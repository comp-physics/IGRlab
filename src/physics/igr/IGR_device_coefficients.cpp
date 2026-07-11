#include "physics/igr/IGR.hpp"
#include "utils/config/env_flags.hpp"
#if !((defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
      (defined(MFEM_USE_HIP) && !defined(__HIPCC__)))
#include "mfem/general/forall.hpp"
#ifndef DG_IGR_HAS_ACCEL_FORALL
#define DG_IGR_HAS_ACCEL_FORALL 1
#endif
#endif
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"
#include "IGR_device_kernels.hpp"
#include "IGR_internal_utils.hpp"

#include <algorithm>
#include <cmath>

namespace dg_igr
{
using namespace igr_internal;

namespace
{

std::unique_ptr<mfem::QuadratureSpace>
MakeVolumeQuadratureSpace(mfem::Mesh& mesh,
                          const mfem::IntegrationRule& ir)
{
  if (mesh.GetNumGeometries(mesh.Dimension()) == 1)
  {
    return std::make_unique<mfem::QuadratureSpace>(mesh, ir);
  }
  return std::make_unique<mfem::QuadratureSpace>(&mesh, ir.GetOrder());
}

} // namespace

void IGRSolver::EnsureH1VelocitySpace_()
{
  if (backend_ != GradBackend::H1Projection)
  {
    return;
  }

  mfem::ParMesh* pm = vfes_.GetParMesh();
  MFEM_VERIFY(pm, "Expected mfem::ParMesh for state space.");

  // Choose H1 order sensibly: ≥ 1, and not below the state space order.
  const int p_state = SpaceOrder(vfes_);
  const int p_h1 = std::max(1, p_state);

  // Recreate only if absent or order/mesh changed
  bool need_make = (!h1_vec_fec_ || !h1_vec_fes_ || !vel_h1_);
  if (!need_make)
  {
    if (h1_vec_fec_->GetOrder() != p_h1) need_make = true;
    if (h1_vec_fes_->GetParMesh() != pm) need_make = true;
  }
  if (need_make)
  {
    h1_vec_fec_ =
      std::make_unique<mfem::H1_FECollection>(p_h1, pm->Dimension());
    h1_vec_fes_ =
      std::make_unique<mfem::ParFiniteElementSpace>(pm,
                                                    h1_vec_fec_.get(),
                                                    /*vdim=*/dim_);
    vel_h1_ = std::make_unique<mfem::ParGridFunction>(h1_vec_fes_.get());
    *vel_h1_ = 0.0;
  }
}

void IGRSolver::EnsureDeviceQuadrature_()
{
  if (!UseDeviceCoefficients_())
  {
    return;
  }

  const bool debug_qf =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  mfem::Mesh* mesh = sfes_.GetMesh();
  MFEM_VERIFY(mesh, "IGR device coefficients require a valid mesh.");
  MFEM_VERIFY(mesh == vfes_.GetMesh(),
              "IGR device coefficients expect sigma and state meshes to match.");

  const long mesh_seq = mesh->GetSequence();
  const long sfes_seq = sfes_.GetSequence();
  const long vfes_seq = vfes_.GetSequence();

  const mfem::FiniteElement& sigma_fe = *sfes_.GetFE(0);
  mfem::ElementTransformation* typical_T =
    mesh->GetTypicalElementTransformation();
  MFEM_VERIFY(typical_T,
              "IGR device coefficients require a typical element transform.");
  const mfem::IntegrationRule& mass_ir =
    mfem::MassIntegrator::GetRule(sigma_fe, sigma_fe, *typical_T);
  const mfem::IntegrationRule& diff_ir =
    mfem::DiffusionIntegrator::GetRule(sigma_fe, sigma_fe);
  const mfem::IntegrationRule& rhs_ir =
    mfem::IntRules.Get(sigma_fe.GetGeomType(), 2 * sigma_fe.GetOrder());
  const int mass_qorder = mass_ir.GetOrder();
  const int diff_qorder = diff_ir.GetOrder();
  const int rhs_qorder = rhs_ir.GetOrder();
  const mfem::Geometry::Type face_geom = mesh->GetTypicalFaceGeometry();
  const mfem::IntegrationRule& face_ir =
    FaceIntegrationRules_().Get(face_geom, diff_qorder);
  const int face_qorder = face_ir.GetOrder();

  const bool need_rebuild =
    (!mass_qs_ || !sigma_qs_ || !rhs_qs_ || !u_mass_qf_ || !u_qf_ ||
     !u_rhs_qf_ || !grad_u_qf_ || !grad_u_rhs_qf_ || !invrho_qf_ ||
     !ck_qf_ || !src_qf_ ||
     mesh_seq != device_mesh_sequence_ || sfes_seq != device_sfes_sequence_ ||
     vfes_seq != device_vfes_sequence_ ||
     mass_qorder != device_mass_qorder_ ||
     diff_qorder != device_diff_qorder_ ||
     rhs_qorder != device_rhs_qorder_ ||
     face_qorder != device_face_qorder_);

  if (!need_rebuild)
  {
    return;
  }
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-rebuild"
              << " mesh_seq=" << mesh_seq
              << " sfes_seq=" << sfes_seq
              << " vfes_seq=" << vfes_seq
              << " mass_qorder=" << mass_qorder
              << " diff_qorder=" << diff_qorder
              << " rhs_qorder=" << rhs_qorder
              << " face_qorder=" << face_qorder
              << " face_geom=" << static_cast<int>(face_geom)
              << " is_dg=" << (IsDGSpace(sfes_) ? 1 : 0)
              << std::endl;
  }

  mass_qs_ = MakeVolumeQuadratureSpace(*mesh, mass_ir);
  sigma_qs_ = MakeVolumeQuadratureSpace(*mesh, diff_ir);
  rhs_qs_ = MakeVolumeQuadratureSpace(*mesh, rhs_ir);

  if (IsDGSpace(sfes_))
  {
    sigma_fqs_ =
      std::make_unique<mfem::FaceQuadratureSpace>(*mesh,
                                                  face_ir,
                                                  mfem::FaceType::Interior);
  }
  else
  {
    sigma_fqs_.reset();
  }

  u_mass_qf_ =
    std::make_unique<mfem::QuadratureFunction>(mass_qs_.get(), num_eq_);
  u_mass_qf_->SetVDim(num_eq_);
  u_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(),
                                                     num_eq_);
  u_qf_->SetVDim(num_eq_);
  u_rhs_qf_ =
    std::make_unique<mfem::QuadratureFunction>(rhs_qs_.get(), num_eq_);
  u_rhs_qf_->SetVDim(num_eq_);
  grad_u_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(),
                                                          num_eq_ * dim_);
  grad_u_qf_->SetVDim(num_eq_ * dim_);
  grad_u_rhs_qf_ =
    std::make_unique<mfem::QuadratureFunction>(rhs_qs_.get(),
                                               num_eq_ * dim_);
  grad_u_rhs_qf_->SetVDim(num_eq_ * dim_);

  invrho_qf_ = std::make_unique<mfem::QuadratureFunction>(mass_qs_.get(), 1);
  invrho_qf_->SetVDim(1);
  ck_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(), 1);
  ck_qf_->SetVDim(1);
  src_qf_ = std::make_unique<mfem::QuadratureFunction>(rhs_qs_.get(), 1);
  src_qf_->SetVDim(1);

  if (sigma_fqs_)
  {
    u_face_qf_ =
      std::make_unique<mfem::QuadratureFunction>(sigma_fqs_.get(), num_eq_);
    u_face_qf_->SetVDim(num_eq_);
    ck_face_qf_ =
      std::make_unique<mfem::QuadratureFunction>(sigma_fqs_.get(), 1);
    ck_face_qf_->SetVDim(1);
  }
  else
  {
    u_face_qf_.reset();
    ck_face_qf_.reset();
  }

  invrho_qf_coef_ =
    std::make_unique<mfem::QuadratureFunctionCoefficient>(*invrho_qf_);
  ck_qf_coef_ =
    std::make_unique<mfem::QuadratureFunctionCoefficient>(*ck_qf_);
  src_qf_coef_ =
    std::make_unique<mfem::QuadratureFunctionCoefficient>(*src_qf_);
  if (ck_face_qf_)
  {
    ck_face_qf_coef_ =
      std::make_unique<mfem::QuadratureFunctionCoefficient>(*ck_face_qf_);
  }
  else
  {
    ck_face_qf_coef_.reset();
  }

  device_mesh_sequence_ = mesh_seq;
  device_sfes_sequence_ = sfes_seq;
  device_vfes_sequence_ = vfes_seq;
  device_qorder_ = diff_qorder;
  device_mass_qorder_ = mass_qorder;
  device_diff_qorder_ = diff_qorder;
  device_rhs_qorder_ = rhs_qorder;
  device_face_qorder_ = face_qorder;
}

void IGRSolver::UpdateDeviceCoefficients_(const mfem::ParGridFunction& U)
{
  EnsureDeviceQuadrature_();
  MFEM_VERIFY(mass_qs_ && sigma_qs_ && rhs_qs_,
              "IGR device coefficients require quadrature spaces.");

  const bool debug_qf =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-start" << "\n";
  }

  if (backend_ == GradBackend::H1Projection)
  {
    if (!device_coeff_logged_ && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=grad-backend-override"
                << " requested=h1"
                << " effective=broken-dg"
                << " reason=device-qf" << "\n";
    }
  }

  const mfem::FiniteElementSpace& ufes = *U.ParFESpace();
  const bool use_tensor = mfem::UsesTensorBasis(ufes);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-uses-tensor"
              << " enabled=" << (use_tensor ? 1 : 0) << "\n";
  }
  const mfem::ElementDofOrdering ordering =
    use_tensor ? mfem::ElementDofOrdering::LEXICOGRAPHIC
               : mfem::ElementDofOrdering::NATIVE;

  const mfem::Operator* R = ufes.GetElementRestriction(ordering);
  MFEM_VERIFY(R, "IGR device coefficients require element restriction.");
  device_qinterp_evec_.SetSize(R->Height());
  R->Mult(U, device_qinterp_evec_);

  const int dim = dim_;
  const int num_eq = num_eq_;
  const mfem::real_t alpha = static_cast<mfem::real_t>(alpha_);
  const mfem::real_t rho_floor = static_cast<mfem::real_t>(rho_floor_);

  auto project_state =
    [&](mfem::QuadratureSpace& qs,
        mfem::QuadratureFunction& values,
        mfem::QuadratureFunction* derivatives)
  {
    const mfem::QuadratureInterpolator* qi =
      ufes.GetQuadratureInterpolator(qs);
    MFEM_VERIFY(qi,
                "IGR device coefficients require quadrature interpolator.");
    qi->SetOutputLayout(mfem::QVectorLayout::byVDIM);
    qi->DisableTensorProducts(!use_tensor);
    qi->Values(device_qinterp_evec_, values);
    if (derivatives)
    {
      qi->PhysDerivatives(device_qinterp_evec_, *derivatives);
    }
  };

  project_state(*mass_qs_, *u_mass_qf_, nullptr);
  project_state(*sigma_qs_, *u_qf_, grad_u_qf_.get());
  project_state(*rhs_qs_, *u_rhs_qf_, grad_u_rhs_qf_.get());

  const int mass_ne = mass_qs_->GetNE();
  const int mass_nq = mass_qs_->GetIntRule(0).GetNPoints();
  const int diff_ne = sigma_qs_->GetNE();
  const int diff_nq = sigma_qs_->GetIntRule(0).GetNPoints();
  const int rhs_ne = rhs_qs_->GetNE();
  const int rhs_nq = rhs_qs_->GetIntRule(0).GetNPoints();

  const mfem::real_t* u_mass_data = u_mass_qf_->Read();
  const mfem::real_t* u_diff_data = u_qf_->Read();
  const mfem::real_t* u_rhs_data = u_rhs_qf_->Read();
  const mfem::real_t* grad_rhs_data = grad_u_rhs_qf_->Read();
  mfem::real_t* invrho_data = invrho_qf_->Write();
  mfem::real_t* ck_data = ck_qf_->Write();
  mfem::real_t* src_data = src_qf_->Write();

#ifdef DG_IGR_HAS_ACCEL_FORALL
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-kernel-start" << "\n";
  }
  igr_device::IGRDeviceInvRhoKernel invrho_kernel;
  invrho_kernel.nq = mass_nq;
  invrho_kernel.ne = mass_ne;
  invrho_kernel.num_eq = num_eq;
  invrho_kernel.rho_floor = rho_floor;
  invrho_kernel.u = u_mass_data;
  invrho_kernel.invrho = invrho_data;
  mfem::forall(mass_ne * mass_nq, invrho_kernel);

  igr_device::IGRDeviceAlphaOverRhoKernel ck_kernel;
  ck_kernel.nq = diff_nq;
  ck_kernel.ne = diff_ne;
  ck_kernel.num_eq = num_eq;
  ck_kernel.alpha = alpha;
  ck_kernel.rho_floor = rho_floor;
  ck_kernel.u = u_diff_data;
  ck_kernel.ck = ck_data;
  mfem::forall(diff_ne * diff_nq, ck_kernel);

  igr_device::IGRDeviceSourceKernel src_kernel;
  src_kernel.nq = rhs_nq;
  src_kernel.ne = rhs_ne;
  src_kernel.dim = dim;
  src_kernel.num_eq = num_eq;
  src_kernel.alpha = alpha;
  src_kernel.rho_floor = rho_floor;
  src_kernel.u = u_rhs_data;
  src_kernel.grad = grad_rhs_data;
  src_kernel.src = src_data;
  mfem::forall(rhs_ne * rhs_nq, src_kernel);
#else
  const mfem::real_t zero = static_cast<mfem::real_t>(0.0);
  const auto u_mass = mfem::Reshape(u_mass_data, num_eq, mass_nq, mass_ne);
  auto invrho = mfem::Reshape(invrho_data, mass_nq, mass_ne);
  for (int e = 0; e < mass_ne; ++e)
  {
    for (int q = 0; q < mass_nq; ++q)
    {
      mfem::real_t rho = u_mass(0, q, e);
      if (rho < rho_floor) { rho = rho_floor; }
      invrho(q, e) = 1.0 / rho;
    }
  }

  const auto u_diff = mfem::Reshape(u_diff_data, num_eq, diff_nq, diff_ne);
  auto ck = mfem::Reshape(ck_data, diff_nq, diff_ne);
  for (int e = 0; e < diff_ne; ++e)
  {
    for (int q = 0; q < diff_nq; ++q)
    {
      mfem::real_t rho = u_diff(0, q, e);
      if (rho < rho_floor) { rho = rho_floor; }
      ck(q, e) = alpha / rho;
    }
  }

  const auto u_rhs = mfem::Reshape(u_rhs_data, num_eq, rhs_nq, rhs_ne);
  const auto grad_rhs =
    mfem::Reshape(grad_rhs_data, num_eq, dim, rhs_nq, rhs_ne);
  auto src = mfem::Reshape(src_data, rhs_nq, rhs_ne);
  for (int e = 0; e < rhs_ne; ++e)
  {
    for (int q = 0; q < rhs_nq; ++q)
    {
      mfem::real_t rho = u_rhs(0, q, e);
      if (rho < rho_floor) { rho = rho_floor; }
      mfem::real_t divu = zero;
      mfem::real_t trA2 = zero;
      mfem::real_t gradu[3][3] = {{zero, zero, zero},
                                  {zero, zero, zero},
                                  {zero, zero, zero}};

      const mfem::real_t rho2 = rho * rho;
      for (int i = 0; i < dim; ++i)
      {
        const mfem::real_t mom_i = u_rhs(1 + i, q, e);
        for (int j = 0; j < dim; ++j)
        {
          const mfem::real_t dmij = grad_rhs(1 + i, j, q, e);
          const mfem::real_t drhoj = grad_rhs(0, j, q, e);
          gradu[i][j] = (rho * dmij - mom_i * drhoj) / rho2;
        }
      }

      for (int i = 0; i < dim; ++i)
      {
        divu += gradu[i][i];
      }
      for (int i = 0; i < dim; ++i)
      {
        for (int j = 0; j < dim; ++j)
        {
          trA2 += gradu[i][j] * gradu[j][i];
        }
      }

      src(q, e) = alpha * (trA2 + divu * divu);
    }
  }
#endif

  if (!device_coeff_logged_ && mfem::Mpi::Root())
  {
    const bool u_mass_ok = VectorIsFinite(*u_mass_qf_);
    const bool u_diff_ok = VectorIsFinite(*u_qf_);
    const bool u_rhs_ok = VectorIsFinite(*u_rhs_qf_);
    const bool grad_diff_ok = VectorIsFinite(*grad_u_qf_);
    const bool grad_rhs_ok = VectorIsFinite(*grad_u_rhs_qf_);
    const bool invrho_ok = VectorIsFinite(*invrho_qf_);
    const bool ck_ok = VectorIsFinite(*ck_qf_);
    const bool src_ok = VectorIsFinite(*src_qf_);
    std::cout << "[igr-linear] event=device-qf-finite"
              << " u_mass=" << (u_mass_ok ? 1 : 0)
              << " u_diff=" << (u_diff_ok ? 1 : 0)
              << " u_rhs=" << (u_rhs_ok ? 1 : 0)
              << " grad_diff=" << (grad_diff_ok ? 1 : 0)
              << " grad_rhs=" << (grad_rhs_ok ? 1 : 0)
              << " invrho=" << (invrho_ok ? 1 : 0)
              << " ck=" << (ck_ok ? 1 : 0)
              << " src=" << (src_ok ? 1 : 0) << "\n";
    if (!(u_mass_ok && u_diff_ok && u_rhs_ok && grad_diff_ok &&
          grad_rhs_ok && invrho_ok && ck_ok && src_ok))
    {
      MFEM_ABORT("IGR device quadrature data contains non-finite values.");
    }

    mfem::Vector inv_vals, ck_vals, src_vals;
    invrho_qf_->GetValues(0, inv_vals);
    ck_qf_->GetValues(0, ck_vals);
    src_qf_->GetValues(0, src_vals);
    auto host_minmax = [](const mfem::Vector& v, double& vmin, double& vmax)
    {
      const double* data = v.HostRead();
      const int n = v.Size();
      vmin = (n > 0) ? data[0] : 0.0;
      vmax = vmin;
      for (int i = 1; i < n; ++i)
      {
        vmin = std::min(vmin, data[i]);
        vmax = std::max(vmax, data[i]);
      }
    };
    double inv_min = 0.0, inv_max = 0.0;
    double ck_min = 0.0, ck_max = 0.0;
    double src_min = 0.0, src_max = 0.0;
    host_minmax(inv_vals, inv_min, inv_max);
    host_minmax(ck_vals, ck_min, ck_max);
    host_minmax(src_vals, src_min, src_max);
    std::cout << "[igr-linear] event=device-qf-range"
              << " invrho_min=" << inv_min
              << " invrho_max=" << inv_max
              << " ck_min=" << ck_min
              << " ck_max=" << ck_max
              << " src_min=" << src_min
              << " src_max=" << src_max << "\n";

    if (inv_vals.Size() > 0 && ck_vals.Size() > 0 && src_vals.Size() > 0)
    {
      const mfem::real_t inv_q = inv_vals(0);
      const mfem::real_t ck_q = ck_vals(0);
      const mfem::real_t src_q = src_vals(0);
      std::cout << "[igr-linear] event=device-qf-sample"
                << " invrho_q=" << inv_q
                << " ck_q=" << ck_q
                << " src_q=" << src_q
                << " mass_qorder=" << device_mass_qorder_
                << " diff_qorder=" << device_diff_qorder_
                << " rhs_qorder=" << device_rhs_qorder_ << "\n";
    }
  }

  if (sigma_fqs_ && ck_face_qf_)
  {
    if (!device_coeff_logged_ && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=face-qf-projector"
                << " mode=single-valued-mfem-face-restriction"
                << " diagnostic_only=1"
                << " reason=side-dependent-coefficient-not-represented"
                << "\n";
    }
    const auto face_info =
      dg_face::ProjectFaceQuadratureValues(ufes,
                                           *sigma_fqs_,
                                           U,
                                           *u_face_qf_,
                                           &device_face_restr_evec_);
    if (debug_qf && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=face-qf-sizes"
                << " dim=" << face_info.dim
                << " vdim=" << face_info.vdim
                << " nd1d=" << face_info.nd1d
                << " nq1d=" << face_info.nq1d
                << " nf=" << face_info.nf
                << " f_vec_size=" << face_info.f_vec_size
                << " qf_size=" << face_info.qf_size
                << std::endl;
    }

    const int nf = sigma_fqs_->GetNE();
    const int nqf = sigma_fqs_->GetIntRule(0).GetNPoints();
    const mfem::real_t* uface_data = u_face_qf_->Read();
    mfem::real_t* ckface_data = ck_face_qf_->Write();
    const auto uface = mfem::Reshape(uface_data, num_eq, nqf, nf);
    auto ckface = mfem::Reshape(ckface_data, nqf, nf);

#ifdef DG_IGR_HAS_ACCEL_FORALL
    igr_device::IGRDeviceFaceCoeffKernel face_kernel;
    face_kernel.nq = nqf;
    face_kernel.nf = nf;
    face_kernel.num_eq = num_eq;
    face_kernel.alpha = alpha;
    face_kernel.rho_floor = rho_floor;
    face_kernel.uface = uface_data;
    face_kernel.ckface = ckface_data;
    mfem::forall(nf * nqf, face_kernel);
#else
    for (int f = 0; f < nf; ++f)
    {
      for (int q = 0; q < nqf; ++q)
      {
        mfem::real_t rho = uface(0, q, f);
        if (rho < rho_floor) { rho = rho_floor; }
        ckface(q, f) = alpha / rho;
      }
    }
#endif
  }

  device_coeff_logged_ = true;
}

} // namespace dg_igr
