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

  const int p_sigma = SpaceOrder(sfes_);
  const int p_state = SpaceOrder(vfes_);
  const int requested_qorder = std::max(1, 2 * std::max(p_sigma, p_state));
  const mfem::FiniteElement& sigma_fe = *sfes_.GetFE(0);
  const bool use_refined =
    (sigma_fe.Space() == mfem::FunctionSpace::rQk);
  const mfem::IntegrationRule& vol_ir =
    use_refined ? mfem::RefinedIntRules.Get(sigma_fe.GetGeomType(),
                                            requested_qorder)
                : mfem::IntRules.Get(sigma_fe.GetGeomType(), requested_qorder);
  const int qorder = vol_ir.GetOrder();
  const mfem::Geometry::Type face_geom = mesh->GetTypicalFaceGeometry();
  const mfem::IntegrationRule& face_ir =
    FaceIntegrationRules_().Get(face_geom, qorder);
  const int face_qorder = face_ir.GetOrder();

  const bool need_rebuild =
    (!sigma_qs_ || !u_qf_ || !invrho_qf_ || !ck_qf_ || !src_qf_ ||
     mesh_seq != device_mesh_sequence_ || sfes_seq != device_sfes_sequence_ ||
     vfes_seq != device_vfes_sequence_ || qorder != device_qorder_ ||
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
              << " qorder=" << qorder
              << " face_qorder=" << face_qorder
              << " face_geom=" << static_cast<int>(face_geom)
              << " is_dg=" << (IsDGSpace(sfes_) ? 1 : 0)
              << std::endl;
  }

  if (mesh->GetNumGeometries(mesh->Dimension()) == 1)
  {
    sigma_qs_ = std::make_unique<mfem::QuadratureSpace>(*mesh, vol_ir);
  }
  else
  {
    sigma_qs_ = std::make_unique<mfem::QuadratureSpace>(mesh, qorder);
  }

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

  u_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(),
                                                     num_eq_);
  u_qf_->SetVDim(num_eq_);
  grad_u_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(),
                                                          num_eq_ * dim_);
  grad_u_qf_->SetVDim(num_eq_ * dim_);

  invrho_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(), 1);
  invrho_qf_->SetVDim(1);
  ck_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(), 1);
  ck_qf_->SetVDim(1);
  src_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(), 1);
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
  device_qorder_ = qorder;
  device_face_qorder_ = face_qorder;
}

void IGRSolver::UpdateDeviceCoefficients_(const mfem::ParGridFunction& U)
{
  EnsureDeviceQuadrature_();
  MFEM_VERIFY(sigma_qs_,
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

  const mfem::QuadratureInterpolator* qi =
    ufes.GetQuadratureInterpolator(*sigma_qs_);
  MFEM_VERIFY(qi, "IGR device coefficients require quadrature interpolator.");
  qi->SetOutputLayout(mfem::QVectorLayout::byVDIM);
  qi->DisableTensorProducts(!use_tensor);
  qi->Values(device_qinterp_evec_, *u_qf_);
  qi->PhysDerivatives(device_qinterp_evec_, *grad_u_qf_);

  const int ne = sigma_qs_->GetNE();
  const int nq = sigma_qs_->GetIntRule(0).GetNPoints();
  const int dim = dim_;
  const int num_eq = num_eq_;

  const mfem::real_t* u_data = u_qf_->Read();
  const mfem::real_t* grad_data = grad_u_qf_->Read();
  mfem::real_t* invrho_data = invrho_qf_->Write();
  mfem::real_t* ck_data = ck_qf_->Write();
  mfem::real_t* src_data = src_qf_->Write();

  const auto u = mfem::Reshape(u_data, num_eq, nq, ne);
  const auto grad = mfem::Reshape(grad_data, num_eq, dim, nq, ne);
  auto invrho = mfem::Reshape(invrho_data, nq, ne);
  auto ck = mfem::Reshape(ck_data, nq, ne);
  auto src = mfem::Reshape(src_data, nq, ne);

  const mfem::real_t alpha = static_cast<mfem::real_t>(alpha_);
  const mfem::real_t rho_floor = static_cast<mfem::real_t>(rho_floor_);

#ifdef DG_IGR_HAS_ACCEL_FORALL
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-kernel-start" << "\n";
  }
  igr_device::IGRDeviceCoeffKernel kernel;
  kernel.nq = nq;
  kernel.ne = ne;
  kernel.dim = dim;
  kernel.num_eq = num_eq;
  kernel.alpha = alpha;
  kernel.rho_floor = rho_floor;
  kernel.u = u_data;
  kernel.grad = grad_data;
  kernel.invrho = invrho_data;
  kernel.ck = ck_data;
  kernel.src = src_data;
  mfem::forall(ne * nq, kernel);
#else
  const mfem::real_t zero = static_cast<mfem::real_t>(0.0);
  for (int e = 0; e < ne; ++e)
  {
    for (int q = 0; q < nq; ++q)
    {
      mfem::real_t rho = u(0, q, e);
      if (rho < rho_floor) { rho = rho_floor; }
      const mfem::real_t inv_rho = 1.0 / rho;
      invrho(q, e) = inv_rho;
      ck(q, e) = alpha * inv_rho;

      mfem::real_t divu = zero;
      mfem::real_t trA2 = zero;
      mfem::real_t gradu[3][3] = {{zero, zero, zero},
                                  {zero, zero, zero},
                                  {zero, zero, zero}};

      const mfem::real_t rho2 = rho * rho;
      for (int i = 0; i < dim; ++i)
      {
        const mfem::real_t mom_i = u(1 + i, q, e);
        for (int j = 0; j < dim; ++j)
        {
          const mfem::real_t dmij = grad(1 + i, j, q, e);
          const mfem::real_t drhoj = grad(0, j, q, e);
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
    const bool u_ok = VectorIsFinite(*u_qf_);
    const bool grad_ok = VectorIsFinite(*grad_u_qf_);
    const bool src_ok = VectorIsFinite(*src_qf_);
    std::cout << "[igr-linear] event=device-qf-finite"
              << " u=" << (u_ok ? 1 : 0)
              << " grad=" << (grad_ok ? 1 : 0)
              << " src=" << (src_ok ? 1 : 0) << "\n";
    if (!(u_ok && grad_ok && src_ok))
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

    if (ne > 0 && nq > 0)
    {
      mfem::ElementTransformation* T = sigma_qs_->GetTransformation(0);
      const mfem::IntegrationRule& ir = sigma_qs_->GetIntRule(0);
      const mfem::IntegrationPoint& ip = ir.IntPoint(0);

      mfem::Vector state_host(num_eq_);
      mfem::DenseMatrix grad_host(num_eq_, dim_);
      (void)U.HostRead();
      U.GetVectorValue(*T, ip, state_host);
      U.GetVectorGradient(*T, grad_host);

      const mfem::real_t rho_h = std::max(state_host(0), rho_floor);
      const mfem::real_t invrho_h = 1.0 / rho_h;
      mfem::real_t divu_h = 0.0;
      mfem::real_t trA2_h = 0.0;
      mfem::real_t gradu_h[3][3] = {{0.0, 0.0, 0.0},
                                    {0.0, 0.0, 0.0},
                                    {0.0, 0.0, 0.0}};
      const mfem::real_t rho2_h = rho_h * rho_h;
      for (int i = 0; i < dim_; ++i)
      {
        const mfem::real_t mom_i = state_host(1 + i);
        for (int j = 0; j < dim_; ++j)
        {
          const mfem::real_t dmij = grad_host(1 + i, j);
          const mfem::real_t drhoj = grad_host(0, j);
          gradu_h[i][j] = (rho_h * dmij - mom_i * drhoj) / rho2_h;
        }
      }
      for (int i = 0; i < dim_; ++i)
      {
        divu_h += gradu_h[i][i];
      }
      for (int i = 0; i < dim_; ++i)
      {
        for (int j = 0; j < dim_; ++j)
        {
          trA2_h += gradu_h[i][j] * gradu_h[j][i];
        }
      }
      const mfem::real_t src_h = alpha_ * (trA2_h + divu_h * divu_h);

      const mfem::real_t inv_q = inv_vals(0);
      const mfem::real_t ck_q = ck_vals(0);
      const mfem::real_t src_q = src_vals(0);
      const mfem::real_t* u_host = u_qf_->HostRead();
      const mfem::real_t* grad_host_dev = grad_u_qf_->HostRead();
      const auto u_host_view = mfem::Reshape(u_host, num_eq_, nq, ne);
      const auto grad_host_view =
        mfem::Reshape(grad_host_dev, num_eq_, dim_, nq, ne);
      const mfem::real_t rho_q = u_host_view(0, 0, 0);
      const mfem::real_t rho_h_val = state_host(0);
      const mfem::real_t mom_q = u_host_view(1, 0, 0);
      const mfem::real_t mom_h = state_host(1);
      const mfem::real_t drho_q = grad_host_view(0, 0, 0, 0);
      const mfem::real_t drho_h = grad_host(0, 0);
      mfem::real_t dm_q = 0.0;
      mfem::real_t dm_h = 0.0;
      if (dim_ > 0)
      {
        dm_q = grad_host_view(1, 0, 0, 0);
        dm_h = grad_host(1 + 0, 0);
      }
      mfem::real_t src_calc = 0.0;
      if (dim_ == 1)
      {
        const mfem::real_t rho2 = rho_q * rho_q;
        const mfem::real_t gradu =
          (rho_q * dm_q - mom_q * drho_q) / rho2;
        src_calc = alpha_ * (gradu * gradu + gradu * gradu);
      }
      std::cout << "[igr-linear] event=device-qf-sample"
                << " invrho_q=" << inv_q
                << " invrho_h=" << invrho_h
                << " ck_q=" << ck_q
                << " ck_h=" << (alpha_ * invrho_h)
                << " src_q=" << src_q
                << " src_h=" << src_h
                << " rho_q=" << rho_q
                << " rho_h=" << rho_h_val
                << " mom_q=" << mom_q
                << " mom_h=" << mom_h
                << " drho_q=" << drho_q
                << " drho_h=" << drho_h
                << " dm_q=" << dm_q
                << " dm_h=" << dm_h
                << " src_calc=" << src_calc << "\n";
    }
  }

  if (sigma_fqs_ && ck_face_qf_)
  {
    if (!device_coeff_logged_ && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=face-qf-projector"
                << " mode=custom" << "\n";
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
