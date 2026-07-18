// Implementation of DGHyperbolicConservationLaws main operator flow.

#include "DGHyperbolicConservationLaws_impl.hpp"
#include "numerics/dgsem/dgsem_operator.hpp"
#include "utils/profiling/nvtx.hpp"

#include <chrono>

namespace mfem
{

DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
  FiniteElementSpace& vfes_,
  std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
  bool preassembleWeakDivergence)
  : TimeDependentOperator(vfes_.GetTrueVSize()),
    num_equations(formIntegrator_->num_equations),
    dim(vfes_.GetMesh()->SpaceDimension()),
    vfes(vfes_),
    formIntegrator(std::move(formIntegrator_)),
    z(vfes_.GetTrueVSize())
{
  // (A) Compute inverse mass matrices (one per element)
  ComputeInvMass();
  BuildInvMassApplyData_();
  BuildNativeMassInverseSolver_();

  // (B) Build the nonlinear form for face/domain contributions.
#ifndef MFEM_USE_MPI
  nonlinearForm = std::make_unique<NonlinearForm>(&vfes);
#else
  if (auto* pvfes = dynamic_cast<ParFiniteElementSpace*>(&vfes))
    nonlinearForm = std::make_unique<ParNonlinearForm>(pvfes);
  else
    nonlinearForm = std::make_unique<NonlinearForm>(&vfes);
#endif

  // (C) Either preassemble weak divergence, or let the integrator contribute volume terms.
  if (preassembleWeakDivergence)
  {
    ComputeWeakDivergence();
    BuildWeakDivApplyData_();
  }
  else
  {
    nonlinearForm->AddDomainIntegrator(formIntegrator.get());
  }

  // (D) Faces are always handled by the form integrator
  nonlinearForm->AddInteriorFaceIntegrator(formIntegrator.get());
  nonlinearForm->UseExternalIntegrators();
}

void DGHyperbolicConservationLaws::SetNumericalFlux(const NumericalFlux* flux)
{
  num_flux_ = flux;
  face_data_dirty_ = true;
}

// -----------------------------------------------------------------------------
// Apply DG operator
// -----------------------------------------------------------------------------
//
// Steps:
// (0) Reset max wave speed.
// (1) nonlinearForm->Mult(x, z) assembles faces (and domain, if matrix-free).
// (2) Add volume term depending on vol_mode_.
// (3) Apply inverse mass element-wise and scatter to y.
// (4) Update max_char_speed from integrator.
//
void DGHyperbolicConservationLaws::Mult(const Vector& x, Vector& y) const
{
  ::dg_igr::NVTXRange mult_range("DG/Mult");

  // (0) Reset the running max wave speed (used for CFL after Mult()).
  formIntegrator->ResetMaxCharSpeed();

  // ParNonlinearForm evaluates in local (L-vector) space as:
  // y_true = P^T A_local (P x_true). The packed DG device path follows the
  // same mapping when a non-trivial prolongation exists.
  const Vector* x_eval = &x;
  Vector x_local;
  Vector z_local;
  Vector y_local;
  const Operator* prolongation = nullptr;
#ifdef MFEM_USE_MPI
  if (Device::Allows(Backend::DEVICE_MASK))
  {
    if (const auto* pvfes = dynamic_cast<const ParFiniteElementSpace*>(&vfes))
    {
      prolongation = pvfes->GetProlongationMatrix();
      if (prolongation)
      {
        ::dg_igr::NVTXRange range("DG/prolong-true-to-local");
        x_local.SetSize(pvfes->GetVSize());
        prolongation->Mult(x, x_local);
        x_eval = &x_local;
        z_local.SetSize(pvfes->GetVSize());
      }
    }
  }
#endif
  Vector& z_eval = prolongation ? z_local : z;

  if (vol_mode_ == VolumeMode::DGSEMFluxDiff)
  {
    MFEM_VERIFY(dgsem_stab_ == DGSEMStabilization::None,
                "DGSEM stabilization hooks are not implemented. "
                "Use --dgsem-stab none.");
    dgsem::DGSEMOperator dgsem_op(*this);

    const bool timing_enabled = IsDGSEMTimingEnabled_();
    const auto t_mult_start = std::chrono::steady_clock::now();

    const auto t_surface_start = std::chrono::steady_clock::now();
    const real_t max_face_speed = dgsem_op.AssembleSurface(x, z);
    const auto t_surface_stop = std::chrono::steady_clock::now();

    const auto t_volume_start = std::chrono::steady_clock::now();
    const real_t max_vol_speed = dgsem_op.AssembleVolume(x, z);
    const auto t_volume_stop = std::chrono::steady_clock::now();

    const auto t_invmass_start = std::chrono::steady_clock::now();
    dgsem_op.ApplyInverseMass(z, y);
    const auto t_invmass_stop = std::chrono::steady_clock::now();

    max_char_speed = std::max(max_face_speed, max_vol_speed);

    if (timing_enabled)
    {
      const auto t_mult_stop = std::chrono::steady_clock::now();
      const double surface_seconds =
        std::chrono::duration<double>(t_surface_stop - t_surface_start).count();
      const double volume_seconds =
        std::chrono::duration<double>(t_volume_stop - t_volume_start).count();
      const double invmass_seconds =
        std::chrono::duration<double>(t_invmass_stop - t_invmass_start).count();
      const double total_seconds =
        std::chrono::duration<double>(t_mult_stop - t_mult_start).count();
      RecordDGSEMTiming_(surface_seconds,
                         volume_seconds,
                         invmass_seconds,
                         total_seconds);
    }
    return;
  }

  // (1) Build the face contribution (and possibly the volume term) into z:
  //     z = - < F̂(u_h, n), [[v]] >_faces
  //     If weak-divergence is NOT preassembled, HyperbolicFormIntegrator also adds
  //     the volume term (F(u_h), ∇v):
  //     z += (F(u_h), ∇v)
  bool used_device_face = false;
  real_t device_face_speed = 0.0;
  if (Device::Allows(Backend::DEVICE_MASK))
  {
    const bool default_force_nonlinearform_face = !UseDeviceFaceResidual_();
    const bool force_nonlinearform_face =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DG_FORCE_NLFORM_FACE",
                                      default_force_nonlinearform_face);
    if (force_nonlinearform_face)
    {
      ::dg_igr::NVTXRange range("DG/face-domain-mfem-nonlinearform");
      nonlinearForm->Mult(*x_eval, z_eval);
      used_device_face = false;
      device_face_speed = formIntegrator->GetMaxCharSpeed();
      if (!dg_face_path_logged_ && mfem::Mpi::Root())
      {
        std::cout << "[dg-device] event=face-domain-path"
                  << " selected=nonlinearForm->Mult"
                  << " backend=device-mfem-face"
                  << " lvec_mode=" << (prolongation ? "P-x-PT" : "identity")
                  << " x_size=" << x.Size()
                  << " local_vsize=" << vfes.GetVSize()
                  << " true_vsize=" << vfes.GetTrueVSize() << "\n";
        dg_face_path_logged_ = true;
      }
    }
    else
    {
      ::dg_igr::NVTXRange range("DG/face-packed-euler-rusanov");
      MFEM_VERIFY(UseDeviceFaceResidual_(),
                  "DG device-closure requires packed device face residual path "
                  "(no host fallback).");
      used_device_face =
        AddDeviceFaceContributionsEulerRusanov_(*x_eval, z_eval, device_face_speed);
      MFEM_VERIFY(used_device_face,
                  "DG device-closure requires packed Euler/Rusanov face kernels "
                  "and supported boundary/sigma metadata (no fallback).");
      if (!dg_face_path_logged_ && mfem::Mpi::Root())
      {
        std::cout << "[dg-device] event=face-domain-path"
                  << " selected=packed-mfem-forall-euler-rusanov"
                  << " backend=device"
                  << " lvec_mode=" << (prolongation ? "P-x-PT" : "identity")
                  << " x_size=" << x.Size()
                  << " local_vsize=" << vfes.GetVSize()
                  << " true_vsize=" << vfes.GetTrueVSize() << "\n";
        dg_face_path_logged_ = true;
      }
    }
  }
  else
  {
    ::dg_igr::NVTXRange range("DG/face-domain-cpu-nonlinearform");
    nonlinearForm->Mult(x, z);
  }

  if (!weakdiv.empty()) // weak divergence WAS preassembled element-wise
  {
    ::dg_igr::NVTXRange range("DG/weak-divergence");
    // Compute (I_h F(u_h), ∇v) and accumulate directly into z.
    AddPreassembledWeakDivergence_(*x_eval, z_eval);
  }

  // Apply inverse mass after all contributions have been assembled in z.
  if (prolongation)
  {
    ::dg_igr::NVTXRange range("DG/inverse-mass-local-and-restrict");
    y_local.SetSize(z_eval.Size());
    ApplyInverseMass_(z_eval, y_local);
    y = 0.0;
    prolongation->MultTranspose(y_local, y);
  }
  else
  {
    ::dg_igr::NVTXRange range("DG/inverse-mass");
    ApplyInverseMass_(z_eval, y);
  }

  // (4) Expose the maximum characteristic speed accumulated inside the integrators
  max_char_speed =
    used_device_face ? device_face_speed : formIntegrator->GetMaxCharSpeed();
}

DGHyperbolicConservationLaws::~DGHyperbolicConservationLaws()
{
  if (IsDGSEMTimingEnabled_())
  {
    PrintDGSEMTimingSummary_();
  }
}

// -----------------------------------------------------------------------------
// Update internal data (mass matrices, etc.) when mesh/space changes
// -----------------------------------------------------------------------------
void DGHyperbolicConservationLaws::Update()
{
  // Refresh the nonlinear form's internal data (quadrature, face pairs, etc.)
  nonlinearForm->Update();

  // The operator is square: height = width = global dof size
  height = nonlinearForm->Height();
  width = height;

  // Scratch workspace for assembled RHS before inverse mass
  z.SetSize(height);

  // Recompute per-element inverse mass matrices
  ComputeInvMass();
  BuildInvMassApplyData_();
  BuildNativeMassInverseSolver_();

  // If we already had weakdiv matrices, refresh them too
  if (!weakdiv.empty())
  {
    ComputeWeakDivergence();
    BuildWeakDivApplyData_();
  }
  else
  {
    BuildWeakDivApplyData_();
  }

  face_data_dirty_ = true;
}

void DGHyperbolicConservationLaws::SetFaceIntegrationRule(
  const IntegrationRule& ir)
{
  face_int_rule_owned_ = std::make_unique<IntegrationRule>(ir);
  face_int_rule_ = face_int_rule_owned_.get();
  formIntegrator->SetIntegrationRule(*face_int_rule_);
  for (auto& bdr_int : ext_bdr_int_)
  {
    bdr_int->SetIntegrationRule(*face_int_rule_);
  }
  face_data_dirty_ = true;
}

bool DGHyperbolicConservationLaws::IsDGSEMTimingEnabled_() const
{
  if (!dgsem_timing_init_)
  {
    dgsem_timing_enabled_ = ::dg_igr::EnvFlagEnabled("DG_IGR_DGSEM_TIMING");
    dgsem_timing_init_ = true;
  }
  return dgsem_timing_enabled_;
}

void DGHyperbolicConservationLaws::RecordDGSEMTiming_(double surface_seconds,
                                                      double volume_seconds,
                                                      double invmass_seconds,
                                                      double total_seconds) const
{
  ++dgsem_timing_stats_.calls;
  dgsem_timing_stats_.surface_seconds += surface_seconds;
  dgsem_timing_stats_.volume_seconds += volume_seconds;
  dgsem_timing_stats_.invmass_seconds += invmass_seconds;
  dgsem_timing_stats_.total_seconds += total_seconds;
}

void DGHyperbolicConservationLaws::PrintDGSEMTimingSummary_() const
{
  if (dgsem_timing_stats_.calls <= 0)
  {
    return;
  }

  const double calls = static_cast<double>(dgsem_timing_stats_.calls);
  const double total = dgsem_timing_stats_.total_seconds;
  const double surface_pct =
    (total > 0.0) ? (100.0 * dgsem_timing_stats_.surface_seconds / total) : 0.0;
  const double volume_pct =
    (total > 0.0) ? (100.0 * dgsem_timing_stats_.volume_seconds / total) : 0.0;
  const double invmass_pct =
    (total > 0.0) ? (100.0 * dgsem_timing_stats_.invmass_seconds / total) : 0.0;
  const int rank =
    (mfem::Mpi::IsInitialized() && !mfem::Mpi::IsFinalized())
      ? mfem::Mpi::WorldRank()
      : 0;

  std::cout << "[dgsem-op] event=run-summary"
            << " rank=" << rank
            << " calls=" << dgsem_timing_stats_.calls
            << " surface_s=" << dgsem_timing_stats_.surface_seconds
            << " volume_s=" << dgsem_timing_stats_.volume_seconds
            << " invmass_s=" << dgsem_timing_stats_.invmass_seconds
            << " total_mult_s=" << total
            << " avg_surface_s=" << (dgsem_timing_stats_.surface_seconds / calls)
            << " avg_volume_s=" << (dgsem_timing_stats_.volume_seconds / calls)
            << " avg_invmass_s=" << (dgsem_timing_stats_.invmass_seconds / calls)
            << " avg_mult_s=" << (total / calls)
            << " surface_pct=" << surface_pct
            << " volume_pct=" << volume_pct
            << " invmass_pct=" << invmass_pct << "\n";
}

// -----------------------------------------------------------------------------
// Add boundary face integrator with attribute marker (keeps objects alive)
// -----------------------------------------------------------------------------
void DGHyperbolicConservationLaws::AddBdrFaceIntegrator(
  std::unique_ptr<NonlinearFormIntegrator> nfi,
  const Array<int>& bdr_marker)
{
  MFEM_VERIFY(nfi != nullptr,
              "AddBdrFaceIntegrator received a null integrator.");
  if (face_int_rule_)
  {
    nfi->SetIntegrationRule(*face_int_rule_);
  }
  // 1) Own the integrator so NonlinearForm can UseExternalIntegrators()
  NonlinearFormIntegrator* nfi_raw = nfi.get();
  ext_bdr_int_.emplace_back(std::move(nfi));

  // 2) Make a heap copy of the marker so its address is stable
  ext_bdr_markers_.emplace_back(std::make_unique<Array<int>>(bdr_marker));

  // 3) Register with NonlinearForm using the *referenced* stable Array<int>
  nonlinearForm->AddBdrFaceIntegrator(nfi_raw, *ext_bdr_markers_.back());
  face_data_dirty_ = true;
}

} // namespace mfem
