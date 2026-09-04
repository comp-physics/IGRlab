// Implementation of DGHyperbolicConservationLaws main operator flow.

#include "DGHyperbolicConservationLaws_impl.hpp"
#include "numerics/dgsem/dgsem_operator.hpp"
#include "utils/profiling/nvtx.hpp"

#include <chrono>
#include <cmath>

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

DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
  DGHyperbolicConservationLaws&&) = default;

IGRFrozenSigmaTelemetry
DGHyperbolicConservationLaws::GetGlobalFrozenSigmaTelemetry() const
{
  IGRFrozenSigmaTelemetry local;
  const auto* flux_igr = dynamic_cast<const FluxWithIGR*>(
    &formIntegrator->GetFluxFunction());
  if (flux_igr)
  {
    local = flux_igr->GetFrozenSigmaTelemetry();
    local.enabled = local.enabled ||
                    (flux_igr->GetInvariantDomainFlux() &&
                     flux_igr->GetSigmaField());
  }

  const auto merge_device_slots = [&local](const Array<double>& storage)
  {
    constexpr int width = dg_igr_detail::kFrozenSigmaTelemetryWidth;
    MFEM_VERIFY(storage.Size() % width == 0,
                "Corrupt packed frozen-sigma telemetry storage.");
    const double* const values = storage.HostRead();
    const int slots = storage.Size() / width;
    for (int slot = 0; slot < slots; ++slot)
    {
      const int offset = slot * width;
      const double samples = values[offset + 0];
      const double nonpositive = values[offset + 1];
      const double nonfinite = values[offset + 2];
      MFEM_VERIFY(std::isfinite(samples) && samples >= 0.0 &&
                    std::isfinite(nonpositive) && nonpositive >= 0.0 &&
                    std::isfinite(nonfinite) && nonfinite >= 0.0,
                  "Invalid packed frozen-sigma telemetry count.");
      local.sample_count += static_cast<std::uint64_t>(samples);
      local.nonpositive_count += static_cast<std::uint64_t>(nonpositive);
      local.nonfinite_count += static_cast<std::uint64_t>(nonfinite);
      local.minimum_raw_q_sigma =
        std::min(local.minimum_raw_q_sigma, values[offset + 3]);
      local.minimum_q_sigma_over_gamma_p =
        std::min(local.minimum_q_sigma_over_gamma_p, values[offset + 4]);
    }
  };
  merge_device_slots(weakdiv_q_sigma_telemetry_);
  merge_device_slots(face_int_q_sigma_telemetry_);
  merge_device_slots(face_bdr_q_sigma_telemetry_);
  merge_device_slots(face_shared_q_sigma_telemetry_);

#ifdef MFEM_USE_MPI
  if (Mpi::IsInitialized() && !Mpi::IsFinalized())
  {
    if (const auto* par_fes =
          dynamic_cast<const ParFiniteElementSpace*>(&vfes))
    {
      int enabled = local.enabled ? 1 : 0;
      unsigned long long counts[3] = {
        static_cast<unsigned long long>(local.sample_count),
        static_cast<unsigned long long>(local.nonpositive_count),
        static_cast<unsigned long long>(local.nonfinite_count)};
      double minima[2] = {local.minimum_raw_q_sigma,
                          local.minimum_q_sigma_over_gamma_p};
      MPI_Allreduce(MPI_IN_PLACE,
                    &enabled,
                    1,
                    MPI_INT,
                    MPI_MAX,
                    par_fes->GetComm());
      MPI_Allreduce(MPI_IN_PLACE,
                    counts,
                    3,
                    MPI_UNSIGNED_LONG_LONG,
                    MPI_SUM,
                    par_fes->GetComm());
      MPI_Allreduce(MPI_IN_PLACE,
                    minima,
                    2,
                    MPI_DOUBLE,
                    MPI_MIN,
                    par_fes->GetComm());
      local.enabled = enabled != 0;
      local.sample_count = static_cast<std::uint64_t>(counts[0]);
      local.nonpositive_count = static_cast<std::uint64_t>(counts[1]);
      local.nonfinite_count = static_cast<std::uint64_t>(counts[2]);
      local.minimum_raw_q_sigma = minima[0];
      local.minimum_q_sigma_over_gamma_p = minima[1];
    }
  }
#endif

  if (!std::isfinite(local.minimum_raw_q_sigma))
  {
    local.minimum_raw_q_sigma = std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(local.minimum_q_sigma_over_gamma_p))
  {
    local.minimum_q_sigma_over_gamma_p =
      std::numeric_limits<double>::quiet_NaN();
  }
  return local;
}

void DGHyperbolicConservationLaws::SetNumericalFlux(const NumericalFlux* flux)
{
  num_flux_ = flux;
  face_data_dirty_ = true;
}

void DGHyperbolicConservationLaws::SetStaticSource(const Vector& source)
{
  MFEM_VERIFY(vol_mode_ == VolumeMode::WeakDivergence,
              "Static Euler sources are currently supported by ordinary DG "
              "only.");
  MFEM_VERIFY(source.Size() == Width(),
              "Projected static source does not match the DG state size.");

  bool nonzero = false;
  const real_t* values = source.HostRead();
  for (int i = 0; i < source.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(values[i]),
                "Projected static source contains a non-finite value.");
    nonzero = nonzero || values[i] != 0.0;
  }
  if (!nonzero)
  {
    static_source_.SetSize(0);
    static_source_requires_rebind_ = false;
    return;
  }

  static_source_.SetSize(source.Size());
  static_source_.UseDevice(Device::Allows(Backend::DEVICE_MASK));
  static_source_ = source;
  static_source_requires_rebind_ = false;
}

dgsem::DGSEMOperator&
DGHyperbolicConservationLaws::GetDGSEMProductionOperator_() const
{
  if (!dgsem_production_operator_)
  {
    dgsem_production_operator_ =
      std::make_unique<dgsem::DGSEMOperator>(*this);
    ++dgsem_production_operator_constructions_;
  }
  else if (dgsem_production_operator_->Rebind(*this))
  {
    ++dgsem_production_operator_rebinds_;
    dgsem_production_dispatch_observed_ = false;
  }
  return *dgsem_production_operator_;
}

bool DGHyperbolicConservationLaws::IsDGSEMProductionLogRoot_() const
{
#ifdef MFEM_USE_MPI
  if (Mpi::IsInitialized() && !Mpi::IsFinalized())
  {
    if (const auto* par_fes =
          dynamic_cast<const ParFiniteElementSpace*>(&vfes))
    {
      int owner_rank = 0;
      MPI_Comm_rank(par_fes->GetComm(), &owner_rank);
      return owner_rank == 0;
    }
  }
#endif
  return true;
}

dgsem::DGSEMProductionSelection
DGHyperbolicConservationLaws::GetDGSEMProductionSelection() const
{
  if (!dgsem_production_operator_)
  {
    return {};
  }
  if (dgsem_production_operator_->Rebind(*this))
  {
    ++dgsem_production_operator_rebinds_;
    dgsem_production_dispatch_observed_ = false;
  }
  return dgsem_production_operator_->GetProductionSelection();
}

dgsem::DGSEMProductionCacheCounters
DGHyperbolicConservationLaws::GetDGSEMProductionCacheCounters() const
{
  if (!dgsem_production_operator_)
  {
    return {};
  }
  if (dgsem_production_operator_->Rebind(*this))
  {
    ++dgsem_production_operator_rebinds_;
    dgsem_production_dispatch_observed_ = false;
  }
  return dgsem_production_operator_->GetProductionCacheCounters();
}

dgsem::DGSEMProductionWorkspaceEvidence
DGHyperbolicConservationLaws::GetDGSEMProductionWorkspaceEvidence() const
{
  if (!dgsem_production_operator_)
  {
    return {};
  }
  if (dgsem_production_operator_->Rebind(*this))
  {
    ++dgsem_production_operator_rebinds_;
    dgsem_production_dispatch_observed_ = false;
  }
  return dgsem_production_operator_->GetProductionWorkspaceEvidence();
}

dgsem::DGSEMProductionBridgeEvidence
DGHyperbolicConservationLaws::GetDGSEMProductionBridgeEvidence() const
{
  if (dgsem_production_operator_ &&
      dgsem_production_operator_->Rebind(*this))
  {
    ++dgsem_production_operator_rebinds_;
    dgsem_production_dispatch_observed_ = false;
  }
  dgsem::DGSEMProductionBridgeEvidence evidence;
  evidence.operator_ready = dgsem_production_operator_ != nullptr;
  evidence.x_local_size = dgsem_x_local_workspace_.Size();
  evidence.z_local_size = dgsem_z_local_workspace_.Size();
  evidence.y_local_size = dgsem_y_local_workspace_.Size();
  evidence.x_local_capacity = dgsem_x_local_workspace_.Capacity();
  evidence.z_local_capacity = dgsem_z_local_workspace_.Capacity();
  evidence.y_local_capacity = dgsem_y_local_workspace_.Capacity();
  evidence.operator_constructions = dgsem_production_operator_constructions_;
  evidence.operator_rebinds = dgsem_production_operator_rebinds_;
  evidence.production_mult_calls = dgsem_production_mult_calls_;
  evidence.fallback_mult_calls = dgsem_production_fallback_calls_;

  std::uint64_t signature = UINT64_C(0x326e0b47cd18a95f);
  const auto append_pointer = [&](const void* pointer)
  {
    signature ^= static_cast<std::uint64_t>(
                   reinterpret_cast<std::uintptr_t>(pointer)) +
                 UINT64_C(0x9e3779b97f4a7c15) + (signature << 6) +
                 (signature >> 2);
  };
  const auto append_workspace = [&](const Vector& workspace)
  {
    append_pointer(workspace.Size() > 0 ? workspace.HostRead() : nullptr);
    append_pointer(workspace.Size() > 0 ? workspace.Read(true) : nullptr);
  };
  append_workspace(dgsem_x_local_workspace_);
  append_workspace(dgsem_z_local_workspace_);
  append_workspace(dgsem_y_local_workspace_);
  evidence.workspace_storage_signature = signature;
  return evidence;
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
bool DGHyperbolicConservationLaws::SupportsPhysicalBoundaryFluxIntegral() const
{
  if (vol_mode_ != VolumeMode::WeakDivergence)
  {
    return false;
  }

  for (size_t i = 0; i < ext_bdr_int_.size(); ++i)
  {
    bool marker_active = false;
    const Array<int>* marker = ext_bdr_markers_[i].get();
    if (marker)
    {
      for (int attr = 0; attr < marker->Size(); ++attr)
      {
        marker_active = marker_active || ((*marker)[attr] != 0);
      }
    }
    if (marker_active &&
        !dynamic_cast<BdrHyperbolicFluxIntegrator*>(ext_bdr_int_[i].get()))
    {
      return false;
    }
  }
  return true;
}

void DGHyperbolicConservationLaws::Mult(const Vector& x, Vector& y) const
{
  ::dg_igr::NVTXRange mult_range("DG/Mult");
  MFEM_VERIFY(!static_source_requires_rebind_,
              "DG operator topology changed after a static source was "
              "installed; project and call SetStaticSource again before "
              "the next RHS evaluation.");

  const bool boundary_flux_requested =
    physical_boundary_flux_integral_armed_;
  physical_boundary_flux_integral_armed_ = false;
  if (boundary_flux_requested)
  {
    physical_boundary_flux_integral_available_ = false;
  }

  // (0) Reset the running max wave speed (used for CFL after Mult()).
  formIntegrator->ResetMaxCharSpeed();
  for (const auto& integrator : ext_bdr_int_)
  {
    if (auto* boundary =
          dynamic_cast<BdrHyperbolicFluxIntegrator*>(integrator.get()))
    {
      boundary->ResetMaxCharSpeed();
      // A failed or interrupted earlier Mult must never leave an integrator
      // accumulating during an unarmed CFL probe.
      boundary->EndBoundaryFluxIntegralCapture();
    }
  }

  const bool capture_boundary_flux =
    boundary_flux_requested && SupportsPhysicalBoundaryFluxIntegral();
  if (capture_boundary_flux)
  {
    physical_boundary_flux_integral_.SetSize(num_equations);
    physical_boundary_flux_integral_.UseDevice(
      Device::Allows(Backend::DEVICE_MASK));
    physical_boundary_flux_integral_ = 0.0;
    for (const auto& integrator : ext_bdr_int_)
    {
      if (auto* boundary =
            dynamic_cast<BdrHyperbolicFluxIntegrator*>(integrator.get()))
      {
        boundary->BeginBoundaryFluxIntegralCapture();
      }
    }
  }

  // ParNonlinearForm evaluates in local (L-vector) space as:
  // y_true = P^T A_local (P x_true). The packed DG device path follows the
  // same mapping when a non-trivial prolongation exists.
  const Vector* x_eval = &x;
  Vector& x_local = dgsem_x_local_workspace_;
  Vector& z_local = dgsem_z_local_workspace_;
  Vector& y_local = dgsem_y_local_workspace_;
  const Operator* prolongation = nullptr;
#ifdef MFEM_USE_MPI
  if (Device::Allows(Backend::DEVICE_MASK) ||
      vol_mode_ == VolumeMode::DGSEMFluxDiff)
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
    dgsem::DGSEMOperator& dgsem_op = GetDGSEMProductionOperator_();
    const bool requested_strict =
      ::dg_igr::EnvFlagEnabled("DG_IGR_DGSEM_PRODUCTION_STRICT");
    const dgsem::DGSEMProductionSelection selection =
      dgsem_op.BindProductionPolicy(requested_strict);
    const bool strict = selection.strict;
    const dgsem::DGSEMProductionReason reason_code = selection.reason;
    const bool dispatch_transition =
      !dgsem_production_dispatch_observed_ ||
      dgsem_production_dispatch_supported_ != selection.supported ||
      dgsem_production_dispatch_strict_ != strict ||
      dgsem_production_dispatch_cuda_ != selection.cuda ||
      dgsem_production_dispatch_reason_ != reason_code;
    const auto record_dispatch = [&]()
    {
      dgsem_production_dispatch_observed_ = true;
      dgsem_production_dispatch_supported_ = selection.supported;
      dgsem_production_dispatch_strict_ = strict;
      dgsem_production_dispatch_cuda_ = selection.cuda;
      dgsem_production_dispatch_reason_ = reason_code;
    };

    const bool timing_enabled = IsDGSEMTimingEnabled_();
    const auto t_mult_start = std::chrono::steady_clock::now();

    if (selection.supported)
    {
      if (y_local.Size() != vfes.GetVSize())
      {
        y_local.SetSize(vfes.GetVSize());
      }
      const dgsem::DGSEMProductionResult result =
        dgsem_op.AssembleProductionResidual(*x_eval, y_local, strict);
      MFEM_VERIFY(result.executed,
                  "DGSEM production execution rejected before output: "
                    << dgsem::DGSEMProductionReasonName(result.reason));

      if (prolongation)
      {
        y = 0.0;
        prolongation->MultTranspose(y_local, y);
      }
      else
      {
        y = y_local;
      }
      max_char_speed = result.max_characteristic_speed;
      ++dgsem_production_mult_calls_;

      if (timing_enabled)
      {
        const double total_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t_mult_start).count();
        RecordDGSEMTiming_(0.0, 0.0, 0.0, total_seconds);
      }
      if (dispatch_transition && IsDGSEMProductionLogRoot_())
      {
        std::cout << "[dgsem-production] selected=mfem-native-packed"
                  << " strict=" << strict
                  << " backend=" << (selection.cuda ? "cuda" : "cpu")
                  << " lvec_mode=" << (prolongation ? "P-x-PT" : "identity")
                  << " order=" << selection.order
                  << " dim=" << selection.dimension << '\n';
      }
      record_dispatch();
      return;
    }

    if (dispatch_transition && IsDGSEMProductionLogRoot_())
    {
      std::cout << "[dgsem-production] selected="
                << (strict ? "strict-rejection" : "host-oracle-fallback")
                << " strict=" << strict
                << " reason="
                << dgsem::DGSEMProductionReasonName(selection.reason)
                << " lvec_mode=" << (prolongation ? "P-x-PT" : "identity")
                << '\n';
    }
    record_dispatch();
    MFEM_VERIFY(!strict,
                "DGSEM production strict policy rejected tuple before output: "
                  << dgsem::DGSEMProductionReasonName(selection.reason));
    MFEM_VERIFY(dgsem::DGSEMOperatorSupportsDimension(dim),
                "DGSEM host oracle cannot execute rejected dimension: "
                  << dgsem::DGSEMProductionReasonName(selection.reason));
    ++dgsem_production_fallback_calls_;

    const bool has_local_elements = vfes.GetNE() > 0;
    const auto t_surface_start = std::chrono::steady_clock::now();
    real_t max_face_speed = 0.0;
    if (has_local_elements)
    {
      max_face_speed = dgsem_op.AssembleSurface(*x_eval, z_eval);
    }
    const auto t_surface_stop = std::chrono::steady_clock::now();

    const auto t_volume_start = std::chrono::steady_clock::now();
    real_t max_vol_speed = 0.0;
    if (has_local_elements)
    {
      max_vol_speed = dgsem_op.AssembleVolume(*x_eval, z_eval);
    }
    const auto t_volume_stop = std::chrono::steady_clock::now();

    const auto t_invmass_start = std::chrono::steady_clock::now();
    if (prolongation)
    {
      y_local.SetSize(z_eval.Size());
      if (has_local_elements)
      {
        dgsem_op.ApplyInverseMass(z_eval, y_local);
      }
      else
      {
        y_local = 0.0;
      }
      y = 0.0;
      prolongation->MultTranspose(y_local, y);
    }
    else if (has_local_elements)
    {
      dgsem_op.ApplyInverseMass(z_eval, y);
    }
    else
    {
      y = 0.0;
    }
    const auto t_invmass_stop = std::chrono::steady_clock::now();

    real_t fallback_max_speed = std::max(max_face_speed, max_vol_speed);
#ifdef MFEM_USE_MPI
    if (Mpi::IsInitialized() && !Mpi::IsFinalized())
    {
      if (const auto* par_fes =
            dynamic_cast<const ParFiniteElementSpace*>(&vfes))
      {
        real_t owning_comm_max = fallback_max_speed;
        MPI_Allreduce(&fallback_max_speed,
                      &owning_comm_max,
                      1,
                      MPITypeMap<real_t>::mpi_type,
                      MPI_MAX,
                      par_fes->GetComm());
        fallback_max_speed = owning_comm_max;
      }
    }
#endif
    max_char_speed = fallback_max_speed;

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
  bool used_fused_element_residual = false;
  real_t device_face_speed = 0.0;
  if (Device::Allows(Backend::DEVICE_MASK))
  {
    const bool default_force_nonlinearform_face = !UseDeviceFaceResidual_();
    const bool force_nonlinearform_face =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DG_FORCE_NLFORM_FACE",
                                      default_force_nonlinearform_face);
    const bool fused_screen_requested = ::dg_igr::EnvFlagEnabledDefault(
      "DG_IGR_DG_FUSED_ELEMENT_RESIDUAL", false);
    if (force_nonlinearform_face)
    {
      if (fused_screen_requested)
      {
        LogFusedElementResidualSelection_(false,
                                          "nonlinearform-face-forced");
      }
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
      used_fused_element_residual =
        fused_screen_requested && UseFusedElementResidualScreen_();
      ::dg_igr::NVTXRange range(
        used_fused_element_residual
          ? "DG/face-flux-records-euler-rusanov"
          : "DG/face-packed-euler-rusanov");
      MFEM_VERIFY(UseDeviceFaceResidual_(),
                  "DG device-closure requires packed device face residual path "
                  "(no host fallback).");
      used_device_face =
        AddDeviceFaceContributionsEulerRusanov_(*x_eval,
                                                z_eval,
                                                device_face_speed,
                                                capture_boundary_flux,
                                                used_fused_element_residual);
      MFEM_VERIFY(used_device_face,
                  "DG device-closure requires packed Euler/Rusanov face kernels "
                  "and supported boundary/sigma metadata (no fallback).");
      if (!dg_face_path_logged_ && mfem::Mpi::Root())
      {
        std::cout << "[dg-device] event=face-domain-path"
                  << " selected="
                  << (used_fused_element_residual
                        ? "face-records-element-owned-euler-rusanov"
                        : "packed-mfem-forall-euler-rusanov")
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

  real_t volume_speed = 0.0;
  if (used_fused_element_residual)
  {
    ::dg_igr::NVTXRange range("DG/fused-volume-face-invmass");
    Vector& fused_output = prolongation ? y_local : y;
    volume_speed = ApplyFusedElementResidualEuler_(*x_eval, fused_output);
    if (prolongation)
    {
      y = 0.0;
      prolongation->MultTranspose(y_local, y);
    }
  }
  else if (!weakdiv.empty()) // weak divergence WAS preassembled element-wise
  {
    ::dg_igr::NVTXRange range("DG/weak-divergence");
    // Compute (I_h F(u_h), ∇v) and accumulate directly into z.
    volume_speed = AddPreassembledWeakDivergence_(*x_eval, z_eval);
  }

  // Apply inverse mass after all contributions have been assembled in z. The
  // fused screen has already completed this work in its element-owned kernel.
  if (!used_fused_element_residual && prolongation)
  {
    ::dg_igr::NVTXRange range("DG/inverse-mass-local-and-restrict");
    y_local.SetSize(z_eval.Size());
    ApplyInverseMass_(z_eval, y_local);
    y = 0.0;
    prolongation->MultTranspose(y_local, y);
  }
  else if (!used_fused_element_residual)
  {
    ::dg_igr::NVTXRange range("DG/inverse-mass");
    ApplyInverseMass_(z_eval, y);
  }

  if (static_source_.Size() != 0)
  {
    // The supplied field is the strong-form source itself, so it is added
    // after inverse mass on both host and device paths.
    y.Add(1.0, static_source_);
  }

  // (4) Expose a domain-wide maximum over volume, interior/shared faces, and
  // physical boundaries. Packed device faces already include their boundary
  // contribution; the separate maximum is then zero and harmless.
  real_t boundary_speed = 0.0;
  for (const auto& integrator : ext_bdr_int_)
  {
    if (auto* boundary =
          dynamic_cast<BdrHyperbolicFluxIntegrator*>(integrator.get()))
    {
      boundary_speed =
        std::max(boundary_speed, boundary->GetMaxCharSpeed());
    }
  }
  const real_t face_speed =
    used_device_face ? device_face_speed : formIntegrator->GetMaxCharSpeed();
  max_char_speed =
    std::max(volume_speed, std::max(face_speed, boundary_speed));

  if (capture_boundary_flux)
  {
    if (!used_device_face)
    {
      real_t* integral = physical_boundary_flux_integral_.HostReadWrite();
      for (const auto& integrator : ext_bdr_int_)
      {
        auto* boundary =
          dynamic_cast<BdrHyperbolicFluxIntegrator*>(integrator.get());
        if (!boundary)
        {
          continue;
        }
        const Vector& local = boundary->GetBoundaryFluxIntegral();
        MFEM_VERIFY(local.Size() == num_equations,
                    "Boundary flux-integral component count mismatch.");
        const real_t* local_values = local.HostRead();
        for (int eq = 0; eq < num_equations; ++eq)
        {
          integral[eq] += local_values[eq];
        }
      }
    }
    for (const auto& integrator : ext_bdr_int_)
    {
      if (auto* boundary =
            dynamic_cast<BdrHyperbolicFluxIntegrator*>(integrator.get()))
      {
        boundary->EndBoundaryFluxIntegralCapture();
      }
    }

#ifdef MFEM_USE_MPI
    if (Mpi::IsInitialized() && !Mpi::IsFinalized())
    {
      if (const auto* par_fes =
            dynamic_cast<const ParFiniteElementSpace*>(&vfes))
      {
        real_t* integral = physical_boundary_flux_integral_.HostReadWrite();
        MPI_Allreduce(MPI_IN_PLACE,
                      integral,
                      num_equations,
                      MPITypeMap<real_t>::mpi_type,
                      MPI_SUM,
                      par_fes->GetComm());
      }
    }
#endif
    physical_boundary_flux_integral_available_ = true;
  }
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
  const bool had_static_source = static_source_.Size() != 0;
  static_source_.SetSize(0);
  static_source_requires_rebind_ =
    static_source_requires_rebind_ || had_static_source;

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
  if (dgsem_production_operator_)
  {
    if (dgsem_production_operator_->Rebind(*this))
    {
      ++dgsem_production_operator_rebinds_;
    }
    dgsem_production_operator_->InvalidateProductionCache(
      dgsem::DGSEMProductionInvalidation::Topology);
  }
  dgsem_x_local_workspace_.SetSize(0);
  dgsem_z_local_workspace_.SetSize(0);
  dgsem_y_local_workspace_.SetSize(0);
  dgsem_production_dispatch_observed_ = false;
  dgsem_production_dispatch_supported_ = false;
  dgsem_production_dispatch_strict_ = false;
  dgsem_production_dispatch_cuda_ = false;
  dgsem_production_dispatch_reason_ = dgsem::DGSEMProductionReason::None;
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
