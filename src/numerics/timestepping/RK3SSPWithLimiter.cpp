#include "igrlab/environment.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"

#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "utils/diagnostics/euler_admissibility.hpp"
#include "utils/diagnostics/non_finite.hpp"
#include "utils/profiling/nvtx.hpp"
#include "numerics/limiters/limiters.hpp" // ApplyZhangShuPositivityLimiter, etc.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace mfem
{

namespace
{

inline const char* AdmissibilityPointSetName(
  mfem::igrlab::EulerAdmissibilityPointSet point_set)
{
  switch (point_set)
  {
  case mfem::igrlab::EulerAdmissibilityPointSet::Dof:
    return "dof";
  case mfem::igrlab::EulerAdmissibilityPointSet::Limiter:
    return "limiter";
  case mfem::igrlab::EulerAdmissibilityPointSet::Output:
    return "output";
  }
  return "unknown";
}

inline std::string StateComponentName(int component, int dim, int vdim)
{
  if (component < 0)
  {
    return "unknown";
  }
  if (vdim == dim + 2)
  {
    if (component == 0)
    {
      return "density";
    }
    if (component >= 1 && component <= dim)
    {
      static const char* kAxes[] = {"x", "y", "z"};
      std::ostringstream os;
      os << "momentum_" << kAxes[component - 1];
      return os.str();
    }
    if (component == dim + 1)
    {
      return "energy";
    }
  }

  std::ostringstream os;
  os << "component_" << component;
  return os.str();
}

inline void WriteDG0ElementField(mfem::ParGridFunction& field,
                                 const mfem::Vector& element_values)
{
  auto* fes = field.ParFESpace();
  MFEM_VERIFY(fes != nullptr, "Limiter diagnostics field is missing an FE space.");
  MFEM_VERIFY(fes->GetNE() == element_values.Size(),
              "Limiter diagnostics field has the wrong number of elements.");

  const double* values = element_values.HostRead();
  double* field_data = field.HostWrite();
  mfem::Array<int> vdofs;
  for (int i = 0; i < fes->GetNE(); ++i)
  {
    fes->GetElementVDofs(i, vdofs);
    MFEM_VERIFY(vdofs.Size() == 1,
                "Limiter diagnostics field must use one scalar dof per element.");
    const int dof = vdofs[0] >= 0 ? vdofs[0] : -1 - vdofs[0];
    field_data[dof] = values[i];
  }
}

inline void VerifyHostLimiter(const char* name)
{
  MFEM_VERIFY(!mfem::Device::Allows(mfem::Backend::DEVICE_MASK),
              name << " is host-only and cannot be added to a CUDA RK "
                   << "pipeline. Use the device-native Zhang-Shu limiter or "
                   << "run this method on CPU.");
}

inline real_t PhysicalBoundaryFluxStageWeight(SSPRKMethod method, int stage)
{
  MFEM_VERIFY(stage >= 0 && stage < SSPRKStageCount(method),
              "Physical-boundary flux tracking received an invalid RK stage.");
  switch (method)
  {
  case SSPRKMethod::SSPRK33:
    return SSPRK33Coefficients::b[static_cast<std::size_t>(stage)];
  case SSPRKMethod::SSPRK104:
    return SSPRK104Coefficients::b[static_cast<std::size_t>(stage)];
  case SSPRKMethod::ResearchRK4:
    return ResearchRK4Coefficients::b[static_cast<std::size_t>(stage)];
  case SSPRKMethod::ResearchDOPRI5:
    return ResearchDOPRI5Coefficients::b[static_cast<std::size_t>(stage)];
  case SSPRKMethod::ResearchRadauPicard5:
    return stage < 10 ? 0.0 : ResearchRadauPicard5Coefficients::b[stage - 10];
  }
  MFEM_ABORT("Unsupported SSPRK method for physical-boundary flux tracking.");
  return 0.0;
}

} // namespace

// ------------------------------- Construction & Init ----------------------------

RK3SSPWithLimiter::RK3SSPWithLimiter(ParFiniteElementSpace& vfes,
                                     SSPRKMethod method)
  : vfes_(vfes), method_(method)
{
}

void RK3SSPWithLimiter::Init(TimeDependentOperator& f_)
{
  ODESolver::Init(f_);
  state_generation_ = 0;
  rhs_evaluation_count_ = 0;
  attempted_characteristic_speed_ = 0.0;
  accepted_characteristic_speed_ = 0.0;
  attempted_characteristic_speed_available_ = false;
  accepted_characteristic_speed_available_ = false;
  const auto* stage_aware = dynamic_cast<const RKStageAwareOperator*>(&f_);
  rhs_requires_step_transaction_ =
    stage_aware && stage_aware->RequiresRKStepTransaction();
  boundary_flux_provider_ =
    dynamic_cast<const PhysicalBoundaryFluxIntegralProvider*>(&f_);
  cumulative_physical_boundary_flux_available_ = false;
  attempted_physical_boundary_flux_integral_.SetSize(0);
  attempted_physical_boundary_flux_absolute_integral_.SetSize(0);
  cumulative_physical_boundary_flux_integral_.SetSize(0);
  cumulative_physical_boundary_flux_absolute_integral_.SetSize(0);
  const int n = f->Width();
  y_.SetSize(n, mem_type);
  k_.SetSize(n, mem_type);

  // If user didn't set a quad order (for helper PP), choose something safe
  if (quad_order_ <= 0)
  {
    quad_order_ = AutoQuadOrder();
  }
}

bool RK3SSPWithLimiter::SupportsPhysicalBoundaryFluxTracking() const
{
  return boundary_flux_provider_ != nullptr &&
         boundary_flux_provider_->SupportsPhysicalBoundaryFluxIntegral();
}

const Vector&
RK3SSPWithLimiter::GetCumulativePhysicalBoundaryFluxIntegral() const
{
  MFEM_VERIFY(cumulative_physical_boundary_flux_available_,
              "No accepted physical-boundary flux integral is available.");
  return cumulative_physical_boundary_flux_integral_;
}

const Vector&
RK3SSPWithLimiter::GetCumulativePhysicalBoundaryFluxAbsoluteIntegral() const
{
  MFEM_VERIFY(cumulative_physical_boundary_flux_available_,
              "No accepted physical-boundary absolute flux integral is "
              "available.");
  return cumulative_physical_boundary_flux_absolute_integral_;
}

int RK3SSPWithLimiter::AutoQuadOrder() const
{
  // Use 2*order + 2 by default (good for averages + positivity points)
  const int order = vfes_.GetFE(0)->GetOrder();
  return 2 * order + 2;
}

// ----------------------------- Callback application ----------------------------

bool RK3SSPWithLimiter::ApplyCallbacks(Vector& v,
                                       double t,
                                       bool apply_optional,
                                       bool report_state_mutation)
{
  // Quick exit if no callbacks at all.
  if (always_callbacks_.empty() &&
      (!apply_optional || optional_callbacks_.empty()))
  {
    return false;
  }

  // Non-owning view of v as a ParGridFunction.
  ParGridFunction U(&vfes_, v);
  bool local_state_modified = false;

  // Apply dissipative/reconstruction limiters first. An always-on admissibility
  // callback such as Zhang--Shu must see their final state before the next RHS
  // or accepted endpoint is published.
  if (apply_optional)
  {
    for (auto& cb : optional_callbacks_)
    {
      const bool callback_modified =
        cb(vfes_, U, t, report_state_mutation);
      local_state_modified = local_state_modified || callback_modified;
    }
  }

  // Always-on callbacks are the terminal stage gate. Preserve insertion order
  // within the group while guaranteeing that optional slope limiting cannot
  // invalidate a preceding positivity projection.
  for (auto& cb : always_callbacks_)
  {
    const bool callback_modified = cb(vfes_, U, t, report_state_mutation);
    local_state_modified = local_state_modified || callback_modified;
  }

  // Callbacks may update the non-owning view on the device. Propagate its
  // memory validity to the owning RK vector before host diagnostics or output.
  U.SyncAliasMemory(v);
  if (!report_state_mutation)
  {
    return false;
  }

  int modified = local_state_modified ? 1 : 0;
  if (mfem::Mpi::IsInitialized() && !mfem::Mpi::IsFinalized())
  {
    MPI_Allreduce(MPI_IN_PLACE,
                  &modified,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  vfes_.GetComm());
  }
  return modified != 0;
}

void RK3SSPWithLimiter::AdvanceStateGeneration()
{
  MFEM_VERIFY(state_generation_ !=
                std::numeric_limits<std::uint64_t>::max(),
              "RK state generation overflow.");
  ++state_generation_;
}

// ----------------------- Convenience helpers to add callbacks ------------------

void RK3SSPWithLimiter::AddZhangShuLimiter(ZhangShuPointSet point_set,
                                           bool always_on,
                                           std::optional<bool>
                                             enforce_vtk_points)
{
  // Capture parameters by value; the callback will call the PP limiter on the stage.
  const int dim = dim_;
  const double gamma = gamma_;
  const double rhomin = rho_min_;
  const double pmin = p_min_;
  const int qord = (quad_order_ > 0) ? quad_order_ : AutoQuadOrder();
  const auto face_qords = face_quad_orders_;
  const int iters = p_bisect_iters_;

  auto cb = [this,
             dim,
             gamma,
             rhomin,
             pmin,
             qord,
             face_qords,
             iters,
             point_set,
             enforce_vtk_points](ParFiniteElementSpace& vf,
                                 ParGridFunction& U,
                                 double /*t*/,
                                 bool report_state_mutation)
  {
    mfem::ZhangShuLimiterElementDiagnostics* diagnostics_ptr =
      (zhang_shu_active_field_ != nullptr && zhang_shu_theta_rho_field_ != nullptr &&
       zhang_shu_theta_p_field_ != nullptr)
        ? &zhang_shu_diagnostics_
        : nullptr;
    bool local_state_modified = false;
    ApplyZhangShuPositivityLimiter(
      vf,
      U,
      dim,
      gamma,
      rhomin,
      pmin,
      qord,
      face_qords,
      iters,
      point_set,
      diagnostics_ptr,
      enforce_vtk_points,
      report_state_mutation ? &local_state_modified : nullptr);
    if (diagnostics_ptr != nullptr)
    {
      // Publish only after the complete RK attempt has been accepted. A later
      // stage may still reject, in which case failure output must retain the
      // last committed diagnostic fields rather than an abandoned stage.
      zhang_shu_diagnostics_pending_ = true;
    }
    return local_state_modified;
  };

  if (always_on)
  {
    always_callbacks_.push_back(std::move(cb));
  }
  else
  {
    optional_callbacks_.push_back(std::move(cb));
  }
}

void RK3SSPWithLimiter::AddCharTVBLimiter(const CharTVBOptions& opts)
{
  VerifyHostLimiter("Characteristic TVB limiter");
  auto local = opts; // make a self-contained copy for the stored lambda
  AddOptionalCallback(
    [local](ParFiniteElementSpace& vf, ParGridFunction& U, double /*t*/)
    { ApplyCharacteristicTVBLimiter(vf, U, local); });
}

void RK3SSPWithLimiter::AddCharTVBFaceNormalLimiter(const CharTVBOptions& opts)
{
  VerifyHostLimiter("Face-normal TVB limiter");
  auto local = opts; // capture by value
  AddOptionalCallback([local](mfem::ParFiniteElementSpace& vf,
                              mfem::ParGridFunction& U,
                              double /*t*/)
                      { ApplyCharacteristicTVBFaceNormal(vf, U, local); });
}

void RK3SSPWithLimiter::AddCharBJLimiter(const CharBJOptions& opts)
{
  VerifyHostLimiter("Characteristic BJ limiter");
  auto local = opts;
  AddOptionalCallback(
    [local](ParFiniteElementSpace& vf, ParGridFunction& U, double /*t*/)
    { ApplyCharacteristicBJLimiter(vf, U, local); });
}

void RK3SSPWithLimiter::AddFaceCharBJLimiter(const FaceBJOptions& opts)
{
  VerifyHostLimiter("Face-normal characteristic BJ limiter");
  auto local = opts; // capture by value for safety
  AddOptionalCallback(
    [local](mfem::ParFiniteElementSpace& vf,
            mfem::ParGridFunction& Ug,
            double /*t*/)
    { ApplyFaceNormalCharacteristicBJLimiter(vf, Ug, local); });
}

void RK3SSPWithLimiter::AddBarthJespersenLimiter(const BJOptions& opts)
{
  VerifyHostLimiter("Barth-Jespersen limiter");
  auto local = opts;
  AddOptionalCallback(
    [local](ParFiniteElementSpace& vf, ParGridFunction& U, double /*t*/)
    { ApplyBarthJespersenLimiter(vf, U, local); });
}

void RK3SSPWithLimiter::AddCharVenkatXYLimiter(const VenkatOptions& opts)
{
  VerifyHostLimiter("Characteristic Venkat XY limiter");
  auto local = opts;
  AddOptionalCallback([local](mfem::ParFiniteElementSpace& vf,
                              mfem::ParGridFunction& U,
                              double /*t*/)
                      { ApplyCharacteristicVenkatXY(vf, U, local); });
}

void RK3SSPWithLimiter::AddCharVenkatFaceNormalLimiter(
  const VenkatOptions& opts)
{
  VerifyHostLimiter("Face-normal characteristic Venkat limiter");
  auto local = opts;
  AddOptionalCallback([local](mfem::ParFiniteElementSpace& vf,
                              mfem::ParGridFunction& U,
                              double /*t*/)
                      { ApplyCharacteristicVenkatFaceNormal(vf, U, local); });
}

void RK3SSPWithLimiter::AddKrivodonovaLimiter(const KrivoOptions& opts)
{
  VerifyHostLimiter("Krivodonova limiter");
  auto local = opts;
  AddOptionalCallback(
    [local](ParFiniteElementSpace& vf, ParGridFunction& Ustage, double /*t*/)
    { ApplyKrivodonovaHierarchicalLimiter(vf, Ustage, local); });
}

void RK3SSPWithLimiter::MaybeScanVector(Vector& v, double t, const char* label)
{
  if (!nan_scan_stages_ || stage_nan_detected_)
  {
    return;
  }

  ::igrlab::NVTXRange range("RK/nonfinite-scan");
  const MPI_Comm comm = vfes_.GetComm();
  if (mfem::igrlab::detail::CanSkipDetailedFiniteScan(
        v, comm, nonfinite_status_scratch_))
  {
    return;
  }
  ParGridFunction stage(&vfes_, v);
  const int vdim = stage.VectorDim();
  const auto ordering = vfes_.GetOrdering();
  const auto scan =
    mfem::igrlab::ScanNonFiniteDetailed(v, comm, vdim, ordering);

  if (scan.summary.global_non_finite > 0)
  {
    stage_nan_detected_ = true;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (mfem::Mpi::Root())
    {
      std::ostringstream os;
      os << "[nan-scan] event=vector-scan"
         << " label=" << label
         << " t=" << t
         << " non_finite=" << scan.summary.global_non_finite
         << " finite_min=" << scan.summary.global_min
         << " finite_max=" << scan.summary.global_max
         << " first_bad_rank=" << scan.first_bad_rank;
      for (int c = 0; c < vdim; ++c)
      {
        if (c < static_cast<int>(scan.global_component_non_finite.size()) &&
            scan.global_component_non_finite[c] > 0)
        {
          os << " " << StateComponentName(c, dim_, vdim)
             << "=" << scan.global_component_non_finite[c];
        }
      }
      std::cout << os.str() << "\n";
    }

    if (scan.first_local_entry.found && rank == scan.first_bad_rank)
    {
      std::cout << "[nan-scan] event=vector-scan"
                << " label=" << label
                << " t=" << t
                << " rank=" << rank
                << " local_index=" << scan.first_local_entry.local_index
                << " scalar_dof=" << scan.first_local_entry.scalar_dof
                << " component="
                << StateComponentName(scan.first_local_entry.component,
                                      dim_,
                                      vdim)
                << " value=" << scan.first_local_entry.value
                << " ordering="
                << ((ordering == mfem::Ordering::byNODES) ? "byNODES"
                                                          : "byVDIM")
                << "\n";
    }
  }
}

void RK3SSPWithLimiter::MaybeScanAdmissibility(Vector& v,
                                               double t,
                                               const char* label)
{
  if (!admissibility_scan_stages_)
  {
    return;
  }

  ::igrlab::NVTXRange range("RK/admissibility-scans");
  const auto report_scan =
    [&](mfem::igrlab::EulerAdmissibilityPointSet point_set)
  {
    ::igrlab::NVTXRange point_set_range(
      point_set == mfem::igrlab::EulerAdmissibilityPointSet::Dof
        ? "RK/admissibility-dof"
        : (point_set == mfem::igrlab::EulerAdmissibilityPointSet::Limiter
             ? "RK/admissibility-limiter"
             : "RK/admissibility-output"));
    const auto scan = admissibility_cache_.Scan(vfes_,
                                                           v,
                                                           dim_,
                                                           gamma_,
                                                           rho_min_,
                                                           p_min_,
                                                           quad_order_,
                                                           face_quad_orders_,
                                                           point_set);
    if (scan.global_bad_rho <= 0 && scan.global_bad_pressure <= 0)
    {
      return;
    }

    int rank = 0;
    MPI_Comm_rank(vfes_.GetComm(), &rank);
    if (mfem::Mpi::Root())
    {
      std::cout << "[admissibility-scan] event=euler"
                << " label=" << label
                << " point_set=" << AdmissibilityPointSetName(point_set)
                << " t=" << t
                << " bad_rho=" << scan.global_bad_rho
                << " bad_pressure=" << scan.global_bad_pressure
                << " min_rho=" << scan.global_min_rho
                << " min_pressure=" << scan.global_min_pressure
                << " first_bad_rank=" << scan.first_bad_rank << "\n";
    }

    if (scan.first_local_issue.found && rank == scan.first_bad_rank)
    {
      std::cout << "[admissibility-scan] event=euler"
                << " label=" << label
                << " point_set=" << AdmissibilityPointSetName(point_set)
                << " t=" << t
                << " rank=" << rank
                << " issue="
                << (scan.first_local_issue.pressure_bad ? "pressure" : "density")
                << " element=" << scan.first_local_issue.element
                << " face=" << scan.first_local_issue.face
                << " qpt=" << scan.first_local_issue.point_index
                << " scalar_dof=" << scan.first_local_issue.scalar_dof
                << " rho=" << scan.first_local_issue.rho
                << " pressure=" << scan.first_local_issue.pressure
                << " face_point=" << (scan.first_local_issue.face_point ? 1 : 0)
                << "\n";
    }
  };

  report_scan(mfem::igrlab::EulerAdmissibilityPointSet::Dof);
  report_scan(mfem::igrlab::EulerAdmissibilityPointSet::Limiter);
  report_scan(mfem::igrlab::EulerAdmissibilityPointSet::Output);
}

void RK3SSPWithLimiter::MaybeCaptureStagePair(Vector& v,
                                              double t,
                                              const char* side)
{
  const char* prefix = ::igrlab::GetEnv("DG_IGR_RK_STAGE_CAPTURE_DIR");
  if (!prefix || prefix[0] == '\0' || stage_pair_capture_written_)
  {
    return;
  }

  const char* min_time_text = ::igrlab::GetEnv("DG_IGR_RK_STAGE_CAPTURE_MIN_TIME");
  const double min_time = min_time_text ? std::strtod(min_time_text, nullptr) : 0.0;
  if (t < min_time)
  {
    return;
  }

  int rank = 0;
  MPI_Comm_rank(vfes_.GetComm(), &rank);
  std::ostringstream rank_name;
  rank_name << std::setfill('0') << std::setw(6) << rank;
  const std::filesystem::path output_dir =
    std::filesystem::path(prefix) / side;
  std::filesystem::create_directories(output_dir);

  auto* mesh = vfes_.GetParMesh();
  {
    std::ofstream output(output_dir /
                         ("euler-mesh-final." + rank_name.str()));
    output.precision(17);
    output << *mesh;
  }

  MFEM_VERIFY(vfes_.GetOrdering() == mfem::Ordering::byNODES,
              "RK stage capture requires byNODES ordering.");
  const int ndofs = vfes_.GetNDofs();
  const double* state_data = v.HostRead();
  mfem::ParFiniteElementSpace scalar_fes(
    mesh, vfes_.FEColl(), 1, mfem::Ordering::byNODES);
  for (int component = 0; component < vfes_.GetVDim(); ++component)
  {
    mfem::ParGridFunction field(&scalar_fes,
                                const_cast<double*>(state_data) +
                                  component * ndofs);
    std::ofstream output(
      output_dir /
      ("euler-" + std::to_string(component) + "-final." + rank_name.str()));
    output.precision(17);
    output << field;
  }

  if (rank == 0)
  {
    std::ofstream output(output_dir / "metadata.txt");
    output << std::setprecision(17)
           << "side=" << side << '\n'
           << "step=" << step_count_ << '\n'
           << "time=" << t << '\n'
           << "order=" << vfes_.GetFE(0)->GetOrder() << '\n'
           << "local_state_size=" << v.Size() << '\n';
    std::cout << "[rk-stage-capture] side=" << side
              << " step=" << step_count_
              << " t=" << t
              << " dir=" << output_dir.string() << std::endl;
  }

  if (std::string(side) == "post")
  {
    stage_pair_capture_written_ = true;
  }
}

// ---------------------------- Explicit SSP RK methods -------------------------

void RK3SSPWithLimiter::Step(Vector& x, real_t& t, real_t& dt)
{
  ::igrlab::NVTXRange range("RK/Step");
  stage_nan_detected_ = false;
  zhang_shu_diagnostics_pending_ = false;
  const int stage_count = GetStageCount();
  MFEM_VERIFY(stage_count > 0, "Unsupported explicit SSPRK method.");
  const bool track_boundary_flux = physical_boundary_flux_tracking_enabled_;
  MFEM_VERIFY(!track_boundary_flux || SupportsPhysicalBoundaryFluxTracking(),
              "Physical-boundary flux tracking was enabled, but the RHS "
              "operator cannot provide exact boundary-flux integrals.");
  const real_t attempted_dt = dt;
  attempted_characteristic_speed_ = 0.0;
  attempted_characteristic_speed_available_ = false;
  if (track_boundary_flux)
  {
    attempted_physical_boundary_flux_integral_ = 0.0;
    attempted_physical_boundary_flux_absolute_integral_ = 0.0;
  }
  RKStageAwareOperator* stage_aware = dynamic_cast<RKStageAwareOperator*>(f);
  if (stage_aware)
  {
    stage_aware->BeginRKStep(
      static_cast<int>(step_count_), t, dt, stage_count);
  }
  // Decide whether to apply optional limiters on this step.
  const bool apply_optional =
    (max_limiter_steps_ < 0) || (step_count_ < max_limiter_steps_);

  // Optionally ensure the incoming state satisfies pipeline constraints.
  //   - Always-callbacks: always applied if limit_initial_ is true.
  //   - Optional-callbacks: only early steps if apply_optional = true.
  if (limit_initial_)
  {
    if (ApplyCallbacks(x,
                       t,
                       apply_optional,
                       /*report_state_mutation=*/true))
    {
      AdvanceStateGeneration();
    }
  }
  MaybeScanVector(x, t, "initial");
  MaybeScanAdmissibility(x, t, "initial");

  auto evaluate_rhs = [&](int stage,
                          Vector& state,
                          real_t stage_time,
                          const char* label)
  {
    if (stage_aware)
    {
      stage_aware->SetRKStageState(stage, state_generation_, stage_time);
    }
    f->SetTime(stage_time);
    ++rhs_evaluation_count_;
    if (track_boundary_flux)
    {
      // Keep the one-shot arm adjacent to the actual production RHS. In
      // particular, separate CFL probes never pass through this path.
      boundary_flux_provider_->ArmPhysicalBoundaryFluxIntegral();
    }
    f->Mult(state, k_);
    if (characteristic_speed_reader_)
    {
      const real_t stage_characteristic_speed =
        characteristic_speed_reader_();
      MFEM_VERIFY(std::isfinite(stage_characteristic_speed) &&
                    stage_characteristic_speed >= 0.0,
                  "The SSPRK RHS reported an invalid characteristic speed.");
      attempted_characteristic_speed_ =
        attempted_characteristic_speed_available_
          ? std::max(attempted_characteristic_speed_,
                     stage_characteristic_speed)
          : stage_characteristic_speed;
      attempted_characteristic_speed_available_ = true;
    }
    if (track_boundary_flux)
    {
      MFEM_VERIFY(
        boundary_flux_provider_->HasLastPhysicalBoundaryFluxIntegral(),
        "An armed RHS did not complete its physical-boundary flux capture.");
      const Vector& stage_flux =
        boundary_flux_provider_->GetLastPhysicalBoundaryFluxIntegral();
      MFEM_VERIFY(stage_flux.Size() > 0,
                  "Physical-boundary flux capture has no components.");
      if (attempted_physical_boundary_flux_integral_.Size() == 0)
      {
        attempted_physical_boundary_flux_integral_.SetSize(stage_flux.Size());
        attempted_physical_boundary_flux_absolute_integral_.SetSize(
          stage_flux.Size());
        attempted_physical_boundary_flux_integral_ = 0.0;
        attempted_physical_boundary_flux_absolute_integral_ = 0.0;
      }
      MFEM_VERIFY(
        stage_flux.Size() == attempted_physical_boundary_flux_integral_.Size(),
        "Physical-boundary flux component count changed within an RK step.");

      const real_t weight =
        attempted_dt * PhysicalBoundaryFluxStageWeight(method_, stage);
      const real_t* flux = stage_flux.HostRead();
      real_t* attempted =
        attempted_physical_boundary_flux_integral_.HostReadWrite();
      real_t* attempted_absolute =
        attempted_physical_boundary_flux_absolute_integral_.HostReadWrite();
      for (int component = 0; component < stage_flux.Size(); ++component)
      {
        attempted[component] += weight * flux[component];
        attempted_absolute[component] +=
          std::abs(weight) * std::abs(flux[component]);
      }
    }
    MaybeScanVector(k_, stage_time, label);
  };

  auto finish_stage = [&](Vector& state,
                          real_t stage_time,
                          bool stage_optional,
                          const char* prelimit_label,
                          const char* postlimit_label)
  {
    MaybeScanVector(state, stage_time, prelimit_label);
    MaybeScanAdmissibility(state, stage_time, prelimit_label);
    ApplyCallbacks(state, stage_time, stage_optional);
    AdvanceStateGeneration();
    MaybeScanVector(state, stage_time, postlimit_label);
    MaybeScanAdmissibility(state, stage_time, postlimit_label);
  };

  const char* final_prelimit_label = "stage3-prelimit";
  const char* final_postlimit_label = "stage3";

  auto tableau_step = [&](const auto& a, const auto& b, const auto& c)
  {
    const int stages = static_cast<int>(b.size());
    for (int stage = 0; stage < stages; ++stage)
    {
      research_stage_work_[stage].SetSize(x.Size(), mem_type);
      if (stage == 0)
      {
        evaluate_rhs(stage, x, t, "research-rk-rhs");
      }
      else
      {
        y_ = x;
        for (int j = 0; j < stage; ++j)
        {
          if (a[stage][j] != 0.0)
          {
            y_.Add(dt * a[stage][j], research_stage_work_[j]);
          }
        }
        const real_t stage_time = t + c[stage] * dt;
        finish_stage(y_, stage_time, /*stage_optional=*/false,
                     "research-rk-stage-prelimit", "research-rk-stage");
        evaluate_rhs(stage, y_, stage_time, "research-rk-rhs");
      }
      research_stage_work_[stage] = k_;
    }
    for (int stage = 0; stage < stages; ++stage)
    {
      if (b[stage] != 0.0)
      {
        x.Add(dt * b[stage], research_stage_work_[stage]);
      }
    }
    final_prelimit_label = "research-rk-final-prelimit";
    final_postlimit_label = "research-rk-final";
  };

  switch (method_)
  {
  case SSPRKMethod::SSPRK33:
    // Preserve the historical SSPRK(3,3) state updates and callback cadence.
    evaluate_rhs(/*stage=*/0, x, t, "rhs0");
    add(x, dt, k_, y_);
    finish_stage(y_,
                 t + dt,
                 apply_optional,
                 "stage1-prelimit",
                 "stage1");

    evaluate_rhs(/*stage=*/1, y_, t + dt, "rhs1");
    y_.Add(dt, k_);
    add(3.0 / 4.0, x, 1.0 / 4.0, y_, y_);
    finish_stage(y_,
                 t + dt / 2.0,
                 /*stage_optional=*/false,
                 "stage2-prelimit",
                 "stage2");

    evaluate_rhs(/*stage=*/2, y_, t + dt / 2.0, "rhs2");
    y_.Add(dt, k_);
    add(1.0 / 3.0, x, 2.0 / 3.0, y_, x);
    break;

  case SSPRKMethod::SSPRK104:
  {
    // Ketcheson's 2-register Shu--Osher implementation. The non-monotone
    // abscissae are intentional: c5=2/3 is followed by c6=1/3.
    constexpr real_t kForwardEulerFraction = 1.0 / 6.0;
    const auto& c = SSPRK104Coefficients::c;
    y_ = x; // q2 retains the accepted input while q1 advances in x.

    for (int stage = 0; stage < 4; ++stage)
    {
      evaluate_rhs(stage, x, t + c[stage] * dt, "ssprk104-rhs");
      x.Add(kForwardEulerFraction * dt, k_);
      finish_stage(x,
                   t + c[stage + 1] * dt,
                   stage == 0 && apply_optional,
                   "ssprk104-stage-prelimit",
                   "ssprk104-stage");
    }

    evaluate_rhs(/*stage=*/4, x, t + c[4] * dt, "ssprk104-rhs");
    x.Add(kForwardEulerFraction * dt, k_);
    y_ *= 1.0 / 25.0;
    y_.Add(9.0 / 25.0, x);
    add(15.0, y_, -5.0, x, x);
    finish_stage(x,
                 t + c[5] * dt,
                 /*stage_optional=*/false,
                 "ssprk104-stage-prelimit",
                 "ssprk104-stage");

    for (int stage = 5; stage < 9; ++stage)
    {
      evaluate_rhs(stage, x, t + c[stage] * dt, "ssprk104-rhs");
      x.Add(kForwardEulerFraction * dt, k_);
      finish_stage(x,
                   t + c[stage + 1] * dt,
                   /*stage_optional=*/false,
                   "ssprk104-stage-prelimit",
                   "ssprk104-stage");
    }

    evaluate_rhs(/*stage=*/9, x, t + c[9] * dt, "ssprk104-rhs");
    x *= 3.0 / 5.0;
    x.Add(1.0, y_);
    x.Add(dt / 10.0, k_);
    final_prelimit_label = "ssprk104-final-prelimit";
    final_postlimit_label = "ssprk104-final";
    break;
  }
  case SSPRKMethod::ResearchRK4:
    tableau_step(ResearchRK4Coefficients::a,
                  ResearchRK4Coefficients::b, ResearchRK4Coefficients::c);
    break;
  case SSPRKMethod::ResearchDOPRI5:
    tableau_step(ResearchDOPRI5Coefficients::a,
                  ResearchDOPRI5Coefficients::b,
                  ResearchDOPRI5Coefficients::c);
    break;
  case SSPRKMethod::ResearchRadauPicard5:
  {
    using Coefficients = ResearchRadauPicard5Coefficients;
    for (Vector& work : research_stage_work_)
    {
      work.SetSize(x.Size(), mem_type);
    }
    evaluate_rhs(0, x, t, "radau-picard-initial-rhs");
    for (int node = 0; node < 3; ++node)
    {
      add(x, dt * Coefficients::c[node], k_, research_stage_work_[node]);
    }
    for (int sweep = 0; sweep < Coefficients::sweeps; ++sweep)
    {
      // Every node consumes the previous sweep. No within-sweep stage
      // dependency is hidden by the serial ordinary-CUDA evaluation here.
      for (int node = 0; node < 3; ++node)
      {
        Vector& state = research_stage_work_[node];
        const real_t stage_time = t + dt * Coefficients::c[node];
        finish_stage(state, stage_time, /*stage_optional=*/false,
                     "radau-picard-node-prelimit", "radau-picard-node");
        evaluate_rhs(1 + 3 * sweep + node, state, stage_time,
                      "radau-picard-node-rhs");
        research_stage_work_[3 + node] = k_;
      }
      if (sweep + 1 < Coefficients::sweeps)
      {
        for (int node = 0; node < 3; ++node)
        {
          Vector& state = research_stage_work_[node];
          state = x;
          for (int j = 0; j < 3; ++j)
          {
            state.Add(dt * Coefficients::q[node][j],
                       research_stage_work_[3 + j]);
          }
        }
      }
    }
    for (int node = 0; node < 3; ++node)
    {
      x.Add(dt * Coefficients::b[node], research_stage_work_[3 + node]);
    }
    final_prelimit_label = "radau-picard-final-prelimit";
    final_postlimit_label = "radau-picard-final";
    break;
  }
  }

  MaybeCaptureStagePair(x, t + dt, "pre");
  MaybeScanVector(x, t + dt, final_prelimit_label);
  MaybeScanAdmissibility(x, t + dt, final_prelimit_label);
  t += dt;
  ApplyCallbacks(x, t, /*apply_optional=*/false);
  AdvanceStateGeneration();
  MaybeCaptureStagePair(x, t, "post");
  MaybeScanVector(x, t, final_postlimit_label);
  MaybeScanAdmissibility(x, t, final_postlimit_label);

  if (stage_aware)
  {
    stage_aware->EndRKStep(x, t, state_generation_);
  }

  // A stage-aware algebraic reconstruction may tighten the characteristic
  // bound at the accepted endpoint (for example after replacing an
  // extrapolated IGR sigma by its exact instantaneous closure). Merge that
  // bound before publishing the transaction. If EndRKStep throws, this code
  // is never reached and rejected work remains unable to alter the accepted
  // CFL state.
  if (characteristic_speed_reader_)
  {
    const real_t endpoint_characteristic_speed =
      characteristic_speed_reader_();
    MFEM_VERIFY(std::isfinite(endpoint_characteristic_speed) &&
                  endpoint_characteristic_speed >= 0.0,
                "The SSPRK endpoint reported an invalid characteristic "
                "speed.");
    attempted_characteristic_speed_ =
      attempted_characteristic_speed_available_
        ? std::max(attempted_characteristic_speed_,
                   endpoint_characteristic_speed)
        : endpoint_characteristic_speed;
    attempted_characteristic_speed_available_ = true;
  }

  if (zhang_shu_diagnostics_pending_)
  {
    WriteDG0ElementField(*zhang_shu_active_field_,
                         zhang_shu_diagnostics_.active);
    WriteDG0ElementField(*zhang_shu_theta_rho_field_,
                         zhang_shu_diagnostics_.theta_rho);
    WriteDG0ElementField(*zhang_shu_theta_p_field_,
                         zhang_shu_diagnostics_.theta_p);
    zhang_shu_diagnostics_pending_ = false;
  }

  if (track_boundary_flux)
  {
    const int component_count =
      attempted_physical_boundary_flux_integral_.Size();
    MFEM_VERIFY(component_count > 0,
                "Tracked RK step completed without boundary-flux samples.");
    if (!cumulative_physical_boundary_flux_available_)
    {
      cumulative_physical_boundary_flux_integral_.SetSize(component_count);
      cumulative_physical_boundary_flux_absolute_integral_.SetSize(
        component_count);
      cumulative_physical_boundary_flux_integral_ = 0.0;
      cumulative_physical_boundary_flux_absolute_integral_ = 0.0;
    }
    MFEM_VERIFY(
      cumulative_physical_boundary_flux_integral_.Size() == component_count,
      "Physical-boundary flux component count changed between RK steps.");
    cumulative_physical_boundary_flux_integral_ +=
      attempted_physical_boundary_flux_integral_;
    cumulative_physical_boundary_flux_absolute_integral_ +=
      attempted_physical_boundary_flux_absolute_integral_;
    cumulative_physical_boundary_flux_available_ = true;
  }

  if (attempted_characteristic_speed_available_)
  {
    accepted_characteristic_speed_ = attempted_characteristic_speed_;
    accepted_characteristic_speed_available_ = true;
  }

  ++step_count_;
}

} // namespace mfem
