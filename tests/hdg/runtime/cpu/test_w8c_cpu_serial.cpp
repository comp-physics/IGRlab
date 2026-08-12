#include "independent_oracle.hpp"

#include "hdg_cpu_serial_lifecycle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dg::hdg::runtime::cpu::sc25::w8c_internal
{
[[gnu::visibility("hidden")]] bool PrimePersistentTrajectoryTransaction(
    PersistentSc25ScalarIgrCpuFoundation& owner,
    PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction& token) noexcept;
[[gnu::visibility("hidden")]] void RequestW8cInitialization(
    const PersistentSc25ScalarIgrCpuFoundation& owner) noexcept;
}

namespace
{

using namespace dg::hdg::runtime;
namespace cpu = dg::hdg::runtime::cpu;
namespace sc25 = dg::hdg::runtime::cpu::sc25;
namespace w8c = dg::hdg::runtime::cpu::w8c;

bool Near(const double left, const double right) noexcept
{
  return std::abs(left - right) <=
      64.0 * std::numeric_limits<double>::epsilon() *
          std::max({1.0, std::abs(left), std::abs(right)});
}

struct W8cSelection : CpuSerialRuntimeSelection
{
  static constexpr std::string_view experiment_id = w8c::experiment_id;
};

struct Checks final
{
  void Require(const bool condition, const char* message)
  {
    if (condition) return;
    ++failures;
    std::cerr << message << '\n';
  }
  int failures = 0;
};

bool KnownCellCoefficients(const double* values, const std::size_t count)
{
  constexpr std::size_t elements_per_direction = 8;
  constexpr std::size_t scalar_dofs = 4;
  if (!values || count != elements_per_direction * elements_per_direction *
                           scalar_dofs * 4)
    return false;
  const std::size_t scalar = count / 4;
  for (int component = 0; component < 4; ++component)
    for (std::size_t index = 0; index < scalar; ++index)
    {
      const double value = values[static_cast<std::size_t>(component) * scalar +
                                  index];
      const std::size_t element = index / scalar_dofs;
      const std::size_t ex = element % elements_per_direction;
      const double x = (static_cast<double>(ex) + 0.5) /
          static_cast<double>(elements_per_direction);
      const auto expected = dg_igr_hdg_cpu_test::W8cCellState(x);
      if (!Near(value, expected[static_cast<std::size_t>(component)]))
        return false;
    }
  return true;
}

bool KnownScalarCellCoefficients(const sc25::PersistentTrajectoryState& state)
{
  constexpr std::size_t elements = 64;
  constexpr std::size_t scalar_dofs = 4;
  if (state.u.size() != elements * 4 * scalar_dofs) return false;
  for (std::size_t element = 0; element < elements; ++element)
    for (int component = 0; component < 4; ++component)
      for (std::size_t scalar = 0; scalar < scalar_dofs; ++scalar)
      {
        const double value = state.u[
            element * 4 * scalar_dofs +
            static_cast<std::size_t>(component) * scalar_dofs + scalar];
        const double x = (static_cast<double>(element % 8) + 0.5) / 8.0;
        const auto expected = dg_igr_hdg_cpu_test::W8cCellState(x);
        if (!Near(value, expected[static_cast<std::size_t>(component)]))
          return false;
      }
  return true;
}

std::vector<double> ScalarUComponentMajor(
    const sc25::PersistentTrajectoryState& state)
{
  constexpr std::size_t elements = 64;
  constexpr std::size_t scalar_dofs = 4;
  std::vector<double> packed(state.u.size());
  for (std::size_t element = 0; element < elements; ++element)
    for (std::size_t component = 0; component < 4; ++component)
      for (std::size_t scalar = 0; scalar < scalar_dofs; ++scalar)
        packed[component * elements * scalar_dofs +
               element * scalar_dofs + scalar] =
            state.u[element * 4 * scalar_dofs +
                    component * scalar_dofs + scalar];
  return packed;
}

bool CloseArrays(const double* left, const double* right,
                 const std::size_t count) noexcept
{
  if ((!left || !right) && count != 0) return false;
  for (std::size_t index = 0; index < count; ++index)
    if (!Near(left[index], right[index])) return false;
  return true;
}

bool SameScalarStateBytes(const sc25::PersistentTrajectoryState& left,
                          const sc25::PersistentTrajectoryState& right) noexcept
{
  const auto equal = [](const std::vector<double>& a,
                        const std::vector<double>& b) noexcept
  {
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(),
                                  a.size() * sizeof(double)) == 0);
  };
  return equal(left.u, right.u) && equal(left.sigma, right.sigma) &&
      equal(left.z, right.z) && equal(left.uhat, right.uhat) &&
      equal(left.sigmahat, right.sigmahat);
}

bool ScalarAccepted(const sc25::TrajectoryStatus status) noexcept
{
  return status == sc25::TrajectoryStatus::HdgSc25Accepted ||
      status == sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
}

bool FinalConventionalUpdateRejectedEveryTrial(
    const cpu::CpuAttemptEvidence& attempt) noexcept
{
  return !attempt.committed && attempt.rollback_byte_equal &&
      attempt.failure_phase == cpu::CpuFailurePhase::recovery &&
      !attempt.admissibility_gate && attempt.work.complete_assemblies > 0 &&
      attempt.final_line_search_trials ==
          cpu::cpu_maximum_line_search_trials &&
      attempt.work.trial_reassemblies == 0 &&
      attempt.work.complete_condensations >=
          attempt.work.accepted_newton_updates &&
      attempt.work.complete_factor_setups ==
          attempt.work.complete_trace_solves &&
      attempt.work.complete_trace_solves >=
          attempt.work.accepted_newton_updates;
}

template <typename Range>
bool PositiveZero(const Range& values)
{
  return std::all_of(values.begin(), values.end(), [](const double value) {
    return value == 0.0 && !std::signbit(value);
  });
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  Checks checks;
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  checks.Require(ranks == 1, "W8C component test requires CPU serial");

  cpu::CpuPersistentConfig config = cpu::MakeCpuSmokeConfig();
  config.experiment = StableIdentity::Copy(w8c::experiment_id);
  config.source = StableIdentity::Copy(w8c::conventional_source_id);
  config.polynomial_order = w8c::polynomial_order;
  config.elements_per_direction = static_cast<int>(w8c::mesh_n);
  config.assembly_quadrature_order = w8c::conventional_q_a;
  config.dt = w8c::dt;
  cpu::PersistentSmoothEulerCpuAdapter<W8cSelection> conventional;
  const bool conventional_ready = conventional.Initialize(config);
  checks.Require(conventional_ready,
                 "conventional W8C N8 initial projection must initialize");
  const cpu::CpuStateSnapshot conventional_state = conventional.Snapshot();
  dg_igr_hdg_cpu_test::W8cTraceGeometryEvidence conventional_trace_evidence;
  const bool conventional_trace_exact = conventional_ready &&
      dg_igr_hdg_cpu_test::VerifyW8cTraceGeometryAndState(
          MPI_COMM_WORLD, conventional_state.trace.data,
          conventional_state.trace.size, conventional_trace_evidence);
  checks.Require(conventional_ready &&
                     KnownCellCoefficients(conventional_state.interior.data,
                                           conventional_state.interior.size),
                 "conventional W8C initial cell known answer");
  checks.Require(conventional_trace_exact,
                 "conventional W8C trace must match independently rebuilt "
                 "face geometry, signed true DOFs, and orientations");
  checks.Require(conventional.evidence().elements_per_direction == 8 &&
                     conventional.evidence().polynomial_order == 1 &&
                     conventional.evidence().assembly_quadrature_order == 11 &&
                     conventional.evidence().dt == w8c::dt,
                 "conventional W8C tuple must remain N8/k1/q11/dt2^-10");

  sc25::FoundationConfig foundation;
  foundation.polynomial_degree = 1;
  foundation.elements_per_direction = 8;

  sc25::FoundationConfig failed_w8c_foundation = foundation;
  failed_w8c_foundation.polynomial_degree = 2;
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
      failed_w8c_owner;
  const bool failed_w8c_created =
      sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, sc25::CpuSerialTag{}, failed_w8c_foundation,
          failed_w8c_owner) == sc25::FoundationStatus::ready &&
      failed_w8c_owner;
  if (failed_w8c_created)
    sc25::w8c_internal::RequestW8cInitialization(*failed_w8c_owner);
  const bool requested_w8c_failed = failed_w8c_created &&
      failed_w8c_owner->InitializePersistentTrajectory() ==
          sc25::TrajectoryStatus::invalid_configuration;

  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
      ordinary_w8b_owner;
  const bool ordinary_w8b_created =
      sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, sc25::CpuSerialTag{}, foundation,
          ordinary_w8b_owner) == sc25::FoundationStatus::ready &&
      ordinary_w8b_owner;
  const bool ordinary_w8b_ready = ordinary_w8b_created &&
      ordinary_w8b_owner->InitializePersistentTrajectory() ==
          sc25::TrajectoryStatus::ready;
  sc25::PersistentTrajectoryState ordinary_w8b_state;
  const bool ordinary_w8b_copied = ordinary_w8b_ready &&
      ordinary_w8b_owner->CopyPublishedTrajectoryState(ordinary_w8b_state);
  checks.Require(requested_w8c_failed && ordinary_w8b_copied &&
                     !KnownScalarCellCoefficients(ordinary_w8b_state),
                 "failed typed W8C selection must clear before an ordinary "
                 "same-process W8B initialization");

  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> scalar;
  const bool scalar_created =
      sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, sc25::CpuSerialTag{}, foundation, scalar) ==
          sc25::FoundationStatus::ready && scalar;
  if (scalar_created)
    sc25::w8c_internal::RequestW8cInitialization(*scalar);
  const bool scalar_ready = scalar_created &&
      scalar->InitializePersistentTrajectory() == sc25::TrajectoryStatus::ready;
  checks.Require(scalar_ready,
                 "traced-scalar-IGR W8C N8 initial projection must initialize");
  sc25::PersistentTrajectoryState scalar_state;
  const bool scalar_copied = scalar_ready &&
      scalar->CopyPublishedTrajectoryState(scalar_state);
  dg_igr_hdg_cpu_test::W8cTraceGeometryEvidence scalar_trace_evidence;
  const bool scalar_trace_exact = scalar_copied &&
      dg_igr_hdg_cpu_test::VerifyW8cTraceGeometryAndState(
          MPI_COMM_WORLD, scalar_state.uhat.data(), scalar_state.uhat.size(),
          scalar_trace_evidence);
  checks.Require(scalar_copied &&
                     KnownScalarCellCoefficients(scalar_state),
                 "scalar-IGR W8C U initial known answer");
  checks.Require(scalar_trace_exact,
                 "scalar-IGR W8C Uhat must match independently rebuilt face "
                 "geometry, signed true DOFs, and orientations");
  checks.Require(scalar_copied && PositiveZero(scalar_state.sigma) &&
                     PositiveZero(scalar_state.z) &&
                     PositiveZero(scalar_state.sigmahat),
                 "scalar-IGR Sigma/z/Sigmahat must initialize to positive zero");
  checks.Require(scalar->trajectory_evidence()
                         .initialization.residual.normalized_maximum <= 1.0e-10 &&
                     scalar->trajectory_evidence()
                             .initialization.trace_true_residual_norm == 0.0 &&
                     scalar->trajectory_evidence()
                             .initialization.trace_right_hand_side_norm == 0.0,
                 "zero scalar auxiliary initialization must pass inherited residual gates");
  const std::vector<double> scalar_u = scalar_copied
      ? ScalarUComponentMajor(scalar_state) : std::vector<double>{};
  checks.Require(scalar_copied &&
                     scalar_state.u.size() == conventional_state.interior.size &&
                     scalar_state.uhat.size() == conventional_state.trace.size &&
                     CloseArrays(scalar_u.data(),
                                 conventional_state.interior.data,
                                 scalar_u.size()) &&
                     CloseArrays(scalar_state.uhat.data(),
                                 conventional_state.trace.data,
                                 scalar_state.uhat.size()),
                 "none and traced-scalar-IGR must share exact initial U/Uhat");

  AttemptOutcome conventional_terminal{};
  cpu::CpuStateSnapshot conventional_before_failure{};
  while (conventional.coordinates().step < w8c::accepted_steps)
  {
    conventional_before_failure = conventional.Snapshot();
    conventional_terminal = conventional.Advance();
    if (!conventional_terminal.committed) break;
  }
  const cpu::CpuStateSnapshot conventional_after_failure =
      conventional.Snapshot();
  const cpu::CpuAttemptEvidence& conventional_failed_attempt =
      conventional.evidence().last_attempt;
  checks.Require(!conventional_terminal.committed &&
                     conventional_terminal.reason ==
                         FailureReason::admissibility_gate &&
                     conventional.coordinates().step == 9 &&
                     conventional_before_failure.interior_hash ==
                         conventional_after_failure.interior_hash &&
                     conventional_before_failure.trace_hash ==
                         conventional_after_failure.trace_hash &&
                     conventional_failed_attempt.rollback_byte_equal,
                 "conventional negative trajectory must preserve committed step 9");
  checks.Require(conventional_failed_attempt.failure_phase ==
                         cpu::CpuFailurePhase::recovery &&
                     FinalConventionalUpdateRejectedEveryTrial(
                         conventional_failed_attempt),
                 "conventional target step 10 must report 13 rejected admissibility trials");

  sc25::TrajectoryStatus scalar_terminal = sc25::TrajectoryStatus::ready;
  sc25::PersistentTrajectoryState scalar_before_failure{};
  sc25::PersistentTrajectoryState scalar_after_failure{};
  sc25::PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction transaction;
  checks.Require(sc25::w8c_internal::PrimePersistentTrajectoryTransaction(
                     *scalar, transaction),
                 "scalar W8C reusable transaction prime");
  while (scalar->trajectory_evidence().coordinates.step <
         w8c::accepted_steps)
  {
    checks.Require(scalar->CopyPublishedTrajectoryState(
                       scalar_before_failure),
                   "scalar W8C pre-attempt state copy");
    scalar_terminal = scalar->PreparePersistentTrajectoryTransaction(
        transaction, sc25::TrajectoryFailureInjection::none);
    if (!ScalarAccepted(scalar_terminal)) break;
    const bool installed =
        scalar->InstallPersistentTrajectoryTransaction(transaction);
    const bool finalized = installed &&
        scalar->FinalizePersistentTrajectoryTransaction(std::move(transaction));
    checks.Require(finalized, "scalar W8C transaction must finalize");
    if (!finalized) break;
  }
  const bool scalar_after_copied =
      scalar->CopyPublishedTrajectoryState(scalar_after_failure);
  const sc25::PersistentTrajectoryEvidence& scalar_evidence =
      scalar->trajectory_evidence();
  const sc25::TrajectoryAttemptEvidence& scalar_failed_attempt =
      scalar_evidence.last_attempt;
  checks.Require(scalar_after_copied &&
                     scalar_terminal ==
                         sc25::TrajectoryStatus::HdgSc25AdmissibilityFailure &&
                     scalar_evidence.contract_last_failure ==
                         FailureReason::admissibility_gate &&
                     scalar_evidence.coordinates.step == 10 &&
                     scalar_evidence.retry_attempts == 0 &&
                     SameScalarStateBytes(scalar_before_failure,
                                          scalar_after_failure),
                 "scalar negative trajectory must preserve committed step 10 without a hidden retry");
  checks.Require(scalar_failed_attempt.target_step == 11 &&
                     scalar_failed_attempt.final_line_search_trials ==
                         sc25::persistent_line_search_trials &&
                     scalar_failed_attempt.attempted_newton_iterations ==
                         scalar_failed_attempt.newton_iterations &&
                     scalar_failed_attempt.newton_iterations > 0 &&
                     scalar_failed_attempt.newton_iterations <=
                         sc25::persistent_newton_maximum_iterations,
                 "scalar target step 11 must not silently accept an inadmissible trial");

  MPI_Finalize();
  return checks.failures == 0 ? 0 : 1;
}
