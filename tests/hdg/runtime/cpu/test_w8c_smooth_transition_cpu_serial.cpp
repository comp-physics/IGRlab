#include "independent_oracle.hpp"
#include "sc25_scalar_igr_independent_oracle.hpp"

#include "hdg_cpu_serial_lifecycle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace dg::hdg::runtime::cpu::sc25::w8c_internal
{
[[gnu::visibility("hidden")]] void RequestSmoothTransitionInitialization(
    const PersistentSc25ScalarIgrCpuFoundation& owner) noexcept;
}

namespace
{

using namespace dg::hdg::runtime;
namespace cpu = dg::hdg::runtime::cpu;
namespace sc25 = dg::hdg::runtime::cpu::sc25;
using SmoothCase = cpu::w8c::SmoothTransitionV2Case;

struct SmoothSelection : CpuSerialRuntimeSelection
{
  static constexpr std::string_view experiment_id = SmoothCase::experiment_id;
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

bool Close(const double actual, const double expected) noexcept
{
  return std::abs(actual - expected) <=
      2048.0 * std::numeric_limits<double>::epsilon() *
          std::max({1.0, std::abs(actual), std::abs(expected)});
}

bool CloseArrays(const double* actual, const std::vector<double>& expected)
    noexcept
{
  if (!actual && !expected.empty()) return false;
  for (std::size_t index = 0; index < expected.size(); ++index)
    if (!Close(actual[index], expected[index])) return false;
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

template <typename Range>
bool PositiveZero(const Range& values)
{
  return std::all_of(values.begin(), values.end(), [](const double value)
  {
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
  checks.Require(ranks == 1,
                 "smooth-transition component test requires CPU serial");

  checks.Require(SmoothCase::case_id == "periodic-smooth-transition-v2" &&
                     SmoothCase::scenario_id == "W8C-V2-N8-DT10" &&
                     SmoothCase::mesh_n == 8 &&
                     SmoothCase::polynomial_order == 1 &&
                     SmoothCase::projection_q == 14 &&
                     SmoothCase::observable_q_o == 18 &&
                     SmoothCase::accepted_steps == 64 &&
                     SmoothCase::output_steps ==
                         std::array<std::uint64_t, 3>{{0, 32, 64}},
                 "smooth-transition descriptor must remain frozen and private");

  std::vector<double> oracle_interior;
  std::vector<double> oracle_trace;
  dg_igr_hdg_cpu_test::SmoothTransitionProjectionEvidence oracle_evidence;
  const bool oracle_ready =
      dg_igr_hdg_cpu_test::BuildSmoothTransitionV2ProjectionOracle(
          MPI_COMM_WORLD, oracle_interior, oracle_trace, oracle_evidence);
  checks.Require(oracle_ready &&
                     Close(oracle_evidence.density_integral, 0.5625) &&
                     Close(oracle_evidence.x_momentum_integral, 0.0) &&
                     Close(oracle_evidence.y_momentum_integral, 0.0) &&
                     Close(oracle_evidence.energy_integral, 1.375) &&
                     oracle_evidence.maximum_reflection_defect <= 1.0e-14 &&
                     oracle_evidence.maximum_half_period_complement_defect <=
                         1.0e-14 &&
                     oracle_evidence.maximum_y_invariance_defect == 0.0 &&
                     oracle_evidence.signed_permuted_trace_entries == 2048 &&
                     oracle_evidence.permuted_trace_entries == 1024 &&
                     oracle_evidence.doubly_covered_trace_dofs == 1024 &&
                     oracle_evidence.maximum_signed_trace_coefficient_defect <=
                         1.0e-14 &&
                     oracle_evidence.maximum_projected_trace_value_defect <=
                         1.0e-14,
                 "independent q14/q18 oracle must recover projections, ownership, means, and symmetries");

  cpu::CpuPersistentConfig conventional_config = cpu::MakeCpuSmokeConfig();
  conventional_config.experiment = StableIdentity::Copy(
      SmoothCase::experiment_id);
  conventional_config.source = StableIdentity::Copy(
      SmoothCase::conventional_source_id);
  conventional_config.polynomial_order = SmoothCase::polynomial_order;
  conventional_config.elements_per_direction =
      static_cast<int>(SmoothCase::mesh_n);
  conventional_config.assembly_quadrature_order = SmoothCase::conventional_q_a;
  conventional_config.dt = SmoothCase::dt;
  cpu::PersistentSmoothEulerCpuAdapter<SmoothSelection> conventional;
  const bool conventional_ready = conventional.Initialize(conventional_config);
  const cpu::CpuStateSnapshot conventional_state = conventional.Snapshot();
  checks.Require(conventional_ready &&
                     conventional_state.interior.size == oracle_interior.size() &&
                     conventional_state.trace.size == oracle_trace.size() &&
                     CloseArrays(conventional_state.interior.data,
                                 oracle_interior) &&
                     CloseArrays(conventional_state.trace.data, oracle_trace),
                 "conventional smooth U/Uhat must match independent L2 oracles");

  sc25::FoundationConfig foundation;
  foundation.polynomial_degree = SmoothCase::polynomial_order;
  foundation.elements_per_direction = static_cast<int>(SmoothCase::mesh_n);
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> scalar;
  const bool scalar_created =
      sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, sc25::CpuSerialTag{}, foundation, scalar) ==
          sc25::FoundationStatus::ready && scalar;
  if (scalar_created)
    sc25::w8c_internal::RequestSmoothTransitionInitialization(*scalar);
  const sc25::TrajectoryStatus scalar_status = scalar_created
      ? scalar->InitializePersistentTrajectory()
      : sc25::TrajectoryStatus::uninitialized;
  const bool scalar_ready = scalar_status == sc25::TrajectoryStatus::ready;
  if (!scalar_ready)
    std::cerr << "smooth scalar initialization status="
              << static_cast<int>(scalar_status) << '\n';
  sc25::PersistentTrajectoryState scalar_state;
  const bool scalar_copied = scalar_ready &&
      scalar->CopyPublishedTrajectoryState(scalar_state);
  const std::vector<double> scalar_u = scalar_copied
      ? ScalarUComponentMajor(scalar_state) : std::vector<double>{};
  checks.Require(scalar_copied && scalar_u.size() == oracle_interior.size() &&
                     CloseArrays(scalar_u.data(), oracle_interior) &&
                     CloseArrays(scalar_state.uhat.data(), oracle_trace),
                 "scalar smooth U/Uhat must match independent L2 oracles");
  checks.Require(scalar_copied && conventional_ready &&
                     scalar_u.size() == conventional_state.interior.size &&
                     scalar_state.uhat.size() == conventional_state.trace.size &&
                     std::memcmp(scalar_u.data(),
                                 conventional_state.interior.data,
                                 scalar_u.size() * sizeof(double)) == 0 &&
                     std::memcmp(scalar_state.uhat.data(),
                                 conventional_state.trace.data,
                                 scalar_state.uhat.size() * sizeof(double)) == 0,
                 "none and scalar policies must start from byte-identical U/Uhat");
  sc25::AuxiliaryResidualMetrics independent_aux_q12;
  sc25::AuxiliaryResidualMetrics independent_aux_q14;
  const bool independent_auxiliary_ready = scalar_copied &&
      dg_igr_hdg_sc25_cpu_test::EvaluateIndependentAuxiliaryResidualOracle(
          *scalar, scalar_state, 12, independent_aux_q12) &&
      dg_igr_hdg_sc25_cpu_test::EvaluateIndependentAuxiliaryResidualOracle(
          *scalar, scalar_state, 14, independent_aux_q14);
  checks.Require(scalar_copied && PositiveZero(scalar_state.sigma) &&
                     PositiveZero(scalar_state.z) &&
                     PositiveZero(scalar_state.sigmahat),
                 "smooth auxiliary fields must be homogeneous positive zero");
  checks.Require(independent_auxiliary_ready &&
                     independent_aux_q12.normalized_maximum <= 1.0e-10 &&
                     independent_aux_q14.normalized_maximum <= 1.0e-10,
                 "complete independent q12/q14 auxiliary weak residual must be homogeneous");
  const sc25::AuxiliaryInitializationEvidence* auxiliary = scalar_ready
      ? &scalar->trajectory_evidence().initialization : nullptr;
  checks.Require(auxiliary && auxiliary->quadrature_order == 12 &&
                     auxiliary->trace_restart_dimension == 50 &&
                     auxiliary->trace_maximum_iterations == 200 &&
                     auxiliary->trace_iterations >= 0 &&
                     auxiliary->trace_true_residual_norm == 0.0 &&
                     auxiliary->trace_right_hand_side_norm == 0.0 &&
                     auxiliary->residual.normalized_maximum <= 1.0e-10 &&
                     independent_auxiliary_ready &&
                     std::abs(auxiliary->residual.normalized_maximum -
                              independent_aux_q12.normalized_maximum) <=
                         2.0e-11,
                 "smooth scalar initialization must complete the full auxiliary solve");

  MPI_Finalize();
  return checks.failures == 0 ? 0 : 1;
}
