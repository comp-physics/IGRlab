#include "euler_committed_observable_oracle.hpp"

#include "hdg_cpu_serial_lifecycle.hpp"
#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace dg::hdg::runtime::cpu::sc25::w8c_internal
{
[[gnu::visibility("hidden")]] void RequestSmoothTransitionInitialization(
    const PersistentSc25ScalarIgrCpuFoundation&) noexcept;
[[gnu::visibility("hidden")]] void RequestSmoothTransitionRank2Initialization(
    const PersistentSc25ScalarIgrCpuFoundation&) noexcept;
} // namespace dg::hdg::runtime::cpu::sc25::w8c_internal

namespace
{

namespace generic = dg::hdg::generic;
namespace cpu = dg::hdg::runtime::cpu;
namespace sc25 = dg::hdg::runtime::cpu::sc25;
namespace transfer = dg_igr_hdg_reference_transfer_test;

constexpr std::uint32_t generic_cpu_capabilities =
    generic::CapabilityBit(generic::Capability::conforming_quadrilateral_mesh) |
    generic::CapabilityBit(generic::Capability::polynomial_order_one_to_three) |
    generic::CapabilityBit(generic::Capability::cpu) |
    generic::CapabilityBit(generic::Capability::external_initial_state) |
    generic::CapabilityBit(generic::Capability::external_source) |
    generic::CapabilityBit(generic::Capability::numerical_boundary_operator);

// These are reference-transfer acceptance tolerances, not scientific-error
// tolerances. They compare the generic CPU path to the retained frozen SC25
// oracle on the same N8/p1/dt trajectory.
constexpr double transfer_observable_absolute_tolerance = 5.0e-8;
constexpr double transfer_observable_relative_tolerance = 5.0e-7;
constexpr double transfer_state_absolute_tolerance = 5.0e-8;
constexpr double transfer_state_relative_tolerance = 5.0e-7;

bool All(const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

bool Close(const double left, const double right, const double absolute,
           const double relative) noexcept
{
  return std::isfinite(left) && std::isfinite(right) &&
      std::abs(left - right) <=
          absolute + relative * std::max(std::abs(left), std::abs(right));
}

bool PeriodicExterior(const void*, const generic::BoundaryPointView&,
                      const double* interior, double* exterior) noexcept
{
  if (!interior || !exterior) return false;
  for (int component = 0; component < 4; ++component)
    exterior[component] = interior[component];
  return true;
}

generic::EulerHdgRequest MakeRequest(
    const mfem::ParMesh& mesh, const std::vector<double>& initial,
    const std::vector<double>& true_trace, const std::vector<double>& source)
{
  generic::EulerHdgNumericalPolicy policy =
      generic::DefaultEulerHdgNumericalPolicy(transfer::n8_polynomial_order);
  policy.volume_projection_quadrature_order = transfer::n8_projection_quadrature_order;
  policy.true_trace_projection_quadrature_order = transfer::n8_projection_quadrature_order;
  policy.conventional_assembly_quadrature_order =
      cpu::w8c::SmoothTransitionV2Case::conventional_q_a;
  policy.igr_assembly_quadrature_orders = {
      cpu::w8c::SmoothTransitionV2Case::scalar_q_a,
      cpu::w8c::SmoothTransitionV2Case::scalar_q_a};
  policy.maximum_nonlinear_updates = sc25::persistent_newton_maximum_iterations;
  policy.maximum_linear_iterations = sc25::persistent_trace_maximum_iterations;
  policy.maximum_line_search_trials = sc25::persistent_line_search_trials;
  policy.nonlinear_relative_tolerance = 1.0e-12;
  policy.linear_relative_tolerance = 1.0e-12;
  policy.true_residual_relative_tolerance = 1.0e-12;
  policy.conservation_relative_tolerance = 1.0e-10;
  policy.armijo_constant = 1.0e-4;
  policy.stabilization = {generic::RegularizationKind::igr,
                           sc25::primary_alpha,
                           sc25::state_stabilization};
  return {&mesh,
          mesh.GetComm(),
          transfer::n8_polynomial_order,
          {generic::BackendKind::cpu,
           -1,
           generic::CudaEngineKind::ordinary},
          generic_cpu_capabilities,
          {initial.data(), initial.size(), 4},
          {true_trace.data(), true_trace.size(), 4},
          {source.data(), source.size(), 4},
          {nullptr, PeriodicExterior, {nullptr, 0, 0, 0.0}},
          policy};
}

bool Sc25Accepted(const sc25::TrajectoryStatus status) noexcept
{
  return status == sc25::TrajectoryStatus::HdgSc25Accepted ||
      status == sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
}

bool ObservablesTransferMatch(
    const transfer::EulerCommittedObservableOracle& generic_observable,
    const transfer::EulerCommittedObservableOracle& sc25_observable) noexcept
{
  if (!generic_observable.available || !sc25_observable.available ||
      generic_observable.quadrature_order != sc25_observable.quadrature_order)
    return false;
  for (int component = 0; component < 4; ++component)
    if (!Close(generic_observable.conserved_integrals[static_cast<std::size_t>(component)],
               sc25_observable.conserved_integrals[static_cast<std::size_t>(component)],
               transfer_observable_absolute_tolerance,
               transfer_observable_relative_tolerance) ||
        !Close(generic_observable.conserved_l2_norms[static_cast<std::size_t>(component)],
               sc25_observable.conserved_l2_norms[static_cast<std::size_t>(component)],
               transfer_observable_absolute_tolerance,
               transfer_observable_relative_tolerance))
      return false;
  return Close(generic_observable.minimum_density, sc25_observable.minimum_density,
               transfer_observable_absolute_tolerance,
               transfer_observable_relative_tolerance) &&
      Close(generic_observable.minimum_pressure, sc25_observable.minimum_pressure,
            transfer_observable_absolute_tolerance,
            transfer_observable_relative_tolerance);
}

template <typename Coordinates>
bool CoordinatesMatch(const Coordinates& coordinates,
                      const std::uint64_t step) noexcept
{
  const double expected_time = static_cast<double>(step) *
      cpu::w8c::SmoothTransitionV2Case::dt;
  const double expected_dt = step == 0
      ? 0.0 : cpu::w8c::SmoothTransitionV2Case::dt;
  return coordinates.step == step && coordinates.generation == step &&
      Close(coordinates.time, expected_time, 1.0e-15, 1.0e-15) &&
      Close(coordinates.dt, expected_dt, 1.0e-15, 1.0e-15);
}

bool Sc25CoordinatesMatch(
    const dg::hdg::runtime::StateCoordinates& coordinates,
    const std::uint64_t step, const int ranks) noexcept
{
  const double expected_time = static_cast<double>(step) *
      cpu::w8c::SmoothTransitionV2Case::dt;
  // The frozen rank-two leaf preloads dt at initialization; the rank-one leaf
  // and the generic session expose zero until the first committed step.
  const double expected_dt = step == 0 && ranks == 1
      ? 0.0 : cpu::w8c::SmoothTransitionV2Case::dt;
  return coordinates.step == step && coordinates.generation == step &&
      coordinates.cursor == step &&
      Close(coordinates.time, expected_time, 1.0e-15, 1.0e-15) &&
      Close(coordinates.dt, expected_dt, 1.0e-15, 1.0e-15);
}

bool CheckTransferCheckpoint(
    cpu::GenericEulerHdgCpuSession& generic_session,
    mfem::ParGridFunction& generic_field, const mfem::ParMesh& generic_mesh,
    const std::vector<double>& sc25_state,
    const dg::hdg::runtime::StateCoordinates& sc25_coordinates,
    const std::uint64_t step)
{
  const bool exported = generic_session.ExportCommittedInterior(generic_field);
  transfer::EulerCommittedObservableOracle generic_oracle{};
  transfer::EulerCommittedObservableOracle scalar_oracle{};
  double maximum_absolute_error = 0.0;
  double maximum_relative_error = 0.0;
  const bool oracle_ready = exported &&
      transfer::EvaluateCommittedEulerObservables(
          generic_field, transfer::n8_observable_quadrature_order,
          generic_oracle) &&
      transfer::EvaluateSc25ElementMajorObservables(
          generic_mesh, sc25_state, transfer::n8_observable_quadrature_order,
          scalar_oracle) &&
      transfer::CompareGenericFieldToSc25ElementMajor(
          generic_field, sc25_state, maximum_absolute_error,
          maximum_relative_error);
  const bool coordinates = CoordinatesMatch(generic_session.coordinates(), step) &&
      Sc25CoordinatesMatch(sc25_coordinates, step, generic_mesh.GetNRanks());
  const bool transfer_matches =
      ObservablesTransferMatch(generic_oracle, scalar_oracle);
  const bool state_absolute_matches =
      maximum_absolute_error <= transfer_state_absolute_tolerance;
  const bool state_relative_matches =
      maximum_relative_error <= transfer_state_relative_tolerance;
  const bool passed = oracle_ready && coordinates && transfer_matches &&
      state_absolute_matches && state_relative_matches;
  if (!passed)
  {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      std::cerr << "N8 reference-transfer checkpoint failed step=" << step
                << " oracle=" << oracle_ready
                << " coordinates=" << coordinates
                << " transfer_observables=" << transfer_matches
                << " generic_oracle_available=" << generic_oracle.available
                << " scalar_oracle_available=" << scalar_oracle.available
                << " state_abs=" << maximum_absolute_error
                << " state_rel=" << maximum_relative_error << '\n';
  }
  return All(passed);
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  bool valid = ranks == 1 || ranks == 2;

  std::unique_ptr<mfem::ParMesh> generic_mesh;
  valid = valid && transfer::MakeN8PeriodicMesh(MPI_COMM_WORLD, generic_mesh);
  std::vector<double> initial, true_trace;
  valid = valid && generic_mesh && transfer::ProjectN8SmoothTransition(
      *generic_mesh, initial, true_trace);
  std::vector<double> source(initial.size(), 0.0);

  mfem::L2_FECollection generic_collection(
      transfer::n8_polynomial_order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  std::unique_ptr<mfem::ParFiniteElementSpace> generic_space;
  std::unique_ptr<mfem::ParGridFunction> generic_field;
  if (valid)
  {
    generic_space = std::make_unique<mfem::ParFiniteElementSpace>(
        generic_mesh.get(), &generic_collection, 4, mfem::Ordering::byNODES);
    generic_field = std::make_unique<mfem::ParGridFunction>(generic_space.get());
  }

  cpu::GenericEulerHdgCpuSession generic_session;
  const cpu::GenericEulerCpuSessionResult generic_initialized = valid
      ? generic_session.Initialize(MakeRequest(*generic_mesh, initial, true_trace, source))
      : cpu::GenericEulerCpuSessionResult{};
  valid = valid && generic_initialized.status ==
      cpu::GenericEulerCpuSessionStatus::initialized;

  sc25::FoundationConfig config{};
  config.polynomial_degree = transfer::n8_polynomial_order;
  config.elements_per_direction = 8;
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> scalar;
  sc25::FoundationStatus foundation_status = sc25::FoundationStatus::uninitialized;
  if (valid)
    foundation_status = ranks == 1
        ? sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
              MPI_COMM_WORLD, sc25::CpuSerialTag{}, config, scalar)
        : sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
              MPI_COMM_WORLD, sc25::CpuMpiRank2Tag{}, config, scalar);
  valid = valid && foundation_status == sc25::FoundationStatus::ready && scalar;
  sc25::TrajectoryStatus scalar_initialized = sc25::TrajectoryStatus::uninitialized;
  if (valid)
  {
    if (ranks == 1)
    {
      sc25::w8c_internal::RequestSmoothTransitionInitialization(*scalar);
      scalar_initialized = scalar->InitializePersistentTrajectory();
    }
    else
    {
      sc25::w8c_internal::RequestSmoothTransitionRank2Initialization(*scalar);
      scalar_initialized = scalar->InitializePersistentRank2Trajectory();
    }
  }
  valid = valid && scalar_initialized == sc25::TrajectoryStatus::ready;

  std::vector<double> scalar_state;
  dg::hdg::runtime::StateCoordinates scalar_coordinates{};
  if (valid)
  {
    if (ranks == 1)
    {
      sc25::PersistentTrajectoryState state;
      valid = scalar->CopyPublishedTrajectoryState(state);
      scalar_state = std::move(state.u);
      scalar_coordinates = scalar->trajectory_evidence().coordinates;
    }
    else
    {
      sc25::PersistentRank2TrajectoryState state;
      valid = scalar->CopyPublishedRank2TrajectoryState(state);
      scalar_state = std::move(state.u);
      scalar_coordinates = scalar->rank2_trajectory_evidence().coordinates;
    }
  }
  valid = valid && CheckTransferCheckpoint(
      generic_session, *generic_field, *generic_mesh, scalar_state,
      scalar_coordinates, 0);

  for (std::uint64_t step = 1;
       step <= cpu::w8c::SmoothTransitionV2Case::accepted_steps && valid;
       ++step)
  {
    const auto generic_step = generic_session.Advance(
        cpu::w8c::SmoothTransitionV2Case::dt);
    valid = generic_step.status == cpu::GenericEulerCpuSessionStatus::advanced;
    sc25::TrajectoryStatus scalar_step = sc25::TrajectoryStatus::uninitialized;
    if (valid)
      scalar_step = ranks == 1
          ? scalar->AdvancePersistentTrajectory()
          : scalar->AdvancePersistentRank2Trajectory();
    valid = valid && Sc25Accepted(scalar_step);
    if (!valid) break;
    if (step == 32 || step == 64)
    {
      if (ranks == 1)
      {
        sc25::PersistentTrajectoryState state;
        valid = scalar->CopyPublishedTrajectoryState(state);
        scalar_state = std::move(state.u);
        scalar_coordinates = scalar->trajectory_evidence().coordinates;
      }
      else
      {
        sc25::PersistentRank2TrajectoryState state;
        valid = scalar->CopyPublishedRank2TrajectoryState(state);
        scalar_state = std::move(state.u);
        scalar_coordinates = scalar->rank2_trajectory_evidence().coordinates;
      }
      valid = valid && CheckTransferCheckpoint(
          generic_session, *generic_field, *generic_mesh, scalar_state,
          scalar_coordinates, step);
    }
  }

  const bool collective_valid = All(valid);
  MPI_Finalize();
  return collective_valid ? 0 : 1;
}
