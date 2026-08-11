#pragma once

/**
 * @file numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr_rank2.cuh
 * @brief Persistent CUDA rank-two leaf for the frozen SC25 projection-v3.
 *
 * This default-off leaf owns only the N=8, k=1, FP64, two-rank/one-GPU
 * experiment. MFEM constructs the
 * periodic mesh, signed trace restriction, and persistent HYPRE graph on the
 * host.  All numerical assembly, local factor/condensation, line-search
 * trials, recovery, checkpoint transposition, and reductions execute in the
 * three CUDA kernels declared by persistent_sc25_scalar_igr_rank2_impl.cuh.
 * Remote matrix/RHS contributions and the owned trace are the only numerical
 * data staged through host MPI; MPI never receives a CUDA pointer. The
 * distributed HYPRE GMRES+ILU(0) solve consumes device-resident data.
 */

#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"

#include <cuda_runtime.h>
#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8b::sc25
{

namespace cpu_sc25 = ::dg::hdg::runtime::cpu::sc25;
namespace cuda_w8a = ::dg::hdg::runtime::cuda_w8a;

inline constexpr std::string_view projection_id = cpu_sc25::projection_id;
inline constexpr std::string_view object_id = cpu_sc25::object_id;
inline constexpr std::string_view cuda_rank2_tuple_id =
    "HDG-SC25-PER-BE-CUDA-R2-HOST-STAGED-FP64-v3";
inline constexpr std::string_view cuda_rank2_fixture_id =
    "C3-MFEM-PERSISTENT-N8-CUDA-R2-HOST-STAGED-01";
inline constexpr std::string_view cuda_rank2_smooth_transition_fixture_id =
    "W8C-V2-MFEM-PERSISTENT-N8-CUDA-R2-HOST-STAGED-01";
inline constexpr std::string_view interior_layout_id =
    "device.field-blocked.U[e][c][a].Sigma[e][a].z[e][c][a].v3";
inline constexpr std::string_view trace_layout_id =
    "device.component-major.Uhat4.Sigmahat1.true-node.v3";
inline constexpr std::string_view checkpoint_layout_id =
    "checkpoint.component-major.seven-plus-five-slabs.v3";
inline constexpr std::string_view device_object_source_id =
    "src/numerics/hdg/runtime/cuda/hdg_sc25_periodic_v3_rank2_device.cu";
inline constexpr std::string_view mfem_pin =
    "d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec";

inline constexpr int dimension = cpu_sc25::dimension;
inline constexpr int state_components = cpu_sc25::state_components;
inline constexpr int flux_components = cpu_sc25::flux_components;
inline constexpr int local_components = cpu_sc25::local_components;
inline constexpr int trace_components = cpu_sc25::trace_components;
inline constexpr int volume_nodes = cpu_sc25::persistent_scalar_volume_nodes;
inline constexpr int face_nodes =
    cpu_sc25::coupled_fixture_scalar_face_nodes;
inline constexpr int local_interior_unknowns =
    cpu_sc25::coupled_fixture_local_interior_unknowns;
inline constexpr int local_trace_unknowns =
    cpu_sc25::coupled_fixture_local_trace_unknowns;
inline constexpr int persistent_elements =
    cpu_sc25::persistent_rank2_local_elements;
inline constexpr int persistent_scalar_trace_nodes =
    cpu_sc25::persistent_scalar_true_trace_nodes;
inline constexpr int persistent_u_unknowns =
    persistent_elements * state_components * volume_nodes;
inline constexpr int persistent_sigma_unknowns =
    persistent_elements * volume_nodes;
inline constexpr int persistent_z_unknowns =
    persistent_elements * flux_components * volume_nodes;
inline constexpr int persistent_interior_unknowns =
    persistent_u_unknowns + persistent_sigma_unknowns +
    persistent_z_unknowns;
inline constexpr int persistent_trace_unknowns =
    cpu_sc25::persistent_true_trace_unknowns;
inline constexpr int persistent_required_steps =
    cpu_sc25::persistent_required_steps;
inline constexpr int smooth_transition_required_steps =
    64;
inline constexpr int maximum_required_steps = smooth_transition_required_steps;
inline constexpr int persistent_max_owned_trace_unknowns =
    cpu_sc25::persistent_rank2_rank0_owned_trace_unknowns;
inline constexpr int persistent_max_full_unknowns =
    persistent_interior_unknowns + persistent_max_owned_trace_unknowns;
inline constexpr int newton_maximum_iterations =
    cpu_sc25::persistent_newton_maximum_iterations;
inline constexpr int line_search_trials =
    cpu_sc25::persistent_line_search_trials;
inline constexpr int trace_restart_dimension =
    cpu_sc25::persistent_trace_restart_dimension;
inline constexpr int trace_maximum_iterations =
    cpu_sc25::persistent_trace_maximum_iterations;
inline constexpr double fixed_dt = cpu_sc25::persistent_dt;
inline constexpr double smooth_transition_dt =
    1.0 / 1024.0;
inline constexpr double relative_trace_tolerance = 1.0e-12;
inline constexpr double absolute_trace_tolerance = 1.0e-14;
inline constexpr double armijo_coefficient = 1.0e-4;
inline constexpr int required_native_architecture = 1200;
inline constexpr int warm_plateau_cycles = 20;

static_assert(persistent_interior_unknowns == 896 &&
                  persistent_u_unknowns == 512 &&
                  persistent_sigma_unknowns == 128 &&
                  persistent_z_unknowns == 256,
              "private CUDA interior must be U/Sigma/z field-blocked");
static_assert(persistent_trace_unknowns == 1280 &&
                  cpu_sc25::persistent_uhat_unknowns == 1024 &&
                  cpu_sc25::persistent_sigmahat_unknowns == 256,
              "private CUDA trace must contain five 256-entry slabs");

struct CudaRank2Tag final
{};

struct CudaRank2Selection final
{
  using outer_model = cpu_sc25::HdgPublishedScalarPeriodicSc25ProjectionV3Tag;
  using source = cpu_sc25::TracedScalarSc25PeriodicExperimentV1Tag;
  using auxiliary_model =
      ::dg::hdg::auxiliary::FullyHybridizedScreenedPoissonV1Tag;
  using trace = cpu_sc25::DualConservativeScalarTraceV1Tag;
  using state_stabilization = cpu_sc25::W7ScalarTau25StateStabilizationTag;
  using auxiliary_stabilization =
      cpu_sc25::DensityHeightP2TraceStabilizationV2Tag;
  using time_treatment = cpu_sc25::BackwardEulerMonolithicV1Tag;
  using precision = cpu_sc25::Fp64Tag;
  using backend = CudaRank2Tag;

  static constexpr RuntimeTupleDescriptor Descriptor() noexcept
  {
    return {StableIdentity::Copy(projection_id),
            StableIdentity::Copy(object_id),
            StableIdentity::Copy("DualConservativeScalarTraceV1"),
            StableIdentity::Copy("W7Tau25.DensityHeightP2.v3"),
            StableIdentity::Copy("BackwardEulerMonolithicV1"),
            StableIdentity::Copy("CudaRank2OneGpuHostStaged"),
            StableIdentity::Copy("Fp64")};
  }
};

static_assert(std::is_empty<CudaRank2Selection>::value,
              "SC25 CUDA rank-two selection must remain compile-time only");

enum class ScalarCudaStatus : std::uint8_t
{
  uninitialized,
  ready,
  transaction_prepared,
  accepted,
  accepted_after_one_retry,
  injected_local_factor_failure,
  injected_trace_linear_failure,
  injected_admissibility_failure,
  injected_newton_iteration_limit,
  invalid_configuration,
  unsupported_communicator,
  unsupported_device,
  mfem_topology_failure,
  allocation_failure,
  transfer_failure,
  assembly_failure,
  local_factor_failure,
  trace_linear_failure,
  admissibility_failure,
  newton_iteration_limit,
  nonlinear_residual_failure,
  conservation_failure,
  publication_failure,
  retry_exhausted,
  restart_identity_failure,
  restart_hash_failure,
  restart_admissibility_failure,
  restart_assembly_failure,
  restart_auxiliary_residual_failure
};

enum class ScalarCudaFailureCause : std::uint8_t
{
  none,
  injected_local_factor,
  injected_trace_linear,
  injected_admissibility,
  injected_newton_limit,
  device_state_admissibility,
  device_nonfinite,
  device_invalid_layout,
  device_local_singularity,
  device_local_application,
  trace_nonconvergence,
  nonlinear_nonconvergence,
  conservation,
  transfer,
  publication
};

using FailureInjection = cpu_sc25::TrajectoryFailureInjection;
using PublishedState = cpu_sc25::PersistentRank2TrajectoryState;
using StateHashes = cpu_sc25::Rank2TrajectoryStateHashes;

/** Closed leaf selection; this is not a public runtime dispatch surface. */
enum class ScalarCudaLifecycle : std::uint8_t
{
  sc25_periodic_v3,
  w8c_smooth_transition_v2
};

struct ScalarCudaConfig final
{
  int polynomial_degree = 1;
  int elements_per_direction = 8;
  int requested_device = 0;
  bool require_host_staged_rank2 = true;
  ScalarCudaLifecycle lifecycle = ScalarCudaLifecycle::sc25_periodic_v3;
};

/** Rank-local checkpoint shard; the canonical manifest binds both ranks. */
struct ScalarCudaCheckpoint final
{
  CheckpointDescriptor descriptor{};
  StableIdentity projection = StableIdentity::Copy(projection_id);
  StableIdentity tuple = StableIdentity::Copy(cuda_rank2_tuple_id);
  StableIdentity scenario = StableIdentity::Copy(cuda_rank2_fixture_id);
  StableIdentity layout = StableIdentity::Copy(checkpoint_layout_id);
  int shard_rank = -1;
  int owned_scalar_trace_nodes = 0;
  std::vector<double> interior_component_slabs;
  std::vector<double> owned_trace_component_slabs;
  StateHashes raw_field_hashes{};
  StateHash shard_manifest_hash{};
  StateHash canonical_manifest_hash{};
};

struct ScalarCudaLaunchLedger final
{
  std::uint64_t assemble_launches = 0;
  std::uint64_t condense_launches = 0;
  std::uint64_t recover_launches = 0;
  std::uint64_t remote_system_exchange_launches = 0;
  std::uint64_t owned_trace_distribution_launches = 0;
  std::uint64_t fallback_calls = 0;
  int last_compiled_architecture = 0;
  bool all_launches_completed = true;
};

struct ScalarCudaResidencyEvidence final
{
  bool mfem_cuda_device = false;
  bool hypre_cuda = false;
  bool umpire_device_pool = false;
  bool state_device_resident = false;
  bool topology_device_resident = false;
  bool workspace_device_resident = false;
  bool trace_solve_device_resident = false;
  bool host_numerical_fallback = false;
  bool host_staged_rank2 = false;
  bool same_device_uuid = false;
};

struct ScalarCudaAllocationEvidence final
{
  std::uintptr_t double_arena = 0;
  std::uintptr_t integer_arena = 0;
  std::uintptr_t true_matrix = 0;
  std::uintptr_t gmres = 0;
  std::uintptr_t ilu = 0;
  std::uint64_t baseline_observed_events = 0;
  std::uint64_t current_observed_events = 0;
  std::uint64_t baseline_owned_calls = 0;
  std::uint64_t current_owned_calls = 0;
  std::uint64_t baseline_deallocations = 0;
  std::uint64_t current_deallocations = 0;
  std::size_t baseline_mfem_high_water = 0;
  std::size_t current_mfem_high_water = 0;
  std::size_t baseline_hypre_high_water = 0;
  std::size_t current_hypre_high_water = 0;
  std::array<std::size_t, warm_plateau_cycles + 1> mfem_high_water{};
  std::array<std::size_t, warm_plateau_cycles + 1> hypre_high_water{};
  bool warm_plateau = false;
  bool zero_post_baseline_observed_growth = false;
  bool zero_post_baseline_owned_allocation = false;
  bool zero_post_baseline_owned_deallocation = false;
};

struct ScalarCudaReuseEvidence final
{
  std::uintptr_t owner = 0;
  std::uintptr_t topology = 0;
  std::uintptr_t workspace = 0;
  std::array<std::uintptr_t, 2> interior_slots{};
  std::array<std::uintptr_t, 2> trace_slots{};
  std::uint64_t topology_generation = 0;
  std::uint64_t storage_generation = 0;
  std::uint64_t factor_generation = 0;
  std::uint64_t solver_generation = 0;
  StateHash identity_hash{};
  StateHash host_topology_hash{};
  StateHash device_topology_hash{};
  StateHash host_quadrature_hash{};
  StateHash device_quadrature_hash{};
  bool device_topology_matches = false;
  bool unchanged = false;
};

struct ScalarCudaNewtonEvidence final
{
  int iteration = -1;
  double entry_residual_norm = 0.0;
  double accepted_residual_norm = 0.0;
  int accepted_line_search_index = -1;
  int trace_iterations = -1;
  double trace_true_residual_norm = 0.0;
  double trace_right_hand_side_norm = 0.0;
  double maximum_local_application_residual = 0.0;
  std::array<double, line_search_trials> trial_residual_norms{};
  std::array<std::uint8_t, line_search_trials> trial_admissible{};
  std::array<std::uint8_t, line_search_trials> trial_armijo{};
};

struct ScalarCudaAttemptEvidence final
{
  ScalarCudaStatus status = ScalarCudaStatus::uninitialized;
  ScalarCudaFailureCause failure_cause = ScalarCudaFailureCause::none;
  FailureInjection injection = FailureInjection::none;
  int target_step = 0;
  int retry_index = 0;
  int iteration = -1;
  int attempted_newton_iterations = 0;
  int newton_iterations = 0;
  double dt = 0.0;
  double entry_residual_norm = 0.0;
  double final_residual_norm = 0.0;
  double nonlinear_residual_bound = 0.0;
  double maximum_conservation_error = 0.0;
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
  double minimum_speed_margin = 0.0;
  StateCoordinates committed_coordinates{};
  StateCoordinates candidate_coordinates{};
  StateHashes entry_hashes{};
  StateHashes candidate_hashes{};
  std::array<ScalarCudaNewtonEvidence, newton_maximum_iterations> newton{};
};

struct ScalarCudaStepEvidence final
{
  bool accepted = false;
  int step = 0;
  double time = 0.0;
  std::uint64_t generation = 0;
  int attempt_index = -1;
  std::array<double, state_components> committed_integrals{};
  std::array<double, state_components> accepted_integrals{};
  StateHashes accepted_hashes{};
};

/**
 * Preallocated rank-local operator/state evidence.  Index zero is the
 * initialization operator; indices one through twenty are the last completed
 * Newton operator of each accepted stage.  Every array has rank-zero maximum
 * capacity so capturing or transactionally copying evidence never allocates.
 */
struct ScalarCudaSnapshotEvidence final
{
  bool valid = false;
  bool initialization = false;
  ScalarCudaStatus status = ScalarCudaStatus::uninitialized;
  StateCoordinates coordinates{};
  int target_step = 0;
  int retry_index = 0;
  int iteration = -1;
  int attempted_newton_iterations = 0;
  int newton_iterations = 0;
  int full_vector_size = 0;
  int owned_trace_size = 0;
  int trace_iterations = -1;
  double entry_residual_norm = 0.0;
  double trace_true_residual_norm = 0.0;
  double trace_right_hand_side_norm = 0.0;
  double maximum_local_application_residual = 0.0;
  double paired_local_application_residual = 0.0;
  double paired_local_right_hand_side_norm = 0.0;
  std::array<double, persistent_max_full_unknowns> nonlinear_residual{};
  std::array<double, persistent_max_full_unknowns> jacobian_action_input{};
  std::array<double, persistent_max_full_unknowns> jacobian_action_result{};
  std::array<double, persistent_max_owned_trace_unknowns>
      condensed_right_hand_side{};
  std::array<double, persistent_max_owned_trace_unknowns>
      true_trace_solution{};
  std::array<double, persistent_interior_unknowns> entry_interior{};
  std::array<double, persistent_max_owned_trace_unknowns> entry_trace{};
  std::array<double, persistent_interior_unknowns> recovered_interior{};
  std::array<double, persistent_max_owned_trace_unknowns> recovered_trace{};
  std::array<double, state_components> committed_integrals{};
  std::array<double, state_components> conserved_integrals{};
};

struct ScalarCudaEvidence final
{
  ScalarCudaStatus status = ScalarCudaStatus::uninitialized;
  ScalarCudaStatus first_failure = ScalarCudaStatus::uninitialized;
  int accepted_steps = 0;
  int attempted_steps = 0;
  int retry_attempts = 0;
  int attempt_count = 0;
  int step_evidence_count = 0;
  int trace_restart = 0;
  int trace_maximum_iterations = 0;
  double trace_relative_tolerance = 0.0;
  double trace_absolute_tolerance = 0.0;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  bool row_partial_pivot_contract = false;
  bool nonzero_small_pivot_contract = false;
  cpu_sc25::Rank2OwnershipCensus census{};
  StateCoordinates coordinates{};
  ScalarCudaResidencyEvidence residency{};
  ScalarCudaAllocationEvidence allocations{};
  ScalarCudaReuseEvidence reuse{};
  ScalarCudaLaunchLedger launches{};
  cuda_w8a::CudaTransferLedgerV1 transfers{};
  cuda_w8a::CudaAllocationLedgerV1 allocation_ledger{};
  std::array<ScalarCudaAttemptEvidence, maximum_required_steps + 4>
      attempts{};
  std::array<ScalarCudaStepEvidence, maximum_required_steps> steps{};
  int snapshot_count = 0;
  std::array<ScalarCudaSnapshotEvidence, maximum_required_steps + 1>
      snapshots{};
};

class PersistentSc25ScalarIgrCudaRank2 final
{
private:
  struct Implementation;

public:
  class Transaction final
  {
  public:
    Transaction() noexcept = default;
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&& other) noexcept
        : incarnation_(other.incarnation_),
          serial_(other.serial_), candidate_slot_(other.candidate_slot_),
          target_(other.target_), valid_(other.valid_),
          installed_(other.installed_)
    {
      other.incarnation_ = 0;
      other.serial_ = 0;
      other.candidate_slot_ = -1;
      other.target_ = {};
      other.valid_ = false;
      other.installed_ = false;
    }
    Transaction& operator=(Transaction&&) = delete;
    ~Transaction() noexcept = default;

    bool valid() const noexcept { return valid_; }
    bool provisionally_installed() const noexcept { return installed_; }
    const StateCoordinates& target_coordinates() const noexcept
    { return target_; }

  private:
    std::uint64_t incarnation_ = 0;
    std::uint64_t serial_ = 0;
    int candidate_slot_ = -1;
    StateCoordinates target_{};
    bool valid_ = false;
    bool installed_ = false;
    friend class PersistentSc25ScalarIgrCudaRank2;
  };

  ~PersistentSc25ScalarIgrCudaRank2() noexcept;
  PersistentSc25ScalarIgrCudaRank2(
      PersistentSc25ScalarIgrCudaRank2&&) noexcept;
  PersistentSc25ScalarIgrCudaRank2& operator=(
      PersistentSc25ScalarIgrCudaRank2&&) = delete;
  PersistentSc25ScalarIgrCudaRank2(
      const PersistentSc25ScalarIgrCudaRank2&) = delete;
  PersistentSc25ScalarIgrCudaRank2& operator=(
      const PersistentSc25ScalarIgrCudaRank2&) = delete;

  static ScalarCudaStatus Create(
      MPI_Comm communicator, const ScalarCudaConfig& config,
      std::unique_ptr<PersistentSc25ScalarIgrCudaRank2>& output) noexcept;

  ScalarCudaStatus status() const noexcept;
  const ScalarCudaConfig& config() const noexcept;

  ScalarCudaStatus InitializePersistentTrajectory() noexcept;
  ScalarCudaStatus AdvancePersistentTrajectory(
      FailureInjection injection = FailureInjection::none) noexcept;
  ScalarCudaStatus PreparePersistentTrajectoryTransaction(
      Transaction& output,
      FailureInjection injection = FailureInjection::none) noexcept;
  bool InstallPersistentTrajectoryTransaction(Transaction& token) noexcept;
  bool StageInstalledTrajectoryTransaction(
      Transaction& token, ScalarCudaCheckpoint& checkpoint,
      ScalarCudaEvidence& prospective_evidence) noexcept;
  bool RevertPersistentTrajectoryTransaction(Transaction&& token) noexcept;
  bool FinalizePersistentTrajectoryTransaction(Transaction&& token) noexcept;

  bool CopyPublishedTrajectoryState(PublishedState& output) const noexcept;
  bool CaptureTrajectoryCheckpoint(ScalarCudaCheckpoint& output) const
      noexcept;
  ScalarCudaStatus RestoreTrajectoryCheckpoint(
      const ScalarCudaCheckpoint& checkpoint) noexcept;

  const ScalarCudaEvidence& trajectory_evidence() const noexcept;
  ScalarCudaReuseEvidence reuse_evidence() const noexcept;
  cuda_w8a::CudaTransferLedgerV1 transfer_ledger() const noexcept;
  cuda_w8a::CudaAllocationLedgerV1 allocation_ledger() const noexcept;

private:
  ScalarCudaStatus PreparePersistentTrajectoryTransactionImpl(
      Transaction& output, FailureInjection injection) noexcept;
  explicit PersistentSc25ScalarIgrCudaRank2(
      std::unique_ptr<Implementation> implementation) noexcept;
  std::unique_ptr<Implementation> implementation_;
};

static_assert(!std::is_polymorphic<PersistentSc25ScalarIgrCudaRank2>::value,
              "SC25 CUDA leaf must not add runtime virtual dispatch");
static_assert(
    std::is_nothrow_move_constructible<
        PersistentSc25ScalarIgrCudaRank2>::value &&
        !std::is_move_assignable<PersistentSc25ScalarIgrCudaRank2>::value &&
        !std::is_copy_constructible<PersistentSc25ScalarIgrCudaRank2>::value &&
        !std::is_copy_assignable<PersistentSc25ScalarIgrCudaRank2>::value,
    "SC25 CUDA owner must remain a nothrow-movable affine handle");
static_assert(!std::is_copy_constructible<
                  PersistentSc25ScalarIgrCudaRank2::Transaction>::value &&
                  !std::is_polymorphic<
                      PersistentSc25ScalarIgrCudaRank2::Transaction>::value &&
                  std::is_nothrow_move_constructible<
                      PersistentSc25ScalarIgrCudaRank2::Transaction>::value &&
                  std::is_move_constructible<
                      PersistentSc25ScalarIgrCudaRank2::Transaction>::value &&
                  !std::is_move_assignable<
                      PersistentSc25ScalarIgrCudaRank2::Transaction>::value,
              "SC25 CUDA transaction authority must remain affine");

} // namespace dg::hdg::runtime::cuda_w8b::sc25
