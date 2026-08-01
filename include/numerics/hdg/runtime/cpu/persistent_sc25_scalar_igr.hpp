#pragma once

/**
 * @file numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp
 * @brief MFEM-native CPU rank-one/rank-two leaves for SC25 projection-v3.
 *
 * This default-disabled component owns the C0 field/space/topology foundation,
 * the exact C1 coupled-operator fixture, and the bounded C2 CPU-rank-one
 * persistent trajectory experiment and the bounded CPU-rank-two peer.  These
 * leaves are experimental evidence for only the frozen N=8/k=1/20-step
 * tuples; they are not a CLI, output, CUDA, supported-method, or broader
 * E2/E3 claim.
 */

#include "numerics/hdg/auxiliary/screened_poisson_cpu.hpp"
#include "numerics/hdg/runtime/run_contract.hpp"
#include "numerics/hdg/trace/trace_dg_mpi.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dg::hdg::runtime::cpu::sc25
{

inline constexpr std::string_view projection_id =
    "exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3";
inline constexpr std::string_view object_id =
    "FullyHybridizedScalarPeriodicSc25ProjectionV3";
inline constexpr std::string_view cpu_rank1_tuple_id =
    "HDG-SC25-PER-BE-CPU-R1-FP64-v3";
inline constexpr std::string_view cpu_rank2_tuple_id =
    "HDG-SC25-PER-BE-CPU-R2-FP64-v3";
inline constexpr std::string_view interior_layout_id =
    "mfem.l2.byNODES.U4.Sigma1.z2.separate.v3";
inline constexpr std::string_view trace_layout_id =
    "mfem.dg-interface.byNODES.Uhat4.Sigmahat1.carrier5.v3";
inline constexpr std::string_view coupled_fixture_id =
    "C1-MFEM-PERIODIC-02";
inline constexpr std::string_view coupled_condensation_id =
    "HDG-SC25-CONDENSE-02";
inline constexpr std::string_view coupled_jacobian_id =
    "HDG-SC25-JAC-02";

inline constexpr int dimension = 2;
inline constexpr int state_components = 4;
inline constexpr int flux_components = 2;
inline constexpr int local_components = 7;
inline constexpr int trace_components = 5;
inline constexpr int faces_per_quadrilateral = 4;
inline constexpr double gamma = 1.4;
inline constexpr double primary_alpha = 0.1;
inline constexpr double state_stabilization = 2.5;
inline constexpr double coupled_fixture_dt = 1.0 / 4096.0;
inline constexpr int coupled_fixture_elements_per_direction = 3;
inline constexpr int coupled_fixture_polynomial_degree = 1;
inline constexpr int coupled_fixture_quadrature_order =
    4 * coupled_fixture_polynomial_degree + 8;
inline constexpr int coupled_fixture_repeat_quadrature_order =
    4 * coupled_fixture_polynomial_degree + 10;
inline constexpr int coupled_fixture_scalar_volume_nodes =
    (coupled_fixture_polynomial_degree + 1) *
    (coupled_fixture_polynomial_degree + 1);
inline constexpr int coupled_fixture_scalar_face_nodes =
    coupled_fixture_polynomial_degree + 1;
inline constexpr int coupled_fixture_elements =
    coupled_fixture_elements_per_direction *
    coupled_fixture_elements_per_direction;
inline constexpr int coupled_fixture_unique_faces =
    dimension * coupled_fixture_elements;
inline constexpr int coupled_fixture_local_scalar_trace_nodes =
    faces_per_quadrilateral * coupled_fixture_scalar_face_nodes;
inline constexpr int coupled_fixture_local_interior_unknowns =
    local_components * coupled_fixture_scalar_volume_nodes;
inline constexpr int coupled_fixture_local_trace_unknowns =
    trace_components * coupled_fixture_local_scalar_trace_nodes;
inline constexpr int coupled_fixture_global_interior_unknowns =
    coupled_fixture_elements * coupled_fixture_local_interior_unknowns;
inline constexpr int coupled_fixture_true_trace_unknowns =
    trace_components * coupled_fixture_unique_faces *
    coupled_fixture_scalar_face_nodes;
inline constexpr int coupled_fixture_restriction_rows =
    coupled_fixture_elements * coupled_fixture_local_trace_unknowns;
inline constexpr int coupled_fixture_full_unknowns =
    coupled_fixture_global_interior_unknowns +
    coupled_fixture_true_trace_unknowns;
static_assert(coupled_fixture_elements == 9 &&
                  coupled_fixture_unique_faces == 18 &&
                  coupled_fixture_quadrature_order == 12 &&
                  coupled_fixture_repeat_quadrature_order == 14 &&
                  coupled_fixture_local_interior_unknowns == 28 &&
                  coupled_fixture_local_trace_unknowns == 40 &&
                  coupled_fixture_global_interior_unknowns == 252 &&
                  coupled_fixture_true_trace_unknowns == 180 &&
                  coupled_fixture_restriction_rows == 360 &&
                  coupled_fixture_full_unknowns == 432,
              "C1-MFEM-PERIODIC-02 dimensions must remain exact");

inline constexpr std::string_view persistent_fixture_id =
    "C2-MFEM-PERSISTENT-N8-R1-01";
inline constexpr std::string_view persistent_rank2_fixture_id =
    "C3-MFEM-PERSISTENT-N8-R2-01";
inline constexpr std::string_view persistent_checkpoint_manifest_id =
    "HDG-SC25-CHECKPOINT-MANIFEST-v3";
inline constexpr int persistent_elements_per_direction = 8;
inline constexpr int persistent_elements = 64;
inline constexpr int persistent_unique_faces = 128;
inline constexpr int persistent_scalar_volume_nodes = 4;
inline constexpr int persistent_scalar_true_trace_nodes = 256;
inline constexpr int persistent_u_unknowns = 1024;
inline constexpr int persistent_sigma_unknowns = 256;
inline constexpr int persistent_z_unknowns = 512;
inline constexpr int persistent_uhat_unknowns = 1024;
inline constexpr int persistent_sigmahat_unknowns = 256;
inline constexpr int persistent_interior_unknowns = 1792;
inline constexpr int persistent_true_trace_unknowns = 1280;
inline constexpr int persistent_restriction_rows = 2560;
inline constexpr int persistent_full_unknowns = 3072;
inline constexpr int persistent_rank2_local_elements = 32;
inline constexpr int persistent_rank2_local_interior_unknowns = 896;
inline constexpr int persistent_rank2_local_faces = 72;
inline constexpr int persistent_rank2_rank0_scalar_true_dofs = 144;
inline constexpr int persistent_rank2_rank1_scalar_true_dofs = 112;
inline constexpr int persistent_rank2_rank0_owned_trace_unknowns = 720;
inline constexpr int persistent_rank2_rank1_owned_trace_unknowns = 560;
inline constexpr int persistent_rank2_global_true_trace_unknowns = 1280;
inline constexpr int persistent_rank2_rank0_ghost_trace_unknowns = 0;
inline constexpr int persistent_rank2_rank1_ghost_trace_unknowns = 160;
inline constexpr int persistent_required_steps = 20;
inline constexpr int persistent_newton_maximum_iterations = 20;
inline constexpr int persistent_line_search_trials = 13;
inline constexpr int persistent_trace_restart_dimension = 50;
inline constexpr int persistent_trace_maximum_iterations = 200;
inline constexpr double persistent_dt = 1.0 / 4096.0;
static_assert(persistent_elements ==
                      persistent_elements_per_direction *
                          persistent_elements_per_direction &&
                  persistent_u_unknowns ==
                      persistent_elements * state_components *
                          persistent_scalar_volume_nodes &&
                  persistent_sigma_unknowns ==
                      persistent_elements * persistent_scalar_volume_nodes &&
                  persistent_z_unknowns ==
                      persistent_elements * flux_components *
                          persistent_scalar_volume_nodes &&
                  persistent_uhat_unknowns ==
                      state_components * persistent_scalar_true_trace_nodes &&
                  persistent_sigmahat_unknowns ==
                      persistent_scalar_true_trace_nodes &&
                  persistent_interior_unknowns ==
                      persistent_u_unknowns + persistent_sigma_unknowns +
                          persistent_z_unknowns &&
                  persistent_true_trace_unknowns ==
                      persistent_uhat_unknowns +
                          persistent_sigmahat_unknowns &&
                  persistent_full_unknowns ==
                      persistent_interior_unknowns +
                          persistent_true_trace_unknowns &&
                  persistent_rank2_local_interior_unknowns ==
                      persistent_rank2_local_elements *
                          coupled_fixture_local_interior_unknowns &&
                  persistent_rank2_rank0_owned_trace_unknowns +
                          persistent_rank2_rank1_owned_trace_unknowns ==
                      persistent_rank2_global_true_trace_unknowns,
              "C2-MFEM-PERSISTENT-N8-R1-01 dimensions must remain exact");

using HdgPublishedScalarPeriodicSc25ProjectionV3Tag =
    auxiliary::HdgPublishedScalarPeriodicSc25ProjectionV3Tag;
using DensityHeightP2TraceStabilizationV2Tag =
    auxiliary::DensityHeightP2TraceStabilizationV2Tag;

struct TracedScalarSc25PeriodicExperimentV1Tag final
{};
struct DualConservativeScalarTraceV1Tag final
{};
struct W7ScalarTau25StateStabilizationTag final
{};
struct BackwardEulerMonolithicV1Tag final
{};
struct Fp64Tag final
{};
struct CpuSerialTag final
{};
struct CpuMpiRank2Tag final
{};

struct CpuRank1Selection final
{
  using outer_model = HdgPublishedScalarPeriodicSc25ProjectionV3Tag;
  using source = TracedScalarSc25PeriodicExperimentV1Tag;
  using auxiliary_model = auxiliary::FullyHybridizedScreenedPoissonV1Tag;
  using trace = DualConservativeScalarTraceV1Tag;
  using state_stabilization = W7ScalarTau25StateStabilizationTag;
  using auxiliary_stabilization = DensityHeightP2TraceStabilizationV2Tag;
  using time_treatment = BackwardEulerMonolithicV1Tag;
  using precision = Fp64Tag;
  using backend = CpuSerialTag;

  static constexpr RuntimeTupleDescriptor Descriptor() noexcept
  {
    return {StableIdentity::Copy(projection_id),
            StableIdentity::Copy(object_id),
            StableIdentity::Copy("DualConservativeScalarTraceV1"),
            StableIdentity::Copy("W7Tau25.DensityHeightP2.v3"),
            StableIdentity::Copy("BackwardEulerMonolithicV1"),
            StableIdentity::Copy("CpuSerial"),
            StableIdentity::Copy("Fp64")};
  }
};

static_assert(std::is_empty<CpuRank1Selection>::value,
              "SC25 rank-one selection must remain compile-time only");

struct CpuRank2Selection final
{
  using outer_model = HdgPublishedScalarPeriodicSc25ProjectionV3Tag;
  using source = TracedScalarSc25PeriodicExperimentV1Tag;
  using auxiliary_model = auxiliary::FullyHybridizedScreenedPoissonV1Tag;
  using trace = DualConservativeScalarTraceV1Tag;
  using state_stabilization = W7ScalarTau25StateStabilizationTag;
  using auxiliary_stabilization = DensityHeightP2TraceStabilizationV2Tag;
  using time_treatment = BackwardEulerMonolithicV1Tag;
  using precision = Fp64Tag;
  using backend = CpuMpiRank2Tag;

  static constexpr RuntimeTupleDescriptor Descriptor() noexcept
  {
    return {StableIdentity::Copy(projection_id),
            StableIdentity::Copy(object_id),
            StableIdentity::Copy("DualConservativeScalarTraceV1"),
            StableIdentity::Copy("W7Tau25.DensityHeightP2.v3"),
            StableIdentity::Copy("BackwardEulerMonolithicV1"),
            StableIdentity::Copy("CpuMpiRank2"),
            StableIdentity::Copy("Fp64")};
  }
};

static_assert(std::is_empty<CpuRank2Selection>::value,
              "SC25 rank-two selection must remain compile-time only");

enum class FoundationStatus : std::uint8_t
{
  uninitialized,
  ready,
  invalid_argument,
  unsupported_communicator,
  allocation_failure,
  mesh_failure,
  space_failure,
  trace_map_failure,
  topology_failure,
  local_factor_failure,
  local_application_failure,
  true_assembly_failure
};

struct FoundationConfig final
{
  int polynomial_degree = 1;
  int elements_per_direction = 8;
};

struct SpaceIdentity final
{
  const mfem::ParFiniteElementSpace* address = nullptr;
  const mfem::FiniteElementCollection* collection = nullptr;
  const mfem::ParMesh* mesh = nullptr;
  int vector_dimension = 0;
  mfem::Ordering::Type ordering = mfem::Ordering::byNODES;
  int local_vector_size = 0;
  int local_true_size = 0;
  std::int64_t global_true_size = 0;
  std::int64_t true_offset_begin = 0;
  std::int64_t true_offset_end = 0;
};

struct SpaceOwnershipSnapshot final
{
  SpaceIdentity u;
  SpaceIdentity sigma;
  SpaceIdentity z;
  SpaceIdentity uhat;
  SpaceIdentity sigmahat;
  SpaceIdentity trace_carrier;
};

struct FieldOwnershipView final
{
  const mfem::ParGridFunction* u = nullptr;
  const mfem::ParGridFunction* sigma = nullptr;
  const mfem::ParGridFunction* z = nullptr;
  const mfem::ParGridFunction* uhat = nullptr;
  const mfem::ParGridFunction* sigmahat = nullptr;
  /** Non-state algebra carrier. Physical trace ownership remains above. */
  const mfem::ParGridFunction* trace_carrier = nullptr;
};

struct ElementVolumeDofs final
{
  std::vector<int> u;
  std::vector<int> sigma;
  std::vector<int> z;
  bool u_transformation_identity = true;
  bool sigma_transformation_identity = true;
  bool z_transformation_identity = true;
};

struct TraceRestrictionEntry final
{
  int element = -1;
  int local_row = -1;
  int local_face = -1;
  int component = -1;
  int oriented_face_dof = -1;
  int canonical_face_dof = -1;
  int signed_carrier_vdof = 0;
  int decoded_carrier_vdof = -1;
  int sign = 0;
  GlobalIndex global_true_dof = invalid_global_index;
  LocalIndex local_true_dof = invalid_local_index;
  RankIndex owner_rank = invalid_rank;
};

struct ElementTopology final
{
  int element = -1;
  int scalar_volume_dofs = 0;
  int scalar_face_dofs = 0;
  int local_interior_unknowns = 0;
  int local_trace_unknowns = 0;
  ElementVolumeDofs volume;
  std::vector<int> faces;
  std::vector<int> face_orientations;
  /** Component-major element-oriented carrier VDofs, including MFEM signs. */
  std::vector<int> signed_trace_vdofs;
  std::vector<TraceRestrictionEntry> restriction;
};

struct ElementFieldCoefficients final
{
  std::vector<double> u;
  std::vector<double> sigma;
  std::vector<double> z;
};

struct InitialStatePoint final
{
  std::array<double, state_components> conservative{};
};

struct AuxiliaryStabilizationPoint final
{
  double tau = 0.0;
  double numerical_flux = 0.0;
  double r_sigma_d_sigma = 0.0;
  double r_sigma_d_sigmahat = 0.0;
  double r_sigma_d_rhohat = 0.0;
  double g_sigma_d_sigma = 0.0;
  double g_sigma_d_sigmahat = 0.0;
  double g_sigma_d_rhohat = 0.0;
};

/** Generic element blocks in the exact residual-row/unknown-column order. */
struct ElementLinearization final
{
  int element = -1;
  int local_interior_unknowns = 0;
  int local_trace_unknowns = 0;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> d;
  std::vector<double> local_residual;
  std::vector<double> trace_residual;
};

struct ElementCondensedBlock final
{
  int element = -1;
  int local_trace_unknowns = 0;
  std::vector<double> schur;
  std::vector<double> right_hand_side;
  double maximum_local_application_residual = 0.0;
};

struct TraceAssemblyIdentity final
{
  /** Incremented only when a complete true system replaces the retained one. */
  std::uint64_t successful_generation = 0;
  int local_height = 0;
  int local_width = 0;
  int local_true_size = 0;
  std::int64_t global_true_size = 0;
  std::size_t local_nonzeros = 0;
};

struct CoupledTraceSolverEvidence final
{
  int restart_dimension = 0;
  int maximum_iterations = 0;
  double relative_tolerance = 0.0;
  double absolute_tolerance = 0.0;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  int setup_status = -1;
  int solve_status = -1;
  int convergence_query_status = -1;
  int iterations = -1;
  int converged = 0;
  double unpreconditioned_residual_norm = 0.0;
  double right_hand_side_norm = 0.0;
};

/** Raw evidence for exact C1-MFEM-PERIODIC-02 (3x3, k=1, 432 unknowns). */
struct CoupledOperatorFixtureSnapshot final
{
  int quadrature_order = 0;
  int repeat_quadrature_order = 0;
  int local_interior_unknowns = 0;
  int local_trace_unknowns = 0;
  int true_trace_unknowns = 0;
  int full_unknowns = 0;
  std::vector<double> old_interior;
  std::vector<double> old_true_trace;
  std::vector<double> candidate_interior;
  std::vector<double> candidate_true_trace;
  std::vector<double> full_residual;
  std::vector<double> full_jacobian;
  std::vector<ElementLinearization> elements;
  std::vector<ElementCondensedBlock> condensed;
  std::vector<double> true_trace_increment;
  std::vector<double> element_trace_increment;
  std::vector<double> recovered_interior_increment;
  std::vector<double> full_increment;
  double maximum_local_application_residual = 0.0;
  double residual_quadrature_repeat_relative_l2 = 0.0;
  double jacobian_quadrature_repeat_relative_l2 = 0.0;
  CoupledTraceSolverEvidence solver;
};

enum class TrajectoryStatus : std::uint8_t
{
  uninitialized,
  ready,
  HdgSc25Accepted,
  HdgSc25AcceptedAfterOneRetry,
  HdgSc25InjectedLocalFactorFailure,
  HdgSc25InjectedTraceLinearFailure,
  HdgSc25InjectedAdmissibilityFailure,
  HdgSc25InjectedNewtonIterationLimit,
  HdgSc25AssemblyFailure,
  HdgSc25LocalFactorFailure,
  HdgSc25TraceLinearFailure,
  HdgSc25AdmissibilityFailure,
  HdgSc25NewtonIterationLimit,
  HdgSc25NonlinearResidualFailure,
  HdgSc25ConservationFailure,
  HdgSc25PublicationFailure,
  HdgSc25RetryExhausted,
  HdgSc25RestartValidationFailureIdentity,
  HdgSc25RestartValidationFailureHash,
  HdgSc25RestartValidationFailureAdmissibility,
  HdgSc25RestartValidationFailureAssembly,
  HdgSc25RestartValidationFailureAuxResidual,
  invalid_configuration,
  initialization_failure
};

enum class TrajectoryFailureInjection : std::uint8_t
{
  none,
  local_factor,
  trace_linear,
  admissibility,
  newton_iteration_limit
};

enum class TrajectoryAssemblyFailureCause : std::uint8_t
{
  none,
  state_admissibility,
  invalid_structure,
  nonfinite_result,
  exception
};

struct PersistentTrajectoryState final
{
  std::array<double, persistent_u_unknowns> u{};
  std::array<double, persistent_sigma_unknowns> sigma{};
  std::array<double, persistent_z_unknowns> z{};
  std::array<double, persistent_uhat_unknowns> uhat{};
  std::array<double, persistent_sigmahat_unknowns> sigmahat{};
};

struct TrajectoryStateHashes final
{
  StateHash u{};
  StateHash sigma{};
  StateHash z{};
  StateHash uhat{};
  StateHash sigmahat{};
  StateHash interior{};
  StateHash trace{};
};

struct TrajectoryCheckpointFieldIdentity final
{
  StableIdentity name{};
  std::uint64_t fp64_entries = 0;
  std::uint32_t canonical_order = 0;
};

struct PersistentTrajectoryCheckpoint final
{
  CheckpointDescriptor descriptor{};
  PersistentTrajectoryState state{};
  StableIdentity projection = StableIdentity::Copy(projection_id);
  StableIdentity tuple = StableIdentity::Copy(cpu_rank1_tuple_id);
  StableIdentity scenario = StableIdentity::Copy(persistent_fixture_id);
  TrajectoryStateHashes raw_field_hashes{};
  std::array<TrajectoryCheckpointFieldIdentity, 5> fields{};
  StateHash canonical_manifest_hash{};
};

struct AuxiliaryResidualMetrics final
{
  double r_z_norm = 0.0;
  double r_z_scale = 0.0;
  double r_sigma_norm = 0.0;
  double r_sigma_scale = 0.0;
  double g_sigma_norm = 0.0;
  double g_sigma_scale = 0.0;
  double normalized_maximum = 0.0;
};

struct AuxiliaryInitializationEvidence final
{
  int quadrature_order = 0;
  int trace_restart_dimension = 0;
  int trace_maximum_iterations = 0;
  double trace_relative_tolerance = 0.0;
  double trace_absolute_tolerance = 0.0;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  int trace_iterations = -1;
  double trace_true_residual_norm = 0.0;
  double trace_right_hand_side_norm = 0.0;
  AuxiliaryResidualMetrics residual{};
};

struct TrajectoryReuseSnapshot final
{
  /**
   * Direct W8-owned boundary census only. Equality detects wrapper/buffer
   * identity, extent, capacity, or generation drift. It does not claim zero
   * total allocation calls, an allocation plateau, performance, or visibility
   * into opaque MFEM/HYPRE internals.
   */
  std::uintptr_t foundation_handle = 0;
  std::uintptr_t topology_handle = 0;
  std::uintptr_t carrier_map_handle = 0;
  std::uintptr_t contract_handle = 0;
  std::uintptr_t assembly_state_handle = 0;
  std::uintptr_t assembly_handle = 0;
  std::uintptr_t trial_assembly_handle = 0;
  std::uintptr_t factor_handle = 0;
  std::uintptr_t workspace_handle = 0;
  std::uintptr_t quadrature_workspace_handle = 0;
  std::uintptr_t local_pattern_handle = 0;
  std::uintptr_t true_matrix_handle = 0;
  std::uintptr_t true_rhs_handle = 0;
  std::uintptr_t true_solution_handle = 0;
  std::uintptr_t ilu_handle = 0;
  std::uintptr_t gmres_handle = 0;
  std::uintptr_t true_residual_handle = 0;
  std::uintptr_t element_to_true_csr_handle = 0;
  StateHash owned_boundary_identity_hash{};
  std::uint64_t topology_generation = 0;
  std::uint64_t storage_generation = 0;
  std::uint64_t factor_generation = 0;
  std::uint64_t workspace_generation = 0;
  std::uint64_t local_pattern_generation = 0;
  std::uint64_t solver_generation = 0;
  std::uint64_t w8_owned_boundary_object_count = 0;
  std::uint64_t w8_owned_boundary_buffer_count = 0;
  /** MFEM/HYPRE internals are outside this direct W8-owned boundary census. */
  bool opaque_mfem_hypre_observed = false;
};

struct TrajectoryNewtonEvidence final
{
  int iteration = -1;
  double entry_residual_norm = 0.0;
  double accepted_residual_norm = 0.0;
  int accepted_line_search_index = -1;
  int trace_iterations = -1;
  double trace_true_residual_norm = 0.0;
  double trace_right_hand_side_norm = 0.0;
  double maximum_local_application_residual = 0.0;
  PersistentTrajectoryState entry{};
  PersistentTrajectoryState direction{};
  std::array<double, persistent_line_search_trials> trial_residual_norms{};
  std::array<std::uint8_t, persistent_line_search_trials> trial_admissible{};
  std::array<std::uint8_t, persistent_line_search_trials> trial_armijo{};
};

struct TrajectoryAttemptEvidence final
{
  TrajectoryStatus status = TrajectoryStatus::uninitialized;
  TrajectoryFailureInjection injection = TrajectoryFailureInjection::none;
  int target_step = 0;
  int retry_index = 0;
  double dt = 0.0;
  int attempted_newton_iterations = 0;
  int newton_iterations = 0;
  double entry_residual_norm = 0.0;
  double final_residual_norm = 0.0;
  double nonlinear_residual_bound = 0.0;
  double maximum_conservation_error = 0.0;
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
  double minimum_speed_margin = 0.0;
  StateCoordinates committed_coordinates{};
  StateCoordinates published_coordinates{};
  TrajectoryStateHashes entry_hashes{};
  TrajectoryStateHashes candidate_hashes{};
  TrajectoryStateHashes published_hashes{};
  TrajectoryReuseSnapshot entry_reuse{};
  TrajectoryReuseSnapshot exit_reuse{};
  std::array<TrajectoryNewtonEvidence,
             persistent_newton_maximum_iterations> newton{};
};

struct TrajectoryStepEvidence final
{
  bool accepted = false;
  int step = 0;
  double time = 0.0;
  double dt = 0.0;
  std::uint64_t generation = 0;
  int attempt_index = -1;
  std::array<double, state_components> committed_integrals{};
  std::array<double, state_components> accepted_integrals{};
  PersistentTrajectoryState accepted_state{};
  TrajectoryStateHashes accepted_hashes{};
  TrajectoryReuseSnapshot reuse{};
};

struct TrajectoryReuseEvidence final
{
  TrajectoryReuseSnapshot baseline{};
  TrajectoryReuseSnapshot current{};
  std::size_t true_matrix_nonzeros = 0;
  /** Detected pointer/extent/capacity/generation drift, not allocation calls. */
  std::uint64_t detected_w8_owned_boundary_identity_changes = 0;
  std::uint64_t w8_device_transfer_calls = 0;
  bool opaque_mfem_hypre_observed = false;
};

struct PersistentTrajectoryEvidence final
{
  TrajectoryStatus status = TrajectoryStatus::uninitialized;
  TrajectoryStatus first_failure = TrajectoryStatus::uninitialized;
  TrajectoryAssemblyFailureCause first_assembly_failure =
      TrajectoryAssemblyFailureCause::none;
  RunStatus contract_status = RunStatus::ready;
  FailureReason contract_last_failure = FailureReason::none;
  FailureReason contract_first_failure = FailureReason::none;
  int accepted_steps = 0;
  int attempted_steps = 0;
  int retry_attempts = 0;
  int attempt_count = 0;
  /** Number of records stored in steps, not the absolute accepted step. */
  int step_evidence_count = 0;
  std::uint64_t checkpoint_step_offset = 0;
  StateCoordinates continuation_start_coordinates{};
  PersistentTrajectoryState continuation_start_state{};
  TrajectoryStateHashes continuation_start_hashes{};
  bool restored_loaded_state_match = false;
  bool restored_loaded_coordinates_match = false;
  bool restored_loaded_hashes_match = false;
  int trace_restart_dimension = 0;
  int trace_maximum_iterations = 0;
  double trace_relative_tolerance = 0.0;
  double trace_absolute_tolerance = 0.0;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  StateCoordinates coordinates{};
  AuxiliaryInitializationEvidence initialization{};
  TrajectoryReuseEvidence reuse{};
  std::array<TrajectoryAttemptEvidence,
             persistent_required_steps + 4> attempts{};
  std::array<TrajectoryStepEvidence, persistent_required_steps> steps{};
};

/** Rank-local committed state. Owned trace values exclude all ghost values. */
struct PersistentRank2TrajectoryState final
{
  std::vector<double> u;
  std::vector<double> sigma;
  std::vector<double> z;
  std::vector<double> uhat;
  std::vector<double> sigmahat;
};

struct Rank2OwnershipCensus final
{
  int rank = -1;
  int ranks = 0;
  int local_elements = 0;
  int local_interior_unknowns = 0;
  int local_faces = 0;
  int owned_unique_faces = 0;
  int scalar_owned_true_dofs = 0;
  int combined_owned_trace_dofs = 0;
  int combined_ghost_trace_dofs = 0;
  int global_unique_faces = 0;
  int global_combined_true_dofs = 0;
};

struct Rank2TrajectoryStateHashes final
{
  StateHash u{};
  StateHash sigma{};
  StateHash z{};
  StateHash uhat{};
  StateHash sigmahat{};
  StateHash interior{};
  StateHash trace{};
  StateHash shard{};
};

struct PersistentRank2TrajectoryCheckpoint final
{
  CheckpointDescriptor descriptor{};
  PersistentRank2TrajectoryState state{};
  StableIdentity projection = StableIdentity::Copy(projection_id);
  StableIdentity tuple = StableIdentity::Copy(cpu_rank2_tuple_id);
  StableIdentity scenario = StableIdentity::Copy(persistent_rank2_fixture_id);
  int shard_rank = -1;
  Rank2TrajectoryStateHashes raw_field_hashes{};
  StateHash canonical_manifest_hash{};
};

struct Rank2TrajectoryNewtonEvidence final
{
  bool attempted = false;
  bool completed = false;
  int iteration = -1;
  double entry_residual_norm = 0.0;
  double accepted_residual_norm = 0.0;
  int accepted_line_search_index = -1;
  int trace_iterations = -1;
  double trace_true_residual_norm = 0.0;
  double trace_right_hand_side_norm = 0.0;
  double maximum_local_application_residual = 0.0;
  PersistentRank2TrajectoryState entry{};
  PersistentRank2TrajectoryState recovered_increment{};
  /** Rank-local element-interior rows followed by owned true-trace rows. */
  std::vector<double> nonlinear_residual{};
  /** A fixed geometry/global-TDof-authored action, independent of Newton. */
  std::vector<double> jacobian_action_input{};
  std::vector<double> jacobian_action_result{};
  /** Owned true-trace vectors in MFEM global-TDof owner order. */
  std::vector<double> condensed_right_hand_side{};
  std::vector<double> true_trace_solution{};
  std::array<double, persistent_line_search_trials> trial_residual_norms{};
  std::array<std::uint8_t, persistent_line_search_trials> trial_visited{};
  std::array<std::uint8_t, persistent_line_search_trials> trial_admissible{};
  std::array<std::uint8_t, persistent_line_search_trials> trial_armijo{};
};

struct Rank2TrajectoryAttemptEvidence final
{
  TrajectoryStatus status = TrajectoryStatus::uninitialized;
  TrajectoryFailureInjection injection = TrajectoryFailureInjection::none;
  int target_step = 0;
  int retry_index = 0;
  int attempted_newton_iterations = 0;
  int newton_iterations = 0;
  int gmres_solves = 0;
  int gmres_iterations = 0;
  int line_search_trials = 0;
  double dt = 0.0;
  double entry_residual_norm = 0.0;
  double final_residual_norm = 0.0;
  double maximum_local_application_residual = 0.0;
  double trace_true_residual_norm = 0.0;
  double trace_right_hand_side_norm = 0.0;
  double nonlinear_residual_bound = 0.0;
  StateCoordinates committed_coordinates{};
  StateCoordinates published_coordinates{};
  Rank2TrajectoryStateHashes entry_hashes{};
  Rank2TrajectoryStateHashes candidate_hashes{};
  Rank2TrajectoryStateHashes published_hashes{};
  PersistentRank2TrajectoryState entry_state{};
  PersistentRank2TrajectoryState accepted_direction{};
  std::array<Rank2TrajectoryNewtonEvidence,
             persistent_newton_maximum_iterations> newton{};
  /** Compatibility summary of the last completed Newton trial sequence. */
  std::array<double, persistent_line_search_trials> trial_residual_norms{};
  std::array<std::uint8_t, persistent_line_search_trials> trial_admissible{};
  std::array<std::uint8_t, persistent_line_search_trials> trial_armijo{};
};

struct Rank2TrajectoryStageEvidence final
{
  bool accepted = false;
  int step = 0;
  double time = 0.0;
  std::uint64_t generation = 0;
  double residual_norm = 0.0;
  double condensed_rhs_norm = 0.0;
  double trace_solution_norm = 0.0;
  double recovery_norm = 0.0;
  std::array<double, state_components> committed_integrals{};
  std::array<double, state_components> conserved_integrals{};
  PersistentRank2TrajectoryState accepted_state{};
  Rank2TrajectoryStateHashes accepted_hashes{};
};

struct PersistentRank2TrajectoryEvidence final
{
  TrajectoryStatus status = TrajectoryStatus::uninitialized;
  TrajectoryStatus first_failure = TrajectoryStatus::uninitialized;
  Rank2OwnershipCensus census{};
  StateCoordinates coordinates{};
  AuxiliaryInitializationEvidence initialization{};
  Rank2TrajectoryNewtonEvidence initialization_operator{};
  int accepted_steps = 0;
  int attempted_steps = 0;
  int retry_attempts = 0;
  int attempt_count = 0;
  int step_evidence_count = 0;
  int trace_restart_dimension = 0;
  int trace_maximum_iterations = 0;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  std::uint64_t topology_generation = 0;
  std::uint64_t storage_generation = 0;
  std::uint64_t solver_generation = 0;
  std::uint64_t detected_w8_owned_boundary_identity_changes = 0;
  std::uint64_t w8_device_transfer_calls = 0;
  StateHash w8_owned_boundary_identity_hash{};
  std::uint64_t w8_owned_boundary_object_count = 0;
  std::uint64_t w8_owned_boundary_buffer_count = 0;
  bool opaque_mfem_hypre_observed = false;
  std::array<Rank2TrajectoryAttemptEvidence,
             persistent_required_steps + 4> attempts{};
  std::array<Rank2TrajectoryStageEvidence,
             persistent_required_steps> stages{};
};

double VelocityGradientInvariant(
    const std::array<std::array<double, dimension>, dimension>& gradient)
    noexcept;

InitialStatePoint SmoothInitialState(double x, double y) noexcept;

bool EulerSigmaNormalFlux(
    const std::array<double, state_components>& conservative,
    double sigma, const std::array<double, dimension>& normal,
    std::array<double, state_components>& flux,
    std::array<std::array<double, trace_components>, state_components>&
        jacobian) noexcept;

bool SourceCorrectionNormalFlux(
    const std::array<double, state_components>& conservative,
    double sigma, const std::array<double, dimension>& normal,
    std::array<double, state_components>& correction,
    std::array<std::array<double, trace_components>, state_components>&
        jacobian) noexcept;

bool EvaluateAuxiliaryStabilization(
    int polynomial_degree, double rhohat, double height, double sigma,
    double sigmahat, double z_dot_normal, double alpha,
    double test_value, AuxiliaryStabilizationPoint& output) noexcept;

bool CondenseElementLinearization(const ElementLinearization& input,
                                  ElementCondensedBlock& output) noexcept;

struct PersistentTrajectoryImplementation;
struct PersistentRank2TrajectoryImplementation;

class PersistentSc25ScalarIgrCpuFoundation final
{
public:
  ~PersistentSc25ScalarIgrCpuFoundation() noexcept;
  PersistentSc25ScalarIgrCpuFoundation(
      PersistentSc25ScalarIgrCpuFoundation&&) noexcept;
  PersistentSc25ScalarIgrCpuFoundation& operator=(
      PersistentSc25ScalarIgrCpuFoundation&&) = delete;
  PersistentSc25ScalarIgrCpuFoundation(
      const PersistentSc25ScalarIgrCpuFoundation&) = delete;
  PersistentSc25ScalarIgrCpuFoundation& operator=(
      const PersistentSc25ScalarIgrCpuFoundation&) = delete;

  static FoundationStatus Create(
      MPI_Comm communicator, CpuSerialTag, const FoundationConfig& config,
      std::unique_ptr<PersistentSc25ScalarIgrCpuFoundation>& output) noexcept;
  static FoundationStatus Create(
      MPI_Comm communicator, CpuMpiRank2Tag, const FoundationConfig& config,
      std::unique_ptr<PersistentSc25ScalarIgrCpuFoundation>& output) noexcept;

  FoundationStatus status() const noexcept;
  const FoundationConfig& config() const noexcept;
  const mfem::ParMesh* mesh() const noexcept;
  const mfem::L2_FECollection* volume_collection() const noexcept;
  const mfem::DG_Interface_FECollection* trace_collection() const noexcept;
  SpaceOwnershipSnapshot spaces() const noexcept;
  FieldOwnershipView fields() const noexcept;
  const trace::DGInterfaceTraceMap& carrier_trace_map() const noexcept;
  const std::vector<ElementTopology>& element_topology() const noexcept;

  bool GatherElementFields(int element,
                           ElementFieldCoefficients& output) const noexcept;
  bool AddElementDualResidual(
      int element, const ElementFieldCoefficients& local_dual,
      mfem::ParGridFunction& u_residual,
      mfem::ParGridFunction& sigma_residual,
      mfem::ParGridFunction& z_residual) const noexcept;
  bool PackTraceCarrierFromPhysicalFields() noexcept;
  bool UnpackPhysicalTraceFromCarrier() noexcept;
  bool SetTraceCarrierFromTrueDofs(const mfem::Vector& true_values) noexcept;

  /**
   * Assemble supplied element Schur/RHS blocks through MFEM's signed local
   * carrier and true restriction. No HYPRE solve is performed in C0. Failure
   * preserves the complete last-successful system and its generation.
   */
  bool AssembleTrueTraceSystem(
      const std::vector<ElementCondensedBlock>& blocks) noexcept;
  const mfem::HypreParMatrix* assembled_true_trace_matrix() const noexcept;
  const mfem::Vector* assembled_true_trace_right_hand_side() const noexcept;
  TraceAssemblyIdentity assembled_trace_identity() const noexcept;

  /** Execute only the exact C1-MFEM-PERIODIC-02 coupled fixture. */
  bool RunCoupledOperatorFixture(
      CoupledOperatorFixtureSnapshot& output) noexcept;

  /** Initialize only C2-MFEM-PERSISTENT-N8-R1-01 on an N=8 foundation. */
  TrajectoryStatus InitializePersistentTrajectory() noexcept;

  /** Advance one accepted step, owning the single same-dt retry if needed. */
  TrajectoryStatus AdvancePersistentTrajectory(
      TrajectoryFailureInjection injection =
          TrajectoryFailureInjection::none) noexcept;

  bool CopyPublishedTrajectoryState(
      PersistentTrajectoryState& output) const noexcept;
  bool CaptureTrajectoryCheckpoint(
      PersistentTrajectoryCheckpoint& output) const noexcept;
  TrajectoryStatus RestoreTrajectoryCheckpoint(
      const PersistentTrajectoryCheckpoint& checkpoint) noexcept;
  const PersistentTrajectoryEvidence& trajectory_evidence() const noexcept;

  /** Initialize only C3-MFEM-PERSISTENT-N8-R2-01 on the frozen x-slab. */
  TrajectoryStatus InitializePersistentRank2Trajectory() noexcept;
  TrajectoryStatus AdvancePersistentRank2Trajectory(
      TrajectoryFailureInjection injection =
          TrajectoryFailureInjection::none) noexcept;
  bool CopyPublishedRank2TrajectoryState(
      PersistentRank2TrajectoryState& output) const noexcept;
  bool CaptureRank2TrajectoryCheckpoint(
      PersistentRank2TrajectoryCheckpoint& output) const noexcept;
  TrajectoryStatus RestoreRank2TrajectoryCheckpoint(
      const PersistentRank2TrajectoryCheckpoint& checkpoint) noexcept;
  const PersistentRank2TrajectoryEvidence& rank2_trajectory_evidence()
      const noexcept;

private:
  friend struct PersistentTrajectoryImplementation;
  friend struct PersistentRank2TrajectoryImplementation;
  struct Implementation;
  explicit PersistentSc25ScalarIgrCpuFoundation(
      std::unique_ptr<Implementation> implementation) noexcept;
  std::unique_ptr<Implementation> implementation_;
};

static_assert(!std::is_polymorphic<
                  PersistentSc25ScalarIgrCpuFoundation>::value,
              "SC25 C0 owner must not add runtime virtual dispatch");

} // namespace dg::hdg::runtime::cpu::sc25
