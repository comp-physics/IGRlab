#pragma once

/**
 * @file numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp
 * @brief MFEM-native rank-one foundation for SC25 projection-v3.
 *
 * This default-disabled component owns the C0 field/space/topology foundation
 * and the exact C1 coupled-operator, condensation, true-solve, and recovery
 * fixture.  It does not claim a nonlinear step, trajectory, restart, CLI,
 * output, rank-two, CUDA, or E2/E3 result.
 */

#include "numerics/hdg/auxiliary/screened_poisson_cpu.hpp"
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
};

static_assert(std::is_empty<CpuRank1Selection>::value,
              "SC25 rank-one selection must remain compile-time only");

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

private:
  struct Implementation;
  explicit PersistentSc25ScalarIgrCpuFoundation(
      std::unique_ptr<Implementation> implementation) noexcept;
  std::unique_ptr<Implementation> implementation_;
};

static_assert(!std::is_polymorphic<
                  PersistentSc25ScalarIgrCpuFoundation>::value,
              "SC25 C0 owner must not add runtime virtual dispatch");

} // namespace dg::hdg::runtime::cpu::sc25
