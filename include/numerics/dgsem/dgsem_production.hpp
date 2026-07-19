#pragma once

/**
 * @file numerics/dgsem/dgsem_production.hpp
 * @brief Fail-closed contracts for the bounded production DGSEM device path.
 * @ingroup numerics_dgsem
 *
 * The production path is intentionally limited to the accepted correctness
 * tuple: 1D/2D p1--p4 Gauss--Lobatto Euler, central volume flux without
 * volume dissipation, Rusanov surfaces, and zero or prescribed stage-frozen
 * scalar sigma.  Unsupported tuples are named here so strict evidence can
 * reject them without mutating the caller's output.
 */

#include "mfem.hpp"

#include <cstdint>

namespace mfem
{
namespace dgsem
{

enum class DGSEMProductionReason : int
{
  None = 0,
  CacheNotBuilt,
  UnsupportedDimension,
  UnsupportedOrder,
  MixedOrder,
  UnsupportedTensorElement,
  UnsupportedNodalBasis,
  UnsupportedEquationCount,
  UnsupportedModel,
  FluxIdentityMismatch,
  CollectivePolicyMismatch,
  UnsupportedVolumeFlux,
  UnsupportedVolumeDissipation,
  UnsupportedStabilization,
  UnsupportedSurfaceFlux,
  UnsupportedBoundary,
  BoundaryOverlap,
  BoundaryDimensionMismatch,
  BoundaryFluxMismatch,
  UnsupportedSigma,
  UnsupportedSigmaSpeedPolicy,
  MultiRankCuda,
  InvalidGamma,
  EulerGammaMismatch,
  InvalidDensityFloor,
  InvalidLexicographicMap,
  InvalidElementVdofMap,
  InvalidFaceOwnership,
  InvalidFaceRange,
  InvalidNormal,
  InvalidWeight,
  InvalidMetric,
  InvalidSigma,
  InvalidDensity,
  DensityBelowBoundaryFloor,
  InvalidPressure,
  InputSizeMismatch,
  OutputSizeMismatch,
  StaleCache,
  WorkspaceIdentityChanged
};

const char* DGSEMProductionReasonName(DGSEMProductionReason reason);

struct DGSEMProductionSelection
{
  bool supported = false;
  DGSEMProductionReason reason = DGSEMProductionReason::CacheNotBuilt;
  bool strict = false;
  bool cuda = false;
  bool uses_sigma = false;
  bool has_local_faces = false;
  bool has_shared_faces = false;
  bool has_boundary_faces = false;
  int dimension = 0;
  int order = 0;
  int elements = 0;
  int ranks = 1;
};

struct DGSEMProductionResult
{
  bool executed = false;
  DGSEMProductionReason reason = DGSEMProductionReason::CacheNotBuilt;
  real_t max_characteristic_speed = 0.0;
};

enum class DGSEMProductionInvalidation : int
{
  PolicyOnly = 0,
  SigmaMetadata = 1,
  Geometry = 2,
  Topology = 3
};

struct DGSEMProductionCacheCounters
{
  std::int64_t policy_binds = 0;
  std::int64_t topology_builds = 0;
  std::int64_t geometry_builds = 0;
  std::int64_t sigma_metadata_builds = 0;
  std::int64_t workspace_builds = 0;
  std::int64_t update_invalidations = 0;
  std::int64_t stage_calls = 0;
  std::int64_t state_exchanges = 0;
  std::int64_t sigma_exchanges = 0;
  std::int64_t mpi_reductions = 0;
  std::int64_t selection_reason_reductions = 0;
  std::int64_t stage_reason_reductions = 0;
  std::int64_t host_synchronizations = 0;
  std::int64_t output_scatters = 0;
  std::int64_t workspace_identity_checks = 0;
  std::int64_t workspace_identity_failures = 0;
  std::int64_t unchanged_cache_fast_paths = 0;
  std::int64_t rebuild_preflight_entries = 0;
};

/** Exact warm-stage workspace snapshot.
 *
 * The signatures cover every packed stage workspace, including MPI send,
 * receive, request, and status storage when present. The storage signature
 * includes the stable MFEM host/device addresses after warmup; size and
 * capacity signatures are kept separate so evidence can distinguish shape
 * changes from allocator relocation. This is not an allocator-wide or
 * performance measurement.
 */
struct DGSEMProductionWorkspaceEvidence
{
  bool ready = false;
  std::uint64_t size_signature = 0;
  std::uint64_t capacity_signature = 0;
  std::uint64_t storage_signature = 0;
  std::int64_t checks = 0;
  std::int64_t failures = 0;
};

/** Evidence-only snapshot for the coordinator-owned production-Mult bridge.
 *
 * Capturing storage identity synchronizes MFEM memory and may materialize
 * host/device views. It is deliberately outside the production hot path and
 * makes no allocation-free observation claim.
 */
struct DGSEMProductionBridgeEvidence
{
  bool operator_ready = false;
  int x_local_size = 0;
  int z_local_size = 0;
  int y_local_size = 0;
  int x_local_capacity = 0;
  int z_local_capacity = 0;
  int y_local_capacity = 0;
  std::uint64_t workspace_storage_signature = 0;
  std::int64_t operator_constructions = 0;
  std::int64_t operator_rebinds = 0;
  std::int64_t production_mult_calls = 0;
  std::int64_t fallback_mult_calls = 0;
};

/** Read-only packed metadata view used by the production preflight.
 *
 * All arrays use the accepted packed layouts.  The element map is ordered
 * `(element, equation, local)` with `local` fastest.  Face matrices use one
 * row per nodal face point.  A signed MFEM vdof `v < 0` addresses `-1-v` with
 * a sign flip.
 */
struct DGSEMProductionMetadataView
{
  int dimension = 0;
  int order = 0;
  int elements = 0;
  int dofs_per_element = 0;
  int equations = 0;
  int local_vector_size = 0;
  int face_neighbor_vector_size = 0;
  const Array<int>* lex_to_local = nullptr;
  const Array<int>* element_vdofs = nullptr;
  const DenseMatrix* metrics = nullptr;
  const DenseMatrix* inverse_mass = nullptr;
  const DenseMatrix* normals = nullptr;
  const Vector* face_weights = nullptr;
  const Array<int>* left_elements = nullptr;
  const Array<int>* left_local_dofs = nullptr;
  const Array<int>* right_elements = nullptr;
  const Array<int>* right_local_dofs = nullptr;
  const Array<int>* ownership = nullptr;
  const Array<int>* shared_state_vdofs = nullptr;
};

DGSEMProductionReason
ValidateDGSEMProductionMetadata(const DGSEMProductionMetadataView& metadata);

DGSEMProductionReason ValidateDGSEMProductionState(
  int dimension,
  real_t gamma,
  real_t velocity_density_floor,
  const DenseMatrix& states,
  const Vector& sigma);

} // namespace dgsem
} // namespace mfem
