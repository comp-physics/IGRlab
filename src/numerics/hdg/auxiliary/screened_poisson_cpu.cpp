#include "numerics/hdg/auxiliary/screened_poisson_cpu.hpp"
#include "numerics/hdg/core/host_reference.hpp"
#include "numerics/hdg/trace/trace_dg_mpi.hpp"
#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
#include <_hypre_parcsr_ls.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>
namespace dg::hdg::auxiliary
{
namespace
{
constexpr std::uint8_t serial_mode = 0;
constexpr std::uint8_t rank_two_mode = 1;
constexpr int faces_per_quadrilateral = 4;
constexpr double quadrature_repeat_tolerance = 1.0e-11;
constexpr double true_absolute_tolerance = 1.0e-14;
constexpr double true_relative_tolerance = 1.0e-12;
constexpr double local_absolute_tolerance = 1.0e-12;
constexpr double local_relative_tolerance = 1.0e-11;
constexpr double frozen_gmres_relative_tolerance = 1.0e-12;
constexpr double frozen_gmres_absolute_tolerance = 1.0e-14;
constexpr int frozen_gmres_krylov_dimension = 50;
constexpr int frozen_gmres_maximum_iterations = 300;
constexpr int forced_nonconvergence_maximum_iterations = 1;
static_assert(std::numeric_limits<double>::is_iec559,
              "HDG-AUX-CPU-03 requires IEC 60559 binary64");
static_assert(sizeof(double) == sizeof(std::uint64_t),
              "HDG-AUX-CPU-03 freezes 64-bit double descriptors");
[[noreturn]] void AbortOpaqueCollectiveConstruction(
    MPI_Comm communicator) noexcept
{
  // MFEM parallel constructors may already have entered communicator-wide
  // collectives before throwing. Their exceptional path is therefore one of
  // the documented process-fatal exclusions, not a recoverable local failure
  // that may safely enter a later application-owned consensus.
  MPI_Abort(communicator, EXIT_FAILURE);
  std::abort();
}
template <typename T>
StridedBlockView<T> View(T* data, const LocalIndex batches,
                         const LocalIndex rows,
                         const LocalIndex columns) noexcept
{
  return {data, batches, rows, columns, rows * columns, columns, 1};
}
bool IsFinite(const double value) noexcept { return std::isfinite(value); }
std::uint64_t DoubleBits(const double value) noexcept
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
int DecodeSignedVDof(const int signed_vdof, int& sign) noexcept
{
  if (signed_vdof == std::numeric_limits<int>::min())
  {
    sign = 0;
    return -1;
  }
  sign = signed_vdof < 0 ? -1 : 1;
  return signed_vdof < 0 ? -1 - signed_vdof : signed_vdof;
}
void InitializeDiagnostic(AttemptDiagnostic& diagnostic,
                          const bool select_all_phases) noexcept
{
  diagnostic = {};
  for (std::size_t index = 0; index < diagnostic.phases.size(); ++index)
  {
    diagnostic.phases[index].phase = static_cast<AttemptPhase>(index);
    diagnostic.phases[index].selected = select_all_phases || index == 0;
  }
}
void SetSetupDiagnostic(AttemptDiagnostic& diagnostic, const Status status,
                        const bool passed) noexcept
{
  diagnostic.status = status;
  auto& phase = diagnostic.phases[0];
  phase.selected = true;
  phase.entered = true;
  phase.completed = true;
  phase.passed = passed;
  phase.disposition = passed ? PhaseDisposition::Passed :
      PhaseDisposition::Failed;
}
int LowestFailingRank(MPI_Comm communicator, const int rank,
                      const int ranks, const bool failed) noexcept
{
  const int local = failed ? rank : ranks;
  int global = ranks;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator);
  return global;
}
Status ConsensusFailure(MPI_Comm communicator, const int rank,
                        const int ranks, const bool failed,
                        const StatusCode local_code,
                        const AttemptPhase phase) noexcept
{
  const int origin = LowestFailingRank(communicator, rank, ranks, failed);
  const int root = origin < ranks ? origin : 0;
  int code = rank == root && origin < ranks ? static_cast<int>(local_code) :
      static_cast<int>(StatusCode::Success);
  MPI_Bcast(&code, 1, MPI_INT, root, communicator);
  return {origin < ranks ? static_cast<StatusCode>(code) : StatusCode::Success,
          phase, origin < ranks ? origin : -1};
}
StatusCode PhaseFailureCode(const AttemptPhase phase) noexcept
{
  switch (phase)
  {
    case AttemptPhase::Setup: return StatusCode::SetupFailure;
    case AttemptPhase::Factorization: return StatusCode::FactorizationFailure;
    case AttemptPhase::TrueAssembly: return StatusCode::TrueAssemblyFailure;
    case AttemptPhase::TraceSolve: return StatusCode::TraceSolveFailure;
    case AttemptPhase::Distribution: return StatusCode::DistributionFailure;
    case AttemptPhase::Recovery: return StatusCode::RecoveryFailure;
    case AttemptPhase::Publication: return StatusCode::PublicationFailure;
    case AttemptPhase::Count: break;
  }
  return StatusCode::SetupFailure;
}
bool FiniteVector(const std::vector<double>& values) noexcept
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return IsFinite(value); });
}
std::vector<double> CopyVector(const mfem::Vector& source)
{
  const double* const data = source.HostRead();
  if (source.Size() != 0 && data == nullptr)
  {
    throw std::runtime_error("MFEM vector has no readable host bytes");
  }
  return data == nullptr ? std::vector<double>{} :
      std::vector<double>(data, data + source.Size());
}
bool AssignVector(mfem::Vector& destination,
                  const std::vector<double>& values)
{
  if (destination.Size() != static_cast<int>(values.size())) { return false; }
  double* const data = destination.HostWrite();
  if (!values.empty() && data == nullptr) { return false; }
  if (!values.empty())
  {
    std::memcpy(data, values.data(), values.size() * sizeof(double));
  }
  return true;
}
double RelativeDifference(const std::vector<double>& left,
                          const std::vector<double>& right) noexcept
{
  if (left.size() != right.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  long double difference = 0.0L;
  long double reference = 0.0L;
  for (std::size_t index = 0; index < left.size(); ++index)
  {
    const long double delta = static_cast<long double>(left[index]) -
        right[index];
    difference += delta * delta;
    reference += static_cast<long double>(right[index]) * right[index];
  }
  return std::sqrt(static_cast<double>(difference)) /
      std::max(1.0, std::sqrt(static_cast<double>(reference)));
}
bool ReadScalar(const mfem::Vector& values,
                const mfem::Array<int>& signed_dofs,
                const mfem::Vector& shape, double& value) noexcept
{
  if (signed_dofs.Size() != shape.Size()) { return false; }
  const double* const data = values.HostRead();
  if (data == nullptr) { return false; }
  value = 0.0;
  for (int dof = 0; dof < signed_dofs.Size(); ++dof)
  {
    int sign = 0;
    const int decoded = DecodeSignedVDof(signed_dofs[dof], sign);
    if (decoded < 0 || decoded >= values.Size()) { return false; }
    value += shape[dof] * static_cast<double>(sign) * data[decoded];
  }
  return IsFinite(value);
}
struct InitializationDescriptor
{
  std::uint8_t mode = 0;
  std::uint8_t output_is_null = 0;
  std::uint16_t reserved = 0;
  int degree = 0;
  int elements_per_direction = 0;
  std::uint64_t alpha_bits = 0;
};
static_assert(std::is_trivially_copyable<InitializationDescriptor>::value,
              "collective initialization descriptor must be trivial");
bool SameDescriptor(const InitializationDescriptor& left,
                    const InitializationDescriptor& right) noexcept
{
  return left.mode == right.mode &&
      left.output_is_null == right.output_is_null &&
      left.degree == right.degree &&
      left.elements_per_direction == right.elements_per_direction &&
      left.alpha_bits == right.alpha_bits;
}
struct CommunicatorGuard
{
  MPI_Comm communicator = MPI_COMM_NULL;
  ~CommunicatorGuard() noexcept
  {
    if (communicator == MPI_COMM_NULL) { return; }
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) { MPI_Comm_free(&communicator); }
  }
  CommunicatorGuard() = default;
  CommunicatorGuard(const CommunicatorGuard&) = delete;
  CommunicatorGuard& operator=(const CommunicatorGuard&) = delete;
  MPI_Comm Release() noexcept
  {
    const MPI_Comm result = communicator;
    communicator = MPI_COMM_NULL;
    return result;
  }
};
double ElementMeasure(mfem::ParMesh& mesh, const int element,
                      const int order)
{
  mfem::ElementTransformation* const transformation =
      mesh.GetElementTransformation(element);
  if (transformation == nullptr) { return 0.0; }
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mesh.GetElementGeometry(element), order);
  double measure = 0.0;
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = rule.IntPoint(point);
    transformation->SetIntPoint(&ip);
    measure += ip.weight * transformation->Weight();
  }
  return measure;
}
double FaceMeasure(mfem::ParMesh& mesh, const int face, const int order)
{
  mfem::ElementTransformation* const transformation =
      mesh.GetFaceTransformation(face);
  if (transformation == nullptr) { return 0.0; }
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mesh.GetFaceGeometry(face), order);
  double measure = 0.0;
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = rule.IntPoint(point);
    transformation->SetIntPoint(&ip);
    measure += ip.weight * transformation->Weight();
  }
  return measure;
}
bool ExpectedTrueDofs(const int ranks, const int rank, const int degree,
                      const int n, int& local, std::int64_t& global) noexcept
{
  const std::int64_t nodes_per_face = degree + 1;
  global = INT64_C(2) * n * n * nodes_per_face;
  if (global > std::numeric_limits<int>::max()) { return false; }
  if (ranks == 1)
  {
    local = static_cast<int>(global);
    return true;
  }
  if (ranks != 2 || (n % 2) != 0 || (rank != 0 && rank != 1))
  {
    return false;
  }
  const std::int64_t faces = static_cast<std::int64_t>(n) *
      (rank == 0 ? n + 1 : n - 1);
  const std::int64_t count = faces * nodes_per_face;
  if (count > std::numeric_limits<int>::max()) { return false; }
  local = static_cast<int>(count);
  return true;
}
using LogicalTraceNodeKey = std::tuple<int, int, int, std::int64_t>;
bool BuildLogicalPeriodicTraceDofMap(
    mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& trace_space,
    const int degree, const int n,
    std::vector<GlobalIndex>& local_to_logical,
    int& logical_true_dofs)
{
  if (n != 2 || degree < 1 || mesh.GetNE() != n * n ||
      trace_space.GetVSize() <= 0)
  {
    return false;
  }
  std::map<LogicalTraceNodeKey, int> canonical;
  std::vector<LogicalTraceNodeKey> local_keys(
      static_cast<std::size_t>(trace_space.GetVSize()));
  std::vector<bool> assigned(local_keys.size(), false);
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* const finite_element =
        trace_space.GetFaceElement(face);
    mfem::ElementTransformation* const transformation =
        mesh.GetFaceTransformation(face);
    mfem::Array<int> signed_dofs;
    trace_space.GetFaceVDofs(face, signed_dofs);
    if (finite_element == nullptr || transformation == nullptr ||
        signed_dofs.Size() != finite_element->GetDof() ||
        finite_element->GetNodes().GetNPoints() != finite_element->GetDof())
    {
      return false;
    }
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    mfem::IntegrationPoint center;
    center.Set1w(0.5, 1.0);
    mfem::Vector center_point(screened_poisson_dimension);
    transformation->Transform(center, center_point);
    transformation->SetIntPoint(&center);
    const mfem::DenseMatrix& jacobian = transformation->Jacobian();
    const bool vertical = std::abs(jacobian(1, 0)) >
        std::abs(jacobian(0, 0));
    const double center_x = center_point[0] - std::floor(center_point[0]);
    const double center_y = center_point[1] - std::floor(center_point[1]);
    const double fixed = vertical ? center_x : center_y;
    const double center_varying = vertical ? center_y : center_x;
    const int line = static_cast<int>(std::llround(fixed * n)) % n;
    int segment = static_cast<int>(std::floor(center_varying * n));
    if (segment == n) { segment = 0; }
    for (int dof = 0; dof < signed_dofs.Size(); ++dof)
    {
      int sign = 0;
      const int decoded = DecodeSignedVDof(signed_dofs[dof], sign);
      if (decoded < 0 ||
          static_cast<std::size_t>(decoded) >= local_keys.size() ||
          assigned[static_cast<std::size_t>(decoded)] ||
          (sign != -1 && sign != 1))
      {
        return false;
      }
      mfem::Vector point(screened_poisson_dimension);
      transformation->Transform(nodes.IntPoint(dof), point);
      const double varying = vertical ?
          point[1] - std::floor(point[1]) :
          point[0] - std::floor(point[0]);
      const LogicalTraceNodeKey key{
          vertical ? 0 : 1, line, segment,
          std::llround(varying * 1000000000000.0)};
      canonical.emplace(key, 0);
      local_keys[static_cast<std::size_t>(decoded)] = key;
      assigned[static_cast<std::size_t>(decoded)] = true;
    }
  }
  int ordinal = 0;
  for (auto& item : canonical) { item.second = ordinal++; }
  int expected_local = 0;
  std::int64_t expected_global = 0;
  if (!ExpectedTrueDofs(1, 0, degree, n, expected_local, expected_global) ||
      ordinal != expected_local || expected_global != expected_local ||
      !std::all_of(assigned.begin(), assigned.end(),
                   [](const bool value) { return value; }))
  {
    return false;
  }
  local_to_logical.assign(local_keys.size(), invalid_global_index);
  std::vector<int> multiplicity(static_cast<std::size_t>(ordinal), 0);
  for (std::size_t local = 0; local < local_keys.size(); ++local)
  {
    const auto found = canonical.find(local_keys[local]);
    if (found == canonical.end()) { return false; }
    local_to_logical[local] = found->second;
    ++multiplicity[static_cast<std::size_t>(found->second)];
  }
  const int nodes_on_four_faces = 4 * (degree + 1);
  const int single = static_cast<int>(std::count(multiplicity.begin(),
                                                 multiplicity.end(), 1));
  const int paired = static_cast<int>(std::count(multiplicity.begin(),
                                                 multiplicity.end(), 2));
  if (single != nodes_on_four_faces || paired != nodes_on_four_faces)
  {
    return false;
  }
  logical_true_dofs = ordinal;
  return true;
}
void ConfigureFrozenIlu(mfem::HypreILU& ilu)
{
  ilu.SetType(0);
  ilu.SetLevelOfFill(0);
  ilu.SetMaxIter(1);
  ilu.SetTol(0.0);
  ilu.SetLocalReordering(1);
  ilu.SetPrintLevel(0);
}
bool InspectFrozenIlu(mfem::HypreILU& ilu,
                      SolverEvidence& evidence) noexcept
{
  auto* const data = reinterpret_cast<hypre_ParILUData*>(
      static_cast<HYPRE_Solver>(ilu));
  if (data == nullptr) { return false; }
  evidence.effective_ilu_type = hypre_ParILUDataIluType(data);
  evidence.effective_ilu_level_of_fill = hypre_ParILUDataLfil(data);
  evidence.effective_ilu_local_reordering =
      hypre_ParILUDataReorderingType(data);
  evidence.exact_ilu_configuration = evidence.effective_ilu_type == 0 &&
      evidence.effective_ilu_level_of_fill == 0 &&
      evidence.effective_ilu_local_reordering == 1;
  return evidence.exact_ilu_configuration;
}
struct PublishedTraceOperatorState
{
  // The logical N=2 seam wraps these arrays without transferring ownership.
  // Keep them alive and destroy the Hypre wrapper before either backing store.
  std::unique_ptr<mfem::SparseMatrix> backing_matrix;
  std::unique_ptr<std::array<HYPRE_BigInt, 2>> partition;
  std::unique_ptr<mfem::HypreParMatrix> matrix;
  std::unique_ptr<mfem::Vector> right_hand_side;
};
} // namespace
struct FullyHybridizedScreenedPoissonCpu::Implementation
{
  struct OwnedCommunicator
  {
    MPI_Comm value = MPI_COMM_NULL;
    ~OwnedCommunicator() noexcept
    {
      if (value == MPI_COMM_NULL) { return; }
      int finalized = 0;
      MPI_Finalized(&finalized);
      if (finalized == 0) { MPI_Comm_free(&value); }
    }
  } communicator;
  std::uint8_t mode = serial_mode;
  int rank = -1;
  int ranks = 0;
  int degree = 0;
  int elements_per_direction = 0;
  int quadrature_order = 0;
  double alpha = screened_poisson_primary_alpha;
  bool logical_periodic_trace = false;
  int logical_true_trace_dofs = 0;
  Status published_status{};
  std::uint64_t generation = 0;
  AttemptDiagnostic last_attempt{};
  std::unique_ptr<mfem::L2_FECollection> volume_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::ParFiniteElementSpace> volume_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> flux_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  long mesh_sequence = 0;
  long volume_space_sequence = 0;
  long flux_space_sequence = 0;
  long trace_space_sequence = 0;
  trace::DGInterfaceTraceMap trace_map;
  trace::DGInterfaceTraceExchangePlan exchange_plan;
  std::vector<GlobalIndex> local_to_logical_true_dof;
  std::unique_ptr<AuxiliarySolution> solution;
  std::unique_ptr<PublishedTraceOperatorState> published_trace_operator;
  bool CollectiveAll(const bool local) const noexcept
  {
    const int input = local ? 1 : 0;
    int output = 0;
    MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                  communicator.value);
    return output == 1;
  }
  int OwnedTrueTraceDofs() const noexcept
  {
    return logical_periodic_trace ? logical_true_trace_dofs :
        trace_space->GetTrueVSize();
  }
  std::int64_t GlobalTrueTraceDofs() const noexcept
  {
    return logical_periodic_trace ? logical_true_trace_dofs :
        trace_space->GlobalTrueVSize();
  }
  bool BuildTopology() noexcept
  {
    std::unique_ptr<mfem::Mesh> periodic;
    std::vector<int> partition;
    bool local_ready = false;
    try
    {
      mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
          elements_per_direction, elements_per_direction,
          mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
      if (logical_periodic_trace)
      {
        // MFEM identifies faces by an unordered pair of distinct vertices.
        // A four-quad torus necessarily has parallel edges with the same
        // endpoints, so retain the valid affine geometry and impose the
        // periodic equivalence on trace nodes below.
        periodic = std::make_unique<mfem::Mesh>(std::move(original));
      }
      else
      {
        std::vector<mfem::Vector> translations(
            2, mfem::Vector(screened_poisson_dimension));
        translations[0][0] = 1.0;
        translations[0][1] = 0.0;
        translations[1][0] = 0.0;
        translations[1][1] = 1.0;
        periodic = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
            original, original.CreatePeriodicVertexMapping(translations)));
      }
      partition.assign(static_cast<std::size_t>(periodic->GetNE()), 0);
      if (ranks == 2)
      {
        for (int ey = 0; ey < elements_per_direction; ++ey)
          for (int ex = 0; ex < elements_per_direction; ++ex)
          {
            const int element = ex + elements_per_direction * ey;
            partition[static_cast<std::size_t>(element)] =
                ex < elements_per_direction / 2 ? 0 : 1;
          }
      }
      volume_collection = std::make_unique<mfem::L2_FECollection>(
          degree, screened_poisson_dimension, mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      trace_collection = std::make_unique<mfem::DG_Interface_FECollection>(
          degree, screened_poisson_dimension, mfem::FiniteElement::VALUE,
          mfem::BasisType::GaussLegendre);
      local_ready = periodic != nullptr && volume_collection != nullptr &&
          trace_collection != nullptr &&
          periodic->GetNE() == elements_per_direction *
              elements_per_direction;
    }
    catch (...) { local_ready = false; }
    // Everything above is rank-local. No rank may enter an MFEM parallel
    // constructor until every peer has completed the same local precheck.
    if (!CollectiveAll(local_ready)) { return false; }
    void* mesh_storage =
        ::operator new(sizeof(mfem::ParMesh), std::nothrow);
    if (!CollectiveAll(mesh_storage != nullptr))
    {
      ::operator delete(mesh_storage);
      return false;
    }
    bool mesh_constructed = false;
    try
    {
      mesh.reset(::new (mesh_storage) mfem::ParMesh(
          communicator.value, *periodic, partition.data()));
      mesh_constructed = mesh != nullptr;
    }
    catch (...)
    {
      AbortOpaqueCollectiveConstruction(communicator.value);
    }
    // Only successful opaque constructors reach application-owned consensus.
    if (!CollectiveAll(mesh_constructed))
    {
      mesh.reset();
      return false;
    }
    const auto construct_space = [&](auto& target, auto&& constructor) noexcept
    {
      using Space = mfem::ParFiniteElementSpace;
      void* storage = ::operator new(sizeof(Space), std::nothrow);
      if (!CollectiveAll(storage != nullptr))
      {
        ::operator delete(storage);
        return false;
      }
      bool constructed = false;
      try
      {
        target.reset(constructor(storage));
        constructed = target != nullptr;
      }
      catch (...)
      {
        AbortOpaqueCollectiveConstruction(communicator.value);
      }
      // A constructor exception is process-fatal; this consensus only gates
      // the successful constructor path and its application-owned storage.
      if (!CollectiveAll(constructed))
      {
        target.reset();
        return false;
      }
      return true;
    };
    if (!construct_space(
            volume_space, [&](void* storage)
            {
              return ::new (storage) mfem::ParFiniteElementSpace(
                  mesh.get(), volume_collection.get(), 1,
                  mfem::Ordering::byNODES);
            }) ||
        !construct_space(
            flux_space, [&](void* storage)
            {
              return ::new (storage) mfem::ParFiniteElementSpace(
                  mesh.get(), volume_collection.get(),
                  screened_poisson_dimension, mfem::Ordering::byNODES);
            }) ||
        !construct_space(
            trace_space, [&](void* storage)
            {
              return ::new (storage) mfem::ParFiniteElementSpace(
                  mesh.get(), trace_collection.get(), 1,
                  mfem::Ordering::byNODES);
            }))
    {
      return false;
    }
    bool map_ready = false;
    try
    {
      map_ready = logical_periodic_trace ?
          BuildLogicalPeriodicTraceDofMap(
              *mesh, *trace_space, degree, elements_per_direction,
              local_to_logical_true_dof, logical_true_trace_dofs) :
          trace::BuildDGInterfaceTraceMap(*trace_space, trace_map);
    }
    catch (...) { map_ready = false; }
    if (!CollectiveAll(map_ready)) { return false; }
    bool exchange_ready = false;
    try
    {
      exchange_ready = logical_periodic_trace ||
          trace::BuildDGInterfaceTraceExchangePlan(
              *trace_space, exchange_plan);
    }
    catch (...) { exchange_ready = false; }
    if (!CollectiveAll(exchange_ready)) { return false; }
    bool topology_ready = false;
    try
    {
      mesh_sequence = mesh->GetSequence();
      volume_space_sequence = volume_space->GetSequence();
      flux_space_sequence = flux_space->GetSequence();
      trace_space_sequence = trace_space->GetSequence();
      int expected_local = 0;
      std::int64_t expected_global = 0;
      const bool common_ready =
          mesh->Dimension() == screened_poisson_dimension &&
          mesh->SpaceDimension() == screened_poisson_dimension &&
          volume_space->GetOrdering() == mfem::Ordering::byNODES &&
          flux_space->GetOrdering() == mfem::Ordering::byNODES &&
          trace_space->GetOrdering() == mfem::Ordering::byNODES &&
          volume_space->GetVDim() == 1 &&
          flux_space->GetVDim() == screened_poisson_dimension &&
          trace_space->GetVDim() == 1 &&
          ExpectedTrueDofs(ranks, rank, degree, elements_per_direction,
                           expected_local, expected_global);
      topology_ready = common_ready && (logical_periodic_trace ?
          (ranks == 1 && rank == 0 &&
           local_to_logical_true_dof.size() ==
               static_cast<std::size_t>(trace_space->GetVSize()) &&
           logical_true_trace_dofs == expected_local &&
           logical_true_trace_dofs == expected_global) :
          (trace_map.IsConsistent() && exchange_plan.IsConsistent() &&
           trace_space->GetTrueVSize() == expected_local &&
           trace_space->GlobalTrueVSize() == expected_global));
    }
    catch (...) { topology_ready = false; }
    return CollectiveAll(topology_ready);
  }
};
namespace
{
struct SolveCandidate
{
  mfem::Vector rho;
  mfem::Vector rho_hat;
  mfem::Vector source;
  LocalBlockSnapshots blocks;
  LocalBlockSnapshots repeat_blocks;
  core::HostReferenceEngine factorization;
  std::unique_ptr<mfem::SparseMatrix> local_matrix;
  std::unique_ptr<std::array<HYPRE_BigInt, 2>> logical_partition;
  std::unique_ptr<mfem::ParBilinearForm> assembly_form;
  std::unique_ptr<mfem::HypreParMatrix> true_matrix;
  mfem::Vector local_right_hand_side;
  mfem::Vector true_right_hand_side;
  mfem::Vector owned_true_solution;
  mfem::Vector local_trace_solution;
  std::unique_ptr<mfem::HypreParVector> hypre_right_hand_side;
  std::unique_ptr<mfem::HypreParVector> hypre_solution;
  std::unique_ptr<mfem::HypreParVector> hypre_residual;
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;
  std::vector<double> element_trace_solution;
  std::vector<double> recovered_interior;
  SolverEvidence solver;
  double true_residual_square = 0.0;
  double true_right_hand_side_square = 0.0;
  double local_recovery_maximum = 0.0;
  int right_hand_side_scale_exponent = 0;
  bool exact_zero_right_hand_side_completion = false;
  std::unique_ptr<AuxiliarySolution> solution;
  std::unique_ptr<PublishedTraceOperatorState> published_trace_operator;
};
bool ExactSnapshotField(const mfem::ParGridFunction* field,
                        const mfem::ParFiniteElementSpace& space) noexcept
{
  return field != nullptr && field->ParFESpace() == &space &&
      field->FESpace() == &space && field->Size() == space.GetVSize() &&
      space.GetParMesh() != nullptr && space.GetVDim() == 1 &&
      space.GetOrdering() == mfem::Ordering::byNODES;
}
template <typename ImplementationType>
bool CopyAndValidateSnapshot(
    ImplementationType& implementation,
    const PrescribedSnapshot& snapshot, SolveCandidate& candidate,
    StatusCode& failure_code)
{
  failure_code = StatusCode::SetupFailure;
  if (implementation.mesh->GetSequence() != implementation.mesh_sequence ||
      implementation.volume_space->GetSequence() !=
          implementation.volume_space_sequence ||
      implementation.flux_space->GetSequence() !=
          implementation.flux_space_sequence ||
      implementation.trace_space->GetSequence() !=
          implementation.trace_space_sequence)
  {
    return false;
  }
  failure_code = StatusCode::InvalidArgument;
  if (snapshot.generation == 0 ||
      !ExactSnapshotField(snapshot.rho, *implementation.volume_space) ||
      !ExactSnapshotField(snapshot.source, *implementation.volume_space) ||
      !ExactSnapshotField(snapshot.rho_hat, *implementation.trace_space))
  {
    return false;
  }
  // Identity and caller-provided coefficient bytes are InvalidArgument.
  // Once those identities are accepted, copy/storage failures are internal
  // setup failures (or AllocationFailure when the outer transaction catches
  // std::bad_alloc).
  failure_code = StatusCode::SetupFailure;
  const std::vector<double> rho = CopyVector(*snapshot.rho);
  const std::vector<double> rho_hat = CopyVector(*snapshot.rho_hat);
  const std::vector<double> source = CopyVector(*snapshot.source);
  if (rho.empty() || rho_hat.empty() || source.empty() ||
      !FiniteVector(rho) || !FiniteVector(rho_hat) ||
      !FiniteVector(source) ||
      !std::all_of(rho.begin(), rho.end(),
                   [](const double value) { return value > 0.0; }) ||
      !std::all_of(rho_hat.begin(), rho_hat.end(),
                   [](const double value) { return value > 0.0; }))
  {
    failure_code = StatusCode::InvalidArgument;
    return false;
  }
  candidate.rho.SetSize(static_cast<int>(rho.size()));
  candidate.rho_hat.SetSize(static_cast<int>(rho_hat.size()));
  candidate.source.SetSize(static_cast<int>(source.size()));
  const bool assigned = AssignVector(candidate.rho, rho) &&
      AssignVector(candidate.rho_hat, rho_hat) &&
      AssignVector(candidate.source, source);
  failure_code = assigned ? StatusCode::Success : failure_code;
  return assigned;
}
template <typename ImplementationType>
bool BuildRestriction(
    ImplementationType& implementation,
    LocalBlockSnapshots& blocks)
{
  blocks.restriction.clear();
  blocks.restriction.reserve(
      static_cast<std::size_t>(blocks.local_elements) *
      blocks.trace_unknowns);
  for (int element = 0; element < implementation.mesh->GetNE(); ++element)
  {
    mfem::Array<int> faces, orientations;
    implementation.mesh->GetElementEdges(element, faces, orientations);
    if (faces.Size() != faces_per_quadrilateral ||
        orientations.Size() != faces.Size())
    {
      return false;
    }
    int row = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      mfem::Array<int> signed_dofs;
      implementation.trace_space->GetFaceVDofs(faces[local_face],
                                                signed_dofs);
      const int orientation = orientations[local_face];
      const int* const permutation =
          implementation.trace_collection->DofOrderForOrientation(
              mfem::Geometry::SEGMENT, orientation);
      if ((orientation != 1 && orientation != -1) || permutation == nullptr)
      {
        return false;
      }
      for (int element_face_dof = 0;
           element_face_dof < signed_dofs.Size(); ++element_face_dof)
      {
        int orientation_sign = 0;
        const int canonical_face_dof = DecodeSignedVDof(
            permutation[element_face_dof], orientation_sign);
        if (canonical_face_dof < 0 ||
            canonical_face_dof >= signed_dofs.Size())
        {
          return false;
        }
        int canonical_sign = 0;
        const int decoded = DecodeSignedVDof(
            signed_dofs[canonical_face_dof], canonical_sign);
        const int sign = orientation_sign * canonical_sign;
        if (decoded < 0 ||
            row >= blocks.trace_unknowns ||
            (sign != -1 && sign != 1))
        {
          return false;
        }
        GlobalIndex global_true_dof = invalid_global_index;
        LocalIndex local_true_dof = invalid_local_index;
        RankIndex owner_rank = invalid_rank;
        if (implementation.logical_periodic_trace)
        {
          if (static_cast<std::size_t>(decoded) >=
              implementation.local_to_logical_true_dof.size())
          {
            return false;
          }
          global_true_dof = implementation.local_to_logical_true_dof[
              static_cast<std::size_t>(decoded)];
          if (global_true_dof < 0 ||
              global_true_dof >= implementation.logical_true_trace_dofs)
          {
            return false;
          }
          local_true_dof = static_cast<LocalIndex>(global_true_dof);
          owner_rank = 0;
        }
        else
        {
          if (static_cast<std::size_t>(decoded) >=
              implementation.trace_map.dofs.entries.size())
          {
            return false;
          }
          // The true column is obtained from the authoritative map entry. The
          // decoded local vdof is never substituted for a true column.
          const trace::TraceMapEntry& entry =
              implementation.trace_map.dofs.entries[
                  static_cast<std::size_t>(decoded)];
          if (entry.local_dof != decoded ||
              entry.global_true_dof == invalid_global_index)
          {
            return false;
          }
          global_true_dof = entry.global_true_dof;
          local_true_dof = entry.local_true_dof;
          owner_rank = entry.owner_rank;
        }
        if (global_true_dof == invalid_global_index)
        {
          return false;
        }
        const int signed_local_vdof = sign < 0 ? -1 - decoded : decoded;
        blocks.restriction.push_back(
            {element, row, signed_local_vdof, decoded, sign,
             global_true_dof, local_true_dof, owner_rank});
        ++row;
      }
    }
    if (row != blocks.trace_unknowns) { return false; }
  }
  const bool complete = blocks.restriction.size() ==
      static_cast<std::size_t>(blocks.local_elements) *
          blocks.trace_unknowns;
  if (!complete || !implementation.logical_periodic_trace) { return complete; }
  std::vector<int> logical_column_counts(
      static_cast<std::size_t>(implementation.logical_true_trace_dofs), 0);
  for (const TraceRestrictionEntry& entry : blocks.restriction)
  {
    if (entry.global_true_dof < 0 ||
        entry.global_true_dof >= implementation.logical_true_trace_dofs)
    {
      return false;
    }
    ++logical_column_counts[
        static_cast<std::size_t>(entry.global_true_dof)];
  }
  return std::all_of(logical_column_counts.begin(),
                     logical_column_counts.end(),
                     [](const int count) { return count == 2; });
}
template <typename ImplementationType>
bool AssembleLocalBlocks(
    ImplementationType& implementation,
    const SolveCandidate& candidate, const int quadrature_order,
    LocalBlockSnapshots& blocks, const bool retain_restriction)
{
  const int elements = implementation.mesh->GetNE();
  if (elements <= 0) { return false; }
  const mfem::FiniteElement* const first_fe =
      implementation.volume_space->GetFE(0);
  if (first_fe == nullptr) { return false; }
  const int scalar_dofs = first_fe->GetDof();
  mfem::Array<int> first_faces, first_orientations;
  implementation.mesh->GetElementEdges(0, first_faces, first_orientations);
  if (first_faces.Size() != faces_per_quadrilateral) { return false; }
  const mfem::FiniteElement* const first_trace =
      implementation.trace_space->GetFaceElement(first_faces[0]);
  if (first_trace == nullptr) { return false; }
  const int trace_dofs = faces_per_quadrilateral * first_trace->GetDof();
  const int interior_unknowns = screened_poisson_dimension * scalar_dofs +
      scalar_dofs;
  blocks = {};
  blocks.local_elements = elements;
  blocks.scalar_interior_dofs = scalar_dofs;
  blocks.interior_unknowns = interior_unknowns;
  blocks.trace_unknowns = trace_dofs;
  const std::size_t ne = static_cast<std::size_t>(elements);
  const std::size_t ni = static_cast<std::size_t>(interior_unknowns);
  const std::size_t nt = static_cast<std::size_t>(trace_dofs);
  blocks.a.assign(ne * ni * ni, 0.0);
  blocks.b.assign(ne * ni * nt, 0.0);
  blocks.c.assign(ne * nt * ni, 0.0);
  blocks.d.assign(ne * nt * nt, 0.0);
  blocks.f.assign(ne * ni, 0.0);
  blocks.g.assign(ne * nt, 0.0);
  if (retain_restriction && !BuildRestriction(implementation, blocks))
  {
    return false;
  }
  const auto interior_index = [scalar_dofs](const int component,
                                             const int scalar) noexcept
  {
    return component * scalar_dofs + scalar;
  };
  const int sigma_component = 0;
  const int zx_component = 1;
  const int zy_component = 2;
  for (int element = 0; element < elements; ++element)
  {
    const mfem::FiniteElement* const finite_element =
        implementation.volume_space->GetFE(element);
    mfem::ElementTransformation* const element_transformation =
        implementation.mesh->GetElementTransformation(element);
    if (finite_element == nullptr || element_transformation == nullptr ||
        finite_element->GetDof() != scalar_dofs)
    {
      return false;
    }
    mfem::Array<int> volume_dofs;
    implementation.volume_space->GetElementVDofs(element, volume_dofs);
    if (volume_dofs.Size() != scalar_dofs) { return false; }
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        finite_element->GetGeomType(), quadrature_order);
    mfem::Vector shape(scalar_dofs);
    mfem::DenseMatrix derivative(scalar_dofs,
                                 screened_poisson_dimension);
    const std::size_t a_offset = static_cast<std::size_t>(element) * ni * ni;
    const std::size_t b_offset = static_cast<std::size_t>(element) * ni * nt;
    const std::size_t c_offset = static_cast<std::size_t>(element) * nt * ni;
    const std::size_t d_offset = static_cast<std::size_t>(element) * nt * nt;
    const std::size_t f_offset = static_cast<std::size_t>(element) * ni;
    for (int point = 0; point < volume_rule.GetNPoints(); ++point)
    {
      const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
      element_transformation->SetIntPoint(&ip);
      finite_element->CalcShape(ip, shape);
      finite_element->CalcPhysDShape(*element_transformation, derivative);
      double rho = 0.0;
      double source = 0.0;
      if (!ReadScalar(candidate.rho, volume_dofs, shape, rho) ||
          !ReadScalar(candidate.source, volume_dofs, shape, source) ||
          !(rho > 0.0))
      {
        return false;
      }
      const double weight = ip.weight * element_transformation->Weight();
      if (!IsFinite(weight) || !(weight > 0.0)) { return false; }
      for (int row = 0; row < scalar_dofs; ++row)
      {
        // Frozen row order is [R_zx, R_zy, R_Sigma]; the independent
        // unknown-column order remains [Sigma, z_x, z_y].
        const int row_zx = interior_index(0, row);
        const int row_zy = interior_index(1, row);
        const int row_sigma = interior_index(2, row);
        blocks.f[f_offset + static_cast<std::size_t>(row_sigma)] +=
            weight * shape[row] * source;
        for (int column = 0; column < scalar_dofs; ++column)
        {
          const int column_sigma = interior_index(sigma_component, column);
          const int column_zx = interior_index(zx_component, column);
          const int column_zy = interior_index(zy_component, column);
          blocks.a[a_offset + static_cast<std::size_t>(row_zx) * ni +
                   column_zx] += weight * shape[row] * rho * shape[column];
          blocks.a[a_offset + static_cast<std::size_t>(row_zy) * ni +
                   column_zy] += weight * shape[row] * rho * shape[column];
          blocks.a[a_offset + static_cast<std::size_t>(row_zx) * ni +
                   column_sigma] +=
              weight * derivative(row, 0) * shape[column];
          blocks.a[a_offset + static_cast<std::size_t>(row_zy) * ni +
                   column_sigma] +=
              weight * derivative(row, 1) * shape[column];
          blocks.a[a_offset + static_cast<std::size_t>(row_sigma) * ni +
                   column_sigma] +=
              weight * shape[row] * shape[column] / rho;
          blocks.a[a_offset + static_cast<std::size_t>(row_sigma) * ni +
                   column_zx] += implementation.alpha * weight *
              derivative(row, 0) * shape[column];
          blocks.a[a_offset + static_cast<std::size_t>(row_sigma) * ni +
                   column_zy] += implementation.alpha * weight *
              derivative(row, 1) * shape[column];
        }
      }
    }
    mfem::Array<int> faces, orientations;
    implementation.mesh->GetElementEdges(element, faces, orientations);
    if (faces.Size() != faces_per_quadrilateral ||
        orientations.Size() != faces.Size())
    {
      return false;
    }
    const double element_measure = ElementMeasure(
        *implementation.mesh, element, quadrature_order);
    if (!IsFinite(element_measure) || !(element_measure > 0.0))
    {
      return false;
    }
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* const transformation =
          implementation.mesh->GetFaceElementTransformations(face);
      const mfem::FiniteElement* const trace_element =
          implementation.trace_space->GetFaceElement(face);
      mfem::Array<int> trace_dofs_array;
      implementation.trace_space->GetFaceVDofs(face, trace_dofs_array);
      if (transformation == nullptr || transformation->Face == nullptr ||
          trace_element == nullptr || trace_dofs_array.Size() !=
              trace_element->GetDof())
      {
        return false;
      }
      const int orientation = orientations[local_face];
      const int* const permutation =
          implementation.trace_collection->DofOrderForOrientation(
              mfem::Geometry::SEGMENT, orientation);
      if ((orientation != 1 && orientation != -1) || permutation == nullptr)
      {
        return false;
      }
      const bool element_one = transformation->Elem1No == element;
      if (!element_one && transformation->Elem2No != element)
      {
        return false;
      }
      const double face_measure = FaceMeasure(*implementation.mesh, face,
                                              quadrature_order);
      const double height = element_measure / face_measure;
      if (!IsFinite(face_measure) || !(face_measure > 0.0) ||
          !IsFinite(height) || !(height > 0.0))
      {
        return false;
      }
      const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
          trace_element->GetGeomType(), quadrature_order);
      mfem::Vector interior_shape(scalar_dofs);
      mfem::Vector trace_shape(trace_element->GetDof());
      mfem::Vector oriented_trace_shape(trace_element->GetDof());
      for (int point = 0; point < face_rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
        transformation->SetAllIntPoints(&ip);
        const mfem::IntegrationPoint& element_ip = element_one ?
            transformation->GetElement1IntPoint() :
            transformation->GetElement2IntPoint();
        finite_element->CalcShape(element_ip, interior_shape);
        trace_element->CalcShape(ip, trace_shape);
        for (int local_dof = 0; local_dof < trace_shape.Size(); ++local_dof)
        {
          int orientation_sign = 0;
          const int canonical_dof = DecodeSignedVDof(
              permutation[local_dof], orientation_sign);
          if (canonical_dof < 0 || canonical_dof >= trace_shape.Size())
          {
            return false;
          }
          oriented_trace_shape[local_dof] =
              static_cast<double>(orientation_sign) *
              trace_shape[canonical_dof];
        }
        mfem::Vector scaled(screened_poisson_dimension);
        mfem::CalcOrtho(transformation->Face->Jacobian(), scaled);
        const double norm = scaled.Norml2();
        const double measure = transformation->Face->Weight();
        const double geometry_tolerance = 256.0 *
            std::numeric_limits<double>::epsilon() *
            std::max(1.0, std::max(norm, measure));
        if (!IsFinite(norm) || !IsFinite(measure) || !(norm > 0.0) ||
            !(measure > 0.0) || std::abs(norm - measure) > geometry_tolerance)
        {
          return false;
        }
        const double side = element_one ? 1.0 : -1.0;
        const double normal[screened_poisson_dimension] = {
            side * scaled[0] / norm, side * scaled[1] / norm};
        double rho_hat = 0.0;
        if (!ReadScalar(candidate.rho_hat, trace_dofs_array, trace_shape,
                        rho_hat) || !(rho_hat > 0.0))
        {
          return false;
        }
        const double tau = static_cast<double>((implementation.degree + 1) *
                                                (implementation.degree + 1)) /
            (rho_hat * height);
        const double weight = ip.weight * measure;
        if (!IsFinite(tau) || !(tau > 0.0) || !IsFinite(weight) ||
            !(weight > 0.0))
        {
          return false;
        }
        for (int row = 0; row < scalar_dofs; ++row)
        {
          const int row_zx = interior_index(0, row);
          const int row_zy = interior_index(1, row);
          const int row_sigma = interior_index(2, row);
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const int column_sigma = interior_index(sigma_component, column);
            const int column_zx = interior_index(zx_component, column);
            const int column_zy = interior_index(zy_component, column);
            blocks.a[a_offset + static_cast<std::size_t>(row_sigma) * ni +
                     column_sigma] += implementation.alpha * weight *
                interior_shape[row] * tau * interior_shape[column];
            blocks.a[a_offset + static_cast<std::size_t>(row_sigma) * ni +
                     column_zx] -= implementation.alpha * weight *
                interior_shape[row] * normal[0] * interior_shape[column];
            blocks.a[a_offset + static_cast<std::size_t>(row_sigma) * ni +
                     column_zy] -= implementation.alpha * weight *
                interior_shape[row] * normal[1] * interior_shape[column];
          }
          for (int column = 0;
               column < oriented_trace_shape.Size(); ++column)
          {
            const int local_column = trace_offset + column;
            blocks.b[b_offset + static_cast<std::size_t>(row_zx) * nt +
                     local_column] -= weight * interior_shape[row] *
                normal[0] * oriented_trace_shape[column];
            blocks.b[b_offset + static_cast<std::size_t>(row_zy) * nt +
                     local_column] -= weight * interior_shape[row] *
                normal[1] * oriented_trace_shape[column];
            blocks.b[b_offset + static_cast<std::size_t>(row_sigma) * nt +
                     local_column] -= implementation.alpha * weight *
                interior_shape[row] * tau * oriented_trace_shape[column];
          }
        }
        for (int row = 0; row < oriented_trace_shape.Size(); ++row)
        {
          const int local_row = trace_offset + row;
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const int column_sigma = interior_index(sigma_component, column);
            const int column_zx = interior_index(zx_component, column);
            const int column_zy = interior_index(zy_component, column);
            blocks.c[c_offset + static_cast<std::size_t>(local_row) * ni +
                     column_sigma] -= implementation.alpha * weight *
                oriented_trace_shape[row] * tau * interior_shape[column];
            blocks.c[c_offset + static_cast<std::size_t>(local_row) * ni +
                     column_zx] += implementation.alpha * weight *
                oriented_trace_shape[row] * normal[0] *
                interior_shape[column];
            blocks.c[c_offset + static_cast<std::size_t>(local_row) * ni +
                     column_zy] += implementation.alpha * weight *
                oriented_trace_shape[row] * normal[1] *
                interior_shape[column];
          }
          for (int column = 0;
               column < oriented_trace_shape.Size(); ++column)
          {
            const int local_column = trace_offset + column;
            blocks.d[d_offset + static_cast<std::size_t>(local_row) * nt +
                     local_column] += implementation.alpha * weight *
                oriented_trace_shape[row] * tau *
                oriented_trace_shape[column];
          }
        }
      }
      trace_offset += oriented_trace_shape.Size();
    }
    if (trace_offset != trace_dofs) { return false; }
  }
  return FiniteVector(blocks.a) && FiniteVector(blocks.b) &&
      FiniteVector(blocks.c) && FiniteVector(blocks.d) &&
      FiniteVector(blocks.f) && FiniteVector(blocks.g);
}
double BlockRepeatDifference(const LocalBlockSnapshots& primary,
                             const LocalBlockSnapshots& repeat) noexcept
{
  return std::max({RelativeDifference(primary.a, repeat.a),
                   RelativeDifference(primary.b, repeat.b),
                   RelativeDifference(primary.c, repeat.c),
                   RelativeDifference(primary.d, repeat.d),
                   RelativeDifference(primary.f, repeat.f),
                   RelativeDifference(primary.g, repeat.g)});
}
template <typename ImplementationType>
bool Factorize(ImplementationType&,
               SolveCandidate& candidate)
{
  LocalBlockSnapshots& blocks = candidate.blocks;
  const LocalBlockShape shape{blocks.scalar_interior_dofs,
                              blocks.trace_unknowns, 3, 1};
  blocks.schur.assign(
      static_cast<std::size_t>(blocks.local_elements) *
          blocks.trace_unknowns * blocks.trace_unknowns,
      0.0);
  blocks.condensed_right_hand_side.assign(
      static_cast<std::size_t>(blocks.local_elements) *
          blocks.trace_unknowns,
      0.0);
  const core::LocalSystemBatchView input{
      shape,
      AsConst(View(blocks.a.data(), blocks.local_elements,
                   blocks.interior_unknowns, blocks.interior_unknowns)),
      AsConst(View(blocks.b.data(), blocks.local_elements,
                   blocks.interior_unknowns, blocks.trace_unknowns)),
      AsConst(View(blocks.c.data(), blocks.local_elements,
                   blocks.trace_unknowns, blocks.interior_unknowns)),
      AsConst(View(blocks.d.data(), blocks.local_elements,
                   blocks.trace_unknowns, blocks.trace_unknowns)),
      AsConst(View(blocks.f.data(), blocks.local_elements,
                   blocks.interior_unknowns, 1)),
      AsConst(View(blocks.g.data(), blocks.local_elements,
                   blocks.trace_unknowns, 1))};
  return candidate.factorization.Condense(
             input,
             {View(blocks.schur.data(), blocks.local_elements,
                   blocks.trace_unknowns, blocks.trace_unknowns),
              View(blocks.condensed_right_hand_side.data(),
                   blocks.local_elements, blocks.trace_unknowns, 1)}) ==
          core::Status::success && candidate.factorization.ready() &&
      FiniteVector(blocks.schur) &&
      FiniteVector(blocks.condensed_right_hand_side);
}
template <typename ImplementationType>
bool PrepareTrueAssemblyLocal(
    ImplementationType& implementation,
    SolveCandidate& candidate)
{
  const LocalBlockSnapshots& blocks = candidate.blocks;
  const int assembly_size = implementation.logical_periodic_trace ?
      implementation.logical_true_trace_dofs :
      implementation.trace_space->GetVSize();
  candidate.local_matrix = std::make_unique<mfem::SparseMatrix>(
      assembly_size);
  if (!implementation.logical_periodic_trace)
  {
    candidate.assembly_form = std::make_unique<mfem::ParBilinearForm>(
        implementation.trace_space.get());
  }
  candidate.local_right_hand_side.SetSize(assembly_size);
  candidate.true_right_hand_side.SetSize(
      implementation.OwnedTrueTraceDofs());
  candidate.local_right_hand_side = 0.0;
  for (int element = 0; element < blocks.local_elements; ++element)
  {
    mfem::Array<int> signed_dofs(blocks.trace_unknowns);
    mfem::DenseMatrix local_schur(blocks.trace_unknowns);
    const std::size_t restriction_offset =
        static_cast<std::size_t>(element) * blocks.trace_unknowns;
    const std::size_t schur_offset =
        restriction_offset * blocks.trace_unknowns;
    for (int row = 0; row < blocks.trace_unknowns; ++row)
    {
      const TraceRestrictionEntry& entry =
          blocks.restriction[restriction_offset + row];
      if (entry.element != element || entry.row != row ||
          entry.decoded_local_vdof < 0 ||
          (entry.sign != -1 && entry.sign != 1))
      {
        return false;
      }
      const int assembly_dof = implementation.logical_periodic_trace ?
          static_cast<int>(entry.global_true_dof) :
          entry.decoded_local_vdof;
      if (assembly_dof < 0 || assembly_dof >= assembly_size)
      {
        return false;
      }
      signed_dofs[row] = entry.sign < 0 ? -1 - assembly_dof : assembly_dof;
      candidate.local_right_hand_side[assembly_dof] +=
          static_cast<double>(entry.sign) *
          blocks.condensed_right_hand_side[restriction_offset + row];
      for (int column = 0; column < blocks.trace_unknowns; ++column)
      {
        local_schur(row, column) =
            blocks.schur[schur_offset +
                         static_cast<std::size_t>(row) *
                             blocks.trace_unknowns +
                         column];
      }
    }
    candidate.local_matrix->AddSubMatrix(signed_dofs, signed_dofs,
                                         local_schur);
  }
  candidate.local_matrix->Finalize();
  return (implementation.logical_periodic_trace ||
          candidate.assembly_form != nullptr) &&
      FiniteVector(CopyVector(candidate.local_right_hand_side));
}
template <typename ImplementationType>
bool ParallelAssembleTrueSystem(
    ImplementationType& implementation, SolveCandidate& candidate)
{
  if (implementation.logical_periodic_trace)
  {
    candidate.logical_partition =
        std::make_unique<std::array<HYPRE_BigInt, 2>>();
    (*candidate.logical_partition)[0] = 0;
    (*candidate.logical_partition)[1] =
        implementation.logical_true_trace_dofs;
    candidate.true_matrix = std::make_unique<mfem::HypreParMatrix>(
        implementation.communicator.value,
        implementation.logical_true_trace_dofs,
        candidate.logical_partition->data(), candidate.local_matrix.get());
  }
  else
  {
    candidate.true_matrix.reset(candidate.assembly_form->ParallelAssemble(
        candidate.local_matrix.get()));
  }
  return candidate.true_matrix != nullptr;
}
template <typename ImplementationType>
bool AssembleTrueRightHandSide(
    ImplementationType& implementation, SolveCandidate& candidate)
{
  if (implementation.logical_periodic_trace)
  {
    return AssignVector(candidate.true_right_hand_side,
                        CopyVector(candidate.local_right_hand_side));
  }
  implementation.trace_space->Dof_TrueDof_Matrix()->MultTranspose(
      candidate.local_right_hand_side, candidate.true_right_hand_side);
  return true;
}
template <typename ImplementationType>
bool ValidateTrueAssemblyLocal(
    ImplementationType& implementation, SolveCandidate& candidate)
{
  return candidate.true_matrix != nullptr &&
      candidate.true_matrix->Height() ==
          implementation.OwnedTrueTraceDofs() &&
      candidate.true_matrix->GetGlobalNumRows() ==
          implementation.GlobalTrueTraceDofs() &&
      FiniteVector(CopyVector(candidate.true_right_hand_side));
}
bool ExactPositiveZeroVector(const mfem::Vector& vector) noexcept
{
  try
  {
    const double* const values = vector.HostRead();
    if (vector.Size() != 0 && values == nullptr) { return false; }
    for (int index = 0; index < vector.Size(); ++index)
    {
      if (DoubleBits(values[index]) != DoubleBits(0.0)) { return false; }
    }
    return true;
  }
  catch (...) { return false; }
}
bool InitializeTraceSolution(SolveCandidate& candidate)
{
  *candidate.hypre_solution = 0.0;
  return ExactPositiveZeroVector(*candidate.hypre_solution);
}
bool ScaleVectorByPowerOfTwo(mfem::Vector& vector,
                             const int exponent) noexcept
{
  try
  {
    double* const values = vector.HostReadWrite();
    if (vector.Size() != 0 && values == nullptr) { return false; }
    for (int index = 0; index < vector.Size(); ++index)
    {
      const double original = values[index];
      const double scaled = std::scalbn(original, exponent);
      const double round_trip = std::scalbn(scaled, -exponent);
      if (!IsFinite(original) || !IsFinite(scaled) ||
          DoubleBits(round_trip) != DoubleBits(original))
      {
        return false;
      }
      values[index] = scaled;
    }
    return true;
  }
  catch (...) { return false; }
}
bool InspectLocalVector(const mfem::Vector& vector, bool& exact_zero,
                        double& square) noexcept
{
  exact_zero = true;
  square = 0.0;
  try
  {
    const double* const values = vector.HostRead();
    if (vector.Size() != 0 && values == nullptr) { return false; }
    for (int index = 0; index < vector.Size(); ++index)
    {
      const double value = values[index];
      const double term = value * value;
      if (!IsFinite(value) || !IsFinite(term) ||
          !IsFinite(square + term))
      {
        return false;
      }
      exact_zero = exact_zero && value == 0.0;
      square += term;
    }
    return true;
  }
  catch (...) { return false; }
}
bool RestoreRightHandSideBytes(SolveCandidate& candidate) noexcept
{
  try
  {
    if (candidate.hypre_right_hand_side->Size() !=
        candidate.true_right_hand_side.Size())
    {
      return false;
    }
    const int size = candidate.true_right_hand_side.Size();
    const double* const original = candidate.true_right_hand_side.HostRead();
    double* const restored = candidate.hypre_right_hand_side->HostWrite();
    if (size != 0 && (original == nullptr || restored == nullptr))
    {
      return false;
    }
    if (size != 0)
    {
      std::memcpy(restored, original,
                  static_cast<std::size_t>(size) * sizeof(double));
    }
    return size == 0 ||
        std::memcmp(restored, original,
                    static_cast<std::size_t>(size) * sizeof(double)) == 0;
  }
  catch (...) { return false; }
}
bool FiniteVectorStorage(const mfem::Vector& vector) noexcept
{
  try
  {
    const double* const values = vector.HostRead();
    if (vector.Size() != 0 && values == nullptr) { return false; }
    for (int index = 0; index < vector.Size(); ++index)
    {
      if (!IsFinite(values[index])) { return false; }
    }
    return true;
  }
  catch (...) { return false; }
}
template <typename ImplementationType>
bool PrepareTraceSolveLocal(
    ImplementationType& implementation, SolveCandidate& candidate,
    const bool force_nonconvergence)
{
  if (implementation.logical_periodic_trace)
  {
    candidate.hypre_right_hand_side =
        std::make_unique<mfem::HypreParVector>(*candidate.true_matrix);
    candidate.hypre_solution =
        std::make_unique<mfem::HypreParVector>(*candidate.true_matrix);
  }
  else
  {
    candidate.hypre_right_hand_side =
        std::make_unique<mfem::HypreParVector>(
            implementation.trace_space.get());
    candidate.hypre_solution = std::make_unique<mfem::HypreParVector>(
        implementation.trace_space.get());
  }
  candidate.hypre_residual = std::make_unique<mfem::HypreParVector>(
      *candidate.true_matrix);
  candidate.ilu = std::make_unique<mfem::HypreILU>();
  candidate.gmres = std::make_unique<mfem::HypreGMRES>(
      implementation.communicator.value);
  if (!candidate.hypre_right_hand_side || !candidate.hypre_solution ||
      !candidate.hypre_residual || !candidate.ilu || !candidate.gmres ||
      !AssignVector(*candidate.hypre_right_hand_side,
                    CopyVector(candidate.true_right_hand_side)) ||
      !InitializeTraceSolution(candidate))
  {
    return false;
  }
  ConfigureFrozenIlu(*candidate.ilu);
  candidate.gmres->SetTol(frozen_gmres_relative_tolerance);
  candidate.gmres->SetAbsTol(frozen_gmres_absolute_tolerance);
  candidate.gmres->SetMaxIter(force_nonconvergence ?
      forced_nonconvergence_maximum_iterations :
      frozen_gmres_maximum_iterations);
  candidate.gmres->SetKDim(frozen_gmres_krylov_dimension);
  candidate.gmres->SetLogging(1);
  candidate.gmres->SetPrintLevel(0);
  candidate.gmres->SetZeroInitialIterate();
  candidate.gmres->SetPreconditioner(*candidate.ilu);
  candidate.gmres->SetOperator(*candidate.true_matrix);
  // MFEM v4.9 resets HypreILU when SetOperator is propagated by GMRES.
  ConfigureFrozenIlu(*candidate.ilu);
  HYPRE_Int effective_kdim = -1;
  HYPRE_Int effective_max_iter = -1;
  HYPRE_Real effective_relative_tolerance =
      std::numeric_limits<HYPRE_Real>::quiet_NaN();
  HYPRE_Real effective_absolute_tolerance =
      std::numeric_limits<HYPRE_Real>::quiet_NaN();
  candidate.solver.gmres_krylov_dimension_query_return =
      HYPRE_GMRESGetKDim(
          static_cast<HYPRE_Solver>(*candidate.gmres), &effective_kdim);
  candidate.solver.gmres_relative_tolerance_query_return =
      HYPRE_GMRESGetTol(
          static_cast<HYPRE_Solver>(*candidate.gmres),
          &effective_relative_tolerance);
  candidate.solver.gmres_absolute_tolerance_query_return =
      HYPRE_GMRESGetAbsoluteTol(
          static_cast<HYPRE_Solver>(*candidate.gmres),
          &effective_absolute_tolerance);
  candidate.solver.gmres_maximum_iterations_query_return =
      HYPRE_GMRESGetMaxIter(
          static_cast<HYPRE_Solver>(*candidate.gmres), &effective_max_iter);
  // In pinned HYPRE 3.1 each getter writes its value and returns the current
  // process-wide hypre_error_flag; it is not a per-query success result.
  candidate.solver.gmres_configuration_query_status =
      candidate.solver.gmres_krylov_dimension_query_return != 0 ?
          candidate.solver.gmres_krylov_dimension_query_return :
      candidate.solver.gmres_relative_tolerance_query_return != 0 ?
          candidate.solver.gmres_relative_tolerance_query_return :
      candidate.solver.gmres_absolute_tolerance_query_return != 0 ?
          candidate.solver.gmres_absolute_tolerance_query_return :
          candidate.solver.gmres_maximum_iterations_query_return;
  candidate.solver.effective_gmres_krylov_dimension =
      static_cast<int>(effective_kdim);
  candidate.solver.effective_gmres_max_iterations =
      static_cast<int>(effective_max_iter);
  candidate.solver.effective_gmres_relative_tolerance_bits = DoubleBits(
      static_cast<double>(effective_relative_tolerance));
  candidate.solver.effective_gmres_absolute_tolerance_bits = DoubleBits(
      static_cast<double>(effective_absolute_tolerance));
  candidate.solver.exact_gmres_configuration =
      candidate.solver.gmres_configuration_query_status == 0 &&
      candidate.solver.gmres_krylov_dimension_query_return == 0 &&
      candidate.solver.gmres_relative_tolerance_query_return == 0 &&
      candidate.solver.gmres_absolute_tolerance_query_return == 0 &&
      candidate.solver.gmres_maximum_iterations_query_return == 0 &&
      candidate.solver.effective_gmres_krylov_dimension ==
          frozen_gmres_krylov_dimension &&
      candidate.solver.effective_gmres_max_iterations ==
          (force_nonconvergence ? forced_nonconvergence_maximum_iterations :
                                  frozen_gmres_maximum_iterations) &&
      candidate.solver.effective_gmres_relative_tolerance_bits ==
          DoubleBits(frozen_gmres_relative_tolerance) &&
      candidate.solver.effective_gmres_absolute_tolerance_bits ==
          DoubleBits(frozen_gmres_absolute_tolerance);
  return candidate.solver.exact_gmres_configuration &&
      InspectFrozenIlu(*candidate.ilu, candidate.solver);
}
template <typename ImplementationType>
bool ClassifyAndScaleRightHandSideCollective(
    ImplementationType& implementation, SolveCandidate& candidate,
    const bool force_nonconvergence)
{
  bool local_exact_zero = false;
  double local_square = 0.0;
  const bool local_ready = InspectLocalVector(
      *candidate.hypre_right_hand_side, local_exact_zero, local_square);
  const int local_flags[2] = {
      local_ready ? 1 : 0,
      local_ready && local_exact_zero ? 1 : 0};
  int global_flags[2] = {0, 0};
  if (MPI_Allreduce(local_flags, global_flags, 2, MPI_INT, MPI_MIN,
                    implementation.communicator.value) != MPI_SUCCESS ||
      global_flags[0] != 1)
  {
    return false;
  }
  double global_square = 0.0;
  if (MPI_Allreduce(&local_square, &global_square, 1, MPI_DOUBLE, MPI_SUM,
                    implementation.communicator.value) != MPI_SUCCESS)
  {
    return false;
  }
  candidate.true_right_hand_side_square = global_square;
  const bool global_square_ready = IsFinite(global_square) &&
      global_square >= 0.0;
  if (!implementation.CollectiveAll(global_square_ready)) { return false; }
  if (global_flags[1] == 1)
  {
    candidate.exact_zero_right_hand_side_completion =
        !force_nonconvergence && global_square == 0.0;
    return global_square == 0.0;
  }
  const double original_norm = std::sqrt(global_square);
  const bool norm_ready = original_norm > 0.0 && IsFinite(original_norm);
  const int original_exponent = norm_ready ? std::ilogb(original_norm) : 0;
  const bool exponent_ready = norm_ready &&
      original_exponent != FP_ILOGB0 &&
      original_exponent != FP_ILOGBNAN;
  candidate.right_hand_side_scale_exponent =
      exponent_ready ? -original_exponent : 0;
  const double scaled_norm = exponent_ready ? std::scalbn(
      original_norm, candidate.right_hand_side_scale_exponent) : 0.0;
  const bool local_scaled = exponent_ready && IsFinite(scaled_norm) &&
      scaled_norm >= 1.0 && scaled_norm < 2.0 &&
      ScaleVectorByPowerOfTwo(
          *candidate.hypre_right_hand_side,
          candidate.right_hand_side_scale_exponent);
  // No rank may enter the outer phase consensus (and then HYPRE) while a peer
  // is still validating or scaling its local vector storage.
  return implementation.CollectiveAll(local_scaled);
}
template <typename ImplementationType>
bool RunGmresCollective(ImplementationType& implementation,
                        SolveCandidate& candidate)
{
  if (candidate.exact_zero_right_hand_side_completion) { return true; }
  candidate.hypre_right_hand_side->HypreRead();
  candidate.hypre_solution->HypreReadWrite();
  candidate.solver.gmres_setup_status = HYPRE_ParCSRGMRESSetup(
      static_cast<HYPRE_Solver>(*candidate.gmres), *candidate.true_matrix,
      *candidate.hypre_right_hand_side, *candidate.hypre_solution);
  // Every rank must return from setup and agree that it succeeded before any
  // peer may enter the collective solve. Preserve HYPRE's status/error state.
  if (!implementation.CollectiveAll(
          candidate.solver.gmres_setup_status == 0))
  {
    return false;
  }
  candidate.solver.gmres_solve_status = HYPRE_ParCSRGMRESSolve(
      static_cast<HYPRE_Solver>(*candidate.gmres), *candidate.true_matrix,
      *candidate.hypre_right_hand_side, *candidate.hypre_solution);
  // The enclosing phase consensus is the common solve-return boundary; all
  // ranks then execute the same local query sequence in ReadGmresLocal.
  return true;
}
bool ReadGmresLocal(SolveCandidate& candidate,
                    const bool force_nonconvergence) noexcept
{
  if (candidate.exact_zero_right_hand_side_completion)
  {
    return !force_nonconvergence &&
        candidate.solver.gmres_setup_status == -1 &&
        candidate.solver.gmres_solve_status == -1 &&
        candidate.solver.gmres_convergence_query_status == -1 &&
        candidate.solver.gmres_converged == -1 &&
        candidate.solver.gmres_iterations == 0 &&
        candidate.solver.gmres_reported_residual == 0.0 &&
        candidate.solver.exact_gmres_configuration &&
        ExactPositiveZeroVector(*candidate.hypre_solution);
  }
  HYPRE_Int converged = -1;
  candidate.solver.gmres_convergence_query_status = HYPRE_GMRESGetConverged(
      static_cast<HYPRE_Solver>(*candidate.gmres), &converged);
  candidate.solver.gmres_converged = static_cast<int>(converged);
  candidate.gmres->GetNumIterations(candidate.solver.gmres_iterations);
  candidate.gmres->GetFinalResidualNorm(
      candidate.solver.gmres_reported_residual);
  // A normal solve that exhausts the frozen budget may proceed only through
  // the original-system norm reductions.  Finalization still rejects it
  // because converged remains false; unrelated HYPRE errors stop here.
  const bool convergence_failure_diagnostic =
      !force_nonconvergence &&
      candidate.solver.gmres_solve_status == HYPRE_ERROR_CONV &&
      candidate.solver.gmres_converged == 0 &&
      candidate.solver.gmres_iterations == frozen_gmres_maximum_iterations &&
      candidate.solver.effective_gmres_max_iterations ==
          frozen_gmres_maximum_iterations;
  const bool common_postconditions =
      candidate.solver.gmres_setup_status == 0 &&
      (candidate.solver.gmres_convergence_query_status == 0 ||
       (convergence_failure_diagnostic &&
        candidate.solver.gmres_convergence_query_status ==
            HYPRE_ERROR_CONV)) &&
      candidate.solver.exact_gmres_configuration &&
      candidate.solver.gmres_iterations >= 0 &&
      IsFinite(candidate.solver.gmres_reported_residual);
  return common_postconditions &&
      ((candidate.solver.gmres_converged == 1 &&
        candidate.solver.gmres_convergence_query_status == 0 &&
        candidate.solver.gmres_solve_status == 0) ||
       convergence_failure_diagnostic);
}
template <typename ImplementationType>
bool RestoreOriginalTraceSystemCollective(
    ImplementationType& implementation, SolveCandidate& candidate) noexcept
{
  bool local_restored = candidate.exact_zero_right_hand_side_completion ||
      ScaleVectorByPowerOfTwo(
          *candidate.hypre_solution,
          -candidate.right_hand_side_scale_exponent);
  local_restored = local_restored && RestoreRightHandSideBytes(candidate);
  local_restored = local_restored &&
      FiniteVectorStorage(*candidate.hypre_solution) &&
      (!candidate.exact_zero_right_hand_side_completion ||
       ExactPositiveZeroVector(*candidate.hypre_solution));
  // The original variable and byte-identical RHS are collectively ready
  // before any rank may enter the original-system residual application.
  return implementation.CollectiveAll(local_restored);
}
bool ApplyTrueResidualCollective(SolveCandidate& candidate)
{
  candidate.true_matrix->Mult(*candidate.hypre_solution,
                              *candidate.hypre_residual);
  *candidate.hypre_residual -= *candidate.hypre_right_hand_side;
  return true;
}
template <typename ImplementationType>
bool ReduceTrueResidualSquareCollective(
    ImplementationType& implementation, SolveCandidate& candidate)
{
  candidate.true_residual_square = mfem::InnerProduct(
      implementation.communicator.value, *candidate.hypre_residual,
      *candidate.hypre_residual);
  return IsFinite(candidate.true_residual_square) &&
      candidate.true_residual_square >= 0.0;
}
template <typename ImplementationType>
bool ReduceTrueRightHandSideSquareCollective(
    ImplementationType& implementation, SolveCandidate& candidate)
{
  candidate.true_right_hand_side_square = mfem::InnerProduct(
      implementation.communicator.value, *candidate.hypre_right_hand_side,
      *candidate.hypre_right_hand_side);
  return IsFinite(candidate.true_right_hand_side_square) &&
      candidate.true_right_hand_side_square >= 0.0;
}
bool FinalizeTraceSolveLocal(SolveCandidate& candidate,
                             const bool force_nonconvergence)
{
  candidate.solver.true_residual_norm =
      std::sqrt(candidate.true_residual_square);
  candidate.solver.true_right_hand_side_norm =
      std::sqrt(candidate.true_right_hand_side_square);
  if (candidate.exact_zero_right_hand_side_completion)
  {
    if (force_nonconvergence ||
        candidate.solver.gmres_setup_status != -1 ||
        candidate.solver.gmres_solve_status != -1 ||
        candidate.solver.gmres_convergence_query_status != -1 ||
        candidate.solver.gmres_converged != -1 ||
        candidate.solver.gmres_iterations != 0 ||
        candidate.solver.gmres_reported_residual != 0.0 ||
        candidate.solver.true_residual_norm != 0.0 ||
        candidate.solver.true_right_hand_side_norm != 0.0 ||
        !ExactPositiveZeroVector(*candidate.hypre_solution))
    {
      return false;
    }
  }
  else if (candidate.solver.gmres_setup_status != 0 ||
           candidate.solver.gmres_solve_status != 0 ||
           candidate.solver.gmres_convergence_query_status != 0 ||
           candidate.solver.gmres_converged != 1)
  {
    return false;
  }
  if (candidate.solver.true_residual_norm > true_absolute_tolerance +
          true_relative_tolerance *
              candidate.solver.true_right_hand_side_norm)
  {
    return false;
  }
  candidate.owned_true_solution.SetSize(candidate.hypre_solution->Size());
  return AssignVector(candidate.owned_true_solution,
                      CopyVector(*candidate.hypre_solution));
}
template <typename ImplementationType>
bool PrepareDistributionLocal(
    ImplementationType& implementation,
    SolveCandidate& candidate)
{
  candidate.local_trace_solution.SetSize(
      implementation.trace_space->GetVSize());
  candidate.local_trace_solution = 0.0;
  return candidate.local_trace_solution.Size() ==
      implementation.trace_space->GetVSize();
}
template <typename ImplementationType>
bool DistributeTraceCollective(
    ImplementationType& implementation, SolveCandidate& candidate)
{
  if (implementation.logical_periodic_trace)
  {
    const double* const owned = candidate.owned_true_solution.HostRead();
    double* const local = candidate.local_trace_solution.HostWrite();
    if (owned == nullptr || local == nullptr ||
        implementation.local_to_logical_true_dof.size() !=
            static_cast<std::size_t>(candidate.local_trace_solution.Size()))
    {
      return false;
    }
    for (int dof = 0; dof < candidate.local_trace_solution.Size(); ++dof)
    {
      const GlobalIndex logical =
          implementation.local_to_logical_true_dof[
              static_cast<std::size_t>(dof)];
      if (logical < 0 || logical >= candidate.owned_true_solution.Size())
      {
        return false;
      }
      local[dof] = owned[static_cast<int>(logical)];
    }
    return true;
  }
  return implementation.exchange_plan.Distribute(
      *implementation.trace_space, candidate.owned_true_solution,
      candidate.local_trace_solution);
}
bool ValidateDistributionLocal(const SolveCandidate& candidate)
{
  return FiniteVector(CopyVector(candidate.local_trace_solution));
}
template <typename ImplementationType>
bool RecoverInteriorLocal(
    ImplementationType&, SolveCandidate& candidate)
{
  const LocalBlockSnapshots& blocks = candidate.blocks;
  const std::size_t trace_count =
      static_cast<std::size_t>(blocks.local_elements) *
      blocks.trace_unknowns;
  candidate.element_trace_solution.assign(trace_count, 0.0);
  const double* const local = candidate.local_trace_solution.HostRead();
  if (local == nullptr) { return false; }
  for (std::size_t row = 0; row < trace_count; ++row)
  {
    const TraceRestrictionEntry& entry = blocks.restriction[row];
    if (entry.decoded_local_vdof < 0 ||
        entry.decoded_local_vdof >= candidate.local_trace_solution.Size())
    {
      return false;
    }
    candidate.element_trace_solution[row] =
        static_cast<double>(entry.sign) * local[entry.decoded_local_vdof];
  }
  candidate.recovered_interior.assign(
      static_cast<std::size_t>(blocks.local_elements) *
          blocks.interior_unknowns,
      0.0);
  if (candidate.factorization.Recover(
          AsConst(View(candidate.element_trace_solution.data(),
                       blocks.local_elements, blocks.trace_unknowns, 1)),
          View(candidate.recovered_interior.data(), blocks.local_elements,
               blocks.interior_unknowns, 1)) != core::Status::success ||
      !FiniteVector(candidate.recovered_interior))
  {
    return false;
  }
  double local_maximum = 0.0;
  for (int element = 0; element < blocks.local_elements; ++element)
  {
    const std::size_t ni = static_cast<std::size_t>(blocks.interior_unknowns);
    const std::size_t nt = static_cast<std::size_t>(blocks.trace_unknowns);
    const std::size_t x_offset = static_cast<std::size_t>(element) * ni;
    const std::size_t lambda_offset = static_cast<std::size_t>(element) * nt;
    const std::size_t a_offset = x_offset * ni;
    const std::size_t b_offset = x_offset * nt;
    long double residual_square = 0.0L;
    long double rhs_square = 0.0L;
    for (std::size_t row = 0; row < ni; ++row)
    {
      long double value = -blocks.f[x_offset + row];
      long double rhs = blocks.f[x_offset + row];
      for (std::size_t column = 0; column < ni; ++column)
      {
        value += blocks.a[a_offset + row * ni + column] *
            candidate.recovered_interior[x_offset + column];
      }
      for (std::size_t column = 0; column < nt; ++column)
      {
        value += blocks.b[b_offset + row * nt + column] *
            candidate.element_trace_solution[lambda_offset + column];
        rhs -= blocks.b[b_offset + row * nt + column] *
            candidate.element_trace_solution[lambda_offset + column];
      }
      residual_square += value * value;
      rhs_square += rhs * rhs;
    }
    const double residual = std::sqrt(static_cast<double>(residual_square));
    const double rhs = std::sqrt(static_cast<double>(rhs_square));
    if (!IsFinite(residual) || !IsFinite(rhs) ||
        residual > local_absolute_tolerance + local_relative_tolerance * rhs)
    {
      return false;
    }
    local_maximum = std::max(local_maximum, residual);
  }
  candidate.local_recovery_maximum = local_maximum;
  return true;
}
template <typename ImplementationType>
bool ReduceRecoveryResidualCollective(
    ImplementationType& implementation, SolveCandidate& candidate) noexcept
{
  MPI_Allreduce(&candidate.local_recovery_maximum,
                &candidate.solver.maximum_local_recovery_residual, 1,
                MPI_DOUBLE, MPI_MAX, implementation.communicator.value);
  return true;
}
bool ValidateRecoveryLocal(const SolveCandidate& candidate) noexcept
{
  return IsFinite(candidate.solver.maximum_local_recovery_residual);
}
template <typename ImplementationType>
bool PrepareSolution(
    ImplementationType& implementation,
    const PrescribedSnapshot& snapshot, SolveCandidate& candidate)
{
  const LocalBlockSnapshots& blocks = candidate.blocks;
  candidate.solution = std::make_unique<AuxiliarySolution>();
  AuxiliarySolution& solution = *candidate.solution;
  solution.polynomial_degree = implementation.degree;
  solution.elements_per_direction = implementation.elements_per_direction;
  solution.alpha = implementation.alpha;
  solution.input_generation = snapshot.generation;
  solution.solution_generation = implementation.generation + 1;
  const int scalar = blocks.scalar_interior_dofs;
  solution.sigma.resize(
      static_cast<std::size_t>(blocks.local_elements) * scalar);
  solution.z.resize(
      static_cast<std::size_t>(blocks.local_elements) *
      screened_poisson_dimension * scalar);
  for (int element = 0; element < blocks.local_elements; ++element)
  {
    const std::size_t input_offset =
        static_cast<std::size_t>(element) * blocks.interior_unknowns;
    const std::size_t sigma_offset =
        static_cast<std::size_t>(element) * scalar;
    const std::size_t z_offset = static_cast<std::size_t>(element) *
        screened_poisson_dimension * scalar;
    for (int dof = 0; dof < scalar; ++dof)
    {
      solution.sigma[sigma_offset + dof] =
          candidate.recovered_interior[input_offset + dof];
      solution.z[z_offset + dof] =
          candidate.recovered_interior[input_offset + scalar + dof];
      solution.z[z_offset + scalar + dof] =
          candidate.recovered_interior[input_offset + 2 * scalar + dof];
    }
  }
  solution.local_sigma_hat = CopyVector(candidate.local_trace_solution);
  solution.owned_true_sigma_hat = CopyVector(candidate.owned_true_solution);
  solution.solver = candidate.solver;
  solution.blocks = candidate.blocks;
  candidate.published_trace_operator =
      std::make_unique<PublishedTraceOperatorState>();
  if (implementation.logical_periodic_trace)
  {
    candidate.published_trace_operator->backing_matrix =
        std::move(candidate.local_matrix);
    candidate.published_trace_operator->partition =
        std::move(candidate.logical_partition);
  }
  candidate.published_trace_operator->matrix = std::move(candidate.true_matrix);
  candidate.published_trace_operator->right_hand_side =
      std::make_unique<mfem::Vector>(candidate.true_right_hand_side);
  solution.census.rank = implementation.rank;
  solution.census.ranks = implementation.ranks;
  solution.census.local_elements = blocks.local_elements;
  solution.census.global_elements = implementation.elements_per_direction *
      implementation.elements_per_direction;
  solution.census.local_trace_dofs =
      implementation.trace_space->GetVSize();
  solution.census.owned_true_trace_dofs =
      implementation.OwnedTrueTraceDofs();
  solution.census.global_true_trace_dofs =
      implementation.GlobalTrueTraceDofs();
  solution.census.owned_trace_entries = implementation.logical_periodic_trace ?
      static_cast<std::size_t>(implementation.logical_true_trace_dofs) :
      implementation.trace_map.OwnedCount();
  solution.census.ghost_trace_entries = implementation.logical_periodic_trace ?
      0U : implementation.trace_map.GhostCount();
  solution.census.negative_restriction_rows = static_cast<int>(
      std::count_if(blocks.restriction.begin(), blocks.restriction.end(),
                    [](const TraceRestrictionEntry& entry)
                    { return entry.sign < 0; }));
  return candidate.published_trace_operator->matrix != nullptr &&
      candidate.published_trace_operator->right_hand_side != nullptr &&
      FiniteVector(solution.sigma) && FiniteVector(solution.z) &&
      FiniteVector(solution.local_sigma_hat) &&
      FiniteVector(solution.owned_true_sigma_hat);
}
bool IsInjected(const FailureInjection& injection,
                const AttemptPhase phase, const int rank) noexcept
{
  return injection.phase == phase && injection.rank == rank;
}
template <typename ImplementationType>
class PhaseTransaction
{
public:
  PhaseTransaction(ImplementationType& implementation,
                   AttemptDiagnostic& diagnostic,
                   const AttemptPhase phase,
                   const FailureInjection& injection, bool& active,
                   Status& terminal_status) noexcept
    : implementation_(implementation),
      record_(diagnostic.phases[static_cast<std::size_t>(phase)]),
      phase_(phase), active_(active), terminal_status_(terminal_status),
      entered_(active), open_(active)
  {
    record_.entered = entered_;
    const bool injected = open_ &&
        IsInjected(injection, phase_, implementation_.rank);
    Consensus(!injected, injected ? StatusCode::InjectedFailure :
              PhaseFailureCode(phase_));
  }
  PhaseTransaction(const PhaseTransaction&) = delete;
  PhaseTransaction& operator=(const PhaseTransaction&) = delete;
  bool open() const noexcept { return open_; }
  void Consensus(const bool local_success,
                 const StatusCode local_failure_code) noexcept
  {
    const Status consensus = ConsensusFailure(
        implementation_.communicator.value, implementation_.rank,
        implementation_.ranks, open_ && !local_success,
        local_failure_code, phase_);
    if (open_ && consensus.code != StatusCode::Success)
    {
      terminal_status_ = consensus;
      open_ = false;
      active_ = false;
    }
  }
  void Finish() noexcept
  {
    record_.completed = entered_;
    if (!entered_)
    {
      record_.disposition = PhaseDisposition::NotReached;
      record_.passed = false;
    }
    else if (open_)
    {
      record_.disposition = PhaseDisposition::Passed;
      record_.passed = true;
    }
    else
    {
      record_.disposition = PhaseDisposition::Failed;
      record_.passed = false;
    }
  }
private:
  ImplementationType& implementation_;
  PhaseDiagnostic& record_;
  AttemptPhase phase_;
  bool& active_;
  Status& terminal_status_;
  bool entered_ = false;
  bool open_ = false;
};
} // namespace
FullyHybridizedScreenedPoissonCpu::FullyHybridizedScreenedPoissonCpu(
    std::unique_ptr<Implementation> implementation) noexcept
  : implementation_(std::move(implementation))
{
}
FullyHybridizedScreenedPoissonCpu::~FullyHybridizedScreenedPoissonCpu()
    noexcept = default;
FullyHybridizedScreenedPoissonCpu::FullyHybridizedScreenedPoissonCpu(
    FullyHybridizedScreenedPoissonCpu&&) noexcept = default;
Status FullyHybridizedScreenedPoissonCpu::Create(
    MPI_Comm communicator, CpuSerialExecutionTag, PeriodicUnitTorusV1Tag,
    const int degree, const int elements_per_direction,
    const ScreenedPoissonPolicy policy,
    std::unique_ptr<FullyHybridizedScreenedPoissonCpu>& output,
    AttemptDiagnostic& diagnostic) noexcept
{
  return CreateInternal(communicator, serial_mode, degree,
                        elements_per_direction, policy, false, output,
                        diagnostic);
}
Status FullyHybridizedScreenedPoissonCpu::Create(
    MPI_Comm communicator, CpuMpiRank2ExecutionTag, PeriodicUnitTorusV1Tag,
    const int degree, const int elements_per_direction,
    const ScreenedPoissonPolicy policy,
    std::unique_ptr<FullyHybridizedScreenedPoissonCpu>& output,
    AttemptDiagnostic& diagnostic) noexcept
{
  return CreateInternal(communicator, rank_two_mode, degree,
                        elements_per_direction, policy, false, output,
                        diagnostic);
}
Status FullyHybridizedScreenedPoissonCpu::CreateInternal(
    MPI_Comm communicator, const std::uint8_t mode, const int degree,
    const int elements_per_direction, const ScreenedPoissonPolicy policy,
    const bool admit_two_by_two_test_mesh,
    std::unique_ptr<FullyHybridizedScreenedPoissonCpu>& output,
    AttemptDiagnostic& diagnostic) noexcept
{
  InitializeDiagnostic(diagnostic, false);
  int rank = -1;
  int ranks = 0;
  // Invalid communicator handles, lifecycle divergence, and failures of MPI
  // collectives themselves are process-fatal exclusions of the contract.
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &ranks);
  const InitializationDescriptor local_descriptor{
      mode, static_cast<std::uint8_t>(output == nullptr ? 1 : 0), 0,
      degree, elements_per_direction, DoubleBits(policy.alpha)};
  InitializationDescriptor root_descriptor{};
  if (rank == 0) { root_descriptor = local_descriptor; }
  MPI_Bcast(&root_descriptor, static_cast<int>(sizeof(root_descriptor)),
            MPI_BYTE, 0, communicator);
  const bool descriptor_mismatch =
      !SameDescriptor(local_descriptor, root_descriptor);
  const int mismatch_origin = LowestFailingRank(
      communicator, rank, ranks, descriptor_mismatch);
  if (mismatch_origin < ranks)
  {
    const Status status{StatusCode::InvalidArgument, AttemptPhase::Setup,
                        mismatch_origin};
    SetSetupDiagnostic(diagnostic, status, false);
    return status;
  }
  StatusCode local_validation_code = StatusCode::Success;
  if ((mode == serial_mode && ranks != 1) ||
      (mode == rank_two_mode && ranks != 2))
  {
    local_validation_code = StatusCode::UnsupportedCommunicator;
  }
  else if (output != nullptr || (mode != serial_mode && mode != rank_two_mode) ||
           degree < 1 || degree > 4 ||
           !IsFinite(policy.alpha) || !(policy.alpha > 0.0) ||
           !((elements_per_direction == 8 ||
              elements_per_direction == 16 ||
              elements_per_direction == 32) ||
             (admit_two_by_two_test_mesh &&
              mode == serial_mode && ranks == 1 &&
              degree == 1 && elements_per_direction == 2)) ||
           (mode == rank_two_mode &&
            (elements_per_direction % 2) != 0))
  {
    local_validation_code = StatusCode::InvalidArgument;
  }
  Status validation = ConsensusFailure(
      communicator, rank, ranks,
      local_validation_code != StatusCode::Success,
      local_validation_code, AttemptPhase::Setup);
  if (validation.code != StatusCode::Success)
  {
    SetSetupDiagnostic(diagnostic, validation, false);
    return validation;
  }
  std::unique_ptr<Implementation> implementation;
  std::unique_ptr<FullyHybridizedScreenedPoissonCpu> owner;
  bool allocated = true;
  try
  {
    implementation.reset(new (std::nothrow) Implementation());
    allocated = implementation != nullptr;
    if (allocated)
    {
      owner.reset(new (std::nothrow) FullyHybridizedScreenedPoissonCpu(
          std::move(implementation)));
      allocated = owner != nullptr && owner->implementation_ != nullptr;
    }
  }
  catch (...)
  {
    allocated = false;
  }
  Status allocation_status = ConsensusFailure(
      communicator, rank, ranks, !allocated,
      StatusCode::AllocationFailure, AttemptPhase::Setup);
  if (allocation_status.code != StatusCode::Success)
  {
    owner.reset();
    SetSetupDiagnostic(diagnostic, allocation_status, false);
    return allocation_status;
  }
  Implementation& next = *owner->implementation_;
  next.mode = mode;
  next.rank = rank;
  next.ranks = ranks;
  next.degree = degree;
  next.elements_per_direction = elements_per_direction;
  next.quadrature_order = 4 * degree + 8;
  next.alpha = policy.alpha;
  next.logical_periodic_trace = admit_two_by_two_test_mesh &&
      mode == serial_mode && ranks == 1 && degree == 1 &&
      elements_per_direction == 2;
  CommunicatorGuard duplicate;
  const int duplicate_result = MPI_Comm_dup(communicator,
                                             &duplicate.communicator);
  const bool duplicate_ok = duplicate_result == MPI_SUCCESS &&
      duplicate.communicator != MPI_COMM_NULL;
  Status duplicate_status = ConsensusFailure(
      communicator, rank, ranks, !duplicate_ok,
      StatusCode::SetupFailure, AttemptPhase::Setup);
  if (duplicate_status.code != StatusCode::Success)
  {
    owner.reset();
    SetSetupDiagnostic(diagnostic, duplicate_status, false);
    return duplicate_status;
  }
  next.communicator.value = duplicate.Release();
  bool topology_ok = true;
  try
  {
    topology_ok = next.BuildTopology();
  }
  catch (...)
  {
    topology_ok = false;
  }
  Status topology_status = ConsensusFailure(
      next.communicator.value, rank, ranks, !topology_ok,
      StatusCode::SetupFailure, AttemptPhase::Setup);
  if (topology_status.code != StatusCode::Success)
  {
    owner.reset();
    SetSetupDiagnostic(diagnostic, topology_status, false);
    return topology_status;
  }
  next.published_status = {StatusCode::Ready, AttemptPhase::Setup, -1};
  next.generation = 1;
  InitializeDiagnostic(next.last_attempt, false);
  next.last_attempt.status = next.published_status;
  next.last_attempt.attempt_generation = 1;
  output = std::move(owner);
  const Status status{StatusCode::Ready, AttemptPhase::Setup, -1};
  SetSetupDiagnostic(diagnostic, status, true);
  diagnostic.attempt_generation = 1;
  return status;
}
Status FullyHybridizedScreenedPoissonCpu::Solve(
    const PrescribedSnapshot& snapshot,
    const FailureInjection& injection) noexcept
{
  if (implementation_ == nullptr)
  {
    return {StatusCode::Uninitialized, AttemptPhase::Setup, -1};
  }
  Implementation& state = *implementation_;
  AttemptDiagnostic diagnostic;
  InitializeDiagnostic(diagnostic, true);
  diagnostic.attempt_generation = state.last_attempt.attempt_generation + 1;
  diagnostic.input_generation = snapshot.generation;
  SolveCandidate candidate;
  bool active = true;
  Status terminal{StatusCode::Success, AttemptPhase::Publication, -1};
  const auto local_step = [&](auto& phase, const StatusCode default_code,
                              auto&& work) noexcept
  {
    bool local_success = true;
    StatusCode local_code = default_code;
    if (phase.open())
    {
      try
      {
        local_success = work(local_code);
      }
      catch (const std::bad_alloc&)
      {
        local_success = false;
        local_code = StatusCode::AllocationFailure;
      }
      catch (...) { local_success = false; }
    }
    phase.Consensus(local_success, local_code);
  };
  {
    PhaseTransaction phase(state, diagnostic, AttemptPhase::Setup,
                           injection, active, terminal);
    local_step(phase, StatusCode::InvalidArgument,
               [&](StatusCode& code)
               { return CopyAndValidateSnapshot(state, snapshot, candidate,
                                                 code); });
    local_step(phase, StatusCode::SetupFailure, [&](StatusCode&)
               { return AssembleLocalBlocks(state, candidate,
                     state.quadrature_order, candidate.blocks, true); });
    local_step(phase, StatusCode::SetupFailure, [&](StatusCode&)
               { return AssembleLocalBlocks(state, candidate,
                     state.quadrature_order + 2, candidate.repeat_blocks,
                     false); });
    local_step(phase, StatusCode::SetupFailure, [&](StatusCode&)
    {
      candidate.solver.quadrature_repeat_relative_change =
          BlockRepeatDifference(candidate.blocks, candidate.repeat_blocks);
      return IsFinite(candidate.solver.quadrature_repeat_relative_change) &&
          candidate.solver.quadrature_repeat_relative_change <=
              quadrature_repeat_tolerance;
    });
    phase.Finish();
  }
  {
    PhaseTransaction phase(state, diagnostic, AttemptPhase::Factorization,
                           injection, active, terminal);
    local_step(phase, StatusCode::FactorizationFailure,
               [&](StatusCode&) { return Factorize(state, candidate); });
    phase.Finish();
  }
  {
    PhaseTransaction phase(state, diagnostic, AttemptPhase::TrueAssembly,
                           injection, active, terminal);
    // The first consensus is the local-readiness gate for ParallelAssemble.
    local_step(phase, StatusCode::TrueAssemblyFailure,
               [&](StatusCode&)
               { return PrepareTrueAssemblyLocal(state, candidate); });
    local_step(phase, StatusCode::TrueAssemblyFailure,
               [&](StatusCode&)
               { return ParallelAssembleTrueSystem(state, candidate); });
    // The post-assembly consensus gates the true-DoF transpose operation.
    local_step(phase, StatusCode::TrueAssemblyFailure,
               [&](StatusCode&)
               { return AssembleTrueRightHandSide(state, candidate); });
    local_step(phase, StatusCode::TrueAssemblyFailure,
               [&](StatusCode&)
               { return ValidateTrueAssemblyLocal(state, candidate); });
    phase.Finish();
  }
  {
    PhaseTransaction phase(state, diagnostic, AttemptPhase::TraceSolve,
                           injection, active, terminal);
    // No rank enters HYPRE until all ranks have constructed and inspected the
    // frozen local solver configuration.
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return PrepareTraceSolveLocal(
                     state, candidate, injection.force_trace_nonconvergence); });
    // Exact-zero RHS completion is classified collectively before any HYPRE
    // setup, solve, or convergence query.  Every nonzero RHS is normalized
    // by one deterministic exact power of two.  HYPRE solves A*y=2^k*b from
    // the verified zero vector; x=2^-k*y restores the unchanged trace system
    // before its independent residual and publication.
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return ClassifyAndScaleRightHandSideCollective(
                     state, candidate,
                     injection.force_trace_nonconvergence); });
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return RunGmresCollective(state, candidate); });
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return ReadGmresLocal(
                     candidate, injection.force_trace_nonconvergence); });
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return RestoreOriginalTraceSystemCollective(
                     state, candidate); });
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return ApplyTrueResidualCollective(candidate); });
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return ReduceTrueResidualSquareCollective(state,
                                                            candidate); });
    // A post-reduction consensus is the readiness gate for the second norm.
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return ReduceTrueRightHandSideSquareCollective(state,
                                                                 candidate); });
    local_step(phase, StatusCode::TraceSolveFailure,
               [&](StatusCode&)
               { return FinalizeTraceSolveLocal(
                     candidate, injection.force_trace_nonconvergence); });
    phase.Finish();
  }
  {
    PhaseTransaction phase(state, diagnostic, AttemptPhase::Distribution,
                           injection, active, terminal);
    local_step(phase, StatusCode::DistributionFailure,
               [&](StatusCode&)
               { return PrepareDistributionLocal(state, candidate); });
    local_step(phase, StatusCode::DistributionFailure,
               [&](StatusCode&)
               { return DistributeTraceCollective(state, candidate); });
    local_step(phase, StatusCode::DistributionFailure,
               [&](StatusCode&)
               { return ValidateDistributionLocal(candidate); });
    phase.Finish();
  }
  {
    PhaseTransaction phase(state, diagnostic, AttemptPhase::Recovery,
                           injection, active, terminal);
    local_step(phase, StatusCode::RecoveryFailure,
               [&](StatusCode&)
               { return RecoverInteriorLocal(state, candidate); });
    local_step(phase, StatusCode::RecoveryFailure,
               [&](StatusCode&)
               { return ReduceRecoveryResidualCollective(state, candidate); });
    local_step(phase, StatusCode::RecoveryFailure,
               [&](StatusCode&) { return ValidateRecoveryLocal(candidate); });
    local_step(phase, StatusCode::RecoveryFailure,
               [&](StatusCode&)
               { return PrepareSolution(state, snapshot, candidate); });
    phase.Finish();
  }
  {
    PhaseTransaction phase(state, diagnostic, AttemptPhase::Publication,
                           injection, active, terminal);
    local_step(phase, StatusCode::PublicationFailure,
               [&](StatusCode&) { return candidate.solution != nullptr; });
    phase.Finish();
  }
  if (active)
  {
    // Publication occurs only after the final publication consensus.  Moving
    // a unique_ptr and assigning scalar metadata are noexcept operations.
    state.solution = std::move(candidate.solution);
    state.published_trace_operator =
        std::move(candidate.published_trace_operator);
    ++state.generation;
    state.published_status =
        {StatusCode::Success, AttemptPhase::Publication, -1};
    terminal = state.published_status;
  }
  diagnostic.status = terminal;
  diagnostic.solver = candidate.solver;
  state.last_attempt = diagnostic;
  return terminal;
}
Status FullyHybridizedScreenedPoissonCpu::status() const noexcept
{
  return implementation_ == nullptr ? Status{} :
      implementation_->published_status;
}
const AttemptDiagnostic&
FullyHybridizedScreenedPoissonCpu::last_attempt() const noexcept
{
  static const AttemptDiagnostic empty{};
  return implementation_ == nullptr ? empty : implementation_->last_attempt;
}
std::uint64_t FullyHybridizedScreenedPoissonCpu::generation() const noexcept
{
  return implementation_ == nullptr ? 0 : implementation_->generation;
}
int FullyHybridizedScreenedPoissonCpu::degree() const noexcept
{
  return implementation_ == nullptr ? 0 : implementation_->degree;
}
int FullyHybridizedScreenedPoissonCpu::elements_per_direction() const noexcept
{
  return implementation_ == nullptr ? 0 :
      implementation_->elements_per_direction;
}
double FullyHybridizedScreenedPoissonCpu::alpha() const noexcept
{
  return implementation_ == nullptr ? 0.0 : implementation_->alpha;
}
const AuxiliarySolution*
FullyHybridizedScreenedPoissonCpu::solution() const noexcept
{
  return implementation_ == nullptr ? nullptr : implementation_->solution.get();
}
const LocalBlockSnapshots*
FullyHybridizedScreenedPoissonCpu::blocks() const noexcept
{
  const AuxiliarySolution* const value = solution();
  return value == nullptr ? nullptr : &value->blocks;
}
const mfem::ParMesh* FullyHybridizedScreenedPoissonCpu::mesh() const noexcept
{
  return implementation_ == nullptr ? nullptr : implementation_->mesh.get();
}
const mfem::ParFiniteElementSpace*
FullyHybridizedScreenedPoissonCpu::volume_space() const noexcept
{
  return implementation_ == nullptr ? nullptr :
      implementation_->volume_space.get();
}
const mfem::ParFiniteElementSpace*
FullyHybridizedScreenedPoissonCpu::flux_space() const noexcept
{
  return implementation_ == nullptr ? nullptr :
      implementation_->flux_space.get();
}
const mfem::ParFiniteElementSpace*
FullyHybridizedScreenedPoissonCpu::trace_space() const noexcept
{
  return implementation_ == nullptr ? nullptr :
      implementation_->trace_space.get();
}
bool FullyHybridizedScreenedPoissonCpu::ApplyPublishedTraceOperator(
    const std::vector<double>& owned_true_input,
    std::vector<double>& owned_true_action,
    std::vector<double>& owned_true_right_hand_side) const noexcept
{
  if (implementation_ == nullptr)
  {
    return false;
  }
  mfem::Vector input;
  mfem::Vector action;
  bool locally_ready = false;
  try
  {
    locally_ready = implementation_->published_trace_operator != nullptr &&
        implementation_->published_trace_operator->matrix != nullptr &&
        implementation_->published_trace_operator->right_hand_side != nullptr &&
        owned_true_input.size() == static_cast<std::size_t>(
            implementation_->published_trace_operator->matrix->Width());
    if (locally_ready)
    {
      input.SetSize(static_cast<int>(owned_true_input.size()));
      action.SetSize(
          implementation_->published_trace_operator->matrix->Height());
      locally_ready = AssignVector(input, owned_true_input);
    }
  }
  catch (...) { locally_ready = false; }
  if (!implementation_->CollectiveAll(locally_ready)) { return false; }
  std::vector<double> action_candidate;
  std::vector<double> rhs_candidate;
  bool locally_complete = false;
  try
  {
    implementation_->published_trace_operator->matrix->Mult(input, action);
    action_candidate = CopyVector(action);
    rhs_candidate = CopyVector(
        *implementation_->published_trace_operator->right_hand_side);
    locally_complete = FiniteVector(action_candidate) &&
        FiniteVector(rhs_candidate) &&
        action_candidate.size() == rhs_candidate.size();
  }
  catch (...) { locally_complete = false; }
  if (!implementation_->CollectiveAll(locally_complete)) { return false; }
  owned_true_action.swap(action_candidate);
  owned_true_right_hand_side.swap(rhs_candidate);
  return true;
}
} // namespace dg::hdg::auxiliary
