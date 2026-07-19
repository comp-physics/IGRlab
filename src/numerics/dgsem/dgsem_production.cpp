#include "numerics/dgsem/dgsem_operator.hpp"

#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_device.hpp"
#include "numerics/dgsem/dgsem_metric.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/igr/igr_flux.hpp"
#include "utils/config/dg_igr_defaults.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace mfem
{
namespace dgsem
{
namespace
{

inline int AbsoluteDof(int vdof)
{
  return vdof >= 0 ? vdof : -1 - vdof;
}

inline bool Finite(real_t value)
{
  return std::isfinite(static_cast<double>(value));
}

inline int FindNode(const Vector& nodes, real_t coordinate)
{
  int best = 0;
  real_t best_error = std::abs(coordinate - nodes(0));
  for (int i = 1; i < nodes.Size(); ++i)
  {
    const real_t error = std::abs(coordinate - nodes(i));
    if (error < best_error)
    {
      best_error = error;
      best = i;
    }
  }
  return best_error < 1.0e-10 ? best : -1;
}

struct ProductionCommunicator
{
  int size = 1;
#ifdef MFEM_USE_MPI
  MPI_Comm comm = MPI_COMM_SELF;
  bool active = false;
#endif
};

ProductionCommunicator GetProductionCommunicator(
  const FiniteElementSpace& fes)
{
  ProductionCommunicator context;
#ifdef MFEM_USE_MPI
  if (Mpi::IsInitialized() && !Mpi::IsFinalized())
  {
    context.active = true;
    const auto* pfes = dynamic_cast<const ParFiniteElementSpace*>(&fes);
    context.comm = pfes ? pfes->GetComm() : MPI_COMM_SELF;
    MPI_Comm_size(context.comm, &context.size);
  }
#else
  (void)fes;
#endif
  return context;
}

DGSEMProductionReason ReduceReason(
  const ProductionCommunicator& context,
  DGSEMProductionReason local,
  std::int64_t& reductions)
{
#ifdef MFEM_USE_MPI
  if (context.active && context.size > 1)
  {
    int local_code = static_cast<int>(local);
    int global_code = local_code;
    MPI_Allreduce(&local_code,
                  &global_code,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  context.comm);
    ++reductions;
    return static_cast<DGSEMProductionReason>(global_code);
  }
#else
  (void)context;
  (void)reductions;
#endif
  return local;
}

bool ReduceAny(const ProductionCommunicator& context,
               bool local,
               std::int64_t& reductions)
{
#ifdef MFEM_USE_MPI
  if (context.active && context.size > 1)
  {
    int local_value = local ? 1 : 0;
    int global_value = local_value;
    MPI_Allreduce(&local_value,
                  &global_value,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  context.comm);
    ++reductions;
    return global_value != 0;
  }
#else
  (void)context;
  (void)reductions;
#endif
  return local;
}

bool ReduceEqual(const ProductionCommunicator& context,
                 std::uint64_t local,
                 std::int64_t& reductions)
{
#ifdef MFEM_USE_MPI
  if (context.active && context.size > 1)
  {
    std::uint64_t minimum = local;
    std::uint64_t maximum = local;
    MPI_Allreduce(&local,
                  &minimum,
                  1,
                  MPI_UINT64_T,
                  MPI_MIN,
                  context.comm);
    MPI_Allreduce(&local,
                  &maximum,
                  1,
                  MPI_UINT64_T,
                  MPI_MAX,
                  context.comm);
    reductions += 2;
    return minimum == maximum;
  }
#else
  (void)context;
  (void)reductions;
#endif
  return true;
}

std::uint64_t HashCombine(std::uint64_t hash, std::uint64_t value)
{
  hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
  return hash;
}

template <typename T>
std::uint64_t HashPointer(std::uint64_t hash, const T* pointer)
{
  return HashCombine(
    hash,
    static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)));
}

std::uint64_t HashReal(std::uint64_t hash, real_t value)
{
  static_assert(sizeof(real_t) <= sizeof(std::uint64_t),
                "real_t fingerprint assumes at most 64 bits");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(real_t));
  return HashCombine(hash, bits);
}

#ifdef MFEM_USE_MPI
void ExchangeFaceNeighborVector(
  ParFiniteElementSpace& pfes,
  const Vector& local,
  Vector& send,
  Vector& neighbor,
  std::vector<MPI_Request>& requests,
  std::vector<MPI_Status>& statuses)
{
  ParMesh* mesh = pfes.GetParMesh();
  const int neighbors = mesh ? mesh->GetNFaceNeighbors() : 0;
  const int send_size = pfes.send_face_nbr_ldof.Size_of_connections();
  MFEM_VERIFY(local.Size() == pfes.GetVSize() &&
                send.Size() == send_size &&
                neighbor.Size() == pfes.GetFaceNbrVSize() &&
                requests.size() == static_cast<std::size_t>(2 * neighbors) &&
                statuses.size() == static_cast<std::size_t>(neighbors),
              "DGSEM production persistent MFEM exchange size mismatch.");
  if (neighbors == 0)
  {
    return;
  }

  const int* send_ldof =
    mfem::Read(pfes.send_face_nbr_ldof.GetJMemory(), send_size);
  const real_t* input = local.Read();
  real_t* send_value = send.Write();
  mfem::forall(
    send_size,
    [=] MFEM_HOST_DEVICE(int i)
    {
      const int ldof = send_ldof[i];
      send_value[i] = input[ldof >= 0 ? ldof : -1 - ldof];
    });

  const real_t* send_host = send.HostRead();
  real_t* neighbor_host = neighbor.HostWrite();
  int* send_offset = pfes.send_face_nbr_ldof.GetI();
  int* receive_offset = pfes.face_nbr_ldof.GetI();
  MPI_Request* send_requests = requests.data();
  MPI_Request* receive_requests = requests.data() + neighbors;
  for (int face_neighbor = 0; face_neighbor < neighbors; ++face_neighbor)
  {
    const int rank = mesh->GetFaceNbrRank(face_neighbor);
    MPI_Isend(send_host + send_offset[face_neighbor],
              send_offset[face_neighbor + 1] - send_offset[face_neighbor],
              MPITypeMap<real_t>::mpi_type,
              rank,
              0,
              pfes.GetComm(),
              send_requests + face_neighbor);
    MPI_Irecv(neighbor_host + receive_offset[face_neighbor],
              receive_offset[face_neighbor + 1] -
                receive_offset[face_neighbor],
              MPITypeMap<real_t>::mpi_type,
              rank,
              0,
              pfes.GetComm(),
              receive_requests + face_neighbor);
  }
  MPI_Waitall(neighbors, send_requests, statuses.data());
  MPI_Waitall(neighbors, receive_requests, statuses.data());
}
#endif

} // namespace

const char* DGSEMProductionReasonName(DGSEMProductionReason reason)
{
  switch (reason)
  {
  case DGSEMProductionReason::None: return "none";
  case DGSEMProductionReason::CacheNotBuilt: return "cache-not-built";
  case DGSEMProductionReason::UnsupportedDimension:
    return "unsupported-dimension";
  case DGSEMProductionReason::UnsupportedOrder: return "unsupported-order";
  case DGSEMProductionReason::MixedOrder: return "mixed-order";
  case DGSEMProductionReason::UnsupportedTensorElement:
    return "unsupported-tensor-element";
  case DGSEMProductionReason::UnsupportedNodalBasis:
    return "unsupported-nodal-basis";
  case DGSEMProductionReason::UnsupportedEquationCount:
    return "unsupported-equation-count";
  case DGSEMProductionReason::UnsupportedModel: return "unsupported-model";
  case DGSEMProductionReason::FluxIdentityMismatch:
    return "flux-identity-mismatch";
  case DGSEMProductionReason::CollectivePolicyMismatch:
    return "collective-policy-mismatch";
  case DGSEMProductionReason::UnsupportedVolumeFlux:
    return "unsupported-volume-flux";
  case DGSEMProductionReason::UnsupportedVolumeDissipation:
    return "unsupported-volume-dissipation";
  case DGSEMProductionReason::UnsupportedStabilization:
    return "unsupported-stabilization";
  case DGSEMProductionReason::UnsupportedSurfaceFlux:
    return "unsupported-surface-flux";
  case DGSEMProductionReason::UnsupportedBoundary:
    return "unsupported-boundary";
  case DGSEMProductionReason::BoundaryOverlap: return "boundary-overlap";
  case DGSEMProductionReason::BoundaryDimensionMismatch:
    return "boundary-dimension-mismatch";
  case DGSEMProductionReason::BoundaryFluxMismatch:
    return "boundary-flux-mismatch";
  case DGSEMProductionReason::UnsupportedSigma: return "unsupported-sigma";
  case DGSEMProductionReason::UnsupportedSigmaSpeedPolicy:
    return "unsupported-sigma-speed-policy";
  case DGSEMProductionReason::MultiRankCuda: return "multi-rank-cuda";
  case DGSEMProductionReason::InvalidGamma: return "invalid-gamma";
  case DGSEMProductionReason::EulerGammaMismatch:
    return "euler-gamma-mismatch";
  case DGSEMProductionReason::InvalidDensityFloor:
    return "invalid-density-floor";
  case DGSEMProductionReason::InvalidLexicographicMap:
    return "invalid-lexicographic-map";
  case DGSEMProductionReason::InvalidElementVdofMap:
    return "invalid-element-vdof-map";
  case DGSEMProductionReason::InvalidFaceOwnership:
    return "invalid-face-ownership";
  case DGSEMProductionReason::InvalidFaceRange: return "invalid-face-range";
  case DGSEMProductionReason::InvalidNormal: return "invalid-normal";
  case DGSEMProductionReason::InvalidWeight: return "invalid-weight";
  case DGSEMProductionReason::InvalidMetric: return "invalid-metric";
  case DGSEMProductionReason::InvalidSigma: return "invalid-sigma";
  case DGSEMProductionReason::InvalidDensity: return "invalid-density";
  case DGSEMProductionReason::DensityBelowBoundaryFloor:
    return "density-below-boundary-floor";
  case DGSEMProductionReason::InvalidPressure: return "invalid-pressure";
  case DGSEMProductionReason::InputSizeMismatch:
    return "input-size-mismatch";
  case DGSEMProductionReason::OutputSizeMismatch:
    return "output-size-mismatch";
  case DGSEMProductionReason::StaleCache: return "stale-cache";
  case DGSEMProductionReason::WorkspaceIdentityChanged:
    return "workspace-identity-changed";
  }
  return "unknown";
}

DGSEMProductionReason
ValidateDGSEMProductionMetadata(const DGSEMProductionMetadataView& m)
{
  if (m.dimension != 1 && m.dimension != 2)
  {
    return DGSEMProductionReason::UnsupportedDimension;
  }
  if (m.order < 1 || m.order > 4)
  {
    return DGSEMProductionReason::UnsupportedOrder;
  }
  const int n1d = m.order + 1;
  const int expected_dof =
    m.dimension == 1 ? n1d : n1d * n1d;
  if (m.elements <= 0 || m.dofs_per_element != expected_dof)
  {
    return DGSEMProductionReason::UnsupportedTensorElement;
  }
  if (m.equations != m.dimension + 2)
  {
    return DGSEMProductionReason::UnsupportedEquationCount;
  }
  const int rows = m.elements * m.dofs_per_element;
  if (!m.lex_to_local || m.lex_to_local->Size() != m.dofs_per_element)
  {
    return DGSEMProductionReason::InvalidLexicographicMap;
  }
  std::vector<char> lex_seen(static_cast<std::size_t>(m.dofs_per_element), 0);
  const int* lex = m.lex_to_local->HostRead();
  for (int i = 0; i < m.dofs_per_element; ++i)
  {
    if (lex[i] < 0 || lex[i] >= m.dofs_per_element || lex_seen[lex[i]])
    {
      return DGSEMProductionReason::InvalidLexicographicMap;
    }
    lex_seen[lex[i]] = 1;
  }

  if (!m.element_vdofs ||
      m.element_vdofs->Size() != rows * m.equations ||
      m.local_vector_size != rows * m.equations)
  {
    return DGSEMProductionReason::InvalidElementVdofMap;
  }
  std::vector<char> vdof_seen(static_cast<std::size_t>(m.local_vector_size), 0);
  const int* vdofs = m.element_vdofs->HostRead();
  for (int i = 0; i < m.element_vdofs->Size(); ++i)
  {
    const int vdof = AbsoluteDof(vdofs[i]);
    if (vdof < 0 || vdof >= m.local_vector_size || vdof_seen[vdof])
    {
      return DGSEMProductionReason::InvalidElementVdofMap;
    }
    vdof_seen[vdof] = 1;
  }

  if (!m.metrics || m.metrics->Height() != rows ||
      m.metrics->Width() != m.dimension * m.dimension)
  {
    return DGSEMProductionReason::InvalidMetric;
  }
  const real_t* metric = m.metrics->HostRead();
  for (int row = 0; row < rows; ++row)
  {
    for (int ref = 0; ref < m.dimension; ++ref)
    {
      real_t norm2 = 0.0;
      for (int physical = 0; physical < m.dimension; ++physical)
      {
        const real_t value =
          metric[row + rows * (ref * m.dimension + physical)];
        if (!Finite(value))
        {
          return DGSEMProductionReason::InvalidMetric;
        }
        norm2 += value * value;
      }
      if (!(norm2 > 0.0) || !Finite(norm2))
      {
        return DGSEMProductionReason::InvalidMetric;
      }
    }
    if (m.dimension == 2)
    {
      const real_t determinant =
        metric[row] * metric[row + rows * 3] -
        metric[row + rows] * metric[row + rows * 2];
      if (!Finite(determinant) || determinant == 0.0)
      {
        return DGSEMProductionReason::InvalidMetric;
      }
    }
  }

  if (!m.inverse_mass || m.inverse_mass->Height() != rows ||
      m.inverse_mass->Width() != m.dofs_per_element)
  {
    return DGSEMProductionReason::InvalidMetric;
  }
  const real_t* inverse = m.inverse_mass->HostRead();
  for (int row = 0; row < rows; ++row)
  {
    real_t row_norm = 0.0;
    for (int j = 0; j < m.dofs_per_element; ++j)
    {
      const real_t value = inverse[row + rows * j];
      if (!Finite(value))
      {
        return DGSEMProductionReason::InvalidMetric;
      }
      row_norm += std::abs(value);
    }
    if (!(row_norm > 0.0) || !Finite(row_norm))
    {
      return DGSEMProductionReason::InvalidMetric;
    }
  }

  if (!m.normals || !m.face_weights || !m.left_elements ||
      !m.left_local_dofs || !m.right_elements || !m.right_local_dofs ||
      !m.ownership || !m.shared_state_vdofs)
  {
    return DGSEMProductionReason::InvalidFaceRange;
  }
  const int nfaces = m.normals->Height();
  if (m.normals->Width() != m.dimension ||
      m.face_weights->Size() != nfaces ||
      m.left_elements->Size() != nfaces ||
      m.left_local_dofs->Size() != nfaces ||
      m.right_elements->Size() != nfaces ||
      m.right_local_dofs->Size() != nfaces ||
      m.ownership->Size() != nfaces ||
      m.shared_state_vdofs->Size() != nfaces * m.equations)
  {
    return DGSEMProductionReason::InvalidFaceRange;
  }
  const real_t* normal = m.normals->HostRead();
  const real_t* weight = m.face_weights->HostRead();
  const int* left_element = m.left_elements->HostRead();
  const int* left_local = m.left_local_dofs->HostRead();
  const int* right_element = m.right_elements->HostRead();
  const int* right_local = m.right_local_dofs->HostRead();
  const int* owner = m.ownership->HostRead();
  const int* shared_vdof = m.shared_state_vdofs->HostRead();
  for (int face = 0; face < nfaces; ++face)
  {
    real_t normal_norm2 = 0.0;
    for (int d = 0; d < m.dimension; ++d)
    {
      const real_t value = normal[face + nfaces * d];
      if (!Finite(value))
      {
        return DGSEMProductionReason::InvalidNormal;
      }
      normal_norm2 += value * value;
    }
    if (!(normal_norm2 > 0.0) || !Finite(normal_norm2))
    {
      return DGSEMProductionReason::InvalidNormal;
    }
    if (!Finite(weight[face]) || !(weight[face] > 0.0))
    {
      return DGSEMProductionReason::InvalidWeight;
    }
    if (left_element[face] < 0 || left_element[face] >= m.elements ||
        left_local[face] < 0 || left_local[face] >= m.dofs_per_element)
    {
      return DGSEMProductionReason::InvalidFaceRange;
    }
    const int ownership_value = owner[face];
    if (ownership_value ==
        static_cast<int>(DGSEMDeviceFaceOwnership::Local))
    {
      if (right_element[face] < 0 || right_element[face] >= m.elements ||
          right_local[face] < 0 ||
          right_local[face] >= m.dofs_per_element)
      {
        return DGSEMProductionReason::InvalidFaceRange;
      }
    }
    else if (ownership_value ==
             static_cast<int>(DGSEMDeviceFaceOwnership::Shared))
    {
      if (right_element[face] != -1 || right_local[face] < 0 ||
          right_local[face] >= m.dofs_per_element ||
          m.face_neighbor_vector_size <= 0)
      {
        return DGSEMProductionReason::InvalidFaceRange;
      }
      for (int equation = 0; equation < m.equations; ++equation)
      {
        const int vdof =
          AbsoluteDof(shared_vdof[face + nfaces * equation]);
        if (vdof < 0 || vdof >= m.face_neighbor_vector_size)
        {
          return DGSEMProductionReason::InvalidFaceRange;
        }
      }
    }
    else if (ownership_value ==
             static_cast<int>(DGSEMDeviceFaceOwnership::Boundary))
    {
      if (right_element[face] != -1 || right_local[face] != -1)
      {
        return DGSEMProductionReason::InvalidFaceRange;
      }
    }
    else
    {
      return DGSEMProductionReason::InvalidFaceOwnership;
    }
  }
  return DGSEMProductionReason::None;
}

DGSEMProductionReason ValidateDGSEMProductionState(
  int dimension,
  real_t gamma,
  real_t velocity_density_floor,
  const DenseMatrix& states,
  const Vector& sigma)
{
  if (dimension != 1 && dimension != 2)
  {
    return DGSEMProductionReason::UnsupportedDimension;
  }
  if (!Finite(gamma) || !(gamma > 1.0))
  {
    return DGSEMProductionReason::InvalidGamma;
  }
  if (!Finite(velocity_density_floor) || !(velocity_density_floor > 0.0))
  {
    return DGSEMProductionReason::InvalidDensityFloor;
  }
  if (states.Width() != dimension + 2 || sigma.Size() != states.Height())
  {
    return DGSEMProductionReason::InputSizeMismatch;
  }
  const int rows = states.Height();
  const real_t* state = states.HostRead();
  const real_t* sigma_value = sigma.HostRead();
  for (int row = 0; row < rows; ++row)
  {
    for (int equation = 0; equation < dimension + 2; ++equation)
    {
      if (!Finite(state[row + rows * equation]))
      {
        return equation == 0 ? DGSEMProductionReason::InvalidDensity
                             : DGSEMProductionReason::InvalidPressure;
      }
    }
    if (!Finite(sigma_value[row]))
    {
      return DGSEMProductionReason::InvalidSigma;
    }
    const real_t density = state[row];
    if (!(density > 0.0))
    {
      return DGSEMProductionReason::InvalidDensity;
    }
    if (density < dg_igr::LimiterDefaults::kRhoMin)
    {
      return DGSEMProductionReason::DensityBelowBoundaryFloor;
    }
    real_t momentum_squared = 0.0;
    for (int d = 0; d < dimension; ++d)
    {
      const real_t momentum = state[row + rows * (1 + d)];
      momentum_squared += momentum * momentum;
    }
    const real_t energy = state[row + rows * (1 + dimension)];
    const real_t pressure =
      (gamma - 1.0) * (energy - 0.5 * momentum_squared / density);
    if (!Finite(pressure) || !(pressure > 0.0))
    {
      return DGSEMProductionReason::InvalidPressure;
    }
  }
  return DGSEMProductionReason::None;
}

struct DGSEMProductionCache
{
  DGSEMProductionSelection selection;
  DGSEMProductionCacheCounters counters;
  bool policy_bound = false;
  bool topology_ready = false;
  bool sigma_ready = false;
  bool workspaces_ready = false;
  long vfes_sequence = -1;
  long mesh_sequence = -1;
  const NumericalFlux* numerical_flux = nullptr;
  const FluxFunction* flux_function = nullptr;
  const ParGridFunction* sigma_field = nullptr;
  const FiniteElementSpace* sigma_fes = nullptr;
  long sigma_fes_sequence = -1;
  VolumeMode volume_mode = VolumeMode::WeakDivergence;
  DGSEMStabilization stabilization = DGSEMStabilization::None;
  DGSEMVolumeFlux volume_flux = DGSEMVolumeFlux::Central;
  DGSEMVolumeDissipation volume_dissipation =
    DGSEMVolumeDissipation::None;
  real_t gamma = 1.4;
  real_t velocity_density_floor = dg_igr::FluxDefaults::kRhoVelFloor;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  std::uint64_t policy_fingerprint = 0;
  DGSEMProductionWorkspaceEvidence workspace_evidence;
  int ne = 0;
  int dof = 0;
  int n1d = 0;
  int equations = 0;
  int rows = 0;
  int local_faces = 0;
  int shared_faces = 0;
  int boundary_faces = 0;
  int face_neighbor_vsize = 0;
  int sigma_ldof = 0;

  Array<int> lex_to_local;
  Array<int> element_vdofs;
  DenseMatrix coordinates;
  DenseMatrix metrics;
  DenseMatrix inverse_mass;

  DenseMatrix normals;
  Vector face_weights;
  Array<int> left_elements;
  Array<int> left_local_dofs;
  Array<int> right_elements;
  Array<int> right_local_dofs;
  Array<int> ownership;
  Array<int> shared_state_vdofs;
  Array<int> shared_neighbor_elements;
  DenseMatrix shared_neighbor_points;

  Array<int> sigma_element_dofs;
  DenseMatrix sigma_interpolation;
  Array<int> sigma_shared_neighbor_dofs;
  DenseMatrix sigma_shared_interpolation;

  DenseMatrix states;
  Vector sigma_nodes;
  DenseMatrix contravariant_flux;
  Vector nodal_speeds;
  DenseMatrix weak_volume;
  DenseMatrix left_states;
  DenseMatrix right_states;
  Vector left_sigma;
  Vector right_sigma;
  Vector shared_sigma;
  DenseMatrix face_flux;
  Vector face_speeds;
  DenseMatrix weak_surface;
  DenseMatrix strong;
  Vector nodal_max;
  Vector face_max;
  Vector empty_face_neighbor;
#ifdef MFEM_USE_MPI
  Vector state_send;
  Vector state_face_neighbor;
  Vector sigma_send;
  std::vector<MPI_Request> exchange_requests;
  std::vector<MPI_Status> exchange_statuses;
#endif
};

void AppendVectorWorkspace(const Vector& vector,
                           DGSEMProductionWorkspaceEvidence& evidence)
{
  evidence.size_signature = HashCombine(
    evidence.size_signature,
    static_cast<std::uint64_t>(vector.Size()));
  evidence.capacity_signature = HashCombine(
    evidence.capacity_signature,
    static_cast<std::uint64_t>(vector.Capacity()));
  evidence.storage_signature = HashPointer(evidence.storage_signature,
                                           vector.HostRead());
  evidence.storage_signature = HashPointer(evidence.storage_signature,
                                           vector.Read(true));
}

void AppendMatrixWorkspace(const DenseMatrix& matrix,
                           DGSEMProductionWorkspaceEvidence& evidence)
{
  evidence.size_signature = HashCombine(
    evidence.size_signature,
    static_cast<std::uint64_t>(matrix.Height()));
  evidence.size_signature = HashCombine(
    evidence.size_signature,
    static_cast<std::uint64_t>(matrix.Width()));
  evidence.capacity_signature = HashCombine(
    evidence.capacity_signature,
    static_cast<std::uint64_t>(matrix.GetMemory().Capacity()));
  evidence.storage_signature = HashPointer(evidence.storage_signature,
                                           matrix.HostRead());
  evidence.storage_signature = HashPointer(evidence.storage_signature,
                                           matrix.Read(true));
}

DGSEMProductionWorkspaceEvidence SnapshotWorkspaces(
  const DGSEMProductionCache& cache)
{
  DGSEMProductionWorkspaceEvidence evidence;
  evidence.ready = true;
  evidence.size_signature = UINT64_C(0x431a7d08c2f9e65b);
  evidence.capacity_signature = UINT64_C(0x70f9bd42e5813ac6);
  evidence.storage_signature = UINT64_C(0xa84c629e17d530fb);
  AppendMatrixWorkspace(cache.states, evidence);
  AppendVectorWorkspace(cache.sigma_nodes, evidence);
  AppendMatrixWorkspace(cache.contravariant_flux, evidence);
  AppendVectorWorkspace(cache.nodal_speeds, evidence);
  AppendMatrixWorkspace(cache.weak_volume, evidence);
  AppendMatrixWorkspace(cache.left_states, evidence);
  AppendMatrixWorkspace(cache.right_states, evidence);
  AppendVectorWorkspace(cache.left_sigma, evidence);
  AppendVectorWorkspace(cache.right_sigma, evidence);
  AppendVectorWorkspace(cache.shared_sigma, evidence);
  AppendMatrixWorkspace(cache.face_flux, evidence);
  AppendVectorWorkspace(cache.face_speeds, evidence);
  AppendMatrixWorkspace(cache.weak_surface, evidence);
  AppendMatrixWorkspace(cache.strong, evidence);
  AppendVectorWorkspace(cache.nodal_max, evidence);
  AppendVectorWorkspace(cache.face_max, evidence);
  AppendVectorWorkspace(cache.empty_face_neighbor, evidence);
#ifdef MFEM_USE_MPI
  AppendVectorWorkspace(cache.state_send, evidence);
  AppendVectorWorkspace(cache.state_face_neighbor, evidence);
  AppendVectorWorkspace(cache.sigma_send, evidence);
  evidence.size_signature = HashCombine(
    evidence.size_signature,
    static_cast<std::uint64_t>(cache.exchange_requests.size()));
  evidence.size_signature = HashCombine(
    evidence.size_signature,
    static_cast<std::uint64_t>(cache.exchange_statuses.size()));
  evidence.capacity_signature = HashCombine(
    evidence.capacity_signature,
    static_cast<std::uint64_t>(cache.exchange_requests.capacity()));
  evidence.capacity_signature = HashCombine(
    evidence.capacity_signature,
    static_cast<std::uint64_t>(cache.exchange_statuses.capacity()));
  evidence.storage_signature = HashPointer(evidence.storage_signature,
                                           cache.exchange_requests.data());
  evidence.storage_signature = HashPointer(evidence.storage_signature,
                                           cache.exchange_statuses.data());
  if (cache.sigma_field)
  {
    const auto* sigma =
      dynamic_cast<const ParGridFunction*>(cache.sigma_field);
    if (sigma)
    {
      AppendVectorWorkspace(sigma->FaceNbrData(), evidence);
    }
  }
#endif
  return evidence;
}

bool SameWorkspaceStorage(const DGSEMProductionWorkspaceEvidence& left,
                          const DGSEMProductionWorkspaceEvidence& right)
{
  return left.ready && right.ready &&
         left.size_signature == right.size_signature &&
         left.capacity_signature == right.capacity_signature &&
         left.storage_signature == right.storage_signature;
}

DGSEMOperator::DGSEMOperator(const DGHyperbolicConservationLaws& op)
  : op_(&op)
{
}

DGSEMOperator::~DGSEMOperator() = default;

bool DGSEMOperator::Rebind(const DGHyperbolicConservationLaws& op)
{
  if (op_ == &op)
  {
    return false;
  }
  op_ = &op;
  production_.reset();
  return true;
}

DGSEMProductionCacheCounters DGSEMOperator::GetProductionCacheCounters() const
{
  return production_ ? production_->counters : DGSEMProductionCacheCounters{};
}

DGSEMProductionSelection DGSEMOperator::GetProductionSelection() const
{
  return production_ ? production_->selection : DGSEMProductionSelection{};
}

DGSEMProductionWorkspaceEvidence
DGSEMOperator::GetProductionWorkspaceEvidence() const
{
  return production_ ? production_->workspace_evidence
                     : DGSEMProductionWorkspaceEvidence{};
}

void DGSEMOperator::InvalidateProductionCache(
  DGSEMProductionInvalidation invalidation) const
{
  if (!production_)
  {
    return;
  }
  production_->policy_bound = false;
  if (invalidation >= DGSEMProductionInvalidation::SigmaMetadata)
  {
    production_->sigma_ready = false;
    production_->workspace_evidence = DGSEMProductionWorkspaceEvidence{};
  }
  if (invalidation >= DGSEMProductionInvalidation::Geometry)
  {
    production_->topology_ready = false;
    production_->workspaces_ready = false;
  }
  if (invalidation == DGSEMProductionInvalidation::Topology)
  {
    production_->topology_ready = false;
    production_->sigma_ready = false;
    production_->workspaces_ready = false;
    ++production_->counters.update_invalidations;
  }
}

DGSEMProductionReason DGSEMOperator::EnsureProductionCache_(bool strict) const
{
  if (!production_)
  {
    production_ = std::make_unique<DGSEMProductionCache>();
  }
  DGSEMProductionCache& c = *production_;
  const ProductionCommunicator context =
    GetProductionCommunicator(op_->vfes);
  const bool collective_strict =
    ReduceAny(context, strict, c.counters.selection_reason_reductions);

  Mesh* mesh = op_->vfes.GetMesh();
  const FluxFunction* flux_function =
    op_->formIntegrator ? &op_->formIntegrator->GetFluxFunction() : nullptr;
  const auto* flux_igr =
    flux_function ? dynamic_cast<const FluxWithIGR*>(flux_function) : nullptr;
  const ParGridFunction* sigma_field =
    flux_igr ? flux_igr->GetSigmaField() : nullptr;
  const FiniteElementSpace* sigma_fes =
    sigma_field ? sigma_field->FESpace() : nullptr;
  const real_t gamma = op_->GetDGSEMGamma();
  const real_t velocity_density_floor =
    flux_igr ? flux_igr->GetVelocityRhoFloor()
             : dg_igr::FluxDefaults::kRhoVelFloor;
  const bool sigma_aware_cfl = flux_igr && flux_igr->GetSigmaAwareCFL();
  const bool sigma_aware_rusanov =
    flux_igr && flux_igr->GetSigmaAwareRusanov();
  const bool cuda = Device::Allows(Backend::CUDA_MASK);
  const long vfes_sequence = op_->vfes.GetSequence();
  const long mesh_sequence = mesh ? mesh->GetSequence() : -1;
  const long sigma_sequence = sigma_fes ? sigma_fes->GetSequence() : -1;
  const FluxFunction* numerical_flux_function =
    op_->num_flux_ ? &op_->num_flux_->GetFluxFunction() : nullptr;
  const FluxFunction* igr_base_flux =
    flux_igr ? &flux_igr->GetBaseFlux() : nullptr;

  std::uint64_t fingerprint = UINT64_C(0xd6730f2e456d9a31);
  std::uint64_t semantic = UINT64_C(0x2a891f347bc61d05);
  fingerprint = HashPointer(fingerprint, mesh);
  fingerprint = HashPointer(fingerprint, op_->vfes.FEColl());
  fingerprint = HashPointer(fingerprint, op_->formIntegrator.get());
  fingerprint = HashPointer(fingerprint, flux_function);
  fingerprint = HashPointer(fingerprint, op_->num_flux_);
  fingerprint = HashPointer(fingerprint, numerical_flux_function);
  fingerprint = HashPointer(fingerprint, igr_base_flux);
  fingerprint = HashPointer(fingerprint, sigma_field);
  fingerprint = HashPointer(fingerprint, sigma_fes);
  fingerprint = HashCombine(
    fingerprint, static_cast<std::uint64_t>(vfes_sequence));
  fingerprint = HashCombine(
    fingerprint, static_cast<std::uint64_t>(mesh_sequence));
  fingerprint = HashCombine(
    fingerprint, static_cast<std::uint64_t>(sigma_sequence));
  fingerprint = HashCombine(fingerprint,
                            static_cast<std::uint64_t>(op_->dim));
  fingerprint = HashCombine(fingerprint,
                            static_cast<std::uint64_t>(op_->num_equations));
  fingerprint = HashCombine(fingerprint,
                            static_cast<std::uint64_t>(op_->GetVolumeMode()));
  fingerprint = HashCombine(
    fingerprint,
    static_cast<std::uint64_t>(op_->GetDGSEMStabilization()));
  fingerprint = HashCombine(
    fingerprint,
    static_cast<std::uint64_t>(op_->GetDGSEMVolumeFlux()));
  fingerprint = HashCombine(
    fingerprint,
    static_cast<std::uint64_t>(op_->GetDGSEMVolumeDissipation()));
  fingerprint = HashReal(fingerprint, gamma);
  fingerprint = HashReal(fingerprint, velocity_density_floor);
  fingerprint = HashCombine(fingerprint, sigma_aware_cfl ? 1 : 0);
  fingerprint = HashCombine(fingerprint, sigma_aware_rusanov ? 1 : 0);
  fingerprint = HashCombine(fingerprint, cuda ? 1 : 0);

  semantic = HashCombine(semantic, static_cast<std::uint64_t>(op_->dim));
  semantic = HashCombine(semantic,
                         static_cast<std::uint64_t>(op_->num_equations));
  semantic = HashCombine(semantic,
                         static_cast<std::uint64_t>(op_->GetVolumeMode()));
  semantic = HashCombine(
    semantic,
    static_cast<std::uint64_t>(op_->GetDGSEMStabilization()));
  semantic = HashCombine(
    semantic,
    static_cast<std::uint64_t>(op_->GetDGSEMVolumeFlux()));
  semantic = HashCombine(
    semantic,
    static_cast<std::uint64_t>(op_->GetDGSEMVolumeDissipation()));
  semantic = HashReal(semantic, gamma);
  semantic = HashReal(semantic, velocity_density_floor);
  semantic = HashCombine(semantic, sigma_field ? 1 : 0);
  semantic = HashCombine(semantic,
                         numerical_flux_function == flux_function ? 1 : 0);
  semantic = HashCombine(
    semantic,
    flux_igr && igr_base_flux ? 1 : 0);
  semantic = HashCombine(semantic, sigma_aware_cfl ? 1 : 0);
  semantic = HashCombine(semantic, sigma_aware_rusanov ? 1 : 0);
  semantic = HashCombine(semantic, cuda ? 1 : 0);

  int surface_type = -1;
  if (const auto* dgsem_flux =
        dynamic_cast<const DGSEMSurfaceFlux*>(op_->num_flux_))
  {
    surface_type = static_cast<int>(dgsem_flux->GetSurfaceFluxType());
  }
  else if (dynamic_cast<const RusanovFluxWithSides*>(op_->num_flux_))
  {
    surface_type = static_cast<int>(DGSEMSurfaceFluxType::Rusanov);
  }
  fingerprint = HashCombine(
    fingerprint,
    static_cast<std::uint64_t>(static_cast<std::int64_t>(surface_type)));
  semantic = HashCombine(
    semantic,
    static_cast<std::uint64_t>(static_cast<std::int64_t>(surface_type)));

  fingerprint = HashCombine(fingerprint, op_->ext_bdr_int_.size());
  semantic = HashCombine(semantic, op_->ext_bdr_int_.size());
  for (std::size_t i = 0; i < op_->ext_bdr_int_.size(); ++i)
  {
    const NonlinearFormIntegrator* generic = op_->ext_bdr_int_[i].get();
    const Array<int>* marker = i < op_->ext_bdr_markers_.size()
                                 ? op_->ext_bdr_markers_[i].get()
                                 : nullptr;
    const auto* boundary =
      dynamic_cast<const BdrHyperbolicFluxIntegrator*>(generic);
    const BoundaryState* boundary_state =
      boundary ? &boundary->GetBoundaryState() : nullptr;
    const auto* wall = boundary
                         ? dynamic_cast<const SlipWallState*>(
                             boundary_state)
                         : nullptr;
    const NumericalFlux* boundary_numerical_flux =
      boundary ? &boundary->GetNumericalFlux() : nullptr;
    const FluxFunction* boundary_physical_flux =
      boundary_numerical_flux
        ? &boundary_numerical_flux->GetFluxFunction()
        : nullptr;
    fingerprint = HashPointer(fingerprint, generic);
    fingerprint = HashPointer(fingerprint, marker);
    fingerprint = HashPointer(fingerprint, boundary);
    fingerprint = HashPointer(fingerprint, boundary_state);
    fingerprint = HashPointer(fingerprint, wall);
    fingerprint = HashPointer(fingerprint, boundary_numerical_flux);
    fingerprint = HashPointer(fingerprint, boundary_physical_flux);
    fingerprint = HashCombine(fingerprint,
                              marker ? marker->Size() : UINT64_C(0));
    fingerprint = HashCombine(
      fingerprint,
      static_cast<std::uint64_t>(wall ? wall->GetDimension() : -1));
    semantic = HashCombine(semantic, boundary ? 1 : 0);
    semantic = HashCombine(
      semantic,
      static_cast<std::uint64_t>(wall ? wall->GetDimension() : -1));
    semantic = HashCombine(
      semantic,
      boundary_numerical_flux == op_->num_flux_ ? 1 : 0);
    semantic = HashCombine(
      semantic,
      boundary_physical_flux == flux_function ? 1 : 0);
    semantic = HashCombine(semantic, marker ? marker->Size() : 0);
    if (marker)
    {
      for (int attr = 0; attr < marker->Size(); ++attr)
      {
        fingerprint = HashCombine(fingerprint,
                                  static_cast<std::uint64_t>((*marker)[attr]));
        semantic = HashCombine(semantic,
                               static_cast<std::uint64_t>((*marker)[attr]));
      }
    }
  }

  const bool semantic_equal = ReduceEqual(
    context, semantic, c.counters.selection_reason_reductions);
  const bool rebuild_preflight = ReduceAny(
    context,
    !c.policy_bound || c.policy_fingerprint != fingerprint,
    c.counters.selection_reason_reductions);

  auto set_collective_selection = [&](DGSEMProductionReason reason)
  {
    if (rebuild_preflight)
    {
      ++c.counters.policy_binds;
    }
    c.policy_bound = true;
    c.policy_fingerprint = fingerprint;
    c.selection = DGSEMProductionSelection{};
    c.selection.supported = reason == DGSEMProductionReason::None;
    c.selection.reason = reason;
    c.selection.strict = collective_strict;
    c.selection.cuda = cuda;
    c.selection.uses_sigma = sigma_field != nullptr;
    c.selection.dimension = op_->dim;
    c.selection.elements = op_->vfes.GetNE();
    c.selection.ranks = context.size;
  };

  if (!semantic_equal)
  {
    set_collective_selection(
      DGSEMProductionReason::CollectivePolicyMismatch);
    return DGSEMProductionReason::CollectivePolicyMismatch;
  }
  if (!rebuild_preflight)
  {
    ++c.counters.unchanged_cache_fast_paths;
    c.selection.strict = collective_strict;
    c.selection.ranks = context.size;
    return c.selection.reason;
  }
  ++c.counters.rebuild_preflight_entries;

  DGSEMProductionReason policy_reason = DGSEMProductionReason::None;
  if (!mesh || (op_->dim != 1 && op_->dim != 2))
  {
    policy_reason = DGSEMProductionReason::UnsupportedDimension;
  }
  else if (op_->num_equations != op_->dim + 2)
  {
    policy_reason = DGSEMProductionReason::UnsupportedEquationCount;
  }
  else if (!op_->formIntegrator || !op_->num_flux_)
  {
    policy_reason = DGSEMProductionReason::UnsupportedSurfaceFlux;
  }
  else if (op_->GetVolumeMode() != VolumeMode::DGSEMFluxDiff ||
           op_->GetDGSEMVolumeFlux() != DGSEMVolumeFlux::Central)
  {
    policy_reason = DGSEMProductionReason::UnsupportedVolumeFlux;
  }
  else if (op_->GetDGSEMVolumeDissipation() !=
           DGSEMVolumeDissipation::None)
  {
    policy_reason = DGSEMProductionReason::UnsupportedVolumeDissipation;
  }
  else if (op_->GetDGSEMStabilization() != DGSEMStabilization::None)
  {
    policy_reason = DGSEMProductionReason::UnsupportedStabilization;
  }
  else if (!Finite(gamma) || !(gamma > 1.0))
  {
    policy_reason = DGSEMProductionReason::InvalidGamma;
  }
  else if (!Finite(velocity_density_floor) ||
           !(velocity_density_floor > 0.0))
  {
    policy_reason = DGSEMProductionReason::InvalidDensityFloor;
  }
  else if (&op_->num_flux_->GetFluxFunction() != flux_function)
  {
    policy_reason = DGSEMProductionReason::FluxIdentityMismatch;
  }

  const EulerFlux* euler_model = nullptr;
  if (policy_reason == DGSEMProductionReason::None)
  {
    euler_model = dynamic_cast<const EulerFlux*>(flux_function);
    if (!euler_model && flux_igr)
    {
      euler_model = dynamic_cast<const EulerFlux*>(&flux_igr->GetBaseFlux());
    }
    if (!euler_model)
    {
      policy_reason = DGSEMProductionReason::UnsupportedModel;
    }
  }

  if (policy_reason == DGSEMProductionReason::None)
  {
    ElementTransformation* transformation =
      op_->vfes.GetNE() > 0 ? op_->vfes.GetElementTransformation(0) : nullptr;
    if (transformation)
    {
      IntegrationPoint point;
      point.x = 0.5;
      point.y = 0.5;
      transformation->SetIntPoint(&point);
      Vector probe(op_->num_equations);
      probe = 0.0;
      probe(0) = 1.0;
      probe(op_->dim + 1) = 1.0 / (gamma - 1.0);
      DenseMatrix physical_flux(op_->num_equations, op_->dim);
      euler_model->ComputeFlux(probe, *transformation, physical_flux);
      const real_t tolerance =
        4096.0 * std::numeric_limits<real_t>::epsilon();
      bool gamma_matches =
        physical_flux.Height() == op_->num_equations &&
        physical_flux.Width() == op_->dim && physical_flux.CheckFinite() == 0;
      for (int d = 0; gamma_matches && d < op_->dim; ++d)
      {
        gamma_matches = std::abs(physical_flux(1 + d, d) - 1.0) <=
                        tolerance;
      }
      if (!gamma_matches)
      {
        policy_reason = DGSEMProductionReason::EulerGammaMismatch;
      }
    }
  }

  if (policy_reason == DGSEMProductionReason::None &&
      (sigma_aware_cfl || sigma_aware_rusanov))
  {
    policy_reason = DGSEMProductionReason::UnsupportedSigmaSpeedPolicy;
  }
  if (policy_reason == DGSEMProductionReason::None &&
      surface_type != static_cast<int>(DGSEMSurfaceFluxType::Rusanov))
  {
    policy_reason = DGSEMProductionReason::UnsupportedSurfaceFlux;
  }
  if (policy_reason == DGSEMProductionReason::None && cuda &&
      context.size != 1)
  {
    policy_reason = DGSEMProductionReason::MultiRankCuda;
  }
  if (policy_reason == DGSEMProductionReason::None && sigma_field &&
      (!sigma_fes || sigma_fes->GetMesh() != mesh ||
       sigma_fes->GetNE() != op_->vfes.GetNE() || sigma_fes->GetVDim() != 1))
  {
    policy_reason = DGSEMProductionReason::UnsupportedSigma;
  }
#ifdef MFEM_USE_MPI
  if (policy_reason == DGSEMProductionReason::None && sigma_field)
  {
    const auto* state_pfes =
      dynamic_cast<const ParFiniteElementSpace*>(&op_->vfes);
    const auto* scalar_pfes =
      dynamic_cast<const ParFiniteElementSpace*>(sigma_fes);
    if (state_pfes && !scalar_pfes)
    {
      policy_reason = DGSEMProductionReason::UnsupportedSigma;
    }
    else if (state_pfes && scalar_pfes)
    {
      int comparison = MPI_UNEQUAL;
      MPI_Comm_compare(state_pfes->GetComm(),
                       scalar_pfes->GetComm(),
                       &comparison);
      if (comparison != MPI_IDENT && comparison != MPI_CONGRUENT)
      {
        policy_reason = DGSEMProductionReason::UnsupportedSigma;
      }
    }
  }
#endif

  if (policy_reason == DGSEMProductionReason::None)
  {
    const int max_attr = mesh->bdr_attributes.Size() > 0
                           ? mesh->bdr_attributes.Max()
                           : 0;
    std::vector<char> active(static_cast<std::size_t>(max_attr + 1), 0);
    for (int b = 0; b < mesh->GetNBE(); ++b)
    {
      FaceElementTransformations* transformation =
        mesh->GetBdrFaceTransformations(b);
      if (transformation && transformation->Attribute > 0 &&
          transformation->Attribute <= max_attr)
      {
        active[static_cast<std::size_t>(transformation->Attribute)] = 1;
      }
    }
    for (int attr = 1;
         policy_reason == DGSEMProductionReason::None && attr <= max_attr;
         ++attr)
    {
      if (!active[static_cast<std::size_t>(attr)])
      {
        continue;
      }
      int applicable = 0;
      const BdrHyperbolicFluxIntegrator* selected = nullptr;
      for (std::size_t i = 0; i < op_->ext_bdr_int_.size(); ++i)
      {
        const Array<int>* marker = i < op_->ext_bdr_markers_.size()
                                     ? op_->ext_bdr_markers_[i].get()
                                     : nullptr;
        if (!marker || attr > marker->Size() || (*marker)[attr - 1] == 0)
        {
          continue;
        }
        ++applicable;
        selected = dynamic_cast<const BdrHyperbolicFluxIntegrator*>(
          op_->ext_bdr_int_[i].get());
      }
      if (applicable == 0 || !selected)
      {
        policy_reason = DGSEMProductionReason::UnsupportedBoundary;
      }
      else if (applicable != 1)
      {
        policy_reason = DGSEMProductionReason::BoundaryOverlap;
      }
      else
      {
        const auto* wall = dynamic_cast<const SlipWallState*>(
          &selected->GetBoundaryState());
        if (!wall)
        {
          policy_reason = DGSEMProductionReason::UnsupportedBoundary;
        }
        else if (wall->GetDimension() != op_->dim)
        {
          policy_reason = DGSEMProductionReason::BoundaryDimensionMismatch;
        }
        else if (&selected->GetNumericalFlux() != op_->num_flux_ ||
                 &selected->GetNumericalFlux().GetFluxFunction() !=
                   flux_function)
        {
          policy_reason = DGSEMProductionReason::BoundaryFluxMismatch;
        }
      }
    }
  }

  policy_reason = ReduceReason(context,
                               policy_reason,
                               c.counters.selection_reason_reductions);
  if (policy_reason != DGSEMProductionReason::None)
  {
    set_collective_selection(policy_reason);
    return policy_reason;
  }

  DGSEMProductionReason topology_reason = DGSEMProductionReason::None;
  int topology_order = -1;
  std::uint64_t topology_semantic = UINT64_C(0xe13c98f04a2576bd);
  if (op_->vfes.GetNE() <= 0 ||
      op_->vfes.GetOrdering() != Ordering::byNODES ||
      !op_->vfes.IsDGSpace())
  {
    topology_reason = DGSEMProductionReason::UnsupportedTensorElement;
  }
  const auto* l2_collection =
    dynamic_cast<const L2_FECollection*>(op_->vfes.FEColl());
  if (topology_reason == DGSEMProductionReason::None &&
      (!l2_collection ||
       l2_collection->GetBasisType() != BasisType::GaussLobatto))
  {
    topology_reason = DGSEMProductionReason::UnsupportedNodalBasis;
  }

  std::vector<int> reference_lex_to_local;
  if (topology_reason == DGSEMProductionReason::None)
  {
    topology_order = op_->vfes.GetFE(0)->GetOrder();
    if (topology_order < 1 || topology_order > 4)
    {
      topology_reason = DGSEMProductionReason::UnsupportedOrder;
    }
  }
  const int n1d = topology_order + 1;
  const int expected_dof = op_->dim == 1 ? n1d : n1d * n1d;
  const Geometry::Type expected_geometry =
    op_->dim == 1 ? Geometry::SEGMENT : Geometry::SQUARE;
  if (topology_reason == DGSEMProductionReason::None)
  {
    const LGLBasis1D& basis = GetLGLBasis1D(topology_order);
    const real_t tolerance =
      4096.0 * std::numeric_limits<real_t>::epsilon();
    for (int e = 0;
         topology_reason == DGSEMProductionReason::None &&
         e < op_->vfes.GetNE();
         ++e)
    {
      const FiniteElement& fe = *op_->vfes.GetFE(e);
      const auto* tensor = dynamic_cast<const TensorBasisElement*>(&fe);
      if (!tensor || fe.GetDim() != op_->dim ||
          fe.GetGeomType() != expected_geometry ||
          fe.GetOrder() != topology_order || fe.GetDof() != expected_dof)
      {
        topology_reason = fe.GetOrder() != topology_order
                            ? DGSEMProductionReason::MixedOrder
                            : DGSEMProductionReason::UnsupportedTensorElement;
        break;
      }
      if (tensor->GetBasisType() != BasisType::GaussLobatto ||
          fe.GetMapType() != FiniteElement::VALUE ||
          fe.GetNodes().GetNPoints() != expected_dof)
      {
        topology_reason = DGSEMProductionReason::UnsupportedNodalBasis;
        break;
      }
      std::vector<int> lex_to_local(static_cast<std::size_t>(expected_dof));
      const Array<int>& map = tensor->GetDofMap();
      std::vector<char> seen(static_cast<std::size_t>(expected_dof), 0);
      for (int lex = 0; lex < expected_dof; ++lex)
      {
        const int local = map.Size() == expected_dof ? map[lex] : lex;
        if (local < 0 || local >= expected_dof ||
            seen[static_cast<std::size_t>(local)])
        {
          topology_reason = DGSEMProductionReason::InvalidLexicographicMap;
          break;
        }
        seen[static_cast<std::size_t>(local)] = 1;
        lex_to_local[static_cast<std::size_t>(lex)] = local;
        const IntegrationPoint& point = fe.GetNodes().IntPoint(local);
        const int i = lex % n1d;
        const int j = op_->dim == 2 ? lex / n1d : 0;
        if (std::abs(point.x - basis.nodes(i)) > tolerance ||
            (op_->dim == 2 &&
             std::abs(point.y - basis.nodes(j)) > tolerance))
        {
          topology_reason = DGSEMProductionReason::UnsupportedNodalBasis;
          break;
        }
      }
      if (e == 0)
      {
        reference_lex_to_local = lex_to_local;
      }
      else if (topology_reason == DGSEMProductionReason::None &&
               lex_to_local != reference_lex_to_local)
      {
        topology_reason = DGSEMProductionReason::InvalidLexicographicMap;
      }
    }
  }

  topology_semantic = HashCombine(
    topology_semantic,
    static_cast<std::uint64_t>(op_->dim));
  topology_semantic = HashCombine(
    topology_semantic,
    static_cast<std::uint64_t>(topology_order));
  topology_semantic = HashCombine(
    topology_semantic,
    static_cast<std::uint64_t>(expected_dof));
  topology_reason = ReduceReason(context,
                                 topology_reason,
                                 c.counters.selection_reason_reductions);
  if (topology_reason == DGSEMProductionReason::None &&
      !ReduceEqual(context,
                   topology_semantic,
                   c.counters.selection_reason_reductions))
  {
    topology_reason = DGSEMProductionReason::MixedOrder;
  }
  if (topology_reason != DGSEMProductionReason::None)
  {
    set_collective_selection(topology_reason);
    return topology_reason;
  }

  const bool local_topology_rebuild =
    !c.topology_ready || c.vfes_sequence != vfes_sequence ||
    c.mesh_sequence != mesh_sequence;
  const bool topology_rebuild = ReduceAny(
    context,
    local_topology_rebuild,
    c.counters.selection_reason_reductions);
  if (topology_rebuild)
  {
    c.topology_ready = false;
    c.sigma_ready = false;
    c.workspaces_ready = false;
#ifdef MFEM_USE_MPI
    if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&op_->vfes))
    {
      pfes->ExchangeFaceNbrData();
    }
#endif
  }

  const bool local_sigma_rebuild =
    topology_rebuild || !c.sigma_ready || c.sigma_field != sigma_field ||
    c.sigma_fes != sigma_fes || c.sigma_fes_sequence != sigma_sequence;
  const bool sigma_rebuild = ReduceAny(context,
                                       local_sigma_rebuild,
                                       c.counters.selection_reason_reductions);
  if (sigma_rebuild)
  {
    c.sigma_ready = false;
    c.workspace_evidence = DGSEMProductionWorkspaceEvidence{};
#ifdef MFEM_USE_MPI
    if (auto* scalar_pfes =
          dynamic_cast<ParFiniteElementSpace*>(
            const_cast<FiniteElementSpace*>(sigma_fes)))
    {
      scalar_pfes->ExchangeFaceNbrData();
    }
#endif
  }

  const DGSEMProductionReason local_build_reason =
    BuildProductionCacheLocal_(collective_strict,
                               context.size,
                               fingerprint);
  const DGSEMProductionReason build_reason = ReduceReason(
    context,
    local_build_reason,
    c.counters.selection_reason_reductions);
  if (build_reason != DGSEMProductionReason::None)
  {
    set_collective_selection(build_reason);
    return build_reason;
  }

#ifdef MFEM_USE_MPI
  if (context.active && context.size > 1)
  {
    int local_elements = c.ne;
    int global_elements = local_elements;
    int local_faces[3] = {c.local_faces > 0 ? 1 : 0,
                          c.shared_faces > 0 ? 1 : 0,
                          c.boundary_faces > 0 ? 1 : 0};
    int global_faces[3] = {0, 0, 0};
    MPI_Allreduce(&local_elements,
                  &global_elements,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  context.comm);
    MPI_Allreduce(local_faces,
                  global_faces,
                  3,
                  MPI_INT,
                  MPI_MAX,
                  context.comm);
    c.counters.selection_reason_reductions += 2;
    c.selection.elements = global_elements;
    c.selection.has_local_faces = global_faces[0] != 0;
    c.selection.has_shared_faces = global_faces[1] != 0;
    c.selection.has_boundary_faces = global_faces[2] != 0;
  }
#endif
  c.selection.strict = collective_strict;
  c.selection.ranks = context.size;
  return DGSEMProductionReason::None;
}

DGSEMProductionReason DGSEMOperator::BuildProductionCacheLocal_(
  bool strict,
  int ranks,
  std::uint64_t fingerprint) const
{
  if (!production_)
  {
    production_ = std::make_unique<DGSEMProductionCache>();
  }
  DGSEMProductionCache& c = *production_;
  c.selection.strict = strict;

  Mesh* mesh = op_->vfes.GetMesh();
  const FluxFunction& flux_function = op_->formIntegrator->GetFluxFunction();
  const auto* flux_igr = dynamic_cast<const FluxWithIGR*>(&flux_function);
  const ParGridFunction* sigma_field =
    flux_igr ? flux_igr->GetSigmaField() : nullptr;
  const real_t velocity_density_floor =
    flux_igr ? flux_igr->GetVelocityRhoFloor()
             : dg_igr::FluxDefaults::kRhoVelFloor;
  const bool cuda = Device::Allows(Backend::CUDA_MASK);
  const long vfes_sequence = op_->vfes.GetSequence();
  const long mesh_sequence = mesh ? mesh->GetSequence() : -1;
  const FiniteElementSpace* sigma_fes =
    sigma_field ? sigma_field->FESpace() : nullptr;
  const long sigma_sequence = sigma_fes ? sigma_fes->GetSequence() : -1;
  const bool sigma_aware_cfl = flux_igr && flux_igr->GetSigmaAwareCFL();
  const bool sigma_aware_rusanov =
    flux_igr && flux_igr->GetSigmaAwareRusanov();

  const bool sigma_metadata_changed =
    c.sigma_field != sigma_field || c.sigma_fes != sigma_fes ||
    c.sigma_fes_sequence != sigma_sequence;
  const bool policy_changed =
    !c.policy_bound || c.policy_fingerprint != fingerprint ||
    c.numerical_flux != op_->num_flux_ ||
    c.flux_function != &flux_function || sigma_metadata_changed ||
    c.volume_mode != op_->GetVolumeMode() ||
    c.stabilization != op_->GetDGSEMStabilization() ||
    c.volume_flux != op_->GetDGSEMVolumeFlux() ||
    c.volume_dissipation != op_->GetDGSEMVolumeDissipation() ||
    c.gamma != op_->GetDGSEMGamma() ||
    c.velocity_density_floor != velocity_density_floor ||
    c.sigma_aware_cfl != sigma_aware_cfl ||
    c.sigma_aware_rusanov != sigma_aware_rusanov;

  if (c.topology_ready &&
      (c.vfes_sequence != vfes_sequence || c.mesh_sequence != mesh_sequence))
  {
    c.topology_ready = false;
    c.sigma_ready = false;
    c.workspaces_ready = false;
  }
  else if (sigma_metadata_changed)
  {
    c.sigma_ready = false;
  }

  if (!policy_changed && c.topology_ready && c.sigma_ready &&
      c.workspaces_ready && c.selection.supported)
  {
    return DGSEMProductionReason::None;
  }

  if (policy_changed)
  {
    ++c.counters.policy_binds;
  }
  c.policy_bound = true;
  c.policy_fingerprint = fingerprint;
  c.numerical_flux = op_->num_flux_;
  c.flux_function = &flux_function;
  c.sigma_field = sigma_field;
  c.sigma_fes = sigma_fes;
  c.sigma_fes_sequence = sigma_sequence;
  c.volume_mode = op_->GetVolumeMode();
  c.stabilization = op_->GetDGSEMStabilization();
  c.volume_flux = op_->GetDGSEMVolumeFlux();
  c.volume_dissipation = op_->GetDGSEMVolumeDissipation();
  c.gamma = op_->GetDGSEMGamma();
  c.velocity_density_floor = velocity_density_floor;
  c.sigma_aware_cfl = sigma_aware_cfl;
  c.sigma_aware_rusanov = sigma_aware_rusanov;

  c.selection = DGSEMProductionSelection{};
  c.selection.strict = strict;
  c.selection.cuda = cuda;
  c.selection.uses_sigma = sigma_field != nullptr;
  c.selection.dimension = op_->dim;
  c.selection.elements = op_->vfes.GetNE();
  c.selection.ranks = ranks;

  auto fail = [&](DGSEMProductionReason reason)
  {
    c.selection.supported = false;
    c.selection.reason = reason;
    return reason;
  };

  if (!mesh || (op_->dim != 1 && op_->dim != 2))
  {
    return fail(DGSEMProductionReason::UnsupportedDimension);
  }
  if (op_->num_equations != op_->dim + 2)
  {
    return fail(DGSEMProductionReason::UnsupportedEquationCount);
  }
  if (op_->GetVolumeMode() != VolumeMode::DGSEMFluxDiff)
  {
    return fail(DGSEMProductionReason::UnsupportedVolumeFlux);
  }
  if (op_->GetDGSEMStabilization() != DGSEMStabilization::None)
  {
    return fail(DGSEMProductionReason::UnsupportedStabilization);
  }
  if (op_->GetDGSEMVolumeFlux() != DGSEMVolumeFlux::Central)
  {
    return fail(DGSEMProductionReason::UnsupportedVolumeFlux);
  }
  if (op_->GetDGSEMVolumeDissipation() != DGSEMVolumeDissipation::None)
  {
    return fail(DGSEMProductionReason::UnsupportedVolumeDissipation);
  }
  if (!Finite(c.gamma) || !(c.gamma > 1.0))
  {
    return fail(DGSEMProductionReason::InvalidGamma);
  }
  if (!Finite(c.velocity_density_floor) ||
      !(c.velocity_density_floor > 0.0))
  {
    return fail(DGSEMProductionReason::InvalidDensityFloor);
  }
  const bool euler_model =
    dynamic_cast<const EulerFlux*>(&flux_function) != nullptr ||
    (flux_igr && flux_igr->HasEulerBase());
  if (!euler_model)
  {
    return fail(DGSEMProductionReason::UnsupportedModel);
  }
  if (sigma_aware_cfl || sigma_aware_rusanov)
  {
    return fail(DGSEMProductionReason::UnsupportedSigmaSpeedPolicy);
  }
  bool surface_supported =
    dynamic_cast<const RusanovFluxWithSides*>(op_->num_flux_) != nullptr;
  if (const auto* dgsem_flux =
        dynamic_cast<const DGSEMSurfaceFlux*>(op_->num_flux_))
  {
    surface_supported =
      dgsem_flux->GetSurfaceFluxType() == DGSEMSurfaceFluxType::Rusanov;
  }
  if (!surface_supported)
  {
    return fail(DGSEMProductionReason::UnsupportedSurfaceFlux);
  }
  if (cuda && ranks != 1)
  {
    return fail(DGSEMProductionReason::MultiRankCuda);
  }
  for (int b = 0; b < mesh->GetNBE(); ++b)
  {
    FaceElementTransformations* tr = mesh->GetBdrFaceTransformations(b);
    if (!tr)
    {
      continue;
    }
    const BdrHyperbolicFluxIntegrator* integrator =
      FindBdrIntegrator(tr->Attribute);
    if (!integrator ||
        !dynamic_cast<const SlipWallState*>(&integrator->GetBoundaryState()))
    {
      return fail(DGSEMProductionReason::UnsupportedBoundary);
    }
  }

  if (!c.topology_ready)
  {
    c.ne = op_->vfes.GetNE();
    c.equations = op_->num_equations;
    if (c.ne <= 0 || op_->vfes.GetOrdering() != Ordering::byNODES)
    {
      return fail(DGSEMProductionReason::UnsupportedTensorElement);
    }
    const FiniteElement& first_fe = *op_->vfes.GetFE(0);
    const int order = first_fe.GetOrder();
    if (order < 1 || order > 4)
    {
      return fail(DGSEMProductionReason::UnsupportedOrder);
    }
    c.selection.order = order;
    c.n1d = order + 1;
    c.dof = first_fe.GetDof();
    c.rows = c.ne * c.dof;
    const int expected_dof = op_->dim == 1 ? c.n1d : c.n1d * c.n1d;
    if (c.dof != expected_dof)
    {
      return fail(DGSEMProductionReason::UnsupportedTensorElement);
    }

    c.lex_to_local.SetSize(c.dof);
    c.lex_to_local = -1;
    const auto& first_ijk = GetTensorDofToIJK(first_fe);
    for (int local = 0; local < c.dof; ++local)
    {
      const int lex = op_->dim == 1
                        ? first_ijk[static_cast<std::size_t>(local)][0]
                        : first_ijk[static_cast<std::size_t>(local)][0] +
                            c.n1d *
                              first_ijk[static_cast<std::size_t>(local)][1];
      if (lex < 0 || lex >= c.dof || c.lex_to_local[lex] != -1)
      {
        return fail(DGSEMProductionReason::InvalidLexicographicMap);
      }
      c.lex_to_local[lex] = local;
    }

    c.element_vdofs.SetSize(c.rows * c.equations);
    c.inverse_mass.SetSize(c.rows, c.dof);
    c.coordinates.SetSize(c.rows, op_->dim);
    Array<int> vdofs;
    Vector physical(op_->dim);
    for (int e = 0; e < c.ne; ++e)
    {
      const FiniteElement& fe = *op_->vfes.GetFE(e);
      if (fe.GetOrder() != order)
      {
        return fail(DGSEMProductionReason::MixedOrder);
      }
      if (fe.GetDof() != c.dof)
      {
        return fail(DGSEMProductionReason::UnsupportedTensorElement);
      }
      Array<int> element_lex(c.dof);
      element_lex = -1;
      const auto& ijk = GetTensorDofToIJK(fe);
      for (int local = 0; local < c.dof; ++local)
      {
        const int lex = op_->dim == 1
                          ? ijk[static_cast<std::size_t>(local)][0]
                          : ijk[static_cast<std::size_t>(local)][0] +
                              c.n1d * ijk[static_cast<std::size_t>(local)][1];
        if (lex < 0 || lex >= c.dof || element_lex[lex] != -1)
        {
          return fail(DGSEMProductionReason::InvalidLexicographicMap);
        }
        element_lex[lex] = local;
      }
      for (int lex = 0; lex < c.dof; ++lex)
      {
        if (element_lex[lex] != c.lex_to_local[lex])
        {
          return fail(DGSEMProductionReason::InvalidLexicographicMap);
        }
      }

      op_->vfes.GetElementVDofs(e, vdofs);
      if (vdofs.Size() != c.dof * c.equations)
      {
        return fail(DGSEMProductionReason::InvalidElementVdofMap);
      }
      for (int equation = 0; equation < c.equations; ++equation)
      {
        for (int local = 0; local < c.dof; ++local)
        {
          c.element_vdofs[e * c.dof + local + c.rows * equation] =
            vdofs[local + c.dof * equation];
        }
      }
      if (e >= static_cast<int>(op_->invmass.size()) ||
          op_->invmass[static_cast<std::size_t>(e)].Height() != c.dof ||
          op_->invmass[static_cast<std::size_t>(e)].Width() != c.dof)
      {
        return fail(DGSEMProductionReason::InvalidMetric);
      }
      for (int i = 0; i < c.dof; ++i)
      {
        for (int j = 0; j < c.dof; ++j)
        {
          c.inverse_mass(e * c.dof + i, j) =
            op_->invmass[static_cast<std::size_t>(e)](i, j);
        }
      }
      ElementTransformation* transformation =
        op_->vfes.GetElementTransformation(e);
      if (!transformation || fe.GetNodes().GetNPoints() != c.dof)
      {
        return fail(DGSEMProductionReason::InvalidMetric);
      }
      for (int local = 0; local < c.dof; ++local)
      {
        transformation->Transform(fe.GetNodes().IntPoint(local), physical);
        for (int d = 0; d < op_->dim; ++d)
        {
          c.coordinates(e * c.dof + local, d) = physical(d);
        }
      }
    }

    if (op_->dim == 1)
    {
      c.metrics.SetSize(c.rows, 1);
      c.metrics = 1.0;
    }
    else
    {
      BuildContravariantMetrics2DDevice(c.ne,
                                         c.n1d,
                                         c.lex_to_local,
                                         GetLGLBasis1D(order).D,
                                         c.coordinates,
                                         c.metrics);
      ++c.counters.host_synchronizations;
      c.metrics.HostRead();
    }
    ++c.counters.geometry_builds;

    std::vector<std::array<real_t, 2>> face_normals;
    std::vector<real_t> face_weights;
    std::vector<int> left_elements;
    std::vector<int> left_local;
    std::vector<int> right_elements;
    std::vector<int> right_local;
    std::vector<int> ownership;
    std::vector<std::array<int, 4>> neighbor_state_maps;
    std::vector<int> neighbor_elements;
    std::vector<std::array<real_t, 2>> neighbor_points;
    c.local_faces = 0;
    c.shared_faces = 0;
    c.boundary_faces = 0;

#ifdef MFEM_USE_MPI
    auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&op_->vfes);
    ParMesh* pmesh = pfes ? pfes->GetParMesh() : nullptr;
    std::vector<char> is_shared_face;
    if (pmesh && pmesh->GetNSharedFaces() > 0)
    {
      is_shared_face.assign(static_cast<std::size_t>(mesh->GetNumFaces()), 0);
      for (int sf = 0; sf < pmesh->GetNSharedFaces(); ++sf)
      {
        const int face = pmesh->GetSharedFace(sf);
        if (face >= 0 && face < mesh->GetNumFaces())
        {
          is_shared_face[static_cast<std::size_t>(face)] = 1;
        }
      }
      c.face_neighbor_vsize = pfes->GetFaceNbrVSize();
    }
    else
    {
      c.face_neighbor_vsize = 0;
    }
#else
    std::vector<char> is_shared_face;
    c.face_neighbor_vsize = 0;
#endif

    auto local_from_point = [&](const FiniteElement& fe,
                                const IntegrationPoint& point) -> int
    {
      if (fe.GetOrder() != order || fe.GetDof() != c.dof)
      {
        return -1;
      }
      const Vector& nodes = GetLGLBasis1D(order).nodes;
      const int i = FindNode(nodes, point.x);
      const int j = op_->dim == 2 ? FindNode(nodes, point.y) : 0;
      if (i < 0 || j < 0)
      {
        return -1;
      }
      return c.lex_to_local[op_->dim == 1 ? i : i + c.n1d * j];
    };
    auto normal_from_point = [&](FaceElementTransformations& transformation,
                                 int element,
                                 int local,
                                 const IntegrationPoint& point,
                                 std::array<real_t, 2>& normal) -> bool
    {
      normal = {0.0, 0.0};
      if (op_->dim == 1)
      {
        normal[0] = (point.x - 0.5) * 2.0;
        return true;
      }
      ContravariantMetric2D metric;
      const int row = element * c.dof + local;
      for (int entry = 0; entry < 4; ++entry)
      {
        metric.data[static_cast<std::size_t>(entry)] = c.metrics(row, entry);
      }
      Vector packed_normal(2);
      BuildNodalFaceNormal2D(metric,
                             transformation.Loc1.Transf.Jacobian(),
                             point,
                             packed_normal);
      normal[0] = packed_normal(0);
      normal[1] = packed_normal(1);
      return true;
    };
    auto append_face = [&](int element_l,
                           int local_l,
                           int element_r,
                           int local_r_value,
                           DGSEMDeviceFaceOwnership owner,
                           const std::array<real_t, 2>& normal,
                           real_t weight,
                           const std::array<int, 4>& neighbor_map,
                           int neighbor_element,
                           const IntegrationPoint* neighbor_point)
    {
      left_elements.push_back(element_l);
      left_local.push_back(local_l);
      right_elements.push_back(element_r);
      right_local.push_back(local_r_value);
      ownership.push_back(static_cast<int>(owner));
      face_normals.push_back(normal);
      face_weights.push_back(weight);
      neighbor_state_maps.push_back(neighbor_map);
      neighbor_elements.push_back(neighbor_element);
      neighbor_points.push_back(
        neighbor_point
          ? std::array<real_t, 2>{{neighbor_point->x, neighbor_point->y}}
          : std::array<real_t, 2>{{0.0, 0.0}});
    };

    const IntegrationRule& face_rule = op_->dim == 1
                                         ? IntRules.Get(Geometry::POINT, 1)
                                         : GetLGLFaceRule(order);
    for (int f = 0; f < mesh->GetNumFaces(); ++f)
    {
      if (!is_shared_face.empty() && is_shared_face[static_cast<std::size_t>(f)])
      {
        continue;
      }
      FaceElementTransformations* transformation =
        mesh->GetFaceElementTransformations(f);
      if (!transformation || transformation->Elem2No < 0)
      {
        continue;
      }
      const int e1 = transformation->Elem1No;
      const int e2 = transformation->Elem2No;
      const FiniteElement& fe1 = *op_->vfes.GetFE(e1);
      const FiniteElement& fe2 = *op_->vfes.GetFE(e2);
      for (int q = 0; q < face_rule.GetNPoints(); ++q)
      {
        const IntegrationPoint& point = face_rule.IntPoint(q);
        transformation->SetAllIntPoints(&point);
        const int local1 =
          local_from_point(fe1, transformation->GetElement1IntPoint());
        const int local2 =
          local_from_point(fe2, transformation->GetElement2IntPoint());
        std::array<real_t, 2> normal;
        if (local1 < 0 || local2 < 0 ||
            !normal_from_point(*transformation,
                               e1,
                               local1,
                               transformation->GetElement1IntPoint(),
                               normal))
        {
          return fail(DGSEMProductionReason::InvalidFaceRange);
        }
        append_face(e1,
                    local1,
                    e2,
                    local2,
                    DGSEMDeviceFaceOwnership::Local,
                    normal,
                    point.weight,
                    {{0, 0, 0, 0}},
                    -1,
                    nullptr);
        ++c.local_faces;
      }
    }

#ifdef MFEM_USE_MPI
    if (pmesh && pmesh->GetNSharedFaces() > 0)
    {
      Array<int> neighbor_vdofs;
      for (int sf = 0; sf < pmesh->GetNSharedFaces(); ++sf)
      {
        FaceElementTransformations* transformation =
          pmesh->GetSharedFaceTransformations(sf, true);
        if (!transformation)
        {
          continue;
        }
        const int e1 = transformation->Elem1No;
        const int e2 = transformation->Elem2No - pmesh->GetNE();
        if (e2 < 0 || !pfes->GetFaceNbrFE(e2) ||
            pfes->GetFaceNbrElementVDofs(e2, neighbor_vdofs) != nullptr ||
            neighbor_vdofs.Size() != c.dof * c.equations)
        {
          return fail(DGSEMProductionReason::InvalidFaceRange);
        }
        const FiniteElement& fe1 = *op_->vfes.GetFE(e1);
        const FiniteElement& fe2 = *pfes->GetFaceNbrFE(e2);
        for (int q = 0; q < face_rule.GetNPoints(); ++q)
        {
          const IntegrationPoint& point = face_rule.IntPoint(q);
          transformation->SetAllIntPoints(&point);
          const int local1 =
            local_from_point(fe1, transformation->GetElement1IntPoint());
          const int local2 =
            local_from_point(fe2, transformation->GetElement2IntPoint());
          std::array<real_t, 2> normal;
          if (local1 < 0 || local2 < 0 ||
              !normal_from_point(*transformation,
                                 e1,
                                 local1,
                                 transformation->GetElement1IntPoint(),
                                 normal))
          {
            return fail(DGSEMProductionReason::InvalidFaceRange);
          }
          std::array<int, 4> map{{0, 0, 0, 0}};
          for (int equation = 0; equation < c.equations; ++equation)
          {
            map[static_cast<std::size_t>(equation)] =
              neighbor_vdofs[local2 + c.dof * equation];
          }
          const IntegrationPoint& point2 = transformation->GetElement2IntPoint();
          append_face(e1,
                      local1,
                      -1,
                      local2,
                      DGSEMDeviceFaceOwnership::Shared,
                      normal,
                      point.weight,
                      map,
                      e2,
                      &point2);
          ++c.shared_faces;
        }
      }
    }
#endif

    for (int b = 0; b < mesh->GetNBE(); ++b)
    {
      FaceElementTransformations* transformation =
        mesh->GetBdrFaceTransformations(b);
      if (!transformation)
      {
        continue;
      }
      const int e1 = transformation->Elem1No;
      const FiniteElement& fe1 = *op_->vfes.GetFE(e1);
      for (int q = 0; q < face_rule.GetNPoints(); ++q)
      {
        const IntegrationPoint& point = face_rule.IntPoint(q);
        transformation->SetAllIntPoints(&point);
        const int local1 =
          local_from_point(fe1, transformation->GetElement1IntPoint());
        std::array<real_t, 2> normal;
        if (local1 < 0 ||
            !normal_from_point(*transformation,
                               e1,
                               local1,
                               transformation->GetElement1IntPoint(),
                               normal))
        {
          return fail(DGSEMProductionReason::InvalidFaceRange);
        }
        append_face(e1,
                    local1,
                    -1,
                    -1,
                    DGSEMDeviceFaceOwnership::Boundary,
                    normal,
                    point.weight,
                    {{0, 0, 0, 0}},
                    -1,
                    nullptr);
        ++c.boundary_faces;
      }
    }

    const int nfaces = static_cast<int>(left_elements.size());
    c.normals.SetSize(nfaces, op_->dim);
    c.face_weights.SetSize(nfaces);
    c.left_elements.SetSize(nfaces);
    c.left_local_dofs.SetSize(nfaces);
    c.right_elements.SetSize(nfaces);
    c.right_local_dofs.SetSize(nfaces);
    c.ownership.SetSize(nfaces);
    c.shared_state_vdofs.SetSize(nfaces * c.equations);
    c.shared_neighbor_elements.SetSize(nfaces);
    c.shared_neighbor_points.SetSize(nfaces, op_->dim);
    for (int face = 0; face < nfaces; ++face)
    {
      c.face_weights(face) = face_weights[static_cast<std::size_t>(face)];
      c.left_elements[face] = left_elements[static_cast<std::size_t>(face)];
      c.left_local_dofs[face] = left_local[static_cast<std::size_t>(face)];
      c.right_elements[face] = right_elements[static_cast<std::size_t>(face)];
      c.right_local_dofs[face] = right_local[static_cast<std::size_t>(face)];
      c.ownership[face] = ownership[static_cast<std::size_t>(face)];
      c.shared_neighbor_elements[face] =
        neighbor_elements[static_cast<std::size_t>(face)];
      for (int d = 0; d < op_->dim; ++d)
      {
        c.normals(face, d) =
          face_normals[static_cast<std::size_t>(face)][static_cast<std::size_t>(d)];
        c.shared_neighbor_points(face, d) =
          neighbor_points[static_cast<std::size_t>(face)][static_cast<std::size_t>(d)];
      }
      for (int equation = 0; equation < c.equations; ++equation)
      {
        c.shared_state_vdofs[face + nfaces * equation] =
          neighbor_state_maps[static_cast<std::size_t>(face)]
                             [static_cast<std::size_t>(equation)];
      }
    }

    DGSEMProductionMetadataView metadata;
    metadata.dimension = op_->dim;
    metadata.order = order;
    metadata.elements = c.ne;
    metadata.dofs_per_element = c.dof;
    metadata.equations = c.equations;
    metadata.local_vector_size = op_->vfes.GetVSize();
    metadata.face_neighbor_vector_size = c.face_neighbor_vsize;
    metadata.lex_to_local = &c.lex_to_local;
    metadata.element_vdofs = &c.element_vdofs;
    metadata.metrics = &c.metrics;
    metadata.inverse_mass = &c.inverse_mass;
    metadata.normals = &c.normals;
    metadata.face_weights = &c.face_weights;
    metadata.left_elements = &c.left_elements;
    metadata.left_local_dofs = &c.left_local_dofs;
    metadata.right_elements = &c.right_elements;
    metadata.right_local_dofs = &c.right_local_dofs;
    metadata.ownership = &c.ownership;
    metadata.shared_state_vdofs = &c.shared_state_vdofs;
    const DGSEMProductionReason metadata_reason =
      ValidateDGSEMProductionMetadata(metadata);
    if (metadata_reason != DGSEMProductionReason::None)
    {
      return fail(metadata_reason);
    }
    c.vfes_sequence = vfes_sequence;
    c.mesh_sequence = mesh_sequence;
    c.topology_ready = true;
    c.sigma_ready = false;
    c.workspaces_ready = false;
    ++c.counters.topology_builds;
  }

  if (!c.sigma_ready)
  {
    const int nfaces = c.normals.Height();
    if (!sigma_field)
    {
      c.sigma_ldof = 0;
      c.sigma_element_dofs.SetSize(0);
      c.sigma_interpolation.SetSize(0, 0);
      c.sigma_shared_neighbor_dofs.SetSize(0);
      c.sigma_shared_interpolation.SetSize(0, 0);
#ifdef MFEM_USE_MPI
      c.sigma_send.SetSize(0);
#endif
      c.sigma_ready = true;
      ++c.counters.sigma_metadata_builds;
    }
    else
    {
      auto* scalar_fes =
        const_cast<FiniteElementSpace*>(sigma_field->FESpace());
      if (!scalar_fes || scalar_fes->GetMesh() != mesh ||
          scalar_fes->GetNE() != c.ne || scalar_fes->GetVDim() != 1)
      {
        return fail(DGSEMProductionReason::UnsupportedSigma);
      }
      c.sigma_ldof = scalar_fes->GetFE(0)->GetDof();
      if (c.sigma_ldof <= 0)
      {
        return fail(DGSEMProductionReason::UnsupportedSigma);
      }
      c.sigma_element_dofs.SetSize(c.rows * c.sigma_ldof);
      c.sigma_interpolation.SetSize(c.rows, c.sigma_ldof);
      Array<int> scalar_dofs;
      Vector shape(c.sigma_ldof);
      for (int e = 0; e < c.ne; ++e)
      {
        const FiniteElement& scalar_fe = *scalar_fes->GetFE(e);
        const FiniteElement& state_fe = *op_->vfes.GetFE(e);
        if (scalar_fe.GetDof() != c.sigma_ldof)
        {
          return fail(DGSEMProductionReason::UnsupportedSigma);
        }
        scalar_fes->GetElementDofs(e, scalar_dofs);
        if (scalar_dofs.Size() != c.sigma_ldof)
        {
          return fail(DGSEMProductionReason::UnsupportedSigma);
        }
        for (int local = 0; local < c.dof; ++local)
        {
          const int row = e * c.dof + local;
          scalar_fe.CalcShape(state_fe.GetNodes().IntPoint(local), shape);
          for (int j = 0; j < c.sigma_ldof; ++j)
          {
            c.sigma_element_dofs[row + c.rows * j] = scalar_dofs[j];
            c.sigma_interpolation(row, j) = shape(j);
          }
        }
      }
      c.sigma_shared_neighbor_dofs.SetSize(nfaces * c.sigma_ldof);
      c.sigma_shared_neighbor_dofs = 0;
      c.sigma_shared_interpolation.SetSize(nfaces, c.sigma_ldof);
      c.sigma_shared_interpolation = 0.0;
#ifdef MFEM_USE_MPI
      auto* scalar_pfes = dynamic_cast<ParFiniteElementSpace*>(scalar_fes);
      if (c.shared_faces > 0)
      {
        if (!scalar_pfes)
        {
          return fail(DGSEMProductionReason::UnsupportedSigma);
        }
        c.sigma_send.SetSize(
          scalar_pfes->send_face_nbr_ldof.Size_of_connections());
        c.sigma_send.UseDevice(true);
        auto* mutable_sigma = const_cast<ParGridFunction*>(sigma_field);
        mutable_sigma->FaceNbrData().SetSize(scalar_pfes->GetFaceNbrVSize());
        mutable_sigma->FaceNbrData().UseDevice(true);
        Array<int> neighbor_dofs;
        for (int face = 0; face < nfaces; ++face)
        {
          if (c.ownership[face] !=
              static_cast<int>(DGSEMDeviceFaceOwnership::Shared))
          {
            continue;
          }
          const int neighbor_element = c.shared_neighbor_elements[face];
          const FiniteElement* neighbor_fe =
            scalar_pfes->GetFaceNbrFE(neighbor_element);
          if (!neighbor_fe || neighbor_fe->GetDof() != c.sigma_ldof ||
              scalar_pfes->GetFaceNbrElementVDofs(neighbor_element,
                                                  neighbor_dofs) != nullptr ||
              neighbor_dofs.Size() != c.sigma_ldof)
          {
            return fail(DGSEMProductionReason::UnsupportedSigma);
          }
          IntegrationPoint point;
          point.x = c.shared_neighbor_points(face, 0);
          if (op_->dim == 2)
          {
            point.y = c.shared_neighbor_points(face, 1);
          }
          neighbor_fe->CalcShape(point, shape);
          for (int j = 0; j < c.sigma_ldof; ++j)
          {
            const int vdof = neighbor_dofs[j];
            const int absolute = AbsoluteDof(vdof);
            if (absolute < 0 || absolute >= scalar_pfes->GetFaceNbrVSize())
            {
              return fail(DGSEMProductionReason::InvalidFaceRange);
            }
            c.sigma_shared_neighbor_dofs[face + nfaces * j] = vdof;
            c.sigma_shared_interpolation(face, j) = shape(j);
          }
        }
      }
#else
      if (c.shared_faces > 0)
      {
        return fail(DGSEMProductionReason::UnsupportedSigma);
      }
#endif
      c.sigma_fes = scalar_fes;
      c.sigma_fes_sequence = scalar_fes->GetSequence();
      c.sigma_ready = true;
      ++c.counters.sigma_metadata_builds;
    }
  }

  if (!c.workspaces_ready)
  {
    const int nfaces = c.normals.Height();
    c.states.SetSize(c.rows, c.equations);
    c.sigma_nodes.SetSize(c.rows);
    c.contravariant_flux.SetSize(c.rows, op_->dim * c.equations);
    c.nodal_speeds.SetSize(c.rows);
    c.weak_volume.SetSize(c.rows, c.equations);
    c.left_states.SetSize(nfaces, c.equations);
    c.right_states.SetSize(nfaces, c.equations);
    c.left_sigma.SetSize(nfaces);
    c.right_sigma.SetSize(nfaces);
    c.shared_sigma.SetSize(nfaces);
    c.face_flux.SetSize(nfaces, c.equations);
    c.face_speeds.SetSize(nfaces);
    c.weak_surface.SetSize(c.rows, c.equations);
    c.strong.SetSize(c.rows, c.equations);
    c.nodal_max.SetSize(1);
    c.face_max.SetSize(1);
    c.empty_face_neighbor.SetSize(0);
    c.states.GetMemory().UseDevice(true);
    c.sigma_nodes.UseDevice(true);
    c.contravariant_flux.GetMemory().UseDevice(true);
    c.nodal_speeds.UseDevice(true);
    c.weak_volume.GetMemory().UseDevice(true);
    c.left_states.GetMemory().UseDevice(true);
    c.right_states.GetMemory().UseDevice(true);
    c.left_sigma.UseDevice(true);
    c.right_sigma.UseDevice(true);
    c.shared_sigma.UseDevice(true);
    c.face_flux.GetMemory().UseDevice(true);
    c.face_speeds.UseDevice(true);
    c.weak_surface.GetMemory().UseDevice(true);
    c.strong.GetMemory().UseDevice(true);
    c.nodal_max.UseDevice(true);
    c.face_max.UseDevice(true);
    if (!sigma_field)
    {
      c.sigma_nodes = 0.0;
      c.shared_sigma = 0.0;
    }
#ifdef MFEM_USE_MPI
    if (c.shared_faces > 0)
    {
      auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&op_->vfes);
      if (!pfes)
      {
        return fail(DGSEMProductionReason::InvalidFaceRange);
      }
      c.state_send.SetSize(
        pfes->send_face_nbr_ldof.Size_of_connections());
      c.state_face_neighbor.SetSize(pfes->GetFaceNbrVSize());
      c.state_send.UseDevice(true);
      c.state_face_neighbor.UseDevice(true);
      ParMesh* pmesh = pfes->GetParMesh();
      const int neighbors = pmesh ? pmesh->GetNFaceNeighbors() : 0;
      c.exchange_requests.resize(static_cast<std::size_t>(2 * neighbors));
      c.exchange_statuses.resize(static_cast<std::size_t>(neighbors));
    }
#endif
    c.workspaces_ready = true;
    c.workspace_evidence = DGSEMProductionWorkspaceEvidence{};
    ++c.counters.workspace_builds;
  }

  c.selection.supported = true;
  c.selection.reason = DGSEMProductionReason::None;
  c.selection.dimension = op_->dim;
  c.selection.order = op_->vfes.GetFE(0)->GetOrder();
  c.selection.elements = c.ne;
  c.selection.uses_sigma = sigma_field != nullptr;
  c.selection.has_local_faces = c.local_faces > 0;
  c.selection.has_shared_faces = c.shared_faces > 0;
  c.selection.has_boundary_faces = c.boundary_faces > 0;
  return DGSEMProductionReason::None;
}

DGSEMProductionSelection DGSEMOperator::BindProductionPolicy(bool strict) const
{
  EnsureProductionCache_(strict);
  return production_ ? production_->selection : DGSEMProductionSelection{};
}

DGSEMProductionResult DGSEMOperator::AssembleProductionResidual(
  const Vector& x_local,
  Vector& y_local,
  bool strict) const
{
  DGSEMProductionResult result;
  const DGSEMProductionReason cache_reason = EnsureProductionCache_(strict);
  if (cache_reason != DGSEMProductionReason::None || !production_)
  {
    result.reason = cache_reason;
    return result;
  }

  DGSEMProductionCache& c = *production_;
  const ProductionCommunicator context =
    GetProductionCommunicator(op_->vfes);
  ++c.counters.stage_calls;

  DGSEMProductionReason local_gate = DGSEMProductionReason::None;
  if (x_local.Size() != op_->vfes.GetVSize())
  {
    local_gate = DGSEMProductionReason::InputSizeMismatch;
  }
  else if (y_local.Size() != op_->vfes.GetVSize())
  {
    local_gate = DGSEMProductionReason::OutputSizeMismatch;
  }
  else if (!op_->vfes.GetMesh() ||
           c.vfes_sequence != op_->vfes.GetSequence() ||
           c.mesh_sequence != op_->vfes.GetMesh()->GetSequence())
  {
    local_gate = DGSEMProductionReason::StaleCache;
  }
  else if (c.sigma_field &&
           (!c.sigma_fes || c.sigma_field->Size() != c.sigma_fes->GetVSize() ||
            c.sigma_fes_sequence != c.sigma_fes->GetSequence()))
  {
    local_gate = DGSEMProductionReason::StaleCache;
  }

  if (local_gate == DGSEMProductionReason::None &&
      c.workspace_evidence.ready)
  {
    ++c.counters.host_synchronizations;
    const DGSEMProductionWorkspaceEvidence current = SnapshotWorkspaces(c);
    ++c.workspace_evidence.checks;
    ++c.counters.workspace_identity_checks;
    if (!SameWorkspaceStorage(c.workspace_evidence, current))
    {
      local_gate = DGSEMProductionReason::WorkspaceIdentityChanged;
    }
  }
  DGSEMProductionReason gate_reason = ReduceReason(
    context,
    local_gate,
    c.counters.stage_reason_reductions);
  if (gate_reason != DGSEMProductionReason::None)
  {
    if (gate_reason == DGSEMProductionReason::WorkspaceIdentityChanged)
    {
      ++c.workspace_evidence.failures;
      ++c.counters.workspace_identity_failures;
    }
    result.reason = gate_reason;
    return result;
  }

  GatherElementStateDevice(c.ne,
                           c.dof,
                           c.equations,
                           c.element_vdofs,
                           x_local,
                           c.states);
  if (c.sigma_field)
  {
    InterpolateElementScalarDevice(c.rows,
                                   c.sigma_ldof,
                                   c.sigma_element_dofs,
                                   c.sigma_interpolation,
                                   *c.sigma_field,
                                   c.sigma_nodes);
  }
  else
  {
    c.sigma_nodes = 0.0;
    c.shared_sigma = 0.0;
  }

  ++c.counters.host_synchronizations;
  DGSEMProductionReason local_state_reason =
    ValidateDGSEMProductionState(op_->dim,
                                 c.gamma,
                                 c.velocity_density_floor,
                                 c.states,
                                 c.sigma_nodes);
  const DGSEMProductionReason state_reason = ReduceReason(
    context,
    local_state_reason,
    c.counters.stage_reason_reductions);
  if (state_reason != DGSEMProductionReason::None)
  {
    result.reason = state_reason;
    return result;
  }

  const Vector* face_neighbor_state = &c.empty_face_neighbor;
  DGSEMProductionReason exchange_reason = DGSEMProductionReason::None;
#ifdef MFEM_USE_MPI
  if (c.shared_faces > 0)
  {
    auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&op_->vfes);
    if (!pfes)
    {
      exchange_reason = DGSEMProductionReason::InvalidFaceRange;
    }
    else
    {
      ExchangeFaceNeighborVector(*pfes,
                                 x_local,
                                 c.state_send,
                                 c.state_face_neighbor,
                                 c.exchange_requests,
                                 c.exchange_statuses);
      ++c.counters.state_exchanges;
      face_neighbor_state = &c.state_face_neighbor;
      if (face_neighbor_state->Size() != c.face_neighbor_vsize)
      {
        exchange_reason = DGSEMProductionReason::InvalidFaceRange;
      }
    }
  }

  if (c.sigma_field && c.shared_faces > 0)
  {
    auto* scalar_pfes = dynamic_cast<ParFiniteElementSpace*>(
      const_cast<FiniteElementSpace*>(c.sigma_field->FESpace()));
    if (!scalar_pfes)
    {
      exchange_reason = DGSEMProductionReason::UnsupportedSigma;
    }
    else
    {
      auto* mutable_sigma = const_cast<ParGridFunction*>(c.sigma_field);
      ExchangeFaceNeighborVector(*scalar_pfes,
                                 *c.sigma_field,
                                 c.sigma_send,
                                 mutable_sigma->FaceNbrData(),
                                 c.exchange_requests,
                                 c.exchange_statuses);
      ++c.counters.sigma_exchanges;
      InterpolateSharedFaceScalarDevice(c.normals.Height(),
                                        c.sigma_ldof,
                                        c.sigma_shared_neighbor_dofs,
                                        c.sigma_shared_interpolation,
                                        mutable_sigma->FaceNbrData(),
                                        c.shared_sigma);
    }
  }
#else
  if (c.shared_faces > 0)
  {
    exchange_reason = DGSEMProductionReason::InvalidFaceRange;
  }
#endif
  exchange_reason = ReduceReason(context,
                                 exchange_reason,
                                 c.counters.stage_reason_reductions);
  if (exchange_reason != DGSEMProductionReason::None)
  {
    result.reason = exchange_reason;
    return result;
  }

  GatherEulerRusanovSlipFacesDevice(op_->dim,
                                    c.ne,
                                    c.dof,
                                    c.states,
                                    c.sigma_nodes,
                                    *face_neighbor_state,
                                    c.shared_state_vdofs,
                                    c.shared_sigma,
                                    c.normals,
                                    c.left_elements,
                                    c.left_local_dofs,
                                    c.right_elements,
                                    c.right_local_dofs,
                                    c.ownership,
                                    c.left_states,
                                    c.right_states,
                                    c.left_sigma,
                                    c.right_sigma);

  ++c.counters.host_synchronizations;
  DGSEMProductionReason local_face_reason =
    ValidateDGSEMProductionState(op_->dim,
                                 c.gamma,
                                 c.velocity_density_floor,
                                 c.left_states,
                                 c.left_sigma);
  if (local_face_reason == DGSEMProductionReason::None)
  {
    ++c.counters.host_synchronizations;
    local_face_reason =
      ValidateDGSEMProductionState(op_->dim,
                                   c.gamma,
                                   c.velocity_density_floor,
                                   c.right_states,
                                   c.right_sigma);
  }
  const DGSEMProductionReason face_reason = ReduceReason(
    context,
    local_face_reason,
    c.counters.stage_reason_reductions);
  if (face_reason != DGSEMProductionReason::None)
  {
    result.reason = face_reason;
    return result;
  }

  BuildEulerContravariantFluxDevice(op_->dim,
                                    c.gamma,
                                    c.velocity_density_floor,
                                    c.states,
                                    c.sigma_nodes,
                                    c.metrics,
                                    c.contravariant_flux,
                                    c.nodal_speeds);
  const LGLBasis1D& basis = GetLGLBasis1D(c.selection.order);
  ApplyCentralVolumeDevice(op_->dim,
                           c.n1d,
                           c.lex_to_local,
                           basis.Dsplit,
                           basis.weights,
                           c.contravariant_flux,
                           c.weak_volume);

  // The accepted face kernel retains its packed host normal preflight. This
  // is one explicit, accounted synchronization per production stage.
  ++c.counters.host_synchronizations;
  AssembleEulerRusanovFacesDevice(op_->dim,
                                  c.ne,
                                  c.dof,
                                  c.gamma,
                                  c.velocity_density_floor,
                                  c.left_states,
                                  c.right_states,
                                  c.normals,
                                  c.left_sigma,
                                  c.right_sigma,
                                  c.face_weights,
                                  c.left_elements,
                                  c.left_local_dofs,
                                  c.right_elements,
                                  c.right_local_dofs,
                                  c.ownership,
                                  c.face_flux,
                                  c.face_speeds,
                                  c.weak_surface);
  AddPackedMatricesDevice(c.weak_surface, c.weak_volume);
  ApplyInverseMassDevice(c.ne,
                         c.dof,
                         c.inverse_mass,
                         c.weak_volume,
                         c.strong);
  DeterministicMaximumDevice(c.nodal_speeds, c.nodal_max);
  DeterministicMaximumDevice(c.face_speeds, c.face_max);

  ++c.counters.host_synchronizations;
  const DGSEMProductionWorkspaceEvidence current = SnapshotWorkspaces(c);
  DGSEMProductionReason local_workspace_reason =
    DGSEMProductionReason::None;
  if (!c.workspace_evidence.ready)
  {
    c.workspace_evidence = current;
  }
  else
  {
    ++c.workspace_evidence.checks;
    ++c.counters.workspace_identity_checks;
    if (!SameWorkspaceStorage(c.workspace_evidence, current))
    {
      local_workspace_reason =
        DGSEMProductionReason::WorkspaceIdentityChanged;
    }
  }
  const DGSEMProductionReason workspace_reason = ReduceReason(
    context,
    local_workspace_reason,
    c.counters.stage_reason_reductions);
  if (workspace_reason != DGSEMProductionReason::None)
  {
    ++c.workspace_evidence.failures;
    ++c.counters.workspace_identity_failures;
    result.reason = workspace_reason;
    return result;
  }

  ++c.counters.host_synchronizations;
  const real_t nodal_max = c.nodal_max.HostRead()[0];
  ++c.counters.host_synchronizations;
  const real_t face_max = c.face_max.HostRead()[0];
  real_t max_speed = std::max(nodal_max, face_max);
#ifdef MFEM_USE_MPI
  if (context.active && context.size > 1)
  {
    real_t global_max = max_speed;
    MPI_Allreduce(&max_speed,
                  &global_max,
                  1,
                  MPITypeMap<real_t>::mpi_type,
                  MPI_MAX,
                  context.comm);
    max_speed = global_max;
    ++c.counters.mpi_reductions;
  }
#endif

  // This is the first write to the caller-owned output in the whole stage.
  ScatterElementStateDevice(c.ne,
                            c.dof,
                            c.equations,
                            c.element_vdofs,
                            c.strong,
                            y_local);
  ++c.counters.output_scatters;
  result.executed = true;
  result.reason = DGSEMProductionReason::None;
  result.max_characteristic_speed = max_speed;
  return result;
}

} // namespace dgsem
} // namespace mfem
