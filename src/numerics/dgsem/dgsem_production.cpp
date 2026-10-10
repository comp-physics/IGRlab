#include "igrlab/environment.hpp"
#include "numerics/dgsem/dgsem_operator.hpp"
#include "dgsem_overintegrated_volume.hpp"

#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_device.hpp"
#include "numerics/dgsem/dgsem_metric.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/bc/detail/euler_boundary_state.hpp"
#include "physics/igr/igr_flux.hpp"
#include "utils/config/dg_igr_defaults.hpp"
#include "utils/config/env_flags.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cerrno>

namespace mfem
{
namespace dgsem
{
namespace overintegration_internal
{
void BuildVolumeQuadratureMaps2D(const FiniteElement& fe,
                                 const IntegrationRule& rule,
                                 DenseMatrix& interpolation,
                                 DenseMatrix& weak_derivative)
{
  MFEM_VERIFY(fe.GetDim()==2&&fe.GetGeomType()==Geometry::SQUARE&&
                fe.GetMapType()==FiniteElement::VALUE,
              "Overintegrated volume maps require value-mapped 2D quadrilaterals.");
  const int dof=fe.GetDof(),nq=rule.GetNPoints();
  interpolation.SetSize(nq,dof);weak_derivative.SetSize(dof,2*nq);
  interpolation.GetMemory().UseDevice(true);weak_derivative.GetMemory().UseDevice(true);
  real_t* b=interpolation.HostWrite();real_t* g=weak_derivative.HostWrite();
  Vector shape(dof);DenseMatrix derivative(dof,2);
  for(int q=0;q<nq;++q)
  {
    const auto& ip=rule.IntPoint(q);fe.CalcShape(ip,shape);fe.CalcDShape(ip,derivative);
    for(int i=0;i<dof;++i)
    {
      b[q+nq*i]=shape[i];
      for(int d=0;d<2;++d)g[i+dof*(q+nq*d)]=ip.weight*derivative(i,d);
    }
  }
}
}

namespace subcell_internal
{
// Private aggregate accounting; its implementation must be CUDA-compiled,
// because this orchestration translation unit is ordinary host C++.
void AccumulateSubcellBlendVisits(const Vector& blend, Vector& visits);
}
namespace
{

int RequestedVolumeQuadratureOrder()
{
  const char* raw=::igrlab::GetEnv("DG_IGR_DGSEM_VOLUME_QUADRATURE_MIN_ORDER");
  if(!raw)return 0;
  errno=0;char* end=nullptr;const long value=std::strtol(raw,&end,10);
  MFEM_VERIFY(raw[0]&&end&&!*end&&errno==0&&value>=0&&value<=26,
    "DG_IGR_DGSEM_VOLUME_QUADRATURE_MIN_ORDER must be an integer in [0,26].");
  return static_cast<int>(value);
}

// Explicit research extension; default qualification remains p1--p4.
// Generic packed kernels and the fused path are independently checked to p7.
inline int MaximumDGSEMProductionOrder()
{
  return ::igrlab::EnvFlagEnabledDefault("DG_IGR_DGSEM_HIGH_ORDER_RESEARCH", false)
      ? 7 : 4;
}

inline int AbsoluteDof(int vdof)
{
  return vdof >= 0 ? vdof : -1 - vdof;
}

inline bool Finite(real_t value)
{
  return std::isfinite(static_cast<double>(value));
}

bool BuildExactScalarNodeMap(FiniteElementSpace& scalar_fes,
                             const FiniteElementSpace& state_fes,
                             int dof,
                             Array<int>& map)
{
  map.SetSize(state_fes.GetNE() * dof);
  Array<int> scalar_dofs;
  for (int element = 0; element < state_fes.GetNE(); ++element)
  {
    const FiniteElement& scalar = *scalar_fes.GetFE(element);
    const FiniteElement& state = *state_fes.GetFE(element);
    if (!dynamic_cast<const NodalFiniteElement*>(&scalar) ||
        scalar.GetMapType() != FiniteElement::VALUE ||
        scalar.GetDim() != state.GetDim() ||
        scalar.GetGeomType() != state.GetGeomType() ||
        scalar.GetDof() != dof || scalar.GetNodes().GetNPoints() != dof ||
        state.GetNodes().GetNPoints() != dof)
      return false;
    scalar_fes.GetElementDofs(element, scalar_dofs);
    if (scalar_dofs.Size() != dof) return false;
    std::vector<char> seen(static_cast<std::size_t>(dof), 0);
    for (int local = 0; local < dof; ++local)
    {
      const IntegrationPoint& target = state.GetNodes().IntPoint(local);
      int match = -1;
      for (int j = 0; j < dof; ++j)
      {
        const IntegrationPoint& node = scalar.GetNodes().IntPoint(j);
        // This is an exact nodal identity, not a sparsification threshold.
        if (target.x == node.x &&
            (state.GetDim() == 1 || target.y == node.y))
        {
          match = j;
          break;
        }
      }
      if (match < 0 || seen[static_cast<std::size_t>(match)]) return false;
      seen[static_cast<std::size_t>(match)] = 1;
      map[element * dof + local] = scalar_dofs[match];
    }
  }
  return true;
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

struct LoweredEulerBoundary
{
  detail::EulerBoundaryStateKind kind =
    detail::EulerBoundaryStateKind::unsupported;
  const Vector* reference_state = nullptr;
  int dimension = -1;
  real_t gamma = 1.4;
  real_t pressure_out = 0.0;
  real_t density_floor = kEulerBoundaryDensityFloor;
  real_t pressure_floor = kEulerBoundaryPressureFloor;
};

LoweredEulerBoundary LowerEulerBoundary(const BoundaryState* state)
{
  LoweredEulerBoundary lowered;
  if (const auto* inlet = dynamic_cast<const SupersonicInletState*>(state))
  {
    lowered.kind = detail::EulerBoundaryStateKind::supersonic_inlet;
    lowered.reference_state = &inlet->GetInletState();
  }
  else if (dynamic_cast<const SupersonicOutletState*>(state))
  {
    lowered.kind = detail::EulerBoundaryStateKind::supersonic_outlet;
  }
  else if (const auto* inlet = dynamic_cast<const SubsonicInletState*>(state))
  {
    lowered.kind = detail::EulerBoundaryStateKind::subsonic_inlet;
    lowered.reference_state = &inlet->GetInletState();
    lowered.dimension = inlet->GetDimension();
    lowered.gamma = inlet->GetGamma();
    lowered.density_floor = inlet->GetDensityFloor();
    lowered.pressure_floor = inlet->GetPressureFloor();
  }
  else if (const auto* outlet =
             dynamic_cast<const SubsonicOutletPressureState*>(state))
  {
    lowered.kind = detail::EulerBoundaryStateKind::subsonic_outlet_pressure;
    lowered.dimension = outlet->GetDimension();
    lowered.gamma = outlet->GetGamma();
    lowered.pressure_out = outlet->GetOutletPressure();
    lowered.density_floor = outlet->GetDensityFloor();
    lowered.pressure_floor = outlet->GetPressureFloor();
  }
  else if (const auto* far_field = dynamic_cast<const FarFieldState*>(state))
  {
    lowered.kind = detail::EulerBoundaryStateKind::far_field;
    lowered.reference_state = &far_field->GetFarFieldState();
    lowered.dimension = far_field->GetDimension();
    lowered.gamma = far_field->GetGamma();
    lowered.density_floor = far_field->GetDensityFloor();
    lowered.pressure_floor = far_field->GetPressureFloor();
  }
  else if (const auto* outlet =
             dynamic_cast<const NonReflectingOutletState*>(state))
  {
    lowered.kind = detail::EulerBoundaryStateKind::nonreflecting_outlet;
    lowered.reference_state = &outlet->GetReferenceState();
    lowered.dimension = outlet->GetDimension();
    lowered.gamma = outlet->GetGamma();
    lowered.density_floor = outlet->GetDensityFloor();
    lowered.pressure_floor = outlet->GetPressureFloor();
  }
  else if (const auto* wall = dynamic_cast<const SlipWallState*>(state))
  {
    lowered.kind = detail::EulerBoundaryStateKind::slip_wall;
    lowered.dimension = wall->GetDimension();
    lowered.density_floor = wall->GetDensityFloor();
    lowered.pressure_floor = wall->GetPressureFloor();
  }
  return lowered;
}

bool IsSupportedEulerBoundary(const LoweredEulerBoundary& boundary,
                              int dimension,
                              int equations)
{
  if (boundary.kind == detail::EulerBoundaryStateKind::unsupported)
  {
    return false;
  }
  if (boundary.dimension >= 0 && boundary.dimension != dimension)
  {
    return false;
  }
  return !boundary.reference_state ||
         boundary.reference_state->Size() == equations;
}

DGSEMProductionReason ValidateRowFaceAdjacency(
  int rows,
  int nfaces,
  int dof,
  const Array<int>& left_elements,
  const Array<int>& left_local,
  const Array<int>& right_elements,
  const Array<int>& right_local,
  const Array<int>& ownership,
  const Array<int>& row_face_offsets,
  const Array<int>& row_face_ids,
  const Array<int>& row_face_signs)
{
  if (row_face_offsets.Size() != rows + 1 ||
      row_face_ids.Size() != row_face_signs.Size() ||
      row_face_offsets[0] != 0 ||
      row_face_offsets[rows] != row_face_ids.Size())
  {
    return DGSEMProductionReason::InvalidFaceRange;
  }
  for (int row = 0; row < rows; ++row)
  {
    if (row_face_offsets[row] > row_face_offsets[row + 1])
    {
      return DGSEMProductionReason::InvalidFaceRange;
    }
    for (int adjacent = row_face_offsets[row];
         adjacent < row_face_offsets[row + 1];
         ++adjacent)
    {
      const int face = row_face_ids[adjacent];
      const int sign = row_face_signs[adjacent];
      if (face < 0 || face >= nfaces || (sign != -1 && sign != 1))
      {
        return DGSEMProductionReason::InvalidFaceRange;
      }
      const bool left = sign == -1 &&
                        left_elements[face] * dof + left_local[face] == row;
      const bool right = sign == 1 &&
                         ownership[face] == static_cast<int>(
                           DGSEMDeviceFaceOwnership::Local) &&
                         right_elements[face] * dof + right_local[face] == row;
      if (!left && !right)
      {
        return DGSEMProductionReason::InvalidFaceRange;
      }
    }
  }
  return DGSEMProductionReason::None;
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

  // This translation unit is intentionally host-compiled.  Stage both the
  // map and values before packing so a CUDA-selected MFEM memory class never
  // exposes device pointers to this host loop.  MPI below is likewise
  // deliberately non-GPU-aware and exchanges these persistent host buffers.
  const int* send_ldof = pfes.send_face_nbr_ldof.HostReadJ();
  const real_t* input = local.HostRead();
  real_t* send_value = send.HostWrite();
  for (int i = 0; i < send_size; ++i)
  {
    const int ldof = send_ldof[i];
    send_value[i] = input[ldof >= 0 ? ldof : -1 - ldof];
  }

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
  if (m.order < 1 || m.order > MaximumDGSEMProductionOrder())
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
      (m.inverse_mass->Width() != m.dofs_per_element &&
       m.inverse_mass->Width() != 1))
  {
    return DGSEMProductionReason::InvalidMetric;
  }
  const real_t* inverse = m.inverse_mass->HostRead();
  for (int row = 0; row < rows; ++row)
  {
    real_t row_norm = 0.0;
    for (int j = 0; j < m.inverse_mass->Width(); ++j)
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
    if (density < igrlab::LimiterDefaults::kRhoMin)
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

namespace
{

struct DGSEMDeviceStateScan
{
  DGSEMProductionReason reason = DGSEMProductionReason::None;
  IGRFrozenSigmaTelemetry q_sigma;
};

/** Validate the packed stage and collect q_sigma telemetry without migrating
 * the state or face panels to the host.
 *
 * A fixed set of deterministic device lanes scans the packed inputs and a
 * bounded ordered fold reduces their summaries. Only six FP64 result scalars
 * cross the device boundary. Element states are admitted before exchange and
 * face inputs afterward; no physical flux or caller output is evaluated until
 * both scans succeed collectively.
 */
DGSEMDeviceStateScan ScanDGSEMProductionStateDevice(
  int dimension,
  real_t gamma,
  real_t density_floor,
  bool scan_owned_states,
  bool scan_face_states,
  const DenseMatrix& states,
  const Vector& sigma,
  const DenseMatrix& left_states,
  const Vector& left_sigma,
  const DenseMatrix& right_states,
  const Vector& right_sigma,
  Array<double>& slot_workspace,
  Array<double>& result_workspace)
{
  ScanEulerIGRStatePanelsDevice(dimension,
                                gamma,
                                density_floor,
                                scan_owned_states,
                                scan_face_states,
                                states,
                                sigma,
                                left_states,
                                left_sigma,
                                right_states,
                                right_sigma,
                                slot_workspace,
                                result_workspace);
  const double* const host_result = result_workspace.HostRead();
  DGSEMDeviceStateScan scan;
  const int expected_samples = scan_owned_states
                                 ? states.Height()
                                 : 2 * left_states.Height();
  auto integral_count = [expected_samples](double value)
  {
    return std::isfinite(value) && value >= 0.0 &&
           value <= static_cast<double>(expected_samples) &&
           std::trunc(value) == value;
  };
  const bool status_well_formed =
    std::isfinite(host_result[0]) && std::trunc(host_result[0]) == host_result[0] &&
    host_result[0] >= static_cast<double>(DGSEMDeviceStateStatus::Valid) &&
    host_result[0] <= static_cast<double>(DGSEMDeviceStateStatus::InvalidPressure);
  const bool counts_well_formed = integral_count(host_result[1]) &&
                                  integral_count(host_result[2]) &&
                                  integral_count(host_result[3]) &&
                                  host_result[2] <= host_result[1] &&
                                  host_result[3] <= host_result[1];
  if (!status_well_formed || !counts_well_formed)
  {
    scan.reason = DGSEMProductionReason::InvalidPressure;
    return scan;
  }
  const auto status =
    static_cast<DGSEMDeviceStateStatus>(static_cast<int>(host_result[0]));
  if (status == DGSEMDeviceStateStatus::Valid &&
      host_result[1] != static_cast<double>(expected_samples))
  {
    scan.reason = DGSEMProductionReason::InvalidPressure;
    return scan;
  }
  switch (status)
  {
  case DGSEMDeviceStateStatus::Valid:
    scan.reason = DGSEMProductionReason::None;
    break;
  case DGSEMDeviceStateStatus::InvalidSigma:
    scan.reason = DGSEMProductionReason::InvalidSigma;
    break;
  case DGSEMDeviceStateStatus::InvalidDensity:
    scan.reason = DGSEMProductionReason::InvalidDensity;
    break;
  case DGSEMDeviceStateStatus::DensityBelowFloor:
    scan.reason = DGSEMProductionReason::DensityBelowBoundaryFloor;
    break;
  case DGSEMDeviceStateStatus::InvalidPressure:
    scan.reason = DGSEMProductionReason::InvalidPressure;
    break;
  default:
    scan.reason = DGSEMProductionReason::InvalidPressure;
    break;
  }
  scan.q_sigma.enabled = true;
  scan.q_sigma.sample_count =
    static_cast<std::uint64_t>(host_result[1]);
  scan.q_sigma.nonpositive_count =
    static_cast<std::uint64_t>(host_result[2]);
  scan.q_sigma.nonfinite_count =
    static_cast<std::uint64_t>(host_result[3]);
  scan.q_sigma.minimum_raw_q_sigma = host_result[4];
  scan.q_sigma.minimum_q_sigma_over_gamma_p = host_result[5];
  return scan;
}

/** Return the fresh q_sigma aggregate over the owning FE-space communicator.
 *
 * This reduction is reserved for the once-per-step endpoint admission scan;
 * ordinary residual calls continue to expose their cumulative telemetry
 * through DGHyperbolicConservationLaws without adding collectives here.
 */
IGRFrozenSigmaTelemetry ReduceFreshQSigmaTelemetry(
  const ProductionCommunicator& context,
  const IGRFrozenSigmaTelemetry& local,
  std::int64_t& reductions)
{
  IGRFrozenSigmaTelemetry global = local;
#ifdef MFEM_USE_MPI
  if (context.active && context.size > 1)
  {
    int enabled = local.enabled ? 1 : 0;
    unsigned long long counts[3] = {
      static_cast<unsigned long long>(local.sample_count),
      static_cast<unsigned long long>(local.nonpositive_count),
      static_cast<unsigned long long>(local.nonfinite_count)};
    double minima[2] = {local.minimum_raw_q_sigma,
                        local.minimum_q_sigma_over_gamma_p};
    MPI_Allreduce(MPI_IN_PLACE,
                  &enabled,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  context.comm);
    MPI_Allreduce(MPI_IN_PLACE,
                  counts,
                  3,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM,
                  context.comm);
    MPI_Allreduce(MPI_IN_PLACE,
                  minima,
                  2,
                  MPI_DOUBLE,
                  MPI_MIN,
                  context.comm);
    reductions += 3;
    global.enabled = enabled != 0;
    global.sample_count = static_cast<std::uint64_t>(counts[0]);
    global.nonpositive_count = static_cast<std::uint64_t>(counts[1]);
    global.nonfinite_count = static_cast<std::uint64_t>(counts[2]);
    global.minimum_raw_q_sigma = minima[0];
    global.minimum_q_sigma_over_gamma_p = minima[1];
  }
#else
  (void)context;
  (void)reductions;
#endif
  return global;
}

real_t ReducePackedCharacteristicSpeed(
  const ProductionCommunicator& context,
  const Vector& nodal_speeds,
  const Vector& face_speeds,
  Vector& nodal_maximum,
  Vector& face_maximum,
  std::int64_t& host_synchronizations,
  std::int64_t& mpi_reductions)
{
  DeterministicMaximumDevice(nodal_speeds, nodal_maximum);
  DeterministicMaximumDevice(face_speeds, face_maximum);

  ++host_synchronizations;
  const real_t nodal_max = nodal_maximum.HostRead()[0];
  ++host_synchronizations;
  const real_t face_max = face_maximum.HostRead()[0];
  real_t maximum = std::max(nodal_max, face_max);
#ifdef MFEM_USE_MPI
  if (context.active && context.size > 1)
  {
    real_t global_maximum = maximum;
    MPI_Allreduce(&maximum,
                  &global_maximum,
                  1,
                  MPITypeMap<real_t>::mpi_type,
                  MPI_MAX,
                  context.comm);
    maximum = global_maximum;
    ++mpi_reductions;
  }
#else
  (void)context;
  (void)mpi_reductions;
#endif
  return maximum;
}

} // namespace

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
  real_t velocity_density_floor = igrlab::FluxDefaults::kRhoVelFloor;
  bool invariant_domain_flux = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  bool frozen_sigma_spectral_speed = false;
  bool llf_volume_active = false;
  bool fused_residual_logged = false;
  int overintegration_order = -1;
  long overintegration_nodes_sequence = -1;
  DenseMatrix volume_interpolation, volume_weak_derivative;
  DenseMatrix volume_quadrature_metrics, volume_quadrature_states, volume_quadrature_flux;
  Vector volume_quadrature_sigma, volume_quadrature_speeds, volume_quadrature_max;
  DenseMatrix subcell_modal_transform;
  Vector subcell_blend;
  Vector subcell_visits;
  std::uint64_t subcell_calls = 0;
  bool subcell_log_summary = false;
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
  Array<int> boundary_kinds;
  DenseMatrix boundary_reference_states;
  Vector boundary_gammas;
  Vector boundary_outlet_pressures;
  Vector boundary_density_floors;
  Vector boundary_pressure_floors;
  Array<int> row_face_offsets;
  Array<int> row_face_ids;
  Array<int> row_face_signs;
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
  DenseMatrix volume_pair_dissipation;
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
  Array<double> state_scan_slots;
  Array<double> state_scan_result;
  Vector empty_face_neighbor;
  void FinishSubcellBlend()
  {
    if (subcell_calls && subcell_log_summary)
    {
      const real_t* visits = subcell_visits.HostRead();
      double full_dg = 0.0, blended = 0.0;
      for (int e = 0; e < ne; ++e)
      {
        full_dg += visits[e];
        blended += visits[ne + e];
      }
      std::cout << "[dgsem-subcell-blend-summary] local_elements=" << ne
                << " rhs_calls=" << subcell_calls
                << " fully_dg_fraction=" << full_dg / (ne * double(subcell_calls))
                << " mean_blend=" << blended / (ne * double(subcell_calls))
                << " native_order=" << n1d - 1 << std::endl;
    }
    subcell_calls = 0;
    subcell_modal_transform.SetSize(0);
    subcell_blend.SetSize(0);
    subcell_visits.SetSize(0);
  }
  ~DGSEMProductionCache() { FinishSubcellBlend(); }
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

void PrepareOverintegratedVolume(DGSEMProductionCache& c,
                                 FiniteElementSpace& vfes, int requested)
{
  Mesh* mesh=vfes.GetMesh();
  if(c.overintegration_order==requested&&
     c.overintegration_nodes_sequence==mesh->GetNodesSequence())return;
  const auto start=std::chrono::steady_clock::now();
  MFEM_VERIFY(sizeof(real_t)==sizeof(double)&&mesh->Dimension()==2&&
                mesh->SpaceDimension()==2&&!mesh->Nonconforming(),
              "Overintegrated DGSEM volume currently requires conforming affine FP64 2D quadrilaterals.");
  const auto& fe=*vfes.GetFE(0);const int p=fe.GetOrder();
  MFEM_VERIFY(p>=1&&p<=7,"Overintegrated DGSEM volume supports p1 through p7.");
  if(c.sigma_field)
  {
    MFEM_VERIFY(c.sigma_fes&&dynamic_cast<const L2_FECollection*>(c.sigma_fes->FEColl())&&
                  c.sigma_ldof==c.dof&&c.sigma_fes->GetFE(0)->GetOrder()==p,
                "Overintegrated DGSEM volume requires current same-degree L2 sigma.");
  }
  if(mesh->GetNodes())
    MFEM_VERIFY(mesh->GetNodes()->FESpace()->GetMaxElementOrder()<=p,
                "Overintegrated DGSEM volume requires geometry degree <= state degree.");
  const auto& ir=IntRules.Get(Geometry::SQUARE,std::max(2*p,requested));
  const int nq=ir.GetNPoints();
  MFEM_VERIFY(std::size_t(c.ne)*nq<=std::size_t(std::numeric_limits<int>::max()),
              "Overintegrated volume exceeds MFEM row indexing.");
  const int qrows=c.ne*nq;
  overintegration_internal::BuildVolumeQuadratureMaps2D(fe,ir,c.volume_interpolation,c.volume_weak_derivative);
  c.volume_quadrature_metrics.SetSize(qrows,4);
  c.volume_quadrature_metrics.GetMemory().UseDevice(true);
  real_t* qmetric=c.volume_quadrature_metrics.HostWrite();const real_t* metric=c.metrics.HostRead();
  for(int e=0;e<c.ne;++e)
  {
    auto* tr=mesh->GetElementTransformation(e);
    for(int q=0;q<nq;++q)
    {
      tr->SetIntPoint(&ir.IntPoint(q));const auto& jac=tr->Jacobian();
      const real_t current[4]={jac(1,1),-jac(0,1),-jac(1,0),jac(0,0)};
      MFEM_VERIFY(tr->Weight()>0.,"Overintegrated volume requires positive Jacobians.");
      for(int k=0;k<4;++k)
      {
        const real_t value=metric[e*c.dof+c.rows*k];
        MFEM_VERIFY(std::abs(current[k]-value)<=1e-11*std::max(real_t(1.),std::abs(value)),
                    "Overintegrated DGSEM volume requires affine geometry; curved Jacobian detected.");
        qmetric[e*nq+q+qrows*k]=value;
      }
    }
    for(int i=0;i<c.dof;++i)for(int k=0;k<4;++k)
      MFEM_VERIFY(std::abs(metric[e*c.dof+i+c.rows*k]-metric[e*c.dof+c.rows*k])<=
                    1e-11*std::max(real_t(1.),std::abs(metric[e*c.dof+c.rows*k])),
                  "Overintegrated volume requires constant contravariant nodal metrics.");
  }
  c.volume_quadrature_states.SetSize(qrows,4);c.volume_quadrature_states.GetMemory().UseDevice(true);
  c.volume_quadrature_flux.SetSize(qrows,8);c.volume_quadrature_flux.GetMemory().UseDevice(true);
  c.volume_quadrature_sigma.SetSize(qrows);c.volume_quadrature_sigma.UseDevice(true);
  c.volume_quadrature_speeds.SetSize(qrows);c.volume_quadrature_speeds.UseDevice(true);
  c.volume_quadrature_max.SetSize(1);c.volume_quadrature_max.UseDevice(true);
  c.overintegration_order=requested;c.overintegration_nodes_sequence=mesh->GetNodesSequence();
  c.workspace_evidence=DGSEMProductionWorkspaceEvidence{};
  std::cout<<"[dgsem-overintegrated-volume] enabled=1 precision=fp64 elements="<<c.ne
           <<" order="<<p<<" requested_quadrature_order="<<requested
           <<" effective_quadrature_order="<<ir.GetOrder()<<" quadrature_points="<<nq
           <<" input_nodes="<<c.dof<<" input_columns=5 projected_columns=4"
           <<" dense_products_per_rhs="<<13*std::size_t(c.ne)*c.dof*nq
           <<" padded_products_per_rhs=0 flux_points_per_rhs="<<qrows
           <<" map_geometry_bytes="<<(3*std::size_t(c.dof)*nq+4*std::size_t(qrows))*sizeof(real_t)
           <<" additional_workspace_bytes="<<(14*std::size_t(qrows)+1)*sizeof(real_t)
           <<" setup_s="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
           <<" mass=lgl faces=lgl-rusanov volume_admission=quadrature-and-nodes affine=1"
           <<" limiter_consumer_points=unchanged new_point_failure=reject-before-output"<<std::endl;
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
  AppendMatrixWorkspace(cache.volume_pair_dissipation, evidence);
  AppendVectorWorkspace(cache.nodal_speeds, evidence);
  AppendMatrixWorkspace(cache.weak_volume, evidence);
  if(cache.volume_quadrature_states.Height()>0)
  {
    AppendMatrixWorkspace(cache.volume_quadrature_states,evidence);
    AppendMatrixWorkspace(cache.volume_quadrature_flux,evidence);
    AppendVectorWorkspace(cache.volume_quadrature_sigma,evidence);
    AppendVectorWorkspace(cache.volume_quadrature_speeds,evidence);
    AppendVectorWorkspace(cache.volume_quadrature_max,evidence);
  }

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
  evidence.size_signature = HashCombine(
    evidence.size_signature,
    static_cast<std::uint64_t>(cache.state_scan_slots.Size()));
  evidence.capacity_signature = HashCombine(
    evidence.capacity_signature,
    static_cast<std::uint64_t>(
      cache.state_scan_slots.GetMemory().Capacity()));
  evidence.storage_signature = HashPointer(
    evidence.storage_signature,
    cache.state_scan_slots.HostRead());
  evidence.storage_signature = HashPointer(
    evidence.storage_signature,
    cache.state_scan_slots.Read(true));
  evidence.size_signature = HashCombine(
    evidence.size_signature,
    static_cast<std::uint64_t>(cache.state_scan_result.Size()));
  evidence.capacity_signature = HashCombine(
    evidence.capacity_signature,
    static_cast<std::uint64_t>(
      cache.state_scan_result.GetMemory().Capacity()));
  evidence.storage_signature = HashPointer(
    evidence.storage_signature,
    cache.state_scan_result.HostRead());
  evidence.storage_signature = HashPointer(
    evidence.storage_signature,
    cache.state_scan_result.Read(true));
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
  if (!production_ || !production_->workspaces_ready)
  {
    return {};
  }
  // Evidence is intentionally sampled only on explicit request. HostRead()
  // here may synchronize device storage, so it must never run in Mult().
  DGSEMProductionWorkspaceEvidence evidence = SnapshotWorkspaces(*production_);
  evidence.checks = production_->workspace_evidence.checks;
  evidence.failures = production_->workspace_evidence.failures;
  return evidence;
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
    production_->overintegration_order = -1;
    production_->workspaces_ready = false;
  }
  if (invalidation == DGSEMProductionInvalidation::Topology)
  {
    production_->topology_ready = false;
    production_->overintegration_order = -1;
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
             : igrlab::FluxDefaults::kRhoVelFloor;
  const bool invariant_domain_flux =
    flux_igr && flux_igr->GetInvariantDomainFlux();
  const bool sigma_aware_cfl = flux_igr && flux_igr->GetSigmaAwareCFL();
  const bool sigma_aware_rusanov =
    flux_igr && flux_igr->GetSigmaAwareRusanov();
  const bool frozen_sigma_spectral_speed =
    flux_igr && flux_igr->GetFrozenSigmaSpectralSpeed();
  const bool cuda = Device::Allows(Backend::CUDA_MASK);
  const bool llf_volume_selected =
    op_->GetDGSEMVolumeDissipation() == DGSEMVolumeDissipation::LLF;
  const int volume_quadrature_order=RequestedVolumeQuadratureOrder();
  if(volume_quadrature_order>0)
  {
    MFEM_VERIFY(collective_strict,
                "Overintegrated DGSEM volume requires strict production dispatch (DG_IGR_DGSEM_PRODUCTION_STRICT=1).");
    MFEM_VERIFY(mesh&&op_->dim==2&&context.size==1&&
                  op_->GetVolumeMode()==VolumeMode::DGSEMFluxDiff&&
                  op_->GetDGSEMVolumeFlux()==DGSEMVolumeFlux::Central&&
                  op_->GetDGSEMStabilization()==DGSEMStabilization::None&&!llf_volume_selected,
                "Overintegrated volume requires rank-one 2D central DGSEM without volume LLF/stabilization.");
    const char* fused_setting=::igrlab::GetEnv("DG_IGR_DGSEM_FUSED_RESIDUAL");
    MFEM_VERIFY(fused_setting&&std::strcmp(fused_setting,"0")==0&&
                  !::igrlab::EnvFlagEnabledDefault("DG_IGR_DGSEM_FUSED_DMMA",false)&&
                  !::igrlab::EnvFlagEnabledDefault("DG_IGR_DGSEM_SUBCELL_BLEND",false),
                "Overintegrated volume requires DG_IGR_DGSEM_FUSED_RESIDUAL=0 and no fused DMMA/subcell blend.");
  }

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
  fingerprint = HashCombine(fingerprint, invariant_domain_flux ? 1 : 0);
  fingerprint = HashCombine(fingerprint, sigma_aware_cfl ? 1 : 0);
  fingerprint = HashCombine(fingerprint, sigma_aware_rusanov ? 1 : 0);
  fingerprint = HashCombine(
    fingerprint, frozen_sigma_spectral_speed ? 1 : 0);
  fingerprint = HashCombine(fingerprint, cuda ? 1 : 0);
  if(volume_quadrature_order>0)
  {
    fingerprint=HashCombine(fingerprint,static_cast<std::uint64_t>(volume_quadrature_order));
    fingerprint=HashCombine(fingerprint,static_cast<std::uint64_t>(mesh->GetNodesSequence()));
  }


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
  semantic = HashCombine(semantic, invariant_domain_flux ? 1 : 0);
  semantic = HashCombine(semantic, sigma_aware_cfl ? 1 : 0);
  semantic = HashCombine(semantic, sigma_aware_rusanov ? 1 : 0);
  semantic = HashCombine(semantic, frozen_sigma_spectral_speed ? 1 : 0);
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
    const LoweredEulerBoundary lowered =
      LowerEulerBoundary(boundary_state);
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
    fingerprint = HashPointer(fingerprint, lowered.reference_state);
    fingerprint = HashPointer(fingerprint, boundary_numerical_flux);
    fingerprint = HashPointer(fingerprint, boundary_physical_flux);
    fingerprint = HashCombine(fingerprint,
                              marker ? marker->Size() : UINT64_C(0));
    fingerprint = HashCombine(
      fingerprint,
      static_cast<std::uint64_t>(static_cast<int>(lowered.kind)));
    fingerprint = HashCombine(
      fingerprint,
      static_cast<std::uint64_t>(lowered.dimension));
    fingerprint = HashReal(fingerprint, lowered.gamma);
    fingerprint = HashReal(fingerprint, lowered.pressure_out);
    fingerprint = HashReal(fingerprint, lowered.density_floor);
    fingerprint = HashReal(fingerprint, lowered.pressure_floor);
    semantic = HashCombine(semantic, boundary ? 1 : 0);
    semantic = HashCombine(
      semantic,
      static_cast<std::uint64_t>(static_cast<int>(lowered.kind)));
    semantic = HashCombine(
      semantic,
      static_cast<std::uint64_t>(lowered.dimension));
    semantic = HashReal(semantic, lowered.gamma);
    semantic = HashReal(semantic, lowered.pressure_out);
    semantic = HashReal(semantic, lowered.density_floor);
    semantic = HashReal(semantic, lowered.pressure_floor);
    semantic = HashCombine(
      semantic,
      boundary_numerical_flux == op_->num_flux_ ? 1 : 0);
    semantic = HashCombine(
      semantic,
      boundary_physical_flux == flux_function ? 1 : 0);
    semantic = HashCombine(semantic, marker ? marker->Size() : 0);
    if (lowered.reference_state)
    {
      fingerprint = HashCombine(
        fingerprint,
        static_cast<std::uint64_t>(lowered.reference_state->Size()));
      semantic = HashCombine(
        semantic,
        static_cast<std::uint64_t>(lowered.reference_state->Size()));
      for (int equation = 0;
           equation < lowered.reference_state->Size();
           ++equation)
      {
        fingerprint = HashReal(
          fingerprint, (*lowered.reference_state)(equation));
        semantic = HashReal(
          semantic, (*lowered.reference_state)(equation));
      }
    }
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
    c.selection.frozen_sigma_spectral_speed =
      frozen_sigma_spectral_speed;
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
             DGSEMVolumeDissipation::None &&
           !llf_volume_selected)
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
      (invariant_domain_flux || sigma_aware_cfl || sigma_aware_rusanov ||
       (frozen_sigma_spectral_speed && !sigma_field)))
  {
    policy_reason = DGSEMProductionReason::UnsupportedSigmaSpeedPolicy;
  }
  if (policy_reason == DGSEMProductionReason::None &&
      surface_type != static_cast<int>(DGSEMSurfaceFluxType::Rusanov))
  {
    policy_reason = DGSEMProductionReason::UnsupportedSurfaceFlux;
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
        const LoweredEulerBoundary boundary =
          LowerEulerBoundary(&selected->GetBoundaryState());
        if (boundary.kind == detail::EulerBoundaryStateKind::unsupported)
        {
          policy_reason = DGSEMProductionReason::UnsupportedBoundary;
        }
        else if (!IsSupportedEulerBoundary(
                   boundary, op_->dim, op_->num_equations))
        {
          policy_reason = DGSEMProductionReason::BoundaryDimensionMismatch;
        }
        else if (!Finite(boundary.density_floor) ||
                 !(boundary.density_floor > 0.0))
        {
          policy_reason = DGSEMProductionReason::InvalidDensityFloor;
        }
        else if (!Finite(boundary.gamma) || !(boundary.gamma > 1.0))
        {
          policy_reason = DGSEMProductionReason::InvalidGamma;
        }
        else if (!Finite(boundary.pressure_out) ||
                 !Finite(boundary.pressure_floor) ||
                 !(boundary.pressure_floor > 0.0))
        {
          policy_reason = DGSEMProductionReason::InvalidPressure;
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
    if (topology_order < 1 || topology_order > MaximumDGSEMProductionOrder())
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
    c.overintegration_order = -1;
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
             : igrlab::FluxDefaults::kRhoVelFloor;
  const bool cuda = Device::Allows(Backend::CUDA_MASK);
  const long vfes_sequence = op_->vfes.GetSequence();
  const long mesh_sequence = mesh ? mesh->GetSequence() : -1;
  const FiniteElementSpace* sigma_fes =
    sigma_field ? sigma_field->FESpace() : nullptr;
  const long sigma_sequence = sigma_fes ? sigma_fes->GetSequence() : -1;
  const bool invariant_domain_flux =
    flux_igr && flux_igr->GetInvariantDomainFlux();
  const bool sigma_aware_cfl = flux_igr && flux_igr->GetSigmaAwareCFL();
  const bool sigma_aware_rusanov =
    flux_igr && flux_igr->GetSigmaAwareRusanov();
  const bool frozen_sigma_spectral_speed =
    flux_igr && flux_igr->GetFrozenSigmaSpectralSpeed();
  const bool llf_volume_selected =
    op_->GetDGSEMVolumeDissipation() == DGSEMVolumeDissipation::LLF;

  const bool sigma_metadata_changed =
    c.sigma_field != sigma_field || c.sigma_fes != sigma_fes ||
    c.sigma_fes_sequence != sigma_sequence;
  const bool topology_policy_changed =
    c.policy_bound && c.policy_fingerprint != fingerprint;
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
    c.invariant_domain_flux != invariant_domain_flux ||
    c.sigma_aware_cfl != sigma_aware_cfl ||
    c.sigma_aware_rusanov != sigma_aware_rusanov ||
    c.frozen_sigma_spectral_speed != frozen_sigma_spectral_speed;

  if (c.topology_ready &&
      (c.vfes_sequence != vfes_sequence || c.mesh_sequence != mesh_sequence ||
       topology_policy_changed))
  {
    c.topology_ready = false;
    c.overintegration_order = -1;
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
  c.invariant_domain_flux = invariant_domain_flux;
  c.sigma_aware_cfl = sigma_aware_cfl;
  c.sigma_aware_rusanov = sigma_aware_rusanov;
  c.frozen_sigma_spectral_speed = frozen_sigma_spectral_speed;
  c.selection = DGSEMProductionSelection{};
  c.selection.strict = strict;
  c.selection.cuda = cuda;
  c.selection.uses_sigma = sigma_field != nullptr;
  c.selection.frozen_sigma_spectral_speed =
    frozen_sigma_spectral_speed;
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
  if (op_->GetDGSEMVolumeDissipation() != DGSEMVolumeDissipation::None &&
      !llf_volume_selected)
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
  if (invariant_domain_flux || sigma_aware_cfl || sigma_aware_rusanov ||
      (frozen_sigma_spectral_speed && !sigma_field))
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
  for (int b = 0; b < mesh->GetNBE(); ++b)
  {
    FaceElementTransformations* tr = mesh->GetBdrFaceTransformations(b);
    if (!tr)
    {
      continue;
    }
    const BdrHyperbolicFluxIntegrator* integrator =
      FindBdrIntegrator(tr->Attribute);
    const LoweredEulerBoundary boundary = integrator
      ? LowerEulerBoundary(&integrator->GetBoundaryState())
      : LoweredEulerBoundary{};
    if (!integrator ||
        !IsSupportedEulerBoundary(boundary, op_->dim, op_->num_equations))
    {
      return fail(DGSEMProductionReason::UnsupportedBoundary);
    }
  }

  if (!c.topology_ready)
  {
    c.FinishSubcellBlend();
    c.ne = op_->vfes.GetNE();
    c.equations = op_->num_equations;
    if (c.ne <= 0 || op_->vfes.GetOrdering() != Ordering::byNODES)
    {
      return fail(DGSEMProductionReason::UnsupportedTensorElement);
    }
    const FiniteElement& first_fe = *op_->vfes.GetFE(0);
    const int order = first_fe.GetOrder();
    if (order < 1 || order > MaximumDGSEMProductionOrder())
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

    // LGL quadrature is collocated, so the assembled inverse mass is normally
    // exactly diagonal. Inspect the supplied matrices once and retain the
    // dense representation if any off-diagonal entry is nonzero; no tolerance
    // or change to the mass discretization is involved.
    bool diagonal_inverse_mass = true;
    for (int e = 0; e < c.ne; ++e)
    {
      if (e >= static_cast<int>(op_->invmass.size()) ||
          op_->invmass[static_cast<std::size_t>(e)].Height() != c.dof ||
          op_->invmass[static_cast<std::size_t>(e)].Width() != c.dof)
      {
        // Defer rejection until after the element topology checks below, so
        // mixed-order and malformed-map diagnostics keep their precedence.
        diagonal_inverse_mass = false;
        continue;
      }
      const DenseMatrix& inverse = op_->invmass[static_cast<std::size_t>(e)];
      for (int j = 0; diagonal_inverse_mass && j < c.dof; ++j)
      {
        for (int i = 0; i < c.dof; ++i)
        {
          if (i != j && inverse(i, j) != 0.0)
          {
            diagonal_inverse_mass = false;
            break;
          }
        }
      }
    }

    c.element_vdofs.SetSize(c.rows * c.equations);
    c.inverse_mass.SetSize(c.rows, diagonal_inverse_mass ? 1 : c.dof);
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
        for (int j = 0; j < c.inverse_mass.Width(); ++j)
        {
          c.inverse_mass(e * c.dof + i, j) =
            op_->invmass[static_cast<std::size_t>(e)](
              i, diagonal_inverse_mass ? i : j);
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
    std::vector<int> boundary_kinds;
    std::vector<std::array<real_t, 4>> boundary_references;
    std::vector<real_t> boundary_gammas;
    std::vector<real_t> boundary_outlet_pressures;
    std::vector<real_t> boundary_density_floors;
    std::vector<real_t> boundary_pressure_floors;
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
                           const IntegrationPoint* neighbor_point,
                           const LoweredEulerBoundary& boundary)
    {
      left_elements.push_back(element_l);
      left_local.push_back(local_l);
      right_elements.push_back(element_r);
      right_local.push_back(local_r_value);
      ownership.push_back(static_cast<int>(owner));
      boundary_kinds.push_back(static_cast<int>(boundary.kind));
      std::array<real_t, 4> reference{{0.0, 0.0, 0.0, 0.0}};
      if (boundary.reference_state)
      {
        for (int equation = 0; equation < c.equations; ++equation)
        {
          reference[static_cast<std::size_t>(equation)] =
            (*boundary.reference_state)(equation);
        }
      }
      boundary_references.push_back(reference);
      boundary_gammas.push_back(boundary.gamma);
      boundary_outlet_pressures.push_back(boundary.pressure_out);
      boundary_density_floors.push_back(boundary.density_floor);
      boundary_pressure_floors.push_back(boundary.pressure_floor);
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
                    nullptr,
                    LoweredEulerBoundary{});
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
                      &point2,
                      LoweredEulerBoundary{});
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
      const BdrHyperbolicFluxIntegrator* boundary_integrator =
        FindBdrIntegrator(transformation->Attribute);
      const LoweredEulerBoundary boundary = boundary_integrator
        ? LowerEulerBoundary(&boundary_integrator->GetBoundaryState())
        : LoweredEulerBoundary{};
      if (!boundary_integrator ||
          !IsSupportedEulerBoundary(
            boundary, op_->dim, c.equations))
      {
        return fail(DGSEMProductionReason::UnsupportedBoundary);
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
                    nullptr,
                    boundary);
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
    c.boundary_kinds.SetSize(nfaces);
    c.boundary_reference_states.SetSize(nfaces, c.equations);
    c.boundary_gammas.SetSize(nfaces);
    c.boundary_outlet_pressures.SetSize(nfaces);
    c.boundary_density_floors.SetSize(nfaces);
    c.boundary_pressure_floors.SetSize(nfaces);
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
      c.boundary_kinds[face] =
        boundary_kinds[static_cast<std::size_t>(face)];
      c.boundary_gammas(face) =
        boundary_gammas[static_cast<std::size_t>(face)];
      c.boundary_outlet_pressures(face) =
        boundary_outlet_pressures[static_cast<std::size_t>(face)];
      c.boundary_density_floors(face) =
        boundary_density_floors[static_cast<std::size_t>(face)];
      c.boundary_pressure_floors(face) =
        boundary_pressure_floors[static_cast<std::size_t>(face)];
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
        c.boundary_reference_states(face, equation) =
          boundary_references[static_cast<std::size_t>(face)]
                             [static_cast<std::size_t>(equation)];
        c.shared_state_vdofs[face + nfaces * equation] =
          neighbor_state_maps[static_cast<std::size_t>(face)]
                             [static_cast<std::size_t>(equation)];
      }
    }

    std::vector<int> row_counts(static_cast<std::size_t>(c.rows), 0);
    for (int face = 0; face < nfaces; ++face)
    {
      const int left_row = c.left_elements[face] * c.dof +
                           c.left_local_dofs[face];
      ++row_counts[static_cast<std::size_t>(left_row)];
      if (c.ownership[face] ==
          static_cast<int>(DGSEMDeviceFaceOwnership::Local))
      {
        const int right_row = c.right_elements[face] * c.dof +
                              c.right_local_dofs[face];
        ++row_counts[static_cast<std::size_t>(right_row)];
      }
    }
    c.row_face_offsets.SetSize(c.rows + 1);
    c.row_face_offsets[0] = 0;
    for (int row = 0; row < c.rows; ++row)
    {
      c.row_face_offsets[row + 1] =
        c.row_face_offsets[row] + row_counts[static_cast<std::size_t>(row)];
    }
    const int adjacent_faces = c.row_face_offsets[c.rows];
    c.row_face_ids.SetSize(adjacent_faces);
    c.row_face_signs.SetSize(adjacent_faces);
    std::vector<int> row_cursor(static_cast<std::size_t>(c.rows));
    for (int row = 0; row < c.rows; ++row)
    {
      row_cursor[static_cast<std::size_t>(row)] = c.row_face_offsets[row];
    }
    for (int face = 0; face < nfaces; ++face)
    {
      const int left_row = c.left_elements[face] * c.dof +
                           c.left_local_dofs[face];
      int adjacent = row_cursor[static_cast<std::size_t>(left_row)]++;
      c.row_face_ids[adjacent] = face;
      c.row_face_signs[adjacent] = -1;
      if (c.ownership[face] ==
          static_cast<int>(DGSEMDeviceFaceOwnership::Local))
      {
        const int right_row = c.right_elements[face] * c.dof +
                              c.right_local_dofs[face];
        adjacent = row_cursor[static_cast<std::size_t>(right_row)]++;
        c.row_face_ids[adjacent] = face;
        c.row_face_signs[adjacent] = 1;
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
    const DGSEMProductionReason adjacency_reason =
      ValidateRowFaceAdjacency(c.rows,
                               nfaces,
                               c.dof,
                               c.left_elements,
                               c.left_local_dofs,
                               c.right_elements,
                               c.right_local_dofs,
                               c.ownership,
                               c.row_face_offsets,
                               c.row_face_ids,
                               c.row_face_signs);
    if (adjacency_reason != DGSEMProductionReason::None)
    {
      return fail(adjacency_reason);
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
      const bool exact_nodal_gather = BuildExactScalarNodeMap(
          *scalar_fes, op_->vfes, c.dof, c.sigma_element_dofs);
      Vector shape(c.sigma_ldof);
      if (exact_nodal_gather)
      {
        c.sigma_interpolation.SetSize(0, 0);
      }
      else
      {
        c.sigma_element_dofs.SetSize(c.rows * c.sigma_ldof);
        c.sigma_interpolation.SetSize(c.rows, c.sigma_ldof);
        Array<int> scalar_dofs;
        for (int e = 0; e < c.ne; ++e)
        {
          const FiniteElement& scalar_fe = *scalar_fes->GetFE(e);
          const FiniteElement& state_fe = *op_->vfes.GetFE(e);
          if (scalar_fe.GetDof() != c.sigma_ldof)
            return fail(DGSEMProductionReason::UnsupportedSigma);
          scalar_fes->GetElementDofs(e, scalar_dofs);
          if (scalar_dofs.Size() != c.sigma_ldof)
            return fail(DGSEMProductionReason::UnsupportedSigma);
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
      if (exact_nodal_gather)
      {
        int logging_rank = 0;
#ifdef MFEM_USE_MPI
        const auto context = GetProductionCommunicator(op_->vfes);
        if (context.active) MPI_Comm_rank(context.comm, &logging_rank);
#endif
        if (logging_rank == 0)
          std::cout << "[dgsem-sigma-interpolation] mode=exact-nodal-gather"
                    << " rows=" << c.rows << " scalar_dofs=" << c.sigma_ldof
                    << " shared_faces=dense" << std::endl;
      }
    }
  }

  c.llf_volume_active = llf_volume_selected;

  if (!c.workspaces_ready)
  {
    const int nfaces = c.normals.Height();
    c.states.SetSize(c.rows, c.equations);
    c.sigma_nodes.SetSize(c.rows);
    c.contravariant_flux.SetSize(c.rows, op_->dim * c.equations);
    if (c.llf_volume_active)
    {
      c.volume_pair_dissipation.SetSize(
        DGSEMVolumePairCount(op_->dim, c.ne, c.n1d), c.equations);
    }
    else
    {
      c.volume_pair_dissipation.SetSize(0, 0);
    }
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
    c.state_scan_slots.SetSize(kDGSEMDeviceStateScanWorkspaceSize);
    c.state_scan_result.SetSize(6);
    c.empty_face_neighbor.SetSize(0);
    c.states.GetMemory().UseDevice(true);
    c.sigma_nodes.UseDevice(true);
    c.contravariant_flux.GetMemory().UseDevice(true);
    c.volume_pair_dissipation.GetMemory().UseDevice(true);
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
    c.state_scan_slots.GetMemory().UseDevice(true);
    c.state_scan_result.GetMemory().UseDevice(true);
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
  c.selection.llf_volume_active = c.llf_volume_active;
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
  bool strict,
  Vector* physical_boundary_flux,
  bool admission_only) const
{
  MFEM_VERIFY(!admission_only || physical_boundary_flux == nullptr,
              "A DGSEM admission-only scan cannot capture boundary flux.");
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
  if (admission_only)
  {
    ++c.counters.admission_calls;
  }
  else
  {
    ++c.counters.stage_calls;
  }

  const int volume_quadrature_order=RequestedVolumeQuadratureOrder();
  const bool overintegrated_volume=volume_quadrature_order>0;
  if(overintegrated_volume)
    PrepareOverintegratedVolume(c,op_->vfes,volume_quadrature_order);

  DGSEMProductionReason local_gate = DGSEMProductionReason::None;
  if (x_local.Size() != op_->vfes.GetVSize())
  {
    local_gate = DGSEMProductionReason::InputSizeMismatch;
  }
  else if (!admission_only && y_local.Size() != op_->vfes.GetVSize())
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

  // Validate owned stage data before any MPI exchange. The packed fields stay
  // resident; only the six-scalar FP64 admission summary is synchronized.
  const DGSEMDeviceStateScan owned_scan = ScanDGSEMProductionStateDevice(
    op_->dim,
    c.gamma,
    c.velocity_density_floor,
    true,
    false,
    c.states,
    c.sigma_nodes,
    c.left_states,
    c.left_sigma,
    c.right_states,
    c.right_sigma,
    c.state_scan_slots,
    c.state_scan_result);
  ++c.counters.host_synchronizations;
  const DGSEMProductionReason owned_state_reason = ReduceReason(
    context,
    owned_scan.reason,
    c.counters.stage_reason_reductions);
  if (owned_state_reason != DGSEMProductionReason::None)
  {
    result.reason = owned_state_reason;
    return result;
  }
  if (c.sigma_field)
  {
    op_->dgsem_q_sigma_telemetry_.Merge(owned_scan.q_sigma);
    if (admission_only)
    {
      result.admission_q_sigma.Merge(owned_scan.q_sigma);
    }
  }

  if(overintegrated_volume)
  {
    overintegration_internal::InterpolateVolumeState2DDevice(c.ne,c.dof,
      c.volume_interpolation,c.states,c.sigma_nodes,
      c.volume_quadrature_states,c.volume_quadrature_sigma);
    const DGSEMDeviceStateScan volume_scan=ScanDGSEMProductionStateDevice(
      op_->dim,c.gamma,c.velocity_density_floor,true,false,
      c.volume_quadrature_states,c.volume_quadrature_sigma,
      c.left_states,c.left_sigma,c.right_states,c.right_sigma,
      c.state_scan_slots,c.state_scan_result);
    ++c.counters.host_synchronizations;
    const auto volume_reason=ReduceReason(context,volume_scan.reason,c.counters.stage_reason_reductions);
    if(volume_reason!=DGSEMProductionReason::None)
    {
      result.reason=volume_reason;return result;
    }
    if(c.sigma_field)
    {
      op_->dgsem_q_sigma_telemetry_.Merge(volume_scan.q_sigma);
      if(admission_only)result.admission_q_sigma.Merge(volume_scan.q_sigma);
    }
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

  GatherEulerRusanovFacesDevice(op_->dim,
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
                                c.boundary_kinds,
                                c.boundary_reference_states,
                                c.boundary_gammas,
                                c.boundary_outlet_pressures,
                                c.boundary_density_floors,
                                c.boundary_pressure_floors,
                                c.left_states,
                                c.right_states,
                                c.left_sigma,
                                c.right_sigma);

  // Validate and qualify every numerical-flux input, including prescribed
  // boundary ghosts, after exchange and before evaluating a physical flux.
  const DGSEMDeviceStateScan face_scan = ScanDGSEMProductionStateDevice(
    op_->dim,
    c.gamma,
    c.velocity_density_floor,
    false,
    true,
    c.states,
    c.sigma_nodes,
    c.left_states,
    c.left_sigma,
    c.right_states,
    c.right_sigma,
    c.state_scan_slots,
    c.state_scan_result);
  ++c.counters.host_synchronizations;
  const DGSEMProductionReason state_reason = ReduceReason(
    context,
    face_scan.reason,
    c.counters.stage_reason_reductions);
  if (state_reason != DGSEMProductionReason::None)
  {
    result.reason = state_reason;
    return result;
  }
  if (c.sigma_field)
  {
    op_->dgsem_q_sigma_telemetry_.Merge(face_scan.q_sigma);
    if (admission_only)
    {
      result.admission_q_sigma.Merge(face_scan.q_sigma);
    }
  }

  if (admission_only)
  {
    result.admission_q_sigma = ReduceFreshQSigmaTelemetry(
      context,
      result.admission_q_sigma,
      c.counters.mpi_reductions);

    // All ranks take the same branch because q_sigma is already reduced on
    // the owning communicator. Do not evaluate the frozen-sigma acoustic
    // radius on a state that the caller must reject.
    const bool speed_admissible =
      !result.admission_q_sigma.enabled ||
      (result.admission_q_sigma.enabled &&
       result.admission_q_sigma.sample_count > 0 &&
       result.admission_q_sigma.nonpositive_count == 0 &&
       result.admission_q_sigma.nonfinite_count == 0 &&
       std::isfinite(result.admission_q_sigma.minimum_raw_q_sigma) &&
       result.admission_q_sigma.minimum_raw_q_sigma > 0.0 &&
       std::isfinite(
         result.admission_q_sigma.minimum_q_sigma_over_gamma_p) &&
       result.admission_q_sigma.minimum_q_sigma_over_gamma_p > 0.0);
    if (speed_admissible)
    {
      ComputeEulerNodalCharacteristicSpeedsDevice(
        op_->dim,
        c.gamma,
        c.velocity_density_floor,
        c.states,
        c.sigma_nodes,
        c.nodal_speeds,
        c.frozen_sigma_spectral_speed);
      ComputeEulerRusanovFaceCharacteristicSpeedsDevice(
        op_->dim,
        c.gamma,
        c.velocity_density_floor,
        c.left_states,
        c.right_states,
        c.normals,
        c.left_sigma,
        c.right_sigma,
        c.face_speeds,
        c.frozen_sigma_spectral_speed);
      result.max_characteristic_speed = ReducePackedCharacteristicSpeed(
        context,
        c.nodal_speeds,
        c.face_speeds,
        c.nodal_max,
        c.face_max,
        c.counters.host_synchronizations,
        c.counters.mpi_reductions);
      if(overintegrated_volume)
      {
        ComputeEulerNodalCharacteristicSpeedsDevice(op_->dim,c.gamma,
          c.velocity_density_floor,c.volume_quadrature_states,
          c.volume_quadrature_sigma,c.volume_quadrature_speeds,c.frozen_sigma_spectral_speed);
        result.max_characteristic_speed=std::max(result.max_characteristic_speed,
          ReducePackedCharacteristicSpeed(context,c.volume_quadrature_speeds,c.face_speeds,
            c.volume_quadrature_max,c.face_max,c.counters.host_synchronizations,c.counters.mpi_reductions));
      }
    }
    result.executed = true;
    result.reason = DGSEMProductionReason::None;
    return result;
  }

  const LGLBasis1D& basis = GetLGLBasis1D(c.selection.order);
  const bool fused_residual = ::igrlab::EnvFlagEnabledDefault(
      "DG_IGR_DGSEM_FUSED_RESIDUAL", false);
  const bool fused_dmma = ::igrlab::EnvFlagEnabledDefault(
      "DG_IGR_DGSEM_FUSED_DMMA", false);
  const bool subcell_blend = ::igrlab::EnvFlagEnabledDefault(
      "DG_IGR_DGSEM_SUBCELL_BLEND", false);
  MFEM_VERIFY(!subcell_blend || (fused_residual && c.n1d >= 4),
      "DG_IGR_DGSEM_SUBCELL_BLEND requires the fused CUDA p3--p7 path.");
  MFEM_VERIFY(!fused_dmma || fused_residual,
              "DG_IGR_DGSEM_FUSED_DMMA requires DG_IGR_DGSEM_FUSED_RESIDUAL.");
  if (fused_residual)
  {
    MFEM_VERIFY(op_->dim == 2 && c.n1d >= 2 && c.n1d <= 8 &&
        c.inverse_mass.Width() == 1 && Device::Allows(Backend::CUDA_MASK),
        "DG_IGR_DGSEM_FUSED_RESIDUAL requires CUDA 2D p1--p7 with "
        "compact diagonal mass.");
    if (!c.fused_residual_logged)
    {
      int logging_rank = 0;
#ifdef MFEM_USE_MPI
      if (context.active) MPI_Comm_rank(context.comm, &logging_rank);
#endif
      if (logging_rank == 0)
        std::cout << "[dgsem-fused-residual] enabled=1 precision=fp64"
                  << " contraction=" << (fused_dmma ? "dmma8x8x4" : "simt")
                  << " elements=" << c.ne << " order=" << c.selection.order
                  << " llf_volume=" << (c.llf_volume_active ? 1 : 0)
                  << " unique_face_flux=1 packed_output=1" << std::endl;
      c.fused_residual_logged = true;
    }
  }
  if(overintegrated_volume)
  {
    BuildEulerContravariantFluxDevice(op_->dim,c.gamma,c.velocity_density_floor,
      c.volume_quadrature_states,c.volume_quadrature_sigma,c.volume_quadrature_metrics,
      c.volume_quadrature_flux,c.volume_quadrature_speeds,c.frozen_sigma_spectral_speed);
    overintegration_internal::ApplyWeakVolumeProjection2DDevice(c.ne,c.dof,
      c.volume_weak_derivative,c.volume_quadrature_flux,c.weak_volume);
    ComputeEulerNodalCharacteristicSpeedsDevice(op_->dim,c.gamma,c.velocity_density_floor,
      c.states,c.sigma_nodes,c.nodal_speeds,c.frozen_sigma_spectral_speed);
  }
  else if (!fused_residual)
  {
    BuildEulerContravariantFluxDevice(op_->dim,
                                    c.gamma,
                                    c.velocity_density_floor,
                                    c.states,
                                    c.sigma_nodes,
                                    c.metrics,
                                    c.contravariant_flux,
                                    c.nodal_speeds,
                                    c.frozen_sigma_spectral_speed);
    ApplyCentralVolumeDevice(op_->dim,
                           c.n1d,
                           c.lex_to_local,
                           basis.Dsplit,
                           basis.weights,
                           c.contravariant_flux,
                           c.weak_volume);
    if (c.llf_volume_active)
    {
      AddEulerLLFVolumeDissipationDevice(op_->dim,
                                       c.ne,
                                       c.n1d,
                                       c.gamma,
                                       c.velocity_density_floor,
                                       c.lex_to_local,
                                       basis.Dsplit,
                                       basis.weights,
                                       c.states,
                                       c.sigma_nodes,
                                       c.metrics,
                                       c.volume_pair_dissipation,
                                       c.weak_volume);
    }
  }

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
                                  c.row_face_offsets,
                                  c.row_face_ids,
                                  c.row_face_signs,
                                  c.face_flux,
                                  c.face_speeds,
                                  c.weak_surface,
                                  c.frozen_sigma_spectral_speed,
                                  !fused_residual);
  if (physical_boundary_flux)
  {
    IntegratePhysicalBoundaryFluxDevice(c.ownership,
                                        c.face_weights,
                                        c.face_flux,
                                        *physical_boundary_flux);
  }
  if (fused_residual)
  {
    if (subcell_blend)
    {
      real_t threshold = 1e-4;
      if (const char* value = ::igrlab::GetEnv("DG_IGR_DGSEM_SUBCELL_THRESHOLD"))
      {
        char* end = nullptr;
        threshold = std::strtod(value, &end);
        MFEM_VERIFY(end != value && *end == '\0' && std::isfinite(threshold) &&
            threshold > 0.0, "Invalid modal subcell sensor threshold.");
      }
      if (c.subcell_modal_transform.Height() != c.n1d)
      {
        // The internal FV flux below uses constant contravariant metrics.
        // Restrict this research variant to affine elements explicitly.
        const real_t* metric = c.metrics.HostRead();
        for (int e = 0; e < c.ne; ++e)
          for (int component = 0; component < 4; ++component)
            for (int i = 1; i < c.dof; ++i)
              MFEM_VERIFY(std::abs(metric[e * c.dof + i + c.rows * component] -
                  metric[e * c.dof + c.rows * component]) <= 1e-12 *
                  std::max(real_t(1.0), std::abs(metric[e * c.dof + c.rows * component])),
                  "Subcell blend research variant requires affine metrics.");
        c.subcell_modal_transform.SetSize(c.n1d);
        for (int i = 0; i < c.n1d; ++i)
        {
          const real_t x = 2.0 * basis.nodes[i] - 1.0;
          real_t previous = 1.0, current = x;
          for (int k = 0; k < c.n1d; ++k)
          {
            const real_t polynomial = k == 0 ? 1.0 : current;
            c.subcell_modal_transform(k, i) = basis.weights[i] * polynomial *
                (k == c.n1d - 1 ? k : 2 * k + 1);
            if (k > 0)
            {
              const real_t next = ((2 * k + 1) * x * current - k * previous) / (k + 1);
              previous = current;
              current = next;
            }
          }
        }
        c.subcell_visits.SetSize(2 * c.ne);
        c.subcell_visits.UseDevice(true);
        c.subcell_visits = 0.0;
        int rank = 0;
#ifdef MFEM_USE_MPI
        if (context.active) MPI_Comm_rank(context.comm, &rank);
#endif
        c.subcell_log_summary = rank == 0;
        if (rank == 0)
          std::cout << "[dgsem-subcell-blend] enabled=1 threshold=" << threshold
                    << " native_order=" << c.n1d - 1
                    << " current_sigma=1 unique_macroface_flux=1 affine=1" << std::endl;
      }
      ComputeModalSubcellBlendDevice(c.ne, c.n1d, c.gamma, threshold,
          c.lex_to_local, c.subcell_modal_transform, c.states, c.subcell_blend);
      subcell_internal::AccumulateSubcellBlendVisits(c.subcell_blend, c.subcell_visits);
      ++c.subcell_calls;
    }
    ApplyFusedEulerResidualDevice(c.ne, c.n1d, c.gamma,
        c.velocity_density_floor, c.llf_volume_active,
        c.frozen_sigma_spectral_speed, c.lex_to_local, basis.Dsplit,
        basis.weights, c.states, c.sigma_nodes, c.metrics, c.inverse_mass,
        c.face_weights, c.row_face_offsets, c.row_face_ids, c.row_face_signs,
        c.face_flux, c.nodal_speeds, c.strong, fused_dmma,
        subcell_blend ? &c.subcell_blend : nullptr);
  }
  else
  {
    AddPackedMatricesDevice(c.weak_surface, c.weak_volume);
    ApplyInverseMassDevice(c.ne, c.dof, c.inverse_mass, c.weak_volume,
                           c.strong);
  }
  real_t max_speed = ReducePackedCharacteristicSpeed(
    context,
    c.nodal_speeds,
    c.face_speeds,
    c.nodal_max,
    c.face_max,
    c.counters.host_synchronizations,
    c.counters.mpi_reductions);

  if(overintegrated_volume)
    max_speed=std::max(max_speed,ReducePackedCharacteristicSpeed(
      context,c.volume_quadrature_speeds,c.face_speeds,c.volume_quadrature_max,
      c.face_max,c.counters.host_synchronizations,c.counters.mpi_reductions));

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
