// Internal-only helper/kernels for DG face quadrature projection.
// This file is intentionally included after MFEM backend headers.
#pragma once

#include "igrlab/compat.hpp"

#include "physics/igr/frozen_scalar_igr_algebra.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace igrlab
{
namespace dg_face
{

enum class DGDiffusionFaceScaling
{
  Arithmetic,
  Max,
  Sum,
  HarmonicWeighted
};

/** Return a process-local revision for one rebuilt face-connectivity pack.
 *
 * Incidence data is derived from the packed face-element array.  Counts alone
 * cannot identify that array: two reordered meshes may have identical
 * element/face counts.  A revision is assigned while the geometry is already
 * being rebuilt on the host, so an apply only needs a scalar comparison and
 * never hashes or synchronizes connectivity.
 */
inline std::uint64_t NextDGFaceConnectivityRevision()
{
  static std::atomic<std::uint64_t> next_revision{1};
  return next_revision.fetch_add(1, std::memory_order_relaxed);
}

struct FaceQFProjectInfo
{
  int dim = 0;
  int vdim = 0;
  int nd1d = 0;
  int nq1d = 0;
  int nf = 0;
  int f_vec_size = 0;
  int qf_size = 0;
};

struct InteriorFaceSideDensityProjectionInfo
{
  int dim = 0;
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  int shape_size = 0;
  int face_element_size = 0;
  int element_vdof_map_size = 0;
  int qf_size = 0;
};

struct InteriorFaceSideDensityProjectionScratch
{
  mfem::Vector shape;
  /// Mapped explicit per-face, per-side alpha; empty for scalar-alpha input.
  mfem::Vector side_alpha;
  mfem::Array<int> face_elements;
  mfem::Array<int> element_vdof_map;
};

struct InteriorFaceScalarTraceProjectionInfo
{
  int dim = 0;
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  int shape_size = 0;
  int normal_dshape_size = 0;
  int face_element_size = 0;
  int element_vdof_map_size = 0;
  int value_qf_size = 0;
  int normal_deriv_qf_size = 0;
};

struct InteriorFaceScalarTraceProjectionScratch
{
  mfem::Vector shape;
  mfem::Vector normal_dshape;
  mfem::Array<int> face_elements;
  mfem::Array<int> element_vdof_map;
};

struct InteriorFaceDGDiffusionApplyInfo
{
  int dim = 0;
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  int shape_size = 0;
  int weighted_normal_dshape_size = 0;
  int penalty_size = 0;
  int face_element_size = 0;
  int element_vdof_map_size = 0;
  /// Changes whenever BuildInteriorFaceDGDiffusionApplyData2D repacks faces.
  std::uint64_t connectivity_revision = 0;
};

// Optional private geometry storage: slots contain byte-identical copies of
// complete [side][q][dof] face blocks.  The full host table remains available
// to the original generic and diagonal paths.  This is prepared only while
// building geometry, never from a pointer/size match during an apply.
struct InteriorFaceShapeDictionary2D
{
  mfem::Vector values;
  mfem::Array<int> face_slots;
  int dof = 0, nq = 0, nf = 0, slot_count = 0;
  std::uint64_t geometry_revision = 0;

  void Reset()
  {
    values.Destroy();
    face_slots.DeleteAll();
    dof = nq = nf = slot_count = 0;
    geometry_revision = 0;
  }

  bool Matches(const InteriorFaceDGDiffusionApplyInfo& info) const
  {
    return geometry_revision != 0 &&
      geometry_revision == info.connectivity_revision &&
      dof == info.dof && nq == info.nq && nf == info.nf &&
      slot_count > 0 && slot_count < nf && face_slots.Size() == nf &&
      std::size_t(values.Size()) ==
        2u * std::size_t(nq) * std::size_t(dof) * std::size_t(slot_count);
  }
};

// Reference derivatives repeat even when physical normal derivatives do not.
// Preserve each face's FP64 geometry packet and deduplicate only byte-identical
// reference blocks.  Geometry and coefficient revisions are separate: the first
// application after a geometry rebuild retains the original dense action.
struct InteriorFaceDGDiffusionFactorization2D
{
  mfem::Vector reference_dshape; // [component][side][q][dof][slot]
  mfem::Vector normal_geometry;  // [component][side][q][face]
  mfem::Array<int> face_slots;
  int dof = 0, nq = 0, nf = 0, slot_count = 0;
  std::uint64_t geometry_revision = 0;
  std::uint64_t coefficient_revision = 0;
  DGDiffusionFaceScaling coefficient_scaling = DGDiffusionFaceScaling::Arithmetic;
  // Independently proved/deduplicated 1D packs, never keyed by DShape slots.
  mfem::Vector tensor_endpoint; // [value/derivative][i][side][slot]
  mfem::Vector tensor_tangent;  // [value/derivative][t][q][side][slot]
  mfem::Array<int> tensor_axes; // [side][slot], reference normal axis
  mfem::Array<int> tensor_face_slots;
  mfem::Array<int> tensor_lex_to_native;
  /// Inverse of the independently proved tensor/native bijection.
  mfem::Array<int> tensor_native_to_lex;
  int tensor_n = 0, tensor_slot_count = 0;
  std::uint64_t tensor_geometry_revision = 0;
  std::uint64_t tensor_inverse_geometry_revision = 0;

  void ResetTensor()
  {
    tensor_endpoint.Destroy();
    tensor_tangent.Destroy();
    tensor_axes.DeleteAll();
    tensor_face_slots.DeleteAll();
    tensor_lex_to_native.DeleteAll();
    tensor_native_to_lex.DeleteAll();
    tensor_n = tensor_slot_count = 0;
    tensor_geometry_revision = 0;
    tensor_inverse_geometry_revision = 0;
  }

  void Reset()
  {
    reference_dshape.Destroy();
    normal_geometry.Destroy();
    face_slots.DeleteAll();
    dof = nq = nf = slot_count = 0;
    geometry_revision = coefficient_revision = 0;
    coefficient_scaling = DGDiffusionFaceScaling::Arithmetic;
    ResetTensor();
  }

  bool Matches(const InteriorFaceDGDiffusionApplyInfo& info) const
  {
    return info.dim == 2 && geometry_revision != 0 &&
      geometry_revision == info.connectivity_revision &&
      dof == info.dof && nq == info.nq && nf == info.nf &&
      slot_count > 0 && slot_count < nf && face_slots.Size() == nf &&
      std::size_t(reference_dshape.Size()) ==
        4u * std::size_t(nq) * std::size_t(dof) * std::size_t(slot_count) &&
      std::size_t(normal_geometry.Size()) ==
        4u * std::size_t(nq) * std::size_t(nf);
  }

  bool TensorMatches(const InteriorFaceDGDiffusionApplyInfo& info) const
  {
    return Matches(info) && info.vdim == 1 && info.dof == 36 && info.nq == 6 &&
      tensor_n == 6 && tensor_slot_count > 0 && tensor_slot_count <= 64 &&
      tensor_geometry_revision == info.connectivity_revision &&
      tensor_endpoint.Size() == 24 * tensor_slot_count &&
      tensor_tangent.Size() == 144 * tensor_slot_count &&
      tensor_axes.Size() == 2 * tensor_slot_count &&
      tensor_face_slots.Size() == nf && tensor_lex_to_native.Size() == 36;
  }
};

struct InteriorFaceDGDiffusionApplyScratch
{
  mfem::Vector shape;
  InteriorFaceShapeDictionary2D shape_dictionary;
  InteriorFaceDGDiffusionFactorization2D normal_factorization;
  mutable mfem::Vector weighted_normal_dshape;
  mfem::Vector penalty;
  /// Geometry-only normal derivative used to refresh state-dependent qdata.
  mfem::Vector normal_dshape_base;
  /// Geometry-only [side][quadrature][face] penalty metric.
  mfem::Vector penalty_metric;
  /// Persistent copy of the current [side][quadrature][face] alpha/rho data.
  mfem::Vector side_coefficient;
  /// Optional tensor-path qdata reduction; exact dense host geometry is kept.
  bool compact_qdata = false;
  /// Optional FP64 tensor lifts; independent of coefficient qdata storage.
  bool compact_lift = false;
  /// Independent stage generation: a geometry revision is not a state revision.
  std::uint64_t coefficient_geometry_revision = 0;
  std::uint64_t coefficient_generation = 0;
  mutable std::uint64_t dense_normal_generation = 0;
  DGDiffusionFaceScaling coefficient_scaling = DGDiffusionFaceScaling::Arithmetic;
  mfem::Array<int> face_elements;
  mfem::Array<int> element_vdof_map;
  /// Fixed-order element-to-(face,side) CSR used by the CUDA row gather.
  mutable mfem::Array<int> element_face_offsets;
  mutable mfem::Array<int> element_face_ids;
  mutable mfem::Array<int> element_face_sides;
  /// Revision from which the incidence arrays were derived.
  mutable std::uint64_t incidence_connectivity_revision = 0;
  /// Per-face row contributions written without atomics before the gather.
  mutable mfem::Vector face_contribution;
  /// Compact [a/b][tangent][side][face] FP64 lifts, 24 entries per face.
  mutable mfem::Vector tensor_lift;
};

inline bool HasCurrentInteriorFaceDGDiffusionTensorData2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work)
{
  const auto& factor = work.normal_factorization;
  return factor.TensorMatches(info) &&
    factor.coefficient_revision == info.connectivity_revision &&
    std::size_t(work.side_coefficient.Size()) ==
      2u * std::size_t(info.nq) * std::size_t(info.nf) &&
    (factor.coefficient_scaling == DGDiffusionFaceScaling::Arithmetic ||
     factor.coefficient_scaling == DGDiffusionFaceScaling::Max ||
     factor.coefficient_scaling == DGDiffusionFaceScaling::Sum);
}

inline bool UsesOrdinaryCUDAForDGDiffusionFaces2D()
{
#if defined(MFEM_USE_CUDA)
  const unsigned long alternative_backends =
    mfem::Backend::OMP_MASK | mfem::Backend::RAJA_MASK |
    mfem::Backend::OCCA_MASK | mfem::Backend::CEED_MASK |
    mfem::Backend::HIP_MASK | mfem::Backend::DEBUG_DEVICE;
  return mfem::Device::Allows(mfem::Backend::CUDA) &&
    !mfem::Device::Allows(alternative_backends);
#else
  return false;
#endif
}

// This private witness is also the dispatch predicate. Its inverse-map proof
// belongs to the geometry build; applies only compare revisions and sizes.
inline bool UsesInteriorFaceDGDiffusionCompactLift2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__) && defined(MFEM_USE_DOUBLE)
  const auto& factor = work.normal_factorization;
  return work.compact_lift && UsesOrdinaryCUDAForDGDiffusionFaces2D() &&
    HasCurrentInteriorFaceDGDiffusionTensorData2D(info, work) &&
    factor.tensor_native_to_lex.Size() == 36 &&
    factor.tensor_inverse_geometry_revision == info.connectivity_revision;
#else
  (void)info;
  (void)work;
  return false;
#endif
}

// A tensor refresh may leave the full normal table from an older stage. Every
// dense reader must materialize it from exact host geometry and current Q,
// independently of whether the optional factorization is still admitted.
void EnsureInteriorFaceDGDiffusionDenseNormals2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work);

inline void PrepareInteriorFaceShapeDictionary2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const mfem::Vector& shape,
  InteriorFaceShapeDictionary2D& dictionary,
  std::size_t memory_budget = 64u * 1024u * 1024u)
{
  dictionary.Reset();
  memory_budget = std::min(memory_budget, std::size_t(64u * 1024u * 1024u));
  if (info.dof <= 0 || info.nq <= 0 || info.nf <= 1 ||
      info.connectivity_revision == 0) { return; }
  const std::size_t nf = std::size_t(info.nf);
  const std::size_t max_entries = std::size_t(std::numeric_limits<int>::max());
  if (std::size_t(info.nq) > max_entries / 2u / std::size_t(info.dof))
  { return; }
  const std::size_t block_entries = 2u * std::size_t(info.nq) * std::size_t(info.dof);
  if (block_entries > max_entries / nf ||
      std::size_t(shape.Size()) != block_entries * nf) { return; }
  if (block_entries * nf > std::numeric_limits<std::size_t>::max() / sizeof(mfem::real_t))
  { return; }
  const std::size_t block_bytes = block_entries * sizeof(mfem::real_t);
  const std::size_t full_bytes = block_bytes * nf;
  const std::size_t map_bytes = nf * sizeof(int);
  struct Entry { std::uint64_t hash; int face; };
  // Charge the sort/hash entries, representative indices and temporary slots,
  // both host/device copies of the final dictionary/map, and fixed bookkeeping.
  // No full-table copy is staged.  The original table is existing geometry.
  if (nf > std::numeric_limits<std::size_t>::max() /
             (sizeof(Entry) + 4u * sizeof(int))) { return; }
  const std::size_t staging_bytes = nf * (sizeof(Entry) + 2u * sizeof(int));
  const std::size_t overhead = 8192u;
  if (memory_budget < overhead || staging_bytes > memory_budget - overhead ||
      2u * map_bytes > memory_budget - overhead - staging_bytes) { return; }
  const std::size_t available = memory_budget - overhead - staging_bytes - 2u * map_bytes;
  try
  {
    const mfem::real_t* source = shape.HostRead();
    std::vector<Entry> entries(nf);
    std::vector<int> representatives;
    representatives.reserve(nf);
    std::vector<int> slots(nf);
    for (int face = 0; face < info.nf; ++face)
    {
      const auto* bytes = reinterpret_cast<const unsigned char*>(
        source + block_entries * std::size_t(face));
      std::uint64_t hash = UINT64_C(14695981039346656037);
      for (std::size_t b = 0; b < block_bytes; ++b)
      {
        hash ^= bytes[b];
        hash *= UINT64_C(1099511628211);
      }
      entries[face] = Entry{hash, face};
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b)
    { return a.hash < b.hash || (a.hash == b.hash && a.face < b.face); });
    for (std::size_t begin = 0; begin < nf; )
    {
      std::size_t end = begin + 1;
      while (end < nf && entries[end].hash == entries[begin].hash) { ++end; }
      const std::size_t group_slots = representatives.size();
      for (std::size_t i = begin; i < end; ++i)
      {
        const int face = entries[i].face;
        std::size_t slot = group_slots;
        for (; slot < representatives.size(); ++slot)
        {
          if (std::memcmp(source + block_entries * std::size_t(face),
                          source + block_entries * std::size_t(representatives[slot]),
                          block_bytes) == 0) { break; }
        }
        if (slot == representatives.size())
        {
          const std::size_t next_count = slot + 1;
          if (next_count > available / 2u / block_bytes ||
              next_count * block_bytes + map_bytes >= full_bytes) { return; }
          representatives.push_back(face);
        }
        slots[face] = int(slot);
      }
      begin = end;
    }
    dictionary.values.SetSize(int(block_entries * representatives.size()));
    dictionary.face_slots.SetSize(info.nf);
    mfem::real_t* values = dictionary.values.HostWrite();
    for (std::size_t slot = 0; slot < representatives.size(); ++slot)
    {
      std::memcpy(values + block_entries * slot,
                  source + block_entries * std::size_t(representatives[slot]),
                  block_bytes);
    }
    std::memcpy(dictionary.face_slots.HostWrite(), slots.data(), map_bytes);
    dictionary.values.UseDevice(true);
    dictionary.face_slots.GetMemory().UseDevice(true);
    dictionary.dof = info.dof;
    dictionary.nq = info.nq;
    dictionary.nf = info.nf;
    dictionary.slot_count = int(representatives.size());
    dictionary.geometry_revision = info.connectivity_revision;
  }
  catch (const std::bad_alloc&)
  {
    dictionary.Reset(); // Optional geometry storage never changes admission.
  }
}

struct SharedFaceDGDiffusionApplyInfo
{
  int dim = 0;
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  int face_nbr_vsize = 0;
  int shape_size = 0;
  int weighted_normal_dshape_size = 0;
  int penalty_size = 0;
  int face_element_size = 0;
  int element_vdof_map_size = 0;
  int face_nbr_dof_map_size = 0;
  /// Changes whenever BuildSharedFaceDGDiffusionApplyData2D repacks faces.
  std::uint64_t connectivity_revision = 0;
};

struct SharedFaceDGDiffusionApplyScratch
{
  mfem::Vector shape;
  mfem::Vector weighted_normal_dshape;
  mfem::Vector penalty;
  mfem::Vector normal_dshape_base;
  mfem::Vector penalty_metric;
  mfem::Vector side_coefficient;
  mfem::Array<int> face_elements;
  mfem::Array<int> element_vdof_map;
  mfem::Array<int> face_nbr_dof_map;
  /// Fixed-order local-element-to-shared-face CSR for the CUDA row gather.
  mutable mfem::Array<int> element_face_offsets;
  mutable mfem::Array<int> element_face_ids;
  /// Revision from which the incidence arrays were derived.
  mutable std::uint64_t incidence_connectivity_revision = 0;
  /// Per-shared-face local-row contributions written before the gather.
  mutable mfem::Vector face_contribution;
};

struct SharedFaceSideDensityProjectionInfo
{
  int dim = 0;
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  int face_nbr_vsize = 0;
  int shape_size = 0;
  int face_element_size = 0;
  int element_vdof_map_size = 0;
  int face_nbr_dof_map_size = 0;
  int qf_size = 0;
};

struct SharedFaceSideDensityProjectionScratch
{
  mfem::Vector shape;
  /// Copied explicit per-face, per-side alpha; empty for scalar-alpha input.
  mfem::Vector side_alpha;
  mfem::Array<int> face_elements;
  mfem::Array<int> element_vdof_map;
  mfem::Array<int> face_nbr_dof_map;
};

void LaunchInteriorFaceDGDiffusionApply2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  mfem::real_t ip_sigma,
  mfem::Vector& y);

// Add the completed face action to an existing base result, retaining the
// original zero-initialized face scratch addition. Unsupported backends
// decline without changing y; the caller owns layout/alias admission.
bool TryLaunchInteriorFaceDGDiffusionAddToBase2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  mfem::real_t ip_sigma,
  mfem::Vector& y);

// Private fixed-sweep finalizer: after the caller writes the current volume
// action into y, replace it with rhs - (volume + completed face action).
// Preflight performs no numerical kernels or logical output writes. Acquiring
// device pointers may allocate or transfer storage. Disjoint read-only aliases
// are admitted. The caller must preflight before computing
// volume and leave the same geometry, coefficients and buffers bound until
// Launch. Unsupported layouts retain the original full-operator path.
bool CanLaunchInteriorFaceDGDiffusionResidual2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  const mfem::Vector& rhs,
  const mfem::Vector& y);

void LaunchInteriorFaceDGDiffusionResidual2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  mfem::real_t ip_sigma,
  const mfem::Vector& rhs,
  mfem::Vector& y);

void LaunchSharedFaceDGDiffusionApply2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  const SharedFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  const mfem::Vector& scalar_face_nbr,
  mfem::real_t ip_sigma,
  mfem::Vector& y);

/** Refresh only the coefficient-dependent packed face data on the device.
 *
 * The shape, geometry-normal derivative, penalty metric, and connectivity are
 * immutable until the mesh/FESpace changes.  Keeping those arrays persistent
 * avoids rebuilding and retransferring the complete face operator at every
 * instantaneous-IGR stage.
 */
void LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& side_q,
  mfem::real_t kappa,
  DGDiffusionFaceScaling face_scaling);

void LaunchSharedFaceDGDiffusionCoefficientUpdate2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  SharedFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& side_q,
  mfem::real_t kappa,
  DGDiffusionFaceScaling face_scaling);

inline int ReferenceNormalComponent2D(int local_face_id)
{
  return (local_face_id == 1 || local_face_id == 3) ? 0 : 1;
}

inline int CountLocalInteriorFaces(const mfem::FiniteElementSpace& fes,
                                   const mfem::FaceQuadratureSpace& fqs)
{
  mfem::Mesh* mesh = fes.GetMesh();
  MFEM_VERIFY(mesh != nullptr, "Face count requires a valid mesh.");
  int nf = 0;
  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    const int mesh_face = fqs.GetMeshFaceIndex(f);
    const mfem::Mesh::FaceInformation face =
      mesh->GetFaceInformation(mesh_face);
    if (face.IsInterior() && !face.IsShared())
    {
      nf++;
    }
  }
  return nf;
}

inline bool SupportsInteriorFaceSideDensityProjection2D(
  const mfem::FiniteElementSpace& vfes,
  const mfem::FaceQuadratureSpace& fqs,
  std::string* reason = nullptr)
{
  auto reject = [reason](const char* why)
  {
    if (reason) { *reason = why; }
    return false;
  };
  if (reason) { reason->clear(); }

  mfem::Mesh* mesh = vfes.GetMesh();
  if (!mesh) { return reject("missing-mesh"); }
  if (mesh->Dimension() != 2) { return reject("requires-2d"); }
  if (vfes.GetVDim() < 1) { return reject("requires-density-component"); }
  if (vfes.GetNE() <= 0) { return reject("requires-elements"); }
  if (fqs.GetFaceType() != mfem::FaceType::Interior)
  {
    return reject("requires-interior-face-quadrature-space");
  }
  if (fqs.GetNE() <= 0) { return reject("requires-interior-faces"); }

  const int dof = vfes.GetFE(0)->GetDof();
  for (int e = 1; e < vfes.GetNE(); ++e)
  {
    if (vfes.GetFE(e)->GetDof() != dof)
    {
      return reject("requires-uniform-dofs");
    }
  }

  const int nq = fqs.GetIntRule(0).GetNPoints();
  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    const int mesh_face = fqs.GetMeshFaceIndex(f);
    const mfem::Mesh::FaceInformation face =
      mesh->GetFaceInformation(mesh_face);
    if (!face.IsInterior())
    {
      return reject("requires-interior-face");
    }
    if (fqs.GetIntRule(f).GetNPoints() != nq)
    {
      return reject("requires-uniform-face-quadrature");
    }
  }

  return true;
}

inline bool SupportsInteriorFaceScalarTraceProjection2D(
  const mfem::FiniteElementSpace& sfes,
  const mfem::FaceQuadratureSpace& fqs,
  std::string* reason = nullptr)
{
  std::string base_reason;
  if (!SupportsInteriorFaceSideDensityProjection2D(sfes, fqs, &base_reason))
  {
    if (reason) { *reason = base_reason; }
    return false;
  }
  if (sfes.GetVDim() != 1)
  {
    if (reason) { *reason = "requires-scalar-space"; }
    return false;
  }
  if (reason) { reason->clear(); }
  return true;
}

inline bool SupportsSharedFaceDGDiffusionApplyData2D(
  const mfem::ParFiniteElementSpace& sfes,
  const mfem::IntegrationRule& ir,
  std::string* reason = nullptr)
{
  auto reject = [reason](const char* why)
  {
    if (reason) { *reason = why; }
    return false;
  };
  if (reason) { reason->clear(); }

#ifndef MFEM_USE_MPI
  (void)sfes;
  (void)ir;
  return reject("requires-mpi");
#else
  mfem::ParMesh* pmesh = sfes.GetParMesh();
  if (!pmesh) { return reject("missing-parmesh"); }
  if (pmesh->Dimension() != 2) { return reject("requires-2d"); }
  if (sfes.GetVDim() != 1) { return reject("requires-scalar-space"); }
  if (sfes.GetNE() <= 0) { return reject("requires-elements"); }
  if (pmesh->GetNSharedFaces() <= 0)
  {
    return reject("requires-shared-faces");
  }
  if (sfes.GetOrdering() != mfem::Ordering::byNODES)
  {
    return reject("requires-by-nodes-ordering");
  }
  if (ir.GetNPoints() <= 0)
  {
    return reject("requires-face-quadrature");
  }

  const int dof = sfes.GetFE(0)->GetDof();
  for (int e = 1; e < sfes.GetNE(); ++e)
  {
    if (sfes.GetFE(e)->GetDof() != dof)
    {
      return reject("requires-uniform-local-dofs");
    }
  }

  auto& mutable_sfes = const_cast<mfem::ParFiniteElementSpace&>(sfes);
  mutable_sfes.ExchangeFaceNbrData();
  if (sfes.GetFaceNbrVSize() <= 0)
  {
    return reject("requires-face-neighbor-dofs");
  }

  mfem::Array<int> nbr_dofs;
  for (int sf = 0; sf < pmesh->GetNSharedFaces(); ++sf)
  {
    mfem::FaceElementTransformations* Tr =
      pmesh->GetSharedFaceTransformations(sf, /*fill2=*/true);
    if (!Tr || !Tr->Elem1 || !Tr->Elem2)
    {
      return reject("missing-shared-face-transformations");
    }
    if (Tr->Elem1No < 0 || Tr->Elem1No >= sfes.GetNE())
    {
      return reject("invalid-local-shared-face-element");
    }
    const int nbr_el = Tr->Elem2No - pmesh->GetNE();
    if (nbr_el < 0)
    {
      return reject("invalid-neighbor-shared-face-element");
    }
    const mfem::FiniteElement* fe2 = sfes.GetFaceNbrFE(nbr_el);
    if (!fe2)
    {
      return reject("missing-neighbor-face-element");
    }
    if (fe2->GetDof() != dof)
    {
      return reject("requires-uniform-neighbor-dofs");
    }
    if (sfes.GetFaceNbrElementVDofs(nbr_el, nbr_dofs) != nullptr)
    {
      return reject("requires-no-neighbor-dof-transformation");
    }
    if (nbr_dofs.Size() != dof)
    {
      return reject("neighbor-dof-size-mismatch");
    }
    for (int d = 0; d < nbr_dofs.Size(); ++d)
    {
      int vd = nbr_dofs[d];
      if (vd < 0) { vd = -1 - vd; }
      if (vd < 0 || vd >= sfes.GetFaceNbrVSize())
      {
        return reject("neighbor-dof-out-of-range");
      }
    }
  }

  return true;
#endif
}

inline bool SupportsSharedFaceSideDensityProjection2D(
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::IntegrationRule& ir,
  std::string* reason = nullptr)
{
  auto reject = [reason](const char* why)
  {
    if (reason) { *reason = why; }
    return false;
  };
  if (reason) { reason->clear(); }

#ifndef MFEM_USE_MPI
  (void)vfes;
  (void)ir;
  return reject("requires-mpi");
#else
  mfem::ParMesh* pmesh = vfes.GetParMesh();
  if (!pmesh) { return reject("missing-parmesh"); }
  if (pmesh->Dimension() != 2) { return reject("requires-2d"); }
  if (vfes.GetVDim() < 1) { return reject("requires-density-component"); }
  if (vfes.GetNE() <= 0) { return reject("requires-elements"); }
  if (pmesh->GetNSharedFaces() <= 0)
  {
    return reject("requires-shared-faces");
  }
  if (vfes.GetOrdering() != mfem::Ordering::byNODES)
  {
    return reject("requires-by-nodes-ordering");
  }
  if (ir.GetNPoints() <= 0)
  {
    return reject("requires-face-quadrature");
  }

  const int dof = vfes.GetFE(0)->GetDof();
  for (int e = 1; e < vfes.GetNE(); ++e)
  {
    if (vfes.GetFE(e)->GetDof() != dof)
    {
      return reject("requires-uniform-local-dofs");
    }
  }

  auto& mutable_vfes = const_cast<mfem::ParFiniteElementSpace&>(vfes);
  mutable_vfes.ExchangeFaceNbrData();
  if (vfes.GetFaceNbrVSize() <= 0)
  {
    return reject("requires-face-neighbor-dofs");
  }

  mfem::Array<int> nbr_dofs;
  for (int sf = 0; sf < pmesh->GetNSharedFaces(); ++sf)
  {
    mfem::FaceElementTransformations* Tr =
      pmesh->GetSharedFaceTransformations(sf, /*fill2=*/true);
    if (!Tr || !Tr->Elem1 || !Tr->Elem2)
    {
      return reject("missing-shared-face-transformations");
    }
    if (Tr->Elem1No < 0 || Tr->Elem1No >= vfes.GetNE())
    {
      return reject("invalid-local-shared-face-element");
    }
    const int nbr_el = Tr->Elem2No - pmesh->GetNE();
    if (nbr_el < 0)
    {
      return reject("invalid-neighbor-shared-face-element");
    }
    const mfem::FiniteElement* fe2 = vfes.GetFaceNbrFE(nbr_el);
    if (!fe2)
    {
      return reject("missing-neighbor-face-element");
    }
    if (fe2->GetDof() != dof)
    {
      return reject("requires-uniform-neighbor-dofs");
    }
    if (vfes.GetFaceNbrElementVDofs(nbr_el, nbr_dofs) != nullptr)
    {
      return reject("requires-no-neighbor-dof-transformation");
    }
    if (nbr_dofs.Size() != dof * vfes.GetVDim())
    {
      return reject("neighbor-vdof-size-mismatch");
    }
    for (int d = 0; d < nbr_dofs.Size(); ++d)
    {
      int vd = nbr_dofs[d];
      if (vd < 0) { vd = -1 - vd; }
      if (vd < 0 || vd >= vfes.GetFaceNbrVSize())
      {
        return reject("neighbor-dof-out-of-range");
      }
    }
  }

  return true;
#endif
}

inline void BuildNativeElementVDofMap(
  const mfem::FiniteElementSpace& vfes,
  mfem::Array<int>& element_vdof_map)
{
  const int ne = vfes.GetNE();
  const int dof = vfes.GetFE(0)->GetDof();
  const int vdim = vfes.GetVDim();
  element_vdof_map.SetSize(dof * vdim * ne);

  mfem::Array<int> sdofs;
  for (int e = 0; e < ne; ++e)
  {
    vfes.GetElementDofs(e, sdofs);
    MFEM_VERIFY(sdofs.Size() == dof,
                "Unexpected scalar dof count while building element map.");
    for (int c = 0; c < vdim; ++c)
    {
      for (int d = 0; d < dof; ++d)
      {
        int vdof = vfes.DofToVDof(sdofs[d], c);
        vdof = mfem::FiniteElementSpace::DecodeDof(vdof);
        element_vdof_map[d + dof * (c + vdim * e)] = vdof;
      }
    }
  }
  element_vdof_map.GetMemory().UseDevice(true);
}

/** Build a persistent, fixed-order local-element incidence map.
 *
 * The face apply used to launch one thread per face and atomically add into
 * both adjacent elements.  That is algebraically correct, but the arrival
 * order of FP64 atomic additions is not reproducible.  This CSR orders each
 * element's incidences by packed face id (and then by side for a self-face),
 * so a later row gather performs the same two-sided SIPG sum deterministically.
 */
inline void EnsureInteriorFaceElementIncidence2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work)
{
  const int incidence_count = 2 * info.nf;
  if (info.connectivity_revision != 0 &&
      work.incidence_connectivity_revision == info.connectivity_revision &&
      work.element_face_offsets.Size() == info.ne + 1 &&
      work.element_face_ids.Size() == incidence_count &&
      work.element_face_sides.Size() == incidence_count)
  {
    return;
  }

  work.element_face_offsets.SetSize(info.ne + 1);
  work.element_face_ids.SetSize(incidence_count);
  work.element_face_sides.SetSize(incidence_count);
  int* offsets = work.element_face_offsets.HostWrite();
  std::fill(offsets, offsets + info.ne + 1, 0);
  const auto faces =
    mfem::Reshape(work.face_elements.HostRead(), 2, info.nf);
  for (int face = 0; face < info.nf; ++face)
  {
    for (int side = 0; side < 2; ++side)
    {
      const int element = faces(side, face);
      MFEM_VERIFY(element >= 0 && element < info.ne,
                  "Interior exact-face incidence received an invalid "
                  "element.");
      ++offsets[element + 1];
    }
  }
  for (int element = 0; element < info.ne; ++element)
  {
    offsets[element + 1] += offsets[element];
  }
  MFEM_VERIFY(offsets[info.ne] == incidence_count,
              "Interior exact-face incidence count mismatch.");

  std::vector<int> cursor(offsets, offsets + info.ne);
  int* face_ids = work.element_face_ids.HostWrite();
  int* face_sides = work.element_face_sides.HostWrite();
  for (int face = 0; face < info.nf; ++face)
  {
    for (int side = 0; side < 2; ++side)
    {
      const int element = faces(side, face);
      const int slot = cursor[static_cast<std::size_t>(element)]++;
      face_ids[slot] = face;
      face_sides[slot] = side;
    }
  }
  work.element_face_offsets.GetMemory().UseDevice(true);
  work.element_face_ids.GetMemory().UseDevice(true);
  work.element_face_sides.GetMemory().UseDevice(true);
  work.incidence_connectivity_revision = info.connectivity_revision;
}

inline void EnsureSharedFaceElementIncidence2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  const SharedFaceDGDiffusionApplyScratch& work)
{
  if (info.connectivity_revision != 0 &&
      work.incidence_connectivity_revision == info.connectivity_revision &&
      work.element_face_offsets.Size() == info.ne + 1 &&
      work.element_face_ids.Size() == info.nf)
  {
    return;
  }

  work.element_face_offsets.SetSize(info.ne + 1);
  work.element_face_ids.SetSize(info.nf);
  int* offsets = work.element_face_offsets.HostWrite();
  std::fill(offsets, offsets + info.ne + 1, 0);
  const int* faces = work.face_elements.HostRead();
  for (int face = 0; face < info.nf; ++face)
  {
    const int element = faces[face];
    MFEM_VERIFY(element >= 0 && element < info.ne,
                "Shared exact-face incidence received an invalid local "
                "element.");
    ++offsets[element + 1];
  }
  for (int element = 0; element < info.ne; ++element)
  {
    offsets[element + 1] += offsets[element];
  }
  MFEM_VERIFY(offsets[info.ne] == info.nf,
              "Shared exact-face incidence count mismatch.");

  std::vector<int> cursor(offsets, offsets + info.ne);
  int* face_ids = work.element_face_ids.HostWrite();
  for (int face = 0; face < info.nf; ++face)
  {
    const int element = faces[face];
    face_ids[cursor[static_cast<std::size_t>(element)]++] = face;
  }
  work.element_face_offsets.GetMemory().UseDevice(true);
  work.element_face_ids.GetMemory().UseDevice(true);
  work.incidence_connectivity_revision = info.connectivity_revision;
}

inline InteriorFaceSideDensityProjectionInfo
BuildInteriorFaceSideDensityProjectionData2D(
  const mfem::FiniteElementSpace& vfes,
  const mfem::FaceQuadratureSpace& fqs,
  mfem::Vector& shape,
  mfem::Array<int>& face_elements,
  mfem::Array<int>& element_vdof_map)
{
  std::string reason;
  MFEM_VERIFY(SupportsInteriorFaceSideDensityProjection2D(vfes, fqs, &reason),
              "Unsupported side-density face projection: " << reason);

  InteriorFaceSideDensityProjectionInfo info;
  info.dim = 2;
  info.dof = vfes.GetFE(0)->GetDof();
  info.vdim = vfes.GetVDim();
  info.nq = fqs.GetIntRule(0).GetNPoints();
  info.nf = CountLocalInteriorFaces(vfes, fqs);
  info.ne = vfes.GetNE();
  info.shape_size = 2 * info.nq * info.dof * info.nf;
  info.face_element_size = 2 * info.nf;
  info.element_vdof_map_size = info.dof * info.vdim * info.ne;

  shape.SetSize(info.shape_size);
  shape.UseDevice(true);
  face_elements.SetSize(info.face_element_size);

  auto shape_out = mfem::Reshape(shape.HostWrite(),
                                 2,
                                 info.nq,
                                 info.dof,
                                 info.nf);
  auto face_out = mfem::Reshape(face_elements.HostWrite(), 2, info.nf);

  mfem::Mesh* mesh = vfes.GetMesh();
  mfem::Vector side_shape(info.dof);
  int of = 0;
  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    const int mesh_face = fqs.GetMeshFaceIndex(f);
    const mfem::Mesh::FaceInformation face =
      mesh->GetFaceInformation(mesh_face);
    if (face.IsShared())
    {
      continue;
    }
    mfem::FaceElementTransformations* Tr =
      mesh->GetFaceElementTransformations(mesh_face);
    MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                "Interior side-density projection requires two elements.");
    face_out(0, of) = Tr->Elem1No;
    face_out(1, of) = Tr->Elem2No;

    const mfem::FiniteElement& fe1 = *vfes.GetFE(Tr->Elem1No);
    const mfem::FiniteElement& fe2 = *vfes.GetFE(Tr->Elem2No);
    const mfem::IntegrationRule& ir = fqs.GetIntRule(f);
    for (int q = 0; q < info.nq; ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetAllIntPoints(&ip);

      fe1.CalcShape(Tr->GetElement1IntPoint(), side_shape);
      for (int d = 0; d < info.dof; ++d)
      {
        shape_out(0, q, d, of) = side_shape(d);
      }

      fe2.CalcShape(Tr->GetElement2IntPoint(), side_shape);
      for (int d = 0; d < info.dof; ++d)
      {
        shape_out(1, q, d, of) = side_shape(d);
      }
    }
    of++;
  }
  MFEM_VERIFY(of == info.nf,
              "Interior side-density projection packed face count mismatch.");
  face_elements.GetMemory().UseDevice(true);
  BuildNativeElementVDofMap(vfes, element_vdof_map);
  return info;
}

inline InteriorFaceScalarTraceProjectionInfo
BuildInteriorFaceScalarTraceProjectionData2D(
  const mfem::FiniteElementSpace& sfes,
  const mfem::FaceQuadratureSpace& fqs,
  mfem::Vector& shape,
  mfem::Vector& normal_dshape,
  mfem::Array<int>& face_elements,
  mfem::Array<int>& element_vdof_map)
{
  std::string reason;
  MFEM_VERIFY(SupportsInteriorFaceScalarTraceProjection2D(sfes, fqs, &reason),
              "Unsupported scalar face trace projection: " << reason);

  InteriorFaceScalarTraceProjectionInfo info;
  info.dim = 2;
  info.dof = sfes.GetFE(0)->GetDof();
  info.vdim = sfes.GetVDim();
  info.nq = fqs.GetIntRule(0).GetNPoints();
  info.nf = CountLocalInteriorFaces(sfes, fqs);
  info.ne = sfes.GetNE();
  info.shape_size = 2 * info.nq * info.dof * info.nf;
  info.normal_dshape_size = info.shape_size;
  info.face_element_size = 2 * info.nf;
  info.element_vdof_map_size = info.dof * info.vdim * info.ne;

  shape.SetSize(info.shape_size);
  normal_dshape.SetSize(info.normal_dshape_size);
  shape.UseDevice(true);
  normal_dshape.UseDevice(true);
  face_elements.SetSize(info.face_element_size);

  auto shape_out = mfem::Reshape(shape.HostWrite(),
                                 2,
                                 info.nq,
                                 info.dof,
                                 info.nf);
  auto dshape_out = mfem::Reshape(normal_dshape.HostWrite(),
                                  2,
                                  info.nq,
                                  info.dof,
                                  info.nf);
  auto face_out = mfem::Reshape(face_elements.HostWrite(), 2, info.nf);

  mfem::Mesh* mesh = sfes.GetMesh();
  mfem::Vector side_shape(info.dof);
  mfem::DenseMatrix side_dshape(info.dof, info.dim);
  int of = 0;
  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    const int mesh_face = fqs.GetMeshFaceIndex(f);
    const mfem::Mesh::FaceInformation face =
      mesh->GetFaceInformation(mesh_face);
    if (face.IsShared())
    {
      continue;
    }
    mfem::FaceElementTransformations* Tr =
      mesh->GetFaceElementTransformations(mesh_face);
    MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                "Interior scalar face trace projection requires two elements.");
    face_out(0, of) = Tr->Elem1No;
    face_out(1, of) = Tr->Elem2No;

    const int local_face_id[2] = {
      face.element[0].local_face_id,
      face.element[1].local_face_id
    };
    const mfem::FiniteElement& fe1 = *sfes.GetFE(Tr->Elem1No);
    const mfem::FiniteElement& fe2 = *sfes.GetFE(Tr->Elem2No);
    const mfem::IntegrationRule& ir = fqs.GetIntRule(f);
    for (int q = 0; q < info.nq; ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetAllIntPoints(&ip);

      fe1.CalcShape(Tr->GetElement1IntPoint(), side_shape);
      fe1.CalcDShape(Tr->GetElement1IntPoint(), side_dshape);
      const int ncomp1 = ReferenceNormalComponent2D(local_face_id[0]);
      for (int d = 0; d < info.dof; ++d)
      {
        shape_out(0, q, d, of) = side_shape(d);
        dshape_out(0, q, d, of) = side_dshape(d, ncomp1);
      }

      fe2.CalcShape(Tr->GetElement2IntPoint(), side_shape);
      fe2.CalcDShape(Tr->GetElement2IntPoint(), side_dshape);
      const int ncomp2 = ReferenceNormalComponent2D(local_face_id[1]);
      for (int d = 0; d < info.dof; ++d)
      {
        shape_out(1, q, d, of) = side_shape(d);
        dshape_out(1, q, d, of) = side_dshape(d, ncomp2);
      }
    }
    of++;
  }
  MFEM_VERIFY(of == info.nf,
              "Interior scalar face trace packed face count mismatch.");
  face_elements.GetMemory().UseDevice(true);
  BuildNativeElementVDofMap(sfes, element_vdof_map);
  return info;
}

inline InteriorFaceDGDiffusionApplyInfo
BuildInteriorFaceDGDiffusionApplyData2D(
  const mfem::FiniteElementSpace& sfes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::Vector& side_q,
  mfem::real_t kappa,
  mfem::Vector& shape,
  mfem::Vector& weighted_normal_dshape,
  mfem::Vector& penalty,
  mfem::Array<int>& face_elements,
  mfem::Array<int>& element_vdof_map,
  DGDiffusionFaceScaling face_scaling = DGDiffusionFaceScaling::Arithmetic,
  mfem::Vector* normal_dshape_base = nullptr,
  mfem::Vector* side_penalty_metric = nullptr,
  InteriorFaceShapeDictionary2D* shape_dictionary = nullptr,
  InteriorFaceDGDiffusionFactorization2D* normal_factorization = nullptr,
  std::size_t factorization_memory_budget = 256u * 1024u * 1024u)
{
  if (shape_dictionary) { shape_dictionary->Reset(); }
  if (normal_factorization) { normal_factorization->Reset(); }
  std::string reason;
  MFEM_VERIFY(SupportsInteriorFaceScalarTraceProjection2D(sfes, fqs, &reason),
              "Unsupported exact DG diffusion face apply: " << reason);

  InteriorFaceDGDiffusionApplyInfo info;
  info.dim = 2;
  info.dof = sfes.GetFE(0)->GetDof();
  info.vdim = sfes.GetVDim();
  info.nq = fqs.GetIntRule(0).GetNPoints();
  info.nf = CountLocalInteriorFaces(sfes, fqs);
  info.ne = sfes.GetNE();
  info.shape_size = 2 * info.nq * info.dof * info.nf;
  info.weighted_normal_dshape_size = info.shape_size;
  info.penalty_size = info.nq * info.nf;
  info.face_element_size = 2 * info.nf;
  info.element_vdof_map_size = info.dof * info.vdim * info.ne;
  info.connectivity_revision = NextDGFaceConnectivityRevision();

  MFEM_VERIFY(side_q.Size() == 2 * info.nq * info.nf,
              "Exact DG diffusion face apply received mismatched side qdata.");

  // Stream one reference block through a bounded dictionary while the existing
  // geometry builder already evaluates DShape.  Never stage a full extra
  // per-face DShape table.  Charge both final host/device copies and temporary
  // storage against the optional cache budget; decline to the dense path.
  auto* factor_cache = normal_factorization;
  mfem::Vector factor_reference;
  std::vector<mfem::real_t> factor_values;
  std::vector<std::uint64_t> factor_hashes;
  std::size_t factor_max_slots = 0;
  const std::size_t factor_block_entries =
    4u * std::size_t(info.nq) * std::size_t(info.dof);
  const std::size_t factor_block_bytes = factor_block_entries * sizeof(mfem::real_t);
  const std::size_t factor_geometry_entries = 4u * std::size_t(info.nq) * info.nf;
  const std::size_t factor_geometry_bytes = factor_geometry_entries * sizeof(mfem::real_t);
  const std::size_t factor_map_bytes = std::size_t(info.nf) * sizeof(int);
  const std::size_t factor_dense_bytes =
    std::size_t(info.weighted_normal_dshape_size) * sizeof(mfem::real_t);
  auto decline_factorization = [&]()
  {
    if (factor_cache) { factor_cache->Reset(); }
    factor_cache = nullptr;
    factor_reference.Destroy();
    std::vector<mfem::real_t>().swap(factor_values);
    std::vector<std::uint64_t>().swap(factor_hashes);
  };
  if (factor_cache)
  {
    const std::size_t budget = std::min(factorization_memory_budget,
                                       std::size_t(256u * 1024u * 1024u));
    // The extra face map is charged on host/device; 1 MiB bounds both final
    // copies and all staging for at most 64 independent 1D packs and FE proof.
    const std::size_t tensor_reserve = info.dof == 36 && info.nq == 6
      ? 2u * factor_map_bytes + 1024u * 1024u : 0u;
    const std::size_t fixed_bytes =
      2u * factor_geometry_bytes + 2u * factor_map_bytes + factor_block_bytes +
      tensor_reserve + 8192u;
    const std::size_t max_entries = std::size_t(std::numeric_limits<int>::max());
    if (info.dof <= 0 || info.nq <= 0 || info.nq > 16 || info.nf <= 1 ||
        face_scaling == DGDiffusionFaceScaling::HarmonicWeighted ||
        factor_geometry_entries > max_entries || factor_block_entries > max_entries ||
        fixed_bytes >= budget || factor_block_bytes == 0)
    { decline_factorization(); }
    else
    {
      factor_max_slots = std::min(std::size_t(info.nf - 1),
        (budget - fixed_bytes) / (4u * factor_block_bytes + 2u * sizeof(std::uint64_t)));
      factor_max_slots = std::min(factor_max_slots, max_entries / factor_block_entries);
      if (factor_max_slots == 0) { decline_factorization(); }
      else
      {
        try
        {
          factor_cache->normal_geometry.SetSize(int(factor_geometry_entries));
          factor_cache->face_slots.SetSize(info.nf);
          factor_reference.SetSize(int(factor_block_entries));
          factor_values.reserve(factor_max_slots * factor_block_entries);
          factor_hashes.reserve(factor_max_slots);
        }
        catch (const std::bad_alloc&) { decline_factorization(); }
      }
    }
  }
  auto factor_geometry = mfem::Reshape(
    factor_cache ? factor_cache->normal_geometry.HostWrite() : nullptr,
    2, 2, info.nq, info.nf);
  auto factor_dshape = mfem::Reshape(
    factor_cache ? factor_reference.HostWrite() : nullptr, 2, 2, info.nq, info.dof);
  int* factor_slots = factor_cache ? factor_cache->face_slots.HostWrite() : nullptr;
  auto finish_factorized_face = [&](int face)
  {
    if (!factor_cache) { return; }
    const mfem::real_t* block = factor_reference.HostRead();
    const auto* bytes = reinterpret_cast<const unsigned char*>(block);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (std::size_t b = 0; b < factor_block_bytes; ++b)
    { hash ^= bytes[b]; hash *= UINT64_C(1099511628211); }
    std::size_t slot = 0;
    for (; slot < factor_hashes.size(); ++slot)
    {
      if (factor_hashes[slot] == hash &&
          std::memcmp(block, factor_values.data() + slot * factor_block_entries,
                      factor_block_bytes) == 0) { break; }
    }
    if (slot == factor_hashes.size())
    {
      if (slot >= factor_max_slots ||
          (slot + 1u) * factor_block_bytes + factor_geometry_bytes +
            factor_map_bytes >= factor_dense_bytes)
      { decline_factorization(); return; }
      factor_hashes.push_back(hash);
      factor_values.insert(factor_values.end(), block, block + factor_block_entries);
    }
    factor_slots[face] = int(slot);
  };

  // One fixed p5/Q6 candidate: prove the actual tensor FE and its native
  // permutation, then build each side from its mapped reference points.
  bool tensor_active = factor_cache && info.dof == 36 && info.nq == 6 && info.vdim == 1;
  const auto& tensor_reference = *sfes.GetFE(0);
  const auto* tensor_fe = dynamic_cast<const mfem::NodalTensorFiniteElement*>(&tensor_reference);
  int tensor_native[36] = {};
  if (tensor_active)
  {
    tensor_active = tensor_fe && tensor_reference.GetOrder() == 5 &&
      tensor_reference.GetGeomType() == mfem::Geometry::SQUARE &&
      tensor_reference.GetMapType() == mfem::FiniteElement::VALUE &&
      tensor_reference.GetRangeType() == mfem::FiniteElement::SCALAR &&
      tensor_reference.GetNodes().Size() == 36 &&
      (tensor_fe->GetBasisType() == mfem::BasisType::GaussLegendre ||
       tensor_fe->GetBasisType() == mfem::BasisType::GaussLobatto);
    if (tensor_active)
    {
      const auto& permutation = tensor_fe->GetDofMap();
      tensor_active = permutation.Size() == 0 || permutation.Size() == 36;
      const mfem::real_t* nodes = mfem::poly1d.GetPoints(5, tensor_fe->GetBasisType());
      bool used[36] = {};
      for (int lex = 0; tensor_active && lex < 36; ++lex)
      {
        const int native = permutation.Size() ? permutation[lex] : lex;
        if (native < 0 || native >= 36 || used[native]) { tensor_active = false; break; }
        used[native] = true;
        tensor_native[lex] = native;
        const auto& point = tensor_reference.GetNodes().IntPoint(native);
        if (point.x != nodes[lex % 6] || point.y != nodes[lex / 6])
        { tensor_active = false; }
      }
      for (int e = 0; tensor_active && e < info.ne; ++e)
      { if (sfes.GetFE(e) != &tensor_reference) { tensor_active = false; } }
    }
  }
  // Staging is bounded independently of face count and of reference slots.
  constexpr int tensor_pack_entries = 24 + 144;
  mfem::real_t tensor_pack[tensor_pack_entries] = {};
  int tensor_pack_axes[2] = {};
  mfem::real_t tensor_fixed[2] = {};
  std::vector<mfem::real_t> tensor_packs;
  std::vector<int> tensor_pack_axes_all;
  mfem::Vector tensor_b(6), tensor_d(6);
  auto decline_tensor = [&]()
  {
    tensor_active = false;
    if (factor_cache) { factor_cache->ResetTensor(); }
    std::vector<mfem::real_t>().swap(tensor_packs);
    std::vector<int>().swap(tensor_pack_axes_all);
  };
  if (tensor_active)
  {
    try
    {
      factor_cache->tensor_lex_to_native.SetSize(36);
      std::memcpy(factor_cache->tensor_lex_to_native.HostWrite(), tensor_native, sizeof(tensor_native));
      factor_cache->tensor_native_to_lex.SetSize(36);
      int* inverse = factor_cache->tensor_native_to_lex.HostWrite();
      for (int lex = 0; lex < 36; ++lex) { inverse[tensor_native[lex]] = lex; }
      factor_cache->tensor_face_slots.SetSize(info.nf);
      tensor_packs.reserve(64 * tensor_pack_entries);
      tensor_pack_axes_all.reserve(128);
    }
    catch (const std::bad_alloc&) { decline_tensor(); }
  }
  auto prepare_tensor_side = [&](int side, int q, const mfem::IntegrationPoint& point,
                                  const mfem::Vector& exact_shape,
                                  const mfem::DenseMatrix& exact_dshape)
  {
    if (!factor_cache || !tensor_active) { return; }
    const int axis = (point.x == 0.0 || point.x == 1.0) ? 0 :
      (point.y == 0.0 || point.y == 1.0) ? 1 : -1;
    if (axis < 0) { decline_tensor(); return; }
    const mfem::real_t fixed = axis == 0 ? point.x : point.y;
    const mfem::real_t tangent = axis == 0 ? point.y : point.x;
    if (!std::isfinite(tangent)) { decline_tensor(); return; }
    if (q == 0)
    {
      tensor_pack_axes[side] = axis;
      tensor_fixed[side] = fixed;
      tensor_fe->GetBasis1D().Eval(fixed, tensor_b, tensor_d);
      for (int i = 0; i < 6; ++i)
      {
        tensor_pack[2 * (i + 6 * side)] = tensor_b(i);
        tensor_pack[1 + 2 * (i + 6 * side)] = tensor_d(i);
      }
    }
    else if (axis != tensor_pack_axes[side] || fixed != tensor_fixed[side])
    { decline_tensor(); return; }
    tensor_fe->GetBasis1D().Eval(tangent, tensor_b, tensor_d);
    for (int t = 0; t < 6; ++t)
    {
      tensor_pack[24 + 2 * (t + 6 * (q + 6 * side))] = tensor_b(t);
      tensor_pack[25 + 2 * (t + 6 * (q + 6 * side))] = tensor_d(t);
    }
    // Verify this full 1D representation against the existing FE evaluations;
    // DShape-slot equality alone cannot prove a tensor contraction layout.
    const mfem::real_t tolerance = 64 * 36 * std::numeric_limits<mfem::real_t>::epsilon();
    for (int t = 0; t < 6; ++t) for (int i = 0; i < 6; ++i)
    {
      const int native = tensor_native[axis == 0 ? i + 6 * t : t + 6 * i];
      const mfem::real_t bn = tensor_pack[2 * (i + 6 * side)];
      const mfem::real_t dn = tensor_pack[1 + 2 * (i + 6 * side)];
      const mfem::real_t reconstructed[3] =
        {bn * tensor_b(t), dn * tensor_b(t), bn * tensor_d(t)};
      const mfem::real_t expected[3] =
        {exact_shape(native), exact_dshape(native, axis), exact_dshape(native, 1 - axis)};
      for (int k = 0; k < 3; ++k)
      {
        if (!std::isfinite(reconstructed[k]) || !std::isfinite(expected[k]) ||
            std::abs(reconstructed[k] - expected[k]) >
              tolerance * std::max(mfem::real_t(1.0), std::abs(expected[k])))
        { decline_tensor(); return; }
      }
    }
  };
  auto finish_tensor_face = [&](int face)
  {
    if (!factor_cache || !tensor_active) { return; }
    const int count = int(tensor_pack_axes_all.size() / 2);
    int slot = 0;
    for (; slot < count; ++slot)
    {
      if (tensor_pack_axes_all[2 * slot] == tensor_pack_axes[0] &&
          tensor_pack_axes_all[2 * slot + 1] == tensor_pack_axes[1] &&
          std::memcmp(tensor_pack, tensor_packs.data() + slot * tensor_pack_entries,
                      sizeof(tensor_pack)) == 0) { break; }
    }
    if (slot == count)
    {
      if (count == 64) { decline_tensor(); return; }
      tensor_packs.insert(tensor_packs.end(), tensor_pack, tensor_pack + tensor_pack_entries);
      tensor_pack_axes_all.insert(tensor_pack_axes_all.end(), tensor_pack_axes, tensor_pack_axes + 2);
    }
    factor_cache->tensor_face_slots.HostWrite()[face] = slot;
  };

  shape.SetSize(info.shape_size);
  weighted_normal_dshape.SetSize(info.weighted_normal_dshape_size);
  penalty.SetSize(info.penalty_size);
  shape.UseDevice(true);
  weighted_normal_dshape.UseDevice(true);
  penalty.UseDevice(true);
  if (normal_dshape_base)
  {
    normal_dshape_base->SetSize(info.weighted_normal_dshape_size);
    normal_dshape_base->UseDevice(true);
  }
  if (side_penalty_metric)
  {
    side_penalty_metric->SetSize(2 * info.nq * info.nf);
    side_penalty_metric->UseDevice(true);
  }
  face_elements.SetSize(info.face_element_size);

  auto shape_out = mfem::Reshape(shape.HostWrite(),
                                 2,
                                 info.nq,
                                 info.dof,
                                 info.nf);
  auto ndshape_out = mfem::Reshape(weighted_normal_dshape.HostWrite(),
                                   2,
                                   info.nq,
                                   info.dof,
                                   info.nf);
  auto penalty_out = mfem::Reshape(penalty.HostWrite(), info.nq, info.nf);
  mfem::real_t* normal_base_data =
    normal_dshape_base ? normal_dshape_base->HostWrite() : nullptr;
  auto normal_base_out = mfem::Reshape(normal_base_data,
                                       2,
                                       info.nq,
                                       info.dof,
                                       info.nf);
  mfem::real_t* penalty_metric_data =
    side_penalty_metric ? side_penalty_metric->HostWrite() : nullptr;
  auto penalty_metric_out =
    mfem::Reshape(penalty_metric_data, 2, info.nq, info.nf);
  auto face_out = mfem::Reshape(face_elements.HostWrite(), 2, info.nf);
  const auto qside = mfem::Reshape(side_q.HostRead(), 2, info.nq, info.nf);

  mfem::Mesh* mesh = sfes.GetMesh();
  mfem::Vector normal(info.dim);
  mfem::Vector ni(info.dim);
  mfem::Vector nh(info.dim);
  mfem::DenseMatrix adjJ(info.dim);
  mfem::Vector side_shape(info.dof);
  mfem::DenseMatrix side_dshape(info.dof, info.dim);
  mfem::Vector side_dshape_dn(info.dof);

  int of = 0;
  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    const int mesh_face = fqs.GetMeshFaceIndex(f);
    const mfem::Mesh::FaceInformation face =
      mesh->GetFaceInformation(mesh_face);
    if (face.IsShared())
    {
      continue;
    }
    mfem::FaceElementTransformations* Tr =
      mesh->GetFaceElementTransformations(mesh_face);
    MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                "Exact DG diffusion face apply requires two elements.");
    face_out(0, of) = Tr->Elem1No;
    face_out(1, of) = Tr->Elem2No;

    const mfem::FiniteElement& fe1 = *sfes.GetFE(Tr->Elem1No);
    const mfem::FiniteElement& fe2 = *sfes.GetFE(Tr->Elem2No);
    const mfem::IntegrationRule& ir = fqs.GetIntRule(f);
    for (int q = 0; q < info.nq; ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetAllIntPoints(&ip);
      mfem::CalcOrtho(Tr->Jacobian(), normal);

      mfem::real_t penalty_metric[2] = {0.0, 0.0};
      for (int side = 0; side < 2; ++side)
      {
        mfem::ElementTransformation& elem_tr =
          (side == 0) ? *Tr->Elem1 : *Tr->Elem2;
        const mfem::IntegrationPoint& eip =
          (side == 0) ? Tr->GetElement1IntPoint()
                      : Tr->GetElement2IntPoint();
        const mfem::FiniteElement& fe = (side == 0) ? fe1 : fe2;

        fe.CalcShape(eip, side_shape);
        fe.CalcDShape(eip, side_dshape);
        prepare_tensor_side(side, q, eip, side_shape, side_dshape);

        const mfem::real_t base_weight =
          ip.weight * static_cast<mfem::real_t>(0.5) / elem_tr.Weight();
        const mfem::real_t side_q_value = qside(side, q, of);
        const mfem::real_t penalty_factor_value =
          ip.weight * (normal * normal) / elem_tr.Weight();
        ni.Set(base_weight, normal);
        mfem::CalcAdjugate(elem_tr.Jacobian(), adjJ);
        adjJ.Mult(ni, nh);
        side_dshape.Mult(nh, side_dshape_dn);
        if (factor_cache)
        {
          for (int component = 0; component < 2; ++component)
          {
            factor_geometry(component, side, q, of) = nh(component);
            for (int d = 0; d < info.dof; ++d)
            { factor_dshape(component, side, q, d) = side_dshape(d, component); }
          }
        }
        penalty_metric[side] = side_q_value * penalty_factor_value;

        if (side_penalty_metric)
        {
          penalty_metric_out(side, q, of) = penalty_factor_value;
        }

        for (int d = 0; d < info.dof; ++d)
        {
          shape_out(side, q, d, of) = side_shape(d);
          ndshape_out(side, q, d, of) =
            side_q_value * side_dshape_dn(d);
          if (normal_dshape_base)
          {
            normal_base_out(side, q, d, of) = side_dshape_dn(d);
          }
        }
      }
      const mfem::real_t metric_sum =
        penalty_metric[0] + penalty_metric[1];
      if (face_scaling == DGDiffusionFaceScaling::HarmonicWeighted &&
          metric_sum > 0.0)
      {
        const mfem::real_t weight[2] = {
          penalty_metric[1] / metric_sum,
          penalty_metric[0] / metric_sum
        };
        for (int side = 0; side < 2; ++side)
        {
          for (int d = 0; d < info.dof; ++d)
          {
            ndshape_out(side, q, d, of) *= 2.0 * weight[side];
          }
        }
      }
      mfem::real_t penalty_q = 0.5 * metric_sum;
      if (face_scaling == DGDiffusionFaceScaling::Max)
      {
        penalty_q = std::max(penalty_metric[0], penalty_metric[1]);
      }
      else if (face_scaling == DGDiffusionFaceScaling::Sum)
      {
        penalty_q = metric_sum;
      }
      else if (face_scaling == DGDiffusionFaceScaling::HarmonicWeighted)
      {
        penalty_q = metric_sum > 0.0
          ? 2.0 * penalty_metric[0] * penalty_metric[1] / metric_sum
          : 0.0;
      }
      penalty_out(q, of) = kappa * penalty_q;
    }
    finish_factorized_face(of);
    finish_tensor_face(of);
    of++;
  }
  MFEM_VERIFY(of == info.nf,
              "Exact DG diffusion face apply packed face count mismatch.");
  if (factor_cache)
  {
    try
    {
      factor_cache->reference_dshape.SetSize(int(factor_values.size()));
      std::memcpy(factor_cache->reference_dshape.HostWrite(), factor_values.data(),
                  factor_values.size() * sizeof(mfem::real_t));
      factor_cache->reference_dshape.UseDevice(true);
      factor_cache->normal_geometry.UseDevice(true);
      factor_cache->face_slots.GetMemory().UseDevice(true);
      factor_cache->dof = info.dof;
      factor_cache->nq = info.nq;
      factor_cache->nf = info.nf;
      factor_cache->slot_count = int(factor_hashes.size());
      factor_cache->geometry_revision = info.connectivity_revision;
      if (tensor_active)
      {
        const int count = int(tensor_pack_axes_all.size() / 2);
        factor_cache->tensor_endpoint.SetSize(24 * count);
        factor_cache->tensor_tangent.SetSize(144 * count);
        factor_cache->tensor_axes.SetSize(2 * count);
        for (int slot = 0; slot < count; ++slot)
        {
          const auto* pack = tensor_packs.data() + slot * tensor_pack_entries;
          std::memcpy(factor_cache->tensor_endpoint.HostWrite() + 24 * slot, pack,
                      24 * sizeof(mfem::real_t));
          std::memcpy(factor_cache->tensor_tangent.HostWrite() + 144 * slot, pack + 24,
                      144 * sizeof(mfem::real_t));
        }
        std::memcpy(factor_cache->tensor_axes.HostWrite(), tensor_pack_axes_all.data(),
                    tensor_pack_axes_all.size() * sizeof(int));
        factor_cache->tensor_endpoint.UseDevice(true);
        factor_cache->tensor_tangent.UseDevice(true);
        factor_cache->tensor_axes.GetMemory().UseDevice(true);
        factor_cache->tensor_face_slots.GetMemory().UseDevice(true);
        factor_cache->tensor_lex_to_native.GetMemory().UseDevice(true);
        factor_cache->tensor_native_to_lex.GetMemory().UseDevice(true);
        factor_cache->tensor_n = 6;
        factor_cache->tensor_slot_count = count;
        factor_cache->tensor_geometry_revision = info.connectivity_revision;
        factor_cache->tensor_inverse_geometry_revision = info.connectivity_revision;
      }
    }
    catch (const std::bad_alloc&) { decline_factorization(); }
  }

  face_elements.GetMemory().UseDevice(true);
  BuildNativeElementVDofMap(sfes, element_vdof_map);
  if (shape_dictionary)
  { PrepareInteriorFaceShapeDictionary2D(info, shape, *shape_dictionary); }
  return info;
}

inline SharedFaceDGDiffusionApplyInfo
BuildSharedFaceDGDiffusionApplyData2D(
  const mfem::ParFiniteElementSpace& sfes,
  const mfem::IntegrationRule& ir,
  const mfem::Vector& side_q,
  mfem::real_t kappa,
  mfem::Vector& shape,
  mfem::Vector& weighted_normal_dshape,
  mfem::Vector& penalty,
  mfem::Array<int>& face_elements,
  mfem::Array<int>& element_vdof_map,
  mfem::Array<int>& face_nbr_dof_map,
  DGDiffusionFaceScaling face_scaling = DGDiffusionFaceScaling::Arithmetic,
  mfem::Vector* normal_dshape_base = nullptr,
  mfem::Vector* side_penalty_metric = nullptr)
{
#ifndef MFEM_USE_MPI
  (void)sfes;
  (void)ir;
  (void)side_q;
  (void)kappa;
  (void)shape;
  (void)weighted_normal_dshape;
  (void)penalty;
  (void)face_elements;
  (void)element_vdof_map;
  (void)face_nbr_dof_map;
  (void)face_scaling;
  (void)normal_dshape_base;
  (void)side_penalty_metric;
  MFEM_ABORT("Shared exact DG diffusion face data requires MFEM MPI support.");
  return {};
#else
  std::string reason;
  MFEM_VERIFY(SupportsSharedFaceDGDiffusionApplyData2D(sfes, ir, &reason),
              "Unsupported shared exact DG diffusion face data: " << reason);

  mfem::ParMesh* pmesh = sfes.GetParMesh();
  MFEM_VERIFY(pmesh != nullptr,
              "Shared exact DG diffusion face data requires ParMesh.");

  SharedFaceDGDiffusionApplyInfo info;
  info.dim = 2;
  info.dof = sfes.GetFE(0)->GetDof();
  info.vdim = sfes.GetVDim();
  info.nq = ir.GetNPoints();
  info.nf = pmesh->GetNSharedFaces();
  info.ne = sfes.GetNE();
  info.face_nbr_vsize = sfes.GetFaceNbrVSize();
  info.shape_size = 2 * info.nq * info.dof * info.nf;
  info.weighted_normal_dshape_size = info.shape_size;
  info.penalty_size = info.nq * info.nf;
  info.face_element_size = info.nf;
  info.element_vdof_map_size = info.dof * info.vdim * info.ne;
  info.face_nbr_dof_map_size = info.dof * info.vdim * info.nf;
  info.connectivity_revision = NextDGFaceConnectivityRevision();

  MFEM_VERIFY(side_q.Size() == 2 * info.nq * info.nf,
              "Shared exact DG diffusion face data received mismatched "
              "side qdata.");

  shape.SetSize(info.shape_size);
  weighted_normal_dshape.SetSize(info.weighted_normal_dshape_size);
  penalty.SetSize(info.penalty_size);
  shape.UseDevice(true);
  weighted_normal_dshape.UseDevice(true);
  penalty.UseDevice(true);
  if (normal_dshape_base)
  {
    normal_dshape_base->SetSize(info.weighted_normal_dshape_size);
    normal_dshape_base->UseDevice(true);
  }
  if (side_penalty_metric)
  {
    side_penalty_metric->SetSize(2 * info.nq * info.nf);
    side_penalty_metric->UseDevice(true);
  }
  face_elements.SetSize(info.face_element_size);
  face_nbr_dof_map.SetSize(info.face_nbr_dof_map_size);

  auto shape_out = mfem::Reshape(shape.HostWrite(),
                                 2,
                                 info.nq,
                                 info.dof,
                                 info.nf);
  auto ndshape_out = mfem::Reshape(weighted_normal_dshape.HostWrite(),
                                   2,
                                   info.nq,
                                   info.dof,
                                   info.nf);
  auto penalty_out = mfem::Reshape(penalty.HostWrite(), info.nq, info.nf);
  mfem::real_t* normal_base_data =
    normal_dshape_base ? normal_dshape_base->HostWrite() : nullptr;
  auto normal_base_out = mfem::Reshape(normal_base_data,
                                       2,
                                       info.nq,
                                       info.dof,
                                       info.nf);
  mfem::real_t* penalty_metric_data =
    side_penalty_metric ? side_penalty_metric->HostWrite() : nullptr;
  auto penalty_metric_out =
    mfem::Reshape(penalty_metric_data, 2, info.nq, info.nf);
  auto face_out = face_elements.HostWrite();
  auto nbr_out = mfem::Reshape(face_nbr_dof_map.HostWrite(),
                               info.dof,
                               info.vdim,
                               info.nf);
  const auto qside = mfem::Reshape(side_q.HostRead(), 2, info.nq, info.nf);

  auto& mutable_sfes = const_cast<mfem::ParFiniteElementSpace&>(sfes);
  mutable_sfes.ExchangeFaceNbrData();

  mfem::Vector normal(info.dim);
  mfem::Vector ni(info.dim);
  mfem::Vector nh(info.dim);
  mfem::DenseMatrix adjJ(info.dim);
  mfem::Vector side_shape(info.dof);
  mfem::DenseMatrix side_dshape(info.dof, info.dim);
  mfem::Vector side_dshape_dn(info.dof);
  mfem::Array<int> nbr_dofs;

  for (int sf = 0; sf < info.nf; ++sf)
  {
    mfem::FaceElementTransformations* Tr =
      pmesh->GetSharedFaceTransformations(sf, /*fill2=*/true);
    MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                "Shared exact DG diffusion face data requires two elements.");
    const int e1 = Tr->Elem1No;
    const int nbr_el = Tr->Elem2No - pmesh->GetNE();
    MFEM_VERIFY(nbr_el >= 0,
                "Shared exact DG diffusion face data received invalid "
                "neighbor element.");
    face_out[sf] = e1;

    MFEM_VERIFY(sfes.GetFaceNbrElementVDofs(nbr_el, nbr_dofs) == nullptr,
                "Shared exact DG diffusion face data does not yet support "
                "neighbor dof transformations.");
    MFEM_VERIFY(nbr_dofs.Size() == info.dof,
                "Shared exact DG diffusion face data neighbor dof mismatch.");
    for (int d = 0; d < info.dof; ++d)
    {
      nbr_out(d, 0, sf) = nbr_dofs[d];
    }

    const mfem::FiniteElement& fe1 = *sfes.GetFE(e1);
    const mfem::FiniteElement* fe2_ptr = sfes.GetFaceNbrFE(nbr_el);
    MFEM_VERIFY(fe2_ptr != nullptr,
                "Shared exact DG diffusion face data missing neighbor FE.");
    const mfem::FiniteElement& fe2 = *fe2_ptr;

    for (int q = 0; q < info.nq; ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetAllIntPoints(&ip);
      mfem::CalcOrtho(Tr->Jacobian(), normal);

      mfem::real_t penalty_metric[2] = {0.0, 0.0};
      for (int side = 0; side < 2; ++side)
      {
        mfem::ElementTransformation& elem_tr =
          (side == 0) ? *Tr->Elem1 : *Tr->Elem2;
        const mfem::IntegrationPoint& eip =
          (side == 0) ? Tr->GetElement1IntPoint()
                      : Tr->GetElement2IntPoint();
        const mfem::FiniteElement& fe = (side == 0) ? fe1 : fe2;

        fe.CalcShape(eip, side_shape);
        fe.CalcDShape(eip, side_dshape);

        const mfem::real_t base_weight =
          ip.weight * static_cast<mfem::real_t>(0.5) / elem_tr.Weight();
        const mfem::real_t side_q_value = qside(side, q, sf);
        const mfem::real_t penalty_factor_value =
          ip.weight * (normal * normal) / elem_tr.Weight();
        ni.Set(base_weight, normal);
        mfem::CalcAdjugate(elem_tr.Jacobian(), adjJ);
        adjJ.Mult(ni, nh);
        side_dshape.Mult(nh, side_dshape_dn);
        penalty_metric[side] = side_q_value * penalty_factor_value;

        if (side_penalty_metric)
        {
          penalty_metric_out(side, q, sf) = penalty_factor_value;
        }

        for (int d = 0; d < info.dof; ++d)
        {
          shape_out(side, q, d, sf) = side_shape(d);
          ndshape_out(side, q, d, sf) =
            side_q_value * side_dshape_dn(d);
          if (normal_dshape_base)
          {
            normal_base_out(side, q, d, sf) = side_dshape_dn(d);
          }
        }
      }
      const mfem::real_t metric_sum =
        penalty_metric[0] + penalty_metric[1];
      if (face_scaling == DGDiffusionFaceScaling::HarmonicWeighted &&
          metric_sum > 0.0)
      {
        const mfem::real_t weight[2] = {
          penalty_metric[1] / metric_sum,
          penalty_metric[0] / metric_sum
        };
        for (int side = 0; side < 2; ++side)
        {
          for (int d = 0; d < info.dof; ++d)
          {
            ndshape_out(side, q, d, sf) *= 2.0 * weight[side];
          }
        }
      }
      mfem::real_t penalty_q = 0.5 * metric_sum;
      if (face_scaling == DGDiffusionFaceScaling::Max)
      {
        penalty_q = std::max(penalty_metric[0], penalty_metric[1]);
      }
      else if (face_scaling == DGDiffusionFaceScaling::Sum)
      {
        penalty_q = metric_sum;
      }
      else if (face_scaling == DGDiffusionFaceScaling::HarmonicWeighted)
      {
        penalty_q = metric_sum > 0.0
          ? 2.0 * penalty_metric[0] * penalty_metric[1] / metric_sum
          : 0.0;
      }
      penalty_out(q, sf) = kappa * penalty_q;
    }
  }

  face_elements.GetMemory().UseDevice(true);
  face_nbr_dof_map.GetMemory().UseDevice(true);
  BuildNativeElementVDofMap(sfes, element_vdof_map);
  return info;
#endif
}

inline SharedFaceSideDensityProjectionInfo
BuildSharedFaceSideDensityProjectionData2D(
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::IntegrationRule& ir,
  mfem::Vector& shape,
  mfem::Array<int>& face_elements,
  mfem::Array<int>& element_vdof_map,
  mfem::Array<int>& face_nbr_dof_map)
{
#ifndef MFEM_USE_MPI
  (void)vfes;
  (void)ir;
  (void)shape;
  (void)face_elements;
  (void)element_vdof_map;
  (void)face_nbr_dof_map;
  MFEM_ABORT("Shared side-density projection requires MFEM MPI support.");
  return {};
#else
  std::string reason;
  MFEM_VERIFY(SupportsSharedFaceSideDensityProjection2D(vfes, ir, &reason),
              "Unsupported shared side-density face projection: " << reason);

  mfem::ParMesh* pmesh = vfes.GetParMesh();
  MFEM_VERIFY(pmesh != nullptr,
              "Shared side-density projection requires ParMesh.");

  SharedFaceSideDensityProjectionInfo info;
  info.dim = 2;
  info.dof = vfes.GetFE(0)->GetDof();
  info.vdim = vfes.GetVDim();
  info.nq = ir.GetNPoints();
  info.nf = pmesh->GetNSharedFaces();
  info.ne = vfes.GetNE();
  info.face_nbr_vsize = vfes.GetFaceNbrVSize();
  info.shape_size = 2 * info.nq * info.dof * info.nf;
  info.face_element_size = info.nf;
  info.element_vdof_map_size = info.dof * info.vdim * info.ne;
  info.face_nbr_dof_map_size = info.dof * info.vdim * info.nf;

  shape.SetSize(info.shape_size);
  shape.UseDevice(true);
  face_elements.SetSize(info.face_element_size);
  face_nbr_dof_map.SetSize(info.face_nbr_dof_map_size);

  auto shape_out = mfem::Reshape(shape.HostWrite(),
                                 2,
                                 info.nq,
                                 info.dof,
                                 info.nf);
  auto face_out = face_elements.HostWrite();
  auto nbr_out = mfem::Reshape(face_nbr_dof_map.HostWrite(),
                               info.dof,
                               info.vdim,
                               info.nf);

  auto& mutable_vfes = const_cast<mfem::ParFiniteElementSpace&>(vfes);
  mutable_vfes.ExchangeFaceNbrData();

  mfem::Vector side_shape(info.dof);
  mfem::Array<int> nbr_dofs;

  for (int sf = 0; sf < info.nf; ++sf)
  {
    mfem::FaceElementTransformations* Tr =
      pmesh->GetSharedFaceTransformations(sf, /*fill2=*/true);
    MFEM_VERIFY(Tr && Tr->Elem1 && Tr->Elem2,
                "Shared side-density projection requires two elements.");
    const int e1 = Tr->Elem1No;
    const int nbr_el = Tr->Elem2No - pmesh->GetNE();
    MFEM_VERIFY(nbr_el >= 0,
                "Shared side-density projection received invalid neighbor.");
    face_out[sf] = e1;

    MFEM_VERIFY(vfes.GetFaceNbrElementVDofs(nbr_el, nbr_dofs) == nullptr,
                "Shared side-density projection does not yet support "
                "neighbor dof transformations.");
    MFEM_VERIFY(nbr_dofs.Size() == info.dof * info.vdim,
                "Shared side-density projection neighbor vdof mismatch.");
    for (int c = 0; c < info.vdim; ++c)
    {
      for (int d = 0; d < info.dof; ++d)
      {
        nbr_out(d, c, sf) = nbr_dofs[d + info.dof * c];
      }
    }

    const mfem::FiniteElement& fe1 = *vfes.GetFE(e1);
    const mfem::FiniteElement* fe2_ptr = vfes.GetFaceNbrFE(nbr_el);
    MFEM_VERIFY(fe2_ptr != nullptr,
                "Shared side-density projection missing neighbor FE.");
    const mfem::FiniteElement& fe2 = *fe2_ptr;

    for (int q = 0; q < info.nq; ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetAllIntPoints(&ip);

      fe1.CalcShape(Tr->GetElement1IntPoint(), side_shape);
      for (int d = 0; d < info.dof; ++d)
      {
        shape_out(0, q, d, sf) = side_shape(d);
      }

      fe2.CalcShape(Tr->GetElement2IntPoint(), side_shape);
      for (int d = 0; d < info.dof; ++d)
      {
        shape_out(1, q, d, sf) = side_shape(d);
      }
    }
  }

  face_elements.GetMemory().UseDevice(true);
  face_nbr_dof_map.GetMemory().UseDevice(true);
  BuildNativeElementVDofMap(vfes, element_vdof_map);
  return info;
#endif
}

struct InteriorFaceSideDensityProjection2D
{
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  mfem::real_t alpha = 0.0;
  mfem::real_t rho_floor = 0.0;
  const mfem::real_t* shape = nullptr;
  const mfem::real_t* side_alpha = nullptr;
  const int* face_elements = nullptr;
  const int* element_vdof_map = nullptr;
  const mfem::real_t* state = nullptr;
  mfem::real_t* side_q = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto S = mfem::Reshape(shape, 2, nq, dof, nf);
    auto F = mfem::Reshape(face_elements, 2, nf);
    auto D = mfem::Reshape(element_vdof_map, dof, vdim, ne);
    auto Q = mfem::Reshape(side_q, 2, nq, nf);
    auto A = mfem::Reshape(side_alpha, 2, nf);

    MFEM_FOREACH_THREAD(qs, x, 2 * nq)
    {
      const int side = qs / nq;
      const int q = qs - side * nq;
      const int e = F(side, face);
      mfem::real_t rho = 0.0;
      for (int d = 0; d < dof; ++d)
      {
        rho += S(side, q, d, face) * state[D(d, 0, e)];
      }
      const mfem::real_t side_alpha_value = side_alpha ? A(side, face) : alpha;
      Q(side, q, face) = frozen_scalar_igr_algebra::AlphaOverDensity(
        side_alpha_value,
        rho,
        rho_floor);
    }
  }
};

struct InteriorFaceScalarTraceProjection2D
{
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  const mfem::real_t* shape = nullptr;
  const mfem::real_t* normal_dshape = nullptr;
  const int* face_elements = nullptr;
  const int* element_vdof_map = nullptr;
  const mfem::real_t* scalar = nullptr;
  mfem::real_t* value_q = nullptr;
  mfem::real_t* normal_deriv_q = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto S = mfem::Reshape(shape, 2, nq, dof, nf);
    auto DS = mfem::Reshape(normal_dshape, 2, nq, dof, nf);
    auto F = mfem::Reshape(face_elements, 2, nf);
    auto D = mfem::Reshape(element_vdof_map, dof, vdim, ne);
    auto V = mfem::Reshape(value_q, 2, nq, nf);
    auto DN = mfem::Reshape(normal_deriv_q, 2, nq, nf);

    MFEM_FOREACH_THREAD(qs, x, 2 * nq)
    {
      const int side = qs / nq;
      const int q = qs - side * nq;
      const int e = F(side, face);
      mfem::real_t value = 0.0;
      mfem::real_t normal_deriv = 0.0;
      for (int d = 0; d < dof; ++d)
      {
        const mfem::real_t dof_value = scalar[D(d, 0, e)];
        value += S(side, q, d, face) * dof_value;
        normal_deriv += DS(side, q, d, face) * dof_value;
      }
      V(side, q, face) = value;
      DN(side, q, face) = normal_deriv;
    }
  }
};

struct InteriorFaceDGDiffusionRecord2D
{
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  mfem::real_t ip_sigma = 0.0;
  const mfem::real_t* shape = nullptr;
  const mfem::real_t* weighted_normal_dshape = nullptr;
  const mfem::real_t* penalty = nullptr;
  const int* face_elements = nullptr;
  const int* element_vdof_map = nullptr;
  const mfem::real_t* x = nullptr;
  mfem::real_t* contribution = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto S = mfem::Reshape(shape, 2, nq, dof, nf);
    auto N = mfem::Reshape(weighted_normal_dshape, 2, nq, dof, nf);
    auto P = mfem::Reshape(penalty, nq, nf);
    auto F = mfem::Reshape(face_elements, 2, nf);
    auto D = mfem::Reshape(element_vdof_map, dof, vdim, ne);
    auto C = mfem::Reshape(contribution, dof, 2, nf);

    for (int side = 0; side < 2; ++side)
    {
      for (int d = 0; d < dof; ++d)
      {
        C(d, side, face) = 0.0;
      }
    }

    for (int q = 0; q < nq; ++q)
    {
      mfem::real_t value[2] = {0.0, 0.0};
      mfem::real_t normal_flux[2] = {0.0, 0.0};
      for (int side = 0; side < 2; ++side)
      {
        const int e = F(side, face);
        for (int d = 0; d < dof; ++d)
        {
          const mfem::real_t dof_value = x[D(d, 0, e)];
          value[side] += S(side, q, d, face) * dof_value;
          normal_flux[side] += N(side, q, d, face) * dof_value;
        }
      }

      const mfem::real_t jump = value[0] - value[1];
      const mfem::real_t avg_flux = normal_flux[0] + normal_flux[1];
      const mfem::real_t penalty_jump = P(q, face) * jump;

      for (int d = 0; d < dof; ++d)
      {
        const mfem::real_t y0 =
          -S(0, q, d, face) * avg_flux +
          ip_sigma * N(0, q, d, face) * jump +
          S(0, q, d, face) * penalty_jump;
        const mfem::real_t y1 =
          S(1, q, d, face) * avg_flux +
          ip_sigma * N(1, q, d, face) * jump -
          S(1, q, d, face) * penalty_jump;
        C(d, 0, face) += y0;
        C(d, 1, face) += y1;
      }
    }
  }
};

// Native CUDA warp specialization.  Each z slice in the (32,1,4) block is
// one whole warp and owns one face, including the final incomplete batch.
// The two sides' quadrature projections must fit in the 32 lanes; output
// rows are tiled so their count does not limit the supported polynomial order.
// Each side/q projection retains the original ascending-d sum, the side-pair
// terms retain their operation order, and each output retains its ascending-q
// sum.  The subsequent deterministic element incidence gather is unchanged.
template <bool FactorizedNormals = false>
struct InteriorFaceDGDiffusionRecord2DWarp
{
  InteriorFaceDGDiffusionRecord2D record;
  const int* shape_face_slots = nullptr;
  int shape_face_count = 0;
  const mfem::real_t* reference_dshape = nullptr;
  const mfem::real_t* normal_geometry = nullptr;
  const int* normal_face_slots = nullptr;
  int reference_face_count = 0;
  const mfem::real_t* side_coefficient = nullptr;

  MFEM_HOST_DEVICE static mfem::real_t ReconstructNormal(
    mfem::real_t dx, mfem::real_t dy, mfem::real_t nx, mfem::real_t ny,
    mfem::real_t coefficient)
  {
    // DenseMatrix::Mult visits its two columns in this order.  Keep the
    // coefficient product rounded before the original DOF/lift accumulation.
    // Host/device contraction can differ by FP64 roundoff; no sum is factored.
#if defined(__CUDA_ARCH__) && defined(MFEM_USE_DOUBLE)
    const mfem::real_t normal = __fma_rn(ny, dy, __dmul_rn(nx, dx));
    return __dmul_rn(coefficient, normal);
#elif defined(__CUDA_ARCH__) && defined(MFEM_USE_SINGLE)
    const mfem::real_t normal = __fmaf_rn(ny, dy, __fmul_rn(nx, dx));
    return __fmul_rn(coefficient, normal);
#else
    mfem::real_t normal = nx * dx;
    normal += ny * dy;
    return coefficient * normal;
#endif
  }

  MFEM_HOST_DEVICE void operator()(int face) const
  {
#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
    const int shape_face = shape_face_slots ? shape_face_slots[face] : face;
    auto S = mfem::Reshape(record.shape, 2, record.nq, record.dof,
                           shape_face_slots ? shape_face_count : record.nf);
    auto N = mfem::Reshape(record.weighted_normal_dshape,
                           2, record.nq, record.dof, record.nf);
    const int normal_face = FactorizedNormals ? normal_face_slots[face] : 0;
    auto RD = mfem::Reshape(reference_dshape,
                            2, 2, record.nq, record.dof, reference_face_count);
    auto G = mfem::Reshape(normal_geometry, 2, 2, record.nq, record.nf);
    auto Q = mfem::Reshape(side_coefficient, 2, record.nq, record.nf);
    auto P = mfem::Reshape(record.penalty, record.nq, record.nf);
    auto F = mfem::Reshape(record.face_elements, 2, record.nf);
    auto D = mfem::Reshape(record.element_vdof_map,
                           record.dof, record.vdim, record.ne);
    auto C = mfem::Reshape(record.contribution, record.dof, 2, record.nf);

    // Each active lane owns one side/quadrature projection.  Initialize all
    // lanes because every lane participates in each full-warp broadcast.
    mfem::real_t value = 0.0;
    mfem::real_t normal_flux = 0.0;
    if (lane < 2 * record.nq)
    {
      const int side = lane / record.nq;
      const int q = lane - side * record.nq;
      const int e = F(side, face);
      const mfem::real_t nx = FactorizedNormals ? G(0, side, q, face) : 0.0;
      const mfem::real_t ny = FactorizedNormals ? G(1, side, q, face) : 0.0;
      const mfem::real_t coefficient = FactorizedNormals ? Q(side, q, face) : 0.0;
      for (int d = 0; d < record.dof; ++d)
      {
        const mfem::real_t dof_value = record.x[D(d, 0, e)];
        value += S(side, q, d, shape_face) * dof_value;
        mfem::real_t normal_value;
        if constexpr (FactorizedNormals)
        {
          normal_value = ReconstructNormal(RD(0, side, q, d, normal_face),
            RD(1, side, q, d, normal_face), nx, ny, coefficient);
        }
        else { normal_value = N(side, q, d, face); }
        normal_flux += normal_value * dof_value;
      }
    }

    // Form the side-pair terms once on each side-zero quadrature lane.
    // Every lane participates in both full-mask broadcasts; inactive lanes
    // select the last valid side-one projection even for the Q1 boundary.
    const int other_q = lane < record.nq ? lane : record.nq - 1;
    const mfem::real_t other_value =
      __shfl_sync(0xffffffffu, value, other_q + record.nq);
    const mfem::real_t other_normal_flux =
      __shfl_sync(0xffffffffu, normal_flux, other_q + record.nq);
    mfem::real_t projected_jump = 0.0;
    mfem::real_t projected_avg_flux = 0.0;
    mfem::real_t projected_penalty_jump = 0.0;
    if (lane < record.nq)
    {
      projected_jump = value - other_value;
      projected_avg_flux = normal_flux + other_normal_flux;
      projected_penalty_jump = P(lane, face) * projected_jump;
    }

    // Keep two independent output sums per lane.  The side-pair terms are
    // broadcast once for both rows, exposing independent table loads without
    // growing the register working set with polynomial order.  Each output
    // still accumulates in ascending q, including the final incomplete group.
    constexpr int rows_per_lane = 2;
    for (int row_base = 0; row_base < 2 * record.dof;
         row_base += 32 * rows_per_lane)
    {
      mfem::real_t sums[rows_per_lane] = {0.0, 0.0};
      int sides[rows_per_lane];
      int dofs[rows_per_lane];
      bool owns_row[rows_per_lane];
#pragma unroll
      for (int i = 0; i < rows_per_lane; ++i)
      {
        const int row = row_base + lane + 32 * i;
        owns_row[i] = row < 2 * record.dof;
        sides[i] = row / record.dof;
        dofs[i] = row - sides[i] * record.dof;
      }
      for (int q = 0; q < record.nq; ++q)
      {
        const mfem::real_t jump =
          __shfl_sync(0xffffffffu, projected_jump, q);
        const mfem::real_t avg_flux =
          __shfl_sync(0xffffffffu, projected_avg_flux, q);
        const mfem::real_t penalty_jump =
          __shfl_sync(0xffffffffu, projected_penalty_jump, q);
#pragma unroll
        for (int i = 0; i < rows_per_lane; ++i)
        {
          if (owns_row[i])
          {
            const int d = dofs[i];
            const int side = sides[i];
            mfem::real_t normal_value;
            if constexpr (FactorizedNormals)
            {
              normal_value = ReconstructNormal(RD(0, side, q, d, normal_face),
                RD(1, side, q, d, normal_face), G(0, side, q, face),
                G(1, side, q, face), Q(side, q, face));
            }
            else { normal_value = N(side, q, d, face); }
            if (side == 0)
            {
              const mfem::real_t y0 =
                -S(0, q, d, shape_face) * avg_flux +
                record.ip_sigma * normal_value * jump +
                S(0, q, d, shape_face) * penalty_jump;
              sums[i] += y0;
            }
            else
            {
              const mfem::real_t y1 =
                S(1, q, d, shape_face) * avg_flux +
                record.ip_sigma * normal_value * jump -
                S(1, q, d, shape_face) * penalty_jump;
              sums[i] += y1;
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < rows_per_lane; ++i)
      {
        if (owns_row[i]) { C(dofs[i], sides[i], face) = sums[i]; }
      }
    }
#else
    record(face);
#endif
  }
};

// Fixed p5/Q6 tensor contractions: each warp owns one face.  Lanes 0..11
// hold side/t or side/q scalars; every lane executes every full-mask shuffle.
// There are no order-sized per-thread arrays and no additional quadrature weight.
struct InteriorFaceDGDiffusionTensorRecord2DWarp
{
  InteriorFaceDGDiffusionRecord2D record;
  const mfem::real_t* endpoint = nullptr;
  const mfem::real_t* tangent = nullptr;
  const int* axes = nullptr;
  const int* face_slots = nullptr;
  const int* lex_to_native = nullptr;
  const mfem::real_t* normal_geometry = nullptr;
  const mfem::real_t* side_coefficient = nullptr;
  int slot_count = 0;

  template <bool CompactLift>
  MFEM_HOST_DEVICE void Apply(int face) const
  {
#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
    const bool active = lane < 12;
    const int side = active ? lane / 6 : 0;
    const int index = active ? lane % 6 : 0;
    const int slot = face_slots[face];
    auto E = mfem::Reshape(endpoint, 2, 6, 2, slot_count);
    auto T = mfem::Reshape(tangent, 2, 6, 6, 2, slot_count);
    auto Axis = mfem::Reshape(axes, 2, slot_count);
    auto G = mfem::Reshape(normal_geometry, 2, 2, 6, record.nf);
    auto Q = mfem::Reshape(side_coefficient, 2, 6, record.nf);
    auto P = mfem::Reshape(record.penalty, 6, record.nf);
    auto F = mfem::Reshape(record.face_elements, 2, record.nf);
    auto D = mfem::Reshape(record.element_vdof_map, 36, record.vdim, record.ne);
    const int axis = Axis(side, slot);

    mfem::real_t trace = 0.0, normal_trace = 0.0;
    if (active)
    {
      const int element = F(side, face);
#pragma unroll 1
      for (int i = 0; i < 6; ++i)
      {
        const int native = lex_to_native[axis == 0 ? i + 6 * index : index + 6 * i];
        const mfem::real_t x = record.x[D(native, 0, element)];
        trace += E(0, i, side, slot) * x;
        normal_trace += E(1, i, side, slot) * x;
      }
    }
    mfem::real_t value = 0.0, gn = 0.0, gt = 0.0;
#pragma unroll 1
    for (int t = 0; t < 6; ++t)
    {
      const mfem::real_t a = __shfl_sync(0xffffffffu, trace, 6 * side + t);
      const mfem::real_t b = __shfl_sync(0xffffffffu, normal_trace, 6 * side + t);
      if (active)
      {
        const mfem::real_t bt = T(0, t, index, side, slot);
        value += bt * a;
        gn += bt * b;
        gt += T(1, t, index, side, slot) * a;
      }
    }
    mfem::real_t flux = 0.0;
    if (active)
    {
      flux = Q(side, index, face) *
        (G(axis, side, index, face) * gn + G(1 - axis, side, index, face) * gt);
    }
    const int other = 6 * (1 - side) + index;
    const mfem::real_t other_value = __shfl_sync(0xffffffffu, value, other);
    const mfem::real_t other_flux = __shfl_sync(0xffffffffu, flux, other);
    mfem::real_t cs = 0.0, cn = 0.0, ct = 0.0;
    if (active)
    {
      const mfem::real_t jump = side == 0 ? value - other_value : other_value - value;
      const mfem::real_t average = side == 0 ? flux + other_flux : other_flux + flux;
      const mfem::real_t penalty_jump = P(index, face) * jump;
      cs = side == 0 ? -average + penalty_jump : average - penalty_jump;
      const mfem::real_t scaled_jump = record.ip_sigma * jump * Q(side, index, face);
      cn = scaled_jump * G(axis, side, index, face);
      ct = scaled_jump * G(1 - axis, side, index, face);
    }
    mfem::real_t lift_a = 0.0, lift_b = 0.0;
#pragma unroll 1
    for (int q = 0; q < 6; ++q)
    {
      const mfem::real_t qs = __shfl_sync(0xffffffffu, cs, 6 * side + q);
      const mfem::real_t qn = __shfl_sync(0xffffffffu, cn, 6 * side + q);
      const mfem::real_t qt = __shfl_sync(0xffffffffu, ct, 6 * side + q);
      if (active)
      {
        const mfem::real_t bt = T(0, index, q, side, slot);
        lift_a += bt * qs + T(1, index, q, side, slot) * qt;
        lift_b += bt * qn;
      }
    }
    if constexpr (CompactLift)
    {
      auto L = mfem::Reshape(record.contribution, 2, 6, 2, record.nf);
      if (active)
      {
        L(0, index, side, face) = lift_a;
        L(1, index, side, face) = lift_b;
      }
    }
    else
    {
      auto C = mfem::Reshape(record.contribution, 36, 2, record.nf);
#pragma unroll 1
      for (int base = 0; base < 72; base += 32)
      {
        const int row = base + lane;
        const bool owns = row < 72;
        const int output_side = owns ? row / 36 : 0;
        const int i = owns ? row % 6 : 0;
        const int t = owns ? (row % 36) / 6 : 0;
        const mfem::real_t a = __shfl_sync(0xffffffffu, lift_a, 6 * output_side + t);
        const mfem::real_t b = __shfl_sync(0xffffffffu, lift_b, 6 * output_side + t);
        if (owns)
        {
          const int output_axis = Axis(output_side, slot);
          const int native = lex_to_native[output_axis == 0 ? i + 6 * t : t + 6 * i];
          C(native, output_side, face) =
            E(0, i, output_side, slot) * a + E(1, i, output_side, slot) * b;
        }
      }
    }
#else
    if constexpr (CompactLift)
    {
      MFEM_ABORT_KERNEL("Compact FP64 tensor face records require native CUDA.");
    }
    else { record(face); }
#endif
  }

  MFEM_HOST_DEVICE void operator()(int face) const { Apply<false>(face); }
};

struct InteriorFaceDGDiffusionTensorLiftRecord2DWarp
  : InteriorFaceDGDiffusionTensorRecord2DWarp
{
  MFEM_HOST_DEVICE void operator()(int face) const { Apply<true>(face); }
};

struct InteriorFaceDGDiffusionGather2D
{
  int dof = 0;
  int vdim = 0;
  int ne = 0;
  int nf = 0;
  const int* element_vdof_map = nullptr;
  const int* element_face_offsets = nullptr;
  const int* element_face_ids = nullptr;
  const int* element_face_sides = nullptr;
  const mfem::real_t* contribution = nullptr;
  mfem::real_t* y = nullptr;
  bool add_to_base = false;

  MFEM_HOST_DEVICE void operator()(int row) const
  {
    const int element = row / dof;
    const int local_dof = row - element * dof;
    auto D = mfem::Reshape(element_vdof_map, dof, vdim, ne);
    auto C = mfem::Reshape(contribution, dof, 2, nf);
    mfem::real_t sum = 0.0;
    for (int incidence = element_face_offsets[element];
         incidence < element_face_offsets[element + 1];
         ++incidence)
    {
      sum += C(local_dof,
               element_face_sides[incidence],
               element_face_ids[incidence]);
    }
    // One CUDA row owns each discontinuous scalar dof.  Interior and shared
    // gathers execute in stream order, so the latter may safely add here.
    const int output = D(local_dof, 0, element);
    if (add_to_base)
    {
      // Original order: scratch = +0; scratch += sum; base += scratch.
      // Explicit CUDA adds retain both roundings, including signed zeros and
      // NaN propagation, without a scratch-vector zero or vector-add launch.
#if defined(__CUDA_ARCH__) && defined(MFEM_USE_DOUBLE)
      y[output] = __dadd_rn(y[output], __dadd_rn(0.0, sum));
#elif defined(__CUDA_ARCH__) && defined(MFEM_USE_SINGLE)
      y[output] = __fadd_rn(y[output], __fadd_rn(0.0f, sum));
#else
      const mfem::real_t scratch_value = mfem::real_t(0.0) + sum;
      y[output] += scratch_value;
#endif
    }
    else
    {
      y[output] += sum;
    }
  }
};

struct InteriorFaceDGDiffusionTensorLiftGather2D
  : InteriorFaceDGDiffusionGather2D
{
  const mfem::real_t* endpoint = nullptr;
  const int* axes = nullptr;
  const int* face_slots = nullptr;
  const int* native_to_lex = nullptr;
  int slot_count = 0;

  MFEM_HOST_DEVICE void operator()(int row) const
  {
    const int element = row / 36;
    const int native = row - element * 36;
    const int lex = native_to_lex[native];
    auto D = mfem::Reshape(element_vdof_map, 36, vdim, ne);
    auto E = mfem::Reshape(endpoint, 2, 6, 2, slot_count);
    auto Axis = mfem::Reshape(axes, 2, slot_count);
    auto L = mfem::Reshape(contribution, 2, 6, 2, nf);
    mfem::real_t sum = 0.0;
    for (int incidence = element_face_offsets[element];
         incidence < element_face_offsets[element + 1]; ++incidence)
    {
      const int side = element_face_sides[incidence];
      const int face = element_face_ids[incidence];
      // Read the current independent tensor slot, including renumbered packs.
      const int slot = face_slots[face];
      const int axis = Axis(side, slot);
      const int i = axis == 0 ? lex % 6 : lex / 6;
      const int t = axis == 0 ? lex / 6 : lex % 6;
      const mfem::real_t a = L(0, t, side, face);
      const mfem::real_t b = L(1, t, side, face);
      const mfem::real_t face_value =
        E(0, i, side, slot) * a + E(1, i, side, slot) * b;
      // The old record stores one rounded FP64 contribution before the ordered
      // gather. Explicit add prevents contraction into that reconstruction.
#if defined(__CUDA_ARCH__) && defined(MFEM_USE_DOUBLE)
      sum = __dadd_rn(sum, face_value);
#else
      volatile mfem::real_t rounded_face_value = face_value;
      sum += rounded_face_value;
#endif
    }
    const int output = D(native, 0, element);
    if (add_to_base)
    {
#if defined(__CUDA_ARCH__) && defined(MFEM_USE_DOUBLE)
      y[output] = __dadd_rn(y[output], __dadd_rn(0.0, sum));
#else
      const mfem::real_t scratch_value = mfem::real_t(0.0) + sum;
      y[output] += scratch_value;
#endif
    }
    else
    {
#if defined(__CUDA_ARCH__) && defined(MFEM_USE_DOUBLE)
      y[output] = __dadd_rn(y[output], sum);
#else
      y[output] += sum;
#endif
    }
  }
};

// Keep residual data and arithmetic out of the legacy gather's functor and
// kernel type. Even a disabled runtime branch changes register allocation and
// can change which of two quiet-NaN operand payloads the legacy add preserves.
struct InteriorFaceDGDiffusionTensorLiftResidual2D
  : InteriorFaceDGDiffusionTensorLiftGather2D
{
  const mfem::real_t* rhs = nullptr;

  MFEM_HOST_DEVICE void operator()(int row) const
  {
    const int element = row / 36;
    const int native = row - element * 36;
    const int lex = native_to_lex[native];
    auto D = mfem::Reshape(element_vdof_map, 36, vdim, ne);
    auto E = mfem::Reshape(endpoint, 2, 6, 2, slot_count);
    auto Axis = mfem::Reshape(axes, 2, slot_count);
    auto L = mfem::Reshape(contribution, 2, 6, 2, nf);
    mfem::real_t sum = 0.0;
    for (int incidence = element_face_offsets[element];
         incidence < element_face_offsets[element + 1]; ++incidence)
    {
      const int side = element_face_sides[incidence];
      const int face = element_face_ids[incidence];
      const int slot = face_slots[face];
      const int axis = Axis(side, slot);
      const int i = axis == 0 ? lex % 6 : lex / 6;
      const int t = axis == 0 ? lex / 6 : lex % 6;
      const mfem::real_t a = L(0, t, side, face);
      const mfem::real_t b = L(1, t, side, face);
      const mfem::real_t face_value =
        E(0, i, side, slot) * a + E(1, i, side, slot) * b;
      // The separate old record rounds before the ordered incidence gather.
#if defined(__CUDA_ARCH__) && defined(MFEM_USE_DOUBLE)
      sum = __dadd_rn(sum, face_value);
#else
      volatile mfem::real_t rounded_face_value = face_value;
      sum += rounded_face_value;
#endif
    }
    const int output = D(native, 0, element);
    // Retain the old scratch/base/subtract sequence as three rounded FP64
    // operations, with no store of the completed action between kernels.
#if defined(__CUDA_ARCH__) && defined(MFEM_USE_DOUBLE)
    const mfem::real_t face = __dadd_rn(0.0, sum);
    const mfem::real_t action = __dadd_rn(y[output], face);
    y[output] = __dsub_rn(rhs[output], action);
#else
    const mfem::real_t face = mfem::real_t(0.0) + sum;
    const mfem::real_t action = y[output] + face;
    y[output] = rhs[output] - action;
#endif
  }
};

struct SharedFaceDGDiffusionRecord2D
{
  int dof = 0;
  int vdim = 0;
  int nq = 0;
  int nf = 0;
  int ne = 0;
  mfem::real_t ip_sigma = 0.0;
  const mfem::real_t* shape = nullptr;
  const mfem::real_t* weighted_normal_dshape = nullptr;
  const mfem::real_t* penalty = nullptr;
  const int* face_elements = nullptr;
  const int* element_vdof_map = nullptr;
  const int* face_nbr_dof_map = nullptr;
  const mfem::real_t* x = nullptr;
  const mfem::real_t* x_face_nbr = nullptr;
  mfem::real_t* contribution = nullptr;

  MFEM_HOST_DEVICE mfem::real_t FaceNbrValue(int vdof) const
  {
    return (vdof >= 0) ? x_face_nbr[vdof] : -x_face_nbr[-1 - vdof];
  }

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto S = mfem::Reshape(shape, 2, nq, dof, nf);
    auto N = mfem::Reshape(weighted_normal_dshape, 2, nq, dof, nf);
    auto P = mfem::Reshape(penalty, nq, nf);
    auto F = mfem::Reshape(face_elements, nf);
    auto D = mfem::Reshape(element_vdof_map, dof, vdim, ne);
    auto DN = mfem::Reshape(face_nbr_dof_map, dof, vdim, nf);
    auto C = mfem::Reshape(contribution, dof, nf);

    const int e0 = F(face);
    for (int d = 0; d < dof; ++d)
    {
      C(d, face) = 0.0;
    }
    for (int q = 0; q < nq; ++q)
    {
      mfem::real_t value[2] = {0.0, 0.0};
      mfem::real_t normal_flux[2] = {0.0, 0.0};
      for (int d = 0; d < dof; ++d)
      {
        const mfem::real_t local_value = x[D(d, 0, e0)];
        const mfem::real_t nbr_value = FaceNbrValue(DN(d, 0, face));
        value[0] += S(0, q, d, face) * local_value;
        normal_flux[0] += N(0, q, d, face) * local_value;
        value[1] += S(1, q, d, face) * nbr_value;
        normal_flux[1] += N(1, q, d, face) * nbr_value;
      }

      const mfem::real_t jump = value[0] - value[1];
      const mfem::real_t avg_flux = normal_flux[0] + normal_flux[1];
      const mfem::real_t penalty_jump = P(q, face) * jump;

      for (int d = 0; d < dof; ++d)
      {
        const mfem::real_t y0 =
          -S(0, q, d, face) * avg_flux +
          ip_sigma * N(0, q, d, face) * jump +
          S(0, q, d, face) * penalty_jump;
        C(d, face) += y0;
      }
    }
  }
};

struct SharedFaceDGDiffusionGather2D
{
  int dof = 0;
  int vdim = 0;
  int ne = 0;
  int nf = 0;
  const int* element_vdof_map = nullptr;
  const int* element_face_offsets = nullptr;
  const int* element_face_ids = nullptr;
  const mfem::real_t* contribution = nullptr;
  mfem::real_t* y = nullptr;

  MFEM_HOST_DEVICE void operator()(int row) const
  {
    const int element = row / dof;
    const int local_dof = row - element * dof;
    auto D = mfem::Reshape(element_vdof_map, dof, vdim, ne);
    auto C = mfem::Reshape(contribution, dof, nf);
    mfem::real_t sum = 0.0;
    for (int incidence = element_face_offsets[element];
         incidence < element_face_offsets[element + 1];
         ++incidence)
    {
      sum += C(local_dof, element_face_ids[incidence]);
    }
    y[D(local_dof, 0, element)] += sum;
  }
};

inline void AddInteriorFaceDGDiffusionApplyDevice2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  mfem::real_t ip_sigma,
  mfem::Vector& y,
  bool add_to_base = false,
  const mfem::Vector* residual_rhs = nullptr)
{
  MFEM_VERIFY(info.vdim == 1 &&
                scalar.Size() == info.ne * info.dof &&
                y.Size() == scalar.Size(),
              "Deterministic exact DG face gather requires one uniquely "
              "owned scalar row per discontinuous element dof.");
  EnsureInteriorFaceElementIncidence2D(info, work);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__) && defined(MFEM_USE_DOUBLE)
  if (UsesInteriorFaceDGDiffusionCompactLift2D(info, work))
  {
    const auto& factor = work.normal_factorization;
    // Keep only the selected live scratch format. A pooled allocator can retain
    // the released block, so this logical reduction need not lower GPU residency.
    if (work.face_contribution.Size()) { work.face_contribution.Destroy(); }
    work.tensor_lift.SetSize(24 * info.nf);
    work.tensor_lift.UseDevice(true);
    InteriorFaceDGDiffusionRecord2D record;
    record.dof = info.dof;
    record.vdim = info.vdim;
    record.nq = info.nq;
    record.nf = info.nf;
    record.ne = info.ne;
    record.ip_sigma = ip_sigma;
    record.penalty = work.penalty.Read();
    record.face_elements = work.face_elements.Read();
    record.element_vdof_map = work.element_vdof_map.Read();
    record.x = scalar.Read();
    record.contribution = work.tensor_lift.Write();
    const InteriorFaceDGDiffusionTensorRecord2DWarp tensor_record{
      record, factor.tensor_endpoint.Read(), factor.tensor_tangent.Read(),
      factor.tensor_axes.Read(), factor.tensor_face_slots.Read(),
      factor.tensor_lex_to_native.Read(), factor.normal_geometry.Read(),
      work.side_coefficient.Read(), factor.tensor_slot_count};
    mfem::forall_2D_batch(info.nf, 32, 1, 4,
      InteriorFaceDGDiffusionTensorLiftRecord2DWarp{tensor_record});

    InteriorFaceDGDiffusionTensorLiftGather2D gather;
    gather.dof = info.dof;
    gather.vdim = info.vdim;
    gather.ne = info.ne;
    gather.nf = info.nf;
    gather.element_vdof_map = work.element_vdof_map.Read();
    gather.element_face_offsets = work.element_face_offsets.Read();
    gather.element_face_ids = work.element_face_ids.Read();
    gather.element_face_sides = work.element_face_sides.Read();
    gather.contribution = work.tensor_lift.Read();
    gather.y = y.ReadWrite();
    gather.add_to_base = add_to_base;
    gather.endpoint = factor.tensor_endpoint.Read();
    gather.axes = factor.tensor_axes.Read();
    gather.face_slots = factor.tensor_face_slots.Read();
    gather.native_to_lex = factor.tensor_native_to_lex.Read();
    gather.slot_count = factor.tensor_slot_count;
    if (residual_rhs)
    {
      InteriorFaceDGDiffusionTensorLiftResidual2D residual;
      static_cast<InteriorFaceDGDiffusionTensorLiftGather2D&>(residual) = gather;
      residual.rhs = residual_rhs->Read();
      mfem::forall(info.ne * info.dof, residual);
    }
    else { mfem::forall(info.ne * info.dof, gather); }
    return;
  }
#endif
  MFEM_VERIFY(!residual_rhs, "Fused face residual requires its preflighted compact lift layout.");
  if (work.tensor_lift.Size()) { work.tensor_lift.Destroy(); }
  work.face_contribution.SetSize(2 * info.dof * info.nf);
  work.face_contribution.UseDevice(true);

  InteriorFaceDGDiffusionRecord2D record;
  record.dof = info.dof;
  record.vdim = info.vdim;
  record.nq = info.nq;
  record.nf = info.nf;
  record.ne = info.ne;
  record.ip_sigma = ip_sigma;
  record.penalty = work.penalty.Read();
  record.face_elements = work.face_elements.Read();
  record.element_vdof_map = work.element_vdof_map.Read();
  record.x = scalar.Read();
  record.contribution = work.face_contribution.Write();
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  if (info.dof > 0 && info.nq > 0 && info.nq <= 16 &&
      mfem::Device::Allows(mfem::Backend::CUDA) &&
      !mfem::Device::Allows(mfem::Backend::RAJA_CUDA))
  {
    const auto& factor = work.normal_factorization;
    const bool factorized = UsesOrdinaryCUDAForDGDiffusionFaces2D() &&
      HasCurrentInteriorFaceDGDiffusionTensorData2D(info, work);
    if (factorized)
    {
      mfem::forall_2D_batch(
        info.nf, 32, 1, 4, InteriorFaceDGDiffusionTensorRecord2DWarp{
          record, factor.tensor_endpoint.Read(), factor.tensor_tangent.Read(),
          factor.tensor_axes.Read(), factor.tensor_face_slots.Read(),
          factor.tensor_lex_to_native.Read(), factor.normal_geometry.Read(),
          work.side_coefficient.Read(), factor.tensor_slot_count});
    }
    else
    {
      EnsureInteriorFaceDGDiffusionDenseNormals2D(info, work);
      const bool compact_shape = work.shape_dictionary.Matches(info);
      record.shape = compact_shape ? work.shape_dictionary.values.Read() : work.shape.Read();
      const int* shape_slots = compact_shape ? work.shape_dictionary.face_slots.Read() : nullptr;
      const int shape_count = compact_shape ? work.shape_dictionary.slot_count : info.nf;
      record.weighted_normal_dshape = work.weighted_normal_dshape.Read();
      mfem::forall_2D_batch(
        info.nf, 32, 1, 4, InteriorFaceDGDiffusionRecord2DWarp<false>{
          record, shape_slots, shape_count});
    }
  }
  else
#endif
  {
    EnsureInteriorFaceDGDiffusionDenseNormals2D(info, work);
    record.shape = work.shape.Read();
    record.weighted_normal_dshape = work.weighted_normal_dshape.Read();
    mfem::forall(info.nf, record);
  }

  InteriorFaceDGDiffusionGather2D gather;
  gather.dof = info.dof;
  gather.vdim = info.vdim;
  gather.ne = info.ne;
  gather.nf = info.nf;
  gather.element_vdof_map = work.element_vdof_map.Read();
  gather.element_face_offsets = work.element_face_offsets.Read();
  gather.element_face_ids = work.element_face_ids.Read();
  gather.element_face_sides = work.element_face_sides.Read();
  gather.contribution = work.face_contribution.Read();
  gather.y = y.ReadWrite();
  gather.add_to_base = add_to_base;
  mfem::forall(info.ne * info.dof, gather);
}

inline void AddSharedFaceDGDiffusionApplyDevice2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  const SharedFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  const mfem::Vector& scalar_face_nbr,
  mfem::real_t ip_sigma,
  mfem::Vector& y)
{
  MFEM_VERIFY(info.vdim == 1 &&
                scalar.Size() == info.ne * info.dof &&
                y.Size() == scalar.Size(),
              "Deterministic shared exact DG face gather requires one "
              "uniquely owned scalar row per discontinuous element dof.");
  EnsureSharedFaceElementIncidence2D(info, work);
  work.face_contribution.SetSize(info.dof * info.nf);
  work.face_contribution.UseDevice(true);

  SharedFaceDGDiffusionRecord2D record;
  record.dof = info.dof;
  record.vdim = info.vdim;
  record.nq = info.nq;
  record.nf = info.nf;
  record.ne = info.ne;
  record.ip_sigma = ip_sigma;
  record.shape = work.shape.Read();
  record.weighted_normal_dshape = work.weighted_normal_dshape.Read();
  record.penalty = work.penalty.Read();
  record.face_elements = work.face_elements.Read();
  record.element_vdof_map = work.element_vdof_map.Read();
  record.face_nbr_dof_map = work.face_nbr_dof_map.Read();
  record.x = scalar.Read();
  record.x_face_nbr = scalar_face_nbr.Read();
  record.contribution = work.face_contribution.Write();
  mfem::forall(info.nf, record);

  SharedFaceDGDiffusionGather2D gather;
  gather.dof = info.dof;
  gather.vdim = info.vdim;
  gather.ne = info.ne;
  gather.nf = info.nf;
  gather.element_vdof_map = work.element_vdof_map.Read();
  gather.element_face_offsets = work.element_face_offsets.Read();
  gather.element_face_ids = work.element_face_ids.Read();
  gather.contribution = work.face_contribution.Read();
  gather.y = y.ReadWrite();
  mfem::forall(info.ne * info.dof, gather);
}

inline InteriorFaceSideDensityProjectionInfo
ProjectInteriorFaceDensityCoefficient2DImpl(
  const mfem::FiniteElementSpace& vfes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& state,
  mfem::real_t alpha,
  const mfem::Vector* element_alpha,
  mfem::real_t rho_floor,
  mfem::Vector& side_q,
  InteriorFaceSideDensityProjectionScratch* scratch,
  const InteriorFaceSideDensityProjectionInfo* prepared_geometry = nullptr)
{
  MFEM_VERIFY(state.ParFESpace() == &vfes,
              "Side-density projection state/FESpace mismatch.");
  InteriorFaceSideDensityProjectionScratch local_scratch;
  InteriorFaceSideDensityProjectionScratch& work =
    scratch ? *scratch : local_scratch;
  const auto info = prepared_geometry ? *prepared_geometry :
    BuildInteriorFaceSideDensityProjectionData2D(
      vfes,
      fqs,
      work.shape,
      work.face_elements,
      work.element_vdof_map);

  side_q.SetSize(2 * info.nq * info.nf);
  side_q.UseDevice(true);

  work.side_alpha.SetSize(element_alpha ? 2 * info.nf : 0);
  if (element_alpha)
  {
    MFEM_VERIFY(element_alpha->Size() == info.ne,
                "Side-density projection received mismatched per-element "
                "alpha data.");
    const mfem::real_t* element_alpha_data = element_alpha->HostRead();
    const auto F = mfem::Reshape(work.face_elements.HostRead(), 2, info.nf);
    auto A = mfem::Reshape(work.side_alpha.HostWrite(), 2, info.nf);
    for (int f = 0; f < info.nf; ++f)
    {
      for (int side = 0; side < 2; ++side)
      {
        const mfem::real_t alpha_e = element_alpha_data[F(side, f)];
        MFEM_VERIFY(alpha_e > 0.0 && std::isfinite(alpha_e),
                    "Side-density projection received invalid per-element "
                    "alpha data.");
        A(side, f) = alpha_e;
      }
    }
    work.side_alpha.UseDevice(true);
  }

#if (defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
    (defined(MFEM_USE_HIP) && !defined(__HIPCC__))
  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), 2, info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  const mfem::real_t* state_data = state.HostRead();
  auto Q = mfem::Reshape(side_q.HostWrite(), 2, info.nq, info.nf);
  const mfem::real_t* side_alpha_data =
    element_alpha ? work.side_alpha.HostRead() : nullptr;
  const auto A = mfem::Reshape(side_alpha_data, 2, info.nf);

  for (int f = 0; f < info.nf; ++f)
  {
    for (int q = 0; q < info.nq; ++q)
    {
      for (int side = 0; side < 2; ++side)
      {
        const int e = F(side, f);
        mfem::real_t rho = 0.0;
        for (int d = 0; d < info.dof; ++d)
        {
          rho += S(side, q, d, f) * state_data[D(d, 0, e)];
        }
        const mfem::real_t side_alpha_value = element_alpha ? A(side, f) : alpha;
        Q(side, q, f) = frozen_scalar_igr_algebra::AlphaOverDensity(
          side_alpha_value,
          rho,
          rho_floor);
      }
    }
  }
#else
  InteriorFaceSideDensityProjection2D kernel;
  kernel.dof = info.dof;
  kernel.vdim = info.vdim;
  kernel.nq = info.nq;
  kernel.nf = info.nf;
  kernel.ne = info.ne;
  kernel.alpha = alpha;
  kernel.rho_floor = rho_floor;
  kernel.shape = work.shape.Read();
  kernel.side_alpha = element_alpha ? work.side_alpha.Read() : nullptr;
  kernel.face_elements = work.face_elements.Read();
  kernel.element_vdof_map = work.element_vdof_map.Read();
  kernel.state = state.Read();
  kernel.side_q = side_q.Write();
  mfem::forall_2D(info.nf, 2 * info.nq, 1, kernel);
#endif

  InteriorFaceSideDensityProjectionInfo out = info;
  out.qf_size = side_q.Size();
  return out;
}

inline SharedFaceSideDensityProjectionInfo
ProjectSharedFaceDensityCoefficient2DImpl(
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::IntegrationRule& ir,
  const mfem::ParGridFunction& state,
  mfem::real_t alpha,
  const mfem::Vector* face_side_alpha,
  mfem::real_t rho_floor,
  mfem::Vector& side_q,
  SharedFaceSideDensityProjectionScratch* scratch)
{
  MFEM_VERIFY(state.ParFESpace() == &vfes,
              "Shared side-density projection state/FESpace mismatch.");
  auto& mutable_state = const_cast<mfem::ParGridFunction&>(state);
  mutable_state.ExchangeFaceNbrData();

  SharedFaceSideDensityProjectionScratch local_scratch;
  SharedFaceSideDensityProjectionScratch& work =
    scratch ? *scratch : local_scratch;
  const auto info = BuildSharedFaceSideDensityProjectionData2D(
    vfes,
    ir,
    work.shape,
    work.face_elements,
    work.element_vdof_map,
    work.face_nbr_dof_map);

  side_q.SetSize(2 * info.nq * info.nf);
  side_q.UseDevice(true);

  work.side_alpha.SetSize(face_side_alpha ? 2 * info.nf : 0);
  if (face_side_alpha)
  {
    MFEM_VERIFY(face_side_alpha->Size() == 2 * info.nf,
                "Shared side-density projection received mismatched per-side "
                "alpha data.");
    const mfem::real_t* alpha_in = face_side_alpha->HostRead();
    mfem::real_t* alpha_out = work.side_alpha.HostWrite();
    for (int i = 0; i < face_side_alpha->Size(); ++i)
    {
      MFEM_VERIFY(alpha_in[i] > 0.0 && std::isfinite(alpha_in[i]),
                  "Shared side-density projection received invalid per-side "
                  "alpha data.");
      alpha_out[i] = alpha_in[i];
    }
    work.side_alpha.UseDevice(true);
  }

  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  const auto DN = mfem::Reshape(work.face_nbr_dof_map.HostRead(),
                                info.dof,
                                info.vdim,
                                info.nf);
  const mfem::real_t* state_data = state.HostRead();
  const mfem::real_t* nbr_data = mutable_state.FaceNbrData().HostRead();
  auto Q = mfem::Reshape(side_q.HostWrite(), 2, info.nq, info.nf);
  const mfem::real_t* side_alpha_data =
    face_side_alpha ? work.side_alpha.HostRead() : nullptr;
  const auto A = mfem::Reshape(side_alpha_data, 2, info.nf);

  for (int f = 0; f < info.nf; ++f)
  {
    const int e0 = F(f);
    for (int q = 0; q < info.nq; ++q)
    {
      mfem::real_t rho0 = 0.0;
      mfem::real_t rho1 = 0.0;
      for (int d = 0; d < info.dof; ++d)
      {
        rho0 += S(0, q, d, f) * state_data[D(d, 0, e0)];
        const int nbr_vdof = DN(d, 0, f);
        const mfem::real_t nbr_value =
          (nbr_vdof >= 0) ? nbr_data[nbr_vdof]
                          : -nbr_data[-1 - nbr_vdof];
        rho1 += S(1, q, d, f) * nbr_value;
      }
      const mfem::real_t alpha0 = face_side_alpha ? A(0, f) : alpha;
      const mfem::real_t alpha1 = face_side_alpha ? A(1, f) : alpha;
      Q(0, q, f) = frozen_scalar_igr_algebra::AlphaOverDensity(
        alpha0,
        rho0,
        rho_floor);
      Q(1, q, f) = frozen_scalar_igr_algebra::AlphaOverDensity(
        alpha1,
        rho1,
        rho_floor);
    }
  }

  SharedFaceSideDensityProjectionInfo out = info;
  out.qf_size = side_q.Size();
  return out;
}

/** Project alpha/rho with one constant alpha on both sides of every local face. */
inline InteriorFaceSideDensityProjectionInfo
ProjectInteriorFaceDensityCoefficient2D(
  const mfem::FiniteElementSpace& vfes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& state,
  mfem::real_t alpha,
  mfem::real_t rho_floor,
  mfem::Vector& side_q,
  InteriorFaceSideDensityProjectionScratch* scratch = nullptr)
{
  return ProjectInteriorFaceDensityCoefficient2DImpl(vfes,
                                                      fqs,
                                                      state,
                                                      alpha,
                                                      nullptr,
                                                      rho_floor,
                                                      side_q,
                                                      scratch);
}

/** Project alpha_e/rho using explicit alpha data indexed by local element. */
inline InteriorFaceSideDensityProjectionInfo
ProjectInteriorFaceDensityCoefficient2D(
  const mfem::FiniteElementSpace& vfes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& state,
  const mfem::Vector& element_alpha,
  mfem::real_t rho_floor,
  mfem::Vector& side_q,
  InteriorFaceSideDensityProjectionScratch* scratch = nullptr)
{
  return ProjectInteriorFaceDensityCoefficient2DImpl(vfes,
                                                      fqs,
                                                      state,
                                                      0.0,
                                                      &element_alpha,
                                                      rho_floor,
                                                      side_q,
                                                      scratch);
}

/** Project alpha/rho with one constant alpha on both sides of every shared face. */
inline SharedFaceSideDensityProjectionInfo
ProjectSharedFaceDensityCoefficient2D(
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::IntegrationRule& ir,
  const mfem::ParGridFunction& state,
  mfem::real_t alpha,
  mfem::real_t rho_floor,
  mfem::Vector& side_q,
  SharedFaceSideDensityProjectionScratch* scratch = nullptr)
{
  return ProjectSharedFaceDensityCoefficient2DImpl(vfes,
                                                    ir,
                                                    state,
                                                    alpha,
                                                    nullptr,
                                                    rho_floor,
                                                    side_q,
                                                    scratch);
}

/** Project alpha/rho using explicit [side][shared-face] alpha data. */
inline SharedFaceSideDensityProjectionInfo
ProjectSharedFaceDensityCoefficient2D(
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::IntegrationRule& ir,
  const mfem::ParGridFunction& state,
  const mfem::Vector& face_side_alpha,
  mfem::real_t rho_floor,
  mfem::Vector& side_q,
  SharedFaceSideDensityProjectionScratch* scratch = nullptr)
{
  return ProjectSharedFaceDensityCoefficient2DImpl(vfes,
                                                    ir,
                                                    state,
                                                    0.0,
                                                    &face_side_alpha,
                                                    rho_floor,
                                                    side_q,
                                                    scratch);
}

inline InteriorFaceDGDiffusionApplyInfo
AddInteriorFaceDGDiffusionApply2D(
  const mfem::FiniteElementSpace& sfes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::Vector& scalar,
  const mfem::Vector& side_q,
  mfem::real_t ip_sigma,
  mfem::real_t kappa,
  mfem::Vector& y,
  InteriorFaceDGDiffusionApplyScratch* scratch = nullptr)
{
  if (y.Size() == 0)
  {
    y.SetSize(scalar.Size());
    y = 0.0;
  }
  MFEM_VERIFY(y.Size() == scalar.Size(),
              "Exact DG diffusion face apply output size mismatch.");

  InteriorFaceDGDiffusionApplyScratch local_scratch;
  InteriorFaceDGDiffusionApplyScratch& work =
    scratch ? *scratch : local_scratch;
  const auto info = BuildInteriorFaceDGDiffusionApplyData2D(
    sfes,
    fqs,
    side_q,
    kappa,
    work.shape,
    work.weighted_normal_dshape,
    work.penalty,
    work.face_elements,
    work.element_vdof_map,
    DGDiffusionFaceScaling::Arithmetic,
    &work.normal_dshape_base,
    &work.penalty_metric,
    &work.shape_dictionary);

#if (defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
    (defined(MFEM_USE_HIP) && !defined(__HIPCC__))
  EnsureInteriorFaceDGDiffusionDenseNormals2D(info, work);
  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto N = mfem::Reshape(work.weighted_normal_dshape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto P = mfem::Reshape(work.penalty.HostRead(), info.nq, info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), 2, info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  const mfem::real_t* x_data = scalar.HostRead();
  mfem::real_t* y_data = y.HostReadWrite();

  for (int f = 0; f < info.nf; ++f)
  {
    for (int q = 0; q < info.nq; ++q)
    {
      mfem::real_t value[2] = {0.0, 0.0};
      mfem::real_t normal_flux[2] = {0.0, 0.0};
      for (int side = 0; side < 2; ++side)
      {
        const int e = F(side, f);
        for (int d = 0; d < info.dof; ++d)
        {
          const mfem::real_t dof_value = x_data[D(d, 0, e)];
          value[side] += S(side, q, d, f) * dof_value;
          normal_flux[side] += N(side, q, d, f) * dof_value;
        }
      }

      const mfem::real_t jump = value[0] - value[1];
      const mfem::real_t avg_flux = normal_flux[0] + normal_flux[1];
      const mfem::real_t penalty_jump = P(q, f) * jump;

      const int e0 = F(0, f);
      const int e1 = F(1, f);
      for (int d = 0; d < info.dof; ++d)
      {
        y_data[D(d, 0, e0)] +=
          -S(0, q, d, f) * avg_flux +
          ip_sigma * N(0, q, d, f) * jump +
          S(0, q, d, f) * penalty_jump;
        y_data[D(d, 0, e1)] +=
          S(1, q, d, f) * avg_flux +
          ip_sigma * N(1, q, d, f) * jump -
          S(1, q, d, f) * penalty_jump;
      }
    }
  }
#else
  AddInteriorFaceDGDiffusionApplyDevice2D(
    info, work, scalar, ip_sigma, y);
#endif

  return info;
}

inline InteriorFaceDGDiffusionApplyInfo
AddInteriorFaceDGDiffusionApply2D(
  const mfem::FiniteElementSpace& sfes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& scalar,
  const mfem::Vector& side_q,
  mfem::real_t ip_sigma,
  mfem::real_t kappa,
  mfem::Vector& y,
  InteriorFaceDGDiffusionApplyScratch* scratch = nullptr)
{
  MFEM_VERIFY(scalar.ParFESpace() == &sfes,
              "Exact DG diffusion face apply state/FESpace mismatch.");
  return AddInteriorFaceDGDiffusionApply2D(sfes,
                                           fqs,
                                           static_cast<const mfem::Vector&>(
                                             scalar),
                                           side_q,
                                           ip_sigma,
                                           kappa,
                                           y,
                                           scratch);
}

inline void AddInteriorFaceDGDiffusionApply2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  mfem::real_t ip_sigma,
  mfem::Vector& y)
{
  MFEM_VERIFY(scalar.Size() == y.Size(),
              "Exact DG diffusion face apply vector size mismatch.");

#if (defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
    (defined(MFEM_USE_HIP) && !defined(__HIPCC__))
  EnsureInteriorFaceDGDiffusionDenseNormals2D(info, work);
  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto N = mfem::Reshape(work.weighted_normal_dshape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto P = mfem::Reshape(work.penalty.HostRead(), info.nq, info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), 2, info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  const mfem::real_t* x_data = scalar.HostRead();
  mfem::real_t* y_data = y.HostReadWrite();

  for (int f = 0; f < info.nf; ++f)
  {
    for (int q = 0; q < info.nq; ++q)
    {
      mfem::real_t value[2] = {0.0, 0.0};
      mfem::real_t normal_flux[2] = {0.0, 0.0};
      for (int side = 0; side < 2; ++side)
      {
        const int e = F(side, f);
        for (int d = 0; d < info.dof; ++d)
        {
          const mfem::real_t dof_value = x_data[D(d, 0, e)];
          value[side] += S(side, q, d, f) * dof_value;
          normal_flux[side] += N(side, q, d, f) * dof_value;
        }
      }

      const mfem::real_t jump = value[0] - value[1];
      const mfem::real_t avg_flux = normal_flux[0] + normal_flux[1];
      const mfem::real_t penalty_jump = P(q, f) * jump;

      const int e0 = F(0, f);
      const int e1 = F(1, f);
      for (int d = 0; d < info.dof; ++d)
      {
        y_data[D(d, 0, e0)] +=
          -S(0, q, d, f) * avg_flux +
          ip_sigma * N(0, q, d, f) * jump +
          S(0, q, d, f) * penalty_jump;
        y_data[D(d, 0, e1)] +=
          S(1, q, d, f) * avg_flux +
          ip_sigma * N(1, q, d, f) * jump -
          S(1, q, d, f) * penalty_jump;
      }
    }
  }
#else
  AddInteriorFaceDGDiffusionApplyDevice2D(
    info, work, scalar, ip_sigma, y);
#endif
}

inline mfem::real_t DGDiffusionFaceDiagonalEntry2D(int side,
                                                   mfem::real_t shape,
                                                   mfem::real_t normal_dshape,
                                                   mfem::real_t penalty,
                                                   mfem::real_t ip_sigma)
{
  const mfem::real_t consistency =
    (side == 0) ? (ip_sigma - 1.0) : (1.0 - ip_sigma);
  return consistency * shape * normal_dshape + penalty * shape * shape;
}

inline void AddInteriorFaceDGDiffusionDiagonal2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  mfem::real_t ip_sigma,
  mfem::Vector& diag)
{
  if (work.compact_qdata && UsesOrdinaryCUDAForDGDiffusionFaces2D() &&
      work.coefficient_geometry_revision == info.connectivity_revision &&
      work.coefficient_generation != 0 &&
      HasCurrentInteriorFaceDGDiffusionTensorData2D(info, work))
  {
    // Reconstruct only each diagonal entry. Materializing the full current N
    // table here would undo both the refresh and device-residency reduction.
    const auto& factor = work.normal_factorization;
    const auto E = mfem::Reshape(factor.tensor_endpoint.HostRead(),
                                  2, 6, 2, factor.tensor_slot_count);
    const auto T = mfem::Reshape(factor.tensor_tangent.HostRead(),
                                  2, 6, 6, 2, factor.tensor_slot_count);
    const auto Axis = mfem::Reshape(factor.tensor_axes.HostRead(),
                                     2, factor.tensor_slot_count);
    const int* slots = factor.tensor_face_slots.HostRead();
    const int* native = factor.tensor_lex_to_native.HostRead();
    const auto G = mfem::Reshape(factor.normal_geometry.HostRead(), 2, 2, 6, info.nf);
    const auto Q = mfem::Reshape(work.side_coefficient.HostRead(), 2, 6, info.nf);
    const auto P = mfem::Reshape(work.penalty.HostRead(), 6, info.nf);
    const auto F = mfem::Reshape(work.face_elements.HostRead(), 2, info.nf);
    const auto D = mfem::Reshape(work.element_vdof_map.HostRead(), 36, 1, info.ne);
    mfem::real_t* diag_data = diag.HostReadWrite();
    for (int face = 0; face < info.nf; ++face)
    {
      const int slot = slots[face];
      for (int side = 0; side < 2; ++side)
      {
        const int axis = Axis(side, slot), element = F(side, face);
        for (int q = 0; q < 6; ++q)
        {
          for (int t = 0; t < 6; ++t) for (int i = 0; i < 6; ++i)
          {
            const int d = native[axis == 0 ? i + 6 * t : t + 6 * i];
            const int vdof = D(d, 0, element);
            MFEM_VERIFY(vdof >= 0 && vdof < diag.Size(),
                        "Compact interior exact-face diagonal vdof out of range.");
            const mfem::real_t bn = E(0, i, side, slot), bt = T(0, t, q, side, slot);
            const mfem::real_t dn = E(1, i, side, slot) * bt;
            const mfem::real_t dt = bn * T(1, t, q, side, slot);
            const mfem::real_t dx = axis == 0 ? dn : dt;
            const mfem::real_t dy = axis == 0 ? dt : dn;
            // Preserve the physical x/y normal accumulation order.
            mfem::real_t normal = G(0, side, q, face) * dx;
            normal += G(1, side, q, face) * dy;
            normal *= Q(side, q, face);
            diag_data[vdof] += DGDiffusionFaceDiagonalEntry2D(
              side, bn * bt, normal, P(q, face), ip_sigma);
          }
        }
      }
    }
    return;
  }
  EnsureInteriorFaceDGDiffusionDenseNormals2D(info, work);
  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto N = mfem::Reshape(work.weighted_normal_dshape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto P = mfem::Reshape(work.penalty.HostRead(), info.nq, info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), 2, info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  mfem::real_t* diag_data = diag.HostReadWrite();

  for (int f = 0; f < info.nf; ++f)
  {
    for (int side = 0; side < 2; ++side)
    {
      const int e = F(side, f);
      for (int q = 0; q < info.nq; ++q)
      {
        for (int d = 0; d < info.dof; ++d)
        {
          const int vdof = D(d, 0, e);
          MFEM_VERIFY(vdof >= 0 && vdof < diag.Size(),
                      "Interior exact-face diagonal vdof out of range.");
          diag_data[vdof] += DGDiffusionFaceDiagonalEntry2D(
            side,
            S(side, q, d, f),
            N(side, q, d, f),
            P(q, f),
            ip_sigma);
        }
      }
    }
  }
}

inline void AddSharedFaceDGDiffusionApply2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  const SharedFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  const mfem::Vector& scalar_face_nbr,
  mfem::real_t ip_sigma,
  mfem::Vector& y)
{
  MFEM_VERIFY(scalar.Size() == y.Size(),
              "Shared exact DG diffusion face apply vector size mismatch.");
  MFEM_VERIFY(scalar_face_nbr.Size() == info.face_nbr_vsize,
              "Shared exact DG diffusion face apply neighbor data size "
              "mismatch.");

#if (defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
    (defined(MFEM_USE_HIP) && !defined(__HIPCC__))
  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto N = mfem::Reshape(work.weighted_normal_dshape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto P = mfem::Reshape(work.penalty.HostRead(), info.nq, info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  const auto DN = mfem::Reshape(work.face_nbr_dof_map.HostRead(),
                                info.dof,
                                info.vdim,
                                info.nf);
  const mfem::real_t* x_data = scalar.HostRead();
  const mfem::real_t* x_nbr_data = scalar_face_nbr.HostRead();
  mfem::real_t* y_data = y.HostReadWrite();

  for (int f = 0; f < info.nf; ++f)
  {
    const int e0 = F(f);
    for (int q = 0; q < info.nq; ++q)
    {
      mfem::real_t value[2] = {0.0, 0.0};
      mfem::real_t normal_flux[2] = {0.0, 0.0};
      for (int d = 0; d < info.dof; ++d)
      {
        const int nbr_vdof = DN(d, 0, f);
        const mfem::real_t nbr_value =
          (nbr_vdof >= 0) ? x_nbr_data[nbr_vdof]
                          : -x_nbr_data[-1 - nbr_vdof];
        const mfem::real_t local_value = x_data[D(d, 0, e0)];
        value[0] += S(0, q, d, f) * local_value;
        normal_flux[0] += N(0, q, d, f) * local_value;
        value[1] += S(1, q, d, f) * nbr_value;
        normal_flux[1] += N(1, q, d, f) * nbr_value;
      }

      const mfem::real_t jump = value[0] - value[1];
      const mfem::real_t avg_flux = normal_flux[0] + normal_flux[1];
      const mfem::real_t penalty_jump = P(q, f) * jump;

      for (int d = 0; d < info.dof; ++d)
      {
        y_data[D(d, 0, e0)] +=
          -S(0, q, d, f) * avg_flux +
          ip_sigma * N(0, q, d, f) * jump +
          S(0, q, d, f) * penalty_jump;
      }
    }
  }
#else
  AddSharedFaceDGDiffusionApplyDevice2D(
    info, work, scalar, scalar_face_nbr, ip_sigma, y);
#endif
}

inline void AddSharedFaceDGDiffusionLocalDiagonal2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  const SharedFaceDGDiffusionApplyScratch& work,
  mfem::real_t ip_sigma,
  mfem::Vector& diag)
{
  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto N = mfem::Reshape(work.weighted_normal_dshape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto P = mfem::Reshape(work.penalty.HostRead(), info.nq, info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  mfem::real_t* diag_data = diag.HostReadWrite();

  for (int f = 0; f < info.nf; ++f)
  {
    const int e0 = F(f);
    for (int q = 0; q < info.nq; ++q)
    {
      for (int d = 0; d < info.dof; ++d)
      {
        const int vdof = D(d, 0, e0);
        MFEM_VERIFY(vdof >= 0 && vdof < diag.Size(),
                    "Shared exact-face diagonal vdof out of range.");
        diag_data[vdof] += DGDiffusionFaceDiagonalEntry2D(
          0,
          S(0, q, d, f),
          N(0, q, d, f),
          P(q, f),
          ip_sigma);
      }
    }
  }
}

inline SharedFaceDGDiffusionApplyInfo
AddSharedFaceDGDiffusionApply2D(
  const mfem::ParFiniteElementSpace& sfes,
  const mfem::IntegrationRule& ir,
  const mfem::Vector& scalar,
  const mfem::Vector& scalar_face_nbr,
  const mfem::Vector& side_q,
  mfem::real_t ip_sigma,
  mfem::real_t kappa,
  mfem::Vector& y,
  SharedFaceDGDiffusionApplyScratch* scratch = nullptr)
{
  if (y.Size() == 0)
  {
    y.SetSize(scalar.Size());
    y = 0.0;
  }
  MFEM_VERIFY(y.Size() == scalar.Size(),
              "Shared exact DG diffusion face apply output size mismatch.");

  SharedFaceDGDiffusionApplyScratch local_scratch;
  SharedFaceDGDiffusionApplyScratch& work =
    scratch ? *scratch : local_scratch;
  const auto info = BuildSharedFaceDGDiffusionApplyData2D(
    sfes,
    ir,
    side_q,
    kappa,
    work.shape,
    work.weighted_normal_dshape,
    work.penalty,
    work.face_elements,
    work.element_vdof_map,
    work.face_nbr_dof_map,
    DGDiffusionFaceScaling::Arithmetic,
    &work.normal_dshape_base,
    &work.penalty_metric);

  MFEM_VERIFY(scalar_face_nbr.Size() == info.face_nbr_vsize,
              "Shared exact DG diffusion face apply neighbor data size "
              "mismatch.");

#if (defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
    (defined(MFEM_USE_HIP) && !defined(__HIPCC__))
  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto N = mfem::Reshape(work.weighted_normal_dshape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto P = mfem::Reshape(work.penalty.HostRead(), info.nq, info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  const auto DN = mfem::Reshape(work.face_nbr_dof_map.HostRead(),
                                info.dof,
                                info.vdim,
                                info.nf);
  const mfem::real_t* x_data = scalar.HostRead();
  const mfem::real_t* x_nbr_data = scalar_face_nbr.HostRead();
  mfem::real_t* y_data = y.HostReadWrite();

  for (int f = 0; f < info.nf; ++f)
  {
    const int e0 = F(f);
    for (int q = 0; q < info.nq; ++q)
    {
      mfem::real_t value[2] = {0.0, 0.0};
      mfem::real_t normal_flux[2] = {0.0, 0.0};
      for (int d = 0; d < info.dof; ++d)
      {
        const int nbr_vdof = DN(d, 0, f);
        const mfem::real_t nbr_value =
          (nbr_vdof >= 0) ? x_nbr_data[nbr_vdof]
                          : -x_nbr_data[-1 - nbr_vdof];
        const mfem::real_t local_value = x_data[D(d, 0, e0)];
        value[0] += S(0, q, d, f) * local_value;
        normal_flux[0] += N(0, q, d, f) * local_value;
        value[1] += S(1, q, d, f) * nbr_value;
        normal_flux[1] += N(1, q, d, f) * nbr_value;
      }

      const mfem::real_t jump = value[0] - value[1];
      const mfem::real_t avg_flux = normal_flux[0] + normal_flux[1];
      const mfem::real_t penalty_jump = P(q, f) * jump;

      for (int d = 0; d < info.dof; ++d)
      {
        y_data[D(d, 0, e0)] +=
          -S(0, q, d, f) * avg_flux +
          ip_sigma * N(0, q, d, f) * jump +
          S(0, q, d, f) * penalty_jump;
      }
    }
  }
#else
  AddSharedFaceDGDiffusionApplyDevice2D(
    info, work, scalar, scalar_face_nbr, ip_sigma, y);
#endif

  return info;
}

inline SharedFaceDGDiffusionApplyInfo
AddSharedFaceDGDiffusionApply2D(
  const mfem::ParFiniteElementSpace& sfes,
  const mfem::IntegrationRule& ir,
  mfem::ParGridFunction& scalar,
  const mfem::Vector& side_q,
  mfem::real_t ip_sigma,
  mfem::real_t kappa,
  mfem::Vector& y,
  SharedFaceDGDiffusionApplyScratch* scratch = nullptr)
{
  MFEM_VERIFY(scalar.ParFESpace() == &sfes,
              "Shared exact DG diffusion face apply state/FESpace mismatch.");
  scalar.ExchangeFaceNbrData();
  return AddSharedFaceDGDiffusionApply2D(
    sfes,
    ir,
    static_cast<const mfem::Vector&>(scalar),
    scalar.FaceNbrData(),
    side_q,
    ip_sigma,
    kappa,
    y,
    scratch);
}

inline InteriorFaceScalarTraceProjectionInfo
ProjectInteriorFaceScalarTrace2D(
  const mfem::FiniteElementSpace& sfes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& scalar,
  mfem::Vector& value_q,
  mfem::Vector& normal_deriv_q,
  InteriorFaceScalarTraceProjectionScratch* scratch = nullptr)
{
  MFEM_VERIFY(scalar.ParFESpace() == &sfes,
              "Scalar face trace projection state/FESpace mismatch.");
  InteriorFaceScalarTraceProjectionScratch local_scratch;
  InteriorFaceScalarTraceProjectionScratch& work =
    scratch ? *scratch : local_scratch;
  const auto info = BuildInteriorFaceScalarTraceProjectionData2D(
    sfes,
    fqs,
    work.shape,
    work.normal_dshape,
    work.face_elements,
    work.element_vdof_map);

  value_q.SetSize(2 * info.nq * info.nf);
  normal_deriv_q.SetSize(2 * info.nq * info.nf);
  value_q.UseDevice(true);
  normal_deriv_q.UseDevice(true);

#if (defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
    (defined(MFEM_USE_HIP) && !defined(__HIPCC__))
  const auto S = mfem::Reshape(work.shape.HostRead(),
                               2,
                               info.nq,
                               info.dof,
                               info.nf);
  const auto DS = mfem::Reshape(work.normal_dshape.HostRead(),
                                2,
                                info.nq,
                                info.dof,
                                info.nf);
  const auto F = mfem::Reshape(work.face_elements.HostRead(), 2, info.nf);
  const auto D = mfem::Reshape(work.element_vdof_map.HostRead(),
                               info.dof,
                               info.vdim,
                               info.ne);
  const mfem::real_t* scalar_data = scalar.HostRead();
  auto V = mfem::Reshape(value_q.HostWrite(), 2, info.nq, info.nf);
  auto DN = mfem::Reshape(normal_deriv_q.HostWrite(), 2, info.nq, info.nf);

  for (int f = 0; f < info.nf; ++f)
  {
    for (int q = 0; q < info.nq; ++q)
    {
      for (int side = 0; side < 2; ++side)
      {
        const int e = F(side, f);
        mfem::real_t value = 0.0;
        mfem::real_t normal_deriv = 0.0;
        for (int d = 0; d < info.dof; ++d)
        {
          const mfem::real_t dof_value = scalar_data[D(d, 0, e)];
          value += S(side, q, d, f) * dof_value;
          normal_deriv += DS(side, q, d, f) * dof_value;
        }
        V(side, q, f) = value;
        DN(side, q, f) = normal_deriv;
      }
    }
  }
#else
  InteriorFaceScalarTraceProjection2D kernel;
  kernel.dof = info.dof;
  kernel.vdim = info.vdim;
  kernel.nq = info.nq;
  kernel.nf = info.nf;
  kernel.ne = info.ne;
  kernel.shape = work.shape.Read();
  kernel.normal_dshape = work.normal_dshape.Read();
  kernel.face_elements = work.face_elements.Read();
  kernel.element_vdof_map = work.element_vdof_map.Read();
  kernel.scalar = scalar.Read();
  kernel.value_q = value_q.Write();
  kernel.normal_deriv_q = normal_deriv_q.Write();
  mfem::forall_2D(info.nf, 2 * info.nq, 1, kernel);
#endif

  InteriorFaceScalarTraceProjectionInfo out = info;
  out.value_qf_size = value_q.Size();
  out.normal_deriv_qf_size = normal_deriv_q.Size();
  return out;
}

struct FaceQFProjectDim1Threaded
{
  int vdim = 0;
  int nf = 0;
  const mfem::real_t* f = nullptr;
  mfem::real_t* q = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto F = mfem::Reshape(f, vdim, nf);
    auto Q = mfem::Reshape(q, vdim, nf);
    MFEM_FOREACH_THREAD(c, x, vdim) { Q(c, face) = F(c, face); }
  }
};

struct FaceQFProjectDim2Threaded
{
  int nd1d = 0;
  int nq1d = 0;
  int vdim = 0;
  int nf = 0;
  const mfem::real_t* B = nullptr;
  const mfem::real_t* f = nullptr;
  mfem::real_t* q = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto Bm = mfem::Reshape(B, nq1d, nd1d);
    auto F = mfem::Reshape(f, nd1d, vdim, nf);
    auto Q = mfem::Reshape(q, vdim, nq1d, nf);
    MFEM_FOREACH_THREAD(qpt, x, nq1d)
    {
      MFEM_FOREACH_THREAD(c, y, vdim)
      {
        mfem::real_t val = 0.0;
        for (int d = 0; d < nd1d; ++d)
        {
          val += Bm(qpt, d) * F(d, c, face);
        }
        Q(c, qpt, face) = val;
      }
    }
  }
};

struct FaceQFProjectDim3Threaded
{
  int nd1d = 0;
  int nq1d = 0;
  int vdim = 0;
  int nf = 0;
  const mfem::real_t* B = nullptr;
  const mfem::real_t* f = nullptr;
  mfem::real_t* q = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto Bm = mfem::Reshape(B, nq1d, nd1d);
    auto F = mfem::Reshape(f, nd1d, nd1d, vdim, nf);
    auto Q = mfem::Reshape(q, vdim, nq1d, nq1d, nf);
    MFEM_FOREACH_THREAD(q1, x, nq1d)
    {
      MFEM_FOREACH_THREAD(q2, y, nq1d)
      {
        MFEM_FOREACH_THREAD(c, z, vdim)
        {
          mfem::real_t val = 0.0;
          for (int d2 = 0; d2 < nd1d; ++d2)
          {
            const mfem::real_t b2 = Bm(q2, d2);
            for (int d1 = 0; d1 < nd1d; ++d1)
            {
              val += Bm(q1, d1) * b2 * F(d1, d2, c, face);
            }
          }
          Q(c, q1, q2, face) = val;
        }
      }
    }
  }
};

inline FaceQFProjectInfo ProjectFaceQuadratureValues(
  const mfem::FiniteElementSpace& fes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& gf,
  mfem::QuadratureFunction& qf,
  mfem::Vector* face_restriction_evec = nullptr)
{
  FaceQFProjectInfo info;
  const bool use_tensor = mfem::UsesTensorBasis(fes);
  MFEM_VERIFY(use_tensor,
              "Face quadrature projection requires tensor-product basis.");
  const mfem::ElementDofOrdering ordering =
    use_tensor ? mfem::ElementDofOrdering::LEXICOGRAPHIC
               : mfem::ElementDofOrdering::NATIVE;
  const mfem::FaceType face_type = fqs.GetFaceType();
  const mfem::Operator* R =
    fes.GetFaceRestriction(ordering, face_type,
                           mfem::L2FaceValues::SingleValued);
  MFEM_VERIFY(R, "Missing face restriction for face quadrature projection.");

  // Reuse caller-owned storage when available to avoid per-update allocations.
  mfem::Vector local_f_vec;
  mfem::Vector* f_vec_ptr = face_restriction_evec ? face_restriction_evec
                                                  : &local_f_vec;
  f_vec_ptr->SetSize(R->Height());
  R->Mult(gf, *f_vec_ptr);

  const mfem::FiniteElement* face_fe = fes.GetTypicalTraceElement();
  const mfem::IntegrationRule& ir = fqs.GetIntRule(0);
  const mfem::DofToQuad& maps = face_fe->GetDofToQuad(ir, mfem::DofToQuad::TENSOR);

  info.nd1d = maps.ndof;
  info.nq1d = maps.nqpt;
  info.vdim = fes.GetVDim();
  info.dim = fes.GetMesh()->Dimension();
  info.nf = fqs.GetNE();
  info.f_vec_size = f_vec_ptr->Size();
  info.qf_size = qf.Size();

  const mfem::real_t* B_data = maps.B.Read();
  const mfem::real_t* f_data = f_vec_ptr->Read();
  mfem::real_t* q_data = qf.Write();

  if (info.dim == 1)
  {
    FaceQFProjectDim1Threaded kernel;
    kernel.vdim = info.vdim;
    kernel.nf = info.nf;
    kernel.f = f_data;
    kernel.q = q_data;
    mfem::forall_2D(info.nf, info.vdim, 1, kernel);
    return info;
  }

  if (info.dim == 2)
  {
    FaceQFProjectDim2Threaded kernel;
    kernel.nd1d = info.nd1d;
    kernel.nq1d = info.nq1d;
    kernel.vdim = info.vdim;
    kernel.nf = info.nf;
    kernel.B = B_data;
    kernel.f = f_data;
    kernel.q = q_data;
    mfem::forall_2D(info.nf, info.nq1d, info.vdim, kernel);
    return info;
  }

  if (info.dim == 3)
  {
    FaceQFProjectDim3Threaded kernel;
    kernel.nd1d = info.nd1d;
    kernel.nq1d = info.nq1d;
    kernel.vdim = info.vdim;
    kernel.nf = info.nf;
    kernel.B = B_data;
    kernel.f = f_data;
    kernel.q = q_data;
    mfem::forall_3D(info.nf, info.nq1d, info.nq1d, info.vdim, kernel);
    return info;
  }

  MFEM_ABORT("Unsupported dimension in face quadrature projection.");
  return info;
}

} // namespace dg_face
} // namespace igrlab
