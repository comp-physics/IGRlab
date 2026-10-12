// Internal-only helper/kernels for DG face quadrature projection.
// This file is intentionally included after MFEM backend headers.
#pragma once

#include "igrlab/compat.hpp"

#include "physics/igr/frozen_scalar_igr_algebra.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
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

struct InteriorFaceDGDiffusionApplyScratch
{
  mfem::Vector shape;
  mfem::Vector weighted_normal_dshape;
  mfem::Vector penalty;
  /// Geometry-only normal derivative used to refresh state-dependent qdata.
  mfem::Vector normal_dshape_base;
  /// Geometry-only [side][quadrature][face] penalty metric.
  mfem::Vector penalty_metric;
  /// Persistent copy of the current [side][quadrature][face] alpha/rho data.
  mfem::Vector side_coefficient;
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
};

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
  mfem::Vector* side_penalty_metric = nullptr)
{
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

        const mfem::real_t base_weight =
          ip.weight * static_cast<mfem::real_t>(0.5) / elem_tr.Weight();
        const mfem::real_t side_q_value = qside(side, q, of);
        const mfem::real_t penalty_factor_value =
          ip.weight * (normal * normal) / elem_tr.Weight();
        ni.Set(base_weight, normal);
        mfem::CalcAdjugate(elem_tr.Jacobian(), adjJ);
        adjJ.Mult(ni, nh);
        side_dshape.Mult(nh, side_dshape_dn);
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
    of++;
  }
  MFEM_VERIFY(of == info.nf,
              "Exact DG diffusion face apply packed face count mismatch.");

  face_elements.GetMemory().UseDevice(true);
  BuildNativeElementVDofMap(sfes, element_vdof_map);
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

// Native CUDA p3/Q4 specialization.  Each z slice in the (32,1,4) block is
// one whole warp and owns one face, including the final incomplete batch.
// Only independent side/q projections are parallelized: each projection
// retains the original ascending-d sum, and each output retains its ascending-q
// sum.  The subsequent deterministic element incidence gather is unchanged.
struct InteriorFaceDGDiffusionRecord2DWarp
{
  InteriorFaceDGDiffusionRecord2D record;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
    auto S = mfem::Reshape(record.shape, 2, record.nq, record.dof, record.nf);
    auto N = mfem::Reshape(record.weighted_normal_dshape,
                           2, record.nq, record.dof, record.nf);
    auto P = mfem::Reshape(record.penalty, record.nq, record.nf);
    auto F = mfem::Reshape(record.face_elements, 2, record.nf);
    auto D = mfem::Reshape(record.element_vdof_map,
                           record.dof, record.vdim, record.ne);
    auto C = mfem::Reshape(record.contribution, record.dof, 2, record.nf);

    // Lanes 0..7 own the two sides at four quadrature points.  Initialize all
    // lanes because every lane participates in each full-warp broadcast.
    mfem::real_t value = 0.0;
    mfem::real_t normal_flux = 0.0;
    if (lane < 8)
    {
      const int side = lane / 4;
      const int q = lane % 4;
      const int e = F(side, face);
      for (int d = 0; d < record.dof; ++d)
      {
        const mfem::real_t dof_value = record.x[D(d, 0, e)];
        value += S(side, q, d, face) * dof_value;
        normal_flux += N(side, q, d, face) * dof_value;
      }
    }

    const int side = lane / 16;
    const int d = lane % 16;
    mfem::real_t sum = 0.0;
    for (int q = 0; q < record.nq; ++q)
    {
      const mfem::real_t value0 = __shfl_sync(0xffffffffu, value, q);
      const mfem::real_t value1 = __shfl_sync(0xffffffffu, value, q + 4);
      const mfem::real_t normal0 = __shfl_sync(0xffffffffu, normal_flux, q);
      const mfem::real_t normal1 = __shfl_sync(0xffffffffu, normal_flux, q + 4);
      const mfem::real_t jump = value0 - value1;
      const mfem::real_t avg_flux = normal0 + normal1;
      const mfem::real_t penalty_jump = P(q, face) * jump;
      if (side == 0)
      {
        const mfem::real_t y0 =
          -S(0, q, d, face) * avg_flux +
          record.ip_sigma * N(0, q, d, face) * jump +
          S(0, q, d, face) * penalty_jump;
        sum += y0;
      }
      else
      {
        const mfem::real_t y1 =
          S(1, q, d, face) * avg_flux +
          record.ip_sigma * N(1, q, d, face) * jump -
          S(1, q, d, face) * penalty_jump;
        sum += y1;
      }
    }
    C(d, side, face) = sum;
#else
    record(face);
#endif
  }
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
    y[D(local_dof, 0, element)] += sum;
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
  mfem::Vector& y)
{
  MFEM_VERIFY(info.vdim == 1 &&
                scalar.Size() == info.ne * info.dof &&
                y.Size() == scalar.Size(),
              "Deterministic exact DG face gather requires one uniquely "
              "owned scalar row per discontinuous element dof.");
  EnsureInteriorFaceElementIncidence2D(info, work);
  work.face_contribution.SetSize(2 * info.dof * info.nf);
  work.face_contribution.UseDevice(true);

  InteriorFaceDGDiffusionRecord2D record;
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
  record.x = scalar.Read();
  record.contribution = work.face_contribution.Write();
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  if (info.dof == 16 && info.nq == 4 &&
      mfem::Device::Allows(mfem::Backend::CUDA) &&
      !mfem::Device::Allows(mfem::Backend::RAJA_CUDA))
  {
    mfem::forall_2D_batch(
      info.nf, 32, 1, 4, InteriorFaceDGDiffusionRecord2DWarp{record});
  }
  else
#endif
  {
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
    &work.penalty_metric);

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
