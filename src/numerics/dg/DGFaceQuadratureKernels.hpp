// Internal-only helper/kernels for DG face quadrature projection.
// This file is intentionally included after MFEM backend headers.
#pragma once

#include <string>

namespace dg_igr
{
namespace dg_face
{

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
};

struct InteriorFaceDGDiffusionApplyScratch
{
  mfem::Vector shape;
  mfem::Vector weighted_normal_dshape;
  mfem::Vector penalty;
  mfem::Array<int> face_elements;
  mfem::Array<int> element_vdof_map;
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
};

struct SharedFaceDGDiffusionApplyScratch
{
  mfem::Vector shape;
  mfem::Vector weighted_normal_dshape;
  mfem::Vector penalty;
  mfem::Array<int> face_elements;
  mfem::Array<int> element_vdof_map;
  mfem::Array<int> face_nbr_dof_map;
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
  mfem::Array<int>& element_vdof_map)
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

  MFEM_VERIFY(side_q.Size() == 2 * info.nq * info.nf,
              "Exact DG diffusion face apply received mismatched side qdata.");

  shape.SetSize(info.shape_size);
  weighted_normal_dshape.SetSize(info.weighted_normal_dshape_size);
  penalty.SetSize(info.penalty_size);
  shape.UseDevice(true);
  weighted_normal_dshape.UseDevice(true);
  penalty.UseDevice(true);
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

      mfem::real_t penalty_q = 0.0;
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
          base_weight * (normal * normal);
        ni.Set(base_weight, normal);
        mfem::CalcAdjugate(elem_tr.Jacobian(), adjJ);
        adjJ.Mult(ni, nh);
        side_dshape.Mult(nh, side_dshape_dn);
        penalty_q += side_q_value * penalty_factor_value;

        for (int d = 0; d < info.dof; ++d)
        {
          shape_out(side, q, d, of) = side_shape(d);
          ndshape_out(side, q, d, of) =
            side_q_value * side_dshape_dn(d);
        }
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
  mfem::Array<int>& face_nbr_dof_map)
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

  MFEM_VERIFY(side_q.Size() == 2 * info.nq * info.nf,
              "Shared exact DG diffusion face data received mismatched "
              "side qdata.");

  shape.SetSize(info.shape_size);
  weighted_normal_dshape.SetSize(info.weighted_normal_dshape_size);
  penalty.SetSize(info.penalty_size);
  shape.UseDevice(true);
  weighted_normal_dshape.UseDevice(true);
  penalty.UseDevice(true);
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

      mfem::real_t penalty_q = 0.0;
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
          base_weight * (normal * normal);
        ni.Set(base_weight, normal);
        mfem::CalcAdjugate(elem_tr.Jacobian(), adjJ);
        adjJ.Mult(ni, nh);
        side_dshape.Mult(nh, side_dshape_dn);
        penalty_q += side_q_value * penalty_factor_value;

        for (int d = 0; d < info.dof; ++d)
        {
          shape_out(side, q, d, sf) = side_shape(d);
          ndshape_out(side, q, d, sf) =
            side_q_value * side_dshape_dn(d);
        }
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
      if (rho < rho_floor) { rho = rho_floor; }
      Q(side, q, face) = alpha / rho;
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

struct InteriorFaceDGDiffusionApply2D
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
  mfem::real_t* y = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto S = mfem::Reshape(shape, 2, nq, dof, nf);
    auto N = mfem::Reshape(weighted_normal_dshape, 2, nq, dof, nf);
    auto P = mfem::Reshape(penalty, nq, nf);
    auto F = mfem::Reshape(face_elements, 2, nf);
    auto D = mfem::Reshape(element_vdof_map, dof, vdim, ne);

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

      const int e0 = F(0, face);
      const int e1 = F(1, face);
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
        ::AtomicAdd(y[D(d, 0, e0)], y0);
        ::AtomicAdd(y[D(d, 0, e1)], y1);
      }
    }
  }
};

struct SharedFaceDGDiffusionApply2D
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
  mfem::real_t* y = nullptr;

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

    const int e0 = F(face);
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
        ::AtomicAdd(y[D(d, 0, e0)], y0);
      }
    }
  }
};

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
  MFEM_VERIFY(state.ParFESpace() == &vfes,
              "Side-density projection state/FESpace mismatch.");
  InteriorFaceSideDensityProjectionScratch local_scratch;
  InteriorFaceSideDensityProjectionScratch& work =
    scratch ? *scratch : local_scratch;
  const auto info = BuildInteriorFaceSideDensityProjectionData2D(
    vfes,
    fqs,
    work.shape,
    work.face_elements,
    work.element_vdof_map);

  side_q.SetSize(2 * info.nq * info.nf);
  side_q.UseDevice(true);

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
        if (rho < rho_floor) { rho = rho_floor; }
        Q(side, q, f) = alpha / rho;
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
ProjectSharedFaceDensityCoefficient2D(
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::IntegrationRule& ir,
  const mfem::ParGridFunction& state,
  mfem::real_t alpha,
  mfem::real_t rho_floor,
  mfem::Vector& side_q,
  SharedFaceSideDensityProjectionScratch* scratch = nullptr)
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
      if (rho0 < rho_floor) { rho0 = rho_floor; }
      if (rho1 < rho_floor) { rho1 = rho_floor; }
      Q(0, q, f) = alpha / rho0;
      Q(1, q, f) = alpha / rho1;
    }
  }

  SharedFaceSideDensityProjectionInfo out = info;
  out.qf_size = side_q.Size();
  return out;
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
    work.element_vdof_map);

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
  InteriorFaceDGDiffusionApply2D kernel;
  kernel.dof = info.dof;
  kernel.vdim = info.vdim;
  kernel.nq = info.nq;
  kernel.nf = info.nf;
  kernel.ne = info.ne;
  kernel.ip_sigma = ip_sigma;
  kernel.shape = work.shape.Read();
  kernel.weighted_normal_dshape = work.weighted_normal_dshape.Read();
  kernel.penalty = work.penalty.Read();
  kernel.face_elements = work.face_elements.Read();
  kernel.element_vdof_map = work.element_vdof_map.Read();
  kernel.x = scalar.Read();
  kernel.y = y.ReadWrite();
  mfem::forall(info.nf, kernel);
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
  InteriorFaceDGDiffusionApply2D kernel;
  kernel.dof = info.dof;
  kernel.vdim = info.vdim;
  kernel.nq = info.nq;
  kernel.nf = info.nf;
  kernel.ne = info.ne;
  kernel.ip_sigma = ip_sigma;
  kernel.shape = work.shape.Read();
  kernel.weighted_normal_dshape = work.weighted_normal_dshape.Read();
  kernel.penalty = work.penalty.Read();
  kernel.face_elements = work.face_elements.Read();
  kernel.element_vdof_map = work.element_vdof_map.Read();
  kernel.x = scalar.Read();
  kernel.y = y.ReadWrite();
  mfem::forall(info.nf, kernel);
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
  SharedFaceDGDiffusionApply2D kernel;
  kernel.dof = info.dof;
  kernel.vdim = info.vdim;
  kernel.nq = info.nq;
  kernel.nf = info.nf;
  kernel.ne = info.ne;
  kernel.ip_sigma = ip_sigma;
  kernel.shape = work.shape.Read();
  kernel.weighted_normal_dshape = work.weighted_normal_dshape.Read();
  kernel.penalty = work.penalty.Read();
  kernel.face_elements = work.face_elements.Read();
  kernel.element_vdof_map = work.element_vdof_map.Read();
  kernel.face_nbr_dof_map = work.face_nbr_dof_map.Read();
  kernel.x = scalar.Read();
  kernel.x_face_nbr = scalar_face_nbr.Read();
  kernel.y = y.ReadWrite();
  mfem::forall(info.nf, kernel);
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
    work.face_nbr_dof_map);

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
  SharedFaceDGDiffusionApply2D kernel;
  kernel.dof = info.dof;
  kernel.vdim = info.vdim;
  kernel.nq = info.nq;
  kernel.nf = info.nf;
  kernel.ne = info.ne;
  kernel.ip_sigma = ip_sigma;
  kernel.shape = work.shape.Read();
  kernel.weighted_normal_dshape = work.weighted_normal_dshape.Read();
  kernel.penalty = work.penalty.Read();
  kernel.face_elements = work.face_elements.Read();
  kernel.element_vdof_map = work.element_vdof_map.Read();
  kernel.face_nbr_dof_map = work.face_nbr_dof_map.Read();
  kernel.x = scalar.Read();
  kernel.x_face_nbr = scalar_face_nbr.Read();
  kernel.y = y.ReadWrite();
  mfem::forall(info.nf, kernel);
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
} // namespace dg_igr
