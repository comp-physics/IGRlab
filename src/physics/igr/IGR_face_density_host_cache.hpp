#pragma once

#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace igrlab
{
namespace igr_internal
{
// Private per-solver acceleration of the existing host projector. Only its
// quadrature/reference geometry and native maps are retained. Current state,
// alpha and rho_floor are always consumed by the original arithmetic below.
// The returned private FQS is for the direct packed face builders only. Those
// builders do not register its owned rule with MFEM's pointer-keyed geometric
// factor or quadrature-interpolator caches (in particular, no GetWeights()).
class HostFaceDensityPreparationCache
{
public:
  explicit HostFaceDensityPreparationCache(
    std::size_t memory_budget = 128u * 1024u * 1024u)
    : memory_budget_(memory_budget) {}

  const mfem::FaceQuadratureSpace* Prepare(
    const mfem::FiniteElementSpace& vfes,
    const mfem::FiniteElementSpace& sfes,
    const mfem::IntegrationRule& ir,
    bool* layout_changed = nullptr)
  {
    if (layout_changed) { *layout_changed = false; }
    if (!Admits(vfes, sfes, ir))
    {
      if (layout_changed) { *layout_changed = bool(pack_); }
      pack_.reset();
      return nullptr;
    }
    if (pack_ && Matches(vfes, sfes, ir)) { return pack_->fqs.get(); }
    if (layout_changed) { *layout_changed = bool(pack_); }
    pack_.reset(); // Release old capacities before charging/building a new pack.
    try
    {
      auto next = std::make_unique<Pack>();
      next->rule = std::make_unique<mfem::IntegrationRule>(ir.GetNPoints());
      next->rule->SetOrder(ir.GetOrder());
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
        next->rule->IntPoint(q) = ir.IntPoint(q);
      }
      next->fqs = std::make_unique<DirectFaceQuadratureSpace>(
        *vfes.GetMesh(), *next->rule, mfem::FaceType::Interior);
      std::string reason;
      if (!dg_face::SupportsInteriorFaceSideDensityProjection2D(
            vfes, *next->fqs, &reason) ||
          !dg_face::SupportsInteriorFaceScalarTraceProjection2D(
            sfes, *next->fqs, &reason))
      {
        return nullptr; // Caller executes the original checks and generic path.
      }
      next->info = dg_face::BuildInteriorFaceSideDensityProjectionData2D(
        vfes, *next->fqs, next->work.shape, next->work.face_elements,
        next->work.element_vdof_map);
      dg_face::BuildNativeElementVDofMap(sfes, next->scalar_map);
      next->mesh = vfes.GetMesh();
      next->vfes = &vfes; next->sfes = &sfes;
      next->vfec = vfes.FEColl(); next->sfec = sfes.FEColl();
      next->vfe = vfes.GetFE(0); next->sfe = sfes.GetFE(0);
      next->mesh_sequence = next->mesh->GetSequence();
      next->nodes_sequence = next->mesh->GetNodesSequence();
      next->vsequence = vfes.GetSequence(); next->ssequence = sfes.GetSequence();
      next->vsize = vfes.GetVSize(); next->ssize = sfes.GetVSize();
      next->vordering = vfes.GetOrdering(); next->sordering = sfes.GetOrdering();
      next->vcomm = static_cast<const mfem::ParFiniteElementSpace&>(vfes).GetComm();
      next->scomm = static_cast<const mfem::ParFiniteElementSpace&>(sfes).GetComm();
      next->nodes = next->mesh->GetNodes();
      next->nodes_fes = next->mesh->GetNodalFESpace();
      next->nodes_fes_sequence = next->nodes_fes ? next->nodes_fes->GetSequence() : -1;
      if (next->nodes_fes)
      {
        next->nodes_fec = next->nodes_fes->FEColl();
        next->nodes_fe = next->nodes_fes->GetFE(0);
        next->nodes_vsize = next->nodes_fes->GetVSize();
        next->nodes_vdim = next->nodes_fes->GetVDim();
        next->nodes_ordering = next->nodes_fes->GetOrdering();
        next->nodes_order = next->nodes_fe->GetOrder();
        next->nodes_dof = next->nodes_fe->GetDof();
        next->nodes_map = next->nodes_fe->GetMapType();
        next->nodes_geometry = next->nodes_fe->GetGeomType();
      }
      next->vfe_order = next->vfe->GetOrder(); next->sfe_order = next->sfe->GetOrder();
      next->vfe_dof = next->vfe->GetDof(); next->sfe_dof = next->sfe->GetDof();
      next->vbasis = static_cast<const mfem::L2_FECollection*>(next->vfec)->GetBasisType();
      next->sbasis = static_cast<const mfem::L2_FECollection*>(next->sfec)->GetBasisType();
      for (int f = 0; f < next->mesh->GetNumFaces(); ++f)
      {
        next->faces.push_back(FaceKey(*next->mesh, f));
      }
      pack_ = std::move(next);
      return pack_->fqs.get();
    }
    catch (const std::bad_alloc&)
    {
      return nullptr; // Optional acceleration never changes generic admission.
    }
  }

  dg_face::InteriorFaceSideDensityProjectionInfo Project(
    const mfem::FiniteElementSpace& vfes,
    const mfem::ParGridFunction& state,
    mfem::real_t scalar_alpha,
    const mfem::Vector* element_alpha,
    mfem::real_t rho_floor,
    mfem::Vector& side_q)
  {
    MFEM_VERIFY(pack_ && pack_->vfes == &vfes,
                "Host face-density projection requires prepared geometry.");
    return dg_face::ProjectInteriorFaceDensityCoefficient2DImpl(
      vfes, *pack_->fqs, state, scalar_alpha, element_alpha, rho_floor,
      side_q, &pack_->work, &pack_->info);
  }

  const mfem::Vector& CurrentSideAlpha() const
  {
    MFEM_VERIFY(pack_, "Host face-density projection requires prepared geometry.");
    return pack_->work.side_alpha;
  }

private:
  class DirectFaceQuadratureSpace final : public mfem::FaceQuadratureSpace
  {
  public:
    using mfem::FaceQuadratureSpace::FaceQuadratureSpace;
  private:
    const mfem::Vector& GetGeometricFactorWeights() const override
    {
      MFEM_ABORT("Private host face-density FQS is for direct face traversal, "
                 "not MFEM pointer-keyed geometric-factor caching.");
      return weights;
    }
  };

  struct Pack
  {
    // FQS borrows its rule; destruction order keeps the rule alive until last.
    std::unique_ptr<mfem::IntegrationRule> rule;
    std::unique_ptr<DirectFaceQuadratureSpace> fqs;
    dg_face::InteriorFaceSideDensityProjectionScratch work;
    dg_face::InteriorFaceSideDensityProjectionInfo info;
    mfem::Array<int> scalar_map;
    std::vector<std::array<int, 6>> faces;
    mfem::Mesh* mesh = nullptr;
    const mfem::FiniteElementSpace* vfes = nullptr;
    const mfem::FiniteElementSpace* sfes = nullptr;
    const mfem::FiniteElementCollection* vfec = nullptr;
    const mfem::FiniteElementCollection* sfec = nullptr;
    const mfem::FiniteElement* vfe = nullptr;
    const mfem::FiniteElement* sfe = nullptr;
    const mfem::GridFunction* nodes = nullptr;
    const mfem::FiniteElementSpace* nodes_fes = nullptr;
    const mfem::FiniteElementCollection* nodes_fec = nullptr;
    const mfem::FiniteElement* nodes_fe = nullptr;
    long mesh_sequence = -1, nodes_sequence = -1, vsequence = -1, ssequence = -1;
    long nodes_fes_sequence = -1;
    int vsize = -1, ssize = -1, vfe_order = -1, sfe_order = -1;
    int vfe_dof = -1, sfe_dof = -1, vbasis = -1, sbasis = -1;
    int nodes_vsize = -1, nodes_vdim = -1, nodes_order = -1, nodes_dof = -1;
    int nodes_map = -1, nodes_geometry = -1;
    mfem::Ordering::Type vordering = mfem::Ordering::byNODES;
    mfem::Ordering::Type sordering = mfem::Ordering::byNODES;
    mfem::Ordering::Type nodes_ordering = mfem::Ordering::byNODES;
    MPI_Comm vcomm = MPI_COMM_NULL, scomm = MPI_COMM_NULL;
  };

  static std::array<int, 6> FaceKey(const mfem::Mesh& mesh, int face)
  {
    int e1, e2, i1, i2;
    mesh.GetFaceElements(face, &e1, &e2);
    mesh.GetFaceInfos(face, &i1, &i2);
    return {{face, e1, e2, i1, i2,
             mesh.GetFaceInformation(face).IsShared() ? 1 : 0}};
  }

  static bool NativeMapMatches(const mfem::FiniteElementSpace& fes,
                               const mfem::Array<int>& map)
  {
    const int dof = fes.GetFE(0)->GetDof(), vdim = fes.GetVDim();
    const int* saved = map.HostRead();
    mfem::Array<int> sdofs;
    for (int e = 0; e < fes.GetNE(); ++e)
    {
      fes.GetElementDofs(e, sdofs);
      if (sdofs.Size() != dof) { return false; }
      for (int c = 0; c < vdim; ++c)
      {
        for (int d = 0; d < dof; ++d)
        {
          const int native = mfem::FiniteElementSpace::DecodeDof(
            fes.DofToVDof(sdofs[d], c));
          if (saved[d + dof * (c + vdim * e)] != native) { return false; }
        }
      }
    }
    return true;
  }

  bool Admits(const mfem::FiniteElementSpace& vfes,
              const mfem::FiniteElementSpace& sfes,
              const mfem::IntegrationRule& ir) const
  {
    auto* mesh = vfes.GetMesh();
    const auto* pvfes = dynamic_cast<const mfem::ParFiniteElementSpace*>(&vfes);
    const auto* psfes = dynamic_cast<const mfem::ParFiniteElementSpace*>(&sfes);
    if (!mesh || sfes.GetMesh() != mesh || !pvfes || !psfes ||
        mesh->Dimension() != 2 || mesh->SpaceDimension() != 2 ||
        mesh->Nonconforming() || vfes.IsVariableOrder() || sfes.IsVariableOrder() ||
        vfes.GetNE() <= 0 || sfes.GetNE() != vfes.GetNE() ||
        vfes.GetVDim() < 1 || sfes.GetVDim() != 1 || ir.GetNPoints() <= 0 ||
        !dynamic_cast<const mfem::L2_FECollection*>(vfes.FEColl()) ||
        !dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl())) { return false; }
    const auto* vfe = vfes.GetFE(0);
    const auto* sfe = sfes.GetFE(0);
    if (vfe->GetDof() <= 0 || sfe->GetDof() <= 0 ||
        vfe->GetMapType() != mfem::FiniteElement::VALUE ||
        sfe->GetMapType() != mfem::FiniteElement::VALUE) { return false; }
    for (int e = 0; e < vfes.GetNE(); ++e)
    {
      if (mesh->GetElementBaseGeometry(e) != mfem::Geometry::SQUARE ||
          vfes.GetFE(e) != vfe || sfes.GetFE(e) != sfe) { return false; }
    }
    std::size_t local_faces = 0;
    for (int f = 0; f < mesh->GetNumFaces(); ++f)
    {
      const auto face = mesh->GetFaceInformation(f);
      if (face.IsInterior() && !face.IsShared()) { ++local_faces; }
    }
    if (!local_faces) { return false; }
    if (std::size_t(ir.GetNPoints()) >
        std::size_t(std::numeric_limits<int>::max()) /
          std::size_t(mesh->GetNumFaces())) { return false; }
    // Checked conservative storage charge, including shape/map arrays, current
    // alpha workspace and FQS/key metadata. The ceiling limits acceleration,
    // not the admissible polynomial order or generic algorithm.
    std::size_t bytes = sizeof(Pack), entries = 1;
    auto charge = [&](std::initializer_list<std::size_t> factors,
                      std::size_t width)
    {
      entries = 1;
      for (const auto factor : factors)
      {
        if (!factor || entries > std::size_t(std::numeric_limits<int>::max()) / factor)
          return false;
        entries *= factor;
      }
      if (entries > (memory_budget_ - std::min(bytes, memory_budget_)) / width)
        return false;
      bytes += entries * width;
      return true;
    };
    return bytes <= memory_budget_ &&
      charge({2u, std::size_t(ir.GetNPoints()), std::size_t(vfe->GetDof()), local_faces}, sizeof(mfem::real_t)) &&
      charge({std::size_t(vfe->GetDof()), std::size_t(vfes.GetVDim()), std::size_t(vfes.GetNE())}, sizeof(int)) &&
      charge({std::size_t(sfe->GetDof()), std::size_t(sfes.GetNE())}, sizeof(int)) &&
      charge({2u, local_faces}, sizeof(int) + sizeof(mfem::real_t)) &&
      charge({std::size_t(mesh->GetNumFaces())}, 128u) &&
      charge({std::size_t(ir.GetNPoints())}, sizeof(mfem::IntegrationPoint));
  }

  bool Matches(const mfem::FiniteElementSpace& vfes,
               const mfem::FiniteElementSpace& sfes,
               const mfem::IntegrationRule& ir) const
  {
    const auto& p = *pack_;
    if (p.mesh != vfes.GetMesh() || p.vfes != &vfes || p.sfes != &sfes ||
        p.vfec != vfes.FEColl() || p.sfec != sfes.FEColl() ||
        p.vfe != vfes.GetFE(0) || p.sfe != sfes.GetFE(0) ||
        p.mesh_sequence != p.mesh->GetSequence() ||
        p.nodes_sequence != p.mesh->GetNodesSequence() ||
        p.nodes != p.mesh->GetNodes() || p.nodes_fes != p.mesh->GetNodalFESpace() ||
        (p.nodes_fes &&
          (p.nodes_fes_sequence != p.nodes_fes->GetSequence() ||
           p.nodes_fec != p.nodes_fes->FEColl() || p.nodes_fe != p.nodes_fes->GetFE(0) ||
           p.nodes_vsize != p.nodes_fes->GetVSize() ||
           p.nodes_vdim != p.nodes_fes->GetVDim() ||
           p.nodes_ordering != p.nodes_fes->GetOrdering() ||
           p.nodes_order != p.nodes_fe->GetOrder() || p.nodes_dof != p.nodes_fe->GetDof() ||
           p.nodes_map != p.nodes_fe->GetMapType() ||
           p.nodes_geometry != p.nodes_fe->GetGeomType())) ||
        p.vsequence != vfes.GetSequence() || p.ssequence != sfes.GetSequence() ||
        p.vsize != vfes.GetVSize() || p.ssize != sfes.GetVSize() ||
        p.info.ne != vfes.GetNE() || p.info.vdim != vfes.GetVDim() ||
        p.vordering != vfes.GetOrdering() || p.sordering != sfes.GetOrdering() ||
        p.vcomm != static_cast<const mfem::ParFiniteElementSpace&>(vfes).GetComm() ||
        p.scomm != static_cast<const mfem::ParFiniteElementSpace&>(sfes).GetComm() ||
        p.vfe_order != p.vfe->GetOrder() || p.sfe_order != p.sfe->GetOrder() ||
        p.vfe_dof != p.vfe->GetDof() || p.sfe_dof != p.sfe->GetDof() ||
        p.vbasis != static_cast<const mfem::L2_FECollection*>(vfes.FEColl())->GetBasisType() ||
        p.sbasis != static_cast<const mfem::L2_FECollection*>(sfes.FEColl())->GetBasisType() ||
        p.rule->GetOrder() != ir.GetOrder() || p.rule->GetNPoints() != ir.GetNPoints() ||
        p.faces.size() != std::size_t(p.mesh->GetNumFaces())) { return false; }
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const auto& old = p.rule->IntPoint(q);
      const auto& now = ir.IntPoint(q);
      if (std::memcmp(&old.x, &now.x, sizeof(old.x)) ||
          std::memcmp(&old.weight, &now.weight, sizeof(old.weight))) { return false; }
    }
    for (int f = 0; f < p.mesh->GetNumFaces(); ++f)
    {
      if (p.faces[f] != FaceKey(*p.mesh, f)) { return false; }
    }
    return NativeMapMatches(vfes, p.work.element_vdof_map) &&
           NativeMapMatches(sfes, p.scalar_map);
  }

  std::size_t memory_budget_;
  std::unique_ptr<Pack> pack_;
};
}
}
