#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "mfem.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"

namespace dg_igr
{
namespace igr_internal
{
// Owned by one solver. Only geometry/reference-basis data are retained;
// side_alpha and side_q are overwritten from the current inputs each call.
struct FaceDensityCUDAProjectionCache
{
  dg_face::InteriorFaceSideDensityProjectionScratch work;
  dg_face::InteriorFaceSideDensityProjectionInfo info;
  mfem::Vector side_q;
  std::unique_ptr<mfem::IntegrationRule> owned_rule;
  std::unique_ptr<mfem::FaceQuadratureSpace> fqs;
  std::uint64_t rebuilds = 0, hits = 0;

  const mfem::FaceQuadratureSpace& Prepare(
    const mfem::FiniteElementSpace& vfes, const mfem::FiniteElementSpace& sfes,
    const mfem::IntegrationRule& ir)
  {
    mfem::Mesh* mesh = vfes.GetMesh();
    MFEM_VERIFY(mesh && sfes.GetMesh() == mesh && vfes.GetNE() > 0 &&
                sfes.GetNE() == vfes.GetNE(), "Face-density cache requires a nonempty shared mesh.");
    std::vector<double> rule;
    rule.reserve(2 * ir.GetNPoints());
    for (int i = 0; i < ir.GetNPoints(); ++i)
    {
      const auto& point = ir.IntPoint(i);
      // Supported faces are segments, so only x and weight define the rule.
      rule.insert(rule.end(), {point.x, point.weight});
    }
    const auto* fe = vfes.GetFE(0);
    const bool same = fqs && mesh_ == mesh && vfes_ == &vfes && sfes_ == &sfes &&
      vfec_ == vfes.FEColl() && sfec_ == sfes.FEColl() &&
      mesh_sequence_ == mesh->GetSequence() && nodes_sequence_ == mesh->GetNodesSequence() &&
      vfes_sequence_ == vfes.GetSequence() && sfes_sequence_ == sfes.GetSequence() &&
      ne_ == vfes.GetNE() && vsize_ == vfes.GetVSize() && vdim_ == vfes.GetVDim() &&
      ordering_ == static_cast<int>(vfes.GetOrdering()) &&
      dof_ == fe->GetDof() && order_ == fe->GetOrder() &&
      rule_order_ == ir.GetOrder() && rule_ == rule;
    if (same) { ++hits; return *fqs; }
    // FaceQuadratureSpace borrows its rule. Keep a private copy whose lifetime
    // includes the quadrature space, including for temporary/custom input rules.
    fqs.reset();
    owned_rule = std::make_unique<mfem::IntegrationRule>(ir.GetNPoints());
    owned_rule->SetOrder(ir.GetOrder());
    for (int i = 0; i < ir.GetNPoints(); ++i)
      owned_rule->IntPoint(i) = ir.IntPoint(i);
    fqs = std::make_unique<mfem::FaceQuadratureSpace>(*mesh, *owned_rule, mfem::FaceType::Interior);
    std::string reason;
    MFEM_VERIFY(dg_face::SupportsInteriorFaceSideDensityProjection2D(vfes, *fqs, &reason),
                "Unsupported cached face-density projection: " << reason);
    MFEM_VERIFY(dg_face::SupportsInteriorFaceScalarTraceProjection2D(sfes, *fqs, &reason),
                "Unsupported cached scalar face trace: " << reason);
    info = dg_face::BuildInteriorFaceSideDensityProjectionData2D(
      vfes, *fqs, work.shape, work.face_elements, work.element_vdof_map);
    mesh_ = mesh; vfes_ = &vfes; sfes_ = &sfes;
    vfec_ = vfes.FEColl(); sfec_ = sfes.FEColl();
    mesh_sequence_ = mesh->GetSequence(); nodes_sequence_ = mesh->GetNodesSequence();
    vfes_sequence_ = vfes.GetSequence(); sfes_sequence_ = sfes.GetSequence();
    ne_ = vfes.GetNE(); vsize_ = vfes.GetVSize(); vdim_ = vfes.GetVDim();
    ordering_ = static_cast<int>(vfes.GetOrdering()); dof_ = fe->GetDof(); order_ = fe->GetOrder();
    rule_order_ = ir.GetOrder(); rule_ = std::move(rule); ++rebuilds;
    return *fqs;
  }
private:
  const mfem::Mesh* mesh_ = nullptr;
  const mfem::FiniteElementSpace* vfes_ = nullptr;
  const mfem::FiniteElementSpace* sfes_ = nullptr;
  const mfem::FiniteElementCollection* vfec_ = nullptr;
  const mfem::FiniteElementCollection* sfec_ = nullptr;
  long mesh_sequence_ = -1, nodes_sequence_ = -1;
  long vfes_sequence_ = -1, sfes_sequence_ = -1;
  int ne_ = -1, vsize_ = -1, vdim_ = -1, ordering_ = -1;
  int dof_ = -1, order_ = -1, rule_order_ = -1;
  std::vector<double> rule_;
};

// Equivalent current-state side-density projection. Geometry is constructed
// on the host, as in the existing helper. State and valid per-element alpha
// are read only on the device. The caller owns alpha-policy validation.
dg_face::InteriorFaceSideDensityProjectionInfo
ProjectInteriorFaceDensityCoefficientCUDA(
  const mfem::FiniteElementSpace& vfes, const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& state, mfem::real_t scalar_alpha,
  const mfem::Vector* element_alpha, mfem::real_t rho_floor,
  mfem::Vector& side_q, dg_face::InteriorFaceSideDensityProjectionScratch& work,
  const dg_face::InteriorFaceSideDensityProjectionInfo* prepared_geometry = nullptr);
}
}
