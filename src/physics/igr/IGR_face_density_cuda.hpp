#pragma once

#include "igrlab/compat.hpp"
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include "mfem.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"
#include "IGR_face_density_tensor.hpp"

namespace igrlab
{
namespace igr_internal
{
// Owned by one solver. Only geometry/reference-basis data are retained;
// side_alpha, side_q and any tensor traces are overwritten from current inputs.
struct FaceDensityCUDAProjectionCache
{
  dg_face::InteriorFaceSideDensityProjectionScratch work;
  dg_face::InteriorFaceSideDensityProjectionInfo info;
  mfem::Vector side_q;
  std::unique_ptr<mfem::IntegrationRule> owned_rule;
  std::unique_ptr<mfem::FaceQuadratureSpace> fqs;
  std::uint64_t rebuilds = 0, hits = 0;

  bool UsesTensor() const
  {
    const auto* space = dynamic_cast<const face_density_detail::QuadratureSpace*>(fqs.get());
    return space && bool(space->tensor);
  }
  const std::string& TensorFallbackReason() const { return tensor_fallback_reason_; }

  const mfem::FaceQuadratureSpace& Prepare(
    const mfem::FiniteElementSpace& vfes, const mfem::FiniteElementSpace& sfes,
    const mfem::IntegrationRule& ir)
  {
    mfem::Mesh* mesh = vfes.GetMesh();
    MFEM_VERIFY(mesh && sfes.GetMesh() == mesh && vfes.GetNE() > 0 &&
                sfes.GetNE() == vfes.GetNE(), "Face-density cache requires a nonempty shared mesh.");
    const auto execution = face_density_detail::RequestedExecution();
    auto key = face_density_detail::MakeLayoutKey(vfes, sfes, ir, execution);
    if (fqs && key_ == key) { ++hits; return *fqs; }
    // FaceQuadratureSpace borrows its rule. Keep a private copy whose lifetime
    // includes the quadrature space, including for temporary/custom input rules.
    fqs.reset();
    owned_rule = std::make_unique<mfem::IntegrationRule>(ir.GetNPoints());
    owned_rule->SetOrder(ir.GetOrder());
    for (int i = 0; i < ir.GetNPoints(); ++i)
      owned_rule->IntPoint(i) = ir.IntPoint(i);
    auto next = std::make_unique<face_density_detail::QuadratureSpace>(
      *mesh, *owned_rule, mfem::FaceType::Interior);
    std::string reason;
    MFEM_VERIFY(dg_face::SupportsInteriorFaceSideDensityProjection2D(vfes, *next, &reason),
                "Unsupported cached face-density projection: " << reason);
    MFEM_VERIFY(dg_face::SupportsInteriorFaceScalarTraceProjection2D(sfes, *next, &reason),
                "Unsupported cached scalar face trace: " << reason);
    // Release prior dense capacities before tensor admission/building. In
    // particular, never construct the dense per-face shape table speculatively.
    work.shape.Destroy(); work.side_alpha.Destroy();
    work.face_elements.DeleteAll(); work.element_vdof_map.DeleteAll();
    tensor_fallback_reason_.clear();
    if (execution == face_density_detail::Execution::Tensor)
    {
      try
      {
        next->tensor = std::make_unique<face_density_detail::TensorData>();
        if (!face_density_detail::BuildTensorData(
              vfes, sfes, *next, *next->tensor, work, tensor_fallback_reason_))
          next->tensor.reset();
      }
      catch (const std::bad_alloc&)
      {
        next->tensor.reset();
        tensor_fallback_reason_ = "allocation";
      }
    }
    info = next->tensor ? next->tensor->info :
      dg_face::BuildInteriorFaceSideDensityProjectionData2D(
        vfes, *next, work.shape, work.face_elements, work.element_vdof_map);
    if (execution == face_density_detail::Execution::Tensor && mfem::Mpi::Root())
      std::cout << "[igr-face-density] event=execution requested=tensor selected="
                << (next->tensor ? "tensor" : "dense") << " reason="
                << (next->tensor ? "supported" : tensor_fallback_reason_) << std::endl;
    fqs = std::move(next);
    key_ = std::move(key); ++rebuilds;
    return *fqs;
  }
private:
  face_density_detail::LayoutKey key_;
  std::string tensor_fallback_reason_;
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
