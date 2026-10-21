#pragma once

#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include <memory>
#include <string>

namespace igrlab
{
namespace igr_internal
{
// Experimental FP64 volume-only mass + scalar diffusion operator. The caller
// retains the original full operator for certificates and adds its exact faces.
// Each QF supplies its own unchanged quadrature rule. Scalar VALUE L2, rank-one
// identity local/true maps, 2D quads, D1D <= 8 and both Q1D <= 8 are required.
// execution is "simt" or "wmma" (FP64 SM80+ 8x8x4). No implicit fallback.
// IGRLAB_IGR_TENSOR_VOLUME_STORAGE=cached|direct defaults to cached. Direct
// storage requires admitted collocated SIMT; other combinations report cached
// storage. Direct storage owns compact immutable geometry, sharing only bitwise
// identical element tiles, and reads the current QFs during every application.
// Factory performs the first coefficient refresh. Later stages must refresh
// after updating QFs. Direct storage retains QF object references until replaced
// by Refresh; each bound QF must outlive that binding (and any submitted device
// work). No QF data pointers are retained. Mesh/space lifetimes must exceed the
// operator's and geometry changes require reconstruction.
std::unique_ptr<mfem::Operator> MakeTensorVolumeOperator(
  mfem::ParFiniteElementSpace& sfes,
  const mfem::QuadratureFunction& invrho,
  const mfem::QuadratureFunction& ck,
  const std::string& execution);

void RefreshTensorVolumeCoefficients(
  mfem::Operator& op,
  const mfem::QuadratureFunction& invrho,
  const mfem::QuadratureFunction& ck);
}
}
