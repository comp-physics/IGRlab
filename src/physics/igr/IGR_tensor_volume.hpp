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
// Factory performs the first coefficient refresh. Later stages must refresh
// after updating QFs; no QF data pointers are retained. Mesh/space lifetimes
// must exceed the operator's and geometry changes require reconstruction.
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
