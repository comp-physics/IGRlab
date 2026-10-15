#pragma once

#include <cstddef>
#include <limits>

namespace dg::hdg::generic
{

inline constexpr int minimum_polynomial_order = 1;

// These are representation checks, not a chosen research-order ceiling.
// Local dense matrices use int row/column products; mesh-scaled storage and
// the CUDA aggregate arena are checked separately before allocation.
constexpr bool DenseLocalDimensionsFitInt(const int ni, const int nt) noexcept
{
  constexpr int maximum = std::numeric_limits<int>::max();
  constexpr std::size_t maximum_doubles =
      std::numeric_limits<std::size_t>::max() / sizeof(double);
  return ni > 0 && nt > 0 && ni <= maximum / ni &&
      nt <= maximum / nt && ni <= maximum / nt &&
      static_cast<std::size_t>(ni) <= maximum_doubles /
          static_cast<std::size_t>(ni) &&
      static_cast<std::size_t>(nt) <= maximum_doubles /
          static_cast<std::size_t>(nt) &&
      static_cast<std::size_t>(ni) <= maximum_doubles /
          static_cast<std::size_t>(nt);
}

constexpr bool ValidAssemblyQuadratureOrder(const int order) noexcept
{
  if(order <= 0) return false;
  // MFEM's tensor quadrilateral rule contains (order/2+1)^2 sites.
  const int points = order / 2 + 1;
  return points <= std::numeric_limits<int>::max() / points;
}

struct PolynomialShapePlan final
{
  int scalar_face_dofs = 0;
  int scalar_interior_dofs = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  int default_quadrature_order = 0;
  bool valid = false;
};

constexpr PolynomialShapePlan PlanPolynomialShape(
    const int order, const bool igr = false) noexcept
{
  PolynomialShapePlan shape{};
  constexpr int maximum = std::numeric_limits<int>::max();
  if(order < minimum_polynomial_order || order >= maximum ||
     order > (maximum - 2) / 2) return shape;
  const int face = order + 1;
  const int interior_components = igr ? 7 : 4;
  const int trace_components = igr ? 5 : 4;
  if(face > maximum / face) return shape;
  const int scalar = face * face;
  if(scalar > maximum / interior_components ||
     face > maximum / (4 * trace_components)) return shape;
  const int interior = interior_components * scalar;
  const int trace = 4 * trace_components * face;
  const int quadrature = 2 * order + 2;
  if(!DenseLocalDimensionsFitInt(interior, trace) ||
     !ValidAssemblyQuadratureOrder(quadrature)) return shape;
  shape.scalar_face_dofs = face;
  shape.scalar_interior_dofs = scalar;
  shape.interior_dofs = interior;
  shape.trace_dofs = trace;
  shape.default_quadrature_order = quadrature;
  shape.valid = true;
  return shape;
}

constexpr bool PolynomialRankIndicesFitInt(
    const PolynomialShapePlan& shape, const int elements,
    const int faces) noexcept
{
  if(!shape.valid || shape.interior_dofs <= 0 || shape.trace_dofs < 4 ||
     shape.trace_dofs % 4 != 0 || elements < 0 || faces < 0) return false;
  constexpr int maximum = std::numeric_limits<int>::max();
  const int face_dofs = shape.trace_dofs / 4;
  return elements <= maximum / shape.interior_dofs &&
      elements <= maximum / shape.trace_dofs &&
      faces <= maximum / face_dofs;
}

// A necessary bound for the rank-scaled packed CUDA geometry/matrix slices.
// The complete arena, including the selected trace solver, is checked later.
constexpr bool PolynomialRankWorkspaceSlicesFitInt(
    const PolynomialShapePlan& shape, const int elements,
    const int volume_order, const int face_order) noexcept
{
  if(!shape.valid || !DenseLocalDimensionsFitInt(
         shape.interior_dofs, shape.trace_dofs) || elements <= 0 ||
     !ValidAssemblyQuadratureOrder(volume_order) ||
     !ValidAssemblyQuadratureOrder(face_order) ||
     shape.scalar_interior_dofs <= 0 || shape.scalar_face_dofs <= 0)
    return false;
  constexpr std::size_t maximum = std::numeric_limits<int>::max();
  const auto product_fits = [](const std::size_t a, const std::size_t b,
                               const std::size_t c, const std::size_t d)
  {
    std::size_t product = a;
    const std::size_t factors[] = {b, c, d};
    for(const std::size_t factor : factors)
    {
      if(factor != 0 && product > maximum / factor) return false;
      product *= factor;
    }
    return true;
  };
  const std::size_t batches = static_cast<std::size_t>(elements);
  const std::size_t ni = static_cast<std::size_t>(shape.interior_dofs);
  const std::size_t nt = static_cast<std::size_t>(shape.trace_dofs);
  const std::size_t si = static_cast<std::size_t>(shape.scalar_interior_dofs);
  const std::size_t sf = static_cast<std::size_t>(shape.scalar_face_dofs);
  const std::size_t volume_axis = static_cast<std::size_t>(volume_order / 2 + 1);
  const std::size_t volume = volume_axis * volume_axis;
  const std::size_t face = static_cast<std::size_t>(face_order / 2 + 1);
  return volume <= maximum - 8 * face &&
      product_fits(batches, ni, ni, 1) &&
      product_fits(batches, ni, nt, 1) &&
      product_fits(batches, nt, nt, 1) &&
      product_fits(batches, ni, nt + 1, 1) &&
      product_fits(batches, volume, si, 2) &&
      product_fits(batches, 4 * face, si, 1) &&
      product_fits(batches, 4 * face, sf, 1);
}

} // namespace dg::hdg::generic
