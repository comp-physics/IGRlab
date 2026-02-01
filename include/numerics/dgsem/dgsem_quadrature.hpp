#pragma once

/**
 * @file numerics/dgsem/dgsem_quadrature.hpp
 * @brief DGSEM quadrature helpers (Gauss-Lobatto rules).
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"

namespace mfem
{
namespace dgsem
{

/**
 * @brief Return the Gauss-Lobatto face integration rule for a DG order.
 *
 * Uses a Gauss-Lobatto 1D rule with p+1 points (exactness 2p-1).
 * Intended for segment faces in 2D DGSEM (tensor-product meshes).
 */
inline const mfem::IntegrationRule& GetLGLFaceRule(int order)
{
  static mfem::IntegrationRules lobatto_rules(0, mfem::Quadrature1D::GaussLobatto);
  const int face_order = (order > 0) ? (2 * order - 1) : 1;
  return lobatto_rules.Get(mfem::Geometry::SEGMENT, face_order);
}

} // namespace dgsem
} // namespace mfem
