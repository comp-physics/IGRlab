#pragma once

/**
 * @file numerics/dgsem/dgsem_basis.hpp
 * @brief DGSEM Gauss-Lobatto basis utilities.
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"

namespace mfem
{
namespace dgsem
{

/// @brief Cached 1D Gauss-Lobatto basis data on MFEM's reference interval [0,1].
struct LGLBasis1D
{
  int order = 0;
  int n = 0;
  mfem::Vector nodes;       ///< LGL nodes on [0,1]
  mfem::Vector weights;     ///< Quadrature weights on [0,1]
  mfem::Vector inv_weights; ///< 1/weights
  mfem::DenseMatrix D;      ///< Strong derivative matrix (d/dx)
  mfem::DenseMatrix Dsplit; ///< Split form matrix: D - 0.5 M^{-1} B
  mfem::DenseMatrix Dhat;   ///< Weak form matrix: -M^{-1} D^T M
};

/// @brief Build a 1D LGL basis bundle for the given polynomial order.
LGLBasis1D BuildLGLBasis1D(int order);

/// @brief Get cached 1D LGL basis data (builds on first use per order).
const LGLBasis1D& GetLGLBasis1D(int order);

} // namespace dgsem
} // namespace mfem
