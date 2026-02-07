#pragma once

/**
 * @file numerics/dgsem/dgsem_flux.hpp
 * @brief DGSEM flux-differencing volume kernels.
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"

#include <array>
#include <functional>
#include <vector>

namespace mfem
{
namespace dgsem
{

/// @brief Symmetric two-point volume flux callback.
/// Signature: flux(u_left, u_right, direction, flux_out)
using VolumeFlux = std::function<void(const mfem::Vector&,
                                      const mfem::Vector&,
                                      int,
                                      mfem::Vector&)>;

/// @brief 1D DGSEM flux differencing kernel.
void FluxDifferencing1D(const mfem::DenseMatrix& u,
                         const mfem::DenseMatrix& dsplit,
                         const VolumeFlux& volume_flux,
                         mfem::DenseMatrix& du,
                         double scale = 1.0);

/// @brief 2D DGSEM flux differencing kernel (tensor-product).
void FluxDifferencing2D(const mfem::DenseMatrix& u,
                         const mfem::DenseMatrix& dsplit,
                         const std::vector<std::array<int, 3>>& dof_to_ijk,
                         const VolumeFlux& volume_flux,
                         mfem::DenseMatrix& du,
                         double scale = 1.0);

} // namespace dgsem
} // namespace mfem
