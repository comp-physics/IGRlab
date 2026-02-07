#pragma once

/**
 * @file numerics/dgsem/dgsem_operator.hpp
 * @brief DGSEM operator helpers (volume + surface assembly).
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"

namespace mfem
{
class DGHyperbolicConservationLaws;

namespace dgsem
{

/// @brief Helper to assemble DGSEM volume/surface contributions.
class DGSEMOperator
{
public:
  explicit DGSEMOperator(const DGHyperbolicConservationLaws& op);

  /// @brief Assemble DGSEM surface contributions into z (overwrites z).
  real_t AssembleSurface(const Vector& x, Vector& z) const;

  /// @brief Add DGSEM volume contributions into z (accumulates into z).
  real_t AssembleVolume(const Vector& x, Vector& z) const;

  /// @brief Apply inverse mass matrices element-wise.
  void ApplyInverseMass(const Vector& z, Vector& y) const;

  /// @brief Assemble surface + volume and apply M^{-1}, returns max wave speed.
  real_t AssembleResidual(const Vector& x, Vector& y) const;

private:
  const BdrHyperbolicFluxIntegrator* FindBdrIntegrator(int attr) const;

  const DGHyperbolicConservationLaws& op_;
};

} // namespace dgsem
} // namespace mfem
