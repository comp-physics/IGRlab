#pragma once

/**
 * @file numerics/riemann/dgsem_surface_flux.hpp
 * @brief DGSEM-aware surface flux selection (Euler).
 * @ingroup numerics_riemann
 */

#include "mfem.hpp"

#include "numerics/dgsem/dgsem_entropy_flux.hpp"
#include "numerics/riemann/diffusive_rusanov.hpp"
#include "numerics/riemann/face_side.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/igr/igr_flux.hpp"

namespace mfem
{

/// @brief DGSEM surface flux selection for Euler.
enum class DGSEMSurfaceFluxType
{
  Rusanov,
  DiffusiveRusanov,
  EntropyStableWinters,
  EntropyStableLFEntropyVars
};

/// @brief DGSEM surface flux wrapper with entropy-stable option.
class DGSEMSurfaceFlux : public NumericalFlux
{
public:
  explicit DGSEMSurfaceFlux(const FluxFunction &fluxFunction);

  void SetSurfaceFluxType(DGSEMSurfaceFluxType type) { type_ = type; }
  void SetEntropyFluxType(dgsem::EntropyFluxType type) { entropy_type_ = type; }
  void SetGamma(double gamma) { gamma_ = gamma; }
  void SetEntropyJacobianMode(dgsem::EntropyJacobianMode mode)
  { entropy_jacobian_mode_ = mode; }

  DGSEMSurfaceFluxType GetSurfaceFluxType() const { return type_; }
  dgsem::EntropyFluxType GetEntropyFluxType() const { return entropy_type_; }
  dgsem::EntropyJacobianMode GetEntropyJacobianMode() const
  { return entropy_jacobian_mode_; }

  real_t Eval(const Vector &state1, const Vector &state2,
              const Vector &nor, FaceElementTransformations &Tr,
              Vector &flux) const override;

  void Grad(int side, const Vector &state1, const Vector &state2,
            const Vector &nor, FaceElementTransformations &Tr,
            DenseMatrix &grad) const override;

  real_t Average(const Vector &state1, const Vector &state2,
                 const Vector &nor, FaceElementTransformations &Tr,
                 Vector &flux) const override;

  void AverageGrad(int side, const Vector &state1, const Vector &state2,
                   const Vector &nor, FaceElementTransformations &Tr,
                   DenseMatrix &grad) const override;

private:
  const FluxFunction &flux_fn_;
  const FaceSideExplicitFlux *side_explicit_flux_ = nullptr;
  DGSEMSurfaceFluxType type_ = DGSEMSurfaceFluxType::Rusanov;
  dgsem::EntropyFluxType entropy_type_ = dgsem::EntropyFluxType::Ranocha;
  dgsem::EntropyJacobianMode entropy_jacobian_mode_ =
      dgsem::EntropyJacobianMode::FiniteDifference;
  double gamma_ = 1.4;
  mutable RusanovFluxWithSides rusanov_;
  mutable DiffusiveRusanovFlux diffusive_rusanov_;
};

} // namespace mfem
