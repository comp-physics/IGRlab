#pragma once

/**
 * @file numerics/dgsem/dgsem_operator.hpp
 * @brief DGSEM operator helpers (volume + surface assembly).
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"
#include "numerics/dgsem/dgsem_production.hpp"

#include <memory>
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"

namespace mfem
{
class DGHyperbolicConservationLaws;

namespace dgsem
{
struct DGSEMProductionCache;

/// DGSEM Wave 1 has complete tensor volume/surface semantics in 1D and 2D.
inline constexpr bool DGSEMOperatorSupportsDimension(int dim)
{
  return dim == 1 || dim == 2;
}

/// @brief Helper to assemble DGSEM volume/surface contributions.
class DGSEMOperator
{
public:
  explicit DGSEMOperator(const DGHyperbolicConservationLaws& op);
  ~DGSEMOperator();

  DGSEMOperator(const DGSEMOperator&) = delete;
  DGSEMOperator& operator=(const DGSEMOperator&) = delete;

  /** Rebind a moved persistent bridge to its new owning operator.
   *
   * A changed owner discards all cache state so no pointer, topology,
   * communicator, or policy fingerprint from the moved-from object survives.
   * Returns true exactly when a rebind occurred.
   */
  bool Rebind(const DGHyperbolicConservationLaws& op);

  /** Assemble DGSEM surface contributions into z (overwrites z).
   *
   * When non-null, `physical_boundary_flux` is overwritten with the
   * rank-local integral of the outward numerical flux actually used by this
   * surface assembly. Communicator reduction remains the parent operator's
   * responsibility.
   */
  real_t AssembleSurface(const Vector& x,
                         Vector& z,
                         Vector* physical_boundary_flux = nullptr) const;

  /// @brief Add DGSEM volume contributions into z (accumulates into z).
  real_t AssembleVolume(const Vector& x, Vector& z) const;

  /// @brief Apply inverse mass matrices element-wise.
  void ApplyInverseMass(const Vector& z, Vector& y) const;

  /// @brief Assemble surface + volume and apply M^{-1}, returns max wave speed.
  real_t AssembleResidual(const Vector& x, Vector& y) const;

  /** Bind and, when supported, build the persistent production policy/cache.
   *
   * This operation does not mutate a residual output. In strict mode every
   * unsupported tuple is returned with a stable reason code for the caller to
   * fail closed. Non-strict callers may use the host oracle only after logging
   * that same reason explicitly.
   */
  DGSEMProductionSelection BindProductionPolicy(bool strict) const;

  /** Execute the accepted production kernels on an MFEM local L-vector.
   *
   * State/sigma admissibility is checked before `y_local` is written. The
   * caller is responsible for true-to-local `P x` before this call and local-
   * to-true `P^T y` afterward. When requested, `physical_boundary_flux` is
   * overwritten with the rank-local outward flux used by the packed face
   * residual; communicator reduction remains the caller's responsibility.
   *
   * With `admission_only=true`, the same packed gather, exchange, and FP64
   * owned/face scans run, but no flux, inverse mass, speed reduction,
   * boundary-flux capture, or caller-output write occurs. The result then
   * carries the fresh communicator-wide q_sigma aggregate for this call.
   * The default remains the complete residual operation.
   */
  DGSEMProductionResult AssembleProductionResidual(
    const Vector& x_local,
    Vector& y_local,
    bool strict,
    Vector* physical_boundary_flux = nullptr,
    bool admission_only = false) const;

  /// Invalidate only the named cache layer after an owning lifecycle change.
  void InvalidateProductionCache(DGSEMProductionInvalidation invalidation) const;

  DGSEMProductionCacheCounters GetProductionCacheCounters() const;
  DGSEMProductionSelection GetProductionSelection() const;
  DGSEMProductionWorkspaceEvidence GetProductionWorkspaceEvidence() const;

private:
  const BdrHyperbolicFluxIntegrator* FindBdrIntegrator(int attr) const;

  DGSEMProductionReason BuildProductionCacheLocal_(bool strict,
                                                    int ranks,
                                                    std::uint64_t fingerprint) const;
  DGSEMProductionReason EnsureProductionCache_(bool strict) const;

  const DGHyperbolicConservationLaws* op_ = nullptr;
  mutable std::unique_ptr<DGSEMProductionCache> production_;
};

} // namespace dgsem
} // namespace mfem
