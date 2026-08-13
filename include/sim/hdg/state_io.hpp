#pragma once

/**
 * @file sim/hdg/state_io.hpp
 * @brief Caller-owned MFEM initial-state projection and publication seam.
 *
 * This V1 seam accepts one two-dimensional, four-component volume
 * mfem::ParFiniteElementSpace/mfem::ParGridFunction association on a
 * caller-owned mfem::ParMesh.  It borrows a VectorCoefficient for one
 * projection, verifies that the resulting host coordinates are finite, and
 * then publishes those coordinates as a read-only snapshot view.
 *
 * The seam owns no mesh, finite-element space, grid function, coefficient,
 * allocator, lifecycle, restart, trace, or derived Euler field.  In
 * particular, it intentionally exposes neither the mutable grid function nor
 * an evidence or ABI record.  Callers retain the supplied MFEM objects and
 * must keep them alive and unchanged while using a published view: publication
 * is intentionally a zero-copy snapshot of the committed IC coordinates.
 */

#include <cstddef>
#include <type_traits>

#include <mfem.hpp>

#include "numerics/hdg/generic/contract.hpp"

namespace dg_sim::hdg
{

inline constexpr int hdg_euler_2d_components =
    dg::hdg::generic::euler_2d_conserved_components;

/**
 * Return whether @p state is exactly the four-component, byNODES volume field
 * on @p space and @p mesh admitted by this seam.  This predicate is public so
 * callers and source-level tests can check an association before constructing
 * the fail-closed adapter.  byNODES is required because the imported raw view
 * is consumed as component-major coordinates.
 */
inline bool IsHdgVolumeStateAssociation(
    const mfem::ParMesh& mesh,
    const mfem::ParFiniteElementSpace& space,
    const mfem::ParGridFunction& state) noexcept
{
  return mesh.Dimension() == 2 && space.GetParMesh() == &mesh &&
      space.GetVDim() == hdg_euler_2d_components &&
      space.GetOrdering() == mfem::Ordering::byNODES &&
      state.ParFESpace() == &space;
}

/**
 * Borrowed initial-condition projection and committed-snapshot adapter.
 *
 * @ref ProjectBorrowedInitialConditionOnce invokes MFEM's vector-coefficient
 * projection exactly once.  On return, @ref published_coordinates exposes the
 * committed finite coordinates only.  A second projection is rejected so a
 * published snapshot has one IC origin.  The caller must not mutate its
 * published grid function while consumers retain this zero-copy view.
 */
class HdgStateIo
{
public:
  HdgStateIo(mfem::ParMesh& mesh,
             mfem::ParFiniteElementSpace& volume_space,
             mfem::ParGridFunction& published_state)
    : volume_space_(volume_space), published_state_(published_state)
  {
    MFEM_VERIFY(IsHdgVolumeStateAssociation(mesh, volume_space, published_state),
                "HDG state I/O requires one 2-D four-component volume "
                "ParFiniteElementSpace/ParGridFunction association.");
  }

  HdgStateIo(const HdgStateIo&) = delete;
  HdgStateIo& operator=(const HdgStateIo&) = delete;
  HdgStateIo(HdgStateIo&&) = delete;
  HdgStateIo& operator=(HdgStateIo&&) = delete;

  /**
   * Project a caller-owned coefficient once and publish its finite result.
   *
   * The coefficient is borrowed for this call only.  Its vector dimension
   * must be the four conserved 2-D Euler components.  This performs no
   * allocation or mesh/space construction.
   */
  void ProjectBorrowedInitialConditionOnce(mfem::VectorCoefficient& initial_condition)
  {
    MFEM_VERIFY(!projection_attempted_,
                "HDG state I/O permits exactly one initial-condition projection.");
    MFEM_VERIFY(initial_condition.GetVDim() == hdg_euler_2d_components,
                "HDG state I/O requires a four-component initial condition.");

    projection_attempted_ = true;
    published_state_.ProjectCoefficient(initial_condition);
    MFEM_VERIFY(published_state_.CheckFinite() == 0,
                "HDG state I/O refuses non-finite published coordinates.");
    published_ = true;
  }

  /** Return only the finite coordinates of the committed published state. */
  [[nodiscard]] const double* published_coordinates() const
  {
    RequirePublishedSnapshot();
    return published_state_.HostRead();
  }

  [[nodiscard]] std::size_t published_scalar_count() const
  {
    RequirePublishedSnapshot();
    return static_cast<std::size_t>(published_state_.Size());
  }

  /**
   * Convert the committed snapshot into the existing generic HDG input view.
   * The view is borrowed and preserves MFEM's volume-space ordering: it maps
   * directly to EulerHdgRequest::initial_state without creating a new ABI,
   * evidence, restart, or ownership record.
   */
  [[nodiscard]] dg::hdg::generic::EulerFieldView
  ImportPublishedInitialState() const
  {
    return {published_coordinates(), published_scalar_count(),
            hdg_euler_2d_components};
  }

private:
  void RequirePublishedSnapshot() const
  {
    MFEM_VERIFY(published_,
                "HDG state I/O coordinates are available only after publication.");
  }

  mfem::ParFiniteElementSpace& volume_space_;
  mfem::ParGridFunction& published_state_;
  // Narrow one-call projection/publication guards, not solver lifecycle state.
  bool projection_attempted_ = false;
  bool published_ = false;
};

static_assert(!std::is_polymorphic<HdgStateIo>::value,
              "HDG state I/O must remain nonvirtual.");

} // namespace dg_sim::hdg
