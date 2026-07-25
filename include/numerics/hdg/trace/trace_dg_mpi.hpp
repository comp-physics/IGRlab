#pragma once

/**
 * @file numerics/hdg/trace/trace_dg_mpi.hpp
 * @brief MFEM-owned parallel maps for 2D vector DG-interface HDG traces.
 *
 * This interface is deliberately separate from the conforming scalar
 * H1/H1_Trace trace path.  It records the signed face VDofs supplied by
 * MFEM's DG_Interface_FECollection and uses MFEM true-DOF ownership and
 * ParGridFunction distribution for owner-to-ghost exchange.
 */

#include "numerics/hdg/trace/trace_mpi.hpp"

#include <mfem.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace dg
{
namespace hdg
{
namespace trace
{

/** MFEM's canonical signed VDofs for one local mesh face. */
struct DGInterfaceFaceMap
{
  int local_face = -1;
  mfem::Geometry::Type geometry = mfem::Geometry::INVALID;
  std::vector<int> signed_local_vdofs;
};

/** Parallel vector trace metadata for 2D DG_Interface_FECollection. */
struct DGInterfaceTraceMap
{
  TraceMap dofs;
  int vector_dimension = 0;
  mfem::Ordering::Type ordering = mfem::Ordering::byNODES;
  std::vector<DGInterfaceFaceMap> faces;

  bool IsConsistent() const noexcept;
  std::size_t OwnedCount() const noexcept { return dofs.OwnedCount(); }
  std::size_t GhostCount() const noexcept { return dofs.GhostCount(); }
};

/**
 * Build a DG-interface trace map without an MPI collective.
 *
 * Only a conforming two-dimensional ParFiniteElementSpace backed by
 * DG_Interface_FECollection segment faces, positive vdim, and
 * Ordering::byNODES is accepted.  Three-dimensional triangle and quadrilateral
 * faces are deliberately rejected until equivalent orientation evidence is
 * admitted.  A local true-DOF partition may be empty.  On failure, @a map is
 * preserved.
 */
bool BuildDGInterfaceTraceMap(const mfem::ParFiniteElementSpace& space,
                              DGInterfaceTraceMap& map);

/**
 * Exact-space exchange metadata.  Construction is noncollective after the
 * ParFiniteElementSpace exists; Distribute is collective and delegates the
 * communication to mfem::ParGridFunction.
 */
struct DGInterfaceTraceExchangePlan
{
  DGInterfaceTraceExchangePlan();
  ~DGInterfaceTraceExchangePlan();
  DGInterfaceTraceExchangePlan(DGInterfaceTraceExchangePlan&&) noexcept;
  DGInterfaceTraceExchangePlan& operator=(
      DGInterfaceTraceExchangePlan&&) noexcept;
  DGInterfaceTraceExchangePlan(
      const DGInterfaceTraceExchangePlan&) = delete;
  DGInterfaceTraceExchangePlan& operator=(
      const DGInterfaceTraceExchangePlan&) = delete;

  const mfem::ParFiniteElementSpace* space = nullptr;
  long space_sequence = -1;
  long mesh_sequence = -1;
  DGInterfaceTraceMap map;

  bool IsConsistent() const noexcept;

  /**
   * Distribute rank-owned true values into all local trace VDofs.
   *
   * This is a collective operation on the bound space communicator.  Every
   * rank must call it in the same order.  Rank-local validation is reduced by
   * an MFEM-owned collective before any rank enters prolongation.  The exact
   * bound space and its saved generations are required, and the bound space,
   * mesh, and finite-element collection must outlive this plan.  Input and
   * output storage may not overlap.  The output must already have local VSize;
   * exchange performs no candidate-owned allocation.  Any failure preserves
   * @a local_values on every participating rank.
   */
  bool Distribute(mfem::ParFiniteElementSpace& space,
                  const mfem::Vector& owned_true_values,
                  mfem::Vector& local_values) const;

private:
  struct Storage;
  std::unique_ptr<Storage> storage_;
  MPI_Comm communicator_{};
  int communicator_size_ = 0;
  mutable mfem::Vector consensus_value_;
  mutable mfem::Vector consensus_unit_;

  bool CollectiveAll(bool local_value) const;

  friend bool BuildDGInterfaceTraceExchangePlan(
      const mfem::ParFiniteElementSpace& space,
      DGInterfaceTraceExchangePlan& plan);
};

/** Build a noncollective, exact-space DG-interface exchange plan. */
bool BuildDGInterfaceTraceExchangePlan(
    const mfem::ParFiniteElementSpace& space,
    DGInterfaceTraceExchangePlan& plan);

} // namespace trace
} // namespace hdg
} // namespace dg
