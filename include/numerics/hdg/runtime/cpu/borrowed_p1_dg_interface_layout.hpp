#pragma once

/**
 * @file numerics/hdg/runtime/cpu/borrowed_p1_dg_interface_layout.hpp
 * @brief Borrowed-mesh p1 DG-interface trace and L2 interior layouts.
 *
 * This small CPU-only adapter owns the finite-element collections, spaces, and
 * the exact-space DG-interface exchange plan.  It never owns or changes the
 * supplied ParMesh, which must outlive the adapter and remain unchanged while
 * it is used.  Local element vectors are component-major:
 *
 * - trace: component * 8 + face * 2 + dof;
 * - interior: component * 4 + dof.
 */

#include <mfem.hpp>

#include <memory>

namespace dg::hdg::runtime::cpu
{

class BorrowedP1DGInterfaceLayout
{
public:
  static constexpr int dimension = 2;
  static constexpr int polynomial_order = 1;
  static constexpr int trace_components = 5;
  static constexpr int interior_components = 7;
  static constexpr int faces_per_quad = 4;
  static constexpr int trace_dofs_per_face = 2;
  static constexpr int interior_dofs_per_quad = 4;
  static constexpr int trace_dofs_per_element =
      faces_per_quad * trace_dofs_per_face;
  static constexpr int trace_values_per_element =
      trace_components * trace_dofs_per_element;
  static constexpr int interior_values_per_element =
      interior_components * interior_dofs_per_quad;

  /**
   * Construct a layout over @a mesh.  Unsupported or inconsistent meshes
   * produce an unusable object; callers can inspect this with IsUsable().
   */
  explicit BorrowedP1DGInterfaceLayout(mfem::ParMesh& mesh) noexcept;
  ~BorrowedP1DGInterfaceLayout();

  BorrowedP1DGInterfaceLayout(const BorrowedP1DGInterfaceLayout&) = delete;
  BorrowedP1DGInterfaceLayout& operator=(
      const BorrowedP1DGInterfaceLayout&) = delete;
  BorrowedP1DGInterfaceLayout(BorrowedP1DGInterfaceLayout&&) noexcept;
  BorrowedP1DGInterfaceLayout& operator=(
      BorrowedP1DGInterfaceLayout&&) noexcept;

  /** True only while the borrowed mesh and all owned layout objects agree. */
  bool IsUsable() const noexcept;

  const mfem::ParFiniteElementSpace* TraceSpace() const noexcept;
  const mfem::ParFiniteElementSpace* InteriorSpace() const noexcept;

  /** The number of rank-owned trace true values required by DistributeTrace. */
  int TraceTrueVSize() const noexcept;
  int TraceLocalVSize() const noexcept;
  int InteriorLocalVSize() const noexcept;
  int LocalElementCount() const noexcept;

  /**
   * Collectively distribute this rank's trace true values to local VDofs.
   * The existing DGInterfaceTraceExchangePlan performs the communication.
   */
  bool DistributeTrace(const mfem::Vector& owned_true_values,
                       mfem::Vector& local_trace_values) const;

  /**
   * Copy one element's trace from local VDOF storage into the exact signed
   * component * 8 + face * 2 + dof layout.  Both vectors must have their
   * exact expected sizes; failure preserves @a element_values.
   */
  bool GatherTraceElement(int element, const mfem::Vector& local_trace_values,
                          mfem::Vector& element_values) const;

  /**
   * Copy one element's interior from local VDOF storage into component * 4 +
   * dof layout.  Both vectors must have their exact expected sizes.
   */
  bool GatherInteriorElement(int element,
                             const mfem::Vector& local_interior_values,
                             mfem::Vector& element_values) const;

  /**
   * Copy component * 4 + dof element values into the signed local interior
   * VDOFs.  Both vectors must have their exact expected sizes.
   */
  bool ScatterInteriorElement(int element, const mfem::Vector& element_values,
                              mfem::Vector& local_interior_values) const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace dg::hdg::runtime::cpu
