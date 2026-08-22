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

#include <cmath>
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

#include <mfem.hpp>
#include <mpi.h>

#include "numerics/hdg/generic/contract.hpp"

namespace dg_sim::hdg
{

inline constexpr int hdg_euler_2d_components =
    dg::hdg::generic::euler_2d_conserved_components;

namespace detail
{

inline bool CollectiveTraceProjectionReady(
    const MPI_Comm communicator, const bool local) noexcept
{
  const int input = local ? 1 : 0;
  int output = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
          MPI_SUCCESS &&
      output == 1;
}

} // namespace detail

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

/** L2-project one four-component coefficient on an element-owned L2 space. */
inline bool ProjectEulerCoefficientL2(
    mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& state,
    mfem::VectorCoefficient& coefficient, const int quadrature_order) noexcept
{
  try
  {
    if (quadrature_order <= 0 || coefficient.GetVDim() != hdg_euler_2d_components ||
        space.GetVDim() != hdg_euler_2d_components ||
        space.GetOrdering() != mfem::Ordering::byNODES ||
        state.ParFESpace() != &space)
    {
      return false;
    }
    state = 0.0;
    for (int element = 0; element < space.GetNE(); ++element)
    {
      const mfem::FiniteElement* const finite_element = space.GetFE(element);
      mfem::ElementTransformation* const transformation =
          space.GetElementTransformation(element);
      if (!finite_element || !transformation) { return false; }
      const int scalar_dofs = finite_element->GetDof();
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      if (scalar_dofs <= 0 || rule.GetNPoints() <= 0) { return false; }
      mfem::DenseMatrix mass(scalar_dofs);
      mass = 0.0;
      std::array<mfem::Vector, hdg_euler_2d_components> rhs;
      for (auto& component : rhs) { component.SetSize(scalar_dofs); component = 0.0; }
      mfem::Vector shape(scalar_dofs);
      mfem::Vector value(hdg_euler_2d_components);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& point = rule.IntPoint(q);
        transformation->SetIntPoint(&point);
        finite_element->CalcShape(point, shape);
        coefficient.Eval(value, *transformation, point);
        const double weight = point.weight * transformation->Weight();
        if (!std::isfinite(weight)) { return false; }
        for (int row = 0; row < scalar_dofs; ++row)
        {
          for (int column = 0; column < scalar_dofs; ++column)
            mass(row, column) += weight * shape[row] * shape[column];
          for (int component = 0; component < hdg_euler_2d_components; ++component)
          {
            if (!std::isfinite(value[component])) { return false; }
            rhs[static_cast<std::size_t>(component)][row] +=
                weight * shape[row] * value[component];
          }
        }
      }
      mfem::DenseMatrixInverse inverse(mass);
      mfem::Array<int> vdofs;
      mfem::DofTransformation transformation_map(
          space.GetVDim(), space.GetOrdering());
      space.GetElementVDofs(element, vdofs, transformation_map);
      mfem::Vector local(vdofs.Size());
      if (local.Size() != scalar_dofs * hdg_euler_2d_components) { return false; }
      for (int component = 0; component < hdg_euler_2d_components; ++component)
      {
        mfem::Vector solution(scalar_dofs);
        inverse.Mult(rhs[static_cast<std::size_t>(component)], solution);
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
          local[component * scalar_dofs + scalar] = solution[scalar];
      }
      transformation_map.TransformPrimal(local);
      state.SetSubVector(vdofs, local);
    }
    return state.CheckFinite() == 0;
  }
  catch (...) { return false; }
}

namespace detail
{

inline bool ProjectEulerTrueTraceCoefficientL2Local(
    mfem::ParFiniteElementSpace& trace_space, mfem::ParGridFunction& local_trace,
    mfem::VectorCoefficient& coefficient, const int quadrature_order) noexcept
{
  try
  {
    if (quadrature_order <= 0 || coefficient.GetVDim() != hdg_euler_2d_components ||
        trace_space.GetVDim() != hdg_euler_2d_components ||
        trace_space.GetOrdering() != mfem::Ordering::byNODES ||
        local_trace.ParFESpace() != &trace_space || trace_space.GetParMesh() == nullptr)
    {
      return false;
    }
    local_trace = 0.0;
    for (int face = 0; face < trace_space.GetParMesh()->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* const finite_element =
          trace_space.GetFaceElement(face);
      mfem::ElementTransformation* const transformation =
          trace_space.GetParMesh()->GetFaceTransformation(face);
      mfem::Array<int> vdofs;
      trace_space.GetFaceVDofs(face, vdofs);
      if (!finite_element || !transformation || finite_element->GetDof() <= 0 ||
          vdofs.Size() != finite_element->GetDof() * hdg_euler_2d_components)
      {
        return false;
      }
      const int scalar_dofs = finite_element->GetDof();
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      if (rule.GetNPoints() <= 0) { return false; }
      mfem::DenseMatrix mass(scalar_dofs);
      mass = 0.0;
      std::array<mfem::Vector, hdg_euler_2d_components> rhs;
      for (auto& component : rhs) { component.SetSize(scalar_dofs); component = 0.0; }
      mfem::Vector shape(scalar_dofs);
      mfem::Vector value(hdg_euler_2d_components);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& point = rule.IntPoint(q);
        transformation->SetIntPoint(&point);
        finite_element->CalcShape(point, shape);
        coefficient.Eval(value, *transformation, point);
        const double weight = point.weight * transformation->Weight();
        if (!std::isfinite(weight)) { return false; }
        for (int row = 0; row < scalar_dofs; ++row)
        {
          for (int column = 0; column < scalar_dofs; ++column)
            mass(row, column) += weight * shape[row] * shape[column];
          for (int component = 0; component < hdg_euler_2d_components; ++component)
          {
            if (!std::isfinite(value[component])) { return false; }
            rhs[static_cast<std::size_t>(component)][row] +=
                weight * shape[row] * value[component];
          }
        }
      }
      mfem::DenseMatrixInverse inverse(mass);
      for (int component = 0; component < hdg_euler_2d_components; ++component)
      {
        mfem::Vector solution(scalar_dofs);
        inverse.Mult(rhs[static_cast<std::size_t>(component)], solution);
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
        {
          const int signed_dof = vdofs[scalar + scalar_dofs * component];
          const int dof = signed_dof >= 0 ? signed_dof : -1 - signed_dof;
          if (dof < 0 || dof >= local_trace.Size()) { return false; }
          local_trace[dof] = (signed_dof >= 0 ? 1.0 : -1.0) * solution[scalar];
        }
      }
    }
    return local_trace.CheckFinite() == 0;
  }
  catch (...) { return false; }
}

} // namespace detail

/**
 * Independently L2-project an analytic Euler state onto local trace faces and
 * publish the resulting owned true-trace coordinates.  This intentionally
 * never samples the already-projected volume field.  Every rank collectively
 * admits both local projection and true-vector storage before entering the
 * collective true-trace projection.
 */
inline bool ProjectEulerTrueTraceCoefficientL2(
    mfem::ParFiniteElementSpace& trace_space, mfem::ParGridFunction& local_trace,
    mfem::Vector& true_trace, mfem::VectorCoefficient& coefficient,
    const int quadrature_order) noexcept
{
  const MPI_Comm communicator = trace_space.GetComm();
  bool local_ready = detail::ProjectEulerTrueTraceCoefficientL2Local(
      trace_space, local_trace, coefficient, quadrature_order);
  if (local_ready)
  {
    try
    {
      true_trace.SetSize(trace_space.GetTrueVSize());
      local_ready = true_trace.Size() == trace_space.GetTrueVSize();
    }
    catch (...) { local_ready = false; }
  }
  if (!detail::CollectiveTraceProjectionReady(communicator, local_ready))
  {
    return false;
  }

  bool output_ready = false;
  try
  {
    local_trace.ParallelProject(true_trace);
    output_ready = true_trace.Size() == trace_space.GetTrueVSize();
    for (int index = 0; output_ready && index < true_trace.Size(); ++index)
      output_ready = std::isfinite(true_trace[index]);
  }
  catch (...) { output_ready = false; }
  return detail::CollectiveTraceProjectionReady(communicator, output_ready);
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
