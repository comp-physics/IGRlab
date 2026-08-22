#include "numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.hpp"

#include "numerics/hdg/runtime/common/traced_scalar_igr_p1_local_algebra.hpp"

#include <mfem.hpp>

#include <array>

namespace dg::hdg::runtime::cpu::traced_scalar_igr_p1
{
namespace
{
namespace algebra = common::traced_scalar_igr_p1_local_algebra;

// The admitted MFEM rules through order 32 are far below these checked fixed
// capacities. Keeping the packing on the stack preserves hot-attempt allocation
// stability while the shared device routine receives only raw pointers.
constexpr int max_volume_quadrature_points = 1024;
constexpr int max_segment_quadrature_points = 64;

Status ToStatus(const algebra::Status status) noexcept
{
  switch (status)
  {
    case algebra::Status::success: return Status::success;
    case algebra::Status::invalid_parameter: return Status::invalid_parameter;
    case algebra::Status::invalid_geometry: return Status::invalid_geometry;
    case algebra::Status::invalid_layout: return Status::invalid_layout;
    case algebra::Status::inadmissible_state: return Status::inadmissible_state;
    case algebra::Status::nonfinite_result: return Status::nonfinite_result;
  }
  return Status::nonfinite_result;
}

bool Finite(const double value) noexcept { return algebra::Finite(value); }
} // namespace

void LocalOperator::Resize()
{
  interior_residual.assign(interior_unknowns, 0.);
  trace_residual.assign(trace_unknowns, 0.);
  A.assign(interior_unknowns * interior_unknowns, 0.);
  B.assign(interior_unknowns * trace_unknowns, 0.);
  C.assign(trace_unknowns * interior_unknowns, 0.);
  D.assign(trace_unknowns * trace_unknowns, 0.);
  f.assign(interior_unknowns, 0.);
  g.assign(trace_unknowns, 0.);
}

Status Assemble(const AffineQuad& quad, const Parameters& parameters,
                const ElementState& state, LocalOperator& output) noexcept
{
  if (!state.interior || !state.previous_interior || !state.trace || !state.previous_trace ||
      !Finite(parameters.gamma) || !(parameters.gamma > 1.) ||
      !Finite(parameters.alpha) || !(parameters.alpha > 0.) ||
      !Finite(parameters.beta) || !(parameters.beta > 0.) ||
      !Finite(parameters.dt) || !(parameters.dt > 0.) ||
      !Finite(parameters.density_floor) || !(parameters.density_floor > 0.) ||
      parameters.quadrature_order < 2 || parameters.quadrature_order > 32)
    return Status::invalid_parameter;
  double geometry_jacobian[4] = {}, determinant = 0.;
  if (!algebra::Geometry(quad.vertices[0].data(), geometry_jacobian, determinant))
    return Status::invalid_geometry;

  const mfem::IntegrationRule* volume_rule_ptr = nullptr;
  const mfem::IntegrationRule* face_rule_ptr = nullptr;
  try
  {
    volume_rule_ptr = &mfem::IntRules.Get(mfem::Geometry::SQUARE,
                                           parameters.quadrature_order);
    face_rule_ptr = &mfem::IntRules.Get(mfem::Geometry::SEGMENT,
                                         parameters.quadrature_order);
  }
  catch (...)
  {
    return Status::invalid_parameter;
  }
  const mfem::IntegrationRule& volume_rule = *volume_rule_ptr;
  const mfem::IntegrationRule& face_rule = *face_rule_ptr;
  if (volume_rule.GetNPoints() <= 0 || face_rule.GetNPoints() <= 0 ||
      volume_rule.GetNPoints() > max_volume_quadrature_points ||
      face_rule.GetNPoints() > max_segment_quadrature_points) return Status::invalid_parameter;
  const std::size_t volume_points = static_cast<std::size_t>(volume_rule.GetNPoints());
  if ((state.static_euler_rhs.data && state.static_euler_rhs.sample_count != volume_points) ||
      (!state.static_euler_rhs.data && state.static_euler_rhs.sample_count != 0))
    return Status::invalid_layout;

  std::array<double, 3 * max_volume_quadrature_points> volume_xyw{};
  std::array<double, 2 * max_segment_quadrature_points> segment_xw{};
  for (int point = 0; point < volume_rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
    volume_xyw[3 * point] = ip.x;
    volume_xyw[3 * point + 1] = ip.y;
    volume_xyw[3 * point + 2] = ip.weight;
  }
  for (int point = 0; point < face_rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
    segment_xw[2 * point] = ip.x;
    segment_xw[2 * point + 1] = ip.weight;
  }
  try { output.Resize(); }
  catch (...) { return Status::invalid_layout; }

  const algebra::Input input{
    quad.vertices[0].data(),
    {parameters.gamma, parameters.alpha, parameters.beta, parameters.dt,
     parameters.density_floor},
    state.interior, state.previous_interior, state.trace, state.previous_trace,
    state.static_euler_rhs.data, static_cast<int>(state.static_euler_rhs.sample_count),
    {volume_xyw.data(), volume_rule.GetNPoints(), segment_xw.data(), face_rule.GetNPoints()}};
  algebra::Output raw_output{output.interior_residual.data(), output.trace_residual.data(),
                             output.A.data(), output.B.data(), output.C.data(), output.D.data(),
                             output.f.data(), output.g.data()};
  return ToStatus(algebra::Assemble(input, raw_output));
}

core::LocalSystemBatchView HostReferenceInput(const LocalOperator& output) noexcept
{
  const auto view = [](const double* data, const int rows, const int columns)
  { return StridedBlockView<const double>{data, 1, rows, columns, rows * columns, columns, 1}; };
  return {{4, 8, 7, 5},
          view(output.A.data(), 28, 28), view(output.B.data(), 28, 40),
          view(output.C.data(), 40, 28), view(output.D.data(), 40, 40),
          view(output.f.data(), 28, 1), view(output.g.data(), 40, 1)};
}
} // namespace dg::hdg::runtime::cpu::traced_scalar_igr_p1
