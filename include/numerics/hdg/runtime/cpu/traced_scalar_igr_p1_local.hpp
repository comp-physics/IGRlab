#pragma once

/** Real affine-Q1 local mixed operator; intentionally has no mesh or MPI API. */

#include "numerics/hdg/core/host_reference.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace dg::hdg::runtime::cpu::traced_scalar_igr_p1
{
inline constexpr int volume_scalar_dofs = 4;
inline constexpr int face_scalar_dofs = 2;
inline constexpr int faces = 4;
inline constexpr int interior_components = 7;
inline constexpr int trace_components = 5;
inline constexpr int interior_unknowns = 28;
inline constexpr int trace_unknowns = 40;

enum class Status : unsigned char
{ success, invalid_parameter, invalid_geometry, invalid_layout, inadmissible_state, nonfinite_result };

struct AffineQuad
{
  // Counter-clockwise physical vertices for reference (0,0),(1,0),(1,1),(0,1).
  std::array<std::array<double, 2>, 4> vertices{};
};

struct Parameters
{
  double gamma = 1.4;
  double alpha = 1.0;
  double beta = 1.0;
  double dt = 1.0;
  double density_floor = 1.0e-12;
  int quadrature_order = 12;
};

struct StaticEulerRhsView
{
  // Volume quadrature point-major samples: [point][U component].
  const double* data = nullptr;
  std::size_t sample_count = 0;
};

struct ElementState
{
  // Component-major: U(4), Sigma, zx, zy; each component has four Q1 DOFs.
  const double* interior = nullptr;
  const double* previous_interior = nullptr;
  // Component-major, each trace component has four faces times two GL p1 DOFs.
  const double* trace = nullptr;
  const double* previous_trace = nullptr;
  StaticEulerRhsView static_euler_rhs{};
};

struct LocalOperator
{
  std::vector<double> interior_residual;
  std::vector<double> trace_residual;
  std::vector<double> A, B, C, D;
  // Owned sign-adapted storage for HostReferenceEngine: f=-r_i, g=-r_t.
  std::vector<double> f, g;
  void Resize();
};

Status Assemble(const AffineQuad&, const Parameters&, const ElementState&,
                LocalOperator&) noexcept;

/** Nonowning one-element HostReferenceEngine view over an assembled operator. */
core::LocalSystemBatchView HostReferenceInput(const LocalOperator&) noexcept;

} // namespace dg::hdg::runtime::cpu::traced_scalar_igr_p1
