#pragma once

/** Fixed-size host/device algebra for one affine-Q1 traced scalar-IGR element. */

#include "physics/igr/frozen_scalar_igr_algebra.hpp"

#include <cmath>

namespace dg::hdg::runtime::common::traced_scalar_igr_p1_local_algebra
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

struct Parameters final
{
  double gamma;
  double alpha;
  double beta;
  double dt;
  double density_floor;
};

/** Host-prepared raw quadrature: [point][x,y,weight] and [point][x,weight]. */
struct QuadratureTables final
{
  const double* volume_xyw;
  int volume_points;
  const double* segment_xw;
  int segment_points;
};

struct Input final
{
  // (x,y) vertices in counter-clockwise Q1 order.
  const double* vertices;
  Parameters parameters;
  const double* interior;
  const double* previous_interior;
  const double* trace;
  const double* previous_trace;
  // Optional point-major [volume point][Euler component] source.
  const double* static_euler_rhs;
  int static_euler_rhs_points;
  QuadratureTables quadrature;
};

struct Output final
{
  double* interior_residual;
  double* trace_residual;
  double* A;
  double* B;
  double* C;
  double* D;
  double* f;
  double* g;
};

MFEM_HOST_DEVICE inline bool Finite(const double value) noexcept
{ return ::isfinite(value); }

/** Hypot-equivalent 2-norm without relying on a device hypot overload. */
MFEM_HOST_DEVICE inline double ScaledNorm2(const double x, const double y) noexcept
{
  if (!Finite(x)) return x;
  if (!Finite(y)) return y;
  const double ax = ::fabs(x);
  const double ay = ::fabs(y);
  const double scale = ax > ay ? ax : ay;
  if (!(scale > 0.)) return scale;
  const double sx = x / scale;
  const double sy = y / scale;
  return scale * ::sqrt(sx * sx + sy * sy);
}

MFEM_HOST_DEVICE inline int I(const int component, const int scalar) noexcept
{ return component * volume_scalar_dofs + scalar; }

MFEM_HOST_DEVICE inline int T(const int component, const int face_scalar) noexcept
{ return component * (faces * face_scalar_dofs) + face_scalar; }

MFEM_HOST_DEVICE inline double& M(double* matrix, const int columns,
                                  const int row, const int column) noexcept
{ return matrix[row * columns + column]; }

MFEM_HOST_DEVICE inline void Shapes(const double xi, const double eta,
                                    double* n, double* dr) noexcept
{
  constexpr double a = 0.21132486540518711775;
  constexpr double b = 0.78867513459481288225;
  constexpr double scale = 1.0 / (b - a);
  const double lx[2] = {(b - xi) * scale, (xi - a) * scale};
  const double ly[2] = {(b - eta) * scale, (eta - a) * scale};
  const double dl[2] = {-scale, scale};
  for (int y = 0; y < 2; ++y)
  {
    for (int x = 0; x < 2; ++x)
    {
      const int q = x + 2 * y;
      n[q] = lx[x] * ly[y];
      dr[2 * q] = dl[x] * ly[y];
      dr[2 * q + 1] = lx[x] * dl[y];
    }
  }
}

MFEM_HOST_DEVICE inline void FacePoint(const int face, const double s,
                                       double& xi, double& eta,
                                       double* tangent) noexcept
{
  switch (face)
  {
    case 0: xi = s; eta = 0.; tangent[0] = 1.; tangent[1] = 0.; break;
    case 1: xi = 1.; eta = s; tangent[0] = 0.; tangent[1] = 1.; break;
    case 2: xi = 1. - s; eta = 1.; tangent[0] = -1.; tangent[1] = 0.; break;
    default: xi = 0.; eta = 1. - s; tangent[0] = 0.; tangent[1] = -1.; break;
  }
}

MFEM_HOST_DEVICE inline bool Admissible(const double* u,
                                        const Parameters& parameters) noexcept
{
  if (!Finite(u[0]) || !(u[0] > parameters.density_floor)) return false;
  const double pressure = (parameters.gamma - 1.) *
    (u[3] - .5 * (u[1] * u[1] + u[2] * u[2]) / u[0]);
  return Finite(pressure) && pressure > 0.;
}

MFEM_HOST_DEVICE inline bool EulerSigma(const double* u, const double sigma,
                                        const double* normal,
                                        const Parameters& parameters,
                                        double* flux, double* jacobian) noexcept
{
  if (!Admissible(u, parameters) || !Finite(sigma)) return false;
  const double rho = u[0], mx = u[1], my = u[2], energy = u[3];
  const double gm1 = parameters.gamma - 1.;
  const double pressure = gm1 * (energy - .5 * (mx * mx + my * my) / rho);
  const double un = (mx * normal[0] + my * normal[1]) / rho;
  flux[0] = mx * normal[0] + my * normal[1];
  flux[1] = mx * un + pressure * normal[0];
  flux[2] = my * un + pressure * normal[1];
  flux[3] = (energy + pressure) * un;
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 5; ++column) jacobian[row * 5 + column] = 0.;
  const double dp[4] = {.5 * gm1 * (mx * mx + my * my) / (rho * rho),
                        -gm1 * mx / rho, -gm1 * my / rho, gm1};
  const double dun[4] = {-un / rho, normal[0] / rho, normal[1] / rho, 0.};
  jacobian[0 * 5 + 1] = normal[0];
  jacobian[0 * 5 + 2] = normal[1];
  for (int column = 0; column < 4; ++column)
  {
    jacobian[1 * 5 + column] = (column == 1 ? un : 0.) + mx * dun[column] +
      dp[column] * normal[0];
    jacobian[2 * 5 + column] = (column == 2 ? un : 0.) + my * dun[column] +
      dp[column] * normal[1];
    jacobian[3 * 5 + column] = ((column == 3 ? 1. : 0.) + dp[column]) * un +
      (energy + pressure) * dun[column];
  }
  ::dg_igr::frozen_scalar_igr_algebra::AddNormalSigmaFlux(
    2, u, 1, normal, 1, sigma, parameters.density_floor, flux, 1);
  ::dg_igr::frozen_scalar_igr_algebra::AddNormalSigmaFluxJacobian(
    2, u, 1, normal, 1, sigma, parameters.density_floor, jacobian, 5, 1);
  // Sigma is a mixed unknown, while the reused frozen helper differentiates U only.
  jacobian[1 * 5 + 4] = normal[0];
  jacobian[2 * 5 + 4] = normal[1];
  jacobian[3 * 5 + 4] = un;
  return Finite(flux[0]) && Finite(flux[1]) && Finite(flux[2]) && Finite(flux[3]);
}

MFEM_HOST_DEVICE inline void Value(const double* x, const double* n,
                                   double* value) noexcept
{
  for (int component = 0; component < 4; ++component)
  {
    value[component] = 0.;
    for (int q = 0; q < 4; ++q) value[component] += x[I(component, q)] * n[q];
  }
}

MFEM_HOST_DEVICE inline double SValue(const double* x, const int component,
                                      const double* n) noexcept
{
  double value = 0.;
  for (int q = 0; q < 4; ++q) value += x[I(component, q)] * n[q];
  return value;
}

MFEM_HOST_DEVICE inline double Derivative(const double* x, const int component,
                                          const double* d, const int direction) noexcept
{
  double value = 0.;
  for (int q = 0; q < 4; ++q) value += x[I(component, q)] * d[2 * q + direction];
  return value;
}

MFEM_HOST_DEVICE inline bool Geometry(const double* vertices, double* j,
                                      double& determinant) noexcept
{
  for (int vertex = 0; vertex < 4; ++vertex)
    if (!Finite(vertices[2 * vertex]) || !Finite(vertices[2 * vertex + 1])) return false;
  j[0] = vertices[2] - vertices[0];
  j[1] = vertices[3] - vertices[1];
  j[2] = vertices[6] - vertices[0];
  j[3] = vertices[7] - vertices[1];
  const double expected_x = vertices[2] + j[2];
  const double expected_y = vertices[3] + j[3];
  if (!Finite(j[0]) || !Finite(j[1]) || !Finite(j[2]) || !Finite(j[3]) ||
      ::fabs(expected_x - vertices[4]) > 1.e-12 ||
      ::fabs(expected_y - vertices[5]) > 1.e-12) return false;
  determinant = j[0] * j[3] - j[1] * j[2];
  return Finite(determinant) && determinant > 0.;
}

MFEM_HOST_DEVICE inline bool AllFinite(const Output& output) noexcept
{
  const double* arrays[] = {output.interior_residual, output.trace_residual,
                            output.A, output.B, output.C, output.D};
  const int counts[] = {interior_unknowns, trace_unknowns,
                        interior_unknowns * interior_unknowns,
                        interior_unknowns * trace_unknowns,
                        trace_unknowns * interior_unknowns,
                        trace_unknowns * trace_unknowns};
  for (int array = 0; array < 6; ++array)
    for (int index = 0; index < counts[array]; ++index)
      if (!Finite(arrays[array][index])) return false;
  return true;
}

MFEM_HOST_DEVICE inline void Zero(Output& output) noexcept
{
  for (int i = 0; i < interior_unknowns; ++i)
  { output.interior_residual[i] = 0.; output.f[i] = 0.; }
  for (int i = 0; i < trace_unknowns; ++i)
  { output.trace_residual[i] = 0.; output.g[i] = 0.; }
  for (int i = 0; i < interior_unknowns * interior_unknowns; ++i) output.A[i] = 0.;
  for (int i = 0; i < interior_unknowns * trace_unknowns; ++i) output.B[i] = 0.;
  for (int i = 0; i < trace_unknowns * interior_unknowns; ++i) output.C[i] = 0.;
  for (int i = 0; i < trace_unknowns * trace_unknowns; ++i) output.D[i] = 0.;
}

MFEM_HOST_DEVICE inline Status Assemble(const Input& input, Output& output) noexcept
{
  const Parameters& p = input.parameters;
  if (!input.vertices || !input.interior || !input.previous_interior || !input.trace ||
      !input.previous_trace || !output.interior_residual || !output.trace_residual ||
      !output.A || !output.B || !output.C || !output.D || !output.f || !output.g ||
      !Finite(p.gamma) || !(p.gamma > 1.) || !Finite(p.alpha) || !(p.alpha > 0.) ||
      !Finite(p.beta) || !(p.beta > 0.) || !Finite(p.dt) || !(p.dt > 0.) ||
      !Finite(p.density_floor) || !(p.density_floor > 0.) ||
      !input.quadrature.volume_xyw || input.quadrature.volume_points <= 0 ||
      !input.quadrature.segment_xw || input.quadrature.segment_points <= 0) return Status::invalid_parameter;
  if ((input.static_euler_rhs && input.static_euler_rhs_points != input.quadrature.volume_points) ||
      (!input.static_euler_rhs && input.static_euler_rhs_points != 0)) return Status::invalid_layout;
  double j[4] = {}, determinant = 0.;
  if (!Geometry(input.vertices, j, determinant)) return Status::invalid_geometry;
  Zero(output);
  const double inverse[4] = {j[3] / determinant, -j[2] / determinant,
                             -j[1] / determinant, j[0] / determinant};
  for (int point = 0; point < input.quadrature.volume_points; ++point)
  {
    const double* quadrature = input.quadrature.volume_xyw + 3 * point;
    double n[4] = {}, dr[8] = {}, d[8] = {};
    Shapes(quadrature[0], quadrature[1], n, dr);
    for (int a = 0; a < 4; ++a)
      for (int k = 0; k < 2; ++k)
        d[2 * a + k] = dr[2 * a] * inverse[k] + dr[2 * a + 1] * inverse[2 + k];
    double u[4] = {}, old[4] = {};
    Value(input.interior, n, u); Value(input.previous_interior, n, old);
    if (!Admissible(u, p) || !Admissible(old, p)) return Status::inadmissible_state;
    const double sigma = SValue(input.interior, 4, n);
    const double z[2] = {SValue(input.interior, 5, n), SValue(input.interior, 6, n)};
    double fx[4] = {}, fy[4] = {}, jx[20] = {}, jy[20] = {};
    const double nx[2] = {1., 0.}, ny[2] = {0., 1.};
    if (!EulerSigma(u, sigma, nx, p, fx, jx) || !EulerSigma(u, sigma, ny, p, fy, jy))
      return Status::inadmissible_state;
    double vg[4] = {};
    for (int velocity = 0; velocity < 2; ++velocity)
      for (int direction = 0; direction < 2; ++direction)
      {
        const double density_derivative = Derivative(input.interior, 0, d, direction);
        const double momentum_derivative = Derivative(input.interior, 1 + velocity, d, direction);
        vg[2 * velocity + direction] = momentum_derivative / u[0] -
          (velocity == 0 ? u[1] : u[2]) * density_derivative / (u[0] * u[0]);
      }
    const double trace = vg[0] + vg[3];
    const double q = vg[0] * vg[0] + 2. * vg[1] * vg[2] + vg[3] * vg[3] + trace * trace;
    double dqg[4] = {};
    for (int velocity = 0; velocity < 2; ++velocity)
      for (int direction = 0; direction < 2; ++direction)
        dqg[2 * velocity + direction] = 2. * vg[2 * direction + velocity] +
          (velocity == direction ? 2. * trace : 0.);
    const double weight = quadrature[2] * determinant;
    for (int row_scalar = 0; row_scalar < 4; ++row_scalar)
    {
      for (int equation = 0; equation < 4; ++equation)
      {
        const int row = I(equation, row_scalar);
        const double rhs = input.static_euler_rhs ? input.static_euler_rhs[4 * point + equation] : 0.;
        if (!Finite(rhs)) return Status::invalid_parameter;
        output.interior_residual[row] += weight *
          (n[row_scalar] * (u[equation] - old[equation]) / p.dt -
           d[2 * row_scalar] * fx[equation] - d[2 * row_scalar + 1] * fy[equation] -
           n[row_scalar] * rhs);
        for (int column_scalar = 0; column_scalar < 4; ++column_scalar)
          for (int variable = 0; variable < 5; ++variable)
          {
            const double zc = (variable == equation ? n[row_scalar] * n[column_scalar] / p.dt : 0.) -
              d[2 * row_scalar] * jx[5 * equation + variable] * n[column_scalar] -
              d[2 * row_scalar + 1] * jy[5 * equation + variable] * n[column_scalar];
            M(output.A, interior_unknowns, row, I(variable, column_scalar)) += weight * zc;
          }
      }
      for (int direction = 0; direction < 2; ++direction)
      {
        const int row = I(4 + direction, row_scalar);
        output.interior_residual[row] += weight *
          (n[row_scalar] * u[0] * z[direction] + d[2 * row_scalar + direction] * sigma);
        for (int column_scalar = 0; column_scalar < 4; ++column_scalar)
        {
          M(output.A, interior_unknowns, row, I(0, column_scalar)) +=
            weight * n[row_scalar] * z[direction] * n[column_scalar];
          M(output.A, interior_unknowns, row, I(4, column_scalar)) +=
            weight * d[2 * row_scalar + direction] * n[column_scalar];
          M(output.A, interior_unknowns, row, I(5 + direction, column_scalar)) +=
            weight * n[row_scalar] * u[0] * n[column_scalar];
        }
      }
      const int sigma_row = I(6, row_scalar);
      output.interior_residual[sigma_row] += weight *
        (n[row_scalar] * sigma / u[0] + p.alpha *
         (d[2 * row_scalar] * z[0] + d[2 * row_scalar + 1] * z[1]) - p.alpha * n[row_scalar] * q);
      for (int column_scalar = 0; column_scalar < 4; ++column_scalar)
      {
        M(output.A, interior_unknowns, sigma_row, I(4, column_scalar)) +=
          weight * n[row_scalar] * n[column_scalar] / u[0];
        for (int direction = 0; direction < 2; ++direction)
          M(output.A, interior_unknowns, sigma_row, I(5 + direction, column_scalar)) +=
            weight * p.alpha * d[2 * row_scalar + direction] * n[column_scalar];
        for (int variable = 0; variable < 4; ++variable)
        {
          double dq = 0.;
          if (variable < 3)
            for (int velocity = 0; velocity < 2; ++velocity)
              for (int direction = 0; direction < 2; ++direction)
              {
                const double momentum = velocity == 0 ? u[1] : u[2];
                const double momentum_derivative = Derivative(input.interior, 1 + velocity, d, direction);
                const double density_derivative = Derivative(input.interior, 0, d, direction);
                const double delta_density = variable == 0 ? n[column_scalar] : 0.;
                const double delta_density_derivative = variable == 0 ? d[2 * column_scalar + direction] : 0.;
                const double delta_momentum = variable == 1 + velocity ? n[column_scalar] : 0.;
                const double delta_momentum_derivative = variable == 1 + velocity ? d[2 * column_scalar + direction] : 0.;
                const double delta_gradient = delta_momentum_derivative / u[0] -
                  momentum_derivative * delta_density / (u[0] * u[0]) -
                  delta_momentum * density_derivative / (u[0] * u[0]) -
                  momentum * delta_density_derivative / (u[0] * u[0]) +
                  2. * momentum * density_derivative * delta_density / (u[0] * u[0] * u[0]);
                dq += dqg[2 * velocity + direction] * delta_gradient;
              }
          double value = -p.alpha * n[row_scalar] * dq;
          if (variable == 0) value -= n[row_scalar] * sigma * n[column_scalar] / (u[0] * u[0]);
          M(output.A, interior_unknowns, sigma_row, I(variable, column_scalar)) += weight * value;
        }
      }
    }
  }
  for (int face = 0; face < faces; ++face)
    for (int point = 0; point < input.quadrature.segment_points; ++point)
    {
      const double* quadrature = input.quadrature.segment_xw + 2 * point;
      double xi = 0., eta = 0., tangent[2] = {};
      FacePoint(face, quadrature[0], xi, eta, tangent);
      double n[4] = {}, dr[8] = {}, d[8] = {};
      Shapes(xi, eta, n, dr);
      for (int a = 0; a < 4; ++a)
        for (int direction = 0; direction < 2; ++direction)
          d[2 * a + direction] = dr[2 * a] * inverse[direction] +
            dr[2 * a + 1] * inverse[2 + direction];
      const double physical_tangent[2] = {j[0] * tangent[0] + j[2] * tangent[1],
                                          j[1] * tangent[0] + j[3] * tangent[1]};
      const double length = ScaledNorm2(physical_tangent[0], physical_tangent[1]);
      if (!(length > 0.) || !Finite(length)) return Status::invalid_geometry;
      const double normal[2] = {physical_tangent[1] / length, -physical_tangent[0] / length};
      const double h = determinant / length;
      const double hshape[2] = {(0.78867513459481288225 - quadrature[0]) * 1.7320508075688772935,
                                (quadrature[0] - 0.21132486540518711775) * 1.7320508075688772935};
      double u[4] = {}, old[4] = {}, uh[4] = {}, oldh[4] = {};
      Value(input.interior, n, u); Value(input.previous_interior, n, old);
      for (int component = 0; component < 4; ++component)
        for (int a = 0; a < 2; ++a)
        {
          uh[component] += input.trace[T(component, face * 2 + a)] * hshape[a];
          oldh[component] += input.previous_trace[T(component, face * 2 + a)] * hshape[a];
        }
      if (!Admissible(u, p) || !Admissible(old, p) || !Admissible(uh, p) || !Admissible(oldh, p))
        return Status::inadmissible_state;
      const double sigma = SValue(input.interior, 4, n);
      const double z[2] = {SValue(input.interior, 5, n), SValue(input.interior, 6, n)};
      double sigmahat = 0.;
      for (int a = 0; a < 2; ++a) sigmahat += input.trace[T(4, face * 2 + a)] * hshape[a];
      double flux[4] = {}, jacobian[20] = {};
      if (!EulerSigma(uh, sigmahat, normal, p, flux, jacobian)) return Status::inadmissible_state;
      const double tau = 4. / (uh[0] * h);
      const double jump = sigma - sigmahat;
      const double zh = z[0] * normal[0] + z[1] * normal[1] - tau * jump;
      const double weight = quadrature[1] * length;
      if (!Finite(tau) || !(tau > 0.) || !Finite(zh)) return Status::nonfinite_result;
      for (int row_scalar = 0; row_scalar < 4; ++row_scalar)
      {
        for (int equation = 0; equation < 4; ++equation)
        {
          const int row = I(equation, row_scalar);
          const double face_flux = flux[equation] + p.beta * (u[equation] - uh[equation]);
          output.interior_residual[row] += weight * n[row_scalar] * face_flux;
          for (int column_scalar = 0; column_scalar < 4; ++column_scalar)
            M(output.A, interior_unknowns, row, I(equation, column_scalar)) +=
              weight * n[row_scalar] * p.beta * n[column_scalar];
          for (int variable = 0; variable < 5; ++variable)
            for (int a = 0; a < 2; ++a)
              M(output.B, trace_unknowns, row, T(variable, face * 2 + a)) += weight * n[row_scalar] *
                (jacobian[5 * equation + variable] - (variable == equation ? p.beta : 0.)) * hshape[a];
        }
        for (int direction = 0; direction < 2; ++direction)
        {
          const int row = I(4 + direction, row_scalar);
          output.interior_residual[row] -= weight * n[row_scalar] * normal[direction] * sigmahat;
          for (int a = 0; a < 2; ++a)
            M(output.B, trace_unknowns, row, T(4, face * 2 + a)) -=
              weight * n[row_scalar] * normal[direction] * hshape[a];
        }
        const int row = I(6, row_scalar);
        output.interior_residual[row] -= p.alpha * weight * n[row_scalar] * zh;
        for (int column_scalar = 0; column_scalar < 4; ++column_scalar)
        {
          M(output.A, interior_unknowns, row, I(4, column_scalar)) +=
            p.alpha * weight * n[row_scalar] * tau * n[column_scalar];
          for (int direction = 0; direction < 2; ++direction)
            M(output.A, interior_unknowns, row, I(5 + direction, column_scalar)) -=
              p.alpha * weight * n[row_scalar] * normal[direction] * n[column_scalar];
        }
        for (int a = 0; a < 2; ++a)
        {
          M(output.B, trace_unknowns, row, T(4, face * 2 + a)) -=
            p.alpha * weight * n[row_scalar] * tau * hshape[a];
          M(output.B, trace_unknowns, row, T(0, face * 2 + a)) -=
            p.alpha * weight * n[row_scalar] * tau * jump / uh[0] * hshape[a];
        }
      }
      for (int component = 0; component < 5; ++component)
        for (int row_scalar = 0; row_scalar < 2; ++row_scalar)
        {
          const int row = T(component, face * 2 + row_scalar);
          if (component < 4)
          {
            const double face_flux = flux[component] + p.beta * (u[component] - uh[component]);
            output.trace_residual[row] += weight * hshape[row_scalar] * face_flux;
            for (int column_scalar = 0; column_scalar < 4; ++column_scalar)
              M(output.C, interior_unknowns, row, I(component, column_scalar)) +=
                weight * hshape[row_scalar] * p.beta * n[column_scalar];
            for (int variable = 0; variable < 5; ++variable)
              for (int a = 0; a < 2; ++a)
                M(output.D, trace_unknowns, row, T(variable, face * 2 + a)) +=
                  weight * hshape[row_scalar] *
                  (jacobian[5 * component + variable] - (variable == component ? p.beta : 0.)) * hshape[a];
          }
          else
          {
            output.trace_residual[row] += p.alpha * weight * hshape[row_scalar] * zh;
            for (int column_scalar = 0; column_scalar < 4; ++column_scalar)
            {
              M(output.C, interior_unknowns, row, I(4, column_scalar)) -=
                p.alpha * weight * hshape[row_scalar] * tau * n[column_scalar];
              for (int direction = 0; direction < 2; ++direction)
                M(output.C, interior_unknowns, row, I(5 + direction, column_scalar)) +=
                  p.alpha * weight * hshape[row_scalar] * normal[direction] * n[column_scalar];
            }
            for (int a = 0; a < 2; ++a)
            {
              M(output.D, trace_unknowns, row, T(4, face * 2 + a)) +=
                p.alpha * weight * hshape[row_scalar] * tau * hshape[a];
              M(output.D, trace_unknowns, row, T(0, face * 2 + a)) +=
                p.alpha * weight * hshape[row_scalar] * tau * jump / uh[0] * hshape[a];
            }
          }
        }
    }
  if (!AllFinite(output)) return Status::nonfinite_result;
  for (int i = 0; i < interior_unknowns; ++i) output.f[i] = -output.interior_residual[i];
  for (int i = 0; i < trace_unknowns; ++i) output.g[i] = -output.trace_residual[i];
  return Status::success;
}
} // namespace dg::hdg::runtime::common::traced_scalar_igr_p1_local_algebra
