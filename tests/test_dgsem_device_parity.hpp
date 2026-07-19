#pragma once

#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_device.hpp"
#include "numerics/dgsem/dgsem_reference.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace dg_igr_tests
{
namespace dgsem_device_parity_detail
{

inline double MaxDifference(const mfem::DenseMatrix& left,
                            const mfem::DenseMatrix& right)
{
  MFEM_VERIFY(left.Height() == right.Height() && left.Width() == right.Width(),
              "DGSEM device parity matrix dimensions differ.");
  double maximum = 0.0;
  const mfem::real_t* left_data = left.HostRead();
  const mfem::real_t* right_data = right.HostRead();
  for (int i = 0; i < left.TotalSize(); ++i)
  {
    maximum =
      std::max(maximum,
               std::abs(static_cast<double>(left_data[i] - right_data[i])));
  }
  return maximum;
}

inline double MaxAbs(const mfem::DenseMatrix& values)
{
  double maximum = 0.0;
  const mfem::real_t* data = values.HostRead();
  for (int i = 0; i < values.TotalSize(); ++i)
  {
    maximum = std::max(maximum, std::abs(static_cast<double>(data[i])));
  }
  return maximum;
}

inline double
EulerPressure(const mfem::DenseMatrix& states, int row, int dim, double gamma)
{
  const double rho = states(row, 0);
  double momentum_squared = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    momentum_squared += states(row, 1 + d) * states(row, 1 + d);
  }
  return (gamma - 1.0) * (states(row, 1 + dim) - 0.5 * momentum_squared / rho);
}

inline double EulerTensorFluxEntry(const mfem::DenseMatrix& states,
                                   int row,
                                   int dim,
                                   int equation,
                                   int physical,
                                   double gamma,
                                   double sigma,
                                   double rho_floor)
{
  const double rho = states(row, 0);
  const double momentum_direction = states(row, 1 + physical);
  const double pressure = EulerPressure(states, row, dim, gamma);
  if (equation == 0)
  {
    return momentum_direction;
  }
  if (equation <= dim)
  {
    return states(row, equation) * momentum_direction / rho +
           (equation == 1 + physical ? pressure + sigma : 0.0);
  }
  return momentum_direction * (states(row, 1 + dim) + pressure) / rho +
         sigma * momentum_direction / std::max(rho, rho_floor);
}

inline double EulerNormalFluxEntry(const mfem::DenseMatrix& states,
                                   int row,
                                   const mfem::DenseMatrix& normals,
                                   int dim,
                                   int equation,
                                   double gamma,
                                   double sigma,
                                   double rho_floor)
{
  const double rho = states(row, 0);
  double momentum_dot_normal = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    momentum_dot_normal += states(row, 1 + d) * normals(row, d);
  }
  const double normal_velocity = momentum_dot_normal / rho;
  const double pressure = EulerPressure(states, row, dim, gamma);
  if (equation == 0)
  {
    return momentum_dot_normal;
  }
  if (equation <= dim)
  {
    const int direction = equation - 1;
    return normal_velocity * states(row, equation) +
           (pressure + sigma) * normals(row, direction);
  }
  return normal_velocity * (states(row, 1 + dim) + pressure) +
         sigma * momentum_dot_normal / std::max(rho, rho_floor);
}

inline double EulerNormalSpeed(const mfem::DenseMatrix& states,
                               int row,
                               const mfem::DenseMatrix& normals,
                               int dim,
                               double gamma)
{
  const double rho = states(row, 0);
  double momentum_dot_normal = 0.0;
  double normal_squared = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    momentum_dot_normal += states(row, 1 + d) * normals(row, d);
    normal_squared += normals(row, d) * normals(row, d);
  }
  return std::abs(momentum_dot_normal / rho) / std::sqrt(normal_squared) +
         std::sqrt(gamma * EulerPressure(states, row, dim, gamma) / rho);
}

inline void HostCentralVolume(int dim,
                              int n1d,
                              const mfem::Array<int>& lex_to_local,
                              const mfem::DenseMatrix& split,
                              const mfem::Vector& weights,
                              const mfem::DenseMatrix& flux,
                              mfem::DenseMatrix& weak)
{
  const int dof = dim == 1 ? n1d : n1d * n1d;
  const int rows = flux.Height();
  const int num_equations = flux.Width() / dim;
  weak.SetSize(rows, num_equations);
  weak = 0.0;
  for (int row = 0; row < rows; ++row)
  {
    const int element = row / dof;
    const int element_local = row - element * dof;
    int lex = 0;
    while (lex_to_local[lex] != element_local)
    {
      ++lex;
    }
    const int i = lex % n1d;
    const int j = dim == 2 ? lex / n1d : 0;
    for (int equation = 0; equation < num_equations; ++equation)
    {
      double value = 0.0;
      for (int m = 0; m < n1d; ++m)
      {
        if (m == i)
        {
          continue;
        }
        const int other = element * dof + lex_to_local[m + n1d * j];
        value -=
          split(i, m) * 0.5 * (flux(row, equation) + flux(other, equation));
      }
      if (dim == 2)
      {
        for (int m = 0; m < n1d; ++m)
        {
          if (m == j)
          {
            continue;
          }
          const int other = element * dof + lex_to_local[i + n1d * m];
          value -= split(j, m) * 0.5 *
                   (flux(row, num_equations + equation) +
                    flux(other, num_equations + equation));
        }
      }
      double tensor_weight = weights(i);
      if (dim == 2)
      {
        tensor_weight *= weights(j);
      }
      weak(row, equation) = tensor_weight * value;
    }
  }
}

inline void HostRusanovFaces(int dim,
                             int ne,
                             int dof,
                             double gamma,
                             double rho_floor,
                             const mfem::DenseMatrix& left,
                             const mfem::DenseMatrix& right,
                             const mfem::DenseMatrix& normals,
                             const mfem::Vector& sigma_left,
                             const mfem::Vector& sigma_right,
                             const mfem::Vector& weights,
                             const mfem::Array<int>& left_elements,
                             const mfem::Array<int>& left_local,
                             const mfem::Array<int>& right_elements,
                             const mfem::Array<int>& right_local,
                             const mfem::Array<int>& ownership,
                             mfem::DenseMatrix& flux,
                             mfem::Vector& speeds,
                             mfem::DenseMatrix& surface)
{
  const int nfaces = left.Height();
  const int num_equations = dim + 2;
  flux.SetSize(nfaces, num_equations);
  speeds.SetSize(nfaces);
  surface.SetSize(ne * dof, num_equations);
  surface = 0.0;
  for (int face = 0; face < nfaces; ++face)
  {
    speeds(face) = std::max(EulerNormalSpeed(left, face, normals, dim, gamma),
                            EulerNormalSpeed(right, face, normals, dim, gamma));
    double normal_norm = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      normal_norm += normals(face, d) * normals(face, d);
    }
    normal_norm = std::sqrt(normal_norm);
    for (int equation = 0; equation < num_equations; ++equation)
    {
      flux(face, equation) =
        0.5 * (speeds(face) * normal_norm *
                 (left(face, equation) - right(face, equation)) +
               EulerNormalFluxEntry(left,
                                    face,
                                    normals,
                                    dim,
                                    equation,
                                    gamma,
                                    sigma_left(face),
                                    rho_floor) +
               EulerNormalFluxEntry(right,
                                    face,
                                    normals,
                                    dim,
                                    equation,
                                    gamma,
                                    sigma_right(face),
                                    rho_floor));
      surface(left_elements[face] * dof + left_local[face], equation) -=
        weights(face) * flux(face, equation);
      if (ownership[face] ==
          static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Local))
      {
        surface(right_elements[face] * dof + right_local[face], equation) +=
          weights(face) * flux(face, equation);
      }
    }
  }
}

inline void FillConservativeState(mfem::DenseMatrix& states,
                                  int row,
                                  double rho,
                                  double u,
                                  double v,
                                  double pressure,
                                  double gamma)
{
  states(row, 0) = rho;
  states(row, 1) = rho * u;
  states(row, 2) = rho * v;
  states(row, 3) = pressure / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
}

inline std::string RequestedDevice()
{
  const char* value = std::getenv("DG_IGR_DGSEM_DEVICE");
  return value ? std::string(value) : std::string("cpu");
}

inline void EnsureDeviceConfigured(TestContext& ctx,
                                   const std::string& requested)
{
  static std::unique_ptr<mfem::Device> owned_device;
  static std::string configured_request;
  if (!mfem::Device::IsConfigured())
  {
    configured_request = requested;
    owned_device = std::make_unique<mfem::Device>(requested);
  }
  const auto info = mfem::dgsem::GetDGSEMDeviceRuntimeInfo();
  const bool request_is_cuda = requested == "cuda";
  ctx.Check(info.configured, "DGSEM MFEM device configured exactly once");
  ctx.Check(request_is_cuda ? info.cuda : !info.cuda,
            "DGSEM requested MFEM backend is selected");
  if (!configured_request.empty())
  {
    ctx.Check(configured_request == requested,
              "DGSEM host-once device request remains static");
  }
}

} // namespace dgsem_device_parity_detail

inline void RunDGSEMDeviceParityTests(TestContext& ctx)
{
  using namespace dgsem_device_parity_detail;
  constexpr int dim = 2;
  constexpr int num_equations = dim + 2;
  constexpr double gamma = 1.4;
  constexpr double rho_floor = 1.0e-14;
  const double pointwise_tol =
    mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture;
  const double metric_tol = mfem::dgsem::PrecisionAwareMetricTolerance(
    mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture);
  const std::string requested = RequestedDevice();
  EnsureDeviceConfigured(ctx, requested);
  const auto runtime = mfem::dgsem::GetDGSEMDeviceRuntimeInfo();

  double max_basis_error = 0.0;
  double max_metric_error = 0.0;
  double max_sigma_flux_error = 0.0;
  double max_volume_error = 0.0;
  double max_surface_error = 0.0;
  double max_inverse_mass_error = 0.0;
  double max_constant_residual = 0.0;
  double max_local_conservation = 0.0;
  double max_reduction_error = 0.0;

  for (int order = 1; order <= 4; ++order)
  {
    const auto& basis = mfem::dgsem::GetLGLBasis1D(order);
    const int n1d = basis.n;
    const int dof = n1d * n1d;
    mfem::Vector moments;
    mfem::DenseMatrix derivative_action;
    mfem::DenseMatrix sbp_residual;
    mfem::DenseMatrix split_residual;
    mfem::dgsem::EvaluateLGLBasisDevice(basis.nodes,
                                        basis.weights,
                                        basis.D,
                                        basis.Dsplit,
                                        order,
                                        moments,
                                        derivative_action,
                                        sbp_residual,
                                        split_residual);
    moments.HostRead();
    derivative_action.HostRead();
    for (int degree = 0; degree < 2 * order; ++degree)
    {
      const double expected = 1.0 / static_cast<double>(degree + 1);
      max_basis_error =
        std::max(max_basis_error,
                 std::abs(static_cast<double>(moments(degree)) - expected));
    }
    for (int degree = 0; degree <= order; ++degree)
    {
      for (int i = 0; i < n1d; ++i)
      {
        const double expected =
          degree == 0 ? 0.0 : degree * std::pow(basis.nodes(i), degree - 1);
        max_basis_error =
          std::max(max_basis_error,
                   std::abs(static_cast<double>(derivative_action(i, degree)) -
                            expected));
      }
    }
    max_basis_error =
      std::max({max_basis_error, MaxAbs(sbp_residual), MaxAbs(split_residual)});

    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1,
                                                  1,
                                                  mfem::Element::QUADRILATERAL,
                                                  true,
                                                  1.0,
                                                  1.0);
    mfem::FiniteElementSpace fes(&mesh, &fec);
    const mfem::FiniteElement& fe = *fes.GetFE(0);
    const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(fe);
    mfem::Array<int> lex_to_local(dof);
    for (int local = 0; local < dof; ++local)
    {
      const int i = dof_to_ijk[static_cast<std::size_t>(local)][0];
      const int j = dof_to_ijk[static_cast<std::size_t>(local)][1];
      lex_to_local[i + n1d * j] = local;
    }

    mfem::DenseMatrix coordinates(dof, dim);
    for (int j = 0; j < n1d; ++j)
    {
      for (int i = 0; i < n1d; ++i)
      {
        const int local = lex_to_local[i + n1d * j];
        const double xi = basis.nodes(i);
        const double eta = basis.nodes(j);
        coordinates(local, 0) = 1.2 * xi + 0.35 * eta;
        coordinates(local, 1) = -0.2 * xi + 0.9 * eta;
      }
    }
    mfem::DenseMatrix metrics;
    mfem::dgsem::BuildContravariantMetrics2DDevice(1,
                                                   n1d,
                                                   lex_to_local,
                                                   basis.D,
                                                   coordinates,
                                                   metrics);
    metrics.HostRead();
    constexpr double exact_metric[4] = {0.9, -0.35, 0.2, 1.2};
    for (int column = 0; column < 4; ++column)
    {
      for (int local = 0; local < dof; ++local)
      {
        max_metric_error =
          std::max(max_metric_error,
                   std::abs(static_cast<double>(metrics(local, column)) -
                            exact_metric[column]));
      }
    }

    mfem::DenseMatrix states(dof, num_equations);
    mfem::Vector sigma(dof);
    for (int j = 0; j < n1d; ++j)
    {
      for (int i = 0; i < n1d; ++i)
      {
        const int local = lex_to_local[i + n1d * j];
        const double xi = basis.nodes(i);
        const double eta = basis.nodes(j);
        FillConservativeState(states,
                              local,
                              1.0 + 0.04 * xi + 0.02 * eta,
                              0.2 + 0.03 * eta,
                              -0.1 + 0.02 * xi,
                              1.0 + 0.05 * xi - 0.03 * eta,
                              gamma);
        sigma(local) = 0.25 + 0.02 * xi - 0.01 * eta;
      }
    }
    mfem::DenseMatrix device_flux;
    mfem::Vector nodal_speeds;
    mfem::dgsem::BuildEulerContravariantFluxDevice(dim,
                                                   gamma,
                                                   rho_floor,
                                                   states,
                                                   sigma,
                                                   metrics,
                                                   device_flux,
                                                   nodal_speeds);
    mfem::DenseMatrix host_flux(dof, dim * num_equations);
    for (int row = 0; row < dof; ++row)
    {
      for (int reference = 0; reference < dim; ++reference)
      {
        for (int equation = 0; equation < num_equations; ++equation)
        {
          double value = 0.0;
          for (int physical = 0; physical < dim; ++physical)
          {
            value += metrics(row, reference * dim + physical) *
                     EulerTensorFluxEntry(states,
                                          row,
                                          dim,
                                          equation,
                                          physical,
                                          gamma,
                                          sigma(row),
                                          rho_floor);
          }
          host_flux(row, reference * num_equations + equation) = value;
        }
      }
    }
    max_sigma_flux_error =
      std::max(max_sigma_flux_error, MaxDifference(device_flux, host_flux));

    mfem::DenseMatrix device_volume;
    mfem::DenseMatrix host_volume;
    mfem::dgsem::ApplyCentralVolumeDevice(dim,
                                          n1d,
                                          lex_to_local,
                                          basis.Dsplit,
                                          basis.weights,
                                          device_flux,
                                          device_volume);
    HostCentralVolume(dim,
                      n1d,
                      lex_to_local,
                      basis.Dsplit,
                      basis.weights,
                      host_flux,
                      host_volume);
    max_volume_error =
      std::max(max_volume_error, MaxDifference(device_volume, host_volume));

    mfem::DenseMatrix constant_states(dof, num_equations);
    mfem::Vector constant_sigma(dof);
    for (int local = 0; local < dof; ++local)
    {
      FillConservativeState(constant_states,
                            local,
                            1.0,
                            0.25,
                            -0.1,
                            1.0,
                            gamma);
      constant_sigma(local) = 0.25;
    }
    mfem::DenseMatrix constant_flux;
    mfem::Vector constant_speeds;
    mfem::dgsem::BuildEulerContravariantFluxDevice(dim,
                                                   gamma,
                                                   rho_floor,
                                                   constant_states,
                                                   constant_sigma,
                                                   metrics,
                                                   constant_flux,
                                                   constant_speeds);
    mfem::DenseMatrix constant_volume;
    mfem::dgsem::ApplyCentralVolumeDevice(dim,
                                          n1d,
                                          lex_to_local,
                                          basis.Dsplit,
                                          basis.weights,
                                          constant_flux,
                                          constant_volume);

    const int boundary_faces = 4 * n1d;
    mfem::DenseMatrix boundary_left(boundary_faces, num_equations);
    mfem::DenseMatrix boundary_right(boundary_faces, num_equations);
    mfem::DenseMatrix boundary_normals(boundary_faces, dim);
    mfem::Vector boundary_sigma_left(boundary_faces);
    mfem::Vector boundary_sigma_right(boundary_faces);
    mfem::Vector boundary_weights(boundary_faces);
    mfem::Array<int> boundary_left_elements(boundary_faces);
    mfem::Array<int> boundary_left_local(boundary_faces);
    mfem::Array<int> boundary_right_elements(boundary_faces);
    mfem::Array<int> boundary_right_local(boundary_faces);
    mfem::Array<int> boundary_ownership(boundary_faces);
    int face = 0;
    auto add_boundary = [&](int local, double nx, double ny, double weight)
    {
      for (int equation = 0; equation < num_equations; ++equation)
      {
        boundary_left(face, equation) = constant_states(local, equation);
        boundary_right(face, equation) = constant_states(local, equation);
      }
      boundary_normals(face, 0) = nx;
      boundary_normals(face, 1) = ny;
      boundary_sigma_left(face) = constant_sigma(local);
      boundary_sigma_right(face) = constant_sigma(local);
      boundary_weights(face) = weight;
      boundary_left_elements[face] = 0;
      boundary_left_local[face] = local;
      boundary_right_elements[face] = -1;
      boundary_right_local[face] = -1;
      boundary_ownership[face] =
        static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Boundary);
      ++face;
    };
    for (int j = 0; j < n1d; ++j)
    {
      add_boundary(lex_to_local[n1d * j],
                   -exact_metric[0],
                   -exact_metric[1],
                   basis.weights(j));
      add_boundary(lex_to_local[n1d - 1 + n1d * j],
                   exact_metric[0],
                   exact_metric[1],
                   basis.weights(j));
    }
    for (int i = 0; i < n1d; ++i)
    {
      add_boundary(lex_to_local[i],
                   -exact_metric[2],
                   -exact_metric[3],
                   basis.weights(i));
      add_boundary(lex_to_local[i + n1d * (n1d - 1)],
                   exact_metric[2],
                   exact_metric[3],
                   basis.weights(i));
    }
    mfem::DenseMatrix boundary_flux;
    mfem::Vector boundary_speeds;
    mfem::DenseMatrix constant_surface;
    mfem::dgsem::AssembleEulerRusanovFacesDevice(dim,
                                                 1,
                                                 dof,
                                                 gamma,
                                                 rho_floor,
                                                 boundary_left,
                                                 boundary_right,
                                                 boundary_normals,
                                                 boundary_sigma_left,
                                                 boundary_sigma_right,
                                                 boundary_weights,
                                                 boundary_left_elements,
                                                 boundary_left_local,
                                                 boundary_right_elements,
                                                 boundary_right_local,
                                                 boundary_ownership,
                                                 boundary_flux,
                                                 boundary_speeds,
                                                 constant_surface);
    constant_volume.HostRead();
    constant_surface.HostRead();
    constant_surface += constant_volume;
    max_constant_residual =
      std::max(max_constant_residual, MaxAbs(constant_surface));

    const int inverse_ne = 2;
    const int inverse_rows = inverse_ne * dof;
    mfem::DenseMatrix inverse_mass(inverse_rows, dof);
    inverse_mass = 0.0;
    mfem::DenseMatrix weak(inverse_rows, num_equations);
    for (int element = 0; element < inverse_ne; ++element)
    {
      for (int local = 0; local < dof; ++local)
      {
        const auto& ijk = dof_to_ijk[static_cast<std::size_t>(local)];
        inverse_mass(element * dof + local, local) =
          (1.0 + 0.125 * element) /
          (basis.weights(ijk[0]) * basis.weights(ijk[1]));
        for (int equation = 0; equation < num_equations; ++equation)
        {
          weak(element * dof + local, equation) =
            0.1 * (1 + element) + 0.03 * local - 0.02 * equation;
        }
      }
    }
    mfem::DenseMatrix device_strong;
    mfem::DenseMatrix host_strong(inverse_rows, num_equations);
    mfem::dgsem::ApplyInverseMassDevice(inverse_ne,
                                        dof,
                                        inverse_mass,
                                        weak,
                                        device_strong);
    for (int element = 0; element < inverse_ne; ++element)
    {
      for (int local = 0; local < dof; ++local)
      {
        for (int equation = 0; equation < num_equations; ++equation)
        {
          double value = 0.0;
          for (int j = 0; j < dof; ++j)
          {
            value += inverse_mass(element * dof + local, j) *
                     weak(element * dof + j, equation);
          }
          host_strong(element * dof + local, equation) = value;
        }
      }
    }
    max_inverse_mass_error =
      std::max(max_inverse_mass_error,
               MaxDifference(device_strong, host_strong));
  }

  for (int face_order = 1; face_order <= 4; ++face_order)
  {
    const auto& face_basis = mfem::dgsem::GetLGLBasis1D(face_order);
    const int face_n1d = face_basis.n;
    const int face_dof = face_n1d * face_n1d;
    const int nfaces = 2 * face_n1d;
    mfem::DenseMatrix left(nfaces, num_equations);
    mfem::DenseMatrix right(nfaces, num_equations);
    mfem::DenseMatrix normals(nfaces, dim);
    mfem::Vector sigma_left(nfaces);
    mfem::Vector sigma_right(nfaces);
    mfem::Vector weights(nfaces);
    mfem::Array<int> left_elements(nfaces);
    mfem::Array<int> left_local(nfaces);
    mfem::Array<int> right_elements(nfaces);
    mfem::Array<int> right_local(nfaces);
    mfem::Array<int> ownership(nfaces);
    for (int q = 0; q < face_n1d; ++q)
    {
      FillConservativeState(left,
                            q,
                            1.0 + 0.02 * q,
                            0.2,
                            -0.05 + 0.01 * q,
                            1.0,
                            gamma);
      FillConservativeState(right,
                            q,
                            0.95 + 0.01 * q,
                            0.18,
                            -0.03,
                            0.98,
                            gamma);
      normals(q, 0) = 0.9;
      normals(q, 1) = -0.35;
      sigma_left(q) = 0.25;
      sigma_right(q) = 0.2;
      weights(q) = face_basis.weights(q);
      left_elements[q] = 0;
      left_local[q] = face_n1d - 1 + face_n1d * q;
      right_elements[q] = 1;
      right_local[q] = face_n1d * (face_n1d - 1 - q);
      ownership[q] =
        static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Local);

      const int shared = face_n1d + q;
      for (int equation = 0; equation < num_equations; ++equation)
      {
        left(shared, equation) = left(q, equation);
        right(shared, equation) = right(q, equation);
      }
      normals(shared, 0) = -0.2;
      normals(shared, 1) = -1.2;
      sigma_left(shared) = 0.25;
      sigma_right(shared) = 0.2;
      weights(shared) = face_basis.weights(q);
      left_elements[shared] = 0;
      left_local[shared] = q;
      right_elements[shared] = -1;
      right_local[shared] = face_n1d - 1 - q;
      ownership[shared] =
        static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Shared);
    }
    mfem::DenseMatrix device_face_flux;
    mfem::Vector device_face_speeds;
    mfem::DenseMatrix device_surface;
    mfem::DenseMatrix host_face_flux;
    mfem::Vector host_face_speeds;
    mfem::DenseMatrix host_surface;
    mfem::dgsem::AssembleEulerRusanovFacesDevice(dim,
                                                 2,
                                                 face_dof,
                                                 gamma,
                                                 rho_floor,
                                                 left,
                                                 right,
                                                 normals,
                                                 sigma_left,
                                                 sigma_right,
                                                 weights,
                                                 left_elements,
                                                 left_local,
                                                 right_elements,
                                                 right_local,
                                                 ownership,
                                                 device_face_flux,
                                                 device_face_speeds,
                                                 device_surface);
    HostRusanovFaces(dim,
                     2,
                     face_dof,
                     gamma,
                     rho_floor,
                     left,
                     right,
                     normals,
                     sigma_left,
                     sigma_right,
                     weights,
                     left_elements,
                     left_local,
                     right_elements,
                     right_local,
                     ownership,
                     host_face_flux,
                     host_face_speeds,
                     host_surface);
    device_face_speeds.HostRead();
    max_surface_error =
      std::max({max_surface_error,
                MaxDifference(device_face_flux, host_face_flux),
                MaxDifference(device_surface, host_surface)});
    for (int face = 0; face < nfaces; ++face)
    {
      max_surface_error =
        std::max(max_surface_error,
                 std::abs(static_cast<double>(device_face_speeds(face) -
                                              host_face_speeds(face))));
    }

    mfem::DenseMatrix local_surface = device_surface;
    local_surface.HostReadWrite();
    for (int q = 0; q < face_n1d; ++q)
    {
      const int shared = face_n1d + q;
      for (int equation = 0; equation < num_equations; ++equation)
      {
        local_surface(left_elements[shared] * face_dof + left_local[shared],
                      equation) +=
          weights(shared) * device_face_flux(shared, equation);
      }
    }
    mfem::Vector component_sums;
    mfem::dgsem::ComponentSumsDevice(local_surface, component_sums);
    component_sums.HostRead();
    double order_local_conservation = 0.0;
    for (int equation = 0; equation < num_equations; ++equation)
    {
      order_local_conservation =
        std::max(order_local_conservation,
                 std::abs(static_cast<double>(component_sums(equation))));
    }
    max_local_conservation =
      std::max(max_local_conservation, order_local_conservation);
    mfem::Vector max_face_speed;
    mfem::dgsem::DeterministicMaximumDevice(device_face_speeds, max_face_speed);
    max_face_speed.HostRead();
    double host_max_face_speed = 0.0;
    for (int face = 0; face < nfaces; ++face)
    {
      host_max_face_speed =
        std::max(host_max_face_speed,
                 static_cast<double>(host_face_speeds(face)));
    }
    max_reduction_error = std::max(
      max_reduction_error,
      std::abs(static_cast<double>(max_face_speed(0)) - host_max_face_speed));
  }

  ctx.Check(max_basis_error <= mfem::dgsem::DGSEMReferenceTolerances::sbp,
            "DGSEM MFEM device p1-p4 basis parity uses frozen SBP gate");
  ctx.Check(max_metric_error <= metric_tol,
            "DGSEM MFEM device p1-p4 sheared metric parity");
  ctx.Check(max_sigma_flux_error <= pointwise_tol,
            "DGSEM MFEM device frozen scalar-sigma tensor-flux parity");
  ctx.Check(max_volume_error <= pointwise_tol,
            "DGSEM MFEM device p1-p4 central volume parity");
  ctx.Check(max_surface_error <= pointwise_tol,
            "DGSEM MFEM device local/shared/oriented surface parity");
  ctx.Check(max_inverse_mass_error <=
              mfem::dgsem::DGSEMReferenceTolerances::inverse_mass,
            "DGSEM MFEM device p1-p4 inverse-mass parity");
  ctx.Check(max_constant_residual <=
              mfem::dgsem::DGSEMReferenceTolerances::constant_affine,
            "DGSEM MFEM device p1-p4 sheared constant-state cancellation");
  ctx.Check(max_local_conservation <=
              mfem::dgsem::DGSEMReferenceTolerances::conservation,
            "DGSEM MFEM device component-wise local-face conservation");
  ctx.Check(max_reduction_error <= pointwise_tol,
            "DGSEM MFEM device p1-p4 deterministic speed reduction parity");
  ctx.Check(
    runtime.device_memory_class ==
      (runtime.cuda ? mfem::MemoryClass::DEVICE : mfem::MemoryClass::HOST),
    "DGSEM MFEM memory class matches selected execution backend");
  ctx.Check(std::string(mfem::dgsem::kDGSEMScalarIGRScope)
                .find("neither repairs B-SCI-001") != std::string::npos,
            "DGSEM device parity preserves B-SCI-001 disclaimer");
  ctx.Check(std::string(mfem::dgsem::kDGSEMScalarIGRScope)
                .find("production wall stability") != std::string::npos,
            "DGSEM device parity preserves wall-stability disclaimer");

  ctx.AddParity(max_basis_error + max_metric_error + max_sigma_flux_error +
                max_volume_error + max_surface_error + max_inverse_mass_error +
                max_constant_residual + max_local_conservation +
                max_reduction_error);

  const char* output_directory = std::getenv("DG_IGR_DGSEM_DEVICE_OUTPUT_DIR");
  if (output_directory && mfem::Mpi::Root())
  {
    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    std::ofstream out(std::filesystem::path(output_directory) /
                      "dgsem_device_parity.json");
    out.setf(std::ios::scientific);
    out.precision(17);
    out << "{\n"
        << "  \"schema\": \"dgsem-mfem-device-parity-v1\",\n"
        << "  \"requested_device\": \"" << requested << "\",\n"
        << "  \"mfem_device_configured\": "
        << (runtime.configured ? "true" : "false") << ",\n"
        << "  \"mfem_device_enabled\": " << (runtime.enabled ? "true" : "false")
        << ",\n"
        << "  \"mfem_cuda_selected\": " << (runtime.cuda ? "true" : "false")
        << ",\n"
        << "  \"device_id\": " << runtime.device_id << ",\n"
        << "  \"device_count\": " << runtime.device_count << ",\n"
        << "  \"host_memory_type\": \""
        << mfem::MemoryTypeName[static_cast<int>(runtime.host_memory_type)]
        << "\",\n"
        << "  \"device_memory_type\": \""
        << mfem::MemoryTypeName[static_cast<int>(runtime.device_memory_type)]
        << "\",\n"
        << "  \"orders\": [1,2,3,4],\n"
        << "  \"kernels_executed\": {\"basis\":true,\"metric\":true,"
           "\"euler_sigma_flux\":true,\"volume\":true,\"surface\":true,"
           "\"inverse_mass\":true,\"component_reduction\":true,"
           "\"maximum_reduction\":true},\n"
        << "  \"max_basis_error\": " << max_basis_error << ",\n"
        << "  \"max_metric_error\": " << max_metric_error << ",\n"
        << "  \"max_sigma_flux_error\": " << max_sigma_flux_error << ",\n"
        << "  \"max_volume_error\": " << max_volume_error << ",\n"
        << "  \"max_surface_error\": " << max_surface_error << ",\n"
        << "  \"max_inverse_mass_error\": " << max_inverse_mass_error << ",\n"
        << "  \"max_constant_residual\": " << max_constant_residual << ",\n"
        << "  \"max_local_conservation\": " << max_local_conservation << ",\n"
        << "  \"max_reduction_error\": " << max_reduction_error << ",\n"
        << "  \"scope\": \"" << mfem::dgsem::kDGSEMScalarIGRScope << "\"\n}\n";
    ctx.Check(!error && out.good(),
              "DGSEM MFEM device machine-readable parity evidence written");
  }
  ctx.LogMetric(
    "DGSEM MFEM backend requested=" + requested +
    " cuda_selected=" + (runtime.cuda ? "true" : "false") + " device_memory=" +
    mfem::MemoryTypeName[static_cast<int>(runtime.device_memory_type)]);
}

} // namespace dg_igr_tests
