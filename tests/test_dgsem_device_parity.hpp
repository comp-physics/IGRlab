#pragma once

#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_device.hpp"
#include "numerics/dgsem/dgsem_reference.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <string>
#include <vector>

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
    if (!std::isfinite(static_cast<double>(left_data[i])) ||
        !std::isfinite(static_cast<double>(right_data[i])))
      return std::numeric_limits<double>::infinity();
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
    if (!std::isfinite(static_cast<double>(data[i])))
      return std::numeric_limits<double>::infinity();
    maximum = std::max(maximum, std::abs(static_cast<double>(data[i])));
  }
  return maximum;
}

inline void BuildRowFaceAdjacency(
  int ne,
  int dof,
  const mfem::Array<int>& left_elements,
  const mfem::Array<int>& left_local,
  const mfem::Array<int>& right_elements,
  const mfem::Array<int>& right_local,
  const mfem::Array<int>& ownership,
  mfem::Array<int>& offsets,
  mfem::Array<int>& face_ids,
  mfem::Array<int>& signs)
{
  const int rows = ne * dof;
  const int nfaces = left_elements.Size();
  std::vector<int> counts(static_cast<std::size_t>(rows), 0);
  for (int face = 0; face < nfaces; ++face)
  {
    ++counts[static_cast<std::size_t>(
      left_elements[face] * dof + left_local[face])];
    if (ownership[face] == static_cast<int>(
          mfem::dgsem::DGSEMDeviceFaceOwnership::Local))
    {
      ++counts[static_cast<std::size_t>(
        right_elements[face] * dof + right_local[face])];
    }
  }
  offsets.SetSize(rows + 1);
  offsets[0] = 0;
  for (int row = 0; row < rows; ++row)
  {
    offsets[row + 1] = offsets[row] + counts[static_cast<std::size_t>(row)];
  }
  face_ids.SetSize(offsets[rows]);
  signs.SetSize(offsets[rows]);
  std::vector<int> cursor(static_cast<std::size_t>(rows));
  for (int row = 0; row < rows; ++row)
  {
    cursor[static_cast<std::size_t>(row)] = offsets[row];
  }
  for (int face = 0; face < nfaces; ++face)
  {
    const int left_row = left_elements[face] * dof + left_local[face];
    int adjacent = cursor[static_cast<std::size_t>(left_row)]++;
    face_ids[adjacent] = face;
    signs[adjacent] = -1;
    if (ownership[face] == static_cast<int>(
          mfem::dgsem::DGSEMDeviceFaceOwnership::Local))
    {
      const int right_row = right_elements[face] * dof + right_local[face];
      adjacent = cursor[static_cast<std::size_t>(right_row)]++;
      face_ids[adjacent] = face;
      signs[adjacent] = 1;
    }
  }
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
                                  int dim,
                                  double rho,
                                  double u,
                                  double v,
                                  double pressure,
                                  double gamma)
{
  states(row, 0) = rho;
  states(row, 1) = rho * u;
  if (dim == 2)
  {
    states(row, 2) = rho * v;
  }
  states(row, 1 + dim) =
    pressure / (gamma - 1.0) + 0.5 * rho * (u * u + (dim == 2 ? v * v : 0.0));
}

struct DimensionalKernelErrors
{
  double metric = 0.0;
  double sigma_flux = 0.0;
  double volume = 0.0;
  double surface = 0.0;
  double inverse_mass = 0.0;
  double constant_residual = 0.0;
  double local_conservation = 0.0;
  double reduction = 0.0;
};

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
  std::array<DimensionalKernelErrors, 2> dimension_errors;

  for (int dim = 1; dim <= 2; ++dim)
  {
    const int num_equations = dim + 2;
    auto& errors = dimension_errors[static_cast<std::size_t>(dim - 1)];
    double& max_metric_error = errors.metric;
    double& max_sigma_flux_error = errors.sigma_flux;
    double& max_volume_error = errors.volume;
    double& max_surface_error = errors.surface;
    double& max_inverse_mass_error = errors.inverse_mass;
    double& max_constant_residual = errors.constant_residual;
    double& max_local_conservation = errors.local_conservation;
    double& max_reduction_error = errors.reduction;

    for (int order = 1; order <= 4; ++order)
    {
      const auto& basis = mfem::dgsem::GetLGLBasis1D(order);
      const int n1d = basis.n;
      const int dof = dim == 1 ? n1d : n1d * n1d;
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
          max_basis_error = std::max(
            max_basis_error,
            std::abs(static_cast<double>(derivative_action(i, degree)) -
                     expected));
        }
      }
      max_basis_error = std::max(
        {max_basis_error, MaxAbs(sbp_residual), MaxAbs(split_residual)});

      mfem::Array<int> lex_to_local(dof);
      if (dim == 1)
      {
        for (int i = 0; i < n1d; ++i)
        {
          lex_to_local[i] = i;
        }
      }
      else
      {
        mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
        mfem::Mesh mesh =
          mfem::Mesh::MakeCartesian2D(1,
                                      1,
                                      mfem::Element::QUADRILATERAL,
                                      true,
                                      1.0,
                                      1.0);
        mfem::FiniteElementSpace fes(&mesh, &fec);
        const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(*fes.GetFE(0));
        for (int local = 0; local < dof; ++local)
        {
          const int i = dof_to_ijk[static_cast<std::size_t>(local)][0];
          const int j = dof_to_ijk[static_cast<std::size_t>(local)][1];
          lex_to_local[i + n1d * j] = local;
        }

        // H1 and L2 use different local ordering at the same LGL nodes.
        // Compare the compact gather against actual FE shape evaluation.
        mfem::H1_FECollection scalar_fec(order, dim,
                                        mfem::BasisType::GaussLobatto);
        mfem::FiniteElementSpace scalar_fes(&mesh, &scalar_fec);
        const auto& scalar_fe = *scalar_fes.GetFE(0);
        mfem::Array<int> scalar_dofs, nodal_map(dof), dense_map(dof * dof);
        scalar_fes.GetElementDofs(0, scalar_dofs);
        mfem::Vector scalar_field(scalar_fes.GetVSize()), shape(dof);
        for (int j = 0; j < scalar_field.Size(); ++j)
          scalar_field[j] = std::sin(0.37 * j) + 0.05 * j;
        mfem::DenseMatrix interpolation(dof, dof), nodal_interpolation;
        bool complete_node_map = true;
        for (int row = 0; row < dof; ++row)
        {
          const auto& point = fes.GetFE(0)->GetNodes().IntPoint(row);
          scalar_fe.CalcShape(point, shape);
          int match = -1;
          for (int j = 0; j < dof; ++j)
          {
            dense_map[row + dof * j] = scalar_dofs[j];
            interpolation(row, j) = shape[j];
            const auto& node = scalar_fe.GetNodes().IntPoint(j);
            if (point.x == node.x && point.y == node.y) match = j;
          }
          complete_node_map = complete_node_map && match >= 0;
          nodal_map[row] = match >= 0 ? scalar_dofs[match] : 0;
        }
        ctx.Check(complete_node_map,
                  "H1/L2 LGL points admit an exact scalar node permutation");
        mfem::Vector dense_scalar, gathered_scalar;
        mfem::dgsem::InterpolateElementScalarDevice(dof, dof, dense_map,
            interpolation, scalar_field, dense_scalar);
        mfem::dgsem::InterpolateElementScalarDevice(dof, dof, nodal_map,
            nodal_interpolation, scalar_field, gathered_scalar);
        dense_scalar.HostRead(); gathered_scalar.HostRead();
        double interpolation_error = 0.0;
        for (int row = 0; row < dof; ++row)
          interpolation_error = std::max(interpolation_error,
              std::abs(double(dense_scalar[row] - gathered_scalar[row])));
        ctx.Check(interpolation_error < pointwise_tol,
                  "Exact scalar node gather matches dense FE interpolation");
      }

      mfem::DenseMatrix metrics;
      constexpr double exact_metric[4] = {0.9, -0.35, 0.2, 1.2};
      if (dim == 1)
      {
        metrics.SetSize(dof, 1);
        metrics = 1.0;
      }
      else
      {
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
        mfem::dgsem::BuildContravariantMetrics2DDevice(1,
                                                       n1d,
                                                       lex_to_local,
                                                       basis.D,
                                                       coordinates,
                                                       metrics);
        metrics.HostRead();
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
      }

      mfem::DenseMatrix states(dof, num_equations);
      mfem::Vector sigma(dof);
      for (int lex = 0; lex < dof; ++lex)
      {
        const int i = lex % n1d;
        const int j = dim == 2 ? lex / n1d : 0;
        const int local = lex_to_local[lex];
        const double xi = basis.nodes(i);
        const double eta = dim == 2 ? basis.nodes(j) : 0.0;
        FillConservativeState(states,
                              local,
                              dim,
                              1.0 + 0.04 * xi + 0.02 * eta,
                              0.2 + 0.03 * eta,
                              -0.1 + 0.02 * xi,
                              1.0 + 0.05 * xi - 0.03 * eta,
                              gamma);
        sigma(local) = 0.25 + 0.02 * xi - 0.01 * eta;
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
                              dim,
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

      const int boundary_faces = dim == 1 ? 2 : 4 * n1d;
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
        if (dim == 2)
        {
          boundary_normals(face, 1) = ny;
        }
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
      if (dim == 1)
      {
        add_boundary(lex_to_local[0], -1.0, 0.0, 1.0);
        add_boundary(lex_to_local[n1d - 1], 1.0, 0.0, 1.0);
      }
      else
      {
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
      }
      mfem::DenseMatrix boundary_flux;
      mfem::Vector boundary_speeds;
      mfem::DenseMatrix constant_surface;
      mfem::Array<int> boundary_row_offsets;
      mfem::Array<int> boundary_row_face_ids;
      mfem::Array<int> boundary_row_face_signs;
      BuildRowFaceAdjacency(1,
                            dof,
                            boundary_left_elements,
                            boundary_left_local,
                            boundary_right_elements,
                            boundary_right_local,
                            boundary_ownership,
                            boundary_row_offsets,
                            boundary_row_face_ids,
                            boundary_row_face_signs);
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
                                                   boundary_row_offsets,
                                                   boundary_row_face_ids,
                                                   boundary_row_face_signs,
                                                   boundary_flux,
                                                   boundary_speeds,
                                                   constant_surface);
      if (runtime.cuda && dim == 2)
      {
        // Nonconstant metric coefficients and arbitrary nonconstant unique
        // face fluxes test the fused algebra independently of free-stream
        // cancellation. Every existing LGL order and native dof map is used.
        mfem::DenseMatrix fused_metric(dof, 4), unique_flux(boundary_faces, 4);
        mfem::DenseMatrix compact_mass(dof, 1), reference_flux(dof, 8);
        metrics.HostRead(); states.HostRead(); sigma.HostRead();
        boundary_flux.HostRead();
        for (int row = 0; row < dof; ++row)
        {
          compact_mass(row, 0) = 1.0 + 0.03 * row;
          for (int component = 0; component < 4; ++component)
            fused_metric(row, component) = metrics(row, component) *
                (1.0 + 0.005 * (component + 1) * row);
          for (int direction = 0; direction < 2; ++direction)
            for (int equation = 0; equation < 4; ++equation)
            {
              double value = 0.0;
              for (int physical = 0; physical < 2; ++physical)
                value += fused_metric(row, direction * 2 + physical) *
                    EulerTensorFluxEntry(states, row, 2, equation, physical,
                                         gamma, sigma(row), rho_floor);
              reference_flux(row, equation + 4 * direction) = value;
            }
        }
        for (int face = 0; face < boundary_faces; ++face)
          for (int equation = 0; equation < 4; ++equation)
            unique_flux(face, equation) = boundary_flux(face, equation) *
                                         (1.0 + 0.017 * face);
        for (bool llf : {false, true})
        {
          mfem::DenseMatrix expected;
          HostCentralVolume(2, n1d, lex_to_local, basis.Dsplit,
                            basis.weights, reference_flux, expected);
          if (llf)
          {
            for (int direction = 0; direction < 2; ++direction)
              for (int line = 0; line < n1d; ++line)
                for (int low = 0; low < n1d; ++low)
                  for (int high = low + 1; high < n1d; ++high)
                  {
                    const int left = lex_to_local[direction == 0
                        ? low + n1d * line : line + n1d * low];
                    const int right = lex_to_local[direction == 0
                        ? high + n1d * line : line + n1d * high];
                    const double mx = 0.5 * (fused_metric(left, 2 * direction) +
                                             fused_metric(right, 2 * direction));
                    const double my = 0.5 * (fused_metric(left, 2 * direction + 1) +
                                             fused_metric(right, 2 * direction + 1));
                    auto speed = [&](int row)
                    {
                      const double rho = states(row, 0);
                      const double px = states(row, 1), py = states(row, 2);
                      const double pressure = (gamma - 1.0) *
                          (states(row, 3) - 0.5 * (px * px + py * py) / rho);
                      const double q = gamma * pressure + (gamma - 1.0) * sigma(row);
                      return std::abs((px / rho) * mx + (py / rho) * my) +
                          std::sqrt(std::abs(q) / rho) * std::sqrt(mx * mx + my * my);
                    };
                    const double scale = std::max(speed(left), speed(right));
                    for (int equation = 0; equation < 4; ++equation)
                    {
                      const double pair = -0.5 * scale *
                          (states(right, equation) - states(left, equation));
                      expected(left, equation) -= basis.weights(low) *
                          basis.weights(line) * basis.Dsplit(low, high) * pair;
                      expected(right, equation) -= basis.weights(high) *
                          basis.weights(line) * basis.Dsplit(high, low) * pair;
                    }
                  }
          }
          for (int row = 0; row < dof; ++row)
            for (int equation = 0; equation < 4; ++equation)
            {
              for (int adjacent = boundary_row_offsets[row];
                   adjacent < boundary_row_offsets[row + 1]; ++adjacent)
              {
                const int face = boundary_row_face_ids[adjacent];
                expected(row, equation) += boundary_row_face_signs[adjacent] *
                    boundary_weights(face) * unique_flux(face, equation);
              }
              expected(row, equation) *= compact_mass(row, 0);
            }
          for (bool frozen_speed : {false, true})
          for (bool tensor_contraction : {false, true})
          {
            mfem::DenseMatrix fused;
            mfem::Vector fused_speeds, ordinary_speeds;
            mfem::dgsem::ApplyFusedEulerResidualDevice(1, n1d, gamma,
                rho_floor, llf, frozen_speed, lex_to_local, basis.Dsplit,
                basis.weights, states, sigma, fused_metric, compact_mass,
                boundary_weights, boundary_row_offsets, boundary_row_face_ids,
                boundary_row_face_signs, unique_flux, fused_speeds, fused,
                tensor_contraction);
            mfem::dgsem::ComputeEulerNodalCharacteristicSpeedsDevice(2, gamma,
                rho_floor, states, sigma, ordinary_speeds, frozen_speed);
            ctx.Check(MaxDifference(fused, expected) < 32.0 * pointwise_tol,
                      tensor_contraction
                        ? "Fused FP64 DMMA residual matches central/LLF host algebra"
                        : "Fused FP64 SIMT residual matches central/LLF host algebra");
            fused_speeds.HostRead(); ordinary_speeds.HostRead();
            double speed_error = 0.0;
            for (int row = 0; row < dof; ++row)
            {
              if (!std::isfinite(double(fused_speeds[row])) ||
                  !std::isfinite(double(ordinary_speeds[row])))
              {
                speed_error = std::numeric_limits<double>::infinity();
                break;
              }
              speed_error = std::max(speed_error,
                  std::abs(double(fused_speeds[row] - ordinary_speeds[row])));
            }
            ctx.Check(speed_error < pointwise_tol,
                      "Fused CUDA residual retains nodal characteristic speeds");
          }
        }
        // LLF floors density inside its pressure calculation, independently
        // of the raw-density physical flux. Compare this rare branch with the
        // existing unfused implementation using substantial kinetic energy.
        mfem::DenseMatrix floored_states(states), floored_flux;
        floored_states.HostReadWrite();
        floored_states(0, 0) = 0.5 * rho_floor;
        floored_states(0, 1) = 0.125 * std::sqrt(rho_floor);
        floored_states(0, 2) = 0.0625 * std::sqrt(rho_floor);
        mfem::DenseMatrix ordinary_weak, pair_dissipation;
        mfem::Vector ordinary_speeds;
        mfem::dgsem::BuildEulerContravariantFluxDevice(2, gamma, rho_floor,
            floored_states, sigma, fused_metric, floored_flux, ordinary_speeds);
        mfem::dgsem::ApplyCentralVolumeDevice(2, n1d, lex_to_local,
            basis.Dsplit, basis.weights, floored_flux, ordinary_weak);
        mfem::dgsem::AddEulerLLFVolumeDissipationDevice(2, 1, n1d, gamma,
            rho_floor, lex_to_local, basis.Dsplit, basis.weights,
            floored_states, sigma, fused_metric, pair_dissipation,
            ordinary_weak);
        ordinary_weak.HostReadWrite();
        for (int row = 0; row < dof; ++row)
          for (int equation = 0; equation < 4; ++equation)
          {
            for (int adjacent = boundary_row_offsets[row];
                 adjacent < boundary_row_offsets[row + 1]; ++adjacent)
            {
              const int face = boundary_row_face_ids[adjacent];
              ordinary_weak(row, equation) += boundary_row_face_signs[adjacent] *
                  boundary_weights(face) * unique_flux(face, equation);
            }
            ordinary_weak(row, equation) *= compact_mass(row, 0);
          }
        for (bool tensor_contraction : {false, true})
        {
          mfem::DenseMatrix fused;
          mfem::Vector fused_speeds;
          mfem::dgsem::ApplyFusedEulerResidualDevice(1, n1d, gamma, rho_floor,
              true, false, lex_to_local, basis.Dsplit, basis.weights,
              floored_states, sigma, fused_metric, compact_mass,
              boundary_weights, boundary_row_offsets, boundary_row_face_ids,
              boundary_row_face_signs, unique_flux, fused_speeds, fused,
              tensor_contraction);
          ctx.Check(MaxDifference(fused, ordinary_weak) <
                        64.0 * pointwise_tol * std::max(1.0, MaxAbs(ordinary_weak)),
                    "Fused LLF residual preserves floored-density helper algebra");
        }
      }
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
          int lex = 0;
          while (lex_to_local[lex] != local)
          {
            ++lex;
          }
          mfem::real_t tensor_weight = basis.weights(lex % n1d);
          if (dim == 2)
          {
            tensor_weight *= basis.weights(lex / n1d);
          }
          inverse_mass(element * dof + local, local) =
            (1.0 + 0.125 * element) / tensor_weight;
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

      // The collocated mass has an exact compact representation. Exercise
      // both device branches on identical data before introducing a truly
      // dense matrix, so a diagonal-only implementation cannot pass parity.
      mfem::DenseMatrix diagonal_mass(inverse_rows, 1);
      for (int row = 0; row < inverse_rows; ++row)
      {
        diagonal_mass(row, 0) = inverse_mass(row, row % dof);
      }
      mfem::DenseMatrix diagonal_strong;
      mfem::dgsem::ApplyInverseMassDevice(inverse_ne,
                                          dof,
                                          diagonal_mass,
                                          weak,
                                          diagonal_strong);
      max_inverse_mass_error =
        std::max(max_inverse_mass_error,
                 MaxDifference(diagonal_strong, device_strong));

      inverse_mass.HostReadWrite();
      for (int row = 0; row < inverse_rows; ++row)
      {
        for (int j = 0; j < dof; ++j)
        {
          if (j != row % dof)
          {
            inverse_mass(row, j) = 0.015 * (1 + row + j) / dof;
          }
        }
      }
      mfem::dgsem::ApplyInverseMassDevice(inverse_ne,
                                          dof,
                                          inverse_mass,
                                          weak,
                                          device_strong);
      for (int row = 0; row < inverse_rows; ++row)
      {
        for (int equation = 0; equation < num_equations; ++equation)
        {
          double value = 0.0;
          for (int j = 0; j < dof; ++j)
          {
            value += inverse_mass(row, j) *
                     weak((row / dof) * dof + j, equation);
          }
          host_strong(row, equation) = value;
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
      const int face_dof = dim == 1 ? face_n1d : face_n1d * face_n1d;
      const int face_points = dim == 1 ? 1 : face_n1d;
      const int nfaces = 3 * face_points;
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
      for (int q = 0; q < face_points; ++q)
      {
        FillConservativeState(left,
                              q,
                              dim,
                              1.0 + 0.02 * q,
                              0.2,
                              -0.05 + 0.01 * q,
                              1.0,
                              gamma);
        FillConservativeState(right,
                              q,
                              dim,
                              0.95 + 0.01 * q,
                              0.18,
                              -0.03,
                              0.98,
                              gamma);
        normals(q, 0) = 0.9;
        if (dim == 2)
        {
          normals(q, 1) = -0.35;
        }
        sigma_left(q) = 0.25;
        sigma_right(q) = 0.2;
        weights(q) = dim == 1 ? 1.0 : face_basis.weights(q);
        left_elements[q] = 0;
        left_local[q] = dim == 1 ? face_n1d - 1 : face_n1d - 1 + face_n1d * q;
        right_elements[q] = 1;
        right_local[q] = dim == 1 ? 0 : face_n1d * (face_n1d - 1 - q);
        ownership[q] =
          static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Local);

        const int shared = face_points + q;
        FillConservativeState(left,
                              shared,
                              dim,
                              1.02 + 0.01 * q,
                              0.16,
                              -0.04,
                              1.01,
                              gamma);
        FillConservativeState(right,
                              shared,
                              dim,
                              0.97 + 0.015 * q,
                              0.12,
                              -0.02,
                              0.99,
                              gamma);
        normals(shared, 0) = dim == 1 ? -1.2 : -0.2;
        if (dim == 2)
        {
          normals(shared, 1) = -1.2;
        }
        sigma_left(shared) = 0.25;
        sigma_right(shared) = 0.2;
        weights(shared) = dim == 1 ? 1.0 : face_basis.weights(q);
        left_elements[shared] = 0;
        left_local[shared] = q;
        right_elements[shared] = -1;
        right_local[shared] = dim == 1 ? face_n1d - 1 : face_n1d - 1 - q;
        ownership[shared] =
          static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Shared);

        const int boundary = 2 * face_points + q;
        FillConservativeState(left,
                              boundary,
                              dim,
                              1.01 + 0.01 * q,
                              0.14,
                              -0.06,
                              1.02,
                              gamma);
        FillConservativeState(right,
                              boundary,
                              dim,
                              0.99 + 0.005 * q,
                              -0.08,
                              0.03,
                              1.0,
                              gamma);
        normals(boundary, 0) = dim == 1 ? 0.7 : 0.6;
        if (dim == 2)
        {
          normals(boundary, 1) = 0.45;
        }
        sigma_left(boundary) = 0.25;
        sigma_right(boundary) = 0.2;
        weights(boundary) = dim == 1 ? 1.0 : face_basis.weights(q);
        left_elements[boundary] = 1;
        left_local[boundary] =
          dim == 1 ? face_n1d - 1 : face_n1d - 1 + face_n1d * q;
        right_elements[boundary] = -1;
        right_local[boundary] = -1;
        ownership[boundary] =
          static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Boundary);
      }
      mfem::DenseMatrix device_face_flux;
      mfem::Vector device_face_speeds;
      mfem::DenseMatrix device_surface;
      mfem::DenseMatrix host_face_flux;
      mfem::Vector host_face_speeds;
      mfem::DenseMatrix host_surface;
      mfem::Array<int> row_offsets;
      mfem::Array<int> row_face_ids;
      mfem::Array<int> row_face_signs;
      BuildRowFaceAdjacency(2,
                            face_dof,
                            left_elements,
                            left_local,
                            right_elements,
                            right_local,
                            ownership,
                            row_offsets,
                            row_face_ids,
                            row_face_signs);
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
                                                   row_offsets,
                                                   row_face_ids,
                                                   row_face_signs,
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
      for (int face = 0; face < nfaces; ++face)
      {
        if (ownership[face] !=
            static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Local))
        {
          for (int equation = 0; equation < num_equations; ++equation)
          {
            local_surface(left_elements[face] * face_dof + left_local[face],
                          equation) +=
              weights(face) * device_face_flux(face, equation);
          }
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
      mfem::dgsem::DeterministicMaximumDevice(device_face_speeds,
                                              max_face_speed);
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
  }

  const double max_metric_error = dimension_errors[1].metric;
  const double max_sigma_flux_error =
    std::max(dimension_errors[0].sigma_flux, dimension_errors[1].sigma_flux);
  const double max_volume_error =
    std::max(dimension_errors[0].volume, dimension_errors[1].volume);
  const double max_surface_error =
    std::max(dimension_errors[0].surface, dimension_errors[1].surface);
  const double max_inverse_mass_error =
    std::max(dimension_errors[0].inverse_mass,
             dimension_errors[1].inverse_mass);
  const double max_constant_residual =
    std::max(dimension_errors[0].constant_residual,
             dimension_errors[1].constant_residual);
  const double max_local_conservation =
    std::max(dimension_errors[0].local_conservation,
             dimension_errors[1].local_conservation);
  const double max_reduction_error =
    std::max(dimension_errors[0].reduction, dimension_errors[1].reduction);

  ctx.Check(max_basis_error <= mfem::dgsem::DGSEMReferenceTolerances::sbp,
            "DGSEM MFEM device p1-p4 basis parity uses frozen SBP gate");
  for (int dim = 1; dim <= 2; ++dim)
  {
    const auto& errors = dimension_errors[static_cast<std::size_t>(dim - 1)];
    const std::string dimension = std::to_string(dim) + "D";
    if (dim == 2)
    {
      ctx.Check(errors.metric <= metric_tol,
                "DGSEM MFEM device direct p1-p4 2D sheared metric parity");
    }
    ctx.Check(errors.sigma_flux <= pointwise_tol,
              "DGSEM MFEM device direct p1-p4 " + dimension +
                " frozen scalar-sigma tensor-flux parity");
    ctx.Check(errors.volume <= pointwise_tol,
              "DGSEM MFEM device direct p1-p4 " + dimension +
                " central volume parity");
    ctx.Check(errors.surface <= pointwise_tol,
              "DGSEM MFEM device direct p1-p4 " + dimension +
                " local/shared/boundary surface parity");
    ctx.Check(errors.inverse_mass <=
                mfem::dgsem::DGSEMReferenceTolerances::inverse_mass,
              "DGSEM MFEM device direct p1-p4 " + dimension +
                " inverse-mass parity");
    ctx.Check(errors.constant_residual <=
                mfem::dgsem::DGSEMReferenceTolerances::constant_affine,
              "DGSEM MFEM device direct p1-p4 " + dimension +
                " constant-state cancellation");
    ctx.Check(errors.local_conservation <=
                mfem::dgsem::DGSEMReferenceTolerances::conservation,
              "DGSEM MFEM device direct p1-p4 " + dimension +
                " component-wise local-face conservation");
    ctx.Check(errors.reduction <= pointwise_tol,
              "DGSEM MFEM device direct p1-p4 " + dimension +
                " deterministic speed reduction parity");
  }
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
        << "  \"schema\": \"dgsem-mfem-device-parity-v2\",\n"
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
        << "  \"dimensions\": [1,2],\n"
        << "  \"orders\": [1,2,3,4],\n"
        << "  \"direct_kernel_dimensions\": {\"basis\":[1,2],"
           "\"metric\":[2],\"euler_sigma_flux\":[1,2],"
           "\"volume\":[1,2],\"surface\":[1,2],"
           "\"inverse_mass\":[1,2],\"component_reduction\":[1,2],"
           "\"maximum_reduction\":[1,2],"
           "\"constant_cancellation\":[1,2]},\n"
        << "  \"face_ownership_dimensions\": {\"local\":[1,2],"
           "\"shared\":[1,2],\"boundary\":[1,2]},\n"
        << "  \"packed_nonzero_normal_preflight_dimensions\": [1,2],\n"
        << "  \"kernels_executed\": {\"basis\":true,\"metric\":true,"
           "\"euler_sigma_flux\":true,\"volume\":true,\"surface\":true,"
           "\"inverse_mass\":true,\"component_reduction\":true,"
           "\"maximum_reduction\":true},\n"
        << "  \"dimension_errors\": {\n"
        << "    \"1\": {\"max_sigma_flux_error\":"
        << dimension_errors[0].sigma_flux
        << ",\"max_volume_error\":" << dimension_errors[0].volume
        << ",\"max_surface_error\":" << dimension_errors[0].surface
        << ",\"max_inverse_mass_error\":" << dimension_errors[0].inverse_mass
        << ",\"max_constant_residual\":"
        << dimension_errors[0].constant_residual
        << ",\"max_local_conservation\":"
        << dimension_errors[0].local_conservation
        << ",\"max_reduction_error\":" << dimension_errors[0].reduction
        << "},\n"
        << "    \"2\": {\"max_metric_error\":" << dimension_errors[1].metric
        << ",\"max_sigma_flux_error\":" << dimension_errors[1].sigma_flux
        << ",\"max_volume_error\":" << dimension_errors[1].volume
        << ",\"max_surface_error\":" << dimension_errors[1].surface
        << ",\"max_inverse_mass_error\":" << dimension_errors[1].inverse_mass
        << ",\"max_constant_residual\":"
        << dimension_errors[1].constant_residual
        << ",\"max_local_conservation\":"
        << dimension_errors[1].local_conservation
        << ",\"max_reduction_error\":" << dimension_errors[1].reduction
        << "}\n  },\n"
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
