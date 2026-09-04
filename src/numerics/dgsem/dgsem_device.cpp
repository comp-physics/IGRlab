#include "numerics/dgsem/dgsem_device.hpp"
#include "physics/bc/detail/euler_boundary_state.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef MFEM_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

#if defined(MFEM_USE_CUDA) && defined(CUBLAS_VERSION) && \
  CUBLAS_VERSION >= 130002
#define DG_IGR_HAS_CUBLAS_FIXEDPOINT_FP64 1
#else
#define DG_IGR_HAS_CUBLAS_FIXEDPOINT_FP64 0
#endif

namespace mfem
{
namespace dgsem
{
namespace
{

MFEM_HOST_DEVICE inline real_t IntegerPower(real_t value, int exponent)
{
  real_t result = 1.0;
  for (int i = 0; i < exponent; ++i)
  {
    result *= value;
  }
  return result;
}

MFEM_HOST_DEVICE inline real_t
EulerPressure(const real_t* state, int stride, int dim, real_t gamma)
{
  const real_t rho = state[0];
  real_t momentum_squared = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t momentum = state[(1 + d) * stride];
    momentum_squared += momentum * momentum;
  }
  const real_t energy = state[(1 + dim) * stride];
  return (gamma - 1.0) * (energy - 0.5 * momentum_squared / rho);
}

MFEM_HOST_DEVICE inline real_t EulerTensorFluxEntry(const real_t* state,
                                                    int stride,
                                                    int dim,
                                                    int equation,
                                                    int physical_direction,
                                                    real_t gamma,
                                                    real_t sigma,
                                                    real_t rho_floor)
{
  const real_t rho = state[0];
  const real_t momentum_direction = state[(1 + physical_direction) * stride];
  const real_t pressure = EulerPressure(state, stride, dim, gamma);
  if (equation == 0)
  {
    return momentum_direction;
  }
  if (equation <= dim)
  {
    const real_t momentum_equation = state[equation * stride];
    const real_t diagonal = equation == 1 + physical_direction
                              ? pressure +
                                  dg_igr::frozen_scalar_igr_algebra::
                                    TensorSigmaMomentumDiagonalTerm(sigma)
                              : 0.0;
    return momentum_equation * momentum_direction / rho + diagonal;
  }
  const real_t energy = state[(1 + dim) * stride];
  return momentum_direction * (energy + pressure) / rho +
         dg_igr::frozen_scalar_igr_algebra::TensorSigmaEnergyTerm(
           sigma, momentum_direction, rho, rho_floor);
}

MFEM_HOST_DEVICE inline real_t EulerNormalFluxEntry(const real_t* state,
                                                    int stride,
                                                    const real_t* normal,
                                                    int normal_stride,
                                                    int dim,
                                                    int equation,
                                                    real_t gamma,
                                                    real_t sigma,
                                                    real_t rho_floor)
{
  const real_t rho = state[0];
  real_t momentum_dot_normal = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    momentum_dot_normal += state[(1 + d) * stride] * normal[d * normal_stride];
  }
  const real_t normal_velocity = momentum_dot_normal / rho;
  const real_t pressure = EulerPressure(state, stride, dim, gamma);
  if (equation == 0)
  {
    return momentum_dot_normal;
  }
  if (equation <= dim)
  {
    const int direction = equation - 1;
    return normal_velocity * state[equation * stride] +
           (pressure +
            dg_igr::frozen_scalar_igr_algebra::
              TensorSigmaMomentumDiagonalTerm(sigma)) *
             normal[direction * normal_stride];
  }
  return normal_velocity * (state[(1 + dim) * stride] + pressure) +
         dg_igr::frozen_scalar_igr_algebra::TensorSigmaEnergyTerm(
           sigma, momentum_dot_normal, rho, rho_floor);
}

MFEM_HOST_DEVICE inline real_t EulerSpeed(const real_t* state,
                                          int stride,
                                          const real_t* normal,
                                          int normal_stride,
                                          int dim,
                                          real_t gamma)
{
  const real_t rho = state[0];
  const real_t pressure = EulerPressure(state, stride, dim, gamma);
  real_t normal_squared = 0.0;
  real_t momentum_dot_normal = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t momentum = state[(1 + d) * stride];
    const real_t normal_value = normal[d * normal_stride];
    momentum_dot_normal += momentum * normal_value;
    normal_squared += normal_value * normal_value;
  }
  const real_t sound = sqrt(gamma * pressure / rho);
  return fabs(momentum_dot_normal / rho) / sqrt(normal_squared) + sound;
}

MFEM_HOST_DEVICE inline real_t EulerDirectionalLLFScale(
  const real_t* state,
  int stride,
  const real_t* direction,
  int dim,
  real_t gamma,
  real_t sigma)
{
  constexpr real_t floor = 1.0e-14;
  const real_t rho = fmax(state[0], floor);
  real_t momentum_squared = 0.0;
  real_t velocity_dot_direction = 0.0;
  real_t direction_squared = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t momentum = state[(1 + d) * stride];
    momentum_squared += momentum * momentum;
    velocity_dot_direction += (momentum / rho) * direction[d];
    direction_squared += direction[d] * direction[d];
  }
  const real_t energy = state[(1 + dim) * stride];
  const real_t kinetic = 0.5 * momentum_squared / rho;
  const real_t pressure =
    fmax((gamma - 1.0) * (energy - kinetic) + sigma, floor);
  const real_t sound = sqrt(fmax(gamma * pressure / rho, floor));
  const real_t direction_norm = sqrt(fmax(direction_squared, 0.0));
  const real_t speed =
    fabs(velocity_dot_direction) + sound * direction_norm;
  return isfinite(speed) ? speed : 0.0;
}

MFEM_HOST_DEVICE inline int PairOrdinal(int low, int high, int n1d)
{
  return low * (2 * n1d - low - 1) / 2 + high - low - 1;
}

} // namespace

DGSEMDeviceRuntimeInfo GetDGSEMDeviceRuntimeInfo()
{
  DGSEMDeviceRuntimeInfo info;
  info.configured = Device::IsConfigured();
  info.enabled = Device::IsEnabled();
  info.cuda = Device::Allows(Backend::CUDA_MASK);
  info.device_id = Device::GetId();
  info.device_count = Device::GetDeviceCount();
  info.host_memory_type = Device::GetHostMemoryType();
  info.device_memory_type = Device::GetDeviceMemoryType();
  info.host_memory_class = Device::GetHostMemoryClass();
  info.device_memory_class = Device::GetDeviceMemoryClass();
  return info;
}

void GatherElementStateDevice(int ne,
                              int dof,
                              int num_equations,
                              const Array<int>& element_vdofs,
                              const Vector& local_state,
                              DenseMatrix& packed_state)
{
  const int rows = ne * dof;
  MFEM_VERIFY(ne >= 0 && dof > 0 && num_equations > 0 &&
                element_vdofs.Size() == rows * num_equations,
              "DGSEM production element-state map size mismatch.");
  if (packed_state.Height() != rows ||
      packed_state.Width() != num_equations)
  {
    packed_state.SetSize(rows, num_equations);
  }
  packed_state.GetMemory().UseDevice(true);
  const int* map = element_vdofs.Read();
  const real_t* input = local_state.Read();
  real_t* output = packed_state.Write();
  mfem::forall(
    rows * num_equations,
    [=] MFEM_HOST_DEVICE(int entry)
    {
      const int vdof = map[entry];
      output[entry] = vdof >= 0 ? input[vdof] : -input[-1 - vdof];
    });
}

void InterpolateElementScalarDevice(int rows,
                                    int scalar_dofs_per_element,
                                    const Array<int>& scalar_element_dofs,
                                    const DenseMatrix& interpolation,
                                    const Vector& scalar_field,
                                    Vector& packed_scalar)
{
  MFEM_VERIFY(rows >= 0 && scalar_dofs_per_element > 0 &&
                scalar_element_dofs.Size() ==
                  rows * scalar_dofs_per_element &&
                interpolation.Height() == rows &&
                interpolation.Width() == scalar_dofs_per_element,
              "DGSEM production sigma interpolation size mismatch.");
  if (packed_scalar.Size() != rows)
  {
    packed_scalar.SetSize(rows);
  }
  packed_scalar.UseDevice(true);
  const int* map = scalar_element_dofs.Read();
  const real_t* shape = interpolation.Read();
  const real_t* field = scalar_field.Read();
  real_t* output = packed_scalar.Write();
  mfem::forall(
    rows,
    [=] MFEM_HOST_DEVICE(int row)
    {
      real_t value = 0.0;
      for (int j = 0; j < scalar_dofs_per_element; ++j)
      {
        const int entry = row + rows * j;
        const int dof_index = map[entry];
        const real_t dof_value =
          dof_index >= 0 ? field[dof_index] : -field[-1 - dof_index];
        value += shape[entry] * dof_value;
      }
      output[row] = value;
    });
}

void InterpolateSharedFaceScalarDevice(
  int nfaces,
  int scalar_dofs_per_element,
  const Array<int>& scalar_face_neighbor_dofs,
  const DenseMatrix& interpolation,
  const Vector& scalar_face_neighbor_data,
  Vector& packed_shared_scalar)
{
  MFEM_VERIFY(nfaces >= 0 && scalar_dofs_per_element > 0 &&
                scalar_face_neighbor_dofs.Size() ==
                  nfaces * scalar_dofs_per_element &&
                interpolation.Height() == nfaces &&
                interpolation.Width() == scalar_dofs_per_element,
              "DGSEM production shared sigma interpolation size mismatch.");
  if (packed_shared_scalar.Size() != nfaces)
  {
    packed_shared_scalar.SetSize(nfaces);
  }
  packed_shared_scalar.UseDevice(true);
  const int* map = scalar_face_neighbor_dofs.Read();
  const real_t* shape = interpolation.Read();
  const real_t* field = scalar_face_neighbor_data.Read();
  real_t* output = packed_shared_scalar.Write();
  mfem::forall(
    nfaces,
    [=] MFEM_HOST_DEVICE(int face)
    {
      real_t value = 0.0;
      for (int j = 0; j < scalar_dofs_per_element; ++j)
      {
        const int entry = face + nfaces * j;
        const real_t coefficient = shape[entry];
        if (coefficient != 0.0)
        {
          const int dof_index = map[entry];
          const real_t dof_value =
            dof_index >= 0 ? field[dof_index] : -field[-1 - dof_index];
          value += coefficient * dof_value;
        }
      }
      output[face] = value;
    });
}

void GatherEulerRusanovFacesDevice(
  int dim,
  int ne,
  int dof,
  real_t gamma,
  const DenseMatrix& packed_states,
  const Vector& packed_sigma,
  const Vector& face_neighbor_state,
  const Array<int>& shared_state_vdofs,
  const Vector& shared_sigma,
  const DenseMatrix& normals,
  const Array<int>& left_elements,
  const Array<int>& left_local_dofs,
  const Array<int>& right_elements,
  const Array<int>& right_local_dofs,
  const Array<int>& ownership,
  const Array<int>& boundary_kinds,
  const DenseMatrix& boundary_reference_states,
  const Vector& boundary_density_floors,
  DenseMatrix& left_states,
  DenseMatrix& right_states,
  Vector& left_sigma,
  Vector& right_sigma)
{
  const int nfaces = normals.Height();
  const int num_equations = dim + 2;
  const int rows = ne * dof;
  MFEM_VERIFY((dim == 1 || dim == 2) && ne >= 0 && dof > 0 &&
                packed_states.Height() == rows &&
                packed_states.Width() == num_equations &&
                packed_sigma.Size() == rows && normals.Width() == dim &&
                left_elements.Size() == nfaces &&
                left_local_dofs.Size() == nfaces &&
                right_elements.Size() == nfaces &&
                right_local_dofs.Size() == nfaces &&
                ownership.Size() == nfaces &&
                boundary_kinds.Size() == nfaces &&
                boundary_reference_states.Height() == nfaces &&
                boundary_reference_states.Width() == num_equations &&
                boundary_density_floors.Size() == nfaces &&
                shared_state_vdofs.Size() == nfaces * num_equations &&
                shared_sigma.Size() == nfaces,
              "DGSEM production face gather size mismatch.");

  if (left_states.Height() != nfaces ||
      left_states.Width() != num_equations)
  {
    left_states.SetSize(nfaces, num_equations);
  }
  if (right_states.Height() != nfaces ||
      right_states.Width() != num_equations)
  {
    right_states.SetSize(nfaces, num_equations);
  }
  if (left_sigma.Size() != nfaces)
  {
    left_sigma.SetSize(nfaces);
  }
  if (right_sigma.Size() != nfaces)
  {
    right_sigma.SetSize(nfaces);
  }
  left_states.GetMemory().UseDevice(true);
  right_states.GetMemory().UseDevice(true);
  left_sigma.UseDevice(true);
  right_sigma.UseDevice(true);

  const real_t* states = packed_states.Read();
  const real_t* sigma = packed_sigma.Read();
  const real_t* neighbor = face_neighbor_state.Read();
  const int* neighbor_map = shared_state_vdofs.Read();
  const real_t* neighbor_sigma = shared_sigma.Read();
  const real_t* normal = normals.Read();
  const int* element_l = left_elements.Read();
  const int* local_l = left_local_dofs.Read();
  const int* element_r = right_elements.Read();
  const int* local_r = right_local_dofs.Read();
  const int* owner = ownership.Read();
  const int* boundary_kind = boundary_kinds.Read();
  const real_t* boundary_reference = boundary_reference_states.Read();
  const real_t* boundary_density_floor = boundary_density_floors.Read();
  real_t* left = left_states.Write();
  real_t* right = right_states.Write();
  real_t* sigma_l = left_sigma.Write();
  real_t* sigma_r = right_sigma.Write();

  mfem::forall(
    nfaces,
    [=] MFEM_HOST_DEVICE(int face)
    {
      const int left_row = element_l[face] * dof + local_l[face];
      for (int equation = 0; equation < num_equations; ++equation)
      {
        left[face + nfaces * equation] =
          states[left_row + rows * equation];
      }
      sigma_l[face] = sigma[left_row];

      if (owner[face] ==
          static_cast<int>(DGSEMDeviceFaceOwnership::Local))
      {
        const int right_row = element_r[face] * dof + local_r[face];
        for (int equation = 0; equation < num_equations; ++equation)
        {
          right[face + nfaces * equation] =
            states[right_row + rows * equation];
        }
        sigma_r[face] = sigma[right_row];
      }
      else if (owner[face] ==
               static_cast<int>(DGSEMDeviceFaceOwnership::Shared))
      {
        for (int equation = 0; equation < num_equations; ++equation)
        {
          const int vdof = neighbor_map[face + nfaces * equation];
          right[face + nfaces * equation] =
            vdof >= 0 ? neighbor[vdof] : -neighbor[-1 - vdof];
        }
        sigma_r[face] = neighbor_sigma[face];
      }
      else
      {
        real_t interior[4] = {0.0, 0.0, 0.0, 0.0};
        real_t exterior[4] = {0.0, 0.0, 0.0, 0.0};
        real_t reference[4] = {0.0, 0.0, 0.0, 0.0};
        real_t scaled_normal[2] = {0.0, 0.0};
        for (int equation = 0; equation < num_equations; ++equation)
        {
          interior[equation] = left[face + nfaces * equation];
          exterior[equation] = interior[equation];
          reference[equation] =
            boundary_reference[face + nfaces * equation];
        }
        for (int d = 0; d < dim; ++d)
        {
          scaled_normal[d] = normal[face + nfaces * d];
        }
        const bool boundary_ok = detail::EvaluateEulerBoundaryState(
          static_cast<detail::EulerBoundaryStateKind>(boundary_kind[face]),
          dim,
          num_equations,
          gamma,
          0.0,
          reference,
          scaled_normal,
          interior,
          exterior,
          boundary_density_floor[face],
          1.0e-12);
        if (!boundary_ok)
        {
          exterior[0] = -1.0;
        }
        for (int equation = 0; equation < num_equations; ++equation)
        {
          right[face + nfaces * equation] = exterior[equation];
        }
        sigma_r[face] = sigma_l[face];
      }
    });
}

void AddPackedMatricesDevice(const DenseMatrix& addend, DenseMatrix& inout)
{
  MFEM_VERIFY(addend.Height() == inout.Height() &&
                addend.Width() == inout.Width(),
              "DGSEM production packed add size mismatch.");
  const int size = addend.Height() * addend.Width();
  const real_t* input = addend.Read();
  real_t* output = inout.ReadWrite();
  mfem::forall(size,
               [=] MFEM_HOST_DEVICE(int entry)
               { output[entry] += input[entry]; });
}

void ScatterElementStateDevice(int ne,
                               int dof,
                               int num_equations,
                               const Array<int>& element_vdofs,
                               const DenseMatrix& packed_state,
                               Vector& local_state)
{
  const int rows = ne * dof;
  MFEM_VERIFY(ne >= 0 && dof > 0 && num_equations > 0 &&
                element_vdofs.Size() == rows * num_equations &&
                packed_state.Height() == rows &&
                packed_state.Width() == num_equations,
              "DGSEM production element scatter size mismatch.");
  const int* map = element_vdofs.Read();
  const real_t* input = packed_state.Read();
  real_t* output = local_state.Write();
  mfem::forall(
    rows * num_equations,
    [=] MFEM_HOST_DEVICE(int entry)
    {
      const int vdof = map[entry];
      if (vdof >= 0)
      {
        output[vdof] = input[entry];
      }
      else
      {
        output[-1 - vdof] = -input[entry];
      }
    });
}

void EvaluateLGLBasisDevice(const Vector& nodes,
                            const Vector& weights,
                            const DenseMatrix& derivative_matrix,
                            const DenseMatrix& split_matrix,
                            int order,
                            Vector& moments,
                            DenseMatrix& derivative_action,
                            DenseMatrix& sbp_residual,
                            DenseMatrix& split_residual)
{
  const int n = order + 1;
  MFEM_VERIFY(nodes.Size() == n && weights.Size() == n,
              "DGSEM device LGL vector size mismatch.");
  MFEM_VERIFY(derivative_matrix.Height() == n &&
                derivative_matrix.Width() == n && split_matrix.Height() == n &&
                split_matrix.Width() == n,
              "DGSEM device LGL matrix size mismatch.");

  moments.SetSize(2 * order);
  derivative_action.SetSize(n, order + 1);
  sbp_residual.SetSize(n);
  split_residual.SetSize(n);
  moments.UseDevice(true);
  derivative_action.GetMemory().UseDevice(true);
  sbp_residual.GetMemory().UseDevice(true);
  split_residual.GetMemory().UseDevice(true);

  const real_t* node = nodes.Read();
  const real_t* weight = weights.Read();
  const real_t* derivative = derivative_matrix.Read();
  const real_t* split = split_matrix.Read();
  real_t* moment = moments.Write();
  real_t* derivative_result = derivative_action.Write();
  real_t* sbp = sbp_residual.Write();
  real_t* split_check = split_residual.Write();

  mfem::forall(2 * order,
               [=] MFEM_HOST_DEVICE(int degree)
               {
                 real_t value = 0.0;
                 for (int i = 0; i < n; ++i)
                 {
                   value += weight[i] * IntegerPower(node[i], degree);
                 }
                 moment[degree] = value;
               });

  mfem::forall(n * (order + 1),
               [=] MFEM_HOST_DEVICE(int entry)
               {
                 const int i = entry % n;
                 const int degree = entry / n;
                 real_t value = 0.0;
                 for (int j = 0; j < n; ++j)
                 {
                   value +=
                     derivative[i + n * j] * IntegerPower(node[j], degree);
                 }
                 derivative_result[i + n * degree] = value;
               });

  mfem::forall(n * n,
               [=] MFEM_HOST_DEVICE(int entry)
               {
                 const int i = entry % n;
                 const int j = entry / n;
                 real_t boundary = 0.0;
                 if (i == j && i == 0)
                 {
                   boundary = -1.0;
                 }
                 else if (i == j && i == n - 1)
                 {
                   boundary = 1.0;
                 }
                 sbp[entry] = weight[i] * derivative[i + n * j] +
                              weight[j] * derivative[j + n * i] - boundary;

                 real_t diagonal_boundary = 0.0;
                 if (i == j && i == 0)
                 {
                   diagonal_boundary = -1.0 / weight[i];
                 }
                 else if (i == j && i == n - 1)
                 {
                   diagonal_boundary = 1.0 / weight[i];
                 }
                 split_check[entry] =
                   split[entry] - (2.0 * derivative[entry] - diagonal_boundary);
               });
}

void BuildContravariantMetrics2DDevice(int ne,
                                       int n1d,
                                       const Array<int>& lex_to_local,
                                       const DenseMatrix& derivative_matrix,
                                       const DenseMatrix& coordinates,
                                       DenseMatrix& metrics)
{
  const int dof = n1d * n1d;
  const int rows = ne * dof;
  MFEM_VERIFY(ne >= 0 && n1d >= 2,
              "DGSEM device metric batch has invalid dimensions.");
  MFEM_VERIFY(lex_to_local.Size() == dof,
              "DGSEM device metric tensor map size mismatch.");
  MFEM_VERIFY(derivative_matrix.Height() == n1d &&
                derivative_matrix.Width() == n1d,
              "DGSEM device metric derivative size mismatch.");
  MFEM_VERIFY(coordinates.Height() == rows && coordinates.Width() == 2,
              "DGSEM device metric coordinate size mismatch.");

  metrics.SetSize(rows, 4);
  metrics.GetMemory().UseDevice(true);
  const int* local = lex_to_local.Read();
  const real_t* derivative = derivative_matrix.Read();
  const real_t* coordinate = coordinates.Read();
  real_t* metric = metrics.Write();
  const auto coordinate_view = Reshape(coordinate, rows, 2);
  auto metric_view = Reshape(metric, rows, 4);

  mfem::forall(
    rows,
    [=] MFEM_HOST_DEVICE(int row)
    {
      const int element = row / dof;
      const int element_local = row - element * dof;
      int lex = 0;
      for (; lex < dof; ++lex)
      {
        if (local[lex] == element_local)
        {
          break;
        }
      }
      const int i = lex % n1d;
      const int j = lex / n1d;
      real_t dx_dxi = 0.0;
      real_t dy_dxi = 0.0;
      real_t dx_deta = 0.0;
      real_t dy_deta = 0.0;
      for (int m = 0; m < n1d; ++m)
      {
        const int row_xi = element * dof + local[m + n1d * j];
        const int row_eta = element * dof + local[i + n1d * m];
        dx_dxi += derivative[i + n1d * m] * coordinate_view(row_xi, 0);
        dy_dxi += derivative[i + n1d * m] * coordinate_view(row_xi, 1);
        dx_deta += derivative[j + n1d * m] * coordinate_view(row_eta, 0);
        dy_deta += derivative[j + n1d * m] * coordinate_view(row_eta, 1);
      }
      metric_view(row, 0) = dy_deta;
      metric_view(row, 1) = -dx_deta;
      metric_view(row, 2) = -dy_dxi;
      metric_view(row, 3) = dx_dxi;
    });
}

void BuildEulerContravariantFluxDevice(int dim,
                                       real_t gamma,
                                       real_t velocity_density_floor,
                                       const DenseMatrix& states,
                                       const Vector& sigma,
                                       const DenseMatrix& metrics,
                                       DenseMatrix& contravariant_flux,
                                       Vector& nodal_speeds)
{
  const int rows = states.Height();
  const int num_equations = dim + 2;
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM device Euler flux supports dimensions 1 and 2 only.");
  MFEM_VERIFY(states.Width() == num_equations && sigma.Size() == rows,
              "DGSEM device Euler state/sigma size mismatch.");
  MFEM_VERIFY(metrics.Height() == rows && metrics.Width() == dim * dim,
              "DGSEM device Euler metric size mismatch.");

  if (contravariant_flux.Height() != rows ||
      contravariant_flux.Width() != dim * num_equations)
  {
    contravariant_flux.SetSize(rows, dim * num_equations);
  }
  if (nodal_speeds.Size() != rows)
  {
    nodal_speeds.SetSize(rows);
  }
  contravariant_flux.GetMemory().UseDevice(true);
  nodal_speeds.UseDevice(true);
  const real_t* state = states.Read();
  const real_t* sigma_value = sigma.Read();
  const real_t* metric = metrics.Read();
  real_t* flux = contravariant_flux.Write();
  real_t* speed = nodal_speeds.Write();

  mfem::forall(
    rows * dim * num_equations,
    [=] MFEM_HOST_DEVICE(int entry)
    {
      const int row = entry % rows;
      const int column = entry / rows;
      const int equation = column % num_equations;
      const int reference_direction = column / num_equations;
      real_t value = 0.0;
      for (int physical = 0; physical < dim; ++physical)
      {
        value += metric[row + rows * (reference_direction * dim + physical)] *
                 EulerTensorFluxEntry(state + row,
                                      rows,
                                      dim,
                                      equation,
                                      physical,
                                      gamma,
                                      sigma_value[row],
                                      velocity_density_floor);
      }
      flux[entry] = value;
    });

  mfem::forall(rows,
               [=] MFEM_HOST_DEVICE(int row)
               {
                 const real_t rho = state[row];
                 const real_t pressure =
                   EulerPressure(state + row, rows, dim, gamma);
                 real_t momentum_squared = 0.0;
                 for (int d = 0; d < dim; ++d)
                 {
                   const real_t momentum = state[row + rows * (1 + d)];
                   momentum_squared += momentum * momentum;
                 }
                 speed[row] =
                   sqrt(momentum_squared) / rho + sqrt(gamma * pressure / rho);
               });
}

void BuildEulerContravariantFluxPanelDevice(
  int dim,
  int ne,
  int dof,
  real_t gamma,
  real_t velocity_density_floor,
  const DenseMatrix& states,
  const Vector& sigma,
  const DenseMatrix& metrics,
  DenseMatrix& contravariant_flux_panel,
  Vector& nodal_speeds)
{
  const int rows = ne * dof;
  const int num_equations = dim + 2;
  const int panel_rows = dim * dof;
  const int panel_columns = ne * num_equations;
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM device Euler flux panel supports dimensions 1 and 2.");
  MFEM_VERIFY(ne > 0 && dof > 0 && states.Height() == rows &&
                states.Width() == num_equations && sigma.Size() == rows,
              "DGSEM device Euler flux panel state/sigma size mismatch.");
  MFEM_VERIFY(metrics.Height() == rows && metrics.Width() == dim * dim,
              "DGSEM device Euler flux panel metric size mismatch.");

  if (contravariant_flux_panel.Height() != panel_rows ||
      contravariant_flux_panel.Width() != panel_columns)
  {
    contravariant_flux_panel.SetSize(panel_rows, panel_columns);
  }
  if (nodal_speeds.Size() != rows)
  {
    nodal_speeds.SetSize(rows);
  }
  contravariant_flux_panel.GetMemory().UseDevice(true);
  nodal_speeds.UseDevice(true);
  const real_t* state = states.Read();
  const real_t* sigma_value = sigma.Read();
  const real_t* metric = metrics.Read();
  real_t* flux = contravariant_flux_panel.Write();
  real_t* speed = nodal_speeds.Write();

  mfem::forall(
    panel_rows * panel_columns,
    [=] MFEM_HOST_DEVICE(int entry)
    {
      const int panel_row = entry % panel_rows;
      const int panel_column = entry / panel_rows;
      const int local = panel_row % dof;
      const int reference_direction = panel_row / dof;
      const int element = panel_column % ne;
      const int equation = panel_column / ne;
      const int row = element * dof + local;
      real_t value = 0.0;
      for (int physical = 0; physical < dim; ++physical)
      {
        value += metric[row + rows * (reference_direction * dim + physical)] *
                 EulerTensorFluxEntry(state + row,
                                      rows,
                                      dim,
                                      equation,
                                      physical,
                                      gamma,
                                      sigma_value[row],
                                      velocity_density_floor);
      }
      flux[entry] = value;
    });

  mfem::forall(rows,
               [=] MFEM_HOST_DEVICE(int row)
               {
                 const real_t rho = state[row];
                 const real_t pressure =
                   EulerPressure(state + row, rows, dim, gamma);
                 real_t momentum_squared = 0.0;
                 for (int d = 0; d < dim; ++d)
                 {
                   const real_t momentum = state[row + rows * (1 + d)];
                   momentum_squared += momentum * momentum;
                 }
                 speed[row] =
                   sqrt(momentum_squared) / rho + sqrt(gamma * pressure / rho);
               });
}

void ApplyCentralVolumeDevice(int dim,
                              int n1d,
                              const Array<int>& lex_to_local,
                              const DenseMatrix& split_matrix,
                              const Vector& weights,
                              const DenseMatrix& contravariant_flux,
                              DenseMatrix& weak_volume)
{
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM device central volume supports dimensions 1 and 2.");
  const int dof = dim == 1 ? n1d : n1d * n1d;
  const int num_equations = contravariant_flux.Width() / dim;
  MFEM_VERIFY(n1d >= 2 && lex_to_local.Size() == dof,
              "DGSEM device central volume tensor map mismatch.");
  MFEM_VERIFY(split_matrix.Height() == n1d && split_matrix.Width() == n1d &&
                weights.Size() == n1d,
              "DGSEM device central volume basis size mismatch.");
  MFEM_VERIFY(num_equations > 0 &&
                contravariant_flux.Width() == dim * num_equations &&
                contravariant_flux.Height() % dof == 0,
              "DGSEM device central volume flux size mismatch.");

  const int rows = contravariant_flux.Height();
  if (weak_volume.Height() != rows || weak_volume.Width() != num_equations)
  {
    weak_volume.SetSize(rows, num_equations);
  }
  weak_volume.GetMemory().UseDevice(true);
  const int* local = lex_to_local.Read();
  const real_t* split = split_matrix.Read();
  const real_t* weight = weights.Read();
  const real_t* flux = contravariant_flux.Read();
  real_t* volume = weak_volume.Write();

  mfem::forall(rows * num_equations,
               [=] MFEM_HOST_DEVICE(int entry)
               {
                 const int row = entry % rows;
                 const int equation = entry / rows;
                 const int element = row / dof;
                 const int element_local = row - element * dof;
                 int lex = 0;
                 for (; lex < dof; ++lex)
                 {
                   if (local[lex] == element_local)
                   {
                     break;
                   }
                 }
                 const int i = lex % n1d;
                 const int j = dim == 2 ? lex / n1d : 0;
                 real_t value = 0.0;
                 for (int m = 0; m < n1d; ++m)
                 {
                   if (m != i)
                   {
                     const int other_local = local[m + n1d * j];
                     const int other_row = element * dof + other_local;
                     const int flux_column = equation;
                     const real_t pair =
                       0.5 * (flux[row + rows * flux_column] +
                              flux[other_row + rows * flux_column]);
                     value -= split[i + n1d * m] * pair;
                   }
                 }
                 if (dim == 2)
                 {
                   for (int m = 0; m < n1d; ++m)
                   {
                     if (m != j)
                     {
                       const int other_local = local[i + n1d * m];
                       const int other_row = element * dof + other_local;
                       const int flux_column = num_equations + equation;
                       const real_t pair =
                         0.5 * (flux[row + rows * flux_column] +
                                flux[other_row + rows * flux_column]);
                       value -= split[j + n1d * m] * pair;
                     }
                   }
                 }
                 real_t tensor_weight = weight[i];
                 if (dim == 2)
                 {
                   tensor_weight *= weight[j];
                 }
                 volume[entry] = tensor_weight * value;
               });
}

void BuildCentralVolumePanelOperator(int dim,
                                     int n1d,
                                     const Array<int>& lex_to_local,
                                     const DenseMatrix& split_matrix,
                                     const Vector& weights,
                                     DenseTensor& panel_operator)
{
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM central volume panel supports dimensions 1 and 2.");
  const int dof = dim == 1 ? n1d : n1d * n1d;
  MFEM_VERIFY(n1d >= 2 && lex_to_local.Size() == dof,
              "DGSEM central volume panel tensor map mismatch.");
  MFEM_VERIFY(split_matrix.Height() == n1d &&
                split_matrix.Width() == n1d && weights.Size() == n1d,
              "DGSEM central volume panel basis size mismatch.");

  panel_operator.SetSize(dof, dim * dof, 1);
  real_t* panel = panel_operator.HostWrite();
  std::fill(panel, panel + panel_operator.TotalSize(), 0.0);
  const int* local = lex_to_local.HostRead();
  const real_t* split = split_matrix.HostRead();
  const real_t* weight = weights.HostRead();
  for (int lex = 0; lex < dof; ++lex)
  {
    const int i = lex % n1d;
    const int j = dim == 2 ? lex / n1d : 0;
    const int local_i = local[lex];
    real_t tensor_weight = weight[i];
    if (dim == 2)
    {
      tensor_weight *= weight[j];
    }
    for (int direction = 0; direction < dim; ++direction)
    {
      const int coordinate = direction == 0 ? i : j;
      real_t row_sum = 0.0;
      for (int other = 0; other < n1d; ++other)
      {
        if (other == coordinate)
        {
          continue;
        }
        const int other_lex = direction == 0
                                ? other + n1d * j
                                : i + n1d * other;
        const int local_other = local[other_lex];
        const real_t derivative = split[coordinate + n1d * other];
        panel[local_i + dof * (direction * dof + local_other)] +=
          -0.5 * tensor_weight * derivative;
        row_sum += derivative;
      }
      panel[local_i + dof * (direction * dof + local_i)] +=
        -0.5 * tensor_weight * row_sum;
    }
  }
}

void ApplyCentralVolumePanelFP64Device(
  int dim,
  int ne,
  int dof,
  int num_equations,
  const DenseTensor& panel_operator,
  const DenseMatrix& contravariant_flux_panel,
  DenseMatrix& weak_volume)
{
  MFEM_VERIFY(sizeof(real_t) == sizeof(double),
              "DGSEM FP64 volume panel requires a double-precision MFEM.");
  MFEM_VERIFY(Device::Allows(Backend::CUDA_MASK),
              "DGSEM FP64 volume panel requires the CUDA backend.");
  MFEM_VERIFY(BatchedLinAlg::IsAvailable(BatchedLinAlg::GPU_BLAS),
              "DGSEM FP64 volume panel requires MFEM GPU BLAS.");
  MFEM_VERIFY((dim == 1 || dim == 2) && ne > 0 && dof > 0 &&
                num_equations == dim + 2,
              "DGSEM FP64 volume panel dimensions are invalid.");
  MFEM_VERIFY(panel_operator.SizeI() == dof &&
                panel_operator.SizeJ() == dim * dof &&
                panel_operator.SizeK() == 1,
              "DGSEM FP64 volume panel operator shape mismatch.");
  MFEM_VERIFY(contravariant_flux_panel.Height() == dim * dof &&
                contravariant_flux_panel.Width() == ne * num_equations,
              "DGSEM FP64 volume panel flux shape mismatch.");
  if (weak_volume.Height() != ne * dof ||
      weak_volume.Width() != num_equations)
  {
    weak_volume.SetSize(ne * dof, num_equations);
  }
  weak_volume.GetMemory().UseDevice(true);

#ifdef MFEM_USE_CUDA
  // Write the owning matrix directly.  A temporary MFEM Vector alias has its
  // own validity flags; writing only that alias can leave the owning
  // DenseMatrix's host view marked current and make an immediately following
  // FP64 audit observe stale zeros.  This is the same column-major GEMM as
  // GPUBlasBatchedLinAlg, without the alias-validity ambiguity.
  const double alpha = 1.0;
  const double beta = 0.0;
  const cublasStatus_t status = cublasGemmEx(
    GPUBlas::Handle(),
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    dof,
    ne * num_equations,
    dim * dof,
    &alpha,
    panel_operator.Read(true),
    CUDA_R_64F,
    dof,
    contravariant_flux_panel.Read(true),
    CUDA_R_64F,
    dim * dof,
    &beta,
    weak_volume.Write(true),
    CUDA_R_64F,
    dof,
    CUBLAS_COMPUTE_64F,
    CUBLAS_GEMM_DEFAULT);
  MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS,
              "DGSEM native FP64 volume panel cuBLAS failure.");
#else
  MFEM_ABORT("DGSEM FP64 volume panel was built without CUDA support.");
#endif
}

#if DG_IGR_HAS_CUBLAS_FIXEDPOINT_FP64
namespace
{

std::size_t FixedPointWorkspaceBytes(int m,
                                     int n,
                                     int k,
                                     int max_mantissa_bits)
{
  // CUDA 13.3 cuBLAS fixed-point emulation workspace upper bound for one
  // real GEMM.  Dynamic mantissa control is intentional: it is the mode that
  // promises native-FP64-or-better accuracy and may fall back to native FP64.
  const long double slices =
    static_cast<long double>((max_mantissa_bits + 8) / 8);
  const long double padded_m =
    static_cast<long double>(((m + 1023) / 1024) * 1024);
  const long double padded_n =
    static_cast<long double>(((n + 1023) / 1024) * 1024);
  const long double padded_k =
    static_cast<long double>(((k + 127) / 128) * 128);
  const long double k_blocks = static_cast<long double>((k + 63) / 64);
  const long double gemm =
    (padded_m * padded_k + padded_n * padded_k) * slices +
    sizeof(std::int32_t) * (padded_m + padded_n);
  const long double adaptive = sizeof(std::int32_t) *
    (static_cast<long double>(m) * k_blocks +
     static_cast<long double>(n) * k_blocks +
     static_cast<long double>(m) * n);
  const long double required =
    1.25L * (gemm > adaptive ? gemm : adaptive) +
    128.0L * 1024.0L * 1024.0L;
  if (!(required > 0.0L) ||
      required > static_cast<long double>(
                   std::numeric_limits<std::size_t>::max() - 255))
  {
    return 0;
  }
  const std::size_t rounded = static_cast<std::size_t>(required + 255.0L);
  return rounded & ~std::size_t(255);
}

} // namespace
#endif

struct DGSEMFixedPointVolumeContext::Implementation
{
  DGSEMFixedPointPanelStatus status;
#if DG_IGR_HAS_CUBLAS_FIXEDPOINT_FP64
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  std::int32_t* retained_mantissa_bits = nullptr;

  void Release()
  {
    if (handle)
    {
      cublasDestroy(handle);
      handle = nullptr;
    }
    if (retained_mantissa_bits)
    {
      cudaFree(retained_mantissa_bits);
      retained_mantissa_bits = nullptr;
    }
    if (workspace)
    {
      cudaFree(workspace);
      workspace = nullptr;
    }
  }
#else
  void Release() {}
#endif
};

DGSEMFixedPointVolumeContext::DGSEMFixedPointVolumeContext()
  : implementation_(std::make_unique<Implementation>())
{
#if DG_IGR_HAS_CUBLAS_FIXEDPOINT_FP64
  implementation_->status.compiled = true;
#endif
}

DGSEMFixedPointVolumeContext::~DGSEMFixedPointVolumeContext()
{
  implementation_->Release();
}

DGSEMFixedPointPanelStatus DGSEMFixedPointVolumeContext::Initialize(
  int m,
  int n,
  int k,
  int max_mantissa_bits,
  std::size_t workspace_limit_bytes)
{
  implementation_->Release();
  DGSEMFixedPointPanelStatus& status = implementation_->status;
  status = DGSEMFixedPointPanelStatus{};
#if DG_IGR_HAS_CUBLAS_FIXEDPOINT_FP64
  status.compiled = true;
  status.m = m;
  status.n = n;
  status.k = k;
  status.max_mantissa_bits = max_mantissa_bits;
  if (m <= 0 || n <= 0 || k <= 0 || max_mantissa_bits <= 0 ||
      max_mantissa_bits > 79)
  {
    status.cublas_status = static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
    return status;
  }

  int device = 0;
  cudaDeviceProp property{};
  cudaError_t cuda_result = cudaGetDevice(&device);
  if (cuda_result == cudaSuccess)
  {
    cuda_result = cudaGetDeviceProperties(&property, device);
  }
  status.cuda_status = static_cast<int>(cuda_result);
  if (cuda_result != cudaSuccess)
  {
    return status;
  }
  status.compute_capability_major = property.major;
  status.compute_capability_minor = property.minor;
  if (property.major < 8 || property.major > 12)
  {
    status.cuda_status = static_cast<int>(cudaErrorNotSupported);
    return status;
  }

  status.workspace_bytes =
    FixedPointWorkspaceBytes(m, n, k, max_mantissa_bits);
  if (status.workspace_bytes == 0 ||
      status.workspace_bytes > workspace_limit_bytes)
  {
    status.cuda_status = static_cast<int>(cudaErrorMemoryAllocation);
    return status;
  }

  cuda_result = cudaMalloc(&implementation_->workspace,
                           status.workspace_bytes);
  if (cuda_result == cudaSuccess)
  {
    cuda_result = cudaMalloc(
      reinterpret_cast<void**>(&implementation_->retained_mantissa_bits),
      sizeof(std::int32_t));
  }
  status.cuda_status = static_cast<int>(cuda_result);
  if (cuda_result != cudaSuccess)
  {
    implementation_->Release();
    return status;
  }
  cuda_result = cudaMemset(
    implementation_->retained_mantissa_bits,
    0xff,
    sizeof(std::int32_t));
  status.cuda_status = static_cast<int>(cuda_result);
  if (cuda_result != cudaSuccess)
  {
    implementation_->Release();
    return status;
  }

  cublasStatus_t cublas_result = cublasCreate(&implementation_->handle);
  if (cublas_result == CUBLAS_STATUS_SUCCESS)
  {
    cublas_result = cublasSetPointerMode(
      implementation_->handle, CUBLAS_POINTER_MODE_HOST);
  }
  if (cublas_result == CUBLAS_STATUS_SUCCESS)
  {
    cublas_result = cublasSetWorkspace(implementation_->handle,
                                       implementation_->workspace,
                                       status.workspace_bytes);
  }
  if (cublas_result == CUBLAS_STATUS_SUCCESS)
  {
    cublas_result = cublasSetEmulationStrategy(
      implementation_->handle, CUBLAS_EMULATION_STRATEGY_EAGER);
  }
  if (cublas_result == CUBLAS_STATUS_SUCCESS)
  {
    cublas_result = cublasSetFixedPointEmulationMantissaControl(
      implementation_->handle,
      CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);
  }
  if (cublas_result == CUBLAS_STATUS_SUCCESS)
  {
    cublas_result = cublasSetFixedPointEmulationMaxMantissaBitCount(
      implementation_->handle, max_mantissa_bits);
  }
  if (cublas_result == CUBLAS_STATUS_SUCCESS)
  {
    cublas_result = cublasSetFixedPointEmulationMantissaBitOffset(
      implementation_->handle, 0);
  }
  if (cublas_result == CUBLAS_STATUS_SUCCESS)
  {
    cublas_result = cublasSetFixedPointEmulationMantissaBitCountPointer(
      implementation_->handle,
      reinterpret_cast<int*>(implementation_->retained_mantissa_bits));
  }
  if (cublas_result == CUBLAS_STATUS_SUCCESS)
  {
    int* configured_pointer = nullptr;
    cublas_result = cublasGetFixedPointEmulationMantissaBitCountPointer(
      implementation_->handle, &configured_pointer);
    if (cublas_result == CUBLAS_STATUS_SUCCESS &&
        configured_pointer !=
          reinterpret_cast<int*>(implementation_->retained_mantissa_bits))
    {
      cublas_result = CUBLAS_STATUS_INTERNAL_ERROR;
    }
  }
  status.cublas_status = static_cast<int>(cublas_result);
  if (cublas_result != CUBLAS_STATUS_SUCCESS)
  {
    implementation_->Release();
    return status;
  }
  status.initialized = true;
#else
  (void)m;
  (void)n;
  (void)k;
  (void)max_mantissa_bits;
  (void)workspace_limit_bytes;
#endif
  return status;
}

DGSEMFixedPointPanelStatus DGSEMFixedPointVolumeContext::Mult(
  const DenseTensor& panel_operator,
  const DenseMatrix& contravariant_flux_panel,
  DenseMatrix& weak_volume)
{
  DGSEMFixedPointPanelStatus& status = implementation_->status;
  status.launch_succeeded = false;
  status.emulation_observed = false;
  status.retained_status_valid = false;
  status.retained_mantissa_bits = -1;
#if DG_IGR_HAS_CUBLAS_FIXEDPOINT_FP64
  if (!status.initialized)
  {
    return status;
  }
  MFEM_VERIFY(sizeof(real_t) == sizeof(double),
              "DGSEM fixed-point panel requires FP64 real_t.");
  MFEM_VERIFY(panel_operator.SizeI() == status.m &&
                panel_operator.SizeJ() == status.k &&
                panel_operator.SizeK() == 1,
              "DGSEM fixed-point panel operator shape mismatch.");
  MFEM_VERIFY(contravariant_flux_panel.Height() == status.k &&
                contravariant_flux_panel.Width() == status.n,
              "DGSEM fixed-point panel input shape mismatch.");
  MFEM_VERIFY(weak_volume.TotalSize() == status.m * status.n,
              "DGSEM fixed-point panel output shape mismatch.");

  const cudaError_t memset_result = cudaMemsetAsync(
    implementation_->retained_mantissa_bits,
    0xff,
    sizeof(std::int32_t),
    nullptr);
  status.cuda_status = static_cast<int>(memset_result);
  if (memset_result != cudaSuccess)
  {
    return status;
  }

  const double alpha = 1.0;
  const double beta = 0.0;
  const cublasStatus_t cublas_result = cublasGemmEx(
    implementation_->handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    status.m,
    status.n,
    status.k,
    &alpha,
    panel_operator.Read(true),
    CUDA_R_64F,
    status.m,
    contravariant_flux_panel.Read(true),
    CUDA_R_64F,
    status.k,
    &beta,
    weak_volume.Write(true),
    CUDA_R_64F,
    status.m,
    CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
    CUBLAS_GEMM_DEFAULT);
  status.cublas_status = static_cast<int>(cublas_result);
  status.launch_succeeded = cublas_result == CUBLAS_STATUS_SUCCESS;
#else
  (void)panel_operator;
  (void)contravariant_flux_panel;
  (void)weak_volume;
#endif
  return status;
}

DGSEMFixedPointPanelStatus
DGSEMFixedPointVolumeContext::SynchronizeStatus()
{
  DGSEMFixedPointPanelStatus& status = implementation_->status;
#if DG_IGR_HAS_CUBLAS_FIXEDPOINT_FP64
  if (status.initialized && status.launch_succeeded)
  {
    static_assert(sizeof(std::int32_t) == sizeof(int),
                  "cuBLAS retained-mantissa telemetry requires int32_t.");
    std::int32_t retained = -1;
    const cudaError_t result = cudaMemcpy(
      &retained,
      implementation_->retained_mantissa_bits,
      sizeof(std::int32_t),
      cudaMemcpyDeviceToHost);
    status.cuda_status = static_cast<int>(result);
    if (result == cudaSuccess)
    {
      status.retained_mantissa_bits = retained;
      status.retained_status_valid =
        retained == -1 ||
        (retained >= 0 && retained <= status.max_mantissa_bits + 7);
      status.emulation_observed =
        status.retained_status_valid && retained >= 0;
    }
    else
    {
      status.launch_succeeded = false;
    }
  }
#endif
  return status;
}

void AuditCentralVolumePanelDevice(const DenseMatrix& candidate,
                                   const DenseMatrix& reference,
                                   Vector& audit)
{
  MFEM_VERIFY(candidate.Height() == reference.Height() &&
                candidate.Width() == reference.Width(),
              "DGSEM volume panel audit shape mismatch.");
  if (audit.Size() != 3)
  {
    audit.SetSize(3);
  }
  audit.UseDevice(true);
  const int entries = candidate.TotalSize();
  const real_t* value = candidate.Read();
  const real_t* truth = reference.Read();
  real_t* result = audit.Write();
  mfem::forall(1,
               [=] MFEM_HOST_DEVICE(int)
               {
                 real_t max_reference = 0.0;
                 real_t max_error = 0.0;
                 real_t non_finite = 0.0;
                 for (int i = 0; i < entries; ++i)
                 {
                   if (!isfinite(value[i]) || !isfinite(truth[i]))
                   {
                     non_finite += 1.0;
                     continue;
                   }
                   max_reference = fmax(max_reference, fabs(truth[i]));
                   max_error = fmax(max_error, fabs(value[i] - truth[i]));
                 }
                 result[0] = max_reference;
                 result[1] = max_error;
                 result[2] = non_finite;
               });
}

void CopyPackedMatrixDevice(const DenseMatrix& source, DenseMatrix& target)
{
  MFEM_VERIFY(source.Height() == target.Height() &&
                source.Width() == target.Width(),
              "DGSEM packed matrix copy shape mismatch.");
  const int entries = source.TotalSize();
  const real_t* input = source.Read();
  real_t* output = target.Write();
  mfem::forall(entries,
               [=] MFEM_HOST_DEVICE(int i) { output[i] = input[i]; });
}

int DGSEMVolumePairCount(int dim, int ne, int n1d)
{
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM volume pairs support dimensions 1 and 2.");
  MFEM_VERIFY(ne >= 0 && n1d >= 2,
              "DGSEM volume pair dimensions are invalid.");
  const int lines = dim == 1 ? 1 : n1d;
  const int pairs_per_line = n1d * (n1d - 1) / 2;
  return ne * dim * lines * pairs_per_line;
}

void AddEulerLLFVolumeDissipationDevice(
  int dim,
  int ne,
  int n1d,
  real_t gamma,
  const Array<int>& lex_to_local,
  const DenseMatrix& split_matrix,
  const Vector& weights,
  const DenseMatrix& states,
  const Vector& sigma,
  const DenseMatrix& metrics,
  DenseMatrix& pair_dissipation,
  DenseMatrix& weak_volume)
{
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM device LLF volume supports dimensions 1 and 2.");
  const int dof = dim == 1 ? n1d : n1d * n1d;
  const int rows = ne * dof;
  const int num_equations = dim + 2;
  const int lines = dim == 1 ? 1 : n1d;
  const int pairs_per_line = n1d * (n1d - 1) / 2;
  const int pairs_per_element = dim * lines * pairs_per_line;
  const int total_pairs = DGSEMVolumePairCount(dim, ne, n1d);
  MFEM_VERIFY(ne > 0 && n1d >= 2 && lex_to_local.Size() == dof,
              "DGSEM device LLF volume tensor map mismatch.");
  MFEM_VERIFY(split_matrix.Height() == n1d &&
                split_matrix.Width() == n1d && weights.Size() == n1d,
              "DGSEM device LLF volume basis size mismatch.");
  MFEM_VERIFY(states.Height() == rows &&
                states.Width() == num_equations && sigma.Size() == rows,
              "DGSEM device LLF volume state/sigma size mismatch.");
  MFEM_VERIFY(metrics.Height() == rows && metrics.Width() == dim * dim,
              "DGSEM device LLF volume metric size mismatch.");
  MFEM_VERIFY(weak_volume.Height() == rows &&
                weak_volume.Width() == num_equations,
              "DGSEM device LLF weak-volume size mismatch.");
  if (pair_dissipation.Height() != total_pairs ||
      pair_dissipation.Width() != num_equations)
  {
    pair_dissipation.SetSize(total_pairs, num_equations);
  }
  pair_dissipation.GetMemory().UseDevice(true);
  weak_volume.GetMemory().UseDevice(true);

  const int* local = lex_to_local.Read();
  const real_t* state = states.Read();
  const real_t* sigma_value = sigma.Read();
  const real_t* metric = metrics.Read();
  real_t* pair = pair_dissipation.Write();

  // Evaluate the nonlinear directional wave speed once per unordered pair.
  // The low/high orientation exactly matches the legacy host pair loop.
  mfem::forall(
    total_pairs,
    [=] MFEM_HOST_DEVICE(int pair_row)
    {
      const int element = pair_row / pairs_per_element;
      const int pair_local = pair_row - element * pairs_per_element;
      const int direction = pair_local / (lines * pairs_per_line);
      const int direction_local = pair_local -
                                  direction * lines * pairs_per_line;
      const int line = direction_local / pairs_per_line;
      int ordinal = direction_local - line * pairs_per_line;
      int low = 0;
      while (ordinal >= n1d - low - 1)
      {
        ordinal -= n1d - low - 1;
        ++low;
      }
      const int high = low + ordinal + 1;
      const int low_lex = direction == 0
                            ? low + n1d * line
                            : line + n1d * low;
      const int high_lex = direction == 0
                             ? high + n1d * line
                             : line + n1d * high;
      const int row_low = element * dof + local[low_lex];
      const int row_high = element * dof + local[high_lex];
      real_t average_metric[2] = {0.0, 0.0};
      for (int physical = 0; physical < dim; ++physical)
      {
        const int metric_column = direction * dim + physical;
        average_metric[physical] =
          0.5 * (metric[row_low + rows * metric_column] +
                 metric[row_high + rows * metric_column]);
      }
      const real_t scale_low = EulerDirectionalLLFScale(state + row_low,
                                                         rows,
                                                         average_metric,
                                                         dim,
                                                         gamma,
                                                         sigma_value[row_low]);
      const real_t scale_high =
        EulerDirectionalLLFScale(state + row_high,
                                 rows,
                                 average_metric,
                                 dim,
                                 gamma,
                                 sigma_value[row_high]);
      const real_t scale = fmax(scale_low, scale_high);
      for (int equation = 0; equation < num_equations; ++equation)
      {
        pair[pair_row + total_pairs * equation] =
          -0.5 * scale *
          (state[row_high + rows * equation] -
           state[row_low + rows * equation]);
      }
    });

  const real_t* split = split_matrix.Read();
  const real_t* weight = weights.Read();
  const real_t* pair_value = pair_dissipation.Read();
  real_t* volume = weak_volume.ReadWrite();
  mfem::forall(
    rows * num_equations,
    [=] MFEM_HOST_DEVICE(int entry)
    {
      const int row = entry % rows;
      const int equation = entry / rows;
      const int element = row / dof;
      const int element_local = row - element * dof;
      int lex = 0;
      for (; lex < dof; ++lex)
      {
        if (local[lex] == element_local)
        {
          break;
        }
      }
      const int i = lex % n1d;
      const int j = dim == 2 ? lex / n1d : 0;
      real_t value = 0.0;
      for (int direction = 0; direction < dim; ++direction)
      {
        const int coordinate = direction == 0 ? i : j;
        const int line = dim == 1 ? 0 : (direction == 0 ? j : i);
        for (int other = 0; other < n1d; ++other)
        {
          if (other == coordinate)
          {
            continue;
          }
          const int low = coordinate < other ? coordinate : other;
          const int high = coordinate < other ? other : coordinate;
          const int pair_local =
            direction * lines * pairs_per_line + line * pairs_per_line +
            PairOrdinal(low, high, n1d);
          const int pair_row = element * pairs_per_element + pair_local;
          value -= split[coordinate + n1d * other] *
                   pair_value[pair_row + total_pairs * equation];
        }
      }
      real_t tensor_weight = weight[i];
      if (dim == 2)
      {
        tensor_weight *= weight[j];
      }
      volume[entry] += tensor_weight * value;
    });
}

void AssembleEulerRusanovFacesDevice(int dim,
                                     int ne,
                                     int dof,
                                     real_t gamma,
                                     real_t velocity_density_floor,
                                     const DenseMatrix& left_states,
                                     const DenseMatrix& right_states,
                                     const DenseMatrix& normals,
                                     const Vector& left_sigma,
                                     const Vector& right_sigma,
                                     const Vector& face_weights,
                                     const Array<int>& row_face_offsets,
                                     const Array<int>& row_face_ids,
                                     const Array<int>& row_face_signs,
                                     DenseMatrix& face_flux,
                                     Vector& face_speeds,
                                     DenseMatrix& weak_surface)
{
  const int nfaces = left_states.Height();
  const int num_equations = dim + 2;
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM device Rusanov faces support dimensions 1 and 2.");
  MFEM_VERIFY(ne >= 0 && dof > 0 && left_states.Width() == num_equations &&
                right_states.Height() == nfaces &&
                right_states.Width() == num_equations,
              "DGSEM device Rusanov state size mismatch.");
  MFEM_VERIFY(normals.Height() == nfaces && normals.Width() == dim &&
                left_sigma.Size() == nfaces && right_sigma.Size() == nfaces &&
                face_weights.Size() == nfaces,
              "DGSEM device Rusanov face data size mismatch.");
  const int rows = ne * dof;
  MFEM_VERIFY(row_face_offsets.Size() == rows + 1 &&
                row_face_ids.Size() == row_face_signs.Size(),
              "DGSEM device Rusanov row-face adjacency size mismatch.");

  // One packed host preflight avoids a separate device synchronization for
  // every point while keeping the bounded correctness fixture well-defined.
  const real_t* host_normal = normals.HostRead();
  for (int face = 0; face < nfaces; ++face)
  {
    real_t normal_squared = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      const real_t value = host_normal[face + nfaces * d];
      normal_squared += value * value;
    }
    MFEM_VERIFY(normal_squared > 0.0,
                "DGSEM device Rusanov face normal must be nonzero.");
  }

  if (face_flux.Height() != nfaces || face_flux.Width() != num_equations)
  {
    face_flux.SetSize(nfaces, num_equations);
  }
  if (face_speeds.Size() != nfaces)
  {
    face_speeds.SetSize(nfaces);
  }
  if (weak_surface.Height() != rows || weak_surface.Width() != num_equations)
  {
    weak_surface.SetSize(rows, num_equations);
  }
  face_flux.GetMemory().UseDevice(true);
  face_speeds.UseDevice(true);
  weak_surface.GetMemory().UseDevice(true);

  const real_t* left = left_states.Read();
  const real_t* right = right_states.Read();
  const real_t* normal = normals.Read();
  const real_t* sigma_l = left_sigma.Read();
  const real_t* sigma_r = right_sigma.Read();
  const real_t* weight = face_weights.Read();
  const int* row_offset = row_face_offsets.Read();
  const int* row_face = row_face_ids.Read();
  const int* row_sign = row_face_signs.Read();
  real_t* numerical_flux = face_flux.Write();
  real_t* speed = face_speeds.Write();
  real_t* surface = weak_surface.Write();

  mfem::forall(
    nfaces,
    [=] MFEM_HOST_DEVICE(int face)
    {
      const real_t speed_l =
        EulerSpeed(left + face, nfaces, normal + face, nfaces, dim, gamma);
      const real_t speed_r =
        EulerSpeed(right + face, nfaces, normal + face, nfaces, dim, gamma);
      const real_t max_speed = fmax(speed_l, speed_r);
      real_t normal_squared = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        const real_t normal_value = normal[face + nfaces * d];
        normal_squared += normal_value * normal_value;
      }
      const real_t scaled_speed = max_speed * sqrt(normal_squared);
      speed[face] = max_speed;
      for (int equation = 0; equation < num_equations; ++equation)
      {
        const real_t flux_l = EulerNormalFluxEntry(left + face,
                                                   nfaces,
                                                   normal + face,
                                                   nfaces,
                                                   dim,
                                                   equation,
                                                   gamma,
                                                   sigma_l[face],
                                                   velocity_density_floor);
        const real_t flux_r = EulerNormalFluxEntry(right + face,
                                                   nfaces,
                                                   normal + face,
                                                   nfaces,
                                                   dim,
                                                   equation,
                                                   gamma,
                                                   sigma_r[face],
                                                   velocity_density_floor);
        numerical_flux[face + nfaces * equation] =
          0.5 * (scaled_speed * (left[face + nfaces * equation] -
                                 right[face + nfaces * equation]) +
                 flux_l + flux_r);
      }
    });

  mfem::forall(
    rows * num_equations,
    [=] MFEM_HOST_DEVICE(int entry)
    {
      const int row = entry % rows;
      const int equation = entry / rows;
      real_t value = 0.0;
      for (int adjacent = row_offset[row];
           adjacent < row_offset[row + 1];
           ++adjacent)
      {
        const int face = row_face[adjacent];
        const real_t weighted_flux =
          weight[face] * numerical_flux[face + nfaces * equation];
        if (row_sign[adjacent] < 0)
        {
          value -= weighted_flux;
        }
        else
        {
          value += weighted_flux;
        }
      }
      surface[entry] = value;
    });
}

void ApplyInverseMassDevice(int ne,
                            int dof,
                            const DenseMatrix& inverse_mass,
                            const DenseMatrix& weak,
                            DenseMatrix& strong)
{
  const int rows = ne * dof;
  MFEM_VERIFY(ne >= 0 && dof > 0 && inverse_mass.Height() == rows &&
                inverse_mass.Width() == dof && weak.Height() == rows,
              "DGSEM device inverse-mass size mismatch.");
  const int num_equations = weak.Width();
  if (strong.Height() != rows || strong.Width() != num_equations)
  {
    strong.SetSize(rows, num_equations);
  }
  strong.GetMemory().UseDevice(true);
  const real_t* mass = inverse_mass.Read();
  const real_t* input = weak.Read();
  real_t* output = strong.Write();

  mfem::forall(rows * num_equations,
               [=] MFEM_HOST_DEVICE(int entry)
               {
                 const int row = entry % rows;
                 const int equation = entry / rows;
                 const int element = row / dof;
                 real_t value = 0.0;
                 for (int j = 0; j < dof; ++j)
                 {
                   value += mass[row + rows * j] *
                            input[element * dof + j + rows * equation];
                 }
                 output[entry] = value;
               });
}

void ComponentSumsDevice(const DenseMatrix& values, Vector& sums)
{
  const int rows = values.Height();
  const int num_equations = values.Width();
  if (sums.Size() != num_equations)
  {
    sums.SetSize(num_equations);
  }
  sums.UseDevice(true);
  const real_t* value = values.Read();
  real_t* sum = sums.Write();
  mfem::forall(num_equations,
               [=] MFEM_HOST_DEVICE(int equation)
               {
                 real_t component_sum = 0.0;
                 for (int row = 0; row < rows; ++row)
                 {
                   component_sum += value[row + rows * equation];
                 }
                 sum[equation] = component_sum;
               });
}

void DeterministicMaximumDevice(const Vector& values, Vector& maximum)
{
  if (maximum.Size() != 1)
  {
    maximum.SetSize(1);
  }
  maximum.UseDevice(true);
  const int size = values.Size();
  const real_t* value = values.Read();
  real_t* result = maximum.Write();
  mfem::forall(1,
               [=] MFEM_HOST_DEVICE(int)
               {
                 real_t current = 0.0;
                 for (int i = 0; i < size; ++i)
                 {
                   current = fmax(current, value[i]);
                 }
                 result[0] = current;
               });
}

} // namespace dgsem
} // namespace mfem
