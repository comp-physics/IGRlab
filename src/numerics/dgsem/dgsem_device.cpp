#include "numerics/dgsem/dgsem_device.hpp"

#include <cmath>

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
    const real_t diagonal =
      equation == 1 + physical_direction ? pressure + sigma : 0.0;
    return momentum_equation * momentum_direction / rho + diagonal;
  }
  const real_t energy = state[(1 + dim) * stride];
  const real_t safe_rho = fmax(rho, rho_floor);
  return momentum_direction * (energy + pressure) / rho +
         sigma * momentum_direction / safe_rho;
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
           (pressure + sigma) * normal[direction * normal_stride];
  }
  const real_t safe_rho = fmax(rho, rho_floor);
  return normal_velocity * (state[(1 + dim) * stride] + pressure) +
         sigma * momentum_dot_normal / safe_rho;
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
  real_t momentum_squared = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t momentum = state[(1 + d) * stride];
    const real_t normal_value = normal[d * normal_stride];
    momentum_dot_normal += momentum * normal_value;
    momentum_squared += momentum * momentum;
    normal_squared += normal_value * normal_value;
  }
  const real_t sound = sqrt(gamma * pressure / rho);
  if (normal)
  {
    return fabs(momentum_dot_normal / rho) / sqrt(normal_squared) + sound;
  }
  return sqrt(momentum_squared) / rho + sound;
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

  contravariant_flux.SetSize(rows, dim * num_equations);
  nodal_speeds.SetSize(rows);
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
  weak_volume.SetSize(rows, num_equations);
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
                                     const Array<int>& left_elements,
                                     const Array<int>& left_local_dofs,
                                     const Array<int>& right_elements,
                                     const Array<int>& right_local_dofs,
                                     const Array<int>& ownership,
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
  MFEM_VERIFY(left_elements.Size() == nfaces &&
                left_local_dofs.Size() == nfaces &&
                right_elements.Size() == nfaces &&
                right_local_dofs.Size() == nfaces && ownership.Size() == nfaces,
              "DGSEM device Rusanov face map size mismatch.");

  const int rows = ne * dof;
  face_flux.SetSize(nfaces, num_equations);
  face_speeds.SetSize(nfaces);
  weak_surface.SetSize(rows, num_equations);
  face_flux.GetMemory().UseDevice(true);
  face_speeds.UseDevice(true);
  weak_surface.GetMemory().UseDevice(true);

  const real_t* left = left_states.Read();
  const real_t* right = right_states.Read();
  const real_t* normal = normals.Read();
  const real_t* sigma_l = left_sigma.Read();
  const real_t* sigma_r = right_sigma.Read();
  const real_t* weight = face_weights.Read();
  const int* element_l = left_elements.Read();
  const int* local_l = left_local_dofs.Read();
  const int* element_r = right_elements.Read();
  const int* local_r = right_local_dofs.Read();
  const int* owner = ownership.Read();
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
      const int element = row / dof;
      const int element_local = row - element * dof;
      real_t value = 0.0;
      for (int face = 0; face < nfaces; ++face)
      {
        const real_t weighted_flux =
          weight[face] * numerical_flux[face + nfaces * equation];
        if (element_l[face] == element && local_l[face] == element_local)
        {
          value -= weighted_flux;
        }
        if (owner[face] == static_cast<int>(DGSEMDeviceFaceOwnership::Local) &&
            element_r[face] == element && local_r[face] == element_local)
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
  strong.SetSize(rows, num_equations);
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
  sums.SetSize(num_equations);
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
  maximum.SetSize(1);
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
