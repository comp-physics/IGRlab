#include "numerics/dgsem/dgsem_device.hpp"
#include "physics/bc/detail/euler_boundary_state.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"

#include <cmath>
#include <limits>

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

MFEM_HOST_DEVICE inline real_t EulerNodalCharacteristicSpeed(
  const real_t* state,
  int stride,
  int dim,
  real_t gamma,
  real_t sigma,
  real_t density_floor,
  bool frozen_sigma_spectral_speed)
{
  const real_t rho = state[0];
  const real_t pressure = EulerPressure(state, stride, dim, gamma);
  real_t momentum_squared = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t momentum = state[(1 + d) * stride];
    momentum_squared += momentum * momentum;
  }
  const real_t transport_speed =
    sqrt(momentum_squared) / fmax(rho, density_floor);
  return frozen_sigma_spectral_speed
           ? dg_igr::frozen_scalar_igr_algebra::
               FrozenSigmaCharacteristicModulusBound(rho,
                                                      pressure,
                                                      transport_speed,
                                                      sigma,
                                                      gamma,
                                                      density_floor)
           : transport_speed + sqrt(gamma * pressure / rho);
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

MFEM_HOST_DEVICE inline real_t FrozenSigmaNormalSpeed(
  const real_t* state,
  int stride,
  const real_t* normal,
  int normal_stride,
  int dim,
  real_t gamma,
  real_t sigma,
  real_t density_floor)
{
  const real_t rho = state[0];
  const real_t pressure = EulerPressure(state, stride, dim, gamma);
  real_t normal_squared = 0.0;
  real_t momentum_dot_normal = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t normal_value = normal[d * normal_stride];
    momentum_dot_normal += state[(1 + d) * stride] * normal_value;
    normal_squared += normal_value * normal_value;
  }
  const real_t normal_norm = sqrt(normal_squared);
  const real_t transport_speed =
    fabs(momentum_dot_normal) / (fmax(rho, density_floor) * normal_norm);
  return dg_igr::frozen_scalar_igr_algebra::
    FrozenSigmaCharacteristicModulusBound(rho,
                                          pressure,
                                          transport_speed,
                                          sigma,
                                          gamma,
                                          density_floor);
}

MFEM_HOST_DEVICE inline real_t EulerRusanovFaceCharacteristicSpeed(
  const real_t* left,
  const real_t* right,
  int state_stride,
  const real_t* normal,
  int normal_stride,
  int dim,
  real_t gamma,
  real_t sigma_left,
  real_t sigma_right,
  real_t density_floor,
  bool frozen_sigma_spectral_speed)
{
  const real_t speed_left =
    frozen_sigma_spectral_speed
      ? FrozenSigmaNormalSpeed(left,
                               state_stride,
                               normal,
                               normal_stride,
                               dim,
                               gamma,
                               sigma_left,
                               density_floor)
      : EulerSpeed(left,
                   state_stride,
                   normal,
                   normal_stride,
                   dim,
                   gamma);
  const real_t speed_right =
    frozen_sigma_spectral_speed
      ? FrozenSigmaNormalSpeed(right,
                               state_stride,
                               normal,
                               normal_stride,
                               dim,
                               gamma,
                               sigma_right,
                               density_floor)
      : EulerSpeed(right,
                   state_stride,
                   normal,
                   normal_stride,
                   dim,
                   gamma);
  return fmax(speed_left, speed_right);
}

MFEM_HOST_DEVICE inline real_t EulerDirectionalLLFScale(
  const real_t* state,
  int stride,
  const real_t* direction,
  int dim,
  real_t gamma,
  real_t sigma,
  real_t density_floor)
{
  const real_t rho = fmax(state[0], density_floor);
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
  const real_t pressure = (gamma - 1.0) * (energy - kinetic);
  const real_t acoustic_numerator =
    dg_igr::frozen_scalar_igr_algebra::
      FrozenSigmaLocalHyperbolicityMargin(pressure, sigma, gamma);
  const real_t direction_norm = sqrt(fmax(direction_squared, 0.0));
  const real_t speed =
    fabs(velocity_dot_direction) +
    sqrt(fabs(acoustic_numerator) / rho) * direction_norm;
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

void ScanEulerIGRStatePanelsDevice(int dim,
                                   real_t gamma,
                                   real_t density_floor,
                                   bool scan_owned_states,
                                   bool scan_face_states,
                                   const DenseMatrix& states,
                                   const Vector& sigma,
                                   const DenseMatrix& left_states,
                                   const Vector& left_sigma,
                                   const DenseMatrix& right_states,
                                   const Vector& right_sigma,
                                   Array<double>& slots,
                                   Array<double>& result)
{
  const int rows = states.Height();
  const int face_rows = left_states.Height();
  MFEM_VERIFY((dim == 1 || dim == 2) && states.Width() == dim + 2 &&
                sigma.Size() == rows && left_states.Width() == dim + 2 &&
                right_states.Height() == face_rows &&
                right_states.Width() == dim + 2 &&
                left_sigma.Size() == face_rows &&
                right_sigma.Size() == face_rows &&
                std::isfinite(static_cast<double>(gamma)) && gamma > 1.0 &&
                std::isfinite(static_cast<double>(density_floor)) &&
                density_floor > 0.0 &&
                scan_owned_states != scan_face_states,
              "DGSEM device state scan received inconsistent packed fields.");
  constexpr int result_width = 6;
  constexpr int slot_width = 7;
  constexpr int lanes = 256;
  static_assert(lanes * slot_width == kDGSEMDeviceStateScanWorkspaceSize,
                "DGSEM state-scan workspace contract drifted.");
  const int samples = scan_owned_states ? rows : 2 * face_rows;
  if (slots.Size() < kDGSEMDeviceStateScanWorkspaceSize)
  {
    slots.SetSize(kDGSEMDeviceStateScanWorkspaceSize);
  }
  slots.GetMemory().UseDevice(true);
  if (result.Size() != result_width)
  {
    result.SetSize(result_width);
  }
  result.GetMemory().UseDevice(true);

  const real_t* const state = scan_owned_states ? states.Read() : nullptr;
  const real_t* const sigma_value =
    scan_owned_states ? sigma.Read() : nullptr;
  const real_t* const left = scan_face_states ? left_states.Read() : nullptr;
  const real_t* const sigma_left =
    scan_face_states ? left_sigma.Read() : nullptr;
  const real_t* const right = scan_face_states ? right_states.Read() : nullptr;
  const real_t* const sigma_right =
    scan_face_states ? right_sigma.Read() : nullptr;
  double* const sample_result = slots.Write();
  double* const reduced = result.Write();
  const double gamma64 = static_cast<double>(gamma);
  const double density_floor64 = static_cast<double>(density_floor);
  const double infinity = std::numeric_limits<double>::infinity();
  const int invalid_density =
    static_cast<int>(DGSEMDeviceStateStatus::InvalidDensity);
  const int density_below_floor =
    static_cast<int>(DGSEMDeviceStateStatus::DensityBelowFloor);
  const int invalid_pressure =
    static_cast<int>(DGSEMDeviceStateStatus::InvalidPressure);
  const int invalid_sigma =
    static_cast<int>(DGSEMDeviceStateStatus::InvalidSigma);

  // Fixed lanes scan coalesced sample strides and write bounded summaries.
  // The second kernel folds only these lane records, not the full state.
  mfem::forall(
    lanes,
    [=] MFEM_HOST_DEVICE(int lane)
    {
      int first_bad = samples;
      int first_status = static_cast<int>(DGSEMDeviceStateStatus::Valid);
      double count = 0.0;
      double nonpositive = 0.0;
      double nonfinite = 0.0;
      double minimum_q = infinity;
      double minimum_ratio = infinity;

      for (int sample = lane; sample < samples; sample += lanes)
      {
        const int stride = scan_owned_states ? rows : face_rows;
        const int row = scan_owned_states ? sample : sample % face_rows;
        const bool right_side = scan_face_states && sample >= face_rows;
        const real_t* const sample_state =
          scan_owned_states ? state : (right_side ? right : left);
        const real_t* const sample_sigma =
          scan_owned_states
            ? sigma_value
            : (right_side ? sigma_right : sigma_left);
        int status = static_cast<int>(DGSEMDeviceStateStatus::Valid);
        for (int equation = 0; equation < dim + 2; ++equation)
        {
          if (!isfinite(static_cast<double>(
                sample_state[row + stride * equation])))
          {
            status = equation == 0 ? invalid_density : invalid_pressure;
            break;
          }
        }
        if (status == static_cast<int>(DGSEMDeviceStateStatus::Valid) &&
            !isfinite(static_cast<double>(sample_sigma[row])))
        {
          status = invalid_sigma;
        }
        const double density = static_cast<double>(sample_state[row]);
        if (status == static_cast<int>(DGSEMDeviceStateStatus::Valid) &&
            !(density > 0.0))
        {
          status = invalid_density;
        }
        if (status == static_cast<int>(DGSEMDeviceStateStatus::Valid) &&
            density < density_floor64)
        {
          status = density_below_floor;
        }
        if (status != static_cast<int>(DGSEMDeviceStateStatus::Valid))
        {
          first_bad = sample;
          first_status = status;
          break;
        }

        double momentum_squared = 0.0;
        for (int d = 0; d < dim; ++d)
        {
          const double momentum = static_cast<double>(
            sample_state[row + stride * (1 + d)]);
          momentum_squared += momentum * momentum;
        }
        const double energy = static_cast<double>(
          sample_state[row + stride * (1 + dim)]);
        const double pressure =
          (gamma64 - 1.0) *
          (energy - 0.5 * momentum_squared / density);
        if (!isfinite(pressure) || !(pressure > 0.0))
        {
          first_bad = sample;
          first_status = invalid_pressure;
          break;
        }

        const double sigma64 = static_cast<double>(sample_sigma[row]);
        const double q_sigma =
          gamma64 * pressure + (gamma64 - 1.0) * sigma64;
        const double ratio = q_sigma / (gamma64 * pressure);
        count += 1.0;
        const bool q_finite = isfinite(q_sigma);
        const bool ratio_finite = isfinite(ratio);
        nonfinite += !q_finite || !ratio_finite ? 1.0 : 0.0;
        if (q_finite)
        {
          nonpositive += q_sigma <= 0.0 ? 1.0 : 0.0;
          minimum_q = fmin(minimum_q, q_sigma);
        }
        if (ratio_finite)
        {
          minimum_ratio = fmin(minimum_ratio, ratio);
        }
      }

      const int offset = lane * slot_width;
      sample_result[offset + 0] = static_cast<double>(first_bad);
      sample_result[offset + 1] = static_cast<double>(first_status);
      sample_result[offset + 2] = count;
      sample_result[offset + 3] = nonpositive;
      sample_result[offset + 4] = nonfinite;
      sample_result[offset + 5] = minimum_q;
      sample_result[offset + 6] = minimum_ratio;
    });

  // Deterministic fold selects the smallest failing sample ordinal and merges
  // bounded lane aggregates. Only the six final scalars leave the device.
  mfem::forall(
    1,
    [=] MFEM_HOST_DEVICE(int)
    {
      reduced[0] =
        static_cast<double>(static_cast<int>(DGSEMDeviceStateStatus::Valid));
      reduced[1] = 0.0;
      reduced[2] = 0.0;
      reduced[3] = 0.0;
      reduced[4] = infinity;
      reduced[5] = infinity;
      int first_bad = samples;
      for (int lane = 0; lane < lanes; ++lane)
      {
        const int offset = lane * slot_width;
        const int lane_bad = static_cast<int>(sample_result[offset + 0]);
        if (lane_bad < first_bad)
        {
          first_bad = lane_bad;
          reduced[0] = sample_result[offset + 1];
        }
        reduced[1] += sample_result[offset + 2];
        reduced[2] += sample_result[offset + 3];
        reduced[3] += sample_result[offset + 4];
        reduced[4] = fmin(reduced[4], sample_result[offset + 5]);
        reduced[5] = fmin(reduced[5], sample_result[offset + 6]);
      }
    });
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
  const Vector& boundary_gammas,
  const Vector& boundary_outlet_pressures,
  const Vector& boundary_density_floors,
  const Vector& boundary_pressure_floors,
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
                boundary_gammas.Size() == nfaces &&
                boundary_outlet_pressures.Size() == nfaces &&
                boundary_density_floors.Size() == nfaces &&
                boundary_pressure_floors.Size() == nfaces &&
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
  const real_t* boundary_gamma = boundary_gammas.Read();
  const real_t* boundary_outlet_pressure =
    boundary_outlet_pressures.Read();
  const real_t* boundary_density_floor = boundary_density_floors.Read();
  const real_t* boundary_pressure_floor = boundary_pressure_floors.Read();
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
          boundary_gamma[face],
          boundary_outlet_pressure[face],
          reference,
          scaled_normal,
          interior,
          exterior,
          boundary_density_floor[face],
          boundary_pressure_floor[face]);
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

void ComputeEulerNodalCharacteristicSpeedsDevice(
  int dim,
  real_t gamma,
  real_t velocity_density_floor,
  const DenseMatrix& states,
  const Vector& sigma,
  Vector& nodal_speeds,
  bool frozen_sigma_spectral_speed)
{
  const int rows = states.Height();
  const int num_equations = dim + 2;
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM device Euler speed supports dimensions 1 and 2 only.");
  MFEM_VERIFY(states.Width() == num_equations && sigma.Size() == rows,
              "DGSEM device Euler speed state/sigma size mismatch.");
  if (nodal_speeds.Size() != rows)
  {
    nodal_speeds.SetSize(rows);
  }
  nodal_speeds.UseDevice(true);
  const real_t* state = states.Read();
  const real_t* sigma_value = sigma.Read();
  real_t* speed = nodal_speeds.Write();

  mfem::forall(rows,
               [=] MFEM_HOST_DEVICE(int row)
               {
                 speed[row] = EulerNodalCharacteristicSpeed(
                   state + row,
                   rows,
                   dim,
                   gamma,
                   sigma_value[row],
                   velocity_density_floor,
                   frozen_sigma_spectral_speed);
               });
}

void BuildEulerContravariantFluxDevice(int dim,
                                       real_t gamma,
                                       real_t velocity_density_floor,
                                       const DenseMatrix& states,
                                       const Vector& sigma,
                                       const DenseMatrix& metrics,
                                       DenseMatrix& contravariant_flux,
                                       Vector& nodal_speeds,
                                       bool frozen_sigma_spectral_speed)
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
  contravariant_flux.GetMemory().UseDevice(true);
  const real_t* state = states.Read();
  const real_t* sigma_value = sigma.Read();
  const real_t* metric = metrics.Read();
  real_t* flux = contravariant_flux.Write();

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
  ComputeEulerNodalCharacteristicSpeedsDevice(dim,
                                              gamma,
                                              velocity_density_floor,
                                              states,
                                              sigma,
                                              nodal_speeds,
                                              frozen_sigma_spectral_speed);
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
  real_t velocity_density_floor,
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
                                                         sigma_value[row_low],
                                                         velocity_density_floor);
      const real_t scale_high =
        EulerDirectionalLLFScale(state + row_high,
                                 rows,
                                 average_metric,
                                 dim,
                                 gamma,
                                 sigma_value[row_high],
                                 velocity_density_floor);
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

void ComputeEulerRusanovFaceCharacteristicSpeedsDevice(
  int dim,
  real_t gamma,
  real_t velocity_density_floor,
  const DenseMatrix& left_states,
  const DenseMatrix& right_states,
  const DenseMatrix& normals,
  const Vector& left_sigma,
  const Vector& right_sigma,
  Vector& face_speeds,
  bool frozen_sigma_spectral_speed)
{
  const int nfaces = left_states.Height();
  const int num_equations = dim + 2;
  MFEM_VERIFY(dim == 1 || dim == 2,
              "DGSEM device Rusanov speeds support dimensions 1 and 2 only.");
  MFEM_VERIFY(left_states.Width() == num_equations &&
                right_states.Height() == nfaces &&
                right_states.Width() == num_equations,
              "DGSEM device Rusanov speed state size mismatch.");
  MFEM_VERIFY(normals.Height() == nfaces && normals.Width() == dim &&
                left_sigma.Size() == nfaces && right_sigma.Size() == nfaces,
              "DGSEM device Rusanov speed face data size mismatch.");

  if (face_speeds.Size() != nfaces)
  {
    face_speeds.SetSize(nfaces);
  }
  face_speeds.UseDevice(true);
  const real_t* left = left_states.Read();
  const real_t* right = right_states.Read();
  const real_t* normal = normals.Read();
  const real_t* sigma_l = left_sigma.Read();
  const real_t* sigma_r = right_sigma.Read();
  real_t* speed = face_speeds.Write();

  mfem::forall(
    nfaces,
    [=] MFEM_HOST_DEVICE(int face)
    {
      speed[face] = EulerRusanovFaceCharacteristicSpeed(
        left + face,
        right + face,
        nfaces,
        normal + face,
        nfaces,
        dim,
        gamma,
        sigma_l[face],
        sigma_r[face],
        velocity_density_floor,
        frozen_sigma_spectral_speed);
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
                                     DenseMatrix& weak_surface,
                                     bool frozen_sigma_spectral_speed)
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
      const real_t max_speed = EulerRusanovFaceCharacteristicSpeed(
        left + face,
        right + face,
        nfaces,
        normal + face,
        nfaces,
        dim,
        gamma,
        sigma_l[face],
        sigma_r[face],
        velocity_density_floor,
        frozen_sigma_spectral_speed);
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

void IntegratePhysicalBoundaryFluxDevice(const Array<int>& ownership,
                                         const Vector& face_weights,
                                         const DenseMatrix& face_flux,
                                         Vector& boundary_flux)
{
  const int nfaces = face_flux.Height();
  const int num_equations = face_flux.Width();
  MFEM_VERIFY(num_equations > 0 && ownership.Size() == nfaces &&
                face_weights.Size() == nfaces,
              "DGSEM physical-boundary flux reduction size mismatch.");

  if (boundary_flux.Size() != num_equations)
  {
    boundary_flux.SetSize(num_equations);
  }
  boundary_flux.UseDevice(true);
  const int* owner = ownership.Read();
  const real_t* weight = face_weights.Read();
  const real_t* flux = face_flux.Read();
  real_t* integral = boundary_flux.Write();
  const int boundary_owner =
    static_cast<int>(DGSEMDeviceFaceOwnership::Boundary);

  // One deterministic loop per conserved component avoids both FP64 atomics
  // and cancellation through local/shared faces. This path is opt-in
  // qualification work, not part of an unarmed production residual.
  mfem::forall(
    num_equations,
    [=] MFEM_HOST_DEVICE(int equation)
    {
      real_t value = 0.0;
      for (int face = 0; face < nfaces; ++face)
      {
        if (owner[face] == boundary_owner)
        {
          value += weight[face] * flux[face + nfaces * equation];
        }
      }
      integral[equation] = value;
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
