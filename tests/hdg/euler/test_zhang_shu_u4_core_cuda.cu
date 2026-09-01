#include "numerics/hdg/euler/zhang_shu_u4.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

namespace
{

namespace zs = dg::hdg::euler::zhang_shu;

struct DeviceWitness final
{
  std::uint32_t checks = 0;
  double density_theta = 1.0;
  double pressure_theta = 1.0;
};

__global__ void EvaluateCore(DeviceWitness* const output)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  DeviceWitness witness{};
  const zs::Parameters parameters{1.4, 0.1, 0.1, 52};

  const double mean[4] = {1.0, 0.0, 0.0, 2.5};
  const double density_point[4] = {-0.2, 0.3, -0.4, 3.0};
  const zs::ThetaResult density =
      zs::DensityTheta(mean, density_point, parameters);
  double density_limited[4]{};
  const zs::Status density_blend = zs::BlendDensityOnlyU4(
      mean, density_point, density.theta, density_limited);
  if (density.ok() && density.active() &&
      density_blend == zs::Status::success &&
      density_limited[0] >= parameters.density_floor &&
      density_limited[1] == density_point[1] &&
      density_limited[2] == density_point[2] &&
      density_limited[3] == density_point[3])
  {
    witness.checks |= 1U << 0U;
  }

  const double pressure_point[4] = {0.4, 1.0, 0.0, 1.0};
  const zs::ThetaResult pressure =
      zs::PressureTheta(mean, pressure_point, parameters);
  double pressure_limited[4]{};
  const zs::Status pressure_blend = zs::BlendU4(
      mean, pressure_point, pressure.theta, pressure_limited);
  double limited_pressure = 0.0;
  if (pressure.ok() && pressure.active() &&
      pressure_blend == zs::Status::success &&
      zs::Pressure(pressure_limited, parameters.gamma, limited_pressure) &&
      limited_pressure >= parameters.pressure_floor)
  {
    witness.checks |= 1U << 1U;
  }

  // Component-major points with a common admissible mean.  Only density needs
  // limiting, so the scale helper must leave the remaining slabs exact.
  const double density_only_mean[4] = {1.0, 0.3, -0.4, 3.0};
  double coefficients[8] = {
    -0.2, 2.2,
     0.3, 0.3,
    -0.4,-0.4,
     3.0, 3.0
  };
  const double original_other[6] = {
      coefficients[2], coefficients[3], coefficients[4],
      coefficients[5], coefficients[6], coefficients[7]};
  const zs::ElementFactors factors = zs::ComputeElementFactors(
      density_only_mean, coefficients, 2,
      /*point_stride=*/1, /*component_stride=*/2, parameters);
  const zs::ScaleResult scale = zs::ScaleU4CoefficientsAboutMean(
      coefficients, 2, 2, 1, density_only_mean, factors, parameters);
  bool other_exact = true;
  for (int index = 0; index < 6; ++index)
  {
    other_exact = other_exact && coefficients[index + 2] == original_other[index];
  }
  if (factors.ok() && factors.density_theta < 1.0 &&
      factors.pressure_theta == 1.0 && scale.ok() && scale.modified &&
      coefficients[0] >= parameters.density_floor && other_exact)
  {
    witness.checks |= 1U << 2U;
  }
  witness.density_theta = factors.density_theta;
  witness.pressure_theta = pressure.theta;

  const double bad_mean[4] = {parameters.density_floor, 0.0, 0.0, 2.5};
  if (zs::ValidateAdmissibleMean(bad_mean, parameters) ==
      zs::Status::inadmissible_mean_density)
  {
    witness.checks |= 1U << 3U;
  }

  *output = witness;
}

}  // namespace

int main()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
  {
    return 75;
  }

  DeviceWitness* device_witness = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&device_witness),
                 sizeof(DeviceWitness)) != cudaSuccess)
  {
    return 2;
  }
  EvaluateCore<<<1, 1>>>(device_witness);
  DeviceWitness witness{};
  const bool copied = cudaGetLastError() == cudaSuccess &&
      cudaDeviceSynchronize() == cudaSuccess &&
      cudaMemcpy(&witness, device_witness, sizeof(DeviceWitness),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
  (void)cudaFree(device_witness);
  constexpr std::uint32_t expected_checks = (1U << 4U) - 1U;
  if (!copied || witness.checks != expected_checks)
  {
    std::cerr << "CUDA Zhang--Shu core checks failed: mask="
              << witness.checks << " density_theta=" << witness.density_theta
              << " pressure_theta=" << witness.pressure_theta << '\n';
    return 3;
  }
  std::cout << "CUDA Zhang--Shu U4 scalar core checks passed\n";
  return 0;
}
