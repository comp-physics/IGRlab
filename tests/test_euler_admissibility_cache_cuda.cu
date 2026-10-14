#include "test_euler_admissibility_cache_cuda.hpp"
#include "mfem/general/cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <limits>

namespace admissibility_cuda_probe
{
namespace
{
__device__ bool NormalOrZero(double x)
{
  const unsigned long long bits = __double_as_longlong(x);
  const unsigned long long exponent = bits & 0x7ff0000000000000ULL;
  const unsigned long long fraction = bits & 0x000fffffffffffffULL;
  return exponent != 0x7ff0000000000000ULL &&
         (exponent != 0 || fraction == 0);
}

__device__ double Multiply(double a, double b, int& declined)
{
  const double result = __dmul_rn(a, b);
  if (!NormalOrZero(result) || (result == 0.0 && a != 0.0 && b != 0.0))
  { declined = 1; }
  return result;
}

__device__ double Add(double a, double b, int& declined)
{
  const double result = __dadd_rn(a, b);
  if (!NormalOrZero(result)) { declined = 1; }
  return result;
}

__device__ double Subtract(double a, double b, int& declined)
{
  const double result = __dsub_rn(a, b);
  if (!NormalOrZero(result)) { declined = 1; }
  return result;
}

__device__ double Divide(double a, double b, int& declined)
{
  const double result = __ddiv_rn(a, b);
  if (!NormalOrZero(result) || (result == 0.0 && a != 0.0))
  { declined = 1; }
  return result;
}

__device__ double Gather(const double* state, int index)
{
  return index >= 0 ? state[index] : -state[-index - 1];
}

__global__ void FillKernel(double* state, int ndofs, double rho, bool invalid)
{
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= ndofs) { return; }
  state[d] = rho;
  state[d + ndofs] = invalid && d == 0 ? CUDART_NAN : 0.0;
  state[d + 2 * ndofs] = -0.0;
  state[d + 3 * ndofs] = 2.5;
}

__global__ void CopyKernel(const double* input, double* state, int size)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) { state[i] = input[i]; }
}

__global__ void EvaluateKernel(const double* state, const int* maps,
                               const double* shapes, const Sample* samples,
                               int size, double gamma, double rho_min,
                               double p_min, Value* output)
{
  const int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_id >= size) { return; }
  const Sample sample = samples[sample_id];
  Value value{};
  if (!NormalOrZero(gamma) || !NormalOrZero(rho_min) || !NormalOrZero(p_min))
  { value.declined = 1; output[sample_id] = value; return; }
  for (int equation = 0; equation < 4; ++equation)
  {
    const int offset = sample.map_offset + equation * sample.dof;
    if (sample.shape_offset < 0)
    {
      value.state[equation] = Gather(state, maps[offset + sample.node]);
      if (!NormalOrZero(value.state[equation])) { value.declined = 1; }
      continue;
    }
    double accumulator = 0.0;
    int j = 0;
    // Match the proved native MFEM paired additions and scalar odd tail.
    for (; j + 1 < sample.dof; j += 2)
    {
      for (int lane = 0; lane < 2; ++lane)
      {
        const double u = Gather(state, maps[offset + j + lane]);
        const double shape = shapes[sample.shape_offset + j + lane];
        if (!NormalOrZero(u) || !NormalOrZero(shape))
        { value.declined = 1; }
        const double product = Multiply(u, shape, value.declined);
        accumulator = Add(accumulator, product, value.declined);
      }
    }
    if (j < sample.dof)
    {
      const double u = Gather(state, maps[offset + j]);
      const double shape = shapes[sample.shape_offset + j];
      if (!NormalOrZero(u) || !NormalOrZero(shape)) { value.declined = 1; }
      const double product = Multiply(u, shape, value.declined);
      accumulator = Add(product, accumulator, value.declined);
    }
    value.state[equation] = accumulator;
  }
  if (value.declined) { output[sample_id] = value; return; }

  const double rho = value.state[0];
  const double safe_rho = rho < 1.0e-16 ? 1.0e-16 : rho;
  double mom2 = 0.0;
  for (int d = 0; d < 2; ++d)
  {
    const double m = value.state[1 + d];
    mom2 = Add(mom2, Multiply(m, m, value.declined), value.declined);
  }
  value.kinetic = Divide(Multiply(0.5, mom2, value.declined), safe_rho,
                         value.declined);
  // Keep the original EOS traversal separate from inspect_state's kinetic.
  double eos_mom2 = 0.0;
  for (int d = 0; d < 2; ++d)
  {
    const double m = value.state[1 + d];
    eos_mom2 = Add(eos_mom2, Multiply(m, m, value.declined), value.declined);
  }
  const double eos_k = Divide(Multiply(0.5, eos_mom2, value.declined),
                              safe_rho, value.declined);
  value.pressure = Multiply(Subtract(gamma, 1.0, value.declined),
                            Subtract(value.state[3], eos_k, value.declined),
                            value.declined);
  value.bad_rho = rho < rho_min;
  value.bad_pressure = value.pressure < p_min;
  output[sample_id] = value;
}

__global__ void FoldKernel(const Value* values, int size, LocalSummary* result)
{
  // Deliberately ordered proof fold. This is not an optimized resident scan.
  LocalSummary out{CUDART_INF, CUDART_INF, 0, 0, -1, 0};
  for (int i = 0; i < size; ++i)
  {
    const Value value = values[i];
    out.declined |= value.declined;
    if (value.declined) { continue; }
    // Strict comparison preserves the first occurrence's signed-zero bits.
    if (value.state[0] < out.min_rho) { out.min_rho = value.state[0]; }
    if (value.pressure < out.min_pressure) { out.min_pressure = value.pressure; }
    out.bad_rho += value.bad_rho;
    out.bad_pressure += value.bad_pressure;
    if (out.first_sample < 0 && (value.bad_rho || value.bad_pressure))
    { out.first_sample = i; }
  }
  result[0] = out;
}
} // namespace

void FillDeviceState(mfem::Vector& state, int scalar_dofs, double rho,
                     bool invalid_first_momentum)
{
  MFEM_VERIFY(state.Size() == 4 * scalar_dofs, "Wrong test Euler layout.");
  FillKernel<<<(scalar_dofs + 127) / 128, 128>>>(
    state.Write(true), scalar_dofs, rho, invalid_first_momentum);
  MFEM_GPU_CHECK(cudaGetLastError());
}

void WriteDeviceState(mfem::Vector& state, const std::vector<double>& values)
{
  MFEM_VERIFY(values.size() == static_cast<std::size_t>(state.Size()),
              "Wrong test state size.");
  mfem::Vector input(state.Size());
  input.UseDevice(true);
  std::copy(values.begin(), values.end(), input.HostWrite());
  CopyKernel<<<(state.Size() + 127) / 128, 128>>>(
    input.Read(true), state.Write(true), state.Size());
  MFEM_GPU_CHECK(cudaGetLastError());
  // Input storage cannot expire while a device copy still consumes it.
  MFEM_GPU_CHECK(cudaDeviceSynchronize());
}

Result Evaluate(const mfem::Vector& state, const std::vector<int>& maps,
                const std::vector<double>& shapes,
                const std::vector<Sample>& samples, double gamma,
                double rho_min, double p_min, bool environment_supported,
                bool require_sticky_flags)
{
  Result result;
  result.environment_declined = !environment_supported;
  result.sticky_flags_declined = require_sticky_flags;
  if (result.environment_declined || result.sticky_flags_declined) { return result; }
  MFEM_VERIFY(!samples.empty() && samples.size() <= INT_MAX &&
                maps.size() <= INT_MAX && shapes.size() <= INT_MAX,
              "Invalid bounded test plan.");
  for (const auto& sample : samples)
  {
    MFEM_VERIFY(sample.dof > 0 && sample.map_offset >= 0 &&
      static_cast<std::size_t>(sample.map_offset) + 4u * sample.dof <= maps.size() &&
      (sample.shape_offset < 0 ? sample.node >= 0 && sample.node < sample.dof :
       static_cast<std::size_t>(sample.shape_offset) + sample.dof <= shapes.size()),
      "Invalid test sample incidence.");
  }
  for (int index : maps)
  {
    const long long decoded = index >= 0 ? index : -1LL - index;
    MFEM_VERIFY(decoded < state.Size(), "Invalid encoded test map.");
  }
  mfem::Array<int> device_maps(static_cast<int>(maps.size()));
  mfem::Array<Sample> device_samples(static_cast<int>(samples.size()));
  mfem::Vector device_shapes(static_cast<int>(shapes.size()));
  device_maps.GetMemory().UseDevice(true);
  device_samples.GetMemory().UseDevice(true);
  device_shapes.UseDevice(true);
  std::copy(maps.begin(), maps.end(), device_maps.HostWrite());
  std::copy(samples.begin(), samples.end(), device_samples.HostWrite());
  if (!shapes.empty())
  { std::copy(shapes.begin(), shapes.end(), device_shapes.HostWrite()); }
  mfem::Array<Value> values(static_cast<int>(samples.size()));
  mfem::Array<LocalSummary> summary(1);
  values.GetMemory().UseDevice(true); summary.GetMemory().UseDevice(true);
  EvaluateKernel<<<(samples.size() + 127) / 128, 128>>>(
    state.Read(true), device_maps.Read(true),
    shapes.empty() ? nullptr : device_shapes.Read(true),
    device_samples.Read(true), static_cast<int>(samples.size()), gamma,
    rho_min, p_min, values.Write(true));
  MFEM_GPU_CHECK(cudaGetLastError());
  FoldKernel<<<1, 1>>>(values.Read(true), values.Size(), summary.Write(true));
  MFEM_GPU_CHECK(cudaGetLastError());
  result.summary = summary.HostRead()[0];
  const Value* host_values = values.HostRead();
  result.values.assign(host_values, host_values + values.Size());
  return result;
}
} // namespace admissibility_cuda_probe
