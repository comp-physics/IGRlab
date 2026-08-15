#include "numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private.cuh"

#include <cstddef>

namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private
{
namespace
{
__global__ void AssemblePacked(const BatchInput input, const BatchOutput output)
{
  const int element = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (element >= input.elements) return;

  const int rhs_points = input.static_euler_rhs_points
    ? input.static_euler_rhs_points[element] : 0;
  const double* const rhs = input.static_euler_rhs
    ? input.static_euler_rhs +
        static_cast<std::size_t>(element) * input.volume_points * 4
    : nullptr;
  const algebra::Input local_input{
    input.vertices + static_cast<std::size_t>(element) * vertices_per_element,
    input.parameters[element],
    input.interior + static_cast<std::size_t>(element) * algebra::interior_unknowns,
    input.previous_interior + static_cast<std::size_t>(element) * algebra::interior_unknowns,
    input.trace + static_cast<std::size_t>(element) * algebra::trace_unknowns,
    input.previous_trace + static_cast<std::size_t>(element) * algebra::trace_unknowns,
    rhs, rhs_points,
    {input.volume_xyw, input.volume_points, input.segment_xw, input.segment_points}};
  algebra::Output local_output{
    output.interior_residual + static_cast<std::size_t>(element) * algebra::interior_unknowns,
    output.trace_residual + static_cast<std::size_t>(element) * algebra::trace_unknowns,
    output.A + static_cast<std::size_t>(element) * algebra::interior_unknowns * algebra::interior_unknowns,
    output.B + static_cast<std::size_t>(element) * algebra::interior_unknowns * algebra::trace_unknowns,
    output.C + static_cast<std::size_t>(element) * algebra::trace_unknowns * algebra::interior_unknowns,
    output.D + static_cast<std::size_t>(element) * algebra::trace_unknowns * algebra::trace_unknowns,
    output.f + static_cast<std::size_t>(element) * algebra::interior_unknowns,
    output.g + static_cast<std::size_t>(element) * algebra::trace_unknowns};
  output.statuses[element] = algebra::Assemble(local_input, local_output);
}

bool ValidEnvelope(const BatchInput& input, const BatchOutput& output) noexcept
{
  return input.volume_points > 0 && input.segment_points > 0 &&
    input.volume_xyw && input.segment_xw &&
    (!input.static_euler_rhs || input.static_euler_rhs_points) &&
    input.vertices && input.parameters && input.interior &&
    input.previous_interior && input.trace && input.previous_trace && output.statuses &&
    output.interior_residual && output.trace_residual && output.A && output.B &&
    output.C && output.D && output.f && output.g;
}
} // namespace

cudaError_t Launch(const BatchInput& input, const BatchOutput& output,
                   const cudaStream_t stream) noexcept
{
  if (input.elements < 0) return cudaErrorInvalidValue;
  if (input.elements == 0) return cudaSuccess;
  if (!ValidEnvelope(input, output)) return cudaErrorInvalidValue;
  constexpr unsigned int threads = 128;
  const unsigned int blocks =
    (static_cast<unsigned int>(input.elements) + threads - 1) / threads;
  AssemblePacked<<<blocks, threads, 0, stream>>>(input, output);
  return cudaPeekAtLastError();
}
} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private
