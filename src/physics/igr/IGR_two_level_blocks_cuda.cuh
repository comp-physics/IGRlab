#pragma once

#include "igrlab/environment.hpp"
#include "mfem.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>

namespace igrlab
{
namespace igr_internal
{
namespace two_level_blocks_internal
{

// One block per element: convert each FP64 RHS entry once and reuse it for
// every row of the column-major inverse. This uses IEEE FP32 FMA, never TF32.
// The original FP64 inverse stays available for bounded per-application
// recovery; the outer flexible solve certifies the current FP64 operator.
static __global__ void ApplyMixed(
    int dof, const int* map, const double* inverse64, const float* inverse32,
    const int* setup_fallback, const double* input, const double* coarse,
    double* output, unsigned long long* fallback_rows)
{
  const int e = blockIdx.x, i = threadIdx.x;
  __shared__ float rhs[256];
  bool unsafe_input = false, nonzero_input = false;
  if (i < dof)
  {
    const double value = input[map[e * dof + i]];
    nonzero_input = value != 0.0;
    // Check the range before converting. Subnormal inputs take the FP64
    // route too, preserving corrections outside the normal FP32 range.
    unsafe_input = !isfinite(value) || fabs(value) > FLT_MAX ||
                   (nonzero_input && fabs(value) < FLT_MIN);
    rhs[i] = unsafe_input ? 0.0f : static_cast<float>(value);
  }
  const bool unsafe = __syncthreads_or(unsafe_input || setup_fallback[e]);
  const bool nonzero = __syncthreads_or(nonzero_input);
  if (i >= dof) { return; }

  bool fallback = unsafe;
  float value32 = 0.0f;
  if (!fallback)
  {
    const std::size_t first = std::size_t(e) * dof * dof + i;
    for (int j = 0; j < dof; ++j)
    {
      value32 = __fmaf_rn(inverse32[first + std::size_t(j) * dof],
                          rhs[j], value32);
    }
    // Zero with a nonzero RHS can be cancellation or underflow. Re-evaluate
    // that row rather than silently erase a potentially useful correction.
    fallback = !isfinite(value32) ||
               (nonzero && fabsf(value32) < FLT_MIN);
  }

  const int row = map[e * dof + i];
  double value = coarse[row];
  if (fallback)
  {
    for (int j = 0; j < dof; ++j)
    {
      value += inverse64[(std::size_t(e) * dof + j) * dof + i] *
               input[map[e * dof + j]];
    }
    atomicAdd(fallback_rows, 1ULL);
  }
  else
  {
    value += static_cast<double>(value32);
  }
  output[row] = value;
}

} // namespace two_level_blocks_internal

// Private storage/application helper shared by DG and DGSEM. FP64 is the
// unchanged default. FP32 changes only the element-block preconditioner action;
// coarse algebra, evolving state, and original-operator residuals remain FP64.
class TwoLevelBlocks
{
public:
  static std::string PrecisionFromEnvironment()
  {
    const char* value = ::igrlab::GetEnv("DG_IGR_IGR_TWO_LEVEL_BLOCK_PRECISION");
    const std::string precision = value && value[0] ? value : "fp64";
    MFEM_VERIFY(precision == "fp64" || precision == "fp32",
                "DG_IGR_IGR_TWO_LEVEL_BLOCK_PRECISION must be fp64|fp32.");
    return precision;
  }

  void Configure(int ne, int dof, const mfem::Vector& inverse,
                 const std::string& precision = PrecisionFromEnvironment())
  {
    static_assert(std::is_same<mfem::real_t, double>::value,
                  "Mixed two-level blocks require FP64 MFEM.");
    MFEM_VERIFY(!inverse_ && ne > 0 && dof > 0 && dof <= 256 &&
                  std::size_t(ne) * dof * dof == std::size_t(inverse.Size()) &&
                  (precision == "fp64" || precision == "fp32"),
                "Invalid two-level block initialization.");
    ne_ = ne;
    dof_ = dof;
    inverse_ = &inverse;
    mixed_ = precision == "fp32";
    if (!mixed_) { return; }

    inverse32_.SetSize(inverse.Size());
    setup_fallback_.SetSize(ne);
    fallback_rows_.SetSize(1);
    fallback_rows_.HostWrite()[0] = 0ULL;
    const double* original = inverse.HostRead();
    float* rounded = inverse32_.HostWrite();
    int* fallback = setup_fallback_.HostWrite();
    std::fill(fallback, fallback + ne, 0);
    for (int k = 0; k < inverse.Size(); ++k)
    {
      const double value = original[k];
      const bool unsafe = !std::isfinite(value) || std::fabs(value) > FLT_MAX ||
                         (value != 0.0 && std::fabs(value) < FLT_MIN);
      // A flagged block never reads this rounded entry during application.
      rounded[k] = unsafe ? 0.0f : static_cast<float>(value);
      fallback[k / (dof * dof)] |= unsafe;
    }
    for (int e = 0; e < ne; ++e) { setup_fallback_blocks_ += fallback[e]; }
    (void)inverse32_.Read();
    (void)setup_fallback_.Read();
    (void)fallback_rows_.Read();
    std::cout << "[igr-two-level-blocks] event=setup precision=ieee-fp32"
              << " elements=" << ne << " block_dof=" << dof
              << " fp32_inverse_bytes=" << std::size_t(inverse.Size()) * sizeof(float)
              << " retained_fp64_inverse_bytes="
              << std::size_t(inverse.Size()) * sizeof(double)
              << " recovery_metadata_bytes="
              << std::size_t(ne) * sizeof(int) + sizeof(unsigned long long)
              << " setup_fallback_blocks=" << setup_fallback_blocks_
              << " state=fp64 coarse=fp64 recovery=fp64-row"
              << std::endl;
  }

  ~TwoLevelBlocks()
  {
    if (mixed_ && inverse_)
    {
      std::cout << "[igr-two-level-blocks] event=summary precision=ieee-fp32"
                << " applications=" << applications_
                << " row_applications=" << applications_ * ne_ * dof_
                << " fp64_fallback_rows=" << FallbackRows()
                << " setup_fallback_blocks=" << setup_fallback_blocks_
                << std::endl;
    }
  }

  void Mult(const mfem::Vector& x, const mfem::Vector& coarse,
            const mfem::Array<int>& element_dofs, mfem::Vector& y) const
  {
    const int height = ne_ * dof_, d = dof_;
    MFEM_VERIFY(inverse_ && x.Size() == height && coarse.Size() == height &&
                  element_dofs.Size() == height && &x != &y && &coarse != &y,
                "Invalid two-level block operands.");
    y.SetSize(height);
    y.UseDevice(true);
    const double* input = x.Read();
    const double* coarse_value = coarse.Read();
    const double* inverse = inverse_->Read();
    const int* map = element_dofs.Read();
    double* output = y.Write();
    if (mixed_)
    {
      int threads = 32;
      while (threads < d) { threads *= 2; }
      two_level_blocks_internal::ApplyMixed<<<ne_, threads>>>(
        d, map, inverse, inverse32_.Read(), setup_fallback_.Read(), input,
        coarse_value, output, fallback_rows_.ReadWrite());
      MFEM_VERIFY(cudaGetLastError() == cudaSuccess,
                  "Two-level FP32 block kernel launch failed.");
      ++applications_;
      return;
    }
    // Preserve the original summation order in the FP64 control.
    mfem::forall(height, [=] MFEM_HOST_DEVICE(int entry)
    {
      const int e = entry / d, i = entry % d;
      double value = coarse_value[map[entry]];
      for (int j = 0; j < d; ++j)
      { value += inverse[(e * d + j) * d + i] * input[map[e * d + j]]; }
      output[map[entry]] = value;
    });
  }

  // Read aggregate recovery counts only at a fixture checkpoint or teardown,
  // never as a synchronization in the production application loop.
  unsigned long long FallbackRows() const
  { return mixed_ ? fallback_rows_.HostRead()[0] : 0ULL; }
  int SetupFallbackBlocks() const { return setup_fallback_blocks_; }

private:
  int ne_ = 0, dof_ = 0, setup_fallback_blocks_ = 0;
  bool mixed_ = false;
  const mfem::Vector* inverse_ = nullptr;
  mfem::Array<float> inverse32_;
  mfem::Array<int> setup_fallback_;
  mutable mfem::Array<unsigned long long> fallback_rows_;
  mutable unsigned long long applications_ = 0;
};

} // namespace igr_internal
} // namespace igrlab
