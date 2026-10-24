#include "igrlab/compat.hpp"
#include "numerics/dgsem/dgsem_device.hpp"
#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace
{
void LiteralOriginalMaximum(const mfem::Vector& values, mfem::Vector& serial)
{
  const int size = values.Size();
  const mfem::real_t* value = values.Read();
  mfem::real_t* reference = serial.Write();
  // Literal original loop, genuinely device-compiled in the CUDA target.
  mfem::forall(1, [=] MFEM_HOST_DEVICE(int)
  {
    mfem::real_t current = 0.0;
    for (int i = 0; i < size; ++i)
    {
      current = fmax(current, value[i]);
    }
    reference[0] = current;
  });
}

int CheckMaximumBits()
{
  const mfem::real_t nan =
    std::numeric_limits<mfem::real_t>::quiet_NaN();
  const mfem::real_t infinity =
    std::numeric_limits<mfem::real_t>::infinity();
  mfem::Vector maximum(2);
  mfem::Vector retained_maximum(3);
  mfem::Vector serial(1);
  serial.UseDevice(true);
  // Cross the 1024-block cap and reuse oversized scratch for smaller grids.
  constexpr int largest_case = 1048593;
  mfem::Vector workspace(
    mfem::dgsem::DeterministicMaximumWorkspaceSize(largest_case));
  workspace.UseDevice(true);
  workspace = infinity;
  const int workspace_size = workspace.Size();
  const auto workspace_capacity = workspace.GetMemory().Capacity();
  const mfem::real_t* workspace_address = workspace.Read();
  int cases = 0;
  const auto check_values = [&](const std::string& label,
                                const mfem::Vector& values)
  {
    LiteralOriginalMaximum(values, serial);
    mfem::dgsem::DeterministicMaximumDevice(values, maximum);
    mfem::dgsem::DeterministicMaximumDevice(values, retained_maximum, workspace);
    const mfem::real_t* expected = serial.HostRead();
    const mfem::real_t* actual = maximum.HostRead();
    MFEM_VERIFY(maximum.Size() == 1 &&
                  std::memcmp(actual, expected, sizeof(mfem::real_t)) == 0,
                "Maximum differs from original serial bits: " << label);
    MFEM_VERIFY(retained_maximum.Size() == 1 &&
                  std::memcmp(retained_maximum.HostRead(), expected,
                              sizeof(mfem::real_t)) == 0,
                "Retained-workspace maximum differs from original serial bits: " << label);
    MFEM_VERIFY(workspace.Size() == workspace_size &&
                  workspace.GetMemory().Capacity() == workspace_capacity &&
                  workspace.Read() == workspace_address,
                "Maximum resized or replaced its retained scratch: " << label);
    ++cases;
  };
  const auto check = [&](const std::string& label,
                         const std::vector<mfem::real_t>& input)
  {
    mfem::Vector values(static_cast<int>(input.size()));
    values.UseDevice(true);
    if (!input.empty())
    {
      std::copy(input.begin(), input.end(), values.HostWrite());
    }
    check_values(label, values);
  };

  for (int size : {0, 1, 2, 31, 32, 33, 255, 256, 257, 511, 513,
                   4095, 4096, 4097, 4099, 8191, 8192, 8193,
                   147456, 147457, 1000003, 1048575, 1048576, 1048577, largest_case})
  {
    std::vector<mfem::real_t> input(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
    {
      input[static_cast<std::size_t>(i)] =
        0.125 * static_cast<mfem::real_t>(i % 37 - 18);
    }
    if (size > 0) { input.back() = 32.0; }
    check("positive tail length=" + std::to_string(size), input);
    if (size > 2)
    {
      input[0] = nan;
      input[static_cast<std::size_t>(size / 2)] = infinity;
      input.back() = -infinity;
      check("NaN and infinities length=" + std::to_string(size), input);
    }
  }
  // Exercise every participating lane, including maxima in a later stride.
  for (int lane = 0; lane < 256; ++lane)
  {
    std::vector<mfem::real_t> input(513, -1.0);
    input[static_cast<std::size_t>(256 + lane)] =
      8.0 + 0.00390625 * static_cast<mfem::real_t>(lane);
    check("positive winner lane=" + std::to_string(lane), input);
  }
  // Boundaries of participating blocks and later grid strides. Keep these
  // inputs modest; the million-entry serial oracle runs only a few times.
  for (int winner : {0, 255, 256, 1023, 1024, 1025, 4095, 4096, 4097,
                     8191, 8192, 16383, 16384, 65535, 65536, 65538})
  {
    std::vector<mfem::real_t> input(65539, -1.0);
    input[static_cast<std::size_t>(winner)] = 64.0;
    check("multiblock boundary winner=" + std::to_string(winner), input);
  }
  // The 17391-entry case has 17 first-pass blocks and a 4352-entry stride.
  // Place a unique maximum in the fourth stride of every participating block.
  constexpr int stride_blocks = 17;
  constexpr int stride = stride_blocks * 256;
  for (int block = 0; block < stride_blocks; ++block)
  {
    std::vector<mfem::real_t> input(4 * stride - 17, -1.0);
    const int winner = 3 * stride + block * 256 + block;
    input[static_cast<std::size_t>(winner)] = 64.0 + block;
    check("later-stride winner block=" + std::to_string(block), input);
  }
  for (const auto& input : std::vector<std::vector<mfem::real_t>>{
         {0.0}, {-0.0}, {0.0, -0.0}, {-0.0, 0.0},
         {nan}, {nan, -nan, -infinity}, {-3.0, -1.0, -infinity},
         {-0.0, nan, 0.0, -0.0, -infinity},
         {infinity, nan, infinity},
         {std::numeric_limits<mfem::real_t>::denorm_min(), nan, -0.0}})
  {
    check("exceptional sequence=" + std::to_string(cases), input);
  }
  if (std::numeric_limits<mfem::real_t>::has_signaling_NaN)
  {
    const mfem::real_t signaling_nan =
      std::numeric_limits<mfem::real_t>::signaling_NaN();
    check("signaling NaN and zero", {signaling_nan, -infinity, -0.0});
    check("signaling NaN and positive maximum", {64.0, signaling_nan, -1.0});
  }
  std::vector<mfem::real_t> zeros(513, nan);
  check("ragged all-NaN partials", zeros);
  zeros[0] = 0.0;
  zeros[256] = -0.0;
  zeros.back() = -0.0;
  check("ragged signed-zero and NaN partials", zeros);
  std::reverse(zeros.begin(), zeros.end());
  check("reversed signed-zero and NaN partials", zeros);
  check("large all-negative fallback", std::vector<mfem::real_t>(4099, -1.0));
  // The final global zero must use the literal original traversal, including
  // NaNs and signed-zero ties across blocks and the ragged fifth grid stride.
  std::vector<mfem::real_t> large(largest_case, -1.0);
  check("capped-grid all-negative fallback", large);
  std::fill(large.begin(), large.end(), nan);
  check("capped-grid all-NaN fallback", large);
  std::fill(large.begin(), large.end(), 0.0);
  check("capped-grid all-zero fallback", large);
  for (int i = 0; i < largest_case; ++i)
  {
    large[static_cast<std::size_t>(i)] = i % 3 == 0 ? nan :
      (i % 3 == 1 ? -0.0 : -infinity);
  }
  large[0] = 0.0;
  large.back() = -0.0;
  check("capped-grid signed-zero NaN fallback", large);

  mfem::Vector rewritten(513);
  rewritten.UseDevice(true);
  mfem::real_t* value = rewritten.Write();
  mfem::forall(rewritten.Size(), [=] MFEM_HOST_DEVICE(int i)
  {
    value[i] = i == 512 ? 64.0 : 0.25 * (i % 17);
  });
  check_values("fresh device-authoritative values", rewritten);
  rewritten.SetSize(4 * stride - 17);
  for (int phase = 0; phase < 3; ++phase)
  {
    const int size = rewritten.Size();
    const int winner = phase == 0 ? size - 1 : 256;
    const mfem::real_t peak = phase == 0 ? 64.0 : (phase == 1 ? 8.0 : -0.0);
    mfem::real_t* current = rewritten.Write();
    mfem::forall(size, [=] MFEM_HOST_DEVICE(int i)
    {
      current[i] = i == winner ? peak : -1.0;
    });
    check_values("rewritten device values retained scratch phase=" +
                   std::to_string(phase), rewritten);
  }
  return cases;
}

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
void BenchmarkMaximum()
{
  constexpr int warmups = 4, repetitions = 32;
  mfem::Vector workspace(mfem::dgsem::DeterministicMaximumWorkspaceSize(9437184));
  workspace.UseDevice(true);
  workspace = 0.0;
  const int workspace_size = workspace.Size();
  const auto workspace_capacity = workspace.GetMemory().Capacity();
  const mfem::real_t* workspace_address = workspace.Read();
  cudaEvent_t begin, end;
  MFEM_VERIFY(cudaEventCreate(&begin) == cudaSuccess &&
                cudaEventCreate(&end) == cudaSuccess,
              "Maximum benchmark event creation failed.");
  for (int size : {9437184, 3151872})
  {
    mfem::Vector values(size), serial(1), old_maximum(1), new_maximum(1);
    values.UseDevice(true); serial.UseDevice(true);
    old_maximum.UseDevice(true); new_maximum.UseDevice(true);
    mfem::real_t* input = values.Write();
    mfem::forall(size, [=] MFEM_HOST_DEVICE(int i)
    {
      input[i] = i == size - 1 ? 64.0 : 1.0 + 0.25 * (i % 37);
    });
    // One serial device oracle per actual production length, outside timing.
    LiteralOriginalMaximum(values, serial);
    const auto apply = [&](int variant)
    {
      if (variant == 0)
      {
        mfem::dgsem::DeterministicMaximumDevice(values, old_maximum);
      }
      else
      {
        mfem::dgsem::DeterministicMaximumDevice(values, new_maximum, workspace);
      }
    };
    for (int repeat = 0; repeat < warmups; ++repeat)
    {
      apply(repeat % 2); apply(1 - repeat % 2);
    }
    const auto check = [&]()
    {
      const mfem::real_t* expected = serial.HostRead();
      MFEM_VERIFY(std::isfinite(expected[0]) && expected[0] == 64.0 &&
                    std::memcmp(old_maximum.HostRead(), expected, sizeof(mfem::real_t)) == 0 &&
                    std::memcmp(new_maximum.HostRead(), expected, sizeof(mfem::real_t)) == 0,
                  "Maximum benchmark variants differ from the literal original bits.");
      MFEM_VERIFY(workspace.Size() == workspace_size &&
                    workspace.GetMemory().Capacity() == workspace_capacity &&
                    workspace.Read() == workspace_address,
                  "Maximum benchmark replaced retained scratch.");
    };
    check();
    std::vector<double> times[2];
    for (int repeat = 0; repeat < repetitions; ++repeat)
    {
      for (int index = 0; index < 2; ++index)
      {
        const int variant = (repeat + index) % 2;
        MFEM_VERIFY(cudaEventRecord(begin) == cudaSuccess,
                    "Maximum benchmark start event failed.");
        apply(variant);
        MFEM_VERIFY(cudaEventRecord(end) == cudaSuccess &&
                      cudaEventSynchronize(end) == cudaSuccess,
                    "Maximum benchmark stop event failed.");
        float elapsed = 0.0f;
        MFEM_VERIFY(cudaEventElapsedTime(&elapsed, begin, end) == cudaSuccess &&
                      std::isfinite(elapsed) && elapsed > 0.0f,
                    "Maximum benchmark event interval is invalid.");
        times[variant].push_back(elapsed);
      }
    }
    check();
    const double old_mean = std::accumulate(times[0].begin(), times[0].end(), 0.0) / repetitions;
    const double new_mean = std::accumulate(times[1].begin(), times[1].end(), 0.0) / repetitions;
    double paired_reduction = 0.0;
    for (int repeat = 0; repeat < repetitions; ++repeat)
    {
      paired_reduction += 100.0 * (1.0 - times[1][repeat] / times[0][repeat]);
    }
    std::cout << std::setprecision(17)
              << "[dgsem-maximum-benchmark] device=cuda n=" << size
              << " warmups_per_variant=" << warmups
              << " repetitions_per_variant=" << repetitions
              << " order=alternating-interleaved old_one_cta_mean_ms=" << old_mean
              << " retained_hierarchical_mean_ms=" << new_mean
              << " mean_paired_reduction_percent=" << paired_reduction / repetitions
              << " bitwise_equal=1 finite_positive_input=1 scratch_reused=1"
              << " host_reads_in_event_interval=0 scope=device-reduction-only\n";
  }
  MFEM_VERIFY(cudaEventDestroy(begin) == cudaSuccess &&
                cudaEventDestroy(end) == cudaSuccess,
              "Maximum benchmark event destruction failed.");
}
#endif
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  const bool benchmark = argc == 3 && std::string(argv[2]) == "--benchmark";
  MFEM_VERIFY(argc <= 2 || benchmark, "Expected [cpu|cuda] [--benchmark].");
  MFEM_VERIFY(backend == "cpu" || backend == "cuda", "Expected cpu or cuda.");
  MFEM_VERIFY(!benchmark || backend == "cuda", "Maximum benchmark requires CUDA.");
#if !defined(MFEM_USE_CUDA) || !defined(__CUDACC__)
  MFEM_VERIFY(backend == "cpu", "CUDA oracle requires a CUDA translation unit.");
#endif
  mfem::Device device(backend.c_str());
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1, "Maximum fixture is single rank.");
  MFEM_VERIFY((backend == "cuda") == mfem::Device::Allows(mfem::Backend::CUDA),
              "Maximum fixture backend differs from requested device.");
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  if (benchmark)
  {
    BenchmarkMaximum();
    return 0;
  }
#endif
  const int cases = CheckMaximumBits();
  std::cout << "[dgsem-maximum-parity] device=" << backend
            << " cases=" << cases << " bitwise_equal=1 retained_workspace=1\n";
  return 0;
}
