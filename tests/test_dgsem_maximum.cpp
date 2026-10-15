#include "igrlab/compat.hpp"
#include "numerics/dgsem/dgsem_device.hpp"
#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
int CheckMaximumBits()
{
  const mfem::real_t nan =
    std::numeric_limits<mfem::real_t>::quiet_NaN();
  const mfem::real_t infinity =
    std::numeric_limits<mfem::real_t>::infinity();
  mfem::Vector maximum(2);
  mfem::Vector serial(1);
  serial.UseDevice(true);
  int cases = 0;
  const auto check_values = [&](const std::string& label,
                                const mfem::Vector& values)
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
    mfem::dgsem::DeterministicMaximumDevice(values, maximum);
    const mfem::real_t* expected = serial.HostRead();
    const mfem::real_t* actual = maximum.HostRead();
    MFEM_VERIFY(maximum.Size() == 1 &&
                  std::memcmp(actual, expected, sizeof(mfem::real_t)) == 0,
                "Maximum differs from original serial bits: " << label);
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
                   4099, 147456, 147457, 1000003})
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

  mfem::Vector rewritten(513);
  rewritten.UseDevice(true);
  mfem::real_t* value = rewritten.Write();
  mfem::forall(rewritten.Size(), [=] MFEM_HOST_DEVICE(int i)
  {
    value[i] = i == 512 ? 64.0 : 0.25 * (i % 17);
  });
  check_values("fresh device-authoritative values", rewritten);
  return cases;
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  MFEM_VERIFY(backend == "cpu" || backend == "cuda", "Expected cpu or cuda.");
#if !defined(MFEM_USE_CUDA) || !defined(__CUDACC__)
  MFEM_VERIFY(backend == "cpu", "CUDA oracle requires a CUDA translation unit.");
#endif
  mfem::Device device(backend.c_str());
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1, "Maximum fixture is single rank.");
  MFEM_VERIFY((backend == "cuda") == mfem::Device::Allows(mfem::Backend::CUDA),
              "Maximum fixture backend differs from requested device.");
  const int cases = CheckMaximumBits();
  std::cout << "[dgsem-maximum-parity] device=" << backend
            << " cases=" << cases << " bitwise_equal=1\n";
  return 0;
}
