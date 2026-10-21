#pragma once

// The canonical CUDA build publishes one exact native architecture to the
// runtime and its consumers.  An explicit supported selection is required.
#ifndef DG_IGR_HDG_NATIVE_CUDA_ARCH
#error "HDG CUDA requires DG_IGR_HDG_NATIVE_CUDA_ARCH from its build target"
#elif DG_IGR_HDG_NATIVE_CUDA_ARCH != 80 && \
      DG_IGR_HDG_NATIVE_CUDA_ARCH != 90 && \
      DG_IGR_HDG_NATIVE_CUDA_ARCH != 120
#error "HDG CUDA supports only native SM80, SM90, or SM120"
#endif

#if defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ != 10 * DG_IGR_HDG_NATIVE_CUDA_ARCH
#error "HDG CUDA device image must match the configured native architecture"
#endif

namespace dg::hdg::runtime::cuda_w8a::detail
{

inline constexpr int configured_native_architecture =
    DG_IGR_HDG_NATIVE_CUDA_ARCH;
inline constexpr int configured_cuda_architecture =
    10 * configured_native_architecture;

constexpr bool CudaDeviceMatchesNativeArchitecture(
    const int major, const int minor) noexcept
{
  return major == configured_native_architecture / 10 &&
      minor == configured_native_architecture % 10;
}

} // namespace dg::hdg::runtime::cuda_w8a::detail
