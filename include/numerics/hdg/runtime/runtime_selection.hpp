#pragma once

/**
 * @file numerics/hdg/runtime/runtime_selection.hpp
 * @brief Closed static selections for the accepted W8A conventional card.
 */

#include "numerics/hdg/runtime/run_contract.hpp"

#include <string_view>

namespace dg::hdg::runtime
{

inline constexpr std::string_view conventional_persistent_card_id =
    "exp.hdg.euler.conventional.smooth_persistent.w8a.v1";
inline constexpr std::string_view conventional_persistent_case_id =
    "conventional-smooth-persistent-v1";

struct ConventionalEulerSmoothPersistentV1Tag
{
  static constexpr std::string_view id = conventional_persistent_card_id;
  static constexpr std::string_view case_id = conventional_persistent_case_id;
};

struct EulerCompressible2dTag
{
  static constexpr std::string_view id = "euler.compressible.2d";
};

struct ConventionalHdgTag
{
  static constexpr std::string_view id = "hdg.conventional";
};

struct MfemDgInterfaceTraceTag
{
  static constexpr std::string_view id = "mfem.dg_interface.trace";
};

struct FixedScalarTauTag
{
  static constexpr std::string_view id = "tau.fixed";
};

struct BackwardEulerRuntimeTag
{
  static constexpr std::string_view id = "backward_euler";
};

struct CpuSerialBackendTag
{
  static constexpr std::string_view id = "cpu.serial";
};

struct CpuRank2BackendTag
{
  static constexpr std::string_view id = "cpu.rank2";
};

struct CudaRank1OneGpuBackendTag
{
  static constexpr std::string_view id = "cuda.rank1.one_gpu";
};

struct CudaRank2OneGpuHostStagedBackendTag
{
  static constexpr std::string_view id =
      "cuda.rank2.one_gpu.host_staged";
};

struct Fp64RuntimeTag
{
  static constexpr std::string_view id = "fp64";
};

template <typename BackendTag>
using ConventionalEulerSmoothPersistentV1Selection =
    StaticRunSelection<EulerCompressible2dTag,
                       ConventionalHdgTag,
                       MfemDgInterfaceTraceTag,
                       FixedScalarTauTag,
                       BackwardEulerRuntimeTag,
                       BackendTag,
                       Fp64RuntimeTag>;

using CpuSerialRuntimeSelection =
    ConventionalEulerSmoothPersistentV1Selection<CpuSerialBackendTag>;
using CpuRank2RuntimeSelection =
    ConventionalEulerSmoothPersistentV1Selection<CpuRank2BackendTag>;
using CudaRank1OneGpuRuntimeSelection =
    ConventionalEulerSmoothPersistentV1Selection<CudaRank1OneGpuBackendTag>;
using CudaRank2OneGpuHostStagedRuntimeSelection =
    ConventionalEulerSmoothPersistentV1Selection<
        CudaRank2OneGpuHostStagedBackendTag>;

} // namespace dg::hdg::runtime
