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
inline constexpr std::string_view scalar_igr_persistent_card_id =
    "exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3";
inline constexpr std::string_view scalar_igr_persistent_case_id =
    "sc25-periodic-smooth-persistent-v3";

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

struct TracedScalarSc25ProjectionV3Tag
{
  static constexpr std::string_view id = scalar_igr_persistent_card_id;
};

struct FullyHybridizedScalarSc25V3Tag
{
  static constexpr std::string_view id =
      "FullyHybridizedScalarPeriodicSc25ProjectionV3";
};

struct DualConservativeScalarTraceV1Tag
{
  static constexpr std::string_view id = "DualConservativeScalarTraceV1";
};

struct W7Tau25DensityHeightP2V3Tag
{
  static constexpr std::string_view id = "W7Tau25.DensityHeightP2.v3";
};

struct BackwardEulerMonolithicV1Tag
{
  static constexpr std::string_view id = "BackwardEulerMonolithicV1";
};

struct MfemDgInterfaceTraceTag
{
  static constexpr std::string_view id = "mfem.dg_interface.trace";
};

struct FixedScalarTauTag
{
  static constexpr std::string_view id = "tau.fixed";
};

inline constexpr std::string_view borrowed_euler_llf_stabilization_id =
    "tau.llf.max-interior-trace.frozen-jacobian.v1";

/**
 * Borrowed Euler data evaluates the local LLF speed from the current interior
 * and trace candidate at each residual assembly.  The selected speed is held
 * fixed only while that residual and its A/B/C/D blocks are formed.
 */
struct BorrowedEulerCurrentLlfTauTag
{
  static constexpr std::string_view id =
      borrowed_euler_llf_stabilization_id;
};

struct BackwardEulerRuntimeTag
{
  static constexpr std::string_view id = "backward_euler";
};

struct PartitionedBackwardEulerRuntimeTag
{
  static constexpr std::string_view id = "partitioned_backward_euler";
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

template <typename BackendTag,
          typename TimeTreatmentTag = BackwardEulerRuntimeTag>
using BorrowedEulerCurrentLlfSelection =
    StaticRunSelection<EulerCompressible2dTag,
                       ConventionalHdgTag,
                       MfemDgInterfaceTraceTag,
                       BorrowedEulerCurrentLlfTauTag,
                       TimeTreatmentTag,
                       BackendTag,
                       Fp64RuntimeTag>;

using BorrowedCudaRank1OneGpuRuntimeSelection =
    BorrowedEulerCurrentLlfSelection<CudaRank1OneGpuBackendTag>;
using BorrowedCudaRank2OneGpuHostStagedRuntimeSelection =
    BorrowedEulerCurrentLlfSelection<
        CudaRank2OneGpuHostStagedBackendTag>;

template <typename BackendTag>
using ScalarIgrSc25PersistentV3Selection =
    StaticRunSelection<TracedScalarSc25ProjectionV3Tag,
                       FullyHybridizedScalarSc25V3Tag,
                       DualConservativeScalarTraceV1Tag,
                       W7Tau25DensityHeightP2V3Tag,
                       BackwardEulerMonolithicV1Tag,
                       BackendTag,
                       Fp64RuntimeTag>;

using ScalarIgrCpuSerialRuntimeSelection =
    ScalarIgrSc25PersistentV3Selection<CpuSerialBackendTag>;
using ScalarIgrCpuRank2RuntimeSelection =
    ScalarIgrSc25PersistentV3Selection<CpuRank2BackendTag>;
using ScalarIgrCudaRank1RuntimeSelection =
    ScalarIgrSc25PersistentV3Selection<CudaRank1OneGpuBackendTag>;
using ScalarIgrCudaRank2RuntimeSelection =
    ScalarIgrSc25PersistentV3Selection<CudaRank2OneGpuHostStagedBackendTag>;

template <typename BackendTag>
using PartitionedBackwardEulerBorrowedEulerCurrentLlfSelection =
    BorrowedEulerCurrentLlfSelection<
        BackendTag, PartitionedBackwardEulerRuntimeTag>;

using PartitionedBackwardEulerCpuRank1RuntimeSelection =
    PartitionedBackwardEulerBorrowedEulerCurrentLlfSelection<
        CpuSerialBackendTag>;
using PartitionedBackwardEulerCpuSerialRuntimeSelection =
    PartitionedBackwardEulerCpuRank1RuntimeSelection;
using PartitionedBackwardEulerCpuRank2RuntimeSelection =
    PartitionedBackwardEulerBorrowedEulerCurrentLlfSelection<
        CpuRank2BackendTag>;
using PartitionedBackwardEulerCudaRank1RuntimeSelection =
    PartitionedBackwardEulerBorrowedEulerCurrentLlfSelection<
        CudaRank1OneGpuBackendTag>;
using PartitionedBackwardEulerCudaRank2RuntimeSelection =
    PartitionedBackwardEulerBorrowedEulerCurrentLlfSelection<
        CudaRank2OneGpuHostStagedBackendTag>;

} // namespace dg::hdg::runtime
