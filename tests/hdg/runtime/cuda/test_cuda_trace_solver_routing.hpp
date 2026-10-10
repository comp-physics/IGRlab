#pragma once

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh"
#include "numerics/hdg/runtime/cuda/tensor_volume_layout.hpp"

#include <cstdint>
#include <limits>

namespace dg::hdg::runtime::cuda_w8a::detail
{

// Validate the fixed aggregate fields already carried by one prepared step.
// This test helper adds no event history or production observability surface.
constexpr bool CudaPreparedTraceSolverRoutingValuesValid(
    const generic::CudaEngineKind engine, const int ranks,
    const int trace_solves, const std::uint64_t native_rank1_calls,
    const std::uint64_t hypre_calls, const bool native_rank1_configured,
    const bool hypre_host_staged_rank2_configured,
    const bool accounting_complete) noexcept
{
  if (!generic::IsKnownCudaEngineKind(engine) ||
      (ranks != 1 && ranks != 2) || trace_solves < 0 ||
      !accounting_complete)
    return false;

  const bool expect_native =
      engine == generic::CudaEngineKind::tensor_core && ranks == 1;
  const bool expect_host_staged_hypre = ranks == 2;
  const std::uint64_t solves = static_cast<std::uint64_t>(trace_solves);
  return native_rank1_configured == expect_native &&
      hypre_host_staged_rank2_configured == expect_host_staged_hypre &&
      native_rank1_calls == (expect_native ? solves : 0) &&
      hypre_calls == (expect_native ? 0 : solves);
}

constexpr bool CudaPreparedTraceSolverRoutingValid(
    const CudaTensorCoreEvidence& tensor, const int trace_solves,
    const generic::CudaEngineKind engine, const int ranks) noexcept
{
  return CudaPreparedTraceSolverRoutingValuesValid(
      engine, ranks, trace_solves,
      tensor.native_rank1_trace_solve_calls,
      tensor.hypre_trace_solve_calls,
      tensor.native_rank1_trace_solver_configured,
      tensor.hypre_host_staged_rank2_trace_solver_configured,
      tensor.trace_solver_accounting_complete);
}

constexpr bool CudaPreparedTensorVolumeEvidenceValuesValid(
    const generic::CudaEngineKind engine,
    const std::uint64_t assembly_launches,
    const std::uint64_t mma_tiles,
    const std::uint64_t mma_tiles_per_assembly) noexcept
{
  if (engine == generic::CudaEngineKind::ordinary)
    return assembly_launches == 0 && mma_tiles == 0 &&
        mma_tiles_per_assembly == 0;
  if (engine != generic::CudaEngineKind::tensor_core ||
      assembly_launches == 0 || mma_tiles_per_assembly == 0)
    return false;
  std::uint64_t expected = 0;
  return TensorVolumeCheckedMultiply(
             assembly_launches, mma_tiles_per_assembly, expected) &&
      mma_tiles == expected;
}

constexpr bool CudaPreparedTensorVolumeEvidenceValid(
    const CudaTensorCoreEvidence& tensor, const CudaMeshShape& shape,
    const generic::CudaEngineKind engine) noexcept
{
  if (engine == generic::CudaEngineKind::ordinary)
    return CudaPreparedTensorVolumeEvidenceValuesValid(
        engine, tensor.volume_assembly_launches,
        tensor.volume_mma_tiles, 0);
  const auto plan = PlanTensorVolumeMmaWork(
      shape.elements, shape.interior_components,
      shape.scalar_interior_dofs, shape.interior_dofs,
      shape.trace_dofs, shape.volume_points, shape.igr_active);
  return plan.valid && CudaPreparedTensorVolumeEvidenceValuesValid(
      engine, tensor.volume_assembly_launches,
      tensor.volume_mma_tiles, plan.mma_tiles);
}

static_assert(CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::tensor_core, 1, 3, 3, 0, true, false, true));
static_assert(CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::ordinary, 1, 3, 0, 3, false, false, true));
static_assert(CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::tensor_core, 2, 3, 0, 3, false, true, true));
static_assert(CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::ordinary, 2, 3, 0, 3, false, true, true));
static_assert(!CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::tensor_core, 1, 3, 0, 3, false, false, true));
static_assert(!CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::ordinary, 1, 3, 3, 0, true, false, true));
static_assert(!CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::tensor_core, 2, 3, 3, 0, true, false, true));
static_assert(!CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::ordinary, 2, 3, 0, 3, false, false, true));
static_assert(!CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::tensor_core, 1, 3, 2, 0, true, false, true));
static_assert(!CudaPreparedTraceSolverRoutingValuesValid(
    generic::CudaEngineKind::tensor_core, 1, 3, 3, 0, true, false, false));
static_assert(CudaPreparedTensorVolumeEvidenceValuesValid(
    generic::CudaEngineKind::ordinary, 0, 0, 0));
static_assert(CudaPreparedTensorVolumeEvidenceValuesValid(
    generic::CudaEngineKind::tensor_core, 3, 1404, 468));
static_assert(!CudaPreparedTensorVolumeEvidenceValuesValid(
    generic::CudaEngineKind::ordinary, 1, 468, 0));
static_assert(!CudaPreparedTensorVolumeEvidenceValuesValid(
    generic::CudaEngineKind::tensor_core, 3, 1403, 468));
static_assert(!CudaPreparedTensorVolumeEvidenceValuesValid(
    generic::CudaEngineKind::tensor_core,
    std::numeric_limits<std::uint64_t>::max(), 1, 2));

} // namespace dg::hdg::runtime::cuda_w8a::detail
