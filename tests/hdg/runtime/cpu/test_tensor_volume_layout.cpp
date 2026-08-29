#include "numerics/hdg/runtime/cuda/tensor_volume_layout.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>

namespace
{
using namespace dg::hdg::runtime::cuda_w8a::detail;

static_assert(TensorVolumeCoefficientStride(false) == 44);
static_assert(TensorVolumeCoefficientStride(true) == 66);
static_assert(TensorVolumeTemporalSlot(0) == 0);
static_assert(TensorVolumeFluxXSlot(3) == 7);
static_assert(TensorVolumeFluxYSlot(0) == 8);
static_assert(TensorVolumeJacobianXSlot(false, 3, 3) == 27);
static_assert(TensorVolumeJacobianYSlot(false, 3, 3) == 43);
static_assert(TensorVolumeJacobianXSlot(true, 3, 4) == 31);
static_assert(TensorVolumeJacobianYSlot(true, 3, 4) == 51);
static_assert(TensorVolumeIgrDqSlot(0, 0) == 57);
static_assert(TensorVolumeIgrDqSlot(2, 2) == 65);
static_assert(TensorVolumeIgrDqSlot(3, 0) == -1);

constexpr TensorVolumeScratchPlan conventional_p1 =
    PlanTensorVolumeScratch(16, 32, 9, false);
constexpr TensorVolumeScratchPlan conventional_p1_q12 =
    PlanTensorVolumeScratch(16, 32, 49, false);
constexpr TensorVolumeScratchPlan conventional_p1_q64 =
    PlanTensorVolumeScratch(16, 32, 64, false);
constexpr TensorVolumeScratchPlan igr_p1 =
    PlanTensorVolumeScratch(28, 40, 16, true);
constexpr TensorVolumeScratchPlan igr_p1_q12 =
    PlanTensorVolumeScratch(28, 40, 49, true);
constexpr TensorVolumeScratchPlan igr_p2_q12 =
    PlanTensorVolumeScratch(63, 60, 49, true);
constexpr TensorVolumeScratchPlan igr_p3_q14 =
    PlanTensorVolumeScratch(112, 80, 64, true);
static_assert(conventional_p1.valid && conventional_p1.capacity_points == 12 &&
              conventional_p1.chunk_points == 9);
static_assert(conventional_p1_q12.valid &&
              conventional_p1_q12.capacity_points == 12 &&
              conventional_p1_q12.chunk_points == 12);
static_assert(conventional_p1_q64.valid &&
              conventional_p1_q64.chunk_points == 12);
static_assert(igr_p1.valid && igr_p1.capacity_points == 17 &&
              igr_p1.chunk_points == 16);
static_assert(igr_p1_q12.valid && igr_p1_q12.chunk_points == 16);
static_assert(igr_p2_q12.valid && igr_p2_q12.chunk_points == 49);
static_assert(igr_p3_q14.valid && igr_p3_q14.chunk_points == 64);
static_assert(!PlanTensorVolumeScratch(4, 1, 4, true).valid);

constexpr TensorVolumeMmaWorkPlan conventional_p1_q12_work =
    PlanTensorVolumeMmaWork(1, 4, 4, 16, 32, 49, false);
constexpr TensorVolumeMmaWorkPlan igr_p1_q12_work =
    PlanTensorVolumeMmaWork(1, 7, 4, 28, 40, 49, true);
constexpr TensorVolumeMmaWorkPlan igr_p2_q12_work =
    PlanTensorVolumeMmaWork(1, 7, 9, 63, 60, 49, true);
constexpr TensorVolumeMmaWorkPlan igr_p3_q14_work =
    PlanTensorVolumeMmaWork(1, 7, 16, 112, 80, 64, true);
static_assert(conventional_p1_q12_work.valid &&
              conventional_p1_q12_work.contraction_launches == 5 &&
              conventional_p1_q12_work.mma_tiles == 468);
static_assert(igr_p1_q12_work.valid &&
              igr_p1_q12_work.contraction_launches == 4 &&
              igr_p1_q12_work.mma_tiles == 1092);
static_assert(igr_p2_q12_work.valid &&
              igr_p2_q12_work.contraction_launches == 1 &&
              igr_p2_q12_work.mma_tiles == 4368);
static_assert(igr_p3_q14_work.valid &&
              igr_p3_q14_work.contraction_launches == 1 &&
              igr_p3_q14_work.mma_tiles == 10080);
static_assert(!PlanTensorVolumeMmaWork(
    std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
    std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
    std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
    true).valid);

constexpr bool CheckAggregateAccounting()
{
  std::uint64_t launches = 0;
  std::uint64_t tiles = 0;
  if (!AccumulateTensorVolumeWork(true, 468, launches, tiles) ||
      launches != 1 || tiles != 468 ||
      !AccumulateTensorVolumeWork(true, 1092, launches, tiles) ||
      launches != 2 || tiles != 1560)
    return false;
  const std::uint64_t retained_launches = launches;
  const std::uint64_t retained_tiles = tiles;
  if (AccumulateTensorVolumeWork(
          true, std::numeric_limits<std::uint64_t>::max(),
          launches, tiles) || launches != retained_launches ||
      tiles != retained_tiles)
    return false;
  launches = std::numeric_limits<std::uint64_t>::max();
  tiles = 0;
  if (AccumulateTensorVolumeWork(true, 1, launches, tiles) ||
      launches != std::numeric_limits<std::uint64_t>::max() || tiles != 0)
    return false;
  launches = 0;
  tiles = 0;
  return AccumulateTensorVolumeWork(false, 0, launches, tiles) &&
      !AccumulateTensorVolumeWork(false, 1, launches, tiles) &&
      launches == 0 && tiles == 0;
}
static_assert(CheckAggregateAccounting());

bool CheckSlotPartition(const bool igr)
{
  const int stride = TensorVolumeCoefficientStride(igr);
  std::array<int, tensor_volume_igr_stride> visits{};
  const auto mark = [&](const int slot)
  {
    if (slot < 0 || slot >= stride) return false;
    ++visits[static_cast<std::size_t>(slot)];
    return true;
  };
  for (int component = 0; component < 4; ++component)
    if (!mark(TensorVolumeTemporalSlot(component)) ||
        !mark(TensorVolumeFluxXSlot(component)) ||
        !mark(TensorVolumeFluxYSlot(component)))
      return false;
  const int columns = TensorVolumeFluxColumns(igr);
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < columns; ++column)
      if (!mark(TensorVolumeJacobianXSlot(igr, row, column)) ||
          !mark(TensorVolumeJacobianYSlot(igr, row, column)))
        return false;
  if (igr)
  {
    for (int slot = tensor_volume_igr_rho_offset;
         slot <= tensor_volume_igr_q_offset; ++slot)
      if (!mark(slot)) return false;
    for (int variable = 0; variable < 3; ++variable)
      for (int feature = 0; feature < 3; ++feature)
        if (!mark(TensorVolumeIgrDqSlot(variable, feature))) return false;
  }
  for (int slot = 0; slot < stride; ++slot)
    if (visits[static_cast<std::size_t>(slot)] != 1) return false;
  return true;
}

bool CheckChunkCoverage(const TensorVolumeScratchPlan& plan,
                        const int points)
{
  if (!plan.valid) return false;
  int covered = 0;
  while (covered < points)
  {
    const int remaining = points - covered;
    const int count = remaining < plan.chunk_points
        ? remaining : plan.chunk_points;
    if (count <= 0 || count > plan.capacity_points ||
        (remaining > plan.chunk_points && count % 4 != 0))
      return false;
    TensorVolumeScratchPlan chunk = plan;
    chunk.chunk_points = count;
    std::size_t last = 0;
    if (!TensorVolumeCoefficientOffset(
            chunk, count - 1, chunk.coefficient_stride - 1, last) ||
        last >= chunk.workspace_doubles)
      return false;
    covered += count;
  }
  return covered == points;
}

template <std::size_t Size>
bool CheckChunkSequence(const TensorVolumeScratchPlan& plan,
                        const int points,
                        const std::array<int, Size>& expected)
{
  if (!plan.valid) return false;
  int covered = 0;
  std::size_t chunk = 0;
  while (covered < points)
  {
    if (chunk >= expected.size()) return false;
    const int remaining = points - covered;
    const int count = remaining < plan.chunk_points
        ? remaining : plan.chunk_points;
    if (count != expected[chunk]) return false;
    covered += count;
    ++chunk;
  }
  return covered == points && chunk == expected.size();
}
} // namespace

int main()
{
  const bool passed = CheckSlotPartition(false) &&
      CheckSlotPartition(true) &&
      CheckChunkCoverage(conventional_p1_q12, 49) &&
      CheckChunkCoverage(conventional_p1_q64, 64) &&
      CheckChunkCoverage(igr_p1_q12, 49) &&
      CheckChunkCoverage(igr_p3_q14, 64) &&
      CheckChunkSequence(conventional_p1_q12, 49,
                         std::array<int, 5>{{12, 12, 12, 12, 1}}) &&
      CheckChunkSequence(igr_p1_q12, 49,
                         std::array<int, 4>{{16, 16, 16, 1}});
  if (!passed)
  {
    std::cerr << "tensor volume layout contract failed\n";
    return 1;
  }
  std::cout << "tensor volume layout contract passed\n";
  return 0;
}
