#pragma once

#include "numerics/hdg/hdg_contract.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

namespace mfem { class Mesh; }

namespace dg::hdg::core
{
enum class Status : unsigned char
{
  success, invalid_shape, invalid_view, unsupported_alias, singular,
  non_finite, overflow, allocation_failure, not_ready
};

/** Logical R=A*x+B*lambda-f and G=C*x+D*lambda-g input blocks. */
struct LocalSystemBatchView
{
  LocalBlockShape shape{};
  StridedBlockView<const double> a{}, b{}, c{}, d{}, f{}, g{};
};

struct CondensedBatchView
{
  StridedBlockView<double> schur{};
  StridedBlockView<double> right_hand_side{};
};

/** One stable shape class suitable for host batching and future packing. */
struct ShapeCensusEntry
{
  LocalBlockShape shape{};
  int polynomial_order = 0;
  int components = 0;
  std::size_t element_count = 0;
};
using ShapeCensus = std::vector<ShapeCensusEntry>;

/** Build p1--p4 scalar and planned five-component smooth-Euler shape classes.
 * This uses MFEM metadata only and creates no physical formulation. */
Status BuildRepresentativeShapeCensus(const mfem::Mesh& mesh,
                                      ShapeCensus& census) noexcept;

/** Deterministic FP64 local HDG reference. Failed calls do not modify outputs. */
class HostReferenceEngine
{
public:
  /** Reserve fixed internal staging for one stable local block batch. */
  Status Prepare(const LocalBlockShape& shape, std::size_t batches) noexcept;
  Status Condense(const LocalSystemBatchView& input,
                  const CondensedBatchView& output) noexcept;
  /**
   * Re-condense only a right-hand side against the most recently retained
   * local operator.  This preserves LU(A), B, C, D, and the Schur operator;
   * it replaces the retained f used by Recover and forms g-C A^{-1}f.
   */
  Status RefreshRightHandSide(StridedBlockView<const double> f,
                              StridedBlockView<const double> g,
                              StridedBlockView<double> right_hand_side) noexcept;
  Status ApplySchur(StridedBlockView<const double> trace,
                    StridedBlockView<double> result) const noexcept;
  Status Recover(StridedBlockView<const double> trace,
                 StridedBlockView<double> interior) const noexcept;
  bool ready() const noexcept { return ready_; }
  LocalBlockShape shape() const noexcept { return shape_; }
  std::size_t batch_count() const noexcept { return batch_count_; }

private:
  LocalBlockShape shape_{};
  std::size_t batch_count_ = 0;
  bool ready_ = false;
  // Per batch: pivoted LU(A), pivots, B, C, D, f. Retaining these permits a
  // matrix-free Schur action instead of multiplying by a stored S.
  std::vector<double> lu_, b_, c_, d_, f_;
  std::vector<int> pivots_;
  // Private fixed staging used by prepared repeated Condense/Recover calls.
  std::vector<double> staged_lu_, staged_b_, staged_c_, staged_d_, staged_f_,
      staged_schur_;
  mutable std::vector<double> staged_rhs_;
  mutable std::vector<double> work_, recovered_;
  std::vector<int> staged_pivots_;
};

static_assert(!std::is_polymorphic<HostReferenceEngine>::value,
              "HDG host reference engine must not use virtual dispatch");
}  // namespace dg::hdg::core
