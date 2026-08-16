#pragma once

// Detail-only retained topology and identity-prime prerequisite for the CUDA
// p1 traced-scalar IGR leaf.  This header deliberately exports neither a
// runtime/session owner nor an assembly/condensation/recovery operation.

#include <mfem.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics
{
class RetainedTopology final
{
public:
  struct Implementation;
  static constexpr int dimension = 2;
  static constexpr int polynomial_order = 1;
  static constexpr int interior_components = 7;
  static constexpr int trace_components = 5;
  static constexpr int interior_values_per_element = 28;
  static constexpr int trace_values_per_element = 40;
  static constexpr int vertices_per_element = 8;

  struct ArenaPlan final
  {
    std::size_t element_count = 0;
    std::size_t interior_values = 0;
    std::size_t trace_values = 0;
    std::size_t integer_logical_bytes = 0;
    std::size_t double_logical_bytes = 0;
    std::size_t integer_physical_bytes = 0;
    std::size_t double_physical_bytes = 0;
    std::int64_t interior_map_offset = 0;
    std::int64_t trace_map_offset = 0;
    std::int64_t vertices_offset = 0;
    std::int64_t volume_quadrature_offset = 0;
    std::int64_t segment_quadrature_offset = 0;
    std::size_t volume_quadrature_values = 0;
    std::size_t segment_quadrature_values = 0;
    std::size_t symbolic_slots = 0;
  };

  // One immutable dense element contribution.  source_slot deliberately
  // names the pre-CSR contribution, rather than a compacted sparse-matrix
  // entry: source_slot = element * 1600 + row * 40 + column.
  struct RetainedCsrSlot final
  {
    std::uint64_t source_slot = 0;
    int signed_row_vdof = -1;
    int signed_column_vdof = -1;
    std::uint64_t global_row = 0;
    std::uint64_t global_column = 0;
    int owner_rank = -1;
    int csr_entry = -1;
    bool csr_entry_is_offd = false;
    int row_sign = 1;
    int column_sign = 1;
  };

  // The plan is rank-major.  Outgoing entries retain local source order
  // within a destination rank; incoming entries retain the sender's source
  // order within its peer-rank segment.  Incoming source_slot is zero because
  // the remote source slot is rank-local; csr_entry identifies its local CSR
  // publication target.
  struct ValueExchangeSlot final
  {
    std::uint64_t source_slot = 0;
    int peer_rank = -1;
    int csr_entry = -1;
    bool csr_entry_is_offd = false;
  };

  RetainedTopology() noexcept;
  ~RetainedTopology();
  RetainedTopology(const RetainedTopology&) = delete;
  RetainedTopology& operator=(const RetainedTopology&) = delete;
  RetainedTopology(RetainedTopology&&) noexcept;
  RetainedTopology& operator=(RetainedTopology&&) noexcept;

  // Collective on MPI_COMM_WORLD after admission.  It accepts an empty local
  // partition and is strictly one-shot whether it succeeds or poisons.
  bool Setup(mfem::ParMesh& mesh) noexcept;

  bool IsReady() const noexcept;
  bool IsPoisoned() const noexcept;
  int LocalElementCount() const noexcept;
  int LocalTraceVSize() const noexcept;
  int LocalTrueTraceSize() const noexcept;
  long long GlobalTrueTraceSize() const noexcept;
  const mfem::ParFiniteElementSpace* TraceSpace() const noexcept;
  const mfem::ParFiniteElementSpace* InteriorSpace() const noexcept;
  const std::vector<int>& SignedElementInteriorVDofs() const noexcept;
  const std::vector<int>& SignedElementTraceVDofs() const noexcept;
  const ArenaPlan& Arena() const noexcept;
  const std::vector<RetainedCsrSlot>& RetainedCsrSlots() const noexcept;
  const std::vector<RetainedCsrSlot>& OwnedCsrSlots() const noexcept;
  const std::vector<ValueExchangeSlot>& OutgoingValueSlots() const noexcept;
  const std::vector<ValueExchangeSlot>& IncomingValueSlots() const noexcept;

  // Collective on MPI_COMM_WORLD.  source_values is indexed by
  // RetainedCsrSlots() and is copied into setup-time buffers only after a world
  // consensus validates every participant.  Received values are published
  // atomically into PublishedIncomingValues() after the collective succeeds.
  // On every failure the prior publication is retained unchanged.
  bool ExchangeOwnedCsrValues(const std::vector<double>& source_values) noexcept;
  const std::vector<double>& PublishedIncomingValues() const noexcept;
  bool IdentityPrimed() const noexcept;
  double PrimeResidual() const noexcept;
  double PrimeMaximumSolutionError() const noexcept;
  bool PrimeUsedDeviceIlu0() const noexcept;

private:
  std::unique_ptr<Implementation> implementation_;
  bool setup_attempted_ = false;
  bool terminal_poisoned_ = false;
};
} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics
