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
