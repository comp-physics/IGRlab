#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include "persistent_smooth_euler_oracle.cuh"

#include <HYPRE_utilities.h>
#include <_hypre_utilities.h>
#include <mfem.hpp>
#include <umpire/ResourceManager.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#error "HDG-RUN-CUDA-01 leaf tests require a native SM120 image"
#endif

namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cuda_w8a;
namespace cuda_detail = dg::hdg::runtime::cuda_w8a::detail;

constexpr cuda_detail::CudaHypreSetupOrderEvidence ExactHypreSetup(
    const std::uint64_t generation, const bool include_mult) noexcept
{
  cuda_detail::CudaHypreSetupOrderEvidence evidence{};
  evidence.matrix_generation = generation;
  evidence.Record(cuda_detail::CudaHypreSetupAction::bind_preconditioner);
  evidence.Record(cuda_detail::CudaHypreSetupAction::set_operator);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_type0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_fill0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_reorder0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_maxiter1);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_tol0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_print0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::rebind_preconditioner);
  evidence.Record(cuda_detail::CudaHypreSetupAction::gmres_settings);
  if (include_mult)
    evidence.Record(cuda_detail::CudaHypreSetupAction::mult);
  return evidence;
}

constexpr cuda_detail::CudaHypreSetupOrderEvidence LegacyPreOperatorIluSetup(
    const std::uint64_t generation) noexcept
{
  cuda_detail::CudaHypreSetupOrderEvidence evidence{};
  evidence.matrix_generation = generation;
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_type0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_fill0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_reorder0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_maxiter1);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_tol0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::ilu_print0);
  evidence.Record(cuda_detail::CudaHypreSetupAction::gmres_settings);
  evidence.Record(cuda_detail::CudaHypreSetupAction::bind_preconditioner);
  evidence.Record(cuda_detail::CudaHypreSetupAction::set_operator);
  evidence.Record(cuda_detail::CudaHypreSetupAction::mult);
  return evidence;
}

constexpr auto exact_configured_hypre = ExactHypreSetup(17, false);
constexpr auto exact_solved_hypre = ExactHypreSetup(17, true);
constexpr auto legacy_pre_operator_hypre = LegacyPreOperatorIluSetup(17);
static_assert(exact_configured_hypre.ConfigurationExactForGeneration(17));
static_assert(!exact_configured_hypre.SolveExactForGeneration(17));
static_assert(exact_solved_hypre.SolveExactForGeneration(17));
static_assert(!exact_solved_hypre.SolveExactForGeneration(18));
static_assert(
    !legacy_pre_operator_hypre.ConfigurationExactForGeneration(17));
static_assert(!legacy_pre_operator_hypre.SolveExactForGeneration(17));

bool HypreIluGenerationSetupRegression() noexcept
{
  const auto configured = ExactHypreSetup(29, false);
  const auto solved = ExactHypreSetup(29, true);
  const auto legacy = LegacyPreOperatorIluSetup(29);
  return configured.ConfigurationExactForGeneration(29) &&
      !configured.SolveExactForGeneration(29) &&
      solved.SolveExactForGeneration(29) &&
      !solved.SolveExactForGeneration(28) &&
      !legacy.ConfigurationExactForGeneration(29) &&
      !legacy.SolveExactForGeneration(29);
}

constexpr std::array<int, 8> TicketWords(
    const CudaFailurePhase phase, const std::uint64_t attempt,
    const std::uint64_t generation, const int device) noexcept
{
  return {{1, 1200, device, static_cast<int>(phase),
           static_cast<int>(attempt), static_cast<int>(attempt >> 32U),
           static_cast<int>(generation),
           static_cast<int>(generation >> 32U)}};
}

constexpr std::uint64_t synthetic_attempt =
    (std::uint64_t{5} << 32U) | 17U;
constexpr std::uint64_t synthetic_generation =
    (std::uint64_t{9} << 32U) | 23U;
constexpr auto exact_ticket_words = TicketWords(
    CudaFailurePhase::trace_solve, synthetic_attempt,
    synthetic_generation, 0);
constexpr auto wrong_attempt_high_words = TicketWords(
    CudaFailurePhase::trace_solve,
    synthetic_attempt ^ (std::uint64_t{1} << 32U),
    synthetic_generation, 0);
constexpr auto wrong_generation_high_words = TicketWords(
    CudaFailurePhase::trace_solve, synthetic_attempt,
    synthetic_generation ^ (std::uint64_t{1} << 32U), 0);
static_assert(cuda_detail::CudaTicketWordsValid(
    exact_ticket_words, CudaFailurePhase::trace_solve, synthetic_attempt,
    synthetic_generation, 0));
static_assert(!cuda_detail::CudaTicketWordsValid(
    wrong_attempt_high_words, CudaFailurePhase::trace_solve,
    synthetic_attempt, synthetic_generation, 0));
static_assert(!cuda_detail::CudaTicketWordsValid(
    wrong_generation_high_words, CudaFailurePhase::trace_solve,
    synthetic_attempt, synthetic_generation, 0));

bool TicketHighHalfRegression() noexcept
{
  return cuda_detail::CudaTicketWordsValid(
             exact_ticket_words, CudaFailurePhase::trace_solve,
             synthetic_attempt, synthetic_generation, 0) &&
      !cuda_detail::CudaTicketWordsValid(
          wrong_attempt_high_words, CudaFailurePhase::trace_solve,
          synthetic_attempt, synthetic_generation, 0) &&
      !cuda_detail::CudaTicketWordsValid(
          wrong_generation_high_words, CudaFailurePhase::trace_solve,
          synthetic_attempt, synthetic_generation, 0);
}

struct TestModelTag
{
  inline static constexpr std::string_view id = "euler.compressible.2d";
};
struct TestFormulationTag
{
  inline static constexpr std::string_view id = "hdg.conventional";
};
struct TestTraceTag
{
  inline static constexpr std::string_view id =
      "mfem.dg_interface.trace";
};
struct TestStabilizationTag
{
  inline static constexpr std::string_view id = "tau.fixed";
};
struct TestTimeTag
{
  inline static constexpr std::string_view id = "backward_euler";
};
struct TestRankOneBackendTag
{
  inline static constexpr std::string_view id = "cuda.rank1.one_gpu";
};
struct TestRankTwoBackendTag
{
  inline static constexpr std::string_view id =
      "cuda.rank2.one_gpu.host_staged";
};
struct TestPrecisionTag
{
  inline static constexpr std::string_view id = "fp64";
};

template <typename Backend>
using TestSelection = StaticRunSelection<
    TestModelTag, TestFormulationTag, TestTraceTag, TestStabilizationTag,
    TestTimeTag, Backend, TestPrecisionTag>;

inline constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
inline constexpr std::uint64_t fnv_prime = 1099511628211ULL;
inline CudaPolicyContext audit_context{};

void MixBytes(std::uint64_t& hash, const void* data,
              const std::size_t size) noexcept
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index)
  {
    hash ^= bytes[index];
    hash *= fnv_prime;
  }
}

template <typename T>
void MixScalar(std::uint64_t& hash, const T value) noexcept
{
  MixBytes(hash, &value, sizeof(value));
}

std::uint64_t IndependentEndpointToken(
    const CudaTransferDirection direction,
    const CudaTransferCategory category, const CudaTransferSite site,
    const CudaPolicyContext& context, const std::uint64_t ordinal,
    const std::uintptr_t source, const std::uintptr_t destination,
    const std::size_t bytes, const std::size_t source_capacity,
    const std::size_t destination_capacity, const bool success) noexcept
{
  std::uint64_t token = fnv_offset;
  MixScalar(token, direction);
  MixScalar(token, category);
  MixScalar(token, site);
  MixScalar(token, context.attempt_sequence);
  MixScalar(token, context.target_step);
  MixScalar(token, context.generation);
  MixScalar(token, context.rank);
  MixScalar(token, context.phase);
  MixScalar(token, ordinal);
  MixScalar(token, source);
  MixScalar(token, destination);
  MixScalar(token, bytes);
  MixScalar(token, source_capacity);
  MixScalar(token, destination_capacity);
  MixScalar(token, success);
  return token;
}

void Account(CudaTransferTotals& totals,
             const CudaTransferDirection direction,
             const std::size_t bytes, const bool success) noexcept
{
  if (totals.event_count == std::numeric_limits<std::uint64_t>::max())
  {
    totals.complete = false;
    return;
  }
  ++totals.event_count;
  totals.complete = totals.complete && success;
  if (!success) { return; }
  const auto add = [&](std::uint64_t& endpoint)
  {
    if (bytes > std::numeric_limits<std::uint64_t>::max() - endpoint)
    {
      totals.complete = false;
      return;
    }
    endpoint += static_cast<std::uint64_t>(bytes);
  };
  if (direction == CudaTransferDirection::host_to_device)
    add(totals.host_to_device_bytes);
  else if (direction == CudaTransferDirection::device_to_host)
    add(totals.device_to_host_bytes);
  else if (direction == CudaTransferDirection::device_to_device)
    add(totals.device_to_device_bytes);
  else
    add(totals.mpi_staging_bytes);
}

struct AuditedAllocator
{
  inline static CudaAllocationLedgerV1 ledger{};
  inline static std::uint64_t ordinal = 0;
  // The production allocator deliberately retains generations across resets.
  inline static std::uint64_t generation = 0;
  inline static std::uint64_t w8_owned_allocation_calls = 0;

  static void ResetLedger() noexcept
  {
    UmpireDeviceAllocatorPolicy::ResetLedger();
    ledger = {};
    ordinal = 0;
  }

  static bool PrepareDeviceArenaAllocator() noexcept
  {
    ++w8_owned_allocation_calls;
    const CudaAllocationLedgerV1 before =
        UmpireDeviceAllocatorPolicy::Ledger();
    const bool prepared =
        UmpireDeviceAllocatorPolicy::PrepareDeviceArenaAllocator();
    const CudaAllocationLedgerV1 after =
        UmpireDeviceAllocatorPolicy::Ledger();
    if (before.overflow || after.overflow || after.count < before.count)
      return false;
    const bool exact_probe_delta = after.count == before.count + 2 &&
        after.observed_count == before.observed_count + 2;
    for (std::size_t index = before.count; index < after.count; ++index)
    {
      const AllocationEventV1& event = after.events[index];
      Record(event.site, static_cast<std::size_t>(event.bytes),
             static_cast<std::size_t>(event.alignment),
             reinterpret_cast<void*>(event.storage_token), event.allocation,
             event.success, event.generation);
    }
    return prepared && exact_probe_delta;
  }

  static void Record(const CudaAllocationSite site, const std::size_t bytes,
                     const std::size_t alignment, void* pointer,
                     const bool allocation, const bool success,
                     const std::uint64_t retained_generation = 0) noexcept
  {
    ++ledger.observed_count;
    if (ledger.count >= ledger.events.size())
    {
      ledger.overflow = true;
      ++ledger.dropped_count;
      return;
    }
    const std::uint64_t event_ordinal = ++ordinal;
    const std::uint64_t event_generation = allocation && success
        ? ++generation : retained_generation;
    const std::size_t index = ledger.count++;
    ledger.events[index] = {
        event_ordinal, site, audit_context, static_cast<std::uint64_t>(bytes),
        static_cast<std::uint64_t>(alignment),
        static_cast<std::uint64_t>(bytes),
        reinterpret_cast<std::uintptr_t>(pointer), event_generation,
        allocation, success};
    std::uint64_t event_digest = fnv_offset;
    MixScalar(event_digest, event_ordinal);
    MixScalar(event_digest, site);
    MixScalar(event_digest, audit_context.attempt_sequence);
    MixScalar(event_digest, audit_context.target_step);
    MixScalar(event_digest, audit_context.generation);
    MixScalar(event_digest, audit_context.rank);
    MixScalar(event_digest, audit_context.phase);
    MixScalar(event_digest, bytes);
    MixScalar(event_digest, alignment);
    MixScalar(event_digest, reinterpret_cast<std::uintptr_t>(pointer));
    MixScalar(event_digest, event_generation);
    MixScalar(event_digest, allocation);
    MixScalar(event_digest, success);
    ledger.ordered_digest ^= event_digest;
    ledger.ordered_digest *= fnv_prime;
  }

  static bool Allocate(const std::size_t bytes, const std::size_t alignment,
                       const CudaAllocationSite site,
                       void*& pointer) noexcept
  {
    ++w8_owned_allocation_calls;
    const bool success = UmpireDeviceAllocatorPolicy::Allocate(
        bytes, alignment, site, pointer);
    Record(site, bytes, alignment, pointer, true, success);
    return success;
  }

  static bool Deallocate(void* pointer,
                         const CudaAllocationSite site) noexcept
  {
    if (pointer == nullptr)
    {
      return UmpireDeviceAllocatorPolicy::Deallocate(pointer, site);
    }
    const auto token = reinterpret_cast<std::uintptr_t>(pointer);
    std::uint64_t bytes = 0;
    std::uint64_t alignment = 0;
    std::uint64_t retained_generation = 0;
    for (std::size_t index = ledger.count; index > 0; --index)
    {
      const AllocationEventV1& event = ledger.events[index - 1];
      if (event.allocation && event.success && event.storage_token == token)
      {
        bytes = event.bytes;
        alignment = event.alignment;
        retained_generation = event.generation;
        break;
      }
    }
    const std::uint64_t before =
        UmpireDeviceAllocatorPolicy::DeallocationEvents();
    const bool policy_success =
        UmpireDeviceAllocatorPolicy::Deallocate(pointer, site);
    const bool success = policy_success &&
        UmpireDeviceAllocatorPolicy::DeallocationEvents() == before + 1;
    Record(site, static_cast<std::size_t>(bytes),
           static_cast<std::size_t>(alignment), pointer, false, success,
           retained_generation);
    return success;
  }

  static std::uint64_t AllocationEvents() noexcept
  { return UmpireDeviceAllocatorPolicy::AllocationEvents(); }
  static std::uint64_t W8OwnedAllocationCalls() noexcept
  { return w8_owned_allocation_calls; }
  static std::uint64_t DeallocationEvents() noexcept
  { return UmpireDeviceAllocatorPolicy::DeallocationEvents(); }
  static bool DevicePoolPolicyValid() noexcept
  { return UmpireDeviceAllocatorPolicy::DevicePoolPolicyValid(); }
  static CudaAllocationLedgerV1 Ledger() noexcept { return ledger; }
};

struct AuditedTransport
{
  inline static CudaTransferLedgerV1 ledger{};
  inline static CudaTransferTotals attempt_totals{};
  inline static std::uint64_t successful_synchronization_events = 0;
  inline static bool fail_checkpoint_trace_once = false;
  inline static bool fail_oracle_export_once = false;
  inline static std::uint64_t injected_oracle_export_failures = 0;

  static void ResetLedger() noexcept
  {
    audit_context = {};
    ledger = {};
    attempt_totals = {};
    successful_synchronization_events = 0;
    fail_checkpoint_trace_once = false;
    fail_oracle_export_once = false;
    injected_oracle_export_failures = 0;
    CudaTransportPolicy::ResetLedger();
  }

  static void BeginAttempt(const CudaPolicyContext& context) noexcept
  {
    audit_context = context;
    attempt_totals = {};
    CudaTransportPolicy::BeginAttempt(context);
  }

  static void SetPhase(const CudaFailurePhase phase,
                       const std::uint64_t generation) noexcept
  {
    audit_context.phase = phase;
    audit_context.generation = generation;
    CudaTransportPolicy::SetPhase(phase, generation);
  }

  static void Record(const CudaTransferDirection direction,
                     const CudaTransferCategory category,
                     const CudaTransferSite site, const void* source,
                     const void* destination, const std::size_t bytes,
                     const std::size_t source_capacity,
                     const std::size_t destination_capacity,
                     const bool success) noexcept
  {
    if (ledger.observed_count == std::numeric_limits<std::uint64_t>::max())
    {
      ledger.overflow = true;
      ledger.totals.complete = false;
      attempt_totals.complete = false;
      return;
    }
    const std::uint64_t event_ordinal = ++ledger.observed_count;
    Account(ledger.totals, direction, bytes, success);
    Account(attempt_totals, direction, bytes, success);
    if (ledger.count >= ledger.events.size())
    {
      ledger.overflow = true;
      ++ledger.dropped_count;
      ledger.totals.complete = false;
      attempt_totals.complete = false;
      return;
    }
    const std::size_t exact_source_capacity =
        source_capacity == 0 ? bytes : source_capacity;
    const std::size_t exact_destination_capacity =
        destination_capacity == 0 ? bytes : destination_capacity;
    const auto source_token = reinterpret_cast<std::uintptr_t>(source);
    const auto destination_token =
        reinterpret_cast<std::uintptr_t>(destination);
    const std::size_t index = ledger.count++;
    ledger.events[index] = {
        event_ordinal, direction, category, site, audit_context,
        static_cast<std::uint64_t>(bytes), source_token, destination_token,
        static_cast<std::uint64_t>(exact_source_capacity),
        static_cast<std::uint64_t>(exact_destination_capacity),
        IndependentEndpointToken(
            direction, category, site, audit_context, event_ordinal,
            source_token, destination_token, bytes, exact_source_capacity,
            exact_destination_capacity, success),
        success};
    ledger.ordered_digest ^= ledger.events[index].endpoint_token;
    ledger.ordered_digest *= fnv_prime;
  }

  static bool HostToDevice(
      void* destination, const void* source, const std::size_t bytes,
      const CudaTransferCategory category, const CudaTransferSite site,
      const std::size_t source_capacity = 0,
      const std::size_t destination_capacity = 0) noexcept
  {
    const bool success = CudaTransportPolicy::HostToDevice(
        destination, source, bytes, category, site, source_capacity,
        destination_capacity);
    Record(CudaTransferDirection::host_to_device, category, site, source,
           destination, bytes, source_capacity, destination_capacity, success);
    return success;
  }

  static bool DeviceToHost(
      void* destination, const void* source, const std::size_t bytes,
      const CudaTransferCategory category, const CudaTransferSite site,
      const std::size_t source_capacity = 0,
      const std::size_t destination_capacity = 0) noexcept
  {
    void* actual_destination = destination;
    if (fail_checkpoint_trace_once &&
        category == CudaTransferCategory::checkpoint_candidate_staging &&
        site == CudaTransferSite::checkpoint_stage_trace)
    {
      actual_destination = nullptr;
      fail_checkpoint_trace_once = false;
    }
    else if (fail_oracle_export_once &&
             category == CudaTransferCategory::oracle_export &&
             site == CudaTransferSite::oracle_export)
    {
      actual_destination = nullptr;
      fail_oracle_export_once = false;
      ++injected_oracle_export_failures;
    }
    const bool success = CudaTransportPolicy::DeviceToHost(
        actual_destination, source, bytes, category, site, source_capacity,
        destination_capacity);
    Record(CudaTransferDirection::device_to_host, category, site, source,
           actual_destination, bytes, source_capacity, destination_capacity,
           success);
    return success;
  }

  static bool DeviceToDevice(
      void* destination, const void* source, const std::size_t bytes,
      const CudaTransferCategory category, const CudaTransferSite site,
      const std::size_t source_capacity = 0,
      const std::size_t destination_capacity = 0) noexcept
  {
    const bool success = CudaTransportPolicy::DeviceToDevice(
        destination, source, bytes, category, site, source_capacity,
        destination_capacity);
    Record(CudaTransferDirection::device_to_device, category, site, source,
           destination, bytes, source_capacity, destination_capacity, success);
    return success;
  }

  static bool RecordMpiStaging(
      const std::size_t bytes, const CudaTransferCategory category,
      const CudaTransferSite site, const void* source,
      const void* destination, const std::size_t source_capacity,
      const std::size_t destination_capacity, const bool success) noexcept
  {
    const bool delegated = CudaTransportPolicy::RecordMpiStaging(
        bytes, category, site, source, destination, source_capacity,
        destination_capacity, success);
    Record(CudaTransferDirection::mpi_host_staging, category, site, source,
           destination, bytes, source_capacity, destination_capacity,
           delegated);
    return delegated;
  }

  static bool Synchronize(const CudaTransferSite site) noexcept
  {
    const bool success = CudaTransportPolicy::Synchronize(site);
    if (success && successful_synchronization_events !=
                       std::numeric_limits<std::uint64_t>::max())
      ++successful_synchronization_events;
    if (ledger.totals.synchronization_events !=
        std::numeric_limits<std::uint64_t>::max())
      ++ledger.totals.synchronization_events;
    else
      ledger.totals.complete = false;
    if (attempt_totals.synchronization_events !=
        std::numeric_limits<std::uint64_t>::max())
      ++attempt_totals.synchronization_events;
    else
      attempt_totals.complete = false;
    ledger.totals.complete = ledger.totals.complete && success;
    attempt_totals.complete = attempt_totals.complete && success;
    return success;
  }

  static CudaTransferTotals Accounting() noexcept { return attempt_totals; }
  static const CudaTransferLedgerV1& Ledger() noexcept { return ledger; }
  static std::uint64_t SuccessfulSynchronizationEvents() noexcept
  { return successful_synchronization_events; }
};

struct AllocatorHighWaterSample
{
  std::size_t mfem = 0;
  std::size_t hypre = 0;
  const char* mfem_identity = "unavailable";
  const char* hypre_identity = "unavailable";
  bool valid = false;
};

AllocatorHighWaterSample SampleAllocatorHighWater() noexcept
{
  AllocatorHighWaterSample sample{};
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    const char* const mfem_name =
        mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    const char* const hypre_name =
        hypre_HandleUmpireDevicePoolName(hypre_handle());
    if (mfem_name == nullptr || hypre_name == nullptr ||
        !manager.isAllocator(mfem_name) || !manager.isAllocator(hypre_name))
      return sample;
    sample.mfem_identity = mfem_name;
    sample.hypre_identity = hypre_name;
    sample.mfem = manager.getAllocator(mfem_name).getHighWatermark();
    sample.hypre = manager.getAllocator(hypre_name).getHighWatermark();
    sample.valid = true;
  }
  catch (...) {}
#endif
  return sample;
}

bool InitialAllocationSampleExact(const std::uint64_t baseline,
                                  const std::uint64_t sample) noexcept
{
  return sample == baseline ||
      (baseline != std::numeric_limits<std::uint64_t>::max() &&
       sample == baseline + 1);
}

bool CollectivelyExact(const bool local) noexcept
{
  const int local_value = local ? 1 : 0;
  int global_value = 0;
  return MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && global_value == 1;
}

bool CollectivelyExactAllocatorState(
    const std::uint64_t allocations,
    const std::uint64_t owned_allocation_calls,
    const std::uint64_t deallocation_events) noexcept
{
  return CollectivelyExact(
      AuditedAllocator::AllocationEvents() == allocations &&
      AuditedAllocator::W8OwnedAllocationCalls() == owned_allocation_calls &&
      AuditedAllocator::DeallocationEvents() == deallocation_events);
}

bool SameContext(const CudaPolicyContext& left,
                 const CudaPolicyContext& right) noexcept
{
  return left.attempt_sequence == right.attempt_sequence &&
      left.target_step == right.target_step &&
      left.generation == right.generation && left.rank == right.rank &&
      left.phase == right.phase;
}

bool SameTotals(const CudaTransferTotals& left,
                const CudaTransferTotals& right) noexcept
{
  return left.host_to_device_bytes == right.host_to_device_bytes &&
      left.device_to_host_bytes == right.device_to_host_bytes &&
      left.device_to_device_bytes == right.device_to_device_bytes &&
      left.mpi_staging_bytes == right.mpi_staging_bytes &&
      left.synchronization_events == right.synchronization_events &&
      left.event_count == right.event_count && left.complete == right.complete;
}

bool SameEvent(const TransferEventV1& left,
               const TransferEventV1& right) noexcept
{
  return left.ordinal == right.ordinal && left.direction == right.direction &&
      left.category == right.category && left.site == right.site &&
      SameContext(left.context, right.context) && left.bytes == right.bytes &&
      left.source_token == right.source_token &&
      left.destination_token == right.destination_token &&
      left.source_capacity == right.source_capacity &&
      left.destination_capacity == right.destination_capacity &&
      left.endpoint_token == right.endpoint_token &&
      left.success == right.success;
}

bool SameEvent(const AllocationEventV1& left,
               const AllocationEventV1& right) noexcept
{
  return left.ordinal == right.ordinal && left.site == right.site &&
      SameContext(left.context, right.context) && left.bytes == right.bytes &&
      left.alignment == right.alignment && left.capacity == right.capacity &&
      left.storage_token == right.storage_token &&
      left.generation == right.generation &&
      left.allocation == right.allocation && left.success == right.success;
}

bool SameLedger(const CudaTransferLedgerV1& left,
                const CudaTransferLedgerV1& right) noexcept
{
  if (left.schema != right.schema || left.capacity != right.capacity ||
      !SameTotals(left.totals, right.totals) || left.count != right.count ||
      left.observed_count != right.observed_count ||
      left.dropped_count != right.dropped_count ||
      left.ordered_digest != right.ordered_digest ||
      left.overflow != right.overflow)
  {
    return false;
  }
  for (std::size_t index = 0; index < left.count; ++index)
  {
    if (!SameEvent(left.events[index], right.events[index])) { return false; }
  }
  return true;
}

bool SameLedger(const CudaAllocationLedgerV1& left,
                const CudaAllocationLedgerV1& right) noexcept
{
  if (left.schema != right.schema || left.capacity != right.capacity ||
      left.count != right.count ||
      left.observed_count != right.observed_count ||
      left.dropped_count != right.dropped_count ||
      left.ordered_digest != right.ordered_digest ||
      left.overflow != right.overflow)
  {
    return false;
  }
  for (std::size_t index = 0; index < left.count; ++index)
  {
    if (!SameEvent(left.events[index], right.events[index])) { return false; }
  }
  return true;
}

bool ExactPreparationProbeEvents(
    const CudaAllocationLedgerV1& ledger) noexcept
{
  const AllocationEventV1* allocation = nullptr;
  const AllocationEventV1* deallocation = nullptr;
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const AllocationEventV1& event = ledger.events[index];
    if (event.site != CudaAllocationSite::mfem_device_pool_probe) { continue; }
    if (!event.success || event.bytes != sizeof(std::max_align_t) ||
        event.alignment != alignof(std::max_align_t) ||
        event.capacity != sizeof(std::max_align_t) ||
        event.storage_token == 0 || event.generation == 0 ||
        event.context.phase != CudaFailurePhase::setup ||
        event.context.rank < 0)
    {
      return false;
    }
    if (event.allocation)
    {
      if (allocation != nullptr) { return false; }
      allocation = &event;
    }
    else
    {
      if (deallocation != nullptr) { return false; }
      deallocation = &event;
    }
  }
  return allocation != nullptr && deallocation != nullptr &&
      allocation->ordinal + 1 == deallocation->ordinal &&
      allocation->storage_token == deallocation->storage_token &&
      allocation->generation == deallocation->generation &&
      SameContext(allocation->context, deallocation->context);
}

std::string PointerTokenText(const std::uintptr_t token)
{
  std::ostringstream stream;
  stream << reinterpret_cast<const void*>(token);
  return stream.str();
}

bool FindMfemBaseKeyForDeviceToken(
    const std::string& pointer_dump, const std::uintptr_t device_token,
    std::string& base_key)
{
  const std::string device = "d_ptr " + PointerTokenText(device_token);
  const std::size_t device_position = pointer_dump.find(device);
  if (device_position == std::string::npos) { return false; }
  const std::size_t line_begin = pointer_dump.rfind('\n', device_position);
  const std::size_t key_position = pointer_dump.find(
      "key ", line_begin == std::string::npos ? 0 : line_begin);
  if (key_position == std::string::npos || key_position > device_position)
    return false;
  const std::size_t key_begin = key_position + 4;
  const std::size_t key_end = pointer_dump.find(',', key_begin);
  if (key_end == std::string::npos || key_end > device_position)
    return false;
  base_key = pointer_dump.substr(key_begin, key_end - key_begin);
  return !base_key.empty();
}

bool HasZeroOffsetAlias(const std::string& alias_dump,
                        const std::string& base_key)
{
  return alias_dump.find(
      "alias: key " + base_key + ", h_ptr " + base_key +
      ", offset 0,") != std::string::npos;
}

bool ExactArenaEventPair(const CudaAllocationLedgerV1& ledger,
                         const CudaAllocationSite site,
                         const std::uintptr_t token,
                         std::uint64_t& allocation_ordinal,
                         std::uint64_t& deallocation_ordinal) noexcept
{
  const AllocationEventV1* allocation = nullptr;
  const AllocationEventV1* deallocation = nullptr;
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const AllocationEventV1& event = ledger.events[index];
    if (event.site != site || event.storage_token != token) { continue; }
    if (!event.success || event.bytes == 0 || event.alignment == 0 ||
        event.capacity != event.bytes || event.generation == 0)
      return false;
    if (event.allocation)
    {
      if (allocation != nullptr) { return false; }
      allocation = &event;
    }
    else
    {
      if (deallocation != nullptr) { return false; }
      deallocation = &event;
    }
  }
  if (allocation == nullptr || deallocation == nullptr ||
      allocation->generation != deallocation->generation ||
      allocation->ordinal >= deallocation->ordinal)
    return false;
  allocation_ordinal = allocation->ordinal;
  deallocation_ordinal = deallocation->ordinal;
  return true;
}

bool ExactPolicyLedgers() noexcept
{
  const auto marker = [](const char* const stage) noexcept
  {
    std::fprintf(stderr,
                 "CUDA_INIT_PROBE stage=policy-ledgers edge=%s result=1\n",
                 stage);
    return true;
  };
  return marker("enter") &&
      ExactPreparationProbeEvents(AuditedAllocator::Ledger()) &&
      marker("audited-preparation-probe") &&
      ExactPreparationProbeEvents(UmpireDeviceAllocatorPolicy::Ledger()) &&
      marker("production-preparation-probe") &&
      SameLedger(AuditedTransport::Ledger(),
                 CudaTransportPolicy::Ledger()) &&
      marker("transfer-ledger-mirror") &&
      SameTotals(AuditedTransport::Accounting(),
                 CudaTransportPolicy::Accounting()) &&
      marker("transfer-totals-mirror") &&
      SameLedger(AuditedAllocator::Ledger(),
                 UmpireDeviceAllocatorPolicy::Ledger()) &&
      marker("allocation-ledger-mirror");
}

bool HypreHostColumnMapCopyFixtures() noexcept
{
  const CudaAllocationLedgerV1 audited_allocation_before =
      AuditedAllocator::Ledger();
  const CudaAllocationLedgerV1 production_allocation_before =
      UmpireDeviceAllocatorPolicy::Ledger();
  const auto audited_transfer_before =
      std::make_unique<CudaTransferLedgerV1>(AuditedTransport::Ledger());
  const auto production_transfer_before =
      std::make_unique<CudaTransferLedgerV1>(CudaTransportPolicy::Ledger());
  const CudaTransferTotals audited_totals_before =
      AuditedTransport::Accounting();
  const CudaTransferTotals production_totals_before =
      CudaTransportPolicy::Accounting();

  HYPRE_BigInt scalar = 19;
  const std::array<HYPRE_BigInt, 3> source{{
      std::numeric_limits<HYPRE_BigInt>::min() + 31,
      -17,
      std::numeric_limits<HYPRE_BigInt>::max() - 31}};
  constexpr HYPRE_BigInt tail_sentinel = 0x5a5a5a5a;
  std::array<HYPRE_BigInt, 4> destination{{
      tail_sentinel, tail_sentinel, tail_sentinel, tail_sentinel}};
  const bool cases_exact =
      cuda_detail::CudaCopyHypreBigIntHost(nullptr, 0, nullptr, 0) &&
      !cuda_detail::CudaCopyHypreBigIntHost(&scalar, 1, nullptr, 1) &&
      !cuda_detail::CudaCopyHypreBigIntHost(nullptr, 1, &scalar, 1) &&
      !cuda_detail::CudaCopyHypreBigIntHost(&scalar, 1, &scalar, -1) &&
      !cuda_detail::CudaCopyHypreBigIntHost(
          destination.data(), 2, source.data(), 3) &&
      cuda_detail::CudaCopyHypreBigIntHost(
          destination.data(), destination.size(), source.data(), 3) &&
      destination[0] == source[0] && destination[1] == source[1] &&
      destination[2] == source[2] && destination[3] == tail_sentinel;

  const CudaAllocationLedgerV1 audited_allocation_after =
      AuditedAllocator::Ledger();
  const CudaAllocationLedgerV1 production_allocation_after =
      UmpireDeviceAllocatorPolicy::Ledger();
  const auto audited_transfer_after =
      std::make_unique<CudaTransferLedgerV1>(AuditedTransport::Ledger());
  const auto production_transfer_after =
      std::make_unique<CudaTransferLedgerV1>(CudaTransportPolicy::Ledger());
  return cases_exact &&
      SameLedger(audited_allocation_before, audited_allocation_after) &&
      SameLedger(production_allocation_before,
                 production_allocation_after) &&
      SameLedger(*audited_transfer_before, *audited_transfer_after) &&
      SameLedger(*production_transfer_before, *production_transfer_after) &&
      SameTotals(audited_totals_before, AuditedTransport::Accounting()) &&
      SameTotals(production_totals_before,
                 CudaTransportPolicy::Accounting());
}

bool SameCoordinates(const StateCoordinates& left,
                     const StateCoordinates& right) noexcept
{
  return std::memcmp(&left.time, &right.time, sizeof(double)) == 0 &&
      left.step == right.step && left.generation == right.generation &&
      left.cursor == right.cursor &&
      std::memcmp(&left.dt, &right.dt, sizeof(double)) == 0;
}

inline constexpr FailureReason checkpoint_cleanup_reason =
    FailureReason::backend_failure;
inline constexpr CudaFailurePhase checkpoint_cleanup_phase =
    CudaFailurePhase::diagnostics;

struct CheckpointPostconditionConsensus
{
  bool local_failure = false;
  bool reduction_complete = false;
  bool cleanup_required = true;
  int failing_ranks = -1;
  FailureReason cleanup_reason = checkpoint_cleanup_reason;
  CudaFailurePhase cleanup_phase = checkpoint_cleanup_phase;
};

CheckpointPostconditionConsensus AgreeCheckpointPostconditionFailure(
    const bool local_failure, const int ranks) noexcept
{
  CheckpointPostconditionConsensus result{};
  result.local_failure = local_failure;
  const int local_count = local_failure ? 1 : 0;
  int failing_ranks = 0;
  result.reduction_complete = MPI_Allreduce(
      &local_count, &failing_ranks, 1, MPI_INT, MPI_SUM,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool count_valid = result.reduction_complete &&
      failing_ranks >= 0 && failing_ranks <= ranks;
  result.failing_ranks = count_valid ? failing_ranks : -1;
  result.cleanup_required = !count_valid || failing_ranks != 0;
  return result;
}

std::uint64_t RangeDigest(const CudaTransferLedgerV1& ledger,
                          const std::uint64_t begin,
                          const std::uint64_t end) noexcept
{
  std::uint64_t digest = fnv_offset;
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const TransferEventV1& event = ledger.events[index];
    if (event.ordinal < begin || event.ordinal > end) { continue; }
    digest ^= event.endpoint_token;
    digest *= fnv_prime;
  }
  return digest;
}

bool ValidRange(const CudaTransferLedgerRangeV1& range,
                const CudaTransferLedgerV1& ledger) noexcept
{
  return range.begin_ordinal != 0 && range.complete &&
      range.end_ordinal <= ledger.observed_count &&
      range.end_ordinal + 1 >= range.begin_ordinal &&
      range.ordered_digest ==
          RangeDigest(ledger, range.begin_ordinal, range.end_ordinal);
}

bool ExactRecordedRange(const CudaTransferLedgerRangeV1& range,
                        const CudaTransferLedgerV1& ledger) noexcept
{
  return range.begin_ordinal != 0 &&
      range.end_ordinal <= ledger.observed_count &&
      range.end_ordinal + 1 >= range.begin_ordinal &&
      range.ordered_digest ==
          RangeDigest(ledger, range.begin_ordinal, range.end_ordinal);
}

// Stable private diagnostic v1. Bit set means the frozen packet predicate
// failed. These masks never participate in test acceptance.
enum : std::uint64_t
{
  ckpt_stage_return=1ULL<<0, ckpt_schema=1ULL<<1,
  ckpt_rank_identity=1ULL<<2, ckpt_attempt_sequence=1ULL<<3,
  ckpt_facade_incarnation=1ULL<<4, ckpt_artifact_flags=1ULL<<5,
  ckpt_descriptor_interior_hash=1ULL<<6,
  ckpt_descriptor_trace_hash=1ULL<<7, ckpt_local_interior_hash=1ULL<<8,
  ckpt_local_trace_hash=1ULL<<9, ckpt_interior_count=1ULL<<10,
  ckpt_trace_count=1ULL<<11, ckpt_staging_generation=1ULL<<12,
  ckpt_source_tokens=1ULL<<13, ckpt_recomputed_interior_hash=1ULL<<14,
  ckpt_recomputed_trace_hash=1ULL<<15, ckpt_range_bounds=1ULL<<16,
  ckpt_range_digest=1ULL<<17, ckpt_ledger_mirror=1ULL<<18,
  ckpt_event_shape_context=1ULL<<19,
  ckpt_sub_d2d_interior=1ULL<<0, ckpt_sub_d2d_trace=1ULL<<1,
  ckpt_sub_first_sync=1ULL<<2, ckpt_sub_d2h_interior=1ULL<<3,
  ckpt_sub_d2h_trace=1ULL<<4, ckpt_sub_second_sync=1ULL<<5,
  ckpt_sub_host_interior_hash=1ULL<<6,
  ckpt_sub_host_trace_hash=1ULL<<7, ckpt_sub_collective_return=1ULL<<8,
  ckpt_sub_range_seal=1ULL<<9
};
struct CheckpointDiagnosticV1
{
  std::uint64_t failure_mask=0, substage_mask=0, event_count=0, sync_delta=0;
  std::uint64_t sync_success_delta=0;
  StateHash interior{}, trace{};
  bool mirror=false;
};

CheckpointDiagnosticV1 BuildCheckpointDiagnosticV1(
    bool staged, const CudaSerializedCheckpointV1& c,
    const CudaPrecommitSnapshot& s, int ranks, int rank,
    const CudaTransferLedgerV1& before,
    const CudaTransferLedgerV1& after,
    const std::uint64_t before_sync_successes,
    const std::uint64_t after_sync_successes, bool mirror) noexcept
{
  CheckpointDiagnosticV1 d{}; d.mirror=mirror;
  const bool ic=c.interior_count<=c.interior.size();
  const bool tc=c.owned_true_trace_count<=c.owned_true_trace.size();
  if(ic)d.interior=cuda_detail::HashHostBytes(c.interior.data(),
      c.interior_count*sizeof(double));
  if(tc)d.trace=cuda_detail::HashHostBytes(c.owned_true_trace.data(),
      c.owned_true_trace_count*sizeof(double));
  const bool ih=ic&&d.interior==c.local_interior_hash;
  const bool th=tc&&d.trace==c.local_trace_hash;
  const bool bounds=c.staging_range.begin_ordinal!=0&&
      c.staging_range.complete&&c.staging_range.end_ordinal<=after.observed_count&&
      c.staging_range.end_ordinal>=c.staging_range.begin_ordinal-1;
  const bool digest=c.staging_range.ordered_digest==RangeDigest(after,
      c.staging_range.begin_ordinal,c.staging_range.end_ordinal);
  bool event_ok[4]={}; unsigned seen[4]={}; bool shape=true;
  const std::uint64_t first=before.observed_count+1;
  const std::uint64_t before_sync_calls=
      before.totals.synchronization_events;
  const std::uint64_t after_sync_calls=after.totals.synchronization_events;
  bool sync_audit_valid=after_sync_calls>=before_sync_calls&&
      after_sync_successes>=before_sync_successes&&
      before_sync_successes<=before_sync_calls&&
      after_sync_successes<=after_sync_calls;
  if(sync_audit_valid)
  {
    d.sync_delta=after_sync_calls-before_sync_calls;
    d.sync_success_delta=after_sync_successes-before_sync_successes;
    sync_audit_valid=d.sync_success_delta<=d.sync_delta;
  }
  const bool first_sync=sync_audit_valid&&d.sync_delta>=1&&
      d.sync_success_delta>=1;
  const bool second_sync=sync_audit_valid&&d.sync_delta>=2&&
      d.sync_success_delta>=2;
  for(std::size_t i=0;i<after.count;++i)
  {
    const auto& e=after.events[i]; if(e.ordinal<first)continue;
    ++d.event_count; int k=-1;
    if(e.site==CudaTransferSite::checkpoint_stage_interior)
      k=e.direction==CudaTransferDirection::device_to_device?0:
        e.direction==CudaTransferDirection::device_to_host?2:-1;
    if(e.site==CudaTransferSite::checkpoint_stage_trace)
      k=e.direction==CudaTransferDirection::device_to_device?1:
        e.direction==CudaTransferDirection::device_to_host?3:-1;
    if(k<0){shape=false;continue;} ++seen[k];
    const bool interior=(k&1)==0, d2h=k>=2;
    const std::uint64_t bytes=(interior?c.interior_count:
        c.owned_true_trace_count)*sizeof(double);
    const std::uintptr_t token=interior?c.interior_source_token:
        c.trace_source_token;
    const std::uintptr_t host=interior?
        reinterpret_cast<std::uintptr_t>(c.interior.data()):
        reinterpret_cast<std::uintptr_t>(c.owned_true_trace.data());
    const std::uint64_t capacity=d2h?(interior?c.interior.size():
        c.owned_true_trace.size())*sizeof(double):bytes;
    event_ok[k]=e.success&&e.category==
        CudaTransferCategory::checkpoint_candidate_staging&&e.bytes==bytes&&
        e.source_token!=0&&e.destination_token==(d2h?host:token)&&
        (!d2h||e.source_token==token)&&e.source_capacity==bytes&&
        e.destination_capacity==capacity&&e.endpoint_token!=0&&
        e.context.attempt_sequence==s.attempt.sequence&&
        e.context.target_step==s.target.step&&
        e.context.generation==s.target.generation&&e.context.rank==rank&&
        e.context.phase==CudaFailurePhase::after_recovery_before_commit;
    shape=shape&&event_ok[k];
  }
  for(int k=0;k<4;++k){event_ok[k]=event_ok[k]&&seen[k]==1;shape=shape&&event_ok[k];}
  shape=shape&&d.event_count==4;
  auto fail=[&](bool p,std::uint64_t b){if(!p)d.failure_mask|=b;};
  fail(staged,ckpt_stage_return); fail(c.schema==1,ckpt_schema);
  fail(c.source_rank==rank&&c.rank_count==ranks,ckpt_rank_identity);
  fail(c.attempt.sequence==s.attempt.sequence,ckpt_attempt_sequence);
  fail(c.facade_incarnation!=0,ckpt_facade_incarnation);
  fail(c.candidate_frozen&&c.checkpoint_staged&&!c.output_staged&&
       !c.successful_commit,ckpt_artifact_flags);
  fail(c.descriptor.interior_hash==s.candidate_interior_hash,
       ckpt_descriptor_interior_hash);
  fail(c.descriptor.trace_hash==s.candidate_trace_hash,
       ckpt_descriptor_trace_hash);
  fail(c.local_interior_hash==s.candidate_local_interior_hash,
       ckpt_local_interior_hash);
  fail(c.local_trace_hash==s.candidate_local_trace_hash,ckpt_local_trace_hash);
  fail(c.interior_count==s.interior_coefficients,ckpt_interior_count);
  fail(c.owned_true_trace_count==s.true_trace_coefficients,ckpt_trace_count);
  fail(c.staging_generation==s.target.generation,ckpt_staging_generation);
  fail(c.interior_source_token&&c.trace_source_token&&
       c.interior_source_token!=c.trace_source_token,ckpt_source_tokens);
  fail(ih,ckpt_recomputed_interior_hash); fail(th,ckpt_recomputed_trace_hash);
  fail(bounds,ckpt_range_bounds); fail(digest,ckpt_range_digest);
  fail(mirror,ckpt_ledger_mirror); fail(shape,ckpt_event_shape_context);
  // Report the first inferred failed substage from this staging call only.
  // Lifetime transfer completeness may already be false from an older stage.
  if(!event_ok[0])d.substage_mask=ckpt_sub_d2d_interior;
  else if(!event_ok[1])d.substage_mask=ckpt_sub_d2d_trace;
  else if(!first_sync)d.substage_mask=ckpt_sub_first_sync;
  else if(!event_ok[2])d.substage_mask=ckpt_sub_d2h_interior;
  else if(!event_ok[3])d.substage_mask=ckpt_sub_d2h_trace;
  else if(!second_sync)d.substage_mask=ckpt_sub_second_sync;
  else if(!ih)d.substage_mask=ckpt_sub_host_interior_hash;
  else if(!th)d.substage_mask=ckpt_sub_host_trace_hash;
  else if(!staged)d.substage_mask=ckpt_sub_collective_return;
  else if(!bounds||!digest)d.substage_mask=ckpt_sub_range_seal;
  return d;
}

void PrintCheckpointHash(std::ostream& out,const StateHash& h) noexcept
{
  const auto flags=out.flags(); const char fill=out.fill();
  out<<std::hex<<std::setfill('0');
  for(unsigned char b:h.bytes)out<<std::setw(2)<<static_cast<unsigned>(b);
  out.flags(flags); out.fill(fill);
}

void ReportCheckpointDiagnosticV1(const char* id,int rank,bool retry,
    std::uint64_t target,
    const CudaSerializedCheckpointV1& c,const CudaPrecommitSnapshot& s,
    const CudaTransferLedgerV1& before,const CudaTransferLedgerV1& after,
    const CudaTransferTotals& totals,const CheckpointDiagnosticV1& d) noexcept
{
  std::cerr<<"CKPT_DIAG v=1 case="<<(id?id:"null")<<" rank="<<rank
    <<" retry="<<retry<<" target="<<target<<" fail_mask=0x"<<std::hex
    <<d.failure_mask<<" substage_mask=0x"<<d.substage_mask<<std::dec
    <<" attempts="<<s.attempt.sequence<<'/'<<c.attempt.sequence<<" counts="
    <<c.interior_count<<'/'<<c.owned_true_trace_count<<" expected="
    <<s.interior_coefficients<<'/'<<s.true_trace_coefficients<<" generation="
    <<c.staging_generation<<'/'<<s.target.generation<<" descriptor=";
  PrintCheckpointHash(std::cerr,c.descriptor.interior_hash);std::cerr<<'/';
  PrintCheckpointHash(std::cerr,c.descriptor.trace_hash);std::cerr<<" local=";
  PrintCheckpointHash(std::cerr,c.local_interior_hash);std::cerr<<'/';
  PrintCheckpointHash(std::cerr,c.local_trace_hash);std::cerr<<" recomputed=";
  PrintCheckpointHash(std::cerr,d.interior);std::cerr<<'/';
  PrintCheckpointHash(std::cerr,d.trace);
  std::cerr<<" tokens=0x"<<std::hex<<c.interior_source_token<<"/0x"
    <<c.trace_source_token<<std::dec<<" range="<<c.staging_range.begin_ordinal
    <<':'<<c.staging_range.end_ordinal<<"/0x"<<std::hex
    <<c.staging_range.ordered_digest<<std::dec<<'/'<<c.staging_range.complete
    <<" ledger="<<after.count<<'/'<<after.observed_count<<'/'
    <<after.dropped_count<<'/'<<after.overflow<<"/0x"<<std::hex
    <<after.ordered_digest<<std::dec<<" attempt="<<totals.event_count<<'/'
    <<totals.synchronization_events<<'/'<<totals.complete<<" stage="
    <<d.event_count<<'/'<<d.sync_delta<<'/'<<d.sync_success_delta
    <<" mirror="<<d.mirror<<'\n';
  std::size_t n=0;const auto first=c.staging_range.begin_ordinal?
      c.staging_range.begin_ordinal:before.observed_count+1;
  const auto last=c.staging_range.end_ordinal>=first?
      c.staging_range.end_ordinal:after.observed_count;
  for(std::size_t i=0;i<after.count&&n<4;++i){const auto& e=after.events[i];
    if(e.ordinal<first||e.ordinal>last)continue;
    std::cerr<<"CKPT_EVENT v=1 case="<<(id?id:"null")<<" rank="<<rank
      <<" ordinal="<<e.ordinal<<" d/c/s="<<static_cast<unsigned>(e.direction)
      <<'/'<<static_cast<unsigned>(e.category)<<'/'
      <<static_cast<unsigned>(e.site)<<" context="<<e.context.attempt_sequence
      <<'/'<<e.context.target_step<<'/'<<e.context.generation<<'/'
      <<e.context.rank<<'/'<<static_cast<unsigned>(e.context.phase)
      <<" bytes="<<e.bytes<<" src/dst=0x"<<std::hex<<e.source_token<<"/0x"
      <<e.destination_token<<std::dec<<" capacities="<<e.source_capacity<<'/'
      <<e.destination_capacity<<" endpoint=0x"<<std::hex<<e.endpoint_token
      <<std::dec<<" success="<<e.success<<'\n';++n;}
}

enum class AffineLocalPhase : int
{
  unknown = 0,
  staging = 1,
  prepared = 2,
  provisional = 3,
  consumed = 4
};

enum class AffineExitAction : unsigned char
{
  terminal_no_cleanup,
  abort_prepared,
  revert_commit,
  verify_consumed
};

struct AffinePhaseCensus
{
  AffineLocalPhase local = AffineLocalPhase::unknown;
  int minimum = 0;
  int maximum = 0;
  int ranks = 0;
  bool complete = false;
  AffineExitAction action = AffineExitAction::terminal_no_cleanup;
};

AffineExitAction SelectAffineExitAction(
    const int minimum, const int maximum, const int observed_ranks,
    const int expected_ranks, const bool reductions_complete) noexcept
{
  if (!reductions_complete || observed_ranks != expected_ranks ||
      minimum != maximum)
    return AffineExitAction::terminal_no_cleanup;
  if (minimum == static_cast<int>(AffineLocalPhase::staging) ||
      minimum == static_cast<int>(AffineLocalPhase::prepared))
    return AffineExitAction::abort_prepared;
  if (minimum == static_cast<int>(AffineLocalPhase::provisional))
    return AffineExitAction::revert_commit;
  if (minimum == static_cast<int>(AffineLocalPhase::consumed))
    return AffineExitAction::verify_consumed;
  return AffineExitAction::terminal_no_cleanup;
}

AffinePhaseCensus CensusAffinePhase(
    const AffineLocalPhase local, const int expected_ranks) noexcept
{
  AffinePhaseCensus census{};
  census.local = local;
  const int local_value = static_cast<int>(local);
  census.minimum = local_value;
  census.maximum = local_value;
  const bool minimum_ready = MPI_Allreduce(
      &local_value, &census.minimum, 1, MPI_INT, MPI_MIN,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool maximum_ready = MPI_Allreduce(
      &local_value, &census.maximum, 1, MPI_INT, MPI_MAX,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool size_ready =
      MPI_Comm_size(MPI_COMM_WORLD, &census.ranks) == MPI_SUCCESS;
  census.complete = minimum_ready && maximum_ready && size_ready;
  census.action = SelectAffineExitAction(
      census.minimum, census.maximum, census.ranks, expected_ranks,
      census.complete);
  return census;
}

bool MixedPreparedProvisionalCensusFixture(
    const int rank, const int ranks) noexcept
{
  if (ranks != 2 || (rank != 0 && rank != 1)) { return false; }
  const AffineLocalPhase local = rank == 0
      ? AffineLocalPhase::prepared : AffineLocalPhase::provisional;
  const AffinePhaseCensus census = CensusAffinePhase(local, ranks);
  const bool exact = census.complete && census.ranks == 2 &&
      census.minimum == static_cast<int>(AffineLocalPhase::prepared) &&
      census.maximum == static_cast<int>(AffineLocalPhase::provisional) &&
      census.action == AffineExitAction::terminal_no_cleanup;
  std::fprintf(stderr,
      "CUDA_AFFINE_SYNTHETIC rank=%d case=mixed-prepared-provisional "
      "local=%d minimum=%d maximum=%d action=%u cleanup_called=0 "
      "local_result=%s\n",
      rank, static_cast<int>(local), census.minimum, census.maximum,
      static_cast<unsigned>(census.action), exact ? "PASS" : "FAIL");
  return exact;
}

AffineLocalPhase PhaseAfterPrepareStep(
    const bool returned, const CudaStepEvidence& evidence) noexcept
{
  if (returned) { return AffineLocalPhase::staging; }
  if (evidence.status == CudaRunStatus::retry_available ||
      evidence.status == CudaRunStatus::committed)
    return AffineLocalPhase::consumed;
  return AffineLocalPhase::unknown;
}

AffineLocalPhase PhaseAfterPrepareCommit(
    const bool returned, const CudaStepEvidence& evidence) noexcept
{
  if (evidence.status == CudaRunStatus::retry_available ||
      evidence.status == CudaRunStatus::committed)
    return AffineLocalPhase::consumed;
  if (returned) { return AffineLocalPhase::prepared; }
  return AffineLocalPhase::unknown;
}

AffineLocalPhase PhaseAfterCommitPrepared(
    const bool returned, const CudaStepEvidence& evidence) noexcept
{
  if (returned && evidence.status != CudaRunStatus::retry_available)
    return AffineLocalPhase::provisional;
  if (evidence.status == CudaRunStatus::retry_available ||
      evidence.status == CudaRunStatus::committed)
    return AffineLocalPhase::consumed;
  return AffineLocalPhase::unknown;
}

AffineLocalPhase PhaseAfterFinalizeCommit(
    const CudaStepEvidence& evidence) noexcept
{
  if (evidence.status == CudaRunStatus::retry_available ||
      evidence.status == CudaRunStatus::committed)
    return AffineLocalPhase::consumed;
  return AffineLocalPhase::unknown;
}

template <typename Engine> bool ApplyUniformAffineCleanup(
    Engine& engine, CudaSerializedCheckpointV1& checkpoint,
    CudaStepEvidence& evidence, const CudaPrecommitSnapshot& snapshot,
    const char* const id, const int rank,
    const FailureReason cleanup_reason,
    const CudaFailurePhase cleanup_phase,
    const AffineExitAction cleanup_action) noexcept
{
  const auto* contract = engine.contract();
  const FailureReason original_reason = evidence.contract_reason;
  const CudaFailurePhase original_phase = evidence.phase;
  const std::uint64_t allocations_before =
      AuditedAllocator::AllocationEvents();
  const std::uint64_t owned_calls_before =
      AuditedAllocator::W8OwnedAllocationCalls();
  const std::uint64_t deallocations_before =
      AuditedAllocator::DeallocationEvents();
  const std::uint64_t commits_before = contract == nullptr
      ? std::numeric_limits<std::uint64_t>::max()
      : contract->telemetry().commits;
  const bool cleanup_return = cleanup_action ==
          AffineExitAction::revert_commit
      ? engine.RevertCommit(checkpoint, cleanup_reason, evidence)
      : engine.AbortPrepared(checkpoint, cleanup_reason, evidence);
  const bool policy_ledgers_exact = ExactPolicyLedgers();
  const bool allocator_exact =
      AuditedAllocator::AllocationEvents() == allocations_before &&
      AuditedAllocator::W8OwnedAllocationCalls() == owned_calls_before &&
      AuditedAllocator::DeallocationEvents() == deallocations_before;
  const bool transfer_evidence_exact =
      SameTotals(evidence.transfers, AuditedTransport::Accounting()) &&
      evidence.transfer_range.complete == evidence.transfers.complete &&
      ExactRecordedRange(
          evidence.transfer_range, AuditedTransport::Ledger());
  const bool cleanup_valid_local = cleanup_return && contract != nullptr &&
      evidence.status == CudaRunStatus::retry_available &&
      evidence.contract_reason == cleanup_reason &&
      evidence.collective_consensus ==
          (cleanup_reason != FailureReason::collective_protocol) &&
      evidence.retry_available && !evidence.committed &&
      evidence.coordinates_preserved && evidence.committed_bytes_preserved &&
      evidence.before_interior_hash == evidence.after_interior_hash &&
      evidence.before_trace_hash == evidence.after_trace_hash &&
      SameCoordinates(evidence.after, snapshot.committed) &&
      SameCoordinates(contract->coordinates(), snapshot.committed) &&
      contract->status() == RunStatus::retry_available &&
      contract->telemetry().commits == commits_before &&
      !checkpoint.candidate_frozen && !checkpoint.checkpoint_staged &&
      !checkpoint.output_staged && !checkpoint.successful_commit &&
      allocator_exact && transfer_evidence_exact && policy_ledgers_exact;
  const bool cleanup_valid_collective =
      CollectivelyExact(cleanup_valid_local);
  std::fprintf(stderr,
      "CUDA_AFFINE_CLEANUP case=%s rank=%d mode=%u return=%u "
      "local=%u collective=%u "
      "original=%u/%u cleanup=%u/%u coordinates=%u bytes=%u "
      "hashes=%u allocator=%u ledgers=%u flags=%u/%u/%u/%u\n",
      id != nullptr ? id : "null", rank,
      static_cast<unsigned>(cleanup_action), cleanup_return ? 1U : 0U,
      cleanup_valid_local ? 1U : 0U,
      cleanup_valid_collective ? 1U : 0U,
      static_cast<unsigned>(original_reason),
      static_cast<unsigned>(original_phase),
      static_cast<unsigned>(cleanup_reason),
      static_cast<unsigned>(cleanup_phase),
      evidence.coordinates_preserved ? 1U : 0U,
      evidence.committed_bytes_preserved ? 1U : 0U,
      evidence.before_interior_hash == evidence.after_interior_hash &&
              evidence.before_trace_hash == evidence.after_trace_hash
          ? 1U : 0U,
      allocator_exact ? 1U : 0U,
      transfer_evidence_exact && policy_ledgers_exact ? 1U : 0U,
      checkpoint.candidate_frozen ? 1U : 0U,
      checkpoint.checkpoint_staged ? 1U : 0U,
      checkpoint.output_staged ? 1U : 0U,
      checkpoint.successful_commit ? 1U : 0U);
  return cleanup_valid_collective;
}

template <typename Engine> bool ResolveUnexpectedAffineExit(
    Engine& engine, CudaSerializedCheckpointV1& checkpoint,
    CudaStepEvidence& evidence, const CudaPrecommitSnapshot& snapshot,
    const char* const id, const int ranks, const int rank,
    const FailureReason cleanup_reason,
    const CudaFailurePhase cleanup_phase,
    const AffineLocalPhase local_phase) noexcept
{
  const AffinePhaseCensus census = CensusAffinePhase(local_phase, ranks);
  std::fprintf(stderr,
      "CUDA_AFFINE_CENSUS case=%s rank=%d local=%d minimum=%d "
      "maximum=%d ranks=%d/%d complete=%u action=%u "
      "reason=%u phase=%u flags=%u/%u/%u/%u\n",
      id != nullptr ? id : "null", rank,
      static_cast<int>(census.local), census.minimum, census.maximum,
      census.ranks, ranks, census.complete ? 1U : 0U,
      static_cast<unsigned>(census.action),
      static_cast<unsigned>(cleanup_reason),
      static_cast<unsigned>(cleanup_phase),
      checkpoint.candidate_frozen ? 1U : 0U,
      checkpoint.checkpoint_staged ? 1U : 0U,
      checkpoint.output_staged ? 1U : 0U,
      checkpoint.successful_commit ? 1U : 0U);
  if (census.action == AffineExitAction::terminal_no_cleanup)
  {
    evidence.status = CudaRunStatus::terminal_failure;
    evidence.phase = CudaFailurePhase::terminal_evidence;
    evidence.contract_reason = FailureReason::collective_protocol;
    evidence.collective_consensus = false;
    std::fprintf(stderr,
        "CUDA_AFFINE_TERMINAL case=%s rank=%d cleanup_called=0 "
        "destroy_engine=1 status=%u reason=%u\n",
        id != nullptr ? id : "null", rank,
        static_cast<unsigned>(evidence.status),
        static_cast<unsigned>(evidence.contract_reason));
    return false;
  }
  if (census.action == AffineExitAction::verify_consumed)
  {
    const auto* contract = engine.contract();
    const bool retry_consumed = contract != nullptr &&
        contract->status() == RunStatus::retry_available &&
        evidence.status == CudaRunStatus::retry_available &&
        evidence.retry_available && !evidence.committed &&
        !checkpoint.candidate_frozen && !checkpoint.checkpoint_staged &&
        !checkpoint.output_staged && !checkpoint.successful_commit;
    const bool commit_consumed = contract != nullptr &&
        contract->status() == RunStatus::committed &&
        evidence.status == CudaRunStatus::committed &&
        evidence.committed && !evidence.retry_available &&
        checkpoint.successful_commit;
    const bool consumed_local = retry_consumed || commit_consumed;
    const bool consumed_collective = CollectivelyExact(consumed_local);
    std::fprintf(stderr,
        "CUDA_AFFINE_CONSUMED case=%s rank=%d retry=%u commit=%u "
        "cleanup_called=0 local=%s collective=%s\n",
        id != nullptr ? id : "null", rank,
        retry_consumed ? 1U : 0U, commit_consumed ? 1U : 0U,
        consumed_local ? "PASS" : "FAIL",
        consumed_collective ? "PASS" : "FAIL");
    return consumed_collective;
  }
  return ApplyUniformAffineCleanup(
      engine, checkpoint, evidence, snapshot, id, rank, cleanup_reason,
      cleanup_phase, census.action);
}

template <typename Engine> bool StagePreparedCheckpointDiagnosed(Engine& e,
    CudaSerializedCheckpointV1& c,CudaStepEvidence& evidence,
    const CudaPrecommitSnapshot& s,int ranks,bool retry,std::uint64_t target,
    const char* id,int rank,const bool cleanup_on_failure=true) noexcept
{
  const auto before=AuditedTransport::Ledger();
  const auto before_sync_successes=
      AuditedTransport::SuccessfulSynchronizationEvents();
  const bool staged=e.StagePreparedCheckpoint(c,evidence);
  const auto after=AuditedTransport::Ledger();
  const auto after_sync_successes=
      AuditedTransport::SuccessfulSynchronizationEvents();
  const CheckpointPostconditionConsensus consensus =
      AgreeCheckpointPostconditionFailure(!staged, ranks);
  if(consensus.cleanup_required)
  {
    const bool mirror=ExactPolicyLedgers();
    const auto d=BuildCheckpointDiagnosticV1(
      staged,c,s,ranks,rank,before,after,before_sync_successes,
      after_sync_successes,mirror);
    ReportCheckpointDiagnosticV1(id,rank,retry,target,c,s,before,after,
      AuditedTransport::Accounting(),d);
    if(cleanup_on_failure)
      (void)ResolveUnexpectedAffineExit(
          e,c,evidence,s,id,ranks,rank,checkpoint_cleanup_reason,
          checkpoint_cleanup_phase,
          AffineLocalPhase::staging);
    return false;
  }
  return true;
}

StateHash IndependentCheckpointHash(const void* data,
                                    const std::size_t bytes) noexcept
{
  StateHash result{};
  const auto* input = static_cast<const unsigned char*>(data);
  std::array<std::uint64_t, 4> hash{{
      1469598103934665603ULL,
      1099511628211ULL ^ 0x9e3779b97f4a7c15ULL,
      1469598103934665603ULL ^ 0xd6e8feb86659fd93ULL,
      1099511628211ULL ^ 0xa0761d6478bd642fULL}};
  for (std::size_t index = 0; index < bytes; ++index)
  {
    for (std::size_t lane = 0; lane < hash.size(); ++lane)
    {
      hash[lane] ^= static_cast<std::uint64_t>(input[index] + 17U * lane);
      hash[lane] *= 1099511628211ULL + 2ULL * lane;
    }
  }
  std::memcpy(result.bytes.data(), hash.data(), sizeof(hash));
  return result;
}

std::uint64_t IndependentCheckpointRangeDigest(
    const CudaTransferLedgerV1& ledger, const std::uint64_t begin,
    const std::uint64_t end) noexcept
{
  std::uint64_t digest = fnv_offset;
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const TransferEventV1& event = ledger.events[index];
    if (event.ordinal < begin || event.ordinal > end) { continue; }
    digest ^= event.endpoint_token;
    digest *= fnv_prime;
  }
  return digest;
}

struct CheckpointDiagnosticFixtureV1
{
  CudaSerializedCheckpointV1 checkpoint{};
  CudaPrecommitSnapshot snapshot{};
  CudaTransferLedgerV1 before{};
  CudaTransferLedgerV1 after{};
  std::uint64_t before_sync_successes = 0;
  std::uint64_t after_sync_successes = 0;
  bool staged = true;
  bool mirror = true;
};

void RefreshCheckpointEventEndpoint(TransferEventV1& event) noexcept
{
  event.endpoint_token = IndependentEndpointToken(
      event.direction, event.category, event.site, event.context,
      event.ordinal, event.source_token, event.destination_token,
      event.bytes, event.source_capacity, event.destination_capacity,
      event.success);
}

void InitializeCheckpointDiagnosticFixture(
    CheckpointDiagnosticFixtureV1& fixture) noexcept
{
  fixture.checkpoint = {};
  fixture.snapshot = {};
  fixture.staged = true;
  fixture.mirror = true;
  fixture.before_sync_successes = 0;
  fixture.after_sync_successes = 2;
  fixture.before.totals = {};
  fixture.before.count = 0;
  fixture.before.observed_count = 0;
  fixture.before.dropped_count = 0;
  fixture.before.ordered_digest = fnv_offset;
  fixture.before.overflow = false;
  fixture.after.totals = {};
  fixture.after.count = 0;
  fixture.after.observed_count = 0;
  fixture.after.dropped_count = 0;
  fixture.after.ordered_digest = fnv_offset;
  fixture.after.overflow = false;
  auto& checkpoint = fixture.checkpoint;
  auto& snapshot = fixture.snapshot;
  auto& after = fixture.after;
  snapshot.attempt.sequence = 7;
  snapshot.target.step = 1;
  snapshot.target.generation = 1;
  snapshot.interior_coefficients = 1;
  snapshot.true_trace_coefficients = 1;
  snapshot.candidate_interior_hash.bytes[0] = 0x31;
  snapshot.candidate_trace_hash.bytes[0] = 0x32;
  checkpoint.schema = 1;
  checkpoint.source_rank = 0;
  checkpoint.rank_count = 1;
  checkpoint.attempt.sequence = snapshot.attempt.sequence;
  checkpoint.facade_incarnation = 9;
  checkpoint.descriptor.interior_hash = snapshot.candidate_interior_hash;
  checkpoint.descriptor.trace_hash = snapshot.candidate_trace_hash;
  checkpoint.interior_count = 1;
  checkpoint.owned_true_trace_count = 1;
  checkpoint.interior_source_token = 0x1000;
  checkpoint.trace_source_token = 0x2000;
  checkpoint.staging_generation = snapshot.target.generation;
  checkpoint.interior[0] = 1.25;
  checkpoint.owned_true_trace[0] = -2.5;
  checkpoint.local_interior_hash = IndependentCheckpointHash(
      checkpoint.interior.data(), sizeof(double));
  checkpoint.local_trace_hash = IndependentCheckpointHash(
      checkpoint.owned_true_trace.data(), sizeof(double));
  snapshot.candidate_local_interior_hash = checkpoint.local_interior_hash;
  snapshot.candidate_local_trace_hash = checkpoint.local_trace_hash;
  checkpoint.candidate_frozen = true;
  checkpoint.checkpoint_staged = true;

  const std::array<CudaTransferDirection, 4> directions{{
      CudaTransferDirection::device_to_device,
      CudaTransferDirection::device_to_device,
      CudaTransferDirection::device_to_host,
      CudaTransferDirection::device_to_host}};
  const std::array<CudaTransferSite, 4> sites{{
      CudaTransferSite::checkpoint_stage_interior,
      CudaTransferSite::checkpoint_stage_trace,
      CudaTransferSite::checkpoint_stage_interior,
      CudaTransferSite::checkpoint_stage_trace}};
  for (std::size_t index = 0; index < directions.size(); ++index)
  {
    TransferEventV1& event = after.events[index];
    event = {};
    const bool interior = (index & 1U) == 0;
    const bool device_to_host = index >= 2;
    const std::uintptr_t token = interior
        ? checkpoint.interior_source_token : checkpoint.trace_source_token;
    const std::uintptr_t host = interior
        ? reinterpret_cast<std::uintptr_t>(checkpoint.interior.data())
        : reinterpret_cast<std::uintptr_t>(
              checkpoint.owned_true_trace.data());
    event.ordinal = index + 1;
    event.direction = directions[index];
    event.category = CudaTransferCategory::checkpoint_candidate_staging;
    event.site = sites[index];
    event.context.attempt_sequence = snapshot.attempt.sequence;
    event.context.target_step = snapshot.target.step;
    event.context.generation = snapshot.target.generation;
    event.context.rank = 0;
    event.context.phase = CudaFailurePhase::after_recovery_before_commit;
    event.bytes = sizeof(double);
    event.source_token = device_to_host ? token : 0x3000 + index;
    event.destination_token = device_to_host ? host : token;
    event.source_capacity = sizeof(double);
    event.destination_capacity = device_to_host
        ? (interior ? checkpoint.interior.size()
                    : checkpoint.owned_true_trace.size()) * sizeof(double)
        : sizeof(double);
    event.success = true;
    RefreshCheckpointEventEndpoint(event);
  }
  after.count = 4;
  after.observed_count = 4;
  after.totals.device_to_device_bytes = 2 * sizeof(double);
  after.totals.device_to_host_bytes = 2 * sizeof(double);
  after.totals.synchronization_events = 2;
  after.totals.event_count = 4;
  after.totals.complete = true;
  checkpoint.staging_range.begin_ordinal = 1;
  checkpoint.staging_range.end_ordinal = 4;
  checkpoint.staging_range.ordered_digest =
      IndependentCheckpointRangeDigest(after, 1, 4);
  checkpoint.staging_range.complete = true;
}

void RestrictCheckpointFixtureToD2D(
    CheckpointDiagnosticFixtureV1& fixture) noexcept
{
  fixture.staged = false;
  fixture.checkpoint.checkpoint_staged = false;
  fixture.after.count = 2;
  fixture.after.observed_count = 2;
  fixture.after.totals.device_to_host_bytes = 0;
  fixture.after.totals.synchronization_events = 1;
  fixture.after_sync_successes = 0;
  fixture.after.totals.event_count = 2;
  fixture.after.totals.complete = false;
  fixture.checkpoint.staging_range.end_ordinal = 2;
  fixture.checkpoint.staging_range.complete = false;
  fixture.checkpoint.staging_range.ordered_digest =
      IndependentCheckpointRangeDigest(fixture.after, 1, 2);
}

bool CheckpointDiagnosticExactFixtures() noexcept
{
  auto fixture = std::make_unique<CheckpointDiagnosticFixtureV1>();
  InitializeCheckpointDiagnosticFixture(*fixture);
  auto diagnose = [&]() noexcept
  {
    return BuildCheckpointDiagnosticV1(
        fixture->staged, fixture->checkpoint, fixture->snapshot, 1, 0,
        fixture->before, fixture->after, fixture->before_sync_successes,
        fixture->after_sync_successes, fixture->mirror);
  };
  const CheckpointDiagnosticV1 baseline = diagnose();
  if (baseline.failure_mask != 0 || baseline.substage_mask != 0 ||
      baseline.event_count != 4 || baseline.sync_delta != 2 ||
      baseline.sync_success_delta != 2)
    return false;

  for (unsigned bit = 0; bit < 20; ++bit)
  {
    InitializeCheckpointDiagnosticFixture(*fixture);
    auto& checkpoint = fixture->checkpoint;
    auto& snapshot = fixture->snapshot;
    switch (bit)
    {
      case 0: fixture->staged = false; break;
      case 1: checkpoint.schema = 0; break;
      case 2: checkpoint.source_rank = -1; break;
      case 3: ++checkpoint.attempt.sequence; break;
      case 4: checkpoint.facade_incarnation = 0; break;
      case 5: checkpoint.output_staged = true; break;
      case 6: ++snapshot.candidate_interior_hash.bytes[0]; break;
      case 7: ++snapshot.candidate_trace_hash.bytes[0]; break;
      case 8: ++snapshot.candidate_local_interior_hash.bytes[0]; break;
      case 9: ++snapshot.candidate_local_trace_hash.bytes[0]; break;
      case 10: ++snapshot.interior_coefficients; break;
      case 11: ++snapshot.true_trace_coefficients; break;
      case 12: ++checkpoint.staging_generation; break;
      case 13:
        checkpoint.trace_source_token = checkpoint.interior_source_token;
        fixture->after.events[1].destination_token =
            checkpoint.trace_source_token;
        fixture->after.events[3].source_token = checkpoint.trace_source_token;
        RefreshCheckpointEventEndpoint(fixture->after.events[1]);
        RefreshCheckpointEventEndpoint(fixture->after.events[3]);
        checkpoint.staging_range.ordered_digest =
            IndependentCheckpointRangeDigest(fixture->after, 1, 4);
        break;
      case 14: checkpoint.interior[0] += 1.0; break;
      case 15: checkpoint.owned_true_trace[0] += 1.0; break;
      case 16: checkpoint.staging_range.complete = false; break;
      case 17: checkpoint.staging_range.ordered_digest ^= 1; break;
      case 18: fixture->mirror = false; break;
      case 19:
        fixture->after.events[0].endpoint_token = 0;
        checkpoint.staging_range.ordered_digest =
            IndependentCheckpointRangeDigest(fixture->after, 1, 4);
        break;
    }
    if (diagnose().failure_mask != (std::uint64_t{1} << bit))
      return false;
  }

  for (unsigned bit = 0; bit < 10; ++bit)
  {
    InitializeCheckpointDiagnosticFixture(*fixture);
    switch (bit)
    {
      case 0:
      case 1:
      case 3:
      case 4:
        fixture->after.events[bit < 2 ? bit : bit - 1].success = false;
        fixture->after.totals.complete = false;
        break;
      case 2:
        RestrictCheckpointFixtureToD2D(*fixture);
        break;
      case 5:
        fixture->staged = false;
        fixture->checkpoint.checkpoint_staged = false;
        fixture->after.totals.complete = false;
        fixture->after_sync_successes = 1;
        break;
      case 6: fixture->checkpoint.interior[0] += 1.0; break;
      case 7: fixture->checkpoint.owned_true_trace[0] += 1.0; break;
      case 8: fixture->staged = false; break;
      case 9: fixture->checkpoint.staging_range.complete = false; break;
    }
    if (diagnose().substage_mask != (std::uint64_t{1} << bit))
      return false;
  }

  // A prior lifetime failure must not shadow a clean current stage.  The
  // current stage contributes exactly two calls and two successes.
  const std::array<std::uint64_t, 5> contaminated_expected{{
      0, std::uint64_t{1} << 6, std::uint64_t{1} << 7,
      std::uint64_t{1} << 8, std::uint64_t{1} << 9}};
  for (std::size_t fault = 0; fault < contaminated_expected.size(); ++fault)
  {
    InitializeCheckpointDiagnosticFixture(*fixture);
    fixture->before.totals.complete = false;
    fixture->after.totals.complete = false;
    fixture->before.totals.synchronization_events = 5;
    fixture->after.totals.synchronization_events = 7;
    fixture->before_sync_successes = 4;
    fixture->after_sync_successes = 6;
    if (fault == 1) fixture->checkpoint.interior[0] += 1.0;
    if (fault == 2) fixture->checkpoint.owned_true_trace[0] += 1.0;
    if (fault == 3) fixture->staged = false;
    if (fault == 4) fixture->checkpoint.staging_range.complete = false;
    const CheckpointDiagnosticV1 diagnostic = diagnose();
    if (diagnostic.substage_mask != contaminated_expected[fault] ||
        diagnostic.sync_delta != 2 || diagnostic.sync_success_delta != 2)
      return false;
  }
  return true;
}

bool ExactTickets(const CudaStepEvidence& evidence) noexcept
{
  if (evidence.device_ticket_count == 0 ||
      evidence.device_ticket_count > evidence.device_tickets.size())
  {
    return false;
  }
  bool assembly = false;
  bool condensation = false;
  bool trace_solve = false;
  bool recovery = false;
  for (std::size_t index = 0; index < evidence.device_ticket_count; ++index)
  {
    const CudaDeviceTicket& ticket = evidence.device_tickets[index];
    if (!ticket.kernel_executed || ticket.compiled_architecture != 1200 ||
        ticket.device_ordinal != 0 ||
        ticket.attempt_sequence != evidence.attempt.sequence ||
        ticket.generation != evidence.attempt.committed_generation + 1)
    {
      return false;
    }
    assembly = assembly || ticket.phase == CudaFailurePhase::assembly;
    condensation =
        condensation || ticket.phase == CudaFailurePhase::condensation;
    trace_solve = trace_solve || ticket.phase == CudaFailurePhase::trace_solve;
    recovery = recovery || ticket.phase == CudaFailurePhase::recovery;
  }
  return assembly && condensation && trace_solve && recovery;
}

bool ExactAttemptEvidence(const CudaStepEvidence& evidence,
                          const int ranks) noexcept
{
  const CudaTransferLedgerV1& ledger = AuditedTransport::Ledger();
  if (!ExactPolicyLedgers() || !ExactTickets(evidence) ||
      !SameTotals(evidence.transfers, AuditedTransport::Accounting()) ||
      !ValidRange(evidence.transfer_range, ledger) ||
      !evidence.transfers.complete ||
      evidence.transfers.synchronization_events == 0 ||
      (ranks == 1 && evidence.transfers.mpi_staging_bytes != 0) ||
      (ranks == 2 && evidence.transfers.mpi_staging_bytes == 0))
  {
    return false;
  }
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const TransferEventV1& event = ledger.events[index];
    if (event.ordinal < evidence.transfer_range.begin_ordinal ||
        event.ordinal > evidence.transfer_range.end_ordinal)
    {
      continue;
    }
    if (!event.success ||
        event.context.attempt_sequence != evidence.attempt.sequence ||
        event.context.target_step != evidence.attempt.committed_step + 1 ||
        event.context.generation != evidence.attempt.committed_generation + 1)
    {
      return false;
    }
  }
  return true;
}

bool ExactFailedStagingEvidence(const CudaStepEvidence& evidence,
                                const int ranks) noexcept
{
  const CudaTransferLedgerV1& ledger = AuditedTransport::Ledger();
  if (!ExactPolicyLedgers() || !ExactTickets(evidence) ||
      !SameTotals(evidence.transfers, AuditedTransport::Accounting()) ||
      evidence.transfer_range.begin_ordinal == 0 ||
      evidence.transfer_range.complete ||
      evidence.transfer_range.end_ordinal > ledger.observed_count ||
      evidence.transfer_range.end_ordinal + 1 <
          evidence.transfer_range.begin_ordinal ||
      evidence.transfer_range.ordered_digest != RangeDigest(
          ledger, evidence.transfer_range.begin_ordinal,
          evidence.transfer_range.end_ordinal) ||
      evidence.transfers.complete || evidence.committed ||
      !evidence.retry_available || !evidence.coordinates_preserved ||
      !evidence.committed_bytes_preserved ||
      evidence.contract_reason != FailureReason::backend_failure ||
      (ranks == 1 && evidence.transfers.mpi_staging_bytes != 0) ||
      (ranks == 2 && evidence.transfers.mpi_staging_bytes == 0))
  {
    return false;
  }
  std::uint64_t failed_checkpoint_trace_events = 0;
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const TransferEventV1& event = ledger.events[index];
    if (event.ordinal < evidence.transfer_range.begin_ordinal ||
        event.ordinal > evidence.transfer_range.end_ordinal)
    {
      continue;
    }
    if (event.context.attempt_sequence != evidence.attempt.sequence ||
        event.context.target_step != evidence.attempt.committed_step + 1 ||
        event.context.generation != evidence.attempt.committed_generation + 1)
    {
      return false;
    }
    if (!event.success)
    {
      if (event.direction != CudaTransferDirection::device_to_host ||
          event.category !=
              CudaTransferCategory::checkpoint_candidate_staging ||
          event.site != CudaTransferSite::checkpoint_stage_trace ||
          event.destination_token != 0)
      {
        return false;
      }
      ++failed_checkpoint_trace_events;
    }
  }
  return failed_checkpoint_trace_events == 1;
}

// Stable SetupValid failure-mask mapping for this private diagnostic TU:
// bits 0..15 respectively cover status, origin, ranks, device, MFEM, HYPRE,
// same-GPU, trace semantics, host staging, no fallback, reuse booleans,
// nonzero tokens, distinct slots, post-baseline counts, scope flags, and the
// independent policy ledgers.
enum SetupFailureBit : std::uint64_t
{
  setup_status_failure = std::uint64_t{1} << 0U,
  setup_origin_failure = std::uint64_t{1} << 1U,
  setup_ranks_failure = std::uint64_t{1} << 2U,
  setup_device_failure = std::uint64_t{1} << 3U,
  setup_mfem_failure = std::uint64_t{1} << 4U,
  setup_hypre_failure = std::uint64_t{1} << 5U,
  setup_same_gpu_failure = std::uint64_t{1} << 6U,
  setup_trace_semantics_failure = std::uint64_t{1} << 7U,
  setup_host_staging_failure = std::uint64_t{1} << 8U,
  setup_no_fallback_failure = std::uint64_t{1} << 9U,
  setup_reuse_booleans_failure = std::uint64_t{1} << 10U,
  setup_tokens_failure = std::uint64_t{1} << 11U,
  setup_distinct_slots_failure = std::uint64_t{1} << 12U,
  setup_post_baseline_counts_failure = std::uint64_t{1} << 13U,
  setup_scope_flags_failure = std::uint64_t{1} << 14U,
  setup_policy_ledgers_failure = std::uint64_t{1} << 15U
};

bool SetupValid(const CudaRunEvidence& setup, const int ranks,
                std::uint64_t& failure_mask) noexcept
{
  std::fprintf(stderr,
      "CUDA_INIT_PROBE rank=%d stage=setup-valid edge=enter result=0\n",
      setup.rank);
  failure_mask = 0;
  if (!(setup.status == CudaRunStatus::ready))
    failure_mask |= setup_status_failure;
  if (!(setup.setup_origin == FailureReason::none))
    failure_mask |= setup_origin_failure;
  if (!(setup.ranks == ranks))
    failure_mask |= setup_ranks_failure;
  if (!(setup.device == 0))
    failure_mask |= setup_device_failure;
  if (!setup.mfem_device_valid)
    failure_mask |= setup_mfem_failure;
  if (!setup.hypre_cuda_umpire_device)
    failure_mask |= setup_hypre_failure;
  if (!setup.same_physical_gpu)
    failure_mask |= setup_same_gpu_failure;
  if (!setup.signed_owner_ghost_semantics)
    failure_mask |= setup_trace_semantics_failure;
  if (!(ranks == 1 || setup.host_staged_rank2))
    failure_mask |= setup_host_staging_failure;
  if (!setup.no_host_solver_fallback)
    failure_mask |= setup_no_fallback_failure;
  if (!(setup.reuse.persistent_graph && setup.reuse.persistent_slots &&
        setup.reuse.persistent_workspace))
  {
    failure_mask |= setup_reuse_booleans_failure;
  }
  if (!(setup.reuse.topology_token != 0 &&
        setup.reuse.double_arena_token != 0 &&
        setup.reuse.integer_arena_token != 0 &&
        setup.reuse.matrix_token != 0 && setup.reuse.factor_token != 0 &&
        setup.reuse.workspace_token != 0 &&
        setup.reuse.interior_slot_tokens[0] != 0 &&
        setup.reuse.interior_slot_tokens[1] != 0 &&
        setup.reuse.trace_slot_tokens[0] != 0 &&
        setup.reuse.trace_slot_tokens[1] != 0))
  {
    failure_mask |= setup_tokens_failure;
  }
  if (!(setup.reuse.interior_slot_tokens[0] !=
            setup.reuse.interior_slot_tokens[1] &&
        setup.reuse.trace_slot_tokens[0] !=
            setup.reuse.trace_slot_tokens[1]))
  {
    failure_mask |= setup_distinct_slots_failure;
  }
  if (!(setup.reuse.post_baseline_allocation_events == 0 &&
        setup.reuse.post_baseline_w8_owned_allocation_calls == 0))
  {
    failure_mask |= setup_post_baseline_counts_failure;
  }
  if (!(setup.reuse.observes_w8_owned_calls_and_pool_growth_only &&
        !setup.reuse.opaque_pool_reuse_churn_observed))
  {
    failure_mask |= setup_scope_flags_failure;
  }
  std::fprintf(stderr,
      "CUDA_INIT_PROBE rank=%d stage=setup-valid-scalars edge=after "
      "result=%u mask=0x%llx\n",
      setup.rank, failure_mask == 0 ? 1U : 0U,
      static_cast<unsigned long long>(failure_mask));
  std::fprintf(stderr,
      "CUDA_INIT_PROBE rank=%d stage=setup-valid-ledgers edge=before "
      "result=0\n",
      setup.rank);
  const bool policy_ledgers_exact = ExactPolicyLedgers();
  std::fprintf(stderr,
      "CUDA_INIT_PROBE rank=%d stage=setup-valid-ledgers edge=after "
      "result=%u\n",
      setup.rank, policy_ledgers_exact ? 1U : 0U);
  if (!policy_ledgers_exact)
    failure_mask |= setup_policy_ledgers_failure;
  std::fprintf(stderr,
      "CUDA_INIT_PROBE rank=%d stage=setup-valid edge=return result=%u "
      "mask=0x%llx\n",
      setup.rank, failure_mask == 0 ? 1U : 0U,
      static_cast<unsigned long long>(failure_mask));
  return failure_mask == 0;
}

void ReportSetupFailure(const char* const context, const bool initialized,
                        const std::uint64_t failure_mask,
                        const CudaRunEvidence& setup,
                        const int expected_ranks) noexcept
{
  if (setup.rank != 0) { return; }
  std::cerr << "SETUP_FAIL initialized=" << (initialized ? 1 : 0)
            << " mask=0x" << std::hex << failure_mask << std::dec
            << " context=" << context
            << " status=" << static_cast<int>(setup.status)
            << " origin=" << static_cast<int>(setup.setup_origin)
            << " ranks=" << setup.ranks
            << " expected_ranks=" << expected_ranks
            << " device=" << setup.device
            << " flags=" << (setup.mfem_device_valid ? 1 : 0)
            << (setup.hypre_cuda_umpire_device ? 1 : 0)
            << (setup.same_physical_gpu ? 1 : 0)
            << (setup.signed_owner_ghost_semantics ? 1 : 0)
            << (setup.host_staged_rank2 ? 1 : 0)
            << (setup.no_host_solver_fallback ? 1 : 0)
            << " reuse=" << (setup.reuse.persistent_graph ? 1 : 0)
            << (setup.reuse.persistent_slots ? 1 : 0)
            << (setup.reuse.persistent_workspace ? 1 : 0)
            << " post_counts="
            << setup.reuse.post_baseline_allocation_events << ','
            << setup.reuse.post_baseline_w8_owned_allocation_calls
            << " scope="
            << (setup.reuse.observes_w8_owned_calls_and_pool_growth_only
                    ? 1 : 0)
            << (setup.reuse.opaque_pool_reuse_churn_observed ? 1 : 0)
            << " tokens=topology:0x" << std::hex
            << setup.reuse.topology_token
            << ",double:0x" << setup.reuse.double_arena_token
            << ",integer:0x" << setup.reuse.integer_arena_token
            << ",matrix:0x" << setup.reuse.matrix_token
            << ",factor:0x" << setup.reuse.factor_token
            << ",workspace:0x" << setup.reuse.workspace_token
            << ",interior0:0x" << setup.reuse.interior_slot_tokens[0]
            << ",interior1:0x" << setup.reuse.interior_slot_tokens[1]
            << ",trace0:0x" << setup.reuse.trace_slot_tokens[0]
            << ",trace1:0x" << setup.reuse.trace_slot_tokens[1]
            << std::dec << '\n';
}

CudaConfig Config(const int ranks, const bool oracle = false) noexcept
{
  CudaConfig config{};
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.capture_oracle = oracle;
  return config;
}

CudaConfig FunctionalConfig(const int ranks,
                            const bool inject_step14) noexcept
{
  CudaConfig config = Config(ranks, false);
  config.order = functional_polynomial_order;
  config.mesh_n = functional_elements_per_direction;
  config.injection.enabled = inject_step14;
  config.injection.target_step = functional_injected_step;
  config.injection.source_rank = ranks - 1;
  return config;
}

bool SamePersistentIdentity(const CudaReuseEvidence& initial,
                            const CudaReuseEvidence& current,
                            const int ranks) noexcept
{
  const bool same_interior_slots =
      (initial.interior_slot_tokens == current.interior_slot_tokens) ||
      (initial.interior_slot_tokens[0] == current.interior_slot_tokens[1] &&
       initial.interior_slot_tokens[1] == current.interior_slot_tokens[0]);
  const bool same_trace_slots =
      (initial.trace_slot_tokens == current.trace_slot_tokens) ||
      (initial.trace_slot_tokens[0] == current.trace_slot_tokens[1] &&
       initial.trace_slot_tokens[1] == current.trace_slot_tokens[0]);
  return initial.topology_token != 0 &&
      initial.topology_token == current.topology_token &&
      initial.double_arena_token == current.double_arena_token &&
      initial.integer_arena_token == current.integer_arena_token &&
      initial.matrix_token == current.matrix_token &&
      initial.factor_token == current.factor_token &&
      initial.workspace_token == current.workspace_token &&
      same_interior_slots && same_trace_slots &&
      current.post_baseline_w8_owned_allocation_calls == 0 &&
      current.post_baseline_allocation_events <=
          static_cast<std::uint64_t>(ranks == 2 ? 1 : 0) &&
      current.observes_w8_owned_calls_and_pool_growth_only &&
      !current.opaque_pool_reuse_churn_observed &&
      current.persistent_graph && current.persistent_slots &&
      current.persistent_workspace;
}

template <typename Backend>
using TestEngine = PersistentCudaSmoothEulerEngine<
    TestSelection<Backend>, AuditedAllocator, AuditedTransport,
    CudaMpiCollectivePolicy>;

std::uint64_t numerical_engine_initialization_count = 0;

std::uint64_t RecordNumericalEngineInitialization(
    const char* const context) noexcept
{
  if (numerical_engine_initialization_count !=
      std::numeric_limits<std::uint64_t>::max())
  {
    ++numerical_engine_initialization_count;
  }
  std::fprintf(stderr,
      "CUDA_ENGINE_INIT_ORDINAL context=%s ordinal=%llu\n",
      context != nullptr ? context : "null",
      static_cast<unsigned long long>(
          numerical_engine_initialization_count));
  return numerical_engine_initialization_count;
}

template <typename Engine>
bool InitializePositive(Engine& engine, CudaRunEvidence& setup,
                        const int ranks,
                        const char* const context,
                        std::uint64_t* const initialization_ordinal = nullptr)
    noexcept
{
  const std::uint64_t ordinal =
      RecordNumericalEngineInitialization(context);
  if (initialization_ordinal != nullptr)
    *initialization_ordinal = ordinal;
  std::fprintf(stderr,
      "CUDA_INIT_PROBE stage=engine-initialize edge=before context=%s "
      "result=0\n",
      context != nullptr ? context : "null");
  const bool initialized = engine.Initialize(setup);
  std::fprintf(stderr,
      "CUDA_INIT_PROBE rank=%d stage=engine-initialize edge=after "
      "context=%s result=%u\n",
      setup.rank, context != nullptr ? context : "null",
      initialized ? 1U : 0U);
  std::uint64_t failure_mask = 0;
  std::fprintf(stderr,
      "CUDA_INIT_PROBE rank=%d stage=setup-valid-call edge=before "
      "context=%s result=0\n",
      setup.rank, context != nullptr ? context : "null");
  const bool setup_valid = SetupValid(setup, ranks, failure_mask);
  std::fprintf(stderr,
      "CUDA_INIT_PROBE rank=%d stage=setup-valid-call edge=after "
      "context=%s result=%u mask=0x%llx\n",
      setup.rank, context != nullptr ? context : "null",
      setup_valid ? 1U : 0U,
      static_cast<unsigned long long>(failure_mask));
  const bool initialized_local = initialized && setup_valid;
  const bool initialized_collective = CollectivelyExact(initialized_local);
  std::fprintf(stderr,
      "CUDA_INIT_CONSENSUS rank=%d context=%s initialized=%u setup=%u "
      "local=%s collective=%s\n",
      setup.rank, context != nullptr ? context : "null",
      initialized ? 1U : 0U, setup_valid ? 1U : 0U,
      initialized_local ? "PASS" : "FAIL",
      initialized_collective ? "PASS" : "FAIL");
  if (!initialized_collective)
  {
    if (!initialized_local)
      ReportSetupFailure(context, initialized, failure_mask, setup, ranks);
    return false;
  }
  return true;
}

enum class ArenaReleaseFault : unsigned char
{
  none,
  vector_alias,
  double_registration,
  integer_registration,
  integer_raw_retryable,
  double_raw_retryable,
  integer_raw_quarantined,
  double_raw_quarantined
};

struct ArenaReleaseFixtureOperations
{
  bool integer_present = true;
  bool double_present = true;
  ArenaReleaseFault fault = ArenaReleaseFault::none;
  bool fault_consumed = false;
  bool wrappers_inert = false;
  bool views_cleared = false;
  std::array<unsigned char, 12> calls{};
  std::size_t call_count = 0;

  void Record(const unsigned char call) noexcept
  { if (call_count < calls.size()) calls[call_count++] = call; }
  bool FailOnce(const ArenaReleaseFault selected) noexcept
  {
    if (fault != selected || fault_consumed) { return false; }
    fault_consumed = true;
    return true;
  }
  bool HasIntegerArena() const noexcept { return integer_present; }
  bool HasDoubleArena() const noexcept { return double_present; }
  void Marker(const char*, const char*,
              const cuda_detail::CudaArenaReleaseState&,
              std::uint64_t) const noexcept {}
  bool ReleaseVectorAlias() noexcept
  { Record(1); return !FailOnce(ArenaReleaseFault::vector_alias); }
  bool ReleaseDoubleRegistration() noexcept
  { Record(2); return !FailOnce(ArenaReleaseFault::double_registration); }
  bool ReleaseIntegerRegistration() noexcept
  { Record(3); return !FailOnce(ArenaReleaseFault::integer_registration); }
  void MakeWrappersInert() noexcept
  { Record(4); wrappers_inert = true; }
  void ClearViews() noexcept
  { Record(5); views_cleared = true; }
  cuda_detail::CudaRawReleaseResult ReleaseIntegerArena() noexcept
  {
    Record(6);
    if (FailOnce(ArenaReleaseFault::integer_raw_retryable))
      return cuda_detail::CudaRawReleaseResult::retryable;
    if (FailOnce(ArenaReleaseFault::integer_raw_quarantined))
      return cuda_detail::CudaRawReleaseResult::quarantined;
    integer_present = false;
    return cuda_detail::CudaRawReleaseResult::released;
  }
  cuda_detail::CudaRawReleaseResult ReleaseDoubleArena() noexcept
  {
    Record(7);
    if (FailOnce(ArenaReleaseFault::double_raw_retryable))
      return cuda_detail::CudaRawReleaseResult::retryable;
    if (FailOnce(ArenaReleaseFault::double_raw_quarantined))
      return cuda_detail::CudaRawReleaseResult::quarantined;
    double_present = false;
    return cuda_detail::CudaRawReleaseResult::released;
  }
};

bool ExactArenaReleaseCalls(const ArenaReleaseFixtureOperations& operations,
                            const std::initializer_list<unsigned char> expected)
    noexcept
{
  if (operations.call_count != expected.size()) { return false; }
  std::size_t index = 0;
  for (const unsigned char call : expected)
    if (operations.calls[index++] != call) { return false; }
  return true;
}

bool ArenaReleaseStateMachineFixtures() noexcept
{
  using cuda_detail::CudaArenaOwnership;
  using cuda_detail::CudaArenaReleaseState;
  using cuda_detail::ReleaseCudaArenas;
  {
    CudaArenaReleaseState state{};
    ArenaReleaseFixtureOperations operations{};
    const auto first = ReleaseCudaArenas(state, operations);
    const std::size_t first_calls = operations.call_count;
    const auto second = ReleaseCudaArenas(state, operations);
    if (!first.complete || first.failure_mask != 0 || first.pass != 1 ||
        first.integer != CudaArenaOwnership::released ||
        first.doubles != CudaArenaOwnership::released ||
        !second.complete || !second.idempotent || second.pass != 2 ||
        operations.call_count != first_calls ||
        !ExactArenaReleaseCalls(operations, {1, 2, 3, 5, 6, 7}))
      return false;
  }
  const std::array<ArenaReleaseFault, 3> wrapper_faults{{
      ArenaReleaseFault::vector_alias,
      ArenaReleaseFault::double_registration,
      ArenaReleaseFault::integer_registration}};
  for (std::size_t index = 0; index < wrapper_faults.size(); ++index)
  {
    CudaArenaReleaseState state{};
    ArenaReleaseFixtureOperations operations{};
    operations.fault = wrapper_faults[index];
    const auto first = ReleaseCudaArenas(state, operations);
    const std::size_t first_calls = operations.call_count;
    const auto second = ReleaseCudaArenas(state, operations);
    const std::uint64_t expected = std::uint64_t{1} << index;
    const bool integer_retained =
        first.integer == CudaArenaOwnership::mfem_retained;
    const bool double_disposed = index < 2
        ? first.doubles == CudaArenaOwnership::mfem_retained
        : first.doubles == CudaArenaOwnership::quarantined;
    if (first.complete || first.failure_mask != expected ||
        !integer_retained || !double_disposed || !operations.wrappers_inert ||
        !operations.views_cleared || !second.idempotent || second.complete ||
        operations.call_count != first_calls)
      return false;
  }
  const std::array<ArenaReleaseFault, 2> retryable_faults{{
      ArenaReleaseFault::integer_raw_retryable,
      ArenaReleaseFault::double_raw_retryable}};
  for (std::size_t index = 0; index < retryable_faults.size(); ++index)
  {
    CudaArenaReleaseState state{};
    ArenaReleaseFixtureOperations operations{};
    operations.fault = retryable_faults[index];
    const auto first = ReleaseCudaArenas(state, operations);
    const std::size_t first_calls = operations.call_count;
    const auto second = ReleaseCudaArenas(state, operations);
    const std::uint64_t expected = std::uint64_t{1} << (index + 3);
    if (first.complete || first.failure_mask != expected ||
        !second.complete || second.idempotent || second.failure_mask != 0 ||
        operations.call_count != first_calls + (index == 0 ? 2 : 1))
      return false;
  }
  const std::array<ArenaReleaseFault, 2> quarantine_faults{{
      ArenaReleaseFault::integer_raw_quarantined,
      ArenaReleaseFault::double_raw_quarantined}};
  for (std::size_t index = 0; index < quarantine_faults.size(); ++index)
  {
    CudaArenaReleaseState state{};
    ArenaReleaseFixtureOperations operations{};
    operations.fault = quarantine_faults[index];
    const auto first = ReleaseCudaArenas(state, operations);
    const std::size_t first_calls = operations.call_count;
    const auto second = ReleaseCudaArenas(state, operations);
    const std::uint64_t expected =
        (std::uint64_t{1} << (index + 3)) |
        cuda_detail::arena_raw_release_quarantined;
    if (first.complete || first.failure_mask != expected ||
        !second.idempotent || second.complete ||
        operations.call_count != first_calls)
      return false;
  }
  {
    CudaArenaReleaseState state{};
    ArenaReleaseFixtureOperations operations{};
    operations.integer_present = false;
    operations.fault = ArenaReleaseFault::double_raw_retryable;
    const auto first = ReleaseCudaArenas(state, operations);
    const std::size_t first_calls = operations.call_count;
    const auto second = ReleaseCudaArenas(state, operations);
    const std::size_t second_calls = operations.call_count;
    const auto third = ReleaseCudaArenas(state, operations);
    if (first.complete ||
        first.failure_mask != cuda_detail::arena_double_raw_release_failure ||
        first.integer != CudaArenaOwnership::none || !second.complete ||
        second.failure_mask != 0 || third.pass != 3 || !third.idempotent ||
        operations.call_count != second_calls || second_calls != first_calls + 1 ||
        !ExactArenaReleaseCalls(operations, {1, 2, 3, 5, 7, 7}))
      return false;
  }
  return true;
}

template <typename Backend>
bool RunArenaTeardown(const int ranks, const char* const context) noexcept
{
  static_assert(std::is_nothrow_destructible<TestEngine<Backend>>::value,
                "CUDA engine teardown must remain noexcept");
  auto engine = std::make_unique<TestEngine<Backend>>(Config(ranks));
  CudaRunEvidence setup{};
  if (!InitializePositive(*engine, setup, ranks, context)) { return false; }
  const std::uintptr_t double_token = setup.reuse.double_arena_token;
  const std::uintptr_t integer_token = setup.reuse.integer_arena_token;
  if (double_token == 0 || integer_token == 0 ||
      double_token == integer_token)
    return false;

  std::ostringstream pointers_before_stream;
  std::ostringstream aliases_before_stream;
  (void)mfem::mm.PrintPtrs(pointers_before_stream);
  (void)mfem::mm.PrintAliases(aliases_before_stream);
  const std::string pointers_before = pointers_before_stream.str();
  const std::string aliases_before = aliases_before_stream.str();
  std::string double_base_key;
  std::string integer_base_key;
  const bool registrations_visible = FindMfemBaseKeyForDeviceToken(
          pointers_before, double_token, double_base_key) &&
      FindMfemBaseKeyForDeviceToken(
          pointers_before, integer_token, integer_base_key) &&
      HasZeroOffsetAlias(aliases_before, double_base_key);

  engine.reset();

  std::ostringstream pointers_after_stream;
  std::ostringstream aliases_after_stream;
  (void)mfem::mm.PrintPtrs(pointers_after_stream);
  (void)mfem::mm.PrintAliases(aliases_after_stream);
  const std::string pointers_after = pointers_after_stream.str();
  const std::string aliases_after = aliases_after_stream.str();
  const bool registrations_released =
      pointers_after.find("d_ptr " + PointerTokenText(double_token)) ==
          std::string::npos &&
      pointers_after.find("d_ptr " + PointerTokenText(integer_token)) ==
          std::string::npos &&
      pointers_after.find("key " + double_base_key + ",") ==
          std::string::npos &&
      pointers_after.find("key " + integer_base_key + ",") ==
          std::string::npos &&
      !HasZeroOffsetAlias(aliases_after, double_base_key);

  std::uint64_t double_allocation_ordinal = 0;
  std::uint64_t double_deallocation_ordinal = 0;
  std::uint64_t integer_allocation_ordinal = 0;
  std::uint64_t integer_deallocation_ordinal = 0;
  const CudaAllocationLedgerV1 ledger = AuditedAllocator::Ledger();
  const bool exact_deallocations = ExactArenaEventPair(
          ledger, CudaAllocationSite::persistent_double_arena,
          double_token, double_allocation_ordinal,
          double_deallocation_ordinal) &&
      ExactArenaEventPair(
          ledger, CudaAllocationSite::persistent_integer_arena,
          integer_token, integer_allocation_ordinal,
          integer_deallocation_ordinal) &&
      double_allocation_ordinal < integer_allocation_ordinal &&
      integer_deallocation_ordinal < double_deallocation_ordinal;
  return registrations_visible && registrations_released &&
      exact_deallocations && ExactPolicyLedgers();
}

template <typename Backend>
bool RunConfigConsensusNegative(const int ranks, const int rank) noexcept
{
  if (ranks == 1)
  {
    const bool local_na = rank == 0 && HYPRE_GetError() == 0 &&
        cudaPeekAtLastError() == cudaSuccess;
    const bool collective_na = CollectivelyExact(local_na);
    std::fprintf(stderr,
        "CUDA_EXPECTED_NEGATIVE case=rank1.config.consensus-negative "
        "rank=%d applicability=N/A expected=collective-config-mismatch "
        "local=%s collective=%s\n",
        rank, local_na ? "PASS" : "FAIL",
        collective_na ? "PASS" : "FAIL");
    return collective_na;
  }
  if (ranks != 2) { return false; }
  CudaConfig config = Config(ranks, rank != 0);
  config.injection.enabled = true;
  config.injection.target_step = 2;
  config.injection.source_rank = rank;
  config.injection.token_fault = rank == 0
      ? CudaTokenFault::omit_solution : CudaTokenFault::stale_attempt;
  const std::uint64_t owned_before =
      AuditedAllocator::W8OwnedAllocationCalls();
  const std::uint64_t deallocations_before =
      AuditedAllocator::DeallocationEvents();
  const CudaAllocationLedgerV1 allocation_before =
      AuditedAllocator::Ledger();
  const CudaTransferLedgerV1 transfer_before = AuditedTransport::Ledger();
  const CudaTransferTotals transfer_totals_before =
      AuditedTransport::Accounting();
  auto engine = std::make_unique<TestEngine<Backend>>(config);
  CudaRunEvidence evidence{};
  (void)RecordNumericalEngineInitialization(
      "rank2.config.consensus-negative");
  const bool initialized = engine->Initialize(evidence);
  const bool local_exact =
      !initialized && evidence.status == CudaRunStatus::setup_failure &&
      evidence.setup_origin == FailureReason::collective_protocol &&
      AuditedAllocator::W8OwnedAllocationCalls() == owned_before &&
      AuditedAllocator::DeallocationEvents() == deallocations_before &&
      SameLedger(allocation_before, AuditedAllocator::Ledger()) &&
      SameLedger(transfer_before, AuditedTransport::Ledger()) &&
      SameTotals(transfer_totals_before, AuditedTransport::Accounting());
  const bool collective_exact = CollectivelyExact(local_exact);
  std::fprintf(stderr,
      "CUDA_EXPECTED_NEGATIVE case=rank2.config.consensus-negative "
      "rank=%d applicability=APPLIES expected_reason=%u actual_reason=%u "
      "expected_status=%u actual_status=%u initialized=%u "
      "local=%s collective=%s\n",
      rank, static_cast<unsigned>(FailureReason::collective_protocol),
      static_cast<unsigned>(evidence.setup_origin),
      static_cast<unsigned>(CudaRunStatus::setup_failure),
      static_cast<unsigned>(evidence.status), initialized ? 1U : 0U,
      local_exact ? "PASS" : "FAIL",
      collective_exact ? "PASS" : "FAIL");
  return collective_exact;
}

using AffineContract = typename TestEngine<TestRankOneBackendTag>::contract_type;
static_assert(!std::is_default_constructible<
                  typename AffineContract::StagingToken>::value &&
              !std::is_copy_constructible<
                  typename AffineContract::StagingToken>::value &&
              std::is_move_constructible<
                  typename AffineContract::StagingToken>::value &&
              !std::is_move_assignable<
                  typename AffineContract::StagingToken>::value);
static_assert(!std::is_default_constructible<
                  typename AffineContract::PreparedCommitToken>::value &&
              !std::is_copy_constructible<
                  typename AffineContract::PreparedCommitToken>::value &&
              std::is_move_constructible<
                  typename AffineContract::PreparedCommitToken>::value &&
              !std::is_move_assignable<
                  typename AffineContract::PreparedCommitToken>::value);
static_assert(!std::is_default_constructible<
                  typename AffineContract::ProvisionalCommitToken>::value &&
              !std::is_copy_constructible<
                  typename AffineContract::ProvisionalCommitToken>::value &&
              std::is_move_constructible<
                  typename AffineContract::ProvisionalCommitToken>::value &&
              !std::is_move_assignable<
                  typename AffineContract::ProvisionalCommitToken>::value);

bool VerifyOracleAndMutations(const CudaOraclePod& pod) noexcept
{
  if (!cuda_w8a_test_oracle::Verify(pod)) { return false; }
  auto mutated = std::make_unique<CudaOraclePod>(pod);
  mutated->residual_r[0] += 1.0 + std::abs(mutated->residual_r[0]);
  if (cuda_w8a_test_oracle::Verify(*mutated)) { return false; }
  mutated = std::make_unique<CudaOraclePod>(pod);
  mutated->schur[0] += 1.0 + std::abs(mutated->schur[0]);
  if (cuda_w8a_test_oracle::Verify(*mutated)) { return false; }
  mutated = std::make_unique<CudaOraclePod>(pod);
  mutated->signed_element_trace[0] = mutated->local_trace_count;
  if (cuda_w8a_test_oracle::Verify(*mutated)) { return false; }
  mutated = std::make_unique<CudaOraclePod>(pod);
  mutated->true_solution[0] +=
      1000.0 * (1.0 + std::abs(mutated->true_solution[0]));
  return !cuda_w8a_test_oracle::Verify(*mutated);
}

bool OracleIdentity(const CudaOraclePod& pod,
                    const CudaPrecommitSnapshot& snapshot,
                    const int ranks, const int rank) noexcept
{
  return pod.rank == rank && pod.ranks == ranks &&
      pod.attempt_sequence == snapshot.attempt.sequence &&
      pod.product_generation == snapshot.products.residual_generation &&
      pod.product_generation != 0 && pod.dt == fixed_dt;
}

void ReportCommitFailure(const int rank, const char* const stage,
                         const bool retry,
                         const std::uint64_t target_step) noexcept
{
  if (rank == 0)
  {
    std::cerr << "COMMIT_FAIL stage=" << stage
              << " target=" << target_step
              << " retry=" << (retry ? 1 : 0) << '\n';
  }
}

void ReportTokenFailure(const int rank, const char* const stage,
                        const CudaTokenFault fault) noexcept
{
  if (rank == 0)
  {
    std::cerr << "TOKEN_FAIL stage=" << stage
              << " fault=" << static_cast<int>(fault) << '\n';
  }
}

void ReportExpectedNegativeEvidence(
    const char* const case_id, const int rank, const char* const stage,
    const FailureReason expected_reason, const CudaRunStatus expected_status,
    const CudaStepEvidence& evidence, const bool local_exact,
    const bool collective_exact,
    const CudaSerializedCheckpointV1* const checkpoint = nullptr) noexcept
{
  std::fprintf(stderr,
      "CUDA_EXPECTED_NEGATIVE case=%s rank=%d stage=%s "
      "expected_reason=%u actual_reason=%u expected_status=%u "
      "actual_status=%u committed=%u retry=%u coordinates=%u bytes=%u "
      "flags=%u/%u/%u/%u local=%s collective=%s\n",
      case_id != nullptr ? case_id : "null", rank,
      stage != nullptr ? stage : "null",
      static_cast<unsigned>(expected_reason),
      static_cast<unsigned>(evidence.contract_reason),
      static_cast<unsigned>(expected_status),
      static_cast<unsigned>(evidence.status), evidence.committed ? 1U : 0U,
      evidence.retry_available ? 1U : 0U,
      evidence.coordinates_preserved ? 1U : 0U,
      evidence.committed_bytes_preserved ? 1U : 0U,
      checkpoint != nullptr && checkpoint->candidate_frozen ? 1U : 0U,
      checkpoint != nullptr && checkpoint->checkpoint_staged ? 1U : 0U,
      checkpoint != nullptr && checkpoint->output_staged ? 1U : 0U,
      checkpoint != nullptr && checkpoint->successful_commit ? 1U : 0U,
      local_exact ? "PASS" : "FAIL",
      collective_exact ? "PASS" : "FAIL");
}

void ReportAttemptTelemetryEvidence(
    const char* const case_id, const int rank, const char* const stage,
    const FailureReason expected_reason, const RunStatus expected_status,
    const AttemptTelemetry& attempt, const AcceptedTelemetry& accepted,
    const RunStatus actual_status, const std::uint64_t commits,
    const std::uint64_t out_of_order_rejections,
    const bool local_exact, const bool collective_exact) noexcept
{
  std::fprintf(stderr,
      "CUDA_ATTEMPT_TELEMETRY case=%s rank=%d stage=%s "
      "expected_reason=%u actual_reason=%u expected_status=%u "
      "actual_status=%u accepted=%u attempt=%llu/%llu "
      "accepted_attempt=%llu/%llu product_mask=0x%x/0x%x "
      "commits=%llu out_of_order_rejections=%llu local=%s collective=%s\n",
      case_id != nullptr ? case_id : "null", rank,
      stage != nullptr ? stage : "null",
      static_cast<unsigned>(expected_reason),
      static_cast<unsigned>(attempt.failure_reason),
      static_cast<unsigned>(expected_status),
      static_cast<unsigned>(actual_status), accepted.valid ? 1U : 0U,
      static_cast<unsigned long long>(attempt.identity.sequence),
      static_cast<unsigned long long>(attempt.identity.committed_step),
      static_cast<unsigned long long>(accepted.attempt.identity.sequence),
      static_cast<unsigned long long>(
          accepted.attempt.identity.committed_step),
      static_cast<unsigned>(attempt.validated_product_mask),
      static_cast<unsigned>(accepted.attempt.validated_product_mask),
      static_cast<unsigned long long>(commits),
      static_cast<unsigned long long>(out_of_order_rejections),
      local_exact ? "PASS" : "FAIL",
      collective_exact ? "PASS" : "FAIL");
}

struct OracleOutputTestSeam
{
  int checkpoint_mismatch_rank = -1;
};

struct FrozenStageResult
{
  bool passed = false;
  bool cleanup_exact = false;
};

template <typename Engine>
FrozenStageResult StageOracleOutputCollective(
    Engine& engine, CudaSerializedCheckpointV1& checkpoint,
    CudaStepEvidence& evidence, const CudaPrecommitSnapshot& snapshot,
    const int ranks, const int rank, const bool retry,
    const std::uint64_t target_step, const bool mutate_oracle,
    const char* const case_id,
    const OracleOutputTestSeam seam = {}) noexcept
{
  auto staged_oracle = std::make_unique<CudaOraclePod>();
  const auto report_stage =
      [&](const char* const stage, const bool local_exact,
          const bool collective_exact) noexcept
  {
    std::fprintf(stderr,
        "CUDA_ORACLE_STAGE case=%s rank=%d target=%llu stage=%s "
        "local=%u collective=%u pod_rank=%d pod_ranks=%d "
        "attempt=%llu/%llu product=%llu/%llu dt=%.17g/%.17g "
        "output=%u committed=%u\n",
        case_id != nullptr ? case_id : "null", rank,
        static_cast<unsigned long long>(target_step), stage,
        local_exact ? 1U : 0U, collective_exact ? 1U : 0U,
        staged_oracle->rank, staged_oracle->ranks,
        static_cast<unsigned long long>(staged_oracle->attempt_sequence),
        static_cast<unsigned long long>(snapshot.attempt.sequence),
        static_cast<unsigned long long>(staged_oracle->product_generation),
        static_cast<unsigned long long>(
            snapshot.products.residual_generation),
        staged_oracle->dt, fixed_dt, checkpoint.output_staged ? 1U : 0U,
        checkpoint.successful_commit ? 1U : 0U);
  };
  const auto fail_and_cleanup =
      [&](const char* const stage,
          const FailureReason reason) noexcept -> FrozenStageResult
  {
    const bool cleanup_exact = ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        reason, CudaFailurePhase::diagnostics,
        AffineLocalPhase::staging);
    ReportCommitFailure(rank, stage, retry, target_step);
    return {false, cleanup_exact};
  };

  // StagePreparedCheckpoint consensus immediately precedes this call and
  // proves the facade's public staging preconditions common across ranks.
  const bool oracle_stage_return =
      engine.StagePreparedOracle(*staged_oracle, evidence);
  const bool oracle_stage_collective =
      CollectivelyExact(oracle_stage_return);
  report_stage(
      "stage-prepared-oracle", oracle_stage_return,
      oracle_stage_collective);
  if (!oracle_stage_collective)
    return fail_and_cleanup("oracle/output-stage", FailureReason::backend_failure);

  const bool oracle_identity_exact =
      OracleIdentity(*staged_oracle, snapshot, ranks, rank);
  const bool oracle_identity_collective =
      CollectivelyExact(oracle_identity_exact);
  report_stage(
      "oracle-identity", oracle_identity_exact,
      oracle_identity_collective);
  if (!oracle_identity_collective)
    return fail_and_cleanup("oracle/output-stage", FailureReason::backend_failure);

  const bool independent_oracle_exact =
      cuda_w8a_test_oracle::Verify(*staged_oracle);
  const bool independent_oracle_collective =
      CollectivelyExact(independent_oracle_exact);
  report_stage(
      "independent-verify", independent_oracle_exact,
      independent_oracle_collective);
  if (!independent_oracle_collective)
  {
    const bool shape_exact =
        cuda_w8a_test_oracle::ValidShape(*staged_oracle);
    const bool shape_collective = CollectivelyExact(shape_exact);
    bool count_collective = false;
    bool physics_exact = false;
    bool physics_collective = false;
    bool condensation_exact = false;
    bool condensation_collective = false;
    bool signs_residual_exact = false;
    bool signs_residual_collective = false;
    if (shape_collective)
    {
      int global_count_min = staged_oracle->global_true_trace_count;
      int global_count_max = staged_oracle->global_true_trace_count;
      const bool count_min_ready = MPI_Allreduce(
          &staged_oracle->global_true_trace_count, &global_count_min,
          1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS;
      const bool count_max_ready = MPI_Allreduce(
          &staged_oracle->global_true_trace_count, &global_count_max,
          1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS;
      count_collective = CollectivelyExact(
          count_min_ready && count_max_ready &&
          global_count_min == global_count_max);
      physics_exact =
          cuda_w8a_test_oracle::VerifyPhysics(*staged_oracle);
      physics_collective = CollectivelyExact(physics_exact);
      condensation_exact =
          cuda_w8a_test_oracle::VerifyCondensationRecovery(*staged_oracle);
      condensation_collective = CollectivelyExact(condensation_exact);
      if (count_collective && physics_collective &&
          condensation_collective)
      {
        signs_residual_exact =
            cuda_w8a_test_oracle::VerifySignsAndTrueResidual(
                *staged_oracle);
        signs_residual_collective =
            CollectivelyExact(signs_residual_exact);
      }
    }
    std::fprintf(stderr,
        "CUDA_ORACLE_COMPONENTS case=%s rank=%d target=%llu "
        "shape=%u/%u count=%u physics=%u/%u condensation=%u/%u "
        "signs_residual=%u/%u global_true=%d local_true=%d\n",
        case_id != nullptr ? case_id : "null", rank,
        static_cast<unsigned long long>(target_step),
        shape_exact ? 1U : 0U, shape_collective ? 1U : 0U,
        count_collective ? 1U : 0U,
        physics_exact ? 1U : 0U, physics_collective ? 1U : 0U,
        condensation_exact ? 1U : 0U,
        condensation_collective ? 1U : 0U,
        signs_residual_exact ? 1U : 0U,
        signs_residual_collective ? 1U : 0U,
        staged_oracle->global_true_trace_count,
        staged_oracle->true_trace_count);
    return fail_and_cleanup("oracle/output-stage", FailureReason::backend_failure);
  }

  const bool mutation_gate_exact =
      !mutate_oracle || VerifyOracleAndMutations(*staged_oracle);
  const bool mutation_gate_collective =
      CollectivelyExact(mutation_gate_exact);
  report_stage(
      "mutation-gate", mutation_gate_exact, mutation_gate_collective);
  if (!mutation_gate_collective)
    return fail_and_cleanup("oracle/output-stage", FailureReason::backend_failure);

  if (rank == seam.checkpoint_mismatch_rank)
    checkpoint.descriptor.interior_hash.bytes[0] ^= 1U;
  const bool mark_output_staged_return =
      engine.MarkOutputStaged(checkpoint);
  const bool output_staged_exact = checkpoint.output_staged;
  const bool successful_commit_clear = !checkpoint.successful_commit;
  const bool output_state_collective = CollectivelyExact(
      mark_output_staged_return && output_staged_exact &&
      successful_commit_clear);
  report_stage(
      "mark-output-staged", mark_output_staged_return,
      output_state_collective);
  std::fprintf(stderr,
      "CUDA_ORACLE_OUTPUT_FLAGS case=%s rank=%d target=%llu "
      "mark=%u output=%u commit_clear=%u collective=%u\n",
      case_id != nullptr ? case_id : "null", rank,
      static_cast<unsigned long long>(target_step),
      mark_output_staged_return ? 1U : 0U,
      output_staged_exact ? 1U : 0U,
      successful_commit_clear ? 1U : 0U,
      output_state_collective ? 1U : 0U);
  if (!output_state_collective)
  {
    const FailureReason reason = seam.checkpoint_mismatch_rank >= 0
        ? FailureReason::collective_protocol : FailureReason::backend_failure;
    return fail_and_cleanup("oracle/output-stage", reason);
  }

  const bool oracle_policy_ledgers_exact = ExactPolicyLedgers();
  const bool oracle_policy_ledgers_collective =
      CollectivelyExact(oracle_policy_ledgers_exact);
  report_stage(
      "policy-ledgers", oracle_policy_ledgers_exact,
      oracle_policy_ledgers_collective);
  if (!oracle_policy_ledgers_collective)
    return fail_and_cleanup("oracle/output-stage", FailureReason::backend_failure);
  return {true, true};
}

template <typename Engine>
bool CommitSuccessfulAttempt(Engine& engine, const bool retry,
                             const std::uint64_t target_step,
                             const int ranks, const int rank,
                             const std::uint64_t allocation_baseline,
                             CudaSerializedCheckpointV1& checkpoint,
                             const bool mutate_oracle = false,
                             CudaPrecommitSnapshot* captured = nullptr,
                             const char* const case_id = "unspecified",
                             const bool allow_initial_shared_growth = false)
    noexcept
{
  const std::uint64_t owned_allocation_calls =
      AuditedAllocator::W8OwnedAllocationCalls();
  const std::uint64_t deallocation_events =
      AuditedAllocator::DeallocationEvents();
  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence evidence{};
  const auto* contract = engine.contract();
  const bool contract_collective = CollectivelyExact(contract != nullptr);
  if (!contract_collective)
  {
    ReportCommitFailure(rank, "contract", retry, target_step);
    return false;
  }
  const StateCoordinates old_coordinates = contract->coordinates();
  const auto old_publication = contract->Published();
  StateCoordinates visible{};
  const bool prepare_step_return =
      engine.PrepareStep(retry, snapshot, evidence);
  const std::uint64_t accepted_allocation_events =
      AuditedAllocator::AllocationEvents();
  const bool local_allocation_sample_exact = allow_initial_shared_growth
      ? InitialAllocationSampleExact(allocation_baseline,
                                     accepted_allocation_events)
      : accepted_allocation_events == allocation_baseline;
  const bool collective_allocation_sample_exact =
      CollectivelyExact(local_allocation_sample_exact &&
          AuditedAllocator::W8OwnedAllocationCalls() ==
              owned_allocation_calls &&
          AuditedAllocator::DeallocationEvents() == deallocation_events);
  const bool prepare_policy_ledgers_exact = ExactPolicyLedgers();
  const bool prepare_step_local = prepare_step_return &&
      snapshot.numerical_candidate_ready &&
      snapshot.attempt.retry_index == (retry ? 1 : 0) &&
      snapshot.target.step == target_step &&
      snapshot.target.generation == old_coordinates.generation + 1 &&
      SameCoordinates(snapshot.committed, old_coordinates) &&
      snapshot.products.CoupledAndOrdered() &&
      snapshot.staging_must_precede_finalize &&
      snapshot.finalize_has_no_fallible_posthook &&
      contract->PublishedCoordinates(old_publication, visible) &&
      SameCoordinates(visible, old_coordinates) &&
      collective_allocation_sample_exact && prepare_policy_ledgers_exact;
  const bool prepare_step_collective =
      CollectivelyExact(prepare_step_local);
  if (!prepare_step_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterPrepareStep(prepare_step_return, evidence));
    ReportCommitFailure(rank, "prepare-step", retry, target_step);
    return false;
  }
  if (captured != nullptr) { *captured = snapshot; }

  const CudaTransferLedgerV1 checkpoint_ledger_before =
      AuditedTransport::Ledger();
  const std::uint64_t checkpoint_sync_successes_before =
      AuditedTransport::SuccessfulSynchronizationEvents();
  const bool checkpoint_stage_return =
      engine.StagePreparedCheckpoint(checkpoint, evidence);
  const CudaTransferLedgerV1 checkpoint_ledger_after =
      AuditedTransport::Ledger();
  const std::uint64_t checkpoint_sync_successes_after =
      AuditedTransport::SuccessfulSynchronizationEvents();
  const bool checkpoint_ledger_mirror = ExactPolicyLedgers();
  const bool local_checkpoint_stage_failed =
      !checkpoint_stage_return ||
      checkpoint.schema != 1 || checkpoint.rank_count != ranks ||
      checkpoint.attempt.sequence != snapshot.attempt.sequence ||
      checkpoint.facade_incarnation == 0 ||
      !checkpoint.candidate_frozen || !checkpoint.checkpoint_staged ||
      checkpoint.output_staged || checkpoint.successful_commit ||
      checkpoint.descriptor.interior_hash !=
          snapshot.candidate_interior_hash ||
      checkpoint.descriptor.trace_hash != snapshot.candidate_trace_hash ||
      checkpoint.local_interior_hash !=
          snapshot.candidate_local_interior_hash ||
      checkpoint.local_trace_hash != snapshot.candidate_local_trace_hash ||
      checkpoint.interior_count != snapshot.interior_coefficients ||
      checkpoint.owned_true_trace_count !=
          snapshot.true_trace_coefficients ||
      checkpoint.staging_generation != snapshot.target.generation ||
      checkpoint.interior_source_token == 0 ||
      checkpoint.trace_source_token == 0 ||
      !ValidRange(checkpoint.staging_range, checkpoint_ledger_after) ||
      !checkpoint_ledger_mirror;
  const CheckpointPostconditionConsensus checkpoint_consensus =
      AgreeCheckpointPostconditionFailure(
          local_checkpoint_stage_failed, ranks);
  if (checkpoint_consensus.cleanup_required)
  {
    if (local_checkpoint_stage_failed)
    {
      const CheckpointDiagnosticV1 diagnostic = BuildCheckpointDiagnosticV1(
          checkpoint_stage_return, checkpoint, snapshot, ranks, rank,
          checkpoint_ledger_before, checkpoint_ledger_after,
          checkpoint_sync_successes_before, checkpoint_sync_successes_after,
          checkpoint_ledger_mirror);
      ReportCheckpointDiagnosticV1(
          case_id, rank, retry, target_step, checkpoint, snapshot,
          checkpoint_ledger_before, checkpoint_ledger_after,
          AuditedTransport::Accounting(), diagnostic);
    }
    std::fprintf(stderr,
        "CUDA_CKPT_CONSENSUS case=%s rank=%d local=%u reduced=%u "
        "failing_ranks=%d cleanup=%u reason=%u phase=%u\n",
        case_id != nullptr ? case_id : "null", rank,
        local_checkpoint_stage_failed ? 1U : 0U,
        checkpoint_consensus.reduction_complete ? 1U : 0U,
        checkpoint_consensus.failing_ranks,
        checkpoint_consensus.cleanup_required ? 1U : 0U,
        static_cast<unsigned>(checkpoint_consensus.cleanup_reason),
        static_cast<unsigned>(checkpoint_consensus.cleanup_phase));
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        checkpoint_consensus.cleanup_reason,
        checkpoint_consensus.cleanup_phase,
        AffineLocalPhase::staging);
    ReportCommitFailure(rank, "checkpoint-stage", retry, target_step);
    return false;
  }

  const FrozenStageResult oracle_output = StageOracleOutputCollective(
      engine, checkpoint, evidence, snapshot, ranks, rank, retry,
      target_step, mutate_oracle, case_id);
  if (!oracle_output.passed)
    return false;

  const bool prepare_commit_return =
      engine.PrepareCommit(checkpoint, evidence);
  const bool prepare_commit_return_collective =
      CollectivelyExact(prepare_commit_return);
  if (!prepare_commit_return_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterPrepareCommit(prepare_commit_return, evidence));
    ReportCommitFailure(rank, "prepare-commit", retry, target_step);
    return false;
  }
  const bool prepare_commit_local =
      evidence.status == CudaRunStatus::ready && !evidence.committed &&
      checkpoint.candidate_frozen && checkpoint.checkpoint_staged &&
      checkpoint.output_staged && !checkpoint.successful_commit &&
      SameCoordinates(contract->coordinates(), old_coordinates) &&
      contract->PublishedCoordinates(old_publication, visible) &&
      SameCoordinates(visible, old_coordinates);
  const bool prepare_commit_collective =
      CollectivelyExact(prepare_commit_local);
  if (!prepare_commit_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterPrepareCommit(prepare_commit_return, evidence));
    ReportCommitFailure(rank, "prepare-commit", retry, target_step);
    return false;
  }

  const bool commit_prepared_return =
      engine.CommitPrepared(checkpoint, evidence);
  const bool commit_prepared_return_collective =
      CollectivelyExact(commit_prepared_return);
  if (!commit_prepared_return_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterCommitPrepared(commit_prepared_return, evidence));
    ReportCommitFailure(rank, "commit-prepared", retry, target_step);
    return false;
  }
  const bool commit_policy_ledgers_exact = ExactPolicyLedgers();
  const bool commit_prepared_local =
      evidence.status == CudaRunStatus::ready && !evidence.committed &&
      checkpoint.candidate_frozen && checkpoint.checkpoint_staged &&
      checkpoint.output_staged && !checkpoint.successful_commit &&
      SameCoordinates(contract->coordinates(), old_coordinates) &&
      contract->PublishedCoordinates(old_publication, visible) &&
      SameCoordinates(visible, old_coordinates) &&
      commit_policy_ledgers_exact;
  const bool commit_prepared_collective =
      CollectivelyExact(commit_prepared_local);
  if (!commit_prepared_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterCommitPrepared(commit_prepared_return, evidence));
    ReportCommitFailure(rank, "commit-prepared", retry, target_step);
    return false;
  }

  const bool finalize_return = engine.FinalizeCommit(checkpoint, evidence);
  const bool finalize_return_collective =
      CollectivelyExact(finalize_return);
  const bool collective_allocation_plateau_exact =
      CollectivelyExactAllocatorState(accepted_allocation_events,
                                      owned_allocation_calls,
                                      deallocation_events);
  const bool finalize_evidence_exact = ExactAttemptEvidence(evidence, ranks);
  const bool finalize_local = finalize_return_collective &&
      checkpoint.successful_commit && evidence.committed &&
      !evidence.retry_available && evidence.collective_consensus &&
      evidence.contract_reason == FailureReason::none &&
      evidence.status == CudaRunStatus::committed &&
      evidence.after.step == target_step &&
      evidence.after.generation == target_step &&
      std::abs(evidence.after.time - fixed_dt * target_step) <= 1.0e-14 &&
      SameCoordinates(contract->coordinates(), snapshot.target) &&
      !contract->PublishedCoordinates(old_publication, visible) &&
      evidence.products.CoupledAndOrdered() && evidence.no_fallback &&
      evidence.device_resident && collective_allocation_plateau_exact &&
      finalize_evidence_exact;
  const bool finalize_collective = CollectivelyExact(finalize_local);
  if (!finalize_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterFinalizeCommit(evidence));
    ReportCommitFailure(rank, "finalize", retry, target_step);
    return false;
  }

  auto exported = std::make_unique<CudaOraclePod>();
  const bool export_return = engine.ExportOracle(*exported);
  const bool export_return_collective = CollectivelyExact(export_return);
  if (!export_return_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        AffineLocalPhase::consumed);
    ReportCommitFailure(rank, "export", retry, target_step);
    return false;
  }
  const bool export_identity_local =
      OracleIdentity(*exported, snapshot, ranks, rank);
  const bool export_identity_collective =
      CollectivelyExact(export_identity_local);
  if (!export_identity_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        AffineLocalPhase::consumed);
    ReportCommitFailure(rank, "export", retry, target_step);
    return false;
  }
  const bool export_oracle_exact =
      cuda_w8a_test_oracle::Verify(*exported);
  const bool export_oracle_collective =
      CollectivelyExact(export_oracle_exact);
  const bool export_allocator_state_exact =
      CollectivelyExactAllocatorState(accepted_allocation_events,
                                      owned_allocation_calls,
                                      deallocation_events);
  const bool export_policy_ledgers_exact = ExactPolicyLedgers();
  const bool export_local = export_oracle_collective &&
      export_allocator_state_exact && export_policy_ledgers_exact;
  const bool export_collective = CollectivelyExact(export_local);
  if (!export_collective)
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        AffineLocalPhase::consumed);
    ReportCommitFailure(rank, "export", retry, target_step);
  }
  return export_collective;
}

/**
 * Functional trajectory commit seam.  The application/codec owner decides
 * which staged payload becomes durable output; this leaf always exercises the
 * existing reversible checkpoint bridge before the atomic commit boundary.
 * No oracle export is requested here: the independent p2 trajectory/balance
 * validator remains a later composed W8A gate.
 */
template <typename Engine>
bool CommitFunctionalAttempt(
    Engine& engine, const bool retry, const std::uint64_t target_step,
    const int ranks, const int rank,
    CudaSerializedCheckpointV1& checkpoint,
    const char* const case_id) noexcept
{
  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence evidence{};
  const auto* contract = engine.contract();
  if (!CollectivelyExact(contract != nullptr)) { return false; }
  const StateCoordinates before = contract->coordinates();
  const std::uint64_t owned_calls =
      AuditedAllocator::W8OwnedAllocationCalls();
  const std::uint64_t deallocations =
      AuditedAllocator::DeallocationEvents();

  const bool prepared = engine.PrepareStep(retry, snapshot, evidence);
  const bool prepared_local = prepared &&
      snapshot.numerical_candidate_ready &&
      snapshot.target.step == target_step &&
      snapshot.target.generation == before.generation + 1 &&
      snapshot.attempt.retry_index == (retry ? 1U : 0U) &&
      SameCoordinates(snapshot.committed, before) &&
      snapshot.products.CoupledAndOrdered() &&
      std::isfinite(evidence.initial_residual) &&
      std::isfinite(evidence.final_residual) &&
      evidence.nonlinear_updates > 0 &&
      evidence.nonlinear_updates <= 20 &&
      evidence.final_residual <=
          1.0e-10 * (1.0 + evidence.initial_residual) &&
      std::isfinite(evidence.true_relative_residual) &&
      evidence.true_relative_residual <= 1.0e-10 &&
      evidence.no_fallback && evidence.device_resident &&
      AuditedAllocator::W8OwnedAllocationCalls() == owned_calls &&
      AuditedAllocator::DeallocationEvents() == deallocations &&
      ExactPolicyLedgers();
  if (!CollectivelyExact(prepared_local))
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterPrepareStep(prepared, evidence));
    return false;
  }

  const bool staged = engine.StagePreparedCheckpoint(checkpoint, evidence);
  const bool staged_local = staged && checkpoint.candidate_frozen &&
      checkpoint.checkpoint_staged && !checkpoint.output_staged &&
      !checkpoint.successful_commit &&
      checkpoint.descriptor.interior_hash ==
          snapshot.candidate_interior_hash &&
      checkpoint.descriptor.trace_hash == snapshot.candidate_trace_hash &&
      checkpoint.local_interior_hash ==
          snapshot.candidate_local_interior_hash &&
      checkpoint.local_trace_hash == snapshot.candidate_local_trace_hash &&
      checkpoint.interior_count == snapshot.interior_coefficients &&
      checkpoint.owned_true_trace_count ==
          snapshot.true_trace_coefficients &&
      ExactId(checkpoint.descriptor.layout, runtime_layout_id) &&
      checkpoint.staging_generation == snapshot.target.generation &&
      ValidRange(checkpoint.staging_range, AuditedTransport::Ledger()) &&
      ExactPolicyLedgers();
  if (!CollectivelyExact(staged_local))
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        AffineLocalPhase::staging);
    return false;
  }
  if (!CollectivelyExact(engine.MarkOutputStaged(checkpoint)))
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        AffineLocalPhase::staging);
    return false;
  }
  const bool prepared_commit = engine.PrepareCommit(checkpoint, evidence);
  if (!CollectivelyExact(prepared_commit))
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterPrepareCommit(prepared_commit, evidence));
    return false;
  }
  const bool commit_prepared = engine.CommitPrepared(checkpoint, evidence);
  if (!CollectivelyExact(commit_prepared))
  {
    (void)ResolveUnexpectedAffineExit(
        engine, checkpoint, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure, CudaFailurePhase::diagnostics,
        PhaseAfterCommitPrepared(commit_prepared, evidence));
    return false;
  }
  const bool finalized = engine.FinalizeCommit(checkpoint, evidence);
  const bool final_local = finalized && checkpoint.successful_commit &&
      evidence.committed && !evidence.retry_available &&
      evidence.collective_consensus &&
      evidence.contract_reason == FailureReason::none &&
      evidence.status == CudaRunStatus::committed &&
      evidence.after.step == target_step &&
      evidence.after.generation == target_step &&
      std::abs(evidence.after.time - fixed_dt * target_step) <= 1.0e-14 &&
      SameCoordinates(contract->coordinates(), snapshot.target) &&
      evidence.products.CoupledAndOrdered() && evidence.no_fallback &&
      evidence.device_resident &&
      AuditedAllocator::W8OwnedAllocationCalls() == owned_calls &&
      AuditedAllocator::DeallocationEvents() == deallocations &&
      ExactAttemptEvidence(evidence, ranks) && ExactPolicyLedgers();
  return CollectivelyExact(final_local);
}

template <typename Engine>
bool InjectFunctionalRollback(
    Engine& engine, const int ranks, const int rank,
    const char* const case_id) noexcept
{
  const auto* contract = engine.contract();
  if (!CollectivelyExact(contract != nullptr)) { return false; }
  const StateCoordinates before = contract->coordinates();
  const auto publication = contract->Published();
  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence evidence{};
  CudaSerializedCheckpointV1 prohibited{};
  const bool prepared = engine.PrepareStep(false, snapshot, evidence);
  const bool injected_local = prepared &&
      snapshot.target.step == functional_injected_step &&
      snapshot.numerical_candidate_ready &&
      evidence.phase == CudaFailurePhase::after_recovery_before_commit &&
      evidence.source_rank == ranks - 1;
  if (!CollectivelyExact(injected_local))
  {
    (void)ResolveUnexpectedAffineExit(
        engine, prohibited, evidence, snapshot, case_id, ranks, rank,
        FailureReason::backend_failure,
        CudaFailurePhase::after_recovery_before_commit,
        PhaseAfterPrepareStep(prepared, evidence));
    return false;
  }
  const bool staging_rejected =
      !engine.StagePreparedCheckpoint(prohibited, evidence);
  const bool rolled_back = ResolveUnexpectedAffineExit(
      engine, prohibited, evidence, snapshot, case_id, ranks, rank,
      FailureReason::backend_failure,
      CudaFailurePhase::after_recovery_before_commit,
      AffineLocalPhase::staging);
  StateCoordinates visible{};
  const bool exact_local = staging_rejected && rolled_back &&
      evidence.status == CudaRunStatus::retry_available &&
      evidence.retry_available && !evidence.committed &&
      evidence.coordinates_preserved && evidence.committed_bytes_preserved &&
      SameCoordinates(contract->coordinates(), before) &&
      contract->PublishedCoordinates(publication, visible) &&
      SameCoordinates(visible, before) && !prohibited.candidate_frozen &&
      !prohibited.checkpoint_staged && !prohibited.output_staged &&
      !prohibited.successful_commit && ExactPolicyLedgers();
  return CollectivelyExact(exact_local);
}

template <typename Backend>
bool RunFunctionalTrajectory(const int ranks, const char* const context)
    noexcept
{
  StateHash reference_interior{};
  StateHash reference_trace{};
  {
    auto reference = std::make_unique<TestEngine<Backend>>(
        FunctionalConfig(ranks, false));
    CudaRunEvidence setup{};
    if (!InitializePositive(*reference, setup, ranks, context) ||
        !setup.reuse.persistent_graph || !setup.reuse.persistent_slots ||
        !setup.reuse.persistent_workspace)
      return false;
    CudaSerializedCheckpointV1 step{};
    for (std::uint64_t target = 1;
         target <= functional_accepted_steps; ++target)
    {
      if (!CommitFunctionalAttempt(
              *reference, false, target, ranks, setup.rank, step, context))
        return false;
    }
    const auto* contract = reference->contract();
    if (!CollectivelyExact(contract != nullptr &&
            contract->coordinates().step == functional_accepted_steps &&
            contract->coordinates().cursor == functional_accepted_steps &&
            contract->telemetry().commits == functional_accepted_steps &&
            SamePersistentIdentity(
                setup.reuse, reference->ReuseEvidence(), ranks)))
      return false;
    reference_interior = step.descriptor.interior_hash;
    reference_trace = step.descriptor.trace_hash;
  }

  CudaSerializedCheckpointV1 bridge{};
  {
    auto writer = std::make_unique<TestEngine<Backend>>(
        FunctionalConfig(ranks, true));
    CudaRunEvidence setup{};
    if (!InitializePositive(*writer, setup, ranks, context) ||
        !setup.reuse.persistent_graph || !setup.reuse.persistent_slots ||
        !setup.reuse.persistent_workspace)
      return false;
    for (std::uint64_t target = 1;
         target <= functional_checkpoint_step; ++target)
    {
      CudaSerializedCheckpointV1 step{};
      if (!CommitFunctionalAttempt(
              *writer, false, target, ranks, setup.rank, step, context))
        return false;
      if (target == functional_checkpoint_step) { bridge = step; }
    }
    const bool bridge_local = bridge.successful_commit &&
        bridge.descriptor.coordinates.step == functional_checkpoint_step &&
        bridge.descriptor.coordinates.generation ==
            functional_checkpoint_step &&
        std::abs(bridge.descriptor.coordinates.time - 0.5) <= 1.0e-14 &&
        SamePersistentIdentity(setup.reuse, writer->ReuseEvidence(), ranks);
    if (!CollectivelyExact(bridge_local)) { return false; }
  }

  auto reader = std::make_unique<TestEngine<Backend>>(
      FunctionalConfig(ranks, true));
  CudaRunEvidence setup{};
  if (!InitializePositive(*reader, setup, ranks, context) ||
      !setup.reuse.persistent_graph || !setup.reuse.persistent_slots ||
      !setup.reuse.persistent_workspace)
    return false;
  CudaStepEvidence restored{};
  const bool restore_local = reader->Restore(bridge, restored) &&
      restored.committed &&
      restored.after.step == functional_checkpoint_step &&
      restored.after.generation == functional_checkpoint_step &&
      restored.after.cursor == functional_checkpoint_step &&
      restored.after_interior_hash == bridge.descriptor.interior_hash &&
      restored.after_trace_hash == bridge.descriptor.trace_hash;
  if (!CollectivelyExact(restore_local)) { return false; }

  CudaSerializedCheckpointV1 final_checkpoint{};
  for (std::uint64_t target = functional_checkpoint_step + 1;
       target < functional_injected_step; ++target)
  {
    if (!CommitFunctionalAttempt(
            *reader, false, target, ranks, setup.rank, final_checkpoint,
            context))
      return false;
  }
  if (!InjectFunctionalRollback(*reader, ranks, setup.rank, context) ||
      !CommitFunctionalAttempt(
          *reader, true, functional_injected_step, ranks, setup.rank,
          final_checkpoint, context))
    return false;
  for (std::uint64_t target = functional_injected_step + 1;
       target <= functional_accepted_steps; ++target)
  {
    if (!CommitFunctionalAttempt(
            *reader, false, target, ranks, setup.rank, final_checkpoint,
            context))
      return false;
  }
  const auto* contract = reader->contract();
  const bool final_local = contract != nullptr &&
      contract->coordinates().step == functional_accepted_steps &&
      contract->coordinates().generation == functional_accepted_steps &&
      contract->coordinates().cursor == functional_accepted_steps + 1 &&
      std::abs(contract->coordinates().time - 1.0) <= 1.0e-14 &&
      contract->telemetry().commits == functional_checkpoint_step &&
      final_checkpoint.successful_commit &&
      final_checkpoint.descriptor.interior_hash == reference_interior &&
      final_checkpoint.descriptor.trace_hash == reference_trace &&
      contract->telemetry().last_attempt.prohibited.fallback_events == 0 &&
      SamePersistentIdentity(setup.reuse, reader->ReuseEvidence(), ranks) &&
      ExactPolicyLedgers();
  std::fprintf(stderr,
      "CUDA_FUNCTIONAL_RESULT rank=%d ranks=%d step=%llu generation=%llu "
      "cursor=%llu restart=10 retry=14 local=%s\n",
      setup.rank, ranks,
      static_cast<unsigned long long>(
          contract == nullptr ? 0 : contract->coordinates().step),
      static_cast<unsigned long long>(
          contract == nullptr ? 0 : contract->coordinates().generation),
      static_cast<unsigned long long>(
          contract == nullptr ? 0 : contract->coordinates().cursor),
      final_local ? "PASS" : "FAIL");
  return CollectivelyExact(final_local);
}

template <typename Backend>
bool RunLifecycle(const int ranks,
                  CudaSerializedCheckpointV1& final_checkpoint,
                  const char* const context,
                  const std::uint64_t expected_initialization_ordinal = 0)
    noexcept
{
  CudaConfig config = Config(ranks, true);
  config.injection.enabled = true;
  config.injection.target_step = 2;
  config.injection.source_rank = ranks - 1;
  auto engine = std::make_unique<TestEngine<Backend>>(config);
  const bool result = [&]() noexcept
  {
    CudaRunEvidence setup{};
    std::uint64_t initialization_ordinal = 0;
    if (!InitializePositive(
            *engine, setup, ranks, context, &initialization_ordinal))
      return false;
    const bool initialization_ordinal_collective = CollectivelyExact(
        expected_initialization_ordinal == 0 ||
        initialization_ordinal == expected_initialization_ordinal);
    std::fprintf(stderr,
        "CUDA_ENGINE_INIT_QUALIFICATION rank=%d context=%s "
        "expected=%llu actual=%llu local=%s collective=%s\n",
        setup.rank, context != nullptr ? context : "null",
        static_cast<unsigned long long>(expected_initialization_ordinal),
        static_cast<unsigned long long>(initialization_ordinal),
        (expected_initialization_ordinal == 0 ||
         initialization_ordinal == expected_initialization_ordinal)
            ? "PASS" : "FAIL",
        initialization_ordinal_collective ? "PASS" : "FAIL");
    if (!initialization_ordinal_collective) { return false; }
    const std::uint64_t allocations = setup.reuse.baseline_allocation_events;
    const std::uint64_t owned_allocation_calls =
        setup.reuse.baseline_w8_owned_allocation_calls;
    const std::uint64_t owned_deallocation_events =
        AuditedAllocator::DeallocationEvents();
    const AllocatorHighWaterSample initial_high_water =
        SampleAllocatorHighWater();
    std::uint64_t plateau_allocations = allocations;
    AllocatorHighWaterSample plateau_high_water{};
    std::uint64_t credited_plateau_cycles = 0;
    const auto credited_allocation_cycle =
        [&](const std::uint64_t cycle) noexcept
    {
      const std::uint64_t current_allocations =
          AuditedAllocator::AllocationEvents();
      const std::uint64_t current_owned_calls =
          AuditedAllocator::W8OwnedAllocationCalls();
      const std::uint64_t current_deallocations =
          AuditedAllocator::DeallocationEvents();
      const AllocatorHighWaterSample current_high_water =
          SampleAllocatorHighWater();
      const bool mfem_nonnegative =
          current_high_water.mfem >= initial_high_water.mfem;
      const bool hypre_nonnegative =
          current_high_water.hypre >= initial_high_water.hypre;
      const bool allocation_nonnegative = current_allocations >= allocations;
      const bool owned_nonnegative =
          current_owned_calls >= owned_allocation_calls;
      const bool deallocation_nonnegative =
          current_deallocations >= owned_deallocation_events;
      const std::size_t mfem_delta = mfem_nonnegative
          ? current_high_water.mfem - initial_high_water.mfem : 0;
      const std::size_t hypre_delta = hypre_nonnegative
          ? current_high_water.hypre - initial_high_water.hypre : 0;
      const std::uint64_t allocation_delta = allocation_nonnegative
          ? current_allocations - allocations : 0;
      const std::uint64_t owned_delta = owned_nonnegative
          ? current_owned_calls - owned_allocation_calls : 0;
      const std::uint64_t deallocation_delta = deallocation_nonnegative
          ? current_deallocations - owned_deallocation_events : 0;
      const bool identities_stable = initial_high_water.valid &&
          current_high_water.valid &&
          std::strcmp(initial_high_water.mfem_identity,
                      current_high_water.mfem_identity) == 0 &&
          std::strcmp(initial_high_water.hypre_identity,
                      current_high_water.hypre_identity) == 0;
      const char* origin = "invalid";
      if (cycle != 1)
        origin = "plateau";
      else if (mfem_delta == 0 && hypre_delta == 0)
        origin = "none";
      else if (mfem_delta != 0 && hypre_delta == 0)
        origin = "mfem";
      else if (mfem_delta == 0 && hypre_delta != 0)
        origin = "hypre";
      bool exact = initial_high_water.valid && current_high_water.valid &&
          identities_stable && mfem_nonnegative && hypre_nonnegative &&
          allocation_nonnegative && owned_nonnegative &&
          deallocation_nonnegative &&
          current_owned_calls == owned_allocation_calls &&
          current_deallocations == owned_deallocation_events;
      if (cycle == 1)
      {
        const bool counter_matches_pools = allocation_delta == 0
            ? mfem_delta == 0 && hypre_delta == 0
            : (mfem_delta != 0) != (hypre_delta != 0);
        exact = exact && std::strcmp(origin, "invalid") != 0 &&
            counter_matches_pools &&
            InitialAllocationSampleExact(allocations, current_allocations) &&
            (ranks == 2 || current_allocations == allocations);
        if (exact)
        {
          plateau_allocations = current_allocations;
          plateau_high_water = current_high_water;
        }
      }
      else
      {
        exact = exact && current_allocations == plateau_allocations &&
            current_high_water.mfem == plateau_high_water.mfem &&
            current_high_water.hypre == plateau_high_water.hypre &&
            std::strcmp(current_high_water.mfem_identity,
                        plateau_high_water.mfem_identity) == 0 &&
            std::strcmp(current_high_water.hypre_identity,
                        plateau_high_water.hypre_identity) == 0;
      }
      const int local_exact = exact ? 1 : 0;
      int global_exact = 0;
      const bool collective_exact = MPI_Allreduce(
          &local_exact, &global_exact, 1, MPI_INT, MPI_MIN,
          MPI_COMM_WORLD) == MPI_SUCCESS && global_exact == 1;
      if (cycle > 1 && collective_exact) { ++credited_plateau_cycles; }
      std::fprintf(stderr,
          "CUDA_ALLOCATION_PLATEAU rank=%d cycle=%llu origin=%s "
          "mfem_id=%s/%s mfem=%zu/%zu/%zu "
          "hypre_id=%s/%s hypre=%zu/%zu/%zu "
          "composite=%llu/%llu/%llu owned=%llu/%llu/%llu "
          "dealloc=%llu/%llu/%llu nonnegative=%u/%u/%u/%u/%u "
          "local=%s collective=%s\n",
          setup.rank, static_cast<unsigned long long>(cycle),
          origin, initial_high_water.mfem_identity,
          current_high_water.mfem_identity,
          initial_high_water.mfem, current_high_water.mfem, mfem_delta,
          initial_high_water.hypre_identity,
          current_high_water.hypre_identity,
          initial_high_water.hypre, current_high_water.hypre, hypre_delta,
          static_cast<unsigned long long>(allocations),
          static_cast<unsigned long long>(current_allocations),
          static_cast<unsigned long long>(allocation_delta),
          static_cast<unsigned long long>(owned_allocation_calls),
          static_cast<unsigned long long>(current_owned_calls),
          static_cast<unsigned long long>(owned_delta),
          static_cast<unsigned long long>(owned_deallocation_events),
          static_cast<unsigned long long>(current_deallocations),
          static_cast<unsigned long long>(deallocation_delta),
          mfem_nonnegative ? 1U : 0U, hypre_nonnegative ? 1U : 0U,
          allocation_nonnegative ? 1U : 0U,
          owned_nonnegative ? 1U : 0U,
          deallocation_nonnegative ? 1U : 0U,
          exact ? "PASS" : "FAIL",
          collective_exact ? "PASS" : "FAIL");
      return collective_exact;
    };
    CudaSerializedCheckpointV1 checkpoint{};
    const bool initial_commit_exact = CollectivelyExact(
        CommitSuccessfulAttempt(*engine, false, 1, ranks, setup.rank,
                                allocations, checkpoint, false, nullptr,
                                context, ranks == 2));
    if (!initial_commit_exact || !credited_allocation_cycle(1))
    {
      return false;
    }

    const auto* contract = engine->contract();
    const bool contract_collective = CollectivelyExact(contract != nullptr);
    if (!contract_collective) { return false; }
    const StateCoordinates old_coordinates = contract->coordinates();
    const auto old_publication = contract->Published();
    CudaPrecommitSnapshot failed_snapshot{};
    CudaStepEvidence failed{};
    CudaSerializedCheckpointV1 prohibited_checkpoint{};
    const bool prepare_return =
        engine->PrepareStep(false, failed_snapshot, failed);
    const bool prepare_post_local = prepare_return &&
        failed_snapshot.target.step == 2 &&
        failed.phase == CudaFailurePhase::after_recovery_before_commit &&
        failed.source_rank == ranks - 1 &&
        failed_snapshot.attempt.retry_index == 0 &&
        failed_snapshot.numerical_candidate_ready;
    const bool prepare_collective = CollectivelyExact(prepare_post_local);
    if (!prepare_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, prohibited_checkpoint, failed, failed_snapshot,
          "lifecycle-injected-prepare", ranks, setup.rank,
          FailureReason::backend_failure,
          CudaFailurePhase::after_recovery_before_commit,
          PhaseAfterPrepareStep(prepare_return, failed));
      return false;
    }
    const bool stage_return = StagePreparedCheckpointDiagnosed(
        *engine, prohibited_checkpoint, failed, failed_snapshot, ranks,
        false, 2, "lifecycle-injected-stage", setup.rank, false);
    const bool stage_failure_local = !stage_return &&
        !prohibited_checkpoint.candidate_frozen &&
        !prohibited_checkpoint.checkpoint_staged &&
        !prohibited_checkpoint.output_staged &&
        !prohibited_checkpoint.successful_commit;
    const bool stage_failure_collective =
        CollectivelyExact(stage_failure_local);
    if (!stage_failure_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, prohibited_checkpoint, failed, failed_snapshot,
          "lifecycle-injected-stage", ranks, setup.rank,
          FailureReason::backend_failure,
          CudaFailurePhase::after_recovery_before_commit,
          AffineLocalPhase::staging);
      return false;
    }
    const bool cleanup_exact = ResolveUnexpectedAffineExit(
        *engine, prohibited_checkpoint, failed, failed_snapshot,
        "lifecycle-injected-stage", ranks, setup.rank,
        FailureReason::backend_failure,
        CudaFailurePhase::after_recovery_before_commit,
        AffineLocalPhase::staging);
    const bool failure_post_local = cleanup_exact && !failed.committed &&
        failed.retry_available &&
        failed.contract_reason == FailureReason::backend_failure &&
        failed.status == CudaRunStatus::retry_available &&
        failed.source_rank == ranks - 1 && failed.collective_consensus &&
        failed.committed_bytes_preserved && failed.coordinates_preserved &&
        SameCoordinates(contract->coordinates(), old_coordinates) &&
        ExactAttemptEvidence(failed, ranks);
    const bool failure_post_collective =
        CollectivelyExact(failure_post_local);
    ReportExpectedNegativeEvidence(
        context, setup.rank, "lifecycle-injected-stage",
        FailureReason::backend_failure, CudaRunStatus::retry_available,
        failed, failure_post_local, failure_post_collective,
        &prohibited_checkpoint);
    if (!failure_post_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, prohibited_checkpoint, failed, failed_snapshot,
          "lifecycle-injected-postcondition", ranks, setup.rank,
          FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::consumed);
      return false;
    }
    StateCoordinates visible{};
    const bool failed_allocation_state_exact =
        CollectivelyExactAllocatorState(
            plateau_allocations, owned_allocation_calls,
            owned_deallocation_events);
    const bool failed_publication_local =
        contract->PublishedCoordinates(old_publication, visible) &&
        SameCoordinates(visible, old_coordinates) &&
        failed_allocation_state_exact && ExactPolicyLedgers();
    const bool failed_publication_collective =
        CollectivelyExact(failed_publication_local);
    if (!failed_publication_collective ||
        !failure_post_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, prohibited_checkpoint, failed, failed_snapshot,
          "lifecycle-injected-publication", ranks, setup.rank,
          FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::consumed);
      return false;
    }

    const bool retry_commit_exact = CollectivelyExact(
        CommitSuccessfulAttempt(*engine, true, 2, ranks, setup.rank,
                                allocations, checkpoint, false, nullptr,
                                context, ranks == 2));
    if (!retry_commit_exact || !credited_allocation_cycle(2))
    {
      return false;
    }
    const bool third_commit_exact = CollectivelyExact(
        CommitSuccessfulAttempt(*engine, false, 3, ranks, setup.rank,
                                allocations, final_checkpoint, true, nullptr,
                                context, ranks == 2));
    if (!third_commit_exact || !credited_allocation_cycle(3))
    {
      return false;
    }
    if (ranks == 2)
    {
      for (std::uint64_t target_step = 4; target_step <= 21; ++target_step)
      {
        CudaSerializedCheckpointV1 continued_checkpoint{};
        const bool continued_commit_exact = CollectivelyExact(
            CommitSuccessfulAttempt(
                *engine, false, target_step, ranks, setup.rank, allocations,
                continued_checkpoint, false, nullptr, context,
                ranks == 2));
        if (!continued_commit_exact ||
            !credited_allocation_cycle(target_step))
        {
          return false;
        }
        final_checkpoint = continued_checkpoint;
      }
    }
    contract = engine->contract();
    const std::uint64_t expected_step = ranks == 2 ? 21 : 3;
    const std::uint64_t expected_cursor = ranks == 2 ? 22 : 4;
    return contract != nullptr &&
        contract->coordinates().step == expected_step &&
        contract->coordinates().generation == expected_step &&
        contract->coordinates().cursor == expected_cursor &&
        std::abs(contract->coordinates().time -
                 fixed_dt * static_cast<double>(expected_step)) <= 1.0e-14 &&
        contract->telemetry().commits == expected_step &&
        (ranks != 2 || credited_plateau_cycles >= 20) &&
        contract->telemetry().last_attempt.prohibited.fallback_events == 0 &&
        final_checkpoint.successful_commit && ExactPolicyLedgers();
  }();
  engine.reset();
  return result && ExactPolicyLedgers();
}

template <typename Backend>
bool RunCorruptRestore(const int ranks,
                       const CudaSerializedCheckpointV1& checkpoint,
                       const char* const context) noexcept
{
  auto engine = std::make_unique<TestEngine<Backend>>(Config(ranks));
  const bool result = [&]() noexcept
  {
    CudaRunEvidence setup{};
    if (!InitializePositive(*engine, setup, ranks, context))
      return false;
    const std::uint64_t owned_allocation_calls =
        setup.reuse.baseline_w8_owned_allocation_calls;
    const std::uint64_t deallocation_events =
        AuditedAllocator::DeallocationEvents();
    CudaSerializedCheckpointV1 corrupt = checkpoint;
    const bool corrupt_shape_collective =
        CollectivelyExact(corrupt.interior_count != 0);
    if (!corrupt_shape_collective) { return false; }
    corrupt.interior[0] += 1.0 + std::abs(corrupt.interior[0]);
    CudaStepEvidence evidence{};
    const auto* contract = engine->contract();
    const bool contract_collective = CollectivelyExact(contract != nullptr);
    if (!contract_collective) { return false; }
    const StateCoordinates before = contract->coordinates();
    const bool restore_return = engine->Restore(corrupt, evidence);
    const bool allocation_state_exact = CollectivelyExactAllocatorState(
        setup.reuse.baseline_allocation_events, owned_allocation_calls,
        deallocation_events);
    const bool restore_local = !restore_return &&
        evidence.contract_reason == FailureReason::restart_incompatible &&
        evidence.status == CudaRunStatus::runtime_failure &&
        !evidence.committed &&
        SameCoordinates(contract->coordinates(), before) &&
        allocation_state_exact &&
        ValidRange(evidence.transfer_range, AuditedTransport::Ledger()) &&
        ExactPolicyLedgers();
    const bool restore_collective = CollectivelyExact(restore_local);
    ReportExpectedNegativeEvidence(
        context, setup.rank, "corrupt-restore",
        FailureReason::restart_incompatible,
        CudaRunStatus::runtime_failure, evidence, restore_local,
        restore_collective, &corrupt);
    return restore_collective;
  }();
  engine.reset();
  return result && ExactPolicyLedgers();
}

template <typename Backend>
bool RunFreshRestore(const int ranks,
                     const CudaSerializedCheckpointV1& checkpoint,
                     const char* const context) noexcept
{
  auto engine = std::make_unique<TestEngine<Backend>>(Config(ranks, true));
  const bool result = [&]() noexcept
  {
    CudaRunEvidence setup{};
    if (!InitializePositive(*engine, setup, ranks, context))
      return false;
    const std::uint64_t owned_allocation_calls =
        setup.reuse.baseline_w8_owned_allocation_calls;
    const std::uint64_t deallocation_events =
        AuditedAllocator::DeallocationEvents();
    const auto* pristine_contract = engine->contract();
    const bool pristine_contract_collective =
        CollectivelyExact(pristine_contract != nullptr);
    if (!pristine_contract_collective) { return false; }
    const auto stale_pre_restore_publication =
        pristine_contract->Published();
    const CheckpointDescriptor source_descriptor = checkpoint.descriptor;
    CudaStepEvidence evidence{};
    const bool restore_return = engine->Restore(checkpoint, evidence);
    const bool allocation_state_exact = CollectivelyExactAllocatorState(
        setup.reuse.baseline_allocation_events, owned_allocation_calls,
        deallocation_events);
    const bool restore_post_local = restore_return && evidence.committed &&
        evidence.status == CudaRunStatus::committed &&
        evidence.contract_reason == FailureReason::none &&
        evidence.after_interior_hash == checkpoint.descriptor.interior_hash &&
        evidence.after_trace_hash == checkpoint.descriptor.trace_hash &&
        SameCoordinates(evidence.after, checkpoint.descriptor.coordinates) &&
        ValidRange(evidence.transfer_range, AuditedTransport::Ledger()) &&
        allocation_state_exact && ExactPolicyLedgers();
    const bool restore_post_collective =
        CollectivelyExact(restore_post_local);
    if (!restore_post_collective) { return false; }
    const auto* contract = engine->contract();
    StateCoordinates stale_visible{};
    const bool restored_contract_local = contract != nullptr &&
        SameCoordinates(contract->coordinates(),
                        checkpoint.descriptor.coordinates) &&
        !contract->PublishedCoordinates(stale_pre_restore_publication,
                                        stale_visible) &&
        dg::hdg::runtime::detail::SameCheckpoint(
            source_descriptor, checkpoint.descriptor);
    const bool restored_contract_collective =
        CollectivelyExact(restored_contract_local);
    if (!restored_contract_collective) { return false; }
    const StateCoordinates restored = contract->coordinates();
    const auto restored_publication = contract->Published();
    CudaSerializedCheckpointV1 continued_checkpoint{};
    CudaPrecommitSnapshot continued{};
    if (!CommitSuccessfulAttempt(
            *engine, false, restored.step + 1, ranks, setup.rank,
            setup.reuse.baseline_allocation_events, continued_checkpoint,
            false, &continued, context))
    {
      return false;
    }
    contract = engine->contract();
    const bool continuation_local = contract != nullptr &&
        continued.attempt.sequence > restored.cursor &&
        continued.attempt.committed_step == restored.step &&
        continued.attempt.committed_generation == restored.generation &&
        continued.target.cursor == continued.attempt.sequence &&
        continued.target.cursor > restored.cursor &&
        continued_checkpoint.attempt.sequence == continued.attempt.sequence &&
        continued_checkpoint.descriptor.coordinates.cursor ==
            continued.target.cursor &&
        continued_checkpoint.descriptor.coordinates.step ==
            restored.step + 1 &&
        !contract->PublishedCoordinates(restored_publication, stale_visible) &&
        dg::hdg::runtime::detail::SameCheckpoint(
            source_descriptor, checkpoint.descriptor);
    return CollectivelyExact(continuation_local);
  }();
  engine.reset();
  return result && ExactPolicyLedgers();
}

template <typename Backend>
bool RunProvisionalRevert(const int ranks,
                          const char* const context) noexcept
{
  auto engine = std::make_unique<TestEngine<Backend>>(Config(ranks, true));
  const bool result = [&]() noexcept
  {
    CudaRunEvidence setup{};
    if (!InitializePositive(*engine, setup, ranks, context))
      return false;
    const auto* contract = engine->contract();
    if (!CollectivelyExact(contract != nullptr)) { return false; }
    const std::uint64_t allocations = setup.reuse.baseline_allocation_events;
    const std::uint64_t owned_allocation_calls =
        setup.reuse.baseline_w8_owned_allocation_calls;
    const std::uint64_t deallocation_events =
        AuditedAllocator::DeallocationEvents();
    const StateCoordinates old_coordinates = contract->coordinates();
    const auto old_publication = contract->Published();
    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    CudaSerializedCheckpointV1 checkpoint{};
    const bool prepare_return = engine->PrepareStep(false, snapshot, evidence);
    const bool prepare_local = prepare_return &&
        snapshot.target.step == 1 && snapshot.numerical_candidate_ready &&
        SameCoordinates(snapshot.committed, old_coordinates);
    const bool prepare_collective = CollectivelyExact(prepare_local);
    const bool prepare_allocator_collective =
        CollectivelyExactAllocatorState(
            allocations, owned_allocation_calls, deallocation_events);
    if (!prepare_collective || !prepare_allocator_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareStep(prepare_return, evidence));
      return false;
    }
    const bool checkpoint_collective = StagePreparedCheckpointDiagnosed(
        *engine, checkpoint, evidence, snapshot, ranks, false, 1,
        context, setup.rank);
    if (!checkpoint_collective) { return false; }
    const FrozenStageResult oracle_output = StageOracleOutputCollective(
        *engine, checkpoint, evidence, snapshot, ranks, setup.rank, false,
        1, false, context);
    if (!oracle_output.passed) { return false; }

    const bool prepare_commit_return =
        engine->PrepareCommit(checkpoint, evidence);
    const bool prepare_commit_collective =
        CollectivelyExact(prepare_commit_return);
    if (!prepare_commit_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareCommit(prepare_commit_return, evidence));
      return false;
    }
    const bool prepared_state_local = checkpoint.candidate_frozen &&
        checkpoint.checkpoint_staged && checkpoint.output_staged &&
        !checkpoint.successful_commit && !evidence.committed &&
        SameCoordinates(contract->coordinates(), old_coordinates);
    const bool prepared_state_collective =
        CollectivelyExact(prepared_state_local);
    if (!prepared_state_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareCommit(prepare_commit_return, evidence));
      return false;
    }

    const bool commit_prepared_return =
        engine->CommitPrepared(checkpoint, evidence);
    const bool commit_prepared_collective =
        CollectivelyExact(commit_prepared_return);
    if (!commit_prepared_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterCommitPrepared(commit_prepared_return, evidence));
      return false;
    }
    const bool provisional_state_local = !checkpoint.successful_commit &&
        SameCoordinates(contract->coordinates(), old_coordinates);
    const bool provisional_state_collective =
        CollectivelyExact(provisional_state_local);
    if (!provisional_state_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterCommitPrepared(commit_prepared_return, evidence));
      return false;
    }
    StateCoordinates visible{};
    const bool provisional_publication_local =
        contract->PublishedCoordinates(old_publication, visible) &&
        SameCoordinates(visible, old_coordinates);
    const bool provisional_publication_collective =
        CollectivelyExact(provisional_publication_local);
    if (!provisional_publication_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::provisional);
      return false;
    }
    const bool cleanup_exact = ResolveUnexpectedAffineExit(
        *engine, checkpoint, evidence, snapshot, context, ranks,
        setup.rank, FailureReason::backend_failure,
        CudaFailurePhase::diagnostics, AffineLocalPhase::provisional);
    const bool revert_local = cleanup_exact &&
        contract->PublishedCoordinates(old_publication, visible) &&
        SameCoordinates(visible, old_coordinates) &&
        ExactAttemptEvidence(evidence, ranks);
    const bool revert_collective = CollectivelyExact(revert_local);
    ReportExpectedNegativeEvidence(
        context, setup.rank, "revert-provisional",
        FailureReason::backend_failure, CudaRunStatus::retry_available,
        evidence, revert_local, revert_collective, &checkpoint);
    const bool revert_allocator_collective =
        CollectivelyExactAllocatorState(
            allocations, owned_allocation_calls, deallocation_events);
    if (!revert_collective || !revert_allocator_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::consumed);
      return false;
    }
    CudaOraclePod unavailable{};
    const bool unavailable_local = !engine->ExportOracle(unavailable);
    const bool unavailable_collective = CollectivelyExact(unavailable_local);
    if (!unavailable_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::consumed);
      return false;
    }
    const bool retry_commit = CommitSuccessfulAttempt(
        *engine, true, 1, ranks, setup.rank, allocations, checkpoint, true,
        nullptr, context);
    const bool retry_commit_collective = CollectivelyExact(retry_commit);
    const auto* final_contract = engine->contract();
    const bool telemetry_collective = CollectivelyExact(
        final_contract != nullptr &&
        final_contract->telemetry().commits == 1);
    return retry_commit_collective && telemetry_collective;
  }();
  engine.reset();
  return result && ExactPolicyLedgers();
}

template <typename Backend>
bool RunPreparedAbort(const int ranks,
                      const char* const context) noexcept
{
  auto engine = std::make_unique<TestEngine<Backend>>(Config(ranks, true));
  const bool result = [&]() noexcept
  {
    CudaRunEvidence setup{};
    if (!InitializePositive(*engine, setup, ranks, context))
      return false;
    const auto* contract = engine->contract();
    if (!CollectivelyExact(contract != nullptr)) { return false; }
    const std::uint64_t allocations = setup.reuse.baseline_allocation_events;
    const std::uint64_t owned_allocation_calls =
        setup.reuse.baseline_w8_owned_allocation_calls;
    const std::uint64_t deallocation_events =
        AuditedAllocator::DeallocationEvents();
    const StateCoordinates old_coordinates = contract->coordinates();
    const auto old_publication = contract->Published();
    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    CudaSerializedCheckpointV1 checkpoint{};
    const bool prepare_return = engine->PrepareStep(false, snapshot, evidence);
    const bool prepare_local = prepare_return &&
        snapshot.target.step == 1 && snapshot.numerical_candidate_ready &&
        SameCoordinates(snapshot.committed, old_coordinates);
    const bool prepare_collective = CollectivelyExact(prepare_local);
    const bool prepare_allocator_collective =
        CollectivelyExactAllocatorState(
            allocations, owned_allocation_calls, deallocation_events);
    if (!prepare_collective || !prepare_allocator_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareStep(prepare_return, evidence));
      return false;
    }
    const bool checkpoint_collective = StagePreparedCheckpointDiagnosed(
        *engine, checkpoint, evidence, snapshot, ranks, false, 1,
        context, setup.rank);
    if (!checkpoint_collective) { return false; }
    const FrozenStageResult oracle_output = StageOracleOutputCollective(
        *engine, checkpoint, evidence, snapshot, ranks, setup.rank, false,
        1, false, context);
    if (!oracle_output.passed) { return false; }
    const bool prepare_commit_return =
        engine->PrepareCommit(checkpoint, evidence);
    const bool prepare_commit_collective =
        CollectivelyExact(prepare_commit_return);
    if (!prepare_commit_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareCommit(prepare_commit_return, evidence));
      return false;
    }
    const bool prepared_local = !evidence.committed &&
        checkpoint.candidate_frozen && checkpoint.checkpoint_staged &&
        checkpoint.output_staged && !checkpoint.successful_commit &&
        SameCoordinates(contract->coordinates(), old_coordinates);
    const bool prepared_collective = CollectivelyExact(prepared_local);
    if (!prepared_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareCommit(prepare_commit_return, evidence));
      return false;
    }
    const bool cleanup_exact = ResolveUnexpectedAffineExit(
        *engine, checkpoint, evidence, snapshot, context, ranks,
        setup.rank, FailureReason::backend_failure,
        CudaFailurePhase::diagnostics, AffineLocalPhase::prepared);
    const bool abort_local = cleanup_exact &&
        SameCoordinates(contract->coordinates(), old_coordinates) &&
        ExactAttemptEvidence(evidence, ranks);
    const bool abort_collective = CollectivelyExact(abort_local);
    ReportExpectedNegativeEvidence(
        context, setup.rank, "abort-prepared",
        FailureReason::backend_failure, CudaRunStatus::retry_available,
        evidence, abort_local, abort_collective, &checkpoint);
    const bool abort_allocator_collective =
        CollectivelyExactAllocatorState(
            allocations, owned_allocation_calls, deallocation_events);
    if (!abort_collective || !abort_allocator_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::consumed);
      return false;
    }
    StateCoordinates visible{};
    const bool publication_local =
        contract->PublishedCoordinates(old_publication, visible) &&
        SameCoordinates(visible, old_coordinates) &&
        contract->telemetry().commits == 0;
    const bool publication_collective = CollectivelyExact(publication_local);
    if (!publication_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::consumed);
      return false;
    }
    const bool retry_commit = CommitSuccessfulAttempt(
        *engine, true, 1, ranks, setup.rank, allocations, checkpoint, true,
        nullptr, context);
    const bool retry_commit_collective = CollectivelyExact(retry_commit);
    const auto* final_contract = engine->contract();
    const bool telemetry_collective = CollectivelyExact(
        final_contract != nullptr &&
        final_contract->telemetry().commits == 1);
    return retry_commit_collective && telemetry_collective;
  }();
  engine.reset();
  return result && ExactPolicyLedgers();
}

template <typename Backend>
bool RunStagingFailure(const int ranks,
                       const char* const context) noexcept
{
  auto engine = std::make_unique<TestEngine<Backend>>(Config(ranks, true));
  const bool result = [&]() noexcept
  {
    CudaRunEvidence setup{};
    if (!InitializePositive(*engine, setup, ranks, context))
      return false;
    const auto* contract = engine->contract();
    if (!CollectivelyExact(contract != nullptr)) { return false; }
    const std::uint64_t allocations = setup.reuse.baseline_allocation_events;
    const std::uint64_t owned_allocation_calls =
        setup.reuse.baseline_w8_owned_allocation_calls;
    const std::uint64_t deallocation_events =
        AuditedAllocator::DeallocationEvents();
    const StateCoordinates old_coordinates = contract->coordinates();
    const auto old_publication = contract->Published();
    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    CudaSerializedCheckpointV1 checkpoint{};
    const bool prepare_return = engine->PrepareStep(false, snapshot, evidence);
    const bool prepare_local = prepare_return &&
        snapshot.target.step == 1 && snapshot.numerical_candidate_ready &&
        SameCoordinates(snapshot.committed, old_coordinates);
    const bool prepare_collective = CollectivelyExact(prepare_local);
    const bool prepare_allocator_collective =
        CollectivelyExactAllocatorState(
            allocations, owned_allocation_calls, deallocation_events);
    if (!prepare_collective || !prepare_allocator_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareStep(prepare_return, evidence));
      return false;
    }
    AuditedTransport::fail_checkpoint_trace_once = true;
    const bool staged = StagePreparedCheckpointDiagnosed(
        *engine, checkpoint, evidence, snapshot, ranks, false, 1, context,
        setup.rank, false);
    const bool staging_failure_local = !staged &&
        !AuditedTransport::fail_checkpoint_trace_once &&
        !checkpoint.checkpoint_staged && !checkpoint.output_staged &&
        !checkpoint.successful_commit && !checkpoint.staging_range.complete;
    const bool staging_failure_collective =
        CollectivelyExact(staging_failure_local);
    if (!staging_failure_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::staging);
      return false;
    }
    const bool cleanup_exact = ResolveUnexpectedAffineExit(
        *engine, checkpoint, evidence, snapshot, context, ranks,
        setup.rank, FailureReason::backend_failure,
        CudaFailurePhase::diagnostics, AffineLocalPhase::staging);
    const bool failed_staging_local = cleanup_exact &&
        SameCoordinates(contract->coordinates(), old_coordinates) &&
        ExactFailedStagingEvidence(evidence, ranks);
    const bool failed_staging_collective =
        CollectivelyExact(failed_staging_local);
    ReportExpectedNegativeEvidence(
        context, setup.rank, "checkpoint-staging",
        FailureReason::backend_failure, CudaRunStatus::retry_available,
        evidence, failed_staging_local, failed_staging_collective,
        &checkpoint);
    const bool failure_allocator_collective =
        CollectivelyExactAllocatorState(
            allocations, owned_allocation_calls, deallocation_events);
    if (!failed_staging_collective || !failure_allocator_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::consumed);
      return false;
    }
    StateCoordinates visible{};
    const bool publication_local =
        contract->PublishedCoordinates(old_publication, visible) &&
        SameCoordinates(visible, old_coordinates) &&
        contract->telemetry().commits == 0;
    const bool publication_collective = CollectivelyExact(publication_local);
    if (!publication_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::consumed);
      return false;
    }
    const bool retry_commit = CommitSuccessfulAttempt(
        *engine, true, 1, ranks, setup.rank, allocations, checkpoint, true,
        nullptr, context);
    const bool retry_commit_collective = CollectivelyExact(retry_commit);
    const auto* final_contract = engine->contract();
    const bool telemetry_collective = CollectivelyExact(
        final_contract != nullptr &&
        final_contract->telemetry().commits == 1);
    return retry_commit_collective && telemetry_collective;
  }();
  engine.reset();
  return result && ExactPolicyLedgers();
}

enum class FrozenAuthorityNegative : unsigned char
{
  oracle_export,
  output_mark,
  finalize_descriptor
};

template <typename Backend>
bool RunFrozenAuthorityNegative(const int ranks, const int rank,
                                const FrozenAuthorityNegative negative,
                                const char* const context) noexcept
{
  auto engine = std::make_unique<TestEngine<Backend>>(Config(ranks, true));
  const bool result = [&]() noexcept
  {
    CudaRunEvidence setup{};
    if (!InitializePositive(*engine, setup, ranks, context))
      return false;
    const auto* contract = engine->contract();
    if (!CollectivelyExact(contract != nullptr)) { return false; }
    const std::uint64_t allocations =
        setup.reuse.baseline_allocation_events;
    const std::uint64_t owned_allocation_calls =
        setup.reuse.baseline_w8_owned_allocation_calls;
    const std::uint64_t deallocation_events =
        AuditedAllocator::DeallocationEvents();
    const StateCoordinates old_coordinates = contract->coordinates();
    const auto old_publication = contract->Published();
    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    CudaSerializedCheckpointV1 checkpoint{};

    const bool prepare_return = engine->PrepareStep(false, snapshot, evidence);
    const bool prepare_local = prepare_return &&
        snapshot.target.step == 1 && snapshot.numerical_candidate_ready &&
        SameCoordinates(snapshot.committed, old_coordinates);
    const bool prepare_collective = CollectivelyExact(prepare_local);
    const bool prepare_allocator_collective =
        CollectivelyExactAllocatorState(
            allocations, owned_allocation_calls, deallocation_events);
    if (!prepare_collective || !prepare_allocator_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareStep(prepare_return, evidence));
      return false;
    }

    const bool checkpoint_collective = StagePreparedCheckpointDiagnosed(
        *engine, checkpoint, evidence, snapshot, ranks, false, 1,
        context, setup.rank);
    if (!checkpoint_collective) { return false; }

    const int source_rank = ranks - 1;
    if (negative == FrozenAuthorityNegative::oracle_export &&
        rank == source_rank)
    {
      AuditedTransport::fail_oracle_export_once = true;
    }
    const OracleOutputTestSeam seam{
        negative == FrozenAuthorityNegative::output_mark ? source_rank : -1};
    const FrozenStageResult oracle_output = StageOracleOutputCollective(
        *engine, checkpoint, evidence, snapshot, ranks, setup.rank, false,
        1, false, context, seam);

    bool negative_collective = false;
    if (negative == FrozenAuthorityNegative::oracle_export ||
        negative == FrozenAuthorityNegative::output_mark)
    {
      const FailureReason expected_reason =
          negative == FrozenAuthorityNegative::oracle_export
          ? FailureReason::backend_failure
          : FailureReason::collective_protocol;
      const std::uint64_t expected_injections =
          negative == FrozenAuthorityNegative::oracle_export &&
              rank == source_rank
          ? 1 : 0;
      StateCoordinates visible{};
      const bool negative_local = !oracle_output.passed &&
          oracle_output.cleanup_exact &&
          AuditedTransport::injected_oracle_export_failures ==
              expected_injections &&
          !AuditedTransport::fail_oracle_export_once &&
          evidence.contract_reason == expected_reason &&
          evidence.status == CudaRunStatus::retry_available &&
          evidence.retry_available && !evidence.committed &&
          evidence.coordinates_preserved &&
          evidence.committed_bytes_preserved &&
          !checkpoint.candidate_frozen &&
          !checkpoint.checkpoint_staged && !checkpoint.output_staged &&
          !checkpoint.successful_commit &&
          SameCoordinates(contract->coordinates(), old_coordinates) &&
          contract->PublishedCoordinates(old_publication, visible) &&
          SameCoordinates(visible, old_coordinates) &&
          (negative == FrozenAuthorityNegative::oracle_export ||
           ExactAttemptEvidence(evidence, ranks)) &&
          ExactPolicyLedgers();
      negative_collective = CollectivelyExact(negative_local);
      ReportExpectedNegativeEvidence(
          context, setup.rank,
          negative == FrozenAuthorityNegative::oracle_export
              ? "source-rank-oracle-export" : "output-mark-mismatch",
          expected_reason, CudaRunStatus::retry_available, evidence,
          negative_local, negative_collective, &checkpoint);
    }
    else
    {
      if (!oracle_output.passed) { return false; }
      const bool prepare_commit_return =
          engine->PrepareCommit(checkpoint, evidence);
      const bool prepare_commit_collective =
          CollectivelyExact(prepare_commit_return);
      if (!prepare_commit_collective)
      {
        (void)ResolveUnexpectedAffineExit(
            *engine, checkpoint, evidence, snapshot, context, ranks,
            setup.rank, FailureReason::backend_failure,
            CudaFailurePhase::diagnostics,
            PhaseAfterPrepareCommit(prepare_commit_return, evidence));
        return false;
      }
      const bool prepared_local = checkpoint.candidate_frozen &&
          checkpoint.checkpoint_staged && checkpoint.output_staged &&
          !checkpoint.successful_commit && !evidence.committed;
      const bool prepared_collective = CollectivelyExact(prepared_local);
      if (!prepared_collective)
      {
        (void)ResolveUnexpectedAffineExit(
            *engine, checkpoint, evidence, snapshot, context, ranks,
            setup.rank, FailureReason::backend_failure,
            CudaFailurePhase::diagnostics,
            PhaseAfterPrepareCommit(prepare_commit_return, evidence));
        return false;
      }
      const bool commit_prepared_return =
          engine->CommitPrepared(checkpoint, evidence);
      const bool commit_prepared_collective =
          CollectivelyExact(commit_prepared_return);
      if (!commit_prepared_collective)
      {
        (void)ResolveUnexpectedAffineExit(
            *engine, checkpoint, evidence, snapshot, context, ranks,
            setup.rank, FailureReason::backend_failure,
            CudaFailurePhase::diagnostics,
            PhaseAfterCommitPrepared(commit_prepared_return, evidence));
        return false;
      }
      if (rank == source_rank)
        checkpoint.descriptor.trace_hash.bytes[0] ^= 1U;
      const bool finalize_return =
          engine->FinalizeCommit(checkpoint, evidence);
      const bool finalize_return_collective =
          CollectivelyExact(finalize_return);
      const bool finalize_affine_exit = ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::collective_protocol,
          CudaFailurePhase::diagnostics,
          PhaseAfterFinalizeCommit(evidence));
      StateCoordinates visible{};
      const bool finalize_local = finalize_return_collective &&
          finalize_affine_exit &&
          evidence.contract_reason == FailureReason::collective_protocol &&
          evidence.status == CudaRunStatus::retry_available &&
          evidence.retry_available && !evidence.committed &&
          evidence.coordinates_preserved &&
          evidence.committed_bytes_preserved &&
          evidence.before_interior_hash == evidence.after_interior_hash &&
          evidence.before_trace_hash == evidence.after_trace_hash &&
          !checkpoint.candidate_frozen &&
          !checkpoint.checkpoint_staged && !checkpoint.output_staged &&
          !checkpoint.successful_commit &&
          SameCoordinates(contract->coordinates(), old_coordinates) &&
          contract->PublishedCoordinates(old_publication, visible) &&
          SameCoordinates(visible, old_coordinates) &&
          ExactAttemptEvidence(evidence, ranks) && ExactPolicyLedgers();
      negative_collective = CollectivelyExact(finalize_local);
      ReportExpectedNegativeEvidence(
          context, setup.rank, "finalize-descriptor-mismatch",
          FailureReason::collective_protocol,
          CudaRunStatus::retry_available, evidence, finalize_local,
          negative_collective, &checkpoint);
    }
    const bool failure_allocator_collective =
        CollectivelyExactAllocatorState(
            allocations, owned_allocation_calls, deallocation_events);
    if (!negative_collective || !failure_allocator_collective)
    {
      if (negative_collective)
      {
        (void)ResolveUnexpectedAffineExit(
            *engine, checkpoint, evidence, snapshot, context, ranks,
            setup.rank, FailureReason::backend_failure,
            CudaFailurePhase::diagnostics,
            AffineLocalPhase::consumed);
      }
      return false;
    }

    CudaSerializedCheckpointV1 retry_checkpoint{};
    const bool retry_return = CommitSuccessfulAttempt(
        *engine, true, 1, ranks, setup.rank, allocations, retry_checkpoint,
        true, nullptr, context, ranks == 2);
    const bool retry_collective = CollectivelyExact(retry_return);
    const auto* final_contract = engine->contract();
    const bool final_collective = CollectivelyExact(
        final_contract != nullptr &&
        final_contract->telemetry().commits == 1 &&
        retry_checkpoint.successful_commit && ExactPolicyLedgers());
    return retry_collective && final_collective;
  }();
  engine.reset();
  return result && ExactPolicyLedgers();
}

template <typename Backend>
bool RunTokenFault(const int ranks, const CudaTokenFault fault,
                   const FailureReason expected,
                   const char* const context) noexcept
{
  CudaConfig config = Config(ranks, true);
  config.injection.source_rank = ranks - 1;
  config.injection.token_fault = fault;
  auto engine = std::make_unique<TestEngine<Backend>>(config);
  const bool result = [&]() noexcept
  {
    CudaRunEvidence setup{};
    if (!InitializePositive(*engine, setup, ranks, context))
      return false;
    const std::uint64_t owned_allocation_calls =
        setup.reuse.baseline_w8_owned_allocation_calls;
    const std::uint64_t deallocation_events =
        AuditedAllocator::DeallocationEvents();
    if (fault == CudaTokenFault::issue_matrix_before_residual)
    {
      CudaSerializedCheckpointV1 checkpoint{};
      const bool commit_return = CommitSuccessfulAttempt(
          *engine, false, 1, ranks, setup.rank,
          setup.reuse.baseline_allocation_events, checkpoint, true, nullptr,
          context);
      const bool commit_collective = CollectivelyExact(commit_return);
      const auto* contract = engine->contract();
      const RunTelemetry telemetry = contract == nullptr
          ? RunTelemetry{} : contract->telemetry();
      const bool final_local = commit_collective && contract != nullptr &&
          contract->status() == RunStatus::committed &&
          telemetry.last_attempt.failure_reason == expected &&
          telemetry.accepted.valid &&
          telemetry.accepted.attempt.failure_reason == expected &&
          dg::hdg::runtime::detail::SameAttempt(
              telemetry.last_attempt.identity,
              telemetry.accepted.attempt.identity) &&
          dg::hdg::runtime::detail::SameAttempt(
              telemetry.last_attempt.identity, checkpoint.attempt) &&
          SameCoordinates(
              telemetry.accepted.coordinates,
              checkpoint.descriptor.coordinates) &&
          telemetry.last_attempt.validated_product_mask == 0x0fU &&
          telemetry.accepted.attempt.validated_product_mask == 0x0fU &&
          telemetry.commits == 1 &&
          telemetry.out_of_order_product_issues_rejected == 1 &&
          telemetry.last_attempt.prohibited.fallback_events == 0 &&
          checkpoint.successful_commit;
      const bool final_collective = CollectivelyExact(final_local);
      if (contract != nullptr)
      {
        ReportAttemptTelemetryEvidence(
            context, setup.rank, "matrix-before-residual",
            expected, RunStatus::committed, telemetry.last_attempt,
            telemetry.accepted, contract->status(), telemetry.commits,
            telemetry.out_of_order_product_issues_rejected,
            final_local, final_collective);
      }
      if (!final_collective)
        ReportTokenFailure(setup.rank, "final", fault);
      return final_collective;
    }

    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    CudaSerializedCheckpointV1 checkpoint{};
    const bool prepare_return = engine->PrepareStep(false, snapshot, evidence);
    const bool prepare_local = prepare_return &&
        snapshot.target.step == 1 && snapshot.numerical_candidate_ready;
    const bool prepare_collective = CollectivelyExact(prepare_local);
    if (!prepare_collective)
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics,
          PhaseAfterPrepareStep(prepare_return, evidence));
      ReportTokenFailure(setup.rank, "precommit", fault);
      return false;
    }
    if (!CollectivelyExactAllocatorState(
            setup.reuse.baseline_allocation_events,
            owned_allocation_calls, deallocation_events))
    {
      (void)ResolveUnexpectedAffineExit(
          *engine, checkpoint, evidence, snapshot, context, ranks,
          setup.rank, FailureReason::backend_failure,
          CudaFailurePhase::diagnostics, AffineLocalPhase::staging);
      ReportTokenFailure(setup.rank, "allocation-plateau", fault);
      return false;
    }
    const bool checkpoint_collective = StagePreparedCheckpointDiagnosed(
        *engine, checkpoint, evidence, snapshot, ranks, false, 1,
        context, setup.rank);
    if (!checkpoint_collective)
    {
      ReportTokenFailure(setup.rank, "precommit", fault);
      return false;
    }
    const FrozenStageResult oracle_output = StageOracleOutputCollective(
        *engine, checkpoint, evidence, snapshot, ranks, setup.rank, false,
        1, false, context);
    if (!oracle_output.passed)
    {
      ReportTokenFailure(setup.rank, "precommit", fault);
      return false;
    }
    const bool progressed_to_rollback =
        engine->PrepareCommit(checkpoint, evidence);
    const bool rollback_return_collective =
        CollectivelyExact(progressed_to_rollback);
    const bool affine_exit_exact = ResolveUnexpectedAffineExit(
        *engine, checkpoint, evidence, snapshot, context, ranks,
        setup.rank, expected, CudaFailurePhase::diagnostics,
        PhaseAfterPrepareCommit(progressed_to_rollback, evidence));
    const auto* contract = engine->contract();
    const bool collective_allocation_plateau_exact =
        CollectivelyExactAllocatorState(
            setup.reuse.baseline_allocation_events,
            owned_allocation_calls, deallocation_events);
    const bool final_local = rollback_return_collective &&
        affine_exit_exact &&
        contract != nullptr &&
        evidence.contract_reason == expected && evidence.retry_available &&
        !evidence.committed && evidence.coordinates_preserved &&
        evidence.committed_bytes_preserved &&
        !checkpoint.candidate_frozen && !checkpoint.checkpoint_staged &&
        !checkpoint.output_staged && !checkpoint.successful_commit &&
        contract->status() == RunStatus::retry_available &&
        contract->telemetry().commits == 0 &&
        contract->telemetry().last_attempt.prohibited.fallback_events == 0 &&
        collective_allocation_plateau_exact &&
        ExactAttemptEvidence(evidence, ranks) && ExactPolicyLedgers();
    const bool final_collective = CollectivelyExact(final_local);
    ReportExpectedNegativeEvidence(
        context, setup.rank, "prepare-commit-token-fault", expected,
        CudaRunStatus::retry_available, evidence, final_local,
        final_collective, &checkpoint);
    if (!final_collective)
    {
      ReportTokenFailure(setup.rank, "final", fault);
    }
    return final_collective;
  }();
  engine.reset();
  return result && ExactPolicyLedgers();
}

// Direct, test-local exercise of the frozen shared affine contract.  These
// values and policies own no CUDA state and expose no production authority.
using TinyInterior = std::array<double, 2>;
using TinyTrace = std::array<double, 2>;
using TinySlot = CompleteStateSlot<TinyInterior, TinyTrace>;

StateHash TinyHash(const TinyInterior& values, const unsigned offset) noexcept
{
  StateHash hash{};
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    hash.bytes[index] = static_cast<std::uint8_t>(
        static_cast<unsigned>(values[index % values.size()] * 7.0) +
        offset + static_cast<unsigned>(index));
  }
  return hash;
}

struct TinyStorage
{
  static constexpr bool preparation_preserves_committed = true;
  static constexpr bool private_copy_is_independent = true;
  static constexpr bool publish_is_atomic = true;
  static constexpr bool publish_preserves_disjoint_backing = true;
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;

  static bool PrepareCandidate(const TinySlot& source,
                               TinySlot& destination) noexcept
  { destination = source; return true; }
  static bool PrepareRestore(const TinySlot& source,
                             TinySlot& destination) noexcept
  { destination = source; return true; }
  static bool StoreCandidate(TinySlot& destination,
                             const TinySlot& source) noexcept
  { destination = source; return true; }
  static bool CopyOut(const TinySlot& source, TinySlot& destination) noexcept
  { destination = source; return true; }
  static void DiscardCandidate(TinySlot& candidate) noexcept
  { candidate = {}; }
  static bool CanPublish(const TinySlot& committed,
                         const TinySlot& candidate) noexcept
  { return Disjoint(committed, candidate); }
  static void Publish(TinySlot& committed, TinySlot& candidate) noexcept
  { using std::swap; swap(committed, candidate); }
  static void RevertPublish(TinySlot& committed,
                            TinySlot& candidate) noexcept
  { using std::swap; swap(committed, candidate); }
  static bool Disjoint(const TinySlot& left, const TinySlot& right) noexcept
  {
    return left.interior.data() != right.interior.data() &&
        left.trace.data() != right.trace.data();
  }
  static StateHash HashInterior(const TinyInterior& values) noexcept
  { return TinyHash(values, 0); }
  static StateHash HashTrace(const TinyTrace& values) noexcept
  { return TinyHash(values, 91); }
};

constexpr std::uint64_t ReasonBit(const FailureReason reason) noexcept
{
  const auto value = static_cast<unsigned>(reason);
  return reason != FailureReason::none &&
          value < static_cast<unsigned>(FailureReason::count)
      ? (std::uint64_t{1} << value) : 0;
}

struct TinyCollective
{
  static CollectiveAdmissionReport Admit(
      const AttemptCollectiveContext& context, const bool prepared,
      const FailureReason reason, const std::uint32_t ranks) noexcept
  {
    return {context, ranks, true, true, prepared,
            reason != FailureReason::none, ReasonBit(reason), reason};
  }
  static CollectiveGateReport Reduce(
      const AttemptCollectiveContext& context, const GateReport& gates,
      const FailureReason reason, const std::uint32_t ranks) noexcept
  {
    return {context, gates, ranks, true, true,
            reason != FailureReason::none, ReasonBit(reason), reason};
  }
  static CollectiveRestoreReport ReduceRestore(
      const RestoreCollectiveContext& context,
      const RestartCompatibility compatibility, const FailureReason reason,
      const std::uint32_t ranks) noexcept
  {
    const auto value = static_cast<unsigned>(compatibility);
    const std::uint64_t compatibility_bit =
        compatibility == RestartCompatibility::compatible
        ? 0 : (std::uint64_t{1} << value);
    return {context, ranks, true, true, reason != FailureReason::none,
            ReasonBit(reason), compatibility_bit, reason, compatibility};
  }
};

using TinyContract = PersistentRunContract<
    TestSelection<TestRankOneBackendTag>, TinyInterior, TinyTrace,
    TinyStorage, TinyCollective>;

TinyContract::ProductBundle TinyProducts(
    TinyContract& contract, const AttemptIdentity& attempt) noexcept
{
  TinyContract::ProductBundle products{};
  products.residual =
      contract.IssueProduct<DerivedProductKind::residual>(attempt);
  products.matrix =
      contract.IssueProduct<DerivedProductKind::matrix>(attempt);
  products.factor =
      contract.IssueProduct<DerivedProductKind::factor>(attempt);
  products.solution =
      contract.IssueProduct<DerivedProductKind::solution>(attempt);
  return products;
}

PersistentResourceDescriptor TinyResources() noexcept
{
  StateHash mesh{};
  StateHash partition{};
  StateHash topology{};
  mesh.bytes[0] = 1;
  partition.bytes[0] = 2;
  topology.bytes[0] = 3;
  return {StableIdentity::Copy("test.mfem.pin"), mesh, partition, topology,
          1, 1, StableIdentity::Copy("test.layout.fp64"),
          {true, true, true, false, 1, 1, 64, 32}};
}

AttemptReport TinyPassingReport() noexcept
{
  AttemptReport report{};
  report.gates = {true, true, true, true, true, true};
  report.telemetry.transfers = {0, 0, 0, 0, 0, true};
  report.telemetry.true_relative_residual = 0.0;
  report.candidate_cursor = 7;
  return report;
}

bool PrepareTiny(TinyContract& contract, AttemptIdentity& attempt,
                 TinyContract::ProductBundle& products,
                 std::optional<TinyContract::StagingToken>& staging) noexcept
{
  const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.05});
  if (!begin.started) { return false; }
  attempt = begin.attempt;
  const auto candidate = contract.AcquireCandidate(attempt);
  const TinySlot accepted{{{11.0, 12.0}}, {{13.0, 14.0}}};
  if (!candidate || !contract.StoreCandidate(*candidate, accepted))
    return false;
  products = TinyProducts(contract, attempt);
  auto frozen = contract.FreezeCandidate(attempt, 7);
  if (!frozen) { return false; }
  staging.emplace(std::move(*frozen));
  return true;
}

bool DirectSharedContractNegatives() noexcept
{
  const TinySlot committed{{{1.0, 2.0}}, {{3.0, 4.0}}};
  const TinySlot candidate{{{-1.0, -2.0}}, {{-3.0, -4.0}}};
  TinyContract first{committed, candidate, {}, TinyResources()};
  TinyContract second{committed, candidate, {}, TinyResources()};
  TinyContract::ProductBundle first_products{};
  TinyContract::ProductBundle second_products{};
  AttemptIdentity first_attempt{};
  AttemptIdentity second_attempt{};
  std::optional<TinyContract::StagingToken> first_staging{};
  std::optional<TinyContract::StagingToken> second_staging{};
  if (!PrepareTiny(first, first_attempt, first_products, first_staging) ||
      !PrepareTiny(second, second_attempt, second_products, second_staging))
  {
    return false;
  }

  TinyContract::StagingToken moved{std::move(*first_staging)};
  TinySlot copy{};
  const bool moved_from_rejected =
      !first.CopyPrepared(*first_staging, copy);
  auto foreign_prepare = second.PrepareCommit(
      std::move(moved), TinyPassingReport(), std::move(second_products));
  const bool cross_owner_rejected = !foreign_prepare;
  const bool owner_authority_survived = first.CopyPrepared(moved, copy);
  auto first_prepared = first.PrepareCommit(
      std::move(moved), TinyPassingReport(), std::move(first_products));
  TinySlot stale_copy{};
  const bool consumed_staging_rejected =
      !first.CopyPrepared(moved, stale_copy);
  auto second_prepared = second.PrepareCommit(
      std::move(*second_staging), TinyPassingReport(),
      std::move(second_products));
  const AttemptOutcome first_aborted = first_prepared
      ? first.AbortPrepared(std::move(*first_prepared),
                            FailureReason::backend_failure)
      : AttemptOutcome{};
  const AttemptOutcome second_aborted = second_prepared
      ? second.AbortPrepared(std::move(*second_prepared),
                             FailureReason::backend_failure)
      : AttemptOutcome{};
  return moved_from_rejected && cross_owner_rejected &&
      owner_authority_survived && first_prepared && second_prepared &&
      consumed_staging_rejected && first_aborted.retry_available &&
      second_aborted.retry_available && !first_aborted.committed &&
      !second_aborted.committed;
}

bool Global(const bool local) noexcept
{
  int value = local ? 1 : 0;
  int all = 0;
  return MPI_Allreduce(&value, &all, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && all == 1;
}

constexpr bool CleanHypreCudaStateAuthorized(
    const int hypre_error, const cudaError_t cuda_error) noexcept
{
  return hypre_error == 0 && cuda_error == cudaSuccess;
}

bool CleanHypreCudaAuthorizationFixtures() noexcept
{
  return CleanHypreCudaStateAuthorized(0, cudaSuccess) &&
      !CleanHypreCudaStateAuthorized(HYPRE_ERROR_GENERIC, cudaSuccess) &&
      !CleanHypreCudaStateAuthorized(HYPRE_ERROR_MEMORY, cudaSuccess) &&
      !CleanHypreCudaStateAuthorized(HYPRE_ERROR_ARG, cudaSuccess) &&
      !CleanHypreCudaStateAuthorized(HYPRE_ERROR_CONV, cudaSuccess) &&
      !CleanHypreCudaStateAuthorized(
          HYPRE_ERROR_GENERIC | HYPRE_ERROR_MEMORY, cudaSuccess) &&
      !CleanHypreCudaStateAuthorized(0, cudaErrorInvalidValue) &&
      !CleanHypreCudaStateAuthorized(0, cudaErrorMemoryAllocation) &&
      !CleanHypreCudaStateAuthorized(0, cudaErrorUnknown);
}

template <typename Backend>
bool RunRankTwoConfigConsensusBoundary(const int ranks,
                                       const int rank) noexcept
{
  // Both collective gates require an exactly clean process-global backend.
  // A stateful negative terminates this process lane before positive cases.
  if (ranks != 2) { return false; }
  const int incoming_hypre = HYPRE_GetError();
  const cudaError_t incoming_cuda = cudaPeekAtLastError();
  const bool clean_prestate = Global(CleanHypreCudaStateAuthorized(
      incoming_hypre, incoming_cuda));
  if (!clean_prestate) { return false; }

  const bool intended_negative =
      RunConfigConsensusNegative<Backend>(ranks, rank);
  const int outgoing_hypre = HYPRE_GetError();
  const cudaError_t outgoing_cuda = cudaPeekAtLastError();
  return Global(intended_negative && CleanHypreCudaStateAuthorized(
      outgoing_hypre, outgoing_cuda));
}

bool CollectiveCaseBoundaryRegression(const int rank, const int ranks) noexcept
{
  if (ranks == 1) { return true; }
  if (ranks != 2) { return false; }
  const bool rank_zero_only_failure = rank == 0;
  const CheckpointPostconditionConsensus decision =
      AgreeCheckpointPostconditionFailure(rank_zero_only_failure, ranks);
  const bool traversed_cleanup_boundary = decision.cleanup_required &&
      decision.reduction_complete && decision.failing_ranks == 1 &&
      decision.local_failure == rank_zero_only_failure &&
      decision.cleanup_reason == checkpoint_cleanup_reason &&
      decision.cleanup_phase == checkpoint_cleanup_phase;
  const bool both_ranks_traversed_cleanup_boundary =
      Global(traversed_cleanup_boundary);
  const bool all_ranks_reenter_next_boundary = Global(true);
  return both_ranks_traversed_cleanup_boundary &&
      all_ranks_reenter_next_boundary;
}

bool PrestageFailureMaskExactFixtures() noexcept
{
  if (cuda_detail::CudaPrestageFailureMask(
          true, true, true, true, true, true, true, false, true) != 0)
    return false;
  for (unsigned bit = 0; bit < 9; ++bit)
  {
    bool finite = true;
    bool admissible = true;
    bool recovery = true;
    bool true_residual = true;
    bool telemetry = true;
    bool no_fallback = true;
    bool success = true;
    bool injected = false;
    bool collective = true;
    switch (bit)
    {
      case 0: finite = false; break;
      case 1: admissible = false; break;
      case 2: recovery = false; break;
      case 3: true_residual = false; break;
      case 4: telemetry = false; break;
      case 5: no_fallback = false; break;
      case 6: success = false; break;
      case 7: injected = true; break;
      case 8: collective = false; break;
      default: return false;
    }
    if (cuda_detail::CudaPrestageFailureMask(
            finite, admissible, recovery, true_residual, telemetry,
            no_fallback, success, injected, collective) !=
        (std::uint64_t{1} << bit))
      return false;
  }
  return true;
}

bool ReportCaseBoundary(const int rank, const int ranks,
                        const char* const label,
                        const char* const edge) noexcept
{
  const int hypre_error = HYPRE_GetError();
  const cudaError_t cuda_error = cudaPeekAtLastError();
  const CudaAllocationLedgerV1 allocation = AuditedAllocator::Ledger();
  const CudaTransferLedgerV1& transfer = AuditedTransport::Ledger();
  const bool ledger_mirrors_exact = SameLedger(
          allocation, UmpireDeviceAllocatorPolicy::Ledger()) &&
      SameLedger(transfer, CudaTransportPolicy::Ledger()) &&
      SameTotals(AuditedTransport::Accounting(),
                 CudaTransportPolicy::Accounting());
  const bool local_exact = ranks >= 1 &&
      CleanHypreCudaStateAuthorized(hypre_error, cuda_error) &&
      ledger_mirrors_exact;
  const bool collective_exact = Global(local_exact);
  std::fprintf(stderr,
      "CUDA_CASE_BOUNDARY rank=%d ranks=%d case=%s edge=%s "
      "hypre=%d cuda=%d allocation=%zu/%llu transfer=%zu/%llu "
      "local=%s collective=%s\n",
      rank, ranks, label, edge, hypre_error, static_cast<int>(cuda_error),
      allocation.count,
      static_cast<unsigned long long>(allocation.observed_count),
      transfer.count,
      static_cast<unsigned long long>(transfer.observed_count),
      local_exact ? "PASS" : "FAIL",
      collective_exact ? "PASS" : "FAIL");
  return collective_exact;
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const bool token_mode = argc == 2 &&
      std::string_view(argv[1]) == "--token-negatives";
  const bool functional_mode = argc == 2 &&
      std::string_view(argv[1]) == "--functional";
  bool pass = true;
  const auto run_case_impl = [&](const char* const case_label,
                                 auto&& case_body)
  {
    const bool before_collective =
        ReportCaseBoundary(rank, ranks, case_label, "before");
    const bool local_result = before_collective && case_body();
    const bool body_collective = Global(local_result);
    const bool after_collective =
        ReportCaseBoundary(rank, ranks, case_label, "after");
    const bool collective_result = Global(
        before_collective && body_collective && after_collective);
    std::fprintf(stderr,
        "CUDA_CASE_RESULT rank=%d ranks=%d case=%s local=%s "
        "body_collective=%s collective=%s\n",
        rank, ranks, case_label, local_result ? "PASS" : "FAIL",
        body_collective ? "PASS" : "FAIL",
        collective_result ? "PASS" : "FAIL");
    if (!collective_result)
    {
      if (rank == 0)
        std::cerr << "FAILED " << case_label << '\n';
    }
    pass = pass && collective_result;
    return collective_result;
  };
#define run_case(case_label, ...) \
  run_case_impl(case_label, [&]() noexcept { return (__VA_ARGS__); })
  if (functional_mode && ranks == 1)
  {
    run_case("rank1.functional.p2-n6-step20-restart-retry",
             RunFunctionalTrajectory<TestRankOneBackendTag>(
                 1, "rank1.functional.p2-n6-step20-restart-retry"));
  }
  else if (functional_mode && ranks == 2)
  {
    run_case("rank2.functional.p2-n6-step20-restart-retry",
             RunFunctionalTrajectory<TestRankTwoBackendTag>(
                 2, "rank2.functional.p2-n6-step20-restart-retry"));
  }
  else if (ranks == 1)
  {
    run_case("rank1.common.hypre-setup-order",
             HypreIluGenerationSetupRegression());
    run_case("rank1.common.ticket-high-half",
             TicketHighHalfRegression());
    run_case("rank1.common.shared-contract-negatives",
             DirectSharedContractNegatives());
    run_case("rank1.common.collective-case-boundary",
             CollectiveCaseBoundaryRegression(rank, ranks));
    run_case("rank1.common.host-offd-map-copy",
             HypreHostColumnMapCopyFixtures());
    run_case("rank1.common.hypre-clean-authorization",
             CleanHypreCudaAuthorizationFixtures());
    run_case("rank1.config.consensus-negative",
             RunConfigConsensusNegative<TestRankOneBackendTag>(1, rank));
    run_case("rank1.negative.oracle-export",
             RunFrozenAuthorityNegative<TestRankOneBackendTag>(
                 1, rank, FrozenAuthorityNegative::oracle_export,
                 "rank1.negative.oracle-export"));
    run_case("rank1.negative.output-mark",
             RunFrozenAuthorityNegative<TestRankOneBackendTag>(
                 1, rank, FrozenAuthorityNegative::output_mark,
                 "rank1.negative.output-mark"));
    run_case("rank1.negative.finalize-descriptor",
             RunFrozenAuthorityNegative<TestRankOneBackendTag>(
                 1, rank, FrozenAuthorityNegative::finalize_descriptor,
                 "rank1.negative.finalize-descriptor"));
    if (token_mode)
    {
      run_case("rank1.token.omit-solution",
               RunTokenFault<TestRankOneBackendTag>(
                   1, CudaTokenFault::omit_solution,
                   FailureReason::missing_product,
                   "rank1.token.omit-solution"));
      run_case("rank1.token.stale-attempt",
               RunTokenFault<TestRankOneBackendTag>(
                   1, CudaTokenFault::stale_attempt,
                   FailureReason::missing_product,
                   "rank1.token.stale-attempt"));
      run_case("rank1.token.matrix-before-residual",
               RunTokenFault<TestRankOneBackendTag>(
                   1, CudaTokenFault::issue_matrix_before_residual,
                   FailureReason::none,
                   "rank1.token.matrix-before-residual"));
    }
    else
    {
      CudaSerializedCheckpointV1 checkpoint{};
      run_case("rank1.lifecycle",
               RunLifecycle<TestRankOneBackendTag>(
                   1, checkpoint, "rank1.lifecycle"));
      run_case("rank1.restore.corrupt",
               RunCorruptRestore<TestRankOneBackendTag>(
                   1, checkpoint, "rank1.restore.corrupt"));
      run_case("rank1.restore.fresh",
               RunFreshRestore<TestRankOneBackendTag>(
                   1, checkpoint, "rank1.restore.fresh"));
      run_case("rank1.provisional-revert",
               RunProvisionalRevert<TestRankOneBackendTag>(
                   1, "rank1.provisional-revert"));
      run_case("rank1.prepared-abort",
               RunPreparedAbort<TestRankOneBackendTag>(
                   1, "rank1.prepared-abort"));
      run_case("rank1.staging-failure",
               RunStagingFailure<TestRankOneBackendTag>(
                   1, "rank1.staging-failure"));
    }
    run_case("rank1.diagnostic.checkpoint-v1-exact",
             CheckpointDiagnosticExactFixtures());
    run_case("rank1.diagnostic.prestage-mask-exact",
             PrestageFailureMaskExactFixtures());
    run_case("rank1.teardown.state-machine",
             ArenaReleaseStateMachineFixtures());
    run_case("rank1.teardown.idempotent",
             RunArenaTeardown<TestRankOneBackendTag>(
                 1, "rank1.teardown.idempotent"));
  }
  else if (ranks == 2)
  {
    run_case("rank2.common.hypre-setup-order",
             HypreIluGenerationSetupRegression());
    run_case("rank2.common.ticket-high-half",
             TicketHighHalfRegression());
    run_case("rank2.common.shared-contract-negatives",
             DirectSharedContractNegatives());
    run_case("rank2.common.collective-case-boundary",
             CollectiveCaseBoundaryRegression(rank, ranks));
    run_case("rank2.common.host-offd-map-copy",
             HypreHostColumnMapCopyFixtures());
    run_case("rank2.common.hypre-clean-authorization",
             CleanHypreCudaAuthorizationFixtures());
    run_case("rank2.common.mixed-affine-census",
             MixedPreparedProvisionalCensusFixture(rank, ranks));
    CudaSerializedCheckpointV1 qualification_checkpoint{};
    const char* const qualification_context = token_mode
        ? "rank2.token.allocation-plateau" : "rank2.lifecycle";
    const bool allocation_plateau_qualified =
        run_case(qualification_context,
                 RunLifecycle<TestRankTwoBackendTag>(
                     2, qualification_checkpoint, qualification_context, 1));
    const bool config_boundary_clean = allocation_plateau_qualified &&
        run_case("rank2.config.consensus-negative",
                 RunRankTwoConfigConsensusBoundary<TestRankTwoBackendTag>(
                     2, rank));
    if (config_boundary_clean)
    {
      run_case("rank2.negative.oracle-export",
               RunFrozenAuthorityNegative<TestRankTwoBackendTag>(
                   2, rank, FrozenAuthorityNegative::oracle_export,
                   "rank2.negative.oracle-export"));
      run_case("rank2.negative.output-mark",
               RunFrozenAuthorityNegative<TestRankTwoBackendTag>(
                   2, rank, FrozenAuthorityNegative::output_mark,
                   "rank2.negative.output-mark"));
      run_case("rank2.negative.finalize-descriptor",
               RunFrozenAuthorityNegative<TestRankTwoBackendTag>(
                   2, rank, FrozenAuthorityNegative::finalize_descriptor,
                   "rank2.negative.finalize-descriptor"));
      if (token_mode)
      {
        if (allocation_plateau_qualified)
        {
          run_case("rank2.token.omit-solution",
                   RunTokenFault<TestRankTwoBackendTag>(
                       2, CudaTokenFault::omit_solution,
                       FailureReason::missing_product,
                       "rank2.token.omit-solution"));
          run_case("rank2.token.stale-attempt",
                   RunTokenFault<TestRankTwoBackendTag>(
                       2, CudaTokenFault::stale_attempt,
                       FailureReason::missing_product,
                       "rank2.token.stale-attempt"));
          run_case("rank2.token.matrix-before-residual",
                   RunTokenFault<TestRankTwoBackendTag>(
                       2,
                       CudaTokenFault::issue_matrix_before_residual,
                       FailureReason::none,
                       "rank2.token.matrix-before-residual"));
        }
      }
      else
      {
        run_case("rank2.restore.corrupt",
                 RunCorruptRestore<TestRankTwoBackendTag>(
                     2, qualification_checkpoint,
                     "rank2.restore.corrupt"));
        run_case("rank2.restore.fresh",
                 RunFreshRestore<TestRankTwoBackendTag>(
                     2, qualification_checkpoint,
                     "rank2.restore.fresh"));
        run_case("rank2.provisional-revert",
                 RunProvisionalRevert<TestRankTwoBackendTag>(
                     2, "rank2.provisional-revert"));
        run_case("rank2.prepared-abort",
                 RunPreparedAbort<TestRankTwoBackendTag>(
                     2, "rank2.prepared-abort"));
        run_case("rank2.staging-failure",
                 RunStagingFailure<TestRankTwoBackendTag>(
                     2, "rank2.staging-failure"));
      }
      run_case("rank2.diagnostic.checkpoint-v1-exact",
               CheckpointDiagnosticExactFixtures());
      run_case("rank2.diagnostic.prestage-mask-exact",
               PrestageFailureMaskExactFixtures());
      run_case("rank2.teardown.state-machine",
               ArenaReleaseStateMachineFixtures());
      run_case("rank2.teardown.idempotent",
               RunArenaTeardown<TestRankTwoBackendTag>(
                   2, "rank2.teardown.idempotent"));
    }
  }
  else
  {
    run_case("unsupported.common.hypre-setup-order",
             HypreIluGenerationSetupRegression());
    run_case("unsupported.common.ticket-high-half",
             TicketHighHalfRegression());
    run_case("unsupported.common.shared-contract-negatives",
             DirectSharedContractNegatives());
    run_case("unsupported.common.collective-case-boundary",
             CollectiveCaseBoundaryRegression(rank, ranks));
    run_case("unsupported.common.host-offd-map-copy",
             HypreHostColumnMapCopyFixtures());
    run_case("unsupported.common.hypre-clean-authorization",
             CleanHypreCudaAuthorizationFixtures());
    run_case("unsupported.diagnostic.checkpoint-v1-exact",
             CheckpointDiagnosticExactFixtures());
    run_case("unsupported.diagnostic.prestage-mask-exact",
             PrestageFailureMaskExactFixtures());
    run_case("unsupported.teardown.state-machine",
             ArenaReleaseStateMachineFixtures());
    pass = false;
  }
#undef run_case

  if (!Global(pass))
  {
    if (rank == 0)
      std::cerr << "persistent CUDA W8A leaf audit failed\n";
    return 1;
  }
  return 0;
}
