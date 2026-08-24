#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include "physics/bc/detail/euler_boundary_state.hpp"

#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
#include <_hypre_parcsr_ls.h>
#include <_hypre_utilities.h>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>

#include <mma.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

// Source-only candidate for the native SM120 W8A path.  Every kernel records
// __CUDA_ARCH__ and refuses to provide a valid ticket for another image.

namespace dg::hdg::runtime::cuda_w8a
{
namespace
{
std::atomic<std::uint64_t> allocation_events{0};
std::atomic<std::uint64_t> w8_owned_allocation_calls{0};
std::atomic<std::uint64_t> deallocation_events{0};
std::atomic<std::uint64_t> allocator_growth_events{0};
std::atomic<std::size_t> mfem_highwater{0};
std::atomic<std::size_t> hypre_highwater{0};
std::atomic<std::uint64_t> allocation_ordinal{0};
std::atomic<std::uint64_t> allocation_generation{0};
CudaAllocationLedgerV1 allocation_ledger{};
thread_local CudaPolicyContext transport_context{};
thread_local CudaTransferLedgerV1 transport_ledger{};
thread_local CudaTransferTotals transport_attempt_totals{};
thread_local CudaTransferLedgerRangeV1 transport_attempt_range{};
thread_local CudaTransferLedgerRangeV1 transport_staging_range{};
thread_local bool transport_staging_active = false;
thread_local bool true_system_diagnostic_active = false;
thread_local int true_system_diagnostic_rank = -1;

bool Copy(void* destination, const void* source, const std::size_t bytes,
          const cudaMemcpyKind kind) noexcept
{
  return bytes == 0 || (destination != nullptr && source != nullptr &&
      cudaMemcpyAsync(destination, source, bytes, kind, nullptr) ==
          cudaSuccess);
}

template <typename T>
void MixScalar(std::uint64_t& hash, T value) noexcept;

std::uint64_t EndpointToken(const CudaTransferDirection direction,
                            const CudaTransferCategory category,
                            const CudaTransferSite site,
                            const CudaPolicyContext& context,
                            const std::uint64_t ordinal,
                            const std::uintptr_t source,
                            const std::uintptr_t destination,
                            const std::size_t bytes,
                            const std::size_t source_capacity,
                            const std::size_t destination_capacity,
                            const bool success) noexcept
{
  std::uint64_t token = 1469598103934665603ULL;
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

void AccumulateRange(CudaTransferLedgerRangeV1& range,
                     const std::uint64_t ordinal,
                     const std::uint64_t token,
                     const bool success) noexcept
{
  if (range.begin_ordinal == 0) { range.begin_ordinal = ordinal; }
  range.end_ordinal = ordinal;
  range.ordered_digest ^= token;
  range.ordered_digest *= 1099511628211ULL;
  range.complete = range.complete && success;
}

void MarkActiveRangesIncomplete() noexcept
{
  transport_attempt_range.complete = false;
  if (transport_staging_active) { transport_staging_range.complete = false; }
}

void RecordTransfer(const CudaTransferDirection direction,
                    const CudaTransferCategory category,
                    const CudaTransferSite site,
                    const void* source,
                    const void* destination,
                    const std::size_t bytes,
                    const std::size_t source_capacity,
                    const std::size_t destination_capacity,
                    const bool success) noexcept
{
  CudaTransferTotals& totals = transport_ledger.totals;
  if (transport_ledger.observed_count ==
      std::numeric_limits<std::uint64_t>::max())
  {
    transport_ledger.overflow = true;
    transport_ledger.totals.complete = false;
    transport_attempt_totals.complete = false;
    MarkActiveRangesIncomplete();
    return;
  }
  const std::uint64_t ordinal = ++transport_ledger.observed_count;
  const auto account = [&](CudaTransferTotals& value)
  {
    if (value.event_count == std::numeric_limits<std::uint64_t>::max())
    {
      value.complete = false;
      return;
    }
    ++value.event_count;
    value.complete = value.complete && success;
    if (!success) { return; }
    auto add = [&](std::uint64_t& endpoint)
    {
      if (bytes > std::numeric_limits<std::uint64_t>::max() - endpoint)
      {
        value.complete = false;
        return;
      }
      endpoint += static_cast<std::uint64_t>(bytes);
    };
    if (direction == CudaTransferDirection::host_to_device)
      add(value.host_to_device_bytes);
    else if (direction == CudaTransferDirection::device_to_host)
      add(value.device_to_host_bytes);
    else if (direction == CudaTransferDirection::device_to_device)
      add(value.device_to_device_bytes);
    else
      add(value.mpi_staging_bytes);
  };
  account(totals);
  account(transport_attempt_totals);
  const auto source_token = reinterpret_cast<std::uintptr_t>(source);
  const auto destination_token = reinterpret_cast<std::uintptr_t>(destination);
  const std::size_t exact_source_capacity =
      source_capacity == 0 ? bytes : source_capacity;
  const std::size_t exact_destination_capacity =
      destination_capacity == 0 ? bytes : destination_capacity;
  const std::uint64_t endpoint_token = EndpointToken(
      direction, category, site, transport_context, ordinal, source_token,
      destination_token, bytes, exact_source_capacity,
      exact_destination_capacity, success);
  transport_ledger.ordered_digest ^= endpoint_token;
  transport_ledger.ordered_digest *= 1099511628211ULL;
  AccumulateRange(transport_attempt_range, ordinal, endpoint_token, success);
  if (transport_staging_active)
    AccumulateRange(transport_staging_range, ordinal, endpoint_token, success);
}

void RecordAllocation(const CudaAllocationSite site,
                      const std::size_t bytes,
                      const std::size_t alignment,
                      void* pointer,
                      const bool allocation,
                      const bool success,
                      const std::uint64_t retained_generation = 0) noexcept
{
  if (allocation_ledger.observed_count ==
      std::numeric_limits<std::uint64_t>::max())
  {
    allocation_ledger.overflow = true;
    return;
  }
  ++allocation_ledger.observed_count;
  const std::uint64_t ordinal = ++allocation_ordinal;
  const std::uint64_t generation = allocation && success
      ? ++allocation_generation : retained_generation;
  std::uint64_t event_digest = 1469598103934665603ULL;
  MixScalar(event_digest, ordinal);
  MixScalar(event_digest, site);
  MixScalar(event_digest, transport_context.attempt_sequence);
  MixScalar(event_digest, transport_context.target_step);
  MixScalar(event_digest, transport_context.generation);
  MixScalar(event_digest, transport_context.rank);
  MixScalar(event_digest, transport_context.phase);
  MixScalar(event_digest, bytes);
  MixScalar(event_digest, alignment);
  MixScalar(event_digest, reinterpret_cast<std::uintptr_t>(pointer));
  MixScalar(event_digest, generation);
  MixScalar(event_digest, allocation);
  MixScalar(event_digest, success);
  allocation_ledger.ordered_digest ^= event_digest;
  allocation_ledger.ordered_digest *= 1099511628211ULL;
}

void Mix(std::uint64_t& hash, const void* data, const std::size_t size) noexcept
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i)
  {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
}

template <typename T>
void MixScalar(std::uint64_t& hash, const T value) noexcept
{ Mix(hash, &value, sizeof(value)); }

void MixId(std::uint64_t& hash, const StableIdentity& id) noexcept
{ Mix(hash, id.data(), id.size()); MixScalar(hash, id.size()); }
void MixStateHash(std::uint64_t& hash, const StateHash& value) noexcept
{ Mix(hash, value.bytes.data(), value.bytes.size()); }
void MixCoordinates(std::uint64_t& hash, const StateCoordinates& value) noexcept
{ MixScalar(hash,value.time);MixScalar(hash,value.step);MixScalar(hash,value.generation);MixScalar(hash,value.cursor);MixScalar(hash,value.dt); }
void MixAttempt(std::uint64_t& hash, const AttemptIdentity& value) noexcept
{ MixScalar(hash,value.sequence);MixScalar(hash,value.committed_step);MixScalar(hash,value.committed_generation);MixScalar(hash,value.retry_index);MixScalar(hash,value.dt); }
void MixTuple(std::uint64_t& hash,const RuntimeTupleDescriptor& value) noexcept
{MixId(hash,value.model);MixId(hash,value.formulation);MixId(hash,value.trace);MixId(hash,value.stabilization);MixId(hash,value.time_treatment);MixId(hash,value.backend);MixId(hash,value.precision);}

std::uint64_t ContextHash(const AttemptCollectiveContext& value) noexcept
{std::uint64_t hash=1469598103934665603ULL;MixAttempt(hash,value.attempt);MixCoordinates(hash,value.committed);MixCoordinates(hash,value.target);MixScalar(hash,value.candidate_cursor);return hash;}

std::uint64_t ContextHash(const RestoreCollectiveContext& value) noexcept
{std::uint64_t hash=1469598103934665603ULL;const auto& c=value.checkpoint;MixScalar(hash,c.schema_version);MixScalar(hash,c.successful_commit);MixScalar(hash,c.restart_mode);MixTuple(hash,c.tuple);MixId(hash,c.mfem_pin);MixStateHash(hash,c.mesh_hash);MixStateHash(hash,c.partition_hash);MixStateHash(hash,c.topology_hash);MixScalar(hash,c.rank_count);MixScalar(hash,c.polynomial_order);MixId(hash,c.layout);MixId(hash,c.precision);MixScalar(hash,c.cursor);MixScalar(hash,c.dt);MixCoordinates(hash,c.coordinates);MixScalar(hash,c.coordinate_fingerprint);MixStateHash(hash,c.interior_hash);MixStateHash(hash,c.trace_hash);MixStateHash(hash,value.loaded_interior_hash);MixStateHash(hash,value.loaded_trace_hash);MixStateHash(hash,value.prepared_interior_hash);MixStateHash(hash,value.prepared_trace_hash);MixScalar(hash,value.preparation_attempted);MixScalar(hash,value.candidate_prepared);return hash;}

bool CollectiveHashConsensus(const std::uint64_t local,
                             const std::uint32_t expected,
                             std::uint32_t& participating,
                             const MPI_Comm communicator) noexcept
{
  int size = 0;
  std::uint64_t minimum = local;
  std::uint64_t maximum = local;
  if (communicator == MPI_COMM_NULL) return false;
  const bool size_ready =
      MPI_Comm_size(communicator, &size) == MPI_SUCCESS;
  const bool minimum_ready = MPI_Allreduce(
      &local, &minimum, 1, MPI_UINT64_T, MPI_MIN,
      communicator) == MPI_SUCCESS;
  const bool maximum_ready = MPI_Allreduce(
      &local, &maximum, 1, MPI_UINT64_T, MPI_MAX,
      communicator) == MPI_SUCCESS;
  if (!size_ready || !minimum_ready || !maximum_ready || size <= 0)
    return false;
  participating = static_cast<std::uint32_t>(size);
  return participating == expected && minimum == maximum;
}

std::uint64_t FailureBit(const FailureReason reason) noexcept
{
  const auto index = static_cast<unsigned>(reason);
  return reason == FailureReason::none || index >= 64 ? 0ULL : 1ULL << index;
}

FailureReason FirstFailure(const std::uint64_t mask) noexcept
{
  for (unsigned index = 1;
       index < static_cast<unsigned>(FailureReason::count) && index < 64;
       ++index)
    if ((mask & (1ULL << index)) != 0)
      return static_cast<FailureReason>(index);
  return FailureReason::none;
}

bool ReduceMask(const std::uint64_t local, std::uint64_t& global,
                const MPI_Comm communicator) noexcept
{
  global = local;
  return MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_BOR,
                       communicator) == MPI_SUCCESS;
}

bool AllRanksReady(const bool local, const MPI_Comm communicator,
                   const int ranks) noexcept
{
  if (communicator == MPI_COMM_NULL || ranks <= 0) return false;
  if (ranks == 1) return local;
  const int local_value = local ? 1 : 0;
  int global_value = 0;
  return MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && global_value == 1;
}

void ObserveHighwater(std::atomic<std::size_t>& endpoint,
                      const std::size_t current) noexcept
{
  std::size_t previous = endpoint.load();
  while (current > previous &&
         !endpoint.compare_exchange_weak(previous, current)) {}
  if (previous != 0 && current > previous) { ++allocator_growth_events; }
}
} // namespace

bool UmpireDeviceAllocatorPolicy::PrepareDeviceArenaAllocator() noexcept
{
  ++w8_owned_allocation_calls;
  constexpr std::size_t probe_bytes = sizeof(std::max_align_t);
  constexpr std::size_t probe_alignment = alignof(std::max_align_t);
  mfem::Memory<std::max_align_t> probe{};
  void* device_pointer = nullptr;
  bool allocation_succeeded = false;
  bool deallocation_succeeded = false;
  try
  {
    // Exercise MFEM's public lazy device-memory path.  This deliberately does
    // not create an Umpire QuickPool or touch HYPRE's independently lazy pool.
    probe.New(1, mfem::MemoryType::DEVICE_UMPIRE);
    device_pointer = probe.Write(mfem::MemoryClass::DEVICE, 1);
    allocation_succeeded = device_pointer != nullptr &&
        reinterpret_cast<std::uintptr_t>(device_pointer) % probe_alignment ==
            0 &&
        probe.DeviceIsValid();
  }
  catch (...) {}

  if (allocation_succeeded) { ++allocation_events; }
  RecordAllocation(CudaAllocationSite::mfem_device_pool_probe, probe_bytes,
                   probe_alignment, device_pointer, true,
                   allocation_succeeded);
  const std::uint64_t allocation_generation_value = allocation_succeeded
      ? allocation_generation.load() : 0;

  try
  {
    probe.Delete();
    deallocation_succeeded = true;
  }
  catch (...) {}
  if (device_pointer != nullptr)
  {
    if (deallocation_succeeded) { ++deallocation_events; }
    RecordAllocation(CudaAllocationSite::mfem_device_pool_probe, probe_bytes,
                     probe_alignment, device_pointer, false,
                     deallocation_succeeded,
                     allocation_generation_value);
  }
  if (!allocation_succeeded || !deallocation_succeeded) { return false; }

#if defined(MFEM_USE_UMPIRE)
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    const char* name = mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    return name != nullptr && name[0] != '\0' && manager.isAllocator(name);
  }
  catch (...) { return false; }
#else
  return false;
#endif
}

bool UmpireDeviceAllocatorPolicy::Allocate(
    const std::size_t bytes, const std::size_t alignment,
    const CudaAllocationSite site, void*& pointer) noexcept
{
  ++w8_owned_allocation_calls;
  pointer = nullptr;
  if (bytes == 0 || alignment == 0 ||
      (alignment & (alignment - 1)) != 0 ||
      alignment > alignof(std::max_align_t))
  {
    RecordAllocation(site, bytes, alignment, nullptr, true, false);
    return false;
  }
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    const char* name = mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    if (name == nullptr || name[0] == '\0' || !manager.isAllocator(name))
    {
      RecordAllocation(site, bytes, alignment, nullptr, true, false);
      return false;
    }
    pointer = manager.getAllocator(name).allocate(bytes);
    if (pointer != nullptr &&
        reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0)
    {
      manager.getAllocator(name).deallocate(pointer);
      pointer = nullptr;
    }
    if (pointer != nullptr) { ++allocation_events; }
    RecordAllocation(site, bytes, alignment, pointer, true,
                     pointer != nullptr);
    return pointer != nullptr;
  }
  catch (...)
  {
    RecordAllocation(site, bytes, alignment, nullptr, true, false);
    return false;
  }
#else
  (void)bytes;
  RecordAllocation(site, bytes, alignment, nullptr, true, false);
  return false;
#endif
}

bool UmpireDeviceAllocatorPolicy::Deallocate(
    void* pointer, const CudaAllocationSite site) noexcept
{
  if (pointer == nullptr) { return true; }
  bool success = false;
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    const char* name = mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    if (name != nullptr && manager.isAllocator(name))
    {
      manager.getAllocator(name).deallocate(pointer);
      ++deallocation_events;
      success = true;
    }
  }
  catch (...) {}
#endif
  RecordAllocation(site, 0, 0, pointer, false, success);
  return success;
}

void UmpireDeviceAllocatorPolicy::ResetLedger() noexcept
{
  allocation_ledger = {};
  allocation_ordinal = 0;
}

std::uint64_t UmpireDeviceAllocatorPolicy::AllocationEvents() noexcept
{
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    const char* mfem_name = mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    const char* hypre_name = hypre_HandleUmpireDevicePoolName(hypre_handle());
    if (mfem_name != nullptr && manager.isAllocator(mfem_name))
      ObserveHighwater(mfem_highwater,
          manager.getAllocator(mfem_name).getHighWatermark());
    if (hypre_name != nullptr && manager.isAllocator(hypre_name))
      ObserveHighwater(hypre_highwater,
          manager.getAllocator(hypre_name).getHighWatermark());
  }
  catch (...) {}
#endif
  return allocation_events.load() + allocator_growth_events.load();
}
std::uint64_t UmpireDeviceAllocatorPolicy::W8OwnedAllocationCalls() noexcept
{ return w8_owned_allocation_calls.load(); }
std::uint64_t UmpireDeviceAllocatorPolicy::DeallocationEvents() noexcept
{ return deallocation_events.load(); }

bool UmpireDeviceAllocatorPolicy::DevicePoolPolicyValid() noexcept
{
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE) && \
    defined(HYPRE_USING_CUDA)
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    const char* mfem_name = mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    const char* hypre_name = hypre_HandleUmpireDevicePoolName(hypre_handle());
    return mfem_name != nullptr && hypre_name != nullptr &&
        manager.isAllocator(mfem_name) && manager.isAllocator(hypre_name);
  }
  catch (...) { return false; }
#else
  return false;
#endif
}

CudaAllocationLedgerV1 UmpireDeviceAllocatorPolicy::Ledger() noexcept
{
  return allocation_ledger;
}

void CudaTransportPolicy::ResetLedger() noexcept
{
  transport_context = {};
  // Compact accounting intentionally retains no event history.
  transport_ledger.schema = 1;
  transport_ledger.totals = {};
  transport_ledger.observed_count = 0;
  transport_ledger.ordered_digest = 1469598103934665603ULL;
  transport_ledger.overflow = false;
  transport_attempt_totals = {};
  transport_attempt_range = {};
  transport_staging_range = {};
  transport_staging_active = false;
}

void CudaTransportPolicy::BeginAttempt(
    const CudaPolicyContext& context) noexcept
{
  transport_context = context;
  transport_attempt_totals = {};
  transport_attempt_range = {};
  transport_attempt_range.begin_ordinal = transport_ledger.observed_count + 1;
  transport_attempt_range.ordered_digest = 1469598103934665603ULL;
  transport_attempt_range.complete = true;
}

void CudaTransportPolicy::SetPhase(const CudaFailurePhase phase,
                                   const std::uint64_t generation) noexcept
{
  transport_context.phase = phase;
  transport_context.generation = generation;
}

bool CudaTransportPolicy::HostToDevice(
    void* destination, const void* source, const std::size_t bytes,
    const CudaTransferCategory category, const CudaTransferSite site,
    const std::size_t source_capacity,
    const std::size_t destination_capacity) noexcept
{
  [[maybe_unused]] ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_device_transfer);
  const bool trace_touch = true_system_diagnostic_active &&
      category == CudaTransferCategory::initial_state &&
      site == CudaTransferSite::initial_trace_upload;
  const bool sparse_staging = true_system_diagnostic_active &&
      category == CudaTransferCategory::topology &&
      site == CudaTransferSite::map_upload;
  if (trace_touch)
    detail::CudaTrueSystemDiagnosticMarker(
        "trace-device-touch", "enter", false);
  if (sparse_staging)
    detail::CudaTrueSystemDiagnosticMarker(
        "sparse-staging", "enter", false);
  const bool success = Copy(destination, source, bytes, cudaMemcpyHostToDevice);
  RecordTransfer(CudaTransferDirection::host_to_device, category, site,
                 source, destination, bytes, source_capacity,
                 destination_capacity, success);
  if (trace_touch)
    detail::CudaTrueSystemDiagnosticMarker(
        "trace-device-touch", "exit", success);
  if (sparse_staging)
    detail::CudaTrueSystemDiagnosticMarker(
        "sparse-staging", "exit", success);
  return success;
}

bool CudaTransportPolicy::DeviceToHost(
    void* destination, const void* source, const std::size_t bytes,
    const CudaTransferCategory category, const CudaTransferSite site,
    const std::size_t source_capacity,
    const std::size_t destination_capacity) noexcept
{
  [[maybe_unused]] ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_device_transfer);
  const bool true_structure_copy = true_system_diagnostic_active &&
      category == CudaTransferCategory::topology &&
      site == CudaTransferSite::map_upload;
  if (true_structure_copy)
    detail::CudaTrueSystemDiagnosticMarker(
        "true-structure-copy", "enter", false);
  const bool success = Copy(destination, source, bytes, cudaMemcpyDeviceToHost);
  RecordTransfer(CudaTransferDirection::device_to_host, category, site,
                 source, destination, bytes, source_capacity,
                 destination_capacity, success);
  if (true_structure_copy)
    detail::CudaTrueSystemDiagnosticMarker(
        "true-structure-copy", "exit", success);
  return success;
}

bool CudaTransportPolicy::DeviceToDevice(
    void* destination, const void* source, const std::size_t bytes,
    const CudaTransferCategory category, const CudaTransferSite site,
    const std::size_t source_capacity,
    const std::size_t destination_capacity) noexcept
{
  [[maybe_unused]] ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_device_transfer);
  const bool success = Copy(destination, source, bytes, cudaMemcpyDeviceToDevice);
  RecordTransfer(CudaTransferDirection::device_to_device, category, site,
                 source, destination, bytes, source_capacity,
                 destination_capacity, success);
  return success;
}

bool CudaTransportPolicy::RecordMpiStaging(
    const std::size_t bytes, const CudaTransferCategory category,
    const CudaTransferSite site, const void* source,
    const void* destination, const std::size_t source_capacity,
    const std::size_t destination_capacity, const bool success) noexcept
{
  [[maybe_unused]] ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_device_transfer);
  const bool sparse_staging = true_system_diagnostic_active &&
      category == CudaTransferCategory::topology &&
      site == CudaTransferSite::map_upload;
  if (sparse_staging)
    detail::CudaTrueSystemDiagnosticMarker(
        "sparse-staging-mpi", "enter", false);
  RecordTransfer(CudaTransferDirection::mpi_host_staging, category, site,
                 source, destination, bytes, source_capacity,
                 destination_capacity, success);
  if (sparse_staging)
    detail::CudaTrueSystemDiagnosticMarker(
        "sparse-staging-mpi", "exit", success);
  return success;
}

bool CudaTransportPolicy::Synchronize(const CudaTransferSite site) noexcept
{
  [[maybe_unused]] ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_synchronization);
  const bool true_system_sync = true_system_diagnostic_active &&
      (site == CudaTransferSite::map_upload ||
       site == CudaTransferSite::initial_trace_upload);
  const char* const sync_stage = site == CudaTransferSite::map_upload
      ? "true-structure-sync" : "sparse-staging-sync";
  if (true_system_sync)
    detail::CudaTrueSystemDiagnosticMarker(
        sync_stage, "enter", false);
  const bool success = cudaDeviceSynchronize() == cudaSuccess;
  if (transport_ledger.totals.synchronization_events !=
      std::numeric_limits<std::uint64_t>::max())
    ++transport_ledger.totals.synchronization_events;
  else
    transport_ledger.totals.complete = false;
  if (transport_attempt_totals.synchronization_events !=
      std::numeric_limits<std::uint64_t>::max())
    ++transport_attempt_totals.synchronization_events;
  else
    transport_attempt_totals.complete = false;
  transport_ledger.totals.complete =
      transport_ledger.totals.complete && success;
  transport_attempt_totals.complete =
      transport_attempt_totals.complete && success;
  if (true_system_sync)
    detail::CudaTrueSystemDiagnosticMarker(
        sync_stage, "exit", success);
  return success;
}

CudaTransferTotals CudaTransportPolicy::Accounting() noexcept
{ return transport_attempt_totals; }

CudaTransferLedgerV1 CudaTransportPolicy::Ledger() noexcept
{ return transport_ledger; }

CudaTransferLedgerRangeV1 CudaTransportPolicy::AttemptRange() noexcept
{
  return transport_attempt_range;
}

void CudaTransportPolicy::BeginStagingRange() noexcept
{
  transport_staging_range = {};
  transport_staging_range.begin_ordinal = transport_ledger.observed_count + 1;
  transport_staging_range.ordered_digest = 1469598103934665603ULL;
  transport_staging_range.complete = true;
  transport_staging_active = true;
}

CudaTransferLedgerRangeV1 CudaTransportPolicy::EndStagingRange() noexcept
{
  transport_staging_active = false;
  return transport_staging_range;
}

CollectiveAdmissionReport CudaMpiCollectivePolicy::Admit(
    const AttemptCollectiveContext& context, const bool prepared,
    FailureReason reason, const std::uint32_t expected) const noexcept
{
  CollectiveAdmissionReport report{};
  report.context = context;
  if (!prepared && reason == FailureReason::none)
    reason = FailureReason::backend_failure;
  report.identity_consistent = CollectiveHashConsensus(
      ContextHash(context), expected, report.participating_ranks,
      communicator_);
  int local_prepared = prepared ? 1 : 0;
  int all_prepared = local_prepared;
  const bool reduced_prepared = MPI_Allreduce(
      &local_prepared, &all_prepared, 1, MPI_INT, MPI_MIN,
      communicator_) == MPI_SUCCESS;
  report.all_prepared = reduced_prepared && all_prepared == 1;
  const std::uint64_t local_mask = FailureBit(reason);
  report.failure_consistent = ReduceMask(
      local_mask, report.failure_mask, communicator_);
  report.any_failure = report.failure_mask != 0 || !report.all_prepared;
  report.resolved_reason = !report.identity_consistent ||
          !report.failure_consistent
      ? FailureReason::collective_protocol
      : FirstFailure(report.failure_mask);
  if (report.any_failure && report.resolved_reason == FailureReason::none)
    report.resolved_reason = FailureReason::backend_failure;
  return report;
}

CollectiveGateReport CudaMpiCollectivePolicy::Reduce(
    const AttemptCollectiveContext& context, const GateReport& gates,
    FailureReason reason, const std::uint32_t expected) const noexcept
{
  CollectiveGateReport report{};
  report.context = context;
  report.identity_consistent = CollectiveHashConsensus(
      ContextHash(context), expected, report.participating_ranks,
      communicator_);
  const int local[6] = {gates.finite, gates.admissible, gates.recovery,
                        gates.true_residual, gates.telemetry, gates.success};
  int global[6] = {};
  const bool gate_ok = MPI_Allreduce(local, global, 6, MPI_INT, MPI_MIN,
                                     communicator_) == MPI_SUCCESS;
  report.gates = {global[0] == 1, global[1] == 1, global[2] == 1,
                  global[3] == 1, global[4] == 1, global[5] == 1};
  if ((!gate_ok || !gates.All()) && reason == FailureReason::none)
    reason = FailureReason::success_gate;
  report.failure_consistent = ReduceMask(
      FailureBit(reason), report.failure_mask, communicator_);
  report.any_failure = !gate_ok || !report.gates.All() ||
      report.failure_mask != 0;
  report.resolved_reason = !report.identity_consistent ||
          !report.failure_consistent
      ? FailureReason::collective_protocol
      : FirstFailure(report.failure_mask);
  if (report.any_failure && report.resolved_reason == FailureReason::none)
    report.resolved_reason = FailureReason::success_gate;
  return report;
}

CollectiveRestoreReport CudaMpiCollectivePolicy::ReduceRestore(
    const RestoreCollectiveContext& context,
    const RestartCompatibility compatibility, FailureReason reason,
    const std::uint32_t expected) const noexcept
{
  CollectiveRestoreReport report{};
  report.context = context;
  report.identity_consistent = CollectiveHashConsensus(
      ContextHash(context), expected, report.participating_ranks,
      communicator_);
  const auto index = static_cast<unsigned>(compatibility);
  const int local_invalid =
      index >= static_cast<unsigned>(RestartCompatibility::count) ? 1 : 0;
  int any_invalid = local_invalid;
  const bool invalid_reduced = MPI_Allreduce(
      &local_invalid, &any_invalid, 1, MPI_INT, MPI_MAX,
      communicator_) == MPI_SUCCESS;
  const std::uint64_t local_compatibility =
      compatibility == RestartCompatibility::compatible ? 0ULL :
      (index < static_cast<unsigned>(RestartCompatibility::count) &&
       index < 64 ? 1ULL << index : 0ULL);
  const bool compatibility_reduced =
      ReduceMask(local_compatibility, report.compatibility_mask,
                 communicator_);
  const bool failure_reduced = ReduceMask(
      FailureBit(reason), report.failure_mask, communicator_);
  report.failure_consistent = compatibility_reduced && failure_reduced &&
      invalid_reduced;
  report.any_failure = any_invalid != 0 || !report.identity_consistent ||
      !report.failure_consistent || report.compatibility_mask != 0 ||
      report.failure_mask != 0;
  report.resolved_reason = !report.identity_consistent ||
          !report.failure_consistent
      ? FailureReason::collective_protocol
      : FirstFailure(report.failure_mask);
  report.resolved_compatibility = RestartCompatibility::compatible;
  if (any_invalid != 0)
  {
    report.resolved_compatibility = RestartCompatibility::collective_protocol;
    report.resolved_reason = FailureReason::collective_protocol;
  }
  else if (report.compatibility_mask != 0 &&
      (report.compatibility_mask & (report.compatibility_mask - 1)) == 0)
  {
    for (unsigned bit = 1;
         bit < static_cast<unsigned>(RestartCompatibility::count) && bit < 64;
         ++bit)
      if ((report.compatibility_mask & (1ULL << bit)) != 0)
        report.resolved_compatibility =
            static_cast<RestartCompatibility>(bit);
  }
  else if (report.compatibility_mask != 0)
  {
    report.resolved_compatibility = RestartCompatibility::collective_protocol;
    report.resolved_reason = FailureReason::collective_protocol;
  }
  if (report.any_failure &&
      report.resolved_reason == FailureReason::none)
    report.resolved_reason = FailureReason::restart_incompatible;
  return report;
}

namespace detail
{
bool CudaCopyHypreBigIntHost(
    HYPRE_BigInt* const destination,
    const std::size_t destination_capacity,
    const HYPRE_BigInt* const source, const int count) noexcept
{
  if (count < 0) { return false; }
  const std::size_t exact_count = static_cast<std::size_t>(count);
  if (exact_count > destination_capacity) { return false; }
  if (exact_count == 0) { return true; }
  if (destination == nullptr || source == nullptr) { return false; }
  std::copy_n(source, exact_count, destination);
  return true;
}

void CudaTrueSystemDiagnosticMarker(
    const char* const stage, const char* const edge, const bool result,
    const std::uint64_t generation) noexcept
{
  (void)stage;
  (void)edge;
  (void)result;
  (void)generation;
}

void CudaBeginTrueSystemDiagnostic(const int rank) noexcept
{
  true_system_diagnostic_rank = rank;
  true_system_diagnostic_active = true;
  CudaTrueSystemDiagnosticMarker("parallel-assemble", "enter", false);
}

void CudaEndTrueSystemDiagnostic(const bool result) noexcept
{
  CudaTrueSystemDiagnosticMarker("build-persistent-true-system", "exit",
                                 result);
  true_system_diagnostic_active = false;
  true_system_diagnostic_rank = -1;
}

bool CudaGetTrueMatrixStructure(
    mfem::HypreParMatrix& matrix, const std::int64_t*, const int owned_count,
    CudaTrueMatrixStructure& structure, CudaCopyOutFunction copy_out,
    CudaSynchronizeFunction synchronize) noexcept
{
  CudaTrueSystemDiagnosticMarker("parallel-assemble", "exit", true);
  CudaTrueSystemDiagnosticScope diagnostic("true-structure");
  try
  {
    matrix.HypreReadWrite();
    hypre_ParCSRMatrix* raw = matrix;
    auto* diag = hypre_ParCSRMatrixDiag(raw);
    auto* offd = hypre_ParCSRMatrixOffd(raw);
    const HYPRE_Int diag_nonzeros = hypre_CSRMatrixNumNonzeros(diag);
    const HYPRE_Int offd_nonzeros = hypre_CSRMatrixNumNonzeros(offd);
    if (owned_count < 0 || diag_nonzeros < 0 || offd_nonzeros < 0)
      return false;
    structure = {};
    structure.rows = owned_count;
    structure.diag_nonzeros = static_cast<int>(diag_nonzeros);
    structure.offd_nonzeros = static_cast<int>(offd_nonzeros);
    structure.diag_offsets.resize(static_cast<std::size_t>(owned_count) + 1);
    structure.diag_columns.resize(static_cast<std::size_t>(diag_nonzeros));
    structure.offd_offsets.resize(static_cast<std::size_t>(owned_count) + 1);
    structure.offd_columns.resize(static_cast<std::size_t>(offd_nonzeros));
    const HYPRE_Int* di = hypre_CSRMatrixI(diag);
    const HYPRE_Int* dj = hypre_CSRMatrixJ(diag);
    const HYPRE_Int* oi = hypre_CSRMatrixI(offd);
    const HYPRE_Int* oj = hypre_CSRMatrixJ(offd);
    const HYPRE_BigInt* cmap = hypre_ParCSRMatrixColMapOffd(raw);
    if (copy_out == nullptr || synchronize == nullptr) { return false; }
    const std::size_t offset_bytes =
        static_cast<std::size_t>(owned_count + 1) * sizeof(HYPRE_Int);
    const std::size_t diag_column_bytes =
        static_cast<std::size_t>(structure.diag_nonzeros) * sizeof(HYPRE_Int);
    const bool diag_offsets_copied = copy_out(
        structure.diag_offsets.data(), di, offset_bytes,
        CudaTransferCategory::topology, CudaTransferSite::map_upload,
        offset_bytes, structure.diag_offsets.size() * sizeof(HYPRE_Int));
    const bool diag_columns_copied = copy_out(
        structure.diag_columns.data(), dj, diag_column_bytes,
        CudaTransferCategory::topology, CudaTransferSite::map_upload,
        diag_column_bytes, structure.diag_columns.size() * sizeof(HYPRE_Int));
    const bool offd_offsets_copied = copy_out(
        structure.offd_offsets.data(), oi, offset_bytes,
        CudaTransferCategory::topology, CudaTransferSite::map_upload,
        offset_bytes, structure.offd_offsets.size() * sizeof(HYPRE_Int));
    const bool first_synchronized = synchronize(CudaTransferSite::map_upload);
    if (!diag_offsets_copied || !diag_columns_copied ||
        !offd_offsets_copied || !first_synchronized)
      return false;
    const HYPRE_Int raw_offd_column_count = hypre_CSRMatrixNumCols(offd);
    if (raw_offd_column_count < 0)
      return false;
    const int offd_column_count = static_cast<int>(raw_offd_column_count);
    if ((structure.offd_nonzeros > 0 && offd_column_count == 0) ||
        (offd_column_count > 0 && cmap == nullptr))
      return false;
    std::vector<HYPRE_Int> offd_columns(
        static_cast<std::size_t>(structure.offd_nonzeros));
    std::vector<HYPRE_BigInt> offd_global(
        static_cast<std::size_t>(offd_column_count));
    const std::size_t offd_column_bytes =
        static_cast<std::size_t>(structure.offd_nonzeros) * sizeof(HYPRE_Int);
    const bool offd_columns_copied = copy_out(
        offd_columns.data(), oj, offd_column_bytes,
        CudaTransferCategory::topology, CudaTransferSite::map_upload,
        offd_column_bytes, offd_columns.size() * sizeof(HYPRE_Int));
    // HYPRE 3.1 keeps col_map_offd in host storage even when the CSR J arrays
    // are device resident.  This checked host copy intentionally emits no
    // audited CUDA transfer event.
    const bool offd_global_copied = CudaCopyHypreBigIntHost(
        offd_global.data(), offd_global.size(), cmap, offd_column_count);
    const bool second_synchronized = synchronize(CudaTransferSite::map_upload);
    if (!offd_columns_copied || !offd_global_copied ||
        !second_synchronized)
      return false;
    for (int entry = 0; entry < structure.offd_nonzeros; ++entry)
    {
      const HYPRE_Int column = offd_columns[entry];
      if (column < 0 || column >= offd_column_count) { return false; }
      structure.offd_columns[entry] = offd_global[column];
    }
    diagnostic.result = true;
    return true;
  }
  catch (...) { return false; }
}

bool CudaGetTrueMatrixDeviceView(mfem::HypreParMatrix& matrix,
                                 CudaTrueMatrixDeviceView& view) noexcept
{
  CudaTrueSystemDiagnosticScope diagnostic("true-device-view");
  hypre_ParCSRMatrix* raw = matrix;
  auto* diag = hypre_ParCSRMatrixDiag(raw);
  auto* offd = hypre_ParCSRMatrixOffd(raw);
  view.diag_values = hypre_CSRMatrixData(diag);
  view.offd_values = hypre_CSRMatrixData(offd);
  view.diag_nonzeros = hypre_CSRMatrixNumNonzeros(diag);
  view.offd_nonzeros = hypre_CSRMatrixNumNonzeros(offd);
  diagnostic.result = view.diag_values != nullptr &&
      (view.offd_nonzeros == 0 || view.offd_values != nullptr);
  return diagnostic.result;
}

bool CudaConfigurePersistentHypre(mfem::HypreParMatrix& matrix,
                                  mfem::HypreILU& ilu,
                                  mfem::HypreGMRES& gmres,
                                  const int ranks,
                                  const CudaTracePolicyDescriptor& policy,
                                  const double relative_tolerance,
                                  const int maximum_iterations,
                                  const int krylov_dimension,
                                  const std::uint64_t matrix_generation,
                                  CudaHypreSetupState& evidence)
    noexcept
{
  (void)ranks;
  CudaTrueSystemDiagnosticScope diagnostic("object-configure",
                                           matrix_generation);
  const CudaTracePreconditionerKind kind = policy.trace_preconditioner;
  const int ilu_type = policy.ilu_type;
  const int level_of_fill = policy.ilu_level_of_fill;
  const int local_reordering = policy.ilu_local_reordering;
  const int application_iterations = policy.ilu_application_iterations;
  const CudaIluTriangularSolveKind triangular_solve_kind =
      policy.ilu_triangular_solve_kind;
  const int triangular_solve = triangular_solve_kind ==
          CudaIluTriangularSolveKind::direct
      ? 1 : triangular_solve_kind ==
              CudaIluTriangularSolveKind::iterative_jacobi ? 0 : -1;
  const int lower_jacobi_iterations =
      policy.ilu_lower_jacobi_iterations;
  const int upper_jacobi_iterations =
      policy.ilu_upper_jacobi_iterations;
  evidence = {};
  evidence.matrix_generation = matrix_generation;
  evidence.trace_preconditioner = kind;
  evidence.ilu_type = ilu_type;
  evidence.ilu_level_of_fill = level_of_fill;
  evidence.ilu_local_reordering = local_reordering;
  evidence.ilu_application_iterations = application_iterations;
  evidence.ilu_triangular_solve_kind = triangular_solve_kind;
  evidence.ilu_lower_jacobi_iterations = lower_jacobi_iterations;
  evidence.ilu_upper_jacobi_iterations = upper_jacobi_iterations;
  const bool exact_kind = CudaTracePolicyDescriptorIsExact(policy) &&
      kind == CudaTracePreconditionerKind::hypre_ilu0;
  if (!exact_kind || ilu_type != 0 || level_of_fill != 0 ||
      (local_reordering != 0 && local_reordering != 1) ||
      application_iterations != 1 ||
      (triangular_solve != 0 && triangular_solve != 1) ||
      (triangular_solve_kind == CudaIluTriangularSolveKind::direct &&
       triangular_solve != 1) ||
      (triangular_solve_kind ==
           CudaIluTriangularSolveKind::iterative_jacobi &&
       triangular_solve != 0) ||
      (triangular_solve == 0 &&
       (lower_jacobi_iterations <= 0 || upper_jacobi_iterations <= 0)))
    return false;
  try
  {
    // Pinned MFEM v4.9 HypreGMRES::SetOperator forwards the new matrix to the
    // preconditioner.  HypreILU::SetOperator then recreates its HYPRE handle
    // with MFEM defaults (fill 1/reorder 1), so every selected ILU setting
    // must follow this generation's SetOperator call.
    gmres.SetPreconditioner(ilu);
    evidence.Advance(CudaHypreSetupTransition::bind_preconditioner);
    gmres.SetOperator(matrix);
    evidence.Advance(CudaHypreSetupTransition::set_operator);
    ilu.SetType(ilu_type);
    evidence.Advance(CudaHypreSetupTransition::ilu_type);
    ilu.SetLevelOfFill(level_of_fill);
    evidence.Advance(CudaHypreSetupTransition::ilu_fill);
    ilu.SetLocalReordering(local_reordering);
    evidence.Advance(CudaHypreSetupTransition::ilu_reordering);
    const HYPRE_Solver raw_ilu = static_cast<HYPRE_Solver>(ilu);
    if (HYPRE_ILUSetTriSolve(raw_ilu, triangular_solve) != 0)
      return false;
    // Pin even inactive direct-solve fields to zero so diagnostics describe
    // exact live HYPRE state rather than its unused 5/5 defaults.
    if (HYPRE_ILUSetLowerJacobiIters(
            raw_ilu, lower_jacobi_iterations) != 0 ||
        HYPRE_ILUSetUpperJacobiIters(
            raw_ilu, upper_jacobi_iterations) != 0)
      return false;
    const auto* const raw_ilu_data =
        reinterpret_cast<const hypre_ParILUData*>(raw_ilu);
    if (raw_ilu_data == nullptr ||
        hypre_ParILUDataTriSolve(raw_ilu_data) != triangular_solve ||
        hypre_ParILUDataLowerJacobiIters(raw_ilu_data) !=
            lower_jacobi_iterations ||
        hypre_ParILUDataUpperJacobiIters(raw_ilu_data) !=
            upper_jacobi_iterations)
      return false;
    evidence.Advance(CudaHypreSetupTransition::ilu_tri_solve_configured);
    if (hypre_ParILUDataIterativeSetupType(raw_ilu_data) != 0)
      return false;
    evidence.Advance(CudaHypreSetupTransition::ilu_iterative_setup_off);
    ilu.SetMaxIter(application_iterations);
    evidence.Advance(CudaHypreSetupTransition::ilu_application_iterations);
    ilu.SetTol(0.0);
    evidence.Advance(CudaHypreSetupTransition::ilu_tol0);
    ilu.SetPrintLevel(0);
    evidence.Advance(CudaHypreSetupTransition::ilu_print0);
    // SetOperator already installs the recreated ILU handle.  Rebinding after
    // the setters makes that handle association explicit without resetting it.
    gmres.SetPreconditioner(ilu);
    evidence.Advance(CudaHypreSetupTransition::rebind_preconditioner);
    if (!(relative_tolerance > 0.0) ||
        !std::isfinite(relative_tolerance) || maximum_iterations <= 0 ||
        krylov_dimension <= 0)
      return false;
    gmres.SetTol(relative_tolerance);
    gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(maximum_iterations);
    gmres.SetKDim(krylov_dimension);
    gmres.SetLogging(1);
    gmres.SetPrintLevel(0);
    gmres.SetZeroInitialIterate();
    evidence.Advance(CudaHypreSetupTransition::gmres_settings);
    auto* const ilu_data = reinterpret_cast<hypre_ParILUData*>(
        static_cast<HYPRE_Solver>(ilu));
    const bool effective_ilu = ilu_data != nullptr &&
        hypre_ParILUDataIluType(ilu_data) == ilu_type &&
        hypre_ParILUDataLfil(ilu_data) == level_of_fill &&
        hypre_ParILUDataReorderingType(ilu_data) == local_reordering &&
        hypre_ParILUDataMaxIter(ilu_data) == application_iterations &&
        hypre_ParILUDataTol(ilu_data) == 0.0 &&
        hypre_ParILUDataPrintLevel(ilu_data) == 0 &&
        hypre_ParILUDataTriSolve(ilu_data) == triangular_solve &&
        hypre_ParILUDataLowerJacobiIters(ilu_data) ==
            lower_jacobi_iterations &&
        hypre_ParILUDataUpperJacobiIters(ilu_data) ==
            upper_jacobi_iterations &&
        hypre_ParILUDataIterativeSetupType(ilu_data) == 0;
    diagnostic.result = effective_ilu &&
        evidence.ConfigurationExactForGeneration(matrix_generation, policy);
    return diagnostic.result;
  }
  catch (...) { return false; }
}

bool CudaConfigurePersistentDiagonal(
    mfem::HypreParMatrix& matrix, mfem::HypreDiagScale& diagonal,
    mfem::HypreGMRES& gmres, const CudaTracePolicyDescriptor& policy,
    const double relative_tolerance, const int maximum_iterations,
    const int krylov_dimension,
    const std::uint64_t matrix_generation,
    CudaDiagScaleSetupState& evidence) noexcept
{
  CudaTrueSystemDiagnosticScope diagnostic("diagonal-configure",
                                           matrix_generation);
  const generic::CudaEngineKind engine = policy.selected_engine;
  const CudaTracePreconditionerKind kind = policy.trace_preconditioner;
  const bool igr_active = policy.igr_active;
  evidence = {};
  evidence.matrix_generation = matrix_generation;
  evidence.selected_engine = engine;
  evidence.trace_preconditioner = kind;
  evidence.igr_active = igr_active;
  const bool exact_kind = CudaTracePolicyDescriptorIsExact(policy) &&
      !igr_active &&
      kind == CudaTracePreconditionerKind::hypre_diag_scale;
  if (!exact_kind) return false;
  try
  {
    gmres.SetPreconditioner(diagonal);
    evidence.Advance(CudaHypreSetupTransition::bind_preconditioner);
    gmres.SetOperator(matrix);
    evidence.Advance(CudaHypreSetupTransition::set_operator);
    gmres.SetPreconditioner(diagonal);
    evidence.Advance(CudaHypreSetupTransition::rebind_preconditioner);
    if (!(relative_tolerance > 0.0) ||
        !std::isfinite(relative_tolerance) || maximum_iterations <= 0 ||
        krylov_dimension <= 0)
      return false;
    gmres.SetTol(relative_tolerance);
    gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(maximum_iterations);
    gmres.SetKDim(krylov_dimension);
    gmres.SetLogging(1);
    gmres.SetPrintLevel(0);
    gmres.SetZeroInitialIterate();
    evidence.Advance(CudaHypreSetupTransition::gmres_settings);
    const auto& configured_diagonal =
        static_cast<const mfem::HypreDiagScale&>(diagonal);
    diagnostic.result = configured_diagonal.GetData() == &matrix &&
        evidence.ConfigurationExactForGeneration(
            matrix_generation, policy);
    return diagnostic.result;
  }
  catch (...) { return false; }
}

bool CudaHypreDevicePolicyValid(mfem::HypreParMatrix& matrix,
                                mfem::HypreParVector& rhs,
                                mfem::HypreParVector& solution) noexcept
{
  CudaTrueSystemDiagnosticScope diagnostic("hypre-vector-touch");
  hypre_ParCSRMatrix* a = matrix;
  hypre_ParVector* b = rhs;
  hypre_ParVector* x = solution;
  const auto am = hypre_ParCSRMatrixMemoryLocation(a);
  const auto bm = hypre_ParVectorMemoryLocation(b);
  const auto xm = hypre_ParVectorMemoryLocation(x);
  diagnostic.result = UmpireDeviceAllocatorPolicy::DevicePoolPolicyValid() &&
      am == HYPRE_MEMORY_DEVICE && bm == HYPRE_MEMORY_DEVICE &&
      xm == HYPRE_MEMORY_DEVICE &&
      hypre_GetExecPolicy1(am) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy2(am, bm) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy2(am, xm) == HYPRE_EXEC_DEVICE;
  return diagnostic.result;
}

namespace
{
struct CudaLiveIluFactorIdentity
{
  std::uintptr_t hypre_matrix_token = 0;
  std::uintptr_t ilu_handle_token = 0;
  std::uintptr_t factor_matrix_token = 0;
  std::uintptr_t factor_values_token = 0;
  std::uintptr_t factor_permutation_token = 0;
  std::uintptr_t jacobi_diagonal_vector_token = 0;
  std::uintptr_t jacobi_diagonal_values_token = 0;
};

bool CudaStableTrueMatrixExact(
    mfem::HypreParMatrix& matrix,
    const CudaIluAttemptFactorState& state,
    CudaLiveIluFactorIdentity& identity) noexcept
{
  CudaTrueMatrixDeviceView current{};
  hypre_ParCSRMatrix* const raw_matrix = matrix;
  const bool current_view_ready =
      CudaGetTrueMatrixDeviceView(matrix, current);
  identity.hypre_matrix_token =
      reinterpret_cast<std::uintptr_t>(raw_matrix);
  return current_view_ready && state.active &&
      state.matrix_object_token ==
          reinterpret_cast<std::uintptr_t>(&matrix) &&
      (state.hypre_matrix_token == 0 ||
       state.hypre_matrix_token == identity.hypre_matrix_token) &&
      state.diag_values_token ==
          reinterpret_cast<std::uintptr_t>(current.diag_values) &&
      state.offd_values_token ==
          reinterpret_cast<std::uintptr_t>(current.offd_values) &&
      state.diag_nonzeros == current.diag_nonzeros &&
      state.offd_nonzeros == current.offd_nonzeros;
}

bool CudaExactIluGmresConfiguration(
    mfem::HypreParMatrix& matrix, mfem::HypreILU& ilu,
    mfem::HypreGMRES& gmres, const CudaTracePolicyDescriptor& policy,
    const double relative_tolerance,
    const int maximum_iterations, const int krylov_dimension,
    const bool factor_required,
    CudaLiveIluFactorIdentity& identity) noexcept
{
  const CudaTracePreconditionerKind kind = policy.trace_preconditioner;
  const bool igr_active = policy.igr_active;
  hypre_ParCSRMatrix* const raw_matrix = matrix;
  const HYPRE_Solver raw_ilu = static_cast<HYPRE_Solver>(ilu);
  const HYPRE_Solver raw_gmres = static_cast<HYPRE_Solver>(gmres);
  auto* const ilu_data =
      reinterpret_cast<hypre_ParILUData*>(raw_ilu);
  auto* const gmres_data =
      reinterpret_cast<hypre_GMRESData*>(raw_gmres);
  if (raw_matrix == nullptr || raw_ilu == nullptr || raw_gmres == nullptr ||
      ilu_data == nullptr || gmres_data == nullptr ||
      !CudaTracePolicyDescriptorIsExact(policy) ||
      kind != CudaTracePreconditionerKind::hypre_ilu0)
    return false;

  const int triangular_solve = policy.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::direct
      ? 1 : policy.ilu_triangular_solve_kind ==
              CudaIluTriangularSolveKind::iterative_jacobi ? 0 : -1;
  const int lower_jacobi_iterations =
      policy.ilu_lower_jacobi_iterations;
  const int upper_jacobi_iterations =
      policy.ilu_upper_jacobi_iterations;

  HYPRE_Real configured_tolerance = 0.0;
  HYPRE_Real configured_absolute_tolerance = 0.0;
  HYPRE_Real configured_convergence_factor_tolerance = 0.0;
  HYPRE_Int configured_maximum_iterations = 0;
  HYPRE_Int configured_krylov_dimension = 0;
  HYPRE_Int configured_relative_change = 0;
  HYPRE_Int configured_logging = 0;
  HYPRE_Int configured_print_level = 0;
  HYPRE_Solver configured_preconditioner = nullptr;
  const bool gmres_getters_ready =
      HYPRE_GMRESGetTol(raw_gmres, &configured_tolerance) == 0 &&
      HYPRE_GMRESGetAbsoluteTol(
          raw_gmres, &configured_absolute_tolerance) == 0 &&
      HYPRE_GMRESGetConvergenceFactorTol(
          raw_gmres, &configured_convergence_factor_tolerance) == 0 &&
      HYPRE_GMRESGetMaxIter(
          raw_gmres, &configured_maximum_iterations) == 0 &&
      HYPRE_GMRESGetKDim(raw_gmres, &configured_krylov_dimension) == 0 &&
      HYPRE_GMRESGetRelChange(
          raw_gmres, &configured_relative_change) == 0 &&
      HYPRE_GMRESGetLogging(raw_gmres, &configured_logging) == 0 &&
      HYPRE_GMRESGetPrintLevel(raw_gmres, &configured_print_level) == 0 &&
      HYPRE_GMRESGetPrecond(raw_gmres, &configured_preconditioner) == 0;

  hypre_CSRMatrix* const factor =
      hypre_ParILUDataMatBILUDevice(ilu_data);
  HYPRE_Int* const factor_permutation =
      hypre_ParILUDataPerm(ilu_data);
  hypre_Vector* const jacobi_diagonal =
      hypre_ParILUDataADiagDiag(ilu_data);
  identity.hypre_matrix_token =
      reinterpret_cast<std::uintptr_t>(raw_matrix);
  identity.ilu_handle_token = reinterpret_cast<std::uintptr_t>(raw_ilu);
  identity.factor_matrix_token = reinterpret_cast<std::uintptr_t>(factor);
  identity.factor_values_token = factor == nullptr ? 0 :
      reinterpret_cast<std::uintptr_t>(hypre_CSRMatrixData(factor));
  identity.factor_permutation_token =
      reinterpret_cast<std::uintptr_t>(factor_permutation);
  identity.jacobi_diagonal_vector_token =
      reinterpret_cast<std::uintptr_t>(jacobi_diagonal);
  identity.jacobi_diagonal_values_token = jacobi_diagonal == nullptr ? 0 :
      reinterpret_cast<std::uintptr_t>(hypre_VectorData(jacobi_diagonal));

  const bool gmres_matrix_exact = !factor_required ||
      (gmres_data->A == raw_matrix &&
       (gmres_data->precond_Mat == nullptr ||
        gmres_data->precond_Mat == raw_matrix));
  hypre_MemoryLocation factor_permutation_location =
      hypre_MEMORY_UNDEFINED;
  const bool factor_permutation_required =
      CudaIluLocalReordering(kind, igr_active) != 0;
  // Pinned HYPRE represents the no-reordering identity permutation by a null
  // pointer.  RCM, in contrast, is live factor state and must remain a stable
  // device allocation for exact reuse.
  const bool factor_permutation_exact = factor_required
      ? (factor_permutation_required
            ? factor_permutation != nullptr &&
                hypre_GetPointerLocation(
                    factor_permutation, &factor_permutation_location) == 0 &&
                factor_permutation_location == hypre_MEMORY_DEVICE
            : factor_permutation == nullptr)
      : factor_permutation == nullptr;
  const bool iterative_triangular_solve = triangular_solve == 0;
  const bool iterative_auxiliary_exact = factor_required
      ? (iterative_triangular_solve
            ? jacobi_diagonal != nullptr &&
                hypre_VectorData(jacobi_diagonal) != nullptr &&
                hypre_VectorMemoryLocation(jacobi_diagonal) ==
                    HYPRE_MEMORY_DEVICE
            : jacobi_diagonal == nullptr)
      : factor_permutation == nullptr && jacobi_diagonal == nullptr;
  const bool factor_exact = factor_required
      ? hypre_ParILUDataMatA(ilu_data) == raw_matrix && factor != nullptr &&
          hypre_CSRMatrixData(factor) != nullptr &&
          hypre_CSRMatrixMemoryLocation(factor) == HYPRE_MEMORY_DEVICE &&
          factor_permutation_exact && iterative_auxiliary_exact
      : hypre_ParILUDataMatA(ilu_data) == nullptr && factor == nullptr &&
          iterative_auxiliary_exact;
  return gmres_getters_ready && factor_exact &&
      gmres_matrix_exact && gmres_data->precond_data == raw_ilu &&
      configured_tolerance == relative_tolerance &&
      configured_absolute_tolerance == 0.0 &&
      configured_convergence_factor_tolerance == 0.0 &&
      configured_maximum_iterations == maximum_iterations &&
      configured_krylov_dimension == krylov_dimension &&
      configured_relative_change == 0 &&
      configured_logging == 1 && configured_print_level == 0 &&
      configured_preconditioner == raw_ilu &&
      hypre_ParILUDataIluType(ilu_data) == policy.ilu_type &&
      hypre_ParILUDataLfil(ilu_data) == policy.ilu_level_of_fill &&
      hypre_ParILUDataReorderingType(ilu_data) ==
          policy.ilu_local_reordering &&
      hypre_ParILUDataMaxIter(ilu_data) ==
          policy.ilu_application_iterations &&
      hypre_ParILUDataTol(ilu_data) == 0.0 &&
      hypre_ParILUDataPrintLevel(ilu_data) == 0 &&
      hypre_ParILUDataTriSolve(ilu_data) == triangular_solve &&
      hypre_ParILUDataLowerJacobiIters(ilu_data) ==
          lower_jacobi_iterations &&
      hypre_ParILUDataUpperJacobiIters(ilu_data) ==
          upper_jacobi_iterations &&
      hypre_ParILUDataIterativeSetupType(ilu_data) == 0 &&
      HYPRE_GetError() == 0;
}

bool CudaRetainedIluFactorExact(
    mfem::HypreParMatrix& matrix, mfem::HypreILU& ilu,
    mfem::HypreGMRES& gmres, const CudaTracePolicyDescriptor& policy,
    const double relative_tolerance,
    const int maximum_iterations, const int krylov_dimension,
    const CudaIluAttemptFactorState& state,
    CudaLiveIluFactorIdentity& identity) noexcept
{
  CudaLiveIluFactorIdentity matrix_identity{};
  const bool matrix_exact =
      CudaStableTrueMatrixExact(matrix, state, matrix_identity);
  const bool configuration_exact = CudaExactIluGmresConfiguration(
      matrix, ilu, gmres, policy, relative_tolerance,
      maximum_iterations, krylov_dimension, true, identity);
  return matrix_exact && configuration_exact && state.factor_ready &&
      state.hypre_matrix_token == identity.hypre_matrix_token &&
      state.ilu_handle_token == identity.ilu_handle_token &&
      state.factor_matrix_token == identity.factor_matrix_token &&
      state.factor_values_token == identity.factor_values_token &&
      state.factor_permutation_token ==
          identity.factor_permutation_token &&
      state.jacobi_diagonal_vector_token ==
          identity.jacobi_diagonal_vector_token &&
      state.jacobi_diagonal_values_token ==
          identity.jacobi_diagonal_values_token;
}
} // namespace

bool CudaSolveWithIluFactorPolicy(
    mfem::HypreParMatrix& matrix, mfem::HypreParVector& rhs,
    mfem::HypreParVector& solution, mfem::HypreILU& ilu,
    mfem::HypreGMRES& gmres, const int ranks,
    const MPI_Comm communicator,
    const CudaTracePolicyDescriptor& policy,
    const AttemptIdentity& attempt,
    const double relative_tolerance, const int maximum_iterations,
    const int krylov_dimension,
    const std::uint64_t matrix_generation,
    CudaIluAttemptFactorState& factor_state,
    CudaSolveDiagnostics& diagnostics) noexcept
{
  CudaTrueSystemDiagnosticScope diagnostic("ilu-factor-policy-solve",
                                           matrix_generation);
  const CudaTracePreconditionerKind kind = policy.trace_preconditioner;
  const bool igr_active = policy.igr_active;
  const CudaIluFactorReusePolicy factor_reuse_policy =
      policy.ilu_factor_reuse_policy;
  diagnostics = {};
  diagnostics.matrix_generation = matrix_generation;
  diagnostics.trace_preconditioner = kind;
  diagnostics.igr_active = igr_active;
  diagnostics.ilu_type = policy.ilu_type;
  diagnostics.ilu_level_of_fill = policy.ilu_level_of_fill;
  diagnostics.ilu_local_reordering = policy.ilu_local_reordering;
  diagnostics.ilu_application_iterations =
      policy.ilu_application_iterations;
  diagnostics.ilu_triangular_solve_kind =
      policy.ilu_triangular_solve_kind;
  diagnostics.ilu_lower_jacobi_iterations =
      policy.ilu_lower_jacobi_iterations;
  diagnostics.ilu_upper_jacobi_iterations =
      policy.ilu_upper_jacobi_iterations;
  diagnostics.krylov_dimension = krylov_dimension;
  diagnostics.ilu_accounting_kind =
      CudaIluAccountingKind::derived_bounds_v1;
  diagnostics.ilu_factor_reuse_policy = factor_reuse_policy;
  diagnostics.ilu_factor_age_generations = 0;

  const bool iterative_factor_expected =
      diagnostics.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::iterative_jacobi;
  const bool factor_permutation_expected =
      diagnostics.ilu_local_reordering != 0;
  const bool state_generation_exact = factor_state.factor_ready
      ? factor_state.hypre_matrix_token != 0 &&
          factor_state.ilu_handle_token != 0 &&
          factor_state.factor_matrix_token != 0 &&
          factor_state.factor_values_token != 0 &&
          (factor_permutation_expected
               ? factor_state.factor_permutation_token != 0
               : factor_state.factor_permutation_token == 0) &&
          (iterative_factor_expected
               ? factor_state.jacobi_diagonal_vector_token != 0 &&
                   factor_state.jacobi_diagonal_values_token != 0
               : factor_state.jacobi_diagonal_vector_token == 0 &&
                   factor_state.jacobi_diagonal_values_token == 0) &&
          factor_state.factor_generation <=
              factor_state.last_solve_generation &&
          matrix_generation > factor_state.last_solve_generation
      : factor_state.hypre_matrix_token == 0 &&
          factor_state.ilu_handle_token == 0 &&
          factor_state.factor_matrix_token == 0 &&
          factor_state.factor_values_token == 0 &&
          factor_state.factor_permutation_token == 0 &&
          factor_state.jacobi_diagonal_vector_token == 0 &&
          factor_state.jacobi_diagonal_values_token == 0 &&
          factor_state.factor_generation == 0 &&
          factor_state.last_solve_generation == 0;
  const bool local_policy_exact = ranks > 0 &&
      CudaTracePolicyDescriptorIsExact(policy) &&
      kind == CudaTracePreconditionerKind::hypre_ilu0 &&
      factor_reuse_policy != CudaIluFactorReusePolicy::not_applicable &&
      (factor_reuse_policy !=
           CudaIluFactorReusePolicy::first_trace_solve_per_attempt ||
       igr_active) &&
      CudaIluFactorAttemptMatches(
          factor_state, attempt, factor_reuse_policy) &&
      state_generation_exact;
  if (!AllRanksReady(local_policy_exact, communicator, ranks))
    return false;

  // A typed refresh-each policy has one possible branch on every rank.  Only
  // an explicitly selected first-per-attempt factor lifetime requires a
  // collective refresh/reuse decision.
  bool refresh_factor = true;
  if (CudaIluRefreshDecisionRequiresConsensus(factor_reuse_policy))
  {
    const int local_refresh =
        CudaIluFactorRefreshRequired(factor_state) ? 1 : 0;
    if (ranks == 1)
      refresh_factor = local_refresh == 1;
    else
    {
      int refresh_votes = 0;
      const bool branch_consensus = MPI_Allreduce(
          &local_refresh, &refresh_votes, 1, MPI_INT, MPI_SUM,
          communicator) == MPI_SUCCESS &&
          (refresh_votes == 0 || refresh_votes == ranks);
      if (!branch_consensus) { return false; }
      refresh_factor = refresh_votes == ranks;
    }
  }

  CudaHypreSetupState setup_order{};
  const bool incoming_hypre_clean = HYPRE_GetError() == 0;
  if (!AllRanksReady(incoming_hypre_clean, communicator, ranks)) return false;
  CudaLiveIluFactorIdentity before_factor{};
  bool policy_ready = false;
  try
  {
    policy_ready = CudaHypreDevicePolicyValid(matrix, rhs, solution);
    if (policy_ready && refresh_factor)
    {
      policy_ready = CudaConfigurePersistentHypre(
          matrix, ilu, gmres, ranks, policy,
          relative_tolerance, maximum_iterations, krylov_dimension,
          matrix_generation, setup_order) &&
          CudaStableTrueMatrixExact(matrix, factor_state, before_factor) &&
          CudaExactIluGmresConfiguration(
              matrix, ilu, gmres, policy,
              relative_tolerance,
              maximum_iterations, krylov_dimension, false, before_factor) &&
          before_factor.factor_matrix_token == 0 &&
          before_factor.factor_values_token == 0 &&
          before_factor.factor_permutation_token == 0 &&
          before_factor.jacobi_diagonal_vector_token == 0 &&
          before_factor.jacobi_diagonal_values_token == 0;
    }
    else if (policy_ready)
    {
      // Reuse is a direct Mult on the already-set-up MFEM GMRES.  No setter,
      // SetOperator call, callback replacement, or adaptive refresh exists on
      // this branch.
      policy_ready = CudaRetainedIluFactorExact(
          matrix, ilu, gmres, policy,
          relative_tolerance,
          maximum_iterations, krylov_dimension, factor_state,
          before_factor);
    }
    policy_ready = policy_ready && HYPRE_GetError() == 0;
  }
  catch (...) { policy_ready = false; }
  if (!AllRanksReady(policy_ready, communicator, ranks)) { return false; }
  bool vectors_ready = false;
  try
  {
    solution = 0.0;
    solution.HypreReadWrite();
    vectors_ready = HYPRE_GetError() == 0;
  }
  catch (...) { vectors_ready = false; }
  if (!AllRanksReady(vectors_ready, communicator, ranks)) { return false; }
  try
  {
    bool mult_returned = false;
    try
    {
      // Keep MFEM's original HYPRE_ILUSolve/HYPRE_ILUSetup callbacks and raw
      // ILU preconditioner data untouched.  Work is bounded analytically from
      // the completed GMRES solve below.
      gmres.Mult(rhs, solution);
      mult_returned = true;
    }
    catch (...) {}
    const HYPRE_Int terminal_hypre_error = HYPRE_GetError();
    // An exact convergence warning is bounded-solver telemetry; the
    // independently rebuilt true residual remains authoritative.  Any other
    // sticky bit is fatal and is never cleared here.
    const bool terminal_clean = AllRanksReady(
        terminal_hypre_error == 0, communicator, ranks);
    const bool terminal_pure_convergence = AllRanksReady(
        terminal_hypre_error == HYPRE_ERROR_CONV, communicator, ranks);
    bool terminal_error_cleared = terminal_clean && HYPRE_GetError() == 0;
    if (terminal_pure_convergence)
    {
      terminal_error_cleared =
          HYPRE_ClearError(HYPRE_ERROR_CONV) == 0 && HYPRE_GetError() == 0;
    }
    const bool lifecycle_ready = mult_returned &&
        (terminal_clean != terminal_pure_convergence) &&
        terminal_error_cleared;
    if (!AllRanksReady(lifecycle_ready, communicator, ranks)) { return false; }

    CudaLiveIluFactorIdentity after_factor{};
    const bool stable_matrix =
        CudaStableTrueMatrixExact(matrix, factor_state, after_factor);
    const bool exact_live_factor = CudaExactIluGmresConfiguration(
        matrix, ilu, gmres, policy, relative_tolerance,
        maximum_iterations, krylov_dimension, true, after_factor);
    const bool factor_operation_exact = refresh_factor
        ? setup_order.Advance(CudaHypreSetupTransition::mult) &&
            setup_order.SolveExactForGeneration(
                matrix_generation, policy)
        : before_factor.hypre_matrix_token ==
                after_factor.hypre_matrix_token &&
            before_factor.ilu_handle_token ==
                after_factor.ilu_handle_token &&
            before_factor.factor_matrix_token ==
                after_factor.factor_matrix_token &&
            before_factor.factor_values_token ==
                after_factor.factor_values_token &&
            before_factor.factor_permutation_token ==
                after_factor.factor_permutation_token &&
            before_factor.jacobi_diagonal_vector_token ==
                after_factor.jacobi_diagonal_vector_token &&
            before_factor.jacobi_diagonal_values_token ==
                after_factor.jacobi_diagonal_values_token &&
            CudaRetainedIluFactorExact(
                matrix, ilu, gmres, policy,
                relative_tolerance,
                maximum_iterations, krylov_dimension, factor_state,
                after_factor);
    diagnostics.device_policy =
        CudaHypreDevicePolicyValid(matrix, rhs, solution);
    diagnostics.configured_ilu = stable_matrix && exact_live_factor &&
        factor_operation_exact;
    gmres.GetNumIterations(diagnostics.iterations);
    gmres.GetFinalResidualNorm(diagnostics.reported_residual);
    const bool solve_observations_clean = HYPRE_GetError() == 0;
    const bool accounting_inputs_valid = solve_observations_clean &&
        diagnostics.iterations >= 0 && krylov_dimension > 0 &&
        diagnostics.ilu_application_iterations > 0 &&
        diagnostics.ilu_lower_jacobi_iterations >= 0 &&
        diagnostics.ilu_upper_jacobi_iterations >= 0;
    const std::uint64_t jacobi_factor_spmv_updates_per_application =
        accounting_inputs_valid
        ? static_cast<std::uint64_t>(
              diagnostics.ilu_lower_jacobi_iterations > 0
                  ? diagnostics.ilu_lower_jacobi_iterations - 1 : 0) +
            static_cast<std::uint64_t>(
              diagnostics.ilu_upper_jacobi_iterations > 0
                  ? diagnostics.ilu_upper_jacobi_iterations - 1 : 0)
        : 0;
    const CudaDerivedIluAccountingBoundsV1 bounds =
        accounting_inputs_valid
        ? CudaDeriveIluAccountingBoundsV1(
              static_cast<std::uint64_t>(diagnostics.iterations),
              static_cast<std::uint64_t>(krylov_dimension),
              static_cast<std::uint64_t>(
                  diagnostics.ilu_application_iterations),
              jacobi_factor_spmv_updates_per_application)
        : CudaDerivedIluAccountingBoundsV1{};
    // Operation counts describe what the selected factor branch actually did
    // and remain valid even when analytical callback bounds are unavailable.
    diagnostics.ilu_factor_setups = refresh_factor ? 1 : 0;
    diagnostics.ilu_factor_reuses = refresh_factor ? 0 : 1;
    diagnostics.ilu_factor_age_generations = refresh_factor ? 0 :
        matrix_generation - factor_state.factor_generation;
    diagnostics.ilu_preconditioner_callbacks_lower_bound =
        bounds.ilu_preconditioner_callbacks_lower_bound;
    diagnostics.ilu_preconditioner_callbacks_upper_bound =
        bounds.ilu_preconditioner_callbacks_upper_bound;
    diagnostics.ilu_inner_iterations_lower_bound =
        bounds.ilu_inner_iterations_lower_bound;
    diagnostics.ilu_inner_iterations_upper_bound =
        bounds.ilu_inner_iterations_upper_bound;
    diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound =
        bounds.ilu_jacobi_factor_spmv_updates_lower_bound;
    diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound =
        bounds.ilu_jacobi_factor_spmv_updates_upper_bound;
    diagnostics.ilu_accounting_complete =
        bounds.ilu_accounting_complete;
    diagnostics.configured_preconditioner = diagnostics.configured_ilu;
    if (!diagnostics.configured_preconditioner)
      diagnostics.trace_preconditioner =
          CudaTracePreconditionerKind::unconfigured;
    diagnostics.finite = solve_observations_clean &&
        std::isfinite(diagnostics.reported_residual);
    diagnostics.reported_tolerance_met = diagnostics.finite &&
        diagnostics.reported_residual <= relative_tolerance;
    // HYPRE owns its stopping test, whose reported value can differ modestly
    // from the requested tolerance.  Match the CPU lifecycle: admit a bounded
    // solve here and let the independently rebuilt original-system residual
    // remain the acceptance authority.  The report remains telemetry only.
    diagnostic.result = solve_observations_clean &&
        CudaTraceSolveReadyForTrueResidualVerification(
            diagnostics, maximum_iterations);
    if (diagnostic.result)
    {
      factor_state.hypre_matrix_token = after_factor.hypre_matrix_token;
      factor_state.ilu_handle_token = after_factor.ilu_handle_token;
      factor_state.factor_matrix_token = after_factor.factor_matrix_token;
      factor_state.factor_values_token = after_factor.factor_values_token;
      factor_state.factor_permutation_token =
          after_factor.factor_permutation_token;
      factor_state.jacobi_diagonal_vector_token =
          after_factor.jacobi_diagonal_vector_token;
      factor_state.jacobi_diagonal_values_token =
          after_factor.jacobi_diagonal_values_token;
      if (refresh_factor)
        factor_state.factor_generation = matrix_generation;
      factor_state.last_solve_generation = matrix_generation;
      factor_state.factor_ready = true;
    }
    return diagnostic.result;
  }
  catch (...) { return false; }
}

bool CudaRefreshFactorAndSolve(
    mfem::HypreParMatrix& matrix, mfem::HypreParVector& rhs,
    mfem::HypreParVector& solution, mfem::HypreILU& ilu,
    mfem::HypreGMRES& gmres, const int ranks,
    const MPI_Comm communicator,
    const CudaTracePolicyDescriptor& policy,
    const double relative_tolerance, const int maximum_iterations,
    const int krylov_dimension,
    const std::uint64_t matrix_generation,
    CudaSolveDiagnostics& diagnostics) noexcept
{
  CudaTrueMatrixDeviceView view{};
  CudaIluAttemptFactorState factor_state{};
  const AttemptIdentity initialization_attempt{};
  if (!CudaGetTrueMatrixDeviceView(matrix, view) ||
      !CudaBeginIluFactorAttempt(
          factor_state, initialization_attempt,
          policy.ilu_factor_reuse_policy,
          reinterpret_cast<std::uintptr_t>(&matrix),
          reinterpret_cast<std::uintptr_t>(view.diag_values),
          reinterpret_cast<std::uintptr_t>(view.offd_values),
          view.diag_nonzeros, view.offd_nonzeros))
    return false;
  return CudaSolveWithIluFactorPolicy(
      matrix, rhs, solution, ilu, gmres, ranks, communicator, policy,
      initialization_attempt,
      relative_tolerance, maximum_iterations, krylov_dimension,
      matrix_generation, factor_state, diagnostics);
}

bool CudaRefreshDiagonalAndSolve(
    mfem::HypreParMatrix& matrix, mfem::HypreParVector& rhs,
    mfem::HypreParVector& solution, mfem::HypreDiagScale& diagonal,
    mfem::HypreGMRES& gmres, const int ranks, const MPI_Comm communicator,
    const CudaTracePolicyDescriptor& policy,
    const double relative_tolerance, const int maximum_iterations,
    const int krylov_dimension, const std::uint64_t matrix_generation,
    CudaSolveDiagnostics& diagnostics) noexcept
{
  CudaTrueSystemDiagnosticScope diagnostic("diagonal-solve",
                                           matrix_generation);
  diagnostics = {};
  diagnostics.matrix_generation = matrix_generation;
  CudaDiagScaleSetupState setup_order{};
  const bool incoming_hypre_clean = HYPRE_GetError() == 0;
  if (!AllRanksReady(incoming_hypre_clean, communicator, ranks)) return false;
  bool policy_ready = false;
  try
  {
    policy_ready = CudaHypreDevicePolicyValid(matrix, rhs, solution) &&
        CudaConfigurePersistentDiagonal(
            matrix, diagonal, gmres, policy,
            relative_tolerance,
            maximum_iterations, krylov_dimension, matrix_generation,
            setup_order) &&
        HYPRE_GetError() == 0;
  }
  catch (...) { policy_ready = false; }
  if (!AllRanksReady(policy_ready, communicator, ranks)) { return false; }
  bool vectors_ready = false;
  try
  {
    solution = 0.0;
    solution.HypreReadWrite();
    vectors_ready = HYPRE_GetError() == 0;
  }
  catch (...) { vectors_ready = false; }
  if (!AllRanksReady(vectors_ready, communicator, ranks)) { return false; }
  try
  {
    bool mult_returned = false;
    try
    {
      gmres.Mult(rhs, solution);
      mult_returned = true;
    }
    catch (...) {}
    const HYPRE_Int terminal_hypre_error = HYPRE_GetError();
    // Match the ILU lifecycle exactly: a pure convergence warning is bounded
    // solver telemetry and the rebuilt current-matrix true residual remains
    // authoritative.  Any other sticky bit is fatal and is never cleared here.
    const bool terminal_clean = AllRanksReady(
        terminal_hypre_error == 0, communicator, ranks);
    const bool terminal_pure_convergence = AllRanksReady(
        terminal_hypre_error == HYPRE_ERROR_CONV, communicator, ranks);
    bool terminal_error_cleared = terminal_clean && HYPRE_GetError() == 0;
    if (terminal_pure_convergence)
    {
      terminal_error_cleared =
          HYPRE_ClearError(HYPRE_ERROR_CONV) == 0 && HYPRE_GetError() == 0;
    }
    const bool lifecycle_ready = mult_returned &&
        (terminal_clean != terminal_pure_convergence) &&
        terminal_error_cleared;
    if (!AllRanksReady(lifecycle_ready, communicator, ranks)) { return false; }

    if (!setup_order.Advance(CudaHypreSetupTransition::mult)) { return false; }
    diagnostics.device_policy =
        CudaHypreDevicePolicyValid(matrix, rhs, solution);
    diagnostics.configured_diagonal =
        setup_order.SolveExactForGeneration(
            matrix_generation, policy);
    diagnostics.configured_preconditioner =
        diagnostics.configured_diagonal;
    diagnostics.trace_preconditioner = diagnostics.configured_diagonal
        ? CudaTracePreconditionerKind::hypre_diag_scale
        : CudaTracePreconditionerKind::unconfigured;
    gmres.GetNumIterations(diagnostics.iterations);
    gmres.GetFinalResidualNorm(diagnostics.reported_residual);
    const bool solve_observations_clean = HYPRE_GetError() == 0;
    diagnostics.finite = solve_observations_clean &&
        std::isfinite(diagnostics.reported_residual);
    diagnostics.reported_tolerance_met = diagnostics.finite &&
        diagnostics.reported_residual <= relative_tolerance;
    diagnostic.result = solve_observations_clean &&
        CudaTraceSolveReadyForTrueResidualVerification(
            diagnostics, maximum_iterations);
    return diagnostic.result;
  }
  catch (...) { return false; }
}

namespace
{
constexpr double cuda_gamma = 1.4;
constexpr double cuda_tau = 2.5;
constexpr double cuda_density_gate = 1.0e-8;
constexpr double cuda_pi = 3.141592653589793238462643383279502884;

__device__ int ByNodes(const int scalar, const int count,
                       const int component)
{ return scalar + count * component; }

__device__ bool Finite4(const double* state)
{
  for (int component = 0; component < conserved_components; ++component)
    if (!isfinite(state[component])) { return false; }
  return true;
}

__device__ bool FluxJacobian(const double* state, const double nx,
                             const double ny, double* flux, double* jacobian)
{
  if (!Finite4(state) || !(state[0] > 0.0) ||
      (!isfinite(nx) || !isfinite(ny)) || (nx == 0.0 && ny == 0.0))
    return false;
  const double rho = state[0], mx = state[1], my = state[2], e = state[3];
  if (!isfinite(mx / rho) || !isfinite(my / rho)) return false;
  const double kinetic = 0.5 * (mx * mx + my * my) / rho;
  const double pressure = (cuda_gamma - 1.0) * (e - kinetic);
  if (!(pressure > 0.0) || !isfinite(pressure)) { return false; }
  const double q = mx * nx + my * ny;
  flux[0] = q;
  flux[1] = mx * q / rho + pressure * nx;
  flux[2] = my * q / rho + pressure * ny;
  flux[3] = (e + pressure) * q / rho;
  const double pr = (cuda_gamma - 1.0) *
      0.5 * (mx * mx + my * my) / (rho * rho);
  const double pmx = -(cuda_gamma - 1.0) * mx / rho;
  const double pmy = -(cuda_gamma - 1.0) * my / rho;
  const double pe = cuda_gamma - 1.0;
  const double h = e + pressure;
  for (int entry = 0; entry < 16; ++entry) jacobian[entry] = 0.0;
  jacobian[1] = nx; jacobian[2] = ny;
  jacobian[4] = -mx * q / (rho * rho) + pr * nx;
  jacobian[5] = q / rho + mx * nx / rho + pmx * nx;
  jacobian[6] = mx * ny / rho + pmy * nx; jacobian[7] = pe * nx;
  jacobian[8] = -my * q / (rho * rho) + pr * ny;
  jacobian[9] = my * nx / rho + pmx * ny;
  jacobian[10] = q / rho + my * ny / rho + pmy * ny;
  jacobian[11] = pe * ny;
  jacobian[12] = pr * q / rho - h * q / (rho * rho);
  jacobian[13] = pmx * q / rho + h * nx / rho;
  jacobian[14] = pmy * q / rho + h * ny / rho;
  jacobian[15] = (1.0 + pe) * q / rho;
  return Finite4(flux) && Finite4(jacobian) && Finite4(jacobian + 4) &&
      Finite4(jacobian + 8) && Finite4(jacobian + 12);
}

// Flux-only form used by post-candidate conservation.  Its admission matches
// the CPU NormalFlux policy exactly and deliberately does not add Jacobian
// finiteness as a stronger diagnostic gate.
__device__ bool EulerNormalFlux(const double* state, const double nx,
                                const double ny, double* flux)
{
  if (!Finite4(state) || !(state[0] > 0.0) || !isfinite(nx) ||
      !isfinite(ny) || (nx == 0.0 && ny == 0.0))
    return false;
  const double rho = state[0];
  const double mx = state[1];
  const double my = state[2];
  const double energy = state[3];
  const double velocity_x = mx / rho;
  const double velocity_y = my / rho;
  if (!isfinite(velocity_x) || !isfinite(velocity_y)) return false;
  const double kinetic = 0.5 * (mx * mx + my * my) / rho;
  const double pressure = (cuda_gamma - 1.0) * (energy - kinetic);
  if (!(pressure > 0.0) || !isfinite(pressure)) return false;
  const double normal_momentum = mx * nx + my * ny;
  flux[0] = normal_momentum;
  flux[1] = mx * normal_momentum / rho + pressure * nx;
  flux[2] = my * normal_momentum / rho + pressure * ny;
  flux[3] = (energy + pressure) * normal_momentum / rho;
  return Finite4(flux);
}

// Euler flux augmented by the scalar IGR stress.  The 4x5 Jacobian has
// columns (rho, rho*u, rho*v, rho*E, sigma), matching the CPU reference.
__device__ bool RegularizedFluxJacobian(
    const double* state, const double sigma, const double nx, const double ny,
    double* flux, double* jacobian)
{
  double conventional[16];
  if (!isfinite(sigma) ||
      !FluxJacobian(state, nx, ny, flux, conventional) || !(state[0] > 0.0))
    return false;
  for (int row = 0; row < conserved_components; ++row)
  {
    for (int column = 0; column < conserved_components; ++column)
      jacobian[row * igr_trace_components + column] =
          conventional[row * conserved_components + column];
    jacobian[row * igr_trace_components + 4] = 0.0;
  }
  const double normal_velocity =
      (state[1] * nx + state[2] * ny) / state[0];
  flux[1] += sigma * nx;
  flux[2] += sigma * ny;
  flux[3] += sigma * normal_velocity;
  jacobian[1 * igr_trace_components + 4] = nx;
  jacobian[2 * igr_trace_components + 4] = ny;
  jacobian[3 * igr_trace_components + 0] -=
      sigma * normal_velocity / state[0];
  jacobian[3 * igr_trace_components + 1] += sigma * nx / state[0];
  jacobian[3 * igr_trace_components + 2] += sigma * ny / state[0];
  jacobian[3 * igr_trace_components + 4] = normal_velocity;
  for (int row = 0; row < conserved_components; ++row)
    for (int column = 0; column < igr_trace_components; ++column)
      if (!isfinite(jacobian[row * igr_trace_components + column]))
        return false;
  return Finite4(flux);
}

__device__ void Manufactured(const double x, const double y, double* state)
{
  const double sx = sin(2.0 * cuda_pi * x), sy = sin(2.0 * cuda_pi * y);
  const double cx = cos(2.0 * cuda_pi * x), cy = cos(2.0 * cuda_pi * y);
  const double rho = 1.0 + 0.1 * sx * sy;
  const double u = 0.35 + 0.05 * cx * sy;
  const double v = -0.20 + 0.05 * sx * cy;
  const double p = 1.0 + 0.1 * cx * cy;
  state[0] = rho; state[1] = rho * u; state[2] = rho * v;
  state[3] = p / (cuda_gamma - 1.0) + 0.5 * rho * (u * u + v * v);
}

__device__ bool Source(const double x, const double y, double* source)
{
  const double sx = sin(2.0 * cuda_pi * x), sy = sin(2.0 * cuda_pi * y);
  const double cx = cos(2.0 * cuda_pi * x), cy = cos(2.0 * cuda_pi * y);
  double state[4]; Manufactured(x, y, state);
  const double rho = state[0], u = state[1] / rho, v = state[2] / rho;
  const double rx = 0.2 * cuda_pi * cx * sy;
  const double ry = 0.2 * cuda_pi * sx * cy;
  const double ux = -0.1 * cuda_pi * sx * sy;
  const double uy = 0.1 * cuda_pi * cx * cy;
  const double vx = 0.1 * cuda_pi * cx * cy;
  const double vy = -0.1 * cuda_pi * sx * sy;
  const double px = -0.2 * cuda_pi * sx * cy;
  const double py = -0.2 * cuda_pi * cx * sy;
  const double dx[4] = {rx, rx*u+rho*ux, rx*v+rho*vx,
      px/(cuda_gamma-1.0)+0.5*rx*(u*u+v*v)+rho*(u*ux+v*vx)};
  const double dy[4] = {ry, ry*u+rho*uy, ry*v+rho*vy,
      py/(cuda_gamma-1.0)+0.5*ry*(u*u+v*v)+rho*(u*uy+v*vy)};
  double unused[4], jx[16], jy[16];
  if (!FluxJacobian(state, 1.0, 0.0, unused, jx) ||
      !FluxJacobian(state, 0.0, 1.0, unused, jy)) return false;
  for (int row = 0; row < 4; ++row)
  {
    source[row] = 0.0;
    for (int column = 0; column < 4; ++column)
      source[row] += jx[row*4+column]*dx[column] +
                     jy[row*4+column]*dy[column];
  }
  return Finite4(source);
}

__device__ bool Admissible(const double* state)
{
  if (!Finite4(state) || !(state[0] >= 1.0e-8)) return false;
  const double kinetic = 0.5 * (state[1]*state[1]+state[2]*state[2])/state[0];
  const double pressure = (cuda_gamma-1.0)*(state[3]-kinetic);
  if (!(pressure >= 1.0e-8) || !isfinite(pressure)) return false;
  const double u = state[1]/state[0], v = state[2]/state[0];
  const double speed = sqrt(u*u+v*v)+sqrt(cuda_gamma*pressure/state[0]);
  return isfinite(speed) && speed < cuda_tau;
}

__device__ bool PhysicallyAdmissible(const double* state)
{
  if (!Finite4(state) || !(state[0] > 0.0)) return false;
  if (!isfinite(state[1] / state[0]) ||
      !isfinite(state[2] / state[0])) return false;
  const double kinetic = 0.5 *
      (state[1] * state[1] + state[2] * state[2]) / state[0];
  const double pressure = (cuda_gamma - 1.0) * (state[3] - kinetic);
  const double sound_squared = cuda_gamma * pressure / state[0];
  return pressure > 0.0 && sound_squared > 0.0 && isfinite(pressure) &&
      isfinite(sound_squared) && isfinite(sqrt(sound_squared));
}

// Borrowed input admission is physical rather than the manufactured W8A
// speed cap: rho, pressure, and sound speed must be finite and strictly
// positive.  A caller decides the lifetime of the returned LLF coefficient;
// production assembly freezes it only across one residual and its A/B/C/D.
__device__ bool LocalLlfSpeed(const double* state, const double nx,
                              const double ny, double& speed)
{
  if (!Finite4(state) || !(state[0] > 0.0) || !isfinite(nx) ||
      !isfinite(ny) || (nx == 0.0 && ny == 0.0)) return false;
  const double rho = state[0];
  const double mx = state[1], my = state[2], energy = state[3];
  if (!isfinite(mx / rho) || !isfinite(my / rho)) return false;
  const double pressure = (cuda_gamma - 1.0) *
      (energy - 0.5 * (mx * mx + my * my) / rho);
  const double sound_squared = cuda_gamma * pressure / rho;
  if (!(pressure > 0.0) || !(sound_squared > 0.0) ||
      !isfinite(pressure) || !isfinite(sound_squared)) return false;
  const double sound = sqrt(sound_squared);
  const double normal_velocity = (mx * nx + my * ny) / rho;
  speed = fabs(normal_velocity) + sound;
  return isfinite(sound) && isfinite(speed) && speed > 0.0;
}

__device__ bool MapStaticBoundaryKind(
    const int kind, mfem::detail::EulerBoundaryStateKind& mapped)
{
  switch (static_cast<CudaStaticBoundaryPolicyKind>(kind))
  {
  case CudaStaticBoundaryPolicyKind::supersonic_inlet:
    mapped = mfem::detail::EulerBoundaryStateKind::supersonic_inlet;
    return true;
  case CudaStaticBoundaryPolicyKind::supersonic_outlet:
    mapped = mfem::detail::EulerBoundaryStateKind::supersonic_outlet;
    return true;
  case CudaStaticBoundaryPolicyKind::slip_wall:
    mapped = mfem::detail::EulerBoundaryStateKind::slip_wall;
    return true;
  case CudaStaticBoundaryPolicyKind::subsonic_inlet:
    mapped = mfem::detail::EulerBoundaryStateKind::subsonic_inlet;
    return true;
  case CudaStaticBoundaryPolicyKind::subsonic_outlet_pressure:
    mapped = mfem::detail::EulerBoundaryStateKind::subsonic_outlet_pressure;
    return true;
  case CudaStaticBoundaryPolicyKind::far_field:
    mapped = mfem::detail::EulerBoundaryStateKind::far_field;
    return true;
  case CudaStaticBoundaryPolicyKind::nonreflecting_outlet:
    mapped = mfem::detail::EulerBoundaryStateKind::nonreflecting_outlet;
    return true;
  case CudaStaticBoundaryPolicyKind::none:
  case CudaStaticBoundaryPolicyKind::subsonic_outlet:
  case CudaStaticBoundaryPolicyKind::nonreflecting:
    return false;
  }
  return false;
}

__device__ bool StaticBoundaryExterior(
    const int kind, const double* reference, const double pressure_out,
    const double gamma, const double* interior, const double normal_x,
    const double normal_y, double* exterior)
{
  if (!PhysicallyAdmissible(interior) || !isfinite(normal_x) ||
      !isfinite(normal_y) || !(gamma > 1.0) || !isfinite(gamma))
    return false;
  const double normal_norm = sqrt(normal_x * normal_x + normal_y * normal_y);
  if (!(normal_norm > 0.0) || !isfinite(normal_norm))
    return false;

  mfem::detail::EulerBoundaryStateKind mapped{};
  if (!MapStaticBoundaryKind(kind, mapped))
    return false;
  const bool reference_required =
      mapped == mfem::detail::EulerBoundaryStateKind::supersonic_inlet ||
      mapped == mfem::detail::EulerBoundaryStateKind::subsonic_inlet ||
      mapped == mfem::detail::EulerBoundaryStateKind::far_field ||
      mapped == mfem::detail::EulerBoundaryStateKind::nonreflecting_outlet;
  if (reference_required &&
      (reference == nullptr || !PhysicallyAdmissible(reference)))
    return false;

  const double normal[dimension] = {normal_x, normal_y};
  if (!mfem::detail::EvaluateEulerBoundaryState(
          mapped, dimension, conserved_components, gamma, pressure_out,
          reference, normal, interior, exterior))
    return false;
  return PhysicallyAdmissible(exterior);
}

__device__ int Decode(const int encoded, double& sign)
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
}

__global__ void TicketKernel(int* words, const int phase,
                             const std::uint64_t attempt,
                             const std::uint64_t generation,
                             const int device)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  words[0] = 1;
#if defined(__CUDA_ARCH__)
  words[1] = __CUDA_ARCH__;
#else
  words[1] = 0;
#endif
  words[2] = device; words[3] = phase;
  words[4] = static_cast<int>(attempt);
  words[5] = static_cast<int>(attempt >> 32U);
  words[6] = static_cast<int>(generation);
  words[7] = static_cast<int>(generation >> 32U);
}

__global__ void GatherKernel(const int* map, const int count,
                             const double* source, double* destination)
{
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= count) return;
  double sign = 1.0;
  const int index = Decode(map[entry], sign);
  destination[entry] = sign * source[index];
}

__global__ void AdmissibilityKernel(CudaDeviceView view, CudaMeshShape shape)
{
  const int element = blockIdx.x;
  if (element >= shape.elements) return;
  __shared__ int volume_failed;
  __shared__ int face_failed;
  if (threadIdx.x == 0)
  {
    volume_failed = 0;
    face_failed = 0;
  }
  __syncthreads();
  const int ni0 = element * shape.interior_dofs;
  const int nt0 = element * shape.trace_dofs;
  for (int q = threadIdx.x; q < shape.volume_points; q += blockDim.x)
  {
    const int qp = element * shape.volume_points + q;
    double state[4] = {};
    for (int scalar = 0; scalar < shape.scalar_interior_dofs; ++scalar)
      for (int component = 0; component < 4; ++component)
        state[component] +=
            view.volume_shape[qp * shape.scalar_interior_dofs + scalar] *
            view.work_interior[ni0 + ByNodes(
                scalar, shape.scalar_interior_dofs, component)];
    if (!(shape.borrowed_host_inputs ? PhysicallyAdmissible(state) :
          Admissible(state))) atomicExch(&volume_failed, 1);
  }
  const int face_points = faces_per_element * shape.face_points;
  for (int fq = threadIdx.x; fq < face_points; fq += blockDim.x)
  {
      const int face = fq / shape.face_points;
      const int q = fq - face * shape.face_points;
    {
      const int qp = (element*faces_per_element+face)*shape.face_points+q;
      const int offset =
          face * shape.trace_components * shape.scalar_face_dofs;
      double interior[4] = {}, trace[4] = {};
      for (int scalar = 0; scalar < shape.scalar_interior_dofs; ++scalar)
        for (int component = 0; component < 4; ++component)
          interior[component] +=
              view.face_interior_shape[
                  qp * shape.scalar_interior_dofs + scalar] *
              view.work_interior[ni0 + ByNodes(
                  scalar, shape.scalar_interior_dofs, component)];
      for (int scalar = 0; scalar < shape.scalar_face_dofs; ++scalar)
        for (int component = 0; component < 4; ++component)
          trace[component] +=
              view.face_trace_shape[qp * shape.scalar_face_dofs + scalar] *
              view.current_local_trace[nt0 + offset + ByNodes(
                  scalar, shape.scalar_face_dofs, component)];
      if (!(shape.borrowed_host_inputs ? PhysicallyAdmissible(interior) :
            Admissible(interior)) ||
          !(shape.borrowed_host_inputs ? PhysicallyAdmissible(trace) :
            Admissible(trace)))
        atomicExch(&face_failed, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    view.status[element] = volume_failed ? -1.0 :
        (face_failed ? -2.0 : 1.0);
}

__global__ void FreezeBorrowedCurrentLlfTauKernel(CudaDeviceView view,
                                                   CudaMeshShape shape)
{
  const int element = blockIdx.x;
  if (element >= shape.elements) return;
  __shared__ int failed;
  if (threadIdx.x == 0) failed = 0;
  __syncthreads();
  const int interior_offset = element * shape.interior_dofs;
  const int trace_offset = element * shape.trace_dofs;
  const int face_points = faces_per_element * shape.face_points;
  for (int fq = threadIdx.x; fq < face_points; fq += blockDim.x)
    {
      const int face = fq / shape.face_points;
      const int q = fq - face * shape.face_points;
      const int qp = (element * faces_per_element + face) *
          shape.face_points + q;
      const int face_offset = face * shape.trace_components *
          shape.scalar_face_dofs;
      double interior[conserved_components] = {};
      double trace[conserved_components] = {};
      for (int scalar = 0; scalar < shape.scalar_interior_dofs; ++scalar)
        for (int component = 0; component < conserved_components; ++component)
          interior[component] += view.face_interior_shape[
              qp * shape.scalar_interior_dofs + scalar] *
              view.work_interior[interior_offset + ByNodes(
                  scalar, shape.scalar_interior_dofs, component)];
      for (int scalar = 0; scalar < shape.scalar_face_dofs; ++scalar)
        for (int component = 0; component < conserved_components; ++component)
          trace[component] += view.face_trace_shape[
              qp * shape.scalar_face_dofs + scalar] *
              view.current_local_trace[trace_offset + face_offset + ByNodes(
                  scalar, shape.scalar_face_dofs, component)];
      double interior_speed = 0.0, trace_speed = 0.0;
      if (!LocalLlfSpeed(interior, view.face_normal[qp * dimension],
                         view.face_normal[qp * dimension + 1], interior_speed) ||
          !LocalLlfSpeed(trace, view.face_normal[qp * dimension],
                         view.face_normal[qp * dimension + 1], trace_speed))
      {
        atomicExch(&failed, 1);
        continue;
      }
      const double tau = fmax(interior_speed, trace_speed);
      if (!(tau > 0.0) || !isfinite(tau))
      {
        atomicExch(&failed, 1);
        continue;
      }
      view.borrowed_lagged_llf_tau[qp] = tau;
    }
  __syncthreads();
  if (threadIdx.x == 0) view.status[element] = failed ? -3.0 : 1.0;
}

// Minimal phase-math port from W7 ac271a0: the signs and A/B/C/D weak
// linearization are retained; no W7 lifecycle/storage code appears here.
__global__ void AssemblyKernel(CudaDeviceView v, CudaMeshShape s,
                               const double dt)
{
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= s.elements) return;
  const int ni = s.interior_dofs, nt = s.trace_dofs;
  const int si = s.scalar_interior_dofs, sf = s.scalar_face_dofs;
  const int e_i = e*ni, e_t = e*nt;
  const std::size_t ao = static_cast<std::size_t>(e)*ni*ni;
  const std::size_t bo = static_cast<std::size_t>(e)*ni*nt;
  const std::size_t co = static_cast<std::size_t>(e)*nt*ni;
  const std::size_t doff = static_cast<std::size_t>(e)*nt*nt;
  v.status[e] = 1.0;
  for (int i=0;i<ni;++i) { v.residual_r[e_i+i]=0; v.f[e_i+i]=0; }
  for (int i=0;i<nt;++i) { v.residual_g[e_t+i]=0; v.g[e_t+i]=0; }
  for (int i=0;i<ni*ni;++i) v.a[ao+i]=0;
  for (int i=0;i<ni*nt;++i) v.b[bo+i]=0;
  for (int i=0;i<nt*ni;++i) v.c[co+i]=0;
  for (int i=0;i<nt*nt;++i) v.d[doff+i]=0;
  for (int q=0;q<s.volume_points;++q)
  {
    const int qp=e*s.volume_points+q;
    const double w=v.volume_weight[qp];
    double state[4]={}, old[4]={};
    for(int a=0;a<si;++a) for(int k=0;k<4;++k)
    {
      const double phi=v.volume_shape[qp*si+a];
      state[k]+=phi*v.work_interior[e_i+ByNodes(a,si,k)];
      old[k]+=phi*v.old_interior[e_i+ByNodes(a,si,k)];
    }
    double source[4]={}, fx[4], fy[4], jx[16], jy[16];
    if (s.borrowed_static_source)
      for(int a=0;a<si;++a) for(int k=0;k<4;++k)
        source[k] += v.volume_shape[qp*si+a] *
            v.static_source[e_i+ByNodes(a,si,k)];
    if (!(w>0) || (!s.borrowed_static_source &&
                    !Source(v.volume_position[qp*2],v.volume_position[qp*2+1],source)) ||
        !FluxJacobian(state,1,0,fx,jx) || !FluxJacobian(state,0,1,fy,jy))
    { v.status[e]=-1; return; }
    for(int row=0;row<si;++row)
    {
      const double phi=v.volume_shape[qp*si+row];
      const double dx=v.volume_derivative[(qp*si+row)*2];
      const double dy=v.volume_derivative[(qp*si+row)*2+1];
      for(int rc=0;rc<4;++rc)
      {
        const int lr=ByNodes(row,si,rc);
        v.residual_r[e_i+lr]+=w*(phi*(state[rc]-old[rc])/dt-dx*fx[rc]-dy*fy[rc]-phi*source[rc]);
        for(int col=0;col<si;++col) for(int cc=0;cc<4;++cc)
        {
          const int lc=ByNodes(col,si,cc);
          const double mass=rc==cc?phi*v.volume_shape[qp*si+col]/dt:0;
          const double flux=-(dx*jx[rc*4+cc]+dy*jy[rc*4+cc])*
              v.volume_shape[qp*si+col];
          v.a[ao+lr*ni+lc]+=w*(mass+flux);
        }
      }
    }
  }
  for(int face=0;face<4;++face) for(int q=0;q<s.face_points;++q)
  {
    const int qp=(e*4+face)*s.face_points+q;
    const int to=face*conserved_components*sf;
    const double w=v.face_weight[qp];
    double u[4]={}, l[4]={};
    for(int a=0;a<si;++a) for(int k=0;k<4;++k)
      u[k]+=v.face_interior_shape[qp*si+a]*
          v.work_interior[e_i+ByNodes(a,si,k)];
    for(int a=0;a<sf;++a) for(int k=0;k<4;++k)
      l[k]+=v.face_trace_shape[qp*sf+a]*
          v.current_local_trace[e_t+to+ByNodes(a,sf,k)];
    double flux[4], jac[16];
    if(!(w>0)||!FluxJacobian(l,v.face_normal[qp*2],v.face_normal[qp*2+1],flux,jac))
    {v.status[e]=-2;return;}
    for(int row=0;row<si;++row)
    {
      const double rp=v.face_interior_shape[qp*si+row];
      for(int rc=0;rc<4;++rc)
      {
        const int lr=ByNodes(row,si,rc);
        v.residual_r[e_i+lr]+=w*rp*(flux[rc]+cuda_tau*(u[rc]-l[rc]));
        for(int col=0;col<si;++col)
          v.a[ao+lr*ni+ByNodes(col,si,rc)]+=w*rp*cuda_tau*
              v.face_interior_shape[qp*si+col];
        for(int col=0;col<sf;++col) for(int cc=0;cc<4;++cc)
          v.b[bo+lr*nt+to+ByNodes(col,sf,cc)]+=w*rp*
              v.face_trace_shape[qp*sf+col]*
              (jac[rc*4+cc]-(rc==cc?cuda_tau:0));
      }
    }
    for(int row=0;row<sf;++row)
    {
      const double rp=v.face_trace_shape[qp*sf+row];
      for(int rc=0;rc<4;++rc)
      {
        const int lr=to+ByNodes(row,sf,rc);
        v.residual_g[e_t+lr]+=w*rp*(flux[rc]+cuda_tau*(u[rc]-l[rc]));
        for(int col=0;col<si;++col)
          v.c[co+lr*ni+ByNodes(col,si,rc)]+=w*rp*cuda_tau*
              v.face_interior_shape[qp*si+col];
        for(int col=0;col<sf;++col) for(int cc=0;cc<4;++cc)
          v.d[doff+lr*nt+to+ByNodes(col,sf,cc)]+=w*rp*
              v.face_trace_shape[qp*sf+col]*
              (jac[rc*4+cc]-(rc==cc?cuda_tau:0));
      }
    }
  }
  for(int i=0;i<ni;++i) v.f[e_i+i]=-v.residual_r[e_i+i];
  for(int i=0;i<nt;++i) v.g[e_t+i]=-v.residual_g[e_t+i];
}

// AssemblyKernel above remains the fixed-tau manufactured path and keeps its
// established arithmetic unchanged.  The live borrowed split kernels below
// evaluate conventional LLF tau from the current interior and trace state at
// each face quadrature point, then hold it fixed while forming the residual and
// Jacobian blocks A/B/C/D; there are deliberately no d-tau terms.  Their
// frozen_llf branch exists only for the private finite-difference audit.
struct BorrowedVolumeCache
{
  double weight;
  double state[conserved_components];
  double old[conserved_components];
  double source[conserved_components];
  double flux_x[conserved_components];
  double flux_y[conserved_components];
  double jacobian_x[conserved_components * igr_trace_components];
  double jacobian_y[conserved_components * igr_trace_components];
  double sigma;
  double z[dimension];
  double conservative_gradient[3][dimension];
  double velocity_gradient[dimension][dimension];
  double dq_dgradient[dimension][dimension];
  double q_value;
  int valid;
};

struct BorrowedFaceCache
{
  double weight;
  double stabilization;
  double interior[conserved_components];
  double trace[conserved_components];
  double flux[conserved_components];
  double jacobian[conserved_components * igr_trace_components];
  double exterior[conserved_components];
  double sigma;
  double sigma_hat;
  double z[dimension];
  double normal[dimension];
  double tau;
  double jump;
  double z_hat_normal;
  int physical_boundary;
  int valid;
};

inline constexpr int borrowed_euler_group_threads = 4;
inline constexpr int borrowed_euler_warp_threads = 32;
inline constexpr int borrowed_euler_groups_per_warp =
    borrowed_euler_warp_threads / borrowed_euler_group_threads;
inline constexpr int borrowed_euler_warps_per_block = 4;
inline constexpr int borrowed_euler_block_threads =
    borrowed_euler_warp_threads * borrowed_euler_warps_per_block;
inline constexpr int borrowed_euler_elements_per_block =
    borrowed_euler_groups_per_warp * borrowed_euler_warps_per_block;
inline constexpr unsigned borrowed_euler_group_mask =
    (1u << borrowed_euler_group_threads) - 1u;

static_assert(borrowed_euler_groups_per_warp == 8);
static_assert(borrowed_euler_elements_per_block == 32);
static_assert(borrowed_euler_group_threads == conserved_components);
static_assert(sizeof(BorrowedVolumeCache) * borrowed_euler_elements_per_block <=
              48 * 1024);
static_assert(sizeof(BorrowedFaceCache) * borrowed_euler_elements_per_block <=
              48 * 1024);

__global__ void BorrowedEulerZeroKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int group=threadIdx.x/borrowed_euler_group_threads;
  const int lane=threadIdx.x%borrowed_euler_group_threads;
  const int e=blockIdx.x*borrowed_euler_elements_per_block+group;
  if(e>=s.elements)return;
  const int ni=s.interior_dofs,nt=s.trace_dofs;
  const int e_i=e*ni,e_t=e*nt;
  const std::size_t ao=static_cast<std::size_t>(e)*ni*ni;
  const std::size_t bo=static_cast<std::size_t>(e)*ni*nt;
  const std::size_t co=static_cast<std::size_t>(e)*nt*ni;
  const std::size_t doff=static_cast<std::size_t>(e)*nt*nt;
  for(int i=lane;i<ni;i+=borrowed_euler_group_threads)
  {v.residual_r[e_i+i]=0.0;v.f[e_i+i]=0.0;}
  for(int i=lane;i<nt;i+=borrowed_euler_group_threads)
  {v.residual_g[e_t+i]=0.0;v.g[e_t+i]=0.0;}
  for(int i=lane;i<ni*ni;i+=borrowed_euler_group_threads)v.a[ao+i]=0.0;
  for(int i=lane;i<ni*nt;i+=borrowed_euler_group_threads)v.b[bo+i]=0.0;
  for(int i=lane;i<nt*ni;i+=borrowed_euler_group_threads)v.c[co+i]=0.0;
  for(int i=lane;i<nt*nt;i+=borrowed_euler_group_threads)v.d[doff+i]=0.0;
  if(lane==0)v.status[e]=1.0;
}

// Each four-lane group owns one element.  Independent cache fields have one
// lane writer and retain their scalar accumulation order; complete matrix rows
// likewise remain lane-owned with their original q and column order.  Exact
// masks and per-group caches isolate the eight groups within each warp.
__global__ void BorrowedEulerVolumeKernel(CudaDeviceView v,CudaMeshShape s,
                                          double dt)
{
  const int group=threadIdx.x/borrowed_euler_group_threads;
  const int lane=threadIdx.x%borrowed_euler_group_threads;
  const int warp_lane=threadIdx.x%borrowed_euler_warp_threads;
  const unsigned group_mask=borrowed_euler_group_mask<<
      (warp_lane-lane);
  const int e=blockIdx.x*borrowed_euler_elements_per_block+group;
  if(e>=s.elements)return;
  const int ni=s.interior_dofs;
  const int si=s.scalar_interior_dofs;
  const int e_i=e*ni;
  const std::size_t ao=static_cast<std::size_t>(e)*ni*ni;
  __shared__ BorrowedVolumeCache caches[borrowed_euler_elements_per_block];
  BorrowedVolumeCache& cache=caches[group];
  for(int q=0;q<s.volume_points;++q)
  {
    const int qp=e*s.volume_points+q;
    if(lane==0)
    {
      cache.weight=v.volume_weight[qp];cache.q_value=0.0;cache.valid=1;
    }
    // Lane c owns conserved field c and its two flux/Jacobian staging rows.
    cache.state[lane]=0.0;cache.old[lane]=0.0;
    cache.source[lane]=0.0;cache.flux_x[lane]=0.0;
    cache.flux_y[lane]=0.0;
    for(int entry=0;entry<igr_trace_components;++entry)
    {
      cache.jacobian_x[lane*igr_trace_components+entry]=0.0;
      cache.jacobian_y[lane*igr_trace_components+entry]=0.0;
    }
    for(int a=0;a<si;++a)
    {
      const double phi=v.volume_shape[qp*si+a];
      cache.state[lane]+=phi*v.work_interior[
          e_i+ByNodes(a,si,lane)];
      cache.old[lane]+=phi*v.old_interior[e_i+ByNodes(a,si,lane)];
      if(s.borrowed_static_source)
        cache.source[lane]+=phi*v.static_source[
            e_i+ByNodes(a,si,lane)];
    }
    if(s.igr_active)
    {
      // Lanes 0--2 own their conservative-gradient component; lane 3 owns
      // sigma and both auxiliary z components.
      if(lane<3)
      {
        for(int direction=0;direction<dimension;++direction)
          cache.conservative_gradient[lane][direction]=0.0;
        for(int a=0;a<si;++a)
          for(int direction=0;direction<dimension;++direction)
            cache.conservative_gradient[lane][direction]+=
                v.volume_derivative[(qp*si+a)*dimension+direction]*
                v.work_interior[e_i+ByNodes(a,si,lane)];
      }
      else
      {
        cache.sigma=0.0;cache.z[0]=0.0;cache.z[1]=0.0;
        for(int a=0;a<si;++a)
        {
          const double phi=v.volume_shape[qp*si+a];
          cache.sigma+=phi*v.work_interior[e_i+ByNodes(a,si,4)];
          cache.z[0]+=phi*v.work_interior[e_i+ByNodes(a,si,5)];
          cache.z[1]+=phi*v.work_interior[e_i+ByNodes(a,si,6)];
        }
      }
    }
    // Publish every independently accumulated field before lane zero performs
    // coupled flux, validity, and IGR derived-field work.
    __syncwarp(group_mask);
    if(lane==0)
    {
      const bool flux_ready=s.igr_active
          ? RegularizedFluxJacobian(cache.state,cache.sigma,1.0,0.0,
                                    cache.flux_x,cache.jacobian_x)&&
            RegularizedFluxJacobian(cache.state,cache.sigma,0.0,1.0,
                                    cache.flux_y,cache.jacobian_y)
          : FluxJacobian(cache.state,1.0,0.0,cache.flux_x,
                         cache.jacobian_x)&&
            FluxJacobian(cache.state,0.0,1.0,cache.flux_y,
                         cache.jacobian_y);
      if(!(cache.weight>0.0)||!flux_ready)cache.valid=0;
      if(cache.valid&&s.igr_active)
      {
        for(int velocity=0;velocity<dimension;++velocity)
          for(int direction=0;direction<dimension;++direction)
            cache.velocity_gradient[velocity][direction]=
                cache.conservative_gradient[1+velocity][direction]/
                    cache.state[0]-cache.state[1+velocity]*
                cache.conservative_gradient[0][direction]/
                    (cache.state[0]*cache.state[0]);
        const double velocity_trace=cache.velocity_gradient[0][0]+
            cache.velocity_gradient[1][1];
        cache.q_value=cache.velocity_gradient[0][0]*
                cache.velocity_gradient[0][0]+
            2.0*cache.velocity_gradient[0][1]*
                cache.velocity_gradient[1][0]+
            cache.velocity_gradient[1][1]*
                cache.velocity_gradient[1][1]+
            velocity_trace*velocity_trace;
        for(int velocity=0;velocity<dimension;++velocity)
          for(int direction=0;direction<dimension;++direction)
            cache.dq_dgradient[velocity][direction]=
                2.0*cache.velocity_gradient[direction][velocity]+
                (velocity==direction?2.0*velocity_trace:0.0);
      }
    }
    __syncwarp(group_mask);
    if(!cache.valid){if(lane==0)v.status[e]=-1.0;return;}
    for(int lr=lane;lr<ni;lr+=borrowed_euler_group_threads)
    {
      const int component=lr/si,row=lr-component*si;
      const double phi=v.volume_shape[qp*si+row];
      const double dx=v.volume_derivative[(qp*si+row)*dimension];
      const double dy=v.volume_derivative[(qp*si+row)*dimension+1];
      const double w=cache.weight;
      if(component<conserved_components)
      {
        v.residual_r[e_i+lr]+=w*(phi*(cache.state[component]-
            cache.old[component])/dt-dx*cache.flux_x[component]-
            dy*cache.flux_y[component]-phi*cache.source[component]);
        const int columns=s.igr_active?igr_trace_components:
            conserved_components;
        for(int col=0;col<si;++col)for(int cc=0;cc<columns;++cc)
        {
          const double column_phi=v.volume_shape[qp*si+col];
          const double mass=component==cc?phi*column_phi/dt:0.0;
          const double flux=-(dx*cache.jacobian_x[component*columns+cc]+
              dy*cache.jacobian_y[component*columns+cc])*column_phi;
          v.a[ao+lr*ni+ByNodes(col,si,cc)]+=w*(mass+flux);
        }
      }
      else if(s.igr_active&&component<6)
      {
        const int direction=component-4;
        const double derivative=v.volume_derivative[
            (qp*si+row)*dimension+direction];
        v.residual_r[e_i+lr]+=w*(phi*cache.state[0]*cache.z[direction]+
                                      derivative*cache.sigma);
        for(int col=0;col<si;++col)
        {
          const double column_phi=v.volume_shape[qp*si+col];
          v.a[ao+lr*ni+ByNodes(col,si,0)]+=
              w*phi*cache.z[direction]*column_phi;
          v.a[ao+lr*ni+ByNodes(col,si,4)]+=
              w*derivative*column_phi;
          v.a[ao+lr*ni+ByNodes(col,si,5+direction)]+=
              w*phi*cache.state[0]*column_phi;
        }
      }
      else if(s.igr_active&&component==6)
      {
        v.residual_r[e_i+lr]+=cache.weight*(phi*cache.sigma/cache.state[0]+
            s.igr_alpha*(dx*cache.z[0]+dy*cache.z[1])-
            s.igr_alpha*phi*cache.q_value);
        for(int col=0;col<si;++col)
        {
          const double column_phi=v.volume_shape[qp*si+col];
          v.a[ao+lr*ni+ByNodes(col,si,4)]+=
              cache.weight*phi*column_phi/cache.state[0];
          for(int direction=0;direction<dimension;++direction)
            v.a[ao+lr*ni+ByNodes(col,si,5+direction)]+=
                cache.weight*s.igr_alpha*v.volume_derivative[
                    (qp*si+row)*dimension+direction]*column_phi;
          for(int variable=0;variable<conserved_components;++variable)
          {
            double dq=0.0;
            if(variable<3)
              for(int velocity=0;velocity<dimension;++velocity)
                for(int direction=0;direction<dimension;++direction)
                {
                  const double momentum=cache.state[1+velocity];
                  const double momentum_derivative=
                      cache.conservative_gradient[1+velocity][direction];
                  const double density_derivative=
                      cache.conservative_gradient[0][direction];
                  const double delta_density=variable==0?column_phi:0.0;
                  const double delta_density_derivative=variable==0?
                      v.volume_derivative[(qp*si+col)*dimension+direction]:0.0;
                  const double delta_momentum=variable==1+velocity?
                      column_phi:0.0;
                  const double delta_momentum_derivative=
                      variable==1+velocity?
                      v.volume_derivative[(qp*si+col)*dimension+direction]:0.0;
                  const double delta_gradient=
                      delta_momentum_derivative/cache.state[0]-
                      momentum_derivative*delta_density/
                          (cache.state[0]*cache.state[0])-
                      delta_momentum*density_derivative/
                          (cache.state[0]*cache.state[0])-
                      momentum*delta_density_derivative/
                          (cache.state[0]*cache.state[0])+
                      2.0*momentum*density_derivative*delta_density/
                          (cache.state[0]*cache.state[0]*cache.state[0]);
                  dq+=cache.dq_dgradient[velocity][direction]*delta_gradient;
                }
            double derivative=-s.igr_alpha*phi*dq;
            if(variable==0)derivative-=phi*cache.sigma*column_phi/
                (cache.state[0]*cache.state[0]);
            v.a[ao+lr*ni+ByNodes(col,si,variable)]+=
                cache.weight*derivative;
          }
        }
      }
    }
    __syncwarp(group_mask);
  }
}

__global__ void BorrowedEulerFaceKernel(CudaDeviceView v,CudaMeshShape s,
                                        bool frozen_llf)
{
  const int group=threadIdx.x/borrowed_euler_group_threads;
  const int lane=threadIdx.x%borrowed_euler_group_threads;
  const int warp_lane=threadIdx.x%borrowed_euler_warp_threads;
  const unsigned group_mask=borrowed_euler_group_mask<<
      (warp_lane-lane);
  const int e=blockIdx.x*borrowed_euler_elements_per_block+group;
  if(e>=s.elements||v.status[e]!=1.0)return;
  const int ni=s.interior_dofs,nt=s.trace_dofs;
  const int si=s.scalar_interior_dofs,sf=s.scalar_face_dofs;
  const int e_i=e*ni,e_t=e*nt;
  const std::size_t ao=static_cast<std::size_t>(e)*ni*ni;
  const std::size_t bo=static_cast<std::size_t>(e)*ni*nt;
  const std::size_t co=static_cast<std::size_t>(e)*nt*ni;
  const std::size_t doff=static_cast<std::size_t>(e)*nt*nt;
  __shared__ BorrowedFaceCache caches[borrowed_euler_elements_per_block];
  BorrowedFaceCache& cache=caches[group];
  for(int face=0;face<faces_per_element;++face)
    for(int q=0;q<s.face_points;++q)
    {
      const int qp=(e*faces_per_element+face)*s.face_points+q;
      const int to=face*s.trace_components*sf;
      if(lane==0)
      {
        cache.weight=v.face_weight[qp];cache.stabilization=s.igr_beta;
        cache.tau=0.0;cache.jump=0.0;cache.z_hat_normal=0.0;
        cache.valid=1;
      }
      // Lane c owns conserved interior/trace field c and its flux, exterior,
      // and Jacobian staging row.
      cache.interior[lane]=0.0;cache.trace[lane]=0.0;
      cache.flux[lane]=0.0;cache.exterior[lane]=0.0;
      for(int entry=0;entry<igr_trace_components;++entry)
        cache.jacobian[lane*igr_trace_components+entry]=0.0;
      for(int a=0;a<si;++a)
        cache.interior[lane]+=v.face_interior_shape[qp*si+a]*
            v.work_interior[e_i+ByNodes(a,si,lane)];
      for(int a=0;a<sf;++a)
        cache.trace[lane]+=v.face_trace_shape[qp*sf+a]*
            v.current_local_trace[e_t+to+ByNodes(a,sf,lane)];
      if(s.igr_active)
      {
        // Auxiliary ownership is lane 0:sigma, 1:sigma_hat, 2:z0, 3:z1.
        if(lane==0)
        {
          cache.sigma=0.0;
          for(int a=0;a<si;++a)
            cache.sigma+=v.face_interior_shape[qp*si+a]*
                v.work_interior[e_i+ByNodes(a,si,4)];
        }
        else if(lane==1)
        {
          cache.sigma_hat=0.0;
          for(int a=0;a<sf;++a)
            cache.sigma_hat+=v.face_trace_shape[qp*sf+a]*
                v.current_local_trace[e_t+to+ByNodes(a,sf,4)];
        }
        else
        {
          const int direction=lane-2;
          cache.z[direction]=0.0;
          for(int a=0;a<si;++a)
            cache.z[direction]+=v.face_interior_shape[qp*si+a]*
                v.work_interior[e_i+ByNodes(a,si,5+direction)];
        }
      }
      // Publish all component and auxiliary interpolants before lane zero
      // performs coupled flux, boundary, stabilization, and validity work.
      __syncwarp(group_mask);
      if(lane==0)
      {
        cache.normal[0]=v.face_normal[qp*dimension];
        cache.normal[1]=v.face_normal[qp*dimension+1];
        if(!s.igr_active)
        {
          if(frozen_llf)cache.stabilization=v.borrowed_lagged_llf_tau[qp];
          else
          {
            double interior_speed=0.0,trace_speed=0.0;
            if(!LocalLlfSpeed(cache.interior,cache.normal[0],cache.normal[1],
                              interior_speed)||
               !LocalLlfSpeed(cache.trace,cache.normal[0],cache.normal[1],
                              trace_speed))cache.valid=-2;
            else cache.stabilization=fmax(interior_speed,trace_speed);
          }
        }
        const bool flux_ready=cache.valid==1&&(s.igr_active?
            RegularizedFluxJacobian(cache.trace,cache.sigma_hat,
                cache.normal[0],cache.normal[1],cache.flux,cache.jacobian):
            FluxJacobian(cache.trace,cache.normal[0],cache.normal[1],
                cache.flux,cache.jacobian));
        if(cache.valid==1&&(!(cache.weight>0.0)||
            !(cache.stabilization>0.0)||!isfinite(cache.stabilization)||
            !flux_ready))cache.valid=-2;
        const int boundary_kind=s.borrowed_static_boundary_policies?
            v.boundary_policy_kind[e*faces_per_element+face]:0;
        cache.physical_boundary=boundary_kind!=0;
        if(cache.valid==1&&cache.physical_boundary&&
           !StaticBoundaryExterior(boundary_kind,
              v.boundary_reference+(e*faces_per_element+face)*
                  conserved_components,
              v.boundary_pressure[e*faces_per_element+face],s.boundary_gamma,
              cache.interior,cache.normal[0],cache.normal[1],cache.exterior))
          cache.valid=-3;
        if(cache.valid==1&&s.igr_active)
        {
          const double h=v.face_h[e*faces_per_element+face];
          const double degree_scale=static_cast<double>((s.order+1)*(s.order+1));
          if(!(cache.trace[0]>cuda_density_gate)||!(h>0.0)||!isfinite(h))
            cache.valid=-4;
          else
          {
            cache.tau=degree_scale/(cache.trace[0]*h);
            cache.jump=cache.sigma-cache.sigma_hat;
            cache.z_hat_normal=cache.z[0]*cache.normal[0]+
                cache.z[1]*cache.normal[1]-cache.tau*cache.jump;
            if(!(cache.tau>0.0)||!isfinite(cache.tau)||
               !isfinite(cache.z_hat_normal))cache.valid=-4;
          }
        }
      }
      __syncwarp(group_mask);
      if(cache.valid!=1)
      {if(lane==0)v.status[e]=static_cast<double>(cache.valid);return;}
      for(int interior_row=lane;interior_row<ni;
          interior_row+=borrowed_euler_group_threads)
      {
        const int component=interior_row/si;
        const int row=interior_row-component*si;
        const double rp=v.face_interior_shape[qp*si+row];
        if(component<conserved_components)
        {
          v.residual_r[e_i+interior_row]+=cache.weight*rp*
              (cache.flux[component]+cache.stabilization*
                  (cache.interior[component]-cache.trace[component]));
          for(int col=0;col<si;++col)
            v.a[ao+interior_row*ni+ByNodes(col,si,component)]+=
                cache.weight*rp*cache.stabilization*
                    v.face_interior_shape[qp*si+col];
          const int columns=s.igr_active?igr_trace_components:
              conserved_components;
          for(int col=0;col<sf;++col)for(int cc=0;cc<columns;++cc)
            v.b[bo+interior_row*nt+to+ByNodes(col,sf,cc)]+=
                cache.weight*rp*v.face_trace_shape[qp*sf+col]*
                (cache.jacobian[component*columns+cc]-
                 (component==cc?cache.stabilization:0.0));
        }
        else if(s.igr_active&&component<6)
        {
          const int direction=component-4;
          const double normal=cache.normal[direction];
          v.residual_r[e_i+interior_row]-=
              cache.weight*rp*normal*cache.sigma_hat;
          for(int col=0;col<sf;++col)
            v.b[bo+interior_row*nt+to+ByNodes(col,sf,4)]-=
                cache.weight*rp*normal*v.face_trace_shape[qp*sf+col];
        }
        else if(s.igr_active&&component==6)
        {
          v.residual_r[e_i+interior_row]-=
              s.igr_alpha*cache.weight*rp*cache.z_hat_normal;
          for(int col=0;col<si;++col)
          {
            const double cp=v.face_interior_shape[qp*si+col];
            v.a[ao+interior_row*ni+ByNodes(col,si,4)]+=
                s.igr_alpha*cache.weight*rp*cache.tau*cp;
            v.a[ao+interior_row*ni+ByNodes(col,si,5)]-=
                s.igr_alpha*cache.weight*rp*cache.normal[0]*cp;
            v.a[ao+interior_row*ni+ByNodes(col,si,6)]-=
                s.igr_alpha*cache.weight*rp*cache.normal[1]*cp;
          }
          for(int col=0;col<sf;++col)
          {
            const double cp=v.face_trace_shape[qp*sf+col];
            v.b[bo+interior_row*nt+to+ByNodes(col,sf,4)]-=
                s.igr_alpha*cache.weight*rp*cache.tau*cp;
            v.b[bo+interior_row*nt+to+ByNodes(col,sf,0)]-=
                s.igr_alpha*cache.weight*rp*cache.tau*cache.jump/
                    cache.trace[0]*cp;
          }
        }
      }
      const int face_trace_dofs=s.trace_components*sf;
      for(int local_trace_row=lane;local_trace_row<face_trace_dofs;
          local_trace_row+=borrowed_euler_group_threads)
      {
        const int component=local_trace_row/sf;
        const int row=local_trace_row-component*sf;
        const int trace_row=to+local_trace_row;
        const double rp=v.face_trace_shape[qp*sf+row];
        if(component<conserved_components)
        {
          v.residual_g[e_t+trace_row]+=cache.weight*rp*
              (cache.physical_boundary?
                  cache.trace[component]-cache.exterior[component]:
                  cache.flux[component]+cache.stabilization*
                      (cache.interior[component]-cache.trace[component]));
          if(!cache.physical_boundary)
            for(int col=0;col<si;++col)
              v.c[co+trace_row*ni+ByNodes(col,si,component)]+=
                  cache.weight*rp*cache.stabilization*
                      v.face_interior_shape[qp*si+col];
          const int columns=s.igr_active?igr_trace_components:
              conserved_components;
          for(int col=0;col<sf;++col)for(int cc=0;cc<columns;++cc)
            v.d[doff+trace_row*nt+to+ByNodes(col,sf,cc)]+=
                cache.weight*rp*v.face_trace_shape[qp*sf+col]*
                (cache.physical_boundary?(component==cc?1.0:0.0):
                 cache.jacobian[component*columns+cc]-
                    (component==cc?cache.stabilization:0.0));
        }
        else if(s.igr_active&&component==4)
        {
          v.residual_g[e_t+trace_row]+=
              s.igr_alpha*cache.weight*rp*cache.z_hat_normal;
          for(int col=0;col<si;++col)
          {
            const double cp=v.face_interior_shape[qp*si+col];
            v.c[co+trace_row*ni+ByNodes(col,si,4)]-=
                s.igr_alpha*cache.weight*rp*cache.tau*cp;
            v.c[co+trace_row*ni+ByNodes(col,si,5)]+=
                s.igr_alpha*cache.weight*rp*cache.normal[0]*cp;
            v.c[co+trace_row*ni+ByNodes(col,si,6)]+=
                s.igr_alpha*cache.weight*rp*cache.normal[1]*cp;
          }
          for(int col=0;col<sf;++col)
          {
            const double cp=v.face_trace_shape[qp*sf+col];
            v.d[doff+trace_row*nt+to+ByNodes(col,sf,4)]+=
                s.igr_alpha*cache.weight*rp*cache.tau*cp;
            v.d[doff+trace_row*nt+to+ByNodes(col,sf,0)]+=
                s.igr_alpha*cache.weight*rp*cache.tau*cache.jump/
                    cache.trace[0]*cp;
          }
        }
      }
      __syncwarp(group_mask);
    }
  for(int i=lane;i<ni;i+=borrowed_euler_group_threads)
    v.f[e_i+i]=-v.residual_r[e_i+i];
  for(int i=lane;i<nt;i+=borrowed_euler_group_threads)
    v.g[e_t+i]=-v.residual_g[e_t+i];
}

__global__ void CondenseKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=s.elements)return;
  const int ni=s.interior_dofs,nt=s.trace_dofs,cols=nt+1;
  const std::size_t ao=static_cast<std::size_t>(e)*ni*ni;
  const std::size_t bo=static_cast<std::size_t>(e)*ni*nt;
  const std::size_t co=static_cast<std::size_t>(e)*nt*ni;
  const std::size_t so=static_cast<std::size_t>(e)*nt*nt;
  double* lu=v.lu+ao; double* p=v.pivots+e*ni;
  double* w=v.solve_workspace+static_cast<std::size_t>(e)*ni*cols;
  for(int i=0;i<ni*ni;++i)lu[i]=v.a[ao+i];
  for(int r=0;r<ni;++r){for(int c=0;c<nt;++c)w[r*cols+c]=v.b[bo+r*nt+c];w[r*cols+nt]=v.f[e*ni+r];}
  if(!CudaFactorSolveRetained(lu,p,ni,w,cols)){v.status[e]=-1;return;}
  for(int r=0;r<nt;++r){double rhs=v.g[e*nt+r];for(int k=0;k<ni;++k)rhs-=v.c[co+r*ni+k]*w[k*cols+nt];v.condensed_rhs[e*nt+r]=rhs;for(int c=0;c<nt;++c){double x=v.d[so+r*nt+c];for(int k=0;k<ni;++k)x-=v.c[co+r*ni+k]*w[k*cols+c];v.schur[so+r*nt+c]=x;}}
  v.status[e]=1;
}

// Ordinary borrowed Euler uses one cooperative block per element.  The
// barriers retain the scalar factor/solve ordering for every matrix and RHS
// entry while distributing independent row, column, and RHS work across the
// block.  Legacy fixed-shape execution keeps CondenseKernel above unchanged.
__global__ void BorrowedCondenseKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x;if(e>=s.elements)return;
  const int ni=s.interior_dofs,nt=s.trace_dofs,cols=nt+1;
  const std::size_t ao=static_cast<std::size_t>(e)*ni*ni;
  const std::size_t bo=static_cast<std::size_t>(e)*ni*nt;
  const std::size_t co=static_cast<std::size_t>(e)*nt*ni;
  const std::size_t so=static_cast<std::size_t>(e)*nt*nt;
  double* lu=v.lu+ao;double* pivots=v.pivots+e*ni;
  double* w=v.solve_workspace+static_cast<std::size_t>(e)*ni*cols;
  __shared__ double scale;
  __shared__ int pivot_row;
  __shared__ int failed;
  for(int i=threadIdx.x;i<ni*ni;i+=blockDim.x)lu[i]=v.a[ao+i];
  for(int i=threadIdx.x;i<ni*cols;i+=blockDim.x)
  {
    const int row=i/cols,column=i-row*cols;
    w[i]=column<nt?v.b[bo+row*nt+column]:v.f[e*ni+row];
  }
  __syncthreads();
  if(threadIdx.x==0)
  {
    scale=0.0;failed=0;
    for(int i=0;i<ni*ni;++i)scale=fmax(scale,fabs(lu[i]));
  }
  __syncthreads();
  for(int column=0;column<ni;++column)
  {
    if(threadIdx.x==0)
    {
      pivot_row=column;
      for(int row=column+1;row<ni;++row)
        if(fabs(lu[row*ni+column])>fabs(lu[pivot_row*ni+column]))
          pivot_row=row;
      pivots[column]=pivot_row;
      if(!(fabs(lu[pivot_row*ni+column])>
            64.0*2.2204460492503131e-16*scale))failed=1;
    }
    __syncthreads();
    if(failed){if(threadIdx.x==0)v.status[e]=-1;return;}
    if(pivot_row!=column)
      for(int entry=threadIdx.x;entry<ni;entry+=blockDim.x)
      {
        const double value=lu[column*ni+entry];
        lu[column*ni+entry]=lu[pivot_row*ni+entry];
        lu[pivot_row*ni+entry]=value;
      }
    __syncthreads();
    for(int row=column+1+threadIdx.x;row<ni;row+=blockDim.x)
      lu[row*ni+column]/=lu[column*ni+column];
    __syncthreads();
    const int trailing=ni-column-1;
    for(int item=threadIdx.x;item<trailing*trailing;item+=blockDim.x)
    {
      const int row=column+1+item/trailing;
      const int entry=column+1+item%trailing;
      lu[row*ni+entry]-=lu[row*ni+column]*lu[column*ni+entry];
    }
    __syncthreads();
  }
  for(int row=0;row<ni;++row)
  {
    const int pivot=static_cast<int>(pivots[row]);
    if(pivot!=row)
      for(int rhs=threadIdx.x;rhs<cols;rhs+=blockDim.x)
      {
        const double value=w[row*cols+rhs];
        w[row*cols+rhs]=w[pivot*cols+rhs];
        w[pivot*cols+rhs]=value;
      }
    __syncthreads();
  }
  for(int column=0;column<ni;++column)
  {
    const int remaining=ni-column-1;
    for(int item=threadIdx.x;item<remaining*cols;item+=blockDim.x)
    {
      const int row=column+1+item/cols,rhs=item%cols;
      w[row*cols+rhs]-=lu[row*ni+column]*w[column*cols+rhs];
    }
    __syncthreads();
  }
  for(int row=ni-1;row>=0;--row)
  {
    for(int rhs=threadIdx.x;rhs<cols;rhs+=blockDim.x)
    {
      double value=w[row*cols+rhs];
      for(int column=row+1;column<ni;++column)
        value-=lu[row*ni+column]*w[column*cols+rhs];
      w[row*cols+rhs]=value/lu[row*ni+row];
    }
    __syncthreads();
  }
  for(int item=threadIdx.x;item<nt*(nt+1);item+=blockDim.x)
  {
    const int row=item/(nt+1),column=item-row*(nt+1);
    if(column==nt)
    {
      double value=v.g[e*nt+row];
      for(int k=0;k<ni;++k)value-=v.c[co+row*ni+k]*w[k*cols+nt];
      v.condensed_rhs[e*nt+row]=value;
    }
    else
    {
      double value=v.d[so+row*nt+column];
      for(int k=0;k<ni;++k)value-=v.c[co+row*ni+k]*w[k*cols+column];
      v.schur[so+row*nt+column]=value;
    }
  }
  __syncthreads();
  if(threadIdx.x==0)v.status[e]=1;
}

// The tensor leaf deliberately retains the ordinary FP64 pivoted LU and
// triangular solves.  Only the material C*A^{-1}*[B f] product is tiled onto
// FP64 tensor cores.  Per-warp shared tiles give every admitted p1-p3
// conventional/IGR shape an aligned m8n8k4 operand, with explicit zero fill
// for row, column, and inner-dimension tails.  Each warp owns a balanced,
// contiguous output-tile range and reuses one staged A operand across two
// adjacent, row-bounded output-column tiles.  The RHS shares a tensor tile when
// it occupies the Schur matrix's final partial column tile; otherwise its
// scalar contraction avoids launching a tensor tile with seven padded columns.
__global__ void TensorBorrowedCondenseKernel(CudaDeviceView v,CudaMeshShape s)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int tensor_warps_per_block=4;
  constexpr int tensor_column_batch_tiles=2;
  const int e=blockIdx.x;if(e>=s.elements)return;
  const int ni=s.interior_dofs,nt=s.trace_dofs,cols=nt+1;
  const std::size_t ao=static_cast<std::size_t>(e)*ni*ni;
  const std::size_t bo=static_cast<std::size_t>(e)*ni*nt;
  const std::size_t co=static_cast<std::size_t>(e)*nt*ni;
  const std::size_t so=static_cast<std::size_t>(e)*nt*nt;
  double* lu=v.lu+ao;double* pivots=v.pivots+e*ni;
  double* w=v.solve_workspace+static_cast<std::size_t>(e)*ni*cols;
  __shared__ double scale;
  __shared__ int pivot_row;
  __shared__ int failed;
  __shared__ __align__(32) double
      tensor_a[tensor_warps_per_block][8*4];
  __shared__ __align__(32) double
      tensor_b[tensor_warps_per_block][tensor_column_batch_tiles][4*8];
  __shared__ __align__(32) double
      tensor_product[tensor_warps_per_block][8*8];
  for(int i=threadIdx.x;i<ni*ni;i+=blockDim.x)lu[i]=v.a[ao+i];
  for(int i=threadIdx.x;i<ni*cols;i+=blockDim.x)
  {
    const int row=i/cols,column=i-row*cols;
    w[i]=column<nt?v.b[bo+row*nt+column]:v.f[e*ni+row];
  }
  __syncthreads();
  if(threadIdx.x==0)
  {
    scale=0.0;failed=0;
    for(int i=0;i<ni*ni;++i)scale=fmax(scale,fabs(lu[i]));
  }
  __syncthreads();
  for(int column=0;column<ni;++column)
  {
    if(threadIdx.x==0)
    {
      pivot_row=column;
      for(int row=column+1;row<ni;++row)
        if(fabs(lu[row*ni+column])>fabs(lu[pivot_row*ni+column]))
          pivot_row=row;
      pivots[column]=pivot_row;
      if(!(fabs(lu[pivot_row*ni+column])>
            64.0*2.2204460492503131e-16*scale))failed=1;
    }
    __syncthreads();
    if(failed){if(threadIdx.x==0)v.status[e]=-1;return;}
    if(pivot_row!=column)
      for(int entry=threadIdx.x;entry<ni;entry+=blockDim.x)
      {
        const double value=lu[column*ni+entry];
        lu[column*ni+entry]=lu[pivot_row*ni+entry];
        lu[pivot_row*ni+entry]=value;
      }
    __syncthreads();
    for(int row=column+1+threadIdx.x;row<ni;row+=blockDim.x)
      lu[row*ni+column]/=lu[column*ni+column];
    __syncthreads();
    const int trailing=ni-column-1;
    for(int item=threadIdx.x;item<trailing*trailing;item+=blockDim.x)
    {
      const int row=column+1+item/trailing;
      const int entry=column+1+item%trailing;
      lu[row*ni+entry]-=lu[row*ni+column]*lu[column*ni+entry];
    }
    __syncthreads();
  }
  for(int row=0;row<ni;++row)
  {
    const int pivot=static_cast<int>(pivots[row]);
    if(pivot!=row)
      for(int rhs=threadIdx.x;rhs<cols;rhs+=blockDim.x)
      {
        const double value=w[row*cols+rhs];
        w[row*cols+rhs]=w[pivot*cols+rhs];
        w[pivot*cols+rhs]=value;
      }
    __syncthreads();
  }
  for(int column=0;column<ni;++column)
  {
    const int remaining=ni-column-1;
    for(int item=threadIdx.x;item<remaining*cols;item+=blockDim.x)
    {
      const int row=column+1+item/cols,rhs=item%cols;
      w[row*cols+rhs]-=lu[row*ni+column]*w[column*cols+rhs];
    }
    __syncthreads();
  }
  for(int row=ni-1;row>=0;--row)
  {
    for(int rhs=threadIdx.x;rhs<cols;rhs+=blockDim.x)
    {
      double value=w[row*cols+rhs];
      for(int column=row+1;column<ni;++column)
        value-=lu[row*ni+column]*w[column*cols+rhs];
      w[row*cols+rhs]=value/lu[row*ni+row];
    }
    __syncthreads();
  }

  const int warp=threadIdx.x/warpSize,lane=threadIdx.x%warpSize;
  const int row_tiles=(nt+7)/8,column_tiles=(nt+7)/8;
  const bool tensorize_rhs=(cols+7)/8==column_tiles;
  const int output_tiles=row_tiles*column_tiles;
  const int first_tile=output_tiles*warp/tensor_warps_per_block;
  const int last_tile=output_tiles*(warp+1)/tensor_warps_per_block;
  for(int tile=first_tile;tile<last_tile;)
  {
    const int row_tile=tile/column_tiles;
    const int column_tile=tile-row_tile*column_tiles;
    int batch_width=tensor_column_batch_tiles;
    const int range_remaining=last_tile-tile;
    const int row_remaining=column_tiles-column_tile;
    if(batch_width>range_remaining)batch_width=range_remaining;
    if(batch_width>row_remaining)batch_width=row_remaining;
    const int row_base=row_tile*8;
    using TensorAccumulator=nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,8,8,4,double>;
    TensorAccumulator accumulators[tensor_column_batch_tiles];
#pragma unroll
    for(int slot=0;slot<tensor_column_batch_tiles;++slot)
      nvcuda::wmma::fill_fragment(accumulators[slot],0.0);
    for(int inner_base=0;inner_base<ni;inner_base+=4)
    {
      const int local_row=lane/4,local_inner=lane%4;
      const int row=row_base+local_row,inner=inner_base+local_inner;
      tensor_a[warp][lane]=row<nt&&inner<ni
          ?v.c[co+row*ni+inner]:0.0;
#pragma unroll
      for(int slot=0;slot<tensor_column_batch_tiles;++slot)
      {
        if(slot<batch_width)
        {
          // Row-major B keeps each solved-workspace row coalesced.
          const int b_local_inner=lane/8,b_local_column=lane%8;
          const int column=(column_tile+slot)*8+b_local_column;
          const int b_inner=inner_base+b_local_inner;
          tensor_b[warp][slot][lane]=column<cols&&b_inner<ni
              ?w[b_inner*cols+column]:0.0;
        }
      }
      __syncwarp();
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,8,8,4,double,
                            nvcuda::wmma::row_major> matrix_a;
      nvcuda::wmma::load_matrix_sync(matrix_a,tensor_a[warp],4);
#pragma unroll
      for(int slot=0;slot<tensor_column_batch_tiles;++slot)
      {
        if(slot<batch_width)
        {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,8,8,4,double,
                                nvcuda::wmma::row_major> matrix_b;
          nvcuda::wmma::load_matrix_sync(
              matrix_b,tensor_b[warp][slot],8);
          nvcuda::wmma::mma_sync(accumulators[slot],matrix_a,matrix_b,
                                 accumulators[slot]);
        }
      }
    }
#pragma unroll
    for(int slot=0;slot<tensor_column_batch_tiles;++slot)
    {
      if(slot<batch_width)
      {
        const int column_base=(column_tile+slot)*8;
        nvcuda::wmma::store_matrix_sync(
            tensor_product[warp],accumulators[slot],8,
            nvcuda::wmma::mem_row_major);
        __syncwarp();
        for(int item=lane;item<8*8;item+=warpSize)
        {
          const int row=row_base+item/8,column=column_base+item%8;
          if(row<nt&&column<cols)
          {
            const double product=tensor_product[warp][item];
            if(column==nt)
              v.condensed_rhs[e*nt+row]=v.g[e*nt+row]-product;
            else
              v.schur[so+row*nt+column]=
                  v.d[so+row*nt+column]-product;
          }
        }
        __syncwarp();
      }
    }
    tile+=batch_width;
  }
  if(!tensorize_rhs)
    for(int row=threadIdx.x;row<nt;row+=blockDim.x)
    {
      double value=v.g[e*nt+row];
      for(int k=0;k<ni;++k)
        value-=v.c[co+row*ni+k]*w[k*cols+nt];
      v.condensed_rhs[e*nt+row]=value;
    }
  __syncthreads();
  if(threadIdx.x==0)v.status[e]=1;
#else
  if(threadIdx.x==0&&blockIdx.x<s.elements)v.status[blockIdx.x]=-2;
#endif
}

__global__ void ZeroSystemKernel(CudaTrueMatrixDeviceView m,double* rhs,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m.diag_nonzeros)m.diag_values[i]=0;if(i<m.offd_nonzeros)m.offd_values[i]=0;if(i<n)rhs[i]=0;}

__device__ double SignedMatrixValue(CudaDeviceView v,CudaMeshShape s,
                                    const int source)
{const int stride=s.trace_dofs,e=source/(stride*stride),rem=source-e*stride*stride,r=rem/stride,c=rem-r*stride;double rs,cs;Decode(v.signed_element_trace[e*stride+r],rs);Decode(v.signed_element_trace[e*stride+c],cs);return rs*cs*v.schur[source];}
__device__ double SignedVectorValue(CudaDeviceView v,CudaMeshShape s,
                                    const double* values,const int source)
{const int stride=s.trace_dofs,e=source/stride,r=source-e*stride;double sign;Decode(v.signed_element_trace[e*stride+r],sign);return sign*values[source];}

__global__ void OwnedMatrixKernel(CudaDeviceView v,CudaTrueMatrixDeviceView m,CudaMeshShape s,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double value=SignedMatrixValue(v,s,v.matrix_owned_source[i]);if(v.matrix_owned_section[i]==0)atomicAdd(m.diag_values+v.matrix_owned_target[i],value);else atomicAdd(m.offd_values+v.matrix_owned_target[i],value);}
__global__ void GatherRemoteMatrixKernel(CudaDeviceView v,CudaMeshShape s,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)v.remote_matrix_send[i]=SignedMatrixValue(v,s,v.matrix_send_source[i]);}
__global__ void ScatterRemoteMatrixKernel(CudaDeviceView v,CudaTrueMatrixDeviceView m,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;if(v.matrix_receive_section[i]==0)atomicAdd(m.diag_values+v.matrix_receive_target[i],v.remote_matrix_receive[i]);else atomicAdd(m.offd_values+v.matrix_receive_target[i],v.remote_matrix_receive[i]);}
__global__ void OwnedVectorKernel(CudaDeviceView v,CudaMeshShape s,const double* values,double* truth,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(truth+v.vector_owned_target[i],SignedVectorValue(v,s,values,v.vector_owned_source[i]));}
__global__ void ZeroVectorKernel(double* values,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)values[i]=0;}
__global__ void StatusReductionKernel(const double* status,int count,
                                      int* failure)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count&&status[i]!=1.0)atomicExch(failure,1);}
__global__ void GatherRemoteVectorKernel(CudaDeviceView v,CudaMeshShape s,const double* values,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)v.remote_vector_send[i]=SignedVectorValue(v,s,values,v.vector_send_source[i]);}
__global__ void ScatterRemoteVectorKernel(CudaDeviceView v,double* truth,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(truth+v.vector_receive_target[i],v.remote_vector_receive[i]);}

__global__ void RecoverKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=s.elements)return;
  const int ni=s.interior_dofs,nt=s.trace_dofs;
  CudaRecoverRetained(v.lu+e*ni*ni,v.pivots+e*ni,ni,
                      v.b+e*ni*nt,nt,v.trace_increment_local+e*nt,
                      v.f+e*ni,v.recovered_interior+e*ni);
  v.status[e]=1;
}
__global__ void BorrowedRecoverKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x;if(e>=s.elements)return;
  const int ni=s.interior_dofs,nt=s.trace_dofs;
  const double* lu=v.lu+static_cast<std::size_t>(e)*ni*ni;
  const double* pivots=v.pivots+e*ni;
  double* recovered=v.recovered_interior+e*ni;
  for(int row=threadIdx.x;row<ni;row+=blockDim.x)
  {
    double value=v.f[e*ni+row];
    for(int column=0;column<nt;++column)
      value-=v.b[(static_cast<std::size_t>(e)*ni+row)*nt+column]*
          v.trace_increment_local[e*nt+column];
    recovered[row]=value;
  }
  __syncthreads();
  for(int row=0;row<ni;++row)
  {
    if(threadIdx.x==0)
    {
      const int pivot=static_cast<int>(pivots[row]);
      if(pivot!=row)
      {
        const double value=recovered[row];
        recovered[row]=recovered[pivot];recovered[pivot]=value;
      }
    }
    __syncthreads();
  }
  for(int column=0;column<ni;++column)
  {
    for(int row=column+1+threadIdx.x;row<ni;row+=blockDim.x)
      recovered[row]-=lu[row*ni+column]*recovered[column];
    __syncthreads();
  }
  for(int row=ni-1;row>=0;--row)
  {
    if(threadIdx.x==0)
    {
      double value=recovered[row];
      for(int column=row+1;column<ni;++column)
        value-=lu[row*ni+column]*recovered[column];
      recovered[row]=value/lu[row*ni+row];
    }
    __syncthreads();
  }
  if(threadIdx.x==0)v.status[e]=1;
}
__global__ void TrialKernel(CudaDeviceView v,CudaMeshShape s,double alpha)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<s.interior_count)v.work_interior[i]=v.base_interior[i]+alpha*v.recovered_interior[i];if(i<s.true_trace_count)v.work_true_trace[i]=v.base_true_trace[i]+alpha*v.true_solution[i];}

__global__ void PackInteriorOutputKernel(CudaMeshShape s,
                                         const double* source,
                                         double* packed,
                                         int output_components)
{
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  const int per_element=output_components*s.scalar_interior_dofs;
  const int count=s.elements*per_element;
  if(index>=count)return;
  const int element=index/per_element;
  const int local=index-element*per_element;
  packed[index]=source[static_cast<std::size_t>(element)*s.interior_dofs+
                       local];
}

inline constexpr int cooperative_reduction_threads = 256;
inline constexpr int cooperative_wave_threads = 128;
inline constexpr int cooperative_conservation_threads = 128;

__global__ void NormPartialsKernel(const double* a,int an,const double* b,
                                   int bn,double* partials)
{
  __shared__ double partial[cooperative_reduction_threads];
  const int thread=threadIdx.x;
  const int count=an+bn;
  double sum=0.0;
  // Treat the two inputs as one ordered sequence.  Each global lane follows a
  // fixed grid-stride subsequence, and each block emits exactly one pairwise-
  // reduced partial without atomics.
  const int global_thread=blockIdx.x*blockDim.x+thread;
  const int stride=gridDim.x*blockDim.x;
  for(int index=global_thread;index<count;index+=stride)
  {
    const double value=index<an?a[index]:b[index-an];
    sum+=value*value;
  }
  partial[thread]=sum;
  __syncthreads();
  for(int offset=blockDim.x/2;offset>0;offset/=2)
  {
    if(thread<offset)partial[thread]+=partial[thread+offset];
    __syncthreads();
  }
  if(thread==0)partials[blockIdx.x]=partial[0];
}

__global__ void NormReduceKernel(const double* partials,int partial_count,
                                 double* out)
{
  __shared__ double partial[cooperative_reduction_threads];
  const int thread=threadIdx.x;
  double sum=0.0;
  for(int index=thread;index<partial_count;index+=blockDim.x)
    sum+=partials[index];
  partial[thread]=sum;
  __syncthreads();
  for(int offset=blockDim.x/2;offset>0;offset/=2)
  {
    if(thread<offset)partial[thread]+=partial[thread+offset];
    __syncthreads();
  }
  if(thread==0)out[0]=partial[0];
}
__device__ double EulerWave(const double rho,const double mx,const double my,
                            const double energy)
{if(!(rho>0.0)||!isfinite(rho)||!isfinite(mx)||!isfinite(my)||!isfinite(energy))return nan("");const double u=mx/rho,v=my/rho;const double p=0.4*(energy-0.5*(mx*mx+my*my)/rho);return p>0.0&&isfinite(p)?sqrt(u*u+v*v)+sqrt(1.4*p/rho):nan("");}
__global__ void MaximumWaveElementsKernel(CudaDeviceView v,CudaMeshShape s,
                                          const double* interior,
                                          const double* trace)
{
  __shared__ double wave[cooperative_wave_threads];
  const int e=blockIdx.x;if(e>=s.elements)return;
  const int thread=threadIdx.x;
  const int ib=e*s.interior_dofs,tb=e*s.trace_dofs;
  const int face_sites=faces_per_element*s.face_points;
  const int sites=s.volume_points+2*face_sites;
  double value=0.0;
  if(thread<s.volume_points)
  {
    const int qp=e*s.volume_points+thread;
    double state[conserved_components]={};
    for(int scalar=0;scalar<s.scalar_interior_dofs;++scalar)
      for(int component=0;component<conserved_components;++component)
        state[component]+=v.volume_shape[
            qp*s.scalar_interior_dofs+scalar]*
            interior[ib+ByNodes(scalar,s.scalar_interior_dofs,component)];
    value=EulerWave(state[0],state[1],state[2],state[3]);
  }
  else if(thread<sites)
  {
    const int face_value=thread-s.volume_points;
    const int face_site=face_value/2;
    const bool trace_site=(face_value&1)!=0;
    const int face=face_site/s.face_points;
    const int q=face_site-face*s.face_points;
    const int qp=(e*faces_per_element+face)*s.face_points+q;
    const int face_offset=face*s.trace_components*s.scalar_face_dofs;
    double state[conserved_components]={};
    if(trace_site)
    {
      for(int scalar=0;scalar<s.scalar_face_dofs;++scalar)
        for(int component=0;component<conserved_components;++component)
          state[component]+=v.face_trace_shape[
              qp*s.scalar_face_dofs+scalar]*
              trace[tb+face_offset+ByNodes(
                  scalar,s.scalar_face_dofs,component)];
    }
    else
    {
      for(int scalar=0;scalar<s.scalar_interior_dofs;++scalar)
        for(int component=0;component<conserved_components;++component)
          state[component]+=v.face_interior_shape[
              qp*s.scalar_interior_dofs+scalar]*
              interior[ib+ByNodes(
                  scalar,s.scalar_interior_dofs,component)];
    }
    value=EulerWave(state[0],state[1],state[2],state[3]);
  }
  // Valid wave speeds are nonnegative.  Convert every invalid site to the
  // existing finite sentinel before the fixed tree so validity propagation is
  // independent of NaN comparison behavior.
  wave[thread]=thread<sites&&(!isfinite(value)||value<0.0)?-1.0:value;
  __syncthreads();
  for(int offset=cooperative_wave_threads/2;offset>0;offset/=2)
  {
    if(thread<offset)
    {
      const double left=wave[thread],right=wave[thread+offset];
      wave[thread]=(left<0.0||right<0.0)?-1.0:fmax(left,right);
    }
    __syncthreads();
  }
  // Valid Euler wave speeds are nonnegative, so -1 is an unambiguous finite
  // sentinel and avoids relying on NaN behavior in the ordered reducer.
  if(thread==0)v.status[e]=wave[0];
}
__global__ void MaximumWaveReduceKernel(const double* element_maxima,
                                        int elements,double* out)
{
  __shared__ double maximums[cooperative_reduction_threads];
  __shared__ int validity[cooperative_reduction_threads];
  const int thread=threadIdx.x;
  double maximum=0.0;
  int valid=1;
  for(int e=thread;e<elements;e+=blockDim.x)
  {
    const double value=element_maxima[e];
    if(!isfinite(value)||value<0.0)valid=0;
    else maximum=fmax(maximum,value);
  }
  maximums[thread]=maximum;
  validity[thread]=valid;
  __syncthreads();
  for(int offset=blockDim.x/2;offset>0;offset/=2)
  {
    if(thread<offset)
    {
      maximums[thread]=fmax(maximums[thread],maximums[thread+offset]);
      validity[thread]=validity[thread]&&validity[thread+offset];
    }
    __syncthreads();
  }
  if(thread==0)
  {
    out[0]=maximums[0];
    out[1]=validity[0]?1.0:0.0;
  }
}

struct EulerConservationSite
{
  double balance[igr_trace_components];
  double scale[igr_trace_components];
  double measure;
};

__device__ __forceinline__ bool EulerConservationVolumeSite(
    CudaDeviceView v, CudaMeshShape s, const int element, const int q,
    const double dt, EulerConservationSite& site)
{
  const int interior_offset = element * s.interior_dofs;
  const int qp = element * s.volume_points + q;
  const double weight = v.volume_weight[qp];
  double current[conserved_components] = {};
  double previous[conserved_components] = {};
  double source[conserved_components] = {};
  double sigma = 0.0;
  double conservative_gradient[3][dimension] = {};
  for (int scalar = 0; scalar < s.scalar_interior_dofs; ++scalar)
  {
    const double phi =
        v.volume_shape[qp * s.scalar_interior_dofs + scalar];
    for (int component = 0; component < conserved_components; ++component)
    {
      const int index = interior_offset +
          ByNodes(scalar, s.scalar_interior_dofs, component);
      current[component] += phi * v.work_interior[index];
      previous[component] += phi * v.old_interior[index];
      source[component] += phi * v.static_source[index];
    }
    if (s.igr_active)
    {
      sigma += phi * v.work_interior[interior_offset +
          ByNodes(scalar, s.scalar_interior_dofs, 4)];
      for (int component = 0; component < 3; ++component)
        for (int direction = 0; direction < dimension; ++direction)
          conservative_gradient[component][direction] +=
              v.volume_derivative[
                  (qp * s.scalar_interior_dofs + scalar) * dimension +
                  direction] *
              v.work_interior[interior_offset +
                  ByNodes(scalar, s.scalar_interior_dofs, component)];
    }
  }
  if (!(weight > 0.0) || !isfinite(weight) ||
      (s.igr_active &&
       (!(current[0] > cuda_density_gate) || !isfinite(sigma))))
    return false;
  site.measure = weight;
  for (int component = 0; component < conserved_components; ++component)
  {
    const double change =
        weight * (current[component] - previous[component]);
    const double forcing = dt * weight * source[component];
    site.balance[component] = change - forcing;
    site.scale[component] = fabs(change) + fabs(forcing);
    if (!isfinite(change) || !isfinite(forcing) ||
        !isfinite(site.balance[component]) ||
        !isfinite(site.scale[component]))
      return false;
  }
  if (s.igr_active)
  {
    double velocity_gradient[dimension][dimension] = {};
    for (int velocity = 0; velocity < dimension; ++velocity)
      for (int direction = 0; direction < dimension; ++direction)
        velocity_gradient[velocity][direction] =
            conservative_gradient[1 + velocity][direction] / current[0] -
            current[1 + velocity] *
                conservative_gradient[0][direction] /
                (current[0] * current[0]);
    const double velocity_trace =
        velocity_gradient[0][0] + velocity_gradient[1][1];
    const double q_value =
        velocity_gradient[0][0] * velocity_gradient[0][0] +
        2.0 * velocity_gradient[0][1] * velocity_gradient[1][0] +
        velocity_gradient[1][1] * velocity_gradient[1][1] +
        velocity_trace * velocity_trace;
    const double sigma_contribution = weight * sigma / current[0];
    const double q_contribution = weight * s.igr_alpha * q_value;
    site.balance[4] = sigma_contribution - q_contribution;
    site.scale[4] = fabs(sigma_contribution) + fabs(q_contribution);
    if (!isfinite(q_value) || !isfinite(sigma_contribution) ||
        !isfinite(q_contribution) || !isfinite(site.balance[4]) ||
        !isfinite(site.scale[4]))
      return false;
  }
  return true;
}

__device__ __forceinline__ bool EulerConservationFaceSite(
    CudaDeviceView v, CudaMeshShape s, const int element, const int face,
    const int q, const double dt, EulerConservationSite& site)
{
  const int element_face = element * faces_per_element + face;
  const bool physical = v.boundary_policy_kind != nullptr &&
      v.boundary_policy_kind[element_face] != 0;
  if (!physical) return true;
  const double h = s.igr_active ? v.face_h[element_face] : 1.0;
  if (s.igr_active && (!(h > 0.0) || !isfinite(h))) return false;
  const int qp = element_face * s.face_points + q;
  const double weight = v.face_weight[qp];
  const double normal[dimension] = {
      v.face_normal[qp * dimension],
      v.face_normal[qp * dimension + 1]};
  const int interior_offset = element * s.interior_dofs;
  const int trace_offset = element * s.trace_dofs;
  const int face_offset =
      face * s.trace_components * s.scalar_face_dofs;
  double interior[conserved_components] = {};
  double trace[conserved_components] = {};
  double sigma = 0.0;
  double sigma_hat = 0.0;
  double z[dimension] = {};
  for (int scalar = 0; scalar < s.scalar_interior_dofs; ++scalar)
  {
    const double phi =
        v.face_interior_shape[qp * s.scalar_interior_dofs + scalar];
    for (int component = 0; component < conserved_components; ++component)
      interior[component] += phi * v.work_interior[interior_offset +
          ByNodes(scalar, s.scalar_interior_dofs, component)];
    if (s.igr_active)
    {
      sigma += phi * v.work_interior[interior_offset +
          ByNodes(scalar, s.scalar_interior_dofs, 4)];
      z[0] += phi * v.work_interior[interior_offset +
          ByNodes(scalar, s.scalar_interior_dofs, 5)];
      z[1] += phi * v.work_interior[interior_offset +
          ByNodes(scalar, s.scalar_interior_dofs, 6)];
    }
  }
  for (int scalar = 0; scalar < s.scalar_face_dofs; ++scalar)
  {
    const double phi =
        v.face_trace_shape[qp * s.scalar_face_dofs + scalar];
    for (int component = 0; component < conserved_components; ++component)
      trace[component] += phi * v.current_local_trace[
          trace_offset + face_offset +
          ByNodes(scalar, s.scalar_face_dofs, component)];
    if (s.igr_active)
      sigma_hat += phi * v.current_local_trace[
          trace_offset + face_offset +
          ByNodes(scalar, s.scalar_face_dofs, 4)];
  }
  double flux[conserved_components] = {};
  double jacobian[conserved_components * igr_trace_components] = {};
  if (!(weight > 0.0) || !isfinite(weight) ||
      !(s.igr_active
            ? RegularizedFluxJacobian(
                  trace, sigma_hat, normal[0], normal[1], flux, jacobian)
            : EulerNormalFlux(trace, normal[0], normal[1], flux)))
    return false;
  double stabilization = s.igr_beta;
  if (!s.igr_active)
  {
    double interior_speed = 0.0;
    double trace_speed = 0.0;
    if (!LocalLlfSpeed(
            interior, normal[0], normal[1], interior_speed) ||
        !LocalLlfSpeed(trace, normal[0], normal[1], trace_speed))
      return false;
    stabilization = fmax(interior_speed, trace_speed);
    if (!(stabilization > 0.0) || !isfinite(stabilization)) return false;
  }
  for (int component = 0; component < conserved_components; ++component)
  {
    const double numerical_flux = flux[component] + stabilization *
        (interior[component] - trace[component]);
    const double contribution = dt * weight * numerical_flux;
    site.balance[component] = contribution;
    site.scale[component] = fabs(contribution);
    if (!isfinite(contribution)) return false;
  }
  if (s.igr_active)
  {
    if (!(trace[0] > cuda_density_gate)) return false;
    const double degree_scale = static_cast<double>(
        (s.order + 1) * (s.order + 1));
    const double tau = degree_scale / (trace[0] * h);
    const double z_hat_normal =
        z[0] * normal[0] + z[1] * normal[1] -
        tau * (sigma - sigma_hat);
    const double contribution = s.igr_alpha * weight * z_hat_normal;
    site.balance[4] = -contribution;
    site.scale[4] = fabs(contribution);
    if (!(tau > 0.0) || !isfinite(tau) || !isfinite(contribution))
      return false;
  }
  return true;
}

// Compact post-candidate balance used by the production borrowed-Euler
// lifecycle.  One block owns one element, and its lanes map the ordered site
// sequence [volume q][face 0 q]...[face 3 q].  Nonphysical face sites are
// exact zeros.  The fixed pairwise tree is deterministic and uses no atomics.
// The first twelve entries of each element's f vector are temporary scratch;
// the second stage restores their exact assembly invariant f=-residual_r.
__global__ void EulerConservationElementsKernel(
    CudaDeviceView v, CudaMeshShape s, const double dt)
{
  constexpr int components = igr_trace_components;
  static_assert(euler_conservation_result_scalars == 2 * components + 2);
  __shared__ double balance[components][cooperative_conservation_threads];
  __shared__ double scale[components][cooperative_conservation_threads];
  __shared__ double measure[cooperative_conservation_threads];
  __shared__ int validity[cooperative_conservation_threads];
  const int element = blockIdx.x;
  const int thread = threadIdx.x;
  const int face_sites = faces_per_element * s.face_points;
  const int sites = s.volume_points + face_sites;
  EulerConservationSite site{};
  bool valid = s.borrowed_host_inputs && isfinite(dt) && dt > 0.0 &&
      (!s.igr_active || v.face_h != nullptr);
  if (valid && thread < s.volume_points)
    valid = EulerConservationVolumeSite(v, s, element, thread, dt, site);
  else if (valid && thread < sites)
  {
    const int face_site = thread - s.volume_points;
    const int face = face_site / s.face_points;
    const int q = face_site - face * s.face_points;
    valid = EulerConservationFaceSite(v, s, element, face, q, dt, site);
  }
  for (int component = 0; component < components; ++component)
  {
    balance[component][thread] = valid ? site.balance[component] : 0.0;
    scale[component][thread] = valid ? site.scale[component] : 0.0;
  }
  measure[thread] = valid ? site.measure : 0.0;
  validity[thread] = valid ? 1 : 0;
  __syncthreads();
  for (int offset = cooperative_conservation_threads / 2;
       offset > 0; offset /= 2)
  {
    if (thread < offset)
    {
      for (int component = 0; component < components; ++component)
      {
        balance[component][thread] +=
            balance[component][thread + offset];
        scale[component][thread] += scale[component][thread + offset];
      }
      measure[thread] += measure[thread + offset];
      validity[thread] =
          validity[thread] && validity[thread + offset];
    }
    __syncthreads();
  }
  if (thread == 0)
  {
    bool element_valid = validity[0] != 0 && isfinite(measure[0]);
    for (int component = 0; component < components; ++component)
      element_valid = element_valid && isfinite(balance[component][0]) &&
          isfinite(scale[component][0]);
    const int scratch = element * s.interior_dofs;
    for (int component = 0; component < components; ++component)
    {
      v.f[scratch + component] =
          element_valid ? balance[component][0] : 0.0;
      v.f[scratch + components + component] =
          element_valid ? scale[component][0] : 0.0;
    }
    v.f[scratch + 2 * components] = element_valid ? measure[0] : 0.0;
    v.f[scratch + 2 * components + 1] = element_valid ? 1.0 : 0.0;
  }
}

__global__ void EulerConservationReduceKernel(
    CudaDeviceView v, CudaMeshShape s, double* out)
{
  constexpr int components = igr_trace_components;
  __shared__ double balance[components][cooperative_conservation_threads];
  __shared__ double scale[components][cooperative_conservation_threads];
  __shared__ double measure[cooperative_conservation_threads];
  __shared__ int validity[cooperative_conservation_threads];
  const int thread = threadIdx.x;
  EulerConservationSite total{};
  bool valid = true;
  for (int element = thread; element < s.elements;
       element += blockDim.x)
  {
    const int scratch = element * s.interior_dofs;
    valid = valid &&
        v.f[scratch + euler_conservation_result_scalars - 1] == 1.0;
    for (int component = 0; component < components; ++component)
    {
      const double element_balance = v.f[scratch + component];
      const double element_scale = v.f[scratch + components + component];
      total.balance[component] += element_balance;
      total.scale[component] += element_scale;
      valid = valid && isfinite(element_balance) &&
          isfinite(element_scale) && isfinite(total.balance[component]) &&
          isfinite(total.scale[component]);
    }
    const double element_measure = v.f[scratch + 2 * components];
    total.measure += element_measure;
    valid = valid && isfinite(element_measure) && isfinite(total.measure);
    // Conservation is called only after coherent assembly/recovery, where the
    // retained local RHS has the exact invariant f=-residual_r.  Restore every
    // overwritten entry so later oracle export observes the original product.
    for (int index = 0; index < euler_conservation_result_scalars; ++index)
      v.f[scratch + index] = -v.residual_r[scratch + index];
  }
  for (int component = 0; component < components; ++component)
  {
    balance[component][thread] = total.balance[component];
    scale[component][thread] = total.scale[component];
  }
  measure[thread] = total.measure;
  validity[thread] = valid ? 1 : 0;
  __syncthreads();
  for (int offset = cooperative_conservation_threads / 2;
       offset > 0; offset /= 2)
  {
    if (thread < offset)
    {
      for (int component = 0; component < components; ++component)
      {
        balance[component][thread] +=
            balance[component][thread + offset];
        scale[component][thread] += scale[component][thread + offset];
      }
      measure[thread] += measure[thread + offset];
      validity[thread] =
          validity[thread] && validity[thread + offset];
    }
    __syncthreads();
  }
  if (thread == 0)
  {
    bool final_valid = validity[0] != 0 && isfinite(measure[0]);
    for (int component = 0; component < components; ++component)
      final_valid = final_valid && isfinite(balance[component][0]) &&
          isfinite(scale[component][0]);
    for (int component = 0; component < components; ++component)
    {
      out[component] = final_valid ? balance[component][0] : 0.0;
      out[components + component] =
          final_valid ? scale[component][0] : 0.0;
    }
    out[2 * components] = final_valid ? measure[0] : 0.0;
    out[2 * components + 1] = final_valid ? 1.0 : 0.0;
  }
}
__device__ std::uint64_t HashBytesLane(const double* values,const int count,
                                      const int lane)
{const std::uint64_t seeds[4]={1469598103934665603ULL,1099511628211ULL^0x9e3779b97f4a7c15ULL,1469598103934665603ULL^0xd6e8feb86659fd93ULL,1099511628211ULL^0xa0761d6478bd642fULL};std::uint64_t h=seeds[lane];const unsigned char* bytes=reinterpret_cast<const unsigned char*>(values);for(int i=0;i<count*static_cast<int>(sizeof(double));++i){h^=static_cast<std::uint64_t>(bytes[i]+17U*lane);h*=1099511628211ULL+2ULL*lane;}return h;}
__global__ void HashKernel(const double* values,int count,std::uint64_t* out)
{const int lane=threadIdx.x;if(blockIdx.x||lane>=4)return;out[lane]=HashBytesLane(values,count,lane);}
__global__ void HashPairKernel(const double* first,int first_count,
                               const double* second,int second_count,
                               std::uint64_t* out)
{const int lane=threadIdx.x;if(blockIdx.x||lane>=8)return;const bool use_second=lane>=4;const int hash_lane=lane&3;out[lane]=HashBytesLane(use_second?second:first,use_second?second_count:first_count,hash_lane);}

bool LaunchOk() noexcept { return cudaPeekAtLastError() == cudaSuccess; }
int Blocks(const int n) noexcept { return (n + 127) / 128; }
bool LaunchBorrowedEulerStages(const CudaDeviceView& v,
                               const CudaMeshShape& s,
                               const double dt,
                               const bool frozen_llf) noexcept
{
  const int blocks=(s.elements-1)/borrowed_euler_elements_per_block+1;
  BorrowedEulerZeroKernel<<<blocks,borrowed_euler_block_threads>>>(v,s);
  if(!LaunchOk())return false;
  BorrowedEulerVolumeKernel<<<blocks,borrowed_euler_block_threads>>>(v,s,dt);
  if(!LaunchOk())return false;
  BorrowedEulerFaceKernel<<<blocks,borrowed_euler_block_threads>>>(
      v,s,frozen_llf);
  return LaunchOk();
}
} // namespace

bool CudaLaunchTicket(int* words,const CudaFailurePhase phase,std::uint64_t attempt,std::uint64_t generation,int device) noexcept
{TicketKernel<<<1,1>>>(words,static_cast<int>(phase),attempt,generation,device);return LaunchOk();}
bool CudaLaunchStatusReduction(const double* status,const int count,
                               int* failure_flag) noexcept
{if(status==nullptr||failure_flag==nullptr||count<0)return false;if(cudaMemsetAsync(failure_flag,0,sizeof(int),nullptr)!=cudaSuccess)return false;if(count)StatusReductionKernel<<<Blocks(count),128>>>(status,count,failure_flag);return LaunchOk();}
bool CudaLaunchGatherLocalTrace(const int* map,int count,const double* source,double* destination) noexcept
{GatherKernel<<<Blocks(count),128>>>(map,count,source,destination);return LaunchOk();}
bool CudaLaunchAdmissibility(const CudaDeviceView& v,const CudaMeshShape& s) noexcept
{AdmissibilityKernel<<<s.elements,128>>>(v,s);return LaunchOk();}
bool CudaLaunchFreezeBorrowedCurrentLlfTau(const CudaDeviceView& v,
                                           const CudaMeshShape& s) noexcept
{FreezeBorrowedCurrentLlfTauKernel<<<s.elements,128>>>(v,s);return LaunchOk();}
bool CudaLaunchAssembly(const CudaDeviceView& v,const CudaMeshShape& s,double dt) noexcept
{
  if(!s.borrowed_host_inputs)
  {AssemblyKernel<<<Blocks(s.elements),128>>>(v,s,dt);return LaunchOk();}
  if(s.elements<=0)return false;
  return LaunchBorrowedEulerStages(v,s,dt,false);
}
bool CudaLaunchFrozenBorrowedLlfAssembly(const CudaDeviceView& v,
                                         const CudaMeshShape& s,
                                         double dt) noexcept
{
  if(!s.borrowed_host_inputs||s.igr_active||s.elements<=0)return false;
  return LaunchBorrowedEulerStages(v,s,dt,true);
}
bool CudaLaunchCondensation(const CudaDeviceView& v,const CudaMeshShape& s,
                            const generic::CudaEngineKind engine,
                            std::uint64_t& tensor_mma_tiles) noexcept
{
  tensor_mma_tiles=0;
  if(engine==generic::CudaEngineKind::ordinary)
  {
    if(s.borrowed_host_inputs)BorrowedCondenseKernel<<<s.elements,128>>>(v,s);
    else CondenseKernel<<<Blocks(s.elements),128>>>(v,s);
    return LaunchOk();
  }
  if(engine!=generic::CudaEngineKind::tensor_core||
     !s.borrowed_host_inputs||s.elements<=0||s.interior_dofs<=0||
     s.trace_dofs<=0||
     s.interior_dofs>generic_maximum_interior_dofs_per_element||
     s.trace_dofs>generic_maximum_trace_dofs_per_element)
    return false;
  const std::uint64_t row_tiles=
      static_cast<std::uint64_t>((s.trace_dofs+7)/8);
  const std::uint64_t column_tiles=
      static_cast<std::uint64_t>((s.trace_dofs+7)/8);
  const std::uint64_t inner_tiles=
      static_cast<std::uint64_t>((s.interior_dofs+3)/4);
  const std::uint64_t elements=static_cast<std::uint64_t>(s.elements);
  if(column_tiles!=0&&
     row_tiles>std::numeric_limits<std::uint64_t>::max()/column_tiles)
    return false;
  const std::uint64_t output_tiles=row_tiles*column_tiles;
  if(inner_tiles!=0&&
     output_tiles>std::numeric_limits<std::uint64_t>::max()/inner_tiles)
    return false;
  const std::uint64_t per_element=output_tiles*inner_tiles;
  if(elements!=0&&
     per_element>std::numeric_limits<std::uint64_t>::max()/elements)
    return false;
  TensorBorrowedCondenseKernel<<<s.elements,128>>>(v,s);
  if(!LaunchOk())return false;
  tensor_mma_tiles=per_element*elements;
  return true;
}
bool CudaLaunchZeroTrueSystem(const CudaTrueMatrixDeviceView& m,double* rhs,int n) noexcept
{const int count=std::max(std::max(m.diag_nonzeros,m.offd_nonzeros),n);ZeroSystemKernel<<<Blocks(count),128>>>(m,rhs,n);return LaunchOk();}
bool CudaLaunchAssembleOwnedMatrix(const CudaDeviceView& v,const CudaTrueMatrixDeviceView& m,const CudaMeshShape& s) noexcept
{OwnedMatrixKernel<<<Blocks(s.matrix_owned_count),128>>>(v,m,s,s.matrix_owned_count);return LaunchOk();}
bool CudaLaunchGatherRemoteMatrix(const CudaDeviceView& v,const CudaMeshShape& s) noexcept
{if(s.matrix_send_count)GatherRemoteMatrixKernel<<<Blocks(s.matrix_send_count),128>>>(v,s,s.matrix_send_count);return LaunchOk();}
bool CudaLaunchScatterRemoteMatrix(const CudaDeviceView& v,const CudaTrueMatrixDeviceView& m,const CudaMeshShape& s) noexcept
{if(s.matrix_receive_count)ScatterRemoteMatrixKernel<<<Blocks(s.matrix_receive_count),128>>>(v,m,s.matrix_receive_count);return LaunchOk();}
bool CudaLaunchAssembleOwnedVector(const CudaDeviceView& v,const CudaMeshShape& s,const double* values,double* truth) noexcept
{ZeroVectorKernel<<<Blocks(s.true_trace_count),128>>>(truth,s.true_trace_count);OwnedVectorKernel<<<Blocks(s.vector_owned_count),128>>>(v,s,values,truth,s.vector_owned_count);return LaunchOk();}
bool CudaLaunchGatherRemoteVector(const CudaDeviceView& v,const CudaMeshShape& s,const double* values) noexcept
{if(s.vector_send_count)GatherRemoteVectorKernel<<<Blocks(s.vector_send_count),128>>>(v,s,values,s.vector_send_count);return LaunchOk();}
bool CudaLaunchScatterRemoteVector(const CudaDeviceView& v,const CudaMeshShape& s,double* truth) noexcept
{if(s.vector_receive_count)ScatterRemoteVectorKernel<<<Blocks(s.vector_receive_count),128>>>(v,truth,s.vector_receive_count);return LaunchOk();}
bool CudaLaunchRecover(const CudaDeviceView& v,const CudaMeshShape& s) noexcept
{if(s.borrowed_host_inputs)BorrowedRecoverKernel<<<s.elements,128>>>(v,s);else RecoverKernel<<<Blocks(s.elements),128>>>(v,s);return LaunchOk();}
bool CudaLaunchTrial(const CudaDeviceView& v,const CudaMeshShape& s,double alpha) noexcept
{TrialKernel<<<Blocks(std::max(s.interior_count,s.true_trace_count)),128>>>(v,s,alpha);return LaunchOk();}
bool CudaLaunchPackInteriorOutput(const CudaMeshShape& s,
                                  const double* source,
                                  double* packed,
                                  int output_components) noexcept
{const int count=s.elements*output_components*s.scalar_interior_dofs;if(source==nullptr||packed==nullptr||output_components<=0||output_components>s.interior_components||count<=0)return false;PackInteriorOutputKernel<<<Blocks(count),128>>>(s,source,packed,output_components);return LaunchOk();}
bool CudaLaunchLocalSquaredNorm(const double* a,int an,const double* b,int bn,
                                double* scratch,int scratch_count,
                                double* out) noexcept
{
  if(out==nullptr||an<0||bn<0||(an>0&&a==nullptr)||(bn>0&&b==nullptr)||
     an>std::numeric_limits<int>::max()-bn)return false;
  const int count=an+bn;
  if(count==0)
    return cudaMemsetAsync(out,0,sizeof(double),nullptr)==cudaSuccess;
  if(scratch==nullptr||scratch==out||scratch_count<=0)return false;
  const int needed_blocks=(count-1)/cooperative_reduction_threads+1;
  const int blocks=std::min(needed_blocks,scratch_count);
  NormPartialsKernel<<<blocks,cooperative_reduction_threads>>>(
      a,an,b,bn,scratch);
  if(!LaunchOk())return false;
  NormReduceKernel<<<1,cooperative_reduction_threads>>>(scratch,blocks,out);
  return LaunchOk();
}
bool CudaLaunchMaximumWave(const CudaDeviceView& v,const CudaMeshShape& s,
                           const double* interior,const double* trace,
                           double* out) noexcept
{const int sites=s.volume_points+2*faces_per_element*s.face_points;if(s.elements<=0||sites<=0||sites>cooperative_wave_threads)return false;MaximumWaveElementsKernel<<<s.elements,cooperative_wave_threads>>>(v,s,interior,trace);if(!LaunchOk())return false;MaximumWaveReduceKernel<<<1,cooperative_reduction_threads>>>(v.status,s.elements,out);return LaunchOk();}
bool CudaLaunchEulerConservation(const CudaDeviceView& v,
                                 const CudaMeshShape& s,
                                 const double dt, double* out) noexcept
{
  if(!s.borrowed_host_inputs||!isfinite(dt)||!(dt>0.0)||out==nullptr||
     v.f==nullptr||v.residual_r==nullptr||s.elements<=0||
     s.interior_dofs<euler_conservation_result_scalars||
     s.interior_count<=0||s.elements>s.interior_count/s.interior_dofs||
     s.volume_points<=0||s.volume_points>cooperative_conservation_threads||
     s.face_points<=0||
     s.face_points>(cooperative_conservation_threads-s.volume_points)/
         faces_per_element||
     (s.igr_active&&v.face_h==nullptr))return false;
  EulerConservationElementsKernel<<<
      s.elements,cooperative_conservation_threads>>>(v,s,dt);
  if(!LaunchOk())return false;
  EulerConservationReduceKernel<<<
      1,cooperative_conservation_threads>>>(v,s,out);
  return LaunchOk();
}
bool CudaLaunchHash(const double* values,int count,std::uint64_t* words) noexcept
{HashKernel<<<1,4>>>(values,count,words);return LaunchOk();}
bool CudaLaunchHashPair(const double* first,int first_count,
                        const double* second,int second_count,
                        std::uint64_t* words) noexcept
{HashPairKernel<<<1,8>>>(first,first_count,second,second_count,words);return LaunchOk();}

} // namespace detail
} // namespace dg::hdg::runtime::cuda_w8a
