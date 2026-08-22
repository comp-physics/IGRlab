#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"

#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
#include <_hypre_parcsr_ls.h>
#include <_hypre_utilities.h>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>

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
  if (communicator == MPI_COMM_NULL ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0 ||
      MPI_Allreduce(&local, &minimum, 1, MPI_UINT64_T, MPI_MIN,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local, &maximum, 1, MPI_UINT64_T, MPI_MAX,
                    communicator) != MPI_SUCCESS)
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

bool AllRanksReady(const bool local, const MPI_Comm communicator) noexcept
{
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
                                  const double relative_tolerance,
                                  const int maximum_iterations,
                                  const int krylov_dimension,
                                  const std::uint64_t matrix_generation,
                                  CudaHypreSetupOrderEvidence& evidence)
    noexcept
{
  (void)ranks;
  CudaTrueSystemDiagnosticScope diagnostic("object-configure",
                                           matrix_generation);
  evidence = {};
  evidence.matrix_generation = matrix_generation;
  try
  {
    // Pinned MFEM v4.9 HypreGMRES::SetOperator forwards the new matrix to the
    // preconditioner.  HypreILU::SetOperator then recreates its HYPRE handle
    // with MFEM defaults (fill 1/reorder 1), so all exact ILU(0) settings must
    // follow this generation's SetOperator call.
    gmres.SetPreconditioner(ilu);
    evidence.Record(CudaHypreSetupAction::bind_preconditioner);
    gmres.SetOperator(matrix);
    evidence.Record(CudaHypreSetupAction::set_operator);
    ilu.SetType(0);
    evidence.Record(CudaHypreSetupAction::ilu_type0);
    ilu.SetLevelOfFill(0);
    evidence.Record(CudaHypreSetupAction::ilu_fill0);
    ilu.SetLocalReordering(0);
    evidence.Record(CudaHypreSetupAction::ilu_reorder0);
    ilu.SetMaxIter(1);
    evidence.Record(CudaHypreSetupAction::ilu_maxiter1);
    ilu.SetTol(0.0);
    evidence.Record(CudaHypreSetupAction::ilu_tol0);
    ilu.SetPrintLevel(0);
    evidence.Record(CudaHypreSetupAction::ilu_print0);
    // SetOperator already installs the recreated ILU handle.  Rebinding after
    // the setters makes that handle association explicit without resetting it.
    gmres.SetPreconditioner(ilu);
    evidence.Record(CudaHypreSetupAction::rebind_preconditioner);
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
    evidence.Record(CudaHypreSetupAction::gmres_settings);
    diagnostic.result =
        evidence.ConfigurationExactForGeneration(matrix_generation);
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

bool CudaRefreshFactorAndSolve(
    mfem::HypreParMatrix& matrix, mfem::HypreParVector& rhs,
    mfem::HypreParVector& solution, mfem::HypreILU& ilu,
    mfem::HypreGMRES& gmres, const int ranks,
    const MPI_Comm communicator,
    const double relative_tolerance, const int maximum_iterations,
    const int krylov_dimension,
    const std::uint64_t matrix_generation,
    CudaSolveDiagnostics& diagnostics) noexcept
{
  CudaTrueSystemDiagnosticScope diagnostic("warm-solve", matrix_generation);
  diagnostics = {};
  diagnostics.matrix_generation = matrix_generation;
  CudaHypreSetupOrderEvidence setup_order{};
  bool policy_ready = false;
  try
  {
    policy_ready = CudaHypreDevicePolicyValid(matrix, rhs, solution) &&
        CudaConfigurePersistentHypre(matrix, ilu, gmres, ranks,
                                     relative_tolerance, maximum_iterations,
                                     krylov_dimension,
                                     matrix_generation, setup_order);
  }
  catch (...) { policy_ready = false; }
  if (!AllRanksReady(policy_ready, communicator)) { return false; }
  bool vectors_ready = false;
  try
  {
    solution = 0.0;
    solution.HypreReadWrite();
    vectors_ready = true;
  }
  catch (...) { vectors_ready = false; }
  if (!AllRanksReady(vectors_ready, communicator)) { return false; }
  try
  {
    gmres.Mult(rhs, solution);
    if (!setup_order.Record(CudaHypreSetupAction::mult)) { return false; }
    diagnostics.device_policy =
        CudaHypreDevicePolicyValid(matrix, rhs, solution);
    diagnostics.configured_ilu0 =
        setup_order.SolveExactForGeneration(matrix_generation);
    gmres.GetNumIterations(diagnostics.iterations);
    gmres.GetFinalResidualNorm(diagnostics.reported_residual);
    diagnostics.finite = std::isfinite(diagnostics.reported_residual);
    diagnostics.converged = diagnostics.finite &&
        diagnostics.reported_residual <= relative_tolerance;
    diagnostic.result = diagnostics.device_policy &&
        diagnostics.configured_ilu0 && diagnostics.converged;
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
  const double kinetic = 0.5 *
      (state[1] * state[1] + state[2] * state[2]) / state[0];
  const double pressure = (cuda_gamma - 1.0) * (state[3] - kinetic);
  const double sound_squared = cuda_gamma * pressure / state[0];
  return pressure > 0.0 && sound_squared > 0.0 && isfinite(pressure) &&
      isfinite(sound_squared) && isfinite(sqrt(sound_squared));
}

// Borrowed input admission is physical rather than the manufactured W8A
// speed cap: rho, pressure, and sound speed must be finite and strictly
// positive.  The local LLF field below is deliberately lagged, so it has no
// derivative contribution to the Newton blocks.
__device__ bool LocalLlfSpeed(const double* state, const double nx,
                              const double ny, double& speed)
{
  if (!Finite4(state) || !(state[0] > 0.0) || !isfinite(nx) ||
      !isfinite(ny) || (nx == 0.0 && ny == 0.0)) return false;
  const double rho = state[0];
  const double mx = state[1], my = state[2], energy = state[3];
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

struct CudaPrimitiveNormalState
{
  double rho, pressure, normal_velocity, tangential_velocity, sound_speed;
};

__device__ CudaPrimitiveNormalState ToPrimitiveNormal(
    const double* state, const double gamma, const double nx, const double ny,
    const double tx, const double ty)
{
  const double rho = fmax(state[0], 1.0e-12);
  const double ux = state[1] / rho;
  const double uy = state[2] / rho;
  const double pressure = fmax((gamma - 1.0) *
      (state[3] - 0.5 * rho * (ux * ux + uy * uy)), 1.0e-12);
  return {rho, pressure, ux * nx + uy * ny, ux * tx + uy * ty,
          sqrt(fmax(gamma * pressure / rho, 0.0))};
}

__device__ void FromPrimitiveNormal(
    const CudaPrimitiveNormalState& primitive, const double gamma,
    const double nx, const double ny, const double tx, const double ty,
    double* state)
{
  const double ux = primitive.normal_velocity * nx +
      primitive.tangential_velocity * tx;
  const double uy = primitive.normal_velocity * ny +
      primitive.tangential_velocity * ty;
  state[0] = primitive.rho;
  state[1] = primitive.rho * ux;
  state[2] = primitive.rho * uy;
  state[3] = primitive.pressure / (gamma - 1.0) +
      0.5 * primitive.rho * (ux * ux + uy * uy);
}

__device__ double EntropyLike(const CudaPrimitiveNormalState& state,
                              const double gamma)
{ return state.pressure / pow(fmax(state.rho, 1.0e-12), gamma); }

__device__ CudaPrimitiveNormalState PressureOutletClosure(
    const CudaPrimitiveNormalState& inner, const double gamma,
    const double pressure_out)
{
  CudaPrimitiveNormalState result = inner;
  if (inner.normal_velocity >= inner.sound_speed ||
      inner.normal_velocity <= -inner.sound_speed)
    return result;
  const double gm1 = gamma - 1.0;
  const double entropy = EntropyLike(inner, gamma);
  const double pressure = fmax(pressure_out, 1.0e-12);
  const double rho = pow(pressure / entropy, 1.0 / gamma);
  const double sound = sqrt(fmax(gamma * pressure / rho, 0.0));
  const double invariant =
      inner.normal_velocity + 2.0 * inner.sound_speed / gm1;
  result.rho = fmax(rho, 1.0e-12);
  result.pressure = pressure;
  result.sound_speed = sound;
  result.normal_velocity = invariant - 2.0 * sound / gm1;
  return result;
}

__device__ CudaPrimitiveNormalState NonreflectingClosure(
    const CudaPrimitiveNormalState& inner,
    const CudaPrimitiveNormalState& reference, const double gamma)
{
  CudaPrimitiveNormalState result = inner;
  if (inner.normal_velocity >= inner.sound_speed ||
      inner.normal_velocity <= -inner.sound_speed)
    return result;
  const double gm1 = gamma - 1.0;
  const double plus = inner.normal_velocity + 2.0 * inner.sound_speed / gm1;
  const double minus = reference.normal_velocity -
      2.0 * reference.sound_speed / gm1;
  const double normal_velocity = 0.5 * (plus + minus);
  const double sound = 0.25 * gm1 * (plus - minus);
  if (!(sound > 0.0)) return result;
  const double entropy = EntropyLike(reference, gamma);
  const double rho = pow((sound * sound) /
      fmax(gamma * entropy, 1.0e-12), 1.0 / gm1);
  result.rho = fmax(rho, 1.0e-12);
  result.pressure = fmax(entropy * pow(result.rho, gamma), 1.0e-12);
  result.sound_speed = sound;
  result.normal_velocity = normal_velocity;
  result.tangential_velocity = inner.tangential_velocity;
  return result;
}

__device__ CudaPrimitiveNormalState SubsonicInletClosure(
    const CudaPrimitiveNormalState& inner,
    const CudaPrimitiveNormalState& target, const double gamma)
{
  if (inner.normal_velocity <= -inner.sound_speed) return target;
  if (inner.normal_velocity >= inner.sound_speed) return inner;
  CudaPrimitiveNormalState result = target;
  const double gm1 = gamma - 1.0;
  const double plus = inner.normal_velocity + 2.0 * inner.sound_speed / gm1;
  const double minus =
      target.normal_velocity - 2.0 * target.sound_speed / gm1;
  const double normal_velocity = 0.5 * (plus + minus);
  const double sound = 0.25 * gm1 * (plus - minus);
  if (!(sound > 0.0)) return target;
  const double entropy = EntropyLike(target, gamma);
  const double rho = pow((sound * sound) /
      fmax(gamma * entropy, 1.0e-12), 1.0 / gm1);
  result.rho = fmax(rho, 1.0e-12);
  result.pressure = fmax(entropy * pow(result.rho, gamma), 1.0e-12);
  result.sound_speed = sound;
  result.normal_velocity = normal_velocity;
  return result;
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
  if (!(normal_norm > 0.0) || !isfinite(normal_norm)) return false;
  const double nx = normal_x / normal_norm;
  const double ny = normal_y / normal_norm;
  const double tx = -ny;
  const double ty = nx;
  if (kind == static_cast<int>(CudaStaticBoundaryPolicyKind::supersonic_inlet))
  {
    if (!reference || !PhysicallyAdmissible(reference)) return false;
    for (int component = 0; component < conserved_components; ++component)
      exterior[component] = reference[component];
    return true;
  }
  if (kind == static_cast<int>(CudaStaticBoundaryPolicyKind::supersonic_outlet))
  {
    for (int component = 0; component < conserved_components; ++component)
      exterior[component] = interior[component];
    return true;
  }
  if (kind == static_cast<int>(CudaStaticBoundaryPolicyKind::slip_wall))
  {
    const double normal_squared = nx * nx + ny * ny;
    if (!(normal_squared > 0.0) || !isfinite(normal_squared)) return false;
    const double rho = fmax(interior[0], 1.0e-12);
    const double ux = interior[1] / rho;
    const double uy = interior[2] / rho;
    const double normal_velocity = ux * nx + uy * ny;
    exterior[0] = interior[0];
    exterior[1] = rho * (ux - 2.0 * normal_velocity * nx / normal_squared);
    exterior[2] = rho * (uy - 2.0 * normal_velocity * ny / normal_squared);
    exterior[3] = interior[3];
    return PhysicallyAdmissible(exterior);
  }
  const CudaPrimitiveNormalState inner =
      ToPrimitiveNormal(interior, gamma, nx, ny, tx, ty);
  CudaPrimitiveNormalState boundary = inner;
  if (kind == static_cast<int>(
          CudaStaticBoundaryPolicyKind::subsonic_outlet_pressure))
    boundary = PressureOutletClosure(inner, gamma, pressure_out);
  else if (kind == static_cast<int>(
               CudaStaticBoundaryPolicyKind::nonreflecting_outlet))
  {
    if (!reference || !PhysicallyAdmissible(reference)) return false;
    boundary = NonreflectingClosure(
        inner, ToPrimitiveNormal(reference, gamma, nx, ny, tx, ty), gamma);
  }
  else if (kind == static_cast<int>(
               CudaStaticBoundaryPolicyKind::subsonic_inlet))
  {
    if (!reference || !PhysicallyAdmissible(reference)) return false;
    boundary = SubsonicInletClosure(
        inner, ToPrimitiveNormal(reference, gamma, nx, ny, tx, ty), gamma);
  }
  else if (kind == static_cast<int>(CudaStaticBoundaryPolicyKind::far_field))
  {
    if (!reference || !PhysicallyAdmissible(reference)) return false;
    const CudaPrimitiveNormalState far =
        ToPrimitiveNormal(reference, gamma, nx, ny, tx, ty);
    if (inner.normal_velocity <= -inner.sound_speed)
    {
      for (int component = 0; component < conserved_components; ++component)
        exterior[component] = reference[component];
      return true;
    }
    if (inner.normal_velocity >= inner.sound_speed)
    {
      for (int component = 0; component < conserved_components; ++component)
        exterior[component] = interior[component];
      return true;
    }
    boundary = inner.normal_velocity < 0.0
        ? SubsonicInletClosure(inner, far, gamma)
        : NonreflectingClosure(inner, far, gamma);
  }
  else
    return false;
  FromPrimitiveNormal(boundary, gamma, nx, ny, tx, ty, exterior);
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
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= shape.elements) return;
  view.status[element] = 1.0;
  const int ni0 = element * shape.interior_dofs;
  const int nt0 = element * shape.trace_dofs;
  for (int q = 0; q < shape.volume_points; ++q)
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
          Admissible(state))) { view.status[element] = -1.0; return; }
  }
  for (int face = 0; face < faces_per_element; ++face)
    for (int q = 0; q < shape.face_points; ++q)
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
      { view.status[element] = -2.0; return; }
    }
}

__global__ void SeedBorrowedLaggedLlfTauKernel(CudaDeviceView view,
                                                CudaMeshShape shape)
{
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= shape.elements) return;
  view.status[element] = 1.0;
  const int interior_offset = element * shape.interior_dofs;
  const int trace_offset = element * shape.trace_dofs;
  for (int face = 0; face < faces_per_element; ++face)
    for (int q = 0; q < shape.face_points; ++q)
    {
      const int qp = (element * faces_per_element + face) *
          shape.face_points + q;
      const int face_offset = face * shape.trace_components *
          shape.scalar_face_dofs;
      double old_interior[conserved_components] = {};
      double old_trace[conserved_components] = {};
      for (int scalar = 0; scalar < shape.scalar_interior_dofs; ++scalar)
        for (int component = 0; component < conserved_components; ++component)
          old_interior[component] += view.face_interior_shape[
              qp * shape.scalar_interior_dofs + scalar] *
              view.old_interior[interior_offset + ByNodes(
                  scalar, shape.scalar_interior_dofs, component)];
      for (int scalar = 0; scalar < shape.scalar_face_dofs; ++scalar)
        for (int component = 0; component < conserved_components; ++component)
          old_trace[component] += view.face_trace_shape[
              qp * shape.scalar_face_dofs + scalar] *
              view.current_local_trace[trace_offset + face_offset + ByNodes(
                  scalar, shape.scalar_face_dofs, component)];
      double interior_speed = 0.0, trace_speed = 0.0;
      if (!LocalLlfSpeed(old_interior, view.face_normal[qp * dimension],
                         view.face_normal[qp * dimension + 1], interior_speed) ||
          !LocalLlfSpeed(old_trace, view.face_normal[qp * dimension],
                         view.face_normal[qp * dimension + 1], trace_speed))
      {
        view.status[element] = -3.0;
        return;
      }
      const double tau = fmax(interior_speed, trace_speed);
      if (!(tau > 0.0) || !isfinite(tau))
      {
        view.status[element] = -3.0;
        return;
      }
      view.borrowed_lagged_llf_tau[qp] = tau;
    }
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

// This is intentionally a separate instantiation from AssemblyKernel: the
// legacy fixed-tau manufactured path above retains its arithmetic verbatim.
// Here tau is a prepared-attempt-lagged coefficient, not a differentiated
// state function, so every residual and A/B/C/D contribution reads the same
// stored per-face quadrature value and adds no d-tau terms.
__global__ void BorrowedLaggedLlfAssemblyKernel(CudaDeviceView v,
                                                CudaMeshShape s,
                                                const double dt)
{
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= s.elements) return;
  const int ni = s.interior_dofs, nt = s.trace_dofs;
  const int si = s.scalar_interior_dofs, sf = s.scalar_face_dofs;
  const int e_i = e * ni, e_t = e * nt;
  const std::size_t ao = static_cast<std::size_t>(e) * ni * ni;
  const std::size_t bo = static_cast<std::size_t>(e) * ni * nt;
  const std::size_t co = static_cast<std::size_t>(e) * nt * ni;
  const std::size_t doff = static_cast<std::size_t>(e) * nt * nt;
  v.status[e] = 1.0;
  for (int i = 0; i < ni; ++i) { v.residual_r[e_i+i] = 0; v.f[e_i+i] = 0; }
  for (int i = 0; i < nt; ++i) { v.residual_g[e_t+i] = 0; v.g[e_t+i] = 0; }
  for (int i = 0; i < ni * ni; ++i) v.a[ao+i] = 0;
  for (int i = 0; i < ni * nt; ++i) v.b[bo+i] = 0;
  for (int i = 0; i < nt * ni; ++i) v.c[co+i] = 0;
  for (int i = 0; i < nt * nt; ++i) v.d[doff+i] = 0;
  for (int q = 0; q < s.volume_points; ++q)
  {
    const int qp = e * s.volume_points + q;
    const double w = v.volume_weight[qp];
    double state[4] = {}, old[4] = {}, source[4] = {}, fx[4], fy[4];
    double jx[conserved_components * igr_trace_components] = {};
    double jy[conserved_components * igr_trace_components] = {};
    double sigma = 0.0, z[dimension] = {};
    double conservative_gradient[3][dimension] = {};
    for (int a = 0; a < si; ++a) for (int k = 0; k < 4; ++k)
    {
      const double phi = v.volume_shape[qp * si + a];
      state[k] += phi * v.work_interior[e_i + ByNodes(a, si, k)];
      old[k] += phi * v.old_interior[e_i + ByNodes(a, si, k)];
      if (s.borrowed_static_source)
        source[k] += phi * v.static_source[e_i + ByNodes(a, si, k)];
    }
    if (s.igr_active)
      for (int a = 0; a < si; ++a)
      {
        const double phi = v.volume_shape[qp * si + a];
        sigma += phi * v.work_interior[e_i + ByNodes(a, si, 4)];
        z[0] += phi * v.work_interior[e_i + ByNodes(a, si, 5)];
        z[1] += phi * v.work_interior[e_i + ByNodes(a, si, 6)];
        for (int component = 0; component < 3; ++component)
          for (int direction = 0; direction < dimension; ++direction)
            conservative_gradient[component][direction] +=
                v.volume_derivative[(qp * si + a) * dimension + direction] *
                v.work_interior[e_i + ByNodes(a, si, component)];
      }
    // A generic empty source is exactly zero.  Manufactured forcing belongs
    // only to the frozen AssemblyKernel path above.
    if (!(w > 0) || !(s.igr_active
              ? RegularizedFluxJacobian(state, sigma, 1, 0, fx, jx) &&
                    RegularizedFluxJacobian(state, sigma, 0, 1, fy, jy)
              : FluxJacobian(state, 1, 0, fx, jx) &&
                    FluxJacobian(state, 0, 1, fy, jy)))
    { v.status[e] = -1; return; }
    for (int row = 0; row < si; ++row)
    {
      const double phi = v.volume_shape[qp * si + row];
      const double dx = v.volume_derivative[(qp * si + row) * 2];
      const double dy = v.volume_derivative[(qp * si + row) * 2 + 1];
      for (int rc = 0; rc < 4; ++rc)
      {
        const int lr = ByNodes(row, si, rc);
        v.residual_r[e_i+lr] += w * (phi * (state[rc]-old[rc]) / dt -
            dx * fx[rc] - dy * fy[rc] - phi * source[rc]);
        const int euler_columns = s.igr_active
            ? igr_trace_components : conserved_components;
        for (int col = 0; col < si; ++col)
          for (int cc = 0; cc < euler_columns; ++cc)
        {
          const int lc = ByNodes(col, si, cc);
          const double mass = rc == cc ? phi * v.volume_shape[qp*si+col] / dt : 0;
          const int stride = s.igr_active
              ? igr_trace_components : conserved_components;
          const double flux = -(dx*jx[rc*stride+cc] +
              dy*jy[rc*stride+cc]) *
              v.volume_shape[qp*si+col];
          v.a[ao+lr*ni+lc] += w * (mass + flux);
        }
      }

      if (!s.igr_active) continue;
      for (int direction = 0; direction < dimension; ++direction)
      {
        const int lr = ByNodes(row, si, 4 + direction);
        const double derivative = v.volume_derivative[
            (qp * si + row) * dimension + direction];
        v.residual_r[e_i + lr] += w *
            (phi * state[0] * z[direction] + derivative * sigma);
        for (int col = 0; col < si; ++col)
        {
          const double column_phi = v.volume_shape[qp * si + col];
          v.a[ao + lr * ni + ByNodes(col, si, 0)] +=
              w * phi * z[direction] * column_phi;
          v.a[ao + lr * ni + ByNodes(col, si, 4)] +=
              w * derivative * column_phi;
          v.a[ao + lr * ni + ByNodes(col, si, 5 + direction)] +=
              w * phi * state[0] * column_phi;
        }
      }

      double velocity_gradient[dimension][dimension] = {};
      for (int velocity = 0; velocity < dimension; ++velocity)
        for (int direction = 0; direction < dimension; ++direction)
          velocity_gradient[velocity][direction] =
              conservative_gradient[1 + velocity][direction] / state[0] -
              state[1 + velocity] * conservative_gradient[0][direction] /
                  (state[0] * state[0]);
      const double velocity_trace =
          velocity_gradient[0][0] + velocity_gradient[1][1];
      const double q_value =
          velocity_gradient[0][0] * velocity_gradient[0][0] +
          2.0 * velocity_gradient[0][1] * velocity_gradient[1][0] +
          velocity_gradient[1][1] * velocity_gradient[1][1] +
          velocity_trace * velocity_trace;
      double dq_dgradient[dimension][dimension] = {};
      for (int velocity = 0; velocity < dimension; ++velocity)
        for (int direction = 0; direction < dimension; ++direction)
          dq_dgradient[velocity][direction] =
              2.0 * velocity_gradient[direction][velocity] +
              (velocity == direction ? 2.0 * velocity_trace : 0.0);
      const int sigma_row = ByNodes(row, si, 6);
      v.residual_r[e_i + sigma_row] += w *
          (phi * sigma / state[0] + s.igr_alpha *
              (dx * z[0] + dy * z[1]) - s.igr_alpha * phi * q_value);
      for (int col = 0; col < si; ++col)
      {
        const double column_phi = v.volume_shape[qp * si + col];
        v.a[ao + sigma_row * ni + ByNodes(col, si, 4)] +=
            w * phi * column_phi / state[0];
        for (int direction = 0; direction < dimension; ++direction)
          v.a[ao + sigma_row * ni + ByNodes(col, si, 5 + direction)] +=
              w * s.igr_alpha * v.volume_derivative[
                  (qp * si + row) * dimension + direction] * column_phi;
        for (int variable = 0; variable < conserved_components; ++variable)
        {
          double dq = 0.0;
          if (variable < 3)
            for (int velocity = 0; velocity < dimension; ++velocity)
              for (int direction = 0; direction < dimension; ++direction)
              {
                const double momentum = state[1 + velocity];
                const double momentum_derivative =
                    conservative_gradient[1 + velocity][direction];
                const double density_derivative =
                    conservative_gradient[0][direction];
                const double delta_density = variable == 0 ? column_phi : 0.0;
                const double delta_density_derivative = variable == 0
                    ? v.volume_derivative[(qp * si + col) * dimension +
                                          direction]
                    : 0.0;
                const double delta_momentum =
                    variable == 1 + velocity ? column_phi : 0.0;
                const double delta_momentum_derivative =
                    variable == 1 + velocity
                    ? v.volume_derivative[(qp * si + col) * dimension +
                                          direction]
                    : 0.0;
                const double delta_gradient =
                    delta_momentum_derivative / state[0] -
                    momentum_derivative * delta_density /
                        (state[0] * state[0]) -
                    delta_momentum * density_derivative /
                        (state[0] * state[0]) -
                    momentum * delta_density_derivative /
                        (state[0] * state[0]) +
                    2.0 * momentum * density_derivative * delta_density /
                        (state[0] * state[0] * state[0]);
                dq += dq_dgradient[velocity][direction] * delta_gradient;
              }
          double derivative = -s.igr_alpha * phi * dq;
          if (variable == 0)
            derivative -= phi * sigma * column_phi /
                (state[0] * state[0]);
          v.a[ao + sigma_row * ni + ByNodes(col, si, variable)] +=
              w * derivative;
        }
      }
    }
  }
  for (int face = 0; face < faces_per_element; ++face)
    for (int q = 0; q < s.face_points; ++q)
    {
      const int qp = (e * faces_per_element + face) * s.face_points + q;
      const int to = face * s.trace_components * sf;
      const double w = v.face_weight[qp];
      const double stabilization = s.igr_active
          ? s.igr_beta : v.borrowed_lagged_llf_tau[qp];
      const int boundary_kind = s.borrowed_static_boundary_policies
          ? v.boundary_policy_kind[e * faces_per_element + face] : 0;
      double u[4] = {}, l[4] = {}, flux[4];
      double jac[conserved_components * igr_trace_components] = {};
      double sigma = 0.0, sigma_hat = 0.0, z[dimension] = {};
      for (int a = 0; a < si; ++a) for (int k = 0; k < 4; ++k)
        u[k] += v.face_interior_shape[qp*si+a] *
            v.work_interior[e_i+ByNodes(a,si,k)];
      for (int a = 0; a < sf; ++a) for (int k = 0; k < 4; ++k)
        l[k] += v.face_trace_shape[qp*sf+a] *
            v.current_local_trace[e_t+to+ByNodes(a,sf,k)];
      if (s.igr_active)
      {
        for (int a = 0; a < si; ++a)
        {
          const double phi = v.face_interior_shape[qp * si + a];
          sigma += phi * v.work_interior[e_i + ByNodes(a, si, 4)];
          z[0] += phi * v.work_interior[e_i + ByNodes(a, si, 5)];
          z[1] += phi * v.work_interior[e_i + ByNodes(a, si, 6)];
        }
        for (int a = 0; a < sf; ++a)
          sigma_hat += v.face_trace_shape[qp * sf + a] *
              v.current_local_trace[e_t + to + ByNodes(a, sf, 4)];
      }
      const double nx = v.face_normal[qp * dimension];
      const double ny = v.face_normal[qp * dimension + 1];
      if (!(w > 0) || !(stabilization > 0) || !isfinite(stabilization) ||
          !(s.igr_active
                ? RegularizedFluxJacobian(
                      l, sigma_hat, nx, ny, flux, jac)
                : FluxJacobian(l, nx, ny, flux, jac)))
      { v.status[e] = -2; return; }
      double exterior[conserved_components] = {};
      const bool physical_boundary = boundary_kind != 0;
      if (physical_boundary && !StaticBoundaryExterior(boundary_kind,
          v.boundary_reference + (e * faces_per_element + face) *
              conserved_components,
          v.boundary_pressure[e * faces_per_element + face],
          s.boundary_gamma, u, nx, ny, exterior))
      { v.status[e] = -3; return; }
      double tau = 0.0, jump = 0.0, z_hat_normal = 0.0;
      if (s.igr_active)
      {
        const double h = v.face_h[e * faces_per_element + face];
        const double degree_scale =
            static_cast<double>((s.order + 1) * (s.order + 1));
        if (!(l[0] > cuda_density_gate) || !(h > 0.0) || !isfinite(h))
        { v.status[e] = -4; return; }
        tau = degree_scale / (l[0] * h);
        jump = sigma - sigma_hat;
        z_hat_normal = z[0] * nx + z[1] * ny - tau * jump;
        if (!(tau > 0.0) || !isfinite(tau) || !isfinite(z_hat_normal))
        { v.status[e] = -4; return; }
      }
      for (int row = 0; row < si; ++row)
      {
        const double rp = v.face_interior_shape[qp*si+row];
        for (int rc = 0; rc < 4; ++rc)
        {
          const int lr = ByNodes(row,si,rc);
          v.residual_r[e_i+lr] += w*rp*
              (flux[rc]+stabilization*(u[rc]-l[rc]));
          for (int col = 0; col < si; ++col)
            v.a[ao+lr*ni+ByNodes(col,si,rc)] += w*rp*stabilization*
                v.face_interior_shape[qp*si+col];
          const int euler_columns = s.igr_active
              ? igr_trace_components : conserved_components;
          for (int col = 0; col < sf; ++col)
            for (int cc = 0; cc < euler_columns; ++cc)
            v.b[bo+lr*nt+to+ByNodes(col,sf,cc)] += w*rp*
                v.face_trace_shape[qp*sf+col] *
                (jac[rc*euler_columns+cc] -
                 (rc == cc ? stabilization : 0));
        }
      }
      for (int row = 0; row < sf; ++row)
      {
        const double rp = v.face_trace_shape[qp*sf+row];
        for (int rc = 0; rc < 4; ++rc)
        {
          const int lr = to + ByNodes(row,sf,rc);
          v.residual_g[e_t+lr] += w*rp*(physical_boundary
              ? l[rc] - exterior[rc]
              : flux[rc]+stabilization*(u[rc]-l[rc]));
          for (int col = 0; col < si; ++col)
            if (!physical_boundary)
              v.c[co+lr*ni+ByNodes(col,si,rc)] += w*rp*stabilization*
                  v.face_interior_shape[qp*si+col];
          const int euler_columns = s.igr_active
              ? igr_trace_components : conserved_components;
          for (int col = 0; col < sf; ++col)
            for (int cc = 0; cc < euler_columns; ++cc)
            v.d[doff+lr*nt+to+ByNodes(col,sf,cc)] += w*rp*
                v.face_trace_shape[qp*sf+col] *
                (physical_boundary
                    ? (rc == cc ? 1.0 : 0.0)
                    : jac[rc*euler_columns+cc] -
                        (rc == cc ? stabilization : 0));
        }
      }

      if (s.igr_active)
      {
        for (int row = 0; row < si; ++row)
        {
          const double rp = v.face_interior_shape[qp * si + row];
          for (int direction = 0; direction < dimension; ++direction)
          {
            const double normal = direction == 0 ? nx : ny;
            const int lr = ByNodes(row, si, 4 + direction);
            v.residual_r[e_i + lr] -= w * rp * normal * sigma_hat;
            for (int col = 0; col < sf; ++col)
              v.b[bo + lr * nt + to + ByNodes(col, sf, 4)] -=
                  w * rp * normal * v.face_trace_shape[qp * sf + col];
          }
          const int sigma_row = ByNodes(row, si, 6);
          v.residual_r[e_i + sigma_row] -=
              s.igr_alpha * w * rp * z_hat_normal;
          for (int col = 0; col < si; ++col)
          {
            const double cp = v.face_interior_shape[qp * si + col];
            v.a[ao + sigma_row * ni + ByNodes(col, si, 4)] +=
                s.igr_alpha * w * rp * tau * cp;
            v.a[ao + sigma_row * ni + ByNodes(col, si, 5)] -=
                s.igr_alpha * w * rp * nx * cp;
            v.a[ao + sigma_row * ni + ByNodes(col, si, 6)] -=
                s.igr_alpha * w * rp * ny * cp;
          }
          for (int col = 0; col < sf; ++col)
          {
            const double cp = v.face_trace_shape[qp * sf + col];
            v.b[bo + sigma_row * nt + to + ByNodes(col, sf, 4)] -=
                s.igr_alpha * w * rp * tau * cp;
            v.b[bo + sigma_row * nt + to + ByNodes(col, sf, 0)] -=
                s.igr_alpha * w * rp * tau * jump / l[0] * cp;
          }
        }
        for (int row = 0; row < sf; ++row)
        {
          const double rp = v.face_trace_shape[qp * sf + row];
          const int lr = to + ByNodes(row, sf, 4);
          v.residual_g[e_t + lr] +=
              s.igr_alpha * w * rp * z_hat_normal;
          for (int col = 0; col < si; ++col)
          {
            const double cp = v.face_interior_shape[qp * si + col];
            v.c[co + lr * ni + ByNodes(col, si, 4)] -=
                s.igr_alpha * w * rp * tau * cp;
            v.c[co + lr * ni + ByNodes(col, si, 5)] +=
                s.igr_alpha * w * rp * nx * cp;
            v.c[co + lr * ni + ByNodes(col, si, 6)] +=
                s.igr_alpha * w * rp * ny * cp;
          }
          for (int col = 0; col < sf; ++col)
          {
            const double cp = v.face_trace_shape[qp * sf + col];
            v.d[doff + lr * nt + to + ByNodes(col, sf, 4)] +=
                s.igr_alpha * w * rp * tau * cp;
            v.d[doff + lr * nt + to + ByNodes(col, sf, 0)] +=
                s.igr_alpha * w * rp * tau * jump / l[0] * cp;
          }
        }
      }
    }
  for (int i = 0; i < ni; ++i) v.f[e_i+i] = -v.residual_r[e_i+i];
  for (int i = 0; i < nt; ++i) v.g[e_t+i] = -v.residual_g[e_t+i];
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
__global__ void TrialKernel(CudaDeviceView v,CudaMeshShape s,double alpha)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<s.interior_count)v.trial_interior[i]=v.base_interior[i]+alpha*v.recovered_interior[i];if(i<s.true_trace_count)v.trial_true_trace[i]=v.base_true_trace[i]+alpha*v.true_solution[i];}
__global__ void NormKernel(const double* a,int an,const double* b,int bn,double* out)
{if(blockIdx.x||threadIdx.x)return;double sum=0;for(int i=0;i<an;++i)sum+=a[i]*a[i];for(int i=0;i<bn;++i)sum+=b[i]*b[i];out[0]=sum;}
__device__ double EulerWave(const double rho,const double mx,const double my,
                            const double energy)
{if(!(rho>0.0)||!isfinite(rho)||!isfinite(mx)||!isfinite(my)||!isfinite(energy))return nan("");const double u=mx/rho,v=my/rho;const double p=0.4*(energy-0.5*(mx*mx+my*my)/rho);return p>0.0&&isfinite(p)?sqrt(u*u+v*v)+sqrt(1.4*p/rho):nan("");}
__device__ void ObserveEulerWave(const double* state,double& maximum,bool& valid)
{const double wave=EulerWave(state[0],state[1],state[2],state[3]);valid=valid&&isfinite(wave);if(valid&&wave>maximum)maximum=wave;}
__global__ void MaximumWaveKernel(CudaDeviceView v,CudaMeshShape s,
                                  const double* interior,
                                  const double* trace,double* out)
{if(blockIdx.x||threadIdx.x)return;double maximum=0.0;bool valid=true;for(int e=0;e<s.elements;++e){const int ib=e*s.interior_dofs,tb=e*s.trace_dofs;for(int q=0;q<s.volume_points;++q){const int qp=e*s.volume_points+q;double state[4]={};for(int a=0;a<s.scalar_interior_dofs;++a)for(int k=0;k<4;++k)state[k]+=v.volume_shape[qp*s.scalar_interior_dofs+a]*interior[ib+ByNodes(a,s.scalar_interior_dofs,k)];ObserveEulerWave(state,maximum,valid);}for(int f=0;f<faces_per_element;++f)for(int q=0;q<s.face_points;++q){const int qp=(e*faces_per_element+f)*s.face_points+q,fo=f*s.trace_components*s.scalar_face_dofs;double volume_state[4]={},trace_state[4]={};for(int a=0;a<s.scalar_interior_dofs;++a)for(int k=0;k<4;++k)volume_state[k]+=v.face_interior_shape[qp*s.scalar_interior_dofs+a]*interior[ib+ByNodes(a,s.scalar_interior_dofs,k)];for(int a=0;a<s.scalar_face_dofs;++a)for(int k=0;k<4;++k)trace_state[k]+=v.face_trace_shape[qp*s.scalar_face_dofs+a]*trace[tb+fo+ByNodes(a,s.scalar_face_dofs,k)];ObserveEulerWave(volume_state,maximum,valid);ObserveEulerWave(trace_state,maximum,valid);}}out[0]=maximum;out[1]=valid?1.0:0.0;}

// Compact post-candidate balance used by the production IGR lifecycle.  One
// thread deliberately accumulates in element/quadrature order so the device
// result mirrors the CPU reference reduction without a second assembler or
// nondeterministic atomics.  out[0:5] are balances, out[5:10] their scales,
// and out[10] is the finite/admissibility flag.
__global__ void IgrConservationKernel(CudaDeviceView v, CudaMeshShape s,
                                      const double dt, double* out)
{
  if (blockIdx.x || threadIdx.x) return;
  constexpr int components = 5;
  for (int index = 0; index < 2 * components; ++index) out[index] = 0.0;
  bool valid = s.igr_active && isfinite(dt) && dt > 0.0 &&
      v.face_h != nullptr;
  const double degree_scale = static_cast<double>(
      (s.order + 1) * (s.order + 1));
  for (int e = 0; e < s.elements && valid; ++e)
  {
    const int interior_offset = e * s.interior_dofs;
    const int trace_offset = e * s.trace_dofs;
    for (int q = 0; q < s.volume_points && valid; ++q)
    {
      const int qp = e * s.volume_points + q;
      const double weight = v.volume_weight[qp];
      double current[conserved_components] = {};
      double previous[conserved_components] = {};
      double source[conserved_components] = {};
      double sigma = 0.0;
      double conservative_gradient[3][dimension] = {};
      for (int scalar = 0; scalar < s.scalar_interior_dofs; ++scalar)
      {
        const double phi = v.volume_shape[
            qp * s.scalar_interior_dofs + scalar];
        for (int component = 0; component < conserved_components; ++component)
        {
          const int index = interior_offset + ByNodes(
              scalar, s.scalar_interior_dofs, component);
          current[component] += phi * v.work_interior[index];
          previous[component] += phi * v.old_interior[index];
          source[component] += phi * v.static_source[index];
        }
        sigma += phi * v.work_interior[interior_offset + ByNodes(
            scalar, s.scalar_interior_dofs, 4)];
        for (int component = 0; component < 3; ++component)
          for (int direction = 0; direction < dimension; ++direction)
            conservative_gradient[component][direction] +=
                v.volume_derivative[
                    (qp * s.scalar_interior_dofs + scalar) * dimension +
                    direction] *
                v.work_interior[interior_offset + ByNodes(
                    scalar, s.scalar_interior_dofs, component)];
      }
      valid = weight > 0.0 && isfinite(weight) &&
          current[0] > cuda_density_gate &&
          isfinite(sigma);
      for (int component = 0;
           component < conserved_components && valid; ++component)
      {
        const double change = weight *
            (current[component] - previous[component]);
        const double forcing = dt * weight * source[component];
        valid = isfinite(change) && isfinite(forcing);
        out[component] += change - forcing;
        out[components + component] += fabs(change) + fabs(forcing);
      }
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
      valid = valid && isfinite(q_value) && isfinite(sigma_contribution) &&
          isfinite(q_contribution);
      out[4] += sigma_contribution - q_contribution;
      out[9] += fabs(sigma_contribution) + fabs(q_contribution);
    }

    for (int face = 0; face < faces_per_element && valid; ++face)
    {
      const int element_face = e * faces_per_element + face;
      const bool physical = v.boundary_policy_kind != nullptr &&
          v.boundary_policy_kind[element_face] != 0;
      if (!physical) continue;
      const double h = v.face_h[element_face];
      valid = isfinite(h) && h > 0.0;
      const int face_offset = face * s.trace_components * s.scalar_face_dofs;
      for (int q = 0; q < s.face_points && valid; ++q)
      {
        const int qp = element_face * s.face_points + q;
        const double weight = v.face_weight[qp];
        const double normal[dimension] = {
            v.face_normal[qp * dimension],
            v.face_normal[qp * dimension + 1]};
        double interior[conserved_components] = {};
        double trace[conserved_components] = {};
        double sigma = 0.0;
        double sigma_hat = 0.0;
        double z[dimension] = {};
        for (int scalar = 0; scalar < s.scalar_interior_dofs; ++scalar)
        {
          const double phi = v.face_interior_shape[
              qp * s.scalar_interior_dofs + scalar];
          for (int component = 0; component < conserved_components;
               ++component)
            interior[component] += phi * v.work_interior[
                interior_offset + ByNodes(
                    scalar, s.scalar_interior_dofs, component)];
          sigma += phi * v.work_interior[interior_offset + ByNodes(
              scalar, s.scalar_interior_dofs, 4)];
          z[0] += phi * v.work_interior[interior_offset + ByNodes(
              scalar, s.scalar_interior_dofs, 5)];
          z[1] += phi * v.work_interior[interior_offset + ByNodes(
              scalar, s.scalar_interior_dofs, 6)];
        }
        for (int scalar = 0; scalar < s.scalar_face_dofs; ++scalar)
        {
          const double phi = v.face_trace_shape[
              qp * s.scalar_face_dofs + scalar];
          for (int component = 0; component < conserved_components;
               ++component)
            trace[component] += phi * v.current_local_trace[
                trace_offset + face_offset + ByNodes(
                    scalar, s.scalar_face_dofs, component)];
          sigma_hat += phi * v.current_local_trace[
              trace_offset + face_offset + ByNodes(
                  scalar, s.scalar_face_dofs, 4)];
        }
        double flux[conserved_components] = {};
        double jacobian[conserved_components * igr_trace_components] = {};
        valid = weight > 0.0 && isfinite(weight) &&
            RegularizedFluxJacobian(trace, sigma_hat, normal[0], normal[1],
                                    flux, jacobian);
        for (int component = 0;
             component < conserved_components && valid; ++component)
        {
          const double numerical_flux = flux[component] + s.igr_beta *
              (interior[component] - trace[component]);
          const double contribution = dt * weight * numerical_flux;
          valid = isfinite(contribution);
          out[component] += contribution;
          out[components + component] += fabs(contribution);
        }
        valid = valid && trace[0] > cuda_density_gate;
        const double tau = degree_scale / (trace[0] * h);
        const double z_hat_normal = z[0] * normal[0] + z[1] * normal[1] -
            tau * (sigma - sigma_hat);
        const double contribution = s.igr_alpha * weight * z_hat_normal;
        valid = valid && tau > 0.0 && isfinite(tau) &&
            isfinite(contribution);
        out[4] -= contribution;
        out[9] += fabs(contribution);
      }
    }
  }
  for (int index = 0; index < 2 * components; ++index)
    valid = valid && isfinite(out[index]);
  out[10] = valid ? 1.0 : 0.0;
}
__global__ void HashKernel(const double* values,int count,std::uint64_t* out)
{if(blockIdx.x||threadIdx.x)return;std::uint64_t h[4]={1469598103934665603ULL,1099511628211ULL^0x9e3779b97f4a7c15ULL,1469598103934665603ULL^0xd6e8feb86659fd93ULL,1099511628211ULL^0xa0761d6478bd642fULL};const unsigned char* bytes=reinterpret_cast<const unsigned char*>(values);for(int i=0;i<count*static_cast<int>(sizeof(double));++i)for(int lane=0;lane<4;++lane){h[lane]^=static_cast<std::uint64_t>(bytes[i]+17U*lane);h[lane]*=1099511628211ULL+2ULL*lane;}for(int lane=0;lane<4;++lane)out[lane]=h[lane];}

__global__ void StaticBoundaryClosureForTestsKernel(
    const CudaStaticBoundaryClosureCaseForTests input, double* const output)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const bool ready = StaticBoundaryExterior(
      input.kind, input.reference, input.pressure_out, input.gamma,
      input.interior, input.normal[0], input.normal[1], output);
  output[conserved_components] = ready ? 1.0 : 0.0;
}

bool LaunchOk() noexcept { return cudaPeekAtLastError() == cudaSuccess; }
int Blocks(const int n) noexcept { return (n + 127) / 128; }
} // namespace

bool CudaLaunchTicket(int* words,const CudaFailurePhase phase,std::uint64_t attempt,std::uint64_t generation,int device) noexcept
{TicketKernel<<<1,1>>>(words,static_cast<int>(phase),attempt,generation,device);return LaunchOk();}
bool CudaLaunchGatherLocalTrace(const int* map,int count,const double* source,double* destination) noexcept
{GatherKernel<<<Blocks(count),128>>>(map,count,source,destination);return LaunchOk();}
bool CudaLaunchAdmissibility(const CudaDeviceView& v,const CudaMeshShape& s) noexcept
{AdmissibilityKernel<<<Blocks(s.elements),128>>>(v,s);return LaunchOk();}
bool CudaLaunchSeedBorrowedLaggedLlfTau(const CudaDeviceView& v,
                                        const CudaMeshShape& s) noexcept
{SeedBorrowedLaggedLlfTauKernel<<<Blocks(s.elements),128>>>(v,s);return LaunchOk();}
bool CudaLaunchAssembly(const CudaDeviceView& v,const CudaMeshShape& s,double dt) noexcept
{if(s.borrowed_host_inputs)BorrowedLaggedLlfAssemblyKernel<<<Blocks(s.elements),128>>>(v,s,dt);else AssemblyKernel<<<Blocks(s.elements),128>>>(v,s,dt);return LaunchOk();}
bool CudaLaunchCondensation(const CudaDeviceView& v,const CudaMeshShape& s) noexcept
{CondenseKernel<<<Blocks(s.elements),128>>>(v,s);return LaunchOk();}
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
{RecoverKernel<<<Blocks(s.elements),128>>>(v,s);return LaunchOk();}
bool CudaLaunchTrial(const CudaDeviceView& v,const CudaMeshShape& s,double alpha) noexcept
{TrialKernel<<<Blocks(std::max(s.interior_count,s.true_trace_count)),128>>>(v,s,alpha);return LaunchOk();}
bool CudaLaunchLocalSquaredNorm(const double* a,int an,const double* b,int bn,double* out) noexcept
{NormKernel<<<1,1>>>(a,an,b,bn,out);return LaunchOk();}
bool CudaLaunchMaximumWave(const CudaDeviceView& v,const CudaMeshShape& s,
                           const double* interior,const double* trace,
                           double* out) noexcept
{MaximumWaveKernel<<<1,1>>>(v,s,interior,trace,out);return LaunchOk();}
bool CudaLaunchIgrConservation(const CudaDeviceView& v,
                               const CudaMeshShape& s,
                               const double dt, double* out) noexcept
{IgrConservationKernel<<<1,1>>>(v,s,dt,out);return LaunchOk();}
bool CudaLaunchStaticBoundaryClosureForTests(
    const CudaStaticBoundaryClosureCaseForTests& input,
    double* const result) noexcept
{StaticBoundaryClosureForTestsKernel<<<1,1>>>(input,result);return LaunchOk();}
bool CudaLaunchHash(const double* values,int count,std::uint64_t* words) noexcept
{HashKernel<<<1,1>>>(values,count,words);return LaunchOk();}

} // namespace detail
} // namespace dg::hdg::runtime::cuda_w8a
