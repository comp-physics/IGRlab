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
  if (transport_ledger.count >= transport_ledger.events.size())
  {
    transport_ledger.overflow = true;
    ++transport_ledger.dropped_count;
    totals.complete = false;
    transport_attempt_totals.complete = false;
    return;
  }
  const std::size_t index = transport_ledger.count++;
  const auto source_token = reinterpret_cast<std::uintptr_t>(source);
  const auto destination_token = reinterpret_cast<std::uintptr_t>(destination);
  const std::size_t exact_source_capacity =
      source_capacity == 0 ? bytes : source_capacity;
  const std::size_t exact_destination_capacity =
      destination_capacity == 0 ? bytes : destination_capacity;
  transport_ledger.events[index] = {
      ordinal, direction, category, site, transport_context,
      static_cast<std::uint64_t>(bytes), source_token,
      destination_token,
      static_cast<std::uint64_t>(exact_source_capacity),
      static_cast<std::uint64_t>(exact_destination_capacity),
      EndpointToken(direction, category, site, transport_context, ordinal,
                    source_token, destination_token, bytes,
                    exact_source_capacity, exact_destination_capacity,
                    success),
      success};
  transport_ledger.ordered_digest ^=
      transport_ledger.events[index].endpoint_token;
  transport_ledger.ordered_digest *= 1099511628211ULL;
}

void RecordAllocation(const CudaAllocationSite site,
                      const std::size_t bytes,
                      const std::size_t alignment,
                      void* pointer,
                      const bool allocation,
                      const bool success,
                      const std::uint64_t retained_generation = 0) noexcept
{
  ++allocation_ledger.observed_count;
  if (allocation_ledger.count >= allocation_ledger.events.size())
  {
    allocation_ledger.overflow = true;
    ++allocation_ledger.dropped_count;
    return;
  }
  const std::size_t index = allocation_ledger.count++;
  const std::uint64_t ordinal = ++allocation_ordinal;
  const std::uint64_t generation = allocation && success
      ? ++allocation_generation : retained_generation;
  allocation_ledger.events[index] = {
      ordinal, site, transport_context, static_cast<std::uint64_t>(bytes),
      static_cast<std::uint64_t>(alignment),
      static_cast<std::uint64_t>(bytes),
      reinterpret_cast<std::uintptr_t>(pointer), generation, allocation,
      success};
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
                             std::uint32_t& participating) noexcept
{
  int size = 0;
  std::uint64_t minimum = local;
  std::uint64_t maximum = local;
  if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS || size <= 0 ||
      MPI_Allreduce(&local, &minimum, 1, MPI_UINT64_T, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&local, &maximum, 1, MPI_UINT64_T, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
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

bool ReduceMask(const std::uint64_t local, std::uint64_t& global) noexcept
{
  global = local;
  return MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_BOR,
                       MPI_COMM_WORLD) == MPI_SUCCESS;
}

bool AllRanksReady(const bool local) noexcept
{
  const int local_value = local ? 1 : 0;
  int global_value = 0;
  return MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && global_value == 1;
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
  std::uint64_t original_bytes = 0;
  std::uint64_t original_alignment = 0;
  std::uint64_t original_generation = 0;
  const auto token = reinterpret_cast<std::uintptr_t>(pointer);
  for (std::size_t index = allocation_ledger.count; index > 0; --index)
  {
    const AllocationEventV1& event = allocation_ledger.events[index - 1];
    if (event.allocation && event.success && event.storage_token == token)
    {
      original_bytes = event.bytes;
      original_alignment = event.alignment;
      original_generation = event.generation;
      break;
    }
  }
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
  RecordAllocation(site, static_cast<std::size_t>(original_bytes),
                   static_cast<std::size_t>(original_alignment), pointer,
                   false, success, original_generation);
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
  transport_ledger = {};
  transport_attempt_totals = {};
}

void CudaTransportPolicy::BeginAttempt(
    const CudaPolicyContext& context) noexcept
{
  transport_context = context;
  transport_attempt_totals = {};
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

CollectiveAdmissionReport CudaMpiCollectivePolicy::Admit(
    const AttemptCollectiveContext& context, const bool prepared,
    FailureReason reason, const std::uint32_t expected) noexcept
{
  CollectiveAdmissionReport report{};
  report.context = context;
  if (!prepared && reason == FailureReason::none)
    reason = FailureReason::backend_failure;
  report.identity_consistent = CollectiveHashConsensus(
      ContextHash(context), expected, report.participating_ranks);
  int local_prepared = prepared ? 1 : 0;
  int all_prepared = local_prepared;
  const bool reduced_prepared = MPI_Allreduce(
      &local_prepared, &all_prepared, 1, MPI_INT, MPI_MIN,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  report.all_prepared = reduced_prepared && all_prepared == 1;
  const std::uint64_t local_mask = FailureBit(reason);
  report.failure_consistent = ReduceMask(local_mask, report.failure_mask);
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
    FailureReason reason, const std::uint32_t expected) noexcept
{
  CollectiveGateReport report{};
  report.context = context;
  report.identity_consistent = CollectiveHashConsensus(
      ContextHash(context), expected, report.participating_ranks);
  const int local[6] = {gates.finite, gates.admissible, gates.recovery,
                        gates.true_residual, gates.telemetry, gates.success};
  int global[6] = {};
  const bool gate_ok = MPI_Allreduce(local, global, 6, MPI_INT, MPI_MIN,
                                     MPI_COMM_WORLD) == MPI_SUCCESS;
  report.gates = {global[0] == 1, global[1] == 1, global[2] == 1,
                  global[3] == 1, global[4] == 1, global[5] == 1};
  if ((!gate_ok || !gates.All()) && reason == FailureReason::none)
    reason = FailureReason::success_gate;
  report.failure_consistent = ReduceMask(FailureBit(reason),
                                         report.failure_mask);
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
    const std::uint32_t expected) noexcept
{
  CollectiveRestoreReport report{};
  report.context = context;
  report.identity_consistent = CollectiveHashConsensus(
      ContextHash(context), expected, report.participating_ranks);
  const auto index = static_cast<unsigned>(compatibility);
  const int local_invalid =
      index >= static_cast<unsigned>(RestartCompatibility::count) ? 1 : 0;
  int any_invalid = local_invalid;
  const bool invalid_reduced = MPI_Allreduce(
      &local_invalid, &any_invalid, 1, MPI_INT, MPI_MAX,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const std::uint64_t local_compatibility =
      compatibility == RestartCompatibility::compatible ? 0ULL :
      (index < static_cast<unsigned>(RestartCompatibility::count) &&
       index < 64 ? 1ULL << index : 0ULL);
  const bool compatibility_reduced =
      ReduceMask(local_compatibility, report.compatibility_mask);
  const bool failure_reduced = ReduceMask(FailureBit(reason),
                                          report.failure_mask);
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
  if (!true_system_diagnostic_active) { return; }
  const int hypre_error = HYPRE_GetError();
  const cudaError_t cuda_error = cudaPeekAtLastError();
  std::fprintf(stderr,
      "CUDA_TRUE_SYSTEM rank=%d stage=%s edge=%s result=%u "
      "generation=%llu hypre=%d cuda=%d\n",
      true_system_diagnostic_rank, stage, edge, result ? 1U : 0U,
      static_cast<unsigned long long>(generation), hypre_error,
      static_cast<int>(cuda_error));
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
    if (owned_count < 0 || owned_count > maximum_true_trace_dofs ||
        diag_nonzeros < 0 || offd_nonzeros < 0 ||
        diag_nonzeros > maximum_true_matrix_entries ||
        offd_nonzeros > maximum_true_matrix_entries - diag_nonzeros)
      return false;
    structure = {};
    structure.rows = owned_count;
    structure.diag_nonzeros = static_cast<int>(diag_nonzeros);
    structure.offd_nonzeros = static_cast<int>(offd_nonzeros);
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
    std::array<HYPRE_Int, maximum_true_matrix_entries> offd_columns{};
    std::array<HYPRE_BigInt, maximum_true_matrix_entries> offd_global{};
    const HYPRE_Int raw_offd_column_count = hypre_CSRMatrixNumCols(offd);
    if (raw_offd_column_count < 0 ||
        raw_offd_column_count > maximum_true_matrix_entries)
      return false;
    const int offd_column_count = static_cast<int>(raw_offd_column_count);
    if ((structure.offd_nonzeros > 0 && offd_column_count == 0) ||
        (offd_column_count > 0 && cmap == nullptr))
      return false;
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
                                  const std::uint64_t matrix_generation,
                                  CudaHypreSetupOrderEvidence& evidence)
    noexcept
{
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
    gmres.SetTol(1.0e-13);
    gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(400);
    gmres.SetKDim(ranks == 1 ? matrix.Height() :
        static_cast<int>(matrix.GetGlobalNumRows()));
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
                                     matrix_generation, setup_order);
  }
  catch (...) { policy_ready = false; }
  if (!AllRanksReady(policy_ready)) { return false; }
  bool vectors_ready = false;
  try
  {
    solution = 0.0;
    solution.HypreReadWrite();
    vectors_ready = true;
  }
  catch (...) { vectors_ready = false; }
  if (!AllRanksReady(vectors_ready)) { return false; }
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
        diagnostics.reported_residual <= 1.0e-10;
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
    if (!Admissible(state)) { view.status[element] = -1.0; return; }
  }
  for (int face = 0; face < faces_per_element; ++face)
    for (int q = 0; q < shape.face_points; ++q)
    {
      const int qp = (element*faces_per_element+face)*shape.face_points+q;
      const int offset =
          face * conserved_components * shape.scalar_face_dofs;
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
      if (!Admissible(interior) || !Admissible(trace))
      { view.status[element] = -2.0; return; }
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
    double source[4], fx[4], fy[4], jx[16], jy[16];
    if (!(w>0) || !Source(v.volume_position[qp*2],v.volume_position[qp*2+1],source) ||
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

__device__ bool FactorSolve(double* lu, double* pivots, const int n,
                            double* rhs, const int columns)
{
  double scale=0; for(int i=0;i<n*n;++i) scale=fmax(scale,fabs(lu[i]));
  for(int k=0;k<n;++k)
  {
    int pivot=k; for(int r=k+1;r<n;++r)
      if(fabs(lu[r*n+k])>fabs(lu[pivot*n+k])) pivot=r;
    pivots[k]=pivot;
    if(!(fabs(lu[pivot*n+k])>64.0*2.2204460492503131e-16*scale)) return false;
    if(pivot!=k) for(int c=0;c<n;++c)
    {double x=lu[k*n+c];lu[k*n+c]=lu[pivot*n+c];lu[pivot*n+c]=x;}
    for(int r=k+1;r<n;++r){lu[r*n+k]/=lu[k*n+k];for(int c=k+1;c<n;++c)lu[r*n+c]-=lu[r*n+k]*lu[k*n+c];}
  }
  // Factorization swaps complete LU rows, including previously stored lower
  // columns.  Apply the retained pivot sequence completely before forward
  // substitution so delayed pivots cannot permute an already-eliminated RHS.
  for(int k=0;k<n;++k){const int p=static_cast<int>(pivots[k]);if(p!=k)for(int c=0;c<columns;++c){double x=rhs[k*columns+c];rhs[k*columns+c]=rhs[p*columns+c];rhs[p*columns+c]=x;}}
  for(int k=0;k<n;++k)for(int r=k+1;r<n;++r)for(int c=0;c<columns;++c)rhs[r*columns+c]-=lu[r*n+k]*rhs[k*columns+c];
  for(int k=n-1;k>=0;--k)for(int c=0;c<columns;++c){for(int j=k+1;j<n;++j)rhs[k*columns+c]-=lu[k*n+j]*rhs[j*columns+c];rhs[k*columns+c]/=lu[k*n+k];}
  return true;
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
  if(!FactorSolve(lu,p,ni,w,cols)){v.status[e]=-1;return;}
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
{const int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=s.elements)return;const int ni=s.interior_dofs,nt=s.trace_dofs;double* x=v.recovered_interior+e*ni;for(int r=0;r<ni;++r){double q=v.f[e*ni+r];for(int c=0;c<nt;++c)q-=v.b[(e*ni+r)*nt+c]*v.trace_increment_local[e*nt+c];x[r]=q;}double* lu=v.lu+e*ni*ni;double* p=v.pivots+e*ni;for(int k=0;k<ni;++k){const int pivot=static_cast<int>(p[k]);if(pivot!=k){double q=x[k];x[k]=x[pivot];x[pivot]=q;}}for(int k=0;k<ni;++k)for(int r=k+1;r<ni;++r)x[r]-=lu[r*ni+k]*x[k];for(int k=ni-1;k>=0;--k){for(int j=k+1;j<ni;++j)x[k]-=lu[k*ni+j]*x[j];x[k]/=lu[k*ni+k];}v.status[e]=1;}
__global__ void TrialKernel(CudaDeviceView v,CudaMeshShape s,double alpha)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<s.interior_count)v.trial_interior[i]=v.base_interior[i]+alpha*v.recovered_interior[i];if(i<s.true_trace_count)v.trial_true_trace[i]=v.base_true_trace[i]+alpha*v.true_solution[i];}
__global__ void NormKernel(const double* a,int an,const double* b,int bn,double* out)
{if(blockIdx.x||threadIdx.x)return;double sum=0;for(int i=0;i<an;++i)sum+=a[i]*a[i];for(int i=0;i<bn;++i)sum+=b[i]*b[i];out[0]=sum;}
__global__ void HashKernel(const double* values,int count,std::uint64_t* out)
{if(blockIdx.x||threadIdx.x)return;std::uint64_t h[4]={1469598103934665603ULL,1099511628211ULL^0x9e3779b97f4a7c15ULL,1469598103934665603ULL^0xd6e8feb86659fd93ULL,1099511628211ULL^0xa0761d6478bd642fULL};const unsigned char* bytes=reinterpret_cast<const unsigned char*>(values);for(int i=0;i<count*static_cast<int>(sizeof(double));++i)for(int lane=0;lane<4;++lane){h[lane]^=static_cast<std::uint64_t>(bytes[i]+17U*lane);h[lane]*=1099511628211ULL+2ULL*lane;}for(int lane=0;lane<4;++lane)out[lane]=h[lane];}

bool LaunchOk() noexcept { return cudaPeekAtLastError() == cudaSuccess; }
int Blocks(const int n) noexcept { return (n + 127) / 128; }
} // namespace

bool CudaLaunchTicket(int* words,const CudaFailurePhase phase,std::uint64_t attempt,std::uint64_t generation,int device) noexcept
{TicketKernel<<<1,1>>>(words,static_cast<int>(phase),attempt,generation,device);return LaunchOk();}
bool CudaLaunchGatherLocalTrace(const int* map,int count,const double* source,double* destination) noexcept
{GatherKernel<<<Blocks(count),128>>>(map,count,source,destination);return LaunchOk();}
bool CudaLaunchAdmissibility(const CudaDeviceView& v,const CudaMeshShape& s) noexcept
{AdmissibilityKernel<<<Blocks(s.elements),128>>>(v,s);return LaunchOk();}
bool CudaLaunchAssembly(const CudaDeviceView& v,const CudaMeshShape& s,double dt) noexcept
{AssemblyKernel<<<Blocks(s.elements),128>>>(v,s,dt);return LaunchOk();}
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
bool CudaLaunchHash(const double* values,int count,std::uint64_t* words) noexcept
{HashKernel<<<1,1>>>(values,count,words);return LaunchOk();}

} // namespace detail
} // namespace dg::hdg::runtime::cuda_w8a
