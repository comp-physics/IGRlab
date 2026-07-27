#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include "persistent_smooth_euler_oracle.cuh"

#include <mfem.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#error "HDG-RUN-CUDA-01 leaf tests require a native SM120 image"
#endif

namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cuda_w8a;

struct TestModelTag { inline static constexpr std::string_view id =
    "euler.compressible.2d"; };
struct TestFormulationTag { inline static constexpr std::string_view id =
    "hdg.conventional"; };
struct TestTraceTag { inline static constexpr std::string_view id =
    "mfem.dg_interface.trace"; };
struct TestStabilizationTag { inline static constexpr std::string_view id =
    "tau.fixed"; };
struct TestTimeTag { inline static constexpr std::string_view id =
    "backward_euler"; };
struct TestRankOneBackendTag { inline static constexpr std::string_view id =
    "cuda.rank1.one_gpu"; };
struct TestRankTwoBackendTag { inline static constexpr std::string_view id =
    "cuda.rank2.one_gpu.host_staged"; };
struct TestPrecisionTag { inline static constexpr std::string_view id =
    "fp64"; };

template <typename Backend>
using TestSelection = StaticRunSelection<
    TestModelTag, TestFormulationTag, TestTraceTag, TestStabilizationTag,
    TestTimeTag, Backend, TestPrecisionTag>;

struct AuditedAllocator
{
  inline static std::array<CudaAllocationEvent, 8> events{};
  inline static std::size_t count = 0;
  static bool Allocate(std::size_t bytes, std::size_t alignment,
                       CudaAllocationSite site, void*& pointer) noexcept
  {
    const bool result = UmpireDeviceAllocatorPolicy::Allocate(
        bytes, alignment, site, pointer);
    if (count < events.size()) events[count++] =
        {site, bytes, alignment, reinterpret_cast<std::uintptr_t>(pointer),
         0, true};
    return result;
  }
  static void Deallocate(void* pointer, CudaAllocationSite site) noexcept
  {
    if (count < events.size()) events[count++] =
        {site, 0, 0, reinterpret_cast<std::uintptr_t>(pointer), 0, false};
    UmpireDeviceAllocatorPolicy::Deallocate(pointer, site);
  }
  static std::uint64_t AllocationEvents() noexcept
  { return UmpireDeviceAllocatorPolicy::AllocationEvents(); }
  static std::uint64_t DeallocationEvents() noexcept
  { return UmpireDeviceAllocatorPolicy::DeallocationEvents(); }
  static bool DevicePoolPolicyValid() noexcept
  { return UmpireDeviceAllocatorPolicy::DevicePoolPolicyValid(); }
};

struct AuditedTransport
{
  inline static std::array<CudaTransferEvent, 16384> events{};
  inline static std::size_t count = 0;
  inline static CudaPolicyContext context{};
  inline static CudaTransferTotals totals{};
  static void BeginAttempt(const CudaPolicyContext& value) noexcept
  { context=value; totals={}; count=0; CudaTransportPolicy::BeginAttempt(value); }
  static void SetPhase(CudaFailurePhase phase,std::uint64_t generation) noexcept
  {context.phase=phase;context.generation=generation;CudaTransportPolicy::SetPhase(phase,generation);}
  static void Record(CudaTransferDirection direction,CudaTransferCategory category,
                     CudaTransferSite site,const void* source,void* destination,
                     std::size_t bytes,bool success) noexcept
  {if(count<events.size())events[count++]={direction,category,site,context,success?bytes:0,reinterpret_cast<std::uintptr_t>(source),reinterpret_cast<std::uintptr_t>(destination)};totals.event_count++;totals.complete=totals.complete&&success;if(success){if(direction==CudaTransferDirection::host_to_device)totals.host_to_device_bytes+=bytes;else if(direction==CudaTransferDirection::device_to_host)totals.device_to_host_bytes+=bytes;else if(direction==CudaTransferDirection::device_to_device)totals.device_to_device_bytes+=bytes;else totals.mpi_staging_bytes+=bytes;}}
  static bool HostToDevice(void* d,const void* s,std::size_t n,CudaTransferCategory c,CudaTransferSite site) noexcept
  {const bool ok=CudaTransportPolicy::HostToDevice(d,s,n,c,site);Record(CudaTransferDirection::host_to_device,c,site,s,d,n,ok);return ok;}
  static bool DeviceToHost(void* d,const void* s,std::size_t n,CudaTransferCategory c,CudaTransferSite site) noexcept
  {const bool ok=CudaTransportPolicy::DeviceToHost(d,s,n,c,site);Record(CudaTransferDirection::device_to_host,c,site,s,d,n,ok);return ok;}
  static bool DeviceToDevice(void* d,const void* s,std::size_t n,CudaTransferCategory c,CudaTransferSite site) noexcept
  {const bool ok=CudaTransportPolicy::DeviceToDevice(d,s,n,c,site);Record(CudaTransferDirection::device_to_device,c,site,s,d,n,ok);return ok;}
  static void RecordMpiStaging(std::size_t n,CudaTransferSite site) noexcept
  {CudaTransportPolicy::RecordMpiStaging(n,site);Record(CudaTransferDirection::mpi_host_staging,CudaTransferCategory::host_staged_trace_exchange,site,nullptr,nullptr,n,true);}
  static bool Synchronize(CudaTransferSite site) noexcept
  {const bool ok=CudaTransportPolicy::Synchronize(site);totals.synchronization_events++;totals.complete=totals.complete&&ok;return ok;}
  static CudaTransferTotals Accounting() noexcept {return totals;}
};

bool Global(const bool local)
{int value=local?1:0,all=0;return MPI_Allreduce(&value,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD)==MPI_SUCCESS&&all==1;}

bool AccountingExact(const CudaStepEvidence& evidence,const int ranks)
{
  if(evidence.device_ticket_count==0)return false;
  for(std::size_t i=0;i<evidence.device_ticket_count;++i)
    if(!evidence.device_tickets[i].kernel_executed||
       evidence.device_tickets[i].compiled_architecture!=1200||
       evidence.device_tickets[i].attempt_sequence!=evidence.attempt.sequence||
       evidence.device_tickets[i].generation!=
           evidence.attempt.committed_generation+1)return false;
  std::uint64_t h2d=0,d2h=0,d2d=0,mpi=0;
  for(std::size_t i=0;i<AuditedTransport::count;++i)
  {const auto& event=AuditedTransport::events[i];if(event.context.attempt_sequence!=evidence.attempt.sequence||event.context.generation!=evidence.attempt.committed_generation+1)return false;switch(event.direction){case CudaTransferDirection::host_to_device:h2d+=event.bytes;break;case CudaTransferDirection::device_to_host:d2h+=event.bytes;break;case CudaTransferDirection::device_to_device:d2d+=event.bytes;break;case CudaTransferDirection::mpi_host_staging:mpi+=event.bytes;break;}}
  return h2d==evidence.transfers.host_to_device_bytes&&
      d2h==evidence.transfers.device_to_host_bytes&&
      d2d==evidence.transfers.device_to_device_bytes&&
      mpi==evidence.transfers.mpi_staging_bytes&&
      evidence.transfers.event_count==AuditedTransport::count&&
      evidence.transfers.synchronization_events>0&&
      evidence.transfers.complete&&
      ((ranks==1&&mpi==0)||(ranks==2&&mpi>0));
}

template <typename Backend>
bool RunLifecycle(const int ranks)
{
  CudaConfig config{};config.expected_rank_count=ranks;
  config.require_host_staged_rank2=ranks==2;config.capture_oracle=true;
  config.injection.enabled=true;config.injection.target_step=2;
  config.injection.source_rank=0;
  PersistentCudaSmoothEulerEngine<TestSelection<Backend>,AuditedAllocator,
      AuditedTransport,CudaMpiCollectivePolicy> engine(config);
  CudaRunEvidence setup{};
  if(!engine.Initialize(setup)||!setup.mfem_device_valid||
      !setup.hypre_cuda_umpire_device||!setup.same_physical_gpu||
      !setup.signed_owner_ghost_semantics||
      (ranks==2&&!setup.host_staged_rank2))return false;
  std::array<CudaStepEvidence,4> attempts{};int cursor=0;
  int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  for(int target=1;target<=3;++target)
  {
    CudaPrecommitSnapshot snapshot{};CudaStepEvidence evidence{};
    if(!engine.PrepareStep(false,snapshot,evidence)||!snapshot.numerical_candidate_ready||
       snapshot.target.step!=static_cast<std::uint64_t>(target)||
       !snapshot.staging_must_precede_finalize||!snapshot.finalize_has_no_fallible_posthook)return false;
    bool checkpoint_staged=true,output_staged=true;
    if(target==2)
    {
      const StateHash candidate_interior=snapshot.candidate_interior_hash;
      checkpoint_staged=false;output_staged=false;
      if(!engine.AbortPrepared(rank==0?FailureReason::backend_failure:
                                      FailureReason::none,evidence)||
         !evidence.retry_available||evidence.committed||
         evidence.phase!=CudaFailurePhase::after_recovery_before_commit||
         evidence.source_rank!=0||!evidence.collective_consensus||
         !evidence.committed_bytes_preserved||!evidence.coordinates_preserved||
         checkpoint_staged||output_staged||!AccountingExact(evidence,ranks))return false;
      attempts[cursor++]=evidence;
      CudaPrecommitSnapshot retry_snapshot{};
      if(!engine.PrepareStep(true,retry_snapshot,evidence)||
         retry_snapshot.attempt.retry_index!=1||
         retry_snapshot.candidate_interior_hash!=candidate_interior||
         !engine.MarkPreparedStagingReady(retry_snapshot.attempt)||
         !engine.FinalizePrepared(evidence)||!evidence.committed)return false;
    }
    else if(!checkpoint_staged||!output_staged||
            !engine.MarkPreparedStagingReady(snapshot.attempt)||
            !engine.FinalizePrepared(evidence)||!evidence.committed)return false;
    if(!AccountingExact(evidence,ranks)||!evidence.products.CoupledAndOrdered()||
       !evidence.no_fallback||!evidence.device_resident||
       !evidence.collective_consensus||
       evidence.after.step!=static_cast<std::uint64_t>(target)||
       std::abs(evidence.after.time-fixed_dt*target)>1e-14)return false;
    attempts[cursor++]=evidence;
  }
  const auto* contract=engine.contract();
  if(contract==nullptr||contract->coordinates().step!=3||
      contract->coordinates().generation!=3||
      contract->coordinates().cursor!=4||
      std::abs(contract->coordinates().time-0.15)>1e-14||cursor!=4||
      !attempts[1].retry_available||!attempts[2].committed)return false;
  CudaOraclePod pod{};
  if(!engine.ExportOracle(pod)||!cuda_w8a_test_oracle::Verify(pod))return false;
  const CudaReuseEvidence reuse=setup.reuse;
  const CudaReuseEvidence final_reuse = [&](){CudaRunEvidence probe{};return engine.contract()!=nullptr?setup.reuse:probe.reuse;}();
  (void)reuse;(void)final_reuse;
  return AuditedAllocator::AllocationEvents()==
      setup.reuse.baseline_allocation_events&&
      contract->telemetry().out_of_order_product_issues_rejected==0&&
      contract->telemetry().last_attempt.prohibited.fallback_events==0;
}

template <typename Backend>
bool RunTokenNegative(const int ranks,CudaTokenFault fault,
                      FailureReason expected)
{
  CudaConfig config{};config.expected_rank_count=ranks;
  config.require_host_staged_rank2=ranks==2;config.injection.token_fault=fault;
  PersistentCudaSmoothEulerEngine<TestSelection<Backend>,AuditedAllocator,
      AuditedTransport,CudaMpiCollectivePolicy> engine(config);
  CudaRunEvidence setup{};if(!engine.Initialize(setup))return false;
  CudaPrecommitSnapshot snapshot{};CudaStepEvidence evidence{};
  if(!engine.PrepareStep(false,snapshot,evidence))return false;
  if(!engine.MarkPreparedStagingReady(snapshot.attempt))return false;
  const bool progressed=engine.FinalizePrepared(evidence);
  const auto* contract=engine.contract();
  if(fault==CudaTokenFault::issue_matrix_before_residual)
    return progressed&&contract!=nullptr&&
        contract->telemetry().out_of_order_product_issues_rejected==1;
  return progressed&&contract!=nullptr&&evidence.contract_reason==expected&&
      evidence.retry_available&&!evidence.committed&&
      evidence.committed_bytes_preserved&&
      contract->telemetry().last_attempt.prohibited.fallback_events==0;
}
} // namespace

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cuda");(void)device;
  int ranks=0;MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  const bool token_mode=argc==2&&std::string_view(argv[1])=="--token-negatives";
  bool pass=false;
  if(ranks==1)
    pass=token_mode?(RunTokenNegative<TestRankOneBackendTag>(1,CudaTokenFault::omit_solution,FailureReason::missing_product)&&RunTokenNegative<TestRankOneBackendTag>(1,CudaTokenFault::stale_attempt,FailureReason::missing_product)&&RunTokenNegative<TestRankOneBackendTag>(1,CudaTokenFault::issue_matrix_before_residual,FailureReason::none)):RunLifecycle<TestRankOneBackendTag>(1);
  else if(ranks==2)
    pass=token_mode?RunTokenNegative<TestRankTwoBackendTag>(2,CudaTokenFault::omit_solution,FailureReason::missing_product):RunLifecycle<TestRankTwoBackendTag>(2);
  if(!Global(pass)){int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);if(rank==0)std::cerr<<"persistent CUDA W8A leaf audit failed\n";return 1;}
  return 0;
}
