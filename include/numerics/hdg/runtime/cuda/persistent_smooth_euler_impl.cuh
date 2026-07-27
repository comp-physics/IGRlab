/* Included by persistent_smooth_euler.cuh inside dg::hdg::runtime::cuda_w8a. */

namespace detail
{

static inline void ReportCudaSetupFailure(const int rank,
                                          const char* const stage) noexcept
{
  if (rank == 0)
    std::fprintf(stderr, "CUDA_SETUP_FAIL stage=%s\n", stage);
}

// Stable private backend-admission mask: bits 0..17 are rank count,
// cudaSetDevice, cudaGetDevice, selected device zero, MFEM configured,
// MFEM CUDA allowed, MFEM debug-device disabled, clean pre-init HYPRE error
// state, normal Hypre::Init return, zero error captured immediately after
// Init, clean policy-query start, Hypre GPU selection, HYPRE memory query,
// HYPRE execution query, device memory, device execution, clean post-policy
// HYPRE error state, and Umpire pools.
enum CudaBackendFailureBit : std::uint64_t
{
  backend_rank_count_failure = std::uint64_t{1} << 0U,
  backend_cuda_set_device_failure = std::uint64_t{1} << 1U,
  backend_cuda_get_device_failure = std::uint64_t{1} << 2U,
  backend_selected_device_failure = std::uint64_t{1} << 3U,
  backend_mfem_configured_failure = std::uint64_t{1} << 4U,
  backend_mfem_cuda_failure = std::uint64_t{1} << 5U,
  backend_mfem_debug_failure = std::uint64_t{1} << 6U,
  backend_hypre_preinit_error_failure = std::uint64_t{1} << 7U,
  backend_hypre_initialize_return_failure = std::uint64_t{1} << 8U,
  backend_hypre_initialize_error_failure = std::uint64_t{1} << 9U,
  backend_hypre_policy_clear_failure = std::uint64_t{1} << 10U,
  backend_hypre_gpu_failure = std::uint64_t{1} << 11U,
  backend_hypre_memory_query_failure = std::uint64_t{1} << 12U,
  backend_hypre_execution_query_failure = std::uint64_t{1} << 13U,
  backend_hypre_memory_device_failure = std::uint64_t{1} << 14U,
  backend_hypre_execution_device_failure = std::uint64_t{1} << 15U,
  backend_hypre_policy_error_failure = std::uint64_t{1} << 16U,
  backend_umpire_pools_failure = std::uint64_t{1} << 17U
};

static inline void ReportCudaBackendFailure(
    const int rank, const std::uint64_t failure_mask) noexcept
{
  if (rank == 0)
  {
    std::fprintf(stderr, "CUDA_BACKEND_FAIL rank0_mask=0x%llx\n",
                 static_cast<unsigned long long>(failure_mask));
  }
}

inline constexpr std::size_t persistent_double_capacity = 220000;
inline constexpr std::size_t persistent_integer_capacity = 120000;

struct CudaTrueMatrixStructure
{
  int rows = 0;
  int diag_nonzeros = 0;
  int offd_nonzeros = 0;
  std::array<int, maximum_true_trace_dofs + 1> diag_offsets{};
  std::array<int, maximum_true_matrix_entries> diag_columns{};
  std::array<int, maximum_true_trace_dofs + 1> offd_offsets{};
  std::array<std::int64_t, maximum_true_matrix_entries> offd_columns{};
};

struct CudaTrueMatrixDeviceView
{
  double* diag_values = nullptr;
  double* offd_values = nullptr;
  int diag_nonzeros = 0;
  int offd_nonzeros = 0;
};

struct CudaDeviceView
{
  double* slot_interior[2]{};
  double* old_interior = nullptr;
  double* work_interior = nullptr;
  double* initial_current_interior = nullptr;
  double* base_interior = nullptr;
  double* trial_interior = nullptr;
  double* delta_interior = nullptr;
  double* recovered_interior = nullptr;
  double* work_true_trace = nullptr;
  double* initial_current_true_trace = nullptr;
  double* base_true_trace = nullptr;
  double* trial_true_trace = nullptr;
  double* distributed_local_trace = nullptr;
  double* current_local_trace = nullptr;
  double* base_local_trace = nullptr;
  double* trace_increment_local = nullptr;
  double* residual_r = nullptr;
  double* residual_g = nullptr;
  double* a = nullptr;
  double* b = nullptr;
  double* c = nullptr;
  double* d = nullptr;
  double* f = nullptr;
  double* g = nullptr;
  double* lu = nullptr;
  double* pivots = nullptr;
  double* solve_workspace = nullptr;
  double* schur = nullptr;
  double* condensed_rhs = nullptr;
  double* true_residual = nullptr;
  double* true_rhs_scratch = nullptr;
  double* true_solution = nullptr;
  double* remote_matrix_send = nullptr;
  double* remote_matrix_receive = nullptr;
  double* remote_vector_send = nullptr;
  double* remote_vector_receive = nullptr;
  double* volume_shape = nullptr;
  double* volume_derivative = nullptr;
  double* volume_weight = nullptr;
  double* volume_position = nullptr;
  double* face_interior_shape = nullptr;
  double* face_trace_shape = nullptr;
  double* face_weight = nullptr;
  double* face_normal = nullptr;
  double* status = nullptr;
  double* scalar = nullptr;
  std::uint64_t* hash_words = nullptr;
  int* ticket_words = nullptr;
  int* signed_element_trace = nullptr;
  int* matrix_owned_source = nullptr;
  int* matrix_owned_section = nullptr;
  int* matrix_owned_target = nullptr;
  int* matrix_send_source = nullptr;
  int* matrix_receive_section = nullptr;
  int* matrix_receive_target = nullptr;
  int* vector_owned_source = nullptr;
  int* vector_owned_target = nullptr;
  int* vector_send_source = nullptr;
  int* vector_receive_target = nullptr;
};

struct CudaMeshShape
{
  int elements = 0;
  int interior_count = 0;
  int local_trace_count = 0;
  int true_trace_count = 0;
  int global_true_trace_count = 0;
  int volume_points = 0;
  int face_points = 0;
  int matrix_owned_count = 0;
  int matrix_send_count = 0;
  int matrix_receive_count = 0;
  int vector_owned_count = 0;
  int vector_send_count = 0;
  int vector_receive_count = 0;
};

enum class CudaHypreSetupAction : std::uint8_t
{
  bind_preconditioner,
  set_operator,
  ilu_type0,
  ilu_fill0,
  ilu_reorder0,
  ilu_maxiter1,
  ilu_tol0,
  ilu_print0,
  rebind_preconditioner,
  gmres_settings,
  mult
};

struct CudaHypreSetupOrderEvidence
{
  inline static constexpr std::size_t configured_action_count = 10;
  inline static constexpr std::size_t solved_action_count = 11;

  std::uint64_t matrix_generation = 0;
  std::array<CudaHypreSetupAction, solved_action_count> actions{};
  std::size_t action_count = 0;
  bool overflow = false;

  constexpr bool Record(const CudaHypreSetupAction action) noexcept
  {
    if (action_count >= actions.size())
    {
      overflow = true;
      return false;
    }
    actions[action_count++] = action;
    return true;
  }

  constexpr bool ConfigurationExactForGeneration(
      const std::uint64_t expected_generation) const noexcept
  {
    return action_count == configured_action_count &&
        PrefixExactForGeneration(expected_generation);
  }

  constexpr bool SolveExactForGeneration(
      const std::uint64_t expected_generation) const noexcept
  {
    return action_count == solved_action_count &&
        PrefixExactForGeneration(expected_generation) &&
        actions[configured_action_count] == CudaHypreSetupAction::mult;
  }

private:
  constexpr bool PrefixExactForGeneration(
      const std::uint64_t expected_generation) const noexcept
  {
    constexpr std::array<CudaHypreSetupAction, configured_action_count>
        expected{{
          CudaHypreSetupAction::bind_preconditioner,
          CudaHypreSetupAction::set_operator,
          CudaHypreSetupAction::ilu_type0,
          CudaHypreSetupAction::ilu_fill0,
          CudaHypreSetupAction::ilu_reorder0,
          CudaHypreSetupAction::ilu_maxiter1,
          CudaHypreSetupAction::ilu_tol0,
          CudaHypreSetupAction::ilu_print0,
          CudaHypreSetupAction::rebind_preconditioner,
          CudaHypreSetupAction::gmres_settings}};
    if (overflow || matrix_generation != expected_generation ||
        action_count < expected.size())
      return false;
    for (std::size_t action = 0; action < expected.size(); ++action)
      if (actions[action] != expected[action]) { return false; }
    return true;
  }
};

constexpr bool CudaTicketWordsValid(
    const std::array<int, 8>& words, const CudaFailurePhase phase,
    const std::uint64_t attempt, const std::uint64_t generation,
    const int device) noexcept
{
  return words[0] == 1 && words[1] == 1200 && words[2] == device &&
      words[3] == static_cast<int>(phase) &&
      static_cast<std::uint32_t>(words[4]) ==
          static_cast<std::uint32_t>(attempt) &&
      static_cast<std::uint32_t>(words[5]) ==
          static_cast<std::uint32_t>(attempt >> 32U) &&
      static_cast<std::uint32_t>(words[6]) ==
          static_cast<std::uint32_t>(generation) &&
      static_cast<std::uint32_t>(words[7]) ==
          static_cast<std::uint32_t>(generation >> 32U);
}

struct CudaSolveDiagnostics
{
  bool device_policy = false;
  bool configured_ilu0 = false;
  bool converged = false;
  bool finite = false;
  std::uint64_t matrix_generation = 0;
  int iterations = 0;
  double reported_residual = std::numeric_limits<double>::infinity();
};

using CudaCopyOutFunction = bool (*)(
    void*, const void*, std::size_t, CudaTransferCategory,
    CudaTransferSite, std::size_t, std::size_t) noexcept;
using CudaSynchronizeFunction = bool (*)(CudaTransferSite) noexcept;

bool CudaGetTrueMatrixStructure(mfem::HypreParMatrix& matrix,
                                const std::int64_t* owned_global,
                                int owned_count,
                                CudaTrueMatrixStructure& structure,
                                CudaCopyOutFunction copy_out,
                                CudaSynchronizeFunction synchronize) noexcept;
bool CudaGetTrueMatrixDeviceView(mfem::HypreParMatrix& matrix,
                                 CudaTrueMatrixDeviceView& view) noexcept;
bool CudaConfigurePersistentHypre(mfem::HypreParMatrix& matrix,
                                  mfem::HypreILU& ilu,
                                  mfem::HypreGMRES& gmres,
                                  int ranks,
                                  std::uint64_t matrix_generation,
                                  CudaHypreSetupOrderEvidence& evidence)
    noexcept;
bool CudaRefreshFactorAndSolve(mfem::HypreParMatrix& matrix,
                               mfem::HypreParVector& rhs,
                               mfem::HypreParVector& solution,
                               mfem::HypreILU& ilu,
                               mfem::HypreGMRES& gmres,
                               int ranks,
                               std::uint64_t matrix_generation,
                               CudaSolveDiagnostics& diagnostics) noexcept;
bool CudaHypreDevicePolicyValid(mfem::HypreParMatrix& matrix,
                                mfem::HypreParVector& rhs,
                                mfem::HypreParVector& solution) noexcept;

bool CudaLaunchTicket(int* words, CudaFailurePhase phase,
                      std::uint64_t attempt, std::uint64_t generation,
                      int device) noexcept;
bool CudaLaunchGatherLocalTrace(const int* signed_map, int count,
                                const double* true_trace,
                                double* local_trace) noexcept;
bool CudaLaunchAdmissibility(const CudaDeviceView& view,
                             const CudaMeshShape& shape) noexcept;
bool CudaLaunchAssembly(const CudaDeviceView& view,
                        const CudaMeshShape& shape, double dt) noexcept;
bool CudaLaunchCondensation(const CudaDeviceView& view,
                            const CudaMeshShape& shape) noexcept;
bool CudaLaunchZeroTrueSystem(const CudaTrueMatrixDeviceView& matrix,
                              double* rhs, int rhs_count) noexcept;
bool CudaLaunchAssembleOwnedMatrix(const CudaDeviceView& view,
                                   const CudaTrueMatrixDeviceView& matrix,
                                   const CudaMeshShape& shape) noexcept;
bool CudaLaunchGatherRemoteMatrix(const CudaDeviceView& view,
                                  const CudaMeshShape& shape) noexcept;
bool CudaLaunchScatterRemoteMatrix(const CudaDeviceView& view,
                                   const CudaTrueMatrixDeviceView& matrix,
                                   const CudaMeshShape& shape) noexcept;
bool CudaLaunchAssembleOwnedVector(const CudaDeviceView& view,
                                   const CudaMeshShape& shape,
                                   const double* element_vector,
                                   double* true_vector) noexcept;
bool CudaLaunchGatherRemoteVector(const CudaDeviceView& view,
                                  const CudaMeshShape& shape,
                                  const double* element_vector) noexcept;
bool CudaLaunchScatterRemoteVector(const CudaDeviceView& view,
                                   const CudaMeshShape& shape,
                                   double* true_vector) noexcept;
bool CudaLaunchRecover(const CudaDeviceView& view,
                       const CudaMeshShape& shape) noexcept;
bool CudaLaunchTrial(const CudaDeviceView& view,
                     const CudaMeshShape& shape, double alpha) noexcept;
bool CudaLaunchLocalSquaredNorm(const double* first, int first_count,
                                const double* second, int second_count,
                                double* result) noexcept;
bool CudaLaunchHash(const double* values, int count,
                    std::uint64_t* words) noexcept;

inline StateHash HashHostBytes(const void* data, const std::size_t bytes) noexcept
{
  StateHash result{};
  const auto* input = static_cast<const unsigned char*>(data);
  std::array<std::uint64_t, 4> hash{{
      1469598103934665603ULL,
      1099511628211ULL ^ 0x9e3779b97f4a7c15ULL,
      1469598103934665603ULL ^ 0xd6e8feb86659fd93ULL,
      1099511628211ULL ^ 0xa0761d6478bd642fULL}};
  for (std::size_t i = 0; i < bytes; ++i)
  {
    for (std::size_t lane = 0; lane < hash.size(); ++lane)
    {
      hash[lane] ^= static_cast<std::uint64_t>(input[i] + 17U * lane);
      hash[lane] *= 1099511628211ULL + 2ULL * lane;
    }
  }
  std::memcpy(result.bytes.data(), hash.data(), sizeof(hash));
  return result;
}

template <typename T>
inline T* TakeArena(T*& cursor, std::size_t count) noexcept
{
  T* result = cursor;
  cursor += count;
  return result;
}

template <typename AllocatorPolicy, typename TransportPolicy>
class CudaPersistentNumerics
{
public:
  using state_slot = CompleteStateSlot<CudaInteriorState, CudaTrueTraceState>;

  CudaPersistentNumerics() noexcept = default;
  ~CudaPersistentNumerics()
  {
    if (integer_arena_ != nullptr)
      AllocatorPolicy::Deallocate(integer_arena_,
                                  CudaAllocationSite::persistent_integer_arena);
    if (double_arena_ != nullptr)
      AllocatorPolicy::Deallocate(double_arena_,
                                  CudaAllocationSite::persistent_double_arena);
  }

  CudaPersistentNumerics(const CudaPersistentNumerics&) = delete;
  CudaPersistentNumerics& operator=(const CudaPersistentNumerics&) = delete;

  bool Initialize(const CudaConfig& config, CudaRunEvidence& evidence,
                  const RuntimeTupleDescriptor& tuple) noexcept
  {
    TransportPolicy::ResetLedger();
    CudaPolicyContext setup{};
    setup.phase = CudaFailurePhase::setup;
    TransportPolicy::BeginAttempt(setup);
    if (!CollectiveContinue(ConfigureDeviceAndRanks(config, evidence)))
    {
      ReportCudaSetupFailure(evidence.rank, "configure-device-ranks");
      return false;
    }
    setup.rank = rank_;
    TransportPolicy::BeginAttempt(setup);
    if (!CollectiveContinue(BuildTopology(config)))
    {
      ReportCudaSetupFailure(evidence.rank, "build-topology");
      return false;
    }
    const int local_ghost_count =
        static_cast<int>(exchange_plan_.map.GhostCount());
    std::array<int, 2> ghost_counts{};
    const bool ghost_census_ready = MPI_Allgather(
        &local_ghost_count, 1, MPI_INT, ghost_counts.data(), 1, MPI_INT,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    if (!CollectiveContinue(ghost_census_ready))
    {
      ReportCudaSetupFailure(evidence.rank, "ghost-census");
      return false;
    }
    host_exchange_bytes_ = ranks_ == 2
        ? static_cast<std::size_t>(ghost_counts[0] + ghost_counts[1]) *
            sizeof(double)
        : 0;
    if (!CollectiveContinue(AllocateArenas()))
    {
      ReportCudaSetupFailure(evidence.rank, "allocate-arenas");
      return false;
    }
    if (!CollectiveContinue(BindArenaViews()))
    {
      ReportCudaSetupFailure(evidence.rank, "bind-arena-views");
      return false;
    }
    if (!CollectiveContinue(UploadTopologyAndState()))
    {
      ReportCudaSetupFailure(evidence.rank, "upload-topology-state");
      return false;
    }
    if (!CollectiveContinue(BuildPersistentTrueSystem()))
    {
      ReportCudaSetupFailure(evidence.rank,
                             "build-persistent-true-system");
      return false;
    }
    if (!CollectiveContinue(BindStateSlots()))
    {
      ReportCudaSetupFailure(evidence.rank, "bind-state-slots");
      return false;
    }
    if (!CollectiveContinue(VerifyResidencyAndTicket(evidence)))
    {
      ReportCudaSetupFailure(evidence.rank, "residency-ticket");
      return false;
    }
    resources_.mfem_pin = StableIdentity::Copy(
        "d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec");
    local_mesh_hash_ = HashHostBytes(element_coordinate_keys_.data(),
        static_cast<std::size_t>(shape_.elements) * dimension *
            sizeof(std::int64_t));
    local_partition_hash_ = HashHostBytes(owned_global_true_.data(),
        static_cast<std::size_t>(shape_.true_trace_count) *
            sizeof(std::int64_t));
    local_topology_hash_ = HashHostBytes(signed_element_trace_.data(),
        static_cast<std::size_t>(shape_.elements) * trace_dofs_per_element *
            sizeof(int));
    if (!AggregateHash(local_mesh_hash_, resources_.mesh_hash) ||
        !AggregateHash(local_partition_hash_, resources_.partition_hash) ||
        !AggregateHash(local_topology_hash_, resources_.topology_hash))
    {
      ReportCudaSetupFailure(evidence.rank, "aggregate-hashes");
      return false;
    }
    resources_.rank_count = static_cast<std::uint32_t>(ranks_);
    resources_.polynomial_order = polynomial_order;
    resources_.layout = StableIdentity::Copy("mfem.byNODES.vdim4.p1.n3");
    resources_.reuse = {true,
                        true,
                        true,
                        false,
                        1,
                        1,
                        persistent_double_capacity * sizeof(double) +
                            persistent_integer_capacity * sizeof(int),
                        WorkspaceBytes()};
    tuple_ = tuple;
    baseline_allocations_ = AllocatorPolicy::AllocationEvents();
    baseline_w8_owned_allocation_calls_ =
        AllocatorPolicy::W8OwnedAllocationCalls();
    initialized_ = true;
    evidence.rank = rank_;
    evidence.ranks = ranks_;
    evidence.device = device_;
    evidence.physical_gpu_uuid = gpu_uuid_;
    evidence.same_physical_gpu = same_physical_gpu_;
    evidence.mfem_device_valid = device_resident_;
    evidence.hypre_cuda_umpire_device = hypre_policy_valid_;
    evidence.signed_owner_ghost_semantics = signed_owner_ghost_valid_;
    evidence.host_staged_rank2 = ranks_ == 1 || host_staged_rank2_;
    evidence.no_host_solver_fallback = no_fallback_;
    return true;
  }

  state_slot CommittedSlot() const noexcept { return committed_slot_; }
  state_slot CandidateSlot() const noexcept { return candidate_slot_; }
  state_slot& WorkspaceSlot() noexcept { return workspace_slot_; }
  const PersistentResourceDescriptor& Resources() const noexcept
  {
    return resources_;
  }
  int rank() const noexcept { return rank_; }
  bool NoFallback() const noexcept { return no_fallback_; }
  bool DeviceResident() const noexcept { return device_resident_; }
  const StateHash& LocalMeshHash() const noexcept { return local_mesh_hash_; }
  const StateHash& LocalPartitionHash() const noexcept
  { return local_partition_hash_; }
  const StateHash& LocalTopologyHash() const noexcept
  { return local_topology_hash_; }
  const std::array<CudaDeviceTicket, maximum_device_tickets>& Tickets()
      const noexcept { return tickets_; }
  std::size_t TicketCount() const noexcept { return ticket_count_; }

  bool CollectiveContinue(const bool local) const noexcept
  {
    int local_value = local ? 1 : 0;
    int global_value = 0;
    return MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                         MPI_COMM_WORLD) == MPI_SUCCESS && global_value == 1;
  }

  bool CollectiveCount(const bool local, int& count) const noexcept
  {
    const int local_value = local ? 1 : 0;
    count = 0;
    return MPI_Allreduce(&local_value, &count, 1, MPI_INT, MPI_SUM,
                         MPI_COMM_WORLD) == MPI_SUCCESS;
  }

  bool StageCheckpoint(const state_slot& frozen,
                       CudaSerializedCheckpointV1& checkpoint) noexcept
  {
    if (frozen.interior.coefficients == nullptr ||
        frozen.trace.coefficients == nullptr ||
        frozen.interior.count > checkpoint.interior.size() ||
        frozen.trace.count > checkpoint.owned_true_trace.size())
      return false;
    checkpoint.source_rank = rank_;
    checkpoint.rank_count = ranks_;
    checkpoint.local_mesh_hash = local_mesh_hash_;
    checkpoint.local_partition_hash = local_partition_hash_;
    checkpoint.local_topology_hash = local_topology_hash_;
    checkpoint.local_interior_hash = frozen.interior.local_hash;
    checkpoint.local_trace_hash = frozen.trace.local_hash;
    checkpoint.interior_count = frozen.interior.count;
    checkpoint.owned_true_trace_count = frozen.trace.count;
    checkpoint.interior_source_token = frozen.interior.storage_token;
    checkpoint.trace_source_token = frozen.trace.storage_token;
    checkpoint.staging_generation = frozen.interior.generation;
    const bool interior_copied = TransportPolicy::DeviceToHost(
            checkpoint.interior.data(), frozen.interior.coefficients,
            frozen.interior.count * sizeof(double),
            CudaTransferCategory::checkpoint_candidate_staging,
            CudaTransferSite::checkpoint_stage_interior,
            frozen.interior.count * sizeof(double),
            checkpoint.interior.size() * sizeof(double));
    const bool trace_copied = TransportPolicy::DeviceToHost(
            checkpoint.owned_true_trace.data(), frozen.trace.coefficients,
            frozen.trace.count * sizeof(double),
            CudaTransferCategory::checkpoint_candidate_staging,
            CudaTransferSite::checkpoint_stage_trace,
            frozen.trace.count * sizeof(double),
            checkpoint.owned_true_trace.size() * sizeof(double));
    const bool synchronized =
        TransportPolicy::Synchronize(CudaTransferSite::checkpoint_stage_trace);
    const bool copied = interior_copied && trace_copied && synchronized;
    if (!copied) { return false; }
    const StateHash interior = HashHostBytes(
        checkpoint.interior.data(),
        frozen.interior.count * sizeof(double));
    const StateHash trace = HashHostBytes(
        checkpoint.owned_true_trace.data(),
        frozen.trace.count * sizeof(double));
    checkpoint.checkpoint_staged = interior == frozen.interior.local_hash &&
        trace == frozen.trace.local_hash;
    return checkpoint.checkpoint_staged;
  }

  bool ImportCheckpoint(const CudaSerializedCheckpointV1& checkpoint,
                        state_slot& loaded) noexcept
  {
    const bool local_descriptor_valid = checkpoint.schema == 1 &&
        checkpoint.successful_commit &&
        checkpoint.rank_count == ranks_ && checkpoint.source_rank == rank_ &&
        checkpoint.local_mesh_hash == local_mesh_hash_ &&
        checkpoint.local_partition_hash == local_partition_hash_ &&
        checkpoint.local_topology_hash == local_topology_hash_ &&
        checkpoint.interior_count == loaded.interior.count &&
        checkpoint.owned_true_trace_count == loaded.trace.count &&
        checkpoint.interior_count <= checkpoint.interior.size() &&
        checkpoint.owned_true_trace_count <=
            checkpoint.owned_true_trace.size();
    if (!CollectiveContinue(local_descriptor_valid)) { return false; }
    const StateHash host_interior = HashHostBytes(
        checkpoint.interior.data(),
        checkpoint.interior_count * sizeof(double));
    const StateHash host_trace = HashHostBytes(
        checkpoint.owned_true_trace.data(),
        checkpoint.owned_true_trace_count * sizeof(double));
    const bool host_hashes_valid =
        host_interior == checkpoint.local_interior_hash &&
        host_trace == checkpoint.local_trace_hash;
    if (!CollectiveContinue(host_hashes_valid)) { return false; }
    const bool interior_copied = TransportPolicy::HostToDevice(
            loaded.interior.coefficients, checkpoint.interior.data(),
            checkpoint.interior_count * sizeof(double),
            CudaTransferCategory::checkpoint_restore_import,
            CudaTransferSite::checkpoint_import_interior,
            checkpoint.interior.size() * sizeof(double),
            loaded.interior.count * sizeof(double));
    const bool trace_copied = TransportPolicy::HostToDevice(
            loaded.trace.coefficients, checkpoint.owned_true_trace.data(),
            checkpoint.owned_true_trace_count * sizeof(double),
            CudaTransferCategory::checkpoint_restore_import,
            CudaTransferSite::checkpoint_import_trace,
            checkpoint.owned_true_trace.size() * sizeof(double),
            loaded.trace.count * sizeof(double));
    const bool synchronized =
        TransportPolicy::Synchronize(CudaTransferSite::checkpoint_import_trace);
    const bool copied = interior_copied && trace_copied && synchronized;
    if (!CollectiveContinue(copied)) { return false; }
    StateHash copied_interior{};
    StateHash copied_trace{};
    const bool local_hash_ready = HashState(
        loaded.interior.coefficients, shape_.interior_count,
        copied_interior) &&
        HashState(loaded.trace.coefficients, shape_.true_trace_count,
                  copied_trace);
    const bool copied_hashes_valid = local_hash_ready &&
        copied_interior == host_interior && copied_trace == host_trace;
    if (!CollectiveContinue(copied_hashes_valid)) { return false; }
    loaded.interior.local_hash = copied_interior;
    loaded.trace.local_hash = copied_trace;
    if (!AggregateHash(copied_interior, loaded.interior.hash) ||
        !AggregateHash(copied_trace, loaded.trace.hash))
      return false;
    loaded.interior.generation = checkpoint.descriptor.coordinates.generation;
    loaded.trace.generation = checkpoint.descriptor.coordinates.generation;
    return CollectiveContinue(
        loaded.interior.hash == checkpoint.descriptor.interior_hash &&
        loaded.trace.hash == checkpoint.descriptor.trace_hash);
  }

  CudaReuseEvidence ReuseEvidence() const noexcept
  {
    CudaReuseEvidence value{};
    value.topology_token = reinterpret_cast<std::uintptr_t>(&resources_);
    value.double_arena_token = reinterpret_cast<std::uintptr_t>(double_arena_);
    value.integer_arena_token = reinterpret_cast<std::uintptr_t>(integer_arena_);
    value.matrix_token = reinterpret_cast<std::uintptr_t>(true_matrix_.get());
    value.factor_token = reinterpret_cast<std::uintptr_t>(
        ilu_ ? static_cast<HYPRE_Solver>(*ilu_) : nullptr);
    value.workspace_token = reinterpret_cast<std::uintptr_t>(view_.a);
    value.interior_slot_tokens = {{committed_slot_.interior.storage_token,
                                   candidate_slot_.interior.storage_token}};
    value.trace_slot_tokens = {{committed_slot_.trace.storage_token,
                                candidate_slot_.trace.storage_token}};
    value.baseline_allocation_events = baseline_allocations_;
    value.post_baseline_allocation_events =
        AllocatorPolicy::AllocationEvents() - baseline_allocations_;
    value.baseline_w8_owned_allocation_calls =
        baseline_w8_owned_allocation_calls_;
    value.post_baseline_w8_owned_allocation_calls =
        AllocatorPolicy::W8OwnedAllocationCalls() -
        baseline_w8_owned_allocation_calls_;
    value.persistent_graph = true_matrix_ && local_trace_matrix_;
    value.persistent_slots = committed_slot_.interior.coefficients !=
            candidate_slot_.interior.coefficients &&
        committed_slot_.trace.coefficients != candidate_slot_.trace.coefficients;
    value.persistent_workspace = view_.a != nullptr && view_.schur != nullptr;
    return value;
  }

  bool PrepareAttempt(const AttemptIdentity& attempt, state_slot& workspace,
                      const bool first_step) noexcept
  {
    current_attempt_ = attempt;
    current_generation_ = attempt.committed_generation + 1;
    ticket_count_ = 0;
    TransportPolicy::SetPhase(CudaFailurePhase::setup, current_generation_);
    bool local_ready = TransportPolicy::DeviceToDevice(
            view_.old_interior, workspace.interior.coefficients,
            workspace.interior.count * sizeof(double),
            CudaTransferCategory::attempt_history,
            CudaTransferSite::attempt_seed_interior);
    if (first_step)
    {
      const bool interior_ready = TransportPolicy::DeviceToDevice(
              workspace.interior.coefficients,
              view_.initial_current_interior,
              workspace.interior.count * sizeof(double),
              CudaTransferCategory::attempt_history,
              CudaTransferSite::attempt_seed_interior);
      const bool trace_ready = TransportPolicy::DeviceToDevice(
              workspace.trace.coefficients,
              view_.initial_current_true_trace,
              workspace.trace.count * sizeof(double),
              CudaTransferCategory::attempt_history,
              CudaTransferSite::attempt_seed_trace);
      local_ready = local_ready && interior_ready && trace_ready;
      workspace.interior.generation = current_generation_;
      workspace.trace.generation = current_generation_;
    }
    const bool synchronized =
        TransportPolicy::Synchronize(CudaTransferSite::attempt_seed_trace);
    local_ready = local_ready && synchronized;
    if (!CollectiveContinue(local_ready)) { return false; }
    return DistributeCurrentTrace(workspace.trace.coefficients);
  }

  CudaNumericalResult Execute(const CudaConfig& config,
                              const AttemptIdentity& attempt,
                              state_slot& workspace) noexcept
  {
    CudaNumericalResult result{};
    result.phase = CudaFailurePhase::assembly;
    const bool assembly_ticket = initialized_ &&
        Ticket(CudaFailurePhase::assembly);
    if (!CollectiveContinue(assembly_ticket) ||
        !AssembleAndNorm(workspace, result.initial_residual))
    {
      result.reason = FailureReason::finite_gate;
      return result;
    }
    result.finite = std::isfinite(result.initial_residual) &&
        result.initial_residual > 1.0e-8;
    result.final_residual = result.initial_residual;
    const double target = 1.0e-10 * (1.0 + result.initial_residual);
    std::uint64_t last_product_generation = 0;
    for (int iteration = 0; iteration < 20; ++iteration)
    {
      if (result.nonlinear_updates >= 2 && result.final_residual <= target)
        break;
      const bool condensed = CopyBase(workspace) &&
          Ticket(CudaFailurePhase::condensation) &&
          CudaLaunchCondensation(view_, shape_) &&
          TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
          StatusPassed();
      if (!CollectiveContinue(condensed))
      {
        result.phase = CudaFailurePhase::condensation;
        result.reason = FailureReason::backend_failure;
        return result;
      }
      const std::uint64_t generation = ++linear_generation_;
      result.phase = CudaFailurePhase::trace_solve;
      if (!AssembleTrueSystem())
      {
        result.reason = FailureReason::true_residual_gate;
        return result;
      }
      const bool solve_ticket = Ticket(CudaFailurePhase::trace_solve);
      if (!CollectiveContinue(solve_ticket))
      {
        result.reason = FailureReason::true_residual_gate;
        return result;
      }
      const bool solve_ok =
          SolveTrueSystem(result.true_relative_residual, generation);
      if (!CollectiveContinue(solve_ok))
      {
        result.reason = FailureReason::true_residual_gate;
        return result;
      }
      ++result.trace_solves;
      const bool trace_distributed = DistributeTraceIncrement();
      const bool recovered = trace_distributed &&
          Ticket(CudaFailurePhase::recovery) &&
          CudaLaunchRecover(view_, shape_) &&
          TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
          StatusPassed();
      if (!CollectiveContinue(recovered))
      {
        result.phase = CudaFailurePhase::recovery;
        result.reason = FailureReason::recovery_gate;
        return result;
      }
      last_product_generation = generation;
      bool accepted = false;
      for (int trial = 0; trial <= 12; ++trial)
      {
        const double alpha = std::ldexp(1.0, -trial);
        const bool trial_ready = CudaLaunchTrial(view_, shape_, alpha) &&
            TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
            SwapTrialIntoWorkspace(workspace);
        if (!CollectiveContinue(trial_ready))
        {
          if (!CollectiveContinue(RestoreBase(workspace)))
          {
            result.reason = FailureReason::backend_failure;
            return result;
          }
          continue;
        }
        const bool trial_admissible =
            DistributeCurrentTrace(workspace.trace.coefficients) &&
            CudaLaunchAdmissibility(view_, shape_) &&
            TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
            StatusPassed();
        if (!CollectiveContinue(trial_admissible))
        {
          if (!CollectiveContinue(RestoreBase(workspace)))
          {
            result.reason = FailureReason::backend_failure;
            return result;
          }
          continue;
        }
        double trial_residual = std::numeric_limits<double>::infinity();
        if (!AssembleAndNorm(workspace, trial_residual) ||
            !std::isfinite(trial_residual))
        {
          result.reason = FailureReason::finite_gate;
          return result;
        }
        if (trial_residual <=
            (1.0 - 1.0e-4 * alpha) * result.final_residual)
        {
          result.final_residual = trial_residual;
          ++result.nonlinear_updates;
          accepted = true;
          break;
        }
        if (!CollectiveContinue(RestoreBase(workspace)))
        {
          result.reason = FailureReason::backend_failure;
          return result;
        }
      }
      if (!accepted)
      {
        result.reason = FailureReason::admissibility_gate;
        return result;
      }
    }
    result.phase = CudaFailurePhase::after_recovery_before_commit;
    result.recovery = last_product_generation != 0;
    result.candidate_ready = result.recovery;
    result.products = {last_product_generation,
                       last_product_generation,
                       last_product_generation,
                       last_product_generation,
                       0};
    workspace.interior.generation = current_generation_;
    workspace.trace.generation = current_generation_;

    // The prescribed injection is agreed before admissibility, hashing,
    // telemetry, or oracle export.  Exactly one principal source is exposed;
    // every rank takes the same rollback branch.
    const int injection_fields[3] = {
        config.injection.enabled ? 1 : 0,
        static_cast<int>(config.injection.target_step),
        config.injection.source_rank};
    int injection_min[3] = {};
    int injection_max[3] = {};
    const bool injection_protocol = MPI_Allreduce(
            injection_fields, injection_min, 3, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD) == MPI_SUCCESS &&
        MPI_Allreduce(injection_fields, injection_max, 3, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) == MPI_SUCCESS &&
        std::equal(std::begin(injection_min), std::end(injection_min),
                   std::begin(injection_max));
    const bool injection_active = injection_protocol &&
        injection_min[0] == 1 &&
        static_cast<std::uint64_t>(injection_min[1]) ==
            attempt.committed_step + 1 && attempt.retry_index == 0;
    const int local_claim = injection_active &&
        rank_ == injection_min[2] ? 1 : 0;
    int claims = 0;
    const bool claims_reduced = MPI_Allreduce(
        &local_claim, &claims, 1, MPI_INT, MPI_SUM,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    if (!injection_protocol || !claims_reduced ||
        (injection_active && claims != 1))
    {
      result.success = false;
      result.reason = FailureReason::collective_protocol;
      result.injection_source_rank = -1;
      return result;
    }
    if (injection_active)
    {
      result.injected = true;
      result.success = false;
      result.reason = FailureReason::backend_failure;
      result.injection_source_rank = injection_min[2];
      return result;
    }

    // Freeze one coherent accepted linearization: A/B/C/D, Schur/RHS,
    // solution, and recovery all describe the same candidate generation.
    const bool coherent_base = result.nonlinear_updates >= 2 &&
        result.final_residual <= target && CopyBase(workspace) &&
        Ticket(CudaFailurePhase::condensation) &&
        CudaLaunchCondensation(view_, shape_) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
        StatusPassed();
    if (!CollectiveContinue(coherent_base) || !AssembleTrueSystem())
    {
      result.reason = FailureReason::backend_failure;
      return result;
    }
    last_product_generation = ++linear_generation_;
    const bool coherent_ticket = Ticket(CudaFailurePhase::trace_solve);
    if (!CollectiveContinue(coherent_ticket))
    {
      result.reason = FailureReason::true_residual_gate;
      return result;
    }
    const bool coherent_solve_ok = SolveTrueSystem(
        result.true_relative_residual, last_product_generation);
    if (!CollectiveContinue(coherent_solve_ok))
    {
      result.reason = FailureReason::true_residual_gate;
      return result;
    }
    const bool coherent_solve = DistributeTraceIncrement() &&
        Ticket(CudaFailurePhase::recovery) &&
        CudaLaunchRecover(view_, shape_) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
        StatusPassed();
    if (!CollectiveContinue(coherent_solve))
    {
      result.reason = FailureReason::recovery_gate;
      return result;
    }
    ++result.trace_solves;
    result.products = {last_product_generation,
                       last_product_generation,
                       last_product_generation,
                       last_product_generation,
                       0};
    const bool local_admissible = CudaLaunchAdmissibility(view_, shape_) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
        StatusPassed();
    result.admissible = CollectiveContinue(local_admissible);
    result.true_residual = std::isfinite(result.true_relative_residual) &&
        result.true_relative_residual <= 1.0e-10;
    result.telemetry = DeviceResident() && NoFallback() &&
        AllocatorPolicy::W8OwnedAllocationCalls() ==
            baseline_w8_owned_allocation_calls_ &&
        AllocatorPolicy::AllocationEvents() == baseline_allocations_;
    result.success = result.finite && result.admissible && result.recovery &&
        result.true_residual && result.telemetry;
    result.reason = result.success ? FailureReason::none :
        FailureReason::success_gate;
    if (!UpdateStateHashes(workspace))
    {
      result.success = false;
      result.telemetry = false;
      result.reason = FailureReason::telemetry_gate;
    }
    if (config.capture_oracle)
    {
      CaptureOracleMetadata(attempt, last_product_generation);
    }
    return result;
  }

  bool ExportOracle(CudaOraclePod& pod) noexcept
  {
    if (!initialized_ || !oracle_captured_) { return false; }
    pod = oracle_;
    TransportPolicy::SetPhase(CudaFailurePhase::terminal_evidence,
                              current_generation_);
    return ExportArray(pod.old_interior.data(), view_.old_interior,
                       shape_.interior_count) &&
        ExportArray(pod.current_interior.data(), view_.work_interior,
                    shape_.interior_count) &&
        ExportArray(pod.current_local_trace.data(), view_.current_local_trace,
                    shape_.elements * trace_dofs_per_element) &&
        ExportArray(pod.distributed_local_trace.data(),
                    ranks_ == 1 ? view_.work_true_trace :
                                  view_.distributed_local_trace,
                    shape_.local_trace_count) &&
        ExportArray(pod.current_true_trace.data(), view_.work_true_trace,
                    shape_.true_trace_count) &&
        ExportArray(pod.residual_r.data(), view_.residual_r,
                    shape_.interior_count) &&
        ExportArray(pod.residual_g.data(), view_.residual_g,
                    shape_.elements * trace_dofs_per_element) &&
        ExportArray(pod.a.data(), view_.a,
                    shape_.elements * interior_dofs_per_element *
                        interior_dofs_per_element) &&
        ExportArray(pod.b.data(), view_.b,
                    shape_.elements * interior_dofs_per_element *
                        trace_dofs_per_element) &&
        ExportArray(pod.c.data(), view_.c,
                    shape_.elements * trace_dofs_per_element *
                        interior_dofs_per_element) &&
        ExportArray(pod.d.data(), view_.d,
                    shape_.elements * trace_dofs_per_element *
                        trace_dofs_per_element) &&
        ExportArray(pod.f.data(), view_.f, shape_.interior_count) &&
        ExportArray(pod.g.data(), view_.g,
                    shape_.elements * trace_dofs_per_element) &&
        ExportArray(pod.schur.data(), view_.schur,
                    shape_.elements * trace_dofs_per_element *
                        trace_dofs_per_element) &&
        ExportArray(pod.condensed_rhs.data(), view_.condensed_rhs,
                    shape_.elements * trace_dofs_per_element) &&
        ExportArray(pod.local_trace_increment.data(),
                    view_.trace_increment_local,
                    shape_.elements * trace_dofs_per_element) &&
        ExportArray(pod.recovered_increment.data(), view_.recovered_interior,
                    shape_.interior_count) &&
        ExportTrueSystem(pod);
  }

private:
  bool ConfigureDeviceAndRanks(const CudaConfig& config,
                               CudaRunEvidence& evidence) noexcept
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks_);
    const bool rank_count_valid = ranks_ == config.expected_rank_count;
    const bool cuda_set_device_valid =
        cudaSetDevice(config.requested_device) == cudaSuccess;
    const bool cuda_get_device_valid =
        cudaGetDevice(&device_) == cudaSuccess;
    const bool selected_device_valid = device_ == 0;
    const bool mfem_configured = mfem::Device::IsConfigured();
    const bool mfem_cuda_allowed =
        mfem::Device::Allows(mfem::Backend::CUDA_MASK);
    const bool mfem_debug_disabled =
        !mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE);

    // Make a prior HYPRE error distinguishable from an Init-produced error.
    // No clearing is allowed between the Init call and its captured result.
    const HYPRE_Int hypre_preinit_clear = HYPRE_ClearAllErrors();
    const HYPRE_Int hypre_preinit_error = HYPRE_GetError();
    const bool hypre_preinit_error_clean =
        hypre_preinit_clear == 0 && hypre_preinit_error == 0;
    bool hypre_init_returned = false;
    try
    {
      // Accepted W7 parity: create/configure the rank-local HYPRE singleton
      // before querying its device policy or allocator pool identity.
      mfem::Hypre::Init();
      hypre_init_returned = true;
    }
    catch (...) {}
    const HYPRE_Int hypre_init_error = HYPRE_GetError();
    const bool hypre_init_error_free = hypre_init_error == 0;
    const bool local_hypre_init_valid = hypre_preinit_error_clean &&
        hypre_init_returned && hypre_init_error_free;

    evidence.device = device_;
    evidence.mfem_device_valid =
        mfem_configured && mfem_cuda_allowed && mfem_debug_disabled;
    evidence.hypre_cuda_umpire_device = false;

    std::uint64_t backend_failure_mask = 0;
    if (!rank_count_valid)
      backend_failure_mask |= backend_rank_count_failure;
    if (!cuda_set_device_valid)
      backend_failure_mask |= backend_cuda_set_device_failure;
    if (!cuda_get_device_valid)
      backend_failure_mask |= backend_cuda_get_device_failure;
    if (!selected_device_valid)
      backend_failure_mask |= backend_selected_device_failure;
    if (!mfem_configured)
      backend_failure_mask |= backend_mfem_configured_failure;
    if (!mfem_cuda_allowed)
      backend_failure_mask |= backend_mfem_cuda_failure;
    if (!mfem_debug_disabled)
      backend_failure_mask |= backend_mfem_debug_failure;
    if (!hypre_preinit_error_clean)
      backend_failure_mask |= backend_hypre_preinit_error_failure;
    if (!hypre_init_returned)
      backend_failure_mask |= backend_hypre_initialize_return_failure;
    if (!hypre_init_error_free)
      backend_failure_mask |= backend_hypre_initialize_error_failure;

    // This gate is intentionally before every later HYPRE/Umpire query.  A
    // rank-local throw or Init error therefore converges before any peer can
    // advance into the policy-admission sequence.
    const bool hypre_init_collectively_valid =
        CollectiveContinue(local_hypre_init_valid);
    if (!hypre_init_collectively_valid)
    {
      ReportCudaBackendFailure(rank_, backend_failure_mask);
      return false;
    }

    // Init's zero error was recorded before this reset.  From this point all
    // ranks execute the same policy/query/pool sequence unconditionally.
    const HYPRE_Int hypre_policy_clear = HYPRE_ClearAllErrors();
    const HYPRE_Int hypre_policy_start_error = HYPRE_GetError();
    const bool hypre_policy_start_clean =
        hypre_policy_clear == 0 && hypre_policy_start_error == 0;
    HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_UNDEFINED;
    HYPRE_ExecutionPolicy execution_policy = HYPRE_EXEC_UNDEFINED;
    const bool hypre_gpu = mfem::HypreUsingGPU();
    const bool hypre_memory_query_valid =
        HYPRE_GetMemoryLocation(&memory_location) == 0;
    const bool hypre_execution_query_valid =
        HYPRE_GetExecutionPolicy(&execution_policy) == 0;
    const bool hypre_memory_device =
        memory_location == HYPRE_MEMORY_DEVICE;
    const bool hypre_execution_device =
        execution_policy == HYPRE_EXEC_DEVICE;
    const HYPRE_Int hypre_policy_error = HYPRE_GetError();
    const bool hypre_policy_error_free = hypre_policy_error == 0;
    const bool umpire_pools_valid =
        AllocatorPolicy::DevicePoolPolicyValid();

    evidence.hypre_cuda_umpire_device = local_hypre_init_valid &&
        hypre_policy_start_clean && hypre_gpu && hypre_memory_query_valid &&
        hypre_execution_query_valid &&
        hypre_memory_device && hypre_execution_device &&
        hypre_policy_error_free && umpire_pools_valid;
    if (!hypre_policy_start_clean)
      backend_failure_mask |= backend_hypre_policy_clear_failure;
    if (!hypre_gpu)
      backend_failure_mask |= backend_hypre_gpu_failure;
    if (!hypre_memory_query_valid)
      backend_failure_mask |= backend_hypre_memory_query_failure;
    if (!hypre_execution_query_valid)
      backend_failure_mask |= backend_hypre_execution_query_failure;
    if (!hypre_memory_device)
      backend_failure_mask |= backend_hypre_memory_device_failure;
    if (!hypre_execution_device)
      backend_failure_mask |= backend_hypre_execution_device_failure;
    if (!hypre_policy_error_free)
      backend_failure_mask |= backend_hypre_policy_error_failure;
    if (!umpire_pools_valid)
      backend_failure_mask |= backend_umpire_pools_failure;

    const bool local_backend_valid = backend_failure_mask == 0;
    const bool backend_collectively_valid =
        CollectiveContinue(local_backend_valid);
    if (!backend_collectively_valid)
    {
      ReportCudaBackendFailure(rank_, backend_failure_mask);
      return false;
    }
    cudaDeviceProp properties{};
    const bool uuid_ready =
        cudaGetDeviceProperties(&properties, device_) == cudaSuccess;
    if (!CollectiveContinue(uuid_ready)) { return false; }
    std::memcpy(gpu_uuid_.data(), properties.uuid.bytes, gpu_uuid_.size());
    std::array<unsigned char, 32> gathered{};
    if (ranks_ == 2)
    {
      if (MPI_Allgather(gpu_uuid_.data(), 16, MPI_UNSIGNED_CHAR,
                        gathered.data(), 16, MPI_UNSIGNED_CHAR,
                        MPI_COMM_WORLD) != MPI_SUCCESS)
        return false;
      same_physical_gpu_ =
          std::equal(gathered.begin(), gathered.begin() + 16,
                     gathered.begin() + 16);
      host_staged_rank2_ = config.require_host_staged_rank2;
    }
    else
    {
      same_physical_gpu_ = true;
      host_staged_rank2_ = false;
    }
    evidence.same_physical_gpu = same_physical_gpu_;
    return same_physical_gpu_;
  }

  bool BuildTopology(const CudaConfig&) noexcept
  {
    try
    {
      mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
          elements_per_direction, elements_per_direction,
          mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
      // MFEM owns this setup-only container and the mapping allocation.
      std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
      translations[0][0] = 1.0;
      translations[1][1] = 1.0;
      serial_mesh_ = mfem::Mesh::MakePeriodic(
          original, original.CreatePeriodicVertexMapping(translations));
      std::array<int, elements_per_direction * elements_per_direction>
          partition{};
      for (int element = 0; element < static_cast<int>(partition.size());
           ++element)
        partition[static_cast<std::size_t>(element)] =
            element * ranks_ / static_cast<int>(partition.size());
      mesh_.emplace(MPI_COMM_WORLD, serial_mesh_, partition.data());
      interior_collection_.emplace(
          polynomial_order, dimension, mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      trace_collection_.emplace(
          polynomial_order, dimension, mfem::FiniteElement::VALUE,
          mfem::BasisType::GaussLegendre);
      interior_space_.emplace(&*mesh_, &*interior_collection_,
                              conserved_components, mfem::Ordering::byNODES);
      trace_space_.emplace(&*mesh_, &*trace_collection_, conserved_components,
                           mfem::Ordering::byNODES);
      interior_space_->ExchangeFaceNbrData();
      trace_space_->ExchangeFaceNbrData();
      if (!dg::hdg::trace::BuildDGInterfaceTraceExchangePlan(
              *trace_space_, exchange_plan_))
        return false;
    }
    catch (...)
    {
      return false;
    }
    shape_.elements = mesh_->GetNE();
    shape_.interior_count = shape_.elements * interior_dofs_per_element;
    shape_.local_trace_count = trace_space_->GetVSize();
    shape_.true_trace_count = trace_space_->GetTrueVSize();
    shape_.global_true_trace_count = static_cast<int>(
        exchange_plan_.map.dofs.global_true_dof_count);
    const mfem::FiniteElement* fe = interior_space_->GetFE(0);
    mfem::Array<int> first_faces, first_orientations;
    mesh_->GetElementEdges(0, first_faces, first_orientations);
    const mfem::FiniteElement* face_fe =
        trace_space_->GetFaceElement(first_faces[0]);
    if (shape_.elements <= 0 || shape_.elements > maximum_rank_elements ||
        shape_.local_trace_count <= 0 ||
        shape_.local_trace_count > maximum_local_trace_dofs ||
        shape_.true_trace_count <= 0 ||
        shape_.true_trace_count > maximum_true_trace_dofs || fe == nullptr ||
        face_fe == nullptr || fe->GetDof() != scalar_dofs_per_element ||
        face_fe->GetDof() != scalar_dofs_per_face ||
        interior_space_->GetOrdering() != mfem::Ordering::byNODES ||
        trace_space_->GetOrdering() != mfem::Ordering::byNODES)
      return false;
    shape_.volume_points = mfem::IntRules.Get(
        fe->GetGeomType(), assembly_quadrature_order).GetNPoints();
    shape_.face_points = mfem::IntRules.Get(
        face_fe->GetGeomType(), assembly_quadrature_order).GetNPoints();
    if (shape_.volume_points > maximum_volume_points ||
        shape_.face_points > maximum_face_points)
      return false;
    return BuildHostGeometryAndMaps() && BuildLocalTraceGraph();
  }

  bool BuildHostGeometryAndMaps() noexcept;
  bool BuildLocalTraceGraph() noexcept;
  bool BuildPersistentTrueSystem() noexcept;
  bool BuildSparseStagingPlan() noexcept;

  bool AllocateArenas() noexcept
  {
    void* doubles = nullptr;
    void* integers = nullptr;
    if (!AllocatorPolicy::Allocate(
            persistent_double_capacity * sizeof(double), alignof(double),
            CudaAllocationSite::persistent_double_arena, doubles))
      return false;
    if (!AllocatorPolicy::Allocate(
            persistent_integer_capacity * sizeof(int), alignof(std::uint64_t),
            CudaAllocationSite::persistent_integer_arena, integers))
    {
      AllocatorPolicy::Deallocate(
          doubles, CudaAllocationSite::persistent_double_arena);
      return false;
    }
    double_arena_ = static_cast<double*>(doubles);
    integer_arena_ = static_cast<int*>(integers);
    return double_arena_ != nullptr && integer_arena_ != nullptr;
  }

  bool BindArenaViews() noexcept;
  bool UploadTopologyAndState() noexcept;
  bool BindStateSlots() noexcept;
  bool VerifyResidencyAndTicket(CudaRunEvidence& evidence) noexcept;
  bool Ticket(CudaFailurePhase phase) noexcept;
  bool StatusPassed() noexcept;
  bool DistributeCurrentTrace(const double* true_values) noexcept;
  bool DistributeTraceIncrement() noexcept;
  bool AssembleTrueVector(const double* element_values,
                          double* true_values) noexcept;
  bool AssembleTrueSystem() noexcept;
  bool SolveTrueSystem(double& true_relative_residual,
                       std::uint64_t matrix_generation) noexcept;
  bool AssembleAndNorm(state_slot& workspace, double& norm) noexcept;
  bool CopyBase(state_slot& workspace) noexcept;
  bool RestoreBase(state_slot& workspace) noexcept;
  bool SwapTrialIntoWorkspace(state_slot& workspace) noexcept;
  bool UpdateStateHashes(state_slot& workspace) noexcept;
  bool HashState(const double* values, int count, StateHash& hash) noexcept;
  bool AggregateHash(const StateHash& local, StateHash& aggregate) const
      noexcept;
  std::size_t WorkspaceBytes() const noexcept;
  void CaptureOracleMetadata(const AttemptIdentity& attempt,
                             std::uint64_t generation) noexcept;
  bool ExportTrueSystem(CudaOraclePod& pod) noexcept;

  bool ExportArray(double* destination, const double* source,
                   const int count) noexcept
  {
    return count >= 0 && TransportPolicy::DeviceToHost(
        destination, source, static_cast<std::size_t>(count) * sizeof(double),
        CudaTransferCategory::oracle_export, CudaTransferSite::oracle_export) &&
        TransportPolicy::Synchronize(CudaTransferSite::oracle_export);
  }

  bool initialized_ = false;
  bool same_physical_gpu_ = false;
  bool host_staged_rank2_ = false;
  bool signed_owner_ghost_valid_ = false;
  bool device_resident_ = false;
  bool hypre_policy_valid_ = false;
  bool no_fallback_ = true;
  bool oracle_captured_ = false;
  int rank_ = -1;
  int ranks_ = 0;
  int device_ = -1;
  std::array<unsigned char, 16> gpu_uuid_{};
  RuntimeTupleDescriptor tuple_{};
  PersistentResourceDescriptor resources_{};
  StateHash local_mesh_hash_{};
  StateHash local_partition_hash_{};
  StateHash local_topology_hash_{};
  CudaMeshShape shape_{};
  CudaDeviceView view_{};
  CudaTrueMatrixStructure true_structure_{};
  CudaTrueMatrixDeviceView true_device_{};
  double* double_arena_ = nullptr;
  int* integer_arena_ = nullptr;
  mfem::Memory<double> double_memory_{};
  mfem::Memory<int> integer_memory_{};
  mfem::Vector double_arena_vector_{};
  mfem::Array<int> integer_arena_array_{};
  mfem::Mesh serial_mesh_{};
  std::optional<mfem::ParMesh> mesh_{};
  std::optional<mfem::L2_FECollection> interior_collection_{};
  std::optional<mfem::DG_Interface_FECollection> trace_collection_{};
  std::optional<mfem::ParFiniteElementSpace> interior_space_{};
  std::optional<mfem::ParFiniteElementSpace> trace_space_{};
  dg::hdg::trace::DGInterfaceTraceExchangePlan exchange_plan_{};
  std::optional<mfem::SparseMatrix> local_trace_matrix_{};
  std::unique_ptr<mfem::HypreParMatrix> true_matrix_{};
  std::array<std::optional<mfem::HypreParVector>, 2> trace_slots_{};
  std::optional<mfem::HypreParVector> hypre_rhs_{};
  std::optional<mfem::HypreParVector> hypre_solution_{};
  std::optional<mfem::HypreParVector> hypre_action_{};
  std::optional<mfem::HypreILU> ilu_{};
  std::optional<mfem::HypreGMRES> gmres_{};
  mfem::Vector owned_stage_{};
  mfem::Vector local_stage_{};
  state_slot committed_slot_{};
  state_slot candidate_slot_{};
  state_slot workspace_slot_{};
  AttemptIdentity current_attempt_{};
  std::uint64_t current_generation_ = 0;
  std::uint64_t linear_generation_ = 0;
  std::uint64_t baseline_allocations_ = 0;
  std::uint64_t baseline_w8_owned_allocation_calls_ = 0;
  std::size_t host_exchange_bytes_ = 0;
  std::array<CudaDeviceTicket, maximum_device_tickets> tickets_{};
  std::size_t ticket_count_ = 0;
  CudaOraclePod oracle_{};

  std::array<int, maximum_rank_elements * trace_dofs_per_element>
      signed_element_trace_{};
  std::array<std::int64_t, maximum_local_trace_dofs> local_global_true_{};
  std::array<int, maximum_local_trace_dofs> local_owner_{};
  std::array<int, maximum_local_trace_dofs> local_true_index_{};
  std::array<std::int64_t, maximum_true_trace_dofs> owned_global_true_{};
  std::array<std::int64_t, maximum_rank_elements * dimension>
      element_coordinate_keys_{};
  std::array<double, maximum_rank_elements * maximum_volume_points *
                         scalar_dofs_per_element>
      host_volume_shape_{};
  std::array<double, maximum_rank_elements * maximum_volume_points *
                         scalar_dofs_per_element * dimension>
      host_volume_derivative_{};
  std::array<double, maximum_rank_elements * maximum_volume_points>
      host_volume_weight_{};
  std::array<double, maximum_rank_elements * maximum_volume_points * dimension>
      host_volume_position_{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * scalar_dofs_per_element>
      host_face_interior_shape_{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * scalar_dofs_per_face>
      host_face_trace_shape_{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points>
      host_face_weight_{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * dimension>
      host_face_normal_{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element>
      host_exact_interior_{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element>
      host_perturbed_interior_{};
  std::array<double, maximum_true_trace_dofs> host_exact_true_trace_{};
  std::array<double, maximum_true_trace_dofs> host_perturbed_true_trace_{};
  std::array<int, maximum_local_matrix_entries> host_matrix_owned_source_{};
  std::array<int, maximum_local_matrix_entries> host_matrix_owned_section_{};
  std::array<int, maximum_local_matrix_entries> host_matrix_owned_target_{};
  std::array<int, maximum_exchange_entries> host_matrix_send_source_{};
  std::array<int, maximum_exchange_entries> host_matrix_receive_section_{};
  std::array<int, maximum_exchange_entries> host_matrix_receive_target_{};
  std::array<int, maximum_rank_elements * trace_dofs_per_element>
      host_vector_owned_source_{};
  std::array<int, maximum_rank_elements * trace_dofs_per_element>
      host_vector_owned_target_{};
  std::array<int, maximum_rank_elements * trace_dofs_per_element>
      host_vector_send_source_{};
  std::array<int, maximum_rank_elements * trace_dofs_per_element>
      host_vector_receive_target_{};
  std::array<std::int64_t, maximum_exchange_entries * 2> matrix_send_keys_{};
  std::array<std::int64_t, maximum_exchange_entries * 2> matrix_receive_keys_{};
  std::array<std::int64_t,
             maximum_rank_elements * trace_dofs_per_element>
      vector_send_keys_{};
  std::array<std::int64_t,
             maximum_rank_elements * trace_dofs_per_element>
      vector_receive_keys_{};
  std::array<double, maximum_exchange_entries> matrix_send_host_{};
  std::array<double, maximum_exchange_entries> matrix_receive_host_{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element>
      vector_send_host_{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element>
      vector_receive_host_{};
};

} // namespace detail

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_methods.cuh"
