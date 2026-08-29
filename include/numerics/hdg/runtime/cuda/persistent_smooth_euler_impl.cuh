/* Included by persistent_smooth_euler.cuh inside dg::hdg::runtime::cuda_w8a. */

namespace detail
{

static inline void ReportCudaSetupFailure(const int rank,
                                          const char* const stage) noexcept
{
  if (rank == 0)
    std::fprintf(stderr, "CUDA_SETUP_FAIL stage=%s\n", stage);
}

// Allocation-free lifetime markers for the isolated HYPRE warm setup.  These
// deliberately do not inspect or mutate CUDA/HYPRE error state and must never
// add an allocator/high-water sample of their own.
static inline void ReportCudaWarmLifetimeMarker(
    const int rank, const char* const stage, const char* const edge,
    const bool result) noexcept
{
  (void)rank;
  (void)stage;
  (void)edge;
  (void)result;
}

// Stable private backend-admission mask: bits 0..15 are rank count,
// cudaSetDevice, cudaGetDevice, selected device zero, MFEM configured,
// MFEM CUDA allowed, MFEM debug-device disabled, zero incoming HYPRE error,
// normal Hypre::Init return, zero error captured immediately after Init, Hypre
// GPU selection, HYPRE memory query, HYPRE execution query, device memory,
// device execution, zero post-policy HYPRE error, and an admitted SM120
// tensor-core target when that explicit engine is selected.  Materialized
// MFEM and HYPRE Umpire pools are admitted only after their first real
// allocations.
enum CudaBackendFailureBit : std::uint64_t
{
  backend_rank_count_failure = std::uint64_t{1} << 0U,
  backend_cuda_set_device_failure = std::uint64_t{1} << 1U,
  backend_cuda_get_device_failure = std::uint64_t{1} << 2U,
  backend_selected_device_failure = std::uint64_t{1} << 3U,
  backend_mfem_configured_failure = std::uint64_t{1} << 4U,
  backend_mfem_cuda_failure = std::uint64_t{1} << 5U,
  backend_mfem_debug_failure = std::uint64_t{1} << 6U,
  backend_hypre_incoming_error_failure = std::uint64_t{1} << 7U,
  backend_hypre_initialize_return_failure = std::uint64_t{1} << 8U,
  backend_hypre_initialize_error_failure = std::uint64_t{1} << 9U,
  backend_hypre_gpu_failure = std::uint64_t{1} << 10U,
  backend_hypre_memory_query_failure = std::uint64_t{1} << 11U,
  backend_hypre_execution_query_failure = std::uint64_t{1} << 12U,
  backend_hypre_memory_device_failure = std::uint64_t{1} << 13U,
  backend_hypre_execution_device_failure = std::uint64_t{1} << 14U,
  backend_hypre_policy_error_failure = std::uint64_t{1} << 15U,
  backend_tensor_core_hardware_failure = std::uint64_t{1} << 16U
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

static inline void ReportCudaTensorCoreAdmissionFailure(
    const int rank, const int architecture) noexcept
{
  if (rank == 0)
    std::fprintf(stderr,
                 "CUDA_TENSOR_CORE_UNSUPPORTED required_sm=120 actual_sm=%d\n",
                 architecture);
}

// Private teardown diagnostics.  MFEM's DEVICE_UMPIRE Wrap(..., own=false)
// still owns a MemoryManager registration, while the raw device allocation
// remains policy-owned.  These bits therefore describe failures to release
// MFEM bookkeeping before the policy is allowed to deallocate either arena.
enum CudaArenaTeardownFailureBit : std::uint64_t
{
  arena_vector_alias_release_failure = std::uint64_t{1} << 0U,
  arena_double_registration_release_failure = std::uint64_t{1} << 1U,
  arena_integer_registration_release_failure = std::uint64_t{1} << 2U,
  arena_integer_raw_release_failure = std::uint64_t{1} << 3U,
  arena_double_raw_release_failure = std::uint64_t{1} << 4U,
  arena_raw_release_quarantined = std::uint64_t{1} << 5U,
  arena_idempotent_state_failure = std::uint64_t{1} << 6U
};

static inline void ReportCudaArenaTeardownFailure(
    const int rank, const std::uint64_t failure_mask) noexcept
{
  if (rank == 0)
  {
    std::fprintf(stderr, "CUDA_TEARDOWN_FAIL rank0_mask=0x%llx\n",
                 static_cast<unsigned long long>(failure_mask));
  }
}

enum class CudaArenaOwnership : unsigned char
{
  none,
  owned,
  released,
  mfem_retained,
  retryable,
  quarantined
};

enum class CudaRawReleaseResult : unsigned char
{
  released,
  retryable,
  quarantined
};

struct CudaArenaReleaseState
{
  CudaArenaOwnership integer = CudaArenaOwnership::none;
  CudaArenaOwnership doubles = CudaArenaOwnership::none;
  std::uint64_t failure_history = 0;
  unsigned pass_count = 0;
  bool wrappers_attempted = false;
  bool views_cleared = false;
  bool complete = false;
};

struct CudaArenaReleasePass
{
  CudaArenaOwnership integer = CudaArenaOwnership::none;
  CudaArenaOwnership doubles = CudaArenaOwnership::none;
  std::uint64_t failure_mask = 0;
  unsigned pass = 0;
  bool complete = false;
  bool idempotent = false;
};

static inline void ReportCudaArenaTeardownMarker(
    const int rank, const char* const stage, const char* const edge,
    const CudaArenaReleaseState& state,
    const std::uint64_t failure_mask) noexcept
{
  (void)rank;
  (void)stage;
  (void)edge;
  (void)state;
  (void)failure_mask;
}

template <typename Operations>
CudaArenaReleasePass ReleaseCudaArenas(
    CudaArenaReleaseState& state, Operations& operations) noexcept
{
  CudaArenaReleasePass result{};
  result.pass = ++state.pass_count;
  auto finish = [&]() noexcept
  {
    result.integer = state.integer;
    result.doubles = state.doubles;
    result.complete = state.complete;
    state.failure_history |= result.failure_mask;
    return result;
  };
  if (state.complete)
  {
    result.idempotent = true;
    return finish();
  }
  if (state.integer == CudaArenaOwnership::mfem_retained ||
      state.doubles == CudaArenaOwnership::mfem_retained ||
      state.integer == CudaArenaOwnership::quarantined ||
      state.doubles == CudaArenaOwnership::quarantined)
  {
    result.idempotent = true;
    return finish();
  }

  if (!state.wrappers_attempted)
  {
    state.wrappers_attempted = true;
    state.integer = operations.HasIntegerArena()
        ? CudaArenaOwnership::owned : CudaArenaOwnership::none;
    state.doubles = operations.HasDoubleArena()
        ? CudaArenaOwnership::owned : CudaArenaOwnership::none;
    operations.Marker("vector-alias", "before", state,
                      result.failure_mask);
    const bool alias_released = operations.ReleaseVectorAlias();
    if (!alias_released)
    {
      operations.MakeWrappersInert();
      if (state.integer == CudaArenaOwnership::owned)
        state.integer = CudaArenaOwnership::mfem_retained;
      if (state.doubles == CudaArenaOwnership::owned)
        state.doubles = CudaArenaOwnership::mfem_retained;
      result.failure_mask = arena_vector_alias_release_failure;
    }
    operations.Marker("vector-alias", "after", state,
                      result.failure_mask);
    if (alias_released)
    {
      operations.Marker("double-registration", "before", state,
                        result.failure_mask);
      const bool double_registration_released =
          operations.ReleaseDoubleRegistration();
      if (!double_registration_released)
      {
        operations.MakeWrappersInert();
        if (state.integer == CudaArenaOwnership::owned)
          state.integer = CudaArenaOwnership::mfem_retained;
        if (state.doubles == CudaArenaOwnership::owned)
          state.doubles = CudaArenaOwnership::mfem_retained;
        result.failure_mask = arena_double_registration_release_failure;
      }
      operations.Marker("double-registration", "after", state,
                        result.failure_mask);
      if (double_registration_released)
      {
        operations.Marker("integer-registration", "before", state,
                          result.failure_mask);
        const bool integer_registration_released =
            operations.ReleaseIntegerRegistration();
        if (!integer_registration_released)
        {
          operations.MakeWrappersInert();
          if (state.integer == CudaArenaOwnership::owned)
            state.integer = CudaArenaOwnership::mfem_retained;
          if (state.doubles == CudaArenaOwnership::owned)
            state.doubles = CudaArenaOwnership::quarantined;
          result.failure_mask = arena_integer_registration_release_failure;
        }
        operations.Marker("integer-registration", "after", state,
                          result.failure_mask);
      }
    }
    operations.ClearViews();
    state.views_cleared = true;
    if (result.failure_mask != 0) { return finish(); }
  }

  auto release_raw = [&](CudaArenaOwnership& ownership,
                         const CudaRawReleaseResult release,
                         const std::uint64_t failure_bit) noexcept
  {
    if (release == CudaRawReleaseResult::released)
    {
      ownership = CudaArenaOwnership::released;
      return true;
    }
    ownership = release == CudaRawReleaseResult::retryable
        ? CudaArenaOwnership::retryable : CudaArenaOwnership::quarantined;
    result.failure_mask |= failure_bit;
    if (release == CudaRawReleaseResult::quarantined)
      result.failure_mask |= arena_raw_release_quarantined;
    return false;
  };
  if (state.integer == CudaArenaOwnership::owned ||
      state.integer == CudaArenaOwnership::retryable)
  {
    operations.Marker("integer-raw", "before", state,
                      result.failure_mask);
    const CudaRawReleaseResult released = operations.ReleaseIntegerArena();
    if (!release_raw(state.integer, released,
                     arena_integer_raw_release_failure))
    {
      operations.Marker("integer-raw", "after", state,
                        result.failure_mask);
      return finish();
    }
    operations.Marker("integer-raw", "after", state,
                      result.failure_mask);
  }
  if (state.doubles == CudaArenaOwnership::owned ||
      state.doubles == CudaArenaOwnership::retryable)
  {
    operations.Marker("double-raw", "before", state,
                      result.failure_mask);
    const CudaRawReleaseResult released = operations.ReleaseDoubleArena();
    if (!release_raw(state.doubles, released,
                     arena_double_raw_release_failure))
    {
      operations.Marker("double-raw", "after", state,
                        result.failure_mask);
      return finish();
    }
    operations.Marker("double-raw", "after", state,
                      result.failure_mask);
  }
  const bool integer_done = state.integer == CudaArenaOwnership::none ||
      state.integer == CudaArenaOwnership::released;
  const bool double_done = state.doubles == CudaArenaOwnership::none ||
      state.doubles == CudaArenaOwnership::released;
  state.complete = integer_done && double_done && state.views_cleared;
  result.complete = state.complete;
  return finish();
}

// One fixed-capacity allocation per arena covers both the admitted p1/N3
// smoke shape and the p2/N6 functional shape.  The arena is partitioned once
// during Initialize and never resized during a process segment.
inline constexpr std::size_t persistent_double_capacity = 1000000;
inline constexpr std::size_t persistent_integer_capacity = 600000;

struct CudaTrueMatrixStructure
{
  int rows = 0;
  int diag_nonzeros = 0;
  int offd_nonzeros = 0;
  std::vector<int> diag_offsets{};
  std::vector<int> diag_columns{};
  std::vector<int> offd_offsets{};
  std::vector<std::int64_t> offd_columns{};
};

// Private post-consensus implementation selection.  It is deliberately not a
// CudaConfig field: the public engine remains ordinary CUDA or tensor CUDA,
// and this leaf is resolved from that typed engine plus the admitted rank and
// generic-lifecycle shape.
enum class CudaTraceSolverKind : unsigned char
{
  unconfigured,
  hypre_parcsr_gmres,
  hypre_host_staged_rank2,
  native_rank1_fp64,
  hypre_fgmres_mgr_rank1
};

constexpr bool IsKnownCudaTraceSolverKind(
    const CudaTraceSolverKind kind) noexcept
{
  return kind == CudaTraceSolverKind::hypre_parcsr_gmres ||
      kind == CudaTraceSolverKind::hypre_host_staged_rank2 ||
      kind == CudaTraceSolverKind::native_rank1_fp64 ||
      kind == CudaTraceSolverKind::hypre_fgmres_mgr_rank1;
}

constexpr CudaTraceSolverKind ResolveCudaTraceSolverKind(
    const generic::CudaEngineKind engine, const int ranks,
    const bool host_staged_rank2, const bool borrowed_topology,
    const bool borrowed_host_inputs,
    const bool generic_numerical_policy,
    const CudaTracePolicyKind trace_policy) noexcept
{
  if (!generic::IsKnownCudaEngineKind(engine) ||
      !IsKnownCudaTracePolicyKind(trace_policy) ||
      (ranks != 1 && ranks != 2) ||
      (ranks == 2 && !host_staged_rank2))
    return CudaTraceSolverKind::unconfigured;
  if (trace_policy == CudaTracePolicyKind::equilibrated_mgr)
    return engine == generic::CudaEngineKind::tensor_core && ranks == 1 &&
            borrowed_topology && borrowed_host_inputs &&
            generic_numerical_policy
        ? CudaTraceSolverKind::hypre_fgmres_mgr_rank1
        : CudaTraceSolverKind::unconfigured;
  if (trace_policy == CudaTracePolicyKind::native_element_schwarz_fp64 ||
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(trace_policy) ||
      trace_policy == CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64)
    return engine == generic::CudaEngineKind::tensor_core && ranks == 1 &&
            borrowed_topology && borrowed_host_inputs &&
            generic_numerical_policy
        ? CudaTraceSolverKind::native_rank1_fp64
        : CudaTraceSolverKind::unconfigured;
  if (ranks == 2)
    return CudaTraceSolverKind::hypre_host_staged_rank2;
  if (engine == generic::CudaEngineKind::tensor_core && ranks == 1 &&
      borrowed_topology && borrowed_host_inputs &&
      generic_numerical_policy)
    return CudaTraceSolverKind::native_rank1_fp64;
  if (engine == generic::CudaEngineKind::tensor_core)
    return CudaTraceSolverKind::unconfigured;
  return CudaTraceSolverKind::hypre_parcsr_gmres;
}

// Private translation seam to the source-only native solver ABI.  Keeping
// these primitive PODs here prevents the native CUDA leaf header from becoming
// part of the public persistent-engine include surface.
struct CudaNativeTraceWorkspaceRequirements
{
  bool valid = false;
  std::size_t basis_doubles = 0;
  std::size_t vector_doubles = 0;
  std::size_t reduction_doubles = 0;
  std::size_t small_doubles = 0;
  std::size_t permutation_integers = 0;
  std::size_t control_double_slots = 0;
};

struct CudaNativeTraceCooperativeSupport
{
  bool supported = false;
  int device = -1;
  int compute_major = 0;
  int compute_minor = 0;
  int multiprocessors = 0;
  int threads_per_block = 0;
  int maximum_active_blocks_per_multiprocessor = 0;
  int cooperative_blocks = 0;
};

struct CudaNativeTraceWorkspaceView
{
  double* basis = nullptr;
  std::size_t basis_doubles = 0;
  double* vectors = nullptr;
  std::size_t vector_doubles = 0;
  double* reduction = nullptr;
  std::size_t reduction_doubles = 0;
  double* small = nullptr;
  std::size_t small_doubles = 0;
  int* permutation_marks = nullptr;
  std::size_t permutation_integers = 0;
  double* device_control = nullptr;
};

struct CudaNativeIlu0FactorView
{
  const int* offsets = nullptr;
  const int* columns = nullptr;
  const double* values = nullptr;
  const int* permutation = nullptr;
  const double* inverse_diagonal = nullptr;
  int rows = 0;
  int nonzeros = 0;
  int lower_jacobi_iterations = 0;
  int upper_jacobi_iterations = 0;
};

struct CudaNativeElementSchwarzView
{
  const double* patch_matrix = nullptr;
  const double* patch_lu = nullptr;
  const int* pivots = nullptr;
  const int* signed_element_true_dofs = nullptr;
  const int* true_incidence_slots = nullptr;
  double* element_rhs = nullptr;
  double* element_solution = nullptr;
  int elements = 0;
  int patch_rows = 0;
  int true_rows = 0;
  int minimum_multiplicity = 0;
  int maximum_multiplicity = 0;
};

enum class CudaNativeTraceOperatorKind : std::uint8_t
{
  assembled_csr_fp64 = 1,
  element_matrix_free_fp64 = 2
};

struct CudaNativeTraceOperatorView
{
  CudaNativeTraceOperatorKind kind =
      CudaNativeTraceOperatorKind::assembled_csr_fp64;
  const double* element_schur = nullptr;
};

struct CudaNativeTraceSolveObservation
{
  int status = 0;
  std::uint32_t failure_mask = 0;
  int iterations = 0;
  int restart_cycles = 0;
  int preconditioner_applications = 0;
  int operator_applications = 0;
  int actual_residual_checks = 0;
  double rhs_norm = std::numeric_limits<double>::infinity();
  double initial_residual_norm = std::numeric_limits<double>::infinity();
  double estimated_residual_norm = std::numeric_limits<double>::infinity();
  double actual_residual_norm = std::numeric_limits<double>::infinity();
  double relative_residual_norm = std::numeric_limits<double>::infinity();
  bool ready_for_true_residual = false;
  bool finite = false;
};

inline constexpr std::size_t native_trace_host_control_double_capacity = 16;

struct CudaTrueMatrixDeviceView
{
  const int* diag_offsets = nullptr;
  const int* diag_columns = nullptr;
  double* diag_values = nullptr;
  const int* offd_offsets = nullptr;
  const int* offd_columns = nullptr;
  double* offd_values = nullptr;
  int rows = 0;
  int diag_columns_count = 0;
  int diag_nonzeros = 0;
  int offd_nonzeros = 0;
};

inline constexpr int cuda_rank1_equilibration_passes = 2;
inline constexpr double cuda_rank1_equilibration_minimum_scale =
    0x1p-40;
inline constexpr double cuda_rank1_equilibration_maximum_scale =
    0x1p40;
inline constexpr std::size_t cuda_rank1_equilibration_control_doubles = 4;

/** One bounded observation of an FP64 two-sided equilibration build. */
struct CudaEquilibrationObservation
{
  int passes = 0;
  double minimum_left_scale = std::numeric_limits<double>::infinity();
  double maximum_left_scale = 0.0;
  double minimum_right_scale = std::numeric_limits<double>::infinity();
  double maximum_right_scale = 0.0;
  bool finite = false;
};

/**
 * Private persistent ownership for the explicit rank-one FlexGMRES/MGR leaf.
 * The vectors are fixed-size configuration storage required by HYPRE; the
 * scalar generation/object identities bind one retained setup and are not a
 * solve history. No per-iteration records are retained.
 */
struct CudaRankOneMgrState
{
  HYPRE_Solver fgmres = nullptr;
  HYPRE_Solver mgr = nullptr;
  HYPRE_Solver f_ilu = nullptr;
  HYPRE_Solver coarse_amg = nullptr;
  std::vector<HYPRE_Int> point_markers{};
  std::array<HYPRE_Int, 1> coarse_point_counts{{1}};
  std::array<HYPRE_Int, 1> coarse_point_indexes{{4}};
  std::array<HYPRE_Int*, 1> coarse_point_index_levels{{nullptr}};
  std::array<HYPRE_Int, 1> f_relax_types{{32}};
  std::array<HYPRE_Int, 1> coarse_grid_methods{{3}};
  std::uint64_t setup_calls = 0;
  std::uint64_t preconditioner_applications = 0;
  std::uint64_t pending_matrix_generation = 0;
  std::uint64_t last_setup_matrix_generation = 0;
  std::uintptr_t setup_matrix_token = 0;
  std::uintptr_t setup_rhs_token = 0;
  std::uintptr_t setup_solution_token = 0;
  double configured_relative_tolerance =
      std::numeric_limits<double>::infinity();
  bool setup_generation_pending = false;
  bool has_setup_generation = false;
  bool configured = false;
};

/**
 * Private bounded state for one attempt's ILU factor lifetime.  Pointer
 * identities are invariants only; none are published as evidence and no
 * per-solve history is retained.
 */
struct CudaIluAttemptFactorState
{
  AttemptIdentity attempt{};
  CudaIluFactorReusePolicy policy =
      CudaIluFactorReusePolicy::not_applicable;
  std::uintptr_t matrix_object_token = 0;
  std::uintptr_t hypre_matrix_token = 0;
  std::uintptr_t diag_offsets_token = 0;
  std::uintptr_t diag_columns_token = 0;
  std::uintptr_t diag_values_token = 0;
  std::uintptr_t offd_offsets_token = 0;
  std::uintptr_t offd_columns_token = 0;
  std::uintptr_t offd_values_token = 0;
  std::uintptr_t ilu_handle_token = 0;
  std::uintptr_t factor_matrix_token = 0;
  std::uintptr_t factor_offsets_token = 0;
  std::uintptr_t factor_columns_token = 0;
  std::uintptr_t factor_values_token = 0;
  std::uintptr_t factor_permutation_token = 0;
  std::uintptr_t native_inverse_diagonal_token = 0;
  std::uintptr_t jacobi_diagonal_vector_token = 0;
  std::uintptr_t jacobi_diagonal_values_token = 0;
  int matrix_rows = 0;
  int diag_nonzeros = 0;
  int offd_nonzeros = 0;
  int factor_rows = 0;
  int factor_nonzeros = 0;
  std::uint64_t factor_generation = 0;
  std::uint64_t last_solve_generation = 0;
  bool active = false;
  bool factor_ready = false;
};

/**
 * End one logical factor authority on every exit from the numerical attempt.
 * This guard owns no HYPRE resource and records no history; it only makes the
 * private constant-size authorization state fail closed before control returns
 * to staging, rollback, or commit code.
 */
class CudaIluFactorAttemptEndGuard final
{
public:
  explicit CudaIluFactorAttemptEndGuard(
      CudaIluAttemptFactorState& state) noexcept
    : state_(state)
  {}

  ~CudaIluFactorAttemptEndGuard() noexcept { state_ = {}; }

  CudaIluFactorAttemptEndGuard(const CudaIluFactorAttemptEndGuard&) = delete;
  CudaIluFactorAttemptEndGuard& operator=(
      const CudaIluFactorAttemptEndGuard&) = delete;
  CudaIluFactorAttemptEndGuard(CudaIluFactorAttemptEndGuard&&) = delete;
  CudaIluFactorAttemptEndGuard& operator=(
      CudaIluFactorAttemptEndGuard&&) = delete;

private:
  CudaIluAttemptFactorState& state_;
};

/** Constant-size derived-factor authority for one Schwarz attempt. */
struct CudaElementSchwarzAttemptState
{
  AttemptIdentity attempt{};
  std::uint64_t factor_generation = 0;
  std::uint64_t last_solve_generation = 0;
  bool active = false;
  bool factor_ready = false;
};

inline bool CudaElementSchwarzFactorTransitionIsExact(
    const CudaElementSchwarzAttemptState& state,
    const AttemptIdentity& attempt, const std::uint64_t matrix_generation,
    const bool same_matrix_retry) noexcept
{
  if (!state.active ||
      !::dg::hdg::runtime::detail::SameAttempt(state.attempt, attempt))
    return false;
  if (same_matrix_retry)
    return state.factor_ready &&
        state.factor_generation == matrix_generation &&
        state.last_solve_generation == matrix_generation;
  return matrix_generation > state.last_solve_generation &&
      (!state.factor_ready ||
       state.factor_generation == state.last_solve_generation);
}

class CudaElementSchwarzAttemptEndGuard final
{
public:
  explicit CudaElementSchwarzAttemptEndGuard(
      CudaElementSchwarzAttemptState& state) noexcept : state_(state) {}
  ~CudaElementSchwarzAttemptEndGuard() noexcept { state_ = {}; }
  CudaElementSchwarzAttemptEndGuard(
      const CudaElementSchwarzAttemptEndGuard&) = delete;
  CudaElementSchwarzAttemptEndGuard& operator=(
      const CudaElementSchwarzAttemptEndGuard&) = delete;

private:
  CudaElementSchwarzAttemptState& state_;
};

constexpr bool CudaBeginIluFactorAttempt(
    CudaIluAttemptFactorState& state, const AttemptIdentity& attempt,
    const CudaIluFactorReusePolicy policy,
    const std::uintptr_t matrix_object_token,
    const std::uintptr_t diag_values_token,
    const std::uintptr_t offd_values_token,
    const int diag_nonzeros, const int offd_nonzeros) noexcept
{
  state = {};
  const bool known_policy =
      policy == CudaIluFactorReusePolicy::not_applicable ||
      policy == CudaIluFactorReusePolicy::refresh_each_trace_solve ||
      policy == CudaIluFactorReusePolicy::first_trace_solve_per_attempt;
  const bool fixed_matrix = matrix_object_token != 0 && diag_nonzeros > 0 &&
      diag_values_token != 0 && offd_nonzeros >= 0 &&
      (offd_nonzeros == 0 || offd_values_token != 0);
  if (!known_policy || !fixed_matrix) return false;
  state.attempt = attempt;
  state.policy = policy;
  state.matrix_object_token = matrix_object_token;
  state.diag_values_token = diag_values_token;
  state.offd_values_token = offd_values_token;
  state.diag_nonzeros = diag_nonzeros;
  state.offd_nonzeros = offd_nonzeros;
  state.active = true;
  return true;
}

inline bool CudaIluFactorAttemptMatches(
    const CudaIluAttemptFactorState& state,
    const AttemptIdentity& attempt,
    const CudaIluFactorReusePolicy policy) noexcept
{
  return state.active && state.policy == policy &&
      ::dg::hdg::runtime::detail::SameAttempt(state.attempt, attempt);
}

inline bool CudaOrdinaryRankOneRefreshAttemptIsExact(
    const CudaIluAttemptFactorState& state,
    const AttemptIdentity& attempt,
    const std::uintptr_t matrix_object_token,
    const std::uintptr_t diag_values_token,
    const std::uintptr_t offd_values_token,
    const int diag_nonzeros, const int offd_nonzeros) noexcept
{
  return CudaIluFactorAttemptMatches(
          state, attempt,
          CudaIluFactorReusePolicy::refresh_each_trace_solve) &&
      state.matrix_object_token == matrix_object_token &&
      state.diag_values_token == diag_values_token &&
      state.offd_values_token == offd_values_token &&
      state.diag_nonzeros == diag_nonzeros &&
      state.offd_nonzeros == offd_nonzeros &&
      state.hypre_matrix_token == 0 &&
      state.diag_offsets_token == 0 &&
      state.diag_columns_token == 0 &&
      state.offd_offsets_token == 0 &&
      state.offd_columns_token == 0 &&
      state.ilu_handle_token == 0 &&
      state.factor_matrix_token == 0 &&
      state.factor_offsets_token == 0 &&
      state.factor_columns_token == 0 &&
      state.factor_values_token == 0 &&
      state.factor_permutation_token == 0 &&
      state.native_inverse_diagonal_token == 0 &&
      state.jacobi_diagonal_vector_token == 0 &&
      state.jacobi_diagonal_values_token == 0 &&
      state.matrix_rows == 0 && state.factor_rows == 0 &&
      state.factor_nonzeros == 0 && state.factor_generation == 0 &&
      state.last_solve_generation == 0 && !state.factor_ready;
}

constexpr bool CudaIluFactorRefreshRequired(
    const CudaIluAttemptFactorState& state) noexcept
{
  return state.policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve ||
      !state.factor_ready;
}

constexpr bool CudaIluRefreshDecisionRequiresConsensus(
    const CudaIluFactorReusePolicy policy) noexcept
{
  return policy ==
      CudaIluFactorReusePolicy::first_trace_solve_per_attempt;
}

/** One checked generic arena plan; BindArenaViews consumes these exact slices. */
struct CudaGenericArenaLayout
{
  std::size_t elements = 0;
  std::size_t interior = 0;
  std::size_t local_trace = 0;
  std::size_t true_trace = 0;
  std::size_t element_coordinates = 0;
  std::size_t element_trace = 0;
  std::size_t element_interior2 = 0;
  std::size_t element_coupling = 0;
  std::size_t local_matrix = 0;
  std::size_t volume_shape = 0;
  std::size_t volume_derivative = 0;
  std::size_t volume_weight = 0;
  std::size_t volume_position = 0;
  std::size_t face_interior = 0;
  std::size_t face_trace = 0;
  std::size_t face_weight = 0;
  std::size_t face_normal = 0;
  std::size_t face_h = 0;
  std::size_t element_faces = 0;
  std::size_t boundary_reference = 0;
  std::size_t boundary_pressure = 0;
  std::size_t native_ilu_inverse_diagonal = 0;
  std::size_t solve_workspace = 0;
  std::size_t two_exchange_keys = 0;
};

// Allocation-free, fail-neutral setup provenance.  The implementation only
// samples HYPRE_GetError/cudaPeekAtLastError; it never clears or synchronizes.
void CudaBeginTrueSystemDiagnostic(int rank) noexcept;
void CudaEndTrueSystemDiagnostic(bool result) noexcept;
void CudaTrueSystemDiagnosticMarker(const char* stage, const char* edge,
                                    bool result,
                                    std::uint64_t generation = 0) noexcept;

struct CudaTrueSystemDiagnosticScope
{
  const char* stage = nullptr;
  std::uint64_t generation = 0;
  bool result = false;

  CudaTrueSystemDiagnosticScope(const char* value,
                                const std::uint64_t value_generation = 0)
      noexcept : stage(value), generation(value_generation)
  {
    CudaTrueSystemDiagnosticMarker(stage, "enter", false, generation);
  }
  ~CudaTrueSystemDiagnosticScope() noexcept
  {
    CudaTrueSystemDiagnosticMarker(stage, "exit", result, generation);
  }
};

struct CudaDeviceView
{
  double* slot_interior[2]{};
  double* old_interior = nullptr;
  double* work_interior = nullptr;
  // Dedicated persistent slice for caller-supplied static L2 source values.
  // It is always allocated; legacy manufactured forcing ignores it.
  double* static_source = nullptr;
  // Borrowed-only face scratch for the private frozen-coefficient Jacobian
  // audit.  Production assembly computes LLF speeds directly from its current
  // candidate and never reads this slice.
  double* borrowed_lagged_llf_tau = nullptr;
  double* initial_current_interior = nullptr;
  double* base_interior = nullptr;
  // Compact four-component element-local staging used only for host output.
  // The nonlinear trial is written directly into work_interior.
  double* output_interior = nullptr;
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
  // Element-volume / face-measure length used by the IGR auxiliary closure.
  double* face_h = nullptr;
  // Borrowed-host physical-boundary data.  A zero kind is an ordinary
  // periodic/shared/interior trace coupling; nonzero values select a copied
  // value-only exterior policy for this element-local face.
  double* boundary_reference = nullptr;
  double* boundary_pressure = nullptr;
  double* status = nullptr;
  double* scalar = nullptr;
  std::uint64_t* hash_words = nullptr;
  int* ticket_words = nullptr;
  int* status_flag = nullptr;
  int* signed_element_trace = nullptr;
  int* boundary_policy_kind = nullptr;
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
  // Rank-one-only aliases of otherwise inactive exchange slices. They are
  // immutable topology or persistent numerical storage, never observations.
  double* element_schwarz_patch_matrix = nullptr;
  double* element_schwarz_patch_lu = nullptr;
  double* element_schwarz_rhs = nullptr;
  double* element_schwarz_solution = nullptr;
  int* element_schwarz_first_source = nullptr;
  int* element_schwarz_second_source = nullptr;
  int* element_schwarz_signed_true_dof = nullptr;
  int* element_schwarz_true_incidence_slots = nullptr;
  int* element_schwarz_pivots = nullptr;
};

struct CudaMeshShape
{
  int elements = 0;
  int interior_components = conserved_components;
  int trace_components = conserved_components;
  int scalar_interior_dofs = 0;
  int scalar_face_dofs = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  int volume_assembly_order = 0;
  int face_assembly_order = 0;
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
  int order = cuda_w8a::polynomial_order;
  double igr_alpha = 0.0;
  double igr_beta = 0.0;
  double boundary_gamma = 1.4;
  bool igr_active = false;
  bool borrowed_static_source = false;
  bool borrowed_host_inputs = false;
  bool borrowed_static_boundary_policies = false;
};

enum class CudaHypreSetupTransition : std::uint8_t
{
  bind_preconditioner,
  set_operator,
  ilu_type,
  ilu_fill,
  ilu_reordering,
  ilu_tri_solve_configured,
  ilu_iterative_setup_off,
  ilu_application_iterations,
  ilu_tol0,
  ilu_print0,
  rebind_preconditioner,
  gmres_settings,
  mult
};

struct CudaHypreSetupState
{
  inline static constexpr std::size_t configured_transition_count = 12;
  inline static constexpr std::size_t solved_transition_count = 13;

  std::uint64_t matrix_generation = 0;
  // One current finite-state ordinal, never an action/event history.
  std::size_t next_transition = 0;
  CudaTracePreconditionerKind trace_preconditioner =
      CudaTracePreconditionerKind::unconfigured;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  int ilu_application_iterations = 0;
  CudaIluTriangularSolveKind ilu_triangular_solve_kind =
      CudaIluTriangularSolveKind::not_applicable;
  int ilu_lower_jacobi_iterations = 0;
  int ilu_upper_jacobi_iterations = 0;
  bool invalid_transition = false;

  constexpr bool Advance(const CudaHypreSetupTransition transition) noexcept
  {
    if (invalid_transition ||
        !TransitionExact(next_transition, transition))
    {
      invalid_transition = true;
      return false;
    }
    ++next_transition;
    return true;
  }

  constexpr bool ConfigurationExactForGeneration(
      const std::uint64_t expected_generation,
      const CudaTracePolicyDescriptor& expected_policy) const noexcept
  {
    return next_transition == configured_transition_count &&
        PrefixExactForGeneration(expected_generation, expected_policy);
  }

  constexpr bool SolveExactForGeneration(
      const std::uint64_t expected_generation,
      const CudaTracePolicyDescriptor& expected_policy) const noexcept
  {
    return next_transition == solved_transition_count &&
        PrefixExactForGeneration(expected_generation, expected_policy);
  }

  // Compatibility overloads retain the established engine-default test seam.
  constexpr bool ConfigurationExactForGeneration(
      const std::uint64_t expected_generation,
      const generic::CudaEngineKind expected_engine,
      const CudaTracePreconditionerKind expected_kind,
      const bool igr_active) const noexcept
  {
    const CudaTracePolicyDescriptor policy = ResolveCudaTracePolicy(
        expected_engine, igr_active, CudaTracePolicyKind::engine_default);
    return expected_kind == policy.trace_preconditioner &&
        ConfigurationExactForGeneration(expected_generation, policy);
  }

  constexpr bool SolveExactForGeneration(
      const std::uint64_t expected_generation,
      const generic::CudaEngineKind expected_engine,
      const CudaTracePreconditionerKind expected_kind,
      const bool igr_active) const noexcept
  {
    const CudaTracePolicyDescriptor policy = ResolveCudaTracePolicy(
        expected_engine, igr_active, CudaTracePolicyKind::engine_default);
    return expected_kind == policy.trace_preconditioner &&
        SolveExactForGeneration(expected_generation, policy);
  }

private:
  static constexpr bool TransitionExact(
      const std::size_t ordinal,
      const CudaHypreSetupTransition transition) noexcept
  {
    switch (ordinal)
    {
    case 0:
      return transition == CudaHypreSetupTransition::bind_preconditioner;
    case 1:
      return transition == CudaHypreSetupTransition::set_operator;
    case 2:
      return transition == CudaHypreSetupTransition::ilu_type;
    case 3:
      return transition == CudaHypreSetupTransition::ilu_fill;
    case 4:
      return transition == CudaHypreSetupTransition::ilu_reordering;
    case 5:
      return transition ==
          CudaHypreSetupTransition::ilu_tri_solve_configured;
    case 6:
      return transition ==
          CudaHypreSetupTransition::ilu_iterative_setup_off;
    case 7:
      return transition ==
          CudaHypreSetupTransition::ilu_application_iterations;
    case 8:
      return transition == CudaHypreSetupTransition::ilu_tol0;
    case 9:
      return transition == CudaHypreSetupTransition::ilu_print0;
    case 10:
      return transition == CudaHypreSetupTransition::rebind_preconditioner;
    case 11:
      return transition == CudaHypreSetupTransition::gmres_settings;
    case 12:
      return transition == CudaHypreSetupTransition::mult;
    default:
      return false;
    }
  }

  constexpr bool PrefixExactForGeneration(
      const std::uint64_t expected_generation,
      const CudaTracePolicyDescriptor& expected_policy) const noexcept
  {
    const CudaTracePreconditionerKind expected_kind =
        expected_policy.trace_preconditioner;
    const int expected_type = expected_policy.ilu_type;
    const int expected_fill = expected_policy.ilu_level_of_fill;
    const int expected_reordering =
        expected_policy.ilu_local_reordering;
    const int expected_application_iterations =
        expected_policy.ilu_application_iterations;
    const CudaIluTriangularSolveKind expected_triangular_solve_kind =
        expected_policy.ilu_triangular_solve_kind;
    const int expected_lower_jacobi_iterations =
        expected_policy.ilu_lower_jacobi_iterations;
    const int expected_upper_jacobi_iterations =
        expected_policy.ilu_upper_jacobi_iterations;
    const bool selected_policy_valid =
        CudaTracePolicyDescriptorIsExact(expected_policy) &&
        expected_kind == CudaTracePreconditionerKind::hypre_ilu0;
    if (invalid_transition || matrix_generation != expected_generation ||
        !selected_policy_valid || trace_preconditioner != expected_kind ||
        ilu_type != expected_type || ilu_level_of_fill != expected_fill ||
        ilu_local_reordering != expected_reordering ||
        ilu_application_iterations != expected_application_iterations ||
        ilu_triangular_solve_kind != expected_triangular_solve_kind ||
        ilu_lower_jacobi_iterations != expected_lower_jacobi_iterations ||
        ilu_upper_jacobi_iterations != expected_upper_jacobi_iterations ||
        expected_type != 0 || expected_fill != 0 ||
        (expected_reordering != 0 && expected_reordering != 1) ||
        expected_application_iterations != 1 ||
        (expected_triangular_solve_kind ==
             CudaIluTriangularSolveKind::direct &&
         (expected_lower_jacobi_iterations != 0 ||
          expected_upper_jacobi_iterations != 0)) ||
        (expected_triangular_solve_kind ==
             CudaIluTriangularSolveKind::iterative_jacobi &&
         (expected_lower_jacobi_iterations <= 0 ||
          expected_upper_jacobi_iterations <= 0)) ||
        next_transition < configured_transition_count)
      return false;
    return true;
  }
};

struct CudaDiagScaleSetupState
{
  inline static constexpr std::size_t configured_transition_count = 4;
  inline static constexpr std::size_t solved_transition_count = 5;

  std::uint64_t matrix_generation = 0;
  // One current finite-state ordinal, never an action/event history.
  std::size_t next_transition = 0;
  generic::CudaEngineKind selected_engine =
      generic::CudaEngineKind::ordinary;
  CudaTracePreconditionerKind trace_preconditioner =
      CudaTracePreconditionerKind::unconfigured;
  bool igr_active = false;
  bool invalid_transition = false;

  constexpr bool Advance(const CudaHypreSetupTransition transition) noexcept
  {
    if (invalid_transition ||
        !TransitionExact(next_transition, transition))
    {
      invalid_transition = true;
      return false;
    }
    ++next_transition;
    return true;
  }

  constexpr bool ConfigurationExactForGeneration(
      const std::uint64_t expected_generation,
      const CudaTracePolicyDescriptor& expected_policy) const noexcept
  {
    return next_transition == configured_transition_count &&
        PrefixExactForGeneration(expected_generation, expected_policy);
  }

  constexpr bool SolveExactForGeneration(
      const std::uint64_t expected_generation,
      const CudaTracePolicyDescriptor& expected_policy) const noexcept
  {
    return next_transition == solved_transition_count &&
        PrefixExactForGeneration(expected_generation, expected_policy);
  }

  // Compatibility overloads retain the established engine-default test seam.
  constexpr bool ConfigurationExactForGeneration(
      const std::uint64_t expected_generation,
      const generic::CudaEngineKind expected_engine,
      const CudaTracePreconditionerKind expected_kind,
      const bool expected_igr_active) const noexcept
  {
    const CudaTracePolicyDescriptor policy = ResolveCudaTracePolicy(
        expected_engine, expected_igr_active,
        CudaTracePolicyKind::engine_default);
    return expected_kind == policy.trace_preconditioner &&
        ConfigurationExactForGeneration(expected_generation, policy);
  }

  constexpr bool SolveExactForGeneration(
      const std::uint64_t expected_generation,
      const generic::CudaEngineKind expected_engine,
      const CudaTracePreconditionerKind expected_kind,
      const bool expected_igr_active) const noexcept
  {
    const CudaTracePolicyDescriptor policy = ResolveCudaTracePolicy(
        expected_engine, expected_igr_active,
        CudaTracePolicyKind::engine_default);
    return expected_kind == policy.trace_preconditioner &&
        SolveExactForGeneration(expected_generation, policy);
  }

private:
  static constexpr bool TransitionExact(
      const std::size_t ordinal,
      const CudaHypreSetupTransition transition) noexcept
  {
    switch (ordinal)
    {
    case 0:
      return transition == CudaHypreSetupTransition::bind_preconditioner;
    case 1:
      return transition == CudaHypreSetupTransition::set_operator;
    case 2:
      return transition == CudaHypreSetupTransition::rebind_preconditioner;
    case 3:
      return transition == CudaHypreSetupTransition::gmres_settings;
    case 4:
      return transition == CudaHypreSetupTransition::mult;
    default:
      return false;
    }
  }

  constexpr bool PrefixExactForGeneration(
      const std::uint64_t expected_generation,
      const CudaTracePolicyDescriptor& expected_policy) const noexcept
  {
    const generic::CudaEngineKind expected_engine =
        expected_policy.selected_engine;
    const CudaTracePreconditionerKind expected_kind =
        expected_policy.trace_preconditioner;
    const bool expected_igr_active = expected_policy.igr_active;
    const bool selected_policy_valid =
        CudaTracePolicyDescriptorIsExact(expected_policy) &&
        !expected_igr_active &&
        expected_kind == CudaTracePreconditionerKind::hypre_diag_scale;
    if (invalid_transition || matrix_generation != expected_generation ||
        !selected_policy_valid || selected_engine != expected_engine ||
        trace_preconditioner != expected_kind ||
        igr_active != expected_igr_active ||
        next_transition < configured_transition_count)
      return false;
    return true;
  }
};

// Private, constexpr seam for focused host tests.  These are analytical
// bounds from HYPRE's restarted right-preconditioned GMRES control flow; no
// callback or HYPRE state is observed to obtain them.  HYPRE's zero-start
// first Jacobi iterate has no factor SpMV, so an L/U configuration contributes
// max(L-1,0) + max(U-1,0) factor-SpMV updates per ILU application.
struct CudaDerivedIluAccountingBoundsV1
{
  std::uint64_t ilu_preconditioner_callbacks_lower_bound = 0;
  std::uint64_t ilu_preconditioner_callbacks_upper_bound = 0;
  std::uint64_t ilu_inner_iterations_lower_bound = 0;
  std::uint64_t ilu_inner_iterations_upper_bound = 0;
  std::uint64_t ilu_jacobi_factor_spmv_updates_lower_bound = 0;
  std::uint64_t ilu_jacobi_factor_spmv_updates_upper_bound = 0;
  bool ilu_accounting_complete = false;
};

constexpr CudaDerivedIluAccountingBoundsV1
CudaDeriveIluAccountingBoundsV1(
    const std::uint64_t gmres_iterations,
    const std::uint64_t krylov_dimension,
    const std::uint64_t application_iterations,
    const std::uint64_t jacobi_factor_spmv_updates_per_application = 0)
    noexcept
{
  CudaDerivedIluAccountingBoundsV1 result{};
  if (krylov_dimension == 0 || application_iterations == 0)
    return result;
  if (gmres_iterations == 0)
  {
    result.ilu_accounting_complete = true;
    return result;
  }
  const std::uint64_t restart_cycles =
      gmres_iterations / krylov_dimension +
      static_cast<std::uint64_t>(gmres_iterations % krylov_dimension != 0);
  const std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  if (gmres_iterations > maximum - restart_cycles ||
      gmres_iterations > maximum - gmres_iterations)
    return result;
  result.ilu_preconditioner_callbacks_lower_bound =
      gmres_iterations + restart_cycles;
  result.ilu_preconditioner_callbacks_upper_bound =
      gmres_iterations + gmres_iterations;
  if (result.ilu_preconditioner_callbacks_lower_bound >
          maximum / application_iterations ||
      result.ilu_preconditioner_callbacks_upper_bound >
          maximum / application_iterations)
  {
    result = {};
    return result;
  }
  result.ilu_inner_iterations_lower_bound =
      result.ilu_preconditioner_callbacks_lower_bound *
          application_iterations;
  result.ilu_inner_iterations_upper_bound =
      result.ilu_preconditioner_callbacks_upper_bound *
          application_iterations;
  if (jacobi_factor_spmv_updates_per_application != 0 &&
      (result.ilu_inner_iterations_lower_bound >
           maximum / jacobi_factor_spmv_updates_per_application ||
       result.ilu_inner_iterations_upper_bound >
           maximum / jacobi_factor_spmv_updates_per_application))
  {
    result = {};
    return result;
  }
  result.ilu_jacobi_factor_spmv_updates_lower_bound =
      result.ilu_inner_iterations_lower_bound *
          jacobi_factor_spmv_updates_per_application;
  result.ilu_jacobi_factor_spmv_updates_upper_bound =
      result.ilu_inner_iterations_upper_bound *
          jacobi_factor_spmv_updates_per_application;
  result.ilu_accounting_complete = true;
  return result;
}

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
  bool configured_preconditioner = false;
  bool configured_ilu = false;
  bool configured_diagonal = false;
  bool configured_element_schwarz = false;
  bool configured_native_solver = false;
  bool configured_mgr = false;
  bool reported_tolerance_met = false;
  bool finite = false;
  std::uint64_t matrix_generation = 0;
  CudaTraceSolverKind trace_solver =
      CudaTraceSolverKind::unconfigured;
  CudaTracePreconditionerKind trace_preconditioner =
      CudaTracePreconditionerKind::unconfigured;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  int ilu_application_iterations = 0;
  CudaIluTriangularSolveKind ilu_triangular_solve_kind =
      CudaIluTriangularSolveKind::not_applicable;
  int ilu_lower_jacobi_iterations = 0;
  int ilu_upper_jacobi_iterations = 0;
  int krylov_dimension = 0;
  bool igr_active = false;
  CudaIluAccountingKind ilu_accounting_kind =
      CudaIluAccountingKind::unavailable;
  CudaIluFactorReusePolicy ilu_factor_reuse_policy =
      CudaIluFactorReusePolicy::not_applicable;
  std::uint64_t ilu_factor_setups = 0;
  std::uint64_t ilu_factor_reuses = 0;
  std::uint64_t ilu_preconditioner_callbacks_lower_bound = 0;
  std::uint64_t ilu_preconditioner_callbacks_upper_bound = 0;
  std::uint64_t ilu_inner_iterations_lower_bound = 0;
  std::uint64_t ilu_inner_iterations_upper_bound = 0;
  std::uint64_t ilu_jacobi_factor_spmv_updates_lower_bound = 0;
  std::uint64_t ilu_jacobi_factor_spmv_updates_upper_bound = 0;
  // One solve's age; the public aggregate retains only the maximum.
  std::uint64_t ilu_factor_age_generations = 0;
  bool ilu_accounting_complete = false;
  std::uint64_t mgr_setup_calls = 0;
  std::uint64_t mgr_preconditioner_applications = 0;
  std::uint64_t mgr_f_ilu_setups = 0;
  std::uint64_t mgr_f_ilu_applications = 0;
  bool mgr_setup_reused = false;
  std::uint64_t element_schwarz_patch_setups = 0;
  std::uint64_t element_schwarz_preconditioner_applications = 0;
  std::uint64_t native_matrix_free_trace_operator_applications = 0;
  std::uint64_t element_schwarz_equilibration_builds = 0;
  std::uint64_t element_schwarz_equilibration_passes = 0;
  int iterations = 0;
  double reported_residual = std::numeric_limits<double>::infinity();
};

constexpr bool CudaTraceSolveReadyForTrueResidualVerification(
    const CudaSolveDiagnostics& diagnostics,
    const int maximum_iterations) noexcept
{
  const bool solver_exact = diagnostics.trace_solver ==
          CudaTraceSolverKind::native_rank1_fp64
      ? diagnostics.configured_native_solver &&
          !diagnostics.configured_mgr
      : diagnostics.trace_solver ==
            CudaTraceSolverKind::hypre_fgmres_mgr_rank1
          ? diagnostics.configured_mgr &&
              !diagnostics.configured_native_solver
          : (diagnostics.trace_solver ==
                 CudaTraceSolverKind::hypre_parcsr_gmres ||
             diagnostics.trace_solver ==
                 CudaTraceSolverKind::hypre_host_staged_rank2) &&
              !diagnostics.configured_native_solver &&
              !diagnostics.configured_mgr;
  const bool mgr_work_exact = diagnostics.trace_solver !=
          CudaTraceSolverKind::hypre_fgmres_mgr_rank1 ||
      (((diagnostics.mgr_setup_calls == 1 &&
         !diagnostics.mgr_setup_reused) ||
        (diagnostics.mgr_setup_calls == 0 &&
         diagnostics.mgr_setup_reused)) &&
       diagnostics.mgr_f_ilu_setups == diagnostics.mgr_setup_calls &&
       diagnostics.mgr_f_ilu_applications ==
           diagnostics.mgr_preconditioner_applications &&
       (diagnostics.iterations == 0 ||
        diagnostics.mgr_preconditioner_applications > 0));
  const bool preconditioner_exact = diagnostics.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_mgr_field_split
      ? diagnostics.configured_mgr && !diagnostics.configured_ilu &&
          !diagnostics.configured_diagonal &&
          !diagnostics.configured_element_schwarz
      : CudaTracePreconditionerIsNativeElementSchwarz(
            diagnostics.trace_preconditioner)
          ? diagnostics.configured_element_schwarz &&
              !diagnostics.configured_ilu &&
              !diagnostics.configured_diagonal &&
              (diagnostics.iterations == 0 ||
               diagnostics.element_schwarz_preconditioner_applications > 0)
          : (diagnostics.configured_ilu !=
                 diagnostics.configured_diagonal) &&
              !diagnostics.configured_element_schwarz &&
          ((diagnostics.configured_ilu &&
            CudaTracePreconditionerIsIlu(
                diagnostics.trace_preconditioner)) ||
           (diagnostics.configured_diagonal &&
            diagnostics.trace_preconditioner ==
                CudaTracePreconditionerKind::hypre_diag_scale));
  return solver_exact && mgr_work_exact && diagnostics.device_policy &&
      diagnostics.configured_preconditioner &&
      preconditioner_exact &&
      diagnostics.iterations >= 0 && diagnostics.iterations <=
          maximum_iterations;
}

constexpr bool CudaInitializeIluEvidence(
    CudaTensorCoreEvidence& evidence,
    const CudaTracePreconditionerKind kind,
    const bool igr_active) noexcept
{
  const CudaTracePolicyDescriptor policy = ResolveCudaTracePolicy(
      evidence.selected_engine, igr_active, evidence.trace_policy);
  const bool selection_exact = CudaTracePolicyDescriptorIsExact(policy) &&
      kind == policy.trace_preconditioner;
  const CudaIluFactorReusePolicy factor_reuse_policy =
      policy.ilu_factor_reuse_policy;
  if (!CudaTracePreconditionerIsIlu(kind))
  {
    const bool empty = selection_exact && evidence.ilu_type == -1 &&
        evidence.ilu_level_of_fill == -1 &&
        evidence.ilu_local_reordering == -1 &&
        evidence.ilu_application_iterations == 0 &&
        evidence.ilu_triangular_solve_kind ==
            CudaIluTriangularSolveKind::not_applicable &&
        evidence.ilu_lower_jacobi_iterations == 0 &&
        evidence.ilu_upper_jacobi_iterations == 0 &&
        evidence.ilu_accounting_kind ==
            CudaIluAccountingKind::unavailable &&
        factor_reuse_policy == CudaIluFactorReusePolicy::not_applicable &&
        evidence.ilu_factor_reuse_policy ==
            CudaIluFactorReusePolicy::not_applicable &&
        evidence.ilu_factor_setups == 0 &&
        evidence.ilu_factor_reuses == 0 &&
        evidence.ilu_preconditioner_callbacks_lower_bound == 0 &&
        evidence.ilu_preconditioner_callbacks_upper_bound == 0 &&
        evidence.ilu_inner_iterations_lower_bound == 0 &&
        evidence.ilu_inner_iterations_upper_bound == 0 &&
        evidence.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
        evidence.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
        evidence.ilu_factor_age_generations == 0;
    evidence.ilu_accounting_complete = empty;
    return empty;
  }
  const int ilu_type = policy.ilu_type;
  const int level_of_fill = policy.ilu_level_of_fill;
  const int local_reordering = policy.ilu_local_reordering;
  const int application_iterations = policy.ilu_application_iterations;
  const CudaIluTriangularSolveKind triangular_solve_kind =
      policy.ilu_triangular_solve_kind;
  const int lower_jacobi_iterations =
      policy.ilu_lower_jacobi_iterations;
  const int upper_jacobi_iterations =
      policy.ilu_upper_jacobi_iterations;
  const bool exact_kind = selection_exact &&
      kind == CudaTracePreconditionerKind::hypre_ilu0;
  const bool empty = evidence.ilu_accounting_kind ==
          CudaIluAccountingKind::unavailable &&
      evidence.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::not_applicable &&
      evidence.ilu_factor_setups == 0 &&
      evidence.ilu_factor_reuses == 0 &&
      evidence.ilu_preconditioner_callbacks_lower_bound == 0 &&
      evidence.ilu_preconditioner_callbacks_upper_bound == 0 &&
      evidence.ilu_inner_iterations_lower_bound == 0 &&
      evidence.ilu_inner_iterations_upper_bound == 0 &&
      evidence.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
      evidence.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
      evidence.ilu_factor_age_generations == 0;
  if (!exact_kind || !empty || ilu_type != 0 || level_of_fill != 0 ||
      (local_reordering != 0 && local_reordering != 1) ||
      application_iterations != 1 ||
      (triangular_solve_kind == CudaIluTriangularSolveKind::direct &&
       (lower_jacobi_iterations != 0 || upper_jacobi_iterations != 0)) ||
      (triangular_solve_kind ==
           CudaIluTriangularSolveKind::iterative_jacobi &&
       (lower_jacobi_iterations <= 0 || upper_jacobi_iterations <= 0)) ||
      triangular_solve_kind ==
          CudaIluTriangularSolveKind::not_applicable)
  {
    evidence.ilu_accounting_complete = false;
    return false;
  }
  evidence.ilu_type = ilu_type;
  evidence.ilu_level_of_fill = level_of_fill;
  evidence.ilu_local_reordering = local_reordering;
  evidence.ilu_application_iterations = application_iterations;
  evidence.ilu_triangular_solve_kind = triangular_solve_kind;
  evidence.ilu_lower_jacobi_iterations = lower_jacobi_iterations;
  evidence.ilu_upper_jacobi_iterations = upper_jacobi_iterations;
  evidence.ilu_accounting_kind =
      CudaIluAccountingKind::derived_bounds_v1;
  evidence.ilu_factor_reuse_policy = factor_reuse_policy;
  evidence.ilu_factor_age_generations = 0;
  evidence.ilu_accounting_complete = true;
  return true;
}

// Return false only for a typed policy/configuration/operation-integrity
// violation.  Derived-bound arithmetic can mark the aggregate incomplete
// while returning true; it is qualification metadata, not the independently
// rebuilt current-matrix true-residual acceptance test.
constexpr bool CudaAccumulateIluSolveEvidence(
    CudaTensorCoreEvidence& evidence,
    const CudaSolveDiagnostics& diagnostics) noexcept
{
  const bool valid_dimensions = diagnostics.iterations >= 0 &&
      diagnostics.krylov_dimension > 0 &&
      diagnostics.ilu_application_iterations > 0 &&
      diagnostics.ilu_lower_jacobi_iterations >= 0 &&
      diagnostics.ilu_upper_jacobi_iterations >= 0;
  const std::uint64_t jacobi_factor_spmv_updates_per_application =
      valid_dimensions
      ? static_cast<std::uint64_t>(
            diagnostics.ilu_lower_jacobi_iterations > 0
                ? diagnostics.ilu_lower_jacobi_iterations - 1 : 0) +
          static_cast<std::uint64_t>(
            diagnostics.ilu_upper_jacobi_iterations > 0
                ? diagnostics.ilu_upper_jacobi_iterations - 1 : 0)
      : 0;
  const CudaDerivedIluAccountingBoundsV1 expected = valid_dimensions
      ? CudaDeriveIluAccountingBoundsV1(
            static_cast<std::uint64_t>(diagnostics.iterations),
            static_cast<std::uint64_t>(diagnostics.krylov_dimension),
            static_cast<std::uint64_t>(
                diagnostics.ilu_application_iterations),
            jacobi_factor_spmv_updates_per_application)
      : CudaDerivedIluAccountingBoundsV1{};
  const int expected_type =
      CudaIluType(diagnostics.trace_preconditioner);
  const int expected_fill =
      CudaIluLevelOfFill(diagnostics.trace_preconditioner);
  const int expected_reordering = CudaIluLocalReordering(
      diagnostics.trace_preconditioner, diagnostics.igr_active);
  const int expected_application_iterations =
      CudaIluApplicationIterations(diagnostics.trace_preconditioner);
  const CudaTracePolicyDescriptor policy = ResolveCudaTracePolicy(
      evidence.selected_engine, diagnostics.igr_active,
      evidence.trace_policy);
  const CudaIluTriangularSolveKind expected_triangular_solve_kind =
      policy.ilu_triangular_solve_kind;
  const int expected_lower_jacobi_iterations =
      policy.ilu_lower_jacobi_iterations;
  const int expected_upper_jacobi_iterations =
      policy.ilu_upper_jacobi_iterations;
  const CudaIluFactorReusePolicy expected_factor_reuse_policy =
      policy.ilu_factor_reuse_policy;
  const bool exact_policy = CudaTracePolicyDescriptorIsExact(policy) &&
      diagnostics.trace_preconditioner == policy.trace_preconditioner &&
      diagnostics.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      diagnostics.ilu_factor_reuse_policy ==
          expected_factor_reuse_policy &&
      evidence.ilu_factor_reuse_policy == expected_factor_reuse_policy;
  const bool first_factor_operation = evidence.ilu_factor_setups == 0 &&
      evidence.ilu_factor_reuses == 0;
  const bool factor_operation_exact =
      expected_factor_reuse_policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve
      ? diagnostics.ilu_factor_setups == 1 &&
          diagnostics.ilu_factor_reuses == 0 &&
          diagnostics.ilu_factor_age_generations == 0
      : expected_factor_reuse_policy ==
            CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
          (first_factor_operation
            ? diagnostics.ilu_factor_setups == 1 &&
                diagnostics.ilu_factor_reuses == 0 &&
                diagnostics.ilu_factor_age_generations == 0
            : evidence.ilu_factor_setups == 1 &&
                diagnostics.ilu_factor_setups == 0 &&
                diagnostics.ilu_factor_reuses == 1 &&
                evidence.ilu_factor_reuses <
                    std::numeric_limits<std::uint64_t>::max() &&
                evidence.ilu_factor_age_generations <=
                    evidence.ilu_factor_reuses &&
                diagnostics.ilu_factor_age_generations >=
                    evidence.ilu_factor_age_generations &&
                diagnostics.ilu_factor_age_generations <=
                    evidence.ilu_factor_age_generations + 1);
  const bool policy_integrity = exact_policy && factor_operation_exact &&
      diagnostics.configured_preconditioner &&
      diagnostics.configured_ilu && !diagnostics.configured_diagonal;
  const bool exact_configuration =
      diagnostics.ilu_type == expected_type &&
      diagnostics.ilu_level_of_fill == expected_fill &&
      diagnostics.ilu_local_reordering == expected_reordering &&
      diagnostics.ilu_application_iterations ==
          expected_application_iterations &&
      CudaTracePreconditionerIsIlu(diagnostics.trace_preconditioner) &&
      diagnostics.ilu_type == evidence.ilu_type &&
      diagnostics.ilu_level_of_fill == evidence.ilu_level_of_fill &&
      diagnostics.ilu_local_reordering == evidence.ilu_local_reordering &&
      diagnostics.ilu_application_iterations ==
          evidence.ilu_application_iterations &&
      diagnostics.ilu_triangular_solve_kind ==
          expected_triangular_solve_kind &&
      diagnostics.ilu_lower_jacobi_iterations ==
          expected_lower_jacobi_iterations &&
      diagnostics.ilu_upper_jacobi_iterations ==
          expected_upper_jacobi_iterations &&
      diagnostics.ilu_triangular_solve_kind ==
          evidence.ilu_triangular_solve_kind &&
      diagnostics.ilu_lower_jacobi_iterations ==
          evidence.ilu_lower_jacobi_iterations &&
      diagnostics.ilu_upper_jacobi_iterations ==
          evidence.ilu_upper_jacobi_iterations &&
      ((expected_triangular_solve_kind ==
            CudaIluTriangularSolveKind::direct &&
        expected_lower_jacobi_iterations == 0 &&
        expected_upper_jacobi_iterations == 0) ||
       (expected_triangular_solve_kind ==
            CudaIluTriangularSolveKind::iterative_jacobi &&
        expected_lower_jacobi_iterations > 0 &&
        expected_upper_jacobi_iterations > 0)) &&
      diagnostics.ilu_accounting_kind ==
          CudaIluAccountingKind::derived_bounds_v1 &&
      evidence.ilu_accounting_kind ==
          CudaIluAccountingKind::derived_bounds_v1;
  if (!policy_integrity || !exact_configuration)
  {
    evidence.ilu_accounting_complete = false;
    return false;
  }

  const bool operation_count_overflow =
      diagnostics.ilu_factor_setups >
          std::numeric_limits<std::uint64_t>::max() -
              evidence.ilu_factor_setups ||
      diagnostics.ilu_factor_reuses >
          std::numeric_limits<std::uint64_t>::max() -
              evidence.ilu_factor_reuses;
  if (operation_count_overflow)
  {
    evidence.ilu_factor_setups = diagnostics.ilu_factor_setups >
            std::numeric_limits<std::uint64_t>::max() -
                evidence.ilu_factor_setups
        ? std::numeric_limits<std::uint64_t>::max()
        : evidence.ilu_factor_setups + diagnostics.ilu_factor_setups;
    evidence.ilu_factor_reuses = diagnostics.ilu_factor_reuses >
            std::numeric_limits<std::uint64_t>::max() -
                evidence.ilu_factor_reuses
        ? std::numeric_limits<std::uint64_t>::max()
        : evidence.ilu_factor_reuses + diagnostics.ilu_factor_reuses;
    evidence.ilu_factor_age_generations = std::max(
        evidence.ilu_factor_age_generations,
        diagnostics.ilu_factor_age_generations);
    evidence.ilu_accounting_complete = false;
    return true;
  }
  evidence.ilu_factor_setups += diagnostics.ilu_factor_setups;
  evidence.ilu_factor_reuses += diagnostics.ilu_factor_reuses;
  evidence.ilu_factor_age_generations = std::max(
      evidence.ilu_factor_age_generations,
      diagnostics.ilu_factor_age_generations);

  const bool derived_bounds_exact = valid_dimensions &&
      expected.ilu_accounting_complete &&
      diagnostics.ilu_accounting_complete &&
      evidence.ilu_accounting_complete &&
      diagnostics.ilu_preconditioner_callbacks_lower_bound ==
          expected.ilu_preconditioner_callbacks_lower_bound &&
      diagnostics.ilu_preconditioner_callbacks_upper_bound ==
          expected.ilu_preconditioner_callbacks_upper_bound &&
      diagnostics.ilu_inner_iterations_lower_bound ==
          expected.ilu_inner_iterations_lower_bound &&
      diagnostics.ilu_inner_iterations_upper_bound ==
          expected.ilu_inner_iterations_upper_bound &&
      diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound ==
          expected.ilu_jacobi_factor_spmv_updates_lower_bound &&
      diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound ==
          expected.ilu_jacobi_factor_spmv_updates_upper_bound &&
      diagnostics.ilu_preconditioner_callbacks_lower_bound <=
          std::numeric_limits<std::uint64_t>::max() -
              evidence.ilu_preconditioner_callbacks_lower_bound &&
      diagnostics.ilu_preconditioner_callbacks_upper_bound <=
          std::numeric_limits<std::uint64_t>::max() -
              evidence.ilu_preconditioner_callbacks_upper_bound &&
      diagnostics.ilu_inner_iterations_lower_bound <=
          std::numeric_limits<std::uint64_t>::max() -
              evidence.ilu_inner_iterations_lower_bound &&
      diagnostics.ilu_inner_iterations_upper_bound <=
          std::numeric_limits<std::uint64_t>::max() -
              evidence.ilu_inner_iterations_upper_bound &&
      diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound <=
          std::numeric_limits<std::uint64_t>::max() -
              evidence.ilu_jacobi_factor_spmv_updates_lower_bound &&
      diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound <=
          std::numeric_limits<std::uint64_t>::max() -
              evidence.ilu_jacobi_factor_spmv_updates_upper_bound;
  if (!derived_bounds_exact)
  {
    evidence.ilu_accounting_complete = false;
    return true;
  }
  evidence.ilu_preconditioner_callbacks_lower_bound +=
      diagnostics.ilu_preconditioner_callbacks_lower_bound;
  evidence.ilu_preconditioner_callbacks_upper_bound +=
      diagnostics.ilu_preconditioner_callbacks_upper_bound;
  evidence.ilu_inner_iterations_lower_bound +=
      diagnostics.ilu_inner_iterations_lower_bound;
  evidence.ilu_inner_iterations_upper_bound +=
      diagnostics.ilu_inner_iterations_upper_bound;
  evidence.ilu_jacobi_factor_spmv_updates_lower_bound +=
      diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound;
  evidence.ilu_jacobi_factor_spmv_updates_upper_bound +=
      diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound;
  return true;
}

using CudaCopyOutFunction = bool (*)(
    void*, const void*, std::size_t, CudaTransferCategory,
    CudaTransferSite, std::size_t, std::size_t) noexcept;
using CudaSynchronizeFunction = bool (*)(CudaTransferSite) noexcept;

#if defined(__CUDACC__)
// Retained element factors are private CUDA implementation details.  Keeping
// these operations static and inline lets the production kernels and the CUDA
// leaf's test-only device seam exercise exactly the same ordering without
// adding an application ABI surface.
static __device__ inline bool CudaFactorRetained(
    double* lu, double* pivots, const int rows)
{
  double scale = 0.0;
  for (int entry = 0; entry < rows * rows; ++entry)
    scale = fmax(scale, fabs(lu[entry]));
  for (int column = 0; column < rows; ++column)
  {
    int pivot = column;
    for (int row = column + 1; row < rows; ++row)
      if (fabs(lu[row * rows + column]) >
          fabs(lu[pivot * rows + column]))
        pivot = row;
    pivots[column] = pivot;
    if (!(fabs(lu[pivot * rows + column]) >
          64.0 * 2.2204460492503131e-16 * scale))
      return false;
    if (pivot != column)
    {
      for (int entry = 0; entry < rows; ++entry)
      {
        const double value = lu[column * rows + entry];
        lu[column * rows + entry] = lu[pivot * rows + entry];
        lu[pivot * rows + entry] = value;
      }
    }
    for (int row = column + 1; row < rows; ++row)
    {
      lu[row * rows + column] /= lu[column * rows + column];
      for (int entry = column + 1; entry < rows; ++entry)
        lu[row * rows + entry] -=
            lu[row * rows + column] * lu[column * rows + entry];
    }
  }
  return true;
}

static __device__ inline void CudaSolveRetained(
    const double* lu, const double* pivots, const int rows,
    double* rhs, const int columns)
{
  // Complete row swaps move previously stored lower columns.  Apply the
  // retained permutation sequence before forward substitution so a delayed
  // pivot cannot permute an already-eliminated right-hand side.
  for (int row = 0; row < rows; ++row)
  {
    const int pivot = static_cast<int>(pivots[row]);
    if (pivot != row)
    {
      for (int column = 0; column < columns; ++column)
      {
        const double value = rhs[row * columns + column];
        rhs[row * columns + column] = rhs[pivot * columns + column];
        rhs[pivot * columns + column] = value;
      }
    }
  }
  for (int column = 0; column < rows; ++column)
    for (int row = column + 1; row < rows; ++row)
      for (int rhs_column = 0; rhs_column < columns; ++rhs_column)
        rhs[row * columns + rhs_column] -=
            lu[row * rows + column] *
            rhs[column * columns + rhs_column];
  for (int row = rows - 1; row >= 0; --row)
  {
    for (int rhs_column = 0; rhs_column < columns; ++rhs_column)
    {
      for (int column = row + 1; column < rows; ++column)
        rhs[row * columns + rhs_column] -=
            lu[row * rows + column] *
            rhs[column * columns + rhs_column];
      rhs[row * columns + rhs_column] /= lu[row * rows + row];
    }
  }
}

static __device__ inline bool CudaFactorSolveRetained(
    double* lu, double* pivots, const int rows,
    double* rhs, const int columns)
{
  if (!CudaFactorRetained(lu, pivots, rows)) { return false; }
  CudaSolveRetained(lu, pivots, rows, rhs, columns);
  return true;
}

static __device__ inline void CudaRecoverRetained(
    const double* lu, const double* pivots, const int interior_rows,
    const double* coupling, const int trace_rows,
    const double* trace_increment, const double* interior_rhs,
    double* recovered)
{
  for (int row = 0; row < interior_rows; ++row)
  {
    double value = interior_rhs[row];
    for (int column = 0; column < trace_rows; ++column)
      value -= coupling[row * trace_rows + column] *
          trace_increment[column];
    recovered[row] = value;
  }
  CudaSolveRetained(lu, pivots, interior_rows, recovered, 1);
}
#endif

bool CudaCopyHypreBigIntHost(HYPRE_BigInt* destination,
                             std::size_t destination_capacity,
                             const HYPRE_BigInt* source,
                             int count) noexcept;
bool CudaGetTrueMatrixStructure(mfem::HypreParMatrix& matrix,
                                const std::int64_t* owned_global,
                                int owned_count,
                                CudaTrueMatrixStructure& structure,
                                CudaCopyOutFunction copy_out,
                                CudaSynchronizeFunction synchronize) noexcept;
bool CudaGetTrueMatrixDeviceView(mfem::HypreParMatrix& matrix,
                                 CudaTrueMatrixDeviceView& view) noexcept;
bool CudaQueryNativeRankOneTraceResources(
    int rows, int krylov_dimension, bool igr_active,
    CudaNativeTraceWorkspaceRequirements& requirements,
    CudaNativeTraceCooperativeSupport& support) noexcept;
bool CudaLaunchNativeRankOneTraceSolver(
    const CudaTrueMatrixDeviceView& matrix, const double* rhs,
    double* solution, CudaTracePreconditionerKind preconditioner_kind,
    const CudaNativeIlu0FactorView& ilu_factor,
    const CudaNativeElementSchwarzView& element_schwarz,
    const CudaNativeTraceOperatorView& trace_operator,
    const CudaNativeTraceWorkspaceView& workspace,
    const CudaNativeTraceCooperativeSupport& support,
    double relative_tolerance, int maximum_iterations,
    int krylov_dimension, cudaStream_t stream) noexcept;
bool CudaLaunchSetupNativeElementSchwarz(
    const CudaDeviceView& view, const CudaMeshShape& shape,
    CudaTracePreconditionerKind preconditioner_kind) noexcept;
bool CudaLaunchAssembleNativeElementVector(
    const CudaDeviceView& view, const CudaMeshShape& shape,
    const double* element_values, double* true_values) noexcept;
bool CudaDecodeNativeRankOneTraceControl(
    const double* host_control, std::size_t host_control_double_slots,
    double relative_tolerance, int maximum_iterations,
    int krylov_dimension,
    CudaNativeTraceSolveObservation& observation) noexcept;
void CudaDestroyRankOneMgrState(CudaRankOneMgrState& state) noexcept;
bool CudaConfigureRankOneMgrState(
    CudaRankOneMgrState& state, MPI_Comm communicator, int rows,
    int scalar_trace_dofs, double relative_tolerance,
    int maximum_iterations, int krylov_dimension) noexcept;
bool CudaLaunchBuildRankOneEquilibratedSystem(
    const CudaTrueMatrixDeviceView& original,
    const CudaTrueMatrixDeviceView& equilibrated,
    const double* original_rhs, double* equilibrated_rhs,
    double* left_scale, double* right_scale, double* row_norm,
    double* column_norm, double* control, int* status, int passes,
    cudaStream_t stream) noexcept;
bool CudaLaunchUnscaleRankOneSolution(
    double* solution, const double* right_scale, int rows,
    cudaStream_t stream) noexcept;
bool CudaSolveRankOneMgr(
    CudaRankOneMgrState& state, mfem::HypreParMatrix& matrix,
    mfem::HypreParVector& rhs, mfem::HypreParVector& solution,
    const CudaTracePolicyDescriptor& policy, double relative_tolerance,
    int maximum_iterations, int krylov_dimension,
    std::uint64_t matrix_generation,
    CudaSolveDiagnostics& diagnostics,
    bool same_matrix_retry = false) noexcept;
bool CudaPrepareNativeRankOneIluFactor(
    mfem::HypreParMatrix& matrix, mfem::HypreParVector& rhs,
    mfem::HypreParVector& solution, mfem::HypreILU& ilu,
    const CudaTracePolicyDescriptor& policy,
    const AttemptIdentity& attempt, std::uint64_t matrix_generation,
    double* inverse_diagonal, std::size_t inverse_diagonal_count,
    CudaIluAttemptFactorState& factor_state,
    CudaNativeIlu0FactorView& factor_view,
    CudaSolveDiagnostics& diagnostics,
    bool same_matrix_retry = false) noexcept;
bool CudaConfigurePersistentHypre(mfem::HypreParMatrix& matrix,
                                  mfem::HypreILU& ilu,
                                  mfem::HypreGMRES& gmres,
                                  int ranks,
                                  const CudaTracePolicyDescriptor& policy,
                                  double relative_tolerance,
                                  int maximum_iterations,
                                  int krylov_dimension,
                                  std::uint64_t matrix_generation,
                                  CudaHypreSetupState& state)
    noexcept;
bool CudaConfigurePersistentDiagonal(mfem::HypreParMatrix& matrix,
                                     mfem::HypreDiagScale& diagonal,
                                     mfem::HypreGMRES& gmres,
                                     const CudaTracePolicyDescriptor& policy,
                                     double relative_tolerance,
                                     int maximum_iterations,
                                     int krylov_dimension,
                                     std::uint64_t matrix_generation,
                                     CudaDiagScaleSetupState& state)
    noexcept;
bool CudaOrdinaryRankOneRefreshFactorAndSolve(
    mfem::HypreParMatrix& matrix,
    mfem::HypreParVector& rhs,
    mfem::HypreParVector& solution,
    mfem::HypreILU& ilu,
    mfem::HypreGMRES& gmres,
    int ranks,
    MPI_Comm communicator,
    const CudaTracePolicyDescriptor& policy,
    double relative_tolerance,
    int maximum_iterations,
    int krylov_dimension,
    std::uint64_t matrix_generation,
    CudaSolveDiagnostics& diagnostics) noexcept;
bool CudaRefreshFactorAndSolve(mfem::HypreParMatrix& matrix,
                               mfem::HypreParVector& rhs,
                               mfem::HypreParVector& solution,
                               mfem::HypreILU& ilu,
                               mfem::HypreGMRES& gmres,
                               int ranks,
                               MPI_Comm communicator,
                               const CudaTracePolicyDescriptor& policy,
                               double relative_tolerance,
                               int maximum_iterations,
                               int krylov_dimension,
                               std::uint64_t matrix_generation,
                               CudaSolveDiagnostics& diagnostics) noexcept;
bool CudaSolveWithIluFactorPolicy(
    mfem::HypreParMatrix& matrix,
    mfem::HypreParVector& rhs,
    mfem::HypreParVector& solution,
    mfem::HypreILU& ilu,
    mfem::HypreGMRES& gmres,
    int ranks,
    MPI_Comm communicator,
    const CudaTracePolicyDescriptor& policy,
    const AttemptIdentity& attempt,
    double relative_tolerance,
    int maximum_iterations,
    int krylov_dimension,
    std::uint64_t matrix_generation,
    CudaIluAttemptFactorState& factor_state,
    CudaSolveDiagnostics& diagnostics,
    bool same_matrix_retry = false) noexcept;
bool CudaRefreshDiagonalAndSolve(mfem::HypreParMatrix& matrix,
                                 mfem::HypreParVector& rhs,
                                 mfem::HypreParVector& solution,
                                 mfem::HypreDiagScale& diagonal,
                                 mfem::HypreGMRES& gmres,
                                 int ranks,
                                 MPI_Comm communicator,
                                 const CudaTracePolicyDescriptor& policy,
                                 double relative_tolerance,
                                 int maximum_iterations,
                                 int krylov_dimension,
                                 std::uint64_t matrix_generation,
                                 CudaSolveDiagnostics& diagnostics) noexcept;
bool CudaHypreDevicePolicyValid(mfem::HypreParMatrix& matrix,
                                mfem::HypreParVector& rhs,
                                mfem::HypreParVector& solution) noexcept;

bool CudaLaunchTicket(int* words, CudaFailurePhase phase,
                      std::uint64_t attempt, std::uint64_t generation,
                      int device) noexcept;
bool CudaLaunchStatusReduction(const double* status, int count,
                               int* failure_flag) noexcept;
bool CudaLaunchGatherLocalTrace(const int* signed_map, int count,
                                const double* true_trace,
                                double* local_trace) noexcept;
bool CudaLaunchAdmissibility(const CudaDeviceView& view,
                             const CudaMeshShape& shape) noexcept;
bool CudaLaunchFreezeBorrowedCurrentLlfTau(const CudaDeviceView& view,
                                           const CudaMeshShape& shape) noexcept;
// Private numerical-leaf selection.  This is intentionally narrower than the
// application CUDA-engine selector and is exposed only inside the persistent
// implementation and its qualification friend.
enum class CudaBorrowedAssemblyKind : std::uint8_t
{
  ordinary_split = 0,
  tensor_fused = 1
};
enum class CudaBorrowedRecoveryKind : std::uint8_t
{
  ordinary_direct_lu = 0,
  tensor_retained_solve = 1,
  invalid = 2
};
constexpr CudaBorrowedRecoveryKind CudaBorrowedRecoveryForEngine(
    const generic::CudaEngineKind engine) noexcept
{
  return engine == generic::CudaEngineKind::ordinary
      ? CudaBorrowedRecoveryKind::ordinary_direct_lu
      : engine == generic::CudaEngineKind::tensor_core
      ? CudaBorrowedRecoveryKind::tensor_retained_solve
      : CudaBorrowedRecoveryKind::invalid;
}
constexpr bool CudaRecoveryRouteSupported(
    const generic::CudaEngineKind engine,
    const bool borrowed_host_inputs) noexcept
{
  return engine == generic::CudaEngineKind::ordinary ||
      (borrowed_host_inputs &&
       engine == generic::CudaEngineKind::tensor_core);
}
bool CudaLaunchBorrowedAssembly(
    const CudaDeviceView& view, const CudaMeshShape& shape, double dt,
    CudaBorrowedAssemblyKind kind) noexcept;
bool CudaLaunchAssembly(const CudaDeviceView& view,
                        const CudaMeshShape& shape, double dt,
                        generic::CudaEngineKind engine) noexcept;
bool CudaLaunchFrozenBorrowedLlfAssembly(
    const CudaDeviceView& view, const CudaMeshShape& shape,
    double dt) noexcept;
bool CudaLaunchCondensation(const CudaDeviceView& view,
                            const CudaMeshShape& shape,
                            generic::CudaEngineKind engine,
                            std::uint64_t& tensor_mma_tiles) noexcept;
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
                       const CudaMeshShape& shape,
                       generic::CudaEngineKind engine) noexcept;
bool CudaLaunchTrial(const CudaDeviceView& view,
                     const CudaMeshShape& shape, double alpha) noexcept;
bool CudaLaunchPackInteriorOutput(const CudaMeshShape& shape,
                                  const double* source,
                                  double* packed,
                                  int output_components) noexcept;
bool CudaLaunchLocalSquaredNorm(const double* first, int first_count,
                                const double* second, int second_count,
                                double* scratch, int scratch_count,
                                double* result) noexcept;
bool CudaLaunchMaximumWave(const CudaDeviceView& view,
                           const CudaMeshShape& shape,
                           const double* interior,
                           const double* element_trace,
                           double* result) noexcept;
bool CudaLaunchEulerConservation(const CudaDeviceView& view,
                                 const CudaMeshShape& shape,
                                 double dt, double* result) noexcept;

bool CudaLaunchHash(const double* values, int count,
                    std::uint64_t* words) noexcept;
bool CudaLaunchHashPair(const double* first, int first_count,
                        const double* second, int second_count,
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
  ~CudaPersistentNumerics() noexcept
  {
    CudaDestroyRankOneMgrState(rank_one_mgr_state_);
    const CudaArenaReleasePass first = ReleaseArenas();
    const CudaArenaReleasePass second = ReleaseArenas();
    if (first.complete && (!second.complete || !second.idempotent))
      ReportCudaArenaTeardownFailure(
          rank_, arena_idempotent_state_failure);
    ReportCudaArenaTeardownMarker(
        rank_, "destructor", "complete", arena_release_state_,
        first.complete && (!second.complete || !second.idempotent)
            ? arena_idempotent_state_failure : 0);
  }

  CudaPersistentNumerics(const CudaPersistentNumerics&) = delete;
  CudaPersistentNumerics& operator=(const CudaPersistentNumerics&) = delete;

  bool Initialize(const CudaConfig& config,
                  const CudaTracePolicyKind trace_policy,
                  CudaRunEvidence& evidence,
                  const RuntimeTupleDescriptor& tuple) noexcept
  {
    communicator_ = CudaCollectiveCommunicator(config);
    cuda_engine_ = config.cuda_engine;
    const bool igr_requested = config.use_generic_numerical_policy &&
        config.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::igr;
    trace_policy_descriptor_ = ResolveCudaTracePolicy(
        cuda_engine_, igr_requested, trace_policy);
    trace_solver_kind_ = CudaTraceSolverKind::unconfigured;
    native_trace_requirements_ = {};
    native_trace_support_ = {};
    native_trace_workspace_ = {};
    native_ilu_inverse_diagonal_ = nullptr;
    native_trace_host_control_.fill(0.0);
    trace_preconditioner_ = CudaTracePreconditionerKind::unconfigured;
    ilu_factor_state_ = {};
    element_schwarz_attempt_state_ = {};
    element_schwarz_plan_ = {};
    initialization_tensor_core_evidence_ = {};
    evidence.tensor_core.selected_engine = cuda_engine_;
    evidence.tensor_core.trace_policy = trace_policy;
    oracle_capture_requested_ = config.capture_oracle;
    TransportPolicy::ResetLedger();
    CudaPolicyContext setup{};
    setup.phase = CudaFailurePhase::setup;
    TransportPolicy::BeginAttempt(setup);
    // Admission of borrowed topology is deliberately ahead of cudaSetDevice,
    // HYPRE initialization, and arena allocation.  Unsupported degree/rank,
    // communicator disagreement and geometry are thus
    // collective host-side rejections rather than partially initialized GPU
    // state.
    if (!ValidateTopologyBeforeDevice(config))
    {
      ReportCudaSetupFailure(evidence.rank, "topology-preflight");
      return false;
    }
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
    newton_correction_linear_relative_tolerance_ =
        config.use_generic_numerical_policy
        ? config.numerical_policy.
              newton_correction_linear_relative_tolerance
        : 1.0e-13;
    final_coherent_linear_relative_tolerance_ =
        config.use_generic_numerical_policy
        ? config.numerical_policy.final_coherent_linear_relative_tolerance
        : 1.0e-13;
    true_residual_tightening_linear_relative_tolerance_ =
        config.use_generic_numerical_policy
        ? config.numerical_policy.
              true_residual_tightening_linear_relative_tolerance
        : 1.0e-13;
    true_residual_relative_tolerance_ = config.use_generic_numerical_policy
        ? config.numerical_policy.true_residual_relative_tolerance : 1.0e-10;
    maximum_linear_iterations_ = config.use_generic_numerical_policy
        ? generic::ResolveEulerHdgMaximumLinearIterations(
              config.numerical_policy.maximum_linear_iterations,
              shape_.global_true_trace_count)
        : 400;
    krylov_dimension_ = config.use_generic_numerical_policy
        ? generic::EulerHdgKrylovRestartDimension(
              shape_.global_true_trace_count)
        : (ranks_ == 1 ? shape_.true_trace_count
                       : shape_.global_true_trace_count);
    if (!CollectiveContinue(
                            newton_correction_linear_relative_tolerance_ > 0.0 &&
                            std::isfinite(
                                newton_correction_linear_relative_tolerance_) &&
                            final_coherent_linear_relative_tolerance_ > 0.0 &&
                            std::isfinite(
                                final_coherent_linear_relative_tolerance_) &&
                            true_residual_tightening_linear_relative_tolerance_ >
                                0.0 &&
                            std::isfinite(
                                true_residual_tightening_linear_relative_tolerance_) &&
                            true_residual_tightening_linear_relative_tolerance_ <=
                                newton_correction_linear_relative_tolerance_ &&
                            true_residual_tightening_linear_relative_tolerance_ <=
                                final_coherent_linear_relative_tolerance_ &&
                            true_residual_relative_tolerance_ > 0.0 &&
                            std::isfinite(true_residual_relative_tolerance_) &&
                            maximum_linear_iterations_ > 0 &&
                            krylov_dimension_ > 0))
    {
      ReportCudaSetupFailure(evidence.rank, "generic-linear-policy");
      return false;
    }
    trace_solver_kind_ = ResolveCudaTraceSolverKind(
        cuda_engine_, ranks_, host_staged_rank2_, borrowed_topology_,
        shape_.borrowed_host_inputs, config.use_generic_numerical_policy,
        trace_policy);
    bool trace_solver_ready = IsKnownCudaTraceSolverKind(trace_solver_kind_);
    if (trace_solver_ready && trace_solver_kind_ ==
            CudaTraceSolverKind::native_rank1_fp64)
    {
      trace_solver_ready = CudaQueryNativeRankOneTraceResources(
          shape_.true_trace_count, krylov_dimension_, shape_.igr_active,
          native_trace_requirements_, native_trace_support_) &&
          native_trace_requirements_.valid &&
          native_trace_support_.supported &&
          native_trace_support_.device == device_ &&
          native_trace_support_.compute_major == 12 &&
          native_trace_support_.compute_minor == 0 &&
          native_trace_support_.threads_per_block == 256 &&
          native_trace_support_.cooperative_blocks > 0 &&
          native_trace_requirements_.control_double_slots > 0 &&
          native_trace_requirements_.control_double_slots <=
              native_trace_host_control_.size();
    }
    if (!CollectiveContinue(trace_solver_ready))
    {
      ReportCudaSetupFailure(evidence.rank, "trace-solver-admission");
      return false;
    }
    const bool topology_storage_ready = AllocateTopologyStorage() &&
        BuildHostGeometryAndMaps(config) && BuildLocalTraceGraph();
    if (!CollectiveContinue(topology_storage_ready))
    {
      ReportCudaSetupFailure(evidence.rank, "build-topology-storage");
      return false;
    }
    if (!CollectiveContinue(InitializeBorrowedHostFields(config)))
    {
      ReportCudaSetupFailure(evidence.rank, "borrowed-host-fields");
      return false;
    }
    const int local_ghost_count =
        static_cast<int>(exchange_plan_.map.GhostCount());
    std::array<int, 2> ghost_counts{};
    const bool ghost_census_ready = MPI_Allgather(
        &local_ghost_count, 1, MPI_INT, ghost_counts.data(), 1, MPI_INT,
        communicator_) == MPI_SUCCESS;
    if (!CollectiveContinue(ghost_census_ready))
    {
      ReportCudaSetupFailure(evidence.rank, "ghost-census");
      return false;
    }
    host_exchange_bytes_ = ranks_ == 2
        ? static_cast<std::size_t>(ghost_counts[0] + ghost_counts[1]) *
            sizeof(double)
        : 0;
    if (!CollectiveContinue(AllocatorPolicy::PrepareDeviceArenaAllocator()))
    {
      ReportCudaSetupFailure(evidence.rank,
                             "prepare-device-arena-allocator");
      return false;
    }
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
    CudaBeginTrueSystemDiagnostic(rank_);
    const bool true_system_ready = BuildPersistentTrueSystem();
    CudaEndTrueSystemDiagnostic(true_system_ready);
    if (!CollectiveContinue(true_system_ready))
    {
      ReportCudaSetupFailure(evidence.rank,
                             "build-persistent-true-system");
      return false;
    }
    const bool trace_preconditioner_ready =
        CudaTracePolicyDescriptorIsExact(trace_policy_descriptor_) &&
        trace_policy_descriptor_.igr_active == shape_.igr_active &&
        trace_preconditioner_ ==
            trace_policy_descriptor_.trace_preconditioner;
    if (!CollectiveContinue(trace_preconditioner_ready))
    {
      ReportCudaSetupFailure(evidence.rank, "trace-preconditioner-selection");
      return false;
    }
    evidence.tensor_core.trace_preconditioner = trace_preconditioner_;
    evidence.tensor_core.trace_preconditioner_configured = true;
    evidence.tensor_core.ilu_type =
        initialization_tensor_core_evidence_.ilu_type;
    evidence.tensor_core.ilu_level_of_fill =
        initialization_tensor_core_evidence_.ilu_level_of_fill;
    evidence.tensor_core.ilu_local_reordering =
        initialization_tensor_core_evidence_.ilu_local_reordering;
    evidence.tensor_core.ilu_application_iterations =
        initialization_tensor_core_evidence_.ilu_application_iterations;
    evidence.tensor_core.ilu_triangular_solve_kind =
        initialization_tensor_core_evidence_.ilu_triangular_solve_kind;
    evidence.tensor_core.ilu_lower_jacobi_iterations =
        initialization_tensor_core_evidence_.ilu_lower_jacobi_iterations;
    evidence.tensor_core.ilu_upper_jacobi_iterations =
        initialization_tensor_core_evidence_.ilu_upper_jacobi_iterations;
    evidence.tensor_core.ilu_accounting_kind =
        initialization_tensor_core_evidence_.ilu_accounting_kind;
    evidence.tensor_core.ilu_factor_reuse_policy =
        initialization_tensor_core_evidence_.ilu_factor_reuse_policy;
    evidence.tensor_core.ilu_factor_setups =
        initialization_tensor_core_evidence_.ilu_factor_setups;
    evidence.tensor_core.ilu_factor_reuses =
        initialization_tensor_core_evidence_.ilu_factor_reuses;
    evidence.tensor_core.ilu_preconditioner_callbacks_lower_bound =
        initialization_tensor_core_evidence_.
            ilu_preconditioner_callbacks_lower_bound;
    evidence.tensor_core.ilu_preconditioner_callbacks_upper_bound =
        initialization_tensor_core_evidence_.
            ilu_preconditioner_callbacks_upper_bound;
    evidence.tensor_core.ilu_inner_iterations_lower_bound =
        initialization_tensor_core_evidence_.
            ilu_inner_iterations_lower_bound;
    evidence.tensor_core.ilu_inner_iterations_upper_bound =
        initialization_tensor_core_evidence_.
            ilu_inner_iterations_upper_bound;
    evidence.tensor_core.ilu_jacobi_factor_spmv_updates_lower_bound =
        initialization_tensor_core_evidence_.
            ilu_jacobi_factor_spmv_updates_lower_bound;
    evidence.tensor_core.ilu_jacobi_factor_spmv_updates_upper_bound =
        initialization_tensor_core_evidence_.
            ilu_jacobi_factor_spmv_updates_upper_bound;
    evidence.tensor_core.ilu_factor_age_generations =
        initialization_tensor_core_evidence_.ilu_factor_age_generations;
    evidence.tensor_core.ilu_accounting_complete =
        initialization_tensor_core_evidence_.ilu_accounting_complete;
    evidence.tensor_core.native_rank1_trace_solve_calls =
        initialization_tensor_core_evidence_.native_rank1_trace_solve_calls;
    evidence.tensor_core.hypre_trace_solve_calls =
        initialization_tensor_core_evidence_.hypre_trace_solve_calls;
    evidence.tensor_core.equilibration_builds =
        initialization_tensor_core_evidence_.equilibration_builds;
    evidence.tensor_core.equilibration_passes =
        initialization_tensor_core_evidence_.equilibration_passes;
    evidence.tensor_core.mgr_setup_calls =
        initialization_tensor_core_evidence_.mgr_setup_calls;
    evidence.tensor_core.mgr_preconditioner_applications =
        initialization_tensor_core_evidence_.
            mgr_preconditioner_applications;
    evidence.tensor_core.mgr_f_ilu_setups =
        initialization_tensor_core_evidence_.mgr_f_ilu_setups;
    evidence.tensor_core.mgr_f_ilu_applications =
        initialization_tensor_core_evidence_.mgr_f_ilu_applications;
    evidence.tensor_core.element_schwarz_patch_setups =
        initialization_tensor_core_evidence_.element_schwarz_patch_setups;
    evidence.tensor_core.element_schwarz_preconditioner_applications =
        initialization_tensor_core_evidence_.
            element_schwarz_preconditioner_applications;
    evidence.tensor_core.native_matrix_free_trace_operator_applications =
        initialization_tensor_core_evidence_.
            native_matrix_free_trace_operator_applications;
    evidence.tensor_core.element_schwarz_patch_rows =
        initialization_tensor_core_evidence_.element_schwarz_patch_rows;
    evidence.tensor_core.element_schwarz_minimum_multiplicity =
        initialization_tensor_core_evidence_.
            element_schwarz_minimum_multiplicity;
    evidence.tensor_core.element_schwarz_maximum_multiplicity =
        initialization_tensor_core_evidence_.
            element_schwarz_maximum_multiplicity;
    evidence.tensor_core.minimum_left_scale =
        initialization_tensor_core_evidence_.minimum_left_scale;
    evidence.tensor_core.maximum_left_scale =
        initialization_tensor_core_evidence_.maximum_left_scale;
    evidence.tensor_core.minimum_right_scale =
        initialization_tensor_core_evidence_.minimum_right_scale;
    evidence.tensor_core.maximum_right_scale =
        initialization_tensor_core_evidence_.maximum_right_scale;
    evidence.tensor_core.native_rank1_trace_solver_configured =
        initialization_tensor_core_evidence_.
            native_rank1_trace_solver_configured;
    evidence.tensor_core.native_element_schwarz_configured =
        initialization_tensor_core_evidence_.
            native_element_schwarz_configured;
    evidence.tensor_core.native_matrix_free_trace_operator_configured =
        initialization_tensor_core_evidence_.
            native_matrix_free_trace_operator_configured;
    evidence.tensor_core.hypre_mgr_rank1_trace_solver_configured =
        initialization_tensor_core_evidence_.
            hypre_mgr_rank1_trace_solver_configured;
    evidence.tensor_core.
        hypre_host_staged_rank2_trace_solver_configured =
            initialization_tensor_core_evidence_.
                hypre_host_staged_rank2_trace_solver_configured;
    evidence.tensor_core.trace_solver_accounting_complete =
        initialization_tensor_core_evidence_.
            trace_solver_accounting_complete;
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
        static_cast<std::size_t>(shape_.elements) * shape_.trace_dofs *
            sizeof(int));
    if (!AggregateHash(local_mesh_hash_, resources_.mesh_hash) ||
        !AggregateHash(local_partition_hash_, resources_.partition_hash) ||
        !AggregateHash(local_topology_hash_, resources_.topology_hash))
    {
      ReportCudaSetupFailure(evidence.rank, "aggregate-hashes");
      return false;
    }
    resources_.rank_count = static_cast<std::uint32_t>(ranks_);
    resources_.polynomial_order = static_cast<std::uint32_t>(config.order);
    resources_.layout = StableIdentity::Copy(runtime_layout_id);
    resources_.reuse = {true,
                        true,
                        true,
                        false,
                        1,
                        1,
                        double_arena_capacity_ * sizeof(double) +
                            integer_arena_capacity_ * sizeof(int),
                        WorkspaceBytes()};
    tuple_ = tuple;
    ReportCudaWarmLifetimeMarker(
        rank_, "baseline-high-water-sampling", "before", false);
    baseline_allocations_ = AllocatorPolicy::AllocationEvents();
    accepted_allocation_events_ = baseline_allocations_;
    allocation_plateau_sampled_ = false;
    ReportCudaWarmLifetimeMarker(
        rank_, "baseline-high-water-sampling", "after", true);
    baseline_w8_owned_allocation_calls_ =
        AllocatorPolicy::W8OwnedAllocationCalls();
    baseline_deallocation_events_ = AllocatorPolicy::DeallocationEvents();
    initialized_ = true;
    evidence.rank = rank_;
    evidence.ranks = ranks_;
    evidence.device = device_;
    evidence.physical_gpu_uuid = gpu_uuid_;
    evidence.same_physical_gpu = same_physical_gpu_;
    evidence.mfem_device_valid = device_resident_;
    evidence.hypre_cuda_umpire_device = hypre_policy_valid_;
    evidence.signed_owner_ghost_semantics = signed_owner_ghost_valid_;
    evidence.borrowed_topology = borrowed_topology_;
    evidence.host_staged_rank2 = ranks_ == 1 || host_staged_rank2_;
    evidence.no_host_solver_fallback = no_fallback_;
    return true;
  }

  state_slot CommittedSlot() const noexcept { return committed_slot_; }
  state_slot CandidateSlot() const noexcept { return candidate_slot_; }
  /** Mirror a final contract slot publication without moving any payload. */
  void SwapPublishedSlotHandles() noexcept
  {
    const state_slot prior_committed = committed_slot_;
    committed_slot_ = candidate_slot_;
    candidate_slot_ = prior_committed;
  }
  state_slot& WorkspaceSlot() noexcept { return workspace_slot_; }
  const PersistentResourceDescriptor& Resources() const noexcept
  {
    return resources_;
  }
  int rank() const noexcept { return rank_; }
  bool NoFallback() const noexcept { return no_fallback_; }
  bool DeviceResident() const noexcept { return device_resident_; }
  bool TensorCoreHardwareSupported() const noexcept
  { return tensor_core_hardware_supported_; }
  CudaTracePreconditionerKind TracePreconditioner() const noexcept
  { return trace_preconditioner_; }
  bool TracePreconditionerConfigured() const noexcept
  {
    return CudaTracePolicyDescriptorIsExact(trace_policy_descriptor_) &&
        trace_policy_descriptor_.igr_active == shape_.igr_active &&
        trace_preconditioner_ == trace_policy_descriptor_.trace_preconditioner;
  }
  const StateHash& LocalMeshHash() const noexcept { return local_mesh_hash_; }
  const StateHash& LocalPartitionHash() const noexcept
  { return local_partition_hash_; }
  const StateHash& LocalTopologyHash() const noexcept
  { return local_topology_hash_; }
  const std::array<CudaDeviceTicket, maximum_device_tickets>& Tickets()
      const noexcept { return tickets_; }
  std::size_t TicketCount() const noexcept { return ticket_count_; }

  bool CommittedBorrowedEulerDiagnostics(const state_slot& committed,
                                         const int order,
                                         double& maximum_wave_speed,
                                         double& backend_stable_dt) noexcept
  {
    maximum_wave_speed = 0.0;
    backend_stable_dt = std::numeric_limits<double>::quiet_NaN();
    const bool local_slot_valid = initialized_ && shape_.borrowed_host_inputs &&
        committed.interior.coefficients != nullptr &&
        committed.trace.coefficients != nullptr && local_hmin_ > 0.0 &&
        std::isfinite(local_hmin_);
    if (!CollectiveContinue(local_slot_valid)) return false;
    const bool trace_ready = DistributeCurrentTrace(committed.trace.coefficients);
    const bool launched = trace_ready && CudaLaunchMaximumWave(
        view_, shape_, committed.interior.coefficients,
        view_.current_local_trace, view_.scalar);
    if (!CollectiveContinue(launched)) return false;
    const bool copied = TransportPolicy::DeviceToHost(
        host_cfl_scalars_.data(), view_.scalar,
        host_cfl_scalars_.size() * sizeof(double),
        CudaTransferCategory::scalar_diagnostic,
        CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!CollectiveContinue(copied && host_cfl_scalars_[1] == 1.0 &&
                            std::isfinite(host_cfl_scalars_[0]) &&
                            host_cfl_scalars_[0] >= 0.0)) return false;
    double global_hmin = local_hmin_;
    maximum_wave_speed = host_cfl_scalars_[0];
    const bool hmin_reduced = MPI_Allreduce(&local_hmin_, &global_hmin, 1,
        MPI_DOUBLE, MPI_MIN, communicator_) == MPI_SUCCESS;
    const bool wave_reduced = MPI_Allreduce(&host_cfl_scalars_[0],
        &maximum_wave_speed, 1, MPI_DOUBLE, MPI_MAX, communicator_) == MPI_SUCCESS;
    const double denominator = 2.0 * static_cast<double>(order) + 1.0;
    backend_stable_dt = maximum_wave_speed > 0.0
        ? global_hmin / (denominator * maximum_wave_speed)
        : global_hmin / denominator;
    return CollectiveContinue(hmin_reduced && wave_reduced &&
        std::isfinite(global_hmin) && global_hmin > 0.0 &&
        std::isfinite(maximum_wave_speed) && maximum_wave_speed >= 0.0 &&
        std::isfinite(backend_stable_dt) && backend_stable_dt > 0.0);
  }

  bool CollectiveContinue(const bool local) const noexcept
  {
    // ConfigureDeviceAndRanks publishes ranks_ before the first eligible
    // local fast path.  A zero/unknown count therefore retains the collective
    // pre-initialization failure behavior, while rank two is unchanged.
    if (ranks_ == 1) return local;
    int local_value = local ? 1 : 0;
    int global_value = 0;
    return MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                         communicator_) == MPI_SUCCESS && global_value == 1;
  }

  bool CollectiveCount(const bool local, int& count) const noexcept
  {
    const int local_value = local ? 1 : 0;
    if (ranks_ == 1)
    {
      count = local_value;
      return true;
    }
    count = 0;
    return MPI_Allreduce(&local_value, &count, 1, MPI_INT, MPI_SUM,
                         communicator_) == MPI_SUCCESS;
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
    value.preconditioner_token = ilu_
        ? reinterpret_cast<std::uintptr_t>(
              static_cast<HYPRE_Solver>(*ilu_))
        : (diagonal_
              ? reinterpret_cast<std::uintptr_t>(&*diagonal_)
              : (rank_one_mgr_state_.configured
                    ? reinterpret_cast<std::uintptr_t>(
                          rank_one_mgr_state_.mgr)
                    : std::uintptr_t{0}));
    value.workspace_token = reinterpret_cast<std::uintptr_t>(view_.a);
    value.interior_slot_tokens = {{committed_slot_.interior.storage_token,
                                   candidate_slot_.interior.storage_token}};
    value.trace_slot_tokens = {{committed_slot_.trace.storage_token,
                                candidate_slot_.trace.storage_token}};
    value.baseline_allocation_events = baseline_allocations_;
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-evidence-sampling", "before", false);
    const std::uint64_t current_allocation_events =
        AllocatorPolicy::AllocationEvents();
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-evidence-sampling", "after", true);
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-allocation-delta", "before", false);
    value.post_baseline_allocation_events =
        current_allocation_events - baseline_allocations_;
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-allocation-delta", "after", true);
    value.baseline_w8_owned_allocation_calls =
        baseline_w8_owned_allocation_calls_;
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-owned-call-sampling", "before", false);
    value.post_baseline_w8_owned_allocation_calls =
        AllocatorPolicy::W8OwnedAllocationCalls() -
        baseline_w8_owned_allocation_calls_;
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-owned-call-sampling", "after", true);
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-persistence-predicates", "before", false);
    const CudaTracePreconditionerKind selected_preconditioner =
        trace_policy_descriptor_.trace_preconditioner;
    const bool selected_preconditioner_ready = trace_solver_kind_ ==
            CudaTraceSolverKind::native_rank1_fp64
        ? !gmres_ && !diagonal_ &&
            (CudaTracePreconditionerIsIlu(selected_preconditioner) ==
                 static_cast<bool>(ilu_)) &&
            trace_preconditioner_ == selected_preconditioner &&
            native_trace_requirements_.valid &&
            native_trace_support_.supported &&
            native_trace_workspace_.device_control != nullptr &&
            (!CudaTracePreconditionerIsNativeElementSchwarz(
                 selected_preconditioner) ||
             (element_schwarz_plan_.valid &&
              view_.element_schwarz_patch_matrix != nullptr &&
              view_.element_schwarz_patch_lu != nullptr &&
              view_.element_schwarz_rhs != nullptr &&
              view_.element_schwarz_solution != nullptr &&
              view_.element_schwarz_first_source != nullptr &&
              view_.element_schwarz_second_source != nullptr &&
              view_.element_schwarz_signed_true_dof != nullptr &&
              view_.element_schwarz_true_incidence_slots != nullptr &&
              view_.element_schwarz_pivots != nullptr))
        : trace_solver_kind_ ==
                  CudaTraceSolverKind::hypre_fgmres_mgr_rank1
            ? !gmres_ && !ilu_ && !diagonal_ &&
                selected_preconditioner ==
                    CudaTracePreconditionerKind::hypre_mgr_field_split &&
                trace_preconditioner_ == selected_preconditioner &&
                rank_one_mgr_state_.configured &&
                rank_one_mgr_state_.fgmres != nullptr &&
                rank_one_mgr_state_.mgr != nullptr &&
                rank_one_mgr_state_.f_ilu != nullptr &&
                rank_one_mgr_state_.coarse_amg != nullptr &&
                equilibrated_true_matrix_ &&
                hypre_equilibrated_rhs_ &&
                std::isfinite(
                    rank_one_mgr_state_.configured_relative_tolerance) &&
                equilibration_left_device_ != nullptr &&
                equilibration_right_device_ != nullptr
            : gmres_ &&
            ((CudaTracePreconditionerIsIlu(selected_preconditioner) &&
              ilu_ && !diagonal_ && trace_preconditioner_ ==
                  selected_preconditioner) ||
             (selected_preconditioner ==
                  CudaTracePreconditionerKind::hypre_diag_scale &&
              diagonal_ && !ilu_ && trace_preconditioner_ ==
                  selected_preconditioner));
    value.persistent_graph = true_matrix_ && local_trace_matrix_ &&
        selected_preconditioner_ready;
    value.persistent_slots = committed_slot_.interior.coefficients !=
            candidate_slot_.interior.coefficients &&
        committed_slot_.trace.coefficients != candidate_slot_.trace.coefficients;
    value.persistent_workspace = view_.a != nullptr && view_.schur != nullptr &&
        (trace_solver_kind_ != CudaTraceSolverKind::native_rank1_fp64 ||
         (native_trace_workspace_.basis != nullptr &&
          native_trace_workspace_.vectors != nullptr &&
          native_trace_workspace_.reduction != nullptr &&
          native_trace_workspace_.small != nullptr &&
          (!CudaTracePreconditionerIsNativeElementSchwarz(
               selected_preconditioner) ||
           (element_schwarz_plan_.valid &&
            view_.element_schwarz_patch_matrix != nullptr &&
            view_.element_schwarz_patch_lu != nullptr &&
            view_.element_schwarz_rhs != nullptr &&
            view_.element_schwarz_solution != nullptr &&
            view_.element_schwarz_first_source != nullptr &&
            view_.element_schwarz_second_source != nullptr &&
            view_.element_schwarz_signed_true_dof != nullptr &&
            view_.element_schwarz_true_incidence_slots != nullptr &&
            view_.element_schwarz_pivots != nullptr)))) &&
        (trace_solver_kind_ !=
                 CudaTraceSolverKind::hypre_fgmres_mgr_rank1 ||
         (equilibration_row_norm_device_ != nullptr &&
          equilibration_column_norm_device_ != nullptr &&
          equilibration_control_device_ != nullptr &&
          equilibrated_rhs_device_ != nullptr));
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-persistence-predicates", "after", true);
    ReportCudaWarmLifetimeMarker(
        rank_, "reuse-evidence-return", "before", true);
    return value;
  }

  bool PrepareAttempt(const AttemptIdentity& attempt, state_slot& workspace,
                      const bool first_step) noexcept
  {
    current_attempt_ = attempt;
    current_generation_ = attempt.committed_generation + 1;
    ticket_count_ = 0;
    TransportPolicy::SetPhase(CudaFailurePhase::setup, current_generation_);
    const CudaIluFactorReusePolicy factor_reuse_policy =
        trace_policy_descriptor_.ilu_factor_reuse_policy;
    const bool factor_attempt_ready = true_matrix_ &&
        CudaBeginIluFactorAttempt(
            ilu_factor_state_, attempt, factor_reuse_policy,
            reinterpret_cast<std::uintptr_t>(true_matrix_.get()),
            reinterpret_cast<std::uintptr_t>(true_device_.diag_values),
            reinterpret_cast<std::uintptr_t>(true_device_.offd_values),
            true_device_.diag_nonzeros, true_device_.offd_nonzeros);
    element_schwarz_attempt_state_ = {};
    const bool use_element_schwarz =
        CudaTracePreconditionerIsNativeElementSchwarz(
            trace_policy_descriptor_.trace_preconditioner);
    bool schwarz_attempt_ready = !use_element_schwarz;
    if (use_element_schwarz &&
        trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64 &&
        ranks_ == 1 && element_schwarz_plan_.valid)
    {
      element_schwarz_attempt_state_.attempt = attempt;
      element_schwarz_attempt_state_.active = true;
      schwarz_attempt_ready = true;
    }
    // Borrowed mode must seed its face LLF speeds from one coherent committed
    // state.  On the initial attempt materialize both committed slots before
    // copying old_interior; the frozen legacy route below remains unchanged.
    bool local_ready = factor_attempt_ready && schwarz_attempt_ready;
    if (first_step && shape_.borrowed_host_inputs)
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
      const bool old_ready = TransportPolicy::DeviceToDevice(
              view_.old_interior, workspace.interior.coefficients,
              workspace.interior.count * sizeof(double),
              CudaTransferCategory::attempt_history,
              CudaTransferSite::attempt_seed_interior);
      local_ready = local_ready && interior_ready && trace_ready && old_ready;
      workspace.interior.generation = current_generation_;
      workspace.trace.generation = current_generation_;
    }
    else
    {
      local_ready = local_ready && TransportPolicy::DeviceToDevice(
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
    }
    if (!CollectiveContinue(local_ready)) { return false; }
    return DistributeCurrentTrace(workspace.trace.coefficients);
  }

private:
  /**
   * Central-difference audit of the live device assembler.  This private
   * implementation stages no output and publishes no slot; the qualification
   * TU reaches it only through the private friend accessor.
   */
  bool AuditBorrowedEulerAssemblerJacobian(
      const double relative_tolerance,
      std::array<double, 6>& relative_errors) noexcept
  {
    relative_errors.fill(std::numeric_limits<double>::infinity());
    const bool component_layout = shape_.igr_active
        ? shape_.interior_components == igr_interior_components &&
            shape_.trace_components == igr_trace_components
        : shape_.interior_components == conserved_components &&
            shape_.trace_components == conserved_components;
    const bool admitted = initialized_ && shape_.borrowed_host_inputs &&
        component_layout &&
        std::isfinite(relative_tolerance) && relative_tolerance > 0.0;
    if (!CollectiveContinue(admitted)) return false;

    std::vector<double> base_interior, base_true_trace;
    std::vector<double> interior_direction, true_trace_direction;
    std::vector<double> element_trace_direction;
    std::vector<double> matrix_a, matrix_b, matrix_c, matrix_d;
    std::vector<double> a_direction, b_direction, c_direction, d_direction;
    bool local_valid = true;
    try
    {
      base_interior.resize(static_cast<std::size_t>(shape_.interior_count));
      base_true_trace.resize(static_cast<std::size_t>(shape_.true_trace_count));
      interior_direction.resize(base_interior.size());
      true_trace_direction.resize(base_true_trace.size());
      element_trace_direction.resize(static_cast<std::size_t>(shape_.elements) *
                                     shape_.trace_dofs);
      matrix_a.resize(static_cast<std::size_t>(shape_.elements) *
                      shape_.interior_dofs * shape_.interior_dofs);
      matrix_b.resize(static_cast<std::size_t>(shape_.elements) *
                      shape_.interior_dofs * shape_.trace_dofs);
      matrix_c.resize(static_cast<std::size_t>(shape_.elements) *
                      shape_.trace_dofs * shape_.interior_dofs);
      matrix_d.resize(static_cast<std::size_t>(shape_.elements) *
                      shape_.trace_dofs * shape_.trace_dofs);
      a_direction.assign(base_interior.size(), 0.0);
      b_direction.assign(base_interior.size(), 0.0);
      c_direction.assign(element_trace_direction.size(), 0.0);
      d_direction.assign(element_trace_direction.size(), 0.0);
    }
    catch (...) { local_valid = false; }
    if (!CollectiveContinue(local_valid)) return false;

    const auto d2h = [](double* destination, const double* source,
                        const std::size_t count) noexcept
    {
      return TransportPolicy::DeviceToHost(
          destination, source, count * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export);
    };
    const auto h2d = [](double* destination, const double* source,
                        const std::size_t count) noexcept
    {
      return TransportPolicy::HostToDevice(
          destination, source, count * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export);
    };
    local_valid = d2h(base_interior.data(),
                      committed_slot_.interior.coefficients,
                      base_interior.size()) &&
        d2h(base_true_trace.data(), committed_slot_.trace.coefficients,
            base_true_trace.size()) &&
        TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
    if (!CollectiveContinue(local_valid)) return false;

    const auto deterministic_unit = [](const std::size_t index,
                                       const std::uint64_t salt) noexcept
    {
      std::uint64_t value =
          (static_cast<std::uint64_t>(index) + 1U) *
              11400714819323198485ULL +
          salt * 7046029254386353131ULL;
      value ^= value >> 29U;
      const double magnitude = 0.25 + 0.75 *
          static_cast<double>((value >> 8U) % 29U) / 28.0;
      return (value & 1U) == 0U ? magnitude : -magnitude;
    };
    if (shape_.igr_active)
      for (int element = 0; element < shape_.elements; ++element)
        for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
        {
          const std::size_t base = static_cast<std::size_t>(element) *
              shape_.interior_dofs;
          const std::size_t ordinal = static_cast<std::size_t>(element) *
              shape_.scalar_interior_dofs + scalar;
          base_interior[base + scalar +
              4 * shape_.scalar_interior_dofs] =
              0.035 * deterministic_unit(ordinal, 11U);
          base_interior[base + scalar +
              5 * shape_.scalar_interior_dofs] =
              0.025 * deterministic_unit(ordinal, 17U);
          base_interior[base + scalar +
              6 * shape_.scalar_interior_dofs] =
              0.025 * deterministic_unit(ordinal, 23U);
        }
    const int true_trace_scalar = shape_.true_trace_count /
        shape_.trace_components;
    local_valid = true_trace_scalar * shape_.trace_components ==
        shape_.true_trace_count;
    if (shape_.igr_active)
      for (int scalar = 0;
           local_valid && scalar < true_trace_scalar; ++scalar)
        base_true_trace[
            static_cast<std::size_t>(4 * true_trace_scalar + scalar)] =
            0.02 * deterministic_unit(
                static_cast<std::size_t>(scalar), 31U);
    for (std::size_t index = 0; index < base_interior.size(); ++index)
      interior_direction[index] = deterministic_unit(index, 41U) *
          std::max(1.0, std::abs(base_interior[index]));
    for (std::size_t index = 0; index < base_true_trace.size(); ++index)
      true_trace_direction[index] = deterministic_unit(index, 47U) *
          std::max(1.0, std::abs(base_true_trace[index]));
    if (!CollectiveContinue(local_valid)) return false;

    const auto install_state = [&](const std::vector<double>& interior,
                                   const std::vector<double>& trace) noexcept
    {
      const bool uploaded = h2d(view_.work_interior, interior.data(),
                                interior.size()) &&
          h2d(view_.work_true_trace, trace.data(), trace.size()) &&
          TransportPolicy::Synchronize(
              CudaTransferSite::generic_host_export);
      if (!CollectiveContinue(uploaded)) return false;
      return DistributeCurrentTrace(view_.work_true_trace);
    };
    const auto assemble = [&](const std::vector<double>& interior,
                              const std::vector<double>& trace,
                              std::vector<double>* residual_interior,
                              std::vector<double>* residual_trace,
                              const bool frozen_llf = false) noexcept
    {
      if (!install_state(interior, trace)) return false;
      TransportPolicy::SetPhase(CudaFailurePhase::assembly,
                                current_generation_);
      const bool assembled = (frozen_llf
              ? CudaLaunchFrozenBorrowedLlfAssembly(
                    view_, shape_, fixed_dt)
              : CudaLaunchAssembly(
                    view_, shape_, fixed_dt, cuda_engine_)) &&
          StatusPassed();
      if (!CollectiveContinue(assembled)) return false;
      if (residual_interior == nullptr || residual_trace == nullptr)
        return true;
      bool copied = true;
      try
      {
        residual_interior->resize(
            static_cast<std::size_t>(shape_.interior_count));
        residual_trace->resize(static_cast<std::size_t>(shape_.elements) *
                               shape_.trace_dofs);
      }
      catch (...) { copied = false; }
      if (!CollectiveContinue(copied)) return false;
      copied = d2h(residual_interior->data(), view_.residual_r,
                   residual_interior->size()) &&
          d2h(residual_trace->data(), view_.residual_g,
              residual_trace->size()) &&
          TransportPolicy::Synchronize(
              CudaTransferSite::generic_host_export);
      return CollectiveContinue(copied);
    };

    local_valid = h2d(view_.old_interior, base_interior.data(),
                      base_interior.size()) &&
        TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
    if (!CollectiveContinue(local_valid) ||
        !install_state(base_interior, base_true_trace))
      return false;
    // The production base assembly evaluates tau at this candidate.  Freeze
    // that same candidate value only for the finite-difference perturbations
    // below: A/B/C/D intentionally omit d-tau while every ordinary residual
    // or line-search assembly refreshes tau from its own current candidate.
    if (!shape_.igr_active)
    {
      const bool tau_frozen = CudaLaunchFreezeBorrowedCurrentLlfTau(
              view_, shape_) && StatusPassed();
      if (!CollectiveContinue(tau_frozen)) return false;
    }
    if (!assemble(base_interior, base_true_trace, nullptr, nullptr))
      return false;
    local_valid = d2h(matrix_a.data(), view_.a, matrix_a.size()) &&
        d2h(matrix_b.data(), view_.b, matrix_b.size()) &&
        d2h(matrix_c.data(), view_.c, matrix_c.size()) &&
        d2h(matrix_d.data(), view_.d, matrix_d.size()) &&
        TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
    if (!CollectiveContinue(local_valid)) return false;

    // Use the production owner/ghost distribution to obtain the exact signed
    // element-local trace direction consumed by B and D.
    local_valid = h2d(view_.work_true_trace, true_trace_direction.data(),
                      true_trace_direction.size()) &&
        TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
    if (!CollectiveContinue(local_valid) ||
        !DistributeCurrentTrace(view_.work_true_trace))
      return false;
    local_valid = d2h(element_trace_direction.data(),
                      view_.current_local_trace,
                      element_trace_direction.size()) &&
        TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
    if (!CollectiveContinue(local_valid)) return false;

    for (int element = 0; element < shape_.elements; ++element)
    {
      const std::size_t interior_offset =
          static_cast<std::size_t>(element) * shape_.interior_dofs;
      const std::size_t trace_offset =
          static_cast<std::size_t>(element) * shape_.trace_dofs;
      const std::size_t a_offset = interior_offset * shape_.interior_dofs;
      const std::size_t b_offset = static_cast<std::size_t>(element) *
          shape_.interior_dofs * shape_.trace_dofs;
      const std::size_t c_offset = static_cast<std::size_t>(element) *
          shape_.trace_dofs * shape_.interior_dofs;
      const std::size_t d_offset = trace_offset * shape_.trace_dofs;
      for (int row = 0; row < shape_.interior_dofs; ++row)
      {
        for (int column = 0; column < shape_.interior_dofs; ++column)
          a_direction[interior_offset + row] += matrix_a[
              a_offset + static_cast<std::size_t>(row) *
                  shape_.interior_dofs + column] *
              interior_direction[interior_offset + column];
        for (int column = 0; column < shape_.trace_dofs; ++column)
          b_direction[interior_offset + row] += matrix_b[
              b_offset + static_cast<std::size_t>(row) *
                  shape_.trace_dofs + column] *
              element_trace_direction[trace_offset + column];
      }
      for (int row = 0; row < shape_.trace_dofs; ++row)
      {
        for (int column = 0; column < shape_.interior_dofs; ++column)
          c_direction[trace_offset + row] += matrix_c[
              c_offset + static_cast<std::size_t>(row) *
                  shape_.interior_dofs + column] *
              interior_direction[interior_offset + column];
        for (int column = 0; column < shape_.trace_dofs; ++column)
          d_direction[trace_offset + row] += matrix_d[
              d_offset + static_cast<std::size_t>(row) *
                  shape_.trace_dofs + column] *
              element_trace_direction[trace_offset + column];
      }
    }

    const double difference_step =
        std::cbrt(std::numeric_limits<double>::epsilon());
    const auto central_difference = [&](const bool perturb_interior,
                                        const bool perturb_trace,
                                        std::vector<double>& difference_r,
                                        std::vector<double>& difference_g)
        noexcept
    {
      std::vector<double> plus_interior, minus_interior;
      std::vector<double> plus_trace, minus_trace;
      std::vector<double> plus_r, plus_g, minus_r, minus_g;
      bool ready = true;
      try
      {
        plus_interior = base_interior;
        minus_interior = base_interior;
        plus_trace = base_true_trace;
        minus_trace = base_true_trace;
        if (perturb_interior)
          for (std::size_t index = 0; index < base_interior.size(); ++index)
          {
            plus_interior[index] += difference_step *
                interior_direction[index];
            minus_interior[index] -= difference_step *
                interior_direction[index];
          }
        if (perturb_trace)
          for (std::size_t index = 0; index < base_true_trace.size(); ++index)
          {
            plus_trace[index] += difference_step * true_trace_direction[index];
            minus_trace[index] -= difference_step * true_trace_direction[index];
          }
      }
      catch (...) { ready = false; }
      if (!CollectiveContinue(ready) ||
          !assemble(plus_interior, plus_trace, &plus_r, &plus_g,
                    !shape_.igr_active) ||
          !assemble(minus_interior, minus_trace, &minus_r, &minus_g,
                    !shape_.igr_active))
        return false;
      try
      {
        difference_r.resize(plus_r.size());
        difference_g.resize(plus_g.size());
        const double inverse_width = 0.5 / difference_step;
        for (std::size_t index = 0; index < difference_r.size(); ++index)
          difference_r[index] = inverse_width *
              (plus_r[index] - minus_r[index]);
        for (std::size_t index = 0; index < difference_g.size(); ++index)
          difference_g[index] = inverse_width *
              (plus_g[index] - minus_g[index]);
      }
      catch (...) { return CollectiveContinue(false); }
      return CollectiveContinue(true);
    };

    std::vector<double> difference_a, difference_c;
    std::vector<double> difference_b, difference_d;
    std::vector<double> difference_combined_r, difference_combined_g;
    if (!central_difference(true, false, difference_a, difference_c) ||
        !central_difference(false, true, difference_b, difference_d) ||
        !central_difference(true, true, difference_combined_r,
                            difference_combined_g))
      return false;
    std::vector<double> combined_r(a_direction.size(), 0.0);
    std::vector<double> combined_g(c_direction.size(), 0.0);
    for (std::size_t index = 0; index < combined_r.size(); ++index)
      combined_r[index] = a_direction[index] + b_direction[index];
    for (std::size_t index = 0; index < combined_g.size(); ++index)
      combined_g[index] = c_direction[index] + d_direction[index];

    const auto relative_error = [&](const std::vector<double>& difference,
                                    const std::vector<double>& linearized,
                                    double& error) noexcept
    {
      if (!CollectiveContinue(difference.size() == linearized.size()))
        return false;
      std::array<double, 4> local{};
      for (std::size_t index = 0; index < difference.size(); ++index)
      {
        const double delta = difference[index] - linearized[index];
        local[0] += delta * delta;
        local[1] += difference[index] * difference[index];
        local[2] += linearized[index] * linearized[index];
      }
      local[3] = static_cast<double>(difference.size());
      std::array<double, 4> global{};
      const bool reduced = MPI_Allreduce(
          local.data(), global.data(), static_cast<int>(global.size()),
          MPI_DOUBLE, MPI_SUM, communicator_) == MPI_SUCCESS;
      if (!CollectiveContinue(reduced)) return false;
      const double denominator = std::max({
          std::sqrt(global[1]), std::sqrt(global[2]),
          difference_step * std::sqrt(std::max(1.0, global[3]))});
      error = std::sqrt(global[0]) / denominator;
      const double activity_floor = 1.0e-12 *
          std::sqrt(std::max(1.0, global[3]));
      return CollectiveContinue(std::isfinite(error) &&
          std::sqrt(global[1]) > activity_floor &&
          std::sqrt(global[2]) > activity_floor);
    };
    local_valid = relative_error(difference_a, a_direction,
                                 relative_errors[0]) &&
        relative_error(difference_b, b_direction, relative_errors[1]) &&
        relative_error(difference_c, c_direction, relative_errors[2]) &&
        relative_error(difference_d, d_direction, relative_errors[3]) &&
        relative_error(difference_combined_r, combined_r,
                       relative_errors[4]) &&
        relative_error(difference_combined_g, combined_g,
                       relative_errors[5]);
    if (!local_valid) return false;

    const bool restored = assemble(
        base_interior, base_true_trace, nullptr, nullptr);
    const bool within_tolerance = std::all_of(
        relative_errors.begin(), relative_errors.end(),
        [relative_tolerance](const double error)
        { return std::isfinite(error) && error <= relative_tolerance; });
    return restored && CollectiveContinue(within_tolerance);
  }

public:
  bool EulerConservationBalance(const double dt, double& maximum_error,
                                double& maximum_relative_error) noexcept
  {
    maximum_error = 0.0;
    maximum_relative_error = 0.0;
    constexpr int components = igr_trace_components;
    static_assert(euler_conservation_result_scalars == 2 * components + 2);
    std::array<double, euler_conservation_result_scalars> local{};
    const bool launched = initialized_ && shape_.borrowed_host_inputs &&
        std::isfinite(dt) && dt > 0.0 &&
        CudaLaunchEulerConservation(view_, shape_, dt, view_.scalar);
    if (!CollectiveContinue(launched)) return false;
    const bool copied = TransportPolicy::DeviceToHost(
            local.data(), view_.scalar, local.size() * sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    bool local_valid = copied && local[2 * components + 1] == 1.0 &&
        std::isfinite(local[2 * components]) &&
        local[2 * components] >= 0.0;
    for (int index = 0; local_valid && index <= 2 * components; ++index)
      local_valid = std::isfinite(local[static_cast<std::size_t>(index)]);
    if (!CollectiveContinue(local_valid)) return false;
    std::array<double, components> global_balance{};
    std::array<double, components> global_scale{};
    double global_domain_measure = 0.0;
    const bool balance_reduced = MPI_Allreduce(
        local.data(), global_balance.data(), components, MPI_DOUBLE,
        MPI_SUM, communicator_) == MPI_SUCCESS;
    const bool scale_reduced = MPI_Allreduce(
        local.data() + components, global_scale.data(), components,
        MPI_DOUBLE, MPI_SUM, communicator_) == MPI_SUCCESS;
    const bool measure_reduced = MPI_Allreduce(
        local.data() + 2 * components, &global_domain_measure, 1,
        MPI_DOUBLE, MPI_SUM, communicator_) == MPI_SUCCESS;
    if (!CollectiveContinue(balance_reduced && scale_reduced &&
                            measure_reduced &&
                            std::isfinite(global_domain_measure) &&
                            global_domain_measure > 0.0))
      return false;
    const int active_components = shape_.igr_active
        ? components : conserved_components;
    for (int component = 0; component < active_components; ++component)
    {
      const double error = std::abs(
          global_balance[static_cast<std::size_t>(component)]);
      const double relative =
          dg::hdg::generic::EulerHdgConservationRelativeError(
              error, global_scale[static_cast<std::size_t>(component)],
              global_domain_measure);
      maximum_error = std::max(maximum_error, error);
      maximum_relative_error = std::max(maximum_relative_error, relative);
    }
    return CollectiveContinue(std::isfinite(maximum_error) &&
                              std::isfinite(maximum_relative_error));
  }

  CudaNumericalResult Execute(const CudaConfig& config,
                              const AttemptIdentity& attempt,
                              state_slot& workspace) noexcept
  {
    ::dg_igr::NVTXRange accepted_step_profile_region(
        ::dg_igr::observability::RegionId::hdg_accepted_step);
    CudaIluFactorAttemptEndGuard end_factor_attempt(ilu_factor_state_);
    CudaElementSchwarzAttemptEndGuard end_schwarz_attempt(
        element_schwarz_attempt_state_);
    CudaNumericalResult result{};
    result.tensor_core.selected_engine = cuda_engine_;
    result.tensor_core.trace_policy = trace_policy_descriptor_.selection;
    result.tensor_core.trace_preconditioner = trace_preconditioner_;
    result.tensor_core.trace_preconditioner_configured =
        CudaTracePolicyDescriptorIsExact(trace_policy_descriptor_) &&
        trace_policy_descriptor_.igr_active == shape_.igr_active &&
        trace_preconditioner_ == trace_policy_descriptor_.trace_preconditioner;
    result.tensor_core.native_rank1_trace_solver_configured =
        trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64;
    const bool use_element_schwarz =
        CudaTracePreconditionerIsNativeElementSchwarz(
            trace_preconditioner_);
    result.tensor_core.native_element_schwarz_configured =
        use_element_schwarz && element_schwarz_plan_.valid;
    result.tensor_core.native_matrix_free_trace_operator_configured =
        CudaTracePolicyUsesNativeElementMatrixFreeOperator(
            trace_policy_descriptor_.selection) &&
        result.tensor_core.native_element_schwarz_configured;
    result.tensor_core.element_schwarz_patch_rows = use_element_schwarz
        ? element_schwarz_plan_.patch_rows : 0;
    result.tensor_core.element_schwarz_minimum_multiplicity =
        use_element_schwarz
        ? element_schwarz_plan_.minimum_multiplicity : 0;
    result.tensor_core.element_schwarz_maximum_multiplicity =
        use_element_schwarz
        ? element_schwarz_plan_.maximum_multiplicity : 0;
    result.tensor_core.hypre_mgr_rank1_trace_solver_configured =
        trace_solver_kind_ ==
            CudaTraceSolverKind::hypre_fgmres_mgr_rank1;
    result.tensor_core.
        hypre_host_staged_rank2_trace_solver_configured =
            trace_solver_kind_ ==
                CudaTraceSolverKind::hypre_host_staged_rank2;
    result.tensor_core.trace_solver_accounting_complete =
        IsKnownCudaTraceSolverKind(trace_solver_kind_);
    // Aggregate-accounting failure is sticky in the evidence but does not
    // redefine the independently verified live preconditioner configuration.
    (void)CudaInitializeIluEvidence(
        result.tensor_core, trace_preconditioner_, shape_.igr_active);
    result.tensor_core.hardware_supported =
        tensor_core_hardware_supported_;
    result.tensor_core.fallback_accounting_complete = true;
    // This production leaf has no mixed-precision or correction route.  Emit
    // that fact from the numerical attempt itself so benchmark accounting is
    // not synthesized by the harness.
    result.precision_work.all_fp64_policy = true;
    result.precision_work.accounting_complete = true;
    // The frozen non-borrowed lifecycle has no Euler conservation contract.
    // Borrowed Euler attempts remain fail-closed until their measured balance
    // has passed below.
    result.conservation = !shape_.borrowed_host_inputs;
    // The application may perform unrelated MFEM output work between solver
    // attempts, and MFEM/HYPRE may grow their opaque shared Umpire pool while
    // doing so.  Admit no growth caused by a warm solver attempt.  The first
    // completed attempt alone may materialize one backend pool block; retain
    // the lifetime-wide counter separately so this remains functionality
    // evidence and never becomes a performance claim.
    accepted_allocation_events_ = AllocatorPolicy::AllocationEvents();
    const bool numerical_injection_phase =
        config.injection.phase == CudaFailurePhase::assembly ||
        config.injection.phase == CudaFailurePhase::condensation ||
        config.injection.phase == CudaFailurePhase::trace_solve ||
        config.injection.phase == CudaFailurePhase::recovery ||
        config.injection.phase ==
            CudaFailurePhase::after_recovery_before_commit ||
        config.injection.phase == CudaFailurePhase::diagnostics;
    const bool injection_config_valid = !config.injection.enabled ||
        (numerical_injection_phase && config.injection.target_step != 0 &&
         config.injection.source_rank >= 0 &&
         config.injection.source_rank < ranks_);
    if (!CollectiveContinue(injection_config_valid))
    {
      result.phase = CudaFailurePhase::setup;
      result.reason = FailureReason::collective_protocol;
      return result;
    }
    const bool injection_active = config.injection.enabled &&
        config.injection.target_step == attempt.committed_step + 1 &&
        attempt.retry_index == 0;
    bool injection_protocol_failure = false;
    const auto inject_at = [&](const CudaFailurePhase phase) noexcept
    {
      result.phase = phase;
      if (!injection_active || config.injection.phase != phase) return false;
      const int local_claim = rank_ == config.injection.source_rank ? 1 : 0;
      int claims = 0;
      if (MPI_Allreduce(&local_claim, &claims, 1, MPI_INT, MPI_SUM,
                        communicator_) != MPI_SUCCESS || claims != 1)
      {
        injection_protocol_failure = true;
        result.reason = FailureReason::collective_protocol;
        result.injection_source_rank = -1;
        return true;
      }
      result.injected = true;
      result.reason = FailureReason::backend_failure;
      result.injection_source_rank = config.injection.source_rank;
      return true;
    };
    result.phase = CudaFailurePhase::assembly;
    if (inject_at(CudaFailurePhase::assembly)) { return result; }
    const bool assembly_ticket = initialized_ &&
        Ticket(CudaFailurePhase::assembly);
    if (!CollectiveContinue(assembly_ticket) ||
        !AssembleAndNorm(workspace, attempt.dt, result.initial_residual))
    {
      result.reason = FailureReason::finite_gate;
      return result;
    }
    result.finite = std::isfinite(result.initial_residual) &&
        (config.use_generic_numerical_policy ||
         result.initial_residual > 1.0e-8);
    result.final_residual = result.initial_residual;
    const double nonlinear_relative_tolerance =
        config.use_generic_numerical_policy
            ? config.numerical_policy.nonlinear_relative_tolerance : 1.0e-10;
    const double armijo_constant = config.use_generic_numerical_policy
        ? config.numerical_policy.armijo_constant : 1.0e-4;
    const int maximum_line_search_trials =
        config.use_generic_numerical_policy
            ? config.numerical_policy.maximum_line_search_trials : 13;
    const double target = nonlinear_relative_tolerance *
        (1.0 + result.initial_residual);
    // Physical boundary rows use the same value-only exterior-state Picard
    // linearization as the CPU path.  Match that bounded envelope without
    // changing the periodic generic or legacy W8A iteration limit.
    const int maximum_nonlinear_updates = config.use_generic_numerical_policy
        ? config.numerical_policy.maximum_nonlinear_updates
        : (shape_.borrowed_static_boundary_policies ? 80 : 20);
    const auto record_linear_work = [&] (
        const generic::EulerHdgLinearSolveRole role,
        const int iterations, const int solve_attempts,
        const int true_residual_tightenings) noexcept
    {
      int& role_iterations =
          role == generic::EulerHdgLinearSolveRole::newton_correction
          ? result.newton_correction_linear_iterations
          : result.final_coherent_linear_iterations;
      if (iterations < 0 || solve_attempts < 0 ||
          true_residual_tightenings < 0 ||
          iterations > std::numeric_limits<int>::max() -
              result.linear_iterations ||
          iterations > std::numeric_limits<int>::max() - role_iterations ||
          solve_attempts > std::numeric_limits<int>::max() -
              result.linear_solve_attempts ||
          true_residual_tightenings > std::numeric_limits<int>::max() -
              result.true_residual_tightenings)
        return false;
      result.linear_iterations += iterations;
      role_iterations += iterations;
      result.linear_solve_attempts += solve_attempts;
      result.true_residual_tightenings += true_residual_tightenings;
      return true;
    };
    const auto record_line_search_trial = [&]() noexcept
    {
      if (result.line_search_trials == std::numeric_limits<int>::max())
        return false;
      ++result.line_search_trials;
      return true;
    };
    const auto record_line_search_rejection = [&]() noexcept
    {
      if (result.rejected_line_search_trials ==
          std::numeric_limits<int>::max())
        return false;
      ++result.rejected_line_search_trials;
      return true;
    };
    const auto launch_condensation = [&]() noexcept
    {
      ::dg_igr::NVTXRange profile_region(
          ::dg_igr::observability::RegionId::hdg_condensation);
      if (cuda_engine_ == generic::CudaEngineKind::tensor_core &&
          result.tensor_core.condensation_launches ==
              std::numeric_limits<std::uint64_t>::max())
        return false;
      std::uint64_t mma_tiles = 0;
      if (!CudaLaunchCondensation(view_, shape_, cuda_engine_, mma_tiles))
        return false;
      if (cuda_engine_ == generic::CudaEngineKind::tensor_core)
      {
        if (mma_tiles > std::numeric_limits<std::uint64_t>::max() -
                result.tensor_core.mma_tiles)
          return false;
        ++result.tensor_core.condensation_launches;
        result.tensor_core.mma_tiles += mma_tiles;
      }
      return true;
    };
    std::uint64_t last_product_generation = 0;
    for (int iteration = 0; iteration < maximum_nonlinear_updates; ++iteration)
    {
      const bool minimum_update_witness = config.use_generic_numerical_policy ||
          result.nonlinear_updates >= 2;
      if (minimum_update_witness && result.final_residual <= target)
        break;
      if (inject_at(CudaFailurePhase::condensation)) { return result; }
      const bool condensed = CopyBase(workspace) &&
          Ticket(CudaFailurePhase::condensation) &&
          launch_condensation() && StatusPassed();
      if (!CollectiveContinue(condensed))
      {
        result.phase = CudaFailurePhase::condensation;
        result.reason = FailureReason::backend_failure;
        return result;
      }
      const std::uint64_t generation = ++linear_generation_;
      result.phase = CudaFailurePhase::trace_solve;
      if (inject_at(CudaFailurePhase::trace_solve)) { return result; }
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
      int solve_iterations = 0;
      int solve_attempts = 0;
      int true_residual_tightenings = 0;
      const bool solve_ok = SolveTrueSystem(
          generic::EulerHdgLinearSolveRole::newton_correction,
          result.true_relative_residual, generation, solve_iterations,
          solve_attempts, true_residual_tightenings, result.tensor_core);
      const bool solve_accounted = record_linear_work(
          generic::EulerHdgLinearSolveRole::newton_correction,
          solve_iterations, solve_attempts, true_residual_tightenings);
      if (!CollectiveContinue(solve_ok && solve_accounted))
      {
        result.reason = FailureReason::true_residual_gate;
        return result;
      }
      ++result.trace_solves;
      const bool trace_distributed = DistributeTraceIncrement();
      if (inject_at(CudaFailurePhase::recovery))
      {
        const bool distribution_complete = ranks_ != 1 ||
            TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
        (void)CollectiveContinue(
            trace_distributed && distribution_complete);
        return result;
      }
      bool recovery_launched = false;
      if (trace_distributed && Ticket(CudaFailurePhase::recovery))
      {
        ::dg_igr::NVTXRange profile_region(
            ::dg_igr::observability::RegionId::hdg_interior_recovery);
        recovery_launched = CudaLaunchRecover(view_, shape_, cuda_engine_);
      }
      const bool recovered = recovery_launched && StatusPassed();
      if (!CollectiveContinue(recovered))
      {
        result.phase = CudaFailurePhase::recovery;
        result.reason = FailureReason::recovery_gate;
        return result;
      }
      last_product_generation = generation;
      bool accepted = false;
      for (int trial = 0; trial < maximum_line_search_trials; ++trial)
      {
        if (!record_line_search_trial())
        {
          result.reason = FailureReason::backend_failure;
          return result;
        }
        const double alpha = std::ldexp(1.0, -trial);
        // TrialKernel writes the candidate directly into the persistent
        // workspace.  The base snapshot remains immutable, so a rejected
        // trial is still restored transactionally without the former pair of
        // full-state device-to-device copies.
        bool trial_ready = false;
        {
          ::dg_igr::NVTXRange profile_region(
              ::dg_igr::observability::RegionId::hdg_nonlinear_update);
          trial_ready = CudaLaunchTrial(view_, shape_, alpha);
        }
        if (!CollectiveContinue(trial_ready))
        {
          if (!CollectiveContinue(RestoreBase(workspace)))
          {
            result.reason = FailureReason::backend_failure;
            return result;
          }
          if (!record_line_search_rejection())
          {
            result.reason = FailureReason::backend_failure;
            return result;
          }
          continue;
        }
        const bool trace_ready =
            DistributeCurrentTrace(workspace.trace.coefficients);
        bool admissibility_launched = false;
        if (trace_ready)
        {
          ::dg_igr::NVTXRange profile_region(
              ::dg_igr::observability::RegionId::hdg_validation);
          admissibility_launched = CudaLaunchAdmissibility(view_, shape_);
        }
        const bool trial_admissible =
            admissibility_launched && StatusPassed();
        if (!CollectiveContinue(trial_admissible))
        {
          if (!CollectiveContinue(RestoreBase(workspace)))
          {
            result.reason = FailureReason::backend_failure;
            return result;
          }
          if (!record_line_search_rejection())
          {
            result.reason = FailureReason::backend_failure;
            return result;
          }
          continue;
        }
        double trial_residual = std::numeric_limits<double>::infinity();
        if (!AssembleAndNorm(workspace, attempt.dt, trial_residual) ||
            !std::isfinite(trial_residual))
        {
          result.reason = FailureReason::finite_gate;
          return result;
        }
        if (trial_residual <=
            (1.0 - armijo_constant * alpha) * result.final_residual)
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
        if (!record_line_search_rejection())
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

    if (inject_at(CudaFailurePhase::after_recovery_before_commit))
      return result;

    // Freeze one coherent accepted linearization: A/B/C/D, Schur/RHS,
    // solution, and recovery all describe the same candidate generation.
    const bool coherent_base =
        (config.use_generic_numerical_policy ||
         result.nonlinear_updates >= 2) &&
        result.final_residual <= target && CopyBase(workspace) &&
        Ticket(CudaFailurePhase::condensation) &&
        launch_condensation() && StatusPassed();
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
    int coherent_solve_iterations = 0;
    int coherent_solve_attempts = 0;
    int coherent_true_residual_tightenings = 0;
    const bool coherent_solve_ok = SolveTrueSystem(
        generic::EulerHdgLinearSolveRole::final_coherent,
        result.true_relative_residual, last_product_generation,
        coherent_solve_iterations, coherent_solve_attempts,
        coherent_true_residual_tightenings, result.tensor_core);
    const bool coherent_solve_accounted =
        record_linear_work(generic::EulerHdgLinearSolveRole::final_coherent,
                           coherent_solve_iterations,
                           coherent_solve_attempts,
                           coherent_true_residual_tightenings);
    if (!CollectiveContinue(
            coherent_solve_ok && coherent_solve_accounted))
    {
      result.reason = FailureReason::true_residual_gate;
      return result;
    }
    const bool coherent_trace_distributed = DistributeTraceIncrement();
    bool coherent_recovery_launched = false;
    if (coherent_trace_distributed && Ticket(CudaFailurePhase::recovery))
    {
      ::dg_igr::NVTXRange profile_region(
          ::dg_igr::observability::RegionId::hdg_interior_recovery);
      coherent_recovery_launched =
          CudaLaunchRecover(view_, shape_, cuda_engine_);
    }
    const bool coherent_solve =
        coherent_recovery_launched && StatusPassed();
    if (!CollectiveContinue(coherent_solve))
    {
      result.reason = FailureReason::recovery_gate;
      return result;
    }
    // A generic steady state may satisfy the nonlinear gate before its first
    // Newton update.  The coherent condensation/trace-solve/recovery sequence
    // above is still formed and validated; it is the recovery witness for
    // that zero-update candidate.  Legacy W8A retains its two-update witness.
    result.recovery = true;
    result.candidate_ready = true;
    // Rank-two increment distribution reuses distributed_local_trace as its
    // host-staged ghost buffer.  Restore the accepted current trace before
    // oracle capture so current_local_trace, distributed_local_trace, and
    // workspace.trace again describe the same candidate generation.
    const bool current_trace_restored = ranks_ != 2 ||
        DistributeCurrentTrace(workspace.trace.coefficients);
    if (!CollectiveContinue(current_trace_restored))
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
    bool admissibility_launched = false;
    {
      ::dg_igr::NVTXRange profile_region(
          ::dg_igr::observability::RegionId::hdg_validation);
      admissibility_launched = CudaLaunchAdmissibility(view_, shape_);
    }
    const bool local_admissible = admissibility_launched && StatusPassed();
    result.admissible = CollectiveContinue(local_admissible);
    result.true_residual = std::isfinite(result.true_relative_residual) &&
        result.true_relative_residual <= true_residual_relative_tolerance_;
    if (config.force_private_scientific_gate_failure)
      result.true_residual = false;
    if (inject_at(CudaFailurePhase::diagnostics)) { return result; }
    if (shape_.borrowed_host_inputs)
    {
      bool conservation_ready = false;
      {
        ::dg_igr::NVTXRange profile_region(
            ::dg_igr::observability::RegionId::hdg_validation);
        conservation_ready = EulerConservationBalance(
            attempt.dt, result.maximum_conservation_error,
            result.maximum_conservation_relative_error);
      }
      if (!CollectiveContinue(conservation_ready))
      {
        result.candidate_ready = false;
        result.conservation = false;
        result.reason = FailureReason::backend_failure;
        return result;
      }
      const bool local_conservation =
          result.maximum_conservation_relative_error <=
              config.numerical_policy.conservation_relative_tolerance;
      result.conservation = CollectiveContinue(local_conservation);
    }
    const std::uint64_t current_allocation_events =
        AllocatorPolicy::AllocationEvents();
    bool allocation_plateau_exact =
        current_allocation_events == accepted_allocation_events_;
    if (!allocation_plateau_sampled_)
    {
      // The rank-two MFEM/HYPRE path may materialize one opaque pool
      // high-water transition on the first fixed-shape solve.  Freeze that
      // one observed transition; rank one and every later attempt remain
      // strict at their exact accepted counts.
      allocation_plateau_exact =
          current_allocation_events == accepted_allocation_events_ ||
          (accepted_allocation_events_ !=
               std::numeric_limits<std::uint64_t>::max() &&
           current_allocation_events == accepted_allocation_events_ + 1);
      if (allocation_plateau_exact)
        accepted_allocation_events_ = current_allocation_events;
      allocation_plateau_sampled_ = true;
    }
    const std::uint64_t completed_linear_solve_attempts =
        static_cast<std::uint64_t>(result.linear_solve_attempts);
    const std::uint64_t completed_trace_solves =
        static_cast<std::uint64_t>(result.trace_solves);
    const bool linear_work_accounting_complete =
        result.linear_iterations ==
            result.newton_correction_linear_iterations +
                result.final_coherent_linear_iterations &&
        result.linear_solve_attempts ==
            result.trace_solves + result.true_residual_tightenings;
    const bool element_schwarz_selected =
        CudaTracePreconditionerIsNativeElementSchwarz(
            trace_preconditioner_);
    const bool equilibrated_field_split_selected =
        CudaTracePreconditionerIsNativeElementSchwarzEquilibratedFieldSplit(
            trace_preconditioner_);
    const bool element_schwarz_accounting_complete =
        element_schwarz_selected
        ? result.tensor_core.native_element_schwarz_configured &&
            result.tensor_core.element_schwarz_patch_rows ==
                NativeElementSchwarzIgrPatchRows(shape_.order) &&
            result.tensor_core.element_schwarz_minimum_multiplicity >= 1 &&
            result.tensor_core.element_schwarz_maximum_multiplicity >=
                result.tensor_core.element_schwarz_minimum_multiplicity &&
            result.tensor_core.element_schwarz_maximum_multiplicity <=
                native_element_schwarz_maximum_contributors &&
            result.tensor_core.element_schwarz_patch_setups ==
                completed_trace_solves &&
            ((result.linear_iterations == 0) ==
             (result.tensor_core.
                  element_schwarz_preconditioner_applications == 0))
        : !result.tensor_core.native_element_schwarz_configured &&
            result.tensor_core.element_schwarz_patch_setups == 0 &&
            result.tensor_core.
                    element_schwarz_preconditioner_applications == 0 &&
            result.tensor_core.element_schwarz_patch_rows == 0 &&
            result.tensor_core.element_schwarz_minimum_multiplicity == 0 &&
            result.tensor_core.element_schwarz_maximum_multiplicity == 0;
    const bool matrix_free_selected =
        CudaTracePolicyUsesNativeElementMatrixFreeOperator(
            trace_policy_descriptor_.selection);
    const bool matrix_free_accounting_complete = matrix_free_selected
        ? result.tensor_core.native_matrix_free_trace_operator_configured &&
            result.tensor_core.
                native_matrix_free_trace_operator_applications >=
                static_cast<std::uint64_t>(result.linear_iterations) &&
            ((result.linear_iterations == 0) ==
             (result.tensor_core.
                  native_matrix_free_trace_operator_applications == 0))
        : !result.tensor_core.
              native_matrix_free_trace_operator_configured &&
            result.tensor_core.
                native_matrix_free_trace_operator_applications == 0;
    result.tensor_core.trace_solver_accounting_complete =
        result.tensor_core.trace_solver_accounting_complete &&
        linear_work_accounting_complete &&
        element_schwarz_accounting_complete &&
        matrix_free_accounting_complete &&
        ((trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64 &&
          result.tensor_core.native_rank1_trace_solve_calls ==
              completed_linear_solve_attempts &&
          result.tensor_core.hypre_trace_solve_calls == 0) ||
         ((trace_solver_kind_ == CudaTraceSolverKind::hypre_parcsr_gmres ||
           trace_solver_kind_ ==
               CudaTraceSolverKind::hypre_host_staged_rank2 ||
           trace_solver_kind_ ==
               CudaTraceSolverKind::hypre_fgmres_mgr_rank1) &&
          result.tensor_core.hypre_trace_solve_calls ==
              completed_linear_solve_attempts &&
          result.tensor_core.native_rank1_trace_solve_calls == 0)) &&
        (trace_solver_kind_ ==
                 CudaTraceSolverKind::hypre_fgmres_mgr_rank1
             ? result.tensor_core.equilibration_builds ==
                       completed_trace_solves &&
                   result.tensor_core.equilibration_passes ==
                       completed_trace_solves *
                           static_cast<std::uint64_t>(
                               cuda_rank1_equilibration_passes) &&
                   result.tensor_core.mgr_setup_calls ==
                       completed_trace_solves &&
                   result.tensor_core.mgr_f_ilu_setups ==
                       result.tensor_core.mgr_setup_calls &&
                   result.tensor_core.mgr_f_ilu_applications ==
                       result.tensor_core.
                           mgr_preconditioner_applications &&
                   (result.linear_iterations == 0 ||
                    result.tensor_core.
                            mgr_preconditioner_applications > 0) &&
                   std::isfinite(
                       result.tensor_core.minimum_left_scale) &&
                   std::isfinite(
                       result.tensor_core.maximum_left_scale) &&
                   std::isfinite(
                       result.tensor_core.minimum_right_scale) &&
                   std::isfinite(
                       result.tensor_core.maximum_right_scale)
             : equilibrated_field_split_selected
                 ? result.tensor_core.equilibration_builds ==
                           completed_trace_solves &&
                       result.tensor_core.equilibration_passes ==
                           completed_trace_solves *
                               static_cast<std::uint64_t>(
                                   native_element_schwarz_equilibration_passes) &&
                       result.tensor_core.mgr_setup_calls == 0 &&
                       result.tensor_core.
                               mgr_preconditioner_applications == 0 &&
                       result.tensor_core.mgr_f_ilu_setups == 0 &&
                       result.tensor_core.mgr_f_ilu_applications == 0
                 : result.tensor_core.equilibration_builds == 0 &&
                       result.tensor_core.equilibration_passes == 0 &&
                       result.tensor_core.mgr_setup_calls == 0 &&
                       result.tensor_core.
                               mgr_preconditioner_applications == 0 &&
                       result.tensor_core.mgr_f_ilu_setups == 0 &&
                       result.tensor_core.mgr_f_ilu_applications == 0);
    result.telemetry = DeviceResident() && NoFallback() &&
        result.tensor_core.trace_solver_accounting_complete &&
        AllocatorPolicy::W8OwnedAllocationCalls() ==
            baseline_w8_owned_allocation_calls_ &&
        AllocatorPolicy::DeallocationEvents() ==
            baseline_deallocation_events_ &&
        allocation_plateau_exact;
    result.success = result.finite && result.admissible && result.recovery &&
        result.true_residual && result.conservation && result.telemetry;
    result.reason = result.success ? FailureReason::none :
        (!result.true_residual ? FailureReason::true_residual_gate
         : !result.telemetry ? FailureReason::telemetry_gate
                             : FailureReason::success_gate);
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
    (void)injection_protocol_failure;
    return result;
  }

  bool ExportOracle(CudaOraclePod& pod) noexcept
  {
    if (!initialized_ || !oracle_captured_ || !OraclePayloadFitsCapacity())
      return false;
    pod = oracle_;
    TransportPolicy::SetPhase(CudaFailurePhase::terminal_evidence,
                              current_generation_);
    return ExportArray(pod.old_interior.data(), view_.old_interior,
                       shape_.interior_count) &&
        ExportArray(pod.current_interior.data(), view_.work_interior,
                    shape_.interior_count) &&
        ExportArray(pod.current_local_trace.data(), view_.current_local_trace,
                    shape_.elements * shape_.trace_dofs) &&
        ExportArray(pod.distributed_local_trace.data(),
                    ranks_ == 1 ? view_.work_true_trace :
                                  view_.distributed_local_trace,
                    shape_.local_trace_count) &&
        ExportArray(pod.current_true_trace.data(), view_.work_true_trace,
                    shape_.true_trace_count) &&
        ExportArray(pod.residual_r.data(), view_.residual_r,
                    shape_.interior_count) &&
        ExportArray(pod.residual_g.data(), view_.residual_g,
                    shape_.elements * shape_.trace_dofs) &&
        ExportArray(pod.a.data(), view_.a,
                    shape_.elements * shape_.interior_dofs *
                        shape_.interior_dofs) &&
        ExportArray(pod.b.data(), view_.b,
                    shape_.elements * shape_.interior_dofs *
                        shape_.trace_dofs) &&
        ExportArray(pod.c.data(), view_.c,
                    shape_.elements * shape_.trace_dofs *
                        shape_.interior_dofs) &&
        ExportArray(pod.d.data(), view_.d,
                    shape_.elements * shape_.trace_dofs *
                        shape_.trace_dofs) &&
        ExportArray(pod.f.data(), view_.f, shape_.interior_count) &&
        ExportArray(pod.g.data(), view_.g,
                    shape_.elements * shape_.trace_dofs) &&
        ExportArray(pod.schur.data(), view_.schur,
                    shape_.elements * shape_.trace_dofs *
                        shape_.trace_dofs) &&
        ExportArray(pod.condensed_rhs.data(), view_.condensed_rhs,
                    shape_.elements * shape_.trace_dofs) &&
        ExportArray(pod.local_trace_increment.data(),
                    view_.trace_increment_local,
                    shape_.elements * shape_.trace_dofs) &&
        ExportArray(pod.recovered_increment.data(), view_.recovered_interior,
                    shape_.interior_count) &&
        ExportTrueSystem(pod);
  }

  /** Map one device element-local state back to a validated byNODES L2 view. */
  bool ExportInterior(const CudaInteriorState& source,
                      CudaMutableHostL2Output destination) noexcept;
  bool ExportFields(
      const CudaInteriorState& source,
      generic::HdgEulerHostInteriorFields destination) noexcept;

private:
  friend struct CudaQualificationTestAccess;

  static bool IgrActive(const CudaConfig& config) noexcept
  {
    return config.use_generic_numerical_policy &&
        config.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::igr;
  }

  static int InteriorComponents(const CudaConfig& config) noexcept
  { return IgrActive(config) ? igr_interior_components : conserved_components; }

  static int TraceComponents(const CudaConfig& config) noexcept
  { return IgrActive(config) ? igr_trace_components : conserved_components; }

  static generic::EulerHdgAssemblyQuadratureOrders AssemblyOrders(
      const CudaConfig& config) noexcept
  {
    if (config.use_generic_numerical_policy)
      return generic::ResolveEulerHdgAssemblyQuadratureOrders(
          config.numerical_policy);
    const int order = config.order == functional_polynomial_order
        ? functional_assembly_quadrature_order : assembly_quadrature_order;
    return {order, order};
  }

  static bool ExactTopologyShapeFitsCapacity(
      const CudaConfig& config, const mfem::ParMesh& mesh,
      const mfem::ParFiniteElementSpace& interior,
      const mfem::ParFiniteElementSpace& trace,
      const dg::hdg::trace::DGInterfaceTraceExchangePlan& exchange) noexcept
  {
    const int elements = mesh.GetNE();
    const bool generic = config.use_borrowed_topology &&
        config.use_borrowed_host_inputs;
    if (elements <= 0 || (!generic && elements > maximum_rank_elements) ||
        !exchange.IsConsistent() || exchange.space != &trace ||
        exchange.map.dofs.entries.size() !=
            static_cast<std::size_t>(trace.GetVSize()) ||
        interior.GetOrdering() != mfem::Ordering::byNODES ||
        trace.GetOrdering() != mfem::Ordering::byNODES)
      return false;

    const mfem::FiniteElement* fe = interior.GetFE(0);
    mfem::Array<int> faces, orientations;
    mesh.GetElementEdges(0, faces, orientations);
    const mfem::FiniteElement* face_fe = faces.Size() == faces_per_element
        ? trace.GetFaceElement(faces[0]) : nullptr;
    if (fe == nullptr || face_fe == nullptr) { return false; }

    const int scalar_interior_dofs = fe->GetDof();
    const int scalar_face_dofs = face_fe->GetDof();
    const int interior_dofs =
        InteriorComponents(config) * scalar_interior_dofs;
    const int trace_dofs =
        faces_per_element * TraceComponents(config) * scalar_face_dofs;
    const auto assembly_orders = AssemblyOrders(config);
    const int volume_points = mfem::IntRules.Get(
        fe->GetGeomType(), assembly_orders.volume_order).GetNPoints();
    const int face_points = mfem::IntRules.Get(
        face_fe->GetGeomType(), assembly_orders.face_order).GetNPoints();
    const bool common_shape = trace.GetVSize() > 0 && trace.GetTrueVSize() > 0 &&
        exchange.map.dofs.global_true_dof_count > 0 &&
        scalar_interior_dofs > 0 &&
        scalar_interior_dofs <= (generic
            ? generic_maximum_scalar_dofs_per_element
            : maximum_scalar_dofs_per_element) &&
        scalar_face_dofs > 0 &&
        scalar_face_dofs <= (generic
            ? generic_maximum_scalar_dofs_per_face
            : maximum_scalar_dofs_per_face) &&
        interior_dofs <= (generic
            ? generic_maximum_interior_dofs_per_element
            : maximum_interior_dofs_per_element) &&
        trace_dofs <= (generic
            ? generic_maximum_trace_dofs_per_element
            : maximum_trace_dofs_per_element) &&
        volume_points <= maximum_volume_points &&
        face_points <= maximum_face_points;
    if (generic) return common_shape;
    return common_shape && trace.GetVSize() <= maximum_local_trace_dofs &&
        trace.GetTrueVSize() <= maximum_true_trace_dofs &&
        exchange.map.dofs.global_true_dof_count <=
            static_cast<std::uint64_t>(maximum_true_trace_dofs);
  }

  bool ValidateBorrowedTopologyCapacity(const CudaConfig& config) const
      noexcept
  {
    bool local_valid = false;
    try
    {
      mfem::L2_FECollection interior_collection(
          config.order, dimension, mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      mfem::DG_Interface_FECollection trace_collection(
          config.order, dimension, mfem::FiniteElement::VALUE,
          mfem::BasisType::GaussLegendre);
      mfem::ParFiniteElementSpace interior_space(
          config.borrowed_mesh, &interior_collection,
          InteriorComponents(config),
          mfem::Ordering::byNODES);
      mfem::ParFiniteElementSpace trace_space(
          config.borrowed_mesh, &trace_collection, TraceComponents(config),
          mfem::Ordering::byNODES);
      interior_space.ExchangeFaceNbrData();
      trace_space.ExchangeFaceNbrData();
      dg::hdg::trace::DGInterfaceTraceExchangePlan exchange;
      local_valid = dg::hdg::trace::BuildDGInterfaceTraceExchangePlan(
          trace_space, exchange) && ExactTopologyShapeFitsCapacity(
          config, *config.borrowed_mesh, interior_space, trace_space,
          exchange);
    }
    catch (...) { local_valid = false; }
    return CollectiveContinue(local_valid);
  }

  bool ValidateTopologyBeforeDevice(const CudaConfig& config) const noexcept
  {
    const bool known_engine =
        generic::IsKnownCudaEngineKind(config.cuda_engine);
    const bool tensor_request_valid =
        config.cuda_engine != generic::CudaEngineKind::tensor_core ||
        (config.use_borrowed_topology && config.use_borrowed_host_inputs &&
         config.use_generic_numerical_policy);
    const bool supported_rank = config.expected_rank_count == 1 ||
        config.expected_rank_count == 2;
    const bool supported_order = config.use_borrowed_topology &&
            config.use_borrowed_host_inputs
        ? config.order >= generic::minimum_polynomial_order &&
            config.order <= generic::maximum_polynomial_order
        : config.order == smoke_polynomial_order ||
            config.order == functional_polynomial_order;
    int collective_ranks = 0;
    const bool communicator_ready = communicator_ != MPI_COMM_NULL &&
        MPI_Comm_size(communicator_, &collective_ranks) == MPI_SUCCESS;
    bool local_valid = known_engine && tensor_request_valid &&
        supported_rank && supported_order && communicator_ready &&
        collective_ranks == config.expected_rank_count;
    int local_elements = 0;

    if (config.use_borrowed_topology)
    {
      local_valid = local_valid && config.borrowed_mesh != nullptr &&
          config.borrowed_communicator != MPI_COMM_NULL;
      const bool static_physical_boundaries =
          config.use_borrowed_host_inputs &&
          config.borrowed_boundary_policies.count != 0;
      local_valid = local_valid && (static_physical_boundaries
          ? IsCudaBorrowedGlobalTopologyIdentity(
              config.borrowed_global_topology)
          : IsCudaAbsentBorrowedGlobalTopologyIdentity(
              config.borrowed_global_topology));
      if (local_valid)
      {
        int caller_collective = MPI_UNEQUAL;
        int mesh_caller = MPI_UNEQUAL;
        const MPI_Comm mesh_communicator = config.borrowed_mesh->GetComm();
        local_valid =
            MPI_Comm_compare(config.borrowed_communicator, communicator_,
                             &caller_collective) == MPI_SUCCESS &&
            (caller_collective == MPI_IDENT ||
             caller_collective == MPI_CONGRUENT) &&
            MPI_Comm_compare(config.borrowed_communicator, mesh_communicator,
                             &mesh_caller) == MPI_SUCCESS &&
            (mesh_caller == MPI_IDENT || mesh_caller == MPI_CONGRUENT) &&
            !config.borrowed_mesh->Nonconforming() &&
            config.borrowed_mesh->Dimension() == dimension;
        local_elements = config.borrowed_mesh->GetNE();
        local_valid = local_valid && local_elements > 0;
        for (int element = 0; local_valid && element < local_elements;
             ++element)
        {
          local_valid = config.borrowed_mesh->GetElementBaseGeometry(element) ==
              mfem::Geometry::SQUARE;
        }
      }
    }
    else
    {
      // Do not silently discard an incomplete caller-owned request.
      local_valid = local_valid && config.borrowed_mesh == nullptr &&
          config.borrowed_communicator == MPI_COMM_NULL;
    }
    if (!CollectiveContinue(local_valid)) { return false; }

    if (!config.use_borrowed_topology) { return true; }
    // A borrowed mesh may contain ordinary periodic/shared faces and true
    // physical boundaries.  Classify every element-local face before device
    // setup: an unpaired nonshared face without a boundary element is a hole,
    // while every true boundary attribute must map to exactly one admitted
    // copied static policy.  This replaces the old periodic-only face census.
    long long local_face_measure = 0;
    long long local_physical_faces = 0;
    std::array<long long, maximum_static_boundary_policies>
        local_policy_uses{};
    std::array<long long, maximum_static_boundary_policies>
        global_policy_uses{};
    try
    {
      const int face_count = config.borrowed_mesh->GetNumFaces();
      if (face_count < 0) local_valid = false;
      std::vector<unsigned char> shared(
          local_valid ? static_cast<std::size_t>(face_count) : 0, 0);
      for (int index = 0; local_valid &&
           index < config.borrowed_mesh->GetNSharedFaces(); ++index)
      {
        const int face = config.borrowed_mesh->GetSharedFace(index);
        if (face < 0 || face >= static_cast<int>(shared.size())) local_valid = false;
        else shared[static_cast<std::size_t>(face)] = 1;
      }
      local_valid = local_valid &&
          ValidateCudaStaticBoundaryPolicyTable(config.borrowed_boundary_policies);
      for (int element = 0; local_valid && element < local_elements; ++element)
      {
        mfem::Array<int> faces, orientations;
        config.borrowed_mesh->GetElementEdges(element, faces, orientations);
        if (faces.Size() != faces_per_element) { local_valid = false; break; }
        for (int local_face = 0; local_face < faces.Size(); ++local_face)
        {
          const int face = faces[local_face];
          mfem::FaceElementTransformations* transformations =
              config.borrowed_mesh->GetFaceElementTransformations(face);
          if (face < 0 || face >= static_cast<int>(shared.size()) ||
              transformations == nullptr || transformations->Face == nullptr ||
              (transformations->Elem1No != element &&
               transformations->Elem2No != element))
          { local_valid = false; break; }
          const int boundary_element = config.borrowed_mesh->GetFaceToBdrElMap()[face];
          const bool is_shared = shared[static_cast<std::size_t>(face)] != 0;
          const bool is_physical = !is_shared && transformations->Elem2No < 0 &&
              boundary_element >= 0;
          if (!is_physical)
          {
            // Shared faces and local paired faces retain ordinary trace
            // coupling.  A nonshared one-sided face with no boundary tag is
            // an unsupported topology hole.
            if (!is_shared && transformations->Elem2No < 0) local_valid = false;
            if (!local_valid) break;
            continue;
          }
          const int attribute = config.borrowed_mesh->GetBdrAttribute(boundary_element);
          const CudaStaticBoundaryPolicy* found = nullptr;
          std::size_t found_index = maximum_static_boundary_policies;
          for (std::size_t policy = 0;
               policy < config.borrowed_boundary_policies.count; ++policy)
            if (config.borrowed_boundary_policies.policies[policy].attribute == attribute)
            {
              found = config.borrowed_boundary_policies.policies + policy;
              found_index = policy;
            }
          if (attribute <= 0 || found == nullptr ||
              found_index >= config.borrowed_boundary_policies.count ||
              !IsCudaAdmittedStaticBoundaryPolicy(found->kind) ||
              local_policy_uses[found_index] ==
                  std::numeric_limits<long long>::max())
          { local_valid = false; break; }
          ++local_policy_uses[found_index];
          ++local_physical_faces;
        }
      }
    }
    catch (...) { local_valid = false; }
    const long long faces = config.borrowed_mesh->GetNumFaces();
    const long long shared_count = config.borrowed_mesh->GetNSharedFaces();
    local_valid = local_valid && faces >= 0 && shared_count >= 0 &&
        shared_count <= faces && faces <= std::numeric_limits<long long>::max() / 2;
    if (local_valid) local_face_measure = 2 * faces - shared_count;
    if (!CollectiveContinue(local_valid)) return false;
    const long long local_element_count = local_elements;
    long long global_elements = 0;
    long long global_face_measure = 0;
    long long global_physical_faces = 0;
    const bool element_census = MPI_Allreduce(&local_element_count,
        &global_elements, 1, MPI_LONG_LONG_INT, MPI_SUM,
        communicator_) == MPI_SUCCESS;
    const bool face_census = MPI_Allreduce(&local_face_measure,
        &global_face_measure, 1, MPI_LONG_LONG_INT, MPI_SUM,
        communicator_) == MPI_SUCCESS;
    const bool boundary_census = MPI_Allreduce(&local_physical_faces,
        &global_physical_faces, 1, MPI_LONG_LONG_INT, MPI_SUM,
        communicator_) == MPI_SUCCESS;
    const bool policy_usage_census = MPI_Allreduce(local_policy_uses.data(),
        global_policy_uses.data(),
        static_cast<int>(global_policy_uses.size()), MPI_LONG_LONG_INT,
        MPI_SUM, communicator_) == MPI_SUCCESS;
    bool every_policy_used = config.borrowed_boundary_policies.count == 0
        ? global_physical_faces == 0
        : global_physical_faces > 0;
    for (std::size_t policy = 0; every_policy_used &&
         policy < config.borrowed_boundary_policies.count; ++policy)
      every_policy_used = global_policy_uses[policy] > 0;
    if (!element_census || !face_census || !boundary_census ||
        !policy_usage_census || !every_policy_used ||
        global_elements < 0 || global_physical_faces < 0 ||
        global_elements > (std::numeric_limits<long long>::max() -
            global_physical_faces) / faces_per_element ||
        global_face_measure != faces_per_element * global_elements +
            global_physical_faces)
      return false;
    return ValidateBorrowedTopologyCapacity(config);
  }

  bool ConfigureDeviceAndRanks(const CudaConfig& config,
                               CudaRunEvidence& evidence) noexcept
  {
    MPI_Comm_rank(communicator_, &rank_);
    MPI_Comm_size(communicator_, &ranks_);
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

    // Capture a stale flag before clearing it.  HYPRE_ClearAllErrors() returns
    // zero after the clear and is therefore never used as prior-error evidence.
    const HYPRE_Int hypre_incoming_error = HYPRE_GetError();
    const bool hypre_incoming_error_free = hypre_incoming_error == 0;
    if (!hypre_incoming_error_free)
      backend_failure_mask |= backend_hypre_incoming_error_failure;
    (void)HYPRE_ClearAllErrors();

    // No rank calls Hypre::Init unless every rank arrived with a clean flag.
    const bool local_preinit_valid = hypre_incoming_error_free;
    const bool preinit_collectively_valid =
        CollectiveContinue(local_preinit_valid);
    if (!preinit_collectively_valid)
    {
      ReportCudaBackendFailure(rank_, backend_failure_mask);
      return false;
    }

    bool hypre_init_returned = false;
    try
    {
      // Accepted W7 parity: create/configure the rank-local HYPRE singleton
      // before querying its device policy or allocator pool identity.
      mfem::Hypre::Init();
      hypre_init_returned = true;
    }
    catch (...) {}
    // Capture Init's error before any later clear can erase it.
    const HYPRE_Int hypre_init_error = HYPRE_GetError();
    const bool hypre_init_error_free = hypre_init_error == 0;
    if (!hypre_init_returned)
      backend_failure_mask |= backend_hypre_initialize_return_failure;
    if (!hypre_init_error_free)
      backend_failure_mask |= backend_hypre_initialize_error_failure;

    // This second gate is intentionally before every later HYPRE/Umpire query.
    // A rank-local throw or Init error converges before any peer advances.
    const bool local_hypre_init_valid =
        hypre_init_returned && hypre_init_error_free;
    const bool hypre_init_collectively_valid =
        CollectiveContinue(local_hypre_init_valid);
    if (!hypre_init_collectively_valid)
    {
      ReportCudaBackendFailure(rank_, backend_failure_mask);
      return false;
    }

    // Init's zero error was recorded before this reset.  From this point all
    // ranks execute the same policy/query/pool sequence unconditionally.
    (void)HYPRE_ClearAllErrors();
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
    const int device_architecture = 10 * properties.major + properties.minor;
    tensor_core_hardware_supported_ = device_architecture == 120;
    evidence.tensor_core.selected_engine = cuda_engine_;
    evidence.tensor_core.trace_policy = trace_policy_descriptor_.selection;
    evidence.tensor_core.hardware_supported =
        tensor_core_hardware_supported_;
    if (cuda_engine_ == generic::CudaEngineKind::tensor_core &&
        !CollectiveContinue(tensor_core_hardware_supported_))
    {
      backend_failure_mask |= backend_tensor_core_hardware_failure;
      ReportCudaTensorCoreAdmissionFailure(rank_, device_architecture);
      ReportCudaBackendFailure(rank_, backend_failure_mask);
      return false;
    }
    std::memcpy(gpu_uuid_.data(), properties.uuid.bytes, gpu_uuid_.size());
    std::array<unsigned char, 32> gathered{};
    if (ranks_ == 2)
    {
      if (MPI_Allgather(gpu_uuid_.data(), 16, MPI_UNSIGNED_CHAR,
                        gathered.data(), 16, MPI_UNSIGNED_CHAR,
                        communicator_) != MPI_SUCCESS)
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

  bool BuildTopology(const CudaConfig& config) noexcept
  {
    try
    {
      borrowed_topology_ = config.use_borrowed_topology;
      if (borrowed_topology_)
      {
        // The caller owns the ParMesh and its partition.  This engine borrows
        // it only to construct FE spaces; it never rebuilds or repartitions it.
        mesh_ = config.borrowed_mesh;
      }
      else
      {
        mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
            config.mesh_n, config.mesh_n,
            mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
        std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
        translations[0][0] = 1.0;
        translations[1][1] = 1.0;
        serial_mesh_ = mfem::Mesh::MakePeriodic(
            original, original.CreatePeriodicVertexMapping(translations));
        std::array<int, maximum_rank_elements> partition{};
        const int global_elements = config.mesh_n * config.mesh_n;
        for (int element = 0; element < global_elements; ++element)
          partition[static_cast<std::size_t>(element)] =
              element * ranks_ / global_elements;
        owned_mesh_.emplace(communicator_, serial_mesh_, partition.data());
        mesh_ = &*owned_mesh_;
      }
      interior_collection_.emplace(
          config.order, dimension, mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      trace_collection_.emplace(
          config.order, dimension, mfem::FiniteElement::VALUE,
          mfem::BasisType::GaussLegendre);
      interior_space_.emplace(mesh_, &*interior_collection_,
                              InteriorComponents(config),
                              mfem::Ordering::byNODES);
      trace_space_.emplace(mesh_, &*trace_collection_, TraceComponents(config),
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
    shape_.local_trace_count = trace_space_->GetVSize();
    shape_.true_trace_count = trace_space_->GetTrueVSize();
    const std::uint64_t global_true = exchange_plan_.map.dofs.global_true_dof_count;
    const mfem::FiniteElement* fe = interior_space_->GetFE(0);
    mfem::Array<int> first_faces, first_orientations;
    mesh_->GetElementEdges(0, first_faces, first_orientations);
    const mfem::FiniteElement* face_fe =
        trace_space_->GetFaceElement(first_faces[0]);
    shape_.scalar_interior_dofs = fe == nullptr ? 0 : fe->GetDof();
    shape_.scalar_face_dofs = face_fe == nullptr ? 0 : face_fe->GetDof();
    shape_.interior_components = InteriorComponents(config);
    shape_.trace_components = TraceComponents(config);
    shape_.interior_dofs =
        shape_.interior_components * shape_.scalar_interior_dofs;
    shape_.trace_dofs = faces_per_element * shape_.trace_components *
        shape_.scalar_face_dofs;
    const auto assembly_orders = AssemblyOrders(config);
    shape_.volume_assembly_order = assembly_orders.volume_order;
    shape_.face_assembly_order = assembly_orders.face_order;
    shape_.order = config.order;
    shape_.igr_active = IgrActive(config);
    shape_.igr_alpha = shape_.igr_active
        ? config.numerical_policy.stabilization.alpha : 0.0;
    shape_.igr_beta = shape_.igr_active
        ? config.numerical_policy.stabilization.beta : 0.0;
    shape_.boundary_gamma = config.borrowed_boundary_policies.gamma;
    const long long expected_interior = static_cast<long long>(shape_.elements) *
        static_cast<long long>(shape_.interior_dofs);
    const int interior_vsize = interior_space_->GetVSize();
    if (shape_.elements <= 0 || shape_.local_trace_count <= 0 ||
        shape_.true_trace_count <= 0 || interior_vsize <= 0 ||
        expected_interior <= 0 || expected_interior >
            std::numeric_limits<int>::max() ||
        interior_vsize != expected_interior || global_true == 0 ||
        global_true > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      return false;
    shape_.interior_count = interior_vsize;
    shape_.global_true_trace_count = static_cast<int>(global_true);
    shape_.borrowed_static_source = false;
    shape_.borrowed_host_inputs = config.use_borrowed_host_inputs;
    shape_.borrowed_static_boundary_policies =
        config.use_borrowed_host_inputs &&
        config.borrowed_boundary_policies.count != 0;
    if (!ExactTopologyShapeFitsCapacity(config, *mesh_, *interior_space_,
                                        *trace_space_, exchange_plan_))
      return false;
    shape_.volume_points = mfem::IntRules.Get(
        fe->GetGeomType(), shape_.volume_assembly_order).GetNPoints();
    shape_.face_points = mfem::IntRules.Get(
        face_fe->GetGeomType(), shape_.face_assembly_order).GetNPoints();
    if (shape_.volume_points > maximum_volume_points ||
        shape_.face_points > maximum_face_points)
      return false;
    return true;
  }

  bool BuildHostGeometryAndMaps(const CudaConfig&) noexcept;
  bool AllocateTopologyStorage() noexcept
  {
    try
    {
      const bool generic = borrowed_topology_ && shape_.borrowed_host_inputs;
      const bool native_trace = trace_solver_kind_ ==
          CudaTraceSolverKind::native_rank1_fp64;
      if ((native_trace && (!generic || !native_trace_requirements_.valid ||
                            !native_trace_support_.supported)) ||
          (!native_trace &&
           (native_trace_requirements_.valid ||
            native_trace_support_.supported)))
        return false;
      const auto checked_product = [](const std::size_t a,
                                      const std::size_t b,
                                      std::size_t& result)
      {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
          return false;
        result = a * b;
        return true;
      };
      const auto checked_add = [](std::size_t& total, const std::size_t value)
      {
        if (value > std::numeric_limits<std::size_t>::max() - total)
          return false;
        total += value;
        return true;
      };
      const std::size_t elements = static_cast<std::size_t>(shape_.elements);
      const std::size_t interior = static_cast<std::size_t>(shape_.interior_count);
      const std::size_t local_trace = static_cast<std::size_t>(shape_.local_trace_count);
      const std::size_t true_trace = static_cast<std::size_t>(shape_.true_trace_count);
      const std::size_t interior_dofs =
          static_cast<std::size_t>(shape_.interior_dofs);
      const std::size_t trace_dofs = static_cast<std::size_t>(shape_.trace_dofs);
      const std::size_t volume_points =
          static_cast<std::size_t>(shape_.volume_points);
      const std::size_t face_points = static_cast<std::size_t>(shape_.face_points);
      const std::size_t scalar_interior_dofs =
          static_cast<std::size_t>(shape_.scalar_interior_dofs);
      const std::size_t scalar_face_dofs =
          static_cast<std::size_t>(shape_.scalar_face_dofs);
      CudaGenericArenaLayout layout{};
      std::size_t interior_square = 0, element_volume_points = 0;
      std::size_t element_faces = 0, face_points_total = 0;
      std::size_t trace_plus_one = 0, solve_base = 0;
      bool valid = elements != 0 && checked_product(elements, trace_dofs,
          layout.element_trace) &&
          checked_product(elements, dimension, layout.element_coordinates) &&
          checked_product(layout.element_trace, trace_dofs, layout.local_matrix) &&
          checked_product(interior_dofs, interior_dofs, interior_square) &&
          checked_product(elements, interior_square, layout.element_interior2) &&
          checked_product(interior_dofs, trace_dofs, interior_square) &&
          checked_product(elements, interior_square, layout.element_coupling) &&
          checked_product(elements, volume_points, element_volume_points) &&
          checked_product(element_volume_points, scalar_interior_dofs,
                          layout.volume_shape) &&
          checked_product(layout.volume_shape, dimension, layout.volume_derivative) &&
          checked_product(elements, volume_points, layout.volume_weight) &&
          checked_product(layout.volume_weight, dimension, layout.volume_position) &&
          checked_product(elements, faces_per_element, element_faces) &&
          checked_product(element_faces, face_points, face_points_total) &&
          checked_product(face_points_total, scalar_interior_dofs,
                          layout.face_interior) &&
          checked_product(face_points_total, scalar_face_dofs, layout.face_trace) &&
          checked_product(element_faces, face_points, layout.face_weight) &&
          checked_product(layout.face_weight, dimension, layout.face_normal) &&
          checked_add(trace_plus_one, trace_dofs) && checked_add(trace_plus_one, 1) &&
          checked_product(elements, interior_dofs, solve_base) &&
          checked_product(solve_base, trace_plus_one, layout.solve_workspace) &&
          checked_product(layout.local_matrix, 2, layout.two_exchange_keys);
      if (!valid) return false;
      layout.elements = elements;
      layout.interior = interior;
      layout.local_trace = local_trace;
      layout.true_trace = true_trace;
      layout.element_faces = element_faces;
      layout.face_h = shape_.igr_active ? element_faces : 0;
      valid = valid && checked_product(element_faces, conserved_components,
                                       layout.boundary_reference);
      layout.boundary_pressure = element_faces;
      layout.native_ilu_inverse_diagonal =
          native_trace && CudaTracePreconditionerIsIlu(
              trace_policy_descriptor_.trace_preconditioner)
          ? true_trace : 0;
      if (!valid) return false;
      const std::size_t host_elements = generic ? layout.elements : maximum_rank_elements;
      const std::size_t host_interior = generic ? interior :
          static_cast<std::size_t>(maximum_rank_elements) * maximum_interior_dofs_per_element;
      const std::size_t host_element_trace = generic ? layout.element_trace :
          static_cast<std::size_t>(maximum_rank_elements) * maximum_trace_dofs_per_element;
      const std::size_t host_local_trace = generic ? local_trace : maximum_local_trace_dofs;
      const std::size_t host_true_trace = generic ? true_trace : maximum_true_trace_dofs;
      const std::size_t host_matrix = generic ? layout.local_matrix : maximum_local_matrix_entries;
      const std::size_t host_exchange = generic ? layout.local_matrix : maximum_exchange_entries;

      signed_element_trace_.assign(host_element_trace, 0);
      local_global_true_.assign(host_local_trace, 0);
      local_owner_.assign(host_local_trace, 0);
      local_true_index_.assign(host_local_trace, 0);
      owned_global_true_.assign(host_true_trace, 0);
      element_coordinate_keys_.assign(generic ? layout.element_coordinates :
          host_elements * dimension, 0);
      host_volume_shape_.assign(generic ? layout.volume_shape :
          static_cast<std::size_t>(maximum_rank_elements) * maximum_volume_points * maximum_scalar_dofs_per_element, 0.0);
      host_volume_derivative_.assign(generic ? layout.volume_derivative :
          static_cast<std::size_t>(maximum_rank_elements) * maximum_volume_points * maximum_scalar_dofs_per_element * dimension, 0.0);
      host_volume_weight_.assign(generic ? layout.volume_weight :
          static_cast<std::size_t>(maximum_rank_elements) * maximum_volume_points, 0.0);
      host_volume_position_.assign(generic ? layout.volume_position :
          static_cast<std::size_t>(maximum_rank_elements) * maximum_volume_points * dimension, 0.0);
      host_face_interior_shape_.assign(generic ? layout.face_interior :
          static_cast<std::size_t>(maximum_rank_elements) * faces_per_element * maximum_face_points * maximum_scalar_dofs_per_element, 0.0);
      host_face_trace_shape_.assign(generic ? layout.face_trace :
          static_cast<std::size_t>(maximum_rank_elements) * faces_per_element * maximum_face_points * maximum_scalar_dofs_per_face, 0.0);
      host_face_weight_.assign(generic ? layout.face_weight :
          static_cast<std::size_t>(maximum_rank_elements) * faces_per_element * maximum_face_points, 0.0);
      host_face_normal_.assign(generic ? layout.face_normal :
          static_cast<std::size_t>(maximum_rank_elements) * faces_per_element * maximum_face_points * dimension, 0.0);
      host_face_h_.assign(generic && shape_.igr_active ? layout.face_h : 0,
                          1.0);
      host_boundary_reference_.assign(generic &&
          shape_.borrowed_static_boundary_policies ? layout.boundary_reference : 0,
          0.0);
      host_boundary_pressure_.assign(generic &&
          shape_.borrowed_static_boundary_policies ? layout.boundary_pressure : 0,
          0.0);
      host_boundary_policy_kind_.assign(generic &&
          shape_.borrowed_static_boundary_policies ? layout.element_faces : 0, 0);
      host_exact_interior_.assign(host_interior, 0.0);
      host_perturbed_interior_.assign(host_interior, 0.0);
      host_exact_true_trace_.assign(host_true_trace, 0.0);
      host_perturbed_true_trace_.assign(host_true_trace, 0.0);
      host_static_source_.assign(host_interior, 0.0);
      host_export_interior_.assign(
          static_cast<std::size_t>(shape_.elements) *
              (shape_.igr_active ? igr_published_components
                                 : conserved_components) *
              shape_.scalar_interior_dofs,
          0.0);
      signed_element_interior_.assign(host_interior, 0);
      host_matrix_owned_source_.assign(host_matrix, 0);
      host_matrix_owned_section_.assign(host_matrix, 0);
      host_matrix_owned_target_.assign(host_matrix, 0);
      host_matrix_send_source_.assign(host_exchange, 0);
      host_matrix_receive_section_.assign(host_exchange, 0);
      host_matrix_receive_target_.assign(host_exchange, 0);
      host_vector_owned_source_.assign(host_element_trace, 0);
      host_vector_owned_target_.assign(host_element_trace, 0);
      host_vector_send_source_.assign(host_element_trace, 0);
      host_vector_receive_target_.assign(host_element_trace, 0);
      matrix_send_keys_.assign(generic ? layout.two_exchange_keys :
          2 * host_exchange, 0);
      matrix_receive_keys_.assign(generic ? layout.two_exchange_keys :
          2 * host_exchange, 0);
      vector_send_keys_.assign(host_element_trace, 0);
      vector_receive_keys_.assign(host_element_trace, 0);
      matrix_send_host_.assign(host_exchange, 0.0);
      matrix_receive_host_.assign(host_exchange, 0.0);
      vector_send_host_.assign(host_element_trace, 0.0);
      vector_receive_host_.assign(host_element_trace, 0.0);

      if (!generic) return true;
      std::size_t doubles = 0;
      std::size_t ten_interior = 0, four_true = 0, three_element_trace = 0;
      std::size_t two_coupling = 0, two_local_matrix = 0;
      std::size_t two_element_trace = 0, three_local_matrix = 0;
      std::size_t four_element_trace = 0;
      valid = checked_product(interior, 10, ten_interior) &&
          checked_product(true_trace, 4, four_true) &&
          checked_product(layout.element_trace, 3, three_element_trace) &&
          checked_product(layout.element_coupling, 2, two_coupling) &&
          checked_product(layout.local_matrix, 2, two_local_matrix) &&
          checked_product(layout.element_trace, 2, two_element_trace) &&
          checked_product(layout.local_matrix, 3, three_local_matrix) &&
          checked_product(layout.element_trace, 4, four_element_trace) &&
          checked_add(doubles, ten_interior) &&
          checked_add(doubles, four_true) && checked_add(doubles, local_trace) &&
          checked_add(doubles, three_element_trace) && checked_add(doubles, interior) &&
          checked_add(doubles, layout.element_trace) &&
          checked_add(doubles, layout.element_interior2) &&
          checked_add(doubles, two_coupling) && checked_add(doubles, layout.local_matrix) &&
          checked_add(doubles, interior) && checked_add(doubles, layout.element_trace) &&
          checked_add(doubles, layout.element_interior2) && checked_add(doubles, interior) &&
          checked_add(doubles, layout.solve_workspace) && checked_add(doubles, layout.local_matrix) &&
          checked_add(doubles, layout.element_trace) && checked_add(doubles, true_trace) &&
          checked_add(doubles, two_local_matrix) && checked_add(doubles, two_element_trace) &&
          checked_add(doubles, layout.volume_shape) && checked_add(doubles, layout.volume_derivative) &&
          checked_add(doubles, layout.volume_weight) && checked_add(doubles, layout.volume_position) &&
          checked_add(doubles, layout.face_interior) && checked_add(doubles, layout.face_trace) &&
          checked_add(doubles, layout.face_weight) && checked_add(doubles, layout.face_normal) &&
          (!shape_.igr_active || checked_add(doubles, layout.face_h)) &&
          checked_add(doubles, elements) &&
          checked_add(doubles, euler_conservation_result_scalars) &&
          checked_add(doubles, layout.face_weight) &&
          (!shape_.borrowed_static_boundary_policies ||
           (checked_add(doubles, layout.boundary_reference) &&
            checked_add(doubles, layout.boundary_pressure))) &&
          (!native_trace ||
           (checked_add(doubles, layout.native_ilu_inverse_diagonal) &&
            checked_add(doubles,
                        native_trace_requirements_.basis_doubles) &&
            checked_add(doubles,
                        native_trace_requirements_.vector_doubles) &&
            checked_add(doubles,
                        native_trace_requirements_.reduction_doubles) &&
            checked_add(doubles,
                        native_trace_requirements_.small_doubles) &&
            checked_add(doubles,
                        native_trace_requirements_.control_double_slots)));
      std::size_t integers = 17;
      valid = valid && checked_add(integers, layout.element_trace) &&
          checked_add(integers, three_local_matrix) &&
          checked_add(integers, three_local_matrix) &&
          checked_add(integers, four_element_trace) &&
          (!shape_.borrowed_static_boundary_policies ||
           checked_add(integers, layout.element_faces)) &&
          (!native_trace || checked_add(
              integers,
              native_trace_requirements_.permutation_integers));
      if (!valid || doubles > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          integers > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return false;
      double_arena_capacity_ = doubles;
      integer_arena_capacity_ = integers;
      generic_arena_layout_ = layout;
      return true;
    }
    catch (...) { return false; }
  }
  bool InitializeBorrowedHostFields(const CudaConfig& config) noexcept;
  bool BuildLocalTraceGraph() noexcept;
  bool BuildPersistentTrueSystem() noexcept;
  bool BuildSparseStagingPlan() noexcept;
  bool BuildAndObserveRankOneEquilibratedSystem(
      CudaEquilibrationObservation& observation) noexcept;

  bool AllocateArenas() noexcept
  {
    void* doubles = nullptr;
    void* integers = nullptr;
    if (!AllocatorPolicy::Allocate(
            double_arena_capacity_ * sizeof(double), alignof(double),
            CudaAllocationSite::persistent_double_arena, doubles))
      return false;
    if (!AllocatorPolicy::Allocate(
            integer_arena_capacity_ * sizeof(int), alignof(std::uint64_t),
            CudaAllocationSite::persistent_integer_arena, integers))
    {
      // Retain the local token until the common release path observes an exact
      // successful deallocation.  A failed cleanup is quarantined, never lost.
      double_arena_ = static_cast<double*>(doubles);
      (void)ReleaseArenas();
      return false;
    }
    double_arena_ = static_cast<double*>(doubles);
    integer_arena_ = static_cast<int*>(integers);
    return double_arena_ != nullptr && integer_arena_ != nullptr;
  }

  struct ArenaReleaseOperations
  {
    CudaPersistentNumerics& owner;

    bool HasIntegerArena() const noexcept
    { return owner.integer_arena_ != nullptr; }
    bool HasDoubleArena() const noexcept
    { return owner.double_arena_ != nullptr; }

    void Marker(const char* const stage, const char* const edge,
                const CudaArenaReleaseState& state,
                const std::uint64_t failure_mask) const noexcept
    {
      ReportCudaArenaTeardownMarker(owner.rank_, stage, edge, state,
                                    failure_mask);
    }

    bool ReleaseVectorAlias() noexcept
    {
      try
      {
        owner.double_arena_vector_.Destroy();
        return owner.double_arena_vector_.Size() == 0 &&
            owner.double_arena_vector_.GetMemory().Empty();
      }
      catch (...) { return false; }
    }
    bool ReleaseDoubleRegistration() noexcept
    {
      try
      {
        owner.double_memory_.Delete();
        return owner.double_memory_.Empty();
      }
      catch (...) { return false; }
    }
    bool ReleaseIntegerRegistration() noexcept
    {
      try
      {
        owner.integer_memory_.Delete();
        return owner.integer_memory_.Empty();
      }
      catch (...) { return false; }
    }
    void MakeWrappersInert() noexcept
    {
      // Reset never unregisters.  It only prevents member destruction from
      // retrying Delete after a caught failure; MFEM retains the registration.
      owner.double_arena_vector_.GetMemory().Reset();
      owner.double_memory_.Reset();
      owner.integer_memory_.Reset();
    }
    void ClearViews() noexcept
    {
      owner.view_ = {};
      owner.committed_slot_ = {};
      owner.candidate_slot_ = {};
      owner.workspace_slot_ = {};
    }

    CudaRawReleaseResult ReleaseIntegerArena() noexcept
    { return ReleaseRaw(owner.integer_arena_,
                        CudaAllocationSite::persistent_integer_arena); }
    CudaRawReleaseResult ReleaseDoubleArena() noexcept
    { return ReleaseRaw(owner.double_arena_,
                        CudaAllocationSite::persistent_double_arena); }

    template <typename T>
    CudaRawReleaseResult ReleaseRaw(
        T*& pointer, const CudaAllocationSite site) noexcept
    {
      if (pointer == nullptr) { return CudaRawReleaseResult::released; }
      const std::uint64_t before = AllocatorPolicy::DeallocationEvents();
      const bool released = AllocatorPolicy::Deallocate(pointer, site);
      const std::uint64_t after = AllocatorPolicy::DeallocationEvents();
      if (released &&
          before != std::numeric_limits<std::uint64_t>::max() &&
          after == before + 1)
      {
        pointer = nullptr;
        return CudaRawReleaseResult::released;
      }
      // The policy attempted the call but did not attest its completion.  The
      // token remains owned and is quarantined from automatic retry.
      return CudaRawReleaseResult::quarantined;
    }
  };

  CudaArenaReleasePass ReleaseArenas() noexcept
  {
    ArenaReleaseOperations operations{*this};
    const CudaArenaReleasePass result =
        ReleaseCudaArenas(arena_release_state_, operations);
    if (result.failure_mask != 0)
      ReportCudaArenaTeardownFailure(rank_, result.failure_mask);
    return result;
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
                          double* true_values,
                          bool completion_required) noexcept;
  bool AssembleTrueSystem() noexcept;
  bool LaunchAndObserveNativeRankOneTraceSolve(
      const CudaNativeIlu0FactorView& factor,
      const CudaNativeElementSchwarzView& element_schwarz,
      std::uint64_t matrix_generation,
      double requested_relative_tolerance,
      CudaSolveDiagnostics& diagnostics) noexcept;
  bool SolveTrueSystemAttempt(double& true_relative_residual,
                       std::uint64_t matrix_generation,
                       double requested_relative_tolerance,
                       bool same_matrix_retry,
                       int& linear_iterations,
                       CudaTensorCoreEvidence& tensor_core_evidence) noexcept;
  bool SolveTrueSystem(generic::EulerHdgLinearSolveRole role,
                       double& true_relative_residual,
                       std::uint64_t matrix_generation,
                       int& linear_iterations,
                       int& linear_solve_attempts,
                       int& true_residual_tightenings,
                       CudaTensorCoreEvidence& tensor_core_evidence) noexcept;
  bool AssembleAndNorm(state_slot& workspace, double dt,
                       double& norm) noexcept;
  bool CopyBase(state_slot& workspace) noexcept;
  bool RestoreBase(state_slot& workspace) noexcept;
  bool UpdateStateHashes(state_slot& workspace) noexcept;
  bool HashState(const double* values, int count, StateHash& hash) noexcept;
  bool AggregateHash(const StateHash& local, StateHash& aggregate) const
      noexcept;
  std::size_t WorkspaceBytes() const noexcept;
  bool OraclePayloadFitsCapacity() const noexcept;
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
  bool tensor_core_hardware_supported_ = false;
  bool oracle_captured_ = false;
  bool oracle_capture_requested_ = false;
  generic::CudaEngineKind cuda_engine_ =
      generic::CudaEngineKind::ordinary;
  CudaTraceSolverKind trace_solver_kind_ =
      CudaTraceSolverKind::unconfigured;
  CudaTracePolicyDescriptor trace_policy_descriptor_{};
  MPI_Comm communicator_ = MPI_COMM_WORLD;
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
  CudaGenericArenaLayout generic_arena_layout_{};
  CudaDeviceView view_{};
  CudaTrueMatrixStructure true_structure_{};
  CudaTrueMatrixDeviceView true_device_{};
  CudaTrueMatrixDeviceView equilibrated_true_device_{};
  CudaNativeTraceWorkspaceRequirements native_trace_requirements_{};
  CudaNativeTraceCooperativeSupport native_trace_support_{};
  CudaNativeTraceWorkspaceView native_trace_workspace_{};
  double* native_ilu_inverse_diagonal_ = nullptr;
  std::array<double, native_trace_host_control_double_capacity>
      native_trace_host_control_{};
  std::array<double, cuda_rank1_equilibration_control_doubles>
      rank_one_equilibration_host_control_{};
  std::vector<double> host_true_diag_values_{};
  std::vector<double> host_true_offd_values_{};
  double* double_arena_ = nullptr;
  int* integer_arena_ = nullptr;
  // False-mode retains the literal W8A capacities and BindArenaViews branch.
  // Borrowed mode receives exact one-time capacities after its actual ParMesh
  // and exchange topology have been constructed.
  std::size_t double_arena_capacity_ = persistent_double_capacity;
  std::size_t integer_arena_capacity_ = persistent_integer_capacity;
  CudaArenaReleaseState arena_release_state_{};
  mfem::Memory<double> double_memory_{};
  mfem::Memory<int> integer_memory_{};
  mfem::Vector double_arena_vector_{};
  mfem::Array<int> integer_arena_array_{};
  mfem::Mesh serial_mesh_{};
  mfem::ParMesh* mesh_ = nullptr;
  std::optional<mfem::ParMesh> owned_mesh_{};
  bool borrowed_topology_ = false;
  std::optional<mfem::L2_FECollection> interior_collection_{};
  std::optional<mfem::DG_Interface_FECollection> trace_collection_{};
  std::optional<mfem::ParFiniteElementSpace> interior_space_{};
  std::optional<mfem::ParFiniteElementSpace> trace_space_{};
  dg::hdg::trace::DGInterfaceTraceExchangePlan exchange_plan_{};
  std::optional<mfem::SparseMatrix> local_trace_matrix_{};
  std::unique_ptr<mfem::HypreParMatrix> true_matrix_{};
  std::unique_ptr<mfem::HypreParMatrix> equilibrated_true_matrix_{};
  std::array<std::optional<mfem::HypreParVector>, 2> trace_slots_{};
  std::optional<mfem::HypreParVector> hypre_rhs_{};
  std::optional<mfem::HypreParVector> hypre_solution_{};
  std::optional<mfem::HypreParVector> hypre_action_{};
  std::optional<mfem::HypreParVector> hypre_equilibrated_rhs_{};
  std::optional<mfem::HypreILU> ilu_{};
  std::optional<mfem::HypreDiagScale> diagonal_{};
  std::optional<mfem::HypreGMRES> gmres_{};
  mfem::Vector equilibration_left_{};
  mfem::Vector equilibration_right_{};
  mfem::Vector equilibration_row_norm_{};
  mfem::Vector equilibration_column_norm_{};
  mfem::Vector equilibration_control_{};
  double* equilibration_left_device_ = nullptr;
  double* equilibration_right_device_ = nullptr;
  double* equilibration_row_norm_device_ = nullptr;
  double* equilibration_column_norm_device_ = nullptr;
  double* equilibration_control_device_ = nullptr;
  double* equilibrated_rhs_device_ = nullptr;
  CudaRankOneMgrState rank_one_mgr_state_{};
  CudaTracePreconditionerKind trace_preconditioner_ =
      CudaTracePreconditionerKind::unconfigured;
  CudaIluAttemptFactorState ilu_factor_state_{};
  CudaElementSchwarzAttemptState element_schwarz_attempt_state_{};
  NativeElementSchwarzPlan element_schwarz_plan_{};
  CudaTensorCoreEvidence initialization_tensor_core_evidence_{};
  mfem::Vector owned_stage_{};
  mfem::Vector local_stage_{};
  state_slot committed_slot_{};
  state_slot candidate_slot_{};
  state_slot workspace_slot_{};
  AttemptIdentity current_attempt_{};
  std::uint64_t current_generation_ = 0;
  std::uint64_t linear_generation_ = 0;
  std::uint64_t baseline_allocations_ = 0;
  std::uint64_t accepted_allocation_events_ = 0;
  std::uint64_t baseline_w8_owned_allocation_calls_ = 0;
  std::uint64_t baseline_deallocation_events_ = 0;
  double local_hmin_ = std::numeric_limits<double>::infinity();
  std::array<double, 2> host_cfl_scalars_{{0.0, 0.0}};
  bool allocation_plateau_sampled_ = false;
  double newton_correction_linear_relative_tolerance_ = 1.0e-13;
  double final_coherent_linear_relative_tolerance_ = 1.0e-13;
  double true_residual_tightening_linear_relative_tolerance_ = 1.0e-13;
  double true_residual_relative_tolerance_ = 1.0e-10;
  int maximum_linear_iterations_ = 400;
  int krylov_dimension_ = 20;
  std::size_t host_exchange_bytes_ = 0;
  std::array<CudaDeviceTicket, maximum_device_tickets> tickets_{};
  std::size_t ticket_count_ = 0;
  CudaOraclePod oracle_{};

  std::vector<int> signed_element_trace_{};
  std::vector<std::int64_t> local_global_true_{};
  std::vector<int> local_owner_{};
  std::vector<int> local_true_index_{};
  std::vector<std::int64_t> owned_global_true_{};
  std::vector<std::int64_t> element_coordinate_keys_{};
  std::vector<double> host_volume_shape_{};
  std::vector<double> host_volume_derivative_{};
  std::vector<double> host_volume_weight_{};
  std::vector<double> host_volume_position_{};
  std::vector<double> host_face_interior_shape_{};
  std::vector<double> host_face_trace_shape_{};
  std::vector<double> host_face_weight_{};
  std::vector<double> host_face_normal_{};
  std::vector<double> host_face_h_{};
  std::vector<double> host_boundary_reference_{};
  std::vector<double> host_boundary_pressure_{};
  std::vector<int> host_boundary_policy_kind_{};
  std::vector<double> host_exact_interior_{};
  std::vector<double> host_perturbed_interior_{};
  std::vector<double> host_exact_true_trace_{};
  std::vector<double> host_perturbed_true_trace_{};
  std::vector<double> host_static_source_{};
  std::vector<double> host_export_interior_{};
  std::vector<int> signed_element_interior_{};
  bool borrowed_host_inputs_ = false;
  std::vector<int> host_matrix_owned_source_{};
  std::vector<int> host_matrix_owned_section_{};
  std::vector<int> host_matrix_owned_target_{};
  std::vector<int> host_matrix_send_source_{};
  std::vector<int> host_matrix_receive_section_{};
  std::vector<int> host_matrix_receive_target_{};
  std::vector<int> host_vector_owned_source_{};
  std::vector<int> host_vector_owned_target_{};
  std::vector<int> host_vector_send_source_{};
  std::vector<int> host_vector_receive_target_{};
  std::vector<std::int64_t> matrix_send_keys_{};
  std::vector<std::int64_t> matrix_receive_keys_{};
  std::vector<std::int64_t> vector_send_keys_{};
  std::vector<std::int64_t> vector_receive_keys_{};
  std::vector<double> matrix_send_host_{};
  std::vector<double> matrix_receive_host_{};
  std::vector<double> vector_send_host_{};
  std::vector<double> vector_receive_host_{};
};

} // namespace detail

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_methods.cuh"
