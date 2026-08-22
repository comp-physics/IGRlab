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
// device execution, and zero post-policy HYPRE error.  Materialized MFEM and
// HYPRE Umpire pools are admitted only after their first real allocations.
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
  backend_hypre_policy_error_failure = std::uint64_t{1} << 15U
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

struct CudaTrueMatrixDeviceView
{
  double* diag_values = nullptr;
  double* offd_values = nullptr;
  int diag_nonzeros = 0;
  int offd_nonzeros = 0;
};

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
  // One frozen local LLF speed for every element/face/quadrature point.  This
  // slice is populated once from the committed state at Prepare and is never
  // differentiated or updated during the nonlinear attempt.
  double* borrowed_lagged_llf_tau = nullptr;
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
  int assembly_order = 0;
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
bool CudaConfigurePersistentHypre(mfem::HypreParMatrix& matrix,
                                  mfem::HypreILU& ilu,
                                  mfem::HypreGMRES& gmres,
                                  int ranks,
                                  double relative_tolerance,
                                  int maximum_iterations,
                                  int krylov_dimension,
                                  std::uint64_t matrix_generation,
                                  CudaHypreSetupOrderEvidence& evidence)
    noexcept;
bool CudaRefreshFactorAndSolve(mfem::HypreParMatrix& matrix,
                               mfem::HypreParVector& rhs,
                               mfem::HypreParVector& solution,
                               mfem::HypreILU& ilu,
                               mfem::HypreGMRES& gmres,
                               int ranks,
                               MPI_Comm communicator,
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
bool CudaLaunchGatherLocalTrace(const int* signed_map, int count,
                                const double* true_trace,
                                double* local_trace) noexcept;
bool CudaLaunchAdmissibility(const CudaDeviceView& view,
                             const CudaMeshShape& shape) noexcept;
bool CudaLaunchSeedBorrowedLaggedLlfTau(const CudaDeviceView& view,
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
bool CudaLaunchMaximumWave(const CudaDeviceView& view,
                           const CudaMeshShape& shape,
                           const double* interior,
                           const double* element_trace,
                           double* result) noexcept;
bool CudaLaunchIgrConservation(const CudaDeviceView& view,
                               const CudaMeshShape& shape,
                               double dt, double* result) noexcept;

/** Fixed-size input used only by the all-closure CUDA qualification seam. */
struct CudaStaticBoundaryClosureCaseForTests
{
  int kind = 0;
  double reference[conserved_components]{};
  double pressure_out = 0.0;
  double gamma = 1.4;
  double interior[conserved_components]{};
  double normal[dimension]{};
};

bool CudaLaunchStaticBoundaryClosureForTests(
    const CudaStaticBoundaryClosureCaseForTests& input,
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
  ~CudaPersistentNumerics() noexcept
  {
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

  bool Initialize(const CudaConfig& config, CudaRunEvidence& evidence,
                  const RuntimeTupleDescriptor& tuple) noexcept
  {
    communicator_ = CudaCollectiveCommunicator(config);
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
    linear_relative_tolerance_ = config.use_generic_numerical_policy
        ? config.numerical_policy.linear_relative_tolerance : 1.0e-13;
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
    if (!CollectiveContinue(linear_relative_tolerance_ > 0.0 &&
                            std::isfinite(linear_relative_tolerance_) &&
                            true_residual_relative_tolerance_ > 0.0 &&
                            std::isfinite(true_residual_relative_tolerance_) &&
                            maximum_linear_iterations_ > 0 &&
                            krylov_dimension_ > 0))
    {
      ReportCudaSetupFailure(evidence.rank, "generic-linear-policy");
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
        view_.current_local_trace, view_.scalar) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
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
    int local_value = local ? 1 : 0;
    int global_value = 0;
    return MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                         communicator_) == MPI_SUCCESS && global_value == 1;
  }

  bool CollectiveCount(const bool local, int& count) const noexcept
  {
    const int local_value = local ? 1 : 0;
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
    value.factor_token = reinterpret_cast<std::uintptr_t>(
        ilu_ ? static_cast<HYPRE_Solver>(*ilu_) : nullptr);
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
    value.persistent_graph = true_matrix_ && local_trace_matrix_;
    value.persistent_slots = committed_slot_.interior.coefficients !=
            candidate_slot_.interior.coefficients &&
        committed_slot_.trace.coefficients != candidate_slot_.trace.coefficients;
    value.persistent_workspace = view_.a != nullptr && view_.schur != nullptr;
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
    // Borrowed mode must seed its face LLF speeds from one coherent committed
    // state.  On the initial attempt materialize both committed slots before
    // copying old_interior; the frozen legacy route below remains unchanged.
    bool local_ready = true;
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
      local_ready = interior_ready && trace_ready && old_ready;
      workspace.interior.generation = current_generation_;
      workspace.trace.generation = current_generation_;
    }
    else
    {
      local_ready = TransportPolicy::DeviceToDevice(
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
    const bool synchronized =
        TransportPolicy::Synchronize(CudaTransferSite::attempt_seed_trace);
    local_ready = local_ready && synchronized;
    if (!CollectiveContinue(local_ready)) { return false; }
    return DistributeCurrentTrace(workspace.trace.coefficients);
  }

  /**
   * Test-only central-difference audit of the production device assembler.
   * It stages no output and publishes no slot; the reusable workspace is
   * restored to the unperturbed test linearization before returning.
   */
  bool VerifyBorrowedEulerAssemblerJacobianForTests(
      const double relative_tolerance,
      std::array<double, 6>& relative_errors) noexcept
  {
    relative_errors.fill(std::numeric_limits<double>::infinity());
    const bool admitted = initialized_ && shape_.borrowed_host_inputs &&
        shape_.igr_active && shape_.interior_components ==
            igr_interior_components &&
        shape_.trace_components == igr_trace_components &&
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
    for (int scalar = 0; local_valid && scalar < true_trace_scalar; ++scalar)
      base_true_trace[static_cast<std::size_t>(4 * true_trace_scalar + scalar)] =
          0.02 * deterministic_unit(static_cast<std::size_t>(scalar), 31U);
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
                              std::vector<double>* residual_trace) noexcept
    {
      if (!install_state(interior, trace)) return false;
      TransportPolicy::SetPhase(CudaFailurePhase::assembly,
                                current_generation_);
      const bool assembled = CudaLaunchAssembly(
              view_, shape_, fixed_dt) &&
          TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
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
        !assemble(base_interior, base_true_trace, nullptr, nullptr))
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
          !assemble(plus_interior, plus_trace, &plus_r, &plus_g) ||
          !assemble(minus_interior, minus_trace, &minus_r, &minus_g))
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
      return CollectiveContinue(std::isfinite(error));
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

  /** Verify the caller-supplied public trace was copied into owned storage. */
  bool VerifyBorrowedInitialTrueTraceForTests(
      CudaBorrowedHostL2Input expected) noexcept;

  /** Direct host/device parity audit of all seven admitted static closures. */
  bool VerifyStaticBoundaryClosuresForTests(
      double relative_tolerance,
      std::array<double, 7>& relative_errors) noexcept;

  bool IgrConservationBalance(const double dt, double& maximum_error,
                              double& maximum_relative_error) noexcept
  {
    maximum_error = 0.0;
    maximum_relative_error = 0.0;
    constexpr int components = 5;
    constexpr int result_scalars = 2 * components + 1;
    std::array<double, result_scalars> local{};
    const bool launched = initialized_ && shape_.igr_active &&
        std::isfinite(dt) && dt > 0.0 &&
        CudaLaunchIgrConservation(view_, shape_, dt, view_.scalar) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!CollectiveContinue(launched)) return false;
    const bool copied = TransportPolicy::DeviceToHost(
            local.data(), view_.scalar, local.size() * sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    bool local_valid = copied && local[2 * components] == 1.0;
    for (int index = 0; local_valid && index < 2 * components; ++index)
      local_valid = std::isfinite(local[static_cast<std::size_t>(index)]);
    if (!CollectiveContinue(local_valid)) return false;
    std::array<double, components> global_balance{};
    std::array<double, components> global_scale{};
    const bool balance_reduced = MPI_Allreduce(
        local.data(), global_balance.data(), components, MPI_DOUBLE,
        MPI_SUM, communicator_) == MPI_SUCCESS;
    const bool scale_reduced = MPI_Allreduce(
        local.data() + components, global_scale.data(), components,
        MPI_DOUBLE, MPI_SUM, communicator_) == MPI_SUCCESS;
    if (!CollectiveContinue(balance_reduced && scale_reduced)) return false;
    for (int component = 0; component < components; ++component)
    {
      const double error = std::abs(
          global_balance[static_cast<std::size_t>(component)]);
      const double relative = error /
          (1.0 + global_scale[static_cast<std::size_t>(component)]);
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
    CudaNumericalResult result{};
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
    // The borrowed path owns a single stabilization field per prepared
    // attempt.  Seed it from the committed interior/local trace before any
    // nonlinear update, then retain it unchanged through every assembly and
    // line-search reassembly.  The legacy manufactured W8A path deliberately
    // does not launch this kernel and continues to use its fixed arithmetic.
    if (shape_.borrowed_host_inputs && !shape_.igr_active)
    {
      const bool tau_seeded = CudaLaunchSeedBorrowedLaggedLlfTau(view_, shape_) &&
          TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
          StatusPassed();
      if (!CollectiveContinue(tau_seeded))
      {
        result.phase = CudaFailurePhase::assembly;
        result.reason = FailureReason::admissibility_gate;
        return result;
      }
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
      const bool solve_ok =
          SolveTrueSystem(result.true_relative_residual, generation);
      if (!CollectiveContinue(solve_ok))
      {
        result.reason = FailureReason::true_residual_gate;
        return result;
      }
      ++result.trace_solves;
      const bool trace_distributed = DistributeTraceIncrement();
      if (inject_at(CudaFailurePhase::recovery)) { return result; }
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
      for (int trial = 0; trial < maximum_line_search_trials; ++trial)
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
    const bool local_admissible = CudaLaunchAdmissibility(view_, shape_) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
        StatusPassed();
    result.admissible = CollectiveContinue(local_admissible);
    result.true_residual = std::isfinite(result.true_relative_residual) &&
        result.true_relative_residual <= true_residual_relative_tolerance_;
    if (config.force_private_scientific_gate_failure)
      result.true_residual = false;
    if (inject_at(CudaFailurePhase::diagnostics)) { return result; }
    if (shape_.igr_active)
    {
      result.conservation = IgrConservationBalance(
          attempt.dt, result.maximum_conservation_error,
          result.maximum_conservation_relative_error) &&
          result.maximum_conservation_relative_error <=
              config.numerical_policy.conservation_relative_tolerance;
      if (!CollectiveContinue(result.conservation))
      {
        result.reason = FailureReason::success_gate;
        return result;
      }
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
    result.telemetry = DeviceResident() && NoFallback() &&
        AllocatorPolicy::W8OwnedAllocationCalls() ==
            baseline_w8_owned_allocation_calls_ &&
        AllocatorPolicy::DeallocationEvents() ==
            baseline_deallocation_events_ &&
        allocation_plateau_exact;
    result.success = result.finite && result.admissible && result.recovery &&
        result.true_residual && result.conservation && result.telemetry;
    result.reason = result.success ? FailureReason::none :
        (!result.true_residual ? FailureReason::true_residual_gate
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

private:
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

  static int AssemblyOrder(const CudaConfig& config) noexcept
  {
    if (config.use_generic_numerical_policy)
      return IgrActive(config)
          ? config.numerical_policy.igr_assembly_quadrature_order
          : config.numerical_policy.conventional_assembly_quadrature_order;
    return config.order == functional_polynomial_order
        ? functional_assembly_quadrature_order : assembly_quadrature_order;
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
    const int assembly_order = AssemblyOrder(config);
    const int volume_points = mfem::IntRules.Get(
        fe->GetGeomType(), assembly_order).GetNPoints();
    const int face_points = mfem::IntRules.Get(
        face_fe->GetGeomType(), assembly_order).GetNPoints();
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
    bool local_valid = supported_rank && supported_order && communicator_ready &&
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
    shape_.assembly_order = AssemblyOrder(config);
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
        fe->GetGeomType(), shape_.assembly_order).GetNPoints();
    shape_.face_points = mfem::IntRules.Get(
        face_fe->GetGeomType(), shape_.assembly_order).GetNPoints();
    if (shape_.volume_points > maximum_volume_points ||
        shape_.face_points > maximum_face_points)
      return false;
    return AllocateTopologyStorage() && BuildHostGeometryAndMaps(config) &&
        BuildLocalTraceGraph();
  }

  bool BuildHostGeometryAndMaps(const CudaConfig&) noexcept;
  bool AllocateTopologyStorage() noexcept
  {
    try
    {
      const bool generic = borrowed_topology_ && shape_.borrowed_host_inputs;
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
      layout.face_h = element_faces;
      valid = valid && checked_product(element_faces, conserved_components,
                                       layout.boundary_reference);
      layout.boundary_pressure = element_faces;
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
      host_face_h_.assign(generic ? layout.face_h : 0, 1.0);
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
      host_export_interior_.assign(host_interior, 0.0);
      host_status_.assign(host_elements, 0.0);
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
          checked_add(doubles, layout.face_h) &&
          checked_add(doubles, elements) && checked_add(doubles, 11) &&
          checked_add(doubles, layout.face_weight) &&
          (!shape_.borrowed_static_boundary_policies ||
           (checked_add(doubles, layout.boundary_reference) &&
            checked_add(doubles, layout.boundary_pressure)));
      std::size_t integers = 16;
      valid = valid && checked_add(integers, layout.element_trace) &&
          checked_add(integers, three_local_matrix) &&
          checked_add(integers, three_local_matrix) &&
          checked_add(integers, four_element_trace) &&
          (!shape_.borrowed_static_boundary_policies ||
           checked_add(integers, layout.element_faces));
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
                          double* true_values) noexcept;
  bool AssembleTrueSystem() noexcept;
  bool SolveTrueSystem(double& true_relative_residual,
                       std::uint64_t matrix_generation) noexcept;
  bool AssembleAndNorm(state_slot& workspace, double dt,
                       double& norm) noexcept;
  bool CopyBase(state_slot& workspace) noexcept;
  bool RestoreBase(state_slot& workspace) noexcept;
  bool SwapTrialIntoWorkspace(state_slot& workspace) noexcept;
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
  bool oracle_captured_ = false;
  bool oracle_capture_requested_ = false;
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
  std::uint64_t accepted_allocation_events_ = 0;
  std::uint64_t baseline_w8_owned_allocation_calls_ = 0;
  std::uint64_t baseline_deallocation_events_ = 0;
  double local_hmin_ = std::numeric_limits<double>::infinity();
  std::array<double, 2> host_cfl_scalars_{{0.0, 0.0}};
  bool allocation_plateau_sampled_ = false;
  double linear_relative_tolerance_ = 1.0e-13;
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
  std::vector<double> host_status_{};
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
