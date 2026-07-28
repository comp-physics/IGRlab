#pragma once

/**
 * @file numerics/hdg/runtime/app/application_contract.hpp
 * @brief Frozen host-side W8A application selection contract.
 *
 * Validation in this file runs before a backend is entered.  The four backend
 * overloads declared at the end are intentionally not implemented by this
 * contract slice.  Their implementations must drive the existing
 * PersistentRunContract lifecycle; they are not a second run lifecycle.
 */

#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace dg::hdg::runtime::app
{

inline constexpr std::string_view smooth_periodic_euler_v1_case_id =
    conventional_persistent_case_id;
inline constexpr std::string_view smooth_periodic_euler_v1_card_id =
    conventional_persistent_card_id;

inline constexpr int smooth_periodic_euler_v1_polynomial_order = 1;
inline constexpr double smooth_periodic_euler_v1_dt = 0.05;
inline constexpr std::uint64_t smooth_periodic_euler_v1_minimum_steps = 20;
inline constexpr std::array<std::uint64_t, 3>
    smooth_periodic_euler_v1_output_steps{{0, 10, 20}};
inline constexpr std::uint64_t smooth_periodic_euler_v1_checkpoint_step = 10;
inline constexpr std::uint64_t smooth_periodic_euler_v1_failure_step = 14;
inline constexpr std::uint32_t smooth_periodic_euler_v1_retry_limit = 1;

enum class ApplicationBackend : std::uint8_t
{
  cpu_serial = 0,
  cpu_rank2 = 1,
  cuda_rank1_one_gpu = 2,
  cuda_rank2_one_gpu_host_staged = 3,
  unknown = 255
};

enum class ApplicationRejection : std::uint8_t
{
  none = 0,
  unknown_case,
  unknown_tuple,
  unsupported_h1,
  invalid_interior_space,
  unknown_backend,
  unbuilt_backend,
  invalid_polynomial_order,
  invalid_mesh,
  insufficient_step_bound,
  invalid_dt,
  invalid_output_schedule,
  invalid_checkpoint_schedule,
  invalid_restart_mode,
  invalid_failure_injection,
  invalid_retry_limit
};

/** Availability is supplied by the normal build, never inferred at runtime. */
struct BuiltBackendSet
{
  bool cpu_serial = false;
  bool cpu_rank2 = false;
  bool cuda_rank1_one_gpu = false;
  bool cuda_rank2_one_gpu_host_staged = false;
};

/** Non-owning command/configuration view consumed only during validation. */
struct ApplicationOptionsView
{
  std::string_view case_id = smooth_periodic_euler_v1_case_id;
  std::string_view model_id = EulerCompressible2dTag::id;
  std::string_view formulation_id = ConventionalHdgTag::id;
  std::string_view trace_id = MfemDgInterfaceTraceTag::id;
  std::string_view stabilization_id = FixedScalarTauTag::id;
  std::string_view time_treatment_id = BackwardEulerRuntimeTag::id;
  std::string_view backend_id = CpuSerialBackendTag::id;
  std::string_view precision_id = Fp64RuntimeTag::id;
  InteriorSpace interior_space = InteriorSpace::l2;
  int polynomial_order = smooth_periodic_euler_v1_polynomial_order;
  std::uint32_t mesh_n = 6;
  std::uint64_t step_bound = smooth_periodic_euler_v1_minimum_steps;
  double dt = smooth_periodic_euler_v1_dt;
  std::array<std::uint64_t, 3> output_steps =
      smooth_periodic_euler_v1_output_steps;
  std::uint64_t checkpoint_step =
      smooth_periodic_euler_v1_checkpoint_step;
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  std::uint64_t injected_failure_step =
      smooth_periodic_euler_v1_failure_step;
  std::uint32_t retry_limit = smooth_periodic_euler_v1_retry_limit;
};

/** Owned, pointer-free result of successful pre-execution validation. */
struct ValidatedApplicationConfig
{
  ApplicationBackend backend = ApplicationBackend::unknown;
  RuntimeTupleDescriptor tuple{};
  InteriorSpace interior_space = InteriorSpace::l2;
  int polynomial_order = -1;
  std::uint32_t mesh_n = 0;
  std::uint64_t step_bound = 0;
  double dt = 0.0;
  std::array<std::uint64_t, 3> output_steps{};
  std::uint64_t checkpoint_step = 0;
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  std::uint64_t injected_failure_step = 0;
  std::uint32_t retry_limit = 0;
};

struct ApplicationValidation
{
  bool accepted = false;
  ApplicationRejection rejection = ApplicationRejection::unknown_tuple;
  ValidatedApplicationConfig config{};
};

ApplicationBackend ParseApplicationBackend(std::string_view id) noexcept;

bool IsBackendBuilt(ApplicationBackend backend,
                    const BuiltBackendSet& built) noexcept;

bool IsAdmittedMesh(ApplicationBackend backend,
                    std::uint32_t mesh_n) noexcept;

/** N=6 is the common trajectory shape for all four admitted backends. */
constexpr bool IsSharedFourShapeMesh(const std::uint32_t mesh_n) noexcept
{
  return mesh_n == 6;
}

ApplicationValidation ValidateApplicationOptions(
    const ApplicationOptionsView& options,
    const BuiltBackendSet& built) noexcept;

struct ApplicationRunResult
{
  RunStatus status = RunStatus::invalid_request;
  FailureReason reason = FailureReason::invalid_request;
  StateCoordinates coordinates{};
  std::uint32_t retries = 0;
};

template <typename BackendTag>
struct IsFrozenApplicationBackend : std::false_type
{};

template <>
struct IsFrozenApplicationBackend<CpuSerialBackendTag> : std::true_type
{};
template <>
struct IsFrozenApplicationBackend<CpuRank2BackendTag> : std::true_type
{};
template <>
struct IsFrozenApplicationBackend<CudaRank1OneGpuBackendTag>
    : std::true_type
{};
template <>
struct IsFrozenApplicationBackend<CudaRank2OneGpuHostStagedBackendTag>
    : std::true_type
{};

// Closed static entry declarations.  A future backend slice defines these
// overloads by composing its existing PersistentRunContract adapter.
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuSerialBackendTag,
    const ValidatedApplicationConfig& config);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuRank2BackendTag,
    const ValidatedApplicationConfig& config);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank1OneGpuBackendTag,
    const ValidatedApplicationConfig& config);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank2OneGpuHostStagedBackendTag,
    const ValidatedApplicationConfig& config);

static_assert(!std::is_polymorphic<ValidatedApplicationConfig>::value,
              "application configuration must remain statically dispatched");
static_assert(std::is_trivially_copyable<ValidatedApplicationConfig>::value,
              "validated application configuration owns no pointers");

} // namespace dg::hdg::runtime::app
