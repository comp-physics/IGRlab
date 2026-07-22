#pragma once

/**
 * @file utils/profiling/observability.hpp
 * @brief Allocation-free, compile-time observability contract.
 *
 * Observability is compiled out unless DG_IGR_ENABLE_OBSERVABILITY is 1.
 * Compiled instrumentation is still inactive until set_runtime_enabled(true)
 * is called during serial program setup. The runtime gate is deliberately
 * non-atomic: configure it before worker threads are launched and do not
 * change it while instrumented work is running.
 */

#include <array>
#include <cstddef>
#include <cstdint>

#ifndef DG_IGR_ENABLE_OBSERVABILITY
#define DG_IGR_ENABLE_OBSERVABILITY 0
#endif

#ifndef DG_IGR_OBSERVABILITY_USE_NVTX
#define DG_IGR_OBSERVABILITY_USE_NVTX 0
#endif

#ifndef DG_IGR_OBSERVABILITY_USE_CALIPER
#define DG_IGR_OBSERVABILITY_USE_CALIPER 0
#endif

#if (DG_IGR_OBSERVABILITY_USE_NVTX || DG_IGR_OBSERVABILITY_USE_CALIPER) && \
    !DG_IGR_ENABLE_OBSERVABILITY
#error "An observability backend requires DG_IGR_ENABLE_OBSERVABILITY"
#endif

#if DG_IGR_OBSERVABILITY_USE_NVTX && DG_IGR_OBSERVABILITY_USE_CALIPER
#error "DG_IGR_OBSERVABILITY_USE_NVTX and DG_IGR_OBSERVABILITY_USE_CALIPER are mutually exclusive"
#endif

#if DG_IGR_ENABLE_OBSERVABILITY && DG_IGR_OBSERVABILITY_USE_NVTX
#if defined(__has_include)
#if !__has_include(<nvtx3/nvToolsExt.h>)
#error "Explicit NVTX observability requested, but nvtx3/nvToolsExt.h is unavailable"
#endif
#endif
#include <nvtx3/nvToolsExt.h>
#endif

#if DG_IGR_ENABLE_OBSERVABILITY && DG_IGR_OBSERVABILITY_USE_CALIPER
#if defined(__has_include)
#if !__has_include(<caliper/cali.h>)
#error "Explicit Caliper observability requested, but caliper/cali.h is unavailable"
#endif
#endif
#include <caliper/cali.h>
#endif

namespace dg_igr
{
namespace observability
{

inline constexpr bool compile_enabled = DG_IGR_ENABLE_OBSERVABILITY != 0;
inline constexpr bool nvtx_backend_enabled =
    DG_IGR_ENABLE_OBSERVABILITY != 0 &&
    DG_IGR_OBSERVABILITY_USE_NVTX != 0;
inline constexpr bool caliper_backend_enabled =
    DG_IGR_ENABLE_OBSERVABILITY != 0 &&
    DG_IGR_OBSERVABILITY_USE_CALIPER != 0;

/** Stable region IDs. Existing string names remain supported by NVTXRange. */
enum class RegionId : std::uint16_t
{
  unknown = 0,
  dg_mult = 100,
  dg_prolong_true_to_local = 101,
  dg_face_domain_mfem_nonlinearform = 102,
  dg_face_packed_euler_rusanov = 103,
  dg_face_domain_cpu_nonlinearform = 104,
  dg_weak_divergence = 105,
  dg_inverse_mass_local_and_restrict = 106,
  dg_inverse_mass = 107,
  limiter_zhang_shu = 150,
  igr_update_from_state = 200,
  igr_krylov_operator_mult = 201,
  igr_krylov_operator_mult_transpose = 202,
  igr_krylov_preconditioner_mult = 203,
  igr_krylov_preconditioner_mult_transpose = 204,
  igr_exact_face_mult = 210,
  igr_exact_face_prolong = 211,
  igr_exact_face_local_apply = 212,
  igr_exact_face_shared_exchange = 213,
  igr_exact_face_shared_apply = 214,
  igr_exact_face_restrict = 215,
  igr_exact_sum_mult = 220,
  igr_exact_sum_base = 221,
  igr_exact_sum_face = 222,
  igr_exact_sum_axpy = 223,
  igr_assemble_system = 230,
  igr_state_face_neighbor_exchange = 231,
  igr_device_coefficients = 232,
  igr_host_coefficients = 233,
  igr_mfem_form_assemble = 234,
  igr_form_linear_system = 235,
  igr_exact_face_setup = 236,
  igr_smoother_sweeps = 240,
  igr_solve_elliptic_system = 241,
  igr_preconditioner_setup = 242,
  igr_chebyshev_setup = 243,
  igr_chebyshev_diagonal = 244,
  igr_warm_start_residual_guard = 245,
  igr_krylov_solve = 246,
  hdg_local_assembly = 300,
  hdg_local_factorization = 301,
  hdg_condensation = 302,
  hdg_schur_action = 303,
  hdg_trace_assembly = 304,
  hdg_interior_recovery = 305,
  hdg_packing = 306,
  hdg_device_transfer = 307
};

/** Stable counter IDs reserved for shared core performance measurements. */
enum class CounterId : std::uint16_t
{
  iterations = 100,
  degrees_of_freedom = 101,
  elements = 102,
  operator_applications = 103,
  bytes = 104
};

constexpr const char* region_name(const RegionId id) noexcept
{
  switch (id)
  {
    case RegionId::dg_mult: return "DG/Mult";
    case RegionId::dg_prolong_true_to_local: return "DG/prolong-true-to-local";
    case RegionId::dg_face_domain_mfem_nonlinearform: return "DG/face-domain-mfem-nonlinearform";
    case RegionId::dg_face_packed_euler_rusanov: return "DG/face-packed-euler-rusanov";
    case RegionId::dg_face_domain_cpu_nonlinearform: return "DG/face-domain-cpu-nonlinearform";
    case RegionId::dg_weak_divergence: return "DG/weak-divergence";
    case RegionId::dg_inverse_mass_local_and_restrict: return "DG/inverse-mass-local-and-restrict";
    case RegionId::dg_inverse_mass: return "DG/inverse-mass";
    case RegionId::limiter_zhang_shu: return "Limiter/Zhang-Shu";
    case RegionId::igr_update_from_state: return "IGR/update-from-state";
    case RegionId::igr_krylov_operator_mult: return "IGR/krylov-operator-mult";
    case RegionId::igr_krylov_operator_mult_transpose: return "IGR/krylov-operator-mult-transpose";
    case RegionId::igr_krylov_preconditioner_mult: return "IGR/krylov-preconditioner-mult";
    case RegionId::igr_krylov_preconditioner_mult_transpose: return "IGR/krylov-preconditioner-mult-transpose";
    case RegionId::igr_exact_face_mult: return "IGR/exact-face/Mult";
    case RegionId::igr_exact_face_prolong: return "IGR/exact-face/prolong";
    case RegionId::igr_exact_face_local_apply: return "IGR/exact-face/local-apply";
    case RegionId::igr_exact_face_shared_exchange: return "IGR/exact-face/shared-exchange";
    case RegionId::igr_exact_face_shared_apply: return "IGR/exact-face/shared-apply";
    case RegionId::igr_exact_face_restrict: return "IGR/exact-face/restrict";
    case RegionId::igr_exact_sum_mult: return "IGR/exact-sum/Mult";
    case RegionId::igr_exact_sum_base: return "IGR/exact-sum/base";
    case RegionId::igr_exact_sum_face: return "IGR/exact-sum/face";
    case RegionId::igr_exact_sum_axpy: return "IGR/exact-sum/axpy";
    case RegionId::igr_assemble_system: return "IGR/assemble-system";
    case RegionId::igr_state_face_neighbor_exchange: return "IGR/state-face-neighbor-exchange";
    case RegionId::igr_device_coefficients: return "IGR/device-coefficients";
    case RegionId::igr_host_coefficients: return "IGR/host-coefficients";
    case RegionId::igr_mfem_form_assemble: return "IGR/mfem-form-assemble";
    case RegionId::igr_form_linear_system: return "IGR/form-linear-system";
    case RegionId::igr_exact_face_setup: return "IGR/exact-face-setup";
    case RegionId::igr_smoother_sweeps: return "IGR/smoother-sweeps";
    case RegionId::igr_solve_elliptic_system: return "IGR/solve-elliptic-system";
    case RegionId::igr_preconditioner_setup: return "IGR/preconditioner-setup";
    case RegionId::igr_chebyshev_setup: return "IGR/chebyshev-setup";
    case RegionId::igr_chebyshev_diagonal: return "IGR/chebyshev-diagonal";
    case RegionId::igr_warm_start_residual_guard: return "IGR/warm-start-residual-guard";
    case RegionId::igr_krylov_solve: return "IGR/krylov-solve";
    case RegionId::hdg_local_assembly: return "HDG/local-assembly";
    case RegionId::hdg_local_factorization: return "HDG/local-factorization";
    case RegionId::hdg_condensation: return "HDG/condensation";
    case RegionId::hdg_schur_action: return "HDG/schur-action";
    case RegionId::hdg_trace_assembly: return "HDG/trace-assembly";
    case RegionId::hdg_interior_recovery: return "HDG/interior-recovery";
    case RegionId::hdg_packing: return "HDG/packing";
    case RegionId::hdg_device_transfer: return "HDG/device-transfer";
    case RegionId::unknown: return "core/unknown";
  }
  return "core/unknown";
}

constexpr const char* counter_name(const CounterId id) noexcept
{
  switch (id)
  {
    case CounterId::iterations: return "core/iterations";
    case CounterId::degrees_of_freedom: return "core/degrees-of-freedom";
    case CounterId::elements: return "core/elements";
    case CounterId::operator_applications: return "core/operator-applications";
    case CounterId::bytes: return "core/bytes";
  }
  return "core/unknown-counter";
}

struct RegionDescriptor
{
  RegionId id;
  const char* name;
};

constexpr RegionDescriptor region_descriptor(const RegionId id) noexcept
{
  return {id, region_name(id)};
}

struct NoopBackend
{
  static constexpr void begin(RegionDescriptor) noexcept {}
  static constexpr void end(RegionDescriptor) noexcept {}
  static constexpr void counter(CounterId, std::int64_t) noexcept {}
};

#if DG_IGR_ENABLE_OBSERVABILITY && DG_IGR_OBSERVABILITY_USE_CALIPER
struct CaliperBackend
{
  // Initialize the fixed counter attributes during serial runtime-gate setup,
  // so the adapter adds no attribute creation to hot counter calls.
  static void initialize() noexcept
  {
    if (initialized_)
    {
      return;
    }

    counter_attributes_ = {
        cali_create_attribute(counter_name(CounterId::iterations), CALI_TYPE_INT,
                              CALI_ATTR_ASVALUE),
        cali_create_attribute(counter_name(CounterId::degrees_of_freedom),
                              CALI_TYPE_INT, CALI_ATTR_ASVALUE),
        cali_create_attribute(counter_name(CounterId::elements), CALI_TYPE_INT,
                              CALI_ATTR_ASVALUE),
        cali_create_attribute(counter_name(CounterId::operator_applications),
                              CALI_TYPE_INT, CALI_ATTR_ASVALUE),
        cali_create_attribute(counter_name(CounterId::bytes), CALI_TYPE_INT,
                              CALI_ATTR_ASVALUE)};
    initialized_ = true;
  }

  static void begin(const RegionDescriptor region) noexcept
  {
    cali_begin_region(region.name);
  }

  static void end(const RegionDescriptor region) noexcept
  {
    cali_end_region(region.name);
  }

  static void counter(const CounterId id, const std::int64_t value) noexcept
  {
    if (!initialized_)
    {
      return;
    }

    switch (id)
    {
      case CounterId::iterations:
        cali_set(counter_attributes_[0], &value, sizeof(value));
        break;
      case CounterId::degrees_of_freedom:
        cali_set(counter_attributes_[1], &value, sizeof(value));
        break;
      case CounterId::elements:
        cali_set(counter_attributes_[2], &value, sizeof(value));
        break;
      case CounterId::operator_applications:
        cali_set(counter_attributes_[3], &value, sizeof(value));
        break;
      case CounterId::bytes:
        cali_set(counter_attributes_[4], &value, sizeof(value));
        break;
    }
  }

private:
  inline static std::array<cali_id_t, 5> counter_attributes_{
      CALI_INV_ID, CALI_INV_ID, CALI_INV_ID, CALI_INV_ID, CALI_INV_ID};
  inline static bool initialized_ = false;
};
#endif

#if DG_IGR_ENABLE_OBSERVABILITY
inline bool& runtime_enabled_storage() noexcept
{
  static bool enabled = false;
  return enabled;
}

inline void set_runtime_enabled(const bool enabled) noexcept
{
#if DG_IGR_OBSERVABILITY_USE_CALIPER
  if (enabled)
  {
    CaliperBackend::initialize();
  }
#endif
  runtime_enabled_storage() = enabled;
}

inline bool runtime_enabled() noexcept
{
  return runtime_enabled_storage();
}
#else
inline constexpr void set_runtime_enabled(bool) noexcept {}
inline constexpr bool runtime_enabled() noexcept { return false; }
#endif

enum class RecordedEventKind : std::uint8_t
{
  region_begin,
  region_end,
  counter
};

struct RecordedEvent
{
  RecordedEventKind kind;
  std::uint16_t id;
  std::int64_t value;
  const char* name;
};

/** Fixed-capacity, unsynchronized backend for focused synthetic tests. */
template <std::size_t Capacity>
struct StaticRecordingBackend
{
  static_assert(Capacity > 0, "recording capacity must be positive");

  static void reset() noexcept
  {
    size_ = 0;
    dropped_ = 0;
  }

  static std::size_t size() noexcept { return size_; }
  static std::size_t dropped() noexcept { return dropped_; }
  static const RecordedEvent& event(const std::size_t index) noexcept
  {
    return events_[index];
  }

  static void begin(const RegionDescriptor region) noexcept
  {
    append({RecordedEventKind::region_begin,
            static_cast<std::uint16_t>(region.id), 0, region.name});
  }

  static void end(const RegionDescriptor region) noexcept
  {
    append({RecordedEventKind::region_end,
            static_cast<std::uint16_t>(region.id), 0, region.name});
  }

  static void counter(const CounterId id, const std::int64_t value) noexcept
  {
    append({RecordedEventKind::counter, static_cast<std::uint16_t>(id),
            value, counter_name(id)});
  }

private:
  static void append(const RecordedEvent event) noexcept
  {
    if (size_ < Capacity)
    {
      events_[size_++] = event;
    }
    else
    {
      ++dropped_;
    }
  }

  inline static std::array<RecordedEvent, Capacity> events_{};
  inline static std::size_t size_ = 0;
  inline static std::size_t dropped_ = 0;
};

#if DG_IGR_ENABLE_OBSERVABILITY && DG_IGR_OBSERVABILITY_USE_NVTX
struct NvtxBackend
{
  static void begin(const RegionDescriptor region) noexcept
  {
    (void)nvtxRangePushA(region.name);
  }

  static void end(RegionDescriptor) noexcept
  {
    (void)nvtxRangePop();
  }

  // NVTX ranges are supported here; counters remain available to later
  // adapters through the same static backend boundary.
  static constexpr void counter(CounterId, std::int64_t) noexcept {}
};

using DefaultBackend = NvtxBackend;
#elif DG_IGR_ENABLE_OBSERVABILITY && DG_IGR_OBSERVABILITY_USE_CALIPER
using DefaultBackend = CaliperBackend;
#else
using DefaultBackend = NoopBackend;
#endif

template <typename Backend>
class ScopedRegion
{
public:
  explicit ScopedRegion(const RegionId id) noexcept
      : ScopedRegion(region_descriptor(id))
  {
  }

  // Compatibility path for the existing NVTXRange("literal") call sites.
  // Callers must provide a name with static storage duration.
  explicit ScopedRegion(const char* name) noexcept
      : ScopedRegion({RegionId::unknown,
                      name == nullptr ? region_name(RegionId::unknown) : name})
  {
  }

  ScopedRegion(const ScopedRegion&) = delete;
  ScopedRegion& operator=(const ScopedRegion&) = delete;

  ~ScopedRegion() noexcept
  {
    if (active_)
    {
      Backend::end(region_);
    }
  }

private:
  explicit ScopedRegion(const RegionDescriptor region) noexcept
      : region_(region), active_(runtime_enabled())
  {
    if (active_)
    {
      Backend::begin(region_);
    }
  }

  RegionDescriptor region_;
  bool active_;
};

template <>
class ScopedRegion<NoopBackend>
{
public:
  explicit constexpr ScopedRegion(RegionId) noexcept {}
  explicit constexpr ScopedRegion(const char*) noexcept {}

  ScopedRegion(const ScopedRegion&) = delete;
  ScopedRegion& operator=(const ScopedRegion&) = delete;
};

template <typename Backend>
struct CounterEmitter
{
  static void emit(const CounterId id, const std::int64_t value) noexcept
  {
    if (runtime_enabled())
    {
      Backend::counter(id, value);
    }
  }
};

template <>
struct CounterEmitter<NoopBackend>
{
  static constexpr void emit(CounterId, std::int64_t) noexcept {}
};

template <typename Backend = DefaultBackend>
inline void record_counter(const CounterId id, const std::int64_t value) noexcept
{
  CounterEmitter<Backend>::emit(id, value);
}

} // namespace observability
} // namespace dg_igr
