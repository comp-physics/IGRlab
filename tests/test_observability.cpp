#include "utils/profiling/nvtx.hpp"

#include <cstdint>
#include <type_traits>

namespace obs = dg_igr::observability;

static_assert(static_cast<std::uint16_t>(obs::RegionId::dg_mult) == 100);
static_assert(static_cast<std::uint16_t>(obs::RegionId::igr_krylov_solve) == 246);
static_assert(static_cast<std::uint16_t>(obs::CounterId::iterations) == 100);
static_assert(obs::region_name(obs::RegionId::dg_mult)[0] == 'D');
static_assert(obs::counter_name(obs::CounterId::iterations)[0] == 'c');
static_assert(!std::is_polymorphic<obs::NoopBackend>::value);

#if DG_IGR_ENABLE_OBSERVABILITY && !DG_IGR_OBSERVABILITY_USE_CALIPER

using Recorder = obs::StaticRecordingBackend<8>;

static int check_recording_backend()
{
  Recorder::reset();
  obs::set_runtime_enabled(false);
  {
    obs::ScopedRegion<Recorder> disabled(obs::RegionId::dg_mult);
    obs::record_counter<Recorder>(obs::CounterId::iterations, 3);
  }
  if (Recorder::size() != 0)
  {
    return 1;
  }

  obs::set_runtime_enabled(true);
  {
    obs::ScopedRegion<Recorder> region(obs::RegionId::dg_mult);
    obs::record_counter<Recorder>(obs::CounterId::iterations, 7);
  }
  {
    obs::ScopedRegion<Recorder> compatibility("DG/Mult");
  }
  obs::set_runtime_enabled(false);

  if (Recorder::size() != 5 || Recorder::dropped() != 0)
  {
    return 2;
  }

  const auto& begin = Recorder::event(0);
  const auto& counter = Recorder::event(1);
  const auto& end = Recorder::event(2);
  const auto& compatibility = Recorder::event(3);
  if (begin.kind != obs::RecordedEventKind::region_begin ||
      begin.id != static_cast<std::uint16_t>(obs::RegionId::dg_mult) ||
      begin.name != obs::region_name(obs::RegionId::dg_mult))
  {
    return 3;
  }
  if (counter.kind != obs::RecordedEventKind::counter ||
      counter.id != static_cast<std::uint16_t>(obs::CounterId::iterations) ||
      counter.value != 7 ||
      counter.name != obs::counter_name(obs::CounterId::iterations))
  {
    return 4;
  }
  if (end.kind != obs::RecordedEventKind::region_end ||
      end.id != begin.id || end.name != begin.name)
  {
    return 5;
  }
  if (compatibility.kind != obs::RecordedEventKind::region_begin ||
      compatibility.id != static_cast<std::uint16_t>(obs::RegionId::unknown) ||
      compatibility.name[0] != 'D')
  {
    return 6;
  }
  return 0;
}

#else

#if DG_IGR_ENABLE_OBSERVABILITY
static_assert(obs::compile_enabled);
static_assert(obs::caliper_backend_enabled);
static_assert(!obs::nvtx_backend_enabled);
static_assert(std::is_same<obs::DefaultBackend, obs::CaliperBackend>::value);
static_assert(!std::is_polymorphic<obs::CaliperBackend>::value);

static int check_caliper_adapter()
{
  // A Caliper configuration is deliberately not required: this exercises the
  // enabled adapter's attribute setup, region enter/exit, and counter path.
  obs::set_runtime_enabled(true);
  {
    dg_igr::NVTXRange region(obs::RegionId::dg_mult);
    obs::record_counter(obs::CounterId::iterations, std::int64_t{1} << 40);
  }
  obs::set_runtime_enabled(false);
  return obs::runtime_enabled() ? 1 : 0;
}

#else
static_assert(!obs::compile_enabled);
static_assert(!obs::nvtx_backend_enabled);
static_assert(!obs::caliper_backend_enabled);
static_assert(std::is_same<obs::DefaultBackend, obs::NoopBackend>::value);
static_assert(std::is_empty<dg_igr::NVTXRange>::value);
static_assert(std::is_trivially_destructible<dg_igr::NVTXRange>::value);
static_assert(noexcept(dg_igr::NVTXRange("DG/Mult")));

static int disabled_probe(const int input) noexcept
{
  dg_igr::NVTXRange range("DG/Mult");
  obs::record_counter(obs::CounterId::iterations, input);
  return input + 1;
}

#endif

#endif

int main()
{
  // Header availability cannot turn on either gate.
#if DG_IGR_ENABLE_OBSERVABILITY && !DG_IGR_OBSERVABILITY_USE_CALIPER
  static_assert(obs::compile_enabled);
  static_assert(!obs::nvtx_backend_enabled);
  dg_igr::NVTXRange compatibility("DG/Mult");
  return check_recording_backend();
#elif DG_IGR_ENABLE_OBSERVABILITY
  return check_caliper_adapter();
#else
  obs::set_runtime_enabled(true);
  if (obs::runtime_enabled())
  {
    return 1;
  }
  return disabled_probe(4) == 5 ? 0 : 2;
#endif
}
