#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <memory>

namespace dg::hdg::runtime::cpu
{

GenericEulerHdgCpuSession::GenericEulerHdgCpuSession()
    : core_(std::make_unique<detail::CpuNumericsCore>())
{}

GenericEulerHdgCpuSession::~GenericEulerHdgCpuSession() = default;

GenericEulerCpuSessionCoordinates
GenericEulerHdgCpuSession::coordinates() const noexcept
{
  return coordinates_;
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Initialize(
    const generic::EulerHdgRequest& request) noexcept
{
  if (initialization_attempted_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  initialization_attempted_ = true;
  const generic::ContractStatus admission =
      generic::ValidateEulerHdgRequest(request);
  if (admission != generic::ContractStatus::accepted ||
      request.backend.kind != generic::BackendKind::cpu ||
      request.mesh == nullptr)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  request_ = request;
  initialized_ = core_->InitializeBorrowedEuler(request_);
  if (!initialized_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  return {GenericEulerCpuSessionStatus::initialized, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Advance(
    const double dt) noexcept
{
  if (!initialized_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  StateCoordinates core_coordinates{coordinates_.time, coordinates_.step,
                                    coordinates_.generation,
                                    coordinates_.step, coordinates_.dt};
  if (!core_->AdvanceBorrowedEuler(dt, core_coordinates))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  coordinates_ = {core_coordinates.time, core_coordinates.step,
                  core_coordinates.generation, core_coordinates.dt};
  return {GenericEulerCpuSessionStatus::advanced, coordinates_};
}

bool GenericEulerHdgCpuSession::ExportCommittedInterior(
    const GenericEulerCpuInteriorBuffer destination) const noexcept
{
  return initialized_ && core_ &&
      core_->ExportCommittedInterior(destination.data, destination.size);
}

bool GenericEulerHdgCpuSession::ExportCommittedInterior(
    mfem::ParGridFunction& destination) const noexcept
{
  return initialized_ && core_ && core_->ExportCommittedInterior(destination);
}

bool ExecuteGenericEulerHdgCpu(const generic::EulerHdgRequest& request) noexcept
{
  GenericEulerHdgCpuSession session;
  return session.Initialize(request).status ==
             GenericEulerCpuSessionStatus::initialized &&
      session.Advance(cpu_backward_euler_dt).status ==
          GenericEulerCpuSessionStatus::advanced;
}

bool RestartGenericEulerHdgCpu(const generic::EulerHdgRequest&) noexcept
{
  // No checkpoint representation is accepted by this borrowed, one-shot API.
  // Leaving the caller's output untouched makes rejection failure-atomic.
  return false;
}

} // namespace dg::hdg::runtime::cpu
