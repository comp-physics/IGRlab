#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace dg::hdg::runtime::cpu
{
namespace
{

StateCoordinates ToCoreCoordinates(
    const GenericEulerCpuSessionCoordinates& coordinates) noexcept
{
  return {coordinates.time, coordinates.step, coordinates.generation,
          coordinates.step, coordinates.dt};
}

GenericEulerCpuSessionCoordinates FromCoreCoordinates(
    const StateCoordinates& coordinates) noexcept
{
  return {coordinates.time, coordinates.step, coordinates.generation,
          coordinates.dt};
}

} // namespace

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
  session_epoch_ = core_->incarnation();
  if (session_epoch_ == 0)
  {
    initialized_ = false;
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  diagnostics_.available = true;
  diagnostics_.coordinates = {coordinates_.time, coordinates_.dt,
                              coordinates_.step, coordinates_.generation};
  diagnostics_.wave_generation = coordinates_.generation;
  diagnostics_.maximum_wave_speed = std::numeric_limits<double>::quiet_NaN();
  // The borrowed CPU core has no characteristic-length/stable-dt estimator.
  // Do not manufacture a CFL value; capabilities() leaves that bit clear.
  return {GenericEulerCpuSessionStatus::initialized, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Prepare(
    const double dt, GenericEulerCpuPreparedStep& token) noexcept
{
  if (!initialized_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::prepare))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  const bool token_empty = !token.valid_;
  if (!core_->CollectiveAll(token_empty))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  StateCoordinates core_coordinates = ToCoreCoordinates(coordinates_);
  std::uint64_t generation = 0;
  if (!core_->PrepareBorrowedEuler(dt, core_coordinates, generation))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  token.owner_ = this;
  token.epoch_ = session_epoch_;
  token.generation_ = generation;
  token.valid_ = true;
  return {GenericEulerCpuSessionStatus::prepared, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Commit(
    GenericEulerCpuPreparedStep&& token) noexcept
{
  if (!initialized_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::commit))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_;
  StateCoordinates core_coordinates = ToCoreCoordinates(coordinates_);
  double maximum_wave_speed = 0.0;
  if (!core_->CommitBorrowedEuler(local_token_valid ? token.generation_ : 0,
                                  core_coordinates, maximum_wave_speed))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  token.valid_ = false;
  coordinates_ = FromCoreCoordinates(core_coordinates);
  diagnostics_.coordinates = {coordinates_.time, coordinates_.dt,
                              coordinates_.step, coordinates_.generation};
  diagnostics_.wave_generation = coordinates_.generation;
  diagnostics_.maximum_wave_speed = maximum_wave_speed;
  return {GenericEulerCpuSessionStatus::committed, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Abort(
    GenericEulerCpuPreparedStep&& token) noexcept
{
  if (!initialized_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::abort))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_;
  const StateCoordinates core_coordinates = ToCoreCoordinates(coordinates_);
  if (!core_->AbortBorrowedEuler(local_token_valid ? token.generation_ : 0,
                                 core_coordinates))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  token.valid_ = false;
  return {GenericEulerCpuSessionStatus::aborted, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Advance(
    const double dt) noexcept
{
  GenericEulerCpuPreparedStep token;
  if (Prepare(dt, token).status != GenericEulerCpuSessionStatus::prepared)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  const GenericEulerCpuSessionResult committed = Commit(std::move(token));
  if (committed.status == GenericEulerCpuSessionStatus::committed)
  {
    return {GenericEulerCpuSessionStatus::advanced, committed.coordinates};
  }
  // Commit can fail only before publication.  Its retained authority is still
  // valid in that case, so clean the prepared candidate collectively rather
  // than leaving this compatibility session wedged.
  static_cast<void>(Abort(std::move(token)));
  return {GenericEulerCpuSessionStatus::rejected, coordinates_};
}

std::uint32_t GenericEulerHdgCpuSession::capabilities() const noexcept
{
  return generic::HdgEulerSessionCapabilityBit(
             generic::HdgEulerSessionCapability::prepared_commit) |
      generic::HdgEulerSessionCapabilityBit(
          generic::HdgEulerSessionCapability::prepared_output) |
      generic::HdgEulerSessionCapabilityBit(
          generic::HdgEulerSessionCapability::generic_borrowed_inputs);
}

generic::HdgEulerCommittedDiagnostics
GenericEulerHdgCpuSession::diagnostics() const noexcept
{
  return diagnostics_;
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

bool GenericEulerHdgCpuSession::ExportPreparedInterior(
    const GenericEulerCpuPreparedStep& token,
    const GenericEulerCpuInteriorBuffer destination) const noexcept
{
  if (!initialized_ || !core_) { return false; }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::export_prepared_buffer))
  {
    return false;
  }
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_;
  return core_->ExportPreparedInterior(local_token_valid ? token.generation_ : 0,
                                       destination.data, destination.size);
}

bool GenericEulerHdgCpuSession::ExportPreparedInterior(
    const GenericEulerCpuPreparedStep& token,
    mfem::ParGridFunction& destination) const noexcept
{
  if (!initialized_ || !core_) { return false; }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::export_prepared_grid_function))
  {
    return false;
  }
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_;
  return core_->ExportPreparedInterior(local_token_valid ? token.generation_ : 0,
                                       destination);
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
