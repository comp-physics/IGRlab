#include "numerics/hdg/runtime/cuda/generic_euler_app.cuh"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <utility>

namespace dg::hdg::runtime::cuda
{

GenericEulerHdgCudaAppSession::GenericEulerHdgCudaAppSession()
    : session_(std::make_unique<GenericEulerHdgCudaSession>())
{}

GenericEulerHdgCudaAppSession::~GenericEulerHdgCudaAppSession() = default;

GenericEulerHdgCudaAppSession::result_type
GenericEulerHdgCudaAppSession::Initialize(
    const generic::EulerHdgRequest& request) noexcept
{ return session_->Initialize(request); }

std::uint32_t GenericEulerHdgCudaAppSession::capabilities() const noexcept
{ return session_->capabilities(); }

GenericEulerHdgCudaAppSession::coordinates_type
GenericEulerHdgCudaAppSession::coordinates() const noexcept
{ return session_->coordinates(); }

generic::HdgEulerCommittedDiagnostics
GenericEulerHdgCudaAppSession::diagnostics() const noexcept
{ return session_->diagnostics(); }

GenericEulerHdgCudaAppSession::result_type
GenericEulerHdgCudaAppSession::Restore(
    const prepared_output_type& metadata,
    const GenericEulerCudaCheckpointPayloadViewV1 payload) noexcept
{ return session_->Restore(metadata, payload); }

bool GenericEulerHdgCudaAppSession::ExportCommittedFields(
    const generic::HdgEulerHostInteriorFields destination) noexcept
{ return session_->ExportCommittedFields(destination); }

GenericEulerHdgCudaAppSession::result_type
GenericEulerHdgCudaAppSession::Prepare(
    const double dt, GenericEulerCudaPreparedStep& authority,
    prepared_output_type& metadata,
    GenericEulerCudaPrepareFailureClass* failure_class) noexcept
{ return session_->Prepare(dt, authority, metadata, failure_class); }

bool GenericEulerHdgCudaAppSession::AssessPrepared(
    const GenericEulerCudaPreparedStep& authority,
    const prepared_output_type& metadata,
    const generic::HdgEulerPreparedAssessmentPolicy policy,
    generic::HdgEulerPreparedAssessment& assessment) noexcept
{ return session_->AssessPrepared(authority, metadata, policy, assessment); }

bool GenericEulerHdgCudaAppSession::ExportPreparedFields(
    const GenericEulerCudaPreparedStep& authority,
    const prepared_output_type& metadata,
    const generic::HdgEulerHostInteriorFields destination) noexcept
{ return session_->ExportPreparedFields(authority, metadata, destination); }

bool GenericEulerHdgCudaAppSession::ExportPreparedCheckpointPayload(
    const GenericEulerCudaPreparedStep& authority,
    const prepared_output_type& metadata,
    const GenericEulerCudaMutableCheckpointPayloadV1 destination) noexcept
{
  return session_->ExportPreparedCheckpointPayload(
      authority, metadata, destination);
}

bool GenericEulerHdgCudaAppSession::AcknowledgePreparedOutput(
    const GenericEulerCudaPreparedStep& authority,
    prepared_output_type& metadata) noexcept
{ return session_->AcknowledgePreparedOutput(authority, metadata); }

GenericEulerHdgCudaAppSession::result_type
GenericEulerHdgCudaAppSession::Commit(
    GenericEulerCudaPreparedStep&& authority,
    prepared_output_type& metadata) noexcept
{ return session_->Commit(std::move(authority), metadata); }

GenericEulerHdgCudaAppSession::result_type
GenericEulerHdgCudaAppSession::Abort(
    GenericEulerCudaPreparedStep&& authority,
    prepared_output_type& metadata) noexcept
{ return session_->Abort(std::move(authority), metadata); }

GenericEulerHdgCudaAppSession::result_type
GenericEulerHdgCudaAppSession::RejectPrepared(
    GenericEulerCudaPreparedStep&& authority,
    prepared_output_type& metadata) noexcept
{ return session_->RejectPrepared(std::move(authority), metadata); }

GenericEulerHdgCudaAppSession::result_type
GenericEulerHdgCudaAppSession::RejectFailedPrepare() noexcept
{ return session_->RejectFailedPrepare(); }

} // namespace dg::hdg::runtime::cuda
