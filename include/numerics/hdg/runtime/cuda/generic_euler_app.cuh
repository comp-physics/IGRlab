#pragma once

#include "numerics/hdg/runtime/cuda/generic_euler_types.cuh"
#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh"

#include <memory>

namespace dg::hdg::runtime::cuda
{

// Application boundary for the existing CUDA facade. Engine construction,
// destruction, and all operations are compiled together in CUDA; the host
// application never allocates or accesses CUDA-conditioned MFEM storage.
class GenericEulerHdgCudaAppSession
{
public:
  using result_type = generic::HdgEulerSessionResult;
  using coordinates_type = generic::HdgEulerSessionCoordinates;
  using prepared_output_type = cuda_w8a::CudaPreparedOutputMetadataV1;

  GenericEulerHdgCudaAppSession();
  ~GenericEulerHdgCudaAppSession();
  GenericEulerHdgCudaAppSession(const GenericEulerHdgCudaAppSession&) = delete;
  GenericEulerHdgCudaAppSession& operator=(
      const GenericEulerHdgCudaAppSession&) = delete;
  GenericEulerHdgCudaAppSession(GenericEulerHdgCudaAppSession&&) = delete;
  GenericEulerHdgCudaAppSession& operator=(GenericEulerHdgCudaAppSession&&) =
      delete;

  result_type Initialize(const generic::EulerHdgRequest& request) noexcept;
  result_type Initialize(
      const generic::EulerHdgRequest& request,
      cuda_w8a::CudaTracePolicyKind trace_policy) noexcept;
  std::uint32_t capabilities() const noexcept;
  coordinates_type coordinates() const noexcept;
  generic::HdgEulerCommittedDiagnostics diagnostics() const noexcept;
  result_type Restore(
      const prepared_output_type& metadata,
      GenericEulerCudaCheckpointPayloadViewV1 payload) noexcept;
  bool ExportCommittedFields(
      generic::HdgEulerHostInteriorFields destination) noexcept;
  result_type Prepare(
      double dt, GenericEulerCudaPreparedStep& authority,
      prepared_output_type& metadata,
      GenericEulerCudaPrepareFailureClass* failure_class = nullptr) noexcept;
  bool AssessPrepared(
      const GenericEulerCudaPreparedStep& authority,
      const prepared_output_type& metadata,
      generic::HdgEulerPreparedAssessmentPolicy policy,
      generic::HdgEulerPreparedAssessment& assessment) noexcept;
  bool ExportPreparedFields(
      const GenericEulerCudaPreparedStep& authority,
      const prepared_output_type& metadata,
      generic::HdgEulerHostInteriorFields destination) noexcept;
  bool ExportPreparedCheckpointPayload(
      const GenericEulerCudaPreparedStep& authority,
      const prepared_output_type& metadata,
      GenericEulerCudaMutableCheckpointPayloadV1 destination) noexcept;
  bool AcknowledgePreparedOutput(
      const GenericEulerCudaPreparedStep& authority,
      prepared_output_type& metadata) noexcept;
  result_type Commit(GenericEulerCudaPreparedStep&& authority,
                     prepared_output_type& metadata) noexcept;
  result_type Abort(GenericEulerCudaPreparedStep&& authority,
                    prepared_output_type& metadata) noexcept;
  result_type RejectPrepared(GenericEulerCudaPreparedStep&& authority,
                             prepared_output_type& metadata) noexcept;
  result_type RejectFailedPrepare() noexcept;

private:
  std::unique_ptr<GenericEulerHdgCudaSession> session_;
};

} // namespace dg::hdg::runtime::cuda
