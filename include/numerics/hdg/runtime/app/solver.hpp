#pragma once

#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <mpi.h>

#include <string>

namespace dg::hdg::runtime::app
{

class MpiApplicationContextV1 final
{
public:
  explicit MpiApplicationContextV1(MPI_Comm communicator = MPI_COMM_WORLD)
      noexcept;

  MPI_Comm communicator() const noexcept { return communicator_; }
  int rank() const noexcept { return rank_; }
  int ranks() const noexcept { return ranks_; }
  const std::string& diagnostic() const noexcept { return diagnostic_; }
  void SetDiagnostic(std::string diagnostic) { diagnostic_ = std::move(diagnostic); }

private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
  int rank_ = -1;
  int ranks_ = 0;
  std::string diagnostic_{};
};

class CudaApplicationContextV1 final
{
public:
  explicit CudaApplicationContextV1(MPI_Comm communicator = MPI_COMM_WORLD)
      noexcept;

  MPI_Comm communicator() const noexcept { return communicator_; }
  int rank() const noexcept { return rank_; }
  int ranks() const noexcept { return ranks_; }
  const std::string& diagnostic() const noexcept { return diagnostic_; }
  void SetDiagnostic(std::string diagnostic) { diagnostic_ = std::move(diagnostic); }

private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
  int rank_ = -1;
  int ranks_ = 0;
  std::string diagnostic_{};
};

/**
 * Thin static bridge for a future common application translator.  It owns no
 * numerical state and deliberately points at the closed request/config and
 * existing concrete CPU/CUDA entry contexts.
 */
struct HdgMethodRequestV1
{
  const ApplicationOptionsView* options = nullptr;
  const HdgRunConfigV1* config = nullptr;
  MpiApplicationContextV1* cpu_context = nullptr;
  CudaApplicationContextV1* cuda_context = nullptr;
};

struct HdgMethodResultV1
{
  ApplicationRunResult run{};
  HdgMethodCapabilitiesV1 capabilities{};
  HdgCheckpointIdentityV1 checkpoint_identity{};
};

/** Static dispatch only; numerical leaves remain their existing owners. */
HdgMethodResultV1 RunHdgMethodV1(const HdgMethodRequestV1& request);

ApplicationRunResult DispatchApplication(const ApplicationOptionsView& request,
                                         MpiApplicationContextV1& cpu_context,
                                         CudaApplicationContextV1* cuda_context);

} // namespace dg::hdg::runtime::app
