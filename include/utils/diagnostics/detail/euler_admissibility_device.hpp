#pragma once

// Private compiled CUDA boundary. Plans retain invariant incidence/shapes and
// bounded scratch storage, overwritten completely before each current result.
#include "mfem.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace mfem
{
namespace igrlab
{
enum class EulerAdmissibilityPointSet;
struct EulerAdmissibilitySummary;
namespace detail
{
struct AdmissibilityElementMap;
struct AdmissibilityShapeTable;
struct AdmissibilityElementSamples;
struct AdmissibilityDevicePlan;

// Internal successful-path shortcut for RK's failure-only detailed scanner.
// True certifies that every rank inspected the current device state and found
// only finite values. Any unsupported/local-host route or nonfinite value makes
// every rank return false, so all ranks enter the unchanged detailed scanner.
// The caller owns one reusable integer of scratch; no state values are copied
// to the host. Full public scan summaries and metadata remain unchanged.
bool CanSkipDetailedFiniteScan(const mfem::Vector& state, MPI_Comm comm,
                              mfem::Array<int>& status);

inline bool AdmissibilityOrdinaryCuda()
{
  const unsigned long native = static_cast<unsigned long>(mfem::Backend::CPU) |
                               static_cast<unsigned long>(mfem::Backend::CUDA);
  return mfem::Device::Allows(mfem::Backend::CUDA) &&
         !mfem::Device::Allows(~native);
}

std::shared_ptr<AdmissibilityDevicePlan> MakeAdmissibilityDevicePlan(
  const std::vector<AdmissibilityElementMap>& elements,
  const std::vector<AdmissibilityShapeTable>& shapes,
  const std::vector<AdmissibilityElementSamples>& samples,
  EulerAdmissibilityPointSet point_set, int state_size,
  std::size_t available_bytes, std::size_t& charged_bytes);

// False declines the entire invocation before any summary is published. The
// caller then executes the unchanged host scanner, including its MPI calls.
bool EvaluateAdmissibilityDevice(const AdmissibilityDevicePlan& plan,
  const mfem::Vector& state, double gamma, double rho_min, double p_min,
  EulerAdmissibilitySummary& local);
} // namespace detail
} // namespace igrlab
} // namespace mfem
