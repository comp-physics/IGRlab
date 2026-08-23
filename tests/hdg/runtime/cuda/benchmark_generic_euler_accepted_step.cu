#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <cuda_runtime_api.h>
#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

// Test-only POD copied while the prepared authority is live.  Keeping this
// observation in the benchmark translation unit avoids a production
// diagnostic API and lets the timed lifecycle exclude the POD copy itself.
struct CudaAcceptedStepBenchmarkObservation
{
  CudaStepEvidence evidence{};
  StateHash candidate_interior_hash{};
  StateHash candidate_trace_hash{};
  double maximum_conservation_error =
      std::numeric_limits<double>::infinity();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::infinity();
  bool candidate_ready = false;
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool conservation = false;
  bool telemetry = false;
  bool success = false;
};

struct CudaAcceptedStepCommittedHashes
{
  StateHash interior{};
  StateHash trace{};
  StateHash local_interior{};
  StateHash local_trace{};
  std::uint64_t generation = 0;
  bool valid = false;
};

struct CudaAcceptedStepProblemMetadata
{
  int elements = 0;
  int interior_components = 0;
  int trace_components = 0;
  int scalar_interior_dofs = 0;
  int scalar_face_dofs = 0;
  int interior_unknowns = 0;
  int true_trace_unknowns = 0;
  int global_true_trace_unknowns = 0;
  bool igr_active = false;
};

// The production facade and engine already grant this exact qualification
// friend.  This executable defines only the two narrow observations needed to
// keep performance work counts and numerical identity bound to every sample.
struct CudaQualificationTestAccess
{
  template <typename Engine>
  static bool CapturePrepared(
      Engine& engine,
      CudaAcceptedStepBenchmarkObservation& observation) noexcept
  {
    observation = {};
    if (!engine.initialized_ || !engine.pending_.active ||
        !engine.pending_.staging || !engine.pending_.prestage_allowed ||
        !engine.pending_.checkpoint_staged)
      return false;

    const auto& pending = engine.pending_;
    const auto& numerical = pending.numerical;
    observation.evidence = pending.evidence;
    observation.candidate_interior_hash = pending.candidate_interior_hash;
    observation.candidate_trace_hash = pending.candidate_trace_hash;
    observation.maximum_conservation_error =
        numerical.maximum_conservation_error;
    observation.maximum_conservation_relative_error =
        numerical.maximum_conservation_relative_error;
    observation.candidate_ready = numerical.candidate_ready;
    observation.finite = numerical.finite;
    observation.admissible = numerical.admissible;
    observation.recovery = numerical.recovery;
    observation.true_residual = numerical.true_residual;
    observation.conservation = numerical.conservation;
    observation.telemetry = numerical.telemetry;
    observation.success = numerical.success;
    return true;
  }

  static bool CapturePrepared(
      cuda::GenericEulerHdgCudaSession& session,
      CudaAcceptedStepBenchmarkObservation& observation) noexcept
  {
    return session.borrowed_rank1_ && !session.borrowed_rank2_ &&
        CapturePrepared(*session.borrowed_rank1_, observation);
  }

  template <typename Engine>
  static bool CaptureCommitted(
      Engine& engine,
      CudaAcceptedStepCommittedHashes& hashes) noexcept
  {
    hashes = {};
    CheckpointDescriptor checkpoint{};
    if (!engine.initialized_ || engine.pending_.active || !engine.contract_ ||
        !engine.contract_->TryMakeCheckpoint(checkpoint))
      return false;
    hashes.interior = checkpoint.interior_hash;
    hashes.trace = checkpoint.trace_hash;
    hashes.generation = checkpoint.coordinates.generation;
    hashes.valid = checkpoint.successful_commit &&
        checkpoint.schema_version == checkpoint_schema_version;
    return hashes.valid;
  }

  static bool CaptureCommitted(
      cuda::GenericEulerHdgCudaSession& session,
      CudaAcceptedStepCommittedHashes& hashes) noexcept
  {
    return session.borrowed_rank1_ && !session.borrowed_rank2_ &&
        CaptureCommitted(*session.borrowed_rank1_, hashes);
  }

  template <typename Engine>
  static bool CaptureProblemMetadata(
      Engine& engine, CudaAcceptedStepProblemMetadata& metadata) noexcept
  {
    metadata = {};
    if (!engine.initialized_) return false;
    const auto& shape = engine.numerics_.shape_;
    metadata.elements = shape.elements;
    metadata.interior_components = shape.interior_components;
    metadata.trace_components = shape.trace_components;
    metadata.scalar_interior_dofs = shape.scalar_interior_dofs;
    metadata.scalar_face_dofs = shape.scalar_face_dofs;
    metadata.interior_unknowns = shape.interior_count;
    metadata.true_trace_unknowns = shape.true_trace_count;
    metadata.global_true_trace_unknowns = shape.global_true_trace_count;
    metadata.igr_active = shape.igr_active;
    return true;
  }

  static bool CaptureProblemMetadata(
      cuda::GenericEulerHdgCudaSession& session,
      CudaAcceptedStepProblemMetadata& metadata) noexcept
  {
    return session.borrowed_rank1_ && !session.borrowed_rank2_ &&
        CaptureProblemMetadata(*session.borrowed_rank1_, metadata);
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::BoundaryPointView;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::EulerBoundaryPolicyTableView;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::StateHash;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaMutableHostL2Output;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;
using dg::hdg::runtime::cuda_w8a::conserved_components;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaAcceptedStepBenchmarkObservation;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaAcceptedStepCommittedHashes;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaAcceptedStepProblemMetadata;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;

constexpr int order = 1;
constexpr int principal_threads = 128;
constexpr int default_mesh_n = 104;
constexpr int warmup_steps = 1;
constexpr int measured_steps = 5;
constexpr int total_steps = warmup_steps + measured_steps;
constexpr double default_timestep = 1.0e-5;
constexpr std::uint32_t cuda_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cuda) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

static_assert(default_mesh_n * default_mesh_n >= 10752,
              "the default benchmark must saturate at least 84 blocks");

constexpr bool ValidDomainPeriods(const int mesh_n,
                                  const int domain_periods) noexcept
{
  return mesh_n >= 2 && domain_periods > 0 && domain_periods <= mesh_n &&
      mesh_n % domain_periods == 0;
}

static_assert(ValidDomainPeriods(default_mesh_n, 1),
              "the default unit domain must remain valid");
static_assert(ValidDomainPeriods(104, 26),
              "the fixed-h saturated IGR workload must remain valid");
static_assert(!ValidDomainPeriods(104, 0) &&
              !ValidDomainPeriods(104, 105) &&
              !ValidDomainPeriods(104, 25),
              "invalid domain replication must be rejected");

struct Options
{
  RegularizationKind regularization = RegularizationKind::none;
  int mesh_n = default_mesh_n;
  int domain_periods = 1;
  double timestep = default_timestep;
  bool help = false;
};

struct BorrowedFields
{
  std::vector<double> initial{};
  std::size_t volume_scalar_count = 0;
};

struct StepSample
{
  bool warmup = false;
  int measured_index = -1;
  double seconds = std::numeric_limits<double>::infinity();
  std::uint64_t exported_interior_checksum = 0;
  CudaAcceptedStepBenchmarkObservation prepared{};
  CudaAcceptedStepCommittedHashes committed{};
  dg::hdg::generic::HdgEulerSessionCoordinates coordinates{};
  dg::hdg::generic::HdgEulerCommittedDiagnostics diagnostics{};
};

bool ParsePositiveInt(const std::string_view text, const int minimum,
                      int& value) noexcept
{
  int parsed = 0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      parsed < minimum || parsed > 46340)
    return false;
  value = parsed;
  return true;
}

bool ParsePositiveDouble(const std::string_view text, double& value) noexcept
{
  double parsed = 0.0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), parsed,
      std::chars_format::general);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !std::isfinite(parsed) || parsed <= 0.0)
    return false;
  value = parsed;
  return true;
}

bool SelectMode(const std::string_view text, Options& options) noexcept
{
  if (text == "conventional")
  {
    options.regularization = RegularizationKind::none;
    return true;
  }
  if (text == "igr")
  {
    options.regularization = RegularizationKind::igr;
    return true;
  }
  return false;
}

bool ParseOptions(const int argc, char* argv[], Options& options,
                  std::string& error)
{
  for (int index = 1; index < argc; ++index)
  {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h")
    {
      options.help = true;
      continue;
    }
    if (argument == "--igr")
    {
      options.regularization = RegularizationKind::igr;
      continue;
    }
    if (argument == "--conventional")
    {
      options.regularization = RegularizationKind::none;
      continue;
    }
    if (argument == "--mode")
    {
      if (++index >= argc || !SelectMode(argv[index], options))
      {
        error = "--mode requires conventional or igr";
        return false;
      }
      continue;
    }
    if (argument == "--mesh-n")
    {
      if (++index >= argc ||
          !ParsePositiveInt(argv[index], 2, options.mesh_n))
      {
        error = "--mesh-n requires an integer in [2,46340]";
        return false;
      }
      continue;
    }
    if (argument == "--domain-periods")
    {
      if (++index >= argc ||
          !ParsePositiveInt(argv[index], 1, options.domain_periods))
      {
        error = "--domain-periods requires an integer in [1,46340]";
        return false;
      }
      continue;
    }
    if (argument == "--dt")
    {
      if (++index >= argc ||
          !ParsePositiveDouble(argv[index], options.timestep))
      {
        error = "--dt requires a finite positive FP64 value";
        return false;
      }
      continue;
    }
    error = "unknown argument: " + std::string(argument);
    return false;
  }
  if (!ValidDomainPeriods(options.mesh_n, options.domain_periods))
  {
    error = "--domain-periods must be positive, no greater than --mesh-n, "
            "and divide --mesh-n exactly";
    return false;
  }
  return true;
}

void PrintUsage(const char* executable)
{
  std::cout << "Usage: " << executable
            << " [--mode conventional|igr] [--mesh-n N]"
               " [--domain-periods K] [--dt DT]\n"
               "       " << executable
            << " [--conventional|--igr] [--mesh-n N]"
               " [--domain-periods K] [--dt DT]\n";
}

mfem::ParMesh MakePeriodicMesh(const int n, const int domain_periods)
{
  const double domain_length = static_cast<double>(domain_periods);
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, true,
      domain_length, domain_length);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = domain_length;
  translations[1][1] = domain_length;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      serial, serial.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

void SmoothPeriodicState(const mfem::Vector& point, const bool wrap_period,
                         mfem::Vector& state)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double x = wrap_period ? point[0] - std::floor(point[0]) : point[0];
  const double y = wrap_period ? point[1] - std::floor(point[1]) : point[1];
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  const double density = 1.0 + 0.04 * sx * sy;
  const double velocity_x = 0.18 + 0.02 * cx * sy;
  const double velocity_y = -0.09 + 0.015 * sx * cy;
  const double pressure = 1.0 + 0.03 * cx * cy;
  state.SetSize(conserved_components);
  state[0] = density;
  state[1] = density * velocity_x;
  state[2] = density * velocity_y;
  state[3] = pressure / 0.4 + 0.5 * density *
      (velocity_x * velocity_x + velocity_y * velocity_y);
}

BorrowedFields MakeFields(mfem::ParMesh& mesh, const int domain_periods)
{
  BorrowedFields fields;
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, conserved_components, mfem::Ordering::byNODES);
  mfem::ParGridFunction projected(&space);
  const auto evaluator = [wrap_period = domain_periods > 1](
      const mfem::Vector& point, mfem::Vector& state)
  { SmoothPeriodicState(point, wrap_period, state); };
  mfem::VectorFunctionCoefficient coefficient(
      conserved_components, evaluator);
  projected.ProjectCoefficient(coefficient);
  fields.volume_scalar_count = static_cast<std::size_t>(
      space.GetVSize() / conserved_components);
  const double* values = projected.HostRead();
  if (values != nullptr && projected.Size() > 0)
    fields.initial.assign(values, values + projected.Size());
  return fields;
}

bool PeriodicExterior(const void*, const BoundaryPointView&,
                      const double* interior, double* exterior) noexcept
{
  if (interior == nullptr || exterior == nullptr) return false;
  std::copy_n(interior, conserved_components, exterior);
  return true;
}

EulerHdgRequest MakeRequest(const mfem::ParMesh& mesh,
                            const BorrowedFields& fields,
                            const RegularizationKind regularization)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {BackendKind::cuda, 0};
  request.capabilities = cuda_capabilities;
  request.initial_state = {
      fields.initial.data(), fields.initial.size(), conserved_components};
  // Exercise the established projected-trace path.  Projection and all other
  // setup happen once before the warmup and are outside every timed sample.
  request.initial_true_trace = {nullptr, 0, conserved_components};
  request.source = {nullptr, 0, conserved_components};
  request.boundary.context = nullptr;
  request.boundary.exterior_state = PeriodicExterior;
  request.boundary.policies = EulerBoundaryPolicyTableView{
      nullptr, 0, 2, 1.4};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  if (regularization == RegularizationKind::igr)
    request.numerical_policy.stabilization = {
        RegularizationKind::igr, 0.2, 0.8};
  return request;
}

std::string HashHex(const StateHash& hash)
{
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.resize(2 * hash.bytes.size());
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    const std::uint8_t value = hash.bytes[index];
    result[2 * index] = digits[value >> 4U];
    result[2 * index + 1] = digits[value & 0x0fU];
  }
  return result;
}

std::uint64_t StableFp64Checksum(const std::vector<double>& values) noexcept
{
  std::uint64_t checksum = 1469598103934665603ULL;
  for (const double value : values)
  {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned int shift = 0; shift < 64; shift += 8)
    {
      checksum ^= (bits >> shift) & 0xffU;
      checksum *= 1099511628211ULL;
    }
  }
  return checksum;
}

bool Healthy(const CudaAcceptedStepBenchmarkObservation& observation) noexcept
{
  const auto& evidence = observation.evidence;
  return observation.candidate_ready && observation.finite &&
      observation.admissible && observation.recovery &&
      observation.true_residual && observation.conservation &&
      observation.telemetry && observation.success &&
      evidence.products.CoupledAndOrdered() &&
      std::isfinite(evidence.initial_residual) &&
      std::isfinite(evidence.final_residual) &&
      std::isfinite(evidence.true_relative_residual) &&
      std::isfinite(observation.maximum_conservation_error) &&
      std::isfinite(observation.maximum_conservation_relative_error) &&
      evidence.nonlinear_updates >= 0 && evidence.trace_solves > 0 &&
      evidence.linear_iterations >= 0 && evidence.line_search_trials >= 0 &&
      evidence.rejected_line_search_trials >= 0;
}

bool RunStep(GenericEulerHdgCudaSession& session,
             const BorrowedFields& fields,
             std::vector<double>& exported_interior,
             const double timestep,
             const bool warmup, const int measured_index,
             StepSample& sample, std::string& error)
{
  using clock = std::chrono::steady_clock;
  sample = {};
  sample.warmup = warmup;
  sample.measured_index = measured_index;
  GenericEulerCudaPreparedStep authority{};
  CudaPreparedOutputMetadataV1 output{};

  const auto started = clock::now();
  const auto prepared = session.Prepare(timestep, authority, output);
  const auto observation_started = clock::now();
  const bool observed = prepared.status == HdgEulerSessionStatus::prepared &&
      CudaQualificationTestAccess::CapturePrepared(session, sample.prepared);
  const auto observation_finished = clock::now();
  const bool exported = observed && session.ExportPreparedInterior(
      authority, output, CudaMutableHostL2Output{
          exported_interior.data(), fields.volume_scalar_count,
          conserved_components});
  const bool acknowledged = exported &&
      session.AcknowledgePreparedOutput(authority, output);
  const auto committed = acknowledged
      ? session.Commit(std::move(authority), output)
      : GenericEulerHdgCudaSession::result_type{};
  const auto finished = clock::now();

  sample.seconds = std::chrono::duration<double>(
      (observation_started - started) +
      (finished - observation_finished)).count();
  if (!observed || !exported || !acknowledged ||
      committed.status != HdgEulerSessionStatus::committed ||
      !output.successful_commit)
  {
    error = "accepted-step transaction failed: prepare_status=" +
        std::to_string(static_cast<unsigned int>(prepared.status)) +
        ", commit_status=" +
        std::to_string(static_cast<unsigned int>(committed.status));
    return false;
  }
  // Hashing and committed-state observation are intentionally outside the
  // timed Prepare + prepared D2H export + Acknowledge + Commit region.
  sample.exported_interior_checksum =
      StableFp64Checksum(exported_interior);
  if (!CudaQualificationTestAccess::CaptureCommitted(
          session, sample.committed))
  {
    error = "committed hash capture failed";
    return false;
  }
  sample.coordinates = session.coordinates();
  sample.diagnostics = session.diagnostics();
  if (!Healthy(sample.prepared))
  {
    error = "accepted-step prepared numerical gates failed";
    return false;
  }
  if (sample.prepared.candidate_interior_hash != sample.committed.interior ||
      sample.prepared.candidate_trace_hash != sample.committed.trace)
  {
    error = "prepared and committed hashes differ: interior=" +
        HashHex(sample.prepared.candidate_interior_hash) + "/" +
        HashHex(sample.committed.interior) + ", trace=" +
        HashHex(sample.prepared.candidate_trace_hash) + "/" +
        HashHex(sample.committed.trace);
    return false;
  }
  if (sample.coordinates.generation != sample.committed.generation)
  {
    error = "committed generation differs from coordinates";
    return false;
  }
  if (!dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(
          sample.diagnostics, session.capabilities()))
  {
    error = "committed diagnostics are invalid";
    return false;
  }
  return true;
}

double Median(std::array<double, measured_steps> values) noexcept
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double PopulationStandardDeviation(
    const std::array<double, measured_steps>& values) noexcept
{
  double mean = 0.0;
  for (const double value : values) mean += value;
  mean /= static_cast<double>(values.size());
  double squared = 0.0;
  for (const double value : values)
  {
    const double difference = value - mean;
    squared += difference * difference;
  }
  return std::sqrt(squared / static_cast<double>(values.size()));
}

void PrintJsonNumber(const double value)
{
  if (std::isfinite(value)) std::cout << value;
  else std::cout << "null";
}

void PrintStep(const StepSample& sample)
{
  const auto& evidence = sample.prepared.evidence;
  std::cout << "{\"phase\":\""
            << (sample.warmup ? "warmup" : "measured") << "\","
            << "\"measured_index\":" << sample.measured_index << ','
            << "\"duration_seconds\":";
  PrintJsonNumber(sample.seconds);
  std::cout << ",\"step\":" << sample.coordinates.step
            << ",\"generation\":" << sample.coordinates.generation
            << ",\"time\":";
  PrintJsonNumber(sample.coordinates.time);
  std::cout << ",\"committed_interior_hash\":\""
            << HashHex(sample.committed.interior) << "\","
            << "\"committed_trace_hash\":\""
            << HashHex(sample.committed.trace) << "\","
            << "\"exported_interior_checksum_fnv1a64\":\""
            << std::hex << std::setw(16) << std::setfill('0')
            << sample.exported_interior_checksum << std::dec
            << std::setfill(' ') << "\","
            << "\"initial_residual\":";
  PrintJsonNumber(evidence.initial_residual);
  std::cout << ",\"final_residual\":";
  PrintJsonNumber(evidence.final_residual);
  std::cout << ",\"true_relative_residual\":";
  PrintJsonNumber(evidence.true_relative_residual);
  std::cout << ",\"maximum_conservation_error\":";
  PrintJsonNumber(sample.prepared.maximum_conservation_error);
  std::cout << ",\"maximum_conservation_relative_error\":";
  PrintJsonNumber(sample.prepared.maximum_conservation_relative_error);
  std::cout << ",\"maximum_wave_speed\":";
  PrintJsonNumber(sample.diagnostics.maximum_wave_speed);
  std::cout << ",\"backend_stable_dt\":";
  PrintJsonNumber(sample.diagnostics.backend_stable_dt);
  std::cout << ",\"nonlinear_updates\":" << evidence.nonlinear_updates
            << ",\"trace_solves\":" << evidence.trace_solves
            << ",\"linear_iterations\":" << evidence.linear_iterations
            << ",\"line_search_trials\":" << evidence.line_search_trials
            << ",\"rejected_line_search_trials\":"
            << evidence.rejected_line_search_trials
            << ",\"gates\":{"
            << "\"candidate_ready\":" << std::boolalpha
            << sample.prepared.candidate_ready
            << ",\"finite\":" << sample.prepared.finite
            << ",\"admissible\":" << sample.prepared.admissible
            << ",\"recovery\":" << sample.prepared.recovery
            << ",\"true_residual\":" << sample.prepared.true_residual
            << ",\"conservation\":" << sample.prepared.conservation
            << ",\"telemetry\":" << sample.prepared.telemetry
            << ",\"products_coupled_and_ordered\":"
            << evidence.products.CoupledAndOrdered()
            << ",\"success\":" << sample.prepared.success
            << std::noboolalpha << "}}";
}

int RunBenchmark(const Options& options)
{
  mfem::Device device("cuda");
  int active_device = -1;
  cudaDeviceProp device_properties{};
  if (cudaGetDevice(&active_device) != cudaSuccess || active_device < 0 ||
      cudaGetDeviceProperties(&device_properties, active_device) !=
          cudaSuccess || device_properties.multiProcessorCount <= 0)
  {
    std::cerr << "{\"schema\":1,\"success\":false,"
                 "\"error\":\"CUDA device metadata query failed\"}\n";
    return 1;
  }

  mfem::ParMesh mesh = MakePeriodicMesh(
      options.mesh_n, options.domain_periods);
  const int elements = mesh.GetNE();
  const int principal_blocks =
      (elements + principal_threads - 1) / principal_threads;
  BorrowedFields fields = MakeFields(mesh, options.domain_periods);
  if (elements != options.mesh_n * options.mesh_n ||
      fields.volume_scalar_count == 0 ||
      fields.initial.size() !=
          fields.volume_scalar_count * conserved_components)
  {
    std::cerr << "{\"schema\":1,\"success\":false,"
                 "\"error\":\"benchmark field initialization failed\"}\n";
    return 1;
  }

  GenericEulerHdgCudaSession session;
  const EulerHdgRequest request = MakeRequest(
      mesh, fields, options.regularization);
  if (session.Initialize(request).status !=
      HdgEulerSessionStatus::initialized)
  {
    std::cerr << "{\"schema\":1,\"success\":false,"
                 "\"error\":\"CUDA session initialization failed\"}\n";
    return 1;
  }

  CudaAcceptedStepProblemMetadata problem{};
  const bool problem_captured =
      CudaQualificationTestAccess::CaptureProblemMetadata(session, problem);
  const int expected_interior_components =
      options.regularization == RegularizationKind::igr
      ? dg::hdg::runtime::cuda_w8a::igr_interior_components
      : conserved_components;
  const int expected_trace_components =
      options.regularization == RegularizationKind::igr
      ? dg::hdg::runtime::cuda_w8a::igr_trace_components
      : conserved_components;
  const std::int64_t expected_interior_unknowns =
      static_cast<std::int64_t>(elements) *
      problem.scalar_interior_dofs * expected_interior_components;
  const std::int64_t expected_true_trace_unknowns =
      2LL * options.mesh_n * options.mesh_n *
      problem.scalar_face_dofs * expected_trace_components;
  if (!problem_captured ||
      problem.elements != elements || problem.scalar_interior_dofs <= 0 ||
      problem.scalar_face_dofs <= 0 ||
      problem.interior_components != expected_interior_components ||
      problem.trace_components != expected_trace_components ||
      problem.igr_active !=
          (options.regularization == RegularizationKind::igr) ||
      problem.interior_unknowns != expected_interior_unknowns ||
      problem.true_trace_unknowns != expected_true_trace_unknowns ||
      problem.true_trace_unknowns != problem.global_true_trace_unknowns)
  {
    std::cerr << "{\"schema\":1,\"success\":false,"
                 "\"error\":\"CUDA problem metadata validation failed\"}\n";
    return 1;
  }

  std::vector<double> exported_interior(fields.initial.size(), 0.0);
  std::array<StepSample, total_steps> samples{};
  std::string error;
  for (int index = 0; index < total_steps; ++index)
  {
    const bool warmup = index < warmup_steps;
    if (!RunStep(session, fields, exported_interior, options.timestep, warmup,
                 warmup ? -1 : index - warmup_steps,
                 samples[static_cast<std::size_t>(index)], error))
    {
      std::cerr << "{\"schema\":1,\"success\":false,\"step_index\":"
                << index << ",\"error\":\"" << error << "\"}\n";
      return 1;
    }
  }

  std::array<double, measured_steps> durations{};
  for (int index = 0; index < measured_steps; ++index)
    durations[static_cast<std::size_t>(index)] =
        samples[static_cast<std::size_t>(index + warmup_steps)].seconds;

  std::cout << std::setprecision(17)
            << "{\"schema\":1,"
            << "\"benchmark\":\"dg_igr_hdg_cuda_accepted_step_fp64\","
            << "\"success\":true,"
            << "\"mode\":\""
            << (options.regularization == RegularizationKind::igr
                    ? "igr" : "conventional") << "\","
            << "\"precision\":\"fp64\","
            << "\"rank_count\":1,"
            << "\"polynomial_order\":" << order << ','
            << "\"mesh_n\":" << options.mesh_n << ','
            << "\"domain_periods\":" << options.domain_periods << ','
            << "\"domain_length\":" << options.domain_periods << ','
            << "\"cells_per_period\":"
            << options.mesh_n / options.domain_periods << ','
            << "\"element_h\":"
            << static_cast<double>(options.domain_periods) /
                static_cast<double>(options.mesh_n) << ','
            << "\"elements\":" << elements << ','
            << "\"interior_unknowns\":" << problem.interior_unknowns << ','
            << "\"true_trace_unknowns\":"
            << problem.true_trace_unknowns << ','
            << "\"gpu_sm_count\":"
            << device_properties.multiProcessorCount << ','
            << "\"principal_threads\":" << principal_threads << ','
            << "\"principal_blocks\":" << principal_blocks << ','
            << "\"dt\":" << options.timestep << ','
            << "\"initialization_timed\":false,"
            << "\"timing_scope\":\"Prepare+ExportPreparedInteriorD2H+"
               "AcknowledgePreparedOutput+Commit\","
            << "\"warmup_steps\":" << warmup_steps << ','
            << "\"measured_steps\":" << measured_steps << ','
            << "\"warmup_duration_seconds\":";
  PrintJsonNumber(samples[0].seconds);
  std::cout << ",\"raw_durations_seconds\":[";
  for (std::size_t index = 0; index < durations.size(); ++index)
  {
    if (index != 0) std::cout << ',';
    PrintJsonNumber(durations[index]);
  }
  std::cout << "],\"median_seconds\":";
  PrintJsonNumber(Median(durations));
  std::cout << ",\"population_stddev_seconds\":";
  PrintJsonNumber(PopulationStandardDeviation(durations));
  std::cout << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index)
  {
    if (index != 0) std::cout << ',';
    PrintStep(samples[index]);
  }
  std::cout << "]}\n";
  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  Options options;
  std::string error;
  if (!ParseOptions(argc, argv, options, error))
  {
    std::cerr << error << '\n';
    PrintUsage(argv[0]);
    return 2;
  }
  if (options.help)
  {
    PrintUsage(argv[0]);
    return 0;
  }
  if (ranks != 1)
  {
    std::cerr << "{\"schema\":1,\"success\":false,"
                 "\"error\":\"benchmark requires exactly one MPI rank\"}\n";
    return 2;
  }
  return RunBenchmark(options);
}
