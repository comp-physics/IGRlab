#pragma once

#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr.cuh"
#include "numerics/hdg/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace dg_igr_hdg_sc25_cuda_test
{

namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25;
using ::dg::hdg::runtime::Sha256;
using ::dg::hdg::runtime::Sha256Digest;
using ::dg::hdg::runtime::StateHash;

struct ComparisonEvidence final
{
  int checked_arrays = 0;
  double maximum_relative_l2 = 0.0;
  std::vector<std::string> failures;
};

struct StateQuadratureAudit final
{
  int volume_points = 0;
  int side_points = 0;
  int trace_points = 0;
  double minimum_density = INFINITY;
  double minimum_pressure = INFINITY;
  double minimum_speed_margin = INFINITY;
  std::array<double, scalar::state_components> integrals{};
};

inline bool AuditPrimitive(const double state[scalar::state_components],
                           StateQuadratureAudit& audit) noexcept
{
  for (int component = 0; component < scalar::state_components; ++component)
    if (!std::isfinite(state[component])) { return false; }
  const double rho = state[0];
  if (!(rho >= 1.0e-8)) { return false; }
  const double vx = state[1] / rho;
  const double vy = state[2] / rho;
  const double pressure = 0.4 *
      (state[3] - 0.5 * (state[1] * state[1] +
                         state[2] * state[2]) / rho);
  if (!(pressure >= 1.0e-8) || !std::isfinite(pressure) ||
      !std::isfinite(vx) || !std::isfinite(vy))
    return false;
  const double speed = std::hypot(vx, vy);
  const double sound = std::sqrt(1.4 * pressure / rho);
  const double margin = 2.5 - speed - sound;
  if (!(margin > 0.0) || !std::isfinite(margin)) { return false; }
  audit.minimum_density = std::min(audit.minimum_density, rho);
  audit.minimum_pressure = std::min(audit.minimum_pressure, pressure);
  audit.minimum_speed_margin = std::min(audit.minimum_speed_margin, margin);
  return true;
}

inline bool AuditPublishedState(
    const scalar::PublishedState& state, const int quadrature_order,
    StateQuadratureAudit& audit)
{
  audit = {};
  audit.minimum_density = INFINITY;
  audit.minimum_pressure = INFINITY;
  audit.minimum_speed_margin = INFINITY;
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        8, 8, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(
        scalar::dimension, mfem::Vector(scalar::dimension));
    translations[0][0] = 1.0;
    translations[1][1] = 1.0;
    mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    std::vector<int> partition(static_cast<std::size_t>(periodic.GetNE()), 0);
    mfem::ParMesh mesh(MPI_COMM_WORLD, periodic, partition.data());
    mfem::L2_FECollection volume_collection(
        1, scalar::dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::DG_Interface_FECollection trace_collection(
        1, scalar::dimension, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace volume_space(
        &mesh, &volume_collection, 1, mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace trace_space(
        &mesh, &trace_collection, scalar::trace_components,
        mfem::Ordering::byNODES);
    if (mesh.GetNE() != scalar::persistent_elements ||
        volume_space.GetVSize() !=
            scalar::persistent_elements * scalar::volume_nodes ||
        trace_space.GetTrueVSize() != scalar::persistent_trace_unknowns)
      return false;

    for (int element = 0; element < mesh.GetNE(); ++element)
    {
      const mfem::FiniteElement* finite_element = volume_space.GetFE(element);
      mfem::ElementTransformation* transformation =
          mesh.GetElementTransformation(element);
      if (finite_element == nullptr || transformation == nullptr ||
          finite_element->GetDof() != scalar::volume_nodes)
        return false;
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      mfem::Vector shape(scalar::volume_nodes);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& integration_point = rule.IntPoint(point);
        finite_element->CalcShape(integration_point, shape);
        transformation->SetIntPoint(&integration_point);
        const double weight = integration_point.weight *
            transformation->Weight();
        double value[scalar::state_components]{};
        for (int component = 0; component < scalar::state_components;
             ++component)
          for (int node = 0; node < scalar::volume_nodes; ++node)
            value[component] += state.u[
                (element * scalar::state_components + component) *
                    scalar::volume_nodes + node] * shape[node];
        if (!(weight > 0.0) || !std::isfinite(weight) ||
            !AuditPrimitive(value, audit))
          return false;
        for (int component = 0; component < scalar::state_components;
             ++component)
          audit.integrals[component] += weight * value[component];
        ++audit.volume_points;
      }
    }

    const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
        mfem::Geometry::SEGMENT, quadrature_order);
    for (int face = 0; face < mesh.GetNumFaces(); ++face)
    {
      mfem::FaceElementTransformations* face_transformations =
          mesh.GetFaceElementTransformations(face);
      const mfem::FiniteElement* face_element = trace_space.GetFaceElement(face);
      mfem::Array<int> face_vdofs;
      trace_space.GetFaceVDofs(face, face_vdofs);
      if (face_transformations == nullptr ||
          face_transformations->Face == nullptr || face_element == nullptr ||
          face_element->GetDof() != scalar::face_nodes ||
          face_vdofs.Size() !=
              scalar::trace_components * scalar::face_nodes)
        return false;
      mfem::Vector face_shape(scalar::face_nodes);
      for (int point = 0; point < face_rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& face_point = face_rule.IntPoint(point);
        face_element->CalcShape(face_point, face_shape);
        double trace_value[scalar::state_components]{};
        for (int component = 0; component < scalar::state_components;
             ++component)
          for (int node = 0; node < scalar::face_nodes; ++node)
          {
            const int encoded = face_vdofs[
                component * scalar::face_nodes + node];
            const int sign = encoded < 0 ? -1 : 1;
            const int local_dof = encoded < 0 ? -1 - encoded : encoded;
            const int true_dof = trace_space.GetLocalTDofNumber(local_dof);
            if (true_dof < 0 || true_dof >=
                    static_cast<int>(state.uhat.size()))
              return false;
            trace_value[component] +=
                sign * state.uhat[static_cast<std::size_t>(true_dof)] *
                    face_shape[node];
          }
        if (!AuditPrimitive(trace_value, audit)) { return false; }
        ++audit.trace_points;

        face_transformations->SetAllIntPoints(&face_point);
        const int elements[2]{face_transformations->Elem1No,
                              face_transformations->Elem2No};
        for (int side = 0; side < 2; ++side)
        {
          if (elements[side] < 0) { continue; }
          const mfem::IntegrationPoint& element_point = side == 0
              ? face_transformations->GetElement1IntPoint()
              : face_transformations->GetElement2IntPoint();
          const mfem::FiniteElement* finite_element =
              volume_space.GetFE(elements[side]);
          if (finite_element == nullptr) { return false; }
          mfem::Vector shape(scalar::volume_nodes);
          finite_element->CalcShape(element_point, shape);
          double side_value[scalar::state_components]{};
          for (int component = 0; component < scalar::state_components;
               ++component)
            for (int node = 0; node < scalar::volume_nodes; ++node)
              side_value[component] += state.u[
                  (elements[side] * scalar::state_components + component) *
                      scalar::volume_nodes + node] * shape[node];
          if (!AuditPrimitive(side_value, audit)) { return false; }
          ++audit.side_points;
        }
      }
    }
    return audit.volume_points > 0 && audit.side_points > 0 &&
        audit.trace_points > 0;
  }
  catch (...) { return false; }
}

inline constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
inline constexpr std::uint64_t fnv_prime = 1099511628211ULL;

inline void MixBytes(std::uint64_t& hash, const void* data,
                     const std::size_t size) noexcept
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index)
  {
    hash ^= bytes[index];
    hash *= fnv_prime;
  }
}

template <typename T>
void MixScalar(std::uint64_t& hash, const T value) noexcept
{
  MixBytes(hash, &value, sizeof(value));
}

inline std::uint64_t IndependentEndpointToken(
    const scalar::cuda_w8a::TransferEventV1& event) noexcept
{
  std::uint64_t token = fnv_offset;
  MixScalar(token, event.direction);
  MixScalar(token, event.category);
  MixScalar(token, event.site);
  MixScalar(token, event.context.attempt_sequence);
  MixScalar(token, event.context.target_step);
  MixScalar(token, event.context.generation);
  MixScalar(token, event.context.rank);
  MixScalar(token, event.context.phase);
  MixScalar(token, event.ordinal);
  MixScalar(token, event.source_token);
  MixScalar(token, event.destination_token);
  MixScalar(token, event.bytes);
  MixScalar(token, event.source_capacity);
  MixScalar(token, event.destination_capacity);
  MixScalar(token, event.success);
  return token;
}

inline bool AddTransfer(
    scalar::cuda_w8a::CudaTransferTotals& totals,
    const scalar::cuda_w8a::TransferEventV1& event) noexcept
{
  if (totals.event_count == std::numeric_limits<std::uint64_t>::max())
    return false;
  ++totals.event_count;
  totals.complete = totals.complete && event.success;
  if (!event.success) { return false; }
  std::uint64_t* endpoint = nullptr;
  using Direction = scalar::cuda_w8a::CudaTransferDirection;
  if (event.direction == Direction::host_to_device)
    endpoint = &totals.host_to_device_bytes;
  else if (event.direction == Direction::device_to_host)
    endpoint = &totals.device_to_host_bytes;
  else if (event.direction == Direction::device_to_device)
    endpoint = &totals.device_to_device_bytes;
  else
    endpoint = &totals.mpi_staging_bytes;
  if (event.bytes > std::numeric_limits<std::uint64_t>::max() - *endpoint)
    return false;
  *endpoint += event.bytes;
  return true;
}

inline bool SameTransferTotals(
    const scalar::cuda_w8a::CudaTransferTotals& left,
    const scalar::cuda_w8a::CudaTransferTotals& right) noexcept
{
  return left.host_to_device_bytes == right.host_to_device_bytes &&
      left.device_to_host_bytes == right.device_to_host_bytes &&
      left.device_to_device_bytes == right.device_to_device_bytes &&
      left.mpi_staging_bytes == right.mpi_staging_bytes &&
      left.synchronization_events == right.synchronization_events &&
      left.event_count == right.event_count &&
      left.complete == right.complete;
}

inline bool ValidateTransferRoute(
    const scalar::cuda_w8a::TransferEventV1& event) noexcept
{
  using Category = scalar::cuda_w8a::CudaTransferCategory;
  using Direction = scalar::cuda_w8a::CudaTransferDirection;
  using Site = scalar::cuda_w8a::CudaTransferSite;
  if (!event.success || event.bytes == 0 || event.source_token == 0 ||
      event.destination_token == 0 || event.source_capacity != event.bytes ||
      event.destination_capacity != event.bytes || event.context.rank != 0)
  {
    return false;
  }
  if (event.context.attempt_sequence == 0)
  {
    if (event.context.target_step != 0 || event.context.generation != 0)
      return false;
  }
  else if (event.context.attempt_sequence != event.context.target_step ||
           event.context.target_step != event.context.generation ||
           event.context.target_step > scalar::persistent_required_steps)
  {
    return false;
  }
  if (event.direction == Direction::host_to_device &&
      event.category == Category::topology &&
      (event.site == Site::map_upload ||
       event.site == Site::quadrature_upload))
    return true;
  if (event.direction == Direction::host_to_device &&
      event.category == Category::scalar_diagnostic &&
      event.site == Site::scalar_status && event.bytes == sizeof(int))
    return true;
  if (event.direction == Direction::device_to_host &&
      event.category == Category::scalar_diagnostic &&
      event.site == Site::scalar_status &&
      (event.bytes == sizeof(int) || event.bytes == 16 * sizeof(double)))
    return true;
  if (event.direction == Direction::device_to_host &&
      event.category == Category::scalar_diagnostic &&
      event.site == Site::state_hash &&
      (event.bytes == 7 * 32 || event.bytes == 2 * 32))
    return true;
  if (event.direction == Direction::device_to_host &&
      event.category == Category::oracle_export &&
      event.site == Site::oracle_export &&
      (event.bytes == scalar::persistent_interior_unknowns * sizeof(double) ||
       event.bytes == scalar::persistent_trace_unknowns * sizeof(double)))
    return true;
  if (event.direction == Direction::device_to_host &&
      event.category == Category::checkpoint_candidate_staging &&
      ((event.site == Site::checkpoint_stage_interior &&
        event.bytes == scalar::persistent_interior_unknowns * sizeof(double)) ||
       (event.site == Site::checkpoint_stage_trace &&
        event.bytes == scalar::persistent_trace_unknowns * sizeof(double))))
    return true;
  if (event.direction == Direction::host_to_device &&
      event.category == Category::checkpoint_restore_import &&
      ((event.site == Site::checkpoint_import_interior &&
        event.bytes == scalar::persistent_interior_unknowns * sizeof(double)) ||
       (event.site == Site::checkpoint_import_trace &&
        event.bytes == scalar::persistent_trace_unknowns * sizeof(double))))
    return true;
  return false;
}

enum class TransferCensusScenario : unsigned char
{
  coupled_operator_c1,
  persistent_trajectory_c2
};

struct TransferCensusExpectation final
{
  TransferCensusScenario scenario =
      TransferCensusScenario::persistent_trajectory_c2;
  int elements = 64;
};

inline constexpr TransferCensusExpectation c1_transfer_expectation{
    TransferCensusScenario::coupled_operator_c1, 9};
inline constexpr TransferCensusExpectation c2_transfer_expectation{
    TransferCensusScenario::persistent_trajectory_c2, 64};

inline bool ValidTransferCensusExpectation(
    const TransferCensusExpectation expectation) noexcept
{
  return (expectation.scenario ==
              TransferCensusScenario::coupled_operator_c1 &&
          expectation.elements == 9) ||
      (expectation.scenario ==
              TransferCensusScenario::persistent_trajectory_c2 &&
       expectation.elements == 64);
}

inline bool ValidateLaunchTransferCoherence(
    const scalar::cuda_w8a::CudaTransferLedgerV1& ledger,
    const scalar::ScalarCudaLaunchLedger& launches) noexcept
{
  using Category = scalar::cuda_w8a::CudaTransferCategory;
  using Direction = scalar::cuda_w8a::CudaTransferDirection;
  using Site = scalar::cuda_w8a::CudaTransferSite;
  if (ledger.schema != 1 ||
      ledger.capacity != scalar::cuda_w8a::maximum_transfer_events ||
      ledger.count != ledger.observed_count ||
      ledger.count != ledger.totals.event_count || ledger.overflow ||
      ledger.dropped_count != 0 || !ledger.totals.complete)
  {
    return false;
  }
  const auto same_context = [](const auto& left, const auto& right)
  {
    return left.attempt_sequence == right.attempt_sequence &&
        left.target_step == right.target_step &&
        left.generation == right.generation && left.rank == right.rank &&
        left.phase == right.phase;
  };
  std::uint64_t kernel_groups = 0;
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const auto& clear_status = ledger.events[index];
    if (clear_status.direction != Direction::host_to_device ||
        clear_status.category != Category::scalar_diagnostic ||
        clear_status.site != Site::scalar_status)
    {
      continue;
    }
    if (index + 3 >= ledger.count) { return false; }
    const auto& clear_architecture = ledger.events[index + 1];
    const auto& read_status = ledger.events[index + 2];
    const auto& read_architecture = ledger.events[index + 3];
    if (clear_status.bytes != sizeof(int) || !clear_status.success ||
        clear_architecture.direction != Direction::host_to_device ||
        clear_architecture.category != Category::scalar_diagnostic ||
        clear_architecture.site != Site::scalar_status ||
        clear_architecture.bytes != sizeof(int) ||
        !clear_architecture.success ||
        read_status.direction != Direction::device_to_host ||
        read_status.category != Category::scalar_diagnostic ||
        read_status.site != Site::scalar_status ||
        read_status.bytes != sizeof(int) || !read_status.success ||
        read_architecture.direction != Direction::device_to_host ||
        read_architecture.category != Category::scalar_diagnostic ||
        read_architecture.site != Site::scalar_status ||
        read_architecture.bytes != sizeof(int) ||
        !read_architecture.success ||
        !same_context(clear_status.context, clear_architecture.context) ||
        !same_context(clear_status.context, read_status.context) ||
        !same_context(clear_status.context, read_architecture.context))
    {
      return false;
    }
    ++kernel_groups;
    index += 3;
  }
  return kernel_groups == launches.assemble_launches +
      launches.condense_launches + launches.recover_launches;
}

inline bool ExpectedReductionReads(
    const scalar::ScalarCudaEvidence& evidence,
    const TransferCensusExpectation expectation,
    std::uint64_t& output) noexcept
{
  output = 0;
  if (!ValidTransferCensusExpectation(expectation) ||
      expectation.scenario !=
          TransferCensusScenario::persistent_trajectory_c2 ||
      evidence.accepted_steps != scalar::persistent_required_steps ||
      evidence.attempted_steps != scalar::persistent_required_steps ||
      evidence.attempt_count != scalar::persistent_required_steps ||
      evidence.retry_attempts != 0)
  {
    return false;
  }
  // Initialization performs one terminal state reduction.  Every accepted
  // C2 attempt then performs one entry and one candidate reduction, one
  // residual reduction per attempted Newton iteration, one recovered-solve
  // reduction per completed Newton update, and one reduction for each trial
  // whose device admissibility audit completed.
  output = 1;
  for (int attempt_index = 0; attempt_index < evidence.attempt_count;
       ++attempt_index)
  {
    const auto& attempt =
        evidence.attempts[static_cast<std::size_t>(attempt_index)];
    if (attempt.status != scalar::ScalarCudaStatus::accepted ||
        attempt.retry_index != 0 ||
        attempt.attempted_newton_iterations !=
            attempt.newton_iterations + 1 ||
        attempt.newton_iterations < 0 ||
        attempt.newton_iterations >= scalar::newton_maximum_iterations)
    {
      return false;
    }
    output += 2 +
        static_cast<std::uint64_t>(attempt.attempted_newton_iterations) +
        static_cast<std::uint64_t>(attempt.newton_iterations);
    for (int iteration = 0; iteration < attempt.newton_iterations;
         ++iteration)
    {
      const auto& newton =
          attempt.newton[static_cast<std::size_t>(iteration)];
      if (newton.iteration != iteration ||
          newton.accepted_line_search_index < 0 ||
          newton.accepted_line_search_index >= scalar::line_search_trials ||
          newton.trial_admissible[static_cast<std::size_t>(
              newton.accepted_line_search_index)] != 1)
      {
        return false;
      }
      for (int trial = 0; trial < scalar::line_search_trials; ++trial)
      {
        const std::uint8_t admissible =
            newton.trial_admissible[static_cast<std::size_t>(trial)];
        if (admissible > 1) { return false; }
        if (admissible != 0)
        {
          ++output;
        }
      }
    }
  }
  return true;
}

inline bool ValidateTransferLedger(
    const scalar::cuda_w8a::CudaTransferLedgerV1& ledger,
    const scalar::ScalarCudaEvidence& evidence,
    const TransferCensusExpectation expectation) noexcept
{
  using Category = scalar::cuda_w8a::CudaTransferCategory;
  using Direction = scalar::cuda_w8a::CudaTransferDirection;
  using Site = scalar::cuda_w8a::CudaTransferSite;
  if (!ValidTransferCensusExpectation(expectation) ||
      expectation.scenario !=
          TransferCensusScenario::persistent_trajectory_c2 ||
      ledger.schema != 1 ||
      ledger.capacity != scalar::cuda_w8a::maximum_transfer_events ||
      ledger.count == 0 || ledger.count != ledger.observed_count ||
      ledger.count != ledger.totals.event_count || ledger.overflow ||
      ledger.dropped_count != 0 || !ledger.totals.complete)
  {
    return false;
  }
  if (ledger.count < 20) { return false; }
  const std::uint64_t elements =
      static_cast<std::uint64_t>(expectation.elements);
  const std::uint64_t true_trace = 20 * elements;
  const std::uint64_t volume_points = 49 * elements;
  const std::uint64_t face_points = 28 * elements;
  const std::array<std::uint64_t, 20> upload_bytes{{
      elements * scalar::local_trace_unknowns * sizeof(int),
      elements * scalar::local_trace_unknowns * sizeof(int),
      elements * scalar::local_trace_unknowns *
          scalar::local_trace_unknowns * sizeof(int),
      (true_trace + 1) * sizeof(int),
      true_trace * 70 * sizeof(int),
      volume_points * sizeof(double),
      volume_points * scalar::volume_nodes * sizeof(double),
      volume_points * scalar::volume_nodes * scalar::dimension *
          sizeof(double),
      volume_points * sizeof(double),
      volume_points * sizeof(double),
      face_points * sizeof(double),
      elements * 4 * sizeof(double),
      face_points * scalar::volume_nodes * sizeof(double),
      face_points * scalar::face_nodes * sizeof(double),
      face_points * scalar::dimension * sizeof(double),
      face_points * sizeof(double),
      face_points * sizeof(double),
      elements * scalar::volume_nodes * sizeof(double),
      elements * scalar::volume_nodes * sizeof(double),
      elements * 4 * scalar::face_nodes * sizeof(double)}};
  for (std::size_t index = 0; index < upload_bytes.size(); ++index)
  {
    const auto& event = ledger.events[index];
    const Site expected_site = index < 5 ? Site::map_upload
                                         : Site::quadrature_upload;
    if (event.direction != Direction::host_to_device ||
        event.category != Category::topology ||
        event.site != expected_site || event.bytes != upload_bytes[index] ||
        event.source_capacity != upload_bytes[index] ||
        event.destination_capacity != upload_bytes[index] ||
        event.context.attempt_sequence != 0 ||
        event.context.target_step != 0 || event.context.generation != 0 ||
        event.context.rank != 0 || !event.success)
    {
      return false;
    }
  }

  scalar::cuda_w8a::CudaTransferTotals totals{};
  std::uint64_t digest = fnv_offset;
  std::uint64_t kernel_groups = 0;
  std::uint64_t reduction_reads = 0;
  std::uint64_t state_hash_reads = 0;
  std::uint64_t topology_hash_reads = 0;
  std::uint64_t oracle_pairs = 0;
  std::uint64_t checkpoint_pairs = 0;
  std::uint64_t restore_pairs = 0;
  std::uint64_t expected_reduction_reads = 0;
  if (!ExpectedReductionReads(evidence, expectation,
                              expected_reduction_reads))
  {
    return false;
  }
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const auto& event = ledger.events[index];
    if (event.ordinal != index + 1 || !ValidateTransferRoute(event) ||
        event.endpoint_token != IndependentEndpointToken(event) ||
        !AddTransfer(totals, event))
    {
      return false;
    }
    digest ^= event.endpoint_token;
    digest *= fnv_prime;
  }

  const auto same_context = [](const auto& left, const auto& right)
  {
    return left.attempt_sequence == right.attempt_sequence &&
        left.target_step == right.target_step &&
        left.generation == right.generation && left.rank == right.rank &&
        left.phase == right.phase;
  };
  bool kernel_payload_open = false;
  scalar::cuda_w8a::CudaPolicyContext kernel_payload_context{};
  for (std::size_t index = 20; index < ledger.count;)
  {
    const auto& event = ledger.events[index];
    if (event.direction == Direction::host_to_device &&
        event.category == Category::scalar_diagnostic &&
        event.site == Site::scalar_status)
    {
      if (index + 3 >= ledger.count) { return false; }
      const auto& clear_architecture = ledger.events[index + 1];
      const auto& read_status = ledger.events[index + 2];
      const auto& read_architecture = ledger.events[index + 3];
      if (event.bytes != sizeof(int) ||
          clear_architecture.direction != Direction::host_to_device ||
          clear_architecture.category != Category::scalar_diagnostic ||
          clear_architecture.site != Site::scalar_status ||
          clear_architecture.bytes != sizeof(int) ||
          read_status.direction != Direction::device_to_host ||
          read_status.category != Category::scalar_diagnostic ||
          read_status.site != Site::scalar_status ||
          read_status.bytes != sizeof(int) ||
          read_architecture.direction != Direction::device_to_host ||
          read_architecture.category != Category::scalar_diagnostic ||
          read_architecture.site != Site::scalar_status ||
          read_architecture.bytes != sizeof(int) ||
          !same_context(event.context, clear_architecture.context) ||
          !same_context(event.context, read_status.context) ||
          !same_context(event.context, read_architecture.context))
      {
        return false;
      }
      ++kernel_groups;
      kernel_payload_open = true;
      kernel_payload_context = event.context;
      index += 4;
      continue;
    }
    if (event.direction == Direction::device_to_host &&
        event.category == Category::scalar_diagnostic &&
        event.site == Site::scalar_status &&
        event.bytes == 16 * sizeof(double))
    {
      if (!kernel_payload_open ||
          !same_context(event.context, kernel_payload_context))
      {
        return false;
      }
      ++reduction_reads;
      kernel_payload_open = false;
      kernel_payload_context = {};
      ++index;
      continue;
    }
    if (event.direction == Direction::device_to_host &&
        event.category == Category::scalar_diagnostic &&
        event.site == Site::state_hash)
    {
      if (!kernel_payload_open ||
          !same_context(event.context, kernel_payload_context))
      {
        return false;
      }
      if (event.bytes == 7 * 32) { ++state_hash_reads; }
      else if (event.bytes == 2 * 32) { ++topology_hash_reads; }
      else { return false; }
      kernel_payload_open = false;
      kernel_payload_context = {};
      ++index;
      continue;
    }
    const auto exact_pair = [&](const Direction direction,
                                const Category category,
                                const Site first_site,
                                const Site second_site,
                                const std::uint64_t first_bytes,
                                const std::uint64_t second_bytes)
    {
      if (index + 1 >= ledger.count) { return false; }
      const auto& second = ledger.events[index + 1];
      return event.direction == direction && event.category == category &&
          event.site == first_site && event.bytes == first_bytes &&
          second.direction == direction && second.category == category &&
          second.site == second_site && second.bytes == second_bytes &&
          same_context(event.context, second.context);
    };
    if (exact_pair(Direction::device_to_host, Category::oracle_export,
                   Site::oracle_export, Site::oracle_export,
                   scalar::persistent_interior_unknowns * sizeof(double),
                   scalar::persistent_trace_unknowns * sizeof(double)))
    {
      ++oracle_pairs;
      kernel_payload_open = false;
      kernel_payload_context = {};
      index += 2;
      continue;
    }
    if (exact_pair(Direction::device_to_host,
                   Category::checkpoint_candidate_staging,
                   Site::checkpoint_stage_interior,
                   Site::checkpoint_stage_trace,
                   scalar::persistent_interior_unknowns * sizeof(double),
                   scalar::persistent_trace_unknowns * sizeof(double)))
    {
      if (!kernel_payload_open ||
          !same_context(event.context, kernel_payload_context))
      {
        return false;
      }
      ++checkpoint_pairs;
      kernel_payload_open = false;
      kernel_payload_context = {};
      index += 2;
      continue;
    }
    if (exact_pair(Direction::host_to_device,
                   Category::checkpoint_restore_import,
                   Site::checkpoint_import_interior,
                   Site::checkpoint_import_trace,
                   scalar::persistent_interior_unknowns * sizeof(double),
                   scalar::persistent_trace_unknowns * sizeof(double)))
    {
      ++restore_pairs;
      kernel_payload_open = false;
      kernel_payload_context = {};
      index += 2;
      continue;
    }
    return false;
  }
  const std::uint64_t recorded_launches = evidence.launches.assemble_launches +
      evidence.launches.condense_launches +
      evidence.launches.recover_launches;
  if (kernel_groups != recorded_launches ||
      reduction_reads != expected_reduction_reads ||
      state_hash_reads != 2 * scalar::persistent_required_steps ||
      topology_hash_reads != scalar::persistent_required_steps + 1 ||
      oracle_pairs != scalar::persistent_required_steps + 2 ||
      checkpoint_pairs != 1 || restore_pairs != 0)
  {
    return false;
  }
  totals.synchronization_events = 1 + kernel_groups +
      expected_reduction_reads +
      state_hash_reads + topology_hash_reads + oracle_pairs +
      checkpoint_pairs;
  return digest == ledger.ordered_digest &&
      SameTransferTotals(totals, ledger.totals);
}

inline bool ValidateTopologyIdentity(
    const scalar::ScalarCudaReuseEvidence& evidence) noexcept
{
  return evidence.device_topology_matches &&
      evidence.host_topology_hash != ::dg::hdg::runtime::StateHash{} &&
      evidence.host_quadrature_hash != ::dg::hdg::runtime::StateHash{} &&
      evidence.host_topology_hash == evidence.device_topology_hash &&
      evidence.host_quadrature_hash == evidence.device_quadrature_hash;
}

inline std::uint64_t IndependentAllocationDigest(
    const scalar::cuda_w8a::AllocationEventV1& event) noexcept
{
  std::uint64_t digest = fnv_offset;
  MixScalar(digest, event.ordinal);
  MixScalar(digest, event.site);
  MixScalar(digest, event.context.attempt_sequence);
  MixScalar(digest, event.context.target_step);
  MixScalar(digest, event.context.generation);
  MixScalar(digest, event.context.rank);
  MixScalar(digest, event.context.phase);
  MixScalar(digest, event.bytes);
  MixScalar(digest, event.alignment);
  MixScalar(digest, event.storage_token);
  MixScalar(digest, event.generation);
  MixScalar(digest, event.allocation);
  MixScalar(digest, event.success);
  return digest;
}

inline bool ValidateAllocationLedger(
    const scalar::cuda_w8a::CudaAllocationLedgerV1& ledger,
    const scalar::ScalarCudaAllocationEvidence& evidence) noexcept
{
  using Site = scalar::cuda_w8a::CudaAllocationSite;
  if (ledger.schema != 1 ||
      ledger.capacity != scalar::cuda_w8a::maximum_allocation_events ||
      ledger.count != 4 || ledger.observed_count != 4 || ledger.overflow ||
      ledger.dropped_count != 0)
    return false;
  std::uint64_t digest = fnv_offset;
  for (std::size_t index = 0; index < ledger.count; ++index)
  {
    const auto& event = ledger.events[index];
    if (event.ordinal != index + 1 || !event.success || event.bytes == 0 ||
        event.capacity != event.bytes || event.alignment == 0 ||
        event.storage_token == 0 || event.generation == 0 ||
        event.context.rank != 0 || event.context.attempt_sequence != 0 ||
        event.context.target_step != 0 || event.context.generation != 0)
      return false;
    const std::uint64_t event_digest = IndependentAllocationDigest(event);
    digest ^= event_digest;
    digest *= fnv_prime;
  }
  const auto& probe_allocate = ledger.events[0];
  const auto& probe_deallocate = ledger.events[1];
  const auto& doubles = ledger.events[2];
  const auto& integers = ledger.events[3];
  return probe_allocate.site == Site::mfem_device_pool_probe &&
      probe_allocate.allocation &&
      probe_deallocate.site == Site::mfem_device_pool_probe &&
      !probe_deallocate.allocation &&
      probe_allocate.storage_token == probe_deallocate.storage_token &&
      probe_allocate.generation == probe_deallocate.generation &&
      doubles.site == Site::persistent_double_arena && doubles.allocation &&
      doubles.storage_token == evidence.double_arena &&
      integers.site == Site::persistent_integer_arena &&
      integers.allocation && integers.storage_token == evidence.integer_arena &&
      digest == ledger.ordered_digest;
}

template <typename Left, typename Right>
double RelativeL2(const Left& left, const Right& right) noexcept
{
  if (left.size() != right.size()) { return INFINITY; }
  long double difference = 0.0L;
  long double reference = 0.0L;
  for (std::size_t i = 0; i < left.size(); ++i)
  {
    const long double lhs = left[i];
    const long double rhs = right[i];
    const long double delta = lhs - rhs;
    difference += delta * delta;
    reference += lhs * lhs;
  }
  return static_cast<double>(
      std::sqrt(difference) / std::max(1.0L, std::sqrt(reference)));
}

template <typename Left, typename Right>
bool CompareArray(const std::string_view name, const Left& reference,
                  const Right& observed, const double tolerance,
                  ComparisonEvidence& evidence)
{
  ++evidence.checked_arrays;
  const double error = RelativeL2(reference, observed);
  evidence.maximum_relative_l2 =
      std::max(evidence.maximum_relative_l2, error);
  if (!(error <= tolerance))
  {
    evidence.failures.emplace_back(name);
    return false;
  }
  return true;
}

inline bool SameStateBytes(const scalar::PublishedState& left,
                           const scalar::PublishedState& right) noexcept
{
  return std::memcmp(left.u.data(), right.u.data(),
                     left.u.size() * sizeof(double)) == 0 &&
      std::memcmp(left.sigma.data(), right.sigma.data(),
                  left.sigma.size() * sizeof(double)) == 0 &&
      std::memcmp(left.z.data(), right.z.data(),
                  left.z.size() * sizeof(double)) == 0 &&
      std::memcmp(left.uhat.data(), right.uhat.data(),
                  left.uhat.size() * sizeof(double)) == 0 &&
      std::memcmp(left.sigmahat.data(), right.sigmahat.data(),
                  left.sigmahat.size() * sizeof(double)) == 0;
}

inline bool CompareState(const scalar::PublishedState& reference,
                         const scalar::PublishedState& observed,
                         const double tolerance,
                         ComparisonEvidence& evidence)
{
  bool valid = true;
  valid = CompareArray("state.U", reference.u, observed.u, tolerance,
                       evidence) && valid;
  valid = CompareArray("state.Sigma", reference.sigma, observed.sigma,
                       tolerance, evidence) && valid;
  valid = CompareArray("state.z", reference.z, observed.z, tolerance,
                       evidence) && valid;
  valid = CompareArray("state.Uhat", reference.uhat, observed.uhat,
                       tolerance, evidence) && valid;
  valid = CompareArray("state.Sigmahat", reference.sigmahat,
                       observed.sigmahat, tolerance, evidence) && valid;
  return valid;
}

inline bool CompareOperator(
    const scalar::CoupledOperatorSnapshot& reference,
    const scalar::CoupledOperatorSnapshot& observed,
    const double tolerance, ComparisonEvidence& evidence)
{
  bool valid = reference.quadrature_order == observed.quadrature_order &&
      reference.repeat_quadrature_order == observed.repeat_quadrature_order &&
      reference.local_interior_unknowns ==
          observed.local_interior_unknowns &&
      reference.local_trace_unknowns == observed.local_trace_unknowns &&
      reference.true_trace_unknowns == observed.true_trace_unknowns &&
      reference.full_unknowns == observed.full_unknowns &&
      reference.elements.size() == observed.elements.size() &&
      reference.condensed.size() == observed.condensed.size();
  if (!valid) { evidence.failures.emplace_back("operator.dimensions"); }
  valid = CompareArray("operator.old_interior", reference.old_interior,
                       observed.old_interior, tolerance, evidence) && valid;
  valid = CompareArray("operator.old_trace", reference.old_true_trace,
                       observed.old_true_trace, tolerance, evidence) && valid;
  valid = CompareArray("operator.candidate_interior",
                       reference.candidate_interior,
                       observed.candidate_interior, tolerance, evidence) &&
      valid;
  valid = CompareArray("operator.candidate_trace",
                       reference.candidate_true_trace,
                       observed.candidate_true_trace, tolerance, evidence) &&
      valid;
  valid = CompareArray("operator.residual", reference.full_residual,
                       observed.full_residual, tolerance, evidence) && valid;
  valid = CompareArray("operator.jacobian", reference.full_jacobian,
                       observed.full_jacobian, tolerance, evidence) && valid;
  const std::size_t elements =
      std::min(reference.elements.size(), observed.elements.size());
  for (std::size_t element = 0; element < elements; ++element)
  {
    const auto& lhs = reference.elements[element];
    const auto& rhs = observed.elements[element];
    valid = lhs.element == rhs.element &&
        lhs.local_interior_unknowns == rhs.local_interior_unknowns &&
        lhs.local_trace_unknowns == rhs.local_trace_unknowns && valid;
    valid = CompareArray("element.A", lhs.a, rhs.a, tolerance, evidence) &&
        valid;
    valid = CompareArray("element.B", lhs.b, rhs.b, tolerance, evidence) &&
        valid;
    valid = CompareArray("element.C", lhs.c, rhs.c, tolerance, evidence) &&
        valid;
    valid = CompareArray("element.D", lhs.d, rhs.d, tolerance, evidence) &&
        valid;
    valid = CompareArray("element.local_residual", lhs.local_residual,
                         rhs.local_residual, tolerance, evidence) && valid;
    valid = CompareArray("element.trace_residual", lhs.trace_residual,
                         rhs.trace_residual, tolerance, evidence) && valid;
  }
  const std::size_t condensed =
      std::min(reference.condensed.size(), observed.condensed.size());
  for (std::size_t element = 0; element < condensed; ++element)
  {
    const auto& lhs = reference.condensed[element];
    const auto& rhs = observed.condensed[element];
    valid = lhs.element == rhs.element &&
        lhs.local_trace_unknowns == rhs.local_trace_unknowns && valid;
    valid = CompareArray("element.schur", lhs.schur, rhs.schur, tolerance,
                         evidence) && valid;
    valid = CompareArray("element.condensed_rhs", lhs.right_hand_side,
                         rhs.right_hand_side, tolerance, evidence) && valid;
  }
  valid = CompareArray("operator.true_increment",
                       reference.true_trace_increment,
                       observed.true_trace_increment, tolerance, evidence) &&
      valid;
  valid = CompareArray("operator.restricted_increment",
                       reference.element_trace_increment,
                       observed.element_trace_increment, tolerance,
                       evidence) && valid;
  valid = CompareArray("operator.recovered_increment",
                       reference.recovered_interior_increment,
                       observed.recovered_interior_increment, tolerance,
                       evidence) && valid;
  valid = CompareArray("operator.full_increment", reference.full_increment,
                       observed.full_increment, tolerance, evidence) && valid;
  return valid;
}

template <typename... Arrays>
StateHash HashArrays(const Arrays&... arrays) noexcept
{
  Sha256 hash;
  const bool updated =
      (hash.Update(arrays.data(), arrays.size() * sizeof(double)) && ...);
  Sha256Digest digest{};
  return updated && hash.Finalize(digest) ? StateHash{digest} : StateHash{};
}

inline scalar::StateHashes HashState(
    const scalar::PublishedState& state) noexcept
{
  return {HashArrays(state.u), HashArrays(state.sigma), HashArrays(state.z),
          HashArrays(state.uhat), HashArrays(state.sigmahat),
          HashArrays(state.u, state.sigma, state.z),
          HashArrays(state.uhat, state.sigmahat)};
}

inline bool SameHashes(const scalar::StateHashes& left,
                       const scalar::StateHashes& right) noexcept
{
  return left.u == right.u && left.sigma == right.sigma &&
      left.z == right.z && left.uhat == right.uhat &&
      left.sigmahat == right.sigmahat &&
      left.interior == right.interior && left.trace == right.trace;
}

inline bool NonzeroHashes(const scalar::StateHashes& hashes) noexcept
{
  return hashes.u != StateHash{} && hashes.sigma != StateHash{} &&
      hashes.z != StateHash{} && hashes.uhat != StateHash{} &&
      hashes.sigmahat != StateHash{} && hashes.interior != StateHash{} &&
      hashes.trace != StateHash{};
}

inline scalar::PublishedState DecodeCheckpoint(
    const scalar::ScalarCudaCheckpoint& checkpoint) noexcept
{
  scalar::PublishedState state{};
  const int scalar_nodes =
      scalar::persistent_elements * scalar::volume_nodes;
  try
  {
    state.u.resize(static_cast<std::size_t>(scalar::state_components) *
                   scalar_nodes);
    state.sigma.resize(scalar_nodes);
    state.z.resize(static_cast<std::size_t>(scalar::flux_components) *
                   scalar_nodes);
    state.uhat.resize(static_cast<std::size_t>(scalar::state_components) *
                      scalar::persistent_scalar_trace_nodes);
    state.sigmahat.resize(scalar::persistent_scalar_trace_nodes);
  }
  catch (...)
  {
    return {};
  }
  for (int element = 0; element < scalar::persistent_elements; ++element)
    for (int node = 0; node < scalar::volume_nodes; ++node)
    {
      const int scalar_node = element * scalar::volume_nodes + node;
      for (int component = 0; component < scalar::state_components;
           ++component)
      {
        state.u[(element * scalar::state_components + component) *
                    scalar::volume_nodes + node] =
            checkpoint.interior_component_slabs[
                component * scalar_nodes + scalar_node];
      }
      state.sigma[scalar_node] = checkpoint.interior_component_slabs[
          scalar::state_components * scalar_nodes +
          scalar_node];
      for (int component = 0; component < scalar::flux_components;
           ++component)
      {
        state.z[(element * scalar::flux_components + component) *
                    scalar::volume_nodes + node] =
            checkpoint.interior_component_slabs[
                (scalar::state_components + 1 + component) *
                    scalar_nodes + scalar_node];
      }
    }
  std::copy_n(checkpoint.trace_component_slabs.begin(), state.uhat.size(),
              state.uhat.begin());
  std::copy_n(checkpoint.trace_component_slabs.begin() + state.uhat.size(),
              state.sigmahat.size(), state.sigmahat.begin());
  return state;
}

inline bool RehashCheckpoint(
    scalar::ScalarCudaCheckpoint& checkpoint) noexcept
{
  const scalar::PublishedState state = DecodeCheckpoint(checkpoint);
  checkpoint.raw_field_hashes = {
      HashArrays(state.u), HashArrays(state.sigma), HashArrays(state.z),
      HashArrays(state.uhat), HashArrays(state.sigmahat),
      HashArrays(state.u, state.sigma, state.z),
      HashArrays(state.uhat, state.sigmahat)};
  checkpoint.descriptor.interior_hash =
      checkpoint.raw_field_hashes.interior;
  checkpoint.descriptor.trace_hash = checkpoint.raw_field_hashes.trace;

  Sha256 hash;
  const auto text = [&hash](const std::string_view value)
  {
    const std::uint64_t size = value.size();
    return hash.Update(&size, sizeof(size)) &&
        hash.Update(value.data(), value.size());
  };
  const auto& descriptor = checkpoint.descriptor;
  const bool valid = text(checkpoint.projection.view()) &&
      text(checkpoint.tuple.view()) && text(checkpoint.scenario.view()) &&
      text(checkpoint.layout.view()) &&
      hash.Update(&descriptor.schema_version,
                  sizeof(descriptor.schema_version)) &&
      hash.Update(&descriptor.successful_commit,
                  sizeof(descriptor.successful_commit)) &&
      hash.Update(&descriptor.restart_mode, sizeof(descriptor.restart_mode)) &&
      hash.Update(&descriptor.tuple, sizeof(descriptor.tuple)) &&
      hash.Update(&descriptor.mfem_pin, sizeof(descriptor.mfem_pin)) &&
      hash.Update(descriptor.mesh_hash.bytes.data(),
                  descriptor.mesh_hash.bytes.size()) &&
      hash.Update(descriptor.partition_hash.bytes.data(),
                  descriptor.partition_hash.bytes.size()) &&
      hash.Update(descriptor.topology_hash.bytes.data(),
                  descriptor.topology_hash.bytes.size()) &&
      hash.Update(&descriptor.rank_count, sizeof(descriptor.rank_count)) &&
      hash.Update(&descriptor.polynomial_order,
                  sizeof(descriptor.polynomial_order)) &&
      hash.Update(&descriptor.layout, sizeof(descriptor.layout)) &&
      hash.Update(&descriptor.precision, sizeof(descriptor.precision)) &&
      hash.Update(&descriptor.cursor, sizeof(descriptor.cursor)) &&
      hash.Update(&descriptor.dt, sizeof(descriptor.dt)) &&
      hash.Update(&descriptor.coordinates.time,
                  sizeof(descriptor.coordinates.time)) &&
      hash.Update(&descriptor.coordinates.step,
                  sizeof(descriptor.coordinates.step)) &&
      hash.Update(&descriptor.coordinates.generation,
                  sizeof(descriptor.coordinates.generation)) &&
      hash.Update(&descriptor.coordinates.cursor,
                  sizeof(descriptor.coordinates.cursor)) &&
      hash.Update(&descriptor.coordinates.dt,
                  sizeof(descriptor.coordinates.dt)) &&
      hash.Update(&descriptor.coordinate_fingerprint,
                  sizeof(descriptor.coordinate_fingerprint)) &&
      hash.Update(descriptor.interior_hash.bytes.data(),
                  descriptor.interior_hash.bytes.size()) &&
      hash.Update(descriptor.trace_hash.bytes.data(),
                  descriptor.trace_hash.bytes.size()) &&
      hash.Update(checkpoint.raw_field_hashes.u.bytes.data(),
                  checkpoint.raw_field_hashes.u.bytes.size()) &&
      hash.Update(checkpoint.raw_field_hashes.sigma.bytes.data(),
                  checkpoint.raw_field_hashes.sigma.bytes.size()) &&
      hash.Update(checkpoint.raw_field_hashes.z.bytes.data(),
                  checkpoint.raw_field_hashes.z.bytes.size()) &&
      hash.Update(checkpoint.raw_field_hashes.uhat.bytes.data(),
                  checkpoint.raw_field_hashes.uhat.bytes.size()) &&
      hash.Update(checkpoint.raw_field_hashes.sigmahat.bytes.data(),
                  checkpoint.raw_field_hashes.sigmahat.bytes.size()) &&
      hash.Update(checkpoint.interior_component_slabs.data(),
                  checkpoint.interior_component_slabs.size() *
                      sizeof(double)) &&
      hash.Update(checkpoint.trace_component_slabs.data(),
                  checkpoint.trace_component_slabs.size() * sizeof(double));
  Sha256Digest digest{};
  if (!valid || !hash.Finalize(digest)) { return false; }
  checkpoint.canonical_manifest_hash = StateHash{digest};
  return true;
}

} // namespace dg_igr_hdg_sc25_cuda_test
