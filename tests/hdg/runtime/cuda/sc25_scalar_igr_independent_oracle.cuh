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
    mfem::ParFiniteElementSpace uhat_space(
        &mesh, &trace_collection, scalar::state_components,
        mfem::Ordering::byNODES);
    if (mesh.GetNE() != scalar::persistent_elements ||
        volume_space.GetVSize() !=
            scalar::persistent_elements * scalar::volume_nodes ||
        uhat_space.GetTrueVSize() != scalar::cpu_sc25::persistent_uhat_unknowns)
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
      const mfem::FiniteElement* face_element = uhat_space.GetFaceElement(face);
      mfem::Array<int> face_vdofs;
      uhat_space.GetFaceVDofs(face, face_vdofs);
      if (face_transformations == nullptr ||
          face_transformations->Face == nullptr || face_element == nullptr ||
          face_element->GetDof() != scalar::face_nodes ||
          face_vdofs.Size() !=
              scalar::state_components * scalar::face_nodes)
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
            const int true_dof = uhat_space.GetLocalTDofNumber(local_dof);
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
