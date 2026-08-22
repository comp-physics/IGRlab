#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

constexpr double qualification_density_gate = 1.0e-8;

struct CudaPreparedIgrConservationCapture
{
  int elements = 0;
  int scalar_interior_dofs = 0;
  int scalar_face_dofs = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  int volume_points = 0;
  int face_points = 0;
  int order = 0;
  int assembly_order = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double dt = 0.0;
  double lifecycle_maximum_error =
      std::numeric_limits<double>::infinity();
  double lifecycle_maximum_relative_error =
      std::numeric_limits<double>::infinity();
  std::vector<double> previous_interior{};
  std::vector<double> current_interior{};
  std::vector<double> current_local_trace{};
  std::array<double, 11> production_local{};
};

// This accessor is defined only in the qualification translation unit.  The
// production engine and generic facade grant it private friendship but ship no
// callable test API or test-only device kernel.
struct CudaQualificationTestAccess
{
  template <typename Engine>
  static bool VerifyAssemblerJacobian(
      Engine& engine, const double relative_tolerance,
      std::array<double, 6>& relative_errors) noexcept
  {
    relative_errors.fill(std::numeric_limits<double>::infinity());
    return engine.initialized_ && !engine.pending_.active &&
        engine.config_.use_generic_numerical_policy &&
        engine.config_.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::igr &&
        engine.numerics_.AuditBorrowedEulerAssemblerJacobian(
            relative_tolerance, relative_errors);
  }

  template <typename Engine>
  static bool CapturePreparedIgrConservation(
      Engine& engine, CudaPreparedIgrConservationCapture& capture) noexcept
  {
    capture = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    bool local_valid = engine.initialized_ && engine.pending_.active &&
        engine.pending_.staging &&
        engine.config_.use_generic_numerical_policy &&
        engine.config_.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::igr && shape.igr_active &&
        std::isfinite(engine.pending_.attempt.dt) &&
        engine.pending_.attempt.dt > 0.0 &&
        numerics.view_.work_interior != nullptr &&
        numerics.view_.old_interior != nullptr &&
        numerics.view_.current_local_trace != nullptr &&
        numerics.view_.scalar != nullptr &&
        shape.interior_components == igr_interior_components &&
        shape.trace_components == igr_trace_components &&
        shape.scalar_interior_dofs > 0 && shape.scalar_face_dofs > 0 &&
        shape.interior_dofs ==
            shape.scalar_interior_dofs * igr_interior_components &&
        shape.trace_dofs == faces_per_element * shape.scalar_face_dofs *
            igr_trace_components;
    try
    {
      capture.current_interior.resize(
          static_cast<std::size_t>(shape.interior_count));
      capture.previous_interior.resize(capture.current_interior.size());
      capture.current_local_trace.resize(
          static_cast<std::size_t>(shape.elements) * shape.trace_dofs);
    }
    catch (...) { local_valid = false; }

    if (!numerics.CollectiveContinue(local_valid)) return false;
    const bool launched = CudaLaunchIgrConservation(
            numerics.view_, shape, engine.pending_.attempt.dt,
            numerics.view_.scalar) &&
        CudaTransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!numerics.CollectiveContinue(launched)) return false;

    const bool copied = CudaTransportPolicy::DeviceToHost(
        capture.current_interior.data(), numerics.view_.work_interior,
        capture.current_interior.size() * sizeof(double),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            capture.previous_interior.data(), numerics.view_.old_interior,
            capture.previous_interior.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            capture.current_local_trace.data(),
            numerics.view_.current_local_trace,
            capture.current_local_trace.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            capture.production_local.data(), numerics.view_.scalar,
            capture.production_local.size() * sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    capture.elements = shape.elements;
    capture.scalar_interior_dofs = shape.scalar_interior_dofs;
    capture.scalar_face_dofs = shape.scalar_face_dofs;
    capture.interior_dofs = shape.interior_dofs;
    capture.trace_dofs = shape.trace_dofs;
    capture.volume_points = shape.volume_points;
    capture.face_points = shape.face_points;
    capture.order = shape.order;
    capture.assembly_order = shape.assembly_order;
    capture.alpha = shape.igr_alpha;
    capture.beta = shape.igr_beta;
    capture.dt = engine.pending_.attempt.dt;
    capture.lifecycle_maximum_error =
        engine.pending_.numerical.maximum_conservation_error;
    capture.lifecycle_maximum_relative_error =
        engine.pending_.numerical.maximum_conservation_relative_error;
    local_valid = copied && capture.production_local[10] == 1.0 &&
        std::all_of(capture.production_local.begin(),
                    capture.production_local.end(),
                    [](const double value) { return std::isfinite(value); });
    return numerics.CollectiveContinue(local_valid);
  }

  static bool CapturePreparedIgrConservation(
      cuda::GenericEulerHdgCudaSession& session,
      CudaPreparedIgrConservationCapture& capture) noexcept
  {
    if (session.borrowed_rank1_)
      return CapturePreparedIgrConservation(
          *session.borrowed_rank1_, capture);
    if (session.borrowed_rank2_)
      return CapturePreparedIgrConservation(
          *session.borrowed_rank2_, capture);
    return false;
  }

  template <typename Engine>
  static bool VerifyIgrCouplingBlocks(
      Engine& engine, std::array<double, 2>& relative_norms) noexcept
  {
    relative_norms.fill(0.0);
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const std::size_t a_count = static_cast<std::size_t>(shape.elements) *
        static_cast<std::size_t>(shape.interior_dofs) *
        static_cast<std::size_t>(shape.interior_dofs);
    const std::size_t b_count = static_cast<std::size_t>(shape.elements) *
        static_cast<std::size_t>(shape.interior_dofs) *
        static_cast<std::size_t>(shape.trace_dofs);
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        shape.igr_active &&
        shape.interior_components == igr_interior_components &&
        shape.trace_components == igr_trace_components &&
        shape.scalar_interior_dofs > 0 && shape.scalar_face_dofs > 0 &&
        numerics.view_.a != nullptr && numerics.view_.b != nullptr;
    std::vector<double> a, b;
    try
    {
      a.resize(a_count);
      b.resize(b_count);
    }
    catch (...) { local_valid = false; }
    const bool copied = local_valid &&
        CudaTransportPolicy::DeviceToHost(
            a.data(), numerics.view_.a, a.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            b.data(), numerics.view_.b, b.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    local_valid = local_valid && copied;

    // Norm 0 is the production Euler residual's sensitivity to IGR unknowns;
    // norm 1 is the auxiliary residual's sensitivity to Euler unknowns.  Both
    // directions must survive the live 7x5 assembler, independently of any
    // nonlinear-solver stopping tolerance.
    std::array<double, 3> local{};
    for (int element = 0; local_valid && element < shape.elements; ++element)
    {
      const std::size_t a_offset = static_cast<std::size_t>(element) *
          static_cast<std::size_t>(shape.interior_dofs) *
          static_cast<std::size_t>(shape.interior_dofs);
      for (int row = 0; row < shape.interior_dofs; ++row)
        for (int column = 0; column < shape.interior_dofs; ++column)
        {
          const double value = a[a_offset +
              static_cast<std::size_t>(row) * shape.interior_dofs + column];
          local_valid = std::isfinite(value);
          local[2] += value * value;
          const int row_component = row / shape.scalar_interior_dofs;
          const int column_component = column /
              shape.scalar_interior_dofs;
          if (row_component < conserved_components &&
              column_component >= conserved_components)
            local[0] += value * value;
          if (row_component >= conserved_components &&
              column_component < conserved_components)
            local[1] += value * value;
        }

      const std::size_t b_offset = static_cast<std::size_t>(element) *
          static_cast<std::size_t>(shape.interior_dofs) *
          static_cast<std::size_t>(shape.trace_dofs);
      for (int row = 0; row < shape.interior_dofs; ++row)
        for (int column = 0; column < shape.trace_dofs; ++column)
        {
          const double value = b[b_offset +
              static_cast<std::size_t>(row) * shape.trace_dofs + column];
          local_valid = std::isfinite(value);
          local[2] += value * value;
          const int row_component = row / shape.scalar_interior_dofs;
          const int face_column = column %
              (shape.trace_components * shape.scalar_face_dofs);
          const int column_component = face_column /
              shape.scalar_face_dofs;
          if (row_component < conserved_components &&
              column_component >= conserved_components)
            local[0] += value * value;
          if (row_component >= conserved_components &&
              column_component < conserved_components)
            local[1] += value * value;
        }
    }
    std::array<double, 3> global{};
    const bool reduced = MPI_Allreduce(
        local.data(), global.data(), static_cast<int>(global.size()),
        MPI_DOUBLE, MPI_SUM, engine.config_.borrowed_communicator) ==
        MPI_SUCCESS;
    const double denominator = reduced && global[2] > 0.0
        ? std::sqrt(global[2]) : 0.0;
    if (denominator > 0.0)
    {
      relative_norms[0] = std::sqrt(global[0]) / denominator;
      relative_norms[1] = std::sqrt(global[1]) / denominator;
    }
    return reduced && numerics.CollectiveContinue(
        local_valid && std::isfinite(relative_norms[0]) &&
        std::isfinite(relative_norms[1]) && relative_norms[0] > 1.0e-12 &&
        relative_norms[1] > 1.0e-12);
  }

  template <typename Engine>
  static bool VerifyBorrowedInitialTrueTrace(
      Engine& engine, const CudaBorrowedHostL2Input expected) noexcept
  {
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const bool shape_valid = shape.trace_components >= conserved_components &&
        shape.true_trace_count >= 0 && shape.trace_components != 0 &&
        shape.true_trace_count % shape.trace_components == 0;
    const std::size_t scalar_count = shape_valid
        ? static_cast<std::size_t>(shape.true_trace_count /
                                   shape.trace_components)
        : 0;
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        numerics.initialized_ && numerics.borrowed_host_inputs_ &&
        shape_valid && expected.coefficients != nullptr &&
        expected.scalar_count == scalar_count &&
        expected.components == conserved_components &&
        numerics.host_exact_true_trace_.size() ==
            static_cast<std::size_t>(shape.true_trace_count) &&
        numerics.committed_slot_.trace.coefficients != nullptr &&
        numerics.committed_slot_.trace.count ==
            static_cast<std::size_t>(shape.true_trace_count);
    std::vector<double> observed;
    try
    {
      observed.resize(static_cast<std::size_t>(shape.true_trace_count));
    }
    catch (...) { local_valid = false; }
    const bool copied = local_valid && CudaTransportPolicy::DeviceToHost(
        observed.data(), numerics.committed_slot_.trace.coefficients,
        observed.size() * sizeof(double),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    local_valid = local_valid && copied;
    for (int component = 0;
         local_valid && component < conserved_components; ++component)
      for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
      {
        const std::size_t index = scalar_count *
            static_cast<std::size_t>(component) + scalar;
        local_valid = std::isfinite(expected.coefficients[index]) &&
            numerics.host_exact_true_trace_[index] ==
                expected.coefficients[index] &&
            observed[index] == expected.coefficients[index];
      }
    for (int component = conserved_components;
         local_valid && component < shape.trace_components; ++component)
      for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
      {
        const std::size_t index = scalar_count *
            static_cast<std::size_t>(component) + scalar;
        const double host_value = numerics.host_exact_true_trace_[index];
        const double device_value = observed[index];
        local_valid = host_value == 0.0 && !std::signbit(host_value) &&
            device_value == 0.0 && !std::signbit(device_value);
      }
    return numerics.CollectiveContinue(local_valid);
  }

  static int LoweredBoundaryKind(
      const generic::EulerBoundaryPolicyKind kind) noexcept
  {
    switch (kind)
    {
    case generic::EulerBoundaryPolicyKind::supersonic_inlet:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::supersonic_inlet);
    case generic::EulerBoundaryPolicyKind::subsonic_inlet:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::subsonic_inlet);
    case generic::EulerBoundaryPolicyKind::supersonic_outlet:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::supersonic_outlet);
    case generic::EulerBoundaryPolicyKind::subsonic_outlet_pressure:
      return static_cast<int>(
          CudaStaticBoundaryPolicyKind::subsonic_outlet_pressure);
    case generic::EulerBoundaryPolicyKind::far_field:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::far_field);
    case generic::EulerBoundaryPolicyKind::nonreflecting_outlet:
      return static_cast<int>(
          CudaStaticBoundaryPolicyKind::nonreflecting_outlet);
    case generic::EulerBoundaryPolicyKind::slip_wall:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::slip_wall);
    case generic::EulerBoundaryPolicyKind::periodic_topology:
      return 0;
    }
    return -1;
  }

  static bool SameDouble(const double left, const double right) noexcept
  {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
  }

  template <typename Engine>
  static bool VerifyBoundaryPolicyTable(
      Engine& engine, const generic::EulerBoundaryPolicy* const expected,
      const std::size_t expected_count, const double expected_gamma) noexcept
  {
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const std::size_t face_count = static_cast<std::size_t>(shape.elements) *
        static_cast<std::size_t>(faces_per_element);
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        numerics.initialized_ && numerics.mesh_ != nullptr &&
        shape.borrowed_static_boundary_policies && expected != nullptr &&
        expected_count == 7 && SameDouble(shape.boundary_gamma,
                                           expected_gamma) &&
        numerics.host_boundary_policy_kind_.size() == face_count &&
        numerics.host_boundary_pressure_.size() == face_count &&
        numerics.host_boundary_reference_.size() ==
            face_count * conserved_components &&
        numerics.view_.boundary_policy_kind != nullptr &&
        numerics.view_.boundary_pressure != nullptr &&
        numerics.view_.boundary_reference != nullptr;
    std::vector<int> device_kinds;
    std::vector<double> device_pressures, device_references;
    try
    {
      device_kinds.resize(face_count);
      device_pressures.resize(face_count);
      device_references.resize(face_count * conserved_components);
    }
    catch (...) { local_valid = false; }
    const bool copied = local_valid && CudaTransportPolicy::DeviceToHost(
        device_kinds.data(), numerics.view_.boundary_policy_kind,
        device_kinds.size() * sizeof(int),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            device_pressures.data(), numerics.view_.boundary_pressure,
            device_pressures.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            device_references.data(), numerics.view_.boundary_reference,
            device_references.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    local_valid = local_valid && copied;

    std::array<long long, 7> local_uses{};
    std::vector<unsigned char> shared;
    try
    {
      shared.resize(static_cast<std::size_t>(
          numerics.mesh_->GetNumFaces()), 0);
      for (int index = 0; index < numerics.mesh_->GetNSharedFaces(); ++index)
      {
        const int face = numerics.mesh_->GetSharedFace(index);
        if (face < 0 || face >= static_cast<int>(shared.size()))
          local_valid = false;
        else
          shared[static_cast<std::size_t>(face)] = 1;
      }
      for (int element = 0; local_valid && element < shape.elements; ++element)
      {
        mfem::Array<int> faces, orientations;
        numerics.mesh_->GetElementEdges(element, faces, orientations);
        local_valid = faces.Size() == faces_per_element;
        for (int local_face = 0;
             local_valid && local_face < faces.Size(); ++local_face)
        {
          const int face = faces[local_face];
          mfem::FaceElementTransformations* transformations =
              numerics.mesh_->GetFaceElementTransformations(face);
          const int boundary_element = face >= 0 &&
                  face < static_cast<int>(shared.size())
              ? numerics.mesh_->GetFaceToBdrElMap()[face] : -1;
          const bool physical = transformations != nullptr &&
              shared[static_cast<std::size_t>(face)] == 0 &&
              transformations->Elem2No < 0 && boundary_element >= 0;
          const std::size_t face_index = static_cast<std::size_t>(element) *
              faces_per_element + local_face;
          int expected_kind = 0;
          double expected_pressure = 0.0;
          std::array<double, conserved_components> expected_reference{};
          if (physical)
          {
            const int attribute =
                numerics.mesh_->GetBdrAttribute(boundary_element);
            local_valid = attribute >= 1 &&
                static_cast<std::size_t>(attribute) <= expected_count;
            if (!local_valid) break;
            const generic::EulerBoundaryPolicy& policy =
                expected[static_cast<std::size_t>(attribute - 1)];
            expected_kind = LoweredBoundaryKind(policy.kind);
            expected_pressure = policy.pressure_out;
            expected_reference = policy.reference_state;
            ++local_uses[static_cast<std::size_t>(attribute - 1)];
          }
          local_valid = expected_kind >= 0 &&
              numerics.host_boundary_policy_kind_[face_index] ==
                  expected_kind &&
              device_kinds[face_index] == expected_kind &&
              SameDouble(numerics.host_boundary_pressure_[face_index],
                         expected_pressure) &&
              SameDouble(device_pressures[face_index], expected_pressure);
          for (int component = 0;
               local_valid && component < conserved_components; ++component)
          {
            const std::size_t index = face_index * conserved_components +
                static_cast<std::size_t>(component);
            local_valid = SameDouble(
                    numerics.host_boundary_reference_[index],
                    expected_reference[static_cast<std::size_t>(component)]) &&
                SameDouble(
                    device_references[index],
                    expected_reference[static_cast<std::size_t>(component)]);
          }
        }
      }
    }
    catch (...) { local_valid = false; }

    std::array<long long, 7> global_uses{};
    const bool reduced = MPI_Allreduce(
        local_uses.data(), global_uses.data(),
        static_cast<int>(global_uses.size()), MPI_LONG_LONG_INT, MPI_SUM,
        engine.config_.borrowed_communicator) == MPI_SUCCESS;
    const bool all_used = reduced && std::all_of(
        global_uses.begin(), global_uses.end(),
        [](const long long uses) { return uses > 0; });
    return numerics.CollectiveContinue(local_valid && all_used);
  }

  static bool VerifyBorrowedInitialTrueTrace(
      cuda::GenericEulerHdgCudaSession& session,
      const CudaBorrowedHostL2Input expected) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyBorrowedInitialTrueTrace(
          *session.borrowed_rank1_, expected);
    if (session.borrowed_rank2_)
      return VerifyBorrowedInitialTrueTrace(
          *session.borrowed_rank2_, expected);
    return false;
  }

  static bool VerifyBoundaryPolicyTable(
      cuda::GenericEulerHdgCudaSession& session,
      const generic::EulerBoundaryPolicy* const expected,
      const std::size_t expected_count, const double expected_gamma) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyBoundaryPolicyTable(
          *session.borrowed_rank1_, expected, expected_count, expected_gamma);
    if (session.borrowed_rank2_)
      return VerifyBoundaryPolicyTable(
          *session.borrowed_rank2_, expected, expected_count, expected_gamma);
    return false;
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::ContractStatus;
using dg::hdg::generic::EulerBoundaryPolicy;
using dg::hdg::generic::EulerBoundaryPolicyKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::BorrowedCudaRank1OneGpuRuntimeSelection;
using dg::hdg::runtime::BorrowedCudaRank2OneGpuHostStagedRuntimeSelection;
using dg::hdg::runtime::cuda_w8a::CudaConfig;
using dg::hdg::runtime::cuda_w8a::CudaBorrowedHostL2Input;
using dg::hdg::runtime::cuda_w8a::CudaMutableHostL2Output;
using dg::hdg::runtime::cuda_w8a::CudaPrecommitSnapshot;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;
using dg::hdg::runtime::cuda_w8a::CudaRunEvidence;
using dg::hdg::runtime::cuda_w8a::CudaStepEvidence;
using dg::hdg::runtime::cuda_w8a::PersistentCudaSmoothEulerEngine;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaPreparedIgrConservationCapture;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;
using dg::hdg::runtime::cuda_w8a::detail::qualification_density_gate;

constexpr int components = dg::hdg::generic::euler_2d_conserved_components;
constexpr std::uint32_t cuda_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cuda) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

bool All(const bool local, const MPI_Comm communicator = MPI_COMM_WORLD)
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
             MPI_SUCCESS &&
      output == 1;
}

mfem::ParMesh MakePeriodicMesh()
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

mfem::ParMesh MakePhysicalMesh()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      3, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  return mfem::ParMesh(MPI_COMM_WORLD, serial);
}

mfem::ParMesh MakeAllSevenPhysicalMesh()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      4, 3, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  for (int boundary = 0; boundary < serial.GetNBE(); ++boundary)
    serial.SetBdrAttribute(boundary, 1 + boundary % 7);
  serial.SetAttributes(false, true);
  return mfem::ParMesh(MPI_COMM_WORLD, serial);
}

struct BoundaryContext
{
  std::array<int, 4> calls{};
  std::array<EulerBoundaryPolicy, 4> policies{};
};

struct AllSevenBoundaryContext
{
  std::array<int, 7> calls{};
  std::array<EulerBoundaryPolicy, 7> policies{};
};

void ConfigureAllSevenPolicies(AllSevenBoundaryContext& context)
{
  const std::array<double, components> inlet{{
      1.0, 0.20, -0.05, 2.52125}};
  const std::array<double, components> subsonic_inlet{{
      0.95, 0.095, 0.019, 2.3812}};
  const std::array<double, components> far_field{{
      1.05, -0.105, 0.0525, 2.5078125}};
  const std::array<double, components> nonreflecting{{
      0.98, 0.049, -0.0294, 2.4522}};
  const std::array<double, components> unused{};
  context.policies = {{
      {EulerBoundaryPolicyKind::supersonic_inlet, inlet, 0.0},
      {EulerBoundaryPolicyKind::subsonic_inlet, subsonic_inlet, 0.0},
      {EulerBoundaryPolicyKind::supersonic_outlet, unused, 0.0},
      {EulerBoundaryPolicyKind::subsonic_outlet_pressure, unused, 0.93},
      {EulerBoundaryPolicyKind::far_field, far_field, 0.0},
      {EulerBoundaryPolicyKind::nonreflecting_outlet, nonreflecting, 0.0},
      {EulerBoundaryPolicyKind::slip_wall, unused, 0.0}}};
}

bool AllSevenPhysicalEulerExterior(
    const void* opaque, const dg::hdg::generic::BoundaryPointView& point,
    const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<AllSevenBoundaryContext*>(
      const_cast<void*>(opaque));
  if (context == nullptr || point.space_dimension != 2 ||
      point.boundary_attribute < 1 || point.boundary_attribute > 7 ||
      interior == nullptr || exterior == nullptr)
    return false;
  ++context->calls[static_cast<std::size_t>(point.boundary_attribute - 1)];
  // Explicit initial_true_trace must bypass compatibility projection and this
  // deliberately failing callback.
  return false;
}

void ConfigurePolicies(BoundaryContext& context, const bool physical)
{
  const std::array<double, components> reference{{1.0, 0.05, -0.02, 2.60}};
  const std::array<double, components> unused{};
  if (!physical)
  {
    for (EulerBoundaryPolicy& policy : context.policies)
    {
      policy = {EulerBoundaryPolicyKind::periodic_topology, unused, 0.0};
    }
    return;
  }
  context.policies = {{
      {EulerBoundaryPolicyKind::slip_wall, unused, 0.0},
      {EulerBoundaryPolicyKind::supersonic_outlet, unused, 0.0},
      {EulerBoundaryPolicyKind::far_field, reference, 0.0},
      {EulerBoundaryPolicyKind::supersonic_inlet, reference, 0.0}}};
}

bool PhysicalEulerExterior(
    const void* opaque, const dg::hdg::generic::BoundaryPointView& point,
    const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<BoundaryContext*>(
      const_cast<void*>(opaque));
  if (context == nullptr || point.position == nullptr ||
      point.outward_normal == nullptr || point.space_dimension != 2 ||
      point.boundary_attribute < 1 || point.boundary_attribute > 4 ||
      interior == nullptr || exterior == nullptr)
  {
    return false;
  }
  ++context->calls[static_cast<std::size_t>(point.boundary_attribute - 1)];
  std::copy_n(interior, components, exterior);
  switch (point.boundary_attribute)
  {
    case 1:
    {
      // Slip-wall exterior state.
      const double normal_momentum =
          interior[1] * point.outward_normal[0] +
          interior[2] * point.outward_normal[1];
      exterior[1] -= 2.0 * normal_momentum * point.outward_normal[0];
      exterior[2] -= 2.0 * normal_momentum * point.outward_normal[1];
      break;
    }
    case 2:
      // Supersonic outlet: the interior state is the exterior state.
      break;
    case 3:
      // Far-field/nonreflecting representative state.
      exterior[0] = 1.0;
      exterior[1] = 0.05;
      exterior[2] = -0.02;
      exterior[3] = 2.60;
      break;
    case 4:
      // Inlet representative state.
      exterior[0] = 1.0;
      exterior[1] = 0.05;
      exterior[2] = -0.02;
      exterior[3] = 2.60;
      break;
  }
  return true;
}

struct BorrowedFields
{
  std::vector<double> initial;
  std::vector<double> initial_true_trace;
  std::vector<double> source;
  std::size_t volume_scalar_count = 0;
};

enum class SourceMode
{
  empty,
  explicit_zero,
  nonzero
};

enum class InitialMode
{
  constant,
  smooth_periodic
};

void EvaluateInitialState(const InitialMode mode, const mfem::Vector& point,
                          mfem::Vector& state)
{
  state.SetSize(components);
  if (mode == InitialMode::constant)
  {
    state[0] = 1.0;
    state[1] = 0.05;
    state[2] = -0.02;
    state[3] = 2.60;
    return;
  }
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double sx = std::sin(2.0 * pi * point[0]);
  const double sy = std::sin(2.0 * pi * point[1]);
  const double cx = std::cos(2.0 * pi * point[0]);
  const double cy = std::cos(2.0 * pi * point[1]);
  const double rho = 1.0 + 0.08 * sx * sy;
  const double u = 0.22 + 0.04 * cx * sy;
  const double v = -0.11 + 0.03 * sx * cy;
  const double pressure = 1.0 + 0.06 * cx * cy;
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = pressure / 0.4 + 0.5 * rho * (u * u + v * v);
}

BorrowedFields MakeBorrowedFields(
    mfem::ParMesh& mesh, const int order, const SourceMode source_mode,
    const InitialMode initial_mode = InitialMode::constant)
{
  BorrowedFields fields;
  mfem::L2_FECollection volume_collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace volume_space(
      &mesh, &volume_collection, components, mfem::Ordering::byNODES);
  fields.volume_scalar_count = static_cast<std::size_t>(
      volume_space.GetVSize() / components);
  mfem::ParGridFunction volume_initial(&volume_space);
  mfem::VectorFunctionCoefficient initial_coefficient(
      components, [initial_mode](const mfem::Vector& point,
                                mfem::Vector& state)
      { EvaluateInitialState(initial_mode, point, state); });
  volume_initial.ProjectCoefficient(initial_coefficient);
  const double* const volume_values = volume_initial.HostRead();
  fields.initial.assign(
      volume_values, volume_values + volume_space.GetVSize());
  if (source_mode != SourceMode::empty)
  {
    fields.source.resize(fields.initial.size(), 0.0);
  }
  for (std::size_t scalar = 0; scalar < fields.volume_scalar_count; ++scalar)
  {
    if (source_mode == SourceMode::nonzero)
    {
      // Keep dt*S far above the nonlinear stopping floor so this checks the
      // source coefficient quantitatively, not merely its sign.
      fields.source[scalar] = 0.25;
      fields.source[fields.volume_scalar_count + scalar] = 0.125;
      fields.source[2 * fields.volume_scalar_count + scalar] = -0.1875;
      fields.source[3 * fields.volume_scalar_count + scalar] = 0.375;
    }
  }

  mfem::DG_Interface_FECollection trace_collection(
      order, 2, mfem::FiniteElement::VALUE,
      mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &trace_collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction local_trace(&trace_space);
  mfem::VectorFunctionCoefficient trace_coefficient(
      components, [initial_mode](const mfem::Vector& point,
                                mfem::Vector& state)
      { EvaluateInitialState(initial_mode, point, state); });
  local_trace.ProjectCoefficient(trace_coefficient);
  mfem::Vector true_trace;
  local_trace.GetTrueDofs(true_trace);
  if (true_trace.Size() > 0)
  {
    const double* const trace_values = true_trace.HostRead();
    fields.initial_true_trace.assign(
        trace_values, trace_values + true_trace.Size());
  }
  return fields;
}

EulerHdgRequest MakeRequest(
    const mfem::ParMesh& mesh, const int order, BorrowedFields& fields,
    BoundaryContext& boundary, const bool physical,
    const RegularizationKind kind)
{
  ConfigurePolicies(boundary, physical);
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {BackendKind::cuda, 0};
  request.capabilities = cuda_capabilities;
  request.initial_state = {
      fields.initial.empty() ? nullptr : fields.initial.data(),
      fields.initial.size(), components};
  request.initial_true_trace = {
      fields.initial_true_trace.empty() ? nullptr
                                        : fields.initial_true_trace.data(),
      fields.initial_true_trace.size(), components};
  request.source = {fields.source.empty() ? nullptr : fields.source.data(),
                    fields.source.size(), components};
  request.boundary.context = &boundary;
  request.boundary.exterior_state = PhysicalEulerExterior;
  request.boundary.policies = physical
      ? dg::hdg::generic::EulerBoundaryPolicyTableView{
            boundary.policies.data(), boundary.policies.size(), 2, 1.4}
      : dg::hdg::generic::EulerBoundaryPolicyTableView{
            nullptr, 0, 2, 1.4};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  request.numerical_policy.stabilization = kind == RegularizationKind::igr
      ? dg::hdg::generic::RegularizationSelection{
            RegularizationKind::igr, 0.2, 0.8}
      : dg::hdg::generic::RegularizationSelection{
            RegularizationKind::none, 0.0, 0.0};
  return request;
}

CudaConfig MakeDirectConfig(const EulerHdgRequest& request,
                            const int ranks, const int order)
{
  CudaConfig config{};
  config.order = order;
  config.mesh_n = 0;
  config.requested_device = 0;
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = const_cast<mfem::ParMesh*>(request.mesh);
  config.borrowed_communicator = request.communicator;
  config.use_borrowed_host_inputs = true;
  config.borrowed_initial = {
      request.initial_state.data,
      request.initial_state.scalar_count / static_cast<std::size_t>(components),
      request.initial_state.components};
  config.borrowed_initial_true_trace = {
      request.initial_true_trace.data,
      request.initial_true_trace.scalar_count /
          static_cast<std::size_t>(components),
      request.initial_true_trace.components};
  config.borrowed_source = {
      request.source.data,
      request.source.scalar_count / static_cast<std::size_t>(components),
      request.source.components};
  config.borrowed_boundary = request.boundary;
  config.use_generic_numerical_policy = true;
  config.numerical_policy = request.numerical_policy;
  return config;
}

struct IndependentIgrConservationResult
{
  std::array<double, 5> production_balance{};
  std::array<double, 5> production_scale{};
  std::array<double, 5> oracle_balance{};
  std::array<double, 5> oracle_scale{};
  // Source components, Euler change, physical Euler flux, sigma/rho,
  // alpha*Q, and the physical auxiliary flux, respectively.
  std::array<double, 9> activity{};
  double maximum_balance_discrepancy =
      std::numeric_limits<double>::infinity();
  double maximum_scale_discrepancy =
      std::numeric_limits<double>::infinity();
  bool passed = false;
};

int DecodeSignedDof(const int encoded, double& sign) noexcept
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
}

bool IndependentRegularizedNormalFlux(
    const std::array<long double, 5>& trace,
    const long double normal_x, const long double normal_y,
    std::array<long double, 4>& flux) noexcept
{
  const long double rho = trace[0];
  const long double momentum_x = trace[1];
  const long double momentum_y = trace[2];
  const long double energy = trace[3];
  const long double sigma_hat = trace[4];
  if (!(rho > 0.0L)) return false;
  const long double normal_momentum =
      momentum_x * normal_x + momentum_y * normal_y;
  const long double pressure = 0.4L *
      (energy - 0.5L *
          (momentum_x * momentum_x + momentum_y * momentum_y) / rho);
  if (!(pressure > 0.0L) || !std::isfinite(pressure)) return false;
  const long double normal_velocity = normal_momentum / rho;
  flux = {{
      normal_momentum,
      momentum_x * normal_velocity + pressure * normal_x +
          sigma_hat * normal_x,
      momentum_y * normal_velocity + pressure * normal_y +
          sigma_hat * normal_y,
      (energy + pressure) * normal_velocity +
          sigma_hat * normal_velocity}};
  return std::all_of(flux.begin(), flux.end(),
                     [](const long double value)
                     { return std::isfinite(value); });
}

bool VerifyIndependentIgrConservation(
    const CudaPreparedIgrConservationCapture& capture,
    const EulerHdgRequest& request,
    IndependentIgrConservationResult& result)
{
  result = {};
  const bool locally_admitted = request.mesh != nullptr &&
      request.source.data != nullptr &&
      request.source.components == components &&
      request.source.scalar_count != 0 &&
      capture.elements >= 0 && capture.scalar_interior_dofs > 0 &&
      capture.scalar_face_dofs > 0 &&
      capture.interior_dofs == capture.scalar_interior_dofs * 7 &&
      capture.trace_dofs == 4 * capture.scalar_face_dofs * 5 &&
      capture.order == request.polynomial_order &&
      capture.assembly_order ==
          request.numerical_policy.igr_assembly_quadrature_order &&
      capture.alpha == request.numerical_policy.stabilization.alpha &&
      capture.beta == request.numerical_policy.stabilization.beta &&
      capture.dt > 0.0 && std::isfinite(capture.dt);
  if (!All(locally_admitted, request.communicator)) return false;

  mfem::ParMesh& mesh = *const_cast<mfem::ParMesh*>(request.mesh);
  bool local_valid = capture.elements == mesh.GetNE() &&
      capture.current_interior.size() ==
          static_cast<std::size_t>(capture.elements) *
              capture.interior_dofs &&
      capture.previous_interior.size() ==
          capture.current_interior.size() &&
      capture.current_local_trace.size() ==
          static_cast<std::size_t>(capture.elements) * capture.trace_dofs;
  std::array<long double, 5> local_balance{};
  std::array<long double, 5> local_scale{};
  std::array<long double, 9> local_activity{};
  std::array<long double, 4> local_change_integral{};
  std::array<long double, 4> local_source_integral{};
  std::array<long double, 4> local_boundary_integral{};
  std::array<long double, 3> local_auxiliary_integral{};

  try
  {
    mfem::L2_FECollection volume_collection(
        capture.order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace source_space(
        &mesh, &volume_collection, components, mfem::Ordering::byNODES);
    mfem::DG_Interface_FECollection trace_collection(
        capture.order, 2, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace trace_space(
        &mesh, &trace_collection, 5, mfem::Ordering::byNODES);
    local_valid = local_valid &&
        request.source.scalar_count ==
            static_cast<std::size_t>(source_space.GetVSize());

    std::vector<unsigned char> shared_faces(
        static_cast<std::size_t>(mesh.GetNumFaces()), 0);
    for (int shared = 0; local_valid && shared < mesh.GetNSharedFaces();
         ++shared)
    {
      const int face = mesh.GetSharedFace(shared);
      local_valid = face >= 0 &&
          face < static_cast<int>(shared_faces.size());
      if (local_valid) shared_faces[static_cast<std::size_t>(face)] = 1;
    }

    for (int element = 0;
         local_valid && element < capture.elements; ++element)
    {
      const mfem::FiniteElement* finite_element =
          source_space.GetFE(element);
      mfem::ElementTransformation* transformation =
          mesh.GetElementTransformation(element);
      mfem::Array<int> source_vdofs;
      source_space.GetElementVDofs(element, source_vdofs);
      local_valid = finite_element != nullptr && transformation != nullptr &&
          finite_element->GetDof() == capture.scalar_interior_dofs &&
          source_vdofs.Size() ==
              capture.scalar_interior_dofs * components;
      if (!local_valid) break;

      const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), capture.assembly_order);
      local_valid = volume_rule.GetNPoints() == capture.volume_points;
      mfem::Vector volume_shape(capture.scalar_interior_dofs);
      mfem::DenseMatrix volume_derivative(
          capture.scalar_interior_dofs, 2);
      const std::size_t interior_offset =
          static_cast<std::size_t>(element) * capture.interior_dofs;
      for (int q = 0; local_valid && q < volume_rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
        transformation->SetIntPoint(&point);
        finite_element->CalcShape(point, volume_shape);
        finite_element->CalcPhysDShape(*transformation, volume_derivative);
        const long double weight = static_cast<long double>(
            point.weight * transformation->Weight());
        std::array<long double, 4> current{};
        std::array<long double, 4> previous{};
        std::array<long double, 4> source{};
        long double sigma = 0.0L;
        long double conservative_gradient[3][2] = {};
        local_valid = weight > 0.0L && std::isfinite(weight);
        for (int scalar = 0;
             local_valid && scalar < capture.scalar_interior_dofs; ++scalar)
        {
          const long double phi = volume_shape[scalar];
          for (int component = 0; component < components; ++component)
          {
            const std::size_t state_index = interior_offset + scalar +
                static_cast<std::size_t>(capture.scalar_interior_dofs) *
                    component;
            current[static_cast<std::size_t>(component)] +=
                phi * capture.current_interior[state_index];
            previous[static_cast<std::size_t>(component)] +=
                phi * capture.previous_interior[state_index];
            double sign = 1.0;
            const int source_dof = DecodeSignedDof(
                source_vdofs[scalar +
                    capture.scalar_interior_dofs * component], sign);
            local_valid = source_dof >= 0 &&
                static_cast<std::size_t>(source_dof) <
                    request.source.scalar_count &&
                std::isfinite(request.source.data[source_dof]);
            if (!local_valid) break;
            source[static_cast<std::size_t>(component)] += phi * sign *
                request.source.data[source_dof];
          }
          sigma += phi * capture.current_interior[
              interior_offset + scalar +
              static_cast<std::size_t>(capture.scalar_interior_dofs) * 4];
          for (int component = 0; component < 3; ++component)
            for (int direction = 0; direction < 2; ++direction)
              conservative_gradient[component][direction] +=
                  static_cast<long double>(
                      volume_derivative(scalar, direction)) *
                  capture.current_interior[
                      interior_offset + scalar +
                      static_cast<std::size_t>(
                          capture.scalar_interior_dofs) * component];
        }
        local_valid = local_valid && current[0] >
            static_cast<long double>(qualification_density_gate) &&
            std::isfinite(sigma);
        for (int component = 0;
             local_valid && component < components; ++component)
        {
          const long double change = weight *
              (current[static_cast<std::size_t>(component)] -
               previous[static_cast<std::size_t>(component)]);
          const long double forcing =
              static_cast<long double>(capture.dt) * weight *
              source[static_cast<std::size_t>(component)];
          local_valid = std::isfinite(change) && std::isfinite(forcing);
          local_balance[static_cast<std::size_t>(component)] +=
              change - forcing;
          local_scale[static_cast<std::size_t>(component)] +=
              std::abs(change) + std::abs(forcing);
          local_change_integral[static_cast<std::size_t>(component)] +=
              change;
          local_source_integral[static_cast<std::size_t>(component)] +=
              forcing;
          local_activity[static_cast<std::size_t>(component)] +=
              std::abs(forcing);
          local_activity[4] += std::abs(change);
        }

        long double velocity_gradient[2][2] = {};
        for (int velocity = 0; velocity < 2; ++velocity)
          for (int direction = 0; direction < 2; ++direction)
            velocity_gradient[velocity][direction] =
                conservative_gradient[1 + velocity][direction] /
                    current[0] -
                current[static_cast<std::size_t>(1 + velocity)] *
                    conservative_gradient[0][direction] /
                    (current[0] * current[0]);
        const long double velocity_trace =
            velocity_gradient[0][0] + velocity_gradient[1][1];
        const long double q_value =
            velocity_gradient[0][0] * velocity_gradient[0][0] +
            2.0L * velocity_gradient[0][1] * velocity_gradient[1][0] +
            velocity_gradient[1][1] * velocity_gradient[1][1] +
            velocity_trace * velocity_trace;
        const long double sigma_term = weight * sigma / current[0];
        const long double q_term = weight *
            static_cast<long double>(capture.alpha) * q_value;
        local_valid = local_valid && std::isfinite(q_value) &&
            std::isfinite(sigma_term) && std::isfinite(q_term);
        local_balance[4] += sigma_term - q_term;
        local_scale[4] += std::abs(sigma_term) + std::abs(q_term);
        local_auxiliary_integral[0] += sigma_term;
        local_auxiliary_integral[1] += q_term;
        local_activity[6] += std::abs(sigma_term);
        local_activity[7] += std::abs(q_term);
      }

      mfem::Array<int> faces;
      mfem::Array<int> orientations;
      mesh.GetElementEdges(element, faces, orientations);
      local_valid = local_valid && faces.Size() == 4;
      for (int local_face = 0;
           local_valid && local_face < faces.Size(); ++local_face)
      {
        const int face = faces[local_face];
        mfem::FaceElementTransformations* face_transformation =
            mesh.GetFaceElementTransformations(face);
        const mfem::FiniteElement* trace_element =
            trace_space.GetFaceElement(face);
        const int boundary_element = face >= 0 &&
                face < static_cast<int>(shared_faces.size())
            ? mesh.GetFaceToBdrElMap()[face] : -1;
        const bool physical = face >= 0 &&
            face < static_cast<int>(shared_faces.size()) &&
            shared_faces[static_cast<std::size_t>(face)] == 0 &&
            face_transformation != nullptr &&
            face_transformation->Elem2No < 0 && boundary_element >= 0;
        if (!physical) continue;
        local_valid = face_transformation->Face != nullptr &&
            trace_element != nullptr &&
            trace_element->GetDof() == capture.scalar_face_dofs &&
            mesh.GetBdrAttribute(boundary_element) >= 1 &&
            mesh.GetBdrAttribute(boundary_element) <= 7;
        if (!local_valid) break;
        const bool element_one =
            face_transformation->Elem1No == element;
        local_valid = element_one ||
            face_transformation->Elem2No == element;
        if (!local_valid) break;

        const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
            trace_element->GetGeomType(), capture.assembly_order);
        local_valid = face_rule.GetNPoints() == capture.face_points;
        long double face_measure = 0.0L;
        for (int q = 0; local_valid && q < face_rule.GetNPoints(); ++q)
        {
          const mfem::IntegrationPoint& point = face_rule.IntPoint(q);
          face_transformation->SetAllIntPoints(&point);
          const long double contribution = point.weight *
              face_transformation->Face->Weight();
          local_valid = contribution > 0.0L &&
              std::isfinite(contribution);
          face_measure += contribution;
        }
        const long double h =
            static_cast<long double>(mesh.GetElementVolume(element)) /
            face_measure;
        local_valid = local_valid && face_measure > 0.0L && h > 0.0L &&
            std::isfinite(h);
        mfem::Vector interior_shape(capture.scalar_interior_dofs);
        mfem::Vector trace_shape(capture.scalar_face_dofs);
        const std::size_t trace_offset =
            static_cast<std::size_t>(element) * capture.trace_dofs +
            static_cast<std::size_t>(local_face) * 5 *
                capture.scalar_face_dofs;
        for (int q = 0;
             local_valid && q < face_rule.GetNPoints(); ++q)
        {
          const mfem::IntegrationPoint& point = face_rule.IntPoint(q);
          face_transformation->SetAllIntPoints(&point);
          const mfem::IntegrationPoint& element_point = element_one
              ? face_transformation->GetElement1IntPoint()
              : face_transformation->GetElement2IntPoint();
          mfem::Vector scaled_normal(2);
          mfem::CalcOrtho(
              face_transformation->Face->Jacobian(), scaled_normal);
          const long double normal_norm = scaled_normal.Norml2();
          const long double measure =
              face_transformation->Face->Weight();
          local_valid = normal_norm > 0.0L && measure > 0.0L &&
              std::isfinite(normal_norm) && std::isfinite(measure);
          if (!local_valid) break;
          const long double side = element_one ? 1.0L : -1.0L;
          const long double normal_x =
              side * scaled_normal[0] / normal_norm;
          const long double normal_y =
              side * scaled_normal[1] / normal_norm;
          finite_element->CalcShape(element_point, interior_shape);
          trace_element->CalcShape(point, trace_shape);
          std::array<long double, 7> interior{};
          std::array<long double, 5> trace{};
          for (int scalar = 0;
               scalar < capture.scalar_interior_dofs; ++scalar)
            for (int component = 0; component < 7; ++component)
              interior[static_cast<std::size_t>(component)] +=
                  static_cast<long double>(interior_shape[scalar]) *
                  capture.current_interior[
                      interior_offset + scalar +
                      static_cast<std::size_t>(
                          capture.scalar_interior_dofs) * component];
          for (int scalar = 0; scalar < capture.scalar_face_dofs; ++scalar)
            for (int component = 0; component < 5; ++component)
              trace[static_cast<std::size_t>(component)] +=
                  static_cast<long double>(trace_shape[scalar]) *
                  capture.current_local_trace[
                      trace_offset + scalar +
                      static_cast<std::size_t>(capture.scalar_face_dofs) *
                          component];
          std::array<long double, 4> flux{};
          local_valid = IndependentRegularizedNormalFlux(
              trace, normal_x, normal_y, flux);
          const long double weight = point.weight * measure;
          for (int component = 0;
               local_valid && component < components; ++component)
          {
            const long double numerical_flux =
                flux[static_cast<std::size_t>(component)] +
                static_cast<long double>(capture.beta) *
                    (interior[static_cast<std::size_t>(component)] -
                     trace[static_cast<std::size_t>(component)]);
            const long double contribution =
                static_cast<long double>(capture.dt) * weight *
                numerical_flux;
            local_valid = std::isfinite(contribution);
            local_balance[static_cast<std::size_t>(component)] +=
                contribution;
            local_scale[static_cast<std::size_t>(component)] +=
                std::abs(contribution);
            local_boundary_integral[static_cast<std::size_t>(component)] +=
                contribution;
            local_activity[5] += std::abs(contribution);
          }
          local_valid = local_valid && trace[0] >
              static_cast<long double>(qualification_density_gate);
          const long double degree_scale =
              static_cast<long double>((capture.order + 1) *
                                       (capture.order + 1));
          const long double tau = degree_scale / (trace[0] * h);
          const long double z_hat_normal =
              interior[5] * normal_x + interior[6] * normal_y -
              tau * (interior[4] - trace[4]);
          const long double auxiliary_contribution =
              static_cast<long double>(capture.alpha) * weight *
              z_hat_normal;
          local_valid = local_valid && tau > 0.0L &&
              std::isfinite(tau) && std::isfinite(auxiliary_contribution);
          local_balance[4] -= auxiliary_contribution;
          local_scale[4] += std::abs(auxiliary_contribution);
          local_auxiliary_integral[2] += auxiliary_contribution;
          local_activity[8] += std::abs(auxiliary_contribution);
        }
      }
    }
  }
  catch (...) { local_valid = false; }

  if (!All(local_valid, request.communicator)) return false;

  std::array<double, 5> production_local_balance{};
  std::array<double, 5> production_local_scale{};
  std::copy_n(capture.production_local.begin(), 5,
              production_local_balance.begin());
  std::copy_n(capture.production_local.begin() + 5, 5,
              production_local_scale.begin());
  std::array<long double, 5> global_balance{};
  std::array<long double, 5> global_scale{};
  std::array<long double, 9> global_activity{};
  std::array<long double, 4> global_change_integral{};
  std::array<long double, 4> global_source_integral{};
  std::array<long double, 4> global_boundary_integral{};
  std::array<long double, 3> global_auxiliary_integral{};
  const int production_balance_status = MPI_Allreduce(
      production_local_balance.data(), result.production_balance.data(), 5,
      MPI_DOUBLE, MPI_SUM, request.communicator);
  const int production_scale_status = MPI_Allreduce(
      production_local_scale.data(), result.production_scale.data(), 5,
      MPI_DOUBLE, MPI_SUM, request.communicator);
  const int oracle_balance_status = MPI_Allreduce(
      local_balance.data(), global_balance.data(), 5, MPI_LONG_DOUBLE,
      MPI_SUM, request.communicator);
  const int oracle_scale_status = MPI_Allreduce(
      local_scale.data(), global_scale.data(), 5, MPI_LONG_DOUBLE,
      MPI_SUM, request.communicator);
  const int activity_status = MPI_Allreduce(
      local_activity.data(), global_activity.data(), 9, MPI_LONG_DOUBLE,
      MPI_SUM, request.communicator);
  const int change_status = MPI_Allreduce(
      local_change_integral.data(), global_change_integral.data(), 4,
      MPI_LONG_DOUBLE, MPI_SUM, request.communicator);
  const int source_status = MPI_Allreduce(
      local_source_integral.data(), global_source_integral.data(), 4,
      MPI_LONG_DOUBLE, MPI_SUM, request.communicator);
  const int boundary_status = MPI_Allreduce(
      local_boundary_integral.data(), global_boundary_integral.data(), 4,
      MPI_LONG_DOUBLE, MPI_SUM, request.communicator);
  const int auxiliary_status = MPI_Allreduce(
      local_auxiliary_integral.data(), global_auxiliary_integral.data(), 3,
      MPI_LONG_DOUBLE, MPI_SUM, request.communicator);
  const bool reduced = production_balance_status == MPI_SUCCESS &&
      production_scale_status == MPI_SUCCESS &&
      oracle_balance_status == MPI_SUCCESS &&
      oracle_scale_status == MPI_SUCCESS &&
      activity_status == MPI_SUCCESS && change_status == MPI_SUCCESS &&
      source_status == MPI_SUCCESS && boundary_status == MPI_SUCCESS &&
      auxiliary_status == MPI_SUCCESS;
  if (!All(reduced, request.communicator)) return false;

  bool parity = true;
  double production_maximum_error = 0.0;
  double production_maximum_relative_error = 0.0;
  result.maximum_balance_discrepancy = 0.0;
  result.maximum_scale_discrepancy = 0.0;
  std::array<double, 5> comparison_tolerance{};
  for (std::size_t component = 0; component < 5; ++component)
  {
    result.oracle_balance[component] =
        static_cast<double>(global_balance[component]);
    result.oracle_scale[component] =
        static_cast<double>(global_scale[component]);
    const double balance_discrepancy = std::abs(
        result.production_balance[component] -
        result.oracle_balance[component]);
    const double scale_discrepancy = std::abs(
        result.production_scale[component] - result.oracle_scale[component]);
    comparison_tolerance[component] =
        2.0e-11 * (1.0 + std::abs(result.oracle_scale[component]));
    result.maximum_balance_discrepancy = std::max(
        result.maximum_balance_discrepancy, balance_discrepancy);
    result.maximum_scale_discrepancy = std::max(
        result.maximum_scale_discrepancy, scale_discrepancy);
    parity = parity && std::isfinite(result.production_balance[component]) &&
        std::isfinite(result.production_scale[component]) &&
        std::isfinite(result.oracle_balance[component]) &&
        std::isfinite(result.oracle_scale[component]) &&
        balance_discrepancy <= comparison_tolerance[component] &&
        scale_discrepancy <= comparison_tolerance[component] &&
        result.production_scale[component] > 0.0;
    const double error = std::abs(result.production_balance[component]);
    production_maximum_error = std::max(
        production_maximum_error, error);
    production_maximum_relative_error = std::max(
        production_maximum_relative_error,
        error / (1.0 + result.production_scale[component]));
  }
  for (std::size_t index = 0; index < result.activity.size(); ++index)
    result.activity[index] = static_cast<double>(global_activity[index]);

  const double lifecycle_error_tolerance = 64.0 *
      std::numeric_limits<double>::epsilon() *
      (1.0 + production_maximum_error);
  const double lifecycle_relative_tolerance = 64.0 *
      std::numeric_limits<double>::epsilon() *
      (1.0 + production_maximum_relative_error);
  const bool lifecycle_parity =
      std::abs(production_maximum_error -
               capture.lifecycle_maximum_error) <=
          lifecycle_error_tolerance &&
      std::abs(production_maximum_relative_error -
               capture.lifecycle_maximum_relative_error) <=
          lifecycle_relative_tolerance;
  const bool active =
      std::all_of(result.activity.begin(), result.activity.begin() + 4,
                  [](const double value)
                  { return std::isfinite(value) && value > 1.0e-12; }) &&
      result.activity[4] > 1.0e-10 &&
      result.activity[5] > 1.0e-10 &&
      result.activity[6] > 1.0e-10 &&
      result.activity[7] > 1.0e-10 &&
      result.activity[8] > 1.0e-10;
  const auto maximum_absolute = [](const auto& values)
  {
    long double maximum = 0.0L;
    for (const long double value : values)
      maximum = std::max(maximum, std::abs(value));
    return maximum;
  };
  const long double euler_omission_tolerance = 100.0L *
      *std::max_element(comparison_tolerance.begin(),
                        comparison_tolerance.begin() + 4);
  const long double auxiliary_omission_tolerance =
      100.0L * comparison_tolerance[4];
  const bool omission_detectable =
      maximum_absolute(global_change_integral) >
          euler_omission_tolerance &&
      maximum_absolute(global_source_integral) >
          euler_omission_tolerance &&
      maximum_absolute(global_boundary_integral) >
          euler_omission_tolerance &&
      std::abs(global_auxiliary_integral[0]) >
          auxiliary_omission_tolerance &&
      std::abs(global_auxiliary_integral[1]) >
          auxiliary_omission_tolerance &&
      std::abs(global_auxiliary_integral[2]) >
          auxiliary_omission_tolerance;
  result.passed = parity && lifecycle_parity && active &&
      omission_detectable;
  return All(result.passed, request.communicator);
}

template <typename Selection>
bool CheckIgrAlgebraForSelection(const EulerHdgRequest& request,
                                 const int ranks,
                                 const int order)
{
  CudaConfig config = MakeDirectConfig(request, ranks, order);

  PersistentCudaSmoothEulerEngine<Selection> engine(config);
  CudaRunEvidence initialization{};
  std::array<double, 6> jacobian_errors{};
  std::array<double, 2> coupling_norms{};
  constexpr double jacobian_tolerance = 5.0e-5;
  const bool initialized = engine.Initialize(initialization);
  const bool initialized_everywhere = All(initialized, request.communicator);
  if (!initialized_everywhere) return false;
  const bool jacobian = CudaQualificationTestAccess::VerifyAssemblerJacobian(
      engine, jacobian_tolerance, jacobian_errors);
  const bool jacobian_everywhere = All(jacobian, request.communicator);
  if (!jacobian_everywhere) return false;
  const bool coupled = CudaQualificationTestAccess::VerifyIgrCouplingBlocks(
      engine, coupling_norms);
  const bool coupled_everywhere = All(coupled, request.communicator);
  if (!coupled_everywhere) return false;

  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence step{};
  constexpr double dt = 1.0e-5;
  const bool prepared = engine.PrepareStep(false, dt, snapshot, step);
  const bool prepared_everywhere = All(prepared, request.communicator);
  if (!prepared_everywhere) return false;

  CudaPreparedOutputMetadataV1 output{};
  const bool staged = engine.StagePreparedOutputMetadata(output, step);
  const bool staged_everywhere = All(staged, request.communicator);
  if (!staged_everywhere) return false;
  const bool acknowledged = engine.MarkOutputStaged(output);
  const bool acknowledged_everywhere = All(
      acknowledged, request.communicator);
  if (!acknowledged_everywhere) return false;
  const bool commit_prepared = engine.PrepareCommit(output, step);
  const bool commit_prepared_everywhere = All(
      commit_prepared, request.communicator);
  if (!commit_prepared_everywhere) return false;
  const bool provisional = engine.CommitPrepared(output, step);
  const bool provisional_everywhere = All(
      provisional, request.communicator);
  if (!provisional_everywhere) return false;
  const bool committed = engine.FinalizeCommit(output, step);
  const bool passed = committed && output.successful_commit;
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA IGR algebra qualification failed order=" << order
                << " initialized=" << initialized
                << " jacobian=" << jacobian
                << " coupling=" << coupled
                << " prepared=" << prepared
                << " jacobian_errors=";
      for (const double error : jacobian_errors) std::cerr << ' ' << error;
      std::cerr << " coupling_norms=" << coupling_norms[0] << ' '
                << coupling_norms[1];
      std::cerr << '\n';
    }
  }
  return All(passed, request.communicator);
}

bool CheckCopiedTraceForGenericSession(
    const EulerHdgRequest& request, BorrowedFields& fields,
    const std::vector<double>& expected_trace, const int order)
{
  GenericEulerHdgCudaSession session;
  const auto initialization = session.Initialize(request);
  const bool initialized =
      initialization.status == HdgEulerSessionStatus::initialized;
  const bool initialized_everywhere = All(
      initialized, request.communicator);
  if (!initialized_everywhere) return false;

  const double poisoned = std::numeric_limits<double>::quiet_NaN();
  std::fill(fields.initial_true_trace.begin(),
            fields.initial_true_trace.end(), poisoned);
  const bool copied_trace =
      CudaQualificationTestAccess::VerifyBorrowedInitialTrueTrace(
          session,
      CudaBorrowedHostL2Input{
          expected_trace.data(), expected_trace.size() /
              static_cast<std::size_t>(components), components});
  const bool passed = initialized && copied_trace;
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA copied-trace qualification failed order="
                << order << " initialized=" << initialized
                << " copied_trace=" << copied_trace << '\n';
    }
  }
  return All(passed, request.communicator);
}

bool CheckCopiedInitialTrueTrace(const int order)
{
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty);
  const std::vector<double> restricted_trace = fields.initial_true_trace;
  const std::size_t trace_scalars =
      fields.initial_true_trace.size() / static_cast<std::size_t>(components);
  for (std::size_t scalar = 0; scalar < trace_scalars; ++scalar)
  {
    fields.initial_true_trace[scalar] +=
        5.0e-4 * static_cast<double>(1 + scalar % 3);
    fields.initial_true_trace[trace_scalars + scalar] -=
        3.0e-4 * static_cast<double>(1 + scalar % 2);
    fields.initial_true_trace[2 * trace_scalars + scalar] +=
        2.0e-4 * static_cast<double>(1 + scalar % 4);
    fields.initial_true_trace[3 * trace_scalars + scalar] += 7.0e-4;
  }
  const std::vector<double> expected_trace = fields.initial_true_trace;
  BoundaryContext boundary{};
  const EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::igr);
  const bool intentionally_distinct = restricted_trace != expected_trace;
  const bool seam = CheckCopiedTraceForGenericSession(
      request, fields, expected_trace, order);
  return All(intentionally_distinct && seam, request.communicator);
}

bool CheckAllSevenLoweredBoundaryPolicies(const int order)
{
  mfem::ParMesh mesh = MakeAllSevenPhysicalMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty);
  BoundaryContext unused_boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, unused_boundary, false,
      RegularizationKind::igr);
  AllSevenBoundaryContext boundary{};
  ConfigureAllSevenPolicies(boundary);
  const std::array<EulerBoundaryPolicy, 7> expected = boundary.policies;
  request.boundary.context = &boundary;
  request.boundary.exterior_state = AllSevenPhysicalEulerExterior;
  request.boundary.policies = {
      boundary.policies.data(), boundary.policies.size(), 2, 1.4};
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;

  GenericEulerHdgCudaSession session;
  const auto initialization = session.Initialize(request);
  const bool initialized = All(
      initialization.status == HdgEulerSessionStatus::initialized,
      request.communicator);
  if (!initialized) return false;
  std::array<int, 7> global_callback_calls{};
  const bool callback_census = MPI_Allreduce(
      boundary.calls.data(), global_callback_calls.data(),
      static_cast<int>(global_callback_calls.size()), MPI_INT, MPI_SUM,
      request.communicator) == MPI_SUCCESS;
  const bool callback_bypassed = callback_census && std::all_of(
      global_callback_calls.begin(), global_callback_calls.end(),
      [](const int calls) { return calls == 0; });

  // The facade's temporary lowered table and the caller's generic table are
  // both gone or poisoned before inspection.  The audit therefore observes
  // the production host-owned per-face map and its uploaded device copy.
  for (EulerBoundaryPolicy& policy : boundary.policies)
  {
    policy.kind = static_cast<EulerBoundaryPolicyKind>(255);
    policy.reference_state.fill(
        std::numeric_limits<double>::quiet_NaN());
    policy.pressure_out = std::numeric_limits<double>::quiet_NaN();
  }
  const bool copied_and_associated = callback_bypassed &&
      CudaQualificationTestAccess::VerifyBoundaryPolicyTable(
          session, expected.data(), expected.size(), 1.4);
  if (!copied_and_associated)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "CUDA all-seven lowered boundary-table audit failed order="
                << order << '\n';
  }
  return All(copied_and_associated, request.communicator);
}

bool CheckIndependentIgrConservation(const int order)
{
  mfem::ParMesh mesh = MakeAllSevenPhysicalMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::smooth_periodic);
  mfem::L2_FECollection source_collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace source_space(
      &mesh, &source_collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction source_field(&source_space);
  mfem::VectorFunctionCoefficient source_coefficient(
      components, [](const mfem::Vector& point, mfem::Vector& source)
      {
        const double modulation = 1.0 + 0.1 * point[0] - 0.05 * point[1];
        source.SetSize(components);
        source[0] = 0.25 * modulation;
        source[1] = 0.125 * modulation;
        source[2] = -0.1875 * modulation;
        source[3] = 0.375 * modulation;
      });
  source_field.ProjectCoefficient(source_coefficient);
  const double* const source_values = source_field.HostRead();
  fields.source.assign(
      source_values, source_values + source_space.GetVSize());

  BoundaryContext unused_boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, unused_boundary, false,
      RegularizationKind::igr);
  AllSevenBoundaryContext boundary{};
  ConfigureAllSevenPolicies(boundary);
  request.boundary.context = &boundary;
  request.boundary.exterior_state = AllSevenPhysicalEulerExterior;
  request.boundary.policies = {
      boundary.policies.data(), boundary.policies.size(), 2, 1.4};
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;

  GenericEulerHdgCudaSession session;
  const auto initialization = session.Initialize(request);
  const bool initialized = All(
      initialization.status == HdgEulerSessionStatus::initialized,
      request.communicator);
  if (!initialized) return false;

  constexpr double dt = 1.0e-5;
  GenericEulerCudaPreparedStep authority;
  GenericEulerHdgCudaSession::prepared_output_type output{};
  const auto prepared_result = session.Prepare(dt, authority, output);
  const bool prepared = All(
      prepared_result.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  if (!prepared) return false;

  CudaPreparedIgrConservationCapture capture;
  const bool captured =
      CudaQualificationTestAccess::CapturePreparedIgrConservation(
          session, capture);
  IndependentIgrConservationResult oracle;
  const bool verified = captured && VerifyIndependentIgrConservation(
      capture, request, oracle);
  const auto aborted = session.Abort(std::move(authority), output);
  const bool aborted_everywhere = All(
      aborted.status == HdgEulerSessionStatus::aborted,
      request.communicator);
  const bool passed = All(
      captured && verified && aborted_everywhere, request.communicator);
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA independent IGR conservation oracle failed order="
                << order << " captured=" << captured
                << " verified=" << verified
                << " aborted=" << aborted_everywhere
                << " balance_discrepancy="
                << oracle.maximum_balance_discrepancy
                << " scale_discrepancy="
                << oracle.maximum_scale_discrepancy << " activity=";
      for (const double value : oracle.activity) std::cerr << ' ' << value;
      std::cerr << '\n';
    }
  }
  return passed;
}

bool CheckIgrAlgebra(const int order, const int ranks)
{
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::smooth_periodic);
  BoundaryContext boundary{};
  const EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::igr);
  return ranks == 1
      ? CheckIgrAlgebraForSelection<
            BorrowedCudaRank1OneGpuRuntimeSelection>(request, ranks, order)
      : CheckIgrAlgebraForSelection<
            BorrowedCudaRank2OneGpuHostStagedRuntimeSelection>(
                request, ranks, order);
}

bool SameValues(const std::vector<double>& left,
                const std::vector<double>& right)
{
  return left == right;
}

bool NearValues(const std::vector<double>& left,
                const std::vector<double>& right)
{
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i)
  {
    const double scale = std::max({1.0, std::abs(left[i]),
                                   std::abs(right[i])});
    if (!std::isfinite(left[i]) || !std::isfinite(right[i]) ||
        std::abs(left[i] - right[i]) > 1.0e-12 * scale)
    {
      return false;
    }
  }
  return true;
}

bool DifferentValues(const std::vector<double>& left,
                     const std::vector<double>& right)
{
  return left.size() == right.size() &&
      !std::equal(left.begin(), left.end(), right.begin());
}

bool ExpectedPeriodicSourceResponse(
    const std::vector<double>& candidate,
    const std::vector<double>& initial,
    const std::vector<double>& source,
    const std::size_t scalar_count,
    const SourceMode source_mode,
    const InitialMode initial_mode,
    const double dt)
{
  if (initial_mode != InitialMode::constant) return true;
  if (source_mode == SourceMode::empty ||
      source_mode == SourceMode::explicit_zero)
    return NearValues(candidate, initial);
  if (candidate.size() != initial.size() ||
      source.size() != initial.size() ||
      initial.size() != scalar_count * static_cast<std::size_t>(components))
    return false;
  for (int component = 0; component < components; ++component)
    for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
    {
      const std::size_t index = scalar_count *
          static_cast<std::size_t>(component) + scalar;
      const double expected_delta = dt * source[index];
      const double actual_delta = candidate[index] - initial[index];
      const double state_scale = std::max(
          {1.0, std::abs(initial[index]), std::abs(candidate[index])});
      const double roundoff_floor = 4096.0 *
          std::numeric_limits<double>::epsilon() * state_scale;
      const double tolerance = roundoff_floor +
          5.0e-5 * std::abs(expected_delta);
      if (!std::isfinite(actual_delta) || expected_delta == 0.0 ||
          std::signbit(actual_delta) != std::signbit(expected_delta) ||
          std::abs(actual_delta - expected_delta) > tolerance)
        return false;
    }
  return true;
}

bool BoundaryCoverage(const BoundaryContext& context,
                      const bool physical,
                      const MPI_Comm communicator)
{
  // CUDA consumes the copied POD table directly; it is not required to call
  // the host callback for physical faces.  Periodic topology, however, must
  // never be reclassified as a physical callback closure.
  if (physical) return true;
  std::array<int, 4> global{};
  if (MPI_Allreduce(context.calls.data(), global.data(),
                    static_cast<int>(global.size()), MPI_INT, MPI_SUM,
                    communicator) != MPI_SUCCESS)
  {
    return false;
  }
  return std::all_of(global.begin(), global.end(),
                     [](const int calls) { return calls == 0; });
}

bool RejectedBeforeMfemDevice(const EulerHdgRequest& request)
{
  const bool unconfigured_before = !mfem::Device::IsConfigured();
  GenericEulerHdgCudaSession session;
  const auto result = session.Initialize(request);
  const auto coordinates = session.coordinates();
  return unconfigured_before &&
      result.status == HdgEulerSessionStatus::rejected &&
      coordinates.step == 0 && coordinates.generation == 0 &&
      coordinates.time == 0.0 && !mfem::Device::IsConfigured();
}

bool CheckContractAdmission(const int order)
{
  mfem::ParMesh mesh = MakePhysicalMesh();
  // Keep this pre-device contract fixture host-only.  ProjectCoefficient on a
  // CUDA-enabled MFEM build requires an admitted Device, while the malformed
  // session cases below deliberately prove rejection before Device setup.
  BorrowedFields fields{};
  fields.volume_scalar_count = 1;
  fields.initial = {1.0, 0.05, -0.02, 2.60};
  fields.initial_true_trace = fields.initial;
  fields.source = {0.25, 0.125, -0.1875, 0.375};
  BoundaryContext boundary{};
  EulerHdgRequest conventional = MakeRequest(
      mesh, order, fields, boundary, true, RegularizationKind::none);
  EulerHdgRequest igr = MakeRequest(
      mesh, order, fields, boundary, true, RegularizationKind::igr);
  EulerHdgRequest invalid_igr = igr;
  invalid_igr.numerical_policy.stabilization.alpha = 0.0;
  EulerHdgRequest null_table = conventional;
  null_table.boundary.policies.policies = nullptr;
  EulerHdgRequest short_table = conventional;
  short_table.boundary.policies.count = boundary.policies.size() - 1;
  EulerHdgRequest wrong_dimension = conventional;
  wrong_dimension.boundary.policies.space_dimension = 3;
  EulerHdgRequest nonfinite_gamma = conventional;
  nonfinite_gamma.boundary.policies.gamma =
      std::numeric_limits<double>::quiet_NaN();

  std::array<EulerBoundaryPolicy, 4> invalid_kind_policies = boundary.policies;
  invalid_kind_policies[0].kind = static_cast<EulerBoundaryPolicyKind>(255);
  EulerHdgRequest invalid_kind = conventional;
  invalid_kind.boundary.policies.policies = invalid_kind_policies.data();

  int ranks = 1;
  int rank = 0;
  MPI_Comm_size(mesh.GetComm(), &ranks);
  MPI_Comm_rank(mesh.GetComm(), &rank);
  std::array<EulerBoundaryPolicy, 4> mismatched_policies = boundary.policies;
  if (ranks == 2 && rank == 0)
  {
    mismatched_policies[0].kind =
        EulerBoundaryPolicyKind::supersonic_outlet;
  }
  EulerHdgRequest mismatched = conventional;
  mismatched.boundary.policies.policies = mismatched_policies.data();

  const bool valid_contracts =
      dg::hdg::generic::ValidateEulerHdgRequest(conventional) ==
          ContractStatus::accepted &&
      dg::hdg::generic::ValidateEulerHdgRequest(igr) ==
          ContractStatus::accepted;
  const bool invalid_igr_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(invalid_igr) !=
          ContractStatus::accepted;
  const bool null_table_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(null_table) !=
          ContractStatus::accepted;
  const bool short_table_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(short_table) !=
          ContractStatus::accepted;
  const bool wrong_dimension_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(wrong_dimension) !=
          ContractStatus::accepted;
  const bool nonfinite_gamma_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(nonfinite_gamma) !=
          ContractStatus::accepted;
  const bool invalid_kind_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(invalid_kind) !=
          ContractStatus::accepted;
  const bool mismatched_contract = ranks != 2 ||
      dg::hdg::generic::ValidateEulerHdgRequest(mismatched) !=
          ContractStatus::accepted;

  // The public session repeats the generic contract before any device or
  // engine allocation.  Each malformed request owns a fresh one-shot facade.
  const bool invalid_igr_session = RejectedBeforeMfemDevice(invalid_igr);
  const bool null_table_session = RejectedBeforeMfemDevice(null_table);
  const bool short_table_session = RejectedBeforeMfemDevice(short_table);
  const bool wrong_dimension_session =
      RejectedBeforeMfemDevice(wrong_dimension);
  const bool nonfinite_gamma_session =
      RejectedBeforeMfemDevice(nonfinite_gamma);
  const bool invalid_kind_session = RejectedBeforeMfemDevice(invalid_kind);
  const bool mismatched_session = ranks != 2 ||
      RejectedBeforeMfemDevice(mismatched);
  return All(valid_contracts && invalid_igr_contract &&
                 null_table_contract && short_table_contract &&
                 wrong_dimension_contract && nonfinite_gamma_contract &&
                 invalid_kind_contract && mismatched_contract &&
                 invalid_igr_session && null_table_session &&
                 short_table_session && wrong_dimension_session &&
                 nonfinite_gamma_session && invalid_kind_session &&
                 mismatched_session,
             mesh.GetComm());
}

bool CheckLifecycle(const int order, const bool physical,
                    const RegularizationKind kind,
                    const SourceMode source_mode,
                    const InitialMode initial_mode,
                    std::vector<double>& committed_result)
{
  mfem::ParMesh mesh = physical ? MakePhysicalMesh() : MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, source_mode, initial_mode);
  const std::vector<double> expected_initial = fields.initial;
  const std::vector<double> expected_source = fields.source;
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, physical, kind);
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
  {
    return false;
  }

  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(request);
  std::vector<double> committed_initial(expected_initial.size(), -1.0);
  const bool initialized_everywhere = All(
      initialized.status == HdgEulerSessionStatus::initialized,
      request.communicator);
  if (!initialized_everywhere) return false;
  const bool initial_export = session.ExportCommittedInterior(
      CudaMutableHostL2Output{
          committed_initial.data(), fields.volume_scalar_count, components});
  if (!All(initial_export && SameValues(committed_initial, expected_initial),
           request.communicator))
  {
    return false;
  }

  // Every borrowed numerical view must have been copied by Initialize.
  const double poisoned = std::numeric_limits<double>::quiet_NaN();
  std::fill(fields.initial.begin(), fields.initial.end(), poisoned);
  std::fill(fields.initial_true_trace.begin(),
            fields.initial_true_trace.end(), poisoned);
  std::fill(fields.source.begin(), fields.source.end(), poisoned);

  constexpr double dt = 1.0e-5;
  GenericEulerCudaPreparedStep first_authority;
  GenericEulerHdgCudaSession::prepared_output_type first_output{};
  const auto before = session.coordinates();
  const auto first_prepared = session.Prepare(
      dt, first_authority, first_output);
  std::vector<double> first_candidate(expected_initial.size(), -2.0);
  const bool first_prepared_everywhere = All(
      first_prepared.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  if (!first_prepared_everywhere) return false;
  const bool first_export = session.ExportPreparedInterior(
      first_authority, first_output,
      CudaMutableHostL2Output{first_candidate.data(),
                             fields.volume_scalar_count, components});
  const bool expected_update = physical
      ? (source_mode != SourceMode::nonzero ||
         DifferentValues(first_candidate, committed_initial))
      : ExpectedPeriodicSourceResponse(
            first_candidate, committed_initial, expected_source,
            fields.volume_scalar_count, source_mode, initial_mode, dt);
  if (!All(first_export && expected_update,
           request.communicator))
  {
    return false;
  }
  const auto aborted = session.Abort(
      std::move(first_authority), first_output);
  std::vector<double> after_abort(expected_initial.size(), -3.0);
  const bool aborted_everywhere = All(
      aborted.status == HdgEulerSessionStatus::aborted,
      request.communicator);
  if (!aborted_everywhere) return false;
  const bool abort_export = session.ExportCommittedInterior(
      CudaMutableHostL2Output{
          after_abort.data(), fields.volume_scalar_count, components});
  if (!All(abort_export && SameValues(after_abort, committed_initial) &&
               session.coordinates().step == before.step &&
               session.coordinates().generation == before.generation &&
               session.coordinates().time == before.time,
           request.communicator))
  {
    return false;
  }

  GenericEulerCudaPreparedStep retry_authority;
  GenericEulerHdgCudaSession::prepared_output_type retry_output{};
  const auto retry_prepared = session.Prepare(
      dt, retry_authority, retry_output);
  std::vector<double> retry_candidate(expected_initial.size(), -4.0);
  const bool retry_prepared_everywhere = All(
      retry_prepared.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  if (!retry_prepared_everywhere) return false;
  const bool retry_export = session.ExportPreparedInterior(
      retry_authority, retry_output,
      CudaMutableHostL2Output{retry_candidate.data(),
                             fields.volume_scalar_count, components});
  if (!All(retry_export && NearValues(first_candidate, retry_candidate),
           request.communicator))
  {
    return false;
  }
  const bool acknowledged = session.AcknowledgePreparedOutput(
      retry_authority, retry_output);
  if (!All(acknowledged, request.communicator)) return false;
  const auto committed = session.Commit(
      std::move(retry_authority), retry_output);
  std::vector<double> committed_final(expected_initial.size(), -5.0);
  const bool committed_everywhere = All(
      committed.status == HdgEulerSessionStatus::committed,
      request.communicator);
  if (!committed_everywhere) return false;
  const bool final_export = session.ExportCommittedInterior(
      CudaMutableHostL2Output{
          committed_final.data(), fields.volume_scalar_count, components});
  const auto after = session.coordinates();
  const bool boundary_coverage = BoundaryCoverage(
      boundary, physical, request.communicator);
  const bool passed = All(
      final_export && SameValues(committed_final, retry_candidate) &&
          after.step == before.step + 1 &&
          after.generation == before.generation + 1 &&
          after.time == before.time + dt &&
          retry_output.successful_commit && boundary_coverage,
      request.communicator);
  if (passed) committed_result = std::move(committed_final);
  return passed;
}

int ParseOrder(const int argc, char* argv[])
{
  if (argc != 3 || std::string(argv[1]) != "--order") return -1;
  try
  {
    const int order = std::stoi(argv[2]);
    return order >= dg::hdg::generic::minimum_polynomial_order &&
            order <= dg::hdg::generic::maximum_polynomial_order
        ? order
        : -1;
  }
  catch (...) { return -1; }
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int order = ParseOrder(argc, argv);
  const bool admission = order > 0 && CheckContractAdmission(order);
  mfem::Device device("cuda");
  (void)device;
  std::vector<double> conventional_periodic_state;
  std::vector<double> conventional_physical_state;
  std::vector<double> igr_periodic_state;
  std::vector<double> igr_physical_state;
  std::vector<double> conventional_empty_source_state;
  std::vector<double> conventional_zero_source_state;
  std::vector<double> igr_empty_source_state;
  std::vector<double> igr_zero_source_state;
  const bool copied_initial_trace = order > 0 &&
      CheckCopiedInitialTrueTrace(order);
  const bool all_seven_boundary_policies = order > 0 &&
      CheckAllSevenLoweredBoundaryPolicies(order);
  const bool independent_igr_conservation = order > 0 &&
      CheckIndependentIgrConservation(order);
  const bool igr_algebra = order > 0 && CheckIgrAlgebra(order, ranks);
  const bool conventional_periodic = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::nonzero,
      InitialMode::constant,
      conventional_periodic_state);
  const bool conventional_physical = order > 0 && CheckLifecycle(
      order, true, RegularizationKind::none, SourceMode::nonzero,
      InitialMode::constant,
      conventional_physical_state);
  const bool igr_periodic = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::nonzero,
      InitialMode::constant,
      igr_periodic_state);
  const bool igr_physical = order > 0 && CheckLifecycle(
      order, true, RegularizationKind::igr, SourceMode::nonzero,
      InitialMode::constant,
      igr_physical_state);
  const bool conventional_empty_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::empty,
      InitialMode::constant,
      conventional_empty_source_state);
  const bool conventional_zero_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::explicit_zero,
      InitialMode::constant,
      conventional_zero_source_state);
  const bool igr_empty_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::empty,
      InitialMode::constant,
      igr_empty_source_state);
  const bool igr_zero_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::explicit_zero,
      InitialMode::constant,
      igr_zero_source_state);
  const bool source_equivalence = conventional_empty_source &&
      conventional_zero_source && igr_empty_source && igr_zero_source &&
      NearValues(conventional_empty_source_state,
                 conventional_zero_source_state) &&
      NearValues(igr_empty_source_state, igr_zero_source_state);
  const bool boundary_effect = conventional_periodic && conventional_physical &&
      igr_periodic && igr_physical &&
      DifferentValues(conventional_periodic_state,
                      conventional_physical_state) &&
      DifferentValues(igr_periodic_state, igr_physical_state);
  const bool passed = (ranks == 1 || ranks == 2) && All(
      admission && copied_initial_trace && all_seven_boundary_policies &&
      independent_igr_conservation && igr_algebra &&
      conventional_periodic && conventional_physical &&
      igr_periodic && igr_physical && boundary_effect && source_equivalence);
  if (!passed)
  {
    std::cerr << "generic CUDA Euler-extension qualification failed rank="
              << rank << " ranks=" << ranks << " order=" << order
              << " admission=" << admission
              << " copied_initial_trace=" << copied_initial_trace
              << " all_seven_boundary_policies="
              << all_seven_boundary_policies
              << " independent_igr_conservation="
              << independent_igr_conservation
              << " igr_algebra=" << igr_algebra
              << " conventional_periodic=" << conventional_periodic
              << " conventional_physical=" << conventional_physical
              << " igr_periodic=" << igr_periodic
              << " igr_physical=" << igr_physical
              << " boundary_effect=" << boundary_effect
              << " source_equivalence=" << source_equivalence << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
