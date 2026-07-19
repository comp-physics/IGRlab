#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "physics/regularization/adapters/frozen_scalar_igr_broken_dg_cpu.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace regularization = dg_igr::regularization;
namespace algebra = dg_igr::frozen_scalar_igr_algebra;

namespace
{

using Model = regularization::FrozenScalarIGRBrokenDGModelTag;

static_assert(regularization::FrozenScalarIGRBrokenDGCPUCombination::value,
              "the exact adapter combination must be supported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::DGDeviceBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyNeutralTag>::value,
              "MFEMDevice must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::DGSEMCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyNeutralTag>::value,
              "DGSEM must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::HDGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyNeutralTag>::value,
              "HDG must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::PointwiseEvaluationTag,
                regularization::DGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyNeutralTag>::value,
              "pointwise evaluation must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::LocalEliminationTag,
                regularization::DGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyNeutralTag>::value,
              "local elimination must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::TraceL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::DGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyNeutralTag>::value,
              "TraceL2 must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::HybridizedTraceTag,
                regularization::GlobalLinearSolveTag,
                regularization::DGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyNeutralTag>::value,
              "hybridized trace must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::DGCPUBackendTag,
                regularization::NoneGradientTreatmentTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyNeutralTag>::value,
              "H1/None gradient treatment must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::DGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::LaggedStepTag,
                regularization::EntropyNeutralTag>::value,
              "lagged time treatment must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::DGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FullyCoupledTag,
                regularization::EntropyNeutralTag>::value,
              "fully coupled time treatment must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::DGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyConservativeTag>::value,
              "entropy-conservative treatment must remain unsupported");
static_assert(!regularization::StaticCombinationSupport<
                Model,
                regularization::ElementL2Tag,
                regularization::GlobalLinearSolveTag,
                regularization::DGCPUBackendTag,
                regularization::ConservativeStateBrokenDGTag,
                regularization::FrozenRKStageTag,
                regularization::EntropyStableTag>::value,
              "entropy-stable treatment must remain unsupported");
static_assert(!regularization::detail::HasWallHook<Model>::value,
              "wall hook must be absent");
static_assert(!regularization::detail::HasDiagnosticsHook<Model>::value,
              "diagnostics hook must be absent");
static_assert(!std::is_polymorphic<
                regularization::FrozenScalarIGRBrokenDGCPUAdapter>::value,
              "adapter must not use runtime polymorphism");

class TestContext
{
public:
  void Check(bool condition, const std::string& message)
  {
    if (!condition)
    {
      ++failures_;
      std::cerr << "[FAIL] " << message << '\n';
    }
  }

  void Code(regularization::ValidationCode actual,
            regularization::ValidationCode expected,
            const std::string& message)
  {
    Check(actual == expected,
          message + " actual_code=" + std::to_string(static_cast<int>(actual)) +
            " expected_code=" + std::to_string(static_cast<int>(expected)));
  }

  void Near(mfem::real_t actual,
            mfem::real_t expected,
            mfem::real_t tolerance,
            const std::string& message)
  {
    const mfem::real_t error = std::abs(actual - expected);
    if (!(error <= tolerance))
    {
      ++failures_;
      std::cerr << "[FAIL] " << message << " actual=" << actual
                << " expected=" << expected << " error=" << error
                << " tolerance=" << tolerance << '\n';
    }
  }

  int Failures() const { return failures_; }

private:
  int failures_ = 0;
};

struct ParityMetrics
{
  mfem::real_t coefficient = 0.0;
  mfem::real_t source = 0.0;
  mfem::real_t tensor = 0.0;
  mfem::real_t normal = 0.0;
  mfem::real_t characteristic = 0.0;
  mfem::real_t two_state = 0.0;
};

regularization::FrozenScalarIGRParameters ValidParameters()
{
  return {static_cast<mfem::real_t>(0.37),
          static_cast<mfem::real_t>(1.0e-7),
          static_cast<mfem::real_t>(2.0e-8),
          true};
}

regularization::FrozenScalarIGRAdapterSelection EnabledSelection()
{
  regularization::FrozenScalarIGRAdapterSelection selection;
  selection.model =
    regularization::FrozenScalarIGRAdapterModelKind::FrozenScalarIGR;
  return selection;
}

regularization::RegularizationValidationContext OperationContextFor(
  const regularization::FrozenScalarIGREvaluatedPointStamp& stamp,
  regularization::RegularizationOperationKind operation)
{
  regularization::RegularizationValidationContext context{
    false, stamp.rk_stage, stamp.state_generation, {}};
  context.expected_producer_generations.state =
    stamp.producer_generations.state;
  if (operation ==
      regularization::RegularizationOperationKind::AuxiliarySource)
  {
    context.expected_producer_generations.conservative_state_gradient =
      stamp.producer_generations.conservative_state_gradient;
  }
  if (operation ==
        regularization::RegularizationOperationKind::VolumeCorrection ||
      operation ==
        regularization::RegularizationOperationKind::NormalCorrection ||
      operation ==
        regularization::RegularizationOperationKind::CharacteristicBounds)
  {
    context.expected_producer_generations.auxiliary =
      stamp.producer_generations.auxiliary;
  }
  if (operation ==
        regularization::RegularizationOperationKind::NormalCorrection ||
      operation ==
        regularization::RegularizationOperationKind::CharacteristicBounds)
  {
    context.expected_producer_generations.normal =
      stamp.producer_generations.normal;
  }
  if (operation ==
      regularization::RegularizationOperationKind::CharacteristicBounds)
  {
    context.expected_producer_generations.base_characteristic_speed =
      stamp.producer_generations.base_characteristic_speed;
  }
  return context;
}

void FillState(mfem::Vector& state, int point_count, int dimension)
{
  const int components = dimension + 2;
  mfem::real_t* values = mfem::Write(state.GetMemory(), state.Size(), false);
  for (int point = 0; point < point_count; ++point)
  {
    const int offset = components * point;
    values[offset] =
      static_cast<mfem::real_t>(0.85 + 0.17 * point + 0.03 * dimension);
    for (int direction = 0; direction < dimension; ++direction)
    {
      values[offset + 1 + direction] = static_cast<mfem::real_t>(
        (direction % 2 == 0 ? 0.31 : -0.23) * (point + 1) * (direction + 1));
    }
    values[offset + components - 1] =
      static_cast<mfem::real_t>(2.4 + 0.21 * point + 0.11 * dimension);
  }
}

void FillGradient(mfem::Vector& gradient, int point_count, int dimension)
{
  const int components = dimension + 2;
  mfem::real_t* values =
    mfem::Write(gradient.GetMemory(), gradient.Size(), false);
  for (int point = 0; point < point_count; ++point)
  {
    const int point_offset = point * components * dimension;
    for (int direction = 0; direction < dimension; ++direction)
    {
      for (int component = 0; component < components; ++component)
      {
        const mfem::real_t sign = ((point + direction + component) % 2 == 0)
                                    ? static_cast<mfem::real_t>(1.0)
                                    : static_cast<mfem::real_t>(-1.0);
        values[point_offset + component + components * direction] =
          sign * static_cast<mfem::real_t>(0.013 * (point + 1) *
                                           (direction + 2) * (component + 1));
      }
    }
  }
}

void FillSigma(mfem::Vector& sigma, int point_count)
{
  mfem::real_t* values = mfem::Write(sigma.GetMemory(), sigma.Size(), false);
  for (int point = 0; point < point_count; ++point)
  {
    values[point] =
      static_cast<mfem::real_t>((point % 2 == 0 ? 0.19 : -0.07) * (point + 1));
  }
}

void FillBaseCharacteristicSpeed(mfem::Vector& speed,
                                 int point_count,
                                 int dimension)
{
  mfem::real_t* values = mfem::Write(speed.GetMemory(), speed.Size(), false);
  for (int point = 0; point < point_count; ++point)
  {
    values[point] =
      static_cast<mfem::real_t>(0.75 + 0.61 * point + 0.09 * dimension);
  }
}

void FillNormal(mfem::Vector& normal, int point_count, int dimension)
{
  mfem::real_t* values = mfem::Write(normal.GetMemory(), normal.Size(), false);
  for (int point = 0; point < point_count; ++point)
  {
    mfem::real_t scale = 0.0;
    for (int direction = 0; direction < dimension; ++direction)
    {
      const mfem::real_t value = static_cast<mfem::real_t>(
        (direction % 2 == 0 ? 0.4 : -0.3) * (point + direction + 1));
      values[point * dimension + direction] = value;
      scale += value * value;
    }
    scale = std::sqrt(scale);
    for (int direction = 0; direction < dimension; ++direction)
    {
      values[point * dimension + direction] /= scale;
    }
  }
}

void FillSentinel(mfem::Vector& vector, mfem::real_t base)
{
  mfem::real_t* values = mfem::Write(vector.GetMemory(), vector.Size(), false);
  for (int index = 0; index < vector.Size(); ++index)
  {
    values[index] = base + static_cast<mfem::real_t>(0.007 * index);
  }
}

std::vector<mfem::real_t> Snapshot(const mfem::Vector& vector)
{
  const mfem::real_t* values = vector.Read(false);
  return std::vector<mfem::real_t>(values, values + vector.Size());
}

void CheckSnapshot(TestContext& test,
                   const mfem::Vector& vector,
                   const std::vector<mfem::real_t>& expected,
                   const std::string& message)
{
  test.Check(vector.Size() == static_cast<int>(expected.size()),
             message + " size unchanged");
  const mfem::real_t* values = vector.Read(false);
  for (int index = 0; index < vector.Size(); ++index)
  {
    test.Near(values[index],
              expected[static_cast<std::size_t>(index)],
              0.0,
              message + " entry " + std::to_string(index));
  }
}

regularization::RegularizationOperationOutputs
OutputFor(regularization::RegularizationOperationKind operation,
          mfem::Vector& storage,
          const regularization::EvaluatedPointBatch& batch)
{
  int components = 0;
  switch (operation)
  {
  case regularization::RegularizationOperationKind::AuxiliaryCoefficient:
  case regularization::RegularizationOperationKind::AuxiliarySource:
    components = 1;
    break;
  case regularization::RegularizationOperationKind::VolumeCorrection:
    components = batch.dimension * (batch.dimension + 2);
    break;
  case regularization::RegularizationOperationKind::NormalCorrection:
  case regularization::RegularizationOperationKind::WallLaw:
    components = batch.dimension + 2;
    break;
  case regularization::RegularizationOperationKind::CharacteristicBounds:
    components = 2;
    break;
  case regularization::RegularizationOperationKind::Diagnostics:
    components = 3;
    break;
  }

  const regularization::EvaluatedPointOutputBinding binding =
    regularization::BindEvaluatedPointOutput(storage,
                                             batch.point_count,
                                             components,
                                             batch.dimension,
                                             batch.site,
                                             batch.point_set_id);
  regularization::RegularizationOperationOutputs outputs;
  switch (operation)
  {
  case regularization::RegularizationOperationKind::AuxiliaryCoefficient:
    outputs.auxiliary_coefficient = binding;
    break;
  case regularization::RegularizationOperationKind::AuxiliarySource:
    outputs.auxiliary_source = binding;
    break;
  case regularization::RegularizationOperationKind::VolumeCorrection:
    outputs.volume_correction = binding;
    break;
  case regularization::RegularizationOperationKind::NormalCorrection:
    outputs.normal_correction = binding;
    break;
  case regularization::RegularizationOperationKind::WallLaw:
    outputs.wall_flux = binding;
    break;
  case regularization::RegularizationOperationKind::CharacteristicBounds:
    outputs.characteristic_bounds = binding;
    break;
  case regularization::RegularizationOperationKind::Diagnostics:
    outputs.diagnostics = binding;
    break;
  }
  return outputs;
}

regularization::FrozenScalarIGREvaluatedPointStamp
Stamp(int point_count,
      int dimension,
      regularization::EvaluationSiteKind site,
      std::uint64_t point_set_id,
      std::uint64_t generation_base)
{
  return {point_count,
          dimension,
          site,
          point_set_id,
          0,
          generation_base + 1u,
          {generation_base + 11u,
           generation_base + 23u,
           generation_base + 37u,
           generation_base + 41u,
           generation_base + 53u,
           generation_base + 67u}};
}

mfem::real_t
UpdateMaximum(mfem::real_t maximum, mfem::real_t actual, mfem::real_t expected)
{
  return std::max(maximum, std::abs(actual - expected));
}

void TestPhysicalParityForDimension(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter,
  int dimension,
  ParityMetrics& metrics)
{
  const int equations = dimension + 2;
  const regularization::FrozenScalarIGRParameters parameters =
    adapter.Parameters();

  const int coefficient_points = dimension + 2;
  mfem::Vector coefficient_state(coefficient_points * equations);
  FillState(coefficient_state, coefficient_points, dimension);
  const auto coefficient_stamp =
    Stamp(coefficient_points,
          dimension,
          regularization::EvaluationSiteKind::Volume,
          1000u + dimension,
          100u + dimension);
  const regularization::EvaluatedPointBatch coefficient_batch =
    regularization::MakeFrozenScalarIGRCoefficientBatch(coefficient_state,
                                                        coefficient_stamp);
  mfem::Vector coefficient_output(coefficient_points);
  FillSentinel(coefficient_output, -8.0);
  const regularization::RegularizationLaunchResult coefficient_result =
    adapter.Launch(
      regularization::RegularizationOperationKind::AuxiliaryCoefficient,
      OperationContextFor(
        coefficient_stamp,
        regularization::RegularizationOperationKind::AuxiliaryCoefficient),
      coefficient_batch,
      OutputFor(
        regularization::RegularizationOperationKind::AuxiliaryCoefficient,
        coefficient_output,
        coefficient_batch));
  test.Check(coefficient_result.launched && coefficient_result.validation.Ok(),
             "coefficient launches in dimension " + std::to_string(dimension));
  test.Check(
    coefficient_batch.auxiliary.Storage() == nullptr &&
      OperationContextFor(
        coefficient_stamp,
        regularization::RegularizationOperationKind::AuxiliaryCoefficient)
          .expected_producer_generations.auxiliary == 0u,
    "coefficient launches with absent sigma and zero auxiliary authority");
  const mfem::real_t* coefficient_state_data = coefficient_state.Read(false);
  const mfem::real_t* coefficient_data = coefficient_output.Read(false);
  for (int point = 0; point < coefficient_points; ++point)
  {
    const mfem::real_t expected =
      algebra::AlphaOverDensity(parameters.alpha,
                                coefficient_state_data[point * equations],
                                parameters.solver_density_floor);
    metrics.coefficient =
      UpdateMaximum(metrics.coefficient, coefficient_data[point], expected);
  }

  const int source_points = dimension + 3;
  mfem::Vector source_state(source_points * equations);
  mfem::Vector source_gradient(source_points * equations * dimension);
  FillState(source_state, source_points, dimension);
  FillGradient(source_gradient, source_points, dimension);
  const auto source_stamp = Stamp(source_points,
                                  dimension,
                                  regularization::EvaluationSiteKind::Volume,
                                  2000u + dimension,
                                  200u + dimension);
  const regularization::EvaluatedPointBatch source_batch =
    regularization::MakeFrozenScalarIGRSourceBatch(source_state,
                                                   source_gradient,
                                                   source_stamp);
  mfem::Vector source_output(source_points);
  FillSentinel(source_output, -7.0);
  const regularization::RegularizationLaunchResult source_result =
    adapter.Launch(
      regularization::RegularizationOperationKind::AuxiliarySource,
      OperationContextFor(
        source_stamp,
        regularization::RegularizationOperationKind::AuxiliarySource),
      source_batch,
      OutputFor(regularization::RegularizationOperationKind::AuxiliarySource,
                source_output,
                source_batch));
  test.Check(source_result.launched && source_result.validation.Ok(),
             "source launches in dimension " + std::to_string(dimension));
  test.Check(
    source_batch.auxiliary.Storage() == nullptr &&
      OperationContextFor(
        source_stamp,
        regularization::RegularizationOperationKind::AuxiliarySource)
          .expected_producer_generations.auxiliary == 0u,
    "source launches with absent sigma and zero auxiliary authority");
  const mfem::real_t* source_state_data = source_state.Read(false);
  const mfem::real_t* source_gradient_data = source_gradient.Read(false);
  const mfem::real_t* source_data = source_output.Read(false);
  for (int point = 0; point < source_points; ++point)
  {
    const mfem::real_t expected = algebra::SourceFromConservativeStateGradient(
      dimension,
      source_state_data + point * equations,
      1,
      source_gradient_data + point * equations * dimension,
      1,
      equations,
      parameters.alpha,
      parameters.solver_density_floor);
    metrics.source =
      UpdateMaximum(metrics.source, source_data[point], expected);
  }

  const int volume_points = dimension + 4;
  mfem::Vector volume_state(volume_points * equations);
  mfem::Vector volume_sigma(volume_points);
  FillState(volume_state, volume_points, dimension);
  FillSigma(volume_sigma, volume_points);
  const auto volume_stamp = Stamp(volume_points,
                                  dimension,
                                  regularization::EvaluationSiteKind::Volume,
                                  3000u + dimension,
                                  300u + dimension);
  const regularization::EvaluatedPointBatch volume_batch =
    regularization::MakeFrozenScalarIGRVolumeCorrectionBatch(volume_state,
                                                             volume_sigma,
                                                             volume_stamp);
  mfem::Vector volume_output(volume_points * equations * dimension);
  FillSentinel(volume_output, 0.13);
  std::vector<mfem::real_t> expected_volume = Snapshot(volume_output);
  const mfem::real_t* volume_state_data = volume_state.Read(false);
  const mfem::real_t* volume_sigma_data = volume_sigma.Read(false);
  for (int point = 0; point < volume_points; ++point)
  {
    algebra::AddTensorSigmaFlux(dimension,
                                volume_state_data + point * equations,
                                1,
                                volume_sigma_data[point],
                                parameters.flux_density_floor,
                                expected_volume.data() +
                                  point * equations * dimension,
                                dimension,
                                1);
  }
  const regularization::RegularizationLaunchResult volume_result =
    adapter.Launch(
      regularization::RegularizationOperationKind::VolumeCorrection,
      OperationContextFor(
        volume_stamp,
        regularization::RegularizationOperationKind::VolumeCorrection),
      volume_batch,
      OutputFor(regularization::RegularizationOperationKind::VolumeCorrection,
                volume_output,
                volume_batch));
  test.Check(volume_result.launched && volume_result.validation.Ok(),
             "volume correction launches in dimension " +
               std::to_string(dimension));
  const mfem::real_t* volume_data = volume_output.Read(false);
  for (int index = 0; index < volume_output.Size(); ++index)
  {
    metrics.tensor =
      UpdateMaximum(metrics.tensor,
                    volume_data[index],
                    expected_volume[static_cast<std::size_t>(index)]);
  }

  const int normal_points = dimension + 5;
  mfem::Vector normal_state(normal_points * equations);
  mfem::Vector normal_sigma(normal_points);
  mfem::Vector normal_vector(normal_points * dimension);
  FillState(normal_state, normal_points, dimension);
  FillSigma(normal_sigma, normal_points);
  FillNormal(normal_vector, normal_points, dimension);
  const auto normal_stamp = Stamp(normal_points,
                                  dimension,
                                  regularization::EvaluationSiteKind::Face,
                                  4000u + dimension,
                                  400u + dimension);
  const regularization::EvaluatedPointBatch normal_batch =
    regularization::MakeFrozenScalarIGRNormalCorrectionBatch(normal_state,
                                                             normal_sigma,
                                                             normal_vector,
                                                             normal_stamp);
  mfem::Vector normal_output(normal_points * equations);
  FillSentinel(normal_output, -0.17);
  std::vector<mfem::real_t> expected_normal = Snapshot(normal_output);
  const mfem::real_t* normal_state_data = normal_state.Read(false);
  const mfem::real_t* normal_sigma_data = normal_sigma.Read(false);
  const mfem::real_t* normal_vector_data = normal_vector.Read(false);
  for (int point = 0; point < normal_points; ++point)
  {
    algebra::AddNormalSigmaFlux(dimension,
                                normal_state_data + point * equations,
                                1,
                                normal_vector_data + point * dimension,
                                1,
                                normal_sigma_data[point],
                                parameters.flux_density_floor,
                                expected_normal.data() + point * equations,
                                1);
  }
  const regularization::RegularizationLaunchResult normal_result =
    adapter.Launch(
      regularization::RegularizationOperationKind::NormalCorrection,
      OperationContextFor(
        normal_stamp,
        regularization::RegularizationOperationKind::NormalCorrection),
      normal_batch,
      OutputFor(regularization::RegularizationOperationKind::NormalCorrection,
                normal_output,
                normal_batch));
  test.Check(normal_result.launched && normal_result.validation.Ok(),
             "normal correction launches in dimension " +
               std::to_string(dimension));
  const mfem::real_t* normal_data = normal_output.Read(false);
  for (int index = 0; index < normal_output.Size(); ++index)
  {
    metrics.normal =
      UpdateMaximum(metrics.normal,
                    normal_data[index],
                    expected_normal[static_cast<std::size_t>(index)]);
  }

  const int bounds_points = dimension + 6;
  mfem::Vector bounds_state(bounds_points * equations);
  mfem::Vector bounds_sigma(bounds_points);
  mfem::Vector bounds_normal(bounds_points * dimension);
  mfem::Vector bounds_base_speed(bounds_points);
  FillState(bounds_state, bounds_points, dimension);
  FillSigma(bounds_sigma, bounds_points);
  FillNormal(bounds_normal, bounds_points, dimension);
  FillBaseCharacteristicSpeed(bounds_base_speed, bounds_points, dimension);
  const auto bounds_stamp = Stamp(bounds_points,
                                  dimension,
                                  regularization::EvaluationSiteKind::Face,
                                  5000u + dimension,
                                  500u + dimension);
  const regularization::EvaluatedPointBatch bounds_batch =
    regularization::MakeFrozenScalarIGRCharacteristicBoundsBatch(
      bounds_state,
      bounds_sigma,
      bounds_normal,
      bounds_base_speed,
      bounds_stamp);
  mfem::Vector bounds_output(2 * bounds_points);
  FillSentinel(bounds_output, -5.0);
  const regularization::RegularizationLaunchResult bounds_result =
    adapter.Launch(
      regularization::RegularizationOperationKind::CharacteristicBounds,
      OperationContextFor(
        bounds_stamp,
        regularization::RegularizationOperationKind::CharacteristicBounds),
      bounds_batch,
      OutputFor(
        regularization::RegularizationOperationKind::CharacteristicBounds,
        bounds_output,
        bounds_batch));
  test.Check(bounds_result.launched && bounds_result.validation.Ok(),
             "characteristic bounds launch in dimension " +
               std::to_string(dimension));
  const mfem::real_t* bounds_state_data = bounds_state.Read(false);
  const mfem::real_t* bounds_sigma_data = bounds_sigma.Read(false);
  const mfem::real_t* bounds_base_speed_data = bounds_base_speed.Read(false);
  const mfem::real_t* bounds_data = bounds_output.Read(false);
  for (int point = 0; point < bounds_points; ++point)
  {
    const mfem::real_t speed =
      algebra::FluxCharacteristicSpeed(bounds_base_speed_data[point],
                                       bounds_state_data[point * equations],
                                       bounds_sigma_data[point],
                                       parameters.flux_density_floor);
    metrics.characteristic =
      UpdateMaximum(metrics.characteristic, bounds_data[2 * point], -speed);
    metrics.characteristic =
      UpdateMaximum(metrics.characteristic, bounds_data[2 * point + 1], speed);
  }

  test.Check(bounds_points >= 3 &&
               bounds_base_speed_data[0] != bounds_base_speed_data[1] &&
               bounds_base_speed_data[1] != bounds_base_speed_data[2] &&
               bounds_data[1] != bounds_data[3] &&
               bounds_data[3] != bounds_data[5],
             "three distinct per-point base speeds produce distinct bounds");

  test.Check(coefficient_batch.point_count != source_batch.point_count &&
               source_batch.point_count != volume_batch.point_count &&
               volume_batch.point_count != normal_batch.point_count &&
               normal_batch.point_count != bounds_batch.point_count,
             "operation-specific point counts remain independent");
  test.Check(coefficient_batch.point_set_id != source_batch.point_set_id &&
               source_batch.point_set_id != volume_batch.point_set_id &&
               volume_batch.point_set_id != normal_batch.point_set_id &&
               normal_batch.point_set_id != bounds_batch.point_set_id,
             "operation-specific point-set identities remain independent");
  test.Check(
    coefficient_stamp.rk_stage == 0 &&
      source_stamp.producer_generations.state !=
        source_stamp.producer_generations.conservative_state_gradient &&
      normal_stamp.producer_generations.auxiliary !=
        normal_stamp.producer_generations.normal &&
      bounds_stamp.producer_generations.base_characteristic_speed !=
        bounds_stamp.producer_generations.normal,
    "stage zero and unequal required authoritative producer epochs launch");
}

void TestTwoStateAcceptedConsumer(TestContext& test, ParityMetrics& metrics)
{
  for (int dimension = 1; dimension <= 3; ++dimension)
  {
    const int equations = dimension + 2;
    std::vector<mfem::real_t> state1(static_cast<std::size_t>(equations), 0.0);
    std::vector<mfem::real_t> state2(static_cast<std::size_t>(equations), 0.0);
    std::vector<mfem::real_t> normal(static_cast<std::size_t>(dimension), 0.0);
    state1[0] = static_cast<mfem::real_t>(0.91 + 0.03 * dimension);
    state2[0] = static_cast<mfem::real_t>(1.37 + 0.05 * dimension);
    for (int direction = 0; direction < dimension; ++direction)
    {
      state1[static_cast<std::size_t>(1 + direction)] =
        static_cast<mfem::real_t>(0.2 * (direction + 1));
      state2[static_cast<std::size_t>(1 + direction)] =
        static_cast<mfem::real_t>(-0.17 * (direction + 2));
      normal[static_cast<std::size_t>(direction)] =
        static_cast<mfem::real_t>(0.3 + 0.1 * direction);
    }
    std::vector<mfem::real_t> flux(static_cast<std::size_t>(equations), 0.0);
    const mfem::real_t sigma = static_cast<mfem::real_t>(0.23);
    algebra::AddTwoStateAverageNormalSigmaFlux(dimension,
                                               state1.data(),
                                               1,
                                               state2.data(),
                                               1,
                                               normal.data(),
                                               1,
                                               sigma,
                                               flux.data(),
                                               1);
    const mfem::real_t expected_energy =
      algebra::ExactTwoStateAverageNormalSigmaEnergyTerm(dimension,
                                                         state1.data(),
                                                         1,
                                                         state2.data(),
                                                         1,
                                                         normal.data(),
                                                         1,
                                                         sigma);
    metrics.two_state =
      UpdateMaximum(metrics.two_state,
                    flux[static_cast<std::size_t>(equations - 1)],
                    expected_energy);
    test.Check(std::isfinite(expected_energy),
               "accepted two-state consumer remains finite");
  }
}

void TestRestartTruthfulness(TestContext& test)
{
  mfem::Vector sigma(4);
  FillSigma(sigma, 4);
  const regularization::AuxiliaryFieldProvenance provenance{0, 71u, 89u};
  const regularization::ScalarAuxiliaryFieldDescriptor descriptor =
    regularization::MakeFrozenScalarIGRSigmaDescriptor(sigma, 4, 2, provenance);
  const regularization::AuxiliaryFieldBinding binding =
    regularization::BindAuxiliaryField(descriptor);
  const regularization::ValidationResult validation =
    regularization::ValidateAuxiliaryFieldBinding(binding);
  test.Check(validation.Ok(),
             "ColdRecomputed scalar sigma descriptor validates at stage zero");
  test.Check(binding.restart_kind ==
                 regularization::AuxiliaryRestartKind::ColdRecomputed &&
               binding.identity.semantic_id != 0u &&
               binding.identity.restart_id == 0u &&
               binding.identity.output_id == 0u,
             "restart identity is honestly ColdRecomputed/zero");
  test.Check(
    !binding.capabilities.Has(
      regularization::AuxiliaryCapability::DeviceAccessible) &&
      !binding.capabilities.Has(
        regularization::AuxiliaryCapability::OutputVisible) &&
      !binding.capabilities.Has(
        regularization::AuxiliaryCapability::TraceCoupled) &&
      binding.capabilities.Has(regularization::AuxiliaryCapability::StageLocal),
    "sigma claims only the stage-local capability");
}

void TestExplicitMFEMVectorMemoryPath(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter)
{
  constexpr int dimension = 1;
  constexpr int equations = dimension + 2;
  constexpr int points = 5;
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(2);
  mfem::L2_FECollection collection(1, dimension);
  mfem::FiniteElementSpace space(&mesh, &collection, equations);
  mfem::GridFunction raw_fe_dofs(&space);
  raw_fe_dofs = 17.0;

  mfem::Vector evaluated_state(points * equations);
  FillState(evaluated_state, points, dimension);
  const auto stamp = Stamp(points,
                           dimension,
                           regularization::EvaluationSiteKind::Volume,
                           6101u,
                           601u);
  const regularization::EvaluatedPointBatch batch =
    regularization::MakeFrozenScalarIGRCoefficientBatch(evaluated_state,
                                                        stamp);
  mfem::Vector output(points);
  FillSentinel(output, -9.0);
  const regularization::RegularizationLaunchResult result = adapter.Launch(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    OperationContextFor(
      stamp,
      regularization::RegularizationOperationKind::AuxiliaryCoefficient),
    batch,
    OutputFor(regularization::RegularizationOperationKind::AuxiliaryCoefficient,
              output,
              batch));
  test.Check(result.launched && result.validation.Ok(),
             "actual MFEM Vector/Memory CPU evaluated-point path launches");
  test.Check(
    raw_fe_dofs.Size() != points && batch.point_count == points &&
      batch.state.Storage() == &evaluated_state &&
      batch.state.Storage() != static_cast<const mfem::Vector*>(&raw_fe_dofs),
    "raw GridFunction DOF Size is not used as an evaluated point count");
}

void TestUnsupportedSelectionsAndCapabilities(TestContext& test)
{
  const regularization::FrozenScalarIGRParameters parameters =
    ValidParameters();
  const regularization::FrozenScalarIGRAdapterSelection supported =
    EnabledSelection();
  mfem::Vector sentinel(3);
  FillSentinel(sentinel, 4.0);
  const std::vector<mfem::real_t> before = Snapshot(sentinel);

  auto check_selection =
    [&](const regularization::FrozenScalarIGRAdapterSelection& selection,
        const std::string& label)
  {
    auto adapter =
      regularization::ConfigureFrozenScalarIGRBrokenDGCPUAdapter(selection,
                                                                 parameters);
    test.Code(adapter.ConfigurationValidation().code,
              regularization::ValidationCode::UnsupportedCombination,
              label + " rejected before static binding");
    const regularization::RegularizationLaunchResult result = adapter.Launch(
      regularization::RegularizationOperationKind::AuxiliaryCoefficient,
      {},
      {},
      {});
    test.Check(!result.launched && adapter.LaunchCount() == 0 &&
                 adapter.HostSelectionCount() == 0,
               label + " never launches or selects a kernel");
    CheckSnapshot(test, sentinel, before, label + " preserves sentinel");
  };

  const regularization::FrozenScalarIGRAdapterSelection default_selection;
  test.Check(
    default_selection.model ==
      regularization::FrozenScalarIGRAdapterModelKind::Disabled,
    "default construction is explicitly disabled");
  check_selection(default_selection, "default disabled model");

  auto selection = supported;
  selection.model = regularization::FrozenScalarIGRAdapterModelKind::Disabled;
  check_selection(selection, "explicitly disabled model");
  selection = supported;
  selection.gradient_treatment = regularization::GradientTreatmentKind::None;
  check_selection(selection, "H1/None gradient");
  selection = supported;
  selection.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::TraceL2;
  check_selection(selection, "TraceL2");
  selection = supported;
  selection.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::HybridizedTrace;
  check_selection(selection, "HybridizedTrace");
  selection = supported;
  selection.auxiliary_solution =
    regularization::AuxiliarySolutionKind::PointwiseEvaluation;
  check_selection(selection, "PointwiseEvaluation");
  selection = supported;
  selection.auxiliary_solution =
    regularization::AuxiliarySolutionKind::LocalElimination;
  check_selection(selection, "LocalElimination");
  selection = supported;
  selection.backend.spatial = regularization::SpatialBackendKind::DGSEM;
  check_selection(selection, "DGSEM");
  selection = supported;
  selection.backend.spatial = regularization::SpatialBackendKind::HDG;
  check_selection(selection, "HDG");
  selection = supported;
  selection.backend.execution = regularization::ExecutionTargetKind::MFEMDevice;
  check_selection(selection, "MFEMDevice");
  selection = supported;
  selection.compatibility.time = regularization::TimeTreatmentKind::LaggedStep;
  check_selection(selection, "LaggedStep");
  selection = supported;
  selection.compatibility.time =
    regularization::TimeTreatmentKind::FullyCoupled;
  check_selection(selection, "FullyCoupled");
  selection = supported;
  selection.compatibility.entropy =
    regularization::EntropyTreatmentKind::EntropyConservative;
  check_selection(selection, "EntropyConservative");
  selection = supported;
  selection.compatibility.entropy =
    regularization::EntropyTreatmentKind::EntropyStable;
  check_selection(selection, "EntropyStable");

  auto adapter =
    regularization::ConfigureFrozenScalarIGRBrokenDGCPUAdapter(supported,
                                                               parameters);
  mfem::Vector wall_output(4);
  mfem::Vector diagnostics_output(3);
  FillSentinel(wall_output, 51.0);
  FillSentinel(diagnostics_output, 52.0);
  const auto wall_before = Snapshot(wall_output);
  const auto diagnostics_before = Snapshot(diagnostics_output);
  regularization::RegularizationOperationOutputs wall_outputs;
  wall_outputs.wall_flux = regularization::BindEvaluatedPointOutput(
    wall_output,
    1,
    4,
    2,
    regularization::EvaluationSiteKind::Wall,
    0x8801u);
  regularization::RegularizationOperationOutputs diagnostics_outputs;
  diagnostics_outputs.diagnostics = regularization::BindEvaluatedPointOutput(
    diagnostics_output,
    1,
    3,
    2,
    regularization::EvaluationSiteKind::Volume,
    0x8802u);
  const regularization::RegularizationLaunchResult wall =
    adapter.Launch(regularization::RegularizationOperationKind::WallLaw,
                   {},
                   {},
                   wall_outputs);
  const regularization::RegularizationLaunchResult diagnostics =
    adapter.Launch(regularization::RegularizationOperationKind::Diagnostics,
                   {},
                   {},
                   diagnostics_outputs);
  test.Code(wall.validation.code,
            regularization::ValidationCode::UnsupportedCapability,
            "wall law rejected before batch/output access");
  test.Code(diagnostics.validation.code,
            regularization::ValidationCode::UnsupportedCapability,
            "diagnostics rejected before batch/output access");
  test.Check(!wall.launched && !diagnostics.launched &&
               adapter.LaunchCount() == 0,
             "wall and diagnostics never launch");
  CheckSnapshot(
    test, wall_output, wall_before, "wall rejection preserves sentinel");
  CheckSnapshot(test,
                diagnostics_output,
                diagnostics_before,
                "diagnostics rejection preserves sentinel");
}

void TestInvalidParameters(TestContext& test)
{
  const regularization::FrozenScalarIGRAdapterSelection selection =
    EnabledSelection();
  const regularization::FrozenScalarIGRParameters valid = ValidParameters();
  mfem::Vector sentinel(2);
  FillSentinel(sentinel, 6.0);
  const std::vector<mfem::real_t> before = Snapshot(sentinel);

  auto check = [&](regularization::FrozenScalarIGRParameters parameters,
                   const std::string& label)
  {
    auto adapter =
      regularization::ConfigureFrozenScalarIGRBrokenDGCPUAdapter(selection,
                                                                 parameters);
    test.Code(adapter.ConfigurationValidation().code,
              regularization::ValidationCode::InvalidModelShape,
              label + " fails model parameter preflight");
    const auto result = adapter.Launch(
      regularization::RegularizationOperationKind::AuxiliaryCoefficient,
      {},
      {},
      {});
    test.Check(!result.launched && adapter.HostSelectionCount() == 0 &&
                 adapter.LaunchCount() == 0,
               label + " fails before MFEM access");
    CheckSnapshot(test, sentinel, before, label + " preserves sentinel");
  };

  auto parameters = valid;
  parameters.alpha = -1.0;
  check(parameters, "negative alpha");
  parameters = valid;
  parameters.alpha = std::numeric_limits<mfem::real_t>::quiet_NaN();
  check(parameters, "NaN alpha");
  parameters = valid;
  parameters.solver_density_floor = 0.0;
  check(parameters, "zero solver density floor");
  parameters = valid;
  parameters.flux_density_floor = -1.0;
  check(parameters, "negative flux density floor");
  parameters = valid;
  parameters.solver_density_floor =
    std::numeric_limits<mfem::real_t>::infinity();
  check(parameters, "infinite solver density floor");
}

struct NegativeFixture
{
  static constexpr int dimension = 2;
  static constexpr int equations = dimension + 2;
  static constexpr int points = 3;

  mfem::Vector state;
  mfem::Vector gradient;
  mfem::Vector sigma;
  mfem::Vector normal;
  mfem::Vector base_speed;
  regularization::FrozenScalarIGREvaluatedPointStamp volume_stamp =
    Stamp(points,
          dimension,
          regularization::EvaluationSiteKind::Volume,
          7001u,
          701u);
  regularization::FrozenScalarIGREvaluatedPointStamp face_stamp =
    Stamp(points,
          dimension,
          regularization::EvaluationSiteKind::Face,
          7002u,
          801u);

  NegativeFixture()
    : state(points * equations),
      gradient(points * equations * dimension),
      sigma(points),
      normal(points * dimension),
      base_speed(points)
  {
    FillState(state, points, dimension);
    FillGradient(gradient, points, dimension);
    FillSigma(sigma, points);
    FillNormal(normal, points, dimension);
    FillBaseCharacteristicSpeed(base_speed, points, dimension);
  }

  regularization::EvaluatedPointBatch Coefficient() const
  {
    return regularization::MakeFrozenScalarIGRCoefficientBatch(state,
                                                               volume_stamp);
  }

  regularization::EvaluatedPointBatch Source() const
  {
    return regularization::MakeFrozenScalarIGRSourceBatch(state,
                                                          gradient,
                                                          volume_stamp);
  }

  regularization::EvaluatedPointBatch Volume() const
  {
    return regularization::MakeFrozenScalarIGRVolumeCorrectionBatch(state,
                                                                    sigma,
                                                                    volume_stamp);
  }

  regularization::EvaluatedPointBatch Normal() const
  {
    return regularization::MakeFrozenScalarIGRNormalCorrectionBatch(state,
                                                                    sigma,
                                                                    normal,
                                                                    face_stamp);
  }

  regularization::EvaluatedPointBatch Bounds() const
  {
    return regularization::MakeFrozenScalarIGRCharacteristicBoundsBatch(
      state,
      sigma,
      normal,
      base_speed,
      face_stamp);
  }
};

void ExpectFailurePreserves(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter,
  regularization::RegularizationOperationKind operation,
  const regularization::RegularizationValidationContext& context,
  const regularization::EvaluatedPointBatch& batch,
  regularization::ValidationCode expected_code,
  const std::string& label)
{
  int output_components = 1;
  if (operation ==
      regularization::RegularizationOperationKind::NormalCorrection)
  {
    output_components = batch.dimension + 2;
  }
  else if (operation ==
           regularization::RegularizationOperationKind::VolumeCorrection)
  {
    output_components =
      batch.dimension > 0 && batch.dimension < 4 ? batch.dimension *
                                                    (batch.dimension + 2)
                                                : 1;
  }
  else if (operation ==
           regularization::RegularizationOperationKind::CharacteristicBounds)
  {
    output_components = 2;
  }
  const int safe_points =
    batch.point_count > 0 && batch.point_count < 1000 ? batch.point_count : 1;
  mfem::Vector output(safe_points * output_components);
  FillSentinel(output, 9.0);
  const std::vector<mfem::real_t> before = Snapshot(output);
  const int launch_count = adapter.LaunchCount();
  const regularization::RegularizationLaunchResult result =
    adapter.Launch(operation,
                   context,
                   batch,
                   OutputFor(operation, output, batch));
  test.Check(!result.launched, label + " does not launch");
  test.Code(result.validation.code, expected_code, label);
  test.Check(adapter.LaunchCount() == launch_count,
             label + " leaves launch count unchanged");
  CheckSnapshot(test, output, before, label + " preserves writable output");
}

void TestOperationLocalRequirements(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter)
{
  NegativeFixture fixture;

  const regularization::EvaluatedPointBatch coefficient = fixture.Coefficient();
  mfem::Vector coefficient_output(NegativeFixture::points);
  FillSentinel(coefficient_output, -31.0);
  const auto coefficient_context = OperationContextFor(
    fixture.volume_stamp,
    regularization::RegularizationOperationKind::AuxiliaryCoefficient);
  const auto coefficient_result = adapter.Launch(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    coefficient_context,
    coefficient,
    OutputFor(regularization::RegularizationOperationKind::AuxiliaryCoefficient,
              coefficient_output,
              coefficient));
  test.Check(coefficient_result.launched &&
               coefficient_result.validation.Ok() &&
               coefficient.auxiliary.Storage() == nullptr &&
               coefficient_context.expected_producer_generations.auxiliary ==
                 0u,
             "coefficient requires only state and no sigma authority");

  const regularization::EvaluatedPointBatch source = fixture.Source();
  mfem::Vector source_output(NegativeFixture::points);
  FillSentinel(source_output, -32.0);
  const auto source_context = OperationContextFor(
    fixture.volume_stamp,
    regularization::RegularizationOperationKind::AuxiliarySource);
  const auto source_result = adapter.Launch(
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    OutputFor(regularization::RegularizationOperationKind::AuxiliarySource,
              source_output,
              source));
  test.Check(source_result.launched && source_result.validation.Ok() &&
               source.auxiliary.Storage() == nullptr &&
               source_context.expected_producer_generations.auxiliary == 0u,
             "source requires only state/gradient and no sigma authority");
}

void TestRequiredFieldFailures(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter)
{
  NegativeFixture fixture;

  auto source = fixture.Source();
  auto metadata = source.state.Metadata();
  source.state = regularization::EvaluatedPointFieldBinding::Missing(metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    source,
    regularization::ValidationCode::MissingBuffer,
    "missing state storage");

  source = fixture.Source();
  metadata = source.state.Metadata();
  metadata.component_extent = NegativeFixture::equations - 1;
  mfem::Vector wrong_state(NegativeFixture::points *
                           (NegativeFixture::equations - 1));
  FillSentinel(wrong_state, 1.0);
  source.state =
    regularization::BindEvaluatedPointField(wrong_state, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    source,
    regularization::ValidationCode::InvalidModelShape,
    "wrong Euler state extent");

  source = fixture.Source();
  mfem::Vector short_state(NegativeFixture::points *
                             NegativeFixture::equations -
                           1);
  metadata = source.state.Metadata();
  source.state =
    regularization::BindEvaluatedPointField(short_state, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    source,
    regularization::ValidationCode::InvalidBufferSize,
    "wrong state storage size");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  source.conservative_state_gradient =
    regularization::EvaluatedPointFieldBinding::Missing(metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    source,
    regularization::ValidationCode::MissingBuffer,
    "missing conservative gradient storage");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  --metadata.component_extent;
  mfem::Vector wrong_gradient(NegativeFixture::points *
                              metadata.component_extent);
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(wrong_gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    source,
    regularization::ValidationCode::InvalidModelShape,
    "wrong conservative gradient extent");

  source = fixture.Source();
  mfem::Vector short_gradient(fixture.gradient.Size() - 1);
  metadata = source.conservative_state_gradient.Metadata();
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(short_gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    source,
    regularization::ValidationCode::InvalidBufferSize,
    "wrong conservative gradient storage size");

  auto volume = fixture.Volume();
  metadata = volume.auxiliary.Metadata();
  volume.auxiliary =
    regularization::EvaluatedPointFieldBinding::Missing(metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::VolumeCorrection,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::VolumeCorrection),
    volume,
    regularization::ValidationCode::MissingBuffer,
    "missing sigma storage");

  volume = fixture.Volume();
  metadata = volume.auxiliary.Metadata();
  metadata.component_extent = 2;
  mfem::Vector vector_sigma(NegativeFixture::points * 2);
  volume.auxiliary =
    regularization::BindEvaluatedPointField(vector_sigma, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::VolumeCorrection,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::VolumeCorrection),
    volume,
    regularization::ValidationCode::InvalidModelShape,
    "wrong scalar sigma extent");

  volume = fixture.Volume();
  mfem::Vector short_sigma(NegativeFixture::points - 1);
  metadata = volume.auxiliary.Metadata();
  volume.auxiliary =
    regularization::BindEvaluatedPointField(short_sigma, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::VolumeCorrection,
    OperationContextFor(
      fixture.volume_stamp,
      regularization::RegularizationOperationKind::VolumeCorrection),
    volume,
    regularization::ValidationCode::InvalidBufferSize,
    "wrong sigma storage size");

  auto normal = fixture.Normal();
  metadata = normal.normal.Metadata();
  normal.normal =
    regularization::EvaluatedPointFieldBinding::Missing(metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::NormalCorrection,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::NormalCorrection),
    normal,
    regularization::ValidationCode::MissingBuffer,
    "missing normal storage");

  normal = fixture.Normal();
  metadata = normal.normal.Metadata();
  metadata.component_extent = NegativeFixture::dimension - 1;
  mfem::Vector scalar_normal(NegativeFixture::points);
  normal.normal =
    regularization::BindEvaluatedPointField(scalar_normal, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::NormalCorrection,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::NormalCorrection),
    normal,
    regularization::ValidationCode::InvalidModelShape,
    "wrong normal extent");

  normal = fixture.Normal();
  mfem::Vector short_normal(fixture.normal.Size() - 1);
  metadata = normal.normal.Metadata();
  normal.normal =
    regularization::BindEvaluatedPointField(short_normal, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::NormalCorrection,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::NormalCorrection),
    normal,
    regularization::ValidationCode::InvalidBufferSize,
    "wrong normal storage size");

  auto bounds = fixture.Bounds();
  metadata = bounds.base_characteristic_speed.Metadata();
  bounds.base_characteristic_speed =
    regularization::EvaluatedPointFieldBinding::Missing(metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::CharacteristicBounds,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::CharacteristicBounds),
    bounds,
    regularization::ValidationCode::MissingBuffer,
    "missing base-speed storage");

  bounds = fixture.Bounds();
  metadata = bounds.base_characteristic_speed.Metadata();
  metadata.component_extent = 2;
  mfem::Vector vector_base_speed(NegativeFixture::points * 2);
  bounds.base_characteristic_speed =
    regularization::BindEvaluatedPointField(vector_base_speed, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::CharacteristicBounds,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::CharacteristicBounds),
    bounds,
    regularization::ValidationCode::InvalidModelShape,
    "wrong scalar base-speed extent");

  bounds = fixture.Bounds();
  mfem::Vector short_base_speed(NegativeFixture::points - 1);
  metadata = bounds.base_characteristic_speed.Metadata();
  bounds.base_characteristic_speed =
    regularization::BindEvaluatedPointField(short_base_speed, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::CharacteristicBounds,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::CharacteristicBounds),
    bounds,
    regularization::ValidationCode::InvalidBufferSize,
    "wrong base-speed storage size");
}

void TestIdentityAndProvenanceFailures(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter)
{
  NegativeFixture fixture;
  const auto source_context = OperationContextFor(
    fixture.volume_stamp,
    regularization::RegularizationOperationKind::AuxiliarySource);
  auto source = fixture.Source();
  auto metadata = source.conservative_state_gradient.Metadata();

  metadata.site = regularization::EvaluationSiteKind::Face;
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(fixture.gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    regularization::ValidationCode::InvalidEvaluationSite,
    "gradient site mismatch");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  ++metadata.point_count;
  mfem::Vector count_gradient(metadata.point_count * metadata.component_extent);
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(count_gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    regularization::ValidationCode::InvalidPointCount,
    "gradient point-count mismatch");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  metadata.dimension = 3;
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(fixture.gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    regularization::ValidationCode::InvalidDimension,
    "gradient dimension mismatch");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  ++metadata.point_set_id;
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(fixture.gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    regularization::ValidationCode::InvalidPointSetIdentity,
    "gradient point-set mismatch");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  metadata.rk_stage = -1;
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(fixture.gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    regularization::ValidationCode::InvalidProvenanceStamp,
    "gradient stage -1");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  ++metadata.rk_stage;
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(fixture.gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    regularization::ValidationCode::StaleProvenance,
    "stale gradient stage");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  metadata.state_generation = 0u;
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(fixture.gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    regularization::ValidationCode::InvalidProvenanceStamp,
    "zero gradient state generation");

  source = fixture.Source();
  metadata = source.conservative_state_gradient.Metadata();
  ++metadata.state_generation;
  source.conservative_state_gradient =
    regularization::BindEvaluatedPointField(fixture.gradient, metadata);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    source,
    regularization::ValidationCode::StaleProvenance,
    "stale gradient state generation");

  auto context = source_context;
  context.expected_rk_stage = -1;
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    context,
    fixture.Source(),
    regularization::ValidationCode::InvalidProvenanceStamp,
    "expected stage -1");

  context = source_context;
  ++context.expected_rk_stage;
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    context,
    fixture.Source(),
    regularization::ValidationCode::StaleProvenance,
    "stale expected stage");

  context = source_context;
  context.expected_state_generation = 0u;
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    context,
    fixture.Source(),
    regularization::ValidationCode::InvalidProvenanceStamp,
    "zero expected state generation");

  context = source_context;
  ++context.expected_state_generation;
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    context,
    fixture.Source(),
    regularization::ValidationCode::StaleProvenance,
    "stale expected state generation");

  auto batch = fixture.Source();
  batch.rk_stage = -1;
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    batch,
    regularization::ValidationCode::InvalidProvenanceStamp,
    "batch stage -1");

  batch = fixture.Source();
  batch.state_generation = 0u;
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    source_context,
    batch,
    regularization::ValidationCode::InvalidProvenanceStamp,
    "zero batch state generation");

  auto bad_stamp = fixture.volume_stamp;
  bad_stamp.point_count = 0;
  mfem::Vector no_state(0);
  mfem::Vector no_gradient(0);
  batch = regularization::MakeFrozenScalarIGRSourceBatch(no_state,
                                                        no_gradient,
                                                        bad_stamp);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      bad_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    batch,
    regularization::ValidationCode::InvalidPointCount,
    "zero batch point count");

  bad_stamp = fixture.volume_stamp;
  bad_stamp.dimension = 4;
  mfem::Vector four_d_state(bad_stamp.point_count *
                            (bad_stamp.dimension + 2));
  mfem::Vector four_d_gradient(
    bad_stamp.point_count * (bad_stamp.dimension + 2) * bad_stamp.dimension);
  batch = regularization::MakeFrozenScalarIGRSourceBatch(four_d_state,
                                                        four_d_gradient,
                                                        bad_stamp);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      bad_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    batch,
    regularization::ValidationCode::InvalidDimension,
    "invalid batch dimension");

  bad_stamp = fixture.volume_stamp;
  bad_stamp.site = regularization::EvaluationSiteKind::Face;
  batch = regularization::MakeFrozenScalarIGRSourceBatch(fixture.state,
                                                        fixture.gradient,
                                                        bad_stamp);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      bad_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    batch,
    regularization::ValidationCode::InvalidEvaluationSite,
    "wrong operation site");

  bad_stamp = fixture.volume_stamp;
  bad_stamp.point_set_id = 0u;
  batch = regularization::MakeFrozenScalarIGRSourceBatch(fixture.state,
                                                        fixture.gradient,
                                                        bad_stamp);
  ExpectFailurePreserves(
    test,
    adapter,
    regularization::RegularizationOperationKind::AuxiliarySource,
    OperationContextFor(
      bad_stamp,
      regularization::RegularizationOperationKind::AuxiliarySource),
    batch,
    regularization::ValidationCode::InvalidPointSetIdentity,
    "zero batch point-set identity");
}

void TestProducerAuthorities(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter)
{
  NegativeFixture fixture;

  auto check = [&](regularization::RegularizationOperationKind operation,
                   const regularization::FrozenScalarIGREvaluatedPointStamp&
                     stamp,
                   const regularization::EvaluatedPointBatch& batch,
                   std::uint64_t regularization::ExpectedEvaluatedPointProducerGenerations::*
                     member,
                   const std::string& label)
  {
    auto context = OperationContextFor(stamp, operation);
    ++(context.expected_producer_generations.*member);
    ExpectFailurePreserves(test,
                           adapter,
                           operation,
                           context,
                           batch,
                           regularization::ValidationCode::StaleProvenance,
                           "wrong " + label + " producer authority");
    context = OperationContextFor(stamp, operation);
    context.expected_producer_generations.*member = 0u;
    ExpectFailurePreserves(
      test,
      adapter,
      operation,
      context,
      batch,
      regularization::ValidationCode::InvalidProvenanceStamp,
      "zero " + label + " producer authority");
  };

  check(regularization::RegularizationOperationKind::AuxiliarySource,
        fixture.volume_stamp,
        fixture.Source(),
        &regularization::ExpectedEvaluatedPointProducerGenerations::state,
        "state");
  check(
    regularization::RegularizationOperationKind::AuxiliarySource,
    fixture.volume_stamp,
    fixture.Source(),
    &regularization::ExpectedEvaluatedPointProducerGenerations::
      conservative_state_gradient,
    "gradient");
  check(regularization::RegularizationOperationKind::VolumeCorrection,
        fixture.volume_stamp,
        fixture.Volume(),
        &regularization::ExpectedEvaluatedPointProducerGenerations::auxiliary,
        "sigma");
  check(regularization::RegularizationOperationKind::NormalCorrection,
        fixture.face_stamp,
        fixture.Normal(),
        &regularization::ExpectedEvaluatedPointProducerGenerations::normal,
        "normal");
  check(
    regularization::RegularizationOperationKind::CharacteristicBounds,
    fixture.face_stamp,
    fixture.Bounds(),
    &regularization::ExpectedEvaluatedPointProducerGenerations::
      base_characteristic_speed,
    "base-speed");

  auto stale_binding =
    [&](regularization::RegularizationOperationKind operation,
        const regularization::FrozenScalarIGREvaluatedPointStamp& stamp,
        regularization::EvaluatedPointBatch batch,
        regularization::EvaluatedPointFieldBinding
          regularization::EvaluatedPointBatch::*field,
        const mfem::Vector& storage,
        const std::string& label)
  {
    auto metadata = (batch.*field).Metadata();
    ++metadata.producer_generation;
    batch.*field =
      regularization::BindEvaluatedPointField(storage, metadata);
    ExpectFailurePreserves(test,
                           adapter,
                           operation,
                           OperationContextFor(stamp, operation),
                           batch,
                           regularization::ValidationCode::StaleProvenance,
                           "stale " + label + " producer binding");
  };

  stale_binding(
    regularization::RegularizationOperationKind::AuxiliarySource,
    fixture.volume_stamp,
    fixture.Source(),
    &regularization::EvaluatedPointBatch::state,
    fixture.state,
    "state");
  stale_binding(
    regularization::RegularizationOperationKind::AuxiliarySource,
    fixture.volume_stamp,
    fixture.Source(),
    &regularization::EvaluatedPointBatch::conservative_state_gradient,
    fixture.gradient,
    "gradient");
  stale_binding(
    regularization::RegularizationOperationKind::VolumeCorrection,
    fixture.volume_stamp,
    fixture.Volume(),
    &regularization::EvaluatedPointBatch::auxiliary,
    fixture.sigma,
    "sigma");
  stale_binding(
    regularization::RegularizationOperationKind::NormalCorrection,
    fixture.face_stamp,
    fixture.Normal(),
    &regularization::EvaluatedPointBatch::normal,
    fixture.normal,
    "normal");
  stale_binding(
    regularization::RegularizationOperationKind::CharacteristicBounds,
    fixture.face_stamp,
    fixture.Bounds(),
    &regularization::EvaluatedPointBatch::base_characteristic_speed,
    fixture.base_speed,
    "base-speed");
}

void TestSigmaAwareCharacteristicDefault(TestContext& test)
{
  NegativeFixture fixture;
  auto parameters = ValidParameters();
  parameters.sigma_aware_characteristic_speed = false;
  auto adapter = regularization::ConfigureFrozenScalarIGRBrokenDGCPUAdapter(
    EnabledSelection(),
    parameters);
  auto batch = fixture.Bounds();
  mfem::Vector output(2 * NegativeFixture::points);
  FillSentinel(output, -61.0);
  const auto result = adapter.Launch(
    regularization::RegularizationOperationKind::CharacteristicBounds,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::CharacteristicBounds),
    batch,
    OutputFor(regularization::RegularizationOperationKind::CharacteristicBounds,
              output,
              batch));
  test.Check(result.launched && result.validation.Ok(),
             "sigma-unaware default bounds launch");
  const mfem::real_t* actual = output.Read(false);
  const mfem::real_t* base = fixture.base_speed.Read(false);
  for (int point = 0; point < NegativeFixture::points; ++point)
  {
    test.Near(actual[2 * point], -base[point], 0.0,
              "sigma-unaware lower bound uses per-point base speed");
    test.Near(actual[2 * point + 1], base[point], 0.0,
              "sigma-unaware upper bound uses per-point base speed");
  }
}

void TestOutputFailures(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter)
{
  NegativeFixture fixture;
  const auto operation =
    regularization::RegularizationOperationKind::AuxiliarySource;
  const auto batch = fixture.Source();
  const auto context = OperationContextFor(fixture.volume_stamp, operation);

  auto run = [&](const regularization::EvaluatedPointOutputBinding& binding,
                 mfem::Vector& output,
                 regularization::ValidationCode code,
                 const std::string& label)
  {
    regularization::RegularizationOperationOutputs outputs;
    outputs.auxiliary_source = binding;
    const auto before = Snapshot(output);
    const int count = adapter.LaunchCount();
    const auto result = adapter.Launch(operation, context, batch, outputs);
    test.Check(!result.launched && adapter.LaunchCount() == count,
               label + " does not launch");
    test.Code(result.validation.code, code, label);
    CheckSnapshot(test, output, before, label + " preserves output");
  };

  mfem::Vector correct(NegativeFixture::points);
  FillSentinel(correct, 44.0);
  run(regularization::EvaluatedPointOutputBinding::Missing(
        batch.point_count,
        1,
        batch.dimension,
        batch.site,
        batch.point_set_id),
      correct,
      regularization::ValidationCode::MissingBuffer,
      "missing output storage");

  mfem::Vector short_output(NegativeFixture::points - 1);
  FillSentinel(short_output, 45.0);
  run(regularization::BindEvaluatedPointOutput(short_output,
                                               batch.point_count,
                                               1,
                                               batch.dimension,
                                               batch.site,
                                               batch.point_set_id),
      short_output,
      regularization::ValidationCode::InvalidBufferSize,
      "wrong output storage size");

  mfem::Vector site_output(NegativeFixture::points);
  FillSentinel(site_output, 46.0);
  run(regularization::BindEvaluatedPointOutput(
        site_output,
        batch.point_count,
        1,
        batch.dimension,
        regularization::EvaluationSiteKind::Face,
        batch.point_set_id),
      site_output,
      regularization::ValidationCode::InvalidEvaluationSite,
      "wrong output site");

  mfem::Vector point_set_output(NegativeFixture::points);
  FillSentinel(point_set_output, 47.0);
  run(regularization::BindEvaluatedPointOutput(point_set_output,
                                               batch.point_count,
                                               1,
                                               batch.dimension,
                                               batch.site,
                                               batch.point_set_id + 1u),
      point_set_output,
      regularization::ValidationCode::InvalidPointSetIdentity,
      "wrong output point-set identity");

  mfem::Vector count_output(NegativeFixture::points + 1);
  FillSentinel(count_output, 48.0);
  run(regularization::BindEvaluatedPointOutput(count_output,
                                               batch.point_count + 1,
                                               1,
                                               batch.dimension,
                                               batch.site,
                                               batch.point_set_id),
      count_output,
      regularization::ValidationCode::InvalidBufferSize,
      "wrong output point count");

  mfem::Vector dimension_output(NegativeFixture::points);
  FillSentinel(dimension_output, 49.0);
  run(regularization::BindEvaluatedPointOutput(dimension_output,
                                               batch.point_count,
                                               1,
                                               batch.dimension + 1,
                                               batch.site,
                                               batch.point_set_id),
      dimension_output,
      regularization::ValidationCode::InvalidBufferSize,
      "wrong output dimension");
}

void TestAliasesAndOverflow(
  TestContext& test,
  regularization::FrozenScalarIGRBrokenDGCPUAdapter& adapter)
{
  NegativeFixture fixture;

  const auto normal_batch = fixture.Normal();
  regularization::RegularizationOperationOutputs state_alias_outputs;
  state_alias_outputs.normal_correction =
    regularization::BindEvaluatedPointOutput(fixture.state,
                                             normal_batch.point_count,
                                             NegativeFixture::equations,
                                             normal_batch.dimension,
                                             normal_batch.site,
                                             normal_batch.point_set_id);
  auto state_before = Snapshot(fixture.state);
  int launch_count = adapter.LaunchCount();
  auto result = adapter.Launch(
    regularization::RegularizationOperationKind::NormalCorrection,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::NormalCorrection),
    normal_batch,
    state_alias_outputs);
  test.Code(result.validation.code,
            regularization::ValidationCode::InvalidBufferAliasing,
            "state/output alias");
  test.Check(!result.launched && adapter.LaunchCount() == launch_count,
             "state alias does not launch");
  CheckSnapshot(test, fixture.state, state_before, "state alias preserves data");

  const auto bounds_batch = fixture.Bounds();
  regularization::RegularizationOperationOutputs normal_alias_outputs;
  normal_alias_outputs.characteristic_bounds =
    regularization::BindEvaluatedPointOutput(fixture.normal,
                                             bounds_batch.point_count,
                                             2,
                                             bounds_batch.dimension,
                                             bounds_batch.site,
                                             bounds_batch.point_set_id);
  const auto normal_before = Snapshot(fixture.normal);
  launch_count = adapter.LaunchCount();
  result = adapter.Launch(
    regularization::RegularizationOperationKind::CharacteristicBounds,
    OperationContextFor(
      fixture.face_stamp,
      regularization::RegularizationOperationKind::CharacteristicBounds),
    bounds_batch,
    normal_alias_outputs);
  test.Code(result.validation.code,
            regularization::ValidationCode::InvalidBufferAliasing,
            "normal/output alias");
  test.Check(!result.launched && adapter.LaunchCount() == launch_count,
             "normal alias does not launch");
  CheckSnapshot(
    test, fixture.normal, normal_before, "normal alias preserves data");

  constexpr int dimension = 3;
  constexpr int equations = dimension + 2;
  const int huge_points = std::numeric_limits<int>::max();
  mfem::Vector tiny_state(1);
  mfem::Vector tiny_sigma(1);
  mfem::Vector tiny_output(1);
  FillSentinel(tiny_state, 1.0);
  FillSentinel(tiny_sigma, 2.0);
  FillSentinel(tiny_output, 3.0);
  const auto output_before = Snapshot(tiny_output);
  const auto huge_stamp = Stamp(huge_points,
                                dimension,
                                regularization::EvaluationSiteKind::Volume,
                                9001u,
                                901u);
  const auto huge_batch =
    regularization::MakeFrozenScalarIGRVolumeCorrectionBatch(tiny_state,
                                                             tiny_sigma,
                                                             huge_stamp);
  regularization::RegularizationOperationOutputs overflow_outputs;
  overflow_outputs.volume_correction = regularization::BindEvaluatedPointOutput(
    tiny_output,
    huge_points,
    equations * dimension,
    dimension,
    regularization::EvaluationSiteKind::Volume,
    huge_stamp.point_set_id);
  launch_count = adapter.LaunchCount();
  result = adapter.Launch(
    regularization::RegularizationOperationKind::VolumeCorrection,
    OperationContextFor(
      huge_stamp,
      regularization::RegularizationOperationKind::VolumeCorrection),
    huge_batch,
    overflow_outputs);
  test.Code(result.validation.code,
            regularization::ValidationCode::InvalidComponentCount,
            "input extent multiplication overflow");
  test.Check(!result.launched && adapter.LaunchCount() == launch_count,
             "overflow does not launch");
  CheckSnapshot(test,
                tiny_output,
                output_before,
                "overflow preserves writable sentinel");
}

} // namespace

int main()
{
  TestContext test;
  ParityMetrics metrics;

  const regularization::FrozenScalarIGRAdapterSelection selection =
    EnabledSelection();
  auto adapter = regularization::ConfigureFrozenScalarIGRBrokenDGCPUAdapter(
    selection,
    ValidParameters());
  test.Check(adapter.IsConfigured() && adapter.HostSelectionCount() == 1,
             "adapter performs exactly one host static selection");

  const int initial_launch_count = adapter.LaunchCount();
  for (int dimension = 1; dimension <= 3; ++dimension)
  {
    TestPhysicalParityForDimension(test, adapter, dimension, metrics);
  }
  test.Check(adapter.LaunchCount() - initial_launch_count == 15,
             "exactly five static operations launch in each of 1D/2D/3D");

  TestTwoStateAcceptedConsumer(test, metrics);
  TestRestartTruthfulness(test);
  TestExplicitMFEMVectorMemoryPath(test, adapter);
  TestUnsupportedSelectionsAndCapabilities(test);
  TestInvalidParameters(test);
  TestOperationLocalRequirements(test, adapter);
  TestRequiredFieldFailures(test, adapter);
  TestIdentityAndProvenanceFailures(test, adapter);
  TestProducerAuthorities(test, adapter);
  TestOutputFailures(test, adapter);
  TestAliasesAndOverflow(test, adapter);
  TestSigmaAwareCharacteristicDefault(test);

  const mfem::real_t tolerance = static_cast<mfem::real_t>(32.0) *
                                 std::numeric_limits<mfem::real_t>::epsilon();
  test.Check(metrics.coefficient <= tolerance,
             "coefficient physical parity tolerance");
  test.Check(metrics.source <= tolerance, "source physical parity tolerance");
  test.Check(metrics.tensor <= tolerance, "tensor physical parity tolerance");
  test.Check(metrics.normal <= tolerance, "normal physical parity tolerance");
  test.Check(metrics.characteristic <= tolerance,
             "characteristic physical parity tolerance");
  test.Check(metrics.two_state <= tolerance,
             "accepted two-state consumer parity tolerance");

  std::cout << "PARITY dimensions=1,2,3"
            << " max_coefficient_error=" << metrics.coefficient
            << " max_source_error=" << metrics.source
            << " max_tensor_error=" << metrics.tensor
            << " max_normal_error=" << metrics.normal
            << " max_characteristic_error=" << metrics.characteristic
            << " max_two_state_error=" << metrics.two_state << '\n';
  std::cout << "STATIC_BINDINGS operations=5 host_selections="
            << adapter.HostSelectionCount()
            << " successful_launches=" << adapter.LaunchCount() << '\n';
  std::cout << "NEGATIVE_GATES selection=14 capabilities=2 parameters=5"
            << " required_fields=15 identity_provenance=16"
            << " producer_authorities=15 outputs=6 aliases=2 overflow=1\n";

  if (test.Failures() != 0)
  {
    std::cerr << test.Failures()
              << " frozen scalar IGR BrokenDG CPU adapter check(s) failed\n";
    return 1;
  }
  std::cout << "Frozen scalar IGR BrokenDG CPU adapter checks passed\n";
  return 0;
}
