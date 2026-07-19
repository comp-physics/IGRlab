#include "physics/regularization/regularization.hpp"

#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <type_traits>

namespace regularization = dg_igr::regularization;

namespace
{

constexpr regularization::AuxiliaryFieldIdentity kScalarIdentity{
  0x1001u, 0x2001u, 0x3001u};
constexpr regularization::AuxiliaryFieldIdentity kScalarIdentityCopy{
  0x1001u, 0x2001u, 0x3001u};
constexpr regularization::AuxiliaryFieldIdentity kTraceIdentity{
  0x1002u, 0x2002u, 0x3002u};

static_assert(kScalarIdentity == kScalarIdentityCopy,
              "field identity equality must be compile-time stable");
static_assert(kScalarIdentity != kTraceIdentity,
              "distinct restart/output identities must remain distinct");
static_assert(
  std::is_trivially_copyable<regularization::RegularizationDevicePolicy>::value,
  "fixture policy must remain trivially copyable");
static_assert(
  regularization::HasRegularizationTraits<
    regularization::ZeroFixtureModelTag>::value,
  "zero fixture trait must be available");
static_assert(
  regularization::HasRegularizationTraits<
    regularization::ConstantFixtureModelTag>::value,
  "constant fixture trait must be available");
static_assert(
  !regularization::HasRegularizationTraits<
    regularization::UnregisteredFixtureModelTag>::value,
  "unknown model trait must be unavailable");
static_assert(regularization::SupportedCpuFixtureCombination::value,
              "supported fixture combination must remain available");
static_assert(!regularization::UnsupportedSolveCombination::value,
              "unsupported solve combination must remain unavailable");
static_assert(!regularization::UnsupportedTimeCombination::value,
              "unsupported time combination must remain unavailable");
static_assert(!regularization::UnsupportedEntropyCombination::value,
              "unsupported entropy combination must remain unavailable");

class TestContext
{
public:
  void Check(bool condition, const char* message)
  {
    if (!condition)
    {
      ++failures_;
      std::cerr << "[FAIL] " << message << '\n';
    }
  }

  void Near(double actual,
            double expected,
            double tolerance,
            const char* message)
  {
    if (std::abs(actual - expected) > tolerance)
    {
      ++failures_;
      std::cerr << "[FAIL] " << message << ": actual=" << actual
                << " expected=" << expected << '\n';
    }
  }

  int Failures() const { return failures_; }

private:
  int failures_ = 0;
};

regularization::AuxiliaryFieldCapabilities VolumeCapabilities()
{
  using regularization::AuxiliaryCapability;
  return {regularization::CapabilityBit(AuxiliaryCapability::DeviceAccessible) |
          regularization::CapabilityBit(AuxiliaryCapability::Restartable) |
          regularization::CapabilityBit(AuxiliaryCapability::OutputVisible) |
          regularization::CapabilityBit(AuxiliaryCapability::StageLocal)};
}

regularization::AuxiliaryFieldCapabilities TraceCapabilities()
{
  using regularization::AuxiliaryCapability;
  return {VolumeCapabilities().bits |
          regularization::CapabilityBit(AuxiliaryCapability::TraceCoupled)};
}

void TestTypedDescriptors(TestContext& test)
{
  constexpr int points = 3;
  constexpr int dimension = 2;
  const regularization::AuxiliaryFieldProvenance provenance{2, 19u, 7u};

  mfem::Vector scalar_storage(points);
  mfem::Vector vector_storage(dimension * points);
  mfem::Vector tensor_storage(dimension * dimension * points);
  mfem::Vector trace_storage(points);

  regularization::ScalarAuxiliaryFieldDescriptor scalar(
    scalar_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::StateOwned,
    provenance,
    kScalarIdentity,
    VolumeCapabilities());
  regularization::VectorAuxiliaryFieldDescriptor vector(
    vector_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    provenance,
    {0x1010u, 0x2010u, 0x3010u},
    VolumeCapabilities());
  regularization::TensorAuxiliaryFieldDescriptor tensor(
    tensor_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    provenance,
    {0x1020u, 0x2020u, 0x3020u},
    VolumeCapabilities());
  regularization::TraceAuxiliaryFieldDescriptor trace(
    trace_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::BorrowedReadOnly,
    provenance,
    kTraceIdentity,
    TraceCapabilities());

  scalar_storage = 2.0;
  const mfem::real_t* scalar_read = scalar.Read(false);
  test.Near(scalar_read[1], 2.0, 0.0, "scalar descriptor MFEM Memory read");

  auto vector_view = mfem::Reshape(vector.Write(false), dimension, points);
  for (int point = 0; point < points; ++point)
  {
    for (int d = 0; d < dimension; ++d)
    {
      vector_view(d, point) = static_cast<double>(10 * point + d);
    }
  }

  tensor_storage = 0.0;
  auto tensor_view =
    mfem::Reshape(tensor.ReadWrite(false), dimension, dimension, points);
  tensor_view(1, 0, 2) = 9.0;
  trace_storage = 4.0;
  const mfem::real_t* trace_read = trace.Read(false);

  test.Check(scalar.Components() == 1 && scalar.HasExpectedSize(),
             "typed scalar descriptor shape");
  test.Check(vector.Components() == dimension && vector.HasExpectedSize(),
             "typed vector descriptor shape");
  test.Check(tensor.Components() == dimension * dimension &&
               tensor.HasExpectedSize(),
             "typed tensor descriptor shape");
  test.Check(trace.Topology() == regularization::AuxiliaryFieldTopology::Trace &&
               trace.HasExpectedSize(),
             "typed trace descriptor topology");
  test.Check(scalar.Ownership() ==
               regularization::AuxiliaryFieldOwnership::StateOwned &&
               trace.Ownership() ==
                 regularization::AuxiliaryFieldOwnership::BorrowedReadOnly,
             "descriptor ownership metadata");
  test.Check(regularization::SameStageState(scalar.Provenance(),
                                             trace.Provenance()),
             "descriptor RK-stage/state-generation provenance");
  test.Check(scalar.Identity() == kScalarIdentity &&
               trace.Identity() == kTraceIdentity,
             "descriptor restart/output identity");
  test.Check(scalar.Capabilities().Has(
               regularization::AuxiliaryCapability::Restartable) &&
               trace.Capabilities().Has(
                 regularization::AuxiliaryCapability::TraceCoupled),
             "descriptor capability metadata");
  test.Near(vector_storage.HostRead()[5], 21.0, 0.0,
            "vector descriptor MFEM Reshape write");
  test.Near(tensor_storage.HostRead()[1 + dimension * dimension * 2],
            9.0,
            0.0,
            "tensor descriptor MFEM ReadWrite/Reshape");
  test.Near(trace_read[0], 4.0, 0.0, "trace descriptor MFEM Memory read");
}

struct FixtureBuffers
{
  static constexpr int points = 3;
  static constexpr int dimension = 2;

  mfem::Vector state = mfem::Vector(points);
  mfem::Vector auxiliary_storage = mfem::Vector(points);
  mfem::Vector trace_storage = mfem::Vector(points);
  mfem::Vector gradient = mfem::Vector(dimension * points);
  mfem::Vector normals = mfem::Vector(dimension * points);
  mfem::Vector coefficient = mfem::Vector(points);
  mfem::Vector source = mfem::Vector(points);
  mfem::Vector volume = mfem::Vector(dimension * points);
  mfem::Vector normal = mfem::Vector(points);
  mfem::Vector wall = mfem::Vector(points);
  mfem::Vector bounds = mfem::Vector(2 * points);
  mfem::Vector diagnostics = mfem::Vector(3 * points);

  regularization::AuxiliaryFieldProvenance provenance{2, 19u, 7u};
  regularization::ScalarAuxiliaryFieldDescriptor auxiliary{
    auxiliary_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::StateOwned,
    provenance,
    kScalarIdentity,
    VolumeCapabilities()};
  regularization::TraceAuxiliaryFieldDescriptor trace{
    trace_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    provenance,
    kTraceIdentity,
    TraceCapabilities()};

  FixtureBuffers()
  {
    double* state_values = state.HostWrite();
    double* trace_values = trace_storage.HostWrite();
    for (int point = 0; point < points; ++point)
    {
      state_values[point] = static_cast<double>(point + 1);
      trace_values[point] = state_values[point] + 0.5;
    }
    auxiliary_storage = 2.0;
    gradient = 0.0;

    auto normal_view =
      mfem::Reshape(normals.HostWrite(), dimension, points);
    normal_view(0, 0) = 1.0;
    normal_view(1, 0) = 0.0;
    normal_view(0, 1) = 0.0;
    normal_view(1, 1) = 1.0;
    normal_view(0, 2) = 0.6;
    normal_view(1, 2) = 0.8;
    ResetOutputs(0.0);
  }

  void ResetOutputs(double value)
  {
    coefficient = value;
    source = value;
    volume = value;
    normal = value;
    wall = value;
    bounds = value;
    diagnostics = value;
  }

  regularization::RegularizationLaunchBuffers LaunchBuffers()
  {
    return {points,
            dimension,
            &state,
            &auxiliary,
            &trace,
            &gradient,
            &normals,
            &coefficient,
            &source,
            &volume,
            &normal,
            &wall,
            &bounds,
            &diagnostics};
  }
};

regularization::RegularizationSelection ConstantCpuSelection()
{
  regularization::RegularizationSelection selection;
  selection.axes.continuum_model =
    regularization::ContinuumModelKind::ConstantFixture;
  selection.axes.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::ElementL2;
  selection.axes.auxiliary_solution =
    regularization::AuxiliarySolutionKind::PointwiseEvaluation;
  selection.axes.execution_backend =
    regularization::ExecutionBackendKind::MFEMCpu;
  selection.compatibility.time =
    regularization::TimeTreatmentKind::FrozenRKStage;
  selection.compatibility.entropy =
    regularization::EntropyTreatmentKind::Neutral;
  return selection;
}

regularization::RegularizationValidationContext ValidationContext()
{
  return {false, 2, 19u};
}

void TestConstantAndNormalConsistency(TestContext& test)
{
  FixtureBuffers fixture;
  const regularization::RegularizationDevicePolicy policy{
    2.5, -3.0, 0.5, 2.0, -4.0};
  const regularization::RegularizationLaunchResult result =
    regularization::ExecuteRegularization(ConstantCpuSelection(),
                                          ValidationContext(),
                                          policy,
                                          fixture.LaunchBuffers());

  test.Check(result.validation.Ok() && result.launched,
             "constant fixture host dispatch launches");
  test.Check(result.host_selection_count == 1,
             "host selection occurs exactly once before static execution");

  const double* coefficient = fixture.coefficient.HostRead();
  const double* source = fixture.source.HostRead();
  const auto volume = mfem::Reshape(fixture.volume.HostRead(),
                                     FixtureBuffers::dimension,
                                     FixtureBuffers::points);
  const auto normals = mfem::Reshape(fixture.normals.HostRead(),
                                      FixtureBuffers::dimension,
                                      FixtureBuffers::points);
  const double* normal = fixture.normal.HostRead();
  const double* wall = fixture.wall.HostRead();
  const auto bounds =
    mfem::Reshape(fixture.bounds.HostRead(), 2, FixtureBuffers::points);
  const auto diagnostics =
    mfem::Reshape(fixture.diagnostics.HostRead(), 3, FixtureBuffers::points);

  for (int point = 0; point < FixtureBuffers::points; ++point)
  {
    test.Near(coefficient[point], 2.5, 0.0,
              "constant coefficient behavior");
    test.Near(source[point], -3.0, 0.0, "constant source behavior");
    double tensor_dot_normal = 0.0;
    for (int d = 0; d < FixtureBuffers::dimension; ++d)
    {
      test.Near(volume(d, point), 1.0, 0.0,
                "constant volume correction behavior");
      tensor_dot_normal += volume(d, point) * normals(d, point);
    }
    test.Near(normal[point],
              tensor_dot_normal,
              1.0e-14,
              "normal correction equals tensor contraction");
    test.Near(wall[point], 1.0, 0.0, "fixture wall-law behavior");
    test.Near(bounds(0, point), -4.0, 0.0,
              "lower characteristic bound");
    test.Near(bounds(1, point), 4.0, 0.0,
              "upper characteristic bound");
    test.Near(diagnostics(0, point), 3.0, 0.0,
              "source diagnostic behavior");
    test.Near(diagnostics(1, point), 2.0, 0.0,
              "correction diagnostic behavior");
    test.Near(diagnostics(2, point), 1.0, 0.0,
              "wall diagnostic behavior");
  }
}

void TestZeroBehavior(TestContext& test)
{
  FixtureBuffers fixture;
  fixture.ResetOutputs(0.0);
  regularization::RegularizationSelection selection = ConstantCpuSelection();
  selection.axes.continuum_model =
    regularization::ContinuumModelKind::ZeroFixture;
  const regularization::RegularizationDevicePolicy nonzero_policy{
    9.0, 8.0, 7.0, 6.0, 5.0};
  const regularization::RegularizationLaunchResult result =
    regularization::ExecuteRegularization(selection,
                                          ValidationContext(),
                                          nonzero_policy,
                                          fixture.LaunchBuffers());
  test.Check(result.validation.Ok() && result.launched,
             "zero fixture launches through static dispatch");

  const mfem::Vector* outputs[] = {&fixture.coefficient,
                                   &fixture.source,
                                   &fixture.volume,
                                   &fixture.normal,
                                   &fixture.wall,
                                   &fixture.bounds,
                                   &fixture.diagnostics};
  for (const mfem::Vector* output : outputs)
  {
    const double* values = output->HostRead();
    for (int i = 0; i < output->Size(); ++i)
    {
      test.Near(values[i], 0.0, 0.0, "zero fixture output behavior");
    }
  }
}

void TestUnsupportedCombinations(TestContext& test)
{
  FixtureBuffers fixture;
  fixture.ResetOutputs(17.0);
  const regularization::RegularizationDevicePolicy policy{};

  regularization::RegularizationSelection unsupported = ConstantCpuSelection();
  unsupported.compatibility.entropy =
    regularization::EntropyTreatmentKind::EntropyStable;
  regularization::RegularizationLaunchResult result =
    regularization::ExecuteRegularization(unsupported,
                                          ValidationContext(),
                                          policy,
                                          fixture.LaunchBuffers());
  test.Check(!result.launched && result.host_selection_count == 0 &&
               result.validation.code ==
                 regularization::ValidationCode::UnsupportedCombination,
             "entropy-incompatible selection fails before dispatch");
  test.Near(fixture.source.HostRead()[0],
            17.0,
            0.0,
            "validation failure leaves output untouched");

  unsupported = ConstantCpuSelection();
  unsupported.compatibility.time =
    regularization::TimeTreatmentKind::LaggedStep;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnsupportedCombination &&
               !result.launched,
             "time-incompatible selection fails before launch");

  unsupported = ConstantCpuSelection();
  unsupported.axes.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::TraceL2;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnsupportedCombination &&
               !result.launched,
             "discretization/solve mismatch fails before launch");

  unsupported = ConstantCpuSelection();
  unsupported.axes.continuum_model =
    regularization::ContinuumModelKind::ZeroFixture;
  unsupported.axes.auxiliary_solution =
    regularization::AuxiliarySolutionKind::GlobalLinearSolve;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnsupportedCombination &&
               !result.launched,
             "model/solve mismatch fails before launch");

  unsupported = ConstantCpuSelection();
  unsupported.axes.execution_backend =
    regularization::ExecutionBackendKind::MFEMDevice;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::BackendUnavailable &&
               !result.launched,
             "unavailable backend fails before launch");

  unsupported = ConstantCpuSelection();
  unsupported.axes.continuum_model =
    static_cast<regularization::ContinuumModelKind>(255);
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnknownAxisValue &&
               !result.launched,
             "unknown model value fails before launch");
}

void TestStaleProvenance(TestContext& test)
{
  FixtureBuffers fixture;
  regularization::RegularizationValidationContext stale = ValidationContext();
  stale.expected_state_generation = 20u;
  const regularization::RegularizationLaunchResult result =
    regularization::ExecuteRegularization(ConstantCpuSelection(),
                                          stale,
                                          {},
                                          fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::StaleProvenance &&
               !result.launched,
             "stale state-generation provenance fails before launch");
}

} // namespace

int main()
{
  mfem::Device device("cpu");
  TestContext test;
  TestTypedDescriptors(test);
  TestConstantAndNormalConsistency(test);
  TestZeroBehavior(test);
  TestUnsupportedCombinations(test);
  TestStaleProvenance(test);

  if (test.Failures() == 0)
  {
    std::cout << "[PASS] static MFEM regularization contract fixtures\n";
  }
  return test.Failures() == 0 ? 0 : 1;
}
