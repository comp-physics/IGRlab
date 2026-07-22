#include "numerics/hdg/hdg_contract.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

namespace
{

using namespace dg::hdg;

struct StatePotential;
struct FluxPotential;
struct AuxiliaryPotential;

using TestTrace =
    TraceTuple<TracePotential<StatePotential, ScalarTrace>,
               TracePotential<FluxPotential, VectorTrace<3>>,
               TracePotential<AuxiliaryPotential, TensorTrace<2, 2>>>;

struct ModelTag;
struct StabilizationTag;
struct TimeTreatmentTag;
struct MfemExecutionTag;
struct UmpireMemoryTag;
struct MpiCommunicationTag;

struct ModelDevicePolicy
{
  double coefficient = 0.0;
  int variant = 0;
};

using TestBackend =
    BackendPolicy<MfemExecutionTag, UmpireMemoryTag, MpiCommunicationTag>;
using TestPrecision = PrecisionPolicy<double, double, double, double, double>;
using TestSpec = KernelSpec<ModelTag,
                            TestTrace,
                            StabilizationTag,
                            TimeTreatmentTag,
                            TestBackend,
                            TestPrecision,
                            ModelDevicePolicy>;

static_assert(TestTrace::potential_count == 3);
static_assert(TestTrace::component_count == 8);
static_assert(TestTrace::ComponentOffset<0>() == 0);
static_assert(TestTrace::ComponentOffset<1>() == 1);
static_assert(TestTrace::ComponentOffset<2>() == 4);
static_assert(
    std::is_same<TestTrace::potential_type<1>,
                 TracePotential<FluxPotential, VectorTrace<3>>>::value);
static_assert(std::is_same<TestSpec::backend_policy, TestBackend>::value);
static_assert(std::is_same<TestSpec::device_policy, ModelDevicePolicy>::value);
static_assert(!std::is_polymorphic<TestSpec>::value);

bool CheckTopology()
{
  HdgTopologyStorage topology;
  topology.geometries.push_back(
      {40, ReferenceGeometry::segment, 1, 2, 3, 5});

  LocalBlockShape shape{4, 2, 2, 1};
  topology.elements.push_back({100, 0, 3, 2, shape});
  topology.faces.push_back({200,
                            0,
                            {0, invalid_local_index},
                            {1, invalid_local_index},
                            {{7, -1}, {}},
                            0,
                            2,
                            0,
                            Ownership::owned,
                            3,
                            1});
  topology.trace_dofs.push_back({500, 0, 0, 0, 0, Ownership::owned});
  topology.trace_dofs.push_back({501, 1, 0, 0, 1, Ownership::ghost});
  topology.element_face_offsets = {0, 1};
  topology.element_faces.push_back({0, 1, {7, -1}});

  if (!topology.HasConsistentRanges())
  {
    return false;
  }

  const HdgTopologyView view = topology.View();
  return view.element_count == 1 && view.face_count == 1 &&
         view.trace_dof_count == 2 && view.geometry_count == 1 &&
         view.Element(0).blocks.InteriorUnknowns() == 8 &&
         view.Element(0).blocks.TraceUnknowns() == 2 &&
         view.Element(0).blocks.Extent(LocalBlock::interior_trace_jacobian)
                 .rows == 8 &&
         view.Element(0).blocks.Extent(LocalBlock::interior_trace_jacobian)
                 .columns == 2 &&
         view.ElementFaceBegin(0) == 0 && view.ElementFaceEnd(0) == 1 &&
         view.element_faces[0].orientation.permutation == 7 &&
         view.element_faces[0].orientation.sign == -1 &&
         view.TraceDof(1).ownership == Ownership::ghost &&
         view.Face(0).owner_rank == 0 &&
         view.Geometry(0).mapping_order == 3;
}

bool CheckStridesAndWorkspace()
{
  std::array<double, 32> storage{};
  StridedBlockView<double> block{
      storage.data(), 2, 2, 3, 16, 4, 1};
  block(0, 1, 2) = 12.5;
  block(1, 0, 1) = 21.5;
  const auto constant = AsConst(block);
  if (constant(0, 1, 2) != 12.5 || constant(1, 0, 1) != 21.5)
  {
    return false;
  }

  const LocalBlockShape shape{2, 1, 2, 2};
  const WorkspaceLayout layout = MakeWorkspaceLayout(shape, 3, 7);
  if (layout.local_jacobian.offset != layout.local_residual.size ||
      layout.factorization.offset !=
          layout.local_jacobian.offset + layout.local_jacobian.size ||
      layout.condensed_system.offset !=
          layout.factorization.offset + layout.factorization.size ||
      layout.temporary.offset !=
          layout.condensed_system.offset + layout.condensed_system.size ||
      layout.ScalarCount() != layout.scalars_per_batch * 3)
  {
    return false;
  }

  HostWorkspaceStorage<double> workspace;
  workspace.Resize(layout);
  const auto view = workspace.View();
  view.RegionData(2, layout.temporary)[6] = 9.0;
  return view.BatchData(2)[layout.temporary.offset + 6] == 9.0;
}

struct StageRecord
{
  std::array<int, 7> calls{};
};

struct LocalOperator
{
  StageRecord* record = nullptr;
  int EvaluateResidual(int value)
  {
    ++record->calls[0];
    return value + 1;
  }
  int EvaluateJacobian(int value)
  {
    ++record->calls[1];
    return value + 2;
  }
};

struct CondensationOperator
{
  StageRecord* record = nullptr;
  int Condense(int value)
  {
    ++record->calls[2];
    return value + 3;
  }
  int ApplySchur(int value)
  {
    ++record->calls[3];
    return value + 4;
  }
};

struct TraceAssembler
{
  StageRecord* record = nullptr;
  int Assemble(int value)
  {
    ++record->calls[4];
    return value + 5;
  }
};

struct TraceSolver
{
  StageRecord* record = nullptr;
  int Solve(int value)
  {
    ++record->calls[5];
    return value + 6;
  }
};

struct RecoveryOperator
{
  StageRecord* record = nullptr;
  int Recover(int value)
  {
    ++record->calls[6];
    return value + 7;
  }
};

bool CheckStaticStageBoundaries()
{
  static_assert(!std::is_polymorphic<LocalOperator>::value);
  static_assert(!std::is_polymorphic<CondensationOperator>::value);
  static_assert(!std::is_polymorphic<TraceAssembler>::value);
  static_assert(!std::is_polymorphic<TraceSolver>::value);
  static_assert(!std::is_polymorphic<RecoveryOperator>::value);

  StageRecord record;
  Pipeline<TestSpec,
           LocalOperator,
           CondensationOperator,
           TraceAssembler,
           TraceSolver,
           RecoveryOperator>
      pipeline({&record}, {&record}, {&record}, {&record}, {&record});

  const bool values = pipeline.EvaluateLocalResidual(10) == 11 &&
                      pipeline.EvaluateLocalJacobian(10) == 12 &&
                      pipeline.CondenseLocalSystem(10) == 13 &&
                      pipeline.ApplySchur(10) == 14 &&
                      pipeline.AssembleTraceSystem(10) == 15 &&
                      pipeline.SolveGlobalTraceSystem(10) == 16 &&
                      pipeline.RecoverInterior(10) == 17;
  for (const int calls : record.calls)
  {
    if (calls != 1)
    {
      return false;
    }
  }
  return values;
}

} // namespace

int main()
{
  return CheckTopology() && CheckStridesAndWorkspace() &&
                 CheckStaticStageBoundaries()
             ? 0
             : 1;
}
