#include "numerics/hdg/hdg_contract.hpp"

#include <array>
#include <cstddef>
#include <limits>
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

struct ModelTag : ModelTagMarker
{};
struct StabilizationTag : StabilizationTagMarker
{};
struct TimeTreatmentTag : TimeTreatmentTagMarker
{};
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
static_assert(IsKernelSpec<TestSpec>::value);
static_assert(TestSpec::trace_component_count == 8);
static_assert(std::is_same<TestSpec::backend_policy, TestBackend>::value);
static_assert(std::is_same<TestSpec::device_policy, ModelDevicePolicy>::value);
static_assert(std::is_same<KernelScalar<TestSpec>, double>::value);

HdgTopologyStorage MakeTopology()
{
  HdgTopologyStorage topology;
  topology.geometries.push_back(
      {40, ReferenceGeometry::segment, 1, 2, 3, 5});
  topology.elements.push_back({100, 0, 3, 1, {4, 2, 2, 8}});
  topology.faces.push_back({200,
                            0,
                            {0, invalid_local_index},
                            {0, invalid_local_index},
                            {{7, -1}, {0, 1}},
                            0,
                            16,
                            0,
                            Ownership::owned,
                            3,
                            1});
  for (LocalIndex i = 0; i != 16; ++i)
  {
    topology.trace_dofs.push_back(
        {500 + i, i, 0, i % TestSpec::trace_component_count, i % 2,
         i % 2 == 0 ? Ownership::owned : Ownership::ghost});
  }
  topology.element_face_offsets = {0, 1};
  topology.element_faces.push_back({0, 0, {7, -1}});
  return topology;
}

bool CheckTopology()
{
  const HdgTopologyStorage topology = MakeTopology();
  if (!topology.HasConsistentRanges() ||
      !HasConsistentRangesForSpec<TestSpec>(topology))
  {
    return false;
  }

  const HdgTopologyView view = topology.View();
  return view.element_count == 1 && view.face_count == 1 &&
         view.trace_dof_count == 16 && view.geometry_count == 1 &&
         view.Element(0).blocks.InteriorUnknowns() == 8 &&
         view.Element(0).blocks.TraceUnknowns() == 16 &&
         view.Element(0).blocks.Extent(LocalBlock::interior_trace_jacobian)
                 .rows == 8 &&
         view.Element(0).blocks.Extent(LocalBlock::interior_trace_jacobian)
                 .columns == 16 &&
         view.ElementFaceBegin(0) == 0 && view.ElementFaceEnd(0) == 1 &&
         view.element_faces[0].orientation.permutation == 7 &&
         view.element_faces[0].orientation.sign == -1 &&
         view.TraceDof(1).ownership == Ownership::ghost &&
         view.Face(0).owner_rank == 0 &&
         view.Geometry(0).mapping_order == 3;
}

bool CheckTopologyRejections()
{
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.elements[0].blocks.trace_components = 1;
    if (!topology.HasConsistentRanges() ||
        HasConsistentRangesForSpec<TestSpec>(topology))
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.elements[0].blocks.interior_dofs = -1;
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.elements[0].blocks.interior_dofs =
        std::numeric_limits<LocalIndex>::max();
    topology.elements[0].blocks.interior_components = 2;
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.faces[0].local_faces[1] = 0;
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.element_faces[0].orientation.sign = 0;
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.trace_dofs[0].face = invalid_local_index;
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.geometries[0].reference_dimension = 2;
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.faces[0].owner_rank = invalid_rank;
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.faces[0].ownership = static_cast<Ownership>(255);
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.trace_dofs[0].ownership = static_cast<Ownership>(255);
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.trace_dofs[0].component = TestSpec::trace_component_count;
    if (!topology.HasConsistentRanges() ||
        HasConsistentRangesForSpec<TestSpec>(topology))
    {
      return false;
    }
  }
  {
    HdgTopologyStorage topology = MakeTopology();
    topology.faces[0].trace_count = 15;
    if (topology.HasConsistentRanges())
    {
      return false;
    }
  }
  return true;
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

  WorkspaceLayout layout;
  if (!TryMakeWorkspaceLayout({2, 1, 2, 8}, 3, 7, layout) ||
      TryMakeWorkspaceLayout({-1, 1, 2, 8}, 3, 7, layout) ||
      TryMakeWorkspaceLayout({std::numeric_limits<LocalIndex>::max(),
                              std::numeric_limits<LocalIndex>::max(),
                              1,
                              1},
                             1,
                             0,
                             layout) ||
      TryMakeWorkspaceLayout({1, 1, 1, 8},
                             std::numeric_limits<std::size_t>::max(),
                             0,
                             layout) ||
      layout.local_jacobian.offset != layout.local_residual.size ||
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
  void EvaluateResidual(const LocalResidualRequest<TestSpec>&,
                        LocalResidualResult<TestSpec>&)
  {
    ++record->calls[0];
  }
  void EvaluateJacobian(const LocalJacobianRequest<TestSpec>&,
                        LocalJacobianResult<TestSpec>&)
  {
    ++record->calls[1];
  }
};

struct CondensationOperator
{
  StageRecord* record = nullptr;
  void Condense(const CondenseRequest<TestSpec>&, CondenseResult<TestSpec>&)
  {
    ++record->calls[2];
  }
  void ApplySchur(const ApplySchurRequest<TestSpec>&,
                  ApplySchurResult<TestSpec>&)
  {
    ++record->calls[3];
  }
};

struct TraceAssembler
{
  StageRecord* record = nullptr;
  void Assemble(const TraceAssemblyRequest<TestSpec>&,
                TraceAssemblyResult<TestSpec>&)
  {
    ++record->calls[4];
  }
};

struct TraceSolver
{
  StageRecord* record = nullptr;
  void Solve(const TraceSolveRequest<TestSpec>&, TraceSolveResult<TestSpec>&)
  {
    ++record->calls[5];
  }
};

struct RecoveryOperator
{
  StageRecord* record = nullptr;
  void Recover(const RecoveryRequest<TestSpec>&, RecoveryResult<TestSpec>&)
  {
    ++record->calls[6];
  }
};

bool CheckStaticStageBoundaries()
{
  StageRecord record;
  Pipeline<TestSpec,
           LocalOperator,
           CondensationOperator,
           TraceAssembler,
           TraceSolver,
           RecoveryOperator>
      pipeline({&record}, {&record}, {&record}, {&record}, {&record});

  LocalResidualRequest<TestSpec> residual_request;
  LocalResidualResult<TestSpec> residual_result;
  LocalJacobianRequest<TestSpec> jacobian_request;
  LocalJacobianResult<TestSpec> jacobian_result;
  CondenseRequest<TestSpec> condense_request;
  CondenseResult<TestSpec> condense_result;
  ApplySchurRequest<TestSpec> schur_request;
  ApplySchurResult<TestSpec> schur_result;
  TraceAssemblyRequest<TestSpec> assembly_request;
  TraceAssemblyResult<TestSpec> assembly_result;
  TraceSolveRequest<TestSpec> solve_request;
  TraceSolveResult<TestSpec> solve_result;
  RecoveryRequest<TestSpec> recovery_request;
  RecoveryResult<TestSpec> recovery_result;
  pipeline.EvaluateLocalResidual(residual_request, residual_result);
  pipeline.EvaluateLocalJacobian(jacobian_request, jacobian_result);
  pipeline.CondenseLocalSystem(condense_request, condense_result);
  pipeline.ApplySchur(schur_request, schur_result);
  pipeline.AssembleTraceSystem(assembly_request, assembly_result);
  pipeline.SolveGlobalTraceSystem(solve_request, solve_result);
  pipeline.RecoverInterior(recovery_request, recovery_result);
  for (const int calls : record.calls)
  {
    if (calls != 1)
    {
      return false;
    }
  }
  return true;
}

} // namespace

int main()
{
  return CheckTopology() && CheckTopologyRejections() &&
                 CheckStridesAndWorkspace() && CheckStaticStageBoundaries()
             ? 0
             : 1;
}
