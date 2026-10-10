// Test-only active CPU HDG wall map A/B through the existing public session.
// A repeated private-core run supplies existing solver evidence and must match
// the public-session exported state. No production callback or API is changed.
#include "../hdg/runtime/generic_euler_convergence_fixture.hpp"
#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "physics/bc/detail/euler_boundary_state.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>

namespace
{
namespace fixture = dg::hdg::test::generic_euler_convergence;
namespace generic = dg::hdg::generic;
namespace cpu = dg::hdg::runtime::cpu;
using Clock = std::chrono::steady_clock;
using Four = std::array<double, 4>;

struct Options
{
  int elements = 2, order = 2;
  double epsilon = .1, dt = 1.e-4, alpha = .0004, beta = .8;
  bool self_test = false;
  std::string output;
};

struct WallContext
{
  bool projected = false;
  std::uint64_t calls = 0;
};

bool WallMap(const void* opaque, const generic::BoundaryPointView& point,
             const double* interior, double* target) noexcept
{
  auto* context = static_cast<WallContext*>(const_cast<void*>(opaque));
  if (!context || !interior || !target || !point.outward_normal ||
      point.space_dimension != 2) return false;
  ++context->calls;
  if (!mfem::detail::EvaluateEulerBoundaryState(
          mfem::detail::EulerBoundaryStateKind::slip_wall, 2, 4, 1.4, 0.,
          static_cast<const double*>(nullptr), point.outward_normal,
          interior, target)) return false;
  if (context->projected)
  {
    // Average the conservative state with its mirror. Keep rho and E exactly:
    // removing normal momentum increases trace thermal pressure; no energy
    // correction may be added without changing the stabilization energy jump.
    target[1] = .5 * (target[1] + interior[1]);
    target[2] = .5 * (target[2] + interior[2]);
  }
  return true;
}

struct Metrics
{
  Four integrals{};
  double minimum_density = std::numeric_limits<double>::infinity();
  double minimum_pressure = std::numeric_limits<double>::infinity();
};

Metrics Measure(const mfem::ParGridFunction& state)
{
  Metrics result;
  auto* mesh = state.ParFESpace()->GetParMesh();
  for (int e = 0; e < mesh->GetNE(); ++e)
  {
    auto* transformation = mesh->GetElementTransformation(e);
    const auto& rule = mfem::IntRules.Get(
        state.ParFESpace()->GetFE(e)->GetGeomType(), 12);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const auto& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      mfem::Vector value(4);
      state.GetVectorValue(e, point, value);
      const double weight = point.weight * transformation->Weight();
      for (int c = 0; c < 4; ++c) result.integrals[c] += weight * value[c];
      const double pressure = .4 *
          (value[3] - .5 * (value[1]*value[1] + value[2]*value[2])/value[0]);
      result.minimum_density = std::min(result.minimum_density, value[0]);
      result.minimum_pressure = std::min(result.minimum_pressure, pressure);
    }
  }
  return result;
}

void WriteFour(std::ostream& out, const Four& value)
{
  out << '[';
  for (int c = 0; c < 4; ++c) out << (c ? "," : "") << value[c];
  out << ']';
}

struct Record
{
  bool projected = false, igr = false, initialized = false, accepted = false;
  bool core_initialized = false, core_accepted = false, exported = false;
  int admission_status = -1, session_initialization_status = -1;
  double dt = 0., reached = 0., session_seconds = 0., witness_seconds = 0.;
  double state_difference = 0.;
  Metrics before, after;
  std::uint64_t callback_calls = 0, core_callback_calls = 0;
  cpu::CpuAttemptEvidence evidence;
};

Record Run(const Options& options, bool projected, bool igr, double dt)
{
  Record record; record.projected = projected; record.igr = igr; record.dt = dt;
  const auto session_start = Clock::now();
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      options.elements, options.elements, mfem::Element::QUADRILATERAL,
      true, 1., 1.);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::L2_FECollection collection(options.order, 2,
      mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(&mesh, &collection, 4,
                                   mfem::Ordering::byNODES);
  mfem::ParGridFunction initial(&space), source(&space), output(&space), witness(&space);
  const auto initial_function = [&](const mfem::Vector& x, mfem::Vector& state)
  {
    state.SetSize(4); state[0] = 1.;
    state[1] = options.epsilon*(x[0]-.5);
    state[2] = options.epsilon*(x[1]-.5);
    state[3] = 2.5;
  };
  mfem::VectorFunctionCoefficient coefficient(4, initial_function);
  initial.ProjectCoefficient(coefficient); source = 0.;
  record.before = Measure(initial);
  fixture::BoundaryAudit unused;
  fixture::CaseSelection selection; selection.order = options.order;
  selection.regularization = igr ? generic::RegularizationKind::igr
                                 : generic::RegularizationKind::none;
  generic::EulerHdgRequest request = fixture::MakeRequest(
      mesh, initial, source, unused, fixture::ExecutionSelection{}, selection, {12,12});
  WallContext context{projected, 0};
  // The custom CPU callback uses the contract's canonical absent POD table.
  // A zero-count table with dimension/gamma set is an empty periodic table,
  // which is correctly rejected on this nonperiodic mesh.
  request.boundary = {&context, WallMap, {nullptr, 0, 0, 0.}};
  request.time_treatment = generic::EulerHdgTimeTreatment::backward_euler;
  request.numerical_policy.stabilization = igr
      ? generic::RegularizationSelection{generic::RegularizationKind::igr,
                                         options.alpha, options.beta}
      : generic::RegularizationSelection{generic::RegularizationKind::none, 0., 0.};
  cpu::GenericEulerHdgCpuSession session;
  record.admission_status = static_cast<int>(generic::ValidateEulerHdgRequest(request));
  const auto initialized = session.Initialize(request);
  record.session_initialization_status = static_cast<int>(initialized.status);
  record.initialized = initialized.status ==
      cpu::GenericEulerCpuSessionStatus::initialized;
  record.accepted = record.initialized && session.Advance(dt).status ==
      cpu::GenericEulerCpuSessionStatus::advanced;
  record.reached = session.coordinates().time;
  record.exported = record.initialized && session.ExportCommittedInterior(output);
  if (record.exported) record.after = Measure(output);
  else record.after = record.before;
  record.callback_calls = context.calls;
  record.session_seconds = std::chrono::duration<double>(Clock::now()-session_start).count();

  // The public session deliberately exposes no solver residual stream. The
  // established private core already has bounded test evidence. Repeat this
  // tiny solve and require identical committed Euler state before attaching
  // its evidence to the session measurement.
  const auto witness_start = Clock::now(); context.calls = 0;
  cpu::detail::CpuNumericsCore core;
  dg::hdg::runtime::StateCoordinates coordinates{};
  record.core_initialized = core.InitializeBorrowedEuler(request);
  record.core_accepted = record.core_initialized && core.AdvanceBorrowedEuler(dt, coordinates);
  record.evidence = core.evidence().last_attempt;
  if (record.core_initialized && core.ExportCommittedInterior(witness) && record.exported)
  {
    const double* a = output.HostRead(); const double* b = witness.HostRead();
    for (int i = 0; i < output.Size(); ++i)
      record.state_difference = std::max(record.state_difference, std::abs(a[i]-b[i]));
  }
  else record.state_difference = -1.;
  record.core_callback_calls = context.calls;
  record.witness_seconds = std::chrono::duration<double>(Clock::now()-witness_start).count();
  return record;
}

void WriteRecord(std::ostream& out, const Record& r)
{
  Four change{}; for (int c=0;c<4;++c) change[c]=r.after.integrals[c]-r.before.integrals[c];
  out << "{\"trace_map\":\"" << (r.projected ? "zero-normal-momentum-rho-E-preserved" : "shared-reflected-ghost")
      << "\",\"igr\":" << r.igr << ",\"dt\":" << r.dt
      << ",\"initialized\":" << r.initialized << ",\"accepted\":" << r.accepted
      << ",\"request_admission_status\":" << r.admission_status
      << ",\"session_initialization_status\":" << r.session_initialization_status
      << ",\"reached_time\":" << r.reached << ",\"exported\":" << r.exported
      << ",\"before_integrals\":"; WriteFour(out,r.before.integrals);
  out << ",\"after_integrals\":"; WriteFour(out,r.after.integrals);
  out << ",\"integral_change\":"; WriteFour(out,change);
  out << ",\"mass_change_per_dt\":" << change[0]/r.dt
      << ",\"energy_change_per_dt\":" << change[3]/r.dt
      << ",\"sampled_minimum_density\":" << r.after.minimum_density
      << ",\"sampled_minimum_pressure\":" << r.after.minimum_pressure
      << ",\"boundary_calls\":" << r.callback_calls
      << ",\"core_boundary_calls\":" << r.core_callback_calls
      << ",\"session_seconds_including_fixture_setup\":" << r.session_seconds
      << ",\"separate_core_witness_seconds\":" << r.witness_seconds
      << ",\"core_initialized\":" << r.core_initialized
      << ",\"core_accepted\":" << r.core_accepted
      << ",\"session_core_maximum_coefficient_difference\":" << r.state_difference
      << ",\"core_attempt_status\":" << static_cast<int>(r.evidence.status)
      << ",\"core_failure_phase\":" << static_cast<int>(r.evidence.failure_phase)
      << ",\"initial_nonlinear_residual\":" << r.evidence.initial_nonlinear_residual
      << ",\"final_nonlinear_residual\":" << r.evidence.final_nonlinear_residual
      << ",\"maximum_true_linear_relative_residual\":" << r.evidence.maximum_true_relative_residual
      << ",\"nonlinear_updates\":" << r.evidence.nonlinear_updates
      << ",\"linear_iterations\":" << r.evidence.linear_iterations
      << ",\"nonlinear_gate\":" << r.evidence.nonlinear_gate
      << ",\"admissibility_gate\":" << r.evidence.admissibility_gate << '}';
}

Options Parse(int argc,char** argv)
{
  Options options;
  for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];
    if(flag=="--self-test") {options.self_test=true;continue;}
    if(++i==argc) throw std::runtime_error("missing option value");
    const std::string value=argv[i];
    if(flag=="--output") options.output=value;
    else if(flag=="--elements") options.elements=std::stoi(value);
    else if(flag=="--order") options.order=std::stoi(value);
    else if(flag=="--epsilon") options.epsilon=std::stod(value);
    else if(flag=="--dt") options.dt=std::stod(value);
    else if(flag=="--alpha") options.alpha=std::stod(value);
    else if(flag=="--beta") options.beta=std::stod(value);
    else throw std::runtime_error("unknown wall-step option: "+flag);
  }
  if(options.elements<1||options.elements>4||options.order<1||options.order>3||
     !std::isfinite(options.epsilon)||options.epsilon<=0.||options.epsilon>.4||
     !std::isfinite(options.dt)||options.dt<=0.||options.dt>.001||
     !std::isfinite(options.alpha)||options.alpha<=0.||
     !std::isfinite(options.beta)||options.beta<=0.)
    throw std::runtime_error("invalid bounded wall-step options");
  if(options.self_test){options.elements=1;options.order=1;options.dt=1.e-4;}
  return options;
}
}

#ifndef IGR_HDG_WALL_STEP_LIBRARY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    int ranks=1;MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(ranks!=1)throw std::runtime_error("bounded wall-step fixture requires one MPI rank");
    const auto options=Parse(argc,argv);
    std::vector<Record> records;
    for(bool igr:{false,true})for(double dt:{options.dt,.5*options.dt})
      for(bool projected:{false,true})records.push_back(Run(options,projected,igr,dt));
    bool passed=true;
    for(const auto& r:records)
    {
      const double mass_change=r.after.integrals[0]-r.before.integrals[0];
      const double energy_change=r.after.integrals[3]-r.before.integrals[3];
      passed=passed&&r.accepted&&r.core_accepted&&r.exported&&r.state_difference>=0.&&
          r.state_difference<1.e-10&&r.callback_calls>0&&r.after.minimum_pressure>0.;
      if(options.self_test)
        passed=passed&&(r.projected?(std::abs(mass_change)<2.e-8&&std::abs(energy_change)<2.e-8)
                                    :mass_change>1.e-8);
    }
    std::ofstream file;
    if(!options.output.empty()){file.open(options.output);if(!file)throw std::runtime_error("cannot open output");}
    std::ostream& out=options.output.empty()?std::cout:file;
    out<<std::setprecision(17)<<std::boolalpha;
    out<<"{\"scope\":\"Actual CPU borrowed HDG backward-Euler sessions and separate identical private-core evidence witnesses; flat closed box, no CUDA or production change. Initial outward normal momentum is an intentional finite-stage wall defect, not a compatible smooth wall IVP.\",\"passed\":"<<passed
       <<",\"elements_per_direction\":"<<options.elements<<",\"order\":"<<options.order
       <<",\"epsilon\":"<<options.epsilon<<",\"alpha\":"<<options.alpha<<",\"beta\":"<<options.beta
       <<",\"expected_initial_reflection_mass_change_rate\":"<<2.*options.epsilon
       <<",\"normal_projection_preserves_total_energy\":true,\"records\":[";
    for(std::size_t i=0;i<records.size();++i){if(i)out<<',';WriteRecord(out,records[i]);}
    out<<"]}\n";return passed?0:2;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif
