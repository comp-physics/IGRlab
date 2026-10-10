// Bounded smooth wall evolution through the existing public CPU HDG session.
// The included wall-step fixture supplies only test helpers, not a new engine.
#define IGR_HDG_WALL_STEP_LIBRARY
#include "igr_hdg_wall_step_probe.cpp"

namespace
{
struct EvolutionOptions
{
  int elements = 4;
  double amplitude = .4, height = .125, dt = .0005, final_time = .05;
  double alpha = .0004, beta = .8;
  bool projected = true, igr = true, self_test = false;
  std::string output;
};

struct WallSnapshot
{
  double time = 0.;
  Metrics volume;
  // Length-averaged interior traces at the two x walls. These are physical
  // field QoIs, not the private hybrid sigma or assembled numerical traction.
  std::array<double, 2> pressure{}, sigma{}, normal_velocity{};
  double transverse_variation = 0.;
};

bool ExportEvolution(cpu::GenericEulerHdgCpuSession& session, bool igr,
                     mfem::ParGridFunction& state,
                     mfem::ParGridFunction& sigma)
{
  sigma = 0.;
  return session.ExportCommittedFields({state.HostWrite(),
      igr ? sigma.HostWrite() : nullptr, static_cast<std::size_t>(sigma.Size())});
}

WallSnapshot SnapshotEvolution(const mfem::ParGridFunction& state,
                               const mfem::ParGridFunction& sigma, double time)
{
  WallSnapshot result; result.time = time; result.volume = Measure(state);
  auto* mesh = state.ParFESpace()->GetParMesh();
  const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 12);
  for (int side = 0; side < 2; ++side)
  {
    int e = -1;
    for (int candidate = 0; candidate < mesh->GetNE(); ++candidate)
    {
      mfem::IntegrationPoint ip; ip.Set2(side, .5);
      mfem::Vector physical(2);
      mesh->GetElementTransformation(candidate)->Transform(ip, physical);
      if (std::abs(physical[0]-side) < 1.e-12) e = candidate;
    }
    if (e < 0) throw std::runtime_error("cannot locate Cartesian x wall");
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const auto& point = rule.IntPoint(q);
      mfem::IntegrationPoint ip; ip.Set2(side, point.x);
      mfem::Vector value(4); state.GetVectorValue(e, ip, value);
      result.pressure[side] += point.weight * .4 *
          (value[3] - .5*(value[1]*value[1]+value[2]*value[2])/value[0]);
      result.sigma[side] += point.weight*sigma.GetValue(e, ip);
      result.normal_velocity[side] += point.weight*(side == 0 ? -1. : 1.)*
          value[1]/value[0];
    }
  }
  for (int e = 0; e < mesh->GetNE(); ++e)
    for (double x : {0., .2113248654051871, .5, .7886751345948129, 1.})
    {
      mfem::IntegrationPoint a, b; a.Set2(x, .2113248654051871);
      b.Set2(x, .7886751345948129);
      mfem::Vector va(4), vb(4); state.GetVectorValue(e, a, va);
      state.GetVectorValue(e, b, vb);
      for (int c = 0; c < 4; ++c)
        result.transverse_variation = std::max(result.transverse_variation,
                                               std::abs(va[c]-vb[c]));
      result.transverse_variation = std::max(result.transverse_variation,
          std::abs(sigma.GetValue(e, a)-sigma.GetValue(e, b)));
    }
  return result;
}

void WritePair(std::ostream& out, const std::array<double, 2>& a)
{ out << '[' << a[0] << ',' << a[1] << ']'; }

void WriteSnapshot(std::ostream& out, const WallSnapshot& s)
{
  out << "{\"time\":" << s.time << ",\"integrals\":";
  WriteFour(out, s.volume.integrals);
  out << ",\"sampled_minimum_density\":" << s.volume.minimum_density
      << ",\"sampled_minimum_pressure\":" << s.volume.minimum_pressure
      << ",\"wall_interior_pressure\":"; WritePair(out, s.pressure);
  out << ",\"wall_interior_sigma\":"; WritePair(out, s.sigma);
  out << ",\"wall_interior_normal_velocity\":"; WritePair(out, s.normal_velocity);
  out << ",\"transverse_variation\":" << s.transverse_variation << '}';
}

void WriteProfile(std::ostream& out, const mfem::ParGridFunction& state,
                  const mfem::ParGridFunction& sigma)
{
  const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 7);
  auto* mesh = state.ParFESpace()->GetParMesh();
  out << '[';
  for (int e = 0; e < mesh->GetNE(); ++e)
  {
    if (e) out << ',';
    auto* transformation = mesh->GetElementTransformation(e);
    mfem::IntegrationPoint ip; mfem::Vector physical(2);
    ip.Set2(0., .5); transformation->Transform(ip, physical);
    out << "{\"left\":" << physical[0];
    ip.Set2(1., .5); transformation->Transform(ip, physical);
    out << ",\"right\":" << physical[0] << ",\"samples\":[";
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      if (q) out << ',';
      ip.Set2(rule.IntPoint(q).x, .5);
      transformation->Transform(ip, physical);
      mfem::Vector value(4); state.GetVectorValue(e, ip, value);
      out << '[' << physical[0];
      for (int c = 0; c < 4; ++c) out << ',' << value[c];
      out << ',' << sigma.GetValue(e, ip) << ']';
    }
    out << "]}";
  }
  out << ']';
}

EvolutionOptions ParseEvolution(int argc, char** argv)
{
  EvolutionOptions o;
  for (int i = 1; i < argc; ++i)
  {
    const std::string flag = argv[i];
    if (flag == "--self-test") { o.self_test = true; continue; }
    if (++i == argc) throw std::runtime_error("missing evolution option value");
    const std::string value = argv[i];
    if (flag == "--output") o.output = value;
    else if (flag == "--elements") o.elements = std::stoi(value);
    else if (flag == "--amplitude") o.amplitude = std::stod(value);
    else if (flag == "--dt") o.dt = std::stod(value);
    else if (flag == "--final-time") o.final_time = std::stod(value);
    else if (flag == "--alpha") o.alpha = std::stod(value);
    else if (flag == "--beta") o.beta = std::stod(value);
    else if (flag == "--model")
    { if (value != "igr" && value != "euler") throw std::runtime_error("bad model");
      o.igr = value == "igr"; }
    else if (flag == "--wall-target")
    { if (value != "projected" && value != "reflected") throw std::runtime_error("bad target");
      o.projected = value == "projected"; }
    else throw std::runtime_error("unknown evolution option: " + flag);
  }
  if (o.elements < 1 || o.elements > 32 || !std::isfinite(o.amplitude) ||
      o.amplitude < 0. || o.amplitude > .9 || !std::isfinite(o.dt) || o.dt <= 0. ||
      o.dt > .01 || !std::isfinite(o.final_time) || o.final_time <= 0. ||
      o.final_time > .2 || o.final_time/o.dt > 4000. ||
      !std::isfinite(o.alpha) || o.alpha <= 0. || !std::isfinite(o.beta) || o.beta <= 0.)
    throw std::runtime_error("invalid bounded wall evolution options");
  if (o.self_test) { o.elements = 2; o.dt = .0001; o.final_time = .0002; }
  return o;
}

bool RunEvolution(const EvolutionOptions& o, std::ostream& out)
{
  const auto start = Clock::now();
  constexpr int order = 3;
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(o.elements, 1,
      mfem::Element::QUADRILATERAL, true, 1., o.height);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::L2_FECollection collection(order, 2, mfem::BasisType::GaussLegendre,
                                   mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(&mesh, &collection, 4, mfem::Ordering::byNODES);
  mfem::ParFiniteElementSpace scalar_space(&mesh, &collection, 1);
  mfem::ParGridFunction initial(&space), source(&space), state(&space), sigma(&scalar_space);
  mfem::VectorFunctionCoefficient coefficient(4,
      [&](const mfem::Vector& x, mfem::Vector& u)
      {
        u.SetSize(4); u[0] = 1.; u[1] = 6.*std::sqrt(3.)*o.amplitude*
            x[0]*(1.-x[0])*(2.*x[0]-1.); u[2] = 0.; u[3] = 2.5;
      });
  initial.ProjectCoefficient(coefficient); source = 0.; state = initial; sigma = 0.;
  const auto before = SnapshotEvolution(state, sigma, 0.);
  fixture::BoundaryAudit unused;
  fixture::CaseSelection selection; selection.order = order;
  selection.regularization = o.igr ? generic::RegularizationKind::igr
                                  : generic::RegularizationKind::none;
  auto request = fixture::MakeRequest(mesh, initial, source, unused,
      fixture::ExecutionSelection{}, selection, {12, 12});
  WallContext context{o.projected, 0};
  request.boundary = {&context, WallMap, {nullptr, 0, 0, 0.}};
  request.time_treatment = generic::EulerHdgTimeTreatment::backward_euler;
  request.numerical_policy.stabilization = o.igr
      ? generic::RegularizationSelection{generic::RegularizationKind::igr, o.alpha, o.beta}
      : generic::RegularizationSelection{generic::RegularizationKind::none, 0., 0.};
  cpu::GenericEulerHdgCpuSession session;
  const int admission = static_cast<int>(generic::ValidateEulerHdgRequest(request));
  const auto initialized = session.Initialize(request);
  bool passed = initialized.status == cpu::GenericEulerCpuSessionStatus::initialized;
  const double setup_seconds = std::chrono::duration<double>(Clock::now()-start).count();
  int rejected = 0, advances = 0; double diagnostic_seconds = 0., advance_seconds = 0.;
  std::vector<WallSnapshot> history;
  const int stride = std::max(1, static_cast<int>(std::ceil(o.final_time/o.dt))/20);
  while (passed && session.coordinates().time < o.final_time-1.e-14)
  {
    const double dt = std::min(o.dt, o.final_time-session.coordinates().time);
    const auto tick = Clock::now();
    const auto advanced = session.Advance(dt);
    advance_seconds += std::chrono::duration<double>(Clock::now()-tick).count();
    if (advanced.status != cpu::GenericEulerCpuSessionStatus::advanced)
    { ++rejected; passed = false; break; }
    ++advances;
    if (advances == 1 || advances % stride == 0 ||
        session.coordinates().time >= o.final_time-1.e-14)
    {
      const auto diagnostic_start = Clock::now();
      passed = ExportEvolution(session, o.igr, state, sigma);
      if (passed) history.push_back(SnapshotEvolution(state, sigma, session.coordinates().time));
      diagnostic_seconds += std::chrono::duration<double>(Clock::now()-diagnostic_start).count();
    }
  }
  const auto final_start = Clock::now();
  bool exported = initialized.status == cpu::GenericEulerCpuSessionStatus::initialized &&
      ExportEvolution(session, o.igr, state, sigma);
  const auto after = SnapshotEvolution(state, sigma, session.coordinates().time);
  passed = passed && exported && after.time >= o.final_time-1.e-14 &&
      after.volume.minimum_density > 0. && after.volume.minimum_pressure > 0.;
  if (o.self_test)
  {
    passed = passed && std::abs(before.normal_velocity[0]) < 1.e-13 &&
        std::abs(before.normal_velocity[1]) < 1.e-13 && after.transverse_variation < 1.e-9;
    if (o.projected) passed = passed &&
        std::abs(after.volume.integrals[0]-before.volume.integrals[0]) < 1.e-9 &&
        std::abs(after.volume.integrals[3]-before.volume.integrals[3]) < 1.e-9;
  }
  diagnostic_seconds += std::chrono::duration<double>(Clock::now()-final_start).count();
  out << std::setprecision(17) << std::boolalpha
      << "{\"scope\":\"Actual CPU public HDG p3 backward-Euler evolution; initially zero wall normal momentum, no source; interior wall p/sigma are QoIs, not exported hybrid traction. No production change or GPU ranking. Initial sigma is not exported until certified after an accepted step.\""
      << ",\"passed\":" << passed << ",\"model\":\"" << (o.igr ? "igr" : "euler")
      << "\",\"wall_target\":\"" << (o.projected ? "projected" : "reflected")
      << "\",\"elements\":" << o.elements << ",\"order\":" << order
      << ",\"height\":" << o.height << ",\"amplitude\":" << o.amplitude
      << ",\"alpha\":" << o.alpha << ",\"beta\":" << o.beta
      << ",\"dt\":" << o.dt << ",\"final_time\":" << o.final_time
      << ",\"request_admission_status\":" << admission
      << ",\"initialization_status\":" << static_cast<int>(initialized.status)
      << ",\"accepted_steps\":" << advances << ",\"rejected_attempts\":" << rejected
      << ",\"exported\":" << exported << ",\"boundary_calls\":" << context.calls
      << ",\"fixture_and_session_setup_seconds\":" << setup_seconds
      << ",\"advance_seconds\":" << advance_seconds
      << ",\"export_and_diagnostic_seconds\":" << diagnostic_seconds
      << ",\"seconds_before_json_serialization\":"
      << std::chrono::duration<double>(Clock::now()-start).count()
      << ",\"before\":"; WriteSnapshot(out, before);
  out << ",\"after\":"; WriteSnapshot(out, after);
  out << ",\"history\": [";
  for (std::size_t i=0; i<history.size(); ++i) { if(i) out << ','; WriteSnapshot(out, history[i]); }
  out << "],\"centerline_polynomial_profile\":"; WriteProfile(out, state, sigma);
  out << "}\n";
  return passed;
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv); mfem::Device device("cpu");
  try
  {
    int ranks = 1; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks != 1) throw std::runtime_error("wall evolution requires one MPI rank");
    const auto o = ParseEvolution(argc, argv);
    std::ofstream file;
    if (!o.output.empty()) { file.open(o.output); if (!file) throw std::runtime_error("cannot open output"); }
    return RunEvolution(o, o.output.empty() ? std::cout : file) ? 0 : 2;
  }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
