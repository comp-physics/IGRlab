#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/io/vis_helpers.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "utils/diagnostics/detail/euler_admissibility_device.hpp"
#include "utils/diagnostics/non_finite.hpp"

#include <cmath>
#include <cfenv>
#include <cstring>
#include <limits>
#include <vector>

#include "test_utils.hpp"

namespace dg_igr_tests
{

namespace
{

dg_sim::RunConfig MakeRunConfig(int order)
{
  dg_sim::RunConfig cfg;
  cfg.case_config.gamma = 1.4;
  cfg.opts.order = order;
  cfg.opts.glvis = false;
  cfg.opts.vtk_save = false;
  cfg.opts.ser_ref_levels = 0;
  cfg.opts.par_ref_levels = 0;
  cfg.dg_backend = dg_sim::DGBackend::DG;
  cfg.mesh_path.clear();
  return cfg;
}

void CheckFiniteScanShortcut(TestContext& ctx)
{
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const bool cuda = mfem::igrlab::detail::AdmissibilityOrdinaryCuda();
  mfem::Array<int> status;
  mfem::Vector empty;
  ctx.Check(mfem::igrlab::detail::CanSkipDetailedFiniteScan(
                empty, MPI_COMM_WORLD, status) == cuda,
            "Empty device finite scan takes the consistent empty-state path.");

  using Scalar = mfem::real_t;
  mfem::Vector state(8);
  const auto fill_finite = [&]()
  {
    auto* values = state.HostWrite();
    values[0] = std::numeric_limits<Scalar>::max();
    values[1] = std::numeric_limits<Scalar>::lowest();
    values[2] = std::numeric_limits<Scalar>::denorm_min();
    values[3] = -std::numeric_limits<Scalar>::denorm_min();
    values[4] = Scalar(0.0);
    values[5] = Scalar(-0.0);
    values[6] = std::numeric_limits<Scalar>::min();
    values[7] = -std::numeric_limits<Scalar>::min();
  };
  const auto upload_current = [&]()
  {
    state.UseDevice(true);
    (void)state.ReadWrite(true);
  };
  fill_finite();
  ctx.Check(!mfem::igrlab::detail::CanSkipDetailedFiniteScan(
                state, MPI_COMM_WORLD, status),
            "A host-authoritative state keeps the full finite scanner.");
  const auto full = mfem::igrlab::ScanNonFiniteDetailed(
      state, MPI_COMM_WORLD, 2, mfem::Ordering::byNODES);
  ctx.Check(full.summary.global_non_finite == 0 &&
                full.summary.global_min == std::numeric_limits<Scalar>::lowest() &&
                full.summary.global_max == std::numeric_limits<Scalar>::max() &&
                full.first_bad_rank == -1 &&
                full.global_component_non_finite == std::vector<int>({0, 0}),
            "Full finite diagnostics retain exact extrema and component metadata.");
  if (!cuda) { return; }

  upload_current();
  ctx.Check(mfem::igrlab::detail::CanSkipDetailedFiniteScan(
                state, MPI_COMM_WORLD, status) &&
                !state.GetMemory().HostIsValid(),
            "Device finite extremes, signed zeros and subnormals pass without state host staging.");

  // One rank declining is sufficient even when every other rank is device-ready.
  if (rank == ranks - 1) { (void)state.HostReadWrite(); }
  ctx.Check(!mfem::igrlab::detail::CanSkipDetailedFiniteScan(
                state, MPI_COMM_WORLD, status),
            "A single rank declining the device route sends all ranks to full diagnostics.");

  const Scalar bad_values[] = {
    std::numeric_limits<Scalar>::quiet_NaN(),
    std::numeric_limits<Scalar>::infinity(),
    -std::numeric_limits<Scalar>::infinity()};
  for (const Scalar bad : bad_values)
  {
    fill_finite();
    if (rank == ranks - 1) { state.HostReadWrite()[2] = bad; }
    upload_current();
    ctx.Check(!mfem::igrlab::detail::CanSkipDetailedFiniteScan(
                  state, MPI_COMM_WORLD, status) &&
                  !state.GetMemory().HostIsValid(),
              "One-rank NaN or signed infinity forces collective fallback without state host staging.");
    const auto detailed = mfem::igrlab::ScanNonFiniteDetailed(
        state, MPI_COMM_WORLD, 2, mfem::Ordering::byNODES);
    ctx.Check(detailed.summary.global_non_finite == 1 &&
                  detailed.summary.global_min == std::numeric_limits<Scalar>::lowest() &&
                  detailed.summary.global_max == std::numeric_limits<Scalar>::max() &&
                  detailed.first_bad_rank == ranks - 1 &&
                  detailed.global_component_non_finite == std::vector<int>({1, 0}) &&
                  detailed.first_local_entry.found == (rank == ranks - 1),
              "Fallback preserves exact global counts, extrema, first rank and component metadata.");
    if (rank == ranks - 1)
    {
      ctx.Check(detailed.first_local_entry.local_index == 2 &&
                    detailed.first_local_entry.scalar_dof == 1 &&
                    detailed.first_local_entry.component == 0 &&
                    (std::isnan(bad) ? std::isnan(detailed.first_local_entry.value) :
                      detailed.first_local_entry.value == bad),
                "Fallback preserves the first nonfinite value and its local mapping.");
    }
  }
  fill_finite();
  upload_current();
  ctx.Check(mfem::igrlab::detail::CanSkipDetailedFiniteScan(
                state, MPI_COMM_WORLD, status) &&
                !state.GetMemory().HostIsValid(),
            "A fresh finite device state clears the previous nonfinite result.");
}

int FindInterfaceSideElement(const dg_sim::ExampleContext& ctx, bool left)
{
  mfem::IntegrationPoint ip_center;
  ip_center.Set2(0.5, 0.5);
  int best_e = -1;
  double best_x = left ? -std::numeric_limits<double>::infinity()
                       : std::numeric_limits<double>::infinity();
  for (int e = 0; e < ctx.Mesh().GetNE(); ++e)
  {
    mfem::ElementTransformation* Tr =
      ctx.spaces.fes.GetElementTransformation(e);
    mfem::Vector x;
    Tr->Transform(ip_center, x);
    if (left)
    {
      if (x(0) < 0.5 && x(0) > best_x)
      {
        best_x = x(0);
        best_e = e;
      }
    }
    else
    {
      if (x(0) > 0.5 && x(0) < best_x)
      {
        best_x = x(0);
        best_e = e;
      }
    }
  }
  return best_e;
}

double ReduceOwnedValue(double local_value, bool owned)
{
  return GlobalSum(owned ? local_value : 0.0);
}

double ReduceOwnedCount(bool owned)
{
  return GlobalSum(owned ? 1.0 : 0.0);
}

} // namespace

int RunDiagnosticsTests(TestContext& ctx)
{
  CheckFiniteScanShortcut(ctx);
  constexpr double gamma = 1.4;
  constexpr double rho_floor = 1.0e-12;

  // Startup may defer unobserved projections, but the first output refresh
  // must use the fully initialized state, including subsequent state changes.
  for (const auto backend : {dg_sim::DGBackend::DG, dg_sim::DGBackend::DGSEM})
  {
    mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
    dg_sim::RunConfig run_cfg = MakeRunConfig(/*order=*/5);
    run_cfg.dg_backend = backend;
    dg_sim::ExampleContext example_ctx(run_cfg, std::move(pmesh), "L2",
                                       /*need_sigma=*/false,
                                       /*need_h1=*/false);
    dg_sim::EulerStateBundle bundle(example_ctx, gamma, rho_floor,
                                     /*project_initial_state=*/false);
    mfem::VectorFunctionCoefficient ic(
      4, [](const mfem::Vector& x, mfem::Vector& y)
      {
        y.SetSize(4);
        const double rho = 1.0 + 0.2 * x(0);
        const double u = 0.25 + 0.1 * x(1);
        const double v = -0.05 + 0.1 * x(0);
        y(0) = rho;
        y(1) = rho * u;
        y(2) = rho * v;
        y(3) = (1.0 + 0.1 * x(1)) / 0.4 +
               0.5 * rho * (u * u + v * v);
      });
    bundle.sol.ProjectCoefficient(ic);
    mfem::PressureCoefficient pressure_coeff(bundle.sol, 2, gamma);
    mfem::VelocityFromConsCoefficient velocity_coeff(2, &bundle.sol, rho_floor);
    mfem::ParGridFunction pressure_reference(&example_ctx.spaces.fes);
    mfem::ParGridFunction velocity_reference(&example_ctx.spaces.dfes);
    const auto same_values = [](const mfem::Vector& a, const mfem::Vector& b)
    {
      if (a.Size() != b.Size()) { return false; }
      const auto* av = a.HostRead();
      const auto* bv = b.HostRead();
      for (int i = 0; i < a.Size(); ++i)
      {
        if (!std::isfinite(av[i]) || av[i] != bv[i]) { return false; }
      }
      return true;
    };
    pressure_reference.ProjectCoefficient(pressure_coeff);
    velocity_reference.ProjectCoefficient(velocity_coeff);
    dg_sim::EulerPressureView eager_pressure(example_ctx.spaces.fes, bundle.sol,
                                             2, gamma);
    dg_sim::EulerVelocityView eager_velocity(example_ctx.spaces.dfes, bundle.sol,
                                             2, rho_floor);
    ctx.Check(same_values(eager_pressure.pressure, pressure_reference) &&
                same_values(eager_velocity.velocity, velocity_reference),
              "Default Euler view constructors retain eager exact projections.");

    // Model an initialization/admission change before the first consumer.
    const int ndofs = example_ctx.spaces.fes.GetNDofs();
    auto* state = bundle.sol.HostReadWrite();
    for (int i = 0; i < ndofs; ++i) { state[3 * ndofs + i] += 0.375; }
    bundle.pressure_view.pressure = std::numeric_limits<double>::quiet_NaN();
    bundle.velocity_view.velocity = std::numeric_limits<double>::quiet_NaN();
    dg_sim::UpdateEulerOutputFields(bundle.field_list);
    pressure_reference.ProjectCoefficient(pressure_coeff);
    velocity_reference.ProjectCoefficient(velocity_coeff);
    ctx.Check(same_values(bundle.pressure_view.pressure, pressure_reference) &&
                same_values(bundle.velocity_view.velocity, velocity_reference),
              "Deferred Euler views exactly refresh the final initialized state.");
    bundle.ProjectIC(ic);
    pressure_reference.ProjectCoefficient(pressure_coeff);
    ctx.Check(same_values(bundle.pressure_view.pressure, pressure_reference),
              "ProjectIC retains its immediate pressure refresh.");

    const bool cuda = mfem::Device::Allows(mfem::Backend::CUDA) &&
                      sizeof(mfem::real_t) == sizeof(double);
    const auto check_current_projection = [&](bool expect_device,
                                               const char* message)
    {
      pressure_reference.ProjectCoefficient(pressure_coeff);
      velocity_reference.ProjectCoefficient(velocity_coeff);
      if (cuda)
      {
        bundle.sol.UseDevice(true);
        (void)bundle.sol.ReadWrite(); // make the current device copy authoritative
      }
      // Test projection residency separately: the complete output boundary
      // also synchronizes the owning state for its conserved host views.
      bundle.pressure_view.Update();
      bundle.velocity_view.Update();
      if (cuda)
      {
        ctx.Check(bundle.sol.GetMemory().HostIsValid() != expect_device,
                  "Derived projection uses the admitted device route or host fallback.");
      }
      const auto same_bits = [](const mfem::Vector& a, const mfem::Vector& b)
      {
        return a.Size() == b.Size() &&
          std::memcmp(a.HostRead(), b.HostRead(),
                      a.Size() * sizeof(mfem::real_t)) == 0;
      };
      ctx.Check(same_bits(bundle.pressure_view.pressure, pressure_reference) &&
                  same_bits(bundle.velocity_view.velocity, velocity_reference),
                message);
    };
    check_current_projection(true,
      "Collocated p5 GL/LGL CUDA projections exactly match generic coefficients.");
    check_current_projection(true,
      "Warm collocation plans still read the current device state exactly.");

    // This pair makes a contracted second square differ by one ULP from the
    // separate momentum sum. The small pressure exposes that distinction.
    {
      const double a = 0x1.3027f08b821d1p+0, b = 0x1.11d4388ee9e79p+0;
      volatile double aa = a * a, bb = b * b;
      auto* values = bundle.sol.HostReadWrite();
      for (int i = 0; i < ndofs; ++i)
      {
        values[i] = 0.5; values[ndofs + i] = a; values[2 * ndofs + i] = b;
        values[3 * ndofs + i] = aa + bb;
      }
      check_current_projection(true,
        "Collocated projection matches the host EOS momentum contraction exactly.");
    }
    if (cuda)
    {
      std::fenv_t saved_environment;
      ctx.Check(std::fegetenv(&saved_environment) == 0,
                "Derived projection can save the host floating-point environment.");
      ctx.Check(std::fesetround(FE_DOWNWARD) == 0,
                "Derived projection can select directed host rounding.");
      check_current_projection(false,
        "Directed host rounding retains the generic derived-field projection.");
      ctx.Check(std::fesetenv(&saved_environment) == 0,
                "Derived projection restores the host floating-point environment.");
#if defined(__aarch64__)
      unsigned long original_fpcr = 0;
      asm volatile("mrs %0, fpcr" : "=r"(original_fpcr));
      // Exercise both a flush control and a trap control, with finite values
      // that do not trigger the selected invalid-operation exception.
      for (unsigned long control : {1ul << 24, 1ul << 8})
      {
        const unsigned long changed_fpcr = original_fpcr | control;
        asm volatile("msr fpcr, %0" : : "r"(changed_fpcr) : "memory");
        unsigned long actual_fpcr = 0;
        asm volatile("mrs %0, fpcr" : "=r"(actual_fpcr));
        const bool applied = (actual_fpcr & control) != 0;
        ctx.Check(control != (1ul << 24) || applied,
                  "Arm flush-to-zero control is applied for the fallback test.");
        // Some Arm CPUs implement exception-trap enables as read-as-zero,
        // write-ignored bits. Route expectations follow the actual FPCR.
        check_current_projection(actual_fpcr == 0,
          applied ? "Applied Arm flush/trap controls retain the host projection."
                  : "Ignored Arm trap control retains the admitted projection.");
        asm volatile("msr fpcr, %0" : : "r"(original_fpcr) : "memory");
      }
#endif
    }

    auto* edge_state = bundle.sol.HostReadWrite();
    for (int i = 0; i < ndofs; ++i)
    {
      edge_state[i] = i % 2 ? 1.0e-18 : -0.0;
      edge_state[ndofs + i] = i % 3 ? 1.0e-18 : -0.0;
      edge_state[2 * ndofs + i] = i % 2 ? -1.0e-18 : -0.0;
      edge_state[3 * ndofs + i] = 1.0;
    }
    check_current_projection(true,
      "Pressure and velocity retain their distinct density floors and signed zeros.");
    bundle.sol.HostReadWrite()[ndofs] =
      std::numeric_limits<mfem::real_t>::quiet_NaN();
    check_current_projection(false,
      "Nonfinite collocated input preserves the full generic NaN projection.");

    example_ctx.Mesh().UniformRefinement();
    example_ctx.spaces.fes.Update();
    example_ctx.spaces.dfes.Update();
    example_ctx.spaces.vfes.Update();
    bundle.sol.Update();
    bundle.pressure_view.pressure.Update();
    bundle.velocity_view.velocity.Update();
    pressure_reference.Update();
    velocity_reference.Update();
    bundle.sol.ProjectCoefficient(ic);
    check_current_projection(true,
      "Mesh/space updates invalidate collocation plans before the next projection.");
  }

  // Mixed geometry and different input/output nodes retain coefficient
  // interpolation, rather than treating nearby or unrelated DOFs as collocated.
  for (const bool mixed_geometry : {false, true})
  {
    mfem::Mesh mesh = mixed_geometry
      ? mfem::Mesh(2, 6, 3, 0, 2)
      : mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL,
                                    true, 1.0, 1.0);
    if (mixed_geometry)
    {
      const mfem::real_t vertices[6][2] = {{0,0},{0.5,0},{1,0},{0,1},{0.5,1},{1,1}};
      for (const auto& vertex : vertices) { mesh.AddVertex(vertex); }
      mesh.AddQuad(0, 1, 4, 3);
      mesh.AddTriangle(1, 2, 5);
      mesh.AddTriangle(1, 5, 4);
      mesh.FinalizeTopology();
      mesh.Finalize();
    }
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::L2_FECollection input_fec(5, 2, mfem::BasisType::GaussLegendre);
    mfem::L2_FECollection output_fec(5, 2, mixed_geometry
      ? mfem::BasisType::GaussLegendre : mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace input_fes(&pmesh, &input_fec, 4,
                                          mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace scalar_fes(&pmesh, &output_fec);
    mfem::ParFiniteElementSpace vector_fes(&pmesh, &output_fec, 2,
                                           mfem::Ordering::byNODES);
    mfem::ParGridFunction state(&input_fes);
    FillConstantState(EulerConsFromPrim2D(1.0, 0.25, -0.05, 1.0, gamma), state);
    // Nonconstant values make a mistaken GL-to-LGL direct gather observable.
    auto* values = state.HostReadWrite();
    for (int i = 0; i < input_fes.GetNDofs(); ++i)
    {
      values[i] += 0.001 * (i % 7);
      values[3 * input_fes.GetNDofs() + i] += 0.002 * (i % 5);
    }
    mfem::PressureCoefficient pressure_coeff(state, 2, gamma);
    mfem::VelocityFromConsCoefficient velocity_coeff(2, &state, rho_floor);
    mfem::ParGridFunction pressure_reference(&scalar_fes), velocity_reference(&vector_fes);
    pressure_reference.ProjectCoefficient(pressure_coeff);
    velocity_reference.ProjectCoefficient(velocity_coeff);
    dg_sim::EulerPressureView pressure(scalar_fes, state, 2, gamma, false);
    dg_sim::EulerVelocityView velocity(vector_fes, state, 2, rho_floor, false);
    if (mfem::Device::Allows(mfem::Backend::CUDA))
    {
      state.UseDevice(true);
      (void)state.ReadWrite();
    }
    pressure.Update(); velocity.Update();
    // A globally mixed mesh can leave a rank with only quadrilaterals, where
    // the matching nodal spaces admit the device path. Require host residency
    // only on nonempty partitions that actually contain unsupported geometry
    // or different input/output nodes; compare values on every rank below.
    bool local_requires_host = !mixed_geometry;
    for (int e = 0; e < pmesh.GetNE(); ++e)
    {
      local_requires_host = local_requires_host ||
          pmesh.GetElementBaseGeometry(e) != mfem::Geometry::SQUARE;
    }
    ctx.Check(state.Size() == 0 || !local_requires_host ||
                  state.GetMemory().HostIsValid(),
              "Locally unsupported or noncollocated spaces retain the host coefficient path.");
    ctx.Check(std::memcmp(pressure.pressure.HostRead(), pressure_reference.HostRead(),
                         pressure_reference.Size() * sizeof(mfem::real_t)) == 0 &&
                std::memcmp(velocity.velocity.HostRead(), velocity_reference.HostRead(),
                            velocity_reference.Size() * sizeof(mfem::real_t)) == 0,
              "Unsupported layouts exactly match generic pressure/velocity projection.");
  }

  // H1 diagnostics scalar outputs must live on scalar spaces (vdim=1).
  {
    mfem::Mesh serial_mesh =
      mfem::Mesh::MakeCartesian2D(1,
                                  1,
                                  mfem::Element::QUADRILATERAL,
                                  /*gen_edges=*/true,
                                  /*sx=*/1.0,
                                  /*sy=*/1.0);
    auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    dg_sim::RunConfig run_cfg = MakeRunConfig(/*order=*/2);
    dg_sim::ExampleContext example_ctx(run_cfg,
                                       std::move(pmesh),
                                       /*sigma_space=*/"H1",
                                       /*need_sigma=*/false,
                                       /*need_h1=*/true);
    dg_sim::EulerStateBundle bundle(example_ctx, gamma, rho_floor);
    FillConstantState(EulerConsFromPrim2D(/*rho=*/1.0,
                                          /*u=*/0.25,
                                          /*v=*/-0.05,
                                          /*p=*/1.0,
                                          gamma),
                      bundle.sol);

    dg_sim::H1Diagnostics h1_diag(*example_ctx.spaces.h1_vfes,
                                  bundle.sol,
                                  example_ctx.dim,
                                  rho_floor);
    h1_diag.Update();

    ctx.Check(h1_diag.u_h1.ParFESpace()->GetVDim() == example_ctx.dim,
              "H1 diagnostics velocity field uses vector space.");
    ctx.Check(h1_diag.g_div.ParFESpace()->GetVDim() == 1,
              "H1 diagnostics div field uses scalar space.");
    ctx.Check(h1_diag.g_gfrob.ParFESpace()->GetVDim() == 1,
              "H1 diagnostics grad-frobenius field uses scalar space.");
    ctx.Check(h1_diag.g_umag.ParFESpace()->GetVDim() == 1,
              "H1 diagnostics speed field uses scalar space.");
  }

  // DG-to-H1 projection must use arithmetic averaging at shared H1 dofs.
  {
    mfem::Mesh serial_mesh =
      mfem::Mesh::MakeCartesian2D(2,
                                  1,
                                  mfem::Element::QUADRILATERAL,
                                  /*gen_edges=*/true,
                                  /*sx=*/1.0,
                                  /*sy=*/1.0);
    auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    dg_sim::RunConfig run_cfg = MakeRunConfig(/*order=*/0);
    dg_sim::ExampleContext example_ctx(run_cfg,
                                       std::move(pmesh),
                                       /*sigma_space=*/"H1",
                                       /*need_sigma=*/false,
                                       /*need_h1=*/true);
    dg_sim::EulerStateBundle bundle(example_ctx, gamma, rho_floor);

    mfem::VectorFunctionCoefficient ic(
      4,
      [](const mfem::Vector& x, mfem::Vector& y)
      {
        y.SetSize(4);
        const double rho = 1.0;
        const double ux = (x(0) < 0.5) ? 0.0 : 2.0;
        const double p = 1.0;
        y(0) = rho;
        y(1) = rho * ux;
        y(2) = 0.0;
        y(3) = p / 0.4 + 0.5 * rho * ux * ux;
      });
    bundle.sol.ProjectCoefficient(ic);

    dg_sim::H1Diagnostics h1_diag(*example_ctx.spaces.h1_vfes,
                                  bundle.sol,
                                  example_ctx.dim,
                                  rho_floor);
    h1_diag.Update();

    const int left_e = FindInterfaceSideElement(example_ctx, /*left=*/true);
    const int right_e = FindInterfaceSideElement(example_ctx, /*left=*/false);
    const bool owns_left = (left_e >= 0);
    const bool owns_right = (right_e >= 0);
    const double left_count = ReduceOwnedCount(owns_left);
    const double right_count = ReduceOwnedCount(owns_right);
    ctx.Check(left_count == 1.0 && right_count == 1.0,
              "Diagnostics test mesh exposes left/right interface elements.");

    mfem::Vector x_left(2);
    x_left(0) = 0.5 - 1.0e-10;
    x_left(1) = 0.5;
    mfem::Vector x_right(2);
    x_right(0) = 0.5 + 1.0e-10;
    x_right(1) = 0.5;
    mfem::IntegrationPoint ip_left;
    mfem::IntegrationPoint ip_right;
    if (owns_left)
    {
      mfem::ElementTransformation* Tr_left =
        example_ctx.spaces.h1_vfes->GetElementTransformation(left_e);
      Tr_left->TransformBack(x_left, ip_left);
    }
    if (owns_right)
    {
      mfem::ElementTransformation* Tr_right =
        example_ctx.spaces.h1_vfes->GetElementTransformation(right_e);
      Tr_right->TransformBack(x_right, ip_right);
    }

    mfem::Vector ul(2);
    mfem::Vector ur(2);
    ul = 0.0;
    ur = 0.0;
    if (owns_left)
    {
      mfem::ElementTransformation* Tr_left =
        example_ctx.spaces.h1_vfes->GetElementTransformation(left_e);
      h1_diag.u_h1.GetVectorValue(*Tr_left, ip_left, ul);
    }
    if (owns_right)
    {
      mfem::ElementTransformation* Tr_right =
        example_ctx.spaces.h1_vfes->GetElementTransformation(right_e);
      h1_diag.u_h1.GetVectorValue(*Tr_right, ip_right, ur);
    }

    const double ul0 = ReduceOwnedValue(ul(0), owns_left);
    const double ur0 = ReduceOwnedValue(ur(0), owns_right);

    // With arithmetic averaging, shared interface velocity is (0 + 2)/2 = 1.
    ctx.CheckNear(ul0,
                  1.0,
                  1.0e-8,
                  "H1 velocity projection averages discontinuous DG states at "
                  "interface (left).");
    ctx.CheckNear(ur0,
                  1.0,
                  1.0e-8,
                  "H1 velocity projection averages discontinuous DG states at "
                  "interface (right).");
  }

  // L2 diagnostics remain broken/local (no H1-style shared-dof averaging).
  {
    mfem::Mesh serial_mesh =
      mfem::Mesh::MakeCartesian2D(2,
                                  1,
                                  mfem::Element::QUADRILATERAL,
                                  /*gen_edges=*/true,
                                  /*sx=*/1.0,
                                  /*sy=*/1.0);
    auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    dg_sim::RunConfig run_cfg = MakeRunConfig(/*order=*/1);
    dg_sim::ExampleContext example_ctx(run_cfg,
                                       std::move(pmesh),
                                       /*sigma_space=*/"L2",
                                       /*need_sigma=*/false,
                                       /*need_h1=*/false);
    dg_sim::EulerStateBundle bundle(example_ctx, gamma, rho_floor);

    mfem::VectorFunctionCoefficient ic(
      4,
      [](const mfem::Vector& x, mfem::Vector& y)
      {
        y.SetSize(4);
        const double rho = 1.0;
        const double ux = (x(0) < 0.5) ? x(0) : (2.0 * x(0) - 0.5);
        const double p = 1.0;
        y(0) = rho;
        y(1) = rho * ux;
        y(2) = 0.0;
        y(3) = p / 0.4 + 0.5 * rho * ux * ux;
      });
    bundle.sol.ProjectCoefficient(ic);

    dg_sim::L2Diagnostics l2_diag(example_ctx.spaces.fes,
                                  bundle.sol,
                                  example_ctx.dim,
                                  rho_floor);
    l2_diag.Update();

    ctx.Check(l2_diag.g_div_l2.ParFESpace()->GetVDim() == 1,
              "L2 diagnostics div field uses scalar space.");
    ctx.Check(l2_diag.g_gfrob_l2.ParFESpace()->GetVDim() == 1,
              "L2 diagnostics grad-frobenius field uses scalar space.");
    ctx.Check(dynamic_cast<const mfem::DG_FECollection*>(
                l2_diag.g_div_l2.ParFESpace()->FEColl()) != nullptr,
              "L2 diagnostics use DG/L2 finite element collection.");

    const int left_e = FindInterfaceSideElement(example_ctx, /*left=*/true);
    const int right_e = FindInterfaceSideElement(example_ctx, /*left=*/false);
    const bool owns_left = (left_e >= 0);
    const bool owns_right = (right_e >= 0);
    const double left_count = ReduceOwnedCount(owns_left);
    const double right_count = ReduceOwnedCount(owns_right);
    ctx.Check(left_count == 1.0 && right_count == 1.0,
              "L2 diagnostics test mesh exposes left/right interface elements.");

    mfem::IntegrationPoint ip_left;
    mfem::IntegrationPoint ip_right;
    ip_left.Set2(0.5, 0.5);
    ip_right.Set2(0.5, 0.5);

    const double div_left_local = [&]()
    {
      if (!owns_left)
      {
        return 0.0;
      }
      mfem::ElementTransformation* Tr_left =
        example_ctx.spaces.fes.GetElementTransformation(left_e);
      return l2_diag.g_div_l2.GetValue(*Tr_left, ip_left);
    }();
    const double div_right_local = [&]()
    {
      if (!owns_right)
      {
        return 0.0;
      }
      mfem::ElementTransformation* Tr_right =
        example_ctx.spaces.fes.GetElementTransformation(right_e);
      return l2_diag.g_div_l2.GetValue(*Tr_right, ip_right);
    }();
    const double div_left = ReduceOwnedValue(div_left_local, owns_left);
    const double div_right = ReduceOwnedValue(div_right_local, owns_right);

    // For u_x = x (left) and u_x = 2x (right), broken gradients stay local:
    // div_left ~= 1, div_right ~= 2 in the two elements.
    ctx.CheckNear(div_left,
                  1.0,
                  1.0e-6,
                  "L2 diagnostics keep local left-element divergence.");
    ctx.CheckNear(div_right,
                  2.0,
                  1.0e-6,
                  "L2 diagnostics keep local right-element divergence.");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests
