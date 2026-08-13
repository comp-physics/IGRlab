#include "physics/bc/boundary_state.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "sim/context/case_context.hpp"
#include "sim/context/mesh_build.hpp"
#include "sim/wiring/bc_wiring.hpp"

#include <memory>
#include <string>
#include <type_traits>

static_assert(std::is_constructible_v<dg_sim::CaseContext,
                                      std::unique_ptr<mfem::ParMesh>,
                                      dg_sim::MeshBuildMetadata>);
static_assert(!std::is_constructible_v<dg_sim::CaseContext,
                                       std::unique_ptr<mfem::ParMesh>,
                                       MPI_Comm,
                                       dg_sim::MeshBuildMetadata>);

#include "test_utils.hpp"

namespace dg_igr_tests
{

int RunBCTests(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr double gamma = 1.4;
  const double tol = 1.0e-12;

  mfem::FaceElementTransformations Tr;
  mfem::Vector nor(dim);
  nor = 0.0;
  nor(0) = 1.0;

  // --- Supersonic inlet enforces the prescribed state ---
  const mfem::Vector Uin = EulerConsFromPrim2D(/*rho=*/1.0,
                                               /*u=*/2.0,
                                               /*v=*/0.5,
                                               /*p=*/1.0,
                                               gamma);
  const mfem::Vector Uminus_inlet = EulerConsFromPrim2D(/*rho=*/0.8,
                                                        /*u=*/-1.0,
                                                        /*v=*/0.25,
                                                        /*p=*/0.7,
                                                        gamma);
  mfem::SupersonicInletState inlet(Uin);
  mfem::Vector Uplus(dim + 2);
  inlet.Eval(Uminus_inlet, nor, Tr, Uplus);
  ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
  for (int i = 0; i < Uplus.Size(); ++i)
  {
    ctx.CheckNear(Uplus(i),
                  Uin(i),
                  tol,
                  "SupersonicInletState enforces Uin component " +
                    std::to_string(i));
  }

  // --- Supersonic outlet is transmissive ---
  mfem::SupersonicOutletState outlet;
  const mfem::Vector Uminus_outlet = EulerConsFromPrim2D(/*rho=*/1.1,
                                                         /*u=*/0.75,
                                                         /*v=*/-0.25,
                                                         /*p=*/0.9,
                                                         gamma);
  outlet.Eval(Uminus_outlet, nor, Tr, Uplus);
  ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
  for (int i = 0; i < Uplus.Size(); ++i)
  {
    ctx.CheckNear(Uplus(i),
                  Uminus_outlet(i),
                  tol,
                  "SupersonicOutletState copies interior state component " +
                    std::to_string(i));
  }

  // --- Subsonic outlet pressure enforces pressure on subsonic branch ---
  {
    const mfem::Vector Uminus_sub_out = EulerConsFromPrim2D(/*rho=*/1.0,
                                                            /*u=*/0.3,
                                                            /*v=*/0.1,
                                                            /*p=*/1.2,
                                                            gamma);
    constexpr double p_out = 1.5;
    mfem::SubsonicOutletPressureState sub_out(dim, gamma, p_out);
    sub_out.Eval(Uminus_sub_out, nor, Tr, Uplus);
    ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));

    mfem::Vector prim(dim + 2);
    mfem::EulerConservativeToPrimitive(Uplus, dim, gamma, prim);
    ctx.CheckNear(prim(1 + dim),
                  p_out,
                  1.0e-10,
                  "SubsonicOutletPressureState enforces outlet pressure");
  }

  // --- Non-reflecting outlet remains finite on subsonic branch ---
  {
    const mfem::Vector Uref = EulerConsFromPrim2D(/*rho=*/1.4,
                                                  /*u=*/0.2,
                                                  /*v=*/0.0,
                                                  /*p=*/1.6,
                                                  gamma);
    const mfem::Vector Uminus_nr = EulerConsFromPrim2D(/*rho=*/1.2,
                                                       /*u=*/0.4,
                                                       /*v=*/-0.1,
                                                       /*p=*/1.3,
                                                       gamma);
    mfem::NonReflectingOutletState nr_out(Uref, dim, gamma);
    nr_out.Eval(Uminus_nr, nor, Tr, Uplus);
    ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
    ctx.Check(std::isfinite(Uplus(0)) && std::isfinite(Uplus(1)) &&
                std::isfinite(Uplus(2)) && std::isfinite(Uplus(3)),
              "NonReflectingOutletState produces finite conservative state");
  }

  // --- Subsonic inlet remains finite and keeps inflow direction ---
  {
    const mfem::Vector Uin_sub = EulerConsFromPrim2D(/*rho=*/1.0,
                                                     /*u=*/-0.6,
                                                     /*v=*/0.0,
                                                     /*p=*/1.0,
                                                     gamma);
    const mfem::Vector Uminus_sub_in = EulerConsFromPrim2D(/*rho=*/1.2,
                                                           /*u=*/-0.2,
                                                           /*v=*/0.05,
                                                           /*p=*/1.1,
                                                           gamma);
    mfem::SubsonicInletState sub_in(Uin_sub, dim, gamma);
    sub_in.Eval(Uminus_sub_in, nor, Tr, Uplus);
    ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));

    mfem::Vector prim(dim + 2);
    mfem::EulerConservativeToPrimitive(Uplus, dim, gamma, prim);
    ctx.Check(prim(1) < 0.0,
              "SubsonicInletState keeps inward normal velocity at inlet");
  }

  // --- Far-field state recovers supersonic branches ---
  {
    const mfem::Vector Ufar = EulerConsFromPrim2D(/*rho=*/1.0,
                                                  /*u=*/-2.0,
                                                  /*v=*/0.0,
                                                  /*p=*/1.0,
                                                  gamma);
    mfem::FarFieldState far(Ufar, dim, gamma);

    // Supersonic inflow at this boundary -> should enforce far-field state.
    const mfem::Vector Uminus_sup_in = EulerConsFromPrim2D(/*rho=*/0.9,
                                                           /*u=*/-1.8,
                                                           /*v=*/0.1,
                                                           /*p=*/0.95,
                                                           gamma);
    far.Eval(Uminus_sup_in, nor, Tr, Uplus);
    ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
    for (int i = 0; i < Uplus.Size(); ++i)
    {
      ctx.CheckNear(Uplus(i),
                    Ufar(i),
                    1.0e-12,
                    "FarFieldState enforces far-field state in supersonic inflow component " +
                      std::to_string(i));
    }

    // Supersonic outflow -> transmissive.
    const mfem::Vector Uminus_sup_out = EulerConsFromPrim2D(/*rho=*/1.1,
                                                            /*u=*/2.2,
                                                            /*v=*/-0.05,
                                                            /*p=*/1.2,
                                                            gamma);
    far.Eval(Uminus_sup_out, nor, Tr, Uplus);
    ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
    for (int i = 0; i < Uplus.Size(); ++i)
    {
      ctx.CheckNear(Uplus(i),
                    Uminus_sup_out(i),
                    1.0e-12,
                    "FarFieldState is transmissive in supersonic outflow component " +
                      std::to_string(i));
    }
  }

  // --- Slip wall reflects the normal momentum and preserves tangential ---
  mfem::SlipWallState slip(dim);
  const mfem::Vector Uminus_wall = EulerConsFromPrim2D(/*rho=*/1.25,
                                                       /*u=*/3.0,
                                                       /*v=*/-1.5,
                                                       /*p=*/1.0,
                                                       gamma);
  slip.Eval(Uminus_wall, nor, Tr, Uplus);

  const double rho = Uminus_wall(0);
  const double mx = Uminus_wall(1);
  const double my = Uminus_wall(2);
  const double energy = Uminus_wall(3);

  ctx.AddParity(Uplus(0) + Uplus(1) + Uplus(2) + Uplus(3));
  ctx.CheckNear(Uplus(0), rho, tol, "SlipWallState preserves density");
  ctx.CheckNear(Uplus(1), -mx, 1.0e-11, "SlipWallState flips normal momentum");
  ctx.CheckNear(Uplus(2),
                my,
                1.0e-11,
                "SlipWallState preserves tangential momentum");
  ctx.CheckNear(Uplus(3), energy, 1.0e-11, "SlipWallState preserves energy");

  // --- Mesh requests retain CLI/default precedence and refinement metadata ---
  {
    dg_sim::RunConfig run_cfg;
    run_cfg.case_config.case_dir = "cases/example";
    run_cfg.case_config.default_mesh_file = "default.mesh";
    run_cfg.opts.ser_ref_levels = 2;
    run_cfg.opts.par_ref_levels = 3;
    run_cfg.mesh_path = "cases/example/default.mesh";
    const dg_sim::MeshBuildRequest default_request =
      dg_sim::ResolveMeshBuildRequest(run_cfg);
    ctx.Check(default_request.mesh_path == run_cfg.mesh_path,
              "Mesh build request retains the resolved default mesh path");
    ctx.Check(default_request.mesh_path_precedence ==
                dg_sim::MeshPathPrecedence::CaseDefault,
              "Mesh build request records default-mesh precedence");
    ctx.Check(default_request.serial_refinement_levels == 2 &&
                default_request.parallel_refinement_levels == 3,
              "Mesh build request records serial then parallel refinement levels");

    run_cfg.opts.mesh_file = "override.mesh";
    run_cfg.mesh_path = "override.mesh";
    const dg_sim::MeshBuildRequest override_request =
      dg_sim::ResolveMeshBuildRequest(run_cfg);
    ctx.Check(override_request.mesh_path == "override.mesh" &&
                override_request.mesh_path_precedence ==
                  dg_sim::MeshPathPrecedence::CommandLineOverride,
              "Mesh build request gives a CLI mesh override precedence");
  }

  // --- Semantic plan lowering is identical to the established DG wiring ---
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/1,
      /*ny=*/1,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::DG_FECollection fec(/*order=*/1, dim);
    mfem::ParFiniteElementSpace vfes(&pmesh,
                                     &fec,
                                     /*vdim=*/dim + 2,
                                     mfem::Ordering::byNODES);
    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides numerical_flux(flux);
    constexpr int int_order_offset = 1;
    mfem::DGHyperbolicConservationLaws direct(
      vfes,
      std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux,
                                                        int_order_offset));
    mfem::DGHyperbolicConservationLaws lowered(
      vfes,
      std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux,
                                                        int_order_offset));
    mfem::Array<int> all_boundaries(pmesh.bdr_attributes.Max());
    all_boundaries = 1;
    const mfem::Vector reference_state = EulerConsFromPrim2D(
      /*rho=*/1.1, /*u=*/0.3, /*v=*/-0.2, /*p=*/1.2, gamma);

    dg_sim::AddSupersonicInlet(direct,
                               numerical_flux,
                               reference_state,
                               int_order_offset,
                               all_boundaries);
    dg_sim::AddSupersonicOutlet(
      direct, numerical_flux, int_order_offset, all_boundaries);
    dg_sim::AddSubsonicInlet(direct,
                             numerical_flux,
                             reference_state,
                             int_order_offset,
                             dim,
                             gamma,
                             all_boundaries);
    dg_sim::AddSubsonicOutletPressure(direct,
                                      numerical_flux,
                                      1.2,
                                      int_order_offset,
                                      dim,
                                      gamma,
                                      all_boundaries);
    dg_sim::AddFarField(direct,
                        numerical_flux,
                        reference_state,
                        int_order_offset,
                        dim,
                        gamma,
                        all_boundaries);
    dg_sim::AddSlipWall(
      direct, numerical_flux, int_order_offset, dim, all_boundaries);
    dg_sim::AddNonReflectingOutlet(direct,
                                   numerical_flux,
                                   reference_state,
                                   int_order_offset,
                                   dim,
                                   gamma,
                                   all_boundaries);

    dg_sim::EulerBoundaryPlan plan;
    plan.AddPeriodic(all_boundaries);
    plan.AddSupersonicInlet(all_boundaries,
                            [reference_state]() { return reference_state; });
    plan.AddSupersonicOutlet(all_boundaries);
    plan.AddSubsonicInlet(all_boundaries,
                          [reference_state]() { return reference_state; });
    plan.AddSubsonicOutletPressure(all_boundaries, []() { return 1.2; });
    plan.AddFarField(all_boundaries,
                     [reference_state]() { return reference_state; });
    plan.AddSlipWall(all_boundaries);
    plan.AddNonReflectingOutlet(all_boundaries,
                                [reference_state]() { return reference_state; });
    dg_sim::LowerDGEulerBoundaryPlan(lowered,
                                     numerical_flux,
                                     int_order_offset,
                                     dim,
                                     gamma,
                                     plan);

    mfem::ParGridFunction state(&vfes);
    FillConstantState(EulerConsFromPrim2D(/*rho=*/1.0,
                                          /*u=*/0.1,
                                          /*v=*/0.0,
                                          /*p=*/1.0,
                                          gamma),
                      state);
    mfem::Vector input;
    state.GetTrueDofs(input);
    mfem::Vector direct_output(vfes.GetTrueVSize());
    mfem::Vector lowered_output(vfes.GetTrueVSize());
    direct.Mult(input, direct_output);
    lowered.Mult(input, lowered_output);
    direct_output -= lowered_output;
    ctx.Check(direct_output.Normlinf() == 0.0,
              "Euler boundary plan lowering exactly reuses DG integrators");
  }

  // --- CaseContext has a single communicator authority: its owned mesh ---
  {
    mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/1,
      /*ny=*/1,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    auto parallel_mesh =
      std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
    dg_sim::MeshBuildMetadata metadata;
    dg_sim::CaseContext case_context(std::move(parallel_mesh),
                                     std::move(metadata));
    int communicator_relation = MPI_UNEQUAL;
    MPI_Comm_compare(case_context.Comm(),
                     case_context.Mesh().GetComm(),
                     &communicator_relation);
    ctx.Check(communicator_relation == MPI_IDENT,
              "CaseContext communicator is the owned mesh communicator");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests
