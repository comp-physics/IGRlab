#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/euler/conversions.hpp"
#include "physics/euler/eos.hpp"
#include "utils/mesh/boundary_markers.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace dg_igr_tests
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

mfem::Mesh MakePeriodicMesh1D(int n)
{
  mfem::Mesh orig_mesh = mfem::Mesh::MakeCartesian1D(n, 1.0);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0})};
  return mfem::Mesh::MakePeriodic(
      orig_mesh, orig_mesh.CreatePeriodicVertexMapping(translations));
}

void ProjectAdvectionState(mfem::ParGridFunction& U,
                           double gamma,
                           double u0,
                           double p0,
                           double rho0,
                           double amp,
                           double time)
{
  auto ic_fun = [=](const mfem::Vector& x, mfem::Vector& Uout)
  {
    const double rho = rho0 + amp * std::sin(2.0 * kPi * (x(0) - u0 * time));
    mfem::Vector prim(3);
    prim(0) = rho;
    prim(1) = u0;
    prim(2) = p0;
    mfem::EulerPrimToConservative(prim, /*dim=*/1, gamma, Uout);
  };
  mfem::VectorFunctionCoefficient ic(3, ic_fun);
  U.ProjectCoefficient(ic);
}

enum class BoundaryMode
{
  InletOutlet,
  AllOutlet,
  AllSlip
};

mfem::Mesh MakePeriodicMesh2D(int nx, int ny)
{
  mfem::Mesh orig_mesh =
      mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL,
                                  /*gen_edges=*/false, 1.0, 1.0);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                            mfem::Vector({0.0, 1.0})};
  return mfem::Mesh::MakePeriodic(
      orig_mesh, orig_mesh.CreatePeriodicVertexMapping(translations));
}

void ProjectAdvectionState2D(mfem::ParGridFunction& U,
                             double gamma,
                             double u0,
                             double v0,
                             double p0,
                             double rho0,
                             double amp,
                             double time)
{
  const double sigma = 0.12;
  auto ic_fun = [=](const mfem::Vector& x, mfem::Vector& Uout)
  {
    const double dx = x(0) - 0.5 - u0 * time;
    const double dy = x(1) - 0.5 - v0 * time;
    const double r2 = dx * dx + dy * dy;
    const double rho = rho0 + amp * std::exp(-r2 / (sigma * sigma));

    mfem::Vector prim(4);
    prim(0) = rho;
    prim(1) = u0;
    prim(2) = v0;
    prim(3) = p0;
    mfem::EulerPrimToConservative(prim, /*dim=*/2, gamma, Uout);
  };
  mfem::VectorFunctionCoefficient ic(4, ic_fun);
  U.ProjectCoefficient(ic);
}

struct MinMaxStats
{
  double rho_min = 0.0;
  double rho_max = 0.0;
  double p_min = 0.0;
  double p_max = 0.0;
};

struct BoundaryFluxDiagnostics
{
  double inlet_state_diff = 0.0;
  double inlet_ghost_diff = 0.0;
  double outlet_ghost_diff = 0.0;
  double inlet_flux_diff = 0.0;
  double outlet_flux_diff = 0.0;
  double inlet_flux_norm = 0.0;
  double outlet_flux_norm = 0.0;
};

BoundaryFluxDiagnostics MaxDiagnostics(const BoundaryFluxDiagnostics& a,
                                       const BoundaryFluxDiagnostics& b)
{
  BoundaryFluxDiagnostics out;
  out.inlet_state_diff = std::max(a.inlet_state_diff, b.inlet_state_diff);
  out.inlet_ghost_diff = std::max(a.inlet_ghost_diff, b.inlet_ghost_diff);
  out.outlet_ghost_diff = std::max(a.outlet_ghost_diff, b.outlet_ghost_diff);
  out.inlet_flux_diff = std::max(a.inlet_flux_diff, b.inlet_flux_diff);
  out.outlet_flux_diff = std::max(a.outlet_flux_diff, b.outlet_flux_diff);
  out.inlet_flux_norm = std::max(a.inlet_flux_norm, b.inlet_flux_norm);
  out.outlet_flux_norm = std::max(a.outlet_flux_norm, b.outlet_flux_norm);
  return out;
}

BoundaryFluxDiagnostics ComputeBoundaryFluxDiagnostics(const mfem::ParGridFunction& U,
                                                       mfem::ParMesh& pmesh,
                                                       const mfem::NumericalFlux& num_flux,
                                                       const mfem::Vector& inlet_state,
                                                       int order,
                                                       int dim)
{
  BoundaryFluxDiagnostics local;
  const int num_eq = dim + 2;
  mfem::Vector state1(num_eq);
  mfem::Vector state2(num_eq);
  mfem::Vector flux(num_eq);
  mfem::Vector flux_ref(num_eq);
  mfem::Vector nor(dim);
  mfem::Vector tmp(num_eq);

  mfem::SupersonicInletState inlet_bc(inlet_state);
  mfem::SupersonicOutletState outlet_bc;
  const mfem::IntegrationRule& ir = mfem::dgsem::GetLGLFaceRule(order);

  for (int b = 0; b < pmesh.GetNBE(); ++b)
  {
    const int attr = pmesh.GetBdrAttribute(b);
    const mfem::BoundaryState* bc = nullptr;
    if (attr == 4)
    {
      bc = &inlet_bc;
    }
    else if (attr == 2)
    {
      bc = &outlet_bc;
    }
    else
    {
      continue;
    }

    mfem::FaceElementTransformations* Tr = pmesh.GetBdrFaceTransformations(b);
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      mfem::ElementTransformation& Tr1 = Tr->GetElement1Transformation();
      const mfem::IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
      U.GetVectorValue(Tr1, ip1, state1);

      if (dim == 1)
      {
        nor(0) = (ip1.x - 0.5) * 2.0;
      }
      else
      {
        mfem::CalcOrtho(Tr->Jacobian(), nor);
      }

      bc->Eval(state1, nor, *Tr, state2);
      num_flux.Eval(state1, state2, nor, *Tr, flux);

      if (attr == 4)
      {
        tmp = state1;
        tmp -= inlet_state;
        local.inlet_state_diff =
            std::max(local.inlet_state_diff, tmp.Norml2());

        tmp = state2;
        tmp -= inlet_state;
        local.inlet_ghost_diff =
            std::max(local.inlet_ghost_diff, tmp.Norml2());

        num_flux.Eval(inlet_state, inlet_state, nor, *Tr, flux_ref);
        tmp = flux;
        tmp -= flux_ref;
        local.inlet_flux_diff =
            std::max(local.inlet_flux_diff, tmp.Norml2());
        local.inlet_flux_norm =
            std::max(local.inlet_flux_norm, flux.Norml2());
      }
      else if (attr == 2)
      {
        tmp = state2;
        tmp -= state1;
        local.outlet_ghost_diff =
            std::max(local.outlet_ghost_diff, tmp.Norml2());

        num_flux.Eval(state1, state1, nor, *Tr, flux_ref);
        tmp = flux;
        tmp -= flux_ref;
        local.outlet_flux_diff =
            std::max(local.outlet_flux_diff, tmp.Norml2());
        local.outlet_flux_norm =
            std::max(local.outlet_flux_norm, flux.Norml2());
      }
    }
  }

  BoundaryFluxDiagnostics global;
  global.inlet_state_diff = GlobalMax(local.inlet_state_diff);
  global.inlet_ghost_diff = GlobalMax(local.inlet_ghost_diff);
  global.outlet_ghost_diff = GlobalMax(local.outlet_ghost_diff);
  global.inlet_flux_diff = GlobalMax(local.inlet_flux_diff);
  global.outlet_flux_diff = GlobalMax(local.outlet_flux_diff);
  global.inlet_flux_norm = GlobalMax(local.inlet_flux_norm);
  global.outlet_flux_norm = GlobalMax(local.outlet_flux_norm);
  return global;
}

MinMaxStats ComputeRhoPressureMinMax(const mfem::ParGridFunction& U,
                                     int dim,
                                     double gamma)
{
  MinMaxStats stats;
  const int num_eq = dim + 2;
  const double* data = U.GetData();
  const int ndofs = U.Size() / num_eq;

  stats.rho_min = std::numeric_limits<double>::infinity();
  stats.rho_max = -std::numeric_limits<double>::infinity();
  stats.p_min = std::numeric_limits<double>::infinity();
  stats.p_max = -std::numeric_limits<double>::infinity();

  mfem::Vector cons(num_eq);
  for (int i = 0; i < ndofs; ++i)
  {
    const double rho = data[i * num_eq];
    cons(0) = rho;
    for (int k = 1; k < num_eq; ++k)
    {
      cons(k) = data[i * num_eq + k];
    }
    const double p = mfem::EulerPressureFromConservative(cons, dim, gamma, 0.0);
    stats.rho_min = std::min(stats.rho_min, rho);
    stats.rho_max = std::max(stats.rho_max, rho);
    stats.p_min = std::min(stats.p_min, p);
    stats.p_max = std::max(stats.p_max, p);
  }

  stats.rho_min = GlobalMin(stats.rho_min);
  stats.rho_max = GlobalMax(stats.rho_max);
  stats.p_min = GlobalMin(stats.p_min);
  stats.p_max = GlobalMax(stats.p_max);
  return stats;
}

double MaxInletBoundaryError(const mfem::ParGridFunction& U,
                             mfem::ParMesh& pmesh,
                             const mfem::Vector& expected,
                             int order)
{
  const mfem::IntegrationRule& ir =
      mfem::dgsem::GetLGLFaceRule(order);
  double local_max = 0.0;
  mfem::Vector state(expected.Size());
  for (int b = 0; b < pmesh.GetNBE(); ++b)
  {
    const int attr = pmesh.GetBdrAttribute(b);
    if (attr != 4) { continue; } // left boundary on Cartesian meshes
    mfem::ElementTransformation* Tr = pmesh.GetBdrElementTransformation(b);
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      U.GetVectorValue(*Tr, ip, state);
      state -= expected;
      local_max = std::max(local_max, state.Norml2());
      state = expected;
    }
  }
  return GlobalMax(local_max);
}

bool AllFinite(const mfem::ParGridFunction& U)
{
  bool local_ok = true;
  for (int i = 0; i < U.Size(); ++i)
  {
    if (!std::isfinite(U(i)))
    {
      local_ok = false;
      break;
    }
  }
  int local = local_ok ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, mfem::MPITypeMap<int>::mpi_type,
                MPI_LAND, MPI_COMM_WORLD);
  return global == 1;
}

} // namespace

int RunDGSEMAdvectionTests(TestContext& ctx)
{
  constexpr int dim = 1;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;
  constexpr int order = 3;
  constexpr int int_order_offset = 2;
  const double tol_resid = 1.0e-12;

  // Constant-state residual (periodic, DGSEM volume mode).
  {
    mfem::Mesh mesh = MakePeriodicMesh1D(8);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq,
                                     mfem::Ordering::byNODES);

    mfem::ParGridFunction U(&vfes);
    const mfem::Vector cons_const =
        EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/0.25, /*p=*/1.0, gamma);
    FillConstantState(cons_const, U);

    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides num_flux(flux);
    auto form_integrator =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                         int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
                                            /*preassembleWeakDivergence=*/true);
    oper.SetNumericalFlux(&num_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order));

    mfem::Vector x;
    mfem::Vector y(vfes.GetTrueVSize());
    U.GetTrueDofs(x);
    oper.Mult(x, y);

    const double resid_norm = GlobalL2Norm(y);
    ctx.AddParity(resid_norm);
    ctx.Check(resid_norm < tol_resid,
              "DGSEM 1D constant-state residual ~0 (norm=" +
                  FormatSci(resid_norm) + ")");
  }

  // 1D smooth advection convergence (periodic).
  {
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double p0 = 1.0;
    const double amp = 0.02;
    const double t_final = 0.1;
    const double a0 = std::sqrt(gamma * p0 / rho0);
    const double wave_speed = std::abs(u0) + a0;
    const double cfl = 0.2;

    const std::vector<int> ns = {20, 40, 80};
    std::vector<double> errors;
    errors.reserve(ns.size());

    for (int n : ns)
    {
      mfem::Mesh mesh = MakePeriodicMesh1D(n);
      mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

      mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
      mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq,
                                       mfem::Ordering::byNODES);

      mfem::ParGridFunction U(&vfes);
      ProjectAdvectionState(U, gamma, u0, p0, rho0, amp, /*time=*/0.0);

      mfem::EulerFlux flux(dim, gamma);
      mfem::RusanovFluxWithSides num_flux(flux);
      auto form_integrator =
          std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                           int_order_offset);
      mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
                                              /*preassembleWeakDivergence=*/true);
      oper.SetNumericalFlux(&num_flux);
      oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
      oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order));

      mfem::RK3SSPWithLimiter rk(vfes);
      rk.Init(oper);

      double t = 0.0;
      const double dx = 1.0 / static_cast<double>(n);
      const double dt_base = cfl * dx / wave_speed;

      mfem::Vector x;
      U.GetTrueDofs(x);
      while (t + 1.0e-14 < t_final)
      {
        double dt = std::min(dt_base, t_final - t);
        rk.Step(x, t, dt);
      }
      U.SetFromTrueDofs(x);

      auto exact_fun = [=](const mfem::Vector& xpos, mfem::Vector& Uout)
      {
        const double rho =
            rho0 + amp * std::sin(2.0 * kPi * (xpos(0) - u0 * t_final));
        mfem::Vector prim(3);
        prim(0) = rho;
        prim(1) = u0;
        prim(2) = p0;
        mfem::EulerPrimToConservative(prim, /*dim=*/1, gamma, Uout);
      };
      mfem::VectorFunctionCoefficient exact(num_eq, exact_fun);
      const double err = U.ComputeL2Error(exact);
      errors.push_back(err);
      ctx.AddParity(err);
      ctx.LogMetric("DGSEM 1D advection L2 error (n=" +
                    std::to_string(n) + ", err=" + FormatSci(err) + ")");
    }

    for (size_t i = 1; i < errors.size(); ++i)
    {
      const double ratio = errors[i - 1] / std::max(errors[i], 1.0e-16);
      const double obs_order = std::log(ratio) / std::log(2.0);
      ctx.Check(obs_order > 1.0,
                "DGSEM 1D advection observed order > 1 (p=" +
                    std::to_string(order) +
                    ", order=" + FormatSci(obs_order) + ")");
    }
  }

  // 2D smooth advection investigation matrix (boundary artifact isolation).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    constexpr int nx = 8;
    constexpr int ny = 4;
    const double rho0 = 1.0;
    const double u0 = 2.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double base_amp = 0.02;
    const double es_amp = 0.0;
    const double a0 = std::sqrt(gamma * p0 / rho0);
    const double wave_speed = std::abs(u0) + a0;
    const double cfl = 0.1;
    const double t_final = 0.005;

    struct CaseConfig
    {
      bool dgsem = true;
      mfem::DGSEMVolumeFlux vol_flux = mfem::DGSEMVolumeFlux::Central;
      mfem::DGSEMSurfaceFluxType surf_flux = mfem::DGSEMSurfaceFluxType::Rusanov;
      BoundaryMode bc_mode = BoundaryMode::InletOutlet;
      bool periodic = false;
      bool evolve = true;
      bool use_positivity_limiter = false;
      double amp = 0.0;
      std::string label;
    };

    auto run_case = [&](const CaseConfig& cfg)
    {
      mfem::Mesh mesh = cfg.periodic
          ? MakePeriodicMesh2D(nx, ny)
          : mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL,
                                        /*gen_edges=*/true, 1.0, 1.0);
      mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

      mfem::DG_FECollection fec(order, dim2, mfem::BasisType::GaussLobatto);
      mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq2,
                                       mfem::Ordering::byNODES);

      mfem::ParGridFunction U(&vfes);
      ProjectAdvectionState2D(U, gamma, u0, v0, p0, rho0, cfg.amp, /*time=*/0.0);

      mfem::NumericalFlux* num_flux = nullptr;
      std::unique_ptr<mfem::DGSEMSurfaceFlux> dgsem_flux;
      std::unique_ptr<mfem::RusanovFluxWithSides> dg_flux;

      mfem::EulerFlux base_flux(dim2, gamma);

      if (cfg.dgsem)
      {
        dgsem_flux = std::make_unique<mfem::DGSEMSurfaceFlux>(base_flux);
        dgsem_flux->SetSurfaceFluxType(cfg.surf_flux);
        dgsem_flux->SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Ranocha);
        dgsem_flux->SetGamma(gamma);
        num_flux = dgsem_flux.get();
      }
      else
      {
        dg_flux = std::make_unique<mfem::RusanovFluxWithSides>(base_flux);
        num_flux = dg_flux.get();
      }

      auto form_integrator =
          std::make_unique<mfem::HyperbolicFormIntegrator>(*num_flux,
                                                           int_order_offset);
      mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
                                              /*preassembleWeakDivergence=*/true);
      oper.SetNumericalFlux(num_flux);

      if (cfg.dgsem)
      {
        oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
        oper.SetDGSEMVolumeFlux(cfg.vol_flux);
        oper.SetDGSEMGamma(gamma);
        oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order));
      }

      if (!cfg.periodic)
      {
        mfem::Array<int> marker_inlet;
        mfem::Array<int> marker_outlet;
        mfem::Array<int> marker_slip;
        mfem::MakeMarker(marker_inlet, pmesh, {4});
        mfem::MakeMarker(marker_outlet, pmesh, {2});
        mfem::MakeMarker(marker_slip, pmesh, {1, 3});

        auto add_inlet = [&](const mfem::Array<int>& marker)
        {
          mfem::Vector prim(4);
          prim(0) = rho0;
          prim(1) = u0;
          prim(2) = v0;
          prim(3) = p0;
          mfem::Vector Uin(4);
          mfem::EulerPrimToConservative(prim, /*dim=*/2, gamma, Uin);
          oper.AddBdrFaceIntegrator(
              new mfem::BdrHyperbolicFluxIntegrator(
                  *num_flux,
                  std::make_unique<mfem::SupersonicInletState>(Uin),
                  int_order_offset),
              marker);
        };

        auto add_outlet = [&](const mfem::Array<int>& marker)
        {
          oper.AddBdrFaceIntegrator(
              new mfem::BdrHyperbolicFluxIntegrator(
                  *num_flux,
                  std::make_unique<mfem::SupersonicOutletState>(),
                  int_order_offset),
              marker);
        };

        auto add_slip = [&](const mfem::Array<int>& marker)
        {
          oper.AddBdrFaceIntegrator(
              new mfem::BdrHyperbolicFluxIntegrator(
                  *num_flux,
                  std::make_unique<mfem::SlipWallState>(dim2),
                  int_order_offset),
              marker);
        };

        if (cfg.bc_mode == BoundaryMode::InletOutlet)
        {
          add_inlet(marker_inlet);
          add_outlet(marker_outlet);
          add_slip(marker_slip);
        }
        else if (cfg.bc_mode == BoundaryMode::AllOutlet)
        {
          add_outlet(marker_inlet);
          add_outlet(marker_outlet);
          add_outlet(marker_slip);
        }
        else if (cfg.bc_mode == BoundaryMode::AllSlip)
        {
          add_slip(marker_inlet);
          add_slip(marker_outlet);
          add_slip(marker_slip);
        }
      }

      if (cfg.evolve)
      {
        mfem::RK3SSPWithLimiter rk(vfes);
        rk.SetEulerParams(dim2, gamma);
        if (cfg.use_positivity_limiter)
        {
          rk.AddZhangShuLimiter(/*always_on=*/true);
        }
        rk.Init(oper);

        double t = 0.0;
        const double dx = 1.0 / static_cast<double>(nx);
        const double dt_base = cfl * dx / wave_speed;

        mfem::Vector x;
        U.GetTrueDofs(x);
        while (t + 1.0e-14 < t_final)
        {
          double dt = std::min(dt_base, t_final - t);
          rk.Step(x, t, dt);
        }
        U.SetFromTrueDofs(x);
      }
      else
      {
        mfem::Vector x;
        mfem::Vector y(vfes.GetTrueVSize());
        U.GetTrueDofs(x);
        oper.Mult(x, y);
        const double resid_norm = GlobalL2Norm(y);
        ctx.LogMetric("DGSEM 2D residual (" + cfg.label + "): " +
                      FormatSci(resid_norm));
      }

      const bool finite = AllFinite(U);
      ctx.Check(finite, "DGSEM 2D advection finite check (" + cfg.label + ")");

      const MinMaxStats stats = ComputeRhoPressureMinMax(U, dim2, gamma);
      ctx.LogMetric("DGSEM 2D advection stats (" + cfg.label + "): rho[min,max]=[" +
                    FormatSci(stats.rho_min) + ", " + FormatSci(stats.rho_max) +
                    "], p[min,max]=[" + FormatSci(stats.p_min) + ", " +
                    FormatSci(stats.p_max) + "]");

      if (!cfg.periodic && cfg.bc_mode == BoundaryMode::InletOutlet)
      {
        mfem::Vector prim(4);
        prim(0) = rho0;
        prim(1) = u0;
        prim(2) = v0;
        prim(3) = p0;
        mfem::Vector Uin(4);
        mfem::EulerPrimToConservative(prim, /*dim=*/2, gamma, Uin);
        const double inlet_err = MaxInletBoundaryError(U, pmesh, Uin, order);
        ctx.LogMetric("DGSEM 2D inlet max diff (" + cfg.label + "): " +
                      FormatSci(inlet_err));
      }
    };

    std::vector<CaseConfig> cases = {
        {true, mfem::DGSEMVolumeFlux::Central,
         mfem::DGSEMSurfaceFluxType::Rusanov, BoundaryMode::InletOutlet,
         false, true, true, base_amp, "dgsem-central+rusanov/inlet-outlet"},
        {false, mfem::DGSEMVolumeFlux::Central,
         mfem::DGSEMSurfaceFluxType::Rusanov, BoundaryMode::InletOutlet,
         false, true, false, base_amp, "dg (weak-div)/rusanov/inlet-outlet"},
        {true, mfem::DGSEMVolumeFlux::Central,
         mfem::DGSEMSurfaceFluxType::Rusanov, BoundaryMode::AllOutlet,
         false, true, true, base_amp, "dgsem-central+rusanov/all-outlet"},
        {true, mfem::DGSEMVolumeFlux::Central,
         mfem::DGSEMSurfaceFluxType::Rusanov, BoundaryMode::AllSlip,
         false, true, true, base_amp, "dgsem-central+rusanov/all-slip"},
        {true, mfem::DGSEMVolumeFlux::Central,
         mfem::DGSEMSurfaceFluxType::Rusanov, BoundaryMode::InletOutlet,
         true, true, false, base_amp, "dgsem-central+rusanov/periodic"}
    };

    const bool enable_es_cases = EnvEnabledDefault("DG_IGR_DGSEM_DIAG_ES", true);
    const bool enable_es_time_cases =
        EnvEnabledDefault("DG_IGR_DGSEM_DIAG_ES_TIME", true);
    const bool enable_pp_cases = EnvEnabledDefault("DG_IGR_DGSEM_DIAG_PP", true);
    if (enable_es_cases)
    {
      cases.push_back({true, mfem::DGSEMVolumeFlux::Ranocha,
                       mfem::DGSEMSurfaceFluxType::EntropyStableWinters,
                       BoundaryMode::InletOutlet, false, false, false, es_amp,
                       "dgsem-ranocha+es-winters/inlet-outlet"});
      cases.push_back({true, mfem::DGSEMVolumeFlux::Ranocha,
                       mfem::DGSEMSurfaceFluxType::EntropyStableLFEntropyVars,
                       BoundaryMode::InletOutlet, false, false, false, es_amp,
                       "dgsem-ranocha+es-lf/inlet-outlet"});
      cases.push_back({true, mfem::DGSEMVolumeFlux::Ranocha,
                       mfem::DGSEMSurfaceFluxType::EntropyStableWinters,
                       BoundaryMode::AllOutlet, false, false, false, es_amp,
                       "dgsem-ranocha+es-winters/all-outlet"});
      cases.push_back({true, mfem::DGSEMVolumeFlux::Ranocha,
                       mfem::DGSEMSurfaceFluxType::EntropyStableWinters,
                       BoundaryMode::AllSlip, false, false, false, es_amp,
                       "dgsem-ranocha+es-winters/all-slip"});
      cases.push_back({true, mfem::DGSEMVolumeFlux::Ranocha,
                       mfem::DGSEMSurfaceFluxType::EntropyStableWinters,
                       BoundaryMode::InletOutlet, true, false, false, es_amp,
                       "dgsem-ranocha+es-winters/periodic"});
    }
    else
    {
      ctx.LogMetric("DGSEM 2D ES diagnostics disabled (set DG_IGR_DGSEM_DIAG_ES=1).");
    }

    if (enable_es_time_cases)
    {
      cases.push_back({true, mfem::DGSEMVolumeFlux::Ranocha,
                       mfem::DGSEMSurfaceFluxType::EntropyStableWinters,
                       BoundaryMode::InletOutlet, false, true, false, base_amp,
                       "dgsem-ranocha+es-winters/inlet-outlet+time"});
    }
    else
    {
      ctx.LogMetric("DGSEM 2D ES time-evolution disabled (set DG_IGR_DGSEM_DIAG_ES_TIME=1).");
    }

    if (enable_pp_cases)
    {
      for (auto& cfg : cases)
      {
        if (cfg.dgsem && !cfg.periodic && cfg.evolve)
        {
          cfg.use_positivity_limiter = true;
        }
      }
      ctx.LogMetric("DGSEM 2D positivity limiter enabled for boundary cases "
                    "(set DG_IGR_DGSEM_DIAG_PP=0 to disable).");
    }
    else
    {
      ctx.LogMetric("DGSEM 2D positivity limiter disabled for diagnostics "
                    "(set DG_IGR_DGSEM_DIAG_PP=1 to enable).");
    }

    for (const auto& cfg : cases)
    {
      run_case(cfg);
    }
  }

  // 2D boundary-artifact regression: log inlet/outlet state + flux mismatches over time.
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    constexpr int nx = 8;
    constexpr int ny = 4;
    constexpr int order2 = 3;
    const double rho0 = 1.0;
    const double u0 = 2.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double amp = 0.02;
    const double a0 = std::sqrt(gamma * p0 / rho0);
    const double wave_speed = std::abs(u0) + a0;
    const double cfl = 0.1;
    const double t_final = 0.01;

    mfem::Mesh mesh =
        mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL,
                                    /*gen_edges=*/true, 1.0, 1.0);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec(order2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq2,
                                     mfem::Ordering::byNODES);

    mfem::ParGridFunction U(&vfes);
    ProjectAdvectionState2D(U, gamma, u0, v0, p0, rho0, amp, /*time=*/0.0);

    mfem::Vector prim(4);
    prim(0) = rho0;
    prim(1) = u0;
    prim(2) = v0;
    prim(3) = p0;
    mfem::Vector Uin(4);
    mfem::EulerPrimToConservative(prim, /*dim=*/2, gamma, Uin);

    const bool use_es = EnvEnabledDefault("DG_IGR_DGSEM_BDR_LOG_ES", true);
    mfem::EulerFlux base_flux(dim2, gamma);
    std::unique_ptr<mfem::DGSEMSurfaceFlux> dgsem_flux =
        std::make_unique<mfem::DGSEMSurfaceFlux>(base_flux);
    if (use_es)
    {
      dgsem_flux->SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::EntropyStableWinters);
      dgsem_flux->SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Ranocha);
      dgsem_flux->SetGamma(gamma);
    }
    else
    {
      dgsem_flux->SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::Rusanov);
    }

    mfem::NumericalFlux* num_flux = dgsem_flux.get();
    auto form_integrator =
        std::make_unique<mfem::HyperbolicFormIntegrator>(*num_flux,
                                                         int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
                                            /*preassembleWeakDivergence=*/true);
    oper.SetNumericalFlux(num_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(use_es ? mfem::DGSEMVolumeFlux::Ranocha
                                   : mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMGamma(gamma);
    oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order2));

    mfem::Array<int> marker_inlet;
    mfem::Array<int> marker_outlet;
    mfem::Array<int> marker_slip;
    mfem::MakeMarker(marker_inlet, pmesh, {4});
    mfem::MakeMarker(marker_outlet, pmesh, {2});
    mfem::MakeMarker(marker_slip, pmesh, {1, 3});

    oper.AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(
            *num_flux,
            std::make_unique<mfem::SupersonicInletState>(Uin),
            int_order_offset),
        marker_inlet);
    oper.AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(
            *num_flux,
            std::make_unique<mfem::SupersonicOutletState>(),
            int_order_offset),
        marker_outlet);
    oper.AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(
            *num_flux,
            std::make_unique<mfem::SlipWallState>(dim2),
            int_order_offset),
        marker_slip);

    mfem::RK3SSPWithLimiter rk(vfes);
    rk.Init(oper);

    mfem::Vector x;
    U.GetTrueDofs(x);

    BoundaryFluxDiagnostics diag0 =
        ComputeBoundaryFluxDiagnostics(U, pmesh, *num_flux, Uin, order2, dim2);
    BoundaryFluxDiagnostics diag_max = diag0;

    double t = 0.0;
    const double dx = 1.0 / static_cast<double>(nx);
    const double dt_base = cfl * dx / wave_speed;
    while (t + 1.0e-14 < t_final)
    {
      double dt = std::min(dt_base, t_final - t);
      rk.Step(x, t, dt);
      U.SetFromTrueDofs(x);
      BoundaryFluxDiagnostics diag_step =
          ComputeBoundaryFluxDiagnostics(U, pmesh, *num_flux, Uin, order2, dim2);
      diag_max = MaxDiagnostics(diag_max, diag_step);
    }

    ctx.LogMetric(std::string("DGSEM boundary diag (") +
                  (use_es ? "es-winters" : "rusanov") + "): "
                  "inlet_state_max=" + FormatSci(diag_max.inlet_state_diff) +
                  ", inlet_ghost_max=" + FormatSci(diag_max.inlet_ghost_diff) +
                  ", outlet_ghost_max=" + FormatSci(diag_max.outlet_ghost_diff));
    ctx.LogMetric(std::string("DGSEM boundary flux diag (") +
                  (use_es ? "es-winters" : "rusanov") + "): "
                  "inlet_flux_diff_max=" + FormatSci(diag_max.inlet_flux_diff) +
                  ", outlet_flux_diff_max=" + FormatSci(diag_max.outlet_flux_diff) +
                  ", inlet_flux_norm_max=" + FormatSci(diag_max.inlet_flux_norm) +
                  ", outlet_flux_norm_max=" + FormatSci(diag_max.outlet_flux_norm));
  }

  return ctx.failures;
}

} // namespace dg_igr_tests
