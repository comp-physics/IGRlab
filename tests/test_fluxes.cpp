#include "numerics/riemann/diffusive_rusanov.hpp"
#include "numerics/riemann/hllc.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/riemann/side_scope_utils.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include "mfem.hpp"
#include "test_utils.hpp"

namespace dg_igr_tests
{

namespace
{

class AdvectionFlux final : public mfem::FluxFunction
{
public:
  explicit AdvectionFlux(const mfem::Vector& a)
    : mfem::FluxFunction(/*num_eq=*/1, /*dim=*/a.Size()), a_(a)
  {}

  mfem::real_t ComputeFlux(const mfem::Vector& state,
                           mfem::ElementTransformation&,
                           mfem::DenseMatrix& flux) const override
  {
    flux.SetSize(num_equations, dim);
    for (int d = 0; d < dim; ++d)
    {
      flux(0, d) = a_(d) * state(0);
    }
    return a_.Norml2();
  }

  mfem::real_t ComputeFluxDotN(const mfem::Vector& state,
                               const mfem::Vector& normal,
                               mfem::FaceElementTransformations&,
                               mfem::Vector& fluxN) const override
  {
    fluxN.SetSize(num_equations);

    double adv = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      adv += a_(d) * normal(d);
    }

    fluxN(0) = adv * state(0);

    const double nmag = normal.Norml2();
    return (nmag > 0.0) ? std::abs(adv) / nmag : 0.0;
  }

private:
  mfem::Vector a_;
};

class SideAwareAdvectionFlux final : public mfem::FluxFunction,
                                     public mfem::FaceSideExplicitFlux
{
public:
  explicit SideAwareAdvectionFlux(const mfem::Vector& a)
    : mfem::FluxFunction(/*num_eq=*/1, /*dim=*/a.Size()), a_(a)
  {}

  mfem::real_t ComputeFlux(const mfem::Vector& state,
                           mfem::ElementTransformation&,
                           mfem::DenseMatrix& flux) const override
  {
    flux.SetSize(num_equations, dim);
    for (int d = 0; d < dim; ++d)
    {
      flux(0, d) = a_(d) * state(0);
    }
    return a_.Norml2();
  }

  mfem::real_t ComputeFluxDotN(const mfem::Vector& state,
                               const mfem::Vector& normal,
                               mfem::FaceElementTransformations&,
                               mfem::Vector& fluxN) const override
  {
    fluxN.SetSize(num_equations);

    double adv = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      adv += a_(d) * normal(d);
    }

    fluxN(0) = adv * state(0);
    const double nmag = normal.Norml2();
    return (nmag > 0.0) ? std::abs(adv) / nmag : 0.0;
  }

  mfem::real_t ComputeFluxDotNAtSide(const mfem::Vector& state,
                                     const mfem::Vector& normal,
                                     mfem::FaceElementTransformations& Tr,
                                     mfem::FaceSide,
                                     mfem::Vector& fluxDotN) const override
  {
    return ComputeFluxDotN(state, normal, Tr, fluxDotN);
  }

  mfem::real_t ComputeAvgFluxDotNAtSide(const mfem::Vector& state1,
                                        const mfem::Vector& state2,
                                        const mfem::Vector& normal,
                                        mfem::FaceElementTransformations& Tr,
                                        mfem::FaceSide,
                                        mfem::Vector& fluxDotN) const override
  {
    fluxDotN.SetSize(num_equations);
    mfem::Vector avg(state1.Size());
    avg = 0.5;
    avg *= state1;
    avg.Add(0.5, state2);
    return ComputeFluxDotN(avg, normal, Tr, fluxDotN);
  }

private:
  mfem::Vector a_;
};

class HLLCFluxTestAdapter final : public mfem::HLLCFlux
{
public:
  using mfem::HLLCFlux::HLLCFlux;

  void ComputePrimitive(const mfem::Vector& U,
                        const mfem::Vector& nhat,
                        mfem::Vector& prim,
                        double& un,
                        double& a) const
  {
    Prim W;
    ConsToPrim(U, nhat, W);
    const int dim = W.u.Size();
    prim.SetSize(dim + 2);
    prim(0) = W.rho;
    for (int d = 0; d < dim; ++d)
    {
      prim(1 + d) = W.u(d);
    }
    prim(1 + dim) = W.p;
    un = W.un;
    a = W.a;
  }
};

void ComputeFluxDotN(const mfem::FluxFunction& flux_fn,
                     const mfem::Vector& state,
                     const mfem::Vector& normal,
                     mfem::Vector& flux_out)
{
  mfem::FaceElementTransformations Tr;
  flux_fn.ComputeFluxDotN(state, normal, Tr, flux_out);
}

double AnalyticDiffusiveFlux(double adv,
                             double nmag,
                             double uL,
                             double uR,
                             double beta,
                             double delta,
                             double cap)
{
  const double lambda = (nmag > 0.0) ? std::abs(adv) / nmag : 0.0;
  double lam_eff = beta * lambda + delta;
  if (lam_eff > cap)
  {
    lam_eff = cap;
  }
  const double scaled = lam_eff * nmag;
  return 0.5 * (adv * (uL + uR) + scaled * (uL - uR));
}

void HLLCConsToPrimReference(const mfem::Vector& U,
                             const mfem::Vector& nhat,
                             double gamma,
                             mfem::Vector& prim,
                             double& un,
                             double& a)
{
  const int dim = U.Size() - 2;
  prim.SetSize(dim + 2);
  mfem::EulerConservativeToPrimitive(U, dim, gamma, prim);
  un = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    un += prim(1 + d) * nhat(d);
  }
  const double rho = prim(0);
  const double p = prim(1 + dim);
  a = std::sqrt(std::max(0.0, gamma * p / rho));
}

template <typename EvalFn>
double TimeEvalSeconds(EvalFn&& fn, int iters, double& sink)
{
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    sink += fn(i);
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

int RunFluxTests(TestContext& ctx)
{
  const double tol = 1e-12;

  // --- Rusanov upwinding on scalar advection ---
  mfem::Vector a(2);
  a = 0.0;
  a(0) = 1.0;

  AdvectionFlux flux(a);
  mfem::RusanovFluxWithSides num_flux(flux);

  mfem::Vector uL(1);
  mfem::Vector uR(1);
  mfem::Vector normal(2);
  mfem::Vector flux_out(1);
  mfem::FaceElementTransformations Tr;

  uL(0) = 2.0;
  uR(0) = 5.0;

  // Positive advection speed selects the left state.
  normal = 0.0;
  normal(0) = 2.0;
  num_flux.Eval(uL, uR, normal, Tr, flux_out);
  ctx.AddParity(flux_out(0));
  ctx.CheckNear(flux_out(0), 2.0 * uL(0), tol,
                "Rusanov advection upwind with positive normal");

  // Negative advection speed selects the right state.
  normal(0) = -2.0;
  num_flux.Eval(uL, uR, normal, Tr, flux_out);
  ctx.AddParity(flux_out(0));
  ctx.CheckNear(flux_out(0), -2.0 * uR(0), tol,
                "Rusanov advection upwind with negative normal");

  // --- Diffusive Rusanov reduces to Rusanov when parameters are neutral ---
  mfem::DiffusiveRusanovFlux diff_neutral(
      flux, /*beta=*/1.0, /*delta=*/0.0,
      /*cap=*/std::numeric_limits<mfem::real_t>::infinity(),
      /*sensor_k=*/0.0, /*sensor_max=*/1.0);

  uL(0) = 1.5;
  uR(0) = -0.25;
  normal(0) = 3.0;
  normal(1) = -1.0;

  mfem::Vector flux_rusanov(1);
  mfem::Vector flux_diff(1);
  num_flux.Eval(uL, uR, normal, Tr, flux_rusanov);
  diff_neutral.Eval(uL, uR, normal, Tr, flux_diff);
  ctx.AddParity(flux_diff(0) + flux_rusanov(0));
  ctx.CheckNear(flux_diff(0), flux_rusanov(0), tol,
                "Diffusive Rusanov matches Rusanov for neutral parameters");

  // --- Diffusive Rusanov matches analytic scalar-advection form ---
  const double adv = mfem::InnerProduct(a, normal);
  const double adv_abs = std::abs(adv);
  const double beta = 2.0;
  mfem::DiffusiveRusanovFlux diff_beta(flux, beta, /*delta=*/0.0);
  diff_beta.Eval(uL, uR, normal, Tr, flux_diff);
  ctx.AddParity(flux_diff(0));
  const double expected =
      0.5 * (beta * adv_abs * (uL(0) - uR(0)) + adv * (uL(0) + uR(0)));
  ctx.CheckNear(flux_diff(0), expected, tol,
                "Diffusive Rusanov matches analytic scalar-advection flux");

  // --- Diffusive Rusanov parameter effects ---
  uL(0) = 2.0;
  uR(0) = 0.25;
  normal = 0.0;
  normal(0) = 2.0;  // adv = 2, |n| = 2, lambda = 1

  const double adv_base = mfem::InnerProduct(a, normal);
  const double nmag_base = normal.Norml2();

  const double flux_beta1_expected = AnalyticDiffusiveFlux(
      adv_base, nmag_base, uL(0), uR(0), /*beta=*/1.0, /*delta=*/0.0,
      /*cap=*/std::numeric_limits<double>::infinity());
  const double flux_beta2_expected = AnalyticDiffusiveFlux(
      adv_base, nmag_base, uL(0), uR(0), /*beta=*/2.0, /*delta=*/0.0,
      /*cap=*/std::numeric_limits<double>::infinity());

  mfem::DiffusiveRusanovFlux diff_beta1(flux, /*beta=*/1.0, /*delta=*/0.0);
  mfem::DiffusiveRusanovFlux diff_beta2(flux, /*beta=*/2.0, /*delta=*/0.0);
  mfem::Vector flux_beta1(1);
  mfem::Vector flux_beta2(1);
  diff_beta1.Eval(uL, uR, normal, Tr, flux_beta1);
  diff_beta2.Eval(uL, uR, normal, Tr, flux_beta2);
  ctx.AddParity(flux_beta1(0) + flux_beta2(0));
  ctx.CheckNear(flux_beta1(0), flux_beta1_expected, tol,
                "Diffusive Rusanov beta=1 matches analytic flux");
  ctx.CheckNear(flux_beta2(0), flux_beta2_expected, tol,
                "Diffusive Rusanov beta=2 matches analytic flux");
  ctx.Check(flux_beta2(0) > flux_beta1(0),
            "Diffusive Rusanov beta>1 increases dissipation for fixed jump");

  // Delta acts as a floor when advective speeds are small.
  normal = 0.0;
  normal(1) = 3.0;  // adv = 0, lambda = 0
  const double adv_orth = mfem::InnerProduct(a, normal);
  const double nmag_orth = normal.Norml2();
  const double delta = 0.3;
  const double flux_delta_expected = AnalyticDiffusiveFlux(
      adv_orth, nmag_orth, uL(0), uR(0), /*beta=*/1.0, delta,
      /*cap=*/std::numeric_limits<double>::infinity());
  mfem::DiffusiveRusanovFlux diff_delta(flux, /*beta=*/1.0, delta);
  mfem::Vector flux_delta(1);
  diff_delta.Eval(uL, uR, normal, Tr, flux_delta);
  ctx.AddParity(flux_delta(0));
  ctx.CheckNear(flux_delta(0), flux_delta_expected, tol,
                "Diffusive Rusanov delta provides a diffusion floor");
  ctx.Check(std::abs(flux_delta(0)) > 0.0,
            "Diffusive Rusanov delta yields non-zero flux when adv=0 and jump!=0");

  // Cap limits the effective lambda.
  normal = 0.0;
  normal(0) = 4.0;  // adv = 4, lambda = 1
  const double adv_cap = mfem::InnerProduct(a, normal);
  const double nmag_cap = normal.Norml2();
  const double cap = 1.5;
  const double flux_capped_expected = AnalyticDiffusiveFlux(
      adv_cap, nmag_cap, uL(0), uR(0), /*beta=*/10.0, /*delta=*/2.0, cap);
  const double flux_uncapped_expected = AnalyticDiffusiveFlux(
      adv_cap, nmag_cap, uL(0), uR(0), /*beta=*/10.0, /*delta=*/2.0,
      /*cap=*/std::numeric_limits<double>::infinity());
  mfem::DiffusiveRusanovFlux diff_capped(flux, /*beta=*/10.0, /*delta=*/2.0, cap);
  mfem::DiffusiveRusanovFlux diff_uncapped(
      flux, /*beta=*/10.0, /*delta=*/2.0,
      std::numeric_limits<mfem::real_t>::infinity());
  mfem::Vector flux_capped(1);
  mfem::Vector flux_uncapped(1);
  diff_capped.Eval(uL, uR, normal, Tr, flux_capped);
  diff_uncapped.Eval(uL, uR, normal, Tr, flux_uncapped);
  ctx.AddParity(flux_capped(0) + flux_uncapped(0));
  ctx.CheckNear(flux_capped(0), flux_capped_expected, tol,
                "Diffusive Rusanov cap matches capped analytic flux");
  ctx.CheckNear(flux_uncapped(0), flux_uncapped_expected, tol,
                "Diffusive Rusanov uncapped matches analytic flux");
  ctx.Check(flux_capped(0) < flux_uncapped(0),
            "Diffusive Rusanov cap reduces dissipation compared to uncapped");

  // --- HLLC equal-state consistency and supersonic upwinding ---
  const double gamma = 1.4;
  mfem::EulerFlux euler_flux(/*dim=*/1, gamma);
  mfem::HLLCFlux hllc_flux(euler_flux, gamma);

  mfem::Vector normal_1d(1);
  normal_1d(0) = 2.5;

  mfem::Vector cons_eq = EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/2.0,
                                             /*p=*/1.0, gamma);
  mfem::Vector flux_base(3);
  mfem::Vector flux_hllc(3);
  ComputeFluxDotN(euler_flux, cons_eq, normal_1d, flux_base);
  hllc_flux.Eval(cons_eq, cons_eq, normal_1d, Tr, flux_hllc);
  ctx.AddParity(flux_hllc(0) + flux_hllc(1) + flux_hllc(2));
  for (int i = 0; i < flux_base.Size(); ++i)
  {
    ctx.CheckNear(flux_hllc(i), flux_base(i), 1e-11,
                  "HLLC matches physical flux for equal states component " +
                      std::to_string(i));
  }

  // Supersonic positive flow should select the left flux.
  mfem::Vector cons_L = EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/8.0,
                                            /*p=*/1.0, gamma);
  mfem::Vector cons_R = EulerConsFromPrim1D(/*rho=*/0.5, /*u=*/1.0,
                                            /*p=*/0.5, gamma);
  ComputeFluxDotN(euler_flux, cons_L, normal_1d, flux_base);
  hllc_flux.Eval(cons_L, cons_R, normal_1d, Tr, flux_hllc);
  ctx.AddParity(flux_hllc(0) + flux_hllc(1) + flux_hllc(2));
  for (int i = 0; i < flux_base.Size(); ++i)
  {
    ctx.CheckNear(flux_hllc(i), flux_base(i), 1e-9,
                  "HLLC supersonic positive upwind component " + std::to_string(i));
  }

  // Supersonic negative flow should select the right flux.
  cons_L = EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/-8.0, /*p=*/1.0, gamma);
  cons_R = EulerConsFromPrim1D(/*rho=*/0.8, /*u=*/-6.5, /*p=*/0.8, gamma);
  ComputeFluxDotN(euler_flux, cons_R, normal_1d, flux_base);
  hllc_flux.Eval(cons_L, cons_R, normal_1d, Tr, flux_hllc);
  ctx.AddParity(flux_hllc(0) + flux_hllc(1) + flux_hllc(2));
  for (int i = 0; i < flux_base.Size(); ++i)
  {
    ctx.CheckNear(flux_hllc(i), flux_base(i), 1e-9,
                  "HLLC supersonic negative upwind component " + std::to_string(i));
  }

  // --- HLLC consistency in 2D normals ---
  mfem::EulerFlux euler_flux_2d(/*dim=*/2, gamma);
  mfem::HLLCFlux hllc_flux_2d(euler_flux_2d, gamma);

  const mfem::Vector cons_eq_2d =
      EulerConsFromPrim2D(/*rho=*/1.2, /*u=*/0.7, /*v=*/-0.3, /*p=*/1.0, gamma);
  mfem::Vector flux_base_2d(4);
  mfem::Vector flux_hllc_2d(4);

  mfem::Vector normal_a(2);
  normal_a(0) = 2.0;
  normal_a(1) = -1.0;
  ComputeFluxDotN(euler_flux_2d, cons_eq_2d, normal_a, flux_base_2d);
  hllc_flux_2d.Eval(cons_eq_2d, cons_eq_2d, normal_a, Tr, flux_hllc_2d);
  for (int i = 0; i < flux_base_2d.Size(); ++i)
  {
    ctx.CheckNear(flux_hllc_2d(i), flux_base_2d(i), 5e-11,
                  "HLLC 2D equal-state matches physical flux (normal A) component " +
                      std::to_string(i));
  }

  mfem::Vector normal_b(2);
  normal_b(0) = -1.5;
  normal_b(1) = 3.0;
  ComputeFluxDotN(euler_flux_2d, cons_eq_2d, normal_b, flux_base_2d);
  hllc_flux_2d.Eval(cons_eq_2d, cons_eq_2d, normal_b, Tr, flux_hllc_2d);
  for (int i = 0; i < flux_base_2d.Size(); ++i)
  {
    ctx.CheckNear(flux_hllc_2d(i), flux_base_2d(i), 5e-11,
                  "HLLC 2D equal-state matches physical flux (normal B) component " +
                      std::to_string(i));
  }

  // Rotation invariance: rotate velocity and normal together.
  mfem::Vector normal_2d(2);
  normal_2d(0) = 1.5;
  normal_2d(1) = -0.75;
  const double angle = 0.6;

  const mfem::Vector flux_orig = [&]() {
    mfem::Vector out(4);
    hllc_flux_2d.Eval(cons_eq_2d, cons_eq_2d, normal_2d, Tr, out);
    return out;
  }();

  mfem::Vector vel_2d(2);
  vel_2d(0) = 0.7;
  vel_2d(1) = -0.3;
  const mfem::Vector vel_rot = Rotate2D(vel_2d, angle);
  const mfem::Vector normal_rot = Rotate2D(normal_2d, angle);
  const mfem::Vector cons_rot =
      EulerConsFromPrim2D(/*rho=*/1.2, vel_rot(0), vel_rot(1), /*p=*/1.0, gamma);
  mfem::Vector flux_rot(4);
  hllc_flux_2d.Eval(cons_rot, cons_rot, normal_rot, Tr, flux_rot);

  // Mass and energy fluxes are rotationally invariant scalars.
  ctx.CheckNear(flux_rot(0), flux_orig(0), 1e-10,
                "HLLC 2D rotation invariance (mass flux)");
  ctx.CheckNear(flux_rot(3), flux_orig(3), 1e-10,
                "HLLC 2D rotation invariance (energy flux)");

  // Momentum flux vector should rotate with the frame.
  mfem::Vector mom_flux_orig(2);
  mom_flux_orig(0) = flux_orig(1);
  mom_flux_orig(1) = flux_orig(2);
  const mfem::Vector mom_flux_rot_expected = Rotate2D(mom_flux_orig, angle);
  ctx.CheckNear(flux_rot(1), mom_flux_rot_expected(0), 1e-9,
                "HLLC 2D rotation invariance (momentum-x)");
  ctx.CheckNear(flux_rot(2), mom_flux_rot_expected(1), 1e-9,
                "HLLC 2D rotation invariance (momentum-y)");

  // --- Helper-dedup regression: HLLC local ConsToPrim == shared Euler conversions ---
  {
    HLLCFluxTestAdapter hllc_adapter(euler_flux_2d, gamma);
    mfem::Vector nhat(2);
    nhat(0) = 0.6;
    nhat(1) = -0.8;

    const mfem::Vector cons_a =
        EulerConsFromPrim2D(/*rho=*/1.1, /*u=*/2.3, /*v=*/-0.4, /*p=*/0.9, gamma);
    const mfem::Vector cons_b =
        EulerConsFromPrim2D(/*rho=*/0.85, /*u=*/-1.7, /*v=*/0.6, /*p=*/1.4, gamma);

    mfem::Vector prim_local;
    mfem::Vector prim_ref;
    double un_local = 0.0, a_local = 0.0;
    double un_ref = 0.0, a_ref = 0.0;

    hllc_adapter.ComputePrimitive(cons_a, nhat, prim_local, un_local, a_local);
    HLLCConsToPrimReference(cons_a, nhat, gamma, prim_ref, un_ref, a_ref);
    for (int i = 0; i < prim_ref.Size(); ++i)
    {
      ctx.CheckNear(prim_local(i), prim_ref(i), 1e-13,
                    "HLLC ConsToPrim matches shared Euler conversion (state A) component "
                        + std::to_string(i));
    }
    ctx.CheckNear(un_local, un_ref, 1e-13,
                  "HLLC ConsToPrim normal velocity matches reference (state A)");
    ctx.CheckNear(a_local, a_ref, 1e-13,
                  "HLLC ConsToPrim sound speed matches reference (state A)");

    hllc_adapter.ComputePrimitive(cons_b, nhat, prim_local, un_local, a_local);
    HLLCConsToPrimReference(cons_b, nhat, gamma, prim_ref, un_ref, a_ref);
    for (int i = 0; i < prim_ref.Size(); ++i)
    {
      ctx.CheckNear(prim_local(i), prim_ref(i), 1e-13,
                    "HLLC ConsToPrim matches shared Euler conversion (state B) component "
                        + std::to_string(i));
    }
    ctx.CheckNear(un_local, un_ref, 1e-13,
                  "HLLC ConsToPrim normal velocity matches reference (state B)");
    ctx.CheckNear(a_local, a_ref, 1e-13,
                  "HLLC ConsToPrim sound speed matches reference (state B)");
  }

  // --- Micro timing gate: cached explicit-side helper should stay near direct call ---
  if (!EnvEnabled("DG_IGR_SKIP_PERF_GATES"))
  {
    mfem::Vector adv_dir(2);
    adv_dir(0) = 1.0;
    adv_dir(1) = -0.2;
    SideAwareAdvectionFlux side_flux(adv_dir);
    const auto* explicit_flux =
        dynamic_cast<const mfem::FaceSideExplicitFlux*>(&side_flux);
    ctx.Check(explicit_flux != nullptr,
              "Side-aware advection flux exposes explicit-side interface");

    mfem::Vector state_perf(1);
    state_perf(0) = 1.25;
    mfem::Vector normal_perf(2);
    normal_perf(0) = 1.2;
    normal_perf(1) = -0.8;
    mfem::Vector flux_perf(1);
    mfem::FaceElementTransformations Tr_perf;

    constexpr int kIters = 200000;
    double sink = 0.0;

    const double t_direct = TimeEvalSeconds(
        [&](int i) -> double
        {
          const auto side = (i & 1) ? mfem::FaceSide::Left : mfem::FaceSide::Right;
          return explicit_flux->ComputeFluxDotNAtSide(state_perf, normal_perf, Tr_perf,
                                                      side, flux_perf) + flux_perf(0);
        }, kIters, sink);

    const double t_cached = TimeEvalSeconds(
        [&](int i) -> double
        {
          const auto side = (i & 1) ? mfem::FaceSide::Left : mfem::FaceSide::Right;
          return mfem::riemann_detail::ComputeFluxDotNForSide(
              explicit_flux, side_flux, side, state_perf, normal_perf, Tr_perf, flux_perf)
              + flux_perf(0);
        }, kIters, sink);

    const double t_uncached = TimeEvalSeconds(
        [&](int i) -> double
        {
          const auto side = (i & 1) ? mfem::FaceSide::Left : mfem::FaceSide::Right;
          return mfem::riemann_detail::ComputeFluxDotNForSide(
              side_flux, side, state_perf, normal_perf, Tr_perf, flux_perf) + flux_perf(0);
        }, kIters, sink);

    const double cached_ratio = (t_direct > 0.0) ? (t_cached / t_direct) : 0.0;
    const double uncached_ratio = (t_direct > 0.0) ? (t_uncached / t_direct) : 0.0;
    ctx.LogMetric("side_helper_perf direct_s=" + FormatSci(t_direct, 6)
                  + " cached_s=" + FormatSci(t_cached, 6)
                  + " uncached_s=" + FormatSci(t_uncached, 6)
                  + " cached_ratio=" + FormatSci(cached_ratio, 6)
                  + " uncached_ratio=" + FormatSci(uncached_ratio, 6));

    // Relative gate: protects hot-path helper migration without machine-specific
    // absolute timing thresholds.
    ctx.Check(cached_ratio <= 1.30,
              "Cached side helper stays within 30% of direct explicit-side call cost");

    // Keep sink in parity so benchmark loops cannot be optimized out in release.
    ctx.AddParity(sink);
  }

  return ctx.failures;
}

} // namespace dg_igr_tests
