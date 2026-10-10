#include "igrlab/environment.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "sim/config/case_config.hpp"
#include "sim/solver/solver_helpers.hpp"

#include "mfem.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace
{

class ScalarODE final : public mfem::TimeDependentOperator,
                        public mfem::RKStageAwareOperator,
                        public mfem::PhysicalBoundaryFluxIntegralProvider
{
public:
  ScalarODE(int size, bool nonautonomous, int expected_stages)
    : mfem::TimeDependentOperator(size), nonautonomous_(nonautonomous),
      expected_stages_(expected_stages), flux_(1)
  {
  }

  void BeginRKStep(int, mfem::real_t, mfem::real_t) override {}
  void BeginRKStep(int, mfem::real_t, mfem::real_t, int stages) override
  {
    ok = ok && stages == expected_stages_ && stages_seen_ == 0;
  }
  void SetRKStage(int stage) override { stage_ = stage; }
  void SetRKStageState(int stage, std::uint64_t generation,
                       mfem::real_t time) override
  {
    stage_ = stage;
    ok = ok && stage == stages_seen_ &&
         (!have_generation_ || generation > last_generation_);
    have_generation_ = true;
    last_generation_ = generation;
    notified_time_ = time;
    ++stages_seen_;
  }
  void EndRKStep(const mfem::Vector&, mfem::real_t,
                  std::uint64_t generation) override
  {
    ok = ok && stages_seen_ == expected_stages_ &&
         generation > last_generation_;
    stages_seen_ = 0;
    ++accepted_steps;
  }
  void Mult(const mfem::Vector& state, mfem::Vector& rhs) const override
  {
    ok = ok && stage_ >= 0 && stage_ < expected_stages_ &&
         std::abs(GetTime() - notified_time_) < 1e-15;
    const double* values = state.HostRead();
    double* result = rhs.HostWrite();
    for (int i = 0; i < state.Size(); ++i)
    {
      result[i] = nonautonomous_ ? (2.0 + 2.0 * GetTime()) * values[i]
                                 : values[i] * values[i];
    }
    if (armed_)
    {
      flux_[0] = 1.0 + GetTime();
      available_ = true;
      armed_ = false;
    }
  }
  bool SupportsPhysicalBoundaryFluxIntegral() const override { return true; }
  void ArmPhysicalBoundaryFluxIntegral() const override
  {
    armed_ = true;
    available_ = false;
  }
  bool HasLastPhysicalBoundaryFluxIntegral() const override
  {
    return available_;
  }
  const mfem::Vector& GetLastPhysicalBoundaryFluxIntegral() const override
  {
    return flux_;
  }

  mutable bool ok = true;
  int accepted_steps = 0;

private:
  bool nonautonomous_;
  int expected_stages_;
  int stages_seen_ = 0;
  int stage_ = -1;
  bool have_generation_ = false;
  std::uint64_t last_generation_ = 0;
  double notified_time_ = 0.0;
  mutable mfem::Vector flux_;
  mutable bool armed_ = false;
  mutable bool available_ = false;
};

double Integrate(mfem::ParFiniteElementSpace& fes, mfem::SSPRKMethod method,
                  bool nonautonomous, int steps, bool& ok)
{
  mfem::RK3SSPWithLimiter stepper(fes, method);
  stepper.LimitInitial(false);
  stepper.EnablePhysicalBoundaryFluxTracking();
  ScalarODE rhs(fes.GetVSize(), nonautonomous, stepper.GetStageCount());
  stepper.Init(rhs);
  mfem::Vector state(fes.GetVSize());
  state = 1.0;
  constexpr double final_time = 0.4;
  double time = 0.0;
  for (int step = 0; step < steps; ++step)
  {
    double dt = final_time / steps;
    stepper.Step(state, time, dt);
  }
  const double exact = nonautonomous
      ? std::exp(2.0 * final_time + final_time * final_time)
                                     : 1.0 / (1.0 - final_time);
  const double signed_flux =
    stepper.GetCumulativePhysicalBoundaryFluxIntegral()[0];
  const double absolute_flux =
    stepper.GetCumulativePhysicalBoundaryFluxAbsoluteIntegral()[0];
  const auto check = [&](bool passed, const char* label)
  {
    if (!passed)
      std::cerr << "research-rk invariant-failed=" << label
                << " method=" << static_cast<int>(method)
                << " nonautonomous=" << nonautonomous
                << " steps=" << steps << " accepted=" << rhs.accepted_steps
                << " rhs_count=" << stepper.GetRHSEvaluationCount()
                << " time=" << time << " signed_flux=" << signed_flux
                << " absolute_flux=" << absolute_flux << '\n';
    ok = passed && ok;
  };
  check(rhs.ok, "stage-callbacks");
  check(rhs.accepted_steps == steps, "accepted-count");
  check(stepper.GetRHSEvaluationCount() == steps * stepper.GetStageCount(),
        "rhs-count");
  check(stepper.GetSSPCoefficient() == 0.0, "no-ssp-coefficient");
  check(std::abs(time - final_time) < 2e-15, "final-time");
  check(std::abs(signed_flux - final_time - 0.5*final_time*final_time) < 2e-14,
        "signed-flux-integral");
  check(absolute_flux >= std::abs(signed_flux) - 2e-14,
        "absolute-flux-bound");
  if (method == mfem::SSPRKMethod::ResearchDOPRI5)
  {
    // The principal formula has a negative output weight. Its absolute flux
    // census must use |b_i|, rather than applying a signed weight to |flux_i|.
    check(absolute_flux > 1.1 * std::abs(signed_flux),
          "negative-output-weight-absolute-flux");
  }
  return std::abs(state[0] - exact);
}

bool CheckMethod(mfem::ParFiniteElementSpace& fes,
                  dg_sim::TimeIntegrator integrator)
{
  const auto method = dg_sim::SSPRKMethodForTimeIntegrator(integrator);
  const int order = mfem::SSPRKOrder(method);
  bool ok = true;
  for (bool nonautonomous : {false, true})
  {
    // y'=(2+2t)y keeps stage-time dependence and a visible leading error.
    // Unlike y'=2ty at t=0, this window avoids coarse-level cancellation in
    // DOPRI5; all finest errors remain well above FP64 rounding error.
    std::array<double, 3> errors;
    for (int level = 0; level < 3; ++level)
    {
      errors[level] = Integrate(fes, method, nonautonomous, 4 << level, ok);
    }
    const double order1 = std::log2(errors[0] / errors[1]);
    const double order2 = std::log2(errors[1] / errors[2]);
    const bool order_ok = std::isfinite(order1) && std::isfinite(order2) &&
         order1 > order - 0.25 && order2 > order - 0.15;
    ok = order_ok && ok;
    std::cout << "research-rk method="
              << dg_sim::TimeIntegratorToString(integrator)
              << " nonautonomous=" << nonautonomous << " errors="
              << errors[0] << ',' << errors[1] << ',' << errors[2]
              << " orders=" << order1 << ',' << order2
              << " order_pass=" << order_ok << '\n';
    if (!order_ok)
      std::cerr << "research-rk order-failed method="
                << dg_sim::TimeIntegratorToString(integrator)
                << " nonautonomous=" << nonautonomous
                << " required_minima=" << order - 0.25 << ',' << order - 0.15
                << '\n';
  }
  dg_sim::RunConfig cfg;
  cfg.time_integrator = integrator;
  cfg.regularization = dg_sim::Regularization::IGR;
  cfg.igr_opts.sigma_cadence = "every-stage";
  cfg.opts.cfl = 0.0;
  cfg.opts.dt = 0.01;
  cfg.case_config.limiter_pipeline.zhang_shu = false;
  ok = ok && !dg_sim::SSPCoefficientForTimeIntegrator(integrator) &&
       dg_sim::ValidateExplicitSSPRKCFLSelection(cfg).empty();
  cfg.opts.cfl = 0.1;
  ok = ok && !dg_sim::ValidateExplicitSSPRKCFLSelection(cfg).empty();
  cfg.opts.cfl = 0.0;
  cfg.case_config.limiter_pipeline.zhang_shu = true;
  ok = ok && !dg_sim::ValidateExplicitSSPRKCFLSelection(cfg).empty();
  cfg.case_config.limiter_pipeline.zhang_shu = false;
  cfg.igr_opts.sigma_cadence = "every-step";
  ok = ok && !dg_sim::ValidateExplicitSSPRKCFLSelection(cfg).empty();
  return ok;
}

bool CheckFrozenSigmaReference()
{
  constexpr const char* name = "DG_IGR_IGR_FROZEN_SIGMA_REFERENCE";
  const char* previous = igrlab::GetEnv(name);
  const bool had_previous = previous != nullptr;
  const std::string previous_value = previous ? previous : "";
  unsetenv(name);
  dg_sim::RunConfig cfg;
  cfg.regularization = dg_sim::Regularization::IGR;
  cfg.discretization = dg_sim::Discretization::DGSEM;
  cfg.igr_opts.sigma_cadence = "every-stage";
  bool ok = !dg_sim::UsesFrozenSigmaSpectralSpeed(cfg);
  cfg.igr_opts.sigma_cadence = "extrapolated-every-step";
  ok = ok && dg_sim::UsesFrozenSigmaSpectralSpeed(cfg);
  cfg.igr_opts.sigma_cadence = "every-stage";
  setenv(name, "1", 1);
  ok = ok && dg_sim::UsesFrozenSigmaSpectralSpeed(cfg);
  const auto rejected = [&]()
  {
    try { (void)dg_sim::UsesFrozenSigmaSpectralSpeed(cfg); }
    catch (const std::invalid_argument&) { return true; }
    return false;
  };
  cfg.igr_opts.sigma_cadence = "every-step";
  ok = rejected() && ok;
  cfg.igr_opts.sigma_cadence = "extrapolated-every-step";
  ok = rejected() && ok;
  cfg.igr_opts.sigma_cadence = "every-stage";
  cfg.regularization = dg_sim::Regularization::None;
  ok = rejected() && ok;
  cfg.regularization = dg_sim::Regularization::IGR;
  cfg.discretization = dg_sim::Discretization::DG;
  ok = rejected() && ok;
  cfg.discretization = dg_sim::Discretization::DGSEM;
  for (const char* value : {"0", "", "true", "01"})
  {
    setenv(name, value, 1);
    ok = rejected() && ok;
  }
  if (had_previous) { setenv(name, previous_value.c_str(), 1); }
  else { unsetenv(name); }
  return ok;
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  mfem::Device device("cpu");
  if (mfem::Mpi::WorldSize() != 1) { return 77; }
  mfem::Mesh serial = mfem::Mesh::MakeCartesian1D(2, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::DG_FECollection collection(1, 1);
  mfem::ParFiniteElementSpace fes(&mesh, &collection);
  std::cout << std::setprecision(16);
  bool ok = CheckFrozenSigmaReference();
  for (auto method : {dg_sim::TimeIntegrator::ResearchRK4,
                      dg_sim::TimeIntegrator::ResearchDOPRI5,
                      dg_sim::TimeIntegrator::ResearchRadauPicard5})
  {
    ok = CheckMethod(fes, method) && ok;
  }
  return ok ? 0 : 1;
}
