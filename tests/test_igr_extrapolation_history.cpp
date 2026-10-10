#include "igrlab/compat.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/igr/IGR.hpp"
#include "physics/igr/igr_coupling.hpp"
#include "physics/igr/igr_flux.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

mfem::Mesh PeriodicMesh()
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(4, 1.0);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0})};
  return mfem::Mesh::MakePeriodic(
    mesh, mesh.CreatePeriodicVertexMapping(translations));
}

// A test-only differential equation coupled to the real assembled IGR solve.
// With rho=1 and momentum=a*shape, sigma=a^2*sigma_shape exactly. Choosing
// a'=mean(sigma)/mean(sigma_shape) gives the independent scalar ODE a'=a^2.
// The synthetic telemetry only isolates temporal coupling from DG transport.
class ClosureODE final : public mfem::DGHyperbolicConservationLaws
{
public:
  ClosureODE(mfem::FiniteElementSpace& fes,
             std::unique_ptr<mfem::HyperbolicFormIntegrator> form,
             const mfem::Vector& sigma,
             const mfem::Vector& shape,
             double normalization)
    : mfem::DGHyperbolicConservationLaws(fes, std::move(form), true),
      sigma_(sigma), shape_(shape), normalization_(normalization)
  {
  }

  void Mult(const mfem::Vector&, mfem::Vector& derivative) const override
  {
    double sum = 0.0;
    const double* sigma = sigma_.HostRead();
    for (int i = 0; i < sigma_.Size(); ++i) { sum += sigma[i]; }
    derivative = 0.0;
    double* data = derivative.HostReadWrite();
    for (int i = 0; i < shape_.Size(); ++i)
    {
      data[shape_.Size() + i] = (sum / normalization_) * shape_[i];
    }
    telemetry_.enabled = true;
    telemetry_.sample_count += 8;
  }

  mfem::IGRFrozenSigmaTelemetry GetGlobalFrozenSigmaTelemetry() const override
  {
    return telemetry_;
  }

  mfem::IGRFrozenSigmaTelemetry InspectFrozenSigmaStateForAdmission(
    const mfem::Vector&) const override
  {
    mfem::IGRFrozenSigmaTelemetry fresh;
    fresh.enabled = true;
    fresh.sample_count = 8;
    if (reject_endpoint_)
    {
      fresh.nonpositive_count = 1;
      reject_endpoint_ = false;
    }
    telemetry_.Merge(fresh);
    return fresh;
  }

  void RejectNextEndpoint() { reject_endpoint_ = true; }

private:
  const mfem::Vector& sigma_;
  const mfem::Vector& shape_;
  double normalization_;
  mutable mfem::IGRFrozenSigmaTelemetry telemetry_;
  mutable bool reject_endpoint_ = false;
};

struct Fixture
{
  mfem::Mesh serial = PeriodicMesh();
  mfem::ParMesh mesh{MPI_COMM_WORLD, serial};
  mfem::DG_FECollection collection{2, 1, mfem::BasisType::GaussLegendre};
  mfem::ParFiniteElementSpace state_space{
    &mesh, &collection, 3, mfem::Ordering::byNODES};
  mfem::ParFiniteElementSpace sigma_space{
    &mesh, &collection, 1, mfem::Ordering::byNODES};
  mfem::ParGridFunction state{&state_space};
  mfem::ParGridFunction shape{&sigma_space};
  igrlab::IGRSolver igr{sigma_space, state_space, 1, 3, 0.1};
  mfem::EulerFlux euler_flux{1, 1.4};
  mfem::FluxWithIGR igr_flux{euler_flux};
  mfem::RusanovFluxWithSides numerical_flux{igr_flux};
  mfem::Vector sigma_shape;
  std::unique_ptr<ClosureODE> op;
  std::unique_ptr<mfem::EulerWithIGR> wrapped;

  explicit Fixture(int degree)
  {
    setenv("DG_IGR_IGR_EXTRAPOLATION_DEGREE",
           std::to_string(degree).c_str(), 1);
    mfem::FunctionCoefficient profile([](const mfem::Vector& x)
    {
      return std::sin(2.0 * std::acos(-1.0) * x[0]);
    });
    shape.ProjectCoefficient(profile);
    SetAmplitude(1.0);
    igr.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
    igr.SetIPKind(igrlab::IGRIPKind::SIPG);
    igr.SetSolveMode(igrlab::IGRSolveMode::Krylov);
    igr.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
    igr.SetPreconditioner(igrlab::IGRPrecondType::None);
    igr.SetLinearSolveParams(1e-13, 0.0, 400, 0);
    igr.SetSigmaUpdatePolicy(
      igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep, 1);
    igr.SetStageRtolScale(1.0);
    igr.UpdateFromState(state, false);
    sigma_shape = igr.GetSigma();
    double normalization = 0.0;
    for (int i = 0; i < sigma_shape.Size(); ++i)
    {
      normalization += sigma_shape[i];
    }
    MFEM_VERIFY(normalization > 0.0, "Closure ODE needs a nonzero IGR source.");
    igr_flux.SetSigmaField(&igr.GetSigma());
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    op = std::make_unique<ClosureODE>(state_space, std::move(form),
                                     igr.GetSigma(), shape, normalization);
    op->SetNumericalFlux(&numerical_flux);
    wrapped = std::make_unique<mfem::EulerWithIGR>(*op, igr, state);
    unsetenv("DG_IGR_IGR_EXTRAPOLATION_DEGREE");
  }

  void SetAmplitude(double amplitude)
  {
    double* data = state.HostWrite();
    for (int i = 0; i < shape.Size(); ++i)
    {
      data[i] = 1.0;
      data[shape.Size() + i] = amplitude * shape[i];
      data[2 * shape.Size() + i] = 3.0;
    }
    state.ExchangeFaceNbrData();
  }

  double SigmaError(double scale)
  {
    mfem::Vector error(igr.GetSigma());
    error.Add(-scale, sigma_shape);
    return error.Normlinf() / sigma_shape.Normlinf();
  }

  double Amplitude() const
  {
    const double* data = state.HostRead();
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < shape.Size(); ++i)
    {
      numerator += data[shape.Size() + i] * shape[i];
      denominator += shape[i] * shape[i];
    }
    return numerator / denominator;
  }
};

bool PolynomialHistory(int degree)
{
  Fixture test(degree);
  const auto polynomial = [degree](double t)
  {
    return 1.0 + 0.4 * t + 0.7 * t * t +
           (degree == 3 ? 0.9 * t * t * t : 0.0);
  };
  std::uint64_t generation = 7;
  test.wrapped->PrepareSigmaForOutput(test.state, generation, 0, 0.0, true);
  const int initial_calls = test.igr.GetLinearSolveCalls();
  const std::array<double, 4> intervals = {{0.03, 0.02, 0.04, 0.025}};
  double time = 0.0;
  bool ok = true;
  mfem::Vector derivative(test.state.Size());
  for (int step = 0; step <= degree; ++step)
  {
    const double dt = intervals[step];
    test.wrapped->BeginRKStep(step, time, dt, 10);
    const int before_stages = test.igr.GetLinearSolveCalls();
    for (int stage = 0; stage < 10; ++stage)
    {
      const double stage_time = time +
        mfem::SSPRK104Coefficients::c[stage] * dt;
      if (stage > 0) { ++generation; }
      test.SetAmplitude(std::sqrt(polynomial(stage_time)));
      test.wrapped->SetRKStageState(stage, generation, stage_time);
      test.wrapped->Mult(test.state, derivative);
      ok = ok && test.SigmaError(polynomial(stage_time)) < 2e-11;
    }
    const int stage_calls = test.igr.GetLinearSolveCalls() - before_stages;
    ok = ok && stage_calls == (step < degree ? 9 : 0);
    const double endpoint = time + dt;
    test.SetAmplitude(std::sqrt(polynomial(endpoint)));
    ++generation;
    test.wrapped->EndRKStep(test.state, endpoint, generation);
    time = endpoint;
  }
  ok = ok && test.igr.GetLinearSolveCalls() - initial_calls ==
               degree * 10 + 1 &&
       test.wrapped->GetSigmaEndpointSolveCount() == degree + 1 &&
       test.wrapped->GetSigmaExtrapolatedStageCount() == 10;

  // Reject a later exact endpoint, restore its accepted anchor, then retry
  // with a different dt. Neither the rejected sigma nor its interval may
  // enter the polynomial history.
  const std::uint64_t accepted_generation = generation;
  const double rejected_dt = 0.03;
  test.wrapped->BeginRKStep(degree + 1, time, rejected_dt, 10);
  test.wrapped->SetRKStageState(0, generation, time);
  test.wrapped->Mult(test.state, derivative);
  test.SetAmplitude(std::sqrt(polynomial(time + rejected_dt)));
  test.op->RejectNextEndpoint();
  bool rejected = false;
  try
  {
    test.wrapped->EndRKStep(test.state, time + rejected_dt, ++generation);
  }
  catch (const mfem::RecoverableRKStageAdmissionFailure&)
  {
    rejected = true;
  }
  test.SetAmplitude(std::sqrt(polynomial(time)));
  generation = accepted_generation;
  test.wrapped->AbortRKStep(test.state, time, generation, 0.012);
  ok = ok && rejected && test.SigmaError(polynomial(time)) < 2e-11;
  test.wrapped->BeginRKStep(degree + 1, time, 0.012, 10);
  test.wrapped->SetRKStageState(0, generation, time);
  test.wrapped->Mult(test.state, derivative);
  const double retry_stage_time = time + 0.008;
  test.SetAmplitude(std::sqrt(polynomial(retry_stage_time)));
  test.wrapped->SetRKStageState(4, ++generation, retry_stage_time);
  test.wrapped->Mult(test.state, derivative);
  ok = ok && test.SigmaError(polynomial(retry_stage_time)) < 2e-11;
  test.SetAmplitude(std::sqrt(polynomial(time)));
  test.wrapped->AbortRKStep(test.state, time, accepted_generation, 0.0);
  std::cout << "history degree=" << degree << " polynomial_startup_rollback="
            << ok << '\n';
  return ok;
}

double CoupledODEError(int degree, int steps)
{
  Fixture test(degree);
  mfem::RK3SSPWithLimiter stepper(test.state_space,
                                 mfem::SSPRKMethod::SSPRK104);
  stepper.Init(*test.wrapped);
  constexpr double final_time = 0.2;
  double time = 0.0;
  for (int step = 0; step < steps; ++step)
  {
    double dt = final_time / steps;
    stepper.Step(test.state, time, dt);
  }
  return std::abs(test.Amplitude() - 1.0 / (1.0 - final_time));
}

bool CoupledOrder(int degree)
{
  std::array<double, 3> errors;
  for (int level = 0; level < 3; ++level)
  {
    errors[level] = CoupledODEError(degree, 20 << level);
  }
  const double order1 = std::log2(errors[0] / errors[1]);
  const double order2 = std::log2(errors[1] / errors[2]);
  std::cout << "history degree=" << degree << " coupled_errors="
            << errors[0] << ',' << errors[1] << ',' << errors[2]
            << " observed_orders=" << order1 << ',' << order2 << '\n';
  return std::isfinite(order1) && std::isfinite(order2) &&
         order1 > degree + 0.65 && order2 > degree + 0.75;
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  mfem::Device device("cpu");
  if (mfem::Mpi::WorldSize() != 1) { return 77; }
  unsetenv("DG_IGR_IGR_DENSE_INVERSE");
  unsetenv("DG_IGR_IGR_EXTRAPOLATION_DEGREE");
  bool ok = true;
  for (int degree : {2, 3})
  {
    ok = PolynomialHistory(degree) && ok;
    ok = CoupledOrder(degree) && ok;
  }
  return ok ? 0 : 1;
}
