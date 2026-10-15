#include "test_dgsem_reference_convergence.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

// Test-only observables. No time-loop or solver API is added for this gate.
namespace
{
using dg_igr_tests::TestContext;
using dg_igr_tests::FormatSci;
using dg_igr_tests::dgsem_convergence_detail::EvaluatePeriodicContact;
using dg_igr_tests::dgsem_convergence_detail::MakeReferenceConvergenceMesh;

static_assert(std::is_same<mfem::real_t, double>::value,
              "These numerical acceptance tolerances require FP64 MFEM.");
constexpr int dimension = 2;
constexpr int equations = 4;
constexpr double gamma = 1.4;
constexpr double horizon = 0.01;
constexpr int smoke_steps = 20;
// Frozen before execution. The conservation budget covers FP64 element/face
// arithmetic and 60 RK stages; independent quadrature accumulation uses wider
// host accumulators. It is tighter than the existing 5e-10 DGSEM total gate.
constexpr double conservation_absolute = 1.0e-12;
constexpr double conservation_relative = 2.0e-11;
constexpr double parity_absolute = 5.0e-11;
constexpr double parity_relative = 2.0e-10;

class StrictProductionScope
{
public:
  StrictProductionScope()
  {
    if (const char* value = std::getenv("DG_IGR_DGSEM_PRODUCTION_STRICT"))
      previous_ = value;
    MFEM_VERIFY(setenv("DG_IGR_DGSEM_PRODUCTION_STRICT", "1", 1) == 0,
                "Cannot select strict production DGSEM for numerical acceptance.");
  }
  ~StrictProductionScope()
  {
    if (previous_)
      setenv("DG_IGR_DGSEM_PRODUCTION_STRICT", previous_->c_str(), 1);
    else
      unsetenv("DG_IGR_DGSEM_PRODUCTION_STRICT");
  }
private:
  std::optional<std::string> previous_;
};

bool Near(double left, double right, double absolute, double relative)
{
  return std::isfinite(left) && std::isfinite(right) &&
         std::abs(left - right) <=
           absolute + relative * std::max(std::abs(left), std::abs(right));
}

struct Sample
{
  std::array<double, equations> totals{};
  std::vector<double> coefficients;
  std::vector<double> quadrature_values;
  std::vector<double> weights;
  double exact_error = std::numeric_limits<double>::infinity();
  bool valid = false;
};

Sample Measure(mfem::ParGridFunction& field, double time)
{
  Sample result;
  const double* coefficients = field.HostRead();
  result.coefficients.assign(coefficients, coefficients + field.Size());
  if (!std::all_of(result.coefficients.begin(), result.coefficients.end(),
                   [](double value) { return std::isfinite(value); }))
    return result;
  std::array<long double, equations> totals{};
  long double error_squared = 0.0L;
  auto* space = field.ParFESpace();
  for (int element = 0; element < space->GetNE(); ++element)
  {
    const auto* finite_element = space->GetFE(element);
    auto* transform = space->GetElementTransformation(element);
    // Independent Gauss quadrature, rather than the solver's mass/reduction
    // workspaces. It exactly integrates polynomial conserved fields on this
    // affine mesh; the extra points also resolve the smooth exact solution.
    const auto& rule = mfem::IntRules.Get(finite_element->GetGeomType(),
                                         2 * finite_element->GetOrder() + 8);
    for (int point = 0; point < rule.GetNPoints(); ++point)
    {
      const auto& integration = rule.IntPoint(point);
      transform->SetIntPoint(&integration);
      mfem::Vector value(equations), exact(equations), position(dimension);
      field.GetVectorValue(element, integration, value);
      transform->Transform(integration, position);
      EvaluatePeriodicContact(position, time, exact);
      const double weight = integration.weight * transform->Weight();
      if (!(weight > 0.0) || !std::isfinite(weight)) return result;
      const double density = value[0];
      if (!(density > 0.0) || !std::isfinite(density)) return result;
      const double pressure = (gamma - 1.0) *
        (value[3] - 0.5 * (value[1] * value[1] + value[2] * value[2]) /
                      density);
      if (!(pressure > 0.0) || !std::isfinite(pressure)) return result;
      result.weights.push_back(weight);
      for (int component = 0; component < equations; ++component)
      {
        if (!std::isfinite(value[component]) ||
            !std::isfinite(exact[component])) return result;
        totals[component] += static_cast<long double>(weight) * value[component];
        const long double difference =
          static_cast<long double>(value[component]) - exact[component];
        error_squared += static_cast<long double>(weight) * difference * difference;
        result.quadrature_values.push_back(value[component]);
      }
    }
  }
  for (int component = 0; component < equations; ++component)
    result.totals[component] = static_cast<double>(totals[component]);
  result.exact_error = std::sqrt(static_cast<double>(error_squared));
  result.valid = std::isfinite(result.exact_error) &&
    std::all_of(result.totals.begin(), result.totals.end(),
                 [](double value) { return std::isfinite(value); });
  return result;
}

double DifferenceL2(const Sample& left, const Sample& right)
{
  if (!left.valid || !right.valid || left.weights != right.weights ||
      left.quadrature_values.size() != right.quadrature_values.size() ||
      left.quadrature_values.size() != left.weights.size() * equations)
    return std::numeric_limits<double>::infinity();
  long double squared = 0.0L;
  for (std::size_t point = 0; point < left.weights.size(); ++point)
    for (int component = 0; component < equations; ++component)
    {
      const auto index = point * equations + component;
      const long double difference =
        static_cast<long double>(left.quadrature_values[index]) -
        right.quadrature_values[index];
      squared += static_cast<long double>(left.weights[point]) *
                 difference * difference;
    }
  return std::sqrt(static_cast<double>(squared));
}

struct Evolution
{
  Sample initial;
  Sample final;
  std::array<double, equations> maximum_drift{};
  bool passed = false;
};

Evolution Evolve(TestContext& context, bool dgsem, int order, int cells,
                 int steps, const std::string& backend)
{
  const std::string description = std::string(dgsem ? "DGSEM" : "DG") +
    " " + backend + " p=" + std::to_string(order) +
    " n=" + std::to_string(cells) + " steps=" + std::to_string(steps);
  Evolution result;
  auto serial = MakeReferenceConvergenceMesh(cells, order + 1, false);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::DG_FECollection collection(order, dimension, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace space(&mesh, &collection, equations,
                                    mfem::Ordering::byNODES);
  mfem::ParGridFunction state(&space);
  mfem::VectorFunctionCoefficient initial_coefficient(
    equations, [](const mfem::Vector& position, mfem::Vector& value)
    { EvaluatePeriodicContact(position, 0.0, value); });
  state.ProjectCoefficient(initial_coefficient);
  result.initial = Measure(state, 0.0);
  context.Check(result.initial.valid, description + " finite admissible initial state");
  if (!result.initial.valid) return result;
  mfem::EulerFlux flux(dimension, gamma);
  mfem::RusanovFluxWithSides numerical_flux(flux);
  auto integrator = std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux, 2);
  mfem::DGHyperbolicConservationLaws oper(space, std::move(integrator), true);
  oper.SetNumericalFlux(&numerical_flux);
  if (dgsem)
  {
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMVolumeDissipation(mfem::DGSEMVolumeDissipation::None);
    oper.SetDGSEMGamma(gamma);
    oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order));
  }
  mfem::RK3SSPWithLimiter rk(space);
  rk.Init(oper);
  mfem::Vector current;
  state.GetTrueDofs(current);
  double time = 0.0;
  bool healthy = true;
  for (int step = 0; step < steps; ++step)
  {
    double dt = horizon / static_cast<double>(steps);
    rk.Step(current, time, dt);
    state.SetFromTrueDofs(current);
    result.final = Measure(state, time);
    healthy = healthy && result.final.valid &&
      Near(time, (step + 1) * horizon / steps,
           64 * std::numeric_limits<double>::epsilon(), 0.0);
    for (int component = 0; component < equations; ++component)
    {
      const double drift = std::abs(result.final.totals[component] -
                                     result.initial.totals[component]);
      result.maximum_drift[component] =
        std::max(result.maximum_drift[component], drift);
      healthy = healthy && Near(result.final.totals[component],
                                 result.initial.totals[component],
                                 conservation_absolute, conservation_relative);
    }
    if (!healthy) break;
  }
  const bool work = rk.GetStepCount() == steps &&
                    rk.GetRHSEvaluationCount() == 3LL * steps;
  const bool horizon_reached = Near(time, horizon,
    64 * std::numeric_limits<double>::epsilon(), 0.0);
  bool production = true;
  if (dgsem)
  {
    const auto selection = oper.GetDGSEMProductionSelection();
    const auto bridge = oper.GetDGSEMProductionBridgeEvidence();
    production = selection.supported && selection.strict &&
      selection.cuda == (backend == "cuda") &&
      selection.reason == mfem::dgsem::DGSEMProductionReason::None &&
      bridge.production_mult_calls == 3LL * steps &&
      bridge.fallback_mult_calls == 0;
  }
  context.Check(healthy && work && horizon_reached && production,
                description + " horizon, work, production route and all conserved totals");
  const double change = DifferenceL2(result.initial, result.final);
  context.Check(std::isfinite(change) && change > 1.0e-8,
                description + " evolves a nonconstant state");
  result.passed = healthy && work && horizon_reached && production &&
                  std::isfinite(change) && change > 1.0e-8;
  std::string metric = description + " final_time=" + FormatSci(time) +
    " initial_projection_L2_error=" + FormatSci(result.initial.exact_error) +
    " exact_L2_error=" + FormatSci(result.final.exact_error) + " max_total_drift=";
  for (double drift : result.maximum_drift) metric += FormatSci(drift) + ",";
  context.LogMetric(metric);
  return result;
}

std::array<Evolution, 4> Smoke(TestContext& context, const std::string& backend)
{
  std::array<Evolution, 4> results;
  int index = 0;
  for (bool dgsem : {false, true})
    for (int order : {1, 3})
      results[index++] = Evolve(context, dgsem, order, 4, smoke_steps, backend);
  return results;
}

void CompareBackends(TestContext& context, const std::array<Evolution, 4>& cpu,
                     const std::array<Evolution, 4>& cuda)
{
  for (std::size_t index = 0; index < cpu.size(); ++index)
  {
    const Sample& host = cpu[index].final;
    const Sample& device = cuda[index].final;
    bool same = cpu[index].passed && cuda[index].passed &&
                host.coefficients.size() == device.coefficients.size();
    if (same)
      for (std::size_t dof = 0; dof < host.coefficients.size(); ++dof)
        same = Near(host.coefficients[dof], device.coefficients[dof],
                     parity_absolute, parity_relative) && same;
    for (int component = 0; component < equations; ++component)
      same = Near(host.totals[component], device.totals[component],
                   conservation_absolute, conservation_relative) && same;
    const double difference = DifferenceL2(host, device);
    context.Check(same && std::isfinite(difference) && difference <=
                    parity_absolute + parity_relative,
                  "production CPU/CUDA final fields and totals case=" +
                    std::to_string(index) + " physical_L2_difference=" +
                    FormatSci(difference));
  }
}

void Convergence(TestContext& context)
{
  const double wave_speed = std::hypot(0.35, -0.2) + std::sqrt(gamma / 0.98);
  for (int order : {1, 3})
  {
    std::array<Evolution, 3> ladder;
    int index = 0;
    int finest_steps = 0;
    for (int cells : {3, 6, 12})
    {
      const double dt_bound = 0.05 / (cells * cells * wave_speed);
      const int steps = static_cast<int>(std::ceil(horizon / dt_bound));
      ladder[index++] = Evolve(context, false, order, cells, steps, "cpu");
      finest_steps = steps;
    }
    for (std::size_t level = 1; level < ladder.size(); ++level)
    {
      const double coarse = ladder[level - 1].final.exact_error;
      const double fine = ladder[level].final.exact_error;
      const double rate = std::log(coarse / fine) / std::log(2.0);
      const double floor = static_cast<double>(order) + 0.25;
      context.Check(ladder[level - 1].passed && ladder[level].passed &&
                      std::isfinite(rate) && fine > 0.0 && fine < coarse && rate > floor,
                    "DG p=" + std::to_string(order) + " spatial level=" +
                      std::to_string(level) + " observed_order=" + FormatSci(rate) +
                      " required_floor=" + FormatSci(floor));
    }
    const Evolution half_dt = Evolve(context, false, order, 12,
                                      2 * finest_steps, "cpu");
    const double time_difference = DifferenceL2(ladder.back().final, half_dt.final);
    context.Check(half_dt.passed && std::isfinite(time_difference) &&
                    time_difference <= 0.1 * ladder.back().final.exact_error,
                  "DG p=" + std::to_string(order) +
                    " finest dt/2 effect <=10% of exact spatial error (difference=" +
                    FormatSci(time_difference) + ")");
  }
}
} // namespace

int main(int argc, char** argv)
{
  // Parse before initializing MPI/device so malformed selections cannot launch
  // a numerical workload. PACE's canonical target launcher supplies rank one.
  if (argc != 3 ||
      (std::string(argv[1]) != "--conservation" &&
       std::string(argv[1]) != "--backend-parity" &&
       std::string(argv[1]) != "--convergence") ||
      (std::string(argv[2]) != "cpu" && std::string(argv[2]) != "cuda") ||
      (std::string(argv[1]) == "--backend-parity" && std::string(argv[2]) != "cuda") ||
      (std::string(argv[1]) == "--convergence" && std::string(argv[2]) != "cpu"))
  {
    std::cerr << "usage: numerical_evolution --conservation cpu|cuda | "
                 "--backend-parity cuda | --convergence cpu\n";
    return 2;
  }
  const std::string mode(argv[1]), backend(argv[2]);
#ifndef MFEM_USE_CUDA
  if (backend == "cuda")
  {
    std::cerr << "CUDA acceptance requires a CUDA-enabled MFEM build\n";
    return 2;
  }
#endif
  mfem::MPI_Session mpi(argc, argv);
  if (mfem::Mpi::WorldSize() != 1 || mfem::Device::IsConfigured())
  {
    std::cerr << "numerical evolution requires rank one and unconfigured MFEM device\n";
    return 2;
  }
  StrictProductionScope strict;
  mfem::Hypre::Init();
  TestContext context;
  // Negative predicates keep NaN/Inf and shape mismatches from masquerading as
  // agreement, a risk with a plain abs(a-b)>tol comparison.
  context.Check(!Near(std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0, 1.0) &&
                  !Near(1.0, std::numeric_limits<double>::infinity(), 1.0, 1.0),
                "numerical agreement rejects nonfinite inputs");
  if (mode == "--convergence") Convergence(context);
  else if (mode == "--backend-parity")
  {
    // Destroy every CPU mesh/operator/vector before configuring CUDA once.
    // Retain plain host observations, as in the existing HDG alignment test.
    const auto reference = Smoke(context, "cpu");
    if (context.failures != 0) return 1;
    mfem::Device device("cuda");
    context.Check(mfem::Device::Allows(mfem::Backend::CUDA_MASK),
                  "CPU/CUDA agreement selects actual CUDA execution");
    const auto actual = Smoke(context, "cuda");
    CompareBackends(context, reference, actual);
  }
  else
  {
    mfem::Device device(backend.c_str());
    context.Check((backend == "cuda") == mfem::Device::Allows(mfem::Backend::CUDA_MASK),
                  "conservation selects requested backend");
    Smoke(context, backend);
  }
  return context.failures == 0 ? 0 : 1;
}
