#pragma once

#include "igrlab/environment.hpp"

#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_reference.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "utils/math/constants.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace dg_igr_tests
{
namespace dgsem_convergence_detail
{

struct ConvergenceRow
{
  const char* mesh_kind = nullptr;
  int order = 0;
  int level = 0;
  int elements_per_direction = 0;
  int steps = 0;
  double h = 0.0;
  double dt_base = 0.0;
  double error = 0.0;
  double observed_rate = std::numeric_limits<double>::quiet_NaN();
  double constant_residual_norm = 0.0;
  double expected_rate_floor = 0.0;
};

inline mfem::Mesh MakeReferenceConvergenceMesh(int elements_per_direction,
                                               int geometry_order,
                                               bool curved)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
    elements_per_direction,
    elements_per_direction,
    mfem::Element::QUADRILATERAL,
    /*gen_edges=*/false,
    1.0,
    1.0,
    /*sfc_ordering=*/false);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                            mfem::Vector({0.0, 1.0})};
  mfem::Mesh mesh = mfem::Mesh::MakePeriodic(
    original, original.CreatePeriodicVertexMapping(translations));
  if (curved)
  {
    // Periodic meshes require discontinuous geometry nodes so the identified
    // boundary can retain both physical representatives (x=0 and x=1).
    mesh.SetCurvature(
      std::max(2, geometry_order), true, 2, mfem::Ordering::byNODES);
    mesh.Transform([](const mfem::Vector& input, mfem::Vector& output)
    {
      output.SetSize(2);
      const double two_pi = 2.0 * dg_sim::math::kPi;
      const double warp =
        mfem::dgsem::DGSEMConvergenceContract::curved_map_amplitude *
        std::sin(two_pi * input(0)) * std::sin(two_pi * input(1));
      output(0) = input(0) + warp;
      output(1) = input(1) + 0.5 * warp;
    });
  }
  return mesh;
}

inline void EvaluatePeriodicContact(const mfem::Vector& position,
                                    double time,
                                    mfem::Vector& conservative)
{
  constexpr int dim = 2;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;
  constexpr double velocity_x = 0.35;
  constexpr double velocity_y = -0.2;
  constexpr double pressure = 1.0;
  constexpr double amplitude = 0.02;
  const double two_pi = 2.0 * dg_sim::math::kPi;
  const double advected_x = position(0) - velocity_x * time;
  const double advected_y = position(1) - velocity_y * time;
  const double density =
    1.0 + amplitude * std::sin(two_pi * advected_x) *
            std::cos(two_pi * advected_y);
  mfem::Vector primitive(num_eq);
  primitive(0) = density;
  primitive(1) = velocity_x;
  primitive(2) = velocity_y;
  primitive(3) = pressure;
  mfem::EulerPrimToConservative(primitive, dim, gamma, conservative);
}

} // namespace dgsem_convergence_detail

inline void RunDGSEMP1P4ConvergenceGate(TestContext& ctx)
{
  using namespace dgsem_convergence_detail;
  constexpr int dim = 2;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;
  constexpr int int_order_offset = 2;
  constexpr double velocity_x = 0.35;
  constexpr double velocity_y = -0.2;
  constexpr double density_min = 0.98;
  constexpr double pressure = 1.0;
  const double wave_speed =
    std::hypot(velocity_x, velocity_y) +
    std::sqrt(gamma * pressure / density_min);
  std::vector<ConvergenceRow> rows;

  const bool mesh_curvature[] = {false, true};
  for (const bool curved : mesh_curvature)
  {
    const char* mesh_kind = curved ? "curved" : "affine";
    for (const int order : mfem::dgsem::DGSEMConvergenceContract::orders)
    {
      std::vector<double> errors;
      errors.reserve(
        mfem::dgsem::DGSEMConvergenceContract::elements_per_direction.size());
      int level = 0;
      for (const int n :
           mfem::dgsem::DGSEMConvergenceContract::elements_per_direction)
      {
        mfem::Mesh mesh =
          MakeReferenceConvergenceMesh(n, order + 1, curved);
        mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
        mfem::DG_FECollection fec(order,
                                  dim,
                                  mfem::BasisType::GaussLobatto);
        mfem::ParFiniteElementSpace vfes(
          &pmesh, &fec, num_eq, mfem::Ordering::byNODES);
        mfem::ParGridFunction state(&vfes);
        auto initial = [](const mfem::Vector& position, mfem::Vector& value)
        {
          EvaluatePeriodicContact(position, 0.0, value);
        };
        mfem::VectorFunctionCoefficient initial_coefficient(num_eq, initial);
        state.ProjectCoefficient(initial_coefficient);

        mfem::EulerFlux flux(dim, gamma);
        mfem::RusanovFluxWithSides numerical_flux(flux);
        auto form_integrator =
          std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux,
                                                           int_order_offset);
        mfem::DGHyperbolicConservationLaws oper(
          vfes, std::move(form_integrator), true);
        oper.SetNumericalFlux(&numerical_flux);
        oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
        oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
        oper.SetDGSEMVolumeDissipation(mfem::DGSEMVolumeDissipation::None);
        oper.SetDGSEMGamma(gamma);
        oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order));

        // Constant-state preservation is checked on every p/mesh ladder before
        // evolving the smooth state used for the convergence norm.
        mfem::ParGridFunction constant_state(&vfes);
        FillConstantState(
          EulerConsFromPrim2D(1.0, velocity_x, velocity_y, pressure, gamma),
          constant_state);
        mfem::Vector constant_x;
        mfem::Vector constant_residual(vfes.GetTrueVSize());
        constant_state.GetTrueDofs(constant_x);
        oper.Mult(constant_x, constant_residual);
        const double constant_norm = GlobalL2Norm(constant_residual);
        const double constant_tolerance =
          curved ? mfem::dgsem::DGSEMReferenceTolerances::constant_curved
                 : mfem::dgsem::DGSEMReferenceTolerances::constant_affine;
        ctx.Check(constant_norm < constant_tolerance,
                  std::string("DGSEM ") + mesh_kind +
                    " constant-state residual (p=" +
                    std::to_string(order) + ", n=" + std::to_string(n) +
                    ")");

        mfem::Vector x;
        state.GetTrueDofs(x);
        mfem::RK3SSPWithLimiter rk(vfes);
        rk.Init(oper);
        const double h = 1.0 / static_cast<double>(n);
        const double dt_base =
          mfem::dgsem::DGSEMConvergenceContract::timestep_coefficient * h * h /
          wave_speed;
        double time = 0.0;
        int steps = 0;
        while (time + 1.0e-14 <
               mfem::dgsem::DGSEMConvergenceContract::final_time)
        {
          double dt =
            std::min(dt_base,
                     mfem::dgsem::DGSEMConvergenceContract::final_time - time);
          rk.Step(x, time, dt);
          ++steps;
        }
        state.SetFromTrueDofs(x);

        auto exact = [](const mfem::Vector& position, mfem::Vector& value)
        {
          EvaluatePeriodicContact(
            position,
            mfem::dgsem::DGSEMConvergenceContract::final_time,
            value);
        };
        mfem::VectorFunctionCoefficient exact_coefficient(num_eq, exact);
        const double error = state.ComputeL2Error(exact_coefficient);
        errors.push_back(error);
        ConvergenceRow row;
        row.mesh_kind = mesh_kind;
        row.order = order;
        row.level = level;
        row.elements_per_direction = n;
        row.steps = steps;
        row.h = h;
        row.dt_base = dt_base;
        row.error = error;
        row.constant_residual_norm = constant_norm;
        row.expected_rate_floor =
          mfem::dgsem::DGSEMConvergenceContract::ExpectedRateFloor(order);
        if (level > 0)
        {
          row.observed_rate =
            std::log(errors[static_cast<std::size_t>(level - 1)] / error) /
            std::log(2.0);
        }
        rows.push_back(row);
        ctx.LogMetric(std::string("DGSEM p1-p4 convergence: mesh=") +
                      mesh_kind + ", p=" + std::to_string(order) +
                      ", n=" + std::to_string(n) +
                      ", steps=" + std::to_string(steps) +
                      ", error=" + FormatSci(error) +
                      (level > 0 ? ", rate=" + FormatSci(row.observed_rate)
                                 : ""));
        ++level;
      }

      for (std::size_t i = 1; i < errors.size(); ++i)
      {
        ctx.Check(errors[i] <
                    mfem::dgsem::DGSEMConvergenceContract::monotone_error_factor *
                      errors[i - 1],
                  std::string("DGSEM ") + mesh_kind +
                    " convergence error decreases (p=" +
                    std::to_string(order) + ", level=" +
                    std::to_string(i) + ")");
        const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);
        ctx.Check(rate >
                    mfem::dgsem::DGSEMConvergenceContract::ExpectedRateFloor(
                      order),
                  std::string("DGSEM ") + mesh_kind +
                    " frozen convergence rate passes (p=" +
                    std::to_string(order) + ", rate=" + FormatSci(rate) +
                    ")");
      }
    }
  }

  const char* output_dir = ::igrlab::GetEnv("DG_IGR_DGSEM_REFERENCE_OUTPUT_DIR");
  if (output_dir && mfem::Mpi::Root())
  {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    const std::filesystem::path base(output_dir);
    std::ofstream csv(base / "p1_p4_convergence.csv");
    csv << std::setprecision(std::numeric_limits<double>::max_digits10);
    csv << "mesh_kind,order,level,elements_per_direction,h,dt_base,steps,"
           "final_time,norm,error,observed_rate,expected_rate_floor,"
           "constant_residual_norm\n";
    for (const auto& row : rows)
    {
      csv << row.mesh_kind << ',' << row.order << ',' << row.level << ','
          << row.elements_per_direction << ',' << row.h << ',' << row.dt_base
          << ',' << row.steps << ','
          << mfem::dgsem::DGSEMConvergenceContract::final_time
          << ",conservative_vector_L2," << row.error << ',';
      if (std::isfinite(row.observed_rate))
      {
        csv << row.observed_rate;
      }
      csv << ',' << row.expected_rate_floor << ','
          << row.constant_residual_norm << '\n';
    }
    std::ofstream contract(base / "convergence_contract.json");
    contract << "{\n"
             << "  \"schema\": \"dgsem-convergence-contract-v1\",\n"
             << "  \"orders\": [1,2,3,4],\n"
             << "  \"elements_per_direction\": [3,6,12],\n"
             << "  \"mesh_kinds\": [\"affine\",\"curved\"],\n"
             << "  \"norm\": \"conservative_vector_L2\",\n"
             << "  \"final_time\": "
             << mfem::dgsem::DGSEMConvergenceContract::final_time << ",\n"
             << "  \"timestep_policy\": \"0.05*h^2/(|velocity|+c_max), "
                "last_step_clipped\",\n"
             << "  \"expected_rate_floor\": \"p+0.25\",\n"
             << "  \"curved_map_amplitude\": "
             << mfem::dgsem::DGSEMConvergenceContract::curved_map_amplitude
             << "\n}\n";
    ctx.Check(!ec && csv.good() && contract.good(),
              "DGSEM machine-readable p1-p4 convergence evidence written");
  }
}

} // namespace igrlab_tests
