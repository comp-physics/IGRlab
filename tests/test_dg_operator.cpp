#include "igrlab/environment.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "mfem.hpp"
#include "test_utils.hpp"

namespace dg_igr_tests
{
namespace
{

class ScopedEnvironment
{
public:
  ScopedEnvironment(const char* name, const char* value) : name_(name)
  {
    if (const char* previous = igrlab::GetEnv(name))
    {
      previous_ = std::string(previous);
    }
    setenv(name, value, 1);
  }

  ~ScopedEnvironment()
  {
    if (previous_)
    {
      setenv(name_.c_str(), previous_->c_str(), 1);
    }
    else
    {
      unsetenv(name_.c_str());
    }
  }

private:
  std::string name_;
  std::optional<std::string> previous_;
};

} // namespace

int RunDGOperatorTests(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;
  const double tol = 1.0e-10;

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    /*nx=*/2,
    /*ny=*/2,
    mfem::Element::QUADRILATERAL,
    /*generate_edges=*/true,
    /*sx=*/1.0,
    /*sy=*/1.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

  mfem::DG_FECollection fec(/*order=*/1, dim);
  mfem::ParFiniteElementSpace vfes(&pmesh,
                                   &fec,
                                   /*vdim=*/num_eq,
                                   mfem::Ordering::byNODES);

  mfem::ParGridFunction U(&vfes);
  const mfem::Vector cons_const = EulerConsFromPrim2D(
    /*rho=*/1.0,
    /*u=*/0.0,
    /*v=*/0.0,
    /*p=*/1.0,
    gamma);
  FillConstantState(cons_const, U);

  mfem::EulerFlux flux(dim, gamma);
  mfem::RusanovFluxWithSides num_flux(flux);

  constexpr int int_order_offset = 2;
  auto form_integrator =
    std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                     int_order_offset);
  mfem::DGHyperbolicConservationLaws oper(vfes,
                                          std::move(form_integrator),
                                          /*preassembleWeakDivergence=*/true);

  // Apply slip-wall boundaries on all attributes to close the domain.
  mfem::Array<int> bdr_marker;
  const int max_attr = pmesh.bdr_attributes.Max();
  if (max_attr > 0)
  {
    bdr_marker.SetSize(max_attr);
    bdr_marker = 1;
    auto slip = std::make_unique<mfem::SlipWallState>(dim);
    oper.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(num_flux,
                                            std::move(slip),
                                            int_order_offset),
      bdr_marker);
  }

  mfem::Vector x;
  mfem::Vector y(vfes.GetTrueVSize());
  U.GetTrueDofs(x);
  oper.Mult(x, y);

  const double resid_norm = GlobalL2Norm(y);
  ctx.AddParity(resid_norm);
  ctx.Check(resid_norm < tol,
            "DG operator residual is near zero for a constant state (norm=" +
              FormatSci(resid_norm) + ")");

  // A prescribed physical-boundary state can dominate all volume and
  // interior-face samples. Its Rusanov speed must reach the operator CFL
  // report rather than remain private to BdrHyperbolicFluxIntegrator.
  {
    mfem::Mesh boundary_mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/1,
      /*ny=*/1,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh boundary_pmesh(MPI_COMM_SELF, boundary_mesh);
    mfem::DG_FECollection boundary_fec(/*order=*/1, dim);
    mfem::ParFiniteElementSpace boundary_vfes(&boundary_pmesh,
                                              &boundary_fec,
                                              num_eq,
                                              mfem::Ordering::byNODES);

    mfem::ParGridFunction boundary_state(&boundary_vfes);
    const mfem::Vector quiet_state = EulerConsFromPrim2D(
      /*rho=*/1.0,
      /*u=*/0.0,
      /*v=*/0.0,
      /*p=*/1.0,
      gamma);
    constexpr double inlet_velocity = 8.0;
    const mfem::Vector fast_inlet = EulerConsFromPrim2D(
      /*rho=*/1.0,
      /*u=*/inlet_velocity,
      /*v=*/0.0,
      /*p=*/1.0,
      gamma);
    FillConstantState(quiet_state, boundary_state);

    mfem::EulerFlux boundary_flux(dim, gamma);
    mfem::RusanovFluxWithSides boundary_num_flux(boundary_flux);
    auto boundary_form =
      std::make_unique<mfem::HyperbolicFormIntegrator>(boundary_num_flux,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws boundary_oper(
      boundary_vfes,
      std::move(boundary_form),
      /*preassembleWeakDivergence=*/true);

    mfem::Array<int> all_boundaries(boundary_pmesh.bdr_attributes.Max());
    all_boundaries = 1;
    boundary_oper.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
        boundary_num_flux,
        std::make_unique<mfem::SupersonicInletState>(fast_inlet),
        int_order_offset),
      all_boundaries);

    mfem::Vector boundary_x;
    mfem::Vector boundary_y(boundary_vfes.GetTrueVSize());
    boundary_state.GetTrueDofs(boundary_x);
    boundary_oper.Mult(boundary_x, boundary_y);

    const double quiet_speed = std::sqrt(gamma);
    const double expected_boundary_speed = inlet_velocity + quiet_speed;
    const double reported_boundary_speed = boundary_oper.GetMaxCharSpeed();
    ctx.AddParity(reported_boundary_speed);
    ctx.Check(reported_boundary_speed > quiet_speed + 1.0,
              "DG CFL accounting includes a boundary-dominant Rusanov speed");
    ctx.CheckNear(reported_boundary_speed,
                  expected_boundary_speed,
                  1.0e-12 * expected_boundary_speed,
                  "DG CFL accounting reports the physical-boundary maximum");
  }

#ifdef MFEM_USE_EXCEPTIONS
  // Exact Euler+IGR boundary construction must enforce the same static
  // admitted-state metadata on the CPU oracle as the packed CUDA path.
  {
    constexpr double density_floor = 0.25;
    constexpr double pressure_floor = 0.4;
    mfem::EulerFlux exact_base_flux(dim, gamma);
    mfem::FluxWithIGR exact_igr_flux(exact_base_flux, gamma);
    exact_igr_flux.SetInvariantDomainFloors(density_floor, pressure_floor);
    exact_igr_flux.SetInvariantDomainFlux(true);
    mfem::RusanovFluxWithSides exact_numerical_flux(exact_igr_flux);

    const mfem::ErrorAction previous_action = mfem::get_error_action();
    mfem::set_error_action(mfem::MFEM_ERROR_THROW);
    bool rejected_mismatched_floors = false;
    try
    {
      mfem::BdrHyperbolicFluxIntegrator rejected(
        exact_numerical_flux,
        std::make_unique<mfem::SlipWallState>(dim),
        int_order_offset);
    }
    catch (const mfem::ErrorException&)
    {
      rejected_mismatched_floors = true;
    }

    mfem::Vector nonfinite_inlet = EulerConsFromPrim2D(
      /*rho=*/1.0,
      /*u=*/0.0,
      /*v=*/0.0,
      /*p=*/1.0,
      gamma);
    nonfinite_inlet(2) = std::numeric_limits<double>::quiet_NaN();
    bool rejected_nonfinite_reference = false;
    try
    {
      mfem::BdrHyperbolicFluxIntegrator rejected(
        exact_numerical_flux,
        std::make_unique<mfem::SupersonicInletState>(nonfinite_inlet),
        int_order_offset);
    }
    catch (const mfem::ErrorException&)
    {
      rejected_nonfinite_reference = true;
    }
    mfem::set_error_action(previous_action);

    auto matching_wall = std::make_unique<mfem::SlipWallState>(
      dim, density_floor, pressure_floor);
    mfem::BdrHyperbolicFluxIntegrator accepted(exact_numerical_flux,
                                                std::move(matching_wall),
                                                int_order_offset);
    (void)accepted;
    ctx.Check(rejected_mismatched_floors &&
                rejected_nonfinite_reference,
              "exact Euler+IGR physical boundaries reject CPU/CUDA contract "
              "mismatches before assembly");
  }
#endif

  // The face-parallel packed CUDA boundary kernel must poison the complete
  // RHS transaction when exact-mode boundary admission fails. Returning a
  // zero face speed here would silently remove a physical boundary face.
  if (mfem::Device::Allows(mfem::Backend::DEVICE_MASK))
  {
    ScopedEnvironment face_parallel("DG_IGR_DG_FACE_Q_PARALLEL", "0");
    ScopedEnvironment packed_faces("DG_IGR_DG_FORCE_NLFORM_FACE", "0");

    mfem::Mesh invalid_mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/1,
      /*ny=*/1,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh invalid_pmesh(MPI_COMM_SELF, invalid_mesh);
    mfem::DG_FECollection invalid_fec(/*order=*/1, dim);
    mfem::ParFiniteElementSpace invalid_vfes(&invalid_pmesh,
                                             &invalid_fec,
                                             num_eq,
                                             mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace invalid_sigma_fes(&invalid_pmesh,
                                                   &invalid_fec,
                                                   1);
    mfem::ParGridFunction invalid_sigma(&invalid_sigma_fes);
    invalid_sigma = 0.0;

    constexpr double density_floor = 0.75;
    constexpr double pressure_floor = 0.25;
    const mfem::Vector inadmissible = EulerConsFromPrim2D(
      /*rho=*/0.5,
      /*u=*/0.0,
      /*v=*/0.0,
      /*p=*/1.0,
      gamma);
    mfem::ParGridFunction invalid_state(&invalid_vfes);
    FillConstantState(inadmissible, invalid_state);

    mfem::EulerFlux invalid_base_flux(dim, gamma);
    mfem::FluxWithIGR invalid_igr_flux(invalid_base_flux, gamma);
    invalid_igr_flux.SetSigmaField(&invalid_sigma);
    invalid_igr_flux.SetInvariantDomainFloors(density_floor, pressure_floor);
    invalid_igr_flux.SetInvariantDomainFlux(true);
    mfem::RusanovFluxWithSides invalid_num_flux(invalid_igr_flux);
    auto invalid_form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      invalid_num_flux,
      int_order_offset);
    mfem::DGHyperbolicConservationLaws invalid_oper(
      invalid_vfes,
      std::move(invalid_form),
      /*preassembleWeakDivergence=*/true);
    invalid_oper.SetDGSEMGamma(gamma);
    invalid_oper.SetNumericalFlux(&invalid_num_flux);

    mfem::Array<int> invalid_boundaries(
      invalid_pmesh.bdr_attributes.Max());
    invalid_boundaries = 1;
    invalid_oper.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
        invalid_num_flux,
        std::make_unique<mfem::SlipWallState>(
          dim, density_floor, pressure_floor),
        int_order_offset),
      invalid_boundaries);

    mfem::Vector invalid_x;
    mfem::Vector invalid_y(invalid_vfes.GetTrueVSize());
    invalid_state.GetTrueDofs(invalid_x);
    invalid_oper.ArmPhysicalBoundaryFluxIntegral();
    invalid_oper.Mult(invalid_x, invalid_y);

    bool residual_nonfinite = false;
    const double* invalid_y_values = invalid_y.HostRead();
    for (int i = 0; i < invalid_y.Size(); ++i)
    {
      residual_nonfinite = residual_nonfinite ||
                           !std::isfinite(invalid_y_values[i]);
    }
    bool boundary_flux_nonfinite = false;
    if (invalid_oper.HasLastPhysicalBoundaryFluxIntegral())
    {
      const mfem::Vector& invalid_boundary_flux =
        invalid_oper.GetLastPhysicalBoundaryFluxIntegral();
      const double* values = invalid_boundary_flux.HostRead();
      for (int eq = 0; eq < invalid_boundary_flux.Size(); ++eq)
      {
        boundary_flux_nonfinite = boundary_flux_nonfinite ||
                                  !std::isfinite(values[eq]);
      }
    }
    ctx.Check(std::isinf(invalid_oper.GetMaxCharSpeed()) &&
                residual_nonfinite && boundary_flux_nonfinite,
              "face-parallel exact CUDA boundary rejection poisons speed, "
              "residual, and armed flux capture");
  }

  // Conservation telemetry is a one-shot capture of the outward numerical
  // flux actually used on physical boundaries. A later unarmed Mult (for
  // example a CFL probe) must leave the completed capture untouched.
  {
    mfem::Mesh flux_mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/1,
      /*ny=*/1,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh flux_pmesh(MPI_COMM_SELF, flux_mesh);
    mfem::DG_FECollection flux_fec(/*order=*/1, dim);
    mfem::ParFiniteElementSpace flux_vfes(&flux_pmesh,
                                          &flux_fec,
                                          num_eq,
                                          mfem::Ordering::byNODES);

    constexpr double density = 1.2;
    constexpr double velocity_x = 2.0;
    constexpr double velocity_y = -0.5;
    constexpr double pressure = 0.8;
    const mfem::Vector flowing_state = EulerConsFromPrim2D(density,
                                                           velocity_x,
                                                           velocity_y,
                                                           pressure,
                                                           gamma);
    mfem::ParGridFunction flux_state(&flux_vfes);
    FillConstantState(flowing_state, flux_state);

    mfem::EulerFlux physical_flux(dim, gamma);
    mfem::RusanovFluxWithSides numerical_flux(physical_flux);
    auto flux_form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux,
      int_order_offset);
    mfem::DGHyperbolicConservationLaws flux_oper(
      flux_vfes,
      std::move(flux_form),
      /*preassembleWeakDivergence=*/true);

    // Cartesian2D boundary attribute 2 is the outward +x boundary. Leave the
    // other attributes without boundary integrators so this one unit-length
    // face is the complete physical-boundary contribution used by the form.
    mfem::Array<int> right_boundary(flux_pmesh.bdr_attributes.Max());
    right_boundary = 0;
    ctx.Check(right_boundary.Size() >= 2,
              "Cartesian flux test exposes its +x boundary attribute");
    if (right_boundary.Size() >= 2)
    {
      right_boundary[1] = 1;
    }
    flux_oper.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
        numerical_flux,
        std::make_unique<mfem::SupersonicOutletState>(),
        int_order_offset),
      right_boundary);

    mfem::Vector flux_x;
    mfem::Vector flux_y(flux_vfes.GetTrueVSize());
    flux_state.GetTrueDofs(flux_x);
    flux_oper.Mult(flux_x, flux_y);
    ctx.Check(!flux_oper.HasLastPhysicalBoundaryFluxIntegral(),
              "unarmed DG Mult does not manufacture conservation telemetry");

    flux_oper.ArmPhysicalBoundaryFluxIntegral();
    ctx.Check(!flux_oper.HasLastPhysicalBoundaryFluxIntegral(),
              "arming invalidates any earlier boundary-flux capture");
    flux_oper.Mult(flux_x, flux_y);
    ctx.Check(flux_oper.HasLastPhysicalBoundaryFluxIntegral(),
              "armed DG Mult publishes boundary-flux telemetry");

    const mfem::Vector& captured =
      flux_oper.GetLastPhysicalBoundaryFluxIntegral();
    ctx.Check(captured.Size() == num_eq,
              "boundary-flux telemetry stores one value per equation");
    const double energy = flowing_state(3);
    mfem::Vector expected(num_eq);
    expected(0) = density * velocity_x;
    expected(1) = density * velocity_x * velocity_x + pressure;
    expected(2) = density * velocity_x * velocity_y;
    expected(3) = (energy + pressure) * velocity_x;
    mfem::Vector captured_snapshot(captured);
    for (int eq = 0; eq < num_eq; ++eq)
    {
      ctx.AddParity(captured(eq));
      ctx.CheckNear(captured(eq),
                    expected(eq),
                    1.0e-12 * std::max(1.0, std::abs(expected(eq))),
                    "DG boundary-flux component " + std::to_string(eq) +
                      " matches the outward Euler numerical flux");
    }

    const mfem::Vector changed_state = EulerConsFromPrim2D(
      /*rho=*/0.9,
      /*u=*/-1.0,
      /*v=*/0.25,
      /*p=*/1.1,
      gamma);
    FillConstantState(changed_state, flux_state);
    flux_state.GetTrueDofs(flux_x);
    flux_oper.Mult(flux_x, flux_y);
    ctx.Check(flux_oper.HasLastPhysicalBoundaryFluxIntegral(),
              "unarmed DG Mult preserves the last completed capture");
    const mfem::Vector& preserved =
      flux_oper.GetLastPhysicalBoundaryFluxIntegral();
    for (int eq = 0; eq < num_eq; ++eq)
    {
      ctx.CheckNear(preserved(eq),
                    captured_snapshot(eq),
                    0.0,
                    "unarmed DG Mult leaves boundary-flux component " +
                      std::to_string(eq) + " unchanged");
    }
  }

  // Volume-node speed accounting is deliberately opt-in and exact-mode-only.
  // This Q2 velocity bubble is zero on every face and peaks at the center
  // interpolation node, separating the legacy face-only report from the
  // additional adaptive-CFL bound.
  {
    mfem::Mesh volume_mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/1,
      /*ny=*/1,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh volume_pmesh(MPI_COMM_SELF, volume_mesh);
    mfem::DG_FECollection volume_fec(/*order=*/2, dim);
    mfem::ParFiniteElementSpace volume_vfes(&volume_pmesh,
                                            &volume_fec,
                                            num_eq,
                                            mfem::Ordering::byNODES);

    constexpr double center_velocity = 8.0;
    constexpr double energy_density = 100.0;
    mfem::VectorFunctionCoefficient bubble_state(
      num_eq,
      [=](const mfem::Vector& point, mfem::Vector& value)
      {
        const double bubble =
          16.0 * point(0) * (1.0 - point(0)) * point(1) * (1.0 - point(1));
        value.SetSize(num_eq);
        value(0) = 1.0;
        value(1) = center_velocity * bubble;
        value(2) = 0.0;
        value(3) = energy_density;
      });
    mfem::ParGridFunction volume_state(&volume_vfes);
    volume_state.ProjectCoefficient(bubble_state);

    mfem::EulerFlux volume_flux(dim, gamma);
    mfem::RusanovFluxWithSides volume_num_flux(volume_flux);
    auto volume_form =
      std::make_unique<mfem::HyperbolicFormIntegrator>(volume_num_flux,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws volume_oper(
      volume_vfes,
      std::move(volume_form),
      /*preassembleWeakDivergence=*/true);

    mfem::Array<int> all_boundaries(volume_pmesh.bdr_attributes.Max());
    all_boundaries = 1;
    volume_oper.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
        volume_num_flux,
        std::make_unique<mfem::SupersonicOutletState>(),
        int_order_offset),
      all_boundaries);

    mfem::Vector volume_x;
    mfem::Vector volume_y(volume_vfes.GetTrueVSize());
    volume_state.GetTrueDofs(volume_x);
    volume_oper.Mult(volume_x, volume_y);

    const double face_pressure = (gamma - 1.0) * energy_density;
    const double face_speed = std::sqrt(gamma * face_pressure);
    const double center_pressure =
      (gamma - 1.0) *
      (energy_density - 0.5 * center_velocity * center_velocity);
    const double expected_volume_speed =
      center_velocity + std::sqrt(gamma * center_pressure);
    const double reported_volume_speed = volume_oper.GetMaxCharSpeed();
    ctx.AddParity(reported_volume_speed);
    ctx.Check(expected_volume_speed > face_speed + 1.0,
              "DG CFL volume test separates its center node from face speeds");
    ctx.CheckNear(reported_volume_speed,
                  face_speed,
                  1.0e-11 * face_speed,
                  "legacy DG retains face-only CFL accounting by default");

    volume_oper.SetInvariantDomainVolumeSpeedTracking(true);
    volume_oper.Mult(volume_x, volume_y);
    ctx.Check(volume_oper.GetInvariantDomainVolumeSpeedTracking(),
              "DG exposes the adaptive-CFL volume-speed request");
    ctx.CheckNear(volume_oper.GetMaxCharSpeed(),
                  face_speed,
                  1.0e-11 * face_speed,
                  "volume-speed request is inert for non-IGR fluxes");

    mfem::ParFiniteElementSpace volume_sigma_fes(&volume_pmesh,
                                                  &volume_fec,
                                                  1);
    mfem::ParGridFunction volume_sigma(&volume_sigma_fes);
    volume_sigma = 0.0;
    mfem::EulerFlux exact_base_flux(dim, gamma);
    mfem::FluxWithIGR exact_igr_flux(exact_base_flux, gamma);
    exact_igr_flux.SetSigmaField(&volume_sigma);
    exact_igr_flux.SetInvariantDomainFloors(1.0e-12, 1.0e-12);
    exact_igr_flux.SetInvariantDomainFlux(true);
    mfem::RusanovFluxWithSides exact_num_flux(exact_igr_flux);
    auto exact_form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      exact_num_flux,
      int_order_offset);
    mfem::DGHyperbolicConservationLaws exact_oper(
      volume_vfes,
      std::move(exact_form),
      /*preassembleWeakDivergence=*/true);
    exact_oper.SetDGSEMGamma(gamma);
    exact_oper.SetNumericalFlux(&exact_num_flux);
    exact_oper.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
        exact_num_flux,
        std::make_unique<mfem::SupersonicOutletState>(),
        int_order_offset),
      all_boundaries);

    mfem::Vector exact_y(volume_vfes.GetTrueVSize());
    exact_oper.Mult(volume_x, exact_y);
    const double exact_face_only_speed = exact_oper.GetMaxCharSpeed();
    const mfem::IGRFrozenSigmaTelemetry fixed_step_telemetry =
      exact_oper.GetGlobalFrozenSigmaTelemetry();
    ctx.Check(!exact_oper.GetInvariantDomainVolumeSpeedTracking(),
              "exact DG leaves volume-speed accounting off for fixed steps");
    ctx.CheckNear(exact_face_only_speed,
                  face_speed,
                  1.0e-11 * face_speed,
                  "fixed-step exact DG reports its face bound only");
    ctx.Check(fixed_step_telemetry.enabled &&
                fixed_step_telemetry.sample_count > 0,
              "fixed-step exact DG retains q_sigma telemetry without volume "
              "speed accounting");

    mfem::Vector fixed_step_residual(exact_y);
    exact_oper.SetInvariantDomainVolumeSpeedTracking(true);
    exact_oper.Mult(volume_x, exact_y);
    const double expected_exact_volume_speed =
      mfem::IGRFluxInvariantDomainSpeed(expected_volume_speed,
                                        1.0,
                                        center_pressure,
                                        center_velocity,
                                        0.0,
                                        gamma,
                                        1.0e-12,
                                        1.0e-12);
    ctx.CheckNear(exact_oper.GetMaxCharSpeed(),
                  expected_exact_volume_speed,
                  1.0e-11 * expected_exact_volume_speed,
                  "adaptive-CFL exact DG includes its volume-node bound");
    mfem::Vector tracking_difference(exact_y);
    tracking_difference -= fixed_step_residual;
    ctx.Check(GlobalL2Norm(tracking_difference) < 1.0e-12,
              "volume-speed accounting does not change the DG residual");
    const mfem::IGRFrozenSigmaTelemetry adaptive_telemetry =
      exact_oper.GetGlobalFrozenSigmaTelemetry();
    ctx.Check(adaptive_telemetry.sample_count ==
                2 * fixed_step_telemetry.sample_count,
              "q_sigma telemetry remains independent of volume-speed "
              "accounting");
  }

  // An Update invalidates the projected source's FE mapping. The operator
  // must discard those coefficients and fail closed until the caller makes an
  // explicit source-free or reprojected SetStaticSource decision.
  {
    mfem::Vector source(x.Size());
    source = 0.125;
    oper.SetStaticSource(source);
    oper.Mult(x, y);
    mfem::Vector source_difference(y);
    source_difference -= source;
    ctx.Check(oper.HasStaticSource() &&
                !oper.StaticSourceRequiresRebind() &&
                GlobalL2Norm(source_difference) < tol,
              "installed DG static source contributes in strong form");

    oper.Update();
    ctx.Check(!oper.HasStaticSource() && oper.StaticSourceRequiresRebind(),
              "DG Update discards the stale source and requires rebinding");
    oper.Update();
    ctx.Check(oper.StaticSourceRequiresRebind(),
              "repeated DG Update cannot clear a pending source rebind");

#ifdef MFEM_USE_EXCEPTIONS
    const mfem::ErrorAction previous_action = mfem::get_error_action();
    mfem::set_error_action(mfem::MFEM_ERROR_THROW);
    bool rejected_stale_source = false;
    try
    {
      oper.Mult(x, y);
    }
    catch (const mfem::ErrorException&)
    {
      rejected_stale_source = true;
    }
    mfem::set_error_action(previous_action);
    ctx.Check(rejected_stale_source,
              "DG Mult fails closed while static-source rebind is pending");
#endif

    oper.SetStaticSource(source);
    oper.Mult(x, y);
    source_difference = y;
    source_difference -= source;
    ctx.Check(oper.HasStaticSource() &&
                !oper.StaticSourceRequiresRebind() &&
                GlobalL2Norm(source_difference) < tol,
              "SetStaticSource re-enables DG Mult after Update");
  }

  // Exact-mode q_sigma telemetry follows the complete host DG speed surface
  // (volume plus physical-boundary states), accumulates attempted Mult work,
  // and exposes only a bounded communicator-wide aggregate.
  {
    mfem::Mesh telemetry_mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/1,
      /*ny=*/1,
      mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh telemetry_pmesh(MPI_COMM_SELF, telemetry_mesh);
    mfem::DG_FECollection telemetry_fec(/*order=*/1, dim);
    mfem::ParFiniteElementSpace telemetry_vfes(&telemetry_pmesh,
                                               &telemetry_fec,
                                               num_eq,
                                               mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace telemetry_sigma_fes(&telemetry_pmesh,
                                                     &telemetry_fec,
                                                     1);
    mfem::ParGridFunction telemetry_sigma(&telemetry_sigma_fes);
    constexpr double sigma_value = 0.25;
    telemetry_sigma = sigma_value;

    const mfem::Vector telemetry_cons = EulerConsFromPrim2D(
      /*rho=*/1.0,
      /*u=*/0.2,
      /*v=*/-0.1,
      /*p=*/1.0,
      gamma);
    mfem::ParGridFunction telemetry_state(&telemetry_vfes);
    FillConstantState(telemetry_cons, telemetry_state);

    mfem::EulerFlux telemetry_euler_flux(dim, gamma);
    mfem::FluxWithIGR telemetry_igr_flux(telemetry_euler_flux, gamma);
    telemetry_igr_flux.SetSigmaField(&telemetry_sigma);
    telemetry_igr_flux.SetInvariantDomainFloors(1.0e-12, 1.0e-12);
    telemetry_igr_flux.SetInvariantDomainFlux(true);
    mfem::RusanovFluxWithSides telemetry_numerical_flux(telemetry_igr_flux);
    auto telemetry_form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      telemetry_numerical_flux,
      int_order_offset);
    mfem::DGHyperbolicConservationLaws telemetry_oper(
      telemetry_vfes,
      std::move(telemetry_form),
      /*preassembleWeakDivergence=*/true);
    telemetry_oper.SetDGSEMGamma(gamma);
    telemetry_oper.SetNumericalFlux(&telemetry_numerical_flux);

    mfem::Array<int> telemetry_boundaries(
      telemetry_pmesh.bdr_attributes.Max());
    telemetry_boundaries = 1;
    telemetry_oper.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
        telemetry_numerical_flux,
        std::make_unique<mfem::SupersonicOutletState>(),
        int_order_offset),
      telemetry_boundaries);

    mfem::Vector telemetry_x;
    mfem::Vector telemetry_y(telemetry_vfes.GetTrueVSize());
    telemetry_state.GetTrueDofs(telemetry_x);
    telemetry_oper.Mult(telemetry_x, telemetry_y);
    const mfem::IGRFrozenSigmaTelemetry first =
      telemetry_oper.GetGlobalFrozenSigmaTelemetry();
    telemetry_oper.Mult(telemetry_x, telemetry_y);
    const mfem::IGRFrozenSigmaTelemetry second =
      telemetry_oper.GetGlobalFrozenSigmaTelemetry();
    const double expected_q_sigma =
      gamma * 1.0 + (gamma - 1.0) * sigma_value;
    ctx.Check(first.enabled && first.sample_count > 0,
              "Exact DG q_sigma telemetry observes host speed samples");
    ctx.Check(second.sample_count == 2 * first.sample_count,
              "Exact DG q_sigma telemetry accumulates attempted Mult work");
    ctx.Check(second.nonpositive_count == 0 && second.nonfinite_count == 0,
              "Admitted constant state has finite positive q_sigma samples");
    ctx.CheckNear(second.minimum_raw_q_sigma,
                  expected_q_sigma,
                  2.0e-13,
                  "Exact DG q_sigma telemetry combines volume and boundary");
    ctx.CheckNear(second.minimum_q_sigma_over_gamma_p,
                  expected_q_sigma / gamma,
                  2.0e-13,
                  "Exact DG q_sigma telemetry records normalized minimum");
  }

  return ctx.failures;
}

} // namespace igrlab_tests
