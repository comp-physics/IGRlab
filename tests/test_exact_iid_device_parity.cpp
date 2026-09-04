#include "mfem.hpp"

#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/igr/igr_flux.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr int kDimension = 2;
constexpr int kEquations = kDimension + 2;
constexpr int kOrder = 2;
constexpr int kFaceOrderOffset = 2;
constexpr double kGamma = 1.4;
constexpr double kDensityFloor = 1.0e-10;
constexpr double kPressureFloor = 1.0e-10;
constexpr char kReferenceMagic[] = "dg_igr_exact_iid_device_parity_v1";

struct Coverage
{
  std::uint64_t volume_samples = 0;
  std::uint64_t interior_face_samples = 0;
  std::uint64_t boundary_face_samples = 0;

  std::uint64_t Total() const
  {
    return volume_samples + interior_face_samples + boundary_face_samples;
  }
};

struct Observation
{
  std::vector<double> residual;
  std::vector<double> boundary_flux;
  double maximum_characteristic_speed = 0.0;
  mfem::IGRFrozenSigmaTelemetry telemetry;
};

int AbsoluteDof(int dof, double& sign)
{
  if (dof >= 0)
  {
    sign = 1.0;
    return dof;
  }
  sign = -1.0;
  return -1 - dof;
}

double EulerPressure(const double* state)
{
  const double momentum_squared =
    state[1] * state[1] + state[2] * state[2];
  return (kGamma - 1.0) *
         (state[3] - 0.5 * momentum_squared / state[0]);
}

mfem::Vector ConservativeState(double density,
                               double velocity_x,
                               double velocity_y,
                               double pressure)
{
  mfem::Vector state(kEquations);
  state(0) = density;
  state(1) = density * velocity_x;
  state(2) = density * velocity_y;
  state(3) = pressure / (kGamma - 1.0) +
             0.5 * density *
               (velocity_x * velocity_x + velocity_y * velocity_y);
  return state;
}

bool FillAdmittedDiscontinuousFields(
  mfem::ParFiniteElementSpace& state_space,
  mfem::ParFiniteElementSpace& sigma_space,
  mfem::ParGridFunction& state,
  mfem::ParGridFunction& sigma)
{
  if (state_space.GetOrdering() != mfem::Ordering::byNODES ||
      state_space.GetVDim() != kEquations || sigma_space.GetVDim() != 1 ||
      state_space.GetNE() != sigma_space.GetNE())
  {
    return false;
  }

  state = 0.0;
  sigma = 0.0;
  double* state_values = state.HostReadWrite();
  double* sigma_values = sigma.HostReadWrite();
  const int state_ndofs = state_space.GetNDofs();
  const int sigma_ndofs = sigma_space.GetNDofs();
  double minimum_density = std::numeric_limits<double>::infinity();
  double maximum_density = -std::numeric_limits<double>::infinity();
  double minimum_pressure = std::numeric_limits<double>::infinity();
  double minimum_sigma = std::numeric_limits<double>::infinity();
  double maximum_sigma = -std::numeric_limits<double>::infinity();

  mfem::Array<int> state_dofs;
  mfem::Array<int> sigma_dofs;
  mfem::Vector point(kDimension);
  for (int element = 0; element < state_space.GetNE(); ++element)
  {
    state_space.GetElementDofs(element, state_dofs);
    sigma_space.GetElementDofs(element, sigma_dofs);
    const mfem::FiniteElement& state_fe = *state_space.GetFE(element);
    const mfem::FiniteElement& sigma_fe = *sigma_space.GetFE(element);
    const mfem::IntegrationRule& state_nodes = state_fe.GetNodes();
    const mfem::IntegrationRule& sigma_nodes = sigma_fe.GetNodes();
    if (state_dofs.Size() != state_nodes.GetNPoints() ||
        sigma_dofs.Size() != sigma_nodes.GetNPoints() ||
        state_dofs.Size() != sigma_dofs.Size())
    {
      return false;
    }

    mfem::ElementTransformation& transformation =
      *state_space.GetElementTransformation(element);
    const double element_bias = static_cast<double>(element) - 1.5;
    for (int local_dof = 0; local_dof < state_dofs.Size(); ++local_dof)
    {
      transformation.Transform(state_nodes.IntPoint(local_dof), point);

      // Each conservative component is affine inside an element, with a
      // small element-dependent jump. This keeps every interpolated state far
      // inside the admitted set while exercising two distinct traces on each
      // interior face.
      const double density =
        1.10 + 0.05 * point(0) - 0.03 * point(1) + 0.015 * element_bias;
      const double momentum_x =
        0.32 + 0.04 * point(0) + 0.02 * point(1) + 0.012 * element_bias;
      const double momentum_y =
        -0.14 + 0.015 * point(0) - 0.025 * point(1) -
        0.008 * element_bias;
      const double energy =
        3.05 + 0.09 * point(0) + 0.04 * point(1) + 0.025 * element_bias;
      const double sigma_value =
        0.18 + 0.045 * point(0) - 0.025 * point(1) +
        0.020 * element_bias;
      const double conservative[kEquations] = {
        density, momentum_x, momentum_y, energy};
      const double pressure = EulerPressure(conservative);

      double state_sign = 1.0;
      const int state_dof = AbsoluteDof(state_dofs[local_dof], state_sign);
      double sigma_sign = 1.0;
      const int sigma_dof = AbsoluteDof(sigma_dofs[local_dof], sigma_sign);
      if (state_dof < 0 || state_dof >= state_ndofs || sigma_dof < 0 ||
          sigma_dof >= sigma_ndofs)
      {
        return false;
      }
      for (int equation = 0; equation < kEquations; ++equation)
      {
        state_values[state_dof + equation * state_ndofs] =
          state_sign * conservative[equation];
      }
      sigma_values[sigma_dof] = sigma_sign * sigma_value;

      minimum_density = std::min(minimum_density, density);
      maximum_density = std::max(maximum_density, density);
      minimum_pressure = std::min(minimum_pressure, pressure);
      minimum_sigma = std::min(minimum_sigma, sigma_value);
      maximum_sigma = std::max(maximum_sigma, sigma_value);
    }
  }

  return minimum_density > 100.0 * kDensityFloor &&
         minimum_pressure > 100.0 * kPressureFloor &&
         minimum_sigma > 0.0 && maximum_density > minimum_density &&
         maximum_sigma > minimum_sigma;
}

Coverage ExpectedCoverage(mfem::ParMesh& mesh,
                          const mfem::ParFiniteElementSpace& state_space)
{
  Coverage coverage;
  for (int element = 0; element < state_space.GetNE(); ++element)
  {
    coverage.volume_samples +=
      static_cast<std::uint64_t>(state_space.GetFE(element)->GetDof());
  }

  int interior_faces = 0;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    const mfem::FaceElementTransformations* transformation =
      mesh.GetFaceElementTransformations(face);
    if (transformation && transformation->Elem2No >= 0)
    {
      ++interior_faces;
    }
  }
  const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
    mfem::Geometry::SEGMENT, 2 * kOrder + kFaceOrderOffset);
  const std::uint64_t samples_per_face =
    2 * static_cast<std::uint64_t>(face_rule.GetNPoints());
  coverage.interior_face_samples =
    static_cast<std::uint64_t>(interior_faces) * samples_per_face;
  coverage.boundary_face_samples =
    static_cast<std::uint64_t>(mesh.GetNBE()) * samples_per_face;
  return coverage;
}

double VectorNorm(const std::vector<double>& values)
{
  double squared_norm = 0.0;
  for (const double value : values)
  {
    squared_norm += value * value;
  }
  return std::sqrt(squared_norm);
}

bool ValidateObservation(const Observation& observation,
                         const Coverage& coverage,
                         const std::string& backend)
{
  const auto finite_vector = [](const std::vector<double>& values)
  {
    return std::all_of(values.begin(), values.end(), [](double value)
    { return std::isfinite(value); });
  };
  const bool complete_coverage = coverage.volume_samples > 0 &&
                                 coverage.interior_face_samples > 0 &&
                                 coverage.boundary_face_samples > 0 &&
                                 observation.telemetry.sample_count ==
                                   coverage.Total();
  const bool valid = !observation.residual.empty() &&
                     observation.boundary_flux.size() == kEquations &&
                     finite_vector(observation.residual) &&
                     finite_vector(observation.boundary_flux) &&
                     VectorNorm(observation.residual) > 1.0e-8 &&
                     VectorNorm(observation.boundary_flux) > 1.0e-8 &&
                     std::isfinite(observation.maximum_characteristic_speed) &&
                     observation.maximum_characteristic_speed > 0.0 &&
                     observation.telemetry.enabled && complete_coverage &&
                     observation.telemetry.nonpositive_count == 0 &&
                     observation.telemetry.nonfinite_count == 0 &&
                     std::isfinite(observation.telemetry.minimum_raw_q_sigma) &&
                     observation.telemetry.minimum_raw_q_sigma > 0.0 &&
                     std::isfinite(
                       observation.telemetry.minimum_q_sigma_over_gamma_p) &&
                     observation.telemetry.minimum_q_sigma_over_gamma_p > 0.0;
  if (!valid)
  {
    std::cerr << "[FAIL] " << backend
              << " exact-IID observation did not exercise a finite, admitted "
                 "volume/interior/boundary DG residual"
              << " residual_norm=" << VectorNorm(observation.residual)
              << " boundary_flux_norm="
              << VectorNorm(observation.boundary_flux)
              << " speed=" << observation.maximum_characteristic_speed
              << " samples=" << observation.telemetry.sample_count
              << " expected_samples=" << coverage.Total()
              << " q_nonpositive="
              << observation.telemetry.nonpositive_count
              << " q_nonfinite=" << observation.telemetry.nonfinite_count
              << " min_q=" << observation.telemetry.minimum_raw_q_sigma
              << " min_q_ratio="
              << observation.telemetry.minimum_q_sigma_over_gamma_p << '\n';
  }
  return valid;
}

bool BuildObservation(Observation& observation, Coverage& coverage)
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    /*nx=*/2,
    /*ny=*/2,
    mfem::Element::QUADRILATERAL,
    /*generate_edges=*/true,
    /*sx=*/1.0,
    /*sy=*/1.0);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial_mesh);
  mfem::DG_FECollection collection(
    kOrder, kDimension, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace state_space(&mesh,
                                          &collection,
                                          kEquations,
                                          mfem::Ordering::byNODES);
  mfem::ParFiniteElementSpace sigma_space(&mesh, &collection);
  mfem::ParGridFunction state(&state_space);
  mfem::ParGridFunction sigma(&sigma_space);
  if (!FillAdmittedDiscontinuousFields(
        state_space, sigma_space, state, sigma))
  {
    std::cerr << "[FAIL] could not construct the admitted discontinuous "
                 "Euler/sigma parity state\n";
    return false;
  }

  coverage = ExpectedCoverage(mesh, state_space);

  mfem::EulerFlux euler_flux(kDimension, kGamma);
  mfem::FluxWithIGR igr_flux(euler_flux, kGamma);
  igr_flux.SetSigmaField(&sigma);
  igr_flux.SetInvariantDomainFloors(kDensityFloor, kPressureFloor);
  igr_flux.SetInvariantDomainFlux(true);
  mfem::RusanovFluxWithSides numerical_flux(igr_flux);
  auto form_integrator = std::make_unique<mfem::HyperbolicFormIntegrator>(
    numerical_flux, kFaceOrderOffset);
  mfem::DGHyperbolicConservationLaws dg_operator(
    state_space,
    std::move(form_integrator),
    /*preassembleWeakDivergence=*/true);
  dg_operator.SetVolumeMode(mfem::VolumeMode::WeakDivergence);
  dg_operator.SetDGSEMGamma(kGamma);
  dg_operator.SetNumericalFlux(&numerical_flux);
  dg_operator.SetFaceQuadratureOrderOffset(kFaceOrderOffset);

  const int maximum_boundary_attribute = mesh.bdr_attributes.Max();
  if (maximum_boundary_attribute <= 0)
  {
    std::cerr << "[FAIL] parity mesh has no physical boundary attributes\n";
    return false;
  }
  mfem::Array<int> all_boundaries(maximum_boundary_attribute);
  all_boundaries = 1;
  const mfem::Vector inlet_state = ConservativeState(
    /*density=*/0.96,
    /*velocity_x=*/0.52,
    /*velocity_y=*/-0.07,
    /*pressure=*/1.08);
  dg_operator.AddBdrFaceIntegrator(
    new mfem::BdrHyperbolicFluxIntegrator(
      numerical_flux,
      std::make_unique<mfem::SupersonicInletState>(inlet_state),
      kFaceOrderOffset),
    all_boundaries);

  if (!igr_flux.GetInvariantDomainFlux() ||
      dg_operator.GetVolumeMode() != mfem::VolumeMode::WeakDivergence ||
      !dg_operator.SupportsPhysicalBoundaryFluxIntegral())
  {
    std::cerr << "[FAIL] parity operator is not the requested ordinary-DG "
                 "exact-IID boundary-capable route\n";
    return false;
  }

  mfem::Vector state_true;
  state.GetTrueDofs(state_true);
  mfem::Vector residual(state_space.GetTrueVSize());
  dg_operator.ArmPhysicalBoundaryFluxIntegral();
  dg_operator.Mult(state_true, residual);
  if (!dg_operator.HasLastPhysicalBoundaryFluxIntegral())
  {
    std::cerr << "[FAIL] physical-boundary flux capture did not complete\n";
    return false;
  }

  const double* residual_values = residual.HostRead();
  observation.residual.assign(residual_values,
                              residual_values + residual.Size());
  const mfem::Vector& boundary_flux =
    dg_operator.GetLastPhysicalBoundaryFluxIntegral();
  const double* boundary_values = boundary_flux.HostRead();
  observation.boundary_flux.assign(boundary_values,
                                   boundary_values + boundary_flux.Size());
  observation.maximum_characteristic_speed = dg_operator.GetMaxCharSpeed();
  observation.telemetry = dg_operator.GetGlobalFrozenSigmaTelemetry();
  return true;
}

bool WriteReference(const std::string& path, const Observation& observation)
{
  std::ofstream output(path, std::ios::trunc);
  if (!output)
  {
    std::cerr << "[FAIL] cannot open CPU reference for writing: " << path
              << '\n';
    return false;
  }
  output << kReferenceMagic << '\n';
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::scientific;
  output << "residual " << observation.residual.size() << '\n';
  for (const double value : observation.residual)
  {
    output << value << '\n';
  }
  output << "boundary_flux " << observation.boundary_flux.size() << '\n';
  for (const double value : observation.boundary_flux)
  {
    output << value << '\n';
  }
  output << "maximum_characteristic_speed "
         << observation.maximum_characteristic_speed << '\n';
  output << "telemetry_enabled "
         << static_cast<int>(observation.telemetry.enabled) << '\n';
  output << "telemetry_sample_count "
         << observation.telemetry.sample_count << '\n';
  output << "telemetry_nonpositive_count "
         << observation.telemetry.nonpositive_count << '\n';
  output << "telemetry_nonfinite_count "
         << observation.telemetry.nonfinite_count << '\n';
  output << "telemetry_minimum_raw_q_sigma "
         << observation.telemetry.minimum_raw_q_sigma << '\n';
  output << "telemetry_minimum_q_sigma_over_gamma_p "
         << observation.telemetry.minimum_q_sigma_over_gamma_p << '\n';
  output << "end\n";
  if (!output)
  {
    std::cerr << "[FAIL] could not complete CPU reference: " << path << '\n';
    return false;
  }
  return true;
}

bool ReadLabel(std::istream& input, const char* expected)
{
  std::string label;
  if (!(input >> label) || label != expected)
  {
    std::cerr << "[FAIL] malformed CPU parity reference: expected label '"
              << expected << "'\n";
    return false;
  }
  return true;
}

bool ReadVector(std::istream& input,
                const char* label,
                std::vector<double>& values)
{
  std::size_t size = 0;
  if (!ReadLabel(input, label) || !(input >> size) || size > 1000000)
  {
    return false;
  }
  values.resize(size);
  for (double& value : values)
  {
    if (!(input >> value))
    {
      std::cerr << "[FAIL] malformed CPU parity reference vector '" << label
                << "'\n";
      return false;
    }
  }
  return true;
}

bool ReadReference(const std::string& path, Observation& observation)
{
  std::ifstream input(path);
  if (!input)
  {
    std::cerr << "[FAIL] cannot open CPU parity reference: " << path << '\n';
    return false;
  }
  std::string magic;
  if (!(input >> magic) || magic != kReferenceMagic)
  {
    std::cerr << "[FAIL] stale or malformed CPU parity reference: " << path
              << '\n';
    return false;
  }
  int enabled = 0;
  unsigned long long sample_count = 0;
  unsigned long long nonpositive_count = 0;
  unsigned long long nonfinite_count = 0;
  if (!ReadVector(input, "residual", observation.residual) ||
      !ReadVector(input, "boundary_flux", observation.boundary_flux) ||
      !ReadLabel(input, "maximum_characteristic_speed") ||
      !(input >> observation.maximum_characteristic_speed) ||
      !ReadLabel(input, "telemetry_enabled") || !(input >> enabled) ||
      !ReadLabel(input, "telemetry_sample_count") ||
      !(input >> sample_count) ||
      !ReadLabel(input, "telemetry_nonpositive_count") ||
      !(input >> nonpositive_count) ||
      !ReadLabel(input, "telemetry_nonfinite_count") ||
      !(input >> nonfinite_count) ||
      !ReadLabel(input, "telemetry_minimum_raw_q_sigma") ||
      !(input >> observation.telemetry.minimum_raw_q_sigma) ||
      !ReadLabel(input, "telemetry_minimum_q_sigma_over_gamma_p") ||
      !(input >> observation.telemetry.minimum_q_sigma_over_gamma_p) ||
      !ReadLabel(input, "end"))
  {
    return false;
  }
  observation.telemetry.enabled = enabled != 0;
  observation.telemetry.sample_count =
    static_cast<std::uint64_t>(sample_count);
  observation.telemetry.nonpositive_count =
    static_cast<std::uint64_t>(nonpositive_count);
  observation.telemetry.nonfinite_count =
    static_cast<std::uint64_t>(nonfinite_count);
  return true;
}

bool NearlyEqual(double reference,
                 double current,
                 double absolute_tolerance,
                 double relative_tolerance)
{
  return std::isfinite(reference) && std::isfinite(current) &&
         std::abs(current - reference) <=
           absolute_tolerance +
             relative_tolerance * std::max(std::abs(reference),
                                           std::abs(current));
}

bool CompareVector(const char* name,
                   const std::vector<double>& reference,
                   const std::vector<double>& current,
                   double absolute_tolerance,
                   double relative_tolerance)
{
  if (reference.size() != current.size())
  {
    std::cerr << "[FAIL] " << name << " size differs: CPU="
              << reference.size() << " CUDA=" << current.size() << '\n';
    return false;
  }
  double maximum_error = 0.0;
  std::size_t maximum_index = 0;
  for (std::size_t index = 0; index < reference.size(); ++index)
  {
    const double error = std::abs(current[index] - reference[index]);
    if (error > maximum_error)
    {
      maximum_error = error;
      maximum_index = index;
    }
    if (!NearlyEqual(reference[index],
                     current[index],
                     absolute_tolerance,
                     relative_tolerance))
    {
      std::cerr << "[FAIL] " << name << " differs at index " << index
                << ": CPU=" << std::setprecision(17) << reference[index]
                << " CUDA=" << current[index] << " error=" << error << '\n';
      return false;
    }
  }
  std::cout << "[PASS] " << name << " CPU/CUDA parity"
            << " max_abs_error=" << std::setprecision(6) << std::scientific
            << maximum_error << " index=" << maximum_index << '\n';
  return true;
}

bool CompareObservations(const Observation& reference,
                         const Observation& current)
{
  bool pass = true;
  pass = CompareVector("ordinary-DG exact-IID residual",
                       reference.residual,
                       current.residual,
                       /*absolute_tolerance=*/2.0e-11,
                       /*relative_tolerance=*/2.0e-11) &&
         pass;
  pass = CompareVector("physical-boundary flux integral",
                       reference.boundary_flux,
                       current.boundary_flux,
                       /*absolute_tolerance=*/2.0e-12,
                       /*relative_tolerance=*/2.0e-12) &&
         pass;

  const auto compare_scalar = [&](const char* name,
                                  double cpu,
                                  double cuda,
                                  double absolute_tolerance,
                                  double relative_tolerance)
  {
    if (!NearlyEqual(cpu,
                     cuda,
                     absolute_tolerance,
                     relative_tolerance))
    {
      std::cerr << "[FAIL] " << name << " differs: CPU="
                << std::setprecision(17) << cpu << " CUDA=" << cuda << '\n';
      pass = false;
    }
  };
  compare_scalar("maximum characteristic speed",
                 reference.maximum_characteristic_speed,
                 current.maximum_characteristic_speed,
                 2.0e-13,
                 2.0e-13);
  compare_scalar("minimum raw q_sigma",
                 reference.telemetry.minimum_raw_q_sigma,
                 current.telemetry.minimum_raw_q_sigma,
                 2.0e-13,
                 2.0e-13);
  compare_scalar("minimum q_sigma/(gamma*p)",
                 reference.telemetry.minimum_q_sigma_over_gamma_p,
                 current.telemetry.minimum_q_sigma_over_gamma_p,
                 2.0e-13,
                 2.0e-13);

  const bool exact_telemetry =
    reference.telemetry.enabled == current.telemetry.enabled &&
    reference.telemetry.sample_count == current.telemetry.sample_count &&
    reference.telemetry.nonpositive_count ==
      current.telemetry.nonpositive_count &&
    reference.telemetry.nonfinite_count ==
      current.telemetry.nonfinite_count;
  if (!exact_telemetry)
  {
    std::cerr << "[FAIL] q_sigma count telemetry differs:"
              << " CPU(enabled,samples,nonpositive,nonfinite)=("
              << reference.telemetry.enabled << ','
              << reference.telemetry.sample_count << ','
              << reference.telemetry.nonpositive_count << ','
              << reference.telemetry.nonfinite_count << ")"
              << " CUDA=(" << current.telemetry.enabled << ','
              << current.telemetry.sample_count << ','
              << current.telemetry.nonpositive_count << ','
              << current.telemetry.nonfinite_count << ")\n";
    pass = false;
  }
  if (pass)
  {
    std::cout << "[PASS] characteristic speed and all q_sigma telemetry "
                 "fields have CPU/CUDA parity\n";
  }
  return pass;
}

void PrintUsage(const char* executable)
{
  std::cerr << "usage: " << executable
            << " <cpu|cuda> <--write-reference|--compare> <reference-file>\n";
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  if (argc != 4)
  {
    PrintUsage(argv[0]);
    return 2;
  }
  const std::string device_name(argv[1]);
  const std::string action(argv[2]);
  const std::string reference_path(argv[3]);
  const bool write_reference = action == "--write-reference";
  const bool compare_reference = action == "--compare";
  if ((device_name != "cpu" && device_name != "cuda") ||
      (!write_reference && !compare_reference) ||
      (write_reference && device_name != "cpu") ||
      (compare_reference && device_name != "cuda") ||
      mfem::Mpi::WorldSize() != 1)
  {
    PrintUsage(argv[0]);
    return 2;
  }

  mfem::Device device(device_name.c_str());
  const bool accelerator_enabled =
    mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  if ((device_name == "cuda") != accelerator_enabled)
  {
    std::cerr << "[FAIL] requested backend '" << device_name
              << "' did not select the expected MFEM execution class\n";
    return 3;
  }

  Observation observation;
  Coverage coverage;
  if (!BuildObservation(observation, coverage) ||
      !ValidateObservation(observation, coverage, device_name))
  {
    return 4;
  }

  if (write_reference)
  {
    if (!WriteReference(reference_path, observation))
    {
      return 5;
    }
    std::cout << "[PASS] wrote exact-IID ordinary-DG CPU reference"
              << " samples(volume,interior,boundary)=("
              << coverage.volume_samples << ','
              << coverage.interior_face_samples << ','
              << coverage.boundary_face_samples << ")\n";
    return 0;
  }

  Observation reference;
  if (!ReadReference(reference_path, reference) ||
      !ValidateObservation(reference, coverage, "CPU reference"))
  {
    return 6;
  }
  return CompareObservations(reference, observation) ? 0 : 7;
}
