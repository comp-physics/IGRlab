#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/sha256.hpp"

#include <HYPRE_krylov.h>
#include <HYPRE_parcsr_ls.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace dg::hdg::runtime::cpu::sc25
{
namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double local_absolute_tolerance = 1.0e-12;
constexpr double local_relative_tolerance = 1.0e-11;

using State = std::array<double, state_components>;
using FluxJacobian =
    std::array<std::array<double, trace_components>, state_components>;

bool Finite(const double value) noexcept { return std::isfinite(value); }

template <std::size_t N>
bool Finite(const std::array<double, N>& values) noexcept
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return Finite(value); });
}

bool Finite(const std::vector<double>& values) noexcept
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return Finite(value); });
}

int DecodeSignedDof(const int signed_dof, int& sign) noexcept
{
  if (signed_dof == std::numeric_limits<int>::min())
  {
    sign = 0;
    return -1;
  }
  sign = signed_dof < 0 ? -1 : 1;
  return signed_dof < 0 ? -1 - signed_dof : signed_dof;
}

int EncodeSignedDof(const int dof, const int sign) noexcept
{
  return sign < 0 ? -1 - dof : dof;
}

std::vector<int> CopyArray(const mfem::Array<int>& values)
{
  return std::vector<int>(values.begin(), values.end());
}

bool Primitive(const State& conservative, double& pressure,
               std::array<double, dimension>& velocity) noexcept
{
  if (!Finite(conservative) || !(conservative[0] > 0.0)) { return false; }
  velocity[0] = conservative[1] / conservative[0];
  velocity[1] = conservative[2] / conservative[0];
  pressure = (gamma - 1.0) *
      (conservative[3] -
       0.5 * (conservative[1] * conservative[1] +
              conservative[2] * conservative[2]) / conservative[0]);
  return Finite(pressure) && pressure > 0.0 && Finite(velocity);
}

struct DenseLu final
{
  int size = 0;
  std::vector<double> values;
  std::vector<int> pivots;
  int factor_failure_column = -1;
  double factor_pivot_magnitude = std::numeric_limits<double>::quiet_NaN();
  double factor_pivot_threshold = 0.0;
  const char* factor_failure_reason = "not-reached";

  bool Factor(const std::vector<double>& matrix, const int n) noexcept
  {
    factor_failure_column = -1;
    factor_pivot_magnitude = std::numeric_limits<double>::quiet_NaN();
    factor_pivot_threshold = 0.0;
    factor_failure_reason = "input-validation";
    try
    {
      if (n <= 0 || matrix.size() != static_cast<std::size_t>(n) * n ||
          !Finite(matrix))
      {
        return false;
      }
      size = n;
      values = matrix;
      pivots.assign(static_cast<std::size_t>(n), 0);
      for (int column = 0; column < n; ++column)
      {
        int pivot = column;
        double maximum = std::abs(values[
            static_cast<std::size_t>(column) * n + column]);
        for (int row = column + 1; row < n; ++row)
        {
          const double candidate = std::abs(values[
              static_cast<std::size_t>(row) * n + column]);
          if (candidate > maximum) { maximum = candidate; pivot = row; }
        }
        factor_failure_column = column;
        factor_pivot_magnitude = maximum;
        if (!Finite(maximum))
        {
          factor_failure_reason = "nonfinite-pivot";
          return false;
        }
        if (!(maximum > factor_pivot_threshold))
        {
          factor_failure_reason = "pivot-not-above-zero-threshold";
          return false;
        }
        pivots[static_cast<std::size_t>(column)] = pivot;
        if (pivot != column)
        {
          for (int entry = 0; entry < n; ++entry)
          {
            std::swap(values[static_cast<std::size_t>(column) * n + entry],
                      values[static_cast<std::size_t>(pivot) * n + entry]);
          }
        }
        const double diagonal = values[
            static_cast<std::size_t>(column) * n + column];
        for (int row = column + 1; row < n; ++row)
        {
          double& multiplier = values[
              static_cast<std::size_t>(row) * n + column];
          multiplier /= diagonal;
          for (int entry = column + 1; entry < n; ++entry)
          {
            values[static_cast<std::size_t>(row) * n + entry] -=
                multiplier * values[
                    static_cast<std::size_t>(column) * n + entry];
          }
        }
      }
      if (!Finite(values))
      {
        factor_failure_reason = "nonfinite-factor";
        return false;
      }
      factor_failure_column = -1;
      factor_pivot_magnitude = std::numeric_limits<double>::quiet_NaN();
      factor_failure_reason = "none";
      return true;
    }
    catch (...)
    {
      factor_failure_reason = "exception";
      return false;
    }
  }

  bool Solve(const std::vector<double>& rhs,
             std::vector<double>& solution) const noexcept
  {
    try
    {
      if (size <= 0 || rhs.size() != static_cast<std::size_t>(size) ||
          pivots.size() != rhs.size() || !Finite(rhs))
      {
        return false;
      }
      solution = rhs;
      for (int column = 0; column < size; ++column)
      {
        const int pivot = pivots[static_cast<std::size_t>(column)];
        if (pivot < column || pivot >= size) { return false; }
        if (pivot != column) { std::swap(solution[column], solution[pivot]); }
      }
      for (int column = 0; column < size; ++column)
      {
        for (int row = column + 1; row < size; ++row)
        {
          solution[row] -= values[
              static_cast<std::size_t>(row) * size + column] *
              solution[column];
        }
      }
      for (int row = size - 1; row >= 0; --row)
      {
        for (int column = row + 1; column < size; ++column)
        {
          solution[row] -= values[
              static_cast<std::size_t>(row) * size + column] *
              solution[column];
        }
        const double diagonal = values[
            static_cast<std::size_t>(row) * size + row];
        if (!Finite(diagonal) || diagonal == 0.0) { return false; }
        solution[row] /= diagonal;
      }
      return Finite(solution);
    }
    catch (...)
    {
      return false;
    }
  }
};

double ApplicationResidual(const std::vector<double>& matrix, const int n,
                           const std::vector<double>& solution,
                           const std::vector<double>& rhs) noexcept
{
  long double residual_square = 0.0L;
  for (int row = 0; row < n; ++row)
  {
    double application = 0.0;
    for (int column = 0; column < n; ++column)
    {
      application += matrix[static_cast<std::size_t>(row) * n + column] *
          solution[static_cast<std::size_t>(column)];
    }
    const double residual = application - rhs[static_cast<std::size_t>(row)];
    residual_square += static_cast<long double>(residual) * residual;
  }
  return std::sqrt(static_cast<double>(residual_square));
}

double VectorNorm(const std::vector<double>& values) noexcept
{
  long double square = 0.0L;
  for (const double value : values)
  {
    square += static_cast<long double>(value) * value;
  }
  return std::sqrt(static_cast<double>(square));
}

SpaceIdentity DescribeSpace(const mfem::ParFiniteElementSpace* space) noexcept
{
  SpaceIdentity identity;
  if (space == nullptr) { return identity; }
  identity.address = space;
  identity.collection = space->FEColl();
  identity.mesh = space->GetParMesh();
  identity.vector_dimension = space->GetVDim();
  identity.ordering = space->GetOrdering();
  identity.local_vector_size = space->GetVSize();
  identity.local_true_size = space->GetTrueVSize();
  identity.global_true_size = space->GlobalTrueVSize();
  identity.true_offset_begin = space->GetMyTDofOffset();
  identity.true_offset_end =
      identity.true_offset_begin + identity.local_true_size;
  return identity;
}

bool ExactSpace(const mfem::ParFiniteElementSpace* space,
                const mfem::ParMesh* mesh,
                const mfem::FiniteElementCollection* collection,
                const int vdim) noexcept
{
  return space != nullptr && space->GetParMesh() == mesh &&
      space->FEColl() == collection && space->GetVDim() == vdim &&
      space->GetOrdering() == mfem::Ordering::byNODES;
}

bool ExactField(const mfem::ParGridFunction* field,
                const mfem::ParFiniteElementSpace* space) noexcept
{
  return field != nullptr && field->ParFESpace() == space &&
      field->FESpace() == space && field->Size() == space->GetVSize();
}

} // namespace

double VelocityGradientInvariant(
    const std::array<std::array<double, dimension>, dimension>& gradient)
    noexcept
{
  const double trace_value = gradient[0][0] + gradient[1][1];
  return gradient[0][0] * gradient[0][0] +
      2.0 * gradient[0][1] * gradient[1][0] +
      gradient[1][1] * gradient[1][1] + trace_value * trace_value;
}

InitialStatePoint SmoothInitialState(const double x, const double y) noexcept
{
  InitialStatePoint output;
  const double rho = 1.0 + 0.05 * std::sin(2.0 * pi * x) *
      std::sin(2.0 * pi * y);
  const double u = 0.1 * std::sin(2.0 * pi * x);
  const double v = 0.1 * std::sin(2.0 * pi * y);
  output.conservative = {{
      rho, rho * u, rho * v,
      1.0 / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)}};
  return output;
}

bool EulerSigmaNormalFlux(const State& conservative, const double sigma,
                          const std::array<double, dimension>& normal,
                          State& flux, FluxJacobian& jacobian) noexcept
{
  double pressure = 0.0;
  std::array<double, dimension> velocity{};
  if (!Primitive(conservative, pressure, velocity) || !Finite(sigma) ||
      !Finite(normal))
  {
    return false;
  }
  const double rho = conservative[0];
  const double mx = conservative[1];
  const double my = conservative[2];
  const double energy = conservative[3];
  const double un = velocity[0] * normal[0] + velocity[1] * normal[1];
  flux = {{mx * normal[0] + my * normal[1],
           mx * un + (pressure + sigma) * normal[0],
           my * un + (pressure + sigma) * normal[1],
           (energy + pressure + sigma) * un}};
  for (auto& row : jacobian) { row.fill(0.0); }
  const double gm1 = gamma - 1.0;
  const double momentum_square = mx * mx + my * my;
  const State pressure_derivative{{
      0.5 * gm1 * momentum_square / (rho * rho),
      -gm1 * mx / rho, -gm1 * my / rho, gm1}};
  const State normal_velocity_derivative{{
      -un / rho, normal[0] / rho, normal[1] / rho, 0.0}};
  jacobian[0][1] = normal[0];
  jacobian[0][2] = normal[1];
  for (int column = 0; column < state_components; ++column)
  {
    jacobian[1][column] = (column == 1 ? un : 0.0) +
        mx * normal_velocity_derivative[column] +
        pressure_derivative[column] * normal[0];
    jacobian[2][column] = (column == 2 ? un : 0.0) +
        my * normal_velocity_derivative[column] +
        pressure_derivative[column] * normal[1];
    jacobian[3][column] =
        ((column == 3 ? 1.0 : 0.0) + pressure_derivative[column]) * un +
        (energy + pressure + sigma) *
            normal_velocity_derivative[column];
  }
  jacobian[1][4] = normal[0];
  jacobian[2][4] = normal[1];
  jacobian[3][4] = un;
  return Finite(flux) &&
      std::all_of(jacobian.begin(), jacobian.end(),
                  [](const auto& row) { return Finite(row); });
}

bool SourceCorrectionNormalFlux(const State& conservative,
                                const double sigma,
                                const std::array<double, dimension>& normal,
                                State& correction,
                                FluxJacobian& jacobian) noexcept
{
  if (!Finite(conservative) || !(conservative[0] > 0.0) || !Finite(sigma) ||
      !Finite(normal))
  {
    return false;
  }
  const double un = (conservative[1] * normal[0] +
                     conservative[2] * normal[1]) / conservative[0];
  correction = {{0.0, sigma * normal[0], sigma * normal[1], sigma * un}};
  for (auto& row : jacobian) { row.fill(0.0); }
  jacobian[1][4] = normal[0];
  jacobian[2][4] = normal[1];
  jacobian[3][0] = -sigma * un / conservative[0];
  jacobian[3][1] = sigma * normal[0] / conservative[0];
  jacobian[3][2] = sigma * normal[1] / conservative[0];
  jacobian[3][4] = un;
  return Finite(correction);
}

bool EvaluateAuxiliaryStabilization(
    const int polynomial_degree, const double rhohat, const double height,
    const double sigma, const double sigmahat, const double z_dot_normal,
    const double alpha, const double test_value,
    AuxiliaryStabilizationPoint& output) noexcept
{
  output = {};
  if (polynomial_degree < 1 || !(rhohat > 0.0) || !(height > 0.0) ||
      !(alpha > 0.0) || !Finite(sigma) || !Finite(sigmahat) ||
      !Finite(z_dot_normal) || !Finite(test_value))
  {
    return false;
  }
  const double degree_scale = static_cast<double>(
      (polynomial_degree + 1) * (polynomial_degree + 1));
  output.tau = degree_scale / (rhohat * height);
  const double jump = sigma - sigmahat;
  output.numerical_flux = z_dot_normal - output.tau * jump;
  output.r_sigma_d_sigma = alpha * test_value * output.tau;
  output.r_sigma_d_sigmahat = -alpha * test_value * output.tau;
  output.r_sigma_d_rhohat =
      -alpha * test_value * output.tau * jump / rhohat;
  output.g_sigma_d_sigma = -alpha * test_value * output.tau;
  output.g_sigma_d_sigmahat = alpha * test_value * output.tau;
  output.g_sigma_d_rhohat =
      alpha * test_value * output.tau * jump / rhohat;
  return Finite(output.tau) && Finite(output.numerical_flux) &&
      Finite(output.r_sigma_d_rhohat) && Finite(output.g_sigma_d_rhohat);
}

bool CondenseElementLinearization(const ElementLinearization& input,
                                  ElementCondensedBlock& output) noexcept
{
  output = {};
  try
  {
    const int ni = input.local_interior_unknowns;
    const int nt = input.local_trace_unknowns;
    if (input.element < 0 || ni <= 0 || nt <= 0 ||
        input.a.size() != static_cast<std::size_t>(ni) * ni ||
        input.b.size() != static_cast<std::size_t>(ni) * nt ||
        input.c.size() != static_cast<std::size_t>(nt) * ni ||
        input.d.size() != static_cast<std::size_t>(nt) * nt ||
        input.local_residual.size() != static_cast<std::size_t>(ni) ||
        input.trace_residual.size() != static_cast<std::size_t>(nt) ||
        !Finite(input.a) || !Finite(input.b) || !Finite(input.c) ||
        !Finite(input.d) || !Finite(input.local_residual) ||
        !Finite(input.trace_residual))
    {
      return false;
    }
    DenseLu factor;
    if (!factor.Factor(input.a, ni)) { return false; }
    std::vector<double> a_inverse_residual;
    if (!factor.Solve(input.local_residual, a_inverse_residual))
    {
      return false;
    }
    double maximum_application_residual = ApplicationResidual(
        input.a, ni, a_inverse_residual, input.local_residual);
    if (maximum_application_residual > local_absolute_tolerance +
            local_relative_tolerance * VectorNorm(input.local_residual))
    {
      return false;
    }
    std::vector<double> a_inverse_b(
        static_cast<std::size_t>(ni) * nt, 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(ni), 0.0);
    std::vector<double> solution;
    for (int trace_column = 0; trace_column < nt; ++trace_column)
    {
      for (int row = 0; row < ni; ++row)
      {
        rhs[static_cast<std::size_t>(row)] = input.b[
            static_cast<std::size_t>(row) * nt + trace_column];
      }
      if (!factor.Solve(rhs, solution)) { return false; }
      const double residual = ApplicationResidual(input.a, ni, solution, rhs);
      maximum_application_residual =
          std::max(maximum_application_residual, residual);
      if (residual > local_absolute_tolerance +
              local_relative_tolerance * VectorNorm(rhs))
      {
        return false;
      }
      for (int row = 0; row < ni; ++row)
      {
        a_inverse_b[static_cast<std::size_t>(row) * nt + trace_column] =
            solution[static_cast<std::size_t>(row)];
      }
    }
    output.element = input.element;
    output.local_trace_unknowns = nt;
    output.schur.assign(static_cast<std::size_t>(nt) * nt, 0.0);
    output.right_hand_side.assign(static_cast<std::size_t>(nt), 0.0);
    output.maximum_local_application_residual = maximum_application_residual;
    for (int row = 0; row < nt; ++row)
    {
      double c_a_inverse_r = 0.0;
      for (int inner = 0; inner < ni; ++inner)
      {
        c_a_inverse_r += input.c[static_cast<std::size_t>(row) * ni + inner] *
            a_inverse_residual[static_cast<std::size_t>(inner)];
      }
      output.right_hand_side[static_cast<std::size_t>(row)] =
          -input.trace_residual[static_cast<std::size_t>(row)] +
          c_a_inverse_r;
      for (int column = 0; column < nt; ++column)
      {
        double c_a_inverse_b = 0.0;
        for (int inner = 0; inner < ni; ++inner)
        {
          c_a_inverse_b += input.c[
              static_cast<std::size_t>(row) * ni + inner] *
              a_inverse_b[static_cast<std::size_t>(inner) * nt + column];
        }
        output.schur[static_cast<std::size_t>(row) * nt + column] =
            input.d[static_cast<std::size_t>(row) * nt + column] -
            c_a_inverse_b;
      }
    }
    return Finite(output.schur) && Finite(output.right_hand_side);
  }
  catch (...)
  {
    output = {};
    return false;
  }
}

struct PersistentSc25ScalarIgrCpuFoundation::Implementation
{
  ~Implementation() noexcept;
  struct OwnedCommunicator final
  {
    MPI_Comm value = MPI_COMM_NULL;
    ~OwnedCommunicator() noexcept
    {
      if (value == MPI_COMM_NULL) { return; }
      int finalized = 0;
      MPI_Finalized(&finalized);
      if (finalized == 0) { MPI_Comm_free(&value); }
    }
  } communicator;

  FoundationConfig config{};
  FoundationStatus status = FoundationStatus::uninitialized;
  int rank = -1;
  int ranks = 0;

  std::unique_ptr<mfem::L2_FECollection> volume_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParMesh> mesh;

  std::unique_ptr<mfem::ParFiniteElementSpace> u_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> sigma_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> z_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> uhat_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> sigmahat_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> carrier_space;

  std::unique_ptr<mfem::ParGridFunction> u;
  std::unique_ptr<mfem::ParGridFunction> sigma;
  std::unique_ptr<mfem::ParGridFunction> z;
  std::unique_ptr<mfem::ParGridFunction> uhat;
  std::unique_ptr<mfem::ParGridFunction> sigmahat;
  std::unique_ptr<mfem::ParGridFunction> carrier;

  trace::DGInterfaceTraceMap carrier_map;
  std::vector<ElementTopology> topology;

  std::unique_ptr<mfem::SparseMatrix> assembled_local_matrix;
  std::unique_ptr<mfem::HypreParMatrix> assembled_true_matrix;
  mfem::Vector assembled_true_rhs;
  TraceAssemblyIdentity assembled_identity{};
  std::unique_ptr<PersistentTrajectoryImplementation> trajectory;

  bool BuildMeshAndSpaces() noexcept
  {
    try
    {
      mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
          config.elements_per_direction, config.elements_per_direction,
          mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
      std::vector<mfem::Vector> translations(
          dimension, mfem::Vector(dimension));
      translations[0][0] = 1.0;
      translations[0][1] = 0.0;
      translations[1][0] = 0.0;
      translations[1][1] = 1.0;
      auto periodic = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
          original, original.CreatePeriodicVertexMapping(translations)));
      std::vector<int> partition(
          static_cast<std::size_t>(periodic->GetNE()), 0);
      volume_collection = std::make_unique<mfem::L2_FECollection>(
          config.polynomial_degree, dimension,
          mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
      trace_collection =
          std::make_unique<mfem::DG_Interface_FECollection>(
              config.polynomial_degree, dimension,
              mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
      mesh = std::make_unique<mfem::ParMesh>(
          communicator.value, *periodic, partition.data());
      u_space = std::make_unique<mfem::ParFiniteElementSpace>(
          mesh.get(), volume_collection.get(), state_components,
          mfem::Ordering::byNODES);
      sigma_space = std::make_unique<mfem::ParFiniteElementSpace>(
          mesh.get(), volume_collection.get(), 1, mfem::Ordering::byNODES);
      z_space = std::make_unique<mfem::ParFiniteElementSpace>(
          mesh.get(), volume_collection.get(), flux_components,
          mfem::Ordering::byNODES);
      uhat_space = std::make_unique<mfem::ParFiniteElementSpace>(
          mesh.get(), trace_collection.get(), state_components,
          mfem::Ordering::byNODES);
      sigmahat_space = std::make_unique<mfem::ParFiniteElementSpace>(
          mesh.get(), trace_collection.get(), 1, mfem::Ordering::byNODES);
      carrier_space = std::make_unique<mfem::ParFiniteElementSpace>(
          mesh.get(), trace_collection.get(), trace_components,
          mfem::Ordering::byNODES);
      if (!ExactSpace(u_space.get(), mesh.get(), volume_collection.get(), 4) ||
          !ExactSpace(sigma_space.get(), mesh.get(), volume_collection.get(),
                      1) ||
          !ExactSpace(z_space.get(), mesh.get(), volume_collection.get(), 2) ||
          !ExactSpace(uhat_space.get(), mesh.get(), trace_collection.get(),
                      4) ||
          !ExactSpace(sigmahat_space.get(), mesh.get(),
                      trace_collection.get(), 1) ||
          !ExactSpace(carrier_space.get(), mesh.get(), trace_collection.get(),
                      5))
      {
        return false;
      }
      u = std::make_unique<mfem::ParGridFunction>(u_space.get());
      sigma = std::make_unique<mfem::ParGridFunction>(sigma_space.get());
      z = std::make_unique<mfem::ParGridFunction>(z_space.get());
      uhat = std::make_unique<mfem::ParGridFunction>(uhat_space.get());
      sigmahat =
          std::make_unique<mfem::ParGridFunction>(sigmahat_space.get());
      carrier = std::make_unique<mfem::ParGridFunction>(carrier_space.get());
      if (!ExactField(u.get(), u_space.get()) ||
          !ExactField(sigma.get(), sigma_space.get()) ||
          !ExactField(z.get(), z_space.get()) ||
          !ExactField(uhat.get(), uhat_space.get()) ||
          !ExactField(sigmahat.get(), sigmahat_space.get()) ||
          !ExactField(carrier.get(), carrier_space.get()))
      {
        return false;
      }
      *u = 0.0;
      *sigma = 0.0;
      *z = 0.0;
      *uhat = 0.0;
      *sigmahat = 0.0;
      *carrier = 0.0;
      return mesh->Dimension() == dimension &&
          mesh->SpaceDimension() == dimension &&
          mesh->GetNE() == config.elements_per_direction *
              config.elements_per_direction;
    }
    catch (...)
    {
      return false;
    }
  }

  bool BuildTopology() noexcept
  {
    try
    {
      if (!trace::BuildDGInterfaceTraceMap(*carrier_space, carrier_map) ||
          !carrier_map.IsConsistent())
      {
        return false;
      }
      topology.clear();
      topology.resize(static_cast<std::size_t>(mesh->GetNE()));
      for (int element = 0; element < mesh->GetNE(); ++element)
      {
        ElementTopology& item = topology[static_cast<std::size_t>(element)];
        item.element = element;
        mfem::DofTransformation u_transform(
            u_space->GetVDim(), u_space->GetOrdering());
        mfem::DofTransformation sigma_transform(
            sigma_space->GetVDim(), sigma_space->GetOrdering());
        mfem::DofTransformation z_transform(
            z_space->GetVDim(), z_space->GetOrdering());
        mfem::Array<int> u_dofs, sigma_dofs, z_dofs;
        u_space->GetElementVDofs(element, u_dofs, u_transform);
        sigma_space->GetElementVDofs(element, sigma_dofs, sigma_transform);
        z_space->GetElementVDofs(element, z_dofs, z_transform);
        item.volume.u = CopyArray(u_dofs);
        item.volume.sigma = CopyArray(sigma_dofs);
        item.volume.z = CopyArray(z_dofs);
        item.volume.u_transformation_identity = u_transform.IsIdentity();
        item.volume.sigma_transformation_identity =
            sigma_transform.IsIdentity();
        item.volume.z_transformation_identity = z_transform.IsIdentity();
        const mfem::FiniteElement* volume_element =
            sigma_space->GetFE(element);
        if (volume_element == nullptr) { return false; }
        item.scalar_volume_dofs = volume_element->GetDof();
        if (u_dofs.Size() != state_components * item.scalar_volume_dofs ||
            sigma_dofs.Size() != item.scalar_volume_dofs ||
            z_dofs.Size() != flux_components * item.scalar_volume_dofs)
        {
          return false;
        }
        mfem::Array<int> faces, orientations;
        mesh->GetElementEdges(element, faces, orientations);
        if (faces.Size() != faces_per_quadrilateral ||
            orientations.Size() != faces.Size())
        {
          return false;
        }
        item.faces = CopyArray(faces);
        item.face_orientations = CopyArray(orientations);
        const mfem::FiniteElement* first_face_element =
            carrier_space->GetFaceElement(faces[0]);
        if (first_face_element == nullptr) { return false; }
        item.scalar_face_dofs = first_face_element->GetDof();
        const int scalar_face_total = faces.Size() * item.scalar_face_dofs;
        item.local_interior_unknowns =
            local_components * item.scalar_volume_dofs;
        item.local_trace_unknowns = trace_components * scalar_face_total;
        item.signed_trace_vdofs.assign(
            static_cast<std::size_t>(item.local_trace_unknowns), 0);
        item.restriction.assign(
            static_cast<std::size_t>(item.local_trace_unknowns), {});
        for (int local_face = 0; local_face < faces.Size(); ++local_face)
        {
          const int face = faces[local_face];
          const mfem::FiniteElement* face_element =
              carrier_space->GetFaceElement(face);
          mfem::Array<int> face_vdofs;
          carrier_space->GetFaceVDofs(face, face_vdofs);
          if (face_element == nullptr ||
              face_element->GetDof() != item.scalar_face_dofs ||
              face_vdofs.Size() !=
                  trace_components * item.scalar_face_dofs)
          {
            return false;
          }
          const int orientation = orientations[local_face];
          const int* permutation = trace_collection->DofOrderForOrientation(
              mfem::Geometry::SEGMENT, orientation);
          if (permutation == nullptr) { return false; }
          for (int component = 0; component < trace_components; ++component)
            for (int oriented_face_dof = 0;
                 oriented_face_dof < item.scalar_face_dofs;
                 ++oriented_face_dof)
            {
              int orientation_sign = 0;
              const int canonical_face_dof = DecodeSignedDof(
                  permutation[oriented_face_dof], orientation_sign);
              if (canonical_face_dof < 0 ||
                  canonical_face_dof >= item.scalar_face_dofs ||
                  orientation_sign == 0)
              {
                return false;
              }
              const int canonical_vdof =
                  component * item.scalar_face_dofs + canonical_face_dof;
              int carrier_sign = 0;
              const int decoded_carrier_vdof = DecodeSignedDof(
                  face_vdofs[canonical_vdof], carrier_sign);
              const int sign = orientation_sign * carrier_sign;
              if (decoded_carrier_vdof < 0 ||
                  decoded_carrier_vdof >= carrier_space->GetVSize() ||
                  sign == 0 ||
                  static_cast<std::size_t>(decoded_carrier_vdof) >=
                      carrier_map.dofs.entries.size())
              {
                return false;
              }
              const int local_row = component * scalar_face_total +
                  local_face * item.scalar_face_dofs + oriented_face_dof;
              const int signed_vdof = EncodeSignedDof(
                  decoded_carrier_vdof, sign);
              const trace::TraceMapEntry& map_entry =
                  carrier_map.dofs.entries[
                      static_cast<std::size_t>(decoded_carrier_vdof)];
              item.signed_trace_vdofs[static_cast<std::size_t>(local_row)] =
                  signed_vdof;
              item.restriction[static_cast<std::size_t>(local_row)] = {
                  element, local_row, local_face, component,
                  oriented_face_dof, canonical_face_dof, signed_vdof,
                  decoded_carrier_vdof, sign, map_entry.global_true_dof,
                  map_entry.local_true_dof, map_entry.owner_rank};
            }
        }
      }
      return true;
    }
    catch (...)
    {
      return false;
    }
  }
};

PersistentSc25ScalarIgrCpuFoundation::PersistentSc25ScalarIgrCpuFoundation(
    std::unique_ptr<Implementation> implementation) noexcept
    : implementation_(std::move(implementation))
{}

PersistentSc25ScalarIgrCpuFoundation::~PersistentSc25ScalarIgrCpuFoundation()
    noexcept = default;

PersistentSc25ScalarIgrCpuFoundation::PersistentSc25ScalarIgrCpuFoundation(
    PersistentSc25ScalarIgrCpuFoundation&&) noexcept = default;

FoundationStatus PersistentSc25ScalarIgrCpuFoundation::Create(
    MPI_Comm communicator, CpuSerialTag, const FoundationConfig& config,
    std::unique_ptr<PersistentSc25ScalarIgrCpuFoundation>& output) noexcept
{
  output.reset();
  if (communicator == MPI_COMM_NULL || config.polynomial_degree < 1 ||
      config.polynomial_degree > 4 ||
      config.elements_per_direction < 3 ||
      (config.elements_per_direction !=
           coupled_fixture_elements_per_direction &&
       config.elements_per_direction != 8 &&
       config.elements_per_direction != 16 &&
       config.elements_per_direction != 32))
  {
    return FoundationStatus::invalid_argument;
  }
  try
  {
    auto implementation = std::make_unique<Implementation>();
    implementation->config = config;
    if (MPI_Comm_dup(communicator, &implementation->communicator.value) !=
            MPI_SUCCESS ||
        MPI_Comm_rank(implementation->communicator.value,
                      &implementation->rank) != MPI_SUCCESS ||
        MPI_Comm_size(implementation->communicator.value,
                      &implementation->ranks) != MPI_SUCCESS ||
        implementation->ranks != 1 || implementation->rank != 0)
    {
      return FoundationStatus::unsupported_communicator;
    }
    if (!implementation->BuildMeshAndSpaces())
    {
      return FoundationStatus::space_failure;
    }
    if (!implementation->BuildTopology())
    {
      return FoundationStatus::topology_failure;
    }
    implementation->status = FoundationStatus::ready;
    output.reset(new (std::nothrow) PersistentSc25ScalarIgrCpuFoundation(
        std::move(implementation)));
    return output ? FoundationStatus::ready
                  : FoundationStatus::allocation_failure;
  }
  catch (const std::bad_alloc&)
  {
    output.reset();
    return FoundationStatus::allocation_failure;
  }
  catch (...)
  {
    output.reset();
    return FoundationStatus::space_failure;
  }
}

FoundationStatus PersistentSc25ScalarIgrCpuFoundation::status() const noexcept
{
  return implementation_ ? implementation_->status
                         : FoundationStatus::uninitialized;
}

const FoundationConfig&
PersistentSc25ScalarIgrCpuFoundation::config() const noexcept
{
  static const FoundationConfig empty{};
  return implementation_ ? implementation_->config : empty;
}

const mfem::ParMesh*
PersistentSc25ScalarIgrCpuFoundation::mesh() const noexcept
{
  return implementation_ ? implementation_->mesh.get() : nullptr;
}

const mfem::L2_FECollection*
PersistentSc25ScalarIgrCpuFoundation::volume_collection() const noexcept
{
  return implementation_ ? implementation_->volume_collection.get() : nullptr;
}

const mfem::DG_Interface_FECollection*
PersistentSc25ScalarIgrCpuFoundation::trace_collection() const noexcept
{
  return implementation_ ? implementation_->trace_collection.get() : nullptr;
}

SpaceOwnershipSnapshot
PersistentSc25ScalarIgrCpuFoundation::spaces() const noexcept
{
  if (!implementation_) { return {}; }
  return {DescribeSpace(implementation_->u_space.get()),
          DescribeSpace(implementation_->sigma_space.get()),
          DescribeSpace(implementation_->z_space.get()),
          DescribeSpace(implementation_->uhat_space.get()),
          DescribeSpace(implementation_->sigmahat_space.get()),
          DescribeSpace(implementation_->carrier_space.get())};
}

FieldOwnershipView PersistentSc25ScalarIgrCpuFoundation::fields() const noexcept
{
  if (!implementation_) { return {}; }
  return {implementation_->u.get(), implementation_->sigma.get(),
          implementation_->z.get(), implementation_->uhat.get(),
          implementation_->sigmahat.get(), implementation_->carrier.get()};
}

const trace::DGInterfaceTraceMap&
PersistentSc25ScalarIgrCpuFoundation::carrier_trace_map() const noexcept
{
  static const trace::DGInterfaceTraceMap empty{};
  return implementation_ ? implementation_->carrier_map : empty;
}

const std::vector<ElementTopology>&
PersistentSc25ScalarIgrCpuFoundation::element_topology() const noexcept
{
  static const std::vector<ElementTopology> empty{};
  return implementation_ ? implementation_->topology : empty;
}

bool PersistentSc25ScalarIgrCpuFoundation::GatherElementFields(
    const int element, ElementFieldCoefficients& output) const noexcept
{
  output = {};
  if (!implementation_ || element < 0 ||
      element >= implementation_->mesh->GetNE())
  {
    return false;
  }
  try
  {
    const auto gather = [element](const mfem::ParFiniteElementSpace& space,
                                  const mfem::ParGridFunction& field,
                                  std::vector<double>& destination)
    {
      mfem::Array<int> vdofs;
      mfem::DofTransformation transformation(space.GetVDim(),
                                              space.GetOrdering());
      space.GetElementVDofs(element, vdofs, transformation);
      mfem::Vector local(vdofs.Size());
      field.GetSubVector(vdofs, local);
      transformation.InvTransformPrimal(local);
      const double* values = local.HostRead();
      destination.assign(values, values + local.Size());
      return Finite(destination);
    };
    return gather(*implementation_->u_space, *implementation_->u, output.u) &&
        gather(*implementation_->sigma_space, *implementation_->sigma,
               output.sigma) &&
        gather(*implementation_->z_space, *implementation_->z, output.z);
  }
  catch (...)
  {
    output = {};
    return false;
  }
}

bool PersistentSc25ScalarIgrCpuFoundation::AddElementDualResidual(
    const int element, const ElementFieldCoefficients& local_dual,
    mfem::ParGridFunction& u_residual,
    mfem::ParGridFunction& sigma_residual,
    mfem::ParGridFunction& z_residual) const noexcept
{
  if (!implementation_ || element < 0 ||
      element >= implementation_->mesh->GetNE() ||
      u_residual.ParFESpace() != implementation_->u_space.get() ||
      sigma_residual.ParFESpace() != implementation_->sigma_space.get() ||
      z_residual.ParFESpace() != implementation_->z_space.get())
  {
    return false;
  }
  try
  {
    const auto validate = [element](
                              const mfem::ParFiniteElementSpace& space,
                              const std::vector<double>& values)
    {
      mfem::Array<int> vdofs;
      mfem::DofTransformation transformation(space.GetVDim(),
                                              space.GetOrdering());
      space.GetElementVDofs(element, vdofs, transformation);
      return values.size() == static_cast<std::size_t>(vdofs.Size()) &&
          Finite(values);
    };
    if (!validate(*implementation_->u_space, local_dual.u) ||
        !validate(*implementation_->sigma_space, local_dual.sigma) ||
        !validate(*implementation_->z_space, local_dual.z))
    {
      return false;
    }
    const auto add = [element](const mfem::ParFiniteElementSpace& space,
                               const std::vector<double>& values,
                               mfem::ParGridFunction& destination)
    {
      mfem::Array<int> vdofs;
      mfem::DofTransformation transformation(space.GetVDim(),
                                              space.GetOrdering());
      space.GetElementVDofs(element, vdofs, transformation);
      mfem::Vector local(vdofs.Size());
      std::copy(values.begin(), values.end(), local.HostWrite());
      transformation.TransformDual(local);
      destination.AddElementVector(vdofs, local);
    };
    add(*implementation_->u_space, local_dual.u, u_residual);
    add(*implementation_->sigma_space, local_dual.sigma, sigma_residual);
    add(*implementation_->z_space, local_dual.z, z_residual);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentSc25ScalarIgrCpuFoundation::
    PackTraceCarrierFromPhysicalFields() noexcept
{
  if (!implementation_) { return false; }
  try
  {
    const int scalar_size = implementation_->sigmahat_space->GetVSize();
    if (implementation_->uhat_space->GetVSize() !=
            state_components * scalar_size ||
        implementation_->carrier_space->GetVSize() !=
            trace_components * scalar_size)
    {
      return false;
    }
    const double* uhat = implementation_->uhat->HostRead();
    const double* sigmahat = implementation_->sigmahat->HostRead();
    double* carrier = implementation_->carrier->HostWrite();
    if ((!uhat || !sigmahat || !carrier) && scalar_size != 0) { return false; }
    std::copy_n(uhat, state_components * scalar_size, carrier);
    std::copy_n(sigmahat, scalar_size,
                carrier + state_components * scalar_size);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentSc25ScalarIgrCpuFoundation::
    UnpackPhysicalTraceFromCarrier() noexcept
{
  if (!implementation_) { return false; }
  try
  {
    const int scalar_size = implementation_->sigmahat_space->GetVSize();
    if (implementation_->uhat_space->GetVSize() !=
            state_components * scalar_size ||
        implementation_->carrier_space->GetVSize() !=
            trace_components * scalar_size)
    {
      return false;
    }
    const double* carrier = implementation_->carrier->HostRead();
    double* uhat = implementation_->uhat->HostWrite();
    double* sigmahat = implementation_->sigmahat->HostWrite();
    if ((!uhat || !sigmahat || !carrier) && scalar_size != 0) { return false; }
    std::copy_n(carrier, state_components * scalar_size, uhat);
    std::copy_n(carrier + state_components * scalar_size, scalar_size,
                sigmahat);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentSc25ScalarIgrCpuFoundation::SetTraceCarrierFromTrueDofs(
    const mfem::Vector& true_values) noexcept
{
  if (!implementation_ ||
      true_values.Size() != implementation_->carrier_space->GetTrueVSize())
  {
    return false;
  }
  try
  {
    const double* values = true_values.HostRead();
    if (values == nullptr && true_values.Size() != 0) { return false; }
    for (int entry = 0; entry < true_values.Size(); ++entry)
    {
      if (!Finite(values[entry])) { return false; }
    }
    implementation_->carrier->SetFromTrueDofs(true_values);
    return UnpackPhysicalTraceFromCarrier();
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentSc25ScalarIgrCpuFoundation::AssembleTrueTraceSystem(
    const std::vector<ElementCondensedBlock>& blocks) noexcept
{
  if (!implementation_ ||
      blocks.size() != implementation_->topology.size())
  {
    return false;
  }
  try
  {
    for (std::size_t element = 0; element < blocks.size(); ++element)
    {
      const ElementCondensedBlock& block = blocks[element];
      const ElementTopology& topology = implementation_->topology[element];
      const int nt = topology.local_trace_unknowns;
      if (nt <= 0 || block.element != topology.element ||
          block.local_trace_unknowns != nt ||
          block.schur.size() != static_cast<std::size_t>(nt) * nt ||
          block.right_hand_side.size() != static_cast<std::size_t>(nt) ||
          topology.signed_trace_vdofs.size() !=
              static_cast<std::size_t>(nt) ||
          !Finite(block.schur) || !Finite(block.right_hand_side))
      {
        return false;
      }
      for (const int signed_dof : topology.signed_trace_vdofs)
      {
        int sign = 0;
        const int dof = DecodeSignedDof(signed_dof, sign);
        if (dof < 0 || dof >= implementation_->carrier_space->GetVSize() ||
            sign == 0)
        {
          return false;
        }
      }
    }
    if (implementation_->assembled_identity.successful_generation ==
        std::numeric_limits<std::uint64_t>::max())
    {
      return false;
    }
    auto local_matrix = std::make_unique<mfem::SparseMatrix>(
        implementation_->carrier_space->GetVSize());
    mfem::ParGridFunction local_residual(
        implementation_->carrier_space.get());
    local_residual = 0.0;
    for (std::size_t element = 0; element < blocks.size(); ++element)
    {
      const ElementCondensedBlock& block = blocks[element];
      const ElementTopology& topology = implementation_->topology[element];
      const int nt = topology.local_trace_unknowns;
      mfem::Array<int> signed_dofs(nt);
      mfem::DenseMatrix schur(nt);
      mfem::Vector rhs(nt);
      for (int row = 0; row < nt; ++row)
      {
        signed_dofs[row] = topology.signed_trace_vdofs[
            static_cast<std::size_t>(row)];
        rhs[row] = block.right_hand_side[static_cast<std::size_t>(row)];
        for (int column = 0; column < nt; ++column)
        {
          schur(row, column) = block.schur[
              static_cast<std::size_t>(row) * nt + column];
        }
      }
      local_matrix->AddSubMatrix(signed_dofs, signed_dofs, schur, 1);
      local_residual.AddElementVector(signed_dofs, rhs);
    }
    local_matrix->Finalize();
    mfem::ParBilinearForm form(implementation_->carrier_space.get());
    std::unique_ptr<mfem::HypreParMatrix> true_matrix(
        form.ParallelAssemble(local_matrix.get()));
    mfem::Vector true_rhs(implementation_->carrier_space->GetTrueVSize());
    local_residual.ParallelAssemble(true_rhs);
    if (!true_matrix ||
        true_matrix->Height() != implementation_->carrier_space->GetTrueVSize() ||
        true_matrix->GetGlobalNumRows() !=
            implementation_->carrier_space->GlobalTrueVSize() ||
        true_rhs.Size() != implementation_->carrier_space->GetTrueVSize())
    {
      return false;
    }
    const std::uint64_t next_generation =
        implementation_->assembled_identity.successful_generation + 1;
    const TraceAssemblyIdentity next_identity = {
        next_generation, true_matrix->Height(), true_matrix->Width(),
        true_rhs.Size(), true_matrix->GetGlobalNumRows(),
        static_cast<std::size_t>(local_matrix->NumNonZeroElems())};
    implementation_->assembled_local_matrix.swap(local_matrix);
    implementation_->assembled_true_matrix.swap(true_matrix);
    implementation_->assembled_true_rhs.Swap(true_rhs);
    implementation_->assembled_identity = next_identity;
    return true;
  }
  catch (...)
  {
    return false;
  }
}

const mfem::HypreParMatrix*
PersistentSc25ScalarIgrCpuFoundation::assembled_true_trace_matrix() const
    noexcept
{
  return implementation_ ? implementation_->assembled_true_matrix.get()
                         : nullptr;
}

const mfem::Vector*
PersistentSc25ScalarIgrCpuFoundation::
    assembled_true_trace_right_hand_side() const noexcept
{
  return implementation_ && implementation_->assembled_true_matrix
      ? &implementation_->assembled_true_rhs : nullptr;
}

TraceAssemblyIdentity
PersistentSc25ScalarIgrCpuFoundation::assembled_trace_identity() const noexcept
{
  return implementation_ ? implementation_->assembled_identity
                         : TraceAssemblyIdentity{};
}

namespace
{

struct CoupledFixturePrivateDiagnostic final
{
  const char* fixture_label = "C1-MFEM-PERIODIC-02";
  const char* phase = "precondition";
  const char* substage = "owner-config";
  int element = -1;
  bool hypre_setup_reached = false;
  bool hypre_solve_reached = false;
  bool hypre_query_reached = false;
  bool iteration_query_reached = false;
  bool true_residual_reached = false;
  int setup_status = -1;
  int solve_status = -1;
  int convergence_query_status = -1;
  int converged = -1;
  int iterations = -1;
  double true_residual_norm = std::numeric_limits<double>::quiet_NaN();
  double right_hand_side_norm = std::numeric_limits<double>::quiet_NaN();
  double true_residual_bound = std::numeric_limits<double>::quiet_NaN();
  int expected_block_element = -1;
  int actual_block_element = -1;
  int block_interior_unknowns = -1;
  int block_trace_unknowns = -1;
  std::size_t block_a_entries = 0;
  std::size_t block_b_entries = 0;
  std::size_t block_c_entries = 0;
  std::size_t block_d_entries = 0;
  std::size_t block_local_residual_entries = 0;
  std::size_t block_trace_residual_entries = 0;
  int lu_failure_column = -1;
  double lu_pivot_magnitude = std::numeric_limits<double>::quiet_NaN();
  double lu_pivot_threshold = std::numeric_limits<double>::quiet_NaN();
  const char* lu_failure_reason = "not-reached";
  const char* local_solve_kind = "not-reached";
  int local_solve_trace_column = -1;
  bool local_solve_reached = false;
  bool local_solve_succeeded = false;
  bool local_application_reached = false;
  double local_application_residual =
      std::numeric_limits<double>::quiet_NaN();
  double local_right_hand_side_norm =
      std::numeric_limits<double>::quiet_NaN();
  double local_application_bound =
      std::numeric_limits<double>::quiet_NaN();
  bool attempt_active = false;
  bool failure_latched = false;
  TrajectoryAssemblyFailureCause failure_cause =
      TrajectoryAssemblyFailureCause::invalid_structure;

  void Begin(const char* next_fixture_label, const char* next_phase,
             const char* next_substage = "precondition") noexcept
  {
    if (failure_latched) { return; }
    fixture_label = next_fixture_label;
    attempt_active = true;
    phase = next_phase;
    substage = next_substage;
    element = -1;
    failure_cause = TrajectoryAssemblyFailureCause::invalid_structure;
  }

  void SetFailureCause(
      const TrajectoryAssemblyFailureCause next_cause) noexcept
  {
    if (!failure_latched) { failure_cause = next_cause; }
  }

  void Set(const char* next_phase, const char* next_substage,
           const int next_element = -1) noexcept
  {
    if (failure_latched) { return; }
    attempt_active = true;
    phase = next_phase;
    substage = next_substage;
    element = next_element;
  }

  void LatchFailure() noexcept
  {
    if (attempt_active) { failure_latched = true; }
    attempt_active = false;
  }

  void CompleteAttempt() noexcept
  {
    if (!failure_latched) { attempt_active = false; }
  }

  void HandleFailure() noexcept
  {
    attempt_active = false;
    failure_latched = false;
  }

  ~CoupledFixturePrivateDiagnostic() noexcept
  {
    if (!failure_latched && !attempt_active) { return; }
    std::fprintf(
        stderr,
        "%s diagnostic phase=%s substage=%s element=%d "
        "hypre_setup_reached=%d setup_status=%d "
        "hypre_solve_reached=%d solve_status=%d "
        "hypre_query_reached=%d convergence_query_status=%d converged=%d "
        "iteration_query_reached=%d iterations=%d "
        "true_residual_reached=%d true_residual=%.17g rhs=%.17g "
        "bound=%.17g\n",
        fixture_label, phase, substage, element,
        hypre_setup_reached ? 1 : 0,
        setup_status, hypre_solve_reached ? 1 : 0, solve_status,
        hypre_query_reached ? 1 : 0, convergence_query_status, converged,
        iteration_query_reached ? 1 : 0, iterations,
        true_residual_reached ? 1 : 0, true_residual_norm,
        right_hand_side_norm, true_residual_bound);
    std::fprintf(
        stderr,
        "%s local-detail expected_element=%d "
        "actual_element=%d ni=%d nt=%d A=%zu B=%zu C=%zu D=%zu "
        "r_local=%zu r_trace=%zu lu_failure_column=%d "
        "lu_pivot_magnitude=%.17g lu_pivot_threshold=%.17g "
        "lu_reason=%s solve_kind=%s trace_column=%d solve_reached=%d "
        "solve_succeeded=%d application_reached=%d "
        "application_residual=%.17g rhs_norm=%.17g bound=%.17g\n",
        fixture_label, expected_block_element, actual_block_element,
        block_interior_unknowns, block_trace_unknowns, block_a_entries,
        block_b_entries, block_c_entries, block_d_entries,
        block_local_residual_entries, block_trace_residual_entries,
        lu_failure_column, lu_pivot_magnitude, lu_pivot_threshold,
        lu_failure_reason, local_solve_kind, local_solve_trace_column,
        local_solve_reached ? 1 : 0, local_solve_succeeded ? 1 : 0,
        local_application_reached ? 1 : 0, local_application_residual,
        local_right_hand_side_norm, local_application_bound);
  }
};

struct CoupledFixtureState final
{
  std::vector<double> old_interior;
  std::vector<double> old_true_trace;
  std::vector<double> candidate_interior;
  std::vector<double> candidate_true_trace;
  std::vector<std::vector<double>> old_by_element;
  std::vector<std::vector<double>> candidate_by_element;
  std::vector<std::vector<double>> old_trace_by_element;
  std::vector<std::vector<double>> trace_by_element;
};

struct CoupledFixtureAssembly final
{
  std::vector<ElementLinearization> elements;
  std::vector<double> full_residual;
  std::vector<double> full_jacobian;
};

struct CoupledAssemblyWorkspace final
{
  mfem::Vector volume_shape;
  mfem::DenseMatrix volume_derivative;
  mfem::Vector face_interior_shape;
  mfem::Vector face_trace_shape;
  mfem::Vector face_oriented_shape;
  mfem::Vector scaled_normal;

  CoupledAssemblyWorkspace()
      : volume_shape(coupled_fixture_scalar_volume_nodes),
        volume_derivative(coupled_fixture_scalar_volume_nodes, dimension),
        face_interior_shape(coupled_fixture_scalar_volume_nodes),
        face_trace_shape(coupled_fixture_scalar_face_nodes),
        face_oriented_shape(coupled_fixture_scalar_face_nodes),
        scaled_normal(dimension)
  {}
};

template <typename Value>
void ResizeAndZero(std::vector<Value>& values, const std::size_t size)
{
  if (values.size() != size) { values.resize(size); }
  std::fill(values.begin(), values.end(), Value{});
}

double RelativeL2Difference(const std::vector<double>& left,
                            const std::vector<double>& right) noexcept
{
  if (left.size() != right.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  long double difference_square = 0.0L;
  long double right_square = 0.0L;
  for (std::size_t entry = 0; entry < left.size(); ++entry)
  {
    const long double difference =
        static_cast<long double>(left[entry]) - right[entry];
    difference_square += difference * difference;
    right_square += static_cast<long double>(right[entry]) * right[entry];
  }
  return std::sqrt(static_cast<double>(difference_square)) /
      std::max(1.0, std::sqrt(static_cast<double>(right_square)));
}

double EvaluateComponent(const std::vector<double>& coefficients,
                         const int component, const int scalar_dofs,
                         const mfem::Vector& shape) noexcept
{
  double value = 0.0;
  for (int scalar = 0; scalar < scalar_dofs; ++scalar)
  {
    value += coefficients[static_cast<std::size_t>(component) * scalar_dofs +
                          scalar] * shape[scalar];
  }
  return value;
}

double EvaluateDerivative(const std::vector<double>& coefficients,
                          const int component, const int scalar_dofs,
                          const mfem::DenseMatrix& derivative,
                          const int direction) noexcept
{
  double value = 0.0;
  for (int scalar = 0; scalar < scalar_dofs; ++scalar)
  {
    value += coefficients[static_cast<std::size_t>(component) * scalar_dofs +
                          scalar] * derivative(scalar, direction);
  }
  return value;
}

bool SampleAdmissible(const State& conservative) noexcept
{
  double pressure = 0.0;
  std::array<double, dimension> velocity{};
  if (!Primitive(conservative, pressure, velocity) ||
      conservative[0] < 1.0e-8 || pressure < 1.0e-8)
  {
    return false;
  }
  const double speed = std::sqrt(velocity[0] * velocity[0] +
                                 velocity[1] * velocity[1]);
  const double sound = std::sqrt(gamma * pressure / conservative[0]);
  return Finite(speed) && Finite(sound) && speed + sound < 2.5;
}

double ElementMeasureAtOrder(mfem::ParMesh& mesh, const int element,
                             const int order)
{
  mfem::ElementTransformation* transformation =
      mesh.GetElementTransformation(element);
  if (transformation == nullptr) { return 0.0; }
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mesh.GetElementGeometry(element), order);
  double measure = 0.0;
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = rule.IntPoint(point);
    transformation->SetIntPoint(&ip);
    measure += ip.weight * transformation->Weight();
  }
  return measure;
}

double FaceMeasureAtOrder(mfem::ParMesh& mesh, const int face,
                          const int order)
{
  mfem::FaceElementTransformations* transformation =
      mesh.GetFaceElementTransformations(face);
  if (transformation == nullptr || transformation->Face == nullptr)
  {
    return 0.0;
  }
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mesh.GetFaceGeometry(face), order);
  double measure = 0.0;
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = rule.IntPoint(point);
    transformation->Face->SetIntPoint(&ip);
    measure += ip.weight * transformation->Face->Weight();
  }
  return measure;
}

bool MaterializeCoupledFixture(
    mfem::ParMesh& mesh, const mfem::ParFiniteElementSpace& volume_space,
    const mfem::ParFiniteElementSpace& carrier_space,
    mfem::ParGridFunction& carrier,
    const std::vector<ElementTopology>& topology,
    CoupledFixtureState& output,
    CoupledFixturePrivateDiagnostic& diagnostic)
{
  output = {};
  diagnostic.Set("materialize", "precondition");
  try
  {
    const int elements = mesh.GetNE();
    if (elements != coupled_fixture_elements ||
        mesh.GetNumFaces() != coupled_fixture_unique_faces ||
        topology.size() != coupled_fixture_elements ||
        carrier_space.GetTrueVSize() != coupled_fixture_true_trace_unknowns ||
        carrier_space.GlobalTrueVSize() !=
            coupled_fixture_true_trace_unknowns)
    {
      return false;
    }
    diagnostic.Set("materialize", "restriction-census");
    std::vector<int> true_incidence(
        coupled_fixture_true_trace_unknowns, 0);
    int restriction_rows = 0;
    for (int element = 0; element < elements; ++element)
    {
      const ElementTopology& item =
          topology[static_cast<std::size_t>(element)];
      if (item.element != element ||
          item.restriction.size() !=
              coupled_fixture_local_trace_unknowns ||
          item.signed_trace_vdofs.size() !=
              coupled_fixture_local_trace_unknowns)
      {
        return false;
      }
      for (int row = 0; row < coupled_fixture_local_trace_unknowns; ++row)
      {
        const TraceRestrictionEntry& entry =
            item.restriction[static_cast<std::size_t>(row)];
        if (entry.local_row != row || entry.local_true_dof < 0 ||
            entry.local_true_dof >= coupled_fixture_true_trace_unknowns)
        {
          return false;
        }
        ++true_incidence[static_cast<std::size_t>(entry.local_true_dof)];
        ++restriction_rows;
      }
    }
    if (restriction_rows != coupled_fixture_restriction_rows ||
        !std::all_of(true_incidence.begin(), true_incidence.end(),
                     [](const int count) { return count == 2; }))
    {
      return false;
    }
    diagnostic.Set("materialize", "volume-state");
    output.old_by_element.resize(static_cast<std::size_t>(elements));
    output.candidate_by_element.resize(static_cast<std::size_t>(elements));
    output.old_trace_by_element.resize(static_cast<std::size_t>(elements));
    output.trace_by_element.resize(static_cast<std::size_t>(elements));
    output.old_interior.assign(
        coupled_fixture_global_interior_unknowns, 0.0);
    output.candidate_interior.assign(
        coupled_fixture_global_interior_unknowns, 0.0);
    for (int element = 0; element < elements; ++element)
    {
      const mfem::FiniteElement* finite_element = volume_space.GetFE(element);
      if (finite_element == nullptr ||
          finite_element->GetDof() != coupled_fixture_scalar_volume_nodes)
      {
        return false;
      }
      const mfem::IntegrationRule& nodes = finite_element->GetNodes();
      if (nodes.GetNPoints() != coupled_fixture_scalar_volume_nodes)
      {
        return false;
      }
      auto& old = output.old_by_element[static_cast<std::size_t>(element)];
      auto& candidate =
          output.candidate_by_element[static_cast<std::size_t>(element)];
      old.assign(coupled_fixture_local_interior_unknowns, 0.0);
      candidate.assign(coupled_fixture_local_interior_unknowns, 0.0);
      const int ex = element % coupled_fixture_elements_per_direction;
      const int ey = element / coupled_fixture_elements_per_direction;
      const double parity = ((ex + ey) % 2 == 0) ? 1.0 : -1.0;
      for (int scalar = 0; scalar < coupled_fixture_scalar_volume_nodes;
           ++scalar)
      {
        const mfem::IntegrationPoint& node = nodes.IntPoint(scalar);
        const double xi = 2.0 * node.x - 1.0;
        const double eta = 2.0 * node.y - 1.0;
        old[scalar] = 1.0 + 0.008 * parity * xi + 0.004 * eta;
        old[4 + scalar] = 0.09 + 0.006 * eta;
        old[8 + scalar] = -0.04 + 0.004 * xi;
        old[12 + scalar] = 2.505 + 0.001 * parity * xi;
        old[16 + scalar] = 0.018 + 0.003 * xi - 0.001 * eta;
        old[20 + scalar] = 0.002 + 0.001 * eta;
        old[24 + scalar] = -0.001 + 0.0005 * xi;
        candidate[scalar] =
            1.0 + 0.010 * parity * xi + 0.005 * eta;
        candidate[4 + scalar] = 0.10 + 0.010 * eta;
        candidate[8 + scalar] = -0.05 + 0.005 * xi;
        candidate[12 + scalar] = 2.510 + 0.002 * parity * xi;
        candidate[16 + scalar] =
            0.020 + 0.004 * xi - 0.002 * eta;
        candidate[20 + scalar] = 0.003 + 0.001 * eta;
        candidate[24 + scalar] = -0.002 + 0.001 * xi;
      }
      std::copy(old.begin(), old.end(),
                output.old_interior.begin() +
                    static_cast<std::size_t>(element) * old.size());
      std::copy(candidate.begin(), candidate.end(),
                output.candidate_interior.begin() +
                    static_cast<std::size_t>(element) * candidate.size());
    }

    diagnostic.Set("materialize", "trace-state");
    output.old_true_trace.assign(
        coupled_fixture_true_trace_unknowns,
        std::numeric_limits<double>::quiet_NaN());
    output.candidate_true_trace.assign(
        coupled_fixture_true_trace_unknowns,
        std::numeric_limits<double>::quiet_NaN());
    std::vector<unsigned char> assigned(
        coupled_fixture_true_trace_unknowns, 0);
    for (int face = 0; face < mesh.GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* trace_element =
          carrier_space.GetFaceElement(face);
      mfem::Array<int> face_vdofs;
      carrier_space.GetFaceVDofs(face, face_vdofs);
      if (trace_element == nullptr ||
          trace_element->GetDof() != coupled_fixture_scalar_face_nodes ||
          face_vdofs.Size() !=
              trace_components * coupled_fixture_scalar_face_nodes)
      {
        return false;
      }
      const mfem::IntegrationRule& nodes = trace_element->GetNodes();
      if (nodes.GetNPoints() != coupled_fixture_scalar_face_nodes)
      {
        return false;
      }
      for (int component = 0; component < trace_components; ++component)
        for (int scalar = 0; scalar < coupled_fixture_scalar_face_nodes;
             ++scalar)
        {
          int sign = 0;
          const int local = DecodeSignedDof(
              face_vdofs[component * coupled_fixture_scalar_face_nodes +
                          scalar], sign);
          if (local < 0 || local >= carrier_space.GetVSize() || sign == 0)
          {
            return false;
          }
          const int local_true = carrier_space.GetLocalTDofNumber(local);
          if (local_true < 0 ||
              local_true >= coupled_fixture_true_trace_unknowns)
          {
            return false;
          }
          const double s = 2.0 * nodes.IntPoint(scalar).x - 1.0;
          double old_value = 0.0;
          double value = 0.0;
          if (component == 0)
          {
            old_value = 1.0 + 0.001 * s;
            value = 1.0 + 0.002 * s;
          }
          else if (component == 1)
          {
            old_value = 0.09 + 0.002 * s;
            value = 0.095 + 0.003 * s;
          }
          else if (component == 2)
          {
            old_value = -0.04 + 0.001 * s;
            value = -0.045 + 0.002 * s;
          }
          else if (component == 3)
          {
            old_value = 2.505 + 0.0005 * s;
            value = 2.508 + 0.001 * s;
          }
          else
          {
            old_value = 0.018 + 0.001 * s;
            value = 0.020 + 0.003 * s;
          }
          const double old_true_value =
              static_cast<double>(sign) * old_value;
          const double true_value = static_cast<double>(sign) * value;
          if (assigned[static_cast<std::size_t>(local_true)] != 0 &&
              (output.old_true_trace[static_cast<std::size_t>(local_true)] !=
                   old_true_value ||
               output.candidate_true_trace[
                   static_cast<std::size_t>(local_true)] != true_value))
          {
            return false;
          }
          output.old_true_trace[static_cast<std::size_t>(local_true)] =
              old_true_value;
          output.candidate_true_trace[
              static_cast<std::size_t>(local_true)] = true_value;
          assigned[static_cast<std::size_t>(local_true)] = 1;
        }
    }
    if (!std::all_of(assigned.begin(), assigned.end(),
                     [](const unsigned char value) { return value == 1; }))
    {
      return false;
    }
    diagnostic.Set("materialize", "carrier-gather");
    const auto gather_trace = [&](const std::vector<double>& true_values,
                                  std::vector<std::vector<double>>& local)
    {
      mfem::Vector true_trace(coupled_fixture_true_trace_unknowns);
      std::copy(true_values.begin(), true_values.end(), true_trace.HostWrite());
      carrier.SetFromTrueDofs(true_trace);
      for (int element = 0; element < elements; ++element)
      {
        const ElementTopology& item =
            topology[static_cast<std::size_t>(element)];
        if (item.local_trace_unknowns !=
                coupled_fixture_local_trace_unknowns ||
            item.signed_trace_vdofs.size() !=
                coupled_fixture_local_trace_unknowns)
        {
          return false;
        }
        mfem::Array<int> signed_dofs(coupled_fixture_local_trace_unknowns);
        for (int row = 0; row < signed_dofs.Size(); ++row)
        {
          signed_dofs[row] = item.signed_trace_vdofs[
              static_cast<std::size_t>(row)];
        }
        mfem::Vector local_trace(signed_dofs.Size());
        carrier.GetSubVector(signed_dofs, local_trace);
        local[static_cast<std::size_t>(element)].assign(
            local_trace.HostRead(),
            local_trace.HostRead() + local_trace.Size());
      }
      return true;
    };
    if (!gather_trace(output.old_true_trace, output.old_trace_by_element) ||
        !gather_trace(output.candidate_true_trace, output.trace_by_element))
    {
      return false;
    }
    diagnostic.Set("materialize", "finite-check");
    return Finite(output.old_interior) && Finite(output.old_true_trace) &&
        Finite(output.candidate_interior) &&
        Finite(output.candidate_true_trace);
  }
  catch (...)
  {
    diagnostic.Set("materialize", "exception");
    output = {};
    return false;
  }
}

bool AssembleCoupledFixture(
    mfem::ParMesh& mesh, const mfem::ParFiniteElementSpace& volume_space,
    const mfem::ParFiniteElementSpace& carrier_space,
    const mfem::DG_Interface_FECollection& trace_collection,
    const std::vector<ElementTopology>& topology,
    const CoupledFixtureState& state, const int quadrature_order,
    const double time_step, const bool assemble_dense_jacobian,
    CoupledFixtureAssembly& output, CoupledAssemblyWorkspace& workspace,
    const char* phase,
    CoupledFixturePrivateDiagnostic& diagnostic)
{
  diagnostic.Set(phase, "precondition");
  try
  {
    constexpr int ns = coupled_fixture_scalar_volume_nodes;
    constexpr int nf = coupled_fixture_local_scalar_trace_nodes;
    constexpr int ni = coupled_fixture_local_interior_unknowns;
    constexpr int nt = coupled_fixture_local_trace_unknowns;
    const int elements = mesh.GetNE();
    const int true_trace = carrier_space.GetTrueVSize();
    const int global_interior = elements * ni;
    const int full = global_interior + true_trace;
    if (elements <= 0 || true_trace <= 0 || !Finite(time_step) ||
        !(time_step > 0.0) ||
        volume_space.GetVDim() != 1 ||
        carrier_space.GetVDim() != trace_components ||
        state.old_by_element.size() != static_cast<std::size_t>(elements) ||
        state.candidate_by_element.size() !=
            static_cast<std::size_t>(elements) ||
        state.old_trace_by_element.size() !=
            static_cast<std::size_t>(elements) ||
        state.trace_by_element.size() !=
            static_cast<std::size_t>(elements) ||
        topology.size() != static_cast<std::size_t>(elements))
    {
      return false;
    }
    output.elements.resize(static_cast<std::size_t>(elements));
    ResizeAndZero(output.full_residual, static_cast<std::size_t>(full));
    if (assemble_dense_jacobian)
    {
      ResizeAndZero(output.full_jacobian,
                    static_cast<std::size_t>(full) * full);
    }
    else
    {
      output.full_jacobian.clear();
    }
    const auto local_index = [](const int component, const int scalar)
    { return component * ns + scalar; };
    const auto trace_index = [](const int component, const int face_scalar)
    { return component * nf + face_scalar; };

    for (int element = 0; element < elements; ++element)
    {
      diagnostic.Set(phase, "volume", element);
      const std::vector<double>& old =
          state.old_by_element[static_cast<std::size_t>(element)];
      const std::vector<double>& current =
          state.candidate_by_element[static_cast<std::size_t>(element)];
      const std::vector<double>& old_trace =
          state.old_trace_by_element[static_cast<std::size_t>(element)];
      const std::vector<double>& trace =
          state.trace_by_element[static_cast<std::size_t>(element)];
      if (old.size() != ni || current.size() != ni ||
          old_trace.size() != nt || trace.size() != nt)
      {
        return false;
      }
      ElementLinearization& block =
          output.elements[static_cast<std::size_t>(element)];
      block.element = element;
      block.local_interior_unknowns = ni;
      block.local_trace_unknowns = nt;
      ResizeAndZero(block.a, static_cast<std::size_t>(ni) * ni);
      ResizeAndZero(block.b, static_cast<std::size_t>(ni) * nt);
      ResizeAndZero(block.c, static_cast<std::size_t>(nt) * ni);
      ResizeAndZero(block.d, static_cast<std::size_t>(nt) * nt);
      ResizeAndZero(block.local_residual, static_cast<std::size_t>(ni));
      ResizeAndZero(block.trace_residual, static_cast<std::size_t>(nt));
      const mfem::FiniteElement* finite_element = volume_space.GetFE(element);
      mfem::ElementTransformation* element_transformation =
          mesh.GetElementTransformation(element);
      if (finite_element == nullptr || element_transformation == nullptr ||
          finite_element->GetDof() != ns)
      {
        return false;
      }
      const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      mfem::Vector& shape = workspace.volume_shape;
      mfem::DenseMatrix& derivative = workspace.volume_derivative;
      for (int point = 0; point < volume_rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
        element_transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        finite_element->CalcPhysDShape(*element_transformation, derivative);
        State conservative{}, previous{};
        std::array<std::array<double, dimension>, state_components>
            conservative_derivative{};
        for (int component = 0; component < state_components; ++component)
        {
          conservative[component] = EvaluateComponent(
              current, component, ns, shape);
          previous[component] = EvaluateComponent(old, component, ns, shape);
          for (int direction = 0; direction < dimension; ++direction)
          {
            conservative_derivative[component][direction] =
                EvaluateDerivative(current, component, ns, derivative,
                                   direction);
          }
        }
        if (!SampleAdmissible(conservative) || !SampleAdmissible(previous))
        {
          diagnostic.SetFailureCause(
              TrajectoryAssemblyFailureCause::state_admissibility);
          return false;
        }
        const double sigma = EvaluateComponent(current, 4, ns, shape);
        const std::array<double, dimension> z{{
            EvaluateComponent(current, 5, ns, shape),
            EvaluateComponent(current, 6, ns, shape)}};
        std::array<State, dimension> flux{};
        std::array<FluxJacobian, dimension> flux_jacobian{};
        if (!EulerSigmaNormalFlux(conservative, sigma, {{1.0, 0.0}},
                                  flux[0], flux_jacobian[0]) ||
            !EulerSigmaNormalFlux(conservative, sigma, {{0.0, 1.0}},
                                  flux[1], flux_jacobian[1]))
        {
          return false;
        }
        const double rho = conservative[0];
        const double mx = conservative[1];
        const double my = conservative[2];
        std::array<std::array<double, dimension>, dimension> velocity_gradient{};
        for (int direction = 0; direction < dimension; ++direction)
        {
          velocity_gradient[0][direction] =
              conservative_derivative[1][direction] / rho -
              mx * conservative_derivative[0][direction] / (rho * rho);
          velocity_gradient[1][direction] =
              conservative_derivative[2][direction] / rho -
              my * conservative_derivative[0][direction] / (rho * rho);
        }
        const double q = VelocityGradientInvariant(velocity_gradient);
        const double trace_gradient =
            velocity_gradient[0][0] + velocity_gradient[1][1];
        std::array<std::array<double, dimension>, dimension> dq_dgradient{};
        for (int velocity = 0; velocity < dimension; ++velocity)
          for (int direction = 0; direction < dimension; ++direction)
          {
            dq_dgradient[velocity][direction] =
                2.0 * velocity_gradient[direction][velocity] +
                (velocity == direction ? 2.0 * trace_gradient : 0.0);
          }
        const double weight = ip.weight * element_transformation->Weight();
        if (!Finite(weight) || !(weight > 0.0)) { return false; }
        for (int row_scalar = 0; row_scalar < ns; ++row_scalar)
        {
          for (int equation = 0; equation < state_components; ++equation)
          {
            const int row = local_index(equation, row_scalar);
            block.local_residual[row] += weight *
                (shape[row_scalar] *
                     (conservative[equation] - previous[equation]) /
                     time_step -
                 derivative(row_scalar, 0) * flux[0][equation] -
                 derivative(row_scalar, 1) * flux[1][equation]);
            for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
              for (int variable = 0; variable < trace_components; ++variable)
              {
                const int column = local_index(variable, column_scalar);
                double value = variable == equation ?
                    shape[row_scalar] * shape[column_scalar] /
                        time_step : 0.0;
                value -= derivative(row_scalar, 0) *
                    flux_jacobian[0][equation][variable] *
                    shape[column_scalar];
                value -= derivative(row_scalar, 1) *
                    flux_jacobian[1][equation][variable] *
                    shape[column_scalar];
                block.a[static_cast<std::size_t>(row) * ni + column] +=
                    weight * value;
              }
          }
          for (int direction = 0; direction < dimension; ++direction)
          {
            const int row = local_index(4 + direction, row_scalar);
            block.local_residual[row] += weight *
                (shape[row_scalar] * rho * z[direction] +
                 derivative(row_scalar, direction) * sigma);
            for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
            {
              block.a[static_cast<std::size_t>(row) * ni +
                      local_index(0, column_scalar)] +=
                  weight * shape[row_scalar] * z[direction] *
                  shape[column_scalar];
              block.a[static_cast<std::size_t>(row) * ni +
                      local_index(4, column_scalar)] +=
                  weight * derivative(row_scalar, direction) *
                  shape[column_scalar];
              block.a[static_cast<std::size_t>(row) * ni +
                      local_index(5 + direction, column_scalar)] +=
                  weight * shape[row_scalar] * rho * shape[column_scalar];
            }
          }
          const int row_sigma = local_index(6, row_scalar);
          block.local_residual[row_sigma] += weight *
              (shape[row_scalar] * sigma / rho +
               primary_alpha * derivative(row_scalar, 0) * z[0] +
               primary_alpha * derivative(row_scalar, 1) * z[1] -
               primary_alpha * shape[row_scalar] * q);
          for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
          {
            block.a[static_cast<std::size_t>(row_sigma) * ni +
                    local_index(4, column_scalar)] +=
                weight * shape[row_scalar] * shape[column_scalar] / rho;
            for (int direction = 0; direction < dimension; ++direction)
            {
              block.a[static_cast<std::size_t>(row_sigma) * ni +
                      local_index(5 + direction, column_scalar)] +=
                  primary_alpha * weight * derivative(row_scalar, direction) *
                  shape[column_scalar];
            }
            for (int variable = 0; variable < state_components; ++variable)
            {
              double dq = 0.0;
              if (variable < 3)
              {
                for (int velocity = 0; velocity < dimension; ++velocity)
                  for (int direction = 0; direction < dimension; ++direction)
                  {
                    const double delta_rho =
                        variable == 0 ? shape[column_scalar] : 0.0;
                    const double delta_drho =
                        variable == 0 ? derivative(column_scalar, direction) :
                                        0.0;
                    const double delta_m =
                        variable == 1 + velocity ? shape[column_scalar] : 0.0;
                    const double delta_dm = variable == 1 + velocity ?
                        derivative(column_scalar, direction) : 0.0;
                    const double momentum = velocity == 0 ? mx : my;
                    const double momentum_derivative =
                        conservative_derivative[1 + velocity][direction];
                    const double rho_derivative =
                        conservative_derivative[0][direction];
                    const double delta_gradient =
                        delta_dm / rho -
                        momentum_derivative * delta_rho / (rho * rho) -
                        delta_m * rho_derivative / (rho * rho) -
                        momentum * delta_drho / (rho * rho) +
                        2.0 * momentum * rho_derivative * delta_rho /
                            (rho * rho * rho);
                    dq += dq_dgradient[velocity][direction] * delta_gradient;
                  }
              }
              double value = -primary_alpha * shape[row_scalar] * dq;
              if (variable == 0)
              {
                value -= shape[row_scalar] * sigma * shape[column_scalar] /
                    (rho * rho);
              }
              block.a[static_cast<std::size_t>(row_sigma) * ni +
                      local_index(variable, column_scalar)] += weight * value;
            }
          }
        }
      }

      diagnostic.Set(phase, "faces", element);
      const double element_measure = ElementMeasureAtOrder(
          mesh, element, quadrature_order);
      if (!Finite(element_measure) || !(element_measure > 0.0))
      {
        return false;
      }
      const ElementTopology& element_topology =
          topology[static_cast<std::size_t>(element)];
      if (element_topology.faces.size() != faces_per_quadrilateral ||
          element_topology.face_orientations.size() !=
              faces_per_quadrilateral)
      {
        return false;
      }
      for (int local_face = 0; local_face < faces_per_quadrilateral;
           ++local_face)
      {
        const int face = element_topology.faces[local_face];
        mfem::FaceElementTransformations* face_transformation =
            mesh.GetFaceElementTransformations(face);
        const mfem::FiniteElement* trace_element =
            carrier_space.GetFaceElement(face);
        if (face_transformation == nullptr ||
            face_transformation->Face == nullptr || trace_element == nullptr ||
            trace_element->GetDof() != 2)
        {
          return false;
        }
        const bool element_one = face_transformation->Elem1No == element;
        if (!element_one && face_transformation->Elem2No != element)
        {
          return false;
        }
        const int orientation =
            element_topology.face_orientations[local_face];
        const int* permutation = trace_collection.DofOrderForOrientation(
            mfem::Geometry::SEGMENT, orientation);
        const double face_measure = FaceMeasureAtOrder(
            mesh, face, quadrature_order);
        const double height = element_measure / face_measure;
        if (permutation == nullptr || !Finite(face_measure) ||
            !(face_measure > 0.0) || !Finite(height) || !(height > 0.0))
        {
          return false;
        }
        const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
            trace_element->GetGeomType(), quadrature_order);
        mfem::Vector& interior_shape = workspace.face_interior_shape;
        mfem::Vector& trace_shape = workspace.face_trace_shape;
        mfem::Vector& oriented_shape = workspace.face_oriented_shape;
        for (int point = 0; point < face_rule.GetNPoints(); ++point)
        {
          const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
          face_transformation->SetAllIntPoints(&ip);
          const mfem::IntegrationPoint& element_ip = element_one ?
              face_transformation->GetElement1IntPoint() :
              face_transformation->GetElement2IntPoint();
          finite_element->CalcShape(element_ip, interior_shape);
          trace_element->CalcShape(ip, trace_shape);
          for (int scalar = 0; scalar < 2; ++scalar)
          {
            int sign = 0;
            const int canonical = DecodeSignedDof(permutation[scalar], sign);
            if (canonical < 0 || canonical >= 2 || sign == 0)
            {
              return false;
            }
            oriented_shape[scalar] = sign * trace_shape[canonical];
          }
          mfem::Vector& scaled = workspace.scaled_normal;
          mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled);
          const double norm = scaled.Norml2();
          const double measure = face_transformation->Face->Weight();
          if (!Finite(norm) || !Finite(measure) || !(norm > 0.0) ||
              !(measure > 0.0))
          {
            return false;
          }
          const double side = element_one ? 1.0 : -1.0;
          const std::array<double, dimension> normal{{
              side * scaled[0] / norm, side * scaled[1] / norm}};
          State conservative{}, previous{}, trace_state{}, previous_trace{};
          for (int component = 0; component < state_components; ++component)
          {
            conservative[component] = EvaluateComponent(
                current, component, ns, interior_shape);
            previous[component] = EvaluateComponent(
                old, component, ns, interior_shape);
          }
          for (int component = 0; component < state_components; ++component)
          {
            trace_state[component] = 0.0;
            previous_trace[component] = 0.0;
            for (int scalar = 0; scalar < 2; ++scalar)
            {
              trace_state[component] += trace[
                  static_cast<std::size_t>(component) * nf +
                  local_face * 2 + scalar] * oriented_shape[scalar];
              previous_trace[component] += old_trace[
                  static_cast<std::size_t>(component) * nf +
                  local_face * 2 + scalar] * oriented_shape[scalar];
            }
          }
          double sigma = EvaluateComponent(current, 4, ns, interior_shape);
          std::array<double, dimension> z{{
              EvaluateComponent(current, 5, ns, interior_shape),
              EvaluateComponent(current, 6, ns, interior_shape)}};
          double sigmahat = 0.0;
          for (int scalar = 0; scalar < 2; ++scalar)
          {
            sigmahat += trace[static_cast<std::size_t>(4) * nf +
                              local_face * 2 + scalar] *
                oriented_shape[scalar];
          }
          if (!SampleAdmissible(conservative) ||
              !SampleAdmissible(previous) ||
              !SampleAdmissible(trace_state) ||
              !SampleAdmissible(previous_trace))
          {
            diagnostic.SetFailureCause(
                TrajectoryAssemblyFailureCause::state_admissibility);
            return false;
          }
          State physical_flux{};
          FluxJacobian physical_jacobian{};
          if (!EulerSigmaNormalFlux(trace_state, sigmahat, normal,
                                    physical_flux, physical_jacobian))
          {
            return false;
          }
          State numerical_flux{};
          for (int equation = 0; equation < state_components; ++equation)
          {
            numerical_flux[equation] = physical_flux[equation] +
                state_stabilization *
                    (conservative[equation] - trace_state[equation]);
          }
          const double rhohat = trace_state[0];
          const double tau = 4.0 / (rhohat * height);
          const double jump = sigma - sigmahat;
          const double auxiliary_flux = z[0] * normal[0] +
              z[1] * normal[1] - tau * jump;
          const double weight = ip.weight * measure;
          if (!Finite(tau) || !(tau > 0.0) || !Finite(weight) ||
              !(weight > 0.0))
          {
            return false;
          }
          for (int row_scalar = 0; row_scalar < ns; ++row_scalar)
          {
            for (int equation = 0; equation < state_components; ++equation)
            {
              const int row = local_index(equation, row_scalar);
              block.local_residual[row] +=
                  weight * interior_shape[row_scalar] *
                  numerical_flux[equation];
              for (int column_scalar = 0; column_scalar < ns;
                   ++column_scalar)
              {
                block.a[static_cast<std::size_t>(row) * ni +
                        local_index(equation, column_scalar)] +=
                    weight * interior_shape[row_scalar] *
                    state_stabilization * interior_shape[column_scalar];
              }
              for (int trace_variable = 0;
                   trace_variable < trace_components; ++trace_variable)
                for (int column_scalar = 0; column_scalar < 2;
                     ++column_scalar)
                {
                  const int column = trace_index(
                      trace_variable, local_face * 2 + column_scalar);
                  const double derivative_value =
                      physical_jacobian[equation][trace_variable] -
                      (trace_variable == equation ? state_stabilization : 0.0);
                  block.b[static_cast<std::size_t>(row) * nt + column] +=
                      weight * interior_shape[row_scalar] * derivative_value *
                      oriented_shape[column_scalar];
                }
            }
            for (int direction = 0; direction < dimension; ++direction)
            {
              const int row = local_index(4 + direction, row_scalar);
              block.local_residual[row] -= weight *
                  interior_shape[row_scalar] * normal[direction] * sigmahat;
              for (int column_scalar = 0; column_scalar < 2;
                   ++column_scalar)
              {
                const int column = trace_index(
                    4, local_face * 2 + column_scalar);
                block.b[static_cast<std::size_t>(row) * nt + column] -=
                    weight * interior_shape[row_scalar] * normal[direction] *
                    oriented_shape[column_scalar];
              }
            }
            const int row_sigma = local_index(6, row_scalar);
            block.local_residual[row_sigma] -=
                primary_alpha * weight * interior_shape[row_scalar] *
                auxiliary_flux;
            for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
            {
              block.a[static_cast<std::size_t>(row_sigma) * ni +
                      local_index(4, column_scalar)] +=
                  primary_alpha * weight * interior_shape[row_scalar] * tau *
                  interior_shape[column_scalar];
              for (int direction = 0; direction < dimension; ++direction)
              {
                block.a[static_cast<std::size_t>(row_sigma) * ni +
                        local_index(5 + direction, column_scalar)] -=
                    primary_alpha * weight * interior_shape[row_scalar] *
                    normal[direction] * interior_shape[column_scalar];
              }
            }
            for (int column_scalar = 0; column_scalar < 2; ++column_scalar)
            {
              block.b[static_cast<std::size_t>(row_sigma) * nt +
                      trace_index(4, local_face * 2 + column_scalar)] -=
                  primary_alpha * weight * interior_shape[row_scalar] * tau *
                  oriented_shape[column_scalar];
              block.b[static_cast<std::size_t>(row_sigma) * nt +
                      trace_index(0, local_face * 2 + column_scalar)] -=
                  primary_alpha * weight * interior_shape[row_scalar] * tau *
                  jump / rhohat * oriented_shape[column_scalar];
            }
          }
          for (int row_component = 0; row_component < trace_components;
               ++row_component)
            for (int row_scalar = 0; row_scalar < 2; ++row_scalar)
            {
              const int row = trace_index(
                  row_component, local_face * 2 + row_scalar);
              if (row_component < state_components)
              {
                block.trace_residual[row] += weight *
                    oriented_shape[row_scalar] *
                    numerical_flux[row_component];
                for (int column_scalar = 0; column_scalar < ns;
                     ++column_scalar)
                {
                  block.c[static_cast<std::size_t>(row) * ni +
                          local_index(row_component, column_scalar)] +=
                      weight * oriented_shape[row_scalar] *
                      state_stabilization * interior_shape[column_scalar];
                }
                for (int trace_variable = 0;
                     trace_variable < trace_components; ++trace_variable)
                  for (int column_scalar = 0; column_scalar < 2;
                       ++column_scalar)
                  {
                    const int column = trace_index(
                        trace_variable, local_face * 2 + column_scalar);
                    const double derivative_value =
                        physical_jacobian[row_component][trace_variable] -
                        (trace_variable == row_component ?
                             state_stabilization : 0.0);
                    block.d[static_cast<std::size_t>(row) * nt + column] +=
                        weight * oriented_shape[row_scalar] *
                        derivative_value * oriented_shape[column_scalar];
                  }
              }
              else
              {
                block.trace_residual[row] += primary_alpha * weight *
                    oriented_shape[row_scalar] * auxiliary_flux;
                for (int column_scalar = 0; column_scalar < ns;
                     ++column_scalar)
                {
                  block.c[static_cast<std::size_t>(row) * ni +
                          local_index(4, column_scalar)] -=
                      primary_alpha * weight * oriented_shape[row_scalar] *
                      tau * interior_shape[column_scalar];
                  for (int direction = 0; direction < dimension; ++direction)
                  {
                    block.c[static_cast<std::size_t>(row) * ni +
                            local_index(5 + direction, column_scalar)] +=
                        primary_alpha * weight * oriented_shape[row_scalar] *
                        normal[direction] * interior_shape[column_scalar];
                  }
                }
                for (int column_scalar = 0; column_scalar < 2;
                     ++column_scalar)
                {
                  block.d[static_cast<std::size_t>(row) * nt +
                          trace_index(4, local_face * 2 + column_scalar)] +=
                      primary_alpha * weight * oriented_shape[row_scalar] *
                      tau * oriented_shape[column_scalar];
                  block.d[static_cast<std::size_t>(row) * nt +
                          trace_index(0, local_face * 2 + column_scalar)] +=
                      primary_alpha * weight * oriented_shape[row_scalar] *
                      tau * jump / rhohat * oriented_shape[column_scalar];
                }
              }
            }
        }
      }
      diagnostic.Set(phase, "block-finite", element);
      if (!Finite(block.a) || !Finite(block.b) || !Finite(block.c) ||
          !Finite(block.d) || !Finite(block.local_residual) ||
          !Finite(block.trace_residual))
      {
        diagnostic.SetFailureCause(
            TrajectoryAssemblyFailureCause::nonfinite_result);
        return false;
      }
    }

    diagnostic.Set(phase, "global-scatter");
    for (int element = 0; element < elements; ++element)
    {
      const ElementLinearization& block =
          output.elements[static_cast<std::size_t>(element)];
      const ElementTopology& map = topology[static_cast<std::size_t>(element)];
      const int local_base = element * ni;
      for (int row = 0; row < ni; ++row)
      {
        output.full_residual[local_base + row] = block.local_residual[row];
        for (int column = 0; assemble_dense_jacobian && column < ni;
             ++column)
        {
          output.full_jacobian[
              static_cast<std::size_t>(local_base + row) * full +
              local_base + column] =
              block.a[static_cast<std::size_t>(row) * ni + column];
        }
        for (int column = 0; column < nt; ++column)
        {
          const TraceRestrictionEntry& entry = map.restriction[column];
          if (entry.local_true_dof < 0 ||
              entry.local_true_dof >= true_trace)
          {
            return false;
          }
          if (assemble_dense_jacobian)
          {
            const int true_column =
                global_interior + entry.local_true_dof;
            output.full_jacobian[
                static_cast<std::size_t>(local_base + row) * full +
                true_column] += entry.sign *
                block.b[static_cast<std::size_t>(row) * nt + column];
          }
        }
      }
      for (int row = 0; row < nt; ++row)
      {
        const TraceRestrictionEntry& row_entry = map.restriction[row];
        if (row_entry.local_true_dof < 0 ||
            row_entry.local_true_dof >= true_trace)
        {
          return false;
        }
        const int true_row = global_interior + row_entry.local_true_dof;
        output.full_residual[true_row] +=
            row_entry.sign * block.trace_residual[row];
        for (int column = 0; assemble_dense_jacobian && column < ni;
             ++column)
        {
          output.full_jacobian[
              static_cast<std::size_t>(true_row) * full +
              local_base + column] += row_entry.sign *
              block.c[static_cast<std::size_t>(row) * ni + column];
        }
        for (int column = 0; column < nt; ++column)
        {
          const TraceRestrictionEntry& column_entry =
              map.restriction[column];
          if (column_entry.local_true_dof < 0 ||
              column_entry.local_true_dof >= true_trace)
          {
            return false;
          }
          if (assemble_dense_jacobian)
          {
            const int true_column =
                global_interior + column_entry.local_true_dof;
            output.full_jacobian[
                static_cast<std::size_t>(true_row) * full + true_column] +=
                row_entry.sign * column_entry.sign *
                block.d[static_cast<std::size_t>(row) * nt + column];
          }
        }
      }
    }
    const bool finite = Finite(output.full_residual) &&
        (!assemble_dense_jacobian || Finite(output.full_jacobian));
    if (!finite)
    {
      diagnostic.SetFailureCause(
          TrajectoryAssemblyFailureCause::nonfinite_result);
    }
    return finite;
  }
  catch (...)
  {
    diagnostic.Set(phase, "exception", diagnostic.element);
    diagnostic.SetFailureCause(TrajectoryAssemblyFailureCause::exception);
    output = {};
    return false;
  }
}

bool SolveChecked(const DenseLu& factor, const std::vector<double>& matrix,
                  const std::vector<double>& rhs,
                  std::vector<double>& solution,
                  double& maximum_residual,
                  CoupledFixturePrivateDiagnostic& diagnostic,
                  const char* solve_kind, const int trace_column) noexcept
{
  diagnostic.local_solve_kind = solve_kind;
  diagnostic.local_solve_trace_column = trace_column;
  diagnostic.local_solve_reached = true;
  diagnostic.local_solve_succeeded = false;
  diagnostic.local_application_reached = false;
  diagnostic.local_application_residual =
      std::numeric_limits<double>::quiet_NaN();
  diagnostic.local_right_hand_side_norm = VectorNorm(rhs);
  diagnostic.local_application_bound = local_absolute_tolerance +
      local_relative_tolerance * diagnostic.local_right_hand_side_norm;
  if (!factor.Solve(rhs, solution)) { return false; }
  diagnostic.local_solve_succeeded = true;
  const double residual = ApplicationResidual(
      matrix, factor.size, solution, rhs);
  diagnostic.local_application_reached = true;
  diagnostic.local_application_residual = residual;
  maximum_residual = std::max(maximum_residual, residual);
  return residual <= diagnostic.local_application_bound;
}

void ConfigureCoupledIlu(mfem::HypreILU& ilu)
{
  ilu.SetType(0);
  ilu.SetLevelOfFill(0);
  ilu.SetMaxIter(1);
  ilu.SetTol(0.0);
  ilu.SetLocalReordering(1);
  ilu.SetPrintLevel(0);
}

struct TrajectoryInteriorBacking final
{
  std::array<double, persistent_u_unknowns> u{};
  std::array<double, persistent_sigma_unknowns> sigma{};
  std::array<double, persistent_z_unknowns> z{};
};

struct TrajectoryTraceBacking final
{
  std::array<double, persistent_uhat_unknowns> uhat{};
  std::array<double, persistent_sigmahat_unknowns> sigmahat{};
};

using TrajectorySlot =
    CompleteStateSlot<TrajectoryInteriorBacking, TrajectoryTraceBacking>;

template <typename... Arrays>
StateHash HashArrays(const Arrays&... arrays) noexcept
{
  Sha256 sha;
  const bool updated =
      (sha.Update(arrays.data(), arrays.size() * sizeof(double)) && ...);
  Sha256Digest digest{};
  if (!updated || !sha.Finalize(digest)) { return {}; }
  return {digest};
}

struct TrajectoryStoragePolicy final
{
  static constexpr bool preparation_preserves_committed = true;
  static constexpr bool private_copy_is_independent = true;
  static constexpr bool publish_is_atomic = true;
  static constexpr bool publish_preserves_disjoint_backing = true;

  static bool PrepareCandidate(const TrajectorySlot& committed,
                               TrajectorySlot& candidate) noexcept
  {
    candidate = committed;
    return true;
  }
  static bool PrepareRestore(const TrajectorySlot& loaded,
                             TrajectorySlot& candidate) noexcept
  {
    candidate = loaded;
    return true;
  }
  static bool StoreCandidate(TrajectorySlot& candidate,
                             const TrajectorySlot& value) noexcept
  {
    candidate = value;
    return true;
  }
  static bool CopyOut(const TrajectorySlot& source,
                      TrajectorySlot& destination) noexcept
  {
    destination = source;
    return true;
  }
  static void DiscardCandidate(TrajectorySlot&) noexcept {}
  static bool CanPublish(const TrajectorySlot& committed,
                         const TrajectorySlot& candidate) noexcept
  {
    return Disjoint(committed, candidate);
  }
  static void Publish(TrajectorySlot& committed,
                      TrajectorySlot& candidate) noexcept
  {
    using std::swap;
    swap(committed, candidate);
  }
  static bool Disjoint(const TrajectorySlot& left,
                       const TrajectorySlot& right) noexcept
  {
    return &left != &right && left.interior.u.data() != right.interior.u.data() &&
        left.trace.uhat.data() != right.trace.uhat.data();
  }
  static StateHash HashInterior(
      const TrajectoryInteriorBacking& state) noexcept
  {
    return HashArrays(state.u, state.sigma, state.z);
  }
  static StateHash HashTrace(const TrajectoryTraceBacking& state) noexcept
  {
    return HashArrays(state.uhat, state.sigmahat);
  }
};

struct TrajectorySerialCollectivePolicy final
{
  static CollectiveAdmissionReport Admit(
      const AttemptCollectiveContext& context, const bool prepared,
      FailureReason reason, const std::uint32_t ranks) noexcept
  {
    if (!prepared && reason == FailureReason::none)
    {
      reason = FailureReason::candidate_preparation;
    }
    return {context,
            ranks,
            ranks == 1,
            ranks == 1,
            prepared,
            reason != FailureReason::none,
            detail::FailureBit(reason),
            reason};
  }

  static CollectiveGateReport Reduce(
      const AttemptCollectiveContext& context, const GateReport& gates,
      const FailureReason reason, const std::uint32_t ranks) noexcept
  {
    return {context,
            gates,
            ranks,
            ranks == 1,
            ranks == 1,
            reason != FailureReason::none,
            detail::FailureBit(reason),
            reason};
  }

  static CollectiveRestoreReport ReduceRestore(
      const RestoreCollectiveContext& context,
      const RestartCompatibility compatibility,
      const FailureReason reason, const std::uint32_t ranks) noexcept
  {
    return {context,
            ranks,
            ranks == 1,
            ranks == 1,
            reason != FailureReason::none,
            detail::FailureBit(reason),
            detail::CompatibilityBit(compatibility),
            reason,
            compatibility};
  }
};

using TrajectoryContract =
    PersistentRunContract<CpuRank1Selection,
                          TrajectoryInteriorBacking,
                          TrajectoryTraceBacking,
                          TrajectoryStoragePolicy,
                          TrajectorySerialCollectivePolicy>;

TrajectorySlot ToSlot(const PersistentTrajectoryState& state) noexcept
{
  TrajectorySlot slot;
  slot.interior.u = state.u;
  slot.interior.sigma = state.sigma;
  slot.interior.z = state.z;
  slot.trace.uhat = state.uhat;
  slot.trace.sigmahat = state.sigmahat;
  return slot;
}

PersistentTrajectoryState ToPublicState(const TrajectorySlot& slot) noexcept
{
  PersistentTrajectoryState state;
  state.u = slot.interior.u;
  state.sigma = slot.interior.sigma;
  state.z = slot.interior.z;
  state.uhat = slot.trace.uhat;
  state.sigmahat = slot.trace.sigmahat;
  return state;
}

template <std::size_t N>
bool SameFp64Bytes(const std::array<double, N>& left,
                   const std::array<double, N>& right) noexcept
{
  return std::memcmp(left.data(), right.data(), N * sizeof(double)) == 0;
}

bool SameTrajectoryStateBytes(const PersistentTrajectoryState& left,
                              const PersistentTrajectoryState& right) noexcept
{
  return SameFp64Bytes(left.u, right.u) &&
      SameFp64Bytes(left.sigma, right.sigma) &&
      SameFp64Bytes(left.z, right.z) &&
      SameFp64Bytes(left.uhat, right.uhat) &&
      SameFp64Bytes(left.sigmahat, right.sigmahat);
}

TrajectoryStateHashes HashState(const TrajectorySlot& state) noexcept
{
  return {HashArrays(state.interior.u),
          HashArrays(state.interior.sigma),
          HashArrays(state.interior.z),
          HashArrays(state.trace.uhat),
          HashArrays(state.trace.sigmahat),
          TrajectoryStoragePolicy::HashInterior(state.interior),
          TrajectoryStoragePolicy::HashTrace(state.trace)};
}

std::array<TrajectoryCheckpointFieldIdentity, 5>
CheckpointFieldIdentities() noexcept
{
  return {{{StableIdentity::Copy("U_h.element-component-node"),
            persistent_u_unknowns, 0},
           {StableIdentity::Copy("Sigma_h.element-node"),
            persistent_sigma_unknowns, 1},
           {StableIdentity::Copy("z_h.element-component-node"),
            persistent_z_unknowns, 2},
           {StableIdentity::Copy("Uhat_h.component-true-node"),
            persistent_uhat_unknowns, 3},
           {StableIdentity::Copy("Sigmahat_h.true-node"),
            persistent_sigmahat_unknowns, 4}}};
}

bool SameFieldIdentity(const TrajectoryCheckpointFieldIdentity& left,
                       const TrajectoryCheckpointFieldIdentity& right)
    noexcept
{
  return left.name == right.name &&
      left.fp64_entries == right.fp64_entries &&
      left.canonical_order == right.canonical_order;
}

class ManifestHashBuilder final
{
public:
  bool Text(const std::string_view value) noexcept
  {
    return U64(value.size()) && hash_.Update(value.data(), value.size());
  }

  bool U64(const std::uint64_t value) noexcept
  {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
      bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
    return hash_.Update(bytes.data(), bytes.size());
  }

  bool Hash(const StateHash& value) noexcept
  {
    return hash_.Update(value.bytes.data(), value.bytes.size());
  }

  StateHash Finish() noexcept
  {
    Sha256Digest digest{};
    return hash_.Finalize(digest) ? StateHash{digest} : StateHash{};
  }

private:
  Sha256 hash_;
};

StateHash CanonicalCheckpointManifestHash(
    const PersistentTrajectoryCheckpoint& checkpoint) noexcept
{
  ManifestHashBuilder builder;
  const CheckpointDescriptor& descriptor = checkpoint.descriptor;
  const RuntimeTupleDescriptor& tuple = descriptor.tuple;
  bool valid = builder.Text(persistent_checkpoint_manifest_id) &&
      builder.Text(checkpoint.projection.view()) &&
      builder.Text(checkpoint.tuple.view()) &&
      builder.Text(checkpoint.scenario.view()) &&
      builder.U64(descriptor.schema_version) &&
      builder.U64(descriptor.successful_commit ? 1U : 0U) &&
      builder.U64(static_cast<std::uint64_t>(descriptor.restart_mode)) &&
      builder.Text(tuple.model.view()) &&
      builder.Text(tuple.formulation.view()) &&
      builder.Text(tuple.trace.view()) &&
      builder.Text(tuple.stabilization.view()) &&
      builder.Text(tuple.time_treatment.view()) &&
      builder.Text(tuple.backend.view()) &&
      builder.Text(tuple.precision.view()) &&
      builder.Text(descriptor.mfem_pin.view()) &&
      builder.Hash(descriptor.mesh_hash) &&
      builder.Hash(descriptor.partition_hash) &&
      builder.Hash(descriptor.topology_hash) &&
      builder.U64(descriptor.rank_count) &&
      builder.U64(static_cast<std::uint64_t>(descriptor.polynomial_order)) &&
      builder.Text(descriptor.layout.view()) &&
      builder.Text(descriptor.precision.view()) &&
      builder.U64(descriptor.cursor) &&
      builder.U64(detail::DoubleBits(descriptor.dt)) &&
      builder.U64(detail::DoubleBits(descriptor.coordinates.time)) &&
      builder.U64(descriptor.coordinates.step) &&
      builder.U64(descriptor.coordinates.generation) &&
      builder.U64(descriptor.coordinates.cursor) &&
      builder.U64(detail::DoubleBits(descriptor.coordinates.dt)) &&
      builder.U64(descriptor.coordinate_fingerprint) &&
      builder.Hash(descriptor.interior_hash) &&
      builder.Hash(descriptor.trace_hash);
  for (const TrajectoryCheckpointFieldIdentity& field : checkpoint.fields)
  {
    valid = valid && builder.Text(field.name.view()) &&
        builder.U64(field.fp64_entries) && builder.U64(field.canonical_order);
  }
  const TrajectoryStateHashes& hashes = checkpoint.raw_field_hashes;
  valid = valid && builder.Hash(hashes.u) && builder.Hash(hashes.sigma) &&
      builder.Hash(hashes.z) && builder.Hash(hashes.uhat) &&
      builder.Hash(hashes.sigmahat);
  return valid ? builder.Finish() : StateHash{};
}

template <typename Owner>
bool LiveResourceHashes(
    const Owner& owner,
    StateHash& mesh_hash, StateHash& partition_hash,
    StateHash& topology_hash) noexcept
{
  mesh_hash = {};
  partition_hash = {};
  topology_hash = {};
  if (!owner.mesh || !owner.carrier_space ||
      !owner.carrier_map.IsConsistent() || owner.topology.empty())
  {
    return false;
  }
  try
  {
    ManifestHashBuilder mesh;
    bool valid = mesh.Text("mfem-live-periodic-mesh-v1") &&
        mesh.U64(static_cast<std::uint64_t>(owner.mesh->Dimension())) &&
        mesh.U64(static_cast<std::uint64_t>(owner.mesh->SpaceDimension())) &&
        mesh.U64(static_cast<std::uint64_t>(owner.mesh->GetNV())) &&
        mesh.U64(static_cast<std::uint64_t>(owner.mesh->GetNE())) &&
        mesh.U64(static_cast<std::uint64_t>(owner.mesh->GetNumFaces()));
    for (int vertex = 0; valid && vertex < owner.mesh->GetNV(); ++vertex)
    {
      const double* coordinates = owner.mesh->GetVertex(vertex);
      if (!coordinates) { return false; }
      for (int direction = 0; direction < owner.mesh->SpaceDimension();
           ++direction)
      {
        valid = valid && mesh.U64(detail::DoubleBits(
            coordinates[direction]));
      }
    }
    mfem::Array<int> vertices;
    for (int element = 0; valid && element < owner.mesh->GetNE(); ++element)
    {
      owner.mesh->GetElementVertices(element, vertices);
      valid = valid && mesh.U64(static_cast<std::uint64_t>(
          owner.mesh->GetElementBaseGeometry(element))) &&
          mesh.U64(static_cast<std::uint64_t>(vertices.Size()));
      for (int vertex : vertices)
      {
        valid = valid && mesh.U64(static_cast<std::uint64_t>(vertex));
      }
    }
    for (int face = 0; valid && face < owner.mesh->GetNumFaces(); ++face)
    {
      owner.mesh->GetFaceVertices(face, vertices);
      valid = valid && mesh.U64(static_cast<std::uint64_t>(
          owner.mesh->GetFaceGeometry(face))) &&
          mesh.U64(static_cast<std::uint64_t>(vertices.Size()));
      for (int vertex : vertices)
      {
        valid = valid && mesh.U64(static_cast<std::uint64_t>(vertex));
      }
    }
    mesh_hash = valid ? mesh.Finish() : StateHash{};

    ManifestHashBuilder partition;
    valid = partition.Text("mfem-live-rank-partition-v1") &&
        partition.U64(static_cast<std::uint64_t>(owner.rank)) &&
        partition.U64(static_cast<std::uint64_t>(owner.ranks)) &&
        partition.U64(static_cast<std::uint64_t>(owner.mesh->GetNE())) &&
        partition.U64(static_cast<std::uint64_t>(
            owner.carrier_space->GetVSize())) &&
        partition.U64(static_cast<std::uint64_t>(
            owner.carrier_space->GetTrueVSize())) &&
        partition.U64(static_cast<std::uint64_t>(
            owner.carrier_space->GlobalTrueVSize()));
    for (int element = 0; valid && element < owner.mesh->GetNE(); ++element)
    {
      valid = partition.U64(static_cast<std::uint64_t>(element));
    }
    partition_hash = valid ? partition.Finish() : StateHash{};

    ManifestHashBuilder topology;
    valid = topology.Text("mfem-live-periodic-trace-topology-v1") &&
        topology.U64(static_cast<std::uint64_t>(
            owner.carrier_map.vector_dimension)) &&
        topology.U64(static_cast<std::uint64_t>(owner.carrier_map.ordering)) &&
        topology.U64(static_cast<std::uint64_t>(
            owner.carrier_map.dofs.rank)) &&
        topology.U64(static_cast<std::uint64_t>(
            owner.carrier_map.dofs.rank_count)) &&
        topology.U64(static_cast<std::uint64_t>(
            owner.carrier_map.dofs.local_true_dof_count)) &&
        topology.U64(static_cast<std::uint64_t>(
            owner.carrier_map.dofs.global_true_dof_count));
    for (const trace::TraceMapEntry& entry : owner.carrier_map.dofs.entries)
    {
      valid = valid && topology.U64(static_cast<std::uint64_t>(entry.local_dof)) &&
          topology.U64(static_cast<std::uint64_t>(entry.global_true_dof)) &&
          topology.U64(static_cast<std::uint64_t>(entry.local_true_dof)) &&
          topology.U64(static_cast<std::uint64_t>(entry.owner_rank)) &&
          topology.U64(static_cast<std::uint64_t>(entry.ownership));
    }
    for (const trace::DGInterfaceFaceMap& face : owner.carrier_map.faces)
    {
      valid = valid && topology.U64(static_cast<std::uint64_t>(face.local_face)) &&
          topology.U64(static_cast<std::uint64_t>(face.geometry)) &&
          topology.U64(static_cast<std::uint64_t>(face.signed_local_vdofs.size()));
      for (const int dof : face.signed_local_vdofs)
      {
        valid = valid && topology.U64(static_cast<std::uint64_t>(
            static_cast<std::int64_t>(dof)));
      }
    }
    for (const ElementTopology& element : owner.topology)
    {
      valid = valid && topology.U64(static_cast<std::uint64_t>(element.element)) &&
          topology.U64(static_cast<std::uint64_t>(element.scalar_volume_dofs)) &&
          topology.U64(static_cast<std::uint64_t>(element.scalar_face_dofs)) &&
          topology.U64(static_cast<std::uint64_t>(element.local_interior_unknowns)) &&
          topology.U64(static_cast<std::uint64_t>(element.local_trace_unknowns));
      const auto hash_ints = [&topology, &valid](const std::vector<int>& values)
      {
        valid = valid && topology.U64(values.size());
        for (const int value : values)
        {
          valid = valid && topology.U64(static_cast<std::uint64_t>(
              static_cast<std::int64_t>(value)));
        }
      };
      hash_ints(element.volume.u);
      hash_ints(element.volume.sigma);
      hash_ints(element.volume.z);
      hash_ints(element.faces);
      hash_ints(element.face_orientations);
      hash_ints(element.signed_trace_vdofs);
      valid = valid && topology.U64(element.restriction.size());
      for (const TraceRestrictionEntry& entry : element.restriction)
      {
        valid = valid && topology.U64(static_cast<std::uint64_t>(entry.element)) &&
            topology.U64(static_cast<std::uint64_t>(entry.local_row)) &&
            topology.U64(static_cast<std::uint64_t>(entry.local_face)) &&
            topology.U64(static_cast<std::uint64_t>(entry.component)) &&
            topology.U64(static_cast<std::uint64_t>(entry.oriented_face_dof)) &&
            topology.U64(static_cast<std::uint64_t>(entry.canonical_face_dof)) &&
            topology.U64(static_cast<std::uint64_t>(
                static_cast<std::int64_t>(entry.signed_carrier_vdof))) &&
            topology.U64(static_cast<std::uint64_t>(entry.decoded_carrier_vdof)) &&
            topology.U64(static_cast<std::uint64_t>(
                static_cast<std::int64_t>(entry.sign))) &&
            topology.U64(static_cast<std::uint64_t>(entry.global_true_dof)) &&
            topology.U64(static_cast<std::uint64_t>(entry.local_true_dof)) &&
            topology.U64(static_cast<std::uint64_t>(entry.owner_rank));
      }
    }
    topology_hash = valid ? topology.Finish() : StateHash{};
    return valid && mesh_hash != StateHash{} &&
        partition_hash != StateHash{} && topology_hash != StateHash{};
  }
  catch (...)
  {
    return false;
  }
}

TrajectorySlot ApplyDirection(
    const TrajectorySlot& entry,
    const PersistentTrajectoryState& direction,
    double theta) noexcept;

struct RuntimeElementWorkspace final
{
  DenseLu factor;
  ElementCondensedBlock condensed;
  std::vector<double> inverse_residual;
  std::vector<double> inverse_b;
  std::vector<double> rhs;
  std::vector<double> solution;
  std::vector<double> recovery_rhs;
  std::vector<double> recovered;

  void Initialize(const int element)
  {
    constexpr int ni = coupled_fixture_local_interior_unknowns;
    constexpr int nt = coupled_fixture_local_trace_unknowns;
    factor.size = ni;
    factor.values.assign(static_cast<std::size_t>(ni) * ni, 0.0);
    factor.pivots.assign(ni, 0);
    condensed.element = element;
    condensed.local_trace_unknowns = nt;
    condensed.schur.assign(static_cast<std::size_t>(nt) * nt, 0.0);
    condensed.right_hand_side.assign(nt, 0.0);
    inverse_residual.assign(ni, 0.0);
    inverse_b.assign(static_cast<std::size_t>(ni) * nt, 0.0);
    rhs.assign(ni, 0.0);
    solution.assign(ni, 0.0);
    recovery_rhs.assign(ni, 0.0);
    recovered.assign(ni, 0.0);
  }
};

} // namespace

struct PersistentTrajectoryImplementation final
{
  std::unique_ptr<TrajectoryContract> contract;
  CoupledFixtureState assembly_state;
  CoupledFixtureAssembly assembly;
  CoupledFixtureAssembly trial_assembly;
  CoupledAssemblyWorkspace quadrature_workspace;
  CoupledFixturePrivateDiagnostic diagnostic;
  std::vector<RuntimeElementWorkspace> elements;
  std::unique_ptr<mfem::SparseMatrix> local_pattern;
  std::unique_ptr<mfem::HypreParMatrix> true_matrix;
  std::unique_ptr<mfem::HypreParVector> true_rhs;
  std::unique_ptr<mfem::HypreParVector> true_solution;
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;
  mfem::Vector true_residual;
  std::vector<int> element_to_true_csr;
  PersistentTrajectoryEvidence evidence{};
  TrajectoryStatus status = TrajectoryStatus::uninitialized;
  TrajectoryStatus first_failure = TrajectoryStatus::uninitialized;
  bool initialized = false;
  std::uint64_t matrix_generation = 0;
  bool reuse_baseline_frozen = false;

  bool InitializeStorage(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner) noexcept;
  bool LoadAssemblyState(
      const PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
      const TrajectorySlot& old_state,
      const TrajectorySlot& candidate) noexcept;
  TrajectoryAssemblyFailureCause Assemble(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
      const TrajectorySlot& old_state, const TrajectorySlot& candidate,
      CoupledFixtureAssembly& output) noexcept;
  bool InitializePhysicalState(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
      TrajectorySlot& state) noexcept;
  bool InitializeAuxiliary(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
      TrajectorySlot& state) noexcept;
  bool BuildPersistentTraceGraph(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner) noexcept;
  bool Condense(const CoupledFixtureAssembly& system,
                double& maximum_residual) noexcept;
  bool AssembleTrueSystem(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner) noexcept;
  bool SolveTrueSystem(int& iterations, double& residual_norm,
                       double& rhs_norm) noexcept;
  bool Recover(
               PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
               const CoupledFixtureAssembly& system,
               PersistentTrajectoryState& direction,
               double& maximum_residual) noexcept;
  bool Integrals(PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
                 const TrajectorySlot& state,
                 std::array<double, state_components>& values) noexcept;
  bool AuxiliaryResidualGate(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
      const TrajectorySlot& state, AuxiliaryResidualMetrics& metrics)
      noexcept;
  TrajectoryStatus ExecuteAttempt(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
      const BeginOutcome& begin, TrajectoryFailureInjection injection,
      TrajectoryAttemptEvidence& attempt) noexcept;
  TrajectoryStatus Advance(
      PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
      TrajectoryFailureInjection injection) noexcept;
  void RefreshEvidence() noexcept;
  void RememberFailure(TrajectoryStatus failure) noexcept;
  void RecordAssemblyFailure(
      TrajectoryAssemblyFailureCause failure) noexcept;
  TrajectoryReuseSnapshot ReuseSnapshot(
      const PersistentSc25ScalarIgrCpuFoundation::Implementation& owner)
      const noexcept;
  bool VerifyReuse(
      const PersistentSc25ScalarIgrCpuFoundation::Implementation& owner)
      noexcept;
};

TrajectoryReuseSnapshot PersistentTrajectoryImplementation::ReuseSnapshot(
    const PersistentSc25ScalarIgrCpuFoundation::Implementation& owner)
    const noexcept
{
  TrajectoryReuseSnapshot snapshot;
  snapshot.foundation_handle = reinterpret_cast<std::uintptr_t>(&owner);
  snapshot.topology_handle =
      reinterpret_cast<std::uintptr_t>(&owner.topology);
  snapshot.carrier_map_handle =
      reinterpret_cast<std::uintptr_t>(&owner.carrier_map);
  snapshot.contract_handle = reinterpret_cast<std::uintptr_t>(contract.get());
  snapshot.assembly_state_handle =
      reinterpret_cast<std::uintptr_t>(&assembly_state);
  snapshot.assembly_handle = reinterpret_cast<std::uintptr_t>(&assembly);
  snapshot.trial_assembly_handle =
      reinterpret_cast<std::uintptr_t>(&trial_assembly);
  snapshot.factor_handle = elements.empty() ? 0 :
      reinterpret_cast<std::uintptr_t>(&elements.front().factor);
  snapshot.workspace_handle = reinterpret_cast<std::uintptr_t>(&elements);
  snapshot.quadrature_workspace_handle =
      reinterpret_cast<std::uintptr_t>(&quadrature_workspace);
  snapshot.local_pattern_handle =
      reinterpret_cast<std::uintptr_t>(local_pattern.get());
  snapshot.true_matrix_handle =
      reinterpret_cast<std::uintptr_t>(true_matrix.get());
  snapshot.true_rhs_handle = reinterpret_cast<std::uintptr_t>(true_rhs.get());
  snapshot.true_solution_handle =
      reinterpret_cast<std::uintptr_t>(true_solution.get());
  snapshot.ilu_handle = reinterpret_cast<std::uintptr_t>(ilu.get());
  snapshot.gmres_handle = reinterpret_cast<std::uintptr_t>(gmres.get());
  snapshot.true_residual_handle =
      reinterpret_cast<std::uintptr_t>(&true_residual);
  snapshot.element_to_true_csr_handle =
      reinterpret_cast<std::uintptr_t>(&element_to_true_csr);
  snapshot.topology_generation = 1;
  snapshot.storage_generation = 1;
  snapshot.factor_generation = 1;
  snapshot.workspace_generation = 1;
  snapshot.local_pattern_generation = 1;
  snapshot.solver_generation = 1;
  snapshot.opaque_mfem_hypre_observed = false;

  Sha256 hash;
  bool valid = true;
  std::uint64_t objects = 0;
  std::uint64_t buffers = 0;
  const auto add_object = [&hash, &valid, &objects](const void* object)
  {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(object);
    valid = valid && hash.Update(&address, sizeof(address));
    if (object != nullptr) { ++objects; }
  };
  const auto add = [&hash, &valid, &buffers](
                       const void* data, const std::size_t size,
                       const std::size_t capacity)
  {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(data);
    valid = valid && hash.Update(&address, sizeof(address)) &&
        hash.Update(&size, sizeof(size)) &&
        hash.Update(&capacity, sizeof(capacity));
    if (data != nullptr) { ++buffers; }
  };
  const auto add_vector = [&add](const auto& values)
  { add(values.data(), values.size(), values.capacity()); };
  add_object(&owner);
  add_object(owner.volume_collection.get());
  add_object(owner.trace_collection.get());
  add_object(owner.mesh.get());
  add_object(owner.u_space.get());
  add_object(owner.sigma_space.get());
  add_object(owner.z_space.get());
  add_object(owner.uhat_space.get());
  add_object(owner.sigmahat_space.get());
  add_object(owner.carrier_space.get());
  add_object(owner.u.get());
  add_object(owner.sigma.get());
  add_object(owner.z.get());
  add_object(owner.uhat.get());
  add_object(owner.sigmahat.get());
  add_object(owner.carrier.get());
  add_object(&owner.carrier_map);
  add_object(&owner.topology);
  add_object(this);
  add_object(contract.get());
  add_object(&assembly_state);
  add_object(&assembly);
  add_object(&trial_assembly);
  add_object(&quadrature_workspace);
  add_object(&diagnostic);
  add_object(&elements);
  add_object(local_pattern.get());
  add_object(true_matrix.get());
  add_object(true_rhs.get());
  add_object(true_solution.get());
  add_object(ilu.get());
  add_object(gmres.get());
  add_object(&true_residual);
  add_object(&element_to_true_csr);
  add_object(&evidence);

  add_vector(owner.carrier_map.dofs.entries);
  add_vector(owner.carrier_map.faces);
  for (const trace::DGInterfaceFaceMap& face : owner.carrier_map.faces)
  {
    add_vector(face.signed_local_vdofs);
  }
  add_vector(owner.topology);
  for (const ElementTopology& topology : owner.topology)
  {
    add_vector(topology.volume.u);
    add_vector(topology.volume.sigma);
    add_vector(topology.volume.z);
    add_vector(topology.faces);
    add_vector(topology.face_orientations);
    add_vector(topology.signed_trace_vdofs);
    add_vector(topology.restriction);
  }
  const auto add_mfem_vector = [&add](const mfem::Vector* values)
  {
    if (values == nullptr)
    {
      add(nullptr, 0, 0);
      return;
    }
    add(values->GetData(), static_cast<std::size_t>(values->Size()),
        static_cast<std::size_t>(values->Size()));
  };
  add_mfem_vector(owner.u.get());
  add_mfem_vector(owner.sigma.get());
  add_mfem_vector(owner.z.get());
  add_mfem_vector(owner.uhat.get());
  add_mfem_vector(owner.sigmahat.get());
  add_mfem_vector(owner.carrier.get());
  add_vector(assembly_state.old_by_element);
  add_vector(assembly_state.candidate_by_element);
  add_vector(assembly_state.old_trace_by_element);
  add_vector(assembly_state.trace_by_element);
  for (const auto& values : assembly_state.old_by_element)
  {
    add_vector(values);
  }
  for (const auto& values : assembly_state.candidate_by_element)
  {
    add_vector(values);
  }
  for (const auto& values : assembly_state.old_trace_by_element)
  {
    add_vector(values);
  }
  for (const auto& values : assembly_state.trace_by_element)
  {
    add_vector(values);
  }
  const auto add_assembly = [&add_vector](const CoupledFixtureAssembly& value)
  {
    add_vector(value.elements);
    add_vector(value.full_residual);
    add_vector(value.full_jacobian);
    for (const ElementLinearization& block : value.elements)
    {
      add_vector(block.a);
      add_vector(block.b);
      add_vector(block.c);
      add_vector(block.d);
      add_vector(block.local_residual);
      add_vector(block.trace_residual);
    }
  };
  add_assembly(assembly);
  add_assembly(trial_assembly);
  add_vector(elements);
  for (const RuntimeElementWorkspace& element : elements)
  {
    add_object(&element);
    add_object(&element.factor);
    add_object(&element.condensed);
    add_vector(element.factor.values);
    add_vector(element.factor.pivots);
    add_vector(element.condensed.schur);
    add_vector(element.condensed.right_hand_side);
    add_vector(element.inverse_residual);
    add_vector(element.inverse_b);
    add_vector(element.rhs);
    add_vector(element.solution);
    add_vector(element.recovery_rhs);
    add_vector(element.recovered);
  }
  add_vector(element_to_true_csr);
  if (local_pattern)
  {
    add(local_pattern->GetI(),
        static_cast<std::size_t>(local_pattern->Height() + 1),
        static_cast<std::size_t>(local_pattern->Height() + 1));
    add(local_pattern->GetJ(),
        static_cast<std::size_t>(local_pattern->NumNonZeroElems()),
        static_cast<std::size_t>(local_pattern->NumNonZeroElems()));
    add(local_pattern->GetData(),
        static_cast<std::size_t>(local_pattern->NumNonZeroElems()),
        static_cast<std::size_t>(local_pattern->NumNonZeroElems()));
  }
  add(quadrature_workspace.volume_shape.GetData(),
      quadrature_workspace.volume_shape.Size(),
      quadrature_workspace.volume_shape.Size());
  add(quadrature_workspace.volume_derivative.GetData(),
      static_cast<std::size_t>(quadrature_workspace.volume_derivative.Height()) *
          quadrature_workspace.volume_derivative.Width(),
      static_cast<std::size_t>(quadrature_workspace.volume_derivative.Height()) *
          quadrature_workspace.volume_derivative.Width());
  add(quadrature_workspace.face_interior_shape.GetData(),
      quadrature_workspace.face_interior_shape.Size(),
      quadrature_workspace.face_interior_shape.Size());
  add(quadrature_workspace.face_trace_shape.GetData(),
      quadrature_workspace.face_trace_shape.Size(),
      quadrature_workspace.face_trace_shape.Size());
  add(quadrature_workspace.face_oriented_shape.GetData(),
      quadrature_workspace.face_oriented_shape.Size(),
      quadrature_workspace.face_oriented_shape.Size());
  add(quadrature_workspace.scaled_normal.GetData(),
      quadrature_workspace.scaled_normal.Size(),
      quadrature_workspace.scaled_normal.Size());
  add(true_residual.GetData(), true_residual.Size(), true_residual.Size());
  const std::array<std::uint64_t, 6> generations{{
      snapshot.topology_generation,
      snapshot.storage_generation,
      snapshot.factor_generation,
      snapshot.workspace_generation,
      snapshot.local_pattern_generation,
      snapshot.solver_generation}};
  valid = valid && hash.Update(generations.data(), sizeof(generations));
  Sha256Digest digest{};
  if (valid && hash.Finalize(digest))
  {
    snapshot.owned_boundary_identity_hash = {digest};
  }
  snapshot.w8_owned_boundary_object_count = objects;
  snapshot.w8_owned_boundary_buffer_count = buffers;
  return snapshot;
}

namespace
{

bool SameReuseSnapshot(const TrajectoryReuseSnapshot& left,
                       const TrajectoryReuseSnapshot& right) noexcept
{
  return left.foundation_handle == right.foundation_handle &&
      left.topology_handle == right.topology_handle &&
      left.carrier_map_handle == right.carrier_map_handle &&
      left.contract_handle == right.contract_handle &&
      left.assembly_state_handle == right.assembly_state_handle &&
      left.assembly_handle == right.assembly_handle &&
      left.trial_assembly_handle == right.trial_assembly_handle &&
      left.factor_handle == right.factor_handle &&
      left.workspace_handle == right.workspace_handle &&
      left.quadrature_workspace_handle == right.quadrature_workspace_handle &&
      left.local_pattern_handle == right.local_pattern_handle &&
      left.true_matrix_handle == right.true_matrix_handle &&
      left.true_rhs_handle == right.true_rhs_handle &&
      left.true_solution_handle == right.true_solution_handle &&
      left.ilu_handle == right.ilu_handle &&
      left.gmres_handle == right.gmres_handle &&
      left.true_residual_handle == right.true_residual_handle &&
      left.element_to_true_csr_handle == right.element_to_true_csr_handle &&
      left.owned_boundary_identity_hash ==
          right.owned_boundary_identity_hash &&
      left.topology_generation == right.topology_generation &&
      left.storage_generation == right.storage_generation &&
      left.factor_generation == right.factor_generation &&
      left.workspace_generation == right.workspace_generation &&
      left.local_pattern_generation == right.local_pattern_generation &&
      left.solver_generation == right.solver_generation &&
      left.w8_owned_boundary_object_count ==
          right.w8_owned_boundary_object_count &&
      left.w8_owned_boundary_buffer_count ==
          right.w8_owned_boundary_buffer_count &&
      !left.opaque_mfem_hypre_observed &&
      !right.opaque_mfem_hypre_observed;
}

} // namespace

bool PersistentTrajectoryImplementation::VerifyReuse(
    const PersistentSc25ScalarIgrCpuFoundation::Implementation& owner)
    noexcept
{
  const TrajectoryReuseSnapshot current = ReuseSnapshot(owner);
  evidence.reuse.current = current;
  if (!reuse_baseline_frozen) { return true; }
  if (!SameReuseSnapshot(evidence.reuse.baseline, current))
  {
    ++evidence.reuse.detected_w8_owned_boundary_identity_changes;
    return false;
  }
  return true;
}

bool PersistentTrajectoryImplementation::InitializeStorage(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner) noexcept
{
  try
  {
    if (owner.config.polynomial_degree != 1 ||
        owner.config.elements_per_direction !=
            persistent_elements_per_direction ||
        owner.mesh->GetNE() != persistent_elements ||
        owner.mesh->GetNumFaces() != persistent_unique_faces ||
        owner.carrier_space->GetTrueVSize() !=
            persistent_true_trace_unknowns ||
        owner.topology.size() != persistent_elements)
    {
      return false;
    }
    const auto initialize_state = [](CoupledFixtureState& state)
    {
      state.old_by_element.resize(persistent_elements);
      state.candidate_by_element.resize(persistent_elements);
      state.old_trace_by_element.resize(persistent_elements);
      state.trace_by_element.resize(persistent_elements);
      for (int element = 0; element < persistent_elements; ++element)
      {
        state.old_by_element[static_cast<std::size_t>(element)].assign(
            coupled_fixture_local_interior_unknowns, 0.0);
        state.candidate_by_element[static_cast<std::size_t>(element)].assign(
            coupled_fixture_local_interior_unknowns, 0.0);
        state.old_trace_by_element[static_cast<std::size_t>(element)].assign(
            coupled_fixture_local_trace_unknowns, 0.0);
        state.trace_by_element[static_cast<std::size_t>(element)].assign(
            coupled_fixture_local_trace_unknowns, 0.0);
      }
    };
    initialize_state(assembly_state);
    const auto initialize_assembly = [](CoupledFixtureAssembly& value)
    {
      value.elements.resize(persistent_elements);
      value.full_residual.assign(persistent_full_unknowns, 0.0);
      for (int element = 0; element < persistent_elements; ++element)
      {
        ElementLinearization& block =
            value.elements[static_cast<std::size_t>(element)];
        block.element = element;
        block.local_interior_unknowns =
            coupled_fixture_local_interior_unknowns;
        block.local_trace_unknowns = coupled_fixture_local_trace_unknowns;
        block.a.assign(
            static_cast<std::size_t>(coupled_fixture_local_interior_unknowns) *
                coupled_fixture_local_interior_unknowns,
            0.0);
        block.b.assign(
            static_cast<std::size_t>(coupled_fixture_local_interior_unknowns) *
                coupled_fixture_local_trace_unknowns,
            0.0);
        block.c.assign(
            static_cast<std::size_t>(coupled_fixture_local_trace_unknowns) *
                coupled_fixture_local_interior_unknowns,
            0.0);
        block.d.assign(
            static_cast<std::size_t>(coupled_fixture_local_trace_unknowns) *
                coupled_fixture_local_trace_unknowns,
            0.0);
        block.local_residual.assign(
            coupled_fixture_local_interior_unknowns, 0.0);
        block.trace_residual.assign(coupled_fixture_local_trace_unknowns,
                                    0.0);
      }
    };
    initialize_assembly(assembly);
    initialize_assembly(trial_assembly);
    elements.resize(persistent_elements);
    for (int element = 0; element < persistent_elements; ++element)
    {
      elements[static_cast<std::size_t>(element)].Initialize(element);
    }
    element_to_true_csr.assign(
        static_cast<std::size_t>(persistent_elements) *
            coupled_fixture_local_trace_unknowns *
            coupled_fixture_local_trace_unknowns,
        -1);
    true_residual.SetSize(persistent_true_trace_unknowns);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentTrajectoryImplementation::LoadAssemblyState(
    const PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    const TrajectorySlot& old_state,
    const TrajectorySlot& candidate) noexcept
{
  constexpr int ns = persistent_scalar_volume_nodes;
  constexpr int ni = coupled_fixture_local_interior_unknowns;
  constexpr int nt = coupled_fixture_local_trace_unknowns;
  try
  {
    for (int element = 0; element < persistent_elements; ++element)
    {
      std::vector<double>& old =
          assembly_state.old_by_element[static_cast<std::size_t>(element)];
      std::vector<double>& current = assembly_state.candidate_by_element[
          static_cast<std::size_t>(element)];
      const std::size_t u_base =
          static_cast<std::size_t>(element) * state_components * ns;
      const std::size_t sigma_base =
          static_cast<std::size_t>(element) * ns;
      const std::size_t z_base =
          static_cast<std::size_t>(element) * flux_components * ns;
      std::copy_n(old_state.interior.u.begin() + u_base,
                  state_components * ns, old.begin());
      std::copy_n(candidate.interior.u.begin() + u_base,
                  state_components * ns, current.begin());
      std::copy_n(old_state.interior.sigma.begin() + sigma_base, ns,
                  old.begin() + state_components * ns);
      std::copy_n(candidate.interior.sigma.begin() + sigma_base, ns,
                  current.begin() + state_components * ns);
      std::copy_n(old_state.interior.z.begin() + z_base,
                  flux_components * ns,
                  old.begin() + (state_components + 1) * ns);
      std::copy_n(candidate.interior.z.begin() + z_base,
                  flux_components * ns,
                  current.begin() + (state_components + 1) * ns);
      if (old.size() != ni || current.size() != ni) { return false; }

      std::vector<double>& old_trace = assembly_state.old_trace_by_element[
          static_cast<std::size_t>(element)];
      std::vector<double>& trace = assembly_state.trace_by_element[
          static_cast<std::size_t>(element)];
      const ElementTopology& topology =
          owner.topology[static_cast<std::size_t>(element)];
      if (topology.restriction.size() != nt) { return false; }
      for (int row = 0; row < nt; ++row)
      {
        const TraceRestrictionEntry& entry =
            topology.restriction[static_cast<std::size_t>(row)];
        const int true_dof = entry.local_true_dof;
        if (true_dof < 0 || true_dof >= persistent_true_trace_unknowns ||
            (entry.sign != 1 && entry.sign != -1))
        {
          return false;
        }
        const auto trace_value = [true_dof](const TrajectorySlot& slot)
        {
          return true_dof < persistent_uhat_unknowns
              ? slot.trace.uhat[static_cast<std::size_t>(true_dof)]
              : slot.trace.sigmahat[static_cast<std::size_t>(
                    true_dof - persistent_uhat_unknowns)];
        };
        old_trace[static_cast<std::size_t>(row)] =
            entry.sign * trace_value(old_state);
        trace[static_cast<std::size_t>(row)] =
            entry.sign * trace_value(candidate);
      }
    }
    return true;
  }
  catch (...)
  {
    return false;
  }
}

TrajectoryAssemblyFailureCause PersistentTrajectoryImplementation::Assemble(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    const TrajectorySlot& old_state, const TrajectorySlot& candidate,
    CoupledFixtureAssembly& output) noexcept
{
  diagnostic.Begin("C2-MFEM-PERSISTENT-N8-R1-01",
                   "persistent-assembly", "load-state");
  if (!LoadAssemblyState(owner, old_state, candidate))
  {
    diagnostic.LatchFailure();
    return TrajectoryAssemblyFailureCause::invalid_structure;
  }
  if (!AssembleCoupledFixture(
          *owner.mesh, *owner.sigma_space, *owner.carrier_space,
          *owner.trace_collection, owner.topology, assembly_state,
          coupled_fixture_quadrature_order, persistent_dt, false, output,
          quadrature_workspace, "persistent-assembly", diagnostic))
  {
    const TrajectoryAssemblyFailureCause cause = diagnostic.failure_cause;
    diagnostic.LatchFailure();
    return cause;
  }
  if (output.elements.size() != persistent_elements ||
      output.full_residual.size() != persistent_full_unknowns)
  {
    diagnostic.Set("persistent-assembly", "output-shape");
    diagnostic.LatchFailure();
    return TrajectoryAssemblyFailureCause::invalid_structure;
  }
  diagnostic.CompleteAttempt();
  return TrajectoryAssemblyFailureCause::none;
}

bool PersistentTrajectoryImplementation::InitializePhysicalState(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    TrajectorySlot& state) noexcept
{
  try
  {
    constexpr int ns = persistent_scalar_volume_nodes;
    constexpr int nf = coupled_fixture_scalar_face_nodes;
    constexpr int order = coupled_fixture_quadrature_order;
    state = {};
    *owner.u = 0.0;
    for (int element = 0; element < persistent_elements; ++element)
    {
      const mfem::FiniteElement* finite_element =
          owner.u_space->GetFE(element);
      mfem::ElementTransformation* element_transformation =
          owner.mesh->GetElementTransformation(element);
      if (!finite_element || !element_transformation ||
          finite_element->GetDof() != ns)
      {
        return false;
      }
      std::vector<double> mass(static_cast<std::size_t>(ns) * ns, 0.0);
      std::array<std::vector<double>, state_components> rhs;
      for (auto& values : rhs) { values.assign(ns, 0.0); }
      mfem::Vector shape(ns), point(dimension);
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), order);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(q);
        element_transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        element_transformation->Transform(ip, point);
        const InitialStatePoint value = SmoothInitialState(point[0], point[1]);
        const double weight = ip.weight * element_transformation->Weight();
        for (int row = 0; row < ns; ++row)
        {
          for (int column = 0; column < ns; ++column)
          {
            mass[static_cast<std::size_t>(row) * ns + column] +=
                weight * shape[row] * shape[column];
          }
          for (int component = 0; component < state_components; ++component)
          {
            rhs[static_cast<std::size_t>(component)]
               [static_cast<std::size_t>(row)] +=
                weight * shape[row] * value.conservative[component];
          }
        }
      }
      DenseLu factor;
      if (!factor.Factor(mass, ns)) { return false; }
      mfem::Vector local(state_components * ns);
      for (int component = 0; component < state_components; ++component)
      {
        std::vector<double> coefficients;
        if (!factor.Solve(rhs[static_cast<std::size_t>(component)],
                          coefficients))
        {
          return false;
        }
        for (int scalar = 0; scalar < ns; ++scalar)
        {
          const double coefficient = coefficients[static_cast<std::size_t>(scalar)];
          state.interior.u[
              static_cast<std::size_t>(element) * state_components * ns +
              static_cast<std::size_t>(component) * ns + scalar] = coefficient;
          local[component * ns + scalar] = coefficient;
        }
      }
      mfem::Array<int> vdofs;
      mfem::DofTransformation transformation(
          owner.u_space->GetVDim(), owner.u_space->GetOrdering());
      owner.u_space->GetElementVDofs(element, vdofs, transformation);
      if (vdofs.Size() != state_components * ns)
      {
        return false;
      }
      transformation.TransformPrimal(local);
      owner.u->SetSubVector(vdofs, local);
    }

    *owner.uhat = 0.0;
    std::array<std::uint8_t, persistent_uhat_unknowns> assigned{};
    for (int face = 0; face < owner.mesh->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* finite_element =
          owner.uhat_space->GetFaceElement(face);
      mfem::ElementTransformation* face_transformation =
          owner.mesh->GetFaceTransformation(face);
      mfem::Array<int> vdofs;
      owner.uhat_space->GetFaceVDofs(face, vdofs);
      if (!finite_element || !face_transformation ||
          finite_element->GetDof() != nf ||
          vdofs.Size() != state_components * nf)
      {
        return false;
      }
      std::vector<double> mass(static_cast<std::size_t>(nf) * nf, 0.0);
      std::array<std::vector<double>, state_components> rhs;
      for (auto& values : rhs) { values.assign(nf, 0.0); }
      mfem::Vector shape(nf), point(dimension);
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), order);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(q);
        face_transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        face_transformation->Transform(ip, point);
        const InitialStatePoint value = SmoothInitialState(point[0], point[1]);
        const double weight = ip.weight * face_transformation->Weight();
        for (int row = 0; row < nf; ++row)
        {
          for (int column = 0; column < nf; ++column)
          {
            mass[static_cast<std::size_t>(row) * nf + column] +=
                weight * shape[row] * shape[column];
          }
          for (int component = 0; component < state_components; ++component)
          {
            rhs[static_cast<std::size_t>(component)]
               [static_cast<std::size_t>(row)] +=
                weight * shape[row] * value.conservative[component];
          }
        }
      }
      DenseLu factor;
      if (!factor.Factor(mass, nf)) { return false; }
      for (int component = 0; component < state_components; ++component)
      {
        std::vector<double> coefficients;
        if (!factor.Solve(rhs[static_cast<std::size_t>(component)],
                          coefficients))
        {
          return false;
        }
        for (int scalar = 0; scalar < nf; ++scalar)
        {
          int sign = 0;
          const int dof = DecodeSignedDof(vdofs[component * nf + scalar], sign);
          const int true_dof = dof >= 0 ?
              owner.uhat_space->GetLocalTDofNumber(dof) : -1;
          if (dof < 0 || dof >= owner.uhat->Size() || true_dof < 0 ||
              true_dof >= persistent_uhat_unknowns || sign == 0)
          {
            return false;
          }
          const double true_value =
              sign * coefficients[static_cast<std::size_t>(scalar)];
          if (assigned[static_cast<std::size_t>(true_dof)] != 0 &&
              state.trace.uhat[static_cast<std::size_t>(true_dof)] !=
                  true_value)
          {
            return false;
          }
          state.trace.uhat[static_cast<std::size_t>(true_dof)] = true_value;
          assigned[static_cast<std::size_t>(true_dof)] = 1;
          (*owner.uhat)[dof] =
              state.trace.uhat[static_cast<std::size_t>(true_dof)];
        }
      }
    }
    if (!std::all_of(assigned.begin(), assigned.end(),
                     [](const std::uint8_t value) { return value == 1; }))
    {
      return false;
    }
    state.interior.sigma.fill(0.0);
    state.interior.z.fill(0.0);
    state.trace.sigmahat.fill(0.0);
    return std::all_of(state.interior.u.begin(), state.interior.u.end(),
                       [](const double value) { return Finite(value); }) &&
        std::all_of(state.trace.uhat.begin(), state.trace.uhat.end(),
                    [](const double value) { return Finite(value); });
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentTrajectoryImplementation::BuildPersistentTraceGraph(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner) noexcept
{
  try
  {
    local_pattern =
        std::make_unique<mfem::SparseMatrix>(owner.carrier_space->GetVSize());
    mfem::DenseMatrix seed(coupled_fixture_local_trace_unknowns);
    for (int row = 0; row < seed.Height(); ++row)
      for (int column = 0; column < seed.Width(); ++column)
      {
        seed(row, column) = 1.0 +
            1.0e-7 * static_cast<double>(row + 41 * column + 1);
      }
    for (const ElementTopology& topology : owner.topology)
    {
      mfem::Array<int> signed_dofs(coupled_fixture_local_trace_unknowns);
      for (int row = 0; row < signed_dofs.Size(); ++row)
      {
        signed_dofs[row] = topology.signed_trace_vdofs[
            static_cast<std::size_t>(row)];
      }
      local_pattern->AddSubMatrix(signed_dofs, signed_dofs, seed, 1);
    }
    local_pattern->Finalize();
    mfem::ParBilinearForm form(owner.carrier_space.get());
    true_matrix.reset(form.ParallelAssemble(local_pattern.get()));
    if (!true_matrix || true_matrix->Height() !=
                            persistent_true_trace_unknowns ||
        true_matrix->Width() != persistent_true_trace_unknowns ||
        true_matrix->GetGlobalNumRows() != persistent_true_trace_unknowns)
    {
      return false;
    }
    const int rows = persistent_true_trace_unknowns;
    const int nonzeros = static_cast<int>(
        true_matrix->GetDiagMemoryData().Capacity());
    const HYPRE_Int* offsets = true_matrix->GetDiagMemoryI().Read(
        mfem::MemoryClass::HOST, rows + 1);
    const HYPRE_Int* columns = true_matrix->GetDiagMemoryJ().Read(
        mfem::MemoryClass::HOST, nonzeros);
    if (offsets == nullptr || columns == nullptr || nonzeros <= 0)
    {
      return false;
    }
    for (int element = 0; element < persistent_elements; ++element)
    {
      const ElementTopology& topology =
          owner.topology[static_cast<std::size_t>(element)];
      for (int row = 0; row < coupled_fixture_local_trace_unknowns; ++row)
        for (int column = 0; column < coupled_fixture_local_trace_unknowns;
             ++column)
        {
          const int true_row = topology.restriction[
              static_cast<std::size_t>(row)].local_true_dof;
          const int true_column = topology.restriction[
              static_cast<std::size_t>(column)].local_true_dof;
          int found = -1;
          for (int entry = offsets[true_row]; entry < offsets[true_row + 1];
               ++entry)
          {
            if (columns[entry] == true_column)
            {
              found = entry;
              break;
            }
          }
          if (found < 0) { return false; }
          element_to_true_csr[
              (static_cast<std::size_t>(element) *
                   coupled_fixture_local_trace_unknowns + row) *
                  coupled_fixture_local_trace_unknowns +
              column] = found;
        }
    }
    true_matrix->HypreReadWrite();
    double* data = true_matrix->GetDiagMemoryData().ReadWrite(
        mfem::MemoryClass::HOST, nonzeros);
    std::fill(data, data + nonzeros, 0.0);
    true_rhs = std::make_unique<mfem::HypreParVector>(
        owner.carrier_space.get());
    true_solution = std::make_unique<mfem::HypreParVector>(
        owner.carrier_space.get());
    *true_rhs = 0.0;
    *true_solution = 0.0;
    ilu = std::make_unique<mfem::HypreILU>();
    gmres = std::make_unique<mfem::HypreGMRES>(owner.communicator.value);
    ConfigureCoupledIlu(*ilu);
    gmres->SetTol(1.0e-12);
    gmres->SetAbsTol(1.0e-14);
    gmres->SetMaxIter(persistent_trace_maximum_iterations);
    gmres->SetKDim(persistent_trace_restart_dimension);
    gmres->SetLogging(1);
    gmres->SetPrintLevel(0);
    gmres->SetZeroInitialIterate();
    gmres->SetPreconditioner(*ilu);
    gmres->SetOperator(*true_matrix);
    ConfigureCoupledIlu(*ilu);

    evidence.trace_restart_dimension = persistent_trace_restart_dimension;
    evidence.trace_maximum_iterations = persistent_trace_maximum_iterations;
    evidence.trace_relative_tolerance = 1.0e-12;
    evidence.trace_absolute_tolerance = 1.0e-14;
    evidence.ilu_type = 0;
    evidence.ilu_level_of_fill = 0;
    evidence.ilu_local_reordering = 1;
    evidence.reuse.true_matrix_nonzeros =
        static_cast<std::size_t>(nonzeros);
    evidence.reuse.opaque_mfem_hypre_observed = false;
    return true;
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentTrajectoryImplementation::InitializeAuxiliary(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    TrajectorySlot& state) noexcept
{
  try
  {
    constexpr int full_ni = coupled_fixture_local_interior_unknowns;
    constexpr int full_nt = coupled_fixture_local_trace_unknowns;
    constexpr int aux_ni = 12;
    constexpr int aux_nt = 8;
    const TrajectoryStateHashes physical_hashes = HashState(state);
    if (!true_matrix || !true_rhs || !true_solution || !ilu || !gmres)
    {
      return false;
    }
    const TrajectoryAssemblyFailureCause assembly_failure =
        Assemble(owner, state, state, assembly);
    if (assembly_failure != TrajectoryAssemblyFailureCause::none)
    {
      RecordAssemblyFailure(assembly_failure);
      return false;
    }
    for (int element = 0; element < persistent_elements; ++element)
    {
      ElementLinearization& target =
          assembly.elements[static_cast<std::size_t>(element)];
      std::array<double, aux_ni * aux_ni> aux_a{};
      std::array<double, aux_ni * aux_nt> aux_b{};
      std::array<double, aux_nt * aux_ni> aux_c{};
      std::array<double, aux_nt * aux_nt> aux_d{};
      std::array<double, aux_ni> aux_local_residual{};
      std::array<double, aux_nt> aux_trace_residual{};
      for (int row = 0; row < aux_ni; ++row)
      {
        aux_local_residual[static_cast<std::size_t>(row)] =
            target.local_residual[static_cast<std::size_t>(16 + row)];
        for (int column = 0; column < aux_ni; ++column)
        {
          aux_a[static_cast<std::size_t>(row) * aux_ni + column] =
              target.a[static_cast<std::size_t>(16 + row) * full_ni +
                       16 + column];
        }
        for (int column = 0; column < aux_nt; ++column)
        {
          aux_b[static_cast<std::size_t>(row) * aux_nt + column] =
              target.b[static_cast<std::size_t>(16 + row) * full_nt +
                       32 + column];
        }
      }
      for (int row = 0; row < aux_nt; ++row)
      {
        aux_trace_residual[static_cast<std::size_t>(row)] =
            target.trace_residual[static_cast<std::size_t>(32 + row)];
        for (int column = 0; column < aux_ni; ++column)
        {
          aux_c[static_cast<std::size_t>(row) * aux_ni + column] =
              target.c[static_cast<std::size_t>(32 + row) * full_ni +
                       16 + column];
        }
        for (int column = 0; column < aux_nt; ++column)
        {
          aux_d[static_cast<std::size_t>(row) * aux_nt + column] =
              target.d[static_cast<std::size_t>(32 + row) * full_nt +
                       32 + column];
        }
      }
      std::fill(target.a.begin(), target.a.end(), 0.0);
      std::fill(target.b.begin(), target.b.end(), 0.0);
      std::fill(target.c.begin(), target.c.end(), 0.0);
      std::fill(target.d.begin(), target.d.end(), 0.0);
      std::fill(target.local_residual.begin(),
                target.local_residual.end(), 0.0);
      std::fill(target.trace_residual.begin(),
                target.trace_residual.end(), 0.0);
      for (int row = 0; row < 16; ++row)
      {
        target.a[static_cast<std::size_t>(row) * full_ni + row] = 1.0;
      }
      for (int row = 0; row < 32; ++row)
      {
        target.d[static_cast<std::size_t>(row) * full_nt + row] = 1.0;
      }
      for (int row = 0; row < aux_ni; ++row)
      {
        target.local_residual[static_cast<std::size_t>(16 + row)] =
            aux_local_residual[static_cast<std::size_t>(row)];
        for (int column = 0; column < aux_ni; ++column)
        {
          target.a[static_cast<std::size_t>(16 + row) * full_ni +
                   16 + column] =
              aux_a[static_cast<std::size_t>(row) * aux_ni + column];
        }
        for (int column = 0; column < aux_nt; ++column)
        {
          target.b[static_cast<std::size_t>(16 + row) * full_nt +
                   32 + column] =
              aux_b[static_cast<std::size_t>(row) * aux_nt + column];
        }
      }
      for (int row = 0; row < aux_nt; ++row)
      {
        target.trace_residual[static_cast<std::size_t>(32 + row)] =
            aux_trace_residual[static_cast<std::size_t>(row)];
        for (int column = 0; column < aux_ni; ++column)
        {
          target.c[static_cast<std::size_t>(32 + row) * full_ni +
                   16 + column] =
              aux_c[static_cast<std::size_t>(row) * aux_ni + column];
        }
        for (int column = 0; column < aux_nt; ++column)
        {
          target.d[static_cast<std::size_t>(32 + row) * full_nt +
                   32 + column] =
              aux_d[static_cast<std::size_t>(row) * aux_nt + column];
        }
      }
    }
    double maximum_local_residual = 0.0;
    int trace_iterations = -1;
    double trace_residual_norm = 0.0;
    double trace_rhs_norm = 0.0;
    PersistentTrajectoryState direction;
    if (!Condense(assembly, maximum_local_residual) ||
        !AssembleTrueSystem(owner) ||
        !SolveTrueSystem(trace_iterations, trace_residual_norm,
                         trace_rhs_norm) ||
        !Recover(owner, assembly, direction, maximum_local_residual))
    {
      return false;
    }
    const TrajectorySlot solved = ApplyDirection(state, direction, 1.0);
    if (HashArrays(solved.interior.u) != physical_hashes.u ||
        HashArrays(solved.trace.uhat) != physical_hashes.uhat)
    {
      return false;
    }
    state = solved;
    AuxiliaryResidualMetrics residual;
    if (!AuxiliaryResidualGate(owner, state, residual) ||
        residual.normalized_maximum > 1.0e-10)
    {
      return false;
    }
    evidence.initialization = {
        coupled_fixture_quadrature_order,
        persistent_trace_restart_dimension,
        persistent_trace_maximum_iterations,
        1.0e-12,
        1.0e-14,
        0,
        0,
        1,
        trace_iterations,
        trace_residual_norm,
        trace_rhs_norm,
        residual};
    return true;
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentTrajectoryImplementation::Condense(
    const CoupledFixtureAssembly& system,
    double& maximum_residual) noexcept
{
  constexpr int ni = coupled_fixture_local_interior_unknowns;
  constexpr int nt = coupled_fixture_local_trace_unknowns;
  maximum_residual = 0.0;
  if (system.elements.size() != persistent_elements) { return false; }
  for (int element = 0; element < persistent_elements; ++element)
  {
    const ElementLinearization& block =
        system.elements[static_cast<std::size_t>(element)];
    RuntimeElementWorkspace& workspace =
        elements[static_cast<std::size_t>(element)];
    if (!workspace.factor.Factor(block.a, ni) ||
        !workspace.factor.Solve(block.local_residual,
                                workspace.inverse_residual))
    {
      return false;
    }
    double residual = ApplicationResidual(
        block.a, ni, workspace.inverse_residual, block.local_residual);
    maximum_residual = std::max(maximum_residual, residual);
    if (residual > local_absolute_tolerance +
            local_relative_tolerance * VectorNorm(block.local_residual))
    {
      return false;
    }
    for (int trace_column = 0; trace_column < nt; ++trace_column)
    {
      for (int row = 0; row < ni; ++row)
      {
        workspace.rhs[static_cast<std::size_t>(row)] = block.b[
            static_cast<std::size_t>(row) * nt + trace_column];
      }
      if (!workspace.factor.Solve(workspace.rhs, workspace.solution))
      {
        return false;
      }
      residual = ApplicationResidual(
          block.a, ni, workspace.solution, workspace.rhs);
      maximum_residual = std::max(maximum_residual, residual);
      if (residual > local_absolute_tolerance +
              local_relative_tolerance * VectorNorm(workspace.rhs))
      {
        return false;
      }
      for (int row = 0; row < ni; ++row)
      {
        workspace.inverse_b[static_cast<std::size_t>(row) * nt +
                            trace_column] =
            workspace.solution[static_cast<std::size_t>(row)];
      }
    }
    ElementCondensedBlock& result = workspace.condensed;
    result.maximum_local_application_residual = maximum_residual;
    std::fill(result.schur.begin(), result.schur.end(), 0.0);
    std::fill(result.right_hand_side.begin(),
              result.right_hand_side.end(), 0.0);
    for (int row = 0; row < nt; ++row)
    {
      double c_inverse_r = 0.0;
      for (int inner = 0; inner < ni; ++inner)
      {
        c_inverse_r +=
            block.c[static_cast<std::size_t>(row) * ni + inner] *
            workspace.inverse_residual[static_cast<std::size_t>(inner)];
      }
      result.right_hand_side[static_cast<std::size_t>(row)] =
          -block.trace_residual[static_cast<std::size_t>(row)] + c_inverse_r;
      for (int column = 0; column < nt; ++column)
      {
        double value = block.d[static_cast<std::size_t>(row) * nt + column];
        for (int inner = 0; inner < ni; ++inner)
        {
          value -= block.c[static_cast<std::size_t>(row) * ni + inner] *
              workspace.inverse_b[static_cast<std::size_t>(inner) * nt +
                                  column];
        }
        result.schur[static_cast<std::size_t>(row) * nt + column] = value;
      }
    }
    if (!Finite(result.schur) || !Finite(result.right_hand_side))
    {
      return false;
    }
  }
  return true;
}

bool PersistentTrajectoryImplementation::AssembleTrueSystem(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner) noexcept
{
  constexpr int nt = coupled_fixture_local_trace_unknowns;
  if (!true_matrix || !true_rhs ||
      owner.topology.size() != persistent_elements)
  {
    return false;
  }
  try
  {
    const int nonzeros = static_cast<int>(
        true_matrix->GetDiagMemoryData().Capacity());
    true_matrix->HypreReadWrite();
    double* matrix_data = true_matrix->GetDiagMemoryData().ReadWrite(
        mfem::MemoryClass::HOST, nonzeros);
    std::fill(matrix_data, matrix_data + nonzeros, 0.0);
    double* rhs = true_rhs->HostWrite();
    std::fill(rhs, rhs + persistent_true_trace_unknowns, 0.0);
    for (int element = 0; element < persistent_elements; ++element)
    {
      const ElementTopology& topology =
          owner.topology[static_cast<std::size_t>(element)];
      const ElementCondensedBlock& block =
          elements[static_cast<std::size_t>(element)].condensed;
      for (int row = 0; row < nt; ++row)
      {
        const TraceRestrictionEntry& row_entry = topology.restriction[
            static_cast<std::size_t>(row)];
        rhs[row_entry.local_true_dof] += row_entry.sign *
            block.right_hand_side[static_cast<std::size_t>(row)];
        for (int column = 0; column < nt; ++column)
        {
          const TraceRestrictionEntry& column_entry = topology.restriction[
              static_cast<std::size_t>(column)];
          const int csr = element_to_true_csr[
              (static_cast<std::size_t>(element) * nt + row) * nt + column];
          if (csr < 0 || csr >= nonzeros) { return false; }
          matrix_data[csr] += row_entry.sign * column_entry.sign *
              block.schur[static_cast<std::size_t>(row) * nt + column];
        }
      }
    }
    ++matrix_generation;
    return std::all_of(matrix_data, matrix_data + nonzeros,
                       [](const double value) { return Finite(value); }) &&
        std::all_of(rhs, rhs + persistent_true_trace_unknowns,
                    [](const double value) { return Finite(value); });
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentTrajectoryImplementation::SolveTrueSystem(
    int& iterations, double& residual_norm, double& rhs_norm) noexcept
{
  iterations = -1;
  residual_norm = std::numeric_limits<double>::infinity();
  rhs_norm = std::numeric_limits<double>::infinity();
  if (!true_matrix || !true_rhs || !true_solution || !ilu || !gmres)
  {
    return false;
  }
  try
  {
    *true_solution = 0.0;
    ConfigureCoupledIlu(*ilu);
    HYPRE_ClearAllErrors();
    true_rhs->HypreRead();
    true_solution->HypreReadWrite();
    const int setup_status = HYPRE_ParCSRGMRESSetup(
        static_cast<HYPRE_Solver>(*gmres), *true_matrix, *true_rhs,
        *true_solution);
    const int solve_status = setup_status == 0
        ? HYPRE_ParCSRGMRESSolve(static_cast<HYPRE_Solver>(*gmres),
                                *true_matrix, *true_rhs, *true_solution)
        : -1;
    HYPRE_Int converged = 0;
    const int query_status = HYPRE_GMRESGetConverged(
        static_cast<HYPRE_Solver>(*gmres), &converged);
    gmres->GetNumIterations(iterations);
    true_matrix->Mult(*true_solution, true_residual);
    true_residual -= *true_rhs;
    residual_norm = true_residual.Norml2();
    rhs_norm = true_rhs->Norml2();
    return setup_status == 0 && solve_status == 0 && query_status == 0 &&
        converged == 1 && iterations >= 0 &&
        iterations <= persistent_trace_maximum_iterations &&
        Finite(residual_norm) && Finite(rhs_norm) &&
        residual_norm <= 1.0e-14 + 1.0e-12 * rhs_norm;
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentTrajectoryImplementation::Recover(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    const CoupledFixtureAssembly& system,
    PersistentTrajectoryState& direction,
    double& maximum_residual) noexcept
{
  constexpr int ns = persistent_scalar_volume_nodes;
  constexpr int ni = coupled_fixture_local_interior_unknowns;
  constexpr int nt = coupled_fixture_local_trace_unknowns;
  direction = {};
  if (!true_solution || system.elements.size() != persistent_elements)
  {
    return false;
  }
  const double* trace_increment = true_solution->HostRead();
  std::copy_n(trace_increment, persistent_uhat_unknowns,
              direction.uhat.begin());
  std::copy_n(trace_increment + persistent_uhat_unknowns,
              persistent_sigmahat_unknowns, direction.sigmahat.begin());
  for (int element = 0; element < persistent_elements; ++element)
  {
    const ElementLinearization& block =
        system.elements[static_cast<std::size_t>(element)];
    RuntimeElementWorkspace& workspace =
        elements[static_cast<std::size_t>(element)];
    const ElementTopology& topology =
        owner.topology[static_cast<std::size_t>(element)];
    workspace.recovery_rhs = block.local_residual;
    for (int row = 0; row < ni; ++row)
      for (int column = 0; column < nt; ++column)
      {
        const TraceRestrictionEntry& entry =
            topology.restriction[static_cast<std::size_t>(column)];
        workspace.recovery_rhs[static_cast<std::size_t>(row)] +=
            block.b[static_cast<std::size_t>(row) * nt + column] *
            entry.sign * trace_increment[entry.local_true_dof];
      }
    if (!workspace.factor.Solve(workspace.recovery_rhs,
                                workspace.recovered))
    {
      return false;
    }
    const double residual = ApplicationResidual(
        block.a, ni, workspace.recovered, workspace.recovery_rhs);
    maximum_residual = std::max(maximum_residual, residual);
    if (residual > local_absolute_tolerance +
            local_relative_tolerance * VectorNorm(workspace.recovery_rhs))
    {
      return false;
    }
    const std::size_t u_base =
        static_cast<std::size_t>(element) * state_components * ns;
    const std::size_t sigma_base = static_cast<std::size_t>(element) * ns;
    const std::size_t z_base =
        static_cast<std::size_t>(element) * flux_components * ns;
    for (int entry = 0; entry < state_components * ns; ++entry)
    {
      direction.u[u_base + entry] =
          -workspace.recovered[static_cast<std::size_t>(entry)];
    }
    for (int entry = 0; entry < ns; ++entry)
    {
      direction.sigma[sigma_base + entry] =
          -workspace.recovered[static_cast<std::size_t>(16 + entry)];
    }
    for (int entry = 0; entry < flux_components * ns; ++entry)
    {
      direction.z[z_base + entry] =
          -workspace.recovered[static_cast<std::size_t>(20 + entry)];
    }
  }
  return true;
}

bool PersistentTrajectoryImplementation::Integrals(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    const TrajectorySlot& state,
    std::array<double, state_components>& values) noexcept
{
  values.fill(0.0);
  try
  {
    constexpr int ns = persistent_scalar_volume_nodes;
    for (int element = 0; element < persistent_elements; ++element)
    {
      const mfem::FiniteElement* finite_element =
          owner.u_space->GetFE(element);
      mfem::ElementTransformation* transformation =
          owner.mesh->GetElementTransformation(element);
      if (!finite_element || !transformation || finite_element->GetDof() != ns)
      {
        return false;
      }
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), coupled_fixture_quadrature_order);
      mfem::Vector& shape = quadrature_workspace.volume_shape;
      const std::size_t base =
          static_cast<std::size_t>(element) * state_components * ns;
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(point);
        transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        const double weight = ip.weight * transformation->Weight();
        for (int component = 0; component < state_components; ++component)
          for (int scalar = 0; scalar < ns; ++scalar)
          {
            values[component] += weight * shape[scalar] *
                state.interior.u[
                    base + static_cast<std::size_t>(component) * ns + scalar];
          }
      }
    }
    return Finite(values);
  }
  catch (...)
  {
    return false;
  }
}

bool PersistentTrajectoryImplementation::AuxiliaryResidualGate(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    const TrajectorySlot& state, AuxiliaryResidualMetrics& metrics) noexcept
{
  metrics = {};
  constexpr int ns = persistent_scalar_volume_nodes;
  constexpr int nf = coupled_fixture_local_scalar_trace_nodes;
  constexpr int rz_rows = persistent_elements * flux_components * ns;
  constexpr int rsigma_rows = persistent_elements * ns;
  std::array<std::array<double, rz_rows>, 3> rz_terms{};
  std::array<std::array<double, rsigma_rows>, 4> rsigma_terms{};
  std::array<double, persistent_sigmahat_unknowns> g_sigma{};
  double g_sigma_term_norm_sum = 0.0;
  if (!LoadAssemblyState(owner, state, state)) { return false; }
  try
  {
    for (int element = 0; element < persistent_elements; ++element)
    {
      const mfem::FiniteElement* finite_element =
          owner.sigma_space->GetFE(element);
      mfem::ElementTransformation* transformation =
          owner.mesh->GetElementTransformation(element);
      if (!finite_element || !transformation || finite_element->GetDof() != ns)
      {
        return false;
      }
      const std::vector<double>& current = assembly_state.candidate_by_element[
          static_cast<std::size_t>(element)];
      const std::vector<double>& trace = assembly_state.trace_by_element[
          static_cast<std::size_t>(element)];
      const ElementTopology& topology =
          owner.topology[static_cast<std::size_t>(element)];
      mfem::Vector& shape = quadrature_workspace.volume_shape;
      mfem::DenseMatrix& derivative = quadrature_workspace.volume_derivative;
      const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), coupled_fixture_quadrature_order);
      for (int point = 0; point < volume_rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
        transformation->SetIntPoint(&ip);
        finite_element->CalcShape(ip, shape);
        finite_element->CalcPhysDShape(*transformation, derivative);
        State conservative{};
        std::array<std::array<double, dimension>, state_components>
            conservative_derivative{};
        for (int component = 0; component < state_components; ++component)
        {
          conservative[component] =
              EvaluateComponent(current, component, ns, shape);
          for (int direction = 0; direction < dimension; ++direction)
          {
            conservative_derivative[component][direction] =
                EvaluateDerivative(current, component, ns, derivative,
                                   direction);
          }
        }
        if (!SampleAdmissible(conservative)) { return false; }
        const double sigma = EvaluateComponent(current, 4, ns, shape);
        const std::array<double, dimension> z{{
            EvaluateComponent(current, 5, ns, shape),
            EvaluateComponent(current, 6, ns, shape)}};
        const double rho = conservative[0];
        std::array<std::array<double, dimension>, dimension> gradient{};
        for (int direction = 0; direction < dimension; ++direction)
        {
          gradient[0][direction] =
              conservative_derivative[1][direction] / rho -
              conservative[1] * conservative_derivative[0][direction] /
                  (rho * rho);
          gradient[1][direction] =
              conservative_derivative[2][direction] / rho -
              conservative[2] * conservative_derivative[0][direction] /
                  (rho * rho);
        }
        const double q = VelocityGradientInvariant(gradient);
        const double weight = ip.weight * transformation->Weight();
        if (!Finite(weight) || !(weight > 0.0)) { return false; }
        for (int scalar = 0; scalar < ns; ++scalar)
        {
          for (int direction = 0; direction < dimension; ++direction)
          {
            const std::size_t row =
                static_cast<std::size_t>(element) * flux_components * ns +
                direction * ns + scalar;
            rz_terms[0][row] += weight * shape[scalar] * rho * z[direction];
            rz_terms[1][row] +=
                weight * derivative(scalar, direction) * sigma;
          }
          const std::size_t row =
              static_cast<std::size_t>(element) * ns + scalar;
          rsigma_terms[0][row] +=
              weight * shape[scalar] * sigma / rho;
          rsigma_terms[1][row] += primary_alpha * weight *
              (derivative(scalar, 0) * z[0] +
               derivative(scalar, 1) * z[1]);
          rsigma_terms[2][row] -=
              primary_alpha * weight * shape[scalar] * q;
        }
      }

      const double element_measure = ElementMeasureAtOrder(
          *owner.mesh, element, coupled_fixture_quadrature_order);
      for (int local_face = 0; local_face < faces_per_quadrilateral;
           ++local_face)
      {
        const int face = topology.faces[static_cast<std::size_t>(local_face)];
        mfem::FaceElementTransformations* face_transformation =
            owner.mesh->GetFaceElementTransformations(face);
        const mfem::FiniteElement* trace_element =
            owner.carrier_space->GetFaceElement(face);
        if (!face_transformation || !face_transformation->Face ||
            !trace_element || trace_element->GetDof() != 2)
        {
          return false;
        }
        const bool element_one = face_transformation->Elem1No == element;
        if (!element_one && face_transformation->Elem2No != element)
        {
          return false;
        }
        const int* permutation = owner.trace_collection->DofOrderForOrientation(
            mfem::Geometry::SEGMENT,
            topology.face_orientations[static_cast<std::size_t>(local_face)]);
        const double face_measure = FaceMeasureAtOrder(
            *owner.mesh, face, coupled_fixture_quadrature_order);
        const double height = element_measure / face_measure;
        if (!permutation || !(element_measure > 0.0) ||
            !(face_measure > 0.0) || !(height > 0.0))
        {
          return false;
        }
        const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
            trace_element->GetGeomType(), coupled_fixture_quadrature_order);
        mfem::Vector& interior_shape =
            quadrature_workspace.face_interior_shape;
        mfem::Vector& trace_shape = quadrature_workspace.face_trace_shape;
        mfem::Vector& oriented_shape =
            quadrature_workspace.face_oriented_shape;
        mfem::Vector& scaled = quadrature_workspace.scaled_normal;
        std::array<double, 2> side_g_sigma{};
        for (int point = 0; point < face_rule.GetNPoints(); ++point)
        {
          const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
          face_transformation->SetAllIntPoints(&ip);
          const mfem::IntegrationPoint& element_ip = element_one ?
              face_transformation->GetElement1IntPoint() :
              face_transformation->GetElement2IntPoint();
          finite_element->CalcShape(element_ip, interior_shape);
          trace_element->CalcShape(ip, trace_shape);
          for (int scalar = 0; scalar < 2; ++scalar)
          {
            int sign = 0;
            const int canonical = DecodeSignedDof(permutation[scalar], sign);
            if (canonical < 0 || canonical >= 2 || sign == 0) { return false; }
            oriented_shape[scalar] = sign * trace_shape[canonical];
          }
          mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled);
          const double normal_norm = scaled.Norml2();
          const double measure = face_transformation->Face->Weight();
          if (!(normal_norm > 0.0) || !(measure > 0.0)) { return false; }
          const double side = element_one ? 1.0 : -1.0;
          const std::array<double, dimension> normal{{
              side * scaled[0] / normal_norm,
              side * scaled[1] / normal_norm}};
          State conservative{}, trace_state{};
          for (int component = 0; component < state_components; ++component)
          {
            conservative[component] =
                EvaluateComponent(current, component, ns, interior_shape);
            for (int scalar = 0; scalar < 2; ++scalar)
            {
              trace_state[component] += trace[
                  static_cast<std::size_t>(component) * nf +
                  local_face * 2 + scalar] * oriented_shape[scalar];
            }
          }
          if (!SampleAdmissible(conservative) ||
              !SampleAdmissible(trace_state))
          {
            return false;
          }
          const double sigma =
              EvaluateComponent(current, 4, ns, interior_shape);
          const std::array<double, dimension> z{{
              EvaluateComponent(current, 5, ns, interior_shape),
              EvaluateComponent(current, 6, ns, interior_shape)}};
          double sigmahat = 0.0;
          for (int scalar = 0; scalar < 2; ++scalar)
          {
            sigmahat += trace[static_cast<std::size_t>(4) * nf +
                              local_face * 2 + scalar] *
                oriented_shape[scalar];
          }
          const double tau = 4.0 / (trace_state[0] * height);
          const double auxiliary_flux = z[0] * normal[0] +
              z[1] * normal[1] - tau * (sigma - sigmahat);
          const double weight = ip.weight * measure;
          if (!Finite(tau) || !(tau > 0.0) || !Finite(auxiliary_flux) ||
              !Finite(weight))
          {
            return false;
          }
          for (int scalar = 0; scalar < ns; ++scalar)
          {
            for (int direction = 0; direction < dimension; ++direction)
            {
              const std::size_t row =
                  static_cast<std::size_t>(element) * flux_components * ns +
                  direction * ns + scalar;
              rz_terms[2][row] -= weight * interior_shape[scalar] *
                  normal[direction] * sigmahat;
            }
            const std::size_t row =
                static_cast<std::size_t>(element) * ns + scalar;
            rsigma_terms[3][row] -= primary_alpha * weight *
                interior_shape[scalar] * auxiliary_flux;
          }
          for (int scalar = 0; scalar < 2; ++scalar)
          {
            side_g_sigma[static_cast<std::size_t>(scalar)] +=
                primary_alpha * weight * oriented_shape[scalar] *
                auxiliary_flux;
          }
        }
        double side_square = 0.0;
        for (int scalar = 0; scalar < 2; ++scalar)
        {
          const TraceRestrictionEntry& entry = topology.restriction[
              static_cast<std::size_t>(32 + local_face * 2 + scalar)];
          const int true_dof = entry.local_true_dof - persistent_uhat_unknowns;
          if (true_dof < 0 || true_dof >= persistent_sigmahat_unknowns)
          {
            return false;
          }
          const double contribution = entry.sign *
              side_g_sigma[static_cast<std::size_t>(scalar)];
          g_sigma[static_cast<std::size_t>(true_dof)] += contribution;
          side_square += contribution * contribution;
        }
        g_sigma_term_norm_sum += std::sqrt(side_square);
      }
    }

    const auto term_norm = [](const auto& values)
    {
      long double square = 0.0L;
      for (const double value : values) { square += value * value; }
      return std::sqrt(static_cast<double>(square));
    };
    std::array<double, rz_rows> r_z{};
    std::array<double, rsigma_rows> r_sigma{};
    for (std::size_t row = 0; row < r_z.size(); ++row)
      for (const auto& term : rz_terms) { r_z[row] += term[row]; }
    for (std::size_t row = 0; row < r_sigma.size(); ++row)
      for (const auto& term : rsigma_terms) { r_sigma[row] += term[row]; }
    metrics.r_z_norm = term_norm(r_z);
    metrics.r_sigma_norm = term_norm(r_sigma);
    metrics.g_sigma_norm = term_norm(g_sigma);
    metrics.r_z_scale = 0.0;
    for (const auto& term : rz_terms)
    {
      metrics.r_z_scale += term_norm(term);
    }
    metrics.r_z_scale = std::max(1.0, metrics.r_z_scale);
    metrics.r_sigma_scale = 0.0;
    for (const auto& term : rsigma_terms)
    {
      metrics.r_sigma_scale += term_norm(term);
    }
    metrics.r_sigma_scale = std::max(1.0, metrics.r_sigma_scale);
    metrics.g_sigma_scale = std::max(1.0, g_sigma_term_norm_sum);
    metrics.normalized_maximum = std::max({
        metrics.r_z_norm / metrics.r_z_scale,
        metrics.r_sigma_norm / metrics.r_sigma_scale,
        metrics.g_sigma_norm / metrics.g_sigma_scale});
    return Finite(metrics.r_z_norm) && Finite(metrics.r_z_scale) &&
        Finite(metrics.r_sigma_norm) && Finite(metrics.r_sigma_scale) &&
        Finite(metrics.g_sigma_norm) && Finite(metrics.g_sigma_scale) &&
        Finite(metrics.normalized_maximum);
  }
  catch (...)
  {
    metrics = {};
    return false;
  }
}

namespace
{

double ResidualNorm(const CoupledFixtureAssembly& assembly) noexcept
{
  return VectorNorm(assembly.full_residual);
}

TrajectorySlot ApplyDirection(const TrajectorySlot& entry,
                              const PersistentTrajectoryState& direction,
                              const double theta) noexcept
{
  TrajectorySlot result = entry;
  for (std::size_t i = 0; i < result.interior.u.size(); ++i)
  {
    result.interior.u[i] += theta * direction.u[i];
  }
  for (std::size_t i = 0; i < result.interior.sigma.size(); ++i)
  {
    result.interior.sigma[i] += theta * direction.sigma[i];
  }
  for (std::size_t i = 0; i < result.interior.z.size(); ++i)
  {
    result.interior.z[i] += theta * direction.z[i];
  }
  for (std::size_t i = 0; i < result.trace.uhat.size(); ++i)
  {
    result.trace.uhat[i] += theta * direction.uhat[i];
  }
  for (std::size_t i = 0; i < result.trace.sigmahat.size(); ++i)
  {
    result.trace.sigmahat[i] += theta * direction.sigmahat[i];
  }
  return result;
}

void NodalAdmissibilityMetrics(const TrajectorySlot& state,
                               double& minimum_density,
                               double& minimum_pressure,
                               double& minimum_speed_margin) noexcept
{
  minimum_density = std::numeric_limits<double>::infinity();
  minimum_pressure = std::numeric_limits<double>::infinity();
  minimum_speed_margin = std::numeric_limits<double>::infinity();
  constexpr int ns = persistent_scalar_volume_nodes;
  for (int element = 0; element < persistent_elements; ++element)
    for (int scalar = 0; scalar < ns; ++scalar)
    {
      State conservative{};
      const std::size_t base =
          static_cast<std::size_t>(element) * state_components * ns;
      for (int component = 0; component < state_components; ++component)
      {
        conservative[component] = state.interior.u[
            base + static_cast<std::size_t>(component) * ns + scalar];
      }
      double pressure = -std::numeric_limits<double>::infinity();
      std::array<double, dimension> velocity{};
      if (!Primitive(conservative, pressure, velocity)) { continue; }
      const double speed = std::hypot(velocity[0], velocity[1]);
      const double sound = std::sqrt(gamma * pressure / conservative[0]);
      minimum_density = std::min(minimum_density, conservative[0]);
      minimum_pressure = std::min(minimum_pressure, pressure);
      minimum_speed_margin =
          std::min(minimum_speed_margin, 2.5 - speed - sound);
    }
  for (int scalar = 0; scalar < persistent_scalar_true_trace_nodes; ++scalar)
  {
    State conservative{};
    for (int component = 0; component < state_components; ++component)
    {
      conservative[component] = state.trace.uhat[
          static_cast<std::size_t>(component) *
              persistent_scalar_true_trace_nodes + scalar];
    }
    double pressure = -std::numeric_limits<double>::infinity();
    std::array<double, dimension> velocity{};
    if (!Primitive(conservative, pressure, velocity)) { continue; }
    const double speed = std::hypot(velocity[0], velocity[1]);
    const double sound = std::sqrt(gamma * pressure / conservative[0]);
    minimum_density = std::min(minimum_density, conservative[0]);
    minimum_pressure = std::min(minimum_pressure, pressure);
    minimum_speed_margin =
        std::min(minimum_speed_margin, 2.5 - speed - sound);
  }
}

FailureReason ContractFailure(const TrajectoryStatus status) noexcept
{
  switch (status)
  {
    case TrajectoryStatus::HdgSc25InjectedAdmissibilityFailure:
    case TrajectoryStatus::HdgSc25AdmissibilityFailure:
      return FailureReason::admissibility_gate;
    case TrajectoryStatus::HdgSc25NonlinearResidualFailure:
      return FailureReason::true_residual_gate;
    case TrajectoryStatus::HdgSc25ConservationFailure:
      return FailureReason::success_gate;
    case TrajectoryStatus::HdgSc25PublicationFailure:
      return FailureReason::publication_precondition;
    default: return FailureReason::backend_failure;
  }
}

bool IsAccepted(const TrajectoryStatus status) noexcept
{
  return status == TrajectoryStatus::HdgSc25Accepted ||
      status == TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
}

} // namespace

void PersistentTrajectoryImplementation::RememberFailure(
    const TrajectoryStatus failure) noexcept
{
  if (!IsAccepted(failure) && failure != TrajectoryStatus::ready &&
      failure != TrajectoryStatus::uninitialized &&
      first_failure == TrajectoryStatus::uninitialized)
  {
    first_failure = failure;
  }
}

void PersistentTrajectoryImplementation::RecordAssemblyFailure(
    const TrajectoryAssemblyFailureCause failure) noexcept
{
  if (failure != TrajectoryAssemblyFailureCause::none &&
      evidence.first_assembly_failure ==
          TrajectoryAssemblyFailureCause::none)
  {
    evidence.first_assembly_failure = failure;
  }
  // The typed cause is now owned evidence. It is a handled result, so it must
  // not survive as destructor-style failure output in an otherwise passing
  // negative-control or rejected-line-search test.
  diagnostic.HandleFailure();
}

void PersistentTrajectoryImplementation::RefreshEvidence() noexcept
{
  evidence.status = status;
  evidence.first_failure = first_failure;
  if (contract)
  {
    evidence.contract_status = contract->status();
    evidence.contract_last_failure = contract->last_failure();
    evidence.contract_first_failure = contract->telemetry().first_failure;
    evidence.coordinates = contract->coordinates();
  }
}

TrajectoryStatus PersistentTrajectoryImplementation::ExecuteAttempt(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    const BeginOutcome& begin, const TrajectoryFailureInjection injection,
    TrajectoryAttemptEvidence& attempt) noexcept
{
  attempt = {};
  attempt.injection = injection;
  attempt.target_step = static_cast<int>(begin.attempt.committed_step + 1);
  attempt.retry_index = begin.attempt.retry_index;
  attempt.dt = begin.attempt.dt;
  attempt.committed_coordinates = {
      begin.attempt.committed_step * persistent_dt,
      begin.attempt.committed_step,
      begin.attempt.committed_generation,
      begin.attempt.committed_step,
      begin.attempt.committed_step == 0 ? 0.0 : persistent_dt};
  attempt.entry_reuse = ReuseSnapshot(owner);
  if (!begin.started || !contract)
  {
    attempt.status = TrajectoryStatus::HdgSc25PublicationFailure;
    return attempt.status;
  }
  TrajectorySlot committed;
  if (!contract->CopyPublished(contract->Published(), committed))
  {
    attempt.status = TrajectoryStatus::HdgSc25PublicationFailure;
    return attempt.status;
  }
  TrajectorySlot candidate = committed;
  attempt.entry_hashes = HashState(committed);
  std::array<double, state_components> committed_integrals{};
  if (!Integrals(owner, committed, committed_integrals))
  {
    attempt.status = TrajectoryStatus::HdgSc25ConservationFailure;
  }

  typename TrajectoryContract::ProductBundle products;
  bool product_residual = false;
  bool product_matrix = false;
  bool product_factor = false;
  bool product_solution = false;
  bool converged = false;
  TrajectoryStatus failure = attempt.status ==
          TrajectoryStatus::HdgSc25ConservationFailure
      ? attempt.status
      : TrajectoryStatus::uninitialized;
  double entry_norm = 0.0;
  double final_norm = std::numeric_limits<double>::infinity();
  double nonlinear_bound = std::numeric_limits<double>::infinity();

  for (int iteration = 0;
       failure == TrajectoryStatus::uninitialized &&
           iteration < persistent_newton_maximum_iterations;
       ++iteration)
  {
    const TrajectoryAssemblyFailureCause assembly_failure =
        Assemble(owner, committed, candidate, assembly);
    if (assembly_failure != TrajectoryAssemblyFailureCause::none)
    {
      RecordAssemblyFailure(assembly_failure);
      failure = assembly_failure ==
              TrajectoryAssemblyFailureCause::state_admissibility
          ? TrajectoryStatus::HdgSc25AdmissibilityFailure
          : TrajectoryStatus::HdgSc25AssemblyFailure;
      break;
    }
    const double residual_norm = ResidualNorm(assembly);
    if (!Finite(residual_norm))
    {
      failure = TrajectoryStatus::HdgSc25NonlinearResidualFailure;
      break;
    }
    attempt.attempted_newton_iterations = iteration + 1;
    if (iteration == 0)
    {
      entry_norm = residual_norm;
      nonlinear_bound = 1.0e-12 + 1.0e-10 * entry_norm;
      products.residual = contract->IssueProduct<
          DerivedProductKind::residual>(begin.attempt);
      product_residual = products.residual.has_value();
      if (!product_residual)
      {
        failure = TrajectoryStatus::HdgSc25PublicationFailure;
        break;
      }
    }
    final_norm = residual_norm;
    if (residual_norm <= nonlinear_bound)
    {
      converged = true;
      attempt.newton_iterations = iteration;
      break;
    }
    if (iteration == 0 &&
        injection == TrajectoryFailureInjection::newton_iteration_limit)
    {
      attempt.attempted_newton_iterations =
          persistent_newton_maximum_iterations;
      attempt.newton_iterations = 0;
      failure = TrajectoryStatus::HdgSc25InjectedNewtonIterationLimit;
      break;
    }
    if (!product_matrix)
    {
      products.matrix = contract->IssueProduct<DerivedProductKind::matrix>(
          begin.attempt);
      product_matrix = products.matrix.has_value();
      if (!product_matrix)
      {
        failure = TrajectoryStatus::HdgSc25PublicationFailure;
        break;
      }
    }
    if (iteration == 0 &&
        injection == TrajectoryFailureInjection::local_factor)
    {
      failure = TrajectoryStatus::HdgSc25InjectedLocalFactorFailure;
      break;
    }
    double maximum_local_residual = 0.0;
    if (!Condense(assembly, maximum_local_residual))
    {
      failure = TrajectoryStatus::HdgSc25LocalFactorFailure;
      break;
    }
    if (!product_factor)
    {
      products.factor = contract->IssueProduct<DerivedProductKind::factor>(
          begin.attempt);
      product_factor = products.factor.has_value();
      if (!product_factor)
      {
        failure = TrajectoryStatus::HdgSc25PublicationFailure;
        break;
      }
    }
    if (!AssembleTrueSystem(owner))
    {
      failure = TrajectoryStatus::HdgSc25TraceLinearFailure;
      break;
    }
    if (iteration == 0 &&
        injection == TrajectoryFailureInjection::trace_linear)
    {
      failure = TrajectoryStatus::HdgSc25InjectedTraceLinearFailure;
      break;
    }
    int trace_iterations = -1;
    double trace_residual_norm = 0.0;
    double trace_rhs_norm = 0.0;
    if (!SolveTrueSystem(trace_iterations, trace_residual_norm,
                         trace_rhs_norm))
    {
      failure = TrajectoryStatus::HdgSc25TraceLinearFailure;
      break;
    }
    PersistentTrajectoryState direction;
    if (!Recover(owner, assembly, direction, maximum_local_residual))
    {
      failure = TrajectoryStatus::HdgSc25LocalFactorFailure;
      break;
    }
    if (!product_solution)
    {
      products.solution = contract->IssueProduct<
          DerivedProductKind::solution>(begin.attempt);
      product_solution = products.solution.has_value();
      if (!product_solution)
      {
        failure = TrajectoryStatus::HdgSc25PublicationFailure;
        break;
      }
    }

    TrajectoryNewtonEvidence& newton =
        attempt.newton[static_cast<std::size_t>(iteration)];
    newton.iteration = iteration;
    newton.entry_residual_norm = residual_norm;
    newton.trace_iterations = trace_iterations;
    newton.trace_true_residual_norm = trace_residual_norm;
    newton.trace_right_hand_side_norm = trace_rhs_norm;
    newton.maximum_local_application_residual = maximum_local_residual;
    newton.entry = ToPublicState(candidate);
    newton.direction = direction;
    newton.trial_residual_norms.fill(
        std::numeric_limits<double>::infinity());
    bool accepted = false;
    for (int line = 0; line < persistent_line_search_trials; ++line)
    {
      const double theta = std::ldexp(1.0, -line);
      if (injection == TrajectoryFailureInjection::admissibility)
      {
        newton.trial_admissible[static_cast<std::size_t>(line)] = 0;
        continue;
      }
      const TrajectorySlot trial = ApplyDirection(candidate, direction, theta);
      const TrajectoryAssemblyFailureCause trial_failure =
          Assemble(owner, committed, trial, trial_assembly);
      if (trial_failure != TrajectoryAssemblyFailureCause::none)
      {
        if (trial_failure ==
            TrajectoryAssemblyFailureCause::state_admissibility)
        {
          diagnostic.HandleFailure();
          newton.trial_admissible[static_cast<std::size_t>(line)] = 0;
          continue;
        }
        RecordAssemblyFailure(trial_failure);
        failure = TrajectoryStatus::HdgSc25AssemblyFailure;
        break;
      }
      newton.trial_admissible[static_cast<std::size_t>(line)] = 1;
      const double trial_norm = ResidualNorm(trial_assembly);
      newton.trial_residual_norms[static_cast<std::size_t>(line)] = trial_norm;
      const bool armijo = Finite(trial_norm) &&
          trial_norm <= (1.0 - 1.0e-4 * theta) * residual_norm;
      newton.trial_armijo[static_cast<std::size_t>(line)] = armijo ? 1 : 0;
      if (armijo)
      {
        candidate = trial;
        newton.accepted_line_search_index = line;
        newton.accepted_residual_norm = trial_norm;
        accepted = true;
        break;
      }
    }
    attempt.newton_iterations = iteration + 1;
    if (failure != TrajectoryStatus::uninitialized) { break; }
    if (!accepted)
    {
      failure = injection == TrajectoryFailureInjection::admissibility
          ? TrajectoryStatus::HdgSc25InjectedAdmissibilityFailure
          : TrajectoryStatus::HdgSc25AdmissibilityFailure;
      break;
    }
  }

  if (failure == TrajectoryStatus::uninitialized && !converged)
  {
    failure = TrajectoryStatus::HdgSc25NewtonIterationLimit;
  }
  attempt.entry_residual_norm = entry_norm;
  attempt.final_residual_norm = final_norm;
  attempt.nonlinear_residual_bound = nonlinear_bound;

  GateReport gates{};
  AttemptTelemetry telemetry{};
  telemetry.transfers.complete = true;
  telemetry.true_relative_residual =
      Finite(final_norm) ? final_norm / std::max(1.0, entry_norm)
                         : std::numeric_limits<double>::infinity();
  FailureReason originating = FailureReason::none;
  TrajectoryStatus result = failure;
  std::array<double, state_components> accepted_integrals{};
  if (failure == TrajectoryStatus::uninitialized && !VerifyReuse(owner))
  {
    failure = TrajectoryStatus::HdgSc25PublicationFailure;
    result = failure;
  }
  if (failure == TrajectoryStatus::uninitialized && converged)
  {
    if (!Integrals(owner, candidate, accepted_integrals))
    {
      result = TrajectoryStatus::HdgSc25ConservationFailure;
    }
    else
    {
      double maximum_conservation = 0.0;
      for (int component = 0; component < state_components; ++component)
      {
        maximum_conservation = std::max(
            maximum_conservation,
            std::abs(accepted_integrals[component] -
                     committed_integrals[component]) /
                std::max(1.0, std::abs(committed_integrals[component])));
      }
      attempt.maximum_conservation_error = maximum_conservation;
      if (maximum_conservation > 2.0e-10)
      {
        result = TrajectoryStatus::HdgSc25ConservationFailure;
      }
      else
      {
        auto capability = contract->AcquireCandidate(begin.attempt);
        if (!capability ||
            !contract->StoreCandidate(*capability, candidate))
        {
          result = TrajectoryStatus::HdgSc25PublicationFailure;
        }
        else
        {
          gates = {true, true, true, true, true, true};
          result = TrajectoryStatus::HdgSc25Accepted;
        }
      }
    }
  }
  if (!IsAccepted(result))
  {
    originating = ContractFailure(result);
  }
  attempt.candidate_hashes = HashState(candidate);
  NodalAdmissibilityMetrics(candidate, attempt.minimum_density,
                            attempt.minimum_pressure,
                            attempt.minimum_speed_margin);
  const AttemptReport report{gates,
                             telemetry,
                             static_cast<std::uint64_t>(attempt.target_step),
                             originating};
  const AttemptOutcome outcome = contract->FinishAttempt(
      begin.attempt, report, std::move(products));
  TrajectorySlot published;
  if (!contract->CopyPublished(contract->Published(), published))
  {
    result = TrajectoryStatus::HdgSc25PublicationFailure;
  }
  attempt.published_hashes = HashState(published);
  if (IsAccepted(result) && !outcome.committed)
  {
    result = TrajectoryStatus::HdgSc25PublicationFailure;
  }
  if (!IsAccepted(result) && outcome.committed)
  {
    result = TrajectoryStatus::HdgSc25PublicationFailure;
  }
  attempt.status = result;
  attempt.published_coordinates = contract->coordinates();
  attempt.exit_reuse = ReuseSnapshot(owner);
  return result;
}

TrajectoryStatus PersistentTrajectoryImplementation::Advance(
    PersistentSc25ScalarIgrCpuFoundation::Implementation& owner,
    const TrajectoryFailureInjection injection) noexcept
{
  if (!initialized || !contract ||
      evidence.attempt_count >= static_cast<int>(evidence.attempts.size()) ||
      contract->coordinates().step >= persistent_required_steps ||
      (injection != TrajectoryFailureInjection::none &&
       contract->coordinates().step + 1 != 7))
  {
    status = TrajectoryStatus::invalid_configuration;
    RememberFailure(status);
    RefreshEvidence();
    return status;
  }
  TrajectorySlot before;
  if (!contract->CopyPublished(contract->Published(), before))
  {
    status = TrajectoryStatus::HdgSc25PublicationFailure;
    RememberFailure(status);
    RefreshEvidence();
    return status;
  }
  std::array<double, state_components> before_integrals{};
  if (!Integrals(owner, before, before_integrals))
  {
    status = TrajectoryStatus::HdgSc25ConservationFailure;
    RememberFailure(status);
    RefreshEvidence();
    return status;
  }

  const BeginOutcome begin =
      contract->BeginStep({InteriorSpace::l2, persistent_dt});
  const int first_attempt_index = evidence.attempt_count++;
  TrajectoryStatus result = ExecuteAttempt(
      owner, begin, injection,
      evidence.attempts[static_cast<std::size_t>(first_attempt_index)]);
  ++evidence.attempted_steps;
  int accepted_attempt_index = first_attempt_index;
  if (!IsAccepted(result))
  {
    RememberFailure(result);
    if (contract->status() != RunStatus::retry_available ||
        evidence.attempt_count >= static_cast<int>(evidence.attempts.size()))
    {
      status = TrajectoryStatus::HdgSc25RetryExhausted;
      RememberFailure(status);
      RefreshEvidence();
      return status;
    }
    const BeginOutcome retry = contract->Retry();
    accepted_attempt_index = evidence.attempt_count++;
    ++evidence.retry_attempts;
    result = ExecuteAttempt(
        owner, retry, TrajectoryFailureInjection::none,
        evidence.attempts[static_cast<std::size_t>(accepted_attempt_index)]);
    if (!IsAccepted(result))
    {
      RememberFailure(result);
      status = TrajectoryStatus::HdgSc25RetryExhausted;
      RememberFailure(status);
      RefreshEvidence();
      return status;
    }
    result = TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
    evidence.attempts[static_cast<std::size_t>(accepted_attempt_index)].status =
        result;
  }

  TrajectorySlot accepted;
  std::array<double, state_components> accepted_integrals{};
  if (!contract->CopyPublished(contract->Published(), accepted) ||
      !Integrals(owner, accepted, accepted_integrals))
  {
    status = TrajectoryStatus::HdgSc25PublicationFailure;
    RememberFailure(status);
    RefreshEvidence();
    return status;
  }
  const StateCoordinates coordinates = contract->coordinates();
  if (coordinates.step == 0 ||
      coordinates.step > persistent_required_steps ||
      coordinates.generation != coordinates.step ||
      coordinates.cursor != coordinates.step ||
      detail::DoubleBits(coordinates.dt) !=
          detail::DoubleBits(persistent_dt))
  {
    status = TrajectoryStatus::HdgSc25PublicationFailure;
    RememberFailure(status);
    RefreshEvidence();
    return status;
  }
  if (evidence.step_evidence_count < 0 ||
      evidence.step_evidence_count >=
          static_cast<int>(evidence.steps.size()))
  {
    status = TrajectoryStatus::HdgSc25PublicationFailure;
    RememberFailure(status);
    RefreshEvidence();
    return status;
  }
  TrajectoryStepEvidence& step = evidence.steps[
      static_cast<std::size_t>(evidence.step_evidence_count)];
  step.accepted = true;
  step.step = static_cast<int>(coordinates.step);
  step.time = coordinates.time;
  step.dt = coordinates.dt;
  step.generation = coordinates.generation;
  step.attempt_index = accepted_attempt_index;
  step.committed_integrals = before_integrals;
  step.accepted_integrals = accepted_integrals;
  step.accepted_state = ToPublicState(accepted);
  step.accepted_hashes = HashState(accepted);
  step.reuse = ReuseSnapshot(owner);
  if (!VerifyReuse(owner))
  {
    status = TrajectoryStatus::HdgSc25PublicationFailure;
    RememberFailure(status);
    RefreshEvidence();
    return status;
  }
  evidence.accepted_steps = static_cast<int>(coordinates.step);
  ++evidence.step_evidence_count;
  status = result;
  RefreshEvidence();
  return status;
}

PersistentSc25ScalarIgrCpuFoundation::Implementation::~Implementation()
    noexcept = default;

bool PersistentSc25ScalarIgrCpuFoundation::RunCoupledOperatorFixture(
    CoupledOperatorFixtureSnapshot& output) noexcept
{
  output = {};
  CoupledFixturePrivateDiagnostic diagnostic;
  if (!implementation_ ||
      implementation_->config.polynomial_degree !=
          coupled_fixture_polynomial_degree ||
      implementation_->config.elements_per_direction !=
          coupled_fixture_elements_per_direction ||
      implementation_->ranks != 1 || implementation_->rank != 0)
  {
    return false;
  }
  try
  {
    CoupledFixtureState state;
    if (!MaterializeCoupledFixture(
            *implementation_->mesh, *implementation_->sigma_space,
            *implementation_->carrier_space, *implementation_->carrier,
            implementation_->topology, state, diagnostic))
    {
      return false;
    }
    CoupledFixtureAssembly primary, repeat;
    CoupledAssemblyWorkspace workspace;
    if (!AssembleCoupledFixture(
            *implementation_->mesh, *implementation_->sigma_space,
            *implementation_->carrier_space,
            *implementation_->trace_collection, implementation_->topology,
            state, coupled_fixture_quadrature_order, coupled_fixture_dt,
            true, primary, workspace,
            "primary-assembly", diagnostic))
    {
      return false;
    }
    if (!AssembleCoupledFixture(
            *implementation_->mesh, *implementation_->sigma_space,
            *implementation_->carrier_space,
            *implementation_->trace_collection, implementation_->topology,
            state, coupled_fixture_repeat_quadrature_order,
            coupled_fixture_dt, true, repeat, workspace,
            "repeat-assembly", diagnostic))
    {
      return false;
    }
    diagnostic.Set("assembly-validation", "element-census");
    if (primary.elements.size() != coupled_fixture_elements ||
        repeat.elements.size() != coupled_fixture_elements)
    {
      return false;
    }

    std::vector<DenseLu> factors(coupled_fixture_elements);
    std::vector<ElementCondensedBlock> condensed(coupled_fixture_elements);
    double maximum_local_residual = 0.0;
    for (std::size_t element = 0; element < primary.elements.size();
         ++element)
    {
      diagnostic.Set("local-condensation", "factor-and-solves",
                     static_cast<int>(element));
      const ElementLinearization& block = primary.elements[element];
      constexpr int ni = coupled_fixture_local_interior_unknowns;
      constexpr int nt = coupled_fixture_local_trace_unknowns;
      diagnostic.expected_block_element = static_cast<int>(element);
      diagnostic.actual_block_element = block.element;
      diagnostic.block_interior_unknowns = block.local_interior_unknowns;
      diagnostic.block_trace_unknowns = block.local_trace_unknowns;
      diagnostic.block_a_entries = block.a.size();
      diagnostic.block_b_entries = block.b.size();
      diagnostic.block_c_entries = block.c.size();
      diagnostic.block_d_entries = block.d.size();
      diagnostic.block_local_residual_entries = block.local_residual.size();
      diagnostic.block_trace_residual_entries = block.trace_residual.size();
      diagnostic.Set("local-condensation", "block-identity-extent",
                     static_cast<int>(element));
      if (block.element != static_cast<int>(element) ||
          block.local_interior_unknowns != ni ||
          block.local_trace_unknowns != nt)
      {
        return false;
      }
      diagnostic.Set("local-condensation", "dense-lu-factor",
                     static_cast<int>(element));
      const bool factor_succeeded = factors[element].Factor(block.a, ni);
      diagnostic.lu_failure_column =
          factors[element].factor_failure_column;
      diagnostic.lu_pivot_magnitude =
          factors[element].factor_pivot_magnitude;
      diagnostic.lu_pivot_threshold =
          factors[element].factor_pivot_threshold;
      diagnostic.lu_failure_reason =
          factors[element].factor_failure_reason;
      if (!factor_succeeded) { return false; }
      std::vector<double> inverse_residual;
      diagnostic.Set("local-condensation", "local-residual-solve",
                     static_cast<int>(element));
      if (!SolveChecked(factors[element], block.a, block.local_residual,
                        inverse_residual, maximum_local_residual, diagnostic,
                        "local-residual", -1))
      {
        return false;
      }
      std::vector<double> inverse_b(static_cast<std::size_t>(ni) * nt, 0.0);
      std::vector<double> rhs(ni), solution;
      for (int trace_column = 0; trace_column < nt; ++trace_column)
      {
        diagnostic.Set("local-condensation", "inverse-b-solve",
                       static_cast<int>(element));
        for (int row = 0; row < ni; ++row)
        {
          rhs[row] = block.b[static_cast<std::size_t>(row) * nt +
                             trace_column];
        }
        if (!SolveChecked(factors[element], block.a, rhs, solution,
                          maximum_local_residual, diagnostic, "A-inverse-B",
                          trace_column))
        {
          return false;
        }
        for (int row = 0; row < ni; ++row)
        {
          inverse_b[static_cast<std::size_t>(row) * nt + trace_column] =
              solution[row];
        }
      }
      ElementCondensedBlock& result = condensed[element];
      result.element = static_cast<int>(element);
      result.local_trace_unknowns = nt;
      result.schur.assign(static_cast<std::size_t>(nt) * nt, 0.0);
      result.right_hand_side.assign(nt, 0.0);
      for (int row = 0; row < nt; ++row)
      {
        result.right_hand_side[row] = -block.trace_residual[row];
        for (int inner = 0; inner < ni; ++inner)
        {
          result.right_hand_side[row] +=
              block.c[static_cast<std::size_t>(row) * ni + inner] *
              inverse_residual[inner];
        }
        for (int column = 0; column < nt; ++column)
        {
          result.schur[static_cast<std::size_t>(row) * nt + column] =
              block.d[static_cast<std::size_t>(row) * nt + column];
          for (int inner = 0; inner < ni; ++inner)
          {
            result.schur[static_cast<std::size_t>(row) * nt + column] -=
                block.c[static_cast<std::size_t>(row) * ni + inner] *
                inverse_b[static_cast<std::size_t>(inner) * nt + column];
          }
        }
      }
      result.maximum_local_application_residual = maximum_local_residual;
      if (!Finite(result.schur) || !Finite(result.right_hand_side))
      {
        return false;
      }
    }

    diagnostic.Set("true-assembly", "parallel-assemble");
    if (!AssembleTrueTraceSystem(condensed)) { return false; }
    diagnostic.Set("true-assembly", "retained-system");
    const mfem::HypreParMatrix* true_matrix =
        assembled_true_trace_matrix();
    const mfem::Vector* true_rhs =
        assembled_true_trace_right_hand_side();
    if (true_matrix == nullptr || true_rhs == nullptr ||
        true_matrix->Height() != coupled_fixture_true_trace_unknowns ||
        true_matrix->Width() != coupled_fixture_true_trace_unknowns ||
        true_rhs->Size() != coupled_fixture_true_trace_unknowns)
    {
      return false;
    }
    diagnostic.Set("trace-solve", "vector-preparation");
    mfem::HypreParVector hypre_rhs(implementation_->carrier_space.get());
    mfem::HypreParVector hypre_solution(
        implementation_->carrier_space.get());
    std::copy(true_rhs->HostRead(), true_rhs->HostRead() + true_rhs->Size(),
              hypre_rhs.HostWrite());
    hypre_solution = 0.0;
    diagnostic.Set("trace-solve", "configuration");
    mfem::HypreILU ilu;
    mfem::HypreGMRES gmres(implementation_->communicator.value);
    ConfigureCoupledIlu(ilu);
    gmres.SetTol(1.0e-12);
    gmres.SetAbsTol(1.0e-14);
    gmres.SetMaxIter(200);
    gmres.SetKDim(50);
    gmres.SetLogging(1);
    gmres.SetPrintLevel(0);
    gmres.SetZeroInitialIterate();
    gmres.SetPreconditioner(ilu);
    gmres.SetOperator(*true_matrix);
    ConfigureCoupledIlu(ilu);
    diagnostic.Set("trace-solve", "hypre-setup");
    hypre_rhs.HypreRead();
    hypre_solution.HypreReadWrite();
    const int setup_status = HYPRE_ParCSRGMRESSetup(
        static_cast<HYPRE_Solver>(gmres), *true_matrix, hypre_rhs,
        hypre_solution);
    diagnostic.hypre_setup_reached = true;
    diagnostic.setup_status = setup_status;
    diagnostic.Set("trace-solve", "hypre-solve");
    const int solve_status = setup_status == 0 ? HYPRE_ParCSRGMRESSolve(
        static_cast<HYPRE_Solver>(gmres), *true_matrix, hypre_rhs,
        hypre_solution) : -1;
    diagnostic.hypre_solve_reached = setup_status == 0;
    diagnostic.solve_status = solve_status;
    diagnostic.Set("trace-solve", "hypre-query");
    HYPRE_Int hypre_converged = -1;
    const int convergence_query_status = HYPRE_GMRESGetConverged(
        static_cast<HYPRE_Solver>(gmres), &hypre_converged);
    const int converged = static_cast<int>(hypre_converged);
    diagnostic.hypre_query_reached = true;
    diagnostic.convergence_query_status = convergence_query_status;
    diagnostic.converged = converged;
    int iterations = -1;
    gmres.GetNumIterations(iterations);
    diagnostic.iteration_query_reached = true;
    diagnostic.iterations = iterations;
    diagnostic.Set("trace-solve", "true-residual");
    mfem::Vector residual(coupled_fixture_true_trace_unknowns);
    true_matrix->Mult(hypre_solution, residual);
    residual -= *true_rhs;
    double local_residual_square = 0.0;
    double local_rhs_square = 0.0;
    for (int entry = 0; entry < residual.Size(); ++entry)
    {
      local_residual_square += residual[entry] * residual[entry];
      local_rhs_square += (*true_rhs)[entry] * (*true_rhs)[entry];
    }
    double residual_square = 0.0;
    double rhs_square = 0.0;
    if (MPI_Allreduce(&local_residual_square, &residual_square, 1,
                      MPI_DOUBLE, MPI_SUM,
                      implementation_->communicator.value) != MPI_SUCCESS ||
        MPI_Allreduce(&local_rhs_square, &rhs_square, 1, MPI_DOUBLE, MPI_SUM,
                      implementation_->communicator.value) != MPI_SUCCESS)
    {
      return false;
    }
    const double residual_norm = std::sqrt(residual_square);
    const double rhs_norm = std::sqrt(rhs_square);
    const double residual_bound = 1.0e-14 + 1.0e-12 * rhs_norm;
    diagnostic.true_residual_reached = true;
    diagnostic.true_residual_norm = residual_norm;
    diagnostic.right_hand_side_norm = rhs_norm;
    diagnostic.true_residual_bound = residual_bound;
    if (setup_status != 0 || solve_status != 0 ||
        convergence_query_status != 0 || converged != 1 ||
        iterations < 0 || iterations > 200 ||
        !Finite(residual_norm) || !Finite(rhs_norm) ||
        residual_norm > residual_bound)
    {
      return false;
    }
    diagnostic.Set("restriction-recovery", "true-to-carrier");
    std::vector<double> true_increment(
        coupled_fixture_true_trace_unknowns);
    std::copy(hypre_solution.HostRead(),
              hypre_solution.HostRead() + hypre_solution.Size(),
              true_increment.begin());
    mfem::Vector true_increment_vector(coupled_fixture_true_trace_unknowns);
    std::copy(true_increment.begin(), true_increment.end(),
              true_increment_vector.HostWrite());
    if (!SetTraceCarrierFromTrueDofs(true_increment_vector)) { return false; }

    std::vector<double> element_trace_increment(
        coupled_fixture_restriction_rows, 0.0);
    std::vector<double> recovered_interior(
        coupled_fixture_global_interior_unknowns, 0.0);
    mfem::ParGridFunction recovered_u(implementation_->u_space.get());
    mfem::ParGridFunction recovered_sigma(
        implementation_->sigma_space.get());
    mfem::ParGridFunction recovered_z(implementation_->z_space.get());
    recovered_u = 0.0;
    recovered_sigma = 0.0;
    recovered_z = 0.0;
    for (int element = 0; element < coupled_fixture_elements; ++element)
    {
      diagnostic.Set("restriction-recovery", "element-recovery", element);
      const ElementTopology& map =
          implementation_->topology[static_cast<std::size_t>(element)];
      mfem::Array<int> signed_dofs(coupled_fixture_local_trace_unknowns);
      for (int row = 0; row < signed_dofs.Size(); ++row)
      {
        signed_dofs[row] = map.signed_trace_vdofs[
            static_cast<std::size_t>(row)];
      }
      mfem::Vector local_trace(coupled_fixture_local_trace_unknowns);
      implementation_->carrier->GetSubVector(signed_dofs, local_trace);
      std::copy(local_trace.HostRead(),
                local_trace.HostRead() + local_trace.Size(),
                element_trace_increment.begin() +
                    static_cast<std::size_t>(element) * local_trace.Size());
      const ElementLinearization& block =
          primary.elements[static_cast<std::size_t>(element)];
      std::vector<double> recovery_rhs = block.local_residual;
      for (int row = 0; row < coupled_fixture_local_interior_unknowns; ++row)
        for (int column = 0; column < coupled_fixture_local_trace_unknowns;
             ++column)
        {
          recovery_rhs[row] += block.b[
              static_cast<std::size_t>(row) *
                  coupled_fixture_local_trace_unknowns + column] *
              local_trace[column];
        }
      std::vector<double> recovered;
      if (!SolveChecked(factors[static_cast<std::size_t>(element)], block.a,
                        recovery_rhs, recovered, maximum_local_residual,
                        diagnostic, "recovery", -1))
      {
        return false;
      }
      for (double& value : recovered) { value = -value; }
      std::copy(recovered.begin(), recovered.end(),
                recovered_interior.begin() +
                    static_cast<std::size_t>(element) * recovered.size());
      const auto scatter_primal = [element](
                                      const mfem::ParFiniteElementSpace& space,
                                      const double* source,
                                      const int source_size,
                                      mfem::ParGridFunction& destination)
      {
        mfem::Array<int> vdofs;
        mfem::DofTransformation transformation(space.GetVDim(),
                                                space.GetOrdering());
        space.GetElementVDofs(element, vdofs, transformation);
        if (vdofs.Size() != source_size) { return false; }
        mfem::Vector local(source_size);
        std::copy(source, source + source_size, local.HostWrite());
        transformation.TransformPrimal(local);
        destination.SetSubVector(vdofs, local);
        return true;
      };
      if (!scatter_primal(*implementation_->u_space, recovered.data(), 16,
                          recovered_u) ||
          !scatter_primal(*implementation_->sigma_space,
                          recovered.data() + 16, 4, recovered_sigma) ||
          !scatter_primal(*implementation_->z_space,
                          recovered.data() + 20, 8, recovered_z))
      {
        return false;
      }
    }
    diagnostic.Set("restriction-recovery", "primal-verification");
    for (int element = 0; element < coupled_fixture_elements; ++element)
    {
      const auto verify_primal = [element](
                                     const mfem::ParFiniteElementSpace& space,
                                     const mfem::ParGridFunction& field,
                                     const double* expected,
                                     const int expected_size)
      {
        mfem::Array<int> vdofs;
        mfem::DofTransformation transformation(space.GetVDim(),
                                                space.GetOrdering());
        space.GetElementVDofs(element, vdofs, transformation);
        if (vdofs.Size() != expected_size) { return false; }
        mfem::Vector local(expected_size);
        field.GetSubVector(vdofs, local);
        transformation.InvTransformPrimal(local);
        for (int entry = 0; entry < expected_size; ++entry)
        {
          if (local[entry] != expected[entry]) { return false; }
        }
        return true;
      };
      const double* expected = recovered_interior.data() +
          static_cast<std::size_t>(element) *
              coupled_fixture_local_interior_unknowns;
      if (!verify_primal(*implementation_->u_space, recovered_u, expected, 16) ||
          !verify_primal(*implementation_->sigma_space, recovered_sigma,
                         expected + 16, 4) ||
          !verify_primal(*implementation_->z_space, recovered_z,
                         expected + 20, 8))
      {
        return false;
      }
    }
    diagnostic.Set("evidence-finalization", "snapshot");
    std::vector<double> full_increment(coupled_fixture_full_unknowns, 0.0);
    std::copy(recovered_interior.begin(), recovered_interior.end(),
              full_increment.begin());
    std::copy(true_increment.begin(), true_increment.end(),
              full_increment.begin() + recovered_interior.size());

    output.quadrature_order = coupled_fixture_quadrature_order;
    output.repeat_quadrature_order =
        coupled_fixture_repeat_quadrature_order;
    output.local_interior_unknowns =
        coupled_fixture_local_interior_unknowns;
    output.local_trace_unknowns = coupled_fixture_local_trace_unknowns;
    output.true_trace_unknowns = coupled_fixture_true_trace_unknowns;
    output.full_unknowns = coupled_fixture_full_unknowns;
    output.old_interior = std::move(state.old_interior);
    output.old_true_trace = std::move(state.old_true_trace);
    output.candidate_interior = std::move(state.candidate_interior);
    output.candidate_true_trace = std::move(state.candidate_true_trace);
    output.full_residual = std::move(primary.full_residual);
    output.full_jacobian = std::move(primary.full_jacobian);
    output.elements = std::move(primary.elements);
    output.condensed = std::move(condensed);
    output.true_trace_increment = std::move(true_increment);
    output.element_trace_increment = std::move(element_trace_increment);
    output.recovered_interior_increment = std::move(recovered_interior);
    output.full_increment = std::move(full_increment);
    output.maximum_local_application_residual = maximum_local_residual;
    output.residual_quadrature_repeat_relative_l2 =
        RelativeL2Difference(output.full_residual, repeat.full_residual);
    output.jacobian_quadrature_repeat_relative_l2 =
        RelativeL2Difference(output.full_jacobian, repeat.full_jacobian);
    output.solver = {50, 200, 1.0e-12, 1.0e-14, 0, 0, 1,
                     setup_status, solve_status, convergence_query_status,
                     iterations, converged, residual_norm, rhs_norm};
    const bool valid = Finite(output.full_residual) &&
        Finite(output.full_jacobian) &&
        Finite(output.true_trace_increment) &&
        Finite(output.recovered_interior_increment) &&
        output.residual_quadrature_repeat_relative_l2 <= 1.0e-11 &&
        output.jacobian_quadrature_repeat_relative_l2 <= 1.0e-11;
    if (valid)
    {
      diagnostic.CompleteAttempt();
    }
    else
    {
      diagnostic.Set("evidence-finalization", "validation");
      diagnostic.LatchFailure();
    }
    return valid;
  }
  catch (...)
  {
    diagnostic.Set("exception", "catch-all", diagnostic.element);
    output = {};
    return false;
  }
}

TrajectoryStatus
PersistentSc25ScalarIgrCpuFoundation::InitializePersistentTrajectory()
    noexcept
{
  if (!implementation_ || implementation_->status != FoundationStatus::ready ||
      implementation_->rank != 0 || implementation_->ranks != 1 ||
      implementation_->config.polynomial_degree != 1 ||
      implementation_->config.elements_per_direction !=
          persistent_elements_per_direction)
  {
    return TrajectoryStatus::invalid_configuration;
  }
  if (implementation_->trajectory &&
      implementation_->trajectory->initialized)
  {
    return TrajectoryStatus::ready;
  }
  try
  {
    auto trajectory = std::make_unique<PersistentTrajectoryImplementation>();
    if (!trajectory->InitializeStorage(*implementation_))
    {
      return TrajectoryStatus::initialization_failure;
    }
    TrajectorySlot initial;
    if (!trajectory->InitializePhysicalState(*implementation_, initial) ||
        !trajectory->BuildPersistentTraceGraph(*implementation_) ||
        !trajectory->InitializeAuxiliary(*implementation_, initial))
    {
      return TrajectoryStatus::initialization_failure;
    }
    StateHash mesh_hash{}, partition_hash{}, topology_hash{};
    if (!LiveResourceHashes(*implementation_, mesh_hash, partition_hash,
                            topology_hash))
    {
      return TrajectoryStatus::initialization_failure;
    }
    const std::size_t persistent_bytes = sizeof(TrajectorySlot) * 2 +
        sizeof(PersistentTrajectoryEvidence);
    const std::size_t workspace_bytes =
        static_cast<std::size_t>(persistent_elements) *
        (sizeof(RuntimeElementWorkspace) +
         (coupled_fixture_local_interior_unknowns *
              coupled_fixture_local_interior_unknowns +
          2 * coupled_fixture_local_interior_unknowns *
              coupled_fixture_local_trace_unknowns +
          2 * coupled_fixture_local_trace_unknowns *
              coupled_fixture_local_trace_unknowns) *
             sizeof(double));
    const PersistentResourceDescriptor resources{
        StableIdentity::Copy("mfem-d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec"),
        mesh_hash,
        partition_hash,
        topology_hash,
        1,
        1,
        StableIdentity::Copy(interior_layout_id),
        {true,
         true,
         true,
         false,
         1,
         1,
         persistent_bytes,
         workspace_bytes}};
    trajectory->contract = std::make_unique<TrajectoryContract>(
        initial, initial, StateCoordinates{}, resources);
    if (trajectory->contract->status() != RunStatus::ready)
    {
      return TrajectoryStatus::initialization_failure;
    }
    const TrajectoryAssemblyFailureCause initial_assembly_failure =
        trajectory->Assemble(*implementation_, initial, initial,
                             trajectory->assembly);
    if (initial_assembly_failure != TrajectoryAssemblyFailureCause::none)
    {
      trajectory->RecordAssemblyFailure(initial_assembly_failure);
      return TrajectoryStatus::initialization_failure;
    }
    double maximum_local_residual = 0.0;
    if (!trajectory->Condense(trajectory->assembly,
                              maximum_local_residual) ||
        !trajectory->AssembleTrueSystem(*implementation_))
    {
      return TrajectoryStatus::initialization_failure;
    }
    trajectory->evidence.reuse.baseline =
        trajectory->ReuseSnapshot(*implementation_);
    trajectory->evidence.reuse.current =
        trajectory->evidence.reuse.baseline;
    trajectory->reuse_baseline_frozen = true;
    if (!trajectory->VerifyReuse(*implementation_))
    {
      return TrajectoryStatus::initialization_failure;
    }
    trajectory->evidence.checkpoint_step_offset = 0;
    trajectory->evidence.continuation_start_coordinates = {};
    trajectory->evidence.continuation_start_state = ToPublicState(initial);
    trajectory->evidence.continuation_start_hashes = HashState(initial);
    trajectory->initialized = true;
    trajectory->status = TrajectoryStatus::ready;
    trajectory->RefreshEvidence();
    implementation_->trajectory = std::move(trajectory);
    return TrajectoryStatus::ready;
  }
  catch (...)
  {
    return TrajectoryStatus::initialization_failure;
  }
}

TrajectoryStatus
PersistentSc25ScalarIgrCpuFoundation::AdvancePersistentTrajectory(
    const TrajectoryFailureInjection injection) noexcept
{
  if (!implementation_ || !implementation_->trajectory)
  {
    return TrajectoryStatus::uninitialized;
  }
  return implementation_->trajectory->Advance(*implementation_, injection);
}

bool PersistentSc25ScalarIgrCpuFoundation::CopyPublishedTrajectoryState(
    PersistentTrajectoryState& output) const noexcept
{
  output = {};
  if (!implementation_ || !implementation_->trajectory ||
      !implementation_->trajectory->contract)
  {
    return false;
  }
  TrajectorySlot state;
  if (!implementation_->trajectory->contract->CopyPublished(
          implementation_->trajectory->contract->Published(), state))
  {
    return false;
  }
  output = ToPublicState(state);
  return true;
}

bool PersistentSc25ScalarIgrCpuFoundation::CaptureTrajectoryCheckpoint(
    PersistentTrajectoryCheckpoint& output) const noexcept
{
  output = {};
  output.projection = StableIdentity::Copy(projection_id);
  output.tuple = StableIdentity::Copy(cpu_rank1_tuple_id);
  output.scenario = StableIdentity::Copy(persistent_fixture_id);
  if (!implementation_ || !implementation_->trajectory ||
      !implementation_->trajectory->contract ||
      !implementation_->trajectory->contract->TryMakeCheckpoint(
          output.descriptor))
  {
    return false;
  }
  TrajectorySlot state;
  if (!implementation_->trajectory->contract->CopyPublished(
          implementation_->trajectory->contract->Published(), state))
  {
    output = {};
    return false;
  }
  output.state = ToPublicState(state);
  output.raw_field_hashes = HashState(state);
  output.fields = CheckpointFieldIdentities();
  output.canonical_manifest_hash =
      CanonicalCheckpointManifestHash(output);
  return true;
}

TrajectoryStatus
PersistentSc25ScalarIgrCpuFoundation::RestoreTrajectoryCheckpoint(
    const PersistentTrajectoryCheckpoint& checkpoint) noexcept
{
  if (!implementation_ || !implementation_->trajectory ||
      !implementation_->trajectory->contract)
  {
    return TrajectoryStatus::uninitialized;
  }
  PersistentTrajectoryImplementation& trajectory =
      *implementation_->trajectory;
  if (!trajectory.VerifyReuse(*implementation_))
  {
    trajectory.status = TrajectoryStatus::HdgSc25PublicationFailure;
    trajectory.RememberFailure(trajectory.status);
    trajectory.RefreshEvidence();
    return trajectory.status;
  }
  const PersistentResourceDescriptor& resources =
      trajectory.contract->resources();
  StateHash live_mesh_hash{}, live_partition_hash{}, live_topology_hash{};
  const bool live_resources = LiveResourceHashes(
      *implementation_, live_mesh_hash, live_partition_hash,
      live_topology_hash);
  const std::array<TrajectoryCheckpointFieldIdentity, 5> expected_fields =
      CheckpointFieldIdentities();
  const auto hashes_equal = [](const TrajectoryStateHashes& left,
                               const TrajectoryStateHashes& right)
  {
    return left.u == right.u && left.sigma == right.sigma &&
        left.z == right.z && left.uhat == right.uhat &&
        left.sigmahat == right.sigmahat &&
        left.interior == right.interior && left.trace == right.trace;
  };
  TrajectoryStatus failure = TrajectoryStatus::uninitialized;
  if (checkpoint.projection != StableIdentity::Copy(projection_id) ||
      checkpoint.tuple != StableIdentity::Copy(cpu_rank1_tuple_id) ||
      checkpoint.scenario != StableIdentity::Copy(persistent_fixture_id) ||
      checkpoint.descriptor.schema_version != checkpoint_schema_version ||
      !checkpoint.descriptor.successful_commit ||
      checkpoint.descriptor.restart_mode !=
          RestartMode::same_backend_rank_partition ||
      checkpoint.descriptor.tuple != CpuRank1Selection::Descriptor() ||
      !live_resources || !resources.Valid() ||
      checkpoint.descriptor.mfem_pin != resources.mfem_pin ||
      checkpoint.descriptor.mesh_hash != resources.mesh_hash ||
      checkpoint.descriptor.partition_hash != resources.partition_hash ||
      checkpoint.descriptor.topology_hash != resources.topology_hash ||
      resources.mesh_hash != live_mesh_hash ||
      resources.partition_hash != live_partition_hash ||
      resources.topology_hash != live_topology_hash ||
      checkpoint.descriptor.rank_count != resources.rank_count ||
      checkpoint.descriptor.polynomial_order != resources.polynomial_order ||
      checkpoint.descriptor.layout != resources.layout ||
      checkpoint.descriptor.precision != StableIdentity::Copy("Fp64") ||
      checkpoint.descriptor.cursor !=
          checkpoint.descriptor.coordinates.cursor ||
      detail::DoubleBits(checkpoint.descriptor.dt) !=
          detail::DoubleBits(persistent_dt) ||
      checkpoint.descriptor.coordinates.step == 0 ||
      checkpoint.descriptor.coordinates.step !=
          checkpoint.descriptor.coordinates.generation ||
      checkpoint.descriptor.coordinates.step !=
          checkpoint.descriptor.coordinates.cursor ||
      detail::DoubleBits(checkpoint.descriptor.coordinates.dt) !=
          detail::DoubleBits(persistent_dt) ||
      detail::DoubleBits(checkpoint.descriptor.coordinates.time) !=
          detail::DoubleBits(checkpoint.descriptor.coordinates.step *
                             persistent_dt) ||
      checkpoint.descriptor.coordinate_fingerprint !=
          detail::CoordinateFingerprint(checkpoint.descriptor.coordinates) ||
      !resources.reuse.persistent_topology ||
      !resources.reuse.persistent_complete_state_slots ||
      !resources.reuse.persistent_workspace ||
      resources.reuse.per_step_allocation_required ||
      resources.reuse.topology_generation != 1 ||
      resources.reuse.storage_generation != 1 ||
      resources.reuse.persistent_bytes == 0 ||
      resources.reuse.workspace_bytes == 0)
  {
    failure = TrajectoryStatus::HdgSc25RestartValidationFailureIdentity;
  }
  else if (!std::equal(
               checkpoint.fields.begin(), checkpoint.fields.end(),
               expected_fields.begin(),
               [](const TrajectoryCheckpointFieldIdentity& left,
                  const TrajectoryCheckpointFieldIdentity& right)
               { return SameFieldIdentity(left, right); }))
  {
    failure = TrajectoryStatus::HdgSc25RestartValidationFailureIdentity;
  }
  else if (CanonicalCheckpointManifestHash(checkpoint) !=
               checkpoint.canonical_manifest_hash)
  {
    failure = TrajectoryStatus::HdgSc25RestartValidationFailureHash;
  }

  const TrajectorySlot loaded = ToSlot(checkpoint.state);
  const TrajectoryStateHashes hashes = HashState(loaded);
  if (failure == TrajectoryStatus::uninitialized &&
      (!hashes_equal(hashes, checkpoint.raw_field_hashes) ||
           hashes.interior != checkpoint.descriptor.interior_hash ||
           hashes.trace != checkpoint.descriptor.trace_hash))
  {
    failure = TrajectoryStatus::HdgSc25RestartValidationFailureHash;
  }
  if (failure == TrajectoryStatus::uninitialized)
  {
    const TrajectoryAssemblyFailureCause assembly_failure =
        trajectory.Assemble(*implementation_, loaded, loaded,
                            trajectory.trial_assembly);
    if (assembly_failure != TrajectoryAssemblyFailureCause::none)
    {
      trajectory.RecordAssemblyFailure(assembly_failure);
      failure = assembly_failure ==
              TrajectoryAssemblyFailureCause::state_admissibility
          ? TrajectoryStatus::HdgSc25RestartValidationFailureAdmissibility
          : TrajectoryStatus::HdgSc25RestartValidationFailureAssembly;
    }
  }
  if (failure == TrajectoryStatus::uninitialized)
  {
    AuxiliaryResidualMetrics auxiliary_residual;
    if (!trajectory.AuxiliaryResidualGate(
            *implementation_, loaded, auxiliary_residual) ||
        auxiliary_residual.normalized_maximum > 1.0e-10)
    {
      failure =
          TrajectoryStatus::HdgSc25RestartValidationFailureAuxResidual;
    }
  }
  if (failure != TrajectoryStatus::uninitialized)
  {
    if (!trajectory.VerifyReuse(*implementation_))
    {
      failure = TrajectoryStatus::HdgSc25PublicationFailure;
    }
    trajectory.status = failure;
    trajectory.RememberFailure(failure);
    trajectory.RefreshEvidence();
    return failure;
  }
  const RestoreOutcome restored = trajectory.contract->Restore(
      checkpoint.descriptor, loaded);
  if (!restored.restored ||
      restored.compatibility != RestartCompatibility::compatible)
  {
    trajectory.status =
        TrajectoryStatus::HdgSc25RestartValidationFailureIdentity;
    trajectory.RememberFailure(trajectory.status);
    trajectory.RefreshEvidence();
    return trajectory.status;
  }
  TrajectorySlot published;
  const bool copied = trajectory.contract->CopyPublished(
      trajectory.contract->Published(), published);
  const StateCoordinates restored_coordinates =
      trajectory.contract->coordinates();
  const TrajectoryStateHashes restored_hashes = copied ?
      HashState(published) : TrajectoryStateHashes{};
  const bool loaded_state_match = copied && SameTrajectoryStateBytes(
      ToPublicState(published), checkpoint.state);
  const bool loaded_coordinates_match = detail::SameCoordinates(
      restored_coordinates, checkpoint.descriptor.coordinates);
  const bool loaded_hashes_match = copied &&
      hashes_equal(restored_hashes, checkpoint.raw_field_hashes);
  if (!loaded_state_match || !loaded_coordinates_match ||
      !loaded_hashes_match)
  {
    trajectory.status = TrajectoryStatus::HdgSc25PublicationFailure;
    trajectory.RememberFailure(trajectory.status);
    trajectory.RefreshEvidence();
    return trajectory.status;
  }
  trajectory.status = TrajectoryStatus::HdgSc25Accepted;
  trajectory.evidence.attempted_steps = 0;
  trajectory.evidence.retry_attempts = 0;
  trajectory.evidence.attempt_count = 0;
  trajectory.evidence.step_evidence_count = 0;
  trajectory.evidence.attempts = {};
  trajectory.evidence.steps = {};
  trajectory.evidence.accepted_steps =
      static_cast<int>(checkpoint.descriptor.coordinates.step);
  trajectory.evidence.checkpoint_step_offset =
      checkpoint.descriptor.coordinates.step;
  trajectory.evidence.continuation_start_coordinates = restored_coordinates;
  trajectory.evidence.continuation_start_state = checkpoint.state;
  trajectory.evidence.continuation_start_hashes = restored_hashes;
  trajectory.evidence.restored_loaded_state_match = true;
  trajectory.evidence.restored_loaded_coordinates_match = true;
  trajectory.evidence.restored_loaded_hashes_match = true;
  if (!trajectory.VerifyReuse(*implementation_))
  {
    trajectory.status = TrajectoryStatus::HdgSc25PublicationFailure;
    trajectory.RememberFailure(trajectory.status);
    trajectory.RefreshEvidence();
    return trajectory.status;
  }
  trajectory.RefreshEvidence();
  return trajectory.status;
}

const PersistentTrajectoryEvidence&
PersistentSc25ScalarIgrCpuFoundation::trajectory_evidence() const noexcept
{
  static const PersistentTrajectoryEvidence empty{};
  return implementation_ && implementation_->trajectory
      ? implementation_->trajectory->evidence
      : empty;
}

} // namespace dg::hdg::runtime::cpu::sc25
