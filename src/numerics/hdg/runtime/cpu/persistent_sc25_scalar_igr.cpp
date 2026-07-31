#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"

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
  bool success = false;

  void Set(const char* next_phase, const char* next_substage,
           const int next_element = -1) noexcept
  {
    phase = next_phase;
    substage = next_substage;
    element = next_element;
  }

  ~CoupledFixturePrivateDiagnostic() noexcept
  {
    if (success) { return; }
    std::fprintf(
        stderr,
        "C1-MFEM-PERIODIC-02 diagnostic phase=%s substage=%s element=%d "
        "hypre_setup_reached=%d setup_status=%d "
        "hypre_solve_reached=%d solve_status=%d "
        "hypre_query_reached=%d convergence_query_status=%d converged=%d "
        "iteration_query_reached=%d iterations=%d "
        "true_residual_reached=%d true_residual=%.17g rhs=%.17g "
        "bound=%.17g\n",
        phase, substage, element, hypre_setup_reached ? 1 : 0,
        setup_status, hypre_solve_reached ? 1 : 0, solve_status,
        hypre_query_reached ? 1 : 0, convergence_query_status, converged,
        iteration_query_reached ? 1 : 0, iterations,
        true_residual_reached ? 1 : 0, true_residual_norm,
        right_hand_side_norm, true_residual_bound);
    std::fprintf(
        stderr,
        "C1-MFEM-PERIODIC-02 local-detail expected_element=%d "
        "actual_element=%d ni=%d nt=%d A=%zu B=%zu C=%zu D=%zu "
        "r_local=%zu r_trace=%zu lu_failure_column=%d "
        "lu_pivot_magnitude=%.17g lu_pivot_threshold=%.17g "
        "lu_reason=%s solve_kind=%s trace_column=%d solve_reached=%d "
        "solve_succeeded=%d application_reached=%d "
        "application_residual=%.17g rhs_norm=%.17g bound=%.17g\n",
        expected_block_element, actual_block_element,
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
    CoupledFixtureAssembly& output, const char* phase,
    CoupledFixturePrivateDiagnostic& diagnostic)
{
  output = {};
  diagnostic.Set(phase, "precondition");
  try
  {
    constexpr int ns = coupled_fixture_scalar_volume_nodes;
    constexpr int nf = coupled_fixture_local_scalar_trace_nodes;
    constexpr int ni = coupled_fixture_local_interior_unknowns;
    constexpr int nt = coupled_fixture_local_trace_unknowns;
    constexpr int full = coupled_fixture_full_unknowns;
    if (mesh.GetNE() != coupled_fixture_elements ||
        mesh.GetNumFaces() != coupled_fixture_unique_faces ||
        volume_space.GetVDim() != 1 ||
        carrier_space.GetVDim() != trace_components ||
        state.old_by_element.size() != coupled_fixture_elements ||
        state.candidate_by_element.size() != coupled_fixture_elements ||
        state.old_trace_by_element.size() != coupled_fixture_elements ||
        state.trace_by_element.size() != coupled_fixture_elements ||
        topology.size() != coupled_fixture_elements)
    {
      return false;
    }
    output.elements.resize(coupled_fixture_elements);
    output.full_residual.assign(full, 0.0);
    output.full_jacobian.assign(static_cast<std::size_t>(full) * full, 0.0);
    const auto local_index = [](const int component, const int scalar)
    { return component * ns + scalar; };
    const auto trace_index = [](const int component, const int face_scalar)
    { return component * nf + face_scalar; };

    for (int element = 0; element < coupled_fixture_elements; ++element)
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
      block.a.assign(static_cast<std::size_t>(ni) * ni, 0.0);
      block.b.assign(static_cast<std::size_t>(ni) * nt, 0.0);
      block.c.assign(static_cast<std::size_t>(nt) * ni, 0.0);
      block.d.assign(static_cast<std::size_t>(nt) * nt, 0.0);
      block.local_residual.assign(ni, 0.0);
      block.trace_residual.assign(nt, 0.0);
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
      mfem::Vector shape(ns);
      mfem::DenseMatrix derivative(ns, dimension);
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
                     coupled_fixture_dt -
                 derivative(row_scalar, 0) * flux[0][equation] -
                 derivative(row_scalar, 1) * flux[1][equation]);
            for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
              for (int variable = 0; variable < trace_components; ++variable)
              {
                const int column = local_index(variable, column_scalar);
                double value = variable == equation ?
                    shape[row_scalar] * shape[column_scalar] /
                        coupled_fixture_dt : 0.0;
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
        mfem::Vector interior_shape(ns), trace_shape(2), oriented_shape(2);
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
          mfem::Vector scaled(dimension);
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
        return false;
      }
    }

    diagnostic.Set(phase, "global-scatter");
    for (int element = 0; element < coupled_fixture_elements; ++element)
    {
      const ElementLinearization& block =
          output.elements[static_cast<std::size_t>(element)];
      const ElementTopology& map = topology[static_cast<std::size_t>(element)];
      const int local_base = element * ni;
      for (int row = 0; row < ni; ++row)
      {
        output.full_residual[local_base + row] = block.local_residual[row];
        for (int column = 0; column < ni; ++column)
        {
          output.full_jacobian[
              static_cast<std::size_t>(local_base + row) * full +
              local_base + column] =
              block.a[static_cast<std::size_t>(row) * ni + column];
        }
        for (int column = 0; column < nt; ++column)
        {
          const TraceRestrictionEntry& entry = map.restriction[column];
          const int true_column =
              coupled_fixture_global_interior_unknowns +
              entry.local_true_dof;
          output.full_jacobian[
              static_cast<std::size_t>(local_base + row) * full +
              true_column] += entry.sign *
              block.b[static_cast<std::size_t>(row) * nt + column];
        }
      }
      for (int row = 0; row < nt; ++row)
      {
        const TraceRestrictionEntry& row_entry = map.restriction[row];
        const int true_row = coupled_fixture_global_interior_unknowns +
            row_entry.local_true_dof;
        output.full_residual[true_row] +=
            row_entry.sign * block.trace_residual[row];
        for (int column = 0; column < ni; ++column)
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
          const int true_column =
              coupled_fixture_global_interior_unknowns +
              column_entry.local_true_dof;
          output.full_jacobian[
              static_cast<std::size_t>(true_row) * full + true_column] +=
              row_entry.sign * column_entry.sign *
              block.d[static_cast<std::size_t>(row) * nt + column];
        }
      }
    }
    return Finite(output.full_residual) && Finite(output.full_jacobian);
  }
  catch (...)
  {
    diagnostic.Set(phase, "exception", diagnostic.element);
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

} // namespace

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
    if (!AssembleCoupledFixture(
            *implementation_->mesh, *implementation_->sigma_space,
            *implementation_->carrier_space,
            *implementation_->trace_collection, implementation_->topology,
            state, coupled_fixture_quadrature_order, primary,
            "primary-assembly", diagnostic))
    {
      return false;
    }
    if (!AssembleCoupledFixture(
            *implementation_->mesh, *implementation_->sigma_space,
            *implementation_->carrier_space,
            *implementation_->trace_collection, implementation_->topology,
            state, coupled_fixture_repeat_quadrature_order, repeat,
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
    diagnostic.success = valid;
    return valid;
  }
  catch (...)
  {
    diagnostic.Set("exception", "catch-all", diagnostic.element);
    output = {};
    return false;
  }
}

} // namespace dg::hdg::runtime::cpu::sc25
