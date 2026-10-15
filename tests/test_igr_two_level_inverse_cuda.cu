#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_two_level_inverse.hpp"
#include "../src/physics/igr/IGR_two_level_blocks_cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
double RelativeDifference(const mfem::Vector& a, const mfem::Vector& b)
{
  const double* av = a.HostRead();
  const double* bv = b.HostRead();
  long double error = 0.0L, scale = 0.0L;
  for (int i = 0; i < a.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(av[i]) && std::isfinite(bv[i]), "Nonfinite test result.");
    const long double delta = static_cast<long double>(av[i]) - bv[i];
    error += delta * delta;
    scale += static_cast<long double>(bv[i]) * bv[i];
  }
  return std::sqrt(static_cast<double>(error / scale));
}

void CheckMixedBlockApplication()
{
  // Uniform batches exercise low/odd/high matrix extents without claiming hp.
  for (const int dof : {4, 9, 16, 25, 36, 64, 121})
  {
    constexpr int ne = 3;
    const int n = ne * dof;
    mfem::Array<int> map(n);
    mfem::Vector inverse(ne * dof * dof), x(n), coarse(n), fp64(n), fp32(n);
    for (int k = 0; k < n; ++k)
    {
      map[k] = n - 1 - k;
      x(k) = 0.7 + 0.2 * std::sin(0.37 * (k + 1));
      coarse(k) = 0.13 * std::cos(0.29 * k);
    }
    for (int e = 0; e < ne; ++e)
      for (int j = 0; j < dof; ++j)
        for (int i = 0; i < dof; ++i)
          inverse((e * dof + j) * dof + i) =
            (i == j ? 1.3 + 0.1 * e : 0.01 / (1 + std::abs(i - j)));
    x.UseDevice(true);
    coarse.UseDevice(true);
    igrlab::igr_internal::TwoLevelBlocks control, mixed;
    control.Configure(ne, dof, inverse, "fp64");
    mixed.Configure(ne, dof, inverse, "fp32");
    control.Mult(x, coarse, map, fp64);
    mixed.Mult(x, coarse, map, fp32);
    const double error = RelativeDifference(fp32, fp64);
    MFEM_VERIFY(error < 3.0e-6 && mixed.FallbackRows() == 0 &&
                  mixed.SetupFallbackBlocks() == 0,
                "IEEE FP32 block application mismatch: dof=" << dof
                << " relative_error=" << error);
    std::cout << "two_level_blocks_test dof=" << dof
              << " fp32_action_error=" << error << " fallback_rows=0\n";
  }
}

void CheckMixedBlockRecovery()
{
  constexpr int ne = 7, dof = 4, n = ne * dof;
  mfem::Array<int> map(n);
  mfem::Vector inverse(ne * dof * dof), x(n), coarse(n), reference(n), actual(n);
  inverse = 0.0;
  coarse = 0.0;
  // Ordinary inputs; unrepresentable inverse; extreme large/small RHS;
  // overflowing/underflowing dot products; and an exactly zero RHS.
  const double diagonal[ne] = {1.0, 1.0e80, 1.0, 1.0, 1.0e30, 1.0e-20, 1.0};
  const double rhs[ne] = {1.0, 1.0e-80, 1.0e80, 1.0e-80, 1.0e30, 1.0e-30, 0.0};
  for (int e = 0; e < ne; ++e)
    for (int i = 0; i < dof; ++i)
    {
      map[e * dof + i] = n - 1 - (e * dof + i);
      inverse((e * dof + i) * dof + i) = diagonal[e];
      x(map[e * dof + i]) = rhs[e];
    }
  x.UseDevice(true);
  coarse.UseDevice(true);
  igrlab::igr_internal::TwoLevelBlocks control, mixed;
  control.Configure(ne, dof, inverse, "fp64");
  mixed.Configure(ne, dof, inverse, "fp32");
  control.Mult(x, coarse, map, reference);
  mixed.Mult(x, coarse, map, actual);
  const double* expected = reference.HostRead();
  const double* measured = actual.HostRead();
  for (int k = 0; k < n; ++k)
  {
    MFEM_VERIFY(std::isfinite(measured[k]) &&
                  std::abs(measured[k] - expected[k]) <=
                    1.0e-13 * std::max(std::abs(expected[k]), 1.0e-300),
                "FP32 block recovery lost FP64 action at row " << k);
  }
  MFEM_VERIFY(mixed.SetupFallbackBlocks() == 1 && mixed.FallbackRows() == 5 * dof,
              "FP32 block recovery accounting mismatch.");
  mixed.Mult(x, coarse, map, actual);
  MFEM_VERIFY(mixed.FallbackRows() == 10 * dof,
              "FP32 block recovery count did not accumulate across applications.");
  std::cout << "two_level_blocks_recovery setup_fallback_blocks=1"
            << " applications=2 fallback_rows=" << mixed.FallbackRows() << '\n';
}

class ScopedBlockPrecision
{
public:
  explicit ScopedBlockPrecision(const std::string& value)
  {
    const char* previous = std::getenv(name_);
    had_previous_ = previous != nullptr;
    if (previous) { previous_ = previous; }
    MFEM_VERIFY(setenv(name_, value.c_str(), 1) == 0, "Could not select test precision.");
  }
  ~ScopedBlockPrecision()
  {
    if (had_previous_) { setenv(name_, previous_.c_str(), 1); }
    else { unsetenv(name_); }
  }
private:
  const char* name_ = "IGRLAB_IGR_TWO_LEVEL_BLOCK_PRECISION";
  bool had_previous_ = false;
  std::string previous_;
};

void Assemble(mfem::BilinearForm& form, mfem::Coefficient& reaction,
              mfem::Coefficient& diffusion)
{
  form.AddDomainIntegrator(new mfem::MassIntegrator(reaction));
  form.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion));
  form.AddInteriorFaceIntegrator(new mfem::DGDiffusionIntegrator(diffusion, -1.0, 36.0));
  form.Assemble();
  form.Finalize();
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    3, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  MFEM_VERIFY(argc <= 2, "Usage: test_igr_two_level_inverse_cuda [fine_order]");
  const int fine_order = argc == 2 ? std::stoi(argv[1]) : 3;
  MFEM_VERIFY(fine_order >= 1 && fine_order <= 10, "Fixture fine_order must be 1..10.");
  CheckMixedBlockApplication();
  CheckMixedBlockRecovery();
  mfem::L2_FECollection collection(fine_order, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace sfes(&pmesh, &collection);
  mfem::FunctionCoefficient reaction([](const mfem::Vector& x)
  { return 1.0 + 0.2 * x(0); });
  mfem::FunctionCoefficient diffusion([](const mfem::Vector& x)
  { return 0.3 + 0.1 * x(1); });
  mfem::FunctionCoefficient changed_reaction([](const mfem::Vector& x)
  { return 1.2 + 0.1 * x(1); });
  mfem::FunctionCoefficient changed_diffusion([](const mfem::Vector& x)
  { return 0.4 + 0.15 * x(0); });
  mfem::BilinearForm initial(&sfes), current(&sfes);
  Assemble(initial, reaction, diffusion);
  Assemble(current, changed_reaction, changed_diffusion);
  const int n = sfes.GetVSize();
  mfem::Vector exact(n), rhs(n), original_rhs(n), dense_reference(n);
  for (int i = 0; i < n; ++i)
  { exact(i) = 1.0 + std::sin(0.17 * (i + 1)) + 0.2 * std::cos(0.71 * i); }
  current.SpMat().Mult(exact, rhs);
  initial.SpMat().Mult(exact, original_rhs);
  mfem::DenseMatrix dense;
  current.SpMat().ToDenseMatrix(dense);
  dense.Invert();
  (void)rhs.HostRead();
  dense.Mult(rhs, dense_reference);
  MFEM_VERIFY(RelativeDifference(dense_reference, exact) < 1.0e-10,
              "Global FP64 reference disagrees with manufactured solution.");

  for (const std::string precision : {"fp64", "fp32"})
  for (const std::string mode : {"dense", "cg"})
  for (int coarse_order = 1; coarse_order <= std::min(fine_order, 3); ++coarse_order)
  {
    ScopedBlockPrecision selected_precision(precision);
    auto preconditioner = ::igrlab::igr_internal::MakeTwoLevelInversePreconditioner(
      initial.SpMat(), sfes, mode, coarse_order);
    mfem::Vector before(n), after(n), solution(n), action(n);
    preconditioner->Mult(original_rhs, before);
    preconditioner->SetOperator(current.SpMat());
    preconditioner->Mult(original_rhs, after);
    MFEM_VERIFY(RelativeDifference(after, before) < 1.0e-13,
                "SetOperator rebuilt the retained two-level preconditioner.");
    mfem::FGMRESSolver solver(MPI_COMM_WORLD);
    solver.SetOperator(current.SpMat());
    solver.SetPreconditioner(*preconditioner);
    solver.SetRelTol(1.0e-12);
    solver.SetAbsTol(0.0);
    solver.SetMaxIter(200);
    solver.SetKDim(60);
    solver.SetPrintLevel(0);
    solver.iterative_mode = false;
    solution.UseDevice(true);
    solver.Mult(rhs, solution);
    current.SpMat().Mult(solution, action);
    const double residual = RelativeDifference(action, rhs);
    const double error = RelativeDifference(solution, dense_reference);
    MFEM_VERIFY(solver.GetConverged() && residual < 1.0e-10 && error < 1.0e-9,
                "Two-level current-equation correction failed: " << mode
                << " coarse_order=" << coarse_order
                << " residual=" << residual << " solution_error=" << error);
    std::cout << "two_level_test coarse=" << mode << " block_precision=" << precision
              << " n=" << n
              << " fine_order=" << fine_order << " coarse_order=" << coarse_order
              << " true_residual=" << residual << " solution_error=" << error
              << " iterations=" << solver.GetNumIterations() << '\n';
  }
  return 0;
}
