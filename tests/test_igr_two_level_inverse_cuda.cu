#include "../src/physics/igr/IGR_two_level_inverse.hpp"
#include <algorithm>
#include <cmath>
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
  constexpr int fine_order = 3;
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

  for (const std::string mode : {"dense", "cg"})
  for (int coarse_order = 1; coarse_order <= fine_order; ++coarse_order)
  {
    auto preconditioner = dg_igr::igr_internal::MakeTwoLevelInversePreconditioner(
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
    std::cout << "two_level_test coarse=" << mode << " n=" << n
              << " fine_order=" << fine_order << " coarse_order=" << coarse_order
              << " true_residual=" << residual << " solution_error=" << error
              << " iterations=" << solver.GetNumIterations() << '\n';
  }
  return 0;
}
