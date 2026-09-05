#include "../src/physics/igr/IGR_screened_patch_inverse.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
double Relative(const mfem::Vector& a, const mfem::Vector& b)
{
  const double* av = a.HostRead();
  const double* bv = b.HostRead();
  long double error = 0.0L, scale = 0.0L;
  for (int i = 0; i < a.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(av[i]) && std::isfinite(bv[i]), "Nonfinite test action.");
    const long double d = static_cast<long double>(av[i]) - bv[i];
    error += d * d; scale += static_cast<long double>(bv[i]) * bv[i];
  }
  return std::sqrt(static_cast<double>(error / std::max(scale, 1.0e-300L)));
}

void Assemble(mfem::BilinearForm& form, mfem::Coefficient& reaction,
              mfem::Coefficient& diffusion, mfem::VectorCoefficient& convection)
{
  form.AddDomainIntegrator(new mfem::MassIntegrator(reaction));
  form.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion));
  form.AddDomainIntegrator(new mfem::ConvectionIntegrator(convection));
  form.AddInteriorFaceIntegrator(new mfem::DGDiffusionIntegrator(diffusion, -1.0, 64.0));
  form.Assemble(); form.Finalize();
}

// Independent oracle: solve each untransposed principal problem for this RHS,
// then restrict its solution. The CUDA factory instead constructs inverse rows.
void Oracle(const mfem::DenseMatrix& a, mfem::ParFiniteElementSpace& fes,
            int radius, const mfem::Vector& x, mfem::Vector& y)
{
  const double* xv = x.HostRead();
  const int ne = fes.GetNE();
  y.SetSize(a.Height()); y = 0.0;
  std::vector<std::vector<int>> neighbors(ne);
  for (int f = 0; f < fes.GetMesh()->GetNumFaces(); ++f)
  {
    int left, right; fes.GetMesh()->GetFaceElements(f, &left, &right);
    if (left >= 0 && right >= 0 && left != right)
    { neighbors[left].push_back(right); neighbors[right].push_back(left); }
  }
  for (int center = 0; center < ne; ++center)
  {
    std::vector<int> distance(ne, ne); distance[center] = 0;
    for (int hop = 0; hop < radius; ++hop)
      for (int e = 0; e < ne; ++e) if (distance[e] == hop)
        for (int j : neighbors[e]) distance[j] = std::min(distance[j], hop + 1);
    std::vector<int> ids, core;
    mfem::Array<int> dofs;
    for (int e = 0; e < ne; ++e) if (distance[e] <= radius)
    {
      fes.GetElementDofs(e, dofs);
      for (int j : dofs)
      {
        if (e == center) core.push_back(static_cast<int>(ids.size()));
        ids.push_back(j);
      }
    }
    const int np = static_cast<int>(ids.size());
    mfem::DenseMatrix local(np);
    mfem::Vector rhs(np), solution(np);
    for (int i = 0; i < np; ++i)
    {
      rhs(i) = xv[ids[i]];
      for (int j = 0; j < np; ++j) local(i, j) = a(ids[i], ids[j]);
    }
    mfem::DenseMatrixInverse inverse(local);
    inverse.Mult(rhs, solution);
    for (int i : core) y(ids[i]) = solution(i);
  }
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    3, 2, mfem::Element::QUADRILATERAL, true, 1.0, 0.7);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::L2_FECollection collection(3, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &collection);
  mfem::FunctionCoefficient reaction([](const mfem::Vector& x) { return 1.0 + 0.2 * x(0); });
  mfem::FunctionCoefficient diffusion([](const mfem::Vector& x) { return 0.3 + 0.1 * x(1); });
  mfem::FunctionCoefficient changed_reaction([](const mfem::Vector& x) { return 1.3 + 0.2 * x(1); });
  mfem::FunctionCoefficient changed_diffusion([](const mfem::Vector& x) { return 0.5 + 0.1 * x(0); });
  mfem::Vector velocity({0.17, -0.09});
  mfem::VectorConstantCoefficient convection(velocity);
  mfem::BilinearForm initial(&fes), current(&fes);
  Assemble(initial, reaction, diffusion, convection);
  Assemble(current, changed_reaction, changed_diffusion, convection);
  mfem::DenseMatrix dense, changed_dense;
  initial.SpMat().ToDenseMatrix(dense);
  current.SpMat().ToDenseMatrix(changed_dense);
  const int n = fes.GetVSize();
  mfem::Vector rhs(n), exact(n), current_rhs(n), reference(n);
  for (int i = 0; i < n; ++i)
  {
    rhs(i) = std::cos(0.31 * (i + 1)) + 0.2 * std::sin(0.53 * i);
    exact(i) = 1.0 + 0.3 * std::sin(0.19 * i);
  }
  changed_dense.Mult(exact, current_rhs);
  mfem::DenseMatrixInverse global_reference(changed_dense);
  global_reference.Mult(current_rhs, reference);
  MFEM_VERIFY(Relative(reference, exact) < 1.0e-10, "Manufactured FP64 reference failed.");
  for (const std::string precision : {"fp64", "fp32"})
  for (int radius : {0, 1, 2, 3})
  {
    auto preconditioner = dg_igr::igr_internal::MakeScreenedPatchInversePreconditioner(
      initial.SpMat(), fes, radius, precision);
    mfem::Vector expected(n), before(n), after(n), solution(n), action(n);
    Oracle(dense, fes, radius, rhs, expected);
    preconditioner->Mult(rhs, before);
    const double action_error = Relative(before, expected);
    MFEM_VERIFY(action_error < (precision == "fp64" ? 1.0e-10 : 1.0e-5),
                "CUDA patch action differs from independent local solves.");
    preconditioner->SetOperator(current.SpMat());
    preconditioner->Mult(rhs, after);
    MFEM_VERIFY(Relative(after, before) < 1.0e-14,
                "SetOperator changed the retained first-operator preconditioner.");
    mfem::FGMRESSolver solver(MPI_COMM_WORLD);
    solver.SetOperator(current.SpMat()); solver.SetPreconditioner(*preconditioner);
    solver.SetRelTol(1.0e-12); solver.SetAbsTol(0.0);
    solver.SetMaxIter(300); solver.SetKDim(80); solver.SetPrintLevel(0);
    solver.iterative_mode = false;
    solution.UseDevice(true);
    solver.Mult(current_rhs, solution);
    current.SpMat().Mult(solution, action);
    const double residual = Relative(action, current_rhs);
    const double error = Relative(solution, reference);
    MFEM_VERIFY(solver.GetConverged() && residual < 1.0e-10 && error < 1.0e-9,
                "Retained patch correction failed current-operator accuracy.");
    std::cout << "screened_patch_test precision=" << precision << " radius=" << radius
              << " action_error=" << action_error << " residual=" << residual
              << " solution_error=" << error << " iterations=" << solver.GetNumIterations()
              << '\n';
  }
  return 0;
}
