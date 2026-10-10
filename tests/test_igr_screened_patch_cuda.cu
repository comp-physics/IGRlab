#include "../src/physics/igr/IGR_screened_patch_inverse.hpp"
#include "test_igr_face_condensed_oracle.hpp"
#include "test_igr_face_density_cuda.hpp"
#include "test_igr_compact_subcell_recovery.hpp"
#include <cuda_runtime.h>
#include <chrono>
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

// Independent representation oracle: form the full congruence using MFEM's
// LGL shape evaluation, solve each vertex principal RHS, then transform back.
// The implementation instead samples face blocks and stores composite GL rows.
void VertexOracle(const mfem::DenseMatrix& a, mfem::ParFiniteElementSpace& fes,
                  const mfem::Vector& x, mfem::Vector& y)
{
  const int n = fes.GetVSize();
  mfem::L2_FECollection corner_collection(1, 2, mfem::BasisType::GaussLobatto);
  const auto* corners = corner_collection.FiniteElementForGeometry(mfem::Geometry::SQUARE);
  mfem::DenseMatrix transform(n), product(n), congruent(n);
  transform = 0.0;
  std::vector<std::vector<int>> groups(fes.GetMesh()->GetNV());
  mfem::Array<int> dofs, vertices;
  mfem::Vector shape(4);
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementDofs(e, dofs); fes.GetMesh()->GetElementVertices(e, vertices);
    for (int j = 0; j < 4; ++j)
    {
      const auto& point = corners->GetNodes().IntPoint(j);
      const int corner = point.y < 0.5 ? (point.x < 0.5 ? 0 : 1)
                                         : (point.x < 0.5 ? 3 : 2);
      groups[vertices[corner]].push_back(4 * e + j);
    }
    for (int i = 0; i < 4; ++i)
    {
      corners->CalcShape(fes.GetFE(e)->GetNodes().IntPoint(i), shape);
      for (int j = 0; j < 4; ++j) transform(dofs[i], 4 * e + j) = shape(j);
    }
  }
  mfem::Mult(a, transform, product); mfem::MultAtB(transform, product, congruent);
  mfem::Vector transformed_rhs(n), transformed_solution(n);
  const double* input = x.HostRead(); transformed_rhs = 0.0; transformed_solution = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) transformed_rhs(j) += transform(i, j) * input[i];
  for (const auto& group : groups)
  {
    const int count = static_cast<int>(group.size());
    if (count == 0) continue;
    mfem::DenseMatrix principal(count); mfem::Vector rhs(count), solution(count);
    for (int i = 0; i < count; ++i)
    {
      rhs(i) = transformed_rhs(group[i]);
      for (int j = 0; j < count; ++j) principal(i, j) = congruent(group[i], group[j]);
    }
    mfem::DenseMatrixInverse inverse(principal); inverse.Mult(rhs, solution);
    for (int i = 0; i < count; ++i) transformed_solution(group[i]) = solution(i);
  }
  y.SetSize(n); transform.Mult(transformed_solution, y);
}

void TestVertex(mfem::ParMesh& mesh, mfem::Coefficient& reaction,
                mfem::Coefficient& diffusion, mfem::Coefficient& changed_reaction,
                mfem::Coefficient& changed_diffusion)
{
  mfem::L2_FECollection collection(1, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &collection);
  mfem::Vector zero({0.0, 0.0}); mfem::VectorConstantCoefficient convection(zero);
  mfem::BilinearForm initial(&fes), current(&fes);
  Assemble(initial, reaction, diffusion, convection);
  Assemble(current, changed_reaction, changed_diffusion, convection);
  mfem::DenseMatrix dense, changed_dense;
  initial.SpMat().ToDenseMatrix(dense); current.SpMat().ToDenseMatrix(changed_dense);
  const int n = fes.GetVSize();
  mfem::Vector rhs(n), exact(n), current_rhs(n), reference(n), expected(n), before(n), after(n);
  for (int i = 0; i < n; ++i)
  { rhs(i) = std::sin(0.31 * (i + 1)) + 0.17 * std::cos(0.43 * i);
    exact(i) = 1.0 + 0.21 * std::cos(0.27 * i); }
  changed_dense.Mult(exact, current_rhs);
  mfem::DenseMatrixInverse inverse(changed_dense); inverse.Mult(current_rhs, reference);
  auto preconditioner = dg_igr::igr_internal::MakeScreenedPatchInversePreconditioner(
    initial.SpMat(), fes, 0, "fp64", "vertex-p1");
  VertexOracle(dense, fes, rhs, expected); preconditioner->Mult(rhs, before);
  MFEM_VERIFY(Relative(before, expected) < 1.0e-11, "Vertex composite action differs from independent congruent solve.");
  // Assemble only this tiny test preconditioner to verify symmetry and SPD.
  mfem::DenseMatrix assembled(n); mfem::Vector unit(n), column(n), oracle_column(n);
  for (int j = 0; j < n; ++j)
  {
    // The previous application has made a device copy valid. Mark the whole
    // host write explicitly before constructing the next basis vector.
    double* input = unit.HostWrite(); std::fill(input, input + n, 0.0); input[j] = 1.0;
    preconditioner->Mult(unit, column);
    VertexOracle(dense, fes, unit, oracle_column);
    MFEM_VERIFY(Relative(column, oracle_column) < 1.0e-11,
                "Vertex basis-column action differs from independent congruent solve: column=" << j);
    const double* values = column.HostRead();
    for (int i = 0; i < n; ++i) assembled(i, j) = values[i];
  }
  double asymmetry = 0.0, scale = 0.0;
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
  { asymmetry = std::max(asymmetry, std::abs(assembled(i, j) - assembled(j, i)));
    scale = std::max(scale, std::abs(assembled(i, j))); }
  MFEM_VERIFY(asymmetry < 1.0e-12 * scale,
              "Vertex application is not symmetric: relative=" << asymmetry / scale
              << " absolute=" << asymmetry << " scale=" << scale);
  mfem::DenseMatrixInverse positive_check(assembled, true);
  preconditioner->SetOperator(current.SpMat()); preconditioner->Mult(rhs, after);
  MFEM_VERIFY(Relative(after, before) < 1.0e-14, "Vertex factors changed unexpectedly with current A.");
  for (bool cg : {false, true})
  {
    std::unique_ptr<mfem::IterativeSolver> solver;
    if (cg) solver = std::make_unique<mfem::CGSolver>(MPI_COMM_WORLD);
    else
    { auto fgmres = std::make_unique<mfem::FGMRESSolver>(MPI_COMM_WORLD);
      fgmres->SetKDim(80); solver = std::move(fgmres); }
    solver->SetOperator(current.SpMat()); solver->SetPreconditioner(*preconditioner);
    solver->SetRelTol(1.0e-12); solver->SetAbsTol(0.0);
    solver->SetMaxIter(300); solver->SetPrintLevel(0); solver->iterative_mode = false;
    mfem::Vector solution(n), action(n); solution.UseDevice(true);
    solver->Mult(current_rhs, solution); current.SpMat().Mult(solution, action);
    const double residual = Relative(action, current_rhs), error = Relative(solution, reference);
    MFEM_VERIFY(solver->GetConverged() && residual < 1.0e-10 && error < 1.0e-9,
                "Retained vertex correction failed with current operator.");
    std::cout << "screened_vertex_test solver=" << (cg ? "cg" : "fgmres")
              << " action_error=" << Relative(before, expected) << " symmetry=" << asymmetry / scale
              << " residual=" << residual << " solution_error=" << error
              << " iterations=" << solver->GetNumIterations() << '\n';
  }
}

void TestCondensed(mfem::ParMesh& mesh, mfem::Coefficient& reaction,
                   mfem::Coefficient& diffusion, mfem::Coefficient& changed_reaction,
                   mfem::Coefficient& changed_diffusion)
{
  for (const std::string kind : {"face-condensed", "trace-pairs"})
  for (int p : {1, 3, 5})
  {
    mfem::L2_FECollection collection(p, 2, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace fes(&mesh, &collection);
    mfem::Vector zero({0.0, 0.0}); mfem::VectorConstantCoefficient convection(zero);
    mfem::BilinearForm initial(&fes), current(&fes);
    Assemble(initial, reaction, diffusion, convection);
    Assemble(current, changed_reaction, changed_diffusion, convection);
    mfem::DenseMatrix dense, changed;
    initial.SpMat().ToDenseMatrix(dense); current.SpMat().ToDenseMatrix(changed);
    const int n = fes.GetVSize();
    mfem::Vector rhs(n), expected(n), exact(n), current_rhs(n);
    for (int i = 0; i < n; ++i)
    { rhs(i) = std::sin(0.37 * i) + 0.3 * std::cos(0.19 * i); exact(i) = 1.0 + 0.1 * std::cos(0.53 * i); }
    FaceCondensedOracle(dense, fes, rhs, expected, kind == "trace-pairs");
    changed.Mult(exact, current_rhs);
    for (const std::string precision : {"fp64", "fp32", "tf32"})
    {
      auto pc = dg_igr::igr_internal::MakeScreenedPatchInversePreconditioner(initial.SpMat(), fes, 0, precision, kind);
      mfem::Vector before(n), after(n), solution(n), action(n);
      pc->Mult(rhs, before);
      const double discrepancy = Relative(before, expected);
      MFEM_VERIFY(discrepancy < (precision == "fp64" ? 1.0e-10 : precision == "fp32" ? 1.0e-5 : 0.01),
                  "Condensed CUDA action differs from independent full algebra: " << discrepancy);
      pc->SetOperator(current.SpMat()); pc->Mult(rhs, after);
      MFEM_VERIFY(Relative(after, before) < 1.0e-14, "Condensed factors changed on SetOperator.");
      std::unique_ptr<mfem::IterativeSolver> solver;
      if (precision == "fp64") solver = std::make_unique<mfem::CGSolver>(MPI_COMM_WORLD);
      else { auto f = std::make_unique<mfem::FGMRESSolver>(MPI_COMM_WORLD); f->SetKDim(80); solver = std::move(f); }
      solver->SetOperator(current.SpMat()); solver->SetPreconditioner(*pc);
      solver->SetRelTol(1.0e-11); solver->SetAbsTol(0.0); solver->SetMaxIter(400); solver->SetPrintLevel(0);
      solver->iterative_mode = false; solution.UseDevice(true);
      solver->Mult(current_rhs, solution); current.SpMat().Mult(solution, action);
      MFEM_VERIFY(solver->GetConverged() && Relative(action, current_rhs) < 1.0e-9 && Relative(solution, exact) < 1.0e-8,
                  "Condensed preconditioner failed current equation correction.");
      std::cout << "screened_condensed_test kind=" << kind << " p=" << p << " precision=" << precision << " action_error=" << discrepancy
                << " residual=" << Relative(action, current_rhs) << " solution_error=" << Relative(solution, exact)
                << " iterations=" << solver->GetNumIterations() << '\n';
    }
  }
}

// Bounded mechanism benchmark; the application harness measures full horizons.
void Benchmark(int nx, int p, double alpha, int repeats, const std::string& precision,
               const std::string& kind, int radius)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(nx, nx, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::L2_FECollection collection(p, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &collection);
  mfem::FunctionCoefficient reaction([](const mfem::Vector& x) { return 1.0 / (1.0 + 0.5 * std::tanh(25.0 * (x(0) - 0.43))); });
  mfem::FunctionCoefficient diffusion([=](const mfem::Vector& x) { return alpha / (1.0 + 0.5 * std::tanh(25.0 * (x(0) - 0.43))); });
  mfem::Vector zero({0.0, 0.0}); mfem::VectorConstantCoefficient convection(zero);
  mfem::BilinearForm form(&fes); Assemble(form, reaction, diffusion, convection);
  const int n = fes.GetVSize(); mfem::Vector x(n), y(n), rhs(n);
  for (int i = 0; i < n; ++i) x(i) = 1.0 + 0.1 * std::sin(0.17 * i);
  x.UseDevice(true); y.UseDevice(true); rhs.UseDevice(true);
  form.SpMat().Mult(x, rhs);
  const auto start = std::chrono::steady_clock::now();
  auto pc = dg_igr::igr_internal::MakeScreenedPatchInversePreconditioner(form.SpMat(), fes, radius, precision, kind);
  cudaDeviceSynchronize();
  const double setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  for (int i = 0; i < 10; ++i) pc->Mult(rhs, y);
  cudaEvent_t begin, end; cudaEventCreate(&begin); cudaEventCreate(&end);
  cudaEventRecord(begin);
  for (int i = 0; i < repeats; ++i) pc->Mult(rhs, y);
  cudaEventRecord(end); cudaEventSynchronize(end);
  float ms = 0.0f; cudaEventElapsedTime(&ms, begin, end);
  cudaEventDestroy(begin); cudaEventDestroy(end);
  std::unique_ptr<mfem::IterativeSolver> solver;
  if ((kind == "face-condensed" || kind == "trace-pairs") && precision == "fp64") solver = std::make_unique<mfem::CGSolver>(MPI_COMM_WORLD);
  else { auto f = std::make_unique<mfem::FGMRESSolver>(MPI_COMM_WORLD); f->SetKDim(80); solver = std::move(f); }
  solver->SetOperator(form.SpMat()); solver->SetPreconditioner(*pc);
  solver->SetRelTol(1.0e-8); solver->SetAbsTol(0.0); solver->SetMaxIter(500); solver->SetPrintLevel(0); solver->iterative_mode = false;
  const auto solve_start = std::chrono::steady_clock::now(); solver->Mult(rhs, y); cudaDeviceSynchronize();
  const double solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
  mfem::Vector action(n); form.SpMat().Mult(y, action);
  std::cout << "screened_benchmark nx=" << nx << " p=" << p << " alpha=" << alpha << " precision=" << precision << " kind=" << kind
            << " n=" << n << " setup_seconds=" << setup << " apply_us=" << 1000.0 * ms / repeats << " solve_seconds=" << solve
            << " iterations=" << solver->GetNumIterations() << " converged=" << solver->GetConverged()
            << " residual=" << Relative(action, rhs) << " solution_error=" << Relative(y, x) << '\n';
  MFEM_VERIFY(solver->GetConverged() && Relative(action, rhs) < 1.0e-7, "Benchmark current residual failed.");
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  if (argc == 9 && std::string(argv[1]) == "--benchmark")
  {
    Benchmark(std::stoi(argv[2]), std::stoi(argv[3]), std::stod(argv[4]), std::stoi(argv[5]), argv[6], argv[7], std::stoi(argv[8]));
    return 0;
  }
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
  for (const std::string precision : {"fp64", "fp32", "tf32"})
  for (int radius : {0, 1, 2, 3})
  {
    auto preconditioner = dg_igr::igr_internal::MakeScreenedPatchInversePreconditioner(
      initial.SpMat(), fes, radius, precision);
    mfem::Vector expected(n), before(n), after(n), solution(n), action(n);
    Oracle(dense, fes, radius, rhs, expected);
    preconditioner->Mult(rhs, before);
    const double action_error = Relative(before, expected);
    MFEM_VERIFY(action_error < (precision == "fp64" ? 1.0e-10 : precision == "fp32" ? 1.0e-5 : 0.01),
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
  TestVertex(mesh, reaction, diffusion, changed_reaction, changed_diffusion);
  TestCondensed(mesh, reaction, diffusion, changed_reaction, changed_diffusion);
  TestFaceDensityCUDA(mesh);
  TestCompactSubcellRecovery(MPI_COMM_WORLD);
  return 0;
}
