#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_device_krylov.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
class Stencil final : public mfem::Operator
{
public:
  Stencil(int n, double off_left, double off_right, double diagonal = 4.0)
    : mfem::Operator(n), left_(off_left), right_(off_right), diagonal_(diagonal) { }
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    y.SetSize(height);
    y.UseDevice(true);
    const double* input = x.Read();
    double* output = y.Write();
    const int n = height;
    const double left = left_, right = right_, diagonal = diagonal_;
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int i)
    {
      output[i] = diagonal * input[i] + (i > 0 ? left * input[i - 1] : 0.0)
                    + (i + 1 < n ? right * input[i + 1] : 0.0);
    });
  }
private:
  double left_, right_, diagonal_;
};

// A varying, positive diagonal action exercises flexible right preconditioning.
class ChangingDiagonal final : public mfem::Solver
{
public:
  explicit ChangingDiagonal(int n) : mfem::Solver(n) { }
  void SetOperator(const mfem::Operator&) override { }
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    y.SetSize(height);
    y.UseDevice(true);
    const double* input = x.Read();
    double* output = y.Write();
    const int phase = calls_++ % 3;
    mfem::forall(height, [=] MFEM_HOST_DEVICE(int i)
    { output[i] = input[i] / (4.0 + 0.1 * ((i + phase) % 5)); });
  }
private:
  mutable int calls_ = 0;
};

double Norm(const mfem::Vector& x)
{
  const double* data = x.HostRead();
  long double result = 0.0L;
  for (int i = 0; i < x.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(data[i]), "Nonfinite device Krylov test vector.");
    result += static_cast<long double>(data[i]) * data[i];
  }
  return std::sqrt(static_cast<double>(result));
}

double Residual(const mfem::Operator& op, const mfem::Vector& b, const mfem::Vector& x)
{
  mfem::Vector r(b.Size());
  op.Mult(x, r);
  r -= b;
  return Norm(r);
}

void Configure(mfem::IterativeSolver& solver, const mfem::Operator& op,
                 bool warm = true, int max_iterations = 200)
{
  solver.SetOperator(op);
  solver.SetRelTol(1.0e-11);
  solver.SetAbsTol(0.0);
  solver.SetMaxIter(max_iterations);
  solver.SetPrintLevel(0);
  solver.iterative_mode = warm;
}

void CheckSolution(const mfem::IterativeSolver& solver, const mfem::Operator& op,
                     const mfem::Vector& b, const mfem::Vector& x,
                     const mfem::Vector& exact, double initial_norm)
{
  const double residual = Residual(op, b, x);
  mfem::Vector error(x);
  error -= exact;
  MFEM_VERIFY(solver.GetConverged() && residual <= 1.0e-11 * initial_norm &&
                Norm(error) / Norm(exact) < 2.0e-11,
              "Device Krylov failed its independent original-operator residual/solution gate.");
  MFEM_VERIFY(std::abs(solver.GetFinalNorm() - residual) <= 1.0e-13 * initial_norm,
              "Device Krylov final norm is not its independently recomputed residual.");
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  constexpr int n = 513; // non-multiple of a block and more than one block
  Stencil nonsymmetric(n, -0.8, 0.3), spd(n, -1.0, -1.0),
          identity(n, 0.0, 0.0, 1.0), negative(n, 0.0, 0.0, -1.0);
  mfem::Vector exact(n), guess(n), rhs(n), x(n), reference(n);
  for (int i = 0; i < n; ++i)
  { exact(i) = 0.5 + std::cos(0.27 * i); guess(i) = 0.2 * std::sin(0.53 * i); }
  auto gmres = ::igrlab::igr_internal::MakeDeviceFGMRESSolver(MPI_COMM_WORLD, 3);
  ChangingDiagonal changing(n);
  nonsymmetric.Mult(exact, rhs);
  Configure(*gmres, nonsymmetric);
  gmres->SetPreconditioner(changing);
  x = guess;
  const double initial = Residual(nonsymmetric, rhs, x);
  gmres->Mult(rhs, x);
  CheckSolution(*gmres, nonsymmetric, rhs, x, exact, initial);
  MFEM_VERIFY(gmres->GetNumIterations() > 3, "Restarted FGMRES test did not restart.");

  mfem::FGMRESSolver mfem_gmres(MPI_COMM_WORLD);
  ChangingDiagonal reference_changing(n);
  Configure(mfem_gmres, nonsymmetric);
  mfem_gmres.SetKDim(3);
  mfem_gmres.SetPreconditioner(reference_changing);
  reference = guess;
  mfem_gmres.Mult(rhs, reference);
  reference -= x;
  MFEM_VERIFY(mfem_gmres.GetConverged() && Norm(reference) / Norm(exact) < 3.0e-11,
              "Device FGMRES disagrees with MFEM on a flexible restarted solve.");

  // Reconfigure the retained workspace, removing the previous preconditioner.
  // Identity produces happy Arnoldi breakdown; no 0/0 normalization is allowed.
  Configure(*gmres, identity, false);
  rhs = 0.0;
  rhs.HostReadWrite()[n - 1] = 1.0;
  gmres->Mult(rhs, x);
  MFEM_VERIFY(gmres->GetConverged() && gmres->GetNumIterations() == 1 &&
                Residual(identity, rhs, x) == 0.0,
              "Device FGMRES happy breakdown failed.");
  rhs = 0.0;
  gmres->Mult(rhs, x);
  MFEM_VERIFY(gmres->GetConverged() && gmres->GetNumIterations() == 0 && Norm(x) == 0.0,
              "Device FGMRES zero RHS failed.");
  nonsymmetric.Mult(exact, rhs);
  Configure(*gmres, nonsymmetric, false, 1);
  gmres->Mult(rhs, x);
  MFEM_VERIFY(!gmres->GetConverged() && gmres->GetNumIterations() == 1,
              "Device FGMRES ignored its iteration budget.");

  auto cg = ::igrlab::igr_internal::MakeDeviceCGSolver(MPI_COMM_WORLD);
  spd.Mult(exact, rhs);
  Configure(*cg, spd);
  x = guess;
  const double cg_initial = Residual(spd, rhs, x);
  cg->Mult(rhs, x);
  CheckSolution(*cg, spd, rhs, x, exact, cg_initial);
  mfem::CGSolver mfem_cg(MPI_COMM_WORLD);
  Configure(mfem_cg, spd);
  reference = guess;
  mfem_cg.Mult(rhs, reference);
  reference -= x;
  MFEM_VERIFY(mfem_cg.GetConverged() && Norm(reference) / Norm(exact) < 3.0e-11,
              "Device CG disagrees with MFEM.");
  Configure(*cg, spd, false);
  rhs = 0.0;
  cg->Mult(rhs, x);
  MFEM_VERIFY(cg->GetConverged() && cg->GetNumIterations() == 0 && Norm(x) == 0.0,
              "Device CG zero RHS failed.");
  Configure(*cg, negative, false);
  rhs = exact;
  cg->Mult(rhs, x);
  MFEM_VERIFY(!cg->GetConverged() && Norm(x) == 0.0,
              "Device CG accepted nonpositive curvature.");
  Stencil nonfinite(n, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN());
  Configure(*cg, nonfinite, false);
  cg->Mult(rhs, x);
  MFEM_VERIFY(!cg->GetConverged() && Norm(x) == 0.0,
              "Device CG accepted nonfinite curvature.");
  std::cout << "device_krylov_test passed restarted_flexible_mgs=true"
               " true_residual=true cached_reconfigure=true happy_breakdown=true"
               " zero_rhs=true iteration_cap=true cg_mfem_parity=true nonpositive_curvature=true nonfinite_curvature=true\n";
  return 0;
}
