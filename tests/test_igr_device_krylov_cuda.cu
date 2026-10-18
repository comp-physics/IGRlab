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

class PositiveDiagonal final : public mfem::Solver
{
public:
  explicit PositiveDiagonal(int n) : mfem::Solver(n) { }
  void SetOperator(const mfem::Operator&) override { }
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    y.SetSize(height);
    y.UseDevice(true);
    const double* input = x.Read();
    double* output = y.Write();
    mfem::forall(height, [=] MFEM_HOST_DEVICE(int i)
    { output[i] = input[i] / (4.0 + 0.2 * (i % 7)); });
  }
};

double PreconditionedResidual(const mfem::Operator& op, mfem::Solver& prec,
                                const mfem::Vector& b, const mfem::Vector& x)
{
  mfem::Vector residual(b.Size()), z(b.Size());
  op.Mult(x, residual);
  residual -= b;
  prec.Mult(residual, z);
  const double* r = residual.HostRead();
  const double* zr = z.HostRead();
  long double product = 0.0L;
  for (int i = 0; i < b.Size(); ++i) { product += static_cast<long double>(r[i]) * zr[i]; }
  MFEM_VERIFY(product >= 0.0L && std::isfinite(product), "Invalid test residual product.");
  return std::sqrt(static_cast<double>(product));
}

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
// Exercise the specialized coarse path with actual MFEM CSR and Jacobi rather
// than a synthetic operator wrapper. The largest CSR exercises many blocks
// and grid-stride work; n=1 and n=31 cover partial blocks and zero-work lanes.
bool CheckCooperativeCG()
{
  auto cooperative = ::igrlab::igr_internal::MakeCooperativeCGSolver(MPI_COMM_WORLD);
  const bool admitted = bool(cooperative);
  if (!cooperative) { cooperative = ::igrlab::igr_internal::MakeDeviceCGSolver(MPI_COMM_WORLD); }
  auto polled = ::igrlab::igr_internal::MakeDeviceCGSolver(MPI_COMM_WORLD);
  mfem::Array<int> essential;
  for (const int size : {1, 31, 513, 262147})
  {
    mfem::SparseMatrix a(size);
    for (int row = 0; row < size; ++row)
    {
      a.Add(row, row, 2.5 + 0.2 * (row % 7));
      if (row > 0) { a.Add(row, row - 1, -0.4); }
      if (row + 1 < size) { a.Add(row, row + 1, -0.4); }
    }
    a.Finalize();
    mfem::Vector diagonal(size), exact(size), b(size), x(size), other(size), zero(size);
    a.GetDiag(diagonal);
    essential.SetSize(size == 31 ? 1 : 0);
    if (size == 31) { essential[0] = 0; }
    mfem::OperatorJacobiSmoother jacobi(diagonal, essential, size == 31 ? 0.75 : 1.0);
    for (int i = 0; i < size; ++i) { exact(i) = 0.5 + std::cos(0.017 * i) + 0.1 * std::sin(0.51 * i); }
    zero = 0.0;
    Configure(*cooperative, a, false, 100);
    cooperative->SetPreconditioner(jacobi);
    Configure(*polled, a, false, 100);
    polled->SetPreconditioner(jacobi);
    const double tolerance = size == 513 ? 1.0e-11 : 1.0e-2;
    cooperative->SetRelTol(tolerance);
    polled->SetRelTol(tolerance);
    for (int solve = 0; solve < 2; ++solve)
    {
      if (solve == 1)
      {
        // The retained solver must use current CSR values and changed RHS,
        // not a stale copied matrix or stale control from the preceding solve.
        const int entries = a.NumNonZeroElems();
        double* values = a.HostReadWriteData();
        for (int entry = 0; entry < entries; ++entry) { values[entry] *= 1.125; }
        double* e = exact.HostReadWrite();
        for (int i = 0; i < size; ++i) { e[i] += 0.2 * std::sin(0.13 * i); }
      }
      a.Mult(exact, b);
      const double initial = PreconditionedResidual(a, jacobi, b, zero);
      x = 123.0; // zero-init contract must discard the supplied iterate
      other = -123.0;
      cooperative->Mult(b, x);
      polled->Mult(b, other);
      const double final = PreconditionedResidual(a, jacobi, b, x);
      const double reference_final = PreconditionedResidual(a, jacobi, b, other);
      MFEM_VERIFY(cooperative->GetConverged() && polled->GetConverged() &&
                    final <= tolerance * initial && reference_final <= tolerance * initial &&
                    std::abs(cooperative->GetInitialNorm() - initial) <= 2.0e-12 * initial &&
                    std::abs(cooperative->GetFinalNorm() - final) <= 2.0e-12 * initial,
                  "Cooperative coarse CG failed its independent original-CSR r.Br certificate.");
      mfem::Vector error(x), difference(x);
      error -= exact;
      difference -= other;
      MFEM_VERIFY(Norm(error) <= 4.0 * tolerance * Norm(exact) &&
                    Norm(difference) <= 4.0 * tolerance * Norm(exact),
                  "Cooperative coarse CG differs from the manufactured solution/device CG.");
    }
    if (size == 513)
    {
      cooperative->SetMaxIter(0);
      cooperative->Mult(b, x);
      MFEM_VERIFY(!cooperative->GetConverged() && cooperative->GetNumIterations() == 0 && Norm(x) == 0.0,
                  "Cooperative coarse CG ignored its zero iteration budget.");
      cooperative->SetMaxIter(1);
      cooperative->Mult(b, x);
      MFEM_VERIFY(!cooperative->GetConverged() && cooperative->GetNumIterations() == 1,
                  "Cooperative coarse CG ignored its one iteration budget.");
      cooperative->SetMaxIter(100);
    }
    b = 0.0;
    x = 123.0;
    cooperative->Mult(b, x);
    MFEM_VERIFY(cooperative->GetConverged() && cooperative->GetNumIterations() == 0 && Norm(x) == 0.0,
                "Cooperative coarse CG zero RHS/reset failed.");
  }

  // Invalid curvature and input must not be mistaken for early convergence.
  // A positive Jacobi action is deliberate even for these non-SPD operators.
  constexpr int bad_size = 7;
  mfem::Vector diagonal(bad_size), b(bad_size), x(bad_size);
  diagonal = 1.0;
  mfem::OperatorJacobiSmoother jacobi(diagonal, essential, 1.0);
  for (int failure = 0; failure < 4; ++failure)
  {
    mfem::SparseMatrix a(bad_size);
    for (int row = 0; row < bad_size; ++row)
    {
      const double value = failure == 0 ? -1.0 : failure == 1 ?
        std::numeric_limits<double>::quiet_NaN() : 1.0;
      a.Add(row, row, value);
      if (failure == 2) { a.Add(row, (row + 1) % bad_size, -1.0); }
    }
    a.Finalize();
    Configure(*cooperative, a, false, 100);
    cooperative->SetPreconditioner(jacobi);
    b = 1.0;
    if (failure == 3) { b.HostReadWrite()[bad_size - 1] = std::numeric_limits<double>::quiet_NaN(); }
    cooperative->Mult(b, x);
    MFEM_VERIFY(!cooperative->GetConverged() && Norm(x) == 0.0 &&
                  cooperative->GetNumIterations() == (failure == 3 ? 0 : 1),
                "Cooperative coarse CG accepted negative/zero/nonfinite curvature or nonfinite input.");
  }
  return admitted;
}

void CheckRHSRelativeFGMRES()
{
  constexpr int n = 65;
  Stencil nonsymmetric(n, -0.8, 0.3), identity(n, 0.0, 0.0, 1.0),
          zero_operator(n, 0.0, 0.0, 0.0);
  ChangingDiagonal changing(n);
  mfem::Vector exact(n), rhs(n), x(n), supplied(n), difference(n);
  for (int i = 0; i < n; ++i) { exact(i) = 0.5 + std::cos(0.27 * i); }
  auto solver = ::igrlab::igr_internal::MakeDeviceFGMRESSolver(
    MPI_COMM_WORLD, 3, ::igrlab::igr_internal::KrylovResidualScale::RHS);
  nonsymmetric.Mult(exact, rhs);
  const double rhs_norm = Norm(rhs);
  Configure(*solver, nonsymmetric, false);
  solver->SetPreconditioner(changing);
  x = 123.0;
  solver->Mult(rhs, x);
  CheckSolution(*solver, nonsymmetric, rhs, x, exact, rhs_norm);
  MFEM_VERIFY(solver->GetNumIterations() > 3,
              "RHS-relative flexible FGMRES test did not restart.");

  constexpr double rtol = 1.0e-8;
  Configure(*solver, nonsymmetric, true, 0);
  solver->SetPreconditioner(changing);
  solver->SetRelTol(rtol);
  x = exact;
  x *= 1.0 - 0.25 * rtol;
  supplied = x;
  const double accepted_initial = Residual(nonsymmetric, rhs, x);
  MFEM_VERIFY(accepted_initial > 0.0 && accepted_initial < 0.5 * rtol * rhs_norm,
              "RHS-relative test did not supply a positive acceptable warm residual.");
  solver->Mult(rhs, x);
  difference = x;
  difference -= supplied;
  MFEM_VERIFY(solver->GetConverged() && solver->GetNumIterations() == 0 &&
                Norm(difference) == 0.0 && Residual(nonsymmetric, rhs, x) <= rtol * rhs_norm &&
                std::abs(solver->GetInitialNorm() - accepted_initial) <= 1.0e-13 * rhs_norm &&
                std::abs(solver->GetFinalNorm() - accepted_initial) <= 1.0e-13 * rhs_norm,
              "RHS-relative FGMRES did not accept its current warm residual with zero budget.");

  // A nearby guess just outside the target must spend iterations. Check the
  // zero-budget rejection first, then solve the same equation and workspace.
  x = exact;
  x *= 1.0 - 1.5 * rtol;
  supplied = x;
  const double insufficient_initial = Residual(nonsymmetric, rhs, x);
  MFEM_VERIFY(insufficient_initial > 1.25 * rtol * rhs_norm &&
                insufficient_initial < 1.75 * rtol * rhs_norm,
              "RHS-relative test did not place its warm residual just outside the target.");
  solver->Mult(rhs, x);
  difference = x;
  difference -= supplied;
  MFEM_VERIFY(!solver->GetConverged() && solver->GetNumIterations() == 0 &&
                Norm(difference) == 0.0,
              "RHS-relative FGMRES ignored an exhausted iteration budget.");
  solver->SetMaxIter(200);
  solver->Mult(rhs, x);
  const double warm_final = Residual(nonsymmetric, rhs, x);
  difference = x;
  difference -= exact;
  MFEM_VERIFY(solver->GetConverged() && solver->GetNumIterations() > 0 &&
                warm_final <= rtol * rhs_norm && Norm(difference) < 2.0 * rtol * Norm(exact) &&
                std::abs(solver->GetInitialNorm() - insufficient_initial) <= 1.0e-13 * rhs_norm &&
                std::abs(solver->GetFinalNorm() - warm_final) <= 1.0e-13 * rhs_norm,
              "RHS-relative warm FGMRES failed its independent residual/solution gate.");

  // The retained solver sees a different operator and a much smaller RHS. A
  // stale RHS norm would incorrectly accept this deliberately insufficient x.
  Stencil changed_operator(n, -0.05, 0.015, 0.25);
  double* changed_exact = exact.HostWrite();
  for (int i = 0; i < n; ++i)
  { changed_exact[i] = 0.03 * (0.5 + std::sin(0.13 * i)) + 0.002 * std::cos(0.41 * i); }
  changed_operator.Mult(exact, rhs);
  const double changed_rhs_norm = Norm(rhs);
  x = exact;
  x *= 1.0 - 4.0 * rtol;
  const double changed_initial = Residual(changed_operator, rhs, x);
  MFEM_VERIFY(changed_initial > 3.0 * rtol * changed_rhs_norm &&
                changed_initial < 0.5 * rtol * rhs_norm,
              "Changed-equation test did not distinguish the current RHS stopping scale.");
  Configure(*solver, changed_operator);
  solver->SetPreconditioner(changing);
  solver->SetRelTol(rtol);
  solver->Mult(rhs, x);
  const double changed_final = Residual(changed_operator, rhs, x);
  difference = x;
  difference -= exact;
  MFEM_VERIFY(solver->GetConverged() && solver->GetNumIterations() > 0 &&
                changed_final <= rtol * changed_rhs_norm &&
                Norm(difference) < 2.0 * rtol * Norm(exact) &&
                std::abs(solver->GetInitialNorm() - changed_initial) <= 1.0e-13 * changed_rhs_norm &&
                std::abs(solver->GetFinalNorm() - changed_final) <= 1.0e-13 * changed_rhs_norm,
              "RHS-relative FGMRES used stale operator/RHS data after reconfiguration.");

  auto unpreconditioned = ::igrlab::igr_internal::MakeDeviceFGMRESSolver(
    MPI_COMM_WORLD, 3, ::igrlab::igr_internal::KrylovResidualScale::RHS);
  Configure(*unpreconditioned, identity, false, 0);
  rhs = 0.0;
  x = 123.0;
  unpreconditioned->Mult(rhs, x);
  MFEM_VERIFY(unpreconditioned->GetConverged() && unpreconditioned->GetNumIterations() == 0 && Norm(x) == 0.0,
              "RHS-relative FGMRES zero RHS/reset failed.");
  constexpr double atol = 1.0e-10;
  Configure(*unpreconditioned, identity, true, 0);
  unpreconditioned->SetAbsTol(atol);
  x.HostReadWrite()[n - 1] = 0.5 * atol;
  supplied = x;
  unpreconditioned->Mult(rhs, x);
  difference = x;
  difference -= supplied;
  MFEM_VERIFY(unpreconditioned->GetConverged() && unpreconditioned->GetNumIterations() == 0 &&
                Norm(difference) == 0.0 && Residual(identity, rhs, x) == 0.5 * atol,
              "RHS-relative FGMRES failed its absolute tolerance with zero RHS.");
  unpreconditioned->SetAbsTol(0.0);
  unpreconditioned->Mult(rhs, x);
  MFEM_VERIFY(!unpreconditioned->GetConverged() && unpreconditioned->GetNumIterations() == 0 &&
                Residual(identity, rhs, x) == 0.5 * atol,
              "RHS-relative FGMRES accepted a nonzero residual with zero RHS/tolerance.");
  unpreconditioned->SetMaxIter(2);
  unpreconditioned->Mult(rhs, x);
  MFEM_VERIFY(unpreconditioned->GetConverged() && unpreconditioned->GetNumIterations() == 1 &&
                Residual(identity, rhs, x) == 0.0 && unpreconditioned->GetFinalNorm() == 0.0,
              "RHS-relative FGMRES failed a zero-RHS happy breakdown.");

  Configure(*solver, zero_operator, false, 5);
  rhs = exact;
  solver->Mult(rhs, x);
  MFEM_VERIFY(!solver->GetConverged() && solver->GetNumIterations() == 1 &&
                Norm(x) == 0.0 && Residual(zero_operator, rhs, x) == Norm(rhs),
              "RHS-relative FGMRES accepted an inconsistent Arnoldi breakdown.");
  Configure(*solver, nonsymmetric, false, 1);
  solver->SetPreconditioner(changing);
  nonsymmetric.Mult(exact, rhs);
  solver->Mult(rhs, x);
  const double capped_final = Residual(nonsymmetric, rhs, x);
  MFEM_VERIFY(!solver->GetConverged() && solver->GetNumIterations() == 1 &&
                capped_final > 1.0e-11 * Norm(rhs) &&
                std::abs(solver->GetFinalNorm() - capped_final) <= 1.0e-13 * Norm(rhs),
              "RHS-relative FGMRES violated its one-iteration budget/residual certificate.");
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

  // Successive RK stages reuse a nearly converged solution while A and b
  // change slightly. MFEM scales the recurrence by ||b-A*x0||, but accepts its
  // estimate before recomputing b-A*x. The independent FP64 gate must retain
  // the configured RHS-relative tolerance instead of multiplying the small
  // warm residual by rtol a second time.
  PositiveDiagonal warm_preconditioner(n), warm_reference_preconditioner(n);
  auto warm_gmres = ::igrlab::igr_internal::MakeDeviceFGMRESSolver(MPI_COMM_WORLD);
  mfem_gmres.SetKDim(50); // use the production restart size for the stage sequence
  mfem::Vector nearby_rhs(n);
  constexpr double warm_rtol = 1.0e-10;
  for (int stage = 1; stage <= 3; ++stage)
  {
    Stencil current_operator(n, -0.8 - stage * 1.0e-12, 0.3);
    current_operator.Mult(exact, nearby_rhs);
    double* nearby = nearby_rhs.HostReadWrite();
    for (int i = 0; i < n; ++i)
    { nearby[i] += stage * 1.0e-12 * (1 + i % 7); }
    reference = x;
    const double rhs_norm = Norm(nearby_rhs);
    const double warm_initial = Residual(current_operator, nearby_rhs, x);
    MFEM_VERIFY(warm_initial > 0.0 && warm_initial < 1.0e-8 * rhs_norm,
                "Repeated FGMRES test did not use a near-solution warm start.");
    Configure(*warm_gmres, current_operator);
    warm_gmres->SetPreconditioner(warm_preconditioner);
    warm_gmres->SetRelTol(warm_rtol);
    warm_gmres->Mult(nearby_rhs, x);
    const double true_final = Residual(current_operator, nearby_rhs, x);
    MFEM_VERIFY(warm_gmres->GetConverged() && warm_gmres->GetNumIterations() > 0 &&
                  warm_gmres->GetNumIterations() < 200 && true_final <= warm_rtol * rhs_norm &&
                  std::abs(warm_gmres->GetInitialNorm() - warm_initial) <= 1.0e-13 * warm_initial &&
                  std::abs(warm_gmres->GetFinalNorm() - true_final) <= 1.0e-13 * rhs_norm,
                "Repeated warm FGMRES violated its current-A RHS-relative true-residual gate.");
    Configure(mfem_gmres, current_operator);
    mfem_gmres.SetPreconditioner(warm_reference_preconditioner);
    mfem_gmres.SetRelTol(warm_rtol);
    mfem_gmres.Mult(nearby_rhs, reference);
    MFEM_VERIFY(mfem_gmres.GetConverged() &&
                  Residual(current_operator, nearby_rhs, reference) <= warm_rtol * rhs_norm,
                "MFEM near-solution reference failed its true-residual gate.");
    reference -= x;
    MFEM_VERIFY(Norm(reference) / Norm(exact) < 3.0e-10,
                "Repeated warm device FGMRES disagrees with MFEM.");
  }

  // A zero operator with nonzero RHS has no solution. Happy Arnoldi breakdown
  // and a zero recurrence estimate must not bypass the original-A certificate.
  Stencil zero_operator(n, 0.0, 0.0, 0.0);
  Configure(*gmres, zero_operator, false, 5);
  rhs = exact;
  gmres->Mult(rhs, x);
  MFEM_VERIFY(!gmres->GetConverged() && gmres->GetNumIterations() == 1 &&
                Norm(x) == 0.0 && Residual(zero_operator, rhs, x) == Norm(rhs),
              "Device FGMRES accepted an inconsistent zero-operator system.");

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

  CheckRHSRelativeFGMRES();

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
  PositiveDiagonal jacobi(n);
  Configure(*cg, spd);
  cg->SetPreconditioner(jacobi);
  x = guess;
  const double preconditioned_initial = PreconditionedResidual(spd, jacobi, rhs, x);
  cg->Mult(rhs, x);
  const double preconditioned_final = PreconditionedResidual(spd, jacobi, rhs, x);
  MFEM_VERIFY(cg->GetConverged() && preconditioned_final <= 1.0e-11 * preconditioned_initial &&
                std::abs(cg->GetInitialNorm() - preconditioned_initial) < 1.0e-13 * preconditioned_initial &&
                std::abs(cg->GetFinalNorm() - preconditioned_final) < 1.0e-13 * preconditioned_initial,
              "Preconditioned device CG violated its original-operator r.Br norm gate.");
  Configure(mfem_cg, spd);
  mfem_cg.SetPreconditioner(jacobi);
  reference = guess;
  mfem_cg.Mult(rhs, reference);
  reference -= x;
  MFEM_VERIFY(mfem_cg.GetConverged() && Norm(reference) / Norm(exact) < 3.0e-11,
              "Preconditioned device CG disagrees with MFEM.");
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
  const bool cooperative_admitted = CheckCooperativeCG();
  std::cout << "device_krylov_test passed cooperative_admitted=" << cooperative_admitted
            << " cooperative_csr_jacobi_certificate=true cooperative_size_reset=true"
               " cooperative_failure_paths=true restarted_flexible_mgs=true"
               " true_residual=true cached_reconfigure=true repeated_near_warm=true"
               " rhs_relative_fgmres=true rhs_relative_warm_budget=true rhs_relative_current_equation=true"
               " rhs_relative_zero_rhs_atol=true rhs_relative_breakdown=true"
               " inconsistent_operator_rejected=true happy_breakdown=true"
               " zero_rhs=true iteration_cap=true cg_mfem_parity=true preconditioned_cg=true nonpositive_curvature=true nonfinite_curvature=true\n";
  return 0;
}
