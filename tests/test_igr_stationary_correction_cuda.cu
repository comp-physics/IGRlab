#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_stationary_correction.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace
{
class DiagonalAction final : public mfem::Solver
{
public:
  DiagonalAction(int n, int change, bool inverse = false) : mfem::Solver(n), values_(n)
  {
    for (int i = 0; i < n; ++i)
    {
      double value = 2.0 + 0.03 * i;
      if (change == 1) { value *= 1.0 + 0.01 * std::cos(0.31 * i); }
      if (change == 2) { value *= 3.0; }
      values_(i) = inverse ? 1.0 / value : value;
    }
  }
  void SetOperator(const mfem::Operator& op) override
  { MFEM_VERIFY(op.Height() == height, "Changed test dimension."); }
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    y.SetSize(height);
    y.UseDevice(true);
    const double* input = x.Read();
    const double* diagonal = values_.Read();
    double* output = y.Write();
    mfem::forall(height, [=] MFEM_HOST_DEVICE(int i)
    { output[i] = diagonal[i] * input[i]; });
  }
private:
  mfem::Vector values_;
};

double Norm(const mfem::Vector& x)
{
  const double* values = x.HostRead();
  long double sum = 0.0L;
  for (int i = 0; i < x.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(values[i]), "Nonfinite stationary test result.");
    sum += static_cast<long double>(values[i]) * values[i];
  }
  return std::sqrt(static_cast<double>(sum));
}

double Residual(const mfem::Operator& op, const mfem::Vector& x,
                const mfem::Vector& rhs)
{
  mfem::Vector action(rhs.Size());
  op.Mult(x, action);
  action -= rhs;
  return Norm(action);
}

void Configure(mfem::IterativeSolver& solver, const mfem::Operator& op,
               mfem::Solver& preconditioner)
{
  solver.SetOperator(op);
  solver.SetPreconditioner(preconditioner);
  solver.SetRelTol(1.0e-10);
  solver.SetAbsTol(0.0);
  solver.iterative_mode = true;
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  constexpr int n = 513; // Multiple reduction strides, including a partial stride.
  DiagonalAction initial(n, 0), changed(n, 1), divergent(n, 2), retained_inverse(n, 0, true);
  mfem::Vector exact(n), guess(n), rhs(n), x(n);
  for (int i = 0; i < n; ++i)
  {
    exact(i) = 0.5 + std::cos(0.27 * i);
    guess(i) = 0.2 * std::sin(0.53 * i);
  }
  changed.Mult(exact, rhs);
  auto solver = ::igrlab::igr_internal::MakeStationaryCorrectionSolver(MPI_COMM_WORLD, 8);
  Configure(*solver, changed, retained_inverse);
  x = guess;
  const double initial_norm = Residual(changed, x, rhs);
  solver->Mult(rhs, x);
  const double final_norm = Residual(changed, x, rhs);
  MFEM_VERIFY(solver->GetConverged() && final_norm <= 1.0e-10 * initial_norm &&
                std::abs(solver->GetInitialNorm() - initial_norm) <= 1.0e-12 * initial_norm &&
                std::abs(solver->GetFinalNorm() - final_norm) <= 1.0e-12 * initial_norm,
              "Stationary changed-operator result violates its true-residual gate.");
  mfem::Vector error(x);
  error -= exact;
  MFEM_VERIFY(Norm(error) / Norm(exact) < 1.0e-10, "Stationary solution error too large.");

  // The cached first inverse cannot stabilize this changed operator at omega=1.
  // Rejection must restore the supplied guess exactly, enabling a safe fallback.
  divergent.Mult(exact, rhs);
  Configure(*solver, divergent, retained_inverse);
  x = guess;
  solver->Mult(rhs, x);
  MFEM_VERIFY(!solver->GetConverged(), "Divergent stationary trial was accepted.");
  const double* restored = x.HostRead();
  const double* saved = guess.HostRead();
  for (int i = 0; i < n; ++i)
  { MFEM_VERIFY(restored[i] == saved[i], "Failed stationary trial did not restore its guess."); }
  mfem::FGMRESSolver fallback(MPI_COMM_WORLD);
  Configure(fallback, divergent, retained_inverse);
  fallback.SetMaxIter(20);
  fallback.SetPrintLevel(0);
  fallback.Mult(rhs, x);
  MFEM_VERIFY(fallback.GetConverged() && Residual(divergent, x, rhs) / Norm(rhs) < 1.0e-10,
              "Charged FGMRES fallback failed after a rejected stationary trial.");

  // Zero RHS and a zero starting residual must pass an exact zero tolerance.
  Configure(*solver, initial, retained_inverse);
  rhs = 0.0;
  x = guess;
  solver->iterative_mode = false;
  solver->Mult(rhs, x);
  MFEM_VERIFY(solver->GetConverged() && solver->GetInitialNorm() == 0.0 &&
                solver->GetFinalNorm() == 0.0 && Norm(x) == 0.0,
              "Zero-RHS stationary solve failed.");
  std::cout << "stationary_correction_test passed changed_operator=true "
               "rejected_trial_restored=true fallback=true zero_rhs=true\n";
  return 0;
}
