#include "../src/physics/igr/IGR_dense_inverse.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
// n=33 deliberately crosses the TF32 panel boundary. Nonsymmetry makes a
// transposition or column-major packing mistake observable.
class TestOperator final : public mfem::Operator
{
public:
  explicit TestOperator(double change) : mfem::Operator(33), entries_(33 * 33)
  {
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        double value = 0.012 * std::sin(0.19 * (i + 1) * (j + 3));
        if (i == j) { value += 3.0 + 0.03 * i + change * (0.5 + 0.01 * i); }
        if (i + 1 == j) { value += 0.55 + 0.1 * change; }
        if (i == j + 1) { value -= 0.27; }
        entries_[i * width + j] = value;
      }
    }
  }

  double Entry(int i, int j) const { return entries_[i * width + j]; }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  { Apply(x, y, false); }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  { Apply(x, y, true); }

private:
  void Apply(const mfem::Vector& x, mfem::Vector& y, bool transpose) const
  {
    const double* input = x.HostRead();
    y.SetSize(height);
    double* output = y.HostWrite();
    for (int i = 0; i < height; ++i)
    {
      long double value = 0.0L;
      for (int j = 0; j < width; ++j)
      {
        value += static_cast<long double>(transpose ? Entry(j, i) : Entry(i, j)) *
                 input[j];
      }
      output[i] = static_cast<double>(value);
    }
  }
  std::vector<double> entries_;
};

// Independent pivoted host elimination, with wider accumulation than the
// CUDA/MFEM FP64 solve. Manufactured solution agreement also checks this oracle.
mfem::Vector ReferenceSolve(const TestOperator& op, const mfem::Vector& rhs)
{
  const int n = op.Height();
  std::vector<long double> a(n * n), b(n);
  const double* input = rhs.HostRead();
  for (int i = 0; i < n; ++i)
  {
    b[i] = input[i];
    for (int j = 0; j < n; ++j) { a[i * n + j] = op.Entry(i, j); }
  }
  for (int k = 0; k < n; ++k)
  {
    int pivot = k;
    for (int i = k + 1; i < n; ++i)
    {
      if (std::abs(a[i * n + k]) > std::abs(a[pivot * n + k])) { pivot = i; }
    }
    MFEM_VERIFY(std::abs(a[pivot * n + k]) > 1.0e-12L, "Singular test oracle.");
    for (int j = k; j < n; ++j) { std::swap(a[k * n + j], a[pivot * n + j]); }
    std::swap(b[k], b[pivot]);
    for (int i = k + 1; i < n; ++i)
    {
      const long double factor = a[i * n + k] / a[k * n + k];
      for (int j = k + 1; j < n; ++j) { a[i * n + j] -= factor * a[k * n + j]; }
      b[i] -= factor * b[k];
    }
  }
  for (int i = n - 1; i >= 0; --i)
  {
    for (int j = i + 1; j < n; ++j) { b[i] -= a[i * n + j] * b[j]; }
    b[i] /= a[i * n + i];
  }
  mfem::Vector result(n);
  for (int i = 0; i < n; ++i) { result(i) = static_cast<double>(b[i]); }
  return result;
}

double RelativeDifference(const mfem::Vector& left, const mfem::Vector& right)
{
  const double* a = left.HostRead();
  const double* b = right.HostRead();
  long double difference = 0.0L, scale = 0.0L;
  for (int i = 0; i < left.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite test vector.");
    const long double delta = static_cast<long double>(a[i]) - b[i];
    difference += delta * delta;
    scale += static_cast<long double>(b[i]) * b[i];
  }
  return std::sqrt(static_cast<double>(difference / scale));
}

double TrueResidual(const TestOperator& op, const mfem::Vector& x,
                    const mfem::Vector& rhs)
{
  mfem::Vector action(op.Height());
  op.Mult(x, action);
  return RelativeDifference(action, rhs);
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  TestOperator initial(0.0), current(1.0);
  mfem::Vector exact(initial.Height()), original_rhs, transpose_rhs, current_rhs;
  for (int i = 0; i < exact.Size(); ++i)
  {
    exact(i) = std::sin(0.23 * (i + 1)) + 0.3 * std::cos(0.41 * (i + 2));
  }
  initial.Mult(exact, original_rhs);
  initial.MultTranspose(exact, transpose_rhs);
  current.Mult(exact, current_rhs);
  const mfem::Vector reference = ReferenceSolve(current, current_rhs);
  MFEM_VERIFY(RelativeDifference(reference, exact) <= 1.0e-13,
              "Host reference disagrees with manufactured solution.");

  for (const std::string mode : {"fp64", "fp32", "tf32"})
  {
    auto preconditioner = dg_igr::igr_internal::MakeDenseInversePreconditioner(
      initial, mode, MPI_COMM_WORLD);
    const double apply_tolerance = mode == "fp64" ? 1.0e-12 :
                                   (mode == "fp32" ? 2.0e-6 : 5.0e-3);
    mfem::Vector direct, transpose, reused, uncorrected;
    preconditioner->Mult(original_rhs, direct);
    const double direct_error = RelativeDifference(direct, exact);
    preconditioner->MultTranspose(transpose_rhs, transpose);
    const double transpose_error = RelativeDifference(transpose, exact);
    MFEM_VERIFY(direct_error <= apply_tolerance && transpose_error <= apply_tolerance,
                "Dense inverse direct/transpose accuracy failed in " << mode);

    preconditioner->SetOperator(current);
    preconditioner->Mult(original_rhs, reused);
    MFEM_VERIFY(RelativeDifference(reused, direct) <= 1.0e-14,
                "SetOperator unexpectedly rebuilt the cached inverse in " << mode);
    preconditioner->Mult(current_rhs, uncorrected);
    const double uncorrected_residual = TrueResidual(current, uncorrected, current_rhs);
    MFEM_VERIFY(uncorrected_residual > 1.0e-3,
                "Changed-operator test must require outer correction.");

    mfem::FGMRESSolver solver(MPI_COMM_WORLD);
    solver.SetOperator(current);
    solver.SetPreconditioner(*preconditioner);
    solver.SetRelTol(1.0e-12);
    solver.SetAbsTol(0.0);
    solver.SetMaxIter(80);
    solver.SetKDim(20);
    solver.SetPrintLevel(0);
    solver.iterative_mode = false;
    mfem::Vector corrected(initial.Height());
    corrected.UseDevice(true);
    solver.Mult(current_rhs, corrected);
    const double residual = TrueResidual(current, corrected, current_rhs);
    const double solution_error = RelativeDifference(corrected, reference);
    MFEM_VERIFY(solver.GetConverged() && residual <= 1.0e-10 && solution_error <= 1.0e-9,
                "Current-operator corrected solve failed in " << mode
                << ": residual=" << residual << " error=" << solution_error);
    std::cout << "dense_inverse_test mode=" << mode
              << " direct_error=" << direct_error
              << " transpose_error=" << transpose_error
              << " uncorrected_residual=" << uncorrected_residual
              << " corrected_residual=" << residual
              << " corrected_solution_error=" << solution_error
              << " iterations=" << solver.GetNumIterations() << '\n';
  }
  return 0;
}
