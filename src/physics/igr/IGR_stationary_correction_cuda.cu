#include "igrlab/compat.hpp"
#include "IGR_stationary_correction.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>

namespace igrlab
{
namespace igr_internal
{
namespace
{
__global__ void ResidualNormPair(int n, const double* initial,
                                const double* final, double* sums)
{
  __shared__ double initial_sums[256];
  __shared__ double final_sums[256];
  const int lane = threadIdx.x;
  double first = 0.0, last = 0.0;
  for (int i = lane; i < n; i += blockDim.x)
  {
    first += initial[i] * initial[i];
    last += final[i] * final[i];
  }
  initial_sums[lane] = first;
  final_sums[lane] = last;
  __syncthreads();
  for (int stride = 128; stride > 0; stride /= 2)
  {
    if (lane < stride)
    {
      initial_sums[lane] += initial_sums[lane + stride];
      final_sums[lane] += final_sums[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0)
  {
    sums[0] = initial_sums[0];
    sums[1] = final_sums[0];
  }
}

class StationaryCorrection final : public mfem::IterativeSolver
{
public:
  StationaryCorrection(MPI_Comm comm, int sweeps, double damping)
    : mfem::IterativeSolver(comm), sweeps_(sweeps), damping_(damping)
  {
    static_assert(std::is_same<mfem::real_t, double>::value,
                  "Stationary correction experiment requires FP64 MFEM.");
    int ranks = 0;
    MPI_Comm_size(comm, &ranks);
    MFEM_VERIFY(ranks == 1 && mfem::Device::Allows(mfem::Backend::CUDA),
                "Stationary correction requires rank-one CUDA.");
    MFEM_VERIFY(sweeps > 0 && sweeps <= 16 && std::isfinite(damping) &&
                  damping > 0.0 && damping <= 2.0,
                "Stationary correction needs 1..16 sweeps and damping in (0,2].");
    norms_.SetSize(2);
  }

  ~StationaryCorrection() override
  {
    std::cout << "[igr-stationary] event=run-summary"
              << " trials=" << trials_ << " accepted=" << accepted_
              << " rejected=" << (trials_ - accepted_)
              << " sweeps_requested=" << sweeps_
              << " sweeps_executed=" << sweeps_executed_ << std::endl;
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() > 0 && op.Height() == op.Width(),
                "Stationary correction requires a nonempty square operator.");
    // Cached trials attach a fresh telemetry wrapper each stage. Do not call
    // SetOperator on a preconditioner pointer retained from the previous stage.
    oper = &op;
    prec = nullptr;
    height = op.Height();
    width = op.Width();
    for (mfem::Vector* work : {&starting_guess_, &initial_residual_,
                               &residual_, &action_, &correction_})
    {
      work->SetSize(height);
      work->UseDevice(true);
    }
  }

  void Mult(const mfem::Vector& b, mfem::Vector& x) const override
  {
    MFEM_VERIFY(oper && prec && b.Size() == height && x.Size() == height,
                "Configure current stationary operator/preconditioner before Mult.");
    MFEM_VERIFY(std::isfinite(rel_tol) && std::isfinite(abs_tol) &&
                  rel_tol >= 0.0 && abs_tol >= 0.0,
                "Invalid stationary residual tolerances.");
    x.UseDevice(true);
    if (!iterative_mode) { x = 0.0; }
    starting_guess_ = x;
    oper->Mult(x, action_);
    subtract(b, action_, initial_residual_);
    residual_ = initial_residual_;
    for (int sweep = 0; sweep < sweeps_; ++sweep)
    {
      prec->Mult(residual_, correction_);
      x.Add(damping_, correction_);
      if (sweep + 1 < sweeps_)
      {
        oper->Mult(x, action_);
        subtract(b, action_, residual_);
      }
    }
    // Re-evaluate the actual current equation, never trust a residual recurrence.
    oper->Mult(x, action_);
    subtract(b, action_, residual_);
    ResidualNormPair<<<1, 256>>>(height, initial_residual_.Read(),
                                residual_.Read(), norms_.Write());
    MFEM_VERIFY(cudaGetLastError() == cudaSuccess,
                "Stationary residual reduction launch failed.");
    const double* sums = norms_.HostRead(); // One combined scalar synchronization.
    initial_norm = std::sqrt(sums[0]);
    final_norm = std::sqrt(sums[1]);
    final_iter = sweeps_;
    converged = std::isfinite(initial_norm) && std::isfinite(final_norm) &&
                final_norm <= std::max(abs_tol, rel_tol * initial_norm);
    ++trials_;
    sweeps_executed_ += sweeps_;
    if (converged) { ++accepted_; }
    if (!converged) { x = starting_guess_; }
  }

private:
  int sweeps_;
  double damping_;
  mutable mfem::Vector starting_guess_, initial_residual_, residual_, action_, correction_;
  mutable mfem::Array<double> norms_;
  mutable long long trials_ = 0, accepted_ = 0, sweeps_executed_ = 0;
};
}

std::unique_ptr<mfem::IterativeSolver> MakeStationaryCorrectionSolver(
  MPI_Comm comm, int sweeps, double damping)
{
  return std::make_unique<StationaryCorrection>(comm, sweeps, damping);
}
}
}
