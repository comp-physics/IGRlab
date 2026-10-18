#include "IGR_device_krylov.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

namespace igrlab
{
namespace igr_internal
{
namespace
{
constexpr int threads = 256;

void CheckCUDA(cudaError_t result)
{ MFEM_VERIFY(result == cudaSuccess, "IGR device Krylov: " << cudaGetErrorString(result)); }
void CheckBLAS(cublasStatus_t result)
{ MFEM_VERIFY(result == CUBLAS_STATUS_SUCCESS, "IGR device Krylov cuBLAS status " << result); }

__global__ void SubtractProjection(int n, const double* v, const double* dot,
                                    double* residual)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { residual[i] -= *dot * v[i]; }
}

__global__ void Normalize(int n, const double* input, const double* norm,
                           double* output)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { output[i] = *norm > 0.0 ? input[i] / *norm : 0.0; }
}

__global__ void StartArnoldi(const double* norm, double* g, double* control)
{ g[0] = *norm; control[1] = 0.0; }

__global__ void RotateColumn(int j, int ld, double* h, double* cs,
                              double* sn, double* g, double* control)
{
  double* column = h + j * ld;
  const double next_norm = column[j + 1];
  for (int k = 0; k < j; ++k)
  {
    const double first = cs[k] * column[k] + sn[k] * column[k + 1];
    column[k + 1] = -sn[k] * column[k] + cs[k] * column[k + 1];
    column[k] = first;
  }
  const double dx = column[j], dy = column[j + 1];
  // Use MFEM's signed plane-rotation convention, including its dy==0 branch.
  if (dy == 0.0) { cs[j] = 1.0; sn[j] = 0.0; }
  else if (fabs(dy) > fabs(dx))
  {
    const double t = dx / dy;
    sn[j] = 1.0 / sqrt(1.0 + t * t);
    cs[j] = t * sn[j];
  }
  else
  {
    const double t = dy / dx;
    cs[j] = 1.0 / sqrt(1.0 + t * t);
    sn[j] = t * cs[j];
  }
  column[j] = cs[j] * dx + sn[j] * dy;
  column[j + 1] = 0.0;
  g[j + 1] = -sn[j] * g[j];
  g[j] *= cs[j];
  control[0] = fabs(g[j + 1]);
  control[1] = next_norm == 0.0 ? 1.0 : 0.0;
  if (!isfinite(control[0]) || !isfinite(column[j])) { control[1] = 2.0; }
}

__global__ void BackSolve(int used, int ld, const double* h, const double* g,
                           double* y, double* control)
{
  for (int j = 0; j < used; ++j) { y[j] = g[j]; }
  for (int j = used - 1; j >= 0; --j)
  {
    const double diagonal = h[j * ld + j];
    if (diagonal == 0.0 || !isfinite(diagonal))
    { control[1] = 2.0; return; }
    y[j] /= diagonal;
    if (!isfinite(y[j])) { control[1] = 2.0; return; }
    for (int i = j - 1; i >= 0; --i) { y[i] -= h[j * ld + i] * y[j]; }
  }
}

__global__ void AddCorrection(int n, const double* direction,
                               const double* weight, const double* control,
                               double* x)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && control[1] != 2.0) { x[i] += *weight * direction[i]; }
}

// CG control layout: alternating r.Br slots [0,1], p.Ap [2], alpha [3],
// beta [4], failure flag [5]. No recurrence scalar is transferred to the host
// for use in vector arithmetic.
__global__ void PrepareCGAlpha(int old_slot, double* c)
{
  if (!(c[2] > 0.0) || !isfinite(c[2]) || !(c[old_slot] >= 0.0))
  { c[5] = 1.0; c[3] = 0.0; }
  else { c[3] = c[old_slot] / c[2]; }
  if (!isfinite(c[3])) { c[5] = 1.0; c[3] = 0.0; }
}

__global__ void CGUpdate(int n, const double* direction, const double* action,
                          const double* c, double* x, double* residual)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && c[5] == 0.0)
  { x[i] += c[3] * direction[i]; residual[i] -= c[3] * action[i]; }
}

__global__ void PrepareCGBeta(int old_slot, double* c)
{
  c[4] = c[1 - old_slot] / c[old_slot];
  if (!(c[4] >= 0.0) || !isfinite(c[4])) { c[5] = 1.0; c[4] = 0.0; }
}

__global__ void CGDirection(int n, const double* z, const double* c, double* p)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && c[5] == 0.0) { p[i] = z[i] + c[4] * p[i]; }
}

// One resident grid owns the entire coarse solve. All control decisions are
// uniform and follow grid barriers; no thread/block may leave a reduction early.
namespace cg = cooperative_groups;
enum CooperativeControl
{
  coop_initial, coop_final, coop_iterations, coop_state,
  coop_norm, coop_next, coop_alpha, coop_beta, coop_tolerance, coop_sum,
  coop_control_size
};
// state: 0 running, 1 certified convergence, 2 numerical failure, 3 budget used.
struct CooperativeCGData
{
  int n, max_iterations;
  const int* rows;
  const int* columns;
  const double* values;
  const double* inverse_diagonal;
  const double* b;
  double* x;
  double* residual;
  double* direction;
  double* action;
  double* partials;
  double* control;
  double relative_squared, absolute_squared;
};

__device__ void CooperativeSum(double value, double* shared,
                                 const CooperativeCGData& v,
                                 const cg::grid_group& grid, double* result)
{
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = threads / 2; stride > 0; stride /= 2)
  {
    if (threadIdx.x < stride) { shared[threadIdx.x] += shared[threadIdx.x + stride]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { v.partials[blockIdx.x] = shared[0]; }
  grid.sync();
  if (grid.thread_rank() == 0)
  {
    double sum = 0.0;
    for (unsigned int block = 0; block < gridDim.x; ++block) { sum += v.partials[block]; }
    *result = sum;
  }
  grid.sync();
}

__device__ double CooperativeRow(const CooperativeCGData& v, int row,
                                   const double* input)
{
  double sum = 0.0;
  for (int entry = v.rows[row]; entry < v.rows[row + 1]; ++entry)
  { sum += v.values[entry] * input[v.columns[entry]]; }
  return sum;
}

__global__ void CooperativeCGKernel(CooperativeCGData v)
{
  __shared__ double shared[threads];
  const cg::grid_group grid = cg::this_grid();
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  double* const c = v.control;
  double local = 0.0;
  for (long long row = first; row < v.n; row += stride)
  {
    v.x[row] = 0.0;
    const double r = v.b[row];
    const double z = v.inverse_diagonal[row] * r;
    v.residual[row] = r;
    v.direction[row] = z;
    local += r * z;
  }
  CooperativeSum(local, shared, v, grid, c + coop_norm);
  if (first == 0)
  {
    c[coop_initial] = c[coop_final] = c[coop_norm];
    c[coop_iterations] = 0.0;
    c[coop_tolerance] = fmax(v.absolute_squared, v.relative_squared * c[coop_norm]);
    c[coop_state] = !(c[coop_norm] >= 0.0) || !isfinite(c[coop_norm]) ? 2.0 :
      c[coop_norm] <= c[coop_tolerance] ? 1.0 : v.max_iterations == 0 ? 3.0 : 0.0;
  }
  grid.sync();
  while (c[coop_state] == 0.0)
  {
    local = 0.0;
    for (long long row = first; row < v.n; row += stride)
    {
      const double ap = CooperativeRow(v, row, v.direction);
      v.action[row] = ap;
      local += v.direction[row] * ap;
    }
    CooperativeSum(local, shared, v, grid, c + coop_sum);
    if (first == 0)
    {
      c[coop_iterations] += 1.0;
      c[coop_alpha] = c[coop_norm] / c[coop_sum];
      if (!(c[coop_sum] > 0.0) || !isfinite(c[coop_sum]) ||
          !isfinite(c[coop_alpha])) { c[coop_state] = 2.0; }
    }
    grid.sync();
    if (c[coop_state] != 0.0) { break; }

    local = 0.0;
    for (long long row = first; row < v.n; row += stride)
    {
      v.x[row] += c[coop_alpha] * v.direction[row];
      const double r = v.residual[row] - c[coop_alpha] * v.action[row];
      v.residual[row] = r;
      local += r * (v.inverse_diagonal[row] * r);
    }
    CooperativeSum(local, shared, v, grid, c + coop_next);
    if (first == 0)
    {
      c[coop_final] = c[coop_next];
      if (!(c[coop_next] >= 0.0) || !isfinite(c[coop_next])) { c[coop_state] = 2.0; }
    }
    grid.sync();
    if (c[coop_state] != 0.0) { break; }

    if (c[coop_next] <= c[coop_tolerance])
    {
      // The original CSR, not a recurrence residual, authorizes convergence.
      // On drift, restart from this residual within the same iteration budget.
      local = 0.0;
      for (long long row = first; row < v.n; row += stride)
      {
        const double r = v.b[row] - CooperativeRow(v, row, v.x);
        const double z = v.inverse_diagonal[row] * r;
        v.residual[row] = r;
        v.direction[row] = z;
        local += r * z;
      }
      CooperativeSum(local, shared, v, grid, c + coop_norm);
      if (first == 0)
      {
        c[coop_final] = c[coop_norm];
        c[coop_state] = !(c[coop_norm] >= 0.0) || !isfinite(c[coop_norm]) ? 2.0 :
          c[coop_norm] <= c[coop_tolerance] ? 1.0 :
          c[coop_iterations] >= v.max_iterations ? 3.0 : 0.0;
      }
      grid.sync();
      continue;
    }

    if (first == 0)
    {
      c[coop_beta] = c[coop_next] / c[coop_norm];
      c[coop_norm] = c[coop_next];
      if (!(c[coop_beta] >= 0.0) || !isfinite(c[coop_beta])) { c[coop_state] = 2.0; }
      else if (c[coop_iterations] >= v.max_iterations) { c[coop_state] = 3.0; }
    }
    grid.sync();
    if (c[coop_state] != 0.0) { break; }
    for (long long row = first; row < v.n; row += stride)
    {
      v.direction[row] = v.inverse_diagonal[row] * v.residual[row] +
                         c[coop_beta] * v.direction[row];
    }
    grid.sync();
  }
}

class DeviceKrylovBase : public mfem::IterativeSolver
{
public:
  explicit DeviceKrylovBase(MPI_Comm comm) : mfem::IterativeSolver(comm)
  {
    static_assert(std::is_same<mfem::real_t, double>::value,
                  "Device Krylov requires FP64 MFEM.");
    int ranks = 0;
    MPI_Comm_size(comm, &ranks);
    MFEM_VERIFY(ranks == 1 && mfem::Device::Allows(mfem::Backend::CUDA) &&
                  !mfem::Device::Allows(mfem::Backend::CEED_MASK | mfem::Backend::OCCA_MASK),
                "Device Krylov requires ordinary rank-one CUDA.");
    CheckBLAS(cublasCreate(&handle_));
    // The application and MFEM ordinary-CUDA operators use the default stream.
    CheckBLAS(cublasSetStream(handle_, nullptr));
    CheckBLAS(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE));
  }
  ~DeviceKrylovBase() override { cublasDestroy(handle_); }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() > 0 && op.Height() == op.Width(),
                "Device Krylov requires a nonempty square operator.");
    // A cached solver must not dereference a previous stage's telemetry wrapper.
    oper = &op;
    prec = nullptr;
    height = op.Height();
    width = op.Width();
    SetVector(residual_);
    SetVector(action_);
  }
  mfem::MemoryClass GetMemoryClass() const override
  { return mfem::MemoryClass::DEVICE; }

protected:
  void SetVector(mfem::Vector& v) const
  { v.SetSize(height); v.UseDevice(true); }
  void Validate(const mfem::Vector& b, mfem::Vector& x) const
  {
    MFEM_VERIFY(oper && b.Size() == height && x.Size() == height &&
                  max_iter >= 0 && std::isfinite(rel_tol) && rel_tol >= 0.0 &&
                  std::isfinite(abs_tol) && abs_tol >= 0.0,
                "Invalid device Krylov configuration.");
    MFEM_VERIFY(controller == nullptr && dot_oper == nullptr,
                "Device Krylov does not implement custom monitors or inner products.");
    b.UseDevice(true);
    x.UseDevice(true);
    converged = false;
    final_iter = 0;
  }
  void TrueResidual(const mfem::Vector& b, const mfem::Vector& x) const
  {
    oper->Mult(x, action_);
    // MFEM declares this vector operation as a hidden friend; use ADL.
    subtract(b, action_, residual_);
  }
  void InitialResidual(const mfem::Vector& b, mfem::Vector& x) const
  {
    if (iterative_mode) { TrueResidual(b, x); }
    else { x = 0.0; residual_ = b; }
  }
  void Dot(const mfem::Vector& x, const mfem::Vector& y, double* out) const
  { CheckBLAS(cublasDdot(handle_, height, x.Read(), 1, y.Read(), 1, out)); }
  void Norm(const mfem::Vector& x, double* out) const
  { CheckBLAS(cublasDnrm2(handle_, height, x.Read(), 1, out)); }
  void ReadControl(double* host, const double* device, int size) const
  {
    CheckCUDA(cudaGetLastError());
    CheckCUDA(cudaMemcpy(host, device, size * sizeof(double), cudaMemcpyDeviceToHost));
  }
  void PrintIteration(const char* method) const
  {
    if (print_options.iterations)
    { mfem::out << method << " iteration=" << final_iter << " residual=" << final_norm << '\n'; }
  }
  cublasHandle_t handle_ = nullptr;
  mutable mfem::Vector residual_, action_;
};

class DeviceFGMRES final : public DeviceKrylovBase
{
public:
  DeviceFGMRES(MPI_Comm comm, int restart_dimension,
               KrylovResidualScale residual_scale)
    : DeviceKrylovBase(comm), m_(restart_dimension), residual_scale_(residual_scale),
      basis_(restart_dimension + 1), directions_(restart_dimension)
  {
    MFEM_VERIFY(m_ >= 1 && m_ <= 256, "Device FGMRES restart dimension must be 1..256.");
    h_.SetSize((m_ + 1) * m_);
    cs_.SetSize(m_);
    sn_.SetSize(m_);
    g_.SetSize(m_ + 1);
    coefficients_.SetSize(m_);
    control_.SetSize(2);
    for (auto* v : {&h_, &cs_, &sn_, &g_, &coefficients_, &control_})
    { v->UseDevice(true); }
  }

  void Mult(const mfem::Vector& b, mfem::Vector& x) const override
  {
    Validate(b, x);
    InitialResidual(b, x);
    double* control = control_.Write();
    CheckCUDA(cudaMemset(control, 0, 2 * sizeof(double)));
    Norm(residual_, control);
    // The second slot temporarily holds ||b||; StartArnoldi resets its flag.
    Norm(b, control + 1);
    double status[2];
    ReadControl(status, control, 2);
    initial_norm = final_norm = status[0];
    // Default to pinned MFEM's initial-residual-scaled recurrence criterion;
    // the opt-in RHS mode uses the same target as the true-residual gate.
    // Certify the corrected iterate separately against the original RHS, since
    // the recurrence estimate can fall below the FP64 b-A*x cancellation floor.
    const double reference_norm = residual_scale_ == KrylovResidualScale::RHS
                                    ? status[1] : initial_norm;
    const double tolerance = std::max(abs_tol, rel_tol * reference_norm);
    const double true_tolerance = std::max(abs_tol, rel_tol * status[1]);
    if (!std::isfinite(initial_norm) || !std::isfinite(status[1])) { return; }
    if (final_norm <= tolerance && final_norm <= true_tolerance)
    { converged = true; return; }
    const int blocks = (height + threads - 1) / threads;
    double* h = h_.Write();
    double* cs = cs_.Write();
    double* sn = sn_.Write();
    double* g = g_.Write();
    double* coefficients = coefficients_.Write();
    while (final_iter < max_iter)
    {
      SetVector(basis_[0]);
      Normalize<<<blocks, threads>>>(height, residual_.Read(), control, basis_[0].Write());
      CheckCUDA(cudaMemset(g, 0, (m_ + 1) * sizeof(double)));
      StartArnoldi<<<1, 1>>>(control, g, control);
      int used = 0;
      for (int j = 0; j < m_ && final_iter < max_iter; ++j)
      {
        SetVector(directions_[j]);
        SetVector(basis_[j + 1]);
        if (prec) { prec->Mult(basis_[j], directions_[j]); }
        else { directions_[j] = basis_[j]; }
        oper->Mult(directions_[j], residual_);
        // Ordered modified Gram-Schmidt: each dot sees the preceding update.
        // A batched basis dot would change the method to classical GS.
        for (int k = 0; k <= j; ++k)
        {
          double* entry = h + j * (m_ + 1) + k;
          Dot(residual_, basis_[k], entry);
          SubtractProjection<<<blocks, threads>>>(height, basis_[k].Read(), entry,
                                                   residual_.ReadWrite());
        }
        double* next_norm = h + j * (m_ + 1) + j + 1;
        Norm(residual_, next_norm);
        Normalize<<<blocks, threads>>>(height, residual_.Read(), next_norm,
                                       basis_[j + 1].Write());
        RotateColumn<<<1, 1>>>(j, m_ + 1, h, cs, sn, g, control);
        ReadControl(status, control, 2); // one host convergence read per iteration
        ++final_iter;
        used = j + 1;
        final_norm = status[0];
        PrintIteration("IGR CUDA FGMRES");
        if (status[1] != 0.0 || !std::isfinite(final_norm) || final_norm <= tolerance)
        { break; }
      }
      if (status[1] == 2.0 || !std::isfinite(final_norm)) { return; }
      const bool recurrence_converged = final_norm <= tolerance;
      BackSolve<<<1, 1>>>(used, m_ + 1, h, g, coefficients, control);
      for (int j = 0; j < used; ++j)
      {
        AddCorrection<<<blocks, threads>>>(height, directions_[j].Read(),
                                            coefficients + j, control, x.ReadWrite());
      }
      // Require both the selected recurrence/restart criterion and the independent
      // current-A, RHS-relative true residual. A failed certificate restarts
      // within the same iteration budget, without shrinking the true target
      // to the preceding warm-start residual.
      TrueResidual(b, x);
      Norm(residual_, control);
      ReadControl(status, control, 2);
      final_norm = status[0];
      if (status[1] == 2.0 || !std::isfinite(final_norm)) { return; }
      if ((recurrence_converged || final_norm <= tolerance) &&
          final_norm <= true_tolerance)
      { converged = true; return; }
      if (status[1] == 1.0) { return; } // Arnoldi breakdown without a true solution
    }
  }

private:
  int m_;
  const KrylovResidualScale residual_scale_;
  mutable std::vector<mfem::Vector> basis_, directions_;
  mutable mfem::Vector h_, cs_, sn_, g_, coefficients_, control_;
};

class DeviceCG final : public DeviceKrylovBase
{
public:
  explicit DeviceCG(MPI_Comm comm) : DeviceKrylovBase(comm)
  { control_.SetSize(6); control_.UseDevice(true); }

  void SetOperator(const mfem::Operator& op) override
  {
    DeviceKrylovBase::SetOperator(op);
    SetVector(z_);
    SetVector(direction_);
  }

  void Mult(const mfem::Vector& b, mfem::Vector& x) const override
  {
    Validate(b, x);
    InitialResidual(b, x);
    if (prec) { prec->Mult(residual_, z_); }
    else { z_ = residual_; }
    direction_ = z_;
    double* c = control_.Write();
    CheckCUDA(cudaMemset(c, 0, 6 * sizeof(double)));
    Dot(residual_, z_, c);
    double status[6];
    ReadControl(status, c, 6);
    initial_norm = final_norm = status[0] >= 0.0 ? std::sqrt(status[0]) : status[0];
    if (!(status[0] >= 0.0) || !std::isfinite(status[0])) { return; }
    const double tolerance_squared = std::max(abs_tol * abs_tol,
                                               rel_tol * rel_tol * status[0]);
    if (status[0] <= tolerance_squared) { converged = true; return; }
    const int blocks = (height + threads - 1) / threads;
    int old_slot = 0;
    while (final_iter < max_iter)
    {
      oper->Mult(direction_, action_);
      Dot(direction_, action_, c + 2);
      PrepareCGAlpha<<<1, 1>>>(old_slot, c);
      CGUpdate<<<blocks, threads>>>(height, direction_.Read(), action_.Read(), c,
                                    x.ReadWrite(), residual_.ReadWrite());
      if (prec) { prec->Mult(residual_, z_); }
      else { z_ = residual_; }
      Dot(residual_, z_, c + 1 - old_slot);
      ReadControl(status, c, 6); // one host convergence read, no host recurrence
      ++final_iter;
      const double norm_squared = status[1 - old_slot];
      final_norm = norm_squared >= 0.0 ? std::sqrt(norm_squared) : norm_squared;
      PrintIteration("IGR CUDA CG");
      if (status[5] != 0.0 || !(norm_squared >= 0.0) || !std::isfinite(norm_squared))
      { return; }
      if (norm_squared <= tolerance_squared)
      {
        // Preserve the coarse solver's preconditioned norm policy, but check
        // it using a freshly evaluated residual from the original operator.
        TrueResidual(b, x);
        if (prec) { prec->Mult(residual_, z_); }
        else { z_ = residual_; }
        Dot(residual_, z_, c + 1 - old_slot);
        ReadControl(status, c, 6);
        const double true_squared = status[1 - old_slot];
        final_norm = true_squared >= 0.0 ? std::sqrt(true_squared) : true_squared;
        converged = std::isfinite(true_squared) && true_squared >= 0.0 &&
                      true_squared <= tolerance_squared;
        if (converged || !std::isfinite(true_squared) || true_squared < 0.0)
        { return; }
        // Recurrence drift does not enlarge the solve budget or tolerance.
        // Restart from the certified residual, retaining its device scalar.
        direction_ = z_;
        old_slot = 1 - old_slot;
        continue;
      }
      PrepareCGBeta<<<1, 1>>>(old_slot, c);
      CGDirection<<<blocks, threads>>>(height, z_.Read(), c, direction_.ReadWrite());
      old_slot = 1 - old_slot;
    }
  }

private:
  mutable mfem::Vector z_, direction_, control_;
};
// Specialized private coarse path: finalized rank-one CSR and a fixed MFEM
// Jacobi action. Snapshot that action, including damping/essential entries, at
// SetPreconditioner; borrow the current CSR device arrays afresh on every Mult.
class CooperativeCG final : public DeviceKrylovBase
{
public:
  CooperativeCG(MPI_Comm comm, int resident_blocks)
    : DeviceKrylovBase(comm), comm_(comm), resident_blocks_(resident_blocks)
  {
    partials_.SetSize(resident_blocks_);
    control_.SetSize(coop_control_size);
    partials_.UseDevice(true);
    control_.UseDevice(true);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    matrix_ = dynamic_cast<const mfem::SparseMatrix*>(&op);
    MFEM_VERIFY(matrix_ && matrix_->Finalized(),
                "Cooperative coarse CG requires a finalized SparseMatrix.");
    DeviceKrylovBase::SetOperator(op);
    SetVector(direction_);
    SetVector(inverse_diagonal_);
    blocks_ = std::min(resident_blocks_, (height - 1) / threads + 1);
  }

  void SetPreconditioner(mfem::Solver& pr) override
  {
    MFEM_VERIFY(oper && dynamic_cast<mfem::OperatorJacobiSmoother*>(&pr) &&
                  pr.Height() == height && pr.Width() == width,
                "Cooperative coarse CG requires a fixed MFEM Jacobi smoother.");
    mfem::IterativeSolver::SetPreconditioner(pr);
    mfem::Vector ones(height);
    ones.UseDevice(true);
    ones = 1.0;
    pr.Mult(ones, inverse_diagonal_);
    const double* inverse = inverse_diagonal_.HostRead();
    for (int i = 0; i < height; ++i)
    { MFEM_VERIFY(inverse[i] > 0.0 && std::isfinite(inverse[i]),
                  "Cooperative coarse CG requires a finite positive Jacobi inverse."); }
  }

  void Mult(const mfem::Vector& b, mfem::Vector& x) const override
  {
    Validate(b, x);
    MFEM_VERIFY(matrix_ && prec && !iterative_mode && max_iter <= 100 &&
                  !print_options.iterations,
                "Cooperative coarse CG requires zero initialization, Jacobi, "
                "at most 100 iterations and no per-iteration printing.");
    if (fallback_) { RunFallback(b, x); return; }
    CooperativeCGData data{height, max_iter, matrix_->ReadI(), matrix_->ReadJ(),
      matrix_->ReadData(), inverse_diagonal_.Read(), b.Read(), x.Write(),
      residual_.Write(), direction_.Write(), action_.Write(), partials_.Write(),
      control_.Write(), rel_tol * rel_tol, abs_tol * abs_tol};
    void* arguments[] = {&data};
    const cudaError_t launch = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(CooperativeCGKernel), dim3(blocks_),
      dim3(threads), arguments, 0, nullptr);
    if (launch == cudaErrorCooperativeLaunchTooLarge || launch == cudaErrorNotSupported ||
        launch == cudaErrorLaunchOutOfResources)
    {
      // These launch rejections execute no kernel. Persist the safe existing
      // host-polled path for this solver; do not conceal asynchronous failures.
      (void)cudaGetLastError();
      mfem::out << "[igr-cooperative-cg] fallback=host-polled reason="
                << cudaGetErrorString(launch) << '\n';
      fallback_ = std::make_unique<DeviceCG>(comm_);
      RunFallback(b, x);
      return;
    }
    CheckCUDA(launch);
    double result[4];
    ReadControl(result, data.control, 4); // the only host read in an admitted solve
    initial_norm = result[0] >= 0.0 ? std::sqrt(result[0]) : result[0];
    final_norm = result[1] >= 0.0 ? std::sqrt(result[1]) : result[1];
    final_iter = static_cast<int>(result[2]);
    converged = result[3] == 1.0;
  }

private:
  void RunFallback(const mfem::Vector& b, mfem::Vector& x) const
  {
    fallback_->SetOperator(*oper);
    fallback_->SetPreconditioner(*prec);
    fallback_->SetRelTol(rel_tol);
    fallback_->SetAbsTol(abs_tol);
    fallback_->SetMaxIter(max_iter);
    fallback_->SetPrintLevel(0);
    fallback_->iterative_mode = false;
    fallback_->Mult(b, x);
    initial_norm = fallback_->GetInitialNorm();
    final_norm = fallback_->GetFinalNorm();
    final_iter = fallback_->GetNumIterations();
    converged = fallback_->GetConverged();
  }
  MPI_Comm comm_;
  int resident_blocks_, blocks_ = 0;
  const mfem::SparseMatrix* matrix_ = nullptr;
  mutable mfem::Vector inverse_diagonal_, direction_, partials_, control_;
  mutable std::unique_ptr<DeviceCG> fallback_;
};

}

std::unique_ptr<mfem::IterativeSolver> MakeDeviceFGMRESSolver(
  MPI_Comm comm, int restart_dimension, KrylovResidualScale residual_scale)
{
  MFEM_VERIFY(restart_dimension >= 1 && restart_dimension <= 256,
              "Device FGMRES restart dimension must be 1..256.");
  return std::make_unique<DeviceFGMRES>(comm, restart_dimension, residual_scale);
}

std::unique_ptr<mfem::IterativeSolver> MakeDeviceCGSolver(MPI_Comm comm)
{ return std::make_unique<DeviceCG>(comm); }
std::unique_ptr<mfem::IterativeSolver> MakeCooperativeCGSolver(MPI_Comm comm)
{
  int ranks = 0;
  MPI_Comm_size(comm, &ranks);
  if (ranks != 1 || !mfem::Device::Allows(mfem::Backend::CUDA) ||
      mfem::Device::Allows(mfem::Backend::CEED_MASK | mfem::Backend::OCCA_MASK))
  { return nullptr; }
  int device = 0, cooperative = 0, multiprocessors = 0, active_blocks = 0;
  CheckCUDA(cudaGetDevice(&device));
  CheckCUDA(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  if (!cooperative) { return nullptr; }
  CheckCUDA(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
  CheckCUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, CooperativeCGKernel, threads, 0));
  if (multiprocessors <= 0 || active_blocks <= 0) { return nullptr; }
  return std::make_unique<CooperativeCG>(comm, multiprocessors * active_blocks);
}

}
}
