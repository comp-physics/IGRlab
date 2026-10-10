#include "igrlab/environment.hpp"
#include "IGR_dense_inverse.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

namespace igrlab
{
namespace igr_internal
{
namespace
{
void CheckCuda(cudaError_t status)
{
  MFEM_VERIFY(status == cudaSuccess, "Dense inverse CUDA: "
                << cudaGetErrorString(status));
}

void CheckBlas(cublasStatus_t status)
{
  MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS,
              "Dense inverse cuBLAS error " << static_cast<int>(status));
}

void CheckSolver(cusolverStatus_t status)
{
  MFEM_VERIFY(status == CUSOLVER_STATUS_SUCCESS,
              "Dense inverse cuSolver error " << static_cast<int>(status));
}

__global__ void SetIdentityDiagonal(double* matrix, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { matrix[i + n * i] = 1.0; }
}

std::size_t InvertWithCuSolver(mfem::DenseMatrix& inverse)
{
  // These buffers exist only during setup and are freed before the retained
  // inverse is uploaded. Report all explicit device setup storage, including LU.
  struct Buffers
  {
    cusolverDnHandle_t handle = nullptr;
    double* lu = nullptr;
    double* solution = nullptr;
    double* workspace = nullptr;
    int* pivots = nullptr;
    int* info = nullptr;
    ~Buffers()
    {
      if (handle) { cusolverDnDestroy(handle); }
      cudaFree(lu);
      cudaFree(solution);
      cudaFree(workspace);
      cudaFree(pivots);
      cudaFree(info);
    }
  } buffers;
  const int n = inverse.Height();
  const std::size_t matrix_bytes = std::size_t(n) * n * sizeof(double);
  CheckSolver(cusolverDnCreate(&buffers.handle));
  CheckSolver(cusolverDnSetStream(buffers.handle, nullptr));
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&buffers.lu), matrix_bytes));
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&buffers.solution), matrix_bytes));
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&buffers.pivots), n * sizeof(int)));
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&buffers.info), sizeof(int)));
  CheckCuda(cudaMemcpy(buffers.lu, inverse.HostRead(), matrix_bytes,
                        cudaMemcpyHostToDevice));
  int workspace_size = 0;
  CheckSolver(cusolverDnDgetrf_bufferSize(buffers.handle, n, n,
                                         buffers.lu, n, &workspace_size));
  MFEM_VERIFY(workspace_size >= 0, "Invalid cuSolver LU workspace size.");
  const std::size_t workspace_bytes = std::size_t(workspace_size) * sizeof(double);
  if (workspace_bytes)
  { CheckCuda(cudaMalloc(reinterpret_cast<void**>(&buffers.workspace), workspace_bytes)); }
  CheckCuda(cudaMemset(buffers.solution, 0, matrix_bytes));
  SetIdentityDiagonal<<<(n + 255) / 256, 256>>>(buffers.solution, n);
  CheckCuda(cudaGetLastError());
  CheckSolver(cusolverDnDgetrf(buffers.handle, n, n, buffers.lu, n,
                              buffers.workspace, buffers.pivots, buffers.info));
  int info = 0;
  CheckCuda(cudaMemcpy(&info, buffers.info, sizeof(int), cudaMemcpyDeviceToHost));
  MFEM_VERIFY(info == 0, "cuSolver pivoted LU failed, info=" << info);
  CheckSolver(cusolverDnDgetrs(buffers.handle, CUBLAS_OP_N, n, n,
                              buffers.lu, n, buffers.pivots,
                              buffers.solution, n, buffers.info));
  CheckCuda(cudaMemcpy(&info, buffers.info, sizeof(int), cudaMemcpyDeviceToHost));
  MFEM_VERIFY(info == 0, "cuSolver identity solve failed, info=" << info);
  CheckCuda(cudaMemcpy(inverse.HostWrite(), buffers.solution, matrix_bytes,
                        cudaMemcpyDeviceToHost));
  return 2 * matrix_bytes + workspace_bytes + std::size_t(n + 1) * sizeof(int);
}

__global__ void PackRhs(const double* input, float* packed,
                        int n, int padded, int columns)
{
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry < padded * columns)
  {
    const int row = entry % padded;
    packed[entry] = row < n ? static_cast<float>(input[row]) : 0.0f;
  }
}

__global__ void UnpackResult(const float* input, double* output, int n)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) { output[row] = static_cast<double>(input[row]); }
}

class DenseInversePreconditioner final : public mfem::Solver
{
public:
  DenseInversePreconditioner(const mfem::Operator& op,
                            const std::string& precision, MPI_Comm comm)
    : mfem::Solver(op.Height()), precision_(precision),
      padded_(precision == "tf32" ? ((op.Height() + 15) / 16) * 16
                                  : op.Height()),
      columns_(precision == "tf32" ? 16 : 1)
  {
    static_assert(std::is_same<mfem::real_t, double>::value,
                  "Dense inverse experiment requires FP64 MFEM.");
    const auto start = std::chrono::steady_clock::now();
    const char* setup_env = ::igrlab::GetEnv("DG_IGR_IGR_DENSE_SETUP");
    const std::string setup = setup_env && setup_env[0] ? setup_env : "host";
    MFEM_VERIFY(setup == "host" || setup == "cusolver",
                "DG_IGR_IGR_DENSE_SETUP must be host|cusolver or unset.");
    int ranks = 0;
    MPI_Comm_size(comm, &ranks);
    MFEM_VERIFY(ranks == 1 && mfem::Device::Allows(mfem::Backend::CUDA),
                "Dense inverse experiment requires one MPI rank and CUDA.");
    const int maximum_size = setup == "cusolver" ? 4096 : 2048;
    MFEM_VERIFY(height > 0 && height <= maximum_size && op.Width() == height,
                "Dense inverse experiment requires a square system, n<="
                << maximum_size << " for setup=" << setup << '.');
    MFEM_VERIFY(precision == "fp64" || precision == "fp32" ||
                  precision == "tf32", "Invalid dense inverse precision.");

    mfem::DenseMatrix inverse(height);
    mfem::Vector basis(height), column(height);
    basis.UseDevice(true);
    column.UseDevice(true);
    for (int j = 0; j < height; ++j)
    {
      double* values = basis.HostWrite();
      std::fill(values, values + height, 0.0);
      values[j] = 1.0;
      op.Mult(basis, column);
      const double* result = column.HostRead();
      for (int i = 0; i < height; ++i)
      {
        MFEM_VERIFY(std::isfinite(result[i]),
                    "Nonfinite sampled dense inverse operator.");
        inverse(i, j) = result[i];
      }
    }
    double matrix_max = 0.0, symmetry_error = 0.0;
    for (int j = 0; j < height; ++j)
    {
      for (int i = 0; i < height; ++i)
      {
        matrix_max = std::max(matrix_max, std::abs(inverse(i, j)));
        symmetry_error = std::max(symmetry_error,
                                  std::abs(inverse(i, j) - inverse(j, i)));
      }
    }
    const auto sampled = std::chrono::steady_clock::now();
    std::size_t setup_device_bytes = 0;
    // Both engines use pivoted FP64 algebra. Preserve the sampled entries;
    // do not impose symmetry or alter the stage equation.
    if (setup == "cusolver") { setup_device_bytes = InvertWithCuSolver(inverse); }
    else { inverse.Invert(); }
    const auto inverted = std::chrono::steady_clock::now();
    for (int entry = 0; entry < inverse.TotalSize(); ++entry)
    {
      MFEM_VERIFY(std::isfinite(inverse.Data()[entry]),
                  "Nonfinite FP64 dense inverse.");
    }
    mfem::Vector rhs(height), solution(height), residual(height);
    double inverse_residual = 0.0;
    for (int probe = 0; probe < 3; ++probe)
    {
      rhs.HostWrite();
      for (int i = 0; i < height; ++i)
      {
        rhs(i) = std::sin((i + 1) * (0.17 + 0.11 * probe));
      }
      solution.HostWrite();
      inverse.Mult(rhs, solution);
      op.Mult(solution, residual);
      residual -= rhs;
      const double relative = residual.Norml2() / rhs.Norml2();
      MFEM_VERIFY(std::isfinite(relative), "Nonfinite inverse probe residual.");
      inverse_residual = std::max(inverse_residual, relative);
    }
    MFEM_VERIFY(std::isfinite(inverse_residual) && inverse_residual < 1.0e-7,
                "Dense inverse FP64 setup residual failed: " << inverse_residual);
    const auto validated = std::chrono::steady_clock::now();
    CheckBlas(cublasCreate(&handle_));
    CheckBlas(cublasSetStream(handle_, nullptr));
    const std::size_t matrix_entries = std::size_t(padded_) * padded_;
    std::size_t device_bytes = 0;
    if (precision_ == "fp64")
    {
      device_bytes = matrix_entries * sizeof(double);
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&inverse64_), device_bytes));
      CheckCuda(cudaMemcpy(inverse64_, inverse.HostRead(), device_bytes,
                            cudaMemcpyHostToDevice));
    }
    else
    {
      std::vector<float> packed(matrix_entries, 0.0f);
      for (int j = 0; j < height; ++j)
      {
        for (int i = 0; i < height; ++i)
        {
          const float value = static_cast<float>(inverse(i, j));
          MFEM_VERIFY(std::isfinite(value), "Dense inverse FP32 overflow.");
          packed[i + std::size_t(padded_) * j] = value;
        }
      }
      const std::size_t panel_bytes = std::size_t(padded_) * columns_ * sizeof(float);
      device_bytes = matrix_entries * sizeof(float) + 2 * panel_bytes;
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&inverse32_),
                             matrix_entries * sizeof(float)));
      CheckCuda(cudaMemcpy(inverse32_, packed.data(), matrix_entries * sizeof(float),
                            cudaMemcpyHostToDevice));
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&rhs32_), panel_bytes));
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&result32_), panel_bytes));
    }
    CheckCuda(cudaDeviceSynchronize());
    const auto finished = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(
      finished - start).count();
    std::cout << "[igr-dense-inverse] n=" << height
              << " precision=" << precision_ << " padded=" << padded_
              << " rhs_columns=" << columns_ << " device_bytes=" << device_bytes
              << " setup_engine=" << setup
              << " setup_explicit_device_bytes=" << setup_device_bytes
              << " sampling_seconds=" << std::chrono::duration<double>(sampled - start).count()
              << " inversion_seconds=" << std::chrono::duration<double>(inverted - sampled).count()
              << " validation_seconds=" << std::chrono::duration<double>(validated - inverted).count()
              << " conversion_upload_seconds=" << std::chrono::duration<double>(finished - validated).count()
              << " setup_seconds=" << elapsed
              << " fp64_inverse_probe_rel=" << inverse_residual
              << " sampled_symmetry_rel=" << symmetry_error / matrix_max
              << " retained=first-operator-preconditioner" << std::endl;
  }

  ~DenseInversePreconditioner() override
  {
    if (handle_) { cublasDestroy(handle_); }
    cudaFree(inverse64_);
    cudaFree(inverse32_);
    cudaFree(rhs32_);
    cudaFree(result32_);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == height && op.Width() == width,
                "Cached dense inverse cannot change system dimensions.");
  }

  mfem::MemoryClass GetMemoryClass() const override
  { return mfem::MemoryClass::DEVICE; }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  { Apply(x, y, CUBLAS_OP_N); }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  { Apply(x, y, CUBLAS_OP_T); }

private:
  void Apply(const mfem::Vector& x, mfem::Vector& y, cublasOperation_t transpose) const
  {
    MFEM_VERIFY(x.Size() == height, "Dense inverse input size mismatch.");
    y.SetSize(height);
    y.UseDevice(true);
    const double* input = x.Read();
    double* output = y.Write();
    if (precision_ == "fp64")
    {
      const double one = 1.0, zero = 0.0;
      CheckBlas(cublasDgemv(handle_, transpose, height, height, &one,
                            inverse64_, height, input, 1, &zero, output, 1));
      return;
    }
    PackRhs<<<(padded_ * columns_ + 255) / 256, 256>>>(
      input, rhs32_, height, padded_, columns_);
    CheckCuda(cudaGetLastError());
    const float one = 1.0f, zero = 0.0f;
    if (precision_ == "fp32")
    {
      CheckBlas(cublasSgemv(handle_, transpose, height, height, &one,
                            inverse32_, height, rhs32_, 1, &zero, result32_, 1));
    }
    else
    {
      CheckBlas(cublasGemmEx(handle_, transpose, CUBLAS_OP_N,
        padded_, columns_, padded_, &one, inverse32_, CUDA_R_32F, padded_,
        rhs32_, CUDA_R_32F, padded_, &zero, result32_, CUDA_R_32F, padded_,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    UnpackResult<<<(height + 255) / 256, 256>>>(result32_, output, height);
    CheckCuda(cudaGetLastError());
  }

  std::string precision_;
  int padded_, columns_;
  cublasHandle_t handle_ = nullptr;
  double* inverse64_ = nullptr;
  float* inverse32_ = nullptr;
  float* rhs32_ = nullptr;
  float* result32_ = nullptr;
};
}

std::unique_ptr<mfem::Solver> MakeDenseInversePreconditioner(
  const mfem::Operator& op, const std::string& precision, MPI_Comm comm)
{
  return std::make_unique<DenseInversePreconditioner>(op, precision, comm);
}
}
}
