#include "igrlab/compat.hpp"
#include "igrlab/environment.hpp"
#include "IGR_local_tensor_jacobi.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace igrlab
{
namespace igr_internal
{
namespace
{
constexpr int max_axis = 8;
constexpr int max_dofs = max_axis * max_axis;
constexpr int max_classes = 512;
constexpr double admission_tolerance = 1.0e-11;

enum class TensorExecution
{
  SimtFP64, SimtFP32, SimtDenseFP32, FactorTF32, FactorFP64, DenseTF32, DenseFP64
};

enum class TensorReference { Exact, Projected };

TensorReference ReadTensorReference()
{
  const char* value = ::igrlab::GetEnv("IGRLAB_IGR_LOCAL_TENSOR_REFERENCE");
  if (!value) { value = ::igrlab::GetEnv("DG_IGR_IGR_LOCAL_TENSOR_REFERENCE"); }
  if (!value) { value = ::igrlab::GetEnv("DG_IGR_LOCAL_TENSOR_REFERENCE"); }
  if (!value || !*value || std::strcmp(value, "exact") == 0)
  { return TensorReference::Exact; }
  if (std::strcmp(value, "projected") == 0) { return TensorReference::Projected; }
  MFEM_ABORT("IGRLAB_IGR_LOCAL_TENSOR_REFERENCE must be exact|projected; got " << value);
  return TensorReference::Exact;
}

const char* ReferenceName(TensorReference reference)
{ return reference == TensorReference::Exact ? "exact" : "projected"; }

TensorExecution ReadTensorExecution()
{
  // The requested canonical name contains IGR_ while its legacy spelling
  // does not. Resolve it explicitly before GetEnv's usual prefix aliasing.
  const char* value = ::igrlab::GetEnv("IGRLAB_IGR_LOCAL_TENSOR_EXECUTION");
  if (!value) { value = ::igrlab::GetEnv("DG_IGR_LOCAL_TENSOR_EXECUTION"); }
  if (!value || !*value || std::strcmp(value, "simt-fp64") == 0)
  { return TensorExecution::SimtFP64; }
  if (std::strcmp(value, "simt-fp32") == 0) { return TensorExecution::SimtFP32; }
  if (std::strcmp(value, "simt-dense-fp32") == 0) { return TensorExecution::SimtDenseFP32; }
  if (std::strcmp(value, "wmma-factor-tf32") == 0) { return TensorExecution::FactorTF32; }
  if (std::strcmp(value, "wmma-factor-fp64") == 0) { return TensorExecution::FactorFP64; }
  if (std::strcmp(value, "wmma-dense-tf32") == 0) { return TensorExecution::DenseTF32; }
  if (std::strcmp(value, "wmma-dense-fp64") == 0) { return TensorExecution::DenseFP64; }
  MFEM_ABORT("IGRLAB_IGR_LOCAL_TENSOR_EXECUTION must be simt-fp64|simt-fp32|"
             "simt-dense-fp32|wmma-factor-tf32|wmma-factor-fp64|wmma-dense-tf32|wmma-dense-fp64; got " << value);
  return TensorExecution::SimtFP64;
}

const char* ExecutionName(TensorExecution execution)
{
  switch (execution)
  {
    case TensorExecution::SimtFP64: return "simt-fp64";
    case TensorExecution::SimtFP32: return "simt-fp32";
    case TensorExecution::SimtDenseFP32: return "simt-dense-fp32";
    case TensorExecution::FactorTF32: return "wmma-factor-tf32";
    case TensorExecution::FactorFP64: return "wmma-factor-fp64";
    case TensorExecution::DenseTF32: return "wmma-dense-tf32";
    case TensorExecution::DenseFP64: return "wmma-dense-fp64";
  }
  return "invalid";
}

bool UsesTensorCores(TensorExecution execution)
{
  return execution == TensorExecution::FactorTF32 ||
         execution == TensorExecution::FactorFP64 ||
         execution == TensorExecution::DenseTF32 || execution == TensorExecution::DenseFP64;
}

bool UsesDenseClasses(TensorExecution execution)
{
  return execution == TensorExecution::SimtDenseFP32 ||
         execution == TensorExecution::DenseTF32 || execution == TensorExecution::DenseFP64;
}

const char* PrecisionName(TensorExecution execution)
{
  if (execution == TensorExecution::FactorTF32 || execution == TensorExecution::DenseTF32)
  { return "tf32"; }
  return execution == TensorExecution::SimtFP32 || execution == TensorExecution::SimtDenseFP32
           ? "fp32" : "fp64";
}

__global__ void RefreshTensorMass(int entries, const double* geometry,
                                 const double* invrho, double* inverse_sqrt_mass)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < entries) { inverse_sqrt_mass[i] = 1.0 / sqrt(geometry[i] * invrho[i]); }
}

// One small CTA per element. All four contractions, diagonal division, native
// gather/scatter and both current-mass factors stay in this single FP64 kernel.
__global__ void ApplyTensorInverse(int axis, const int* map, const int* classes,
                                   const double* factors, const double* element_diagonals,
                                   const double* scaling,
                                   const double* input, double* output)
{
  const int e = blockIdx.x, t = threadIdx.x, dof = axis * axis;
  const int x = t % axis, y = t / axis;
  const double* qx = factors + std::size_t(classes[e]) * (element_diagonals ? 2 : 3) * dof;
  const double* qy = qx + dof;
  const double* denominator = element_diagonals ? element_diagonals + std::size_t(e) * dof : qy + dof;
  __shared__ double u[max_dofs], v[max_dofs];
  if (t < dof) { u[t] = scaling[e * dof + t] * input[map[e * dof + t]]; }
  __syncthreads();
  if (t < dof)
  {
    double value = 0.0;
    for (int i = 0; i < axis; ++i) { value += qx[i + axis * x] * u[i + axis * y]; }
    v[t] = value;
  }
  __syncthreads();
  if (t < dof)
  {
    double value = 0.0;
    for (int j = 0; j < axis; ++j) { value += qy[j + axis * y] * v[x + axis * j]; }
    u[t] = value * denominator[t];
  }
  __syncthreads();
  if (t < dof)
  {
    double value = 0.0;
    for (int i = 0; i < axis; ++i) { value += qx[x + axis * i] * u[i + axis * y]; }
    v[t] = value;
  }
  __syncthreads();
  if (t < dof)
  {
    double value = 0.0;
    for (int j = 0; j < axis; ++j) { value += qy[y + axis * j] * v[x + axis * j]; }
    output[map[e * dof + t]] = scaling[e * dof + t] * value;
  }
}

#include "IGR_local_tensor_jacobi_kernels.cuh"

const mfem::QuadratureSpace& CheckMassQF(const mfem::QuadratureFunction& qf,
                                       mfem::ParFiniteElementSpace& space)
{
  const auto* qs = dynamic_cast<const mfem::QuadratureSpace*>(qf.GetSpace());
  MFEM_VERIFY(qs && qs->GetMesh() == space.GetMesh() && qs->GetNE() == space.GetNE() &&
                qf.GetVDim() == 1 && qf.Size() == qs->GetSize(),
              "Local tensor Jacobi requires a scalar volume invrho QF on this mesh.");
  return *qs;
}

// A dependency-free FP64 eigensolver for the admitted one-dimensional axes
// (at most 8 by 8). Eigenvectors are column-major, as used by the CUDA kernel.
void SymmetricEigen(int n, const std::vector<double>& input,
                    std::vector<double>& eigenvalues, std::vector<double>& vectors)
{
  std::vector<double> a(input);
  vectors.assign(n * n, 0.0);
  for (int i = 0; i < n; ++i) { vectors[i + n * i] = 1.0; }
  double norm = 0.0;
  for (double value : a) { norm = std::max(norm, std::abs(value)); }
  const double stop = 8.0 * std::numeric_limits<double>::epsilon() * norm;
  bool converged = n == 1;
  for (int iteration = 0; iteration < 128 * n * n && !converged; ++iteration)
  {
    int p = 0, q = 0;
    double off = 0.0;
    for (int j = 1; j < n; ++j)
      for (int i = 0; i < j; ++i)
        if (std::abs(a[i + n * j]) > off)
        { p = i; q = j; off = std::abs(a[i + n * j]); }
    if (off <= stop) { converged = true; break; }
    const double apq = a[p + n * q];
    const double tau = (a[q + n * q] - a[p + n * p]) / (2.0 * apq);
    const double tangent = std::copysign(1.0 / (std::abs(tau) + std::hypot(1.0, tau)), tau);
    const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
    const double sine = tangent * cosine;
    a[p + n * p] -= tangent * apq;
    a[q + n * q] += tangent * apq;
    a[p + n * q] = a[q + n * p] = 0.0;
    for (int k = 0; k < n; ++k)
    {
      if (k != p && k != q)
      {
        const double akp = a[k + n * p], akq = a[k + n * q];
        a[k + n * p] = a[p + n * k] = cosine * akp - sine * akq;
        a[k + n * q] = a[q + n * k] = sine * akp + cosine * akq;
      }
      const double vkp = vectors[k + n * p], vkq = vectors[k + n * q];
      vectors[k + n * p] = cosine * vkp - sine * vkq;
      vectors[k + n * q] = sine * vkp + cosine * vkq;
    }
  }
  MFEM_VERIFY(converged, "Local tensor Jacobi small symmetric eigensolve did not converge.");
  eigenvalues.resize(n);
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int i, int j)
  { return a[i + n * i] < a[j + n * j]; });
  const auto unsorted = vectors;
  for (int j = 0; j < n; ++j)
  {
    eigenvalues[j] = a[order[j] + n * order[j]];
    for (int i = 0; i < n; ++i) { vectors[i + n * j] = unsorted[i + n * order[j]]; }
  }
  double error = 0.0, orthogonal_error = 0.0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
    {
      long double reconstructed = 0.0L, orthogonal = 0.0L;
      for (int k = 0; k < n; ++k)
      {
        reconstructed += static_cast<long double>(vectors[i + n * k]) *
                         eigenvalues[k] * vectors[j + n * k];
        orthogonal += static_cast<long double>(vectors[k + n * i]) * vectors[k + n * j];
      }
      error = std::max(error, std::abs(double(reconstructed) - input[i + n * j]));
      orthogonal_error = std::max(orthogonal_error, std::abs(double(orthogonal) - double(i == j)));
    }
  MFEM_VERIFY(error <= 1.0e-12 * std::max(norm, std::numeric_limits<double>::min()) &&
                orthogonal_error < 1.0e-12,
              "Local tensor Jacobi small-axis eigen reconstruction failed.");
}

struct TensorClass
{
  std::vector<double> x, y, packed;
};

std::uint64_t AxisBucket(const std::vector<double>& x, const std::vector<double>& y,
                         double norm)
{
  // A deliberately coarse quantization groups roundoff-equivalent Cartesian
  // cells. A bucket match is never admission: every candidate is subsequently
  // checked against the complete normalized matrix at admission_tolerance.
  // Near a bucket boundary equivalent matrices can create an extra class, still
  // bounded by max_classes; no search grows with the number of elements.
  int exponent = 0;
  const double mantissa = std::frexp(norm, &exponent);
  std::uint64_t hash = 1469598103934665603ULL;
  const auto append = [&](long long value)
  {
    hash ^= static_cast<std::uint64_t>(value);
    hash *= 1099511628211ULL;
  };
  append(exponent); append(std::llround(1.0e7 * mantissa));
  for (double value : x) { append(std::llround(1.0e7 * value / norm)); }
  for (double value : y) { append(std::llround(1.0e7 * value / norm)); }
  return hash;
}

// H(row_x,row_y;col_x,col_y) = X(row_x,col_x) delta_y +
// Y(row_y,col_y) delta_x. The trace is split equally to fix the scalar gauge.
void ExtractAxes(int axis, const std::vector<double>& h,
                  std::vector<double>& x, std::vector<double>& y)
{
  const int dof = axis * axis;
  x.assign(dof, 0.0); y.assign(dof, 0.0);
  double mean = 0.0;
  for (int i = 0; i < dof; ++i) { mean += h[i + dof * i] / dof; }
  for (int j = 0; j < axis; ++j)
    for (int i = 0; i < axis; ++i)
    {
      for (int k = 0; k < axis; ++k)
      {
        x[i + axis * j] += h[i + axis * k + dof * (j + axis * k)] / axis;
        y[i + axis * j] += h[k + axis * i + dof * (k + axis * j)] / axis;
      }
      if (i == j) { x[i + axis * j] -= mean / 2.0; y[i + axis * j] -= mean / 2.0; }
    }
}

double ReconstructionError(int axis, const std::vector<double>& h,
                            const std::vector<double>& x, const std::vector<double>& y)
{
  const int dof = axis * axis;
  double error = 0.0;
  for (int col = 0; col < dof; ++col)
    for (int row = 0; row < dof; ++row)
    {
      const double expected = (row / axis == col / axis ? x[row % axis + axis * (col % axis)] : 0.0) +
                              (row % axis == col % axis ? y[row / axis + axis * (col / axis)] : 0.0);
      error = std::max(error, std::abs(h[row + dof * col] - expected));
    }
  return error;
}

TensorClass FactorAxes(int axis, std::vector<double> x, std::vector<double> y)
{
  TensorClass result;
  result.x = std::move(x); result.y = std::move(y);
  std::vector<double> lx, ly, qx, qy;
  SymmetricEigen(axis, result.x, lx, qx);
  SymmetricEigen(axis, result.y, ly, qy);
  const double minimum = lx.front() + ly.front();
  const double maximum = lx.back() + ly.back();
  MFEM_VERIFY(std::isfinite(minimum) && std::isfinite(maximum) && minimum >
                512.0 * std::numeric_limits<double>::epsilon() * maximum,
              "Local tensor Jacobi requires a positive, resolved Kronecker-sum spectrum.");
  result.packed = qx;
  result.packed.insert(result.packed.end(), qy.begin(), qy.end());
  for (int j = 0; j < axis; ++j)
    for (int i = 0; i < axis; ++i) { result.packed.push_back(1.0 / (lx[i] + ly[j])); }
  return result;
}

TensorClass FactorProjectedAxes(int axis, std::vector<double> x, std::vector<double> y)
{
  TensorClass result;
  result.x = std::move(x); result.y = std::move(y);
  std::vector<double> values, qx, qy;
  SymmetricEigen(axis, result.x, values, qx);
  SymmetricEigen(axis, result.y, values, qy);
  // These partial-trace axes select a shared orthogonal basis only. Their
  // gauge-dependent eigensums are not inverted or required to be positive.
  result.packed = std::move(qx);
  result.packed.insert(result.packed.end(), qy.begin(), qy.end());
  return result;
}

// Form the inverse of one already-verified factor class, never an element's
// current block. Mirrored entries are identical, and the positive eigensums
// admitted by FactorAxes retain the reference operator's SPD qualification.
void StoreDenseClassInverse(int axis, const std::vector<double>& packed, double* inverse)
{
  const int dof = axis * axis;
  const double* qx = packed.data();
  const double* qy = qx + dof;
  const double* denominator = qy + dof;
  for (int row = 0; row < dof; ++row)
    for (int col = 0; col <= row; ++col)
    {
      double value = 0.0;
      for (int ky = 0; ky < axis; ++ky)
        for (int kx = 0; kx < axis; ++kx)
        {
          const double left = qx[row % axis + axis * kx] * qy[row / axis + axis * ky];
          const double right = qx[col % axis + axis * kx] * qy[col / axis + axis * ky];
          value += left * denominator[kx + axis * ky] * right;
        }
      MFEM_VERIFY(std::isfinite(value) && (row != col || value > 0.0),
                  "Local tensor Jacobi cached class inverse is not finite/positive on its diagonal.");
      inverse[row * dof + col] = inverse[col * dof + row] = value;
    }
}

class LocalTensorJacobi final : public ElementBlockJacobi
{
public:
  LocalTensorJacobi(mfem::ParFiniteElementSpace& space,
                    const mfem::QuadratureFunction& invrho)
    : ElementBlockJacobi(space.GetVSize()), space_(space),
      mass_space_(&CheckMassQF(invrho, space)),
      ne_(space.GetNE()), axis_(ne_ ? space.GetFE(0)->GetOrder() + 1 : 0),
      dof_(axis_ * axis_), space_sequence_(space.GetSequence()),
      mesh_sequence_(space.GetMesh() ? space.GetMesh()->GetSequence() : -1),
      nodes_sequence_(space.GetMesh() ? space.GetMesh()->GetNodesSequence() : -1),
      execution_(ReadTensorExecution()), reference_(ReadTensorReference())
  {
    static_assert(std::is_same<mfem::real_t, double>::value, "Local tensor Jacobi requires FP64 MFEM.");
    MFEM_VERIFY(dg_face::UsesOrdinaryCUDAForDGDiffusionFaces2D() && space.GetNRanks() == 1 &&
                  dynamic_cast<const mfem::L2_FECollection*>(space.FEColl()) &&
                  space.GetMesh() && space.GetMesh()->Dimension() == 2 &&
                  space.GetMesh()->SpaceDimension() == 2 && space.GetVDim() == 1 &&
                  space.Conforming() && ne_ > 0 && axis_ > 0 && axis_ <= max_axis &&
                  std::size_t(ne_) * dof_ == std::size_t(height) &&
                  space.GetTrueVSize() == height &&
                  mfem::IsIdentityProlongation(space.GetProlongationMatrix()) &&
                  mfem::IsIdentityProlongation(space.GetRestrictionTransposeOperator()),
                "Local tensor Jacobi requires rank-one native CUDA, uniform scalar tensor L2 quads, "
                "D1D <= 8 and an identity true/local map.");
    MFEM_VERIFY(std::size_t(ne_) * dof_ * dof_ <= std::size_t(std::numeric_limits<int>::max()) &&
                  (reference_ == TensorReference::Projected || invrho.Size() == height),
                "Local tensor Jacobi dense setup/QF indexing bound exceeded.");
    if (reference_ == TensorReference::Projected)
    {
      MFEM_VERIFY(axis_ == tensor_axis,
                  "Projected local tensor Jacobi requires p5/D1D=6.");
      MFEM_VERIFY(!UsesDenseClasses(execution_),
                  "Projected local tensor Jacobi has an element-specific spectral diagonal; "
                  "shared dense execution is unavailable for " << ExecutionName(execution_) <<
                  ". Select simt-fp64|simt-fp32|wmma-factor-tf32|wmma-factor-fp64. No fallback is provided.");
    }
    if (UsesTensorCores(execution_) || UsesDenseClasses(execution_))
    {
      MFEM_VERIFY(axis_ == tensor_axis,
                  "Grouped local tensor Jacobi backends require p5/D1D=6; selected " <<
                  ExecutionName(execution_) << ". No fallback is provided.");
    }
    if (UsesTensorCores(execution_))
    {
      int device = 0;
      cudaDeviceProp properties;
      MFEM_GPU_CHECK(cudaGetDevice(&device));
      MFEM_GPU_CHECK(cudaGetDeviceProperties(&properties, device));
      MFEM_VERIFY(properties.major >= 8,
                  "Local tensor Jacobi WMMA requires native SM80+ tensor cores; selected " <<
                  ExecutionName(execution_) << ". No fallback is provided.");
    }
    BuildLayoutAndMass(invrho);
  }

  void SetOperator(const mfem::Operator&) override
  { MFEM_ABORT("Initialize local tensor Jacobi through initial exact element blocks."); }

  mfem::MemoryClass GetMemoryClass() const override { return mfem::MemoryClass::DEVICE; }

  const char* BackendName() const override
  {
    return reference_ == TensorReference::Exact && execution_ == TensorExecution::SimtFP64 ?
      "tensor-element-density-fp64" : ExecutionName(execution_);
  }

  const char* ReferencePolicy() const override { return ReferenceName(reference_); }

  mfem::DenseTensor& PrepareVolumeBlocks(mfem::BilinearForm& form) override
  {
    CheckTopology();
    MFEM_VERIFY(!prepared_ && !ready_ && form.FESpace() == &space_ &&
                  form.GetFBFI()->Size() == 0 && form.GetBBFI()->Size() == 0 && form.GetBFBFI()->Size() == 0,
                "Local tensor Jacobi admits exactly one initial domain assembly, with separate exact face blocks.");
    ea_ = std::make_unique<mfem::EABilinearFormExtension>(&form);
    ea_->Assemble();
    ea_->GetElementMatrices(blocks_, mfem::ElementDofOrdering::NATIVE, false);
    blocks_.GetMemory().UseDevice(true);
    MFEM_VERIFY(blocks_.SizeI() == dof_ && blocks_.SizeJ() == dof_ && blocks_.SizeK() == ne_,
                "Unexpected local tensor Jacobi initial EA layout.");
    prepared_ = true;
    return blocks_;
  }

  void FactorBlocks() override
  {
    CheckTopology();
    MFEM_VERIFY(prepared_ && !ready_, "Prepare initial exact blocks before local tensor factorization.");
    if (reference_ == TensorReference::Projected) { FactorProjectedBlocks(); return; }
    const double* blocks = blocks_.HostRead();
    const double* scale = inverse_sqrt_mass_.HostRead();
    const int* native = lex_to_native_.HostRead();
    element_classes_.SetSize(ne_);
    int* ids = element_classes_.HostWrite();
    std::vector<TensorClass> classes;
    std::unordered_map<std::uint64_t, std::vector<int>> buckets;
    std::vector<double> h(dof_ * dof_), x, y;
    double maximum_reconstruction = 0.0;
    for (int e = 0; e < ne_; ++e)
    {
      double norm = 0.0, asymmetry = 0.0;
      for (int j = 0; j < dof_; ++j)
        for (int i = 0; i < dof_; ++i)
        {
          const double a = blocks[std::size_t(e) * dof_ * dof_ + native[i] + dof_ * native[j]];
          const double value = scale[e * dof_ + i] * a * scale[e * dof_ + j];
          MFEM_VERIFY(std::isfinite(value), "Nonfinite initial local tensor Jacobi block at element " << e);
          h[i + dof_ * j] = value;
          norm = std::max(norm, std::abs(value));
        }
      MFEM_VERIFY(norm > 0.0, "Zero initial local tensor Jacobi block at element " << e);
      for (int j = 0; j < dof_; ++j)
        for (int i = 0; i < j; ++i)
        {
          asymmetry = std::max(asymmetry, std::abs(h[i + dof_ * j] - h[j + dof_ * i]));
          const double average = 0.5 * (h[i + dof_ * j] + h[j + dof_ * i]);
          h[i + dof_ * j] = h[j + dof_ * i] = average;
        }
      MFEM_VERIFY(asymmetry <= admission_tolerance * norm,
                  "Local tensor Jacobi requires symmetric initial SIPG blocks; element " << e);
      ExtractAxes(axis_, h, x, y);
      double error = ReconstructionError(axis_, h, x, y);
      const double allowance = admission_tolerance * norm - 0.5 * asymmetry;
      MFEM_VERIFY(error <= allowance,
                  "Initial mass-normalized block is not a Kronecker sum at element " << e <<
                  "; relative reconstruction=" << error / norm);
      auto& candidates = buckets[AxisBucket(x, y, norm)];
      int id = -1;
      for (int c : candidates)
      {
        bool possible = true;
        for (int i = 0; i < dof_ && possible; ++i)
        {
          possible = std::abs(x[i] - classes[c].x[i]) <= 0.25 * admission_tolerance * norm &&
                     std::abs(y[i] - classes[c].y[i]) <= 0.25 * admission_tolerance * norm;
        }
        if (!possible) { continue; }
        const double cached_error = ReconstructionError(axis_, h, classes[c].x, classes[c].y);
        if (cached_error <= allowance)
        { id = int(c); error = cached_error; break; }
      }
      if (id < 0)
      {
        MFEM_VERIFY(classes.size() < max_classes,
                    "Local tensor Jacobi exceeded its bounded 512 verified tensor factor classes.");
        id = int(classes.size());
        classes.push_back(FactorAxes(axis_, x, y));
        candidates.push_back(id);
      }
      ids[e] = id;
      maximum_reconstruction = std::max(maximum_reconstruction, (error + 0.5 * asymmetry) / norm);
    }
    factors_.SetSize(int(classes.size()) * 3 * dof_);
    double* packed = factors_.HostWrite();
    for (std::size_t c = 0; c < classes.size(); ++c)
    { std::copy(classes[c].packed.begin(), classes[c].packed.end(), packed + c * 3 * dof_); }
    if (UsesDenseClasses(execution_))
    {
      class_inverses_.SetSize(int(classes.size()) * dof_ * dof_);
      double* inverses = class_inverses_.HostWrite();
      for (std::size_t c = 0; c < classes.size(); ++c)
      { StoreDenseClassInverse(axis_, classes[c].packed, inverses + c * dof_ * dof_); }
      class_inverses_.UseDevice(true);
      (void)class_inverses_.Read();
    }
    if (UsesTensorCores(execution_) || UsesDenseClasses(execution_))
    { BuildClassGroups(ids, int(classes.size())); }
    CompleteSetup(int(classes.size()), maximum_reconstruction);
  }

  void RefreshMassScaling(const mfem::QuadratureFunction& invrho) override
  {
    CheckTopology();
    MFEM_VERIFY(ready_ && &CheckMassQF(invrho, space_) == mass_space_ &&
                  invrho.Size() == geometry_mass_.Size(),
                "Local tensor Jacobi mass layout changed; reconstruct the backend.");
    // invrho uses the established coefficient floor semantics. This update
    // intentionally adds no per-stage host copy, coefficient scan or reduction.
    if (reference_ == TensorReference::Projected)
    {
      RefreshProjectedTensorMass<<<ne_, 64>>>(mass_axis_, mass_basis_squared_.Read(),
        geometry_mass_.Read(), invrho.Read(), inverse_sqrt_mass_.Write());
    }
    else
    {
      RefreshTensorMass<<<1 + (height - 1) / 256, 256>>>(
        height, geometry_mass_.Read(), invrho.Read(), inverse_sqrt_mass_.Write());
    }
    MFEM_GPU_CHECK(cudaGetLastError());
  }

  void Mult(const mfem::Vector& rhs, mfem::Vector& result) const override
  {
    CheckTopology();
    MFEM_VERIFY(ready_ && rhs.Size() == height && &rhs != &result,
                "Local tensor Jacobi requires ready factors and distinct input/output vectors.");
    result.SetSize(height); result.UseDevice(true);
    const int* map = element_dofs_.Read();
    const double* scaling = inverse_sqrt_mass_.Read();
    const double* input = rhs.Read();
    const double* diagonal = reference_ == TensorReference::Projected ? element_diagonals_.Read() : nullptr;
    double* output = result.Write();
    if (execution_ == TensorExecution::SimtFP64 || execution_ == TensorExecution::SimtFP32)
    {
      int threads = 32;
      while (threads < dof_) { threads *= 2; }
      if (execution_ == TensorExecution::SimtFP64)
      {
        ApplyTensorInverse<<<ne_, threads>>>(axis_, map, element_classes_.Read(),
          factors_.Read(), diagonal, scaling, input, output);
      }
      else
      {
        ApplyTensorInverseFP32<<<ne_, threads>>>(axis_, map, element_classes_.Read(),
          factors_.Read(), diagonal, scaling, input, output);
      }
    }
    else
    {
      const int groups = grouped_classes_.Size();
      const int* elements = grouped_elements_.Read();
      const int* classes = grouped_classes_.Read();
      if (execution_ == TensorExecution::FactorTF32)
      {
        ApplyTensorFactorTF32<<<groups, 6 * 32>>>(map, elements, classes, factors_.Read(),
          diagonal, scaling, input, output);
      }
      else if (execution_ == TensorExecution::FactorFP64)
      {
        ApplyTensorFactorFP64<false><<<groups, 6 * 32>>>(map, elements, classes, factors_.Read(),
          diagonal, scaling, input, output);
      }
      else if (execution_ == TensorExecution::DenseTF32)
      {
        ApplyTensorDenseTF32<<<groups, 3 * 32>>>(map, elements, classes, class_inverses_.Read(),
          scaling, input, output);
      }
      else if (execution_ == TensorExecution::DenseFP64)
      {
        ApplyTensorDenseFP64<<<groups, 5 * 32>>>(map, elements, classes, class_inverses_.Read(),
          scaling, input, output);
      }
      else
      {
        ApplyTensorDenseFP32<<<groups, 3 * 32>>>(map, elements, classes, class_inverses_.Read(),
          scaling, input, output);
      }
    }
    MFEM_GPU_CHECK(cudaGetLastError());
  }

  bool TryAddMult(const mfem::Vector& rhs, mfem::real_t omega,
                  mfem::Vector& solution) const override
  {
    if (reference_ != TensorReference::Projected || execution_ != TensorExecution::FactorFP64 ||
        !ready_ || rhs.Size() != height || solution.Size() != height)
    { return false; }
    CheckTopology();
    const double* input = rhs.Read();
    const double* current = solution.Read();
    const auto a = reinterpret_cast<std::uintptr_t>(input);
    const auto b = reinterpret_cast<std::uintptr_t>(current);
    const auto bytes = std::size_t(height) * sizeof(double);
    if ((a < b ? b - a : a - b) < bytes) { return false; }
    // Setup proves that every native dof occurs in exactly one grouped real
    // element. Padding owns no row; no other group can write this solution.
    ApplyTensorFactorFP64<true><<<grouped_classes_.Size(), 6 * 32>>>(
      element_dofs_.Read(), grouped_elements_.Read(), grouped_classes_.Read(), factors_.Read(),
      element_diagonals_.Read(), inverse_sqrt_mass_.Read(), input, solution.ReadWrite(), omega);
    MFEM_GPU_CHECK(cudaGetLastError());
    if (!update_execution_reported_)
    {
      std::cout << "[igr-local-tensor-jacobi] event=execution path=fused-solution-update"
                << " reference=projected backend=wmma-factor-fp64 correction_rounding=fp64"
                << " elements=" << ne_ << " groups=" << grouped_classes_.Size() << std::endl;
      update_execution_reported_ = true;
    }
    return true;
  }

private:
  void CompleteSetup(int class_count, double maximum_reconstruction,
                     double maximum_offdiag = 0.0, double diagonal_minimum = 0.0,
                     double diagonal_maximum = 0.0)
  {
    // Exact EA, its host copy, and the native-only permutation are setup-only.
    // Runtime retains O(N) mass/maps/diagonals and at most 16 projected bases.
    ea_.reset();
    blocks_.Clear();
    lex_to_native_.DeleteAll();
    factors_.UseDevice(true);
    (void)element_classes_.Read(); (void)factors_.Read();
    ready_ = true; prepared_ = false;
    const std::size_t bytes = sizeof(double) * (std::size_t(geometry_mass_.Size()) +
      inverse_sqrt_mass_.Size() + factors_.Size() + class_inverses_.Size() +
      mass_basis_squared_.Size() + element_diagonals_.Size()) +
      sizeof(int) * (std::size_t(element_dofs_.Size()) + element_classes_.Size() +
                     grouped_elements_.Size() + grouped_classes_.Size());
    const int dense_classes = UsesDenseClasses(execution_) ? class_count : 0;
    const bool projected = reference_ == TensorReference::Projected;
    const bool grouped = grouped_classes_.Size() > 0;
    const auto& tensor = dynamic_cast<const mfem::TensorBasisElement&>(*space_.GetFE(0));
    std::cout << "[igr-local-tensor-jacobi] event=setup precision=" << PrecisionName(execution_)
              << " execution=" << (UsesTensorCores(execution_) ? "wmma" : "simt")
              << " backend=" << ExecutionName(execution_)
              << " tensor_core=" << int(UsesTensorCores(execution_))
              << " reference=" << ReferenceName(reference_)
              << " nodes=" << (tensor.GetBasisType() == mfem::BasisType::GaussLegendre ? "gl" : "gll")
              << " d1d=" << axis_ << " elements=" << ne_ << " factor_classes=" << class_count
              << " coefficient_policy=" << (projected ? "current-mass-diagonal" : "current-collocated-mass")
              << " initial_blocks=exact-sipg"
              << " geometry=" << (projected ? "fixed-q1q2-mapped" : "affine-cartesian")
              << " geometry_order_max=" << maximum_geometry_order_
              << " mass_rule=gl" << mass_axis_ << "x" << mass_axis_
              << " mass_quadrature_entries=" << geometry_mass_.Size();
    if (projected)
    {
      std::cout << " jacobian_validation=sampled"
                << " geometry_approximation=projected-initial-block"
                << " projected_offdiag_frobenius_relative_max=" << maximum_offdiag
                << " projected_diagonal_min=" << diagonal_minimum
                << " projected_diagonal_max=" << diagonal_maximum;
    }
    else { std::cout << " reconstruction_relative=" << maximum_reconstruction; }
    std::cout << " density_approximation=heterogeneous-current-mass"
              << " owned_bytes=" << bytes << " dense_after_setup=" << dense_classes
              << " class_dense_matrices=" << dense_classes
              << " class_dense_entries=" << class_inverses_.Size() << " dense_per_element=0"
              << " class_cache_bytes=" << sizeof(double) * (std::size_t(factors_.Size()) + class_inverses_.Size())
              << " element_diagonal_bytes=" << sizeof(double) * std::size_t(element_diagonals_.Size())
              << " grouping=" << (grouped ? (projected ? "boundary-mask" : "verified-class") : "element")
              << " group_elements=" << (grouped ? tensor_group_elements : 1)
              << " groups=" << (grouped ? grouped_classes_.Size() : ne_)
              << " group_padding=" << (grouped ? grouped_elements_.Size() - ne_ : 0)
              << " mass_precision=fp64 output_precision=fp64"
              << " arithmetic_spd=" << (execution_ == TensorExecution::SimtFP64 ||
                  execution_ == TensorExecution::FactorFP64 || execution_ == TensorExecution::DenseFP64 ?
                  "fp64-roundoff" : "rounding-perturbed")
              << " fallback=none" << std::endl;
  }

  void FactorProjectedBlocks()
  {
    const double* blocks = blocks_.HostRead();
    const double* scale = inverse_sqrt_mass_.HostRead();
    const int* native = lex_to_native_.HostRead();
    element_classes_.SetSize(ne_);
    int* ids = element_classes_.HostWrite();
    std::array<int, 16> counts{}, class_for_mask;
    class_for_mask.fill(-1);
    mfem::Array<int> edges, orientations;
    for (int e = 0; e < ne_; ++e)
    {
      space_.GetMesh()->GetElementEdges(e, edges, orientations);
      MFEM_VERIFY(edges.Size() == 4, "Projected local tensor Jacobi requires four local quad edges.");
      int mask = 0;
      for (int edge = 0; edge < 4; ++edge)
      {
        int first = -1, second = -1;
        space_.GetMesh()->GetFaceElements(edges[edge], &first, &second);
        MFEM_VERIFY(first == e || second == e, "Invalid projected local tensor edge adjacency.");
        // Rank one and conforming: a single adjacent element is a physical
        // boundary. Periodic faces have two sides, even when both are e.
        if (first < 0 || second < 0) { mask |= 1 << edge; }
      }
      ids[e] = mask;
      ++counts[mask];
    }
    int class_count = 0;
    for (int mask = 0; mask < 16; ++mask)
    { if (counts[mask]) { class_for_mask[mask] = class_count++; } }
    std::vector<std::vector<double>> averages(class_count, std::vector<double>(dof_ * dof_, 0.0));
    std::vector<double> h(dof_ * dof_), x, y;
    for (int e = 0; e < ne_; ++e)
    {
      const int mask = ids[e], id = class_for_mask[mask];
      ids[e] = id;
      double norm = 0.0, asymmetry = 0.0;
      for (int j = 0; j < dof_; ++j)
        for (int i = 0; i < dof_; ++i)
        {
          const double a = blocks[std::size_t(e) * dof_ * dof_ + native[i] + dof_ * native[j]];
          const double value = scale[e * dof_ + i] * a * scale[e * dof_ + j];
          MFEM_VERIFY(std::isfinite(value) && (i != j || value > 0.0),
                      "Projected local tensor Jacobi initial block is nonfinite/nonpositive on its diagonal; element " << e);
          h[i + dof_ * j] = value;
          norm = std::max(norm, std::abs(value));
        }
      MFEM_VERIFY(norm > 0.0, "Zero projected local tensor Jacobi initial block at element " << e);
      for (int j = 0; j < dof_; ++j)
        for (int i = 0; i < j; ++i)
        {
          asymmetry = std::max(asymmetry, std::abs(h[i + dof_ * j] - h[j + dof_ * i]));
          const double average = 0.5 * (h[i + dof_ * j] + h[j + dof_ * i]);
          h[i + dof_ * j] = h[j + dof_ * i] = average;
        }
      MFEM_VERIFY(asymmetry <= admission_tolerance * norm,
                  "Projected local tensor Jacobi requires symmetric initial SIPG blocks; element " << e);
      const double class_weight = 1.0 / counts[mask];
      for (int i = 0; i < dof_ * dof_; ++i) { averages[id][i] += class_weight * h[i]; }
    }
    factors_.SetSize(class_count * 2 * dof_);
    double* packed = factors_.HostWrite();
    for (int c = 0; c < class_count; ++c)
    {
      for (double value : averages[c])
      { MFEM_VERIFY(std::isfinite(value), "Nonfinite averaged projected local tensor reference class " << c); }
      ExtractAxes(axis_, averages[c], x, y);
      const auto basis = FactorProjectedAxes(axis_, x, y);
      std::copy(basis.packed.begin(), basis.packed.end(), packed + c * 2 * dof_);
    }
    factors_.UseDevice(true);
    element_diagonals_.SetSize(height); element_diagonals_.UseDevice(true);
    mfem::Vector relative_offdiag(ne_); relative_offdiag.UseDevice(true);
    ProjectTensorDiagonals<<<ne_, 128>>>(lex_to_native_.Read(), element_classes_.Read(),
      blocks_.Read(), inverse_sqrt_mass_.Read(), factors_.Read(), element_diagonals_.Write(),
      relative_offdiag.Write());
    MFEM_GPU_CHECK(cudaGetLastError());
    const double* reciprocals = element_diagonals_.HostRead();
    const double* errors = relative_offdiag.HostRead();
    double minimum = std::numeric_limits<double>::infinity(), maximum = 0.0, maximum_offdiag = 0.0;
    for (int e = 0; e < ne_; ++e)
    {
      double element_minimum = std::numeric_limits<double>::infinity(), element_maximum = 0.0;
      for (int i = 0; i < dof_; ++i)
      {
        const double reciprocal = reciprocals[e * dof_ + i];
        const double value = 1.0 / reciprocal;
        MFEM_VERIFY(std::isfinite(reciprocal) && reciprocal > 0.0 && std::isfinite(value) && value > 0.0,
                    "Projected local tensor Jacobi requires finite positive projected diagonals; element " << e);
        element_minimum = std::min(element_minimum, value);
        element_maximum = std::max(element_maximum, value);
      }
      MFEM_VERIFY(element_minimum > 512.0 * std::numeric_limits<double>::epsilon() * element_maximum,
                  "Projected local tensor Jacobi requires resolved positive projected diagonals; element " << e);
      MFEM_VERIFY(std::isfinite(errors[e]), "Nonfinite projected local tensor off-diagonal metric; element " << e);
      minimum = std::min(minimum, element_minimum); maximum = std::max(maximum, element_maximum);
      maximum_offdiag = std::max(maximum_offdiag, errors[e]);
    }
    if (UsesTensorCores(execution_)) { BuildClassGroups(ids, class_count); }
    CompleteSetup(class_count, 0.0, maximum_offdiag, minimum, maximum);
  }

  void BuildClassGroups(const int* ids, int class_count)
  {
    std::vector<std::vector<int>> members(class_count);
    for (int e = 0; e < ne_; ++e) { members[ids[e]].push_back(e); }
    int groups = 0;
    for (const auto& elements : members)
    { groups += (int(elements.size()) + tensor_group_elements - 1) / tensor_group_elements; }
    grouped_classes_.SetSize(groups);
    grouped_elements_.SetSize(groups * tensor_group_elements);
    int* classes = grouped_classes_.HostWrite();
    int* elements = grouped_elements_.HostWrite();
    std::fill(elements, elements + grouped_elements_.Size(), -1);
    int group = 0;
    for (int c = 0; c < class_count; ++c)
      for (std::size_t begin = 0; begin < members[c].size(); begin += tensor_group_elements)
      {
        classes[group] = c;
        const int count = int(std::min<std::size_t>(tensor_group_elements, members[c].size() - begin));
        std::copy_n(members[c].data() + begin, count, elements + group * tensor_group_elements);
        ++group;
      }
    (void)grouped_elements_.Read(); (void)grouped_classes_.Read();
  }

  double CheckProjectedJacobian(mfem::ElementTransformation& transformation,
                                const mfem::IntegrationPoint& ip, int element) const
  {
    transformation.SetIntPoint(&ip);
    const auto& jacobian = transformation.Jacobian();
    const double xx = jacobian(0, 0), xy = jacobian(0, 1);
    const double yx = jacobian(1, 0), yy = jacobian(1, 1);
    const double determinant = xx * yy - xy * yx;
    MFEM_VERIFY(std::isfinite(xx) && std::isfinite(xy) && std::isfinite(yx) && std::isfinite(yy) &&
                  std::isfinite(determinant) && determinant > 0.0,
                "Projected local tensor Jacobi requires finite positive sampled signed Jacobians; element " << element);
    const double weight = transformation.Weight();
    MFEM_VERIFY(std::isfinite(weight) && weight > 0.0,
                "Projected local tensor Jacobi requires positive sampled geometry weights; element " << element);
    return weight;
  }

  void BuildProjectedLayoutAndMass(const mfem::QuadratureFunction& invrho)
  {
    const auto& fe = *space_.GetFE(0);
    const auto* tensor = dynamic_cast<const mfem::NodalTensorFiniteElement*>(&fe);
    MFEM_VERIFY(typeid(fe) == typeid(mfem::L2_QuadrilateralElement) && tensor &&
                  (tensor->GetBasisType() == mfem::BasisType::GaussLegendre ||
                   tensor->GetBasisType() == mfem::BasisType::GaussLobatto) &&
                  fe.GetGeomType() == mfem::Geometry::SQUARE && fe.GetDof() == dof_ &&
                  fe.GetMapType() == mfem::FiniteElement::VALUE && fe.GetNodes().GetNPoints() == dof_,
                "Projected local tensor Jacobi requires builtin p5 GL/GLL nodal tensor quads.");
    const auto& lex = tensor->GetDofMap();
    MFEM_VERIFY(lex.Size() == 0 || lex.Size() == dof_, "Invalid projected tensor native permutation.");
    lex_to_native_.SetSize(dof_);
    int* native = lex_to_native_.HostWrite();
    std::vector<bool> local_seen(dof_, false), seen(height, false);
    mfem::IntegrationRules gl_rules(0, mfem::Quadrature1D::GaussLegendre);
    const mfem::real_t* nodes = mfem::poly1d.GetPoints(fe.GetOrder(), tensor->GetBasisType());
    for (int i = 0; i < dof_; ++i)
    {
      native[i] = lex.Size() ? lex[i] : i;
      MFEM_VERIFY(native[i] >= 0 && native[i] < dof_ && !local_seen[native[i]],
                  "Invalid unsigned projected tensor native permutation.");
      local_seen[native[i]] = true;
      const auto& node = fe.GetNodes().IntPoint(native[i]);
      MFEM_VERIFY(std::isfinite(node.x) && std::isfinite(node.y) &&
                    std::abs(node.x - nodes[i % axis_]) < 1.0e-13 &&
                    std::abs(node.y - nodes[i / axis_]) < 1.0e-13,
                  "Projected local tensor Jacobi requires actual lexicographic GL/GLL nodes.");
    }
    const auto& rule = mass_space_->GetIntRule(0);
    mass_axis_ = int(std::sqrt(double(rule.GetNPoints())));
    MFEM_VERIFY((mass_axis_ == 6 || mass_axis_ == 7) &&
                  mass_axis_ * mass_axis_ == rule.GetNPoints() &&
                  std::size_t(ne_) * rule.GetNPoints() == std::size_t(invrho.Size()),
                "Projected local tensor Jacobi requires uniform actual GL6x6 or GL7x7 mass quadrature.");
    const auto& mass_rule_1d = gl_rules.Get(mfem::Geometry::SEGMENT, 2 * mass_axis_ - 1);
    MFEM_VERIFY(mass_rule_1d.GetNPoints() == mass_axis_, "Unexpected projected mass quadrature axis.");
    mass_basis_squared_.SetSize(mass_axis_ * axis_);
    double* squared_basis = mass_basis_squared_.HostWrite();
    std::vector<double> basis(mass_axis_ * axis_);
    mfem::Vector values(axis_), shape(dof_);
    for (int q = 0; q < mass_axis_; ++q)
    {
      tensor->GetBasis1D().Eval(mass_rule_1d.IntPoint(q).x, values);
      for (int i = 0; i < axis_; ++i)
      {
        const double value = values[i];
        MFEM_VERIFY(std::isfinite(value), "Nonfinite projected mass basis value.");
        basis[q + mass_axis_ * i] = value;
        squared_basis[q + mass_axis_ * i] = value * value;
      }
    }
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const int qx = q % mass_axis_, qy = q / mass_axis_;
      const auto& ip = rule.IntPoint(q);
      const auto& ix = mass_rule_1d.IntPoint(qx);
      const auto& iy = mass_rule_1d.IntPoint(qy);
      MFEM_VERIFY(std::isfinite(ip.x) && std::isfinite(ip.y) && std::isfinite(ip.weight) &&
                    ip.weight > 0.0 && std::abs(ip.x - ix.x) < 1.0e-13 &&
                    std::abs(ip.y - iy.x) < 1.0e-13 &&
                    std::abs(ip.weight - ix.weight * iy.weight) < 1.0e-13 * ip.weight,
                  "Projected local tensor Jacobi requires lexicographic Gauss-Legendre mass quadrature.");
      fe.CalcShape(ip, shape);
      for (int i = 0; i < dof_; ++i)
      {
        const double expected = basis[qx + mass_axis_ * (i % axis_)] *
                                basis[qy + mass_axis_ * (i / axis_)];
        MFEM_VERIFY(std::abs(shape[native[i]] - expected) < 1.0e-12,
                    "Projected local tensor Jacobi mass shapes do not match the tensor basis.");
      }
    }
    element_dofs_.SetSize(height); geometry_mass_.SetSize(invrho.Size());
    inverse_sqrt_mass_.SetSize(height);
    int* map = element_dofs_.HostWrite();
    double* geometry = geometry_mass_.HostWrite();
    const double* coefficient = invrho.HostRead();
    mfem::Array<int> dofs;
    const auto& vertices = *mfem::Geometries.GetVertices(mfem::Geometry::SQUARE);
    for (int e = 0; e < ne_; ++e)
    {
      MFEM_VERIFY(space_.GetFE(e) == &fe && mass_space_->Offset(e) == e * rule.GetNPoints() &&
                    &mass_space_->GetIntRule(e) == &rule &&
                    space_.GetElementDofs(e, dofs) == nullptr && dofs.Size() == dof_,
                  "Projected local tensor Jacobi requires uniform FE/QF layout without dof transformations.");
      auto* transformation = space_.GetMesh()->GetElementTransformation(e);
      const auto* isoparametric = dynamic_cast<const mfem::IsoparametricTransformation*>(transformation);
      const int order = transformation->Order();
      const auto* geometry_fe = isoparametric ? isoparametric->GetFE() : nullptr;
      MFEM_VERIFY(geometry_fe && (order == 1 || order == 2) &&
                    geometry_fe->GetGeomType() == mfem::Geometry::SQUARE &&
                    geometry_fe->GetDof() == (order + 1) * (order + 1),
                  "Projected local tensor Jacobi admits fixed Q1/Q2 quadrilateral mappings; element " << e);
      MFEM_VERIFY(mass_axis_ >= axis_ + order - 1,
                  "Projected Q2 local tensor Jacobi requires the actual GL7x7 mass rule, not collocated GL6; element " << e);
      maximum_geometry_order_ = std::max(maximum_geometry_order_, order);
      // These samples reject invalid metrics where observed; they do not
      // certify positivity between the sampled points of a curved map.
      for (int i = 0; i < vertices.GetNPoints(); ++i)
      { CheckProjectedJacobian(*transformation, vertices.IntPoint(i), e); }
      const auto& geometry_nodes = geometry_fe->GetNodes();
      for (int i = 0; i < geometry_nodes.GetNPoints(); ++i)
      { CheckProjectedJacobian(*transformation, geometry_nodes.IntPoint(i), e); }
      for (int i = 0; i < dof_; ++i)
      {
        const int row = dofs[native[i]];
        MFEM_VERIFY(row >= 0 && row < height && !seen[row],
                    "Projected local tensor Jacobi requires disjoint unsigned scalar element dofs.");
        seen[row] = true; map[e * dof_ + i] = row;
        CheckProjectedJacobian(*transformation, fe.GetNodes().IntPoint(native[i]), e);
      }
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const auto& ip = rule.IntPoint(q);
        const double weight = ip.weight * CheckProjectedJacobian(*transformation, ip, e);
        const int entry = e * rule.GetNPoints() + q;
        const double mass = weight * coefficient[entry];
        MFEM_VERIFY(std::isfinite(weight) && weight > 0.0 && std::isfinite(mass) && mass > 0.0,
                    "Projected local tensor Jacobi requires positive finite initial quadrature mass; element " << e);
        geometry[entry] = weight;
      }
    }
    geometry_mass_.UseDevice(true); inverse_sqrt_mass_.UseDevice(true); mass_basis_squared_.UseDevice(true);
    (void)element_dofs_.Read();
    RefreshProjectedTensorMass<<<ne_, 64>>>(mass_axis_, mass_basis_squared_.Read(), geometry_mass_.Read(),
      invrho.Read(), inverse_sqrt_mass_.Write());
    MFEM_GPU_CHECK(cudaGetLastError());
    const double* initial_scale = inverse_sqrt_mass_.HostRead();
    for (int i = 0; i < height; ++i)
    { MFEM_VERIFY(std::isfinite(initial_scale[i]) && initial_scale[i] > 0.0,
                  "Projected local tensor Jacobi requires positive finite initial mass diagonals; entry " << i); }
  }

  void BuildLayoutAndMass(const mfem::QuadratureFunction& invrho)
  {
    if (reference_ == TensorReference::Projected) { BuildProjectedLayoutAndMass(invrho); return; }
    mass_axis_ = axis_;
    const auto& fe = *space_.GetFE(0);
    const auto* tensor = dynamic_cast<const mfem::NodalTensorFiniteElement*>(&fe);
    MFEM_VERIFY(tensor && tensor->GetBasisType() == mfem::BasisType::GaussLegendre &&
                  fe.GetGeomType() == mfem::Geometry::SQUARE && fe.GetDof() == dof_ &&
                  fe.GetMapType() == mfem::FiniteElement::VALUE && fe.GetNodes().GetNPoints() == dof_,
                "Local tensor Jacobi requires actual Gauss-Legendre nodal tensor elements.");
    const auto& lex = tensor->GetDofMap();
    MFEM_VERIFY(lex.Size() == 0 || lex.Size() == dof_, "Invalid local tensor native permutation.");
    lex_to_native_.SetSize(dof_);
    int* native = lex_to_native_.HostWrite();
    const auto& rule = mass_space_->GetIntRule(0);
    MFEM_VERIFY(rule.GetNPoints() == dof_, "Local tensor Jacobi requires collocated mass quadrature.");
    std::vector<bool> local_seen(dof_, false), seen(height, false);
    mfem::Vector shape(dof_);
    for (int i = 0; i < dof_; ++i)
    {
      native[i] = lex.Size() ? lex[i] : i;
      MFEM_VERIFY(native[i] >= 0 && native[i] < dof_ && !local_seen[native[i]],
                  "Invalid unsigned local tensor native permutation.");
      local_seen[native[i]] = true;
      const auto& ip = rule.IntPoint(i);
      const auto& node = fe.GetNodes().IntPoint(native[i]);
      MFEM_VERIFY(std::isfinite(ip.x) && std::isfinite(ip.y) && std::isfinite(ip.weight) &&
                    ip.weight > 0.0 && ip.x == rule.IntPoint(i % axis_).x &&
                    ip.y == rule.IntPoint(i / axis_).x &&
                    std::abs(ip.x - node.x) < 1.0e-13 && std::abs(ip.y - node.y) < 1.0e-13,
                  "Local tensor Jacobi requires matching GL nodes and lexicographic tensor quadrature.");
      fe.CalcShape(ip, shape);
      for (int j = 0; j < dof_; ++j)
      {
        MFEM_VERIFY(std::abs(shape[j] - double(j == native[i])) < 1.0e-12,
                    "Local tensor Jacobi mass quadrature is not nodally collocated.");
      }
    }
    element_dofs_.SetSize(height); geometry_mass_.SetSize(height); inverse_sqrt_mass_.SetSize(height);
    int* map = element_dofs_.HostWrite();
    double* geometry = geometry_mass_.HostWrite();
    double* scale = inverse_sqrt_mass_.HostWrite();
    const double* coefficient = invrho.HostRead();
    mfem::Array<int> dofs;
    for (int e = 0; e < ne_; ++e)
    {
      MFEM_VERIFY(space_.GetFE(e) == &fe && mass_space_->Offset(e) == e * dof_ &&
                    &mass_space_->GetIntRule(e) == &rule &&
                    space_.GetElementDofs(e, dofs) == nullptr && dofs.Size() == dof_,
                  "Local tensor Jacobi requires a uniform FE/QF layout without dof transformations.");
      auto* transformation = space_.GetMesh()->GetElementTransformation(e);
      // Restrict the geometry representation as well as its sampled metric.
      // For a bilinear Q1 map, constant diagonal Jacobians at all four corners
      // certify an affine Cartesian map, including the D1D=1 case.
      MFEM_VERIFY(transformation->Order() == 1,
                  "Local tensor Jacobi admits only fixed affine Cartesian geometry; element " << e);
      double jx = 0.0, jy = 0.0;
      const auto& vertices = *mfem::Geometries.GetVertices(mfem::Geometry::SQUARE);
      for (int vertex = 0; vertex < vertices.GetNPoints(); ++vertex)
      {
        transformation->SetIntPoint(&vertices.IntPoint(vertex));
        const auto& jacobian = transformation->Jacobian();
        if (vertex == 0) { jx = jacobian(0, 0); jy = jacobian(1, 1); }
        const double length = std::max(std::abs(jx), std::abs(jy));
        MFEM_VERIFY(std::isfinite(jx) && std::isfinite(jy) && jx > 0.0 && jy > 0.0 &&
                      std::abs(jacobian(0, 0) - jx) <= 1.0e-12 * length &&
                      std::abs(jacobian(1, 1) - jy) <= 1.0e-12 * length &&
                      std::abs(jacobian(0, 1)) <= 1.0e-12 * length &&
                      std::abs(jacobian(1, 0)) <= 1.0e-12 * length,
                    "Local tensor Jacobi admits only fixed affine Cartesian geometry; element " << e);
      }
      for (int i = 0; i < dof_; ++i)
      {
        const int row = dofs[native[i]];
        MFEM_VERIFY(row >= 0 && row < height && !seen[row],
                    "Local tensor Jacobi requires disjoint unsigned scalar element dofs.");
        seen[row] = true; map[e * dof_ + i] = row;
        const auto& ip = rule.IntPoint(i);
        transformation->SetIntPoint(&ip);
        const auto& jacobian = transformation->Jacobian();
        const double xx = jacobian(0, 0), yy = jacobian(1, 1);
        const double length = std::max(std::abs(jx), std::abs(jy));
        MFEM_VERIFY(std::isfinite(xx) && std::isfinite(yy) && jx > 0.0 && jy > 0.0 &&
                      std::abs(xx - jx) <= 1.0e-12 * length && std::abs(yy - jy) <= 1.0e-12 * length &&
                      std::abs(jacobian(0, 1)) <= 1.0e-12 * length &&
                      std::abs(jacobian(1, 0)) <= 1.0e-12 * length,
                    "Local tensor Jacobi admits only fixed affine Cartesian geometry; element " << e);
        const double weight = ip.weight * transformation->Weight();
        const double mass = weight * coefficient[e * dof_ + i];
        MFEM_VERIFY(std::isfinite(weight) && weight > 0.0 && std::isfinite(mass) && mass > 0.0,
                    "Local tensor Jacobi requires positive initial physical nodal mass; element " << e);
        geometry[e * dof_ + i] = weight;
        scale[e * dof_ + i] = 1.0 / std::sqrt(mass);
      }
    }
    geometry_mass_.UseDevice(true); inverse_sqrt_mass_.UseDevice(true);
    (void)element_dofs_.Read(); (void)geometry_mass_.Read(); (void)inverse_sqrt_mass_.Read();
  }

  void CheckTopology() const
  {
    MFEM_VERIFY(space_.GetSequence() == space_sequence_ && space_.GetMesh() &&
                  space_.GetMesh()->GetSequence() == mesh_sequence_ &&
                  space_.GetMesh()->GetNodesSequence() == nodes_sequence_ &&
                  space_.GetNE() == ne_ && space_.GetVSize() == height,
                "Local tensor Jacobi topology/geometry changed; reconstruct the backend.");
  }

  mfem::ParFiniteElementSpace& space_;
  const mfem::QuadratureSpace* mass_space_;
  const int ne_, axis_, dof_;
  const long space_sequence_, mesh_sequence_, nodes_sequence_;
  const TensorExecution execution_;
  const TensorReference reference_;
  std::unique_ptr<mfem::EABilinearFormExtension> ea_;
  mfem::DenseTensor blocks_;
  mfem::Array<int> lex_to_native_, element_dofs_, element_classes_;
  mfem::Array<int> grouped_elements_, grouped_classes_;
  mfem::Vector geometry_mass_, inverse_sqrt_mass_, factors_, class_inverses_;
  mfem::Vector mass_basis_squared_, element_diagonals_;
  int mass_axis_ = 0, maximum_geometry_order_ = 1;
  bool prepared_ = false, ready_ = false;
  mutable bool update_execution_reported_ = false;
};
}

std::unique_ptr<ElementBlockJacobi> MakeTensorElementBlockJacobi(
  mfem::ParFiniteElementSpace& space, const mfem::QuadratureFunction& initial_invrho)
{
  return std::make_unique<LocalTensorJacobi>(space, initial_invrho);
}

}
}
