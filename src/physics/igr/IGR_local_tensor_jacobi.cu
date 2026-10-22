#include "igrlab/compat.hpp"
#include "igrlab/environment.hpp"
#include "IGR_local_tensor_jacobi.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <type_traits>
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
  SimtFP64, SimtFP32, SimtDenseFP32, FactorTF32, DenseTF32, DenseFP64
};

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
  if (std::strcmp(value, "wmma-dense-tf32") == 0) { return TensorExecution::DenseTF32; }
  if (std::strcmp(value, "wmma-dense-fp64") == 0) { return TensorExecution::DenseFP64; }
  MFEM_ABORT("IGRLAB_IGR_LOCAL_TENSOR_EXECUTION must be simt-fp64|simt-fp32|"
             "simt-dense-fp32|wmma-factor-tf32|wmma-dense-tf32|wmma-dense-fp64; got " << value);
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
    case TensorExecution::DenseTF32: return "wmma-dense-tf32";
    case TensorExecution::DenseFP64: return "wmma-dense-fp64";
  }
  return "invalid";
}

bool UsesTensorCores(TensorExecution execution)
{
  return execution == TensorExecution::FactorTF32 ||
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
                                   const double* factors, const double* scaling,
                                   const double* input, double* output)
{
  const int e = blockIdx.x, t = threadIdx.x, dof = axis * axis;
  const int x = t % axis, y = t / axis;
  const double* qx = factors + std::size_t(classes[e]) * 3 * dof;
  const double* qy = qx + dof;
  const double* denominator = qy + dof;
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
      execution_(ReadTensorExecution())
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
                "Local tensor Jacobi requires rank-one native CUDA, uniform scalar GL L2 quads, "
                "D1D <= 8 and an identity true/local map.");
    MFEM_VERIFY(std::size_t(ne_) * dof_ * dof_ <= std::size_t(std::numeric_limits<int>::max()) &&
                  invrho.Size() == height,
                "Local tensor Jacobi dense setup/QF indexing bound exceeded.");
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
    // The complete exact blocks, their host copy, EA buffers, and native-only
    // permutation are setup-only. Persistent storage is O(N) plus small classes.
    ea_.reset();
    blocks_.Clear();
    lex_to_native_.DeleteAll();
    factors_.UseDevice(true);
    (void)element_classes_.Read(); (void)factors_.Read();
    ready_ = true; prepared_ = false;
    const std::size_t bytes = sizeof(double) * (std::size_t(geometry_mass_.Size()) +
      inverse_sqrt_mass_.Size() + factors_.Size() + class_inverses_.Size()) +
      sizeof(int) * (std::size_t(element_dofs_.Size()) + element_classes_.Size() +
                     grouped_elements_.Size() + grouped_classes_.Size());
    const int dense_classes = UsesDenseClasses(execution_) ? int(classes.size()) : 0;
    const bool grouped = grouped_classes_.Size() > 0;
    std::cout << "[igr-local-tensor-jacobi] event=setup precision=" << PrecisionName(execution_)
              << " execution=" << (UsesTensorCores(execution_) ? "wmma" : "simt")
              << " backend=" << ExecutionName(execution_)
              << " tensor_core=" << int(UsesTensorCores(execution_))
              << " d1d=" << axis_ << " elements=" << ne_ << " factor_classes=" << classes.size()
              << " coefficient_policy=current-collocated-mass initial_blocks=exact-sipg"
              << " geometry=affine-cartesian reconstruction_relative=" << maximum_reconstruction
              << " owned_bytes=" << bytes << " dense_after_setup=" << dense_classes
              << " class_dense_matrices=" << dense_classes
              << " class_dense_entries=" << class_inverses_.Size() << " dense_per_element=0"
              << " grouping=" << (grouped ? "verified-class" : "element")
              << " group_elements=" << (grouped ? tensor_group_elements : 1)
              << " groups=" << (grouped ? grouped_classes_.Size() : ne_)
              << " group_padding=" << (grouped ? grouped_elements_.Size() - ne_ : 0)
              << " mass_precision=fp64 output_precision=fp64"
              << " arithmetic_spd=" << (execution_ == TensorExecution::SimtFP64 ||
                  execution_ == TensorExecution::DenseFP64 ? "fp64-roundoff" : "rounding-perturbed")
              << " fallback=none" << std::endl;
  }

  void RefreshMassScaling(const mfem::QuadratureFunction& invrho) override
  {
    CheckTopology();
    MFEM_VERIFY(ready_ && &CheckMassQF(invrho, space_) == mass_space_ && invrho.Size() == height,
                "Local tensor Jacobi mass layout changed; reconstruct the backend.");
    // invrho uses the established coefficient floor semantics. This update
    // intentionally adds no per-stage host copy, coefficient scan or reduction.
    RefreshTensorMass<<<1 + (height - 1) / 256, 256>>>(
      height, geometry_mass_.Read(), invrho.Read(), inverse_sqrt_mass_.Write());
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
    double* output = result.Write();
    if (execution_ == TensorExecution::SimtFP64 || execution_ == TensorExecution::SimtFP32)
    {
      int threads = 32;
      while (threads < dof_) { threads *= 2; }
      if (execution_ == TensorExecution::SimtFP64)
      {
        ApplyTensorInverse<<<ne_, threads>>>(axis_, map, element_classes_.Read(),
          factors_.Read(), scaling, input, output);
      }
      else
      {
        ApplyTensorInverseFP32<<<ne_, threads>>>(axis_, map, element_classes_.Read(),
          factors_.Read(), scaling, input, output);
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
          scaling, input, output);
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

private:
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

  void BuildLayoutAndMass(const mfem::QuadratureFunction& invrho)
  {
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
  std::unique_ptr<mfem::EABilinearFormExtension> ea_;
  mfem::DenseTensor blocks_;
  mfem::Array<int> lex_to_native_, element_dofs_, element_classes_;
  mfem::Array<int> grouped_elements_, grouped_classes_;
  mfem::Vector geometry_mass_, inverse_sqrt_mass_, factors_, class_inverses_;
  bool prepared_ = false, ready_ = false;
};
}

std::unique_ptr<ElementBlockJacobi> MakeTensorElementBlockJacobi(
  mfem::ParFiniteElementSpace& space, const mfem::QuadratureFunction& initial_invrho)
{
  return std::make_unique<LocalTensorJacobi>(space, initial_invrho);
}

}
}
