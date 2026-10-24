#include "igrlab/compat.hpp"
#include "IGR_element_block_jacobi.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace igrlab
{
namespace igr_internal
{
namespace
{
void CheckBLAS(cublasStatus_t status)
{
  MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS,
              "Element-block Jacobi cuBLAS failure: " << int(status));
}

__global__ void PrepareFactorPointers(int dof, double* blocks,
                                      double** pointers, int count)
{
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < count)
  { pointers[element] = blocks + std::size_t(element) * dof * dof; }
}

// getrf info is mandatory singularity/API handling, not an optional finite scan.
__global__ void CheckFactorInfo(int count, const int* info, int* failure)
{
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < count && info[element] != 0)
  { atomicCAS(failure, 0, element + 1); }
}

__global__ void ApplyElementInverse(int dof, const int* map, const double* inverse,
                                    const double* input, double* output)
{
  const int element = blockIdx.x, row = threadIdx.x;
  __shared__ double rhs[256];
  if (row < dof) { rhs[row] = input[map[element * dof + row]]; }
  __syncthreads();
  if (row >= dof) { return; }
  double value = 0.0;
  for (int j = 0; j < dof; ++j)
  {
    value += inverse[(std::size_t(element) * dof + j) * dof + row] * rhs[j];
  }
  output[map[element * dof + row]] = value;
}

class CurrentElementBlockJacobi final : public ElementBlockJacobi
{
public:
  CurrentElementBlockJacobi(mfem::ParFiniteElementSpace& space, bool explicit_inverse)
    : ElementBlockJacobi(space.GetVSize()), space_(space),
      explicit_inverse_(explicit_inverse),
      ne_(space.GetNE()), dof_(ne_ ? space.GetFE(0)->GetDof() : 0),
      space_sequence_(space.GetSequence()),
      mesh_sequence_(space.GetMesh() ? space.GetMesh()->GetSequence() : -1)
  {
    static_assert(std::is_same<mfem::real_t, double>::value,
                  "Element-block Jacobi requires FP64 MFEM.");
    int ranks = 0;
    MPI_Comm_size(space.GetComm(), &ranks);
    MFEM_VERIFY(ranks == 1 && dg_face::UsesOrdinaryCUDAForDGDiffusionFaces2D() &&
                  dynamic_cast<const mfem::L2_FECollection*>(space.FEColl()) &&
                  space.GetVDim() == 1 && space.GetMesh() &&
                  space.GetMesh()->Dimension() == 2 &&
                  space.GetMesh()->SpaceDimension() == 2 && space.Conforming() &&
                  ne_ > 0 && dof_ > 0 && dof_ <= 256 &&
                  std::size_t(ne_) * dof_ == std::size_t(height) &&
                  space.GetTrueVSize() == height &&
                  mfem::IsIdentityProlongation(space.GetProlongationMatrix()) &&
                  mfem::IsIdentityProlongation(space.GetRestrictionTransposeOperator()),
                "Element-block Jacobi requires rank-one native CUDA, scalar uniform "
                "nodal L2 quads and an identity true/local map.");
    MFEM_VERIFY(std::size_t(ne_) * dof_ * dof_ <=
                  std::size_t(std::numeric_limits<int>::max()),
                "Element blocks exceed MFEM integer indexing.");
    element_dofs_.SetSize(height);
    std::vector<char> seen(height, 0);
    mfem::Array<int> dofs;
    const auto& reference = *space.GetFE(0);
    MFEM_VERIFY(dynamic_cast<const mfem::NodalTensorFiniteElement*>(&reference),
                "Element-block Jacobi requires an actual nodal tensor element.");
    for (int e = 0; e < ne_; ++e)
    {
      const auto& fe = *space.GetFE(e);
      MFEM_VERIFY(&fe == &reference && fe.GetGeomType() == mfem::Geometry::SQUARE &&
                    fe.GetOrder() == reference.GetOrder() && fe.GetDof() == dof_ &&
                    fe.GetMapType() == mfem::FiniteElement::VALUE &&
                    fe.GetNodes().GetNPoints() == dof_,
                  "Element-block Jacobi requires uniform value-mapped nodal quads.");
      MFEM_VERIFY(space.GetElementDofs(e, dofs) == nullptr && dofs.Size() == dof_,
                  "Element-block Jacobi does not admit dof transformations.");
      for (int i = 0; i < dof_; ++i)
      {
        const int row = dofs[i];
        MFEM_VERIFY(row >= 0 && row < height && !seen[row],
                    "Element-block Jacobi needs a disjoint unsigned native map.");
        seen[row] = 1;
        element_dofs_[e * dof_ + i] = row;
      }
    }
    blocks_.SetSize(dof_, dof_, ne_);
    blocks_.GetMemory().UseDevice(true);
    factor_pointers_.SetSize(ne_);
    pivots_.SetSize(height);
    factor_info_.SetSize(ne_);
    failure_.SetSize(1);
    if (explicit_inverse_)
    {
      inverse_.SetSize(dof_, dof_, ne_);
      inverse_.GetMemory().UseDevice(true);
      inverse_pointers_.SetSize(ne_);
    }
    else
    {
      rhs_pointers_.SetSize(ne_);
      rhs_.SetSize(height);
      rhs_.UseDevice(true);
    }
    CheckBLAS(cublasCreate(&handle_));
    CheckBLAS(cublasSetStream(handle_, nullptr));
    CheckBLAS(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
    (void)element_dofs_.Read();
  }

  ~CurrentElementBlockJacobi() override
  {
    if (handle_) { cublasDestroy(handle_); }
  }

  void SetOperator(const mfem::Operator&) override
  {
    MFEM_ABORT("Element blocks must be refreshed with PrepareVolumeBlocks/FactorBlocks.");
  }

  mfem::MemoryClass GetMemoryClass() const override
  { return mfem::MemoryClass::DEVICE; }

  mfem::DenseTensor& PrepareVolumeBlocks(mfem::BilinearForm& form) override
  {
    CheckTopology();
    ready_ = prepared_ = false;
    MFEM_VERIFY(form.FESpace() == &space_ && form.GetFBFI()->Size() == 0 &&
                  form.GetBBFI()->Size() == 0 && form.GetBFBFI()->Size() == 0,
                "PrepareVolumeBlocks expects this space's domain-only form; "
                "exact face terms are added separately.");
    if (form_ != &form)
    {
      ea_ = std::make_unique<mfem::EABilinearFormExtension>(&form);
      form_ = &form;
    }
    // Always refresh numerical matrices, even if form/coefficient addresses
    // and dimensions are unchanged. EA shares the actual domain integrators.
    ea_->Assemble();
    ea_->GetElementMatrices(blocks_, mfem::ElementDofOrdering::NATIVE, false);
    if (explicit_inverse_)
    {
      // Reference mode releases its build-only storage after inversion. A
      // subsequent explicit refresh is still supported (and uses current A).
      blocks_.GetMemory().UseDevice(true);
      factor_pointers_.SetSize(ne_);
      pivots_.SetSize(height);
      factor_info_.SetSize(ne_);
      inverse_pointers_.SetSize(ne_);
    }
    MFEM_VERIFY(blocks_.SizeI() == dof_ && blocks_.SizeJ() == dof_ &&
                  blocks_.SizeK() == ne_, "Unexpected EA element block layout.");
    prepared_ = true;
    return blocks_;
  }

  void FactorBlocks() override
  {
    CheckTopology();
    MFEM_VERIFY(prepared_ && !ready_, "Prepare current element blocks before factorization.");
    PrepareFactorPointers<<<1 + (ne_ - 1) / 256, 256>>>(
      dof_, blocks_.ReadWrite(), factor_pointers_.Write(), ne_);
    MFEM_GPU_CHECK(cudaGetLastError());
    CheckBLAS(cublasDgetrfBatched(handle_, dof_, factor_pointers_.ReadWrite(), dof_,
                                  pivots_.Write(), factor_info_.Write(), ne_));
    failure_.HostWrite()[0] = 0;
    CheckFactorInfo<<<1 + (ne_ - 1) / 256, 256>>>(ne_, factor_info_.Read(), failure_.ReadWrite());
    MFEM_GPU_CHECK(cudaGetLastError());
    MFEM_VERIFY(failure_.HostRead()[0] == 0,
                "Current element LU factorization failed at element " << failure_.HostRead()[0] - 1);
    if (explicit_inverse_)
    {
      PrepareFactorPointers<<<1 + (ne_ - 1) / 256, 256>>>(
        dof_, inverse_.Write(), inverse_pointers_.Write(), ne_);
      MFEM_GPU_CHECK(cudaGetLastError());
      const double** factors = const_cast<const double**>(
        reinterpret_cast<const double* const*>(factor_pointers_.Read()));
      CheckBLAS(cublasDgetriBatched(handle_, dof_, factors, dof_, pivots_.Read(),
        inverse_pointers_.ReadWrite(), dof_, factor_info_.Write(), ne_));
      failure_.HostWrite()[0] = 0;
      CheckFactorInfo<<<1 + (ne_ - 1) / 256, 256>>>(ne_, factor_info_.Read(), failure_.ReadWrite());
      MFEM_GPU_CHECK(cudaGetLastError());
      MFEM_VERIFY(failure_.HostRead()[0] == 0,
                  "Explicit element inverse failed at element " << failure_.HostRead()[0] - 1);
      // The checked device info above synchronizes inversion. Neither the
      // inverse application nor a future assembly needs the old EA/LU data.
      ea_.reset();
      form_ = nullptr;
      blocks_.Clear();
      factor_pointers_.DeleteAll();
      inverse_pointers_.DeleteAll();
      pivots_.DeleteAll();
      factor_info_.DeleteAll();
    }
    ready_ = true;
    prepared_ = false;
  }

  void Mult(const mfem::Vector& r, mfem::Vector& z) const override
  {
    CheckTopology();
    MFEM_VERIFY(ready_ && r.Size() == height && &r != &z,
                "Current element factors and distinct input/output are required.");
    z.SetSize(height);
    z.UseDevice(true);
    const int d = dof_;
    const int* map = element_dofs_.Read();
    const double* input = r.Read();
    if (explicit_inverse_)
    {
      int threads = 32;
      while (threads < d) { threads *= 2; }
      ApplyElementInverse<<<ne_, threads>>>(d, map, inverse_.Read(), input, z.Write());
      MFEM_GPU_CHECK(cudaGetLastError());
      return;
    }
    double* packed = rhs_.Write();
    double** pointers = rhs_pointers_.Write();
    mfem::forall(height, [=] MFEM_HOST_DEVICE(int entry)
    {
      packed[entry] = input[map[entry]];
      if (entry % d == 0) { pointers[entry / d] = packed + entry; }
    });
    int info = 0; // getrsBatched's scalar argument-status output is host memory.
    // cuBLAS treats the pointer arrays as inputs, including in header versions
    // whose batched signatures omit top-level const on those arrays.
    const double** factors = const_cast<const double**>(
      reinterpret_cast<const double* const*>(factor_pointers_.Read()));
    CheckBLAS(cublasDgetrsBatched(handle_, CUBLAS_OP_N, d, 1,
      factors, d, pivots_.Read(), rhs_pointers_.ReadWrite(), d, &info, ne_));
    MFEM_VERIFY(info == 0, "Element-block triangular solve rejected its arguments: " << info);
    const double* solution = rhs_.Read();
    double* output = z.Write();
    mfem::forall(height, [=] MFEM_HOST_DEVICE(int entry)
    { output[map[entry]] = solution[entry]; });
    MFEM_GPU_CHECK(cudaGetLastError());
  }

private:
  void CheckTopology() const
  {
    MFEM_VERIFY(space_.GetSequence() == space_sequence_ && space_.GetMesh() &&
                  space_.GetMesh()->GetSequence() == mesh_sequence_ &&
                  space_.GetNE() == ne_ && space_.GetVSize() == height,
                "Element-block topology changed; reconstruct the backend.");
  }

  mfem::ParFiniteElementSpace& space_;
  const bool explicit_inverse_;
  const int ne_, dof_;
  const long space_sequence_, mesh_sequence_;
  mfem::BilinearForm* form_ = nullptr;
  std::unique_ptr<mfem::EABilinearFormExtension> ea_;
  mfem::DenseTensor blocks_, inverse_;
  mfem::Array<int> element_dofs_, pivots_, factor_info_, failure_;
  mfem::Array<double*> factor_pointers_, inverse_pointers_;
  mutable mfem::Array<double*> rhs_pointers_;
  mutable mfem::Vector rhs_;
  cublasHandle_t handle_ = nullptr;
  bool prepared_ = false, ready_ = false;
};

template <bool FactorizedNormals>
struct AddFaceBlocks
{
  int dof, nq, ne, nf, shape_slots, normal_slots;
  double ip_sigma;
  const int *offsets, *face_ids, *row_sides, *face_elements;
  const int *shape_map, *normal_map;
  const double *shape, *normal, *reference_dshape, *geometry, *coefficient, *penalty;
  double* blocks;

  MFEM_HOST_DEVICE double Shape(int side, int q, int i, int face) const
  {
    const int slot = shape_map ? shape_map[face] : face;
    return shape[side + 2 * (q + nq * (i + dof * slot))];
  }

  MFEM_HOST_DEVICE double Normal(int side, int q, int i, int face) const
  {
    if constexpr (FactorizedNormals)
    {
      const int slot = normal_map[face];
      const int reference = 2 * (side + 2 * (q + nq * (i + dof * slot)));
      const int physical = 2 * (side + 2 * (q + nq * face));
      const double dx = reference_dshape[reference];
      const double dy = reference_dshape[reference + 1];
      const double nx = geometry[physical], ny = geometry[physical + 1];
      return dg_face::InteriorFaceDGDiffusionRecord2DWarp<true>::ReconstructNormal(
        dx, dy, nx, ny, coefficient[side + 2 * (q + nq * face)]);
    }
    else { return normal[side + 2 * (q + nq * (i + dof * face))]; }
  }

  MFEM_HOST_DEVICE void operator()(int entry) const
  {
    const int element = entry / (dof * dof);
    const int i = entry % dof, j = (entry / dof) % dof;
    double value = blocks[entry];
    for (int incidence = offsets[element]; incidence < offsets[element + 1]; ++incidence)
    {
      const int face = face_ids[incidence], row_side = row_sides[incidence];
      const double row_sign = row_side == 0 ? 1.0 : -1.0;
      for (int col_side = 0; col_side < 2; ++col_side)
      {
        if (face_elements[col_side + 2 * face] != element) { continue; }
        const double col_sign = col_side == 0 ? 1.0 : -1.0;
        for (int q = 0; q < nq; ++q)
        {
          const double si = Shape(row_side, q, i, face);
          const double sj = Shape(col_side, q, j, face);
          const double ni = Normal(row_side, q, i, face);
          const double nj = Normal(col_side, q, j, face);
          value += -row_sign * si * nj + ip_sigma * col_sign * ni * sj +
                   row_sign * col_sign * penalty[q + nq * face] * si * sj;
        }
      }
    }
    blocks[entry] = value;
  }
};
}

std::unique_ptr<ElementBlockJacobi> MakeElementBlockJacobi(mfem::ParFiniteElementSpace& space,
                                                         bool explicit_inverse)
{
  return std::make_unique<CurrentElementBlockJacobi>(space, explicit_inverse);
}

void AddExactFaceBlocks(const dg_face::InteriorFaceDGDiffusionApplyInfo& info,
                        const dg_face::InteriorFaceDGDiffusionApplyScratch& work,
                        mfem::real_t ip_sigma, mfem::DenseTensor& blocks)
{
  MFEM_VERIFY(dg_face::UsesOrdinaryCUDAForDGDiffusionFaces2D() &&
                info.dim == 2 && info.vdim == 1 && info.dof > 0 && info.ne > 0 &&
                info.nq > 0 && info.nf >= 0 && std::isfinite(ip_sigma) &&
                blocks.SizeI() == info.dof && blocks.SizeJ() == info.dof && blocks.SizeK() == info.ne &&
                std::size_t(info.ne) * info.dof * info.dof <= std::size_t(std::numeric_limits<int>::max()) &&
                std::size_t(info.nq) * info.nf == std::size_t(work.penalty.Size()) &&
                std::size_t(2) * info.nf == std::size_t(work.face_elements.Size()),
              "Exact element-face blocks received an invalid scalar layout.");
  if (info.nf == 0) { return; }
  dg_face::EnsureInteriorFaceElementIncidence2D(info, work);
  const auto& factor = work.normal_factorization;
  const bool factorized = factor.Matches(info) &&
    factor.coefficient_revision == info.connectivity_revision &&
    work.side_coefficient.Size() == 2 * info.nq * info.nf &&
    (factor.coefficient_scaling == dg_face::DGDiffusionFaceScaling::Arithmetic ||
     factor.coefficient_scaling == dg_face::DGDiffusionFaceScaling::Max ||
     factor.coefficient_scaling == dg_face::DGDiffusionFaceScaling::Sum);
  const bool compact_shape = work.shape_dictionary.Matches(info);
  MFEM_VERIFY(compact_shape || std::size_t(work.shape.Size()) ==
                std::size_t(2) * info.nq * info.dof * info.nf,
              "Exact face shape storage is incomplete.");
  if (!factorized)
  {
    dg_face::EnsureInteriorFaceDGDiffusionDenseNormals2D(info, work);
    MFEM_VERIFY(std::size_t(work.weighted_normal_dshape.Size()) ==
                  std::size_t(2) * info.nq * info.dof * info.nf,
                "Exact current dense face normals are incomplete.");
  }
  AddFaceBlocks<true> add{info.dof, info.nq, info.ne, info.nf,
    compact_shape ? work.shape_dictionary.slot_count : info.nf, factor.slot_count,
    ip_sigma, work.element_face_offsets.Read(), work.element_face_ids.Read(),
    work.element_face_sides.Read(), work.face_elements.Read(),
    compact_shape ? work.shape_dictionary.face_slots.Read() : nullptr,
    factorized ? factor.face_slots.Read() : nullptr,
    compact_shape ? work.shape_dictionary.values.Read() : work.shape.Read(),
    factorized ? nullptr : work.weighted_normal_dshape.Read(),
    factorized ? factor.reference_dshape.Read() : nullptr,
    factorized ? factor.normal_geometry.Read() : nullptr,
    factorized ? work.side_coefficient.Read() : nullptr, work.penalty.Read(), blocks.ReadWrite()};
  const int entries = blocks.TotalSize();
  if (factorized) { mfem::forall(entries, add); }
  else
  {
    const AddFaceBlocks<false> dense{add.dof, add.nq, add.ne, add.nf, add.shape_slots, add.normal_slots,
      add.ip_sigma, add.offsets, add.face_ids, add.row_sides, add.face_elements,
      add.shape_map, add.normal_map, add.shape, add.normal, add.reference_dshape,
      add.geometry, add.coefficient, add.penalty, add.blocks};
    mfem::forall(entries, dense);
  }
  MFEM_GPU_CHECK(cudaGetLastError());
}

}
}
