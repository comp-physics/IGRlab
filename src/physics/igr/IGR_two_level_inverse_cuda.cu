#include "igrlab/compat.hpp"
#include "IGR_two_level_inverse.hpp"
#include "IGR_device_krylov.hpp"
#include "IGR_two_level_blocks_cuda.cuh"
#include "IGR_shared_patch_cuda.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

namespace igrlab
{
namespace igr_internal
{
namespace
{
using Graph = std::vector<std::set<int>>;

std::vector<int> ColorGraph(const Graph& graph)
{
  std::vector<int> colors(graph.size(), -1), used(graph.size(), -1);
  for (int i = 0; i < static_cast<int>(graph.size()); ++i)
  {
    for (int j : graph[i])
    {
      if (colors[j] >= 0) { used[colors[j]] = i; }
    }
    int color = 0;
    while (used[color] == i) { ++color; }
    colors[i] = color;
  }
  return colors;
}

int ColorCount(const std::vector<int>& colors)
{ return 1 + *std::max_element(colors.begin(), colors.end()); }

std::vector<int> ColorColumns(const Graph& rows)
{
  Graph conflict(rows.size());
  for (const auto& row : rows)
  {
    for (int i : row)
    {
      for (int j : row)
      {
        if (i != j) { conflict[i].insert(j); }
      }
    }
  }
  return ColorGraph(conflict);
}

double RelativeDifference(const mfem::Vector& a, const mfem::Vector& b)
{
  const double* av = a.HostRead();
  const double* bv = b.HostRead();
  long double error = 0.0L, scale = 0.0L;
  for (int i = 0; i < a.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(av[i]) && std::isfinite(bv[i]),
                "Nonfinite two-level setup probe.");
    const long double delta = static_cast<long double>(av[i]) - bv[i];
    error += delta * delta;
    scale += static_cast<long double>(bv[i]) * bv[i];
  }
  return std::sqrt(static_cast<double>(error / std::max(scale, 1.0e-300L)));
}

void InvertChecked(mfem::DenseMatrix& matrix)
{
  matrix.Invert(); // Pivoted FP64 inversion; preserve the measured entries.
  for (int i = 0; i < matrix.TotalSize(); ++i)
  {
    MFEM_VERIFY(std::isfinite(matrix.Data()[i]),
                "Nonfinite two-level dense inverse.");
  }
}

class TwoLevelInverse final : public mfem::Solver
{
public:
  TwoLevelInverse(const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes,
                  const std::string& coarse_mode, int coarse_order)
    : mfem::Solver(op.Height()), ne_(sfes.GetNE()),
      dof_(ne_ > 0 ? sfes.GetFE(0)->GetDof() : 0), coarse_mode_(coarse_mode),
      local_mode_(TwoLevelLocalMode()), patch_composition_(TwoLevelPatchComposition())
  {
    const auto start = std::chrono::steady_clock::now();
    int ranks = 0;
    MPI_Comm_size(sfes.GetComm(), &ranks);
    MFEM_VERIFY(ranks == 1 && mfem::Device::Allows(mfem::Backend::CUDA),
                "Two-level inverse requires rank-one CUDA.");
    MFEM_VERIFY(dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl()) &&
                  sfes.GetVDim() == 1 && sfes.GetMesh()->Dimension() == 2 &&
                  ne_ > 0 && dof_ > 0 && dof_ <= 256 &&
                  op.Width() == height && height == sfes.GetVSize() &&
                  height == sfes.GetTrueVSize() && height == ne_ * dof_,
                "Two-level inverse requires scalar, uniform nodal L2 quads "
                "with identity true/local map and <=256 local DOFs.");
    MFEM_VERIFY(coarse_mode == "dense" || coarse_mode == "cg",
                "Two-level coarse mode must be dense|cg.");
    const int fine_order = sfes.GetFE(0)->GetOrder();
    MFEM_VERIFY(coarse_order >= 1 && coarse_order <= fine_order,
                "Two-level coarse order must be between 1 and the sigma FE order.");

    Graph adjacent(ne_);
    for (int face = 0; face < sfes.GetMesh()->GetNumFaces(); ++face)
    {
      int left = -1, right = -1;
      sfes.GetMesh()->GetFaceElements(face, &left, &right);
      if (left >= 0 && right >= 0 && left != right)
      {
        adjacent[left].insert(right);
        adjacent[right].insert(left);
      }
    }
    const bool shared_patch = local_mode_ != "element";
    MFEM_VERIFY(shared_patch || patch_composition_ == "additive",
                "Multiplicative patch composition requires a shared-patch local mode.");
    const auto patch = shared_patch ? SharedPatchTopology(sfes) : std::vector<std::array<int,5>>{};
    Graph sample_support = adjacent;
    if (shared_patch) for (int e = 0; e < ne_; ++e) sample_support[e].insert(e);
    const auto element_colors = shared_patch ? ColorColumns(sample_support) : ColorGraph(adjacent);
    const int element_color_count = ColorCount(element_colors);
    mfem::H1_FECollection coarse_collection(coarse_order, 2);
    mfem::FiniteElementSpace coarse_space(sfes.GetMesh(), &coarse_collection);
    MFEM_VERIFY(coarse_space.GetConformingProlongation() == nullptr,
                "Two-level inverse requires a conforming mesh.");
    const int nc = coarse_space.GetVSize();
    MFEM_VERIFY(coarse_mode != "dense" || nc <= 512,
                "Dense two-level coarse solve is bounded to 512 unknowns; "
                "select coarse mode cg explicitly for larger systems.");
    prolong_ = std::make_unique<mfem::SparseMatrix>(height, nc);
    std::vector<std::vector<int>> element_coarse(ne_);
    element_dofs_.SetSize(height);
    std::vector<char> seen(height, 0);
    mfem::Array<int> fine_dofs, coarse_dofs;
    mfem::Vector shape;
    for (int e = 0; e < ne_; ++e)
    {
      const auto& fine_fe = *sfes.GetFE(e);
      const auto& coarse_fe = *coarse_space.GetFE(e);
      MFEM_VERIFY(fine_fe.GetGeomType() == mfem::Geometry::SQUARE &&
                    fine_fe.GetDof() == dof_ && fine_fe.GetOrder() == fine_order &&
                    fine_fe.GetMapType() == mfem::FiniteElement::VALUE &&
                    fine_fe.GetNodes().GetNPoints() == dof_,
                  "Two-level inverse requires uniform nodal quadrilaterals.");
      sfes.GetElementDofs(e, fine_dofs);
      coarse_space.GetElementDofs(e, coarse_dofs);
      element_coarse[e].assign(coarse_dofs.begin(), coarse_dofs.end());
      shape.SetSize(coarse_fe.GetDof());
      for (int i = 0; i < dof_; ++i)
      {
        const int row = fine_dofs[i];
        MFEM_VERIFY(row >= 0 && row < height && !seen[row],
                    "Two-level inverse requires a disjoint unsigned L2 map.");
        seen[row] = 1;
        element_dofs_[e * dof_ + i] = row;
        // Sigma may use Gauss-Legendre nodes; evaluate at its actual nodes.
        coarse_fe.CalcShape(fine_fe.GetNodes().IntPoint(i), shape);
        for (int j = 0; j < coarse_dofs.Size(); ++j)
        {
          MFEM_VERIFY(coarse_dofs[j] >= 0, "Signed H1 coarse map unsupported.");
          prolong_->Add(row, coarse_dofs[j], shape(j));
        }
      }
    }
    prolong_->Finalize();
    prolong_->BuildTranspose();

    const std::size_t block_entries = std::size_t(ne_) * dof_ * dof_;
    MFEM_VERIFY(block_entries <= std::numeric_limits<int>::max(),
                "Two-level block storage exceeds MFEM vector index range.");
    std::vector<double> blocks(block_entries);
    MFEM_VERIFY(!shared_patch || 5 * block_entries <= (std::size_t(1) << 26),
                "Shared patch setup exceeds bounded512MiB stencil storage.");
    std::vector<double> stencil(shared_patch ? 5 * block_entries : 0, 0.0);
    mfem::Vector basis(height), action(height);
    basis.UseDevice(true);
    action.UseDevice(true);
    int operator_probes = 0;
    for (int color = 0; color < element_color_count; ++color)
    {
      for (int j = 0; j < dof_; ++j)
      {
        double* values = basis.HostWrite();
        std::fill(values, values + height, 0.0);
        for (int e = 0; e < ne_; ++e)
        {
          if (element_colors[e] == color) { values[element_dofs_[e * dof_ + j]] = 1.0; }
        }
        op.Mult(basis, action);
        ++operator_probes;
        const double* result = action.HostRead();
        for (int e = 0; e < ne_; ++e)
        {
          if (element_colors[e] != color) { continue; }
          for (int i = 0; i < dof_; ++i)
          { blocks[(std::size_t(e) * dof_ + j) * dof_ + i] = result[element_dofs_[e * dof_ + i]]; }
        }
        if (shared_patch)
          for (int e = 0; e < ne_; ++e) for (int slot = 0; slot < 5; ++slot)
            if (element_colors[patch[e][slot]] == color)
              for (int i = 0; i < dof_; ++i)
                stencil[((std::size_t(e) * 5 + slot) * dof_ + j) * dof_ + i] =
                    result[element_dofs_[e * dof_ + i]];
      }
    }
    // Independently check a few blocks against single-element probes. This
    // catches operator couplings outside the mesh-face graph used in coloring.
    double block_probe_error = 0.0;
    for (int sample = 0; sample < std::min(ne_, 3); ++sample)
    {
      const int e = sample * (ne_ - 1) / std::max(1, std::min(ne_, 3) - 1);
      double* values = basis.HostWrite();
      std::fill(values, values + height, 0.0);
      for (int j = 0; j < dof_; ++j)
      { values[element_dofs_[e * dof_ + j]] = std::sin(0.31 * (j + 1)); }
      op.Mult(basis, action);
      ++operator_probes;
      const double* result = action.HostRead();
      mfem::Vector measured(dof_), expected(dof_);
      for (int i = 0; i < dof_; ++i)
      {
        expected(i) = result[element_dofs_[e * dof_ + i]];
        measured(i) = 0.0;
        for (int j = 0; j < dof_; ++j)
        { measured(i) += blocks[(std::size_t(e) * dof_ + j) * dof_ + i] * std::sin(0.31 * (j + 1)); }
      }
      block_probe_error = std::max(block_probe_error, RelativeDifference(measured, expected));
    }
    MFEM_VERIFY(block_probe_error < 1.0e-10,
                "Two-level block-coloring reconstruction failed: " << block_probe_error);
    if (shared_patch)
    {
      // Independently validate the full sampled face graph, not only the
      // diagonal blocks; omitted/colliding couplings invalidate patch setup.
      mfem::Vector expected(height);
      double graph_error = 0.0;
      for (int probe = 0; probe < 3; ++probe)
      {
        double* values = basis.HostWrite();
        for (int i = 0; i < height; ++i) values[i] = std::sin((0.23 + 0.13 * probe) * (i + 1));
        op.Mult(basis, action); ++operator_probes;
        const double* input = basis.HostRead(); double* output = expected.HostWrite();
        for (int e = 0; e < ne_; ++e) for (int i = 0; i < dof_; ++i)
        {
          long double sum = 0.0L;
          for (int slot = 0; slot < 5; ++slot) for (int j = 0; j < dof_; ++j)
            sum += static_cast<long double>(stencil[((std::size_t(e) * 5 + slot) * dof_ + j) * dof_ + i]) *
                   input[element_dofs_[patch[e][slot] * dof_ + j]];
          output[element_dofs_[e * dof_ + i]] = static_cast<double>(sum);
        }
        graph_error = std::max(graph_error, RelativeDifference(expected, action));
      }
      MFEM_VERIFY(graph_error < 1e-10, "Shared patch graph reconstruction failed: " << graph_error);
      ConfigureSharedPatch(shared_patch_, dof_, patch, element_dofs_, stencil, local_mode_, patch_composition_);
      std::cout << "[igr-shared-patch] event=graph-certificate relative_error=" << graph_error
                << " probes=3 colors=" << element_color_count << '\n';
    }
    else
    {
      inverse_blocks_.SetSize(static_cast<int>(block_entries));
      double* inverse_data = inverse_blocks_.HostWrite();
      for (int e = 0; e < ne_; ++e)
      {
        mfem::DenseMatrix block(dof_);
        std::copy(blocks.begin() + std::size_t(e) * dof_ * dof_,
                  blocks.begin() + std::size_t(e + 1) * dof_ * dof_, block.Data());
        InvertChecked(block);
        std::copy(block.Data(), block.Data() + dof_ * dof_,
                  inverse_data + std::size_t(e) * dof_ * dof_);
      }

      blocks_.Configure(ne_, dof_, inverse_blocks_);
    }

    Graph coarse_graph(nc);
    for (int e = 0; e < ne_; ++e)
    {
      std::set<int> columns(element_coarse[e].begin(), element_coarse[e].end());
      for (int neighbor : adjacent[e])
      { columns.insert(element_coarse[neighbor].begin(), element_coarse[neighbor].end()); }
      for (int row : element_coarse[e])
      { coarse_graph[row].insert(columns.begin(), columns.end()); }
    }
    const auto coarse_colors = ColorColumns(coarse_graph);
    const int coarse_color_count = ColorCount(coarse_colors);
    coarse_ = std::make_unique<mfem::SparseMatrix>(nc);
    coarse_rhs_.SetSize(nc);
    coarse_solution_.SetSize(nc);
    coarse_fine_.SetSize(height);
    coarse_rhs_.UseDevice(true);
    coarse_solution_.UseDevice(true);
    coarse_fine_.UseDevice(true);
    for (int color = 0; color < coarse_color_count; ++color)
    {
      double* values = coarse_solution_.HostWrite();
      for (int j = 0; j < nc; ++j) { values[j] = coarse_colors[j] == color ? 1.0 : 0.0; }
      prolong_->Mult(coarse_solution_, basis);
      op.Mult(basis, action);
      ++operator_probes;
      prolong_->MultTranspose(action, coarse_rhs_);
      const double* result = coarse_rhs_.HostRead();
      for (int i = 0; i < nc; ++i)
      {
        for (int j : coarse_graph[i])
        { if (coarse_colors[j] == color) { coarse_->Add(i, j, result[i]); } }
      }
    }
    coarse_->Finalize();
    double coarse_probe_error = 0.0;
    mfem::Vector coarse_action(nc);
    for (int probe = 0; probe < 3; ++probe)
    {
      double* values = coarse_solution_.HostWrite();
      for (int j = 0; j < nc; ++j) { values[j] = std::sin((0.23 + 0.13 * probe) * (j + 1)); }
      prolong_->Mult(coarse_solution_, basis);
      op.Mult(basis, action);
      ++operator_probes;
      prolong_->MultTranspose(action, coarse_rhs_);
      coarse_->Mult(coarse_solution_, coarse_action);
      coarse_probe_error = std::max(coarse_probe_error,
                                    RelativeDifference(coarse_action, coarse_rhs_));
    }
    MFEM_VERIFY(coarse_probe_error < 1.0e-10,
                "Two-level sparse Galerkin reconstruction failed: " << coarse_probe_error);
    if (coarse_mode == "dense")
    {
      mfem::DenseMatrix inverse;
      coarse_->ToDenseMatrix(inverse);
      InvertChecked(inverse);
      coarse_inverse_.SetSize(nc * nc);
      std::copy(inverse.Data(), inverse.Data() + nc * nc, coarse_inverse_.HostWrite());
      MFEM_VERIFY(cublasCreate(&handle_) == CUBLAS_STATUS_SUCCESS,
                  "Two-level coarse cuBLAS handle creation failed.");
      (void)coarse_inverse_.Read();
    }
    else
    {
      mfem::Vector diagonal(nc);
      coarse_->GetDiag(diagonal);
      (void)diagonal.HostRead();
      for (int i = 0; i < nc; ++i)
      { MFEM_VERIFY(diagonal(i) > 0.0 && std::isfinite(diagonal(i)), "Nonpositive H1 coarse diagonal."); }
      coarse_jacobi_ = std::make_unique<mfem::OperatorJacobiSmoother>(
        diagonal, coarse_essential_, 1.0);
      device_coarse_cg_ = DeviceKrylovRequested("DG_IGR_IGR_COARSE_CG_BACKEND",
        DeviceKrylovRequested("DG_IGR_IGR_KRYLOV_BACKEND"));
      if (CooperativeCoarseCGRequested())
      {
        MFEM_VERIFY(device_coarse_cg_,
                    "Cooperative coarse execution requires cuda-fp64 coarse CG.");
        coarse_cg_ = MakeCooperativeCGSolver(sfes.GetComm());
        cooperative_coarse_cg_ = bool(coarse_cg_);
        if (!cooperative_coarse_cg_)
        { mfem::out << "[igr-cooperative-cg] fallback=host-polled reason=unsupported-hardware\n"; }
      }
      if (device_coarse_cg_ && !coarse_cg_) { coarse_cg_ = MakeDeviceCGSolver(sfes.GetComm()); }
      else if (!device_coarse_cg_) { coarse_cg_ = std::make_unique<mfem::CGSolver>(sfes.GetComm()); }
      coarse_cg_->SetOperator(*coarse_);
      coarse_cg_->SetPreconditioner(*coarse_jacobi_);
      coarse_cg_->SetRelTol(1.0e-2);
      coarse_cg_->SetAbsTol(0.0);
      coarse_cg_->SetMaxIter(100);
      coarse_cg_->SetPrintLevel(0);
      coarse_cg_->iterative_mode = false;
    }
    if (inverse_blocks_.Size()) { (void)inverse_blocks_.Read(); }
    (void)element_dofs_.Read();
    MFEM_VERIFY(cudaDeviceSynchronize() == cudaSuccess, "Two-level setup CUDA failure.");
    const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
    std::cout << "[igr-two-level] n=" << height << " elements=" << ne_
              << " block_dof=" << dof_ << " coarse_n=" << nc
              << " coarse_mode=" << coarse_mode << " local_mode=" << local_mode_
              << " patch_composition=" << patch_composition_
              << " coarse_cg_backend=" << (device_coarse_cg_ ? "cuda-fp64" : "mfem")
              << " coarse_cg_execution=" << (cooperative_coarse_cg_ ? "cooperative" : "host-polled")
              << " coarse_order=" << coarse_order
              << " element_colors=" << element_color_count
              << " coarse_colors=" << coarse_color_count
              << " operator_probes=" << operator_probes
              << " block_inverse_bytes=" << std::size_t(inverse_blocks_.Size()) * sizeof(double)
              << " dense_coarse_bytes=" << coarse_inverse_.Size() * sizeof(double)
              << " block_probe_rel=" << block_probe_error
              << " coarse_probe_rel=" << coarse_probe_error
              << " setup_seconds=" << elapsed
              << " retained=first-operator-preconditioner" << std::endl;
  }

  ~TwoLevelInverse() override { if (handle_) { cublasDestroy(handle_); } }

  void SetOperator(const mfem::Operator& op) override
  {
    MFEM_VERIFY(op.Height() == height && op.Width() == width,
                "Cached two-level inverse cannot change dimensions.");
  }

  mfem::MemoryClass GetMemoryClass() const override
  { return mfem::MemoryClass::DEVICE; }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    MFEM_VERIFY(x.Size() == height, "Two-level inverse input size mismatch.");
    prolong_->MultTranspose(x, coarse_rhs_);
    if (coarse_mode_ == "dense")
    {
      const int nc = coarse_rhs_.Size();
      const double one = 1.0, zero = 0.0;
      MFEM_VERIFY(cublasDgemv(handle_, CUBLAS_OP_N, nc, nc, &one,
                    coarse_inverse_.Read(), nc, coarse_rhs_.Read(), 1,
                    &zero, coarse_solution_.Write(), 1) == CUBLAS_STATUS_SUCCESS,
                  "Two-level coarse Dgemv failed.");
    }
    else
    {
      coarse_cg_->Mult(coarse_rhs_, coarse_solution_);
      MFEM_VERIFY(coarse_cg_->GetConverged(),
                  "Two-level coarse CG did not reach its 1e-2 tolerance in 100 iterations.");
    }
    prolong_->Mult(coarse_solution_, coarse_fine_);
    if (local_mode_ == "element") blocks_.Mult(x, coarse_fine_, element_dofs_, y);
    else shared_patch_.Mult(x, coarse_fine_, y);
  }

private:
  int ne_, dof_;
  std::string coarse_mode_, local_mode_, patch_composition_;
  mfem::Array<int> element_dofs_;
  mfem::Array<int> coarse_essential_;
  mfem::Vector inverse_blocks_, coarse_inverse_;
  TwoLevelBlocks blocks_;
  SharedPatchBatch shared_patch_;
  std::unique_ptr<mfem::SparseMatrix> prolong_, coarse_;
  std::unique_ptr<mfem::OperatorJacobiSmoother> coarse_jacobi_;
  std::unique_ptr<mfem::IterativeSolver> coarse_cg_;
  bool device_coarse_cg_ = false;
  bool cooperative_coarse_cg_ = false;
  mutable mfem::Vector coarse_rhs_, coarse_solution_, coarse_fine_;
  cublasHandle_t handle_ = nullptr;
};
}

std::unique_ptr<mfem::Solver> MakeTwoLevelInversePreconditioner(
  const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes,
  const std::string& coarse_mode, int coarse_order)
{
  return std::make_unique<TwoLevelInverse>(op, sfes, coarse_mode, coarse_order);
}
}
}
