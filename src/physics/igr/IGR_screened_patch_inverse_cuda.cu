#include "IGR_screened_patch_inverse.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

namespace dg_igr
{
namespace igr_internal
{
namespace
{
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point start)
{ return std::chrono::duration<double>(Clock::now() - start).count(); }

template <typename Real>
__global__ void ApplyPatchRows(int n, int dof, const int* element_dofs,
                               const int* offsets, const int* inputs,
                               const Real* rows, const double* x, double* y)
{
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = thread / 32, lane = thread % 32;
  if (row >= n) return;
  const int e = row / dof, i = row % dof;
  const int begin = offsets[e], count = offsets[e + 1] - begin;
  const std::size_t row_begin = std::size_t(begin) * dof + std::size_t(i) * count;
  Real sum = Real(0.0);
  for (int j = lane; j < count; j += 32)
    sum += rows[row_begin + j] * static_cast<Real>(x[inputs[begin + j]]);
  for (int delta = 16; delta; delta /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, delta);
  if (lane == 0) y[element_dofs[row]] = static_cast<double>(sum);
}

class ScreenedPatchInverse final : public mfem::Solver
{
  int dof_ = 0;
  bool fp32_ = false;
  mfem::Array<int> element_dofs_, offsets_, inputs_;
  mfem::Vector rows_;
  mfem::Array<float> rows32_;
public:
  ScreenedPatchInverse(const mfem::Operator& op,
                       mfem::ParFiniteElementSpace& sfes, int radius,
                       const std::string& precision) : mfem::Solver(op.Height())
  {
    const auto start = Clock::now();
    int ranks = 0;
    MPI_Comm_size(sfes.GetComm(), &ranks);
    const int ne = sfes.GetNE();
    dof_ = ne > 0 ? sfes.GetFE(0)->GetDof() : 0;
    MFEM_VERIFY(ranks == 1 && mfem::Device::Allows(mfem::Backend::CUDA),
                "Screened patch inverse requires rank-one CUDA.");
    MFEM_VERIFY((precision == "fp64" || precision == "fp32") && radius >= 0 && radius <= 6,
                "Screened patch inverse requires fp64|fp32 and radius 0..6.");
    fp32_ = precision == "fp32";
    MFEM_VERIFY(dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl()) &&
                  sfes.GetVDim() == 1 && sfes.GetMesh()->Dimension() == 2 &&
                  ne > 0 && dof_ > 0 && dof_ <= 256 && op.Width() == height &&
                  height == sfes.GetVSize() && height == sfes.GetTrueVSize() &&
                  static_cast<long long>(ne) * dof_ == height &&
                  !sfes.GetMesh()->Nonconforming(),
                "Screened patch inverse requires conforming-mesh uniform scalar L2 quads.");
    element_dofs_.SetSize(height);
    std::vector<char> seen(height, 0);
    mfem::Array<int> dofs;
    const int order = sfes.GetFE(0)->GetOrder();
    for (int e = 0; e < ne; ++e)
    {
      const auto& fe = *sfes.GetFE(e);
      MFEM_VERIFY(fe.GetGeomType() == mfem::Geometry::SQUARE &&
                    fe.GetDof() == dof_ && fe.GetOrder() == order &&
                    fe.GetMapType() == mfem::FiniteElement::VALUE &&
                    fe.GetNodes().GetNPoints() == dof_,
                  "Screened patch inverse requires uniform nodal VALUE elements.");
      sfes.GetElementDofs(e, dofs);
      for (int i = 0; i < dof_; ++i)
      {
        MFEM_VERIFY(dofs[i] >= 0 && dofs[i] < height && !seen[dofs[i]],
                    "Screened patch inverse requires disjoint unsigned element DOFs.");
        seen[dofs[i]] = 1;
        element_dofs_[e * dof_ + i] = dofs[i];
      }
    }
    // Each row support includes self and face neighbors. Column colors must
    // differ throughout each support, hence coloring the distance-two graph.
    std::vector<std::set<int>> support(ne), conflict(ne);
    for (int e = 0; e < ne; ++e) support[e].insert(e);
    for (int f = 0; f < sfes.GetMesh()->GetNumFaces(); ++f)
    {
      int a, b;
      sfes.GetMesh()->GetFaceElements(f, &a, &b);
      if (a >= 0 && b >= 0 && a != b)
      { support[a].insert(b); support[b].insert(a); }
    }
    for (const auto& row : support)
      for (int a : row) for (int b : row) if (a != b) conflict[a].insert(b);
    std::vector<int> colors(ne, -1), used(ne, -1), block_offsets(ne + 1, 0);
    int color_count = 0;
    for (int e = 0; e < ne; ++e)
    {
      for (int j : conflict[e]) if (colors[j] >= 0) used[colors[j]] = e;
      int color = 0;
      while (color < ne && used[color] == e) ++color;
      MFEM_VERIFY(color < ne, "Invalid screened block graph coloring.");
      colors[e] = color;
      color_count = std::max(color_count, color + 1);
      block_offsets[e + 1] = block_offsets[e] + static_cast<int>(support[e].size());
    }
    const std::size_t block_entries = std::size_t(block_offsets.back()) * dof_ * dof_;
    MFEM_VERIFY(block_entries <= (std::size_t(1) << 27),
                "Screened face-block samples exceed bounded index storage.");
    std::vector<double> blocks(block_entries, 0.0);
    mfem::Vector basis(height), action(height);
    basis.UseDevice(true); action.UseDevice(true);
    const auto sample_start = Clock::now();
    for (int color = 0; color < color_count; ++color)
      for (int j = 0; j < dof_; ++j)
      {
        double* values = basis.HostWrite();
        std::fill(values, values + height, 0.0);
        for (int e = 0; e < ne; ++e)
          if (colors[e] == color) values[element_dofs_[e * dof_ + j]] = 1.0;
        op.Mult(basis, action);
        const double* result = action.HostRead();
        for (int e = 0; e < ne; ++e)
        {
          int slot = block_offsets[e];
          for (int neighbor : support[e])
          {
            if (colors[neighbor] == color)
              for (int i = 0; i < dof_; ++i)
                blocks[(std::size_t(slot) * dof_ + j) * dof_ + i] =
                  result[element_dofs_[e * dof_ + i]];
            ++slot;
          }
        }
      }
    const double sample_seconds = Seconds(sample_start);
    const auto check_start = Clock::now();
    double reconstruction_error = 0.0;
    for (int probe = 0; probe < 3; ++probe)
    {
      double* values = basis.HostWrite();
      for (int i = 0; i < height; ++i)
        values[i] = std::sin((0.23 + 0.13 * probe) * (i + 1)) +
                    0.17 * std::cos((0.51 + 0.07 * probe) * (i + 1));
      op.Mult(basis, action);
      const double* measured = action.HostRead();
      values = basis.HostWrite();
      long double error = 0.0L, scale = 0.0L;
      for (int e = 0; e < ne; ++e)
        for (int i = 0; i < dof_; ++i)
        {
          long double expected = 0.0L;
          int slot = block_offsets[e];
          for (int neighbor : support[e])
          {
            for (int j = 0; j < dof_; ++j)
              expected += blocks[(std::size_t(slot) * dof_ + j) * dof_ + i] *
                          static_cast<long double>(values[element_dofs_[neighbor * dof_ + j]]);
            ++slot;
          }
          const double actual = measured[element_dofs_[e * dof_ + i]];
          MFEM_VERIFY(std::isfinite(actual) && std::isfinite(expected),
                      "Nonfinite screened graph reconstruction.");
          error += (expected - actual) * (expected - actual);
          scale += static_cast<long double>(actual) * actual;
        }
      reconstruction_error = std::max(reconstruction_error,
        std::sqrt(static_cast<double>(error / std::max(scale, 1.0e-300L))));
    }
    MFEM_VERIFY(reconstruction_error < 1.0e-10,
                "Current IGR operator is not reconstructed by the sampled face graph: "
                << reconstruction_error);
    const double check_seconds = Seconds(check_start);
    const auto inverse_start = Clock::now();
    std::vector<int> input_indices, row_offsets(ne + 1, 0), visited(ne, -1),
                     positions(ne, -1);
    std::vector<double> packed_rows;
    int max_patch_dofs = 0;
    long long factor_entries = 0;
    double local_inverse_error = 0.0;
    for (int center = 0; center < ne; ++center)
    {
      std::vector<int> patch{center}, hops{0};
      visited[center] = center;
      for (std::size_t k = 0; k < patch.size(); ++k)
        if (hops[k] < radius)
          for (int next : support[patch[k]]) if (visited[next] != center)
          { visited[next] = center; patch.push_back(next); hops.push_back(hops[k] + 1); }
      const int np = static_cast<int>(patch.size()) * dof_;
      MFEM_VERIFY(np <= 512, "Screened patch exceeds 512 DOFs; reduce radius or p.");
      MFEM_VERIFY(packed_rows.size() + std::size_t(np) * dof_ <= (std::size_t(1) << 27),
                  "Screened inverse rows exceed bounded 1 GiB storage.");
      max_patch_dofs = std::max(max_patch_dofs, np);
      factor_entries += static_cast<long long>(np) * np;
      for (int k = 0; k < static_cast<int>(patch.size()); ++k) positions[patch[k]] = k;
      mfem::DenseMatrix transpose(np);
      transpose = 0.0;
      for (int k = 0; k < static_cast<int>(patch.size()); ++k)
      {
        const int e = patch[k];
        int slot = block_offsets[e];
        for (int neighbor : support[e])
        {
          if (visited[neighbor] == center)
            for (int j = 0; j < dof_; ++j)
              for (int i = 0; i < dof_; ++i)
                transpose(positions[neighbor] * dof_ + j, k * dof_ + i) =
                  blocks[(std::size_t(slot) * dof_ + j) * dof_ + i];
          ++slot;
        }
        for (int i = 0; i < dof_; ++i) input_indices.push_back(element_dofs_[e * dof_ + i]);
      }
      mfem::DenseMatrixInverse inverse(transpose);
      mfem::Vector unit(np), row(np), residual(np);
      for (int i = 0; i < dof_; ++i)
      {
        unit = 0.0; unit(i) = 1.0; // center is first in the BFS patch
        inverse.Mult(unit, row);
        if (i == 0 || i == dof_ - 1)
        {
          transpose.Mult(row, residual); residual -= unit;
          local_inverse_error = std::max(local_inverse_error, residual.Norml2());
        }
        for (int j = 0; j < np; ++j)
        {
          MFEM_VERIFY(std::isfinite(row(j)), "Nonfinite screened local inverse row.");
          packed_rows.push_back(row(j));
        }
      }
      MFEM_VERIFY(input_indices.size() <= std::numeric_limits<int>::max() &&
                    packed_rows.size() <= std::numeric_limits<int>::max(),
                  "Screened patch storage exceeds bounded index range.");
      row_offsets[center + 1] = static_cast<int>(input_indices.size());
    }
    MFEM_VERIFY(local_inverse_error < 1.0e-8,
                "Screened local LU inverse residual failed: " << local_inverse_error);
    const double inverse_seconds = Seconds(inverse_start);
    const auto upload_start = Clock::now();
    offsets_.SetSize(ne + 1); inputs_.SetSize(static_cast<int>(input_indices.size()));
    std::copy(row_offsets.begin(), row_offsets.end(), offsets_.HostWrite());
    std::copy(input_indices.begin(), input_indices.end(), inputs_.HostWrite());
    if (fp32_)
    {
      rows32_.SetSize(static_cast<int>(packed_rows.size()));
      float* converted = rows32_.HostWrite();
      for (std::size_t i = 0; i < packed_rows.size(); ++i)
      {
        converted[i] = static_cast<float>(packed_rows[i]);
        MFEM_VERIFY(std::isfinite(converted[i]), "Screened inverse row overflows FP32.");
      }
      (void)rows32_.Read();
    }
    else
    {
      rows_.SetSize(static_cast<int>(packed_rows.size())); rows_.UseDevice(true);
      std::copy(packed_rows.begin(), packed_rows.end(), rows_.HostWrite());
      (void)rows_.Read();
    }
    (void)element_dofs_.Read(); (void)offsets_.Read(); (void)inputs_.Read();
    MFEM_VERIFY(cudaDeviceSynchronize() == cudaSuccess, "Screened inverse upload failed.");
    const double upload_seconds = Seconds(upload_start);
    const long long row_bytes = static_cast<long long>(packed_rows.size()) *
                                (fp32_ ? sizeof(float) : sizeof(double));
    const long long index_bytes = static_cast<long long>(height + ne + 1 + input_indices.size()) * sizeof(int);
    std::cout << "[igr-screened-patch] event=setup n=" << height << " ne=" << ne
              << " p=" << order << " dof=" << dof_ << " radius=" << radius
              << " precision=" << precision << " arithmetic=ordinary_cuda"
              << " accumulation=" << precision << " colors=" << color_count
              << " operator_probes=" << color_count * dof_ + 3
              << " max_patch_dofs=" << max_patch_dofs
              << " sampled_block_bytes=" << block_entries * sizeof(double)
              << " factor_entries=" << factor_entries << " row_bytes=" << row_bytes
              << " index_bytes=" << index_bytes
              << " retained_device_bytes=" << row_bytes + index_bytes
              << " apply_products=" << packed_rows.size()
              << " graph_error=" << reconstruction_error
              << " local_inverse_error=" << local_inverse_error
              << " sampling_seconds=" << sample_seconds << " validation_seconds=" << check_seconds
              << " inversion_seconds=" << inverse_seconds << " upload_seconds=" << upload_seconds
              << " conversion_upload_seconds=" << upload_seconds
              << " setup_seconds=" << Seconds(start) << " reuse=first_operator_preconditioner_only"
              << std::endl;
  }
  void SetOperator(const mfem::Operator& op) override
  { MFEM_VERIFY(op.Height() == height && op.Width() == width, "Screened inverse size changed."); }
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    MFEM_VERIFY(x.Size() == width, "Screened inverse input size mismatch.");
    y.SetSize(height);
    y.UseDevice(true);
    if (fp32_)
      ApplyPatchRows<float><<<(height + 7) / 8, 256>>>(height, dof_, element_dofs_.Read(),
        offsets_.Read(), inputs_.Read(), rows32_.Read(), x.Read(), y.Write());
    else
      ApplyPatchRows<double><<<(height + 7) / 8, 256>>>(height, dof_, element_dofs_.Read(),
        offsets_.Read(), inputs_.Read(), rows_.Read(), x.Read(), y.Write());
    MFEM_VERIFY(cudaGetLastError() == cudaSuccess, "Screened patch application launch failed.");
  }
};
}

std::unique_ptr<mfem::Solver> MakeScreenedPatchInversePreconditioner(
  const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes, int radius,
  const std::string& precision)
{ return std::make_unique<ScreenedPatchInverse>(op, sfes, radius, precision); }
}
}
