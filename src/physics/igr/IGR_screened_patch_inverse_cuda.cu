#include "igrlab/compat.hpp"
#include "IGR_screened_patch_inverse.hpp"
#include "IGR_face_condensed_setup.hpp"
#include <cuda_runtime.h>
#include <mma.h>
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

// One matrix-vector product is deliberately replicated across 16 columns.
// This is a research comparison: all padded/redundant products are reported.
// Matrix and vector TF32 rounding happen on chip; FP64 current-A correction
// remains the responsibility of the outer flexible solver.
__global__ void ApplyPatchRowsTF32(int dof, const int* element_dofs,
                                 const int* offsets, const int* inputs,
                                 const float* rows, const double* x, double* y)
{
  using namespace nvcuda;
  const int e = blockIdx.x, row0 = 16 * blockIdx.y, lane = threadIdx.x;
  const int begin = offsets[e], count = offsets[e + 1] - begin;
  __shared__ __align__(32) float a[16 * 8], b[8 * 16], c[16 * 16];
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int k = 0; k < count; k += 8)
  {
    for (int t = lane; t < 128; t += 32)
    {
      const int i = row0 + t / 8, j = k + t % 8;
      a[t] = (i < dof && j < count) ? wmma::__float_to_tf32(rows[std::size_t(begin) * dof + std::size_t(i) * count + j]) : 0.0f;
      const int v = k + t / 16;
      b[t] = v < count ? wmma::__float_to_tf32(static_cast<float>(x[inputs[begin + v]])) : 0.0f;
    }
    __syncwarp();
    wmma::load_matrix_sync(af, a, 8);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::mma_sync(acc, af, bf, acc);
    __syncwarp();
  }
  wmma::store_matrix_sync(c, acc, 16, wmma::mem_row_major);
  __syncwarp();
  if (lane < 16 && row0 + lane < dof)
    y[element_dofs[e * dof + row0 + lane]] = static_cast<double>(c[16 * lane]);
}

class ScreenedPatchInverse final : public mfem::Solver
{
  int dof_ = 0;
  bool fp32_ = false;
  bool tf32_ = false;
  mfem::Array<int> element_dofs_, offsets_, inputs_;
  mfem::Vector rows_;
  mfem::Array<float> rows32_;
public:
  ScreenedPatchInverse(const mfem::Operator& op,
                       mfem::ParFiniteElementSpace& sfes, int radius,
                       const std::string& precision, const std::string& kind)
    : mfem::Solver(op.Height())
  {
    const auto start = Clock::now();
    int ranks = 0;
    MPI_Comm_size(sfes.GetComm(), &ranks);
    const int ne = sfes.GetNE();
    dof_ = ne > 0 ? sfes.GetFE(0)->GetDof() : 0;
    MFEM_VERIFY(ranks == 1 && mfem::Device::Allows(mfem::Backend::CUDA),
                "Screened patch inverse requires rank-one CUDA.");
    MFEM_VERIFY((precision == "fp64" || precision == "fp32" || precision == "tf32") && radius >= 0 && radius <= 6,
                "Screened patch inverse requires fp64|fp32|tf32 and radius 0..6.");
    tf32_ = precision == "tf32";
    fp32_ = precision != "fp64";
    MFEM_VERIFY(kind == "ras" || kind == "vertex-p1" || kind == "face-condensed" || kind == "trace-pairs", "Unknown screened patch kind.");
    const bool vertex = kind == "vertex-p1";
    const bool condensed = kind == "face-condensed" || kind == "trace-pairs";
    MFEM_VERIFY(!condensed || radius == 0, "Face condensation requires radius=0.");
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
    const auto* l2 = dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl());
    MFEM_VERIFY(!vertex || (order == 1 && dof_ == 4 && radius == 0 && !fp32_ &&
                            l2->GetBasisType() == mfem::BasisType::GaussLegendre),
                "Vertex inverse requires p1 Gauss-Legendre L2 quads, fp64 and radius=0.");
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
    int max_factor_dofs = 0, vertex_groups = 0;
    long long factor_entries = 0;
    long long transform_products = 0, elimination_products = 0, composition_products = 0;
    long long interior_factor_entries = 0, face_factor_entries = 0;
    double local_inverse_error = 0.0;
    if (condensed)
    {
      std::vector<int> ids(element_dofs_.HostRead(), element_dofs_.HostRead() + height);
      auto result = BuildFaceCondensedRows(sfes, support, block_offsets, blocks, ids, kind == "trace-pairs");
      input_indices = std::move(result.inputs);
      row_offsets = std::move(result.offsets);
      packed_rows = std::move(result.rows);
      factor_entries = result.factor_entries;
      max_factor_dofs = result.max_factor_dofs;
      vertex_groups = result.group_count;
      local_inverse_error = result.local_error;
      transform_products = result.transform_products;
      elimination_products = result.elimination_products;
      composition_products = result.composition_products;
      interior_factor_entries = result.interior_factor_entries;
      face_factor_entries = result.face_factor_entries;
      for (int e = 0; e < ne; ++e)
        max_patch_dofs = std::max(max_patch_dofs, row_offsets[e + 1] - row_offsets[e]);
    }
    else if (vertex)
    {
      // Corner columns follow MFEM's cyclic local vertex order. For p1 this
      // evaluates the LGL/corner polynomial directly at the current GL nodes.
      mfem::DenseMatrix transform(4);
      const auto& nodes = sfes.GetFE(0)->GetNodes();
      for (int i = 0; i < 4; ++i)
      {
        const double x = nodes.IntPoint(i).x, y = nodes.IntPoint(i).y;
        transform(i, 0) = (1.0 - x) * (1.0 - y);
        transform(i, 1) = x * (1.0 - y);
        transform(i, 2) = x * y;
        transform(i, 3) = (1.0 - x) * y;
      }
      struct Copy { int element, corner; };
      const int nv = sfes.GetMesh()->GetNV();
      std::vector<std::vector<Copy>> copies(nv);
      std::vector<std::vector<int>> vertices(ne), copy_positions(ne, std::vector<int>(4));
      mfem::Array<int> ids;
      for (int e = 0; e < ne; ++e)
      {
        sfes.GetMesh()->GetElementVertices(e, ids);
        MFEM_VERIFY(ids.Size() == 4, "Vertex inverse requires quadrilateral topology.");
        vertices[e].assign(ids.begin(), ids.end());
        for (int c = 0; c < 4; ++c)
        {
          MFEM_VERIFY(ids[c] >= 0 && ids[c] < nv, "Invalid physical vertex index.");
          copy_positions[e][c] = static_cast<int>(copies[ids[c]].size());
          copies[ids[c]].push_back({e, c});
        }
      }
      const auto entry = [&](int e, int f, int i, int j)
      {
        const auto found = support[e].find(f);
        if (found == support[e].end()) return 0.0;
        const int slot = block_offsets[e] + static_cast<int>(std::distance(support[e].begin(), found));
        return blocks[(std::size_t(slot) * 4 + j) * 4 + i];
      };
      std::vector<mfem::DenseMatrix> inverses(nv);
      for (int v = 0; v < nv; ++v)
      {
        const int count = static_cast<int>(copies[v].size());
        if (count == 0) continue;
        MFEM_VERIFY(count <= 512, "Physical vertex exceeds bounded local factor size.");
        ++vertex_groups;
        max_factor_dofs = std::max(max_factor_dofs, count);
        factor_entries += static_cast<long long>(count) * count;
        mfem::DenseMatrix group(count);
        for (int k = 0; k < count; ++k)
          for (int l = 0; l < count; ++l)
          {
            double value = 0.0;
            for (int i = 0; i < 4; ++i)
              for (int j = 0; j < 4; ++j)
                value += transform(i, copies[v][k].corner) *
                         entry(copies[v][k].element, copies[v][l].element, i, j) *
                         transform(j, copies[v][l].corner);
            group(k, l) = value;
          }
        double asymmetry = 0.0, scale = 0.0;
        for (int k = 0; k < count; ++k)
          for (int l = 0; l < count; ++l)
          { asymmetry = std::max(asymmetry, std::abs(group(k, l) - group(l, k)));
            scale = std::max(scale, std::abs(group(k, l))); }
        MFEM_VERIFY(asymmetry <= 1.0e-10 * std::max(scale, 1.0e-300),
                    "Vertex inverse requires symmetric transformed principal blocks.");
        for (int k = 0; k < count; ++k)
          for (int l = 0; l < k; ++l)
            group(k, l) = group(l, k) = 0.5 * (group(k, l) + group(l, k));
        mfem::DenseMatrixInverse inverse(group, true);
        inverses[v].SetSize(count);
        mfem::Vector unit(count), column(count), residual(count);
        for (int l = 0; l < count; ++l)
        {
          unit = 0.0; unit(l) = 1.0; inverse.Mult(unit, column);
          group.Mult(column, residual); residual -= unit;
          local_inverse_error = std::max(local_inverse_error, residual.Norml2());
          for (int k = 0; k < count; ++k)
          { MFEM_VERIFY(std::isfinite(column(k)), "Nonfinite vertex inverse.");
            inverses[v](k, l) = column(k); }
        }
      }
      for (int center = 0; center < ne; ++center)
      {
        std::set<int> patch;
        for (int v : vertices[center])
          for (const auto& copy : copies[v]) patch.insert(copy.element);
        const int np = static_cast<int>(patch.size()) * 4;
        MFEM_VERIFY(np <= 512 && packed_rows.size() + std::size_t(np) * 4 <= (std::size_t(1) << 27),
                    "Vertex composite rows exceed bounded storage.");
        max_patch_dofs = std::max(max_patch_dofs, np);
        int k = 0;
        for (int e : patch)
        { positions[e] = k++; for (int i = 0; i < 4; ++i) input_indices.push_back(element_dofs_[e * 4 + i]); }
        const std::size_t first = packed_rows.size();
        packed_rows.resize(first + std::size_t(np) * 4, 0.0);
        for (int c = 0; c < 4; ++c)
        {
          const int v = vertices[center][c], local = copy_positions[center][c];
          for (int l = 0; l < static_cast<int>(copies[v].size()); ++l)
          {
            const auto& copy = copies[v][l];
            for (int i = 0; i < 4; ++i)
              for (int j = 0; j < 4; ++j)
                packed_rows[first + std::size_t(i) * np + positions[copy.element] * 4 + j] +=
                  transform(i, c) * inverses[v](local, l) * transform(j, copy.corner);
          }
        }
        row_offsets[center + 1] = static_cast<int>(input_indices.size());
      }
    }
    else
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
      max_factor_dofs = std::max(max_factor_dofs, np);
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
    long long executed_products = 0;
    for (int e = 0; e < ne; ++e)
    {
      const long long count = row_offsets[e + 1] - row_offsets[e];
      executed_products += tf32_ ? ((dof_ + 15) / 16) * ((count + 7) / 8) * 16LL * 16 * 8
                                : count * dof_;
    }
    std::cout << "[igr-screened-patch] event=setup n=" << height << " ne=" << ne
              << " p=" << order << " dof=" << dof_ << " radius=" << radius
              << " kind=" << kind << " vertex_groups=" << vertex_groups
              << " precision=" << precision << " arithmetic=" << (tf32_ ? "wmma_tf32" : "ordinary_cuda")
              << " accumulation=" << (fp32_ ? "fp32" : "fp64") << " colors=" << color_count
              << " operator_probes=" << color_count * dof_ + 3
              << " max_patch_dofs=" << max_patch_dofs
              << " max_factor_dofs=" << max_factor_dofs
              << " sampled_block_bytes=" << block_entries * sizeof(double)
              << " factor_entries=" << factor_entries << " row_bytes=" << row_bytes
              << " interior_factor_entries=" << interior_factor_entries
              << " face_factor_entries=" << face_factor_entries
              << " transform_products=" << transform_products
              << " elimination_products=" << elimination_products
              << " composition_products=" << composition_products
              << " index_bytes=" << index_bytes
              << " retained_device_bytes=" << row_bytes + index_bytes
              << " apply_products=" << packed_rows.size()
              << " executed_products=" << executed_products
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
    if (tf32_)
      ApplyPatchRowsTF32<<<dim3(height / dof_, (dof_ + 15) / 16), 32>>>(dof_,
        element_dofs_.Read(), offsets_.Read(), inputs_.Read(), rows32_.Read(), x.Read(), y.Write());
    else if (fp32_)
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
  const std::string& precision, const std::string& kind)
{ return std::make_unique<ScreenedPatchInverse>(op, sfes, radius, precision, kind); }
}
}
