#pragma once

#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace dg_igr
{
namespace igr_internal
{

struct FaceCondensedRows
{
  std::vector<int> offsets, inputs;
  std::vector<double> rows;
  long long factor_entries = 0;
  int max_factor_dofs = 0, group_count = 0;
  double local_error = 0.0, symmetry_error = 0.0, bubble_coupling_error = 0.0;
  long long transform_products = 0, elimination_products = 0;
  long long composition_products = 0;
  long long interior_factor_entries = 0, face_factor_entries = 0;
  int max_group_patch_dofs = 0;
};

// A bounded setup for the SPD block-LDL preconditioner of a sampled SIPG
// operator. GL coefficients equal T times the LGL coefficients. In the latter
// basis A=[D E; E^T F], with element-local bubble blocks D. E includes the
// neighboring traces coupled by normal derivatives of the bubbles. Only the
// principal groups of S=F-E^T D^-1 E are formed; no global Schur matrix is built.
//
// The composed inverse is T diag(D^-1,0) T^T + sum_g R_g^T S_gg^-1 R_g,
// where R_g selects the trace group minus E_g^T D^-1 times the bubble residual.
// Retaining this object changes only the preconditioner: a current-operator
// outer solve is still required. Composition spends additional arithmetic and
// storage to reuse the existing single-kernel row application.
inline FaceCondensedRows BuildFaceCondensedRows(
  mfem::ParFiniteElementSpace& sfes,
  const std::vector<std::set<int>>& support,
  const std::vector<int>& block_offsets, const std::vector<double>& blocks,
  const std::vector<int>& element_dofs, bool trace_pairs = false)
{
  FaceCondensedRows out;
  const int ne = sfes.GetNE();
  MFEM_VERIFY(ne > 0 && static_cast<int>(support.size()) == ne &&
              static_cast<int>(block_offsets.size()) == ne + 1,
              "Invalid face-condensation element graph.");
  const int p = sfes.GetFE(0)->GetOrder(), q = sfes.GetFE(0)->GetDof();
  const auto* collection = dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl());
  MFEM_VERIFY(collection && collection->GetBasisType() == mfem::BasisType::GaussLegendre &&
              p >= 1 && p <= 7 && q == (p + 1) * (p + 1) &&
              sfes.GetMesh()->Dimension() == 2 && !sfes.GetMesh()->Nonconforming() &&
              element_dofs.size() == std::size_t(ne) * q &&
              blocks.size() == std::size_t(block_offsets.back()) * q * q,
              "Face condensation requires uniform conforming p1-p7 GL L2 quads.");
  int ranks = 0;
  MPI_Comm_size(sfes.GetComm(), &ranks);
  MFEM_VERIFY(ranks == 1, "Face condensation currently requires rank one.");

  mfem::L2_FECollection lgl_collection(p, 2, mfem::BasisType::GaussLobatto);
  const auto* lgl = lgl_collection.FiniteElementForGeometry(mfem::Geometry::SQUARE);
  mfem::DenseMatrix transform(q);
  mfem::Vector shape(q);
  std::vector<int> interior, trace;
  for (int i = 0; i < q; ++i)
  {
    lgl->CalcShape(sfes.GetFE(0)->GetNodes().IntPoint(i), shape);
    for (int j = 0; j < q; ++j) transform(i, j) = shape(j);
    const auto& point = lgl->GetNodes().IntPoint(i);
    const bool boundary = point.x < 1e-12 || point.x > 1.0 - 1e-12 ||
                          point.y < 1e-12 || point.y > 1.0 - 1e-12;
    (boundary ? trace : interior).push_back(i);
  }
  const int ni = static_cast<int>(interior.size());
  MFEM_VERIFY(ni == (p - 1) * (p - 1), "Unexpected tensor bubble ordering.");

  std::vector<double> transformed(blocks.size());
  for (int slot = 0; slot < block_offsets.back(); ++slot)
  {
    mfem::DenseMatrix original(q), product(q), congruent(q);
    for (int j = 0; j < q; ++j)
      for (int i = 0; i < q; ++i)
        original(i, j) = blocks[(std::size_t(slot) * q + j) * q + i];
    mfem::Mult(original, transform, product);
    mfem::MultAtB(transform, product, congruent);
    for (int j = 0; j < q; ++j)
      for (int i = 0; i < q; ++i)
        transformed[(std::size_t(slot) * q + j) * q + i] = congruent(i, j);
  }
  out.transform_products = 2LL * block_offsets.back() * q * q * q;
  const auto entry = [&](int e, int f, int i, int j)
  {
    const auto found = support[e].find(f);
    if (found == support[e].end()) return 0.0;
    const int slot = block_offsets[e] +
      static_cast<int>(std::distance(support[e].begin(), found));
    return transformed[(std::size_t(slot) * q + j) * q + i];
  };
  double scale = 0.0, asymmetry = 0.0, cross_bubble = 0.0;
  for (int e = 0; e < ne; ++e)
    for (int f : support[e])
    {
      MFEM_VERIFY(support[f].count(e), "Face graph is not symmetric.");
      for (int j = 0; j < q; ++j)
        for (int i = 0; i < q; ++i)
        {
          const double a = entry(e, f, i, j);
          MFEM_VERIFY(std::isfinite(a), "Nonfinite transformed SIPG entry.");
          scale = std::max(scale, std::abs(a));
          asymmetry = std::max(asymmetry, std::abs(a - entry(f, e, j, i)));
        }
      if (f != e)
        for (int i : interior) for (int j : interior)
          cross_bubble = std::max(cross_bubble, std::abs(entry(e, f, i, j)));
    }
  out.symmetry_error = asymmetry / std::max(scale, 1e-300);
  out.bubble_coupling_error = cross_bubble / std::max(scale, 1e-300);
  MFEM_VERIFY(out.symmetry_error < 1e-10 && out.bubble_coupling_error < 1e-10,
              "Face condensation requires symmetric SIPG with element-local bubbles: "
              << out.symmetry_error << ", " << out.bubble_coupling_error);

  const auto inverse_spd = [&](mfem::DenseMatrix& a, mfem::DenseMatrix& inverse)
  {
    const int n = a.Height();
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j)
        a(i, j) = a(j, i) = 0.5 * (a(i, j) + a(j, i));
    inverse.SetSize(n);
    mfem::DenseMatrixInverse factor(a, true);
    mfem::Vector unit(n), column(n), residual(n);
    for (int j = 0; j < n; ++j)
    {
      unit = 0.0; unit(j) = 1.0;
      factor.Mult(unit, column); a.Mult(column, residual); residual -= unit;
      out.local_error = std::max(out.local_error, residual.Norml2());
      for (int i = 0; i < n; ++i)
      {
        MFEM_VERIFY(std::isfinite(column(i)), "Nonfinite condensed inverse.");
        inverse(i, j) = column(i);
      }
    }
    // Symmetric storage keeps separately rounded solve columns from creating
    // an avoidable antisymmetric part in the composed CG preconditioner.
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < i; ++j)
        inverse(i, j) = inverse(j, i) = 0.5 * (inverse(i, j) + inverse(j, i));
    out.factor_entries += static_cast<long long>(n) * n;
    out.max_factor_dofs = std::max(out.max_factor_dofs, n);
  };
  std::vector<mfem::DenseMatrix> dinverse(ne);
  if (ni)
    for (int e = 0; e < ne; ++e)
    {
      mfem::DenseMatrix d(ni);
      for (int j = 0; j < ni; ++j)
        for (int i = 0; i < ni; ++i)
          d(i, j) = entry(e, e, interior[i], interior[j]);
      inverse_spd(d, dinverse[e]);
      out.interior_factor_entries += static_cast<long long>(ni) * ni;
    }

  // A vertex group contains all its discontinuous copies. An edge group
  // contains the interior edge nodes from its one or two incident elements;
  // corners belong only to the vertex groups, so the groups are disjoint.
  struct Copy { int element, local; };
  using Key = std::tuple<int, int, int>;
  std::map<Key, std::vector<Copy>> groups;
  mfem::Array<int> vertices;
  for (int e = 0; e < ne; ++e)
  {
    MFEM_VERIFY(sfes.GetFE(e)->GetGeomType() == mfem::Geometry::SQUARE &&
                sfes.GetFE(e)->GetOrder() == p && sfes.GetFE(e)->GetDof() == q,
                "Face condensation requires uniform quadrilaterals.");
    sfes.GetMesh()->GetElementVertices(e, vertices);
    MFEM_VERIFY(vertices.Size() == 4, "Invalid quadrilateral vertex topology.");
    for (int t : trace)
    {
      const auto& point = lgl->GetNodes().IntPoint(t);
      const bool left = point.x < 1e-12, right = point.x > 1.0 - 1e-12;
      const bool bottom = point.y < 1e-12, top = point.y > 1.0 - 1e-12;
      Key key;
      if ((left || right) && (bottom || top))
      {
        const int corner = bottom ? (left ? 0 : 1) : (left ? 3 : 2);
        key = Key(0, vertices[corner], -1);
      }
      else
      {
        const int c0 = bottom ? 0 : top ? 3 : left ? 0 : 1;
        const int c1 = bottom ? 1 : top ? 2 : left ? 3 : 2;
        key = Key(1, std::min(vertices[c0], vertices[c1]),
                  std::max(vertices[c0], vertices[c1]));
      }
      groups[key].push_back({e, t});
    }
  }
  if (trace_pairs)
  {
    // Each face-adjacent pair contains all trace DOFs of both elements.
    // Boundary elements appear in as many groups as they have neighbors;
    // an isolated element gets one singleton group. The factors below use
    // unweighted Schur principals; multiplicity weights act on both sides
    // of each inverse, preserving the symmetric additive construction.
    groups.clear();
    for (int e = 0; e < ne; ++e)
    {
      bool connected = false;
      for (int f : support[e])
      {
        if (f == e) continue;
        connected = true;
        if (f < e) continue;
        auto& copies = groups[Key(2, e, f)];
        for (int t : trace) copies.push_back({e, t});
        for (int t : trace) copies.push_back({f, t});
      }
      if (!connected)
      {
        auto& copies = groups[Key(2, e, e)];
        for (int t : trace) copies.push_back({e, t});
      }
    }
  }
  std::vector<int> multiplicity(std::size_t(ne) * q, 0);
  for (const auto& item : groups)
    for (const auto& copy : item.second) ++multiplicity[copy.element * q + copy.local];
  for (int e = 0; e < ne; ++e)
    for (int t : trace)
      MFEM_VERIFY(multiplicity[e * q + t] > 0, "Uncovered condensed trace DOF.");
  out.group_count = static_cast<int>(groups.size());

  constexpr std::size_t max_entries = std::size_t(1) << 27;
  std::size_t composed_entries = 0;
  std::vector<std::map<int, std::vector<double>>> composed(ne);
  const auto row_block = [&](int e, int f) -> std::vector<double>&
  {
    auto found = composed[e].find(f);
    if (found == composed[e].end())
    {
      MFEM_VERIFY(composed_entries + std::size_t(q) * q <= max_entries,
                  "Composed face inverse exceeds bounded one-GiB row storage.");
      composed_entries += std::size_t(q) * q;
      found = composed[e].emplace(f, std::vector<double>(std::size_t(q) * q, 0.0)).first;
    }
    return found->second;
  };
  if (ni)
    for (int e = 0; e < ne; ++e)
    {
      mfem::DenseMatrix ti(q, ni), work(q, ni), base(q);
      for (int j = 0; j < ni; ++j)
        for (int i = 0; i < q; ++i) ti(i, j) = transform(i, interior[j]);
      mfem::Mult(ti, dinverse[e], work); mfem::MultABt(work, ti, base);
      auto& target = row_block(e, e);
      for (int i = 0; i < q; ++i)
        for (int j = 0; j < q; ++j) target[std::size_t(i) * q + j] += base(i, j);
      out.composition_products += static_cast<long long>(q) * ni * ni +
                                  static_cast<long long>(q) * q * ni;
    }

  for (const auto& item : groups)
  {
    const auto& copies = item.second;
    const int m = static_cast<int>(copies.size());
    MFEM_VERIFY(m > 0 && m <= 512, "Unbounded condensed face group.");
    std::set<int> patch_set;
    for (const auto& copy : copies)
    {
      patch_set.insert(copy.element);
      if (ni) patch_set.insert(support[copy.element].begin(), support[copy.element].end());
    }
    const std::vector<int> patch(patch_set.begin(), patch_set.end());
    const int np = static_cast<int>(patch.size()) * q;
    MFEM_VERIFY(np <= 8192, "Condensed group patch exceeds bounded local setup.");
    out.max_group_patch_dofs = std::max(out.max_group_patch_dofs, np);
    mfem::DenseMatrix schur(m), r(m, np);
    r = 0.0;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        schur(i, j) = entry(copies[i].element, copies[j].element,
                            copies[i].local, copies[j].local);
    for (int k = 0; k < static_cast<int>(patch.size()); ++k)
    {
      const int e = patch[k];
      mfem::DenseMatrix coupling(ni, m), response(ni, m);
      if (ni)
      {
        for (int j = 0; j < m; ++j)
          for (int i = 0; i < ni; ++i)
            coupling(i, j) = entry(e, copies[j].element, interior[i], copies[j].local);
        mfem::Mult(dinverse[e], coupling, response);
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j)
            for (int a = 0; a < ni; ++a)
              schur(i, j) -= coupling(a, i) * response(a, j);
        out.elimination_products += static_cast<long long>(ni) * ni * m +
                                    static_cast<long long>(m) * m * ni;
      }
      for (int j = 0; j < m; ++j)
        for (int a = 0; a < q; ++a)
        {
          double value = copies[j].element == e ? transform(a, copies[j].local) : 0.0;
          for (int i = 0; i < ni; ++i) value -= response(i, j) * transform(a, interior[i]);
          r(j, k * q + a) = value;
        }
      out.composition_products += static_cast<long long>(m) * q * ni;
    }
    if (trace_pairs)
      for (int j = 0; j < m; ++j)
      {
        const double weight = 1.0 / std::sqrt(static_cast<double>(
          multiplicity[copies[j].element * q + copies[j].local]));
        for (int a = 0; a < np; ++a) r(j, a) *= weight;
        out.composition_products += np;
      }
    mfem::DenseMatrix sinverse(m), weighted(m, np);
    inverse_spd(schur, sinverse);
    out.face_factor_entries += static_cast<long long>(m) * m;
    mfem::Mult(sinverse, r, weighted);
    out.composition_products += static_cast<long long>(m) * m * np;
    // Accumulate matching transpose blocks together. This realizes exactly
    // the symmetric R^T W R action without materializing a global dense map.
    for (int k = 0; k < static_cast<int>(patch.size()); ++k)
      for (int l = k; l < static_cast<int>(patch.size()); ++l)
      {
        auto& forward = row_block(patch[k], patch[l]);
        auto& backward = row_block(patch[l], patch[k]);
        for (int i = 0; i < q; ++i)
          for (int j = (k == l ? i : 0); j < q; ++j)
          {
            double value = 0.0;
            for (int a = 0; a < m; ++a)
              value += r(a, k * q + i) * weighted(a, l * q + j);
            forward[std::size_t(i) * q + j] += value;
            if (k != l || i != j) backward[std::size_t(j) * q + i] += value;
            out.composition_products += m;
          }
      }
  }
  MFEM_VERIFY(out.local_error < 1e-8, "Condensed factor inverse residual failed: " << out.local_error);
  out.offsets.resize(ne + 1, 0);
  out.rows.reserve(composed_entries);
  for (int e = 0; e < ne; ++e)
  {
    const int count = static_cast<int>(composed[e].size()) * q;
    MFEM_VERIFY(out.inputs.size() + count <= std::size_t(std::numeric_limits<int>::max()),
                "Composed face input indexing overflow.");
    for (const auto& item : composed[e])
      for (int j = 0; j < q; ++j) out.inputs.push_back(element_dofs[item.first * q + j]);
    for (int i = 0; i < q; ++i)
      for (const auto& item : composed[e])
        for (int j = 0; j < q; ++j)
        {
          const double value = item.second[std::size_t(i) * q + j];
          MFEM_VERIFY(std::isfinite(value), "Nonfinite composed face row.");
          out.rows.push_back(value);
        }
    out.offsets[e + 1] = static_cast<int>(out.inputs.size());
  }
  return out;
}

} // namespace igr_internal
} // namespace dg_igr
