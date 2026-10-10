#pragma once

#include "mfem.hpp"
#include <map>
#include <tuple>
#include <vector>

// Deliberately dense, independently assembled reference for tiny tests only.
// It constructs the entire congruence and Schur complement before applying
// block LDL; production setup constructs only the individual Schur groups.
inline void FaceCondensedOracle(const mfem::DenseMatrix& a,
                               mfem::ParFiniteElementSpace& fes,
                               const mfem::Vector& x, mfem::Vector& y,
                               bool trace_pairs = false)
{
  const int ne = fes.GetNE(), n = fes.GetVSize();
  const int p = fes.GetFE(0)->GetOrder(), q = fes.GetFE(0)->GetDof();
  mfem::L2_FECollection lgl_collection(p, 2, mfem::BasisType::GaussLobatto);
  const auto* lgl = lgl_collection.FiniteElementForGeometry(mfem::Geometry::SQUARE);
  mfem::DenseMatrix transform(n), product(n), congruent(n);
  transform = 0.0;
  std::vector<int> bubbles, traces;
  std::map<std::tuple<int, int, int>, std::vector<int>> groups;
  mfem::Array<int> dofs, vertices;
  mfem::Vector shape(q);
  for (int e = 0; e < ne; ++e)
  {
    fes.GetElementDofs(e, dofs); fes.GetMesh()->GetElementVertices(e, vertices);
    for (int i = 0; i < q; ++i)
    {
      lgl->CalcShape(fes.GetFE(e)->GetNodes().IntPoint(i), shape);
      for (int j = 0; j < q; ++j) transform(dofs[i], e * q + j) = shape(j);
    }
    for (int j = 0; j < q; ++j)
    {
      const auto& point = lgl->GetNodes().IntPoint(j);
      const bool left = point.x < 1e-12, right = point.x > 1.0 - 1e-12;
      const bool bottom = point.y < 1e-12, top = point.y > 1.0 - 1e-12;
      if (!(left || right || bottom || top))
      { bubbles.push_back(e * q + j); continue; }
      const int position = static_cast<int>(traces.size());
      traces.push_back(e * q + j);
      std::tuple<int, int, int> key;
      if ((left || right) && (bottom || top))
      {
        const int corner = bottom ? (left ? 0 : 1) : (left ? 3 : 2);
        key = std::make_tuple(0, vertices[corner], -1);
      }
      else
      {
        const int c0 = bottom ? 0 : top ? 3 : left ? 0 : 1;
        const int c1 = bottom ? 1 : top ? 2 : left ? 3 : 2;
        key = std::make_tuple(1, std::min(vertices[c0], vertices[c1]),
                             std::max(vertices[c0], vertices[c1]));
      }
      groups[key].push_back(position);
    }
  }
  if (trace_pairs)
  {
    groups.clear();
    std::vector<std::vector<int>> element_traces(ne);
    for (int t = 0; t < static_cast<int>(traces.size()); ++t)
      element_traces[traces[t] / q].push_back(t);
    std::vector<bool> connected(ne, false);
    for (int face = 0; face < fes.GetMesh()->GetNumFaces(); ++face)
    {
      int first, second; fes.GetMesh()->GetFaceElements(face, &first, &second);
      if (first < 0 || second < 0 || first == second) continue;
      if (first > second) std::swap(first, second);
      auto& ids = groups[std::make_tuple(2, first, second)];
      // A pair connected through more than one topological face still has
      // one group, matching the element adjacency graph used in production.
      if (!ids.empty()) continue;
      ids = element_traces[first];
      ids.insert(ids.end(), element_traces[second].begin(), element_traces[second].end());
      connected[first] = connected[second] = true;
    }
    for (int e = 0; e < ne; ++e)
      if (!connected[e]) groups[std::make_tuple(2, e, e)] = element_traces[e];
  }
  mfem::Mult(a, transform, product); mfem::MultAtB(transform, product, congruent);
  const int ni = static_cast<int>(bubbles.size()), nt = static_cast<int>(traces.size());
  std::vector<int> multiplicity(nt, 0);
  for (const auto& item : groups) for (int i : item.second) ++multiplicity[i];
  for (int count : multiplicity) MFEM_VERIFY(count > 0, "Uncovered oracle trace DOF.");
  mfem::DenseMatrix d(ni), e(ni, nt), response(ni, nt), s(nt);
  mfem::Vector transformed_rhs(n), ri(ni), rt(nt), u(ni), reduced(nt), z(nt);
  transform.MultTranspose(x, transformed_rhs);
  for (int i = 0; i < ni; ++i)
  {
    ri(i) = transformed_rhs(bubbles[i]);
    for (int j = 0; j < ni; ++j) d(i, j) = congruent(bubbles[i], bubbles[j]);
    for (int j = 0; j < nt; ++j) e(i, j) = congruent(bubbles[i], traces[j]);
  }
  for (int i = 0; i < nt; ++i)
  {
    rt(i) = transformed_rhs(traces[i]);
    for (int j = 0; j < nt; ++j) s(i, j) = congruent(traces[i], traces[j]);
  }
  if (ni)
  {
    // A full D solve is intentionally used here, independently of the
    // production assumption/check that bubbles are element-block diagonal.
    mfem::DenseMatrixInverse interior_solve(d);
    interior_solve.Mult(ri, u);
    mfem::Vector rhs_column(ni), solved_column(ni);
    for (int j = 0; j < nt; ++j)
    {
      for (int i = 0; i < ni; ++i) rhs_column(i) = e(i, j);
      interior_solve.Mult(rhs_column, solved_column);
      for (int i = 0; i < ni; ++i) response(i, j) = solved_column(i);
    }
    for (int i = 0; i < nt; ++i)
      for (int j = 0; j < nt; ++j)
        for (int k = 0; k < ni; ++k) s(i, j) -= e(k, i) * response(k, j);
  }
  reduced = rt;
  for (int i = 0; i < nt; ++i)
    for (int j = 0; j < ni; ++j) reduced(i) -= e(j, i) * u(j);
  z = 0.0;
  for (const auto& item : groups)
  {
    const auto& ids = item.second;
    const int m = static_cast<int>(ids.size());
    mfem::DenseMatrix principal(m); mfem::Vector rhs(m), solution(m);
    for (int i = 0; i < m; ++i)
    {
      const double weight = trace_pairs ? 1.0 / std::sqrt(static_cast<double>(multiplicity[ids[i]])) : 1.0;
      rhs(i) = weight * reduced(ids[i]);
      for (int j = 0; j < m; ++j) principal(i, j) = s(ids[i], ids[j]);
    }
    mfem::DenseMatrixInverse solve(principal); solve.Mult(rhs, solution);
    for (int i = 0; i < m; ++i)
    {
      const double weight = trace_pairs ? 1.0 / std::sqrt(static_cast<double>(multiplicity[ids[i]])) : 1.0;
      z(ids[i]) += weight * solution(i);
    }
  }
  mfem::Vector solution(n); solution = 0.0;
  for (int i = 0; i < nt; ++i) solution(traces[i]) = z(i);
  for (int i = 0; i < ni; ++i)
  {
    double value = u(i);
    for (int j = 0; j < nt; ++j) value -= response(i, j) * z(j);
    solution(bubbles[i]) = value;
  }
  y.SetSize(n); transform.Mult(solution, y);
}
