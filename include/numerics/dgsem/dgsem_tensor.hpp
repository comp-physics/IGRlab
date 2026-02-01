#pragma once

/**
 * @file numerics/dgsem/dgsem_tensor.hpp
 * @brief Tensor-product DOF mapping helpers for DGSEM.
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"

#include <array>
#include <map>
#include <vector>

namespace mfem
{
namespace dgsem
{

struct TensorDofMapKey
{
  int order = 0;
  int dim = 0;
  int basis_type = 0;
  int dof = 0;
};

struct TensorDofMapKeyLess
{
  bool operator()(const TensorDofMapKey& a, const TensorDofMapKey& b) const
  {
    if (a.order != b.order) { return a.order < b.order; }
    if (a.dim != b.dim) { return a.dim < b.dim; }
    if (a.basis_type != b.basis_type) { return a.basis_type < b.basis_type; }
    return a.dof < b.dof;
  }
};

inline std::array<int, 3> LexToIJK(int lex, int n1d, int dim)
{
  const int i = lex % n1d;
  const int j = (dim > 1) ? (lex / n1d) % n1d : 0;
  const int k = (dim > 2) ? (lex / (n1d * n1d)) : 0;
  return {i, j, k};
}

inline std::vector<int> BuildLocalToLex(const FiniteElement& fe)
{
  const int dof = fe.GetDof();
  std::vector<int> local_to_lex(dof, 0);

  const auto* tbe = dynamic_cast<const TensorBasisElement*>(&fe);
  const auto* nfe = dynamic_cast<const NodalFiniteElement*>(&fe);
  const Array<int>* lex_to_local = nullptr;

  if (tbe && tbe->GetDofMap().Size() > 0)
  {
    lex_to_local = &tbe->GetDofMap();
  }
  else if (nfe && nfe->GetLexicographicOrdering().Size() > 0)
  {
    lex_to_local = &nfe->GetLexicographicOrdering();
  }

  if (lex_to_local)
  {
    for (int lex = 0; lex < dof; ++lex)
    {
      const int local = (*lex_to_local)[lex];
      local_to_lex[local] = lex;
    }
  }
  else
  {
    for (int local = 0; local < dof; ++local)
    {
      local_to_lex[local] = local;
    }
  }

  return local_to_lex;
}

inline std::vector<std::array<int, 3>> BuildTensorDofToIJK(const FiniteElement& fe)
{
  const int dim = fe.GetDim();
  const int order = fe.GetOrder();
  const int n1d = order + 1;
  const int dof = fe.GetDof();
  MFEM_VERIFY(dim >= 1 && dim <= 3, "Tensor DOF map supports dim=1..3.");
  MFEM_VERIFY(dof == TensorBasisElement::Pow(n1d, dim),
              "Finite element is not tensor-product of order p=" << order << ".");

  const std::vector<int> local_to_lex = BuildLocalToLex(fe);
  std::vector<std::array<int, 3>> dof_to_ijk(dof);
  for (int local = 0; local < dof; ++local)
  {
    const int lex = local_to_lex[local];
    dof_to_ijk[local] = LexToIJK(lex, n1d, dim);
  }
  return dof_to_ijk;
}

inline const std::vector<std::array<int, 3>>& GetTensorDofToIJK(const FiniteElement& fe)
{
  static std::map<TensorDofMapKey, std::vector<std::array<int, 3>>, TensorDofMapKeyLess> cache;
  const auto* tbe = dynamic_cast<const TensorBasisElement*>(&fe);
  const int basis_type = tbe ? tbe->GetBasisType() : -1;
  const TensorDofMapKey key{fe.GetOrder(), fe.GetDim(), basis_type, fe.GetDof()};
  auto it = cache.find(key);
  if (it != cache.end())
  {
    return it->second;
  }
  auto inserted = cache.emplace(key, BuildTensorDofToIJK(fe));
  return inserted.first->second;
}

} // namespace dgsem
} // namespace mfem
