#include "numerics/dgsem/dgsem_flux.hpp"

namespace mfem
{
namespace dgsem
{
namespace
{

void AddScaledRow(mfem::DenseMatrix& mat,
                  int row,
                  double scale,
                  const mfem::Vector& vec)
{
  const int n = vec.Size();
  for (int i = 0; i < n; ++i)
  {
    mat(row, i) += scale * vec(i);
  }
}

std::vector<int>
BuildLexToLocal(const std::vector<std::array<int, 3>>& dof_to_ijk,
                int n1d,
                int dim)
{
  const int dof = static_cast<int>(dof_to_ijk.size());
  std::vector<int> lex_to_local(n1d * n1d * (dim > 2 ? n1d : 1), -1);
  for (int local = 0; local < dof; ++local)
  {
    const auto& ijk = dof_to_ijk[local];
    const int i = ijk[0];
    const int j = ijk[1];
    const int k = ijk[2];
    const int lex =
      (dim == 1) ? i : (dim == 2 ? i + n1d * j : i + n1d * (j + n1d * k));
    lex_to_local[lex] = local;
  }
  return lex_to_local;
}

} // namespace

void FluxDifferencing1D(const mfem::DenseMatrix& u,
                        const mfem::DenseMatrix& dsplit,
                        const VolumeFlux& volume_flux,
                        mfem::DenseMatrix& du,
                        double scale)
{
  const int dof = u.Height();
  const int num_eq = u.Width();
  MFEM_VERIFY(dsplit.Height() == dsplit.Width(), "dsplit must be square.");
  MFEM_VERIFY(dsplit.Height() == dof, "dsplit size must match dof count.");

  du.SetSize(dof, num_eq);
  du = 0.0;

  mfem::Vector ui(num_eq);
  mfem::Vector uj(num_eq);
  mfem::Vector flux(num_eq);

  for (int i = 0; i < dof; ++i)
  {
    u.GetRow(i, ui);
    for (int j = i + 1; j < dof; ++j)
    {
      u.GetRow(j, uj);
      volume_flux(ui, uj, /*dir=*/0, flux);
      AddScaledRow(du, i, scale * dsplit(i, j), flux);
      AddScaledRow(du, j, scale * dsplit(j, i), flux);
    }
  }
}

void FluxDifferencing2D(const mfem::DenseMatrix& u,
                        const mfem::DenseMatrix& dsplit,
                        const std::vector<std::array<int, 3>>& dof_to_ijk,
                        const VolumeFlux& volume_flux,
                        mfem::DenseMatrix& du,
                        double scale)
{
  const int dof = u.Height();
  const int num_eq = u.Width();
  const int n1d = dsplit.Height();
  MFEM_VERIFY(dsplit.Height() == dsplit.Width(), "dsplit must be square.");
  MFEM_VERIFY(dof == n1d * n1d, "2D tensor-product dof mismatch.");
  MFEM_VERIFY(static_cast<int>(dof_to_ijk.size()) == dof,
              "dof_to_ijk size mismatch.");

  du.SetSize(dof, num_eq);
  du = 0.0;

  const std::vector<int> lex_to_local = BuildLexToLocal(dof_to_ijk, n1d, 2);

  mfem::Vector ui(num_eq);
  mfem::Vector uj(num_eq);
  mfem::Vector flux(num_eq);

  // x-direction flux differencing (vary i, fixed j)
  for (int j = 0; j < n1d; ++j)
  {
    for (int i = 0; i < n1d; ++i)
    {
      const int local_i = lex_to_local[i + n1d * j];
      u.GetRow(local_i, ui);
      for (int ii = i + 1; ii < n1d; ++ii)
      {
        const int local_ii = lex_to_local[ii + n1d * j];
        u.GetRow(local_ii, uj);
        volume_flux(ui, uj, /*dir=*/0, flux);
        AddScaledRow(du, local_i, scale * dsplit(i, ii), flux);
        AddScaledRow(du, local_ii, scale * dsplit(ii, i), flux);
      }
    }
  }

  // y-direction flux differencing (vary j, fixed i)
  for (int i = 0; i < n1d; ++i)
  {
    for (int j = 0; j < n1d; ++j)
    {
      const int local_j = lex_to_local[i + n1d * j];
      u.GetRow(local_j, ui);
      for (int jj = j + 1; jj < n1d; ++jj)
      {
        const int local_jj = lex_to_local[i + n1d * jj];
        u.GetRow(local_jj, uj);
        volume_flux(ui, uj, /*dir=*/1, flux);
        AddScaledRow(du, local_j, scale * dsplit(j, jj), flux);
        AddScaledRow(du, local_jj, scale * dsplit(jj, j), flux);
      }
    }
  }
}

} // namespace dgsem
} // namespace mfem
