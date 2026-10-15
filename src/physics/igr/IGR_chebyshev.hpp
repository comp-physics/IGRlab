#pragma once

#include "mfem.hpp"

#include <memory>

namespace igrlab
{
namespace igr_internal
{

// Private IGR equivalent of the pinned MFEM Chebyshev polynomial. The caller
// must establish an operator with ordered launches and stable vector storage.
// Only the ownership of its two scratch buffers changes during application.
class OwnedScratchChebyshev final : public mfem::Solver
{
public:
  static bool HasCUDAKernels();
  static bool UsesOrdinaryCUDA();

  OwnedScratchChebyshev(const mfem::Operator& op,
                       const mfem::Vector& diagonal,
                       const mfem::Array<int>& essential_tdofs,
                       int degree,
                       mfem::real_t max_eigenvalue);

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  { Mult(x, y); }

  void SetOperator(const mfem::Operator& op) override
  {
    op_ = &op;
    // A later rebind does not establish the original stream/storage contract.
    // As in MFEM, it does not refresh the polynomial or its dimensions.
    swap_owned_buffers_ = op_ == ordered_op_;
  }

private:
  const int degree_;
  const int size_;
  mfem::Vector inverse_diagonal_;
  mfem::Array<mfem::real_t> coefficients_;
  mutable mfem::Vector residual_;
  mutable mfem::Vector helper_;
  const mfem::Operator* op_;
  const mfem::Operator* const ordered_op_;
  bool swap_owned_buffers_ = true;
};

// Unsupported production routes retain the original MFEM implementation.
std::unique_ptr<mfem::Solver> MakeIGRChebyshevSmoother(
  const mfem::Operator& op,
  const mfem::Vector& diagonal,
  const mfem::Array<int>& essential_tdofs,
  int degree,
  mfem::real_t max_eigenvalue,
  bool ordered_owned_cuda_operator);

} // namespace igr_internal
} // namespace igrlab
