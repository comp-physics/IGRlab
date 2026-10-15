#include "IGR_chebyshev.hpp"

#include "mfem/general/forall.hpp"

#include <cmath>

namespace igrlab
{
namespace igr_internal
{

namespace
{
// NVCC extended lambdas need an addressable enclosing function.
void SetupInverseDiagonal(int size,
                          const mfem::Vector& diagonal,
                          const mfem::Array<int>& essential_tdofs,
                          mfem::Vector& inverse_diagonal)
{
  const auto D = diagonal.Read();
  const auto X = inverse_diagonal.Write();
  mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) { X[i] = 1.0 / D[i]; });
  const auto I = essential_tdofs.Read();
  mfem::forall(essential_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
  { X[I[i]] = 1.0; });
}
} // namespace

bool OwnedScratchChebyshev::HasCUDAKernels()
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  return true;
#else
  return false;
#endif
}

bool OwnedScratchChebyshev::UsesOrdinaryCUDA()
{
  // MFEM implicitly adds CUDA for some CEED/OCCA configurations. That flag
  // alone does not prove the stream contract of the native PA implementation.
  const unsigned long alternative_backends =
    mfem::Backend::OMP_MASK | mfem::Backend::RAJA_MASK |
    mfem::Backend::OCCA_MASK | mfem::Backend::CEED_MASK |
    mfem::Backend::HIP_MASK | mfem::Backend::DEBUG_DEVICE;
  return HasCUDAKernels() && mfem::Device::Allows(mfem::Backend::CUDA) &&
         !mfem::Device::Allows(alternative_backends);
}

OwnedScratchChebyshev::OwnedScratchChebyshev(
  const mfem::Operator& op,
  const mfem::Vector& diagonal,
  const mfem::Array<int>& essential_tdofs,
  int degree,
  mfem::real_t max_eigenvalue)
  : mfem::Solver(diagonal.Size()),
    degree_(degree),
    size_(diagonal.Size()),
    inverse_diagonal_(size_),
    coefficients_(degree),
    residual_(size_),
    op_(&op),
    ordered_op_(&op)
{
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
  MFEM_VERIFY(!mfem::Device::Allows(mfem::Backend::CUDA_MASK),
              "Private Chebyshev CUDA arithmetic requires local CUDA compilation.");
#endif
  // Retain the Setup/Mult arithmetic from MFEM d9d6526 linalg/solvers.cpp.
  // The fused scale/accumulate below preserves the rounded residual between
  // those operations. Neither coefficients nor state are reused across the
  // application's changing closure operators.
  residual_.UseDevice(true);
  helper_.UseDevice(true);
  SetupInverseDiagonal(size_, diagonal, essential_tdofs, inverse_diagonal_);

  mfem::real_t upper_bound = 1.2 * max_eigenvalue;
  mfem::real_t lower_bound = 0.3 * max_eigenvalue;
  mfem::real_t theta = 0.5 * (upper_bound + lower_bound);
  mfem::real_t delta = 0.5 * (upper_bound - lower_bound);
  switch (degree_ - 1)
  {
  case 0:
  {
    coefficients_[0] = 1.0 / theta;
    break;
  }
  case 1:
  {
    mfem::real_t tmp_0 = 1.0/(pow(delta, 2) - 2*pow(theta, 2));
    coefficients_[0] = -4*theta*tmp_0;
    coefficients_[1] = 2*tmp_0;
    break;
  }
  case 2:
  {
    mfem::real_t tmp_0 = 3*pow(delta, 2);
    mfem::real_t tmp_1 = pow(theta, 2);
    mfem::real_t tmp_2 = 1.0/(-4*pow(theta, 3) + theta*tmp_0);
    coefficients_[0] = tmp_2*(tmp_0 - 12*tmp_1);
    coefficients_[1] = 12/(tmp_0 - 4*tmp_1);
    coefficients_[2] = -4*tmp_2;
    break;
  }
  case 3:
  {
    mfem::real_t tmp_0 = pow(delta, 2);
    mfem::real_t tmp_1 = pow(theta, 2);
    mfem::real_t tmp_2 = 8*tmp_0;
    mfem::real_t tmp_3 = 1.0/(pow(delta, 4) + 8*pow(theta, 4) - tmp_1*tmp_2);
    coefficients_[0] = tmp_3*(32*pow(theta, 3) - 16*theta*tmp_0);
    coefficients_[1] = tmp_3*(-48*tmp_1 + tmp_2);
    coefficients_[2] = 32*theta*tmp_3;
    coefficients_[3] = -8*tmp_3;
    break;
  }
  case 4:
  {
    mfem::real_t tmp_0 = 5*pow(delta, 4);
    mfem::real_t tmp_1 = pow(theta, 4);
    mfem::real_t tmp_2 = pow(theta, 2);
    mfem::real_t tmp_3 = pow(delta, 2);
    mfem::real_t tmp_4 = 60*tmp_3;
    mfem::real_t tmp_5 = 20*tmp_3;
    mfem::real_t tmp_6 = 1.0/(16*pow(theta, 5) - pow(theta, 3)*tmp_5 + theta*tmp_0);
    mfem::real_t tmp_7 = 160*tmp_2;
    mfem::real_t tmp_8 = 1.0/(tmp_0 + 16*tmp_1 - tmp_2*tmp_5);
    coefficients_[0] = tmp_6*(tmp_0 + 80*tmp_1 - tmp_2*tmp_4);
    coefficients_[1] = tmp_8*(tmp_4 - tmp_7);
    coefficients_[2] = tmp_6*(-tmp_5 + tmp_7);
    coefficients_[3] = -80*tmp_8;
    coefficients_[4] = 16*tmp_6;
    break;
  }
  default:
    MFEM_ABORT("Chebyshev smoother not implemented for order = " << degree_);
  }
}

void OwnedScratchChebyshev::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
  if (iterative_mode)
  { MFEM_ABORT("Chebyshev smoother not implemented for iterative mode"); }
  if (!op_)
  { MFEM_ABORT("Chebyshev smoother requires operator"); }

  residual_ = x;
  helper_.SetSize(x.Size());
  helper_.UseDevice(true);
  y.UseDevice(true);
  y = 0.0;

  for (int k = 0; k < degree_; ++k)
  {
    if (k > 0)
    {
      op_->Mult(residual_, helper_);
      if (swap_owned_buffers_ && residual_.OwnsData() && helper_.OwnsData() &&
          residual_.Size() == size_ && helper_.Size() == size_)
      {
        // Both allocations remain alive. All admitted producer/consumer
        // launches use the same stream; Swap carries complete Memory state.
        residual_.Swap(helper_);
      }
      else
      {
        residual_ = helper_;
      }
    }

    const auto Dinv = inverse_diagonal_.Read();
    const auto R = residual_.ReadWrite();
    const auto Y = y.ReadWrite();
    const auto C = coefficients_.Read();
    mfem::forall(size_, [=] MFEM_HOST_DEVICE (int i)
    {
      // The original first kernel stored this product before accumulation.
      // Preserve that rounding boundary while avoiding a second launch/read.
#if defined(__CUDA_ARCH__) && defined(MFEM_USE_DOUBLE)
      const mfem::real_t scaled = __dmul_rn(R[i], Dinv[i]);
#elif defined(__CUDA_ARCH__) && defined(MFEM_USE_SINGLE)
      const mfem::real_t scaled = __fmul_rn(R[i], Dinv[i]);
#else
      const volatile mfem::real_t scaled = R[i] * Dinv[i];
#endif
      R[i] = scaled;
      Y[i] += C[k] * scaled;
    });
  }
}

std::unique_ptr<mfem::Solver> MakeIGRChebyshevSmoother(
  const mfem::Operator& op,
  const mfem::Vector& diagonal,
  const mfem::Array<int>& essential_tdofs,
  int degree,
  mfem::real_t max_eigenvalue,
  bool ordered_owned_cuda_operator)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  if (ordered_owned_cuda_operator && OwnedScratchChebyshev::UsesOrdinaryCUDA())
  {
    return std::make_unique<OwnedScratchChebyshev>(
      op, diagonal, essential_tdofs, degree, max_eigenvalue);
  }
#else
  (void)ordered_owned_cuda_operator;
#endif
  return std::make_unique<mfem::OperatorChebyshevSmoother>(
    op, diagonal, essential_tdofs, degree, max_eigenvalue);
}

} // namespace igr_internal
} // namespace igrlab
