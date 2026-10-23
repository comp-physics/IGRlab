#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#error "The exact-face host-query regression must be compiled as ordinary C++."
#endif

// Give this test-only header instantiation a unique name. With the old
// compiler-dependent body, an uninlined C++ call must not bind to an NVCC
// weak inline definition and hide the regression through an ODR violation.
#define UsesInteriorFaceDGDiffusionCompactLift2D CompactLiftHostPredicateForTest
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"
#undef UsesInteriorFaceDGDiffusionCompactLift2D

// Deliberately compiled as ordinary C++, like IGR_linear_solve.cpp. The CUDA
// fixture must not validate a compiler-dependent predicate solely from NVCC.
namespace exact_face_test
{
bool CompactLiftHostQuery(
  const ::igrlab::dg_face::InteriorFaceDGDiffusionApplyInfo& info,
  const ::igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch& work)
{
  return ::igrlab::dg_face::CompactLiftHostPredicateForTest(info, work);
}
} // namespace exact_face_test
