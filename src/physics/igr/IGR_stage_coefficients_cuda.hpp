#pragma once

#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include <memory>
#include <string>

namespace igrlab
{
namespace igr_internal
{
// Private solver cache. It owns reference-basis data and records the identities
// and sequences of borrowed geometry; it never retains a stage field.
struct FusedStageCoefficientCache;

// restricted_state must be the current LEXICOGRAPHIC element restriction of
// ufes. An admitted call overwrites the three original scalar QFs on CUDA.
// Unsupported configurations return false, with a reason, before writing any
// output. The caller retains the MFEM path for those configurations.
bool TryUpdateFusedStageCoefficientsCUDA(
  const mfem::ParFiniteElementSpace& ufes,
  const mfem::QuadratureSpace& mass_qs,
  const mfem::QuadratureSpace& diff_qs,
  const mfem::QuadratureSpace& rhs_qs,
  const mfem::Vector& restricted_state,
  double alpha, double rho_floor, const mfem::Vector* element_alpha,
  mfem::QuadratureFunction& invrho, mfem::QuadratureFunction& ck,
  mfem::QuadratureFunction& src,
  std::shared_ptr<FusedStageCoefficientCache>& cache,
  std::string* reason = nullptr);
}
}
