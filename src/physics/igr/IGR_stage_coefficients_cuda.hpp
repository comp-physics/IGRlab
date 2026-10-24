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

enum class StageCoefficientStateLayout { Restricted, Native };

// IGRLAB_IGR_STAGE_STATE_DIRECT=1 lets the solver request Native only for its
// admitted fused coefficient path. Diagnostics and unsupported inputs retain
// the existing current-state restriction and MFEM/fused fallback.
// state must be the current LEXICOGRAPHIC element restriction of ufes, or its
// native L-vector when state_layout is Native. Native admission proves the
// same disjoint identity L2 element layout and actual MFEM component strides;
// it changes only state loads, not interpolation or coefficient arithmetic.
// An admitted call overwrites the three original scalar QFs on CUDA.
// Unsupported configurations return false, with a reason, before writing any
// output. The caller retains the MFEM path for those configurations.
bool TryUpdateFusedStageCoefficientsCUDA(
  const mfem::ParFiniteElementSpace& ufes,
  const mfem::QuadratureSpace& mass_qs,
  const mfem::QuadratureSpace& diff_qs,
  const mfem::QuadratureSpace& rhs_qs,
  const mfem::Vector& state,
  double alpha, double rho_floor, const mfem::Vector* element_alpha,
  mfem::QuadratureFunction& invrho, mfem::QuadratureFunction& ck,
  mfem::QuadratureFunction& src,
  std::shared_ptr<FusedStageCoefficientCache>& cache,
  std::string* reason = nullptr,
  StageCoefficientStateLayout state_layout = StageCoefficientStateLayout::Restricted);
}
}
