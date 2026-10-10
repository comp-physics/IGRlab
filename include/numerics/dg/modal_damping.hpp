#pragma once

#include "mfem.hpp"

namespace mfem
{

/** Physical-time conservative modal relaxation on affine tensor DG cells.
 *
 * In one dimension the M16 profile is q(k) = max((39*k-29*p)/(10*p),0)^4
 * with rate 80000. In multiple dimensions q is evaluated at max(k_1,...,k_d).
 * This tensor extension retains a maximum decay rate equal to rate, and
 * reduces to the original law for a field depending on one reference axis.
 * The source is added to every conservative component after mass inversion.
 * A zero constant-mode rate preserves physical cell integrals on affine cells.
 * No positivity, entropy, or gradient-monotonicity guarantee is implied.
 */
class ConservativeModalDamping
{
public:
  ConservativeModalDamping(FiniteElementSpace& space, real_t rate);

  static real_t ModeWeight(int order, int mode);
  real_t GetRate() const { return rate_; }

  /// Add the strong source to a local FE vector (before true-DOF restriction).
  void AddMult(const Vector& state, Vector& rhs) const;

private:
  FiniteElementSpace& space_;
  real_t rate_;
  int dimension_ = 0;
  int n1d_ = 0;
  int dof_ = 0;
  int entries_ = 0;
  long space_sequence_ = 0;
  long mesh_sequence_ = 0;
  long nodes_sequence_ = 0;
  Array<int> lex_vdofs_;
  DenseMatrix to_modal_;
  DenseMatrix to_nodal_;
  Vector weights_;
  mutable Vector work_a_;
  mutable Vector work_b_;
};

} // namespace mfem
