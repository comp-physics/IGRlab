#pragma once

#include "mfem.hpp"

namespace mfem
{

/** Physical-time conservative modal relaxation on tensor DG cells.
 *
 * In one dimension the M16 profile is q(k) = max((39*k-29*p)/(10*p),0)^4
 * with rate 80000. In multiple dimensions q is evaluated at max(k_1,...,k_d).
 * The maximum decay rate equals rate. On affine cells the tensor extension
 * reduces to the original law for a field depending on one reference axis.
 * The source is added to every conservative component after mass inversion.
 * Affine cells retain the tensor Legendre transform. Curved 2D polynomial
 * quadrilaterals (solution and geometry orders at most eight) use a cached
 * physical-mass orthonormalization ordered by increasing q. This preserves
 * physical cell integrals and the entire zero-q polynomial space, dissipates
 * the physical mass norm, and retains decay eigenvalues between zero and rate.
 * Curved operators require finite positive Jacobians, checked conditioning,
 * and at most 256 MiB of local matrix storage. Geometry is fixed until the
 * enclosing DG operator is updated.
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
  Array<int> curved_elements_;
  Array<int> curved_operator_index_;
  Vector curved_decay_;
  mutable Vector curved_work_;
  mutable Vector work_a_;
  mutable Vector work_b_;
};

} // namespace mfem
