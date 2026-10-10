#include "numerics/dg/modal_damping.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mfem
{
namespace
{

void RequireAffine(ElementTransformation& transformation,
                   Geometry::Type geometry)
{
  // More points per direction than the polynomial geometry degree also
  // detect curved elements whose Jacobian happens to agree at the corners.
  const IntegrationRule& rule =
    IntRules.Get(geometry, 2 * transformation.Order() + 2);
  transformation.SetIntPoint(&rule.IntPoint(0));
  const DenseMatrix first(transformation.Jacobian());
  real_t scale = 0.0;
  for (int j = 0; j < first.Width(); ++j)
  {
    for (int i = 0; i < first.Height(); ++i)
    {
      scale = std::max(scale, std::abs(first(i, j)));
    }
  }
  MFEM_VERIFY(std::isfinite(scale) && scale > 0.0,
              "Modal damping requires a nonsingular affine cell.");
  const real_t tolerance = 512 * std::numeric_limits<real_t>::epsilon() * scale;
  for (int q = 0; q < rule.GetNPoints(); ++q)
  {
    transformation.SetIntPoint(&rule.IntPoint(q));
    MFEM_VERIFY(std::isfinite(transformation.Weight()) &&
                  transformation.Weight() > 0.0,
                "Modal damping requires a finite positive cell measure.");
    const DenseMatrix& jacobian = transformation.Jacobian();
    for (int j = 0; j < first.Width(); ++j)
    {
      for (int i = 0; i < first.Height(); ++i)
      {
        MFEM_VERIFY(std::isfinite(jacobian(i, j)) &&
                      std::abs(jacobian(i, j) - first(i, j)) <= tolerance,
                    "Modal damping currently requires affine tensor cells; "
                    "curved or non-affine cells need a different "
                    "physical-mean-preserving operator.");
      }
    }
  }
}

real_t Coordinate(const IntegrationPoint& point, int axis)
{
  return axis == 0 ? point.x : (axis == 1 ? point.y : point.z);
}

void TransformAxis(const DenseMatrix& matrix, int n, int entries, int axis,
                   const Vector& input, Vector& output)
{
  const int stride = axis == 0 ? 1 : TensorBasisElement::Pow(n, axis);
  const real_t* a = matrix.Read();
  const real_t* x = input.Read();
  real_t* y = output.Write();
  mfem::forall(entries, [=] MFEM_HOST_DEVICE(int i)
  {
    const int row = (i / stride) % n;
    const int base = i - row * stride;
    real_t value = 0.0;
    for (int k = 0; k < n; ++k)
    {
      value += a[row + n * k] * x[base + stride * k];
    }
    y[i] = value;
  });
}

} // namespace

real_t ConservativeModalDamping::ModeWeight(int order, int mode)
{
  MFEM_VERIFY(order >= 0 && mode >= 0 && mode <= order,
              "Invalid modal damping order or mode index.");
  if (order == 0)
  {
    return 0.0;
  }
  const real_t ramp =
    std::max((39.0 * mode - 29.0 * order) / (10.0 * order), 0.0);
  const real_t square = ramp * ramp;
  return square * square;
}

ConservativeModalDamping::ConservativeModalDamping(FiniteElementSpace& space,
                                                 real_t rate)
  : space_(space), rate_(rate), dimension_(space.GetMesh()->Dimension()),
    space_sequence_(space.GetSequence()),
    mesh_sequence_(space.GetMesh()->GetSequence()),
    nodes_sequence_(space.GetMesh()->GetNodesSequence())
{
  MFEM_VERIFY(std::isfinite(rate_) && rate_ >= 0.0,
              "Modal damping rate must be finite and nonnegative.");
  MFEM_VERIFY(space.FEColl()->GetContType() == FiniteElementCollection::DISCONTINUOUS,
              "Modal damping requires a discontinuous finite-element space.");
  MFEM_VERIFY(dimension_ >= 1 && dimension_ <= 3 &&
                space.GetMesh()->SpaceDimension() == dimension_,
              "Modal damping supports full-dimensional tensor cells in 1D--3D.");
  if (space.GetNE() == 0)
  {
    MFEM_VERIFY(space.GetVSize() == 0, "Invalid empty modal damping partition.");
    return;
  }

  const FiniteElement& first = *space.GetFE(0);
  const int order = first.GetOrder();
  n1d_ = order + 1;
  dof_ = TensorBasisElement::Pow(n1d_, dimension_);
  MFEM_VERIFY(dynamic_cast<const NodalTensorFiniteElement*>(&first) &&
                first.GetDof() == dof_ && first.GetNodes().GetNPoints() == dof_,
              "Modal damping requires nodal tensor finite elements.");
  const auto first_map = dgsem::BuildLocalToLex(first);
  std::vector<int> first_lex_to_local(dof_);
  for (int local = 0; local < dof_; ++local)
  {
    first_lex_to_local[first_map[local]] = local;
  }
  std::vector<real_t> nodes(n1d_);
  to_nodal_.SetSize(n1d_);
  for (int i = 0; i < n1d_; ++i)
  {
    nodes[i] = first.GetNodes().IntPoint(first_lex_to_local[i]).x;
    const real_t xi = 2.0 * nodes[i] - 1.0;
    to_nodal_(i, 0) = 1.0;
    if (order > 0)
    {
      to_nodal_(i, 1) = xi;
    }
    for (int k = 2; k <= order; ++k)
    {
      to_nodal_(i, k) =
        ((2 * k - 1) * xi * to_nodal_(i, k - 1) -
         (k - 1) * to_nodal_(i, k - 2)) / k;
    }
  }
  DenseMatrixInverse inverse(to_nodal_);
  inverse.GetInverseMatrix(to_modal_);

  entries_ = space.GetVSize();
  MFEM_VERIFY(entries_ == space.GetNE() * space.GetVDim() * dof_,
              "Modal damping requires element-local nodal DG degrees of freedom.");
  lex_vdofs_.SetSize(entries_);
  std::vector<bool> visited(entries_, false);
  Array<int> vdofs;
  const Geometry::Type geometry = dimension_ == 1 ? Geometry::SEGMENT :
                                  dimension_ == 2 ? Geometry::SQUARE : Geometry::CUBE;
  for (int e = 0; e < space.GetNE(); ++e)
  {
    const FiniteElement& fe = *space.GetFE(e);
    MFEM_VERIFY(fe.GetGeomType() == geometry && fe.GetOrder() == order &&
                  fe.GetDof() == dof_ && fe.GetNodes().GetNPoints() == dof_ &&
                  dynamic_cast<const NodalTensorFiniteElement*>(&fe) &&
                  fe.GetMapType() == FiniteElement::VALUE,
                "Modal damping requires uniform-order, VALUE-mapped nodal "
                "segment/quad/hex DG elements.");
    RequireAffine(*space.GetElementTransformation(e), geometry);
    const auto local_to_lex = dgsem::BuildLocalToLex(fe);
    space.GetElementVDofs(e, vdofs);
    for (int local = 0; local < dof_; ++local)
    {
      const int lex = local_to_lex[local];
      const auto ijk = dgsem::LexToIJK(lex, n1d_, dimension_);
      for (int axis = 0; axis < dimension_; ++axis)
      {
        MFEM_VERIFY(std::abs(Coordinate(fe.GetNodes().IntPoint(local), axis) -
                              nodes[ijk[axis]]) < 64 * std::numeric_limits<real_t>::epsilon(),
                    "Modal damping requires matching tensor nodes in every direction.");
      }
      for (int component = 0; component < space.GetVDim(); ++component)
      {
        // MFEM's element vector is component-blocked for either global ordering.
        const int vdof = vdofs[component * dof_ + local];
        MFEM_VERIFY(vdof >= 0 && vdof < entries_ && !visited[vdof],
                    "Modal damping requires unique unsigned nodal element DOFs.");
        visited[vdof] = true;
        lex_vdofs_[(e * space.GetVDim() + component) * dof_ + lex] = vdof;
      }
    }
  }
  weights_.SetSize(dof_);
  for (int lex = 0; lex < dof_; ++lex)
  {
    const auto ijk = dgsem::LexToIJK(lex, n1d_, dimension_);
    weights_[lex] = ModeWeight(order, std::max({ijk[0], ijk[1], ijk[2]}));
  }
  work_a_.SetSize(entries_);
  work_b_.SetSize(entries_);
  work_a_.UseDevice(true);
  work_b_.UseDevice(true);
}

void ConservativeModalDamping::AddMult(const Vector& state, Vector& rhs) const
{
  MFEM_VERIFY(space_.GetSequence() == space_sequence_ &&
                space_.GetMesh()->GetSequence() == mesh_sequence_ &&
                space_.GetMesh()->GetNodesSequence() == nodes_sequence_,
              "Modal damping geometry/space changed; update the DG operator first.");
  MFEM_VERIFY(state.Size() == entries_ && rhs.Size() == entries_,
              "Modal damping expects local FE vectors.");
  if (entries_ == 0 || rate_ == 0.0 || n1d_ == 1)
  {
    return;
  }
  const int dof = dof_;
  const int* map = lex_vdofs_.Read();
  const real_t* x = state.Read();
  real_t* packed = work_a_.Write();
  mfem::forall(entries_, [=] MFEM_HOST_DEVICE(int i)
  {
    // A constant shift lies in the undamped kernel. Subtract it before the
    // transform so uniform states receive exactly zero numerical damping.
    packed[i] = x[map[i]] - x[map[(i / dof) * dof]];
  });
  Vector* input = &work_a_;
  Vector* output = &work_b_;
  for (int axis = 0; axis < dimension_; ++axis)
  {
    TransformAxis(to_modal_, n1d_, entries_, axis, *input, *output);
    std::swap(input, output);
  }
  const real_t* weights = weights_.Read();
  real_t* modes = input->ReadWrite();
  mfem::forall(entries_, [=] MFEM_HOST_DEVICE(int i)
  {
    modes[i] *= weights[i % dof];
  });
  for (int axis = 0; axis < dimension_; ++axis)
  {
    TransformAxis(to_nodal_, n1d_, entries_, axis, *input, *output);
    std::swap(input, output);
  }
  const real_t rate = rate_;
  const real_t* source = input->Read();
  real_t* y = rhs.ReadWrite();
  mfem::forall(entries_, [=] MFEM_HOST_DEVICE(int i)
  {
    y[map[i]] -= rate * source[i];
  });
}

} // namespace mfem
