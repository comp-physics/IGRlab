#include "numerics/dg/modal_damping.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace mfem
{
namespace
{

bool IsAffine(ElementTransformation& transformation, Geometry::Type geometry)
{
  // More points per direction than the polynomial geometry degree also
  // detect curved elements whose Jacobian happens to agree at the corners.
  const int geometry_order = transformation.Order();
  MFEM_VERIFY(geometry_order >= 0 &&
                geometry_order <= (std::numeric_limits<int>::max() - 2) / 2,
              "Invalid modal damping geometry order.");
  const IntegrationRule& rule = IntRules.Get(geometry, 2 * geometry_order + 2);
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
              "Modal damping requires a nonsingular cell.");
  const real_t tolerance = 512 * std::numeric_limits<real_t>::epsilon() * scale;
  bool affine = true;
  bool positive_jacobians = true;
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
        MFEM_VERIFY(std::isfinite(jacobian(i, j)),
                    "Modal damping requires finite cell Jacobians.");
        affine = affine &&
                 std::abs(jacobian(i, j) - first(i, j)) <= tolerance;
      }
    }
    const real_t determinant = jacobian.Det();
    positive_jacobians = positive_jacobians &&
                         std::isfinite(determinant) && determinant > 0.0;
  }
  MFEM_VERIFY(affine || positive_jacobians,
              "Curved modal damping requires finite positive Jacobians.");
  return affine;
}

// A dense curved operator is deliberately bounded; affine tensor transforms
// retain their existing dimensional and high-order coverage.
constexpr int max_curved_order = 8;
constexpr std::size_t max_curved_bytes = 256u * 1024u * 1024u;
using HostMatrix = std::vector<long double>;

long double InfinityNorm(const HostMatrix& matrix, int n)
{
  long double norm = 0.0L;
  for (int i = 0; i < n; ++i)
  {
    long double row_sum = 0.0L;
    for (int j = 0; j < n; ++j)
    {
      row_sum += std::abs(matrix[i + n * j]);
    }
    norm = std::max(norm, row_sum);
  }
  return norm;
}

void LegendreValues(real_t coordinate, int n, std::vector<long double>& values)
{
  const long double xi = 2.0L * coordinate - 1.0L;
  values[0] = 1.0L;
  if (n > 1)
  {
    values[1] = xi;
  }
  for (int k = 2; k < n; ++k)
  {
    values[k] = ((2 * k - 1) * xi * values[k - 1] -
                 (k - 1) * values[k - 2]) / k;
  }
}

real_t PositiveJacobian(ElementTransformation& transformation,
                        const IntegrationPoint& point)
{
  transformation.SetIntPoint(&point);
  const DenseMatrix& jacobian = transformation.Jacobian();
  for (int j = 0; j < jacobian.Width(); ++j)
  {
    for (int i = 0; i < jacobian.Height(); ++i)
    {
      MFEM_VERIFY(std::isfinite(jacobian(i, j)),
                  "Curved modal damping requires finite cell Jacobians.");
    }
  }
  const real_t determinant = jacobian.Det();
  MFEM_VERIFY(std::isfinite(determinant) && determinant > 0.0,
              "Curved modal damping requires finite positive Jacobians.");
  return determinant;
}

void BuildCurvedDecay(ElementTransformation& transformation, int order,
                      const std::vector<int>& mode_order,
                      const Vector& weights, const DenseMatrix& to_nodal,
                      const DenseMatrix& to_modal, real_t* decay)
{
  const auto* iso =
    dynamic_cast<const IsoparametricTransformation*>(&transformation);
  const FiniteElement* geometry_fe = iso ? iso->GetFE() : nullptr;
  MFEM_VERIFY(geometry_fe && geometry_fe->Space() == FunctionSpace::Qk &&
                dynamic_cast<const NodalFiniteElement*>(geometry_fe) &&
                transformation.Order() >= 1 &&
                transformation.Order() <= max_curved_order,
              "Curved modal damping supports polynomial nodal quadrilateral "
              "geometry of orders one through eight.");
  const int n = order + 1;
  const int dof = n * n;
  const int size = dof * dof;
  const long double epsilon = std::numeric_limits<real_t>::epsilon();
  const long double host_epsilon = std::numeric_limits<long double>::epsilon();

  // For Q_g geometry det(J) has coordinate degree at most 2*g-1.
  // MFEM's OrderW() supplies this bound; the Gauss rule integrates the full
  // product of two degree-p tensor modes with this physical measure.
  const int weight_order = transformation.OrderW();
  MFEM_VERIFY(weight_order >= 0 && weight_order <= 2 * max_curved_order - 1,
              "Unsupported curved modal damping Jacobian polynomial order.");
  const IntegrationRule& rule =
    IntRules.Get(Geometry::SQUARE, 2 * order + weight_order);
  const IntegrationRule& geometry_nodes = geometry_fe->GetNodes();
  for (int q = 0; q < geometry_nodes.GetNPoints(); ++q)
  {
    PositiveJacobian(transformation, geometry_nodes.IntPoint(q));
  }
  const IntegrationRule& vertices = *Geometries.GetVertices(Geometry::SQUARE);
  for (int q = 0; q < vertices.GetNPoints(); ++q)
  {
    PositiveJacobian(transformation, vertices.IntPoint(q));
  }

  HostMatrix gram(size, 0.0L);
  std::vector<long double> px(n), py(n), phi(dof);
  long double volume = 0.0L;
  for (int q = 0; q < rule.GetNPoints(); ++q)
  {
    const IntegrationPoint& point = rule.IntPoint(q);
    const long double weight =
      static_cast<long double>(point.weight) *
      PositiveJacobian(transformation, point);
    MFEM_VERIFY(std::isfinite(weight) && weight > 0.0L,
                "Curved modal damping requires positive Gauss weights.");
    volume += weight;
    LegendreValues(point.x, n, px);
    LegendreValues(point.y, n, py);
    for (int j = 0; j < dof; ++j)
    {
      const int lex = mode_order[j];
      phi[j] = px[lex % n] * py[lex / n];
      for (int i = 0; i <= j; ++i)
      {
        gram[i + dof * j] += weight * phi[i] * phi[j];
      }
    }
  }
  MFEM_VERIFY(std::isfinite(volume) && volume > 0.0L,
              "Curved modal damping requires a finite positive cell volume.");
  for (int j = 0; j < dof; ++j)
  {
    for (int i = 0; i <= j; ++i)
    {
      gram[j + dof * i] = gram[i + dof * j] /= volume;
    }
  }

  // H = U^T U with upper U. Ordering the zero-q polynomial space first is
  // essential: upper-triangular U maps that whole space into the zero-q prefix.
  HostMatrix upper(size, 0.0L), inverse_upper(size, 0.0L);
  const long double gram_norm = InfinityNorm(gram, dof);
  for (int j = 0; j < dof; ++j)
  {
    for (int i = 0; i <= j; ++i)
    {
      long double value = gram[i + dof * j];
      for (int k = 0; k < i; ++k)
      {
        value -= upper[k + dof * i] * upper[k + dof * j];
      }
      if (i == j)
      {
        MFEM_VERIFY(std::isfinite(value) &&
                      value > 64 * dof * host_epsilon * gram_norm,
                    "Curved modal damping physical Gram matrix is singular "
                    "or too ill-conditioned.");
        upper[i + dof * j] = std::sqrt(value);
      }
      else
      {
        upper[i + dof * j] = value / upper[i + dof * i];
      }
    }
    for (int i = j; i >= 0; --i)
    {
      long double value = i == j ? 1.0L : 0.0L;
      for (int k = i + 1; k <= j; ++k)
      {
        value -= upper[i + dof * k] * inverse_upper[k + dof * j];
      }
      inverse_upper[i + dof * j] = value / upper[i + dof * i];
    }
  }
  const long double condition = InfinityNorm(upper, dof) *
                                InfinityNorm(inverse_upper, dof);
  MFEM_VERIFY(std::isfinite(condition) &&
                condition * condition * epsilon <= 1.0e-4L,
              "Curved modal damping physical Gram matrix is too ill-conditioned.");
  long double factor_residual = 0.0L;
  long double inverse_residual = 0.0L;
  for (int j = 0; j < dof; ++j)
  {
    for (int i = 0; i < dof; ++i)
    {
      long double product = 0.0L;
      long double identity = 0.0L;
      for (int k = 0; k < dof; ++k)
      {
        product += upper[k + dof * i] * upper[k + dof * j];
        identity += upper[i + dof * k] * inverse_upper[k + dof * j];
      }
      factor_residual = std::max(factor_residual,
                                 std::abs(product - gram[i + dof * j]));
      inverse_residual = std::max(inverse_residual,
                                  std::abs(identity - (i == j ? 1.0L : 0.0L)));
    }
  }
  MFEM_VERIFY(factor_residual <= 128 * dof * host_epsilon * gram_norm &&
                inverse_residual <= 128 * dof * host_epsilon * condition,
              "Curved modal damping physical Gram factorization failed "
              "reconstruction.");

  HostMatrix nodal(size), modal(size), modal_decay(size, 0.0L), image(size, 0.0L);
  for (int j = 0; j < dof; ++j)
  {
    const int mode = mode_order[j];
    for (int i = 0; i < dof; ++i)
    {
      nodal[i + dof * j] =
        static_cast<long double>(to_nodal(i % n, mode % n)) *
        to_nodal(i / n, mode / n);
      modal[j + dof * i] =
        static_cast<long double>(to_modal(mode % n, i % n)) *
        to_modal(mode / n, i / n);
    }
    for (int i = 0; i <= j; ++i)
    {
      for (int k = i; k <= j; ++k)
      {
        modal_decay[i + dof * j] += inverse_upper[i + dof * k] *
          weights[mode_order[k]] * upper[k + dof * j];
      }
    }
  }
  const long double nodal_norm = InfinityNorm(nodal, dof);
  const long double modal_norm = InfinityNorm(modal, dof);
  const long double basis_condition = nodal_norm * modal_norm;
  MFEM_VERIFY(std::isfinite(basis_condition) &&
                basis_condition * epsilon <= 1.0e-5L,
              "Curved modal damping nodal basis is too ill-conditioned.");
  long double basis_residual = 0.0L;
  for (int j = 0; j < dof; ++j)
  {
    for (int i = 0; i < dof; ++i)
    {
      long double identity = 0.0L;
      for (int k = 0; k < dof; ++k)
      {
        identity += nodal[i + dof * k] * modal[k + dof * j];
        image[i + dof * j] += nodal[i + dof * k] * modal_decay[k + dof * j];
      }
      basis_residual = std::max(basis_residual,
                                std::abs(identity - (i == j ? 1.0L : 0.0L)));
    }
  }
  MFEM_VERIFY(basis_residual <= 128 * dof * epsilon * basis_condition,
              "Curved modal damping nodal basis inversion failed reconstruction.");
  HostMatrix stored_decay(size);
  for (int j = 0; j < dof; ++j)
  {
    for (int i = 0; i < dof; ++i)
    {
      long double value = 0.0L;
      for (int k = 0; k < dof; ++k)
      {
        value += image[i + dof * k] * modal[k + dof * j];
      }
      decay[i + dof * j] = static_cast<real_t>(value);
      MFEM_VERIFY(std::isfinite(decay[i + dof * j]),
                  "Curved modal damping produced a nonfinite source matrix.");
      stored_decay[i + dof * j] = decay[i + dof * j];
    }
  }
  long double decay_residual = 0.0L;
  for (int j = 0; j < dof; ++j)
  {
    for (int i = 0; i < dof; ++i)
    {
      long double value = 0.0L;
      for (int k = 0; k < dof; ++k)
      {
        value += stored_decay[i + dof * k] * nodal[k + dof * j];
      }
      decay_residual = std::max(decay_residual,
                                std::abs(value - image[i + dof * j]));
    }
  }
  MFEM_VERIFY(decay_residual <= 128 * dof * epsilon *
                (InfinityNorm(stored_decay, dof) * nodal_norm +
                 InfinityNorm(image, dof)),
              "Curved modal damping source matrix failed modal reconstruction.");
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
  MFEM_VERIFY(order >= 0 && order < std::numeric_limits<int>::max(),
              "Invalid modal damping solution order.");
  n1d_ = order + 1;
  MFEM_VERIFY(n1d_ <= std::numeric_limits<int>::max() / n1d_,
              "Modal damping basis matrix size exceeds integer indexing.");
  dof_ = 1;
  for (int axis = 0; axis < dimension_; ++axis)
  {
    MFEM_VERIFY(dof_ <= std::numeric_limits<int>::max() / n1d_,
                "Modal damping tensor size exceeds integer indexing.");
    dof_ *= n1d_;
  }
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
  MFEM_VERIFY(space.GetVDim() > 0 &&
                space.GetVDim() <= std::numeric_limits<int>::max() / dof_ &&
                space.GetNE() <= std::numeric_limits<int>::max() /
                                   (space.GetVDim() * dof_) &&
                entries_ == space.GetNE() * space.GetVDim() * dof_,
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
    if (!IsAffine(*space.GetElementTransformation(e), geometry))
    {
      MFEM_VERIFY(dimension_ == 2 && order <= max_curved_order,
                  "Curved modal damping supports 2D quadrilaterals with "
                  "solution orders zero through eight; affine tensor cells "
                  "retain their 1D--3D support.");
      if (curved_elements_.Size() == 0)
      {
        curved_operator_index_.SetSize(space.GetNE());
        curved_operator_index_ = -1;
      }
      curved_operator_index_[e] = curved_elements_.Size();
      curved_elements_.Append(e);
    }
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
  if (curved_elements_.Size() > 0)
  {
    const std::size_t matrix_entries = static_cast<std::size_t>(dof_) * dof_;
    const std::size_t max_entries = std::min(
      static_cast<std::size_t>(std::numeric_limits<int>::max()),
      max_curved_bytes / sizeof(real_t));
    MFEM_VERIFY(static_cast<std::size_t>(curved_elements_.Size()) <=
                  max_entries / matrix_entries,
                "Curved modal damping exceeds the 256 MiB local matrix "
                "storage bound or integer indexing.");
    curved_decay_.SetSize(
      static_cast<int>(matrix_entries * curved_elements_.Size()));
    std::vector<int> mode_order(dof_);
    std::iota(mode_order.begin(), mode_order.end(), 0);
    std::stable_sort(mode_order.begin(), mode_order.end(),
                     [this](int a, int b) { return weights_[a] < weights_[b]; });
    real_t* decay = curved_decay_.HostWrite();
    for (int i = 0; i < curved_elements_.Size(); ++i)
    {
      BuildCurvedDecay(*space.GetElementTransformation(curved_elements_[i]),
                       order, mode_order, weights_, to_nodal_, to_modal_,
                       decay + i * matrix_entries);
    }
    curved_decay_.UseDevice(true);
    curved_work_.SetSize(curved_elements_.Size() * space.GetVDim() * dof_);
    curved_work_.UseDevice(true);
  }
  if (curved_elements_.Size() < space.GetNE())
  {
    work_a_.SetSize(entries_);
    work_b_.SetSize(entries_);
    work_a_.UseDevice(true);
    work_b_.UseDevice(true);
  }
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
  const int vdim = space_.GetVDim();
  const int* map = lex_vdofs_.Read();
  const real_t* x = state.Read();
  const real_t rate = rate_;
  const int curved_count = curved_elements_.Size();
  const int* curved_elements =
    curved_count > 0 ? curved_elements_.Read() : nullptr;
  if (curved_count > 0)
  {
    real_t* packed = curved_work_.Write();
    mfem::forall(curved_work_.Size(), [=] MFEM_HOST_DEVICE(int i)
    {
      const int element = curved_elements[i / (vdim * dof)];
      const int component = (i / dof) % vdim;
      const int base = (element * vdim + component) * dof;
      // Apply the same exact constant anchor as in the affine transform.
      packed[i] = x[map[base + i % dof]] - x[map[base]];
    });
  }
  if (curved_count < space_.GetNE())
  {
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
    const real_t* source = input->Read();
    real_t* y = rhs.ReadWrite();
    if (curved_count == 0)
    {
      mfem::forall(entries_, [=] MFEM_HOST_DEVICE(int i)
      {
        y[map[i]] -= rate * source[i];
      });
    }
    else
    {
      const int* curved_index = curved_operator_index_.Read();
      mfem::forall(entries_, [=] MFEM_HOST_DEVICE(int i)
      {
        if (curved_index[i / (vdim * dof)] < 0)
        {
          y[map[i]] -= rate * source[i];
        }
      });
    }
  }
  if (curved_count > 0)
  {
    const real_t* decay = curved_decay_.Read();
    const real_t* packed = curved_work_.Read();
    real_t* y = rhs.ReadWrite();
    mfem::forall(curved_work_.Size(), [=] MFEM_HOST_DEVICE(int i)
    {
      const int cell = i / (vdim * dof);
      const int row = i % dof;
      const int packed_base = (i / dof) * dof;
      const int matrix_base = cell * dof * dof;
      real_t value = 0.0;
      for (int k = 0; k < dof; ++k)
      {
        value += decay[matrix_base + row + dof * k] * packed[packed_base + k];
      }
      const int element = curved_elements[cell];
      const int component = (i / dof) % vdim;
      const int base = (element * vdim + component) * dof;
      y[map[base + row]] -= rate * value;
    });
  }
}

} // namespace mfem
