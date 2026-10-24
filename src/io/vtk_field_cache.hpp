#pragma once

#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "physics/euler/diagnostics.hpp"

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace mfem
{
namespace io
{
namespace detail
{

// Private, per-collection reference data. The common writer sequence hits the
// next cached point directly; coordinates, not IP ordinals, establish identity.
// Signed zero is retained. Rule weights do not enter scalar VALUE CalcShape.
class VTKReferenceShapes
{
public:
  struct Point
  {
    std::array<unsigned char, 3*sizeof(real_t)> coordinates{};
    Vector shape;
  };
  struct Entry { std::vector<Point> points; };

  void Reset() { entries_.clear(); bytes_ = 0; }

  Entry* Find(const FiniteElement& fe)
  {
    auto found = entries_.find(&fe);
    if (found != entries_.end()) { return &found->second; }
    if (entries_.size() >= 32) { return nullptr; }
    return &entries_[&fe];
  }

  const Vector& Shape(const FiniteElement& fe, const IntegrationPoint& ip,
                      Entry* entry, std::size_t& cursor, Vector& temporary)
  {
    std::array<unsigned char, 3*sizeof(real_t)> key{};
    const real_t coordinates[3] = {ip.x, fe.GetDim() > 1 ? ip.y : real_t(0),
                                  fe.GetDim() > 2 ? ip.z : real_t(0)};
    std::memcpy(key.data(), coordinates, sizeof(coordinates));
    if (entry)
    {
      if (cursor < entry->points.size() && entry->points[cursor].coordinates == key)
      { return entry->points[cursor++].shape; }
      for (std::size_t i = 0; i < entry->points.size(); ++i)
      {
        if (entry->points[i].coordinates == key)
        { cursor = i+1; return entry->points[i].shape; }
      }
      const std::size_t bytes = sizeof(Point)+std::size_t(fe.GetDof())*sizeof(real_t);
      if (entry->points.size() < 4096 && bytes <= max_bytes_-bytes_)
      {
        entry->points.emplace_back();
        Point& point = entry->points.back();
        point.coordinates = key;
        point.shape.UseDevice(false);
        point.shape.SetSize(fe.GetDof());
        fe.CalcShape(ip, point.shape);
        bytes_ += bytes;
        cursor = entry->points.size();
        return point.shape;
      }
    }
    temporary.UseDevice(false);
    temporary.SetSize(fe.GetDof());
    fe.CalcShape(ip, temporary);
    return temporary;
  }

private:
  static constexpr std::size_t max_bytes_ = 8u*1024u*1024u;
  std::map<const FiniteElement*, Entry> entries_;
  std::size_t bytes_ = 0;
};

inline bool CanCacheVTKField(const GridFunction& field, const Mesh* mesh)
{
  const auto* fes = field.FESpace();
  if (!fes || fes->GetMesh() != mesh || fes->GetNE() <= 0 || fes->GetNURBSext() ||
      fes->IsVariableOrder() ||
      field.Size() != fes->GetVSize() || field.VectorDim() != fes->GetVDim())
  { return false; }
  for (int e = 0; e < fes->GetNE(); ++e)
  {
    const auto* fe = fes->GetFE(e);
    if (!fe || fe->GetRangeType() != FiniteElement::SCALAR ||
        fe->GetMapType() != FiniteElement::VALUE || fe->GetDim() < 1 || fe->GetDim() > 3 ||
        !dynamic_cast<const TensorBasisElement*>(fe))
    { return false; }
  }
  return true;
}

// Only one element's values are retained per field. Reset at every Save even
// when the mesh has one element and its identity/sequence did not change.
class VTKFieldEvaluation
{
public:
  VTKFieldEvaluation(const GridFunction& field, VTKReferenceShapes& shapes)
    : field_(field), shapes_(shapes)
  { local_.UseDevice(false); temporary_.UseDevice(false); }

  void Reset()
  {
    element_ = -1; fes_ = nullptr; fe_ = nullptr; reference_ = nullptr; cursor_ = 0;
    host_values_ = nullptr;
  }

  real_t Scalar(ElementTransformation& transformation, const IntegrationPoint& ip)
  {
    if (!Load(transformation)) { return field_.GetValue(transformation, ip); }
    const Vector& shape = shapes_.Shape(*fe_, ip, reference_, cursor_, temporary_);
    // Same MFEM Vector inner product and local DOF transform as GetValues.
    return shape*local_;
  }

  void VectorValue(Vector& value, ElementTransformation& transformation,
                   const IntegrationPoint& ip)
  {
    if (!Load(transformation)) { field_.GetVectorValue(transformation, ip, value); return; }
    const Vector& shape = shapes_.Shape(*fe_, ip, reference_, cursor_, temporary_);
    const int dof = fe_->GetDof(), components = fes_->GetVDim();
    value.SetSize(components);
    const real_t* local = local_.HostRead();
    for (int c = 0; c < components; ++c)
    { value(c) = shape*(local+dof*c); }
  }

private:
  bool Load(ElementTransformation& transformation)
  {
    const auto* fes = field_.FESpace();
    if (!fes || transformation.ElementType != ElementTransformation::ELEMENT ||
        transformation.ElementNo < 0 || transformation.ElementNo >= fes->GetNE())
    { return false; }
    const auto* fe = fes->GetFE(transformation.ElementNo);
    if (!fe || fe->GetRangeType() != FiniteElement::SCALAR ||
        fe->GetMapType() != FiniteElement::VALUE || fe->GetDim() < 1 || fe->GetDim() > 3)
    { return false; }
    if (fes_ == fes && element_ == transformation.ElementNo && fe_ == fe)
    { return true; }
    if (fes->GetNURBSext() || fes->IsVariableOrder() ||
        !dynamic_cast<const TensorBasisElement*>(fe)) { return false; }
    DofTransformation dof_transformation;
    if (fes->GetVDim() == 1)
    { fes->GetElementDofs(transformation.ElementNo, dofs_, dof_transformation); }
    else { fes->GetElementVDofs(transformation.ElementNo, dofs_, dof_transformation); }
    // GridFunction memory can be device-current. Synchronize once per field
    // and frame, then use the exact signed native gather on host. In particular
    // no Array/Vector device flags can trigger per-element CUDA gathers here.
    if (!host_values_) { host_values_ = field_.HostRead(); }
    local_.UseDevice(false);
    local_.SetSize(dofs_.Size());
    real_t* local = local_.HostWrite();
    const int* dofs = dofs_.HostRead();
    for (int i = 0; i < dofs_.Size(); ++i)
    {
      const int index = dofs[i];
      local[i] = index >= 0 ? host_values_[index] : -host_values_[-1-index];
    }
    dof_transformation.InvTransformPrimal(local_);
    local_.UseDevice(false);
    if (fe_ != fe) { reference_ = shapes_.Find(*fe); }
    fes_ = fes; fe_ = fe; element_ = transformation.ElementNo; cursor_ = 0;
    return true;
  }

  const GridFunction& field_;
  VTKReferenceShapes& shapes_;
  const FiniteElementSpace* fes_ = nullptr;
  const FiniteElement* fe_ = nullptr;
  const real_t* host_values_ = nullptr;
  VTKReferenceShapes::Entry* reference_ = nullptr;
  int element_ = -1;
  std::size_t cursor_ = 0;
  Array<int> dofs_;
  Vector local_, temporary_;
};

class CachedVTKScalar final : public Coefficient
{
public:
  explicit CachedVTKScalar(VTKFieldEvaluation& field) : field_(field) { }
  real_t Eval(ElementTransformation& transformation, const IntegrationPoint& ip) override
  { transformation.SetIntPoint(&ip); return field_.Scalar(transformation, ip); }
private:
  VTKFieldEvaluation& field_;
};

class CachedVTKVector final : public VectorCoefficient
{
public:
  CachedVTKVector(int dimension, VTKFieldEvaluation& field)
    : VectorCoefficient(dimension), field_(field) { }
  void Eval(Vector& value, ElementTransformation& transformation,
            const IntegrationPoint& ip) override
  { transformation.SetIntPoint(&ip); field_.VectorValue(value, transformation, ip); }
private:
  VTKFieldEvaluation& field_;
};

// Exact known pressure policy only. Interpolation is cached; the native
// PressureCoefficient's current U/dim/gamma and rho-floor/math are unchanged.
class CachedVTKPressure final : public Coefficient
{
public:
  CachedVTKPressure(PressureCoefficient& original, VTKReferenceShapes& shapes)
    : original_(original), shapes_(shapes) { }
  void Reset()
  {
    if (original_.U != source_)
    {
      source_ = original_.U;
      field_ = source_ ? std::make_unique<VTKFieldEvaluation>(*source_, shapes_) : nullptr;
    }
    if (field_) { field_->Reset(); }
  }
  real_t Eval(ElementTransformation& transformation, const IntegrationPoint& ip) override
  {
    if (original_.U != source_) { Reset(); }
    MFEM_VERIFY(field_, "PressureCoefficient: U pointer not set.");
    transformation.SetIntPoint(&ip);
    field_->VectorValue(state_, transformation, ip);
    return PressureFromCons(state_, original_.dim, original_.gamma, /*rho_floor=*/1e-16);
  }
private:
  PressureCoefficient& original_;
  VTKReferenceShapes& shapes_;
  const ParGridFunction* source_ = nullptr;
  std::unique_ptr<VTKFieldEvaluation> field_;
  Vector state_;
};

} // namespace detail
} // namespace io
} // namespace mfem
