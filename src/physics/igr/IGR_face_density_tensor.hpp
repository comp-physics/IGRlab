#pragma once

#include "igrlab/compat.hpp"
#include "igrlab/environment.hpp"
#include "mfem.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace igrlab
{
namespace igr_internal
{
namespace face_density_detail
{
enum class Execution { Dense, Tensor };

inline Execution RequestedExecution()
{
  const char* raw = ::igrlab::GetEnv("DG_IGR_IGR_FACE_DENSITY_EXECUTION");
  const std::string value = raw ? raw : "dense";
  MFEM_VERIFY(value == "dense" || value == "tensor",
              "IGRLAB_IGR_FACE_DENSITY_EXECUTION must be dense or tensor.");
  return value == "tensor" ? Execution::Tensor : Execution::Dense;
}

// Reference/layout data only. The density traces are overwritten from the
// current device state on every projection, including cache hits.
struct TensorData
{
  static constexpr int n = 6;
  dg_face::InteriorFaceSideDensityProjectionInfo info;
  // Native density vdof, in lexicographic tensor order [element][j][i].
  mfem::Array<int> density_map;
  // Incidence = 4*element + reference side, in the original side_q face order.
  mfem::Array<int> face_incidence, side_maps;
  mfem::Array<int> tangent_sparse, tangent_permutation;
  mfem::Vector endpoints, tangents, density_traces;
};

// This private carrier lets the existing Prepare -> Project API retain its
// geometry ownership without a process-global registry or a state cache.
class QuadratureSpace final : public mfem::FaceQuadratureSpace
{
public:
  using mfem::FaceQuadratureSpace::FaceQuadratureSpace;
  std::unique_ptr<TensorData> tensor;
};

struct LayoutKey
{
  std::array<const void*, 11> objects{};
  std::vector<long> metadata;
  std::vector<mfem::real_t> rule;
  bool operator==(const LayoutKey& other) const
  {
    return objects == other.objects && metadata == other.metadata && rule == other.rule;
  }
};

inline LayoutKey MakeLayoutKey(const mfem::FiniteElementSpace& vfes,
                               const mfem::FiniteElementSpace& sfes,
                               const mfem::IntegrationRule& ir,
                               Execution execution)
{
  const auto* mesh = vfes.GetMesh();
  const auto* nodes = mesh->GetNodes();
  const auto* nfes = mesh->GetNodalFESpace();
  const auto* nfe = nfes && nfes->GetNE() ? nfes->GetFE(0) : nullptr;
  const auto* vfe = vfes.GetFE(0);
  const auto* sfe = sfes.GetFE(0);
  LayoutKey key;
  key.objects = {{mesh, &vfes, &sfes, vfes.FEColl(), sfes.FEColl(),
                  vfe, sfe, nodes, nfes, nfes ? nfes->FEColl() : nullptr, nfe}};
  key.metadata = {mesh->GetSequence(), mesh->GetNodesSequence(),
    vfes.GetSequence(), sfes.GetSequence(), vfes.GetNE(), sfes.GetNE(),
    vfes.GetVSize(), sfes.GetVSize(), vfes.GetNDofs(), sfes.GetNDofs(),
    vfes.GetVDim(), sfes.GetVDim(), static_cast<long>(vfes.GetOrdering()),
    static_cast<long>(sfes.GetOrdering()), vfe->GetDof(), sfe->GetDof(),
    vfe->GetOrder(), sfe->GetOrder(), vfe->GetMapType(), sfe->GetMapType(),
    vfe->GetGeomType(), sfe->GetGeomType(), mesh->GetNumFaces(),
    nfes ? nfes->GetSequence() : -1, nfes ? nfes->GetVSize() : -1,
    nfes ? nfes->GetVDim() : -1, nfes ? static_cast<long>(nfes->GetOrdering()) : -1,
    nfe ? nfe->GetOrder() : -1, nfe ? nfe->GetDof() : -1,
    nfe ? nfe->GetMapType() : -1, nfe ? nfe->GetGeomType() : -1,
    ir.GetOrder(), static_cast<long>(execution)};
  key.rule.reserve(2 * ir.GetNPoints());
  for (int q = 0; q < ir.GetNPoints(); ++q)
  {
    const auto& point = ir.IntPoint(q);
    // Segment rules are completely determined by x and weight.
    key.rule.insert(key.rule.end(), {point.x, point.weight});
  }
  return key;
}

inline bool PositiveJacobian(mfem::ElementTransformation& transform,
                             const mfem::IntegrationPoint& point)
{
  transform.SetIntPoint(&point);
  const auto& jacobian = transform.Jacobian();
  if (jacobian.Height() != 2 || jacobian.Width() != 2) { return false; }
  const mfem::real_t a = jacobian(0, 0), b = jacobian(0, 1);
  const mfem::real_t c = jacobian(1, 0), d = jacobian(1, 1);
  const mfem::real_t determinant = a * d - b * c;
  return std::isfinite(a) && std::isfinite(b) && std::isfinite(c) &&
    std::isfinite(d) && std::isfinite(determinant) && determinant > 0.0 &&
    std::isfinite(transform.Weight()) && transform.Weight() > 0.0;
}

inline bool BuildTensorData(const mfem::FiniteElementSpace& vfes,
                            const mfem::FiniteElementSpace& sfes,
                            const mfem::FaceQuadratureSpace& fqs,
                            TensorData& data,
                            dg_face::InteriorFaceSideDensityProjectionScratch& work,
                            std::string& reason)
{
  const auto reject = [&](const char* why) { reason = why; return false; };
  reason.clear();
  auto* mesh = vfes.GetMesh();
  if (!mfem::Device::Allows(mfem::Backend::CUDA) ||
      sizeof(mfem::real_t) != sizeof(double)) { return reject("backend"); }
  if (!mesh || sfes.GetMesh() != mesh || fqs.GetMesh() != mesh ||
      mesh->Dimension() != 2 || mesh->SpaceDimension() != 2 ||
      !mesh->Conforming() || mesh->NURBSext || vfes.GetNE() <= 0 ||
      sfes.GetNE() != vfes.GetNE() || vfes.IsVariableOrder() || sfes.IsVariableOrder() ||
      vfes.GetVDim() != 4 || sfes.GetVDim() != 1 ||
      !dynamic_cast<const mfem::L2_FECollection*>(vfes.FEColl()) ||
      !dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl()))
    { return reject("state-layout"); }
#ifdef MFEM_USE_MPI
  const auto* pvfes = dynamic_cast<const mfem::ParFiniteElementSpace*>(&vfes);
  const auto* psfes = dynamic_cast<const mfem::ParFiniteElementSpace*>(&sfes);
  if (!pvfes || !psfes) { return reject("mpi-space"); }
  int vranks = 0, sranks = 0;
  MPI_Comm_size(pvfes->GetComm(), &vranks);
  MPI_Comm_size(psfes->GetComm(), &sranks);
  if (vranks != 1 || sranks != 1 || pvfes->GetParMesh()->GetNSharedFaces() != 0)
    { return reject("mpi"); }
#endif
  constexpr int n = TensorData::n, dof = n * n, order = n - 1;
  const auto valid_fe = [](const mfem::FiniteElement& fe)
  {
    const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&fe);
    return typeid(fe) == typeid(mfem::L2_QuadrilateralElement) && tensor &&
      tensor->GetBasisType() == mfem::BasisType::GaussLegendre &&
      fe.GetOrder() == order && fe.GetDof() == dof && fe.GetNodes().Size() == dof &&
      fe.GetGeomType() == mfem::Geometry::SQUARE &&
      fe.GetMapType() == mfem::FiniteElement::VALUE &&
      fe.GetRangeType() == mfem::FiniteElement::SCALAR;
  };
  const auto& reference = *vfes.GetFE(0);
  const auto& scalar_reference = *sfes.GetFE(0);
  if (!valid_fe(reference) || !valid_fe(scalar_reference))
    { return reject("requires-p5-gl-l2"); }
  const auto& line_rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * order + 1);
  const auto rule_matches = [&](const mfem::IntegrationRule& rule)
  {
    if (rule.GetNPoints() != n || line_rule.GetNPoints() != n) { return false; }
    for (int q = 0; q < n; ++q)
      if (rule.IntPoint(q).x != line_rule.IntPoint(q).x ||
          rule.IntPoint(q).weight != line_rule.IntPoint(q).weight) { return false; }
    return true;
  };
  if (fqs.GetFaceType() != mfem::FaceType::Interior || fqs.GetNE() <= 0 ||
      !rule_matches(fqs.GetIntRule(0))) { return reject("requires-gl6-face-rule"); }
  const auto checked = [](std::size_t count)
  { return count <= static_cast<std::size_t>(std::numeric_limits<int>::max()); };
  const int ne = vfes.GetNE(), nf = dg_face::CountLocalInteriorFaces(vfes, fqs);
  if (!checked(std::size_t(ne) * dof) || !checked(std::size_t(ne) * 4 * n) ||
      !checked(std::size_t(nf) * 2 * n)) { return reject("native-index-range"); }

  // Validate the actual native-node permutation, including scalar layout.
  std::array<int, dof> local{};
  for (int field = 0; field < 2; ++field)
  {
    const auto& fe = field ? scalar_reference : reference;
    const auto& permutation = dynamic_cast<const mfem::TensorBasisElement&>(fe).GetDofMap();
    if (permutation.Size() && permutation.Size() != dof) { return reject("native-node-map"); }
    std::array<bool, dof> used{};
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
    {
      const int lex = i + n * j, native = permutation.Size() ? permutation[lex] : lex;
      if (native < 0 || native >= dof || used[static_cast<std::size_t>(native)])
        { return reject("native-node-map"); }
      used[static_cast<std::size_t>(native)] = true;
      const auto& point = fe.GetNodes().IntPoint(native);
      if (point.x != line_rule.IntPoint(i).x || point.y != line_rule.IntPoint(j).x)
        { return reject("native-node-map"); }
      if (!field) { local[static_cast<std::size_t>(lex)] = native; }
    }
  }

  data.density_map.SetSize(ne * dof);
  auto* density_map = data.density_map.HostWrite();
  mfem::Array<int> element_dofs;
  for (int e = 0; e < ne; ++e)
  {
    if (mesh->GetElementBaseGeometry(e) != mfem::Geometry::SQUARE ||
        vfes.GetFE(e) != &reference || sfes.GetFE(e) != &scalar_reference ||
        vfes.GetElementDofs(e, element_dofs) || element_dofs.Size() != dof)
      { return reject("state-layout"); }
    for (int lex = 0; lex < dof; ++lex)
    {
      const int sdof = element_dofs[local[static_cast<std::size_t>(lex)]];
      if (sdof < 0 || sdof >= vfes.GetNDofs()) { return reject("native-density-map"); }
      const int native = vfes.DofToVDof(sdof, 0);
      if (native < 0 || native >= vfes.GetVSize()) { return reject("native-density-map"); }
      density_map[e * dof + lex] = native;
    }
    if (sfes.GetElementDofs(e, element_dofs) || element_dofs.Size() != dof)
      { return reject("scalar-layout"); }
    for (int d = 0; d < dof; ++d)
      if (element_dofs[d] < 0 || element_dofs[d] >= sfes.GetNDofs())
        { return reject("scalar-layout"); }
    auto* transform = vfes.GetElementTransformation(e);
    const auto* iso = dynamic_cast<const mfem::IsoparametricTransformation*>(transform);
    const auto* geometry = iso ? iso->GetFE() : nullptr;
    const int geometry_order = transform ? transform->Order() : 0;
    if (!geometry || (geometry_order != 1 && geometry_order != 2) ||
        geometry->GetGeomType() != mfem::Geometry::SQUARE ||
        geometry->GetMapType() != mfem::FiniteElement::VALUE ||
        geometry->GetRangeType() != mfem::FiniteElement::SCALAR ||
        geometry->GetDof() != (geometry_order + 1) * (geometry_order + 1) ||
        (typeid(*geometry) != typeid(mfem::BiLinear2DFiniteElement) &&
         typeid(*geometry) != typeid(mfem::H1_QuadrilateralElement) &&
         typeid(*geometry) != typeid(mfem::L2_QuadrilateralElement)))
      { return reject("requires-q1-q2-geometry"); }
    const auto& coordinates = iso->GetPointMat();
    if (coordinates.Height() != 2 || coordinates.Width() != geometry->GetDof())
      { return reject("geometry-layout"); }
    for (int j = 0; j < coordinates.Width(); ++j) for (int d = 0; d < 2; ++d)
      if (!std::isfinite(coordinates(d, j))) { return reject("geometry"); }
    // Setup samples reject observed invalid metrics. They do not certify Q2
    // positivity between samples; density evaluation itself needs no metric.
    const mfem::IntegrationRule* samples[] = {
      mfem::Geometries.GetVertices(mfem::Geometry::SQUARE),
      &geometry->GetNodes(), &reference.GetNodes()};
    for (const auto* samples_rule : samples)
      for (int q = 0; q < samples_rule->GetNPoints(); ++q)
        if (!PositiveJacobian(*transform, samples_rule->IntPoint(q)))
          { return reject("geometry"); }
  }

  mfem::L2_FECollection line_collection(order, 1, mfem::BasisType::GaussLegendre);
  const auto& line = *line_collection.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
  mfem::Vector line_shape(n), shape(dof);
  data.endpoints.SetSize(2 * n);
  auto* endpoints = data.endpoints.HostWrite();
  for (int end = 0; end < 2; ++end)
  {
    mfem::IntegrationPoint point;
    point.x = end;
    line.CalcShape(point, line_shape);
    for (int k = 0; k < n; ++k) { endpoints[end * n + k] = line_shape[k]; }
  }
  data.tangents.SetSize(8 * n * n);
  data.tangents = 0.0;
  auto* tangents = data.tangents.HostWrite();
  data.side_maps.SetSize(4 * ne);
  data.side_maps = -1;
  auto* side_maps = data.side_maps.HostWrite();
  data.face_incidence.SetSize(2 * nf);
  auto* incidence = data.face_incidence.HostWrite();
  work.face_elements.SetSize(2 * nf);
  auto* elements = work.face_elements.HostWrite();
  std::array<bool, 8> ready{};
  std::array<mfem::real_t, 8 * n> tangent_points{};
  const auto map_side = [&](mfem::FaceElementTransformations& tr, bool second,
                            const mfem::IntegrationRule& rule, int& result)
  {
    tr.SetAllIntPoints(&rule.IntPoint(0));
    const auto& first = second ? tr.GetElement2IntPoint() : tr.GetElement1IntPoint();
    const int side = first.x == 0.0 ? 0 : first.x == 1.0 ? 1 :
                     first.y == 0.0 ? 2 : first.y == 1.0 ? 3 : -1;
    if (side < 0) { return false; }
    const auto t0 = side < 2 ? first.y : first.x;
    const bool reverse = std::abs(t0 - line_rule.IntPoint(n - 1).x) <
                         std::abs(t0 - line_rule.IntPoint(0).x);
    const int signature = 2 * side + static_cast<int>(reverse);
    const int e = second ? tr.Elem2No : tr.Elem1No;
    if (e < 0 || e >= ne) { return false; }
    result = 4 * e + side;
    if (side_maps[result] >= 0) { return false; }
    for (int q = 0; q < n; ++q)
    {
      tr.SetAllIntPoints(&rule.IntPoint(q));
      const auto& point = second ? tr.GetElement2IntPoint() : tr.GetElement1IntPoint();
      const auto fixed = side < 2 ? point.x : point.y;
      const auto t = side < 2 ? point.y : point.x;
      const auto expected = reverse ? 1.0 - rule.IntPoint(q).x : rule.IntPoint(q).x;
      auto* transform = second ? tr.Elem2 : tr.Elem1;
      if (fixed != mfem::real_t(side % 2) || !std::isfinite(t) ||
          std::abs(t - expected) > 32 * std::numeric_limits<mfem::real_t>::epsilon() ||
          !transform || !PositiveJacobian(*transform, point)) { return false; }
      if (ready[static_cast<std::size_t>(signature)])
      {
        if (t != tangent_points[static_cast<std::size_t>(signature * n + q)])
          { return false; }
      }
      else
      {
        tangent_points[static_cast<std::size_t>(signature * n + q)] = t;
        mfem::IntegrationPoint line_point;
        line_point.x = t;
        line.CalcShape(line_point, line_shape);
        for (int k = 0; k < n; ++k)
          tangents[(signature * n + q) * n + k] = line_shape[k];
        reference.CalcShape(point, shape);
        for (int lex = 0; lex < dof; ++lex)
        {
          const int i = lex % n, j = lex / n;
          const auto value = endpoints[(side % 2) * n + (side < 2 ? i : j)] *
                             line_shape[side < 2 ? j : i];
          const auto native = shape[local[static_cast<std::size_t>(lex)]];
          if (!std::isfinite(value) || !std::isfinite(native) ||
              std::abs(native - value) > 64 * dof * std::numeric_limits<mfem::real_t>::epsilon() *
                std::max(mfem::real_t(1.0), std::abs(native))) { return false; }
        }
      }
    }
    ready[static_cast<std::size_t>(signature)] = true;
    side_maps[result] = signature;
    return true;
  };
  int packed = 0;
  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    const int mesh_face = fqs.GetMeshFaceIndex(f);
    if (mesh->GetFaceInformation(mesh_face).IsShared()) { return reject("mpi"); }
    const auto& rule = fqs.GetIntRule(f);
    if (fqs.GetGeometry(f) != mfem::Geometry::SEGMENT || !rule_matches(rule))
      { return reject("requires-gl6-face-rule"); }
    auto* tr = mesh->GetFaceElementTransformations(mesh_face);
    if (!tr || !tr->Elem1 || !tr->Elem2 || packed >= nf)
      { return reject("face-layout"); }
    for (int side = 0; side < 2; ++side)
    {
      int mapped = -1;
      if (!map_side(*tr, side != 0, rule, mapped)) { return reject("reference-face-map"); }
      incidence[side + 2 * packed] = mapped;
      elements[side + 2 * packed] = mapped / 4;
    }
    ++packed;
  }
  if (packed != nf) { return reject("face-count"); }
  data.tangent_sparse.SetSize(8);
  data.tangent_permutation.SetSize(8 * n);
  auto* sparse = data.tangent_sparse.HostWrite();
  auto* permutation = data.tangent_permutation.HostWrite();
  for (int map = 0; map < 8; ++map)
  {
    bool collocated = ready[static_cast<std::size_t>(map)];
    for (int q = 0; q < n; ++q)
    {
      const int pivot = map % 2 ? n - 1 - q : q;
      permutation[map * n + q] = pivot;
      for (int k = 0; k < n; ++k)
      {
        const auto value = tangents[(map * n + q) * n + k];
        if ((k != pivot && value != 0.0) ||
            (k == pivot && (!std::isfinite(value) || value == 0.0))) { collocated = false; }
      }
    }
    sparse[map] = collocated ? 1 : 0;
  }
  data.density_traces.SetSize(4 * n * ne);
  data.density_map.GetMemory().UseDevice(true);
  data.face_incidence.GetMemory().UseDevice(true);
  data.side_maps.GetMemory().UseDevice(true);
  data.tangent_sparse.GetMemory().UseDevice(true);
  data.tangent_permutation.GetMemory().UseDevice(true);
  work.face_elements.GetMemory().UseDevice(true);
  data.endpoints.UseDevice(true); data.tangents.UseDevice(true);
  data.density_traces.UseDevice(true);
  data.info.dim = 2; data.info.dof = dof; data.info.vdim = vfes.GetVDim();
  data.info.ne = ne; data.info.nf = nf; data.info.nq = n;
  data.info.face_element_size = work.face_elements.Size();
  // These report the actual generic arrays, which remain unallocated here.
  data.info.shape_size = 0; data.info.element_vdof_map_size = 0;
  return true;
}
} // namespace face_density_detail
} // namespace igr_internal
} // namespace igrlab
