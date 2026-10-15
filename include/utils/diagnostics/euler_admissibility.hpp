#pragma once

#include "igrlab/compat.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cmath>
#include <limits>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mfem.hpp"
#include "physics/euler/eos.hpp"
#include "utils/config/env_flags.hpp"
#include "utils/diagnostics/detail/euler_admissibility_device.hpp"
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

/**
 * @file utils/diagnostics/euler_admissibility.hpp
 * @brief Helpers to scan Euler states for density/pressure admissibility.
 * @ingroup utils_config
 */

namespace mfem
{
namespace igrlab
{

enum class EulerAdmissibilityPointSet
{
  Dof,
  Limiter,
  Output
};

struct EulerAdmissibilityIssue
{
  bool found = false;
  bool pressure_bad = false;
  bool face_point = false;
  int element = -1;
  int face = -1;
  int point_index = -1;
  int scalar_dof = -1;
  double rho = std::numeric_limits<double>::infinity();
  double pressure = std::numeric_limits<double>::infinity();
};

struct EulerAdmissibilitySummary
{
  double global_min_rho = std::numeric_limits<double>::infinity();
  double global_min_pressure = std::numeric_limits<double>::infinity();
  int global_bad_rho = 0;
  int global_bad_pressure = 0;
  int first_bad_rank = -1;
  EulerAdmissibilityIssue first_local_issue;
};

namespace detail
{

struct ElementFaceCacheEntry
{
  long mesh_sequence = -1;
  int ne = -1;
  std::vector<std::vector<int>> elem_faces;
};

inline const std::vector<std::vector<int>>& GetElementFacesForAdmissibility(
  mfem::Mesh& mesh)
{
  static std::unordered_map<const mfem::Mesh*, ElementFaceCacheEntry> cache;
  auto& entry = cache[&mesh];
  if (entry.mesh_sequence == mesh.GetSequence() && entry.ne == mesh.GetNE())
  {
    return entry.elem_faces;
  }

  entry.mesh_sequence = mesh.GetSequence();
  entry.ne = mesh.GetNE();
  entry.elem_faces.clear();
  entry.elem_faces.resize(entry.ne);

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1 = -1, e2 = -1;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e1 >= 0 && e1 < entry.ne)
    {
      entry.elem_faces[e1].push_back(f);
    }
    if (e2 >= 0 && e2 < entry.ne)
    {
      entry.elem_faces[e2].push_back(f);
    }
  }

  return entry.elem_faces;
}

inline std::vector<int>
BuildFaceQuadratureOrdersForAdmissibility(const mfem::ParFiniteElementSpace& vfes,
                                          int volume_qord,
                                          const std::vector<int>& configured_face_qords)
{
  std::vector<int> qords;
  qords.reserve(configured_face_qords.size() + 2);

  auto push_unique_positive = [&](int qord)
  {
    if (qord <= 0)
    {
      return;
    }
    if (std::find(qords.begin(), qords.end(), qord) == qords.end())
    {
      qords.push_back(qord);
    }
  };

  push_unique_positive(volume_qord);
  for (const int qord : configured_face_qords)
  {
    push_unique_positive(qord);
  }

  if (qords.size() == 1)
  {
    push_unique_positive(2 * vfes.GetFE(0)->GetOrder() + 1);
  }

  return qords;
}

inline int FindFirstBadRank(bool found, MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int first_bad_rank = found ? rank : std::numeric_limits<int>::max();
  MPI_Allreduce(MPI_IN_PLACE,
                &first_bad_rank,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_MIN,
                comm);
  if (first_bad_rank == std::numeric_limits<int>::max())
  {
    return -1;
  }
  return first_bad_rank;
}

// These plans contain only invariant reference interpolation and connectivity.
// State values, diagnostic results and MPI reductions are never cached.
struct AdmissibilityShapeTable
{
  int dof = 0;
  int points = 0;
  std::vector<mfem::real_t> values;
};

struct AdmissibilityElementMap
{
  const mfem::FiniteElement* fe = nullptr;
  mfem::Array<int> vdofs;
  int dof = 0;
  int order = 0;
  bool enforce_nodes = false;
};

struct AdmissibilityFaceRule
{
  int face = -1;
  int shape_table = -1;
};

struct AdmissibilityElementSamples
{
  int volume_shape_table = -1;
  std::vector<AdmissibilityFaceRule> face_rules;
};

struct AdmissibilityReferenceRule
{
  int geometry = -1;
  int order = -1;
  std::vector<std::array<mfem::real_t, 2>> points;
};

inline bool AdmissibilityNativeHostEnvironment()
{
#if defined(__SSE2__) && !defined(__FAST_MATH__)
  const unsigned int csr = _mm_getcsr();
  return std::numeric_limits<mfem::real_t>::is_iec559 &&
    sizeof(mfem::real_t) == sizeof(double) &&
    std::fegetround() == FE_TONEAREST && (csr & 0xe040u) == 0 &&
    (csr & 0x1f80u) == 0x1f80u;
#else
  return false;
#endif
}

class EulerAdmissibilityCache
{
public:
  EulerAdmissibilitySummary Scan(
    const mfem::ParFiniteElementSpace& vfes,
    const mfem::Vector& v,
    int dim,
    double gamma,
    double rho_min,
    double p_min,
    int quad_order,
    const std::vector<int>& face_quad_orders,
    EulerAdmissibilityPointSet point_set);

private:
  // An optional acceleration budget, not a restriction on supported orders.
  // The three p5, 128x128 scan plans charge about 121 MiB together, including
  // packing temporaries and both host/device mirrors. Keep all three within
  // this bounded allowance; otherwise the later scans silently fall back to
  // the host even though their device payloads fit. Larger or unusual spaces
  // retain the original scanner, and device-memory admission is unchanged.
  static constexpr std::size_t byte_budget = 128u * 1024u * 1024u;
  struct BudgetExceeded {};

  void Charge(std::size_t bytes)
  {
    if (bytes > byte_budget - bytes_used_) { throw BudgetExceeded{}; }
    bytes_used_ += bytes;
  }

  void ChargeProduct(std::size_t count, std::size_t bytes)
  {
    if (bytes != 0 && count > (byte_budget - bytes_used_) / bytes)
    { throw BudgetExceeded{}; }
    Charge(count * bytes);
  }

  bool Matches(const mfem::ParFiniteElementSpace& vfes,
               int dim,
               int quad_order,
               const std::vector<int>& face_quad_orders) const
  {
    const mfem::Mesh* mesh = vfes.GetMesh();
    if (&vfes != vfes_ || mesh != mesh_ || !mesh ||
        vfes.GetSequence() != fes_sequence_ ||
        mesh->GetSequence() != mesh_sequence_ || vfes.FEColl() != fec_ ||
        vfes.GetNE() != ne_ || vfes.GetVDim() != num_eq_ || dim != dim_ ||
        vfes.GetOrdering() != ordering_ || quad_order != quad_order_ ||
        face_quad_orders != face_quad_orders_)
    { return false; }
    if (eligible_)
    {
      const auto* l2 = dynamic_cast<const mfem::L2_FECollection*>(vfes.FEColl());
      if (!l2 || l2->GetBasisType() != basis_) { return false; }
      for (int e = 0; e < ne_; ++e)
      {
        const auto* fe = vfes.GetFE(e);
        if (fe != elements_[e].fe || fe->GetDof() != elements_[e].dof ||
            fe->GetOrder() != elements_[e].order ||
            fe->GetGeomType() != mfem::Geometry::SQUARE ||
            fe->GetMapType() != mfem::FiniteElement::VALUE)
        { return false; }
      }
      // Public MFEM rules can be replaced in place without a mesh update.
      // Check their reference coordinates as well as the selected orders.
      for (const auto& saved : reference_rules_)
      {
        const auto& rule = mfem::IntRules.Get(saved.geometry, saved.order);
        if (rule.GetNPoints() != static_cast<int>(saved.points.size()))
        { return false; }
        for (int q = 0; q < rule.GetNPoints(); ++q)
        {
          const auto& point = rule.IntPoint(q);
          if (std::memcmp(&point.x, &saved.points[q][0], sizeof(point.x)) ||
              (saved.geometry == mfem::Geometry::SQUARE &&
               std::memcmp(&point.y, &saved.points[q][1], sizeof(point.y))))
          { return false; }
        }
      }
    }
    return true;
  }

  void Reset(const mfem::ParFiniteElementSpace& vfes,
             int dim,
             int quad_order,
             const std::vector<int>& face_quad_orders)
  {
    // Release prior generations, including capacities, before charging the
    // next bounded plan. Nothing here owns device work or conservative data.
    std::vector<AdmissibilityElementMap>().swap(elements_);
    std::vector<AdmissibilityShapeTable>().swap(shape_tables_);
    std::unordered_map<std::string, int>().swap(shape_indices_);
    std::vector<AdmissibilityReferenceRule>().swap(reference_rules_);
    for (auto& plan : plans_)
    { std::vector<AdmissibilityElementSamples>().swap(plan); }
    for (auto& plan : device_plans_) { plan.reset(); }
    device_declined_.fill(false);
    device_shape_environment_.fill(false);
    device_shape_values_qualified_ = true;
    ready_.fill(false);
    declined_.fill(false);
    bytes_used_ = 0;
    eligible_ = false;
    vfes_ = &vfes;
    mesh_ = vfes.GetMesh();
    fes_sequence_ = vfes.GetSequence();
    mesh_sequence_ = mesh_ ? mesh_->GetSequence() : -1;
    fec_ = vfes.FEColl();
    ne_ = vfes.GetNE();
    num_eq_ = vfes.GetVDim();
    ordering_ = vfes.GetOrdering();
    dim_ = dim;
    quad_order_ = quad_order;
    face_quad_orders_ = face_quad_orders;

    const auto* l2 = dynamic_cast<const mfem::L2_FECollection*>(fec_);
    if (!mesh_ || !l2 || dim != 2 || ne_ <= 0 || num_eq_ != 4 ||
        ordering_ != mfem::Ordering::byNODES || mesh_->Nonconforming() ||
        vfes.Nonconforming())
    { return; }
    basis_ = l2->GetBasisType();
    const auto* first_fe = vfes.GetFE(0);
    if (first_fe->GetGeomType() != mfem::Geometry::SQUARE ||
        first_fe->GetMapType() != mfem::FiniteElement::VALUE) { return; }
    for (int e = 1; e < ne_; ++e)
    {
      // A conservative initial fast path; mixed/variable spaces retain the
      // original exact element-by-element implementation.
      if (vfes.GetFE(e) != first_fe) { return; }
    }

    try
    {
      ChargeProduct(static_cast<std::size_t>(ne_),
                    sizeof(AdmissibilityElementMap));
      elements_.resize(ne_);
      for (int e = 0; e < ne_; ++e)
      {
        auto& element = elements_[e];
        element.fe = vfes.GetFE(e);
        element.dof = element.fe->GetDof();
        element.order = element.fe->GetOrder();
        element.enforce_nodes = basis_ == mfem::BasisType::GaussLobatto;
        ChargeProduct(static_cast<std::size_t>(element.dof),
                      static_cast<std::size_t>(num_eq_) * sizeof(int));
        vfes.GetElementVDofs(e, element.vdofs);
      }
      eligible_ = true;
    }
    catch (const BudgetExceeded&)
    { std::vector<AdmissibilityElementMap>().swap(elements_); }
    catch (const std::bad_alloc&)
    { std::vector<AdmissibilityElementMap>().swap(elements_); }
  }

  void RememberRule(int geometry, int order, const mfem::IntegrationRule& rule)
  {
    for (const auto& saved : reference_rules_)
    {
      if (saved.geometry == geometry && saved.order == order) { return; }
    }
    ChargeProduct(static_cast<std::size_t>(rule.GetNPoints()),
                  sizeof(std::array<mfem::real_t, 2>));
    Charge(2 * sizeof(AdmissibilityReferenceRule));
    AdmissibilityReferenceRule saved;
    saved.geometry = geometry;
    saved.order = order;
    saved.points.resize(rule.GetNPoints());
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      saved.points[q] = {rule.IntPoint(q).x,
        geometry == mfem::Geometry::SQUARE ? rule.IntPoint(q).y : 0.0};
    }
    reference_rules_.push_back(std::move(saved));
  }

  int ShapeTable(const mfem::FiniteElement& fe,
                 const std::vector<mfem::IntegrationPoint>& points)
  {
    // CalcShape for the admitted scalar 2D L2 basis depends on reference x/y.
    // Exact coordinate bytes retain signed zero and do not merge nearby points.
    const auto identity = reinterpret_cast<std::uintptr_t>(&fe);
    std::string key(reinterpret_cast<const char*>(&identity), sizeof(identity));
    if (points.size() > byte_budget / (2 * sizeof(mfem::real_t)))
    { throw BudgetExceeded{}; }
    key.reserve(sizeof(identity) + points.size() * 2 * sizeof(mfem::real_t));
    for (const auto& point : points)
    {
      key.append(reinterpret_cast<const char*>(&point.x), sizeof(point.x));
      key.append(reinterpret_cast<const char*>(&point.y), sizeof(point.y));
    }
    const auto existing = shape_indices_.find(key);
    if (existing != shape_indices_.end()) { return existing->second; }
    // Tables may be shared between point sets. One unsupported construction
    // context conservatively keeps all interpolated scans on the host until
    // the original cache generation is invalidated.
    device_shape_values_qualified_ = device_shape_values_qualified_ &&
                                      AdmissibilityNativeHostEnvironment();

    const int dof = fe.GetDof();
    ChargeProduct(points.size(), static_cast<std::size_t>(dof) *
                                  sizeof(mfem::real_t));
    // Account conservatively for vector/map nodes and capacity growth.
    Charge(key.size() + 2 * sizeof(AdmissibilityShapeTable) + 128);
    AdmissibilityShapeTable table;
    table.dof = dof;
    table.points = static_cast<int>(points.size());
    table.values.resize(points.size() * static_cast<std::size_t>(dof));
    mfem::Vector shape(dof);
    for (std::size_t q = 0; q < points.size(); ++q)
    {
      fe.CalcShape(points[q], shape);
      std::copy_n(shape.GetData(), dof,
                  table.values.data() + q * static_cast<std::size_t>(dof));
    }
    const int index = static_cast<int>(shape_tables_.size());
    shape_tables_.push_back(std::move(table));
    shape_indices_.emplace(std::move(key), index);
    return index;
  }

  bool Prepare(const mfem::ParFiniteElementSpace& vfes,
               int dim,
               int quad_order,
               const std::vector<int>& face_quad_orders,
               EulerAdmissibilityPointSet point_set)
  {
    const int slot = static_cast<int>(point_set);
    if (slot < 0 || slot >= static_cast<int>(plans_.size())) { return false; }
    if (!Matches(vfes, dim, quad_order, face_quad_orders))
    { Reset(vfes, dim, quad_order, face_quad_orders); }
    if (!eligible_) { return false; }
    if (ready_[slot]) { return true; }
    if (declined_[slot]) { return false; }
    auto& plan = plans_[slot];
    if (point_set == EulerAdmissibilityPointSet::Dof)
    {
      device_shape_environment_[slot] = true; // No constructed shape arithmetic.
      ready_[slot] = true;
      return true;
    }

    try
    {
      const int order = elements_[0].order;
      const int base_qord = quad_order > 0 ? quad_order : 2 * order + 2;
      const int output_qord = std::max(base_qord + 2, 2 * order + 4);
      const int scan_qord = point_set == EulerAdmissibilityPointSet::Output
                              ? output_qord : base_qord;
      auto scan_face_qords = BuildFaceQuadratureOrdersForAdmissibility(
        vfes, scan_qord, face_quad_orders);
      if (point_set == EulerAdmissibilityPointSet::Output)
      {
        for (int& qord : scan_face_qords)
        { qord = std::max(qord + 2, 2 * order + 3); }
        std::sort(scan_face_qords.begin(), scan_face_qords.end());
        scan_face_qords.erase(
          std::unique(scan_face_qords.begin(), scan_face_qords.end()),
          scan_face_qords.end());
      }
      const auto& elem_faces = GetElementFacesForAdmissibility(*mesh_);
      ChargeProduct(static_cast<std::size_t>(ne_),
                    sizeof(AdmissibilityElementSamples));
      plan.resize(ne_);
      for (int e = 0; e < ne_; ++e)
      {
        const auto& fe = *elements_[e].fe;
        auto* transformation = vfes.GetElementTransformation(e);
        const auto& rule = mfem::IntRules.Get(
          transformation->GetGeometryType(), scan_qord);
        RememberRule(transformation->GetGeometryType(), scan_qord, rule);
        if (rule.GetNPoints() > static_cast<int>(byte_budget /
                                                sizeof(mfem::IntegrationPoint)))
        { throw BudgetExceeded{}; }
        std::vector<mfem::IntegrationPoint> points(rule.GetNPoints());
        for (int q = 0; q < rule.GetNPoints(); ++q)
        {
          points[q].Init(q);
          points[q].Set2(rule.IntPoint(q).x, rule.IntPoint(q).y);
        }
        plan[e].volume_shape_table = ShapeTable(fe, points);
        for (const int face : elem_faces[e])
        {
          auto* face_transformation = mesh_->GetFaceElementTransformations(face);
          if (!face_transformation) { continue; }
          int e1 = -1, e2 = -1;
          mesh_->GetFaceElements(face, &e1, &e2);
          const bool first = e1 == e;
          if (!first && e2 != e) { continue; }
          for (const int qord : scan_face_qords)
          {
            const auto& face_rule = mfem::IntRules.Get(
              face_transformation->GetGeometryType(), qord);
            RememberRule(face_transformation->GetGeometryType(), qord, face_rule);
            if (face_rule.GetNPoints() > static_cast<int>(byte_budget /
                                                   sizeof(mfem::IntegrationPoint)))
            { throw BudgetExceeded{}; }
            points.resize(face_rule.GetNPoints());
            for (int q = 0; q < face_rule.GetNPoints(); ++q)
            {
              face_transformation->SetAllIntPoints(&face_rule.IntPoint(q));
              const auto& reference_point = first
                ? face_transformation->GetElement1IntPoint()
                : face_transformation->GetElement2IntPoint();
              points[q].Init(q);
              points[q].Set2(reference_point.x, reference_point.y);
            }
            const int table = ShapeTable(fe, points);
            Charge(2 * sizeof(AdmissibilityFaceRule));
            plan[e].face_rules.push_back({face, table});
          }
        }
      }
      // A later change back to nearest rounding must not bless shape constants
      // constructed in an unsupported caller environment.
      device_shape_environment_[slot] = AdmissibilityNativeHostEnvironment();
      ready_[slot] = true;
      return true;
    }
    catch (const BudgetExceeded&)
    { std::vector<AdmissibilityElementSamples>().swap(plan); }
    catch (const std::bad_alloc&)
    { std::vector<AdmissibilityElementSamples>().swap(plan); }
    declined_[slot] = true;
    return false;
  }

  const mfem::ParFiniteElementSpace* vfes_ = nullptr;
  mfem::Mesh* mesh_ = nullptr;
  const mfem::FiniteElementCollection* fec_ = nullptr;
  long fes_sequence_ = -1;
  long mesh_sequence_ = -1;
  int ne_ = -1;
  int num_eq_ = -1;
  int dim_ = -1;
  int ordering_ = -1;
  int basis_ = -1;
  int quad_order_ = -1;
  std::vector<int> face_quad_orders_;
  bool eligible_ = false;
  std::size_t bytes_used_ = 0;
  std::vector<AdmissibilityElementMap> elements_;
  std::vector<AdmissibilityShapeTable> shape_tables_;
  std::unordered_map<std::string, int> shape_indices_;
  std::vector<AdmissibilityReferenceRule> reference_rules_;
  std::array<std::vector<AdmissibilityElementSamples>, 3> plans_;
  std::array<bool, 3> ready_{};
  std::array<bool, 3> declined_{};
  std::array<std::shared_ptr<AdmissibilityDevicePlan>, 3> device_plans_;
  std::array<bool, 3> device_declined_{};
  std::array<bool, 3> device_shape_environment_{};
  bool device_shape_values_qualified_ = true;
};

inline EulerAdmissibilitySummary
ScanEulerAdmissibilityImpl(const mfem::ParFiniteElementSpace& vfes,
                       const mfem::Vector& v,
                       int dim,
                       double gamma,
                       double rho_min,
                       double p_min,
                       int quad_order,
                       const std::vector<int>& face_quad_orders,
                       EulerAdmissibilityPointSet point_set,
                       const std::vector<AdmissibilityElementMap>* cached_elements,
                       const std::vector<AdmissibilityShapeTable>* cached_shapes,
                       const std::vector<AdmissibilityElementSamples>* cached_samples)
{
  EulerAdmissibilitySummary out;
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Euler admissibility scan assumes byNODES ordering.");

  const int num_eq = dim + 2;
  MFEM_VERIFY(vfes.GetVDim() == num_eq,
              "Euler admissibility scan expects Euler state layout.");

  mfem::Mesh& mesh = *vfes.GetMesh();
  const auto* elem_faces = cached_elements ? nullptr
    : &detail::GetElementFacesForAdmissibility(mesh);
  const int NE = vfes.GetNE();

  int scan_qord = 0;
  std::vector<int> scan_face_qords;
  if (!cached_elements)
  {
    // Prepared scans retain these occurrences in their reference shape plans.
    // Keep the original setup and getter order for the generic fallback.
    const int base_qord =
      (quad_order > 0) ? quad_order : (2 * vfes.GetFE(0)->GetOrder() + 2);
    const int output_qord = std::max(base_qord + 2, 2 * vfes.GetFE(0)->GetOrder() + 4);
    scan_qord =
      (point_set == EulerAdmissibilityPointSet::Output) ? output_qord : base_qord;

    scan_face_qords =
      detail::BuildFaceQuadratureOrdersForAdmissibility(vfes,
                                                        scan_qord,
                                                        face_quad_orders);
    if (point_set == EulerAdmissibilityPointSet::Output)
    {
      for (int& qord : scan_face_qords)
      {
        qord = std::max(qord + 2, 2 * vfes.GetFE(0)->GetOrder() + 3);
      }
      std::sort(scan_face_qords.begin(), scan_face_qords.end());
      scan_face_qords.erase(
        std::unique(scan_face_qords.begin(), scan_face_qords.end()),
        scan_face_qords.end());
    }
  }

  double local_min_rho = std::numeric_limits<double>::infinity();
  double local_min_pressure = std::numeric_limits<double>::infinity();
  int local_bad_rho = 0;
  int local_bad_pressure = 0;

  mfem::Array<int> vdofs;
  mfem::Vector elvec;
  mfem::DenseMatrix elmat;
  mfem::Vector shape, state(num_eq);
  // Prepared maps and the local element buffer are host-only in the original
  // GetSubVector dispatch. Acquire this scan's current input once; no values
  // or results are retained by the cache.
  const mfem::real_t* cached_input = cached_elements ? v.Read(false) : nullptr;

  auto inspect_state = [&](const mfem::Vector& U,
                           int elem,
                           int face,
                           int point_index,
                           bool face_point,
                           int scalar_dof)
  {
    const double rho = U(0);
    bool conservative_finite = true;
    for (int eq = 0; eq < num_eq; ++eq)
    {
      conservative_finite = conservative_finite && std::isfinite(U(eq));
    }

    double momentum_squared = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      momentum_squared += U(1 + d) * U(1 + d);
    }
    const double kinetic =
      0.5 * momentum_squared / std::max(rho, 1.0e-16);
    const double pressure =
      mfem::EulerPressureFromConservative(U, dim, gamma, /*rho_floor=*/1e-16);

    const bool rho_finite = std::isfinite(rho);
    const bool kinetic_finite = std::isfinite(kinetic);
    const bool pressure_finite = std::isfinite(pressure);
    if (rho_finite)
    {
      local_min_rho = std::min(local_min_rho, rho);
    }
    if (pressure_finite)
    {
      local_min_pressure = std::min(local_min_pressure, pressure);
    }

    const bool bad_rho = !rho_finite || rho < rho_min;
    // The existing pressure counter is also the thermodynamic/state-invalid
    // counter.  In particular, do not let NaN comparisons make a nonfinite
    // conservative component or derived kinetic/pressure value look valid.
    const bool bad_pressure = !conservative_finite || !kinetic_finite ||
                              !pressure_finite || pressure < p_min;
    local_bad_rho += bad_rho ? 1 : 0;
    local_bad_pressure += bad_pressure ? 1 : 0;

    if (!out.first_local_issue.found && (bad_rho || bad_pressure))
    {
      out.first_local_issue.found = true;
      out.first_local_issue.pressure_bad = !bad_rho && bad_pressure;
      out.first_local_issue.face_point = face_point;
      out.first_local_issue.element = elem;
      out.first_local_issue.face = face;
      out.first_local_issue.point_index = point_index;
      out.first_local_issue.scalar_dof = scalar_dof;
      out.first_local_issue.rho = rho;
      out.first_local_issue.pressure = pressure;
    }
  };

  for (int i = 0; i < NE; ++i)
  {
    const mfem::FiniteElement* fe = cached_elements ? (*cached_elements)[i].fe
      : vfes.GetFE(i);
    mfem::ElementTransformation* Tr = cached_elements ? nullptr
      : vfes.GetElementTransformation(i);
    const int dof = cached_elements ? (*cached_elements)[i].dof : fe->GetDof();
    const auto* fec_l2 = cached_elements ? nullptr
      : dynamic_cast<const mfem::L2_FECollection*>(vfes.FEColl());
    const bool enforce_nodes = cached_elements ? (*cached_elements)[i].enforce_nodes
      : (fec_l2 != nullptr && fec_l2->GetBasisType() == mfem::BasisType::GaussLobatto);

    if (cached_elements)
    {
      const auto& map = (*cached_elements)[i].vdofs;
      elvec.SetSize(map.Size());
      mfem::real_t* element_data = elvec.HostWrite();
      const int* indices = map.HostRead();
      for (int j = 0; j < map.Size(); ++j)
      {
        const int index = indices[j];
        element_data[j] = index >= 0 ? cached_input[index] : -cached_input[-index-1];
      }
    }
    else
    {
      vfes.GetElementVDofs(i, vdofs);
      v.GetSubVector(vdofs, elvec);
    }
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);
    if (!cached_elements) { shape.SetSize(dof); }
    mfem::real_t* cached_state_data = cached_elements ? state.HostWrite() : nullptr;

    const bool do_dof_scan = (point_set == EulerAdmissibilityPointSet::Dof);
    if (do_dof_scan || enforce_nodes)
    {
      for (int j = 0; j < dof; ++j)
      {
        if (cached_elements)
        {
          const mfem::real_t* element_data = elvec.GetData();
          for (int eq = 0; eq < num_eq; ++eq)
          { cached_state_data[eq] = element_data[j + eq * dof]; }
        }
        else { elmat.GetRow(j, state); }
        inspect_state(state, i, -1, j, false, j);
      }
      if (do_dof_scan)
      {
        continue;
      }
    }

    if (cached_samples)
    {
      // Prepared node copies preserve state size, so the same output storage
      // remains valid. Keep the compiled MFEM pointer-overload dot unchanged.
      mfem::real_t* state_data = cached_elements ? cached_state_data : state.HostWrite();
      const auto inspect_table = [&](int table_index, int face)
      {
        const auto& table = (*cached_shapes)[table_index];
        for (int q = 0; q < table.points; ++q)
        {
          // Same native shape values and MFEM summation as the reference scan.
          elmat.MultTranspose(
            table.values.data() + static_cast<std::size_t>(q) * dof,
            state_data);
          inspect_state(state, i, face, q, face >= 0, -1);
        }
      };
      const auto& samples = (*cached_samples)[i];
      inspect_table(samples.volume_shape_table, -1);
      for (const auto& face_rule : samples.face_rules)
      { inspect_table(face_rule.shape_table, face_rule.face); }
      continue;
    }

    const mfem::IntegrationRule& ir =
      mfem::IntRules.Get(Tr->GetGeometryType(), scan_qord);
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, state);
      inspect_state(state, i, -1, q, false, -1);
    }

    for (const int face : (*elem_faces)[i])
    {
      mfem::FaceElementTransformations* FTr =
        mesh.GetFaceElementTransformations(face);
      if (FTr == nullptr)
      {
        continue;
      }

      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(face, &e1, &e2);
      const bool elem_is_first = (e1 == i);
      const bool elem_is_second = (e2 == i);
      if (!elem_is_first && !elem_is_second)
      {
        continue;
      }

      for (const int fqord : scan_face_qords)
      {
        const mfem::IntegrationRule& fir =
          mfem::IntRules.Get(FTr->GetGeometryType(), fqord);
        for (int q = 0; q < fir.GetNPoints(); ++q)
        {
          const mfem::IntegrationPoint& fip = fir.IntPoint(q);
          FTr->SetAllIntPoints(&fip);
          const mfem::IntegrationPoint& eip = elem_is_first
                                                ? FTr->GetElement1IntPoint()
                                                : FTr->GetElement2IntPoint();
          fe->CalcShape(eip, shape);
          elmat.MultTranspose(shape, state);
          inspect_state(state, i, face, q, true, -1);
        }
      }
    }
  }

  MPI_Allreduce(&local_min_rho,
                &out.global_min_rho,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN,
                vfes.GetComm());
  MPI_Allreduce(&local_min_pressure,
                &out.global_min_pressure,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN,
                vfes.GetComm());
  MPI_Allreduce(&local_bad_rho,
                &out.global_bad_rho,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM,
                vfes.GetComm());
  MPI_Allreduce(&local_bad_pressure,
                &out.global_bad_pressure,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM,
                vfes.GetComm());
  out.first_bad_rank =
    detail::FindFirstBadRank(out.first_local_issue.found, vfes.GetComm());

  return out;
}

inline EulerAdmissibilitySummary EulerAdmissibilityCache::Scan(
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::Vector& v,
  int dim,
  double gamma,
  double rho_min,
  double p_min,
  int quad_order,
  const std::vector<int>& face_quad_orders,
  EulerAdmissibilityPointSet point_set)
{
  const bool prepared = Prepare(vfes, dim, quad_order, face_quad_orders, point_set);
  const bool dof_scan = point_set == EulerAdmissibilityPointSet::Dof;
  // Explicit opt-in is a native compiled-arithmetic qualification contract,
  // not a portable promise for an arbitrary MFEM binary or observable sticky
  // exception flags. The original CXX/MFEM implementation remains the oracle.
  if (prepared && AdmissibilityNativeHostEnvironment() &&
      device_shape_environment_[static_cast<int>(point_set)] &&
      (dof_scan || device_shape_values_qualified_) &&
      v.Size() == vfes.GetVSize() &&
      v.GetMemory().DeviceIsValid() &&
      (basis_ == mfem::BasisType::GaussLegendre ||
       basis_ == mfem::BasisType::GaussLobatto) &&
      AdmissibilityOrdinaryCuda() &&
      ::igrlab::EnvFlagEnabled("DG_IGR_EULER_ADMISSIBILITY_CUDA"))
  {
    int ranks = 0;
    MPI_Comm_size(vfes.GetComm(), &ranks);
    const int slot = static_cast<int>(point_set);
    if (ranks == 1 && !device_declined_[slot])
    {
      if (!device_plans_[slot])
      {
        std::size_t charged = 0;
        device_plans_[slot] = MakeAdmissibilityDevicePlan(
          elements_, shape_tables_, plans_[slot], point_set, v.Size(),
          byte_budget - bytes_used_, charged);
        if (device_plans_[slot]) { Charge(charged); }
        else { device_declined_[slot] = true; }
      }
      EulerAdmissibilitySummary local;
      if (device_plans_[slot] && EvaluateAdmissibilityDevice(
            *device_plans_[slot], v, gamma, rho_min, p_min, local))
      {
        EulerAdmissibilitySummary out;
        out.first_local_issue = local.first_local_issue;
        // Retain all five original communicator operations and their types.
        MPI_Allreduce(&local.global_min_rho, &out.global_min_rho, 1,
          mfem::MPITypeMap<double>::mpi_type, MPI_MIN, vfes.GetComm());
        MPI_Allreduce(&local.global_min_pressure, &out.global_min_pressure, 1,
          mfem::MPITypeMap<double>::mpi_type, MPI_MIN, vfes.GetComm());
        MPI_Allreduce(&local.global_bad_rho, &out.global_bad_rho, 1,
          mfem::MPITypeMap<int>::mpi_type, MPI_SUM, vfes.GetComm());
        MPI_Allreduce(&local.global_bad_pressure, &out.global_bad_pressure, 1,
          mfem::MPITypeMap<int>::mpi_type, MPI_SUM, vfes.GetComm());
        out.first_bad_rank = FindFirstBadRank(out.first_local_issue.found,
                                             vfes.GetComm());
        return out;
      }
    }
  }
  return ScanEulerAdmissibilityImpl(
    vfes, v, dim, gamma, rho_min, p_min, quad_order, face_quad_orders, point_set,
    prepared ? &elements_ : nullptr,
    prepared && !dof_scan ? &shape_tables_ : nullptr,
    prepared && !dof_scan ? &plans_[static_cast<int>(point_set)] : nullptr);
}

} // namespace detail

// Keep the original scanner as the generic path and numerical reference.
inline EulerAdmissibilitySummary
ScanEulerAdmissibility(const mfem::ParFiniteElementSpace& vfes,
                       const mfem::Vector& v,
                       int dim,
                       double gamma,
                       double rho_min,
                       double p_min,
                       int quad_order,
                       const std::vector<int>& face_quad_orders,
                       EulerAdmissibilityPointSet point_set)
{
  return detail::ScanEulerAdmissibilityImpl(
    vfes, v, dim, gamma, rho_min, p_min, quad_order, face_quad_orders, point_set,
    nullptr, nullptr, nullptr);
}

} // namespace igrlab
} // namespace mfem
