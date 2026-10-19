#include "igrlab/environment.hpp"
#include "physics/igr/IGR.hpp"
#include "utils/config/env_flags.hpp"
#if !((defined(MFEM_USE_CUDA) && !defined(__CUDACC__)) || \
      (defined(MFEM_USE_HIP) && !defined(__HIPCC__)))
#include "mfem/general/forall.hpp"
#ifndef DG_IGR_HAS_ACCEL_FORALL
#define DG_IGR_HAS_ACCEL_FORALL 1
#endif
#endif
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"
#include "IGR_device_kernels.hpp"
#include "IGR_internal_utils.hpp"
#include "IGR_nodal_source.hpp"
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <string>
#include <sstream>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace igrlab
{
using namespace igr_internal;

namespace igr_internal
{
NodalSourceMode GetNodalSourceMode()
{
  const char* raw = ::igrlab::GetEnv("DG_IGR_IGR_NODAL_SOURCE");
  const std::string mode = raw ? raw : "baseline";
  if (mode == "baseline" || mode == "default") return NodalSourceMode::Baseline;
  if (mode == "volume") return NodalSourceMode::Volume;
  if (mode == "central") return NodalSourceMode::Central;
  MFEM_ABORT("DG_IGR_IGR_NODAL_SOURCE must be baseline, default, volume, or central.");
  return NodalSourceMode::Baseline;
}
const char* NodalSourceModeName(NodalSourceMode mode)
{
  return mode == NodalSourceMode::Central ? "central" :
         mode == NodalSourceMode::Volume ? "volume" : "baseline";
}
int NodalSourceQuadratureMinimumOrder()
{
  const char* raw = ::igrlab::GetEnv("DG_IGR_IGR_SOURCE_QUADRATURE_MIN_ORDER");
  if (!raw) return 0;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(raw, &end, 10);
  MFEM_VERIFY(raw[0] && end && !*end && errno == 0 && value >= 0 && value <= 128,
              "DG_IGR_IGR_SOURCE_QUADRATURE_MIN_ORDER must be an integer in [0,128].");
  return static_cast<int>(value);
}
int OperatorQuadratureMinimumOrder()
{
  const char* raw = ::igrlab::GetEnv("DG_IGR_IGR_OPERATOR_QUADRATURE_MIN_ORDER");
  if (!raw) return 0;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(raw, &end, 10);
  MFEM_VERIFY(raw[0] && end && !*end && errno == 0 && value >= 0 && value <= 128,
              "DG_IGR_IGR_OPERATOR_QUADRATURE_MIN_ORDER must be an integer in [0,128].");
  return static_cast<int>(value);
}
bool SupportsSourceQuadratureTensor(int dimension, int d1d, int q1d,
                                    std::string* reason)
{
  if (reason) reason->clear();
  if (dimension < 1 || dimension > 3 || d1d <= 0 || q1d <= 0)
  {
    if (reason) *reason = "invalid tensor dimension or point count";
    return false;
  }
  // The MFEM 1D interpolation kernels have no fixed dof/quadrature arrays.
  if (dimension == 1) return true;
#ifdef DG_IGR_HAS_ACCEL_FORALL
  const auto& limits = mfem::DeviceDofQuadLimits::Get();
  const int max_d = dimension == 2 ? limits.MAX_D1D : limits.MAX_INTERP_1D;
  const int max_q = dimension == 2 ? limits.MAX_Q1D : limits.MAX_INTERP_1D;
  if (d1d <= max_d && q1d <= max_q) return true;
  if (reason)
  {
    std::ostringstream message;
    message << "MFEM generic tensor QI limit: dimension=" << dimension
            << " D1D=" << d1d << " Q1D=" << q1d
            << " maximum_D1D=" << max_d << " maximum_Q1D=" << max_q;
    *reason = message.str();
  }
#else
  if (reason) *reason = "source quadrature override requires accelerator-aware coefficient compilation";
#endif
  return false;
}
namespace
{
void ValidateCoefficientQuadrature(const mfem::FiniteElementSpace& vfes,
                              const mfem::FiniteElementSpace& sfes,
                              const mfem::IntegrationRule& ir,
                              const char* control, int requested)
{
  MFEM_VERIFY(mfem::UsesTensorBasis(vfes) && mfem::UsesTensorBasis(sfes),
              control << " currently requires tensor state/scalar spaces.");
  auto check = [&](const mfem::FiniteElementSpace& fes, const char* role)
  {
    for (int e = 0; e < fes.GetNE(); ++e)
    {
      const auto& fe = *fes.GetFE(e);
      const auto& map = fe.GetDofToQuad(ir, mfem::DofToQuad::TENSOR);
      std::string reason;
      MFEM_VERIFY(SupportsSourceQuadratureTensor(fe.GetDim(),map.ndof,map.nqpt,&reason),
        "Unsupported " << control << "=" << requested
        << " for " << role << " element=" << e << ": " << reason
        << ". Reduce the requested coefficient rule before launching quadrature interpolation.");
    }
  };
  check(vfes,"state"); check(sfes,"scalar");
  const auto* nodes = vfes.GetMesh()->GetNodes();
  if (nodes)
  {
    MFEM_VERIFY(mfem::UsesTensorBasis(*nodes->FESpace()),
                "Coefficient quadrature override currently requires tensor geometry.");
    check(*nodes->FESpace(),"geometry");
  }
}
void PrepareNodalSource(const mfem::FiniteElementSpace& fes,
                        const mfem::QuadratureSpace& qs, bool central,
                        NodalVelocitySourceCache& c,
                        const mfem::Array<int>* slip_marker)
{
  auto* mesh = fes.GetMesh();
  MFEM_VERIFY(sizeof(mfem::real_t) == sizeof(double), "Nodal IGR source requires FP64.");
  MFEM_VERIFY(mesh && mesh->Dimension() == 2 && mesh->SpaceDimension() == 2 &&
                fes.GetVDim() == 4 && fes.GetNE() > 0 && !mesh->Nonconforming(),
              "Nodal IGR source supports conforming two-dimensional full-energy quadrilaterals.");
  if (const auto* pfes = dynamic_cast<const mfem::ParFiniteElementSpace*>(&fes))
  {
    int ranks = 0; MPI_Comm_size(pfes->GetComm(), &ranks);
    MFEM_VERIFY(ranks == 1, "Nodal IGR source currently supports one MPI rank.");
  }
  MFEM_VERIFY(dynamic_cast<const mfem::L2_FECollection*>(fes.FEColl()),
              "Nodal IGR source requires a nodal conservative L2 space.");
  const auto& fe = *fes.GetFE(0);
  const int p = fe.GetOrder(), ndof = fe.GetDof(), ne = fes.GetNE();
  MFEM_VERIFY(p >= 1 && p <= 7, "Nodal IGR source currently supports p=1 through 7.");
  const auto& ir = qs.GetIntRule(0);
  const int nq = ir.GetNPoints();
  const bool mirror_walls = slip_marker != nullptr;
  MFEM_VERIFY(!mirror_walls || central,
              "Mirror wall source requires central velocity recovery.");
  std::vector<int> wall_marker;
  if (slip_marker)
    for (int i = 0; i < slip_marker->Size(); ++i)
      wall_marker.push_back((*slip_marker)[i] ? 1 : 0);
  std::vector<double> rule;
  for (int q = 0; q < nq; ++q)
  {
    const auto& ip = ir.IntPoint(q);
    rule.insert(rule.end(), {ip.x, ip.y, ip.weight});
  }
  if (c.mesh == mesh && c.fes == &fes && c.fec == fes.FEColl() &&
      c.mesh_sequence == mesh->GetSequence() && c.nodes_sequence == mesh->GetNodesSequence() &&
      c.fes_sequence == fes.GetSequence() && c.order == p && c.ne == ne &&
      c.rule_order == ir.GetOrder() && c.rule == rule && c.central == central &&
      c.mirror_walls == mirror_walls && c.wall_marker == wall_marker)
  { ++c.hits; return; }
  const auto start = std::chrono::steady_clock::now();
  int geometry_order = 1;
  if (mesh->GetNodes()) geometry_order = mesh->GetNodes()->FESpace()->GetMaxElementOrder();
  MFEM_VERIFY(geometry_order <= p, "Nodal IGR source requires geometry degree <= state degree.");
  for (int e = 0; e < ne; ++e)
  {
    const auto& local = *fes.GetFE(e);
    const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&local);
    MFEM_VERIFY(tensor && local.GetGeomType() == mfem::Geometry::SQUARE &&
                  local.GetDof() == ndof && local.GetOrder() == p &&
                  local.GetMapType() == mfem::FiniteElement::VALUE &&
                  (tensor->GetBasisType() == mfem::BasisType::GaussLegendre ||
                   tensor->GetBasisType() == mfem::BasisType::GaussLobatto),
                "Nodal IGR source requires uniform-order GL/LGL tensor L2 quadrilaterals.");
    MFEM_VERIFY(qs.GetIntRule(e).GetNPoints() == nq &&
                  &qs.GetIntRule(e) == &ir,
                "Nodal IGR source requires one shared RHS quadrature rule.");
  }
  c.ndof = ndof; c.nq = nq; c.ne = ne; c.slots = central ? 5 : 1;
  const std::size_t entries = std::size_t(ndof) * c.slots * 2 * nq * ne;
  MFEM_VERIFY(entries <= std::size_t(std::numeric_limits<int>::max()),
              "Nodal IGR source geometry cache exceeds MFEM vector indexing.");
  c.derivative.SetSize(static_cast<int>(entries)); c.derivative.UseDevice(true);
  c.neighbors.SetSize(c.slots * ne); c.neighbors.GetMemory().UseDevice(true);
  double* maps = c.derivative.HostWrite(); std::fill(maps, maps + entries, 0.0);
  int* neighbors = c.neighbors.HostWrite(); std::fill(neighbors, neighbors + c.slots * ne, -1);
  c.wall_element_count = 0;
  c.wall_elements.SetSize(mirror_walls ? ne : 0);
  c.wall_elements.GetMemory().UseDevice(true);
  int* wall_elements = mirror_walls ? c.wall_elements.HostWrite() : nullptr;
  if (mirror_walls)
  {
    std::fill(wall_elements, wall_elements + ne, -1);
    for (int b = 0; b < mesh->GetNBE(); ++b)
    {
      const int attr = mesh->GetBdrAttribute(b);
      if (attr <= 0 || attr > static_cast<int>(wall_marker.size()) || !wall_marker[attr-1]) continue;
      auto* tr = mesh->GetBdrFaceTransformations(b);
      MFEM_VERIFY(tr && tr->Elem1No >= 0 && tr->Elem1No < ne,
                  "Mirror source wall must belong to a local element.");
      if (wall_elements[tr->Elem1No] < 0)
        wall_elements[tr->Elem1No] = c.wall_element_count++;
    }
  }
  // The rank-three normal tensor has only four distinct components in 2D.
  // Retain compact maps for wall elements, not padded copies for every cell.
  const std::size_t wall_entries = 4*std::size_t(ndof)*nq*c.wall_element_count;
  MFEM_VERIFY(wall_entries <= std::size_t(std::numeric_limits<int>::max()),
              "Mirror source geometry exceeds MFEM vector indexing.");
  c.wall_derivative.SetSize(static_cast<int>(wall_entries));
  c.wall_derivative.UseDevice(true);
  double* wall_maps = wall_entries ? c.wall_derivative.HostWrite() : nullptr;
  if (wall_maps) std::fill(wall_maps, wall_maps + wall_entries, 0.0);
  auto index = [&](int e, int q, int d, int slot, int j)
  { return j + ndof * (slot + c.slots * (d + 2 * (q + nq * e))); };
  // q-evaluation of each exact local mass inverse is only needed for the lift.
  std::vector<mfem::DenseMatrix> eval_minv;
  if (central) eval_minv.resize(ne);
  const auto& mass_ir = mfem::IntRules.Get(mfem::Geometry::SQUARE, 2*p + 2*geometry_order + 2);
  for (int e = 0; e < ne; ++e)
  {
    neighbors[c.slots * e] = e;
    const auto& local = *fes.GetFE(e);
    const auto& map = dynamic_cast<const mfem::TensorBasisElement&>(local).GetDofMap();
    auto native = [&](int j) { return map.Size() ? map[j] : j; };
    auto* tr = mesh->GetElementTransformation(e);
    mfem::DenseMatrix ds(ndof, 2), mass(ndof), inverse(ndof);
    mfem::Vector shape(ndof);
    if (central)
    {
      mass = 0.0;
      for (int q = 0; q < mass_ir.GetNPoints(); ++q)
      {
        const auto& ip = mass_ir.IntPoint(q); tr->SetIntPoint(&ip);
        local.CalcShape(ip, shape); const double weight = ip.weight * tr->Weight();
        MFEM_VERIFY(weight > 0, "Nodal IGR source requires positive geometry weights.");
        for (int j = 0; j < ndof; ++j) for (int i = 0; i < ndof; ++i)
          mass(i,j) += weight * shape(i) * shape(j);
      }
      mfem::DenseMatrixInverse factor(mass, true); factor.GetInverseMatrix(inverse);
      eval_minv[e].SetSize(nq, ndof);
    }
    for (int q = 0; q < nq; ++q)
    {
      const auto& ip = ir.IntPoint(q); tr->SetIntPoint(&ip);
      local.CalcPhysDShape(*tr, ds);
      for (int d = 0; d < 2; ++d) for (int j = 0; j < ndof; ++j)
        maps[index(e,q,d,0,j)] = ds(native(j),d);
      if (central)
      {
        local.CalcShape(ip, shape);
        for (int j = 0; j < ndof; ++j)
        {
          double value = 0.0;
          for (int i = 0; i < ndof; ++i) value += shape(i) * inverse(i,j);
          eval_minv[e](q,j) = value;
        }
      }
    }
  }
  if (central)
  {
    const auto& face_ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2*p + 2*geometry_order + 2);
    mfem::Vector s1(ndof), s2(ndof), normal(2);
    for (int f = 0; f < mesh->GetNumFaces(); ++f)
    {
      auto* tr = mesh->GetInteriorFaceTransformations(f);
      if (!tr) continue; // physical boundaries are handled separately below
      const int elems[2] = {tr->Elem1No, tr->Elem2No};
      MFEM_VERIFY(elems[0] >= 0 && elems[1] >= 0 && elems[0] < ne && elems[1] < ne,
                  "Nodal IGR source requires local conforming face neighbors.");
      int slots[2];
      for (int side = 0; side < 2; ++side)
      {
        int slot = 1;
        while (slot < c.slots && neighbors[slot+c.slots*elems[side]] != -1 &&
               neighbors[slot+c.slots*elems[side]] != elems[1-side]) ++slot;
        MFEM_VERIFY(slot < c.slots, "Nodal IGR source quadrilateral has too many neighbors.");
        neighbors[slot+c.slots*elems[side]] = elems[1-side]; slots[side] = slot;
      }
      for (int fq = 0; fq < face_ir.GetNPoints(); ++fq)
      {
        const auto& ip = face_ir.IntPoint(fq); tr->SetAllIntPoints(&ip);
        fes.GetFE(elems[0])->CalcShape(tr->GetElement1IntPoint(), s1);
        fes.GetFE(elems[1])->CalcShape(tr->GetElement2IntPoint(), s2);
        mfem::CalcOrtho(tr->Jacobian(), normal);
        for (int side = 0; side < 2; ++side)
        {
          const int e = elems[side];
          const mfem::Vector& own = side ? s2 : s1;
          const mfem::Vector& other = side ? s1 : s2;
          const auto& own_map = dynamic_cast<const mfem::TensorBasisElement&>(*fes.GetFE(e)).GetDofMap();
          const auto& other_map = dynamic_cast<const mfem::TensorBasisElement&>(*fes.GetFE(elems[1-side])).GetDofMap();
          for (int q = 0; q < nq; ++q)
          {
            double lift = 0.0;
            for (int i = 0; i < ndof; ++i) lift += eval_minv[e](q,i) * own(i);
            for (int d = 0; d < 2; ++d)
            {
              const double weight = .5 * ip.weight * (side ? -normal(d) : normal(d)) * lift;
              for (int j = 0; j < ndof; ++j)
              {
                maps[index(e,q,d,0,j)] -= weight * own(own_map.Size() ? own_map[j] : j);
                maps[index(e,q,d,slots[side],j)] += weight * other(other_map.Size() ? other_map[j] : j);
              }
            }
          }
        }
      }
    }
  }
  if (mirror_walls && c.wall_element_count)
  {
    // Unit normals on curved polynomial faces make this integrand rational.
    // Use a more resolved setup-only face rule; no exact-polynomial claim is
    // made for curved normals. The oracle independently refines this rule.
    const auto& wall_ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT,
                                            4*p + 2*geometry_order + 8);
    mfem::Vector shape(ndof), normal(2);
    for (int b = 0; b < mesh->GetNBE(); ++b)
    {
      const int attr = mesh->GetBdrAttribute(b);
      if (attr <= 0 || attr > static_cast<int>(wall_marker.size()) || !wall_marker[attr-1]) continue;
      auto* tr = mesh->GetBdrFaceTransformations(b);
      const int e = tr->Elem1No, wall = wall_elements[e];
      const auto& fe = *fes.GetFE(e);
      const auto& dof_map = dynamic_cast<const mfem::TensorBasisElement&>(fe).GetDofMap();
      for (int fq = 0; fq < wall_ir.GetNPoints(); ++fq)
      {
        const auto& ip = wall_ir.IntPoint(fq); tr->SetAllIntPoints(&ip);
        fe.CalcShape(tr->GetElement1IntPoint(), shape);
        mfem::CalcOrtho(tr->Jacobian(), normal);
        const double jac = normal.Norml2();
        MFEM_VERIFY(jac > 0 && std::isfinite(jac), "Degenerate mirror wall face.");
        const double nx = normal[0]/jac, ny = normal[1]/jac;
        const double moments[4] = {nx*nx*nx, nx*nx*ny, nx*ny*ny, ny*ny*ny};
        for (int q = 0; q < nq; ++q)
        {
          double lift = 0.;
          for (int i = 0; i < ndof; ++i) lift += eval_minv[e](q,i)*shape[i];
          for (int j = 0; j < ndof; ++j)
          {
            const double weight = -ip.weight*jac*lift*shape[dof_map.Size() ? dof_map[j] : j];
            for (int m = 0; m < 4; ++m)
              wall_maps[j+ndof*(m+4*(q+nq*wall))] += weight*moments[m];
          }
        }
      }
    }
  }
  c.mesh = mesh; c.fes = &fes; c.fec = fes.FEColl();
  c.mesh_sequence = mesh->GetSequence(); c.nodes_sequence = mesh->GetNodesSequence();
  c.fes_sequence = fes.GetSequence(); c.order = p; c.rule_order = ir.GetOrder();
  c.rule = std::move(rule); c.central = central;
  c.mirror_walls = mirror_walls; c.wall_marker = std::move(wall_marker); ++c.rebuilds;
  std::size_t active_slots = 0;
  for (int i = 0; i < c.neighbors.Size(); ++i) if (neighbors[i] >= 0) ++active_slots;
  if (mfem::Mpi::Root())
    std::cout << "[igr-linear] event=nodal-source-setup mode=" << (central ? "central" : "volume")
              << " p=" << p << " elements=" << ne << " nodes=" << ndof << " rhs_points=" << nq
              << " geometry_order=" << geometry_order << " matrices=" << ne*2*c.slots
              << " matrix_rows=" << nq << " matrix_columns=" << ndof
              << " useful_products_per_apply=" << 4*std::size_t(ndof)*nq*active_slots
              << " allocated_products_per_apply=" << 2*entries
              << " geometry_bytes=" << (entries+wall_entries)*sizeof(double)+(c.neighbors.Size()+c.wall_elements.Size())*sizeof(int)
              << " wall_elements=" << c.wall_element_count
              << " wall_geometry_bytes=" << wall_entries*sizeof(double)+c.wall_elements.Size()*sizeof(int)
              << " wall_products_per_apply=" << 6*std::size_t(ndof)*nq*c.wall_element_count
              << " setup_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << " lift_mass=integrated physical_boundary="
              << (mirror_walls ? "central-slip-mirror" : "own-trace") << '\n';
}
struct NodalVelocityKernel
{
  int ndof; double floor; const double* state; double* velocity;
  MFEM_HOST_DEVICE void operator()(int k) const
  {
    const int j = k % ndof, e = k / ndof;
    const double inverse = frozen_scalar_igr_algebra::SafeInverseDensity(state[j+ndof*(4*e)],floor);
    for (int v = 0; v < 2; ++v)
      velocity[j+ndof*(v+2*e)] = state[j+ndof*(1+v+4*e)] * inverse;
  }
};
struct NodalSourceKernel
{
  int ndof, nq, slots; double alpha;
  const double *maps, *velocity, *element_alpha; const int* neighbors; double* source;
  const double* wall_maps; const int* wall_elements;
  MFEM_HOST_DEVICE void operator()(int k) const
  {
    const int e = k/nq;
    double grad[4] = {0,0,0,0};
    for (int slot = 0; slot < slots; ++slot)
    {
      const int other = neighbors[slot+slots*e]; if (other < 0) continue;
      for (int d = 0; d < 2; ++d) for (int j = 0; j < ndof; ++j)
      {
        const double a = maps[j+ndof*(slot+slots*(d+2*k))];
        for (int v = 0; v < 2; ++v) grad[v+2*d] += a*velocity[j+ndof*(v+2*other)];
      }
    }
    const int wall = wall_elements ? wall_elements[e] : -1;
    if (wall >= 0)
    {
      const int q = k-nq*e;
      for (int j = 0; j < ndof; ++j)
      {
        const int offset = j+ndof*(4*(q+nq*wall));
        const double ux = velocity[j+ndof*(2*e)], uy = velocity[j+ndof*(1+2*e)];
        grad[0] += wall_maps[offset]*ux + wall_maps[offset+ndof]*uy;
        const double cross = wall_maps[offset+ndof]*ux + wall_maps[offset+2*ndof]*uy;
        grad[1] += cross; grad[2] += cross;
        grad[3] += wall_maps[offset+2*ndof]*ux + wall_maps[offset+3*ndof]*uy;
      }
    }
    source[k] = frozen_scalar_igr_algebra::SourceFromVelocityGradient(2,grad,1,2,
                    element_alpha ? element_alpha[e] : alpha);
  }
};
}
void ApplyNodalVelocitySource2D(const mfem::FiniteElementSpace& fes,
  const mfem::QuadratureSpace& qs, const mfem::Vector& evec, mfem::real_t alpha,
  const mfem::Vector* element_alpha, mfem::real_t rho_floor, bool central,
  NodalVelocitySourceCache& c, mfem::Vector& source,
  const mfem::Array<int>* slip_marker)
{
  PrepareNodalSource(fes,qs,central,c,slip_marker);
  MFEM_VERIFY(evec.Size() == 4*c.ndof*c.ne && (!element_alpha || element_alpha->Size() == c.ne),
              "Nodal IGR source input layout mismatch.");
  c.velocity.SetSize(2*c.ndof*c.ne); c.velocity.UseDevice(true);
  source.SetSize(c.nq*c.ne); source.UseDevice(true);
  NodalVelocityKernel velocity{c.ndof,rho_floor,evec.Read(),c.velocity.Write()};
  NodalSourceKernel apply{c.ndof,c.nq,c.slots,alpha,c.derivative.Read(),c.velocity.Read(),
                         element_alpha ? element_alpha->Read() : nullptr,c.neighbors.Read(),source.Write(),
                         c.wall_derivative.Size() ? c.wall_derivative.Read() : nullptr,
                         c.wall_elements.Size() ? c.wall_elements.Read() : nullptr};
#ifdef DG_IGR_HAS_ACCEL_FORALL
  mfem::forall(c.ndof*c.ne,velocity);
  mfem::forall(c.nq*c.ne,apply);
#else
  MFEM_VERIFY(!mfem::Device::Allows(mfem::Backend::DEVICE_MASK),
              "Nodal IGR source requires accelerator compilation on a device backend.");
  for (int i=0;i<c.ndof*c.ne;++i) velocity(i);
  for (int i=0;i<c.nq*c.ne;++i) apply(i);
#endif
}
}


namespace
{

std::unique_ptr<mfem::QuadratureSpace>
MakeVolumeQuadratureSpace(mfem::Mesh& mesh,
                          const mfem::IntegrationRule& ir)
{
  if (mesh.GetNumGeometries(mesh.Dimension()) == 1)
  {
    return std::make_unique<mfem::QuadratureSpace>(mesh, ir);
  }
  return std::make_unique<mfem::QuadratureSpace>(&mesh, ir.GetOrder());
}

// Values are always produced in byVDIM layout with the same state vdim.
// A volume QuadratureSpace stores entity e as mesh element e without a point
// permutation.  Prove the remaining layout identity once per cache rebuild.
bool SameStateQuadratureLayout(const mfem::QuadratureSpace& a,
                               const mfem::QuadratureSpace& b)
{
  if (a.GetMesh() != b.GetMesh() || a.GetNE() != b.GetNE() ||
      a.GetNE() != a.GetMesh()->GetNE() || a.GetSize() != b.GetSize())
  { return false; }
  bool checked_geometry[mfem::Geometry::NumGeom] = {};
  const int dim = a.GetMesh()->Dimension();
  for (int e = 0; e < a.GetNE(); ++e)
  {
    if (a.Offset(e) != b.Offset(e) || a.GetGeometry(e) != b.GetGeometry(e))
    { return false; }
    const int geom = static_cast<int>(a.GetGeometry(e));
    if (checked_geometry[geom]) { continue; }
    checked_geometry[geom] = true;
    const auto& ar = a.GetIntRule(e);
    const auto& br = b.GetIntRule(e);
    if (ar.GetNPoints() != br.GetNPoints()) { return false; }
    for (int q = 0; q < ar.GetNPoints(); ++q)
    {
      const auto& ap = ar.IntPoint(q);
      const auto& bp = br.IntPoint(q);
      if (std::memcmp(&ap.x, &bp.x, sizeof(ap.x)) ||
          (dim > 1 && std::memcmp(&ap.y, &bp.y, sizeof(ap.y))) ||
          (dim > 2 && std::memcmp(&ap.z, &bp.z, sizeof(ap.z))) ||
          std::memcmp(&ap.weight, &bp.weight, sizeof(ap.weight)))
      { return false; }
    }
  }
  return a.Offset(a.GetNE()) == b.Offset(b.GetNE()) &&
         a.Offset(a.GetNE()) == a.GetSize();
}

void LogStateQuadratureAudit(const char* name,
                             const mfem::QuadratureFunction& values,
                             const mfem::QuadratureSpace& space,
                             int num_eq,
                             int dim,
                             double rho_floor,
                             int solve_call,
                             MPI_Comm comm)
{
  const int ne = space.GetNE();
  const int nq = space.GetIntRule(0).GetNPoints();
  const auto state = mfem::Reshape(values.HostRead(), num_eq, nq, ne);
  double local_min_rho = std::numeric_limits<double>::infinity();
  double local_max_rho = -std::numeric_limits<double>::infinity();
  double local_min_internal = std::numeric_limits<double>::infinity();
  long long local_floor = 0;
  long long local_negative_internal = 0;
  long long local_nonfinite = 0;
  for (int e = 0; e < ne; ++e)
  {
    for (int q = 0; q < nq; ++q)
    {
      const double rho = state(0, q, e);
      double momentum_sq = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        momentum_sq += state(1 + d, q, e) * state(1 + d, q, e);
      }
      const double internal = rho > 0.0
        ? state(1 + dim, q, e) - 0.5 * momentum_sq / rho
        : -std::numeric_limits<double>::infinity();
      local_min_rho = std::min(local_min_rho, rho);
      local_max_rho = std::max(local_max_rho, rho);
      local_min_internal = std::min(local_min_internal, internal);
      local_floor += rho < rho_floor ? 1 : 0;
      local_negative_internal += internal < 0.0 ? 1 : 0;
      local_nonfinite += (!std::isfinite(rho) || !std::isfinite(internal)) ? 1 : 0;
    }
  }

  double global_min_rho = local_min_rho;
  double global_max_rho = local_max_rho;
  double global_min_internal = local_min_internal;
  long long global_floor = local_floor;
  long long global_negative_internal = local_negative_internal;
  long long global_nonfinite = local_nonfinite;
  long long global_evaluations = static_cast<long long>(ne) * nq;
  MPI_Allreduce(MPI_IN_PLACE, &global_min_rho, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &global_max_rho, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_min_internal,
                1,
                MPI_DOUBLE,
                MPI_MIN,
                comm);
  MPI_Allreduce(MPI_IN_PLACE, &global_floor, 1, MPI_LONG_LONG, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_negative_internal,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_nonfinite,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_evaluations,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  if (mfem::Mpi::Root())
  {
    std::cout << "[igr-ingestion-audit] kind=state"
              << " solve=" << solve_call
              << " space=" << name
              << " evaluations=" << global_evaluations
              << " min_rho=" << global_min_rho
              << " max_rho=" << global_max_rho
              << " min_internal_energy=" << global_min_internal
              << " density_floor_activations=" << global_floor
              << " negative_internal_energy=" << global_negative_internal
              << " nonfinite=" << global_nonfinite << std::endl;
  }
}

void LogScalarQuadratureAudit(const char* name,
                              const mfem::Vector& values,
                              int solve_call,
                              MPI_Comm comm)
{
  const double* data = values.HostRead();
  double local_min = std::numeric_limits<double>::infinity();
  double local_max = -std::numeric_limits<double>::infinity();
  long long local_nonfinite = 0;
  for (int i = 0; i < values.Size(); ++i)
  {
    local_min = std::min(local_min, data[i]);
    local_max = std::max(local_max, data[i]);
    local_nonfinite += std::isfinite(data[i]) ? 0 : 1;
  }
  double global_min = local_min;
  double global_max = local_max;
  long long global_nonfinite = local_nonfinite;
  long long global_values = values.Size();
  MPI_Allreduce(MPI_IN_PLACE, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_nonfinite,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_values,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  if (mfem::Mpi::Root())
  {
    std::cout << "[igr-ingestion-audit] kind=derived"
              << " solve=" << solve_call
              << " field=" << name
              << " values=" << global_values
              << " min=" << global_min
              << " max=" << global_max
              << " nonfinite=" << global_nonfinite << std::endl;
  }
}

} // namespace

void IGRSolver::EnsureH1VelocitySpace_()
{
  if (backend_ != GradBackend::H1Projection)
  {
    return;
  }

  mfem::ParMesh* pm = vfes_.GetParMesh();
  MFEM_VERIFY(pm, "Expected mfem::ParMesh for state space.");

  // Choose H1 order sensibly: ≥ 1, and not below the state space order.
  const int p_state = SpaceOrder(vfes_);
  const int p_h1 = std::max(1, p_state);

  // Recreate only if absent or order/mesh changed
  bool need_make = (!h1_vec_fec_ || !h1_vec_fes_ || !vel_h1_);
  if (!need_make)
  {
    if (h1_vec_fec_->GetOrder() != p_h1) need_make = true;
    if (h1_vec_fes_->GetParMesh() != pm) need_make = true;
  }
  if (need_make)
  {
    h1_vec_fec_ =
      std::make_unique<mfem::H1_FECollection>(p_h1, pm->Dimension());
    h1_vec_fes_ =
      std::make_unique<mfem::ParFiniteElementSpace>(pm,
                                                    h1_vec_fec_.get(),
                                                    /*vdim=*/dim_);
    vel_h1_ = std::make_unique<mfem::ParGridFunction>(h1_vec_fes_.get());
    *vel_h1_ = 0.0;
  }
}

void IGRSolver::EnsureDeviceQuadrature_()
{
  if (!UseDeviceCoefficients_())
  {
    return;
  }

  const bool debug_qf =
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  mfem::Mesh* mesh = sfes_.GetMesh();
  MFEM_VERIFY(mesh, "IGR device coefficients require a valid mesh.");
  MFEM_VERIFY(mesh == vfes_.GetMesh(),
              "IGR device coefficients expect sigma and state meshes to match.");

  const long mesh_seq = mesh->GetSequence();
  const long sfes_seq = sfes_.GetSequence();
  const long vfes_seq = vfes_.GetSequence();

  const mfem::FiniteElement& sigma_fe = *sfes_.GetFE(0);
  mfem::ElementTransformation* typical_T =
    mesh->GetTypicalElementTransformation();
  MFEM_VERIFY(typical_T,
              "IGR device coefficients require a typical element transform.");
  const mfem::IntegrationRule& native_mass_ir =
    mfem::MassIntegrator::GetRule(sigma_fe, sigma_fe, *typical_T);
  const mfem::IntegrationRule& native_diff_ir =
    mfem::DiffusionIntegrator::GetRule(sigma_fe, sigma_fe);
  const int operator_minimum = std::max(subcell_quadrature_minimum_order_,
                                         OperatorQuadratureMinimumOrder());
  const mfem::IntegrationRule& mass_ir =
    operator_minimum > native_mass_ir.GetOrder()
      ? mfem::IntRules.Get(sigma_fe.GetGeomType(),operator_minimum) : native_mass_ir;
  const mfem::IntegrationRule& diff_ir =
    operator_minimum > native_diff_ir.GetOrder()
      ? mfem::IntRules.Get(sigma_fe.GetGeomType(),operator_minimum) : native_diff_ir;
  const mfem::IntegrationRule& rhs_ir =
    mfem::IntRules.Get(sigma_fe.GetGeomType(), std::max({2 * sigma_fe.GetOrder(), subcell_quadrature_minimum_order_, NodalSourceQuadratureMinimumOrder()}));
  const int mass_qorder = mass_ir.GetOrder();
  const int diff_qorder = diff_ir.GetOrder();
  const int rhs_qorder = rhs_ir.GetOrder();
  const mfem::Geometry::Type face_geom = mesh->GetTypicalFaceGeometry();
  const mfem::IntegrationRule& face_ir =
    FaceIntegrationRules_().Get(face_geom, diff_qorder);
  const int face_qorder = face_ir.GetOrder();

  const bool need_rebuild =
    (!mass_qs_ || !sigma_qs_ || !rhs_qs_ || !u_mass_qf_ || !u_qf_ ||
     !u_rhs_qf_ || !grad_u_rhs_qf_ || !invrho_qf_ ||
     !ck_qf_ || !src_qf_ ||
     mesh_seq != device_mesh_sequence_ || sfes_seq != device_sfes_sequence_ ||
     vfes_seq != device_vfes_sequence_ ||
     mass_qorder != device_mass_qorder_ ||
     diff_qorder != device_diff_qorder_ ||
     rhs_qorder != device_rhs_qorder_ ||
     face_qorder != device_face_qorder_);

  if (!need_rebuild)
  {
    return;
  }
  if (NodalSourceQuadratureMinimumOrder() > 0)
  {
    // Before QuadratureSpace construction can request geometry QI kernels,
    // check actual tensor DofToQuad sizes against the active backend limits.
    ValidateCoefficientQuadrature(vfes_,sfes_,rhs_ir,
      "DG_IGR_IGR_SOURCE_QUADRATURE_MIN_ORDER",NodalSourceQuadratureMinimumOrder());
  }
  if (OperatorQuadratureMinimumOrder() > 0)
  {
    ValidateCoefficientQuadrature(vfes_,sfes_,mass_ir,
      "DG_IGR_IGR_OPERATOR_QUADRATURE_MIN_ORDER",OperatorQuadratureMinimumOrder());
    ValidateCoefficientQuadrature(vfes_,sfes_,diff_ir,
      "DG_IGR_IGR_OPERATOR_QUADRATURE_MIN_ORDER",OperatorQuadratureMinimumOrder());
  }
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-rebuild"
              << " mesh_seq=" << mesh_seq
              << " sfes_seq=" << sfes_seq
              << " vfes_seq=" << vfes_seq
              << " mass_qorder=" << mass_qorder
              << " diff_qorder=" << diff_qorder
              << " rhs_qorder=" << rhs_qorder
              << " source_qorder_min_requested=" << NodalSourceQuadratureMinimumOrder()
              << " operator_qorder_min_requested=" << OperatorQuadratureMinimumOrder()
              << " face_qorder=" << face_qorder
              << " face_geom=" << static_cast<int>(face_geom)
              << " is_dg=" << (IsDGSpace(sfes_) ? 1 : 0)
              << std::endl;
  }

  // The diagnostic-only gradient belongs to the old diffusion space.  Keep
  // lifetime-once scan state unchanged when rebuilding this local cache.
  grad_u_qf_.reset();

  mass_qs_ = MakeVolumeQuadratureSpace(*mesh, mass_ir);
  sigma_qs_ = MakeVolumeQuadratureSpace(*mesh, diff_ir);
  rhs_qs_ = MakeVolumeQuadratureSpace(*mesh, rhs_ir);

  if (IsDGSpace(sfes_))
  {
    sigma_fqs_ =
      std::make_unique<mfem::FaceQuadratureSpace>(*mesh,
                                                  face_ir,
                                                  mfem::FaceType::Interior);
  }
  else
  {
    sigma_fqs_.reset();
  }

  const bool share_state = ::igrlab::EnvFlagEnabledDefault(
    "DG_IGR_IGR_SHARE_STATE_INTERPOLATION", true);
  device_diff_values_share_mass_ = share_state &&
    SameStateQuadratureLayout(*mass_qs_, *sigma_qs_);
  device_rhs_values_share_mass_ = share_state &&
    SameStateQuadratureLayout(*mass_qs_, *rhs_qs_);
  device_rhs_values_share_diff_ = share_state &&
    !device_rhs_values_share_mass_ && !device_diff_values_share_mass_ &&
    SameStateQuadratureLayout(*sigma_qs_, *rhs_qs_);
  u_mass_qf_ =
    std::make_unique<mfem::QuadratureFunction>(mass_qs_.get(), num_eq_);
  u_qf_ =
    std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(), num_eq_);
  u_rhs_qf_ =
    std::make_unique<mfem::QuadratureFunction>(rhs_qs_.get(), num_eq_);
  grad_u_rhs_qf_ =
    std::make_unique<mfem::QuadratureFunction>(rhs_qs_.get(),
                                               num_eq_ * dim_);
  grad_u_rhs_qf_->SetVDim(num_eq_ * dim_);

  invrho_qf_ = std::make_unique<mfem::QuadratureFunction>(mass_qs_.get(), 1);
  invrho_qf_->SetVDim(1);
  ck_qf_ = std::make_unique<mfem::QuadratureFunction>(sigma_qs_.get(), 1);
  ck_qf_->SetVDim(1);
  src_qf_ = std::make_unique<mfem::QuadratureFunction>(rhs_qs_.get(), 1);
  src_qf_->SetVDim(1);

  if (sigma_fqs_)
  {
    u_face_qf_ =
      std::make_unique<mfem::QuadratureFunction>(sigma_fqs_.get(), num_eq_);
    u_face_qf_->SetVDim(num_eq_);
    ck_face_qf_ =
      std::make_unique<mfem::QuadratureFunction>(sigma_fqs_.get(), 1);
    ck_face_qf_->SetVDim(1);
  }
  else
  {
    u_face_qf_.reset();
    ck_face_qf_.reset();
  }

  invrho_qf_coef_ =
    std::make_unique<mfem::QuadratureFunctionCoefficient>(*invrho_qf_);
  ck_qf_coef_ =
    std::make_unique<mfem::QuadratureFunctionCoefficient>(*ck_qf_);
  src_qf_coef_ =
    std::make_unique<mfem::QuadratureFunctionCoefficient>(*src_qf_);
  if (ck_face_qf_)
  {
    ck_face_qf_coef_ =
      std::make_unique<mfem::QuadratureFunctionCoefficient>(*ck_face_qf_);
  }
  else
  {
    ck_face_qf_coef_.reset();
  }

  device_mesh_sequence_ = mesh_seq;
  device_sfes_sequence_ = sfes_seq;
  device_vfes_sequence_ = vfes_seq;
  device_qorder_ = diff_qorder;
  device_mass_qorder_ = mass_qorder;
  device_diff_qorder_ = diff_qorder;
  device_rhs_qorder_ = rhs_qorder;
  device_face_qorder_ = face_qorder;
}

void IGRSolver::EnsureDeviceElementAlpha_()
{
  if (alpha_length_scale_policy_ != AlphaLengthScalePolicy::Local)
  {
    return;
  }

  mfem::Mesh* mesh = sfes_.GetMesh();
  MFEM_VERIFY(mesh,
              "IGR local-alpha device coefficients require a valid mesh.");
  MFEM_VERIFY(mesh == vfes_.GetMesh(),
              "IGR local-alpha device coefficients expect sigma and state "
              "meshes to match.");

  const int ne = mesh->GetNE();
  const long mesh_sequence = mesh->GetSequence();
  const bool require_positive = IsDGSpace(sfes_) && UseExactDGFaceOperator_();
  const bool needs_refresh =
    device_element_alpha_.Size() != ne ||
    device_element_alpha_mesh_sequence_ != mesh_sequence ||
    device_element_alpha_c_alpha_ != alpha_ ||
    (require_positive && !device_element_alpha_positive_validated_);
  if (!needs_refresh)
  {
    return;
  }

  device_element_alpha_.SetSize(ne);
  device_element_alpha_.UseDevice(true);
  mfem::real_t* element_alpha = device_element_alpha_.HostWrite();
  for (int e = 0; e < ne; ++e)
  {
    const double h = static_cast<double>(mesh->GetElementSize(e, 1));
    element_alpha[e] = static_cast<mfem::real_t>(alpha_ * h * h);
    if (require_positive)
      MFEM_VERIFY(element_alpha[e] > 0.0 && std::isfinite(element_alpha[e]),
                  "IGR local-alpha policy produced a nonpositive or nonfinite element coefficient.");
  }
  device_element_alpha_mesh_sequence_ = mesh_sequence;
  device_element_alpha_c_alpha_ = alpha_;
  device_element_alpha_positive_validated_ = require_positive;
}

void IGRSolver::UpdateDeviceCoefficients_(const mfem::ParGridFunction& U)
{
  // Configuration setters run after the constructor builds its initial form.
  // Validate the final operator policy before any current-state interpolation.
  MFEM_VERIFY(OperatorQuadratureMinimumOrder() == 0 ||
                (dim_ == 2 && UseExactDGFaceOperator_()),
              "The IGR operator-quadrature experiment requires 2D exact side-aware DG faces.");
  EnsureDeviceQuadrature_();
  MFEM_VERIFY(mass_qs_ && sigma_qs_ && rhs_qs_,
              "IGR device coefficients require quadrature spaces.");
  const bool use_element_local_alpha =
    alpha_length_scale_policy_ == AlphaLengthScalePolicy::Local;
  if (use_element_local_alpha)
  {
    // Refresh this host-derived geometry cache before launching state
    // interpolation so the common steady-state path remains device-only.
    EnsureDeviceElementAlpha_();
  }

  const bool ingestion_audit =
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_INGESTION_AUDIT", false);
  const bool need_diff_gradient = ingestion_audit ||
    (diagnostic_nan_scan_ && !device_coeff_diagnostics_logged_);
  if (need_diff_gradient && !grad_u_qf_)
  {
    grad_u_qf_ = std::make_unique<mfem::QuadratureFunction>(
      sigma_qs_.get(), num_eq_ * dim_);
  }

  const bool debug_qf =
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-start" << "\n";
  }

  if (backend_ == GradBackend::H1Projection)
  {
    if (!device_coeff_logged_ && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=grad-backend-override"
                << " requested=h1"
                << " effective=broken-dg"
                << " reason=device-qf" << "\n";
    }
  }

  const mfem::FiniteElementSpace& ufes = *U.ParFESpace();
  const bool use_tensor = mfem::UsesTensorBasis(ufes);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-uses-tensor"
              << " enabled=" << (use_tensor ? 1 : 0) << "\n";
  }
  const mfem::ElementDofOrdering ordering =
    use_tensor ? mfem::ElementDofOrdering::LEXICOGRAPHIC
               : mfem::ElementDofOrdering::NATIVE;

  const mfem::Operator* R = ufes.GetElementRestriction(ordering);
  MFEM_VERIFY(R, "IGR device coefficients require element restriction.");
  device_qinterp_evec_.SetSize(R->Height());
  R->Mult(U, device_qinterp_evec_);

  const int dim = dim_;
  const int num_eq = num_eq_;
  const mfem::real_t alpha = static_cast<mfem::real_t>(alpha_);
  const mfem::real_t rho_floor = static_cast<mfem::real_t>(rho_floor_);

  auto project_state =
    [&](mfem::QuadratureSpace& qs,
        mfem::QuadratureFunction* values,
        mfem::QuadratureFunction* derivatives)
  {
    if (!values && !derivatives) { return; }
    const mfem::QuadratureInterpolator* qi =
      ufes.GetQuadratureInterpolator(qs);
    MFEM_VERIFY(qi,
                "IGR device coefficients require quadrature interpolator.");
    qi->SetOutputLayout(mfem::QVectorLayout::byVDIM);
    qi->DisableTensorProducts(!use_tensor);
    if (values) { qi->Values(device_qinterp_evec_, *values); }
    if (derivatives)
    {
      qi->PhysDerivatives(device_qinterp_evec_, *derivatives);
    }
  };

  project_state(*mass_qs_, u_mass_qf_.get(), nullptr);
  project_state(*sigma_qs_, device_diff_values_share_mass_ ? nullptr : u_qf_.get(),
                need_diff_gradient ? grad_u_qf_.get() : nullptr);
  project_state(*rhs_qs_,
                (device_rhs_values_share_mass_ || device_rhs_values_share_diff_)
                  ? nullptr : u_rhs_qf_.get(),
                grad_u_rhs_qf_.get());

  // Read the owner directly; unused wrappers never receive Values or Read
  // and therefore need no device buffer.  These references are chosen anew for
  // this update, with no memory aliases or persistent current-state reuse.
  const mfem::QuadratureFunction& mass_values = *u_mass_qf_;
  const mfem::QuadratureFunction& diff_values = device_diff_values_share_mass_
    ? mass_values : *u_qf_;
  const mfem::QuadratureFunction& rhs_values = device_rhs_values_share_mass_
    ? mass_values : (device_rhs_values_share_diff_ ? diff_values : *u_rhs_qf_);

  const int mass_ne = mass_qs_->GetNE();
  const int mass_nq = mass_qs_->GetIntRule(0).GetNPoints();
  const int diff_ne = sigma_qs_->GetNE();
  const int diff_nq = sigma_qs_->GetIntRule(0).GetNPoints();
  const int rhs_ne = rhs_qs_->GetNE();
  const int rhs_nq = rhs_qs_->GetIntRule(0).GetNPoints();
  if (use_element_local_alpha)
  {
    MFEM_VERIFY(device_element_alpha_.Size() == diff_ne &&
                  device_element_alpha_.Size() == rhs_ne,
                "IGR local-alpha element cache must match volume quadrature "
                "element counts.");
  }

  const mfem::real_t* u_mass_data = mass_values.Read();
  const mfem::real_t* u_diff_data = diff_values.Read();
  const mfem::real_t* u_rhs_data = rhs_values.Read();
  const mfem::real_t* grad_rhs_data = grad_u_rhs_qf_->Read();
  mfem::real_t* invrho_data = invrho_qf_->Write();
  mfem::real_t* ck_data = ck_qf_->Write();
  mfem::real_t* src_data = src_qf_->Write();

#ifdef DG_IGR_HAS_ACCEL_FORALL
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=device-qf-kernel-start" << "\n";
  }
  igr_device::IGRDeviceInvRhoKernel invrho_kernel;
  invrho_kernel.nq = mass_nq;
  invrho_kernel.ne = mass_ne;
  invrho_kernel.num_eq = num_eq;
  invrho_kernel.rho_floor = rho_floor;
  invrho_kernel.u = u_mass_data;
  invrho_kernel.invrho = invrho_data;
  mfem::forall(mass_ne * mass_nq, invrho_kernel);

  if (use_element_local_alpha)
  {
    const mfem::real_t* element_alpha = device_element_alpha_.Read();

    igr_device::IGRDeviceLocalAlphaOverRhoKernel ck_kernel;
    ck_kernel.nq = diff_nq;
    ck_kernel.ne = diff_ne;
    ck_kernel.num_eq = num_eq;
    ck_kernel.rho_floor = rho_floor;
    ck_kernel.element_alpha = element_alpha;
    ck_kernel.u = u_diff_data;
    ck_kernel.ck = ck_data;
    mfem::forall(diff_ne * diff_nq, ck_kernel);

    igr_device::IGRDeviceLocalSourceKernel src_kernel;
    src_kernel.nq = rhs_nq;
    src_kernel.ne = rhs_ne;
    src_kernel.dim = dim;
    src_kernel.num_eq = num_eq;
    src_kernel.rho_floor = rho_floor;
    src_kernel.element_alpha = element_alpha;
    src_kernel.u = u_rhs_data;
    src_kernel.grad = grad_rhs_data;
    src_kernel.src = src_data;
    mfem::forall(rhs_ne * rhs_nq, src_kernel);
  }
  else
  {
    igr_device::IGRDeviceAlphaOverRhoKernel ck_kernel;
    ck_kernel.nq = diff_nq;
    ck_kernel.ne = diff_ne;
    ck_kernel.num_eq = num_eq;
    ck_kernel.alpha = alpha;
    ck_kernel.rho_floor = rho_floor;
    ck_kernel.u = u_diff_data;
    ck_kernel.ck = ck_data;
    mfem::forall(diff_ne * diff_nq, ck_kernel);

    igr_device::IGRDeviceSourceKernel src_kernel;
    src_kernel.nq = rhs_nq;
    src_kernel.ne = rhs_ne;
    src_kernel.dim = dim;
    src_kernel.num_eq = num_eq;
    src_kernel.alpha = alpha;
    src_kernel.rho_floor = rho_floor;
    src_kernel.u = u_rhs_data;
    src_kernel.grad = grad_rhs_data;
    src_kernel.src = src_data;
    mfem::forall(rhs_ne * rhs_nq, src_kernel);
  }
#else
  const auto u_mass = mfem::Reshape(u_mass_data, num_eq, mass_nq, mass_ne);
  auto invrho = mfem::Reshape(invrho_data, mass_nq, mass_ne);
  for (int e = 0; e < mass_ne; ++e)
  {
    for (int q = 0; q < mass_nq; ++q)
    {
      invrho(q, e) =
        frozen_scalar_igr_algebra::SafeInverseDensity(u_mass(0, q, e),
                                                      rho_floor);
    }
  }

  const auto u_diff = mfem::Reshape(u_diff_data, num_eq, diff_nq, diff_ne);
  auto ck = mfem::Reshape(ck_data, diff_nq, diff_ne);
  const mfem::real_t* element_alpha =
    use_element_local_alpha ? device_element_alpha_.HostRead() : nullptr;
  for (int e = 0; e < diff_ne; ++e)
  {
    const mfem::real_t alpha_e =
      use_element_local_alpha ? element_alpha[e] : alpha;
    for (int q = 0; q < diff_nq; ++q)
    {
      ck(q, e) = frozen_scalar_igr_algebra::AlphaOverDensity(alpha_e,
                                                             u_diff(0, q, e),
                                                             rho_floor);
    }
  }

  auto src = mfem::Reshape(src_data, rhs_nq, rhs_ne);
  for (int e = 0; e < rhs_ne; ++e)
  {
    const mfem::real_t alpha_e =
      use_element_local_alpha ? element_alpha[e] : alpha;
    for (int q = 0; q < rhs_nq; ++q)
    {
      src(q, e) =
        frozen_scalar_igr_algebra::SourceFromConservativeStateGradient(
          dim,
          u_rhs_data + num_eq * (q + rhs_nq * e),
          1,
          grad_rhs_data + num_eq * dim * (q + rhs_nq * e),
          1,
          num_eq,
          alpha_e,
          rho_floor);
    }
  }
#endif

  const auto nodal_source_mode = GetNodalSourceMode();
  if (nodal_source_mode != NodalSourceMode::Baseline)
  {
    if (!nodal_velocity_source_cache_)
      nodal_velocity_source_cache_ = std::make_shared<NodalVelocitySourceCache>();
    MFEM_VERIFY(!NodalSourceMirrorWalls() || HasNodalSourceBoundaryMarker(),
                "Mirror source requires the actual Euler slip boundary marker before state update.");
    ApplyNodalVelocitySource2D(ufes, *rhs_qs_, device_qinterp_evec_, alpha,
      use_element_local_alpha ? &device_element_alpha_ : nullptr, rho_floor,
      nodal_source_mode == NodalSourceMode::Central,
      *nodal_velocity_source_cache_, *src_qf_,
      NodalSourceMirrorWalls() ? &nodal_source_slip_marker_ : nullptr);
  }

  if (ingestion_audit)
  {
    const int solve_call = linear_solve_calls_ + 1;
    const MPI_Comm comm = vfes_.GetComm();
    LogStateQuadratureAudit("mass",
                            mass_values,
                            *mass_qs_,
                            num_eq,
                            dim,
                            rho_floor,
                            solve_call,
                            comm);
    LogStateQuadratureAudit("diffusion",
                            diff_values,
                            *sigma_qs_,
                            num_eq,
                            dim,
                            rho_floor,
                            solve_call,
                            comm);
    LogStateQuadratureAudit("rhs",
                            rhs_values,
                            *rhs_qs_,
                            num_eq,
                            dim,
                            rho_floor,
                            solve_call,
                            comm);
    LogScalarQuadratureAudit("grad_diffusion",
                             *grad_u_qf_,
                             solve_call,
                             comm);
    LogScalarQuadratureAudit("grad_rhs",
                             *grad_u_rhs_qf_,
                             solve_call,
                             comm);
    LogScalarQuadratureAudit("invrho", *invrho_qf_, solve_call, comm);
    LogScalarQuadratureAudit("alpha_over_rho", *ck_qf_, solve_call, comm);
    LogScalarQuadratureAudit("rhs_source", *src_qf_, solve_call, comm);
  }

  if (diagnostic_nan_scan_ && !device_coeff_diagnostics_logged_)
  {
    const MPI_Comm comm = vfes_.GetComm();
    int finite[8] = {VectorIsFinite(mass_values) ? 1 : 0,
                     VectorIsFinite(diff_values) ? 1 : 0,
                     VectorIsFinite(rhs_values) ? 1 : 0,
                     VectorIsFinite(*grad_u_qf_) ? 1 : 0,
                     VectorIsFinite(*grad_u_rhs_qf_) ? 1 : 0,
                     VectorIsFinite(*invrho_qf_) ? 1 : 0,
                     VectorIsFinite(*ck_qf_) ? 1 : 0,
                     VectorIsFinite(*src_qf_) ? 1 : 0};
    MPI_Allreduce(MPI_IN_PLACE, finite, 8, MPI_INT, MPI_MIN, comm);
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=device-qf-finite"
                << " u_mass=" << finite[0]
                << " u_diff=" << finite[1]
                << " u_rhs=" << finite[2]
                << " grad_diff=" << finite[3]
                << " grad_rhs=" << finite[4]
                << " invrho=" << finite[5]
                << " ck=" << finite[6]
                << " src=" << finite[7] << "\n";
    }
    if (!(finite[0] && finite[1] && finite[2] && finite[3] && finite[4] &&
          finite[5] && finite[6] && finite[7]))
    {
      MFEM_ABORT("IGR device quadrature data contains non-finite values.");
    }

    const mfem::Vector& inv_vals = *invrho_qf_;
    const mfem::Vector& ck_vals = *ck_qf_;
    const mfem::Vector& src_vals = *src_qf_;
    auto host_minmax = [](const mfem::Vector& v, double& vmin, double& vmax)
    {
      const double* data = v.HostRead();
      const int n = v.Size();
      vmin = (n > 0) ? data[0] : 0.0;
      vmax = vmin;
      for (int i = 1; i < n; ++i)
      {
        vmin = std::min(vmin, data[i]);
        vmax = std::max(vmax, data[i]);
      }
    };
    double inv_min = 0.0, inv_max = 0.0;
    double ck_min = 0.0, ck_max = 0.0;
    double src_min = 0.0, src_max = 0.0;
    host_minmax(inv_vals, inv_min, inv_max);
    host_minmax(ck_vals, ck_min, ck_max);
    host_minmax(src_vals, src_min, src_max);
    MPI_Allreduce(MPI_IN_PLACE, &inv_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &inv_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &ck_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &ck_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &src_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &src_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=device-qf-range"
                << " invrho_min=" << inv_min
                << " invrho_max=" << inv_max
                << " ck_min=" << ck_min
                << " ck_max=" << ck_max
                << " src_min=" << src_min
                << " src_max=" << src_max << "\n";
    }

    if (mfem::Mpi::Root() && inv_vals.Size() > 0 && ck_vals.Size() > 0 &&
        src_vals.Size() > 0)
    {
      const mfem::real_t inv_q = inv_vals(0);
      const mfem::real_t ck_q = ck_vals(0);
      const mfem::real_t src_q = src_vals(0);
      std::cout << "[igr-linear] event=device-qf-sample"
                << " invrho_q=" << inv_q
                << " ck_q=" << ck_q
                << " src_q=" << src_q
                << " mass_qorder=" << device_mass_qorder_
                << " diff_qorder=" << device_diff_qorder_
                << " rhs_qorder=" << device_rhs_qorder_
                << " source_qorder_min_requested=" << NodalSourceQuadratureMinimumOrder()
                << " operator_qorder_min_requested=" << OperatorQuadratureMinimumOrder()
                << " face_qorder=" << device_face_qorder_
                << " source_mode=" << NodalSourceModeName(GetNodalSourceMode()) << "\n";
    }
    device_coeff_diagnostics_logged_ = true;
  }

  // Exact faces consume their independently projected side coefficients. The
  // single-valued MFEM projection below is unused by that operator and has a
  // separate Q1D<=10 limit. Skip it only for this explicit quadrature study.
  const bool skip_unused_face_projection = OperatorQuadratureMinimumOrder() > 0;
  if (skip_unused_face_projection && !device_coeff_logged_ && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=face-qf-projector"
              << " mode=skipped-unused-single-valued"
              << " reason=oversampled-exact-side-operator\n";
  }
  if (sigma_fqs_ && ck_face_qf_ && !skip_unused_face_projection)
  {
    if (!device_coeff_logged_ && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=face-qf-projector"
                << " mode=single-valued-mfem-face-restriction"
                << " diagnostic_only=1"
                << " reason=side-dependent-coefficient-not-represented"
                << "\n";
    }
    const auto face_info =
      dg_face::ProjectFaceQuadratureValues(ufes,
                                           *sigma_fqs_,
                                           U,
                                           *u_face_qf_,
                                           &device_face_restr_evec_);
    if (debug_qf && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=face-qf-sizes"
                << " dim=" << face_info.dim
                << " vdim=" << face_info.vdim
                << " nd1d=" << face_info.nd1d
                << " nq1d=" << face_info.nq1d
                << " nf=" << face_info.nf
                << " f_vec_size=" << face_info.f_vec_size
                << " qf_size=" << face_info.qf_size
                << std::endl;
    }

    const int nf = sigma_fqs_->GetNE();
    const int nqf = sigma_fqs_->GetIntRule(0).GetNPoints();
    const mfem::real_t* uface_data = u_face_qf_->Read();
    mfem::real_t* ckface_data = ck_face_qf_->Write();
    const auto uface = mfem::Reshape(uface_data, num_eq, nqf, nf);
    auto ckface = mfem::Reshape(ckface_data, nqf, nf);

#ifdef DG_IGR_HAS_ACCEL_FORALL
    igr_device::IGRDeviceFaceCoeffKernel face_kernel;
    face_kernel.nq = nqf;
    face_kernel.nf = nf;
    face_kernel.num_eq = num_eq;
    face_kernel.alpha = alpha;
    face_kernel.rho_floor = rho_floor;
    face_kernel.uface = uface_data;
    face_kernel.ckface = ckface_data;
    mfem::forall(nf * nqf, face_kernel);
#else
    for (int f = 0; f < nf; ++f)
    {
      for (int q = 0; q < nqf; ++q)
      {
        ckface(q, f) =
          frozen_scalar_igr_algebra::AlphaOverDensity(alpha,
                                                      uface(0, q, f),
                                                      rho_floor);
      }
    }
#endif
  }

  device_coeff_logged_ = true;
}

} // namespace igrlab
