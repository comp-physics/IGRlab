#include "IGR_stage_coefficients_cuda.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "mfem/linalg/kernels.hpp"
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <typeinfo>

namespace igrlab
{
namespace igr_internal
{
namespace
{
constexpr int d1d = 6;
constexpr int ndof = d1d*d1d;
constexpr int mass_q1d = 7;
constexpr int mass_nq = mass_q1d*mass_q1d;
constexpr int components = 4;
constexpr int warps = 4;
constexpr int b6_offset = 0;
constexpr int g6_offset = ndof;
constexpr int b7_offset = 2*ndof;
constexpr int basis_size = b7_offset + mass_q1d*d1d;

static_assert(std::is_same<mfem::real_t, double>::value,
              "Fused stage coefficients require MFEM FP64.");

bool Reject(std::string* reason, const char* message)
{
  if (reason) { *reason = message; }
  return false;
}

// Exact coordinates, weights and ordering are checked against an independent
// GL rule container, not inferred from the rule's order or point count. This
// bounded reference-data comparison also catches in-place rule changes.
bool SameRule(const mfem::IntegrationRule& a, const mfem::IntegrationRule& b)
{
  if (a.GetNPoints() != b.GetNPoints() || a.GetOrder() != b.GetOrder())
  { return false; }
  for (int q = 0; q < a.GetNPoints(); ++q)
  {
    const auto& x = a.IntPoint(q);
    const auto& y = b.IntPoint(q);
    if (x.x != y.x || x.y != y.y || x.weight != y.weight) { return false; }
  }
  return true;
}

struct CanonicalRules
{
  mfem::IntegrationRules rules{0, mfem::Quadrature1D::GaussLegendre};
  const mfem::IntegrationRule& gl6 = rules.Get(mfem::Geometry::SQUARE, 11);
  const mfem::IntegrationRule& gl7 = rules.Get(mfem::Geometry::SQUARE, 13);
};

const mfem::FiniteElementSpace* GeometrySpace(const mfem::Mesh& mesh)
{
  const auto* nodes = mesh.GetNodes();
  return nodes ? nodes->FESpace() : nullptr;
}

bool StandardStateElement(const mfem::FiniteElement& fe)
{
  const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&fe);
  return typeid(fe) == typeid(mfem::L2_QuadrilateralElement) && tensor &&
    fe.GetDim() == 2 && fe.GetOrder() == 5 && fe.GetDof() == ndof &&
    fe.GetGeomType() == mfem::Geometry::SQUARE &&
    fe.GetMapType() == mfem::FiniteElement::VALUE &&
    tensor->GetBasisType() == mfem::BasisType::GaussLegendre;
}

bool StandardGeometryElement(const mfem::FiniteElement& fe)
{
  const bool standard = typeid(fe) == typeid(mfem::H1_QuadrilateralElement) ||
                        typeid(fe) == typeid(mfem::L2_QuadrilateralElement);
  const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&fe);
  const int p = fe.GetOrder();
  return standard && tensor && p >= 1 && p <= 2 &&
    fe.GetDim() == 2 && fe.GetGeomType() == mfem::Geometry::SQUARE &&
    fe.GetMapType() == mfem::FiniteElement::VALUE &&
    fe.GetDof() == (p+1)*(p+1) &&
    (tensor->GetBasisType() == mfem::BasisType::GaussLegendre ||
     tensor->GetBasisType() == mfem::BasisType::GaussLobatto);
}

bool CheckGeometrySpace(const mfem::Mesh& mesh, std::string* reason)
{
  const auto* nodes = mesh.GetNodes();
  if (!nodes) { return true; } // MFEM supplies its built-in Q1 transformation.
  const auto* gfes = nodes->FESpace();
  if (!gfes || gfes->GetMesh() != &mesh || gfes->GetVDim() != 2 ||
      gfes->GetNE() != mesh.GetNE() || gfes->IsVariableOrder() ||
      gfes->GetNURBSext() || nodes->Size() != gfes->GetVSize())
  { return Reject(reason, "unsupported-geometry-space"); }
  const auto* fec = gfes->FEColl();
  if (!fec || (typeid(*fec) != typeid(mfem::H1_FECollection) &&
               typeid(*fec) != typeid(mfem::L2_FECollection)) ||
      !StandardGeometryElement(*gfes->GetFE(0)))
  { return Reject(reason, "requires-builtin-q1-q2-geometry"); }
  return true;
}

// A setup-only sample checks the actual transformations of the first, middle
// and last elements at both rules. This is not a proof of positivity on every
// cell or throughout a curved element. No geometry/state host scan is done
// when the cache is reused.
bool CheckSampledGeometry(mfem::Mesh& mesh,
                          const mfem::IntegrationRule& mass_rule,
                          const mfem::IntegrationRule& source_rule,
                          std::string* reason)
{
  const int samples[3] = {0, mesh.GetNE()/2, mesh.GetNE()-1};
  for (int s = 0; s < 3; ++s)
  {
    if (s && samples[s] == samples[s-1]) { continue; }
    auto* tr = mesh.GetElementTransformation(samples[s]);
    const auto* iso = dynamic_cast<const mfem::IsoparametricTransformation*>(tr);
    if (!iso || typeid(*tr) != typeid(mfem::IsoparametricTransformation) ||
        !iso->GetFE() || tr->GetSpaceDim() != 2 ||
        !(StandardGeometryElement(*iso->GetFE()) ||
          (!mesh.GetNodes() &&
           typeid(*iso->GetFE()) == typeid(mfem::BiLinear2DFiniteElement))))
    { return Reject(reason, "unsupported-sampled-geometry-mapping"); }
    const mfem::IntegrationRule* rules[2] = {&mass_rule, &source_rule};
    for (const auto* rule : rules)
    {
      for (int q = 0; q < rule->GetNPoints(); ++q)
      {
        tr->SetIntPoint(&rule->IntPoint(q));
        const auto& j = tr->Jacobian();
        const double a = j(0,0), b = j(1,0), c = j(0,1), d = j(1,1);
        const double det = a*d-b*c;
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
            !std::isfinite(d) || !std::isfinite(det) || det <= 0.0)
        { return Reject(reason, "invalid-sampled-geometry-jacobian"); }
      }
    }
  }
  return true;
}

// Four independent element warps per CTA. Only rho, mx and my enter shared
// memory. The current conservative values/reference derivatives and physical
// gradients remain on chip; the energy field is never loaded.
template <bool identity_b6>
__global__ void SourceAndDiffusionKernel(
  int ne, const double* basis, const double* jacobians,
  const double* state, double alpha, double rho_floor,
  const double* element_alpha, double* ck, double* src)
{
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int e = blockIdx.x*warps+warp;
  __shared__ double b[ndof], g[ndof];
  __shared__ double u[warps][3][ndof];
  __shared__ double bx[warps][3][ndof], gx[warps][3][ndof];
  if (threadIdx.x < ndof)
  {
    b[threadIdx.x] = basis[b6_offset+threadIdx.x];
    g[threadIdx.x] = basis[g6_offset+threadIdx.x];
  }
  __syncthreads();
  if (e >= ne) { return; }
  for (int i = lane; i < 3*ndof; i += 32)
  {
    const int c = i/ndof, j = i%ndof;
    u[warp][c][j] = state[j+ndof*(c+components*e)];
  }
  __syncwarp();
  if constexpr (!identity_b6)
  {
    for (int i = lane; i < 3*ndof; i += 32)
    {
      const int c = i/ndof, qx = i%d1d, dy = (i%ndof)/d1d;
      double value = 0.0, derivative = 0.0;
#pragma unroll
      for (int dx = 0; dx < d1d; ++dx)
      {
        const double input = u[warp][c][dx+d1d*dy];
        value += input*b[qx+d1d*dx];
        derivative += input*g[qx+d1d*dx];
      }
      bx[warp][c][qx+d1d*dy] = value;
      gx[warp][c][qx+d1d*dy] = derivative;
    }
    __syncwarp();
  }
  const double alpha_e = element_alpha ? element_alpha[e] : alpha;
  for (int q = lane; q < ndof; q += 32)
  {
    const int qx = q%d1d, qy = q/d1d;
    double values[3], physical_gradient[6];
    double j[4], ji[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) { j[k] = jacobians[q+ndof*(k+4*e)]; }
    // Exactly the full 2x2 inverse used by MFEM PhysDerivatives, including
    // both off-diagonal entries on mapped/curved elements.
    mfem::kernels::CalcInverse<2>(j, ji);
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
      double value = 0.0, dr = 0.0, ds = 0.0;
      if constexpr (identity_b6)
      {
        value = u[warp][c][q];
#pragma unroll
        for (int k = 0; k < d1d; ++k)
        {
          dr += u[warp][c][k+d1d*qy]*g[qx+d1d*k];
          ds += u[warp][c][qx+d1d*k]*g[qy+d1d*k];
        }
      }
      else
      {
#pragma unroll
        for (int k = 0; k < d1d; ++k)
        {
          const double v = bx[warp][c][qx+d1d*k];
          value += v*b[qy+d1d*k];
          dr += gx[warp][c][qx+d1d*k]*b[qy+d1d*k];
          ds += v*g[qy+d1d*k];
        }
      }
      values[c] = value;
      physical_gradient[c] = ji[0]*dr+ji[1]*ds;
      physical_gradient[c+3] = ji[2]*dr+ji[3]*ds;
    }
    ck[q+ndof*e] = frozen_scalar_igr_algebra::AlphaOverDensity(
      alpha_e, values[0], rho_floor);
    // The existing algebra applies the conservative quotient rule and
    // trace(grad(u)^2), including the signed cross term, without needing E.
    src[q+ndof*e] =
      frozen_scalar_igr_algebra::SourceFromConservativeStateGradient(
        2, values, 1, physical_gradient, 1, 3, alpha_e, rho_floor);
  }
}

// The original GL7 mass rule remains separate. A density-only pair of tensor
// contractions writes 1/rho directly, without a four-component QF temporary.
__global__ void InverseDensityKernel(
  int ne, const double* basis, const double* state,
  double rho_floor, double* invrho)
{
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int e = blockIdx.x*warps+warp;
  __shared__ double b[mass_q1d*d1d];
  __shared__ double rho[warps][ndof], bx[warps][mass_q1d*d1d];
  if (threadIdx.x < mass_q1d*d1d)
  { b[threadIdx.x] = basis[b7_offset+threadIdx.x]; }
  __syncthreads();
  if (e >= ne) { return; }
  for (int i = lane; i < ndof; i += 32)
  { rho[warp][i] = state[i+ndof*components*e]; }
  __syncwarp();
  for (int i = lane; i < mass_q1d*d1d; i += 32)
  {
    const int qx = i%mass_q1d, dy = i/mass_q1d;
    double value = 0.0;
#pragma unroll
    for (int dx = 0; dx < d1d; ++dx)
    { value += rho[warp][dx+d1d*dy]*b[qx+mass_q1d*dx]; }
    bx[warp][i] = value;
  }
  __syncwarp();
  for (int q = lane; q < mass_nq; q += 32)
  {
    const int qx = q%mass_q1d, qy = q/mass_q1d;
    double density = 0.0;
#pragma unroll
    for (int dy = 0; dy < d1d; ++dy)
    { density += bx[warp][qx+mass_q1d*dy]*b[qy+mass_q1d*dy]; }
    invrho[q+mass_nq*e] =
      frozen_scalar_igr_algebra::SafeInverseDensity(density, rho_floor);
  }
}
}

struct FusedStageCoefficientCache
{
  mfem::Vector basis;
  bool identity_b6 = false;
  const mfem::ParFiniteElementSpace* fes = nullptr;
  const mfem::Mesh* mesh = nullptr;
  const mfem::FiniteElementCollection* fec = nullptr;
  const mfem::FiniteElement* state_fe = nullptr;
  const mfem::GridFunction* nodes = nullptr;
  const mfem::FiniteElementSpace* nodes_fes = nullptr;
  const mfem::FiniteElementCollection* nodes_fec = nullptr;
  const mfem::FiniteElement* geometry_fe = nullptr;
  const mfem::QuadratureSpace* spaces[3] = {};
  const mfem::IntegrationRule* rules[3] = {};
  const mfem::ElementRestrictionOperator* restriction = nullptr;
  long mesh_sequence = -1, nodes_sequence = -1, fes_sequence = -1;
  long nodes_fes_sequence = -1;
  int ne = 0, vsize = 0, ordering = -1, nodes_size = 0;

  void Capture(const mfem::ParFiniteElementSpace& ufes,
               const mfem::QuadratureSpace& mass,
               const mfem::QuadratureSpace& diff,
               const mfem::QuadratureSpace& rhs)
  {
    fes = &ufes; mesh = ufes.GetMesh(); fec = ufes.FEColl();
    state_fe = ufes.GetFE(0); nodes = mesh->GetNodes();
    nodes_fes = GeometrySpace(*mesh);
    nodes_fec = nodes_fes ? nodes_fes->FEColl() : nullptr;
    geometry_fe = nodes_fes ? nodes_fes->GetFE(0) : nullptr;
    mesh_sequence = mesh->GetSequence(); nodes_sequence = mesh->GetNodesSequence();
    fes_sequence = ufes.GetSequence();
    nodes_fes_sequence = nodes_fes ? nodes_fes->GetSequence() : -1;
    ne = ufes.GetNE(); vsize = ufes.GetVSize();
    ordering = static_cast<int>(ufes.GetOrdering());
    nodes_size = nodes ? nodes->Size() : 0;
    spaces[0] = &mass; spaces[1] = &diff; spaces[2] = &rhs;
    for (int k = 0; k < 3; ++k) { rules[k] = &spaces[k]->GetIntRule(0); }
    restriction = ufes.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
  }

  bool Matches(const mfem::ParFiniteElementSpace& ufes,
               const mfem::QuadratureSpace& mass,
               const mfem::QuadratureSpace& diff,
               const mfem::QuadratureSpace& rhs) const
  {
    const auto* current_mesh = ufes.GetMesh();
    const auto* current_nodes = current_mesh->GetNodes();
    const auto* gfes = GeometrySpace(*current_mesh);
    return fes == &ufes && mesh == current_mesh && fec == ufes.FEColl() &&
      state_fe == ufes.GetFE(0) && nodes == current_nodes && nodes_fes == gfes &&
      nodes_fec == (gfes ? gfes->FEColl() : nullptr) &&
      geometry_fe == (gfes ? gfes->GetFE(0) : nullptr) &&
      mesh_sequence == current_mesh->GetSequence() &&
      nodes_sequence == current_mesh->GetNodesSequence() &&
      fes_sequence == ufes.GetSequence() &&
      nodes_fes_sequence == (gfes ? gfes->GetSequence() : -1) &&
      ne == ufes.GetNE() && vsize == ufes.GetVSize() &&
      ordering == static_cast<int>(ufes.GetOrdering()) &&
      nodes_size == (current_nodes ? current_nodes->Size() : 0) &&
      spaces[0] == &mass && spaces[1] == &diff && spaces[2] == &rhs &&
      rules[0] == &mass.GetIntRule(0) && rules[1] == &diff.GetIntRule(0) &&
      rules[2] == &rhs.GetIntRule(0) &&
      restriction == ufes.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
  }
};

namespace
{
bool BuildReferenceData(const mfem::ParFiniteElementSpace& ufes,
                         const mfem::QuadratureSpace& mass,
                         const mfem::QuadratureSpace& diff,
                         const mfem::QuadratureSpace& rhs,
                         FusedStageCoefficientCache& cache, std::string* reason)
{
  const auto* fec = ufes.FEColl();
  const auto* fe = ufes.GetFE(0);
  const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(fe);
  if (!fec || typeid(*fec) != typeid(mfem::L2_FECollection) ||
      !fe || !StandardStateElement(*fe))
  { return Reject(reason, "requires-builtin-p5-gl-l2-quads"); }
  const auto& lex = tensor->GetDofMap();
  if (lex.Size() != 0 && lex.Size() != ndof)
  { return Reject(reason, "unsupported-l2-element-permutation"); }
  for (int j = 0; j < lex.Size(); ++j)
  {
    if (lex[j] != j) { return Reject(reason, "requires-identity-l2-element-ordering"); }
  }
  const auto* restriction = ufes.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
  if (!restriction || typeid(*restriction) != typeid(mfem::L2ElementRestriction) ||
      restriction->Height() != components*ndof*ufes.GetNE() ||
      restriction->Width() != ufes.GetVSize())
  { return Reject(reason, "unsupported-element-restriction-layout"); }
  auto* mesh = ufes.GetMesh();
  if (!CheckGeometrySpace(*mesh, reason)) { return false; }
  const auto* gfes = GeometrySpace(*mesh);
  const auto* gfe = gfes ? gfes->GetFE(0) : nullptr;
  mfem::Array<int> dofs;
  // This full layout/type check runs only while constructing a cache. MFEM's
  // L2 restriction lays out ND x VDIM x NE, independent of L-vector ordering.
  for (int e = 0; e < ufes.GetNE(); ++e)
  {
    if (mesh->GetElementBaseGeometry(e) != mfem::Geometry::SQUARE ||
        ufes.GetFE(e) != fe || (gfes && gfes->GetFE(e) != gfe) ||
        &mass.GetIntRule(e) != &mass.GetIntRule(0) ||
        &diff.GetIntRule(e) != &diff.GetIntRule(0) ||
        &rhs.GetIntRule(e) != &rhs.GetIntRule(0) ||
        mass.Offset(e) != mass_nq*e || diff.Offset(e) != ndof*e ||
        rhs.Offset(e) != ndof*e)
    { return Reject(reason, "requires-uniform-quad-qf-layouts"); }
    ufes.GetElementDofs(e, dofs);
    if (dofs.Size() != ndof)
    { return Reject(reason, "unexpected-element-dof-count"); }
    for (int j = 0; j < ndof; ++j)
    {
      if (dofs[j] != ndof*e+j)
      { return Reject(reason, "unsupported-global-l2-element-layout"); }
    }
  }
  const auto& m = fe->GetDofToQuad(mass.GetIntRule(0), mfem::DofToQuad::TENSOR);
  const auto& d = fe->GetDofToQuad(diff.GetIntRule(0), mfem::DofToQuad::TENSOR);
  const auto& r = fe->GetDofToQuad(rhs.GetIntRule(0), mfem::DofToQuad::TENSOR);
  if (m.ndof != d1d || m.nqpt != mass_q1d || m.B.Size() != mass_q1d*d1d ||
      d.ndof != d1d || d.nqpt != d1d || d.B.Size() != ndof || d.G.Size() != ndof ||
      r.ndof != d1d || r.nqpt != d1d || r.B.Size() != ndof || r.G.Size() != ndof)
  { return Reject(reason, "unexpected-tensor-interpolation-maps"); }
  const double* bm = m.B.HostRead();
  const double* bd = d.B.HostRead(); const double* gd = d.G.HostRead();
  const double* br = r.B.HostRead(); const double* gr = r.G.HostRead();
  cache.basis.SetSize(basis_size); cache.basis.UseDevice(true);
  double* out = cache.basis.HostWrite();
  cache.identity_b6 = true;
  for (int i = 0; i < ndof; ++i)
  {
    if (!std::isfinite(br[i]) || !std::isfinite(gr[i]) ||
        bd[i] != br[i] || gd[i] != gr[i])
    { return Reject(reason, "diffusion-source-interpolation-mismatch"); }
    out[b6_offset+i] = br[i]; out[g6_offset+i] = gr[i];
    if (br[i] != (i%d1d == i/d1d ? 1.0 : 0.0)) { cache.identity_b6 = false; }
  }
  for (int i = 0; i < mass_q1d*d1d; ++i)
  {
    if (!std::isfinite(bm[i])) { return Reject(reason, "nonfinite-gl7-interpolation-map"); }
    out[b7_offset+i] = bm[i];
  }
  return CheckSampledGeometry(*mesh, mass.GetIntRule(0), rhs.GetIntRule(0), reason);
}
}

bool TryUpdateFusedStageCoefficientsCUDA(
  const mfem::ParFiniteElementSpace& ufes,
  const mfem::QuadratureSpace& mass_qs,
  const mfem::QuadratureSpace& diff_qs,
  const mfem::QuadratureSpace& rhs_qs,
  const mfem::Vector& restricted_state,
  double alpha, double rho_floor, const mfem::Vector* element_alpha,
  mfem::QuadratureFunction& invrho, mfem::QuadratureFunction& ck,
  mfem::QuadratureFunction& src,
  std::shared_ptr<FusedStageCoefficientCache>& cache, std::string* reason)
{
  if (reason) { reason->clear(); }
  if (!mfem::Device::Allows(mfem::Backend::CUDA))
  { return Reject(reason, "requires-cuda"); }
  auto* mesh = ufes.GetMesh();
  const int ne = ufes.GetNE();
  if (!mesh || mesh->Dimension() != 2 || mesh->SpaceDimension() != 2 ||
      ne <= 0 || mesh->GetNE() != ne || mesh->Nonconforming() ||
      ufes.Nonconforming() || ufes.IsVariableOrder() || ufes.GetNURBSext() ||
      ufes.GetVDim() != components || ne > std::numeric_limits<int>::max()/(components*ndof))
  { return Reject(reason, "requires-fixed-conforming-2d-four-component-space"); }
  const auto* current_fe = ufes.GetFE(0);
  if (!ufes.FEColl() || typeid(*ufes.FEColl()) != typeid(mfem::L2_FECollection) ||
      !current_fe || !StandardStateElement(*current_fe))
  { return Reject(reason, "requires-builtin-p5-gl-l2-quads"); }
  if (!CheckGeometrySpace(*mesh, reason)) { return false; }
  int ranks = 0; MPI_Comm_size(ufes.GetComm(), &ranks);
  if (ranks != 1) { return Reject(reason, "requires-one-mpi-rank"); }
  if (ufes.GetNDofs() != ndof*ne || ufes.GetVSize() != components*ndof*ne ||
      restricted_state.Size() != components*ndof*ne ||
      (ufes.GetOrdering() != mfem::Ordering::byNODES &&
       ufes.GetOrdering() != mfem::Ordering::byVDIM))
  { return Reject(reason, "unexpected-conservative-evec-layout"); }
  if (element_alpha && element_alpha->Size() != ne)
  { return Reject(reason, "element-alpha-size-mismatch"); }
  if (!std::isfinite(alpha) || !std::isfinite(rho_floor) || rho_floor <= 0.0)
  { return Reject(reason, "invalid-alpha-or-density-floor"); }
  if (mass_qs.GetMesh() != mesh || diff_qs.GetMesh() != mesh || rhs_qs.GetMesh() != mesh ||
      typeid(mass_qs) != typeid(mfem::QuadratureSpace) ||
      typeid(diff_qs) != typeid(mfem::QuadratureSpace) ||
      typeid(rhs_qs) != typeid(mfem::QuadratureSpace) ||
      mass_qs.GetNE() != ne || diff_qs.GetNE() != ne || rhs_qs.GetNE() != ne ||
      mass_qs.GetSize() != mass_nq*ne || diff_qs.GetSize() != ndof*ne ||
      rhs_qs.GetSize() != ndof*ne ||
      invrho.GetSpace() != &mass_qs || ck.GetSpace() != &diff_qs || src.GetSpace() != &rhs_qs ||
      invrho.GetVDim() != 1 || ck.GetVDim() != 1 || src.GetVDim() != 1 ||
      invrho.Size() != mass_nq*ne || ck.Size() != ndof*ne || src.Size() != ndof*ne ||
      &invrho == &ck || &invrho == &src || &ck == &src)
  { return Reject(reason, "unexpected-scalar-volume-qf-layouts"); }
  static CanonicalRules canonical;
  if (!SameRule(mass_qs.GetIntRule(0), canonical.gl7) ||
      !SameRule(diff_qs.GetIntRule(0), canonical.gl6) ||
      !SameRule(rhs_qs.GetIntRule(0), canonical.gl6))
  { return Reject(reason, "requires-exact-gl7-mass-matching-gl6-diffusion-source"); }

  const bool rebuild = !cache || !cache->Matches(ufes, mass_qs, diff_qs, rhs_qs);
  auto prepared = rebuild ? std::make_shared<FusedStageCoefficientCache>() : cache;
  if (rebuild && !BuildReferenceData(ufes, mass_qs, diff_qs, rhs_qs, *prepared, reason))
  { return false; }
  // Borrow MFEM's existing per-quadrature geometry buffer, just as its physical
  // derivative evaluator does. No duplicate inverse/gradient geometry is kept.
  // MFEM invalidates it through NodesUpdated()/mesh nodes sequence changes.
  const auto* geometry = mesh->GetGeometricFactors(
    rhs_qs.GetIntRule(0), mfem::GeometricFactors::JACOBIANS);
  if (!geometry || geometry->mesh != mesh || geometry->IntRule != &rhs_qs.GetIntRule(0) ||
      geometry->J.Size() != 4*ndof*ne)
  { return Reject(reason, "unexpected-physical-jacobian-layout"); }
  if (rebuild)
  {
    // GetGeometricFactors may install MFEM's equivalent Q1 nodal geometry on a
    // mesh initially represented only by vertices. Capture after that change.
    if (!CheckGeometrySpace(*mesh, reason)) { return false; }
    prepared->Capture(ufes, mass_qs, diff_qs, rhs_qs);
    cache = prepared;
  }
  const double* basis = prepared->basis.Read();
  const double* jacobians = geometry->J.Read();
  const double* state = restricted_state.Read();
  const double* local_alpha = element_alpha ? element_alpha->Read() : nullptr;
  const int blocks = (ne+warps-1)/warps;
  double* diffusion = ck.Write();
  double* source = src.Write();
  if (prepared->identity_b6)
  {
    SourceAndDiffusionKernel<true><<<blocks,32*warps>>>(ne, basis, jacobians,
      state, alpha, rho_floor, local_alpha, diffusion, source);
  }
  else
  {
    SourceAndDiffusionKernel<false><<<blocks,32*warps>>>(ne, basis, jacobians,
      state, alpha, rho_floor, local_alpha, diffusion, source);
  }
  MFEM_VERIFY(cudaGetLastError() == cudaSuccess,
              "Fused stage source/diffusion coefficient launch failed.");
  InverseDensityKernel<<<blocks,32*warps>>>(ne, basis, state, rho_floor, invrho.Write());
  MFEM_VERIFY(cudaGetLastError() == cudaSuccess,
              "Fused stage mass coefficient launch failed.");
  return true;
}
}
}
