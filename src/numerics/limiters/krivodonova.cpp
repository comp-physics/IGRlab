#include "numerics/limiters/krivodonova.hpp"

#include "limiters_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace {

// Build monomial index list (a,b) with total degree <= p
static void BuildMonomialAB(int p, std::vector<std::pair<int,int>> &ab)
{
  ab.clear();
  for (int t=0; t<=p; ++t) for (int a=0; a<=t; ++a) { int b = t - a; ab.emplace_back(a,b); }
}

// Evaluate monomial basis at reference point (xi,eta)
static inline double Mono(int a, int b, double xi, double eta)
{ double v=1.0; for(int i=0;i<a;++i) v*=xi; for(int j=0;j<b;++j) v*=eta; return v; }

// Get ref nodes (xi,eta) from FE nodes (works for L2 quads). Returns 2x dof.
static void RefDofXY(const mfem::FiniteElement &fe, mfem::Array2D<double> &xy)
{
  const mfem::IntegrationRule &nodes = fe.GetNodes();
  const int dof = fe.GetDof();
  xy.SetSize(2, dof);
  for (int j=0; j<dof; ++j) { const mfem::IntegrationPoint &ip = nodes.IntPoint(j); xy(0,j) = ip.x; xy(1,j) = ip.y; }
}

static void DenseTranspose(const mfem::DenseMatrix &A, mfem::DenseMatrix &AT)
{
  AT.SetSize(A.Width(), A.Height());
  for (int i = 0; i < A.Height(); ++i)
    for (int j = 0; j < A.Width(); ++j)
      AT(j, i) = A(i, j);
}

// Build Vandermonde V(dof x ncoef) and a robust LSQ projector
//   P(ncoef x dof) ~= (V^T V + lambda I)^-1 V^T
// where lambda is tiny Tikhonov regularization to avoid catastrophic inverses.
static void BuildVandermondeAndProjector(const mfem::FiniteElement &fe, int p,
                                         mfem::DenseMatrix &V, mfem::DenseMatrix &P,
                                         std::vector<std::pair<int,int>> &ab)
{
  // 1) reference DOF coordinates
  mfem::Array2D<double> xy;
  RefDofXY(fe, xy);

  // 2) monomial index list
  BuildMonomialAB(p, ab);

  const int dof   = fe.GetDof();
  const int ncoef = (p+1)*(p+2)/2;

  // 3) Vandermonde
  V.SetSize(dof, ncoef);
  for (int j = 0; j < dof; ++j)
  {
    const double xi  = xy(0, j);
    const double eta = xy(1, j);
    for (int k = 0; k < ncoef; ++k)
    {
      const int a = ab[k].first, b = ab[k].second;
      V(j, k) = Mono(a, b, xi, eta);
    }
  }

  // 4) Robust projector: P = (V^T V + lambda I)^-1 V^T
  mfem::DenseMatrix VT, G;
  DenseTranspose(V, VT);         // (ncoef x dof)
  mfem::Mult(VT, V, G);          // (ncoef x ncoef)

  // Tikhonov regularization lambda = eps * trace(G) / ncoef
  double tr = 0.0; for (int i = 0; i < G.Height(); ++i) tr += G(i,i);
  const double lam = std::max(1e-14, 1e-14 * tr / std::max(1, G.Height()));
  for (int i = 0; i < G.Height(); ++i) { G(i,i) += lam; }

  // Solve G * X = VT  => P = X
  // Use MFEM's DenseMatrixInverse (LU+pivoting) instead of CalcInverse
  mfem::DenseMatrixInverse inv(G);
  inv.Factor();                  // LU with pivoting
  inv.Mult(VT, P);               // P: (ncoef x dof)

  // Guard: if anything bad slipped in, fall back to zero projector (no limiting)
  auto sane = [](double v){ return std::isfinite(v); };
  bool ok = true;
  for (int i = 0; i < P.Height() && ok; ++i)
    for (int j = 0; j < P.Width(); ++j)
      if (!sane(P(i,j))) { ok = false; break; }

  if (!ok) { P.SetSize(ncoef, dof); P = 0.0; } // conservative fallback
}

// Compute element average Ubar by quadrature (reusing helper)
static void ElementAverageQuad(const mfem::FiniteElement &fe, mfem::ElementTransformation &Tr,
                               const mfem::DenseMatrix &elfun_mat, int num_eq, int qord,
                               mfem::Vector &Ubar_out)
{
  const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr.GetGeometryType(), qord);
  mfem::Vector shape(fe.GetDof()), Upt(num_eq);
  Ubar_out.SetSize(num_eq); Ubar_out = 0.0;
  double vol = 0.0;
  for (int i = 0; i < ir.GetNPoints(); i++) {
    const mfem::IntegrationPoint &ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    fe.CalcShape(ip, shape);
    elfun_mat.MultTranspose(shape, Upt);
    const double w = ip.weight * Tr.Weight();
    Ubar_out.Add(w, Upt); vol += w;
  }
  Ubar_out /= vol;
}

// Face quadrature points in reference space for each face of the quad
static void FaceRefQuadrature(int p, std::array<std::vector<mfem::IntegrationPoint>,4> &fqs)
{
  const int qord = std::max(2, p+2);
  const mfem::IntegrationRule &ir1d = mfem::IntRules.Get(mfem::Geometry::SEGMENT, qord);
  fqs.fill({});
  // face 0: xi=-1, eta in [-1,1]
  for (int i=0;i<ir1d.GetNPoints();++i){ mfem::IntegrationPoint ip; ip.x=-1.0; ip.y=ir1d.IntPoint(i).x; ip.weight=ir1d.IntPoint(i).weight; fqs[0].push_back(ip); }
  // face 1: xi=+1
  for (int i=0;i<ir1d.GetNPoints();++i){ mfem::IntegrationPoint ip; ip.x=+1.0; ip.y=ir1d.IntPoint(i).x; ip.weight=ir1d.IntPoint(i).weight; fqs[1].push_back(ip); }
  // face 2: eta=-1
  for (int i=0;i<ir1d.GetNPoints();++i){ mfem::IntegrationPoint ip; ip.y=-1.0; ip.x=ir1d.IntPoint(i).x; ip.weight=ir1d.IntPoint(i).weight; fqs[2].push_back(ip); }
  // face 3: eta=+1
  for (int i=0;i<ir1d.GetNPoints();++i){ mfem::IntegrationPoint ip; ip.y=+1.0; ip.x=ir1d.IntPoint(i).x; ip.weight=ir1d.IntPoint(i).weight; fqs[3].push_back(ip); }
}

} // namespace

namespace mfem
{

void ApplyKrivodonovaHierarchicalLimiter(mfem::ParFiniteElementSpace &vfes,
                                         mfem::ParGridFunction &U,
                                         const KrivoOptions &opt)
{
  MFEM_VERIFY(opt.dim == 2, "Krivodonova limiter implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES, "Assumes byNODES.");
  // Only quads for now
  for (int i = 0; i < vfes.GetNE(); ++i) {
    MFEM_VERIFY(vfes.GetFE(i)->GetGeomType() == mfem::Geometry::SQUARE,
                "This implementation currently supports quadrilateral elements.");
  }

  mfem::Mesh *pm = vfes.GetParMesh();
  mfem::Mesh &mesh = pm ? static_cast<mfem::Mesh&>(*pm) : *vfes.GetMesh();

  const int NE = vfes.GetNE();
  const int num_eq = opt.dim + 2; // 4
  const int p = vfes.GetFE(0)->GetOrder();
  const int qord_avg = 2*p + 2;
  const int face_q = (opt.face_qord > 0) ? opt.face_qord : (p + 2);

  // neighbor means for bounds
  mfem::Array<dg_igr::limiters_detail::Nbr4> nbr; dg_igr::limiters_detail::Build4Nbrs(mesh, nbr);

  // precompute per-element means
  std::vector<mfem::Vector> Ubar_all(NE, mfem::Vector(num_eq));
  {
    mfem::Array<int> vdofs; mfem::Vector elvec; mfem::DenseMatrix elmat;
    for (int e=0; e<NE; ++e)
    {
      const mfem::FiniteElement *fe = vfes.GetFE(e);
      mfem::ElementTransformation *Tr = vfes.GetElementTransformation(e);
      const int dof = fe->GetDof();
      vfes.GetElementVDofs(e, vdofs);
      U.GetSubVector(vdofs, elvec);
      elmat.UseExternalData(elvec.GetData(), dof, num_eq);
      ElementAverageQuad(*fe, *Tr, elmat, num_eq, qord_avg, /*out*/Ubar_all[e]);
    }
  }

  // face quadrature on reference
  std::array<std::vector<mfem::IntegrationPoint>,4> face_qpts;
  FaceRefQuadrature(p, face_qpts);

  // monomial index and projection matrices (cached per order)
  std::vector<std::pair<int,int>> ab;
  mfem::DenseMatrix V, P; // V: dof x ncoef, P: ncoef x dof

  // main loop
  mfem::Array<int> vdofs; mfem::Vector elvec; mfem::DenseMatrix elmat;
  mfem::Vector row(num_eq);

  for (int e=0; e<NE; ++e)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(e);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(e, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // Build/calc V,P once for this FE
    BuildVandermondeAndProjector(*fe, p, V, P, ab);
    const int ncoef = P.Height();

    // Split coefficient indices by total degree
    std::vector<int> idx_low, idx_high;
    idx_low.reserve(ncoef); idx_high.reserve(ncoef);
    for (int k=0; k<ncoef; ++k) {
      const int a = ab[k].first, b = ab[k].second;
      if (a + b < p) idx_low.push_back(k); else idx_high.push_back(k);
    }

    // Form coefficients per component by LSQ projection: c = P * values
    // values_j = elmat(j, comp)
    mfem::DenseMatrix c_all(ncoef, num_eq);
    for (int comp=0; comp<num_eq; ++comp) {
      // column vector of nodal values for this component
      mfem::Vector vals(dof);
      for (int j=0;j<dof;++j){ vals(j) = elmat(j, comp); }
      mfem::Vector c_comp(ncoef); P.Mult(vals, c_comp);
      for (int k=0;k<ncoef;++k) c_all(k,comp) = c_comp(k);
    }

    // Build low- and high-degree reconstructions at (xi,eta): u = phi_low·c_low + phi_high·c_high
    auto eval_poly = [&](double xi, double eta, int comp, const mfem::DenseMatrix &cMat, double alpha)->double
    {
      double u = 0.0;
      // low degrees (unchanged)
      for (int kk : idx_low) { const int a=ab[kk].first, b=ab[kk].second; u += cMat(kk,comp) * Mono(a,b,xi,eta); }
      // scaled high-degree block
      for (int kk : idx_high){ const int a=ab[kk].first, b=ab[kk].second; u += alpha * cMat(kk,comp) * Mono(a,b,xi,eta); }
      return u;
    };

    // Neighbor bounds per face, per component (use neighbor cell means; inflate)
    auto bounds_for_face = [&](int face, int comp, double &umin, double &umax)
    {
      const mfem::Vector &Ui = Ubar_all[e];
      const auto pick = [&](int en)->const mfem::Vector& { return (en>=0)? Ubar_all[en] : Ui; };
      const mfem::Vector *Unbr = nullptr;
      if (face==0) Unbr = &pick(nbr[e].xm);
      if (face==1) Unbr = &pick(nbr[e].xp);
      if (face==2) Unbr = &pick(nbr[e].ym);
      if (face==3) Unbr = &pick(nbr[e].yp);
      const double u0 = Ui(comp);
      const double un = Unbr->operator()(comp);
      // min/max from (cell mean, neighbor mean)
      double lo = std::min(u0, un), hi = std::max(u0, un);
      // inflate the interval a bit to avoid over-clipping in smooth regions
      const double eps = opt.inflate * (hi - lo);
      umin = lo - eps; umax = hi + eps;
    };

    // Determine scaling alpha for high-order block (minimum over faces/points/components)
    double alpha = 1.0;
    for (int face = 0; face < 4; ++face)
    {
      for (const auto &ip : face_qpts[face])
      {
        const double xi = ip.x;
        const double eta = ip.y;
        for (int comp=0; comp<num_eq; ++comp)
        {
          double umin, umax;
          bounds_for_face(face, comp, umin, umax);
          const double ubar = Ubar_all[e](comp);
          const double upred = eval_poly(xi, eta, comp, c_all, /*alpha=*/1.0);

          if (upred < umin)
          {
            const double den = (upred - ubar);
            if (den < 0.0)
            {
              const double a = (umin - ubar) / den;
              alpha = std::min(alpha, a);
            }
          }
          else if (upred > umax)
          {
            const double den = (upred - ubar);
            if (den > 0.0)
            {
              const double a = (umax - ubar) / den;
              alpha = std::min(alpha, a);
            }
          }
        }
      }
    }

    alpha = std::max(0.0, std::min(1.0, alpha));

    if (alpha < 1.0)
    {
      // Reconstruct nodal values from limited coefficients
      for (int j=0; j<dof; ++j)
      {
        const mfem::IntegrationPoint &ip = fe->GetNodes().IntPoint(j);
        const double xi = ip.x;
        const double eta = ip.y;
        for (int comp=0; comp<num_eq; ++comp)
        {
          row(comp) = eval_poly(xi, eta, comp, c_all, alpha);
        }
        elmat.SetRow(j, row);
      }
      U.SetSubVector(vdofs, elmat.GetData());
    }
  }

  if (vfes.GetParMesh()) { U.ExchangeFaceNbrData(); }
}

} // namespace mfem
