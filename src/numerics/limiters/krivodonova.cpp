#include "numerics/limiters/krivodonova.hpp"

#include "limiters_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace {

static void LegendreValues(int p, double x, mfem::Vector &vals)
{
  vals.SetSize(p + 1);
  vals(0) = 1.0;
  if (p >= 1) { vals(1) = x; }
  for (int n = 2; n <= p; ++n)
  {
    vals(n) = ((2.0 * n - 1.0) * x * vals(n - 1) - (n - 1.0) * vals(n - 2)) /
              static_cast<double>(n);
  }
}

static inline double LegendreNorm2D(int i, int j)
{
  // ∫_{-1}^1 P_i^2 dx = 2/(2i+1), so tensor-product norm is 4/((2i+1)(2j+1)).
  return 4.0 / ((2.0 * i + 1.0) * (2.0 * j + 1.0));
}

static void BuildModalCoefficients(const mfem::FiniteElement &fe,
                                   const mfem::DenseMatrix &elfun_mat,
                                   int p,
                                   int num_eq,
                                   const mfem::Vector &Ubar,
                                   mfem::DenseMatrix &c_all)
{
  const int ncoef = (p + 1) * (p + 1);
  const int dof = fe.GetDof();
  c_all.SetSize(ncoef, num_eq);
  c_all = 0.0;

  const int qord = std::max(2, 2 * p + 2);
  const mfem::IntegrationRule &ir =
      mfem::IntRules.Get(fe.GetGeomType(), qord);

  mfem::Vector shape(dof);
  mfem::Vector state(num_eq);
  mfem::Vector Px, Py;

  for (int q = 0; q < ir.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(q);
    fe.CalcShape(ip, shape);
    elfun_mat.MultTranspose(shape, state);

    LegendreValues(p, ip.x, Px);
    LegendreValues(p, ip.y, Py);

    int k = 0;
    for (int i = 0; i <= p; ++i)
    {
      for (int j = 0; j <= p; ++j, ++k)
      {
        const double phi = Px(i) * Py(j);
        for (int comp = 0; comp < num_eq; ++comp)
        {
          const double r = state(comp) - Ubar(comp);
          c_all(k, comp) += ip.weight * r * phi;
        }
      }
    }
  }

  int k = 0;
  for (int i = 0; i <= p; ++i)
  {
    for (int j = 0; j <= p; ++j, ++k)
    {
      const double norm = LegendreNorm2D(i, j);
      for (int comp = 0; comp < num_eq; ++comp)
      {
        c_all(k, comp) /= norm;
      }
    }
  }

  for (int comp = 0; comp < num_eq; ++comp)
  {
    c_all(0, comp) = 0.0;
  }
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
static void FaceRefQuadrature(int qord_in,
                              std::array<std::vector<mfem::IntegrationPoint>,4> &fqs)
{
  const int qord = std::max(2, qord_in);
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
  MFEM_VERIFY(opt.conservative_only,
              "Krivodonova limiter currently supports conservative limiting only.");
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
  FaceRefQuadrature(face_q, face_qpts);

  const int ncoef = (p + 1) * (p + 1);
  std::vector<int> low_modes;
  std::vector<int> high_modes;
  low_modes.reserve(ncoef);
  high_modes.reserve(ncoef);
  {
    int k = 0;
    for (int i = 0; i <= p; ++i)
    {
      for (int j = 0; j <= p; ++j, ++k)
      {
        if (i == 0 && j == 0)
        {
          continue;
        }
        if (std::max(i, j) < p)
        {
          low_modes.push_back(k);
        }
        else
        {
          high_modes.push_back(k);
        }
      }
    }
  }

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

    const mfem::Vector &Ubar = Ubar_all[e];
    mfem::DenseMatrix c_all;
    BuildModalCoefficients(*fe, elmat, p, num_eq, Ubar, c_all);

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
    mfem::Vector Px, Py;
    std::vector<double> phi(ncoef, 0.0);
    for (int face = 0; face < 4; ++face)
    {
      for (const auto &ip : face_qpts[face])
      {
        LegendreValues(p, ip.x, Px);
        LegendreValues(p, ip.y, Py);
        int k = 0;
        for (int i = 0; i <= p; ++i)
        {
          for (int j = 0; j <= p; ++j, ++k)
          {
            phi[k] = Px(i) * Py(j);
          }
        }

        for (int comp=0; comp<num_eq; ++comp)
        {
          double umin, umax;
          bounds_for_face(face, comp, umin, umax);
          const double ubar = Ubar(comp);
          double upred = ubar;
          for (int kk : low_modes)
          {
            upred += c_all(kk, comp) * phi[kk];
          }
          for (int kk : high_modes)
          {
            upred += c_all(kk, comp) * phi[kk];
          }

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
        LegendreValues(p, ip.x, Px);
        LegendreValues(p, ip.y, Py);
        int k = 0;
        for (int i = 0; i <= p; ++i)
        {
          for (int l = 0; l <= p; ++l, ++k)
          {
            phi[k] = Px(i) * Py(l);
          }
        }
        for (int comp=0; comp<num_eq; ++comp)
        {
          double val = Ubar(comp);
          for (int kk : low_modes)
          {
            val += c_all(kk, comp) * phi[kk];
          }
          for (int kk : high_modes)
          {
            val += alpha * c_all(kk, comp) * phi[kk];
          }
          row(comp) = val;
        }
        elmat.SetRow(j, row);
      }
      U.SetSubVector(vdofs, elmat.GetData());
    }
  }

  if (vfes.GetParMesh()) { U.ExchangeFaceNbrData(); }
}

} // namespace mfem
