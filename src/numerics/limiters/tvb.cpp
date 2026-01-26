#include "numerics/limiters/tvb.hpp"

#include "limiters_common.hpp"
#include "physics/euler/eos.hpp"
#include "physics/euler/characteristics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

static inline double tvb_minmod(double a, double b, double c, double theta, double t)
{
  const double sa = (a>0)-(a<0), sb=(b>0)-(b<0), sc=(c>0)-(c<0);
  const double aa = std::abs(a), ab=std::abs(b), ac=std::abs(c);
  if (aa <= t) return a;
  if (sa!=0 && sa==sb && sa==sc) return sa*std::min({theta*aa, ab, ac});
  return 0.0;
}

} // namespace

namespace mfem
{

void ApplyCharacteristicTVBLimiter(ParFiniteElementSpace &vfes,
                                   ParGridFunction &U,
                                   const CharTVBOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Characteristic TVB limiter currently implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Characteristic TVB limiter assumes byNODES ordering.");

  Mesh *pm = vfes.GetParMesh();
  Mesh &mesh = pm ? static_cast<Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  Array<int> vdofs;
  Vector elvec;
  DenseMatrix elmat;
  Vector Ubar(num_eq), row(num_eq);
  Vector wx, wy;
  Array2D<double> dof_xy;
  Array<dg_igr::limiters_detail::Nbr4> nbr; dg_igr::limiters_detail::Build4Nbrs(mesh, nbr);

  std::vector<Vector> Ubar_all(NE, Vector(num_eq));
  std::vector<std::pair<double,double>> dxy_all(NE);
  for (int i=0; i<NE; ++i)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);
    dg_igr::limiters_detail::ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
    dxy_all[i] = dg_igr::limiters_detail::ElementDXDYFromArea(*Tr, qord_avg);
  }

  for (int i=0; i<NE; ++i)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    const std::pair<double,double> dxy = dxy_all[i];
    const double dx = dxy.first;
    const double dy = dxy.second;

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);
    Ubar = Ubar_all[i];

    double area=0.0;
    dg_igr::limiters_detail::SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    Vector sDGx(num_eq); sDGx = 0.0;
    Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, row);
      sDGx.Add(wx(j), row);
      sDGy.Add(wy(j), row);
    }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    auto mean_or_self = [&](int j)->const Vector& { return (j>=0) ? Ubar_all[j] : Ubar_all[i]; };
    const Vector &Uxm = mean_or_self(nbr[i].xm);
    const Vector &Uxp = mean_or_self(nbr[i].xp);
    const Vector &Uym = mean_or_self(nbr[i].ym);
    const Vector &Uyp = mean_or_self(nbr[i].yp);

    Vector dlx(num_eq), drx(num_eq), dly(num_eq), dry(num_eq);
    dg_igr::limiters_detail::Subtract(Ubar, Uxm, dlx); dlx /= std::max(dx,1e-16);
    dg_igr::limiters_detail::Subtract(Uxp, Ubar, drx); drx /= std::max(dx,1e-16);
    dg_igr::limiters_detail::Subtract(Ubar, Uym, dly); dly /= std::max(dy,1e-16);
    dg_igr::limiters_detail::Subtract(Uyp, Ubar, dry); dry /= std::max(dy,1e-16);

    const double rho = std::max(Ubar(0), 1e-16);
    const double p   = PressureFromCons(Ubar, 2, opt.gamma, /*rho_floor=*/1e-16);
    const double c   = std::sqrt(std::max(opt.gamma*p/rho, 0.0));
    const double H   = (Ubar(3) + p)/rho;

    DenseMatrix R_x(4,4), R_y(4,4), L_x(4,4), L_y(4,4);
    EulerRightEigen2D(Ubar, c, H, R_x, R_y);
    EulerLeftEigen2D (Ubar, c, H, L_x, L_y);

    Vector A_x(4), B_x(4), C_x(4), A_y(4), B_y(4), C_y(4);
    L_x.Mult(sDGx, A_x);  L_x.Mult(dlx, B_x);  L_x.Mult(drx, C_x);
    L_y.Mult(sDGy, A_y);  L_y.Mult(dly, B_y);  L_y.Mult(dry, C_y);

    Vector W_x(4), W_y(4);
    const double tx = opt.M_tvb * dx*dx;
    const double ty = opt.M_tvb * dy*dy;
    for (int k=0; k<4; ++k)
    {
      W_x(k) = tvb_minmod(A_x(k), B_x(k), C_x(k), opt.theta, tx);
      W_y(k) = tvb_minmod(A_y(k), B_y(k), C_y(k), opt.theta, ty);
    }

    Vector sUx(4), sUy(4);
    R_x.Mult(W_x, sUx);
    R_y.Mult(W_y, sUy);

    // Simple KXRCF-like density jump (decide whether to reconstruct)
    auto reljump = [](const Vector &a, const Vector &b)->double {
      const double ar = std::max(a(0), 1e-14);
      return std::abs(a(0)-b(0))/ar;
    };
    double Ji = 0.0;
    Ji = std::max(Ji, reljump(Ubar, Uxm));
    Ji = std::max(Ji, reljump(Uxp, Ubar));
    Ji = std::max(Ji, reljump(Ubar, Uym));
    Ji = std::max(Ji, reljump(Uyp, Ubar));

    const bool troubled = (Ji > opt.tau_flag);

    if (troubled)
    {
      IntegrationPoint ipc; ipc.Set2(0.5,0.5);
      Vector Xc(3); Tr->Transform(ipc, Xc);
      Array2D<double> dof_xy;
      dg_igr::limiters_detail::ElementDofXY(*fe, *Tr, dof_xy);

      Vector rowv(num_eq);
      for (int j=0; j<dof; ++j)
      {
        const double dxj = dof_xy(0,j) - Xc(0);
        const double dyj = dof_xy(1,j) - Xc(1);
        elmat.GetRow(j, rowv);
        rowv = Ubar;
        rowv.Add(dxj, sUx);
        rowv.Add(dyj, sUy);
        elmat.SetRow(j, rowv);
      }
      U.SetSubVector(vdofs, elmat.GetData());
    }
  }

  if (vfes.GetParMesh()) { U.ExchangeFaceNbrData(); }
}

void ApplyCharacteristicTVBFaceNormal( ParFiniteElementSpace &vfes,
                                       ParGridFunction &U,
                                       const CharTVBOptions &opt )
{
  MFEM_VERIFY(opt.dim==2, "Face-normal TVB is implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Face-normal TVB assumes byNODES ordering.");

  Mesh *pm = vfes.GetParMesh();
  Mesh &mesh = pm ? static_cast<Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  // Per-element face normals & neighbors
  std::vector<std::vector<dg_igr::limiters_detail::FaceNbrRec>> faces;
  dg_igr::limiters_detail::BuildPerElementFaceNormals(mesh, faces);

  // Precompute cell averages and DG slope weights
  std::vector<Vector> Ubar_all(NE, Vector(num_eq));

  Array<int> vdofs;
  Vector elvec, Ubar(num_eq), row(num_eq);
  DenseMatrix elmat;
  Vector wx, wy;
  double area = 0.0;

  for (int i=0; i<NE; ++i)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    dg_igr::limiters_detail::ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
  }

  // Main loop: compute sDG, limit per face normal, reconstruct
  for (int i=0; i<NE; ++i)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    Ubar = Ubar_all[i];

    // Candidate DG slopes via ∫ grad phi_j (vector gradient components)
    dg_igr::limiters_detail::SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    Vector sDGx(num_eq); sDGx = 0.0;
    Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, row);
      sDGx.Add(wx(j), row);
      sDGy.Add(wy(j), row);
    }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    // Accumulate limited gradient from face-normal contributions
    Vector sUx(num_eq); sUx = 0.0;
    Vector sUy(num_eq); sUy = 0.0;

    for (const dg_igr::limiters_detail::FaceNbrRec &fr : faces[i])
    {
      const double nx = fr.nx, ny = fr.ny;
      const double dn = std::max(fr.dn, 1e-14);

      // Project candidate slope onto normal
      Vector sDGn(num_eq);
      sDGn = 0.0;
      sDGn.Add(nx, sDGx);
      sDGn.Add(ny, sDGy);

      // One-sided mean slopes along normal
      const Vector &Ui = Ubar_all[i];
      const Vector &Uo = (fr.nbr >= 0) ? Ubar_all[fr.nbr] : Ubar_all[i];

      Vector dmn(num_eq), dpn(num_eq);
      dg_igr::limiters_detail::Subtract(Ui, Uo, dmn); dmn /= dn;   // "minus" (this - other) / dn
      dg_igr::limiters_detail::Subtract(Uo, Ui, dpn); dpn /= dn;   // "plus"  (other - this) / dn

      // Characteristics along n (approximate)
      DenseMatrix L_n(4,4), R_n(4,4);
      EulerLeftRightInDirection(Ubar, opt.gamma, nx, ny, L_n, R_n);

      Vector A(4), B(4), C(4);
      L_n.Mult(sDGn, A);
      L_n.Mult(dmn , B);
      L_n.Mult(dpn , C);

      // TVB threshold with directional size
      const double t = opt.M_tvb * dn * dn;

      Vector W(4);
      for (int k=0; k<4; ++k)
      {
        W(k) = tvb_minmod(A(k), B(k), C(k), opt.theta, t);
      }

      // Back to conserved slope in normal direction, then split to x/y
      Vector sUn(num_eq);
      R_n.Mult(W, sUn);
      sUx.Add(nx, sUn);
      sUy.Add(ny, sUn);
    }

    // Reconstruct linear polynomial at DOFs
    IntegrationPoint ipc; ipc.Set2(0.5,0.5);
    Vector Xc(3); Tr->Transform(ipc, Xc);
    Array2D<double> dof_xy;
    dg_igr::limiters_detail::ElementDofXY(*fe, *Tr, dof_xy);

    Vector rowv(num_eq);
    for (int j=0; j<dof; ++j)
    {
      const double dxj = dof_xy(0,j) - Xc(0);
      const double dyj = dof_xy(1,j) - Xc(1);
      elmat.GetRow(j, rowv);
      rowv = Ubar;
      rowv.Add(dxj, sUx);
      rowv.Add(dyj, sUy);
      elmat.SetRow(j, rowv);
    }
    U.SetSubVector(vdofs, elmat.GetData());
  }

  if (vfes.GetParMesh()) { U.ExchangeFaceNbrData(); }
}

} // namespace mfem
