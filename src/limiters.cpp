#include "limiters.hpp"
#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <utility> // std::pair
#include <limits>


// ======================= Common, small utilities =======================

static inline void BlendRow(mfem::Vector &row, const mfem::Vector &ubar, double theta)
{
  for (int k = 0; k < row.Size(); ++k) { row(k) = ubar(k) + theta * (row(k) - ubar(k)); }
}

static inline void Subtract(const mfem::Vector &a, const mfem::Vector &b, mfem::Vector &out)
{
  const int n = a.Size();
  out.SetSize(n);
  for (int i = 0; i < n; ++i) out(i) = a(i) - b(i);
}

static inline double PressureCons(const mfem::Vector &U, int dim, double gamma)
{
  const double rho = std::max(U(0), 1e-16);
  const double E   = U(1 + dim);
  double mom2 = 0.0;
  for (int d = 0; d < dim; ++d) mom2 += U(1 + d) * U(1 + d);
  const double k = 0.5 * mom2 / rho;
  return (gamma - 1.0) * (E - k);
}

// ========================= PressureCoefficient =========================

double PressureCoefficient::Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
{
  MFEM_VERIFY(U, "PressureCoefficient: U pointer not set.");
  T.SetIntPoint(&ip);
  U->GetVectorValue(T, ip, state);   // [rho, m, E]
  const double rho = std::max(state(0), 1e-16);
  const double E   = state(1 + dim);
  double mom2 = 0.0; for (int d = 0; d < dim; ++d) { const double m = state(1+d); mom2 += m*m; }
  const double k = 0.5 * mom2 / rho;
  return (gamma - 1.0) * (E - k);
}

// ===================== Zhang–Shu positivity limiter ====================

static inline double ThetaForPressure(const mfem::Vector &Ubar,
                                      const mfem::Vector &Up,
                                      int dim, double gamma, double p_min,
                                      double theta_max = 1.0, int iters = 12)
{
  mfem::Vector Utest(Ubar.Size());
  Utest = Ubar; Utest.Add(theta_max, Up); Utest.Add(-theta_max, Ubar);
  if (PressureCons(Utest, dim, gamma) >= p_min) { return theta_max; }

  double lo = 0.0, hi = theta_max;
  for (int it = 0; it < iters; ++it)
  {
    const double mid = 0.5*(lo + hi);
    Utest = Ubar; Utest.Add(mid, Up); Utest.Add(-mid, Ubar);
    const double p = PressureCons(Utest, dim, gamma);
    if (p >= p_min) { lo = mid; } else { hi = mid; }
  }
  return lo;
}

static void ElementAverage(const mfem::FiniteElement &fe,
                           mfem::ElementTransformation &Tr,
                           const mfem::DenseMatrix &elfun_mat,
                           int num_eq, int qord,
                           mfem::Vector &Ubar_out)
{
  const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr.GetGeometryType(), qord);
  mfem::Vector shape(fe.GetDof()), Upt(num_eq);
  Ubar_out.SetSize(num_eq); Ubar_out = 0.0;
  double vol = 0.0;
  for (int i = 0; i < ir.GetNPoints(); i++)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    fe.CalcShape(ip, shape);
    elfun_mat.MultTranspose(shape, Upt);
    const double w = ip.weight * Tr.Weight();
    Ubar_out.Add(w, Upt);
    vol += w;
  }
  Ubar_out /= vol;
}

void ApplyZhangShuPositivityLimiter(mfem::ParFiniteElementSpace &vfes,
                                    mfem::ParGridFunction &U,
                                    int dim, double gamma,
                                    double rho_min, double p_min,
                                    int quad_order, int p_bisect_iters)
{
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Positivity limiter assumes byNODES ordering.");

  const int num_eq = dim + 2;
  const int NE = vfes.GetNE();

  mfem::Array<int> vdofs;
  mfem::Vector elvec;                 // element slice (size = dof*num_eq)
  mfem::DenseMatrix elmat;            // (dof × num_eq) view of element DOFs
  mfem::Vector shape, Up(num_eq), Ubar(num_eq), row(num_eq);

  const int qord = (quad_order > 0) ? quad_order
                                    : (2*vfes.GetFE(0)->GetOrder() + 2);

  for (int i = 0; i < NE; i++)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);

    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // --- compute cell average ---
    ElementAverage(*fe, *Tr, elmat, num_eq, qord, Ubar);

    // If the average itself is unphysical, collapse element to constant admissible state
    {
      const double rho_bar = std::max(Ubar(0), rho_min);
      double mom2_bar = 0.0;
      for (int d = 0; d < dim; ++d) mom2_bar += Ubar(1 + d) * Ubar(1 + d);
      const double k_bar = 0.5 * mom2_bar / std::max(rho_bar, 1e-16);
      double E_bar = Ubar(1 + dim);
      double p_bar = (gamma - 1.0) * (E_bar - k_bar);

      if (rho_bar <= rho_min || p_bar <= p_min)
      {
        Ubar(0) = rho_bar;
        E_bar = k_bar + p_min/(gamma - 1.0);
        Ubar(1 + dim) = E_bar;

        for (int j = 0; j < dof; ++j) { elmat.GetRow(j, row); row = Ubar; elmat.SetRow(j, row); }
        U.SetSubVector(vdofs, elmat.GetData());
        continue; // done with this element
      }
    }

    // --- positivity points (use same IR as average) ---
    const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr->GetGeometryType(), qord);
    shape.SetSize(dof);

    // Step 1: density limiting
    double theta_rho = 1.0;
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up);
      const double rho = Up(0);
      if (rho < rho_min)
      {
        const double num = Ubar(0) - rho_min;
        const double den = Ubar(0) - rho;
        if (den > 0.0) { theta_rho = std::min(theta_rho, num / den); }
        else           { theta_rho = 0.0; }
      }
    }
    if (theta_rho < 1.0)
    {
      for (int j = 0; j < dof; ++j) { elmat.GetRow(j, row); BlendRow(row, Ubar, theta_rho); elmat.SetRow(j, row); }
    }

    // Step 2: pressure limiting
    double theta_p = 1.0;
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up); // current point state (after rho limit)
      const double p = PressureCons(Up, dim, gamma);
      if (p < p_min)
      {
        theta_p = std::min(theta_p,
                           ThetaForPressure(Ubar, Up, dim, gamma, p_min,
                                            /*theta_max=*/theta_rho,
                                            p_bisect_iters));
      }
    }
    if (theta_p < 1.0)
    {
      for (int j = 0; j < dof; ++j) { elmat.GetRow(j, row); BlendRow(row, Ubar, theta_p); elmat.SetRow(j, row); }
    }

    U.SetSubVector(vdofs, elmat.GetData());
  }
}

// ================== Shared helpers for Char-TVB and Char-BJ ==================

// Make POD for mfem::Array<T>
struct Nbr4 { int xm, xp, ym, yp; };

static void ComputeElementCenter(mfem::Mesh &mesh, int e, mfem::Vector &c)
{
  mfem::Array<int> verts;
  mesh.GetElementVertices(e, verts);
  c.SetSize(3); c = 0.0;
  for (int vi = 0; vi < verts.Size(); ++vi)
  {
    const double *X = mesh.GetVertex(verts[vi]);
    c(0) += X[0]; c(1) += X[1]; c(2) += (mesh.Dimension() > 2 ? X[2] : 0.0);
  }
  const double inv = 1.0 / std::max(1, verts.Size());
  c *= inv;
}

static void ComputeFaceCenter(mfem::Mesh &mesh, int f, mfem::Vector &c)
{
  mfem::Array<int> verts;
  mesh.GetFaceVertices(f, verts);
  c.SetSize(3); c = 0.0;
  for (int vi = 0; vi < verts.Size(); ++vi)
  {
    const double *X = mesh.GetVertex(verts[vi]);
    c(0) += X[0]; c(1) += X[1]; c(2) += (mesh.Dimension() > 2 ? X[2] : 0.0);
  }
  const double inv = 1.0 / std::max(1, verts.Size());
  c *= inv;
}

static void Build4Nbrs(mfem::Mesh &mesh, mfem::Array<Nbr4> &nbr)
{
  const int NE = mesh.GetNE();
  nbr.SetSize(NE);
  for (int i = 0; i < NE; ++i) { nbr[i].xm = nbr[i].xp = nbr[i].ym = nbr[i].yp = -1; }

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1=-1, e2=-1;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e1 < 0 || e2 < 0) { continue; } // boundary face

    mfem::Vector c1(3), c2(3), fc(3);
    ComputeFaceCenter(mesh, f, fc);
    ComputeElementCenter(mesh, e1, c1);
    ComputeElementCenter(mesh, e2, c2);

    const double dx = fc(0) - c1(0);
    const double dy = fc(1) - c1(1);

    if (std::abs(dx) > std::abs(dy))
    {
      if (dx > 0) { nbr[e1].xp = e2; nbr[e2].xm = e1; }
      else        { nbr[e1].xm = e2; nbr[e2].xp = e1; }
    }
    else
    {
      if (dy > 0) { nbr[e1].yp = e2; nbr[e2].ym = e1; }
      else        { nbr[e1].ym = e2; nbr[e2].yp = e1; }
    }
  }
}

// DOF physical coordinates (use FE nodes).
static void ElementDofXY(const mfem::FiniteElement &fe, mfem::ElementTransformation &Tr,
                         mfem::Array2D<double> &xy)
{
  const mfem::IntegrationRule &nodes = fe.GetNodes();
  const int dof = fe.GetDof();
  xy.SetSize(2, dof);
  for (int j=0; j<dof; ++j)
  {
    const mfem::IntegrationPoint &ip = nodes.IntPoint(j);
    mfem::Vector XP(3); Tr.Transform(ip, XP);
    xy(0,j) = XP(0); xy(1,j) = XP(1);
  }
}

// Slope weights: wx[j] ≈ ∫ ∂x φ_j, wy[j] ≈ ∫ ∂y φ_j  (over the physical element).
static void SlopeWeights(const mfem::FiniteElement &fe, mfem::ElementTransformation &Tr,
                         int qord_slope, mfem::Vector &wx, mfem::Vector &wy, double &area_out)
{
  const int dof = fe.GetDof();
  wx.SetSize(dof); wy.SetSize(dof); wx = 0.0; wy = 0.0;
  area_out = 0.0;

  const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr.GetGeometryType(), qord_slope);
  mfem::DenseMatrix dshape(dof, Tr.GetSpaceDim());
  for (int q=0; q<ir.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(q);
    Tr.SetIntPoint(&ip);
    fe.CalcDShape(ip, dshape);
    const double w = ip.weight * Tr.Weight();
    for (int j=0; j<dof; ++j) { wx(j) += w * dshape(j,0); wy(j) += w * dshape(j,1); }
    area_out += w;
  }
}

// Return isotropic element length h from area (dx=dy=h); simple and robust.
static inline std::pair<double,double>
ElementDXDYFromArea(mfem::ElementTransformation &Tr, int qord)
{
  const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr.GetGeometryType(), qord);
  double area = 0.0;
  for (int q=0; q<ir.GetNPoints(); ++q)
  { const mfem::IntegrationPoint &ip = ir.IntPoint(q); Tr.SetIntPoint(&ip); area += ip.weight * Tr.Weight(); }
  const double h = std::sqrt(std::max(area, 1e-30));
  return std::make_pair(h, h);
}

// Euler eigenvectors (2D) for x/y directions at mean state
static inline void EulerRightEigen2D(const mfem::Vector &Ubar, double c, double H,
                                     mfem::DenseMatrix &R_x, mfem::DenseMatrix &R_y)
{
  const double rho = std::max(Ubar(0),1e-16);
  const double ux = Ubar(1)/rho;
  const double uy = Ubar(2)/rho;
  const double u2 = ux*ux + uy*uy;

  // R_x
  R_x.SetSize(4,4); R_x = 0.0;
  R_x(0,0)=1; R_x(1,0)=ux-c; R_x(2,0)=uy;   R_x(3,0)=H-ux*c;
  R_x(0,1)=1; R_x(1,1)=ux;   R_x(2,1)=uy;   R_x(3,1)=0.5*u2;
  R_x(0,2)=0; R_x(1,2)=0;    R_x(2,2)=1;    R_x(3,2)=uy;
  R_x(0,3)=1; R_x(1,3)=ux+c; R_x(2,3)=uy;   R_x(3,3)=H+ux*c;

  // R_y
  R_y.SetSize(4,4); R_y = 0.0;
  R_y(0,0)=1; R_y(1,0)=ux;   R_y(2,0)=uy-c; R_y(3,0)=H-uy*c;
  R_y(0,1)=1; R_y(1,1)=ux;   R_y(2,1)=uy;   R_y(3,1)=0.5*u2;
  R_y(0,2)=0; R_y(1,2)=1;    R_y(2,2)=0;    R_y(3,2)=ux;
  R_y(0,3)=1; R_y(1,3)=ux;   R_y(2,3)=uy+c; R_y(3,3)=H+uy*c;
}

static inline void EulerLeftEigen2D(const mfem::Vector &Ubar, double c, double H,
                                    mfem::DenseMatrix &L_x, mfem::DenseMatrix &L_y)
{
  const double rho = std::max(Ubar(0),1e-16);
  const double ux = Ubar(1)/rho;
  const double uy = Ubar(2)/rho;
  const double u2 = ux*ux + uy*uy;
  const double q = 0.5*u2;
  const double h = H - q;

  L_x.SetSize(4,4); L_x = 0.0;
  L_x(0,0)=0.5*(q/h + ux/c); L_x(0,1)=-0.5*(ux/h + 1./c); L_x(0,2)=-0.5*(uy/h); L_x(0,3)=0.5*(1./h);
  L_x(1,0)=1.-q/h;           L_x(1,1)= ux/h;               L_x(1,2)= uy/h;      L_x(1,3)=-1./h;
  L_x(2,0)=-uy;              L_x(2,1)= 0.;                 L_x(2,2)= 1.;        L_x(2,3)= 0.;
  L_x(3,0)=0.5*(q/h - ux/c); L_x(3,1)=-0.5*(ux/h - 1./c);  L_x(3,2)=-0.5*(uy/h);L_x(3,3)=0.5*(1./h);

  L_y.SetSize(4,4); L_y = 0.0;
  L_y(0,0)=0.5*(q/h + uy/c); L_y(0,1)=-0.5*(ux/h);         L_y(0,2)=-0.5*(uy/h + 1./c); L_y(0,3)=0.5*(1./h);
  L_y(1,0)=1.-q/h;           L_y(1,1)= ux/h;               L_y(1,2)= uy/h;              L_y(1,3)=-1./h;
  L_y(2,0)=-ux;              L_y(2,1)= 1.;                 L_y(2,2)= 0.;                L_y(2,3)= 0.;
  L_y(3,0)=0.5*(q/h - uy/c); L_y(3,1)=-0.5*(ux/h);         L_y(3,2)=-0.5*(uy/h - 1./c); L_y(3,3)=0.5*(1./h);
}

// =================== Characteristic TVB limiter ===================

static inline double tvb_minmod(double a, double b, double c, double theta, double t)
{
  const double sa = (a>0)-(a<0), sb=(b>0)-(b<0), sc=(c>0)-(c<0);
  const double aa = std::abs(a), ab=std::abs(b), ac=std::abs(c);
  if (aa <= t) return a;
  if (sa!=0 && sa==sb && sa==sc) return sa*std::min({theta*aa, ab, ac});
  return 0.0;
}

void ApplyCharacteristicTVBLimiter(mfem::ParFiniteElementSpace &vfes,
                                   mfem::ParGridFunction &U,
                                   const CharTVBOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Characteristic TVB limiter currently implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Characteristic TVB limiter assumes byNODES ordering.");

  mfem::Mesh *pm = vfes.GetParMesh();
  mfem::Mesh &mesh = pm ? static_cast<mfem::Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  mfem::Array<int> vdofs;
  mfem::Vector elvec;
  mfem::DenseMatrix elmat;
  mfem::Vector Ubar(num_eq), row(num_eq);
  mfem::Vector wx, wy;
  mfem::Array2D<double> dof_xy;
  mfem::Array<Nbr4> nbr; Build4Nbrs(mesh, nbr);

  std::vector<mfem::Vector> Ubar_all(NE, mfem::Vector(num_eq));
  std::vector<std::pair<double,double>> dxy_all(NE);
  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);
    ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
    dxy_all[i] = ElementDXDYFromArea(*Tr, qord_avg);
  }

  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    const std::pair<double,double> dxy = dxy_all[i];
    const double dx = dxy.first;
    const double dy = dxy.second;

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);
    Ubar = Ubar_all[i];

    double area=0.0;
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    mfem::Vector sDGx(num_eq); sDGx = 0.0;
    mfem::Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, row);
      sDGx.Add(wx(j), row);
      sDGy.Add(wy(j), row);
    }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    auto mean_or_self = [&](int j)->const mfem::Vector& { return (j>=0) ? Ubar_all[j] : Ubar_all[i]; };
    const mfem::Vector &Uxm = mean_or_self(nbr[i].xm);
    const mfem::Vector &Uxp = mean_or_self(nbr[i].xp);
    const mfem::Vector &Uym = mean_or_self(nbr[i].ym);
    const mfem::Vector &Uyp = mean_or_self(nbr[i].yp);

    mfem::Vector dlx(num_eq), drx(num_eq), dly(num_eq), dry(num_eq);
    Subtract(Ubar, Uxm, dlx); dlx /= std::max(dx,1e-16);
    Subtract(Uxp, Ubar, drx); drx /= std::max(dx,1e-16);
    Subtract(Ubar, Uym, dly); dly /= std::max(dy,1e-16);
    Subtract(Uyp, Ubar, dry); dry /= std::max(dy,1e-16);

    const double rho = std::max(Ubar(0), 1e-16);
    const double p   = PressureCons(Ubar, 2, opt.gamma);
    const double c   = std::sqrt(std::max(opt.gamma*p/rho, 0.0));
    const double H   = (Ubar(3) + p)/rho;

    mfem::DenseMatrix R_x(4,4), R_y(4,4), L_x(4,4), L_y(4,4);
    EulerRightEigen2D(Ubar, c, H, R_x, R_y);
    EulerLeftEigen2D (Ubar, c, H, L_x, L_y);

    mfem::Vector A_x(4), B_x(4), C_x(4), A_y(4), B_y(4), C_y(4);
    L_x.Mult(sDGx, A_x);  L_x.Mult(dlx, B_x);  L_x.Mult(drx, C_x);
    L_y.Mult(sDGy, A_y);  L_y.Mult(dly, B_y);  L_y.Mult(dry, C_y);

    mfem::Vector W_x(4), W_y(4);
    const double tx = opt.M_tvb * dx*dx;
    const double ty = opt.M_tvb * dy*dy;
    for (int k=0; k<4; ++k)
    {
      W_x(k) = tvb_minmod(A_x(k), B_x(k), C_x(k), opt.theta, tx);
      W_y(k) = tvb_minmod(A_y(k), B_y(k), C_y(k), opt.theta, ty);
    }

    mfem::Vector sUx(4), sUy(4);
    R_x.Mult(W_x, sUx);
    R_y.Mult(W_y, sUy);

    // Simple KXRCF-like density jump (decide whether to reconstruct)
    auto reljump = [](const mfem::Vector &a, const mfem::Vector &b)->double {
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
      mfem::IntegrationPoint ipc; ipc.Set2(0.5,0.5);
      mfem::Vector Xc(3); Tr->Transform(ipc, Xc);
      mfem::Array2D<double> dof_xy;
      ElementDofXY(*fe, *Tr, dof_xy);

      mfem::Vector rowv(num_eq);
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

// ===================== Face-normal characteristic TVB implementation =====================

// Per-face neighbor info for each element (2D). We keep POD types only.
struct FaceNbrRec
{
  int    nbr;    // neighbor element id, -1 for boundary
  double nx, ny; // unit normal from this element toward the face
  double dn;     // spacing along normal (centroid-to-(nbr|face)) projected on n
};

// Build per-element face records with (nx,ny) and neighbor id.
// Strategy: for each face, get face center 'fc' and the centers of the adjacent
// elements. For e1, normal is n = normalize(fc - c1). For e2 (if interior),
// n = normalize(fc - c2). The spacing dn is |(c_other - c_this)·n| for interior,
// and |(fc - c_this)·n| for boundary.
static void BuildPerElementFaceNormals(mfem::Mesh &mesh,
                                       std::vector<std::vector<FaceNbrRec>> &per_elem)
{
  const int NE = mesh.GetNE();
  per_elem.clear();
  per_elem.resize(NE);

  mfem::Vector c1(3), c2(3), fc(3), tmp(3);

  auto norm2 = [](double x, double y) { return std::sqrt(x*x + y*y); };

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1=-1, e2=-1;
    mesh.GetFaceElements(f, &e1, &e2);

    // centers
    ComputeFaceCenter(mesh, f, fc);
    if (e1 >= 0) ComputeElementCenter(mesh, e1, c1);
    if (e2 >= 0) ComputeElementCenter(mesh, e2, c2);

    // element e1 side
    if (e1 >= 0)
    {
      const double nxe1 = fc(0) - c1(0);
      const double nye1 = fc(1) - c1(1);
      const double len  = std::max(1e-14, norm2(nxe1, nye1));
      FaceNbrRec rec;
      rec.nx  = nxe1 / len;
      rec.ny  = nye1 / len;
      rec.nbr = (e2 >= 0) ? e2 : -1;
      if (e2 >= 0) { // interior: spacing is centroid diff projected on n
        const double dx = c2(0) - c1(0);
        const double dy = c2(1) - c1(1);
        rec.dn = std::abs(dx*rec.nx + dy*rec.ny);
      } else {       // boundary: centroid-to-face projected distance
        const double dx = fc(0) - c1(0);
        const double dy = fc(1) - c1(1);
        rec.dn = std::abs(dx*rec.nx + dy*rec.ny);
      }
      per_elem[e1].push_back(rec);
    }

    // element e2 side
    if (e2 >= 0)
    {
      const double nxe2 = fc(0) - c2(0);
      const double nye2 = fc(1) - c2(1);
      const double len  = std::max(1e-14, norm2(nxe2, nye2));
      FaceNbrRec rec;
      rec.nx  = nxe2 / len;
      rec.ny  = nye2 / len;
      rec.nbr = (e1 >= 0) ? e1 : -1;
      if (e1 >= 0) {
        const double dx = c1(0) - c2(0);
        const double dy = c1(1) - c2(1);
        rec.dn = std::abs(dx*rec.nx + dy*rec.ny);
      } else {
        const double dx = fc(0) - c2(0);
        const double dy = fc(1) - c2(1);
        rec.dn = std::abs(dx*rec.nx + dy*rec.ny);
      }
      per_elem[e2].push_back(rec);
    }
  }

  // Fallback: if an element has no faces in the list (degenerate), leave empty.
}

static inline void EulerLeftRightInDirection(const mfem::Vector &Ubar, double gamma,
                                             double nx, double ny,
                                             mfem::DenseMatrix &L_n, mfem::DenseMatrix &R_n)
{
  // Build L_n ≈ nx L_x + ny L_y and R_n ≈ nx R_x + ny R_y (common approximation).
  const double rho = std::max(Ubar(0), 1e-16);
  const double p   = PressureCons(Ubar, /*dim=*/2, gamma);
  const double c   = std::sqrt(std::max(gamma*p/rho, 0.0));
  const double H   = (Ubar(3) + p)/rho;

  mfem::DenseMatrix Lx(4,4), Ly(4,4), Rx(4,4), Ry(4,4);
  EulerLeftEigen2D (Ubar, c, H, Lx, Ly);
  EulerRightEigen2D(Ubar, c, H, Rx, Ry);

  L_n.SetSize(4,4); R_n.SetSize(4,4);
  for (int i=0;i<4;++i)
    for (int j=0;j<4;++j)
    {
      L_n(i,j) = nx*Lx(i,j) + ny*Ly(i,j);
      R_n(i,j) = nx*Rx(i,j) + ny*Ry(i,j);
    }
}

void ApplyCharacteristicTVBFaceNormal( mfem::ParFiniteElementSpace &vfes,
                                       mfem::ParGridFunction &U,
                                       const CharTVBOptions &opt )
{
  MFEM_VERIFY(opt.dim==2, "Face-normal TVB is implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Face-normal TVB assumes byNODES ordering.");

  mfem::Mesh *pm = vfes.GetParMesh();
  mfem::Mesh &mesh = pm ? static_cast<mfem::Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  // Per-element face normals & neighbors
  std::vector<std::vector<FaceNbrRec>> faces;
  BuildPerElementFaceNormals(mesh, faces);

  // Precompute cell averages and DG slope weights
  std::vector<mfem::Vector> Ubar_all(NE, mfem::Vector(num_eq));

  mfem::Array<int> vdofs;
  mfem::Vector elvec, Ubar(num_eq), row(num_eq);
  mfem::DenseMatrix elmat;
  mfem::Vector wx, wy;
  double area = 0.0;

  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
  }

  // Main loop: compute sDG, limit per face normal, reconstruct
  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    Ubar = Ubar_all[i];

    // Candidate DG slopes via ∫ grad φ_j (vector gradient components)
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    mfem::Vector sDGx(num_eq); sDGx = 0.0;
    mfem::Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, row);
      sDGx.Add(wx(j), row);
      sDGy.Add(wy(j), row);
    }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    // Accumulate limited gradient from face-normal contributions
    mfem::Vector sUx(num_eq); sUx = 0.0;
    mfem::Vector sUy(num_eq); sUy = 0.0;

    for (const FaceNbrRec &fr : faces[i])
    {
      const double nx = fr.nx, ny = fr.ny;
      const double dn = std::max(fr.dn, 1e-14);

      // Project candidate slope onto normal
      mfem::Vector sDGn(num_eq);
      sDGn = 0.0;
      sDGn.Add(nx, sDGx);
      sDGn.Add(ny, sDGy);

      // One-sided mean slopes along normal
      const mfem::Vector &Ui = Ubar_all[i];
      const mfem::Vector &Uo = (fr.nbr >= 0) ? Ubar_all[fr.nbr] : Ubar_all[i];

      mfem::Vector dmn(num_eq), dpn(num_eq);
      Subtract(Ui, Uo, dmn); dmn /= dn;   // "minus" (this - other) / dn
      Subtract(Uo, Ui, dpn); dpn /= dn;   // "plus"  (other - this) / dn

      // Characteristics along n (approximate)
      mfem::DenseMatrix L_n(4,4), R_n(4,4);
      EulerLeftRightInDirection(Ubar, opt.gamma, nx, ny, L_n, R_n);

      mfem::Vector A(4), B(4), C(4);
      L_n.Mult(sDGn, A);
      L_n.Mult(dmn , B);
      L_n.Mult(dpn , C);

      // TVB threshold with directional size
      const double t = opt.M_tvb * dn * dn;

      mfem::Vector W(4);
      for (int k=0; k<4; ++k)
      {
        W(k) = tvb_minmod(A(k), B(k), C(k), opt.theta, t);
      }

      // Back to conserved slope in normal direction, then split to x/y
      mfem::Vector sUn(num_eq);
      R_n.Mult(W, sUn);
      sUx.Add(nx, sUn);
      sUy.Add(ny, sUn);
    }

    // Reconstruct linear polynomial at DOFs
    mfem::IntegrationPoint ipc; ipc.Set2(0.5,0.5);
    mfem::Vector Xc(3); Tr->Transform(ipc, Xc);
    mfem::Array2D<double> dof_xy;
    ElementDofXY(*fe, *Tr, dof_xy);

    mfem::Vector rowv(num_eq);
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

// =================== Characteristic Barth–Jespersen limiter ===================

static inline double clamp01(double a) { return (a <= 0.0) ? 0.0 : (a >= 1.0 ? 1.0 : a); }

// Compute theta for one face point (BJ rule) for a single scalar (characteristic) variable.
static inline double BJThetaOnePoint(double wbar, double wpred,
                                     double wmin, double wmax)
{
  if (wpred > wmax)
  {
    const double den = (wpred - wbar);
    if (den <= 0) return 0.0;
    return std::max(0.0, (wmax - wbar) / den);
  }
  if (wpred < wmin)
  {
    const double den = (wpred - wbar);
    if (den >= 0) return 0.0;
    return std::max(0.0, (wmin - wbar) / den);
  }
  return 1.0; // within bounds, no restriction
}

void ApplyCharacteristicBJLimiter(mfem::ParFiniteElementSpace &vfes,
                                  mfem::ParGridFunction &U,
                                  const CharBJOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Characteristic BJ limiter currently implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Characteristic BJ limiter assumes byNODES ordering.");

  mfem::Mesh *pm = vfes.GetParMesh();
  mfem::Mesh &mesh = pm ? static_cast<mfem::Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  mfem::Array<Nbr4> nbr; Build4Nbrs(mesh, nbr);

  // Precompute means and "sizes"
  std::vector<mfem::Vector> Ubar_all(NE, mfem::Vector(num_eq));
  std::vector<std::pair<double,double>> dxy_all(NE);

  {
    mfem::Array<int> vdofs; mfem::Vector elvec; mfem::DenseMatrix elmat; mfem::Vector Ubar(num_eq);
    for (int i=0; i<NE; ++i)
    {
      const mfem::FiniteElement *fe = vfes.GetFE(i);
      mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
      const int dof = fe->GetDof();
      vfes.GetElementVDofs(i, vdofs);
      U.GetSubVector(vdofs, elvec);
      elmat.UseExternalData(elvec.GetData(), dof, num_eq);
      ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
      Ubar_all[i] = Ubar;
      dxy_all[i] = ElementDXDYFromArea(*Tr, qord_avg);
    }
  }

  // Main per-element loop
  mfem::Array<int> vdofs; mfem::Vector elvec; mfem::DenseMatrix elmat;
  mfem::Vector wx, wy, row(num_eq), Ubar(num_eq);

  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);

    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    Ubar = Ubar_all[i];
    const std::pair<double,double> dxy = dxy_all[i];
    const double dx = dxy.first;
    const double dy = dxy.second;

    // Candidate DG slopes from ∫ grad φ_j
    double area=0.0;
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    mfem::Vector sDGx(num_eq); sDGx = 0.0;
    mfem::Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, row);
      sDGx.Add(wx(j), row);
      sDGy.Add(wy(j), row);
    }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    // Neighbors (fallback to self on boundary)
    auto mean_or_self = [&](int j)->const mfem::Vector& { return (j>=0) ? Ubar_all[j] : Ubar_all[i]; };
    const mfem::Vector &Uxm = mean_or_self(nbr[i].xm);
    const mfem::Vector &Uxp = mean_or_self(nbr[i].xp);
    const mfem::Vector &Uym = mean_or_self(nbr[i].ym);
    const mfem::Vector &Uyp = mean_or_self(nbr[i].yp);

    // Characteristic transforms at the mean state
    const double rho = std::max(Ubar(0), 1e-16);
    const double p   = PressureCons(Ubar, 2, opt.gamma);
    const double c   = std::sqrt(std::max(opt.gamma*p/rho, 0.0));
    const double H   = (Ubar(3) + p)/rho;

    mfem::DenseMatrix R_x(4,4), R_y(4,4), L_x(4,4), L_y(4,4);
    EulerRightEigen2D(Ubar, c, H, R_x, R_y);
    EulerLeftEigen2D (Ubar, c, H, L_x, L_y);

    // Project means and slopes to characteristics
    mfem::Vector wbar_x(4), wbar_y(4); L_x.Mult(Ubar, wbar_x); L_y.Mult(Ubar, wbar_y);

    mfem::Vector wxm(4), wxp(4), wym(4), wyp(4);
    L_x.Mult(Uxm, wxm); L_x.Mult(Uxp, wxp);
    L_y.Mult(Uym, wym); L_y.Mult(Uyp, wyp);

    mfem::Vector W_x(4), W_y(4);
    L_x.Mult(sDGx, W_x);
    L_y.Mult(sDGy, W_y);

    // BJ: compute theta per wave from two face centers per direction
    mfem::Vector theta_x(4); theta_x = 1.0;
    mfem::Vector theta_y(4); theta_y = 1.0;

    // face-center offsets (in physical coords)
    const double hx2 = 0.5*dx;
    const double hy2 = 0.5*dy;

    // X-direction
    if (opt.limit_x)
    {
      for (int k=0; k<4; ++k)
      {
        double th = 1.0;
        // left face
        {
          const double wmin = std::min(wbar_x(k), wxm(k));
          const double wmax = std::max(wbar_x(k), wxm(k));
          const double wpred = wbar_x(k) + W_x(k)*(-hx2); // dy=0 at face center
          th = std::min(th, BJThetaOnePoint(wbar_x(k), wpred, wmin, wmax));
        }
        // right face
        {
          const double wmin = std::min(wbar_x(k), wxp(k));
          const double wmax = std::max(wbar_x(k), wxp(k));
          const double wpred = wbar_x(k) + W_x(k)*(+hx2);
          th = std::min(th, BJThetaOnePoint(wbar_x(k), wpred, wmin, wmax));
        }
        theta_x(k) = clamp01(th);
      }
      for (int k=0; k<4; ++k) { W_x(k) *= theta_x(k); }
    }

    // Y-direction
    if (opt.limit_y)
    {
      for (int k=0; k<4; ++k)
      {
        double th = 1.0;
        // bottom face
        {
          const double wmin = std::min(wbar_y(k), wym(k));
          const double wmax = std::max(wbar_y(k), wym(k));
          const double wpred = wbar_y(k) + W_y(k)*(-hy2);
          th = std::min(th, BJThetaOnePoint(wbar_y(k), wpred, wmin, wmax));
        }
        // top face
        {
          const double wmin = std::min(wbar_y(k), wyp(k));
          const double wmax = std::max(wbar_y(k), wyp(k));
          const double wpred = wbar_y(k) + W_y(k)*(+hy2);
          th = std::min(th, BJThetaOnePoint(wbar_y(k), wpred, wmin, wmax));
        }
        theta_y(k) = clamp01(th);
      }
      for (int k=0; k<4; ++k) { W_y(k) *= theta_y(k); }
    }

    // Back to conserved slopes
    mfem::Vector sUx(4), sUy(4);
    R_x.Mult(W_x, sUx);
    R_y.Mult(W_y, sUy);

    // Overwrite DOFs with the linear reconstruction Ū + sUx*dxj + sUy*dyj
    mfem::IntegrationPoint ipc; ipc.Set2(0.5,0.5);
    mfem::Vector Xc(3); Tr->Transform(ipc, Xc);
    mfem::Array2D<double> dof_xy;
    ElementDofXY(*fe, *Tr, dof_xy);

    mfem::Vector rowv(num_eq);
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

// ===================== Face-normal characteristic BJ limiter =====================

// Small helper: rotate (m_x, m_y) to normal/tangent (m_n, m_t) using n=(nx,ny), t=(-ny,nx)
static inline void RotateMomToNT(double mx, double my, double nx, double ny,
                                 double &mn, double &mt)
{
  const double tx = -ny, ty = nx;      // right-handed tangent
  mn =  nx*mx + ny*my;
  mt =  tx*mx + ty*my;
}

static inline void RotateMomFromNT(double mn, double mt, double nx, double ny,
                                   double &mx, double &my)
{
  const double tx = -ny, ty = nx;
  mx = nx*mn + tx*mt;
  my = ny*mn + ty*mt;
}

// Build (approximate) outward normal for element 'ei' on face 'f' as
// the (face_center - elem_center) direction. Also return face center.
struct FaceInfo { int face=-1, nbr=-1; double nx=0, ny=0, fx=0, fy=0; };

static void BuildFaceInfosPerElem(mfem::Mesh &mesh,
                                  std::vector<mfem::Vector> &xc_all,
                                  std::vector<std::vector<FaceInfo>> &faces_of_elem)
{
  const int NE   = mesh.GetNE();
  const int NFac = mesh.GetNumFaces();

  faces_of_elem.assign(NE, std::vector<FaceInfo>());

  // Precompute element centers if not given
  if ((int)xc_all.size() != NE) { xc_all.assign(NE, mfem::Vector(3)); }
  for (int i = 0; i < NE; ++i) {
    mfem::Array<int> v; mesh.GetElementVertices(i, v);
    mfem::Vector c(3); c = 0.0;
    for (int k=0; k<v.Size(); ++k) { const double *X = mesh.GetVertex(v[k]); c(0)+=X[0]; c(1)+=X[1]; c(2)+=(mesh.Dimension()>2?X[2]:0.0); }
    c /= std::max(1, v.Size());
    xc_all[i] = c;
  }

  // Face loop: register the face once for each of its adjacent elements
  for (int f = 0; f < NFac; ++f)
  {
    int e1=-1, e2=-1; mesh.GetFaceElements(f, &e1, &e2);
    // face center
    mfem::Array<int> fv; mesh.GetFaceVertices(f, fv);
    mfem::Vector fc(3); fc = 0.0;
    for (int k=0; k<fv.Size(); ++k) { const double *X = mesh.GetVertex(fv[k]); fc(0)+=X[0]; fc(1)+=X[1]; fc(2)+=(mesh.Dimension()>2?X[2]:0.0); }
    fc /= std::max(1, fv.Size());

    auto add_for = [&](int ei, int ej)
    {
      if (ei < 0) return;
      FaceInfo info; info.face = f; info.nbr = ej; info.fx = fc(0); info.fy = fc(1);
      const mfem::Vector &ci = xc_all[ei];
      const double vx = fc(0) - ci(0), vy = fc(1) - ci(1);
      const double norm = std::sqrt(vx*vx + vy*vy) + 1e-300;
      info.nx = vx / norm; info.ny = vy / norm;
      faces_of_elem[ei].push_back(info);
    };
    add_for(e1, e2);
    add_for(e2, e1);
  }
}

// Build 1D (normal) Euler eigenvectors for a state rotated so that 'x' is the normal.
// Reuse your existing 2D helpers by passing (u_n,u_t) as (ux,uy) and taking the x-block.
static inline void EulerCharNormalFromRotatedUbar(const mfem::Vector &Ubar_rot,
                                                  double gamma,
                                                  mfem::DenseMatrix &L_n,
                                                  mfem::DenseMatrix &R_n)
{
  // Ubar_rot = [rho, m_n, m_t, E]
  const double rho = std::max(Ubar_rot(0), 1e-16);
  const double ux  = Ubar_rot(1)/rho;   // u_n
  const double uy  = Ubar_rot(2)/rho;   // u_t (acts like 'transverse' velocity)
  const double E   = Ubar_rot(3);
  double mom2 = Ubar_rot(1)*Ubar_rot(1) + Ubar_rot(2)*Ubar_rot(2);
  const double p = (gamma-1.0)*(E - 0.5*mom2/rho);
  const double c = std::sqrt(std::max(gamma*p/rho, 0.0));
  const double H = (E + p)/rho;

  // Use your earlier functions, then use the x-direction blocks (since x ≡ n)
  mfem::DenseMatrix L(4,4), R(4,4), Ltmp(4,4), Rtmp(4,4);
  EulerRightEigen2D(Ubar_rot, c, H, Rtmp, R); // R : y-dir, Rtmp : x-dir
  EulerLeftEigen2D (Ubar_rot, c, H, Ltmp, L); // L : y-dir, Ltmp : x-dir
  R_n = Rtmp;  // x-direction
  L_n = Ltmp;  // x-direction
}

void ApplyFaceNormalCharacteristicBJLimiter(mfem::ParFiniteElementSpace &vfes,
                                            mfem::ParGridFunction &U,
                                            const FaceBJOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Face-normal characteristic BJ implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Limiter assumes byNODES ordering.");

  mfem::Mesh *pm = vfes.GetParMesh();
  mfem::Mesh &mesh = pm ? static_cast<mfem::Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = 4;
  const int NE     = vfes.GetNE();
  const int order  = vfes.GetFE(0)->GetOrder();
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);
  const int qord_avg   = 2*order + 2;

  // Precompute means and element centers
  std::vector<mfem::Vector> Ubar_all(NE, mfem::Vector(num_eq));
  std::vector<mfem::Vector> Xc_all(NE, mfem::Vector(3));
  for (int i=0; i<NE; ++i) {
    mfem::Array<int> v; mesh.GetElementVertices(i, v);
    mfem::Vector c(3); c=0.0;
    for (int k=0; k<v.Size(); ++k) { const double *X = mesh.GetVertex(v[k]); c(0)+=X[0]; c(1)+=X[1]; }
    c /= std::max(1, v.Size()); Xc_all[i] = c;

    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    mfem::Array<int> vdofs; vfes.GetElementVDofs(i, vdofs);
    mfem::Vector elvec; U.GetSubVector(vdofs, elvec);
    mfem::DenseMatrix elmat; elmat.UseExternalData(elvec.GetData(), fe->GetDof(), num_eq);
    mfem::Vector Ubar(num_eq); ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
  }

  // Per-element face info (face center, outward approx normal, neighbor id)
  std::vector<std::vector<FaceInfo>> faces_of_elem;
  BuildFaceInfosPerElem(mesh, Xc_all, faces_of_elem);

  // Work buffers
  mfem::Array<int> vdofs;
  mfem::Vector elvec, row(num_eq), Ubar(num_eq);
  mfem::DenseMatrix elmat;
  mfem::Vector wx, wy; double area=0.0;

  // Main element loop
  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // Mean and center
    Ubar = Ubar_all[i];
    const mfem::Vector &Xci = Xc_all[i];

    // Candidate average slopes via ∫ grad φ_j / area
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    mfem::Vector sDGx(num_eq); sDGx = 0.0;
    mfem::Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0; j<dof; ++j) { elmat.GetRow(j, row); sDGx.Add(wx(j), row); sDGy.Add(wy(j), row); }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    // Compute θ from all faces (min over faces & waves)
    double theta = 1.0;

    for (const FaceInfo &fi : faces_of_elem[i])
    {
      // Face delta from center
      const double dxf = fi.fx - Xci(0);
      const double dyf = fi.fy - Xci(1);

      // Predict state at face center
      mfem::Vector Up(num_eq); Up = Ubar; Up.Add(dxf, sDGx); Up.Add(dyf, sDGy);

      // Rotate Ubar and Up into (n,t) frame
      auto rotU = [&](const mfem::Vector &Uin)->mfem::Vector {
        mfem::Vector Ur(num_eq);
        const double rho = std::max(Uin(0), opt.eps_rho);
        double mn, mt; RotateMomToNT(Uin(1), Uin(2), fi.nx, fi.ny, mn, mt);
        Ur(0)=rho; Ur(1)=mn; Ur(2)=mt; Ur(3)=Uin(3);
        return Ur;
      };
      const mfem::Vector Ubar_r = rotU(Ubar);
      const mfem::Vector Up_r   = rotU(Up);

      // Neighbor mean (or self for boundary)
      const mfem::Vector &Unb = (fi.nbr>=0) ? Ubar_all[fi.nbr] : Ubar_all[i];
      const mfem::Vector Unb_r = rotU(Unb);

      // Characteristic left/right in the *normal* (x) direction for rotated state
      mfem::DenseMatrix L_n(4,4), R_n(4,4);
      EulerCharNormalFromRotatedUbar(Ubar_r, opt.gamma, L_n, R_n);

      // Project to char space
      mfem::Vector w_bar(4), w_pred(4), w_nbr(4);
      L_n.Mult(Ubar_r, w_bar);
      L_n.Mult(Up_r,   w_pred);
      L_n.Mult(Unb_r,  w_nbr);

      // Per-wave BJ clamp vs neighbor; compute a face θ_f ∈ [0,1]
      double theta_f = 1.0;
      for (int k=0; k<4; ++k)
      {
        const double ubar = w_bar(k);
        const double up   = w_pred(k);
        const double unb  = w_nbr(k);

        // If prediction does not overshoot, skip
        const double diff = up - ubar;
        if (std::abs(diff) <= 0.0) continue;

        const double umin = std::min(ubar, unb);
        const double umax = std::max(ubar, unb);

        double alpha = 1.0;
        if (up > umax) {
          alpha = (umax - ubar) / (up - ubar);
        } else if (up < umin) {
          alpha = (umin - ubar) / (up - ubar);
        }
        // Guard
        alpha = std::max(0.0, std::min(1.0, alpha));
        theta_f = std::min(theta_f, alpha);
      }

      theta = std::min(theta, theta_f);
      if (theta == 0.0) break;
    }

    // Rescale slopes and reconstruct linear DOF values
    if (theta < 1.0)
    {
      sDGx *= theta; sDGy *= theta;

      // Reconstruct: U_j = Ubar + sDGx*dx + sDGy*dy at DOF points
      mfem::Array2D<double> xy; ElementDofXY(*fe, *Tr, xy);
      for (int j=0; j<dof; ++j)
      {
        const double dxj = xy(0,j) - Xci(0);
        const double dyj = xy(1,j) - Xci(1);
        elmat.GetRow(j, row);
        row = Ubar;
        row.Add(dxj, sDGx);
        row.Add(dyj, sDGy);
        elmat.SetRow(j, row);
      }
      U.SetSubVector(vdofs, elmat.GetData());
    }
    // else unchanged
  }

  if (vfes.GetParMesh()) { U.ExchangeFaceNbrData(); }
}

// ========================== Barth–Jespersen limiter ==========================

void ApplyBarthJespersenLimiter(mfem::ParFiniteElementSpace &vfes,
                                mfem::ParGridFunction &U,
                                const BJOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Barth–Jespersen limiter currently implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Barth–Jespersen limiter assumes byNODES ordering.");

  mfem::Mesh *pm = vfes.GetParMesh();
  mfem::Mesh &mesh = pm ? static_cast<mfem::Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4 for Euler 2D
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = (opt.qord_avg  >0) ? opt.qord_avg  : (2*order + 2);
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord: (2*order + 2);

  // neighbor map and working arrays
  mfem::Array<Nbr4> nbr; Build4Nbrs(mesh, nbr);

  mfem::Array<int> vdofs;
  mfem::Vector elvec;
  mfem::DenseMatrix elmat;                 // (dof × num_eq)
  mfem::Vector Ubar(num_eq), row(num_eq);
  mfem::Vector wx, wy;                     // slope weights
  mfem::Array2D<double> dof_xy;

  // Precompute all cell averages
  std::vector<mfem::Vector> Ubar_all(NE, mfem::Vector(num_eq));
  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);
    ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
  }

  // Helper: bounds from neighbor means
  auto comp_bounds = [&](int i, int m)->std::pair<double,double>
  {
    double rmin = Ubar_all[i](m), rmax = Ubar_all[i](m);
    auto upd = [&](int j){
      if (j>=0) { rmin = std::min(rmin, Ubar_all[j](m)); rmax = std::max(rmax, Ubar_all[j](m)); }
    };
    upd(nbr[i].xm); upd(nbr[i].xp); upd(nbr[i].ym); upd(nbr[i].yp);
    return {rmin, rmax};
  };

  // Main element loop
  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    Ubar = Ubar_all[i];

    // Slope weights and candidate slopes
    double area = 0.0;
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    mfem::Vector sx(num_eq); sx = 0.0;
    mfem::Vector sy(num_eq); sy = 0.0;
    mfem::Vector rowloc(num_eq);
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, rowloc);
      sx.Add(wx(j), rowloc);
      sy.Add(wy(j), rowloc);
    }
    sx /= std::max(area, 1e-30);
    sy /= std::max(area, 1e-30);

    // Element center and DOF coordinates
    mfem::IntegrationPoint ipc; ipc.Set2(0.5,0.5);
    mfem::Vector Xc(3); Tr->Transform(ipc, Xc);
    ElementDofXY(*fe, *Tr, dof_xy);

    // Decide which components to limit
    bool limit_comp[4] = {opt.limit_rho, opt.limit_momentum, opt.limit_momentum, opt.limit_energy};

    // Compute theta per component via BJ formula using FE nodes as eval points
    double theta_c[4] = {1.0,1.0,1.0,1.0};
    for (int m = 0; m < num_eq; ++m)
    {
      if (!limit_comp[m]) { continue; }

      const auto [Umin, Umax] = comp_bounds(i, m);
      const double Ubar_m = Ubar(m);
      double theta_m = 1.0;

      // Evaluate candidate linear polynomial at all nodes
      for (int j=0; j<dof; ++j)
      {
        const double dx = dof_xy(0,j) - Xc(0);
        const double dy = dof_xy(1,j) - Xc(1);
        const double Up = Ubar_m + sx(m)*dx + sy(m)*dy;

        const double num_p = Umax - Ubar_m;
        const double den_p = Up   - Ubar_m;
        if (den_p > 1e-14) { theta_m = std::min(theta_m, num_p / den_p); }

        const double num_m = Ubar_m - Umin;
        const double den_m = Ubar_m - Up;
        if (den_m > 1e-14) { theta_m = std::min(theta_m, num_m / den_m); }
      }

      theta_c[m] = std::max(0.0, std::min(1.0, theta_m));
    }

    // Apply theta to slopes and reconstruct *linear* values at DOFs
    for (int j=0; j<dof; ++j)
    {
      const double dx = dof_xy(0,j) - Xc(0);
      const double dy = dof_xy(1,j) - Xc(1);

      elmat.GetRow(j, row);
      // Start from mean
      row = Ubar;

      // Add limited linear parts per component
      for (int m=0; m<num_eq; ++m)
      {
        const double th = theta_c[m];
        if (th == 0.0) { continue; }
        const double incr = th * ( sx(m)*dx + sy(m)*dy );
        row(m) += incr;
      }
      elmat.SetRow(j, row);
    }

    // Scatter back
    U.SetSubVector(vdofs, elmat.GetData());
  }

  if (vfes.GetParMesh()) { U.ExchangeFaceNbrData(); }
}

// ===================== Venkatakrishnan limiter (face-normal, characteristic) =====================

// Smooth limiter function φ_VK(a,b,ε), where
//   a = predicted increment to face (characteristic component),
//   b = allowed increment (same sign as a; else b = 0),
//   ε = smoothness parameter (per face).
// This form behaves like: φ≈1 when |a| << |b|; φ→0 as |a| >> |b|; smooth at extrema.
static inline double venkat_phi(double a, double b, double eps)
{
  // enforce same-sign allowed; if opposite sign, limit to 0
  if (a*b <= 0.0) return 0.0;

  const double aa = a*a;
  const double bb = b*b;
  const double ab = a*b;
  // Common smooth form used in practice:
  //   φ = (aa + 2ab + eps) / (aa + ab + 2bb + eps)
  const double num = aa + 2.0*ab + eps;
  const double den = aa + ab + 2.0*bb + eps;
  return (den > 0.0) ? std::min(1.0, std::max(0.0, num/den)) : 0.0;
}

// Reuse the face-normal builder from the face-normal TVB path.
// If you don't already have BuildPerElementFaceNormals(...) and FaceNbrRec,
// copy them from your face-normal TVB block (identical).

void ApplyCharacteristicVenkatFaceNormal(mfem::ParFiniteElementSpace &vfes,
                                         mfem::ParGridFunction &U,
                                         const VenkatOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Venkat face-normal limiter implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Venkat face-normal limiter assumes byNODES ordering.");

  mfem::Mesh *pm = vfes.GetParMesh();
  mfem::Mesh &mesh = pm ? static_cast<mfem::Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  // Per-element face normals & neighbors
  std::vector<std::vector<FaceNbrRec>> faces;
  BuildPerElementFaceNormals(mesh, faces);

  // Precompute cell averages
  std::vector<mfem::Vector> Ubar_all(NE, mfem::Vector(num_eq));

  mfem::Array<int> vdofs;
  mfem::Vector elvec, Ubar(num_eq), row(num_eq);
  mfem::DenseMatrix elmat;

  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
  }

  // Main loop
  mfem::Vector wx, wy; double area=0.0;
  for (int i=0; i<NE; ++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    Ubar = Ubar_all[i];

    // Candidate DG slopes via ∫ grad φ_j
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    mfem::Vector sDGx(num_eq); sDGx = 0.0;
    mfem::Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, row);
      sDGx.Add(wx(j), row);
      sDGy.Add(wy(j), row);
    }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    // Accumulate limited gradient from faces; average at the end
    mfem::Vector sUx(num_eq); sUx = 0.0;
    mfem::Vector sUy(num_eq); sUy = 0.0;
    int          nf_used = 0;

    for (const FaceNbrRec &fr : faces[i])
    {
      const double nx = fr.nx, ny = fr.ny;
      const double dn = std::max(fr.dn, 1e-14);

      // Distance from centroid to face center along normal (half spacing)
      const double delta = 0.5 * dn;

      // Project candidate slope onto normal ⇒ predicted increment to face: a_k
      mfem::Vector sDGn(num_eq); sDGn = 0.0;
      sDGn.Add(nx, sDGx);
      sDGn.Add(ny, sDGy);

      // Characteristics along n
      mfem::DenseMatrix L_n(4,4), R_n(4,4);
      EulerLeftRightInDirection(Ubar, opt.gamma, nx, ny, L_n, R_n);

      mfem::Vector A(4);
      L_n.Mult(sDGn, A);        // A ≈ characteristic slope along n
      A *= delta;               // a_k: predicted increment from center to face

      // Allowed increment b_k from neighbor mean along the same char dir:
      const mfem::Vector &Ui = Ubar_all[i];
      const mfem::Vector &Uo = (fr.nbr >= 0) ? Ubar_all[fr.nbr] : Ubar_all[i];

      mfem::Vector dU(num_eq); Subtract(Uo, Ui, dU); // center-to-center jump
      mfem::Vector D(4); L_n.Mult(dU, D);            // project to chars
      D *= 0.5;                                      // face is ~halfway

      // Smoothness parameter ε
      const double eps = opt.eps_kappa * dn*dn*dn;

      // Limit per wave with smooth φ, then map back
      mfem::Vector W(4);
      for (int k=0; k<4; ++k)
      {
        const double a = A(k);
        const double b = (a >= 0.0) ? std::max(0.0, D(k))
                                    : std::min(0.0, D(k));
        const double phi = venkat_phi(a, b, eps);
        W(k) = (delta > 0.0) ? (phi * (A(k) / delta)) : 0.0; // back to slope
      }

      mfem::Vector sUn(num_eq);
      R_n.Mult(W, sUn);

      sUx.Add(nx, sUn);
      sUy.Add(ny, sUn);
      nf_used++;
    }

    if (nf_used > 0) { sUx /= nf_used; sUy /= nf_used; }

    // Reconstruct linear polynomial at DOFs
    mfem::IntegrationPoint ipc; ipc.Set2(0.5,0.5);
    mfem::Vector Xc(3); Tr->Transform(ipc, Xc);
    mfem::Array2D<double> dof_xy;
    ElementDofXY(*fe, *Tr, dof_xy);

    mfem::Vector rowv(num_eq);
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

// ========================== Venkatakrishnan limiter ==========================
void ApplyCharacteristicVenkatXY(mfem::ParFiniteElementSpace &vfes,
                                 mfem::ParGridFunction &U,
                                 const VenkatOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Venkat XY implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Assumes byNODES ordering.");

  mfem::Mesh *pm = vfes.GetParMesh();
  mfem::Mesh &mesh = pm ? static_cast<mfem::Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  mfem::Array<Nbr4> nbr; Build4Nbrs(mesh, nbr);

  // Precompute means and nominal dx,dy
  std::vector<mfem::Vector> Ubar_all(NE, mfem::Vector(num_eq));
  std::vector<std::pair<double,double>> dxy_all(NE);

  mfem::Array<int> vdofs;
  mfem::Vector elvec, Ubar(num_eq), row(num_eq);
  mfem::DenseMatrix elmat;

  for (int i=0;i<NE;++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
    dxy_all[i]  = ElementDXDYFromArea(*Tr, qord_avg);
  }

  // Main loop
  mfem::Vector wx, wy; double area=0.0;
  for (int i=0;i<NE;++i)
  {
    const mfem::FiniteElement *fe = vfes.GetFE(i);
    mfem::ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    const auto dxy = dxy_all[i];
    const double dx = dxy.first, dy = dxy.second;
    const double epsx = opt.eps_kappa * dx*dx*dx;
    const double epsy = opt.eps_kappa * dy*dy*dy;

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    Ubar = Ubar_all[i];

    // Candidate DG slopes
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    mfem::Vector sDGx(num_eq); sDGx = 0.0;
    mfem::Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0;j<dof;++j) { elmat.GetRow(j,row); sDGx.Add(wx(j),row); sDGy.Add(wy(j),row); }
    sDGx /= std::max(area,1e-30);
    sDGy /= std::max(area,1e-30);

    // One-sided mean differences
    auto mean_or_self = [&](int j)->const mfem::Vector& { return (j>=0) ? Ubar_all[j] : Ubar_all[i]; };
    const mfem::Vector &Uxm = mean_or_self(nbr[i].xm);
    const mfem::Vector &Uxp = mean_or_self(nbr[i].xp);
    const mfem::Vector &Uym = mean_or_self(nbr[i].ym);
    const mfem::Vector &Uyp = mean_or_self(nbr[i].yp);

    // Characteristic transforms for x and y axes
    // (use EulerLeft/Right in pure x and y)
    mfem::DenseMatrix Lx(4,4), Rx(4,4), Ly(4,4), Ry(4,4);
    EulerLeftRightInDirection(Ubar, opt.gamma, /*nx=*/1.0, /*ny=*/0.0, Lx, Rx);
    EulerLeftRightInDirection(Ubar, opt.gamma, /*nx=*/0.0, /*ny=*/1.0, Ly, Ry);

    // Project slopes and neighbor jumps to characteristics
    mfem::Vector A_x(4), A_y(4); Lx.Mult(sDGx, A_x); Ly.Mult(sDGy, A_y);
    mfem::Vector Dxp(4), Dxm(4), Dyp(4), Dym(4);
    { mfem::Vector d; d.SetSize(num_eq);
      Subtract(Uxp, Ubar, d); Lx.Mult(d, Dxp);
      Subtract(Ubar, Uxm, d); Lx.Mult(d, Dxm);
      Subtract(Uyp, Ubar, d); Ly.Mult(d, Dyp);
      Subtract(Ubar, Uym, d); Ly.Mult(d, Dym);
    }

    // Predicted increments to faces at ±x: a± = A_x * (±dx/2)
    // Allowed increments b± ≈ D±/2 (center-to-face)
    mfem::Vector W_x(4), W_y(4);
    for (int k=0;k<4;++k)
    {
      const double axp =  A_x(k) * (0.5*dx);
      const double axm = -A_x(k) * (0.5*dx);
      const double bxp =  0.5*Dxp(k);
      const double bxm =  0.5*Dxm(k);
      const double phix = std::min( venkat_phi(axp, bxp, epsx),
                                    venkat_phi(axm, bxm, epsx) );
      W_x(k) = phix * A_x(k); // limited char slope in x

      const double ayp =  A_y(k) * (0.5*dy);
      const double aym = -A_y(k) * (0.5*dy);
      const double byp =  0.5*Dyp(k);
      const double bym =  0.5*Dym(k);
      const double phiy = std::min( venkat_phi(ayp, byp, epsy),
                                    venkat_phi(aym, bym, epsy) );
      W_y(k) = phiy * A_y(k); // limited char slope in y
    }

    // Back to conserved slopes
    mfem::Vector sUx(num_eq), sUy(num_eq);
    Rx.Mult(W_x, sUx);
    Ry.Mult(W_y, sUy);

    // Reconstruct linear polynomial at DOFs
    mfem::IntegrationPoint ipc; ipc.Set2(0.5,0.5);
    mfem::Vector Xc(3); Tr->Transform(ipc, Xc);
    mfem::Array2D<double> dof_xy; ElementDofXY(*fe, *Tr, dof_xy);

    mfem::Vector rowv(num_eq);
    for (int j=0;j<dof;++j)
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

// =================== Krivodonova / BDS hierarchical modal limiter (quads) ===================

// Build monomial index list (a,b) with total degree <= p
static void BuildMonomialAB(int p, std::vector<std::pair<int,int>> &ab)
{
  ab.clear();
  for (int t=0; t<=p; ++t) for (int a=0; a<=t; ++a) { int b = t - a; ab.emplace_back(a,b); }
}

// Evaluate monomial basis at reference point (xi,eta)
static inline double Mono(int a, int b, double xi, double eta)
{ double v=1.0; for(int i=0;i<a;++i) v*=xi; for(int j=0;j<b;++j) v*=eta; return v; }

// Get ref nodes (ξ,η) from FE nodes (works for L2 quads). Returns 2×dof.
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

// Build Vandermonde V(dof × ncoef) and a robust LSQ projector
//   P(ncoef × dof) ≈ (VᵀV + λ I)⁻¹ Vᵀ
// where λ is tiny Tikhonov regularization to avoid catastrophic inverses.
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

  // 4) Robust projector: P = (V^T V + λI)^-1 V^T
  mfem::DenseMatrix VT, G;
  DenseTranspose(V, VT);         // (ncoef × dof)
  mfem::Mult(VT, V, G);          // (ncoef × ncoef)

  // Tikhonov regularization λ = eps * trace(G) / ncoef
  double tr = 0.0; for (int i = 0; i < G.Height(); ++i) tr += G(i,i);
  const double lam = std::max(1e-14, 1e-14 * tr / std::max(1, G.Height()));
  for (int i = 0; i < G.Height(); ++i) { G(i,i) += lam; }

  // Solve G * X = VT  => P = X
  // Use MFEM's DenseMatrixInverse (LU+pivoting) instead of CalcInverse
  mfem::DenseMatrixInverse inv(G);
  inv.Factor();                  // LU with pivoting
  inv.Mult(VT, P);               // P: (ncoef × dof)

  // Guard: if anything bad slipped in, fall back to zero projector (no limiting)
  auto sane = [](double v){ return std::isfinite(v); };
  bool ok = true;
  for (int i = 0; i < P.Height() && ok; ++i)
    for (int j = 0; j < P.Width(); ++j)
      if (!sane(P(i,j))) { ok = false; break; }

  if (!ok) { P.SetSize(ncoef, dof); P = 0.0; } // conservative fallback
}

// Compute element average Ū by quadrature (reusing your existing helper)
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
  // face 0: xi=-1, eta ∈ [-1,1]
  for (int i=0;i<ir1d.GetNPoints();++i){ mfem::IntegrationPoint ip; ip.x=-1.0; ip.y=ir1d.IntPoint(i).x; ip.weight=ir1d.IntPoint(i).weight; fqs[0].push_back(ip); }
  // face 1: xi=+1
  for (int i=0;i<ir1d.GetNPoints();++i){ mfem::IntegrationPoint ip; ip.x=+1.0; ip.y=ir1d.IntPoint(i).x; ip.weight=ir1d.IntPoint(i).weight; fqs[1].push_back(ip); }
  // face 2: eta=-1
  for (int i=0;i<ir1d.GetNPoints();++i){ mfem::IntegrationPoint ip; ip.y=-1.0; ip.x=ir1d.IntPoint(i).x; ip.weight=ir1d.IntPoint(i).weight; fqs[2].push_back(ip); }
  // face 3: eta=+1
  for (int i=0;i<ir1d.GetNPoints();++i){ mfem::IntegrationPoint ip; ip.y=+1.0; ip.x=ir1d.IntPoint(i).x; ip.weight=ir1d.IntPoint(i).weight; fqs[3].push_back(ip); }
}

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
  mfem::Array<Nbr4> nbr; Build4Nbrs(mesh, nbr);

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
  mfem::DenseMatrix V, P; // V: dof×ncoef, P: ncoef×dof

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

    // Build low- and high-degree reconstructions at (xi,eta): u = φ_low·c_low + φ_high·c_high
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

    // Find largest alpha in [0,1] keeping every face-qp within neighbor bounds
    double alpha = 1.0;
    for (int face=0; face<4; ++face)
    {
      for (int comp=0; comp<num_eq; ++comp)
      {
        bool has_high = false;
        for (int kk : idx_high) if (std::abs(c_all(kk,comp)) > 0) { has_high = true; break; }
        if (!has_high) continue;

        double umin, umax; bounds_for_face(face, comp, umin, umax);

        for (const auto &ip : face_qpts[face])
        {
          const double u_low = eval_poly(ip.x, ip.y, comp, c_all, /*alpha=*/0.0);
          const double u_hi  = eval_poly(ip.x, ip.y, comp, c_all, /*alpha=*/1.0);
          const double du    = u_hi - u_low;

          // If high-order block does nothing at this point, skip
          if (std::abs(du) < 1e-30) { continue; }

          // We need u(α) = u_low + α du ∈ [umin, umax]
          // => α ∈ [(umin - u_low)/du, (umax - u_low)/du]
          double a1 = (umin - u_low) / du;
          double a2 = (umax - u_low) / du;

          // normalize to [a_min, a_max] with a_min <= a_max
          double a_min = std::min(a1, a2);
          double a_max = std::max(a1, a2);

          // intersect with feasible [0,1]: require alpha ≤ a_max and ≥ a_min
          // we want largest feasible alpha (be as non-dissipative as possible)
          // So the active constraint is the upper end a_max.
          alpha = std::min(alpha, a_max);

          if (alpha <= std::max(0.0, a_min)) { alpha = 0.0; break; }
        }
        if (alpha <= 0.0) break;
      }
      if (alpha <= 0.0) break;
    }
    // alpha = std::max(0.0, std::min(1.0, alpha));

    // If no clipping, continue
    alpha = std::max(0.0, std::min(1.0, alpha));
    if (alpha >= 0.999999) { continue; }

    // Rebuild nodal values with scaled high block and scatter back
    // For each node j, u_j(comp) = sum_k φ_k(xj) * c_low + alpha * sum_k φ_k(xj) * c_high
    // That is just evaluating eval_poly at the FE nodes
    mfem::Array2D<double> ref_nodes; RefDofXY(*fe, ref_nodes);
    for (int j=0; j<dof; ++j)
    {
      const double xi = ref_nodes(0,j), eta = ref_nodes(1,j);
      for (int comp=0; comp<num_eq; ++comp)
      {
        const double u = eval_poly(xi, eta, comp, c_all, alpha);
        elmat(j, comp) = u;
      }
    }
    U.SetSubVector(vdofs, elmat.GetData());
  }

  if (vfes.GetParMesh()) { U.ExchangeFaceNbrData(); }
}
