#include "limiters.hpp"
#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <utility> // std::pair

using namespace mfem;

// ======================= Common, small utilities =======================

static inline void BlendRow(Vector &row, const Vector &ubar, double theta)
{
  for (int k = 0; k < row.Size(); ++k) { row(k) = ubar(k) + theta * (row(k) - ubar(k)); }
}

static inline void RowAssign(Vector &row, const Vector &val)
{
  for (int k = 0; k < row.Size(); ++k) row(k) = val(k);
}

static inline void Subtract(const Vector &a, const Vector &b, Vector &out)
{
  const int n = a.Size();
  out.SetSize(n);
  for (int i = 0; i < n; ++i) out(i) = a(i) - b(i);
}

static inline double PressureCons(const Vector &U, int dim, double gamma)
{
  const double rho = U(0);
  const double E   = U(1 + dim);
  double mom2 = 0.0;
  for (int d = 0; d < dim; ++d) mom2 += U(1 + d) * U(1 + d);
  const double k = 0.5 * mom2 / std::max(rho, 1e-16);
  return (gamma - 1.0) * (E - k);
}

// ========================= PressureCoefficient =========================

double PressureCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
  MFEM_VERIFY(U, "PressureCoefficient: U pointer not set.");
  T.SetIntPoint(&ip);
  U->GetVectorValue(T, ip, state);   // [rho, m, E]
  const double rho = state(0);
  const double E   = state(1 + dim);
  double mom2 = 0.0; for (int d = 0; d < dim; ++d) { const double m = state(1+d); mom2 += m*m; }
  const double k = 0.5 * mom2 / std::max(rho, 1e-16);
  return (gamma - 1.0) * (E - k);
}

// ===================== Zhang–Shu positivity limiter ====================

static inline double ThetaForPressure(const Vector &Ubar,
                                      const Vector &Up,
                                      int dim, double gamma, double p_min,
                                      double theta_max = 1.0, int iters = 12)
{
  Vector Utest(Ubar.Size());
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

static void ElementAverage(const FiniteElement &fe,
                           ElementTransformation &Tr,
                           const DenseMatrix &elfun_mat,
                           int num_eq, int qord,
                           Vector &Ubar_out)
{
  const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), qord);
  Vector shape(fe.GetDof()), Upt(num_eq);
  Ubar_out.SetSize(num_eq); Ubar_out = 0.0;
  double vol = 0.0;
  for (int i = 0; i < ir.GetNPoints(); i++)
  {
    const IntegrationPoint &ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    fe.CalcShape(ip, shape);
    elfun_mat.MultTranspose(shape, Upt);
    const double w = ip.weight * Tr.Weight();
    Ubar_out.Add(w, Upt);
    vol += w;
  }
  Ubar_out /= vol;
}

void ApplyZhangShuPositivityLimiter(ParFiniteElementSpace &vfes,
                                    ParGridFunction &U,
                                    int dim, double gamma,
                                    double rho_min, double p_min,
                                    int quad_order, int p_bisect_iters)
{
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Positivity limiter assumes byNODES ordering.");

  const int num_eq = dim + 2;
  const int NE = vfes.GetNE();

  Array<int> vdofs;
  Vector elvec;                 // element slice (size = dof*num_eq)
  DenseMatrix elmat;            // (dof × num_eq) view of element DOFs
  Vector shape, Up(num_eq), Ubar(num_eq), row(num_eq);

  const int qord = (quad_order > 0) ? quad_order
                                    : (2*vfes.GetFE(0)->GetOrder() + 2);

  for (int i = 0; i < NE; i++)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
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
    const IntegrationRule &ir = IntRules.Get(Tr->GetGeometryType(), qord);
    shape.SetSize(dof);

    // Step 1: density limiting
    double theta_rho = 1.0;
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const IntegrationPoint &ip = ir.IntPoint(q);
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
      const IntegrationPoint &ip = ir.IntPoint(q);
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

// ================== Characteristic TVB + KXRCF helper pieces ==================

static inline void EulerRightEigen2D(const Vector &Ubar, double c, double H,
                                     DenseMatrix &R_x, DenseMatrix &R_y)
{
  const double rho = Ubar(0);
  const double ux = Ubar(1)/std::max(rho,1e-16);
  const double uy = Ubar(2)/std::max(rho,1e-16);
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

static inline void EulerLeftEigen2D(const Vector &Ubar, double c, double H,
                                    DenseMatrix &L_x, DenseMatrix &L_y)
{
  const double rho = Ubar(0);
  const double ux = Ubar(1)/std::max(rho,1e-16);
  const double uy = Ubar(2)/std::max(rho,1e-16);
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
  L_y(1,0)=1.-q/h;           L_y(1,1)= ux/h;               L_y(1,2)= uy/h;            L_y(1,3)=-1./h;
  L_y(2,0)=-ux;              L_y(2,1)= 1.;                 L_y(2,2)= 0.;              L_y(2,3)= 0.;
  L_y(3,0)=0.5*(q/h - uy/c); L_y(3,1)=-0.5*(ux/h);         L_y(3,2)=-0.5*(uy/h - 1./c);L_y(3,3)=0.5*(1./h);
}

static inline double tvb_minmod(double a, double b, double c, double theta, double t)
{
  const double sa = (a>0)-(a<0), sb=(b>0)-(b<0), sc=(c>0)-(c<0);
  const double aa = std::abs(a), ab=std::abs(b), ac=std::abs(c);
  if (aa <= t) return a;
  if (sa!=0 && sa==sb && sa==sc) return sa*std::min({theta*aa, ab, ac});
  return 0.0;
}

// Trivial POD for mfem::Array<T>
struct Nbr4 { int xm, xp, ym, yp; };

static void ComputeElementCenter(Mesh &mesh, int e, Vector &c)
{
  Array<int> verts;
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

static void ComputeFaceCenter(Mesh &mesh, int f, Vector &c)
{
  Array<int> verts;
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

static void Build4Nbrs(Mesh &mesh, Array<Nbr4> &nbr)
{
  const int NE = mesh.GetNE();
  nbr.SetSize(NE);
  // initialize to -1 (boundary)
  for (int i = 0; i < NE; ++i) { nbr[i].xm = nbr[i].xp = nbr[i].ym = nbr[i].yp = -1; }

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1=-1, e2=-1;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e1 < 0 || e2 < 0) { continue; } // boundary face

    Vector c1(3), c2(3), fc(3);
    ComputeFaceCenter(mesh, f, fc);
    ComputeElementCenter(mesh, e1, c1);
    ComputeElementCenter(mesh, e2, c2);

    // direction from e1 center to face center
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
static void ElementDofXY(const FiniteElement &fe, ElementTransformation &Tr,
                         Array2D<double> &xy)
{
  const IntegrationRule &nodes = fe.GetNodes();
  const int dof = fe.GetDof();
  xy.SetSize(2, dof);
  for (int j=0; j<dof; ++j)
  {
    const IntegrationPoint &ip = nodes.IntPoint(j);
    Vector XP(3); Tr.Transform(ip, XP);
    xy(0,j) = XP(0); xy(1,j) = XP(1);
  }
}

// Slope weights: wx[j] ≈ ∫ ∂x φ_j, wy[j] ≈ ∫ ∂y φ_j  (over the physical element).
static void SlopeWeights(const FiniteElement &fe, ElementTransformation &Tr,
                         int qord_slope, Vector &wx, Vector &wy, double &area_out)
{
  const int dof = fe.GetDof();
  wx.SetSize(dof); wy.SetSize(dof); wx = 0.0; wy = 0.0;
  area_out = 0.0;

  const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), qord_slope);
  DenseMatrix dshape(dof, Tr.GetSpaceDim());
  for (int q=0; q<ir.GetNPoints(); ++q)
  {
    const IntegrationPoint &ip = ir.IntPoint(q);
    Tr.SetIntPoint(&ip);
    fe.CalcDShape(ip, dshape);
    const double w = ip.weight * Tr.Weight();
    for (int j=0; j<dof; ++j) { wx(j) += w * dshape(j,0); wy(j) += w * dshape(j,1); }
    area_out += w;
  }
}

// Return isotropic element length h from area (dx=dy=h); simple and robust.
static inline std::pair<double,double>
ElementDXDYFromArea(ElementTransformation &Tr, int qord)
{
  const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), qord);
  double area = 0.0;
  for (int q=0; q<ir.GetNPoints(); ++q)
  { const IntegrationPoint &ip = ir.IntPoint(q); Tr.SetIntPoint(&ip); area += ip.weight * Tr.Weight(); }
  const double h = std::sqrt(std::max(area, 1e-30));
  return std::make_pair(h, h);
}

void ApplyCharacteristicTVBLimiter(ParFiniteElementSpace &vfes,
                                   ParGridFunction &U,
                                   const CharTVBOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Characteristic TVB limiter currently implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Characteristic TVB limiter assumes byNODES ordering.");

  // Use non-const mesh since some helpers are non-const in some MFEM builds
  Mesh *pm = vfes.GetParMesh();
  Mesh &mesh = pm ? static_cast<Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  // Work arrays
  Array<int> vdofs;
  Vector elvec;
  DenseMatrix elmat;                  // (dof × num_eq)
  Vector Ubar(num_eq), row(num_eq);
  Vector wx, wy;                      // slope weights per DOF
  Array2D<double> dof_xy;             // physical coords of DOFs
  Array<Nbr4> nbr; Build4Nbrs(mesh, nbr);

  // Precompute cell averages and lengths for all elements
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
    ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
    dxy_all[i] = ElementDXDYFromArea(*Tr, qord_avg);
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

    // Candidate DG slopes via ∫ grad φ_j
    double area=0.0;
    Vector wx, wy;
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    Vector sDGx(num_eq); sDGx = 0.0;
    Vector sDGy(num_eq); sDGy = 0.0;
    Vector rowloc(num_eq);
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, rowloc);
      sDGx.Add(wx(j), rowloc);
      sDGy.Add(wy(j), rowloc);
    }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    // One-sided mean slopes (boundary => copy self mean)
    auto mean_or_self = [&](int j)->const Vector& { return (j>=0) ? Ubar_all[j] : Ubar_all[i]; };

    const Vector &Uxm = mean_or_self(nbr[i].xm);
    const Vector &Uxp = mean_or_self(nbr[i].xp);
    const Vector &Uym = mean_or_self(nbr[i].ym);
    const Vector &Uyp = mean_or_self(nbr[i].yp);

    Vector dlx(num_eq), drx(num_eq), dly(num_eq), dry(num_eq);
    Subtract(Ubar, Uxm, dlx); dlx /= std::max(dx,1e-16);
    Subtract(Uxp, Ubar, drx); drx /= std::max(dx,1e-16);
    Subtract(Ubar, Uym, dly); dly /= std::max(dy,1e-16);
    Subtract(Uyp, Ubar, dry); dry /= std::max(dy,1e-16);

    // Characteristic transforms (local)
    const double rho = std::max(Ubar(0), 1e-16);
    const double p   = PressureCons(Ubar, 2, opt.gamma);
    const double c   = std::sqrt(std::max(opt.gamma*p/rho, 0.0));
    const double H   = (Ubar(3) + p)/rho;

    DenseMatrix R_x(4,4), R_y(4,4), L_x(4,4), L_y(4,4);
    EulerRightEigen2D(Ubar, c, H, R_x, R_y);
    EulerLeftEigen2D (Ubar, c, H, L_x, L_y);

    Vector A_x(4), B_x(4), C_x(4), A_y(4), B_y(4), C_y(4);
    L_x.Mult(sDGx, A_x);  L_x.Mult(dlx, B_x);  L_x.Mult(drx, C_x);
    L_y.Mult(sDGy, A_y);  L_y.Mult(dly, B_y);  L_y.Mult(dry, C_y);

    // Shu–TVB limiter per wave
    Vector W_x(4), W_y(4);
    const double tx = opt.M_tvb * dx*dx;
    const double ty = opt.M_tvb * dy*dy;
    for (int k=0; k<4; ++k)
    {
      W_x(k) = tvb_minmod(A_x(k), B_x(k), C_x(k), opt.theta, tx);
      W_y(k) = tvb_minmod(A_y(k), B_y(k), C_y(k), opt.theta, ty);
    }

    // Back to conserved slopes
    Vector sUx(4), sUy(4);
    R_x.Mult(W_x, sUx);
    R_y.Mult(W_y, sUy);

    // Simple KXRCF-like density jump
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
      // Reconstruct linear polynomial at DOFs: U_j = Ubar + sUx*dx + sUy*dy
      IntegrationPoint ipc; ipc.Set2(0.5,0.5);
      Vector Xc(3); Tr->Transform(ipc, Xc);
      Array2D<double> dof_xy; ElementDofXY(*fe, *Tr, dof_xy);

      for (int j=0; j<dof; ++j)
      {
        const double dxj = dof_xy(0,j) - Xc(0);
        const double dyj = dof_xy(1,j) - Xc(1);
        Vector row(num_eq);
        elmat.GetRow(j, row);
        row = Ubar;
        row.Add(dxj, sUx);
        row.Add(dyj, sUy);
        elmat.SetRow(j, row);
      }
      U.SetSubVector(vdofs, elmat.GetData());
    }
    // else: leave DOFs unchanged
  }

  if (vfes.GetParMesh()) { U.ExchangeFaceNbrData(); }
}

// ========================== Barth–Jespersen limiter ==========================

void ApplyBarthJespersenLimiter(mfem::ParFiniteElementSpace &vfes,
                                mfem::ParGridFunction &U,
                                const BJOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Barth–Jespersen limiter currently implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Barth–Jespersen limiter assumes byNODES ordering.");

  Mesh *pm = vfes.GetParMesh();
  Mesh &mesh = pm ? static_cast<Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4 for Euler 2D
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = (opt.qord_avg  >0) ? opt.qord_avg  : (2*order + 2);
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord: (2*order + 2);

  // neighbor map and working arrays
  Array<Nbr4> nbr; Build4Nbrs(mesh, nbr);

  Array<int> vdofs;
  Vector elvec;
  DenseMatrix elmat;                 // (dof × num_eq)
  Vector Ubar(num_eq), row(num_eq);
  Vector wx, wy;                     // slope weights
  Array2D<double> dof_xy;

  // Precompute all cell averages
  std::vector<Vector> Ubar_all(NE, Vector(num_eq));
  for (int i=0; i<NE; ++i)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
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
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    Ubar = Ubar_all[i];

    // Slope weights and candidate slopes
    double area = 0.0;
    SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    Vector sx(num_eq); sx = 0.0;
    Vector sy(num_eq); sy = 0.0;
    Vector rowloc(num_eq);
    for (int j=0; j<dof; ++j)
    {
      elmat.GetRow(j, rowloc);
      sx.Add(wx(j), rowloc);
      sy.Add(wy(j), rowloc);
    }
    sx /= std::max(area, 1e-30);
    sy /= std::max(area, 1e-30);

    // Element center and DOF coordinates
    IntegrationPoint ipc; ipc.Set2(0.5,0.5);
    Vector Xc(3); Tr->Transform(ipc, Xc);
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
