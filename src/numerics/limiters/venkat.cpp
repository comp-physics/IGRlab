#include "numerics/limiters/venkat.hpp"

#include "limiters_common.hpp"
#include "physics/euler/eos.hpp"
#include "physics/euler/characteristics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Smooth limiter function phi_VK(a,b,eps), where
//   a = predicted increment to face (characteristic component),
//   b = allowed increment (same sign as a; else b = 0),
//   eps = smoothness parameter (per face).
// This form behaves like: phi≈1 when |a| << |b|; phi→0 as |a| >> |b|; smooth at extrema.
static inline double venkat_phi(double a, double b, double eps)
{
  // enforce same-sign allowed; if opposite sign, limit to 0
  if (a*b <= 0.0) return 0.0;

  const double aa = a*a;
  const double bb = b*b;
  const double ab = a*b;
  // Common smooth form used in practice:
  //   phi = (aa + 2ab + eps) / (aa + ab + 2bb + eps)
  const double num = aa + 2.0*ab + eps;
  const double den = aa + ab + 2.0*bb + eps;
  return (den > 0.0) ? std::min(1.0, std::max(0.0, num/den)) : 0.0;
}

} // namespace

namespace mfem
{

void ApplyCharacteristicVenkatFaceNormal(ParFiniteElementSpace &vfes,
                                         ParGridFunction &U,
                                         const VenkatOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Venkat face-normal limiter implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Venkat face-normal limiter assumes byNODES ordering.");

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

  // Precompute cell averages
  std::vector<Vector> Ubar_all(NE, Vector(num_eq));

  Array<int> vdofs;
  Vector elvec, Ubar(num_eq), row(num_eq);
  DenseMatrix elmat;

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

  // Main loop: per element
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

    Ubar = Ubar_all[i];

    // Candidate DG slopes via ∫ grad phi_j
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

      // Venkat threshold with directional size
      const double h = dn;
      const double eps = opt.eps_kappa * h*h*h;

      Vector W(4);
      for (int k=0; k<4; ++k)
      {
        W(k) = venkat_phi(A(k), B(k), eps);
        const double wplus = venkat_phi(A(k), C(k), eps);
        W(k) = std::min(W(k), wplus);
        W(k) *= A(k);
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

void ApplyCharacteristicVenkatXY(ParFiniteElementSpace &vfes,
                                 ParGridFunction &U,
                                 const VenkatOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Venkat XY implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Venkat limiter assumes byNODES ordering.");

  Mesh *pm = vfes.GetParMesh();
  Mesh &mesh = pm ? static_cast<Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  Array<dg_igr::limiters_detail::Nbr4> nbr; dg_igr::limiters_detail::Build4Nbrs(mesh, nbr);

  // Precompute cell averages
  std::vector<Vector> Ubar_all(NE, Vector(num_eq));
  std::vector<std::pair<double,double>> dxy_all(NE);

  Array<int> vdofs;
  Vector elvec; DenseMatrix elmat; Vector Ubar(num_eq);
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

  // Main loop
  Vector wx, wy, row(num_eq);
  for (int i=0; i<NE; ++i)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();

    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    Ubar = Ubar_all[i];
    const std::pair<double,double> dxy = dxy_all[i];
    const double dx = dxy.first;
    const double dy = dxy.second;

    // Candidate DG slopes
    double area = 0.0;
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

    // Neighbors (fallback to self)
    auto mean_or_self = [&](int j)->const Vector& { return (j>=0) ? Ubar_all[j] : Ubar_all[i]; };
    const Vector &Uxm = mean_or_self(nbr[i].xm);
    const Vector &Uxp = mean_or_self(nbr[i].xp);
    const Vector &Uym = mean_or_self(nbr[i].ym);
    const Vector &Uyp = mean_or_self(nbr[i].yp);

    // Characteristic transforms at the mean state
    const double rho = std::max(Ubar(0), 1e-16);
    const double p   = PressureFromCons(Ubar, 2, opt.gamma, /*rho_floor=*/1e-16);
    const double c   = std::sqrt(std::max(opt.gamma*p/rho, 0.0));
    const double H   = (Ubar(3) + p)/rho;

    DenseMatrix R_x(4,4), R_y(4,4), L_x(4,4), L_y(4,4);
    EulerRightEigen2D(Ubar, c, H, R_x, R_y);
    EulerLeftEigen2D (Ubar, c, H, L_x, L_y);

    Vector W_x(4), W_y(4);
    L_x.Mult(sDGx, W_x);
    L_y.Mult(sDGy, W_y);

    // Neighbor means in characteristic space
    Vector wxm(4), wxp(4), wym(4), wyp(4);
    L_x.Mult(Uxm, wxm); L_x.Mult(Uxp, wxp);
    L_y.Mult(Uym, wym); L_y.Mult(Uyp, wyp);

    // mean in characteristic space
    Vector wbar_x(4), wbar_y(4);
    L_x.Mult(Ubar, wbar_x); L_y.Mult(Ubar, wbar_y);

    // Venkat limiter on each characteristic component
    const double epsx = opt.eps_kappa * dx*dx*dx;
    const double epsy = opt.eps_kappa * dy*dy*dy;

    for (int k=0; k<4; ++k)
    {
      // X direction: use neighbor mean bounds
      const double wmin = std::min(wxm(k), wxp(k));
      const double wmax = std::max(wxm(k), wxp(k));
      const double a = W_x(k);
      const double b = (a > 0.0) ? (wmax - wbar_x(k)) : (wmin - wbar_x(k));
      const double phi = venkat_phi(a, b, epsx);
      W_x(k) = phi * a;

      // Y direction
      const double wminy = std::min(wym(k), wyp(k));
      const double wmaxy = std::max(wym(k), wyp(k));
      const double ay = W_y(k);
      const double by = (ay > 0.0) ? (wmaxy - wbar_y(k)) : (wminy - wbar_y(k));
      const double phiy = venkat_phi(ay, by, epsy);
      W_y(k) = phiy * ay;
    }

    // Back to conserved slopes
    Vector sUx(4), sUy(4);
    R_x.Mult(W_x, sUx);
    R_y.Mult(W_y, sUy);

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
