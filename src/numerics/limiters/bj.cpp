#include "numerics/limiters/bj.hpp"

#include "limiters_common.hpp"
#include "physics/euler/eos.hpp"
#include "physics/euler/characteristics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

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

} // namespace

namespace mfem
{

void ApplyCharacteristicBJLimiter(ParFiniteElementSpace &vfes,
                                  ParGridFunction &U,
                                  const CharBJOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Characteristic BJ limiter currently implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Characteristic BJ limiter assumes byNODES ordering.");

  Mesh *pm = vfes.GetParMesh();
  Mesh &mesh = pm ? static_cast<Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = opt.dim + 2;  // 4
  const int NE = vfes.GetNE();
  const int order = vfes.GetFE(0)->GetOrder();
  const int qord_avg   = 2*order + 2;
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);

  Array<dg_igr::limiters_detail::Nbr4> nbr; dg_igr::limiters_detail::Build4Nbrs(mesh, nbr);

  // Precompute means and "sizes"
  std::vector<Vector> Ubar_all(NE, Vector(num_eq));
  std::vector<std::pair<double,double>> dxy_all(NE);

  {
    Array<int> vdofs; Vector elvec; DenseMatrix elmat; Vector Ubar(num_eq);
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
  }

  // Main per-element loop
  Array<int> vdofs; Vector elvec; DenseMatrix elmat;
  Vector wx, wy, row(num_eq), Ubar(num_eq);

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

    // Candidate DG slopes from ∫ grad phi_j
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

    // Neighbors (fallback to self on boundary)
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

    // Project means and slopes to characteristics
    Vector wbar_x(4), wbar_y(4); L_x.Mult(Ubar, wbar_x); L_y.Mult(Ubar, wbar_y);

    Vector wxm(4), wxp(4), wym(4), wyp(4);
    L_x.Mult(Uxm, wxm); L_x.Mult(Uxp, wxp);
    L_y.Mult(Uym, wym); L_y.Mult(Uyp, wyp);

    Vector W_x(4), W_y(4);
    L_x.Mult(sDGx, W_x);
    L_y.Mult(sDGy, W_y);

    // BJ: compute theta per wave from two face centers per direction
    Vector theta_x(4); theta_x = 1.0;
    Vector theta_y(4); theta_y = 1.0;

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
    Vector sUx(4), sUy(4);
    R_x.Mult(W_x, sUx);
    R_y.Mult(W_y, sUy);

    // Overwrite DOFs with the linear reconstruction Ubar + sUx*dxj + sUy*dyj
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

void ApplyFaceNormalCharacteristicBJLimiter(ParFiniteElementSpace &vfes,
                                            ParGridFunction &U,
                                            const FaceBJOptions &opt)
{
  MFEM_VERIFY(opt.dim==2, "Face-normal characteristic BJ implemented for 2D.");
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Limiter assumes byNODES ordering.");

  Mesh *pm = vfes.GetParMesh();
  Mesh &mesh = pm ? static_cast<Mesh&>(*pm) : *vfes.GetMesh();

  const int num_eq = 4;
  const int NE     = vfes.GetNE();
  const int order  = vfes.GetFE(0)->GetOrder();
  const int qord_slope = (opt.slope_qord>0) ? opt.slope_qord : (2*order + 2);
  const int qord_avg   = 2*order + 2;

  // Precompute means and element centers
  std::vector<Vector> Ubar_all(NE, Vector(num_eq));
  std::vector<Vector> Xc_all(NE, Vector(3));
  for (int i=0; i<NE; ++i) {
    Array<int> v; mesh.GetElementVertices(i, v);
    Vector c(3); c=0.0;
    for (int k=0; k<v.Size(); ++k) { const double *X = mesh.GetVertex(v[k]); c(0)+=X[0]; c(1)+=X[1]; }
    c /= std::max(1, v.Size()); Xc_all[i] = c;

    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    Array<int> vdofs; vfes.GetElementVDofs(i, vdofs);
    Vector elvec; U.GetSubVector(vdofs, elvec);
    DenseMatrix elmat; elmat.UseExternalData(elvec.GetData(), fe->GetDof(), num_eq);
    Vector Ubar(num_eq); dg_igr::limiters_detail::ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
    Ubar_all[i] = Ubar;
  }

  // Per-element face info (face center, outward approx normal, neighbor id)
  std::vector<std::vector<FaceInfo>> faces_of_elem;
  BuildFaceInfosPerElem(mesh, Xc_all, faces_of_elem);

  // Work buffers
  Array<int> vdofs;
  Vector elvec, row(num_eq), Ubar(num_eq);
  DenseMatrix elmat;
  Vector wx, wy; double area=0.0;

  // Main element loop
  for (int i=0; i<NE; ++i)
  {
    const FiniteElement *fe = vfes.GetFE(i);
    ElementTransformation *Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // Mean and center
    Ubar = Ubar_all[i];
    const Vector &Xci = Xc_all[i];

    // Candidate average slopes via ∫ grad phi_j / area
    dg_igr::limiters_detail::SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
    Vector sDGx(num_eq); sDGx = 0.0;
    Vector sDGy(num_eq); sDGy = 0.0;
    for (int j=0; j<dof; ++j) { elmat.GetRow(j, row); sDGx.Add(wx(j), row); sDGy.Add(wy(j), row); }
    sDGx /= std::max(area, 1e-30);
    sDGy /= std::max(area, 1e-30);

    // Compute theta from all faces (min over faces & waves)
    double theta = 1.0;

    for (const FaceInfo &fi : faces_of_elem[i])
    {
      // Face delta from center
      const double dxf = fi.fx - Xci(0);
      const double dyf = fi.fy - Xci(1);

      // Predict state at face center
      Vector Up(num_eq); Up = Ubar; Up.Add(dxf, sDGx); Up.Add(dyf, sDGy);

      // Rotate Ubar and Up into (n,t) frame
      auto rotU = [&](const Vector &Uin)->Vector {
        Vector Ur(num_eq);
        const double rho = std::max(Uin(0), opt.eps_rho);
        double mn, mt; RotateMomToNT(Uin(1), Uin(2), fi.nx, fi.ny, mn, mt);
        Ur(0)=rho; Ur(1)=mn; Ur(2)=mt; Ur(3)=Uin(3);
        return Ur;
      };
      const Vector Ubar_r = rotU(Ubar);
      const Vector Up_r   = rotU(Up);

      // Neighbor mean (or self for boundary)
      const Vector &Unb = (fi.nbr>=0) ? Ubar_all[fi.nbr] : Ubar_all[i];
      const Vector Unb_r = rotU(Unb);

      // Characteristic left/right in the *normal* (x) direction for rotated state
      DenseMatrix L_n(4,4), R_n(4,4);
      EulerCharNormalFromRotatedUbar(Ubar_r, opt.gamma, L_n, R_n);

      // Project to char space
      Vector w_bar(4), w_pred(4), w_nbr(4);
      L_n.Mult(Ubar_r, w_bar);
      L_n.Mult(Up_r,   w_pred);
      L_n.Mult(Unb_r,  w_nbr);

      // Per-wave BJ clamp vs neighbor; compute a face theta_f in [0,1]
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
      Array2D<double> xy; dg_igr::limiters_detail::ElementDofXY(*fe, *Tr, xy);
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

void ApplyBarthJespersenLimiter(ParFiniteElementSpace &vfes,
                                ParGridFunction &U,
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
  Array<dg_igr::limiters_detail::Nbr4> nbr; dg_igr::limiters_detail::Build4Nbrs(mesh, nbr);

  Array<int> vdofs;
  Vector elvec;
  DenseMatrix elmat;                 // (dof x num_eq)
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
    dg_igr::limiters_detail::ElementAverage(*fe, *Tr, elmat, num_eq, qord_avg, Ubar);
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
    dg_igr::limiters_detail::SlopeWeights(*fe, *Tr, qord_slope, wx, wy, area);
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
    dg_igr::limiters_detail::ElementDofXY(*fe, *Tr, dof_xy);

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

} // namespace mfem
