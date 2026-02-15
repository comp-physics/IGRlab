#include "physics/euler/characteristics.hpp"
#include "physics/euler/eos.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{

void EulerRightEigen2D(const mfem::Vector& Ubar,
                       double c,
                       double H,
                       mfem::DenseMatrix& R_x,
                       mfem::DenseMatrix& R_y)
{
  const double rho = std::max(Ubar(0), 1e-16);
  const double ux = Ubar(1) / rho;
  const double uy = Ubar(2) / rho;
  const double u2 = ux * ux + uy * uy;

  // R_x
  R_x.SetSize(4, 4);
  R_x = 0.0;
  R_x(0, 0) = 1;
  R_x(1, 0) = ux - c;
  R_x(2, 0) = uy;
  R_x(3, 0) = H - ux * c;
  R_x(0, 1) = 1;
  R_x(1, 1) = ux;
  R_x(2, 1) = uy;
  R_x(3, 1) = 0.5 * u2;
  R_x(0, 2) = 0;
  R_x(1, 2) = 0;
  R_x(2, 2) = 1;
  R_x(3, 2) = uy;
  R_x(0, 3) = 1;
  R_x(1, 3) = ux + c;
  R_x(2, 3) = uy;
  R_x(3, 3) = H + ux * c;

  // R_y
  R_y.SetSize(4, 4);
  R_y = 0.0;
  R_y(0, 0) = 1;
  R_y(1, 0) = ux;
  R_y(2, 0) = uy - c;
  R_y(3, 0) = H - uy * c;
  R_y(0, 1) = 1;
  R_y(1, 1) = ux;
  R_y(2, 1) = uy;
  R_y(3, 1) = 0.5 * u2;
  R_y(0, 2) = 0;
  R_y(1, 2) = 1;
  R_y(2, 2) = 0;
  R_y(3, 2) = ux;
  R_y(0, 3) = 1;
  R_y(1, 3) = ux;
  R_y(2, 3) = uy + c;
  R_y(3, 3) = H + uy * c;
}

void EulerLeftEigen2D(const mfem::Vector& Ubar,
                      double c,
                      double H,
                      mfem::DenseMatrix& L_x,
                      mfem::DenseMatrix& L_y)
{
  const double rho = std::max(Ubar(0), 1e-16);
  const double ux = Ubar(1) / rho;
  const double uy = Ubar(2) / rho;
  const double u2 = ux * ux + uy * uy;
  const double q = 0.5 * u2;
  const double h = H - q;

  L_x.SetSize(4, 4);
  L_x = 0.0;
  L_x(0, 0) = 0.5 * (q / h + ux / c);
  L_x(0, 1) = -0.5 * (ux / h + 1. / c);
  L_x(0, 2) = -0.5 * (uy / h);
  L_x(0, 3) = 0.5 * (1. / h);
  L_x(1, 0) = 1. - q / h;
  L_x(1, 1) = ux / h;
  L_x(1, 2) = uy / h;
  L_x(1, 3) = -1. / h;
  L_x(2, 0) = -uy;
  L_x(2, 1) = 0.;
  L_x(2, 2) = 1.;
  L_x(2, 3) = 0.;
  L_x(3, 0) = 0.5 * (q / h - ux / c);
  L_x(3, 1) = -0.5 * (ux / h - 1. / c);
  L_x(3, 2) = -0.5 * (uy / h);
  L_x(3, 3) = 0.5 * (1. / h);

  L_y.SetSize(4, 4);
  L_y = 0.0;
  L_y(0, 0) = 0.5 * (q / h + uy / c);
  L_y(0, 1) = -0.5 * (ux / h);
  L_y(0, 2) = -0.5 * (uy / h + 1. / c);
  L_y(0, 3) = 0.5 * (1. / h);
  L_y(1, 0) = 1. - q / h;
  L_y(1, 1) = ux / h;
  L_y(1, 2) = uy / h;
  L_y(1, 3) = -1. / h;
  L_y(2, 0) = -ux;
  L_y(2, 1) = 1.;
  L_y(2, 2) = 0.;
  L_y(2, 3) = 0.;
  L_y(3, 0) = 0.5 * (q / h - uy / c);
  L_y(3, 1) = -0.5 * (ux / h);
  L_y(3, 2) = -0.5 * (uy / h - 1. / c);
  L_y(3, 3) = 0.5 * (1. / h);
}

void EulerLeftRightInDirection(const mfem::Vector& Ubar,
                               double gamma,
                               double nx,
                               double ny,
                               mfem::DenseMatrix& L_n,
                               mfem::DenseMatrix& R_n)
{
  // Build L_n ~= nx L_x + ny L_y and R_n ~= nx R_x + ny R_y (common approximation).
  const double rho = std::max(Ubar(0), 1e-16);
  const double p =
    EulerPressureFromConservative(Ubar, /*dim=*/2, gamma, /*rho_floor=*/0.0);
  const double c = std::sqrt(std::max(gamma * p / rho, 0.0));
  const double H = (Ubar(3) + p) / rho;

  mfem::DenseMatrix Lx(4, 4), Ly(4, 4), Rx(4, 4), Ry(4, 4);
  EulerLeftEigen2D(Ubar, c, H, Lx, Ly);
  EulerRightEigen2D(Ubar, c, H, Rx, Ry);

  L_n.SetSize(4, 4);
  R_n.SetSize(4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
    {
      L_n(i, j) = nx * Lx(i, j) + ny * Ly(i, j);
      R_n(i, j) = nx * Rx(i, j) + ny * Ry(i, j);
    }
}

void EulerCharNormalFromRotatedUbar(const mfem::Vector& Ubar_rot,
                                    double gamma,
                                    mfem::DenseMatrix& L_n,
                                    mfem::DenseMatrix& R_n)
{
  // Ubar_rot = [rho, m_n, m_t, E]
  const double rho = std::max(Ubar_rot(0), 1e-16);
  const double ux = Ubar_rot(1) / rho; // u_n
  const double uy = Ubar_rot(2) / rho; // u_t (acts like 'transverse' velocity)
  const double E = Ubar_rot(3);
  double mom2 = Ubar_rot(1) * Ubar_rot(1) + Ubar_rot(2) * Ubar_rot(2);
  const double p = (gamma - 1.0) * (E - 0.5 * mom2 / rho);
  const double c = std::sqrt(std::max(gamma * p / rho, 0.0));
  const double H = (E + p) / rho;

  // Use your earlier functions, then use the x-direction blocks (since x ≡ n)
  mfem::DenseMatrix L(4, 4), R(4, 4), Ltmp(4, 4), Rtmp(4, 4);
  EulerRightEigen2D(Ubar_rot, c, H, Rtmp, R); // R : y-dir, Rtmp : x-dir
  EulerLeftEigen2D(Ubar_rot, c, H, Ltmp, L);  // L : y-dir, Ltmp : x-dir
  R_n = Rtmp;                                 // x-direction
  L_n = Ltmp;                                 // x-direction
}

} // namespace mfem
