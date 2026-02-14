#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include "mfem.hpp"

namespace mfem::dg_igr::limiters_detail
{

// Make POD for mfem::Array<T>
struct Nbr4
{
  int xm, xp, ym, yp;
};

inline void BlendRow(mfem::Vector& row, const mfem::Vector& ubar, double theta)
{
  for (int k = 0; k < row.Size(); ++k)
  {
    row(k) = ubar(k) + theta * (row(k) - ubar(k));
  }
}

inline void
Subtract(const mfem::Vector& a, const mfem::Vector& b, mfem::Vector& out)
{
  const int n = a.Size();
  out.SetSize(n);
  for (int i = 0; i < n; ++i)
    out(i) = a(i) - b(i);
}

inline void ElementAverage(const mfem::FiniteElement& fe,
                           mfem::ElementTransformation& Tr,
                           const mfem::DenseMatrix& elfun_mat,
                           int num_eq,
                           int qord,
                           mfem::Vector& Ubar_out)
{
  const mfem::IntegrationRule& ir =
    mfem::IntRules.Get(Tr.GetGeometryType(), qord);
  mfem::Vector shape(fe.GetDof()), Upt(num_eq);
  Ubar_out.SetSize(num_eq);
  Ubar_out = 0.0;
  double vol = 0.0;
  for (int i = 0; i < ir.GetNPoints(); i++)
  {
    const mfem::IntegrationPoint& ip = ir.IntPoint(i);
    Tr.SetIntPoint(&ip);
    fe.CalcShape(ip, shape);
    elfun_mat.MultTranspose(shape, Upt);
    const double w = ip.weight * Tr.Weight();
    Ubar_out.Add(w, Upt);
    vol += w;
  }
  Ubar_out /= vol;
}

inline void ComputeElementCenter(mfem::Mesh& mesh, int e, mfem::Vector& c)
{
  mfem::Array<int> verts;
  mesh.GetElementVertices(e, verts);
  c.SetSize(3);
  c = 0.0;
  for (int vi = 0; vi < verts.Size(); ++vi)
  {
    const double* X = mesh.GetVertex(verts[vi]);
    c(0) += X[0];
    c(1) += X[1];
    c(2) += (mesh.Dimension() > 2 ? X[2] : 0.0);
  }
  const double inv = 1.0 / std::max(1, verts.Size());
  c *= inv;
}

inline void ComputeFaceCenter(mfem::Mesh& mesh, int f, mfem::Vector& c)
{
  mfem::Array<int> verts;
  mesh.GetFaceVertices(f, verts);
  c.SetSize(3);
  c = 0.0;
  for (int vi = 0; vi < verts.Size(); ++vi)
  {
    const double* X = mesh.GetVertex(verts[vi]);
    c(0) += X[0];
    c(1) += X[1];
    c(2) += (mesh.Dimension() > 2 ? X[2] : 0.0);
  }
  const double inv = 1.0 / std::max(1, verts.Size());
  c *= inv;
}

inline void Build4Nbrs(mfem::Mesh& mesh, mfem::Array<Nbr4>& nbr)
{
  const int NE = mesh.GetNE();
  nbr.SetSize(NE);
  for (int i = 0; i < NE; ++i)
  {
    nbr[i].xm = nbr[i].xp = nbr[i].ym = nbr[i].yp = -1;
  }

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1 = -1, e2 = -1;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e1 < 0 || e2 < 0)
    {
      continue;
    } // boundary face

    mfem::Vector c1(3), c2(3), fc(3);
    ComputeFaceCenter(mesh, f, fc);
    ComputeElementCenter(mesh, e1, c1);
    ComputeElementCenter(mesh, e2, c2);

    const double dx = fc(0) - c1(0);
    const double dy = fc(1) - c1(1);

    if (std::abs(dx) > std::abs(dy))
    {
      if (dx > 0)
      {
        nbr[e1].xp = e2;
        nbr[e2].xm = e1;
      }
      else
      {
        nbr[e1].xm = e2;
        nbr[e2].xp = e1;
      }
    }
    else
    {
      if (dy > 0)
      {
        nbr[e1].yp = e2;
        nbr[e2].ym = e1;
      }
      else
      {
        nbr[e1].ym = e2;
        nbr[e2].yp = e1;
      }
    }
  }
}

// DOF physical coordinates (use FE nodes).
inline void ElementDofXY(const mfem::FiniteElement& fe,
                         mfem::ElementTransformation& Tr,
                         mfem::Array2D<double>& xy)
{
  const mfem::IntegrationRule& nodes = fe.GetNodes();
  const int dof = fe.GetDof();
  xy.SetSize(2, dof);
  for (int j = 0; j < dof; ++j)
  {
    const mfem::IntegrationPoint& ip = nodes.IntPoint(j);
    mfem::Vector XP(3);
    Tr.Transform(ip, XP);
    xy(0, j) = XP(0);
    xy(1, j) = XP(1);
  }
}

// Slope weights: wx[j] ≈ ∫ ∂x phi_j, wy[j] ≈ ∫ ∂y phi_j (over the physical element).
inline void SlopeWeights(const mfem::FiniteElement& fe,
                         mfem::ElementTransformation& Tr,
                         int qord_slope,
                         mfem::Vector& wx,
                         mfem::Vector& wy,
                         double& area_out)
{
  const int dof = fe.GetDof();
  wx.SetSize(dof);
  wy.SetSize(dof);
  wx = 0.0;
  wy = 0.0;
  area_out = 0.0;

  const mfem::IntegrationRule& ir =
    mfem::IntRules.Get(Tr.GetGeometryType(), qord_slope);
  mfem::DenseMatrix dshape(dof, Tr.GetSpaceDim());
  for (int q = 0; q < ir.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint& ip = ir.IntPoint(q);
    Tr.SetIntPoint(&ip);
    fe.CalcDShape(ip, dshape);
    const double w = ip.weight * Tr.Weight();
    for (int j = 0; j < dof; ++j)
    {
      wx(j) += w * dshape(j, 0);
      wy(j) += w * dshape(j, 1);
    }
    area_out += w;
  }
}

// Return isotropic element length h from area (dx=dy=h); simple and robust.
inline std::pair<double, double>
ElementDXDYFromArea(mfem::ElementTransformation& Tr, int qord)
{
  const mfem::IntegrationRule& ir =
    mfem::IntRules.Get(Tr.GetGeometryType(), qord);
  double area = 0.0;
  for (int q = 0; q < ir.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint& ip = ir.IntPoint(q);
    Tr.SetIntPoint(&ip);
    area += ip.weight * Tr.Weight();
  }
  const double h = std::sqrt(std::max(area, 1e-30));
  return std::make_pair(h, h);
}

// Per-face neighbor info for each element (2D). We keep POD types only.
struct FaceNbrRec
{
  int nbr;       // neighbor element id, -1 for boundary
  double nx, ny; // unit normal from this element toward the face
  double dn;     // spacing along normal (centroid-to-(nbr|face)) projected on n
};

// Build per-element face records with (nx,ny) and neighbor id.
inline void
BuildPerElementFaceNormals(mfem::Mesh& mesh,
                           std::vector<std::vector<FaceNbrRec>>& per_elem)
{
  const int NE = mesh.GetNE();
  per_elem.clear();
  per_elem.resize(NE);

  mfem::Vector c1(3), c2(3), fc(3);

  auto norm2 = [](double x, double y)
  {
    return std::sqrt(x * x + y * y);
  };

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1 = -1, e2 = -1;
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
      const double len = std::max(1e-14, norm2(nxe1, nye1));
      FaceNbrRec rec;
      rec.nx = nxe1 / len;
      rec.ny = nye1 / len;
      rec.nbr = (e2 >= 0) ? e2 : -1;
      if (e2 >= 0)
      { // interior: spacing is centroid diff projected on n
        const double dx = c2(0) - c1(0);
        const double dy = c2(1) - c1(1);
        rec.dn = std::abs(dx * rec.nx + dy * rec.ny);
      }
      else
      { // boundary: centroid-to-face projected distance
        const double dx = fc(0) - c1(0);
        const double dy = fc(1) - c1(1);
        rec.dn = std::abs(dx * rec.nx + dy * rec.ny);
      }
      per_elem[e1].push_back(rec);
    }

    // element e2 side
    if (e2 >= 0)
    {
      const double nxe2 = fc(0) - c2(0);
      const double nye2 = fc(1) - c2(1);
      const double len = std::max(1e-14, norm2(nxe2, nye2));
      FaceNbrRec rec;
      rec.nx = nxe2 / len;
      rec.ny = nye2 / len;
      rec.nbr = (e1 >= 0) ? e1 : -1;
      if (e1 >= 0)
      {
        const double dx = c1(0) - c2(0);
        const double dy = c1(1) - c2(1);
        rec.dn = std::abs(dx * rec.nx + dy * rec.ny);
      }
      else
      {
        const double dx = fc(0) - c2(0);
        const double dy = fc(1) - c2(1);
        rec.dn = std::abs(dx * rec.nx + dy * rec.ny);
      }
      per_elem[e2].push_back(rec);
    }
  }
}

} // namespace mfem::dg_igr::limiters_detail
