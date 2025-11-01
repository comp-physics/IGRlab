// DGSEMCurvedVolume.cpp
#include "DGSEMCurvedVolume.hpp"
#include <algorithm>
#include <cmath>

namespace mfem
{

// ========================= VolumeRusanovFlux =========================

real_t VolumeRusanovFlux::Eval(const Vector &U1, const Vector &U2,
                               const Vector &nor, ElementTransformation &Tr,
                               Vector &flux_out) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction_.num_equations);
   Vector fluxN2(fluxFunction_.num_equations);
#else
   Vector &fluxN1 = fluxN1_;
   Vector &fluxN2 = fluxN2_;
   if (fluxN1.Size() != fluxFunction_.num_equations) fluxN1.SetSize(fluxFunction_.num_equations);
   if (fluxN2.Size() != fluxFunction_.num_equations) fluxN2.SetSize(fluxFunction_.num_equations);
#endif

   // Fake a face transform referencing the same element on both sides.
   FaceElementTransformations fake_face;
   fake_face.Elem1 = &Tr;
   fake_face.Elem2 = &Tr;

   const real_t s1 = fluxFunction_.ComputeFluxDotN(U1, nor, fake_face, fluxN1);
   const real_t s2 = fluxFunction_.ComputeFluxDotN(U2, nor, fake_face, fluxN2);
   const real_t maxE = std::max(s1, s2);
   const real_t scaledMaxE = maxE * nor.Norml2(); // MFEM convention

   flux_out.SetSize(fluxFunction_.num_equations);
   for (int i = 0; i < fluxFunction_.num_equations; i++)
   {
      // ½(F·n(u⁻) + F·n(u⁺)) − ½ |λ|max ||n|| (u⁺ − u⁻)
      flux_out(i) = 0.5*(fluxN1(i) + fluxN2(i)) - 0.5*scaledMaxE*(U2(i) - U1(i));
   }
   return maxE;
}

// ============================== helpers ==============================

static void BaryWeights1D(const Array<double> &x, Vector &w)
{
   const int n = x.Size();
   w.SetSize(n);
   for (int j = 0; j < n; ++j)
   {
      double prod = 1.0;
      for (int k = 0; k < n; ++k) if (k != j) prod *= (x[j] - x[k]);
      w[j] = 1.0 / prod;
   }
}

static void LagrangeD1D(const Array<double> &x, DenseMatrix &D)
{
   const int n = x.Size();
   Vector w; BaryWeights1D(x, w);
   D.SetSize(n, n);
   for (int i = 0; i < n; ++i)
   {
      double sum = 0.0;
      for (int j = 0; j < n; ++j)
      {
         if (i == j) { D(i,j) = 0.0; continue; }
         const double v = (w[j]/w[i]) / (x[i]-x[j]);
         D(i,j) = v; sum += v;
      }
      D(i,i) = -sum; // SBP: rows sum to zero
   }
}

static void LumpedMass(const FiniteElement &fe, ElementTransformation &Tr, Vector &m)
{
   MassIntegrator mi; DenseMatrix M;
   mi.AssembleElementMatrix(fe, Tr, M);
   const int dof = fe.GetDof();
   m.SetSize(dof);
   for (int i = 0; i < dof; ++i)
   {
      double s = 0.0;
      for (int j = 0; j < dof; ++j) s += M(i,j);
      m[i] = s;
   }
}

static void RefTensorNodes2D(const FiniteElement &fe,
                             Array<double> &x1d, Array<double> &y1d)
{
   const IntegrationRule &nodes = fe.GetNodes();
   const int dof = fe.GetDof();
   Array<double> xs, ys; xs.Reserve(dof); ys.Reserve(dof);
   for (int j = 0; j < dof; ++j)
   {
      const IntegrationPoint &ip = nodes.IntPoint(j);
      xs.Append(ip.x); ys.Append(ip.y);
   }
   auto uniq_sorted = [](Array<double> &a, Array<double> &out)
   {
      std::vector<double> v(a.GetData(), a.GetData()+a.Size());
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end(),
                          [](double A,double B){ return std::abs(A-B) < 1e-14; }),
              v.end());
      out.SetSize((int)v.size());
      for (int i=0;i<(int)v.size();++i) out[i]=v[i];
   };
   uniq_sorted(xs, x1d);
   uniq_sorted(ys, y1d);
}

// ===================== main DGSEM curved volume kernel =======================

void AddDGSEMVolumeContribution(const ParFiniteElementSpace &vfes,
                                const ParGridFunction       &U,
                                const FluxFunction          &fluxF,
                                const DGSEMOptions          &opt,
                                Vector                      &z)
{
   MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
               "DGSEM volume expects nodal byNODES DG space.");

   Mesh *pm = vfes.GetParMesh();
   Mesh &mesh = pm ? static_cast<Mesh&>(*pm) : *vfes.GetMesh();

   const int dim     = mesh.Dimension();
   MFEM_VERIFY(dim == 2, "This implementation currently supports 2D quads.");
   const int num_eq  = vfes.GetVDim();
   const int NE      = vfes.GetNE();

   VolumeRusanovFlux vol_flux(fluxF);

   Array<int> vdofs;
   Vector elvec;
   DenseMatrix Uloc, Zloc;
   Vector m_lumped;

//    MFEM_VERIFY(U.IsFinite(), "Non-finite values in global U.");

   for (int e = 0; e < NE; ++e)
   {
      const FiniteElement *fe = vfes.GetFE(e);
      MFEM_VERIFY(fe->GetGeomType() == Geometry::SQUARE ||
                  fe->GetGeomType() == Geometry::CUBE,
                  "DGSEM volume requires tensor-product quads/hex.");
      const int dof = fe->GetDof();

      ElementTransformation *Tr = vfes.GetElementTransformation(e);
      vfes.GetElementVDofs(e, vdofs);

      // local views
      U.GetSubVector(vdofs, elvec);
      Uloc.UseExternalData(elvec.GetData(), dof, num_eq);

      z.GetSubVector(vdofs, elvec);
      Zloc.UseExternalData(elvec.GetData(), dof, num_eq);

    //   MFEM_VERIFY(Uloc.IsFinite(), "Non-finite in Uloc.");

      // Lumped mass
      LumpedMass(*fe, *Tr, m_lumped);

      // 1D nodes and derivative matrices
      Array<double> x1d, y1d;
      RefTensorNodes2D(*fe, x1d, y1d);
      const int nx = x1d.Size(), ny = y1d.Size();
      MFEM_VERIFY(nx*ny == dof, "Element is not nodal tensor-product.");
      DenseMatrix Dx(nx, nx), Dy(ny, ny);
      LagrangeD1D(x1d, Dx);
      LagrangeD1D(y1d, Dy);

      auto idx = [ny](int ix,int iy){ return ix*ny + iy; };

      DenseMatrix Div(dof, num_eq); Div = 0.0;
      Vector Ui(num_eq), Uj(num_eq), nor(dim), flux(num_eq);

      const IntegrationRule &nodes = fe->GetNodes();

      auto contravariant_col_at_node = [&](int node, int m, Vector &g_out) -> bool
      {
         const IntegrationPoint &ip = nodes.IntPoint(node);
         Tr->SetIntPoint(&ip);

         const DenseMatrix &J = Tr->Jacobian();
         double detJ = J.Det();
         if (!std::isfinite(detJ) || std::abs(detJ) <= opt.small_detj) return false;

         DenseMatrix JinvT(dim, dim);
         CalcInverseTranspose(J, JinvT);

         g_out.SetSize(dim);
         for (int k = 0; k < dim; ++k) g_out(k) = detJ * JinvT(k, m);
        //  return g_out.IsFinite() && (g_out.Norml2() > 0.0);
      };

      // xi-direction (m=0)
      for (int ix = 0; ix < nx; ++ix)
      for (int jx = 0; jx < nx; ++jx)
      {
         const double d = Dx(ix, jx);
         if (std::abs(d) < 1e-14) continue;

         for (int iy = 0; iy < ny; ++iy)
         {
            const int i = idx(ix, iy);
            const int j = idx(jx, iy);

            if (!contravariant_col_at_node(i, 0, nor)) continue;

            Uloc.GetRow(i, Ui);
            Uloc.GetRow(j, Uj);

            vol_flux.Eval(Ui, Uj, nor, *Tr, flux);

            for (int c = 0; c < num_eq; ++c) Div(i, c) += d * flux(c);
         }
      }

      // eta-direction (m=1)
      for (int iy = 0; iy < ny; ++iy)
      for (int jy = 0; jy < ny; ++jy)
      {
         const double d = Dy(iy, jy);
         if (std::abs(d) < 1e-14) continue;

         for (int ix = 0; ix < nx; ++ix)
         {
            const int i = idx(ix, iy);
            const int j = idx(ix, jy);

            if (!contravariant_col_at_node(i, 1, nor)) continue;

            Uloc.GetRow(i, Ui);
            Uloc.GetRow(j, Uj);

            vol_flux.Eval(Ui, Uj, nor, *Tr, flux);

            for (int c = 0; c < num_eq; ++c) Div(i, c) += d * flux(c);
         }
      }

    //   MFEM_VERIFY(Div.IsFinite(), "Non-finite Div in DGSEM volume.");

      // Deposit into Zloc with w_i/detJ(i)
      for (int i = 0; i < dof; ++i)
      {
         const IntegrationPoint &ip = nodes.IntPoint(i);
         Tr->SetIntPoint(&ip);
         const DenseMatrix &J = Tr->Jacobian();
         const double detJ = J.Det();
         if (!std::isfinite(detJ) || std::abs(detJ) <= opt.small_detj) continue;

         const double inv_detJ = 1.0 / detJ;
         const double wi = m_lumped(i);

         for (int c = 0; c < num_eq; ++c)
         {
            Zloc(i, c) += wi * inv_detJ * Div(i, c);
         }
      }

    //   MFEM_VERIFY(Zloc.IsFinite(), "Non-finite Zloc after DGSEM volume.");
      z.SetSubVector(vdofs, Zloc.GetData());
   }
}

} // namespace mfem
