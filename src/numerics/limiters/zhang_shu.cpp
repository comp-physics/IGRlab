#include "numerics/limiters/zhang_shu.hpp"

#include "physics/euler/eos.hpp"
#include "limiters_common.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace
{

struct ElementFaceCacheEntry
{
  long mesh_sequence = -1;
  int ne = -1;
  std::vector<std::vector<int>> elem_faces;
};

inline const std::vector<std::vector<int>>& GetElementFaces(mfem::Mesh& mesh)
{
  static std::unordered_map<const mfem::Mesh*, ElementFaceCacheEntry> cache;
  auto& entry = cache[&mesh];
  if (entry.mesh_sequence == mesh.GetSequence() && entry.ne == mesh.GetNE())
  {
    return entry.elem_faces;
  }

  entry.mesh_sequence = mesh.GetSequence();
  entry.ne = mesh.GetNE();
  entry.elem_faces.clear();
  entry.elem_faces.resize(entry.ne);

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1 = -1, e2 = -1;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e1 >= 0 && e1 < entry.ne)
    {
      entry.elem_faces[e1].push_back(f);
    }
    if (e2 >= 0 && e2 < entry.ne)
    {
      entry.elem_faces[e2].push_back(f);
    }
  }

  return entry.elem_faces;
}

inline std::vector<int>
BuildFaceQuadratureOrders(const mfem::ParFiniteElementSpace& vfes,
                          int volume_qord,
                          const std::vector<int>& configured_face_qords)
{
  std::vector<int> qords;
  qords.reserve(configured_face_qords.size() + 2);

  auto push_unique_positive = [&](int qord)
  {
    if (qord <= 0)
    {
      return;
    }
    if (std::find(qords.begin(), qords.end(), qord) == qords.end())
    {
      qords.push_back(qord);
    }
  };

  push_unique_positive(volume_qord);
  for (const int qord : configured_face_qords)
  {
    push_unique_positive(qord);
  }

  if (qords.size() == 1)
  {
    push_unique_positive(2 * vfes.GetFE(0)->GetOrder() + 1);
  }

  return qords;
}

static inline double ThetaForPressure(const mfem::Vector& Ubar,
                                      const mfem::Vector& Up,
                                      int dim,
                                      double gamma,
                                      double p_min,
                                      double theta_max = 1.0,
                                      int iters = 12)
{
  mfem::Vector Utest(Ubar.Size());
  Utest = Ubar;
  Utest.Add(theta_max, Up);
  Utest.Add(-theta_max, Ubar);
  if (mfem::EulerPressureFromConservative(Utest,
                                          dim,
                                          gamma,
                                          /*rho_floor=*/1e-16) >= p_min)
  {
    return theta_max;
  }

  double lo = 0.0, hi = theta_max;
  for (int it = 0; it < iters; ++it)
  {
    const double mid = 0.5 * (lo + hi);
    Utest = Ubar;
    Utest.Add(mid, Up);
    Utest.Add(-mid, Ubar);
    const double p = mfem::EulerPressureFromConservative(Utest,
                                                         dim,
                                                         gamma,
                                                         /*rho_floor=*/1e-16);
    if (p >= p_min)
    {
      lo = mid;
    }
    else
    {
      hi = mid;
    }
  }
  return lo;
}

} // namespace

namespace mfem
{

void ApplyZhangShuPositivityLimiter(ParFiniteElementSpace& vfes,
                                    ParGridFunction& U,
                                    int dim,
                                    double gamma,
                                    double rho_min,
                                    double p_min,
                                    int quad_order,
                                    const std::vector<int>& face_quad_orders,
                                    int p_bisect_iters,
                                    ZhangShuLimiterElementDiagnostics* diagnostics)
{
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Positivity limiter assumes byNODES ordering.");

  const int num_eq = dim + 2;
  const int NE = vfes.GetNE();
  Mesh& mesh = *vfes.GetMesh();
  const auto& elem_faces = GetElementFaces(mesh);

  mfem::Array<int> vdofs;
  mfem::Vector elvec;      // element slice (size = dof*num_eq)
  mfem::DenseMatrix elmat; // (dof x num_eq) view of element DOFs
  mfem::Vector shape, Up(num_eq), Ubar(num_eq), row(num_eq);

  const int qord =
    (quad_order > 0) ? quad_order : (2 * vfes.GetFE(0)->GetOrder() + 2);
  const std::vector<int> limiter_face_qords =
    BuildFaceQuadratureOrders(vfes, qord, face_quad_orders);

  if (diagnostics)
  {
    diagnostics->SetSize(NE);
  }

  for (int i = 0; i < NE; i++)
  {
    const FiniteElement* fe = vfes.GetFE(i);
    const auto* fec_l2 = dynamic_cast<const L2_FECollection*>(vfes.FEColl());
    const bool enforce_nodes =
      (fec_l2 != nullptr && fec_l2->GetBasisType() == BasisType::GaussLobatto);
    ElementTransformation* Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);

    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // --- compute cell average ---
    dg_igr::limiters_detail::ElementAverage(*fe,
                                            *Tr,
                                            elmat,
                                            num_eq,
                                            qord,
                                            Ubar);

    // If the average itself is unphysical, collapse element to constant admissible state
    {
      const double rho_bar = std::max(Ubar(0), rho_min);
      double mom2_bar = 0.0;
      for (int d = 0; d < dim; ++d)
        mom2_bar += Ubar(1 + d) * Ubar(1 + d);
      const double k_bar = 0.5 * mom2_bar / std::max(rho_bar, 1e-16);
      double E_bar = Ubar(1 + dim);
      double p_bar = (gamma - 1.0) * (E_bar - k_bar);

      if (rho_bar <= rho_min || p_bar <= p_min)
      {
        Ubar(0) = rho_bar;
        E_bar = k_bar + p_min / (gamma - 1.0);
        Ubar(1 + dim) = E_bar;

        for (int j = 0; j < dof; ++j)
        {
          elmat.GetRow(j, row);
          row = Ubar;
          elmat.SetRow(j, row);
        }
        U.SetSubVector(vdofs, elmat.GetData());
        if (diagnostics)
        {
          diagnostics->active(i) = 1.0;
          diagnostics->theta_rho(i) = 0.0;
          diagnostics->theta_p(i) = 0.0;
        }
        continue; // done with this element
      }
    }

    // --- positivity points (use same IR as average) ---
    const IntegrationRule& ir = IntRules.Get(Tr->GetGeometryType(), qord);
    shape.SetSize(dof);

    // Step 1: density limiting
    double theta_rho = 1.0;
    auto rho_update = [&](const mfem::Vector& state)
    {
      const double rho = state(0);
      if (rho < rho_min)
      {
        const double num = Ubar(0) - rho_min;
        const double den = Ubar(0) - rho;
        if (den > 0.0)
        {
          theta_rho = std::min(theta_rho, num / den);
        }
        else
        {
          theta_rho = 0.0;
        }
      }
    };

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up);
      rho_update(Up);
    }
    for (const int face : elem_faces[i])
    {
      FaceElementTransformations* FTr = mesh.GetFaceElementTransformations(face);
      if (FTr == nullptr)
      {
        continue;
      }

      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(face, &e1, &e2);
      const bool elem_is_first = (e1 == i);
      const bool elem_is_second = (e2 == i);
      if (!elem_is_first && !elem_is_second)
      {
        continue;
      }

      for (const int fqord : limiter_face_qords)
      {
        const IntegrationRule& fir =
          IntRules.Get(FTr->GetGeometryType(), fqord);
        for (int q = 0; q < fir.GetNPoints(); ++q)
        {
          const IntegrationPoint& fip = fir.IntPoint(q);
          FTr->SetAllIntPoints(&fip);
          const IntegrationPoint& eip = elem_is_first
                                          ? FTr->GetElement1IntPoint()
                                          : FTr->GetElement2IntPoint();
          fe->CalcShape(eip, shape);
          elmat.MultTranspose(shape, Up);
          rho_update(Up);
        }
      }
    }
    if (enforce_nodes)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, Up);
        rho_update(Up);
      }
    }
    if (theta_rho < 1.0)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, row);
        dg_igr::limiters_detail::BlendRow(row, Ubar, theta_rho);
        elmat.SetRow(j, row);
      }
    }

    // Step 2: pressure limiting
    double theta_p = 1.0;
    auto p_update = [&](const mfem::Vector& state)
    {
      const double p = EulerPressureFromConservative(state,
                                                     dim,
                                                     gamma,
                                                     /*rho_floor=*/1e-16);
      if (p < p_min)
      {
        theta_p = std::min(theta_p,
                           ThetaForPressure(Ubar,
                                            state,
                                            dim,
                                            gamma,
                                            p_min,
                                            /*theta_max=*/theta_rho,
                                            p_bisect_iters));
      }
    };

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up); // current point state (after rho limit)
      p_update(Up);
    }
    for (const int face : elem_faces[i])
    {
      FaceElementTransformations* FTr = mesh.GetFaceElementTransformations(face);
      if (FTr == nullptr)
      {
        continue;
      }

      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(face, &e1, &e2);
      const bool elem_is_first = (e1 == i);
      const bool elem_is_second = (e2 == i);
      if (!elem_is_first && !elem_is_second)
      {
        continue;
      }

      for (const int fqord : limiter_face_qords)
      {
        const IntegrationRule& fir =
          IntRules.Get(FTr->GetGeometryType(), fqord);
        for (int q = 0; q < fir.GetNPoints(); ++q)
        {
          const IntegrationPoint& fip = fir.IntPoint(q);
          FTr->SetAllIntPoints(&fip);
          const IntegrationPoint& eip = elem_is_first
                                          ? FTr->GetElement1IntPoint()
                                          : FTr->GetElement2IntPoint();
          fe->CalcShape(eip, shape);
          elmat.MultTranspose(shape, Up);
          p_update(Up);
        }
      }
    }
    if (enforce_nodes)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, Up);
        p_update(Up);
      }
    }
    if (theta_p < 1.0)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, row);
        dg_igr::limiters_detail::BlendRow(row, Ubar, theta_p);
        elmat.SetRow(j, row);
      }
    }

    if (diagnostics)
    {
      diagnostics->theta_rho(i) = theta_rho;
      diagnostics->theta_p(i) = theta_p;
      diagnostics->active(i) = (theta_rho < 1.0 || theta_p < 1.0) ? 1.0 : 0.0;
    }

    U.SetSubVector(vdofs, elmat.GetData());
  }
}

} // namespace mfem
