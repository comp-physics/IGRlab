#include "numerics/dgsem/dgsem_operator.hpp"

#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_entropy_flux.hpp"
#include "numerics/dgsem/dgsem_metric.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"
#include "numerics/dgsem/dgsem_volume_flux.hpp"
#include "physics/igr/igr_flux.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{
namespace dgsem
{

const BdrHyperbolicFluxIntegrator*
DGSEMOperator::FindBdrIntegrator(int attr) const
{
  if (attr <= 0)
  {
    return nullptr;
  }
  for (size_t i = 0; i < op_->ext_bdr_int_.size(); ++i)
  {
    const auto* marker = op_->ext_bdr_markers_[i].get();
    if (!marker || attr > marker->Size())
    {
      continue;
    }
    if ((*marker)[attr - 1] == 0)
    {
      continue;
    }
    return dynamic_cast<const BdrHyperbolicFluxIntegrator*>(
      op_->ext_bdr_int_[i].get());
  }
  return nullptr;
}

real_t DGSEMOperator::AssembleSurface(const Vector& x, Vector& z) const
{
  MFEM_VERIFY(DGSEMOperatorSupportsDimension(op_->dim),
              "DGSEM legacy surface oracle supports only 1D/2D.");
  MFEM_VERIFY(op_->num_flux_,
              "DGSEM surface assembly requires a NumericalFlux; call "
              "DGHyperbolicConservationLaws::SetNumericalFlux().");

  z = 0.0;
  real_t max_face_speed = 0.0;

  Mesh& mesh = *op_->vfes.GetMesh();
  auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&op_->vfes);
  ParMesh* pmesh = pfes ? pfes->GetParMesh() : nullptr;
  const int nfaces = mesh.GetNumFaces();
  Vector state1(op_->num_equations);
  Vector state2(op_->num_equations);
  Vector fluxN(op_->num_equations);
  Vector nor(op_->dim);
  ParGridFunction x_par;
  const Vector* face_nbr = nullptr;
  int n_shared_faces = 0;
  std::vector<std::vector<dgsem::ContravariantMetric2D>> metric_cache(
    static_cast<std::size_t>(mesh.GetNE()));
  std::vector<char> metric_cache_ready(static_cast<std::size_t>(mesh.GetNE()),
                                       0);
  if (pmesh && pmesh->GetNSharedFaces() > 0)
  {
    MFEM_VERIFY(x.Size() == pfes->GetVSize(),
                "DGSEM shared-face assembly expects local DOF vector.");
    x_par.MakeRef(pfes, const_cast<Vector&>(x), 0);
    x_par.ExchangeFaceNbrData();
    face_nbr = &x_par.FaceNbrData();
    n_shared_faces = pmesh->GetNSharedFaces();
  }

  auto find_node = [](const Vector& nodes, double coord) -> int
  {
    int best = 0;
    double best_err = std::abs(coord - nodes(0));
    for (int i = 1; i < nodes.Size(); ++i)
    {
      const double err = std::abs(coord - nodes(i));
      if (err < best_err)
      {
        best_err = err;
        best = i;
      }
    }
    MFEM_VERIFY(best_err < 1.0e-10,
                "DGSEM face point does not match a nodal coordinate.");
    return best;
  };
  auto build_lex_to_local =
    [&](const FiniteElement& fe, int n1d, std::vector<int>& lex_to_local)
  {
    const int dof = fe.GetDof();
    const std::vector<std::array<int, 3>>& dof_to_ijk =
      mfem::dgsem::GetTensorDofToIJK(fe);
    lex_to_local.assign(dof, -1);
    for (int local = 0; local < dof; ++local)
    {
      const int i = dof_to_ijk[local][0];
      const int j = dof_to_ijk[local][1];
      const int lex = (op_->dim == 1) ? i : i + n1d * j;
      lex_to_local[lex] = local;
    }
  };
  auto set_nodal_normal = [&](FaceElementTransformations& face_tr,
                              int element,
                              const FiniteElement& fe,
                              int local,
                              const IntegrationPoint& element_ip)
  {
    if (op_->dim == 1)
    {
      nor(0) = (element_ip.x - 0.5) * 2.0;
      return;
    }
    if (!metric_cache_ready[static_cast<std::size_t>(element)])
    {
      dgsem::BuildContravariantMetrics2D(
        fe,
        *op_->vfes.GetElementTransformation(element),
        metric_cache[static_cast<std::size_t>(element)]);
      metric_cache_ready[static_cast<std::size_t>(element)] = 1;
    }

    const auto& metric = metric_cache[static_cast<std::size_t>(element)]
                                     [static_cast<std::size_t>(local)];
    dgsem::BuildNodalFaceNormal2D(
      metric, face_tr.Loc1.Transf.Jacobian(), element_ip, nor);
  };

  for (int f = 0; f < nfaces; ++f)
  {
    FaceElementTransformations* Tr = mesh.GetFaceElementTransformations(f);
    if (!Tr)
    {
      continue;
    }

    const int e1 = Tr->Elem1No;
    const int e2 = Tr->Elem2No;
    if (e2 < 0)
    {
      continue;
    }

    const FiniteElement& fe1 = *op_->vfes.GetFE(e1);
    const int dof1 = fe1.GetDof();
    const int order = fe1.GetOrder();
    const auto& basis1d = mfem::dgsem::GetLGLBasis1D(order);
    const Vector& nodes1d = basis1d.nodes;
    const int n1d = basis1d.n;

    Array<int> vdofs1;
    Vector xval1;
    op_->vfes.GetElementVDofs(e1, vdofs1);
    x.GetSubVector(vdofs1, xval1);
    DenseMatrix Xmat1(xval1.GetData(), dof1, op_->num_equations);

    std::vector<int> lex_to_local1;
    build_lex_to_local(fe1, n1d, lex_to_local1);

    DenseMatrix elvect1(dof1, op_->num_equations);
    elvect1 = 0.0;

    DenseMatrix Xmat2;
    DenseMatrix elvect2;
    Array<int> vdofs2;
    Vector xval2;
    std::vector<int> lex_to_local2;
    int dof2 = 0;

    if (e2 >= 0)
    {
      const FiniteElement& fe2 = *op_->vfes.GetFE(e2);
      dof2 = fe2.GetDof();
      MFEM_VERIFY(fe2.GetOrder() == order,
                  "DGSEM surface assembly requires matching orders on faces.");

      op_->vfes.GetElementVDofs(e2, vdofs2);
      x.GetSubVector(vdofs2, xval2);
      Xmat2.UseExternalData(xval2.GetData(), dof2, op_->num_equations);

      build_lex_to_local(fe2, n1d, lex_to_local2);
      elvect2.SetSize(dof2, op_->num_equations);
      elvect2 = 0.0;
    }

    const IntegrationRule& ir = (op_->dim == 1)
                                  ? IntRules.Get(Geometry::POINT, 1)
                                  : mfem::dgsem::GetLGLFaceRule(order);

    for (int ip_id = 0; ip_id < ir.GetNPoints(); ++ip_id)
    {
      const IntegrationPoint& ip = ir.IntPoint(ip_id);
      Tr->SetAllIntPoints(&ip);

      const IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
      const int i1 = find_node(nodes1d, ip1.x);
      const int j1 = (op_->dim > 1) ? find_node(nodes1d, ip1.y) : 0;
      const int local1 = lex_to_local1[(op_->dim == 1) ? i1 : i1 + n1d * j1];

      Xmat1.GetRow(local1, state1);

      set_nodal_normal(*Tr, e1, fe1, local1, ip1);

      const IntegrationPoint& ip2 = Tr->GetElement2IntPoint();
      const int i2 = find_node(nodes1d, ip2.x);
      const int j2 = (op_->dim > 1) ? find_node(nodes1d, ip2.y) : 0;
      const int local2 = lex_to_local2[(op_->dim == 1) ? i2 : i2 + n1d * j2];

      Xmat2.GetRow(local2, state2);

      const real_t speed = op_->num_flux_->Eval(state1, state2, nor, *Tr, fluxN);
      max_face_speed = std::max(max_face_speed, speed);

      for (int eq = 0; eq < op_->num_equations; ++eq)
      {
        elvect1(local1, eq) += -ip.weight * fluxN(eq);
        elvect2(local2, eq) += +ip.weight * fluxN(eq);
      }
    }

    z.AddElementVector(vdofs1, elvect1.GetData());
    z.AddElementVector(vdofs2, elvect2.GetData());
  }

  if (pmesh && n_shared_faces > 0)
  {
    MFEM_VERIFY(face_nbr, "DGSEM shared-face assembly requires neighbor data.");

    for (int sf = 0; sf < n_shared_faces; ++sf)
    {
      FaceElementTransformations* Tr =
        pmesh->GetSharedFaceTransformations(sf, /*fill2=*/true);
      if (!Tr)
      {
        continue;
      }

      const int e1 = Tr->Elem1No;
      const int e2 = Tr->Elem2No - pmesh->GetNE();
      MFEM_VERIFY(e2 >= 0, "DGSEM shared face neighbor index is invalid.");

      const FiniteElement& fe1 = *op_->vfes.GetFE(e1);
      const FiniteElement& fe2 = *pfes->GetFaceNbrFE(e2);
      const int dof1 = fe1.GetDof();
      const int dof2 = fe2.GetDof();
      const int order = fe1.GetOrder();
      MFEM_VERIFY(fe2.GetOrder() == order,
                  "DGSEM shared-face assembly requires matching orders.");

      const auto& basis1d = mfem::dgsem::GetLGLBasis1D(order);
      const Vector& nodes1d = basis1d.nodes;
      const int n1d = basis1d.n;

      Array<int> vdofs1;
      Vector xval1;
      op_->vfes.GetElementVDofs(e1, vdofs1);
      x.GetSubVector(vdofs1, xval1);
      DenseMatrix Xmat1(xval1.GetData(), dof1, op_->num_equations);

      Array<int> vdofs2;
      Vector xval2;
      pfes->GetFaceNbrElementVDofs(e2, vdofs2);
      face_nbr->GetSubVector(vdofs2, xval2);
      DenseMatrix Xmat2(xval2.GetData(), dof2, op_->num_equations);

      std::vector<int> lex_to_local1;
      std::vector<int> lex_to_local2;
      build_lex_to_local(fe1, n1d, lex_to_local1);
      build_lex_to_local(fe2, n1d, lex_to_local2);

      DenseMatrix elvect1(dof1, op_->num_equations);
      elvect1 = 0.0;

      const IntegrationRule& ir = (op_->dim == 1)
                                    ? IntRules.Get(Geometry::POINT, 1)
                                    : mfem::dgsem::GetLGLFaceRule(order);

      for (int ip_id = 0; ip_id < ir.GetNPoints(); ++ip_id)
      {
        const IntegrationPoint& ip = ir.IntPoint(ip_id);
        Tr->SetAllIntPoints(&ip);

        const IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
        const int i1 = find_node(nodes1d, ip1.x);
        const int j1 = (op_->dim > 1) ? find_node(nodes1d, ip1.y) : 0;
        const int local1 = lex_to_local1[(op_->dim == 1) ? i1 : i1 + n1d * j1];

        Xmat1.GetRow(local1, state1);

        set_nodal_normal(*Tr, e1, fe1, local1, ip1);

        const IntegrationPoint& ip2 = Tr->GetElement2IntPoint();
        const int i2 = find_node(nodes1d, ip2.x);
        const int j2 = (op_->dim > 1) ? find_node(nodes1d, ip2.y) : 0;
        const int local2 = lex_to_local2[(op_->dim == 1) ? i2 : i2 + n1d * j2];

        Xmat2.GetRow(local2, state2);

        const real_t speed =
          op_->num_flux_->Eval(state1, state2, nor, *Tr, fluxN);
        max_face_speed = std::max(max_face_speed, speed);

        for (int eq = 0; eq < op_->num_equations; ++eq)
        {
          elvect1(local1, eq) += -ip.weight * fluxN(eq);
        }
      }

      z.AddElementVector(vdofs1, elvect1.GetData());
    }
  }

  const int nbdr = mesh.GetNBE();
  for (int b = 0; b < nbdr; ++b)
  {
    FaceElementTransformations* Tr = mesh.GetBdrFaceTransformations(b);
    if (!Tr)
    {
      continue;
    }

    const int e1 = Tr->Elem1No;
    const FiniteElement& fe1 = *op_->vfes.GetFE(e1);
    const int dof1 = fe1.GetDof();
    const int order = fe1.GetOrder();
    const auto& basis1d = mfem::dgsem::GetLGLBasis1D(order);
    const Vector& nodes1d = basis1d.nodes;
    const int n1d = basis1d.n;

    Array<int> vdofs1;
    Vector xval1;
    op_->vfes.GetElementVDofs(e1, vdofs1);
    x.GetSubVector(vdofs1, xval1);
    DenseMatrix Xmat1(xval1.GetData(), dof1, op_->num_equations);

    std::vector<int> lex_to_local1;
    build_lex_to_local(fe1, n1d, lex_to_local1);

    DenseMatrix elvect1(dof1, op_->num_equations);
    elvect1 = 0.0;

    const IntegrationRule& ir = (op_->dim == 1)
                                  ? IntRules.Get(Geometry::POINT, 1)
                                  : mfem::dgsem::GetLGLFaceRule(order);

    for (int ip_id = 0; ip_id < ir.GetNPoints(); ++ip_id)
    {
      const IntegrationPoint& ip = ir.IntPoint(ip_id);
      Tr->SetAllIntPoints(&ip);

      const IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
      const int i1 = find_node(nodes1d, ip1.x);
      const int j1 = (op_->dim > 1) ? find_node(nodes1d, ip1.y) : 0;
      const int local1 = lex_to_local1[(op_->dim == 1) ? i1 : i1 + n1d * j1];

      Xmat1.GetRow(local1, state1);

      set_nodal_normal(*Tr, e1, fe1, local1, ip1);

      const BdrHyperbolicFluxIntegrator* bdr = FindBdrIntegrator(Tr->Attribute);
      MFEM_VERIFY(bdr, "DGSEM boundary face is missing a boundary integrator.");
      bdr->GetBoundaryState().Eval(state1, nor, *Tr, state2);

      const real_t speed = op_->num_flux_->Eval(state1, state2, nor, *Tr, fluxN);
      max_face_speed = std::max(max_face_speed, speed);

      for (int eq = 0; eq < op_->num_equations; ++eq)
      {
        elvect1(local1, eq) += -ip.weight * fluxN(eq);
      }
    }

    z.AddElementVector(vdofs1, elvect1.GetData());
  }

  return max_face_speed;
}

real_t DGSEMOperator::AssembleVolume(const Vector& x, Vector& z) const
{
  MFEM_VERIFY(DGSEMOperatorSupportsDimension(op_->dim),
              "DGSEM legacy volume oracle supports only 1D/2D.");
  MFEM_VERIFY(!op_->weakdiv.empty(),
              "DGSEM volume mode requires preassembled weak divergence "
              "(face-only form).");

  Vector state;
  DenseMatrix flux;      // contravariant flux (num_eq × (dim*dof))
  DenseMatrix Xmat;      // element states (dof × num_eq)
  DenseMatrix volume;    // volume contribution (dof × num_eq)
  DenseMatrix phys_flux; // physical flux at node (num_eq × dim)
  DenseMatrix adj;       // adjugate of Jacobian (dim × dim)
  std::vector<double> adj_storage;
  std::vector<double> sigma_nodes;
  std::vector<dgsem::ContravariantMetric2D> discrete_metrics;
  std::vector<int> lex_to_local;
  const FluxFunction& fluxFunction = op_->formIntegrator->GetFluxFunction();
  const FluxWithIGR* flux_igr = dynamic_cast<const FluxWithIGR*>(&fluxFunction);
  const bool use_entropy_flux =
    (op_->GetDGSEMVolumeFlux() != DGSEMVolumeFlux::Central);
  const DGSEMVolumeDissipation vol_diss_mode = op_->GetDGSEMVolumeDissipation();
  const bool use_volume_dissipation =
    (vol_diss_mode != DGSEMVolumeDissipation::None);
  const dgsem::EntropyJacobianMode entropy_jacobian_mode =
    op_->GetDGSEMEntropyJacobianAnalytic()
      ? dgsem::EntropyJacobianMode::Analytic
      : dgsem::EntropyJacobianMode::FiniteDifference;
  dgsem::EntropyFluxType entropy_type = dgsem::EntropyFluxType::Ranocha;
  if (op_->GetDGSEMVolumeFlux() == DGSEMVolumeFlux::Chandrashekar)
  {
    entropy_type = dgsem::EntropyFluxType::Chandrashekar;
  }
  const double gamma = op_->GetDGSEMGamma();
  Array<int> vdofs;
  Vector xval;

  int max_dof = 0;
  for (int e = 0; e < op_->vfes.GetNE(); ++e)
  {
    max_dof = std::max(max_dof, op_->vfes.GetFE(e)->GetDof());
  }
  lex_to_local.reserve(max_dof);
  if (op_->dim == 2 || use_entropy_flux || use_volume_dissipation)
  {
    adj_storage.reserve(max_dof * op_->dim * op_->dim);
    if (flux_igr)
    {
      sigma_nodes.reserve(max_dof);
    }
  }

  Vector flux_ij(op_->num_equations);
  Vector diss_ij(op_->num_equations);
  Vector state_i(op_->num_equations);
  Vector state_j(op_->num_equations);
  Vector contravariant_dir(op_->dim);
  Vector flux_dir(op_->num_equations);

  real_t max_vol_speed = 0.0;

  for (int e = 0; e < op_->vfes.GetNE(); ++e)
  {
    ElementTransformation* Tr = op_->vfes.GetElementTransformation(e);
    const FiniteElement& fe = *op_->vfes.GetFE(e);
    const int dof = fe.GetDof();
    const int order = fe.GetOrder();
    const auto& basis1d = mfem::dgsem::GetLGLBasis1D(order);
    const DenseMatrix& dsplit = basis1d.Dsplit;
    const Vector& weights = basis1d.weights;
    const int n1d = basis1d.n;

    MFEM_VERIFY(dof == mfem::TensorBasisElement::Pow(n1d, op_->dim),
                "DGSEM requires tensor-product elements (order "
                  << order << ", dim " << op_->dim << ").");

    const std::vector<std::array<int, 3>>& dof_to_ijk =
      mfem::dgsem::GetTensorDofToIJK(fe);

    lex_to_local.assign(dof, 0);
    for (int local = 0; local < dof; ++local)
    {
      const int i = dof_to_ijk[local][0];
      const int j = dof_to_ijk[local][1];
      const int k = dof_to_ijk[local][2];
      const int lex =
        (op_->dim == 1) ? i
                       : (op_->dim == 2 ? i + n1d * j : i + n1d * (j + n1d * k));
      lex_to_local[lex] = local;
    }

    op_->vfes.GetElementVDofs(e, vdofs);
    x.GetSubVector(vdofs, xval);
    Xmat.UseExternalData(xval.GetData(), dof, op_->num_equations);

    // Build contravariant flux G = adj(J) * F(U) at nodes.
    flux.SetSize(op_->num_equations, op_->dim * dof);
    phys_flux.SetSize(op_->num_equations, op_->dim);
    adj.SetSize(op_->dim);
    if (op_->dim == 2 || use_entropy_flux || use_volume_dissipation)
    {
      adj_storage.resize(dof * op_->dim * op_->dim);
      if (flux_igr)
      {
        sigma_nodes.resize(dof);
      }
    }
    if (op_->dim == 2)
    {
      dgsem::BuildContravariantMetrics2D(
        fe, *Tr, discrete_metrics);
    }
    const IntegrationRule& nodes = fe.GetNodes();
    MFEM_ASSERT(nodes.Size() == dof,
                "FE is not nodal or GetNodes() != dof; "
                "need to evaluate at an appropriate rule.");

    for (int j = 0; j < dof; ++j)
    {
      const IntegrationPoint& ip = nodes.IntPoint(j);
      Tr->SetIntPoint(&ip);

      Xmat.GetRow(j, state);
      const real_t mcs = fluxFunction.ComputeFlux(state, *Tr, phys_flux);
      max_vol_speed = std::max(max_vol_speed, mcs);

      if (op_->dim == 2)
      {
        const auto& metric =
          discrete_metrics[static_cast<std::size_t>(j)];
        for (int d = 0; d < op_->dim; ++d)
        {
          for (int k = 0; k < op_->dim; ++k)
          {
            adj(d, k) = metric(d, k);
          }
        }
      }
      else
      {
        mfem::CalcAdjugate(Tr->Jacobian(), adj);
      }

      if (use_entropy_flux || use_volume_dissipation)
      {
        if (flux_igr)
        {
          sigma_nodes[j] = flux_igr->GetSigmaAtPoint(*Tr);
        }
        for (int d = 0; d < op_->dim; ++d)
        {
          for (int k = 0; k < op_->dim; ++k)
          {
            const int idx = (j * op_->dim + d) * op_->dim + k;
            adj_storage[idx] = adj(d, k);
          }
        }
      }

      for (int d = 0; d < op_->dim; ++d)
      {
        for (int eq = 0; eq < op_->num_equations; ++eq)
        {
          double g = 0.0;
          for (int k = 0; k < op_->dim; ++k)
          {
            g += phys_flux(eq, k) * adj(d, k);
          }
          flux(eq, j * op_->dim + d) = g;
        }
      }
    }

    volume.SetSize(dof, op_->num_equations);
    volume = 0.0;

    const double vol_sign = -1.0; // match DGSEM split-form sign convention
    auto adj_at = [&](int node, int d, int k) -> double
    {
      const int idx = (node * op_->dim + d) * op_->dim + k;
      return adj_storage[idx];
    };
    auto eval_ec_contravariant =
      [&](int local_i, int local_j, int dir, Vector& out)
    {
      Xmat.GetRow(local_i, state_i);
      Xmat.GetRow(local_j, state_j);
      const double sigma_i =
        (flux_igr && !sigma_nodes.empty()) ? sigma_nodes[local_i] : 0.0;
      const double sigma_j =
        (flux_igr && !sigma_nodes.empty()) ? sigma_nodes[local_j] : 0.0;
      out = 0.0;
      for (int k = 0; k < op_->dim; ++k)
      {
        dgsem::EntropyConservativeFlux(state_i,
                                       state_j,
                                       k,
                                       gamma,
                                       entropy_type,
                                       flux_dir,
                                       sigma_i,
                                       sigma_j);
        const double adj_avg =
          0.5 * (adj_at(local_i, dir, k) + adj_at(local_j, dir, k));
        for (int eq = 0; eq < op_->num_equations; ++eq)
        {
          out(eq) += adj_avg * flux_dir(eq);
        }
      }
    };
    auto eval_volume_dissipation =
      [&](int local_i, int local_j, int dir, Vector& out)
    {
      out = 0.0;
      if (!use_volume_dissipation)
      {
        return;
      }

      Xmat.GetRow(local_i, state_i);
      Xmat.GetRow(local_j, state_j);
      const double sigma_i =
        (flux_igr && !sigma_nodes.empty()) ? sigma_nodes[local_i] : 0.0;
      const double sigma_j =
        (flux_igr && !sigma_nodes.empty()) ? sigma_nodes[local_j] : 0.0;

      for (int k = 0; k < op_->dim; ++k)
      {
        contravariant_dir(k) =
          0.5 * (adj_at(local_i, dir, k) + adj_at(local_j, dir, k));
      }
      const double lf_i =
        dgsem::EulerDirectionalLaxFriedrichsScale(state_i,
                                                  contravariant_dir,
                                                  op_->dim,
                                                  gamma,
                                                  sigma_i);
      const double lf_j =
        dgsem::EulerDirectionalLaxFriedrichsScale(state_j,
                                                  contravariant_dir,
                                                  op_->dim,
                                                  gamma,
                                                  sigma_j);
      const double scale = std::max(lf_i, lf_j);

      if (vol_diss_mode == DGSEMVolumeDissipation::LLF)
      {
        for (int eq = 0; eq < op_->num_equations; ++eq)
        {
          out(eq) = -0.5 * scale * (state_j(eq) - state_i(eq));
        }
        return;
      }

      MFEM_ASSERT(vol_diss_mode == DGSEMVolumeDissipation::EntropyVarsLF,
                  "Unexpected DGSEM volume dissipation mode in "
                  "entropy-variable LF branch.");
      const bool valid_i =
        dgsem::EulerStateIsValid(state_i, op_->dim, gamma, sigma_i);
      const bool valid_j =
        dgsem::EulerStateIsValid(state_j, op_->dim, gamma, sigma_j);
      if (valid_i && valid_j)
      {
        dgsem::EntropyVariableLFDissipation(state_i,
                                            state_j,
                                            gamma,
                                            scale,
                                            out,
                                            sigma_i,
                                            sigma_j,
                                            entropy_jacobian_mode);
      }
      else
      {
        // Conservative fallback when entropy variables are undefined.
        for (int eq = 0; eq < op_->num_equations; ++eq)
        {
          out(eq) = -0.5 * scale * (state_j(eq) - state_i(eq));
        }
      }
    };
    auto eval_pair_flux = [&](int local_i, int local_j, int dir, Vector& out)
    {
      if (use_entropy_flux)
      {
        eval_ec_contravariant(local_i, local_j, dir, out);
      }
      else
      {
        mfem::dgsem::SymmetricAverageContravariantFlux(flux,
                                                       local_i,
                                                       local_j,
                                                       dir,
                                                       op_->dim,
                                                       out);
      }
      eval_volume_dissipation(local_i, local_j, dir, diss_ij);
      if (use_volume_dissipation)
      {
        out += diss_ij;
      }
    };

    if (op_->dim == 1)
    {
      for (int i = 0; i < n1d; ++i)
      {
        const int local_i = lex_to_local[i];
        for (int ii = i + 1; ii < n1d; ++ii)
        {
          const int local_ii = lex_to_local[ii];
          eval_pair_flux(local_i, local_ii, /*dir=*/0, flux_ij);
          for (int eq = 0; eq < op_->num_equations; ++eq)
          {
            volume(local_i, eq) += vol_sign * dsplit(i, ii) * flux_ij(eq);
            volume(local_ii, eq) += vol_sign * dsplit(ii, i) * flux_ij(eq);
          }
        }
      }
    }
    else
    {
      for (int j = 0; j < n1d; ++j)
      {
        for (int i = 0; i < n1d; ++i)
        {
          const int local_i = lex_to_local[i + n1d * j];
          for (int ii = i + 1; ii < n1d; ++ii)
          {
            const int local_ii = lex_to_local[ii + n1d * j];
            eval_pair_flux(local_i, local_ii, /*dir=*/0, flux_ij);
            for (int eq = 0; eq < op_->num_equations; ++eq)
            {
              volume(local_i, eq) += vol_sign * dsplit(i, ii) * flux_ij(eq);
              volume(local_ii, eq) += vol_sign * dsplit(ii, i) * flux_ij(eq);
            }
          }
        }
      }
      for (int i = 0; i < n1d; ++i)
      {
        for (int j = 0; j < n1d; ++j)
        {
          const int local_j = lex_to_local[i + n1d * j];
          for (int jj = j + 1; jj < n1d; ++jj)
          {
            const int local_jj = lex_to_local[i + n1d * jj];
            eval_pair_flux(local_j, local_jj, /*dir=*/1, flux_ij);
            for (int eq = 0; eq < op_->num_equations; ++eq)
            {
              volume(local_j, eq) += vol_sign * dsplit(j, jj) * flux_ij(eq);
              volume(local_jj, eq) += vol_sign * dsplit(jj, j) * flux_ij(eq);
            }
          }
        }
      }
    }

    for (int local = 0; local < dof; ++local)
    {
      const int i = dof_to_ijk[local][0];
      const int j = dof_to_ijk[local][1];
      double w = weights(i);
      if (op_->dim > 1)
      {
        w *= weights(j);
      }
      for (int eq = 0; eq < op_->num_equations; ++eq)
      {
        volume(local, eq) *= w;
      }
    }

    z.AddElementVector(vdofs, volume.GetData());
  }

  return max_vol_speed;
}

void DGSEMOperator::ApplyInverseMass(const Vector& z, Vector& y) const
{
  MFEM_VERIFY(DGSEMOperatorSupportsDimension(op_->dim),
              "DGSEM legacy inverse-mass oracle supports only 1D/2D.");
  DenseMatrix Zmat_elem, Ymat;
  Vector zval_elem;
  Array<int> vdofs_elem;
  for (int e = 0; e < op_->vfes.GetNE(); ++e)
  {
    const int dof = op_->vfes.GetFE(e)->GetDof();
    op_->vfes.GetElementVDofs(e, vdofs_elem);
    z.GetSubVector(vdofs_elem, zval_elem);
    Zmat_elem.UseExternalData(zval_elem.GetData(), dof, op_->num_equations);

    Ymat.SetSize(dof, op_->num_equations);
    mfem::Mult(op_->invmass[e], Zmat_elem, Ymat);
    y.SetSubVector(vdofs_elem, Ymat.GetData());
  }
}

real_t DGSEMOperator::AssembleResidual(const Vector& x, Vector& y) const
{
  MFEM_VERIFY(DGSEMOperatorSupportsDimension(op_->dim),
              "DGSEM legacy residual oracle supports only 1D/2D.");
  Vector& z = op_->z;
  const real_t max_face_speed = AssembleSurface(x, z);
  const real_t max_vol_speed = AssembleVolume(x, z);
  ApplyInverseMass(z, y);
  return std::max(max_face_speed, max_vol_speed);
}

} // namespace dgsem
} // namespace mfem
