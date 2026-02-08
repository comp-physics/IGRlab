#include "numerics/dgsem/dgsem_operator.hpp"

#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/bc/boundary_state.hpp"
#include "utils/math/constants.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace dg_igr_tests
{

namespace
{

struct FaceNodeAudit
{
  double mismatched = 0.0;
  double total = 0.0;
  double shape_local_err_max = 0.0;
  double shape_peak_err_max = 0.0;
  double state_interp_err_max = 0.0;
};

struct BoundaryTransformAudit
{
  double max_ip_diff = 0.0;
  double max_nor_diff = 0.0;
};

int FindNodeIndex(const mfem::Vector& nodes, double coord, double tol)
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
  MFEM_VERIFY(best_err < tol, "DGSEM face node coordinate mismatch.");
  return best;
}

BoundaryTransformAudit AuditBoundaryTransforms(mfem::ParMesh& pmesh, int order)
{
  BoundaryTransformAudit local;
  const int dim = pmesh.Dimension();
  const mfem::IntegrationRule& ir = mfem::dgsem::GetLGLFaceRule(order);
  mfem::Vector nor_face(dim);
  mfem::Vector nor_bdr(dim);

  for (int b = 0; b < pmesh.GetNBE(); ++b)
  {
    mfem::FaceElementTransformations* Tr_bdr = pmesh.GetBdrFaceTransformations(b);
    if (!Tr_bdr) { continue; }

    const int face = pmesh.GetBdrElementFaceIndex(b);
    mfem::FaceElementTransformations* Tr_face =
        pmesh.GetFaceElementTransformations(face);
    if (!Tr_face) { continue; }

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr_bdr->SetAllIntPoints(&ip);
      Tr_face->SetAllIntPoints(&ip);

      const mfem::IntegrationPoint& ip_bdr = Tr_bdr->GetElement1IntPoint();
      const mfem::IntegrationPoint& ip_face = Tr_face->GetElement1IntPoint();

      double ip_diff = std::abs(ip_bdr.x - ip_face.x);
      if (dim > 1)
      {
        const double dy = ip_bdr.y - ip_face.y;
        ip_diff = std::max(ip_diff, std::abs(dy));
      }
      if (dim > 2)
      {
        const double dz = ip_bdr.z - ip_face.z;
        ip_diff = std::max(ip_diff, std::abs(dz));
      }
      local.max_ip_diff = std::max(local.max_ip_diff, ip_diff);

      if (dim == 1)
      {
        nor_face(0) = (ip_face.x - 0.5) * 2.0;
        nor_bdr(0) = (ip_bdr.x - 0.5) * 2.0;
      }
      else
      {
        mfem::CalcOrtho(Tr_face->Jacobian(), nor_face);
        mfem::CalcOrtho(Tr_bdr->Jacobian(), nor_bdr);
      }
      mfem::Vector diff(nor_face.Size());
      diff = nor_face;
      diff -= nor_bdr;
      local.max_nor_diff = std::max(local.max_nor_diff, diff.Norml2());
    }
  }

  BoundaryTransformAudit global;
  global.max_ip_diff = GlobalMax(local.max_ip_diff);
  global.max_nor_diff = GlobalMax(local.max_nor_diff);
  return global;
}

FaceNodeAudit AuditDGSEMFaceNodes(const mfem::ParGridFunction& U,
                                  mfem::ParMesh& pmesh,
                                  int order)
{
  FaceNodeAudit local;
  const bool dump = EnvEnabled("DG_IGR_DGSEM_FACE_AUDIT_DUMP");
  int dump_remaining = 32;
  const int dim = pmesh.Dimension();
  const int num_eq = U.VectorDim();
  const double tol = 1.0e-10;
  const auto& basis1d = mfem::dgsem::GetLGLBasis1D(order);
  const mfem::Vector& nodes1d = basis1d.nodes;
  const int n1d = basis1d.n;
  const mfem::IntegrationRule& ir = mfem::dgsem::GetLGLFaceRule(order);

  const auto* vfes = dynamic_cast<const mfem::ParFiniteElementSpace*>(U.ParFESpace());
  MFEM_VERIFY(vfes, "AuditDGSEMFaceNodes requires a ParFiniteElementSpace.");

  for (int b = 0; b < pmesh.GetNBE(); ++b)
  {
    mfem::FaceElementTransformations* Tr = pmesh.GetBdrFaceTransformations(b);
    if (!Tr) { continue; }

    int info = 0;
    int elem_id = -1;
    pmesh.GetBdrElementAdjacentElement(b, elem_id, info);
    const int face_id = mfem::Mesh::DecodeFaceInfoLocalIndex(info);
    const int face_orientation = mfem::Mesh::DecodeFaceInfoOrientation(info);

    const int e1 = Tr->Elem1No;
    const mfem::FiniteElement& fe1 = *vfes->GetFE(e1);
    const int dof1 = fe1.GetDof();

    mfem::Array<int> vdofs1;
    vfes->GetElementVDofs(e1, vdofs1);
    mfem::Vector xval1;
    U.GetSubVector(vdofs1, xval1);
    mfem::DenseMatrix Xmat1(xval1.GetData(), dof1, num_eq);

    const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(fe1);
    std::vector<int> lex_to_local(dof1, -1);
    for (int local_idx = 0; local_idx < dof1; ++local_idx)
    {
      const int i = dof_to_ijk[local_idx][0];
      const int j = dof_to_ijk[local_idx][1];
      const int lex = (dim == 1) ? i : i + n1d * j;
      lex_to_local[lex] = local_idx;
    }

    mfem::Vector shape(dof1);
    mfem::Vector state_from_shape(num_eq);
    mfem::Vector state_node(num_eq);
    mfem::Vector diff(num_eq);

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetAllIntPoints(&ip);
      const mfem::IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
      const int i = FindNodeIndex(nodes1d, ip1.x, tol);
      const int j = (dim > 1) ? FindNodeIndex(nodes1d, ip1.y, tol) : 0;
      const int lex = (dim == 1) ? i : i + n1d * j;
      const int local_idx = lex_to_local[lex];

      fe1.CalcShape(ip1, shape);

      int max_idx = 0;
      double max_val = shape(0);
      for (int k = 1; k < dof1; ++k)
      {
        if (shape(k) > max_val)
        {
          max_val = shape(k);
          max_idx = k;
        }
      }

      if (local_idx != max_idx)
      {
        local.mismatched += 1.0;
        if (dump && dump_remaining > 0)
        {
          std::cout << "[AUDIT] bdr=" << b
                    << " elem=" << e1
                    << " face_id=" << face_id
                    << " orient=" << face_orientation
                    << " ip=(" << FormatSci(ip1.x) << "," << FormatSci(ip1.y) << ")"
                    << " local_idx=" << local_idx
                    << " max_idx=" << max_idx
                    << " shape_local=" << FormatSci(shape(local_idx))
                    << " shape_max=" << FormatSci(max_val)
                    << "\n";
          --dump_remaining;
        }
      }
      local.total += 1.0;

      local.shape_local_err_max =
          std::max(local.shape_local_err_max, std::abs(1.0 - shape(local_idx)));
      local.shape_peak_err_max =
          std::max(local.shape_peak_err_max, std::abs(1.0 - max_val));

      Xmat1.GetRow(local_idx, state_node);
      Xmat1.MultTranspose(shape, state_from_shape);
      diff = state_from_shape;
      diff -= state_node;
      local.state_interp_err_max =
          std::max(local.state_interp_err_max, diff.Norml2());
    }
  }

  FaceNodeAudit global;
  global.mismatched = GlobalSum(local.mismatched);
  global.total = GlobalSum(local.total);
  global.shape_local_err_max = GlobalMax(local.shape_local_err_max);
  global.shape_peak_err_max = GlobalMax(local.shape_peak_err_max);
  global.state_interp_err_max = GlobalMax(local.state_interp_err_max);
  return global;
}

void AssembleBoundaryShape(const mfem::ParFiniteElementSpace& vfes,
                           const mfem::Vector& px,
                           mfem::ParMesh& pmesh,
                           mfem::BdrHyperbolicFluxIntegrator& bdr_int,
                           mfem::Vector& z)
{
  const int num_eq = vfes.GetVDim();
  z.SetSize(vfes.GetTrueVSize());
  z = 0.0;

  mfem::Vector elfun;
  mfem::Vector elvect;

  for (int b = 0; b < pmesh.GetNBE(); ++b)
  {
    mfem::FaceElementTransformations* Tr = pmesh.GetBdrFaceTransformations(b);
    if (!Tr) { continue; }

    const int e1 = Tr->Elem1No;
    const mfem::FiniteElement& fe1 = *vfes.GetFE(e1);
    const int dof1 = fe1.GetDof();

    mfem::Array<int> vdofs1;
    vfes.GetElementVDofs(e1, vdofs1);
    px.GetSubVector(vdofs1, elfun);
    bdr_int.AssembleFaceVector(fe1, fe1, *Tr, elfun, elvect);

    mfem::DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_eq);
    z.AddElementVector(vdofs1, elvect1_mat.GetData());
  }
}

} // namespace

int RunDGSEMOperatorTests(TestContext& ctx)
{
  const double tol = 1.0e-10;

  // 1D constant-state cancellation.
  constexpr int dim = 1;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(2, 1.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

  mfem::DG_FECollection fec(/*order=*/2, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq,
                                   mfem::Ordering::byNODES);

  mfem::ParGridFunction U(&vfes);
  const mfem::Vector cons_const =
      EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/0.0, /*p=*/1.0, gamma);
  FillConstantState(cons_const, U);

  mfem::EulerFlux flux(dim, gamma);
  mfem::RusanovFluxWithSides num_flux(flux);

  constexpr int int_order_offset = 2;
  auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                       int_order_offset);
  mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
                                          /*preassembleWeakDivergence=*/true);
  oper.SetNumericalFlux(&num_flux);
  oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

  mfem::Array<int> bdr_marker;
  const int max_attr = pmesh.bdr_attributes.Max();
  if (max_attr > 0)
  {
    bdr_marker.SetSize(max_attr);
    bdr_marker = 1;
    auto outlet = std::make_unique<mfem::SupersonicOutletState>();
    oper.AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(num_flux, std::move(outlet),
                                              int_order_offset),
        bdr_marker);
  }

  mfem::Vector x;
  U.GetTrueDofs(x);

  mfem::dgsem::DGSEMOperator dgsem_op(oper);

  mfem::Vector z_surf(vfes.GetTrueVSize());
  mfem::Vector z_vol(vfes.GetTrueVSize());

  dgsem_op.AssembleSurface(x, z_surf);
  z_vol = 0.0;
  dgsem_op.AssembleVolume(x, z_vol);

  mfem::Vector z_sum(z_surf.Size());
  z_sum = z_surf;
  z_sum += z_vol;

  const double norm_surf = GlobalL2Norm(z_surf);
  const double norm_vol = GlobalL2Norm(z_vol);
  const double norm_sum = GlobalL2Norm(z_sum);

  ctx.Check(norm_surf > 0.0, "DGSEM surface contribution is nonzero");
  ctx.Check(norm_vol > 0.0, "DGSEM volume contribution is nonzero");
  ctx.Check(norm_sum < tol,
            "DGSEM surface + volume cancel for constant state (norm=" +
                FormatSci(norm_sum) + ")");

  mfem::Vector y(vfes.GetTrueVSize());
  dgsem_op.ApplyInverseMass(z_sum, y);
  const double resid_norm = GlobalL2Norm(y);
  ctx.Check(resid_norm < tol,
            "DGSEM operator residual ~0 after M^{-1} (norm=" +
                FormatSci(resid_norm) + ")");

  ctx.AddParity(norm_surf + norm_vol + norm_sum + resid_norm);

  // 2D constant-state cancellation on a single quad (outlet + inlet BCs).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    constexpr int order2 = 2;

    mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                   /*gen_edges=*/true, 1.0, 1.0);
    mfem::ParMesh pmesh2(MPI_COMM_WORLD, mesh2);

    mfem::DG_FECollection fec2(order2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh2, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    mfem::ParGridFunction U2(&vfes2);
    const mfem::Vector cons_const2 =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const2, U2);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);

    constexpr int int_order_offset2 = 2;
    const int max_attr2 = pmesh2.bdr_attributes.Max();
    mfem::Array<int> bdr_marker2;
    if (max_attr2 > 0)
    {
      bdr_marker2.SetSize(max_attr2);
      bdr_marker2 = 1;
    }

    auto run_boundary_case = [&](const std::string& label,
                                 std::unique_ptr<mfem::BoundaryState> bc_state)
    {
      auto form_integrator2 =
          std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                           int_order_offset2);
      mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                               /*preassembleWeakDivergence=*/true);
      oper2.SetNumericalFlux(&num_flux2);
      oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

      if (max_attr2 > 0)
      {
        oper2.AddBdrFaceIntegrator(
            new mfem::BdrHyperbolicFluxIntegrator(num_flux2, std::move(bc_state),
                                                  int_order_offset2),
            bdr_marker2);
      }

      mfem::Vector x2;
      U2.GetTrueDofs(x2);

      mfem::dgsem::DGSEMOperator dgsem_op2(oper2);

      mfem::Vector z_surf2(vfes2.GetTrueVSize());
      mfem::Vector z_vol2(vfes2.GetTrueVSize());
      dgsem_op2.AssembleSurface(x2, z_surf2);
      z_vol2 = 0.0;
      dgsem_op2.AssembleVolume(x2, z_vol2);

      mfem::Vector z_sum2(z_surf2.Size());
      z_sum2 = z_surf2;
      z_sum2 += z_vol2;

      const double norm_sum2 = GlobalL2Norm(z_sum2);
      ctx.Check(norm_sum2 < tol,
                "DGSEM 2D surface + volume cancel (" + label + ", norm=" +
                    FormatSci(norm_sum2) + ")");

      mfem::Vector y2(vfes2.GetTrueVSize());
      dgsem_op2.ApplyInverseMass(z_sum2, y2);
      const double resid_norm2 = GlobalL2Norm(y2);
      ctx.Check(resid_norm2 < tol,
                "DGSEM 2D residual ~0 after M^{-1} (" + label + ", norm=" +
                    FormatSci(resid_norm2) + ")");

      ctx.AddParity(norm_sum2 + resid_norm2);
    };

    run_boundary_case("outlet",
                      std::make_unique<mfem::SupersonicOutletState>());

    run_boundary_case("inlet",
                      std::make_unique<mfem::SupersonicInletState>(cons_const2));
  }

  // Boundary face nodes + normals sanity (LGL alignment + outward normals).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    constexpr int order2 = 3;
    const double tol_face = 1.0e-10;

    mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                   /*gen_edges=*/true, 1.0, 1.0);
    mfem::ParMesh pmesh2(MPI_COMM_WORLD, mesh2);

    mfem::DG_FECollection fec2(order2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh2, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    const auto& basis = mfem::dgsem::GetLGLBasis1D(order2);
    const mfem::Vector& nodes1d = basis.nodes;
    const mfem::IntegrationRule& ir = mfem::dgsem::GetLGLFaceRule(order2);

    auto find_node = [&](double coord) -> int
    {
      int best = 0;
      double best_err = std::abs(coord - nodes1d(0));
      for (int i = 1; i < nodes1d.Size(); ++i)
      {
        const double err = std::abs(coord - nodes1d(i));
        if (err < best_err)
        {
          best_err = err;
          best = i;
        }
      }
      ctx.Check(best_err < tol_face,
                "LGL face IP aligns with nodal coordinate");
      return best;
    };

    mfem::Mesh& mesh_ref = *vfes2.GetMesh();
    const int nfaces = mesh_ref.GetNumFaces();
    int bdr_faces = 0;
    for (int f = 0; f < nfaces; ++f)
    {
      mfem::FaceElementTransformations* Tr =
          mesh_ref.GetFaceElementTransformations(f);
      if (!Tr || Tr->Elem2No >= 0) { continue; }
      ++bdr_faces;

      mfem::Array<int> face_verts;
      mesh_ref.GetFaceVertices(f, face_verts);
      mfem::Vector center(dim2);
      center = 0.0;
      for (int i = 0; i < face_verts.Size(); ++i)
      {
        const double* v = mesh_ref.GetVertex(face_verts[i]);
        center(0) += v[0];
        center(1) += v[1];
      }
      center /= static_cast<double>(face_verts.Size());

      mfem::Vector expected(dim2);
      expected = 0.0;
      if (std::abs(center(0)) < tol_face)
      {
        expected(0) = -1.0;
      }
      else if (std::abs(center(0) - 1.0) < tol_face)
      {
        expected(0) = 1.0;
      }
      else if (std::abs(center(1)) < tol_face)
      {
        expected(1) = -1.0;
      }
      else if (std::abs(center(1) - 1.0) < tol_face)
      {
        expected(1) = 1.0;
      }
      else
      {
        ctx.Check(false, "Boundary face center lies on unit-square boundary");
      }

      double ref_norm = -1.0;
      for (int ip_id = 0; ip_id < ir.GetNPoints(); ++ip_id)
      {
        const mfem::IntegrationPoint& ip = ir.IntPoint(ip_id);
        Tr->SetAllIntPoints(&ip);
        const mfem::IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
        (void)find_node(ip1.x);
        (void)find_node(ip1.y);

        mfem::Vector nor(dim2);
        mfem::CalcOrtho(Tr->Jacobian(), nor);
        const double nmag = nor.Norml2();
        ctx.Check(nmag > 0.0, "Boundary normal magnitude is nonzero");
        if (ref_norm < 0.0) { ref_norm = nmag; }
        ctx.CheckNear(nmag, ref_norm, 1.0e-10,
                      "Boundary normal magnitude consistent along face");

        const double dot = nor * expected;
        ctx.Check(dot > 0.0, "Boundary normal points outward");

        if (std::abs(expected(0)) > 0.0)
        {
          ctx.Check(std::abs(nor(1)) <= 1.0e-10 * nmag,
                    "Boundary normal is axis-aligned (x-face)");
        }
        else
        {
          ctx.Check(std::abs(nor(0)) <= 1.0e-10 * nmag,
                    "Boundary normal is axis-aligned (y-face)");
        }
      }
    }
    ctx.Check(bdr_faces > 0, "Found boundary faces to check");
  }

  // Boundary integrators use LGL face rule when configured.
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    constexpr int order2 = 3;
    constexpr int int_order_offset2 = 2;

    mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                   /*gen_edges=*/true, 1.0, 1.0);
    mfem::ParMesh pmesh2(MPI_COMM_WORLD, mesh2);
    mfem::DG_FECollection fec2(order2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh2, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    const mfem::IntegrationRule& lgl_rule = mfem::dgsem::GetLGLFaceRule(order2);
    oper2.SetFaceIntegrationRule(lgl_rule);

    mfem::Array<int> bdr_marker2;
    const int max_attr2 = pmesh2.bdr_attributes.Max();
    bdr_marker2.SetSize(max_attr2);
    bdr_marker2 = 1;

    auto inlet2 = std::make_unique<mfem::SupersonicInletState>(
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma));
    auto bdr = std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
        num_flux2, std::move(inlet2), int_order_offset2);
    mfem::BdrHyperbolicFluxIntegrator* bdr_ptr = bdr.get();
    oper2.AddBdrFaceIntegrator(std::move(bdr), bdr_marker2);
    const mfem::IntegrationRule* bdr_rule = bdr_ptr->GetIntegrationRule();
    bool same_rule = (bdr_rule != nullptr) &&
                     (bdr_rule->GetNPoints() == lgl_rule.GetNPoints());
    double rule_err_max = 0.0;
    if (same_rule)
    {
      for (int q = 0; q < lgl_rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& a = bdr_rule->IntPoint(q);
        const mfem::IntegrationPoint& b = lgl_rule.IntPoint(q);
        const double err = std::max(std::abs(a.x - b.x),
                                    std::abs(a.weight - b.weight));
        rule_err_max = std::max(rule_err_max, err);
        if (err > 1.0e-14)
        {
          same_rule = false;
          break;
        }
      }
    }
    if (ctx.verbose_metrics)
    {
      const int bdr_npts = bdr_rule ? bdr_rule->GetNPoints() : -1;
      const int bdr_order = bdr_rule ? bdr_rule->GetOrder() : -1;
      ctx.LogMetric("Boundary rule audit: npts=" + std::to_string(bdr_npts) +
                    ", order=" + std::to_string(bdr_order) +
                    ", err_max=" + FormatSci(rule_err_max));
    }
    ctx.Check(same_rule,
              "Boundary integrator uses DGSEM LGL face rule");

    // Updating the face rule after BC registration must retarget existing
    // boundary integrators to the new rule (no stale pointer to prior rule).
    const mfem::IntegrationRule& lgl_rule_updated = mfem::dgsem::GetLGLFaceRule(1);
    oper2.SetFaceIntegrationRule(lgl_rule_updated);
    const mfem::IntegrationRule* bdr_rule_updated = bdr_ptr->GetIntegrationRule();
    bool same_updated_rule = (bdr_rule_updated != nullptr) &&
                             (bdr_rule_updated->GetNPoints() ==
                              lgl_rule_updated.GetNPoints());
    if (same_updated_rule)
    {
      for (int q = 0; q < lgl_rule_updated.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& a = bdr_rule_updated->IntPoint(q);
        const mfem::IntegrationPoint& b = lgl_rule_updated.IntPoint(q);
        const double err = std::max(std::abs(a.x - b.x),
                                    std::abs(a.weight - b.weight));
        if (err > 1.0e-14)
        {
          same_updated_rule = false;
          break;
        }
      }
    }
    ctx.Check(same_updated_rule,
              "Boundary integrator tracks updated DGSEM face rule");
  }

  // Boundary-only constant inflow diagnostic (surface term alone).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    constexpr int order2 = 3;
    constexpr int int_order_offset2 = 2;
    const double tol_bdr = 1.0e-12;

    mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                   /*gen_edges=*/true, 1.0, 1.0);
    mfem::ParMesh pmesh2(MPI_COMM_WORLD, mesh2);
    mfem::DG_FECollection fec2(order2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh2, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    mfem::ParGridFunction U2(&vfes2);
    const mfem::Vector cons_const2 =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.25, /*v=*/0.1, /*p=*/1.0, gamma);
    FillConstantState(cons_const2, U2);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper2.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order2));

    mfem::Array<int> bdr_marker2;
    const int max_attr2 = pmesh2.bdr_attributes.Max();
    bdr_marker2.SetSize(max_attr2);
    bdr_marker2 = 1;
    auto inlet2 = std::make_unique<mfem::SupersonicInletState>(cons_const2);
    oper2.AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(num_flux2, std::move(inlet2),
                                              int_order_offset2),
        bdr_marker2);

    mfem::Vector x2;
    U2.GetTrueDofs(x2);
    mfem::dgsem::DGSEMOperator dgsem_op2(oper2);
    mfem::Vector z_surf2(vfes2.GetTrueVSize());
    dgsem_op2.AssembleSurface(x2, z_surf2);
    const double norm_surf2 = GlobalL2Norm(z_surf2);
    ctx.LogMetric("DGSEM boundary-only residual (constant inflow): " +
                  FormatSci(norm_surf2));
    ctx.LogMetric("DGSEM boundary-only residual is expected to be nonzero; "
                  "volume term cancels it for constant states.");
  }

  // Boundary residual match: DGSEM surface vs MFEM Bdr integrator (same BC data).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    constexpr int order2 = 3;
    constexpr int int_order_offset2 = 2;
    const double tol_match = 1.0e-12;

    mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                   /*gen_edges=*/true, 1.0, 1.0);
    mfem::ParMesh pmesh2(MPI_COMM_WORLD, mesh2);
    mfem::DG_FECollection fec2(order2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh2, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    mfem::ParGridFunction U2(&vfes2);
    const mfem::Vector cons_const2 =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const2, U2);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);
    const mfem::IntegrationRule& lgl_rule = mfem::dgsem::GetLGLFaceRule(order2);

    // DGSEM surface assembly.
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper2.SetFaceIntegrationRule(lgl_rule);

    mfem::Array<int> bdr_marker2;
    const int max_attr2 = pmesh2.bdr_attributes.Max();
    bdr_marker2.SetSize(max_attr2);
    bdr_marker2 = 1;
    auto inlet2 = std::make_unique<mfem::SupersonicInletState>(cons_const2);
    oper2.AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(num_flux2, std::move(inlet2),
                                              int_order_offset2),
        bdr_marker2);

    mfem::Vector x2;
    U2.GetTrueDofs(x2);

    mfem::dgsem::DGSEMOperator dgsem_op2(oper2);
    mfem::Vector z_dgsem(vfes2.GetTrueVSize());
    dgsem_op2.AssembleSurface(x2, z_dgsem);

    // MFEM boundary integrator assembly.
    mfem::ParNonlinearForm bdr_form(&vfes2);
    auto inlet_ref = std::make_unique<mfem::SupersonicInletState>(cons_const2);
    auto* bdr_int = new mfem::BdrHyperbolicFluxIntegrator(num_flux2,
                                                          std::move(inlet_ref),
                                                          int_order_offset2);
    bdr_int->SetIntegrationRule(lgl_rule);
    bdr_form.AddBdrFaceIntegrator(bdr_int, bdr_marker2);

    mfem::Vector y_bdr(vfes2.GetTrueVSize());
    bdr_form.Mult(x2, y_bdr);

    mfem::Vector diff(z_dgsem.Size());
    diff = z_dgsem;
    diff -= y_bdr;
    const double diff_norm = GlobalL2Norm(diff);
    ctx.LogMetric("DGSEM vs Bdr boundary residual diff: " +
                  FormatSci(diff_norm));

    // Compare boundary element vs face-element transforms at LGL nodes.
    BoundaryTransformAudit tr_audit = AuditBoundaryTransforms(pmesh2, order2);
    ctx.LogMetric("DGSEM bdr/face transform audit: ip_diff_max=" +
                  FormatSci(tr_audit.max_ip_diff) +
                  ", nor_diff_max=" + FormatSci(tr_audit.max_nor_diff));

    // Manual boundary assembly using BdrHyperbolicFluxIntegrator + bdr transforms.
    auto inlet_manual = std::make_unique<mfem::SupersonicInletState>(cons_const2);
    mfem::BdrHyperbolicFluxIntegrator bdr_manual(num_flux2, std::move(inlet_manual),
                                                 int_order_offset2);
    bdr_manual.SetIntegrationRule(lgl_rule);
    mfem::Vector px;
    if (const mfem::Operator* P = vfes2.GetProlongationMatrix())
    {
      px.SetSize(P->Height());
      P->Mult(x2, px);
    }
    else
    {
      px = x2;
    }

    mfem::Vector y_manual;
    AssembleBoundaryShape(vfes2, px, pmesh2, bdr_manual, y_manual);

    mfem::Vector diff_manual(y_manual.Size());
    diff_manual = y_manual;
    diff_manual -= y_bdr;
    const double diff_manual_norm = GlobalL2Norm(diff_manual);
    ctx.LogMetric("Bdr manual vs Bdr form diff: " +
                  FormatSci(diff_manual_norm));

    mfem::Vector diff_dgsem_manual(z_dgsem.Size());
    diff_dgsem_manual = z_dgsem;
    diff_dgsem_manual -= y_manual;
    const double diff_dgsem_manual_norm = GlobalL2Norm(diff_dgsem_manual);
    ctx.LogMetric("DGSEM vs Bdr manual diff: " +
                  FormatSci(diff_dgsem_manual_norm));

    FaceNodeAudit audit = AuditDGSEMFaceNodes(U2, pmesh2, order2);
    const long long mismatched = static_cast<long long>(audit.mismatched);
    const long long total = static_cast<long long>(audit.total);
    ctx.LogMetric("DGSEM face-node audit: mismatched=" + std::to_string(mismatched) +
                  "/" + std::to_string(total) +
                  ", shape_local_err_max=" + FormatSci(audit.shape_local_err_max) +
                  ", shape_peak_err_max=" + FormatSci(audit.shape_peak_err_max) +
                  ", state_interp_err_max=" + FormatSci(audit.state_interp_err_max));

    const double tol_audit = 1.0e-12;
    const bool strict_audit =
        EnvEnabledDefault("DG_IGR_DGSEM_FACE_AUDIT_STRICT", true);
    if (strict_audit)
    {
      ctx.Check(mismatched == 0,
                "DGSEM face-node audit has no mismatches");
      ctx.Check(audit.shape_local_err_max < tol_audit,
                "DGSEM face-node audit local shape ~1 (err=" +
                    FormatSci(audit.shape_local_err_max) + ")");
      ctx.Check(audit.state_interp_err_max < tol_audit,
                "DGSEM face-node audit state interpolation matches node value "
                "(err=" + FormatSci(audit.state_interp_err_max) + ")");
    }
    else
    {
      ctx.LogMetric("DGSEM face-node strict audit disabled "
                    "(set DG_IGR_DGSEM_FACE_AUDIT_STRICT=1).");
    }
    const bool strict_match =
        EnvEnabledDefault("DG_IGR_DGSEM_BDR_MATCH", true);
    if (strict_match)
    {
      ctx.Check(diff_norm < tol_match,
                "DGSEM boundary residual matches Bdr integrator (norm=" +
                    FormatSci(diff_norm) + ")");
    }
    else
    {
      ctx.LogMetric("DGSEM boundary residual match check disabled "
                    "(set DG_IGR_DGSEM_BDR_MATCH=1).");
    }
  }

  // 2D periodic boundary cancellation (no BC integrators).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;

    const int n = 3;
    const bool sfc = false;
    mfem::Mesh orig_mesh =
        mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::QUADRILATERAL,
                                    /*gen_edges=*/false, 1.0, 1.0, sfc);
    std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                              mfem::Vector({0.0, 1.0})};
    mfem::Mesh mesh =
        mfem::Mesh::MakePeriodic(orig_mesh,
                                 orig_mesh.CreatePeriodicVertexMapping(translations));

    mfem::DG_FECollection fec2(/*order=*/2, dim2, mfem::BasisType::GaussLobatto);
    mfem::FiniteElementSpace vfes2(&mesh, &fec2, num_eq2,
                                   mfem::Ordering::byNODES);

    mfem::GridFunction U2(&vfes2);
    const mfem::Vector cons_const2 =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const2, U2);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);

    constexpr int int_order_offset2 = 2;
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

    mfem::Vector x2;
    U2.GetTrueDofs(x2);

    mfem::dgsem::DGSEMOperator dgsem_op2(oper2);
    mfem::Vector z_surf2(vfes2.GetTrueVSize());
    mfem::Vector z_vol2(vfes2.GetTrueVSize());
    dgsem_op2.AssembleSurface(x2, z_surf2);
    z_vol2 = 0.0;
    dgsem_op2.AssembleVolume(x2, z_vol2);

    mfem::Vector z_sum2(z_surf2.Size());
    z_sum2 = z_surf2;
    z_sum2 += z_vol2;

    const double norm_sum2 = GlobalL2Norm(z_sum2);
    ctx.Check(norm_sum2 < tol,
              "DGSEM periodic surface + volume cancel (norm=" +
                  FormatSci(norm_sum2) + ")");

    mfem::Vector y2(vfes2.GetTrueVSize());
    dgsem_op2.ApplyInverseMass(z_sum2, y2);
    const double resid_norm2 = GlobalL2Norm(y2);
    ctx.Check(resid_norm2 < tol,
              "DGSEM periodic residual ~0 after M^{-1} (norm=" +
                  FormatSci(resid_norm2) + ")");

    ctx.AddParity(norm_sum2 + resid_norm2);
  }

  // Face orientation/permute invariance across a shared face.
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;

    mfem::Mesh mesh =
        mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL,
                                    /*gen_edges=*/true, 1.0, 1.0);
    // Rotate vertex ordering for element 0 (permute, preserves orientation).
    {
      mfem::Element* el0 = mesh.GetElement(0);
      int* v = el0->GetVertices();
      const int tmp = v[0];
      v[0] = v[1];
      v[1] = v[2];
      v[2] = v[3];
      v[3] = tmp;
    }
    // Rotate vertex ordering for element 1 with a different permutation.
    {
      mfem::Element* el1 = mesh.GetElement(1);
      int* v = el1->GetVertices();
      const int v0 = v[0];
      const int v1 = v[1];
      const int v2 = v[2];
      const int v3 = v[3];
      v[0] = v2;
      v[1] = v3;
      v[2] = v0;
      v[3] = v1;
    }
    mesh.FinalizeTopology();
    mesh.Finalize(/*refine=*/false, /*fix_orientation=*/false);

    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec2(/*order=*/2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    mfem::ParGridFunction U2(&vfes2);
    const mfem::Vector cons_const2 =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const2, U2);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);

    constexpr int int_order_offset2 = 2;
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

    mfem::Array<int> bdr_marker2;
    const int max_attr2 = pmesh.bdr_attributes.Max();
    if (max_attr2 > 0)
    {
      bdr_marker2.SetSize(max_attr2);
      bdr_marker2 = 1;
      auto outlet2 = std::make_unique<mfem::SupersonicOutletState>();
      oper2.AddBdrFaceIntegrator(
          new mfem::BdrHyperbolicFluxIntegrator(num_flux2, std::move(outlet2),
                                                int_order_offset2),
          bdr_marker2);
    }

    mfem::Vector x2;
    U2.GetTrueDofs(x2);

    mfem::dgsem::DGSEMOperator dgsem_op2(oper2);
    mfem::Vector z_sum2(vfes2.GetTrueVSize());
    z_sum2 = 0.0;
    dgsem_op2.AssembleSurface(x2, z_sum2);
    dgsem_op2.AssembleVolume(x2, z_sum2);

    mfem::Vector y2(vfes2.GetTrueVSize());
    dgsem_op2.ApplyInverseMass(z_sum2, y2);
    const double resid_norm2 = GlobalL2Norm(y2);
    ctx.Check(resid_norm2 < tol,
              "DGSEM orientation/permute invariance (norm=" +
                  FormatSci(resid_norm2) + ")");
  }

  // High-order constant-state residuals (p=4/5).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    const double tol_hi = 1.0e-9;
    const int orders[] = {4, 5};

    for (const int order : orders)
    {
      mfem::Mesh mesh2 =
          mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                      /*gen_edges=*/true, 1.0, 1.0);
      mfem::ParMesh pmesh2(MPI_COMM_WORLD, mesh2);

      mfem::DG_FECollection fec2(order, dim2, mfem::BasisType::GaussLobatto);
      mfem::ParFiniteElementSpace vfes2(&pmesh2, &fec2, num_eq2,
                                        mfem::Ordering::byNODES);

      mfem::ParGridFunction U2(&vfes2);
      const mfem::Vector cons_const2 =
          EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
      FillConstantState(cons_const2, U2);

      mfem::EulerFlux flux2(dim2, gamma);
      mfem::RusanovFluxWithSides num_flux2(flux2);

      constexpr int int_order_offset2 = 2;
      auto form_integrator2 =
          std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                           int_order_offset2);
      mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                               /*preassembleWeakDivergence=*/true);
      oper2.SetNumericalFlux(&num_flux2);
      oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

      mfem::Array<int> bdr_marker2;
      const int max_attr2 = pmesh2.bdr_attributes.Max();
      if (max_attr2 > 0)
      {
        bdr_marker2.SetSize(max_attr2);
        bdr_marker2 = 1;
        auto outlet2 = std::make_unique<mfem::SupersonicOutletState>();
        oper2.AddBdrFaceIntegrator(
            new mfem::BdrHyperbolicFluxIntegrator(num_flux2, std::move(outlet2),
                                                  int_order_offset2),
            bdr_marker2);
      }

      mfem::Vector x2;
      U2.GetTrueDofs(x2);

      mfem::dgsem::DGSEMOperator dgsem_op2(oper2);
      mfem::Vector z_sum2(vfes2.GetTrueVSize());
      z_sum2 = 0.0;
      dgsem_op2.AssembleSurface(x2, z_sum2);
      dgsem_op2.AssembleVolume(x2, z_sum2);

      mfem::Vector y2(vfes2.GetTrueVSize());
      dgsem_op2.ApplyInverseMass(z_sum2, y2);
      const double resid_norm2 = GlobalL2Norm(y2);
      ctx.Check(resid_norm2 < tol_hi,
                "DGSEM high-order residual ~0 (p=" +
                    std::to_string(order) +
                    ", norm=" + FormatSci(resid_norm2) + ")");
    }
  }

  // Conservation over one RK step on a periodic mesh (mass/energy invariants).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    const double tol_cons = 1.0e-10;

    const int n = 3;
    const bool sfc = false;
    mfem::Mesh orig_mesh =
        mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::QUADRILATERAL,
                                    /*gen_edges=*/false, 1.0, 1.0, sfc);
    std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                              mfem::Vector({0.0, 1.0})};
    mfem::Mesh mesh =
        mfem::Mesh::MakePeriodic(orig_mesh,
                                 orig_mesh.CreatePeriodicVertexMapping(translations));
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec2(/*order=*/2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    class SmoothConservativeCoeff final : public mfem::VectorCoefficient
    {
    public:
      SmoothConservativeCoeff(double gamma)
        : mfem::VectorCoefficient(kNumEq), gamma_(gamma) {}

      void Eval(mfem::Vector& V,
                mfem::ElementTransformation& Tr,
                const mfem::IntegrationPoint& ip) override
      {
        mfem::Vector x(kDim);
        Tr.Transform(ip, x);

        constexpr double kPi = dg_sim::math::kPi;
        const double sx = std::sin(2.0 * kPi * x(0));
        const double sy = std::sin(2.0 * kPi * x(1));
        const double rho = 1.0 + 0.1 * sx * sy;
        const double u = 0.2 + 0.05 * std::cos(2.0 * kPi * x(1));
        const double v = -0.1 + 0.05 * std::sin(2.0 * kPi * x(0));
        const double p = 1.0 + 0.05 * sx;

        V = EulerConsFromPrim2D(rho, u, v, p, gamma_);
      }

    private:
      enum { kDim = 2, kNumEq = kDim + 2 };
      double gamma_;
    };

    mfem::ParGridFunction U2(&vfes2);
    SmoothConservativeCoeff coeff(gamma);
    U2.ProjectCoefficient(coeff);

    const double mass_before = IntegrateComponent(U2, 0);
    const double momx_before = IntegrateComponent(U2, 1);
    const double momy_before = IntegrateComponent(U2, 2);
    const double energy_before = IntegrateComponent(U2, 3);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);

    constexpr int int_order_offset2 = 2;
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper2.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(/*order=*/2));

    mfem::RK3SSPWithLimiter rk(vfes2);
    rk.Init(oper2);

    double t = 0.0;
    double dt = 1.0e-3;
    mfem::Vector x2;
    U2.GetTrueDofs(x2);
    rk.Step(x2, t, dt);
    U2.SetFromTrueDofs(x2);

    const double mass_after = IntegrateComponent(U2, 0);
    const double momx_after = IntegrateComponent(U2, 1);
    const double momy_after = IntegrateComponent(U2, 2);
    const double energy_after = IntegrateComponent(U2, 3);

    ctx.CheckNear(mass_after, mass_before, tol_cons,
                  "DGSEM periodic RK3 mass conserved");
    ctx.CheckNear(momx_after, momx_before, tol_cons,
                  "DGSEM periodic RK3 x-momentum conserved");
    ctx.CheckNear(momy_after, momy_before, tol_cons,
                  "DGSEM periodic RK3 y-momentum conserved");
    ctx.CheckNear(energy_after, energy_before, tol_cons,
                  "DGSEM periodic RK3 energy conserved");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests
