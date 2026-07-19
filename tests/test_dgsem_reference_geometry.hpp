#pragma once

#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_metric.hpp"
#include "numerics/dgsem/dgsem_operator.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_reference.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/euler/conversions.hpp"
#include "utils/math/constants.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace dg_igr_tests
{
namespace dgsem_reference_geometry
{

struct MetricOracleEvidence
{
  std::array<mfem::real_t, 4> affine_adjugate_max_error{{}};
  std::array<mfem::real_t, 4> affine_face_normal_max_error{{}};
  std::array<mfem::real_t, 4> affine_cross_term_max{{}};
  std::array<mfem::real_t, 4> curved_metric_identity_max_error{{}};
};

struct PermutationOracleEvidence
{
  double geometry_max_error = 0.0;
  double component_moment_max_error = 0.0;
  double canonical_surface_norm = 0.0;
  double permuted_surface_norm = 0.0;
  int local_interior_faces = 0;
  int shared_faces = 0;
};

inline int FindLGLIndex(const mfem::Vector& nodes, double coordinate)
{
  int best = 0;
  double best_error = std::abs(coordinate - nodes(0));
  for (int node = 1; node < nodes.Size(); ++node)
  {
    const double error = std::abs(coordinate - nodes(node));
    if (error < best_error)
    {
      best = node;
      best_error = error;
    }
  }
  MFEM_VERIFY(best_error < 1.0e-10,
              "DGSEM reference face point is not an LGL node.");
  return best;
}

inline std::vector<int> BuildLexToLocal(const mfem::FiniteElement& fe)
{
  const int order = fe.GetOrder();
  const int n1d = order + 1;
  std::vector<int> lex_to_local(static_cast<std::size_t>(fe.GetDof()), -1);
  const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(fe);
  for (int local = 0; local < fe.GetDof(); ++local)
  {
    const int i = dof_to_ijk[static_cast<std::size_t>(local)][0];
    const int j = dof_to_ijk[static_cast<std::size_t>(local)][1];
    lex_to_local[static_cast<std::size_t>(i + n1d * j)] = local;
  }
  return lex_to_local;
}

inline int FaceLocalDof(const mfem::FiniteElement& fe,
                        const mfem::IntegrationPoint& point)
{
  const auto& basis = mfem::dgsem::GetLGLBasis1D(fe.GetOrder());
  const std::vector<int> lex_to_local = BuildLexToLocal(fe);
  const int i = FindLGLIndex(basis.nodes, point.x);
  const int j = FindLGLIndex(basis.nodes, point.y);
  return lex_to_local[static_cast<std::size_t>(i + basis.n * j)];
}

inline double AuditCurvedFaceMetricPair(
  mfem::FaceElementTransformations& transformations,
  const mfem::FiniteElement& element1,
  const mfem::FiniteElement& element2)
{
  MFEM_VERIFY(transformations.Elem1 && transformations.Elem2,
              "Curved face metric audit requires both element transforms.");
  MFEM_VERIFY(element1.GetOrder() == element2.GetOrder(),
              "Curved face metric audit requires matching orders.");
  std::vector<mfem::dgsem::ContravariantMetric2D> metric1;
  std::vector<mfem::dgsem::ContravariantMetric2D> metric2;
  mfem::dgsem::BuildContravariantMetrics2D(
    element1, *transformations.Elem1, metric1);
  mfem::dgsem::BuildContravariantMetrics2D(
    element2, *transformations.Elem2, metric2);

  const mfem::IntegrationRule& rule =
    mfem::dgsem::GetLGLFaceRule(element1.GetOrder());
  mfem::Vector normal1(2);
  mfem::Vector normal2(2);
  double max_error = 0.0;
  for (int q = 0; q < rule.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint& point = rule.IntPoint(q);
    transformations.SetAllIntPoints(&point);
    const mfem::IntegrationPoint& point1 =
      transformations.GetElement1IntPoint();
    const mfem::IntegrationPoint& point2 =
      transformations.GetElement2IntPoint();
    const int local1 = FaceLocalDof(element1, point1);
    const int local2 = FaceLocalDof(element2, point2);
    mfem::dgsem::BuildNodalFaceNormal2D(
      metric1[static_cast<std::size_t>(local1)],
      transformations.Loc1.Transf.Jacobian(),
      point1,
      normal1);
    mfem::dgsem::BuildNodalFaceNormal2D(
      metric2[static_cast<std::size_t>(local2)],
      transformations.Loc2.Transf.Jacobian(),
      point2,
      normal2);
    normal1 += normal2;
    max_error = std::max(max_error, normal1.Normlinf());
  }
  return max_error;
}

inline MetricOracleEvidence RunP1P4MetricOracles(TestContext& ctx)
{
  MetricOracleEvidence evidence;
  constexpr int dim = 2;
  const mfem::real_t fixture_tolerance =
    mfem::dgsem::PrecisionAwareMetricTolerance(
      mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture);
  const mfem::real_t identity_tolerance =
    mfem::dgsem::PrecisionAwareMetricTolerance(
      mfem::dgsem::DGSEMReferenceTolerances::sbp);

  for (const int order : mfem::dgsem::DGSEMConvergenceContract::orders)
  {
    const std::size_t order_index = static_cast<std::size_t>(order - 1);

    // Positive-Jacobian sheared parallelogram. Its nonzero cross terms catch
    // row/column swaps and off-diagonal sign errors that an axis-aligned map
    // cannot distinguish.
    mfem::Mesh affine = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    affine.Transform([](const mfem::Vector& reference, mfem::Vector& physical)
    {
      physical.SetSize(2);
      physical(0) = 1.2 * reference(0) + 0.35 * reference(1);
      physical(1) = -0.2 * reference(0) + 0.9 * reference(1);
    });
    mfem::DG_FECollection affine_fec(
      order, dim, mfem::BasisType::GaussLobatto);
    mfem::FiniteElementSpace affine_fes(&affine, &affine_fec);
    const mfem::FiniteElement& affine_fe = *affine_fes.GetFE(0);
    mfem::ElementTransformation& affine_transformation =
      *affine_fes.GetElementTransformation(0);
    std::vector<mfem::dgsem::ContravariantMetric2D> affine_metrics;
    mfem::dgsem::BuildContravariantMetrics2D(
      affine_fe, affine_transformation, affine_metrics);
    mfem::DenseMatrix exact_adjugate(dim);
    const mfem::IntegrationRule& affine_nodes = affine_fe.GetNodes();
    for (int local = 0; local < affine_fe.GetDof(); ++local)
    {
      affine_transformation.SetIntPoint(&affine_nodes.IntPoint(local));
      mfem::CalcAdjugate(
        affine_transformation.Jacobian(), exact_adjugate);
      for (int reference = 0; reference < dim; ++reference)
      {
        for (int physical = 0; physical < dim; ++physical)
        {
          evidence.affine_adjugate_max_error[order_index] = std::max(
            evidence.affine_adjugate_max_error[order_index],
            std::abs(
              affine_metrics[static_cast<std::size_t>(local)](
                reference, physical) -
              exact_adjugate(reference, physical)));
        }
      }
      evidence.affine_cross_term_max[order_index] = std::max(
        evidence.affine_cross_term_max[order_index],
        std::max(std::abs(exact_adjugate(0, 1)),
                 std::abs(exact_adjugate(1, 0))));
    }
    mfem::Vector nodal_normal(dim);
    mfem::Vector exact_normal(dim);
    const mfem::IntegrationRule& face_rule =
      mfem::dgsem::GetLGLFaceRule(order);
    for (int boundary = 0; boundary < affine.GetNBE(); ++boundary)
    {
      mfem::FaceElementTransformations* face =
        affine.GetBdrFaceTransformations(boundary);
      for (int q = 0; q < face_rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& point = face_rule.IntPoint(q);
        face->SetAllIntPoints(&point);
        const mfem::IntegrationPoint& element_point =
          face->GetElement1IntPoint();
        const int local = FaceLocalDof(affine_fe, element_point);
        mfem::dgsem::BuildNodalFaceNormal2D(
          affine_metrics[static_cast<std::size_t>(local)],
          face->Loc1.Transf.Jacobian(),
          element_point,
          nodal_normal);
        mfem::CalcOrtho(face->Jacobian(), exact_normal);
        nodal_normal -= exact_normal;
        evidence.affine_face_normal_max_error[order_index] = std::max(
          evidence.affine_face_normal_max_error[order_index],
          nodal_normal.Normlinf());
      }
    }

    // Curved geometry has degree p+1. Curl-form metrics must retain the
    // discrete divergence identity when sampled on the solution's p-grid.
    mfem::Mesh curved = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    curved.SetCurvature(order + 1, false, dim, mfem::Ordering::byNODES);
    curved.Transform([](const mfem::Vector& reference,
                        mfem::Vector& physical)
    {
      physical.SetSize(2);
      physical(0) =
        reference(0) + 0.1 * reference(0) * reference(1);
      physical(1) = reference(1) + 0.05 * reference(0) * reference(0);
    });
    mfem::DG_FECollection curved_fec(
      order, dim, mfem::BasisType::GaussLobatto);
    mfem::FiniteElementSpace curved_fes(&curved, &curved_fec);
    const mfem::FiniteElement& curved_fe = *curved_fes.GetFE(0);
    std::vector<mfem::dgsem::ContravariantMetric2D> curved_metrics;
    mfem::dgsem::BuildContravariantMetrics2D(
      curved_fe,
      *curved_fes.GetElementTransformation(0),
      curved_metrics);
    const auto& basis = mfem::dgsem::GetLGLBasis1D(order);
    const std::vector<int> lex_to_local = BuildLexToLocal(curved_fe);
    for (int j = 0; j < basis.n; ++j)
    {
      for (int i = 0; i < basis.n; ++i)
      {
        for (int physical = 0; physical < dim; ++physical)
        {
          mfem::real_t divergence = 0.0;
          for (int m = 0; m < basis.n; ++m)
          {
            const int xi_local =
              lex_to_local[static_cast<std::size_t>(m + basis.n * j)];
            const int eta_local =
              lex_to_local[static_cast<std::size_t>(i + basis.n * m)];
            divergence +=
              basis.D(i, m) *
                curved_metrics[static_cast<std::size_t>(xi_local)](
                  0, physical) +
              basis.D(j, m) *
                curved_metrics[static_cast<std::size_t>(eta_local)](
                  1, physical);
          }
          evidence.curved_metric_identity_max_error[order_index] = std::max(
            evidence.curved_metric_identity_max_error[order_index],
            std::abs(divergence));
        }
      }
    }

    evidence.affine_adjugate_max_error[order_index] =
      GlobalMax(evidence.affine_adjugate_max_error[order_index]);
    evidence.affine_face_normal_max_error[order_index] =
      GlobalMax(evidence.affine_face_normal_max_error[order_index]);
    evidence.affine_cross_term_max[order_index] =
      GlobalMax(evidence.affine_cross_term_max[order_index]);
    evidence.curved_metric_identity_max_error[order_index] =
      GlobalMax(evidence.curved_metric_identity_max_error[order_index]);
    ctx.Check(evidence.affine_cross_term_max[order_index] > 0.1,
              "DGSEM sheared affine oracle has nonzero cross terms p=" +
                std::to_string(order));
    ctx.Check(evidence.affine_adjugate_max_error[order_index] <
                fixture_tolerance,
              "DGSEM affine adjugate equivalence p=" +
                std::to_string(order));
    ctx.Check(evidence.affine_face_normal_max_error[order_index] <
                fixture_tolerance,
              "DGSEM affine endpoint metric/face-normal consistency p=" +
                std::to_string(order));
    ctx.Check(evidence.curved_metric_identity_max_error[order_index] <
                identity_tolerance,
              "DGSEM curved discrete metric identity p=" +
                std::to_string(order));
  }
  return evidence;
}

inline PermutationOracleEvidence
RunNonconstantPermutationOracle(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr int num_eq = dim + 2;
  constexpr int order = 3;
  constexpr double gamma = 1.4;
  constexpr int moments_per_equation = 6;
  const int world_size = mfem::Mpi::WorldSize();
  ctx.Check(world_size == 1 || world_size == 2,
            "DGSEM permutation oracle is bounded to one/two ranks");

  PermutationOracleEvidence evidence;
  std::vector<double> canonical_moments;
  std::vector<double> permuted_moments;

  auto evaluate = [&](bool permuted,
                      std::vector<double>& global_moments,
                      double& surface_norm,
                      int& local_interior_faces,
                      int& shared_faces)
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    if (permuted)
    {
      int* left = mesh.GetElement(0)->GetVertices();
      const int left0 = left[0];
      left[0] = left[1];
      left[1] = left[2];
      left[2] = left[3];
      left[3] = left0;
      int* right = mesh.GetElement(1)->GetVertices();
      const int right0 = right[0];
      const int right1 = right[1];
      const int right2 = right[2];
      const int right3 = right[3];
      right[0] = right2;
      right[1] = right3;
      right[2] = right0;
      right[3] = right1;
      mesh.FinalizeTopology();
      mesh.Finalize(false, false);
    }
    const int partition[2] = {0, world_size == 2 ? 1 : 0};
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh, partition);
    int local_faces_local = 0;
    for (int face_index = 0; face_index < pmesh.GetNumFaces(); ++face_index)
    {
      mfem::FaceElementTransformations* face =
        pmesh.GetFaceElementTransformations(face_index);
      if (face && face->Elem2No >= 0)
      {
        ++local_faces_local;
      }
    }
    local_interior_faces = static_cast<int>(GlobalSum(local_faces_local));
    shared_faces =
      static_cast<int>(GlobalSum(pmesh.GetNSharedFaces()));

    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(
      &pmesh, &fec, num_eq, mfem::Ordering::byNODES);
    mfem::ParGridFunction state(&vfes);
    auto nonconstant = [=](const mfem::Vector& position, mfem::Vector& value)
    {
      mfem::Vector primitive(num_eq);
      primitive(0) = 1.0 + 0.07 * position(0) + 0.03 * position(1);
      primitive(1) = 0.2 + 0.05 * position(1);
      primitive(2) = -0.1 + 0.04 * position(0);
      primitive(3) = 1.0 + 0.02 * position(0) * position(1);
      mfem::EulerPrimToConservative(primitive, dim, gamma, value);
    };
    mfem::VectorFunctionCoefficient coefficient(num_eq, nonconstant);
    state.ProjectCoefficient(coefficient);

    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides numerical_flux(flux);
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(
      vfes, std::move(form_integrator), true);
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order));
    mfem::Array<int> all_boundary(pmesh.bdr_attributes.Max());
    all_boundary = 1;
    oper.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
        numerical_flux,
        std::make_unique<mfem::SupersonicOutletState>(),
        2),
      all_boundary);

    mfem::Vector x;
    state.GetTrueDofs(x);
    mfem::Vector surface(vfes.GetVSize());
    mfem::dgsem::DGSEMOperator(oper).AssembleSurface(x, surface);
    surface_norm = GlobalL2Norm(surface);

    std::vector<double> local_moments(
      static_cast<std::size_t>(num_eq * moments_per_equation), 0.0);
    mfem::Array<int> vdofs;
    mfem::Vector element_values;
    mfem::Vector physical(dim);
    for (int element = 0; element < vfes.GetNE(); ++element)
    {
      const mfem::FiniteElement& fe = *vfes.GetFE(element);
      vfes.GetElementVDofs(element, vdofs);
      surface.GetSubVector(vdofs, element_values);
      mfem::DenseMatrix values(
        element_values.GetData(), fe.GetDof(), num_eq);
      mfem::ElementTransformation* transformation =
        vfes.GetElementTransformation(element);
      for (int local = 0; local < fe.GetDof(); ++local)
      {
        transformation->Transform(fe.GetNodes().IntPoint(local), physical);
        const double moment[moments_per_equation] = {
          1.0,
          physical(0),
          physical(1),
          physical(0) * physical(0),
          physical(0) * physical(1),
          physical(1) * physical(1)};
        for (int equation = 0; equation < num_eq; ++equation)
        {
          for (int m = 0; m < moments_per_equation; ++m)
          {
            local_moments[static_cast<std::size_t>(
              equation * moments_per_equation + m)] +=
              values(local, equation) * moment[m];
          }
        }
      }
    }
    global_moments.resize(local_moments.size());
    for (std::size_t entry = 0; entry < local_moments.size(); ++entry)
    {
      global_moments[entry] = GlobalSum(local_moments[entry]);
    }

    if (permuted)
    {
      const mfem::IntegrationRule& rule = mfem::dgsem::GetLGLFaceRule(order);
      double local_geometry_error = 0.0;
      auto audit_geometry = [&](mfem::FaceElementTransformations* face)
      {
        if (!face || !face->Elem1 || !face->Elem2)
        {
          return;
        }
        mfem::Vector point1(dim);
        mfem::Vector point2(dim);
        for (int q = 0; q < rule.GetNPoints(); ++q)
        {
          face->SetAllIntPoints(&rule.IntPoint(q));
          face->Elem1->Transform(face->GetElement1IntPoint(), point1);
          face->Elem2->Transform(face->GetElement2IntPoint(), point2);
          point1 -= point2;
          local_geometry_error =
            std::max(local_geometry_error, point1.Normlinf());
        }
      };
      for (int face = 0; face < pmesh.GetNumFaces(); ++face)
      {
        audit_geometry(pmesh.GetFaceElementTransformations(face));
      }
      for (int shared = 0; shared < pmesh.GetNSharedFaces(); ++shared)
      {
        audit_geometry(pmesh.GetSharedFaceTransformations(shared, true));
      }
      evidence.geometry_max_error = GlobalMax(local_geometry_error);
    }
  };

  int canonical_local_faces = 0;
  int canonical_shared_faces = 0;
  evaluate(false,
           canonical_moments,
           evidence.canonical_surface_norm,
           canonical_local_faces,
           canonical_shared_faces);
  evaluate(true,
           permuted_moments,
           evidence.permuted_surface_norm,
           evidence.local_interior_faces,
           evidence.shared_faces);

  for (std::size_t entry = 0; entry < canonical_moments.size(); ++entry)
  {
    evidence.component_moment_max_error = std::max(
      evidence.component_moment_max_error,
      std::abs(canonical_moments[entry] - permuted_moments[entry]));
  }
  ctx.Check(evidence.geometry_max_error <
              mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture,
            "DGSEM permuted face LGL points coincide physically");
  ctx.Check(evidence.canonical_surface_norm > 1.0e-8 &&
              evidence.permuted_surface_norm > 1.0e-8,
            "DGSEM nonconstant permutation fixture has nonzero action");
  ctx.Check(evidence.component_moment_max_error <
              mfem::dgsem::DGSEMReferenceTolerances::mpi_parity,
            "DGSEM nonconstant oriented/permuted component moments match");
  if (world_size == 1)
  {
    ctx.Check(evidence.local_interior_faces > 0 &&
                evidence.shared_faces == 0,
              "DGSEM permuted serial fixture encounters local faces only");
  }
  else
  {
    ctx.Check(evidence.shared_faces > 0,
              "DGSEM permuted two-rank fixture encounters a shared face");
  }
  return evidence;
}

} // namespace dgsem_reference_geometry
} // namespace dg_igr_tests
