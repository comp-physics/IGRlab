#pragma once

#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_metric.hpp"
#include "numerics/dgsem/dgsem_operator.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_reference.hpp"
#include "test_dgsem_reference_geometry.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/igr/igr_flux.hpp"
#include "utils/math/constants.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace dg_igr_tests
{
namespace dgsem_reference_detail
{

inline mfem::Mesh MakePeriodicReferenceMesh2D(int nx, int ny, int geometry_order = 0)
{
  mfem::Mesh original =
    mfem::Mesh::MakeCartesian2D(nx,
                                ny,
                                mfem::Element::QUADRILATERAL,
                                /*gen_edges=*/false,
                                1.0,
                                1.0,
                                /*sfc_ordering=*/false);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                            mfem::Vector({0.0, 1.0})};
  mfem::Mesh mesh = mfem::Mesh::MakePeriodic(
    original, original.CreatePeriodicVertexMapping(translations));
  if (geometry_order > 0)
  {
    mesh.SetCurvature(
      geometry_order, true, 2, mfem::Ordering::byNODES);
    mesh.Transform([](const mfem::Vector& input, mfem::Vector& output)
    {
      output.SetSize(2);
      const double two_pi = 2.0 * dg_sim::math::kPi;
      const double warp =
        mfem::dgsem::DGSEMConvergenceContract::curved_map_amplitude *
        std::sin(two_pi * input(0)) * std::sin(two_pi * input(1));
      output(0) = input(0) + warp;
      output(1) = input(1) + 0.5 * warp;
    });
  }
  return mesh;
}

inline double WeakComponentSum(const mfem::ParFiniteElementSpace& vfes,
                               const mfem::Vector& weak,
                               int component)
{
  double local = 0.0;
  for (int sdof = 0; sdof < vfes.GetNDofs(); ++sdof)
  {
    int vdof = vfes.DofToVDof(sdof, component);
    if (vdof < 0)
    {
      vdof = -1 - vdof;
    }
    local += weak(vdof);
  }
  return GlobalSum(local);
}

inline double WeakComponentL2Norm(const mfem::ParFiniteElementSpace& vfes,
                                  const mfem::Vector& weak,
                                  int component)
{
  double local_square = 0.0;
  for (int sdof = 0; sdof < vfes.GetNDofs(); ++sdof)
  {
    int vdof = vfes.DofToVDof(sdof, component);
    if (vdof < 0)
    {
      vdof = -1 - vdof;
    }
    local_square += weak(vdof) * weak(vdof);
  }
  return std::sqrt(GlobalSum(local_square));
}

struct TimingStatistic
{
  double median = 0.0;
  double mad = 0.0;
};

inline TimingStatistic MedianMAD(std::vector<double> samples)
{
  MFEM_VERIFY(!samples.empty(), "MedianMAD requires at least one sample.");
  std::sort(samples.begin(), samples.end());
  const std::size_t mid = samples.size() / 2;
  const double median = samples[mid];
  std::vector<double> deviations(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i)
  {
    deviations[i] = std::abs(samples[i] - median);
  }
  std::sort(deviations.begin(), deviations.end());
  return {median, deviations[mid]};
}

inline double GlobalMaxSeconds(double local)
{
  double global = 0.0;
  MPI_Allreduce(&local,
                &global,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MAX,
                MPI_COMM_WORLD);
  return global;
}

template <typename Callable>
inline double TimeGlobalMaximum(Callable&& callable)
{
  const auto start = std::chrono::steady_clock::now();
  callable();
  const auto stop = std::chrono::steady_clock::now();
  const double local = std::chrono::duration<double>(stop - start).count();
  return GlobalMaxSeconds(local);
}

struct AttributionRecord
{
  const char* category = nullptr;
  TimingStatistic statistic;
};

} // namespace dgsem_reference_detail

inline void RunDGSEMReferenceOperatorGate(TestContext& ctx)
{
  using namespace dgsem_reference_detail;
  constexpr int dim = 2;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;
  constexpr int int_order_offset = 2;
  const double fixture_tol =
    mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture;

  double metric_fixture_max_error = 0.0;
  double metric_identity_max_error = 0.0;
  double inverse_mass_max_error = 0.0;
  double orientation_max_error = 0.0;
  double shared_surface_norm = 0.0;
  std::vector<double> shared_balances(num_eq, 0.0);
  std::vector<double> surface_component_norms(num_eq, 0.0);
  int local_interior_face_count = 0;
  int shared_face_count = 0;
  double curved_local_face_normal_error = 0.0;
  double curved_shared_face_normal_error = 0.0;
  double wall_constant_norm = 0.0;
  double periodic_constant_norm = 0.0;
  std::vector<double> periodic_conservation_deltas(num_eq, 0.0);
  double wall_mass_delta = 0.0;
  double wall_energy_delta = 0.0;
  const auto p1p4_metric_evidence =
    dgsem_reference_geometry::RunP1P4MetricOracles(ctx);
  const auto permutation_evidence =
    dgsem_reference_geometry::RunNonconstantPermutationOracle(ctx);
  orientation_max_error = permutation_evidence.geometry_max_error;

  // Frozen polynomial geometry for the curl-form contravariant metrics:
  // x=xi+0.1*xi*eta, y=eta+0.05*xi^2.  The analytic adjugate and its
  // discrete divergence are both deterministic at every p3 LGL node.
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mesh.SetCurvature(4, false, dim, mfem::Ordering::byNODES);
    mesh.Transform([](const mfem::Vector& reference, mfem::Vector& physical)
    {
      physical.SetSize(2);
      physical(0) =
        reference(0) + 0.1 * reference(0) * reference(1);
      physical(1) = reference(1) + 0.05 * reference(0) * reference(0);
    });
    constexpr int order = 3;
    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
    mfem::FiniteElementSpace fes(&mesh, &fec);
    const mfem::FiniteElement& fe = *fes.GetFE(0);
    std::vector<mfem::dgsem::ContravariantMetric2D> metrics;
    mfem::dgsem::BuildContravariantMetrics2D(
      fe, *fes.GetElementTransformation(0), metrics);
    const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(fe);
    const auto& basis = mfem::dgsem::GetLGLBasis1D(order);
    std::vector<int> lex_to_local(static_cast<std::size_t>(fe.GetDof()), -1);
    for (int local = 0; local < fe.GetDof(); ++local)
    {
      const int i = dof_to_ijk[static_cast<std::size_t>(local)][0];
      const int j = dof_to_ijk[static_cast<std::size_t>(local)][1];
      lex_to_local[static_cast<std::size_t>(i + basis.n * j)] = local;
      const double xi = basis.nodes(i);
      const double eta = basis.nodes(j);
      const double expected[4] = {1.0,
                                  -0.1 * xi,
                                  -0.1 * xi,
                                  1.0 + 0.1 * eta};
      for (int entry = 0; entry < 4; ++entry)
      {
        metric_fixture_max_error = std::max(
          metric_fixture_max_error,
          std::abs(metrics[static_cast<std::size_t>(local)].data[entry] -
                   expected[entry]));
      }
    }
    for (int j = 0; j < basis.n; ++j)
    {
      for (int i = 0; i < basis.n; ++i)
      {
        for (int physical = 0; physical < dim; ++physical)
        {
          double divergence = 0.0;
          for (int m = 0; m < basis.n; ++m)
          {
            const int xi_local =
              lex_to_local[static_cast<std::size_t>(m + basis.n * j)];
            const int eta_local =
              lex_to_local[static_cast<std::size_t>(i + basis.n * m)];
            divergence +=
              basis.D(i, m) *
                metrics[static_cast<std::size_t>(xi_local)](0, physical) +
              basis.D(j, m) *
                metrics[static_cast<std::size_t>(eta_local)](1, physical);
          }
          metric_identity_max_error =
            std::max(metric_identity_max_error, std::abs(divergence));
        }
      }
    }
    metric_fixture_max_error = GlobalMax(metric_fixture_max_error);
    metric_identity_max_error = GlobalMax(metric_identity_max_error);
    ctx.Check(metric_fixture_max_error < fixture_tol,
              "DGSEM frozen contravariant polynomial metric fixture");
    ctx.Check(metric_identity_max_error <
                mfem::dgsem::DGSEMReferenceTolerances::sbp,
              "DGSEM discrete contravariant metric identity");
  }
  // Frozen affine p1 inverse-mass fixture. The unit-square tensor LGL mass is
  // diag(1/4), hence every equation/node entry is multiplied by exactly four.
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mfem::DG_FECollection fec(1, dim, mfem::BasisType::GaussLobatto);
    mfem::FiniteElementSpace vfes(
      &mesh, &fec, num_eq, mfem::Ordering::byNODES);
    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides numerical_flux(flux);
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(
      vfes, std::move(form_integrator), true);
    mfem::dgsem::DGSEMOperator dgsem_oper(oper);
    mfem::Vector weak(vfes.GetVSize());
    mfem::Vector strong(vfes.GetVSize());
    for (int i = 0; i < weak.Size(); ++i)
    {
      weak(i) = 0.125 * static_cast<double>(i + 1);
    }
    dgsem_oper.ApplyInverseMass(weak, strong);
    for (int i = 0; i < weak.Size(); ++i)
    {
      inverse_mass_max_error =
        std::max(inverse_mass_max_error, std::abs(strong(i) - 4.0 * weak(i)));
    }
    ctx.Check(inverse_mass_max_error <
                mfem::dgsem::DGSEMReferenceTolerances::inverse_mass,
              "DGSEM frozen affine p1 inverse-mass action");
  }

  // The same curved periodic surface-action fixture is local-only on one rank
  // and shared-only with the explicit checkerboard partition on two ranks.
  {
    constexpr int order = 2;
    const int world_size = mfem::Mpi::WorldSize();
    ctx.Check(world_size == 1 || world_size == 2,
              "DGSEM curved surface fixture is bounded to one/two ranks");
    mfem::Mesh mesh = MakePeriodicReferenceMesh2D(4, 4, order + 1);
    std::vector<int> partition(static_cast<std::size_t>(mesh.GetNE()), 0);
    if (world_size == 2)
    {
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
        const int i = element % 4;
        const int j = element / 4;
        partition[static_cast<std::size_t>(element)] = (i + j) % 2;
      }
    }
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.data());
    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(
      &pmesh, &fec, num_eq, mfem::Ordering::byNODES);

    // Shared element transformations require MFEM's geometry-neighbor data.
    // The production surface path performs the equivalent state exchange.
    pmesh.ExchangeFaceNbrData();
    int local_faces_local = 0;
    double local_normal_error = 0.0;
    for (int face_index = 0; face_index < pmesh.GetNumFaces(); ++face_index)
    {
      mfem::FaceElementTransformations* face =
        pmesh.GetFaceElementTransformations(face_index);
      if (!face || face->Elem2No < 0)
      {
        continue;
      }
      ++local_faces_local;
      local_normal_error = std::max(
        local_normal_error,
        dgsem_reference_geometry::AuditCurvedFaceMetricPair(
          *face, *vfes.GetFE(face->Elem1No), *vfes.GetFE(face->Elem2No)));
    }
    double shared_normal_error = 0.0;
    for (int shared = 0; shared < pmesh.GetNSharedFaces(); ++shared)
    {
      mfem::FaceElementTransformations* face =
        pmesh.GetSharedFaceTransformations(shared, true);
      const int neighbor = face->Elem2No - pmesh.GetNE();
      shared_normal_error = std::max(
        shared_normal_error,
        dgsem_reference_geometry::AuditCurvedFaceMetricPair(
          *face, *vfes.GetFE(face->Elem1No), *vfes.GetFaceNbrFE(neighbor)));
    }
    local_interior_face_count = static_cast<int>(
      GlobalSum(static_cast<double>(local_faces_local)));
    shared_face_count = static_cast<int>(
      GlobalSum(static_cast<double>(pmesh.GetNSharedFaces())));
    curved_local_face_normal_error = GlobalMax(local_normal_error);
    curved_shared_face_normal_error = GlobalMax(shared_normal_error);
    if (world_size == 1)
    {
      ctx.Check(local_interior_face_count > 0 && shared_face_count == 0,
                "DGSEM curved serial fixture encounters local faces only");
      ctx.Check(curved_local_face_normal_error < fixture_tol,
                "DGSEM curved local endpoint normals are pair-consistent");
    }
    else
    {
      ctx.Check(local_interior_face_count == 0 && shared_face_count > 0,
                "DGSEM curved two-rank fixture encounters shared faces only");
      ctx.Check(curved_shared_face_normal_error < fixture_tol,
                "DGSEM curved shared endpoint normals are pair-consistent");
    }
    mfem::ParGridFunction state(&vfes);
    auto state_function = [=](const mfem::Vector& x, mfem::Vector& value)
    {
      const double two_pi = 2.0 * dg_sim::math::kPi;
      const double rho =
        1.0 + 0.08 * std::sin(two_pi * x(0)) +
        0.04 * std::cos(two_pi * x(1));
      mfem::Vector primitive(num_eq);
      primitive(0) = rho;
      primitive(1) = 0.2;
      primitive(2) = -0.1;
      primitive(3) = 1.0;
      mfem::EulerPrimToConservative(primitive, dim, gamma, value);
    };
    mfem::VectorFunctionCoefficient coefficient(num_eq, state_function);
    state.ProjectCoefficient(coefficient);

    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides numerical_flux(flux);
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(
      vfes, std::move(form_integrator), true);
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order));
    mfem::Vector x;
    state.GetTrueDofs(x);

    mfem::ParGridFunction constant_state(&vfes);
    FillConstantState(
      EulerConsFromPrim2D(1.0, 0.2, -0.1, 1.0, gamma), constant_state);
    mfem::Vector constant_x;
    mfem::Vector constant_residual(vfes.GetTrueVSize());
    constant_state.GetTrueDofs(constant_x);
    oper.Mult(constant_x, constant_residual);
    periodic_constant_norm = GlobalL2Norm(constant_residual);
    ctx.Check(periodic_constant_norm <
                mfem::dgsem::DGSEMReferenceTolerances::constant_curved,
              "DGSEM curved periodic constant state is preserved");
    mfem::dgsem::DGSEMOperator dgsem_oper(oper);
    mfem::Vector surface(vfes.GetVSize());
    dgsem_oper.AssembleSurface(x, surface);
    shared_surface_norm = GlobalL2Norm(surface);
    ctx.Check(shared_surface_norm > 1.0e-8,
              "DGSEM curved local/shared surface fixture has nonzero action");
    for (int eq = 0; eq < num_eq; ++eq)
    {
      surface_component_norms[eq] =
        WeakComponentL2Norm(vfes, surface, eq);
      shared_balances[eq] = WeakComponentSum(vfes, surface, eq);
      ctx.Check(surface_component_norms[eq] > 1.0e-10,
                "DGSEM curved surface component has nonzero action " +
                  std::to_string(eq));
      ctx.Check(std::abs(shared_balances[eq]) <
                  mfem::dgsem::DGSEMReferenceTolerances::face_balance,
                "DGSEM separated local/shared surface conservation equation " +
                  std::to_string(eq));
    }

    // Component-wise one/two-rank parity. This intentionally does not use the
    // suite-level aggregate parity scalar, which could hide component swaps.
    const char* component_parity_file =
      std::getenv("DG_IGR_DGSEM_COMPONENT_PARITY_FILE");
    if (component_parity_file && mfem::Mpi::Root())
    {
      if (world_size == 1)
      {
        std::ofstream reference(component_parity_file);
        reference << std::setprecision(
          std::numeric_limits<double>::max_digits10);
        for (int eq = 0; eq < num_eq; ++eq)
        {
          reference << surface_component_norms[eq] << '\n';
        }
        ctx.Check(reference.good(),
                  "DGSEM component-wise one-rank reference written");
      }
      else
      {
        std::ifstream reference(component_parity_file);
        for (int eq = 0; eq < num_eq; ++eq)
        {
          double expected_norm = 0.0;
          reference >> expected_norm;
          ctx.CheckNear(surface_component_norms[eq],
                        expected_norm,
                        mfem::dgsem::DGSEMReferenceTolerances::mpi_parity,
                        "DGSEM local/shared component-wise rank parity " +
                          std::to_string(eq));
        }
        ctx.Check(reference.good() || reference.eof(),
                  "DGSEM component-wise one-rank reference read");
      }
    }
    std::vector<double> conserved_before(num_eq, 0.0);
    for (int eq = 0; eq < num_eq; ++eq)
    {
      conserved_before[eq] = IntegrateComponent(state, eq);
    }
    mfem::RK3SSPWithLimiter rk(vfes);
    rk.Init(oper);
    double time = 0.0;
    double dt = 1.0e-4;
    rk.Step(x, time, dt);
    state.SetFromTrueDofs(x);
    for (int eq = 0; eq < num_eq; ++eq)
    {
      const double conserved_after = IntegrateComponent(state, eq);
      periodic_conservation_deltas[eq] =
        conserved_after - conserved_before[eq];
      ctx.CheckNear(conserved_after,
                    conserved_before[eq],
                    mfem::dgsem::DGSEMReferenceTolerances::conservation,
                    "DGSEM periodic conserved integral for equation " +
                      std::to_string(eq));
    }
  }

  // Wall constant-state preservation and the physically admissible closed-wall
  // conservation claim: mass and total energy, not momentum, are invariant.
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      3, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    constexpr int order = 3;
    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(
      &pmesh, &fec, num_eq, mfem::Ordering::byNODES);
    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides numerical_flux(flux);
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux,
                                                       int_order_offset);
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
        std::make_unique<mfem::SlipWallState>(dim),
        int_order_offset),
      all_boundary);

    mfem::ParGridFunction state(&vfes);
    FillConstantState(EulerConsFromPrim2D(1.0, 0.0, 0.0, 1.0, gamma), state);
    mfem::Vector x;
    mfem::Vector residual(vfes.GetTrueVSize());
    state.GetTrueDofs(x);
    oper.Mult(x, residual);
    wall_constant_norm = GlobalL2Norm(residual);
    ctx.Check(wall_constant_norm <
                mfem::dgsem::DGSEMReferenceTolerances::constant_affine,
              "DGSEM slip-wall rest state is preserved");

    auto closed_state = [=](const mfem::Vector& pos, mfem::Vector& value)
    {
      const double rho =
        1.0 + 0.06 * std::cos(2.0 * dg_sim::math::kPi * pos(0)) *
                std::cos(2.0 * dg_sim::math::kPi * pos(1));
      const double pressure =
        1.0 + 0.03 * std::cos(dg_sim::math::kPi * pos(0)) *
                std::cos(dg_sim::math::kPi * pos(1));
      mfem::Vector primitive(num_eq);
      primitive(0) = rho;
      primitive(1) = 0.0;
      primitive(2) = 0.0;
      primitive(3) = pressure;
      mfem::EulerPrimToConservative(primitive, dim, gamma, value);
    };
    mfem::VectorFunctionCoefficient closed_coefficient(num_eq, closed_state);
    state.ProjectCoefficient(closed_coefficient);
    const double mass_before = IntegrateComponent(state, 0);
    const double energy_before = IntegrateComponent(state, 3);
    state.GetTrueDofs(x);
    mfem::RK3SSPWithLimiter rk(vfes);
    rk.Init(oper);
    double time = 0.0;
    double dt = 1.0e-4;
    rk.Step(x, time, dt);
    state.SetFromTrueDofs(x);
    const double mass_after = IntegrateComponent(state, 0);
    const double energy_after = IntegrateComponent(state, 3);
    wall_mass_delta = mass_after - mass_before;
    wall_energy_delta = energy_after - energy_before;
    ctx.CheckNear(mass_after,
                  mass_before,
                  mfem::dgsem::DGSEMReferenceTolerances::conservation,
                  "DGSEM closed slip-wall mass is conserved");
    ctx.CheckNear(energy_after,
                  energy_before,
                  mfem::dgsem::DGSEMReferenceTolerances::conservation,
                  "DGSEM closed slip-wall total energy is conserved");
  }

  // Bounded micro-attribution: setup is separate and each isolated operation
  // uses nine samples with median and median absolute deviation. Categories
  // can overlap and are not additive whole-stage timings. This is reference
  // attribution, not an optimization or a performance threshold.
  std::vector<AttributionRecord> attribution;
  double setup_seconds = 0.0;
  {
    const auto setup_start = std::chrono::steady_clock::now();
    mfem::Mesh mesh = MakePeriodicReferenceMesh2D(6, 6);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    constexpr int order = 3;
    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(
      &pmesh, &fec, num_eq, mfem::Ordering::byNODES);
    mfem::ParGridFunction state(&vfes);
    auto state_function = [=](const mfem::Vector& pos, mfem::Vector& value)
    {
      const double rho =
        1.0 + 0.03 * std::sin(2.0 * dg_sim::math::kPi * pos(0)) *
                std::cos(2.0 * dg_sim::math::kPi * pos(1));
      mfem::Vector primitive(num_eq);
      primitive(0) = rho;
      primitive(1) = 0.25;
      primitive(2) = -0.15;
      primitive(3) = 1.0;
      mfem::EulerPrimToConservative(primitive, dim, gamma, value);
    };
    mfem::VectorFunctionCoefficient coefficient(num_eq, state_function);
    state.ProjectCoefficient(coefficient);

    mfem::DG_FECollection sigma_fec(2, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace sigma_fes(
      &pmesh, &sigma_fec, 1, mfem::Ordering::byNODES);
    mfem::ParGridFunction sigma(&sigma_fes);
    mfem::ConstantCoefficient sigma_coefficient(0.125);
    sigma.ProjectCoefficient(sigma_coefficient);
    sigma.ExchangeFaceNbrData();

    mfem::EulerFlux euler_flux(dim, gamma);
    mfem::FluxWithIGR igr_flux(euler_flux);
    igr_flux.SetSigmaField(&sigma);
    mfem::DGSEMSurfaceFlux numerical_flux(igr_flux);
    numerical_flux.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::Rusanov);
    numerical_flux.SetGamma(gamma);
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(
      vfes, std::move(form_integrator), true);
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(order));
    mfem::dgsem::DGSEMOperator dgsem_oper(oper);
    mfem::Vector x;
    state.GetTrueDofs(x);
    mfem::Vector weak(vfes.GetVSize());
    mfem::Vector strong(vfes.GetVSize());
    const auto setup_stop = std::chrono::steady_clock::now();
    setup_seconds = GlobalMaxSeconds(
      std::chrono::duration<double>(setup_stop - setup_start).count());

    // One untimed warmup of the assembled phases.
    dgsem_oper.AssembleSurface(x, weak);
    weak = 0.0;
    dgsem_oper.AssembleVolume(x, weak);
    dgsem_oper.ApplyInverseMass(weak, strong);
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> surface_samples;
    std::vector<double> volume_samples;
    std::vector<double> auxiliary_samples;
    std::vector<double> inverse_mass_samples;
    std::vector<double> reduction_samples;
    std::vector<double> exchange_samples;
    std::vector<double> output_samples;
    std::vector<double> synchronization_samples;
    std::vector<double> residual_samples;
    constexpr std::size_t samples = mfem::dgsem::kDGSEMAttributionSamples;
    surface_samples.reserve(samples);
    volume_samples.reserve(samples);
    auxiliary_samples.reserve(samples);
    inverse_mass_samples.reserve(samples);
    reduction_samples.reserve(samples);
    exchange_samples.reserve(samples);
    output_samples.reserve(samples);
    synchronization_samples.reserve(samples);
    residual_samples.reserve(samples);

    const char* output_env =
      std::getenv("DG_IGR_DGSEM_REFERENCE_OUTPUT_DIR");
    const bool filesystem_output = output_env && output_env[0];
    std::filesystem::path output_path;
    if (filesystem_output)
    {
      output_path = output_env;
      std::error_code ec;
      std::filesystem::create_directories(output_path, ec);
      ctx.Check(!ec, "DGSEM attribution output directory is writable");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    volatile double sink = 0.0;
    for (std::size_t sample = 0; sample < samples; ++sample)
    {
      surface_samples.push_back(TimeGlobalMaximum([&]()
      {
        dgsem_oper.AssembleSurface(x, weak);
      }));
      volume_samples.push_back(TimeGlobalMaximum([&]()
      {
        weak = 0.0;
        dgsem_oper.AssembleVolume(x, weak);
      }));
      auxiliary_samples.push_back(TimeGlobalMaximum([&]()
      {
        double local_sigma_sum = 0.0;
        for (int element = 0; element < vfes.GetNE(); ++element)
        {
          mfem::ElementTransformation* tr =
            vfes.GetElementTransformation(element);
          const mfem::IntegrationRule& nodes = vfes.GetFE(element)->GetNodes();
          for (int node = 0; node < nodes.GetNPoints(); ++node)
          {
            const mfem::IntegrationPoint& ip = nodes.IntPoint(node);
            tr->SetIntPoint(&ip);
            local_sigma_sum += igr_flux.GetSigmaAtPoint(*tr);
          }
        }
        sink += local_sigma_sum;
      }));
      inverse_mass_samples.push_back(TimeGlobalMaximum([&]()
      {
        dgsem_oper.ApplyInverseMass(weak, strong);
      }));
      reduction_samples.push_back(TimeGlobalMaximum([&]()
      {
        double local = static_cast<double>(sample + 1);
        double global = 0.0;
        MPI_Allreduce(&local,
                      &global,
                      1,
                      mfem::MPITypeMap<double>::mpi_type,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        sink += global;
      }));
      exchange_samples.push_back(TimeGlobalMaximum([&]()
      {
        sigma.ExchangeFaceNbrData();
      }));
      output_samples.push_back(TimeGlobalMaximum([&]()
      {
        if (filesystem_output)
        {
          const std::filesystem::path sample_path =
            output_path /
            ("attribution_state_rank" +
             std::to_string(mfem::Mpi::WorldRank()) + ".gf");
          std::ofstream stream(sample_path, std::ios::trunc);
          state.Save(stream);
          stream.flush();
        }
        else
        {
          std::ostringstream stream;
          state.Save(stream);
          sink += static_cast<double>(stream.str().size());
        }
      }));
      synchronization_samples.push_back(TimeGlobalMaximum([&]()
      {
        MPI_Barrier(MPI_COMM_WORLD);
      }));
      residual_samples.push_back(TimeGlobalMaximum([&]()
      {
        oper.Mult(x, strong);
      }));
    }
    ctx.Check(std::isfinite(static_cast<double>(sink)),
              "DGSEM attribution work is retained");
    attribution = {{"volume", MedianMAD(volume_samples)},
                   {"surface", MedianMAD(surface_samples)},
                   {"auxiliary", MedianMAD(auxiliary_samples)},
                   {"inverse_mass", MedianMAD(inverse_mass_samples)},
                   {"reductions", MedianMAD(reduction_samples)},
                   {"exchange", MedianMAD(exchange_samples)},
                   {"output", MedianMAD(output_samples)},
                   {"synchronization", MedianMAD(synchronization_samples)},
                   {"total_residual", MedianMAD(residual_samples)}};
    for (const auto& record : attribution)
    {
      ctx.Check(record.statistic.median >= 0.0 &&
                  record.statistic.mad >= 0.0 &&
                  std::isfinite(record.statistic.median) &&
                  std::isfinite(record.statistic.mad),
                std::string("DGSEM bounded CPU micro-attribution recorded: ") +
                  record.category);
      ctx.LogMetric(std::string("DGSEM CPU micro-attribution ") +
                      record.category +
                    ": median_s=" + FormatSci(record.statistic.median) +
                    ", mad_s=" + FormatSci(record.statistic.mad));
    }

    if (filesystem_output && mfem::Mpi::Root())
    {
      std::ofstream out(output_path / "cpu_attribution.json");
      out << "{\n"
          << "  \"schema\": \"dgsem-cpu-micro-attribution-v1\",\n"
          << "  \"ranks\": " << mfem::Mpi::WorldSize() << ",\n"
          << "  \"samples\": " << samples << ",\n"
          << "  \"dispersion\": \"median_absolute_deviation\",\n"
          << "  \"setup_seconds\": " << setup_seconds << ",\n"
          << "  \"output_mode\": \"filesystem\",\n"
          << "  \"scope\": \"isolated overlapping operations; not additive "
             "whole-stage attribution\",\n"
          << "  \"volume_scope\": \"AssembleVolume-only\",\n"
          << "  \"surface_scope\": \"AssembleSurface including state "
             "ExchangeFaceNbrData\",\n"
          << "  \"auxiliary_scope\": \"GetSigmaAtPoint-only\",\n"
          << "  \"exchange_scope\": \"sigma.ExchangeFaceNbrData-only\",\n"
          << "  \"reductions_scope\": \"standalone MPI_Allreduce-only\",\n"
          << "  \"synchronization_scope\": \"standalone MPI_Barrier-only\",\n"
          << "  \"categories\": {\n";
      for (std::size_t i = 0; i < attribution.size(); ++i)
      {
        const auto& record = attribution[i];
        out << "    \"" << record.category << "\": {\"median_seconds\": "
            << record.statistic.median << ", \"mad_seconds\": "
            << record.statistic.mad << "}"
            << (i + 1 == attribution.size() ? "\n" : ",\n");
      }
      out << "  }\n}\n";
      ctx.Check(out.good(),
                "DGSEM machine-readable CPU micro-attribution written");
    }
  }

  const char* output_dir = std::getenv("DG_IGR_DGSEM_REFERENCE_OUTPUT_DIR");
  if (output_dir && mfem::Mpi::Root())
  {
    const std::filesystem::path path =
      std::filesystem::path(output_dir) /
      ("operator_fixtures_ranks_" +
       std::to_string(mfem::Mpi::WorldSize()) + ".json");
    std::ofstream out(path);
    out << "{\n"
        << "  \"schema\": \"dgsem-operator-fixtures-v2\",\n"
        << "  \"ranks\": " << mfem::Mpi::WorldSize() << ",\n";
    auto write_p1p4 = [&](const char* name,
                          const std::array<double, 4>& values)
    {
      out << "  \"" << name << "\": [";
      for (std::size_t i = 0; i < values.size(); ++i)
      {
        out << values[i] << (i + 1 == values.size() ? "" : ",");
      }
      out << "],\n";
    };
    write_p1p4("p1_p4_affine_adjugate_max_error",
               p1p4_metric_evidence.affine_adjugate_max_error);
    write_p1p4("p1_p4_affine_face_normal_max_error",
               p1p4_metric_evidence.affine_face_normal_max_error);
    write_p1p4("p1_p4_curved_metric_identity_max_error",
               p1p4_metric_evidence.curved_metric_identity_max_error);
    out << "  \"permutation_geometry_max_error\": "
        << permutation_evidence.geometry_max_error << ",\n"
        << "  \"permutation_component_moment_max_error\": "
        << permutation_evidence.component_moment_max_error << ",\n"
        << "  \"permutation_canonical_surface_norm\": "
        << permutation_evidence.canonical_surface_norm << ",\n"
        << "  \"permutation_surface_norm\": "
        << permutation_evidence.permuted_surface_norm << ",\n"
        << "  \"permutation_local_interior_faces\": "
        << permutation_evidence.local_interior_faces << ",\n"
        << "  \"permutation_shared_faces\": "
        << permutation_evidence.shared_faces << ",\n"
        << "  \"contravariant_metric_max_error\": "
        << metric_fixture_max_error << ",\n"
        << "  \"metric_identity_max_error\": "
        << metric_identity_max_error << ",\n"
        << "  \"inverse_mass_max_error\": " << inverse_mass_max_error
        << ",\n"
        << "  \"orientation_max_error\": " << orientation_max_error
        << ",\n"
        << "  \"surface_action_norm\": " << shared_surface_norm << ",\n"
        << "  \"surface_action_class\": \""
        << (mfem::Mpi::WorldSize() == 1 ? "local-only" : "shared-only")
        << "\",\n"
        << "  \"local_interior_face_count\": "
        << local_interior_face_count << ",\n"
        << "  \"shared_face_count\": " << shared_face_count << ",\n"
        << "  \"curved_local_face_normal_max_error\": "
        << curved_local_face_normal_error << ",\n"
        << "  \"curved_shared_face_normal_max_error\": "
        << curved_shared_face_normal_error << ",\n"
        << "  \"surface_component_norms\": [";
    for (int eq = 0; eq < num_eq; ++eq)
    {
      out << surface_component_norms[eq]
          << (eq + 1 == num_eq ? "" : ",");
    }
    out << "],\n"
        << "  \"surface_equation_balances\": [";
    for (int eq = 0; eq < num_eq; ++eq)
    {
      out << shared_balances[eq] << (eq + 1 == num_eq ? "" : ",");
    }
    out << "],\n"
        << "  \"periodic_constant_residual_norm\": "
        << periodic_constant_norm << ",\n"
        << "  \"periodic_conservation_deltas\": [";
    for (int eq = 0; eq < num_eq; ++eq)
    {
      out << periodic_conservation_deltas[eq]
          << (eq + 1 == num_eq ? "" : ",");
    }
    out << "],\n"
        << "  \"wall_constant_residual_norm\": " << wall_constant_norm
        << ",\n"
        << "  \"wall_mass_delta\": " << wall_mass_delta << ",\n"
        << "  \"wall_energy_delta\": " << wall_energy_delta << "\n"
        << "}\n";
    ctx.Check(out.good(), "DGSEM machine-readable operator fixtures written");
  }
}

} // namespace dg_igr_tests
