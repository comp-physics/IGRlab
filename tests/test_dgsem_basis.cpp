#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_reference.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

namespace dg_igr_tests
{

namespace
{

double MaxAbsEntry(const mfem::DenseMatrix& mat)
{
  double max_val = 0.0;
  for (int i = 0; i < mat.Height(); ++i)
  {
    for (int j = 0; j < mat.Width(); ++j)
    {
      max_val = std::max(max_val, std::abs(mat(i, j)));
    }
  }
  return max_val;
}

} // namespace

int RunDGSEMBasisTests(TestContext& ctx)
{
  const double tol = mfem::dgsem::DGSEMReferenceTolerances::lgl;
  const int orders[] = {1, 2, 3, 4};
  constexpr double frozen_base_lgl_and_lex = 1.0e-12;
  constexpr double frozen_base_sbp_and_dsplit = 1.0e-10;
  static_assert(mfem::dgsem::DGSEMReferenceTolerances::lgl ==
                  frozen_base_lgl_and_lex,
                "DGSEM LGL/lex gate must equal the frozen base.");
  ctx.Check(tol == frozen_base_lgl_and_lex,
            "DGSEM LGL/lex tolerance equals frozen-base 1e-12");
  ctx.Check(mfem::dgsem::DGSEMReferenceTolerances::sbp <=
              frozen_base_sbp_and_dsplit,
            "DGSEM SBP/Dsplit tolerance is no weaker than frozen base");
  const bool preexisting_tolerances_relaxed =
    tol > frozen_base_lgl_and_lex ||
    mfem::dgsem::DGSEMReferenceTolerances::sbp >
      frozen_base_sbp_and_dsplit;

  for (int order : orders)
  {
    const auto& basis = mfem::dgsem::GetLGLBasis1D(order);

    double weight_sum = 0.0;
    for (int i = 0; i < basis.n; ++i)
    {
      weight_sum += basis.weights(i);
    }
    ctx.CheckNear(
      weight_sum,
      1.0,
      tol,
      "LGL weights sum to 1 on [0,1] (order=" + std::to_string(order) + ")");

    // Freeze quadrature exactness through degree 2p-1.
    for (int degree = 0; degree <= 2 * order - 1; ++degree)
    {
      double moment = 0.0;
      for (int i = 0; i < basis.n; ++i)
      {
        moment += basis.weights(i) * std::pow(basis.nodes(i), degree);
      }
      ctx.CheckNear(moment,
                    1.0 / static_cast<double>(degree + 1),
                    tol,
                    "LGL polynomial moment is exact (p=" +
                      std::to_string(order) + ", degree=" +
                      std::to_string(degree) + ")");
    }

    // Freeze nodal interpolation for every represented polynomial degree.
    mfem::Poly_1D::Basis& poly_basis =
      mfem::poly1d.GetBasis(order, mfem::BasisType::GaussLobatto);
    mfem::Vector shape(basis.n);
    const double sample_points[] = {0.137, 0.431, 0.793};
    for (int degree = 0; degree <= order; ++degree)
    {
      for (const double x : sample_points)
      {
        poly_basis.Eval(x, shape);
        double interpolated = 0.0;
        for (int i = 0; i < basis.n; ++i)
        {
          interpolated += shape(i) * std::pow(basis.nodes(i), degree);
        }
        ctx.CheckNear(interpolated,
                      std::pow(x, degree),
                      tol,
                      "LGL interpolation is exact (p=" +
                        std::to_string(order) + ", degree=" +
                        std::to_string(degree) + ")");
      }
    }

    // SBP identity: M D + D^T M = B.
    mfem::DenseMatrix M(basis.n);
    M = 0.0;
    for (int i = 0; i < basis.n; ++i)
    {
      M(i, i) = basis.weights(i);
    }

    mfem::DenseMatrix MD(basis.n);
    mfem::DenseMatrix DtM(basis.n);
    mfem::Mult(M, basis.D, MD);
    mfem::DenseMatrix Dt;
    Dt.Transpose(basis.D);
    mfem::Mult(Dt, M, DtM);

    mfem::DenseMatrix sbp(basis.n);
    sbp = 0.0;
    sbp += MD;
    sbp += DtM;

    mfem::DenseMatrix B(basis.n);
    B = 0.0;
    if (basis.n > 0)
    {
      B(0, 0) = -1.0;
      B(basis.n - 1, basis.n - 1) = 1.0;
    }

    sbp -= B;
    const double sbp_err = MaxAbsEntry(sbp);
    ctx.Check(sbp_err < mfem::dgsem::DGSEMReferenceTolerances::sbp,
              "SBP identity holds (order=" + std::to_string(order) + ")");

    // Dsplit consistency: 2 D - M^{-1} B.
    mfem::DenseMatrix Dsplit_expected(basis.n);
    Dsplit_expected = basis.D;
    for (int i = 0; i < basis.n; ++i)
    {
      for (int j = 0; j < basis.n; ++j)
      {
        Dsplit_expected(i, j) *= 2.0;
      }
    }
    for (int i = 0; i < basis.n; ++i)
    {
      const double b = (i == 0) ? -1.0 : (i == basis.n - 1 ? 1.0 : 0.0);
      Dsplit_expected(i, i) -= b * basis.inv_weights(i);
    }
    Dsplit_expected -= basis.Dsplit;
    const double dsplit_err = MaxAbsEntry(Dsplit_expected);
    ctx.Check(
      dsplit_err < mfem::dgsem::DGSEMReferenceTolerances::sbp,
      "Dsplit matches 2 D - M^{-1} B (order=" + std::to_string(order) + ")");
  }

  // Lexicographic tensor mapping (1D + 2D, p1-p4, Gauss-Lobatto).
  for (const int order : orders)
  {
    const mfem::real_t* nodes =
      mfem::poly1d.GetPoints(order, mfem::BasisType::GaussLobatto);

    mfem::DG_FECollection fec_1d(order, 1, mfem::BasisType::GaussLobatto);
    const mfem::FiniteElement* fe_1d =
      fec_1d.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
    const auto& map_1d = mfem::dgsem::GetTensorDofToIJK(*fe_1d);
    for (int ld = 0; ld < fe_1d->GetDof(); ++ld)
    {
      const int i = map_1d[ld][0];
      const double x = fe_1d->GetNodes().IntPoint(ld).x;
      ctx.CheckNear(x,
                    nodes[i],
                    tol,
                    "1D lexicographic mapping matches nodes (p=" +
                      std::to_string(order) + ")");
    }

    mfem::DG_FECollection fec_2d(order, 2, mfem::BasisType::GaussLobatto);
    const mfem::FiniteElement* fe_2d =
      fec_2d.FiniteElementForGeometry(mfem::Geometry::SQUARE);
    const auto& map_2d = mfem::dgsem::GetTensorDofToIJK(*fe_2d);
    for (int ld = 0; ld < fe_2d->GetDof(); ++ld)
    {
      const int i = map_2d[ld][0];
      const int j = map_2d[ld][1];
      const mfem::IntegrationPoint& ip = fe_2d->GetNodes().IntPoint(ld);
      ctx.CheckNear(ip.x,
                    nodes[i],
                    tol,
                    "2D lexicographic x matches nodes (p=" +
                      std::to_string(order) + ")");
      ctx.CheckNear(ip.y,
                    nodes[j],
                    tol,
                    "2D lexicographic y matches nodes (p=" +
                      std::to_string(order) + ")");
    }
  }

  // Freeze the production/reference state and auxiliary consumer inventory.
  const std::set<std::string> required_consumers = {
    "DGHyperbolicConservationLaws::Mult",
    "DGSEMOperator::AssembleVolume",
    "DGSEMOperator::AssembleSurface",
    "DGSEMOperator::ApplyInverseMass",
    "DGSEMSurfaceFlux",
    "RusanovFluxWithSides",
    "DiffusiveRusanovFlux",
    "FluxWithIGR",
    "EulerIGRFluxBundle sigma ownership handoff",
    "BdrHyperbolicFluxIntegrator/BoundaryState",
    "RK3SSPWithLimiter",
    "limiter pipeline",
    "solver/backend wiring",
    "DGSEM flux wiring",
    "diagnostics/output/restart",
    "smooth-advection examples",
    "wall/shock examples",
    "blast/vortex examples",
    "registered DGSEM reference suites"};
  bool saw_flux_bundle_handoff = false;
  std::set<std::string> consumer_names;
  for (const auto& consumer : mfem::dgsem::kDGSEMReferenceConsumers)
  {
    const bool complete = consumer.consumer && consumer.consumer[0] &&
                          consumer.path && consumer.path[0] &&
                          consumer.state_contract && consumer.state_contract[0] &&
                          consumer.auxiliary_contract &&
                          consumer.auxiliary_contract[0];
    ctx.Check(complete,
              std::string("DGSEM consumer inventory record is complete: ") +
                consumer.consumer);
    const bool relative_specific_path = consumer.path &&
                                        consumer.path[0] != '/' &&
                                        std::string(consumer.path).find('/') !=
                                          std::string::npos;
    ctx.Check(consumer_names.insert(consumer.consumer).second,
              std::string("DGSEM consumer inventory name is unique: ") +
                consumer.consumer);
    ctx.Check(relative_specific_path,
              std::string("DGSEM consumer inventory path is scoped: ") +
                consumer.consumer);
    if (std::string(consumer.consumer) ==
        "EulerIGRFluxBundle sigma ownership handoff")
    {
      saw_flux_bundle_handoff =
        std::string(consumer.path) ==
          "include/sim/wiring/flux_bundle.hpp:29" &&
        std::string(consumer.auxiliary_contract).find("IGRSolver-owned") !=
          std::string::npos &&
        std::string(consumer.auxiliary_contract).find("non-owning") !=
          std::string::npos;
    }
  }
  ctx.Check(std::string(mfem::dgsem::kDGSEMScalarIGRScope).find("B-SCI-001") !=
              std::string::npos,
            "DGSEM scalar-IGR scope explicitly preserves B-SCI-001");
  ctx.Check(std::string(mfem::dgsem::kDGSEMScalarIGRScope).find("CONTRACT-01") !=
              std::string::npos,
            "DGSEM scalar-IGR scope explicitly avoids CONTRACT-01 dependency");

  ctx.Check(consumer_names == required_consumers,
            "DGSEM consumer inventory matches the frozen required set");
  ctx.Check(mfem::dgsem::kDGSEMReferenceConsumers.size() ==
              required_consumers.size(),
            "DGSEM consumer inventory has no missing or extra records");
  ctx.Check(saw_flux_bundle_handoff,
            "DGSEM inventory freezes the actual sigma ownership handoff");
  // Optional machine-readable inventory for the reference-gate evidence run.
  const char* output_dir = std::getenv("DG_IGR_DGSEM_REFERENCE_OUTPUT_DIR");
  if (output_dir && mfem::Mpi::Root())
  {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    const std::filesystem::path path =
      std::filesystem::path(output_dir) / "consumer_inventory.json";
    std::ofstream out(path);
    out << "{\n  \"schema\": \"dgsem-consumer-inventory-v1\",\n"
        << "  \"scalar_igr_scope\": \""
        << mfem::dgsem::kDGSEMScalarIGRScope << "\",\n"
        << "  \"consumers\": [\n";
    for (std::size_t i = 0;
         i < mfem::dgsem::kDGSEMReferenceConsumers.size();
         ++i)
    {
      const auto& consumer = mfem::dgsem::kDGSEMReferenceConsumers[i];
      out << "    {\"consumer\": \"" << consumer.consumer
          << "\", \"path\": \"" << consumer.path
          << "\", \"state_contract\": \"" << consumer.state_contract
          << "\", \"auxiliary_contract\": \""
          << consumer.auxiliary_contract << "\"}"
          << (i + 1 == mfem::dgsem::kDGSEMReferenceConsumers.size() ? "\n"
                                                                    : ",\n");
    }
    out << "  ]\n}\n";
    ctx.Check(!ec && out.good(),
              "DGSEM machine-readable consumer inventory written");

    std::ofstream tolerance_out(
      std::filesystem::path(output_dir) / "tolerance_audit.json");
    tolerance_out
      << "{\n"
      << "  \"schema\": \"dgsem-base-child-tolerance-audit-v1\",\n"
      << "  \"frozen_base\": {\"lgl_and_lex\": 1e-12, "
         "\"sbp_and_dsplit\": 1e-10},\n"
      << "  \"child\": {\"lgl_and_lex\": " << tol
      << ", \"sbp_and_dsplit\": "
      << mfem::dgsem::DGSEMReferenceTolerances::sbp << "},\n"
      << "  \"preexisting_tolerances_relaxed\": "
      << (preexisting_tolerances_relaxed ? "true" : "false") << ",\n"
      << "  \"metric_real_t_bytes\": " << sizeof(mfem::real_t) << ",\n"
      << "  \"double_pointwise_gate\": "
      << mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture << ",\n"
      << "  \"precision_aware_pointwise_gate\": "
      << mfem::dgsem::PrecisionAwareMetricTolerance(
           mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture)
      << ",\n"
      << "  \"mandatory_assertions_executed\": true\n"
      << "}\n";
    ctx.Check(!ec && tolerance_out.good(),
              "DGSEM machine-readable tolerance audit written");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests
