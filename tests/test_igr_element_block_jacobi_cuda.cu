#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_element_block_jacobi.hpp"
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/igr/IGR.hpp"
#include "physics/igr/igr_coupling.hpp"
#include "physics/igr/igr_flux.hpp"
#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
// Do not import mfem::igrlab into the translation-unit lookup scope: NVCC's
// generated registration stubs use unqualified igrlab::dg_face type names.
using mfem::Array;
using mfem::AssemblyLevel;
using mfem::BasisType;
using mfem::BdrHyperbolicFluxIntegrator;
using mfem::BilinearForm;
using mfem::Coefficient;
using mfem::DenseMatrix;
using mfem::DenseMatrixInverse;
using mfem::DenseTensor;
using mfem::DG_FECollection;
using mfem::DGDiffusionIntegrator;
using mfem::DGHyperbolicConservationLaws;
using mfem::DiffusionIntegrator;
using mfem::Element;
using mfem::EulerFlux;
using mfem::EulerWithIGR;
using mfem::FaceQuadratureSpace;
using mfem::FaceType;
using mfem::FluxWithIGR;
using mfem::FunctionCoefficient;
using mfem::Geometry;
using mfem::HyperbolicFormIntegrator;
using mfem::IntegrationRule;
using mfem::IntRules;
using mfem::MassIntegrator;
using mfem::Mesh;
using mfem::Ordering;
using mfem::ParFiniteElementSpace;
using mfem::ParGridFunction;
using mfem::ParMesh;
using mfem::Reshape;
using mfem::RusanovFluxWithSides;
using mfem::SlipWallState;
using mfem::Vector;
using mfem::VectorFunctionCoefficient;
using namespace ::igrlab::dg_face;

void AddDomain(BilinearForm& form, Coefficient& mass, Coefficient& diffusion,
                const IntegrationRule& mass_rule, const IntegrationRule& diffusion_rule)
{
  auto* m = new MassIntegrator(mass);
  auto* k = new DiffusionIntegrator(diffusion);
  m->SetIntRule(&mass_rule);
  k->SetIntRule(&diffusion_rule);
  form.AddDomainIntegrator(m);
  form.AddDomainIntegrator(k);
}

void FillFaceCoefficients(ParMesh& mesh, const FaceQuadratureSpace& fqs,
                           Coefficient& coefficient, Vector& side_q)
{
  const auto& rule = fqs.GetIntRule(0);
  side_q.SetSize(2 * rule.GetNPoints() * fqs.GetNE());
  auto qdata = Reshape(side_q.HostWrite(), 2, rule.GetNPoints(), fqs.GetNE());
  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    auto* tr = mesh.GetFaceElementTransformations(fqs.GetMeshFaceIndex(f));
    MFEM_VERIFY(tr && tr->Elem1 && tr->Elem2, "Expected a local interior face.");
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      tr->SetAllIntPoints(&rule.IntPoint(q));
      qdata(0, q, f) = coefficient.Eval(*tr->Elem1, tr->GetElement1IntPoint());
      qdata(1, q, f) = coefficient.Eval(*tr->Elem2, tr->GetElement2IntPoint());
    }
  }
}

double RelativeDifference(const Vector& x, const Vector& y)
{
  MFEM_VERIFY(x.Size() == y.Size(), "Mismatched fixture vectors.");
  const auto* a = x.HostRead();
  const auto* b = y.HostRead();
  long double delta = 0.0L, scale = 0.0L;
  for (int i = 0; i < x.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite fixture result.");
    const long double d = static_cast<long double>(a[i]) - b[i];
    delta += d * d;
    scale += static_cast<long double>(b[i]) * b[i];
  }
  return std::sqrt(static_cast<double>(delta / std::max(scale, 1.0e-300L)));
}

void CheckCase(int order, bool curved, bool explicit_inverse = false)
{
  Mesh serial = Mesh::MakeCartesian2D(curved ? 6 : 4, 3,
    Element::QUADRILATERAL, true, 1.0, 0.8);
  Array<int> permutation(serial.GetNE());
  // 5 is coprime to both 12 and 18; change global element order.
  for (int e = 0; e < permutation.Size(); ++e)
  { permutation[e] = (5 * e + 3) % permutation.Size(); }
  serial.ReorderElements(permutation, false);
  if (curved)
  {
    serial.SetCurvature(2, false, 2, Ordering::byVDIM);
    serial.Transform([](const Vector& x, Vector& y)
    {
      y.SetSize(2);
      y[0] = x[0] + 0.08 * x[0] * x[1];
      y[1] = x[1] + 0.04 * x[0] * (1.0 - x[0]);
    });
  }
  else
  {
    serial.Transform([](const Vector& x, Vector& y)
    {
      y.SetSize(2);
      y[0] = 1.1 * x[0] + 0.23 * x[1];
      y[1] = -0.12 * x[0] + 0.9 * x[1];
    });
  }
  ParMesh mesh(MPI_COMM_WORLD, serial);
  DG_FECollection collection(order, 2, BasisType::GaussLegendre);
  ParFiniteElementSpace space(&mesh, &collection);
  const int n = space.GetVSize(), d = space.GetFE(0)->GetDof();
  int phase = 0;
  FunctionCoefficient mass([&](const Vector& x)
  { return (1.0 + 0.3 * phase) / (1.0 + 0.2 * x[0] + 0.1 * x[1]); });
  FunctionCoefficient diffusion([&](const Vector& x)
  { return 0.015 * (1.0 + 0.2 * phase + 0.05 * x[0] + 0.03 * x[1]); });
  // Deliberately distinct domain rules: EA must follow each actual integrator.
  const auto& mass_rule = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
  const auto& diffusion_rule = IntRules.Get(Geometry::SQUARE, 2 * order + 3);
  const auto& face_rule = IntRules.Get(Geometry::SEGMENT, 2 * order + 1);
  FaceQuadratureSpace face_space(mesh, face_rule, FaceType::Interior);
  BilinearForm volume(&space);
  AddDomain(volume, mass, diffusion, mass_rule, diffusion_rule);
  volume.SetAssemblyLevel(AssemblyLevel::PARTIAL);
  auto backend = ::igrlab::igr_internal::MakeElementBlockJacobi(space, explicit_inverse);
  InteriorFaceDGDiffusionApplyScratch work;
  work.compact_qdata = true;
  work.compact_lift = true;
  InteriorFaceDGDiffusionApplyInfo info;
  Vector side_q;
  const double kappa = 2.0 * (order + 1) * (order + 1);
  Vector prior_blocks;
  for (int pass = 0; pass < 3; ++pass)
  {
    phase = pass == 0 ? 0 : 1;
    // Two SIPG coefficient generations, followed by a nonsymmetric IIPG
    // orientation check and the current dense-normal fallback.
    const double ip_sigma = pass == 2 ? 0.0 : -1.0;
    FillFaceCoefficients(mesh, face_space, diffusion, side_q);
    if (pass == 0)
    {
      info = BuildInteriorFaceDGDiffusionApplyData2D(
        space, face_space, side_q, kappa, work.shape, work.weighted_normal_dshape,
        work.penalty, work.face_elements, work.element_vdof_map,
        DGDiffusionFaceScaling::Arithmetic, &work.normal_dshape_base,
        &work.penalty_metric, &work.shape_dictionary, &work.normal_factorization);
    }
    LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(info, work, side_q, kappa,
                                                    DGDiffusionFaceScaling::Arithmetic);
    if (pass == 2) { work.normal_factorization.coefficient_revision = 0; }
    const bool compact_normal = work.normal_factorization.Matches(info) &&
      work.normal_factorization.coefficient_revision == info.connectivity_revision;
    const bool no_dense_device_normals = !work.weighted_normal_dshape.GetMemory().DeviceIsValid();
    if (order == 5 && pass != 2)
    {
      MFEM_VERIFY(compact_normal && work.shape_dictionary.Matches(info) && no_dense_device_normals,
                  "p5 fixture must exercise compact current normals/shapes.");
    }
    volume.Assemble();
    DenseTensor& blocks = backend->PrepareVolumeBlocks(volume);
    ::igrlab::igr_internal::AddExactFaceBlocks(info, work, ip_sigma, blocks);
    if (order == 5 && pass != 2)
    {
      MFEM_VERIFY(!work.weighted_normal_dshape.GetMemory().DeviceIsValid(),
                  "Compact block assembly materialized the full normal table.");
    }

    BilinearForm assembled(&space);
    AddDomain(assembled, mass, diffusion, mass_rule, diffusion_rule);
    auto* face = new DGDiffusionIntegrator(diffusion, ip_sigma, kappa);
    face->SetIntRule(&face_rule);
    assembled.AddInteriorFaceIntegrator(face);
    assembled.Assemble();
    assembled.Finalize();
    DenseMatrix full;
    assembled.SpMat().ToDenseMatrix(full);
    DenseTensor expected(d, d, space.GetNE());
    Array<int> dofs;
    double* expected_data = expected.HostWrite();
    for (int e = 0; e < space.GetNE(); ++e)
    {
      space.GetElementDofs(e, dofs);
      for (int j = 0; j < d; ++j)
        for (int i = 0; i < d; ++i)
          expected_data[i + d * (j + d * e)] = full(dofs[i], dofs[j]);
    }
    Vector actual_view(const_cast<double*>(blocks.HostRead()), blocks.TotalSize());
    Vector expected_view(expected.HostWrite(), expected.TotalSize());
    const double block_error = RelativeDifference(actual_view, expected_view);
    MFEM_VERIFY(block_error < 5.0e-11, "Current exact element blocks differ from assembled MFEM: " << block_error);
    double nonsymmetric_ratio = 0.0;
    if (pass == 2)
    {
      long double difference = 0.0L, scale = 0.0L;
      for (int e = 0; e < space.GetNE(); ++e)
        for (int j = 0; j < d; ++j)
          for (int i = 0; i < d; ++i)
          {
            const long double a = expected_data[i + d * (j + d * e)];
            const long double b = expected_data[j + d * (i + d * e)];
            difference += (a - b) * (a - b);
            scale += a * a;
          }
      nonsymmetric_ratio = std::sqrt(static_cast<double>(difference / scale));
      MFEM_VERIFY(nonsymmetric_ratio > 1.0e-8,
                  "IIPG fixture must distinguish an inverse from its transpose.");
    }
    if (pass == 0) { prior_blocks = actual_view; }
    if (pass == 1)
    {
      MFEM_VERIFY(RelativeDifference(actual_view, prior_blocks) > 0.05,
                  "Changed coefficients did not refresh the current blocks.");
    }
    // The explicit inverse releases blocks after factorization. End this
    // non-owning host view before that release.
    actual_view.Destroy();

    // Independent current-operator column oracle after EA has refreshed the
    // same domain integrators. No full-operator probing is used in production.
    Vector basis(n), action(n), expected_column(d), actual_column(d);
    basis.UseDevice(true); action.UseDevice(true);
    double column_error = 0.0;
    for (int e : {0, space.GetNE() - 1})
    {
      space.GetElementDofs(e, dofs);
      for (int j : {0, d / 2, d - 1})
      {
        basis = 0.0;
        basis.HostReadWrite()[dofs[j]] = 1.0;
        volume.Mult(basis, action);
        LaunchInteriorFaceDGDiffusionApply2D(info, work, basis, ip_sigma, action);
        const double* current = action.HostRead();
        for (int i = 0; i < d; ++i)
        {
          actual_column[i] = current[dofs[i]];
          expected_column[i] = expected_data[i + d * (j + d * e)];
        }
        column_error = std::max(column_error, RelativeDifference(actual_column, expected_column));
      }
    }
    MFEM_VERIFY(column_error < 5.0e-11, "Full current-A column oracle differs: " << column_error);

    backend->FactorBlocks();
    MFEM_VERIFY(!explicit_inverse || blocks.TotalSize() == 0,
                "Explicit inverse retained its build-only block matrices.");
    Vector rhs(n), actual(n), reference(n);
    for (int i = 0; i < n; ++i) { rhs[i] = std::sin(0.13 * (i + 1)) + 0.2 * std::cos(0.07 * i); }
    backend->Mult(rhs, actual);
    Vector local_rhs(d), local_solution(d);
    for (int e = 0; e < space.GetNE(); ++e)
    {
      space.GetElementDofs(e, dofs);
      DenseMatrix diagonal(d);
      std::copy(expected_data + std::size_t(e) * d * d,
                expected_data + std::size_t(e + 1) * d * d, diagonal.Data());
      DenseMatrixInverse inverse(diagonal);
      for (int i = 0; i < d; ++i) { local_rhs[i] = rhs[dofs[i]]; }
      inverse.Mult(local_rhs, local_solution);
      for (int i = 0; i < d; ++i) { reference[dofs[i]] = local_solution[i]; }
    }
    const double solve_error = RelativeDifference(actual, reference);
    MFEM_VERIFY(solve_error < 1.0e-10, "FP64 element application differs from independent block solves: " << solve_error);
    rhs *= -0.25;
    reference *= -0.25;
    backend->Mult(rhs, actual);
    MFEM_VERIFY(RelativeDifference(actual, reference) < 1.0e-10,
                "Reused current factors produced a stale solve.");
    std::cout << "element_block_jacobi_case p=" << order << " curved=" << int(curved)
              << " explicit_inverse=" << int(explicit_inverse)
              << " coefficient_phase=" << phase << " ip_sigma=" << ip_sigma
              << " compact_normal=" << int(compact_normal)
              << " block_relative=" << block_error << " current_A_columns_relative=" << column_error
              << " solve_relative=" << solve_error
              << " nonsymmetric_ratio=" << nonsymmetric_ratio << " passed\n";
  }
}

void CheckSelfNeighbor()
{
  InteriorFaceDGDiffusionApplyInfo info;
  info.dim = 2; info.vdim = 1; info.ne = 1; info.nf = 1; info.nq = 2; info.dof = 4;
  info.shape_size = info.weighted_normal_dshape_size = 16;
  info.penalty_size = 2; info.face_element_size = 2; info.element_vdof_map_size = 4;
  info.connectivity_revision = NextDGFaceConnectivityRevision();
  InteriorFaceDGDiffusionApplyScratch work;
  work.face_elements.SetSize(2); work.face_elements = 0;
  work.element_vdof_map.SetSize(4);
  for (int i = 0; i < 4; ++i) { work.element_vdof_map[i] = i; }
  work.shape.SetSize(16); work.weighted_normal_dshape.SetSize(16); work.penalty.SetSize(2);
  for (int i = 0; i < 16; ++i)
  {
    work.shape[i] = 0.03 * (i + 1);
    work.weighted_normal_dshape[i] = -0.1 + 0.017 * i;
  }
  work.penalty = 2.0;
  DenseTensor blocks(4, 4, 1);
  std::fill(blocks.HostWrite(), blocks.HostWrite() + 16, 0.0);
  constexpr double sigma = -1.0;
  ::igrlab::igr_internal::AddExactFaceBlocks(info, work, sigma, blocks);
  const double* data = blocks.HostRead();
  const double* shape = work.shape.HostRead();
  const double* normal = work.weighted_normal_dshape.HostRead();
  const double* penalty = work.penalty.HostRead();
  Vector basis(4), actual(4), measured(4), mathematical(4);
  basis.UseDevice(true); actual.UseDevice(true);
  double maximum_error = 0.0;
  long double cross_side_witness = 0.0L;
  for (int j = 0; j < 4; ++j)
  {
    // A previous CUDA Read promotes UseDevice. Writing basis[j] directly
    // after a device zero would leave the next column stale on the device.
    double* input = basis.HostWrite();
    std::fill(input, input + 4, 0.0);
    input[j] = 1.0;
    actual = 0.0;
    LaunchInteriorFaceDGDiffusionApply2D(info, work, basis, sigma, actual);
    double* column = measured.HostWrite();
    double* expected = mathematical.HostWrite();
    for (int i = 0; i < 4; ++i)
    {
      // Independent bilinear form for one global element on both sides:
      // J = S0-S1, F = N0+N1, A = -J^T F + sigma F^T J + J^T P J.
      // This sums neither the production CSR nor its four side-pair entries.
      long double entry = 0.0L, same_side_only = 0.0L;
      for (int q = 0; q < 2; ++q)
      {
        const int ri = 2 * (q + 2 * i), cj = 2 * (q + 2 * j);
        const long double si0 = shape[ri], si1 = shape[ri + 1];
        const long double sj0 = shape[cj], sj1 = shape[cj + 1];
        const long double ni0 = normal[ri], ni1 = normal[ri + 1];
        const long double nj0 = normal[cj], nj1 = normal[cj + 1];
        const long double jump_i = si0 - si1, jump_j = sj0 - sj1;
        entry += -jump_i * (nj0 + nj1) + sigma * (ni0 + ni1) * jump_j +
                 penalty[q] * jump_i * jump_j;
        same_side_only += -si0 * nj0 + sigma * ni0 * sj0 + penalty[q] * si0 * sj0 +
                           si1 * nj1 - sigma * ni1 * sj1 + penalty[q] * si1 * sj1;
      }
      cross_side_witness += std::abs(entry - same_side_only);
      column[i] = data[i + 4 * j];
      expected[i] = static_cast<double>(entry);
    }
    maximum_error = std::max(maximum_error, RelativeDifference(measured, mathematical));
    maximum_error = std::max(maximum_error, RelativeDifference(actual, mathematical));
    MFEM_VERIFY(maximum_error < 1.0e-12,
                "Periodic self-neighbor differs from the independent jump/normal-sum form: " << maximum_error);
  }
  MFEM_VERIFY(cross_side_witness > 1.0e-3L,
              "Self-neighbor fixture does not distinguish omitted cross-side terms.");
  std::cout << "element_block_jacobi_self_neighbor passed columns=4 independent_jump_normal_sum=true"
            << " explicit_device_column_writes=true relative_error=" << maximum_error << '\n';
}

class LifecycleEnvironment
{
public:
  LifecycleEnvironment()
  {
    const std::pair<const char*, const char*> settings[] = {
      {"IGR_TENSOR_MG", "off"}, {"IGR_TENSOR_MG_OPERATOR", "sampled"},
      {"IGR_DENSE_INVERSE", ""}, {"IGR_TWO_LEVEL_INVERSE", ""},
      {"IGR_PATCH_RADIUS", ""}, {"IGR_STATIONARY_SWEEPS", ""},
      {"IGR_PARTIAL_ASSEMBLY", "1"}, {"IGR_DEVICE_QF", "1"},
      {"IGR_DIRECT_IDENTITY_SYSTEM", "1"}, {"IGR_FACE_DENSITY_CUDA", "1"},
      {"IGR_TENSOR_VOLUME", "simt"}, {"IGR_TENSOR_VOLUME_COLLOCATED", "0"},
      {"IGR_TENSOR_VOLUME_STORAGE", "cached"}, {"IGR_KRYLOV_BACKEND", "cuda-fp64"},
      {"IGR_TRUE_RESIDUAL", "1"}};
    for (const char* prefix : {"IGRLAB_", "DG_IGR_"})
      for (const auto& setting : settings)
      {
        const std::string name = std::string(prefix) + setting.first;
        const char* previous = std::getenv(name.c_str());
        entries_.push_back({name, previous ? previous : "", previous != nullptr});
        setenv(name.c_str(), setting.second, 1);
      }
  }
  ~LifecycleEnvironment()
  {
    for (const auto& entry : entries_)
    {
      if (entry.had_value) { setenv(entry.name.c_str(), entry.value.c_str(), 1); }
      else { unsetenv(entry.name.c_str()); }
    }
  }
private:
  struct Entry { std::string name, value; bool had_value; };
  std::vector<Entry> entries_;
};

void CheckLifecycle(bool reference_blocks)
{
  LifecycleEnvironment environment;
  Mesh serial = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0, 1.0);
  ParMesh mesh(MPI_COMM_WORLD, serial);
  DG_FECollection collection(2, 2, BasisType::GaussLegendre);
  ParFiniteElementSpace scalar_space(&mesh, &collection);
  ParFiniteElementSpace vector_space(&mesh, &collection, 4, Ordering::byNODES);
  ParGridFunction state(&vector_space);
  int phase = 0;
  VectorFunctionCoefficient initial(4, [&](const Vector& x, Vector& value)
  {
    const double rho = 1.0 + 0.1 * x[0] + 0.03 * phase * x[1];
    const double u = 0.2 + 0.04 * x[0] + 0.07 * phase * x[1];
    const double v = -0.1 + 0.05 * x[1] + 0.03 * phase * x[0];
    const double pressure = 1.0 + 0.1 * x[0];
    value.SetSize(4);
    value[0] = rho; value[1] = rho * u; value[2] = rho * v;
    value[3] = pressure / 0.4 + 0.5 * rho * (u * u + v * v);
  });
  state.ProjectCoefficient(initial);
  ::igrlab::IGRConfig config;
  config.alpha = 0.02;
  config.backend = ::igrlab::IGRGradBackend::BrokenDG;
  config.ip_kind = ::igrlab::IGRIPKind::SIPG;
  config.solve_mode = ::igrlab::IGRSolveMode::FixedElementJacobi;
  config.regular_sweeps = 2; config.warm_start_sweeps = 50;
  config.element_jacobi_omega = 0.7;
  config.element_jacobi_reference_blocks = reference_blocks;
  config.exact_dg_face_operator = true;
  config.sigma_cadence = ::igrlab::IGRSigmaUpdateCadence::EveryStage;
  config.lin_solver_type = ::igrlab::IGRLinearSolverType::FGMRES;
  config.prec_type = ::igrlab::IGRPrecondType::None;
  // A hidden one-iteration Krylov fallback would not satisfy this fixture.
  config.lin_rtol = 1.0e-14; config.lin_atol = 0.0; config.lin_maxit = 1;
  config.nonconvergence_policy = ::igrlab::IGRNonconvergencePolicy::Error;
  const auto make_solver = [&]()
  {
    auto solver = std::make_unique<::igrlab::IGRSolver>(scalar_space, vector_space, 2, 4, config.alpha);
    solver->ApplyConfig(config);
    return solver;
  };
  const auto check_solve = [reference_blocks](const ::igrlab::IGRSolver& solver, int sweeps, bool warm)
  {
    MFEM_VERIFY(solver.GetLastLinearIters() == sweeps && !solver.GetLastLinearConverged() &&
                  solver.GetLastWarmStartEffective() == warm &&
                  solver.GetLastLinearSolverName() == "fixed-element-jacobi" &&
                  solver.GetLastPrecondName() == (reference_blocks
                    ? "reference-element-inverse-fp64" : "current-element-lu-fp64") &&
                  std::isfinite(solver.GetLastLinearTrueResidualRelRHS()),
                "Fixed-sweep lifecycle was relabeled, cold-started or changed its update budget.");
  };

  auto direct = make_solver(), interposed = make_solver();
  for (auto* solver : {direct.get(), interposed.get()})
  {
    solver->UpdateFromState(state, true);
    check_solve(*solver, 50, false);
  }
  MFEM_VERIFY(std::memcmp(direct->GetSigma().HostRead(), interposed->GetSigma().HostRead(),
                          std::size_t(scalar_space.GetVSize()) * sizeof(double)) == 0,
              "Identical fixed initialization differs.");
  phase = 1;
  state.ProjectCoefficient(initial);
  interposed->UpdateFromState(state, true, ::igrlab::IGRSolvePurpose::OutputOnly);
  check_solve(*interposed, 2, true);
  direct->UpdateFromState(state, true);
  interposed->UpdateFromState(state, true);
  check_solve(*direct, 2, true); check_solve(*interposed, 2, true);
  MFEM_VERIFY(direct->GetProductionLinearIterations() == 52 &&
                interposed->GetProductionLinearIterations() == 52 &&
                interposed->GetOutputOnlyLinearIterations() == 2 &&
                std::memcmp(direct->GetSigma().HostRead(), interposed->GetSigma().HostRead(),
                            std::size_t(scalar_space.GetVSize()) * sizeof(double)) == 0,
              "Output-only work changed subsequent production sigma or sweep accounting.");

  const auto same_sigma = [&](const ::igrlab::IGRSolver& a, const ::igrlab::IGRSolver& b)
  {
    return std::memcmp(a.GetSigma().HostRead(), b.GetSigma().HostRead(),
                       std::size_t(scalar_space.GetVSize()) * sizeof(double)) == 0;
  };
  // Before production history exists, a different-state output/prefetch must
  // seed neither sigma history nor the retained reference block inverse.
  for (bool prefetch : {false, true})
  {
    auto fresh = make_solver(), temporary = make_solver();
    phase = 0;
    state.ProjectCoefficient(initial);
    temporary->UpdateFromState(state, true, prefetch
      ? ::igrlab::IGRSolvePurpose::ProductionStage0Prefetch
      : ::igrlab::IGRSolvePurpose::OutputOnly);
    check_solve(*temporary, 50, false);
    if (prefetch)
    {
      MFEM_VERIFY(temporary->HasPendingStage0PrefetchAccounting() &&
                    temporary->ReclassifyPendingStage0PrefetchAsOutputOnly(true),
                  "Fixture did not discard its pending initialization prefetch.");
    }
    phase = 3;
    state.ProjectCoefficient(initial);
    fresh->UpdateFromState(state, true);
    temporary->UpdateFromState(state, true);
    check_solve(*fresh, 50, false); check_solve(*temporary, 50, false);
    MFEM_VERIFY(same_sigma(*fresh, *temporary),
                "Different-state initial output/prefetch seeded production history or reference blocks.");
    // The next two-sweep update also witnesses which reference matrix was
    // retained, even if fifty initialization sweeps nearly solved the system.
    phase = 4;
    state.ProjectCoefficient(initial);
    fresh->UpdateFromState(state, true);
    temporary->UpdateFromState(state, true);
    check_solve(*fresh, 2, true); check_solve(*temporary, 2, true);
    MFEM_VERIFY(same_sigma(*fresh, *temporary) &&
                  fresh->GetProductionLinearIterations() == 52 &&
                  temporary->GetProductionLinearIterations() == 52 &&
                  temporary->GetOutputOnlyLinearIterations() == 50,
                "Uncommitted initialization changed later production sigma/reference blocks.");
  }

  // Real ordinary-DG wrapper: initialization is prefetched once, then consumed
  // by stage zero without another solve; the next stage performs two sweeps.
  auto wrapped_solver = make_solver();
  EulerFlux euler(2, 1.4);
  FluxWithIGR physical(euler);
  physical.SetSigmaField(&wrapped_solver->GetSigma());
  RusanovFluxWithSides flux(physical);
  auto integrator = std::make_unique<HyperbolicFormIntegrator>(flux, 2);
  DGHyperbolicConservationLaws base(vector_space, std::move(integrator), true);
  base.SetNumericalFlux(&flux);
  Array<int> boundary(mesh.bdr_attributes.Max()); boundary = 1;
  base.AddBdrFaceIntegrator(std::make_unique<BdrHyperbolicFluxIntegrator>(
    flux, std::make_unique<SlipWallState>(2), 2), boundary);
  EulerWithIGR wrapped(base, *wrapped_solver, state);
  const auto preparation = wrapped.PrepareSigmaForOutput(state, 0, 0, 0.0, true, false, false);
  check_solve(*wrapped_solver, 50, false);
  MFEM_VERIFY(preparation.solved && preparation.reusable_stage0 &&
                wrapped_solver->HasPendingStage0PrefetchAccounting() &&
                !wrapped.SupportsIndependentResidualCertification(),
              "Approximate wrapper initialization did not create an uncertified prefetch.");
  Vector rhs(state.Size());
  wrapped.BeginRKStep(0, 0.0, 1.0e-4, 3);
  wrapped.SetRKStageState(0, 0);
  wrapped.Mult(state, rhs);
  MFEM_VERIFY(wrapped_solver->GetLinearSolveCalls() == 1 &&
                !wrapped_solver->HasPendingStage0PrefetchAccounting() &&
                wrapped.GetSigmaStage0ReuseCount() == 1,
              "Stage zero failed to consume the initialization prefetch exactly once.");
  wrapped.SetRKStageState(1, 1);
  wrapped.Mult(state, rhs);
  check_solve(*wrapped_solver, 2, true);
  MFEM_VERIFY(wrapped_solver->GetLinearSolveCalls() == 2 &&
                wrapped_solver->GetProductionLinearIterations() == 52 &&
                wrapped_solver->GetProductionLinearConvergedCalls() == 0,
              "Real wrapper changed fixed sweep counts or claimed convergence.");
  std::cout << "element_block_jacobi_lifecycle passed reference_blocks=" << int(reference_blocks)
            << " initial_sweeps=50 regular_sweeps=2 "
               "omega=0.7 output_history=bitwise prefetch=consume-only "
               "discarded_prefetch=uncommitted different_state_initial_output=uncommitted "
               "reference_cache_history=bitwise convergence_claim=false fallback=none\n";
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1 &&
                ::igrlab::dg_face::UsesOrdinaryCUDAForDGDiffusionFaces2D(),
              "Element-block fixture requires one native CUDA rank.");
  CheckCase(2, false);
  CheckCase(5, true);
  CheckCase(2, false, true);
  CheckCase(5, true, true);
  CheckSelfNeighbor();
  CheckLifecycle(false);
  CheckLifecycle(true);
  std::cout << "element_block_jacobi_test passed current_coefficients=true "
               "assembled_blocks=true full_A_columns=true fp64_lu=true fp64_explicit_inverse=true "
               "coarse=false skew=true curved=true dense_fallback=true self_neighbor=true\n";
  return 0;
}
