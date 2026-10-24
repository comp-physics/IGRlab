#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_element_block_jacobi.hpp"
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/dgsem/dgsem_operator.hpp"
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
#include <sstream>
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
      {"IGR_LOCAL_TENSOR_PRECONDITIONER", "0"},
      {"IGR_LOCAL_TENSOR_REFERENCE", "exact"},
      {"IGR_LOCAL_TENSOR_EXECUTION", "simt-fp64"},
      {"IGR_ELEMENT_JACOBI_FUSED_RESIDUAL", "0"},
      {"IGR_ELEMENT_JACOBI_FUSED_UPDATE", "0"},
      {"IGR_TRUE_RESIDUAL", "1"}};
    for (const char* prefix : {"IGRLAB_", "DG_IGR_"})
      for (const auto& setting : settings)
      {
        const std::string name = std::string(prefix) + setting.first;
        Set(name.c_str(), setting.second);
      }
  }
  void Set(const char* name, const char* value)
  {
    const auto found = std::find_if(entries_.begin(), entries_.end(),
      [name](const Entry& entry) { return entry.name == name; });
    if (found == entries_.end())
    {
      const char* previous = std::getenv(name);
      entries_.push_back({name, previous ? previous : "", previous != nullptr});
    }
    setenv(name, value, 1);
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

void CheckConvergedLocalTensor()
{
  LifecycleEnvironment environment;
  environment.Set("IGRLAB_IGR_LOCAL_TENSOR_REFERENCE", "projected");
  environment.Set("IGRLAB_IGR_KRYLOV_RESIDUAL_SCALE", "rhs");
  Mesh serial = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0, 1.0);
  serial.SetCurvature(2, false, 2, Ordering::byVDIM);
  serial.Transform([](const Vector& x, Vector& y)
  {
    y.SetSize(2);
    y[0] = x[0] + 0.08 * x[0] * x[1];
    y[1] = x[1] + 0.04 * x[0] * (1.0 - x[0]);
  });
  ParMesh mesh(MPI_COMM_WORLD, serial);
  DG_FECollection collection(5, 2, BasisType::GaussLegendre);
  ParFiniteElementSpace scalar_space(&mesh, &collection);
  ParFiniteElementSpace vector_space(&mesh, &collection, 4, Ordering::byNODES);
  ParGridFunction state(&vector_space);
  int phase = 0;
  VectorFunctionCoefficient initial(4, [&](const Vector& x, Vector& value)
  {
    const double rho = 0.8 + 0.2 * x[0] + 0.1 * x[1] +
      0.15 * phase * (1.0 + x[0] * x[1]);
    const double u = 0.2 + 0.11 * x[0] + 0.06 * phase * x[1];
    const double v = -0.1 + 0.08 * x[1] + 0.04 * phase * x[0];
    value.SetSize(4);
    value[0] = rho; value[1] = rho * u; value[2] = rho * v;
    value[3] = 2.5 + 0.5 * rho * (u * u + v * v);
  });
  ::igrlab::IGRConfig config;
  config.alpha = 0.005;
  config.backend = ::igrlab::IGRGradBackend::BrokenDG;
  config.ip_kind = ::igrlab::IGRIPKind::SIPG;
  config.solve_mode = ::igrlab::IGRSolveMode::Krylov;
  config.exact_dg_face_operator = true;
  config.lin_solver_type = ::igrlab::IGRLinearSolverType::FGMRES;
  config.prec_type = ::igrlab::IGRPrecondType::None;
  config.lin_rtol = 1.0e-6; config.lin_atol = 0.0; config.lin_maxit = 1000;
  config.nonconvergence_policy = ::igrlab::IGRNonconvergencePolicy::Error;
  const auto make_solver = [&](bool control)
  {
    auto solver = std::make_unique<::igrlab::IGRSolver>(
      scalar_space, vector_space, 2, 4, config.alpha);
    auto selected = config;
    if (control) { selected.lin_rtol = 1.0e-11; }
    solver->ApplyConfig(selected);
    return solver;
  };
  const auto solve_local = [&](::igrlab::IGRSolver& solver,
                               ::igrlab::IGRSolvePurpose purpose)
  {
    environment.Set("IGRLAB_IGR_LOCAL_TENSOR_PRECONDITIONER", "1");
    solver.UpdateFromState(state, false, purpose);
    MFEM_VERIFY(solver.GetLastLinearConverged() &&
                  solver.GetLastLinearSolverName() == "fgmres-cuda-fp64" &&
                  solver.GetLastPrecondName() == "tensor-element-projected-simt-fp64" &&
                  solver.GetLastLinearTrueResidualRelRHS() <= config.lin_rtol &&
                  solver.GetLastIndependentCertificateRelativeLimit() == config.lin_rtol,
                "Local tensor FGMRES failed its original-current FP64 RHS residual certificate.");
  };
  auto local = make_solver(false), control = make_solver(true);
  double maximum_difference = 0.0, maximum_residual = 0.0;
  for (phase = 0; phase < 2; ++phase)
  {
    // Exercise the local-preconditioner certificate both without a volume
    // override and with the existing current-coefficient SIMT optimization.
    environment.Set("IGRLAB_IGR_TENSOR_VOLUME", phase == 0 ? "off" : "simt");
    state.ProjectCoefficient(initial);
    solve_local(*local, ::igrlab::IGRSolvePurpose::ProductionStage);
    maximum_residual = std::max(maximum_residual, local->GetLastLinearTrueResidualRelRHS());
    environment.Set("IGRLAB_IGR_LOCAL_TENSOR_PRECONDITIONER", "0");
    control->UpdateFromState(state, false);
    const double difference = RelativeDifference(local->GetSigma(), control->GetSigma());
    maximum_difference = std::max(maximum_difference, difference);
    MFEM_VERIFY(control->GetLastLinearConverged() && difference <= 1.0e-4,
                "Projected local preconditioning changed the converged current-operator solution: "
                << difference);
  }
  // An output-only or abandoned initialization must not select the reference
  // density for the first production preconditioner. Cold starts make equal
  // production caches observable through identical solve work and sigma.
  for (bool prefetch : {false, true})
  {
    auto fresh = make_solver(false), temporary = make_solver(false);
    phase = 0; state.ProjectCoefficient(initial);
    solve_local(*temporary, prefetch
      ? ::igrlab::IGRSolvePurpose::ProductionStage0Prefetch
      : ::igrlab::IGRSolvePurpose::OutputOnly);
    if (prefetch)
    {
      MFEM_VERIFY(temporary->ReclassifyPendingStage0PrefetchAsOutputOnly(true),
                  "Local tensor fixture did not discard its initialization prefetch.");
    }
    phase = 1; state.ProjectCoefficient(initial);
    solve_local(*fresh, ::igrlab::IGRSolvePurpose::ProductionStage);
    solve_local(*temporary, ::igrlab::IGRSolvePurpose::ProductionStage);
    MFEM_VERIFY(fresh->GetLastLinearIters() == temporary->GetLastLinearIters() &&
                  std::memcmp(fresh->GetSigma().HostRead(), temporary->GetSigma().HostRead(),
                              std::size_t(scalar_space.GetVSize()) * sizeof(double)) == 0,
                "Uncommitted local tensor initialization changed the production reference.");
  }
  std::cout << "element_block_local_tensor_fgmres passed curved_q2=true order=5"
            << " reference=projected execution=simt-fp64 coefficient_refresh=true"
            << " original_current_fp64_certificate=true relative_limit=" << config.lin_rtol
            << " maximum_true_residual_rel_rhs=" << maximum_residual
            << " maximum_control_sigma_relative_difference=" << maximum_difference
            << " uncommitted_initial_reference=discarded\n";
}

void CheckLifecycle(bool reference_blocks, bool tensor_blocks = false)
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
    // Tensor setup needs a separable initial operator. Elementwise constant
    // phase 3 changes normalized face terms, so a leaked phase-0 cache cannot
    // pass the initial output/discarded-prefetch comparison by mass scaling.
    const double rho = tensor_blocks && phase == 0 ? 1.0 :
      (tensor_blocks && phase == 3 ? (x[0] < 0.5 ? 0.75 : 1.25) :
        (tensor_blocks && phase == 5 ? 0.45 :
          1.0 + 0.1 * x[0] + 0.03 * phase * x[1]));
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
  config.element_jacobi_tensor_blocks = tensor_blocks;
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
  const auto check_solve = [reference_blocks, tensor_blocks](
    const ::igrlab::IGRSolver& solver, int sweeps, bool warm)
  {
    MFEM_VERIFY(solver.GetLastLinearIters() == sweeps && !solver.GetLastLinearConverged() &&
                  solver.GetLastWarmStartEffective() == warm &&
                  solver.GetLastLinearSolverName() == "fixed-element-jacobi" &&
                  solver.GetLastPrecondName() == (tensor_blocks
                    ? "tensor-element-density-fp64" : (reference_blocks
                        ? "reference-element-inverse-fp64" : "current-element-lu-fp64")) &&
                  std::isfinite(solver.GetLastLinearTrueResidualRelRHS()),
                "Fixed-sweep lifecycle was relabeled, cold-started or changed its update budget.");
  };

  if (tensor_blocks)
  {
    auto scaled = make_solver();
    auto current_config = config;
    current_config.element_jacobi_tensor_blocks = false;
    auto current = std::make_unique<::igrlab::IGRSolver>(
      scalar_space, vector_space, 2, 4, current_config.alpha);
    current->ApplyConfig(current_config);
    for (auto* solver : {scaled.get(), current.get()})
    {
      solver->UpdateFromState(state, true);
    }
    phase = 5;
    state.ProjectCoefficient(initial);
    for (auto* solver : {scaled.get(), current.get()})
    {
      solver->UpdateFromState(state, true);
    }
    check_solve(*scaled, 2, true);
    Vector difference(scaled->GetSigma());
    difference -= current->GetSigma();
    MFEM_VERIFY(difference.Normlinf() <=
                  5.0e-11 * std::max(1.0, current->GetSigma().Normlinf()),
                "Tensor integration did not refresh mass scaling for a uniform density change.");
    phase = 0;
    state.ProjectCoefficient(initial);
  }

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
  // seed neither sigma history nor the retained inverse/tensor structure.
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
  if (tensor_blocks)
  {
    phase = 0;
    state.ProjectCoefficient(initial);
  }
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
            << " tensor_blocks=" << int(tensor_blocks)
            << " current_mass_refresh=" << int(tensor_blocks)
            << " initial_sweeps=50 regular_sweeps=2 "
               "omega=0.7 output_history=bitwise prefetch=consume-only "
               "discarded_prefetch=uncommitted different_state_initial_output=uncommitted "
               "reference_cache_history=bitwise convergence_claim=false fallback=none\n";
}

void CheckFusedSweepLifecycle(bool compact_lift, int regular_sweeps,
                              bool fuse_residual, bool fuse_update,
                              const char* execution = "wmma-factor-fp64",
                              int basis = BasisType::GaussLegendre)
{
  LifecycleEnvironment environment;
  environment.Set("IGRLAB_IGR_LOCAL_TENSOR_REFERENCE", "projected");
  environment.Set("IGRLAB_IGR_LOCAL_TENSOR_EXECUTION", execution);
  environment.Set("IGRLAB_IGR_COMPACT_FACE_QDATA", "1");
  environment.Set("IGRLAB_IGR_COMPACT_FACE_LIFT", compact_lift ? "1" : "0");
  Mesh serial = Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true, 1.0, 0.8);
  serial.SetCurvature(2, false, 2, Ordering::byVDIM);
  serial.Transform([](const Vector& x, Vector& y)
  {
    y.SetSize(2);
    y[0] = x[0] + 0.08 * x[0] * x[1];
    y[1] = x[1] + 0.04 * x[0] * (1.0 - x[0]);
  });
  ParMesh mesh(MPI_COMM_WORLD, serial);
  DG_FECollection collection(5, 2, basis);
  ParFiniteElementSpace scalar_space(&mesh, &collection);
  ParFiniteElementSpace vector_space(&mesh, &collection, 4, Ordering::byNODES);
  ParGridFunction state(&vector_space);
  int phase = 0;
  VectorFunctionCoefficient initial(4, [&](const Vector& x, Vector& value)
  {
    const double rho = 0.8 + 0.2 * x[0] + 0.1 * x[1] + 0.15 * phase * (1.0 + x[0] * x[1]);
    const double u = 0.2 + 0.11 * x[0] + 0.06 * phase * x[1];
    const double v = -0.1 + 0.08 * x[1] + 0.04 * phase * x[0];
    value.SetSize(4);
    value[0] = rho; value[1] = rho * u; value[2] = rho * v;
    value[3] = 2.5 + 0.5 * rho * (u * u + v * v);
  });
  ::igrlab::IGRConfig config;
  config.alpha = 0.002;
  config.backend = ::igrlab::IGRGradBackend::BrokenDG;
  config.ip_kind = ::igrlab::IGRIPKind::SIPG;
  config.solve_mode = ::igrlab::IGRSolveMode::FixedElementJacobi;
  config.regular_sweeps = regular_sweeps; config.warm_start_sweeps = 50;
  config.element_jacobi_omega = 0.7;
  config.element_jacobi_tensor_blocks = true;
  config.exact_dg_face_operator = true;
  config.sigma_cadence = ::igrlab::IGRSigmaUpdateCadence::EveryStage;
  config.lin_print = 1;
  config.nan_scan = false;
  const auto make_solver = [&]()
  {
    auto result = std::make_unique<::igrlab::IGRSolver>(scalar_space, vector_space, 2, 4, config.alpha);
    result->ApplyConfig(config);
    return result;
  };
  auto original = make_solver(), fused = make_solver();
  const auto update = [&](::igrlab::IGRSolver& solver, bool enabled,
                          ::igrlab::IGRSolvePurpose purpose, bool initialized)
  {
    environment.Set("IGRLAB_IGR_ELEMENT_JACOBI_FUSED_RESIDUAL", enabled && fuse_residual ? "1" : "0");
    environment.Set("IGRLAB_IGR_ELEMENT_JACOBI_FUSED_UPDATE", enabled && fuse_update ? "1" : "0");
    std::ostringstream captured;
    {
      struct RestoreOutput
      {
        std::streambuf* previous;
        ~RestoreOutput() { std::cout.rdbuf(previous); }
      } restore{std::cout.rdbuf(captured.rdbuf())};
      solver.UpdateFromState(state, true, purpose);
    }
    std::cout << captured.str();
    const int sweeps = initialized ? regular_sweeps : 50;
    const int fast_sweeps = enabled && fuse_residual && compact_lift && initialized ? sweeps : 0;
    const int update_sweeps = enabled && fuse_update && initialized &&
      std::strcmp(execution, "wmma-factor-fp64") == 0 ? sweeps : 0;
    const auto expected = " fused_residual_sweeps=" + std::to_string(fast_sweeps) + " ";
    const auto expected_update = " fused_update_sweeps=" + std::to_string(update_sweeps) + " ";
    MFEM_VERIFY(captured.str().find(expected) != std::string::npos &&
                  captured.str().find(expected_update) != std::string::npos &&
                  solver.GetLastLinearIters() == sweeps &&
                  solver.GetLastKrylovOperatorApplyCount() == sweeps + 1 &&
                  solver.GetLastKrylovPreconditionerApplyCount() == sweeps &&
                  solver.GetLastWarmStartEffective() == initialized &&
                  !solver.GetLastLinearConverged() && std::isfinite(solver.GetLastLinearTrueResidualRelRHS()),
                "Fused fixed-sweep path changed initialization, fallback, certificate or application accounting.");
    if (!initialized)
    { MFEM_VERIFY(captured.str().find("path=fused-") == std::string::npos,
                  "Initialization must retain the original operator/inverse sequence."); }
  };
  const auto same = [&]()
  {
    MFEM_VERIFY(std::memcmp(original->GetSigma().HostRead(), fused->GetSigma().HostRead(),
                            std::size_t(scalar_space.GetVSize()) * sizeof(double)) == 0 &&
                  original->GetLastLinearTrueResidualRelRHS() == fused->GetLastLinearTrueResidualRelRHS(),
                "Fused residual changed production sigma or the original current-FP64 residual.");
  };
  const auto production = ::igrlab::IGRSolvePurpose::ProductionStage;
  state.ProjectCoefficient(initial);
  update(*original, false, production, false);
  update(*fused, true, production, false);
  same();
  phase = 1; state.ProjectCoefficient(initial);
  update(*fused, true, ::igrlab::IGRSolvePurpose::OutputOnly, true);
  update(*original, false, production, true);
  update(*fused, true, production, true);
  same();
  phase = 2; state.ProjectCoefficient(initial);
  update(*original, false, production, true);
  update(*fused, true, ::igrlab::IGRSolvePurpose::ProductionStage0Prefetch, true);
  fused->ConsumePendingStage0PrefetchAccounting();
  same();
  phase = 3; state.ProjectCoefficient(initial);
  update(*original, false, production, true);
  update(*fused, true, production, true);
  same();
  MFEM_VERIFY(original->GetProductionLinearIterations() == 50 + 3 * regular_sweeps &&
                fused->GetProductionLinearIterations() == 50 + 3 * regular_sweeps &&
                fused->GetOutputOnlyLinearIterations() == regular_sweeps,
              "Fused residual changed output-only or consumed-prefetch history.");
  std::cout << "element_block_jacobi_fused_sweep passed p=5 geometry=q2 reference=projected"
            << " basis=" << (basis==BasisType::GaussLobatto ? "gll" : "gl")
            << " execution=" << execution << " volume=simt compact_lift=" << compact_lift
            << " fuse_residual=" << fuse_residual << " fuse_update=" << fuse_update
            << " initial_sweeps=50 regular_sweeps=" << regular_sweeps
            << " production_sigma=bitwise output_history=bitwise original_fp64_diagnostic=true\n";
}
void CheckDGSEMMappedFixedLifecycle()
{
  LifecycleEnvironment environment;
  const std::pair<const char*, const char*> settings[] = {
    {"IGRLAB_IGR_LOCAL_TENSOR_REFERENCE", "projected"},
    {"IGRLAB_IGR_LOCAL_TENSOR_EXECUTION", "wmma-factor-fp64"},
    {"IGRLAB_IGR_COEFFICIENT_EXECUTION", "fused"},
    {"IGRLAB_IGR_FACE_DENSITY_EXECUTION", "tensor"},
    {"IGRLAB_IGR_STAGE_STATE_DIRECT", "1"},
    {"IGRLAB_IGR_TENSOR_VOLUME_COMPACT", "1"},
    {"IGRLAB_IGR_COMPACT_FACE_QDATA", "1"},
    {"IGRLAB_IGR_COMPACT_FACE_LIFT", "1"},
    {"IGRLAB_IGR_ELEMENT_JACOBI_FUSED_RESIDUAL", "1"},
    {"IGRLAB_IGR_ELEMENT_JACOBI_FUSED_UPDATE", "1"},
    {"IGRLAB_DGSEM_PRODUCTION_STRICT", "1"},
    {"IGRLAB_DGSEM_FUSED_RESIDUAL", "1"},
    {"IGRLAB_DGSEM_FUSED_DMMA", "0"},
    {"IGRLAB_DGSEM_SUBCELL_BLEND", "0"},
    {"IGRLAB_DGSEM_VOLUME_QUADRATURE_MIN_ORDER", "0"}};
  for (const auto& setting : settings) { environment.Set(setting.first, setting.second); }
  Mesh serial = Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL, true, 1.0, 0.8);
  serial.SetCurvature(2, false, 2, Ordering::byVDIM);
  serial.Transform([](const Vector& x, Vector& y)
  {
    y.SetSize(2);
    y[0] = x[0] + .08*x[0]*x[1];
    y[1] = x[1] + .04*x[0]*(1.0-x[0]);
  });
  ParMesh mesh(MPI_COMM_WORLD, serial);
  DG_FECollection collection(5, 2, BasisType::GaussLobatto);
  ParFiniteElementSpace scalar_space(&mesh, &collection);
  ParFiniteElementSpace vector_space(&mesh, &collection, 4, Ordering::byNODES);
  ParGridFunction state(&vector_space);
  int phase = 0;
  VectorFunctionCoefficient initial(4, [&](const Vector& x, Vector& value)
  {
    const double rho = .9+.15*x[0]+.08*x[1]+.02*phase*x[0]*x[1];
    const double u = .2+.06*x[0]+.03*phase*x[1];
    const double v = -.1+.08*x[1]-.02*phase*x[0];
    value.SetSize(4);
    value[0]=rho; value[1]=rho*u; value[2]=rho*v;
    value[3]=2.5+.5*rho*(u*u+v*v);
  });
  state.ProjectCoefficient(initial);
  ::igrlab::IGRConfig config;
  config.alpha=.002;
  config.backend=::igrlab::IGRGradBackend::BrokenDG;
  config.ip_kind=::igrlab::IGRIPKind::SIPG;
  config.solve_mode=::igrlab::IGRSolveMode::FixedElementJacobi;
  config.element_jacobi_tensor_blocks=true;
  config.regular_sweeps=2; config.warm_start_sweeps=50; config.element_jacobi_omega=.7;
  config.exact_dg_face_operator=true;
  config.sigma_cadence=::igrlab::IGRSigmaUpdateCadence::EveryStage;
  config.nan_scan=false;
  const auto make_solver = [&]()
  {
    auto result=std::make_unique<::igrlab::IGRSolver>(scalar_space, vector_space, 2, 4, config.alpha);
    result->ApplyConfig(config);
    return result;
  };
  auto direct=make_solver(), interposed=make_solver();
  EulerFlux euler(2,1.4);
  FluxWithIGR physical(euler);
  physical.SetSigmaField(&interposed->GetSigma());
  RusanovFluxWithSides flux(physical);
  auto integrator=std::make_unique<HyperbolicFormIntegrator>(flux,2);
  DGHyperbolicConservationLaws base(vector_space,std::move(integrator),true);
  base.SetNumericalFlux(&flux);
  base.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
  base.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
  base.SetDGSEMVolumeDissipation(mfem::DGSEMVolumeDissipation::None);
  base.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
  base.SetDGSEMGamma(1.4);
  base.SetDGSEMDiagnosticStateScan(false);
  Array<int> boundary(mesh.bdr_attributes.Max()); boundary=1;
  base.AddBdrFaceIntegrator(std::make_unique<BdrHyperbolicFluxIntegrator>(
    flux,std::make_unique<SlipWallState>(2),2),boundary);
  EulerWithIGR wrapped(base,*interposed,state);
  direct->UpdateFromState(state,true);
  const auto preparation=wrapped.PrepareSigmaForOutput(state,0,0,0.0,true,false,false);
  MFEM_VERIFY(preparation.solved && preparation.reusable_stage0 &&
                interposed->GetLastLinearIters()==50 &&
                !wrapped.SupportsIndependentResidualCertification(),
              "Mapped GLL fixed initialization must use exactly 50 uncertified corrections.");
  wrapped.BeginRKStep(0,0.0,1.0e-4,3);
  Vector rhs(state.Size()); rhs.UseDevice(true);
  double max_rhs_error=0.0, max_speed_error=0.0;
  for (int stage=0; stage<3; ++stage)
  {
    if (stage)
    {
      phase=stage; state.ProjectCoefficient(initial);
      // Output must not change the next production approximation's history.
      interposed->UpdateFromState(state,true,::igrlab::IGRSolvePurpose::OutputOnly);
      direct->UpdateFromState(state,true);
    }
    wrapped.SetRKStageState(stage,stage);
    wrapped.Mult(state,rhs);
    MFEM_VERIFY(interposed->GetLastPrecondName()=="tensor-element-projected-wmma-factor-fp64" &&
                  !interposed->GetLastLinearConverged() &&
                  interposed->GetLastLinearIters()==(stage ? 2 : 50) &&
                  std::memcmp(direct->GetSigma().HostRead(),interposed->GetSigma().HostRead(),
                    std::size_t(scalar_space.GetVSize())*sizeof(double))==0,
                "Mapped GLL stage/output changed the fixed history, backend or sweep budget.");
    const auto selection=base.GetDGSEMProductionSelection();
    const auto bridge=base.GetDGSEMProductionBridgeEvidence();
    MFEM_VERIFY(selection.supported && selection.strict && selection.uses_sigma &&
                  selection.order==5 && bridge.fallback_mult_calls==0 &&
                  bridge.production_mult_calls==stage+1 && !base.GetDGSEMDiagnosticStateScan(),
                "Mapped GLL wrapper must execute the packed DGSEM sigma path without fallback/scans.");
    const double actual_speed=base.GetMaxCharSpeed();
    // Independent host assembly uses the same physical GLL polynomial and
    // current sigma, with separate volume/face evaluation and host application
    // of the base operator's inverse mass (not independent mass assembly).
    mfem::dgsem::DGSEMOperator host(base);
    Vector weak(state.Size()), expected(state.Size()); weak=0.0;
    const double surface_speed=host.AssembleSurface(state,weak);
    const double volume_speed=host.AssembleVolume(state,weak);
    host.ApplyInverseMass(weak,expected);
    Vector difference(rhs); difference-=expected;
    const double error=difference.Normlinf()/std::max(1.0,expected.Normlinf());
    const double speed=std::max(surface_speed,volume_speed);
    const double speed_error=std::abs(actual_speed-speed)/std::max(1.0,std::abs(speed));
    MFEM_VERIFY(std::isfinite(error) && error<2e-10 && std::isfinite(speed_error) && speed_error<2e-12,
                "Mapped GLL production Euler residual/CFL differs from host assembly.");
    max_rhs_error=std::max(max_rhs_error,error); max_speed_error=std::max(max_speed_error,speed_error);
  }
  MFEM_VERIFY(interposed->GetProductionLinearIterations()==54 &&
                interposed->GetOutputOnlyLinearIterations()==4 &&
                wrapped.GetSigmaStage0ReuseCount()==1 &&
                interposed->GetProductionLinearConvergedCalls()==0,
              "Mapped GLL wrapper changed initialization, output or production work accounting.");
  std::cout << "dgsem_mapped_fixed_lifecycle passed basis=gll geometry=q2 p=5 "
               "initial_sweeps=50 regular_sweeps=2 omega=0.7 history=bitwise "
               "prefetch=consume-only fallback=none scans=off convergence_claim=false "
            << "rhs_relative_max=" << max_rhs_error << " cfl_relative_error=" << max_speed_error << '\n';
}

}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1 &&
                ::igrlab::dg_face::UsesOrdinaryCUDAForDGDiffusionFaces2D(),
              "Element-block fixture requires one native CUDA rank.");
  const bool gll_only = argc==2 && std::strcmp(argv[1], "--qualify-dgsem-mapped")==0;
  MFEM_VERIFY(argc==1 || gll_only, "Expected no arguments or --qualify-dgsem-mapped.");
  if (gll_only)
  {
    int current_device = 0;
    cudaDeviceProp properties{};
    MFEM_VERIFY(cudaGetDevice(&current_device) == cudaSuccess &&
                  cudaGetDeviceProperties(&properties, current_device) == cudaSuccess,
                "Could not inspect the CUDA device for mapped DGSEM qualification.");
    if (properties.major < 8)
    {
      std::cout << "SKIP: mapped DGSEM WMMA qualification requires SM80 or newer.\n";
      return 77;
    }
    CheckDGSEMMappedFixedLifecycle();
    CheckFusedSweepLifecycle(true, 2, true, true, "wmma-factor-fp64", BasisType::GaussLobatto);
    CheckFusedSweepLifecycle(true, 2, true, true, "simt-fp64", BasisType::GaussLobatto);
    return 0;
  }
  CheckCase(2, false);
  CheckCase(5, true);
  CheckCase(2, false, true);
  CheckCase(5, true, true);
  CheckSelfNeighbor();
  CheckLifecycle(false);
  CheckLifecycle(true);
  CheckLifecycle(false, true);
  CheckConvergedLocalTensor();
  int current_device = 0;
  cudaDeviceProp properties{};
  MFEM_VERIFY(cudaGetDevice(&current_device) == cudaSuccess &&
                cudaGetDeviceProperties(&properties, current_device) == cudaSuccess,
              "Could not inspect the CUDA device for fused projected-WMMA qualification.");
  if (properties.major >= 8)
  {
    CheckFusedSweepLifecycle(true, 2, true, false);
    CheckFusedSweepLifecycle(true, 2, false, true);
    CheckFusedSweepLifecycle(true, 2, true, true);
    CheckFusedSweepLifecycle(true, 3, true, true);
    CheckFusedSweepLifecycle(false, 2, true, true);
    CheckFusedSweepLifecycle(true, 2, true, true, "simt-fp64");
  }
  std::cout << "element_block_jacobi_test passed current_coefficients=true "
               "assembled_blocks=true full_A_columns=true fp64_lu=true fp64_explicit_inverse=true "
               "coarse=false skew=true curved=true dense_fallback=true self_neighbor=true\n";
  return 0;
}
