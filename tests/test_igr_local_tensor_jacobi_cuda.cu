#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_local_tensor_jacobi.hpp"
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"
#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
// Keep mfem::igrlab out of the translation-unit scope for NVCC registration.
using mfem::Array;
using mfem::AssemblyLevel;
using mfem::BasisType;
using mfem::BilinearForm;
using mfem::Coefficient;
using mfem::DenseMatrix;
using mfem::DenseMatrixInverse;
using mfem::DenseTensor;
using mfem::DGDiffusionIntegrator;
using mfem::DG_FECollection;
using mfem::DiffusionIntegrator;
using mfem::Element;
using mfem::FaceQuadratureSpace;
using mfem::FaceType;
using mfem::FunctionCoefficient;
using mfem::Geometry;
using mfem::IntegrationRule;
using mfem::IntRules;
using mfem::MassIntegrator;
using mfem::Mesh;
using mfem::Ordering;
using mfem::ParFiniteElementSpace;
using mfem::ParMesh;
using mfem::QuadratureFunction;
using mfem::QuadratureSpace;
using mfem::Reshape;
using mfem::Vector;
using namespace ::igrlab::dg_face;

void AddDomain(BilinearForm& form, Coefficient& mass, Coefficient& diffusion,
                const IntegrationRule& mass_rule, const IntegrationRule& diffusion_rule)
{
  auto* m = new MassIntegrator(mass);
  auto* k = new DiffusionIntegrator(diffusion);
  m->SetIntRule(&mass_rule); k->SetIntRule(&diffusion_rule);
  form.AddDomainIntegrator(m); form.AddDomainIntegrator(k);
}

void FillMass(ParMesh& mesh, QuadratureFunction& qf, Coefficient& coefficient)
{
  auto* qs = dynamic_cast<QuadratureSpace*>(qf.GetSpace());
  MFEM_VERIFY(qs, "Fixture requires volume quadrature.");
  double* values = qf.HostWrite();
  for (int e = 0; e < qs->GetNE(); ++e)
  {
    auto* tr = mesh.GetElementTransformation(e);
    const auto& rule = qs->GetIntRule(e);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const auto& ip = rule.IntPoint(q); tr->SetIntPoint(&ip);
      values[qs->Offset(e) + q] = coefficient.Eval(*tr, ip);
    }
  }
}

void FillFace(ParMesh& mesh, const FaceQuadratureSpace& qs,
              Coefficient& coefficient, Vector& side)
{
  const auto& rule = qs.GetIntRule(0);
  side.SetSize(2 * rule.GetNPoints() * qs.GetNE());
  auto values = Reshape(side.HostWrite(), 2, rule.GetNPoints(), qs.GetNE());
  for (int f = 0; f < qs.GetNE(); ++f)
  {
    auto* tr = mesh.GetFaceElementTransformations(qs.GetMeshFaceIndex(f));
    MFEM_VERIFY(tr && tr->Elem1 && tr->Elem2, "Fixture expects local interior faces.");
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      tr->SetAllIntPoints(&rule.IntPoint(q));
      values(0, q, f) = coefficient.Eval(*tr->Elem1, tr->GetElement1IntPoint());
      values(1, q, f) = coefficient.Eval(*tr->Elem2, tr->GetElement2IntPoint());
    }
  }
}

double RelativeDifference(const Vector& actual, const Vector& expected)
{
  MFEM_VERIFY(actual.Size() == expected.Size(), "Fixture vector sizes differ.");
  const double* a = actual.HostRead(); const double* b = expected.HostRead();
  long double delta = 0.0L, scale = 0.0L;
  for (int i = 0; i < actual.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite fixture result.");
    const long double d = static_cast<long double>(a[i]) - b[i];
    delta += d * d; scale += static_cast<long double>(b[i]) * b[i];
  }
  return std::sqrt(double(delta / std::max(scale, 1.0e-300L)));
}

Vector AssembledDiagonal(ParFiniteElementSpace& space, Coefficient& mass,
                          Coefficient& diffusion, const IntegrationRule& mass_rule,
                          const IntegrationRule& diffusion_rule,
                          const IntegrationRule& face_rule, double kappa)
{
  BilinearForm form(&space);
  AddDomain(form, mass, diffusion, mass_rule, diffusion_rule);
  auto* face = new DGDiffusionIntegrator(diffusion, -1.0, kappa);
  face->SetIntRule(&face_rule); form.AddInteriorFaceIntegrator(face);
  form.Assemble(); form.Finalize();
  DenseMatrix full; form.SpMat().ToDenseMatrix(full);
  const int dof = space.GetFE(0)->GetDof();
  Vector blocks(space.GetNE() * dof * dof);
  double* data = blocks.HostWrite();
  Array<int> dofs;
  for (int e = 0; e < space.GetNE(); ++e)
  {
    space.GetElementDofs(e, dofs);
    for (int j = 0; j < dof; ++j)
      for (int i = 0; i < dof; ++i)
      { data[std::size_t(e) * dof * dof + i + dof * j] = full(dofs[i], dofs[j]); }
  }
  return blocks;
}

Vector SolveBlocks(ParFiniteElementSpace& space, const Vector& matrices,
                     const Vector& rhs)
{
  const int dof = space.GetFE(0)->GetDof();
  Vector result(rhs.Size()), local_rhs(dof), local_result(dof);
  const double* data = matrices.HostRead(); const double* input = rhs.HostRead();
  double* output = result.HostWrite();
  Array<int> dofs;
  for (int e = 0; e < space.GetNE(); ++e)
  {
    space.GetElementDofs(e, dofs);
    DenseMatrix block(dof);
    std::copy(data + std::size_t(e) * dof * dof,
              data + std::size_t(e + 1) * dof * dof, block.Data());
    DenseMatrixInverse inverse(block);
    for (int i = 0; i < dof; ++i) { local_rhs[i] = input[dofs[i]]; }
    inverse.Mult(local_rhs, local_result);
    for (int i = 0; i < dof; ++i) { output[dofs[i]] = local_result[i]; }
  }
  return result;
}

Vector DenseSurrogate(ParFiniteElementSpace& space, const Vector& initial_blocks,
                        const QuadratureFunction& initial_mass,
                        const QuadratureFunction& current_mass)
{
  const auto& fe = *space.GetFE(0);
  const auto& lex = dynamic_cast<const mfem::TensorBasisElement&>(fe).GetDofMap();
  const int dof = fe.GetDof();
  const auto* initial = initial_mass.HostRead(); const auto* current = current_mass.HostRead();
  const double* blocks = initial_blocks.HostRead();
  Vector surrogate(initial_blocks.Size()), ratio(dof);
  double* data = surrogate.HostWrite();
  for (int e = 0; e < space.GetNE(); ++e)
  {
    for (int q = 0; q < dof; ++q)
    {
      const int native = lex.Size() ? lex[q] : q;
      // Fixed physical mass weights cancel in M_current / M_initial.
      ratio[native] = std::sqrt(current[e * dof + q] / initial[e * dof + q]);
    }
    for (int j = 0; j < dof; ++j)
      for (int i = 0; i < dof; ++i)
      {
        const std::size_t index = std::size_t(e) * dof * dof + i + dof * j;
        data[index] = ratio[i] * blocks[index] * ratio[j];
      }
  }
  return surrogate;
}

void CheckCase(int order)
{
  Mesh serial = Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true, 1.3, 0.7);
  Array<int> permutation(serial.GetNE());
  for (int e = 0; e < permutation.Size(); ++e) { permutation[e] = (5 * e + 3) % permutation.Size(); }
  serial.ReorderElements(permutation, false);
  ParMesh mesh(MPI_COMM_WORLD, serial);
  DG_FECollection collection(order, 2, BasisType::GaussLegendre);
  ParFiniteElementSpace space(&mesh, &collection);
  const int n = space.GetVSize(), dof = space.GetFE(0)->GetDof();
  const auto& mass_rule = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
  // The exact initial inverse must honor the actual, different diffusion rule.
  const auto& diffusion_rule = IntRules.Get(Geometry::SQUARE, 2 * order + 3);
  const auto& face_rule = IntRules.Get(Geometry::SEGMENT, 2 * order + 1);
  QuadratureSpace mass_space(mesh, mass_rule);
  QuadratureFunction invrho(mass_space), initial_invrho(mass_space);
  int phase = 0;
  const auto density = [&](const Vector& x)
  { return phase == 0 ? 1.0 : (phase == 1 ? 0.45 : 0.45 + 0.08 * x[0] + 0.05 * x[1]); };
  FunctionCoefficient mass([&](const Vector& x) { return 1.0 / density(x); });
  FunctionCoefficient diffusion([&](const Vector& x) { return 0.017 / density(x); });
  FillMass(mesh, invrho, mass); FillMass(mesh, initial_invrho, mass);
  BilinearForm volume(&space);
  AddDomain(volume, mass, diffusion, mass_rule, diffusion_rule);
  volume.SetAssemblyLevel(AssemblyLevel::PARTIAL); volume.Assemble();
  FaceQuadratureSpace face_space(mesh, face_rule, FaceType::Interior);
  InteriorFaceDGDiffusionApplyScratch work;
  work.compact_qdata = true; work.compact_lift = true;
  Vector side; FillFace(mesh, face_space, diffusion, side);
  const double kappa = 2.0 * (order + 1) * (order + 1);
  const auto info = BuildInteriorFaceDGDiffusionApplyData2D(
    space, face_space, side, kappa, work.shape, work.weighted_normal_dshape,
    work.penalty, work.face_elements, work.element_vdof_map,
    DGDiffusionFaceScaling::Arithmetic, &work.normal_dshape_base,
    &work.penalty_metric, &work.shape_dictionary, &work.normal_factorization);
  LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(info, work, side, kappa,
                                                DGDiffusionFaceScaling::Arithmetic);
  // PA/face preparation may install mesh Nodes lazily. Capture the fixed
  // geometry sequence only after those initial caches have been established.
  auto backend = ::igrlab::igr_internal::MakeTensorElementBlockJacobi(space, invrho);
  DenseTensor& built = backend->PrepareVolumeBlocks(volume);
  ::igrlab::igr_internal::AddExactFaceBlocks(info, work, -1.0, built);
  Vector initial_blocks = AssembledDiagonal(space, mass, diffusion, mass_rule,
                                             diffusion_rule, face_rule, kappa);
  {
    Vector actual(const_cast<double*>(built.HostRead()), built.TotalSize());
    MFEM_VERIFY(RelativeDifference(actual, initial_blocks) < 5.0e-11,
                "Initial exact EA/face blocks differ from assembled MFEM.");
  }
  backend->FactorBlocks();
  MFEM_VERIFY(built.TotalSize() == 0, "Local tensor Jacobi retained dense element build storage.");
  Vector rhs(n), second(n), actual(n), initial_action(n), action2(n);
  for (int i = 0; i < n; ++i)
  {
    rhs[i] = std::sin(0.17 * (i + 1)) + 0.11 * std::cos(0.037 * i);
    second[i] = std::cos(0.09 * (i + 2)) - 0.13 * std::sin(0.12 * i);
  }
  bool boundary = false, interior = false;
  Array<int> adjacent, orientations;
  for (int e = 0; e < mesh.GetNE(); ++e)
  {
    mesh.GetElementEdges(e, adjacent, orientations);
    int neighbors = 0;
    for (int j = 0; j < adjacent.Size(); ++j)
    {
      int e1 = -1, e2 = -1; mesh.GetFaceElements(adjacent[j], &e1, &e2);
      if (e1 >= 0 && e2 >= 0) { ++neighbors; }
    }
    boundary = boundary || neighbors < 4; interior = interior || neighbors == 4;
  }
  MFEM_VERIFY(boundary && interior, "Tensor fixture missed boundary/interior face masks.");
  for (int pass = 0; pass < 4; ++pass)
  {
    phase = pass < 3 ? pass : 1;
    FillMass(mesh, invrho, mass);
    backend->RefreshMassScaling(invrho);
    backend->Mult(rhs, actual); backend->Mult(second, action2);
    const Vector surrogate = DenseSurrogate(space, initial_blocks, initial_invrho, invrho);
    const Vector expected = SolveBlocks(space, surrogate, rhs);
    const double surrogate_error = RelativeDifference(actual, expected);
    MFEM_VERIFY(surrogate_error < 5.0e-10,
                "FP64 tensor action differs from independent dense current-mass surrogate: " << surrogate_error);
    const double xpy = rhs * action2, ypx = second * actual;
    const double symmetry = std::abs(xpy - ypx) / std::max({std::abs(xpy), std::abs(ypx), 1.0e-300});
    MFEM_VERIFY(std::isfinite(xpy) && std::isfinite(ypx) && symmetry < 2.0e-11 &&
                  rhs * actual > 0.0 && second * action2 > 0.0,
                "Current-mass tensor inverse lost symmetry or positive action.");
    Vector current_blocks = AssembledDiagonal(space, mass, diffusion, mass_rule,
                                               diffusion_rule, face_rule, kappa);
    const Vector current = SolveBlocks(space, current_blocks, rhs);
    const double current_error = RelativeDifference(actual, current);
    if (phase < 2)
    {
      MFEM_VERIFY(current_error < 5.0e-10,
                  "Uniform-density tensor action differs from exact current block inverse: " << current_error);
      if (phase == 0) { initial_action = actual; }
      else
      {
        MFEM_VERIFY(RelativeDifference(actual, initial_action) > 0.5,
                    "rho=.45 tensor update reused stale initial mass scaling.");
      }
    }
    else
    {
      MFEM_VERIFY(current_error > 1.0e-5,
                  "Heterogeneous fixture did not distinguish the surrogate from exact current blocks.");
    }
    MFEM_VERIFY(built.TotalSize() == 0, "Current tensor update restored dense element storage.");
    std::cout << "local_tensor_jacobi_case p=" << order << " phase=" << phase
              << " rho=" << (phase == 0 ? "1" : (phase == 1 ? "0.45" : "heterogeneous"))
              << " surrogate_relative=" << surrogate_error << " current_block_relative=" << current_error
              << " symmetry_relative=" << symmetry
              << " constant_density_exact=" << (phase < 2 ? "true" : "not-applicable")
              << " current_mass_refresh=true dense_after_setup=0 passed\n";
  }
}

// Separate expected-failure invocations avoid exception/build-option assumptions
// and prove that unsupported inputs are rejected before any fallback can run.
void ExpectedRejection(const std::string& kind)
{
  Mesh serial = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.1, 0.9);
  if (kind == "skew")
  {
    serial.Transform([](const Vector& x, Vector& y)
    { y.SetSize(2); y[0] = x[0] + 0.2 * x[1]; y[1] = x[1]; });
  }
  if (kind == "curved")
  {
    serial.SetCurvature(2, false, 2, Ordering::byVDIM);
    serial.Transform([](const Vector& x, Vector& y)
    { y.SetSize(2); y[0] = x[0] + 0.1 * x[0] * x[1]; y[1] = x[1]; });
  }
  ParMesh mesh(MPI_COMM_WORLD, serial);
  DG_FECollection collection(5, 2, BasisType::GaussLegendre);
  ParFiniteElementSpace space(&mesh, &collection);
  const auto& rule = IntRules.Get(Geometry::SQUARE, 11);
  QuadratureSpace qs(mesh, rule); QuadratureFunction invrho(qs);
  invrho = 1.0;
  std::cout << "local_tensor_jacobi_expected_rejection kind=" << kind << std::endl;
  if (kind == "nonseparable")
  {
    mfem::ConstantCoefficient mass(1.0);
    FunctionCoefficient diffusion([](const Vector& x) { return 0.017 * (1.0 + 0.4 * x[0] * x[1]); });
    BilinearForm volume(&space);
    AddDomain(volume, mass, diffusion, rule, rule);
    volume.SetAssemblyLevel(AssemblyLevel::PARTIAL); volume.Assemble();
    auto backend = ::igrlab::igr_internal::MakeTensorElementBlockJacobi(space, invrho);
    backend->PrepareVolumeBlocks(volume); backend->FactorBlocks();
  }
  else
  {
    auto backend = ::igrlab::igr_internal::MakeTensorElementBlockJacobi(space, invrho);
  }
  // Returning success makes a CTest WILL_FAIL rejection test fail.
  std::cout << "local_tensor_jacobi_unexpected_admission kind=" << kind << std::endl;
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1 && UsesOrdinaryCUDAForDGDiffusionFaces2D(),
              "Local tensor Jacobi fixture requires one native CUDA rank.");
  if (argc == 2 && std::strncmp(argv[1], "--reject-", 9) == 0)
  {
    const std::string kind(argv[1] + 9);
    MFEM_VERIFY(kind == "skew" || kind == "curved" || kind == "nonseparable", "Unknown rejection fixture.");
    mfem::set_error_action(mfem::MFEM_ERROR_ABORT);
    ExpectedRejection(kind);
    return 0;
  }
  MFEM_VERIFY(argc == 1, "Unexpected local tensor Jacobi fixture arguments.");
  CheckCase(2); CheckCase(5);
  std::cout << "local_tensor_jacobi_test passed constant_density_exact=true "
               "heterogeneous_current_mass=true dense_surrogate=true spd=true "
               "native_ordering=true no_dense_after_setup=true fallback=none\n";
  return 0;
}
