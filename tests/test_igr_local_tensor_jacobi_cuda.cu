#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_local_tensor_jacobi.hpp"
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"
#include "mfem.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
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

const std::vector<std::string>& ExecutionModes()
{
  static const std::vector<std::string> modes{
    "simt-fp64", "simt-fp32", "simt-dense-fp32", "wmma-factor-tf32",
    "wmma-factor-fp64", "wmma-dense-tf32", "wmma-dense-fp64"};
  return modes;
}

bool IsFP64(const std::string& mode)
{ return mode == "simt-fp64" || mode == "wmma-factor-fp64" || mode == "wmma-dense-fp64"; }

bool IsDense(const std::string& mode)
{ return mode.find("dense") != std::string::npos; }

class ExecutionSelection
{
public:
  explicit ExecutionSelection(const std::string& mode)
  {
    const char* previous = std::getenv("IGRLAB_IGR_LOCAL_TENSOR_EXECUTION");
    had_previous_ = previous != nullptr;
    if (previous) { previous_ = previous; }
    MFEM_VERIFY(::setenv("IGRLAB_IGR_LOCAL_TENSOR_EXECUTION", mode.c_str(), 1) == 0,
                "Could not select local tensor execution mode.");
  }
  ~ExecutionSelection()
  {
    if (had_previous_)
    { ::setenv("IGRLAB_IGR_LOCAL_TENSOR_EXECUTION", previous_.c_str(), 1); }
    else { ::unsetenv("IGRLAB_IGR_LOCAL_TENSOR_EXECUTION"); }
  }
private:
  bool had_previous_ = false;
  std::string previous_;
};

class ReferenceSelection
{
public:
  explicit ReferenceSelection(const std::string& policy)
  {
    const char* previous = std::getenv("IGRLAB_IGR_LOCAL_TENSOR_REFERENCE");
    had_previous_ = previous != nullptr;
    if (previous) { previous_ = previous; }
    MFEM_VERIFY(::setenv("IGRLAB_IGR_LOCAL_TENSOR_REFERENCE", policy.c_str(), 1) == 0,
                "Could not select local tensor reference policy.");
  }
  ~ReferenceSelection()
  {
    if (had_previous_)
    { ::setenv("IGRLAB_IGR_LOCAL_TENSOR_REFERENCE", previous_.c_str(), 1); }
    else { ::unsetenv("IGRLAB_IGR_LOCAL_TENSOR_REFERENCE"); }
  }
private:
  bool had_previous_ = false;
  std::string previous_;
};

Mesh QualificationMesh(int nx, int ny, double width, double height, bool reorder)
{
  Mesh serial = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                     true, width, height, false);
  if (reorder)
  {
    Array<int> permutation(serial.GetNE());
    for (int e = 0; e < permutation.Size(); ++e)
    { permutation[e] = (5 * e + 3) % permutation.Size(); }
    serial.ReorderElements(permutation, false);
  }
  return serial;
}

// This is only fixture setup. Every execution mode is constructed through the
// production factory and uses its PrepareVolumeBlocks/FactorBlocks/Mult path.
struct TensorFixture
{
  int nx, ny;
  double width, height, alpha;
  Mesh serial;
  ParMesh mesh;
  DG_FECollection collection;
  ParFiniteElementSpace space;
  const IntegrationRule &mass_rule, &diffusion_rule, &face_rule;
  QuadratureSpace mass_space;
  QuadratureFunction invrho, initial_invrho;
  int phase = 0;
  FunctionCoefficient mass, diffusion;
  BilinearForm volume;
  FaceQuadratureSpace face_space;
  InteriorFaceDGDiffusionApplyScratch work;
  InteriorFaceDGDiffusionApplyInfo info;
  Vector side;
  const double kappa;

  TensorFixture(int nx_, int ny_, double width_, double height_, double alpha_,
                bool reorder = false, bool production_quadrature = false)
    : nx(nx_), ny(ny_), width(width_), height(height_), alpha(alpha_),
      serial(QualificationMesh(nx, ny, width, height, reorder)),
      mesh(MPI_COMM_WORLD, serial), collection(5, 2, BasisType::GaussLegendre),
      space(&mesh, &collection), mass_rule(IntRules.Get(Geometry::SQUARE, 11)),
      diffusion_rule(IntRules.Get(Geometry::SQUARE, production_quadrature ? 11 : 13)),
      face_rule(IntRules.Get(Geometry::SEGMENT, 11)), mass_space(mesh, mass_rule),
      invrho(mass_space), initial_invrho(mass_space),
      mass([this](const Vector& x) { return 1.0 / Density(x); }),
      diffusion([this](const Vector& x) { return alpha / Density(x); }),
      volume(&space), face_space(mesh, face_rule, FaceType::Interior),
      kappa(production_quadrature ? 144.0 : 72.0)
  {
    FillMass(mesh, invrho, mass); FillMass(mesh, initial_invrho, mass);
    AddDomain(volume, mass, diffusion, mass_rule, diffusion_rule);
    volume.SetAssemblyLevel(AssemblyLevel::PARTIAL); volume.Assemble();
    work.compact_qdata = true; work.compact_lift = true;
    FillFace(mesh, face_space, diffusion, side);
    info = BuildInteriorFaceDGDiffusionApplyData2D(
      space, face_space, side, kappa, work.shape, work.weighted_normal_dshape,
      work.penalty, work.face_elements, work.element_vdof_map,
      DGDiffusionFaceScaling::Arithmetic, &work.normal_dshape_base,
      &work.penalty_metric, &work.shape_dictionary, &work.normal_factorization);
    LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
      info, work, side, kappa, DGDiffusionFaceScaling::Arithmetic);
  }

  double Density(const Vector& x) const
  {
    if (phase == 0) { return 1.0; }
    if (phase == 1) { return 0.45; }
    const double xx = x[0] / width, yy = x[1] / height;
    if (phase == 2)
    {
      return 0.55 + 0.19 * std::sin(5.7 * xx + 0.3) +
             0.13 * std::cos(4.9 * yy - 0.2) + 0.08 * xx * yy;
    }
    // The dynamic-density case spans almost four decades across GL points.
    return std::pow(10.0, 4.0 * (xx - 0.5)) * (0.8 + 0.4 * yy);
  }

  void SetPhase(int next)
  { phase = next; FillMass(mesh, invrho, mass); }

  std::unique_ptr<::igrlab::igr_internal::ElementBlockJacobi> Prepare()
  {
    auto backend = ::igrlab::igr_internal::MakeTensorElementBlockJacobi(space, initial_invrho);
    DenseTensor& blocks = backend->PrepareVolumeBlocks(volume);
    ::igrlab::igr_internal::AddExactFaceBlocks(info, work, -1.0, blocks);
    return backend;
  }
};

struct ErrorMetrics
{
  double relative_l2 = 0.0, relative_max = 0.0;
  long double norm_l2 = 0.0L, norm_max = 0.0L;
};

template <typename Accumulator>
ErrorMetrics MeasureErrorWith(const Vector& actual, const Vector& expected)
{
  MFEM_VERIFY(actual.Size() == expected.Size(), "Correction dimensions differ.");
  const double* a = actual.HostRead(); const double* b = expected.HostRead();
  Accumulator error2 = 0, norm2 = 0, error_max = 0, norm_max = 0;
  for (int i = 0; i < actual.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite correction.");
    const Accumulator d = static_cast<Accumulator>(a[i]) - b[i];
    error2 += d * d; norm2 += static_cast<Accumulator>(b[i]) * b[i];
    error_max = std::max(error_max, std::abs(d));
    norm_max = std::max(norm_max, std::abs(static_cast<Accumulator>(b[i])));
  }
  MFEM_VERIFY(norm2 > 0.0L && norm_max > 0.0L, "Correction oracle is zero.");
  return {std::sqrt(double(error2 / norm2)), double(error_max / norm_max),
          std::sqrt(norm2), norm_max};
}

ErrorMetrics MeasureError(const Vector& actual, const Vector& expected)
{ return MeasureErrorWith<long double>(actual, expected); }

long double Dot(const Vector& x, const Vector& y)
{
  MFEM_VERIFY(x.Size() == y.Size(), "Quadratic form dimensions differ.");
  const double* a = x.HostRead(); const double* b = y.HostRead();
  long double sum = 0.0L;
  for (int i = 0; i < x.Size(); ++i) { sum += static_cast<long double>(a[i]) * b[i]; }
  return sum;
}

double DotDouble(const Vector& x, const Vector& y)
{
  const double* a = x.HostRead(); const double* b = y.HostRead();
  double sum = 0.0;
  for (int i = 0; i < x.Size(); ++i) { sum += a[i] * b[i]; }
  return sum;
}

struct OracleBudget
{
  long double absolute_l2 = 0.0L, absolute_max = 0.0L;
};

struct UnscaledOracleBudgets
{
  OracleBudget dense, factor;
};

struct DenseClass
{
  std::vector<double> block, inverse, physical_mass;
  double inverse_h_frobenius = 0.0, condition_h_inf = 0.0;
};

// Independent pivoted FP64 dense inverses of native local matrices. The small
// qualification uses separately assembled MFEM matrices. The full benchmark
// copies only its nine exact SIPG class matrices before production factorization;
// no production eigenvectors, packed inverses or kernel internals are inspected.
class CompactDenseReference
{
public:
  CompactDenseReference(TensorFixture& fixture, const double* blocks, bool nine_classes)
    : fixture_(fixture), dof_(fixture.space.GetFE(0)->GetDof()),
      element_class_(fixture.space.GetNE()), native_to_lex_(dof_)
  {
    const auto& lex = dynamic_cast<const mfem::TensorBasisElement&>(
      *fixture.space.GetFE(0)).GetDofMap();
    for (int q = 0; q < dof_; ++q) { native_to_lex_[lex.Size() ? lex[q] : q] = q; }
    classes_.resize(nine_classes ? 9 : fixture.space.GetNE());
    mfem::IntegrationPoint center; center.Set2(0.5, 0.5);
    Vector position(2);
    const double* coefficient = fixture.initial_invrho.HostRead();
    for (int e = 0; e < fixture.space.GetNE(); ++e)
    {
      auto* tr = fixture.mesh.GetElementTransformation(e);
      tr->Transform(center, position);
      const int xclass = position[0] < fixture.width / fixture.nx ? 0 :
                         (position[0] > fixture.width * (1.0 - 1.0 / fixture.nx) ? 2 : 1);
      const int yclass = position[1] < fixture.height / fixture.ny ? 0 :
                         (position[1] > fixture.height * (1.0 - 1.0 / fixture.ny) ? 2 : 1);
      const int c = nine_classes ? xclass + 3 * yclass : e;
      element_class_[e] = c;
      auto& cls = classes_[c];
      const double* block = blocks + std::size_t(e) * dof_ * dof_;
      if (!cls.block.empty())
      {
        double norm = 0.0, error = 0.0;
        for (int i = 0; i < dof_ * dof_; ++i)
        {
          norm = std::max(norm, std::abs(block[i]));
          error = std::max(error, std::abs(block[i] - cls.block[i]));
        }
        MFEM_VERIFY(error < 1.0e-11 * norm, "Full benchmark geometry class mismatch.");
        continue;
      }
      cls.block.assign(block, block + dof_ * dof_);
      cls.physical_mass.resize(dof_);
      for (int i = 0; i < dof_; ++i)
      {
        const int q = native_to_lex_[i];
        const auto& ip = fixture.mass_rule.IntPoint(q); tr->SetIntPoint(&ip);
        cls.physical_mass[i] = ip.weight * tr->Weight() * coefficient[e * dof_ + q];
      }
      DenseMatrix matrix(dof_), inverse(dof_);
      std::copy(cls.block.begin(), cls.block.end(), matrix.Data());
      DenseMatrixInverse factor(matrix);
      factor.GetInverseMatrix(inverse);
      cls.inverse.assign(inverse.Data(), inverse.Data() + dof_ * dof_);
      double h_inf = 0.0, inverse_h_inf = 0.0;
      long double inverse_h_f2 = 0.0L;
      for (int i = 0; i < dof_; ++i)
      {
        double h_row = 0.0, inverse_h_row = 0.0;
        for (int j = 0; j < dof_; ++j)
        {
          const double mass = std::sqrt(cls.physical_mass[i] * cls.physical_mass[j]);
          const double hinv = mass * cls.inverse[i + dof_ * j];
          h_row += std::abs(cls.block[i + dof_ * j] / mass);
          inverse_h_row += std::abs(hinv);
          inverse_h_f2 += static_cast<long double>(hinv) * hinv;
        }
        h_inf = std::max(h_inf, h_row);
        inverse_h_inf = std::max(inverse_h_inf, inverse_h_row);
      }
      cls.inverse_h_frobenius = double(std::sqrt(inverse_h_f2));
      cls.condition_h_inf = h_inf * inverse_h_inf;
    }
    for (const auto& cls : classes_)
    { MFEM_VERIFY(!cls.block.empty(), "Benchmark missed a boundary/interior class."); }
  }

  int ClassCount() const { return int(classes_.size()); }

  double MaximumCondition() const
  {
    double value = 0.0;
    for (const auto& cls : classes_) { value = std::max(value, cls.condition_h_inf); }
    return value;
  }

  OracleBudget Apply(const Vector& rhs, Vector& expected, const std::string& mode) const
  { return ScaleBudget(ApplyUnscaled<long double>(rhs, expected), mode); }

  // Full-size expected corrections and both unscaled envelopes are computed
  // once with hardware FP64 arithmetic. AArch64 long double is software IEEE128;
  // reserve that cost for the small extreme-exponent qualification fixtures.
  UnscaledOracleBudgets ApplyDouble(const Vector& rhs, Vector& expected) const
  { return ApplyUnscaled<double>(rhs, expected); }

  static OracleBudget ScaleBudget(const UnscaledOracleBudgets& budgets, const std::string& mode)
  {
    const double float_u = std::ldexp(1.0, -24);
    const double accumulation_u = IsFP64(mode) ? std::ldexp(1.0, -53) : float_u;
    const double operand_u = mode.find("tf32") != std::string::npos ? std::ldexp(1.0, -11) : accumulation_u;
    const double gamma = 38.0 * accumulation_u / (1.0 - 38.0 * accumulation_u);
    const double gamma_axis = 8.0 * accumulation_u / (1.0 - 8.0 * accumulation_u);
    // Dense: rounding both operands plus FP32 accumulation of a length-36 dot.
    // Factor: each of four orthogonal axis contractions has ||delta Q||_2
    // <= sqrt(6)*u, operand rounding <= u, and FP32 accumulation. The diagonal
    // division adds FP32 rounding. Current mass factors stay FP64. Frobenius
    // ||H^-1|| bounds its spectral norm, so this is an explicit forward-error
    // envelope, not a claim that TF32 meets a 1e-6 correction tolerance.
    const double delta = (std::sqrt(6.0) + 1.0) * operand_u + std::sqrt(6.0) * gamma_axis;
    const double arithmetic = IsDense(mode) ? (2.0 * operand_u + operand_u * operand_u + gamma) :
      std::pow(1.0 + delta, 4) * (1.0 + 3.0 * accumulation_u) - 1.0;
    const auto& unit = IsDense(mode) ? budgets.dense : budgets.factor;
    return {arithmetic * unit.absolute_l2, arithmetic * unit.absolute_max};
  }

private:
  template <typename Accumulator>
  UnscaledOracleBudgets ApplyUnscaled(const Vector& rhs, Vector& expected) const
  {
    expected.SetSize(rhs.Size());
    const double* input = rhs.HostRead(); double* output = expected.HostWrite();
    const double* current = fixture_.invrho.HostRead();
    const double* initial = fixture_.initial_invrho.HostRead();
    Accumulator dense_bound2 = 0, dense_bound_max = 0, factor_bound2 = 0, factor_bound_max = 0;
    Array<int> dofs;
    std::vector<double> ratio(dof_), local(dof_);
    for (int e = 0; e < fixture_.space.GetNE(); ++e)
    {
      fixture_.space.GetElementDofs(e, dofs);
      const auto& cls = classes_[element_class_[e]];
      double scaling_max = 0.0;
      Accumulator scaled_norm2 = 0;
      for (int j = 0; j < dof_; ++j)
      {
        const int q = e * dof_ + native_to_lex_[j];
        ratio[j] = std::sqrt(current[q] / initial[q]);
        local[j] = input[dofs[j]] / ratio[j];
        const double scaling = 1.0 / (std::sqrt(cls.physical_mass[j]) * ratio[j]);
        const Accumulator z = static_cast<Accumulator>(scaling) * input[dofs[j]];
        scaled_norm2 += z * z; scaling_max = std::max(scaling_max, scaling);
      }
      for (int i = 0; i < dof_; ++i)
      {
        Accumulator sum = 0, absolute_sum = 0;
        for (int j = 0; j < dof_; ++j)
        {
          const Accumulator term = static_cast<Accumulator>(cls.inverse[i + dof_ * j]) * local[j];
          sum += term; absolute_sum += std::abs(term);
        }
        output[dofs[i]] = double(sum / ratio[i]);
        const Accumulator row_bound = absolute_sum / ratio[i];
        dense_bound2 += row_bound * row_bound;
        dense_bound_max = std::max(dense_bound_max, row_bound);
      }
      const Accumulator element_bound = scaling_max *
        cls.inverse_h_frobenius * std::sqrt(scaled_norm2);
      factor_bound2 += element_bound * element_bound;
      factor_bound_max = std::max(factor_bound_max, element_bound);
    }
    return {{std::sqrt(dense_bound2), dense_bound_max}, {std::sqrt(factor_bound2), factor_bound_max}};
  }
  TensorFixture& fixture_;
  int dof_;
  std::vector<int> element_class_, native_to_lex_;
  std::vector<DenseClass> classes_;
};

void CheckCorrection(const ErrorMetrics& error, const OracleBudget& budget,
                     const std::string& mode)
{
  const double l2_limit = IsFP64(mode) ? 5.0e-10 :
    double(budget.absolute_l2 / error.norm_l2) + 5.0e-10;
  const double max_limit = IsFP64(mode) ? 5.0e-10 :
    double(budget.absolute_max / error.norm_max) + 5.0e-10;
  MFEM_VERIFY(error.relative_l2 <= l2_limit && error.relative_max <= max_limit,
              "Local tensor correction exceeded its FP64 oracle/roundoff envelope: backend=" << mode <<
              " relative_l2=" << error.relative_l2 << " relative_max=" << error.relative_max <<
              " l2_limit=" << l2_limit << " max_limit=" << max_limit);
}

void FillSignedResidual(Vector& rhs, int exponent, bool second = false)
{
  double* values = rhs.HostWrite();
  for (int i = 0; i < rhs.Size(); ++i)
  {
    const double value = second ? std::cos(0.09 * (i + 2)) - 0.13 * std::sin(0.12 * i) :
      std::sin(0.17 * (i + 1)) + 0.11 * std::cos(0.037 * i) + 0.03 * std::sin(0.71 * i);
    values[i] = std::scalbn(value, exponent);
  }
}

void CheckFusedSolutionUpdate(::igrlab::igr_internal::ElementBlockJacobi& backend,
                              const Vector& rhs, bool expected_admitted)
{
  const int n = rhs.Size(), padding = 7;
  Vector live_rhs(rhs), input_view, correction(n), expected(n), padded(n + 2 * padding), output;
  for (Vector* value : {&live_rhs, &correction, &expected, &padded}) { value->UseDevice(true); }
  const auto same = [](const Vector& a, const Vector& b)
  {
    const double* av = a.HostRead(); const double* bv = b.HostRead();
    for (int i = 0; i < a.Size(); ++i)
    {
      if (std::isnan(av[i]) && std::isnan(bv[i])) { continue; }
      if (std::memcmp(av + i, bv + i, sizeof(double)) != 0) { return false; }
    }
    return true;
  };
  for (const bool nonfinite : {false, true, false})
  {
    const double* initial = rhs.Read();
    double* current = live_rhs.Write();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int i)
    {
      current[i] = nonfinite && i == 0 ? nan :
        (nonfinite && i == 1 ? inf : initial[i] + 0.003 * std::cos(0.13 * (i + 1)));
    });
    input_view.MakeRef(live_rhs, 0, n);
    MFEM_VERIFY(!input_view.GetMemory().HostIsValid(),
                "Fused update fixture must present a current device RHS and a stale host copy.");
    backend.Mult(input_view, correction);
    for (const double omega : {0.7, 1.0, -0.5, 0.0, -0.0})
    {
      double* values = padded.HostWrite();
      for (int i = 0; i < padded.Size(); ++i) { values[i] = 12345.0; }
      for (int i = 0; i < n; ++i)
      { values[padding + i] = i == 0 ? -0.0 : 0.125 * std::sin(0.09 * (i + 1)); }
      output.MakeRef(padded, padding, n);
      const Vector before(output);
      expected = output; expected.Add(omega, correction);
      const bool admitted = backend.TryAddMult(input_view, omega, output);
      MFEM_VERIFY(admitted == expected_admitted,
                  "Only the projected FP64 WMMA backend may admit the fused solution update.");
      if (!admitted)
      {
        MFEM_VERIFY(same(output, before), "A declined local update modified the solution.");
        backend.Mult(input_view, correction); output.Add(omega, correction);
      }
      MFEM_VERIFY(same(output, expected),
                  "Fused update changed rounded correction, damping, native ordering or nonfinite propagation.");
      const double* result = padded.HostRead();
      for (int i = 0; i < padding; ++i)
      { MFEM_VERIFY(result[i] == 12345.0 && result[n + padding + i] == 12345.0,
                    "Fused update modified a padded alias boundary."); }
    }
  }
  Vector overlap(n + 1), left, right;
  overlap.UseDevice(true); overlap = 0.25;
  left.MakeRef(overlap, 0, n); right.MakeRef(overlap, 1, n);
  const Vector before(overlap);
  MFEM_VERIFY(!backend.TryAddMult(left, 0.7, right) &&
                !backend.TryAddMult(left, 0.7, left) && same(overlap, before),
              "Fused local update must decline complete and offset RHS/solution overlap without writes.");
  std::cout << "local_tensor_fused_update admitted=" << expected_admitted
            << " ordinary_Mult_Add_parity=bitwise finite=true nonfinite=true recovery=true"
            << " damping=positive:negative:signed-zero alias_boundaries=true overlap_declined=true"
            << " device_current_rhs=true native_order=true\n";
}

void QualifyBackends(const std::vector<std::string>& modes)
{
  TensorFixture fixture(4, 3, 1.3, 0.7, 0.017, true);
  const Vector independent = AssembledDiagonal(fixture.space, fixture.mass, fixture.diffusion,
    fixture.mass_rule, fixture.diffusion_rule, fixture.face_rule, fixture.kappa);
  CompactDenseReference reference(fixture, independent.HostRead(), false);
  const int n = fixture.space.GetVSize();
  Vector rhs(n), second(n), actual(n), action2(n), expected(n), expected2(n);
  for (const auto& mode : modes)
  {
    ExecutionSelection select(mode);
    fixture.SetPhase(0);
    auto backend = fixture.Prepare(); backend->FactorBlocks();
    FillSignedResidual(rhs, 0);
    CheckFusedSolutionUpdate(*backend, rhs, false);
    struct Case { int phase, exponent; };
    std::vector<Case> cases{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {2, 80}, {2, -80}};
    if (IsFP64(mode)) { cases.push_back({2, 700}); cases.push_back({2, -700}); }
    for (const auto& test : cases)
    {
      fixture.SetPhase(test.phase); backend->RefreshMassScaling(fixture.invrho);
      FillSignedResidual(rhs, test.exponent); FillSignedResidual(second, test.exponent, true);
      backend->Mult(rhs, actual); backend->Mult(second, action2);
      const auto budget = reference.Apply(rhs, expected, mode);
      const auto budget2 = reference.Apply(second, expected2, mode);
      const auto error = MeasureError(actual, expected), error2 = MeasureError(action2, expected2);
      CheckCorrection(error, budget, mode); CheckCorrection(error2, budget2, mode);
      const long double q1 = Dot(rhs, actual), q2 = Dot(second, action2);
      MFEM_VERIFY(std::isfinite(q1) && std::isfinite(q2) && q1 > 0.0L && q2 > 0.0L,
                  "Local tensor correction lost positive quadratic action.");
      const long double scale = std::sqrt(q1) * std::sqrt(q2);
      const double symmetry = double(std::abs(Dot(rhs, action2) - Dot(second, actual)) / scale);
      const long double rnorm = std::sqrt(Dot(rhs, rhs)), snorm = std::sqrt(Dot(second, second));
      const double symmetry_limit = IsFP64(mode) ? 5.0e-10 :
        double((budget.absolute_l2 * snorm + budget2.absolute_l2 * rnorm) / scale) + 5.0e-10;
      MFEM_VERIFY(symmetry <= symmetry_limit, "Local tensor inverse exceeded symmetry roundoff envelope.");
      std::cout << std::setprecision(17)
                << "local_tensor_jacobi_qualification backend=" << mode << " p=5 phase=" << test.phase
                << " residual_exponent=" << test.exponent
                << " correction_relative_l2=" << error.relative_l2
                << " correction_relative_max=" << error.relative_max
                << " roundoff_envelope_relative_l2=" << double(budget.absolute_l2 / error.norm_l2)
                << " roundoff_envelope_relative_max=" << double(budget.absolute_max / error.norm_max)
                << " symmetry_relative=" << symmetry << " positive_quadratic=true"
                << " initial_h_condition_inf_max=" << reference.MaximumCondition()
                << " oracle=independent-mfem-dense-fp64"
                << " float_input_scaling=" << (IsFP64(mode) ? "fp64" : "direct-cast-no-rescaling")
                << " passed\n";
    }
  }
  std::cout << "local_tensor_jacobi_backend_test passed modes=" << modes.size()
            << " fp32_tf32_exponents=-80:80 fp64_exponents=-700:700"
            << " float_extreme_exponents_unsupported=true physical_accuracy_claim=false\n";
}

struct Options
{
  bool benchmark = false, profile = false, qualify = false, qualify_projected = false, help = false;
  int mesh_n = 512, repetitions = 64, warmup = 8, rounds = 3;
  std::string backend = "all", reference = "exact", write_mapped_mesh;
};

Options ParseOptions(int argc, char** argv)
{
  Options options;
  bool repetitions_set = false, warmup_set = false, rounds_set = false;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);
    auto value = [&]() -> const char*
    { MFEM_VERIFY(i + 1 < argc, "Missing value for " << arg); return argv[++i]; };
    if (arg == "--benchmark") { options.benchmark = true; }
    else if (arg == "--profile") { options.benchmark = true; options.profile = true; }
    else if (arg == "--qualify-backends") { options.qualify = true; }
    else if (arg == "--qualify-projected") { options.qualify_projected = true; options.reference = "projected"; }
    else if (arg == "--reference") { options.reference = value(); }
    else if (arg == "--write-mapped-mesh") { options.write_mapped_mesh = value(); }
    else if (arg == "--backend") { options.backend = value(); }
    else if (arg == "--mesh-n") { options.mesh_n = std::stoi(value()); }
    else if (arg == "--repetitions")
    { options.repetitions = std::stoi(value()); repetitions_set = true; }
    else if (arg == "--warmup") { options.warmup = std::stoi(value()); warmup_set = true; }
    else if (arg == "--rounds") { options.rounds = std::stoi(value()); rounds_set = true; }
    else if (arg == "--help" || arg == "-h") { options.help = true; }
    else { MFEM_ABORT("Unknown local tensor fixture argument: " << arg); }
  }
  MFEM_VERIFY(options.backend == "all" ||
    std::find(ExecutionModes().begin(), ExecutionModes().end(), options.backend) != ExecutionModes().end(),
    "Unknown local tensor backend: " << options.backend);
  MFEM_VERIFY(options.reference == "exact" || options.reference == "projected", "Unknown tensor reference policy.");
  const int actions = int(options.qualify) + int(options.qualify_projected) +
                      int(options.benchmark) + int(!options.write_mapped_mesh.empty());
  MFEM_VERIFY(actions <= 1, "Select one qualification, benchmark or mesh-generation action.");
  MFEM_VERIFY(!options.qualify || options.reference == "exact", "--qualify-backends checks the exact reference.");
  MFEM_VERIFY(!options.qualify_projected || options.reference == "projected", "--qualify-projected requires projected reference.");
  MFEM_VERIFY(argc == 1 || options.help || actions == 1,
              "Select --qualify-backends, --qualify-projected, --benchmark, --profile or --write-mapped-mesh.");
  if (options.profile)
  {
    MFEM_VERIFY(options.backend != "all", "--profile requires one explicit --backend.");
    if (!repetitions_set) { options.repetitions = 1; }
    if (!warmup_set) { options.warmup = 0; }
    if (!rounds_set) { options.rounds = 1; }
    MFEM_VERIFY(options.rounds == 1, "--profile supports one backend construction/round.");
  }
  MFEM_VERIFY(options.mesh_n >= 3 && options.mesh_n <= 512 &&
    options.repetitions >= 1 && options.repetitions <= 4096 &&
    options.warmup >= 0 && options.warmup <= 4096 && options.rounds >= 1 && options.rounds <= 8,
    "Invalid local tensor benchmark dimensions/repetitions/warmup/rounds.");
  return options;
}

std::vector<std::string> SelectedModes(const std::string& selected, const std::string& reference = "exact")
{
  if (selected != "all")
  {
    MFEM_VERIFY(reference != "projected" || !IsDense(selected), "Projected reference does not admit dense execution.");
    return {selected};
  }
  // Mapped benchmark candidates keep both the reference and WMMA arithmetic
  // bounded to shared axis factors plus one 36-entry diagonal per element.
  if (reference == "projected") { return {"simt-fp64", "wmma-factor-fp64", "wmma-factor-tf32"}; }
  return ExecutionModes();
}

class MultTimer
{
public:
  explicit MultTimer(int repetitions) : starts_(repetitions), stops_(repetitions)
  {
    for (std::size_t i = 0; i < starts_.size(); ++i)
    {
      MFEM_GPU_CHECK(cudaEventCreate(&starts_[i]));
      MFEM_GPU_CHECK(cudaEventCreate(&stops_[i]));
    }
  }
  ~MultTimer()
  {
    for (std::size_t i = 0; i < starts_.size(); ++i)
    { cudaEventDestroy(starts_[i]); cudaEventDestroy(stops_[i]); }
  }

  std::vector<double> Measure(const ::igrlab::igr_internal::ElementBlockJacobi& backend,
                              const Vector& rhs, Vector& result, int warmup)
  {
    for (int i = 0; i < warmup; ++i) { backend.Mult(rhs, result); }
    MFEM_GPU_CHECK(cudaDeviceSynchronize());
    for (std::size_t i = 0; i < starts_.size(); ++i)
    {
      MFEM_GPU_CHECK(cudaEventRecord(starts_[i]));
      backend.Mult(rhs, result);
      MFEM_GPU_CHECK(cudaEventRecord(stops_[i]));
    }
    MFEM_GPU_CHECK(cudaEventSynchronize(stops_.back()));
    MFEM_GPU_CHECK(cudaGetLastError());
    std::vector<double> timings(starts_.size());
    for (std::size_t i = 0; i < starts_.size(); ++i)
    {
      float elapsed = 0.0f;
      MFEM_GPU_CHECK(cudaEventElapsedTime(&elapsed, starts_[i], stops_[i]));
      MFEM_VERIFY(std::isfinite(elapsed) && elapsed > 0.0f, "Invalid whole-Mult CUDA event timing.");
      timings[i] = elapsed;
    }
    return timings;
  }

  std::vector<double> MeasureRefresh(::igrlab::igr_internal::ElementBlockJacobi& backend,
                                     const QuadratureFunction& mass, int warmup)
  {
    for (int i = 0; i < warmup; ++i) { backend.RefreshMassScaling(mass); }
    MFEM_GPU_CHECK(cudaDeviceSynchronize());
    for (std::size_t i = 0; i < starts_.size(); ++i)
    {
      MFEM_GPU_CHECK(cudaEventRecord(starts_[i]));
      backend.RefreshMassScaling(mass);
      MFEM_GPU_CHECK(cudaEventRecord(stops_[i]));
    }
    MFEM_GPU_CHECK(cudaEventSynchronize(stops_.back()));
    MFEM_GPU_CHECK(cudaGetLastError());
    std::vector<double> timings(starts_.size());
    for (std::size_t i = 0; i < starts_.size(); ++i)
    {
      float elapsed = 0.0f;
      MFEM_GPU_CHECK(cudaEventElapsedTime(&elapsed, starts_[i], stops_[i]));
      MFEM_VERIFY(std::isfinite(elapsed) && elapsed > 0.0f, "Invalid mass-refresh CUDA event timing.");
      timings[i] = elapsed;
    }
    return timings;
  }
private:
  std::vector<cudaEvent_t> starts_, stops_;
};

struct TimingSummary
{
  double min, median, max;
};

TimingSummary Summarize(std::vector<double> values)
{
  MFEM_VERIFY(!values.empty(), "No Mult timing samples.");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return {values.front(), values.size() % 2 ? values[middle] :
          0.5 * (values[middle - 1] + values[middle]), values.back()};
}

std::string Provenance(const char* variable)
{
  const char* value = std::getenv(variable);
  std::string token = value ? value : "unspecified";
  // Keep every output record parseable as whitespace-separated key=value.
  for (char& c : token) { if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { c = '_'; } }
  return token;
}

void Benchmark(const Options& options, const std::vector<std::string>& modes)
{
  constexpr double physical_alpha = 6.357828776041666e-7;
  TensorFixture fixture(options.mesh_n, options.mesh_n, 1.0, 1.0, physical_alpha, false, true);
  fixture.SetPhase(2);
  const int n = fixture.space.GetVSize();
  Vector rhs(n), result(n), expected(n);
  FillSignedResidual(rhs, 0);
  // One deterministic, nonconstant signed residual and heterogeneous current
  // density are reused unchanged by every execution mode and every round.
  rhs.UseDevice(true); result.UseDevice(true);
  (void)rhs.Read(); (void)result.Write();
  MultTimer timer(options.repetitions);
  std::unique_ptr<CompactDenseReference> reference;
  UnscaledOracleBudgets unscaled_budgets;
  bool oracle_action_ready = false;
  std::vector<std::vector<double>> round_medians(modes.size());
  std::vector<ErrorMetrics> errors(modes.size());
  std::vector<OracleBudget> budgets(modes.size());
  std::cout << std::setprecision(17)
            << "local_tensor_jacobi_benchmark event=configuration"
            << " source=" << Provenance("IGRLAB_BENCHMARK_SOURCE")
            << " build=" << Provenance("IGRLAB_BENCHMARK_BUILD")
            << " mesh_n=" << options.mesh_n << " elements=" << fixture.space.GetNE()
            << " p=5 d1d=6 sigma_dofs=" << n << " alpha=" << physical_alpha
            << " kappa=" << fixture.kappa
            << " mass_nq=" << fixture.mass_rule.GetNPoints()
            << " diffusion_nq=" << fixture.diffusion_rule.GetNPoints()
            << " face_nq=" << fixture.face_rule.GetNPoints()
            << " repetitions=" << options.repetitions << " warmup=" << options.warmup
            << " rounds=" << options.rounds << " mode_count=" << modes.size()
            << " scope=complete-production-Mult gather_current_S_pack_scatter_included=true"
            << " coefficient_refresh_timed=false setup_timed=false input=deterministic-signed-varying"
            << " current_density=heterogeneous profile=" << (options.profile ? "true" : "false") << '\n';
  for (int round = 0; round < options.rounds; ++round)
  {
    std::vector<int> order(modes.size()); std::iota(order.begin(), order.end(), 0);
    std::rotate(order.begin(), order.begin() + round % order.size(), order.end());
    if (round % 2) { std::reverse(order.begin(), order.end()); }
    for (int position = 0; position < int(order.size()); ++position)
    {
      const int index = order[position]; const auto& mode = modes[index];
      ExecutionSelection select(mode);
      // The exact initial blocks remain rho=1 even though every timed Mult uses
      // the same evolved current invrho. Only this setup coefficient changes.
      fixture.phase = 0;
      const auto setup_begin = std::chrono::steady_clock::now();
      auto backend = ::igrlab::igr_internal::MakeTensorElementBlockJacobi(
        fixture.space, fixture.initial_invrho);
      DenseTensor& blocks = backend->PrepareVolumeBlocks(fixture.volume);
      ::igrlab::igr_internal::AddExactFaceBlocks(fixture.info, fixture.work, -1.0, blocks);
      if (!reference && !options.profile)
      { reference = std::make_unique<CompactDenseReference>(fixture, blocks.HostRead(), true); }
      backend->FactorBlocks();
      MFEM_VERIFY(blocks.TotalSize() == 0, "Full benchmark retained per-element dense build blocks.");
      fixture.phase = 2;
      backend->RefreshMassScaling(fixture.invrho);
      MFEM_GPU_CHECK(cudaDeviceSynchronize());
      const double setup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_begin).count();
      const auto timing = Summarize(timer.Measure(*backend, rhs, result, options.warmup));
      round_medians[index].push_back(timing.median);
      std::cout << "local_tensor_jacobi_benchmark event=" << (options.profile ? "profile" : "round")
                << " round=" << round << " order=" << position << " backend=" << mode
                << " mesh_n=" << options.mesh_n << " p=5 sigma_dofs=" << n
                << " apply_ms_min=" << timing.min << " apply_ms_median=" << timing.median
                << " apply_ms_max=" << timing.max << " samples=" << options.repetitions
                << " setup_seconds=" << setup_seconds;
      if (!options.profile)
      {
        if (!oracle_action_ready)
        {
          unscaled_budgets = reference->ApplyDouble(rhs, expected);
          oracle_action_ready = true;
        }
        budgets[index] = CompactDenseReference::ScaleBudget(unscaled_budgets, mode);
        errors[index] = MeasureErrorWith<double>(result, expected);
        CheckCorrection(errors[index], budgets[index], mode);
        MFEM_VERIFY(DotDouble(rhs, result) > 0.0, "Full-size correction lost positive quadratic action.");
        std::cout << " correction_relative_l2=" << errors[index].relative_l2
                  << " correction_relative_max=" << errors[index].relative_max
                  << " roundoff_envelope_relative_l2=" << double(budgets[index].absolute_l2 / errors[index].norm_l2)
                  << " roundoff_envelope_relative_max=" << double(budgets[index].absolute_max / errors[index].norm_max)
                  << " oracle_classes=" << reference->ClassCount()
                  << " initial_h_condition_inf_max=" << reference->MaximumCondition()
                  << " oracle=compact-class-dense-fp64 oracle_action_evaluations=1 positive_quadratic=true";
      }
      else { std::cout << " oracle=not-run other_backend_Mult_calls=0"; }
      std::cout << " live_backend_instances=1 timed_diagnostic_scans=0 passed" << std::endl;
    }
  }
  if (options.profile) { return; }
  double best = std::numeric_limits<double>::infinity();
  std::string winner;
  for (int i = 0; i < int(modes.size()); ++i)
  {
    const auto timing = Summarize(round_medians[i]);
    if (timing.median < best) { best = timing.median; winner = modes[i]; }
    std::cout << "local_tensor_jacobi_benchmark event=summary backend=" << modes[i]
              << " mesh_n=" << options.mesh_n << " p=5 sigma_dofs=" << n
              << " round_count=" << round_medians[i].size()
              << " apply_ms_min=" << timing.min << " apply_ms_median=" << timing.median
              << " apply_ms_max=" << timing.max << " statistic=per-round-medians"
              << " correction_relative_l2=" << errors[i].relative_l2
              << " correction_relative_max=" << errors[i].relative_max
              << " oracle_classes=" << reference->ClassCount() << '\n';
  }
  std::cout << "local_tensor_jacobi_benchmark event=winner backend=" << winner
            << " apply_ms_median=" << best << " criterion=complete-production-Mult"
            << " physical_accuracy_claim=false" << std::endl;
}

#include "igr_local_tensor_projected_fixture.hpp"

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
  if (argc == 2 && std::strncmp(argv[1], "--reject-", 9) == 0)
  {
    mfem::Device device("cuda");
    ReferenceSelection exact_reference("exact");
    ExecutionSelection baseline("simt-fp64");
    MFEM_VERIFY(mfem::Mpi::WorldSize() == 1 && UsesOrdinaryCUDAForDGDiffusionFaces2D(),
                "Local tensor rejection fixture requires one native CUDA rank.");
    const std::string kind(argv[1] + 9);
    mfem::set_error_action(mfem::MFEM_ERROR_ABORT);
    if (kind == "projected-dense")
    {
      ReferenceSelection projected("projected"); ExecutionSelection dense("simt-dense-fp32");
      MappedTensorFixture fixture(3, 3, "curved-q2", 0.017, false);
      auto backend = fixture.Prepare(); backend->FactorBlocks();
      std::cout << "local_tensor_projected_unexpected_dense_admission\n";
      return 0;
    }
    MFEM_VERIFY(kind == "skew" || kind == "curved" || kind == "nonseparable", "Unknown rejection fixture.");
    ExpectedRejection(kind);
    return 0;
  }
  const Options options = ParseOptions(argc, argv);
  if (options.help)
  {
    std::cout << "Usage: dg_igr_local_tensor_jacobi_test [--qualify-backends [--backend MODE] | "
                 "--qualify-projected [--backend all|MODE] | "
                 "--benchmark [--mesh-n 512] [--backend all|MODE] [--repetitions 64] "
                 "[--warmup 8] [--rounds 3] [--reference exact|projected] | --profile --backend MODE "
                 "[--reference exact|projected] [--mesh-n 512] [--repetitions 1] [--warmup 0] | "
                 "--write-mapped-mesh PATH [--mesh-n 512]]\n";
    return 0;
  }
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1, "Local tensor fixture requires one rank.");
  if (!options.write_mapped_mesh.empty()) { WriteMappedMesh(options); return 0; }
  mfem::Device device("cuda");
  MFEM_VERIFY(UsesOrdinaryCUDAForDGDiffusionFaces2D(), "Local tensor fixture requires native CUDA.");
  ReferenceSelection reference(options.reference);
  if (options.qualify || options.qualify_projected || options.benchmark)
  {
    const auto modes = SelectedModes(options.backend, options.reference);
    const bool needs_wmma = std::any_of(modes.begin(), modes.end(), [](const std::string& mode)
    { return mode.find("wmma") == 0; });
    int ordinal = 0; cudaDeviceProp properties{};
    MFEM_GPU_CHECK(cudaGetDevice(&ordinal));
    MFEM_GPU_CHECK(cudaGetDeviceProperties(&properties, ordinal));
    if (needs_wmma && properties.major < 8)
    {
      std::cout << "SKIP: local tensor WMMA qualification requires SM80 or newer.\n";
      return 77;
    }
    if (options.qualify) { QualifyBackends(modes); }
    else if (options.qualify_projected) { QualifyProjected(modes); }
    else if (options.reference == "projected") { BenchmarkProjected(options, modes); }
    else { Benchmark(options, modes); }
    return 0;
  }
  ExecutionSelection baseline("simt-fp64");
  CheckCase(2); CheckCase(5);
  std::cout << "local_tensor_jacobi_test passed constant_density_exact=true "
               "heterogeneous_current_mass=true dense_surrogate=true spd=true "
               "native_ordering=true no_dense_after_setup=true fallback=none\n";
  return 0;
}
