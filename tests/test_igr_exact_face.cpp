#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
// Keep namespace directives out of this CUDA translation unit: NVCC's
// generated global launch stubs spell igrlab unqualified, and importing mfem
// also exposes its legacy mfem::igrlab extension namespace.
using mfem::Array;
using mfem::DenseMatrix;
using mfem::DG_FECollection;
using mfem::DGDiffusionIntegrator;
using mfem::Element;
using mfem::FaceQuadratureSpace;
using mfem::FaceType;
using mfem::FiniteElementSpace;
using mfem::FunctionCoefficient;
using mfem::Geometry;
using mfem::IntegrationRule;
using mfem::IntRules;
using mfem::Mesh;
using mfem::Ordering;
using mfem::ParFiniteElementSpace;
using mfem::ParGridFunction;
using mfem::ParMesh;
using mfem::Reshape;
using mfem::Vector;
using mfem::forall;
using mfem::real_t;
using ::igrlab::dg_face::AddInteriorFaceDGDiffusionApplyDevice2D;
using ::igrlab::dg_face::BuildInteriorFaceDGDiffusionApplyData2D;
using ::igrlab::dg_face::DGDiffusionFaceScaling;
using ::igrlab::dg_face::InteriorFaceDGDiffusionApplyInfo;
using ::igrlab::dg_face::InteriorFaceDGDiffusionApplyScratch;
using ::igrlab::dg_face::InteriorFaceDGDiffusionGather2D;
using ::igrlab::dg_face::InteriorFaceDGDiffusionRecord2D;
using ::igrlab::dg_face::InteriorFaceShapeDictionary2D;
using ::igrlab::dg_face::LaunchInteriorFaceDGDiffusionCoefficientUpdate2D;
using ::igrlab::dg_face::PrepareInteriorFaceShapeDictionary2D;

bool SameBits(const Vector& a, const Vector& b)
{
  return a.Size() == b.Size() &&
    (a.Size() == 0 || std::memcmp(a.HostRead(), b.HostRead(),
      static_cast<std::size_t>(a.Size()) * sizeof(real_t)) == 0);
}

void RequireFinite(const Vector& v)
{
  const real_t* data = v.HostRead();
  for (int i = 0; i < v.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(data[i]), "Exact face output was not written.");
  }
}

void CheckDictionaryCopies(const InteriorFaceDGDiffusionApplyInfo& info,
                           const InteriorFaceDGDiffusionApplyScratch& work)
{
  if (!work.shape_dictionary.Matches(info)) { return; }
  const int block = 2 * info.nq * info.dof;
  const real_t* full = work.shape.HostRead();
  const real_t* compact = work.shape_dictionary.values.HostRead();
  const int* slots = work.shape_dictionary.face_slots.HostRead();
  for (int face = 0; face < info.nf; ++face)
  {
    MFEM_VERIFY(slots[face] >= 0 && slots[face] < work.shape_dictionary.slot_count &&
                  std::memcmp(full + block * face, compact + block * slots[face],
                              std::size_t(block) * sizeof(real_t)) == 0,
                "Shape dictionary must copy every original bit pattern.");
  }
}

void CheckDictionaryAdmission()
{
  InteriorFaceDGDiffusionApplyInfo info;
  info.dof = 2; info.nq = 1; info.nf = 6;
  info.connectivity_revision = ::igrlab::dg_face::NextDGFaceConnectivityRevision();
  Vector shape(2 * info.dof * info.nq * info.nf);
  real_t* s = shape.HostWrite();
  for (int face = 0; face < info.nf; ++face)
  {
    s[4 * face] = face < 2 ? 0.0 : -0.0;
    s[4 * face + 1] = 1.0;
    s[4 * face + 2] = -1.0;
    s[4 * face + 3] = 2.0;
    if (face >= 4)
    {
      static_assert(sizeof(real_t) == sizeof(std::uint64_t), "FP64 fixture required.");
      const std::uint64_t payload = UINT64_C(0x7ff8000000000042);
      std::memcpy(s + 4 * face, &payload, sizeof(payload));
    }
  }
  InteriorFaceShapeDictionary2D dictionary;
  PrepareInteriorFaceShapeDictionary2D(info, shape, dictionary);
  MFEM_VERIFY(dictionary.Matches(info) && dictionary.slot_count == 3,
              "Exact dictionary must distinguish signed zero and copy NaN payloads.");
  const int* slots = dictionary.face_slots.HostRead();
  MFEM_VERIFY(slots[0] == slots[1] && slots[2] == slots[3] && slots[4] == slots[5] &&
                slots[0] != slots[2] && slots[0] != slots[4] && slots[2] != slots[4],
              "Only byte-identical face blocks may share a slot.");
  for (int face = 0; face < info.nf; ++face)
  {
    MFEM_VERIFY(std::memcmp(shape.HostRead() + 4 * face,
                            dictionary.values.HostRead() + 4 * slots[face],
                            4 * sizeof(real_t)) == 0, "Dictionary bytes changed.");
  }
  PrepareInteriorFaceShapeDictionary2D(info, shape, dictionary, 1);
  MFEM_VERIFY(!dictionary.Matches(info) && dictionary.values.Size() == 0 &&
                dictionary.face_slots.Size() == 0, "Budget miss must discard old geometry.");
  s = shape.HostWrite();
  for (int i = 0; i < shape.Size(); ++i) { s[i] = real_t(i + 1); }
  PrepareInteriorFaceShapeDictionary2D(info, shape, dictionary);
  MFEM_VERIFY(dictionary.values.Size() == 0 && dictionary.face_slots.Size() == 0,
              "An all-unique dictionary must use the original table.");
  info.dof = info.nq = std::numeric_limits<int>::max();
  PrepareInteriorFaceShapeDictionary2D(info, shape, dictionary);
  MFEM_VERIFY(dictionary.values.Size() == 0, "Overflowing optional dimensions must decline.");
}

// Test-only use of the unchanged one-thread/face device functor.  Compiling
// this source as CUDA makes the comparison a CUDA arithmetic oracle, rather
// than a CPU sum with potentially different multiply/add contraction.
Vector OriginalRecordAndGather(const InteriorFaceDGDiffusionApplyInfo& info,
                              const InteriorFaceDGDiffusionApplyScratch& work,
                              const Vector& x, real_t ip_sigma,
                              Vector& contribution)
{
  contribution.SetSize(2 * info.dof * info.nf);
  contribution.UseDevice(true);
  contribution = std::numeric_limits<real_t>::quiet_NaN();
  InteriorFaceDGDiffusionRecord2D record;
  record.dof = info.dof;
  record.vdim = info.vdim;
  record.nq = info.nq;
  record.nf = info.nf;
  record.ne = info.ne;
  record.ip_sigma = ip_sigma;
  record.shape = work.shape.Read();
  record.weighted_normal_dshape = work.weighted_normal_dshape.Read();
  record.penalty = work.penalty.Read();
  record.face_elements = work.face_elements.Read();
  record.element_vdof_map = work.element_vdof_map.Read();
  record.x = x.Read();
  record.contribution = contribution.Write();
  forall(info.nf, record);

  Vector result(x.Size());
  result.UseDevice(true);
  result = 0.03125; // Exercise addition to an existing volume contribution.
  InteriorFaceDGDiffusionGather2D gather;
  gather.dof = info.dof;
  gather.vdim = info.vdim;
  gather.ne = info.ne;
  gather.nf = info.nf;
  gather.element_vdof_map = work.element_vdof_map.Read();
  gather.element_face_offsets = work.element_face_offsets.Read();
  gather.element_face_ids = work.element_face_ids.Read();
  gather.element_face_sides = work.element_face_sides.Read();
  gather.contribution = contribution.Read();
  gather.y = result.ReadWrite();
  forall(info.ne * info.dof, gather);
  return result;
}

Mesh MakeMesh(int geometry)
{
  // Periodic vertex identification needs at least three cells in each
  // direction to retain distinct topological edges. Four by three also
  // covers full four-face batches; the nonperiodic meshes retain a 7-face
  // tail fixture below.
  const bool periodic = geometry == 2;
  Mesh mesh = Mesh::MakeCartesian2D(periodic ? 4 : 3,
                                    periodic ? 3 : 2, Element::QUADRILATERAL,
                                    true, 1.0, 0.7);
  if (geometry == 2)
  {
    const std::vector<Vector> shifts = {Vector({1.0, 0.0}), Vector({0.0, 0.7})};
    return Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(shifts));
  }
  if (geometry == 1)
  {
    // Positive rotations change the local face/dof orientation.  Curvature
    // then exercises nonconstant physical normals and inverse Jacobians.
    for (const int e : {0, 5})
    {
      int* v = mesh.GetElement(e)->GetVertices();
      const int first = v[0];
      v[0] = v[1]; v[1] = v[2]; v[2] = v[3]; v[3] = first;
    }
    mesh.FinalizeTopology();
    mesh.Finalize(false, false);
    mesh.SetCurvature(2, false, 2, Ordering::byVDIM);
    mesh.Transform([](const Vector& x, Vector& y)
    {
      y.SetSize(2);
      y[0] = x[0] + 0.08 * x[0] * x[1];
      y[1] = x[1] + 0.06 * x[0] * (1.0 - x[0]);
    });
  }
  return mesh;
}

void CheckCase(int order, int face_order, int basis, int geometry,
               int& cases, double& max_assembled_error)
{
  Mesh serial = MakeMesh(geometry);
  ParMesh mesh(MPI_COMM_SELF, serial);
  DG_FECollection fec(order, 2, basis);
  ParFiniteElementSpace fes(&mesh, &fec, 1, Ordering::byNODES);
  ParGridFunction x(&fes);
  x.UseDevice(true);
  Array<int> dofs;
  real_t* values = x.HostWrite();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementDofs(e, dofs);
    for (int d = 0; d < dofs.Size(); ++d)
    {
      values[FiniteElementSpace::DecodeDof(dofs[d])] =
        0.01 * std::sin(0.13 * (d + 1) * (e + 2)) + 0.002 * e;
    }
  }
  FunctionCoefficient coefficient([](const Vector& point)
  { return 0.043 / (0.9 + 0.08 * point[0] + 0.03 * point[1]); });
  const IntegrationRule& ir = IntRules.Get(Geometry::SEGMENT, face_order);
  FaceQuadratureSpace fqs(mesh, ir, FaceType::Interior);
  Vector side_q(2 * ir.GetNPoints() * fqs.GetNE());
  auto Q = Reshape(side_q.HostWrite(), 2, ir.GetNPoints(), fqs.GetNE());
  for (int face = 0; face < fqs.GetNE(); ++face)
  {
    auto* tr = mesh.GetFaceElementTransformations(fqs.GetMeshFaceIndex(face));
    MFEM_VERIFY(tr && tr->Elem1 && tr->Elem2, "Expected two face elements.");
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      tr->SetAllIntPoints(&ir.IntPoint(q));
      Q(0, q, face) = coefficient.Eval(*tr->Elem1, tr->GetElement1IntPoint());
      Q(1, q, face) = coefficient.Eval(*tr->Elem2, tr->GetElement2IntPoint());
    }
  }

  constexpr real_t kappa = 2.75;
  for (const auto scaling : {DGDiffusionFaceScaling::Arithmetic,
                             DGDiffusionFaceScaling::Max,
                             DGDiffusionFaceScaling::Sum,
                             DGDiffusionFaceScaling::HarmonicWeighted})
  {
    InteriorFaceDGDiffusionApplyScratch work;
    const auto info = BuildInteriorFaceDGDiffusionApplyData2D(
      fes, fqs, side_q, kappa, work.shape, work.weighted_normal_dshape,
      work.penalty, work.face_elements, work.element_vdof_map, scaling,
      &work.normal_dshape_base, &work.penalty_metric, &work.shape_dictionary);
    MFEM_VERIFY(info.nf == (geometry == 2 ? 24 : 7),
                "Fixture must cover whole batches and a three-face tail.");
    MFEM_VERIFY(info.dof == (order + 1) * (order + 1) &&
                info.nq == ir.GetNPoints(), "Unexpected packed dimensions.");
    CheckDictionaryCopies(info, work);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
    if (geometry == 0 && info.nq <= 16 &&
        mfem::Device::Allows(mfem::Backend::CUDA))
    {
      MFEM_VERIFY(work.shape_dictionary.Matches(info),
                  "Repeating Cartesian geometry must exercise compact CUDA S loads.");
    }
#endif
    const Vector compact_before(work.shape_dictionary.values);
    const auto dictionary_revision = work.shape_dictionary.geometry_revision;
    for (const real_t ip_sigma : {-1.0, 1.0})
    {
      // A second coefficient pack exercises the persistent geometry refresh.
      for (int refresh = 0; refresh < 2; ++refresh)
      {
        Vector current_q(side_q);
        if (refresh)
        {
          real_t* qdata = current_q.HostReadWrite();
          for (int i = 0; i < current_q.Size(); ++i)
          { qdata[i] *= 0.71 + 0.037 * (i % 7); }
          LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
            info, work, current_q, kappa, scaling);
        }
        else
        {
          LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
            info, work, side_q, kappa, scaling);
        }
        MFEM_VERIFY(work.shape_dictionary.geometry_revision == dictionary_revision &&
                      SameBits(compact_before, work.shape_dictionary.values),
                    "Current coefficients must not change the immutable shape dictionary.");
        work.face_contribution.SetSize(2 * info.dof * info.nf);
        work.face_contribution.UseDevice(true);
        work.face_contribution = std::numeric_limits<real_t>::quiet_NaN();
        Vector actual(x.Size());
        actual.UseDevice(true);
        actual = 0.03125;
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
        const bool first_compact_apply = ip_sigma == -1.0 && refresh == 0 &&
          info.nq <= 16 && work.shape_dictionary.Matches(info) &&
          mfem::Device::Allows(mfem::Backend::CUDA);
        if (first_compact_apply)
        {
          MFEM_VERIFY(!work.shape.GetMemory().DeviceIsValid(),
                      "Compact fixture must start with the full S table on host only.");
        }
#endif
        // This internal entry point also exercises the record/gather CPU
        // fallback when this source is compiled without CUDA support.
        AddInteriorFaceDGDiffusionApplyDevice2D(info, work, x, ip_sigma, actual);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
        if (first_compact_apply)
        {
          MFEM_VERIFY(!work.shape.GetMemory().DeviceIsValid() &&
                        work.shape_dictionary.values.GetMemory().DeviceIsValid() &&
                        work.shape_dictionary.face_slots.GetMemory().DeviceIsValid(),
                      "Native compact apply must read its dictionary without migrating full S.");
        }
#endif
        Vector original_contribution;
        Vector original = OriginalRecordAndGather(
          info, work, x, ip_sigma, original_contribution);
        RequireFinite(actual);
        RequireFinite(work.face_contribution);
        RequireFinite(original_contribution);
        MFEM_VERIFY(work.face_contribution.Normlinf() > 1.0e-6,
                    "Fixture must exercise nontrivial face contributions.");
        MFEM_VERIFY(SameBits(work.face_contribution, original_contribution),
                    "Exact face record changed original FP64 arithmetic: "
                    << "p=" << order << " nq=" << info.nq
                    << " geometry=" << geometry << " basis=" << basis);
        MFEM_VERIFY(SameBits(actual, original),
                    "Exact face deterministic gathered action changed bits.");

        Vector repeat(x.Size());
        repeat.UseDevice(true);
        repeat = 0.03125;
        AddInteriorFaceDGDiffusionApplyDevice2D(info, work, x, ip_sigma, repeat);
        MFEM_VERIFY(SameBits(actual, repeat), "Exact face repeat changed bits.");

        if (!refresh && scaling == DGDiffusionFaceScaling::Arithmetic)
        {
          DGDiffusionIntegrator integrator(coefficient, ip_sigma, kappa);
          integrator.SetIntegrationRule(ir);
          Vector assembled(x.Size());
          assembled = 0.0;
          real_t* out = assembled.HostReadWrite();
          const real_t* input = x.HostRead();
          DenseMatrix matrix;
          Vector local_x(2 * info.dof), local_y(2 * info.dof);
          Array<int> left, right;
          for (int face = 0; face < info.nf; ++face)
          {
            auto* tr = mesh.GetFaceElementTransformations(
              fqs.GetMeshFaceIndex(face));
            integrator.AssembleFaceMatrix(*fes.GetFE(tr->Elem1No),
                                          *fes.GetFE(tr->Elem2No), *tr, matrix);
            fes.GetElementDofs(tr->Elem1No, left);
            fes.GetElementDofs(tr->Elem2No, right);
            for (int d = 0; d < info.dof; ++d)
            {
              local_x[d] = input[FiniteElementSpace::DecodeDof(left[d])];
              local_x[info.dof + d] = input[FiniteElementSpace::DecodeDof(right[d])];
            }
            matrix.Mult(local_x, local_y);
            for (int d = 0; d < info.dof; ++d)
            {
              out[FiniteElementSpace::DecodeDof(left[d])] += local_y[d];
              out[FiniteElementSpace::DecodeDof(right[d])] += local_y[info.dof + d];
            }
          }
          RequireFinite(assembled);
          const double reference_norm = assembled.Norml2();
          MFEM_VERIFY(reference_norm > 1.0e-6,
                      "Assembled oracle must contain a nonzero face action.");
          Vector expected(assembled);
          expected += 0.03125;
          Vector delta(actual);
          delta -= expected;
          const double max_error = delta.Normlinf();
          const double relative_error = delta.Norml2() / reference_norm;
          max_assembled_error = std::max(max_assembled_error, max_error);
          MFEM_VERIFY(max_error < 5.0e-13 && relative_error < 5.0e-13,
                      "Exact face action differs from assembled MFEM: "
                      << max_error << " relative=" << relative_error);
        }
        ++cases;
      }
    }
  }
}

void CheckDictionaryGeometryRebuild()
{
  Mesh serial = MakeMesh(0);
  ParMesh mesh(MPI_COMM_SELF, serial);
  DG_FECollection fec(5, 2, mfem::BasisType::GaussLegendre);
  ParFiniteElementSpace fes(&mesh, &fec);
  IntegrationRule rule(IntRules.Get(Geometry::SEGMENT, 11));
  Vector side_q(2 * rule.GetNPoints() * 7);
  side_q = 0.07;
  Vector x(fes.GetVSize());
  x.UseDevice(true);
  real_t* input = x.HostWrite();
  for (int i = 0; i < x.Size(); ++i) { input[i] = 0.01 * std::sin(0.07 * (i + 1)); }
  InteriorFaceDGDiffusionApplyScratch work;
  auto build = [&](bool compact)
  {
    FaceQuadratureSpace fqs(mesh, rule, FaceType::Interior);
    return BuildInteriorFaceDGDiffusionApplyData2D(
      fes, fqs, side_q, 2.75, work.shape, work.weighted_normal_dshape,
      work.penalty, work.face_elements, work.element_vdof_map,
      DGDiffusionFaceScaling::Arithmetic, &work.normal_dshape_base,
      &work.penalty_metric, compact ? &work.shape_dictionary : nullptr);
  };
  auto parity = [&](const InteriorFaceDGDiffusionApplyInfo& info)
  {
    Vector actual(x.Size());
    actual.UseDevice(true);
    actual = 0.03125;
    AddInteriorFaceDGDiffusionApplyDevice2D(info, work, x, -1.0, actual);
    Vector original_contribution;
    const Vector original = OriginalRecordAndGather(info, work, x, -1.0,
                                                   original_contribution);
    RequireFinite(actual);
    MFEM_VERIFY(SameBits(actual, original) &&
                  SameBits(work.face_contribution, original_contribution),
                "Rebuilt or declined dictionary changed the original action.");
  };
  auto info = build(true);
  MFEM_VERIFY(work.shape_dictionary.Matches(info), "Rebuild fixture must admit compression.");
  const auto first_revision = info.connectivity_revision;
  const Vector old_shape(work.shape);
  parity(info);
  // Legacy vector-only builders can reuse the same allocations; an epoch miss
  // must reject the old dictionary rather than lazily accepting its pointers.
  info = build(false);
  MFEM_VERIFY(info.connectivity_revision != first_revision &&
                !work.shape_dictionary.Matches(info), "Fresh geometry must reject stale slots.");
  parity(info);
  mesh.Transform([](const Vector& point, Vector& out)
  { out.SetSize(2); out[0] = 1.03 * point[0]; out[1] = 0.97 * point[1]; });
  rule.IntPoint(0).x += 0.001;
  info = build(true);
  MFEM_VERIFY(work.shape_dictionary.Matches(info) && !SameBits(old_shape, work.shape) &&
                info.connectivity_revision != first_revision,
              "Same-size changed geometry/rule must rebuild immutable slots.");
  CheckDictionaryCopies(info, work);
  parity(info);
  PrepareInteriorFaceShapeDictionary2D(info, work.shape, work.shape_dictionary, 1);
  parity(info); // A budget decline follows the original full-S warp path.
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  MFEM_VERIFY(backend == "cpu" || backend == "cuda", "Expected cpu or cuda.");
  mfem::Device device(backend.c_str());
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1, "Exact face fixture is single rank.");
  CheckDictionaryAdmission();
  CheckDictionaryGeometryRebuild();
  int cases = 0;
  double max_assembled_error = 0.0;
  for (const int basis : {mfem::BasisType::GaussLegendre,
                          mfem::BasisType::GaussLobatto})
  {
    for (int geometry = 0; geometry < 3; ++geometry)
    { CheckCase(3, 7, basis, geometry, cases, max_assembled_error); }
  }
  // Smaller output tails and quadrature counts differing from p3/Q4.
  for (const int order : {2, 4})
  { CheckCase(order, 7, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error); }
  CheckCase(3, 9, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  // Native p4/p5/p6 quadrature exercises multiple output tiles, including
  // tiles that span the two sides and partially filled final output tiles.
  for (const int order : {4, 5, 6})
  {
    for (const int basis : {mfem::BasisType::GaussLegendre,
                            mfem::BasisType::GaussLobatto})
    { CheckCase(order, 2 * order + 1, basis, 0, cases, max_assembled_error); }
  }
  CheckCase(5, 11, mfem::BasisType::GaussLegendre, 2, cases, max_assembled_error);
  CheckCase(6, 13, mfem::BasisType::GaussLegendre, 1, cases, max_assembled_error);
  // P10 has 242 output rows: several bounded row groups and a final tail.
  for (const int basis : {mfem::BasisType::GaussLegendre,
                          mfem::BasisType::GaussLobatto})
  { CheckCase(10, 21, basis, 0, cases, max_assembled_error); }
  // Q1 tests valid side-one sources for all inactive warp lanes while P5
  // still exercises multiple output groups and a partially filled tail.
  MFEM_VERIFY(IntRules.Get(Geometry::SEGMENT, 0).GetNPoints() == 1,
              "Lower-bound shuffle fixture must contain one quadrature point.");
  CheckCase(5, 0, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  // Q16 fills all projection lanes; Q17 exercises the generic fallback.
  CheckCase(4, 31, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  CheckCase(4, 33, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  std::cout << "Exact face record/gather parity: backend=" << backend
            << " cases=" << cases << " max_assembled_error="
            << max_assembled_error << '\n';
  return 0;
}
