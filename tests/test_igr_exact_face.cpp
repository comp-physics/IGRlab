#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
#include "mfem/general/cuda.hpp"
#include <cuda_runtime.h>
#endif
#include "physics/igr/IGR.hpp"
#include "../src/numerics/dg/DGFaceQuadratureKernels.hpp"
#include "../src/physics/igr/IGR_chebyshev.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
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
using ::igrlab::dg_face::LaunchInteriorFaceDGDiffusionApply2D;
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

// Factorization changes the order of FP64 products, so compare norm-scaled
// errors rather than individual near-zero rows. This bound allows roughly
// 2e4 eps for the longest tested p10 contractions without admitting a changed
// discretization; measured maxima are printed by the test executable.
constexpr double factorization_tolerance = 5.0e-12;
struct FactorizationTestStats
{
  int prepared = 0, cuda_applications = 0;
  double max_relative_l2 = 0.0, max_relative_linf = 0.0;
} factorization_stats;

void CheckFactorizationError(const Vector& actual, const Vector& reference,
                              const char* label)
{
  MFEM_VERIFY(actual.Size() == reference.Size(), "Factorized action size changed.");
  RequireFinite(actual);
  RequireFinite(reference);
  Vector difference(actual);
  difference -= reference;
  const double l2 = difference.Norml2() / std::max(reference.Norml2(), 1.0e-300);
  const double linf = difference.Normlinf() / std::max(reference.Normlinf(), 1.0e-300);
  factorization_stats.max_relative_l2 = std::max(factorization_stats.max_relative_l2, l2);
  factorization_stats.max_relative_linf = std::max(factorization_stats.max_relative_linf, linf);
  MFEM_VERIFY(l2 <= factorization_tolerance && linf <= factorization_tolerance,
              label << " relative_l2=" << l2 << " relative_linf=" << linf);
}

// Independent scalar CPU evaluation of the stored reference derivatives and
// physical mapping factors. Production CPU dispatch intentionally stays dense.
Vector HostFactorizedAction(const InteriorFaceDGDiffusionApplyInfo& info,
                            const InteriorFaceDGDiffusionApplyScratch& work,
                            const Vector& x, real_t ip_sigma,
                            Vector& contribution)
{
  const auto& factors = work.normal_factorization;
  MFEM_VERIFY(factors.Matches(info) &&
                factors.coefficient_revision == info.connectivity_revision,
              "Host factor oracle requires current geometry and coefficients.");
  const auto DS = Reshape(factors.reference_dshape.HostRead(),
                          2, 2, info.nq, info.dof, factors.slot_count);
  const auto NH = Reshape(factors.normal_geometry.HostRead(), 2, 2, info.nq, info.nf);
  const auto Q = Reshape(work.side_coefficient.HostRead(), 2, info.nq, info.nf);
  const auto PM = Reshape(work.penalty_metric.HostRead(), 2, info.nq, info.nf);
  const int* slots = factors.face_slots.HostRead();
  Vector normals(work.weighted_normal_dshape.Size());
  auto N = Reshape(normals.HostWrite(), 2, info.nq, info.dof, info.nf);
  for (int face = 0; face < info.nf; ++face)
  {
    for (int q = 0; q < info.nq; ++q)
    {
      const real_t metric0 = Q(0, q, face) * PM(0, q, face);
      const real_t metric1 = Q(1, q, face) * PM(1, q, face);
      for (int side = 0; side < 2; ++side)
      {
        real_t scale = Q(side, q, face);
        if (factors.coefficient_scaling == DGDiffusionFaceScaling::HarmonicWeighted &&
            metric0 + metric1 > 0.0)
        { scale *= 2.0 * (side == 0 ? metric1 : metric0) / (metric0 + metric1); }
        for (int d = 0; d < info.dof; ++d)
        {
          N(side, q, d, face) = scale *
            (DS(0, side, q, d, slots[face]) * NH(0, side, q, face) +
             DS(1, side, q, d, slots[face]) * NH(1, side, q, face));
        }
      }
    }
  }
  CheckFactorizationError(normals, work.weighted_normal_dshape,
                          "Factorized normal reconstruction changed the physical mapping");
  contribution.SetSize(2 * info.dof * info.nf);
  InteriorFaceDGDiffusionRecord2D record;
  record.dof = info.dof; record.vdim = info.vdim;
  record.nq = info.nq; record.nf = info.nf; record.ne = info.ne;
  record.ip_sigma = ip_sigma;
  record.shape = work.shape.HostRead();
  record.weighted_normal_dshape = normals.HostRead();
  record.penalty = work.penalty.HostRead();
  record.face_elements = work.face_elements.HostRead();
  record.element_vdof_map = work.element_vdof_map.HostRead();
  record.x = x.HostRead();
  record.contribution = contribution.HostWrite();
  for (int face = 0; face < info.nf; ++face) { record(face); }
  Vector result(x.Size());
  result = 0.03125;
  InteriorFaceDGDiffusionGather2D gather;
  gather.dof = info.dof; gather.vdim = info.vdim;
  gather.ne = info.ne; gather.nf = info.nf;
  gather.element_vdof_map = work.element_vdof_map.HostRead();
  gather.element_face_offsets = work.element_face_offsets.HostRead();
  gather.element_face_ids = work.element_face_ids.HostRead();
  gather.element_face_sides = work.element_face_sides.HostRead();
  gather.contribution = contribution.HostRead();
  gather.y = result.HostReadWrite();
  for (int row = 0; row < info.ne * info.dof; ++row) { gather(row); }
  return result;
}

void CheckGatherToBaseRounding()
{
  // The combined production path admits ordinary CUDA only. Host arithmetic
  // may select a different quiet-NaN operand payload; CPU production fallback
  // equivalence is covered by CheckProductionIdentityCopies instead.
  if (!::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA())
  { return; }
  // Compare the original two device launches with the combined gather. In
  // particular, do not reassociate the base value into the incidence sum.
  constexpr int n = 8, nf = 2 * (n - 1);
  Array<int> map(n), offsets(n + 1), ids(nf), sides(nf);
  for (int i = 0; i < n; ++i)
  {
    map[i] = i;
    offsets[i] = 2 * i;
  }
  offsets[n] = nf; // The final row has no incident interior faces.
  for (int i = 0; i < nf; ++i) { ids[i] = i; sides[i] = 0; }
  real_t nan1, nan2;
#if defined(MFEM_USE_SINGLE)
  const std::uint32_t payload1 = UINT32_C(0x7fc00042);
  const std::uint32_t payload2 = UINT32_C(0x7fc00017);
#else
  const std::uint64_t payload1 = UINT64_C(0x7ff8000000000042);
  const std::uint64_t payload2 = UINT64_C(0x7ff8000000000017);
#endif
  static_assert(sizeof(real_t) == sizeof(payload1), "Matching NaN payload precision required.");
  std::memcpy(&nan1, &payload1, sizeof(real_t));
  std::memcpy(&nan2, &payload2, sizeof(real_t));
  const real_t tiny = std::numeric_limits<real_t>::denorm_min();
  const real_t inf = std::numeric_limits<real_t>::infinity();
  const real_t bases[n] = {-0.0, tiny, 1.0, 7.0, 3.0, nan1, nan1, -0.0};
  const real_t terms[n - 1][2] = {
    {-0.0, -0.0}, {-tiny, 0.0}, {1.0e16, -1.0e16},
    {nan2, 0.0}, {inf, -inf}, {2.0, 3.0}, {nan2, 0.0}};
  Vector contribution(2 * nf), scratch(n), expected(n), actual(n);
  contribution.UseDevice(true); scratch.UseDevice(true);
  expected.UseDevice(true); actual.UseDevice(true);
  real_t* c = contribution.HostWrite();
  std::fill(c, c + contribution.Size(), 0.0);
  for (int row = 0; row < n - 1; ++row)
  {
    for (int incidence = 0; incidence < 2; ++incidence)
    { c[2 * (2 * row + incidence)] = terms[row][incidence]; }
  }
  std::copy(bases, bases + n, expected.HostWrite());
  std::copy(bases, bases + n, actual.HostWrite());
  scratch = 0.0;
  InteriorFaceDGDiffusionGather2D gather;
  gather.dof = gather.vdim = 1; gather.ne = n; gather.nf = nf;
  gather.element_vdof_map = map.Read();
  gather.element_face_offsets = offsets.Read();
  gather.element_face_ids = ids.Read();
  gather.element_face_sides = sides.Read();
  gather.contribution = contribution.Read();
  gather.y = scratch.ReadWrite();
  forall(n, gather);
  expected += scratch;
  gather.y = actual.ReadWrite();
  gather.add_to_base = true;
  forall(n, gather);
  MFEM_VERIFY(SameBits(expected, actual),
              "Combined gather changed base + (+0 + face sum) rounding.");
  MFEM_VERIFY(actual.HostRead()[2] == 1.0,
              "The base value must not enter the incidence reduction.");
}

Mesh MakeMesh(int geometry)
{
  // Periodic vertex identification needs at least three cells in each
  // direction to retain distinct topological edges. Four by three also
  // covers full four-face batches; the nonperiodic meshes retain a 7-face
  // tail fixture below.
  const bool periodic = geometry == 2;
  const bool factor_fixture = geometry >= 3;
  Mesh mesh = Mesh::MakeCartesian2D(factor_fixture ? 4 : (periodic ? 4 : 3),
                                    factor_fixture ? 4 : (periodic ? 3 : 2),
                                    Element::QUADRILATERAL,
                                    true, 1.0, factor_fixture ? 1.0 : 0.7);
  if (geometry == 3)
  {
    mesh.Transform([](const Vector& x, Vector& y)
    { y.SetSize(2); y[0] = 1.1 * x[0] + 0.2 * x[1];
      y[1] = 0.15 * x[0] + 0.9 * x[1]; });
  }
  if (geometry == 2)
  {
    const std::vector<Vector> shifts = {Vector({1.0, 0.0}), Vector({0.0, 0.7})};
    return Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(shifts));
  }
  if (geometry == 1 || geometry == 4)
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
      &work.normal_dshape_base, &work.penalty_metric, &work.shape_dictionary,
      &work.normal_factorization);
    MFEM_VERIFY(info.nf == (geometry >= 2 ? 24 : 7),
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
    const Vector reference_before(work.normal_factorization.reference_dshape);
    const Vector geometry_before(work.normal_factorization.normal_geometry);
    const auto dictionary_revision = work.shape_dictionary.geometry_revision;
    if (geometry >= 3)
    {
      MFEM_VERIFY(work.normal_factorization.Matches(info) ==
                    (scaling != DGDiffusionFaceScaling::HarmonicWeighted),
                  "Skew/nonaffine p5 fixtures must admit supported factorized scalings only.");
    }
    if (work.normal_factorization.Matches(info)) { ++factorization_stats.prepared; }
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
        MFEM_VERIFY(SameBits(reference_before, work.normal_factorization.reference_dshape) &&
                      SameBits(geometry_before, work.normal_factorization.normal_geometry),
                    "Current qdata must not modify immutable reference or mapping factors.");
        const auto factor_revision = work.normal_factorization.coefficient_revision;
        work.normal_factorization.coefficient_revision = 0; // Original dense-path oracle.
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

        work.normal_factorization.coefficient_revision = factor_revision;
        Vector factorized(x.Size());
        factorized.UseDevice(true);
        factorized = 0.03125;
        work.weighted_normal_dshape.HostReadWrite();
        LaunchInteriorFaceDGDiffusionApply2D(info, work, x, ip_sigma, factorized);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
        if (::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA())
        {
          const bool admitted = work.normal_factorization.Matches(info) && info.nq <= 16 &&
            factor_revision == info.connectivity_revision &&
            scaling != DGDiffusionFaceScaling::HarmonicWeighted;
          MFEM_VERIFY(work.weighted_normal_dshape.GetMemory().DeviceIsValid() != admitted,
                      "Public apply must select factorized storage only when admitted.");
          if (admitted) { ++factorization_stats.cuda_applications; }
        }
#endif
        CheckFactorizationError(factorized, original, "Factorized public face action");
        CheckFactorizationError(work.face_contribution, original_contribution,
                                "Factorized face contributions");
        if (work.normal_factorization.Matches(info))
        {
          Vector host_contribution;
          const Vector host_action = HostFactorizedAction(info, work, x, ip_sigma,
                                                          host_contribution);
          CheckFactorizationError(host_action, original, "Scalar CPU factorized action");
          CheckFactorizationError(host_contribution, original_contribution,
                                  "Scalar CPU factorized contributions");
        }

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
          CheckFactorizationError(factorized, expected, "Factorized assembled-MFEM oracle");
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
  auto build = [&](bool compact_shape, bool factorize = true,
                    std::size_t budget = 256u * 1024u * 1024u)
  {
    FaceQuadratureSpace fqs(mesh, rule, FaceType::Interior);
    return BuildInteriorFaceDGDiffusionApplyData2D(
      fes, fqs, side_q, 2.75, work.shape, work.weighted_normal_dshape,
      work.penalty, work.face_elements, work.element_vdof_map,
      DGDiffusionFaceScaling::Arithmetic, &work.normal_dshape_base,
      &work.penalty_metric, compact_shape ? &work.shape_dictionary : nullptr,
      factorize ? &work.normal_factorization : nullptr, budget);
  };
  auto parity = [&](const InteriorFaceDGDiffusionApplyInfo& info, bool factorized)
  {
    Vector actual(x.Size());
    actual.UseDevice(true);
    actual = 0.03125;
    work.weighted_normal_dshape.HostReadWrite();
    LaunchInteriorFaceDGDiffusionApply2D(info, work, x, -1.0, actual);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
    if (::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA())
    {
      MFEM_VERIFY(work.weighted_normal_dshape.GetMemory().DeviceIsValid() != factorized,
                  "Factorized dispatch must reject stale or unavailable geometry/coefficient data.");
    }
#else
    (void)factorized;
#endif
    Vector original_contribution;
    const Vector original = OriginalRecordAndGather(info, work, x, -1.0,
                                                   original_contribution);
    CheckFactorizationError(actual, original, "Rebuilt factorized face action");
    CheckFactorizationError(work.face_contribution, original_contribution,
                            "Rebuilt factorized face records");
    return actual;
  };
  auto refresh = [&](const InteriorFaceDGDiffusionApplyInfo& info)
  {
    LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
      info, work, side_q, 2.75, DGDiffusionFaceScaling::Arithmetic);
  };
  auto info = build(true);
  MFEM_VERIFY(work.shape_dictionary.Matches(info) && work.normal_factorization.Matches(info),
              "Rebuild fixture must admit shape and reference-derivative storage.");
  MFEM_VERIFY(work.normal_factorization.coefficient_revision == 0,
              "An initial geometry build must retain dense normal dispatch.");
  const auto first_revision = info.connectivity_revision;
  const Vector old_shape(work.shape), old_geometry(work.normal_factorization.normal_geometry);
  parity(info, false);
  // Reference-derivative slots are independent of shape slots. Renumber the
  // former without changing any block to expose accidental shape-slot reuse.
  auto& factors = work.normal_factorization;
  MFEM_VERIFY(factors.slot_count > 1, "Reference slot test requires distinct orientations.");
  const int block = 4 * info.nq * info.dof;
  const Vector original_reference(factors.reference_dshape);
  real_t* reordered = factors.reference_dshape.HostWrite();
  for (int slot = 0; slot < factors.slot_count; ++slot)
  {
    std::copy(original_reference.HostRead() + block * slot,
              original_reference.HostRead() + block * (slot + 1),
              reordered + block * (factors.slot_count - 1 - slot));
  }
  int* slots = factors.face_slots.HostReadWrite();
  for (int face = 0; face < info.nf; ++face)
  { slots[face] = factors.slot_count - 1 - slots[face]; }
  refresh(info);
  const Vector old_action = parity(info, true);
  side_q *= 1.19;
  refresh(info);
  const Vector new_action = parity(info, true);
  MFEM_VERIFY(!SameBits(old_action, new_action) && SameBits(old_shape, work.shape) &&
                SameBits(old_geometry, work.normal_factorization.normal_geometry),
              "Coefficient refresh must change action without changing geometric factors.");

  Vector harmonic_q(side_q);
  real_t* harmonic_values = harmonic_q.HostReadWrite();
  for (int i = 0; i < harmonic_q.Size(); ++i)
  { harmonic_values[i] *= (i % 2) ? 1.31 : 0.73; }
  LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
    info, work, harmonic_q, 2.75, DGDiffusionFaceScaling::HarmonicWeighted);
  MFEM_VERIFY(work.normal_factorization.Matches(info),
              "Harmonic fallback must be tested with otherwise admitted geometric factors.");
  const Vector harmonic_action = parity(info, false);
  Vector harmonic_contribution;
  const Vector host_harmonic = HostFactorizedAction(info, work, x, -1.0,
                                                   harmonic_contribution);
  CheckFactorizationError(host_harmonic, harmonic_action, "Host harmonic geometric reconstruction");
  refresh(info);

  // Legacy vector-only rebuilds can reuse allocations but never old epochs.
  info = build(false, false);
  MFEM_VERIFY(info.connectivity_revision != first_revision &&
                !work.shape_dictionary.Matches(info) && !work.normal_factorization.Matches(info),
              "Same-size rebuild must reject stale shape and derivative slots.");
  parity(info, false);
  refresh(info);
  parity(info, false); // New qdata cannot make stale geometry valid.
  mesh.Transform([](const Vector& point, Vector& out)
  { out.SetSize(2); out[0] = 1.03 * point[0] + 0.07 * point[1];
    out[1] = 0.97 * point[1]; });
  rule.IntPoint(0).x += 0.001;
  info = build(true);
  MFEM_VERIFY(work.shape_dictionary.Matches(info) && work.normal_factorization.Matches(info) &&
                !SameBits(old_shape, work.shape) &&
                !SameBits(old_geometry, work.normal_factorization.normal_geometry) &&
                info.connectivity_revision != first_revision,
              "Changed geometry/rule must rebuild the reference and mapping factors.");
  CheckDictionaryCopies(info, work);
  parity(info, false);
  refresh(info);
  parity(info, true);
  info = build(true, false);
  refresh(info);
  MFEM_VERIFY(work.shape_dictionary.Matches(info) && !work.normal_factorization.Matches(info),
              "Missing factor storage must leave the original shape optimization available.");
  parity(info, false);
  info = build(true, true, 1);
  refresh(info);
  MFEM_VERIFY(!work.normal_factorization.Matches(info) &&
                work.normal_factorization.reference_dshape.Size() == 0 &&
                work.normal_factorization.normal_geometry.Size() == 0 &&
                work.normal_factorization.face_slots.Size() == 0,
              "Factor memory-budget decline must discard old storage.");
  parity(info, false);
  PrepareInteriorFaceShapeDictionary2D(info, work.shape, work.shape_dictionary, 1);
  parity(info, false);
}

class ScopedEnvironment
{
public:
  ScopedEnvironment(const char* name, const char* value) : name_(name)
  {
    if (const char* old = std::getenv(name))
    { had_value_ = true; value_ = old; }
    setenv(name, value, 1);
  }
  ~ScopedEnvironment()
  {
    if (had_value_) { setenv(name_.c_str(), value_.c_str(), 1); }
    else { unsetenv(name_.c_str()); }
  }
private:
  std::string name_, value_;
  bool had_value_ = false;
};

void CheckProductionFactorizedDispatch()
{
  Mesh serial = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL,
                                      true, 1.0, 1.0);
  ParMesh mesh(MPI_COMM_SELF, serial);
  DG_FECollection fec(5, 2, mfem::BasisType::GaussLegendre);
  ParFiniteElementSpace scalar_fes(&mesh, &fec, 1, Ordering::byNODES);
  ParFiniteElementSpace state_fes(&mesh, &fec, 4, Ordering::byNODES);
  ParGridFunction state(&state_fes);
  state.UseDevice(true);
  Vector input(scalar_fes.GetVSize());
  input.UseDevice(true);
  real_t* x = input.HostWrite();
  for (int d = 0; d < input.Size(); ++d)
  { x[d] = 0.01 * std::sin(0.13 * (d + 1)); }
  Vector dense_actions[2];
  for (const bool compact : {false, true})
  {
    // This switch is sampled when the production face operator is built.
    ScopedEnvironment selection("DG_IGR_IGR_FACTORIZED_FACE_NORMALS",
                                compact ? "true" : "false");
    ScopedEnvironment canonical_selection("IGRLAB_IGR_FACTORIZED_FACE_NORMALS",
                                          compact ? "true" : "false");
    ::igrlab::IGRSolver solver(scalar_fes, state_fes, 2, 4, 0.07);
    solver.SetGradBackend(::igrlab::IGRGradBackend::BrokenDG);
    solver.SetIPKind(::igrlab::IGRIPKind::SIPG);
    solver.SetExactDGFaceOperator(true);
    solver.SetSolveMode(::igrlab::IGRSolveMode::SmootherSweeps);
    solver.SetSweeps(0, 0);
    for (int refresh = 0; refresh < 2; ++refresh)
    {
      const int ndofs = state_fes.GetNDofs();
      real_t* u = state.HostWrite();
      for (int d = 0; d < ndofs; ++d)
      {
        const real_t rho = (refresh ? 1.19 : 1.0) *
          (1.0 + 0.03 * std::sin(0.17 * (d + 1)));
        u[d] = rho;
        u[d + ndofs] = 0.04 * rho;
        u[d + 2 * ndofs] = -0.02 * rho;
        u[d + 3 * ndofs] = 2.5 + 0.001 * rho;
      }
      solver.UpdateFromState(state, false);
      const auto* face = solver.GetLinearFaceOperator();
      MFEM_VERIFY(face && solver.GetLastExactFaceEffective() &&
                    solver.GetLastExactFaceSharedFaceCount() == 0,
                  "Factorized-normal switch fixture must exercise local exact faces.");
      Vector actual(input.Size());
      actual.UseDevice(true);
      face->Mult(input, actual);
      RequireFinite(actual);
      MFEM_VERIFY(actual.Normlinf() > 1.0e-6,
                  "Factorized-normal production fixture needs a nonzero face action.");
      if (!compact) { dense_actions[refresh] = actual; }
      else
      {
        CheckFactorizationError(actual, dense_actions[refresh],
                                "Production factorized face dispatch");
      }
    }
  }
  MFEM_VERIFY(!SameBits(dense_actions[0], dense_actions[1]),
              "Production factorized-normal comparison must evolve its coefficients.");
}

// A test-only virtual-access witness distinguishes the production direct
// read from the old identity-copy path, without adding a production counter.
class ObservedInput final : public Vector
{
public:
  using Vector::Vector;
  using Vector::operator=;
  const real_t* Read(bool on_device = true) const override
  { ++reads; return Vector::Read(on_device); }
  const real_t* HostRead() const override
  { ++reads; return Vector::HostRead(); }
  mutable int reads = 0;
};

// Inspect allocation addresses without transferring intermediate results to
// the host or inserting a synchronization between polynomial stages.
class ChebyshevOperatorWitness final : public mfem::Operator
{
public:
  explicit ChebyshevOperatorWitness(const mfem::Operator& op)
    : mfem::Operator(op.Height(), op.Width()), op_(op) { }
  void Mult(const Vector& x, Vector& y) const override
  {
    input_addresses.push_back(x.Read());
    op_.Mult(x, y);
  }
  mfem::MemoryClass GetMemoryClass() const override
  { return op_.GetMemoryClass(); }
  mutable std::vector<const real_t*> input_addresses;
private:
  const mfem::Operator& op_;
};

class ChebyshevDiagonalOracle final : public mfem::Operator
{
public:
  explicit ChebyshevDiagonalOracle(int n) : mfem::Operator(n) { }
  void Mult(const Vector& x, Vector& y) const override
  {
    const auto X = x.Read();
    const auto Y = y.Write();
    const real_t scale = scale_;
    forall(height, [=] MFEM_HOST_DEVICE (int i)
    { Y[i] = (scale + 0.0625 * (i % 3)) * X[i]; });
  }
  real_t scale_ = 0.75;
};

void CheckChebyshevOracle(const mfem::Operator& op, const Vector& diagonal,
                          const Array<int>& essential, real_t eigenvalue,
                          bool exceptional = false)
{
  if (mfem::Device::Allows(mfem::Backend::CUDA_MASK) &&
      !::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA())
  { return; } // Unsupported accelerator/local-CXX routes test factory fallback below.
  const int n = diagonal.Size();
  for (int degree = 1; degree <= 5; ++degree)
  {
    ChebyshevOperatorWitness native_op(op), owned_op(op);
    mfem::OperatorChebyshevSmoother native(
      native_op, diagonal, essential, degree, eigenvalue);
    ::igrlab::igr_internal::OwnedScratchChebyshev owned(
      owned_op, diagonal, essential, degree, eigenvalue);
    Vector x(n), native_y(n), owned_y(n);
    x.UseDevice(true); native_y.UseDevice(true); owned_y.UseDevice(true);
    std::vector<Vector> native_results, owned_results;
    native_results.reserve(4); owned_results.reserve(4);
    for (int call = 0; call < 4; ++call)
    {
      native_results.emplace_back(n);
      owned_results.emplace_back(n);
      native_results.back().UseDevice(true);
      owned_results.back().UseDevice(true);
    }
    for (int call = 0; call < 4; ++call)
    {
      // Alternate current device and host authority. Preserve each output
      // without a host read so repeated applications exercise queued reuse.
      auto fill = [=] MFEM_HOST_DEVICE (int i)
      {
        if (call == 0) { return real_t(i % 2 ? 0.0 : -0.0); }
        return real_t((i % 2 ? -1.0 : 1.0) *
                      (0.03125 * (i + 1) + 0.0078125 * call));
      };
      if (call % 2)
      {
        real_t* data = x.HostWrite();
        for (int i = 0; i < n; ++i) { data[i] = fill(i); }
      }
      else
      {
        real_t* data = x.Write();
        forall(n, [=] MFEM_HOST_DEVICE (int i) { data[i] = fill(i); });
      }
      if (exceptional && n >= 6)
      {
        real_t* data = x.HostReadWrite();
        data[0] = std::numeric_limits<real_t>::infinity();
        data[1] = -std::numeric_limits<real_t>::infinity();
        data[2] = std::numeric_limits<real_t>::quiet_NaN();
        data[3] = std::numeric_limits<real_t>::max();
        data[4] = std::numeric_limits<real_t>::denorm_min();
        data[5] = -std::numeric_limits<real_t>::denorm_min();
      }
      auto& native_result = native_results[call];
      auto& owned_result = owned_results[call];
      native_result = 19.0;
      owned_result = -23.0;
      if (call % 2)
      { native.MultTranspose(x, native_result); owned.MultTranspose(x, owned_result); }
      else
      { native.Mult(x, native_result); owned.Mult(x, owned_result); }
    }
    MFEM_VERIFY(native_op.input_addresses.size() == std::size_t(4 * (degree - 1)) &&
                  owned_op.input_addresses.size() == native_op.input_addresses.size(),
                "Chebyshev changed the prescribed operator call count.");
    if (degree > 1 && n > 0)
    {
      const auto& addresses = owned_op.input_addresses;
      MFEM_VERIFY(std::find_if(addresses.begin() + 1, addresses.end(),
                      [&](const real_t* p) { return p != addresses.front(); }) !=
                      addresses.end(),
                  "Owned Chebyshev must actually alternate scratch allocations.");
      MFEM_VERIFY(std::all_of(native_op.input_addresses.begin(),
                              native_op.input_addresses.end(),
                      [&](const real_t* p)
                      { return p == native_op.input_addresses.front(); }),
                  "Pinned MFEM oracle must retain its original copied residual.");
    }
    for (int call = 0; call < 4; ++call)
    {
      MFEM_VERIFY(SameBits(native_results[call], owned_results[call]),
                  "Owned Chebyshev changed pinned MFEM output bits: degree="
                  << degree << " call=" << call << " exceptional=" << exceptional);
      if (!exceptional && std::isfinite(eigenvalue) && eigenvalue > 0)
      { RequireFinite(owned_results[call]); }
    }

    // The initial residual copy must precede clearing y for all caller aliases.
    Vector native_in_place(x), owned_in_place(x);
    native.Mult(native_in_place, native_in_place);
    owned.Mult(owned_in_place, owned_in_place);
    MFEM_VERIFY(SameBits(native_in_place, owned_in_place),
                "Owned Chebyshev changed in-place input/output semantics.");
    Vector native_backing(n + 1), owned_backing(n + 1);
    native_backing.UseDevice(true); owned_backing.UseDevice(true);
    const real_t* input = x.HostRead();
    real_t* native_data = native_backing.HostWrite();
    real_t* owned_data = owned_backing.HostWrite();
    if (n > 0)
    {
      std::copy(input, input + n, native_data);
      std::copy(input, input + n, owned_data);
    }
    native_data[n] = owned_data[n] = 29.0;
    Vector native_view, owned_view, native_output, owned_output;
    native_view.MakeRef(native_backing, 0, n);
    native_output.MakeRef(native_backing, 1, n);
    owned_view.MakeRef(owned_backing, 0, n);
    owned_output.MakeRef(owned_backing, 1, n);
    native.Mult(native_view, native_output);
    owned.Mult(owned_view, owned_output);
    MFEM_VERIFY(SameBits(native_output, owned_output),
                "Owned Chebyshev changed overlapping caller-view semantics.");

    // Unknown subsequent operator bindings take the original copy sequence.
    ChebyshevOperatorWitness rebound(op);
    owned.SetOperator(rebound);
    native.SetOperator(rebound);
    owned.Mult(x, owned_y);
    native.Mult(x, native_y);
    MFEM_VERIFY(SameBits(native_y, owned_y),
                "Chebyshev operator rebinding changed original behavior.");
    if (degree > 2 && n > 0)
    {
      MFEM_VERIFY(rebound.input_addresses[0] == rebound.input_addresses[1],
                  "Unknown rebound operator must retain the copied-residual path.");
    }
  }
}

void CheckChebyshevOwnedScratch()
{
  for (int n : {0, 1, 7, 255, 257})
  {
    ChebyshevDiagonalOracle op(n);
    Vector diagonal(n);
    diagonal.UseDevice(true);
    real_t* d = diagonal.HostWrite();
    for (int i = 0; i < n; ++i) { d[i] = 1.0 + 0.125 * (i % 5); }
    Array<int> essential;
    if (n > 1)
    {
      essential.SetSize(1);
      essential[0] = n / 2;
      d[n / 2] = 0.0; // Original essential inverse is overwritten with one.
    }
    CheckChebyshevOracle(op, diagonal, essential, 2.125);
    // A new coefficient/diagonal setup uses the current values rather than
    // retaining an old action or preconditioner result.
    op.scale_ = 1.125;
    d = diagonal.HostWrite();
    for (int i = 0; i < n; ++i) { d[i] = 1.25 + 0.0625 * (i % 7); }
    CheckChebyshevOracle(op, diagonal, essential, 3.375);
    if (n == 7)
    {
      CheckChebyshevOracle(op, diagonal, essential, 2.125, true);
      CheckChebyshevOracle(op, diagonal, essential, 0.0, true);
      CheckChebyshevOracle(op, diagonal, essential, -2.0, true);
    }
    auto fallback = ::igrlab::igr_internal::MakeIGRChebyshevSmoother(
      op, diagonal, essential, 3, 2.125, false);
    MFEM_VERIFY(dynamic_cast<mfem::OperatorChebyshevSmoother*>(fallback.get()),
                "Unsupported production route must retain original MFEM.");
    if (!::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA())
    {
      auto cpu_fallback = ::igrlab::igr_internal::MakeIGRChebyshevSmoother(
        op, diagonal, essential, 3, 2.125, true);
      MFEM_VERIFY(dynamic_cast<mfem::OperatorChebyshevSmoother*>(cpu_fallback.get()),
                  "Unsupported backend must retain MFEM even if an eligibility flag is supplied.");
    }
  }
}

void CheckProductionIdentityCopies()
{
  ScopedEnvironment direct_system("IGRLAB_IGR_DIRECT_IDENTITY_SYSTEM", "1");
  for (const int order : {3, 5})
  {
    for (const bool timed : {false, true})
    {
      ScopedEnvironment timing("IGRLAB_IGR_EXACT_FACE_TIMING", timed ? "1" : "0");
      Mesh serial = MakeMesh(1);
      ParMesh mesh(MPI_COMM_SELF, serial);
      DG_FECollection fec(order, 2, mfem::BasisType::GaussLegendre);
      ParFiniteElementSpace scalar_fes(&mesh, &fec, 1, Ordering::byNODES);
      ParFiniteElementSpace state_fes(&mesh, &fec, 4, Ordering::byNODES);
      MFEM_VERIFY(scalar_fes.Conforming() && scalar_fes.GetNRanks() == 1 &&
                    mfem::IsIdentityProlongation(scalar_fes.GetProlongationMatrix()) &&
                    mfem::IsIdentityProlongation(
                      scalar_fes.GetRestrictionTransposeOperator()) &&
                    !mfem::IsIdentityProlongation(scalar_fes.GetRestrictionOperator()),
                  "Production fixture needs the actual wrapped identity restriction.");
      ParGridFunction state(&state_fes);
      state.UseDevice(true);
      ::igrlab::IGRSolver solver(scalar_fes, state_fes, 2, 4, 0.07);
      solver.SetGradBackend(::igrlab::IGRGradBackend::BrokenDG);
      solver.SetIPKind(::igrlab::IGRIPKind::SIPG);
      solver.SetExactDGFaceOperator(true);
      // Assemble the real production operator without an unrelated Krylov
      // solve: the comparisons below apply its public existing components.
      solver.SetSolveMode(::igrlab::IGRSolveMode::SmootherSweeps);
      solver.SetSweeps(0, 0);
      Vector previous_face;
      for (int refresh = 0; refresh < 2; ++refresh)
      {
        const int ndofs = state_fes.GetNDofs();
        real_t* u = state.HostWrite();
        for (int d = 0; d < ndofs; ++d)
        {
          const real_t rho = (refresh ? 1.19 : 1.0) *
            (1.0 + 0.03 * std::sin(0.17 * (d + 1)));
          u[d] = rho;
          u[d + ndofs] = 0.04 * rho;
          u[d + 2 * ndofs] = -0.02 * rho;
          u[d + 3 * ndofs] = 2.5 + 0.001 * rho;
        }
        solver.UpdateFromState(state, false);
        const auto* face = solver.GetLinearFaceOperator();
        const auto* base = solver.GetLinearBaseOperator();
        const auto* full = solver.GetLinearOperator();
        MFEM_VERIFY(face && base && full && solver.GetLastExactFaceEffective() &&
                      solver.GetLastExactFaceSharedFaceCount() == 0,
                    "Fixture must use the real no-shared production exact face.");
        ObservedInput input(face->Width());
        input.UseDevice(true);
        real_t* x = input.Write();
        const int n = input.Size();
        forall(n, [=] MFEM_HOST_DEVICE(int i)
        { x[i] = i == 0 ? -0.0 : 0.01 * std::sin(0.13 * (i + 1)); });
        Vector direct(face->Height());
        direct.UseDevice(true);
        MFEM_VERIFY(input.OwnsData() && direct.OwnsData(),
                    "Direct fixture requires distinct owned allocations.");
        if (mfem::Device::Allows(mfem::Backend::CUDA))
        { MFEM_VERIFY(!input.GetMemory().HostIsValid(), "Fresh device input required."); }
        face->Mult(input, direct);
        MFEM_VERIFY(input.reads > 0,
                    "Production identity path must read the current input directly.");

        // A nonowning input forces the unchanged original staging path. Both
        // paths use the same original face kernels/current coefficients.
        ObservedInput view;
        view.MakeRef(input, 0, n);
        Vector original(face->Height());
        original.UseDevice(true);
        face->Mult(view, original);
        MFEM_VERIFY(view.reads == 0 && SameBits(direct, original),
                    "Direct action must match the original alias fallback bitwise.");
        RequireFinite(direct);
        MFEM_VERIFY(direct.Normlinf() > 1.0e-6,
                    "Production fixture needs a nontrivial face action.");

        // Preserve the completed original base + face addition. The combined
        // CUDA gather must not accumulate individual incidences into the base.
        Vector expected(full->Height()), actual(full->Height());
        expected.UseDevice(true); actual.UseDevice(true);
        base->Mult(input, expected);
        expected += original;
        full->Mult(input, actual);
        MFEM_VERIFY(SameBits(expected, actual), "Production sum changed base + face bits.");
        Vector full_fallback(full->Height());
        full_fallback.UseDevice(true);
        full->Mult(view, full_fallback);
        MFEM_VERIFY(SameBits(expected, full_fallback),
                    "Nonowning sum input must preserve the original scratch path.");
        Vector output_backing(full->Height() + 1), output_view;
        output_backing.UseDevice(true);
        output_view.MakeRef(output_backing, 1, full->Height());
        full->Mult(input, output_view);
        MFEM_VERIFY(SameBits(expected, output_view),
                    "Nonowning sum output must preserve the original scratch path.");

        Vector diagonal(full->Height());
        full->AssembleDiagonal(diagonal);
        Array<int> essential;
        CheckChebyshevOracle(*full, diagonal, essential, 4.125);
        const bool admitted_cuda_layout =
          ::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA() &&
          solver.GetLastDirectIdentitySystem() && scalar_fes.Conforming() &&
          scalar_fes.GetNRanks() == 1 &&
          solver.GetLastExactFaceSharedFaceCount() == 0;
        if (::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA() &&
            scalar_fes.GetNRanks() == 1)
        {
          MFEM_VERIFY(admitted_cuda_layout,
                      "CUDA fixture must exercise the admitted direct-identity face sum.");
        }
        auto selected = ::igrlab::igr_internal::MakeIGRChebyshevSmoother(
          *full, diagonal, essential, 3, 4.125, admitted_cuda_layout);
        const bool owned_selected = dynamic_cast<
          ::igrlab::igr_internal::OwnedScratchChebyshev*>(selected.get()) != nullptr;
        MFEM_VERIFY(owned_selected ==
                      (admitted_cuda_layout &&
                       ::igrlab::igr_internal::OwnedScratchChebyshev::HasCUDAKernels()),
                    "Qualified CUDA production must select owned Chebyshev scratch.");
        mfem::OperatorChebyshevSmoother reference(*full, diagonal, essential, 3, 4.125);
        Vector selected_y(full->Height()), reference_y(full->Height());
        selected->Mult(input, selected_y);
        reference.Mult(input, reference_y);
        MFEM_VERIFY(SameBits(selected_y, reference_y),
                    "Production Chebyshev selection changed the original full operator action.");
        if (!admitted_cuda_layout)
        {
          MFEM_VERIFY(dynamic_cast<mfem::OperatorChebyshevSmoother*>(selected.get()),
                      "Non-admitted production layout must retain MFEM.");
        }

        ObservedInput in_place(n);
        in_place.UseDevice(true);
        in_place = static_cast<const Vector&>(input);
        face->Mult(in_place, in_place);
        MFEM_VERIFY(in_place.reads == 0 && SameBits(in_place, original),
                    "In-place action must retain the original staging fallback.");

        Vector backing(n + 1);
        backing.UseDevice(true);
        const real_t* original_input = input.HostRead();
        real_t* backing_input = backing.HostWrite();
        std::copy(original_input, original_input + n, backing_input);
        backing_input[n] = 0.0;
        ObservedInput overlap_input;
        overlap_input.MakeRef(backing, 0, n);
        Vector overlap_output;
        overlap_output.MakeRef(backing, 1, n);
        face->Mult(overlap_input, overlap_output);
        MFEM_VERIFY(overlap_input.reads == 0 && SameBits(overlap_output, original),
                    "Partially overlapping views must retain the original staging fallback.");
        if (refresh)
        { MFEM_VERIFY(!SameBits(previous_face, original), "Coefficient refresh must change action."); }
        previous_face = original;
      }
    }
  }
}
// Opt-in fixed-workload record diagnostic; timings exclude setup, gather and solves.
void BenchmarkFaceRecords512(const std::string& mesh_path)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  MFEM_VERIFY(::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA(),
              "The large face-record benchmark requires ordinary native CUDA.");
  constexpr int n = 512, warmups = 4, repeats = 64;
  // Match production -rs 0 -rp 4 exactly, including mesh-file element and
  // face ordering; direct Cartesian construction has a different packing.
  Mesh serial(mesh_path);
  ParMesh mesh(MPI_COMM_SELF, serial);
  for (int level = 0; level < 4; ++level) { mesh.UniformRefinement(); }
  MFEM_VERIFY(mesh.Dimension() == 2 && mesh.SpaceDimension() == 2 &&
                mesh.GetNE() == n * n,
              "The supplied Sedov mesh plus four parallel refinements must have 512-square cells.");
  real_t lower[2] = {1.0, 1.0}, upper[2] = {0.0, 0.0};
  for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
  {
    const real_t* coordinate = mesh.GetVertex(vertex);
    for (int d = 0; d < 2; ++d)
    {
      lower[d] = std::min(lower[d], coordinate[d]);
      upper[d] = std::max(upper[d], coordinate[d]);
    }
  }
  MFEM_VERIFY(std::abs(lower[0]) < 1.0e-13 && std::abs(lower[1]) < 1.0e-13 &&
                std::abs(upper[0] - 1.0) < 1.0e-13 &&
                std::abs(upper[1] - 1.0) < 1.0e-13,
              "The large face benchmark requires the production unit-square domain.");
  DG_FECollection fec(5, 2, mfem::BasisType::GaussLegendre);
  ParFiniteElementSpace fes(&mesh, &fec, 1, Ordering::byNODES);
  const IntegrationRule& rule = IntRules.Get(Geometry::SEGMENT, 11);
  FaceQuadratureSpace fqs(mesh, rule, FaceType::Interior);
  Vector side_q(2 * rule.GetNPoints() * fqs.GetNE()), x(fes.GetVSize());
  side_q.UseDevice(true);
  x.UseDevice(true);
  real_t* q = side_q.HostWrite();
  for (int i = 0; i < side_q.Size(); ++i)
  { q[i] = 0.043 / (0.9 + 0.003 * (i % 31) + 0.002 * ((i / 2) % 17)); }
  real_t* values = x.HostWrite();
  for (int i = 0; i < x.Size(); ++i)
  { values[i] = 0.01 * std::sin(0.013 * (i + 1)) + 0.002 * (i % 7); }

  InteriorFaceDGDiffusionApplyScratch work;
  const auto info = BuildInteriorFaceDGDiffusionApplyData2D(
    fes, fqs, side_q, 2.75, work.shape, work.weighted_normal_dshape,
    work.penalty, work.face_elements, work.element_vdof_map,
    DGDiffusionFaceScaling::Arithmetic, &work.normal_dshape_base,
    &work.penalty_metric, &work.shape_dictionary, &work.normal_factorization);
  MFEM_VERIFY(info.dof == 36 && info.nq == 6 && info.ne == n * n &&
                info.nf == 2 * n * (n - 1),
              "The face benchmark must retain the prescribed 512-square p5/Q6 case.");
  LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
    info, work, side_q, 2.75, DGDiffusionFaceScaling::Arithmetic);
  const bool compact_shape = work.shape_dictionary.Matches(info);
  const auto& factors = work.normal_factorization;
  const bool factorized = factors.Matches(info);
  MFEM_VERIFY(factors.coefficient_revision == info.connectivity_revision &&
                factors.coefficient_scaling == DGDiffusionFaceScaling::Arithmetic &&
                work.side_coefficient.Size() == 2 * info.nq * info.nf,
              "The face benchmark must use refreshed current coefficients.");
  std::cout << std::setprecision(17)
            << "ExactFaceBenchmark512 cells=" << info.ne << " scalar_dofs=" << x.Size()
            << " interior_faces=" << info.nf << " dof=" << info.dof << " nq=" << info.nq
            << " shape_dictionary_admitted=" << compact_shape
            << " shape_slots=" << work.shape_dictionary.slot_count
            << " factorization_admitted=" << factorized
            << " reference_slots=" << factors.slot_count
            << " dense_shape_bytes=" << std::size_t(work.shape.Size()) * sizeof(real_t)
            << " dense_geometry_normal_bytes=" << std::size_t(work.normal_dshape_base.Size()) * sizeof(real_t)
            << " dense_weighted_normal_bytes=" << std::size_t(work.weighted_normal_dshape.Size()) * sizeof(real_t)
            << " factor_reference_bytes=" << std::size_t(factors.reference_dshape.Size()) * sizeof(real_t)
            << " factor_geometry_bytes=" << std::size_t(factors.normal_geometry.Size()) * sizeof(real_t)
            << " factor_slot_bytes=" << std::size_t(factors.face_slots.Size()) * sizeof(int)
            << " mesh_path=" << std::quoted(mesh_path) << " serial_refinements=0 parallel_refinements=4"
            << " warmups=" << warmups << " repeats=" << repeats
            << " scope=record_kernel_only" << std::endl;

  MFEM_VERIFY(factorized,
              "The prescribed large case must admit factorization before timing a candidate.");
  work.face_contribution.SetSize(2 * info.dof * info.nf);
  work.face_contribution.UseDevice(true);
  InteriorFaceDGDiffusionRecord2D record;
  record.dof = info.dof; record.vdim = info.vdim;
  record.nq = info.nq; record.nf = info.nf; record.ne = info.ne;
  record.ip_sigma = -1.0;
  record.shape = compact_shape ? work.shape_dictionary.values.Read() : work.shape.Read();
  record.weighted_normal_dshape = work.weighted_normal_dshape.Read();
  record.penalty = work.penalty.Read();
  record.face_elements = work.face_elements.Read();
  record.element_vdof_map = work.element_vdof_map.Read();
  record.x = x.Read();
  record.contribution = work.face_contribution.Write();
  const int* shape_slots = compact_shape ? work.shape_dictionary.face_slots.Read() : nullptr;
  const int shape_count = compact_shape ? work.shape_dictionary.slot_count : info.nf;
  using ::igrlab::dg_face::InteriorFaceDGDiffusionRecord2DWarp;
  const auto dense = InteriorFaceDGDiffusionRecord2DWarp<false>{
    record, shape_slots, shape_count};
  mfem::forall_2D_batch(info.nf, 32, 1, 4, dense);
  MFEM_GPU_CHECK(cudaGetLastError());
  MFEM_GPU_CHECK(cudaDeviceSynchronize());
  Vector reference(work.face_contribution);
  RequireFinite(reference);
  MFEM_VERIFY(reference.Normlinf() > 1.0e-6,
              "Large record benchmark must exercise a nontrivial face action.");

  cudaEvent_t start, stop;
  MFEM_GPU_CHECK(cudaEventCreate(&start));
  MFEM_GPU_CHECK(cudaEventCreate(&stop));
  auto measure = [&](const char* name, auto kernel)
  {
    // Reestablish device-write authority after the preceding host comparison.
    kernel.record.contribution = work.face_contribution.Write();
    for (int i = 0; i < warmups; ++i)
    { mfem::forall_2D_batch(info.nf, 32, 1, 4, kernel); }
    MFEM_GPU_CHECK(cudaGetLastError());
    MFEM_GPU_CHECK(cudaDeviceSynchronize());
    MFEM_GPU_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i)
    { mfem::forall_2D_batch(info.nf, 32, 1, 4, kernel); }
    MFEM_GPU_CHECK(cudaEventRecord(stop));
    MFEM_GPU_CHECK(cudaEventSynchronize(stop));
    MFEM_GPU_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0f;
    MFEM_GPU_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    Vector difference(work.face_contribution);
    difference -= reference;
    const double relative_l2 = difference.Norml2() / reference.Norml2();
    const double relative_linf = difference.Normlinf() / reference.Normlinf();
    std::cout << "ExactFaceBenchmark512 variant=" << name
              << " total_ms=" << elapsed_ms
              << " mean_record_ms=" << double(elapsed_ms) / repeats
              << " relative_l2=" << relative_l2
              << " relative_linf=" << relative_linf
              << " bitwise_dense_reference=" << SameBits(work.face_contribution, reference)
              << std::endl;
    MFEM_VERIFY(std::isfinite(relative_l2) && std::isfinite(relative_linf) &&
                  relative_l2 <= factorization_tolerance && relative_linf <= factorization_tolerance,
                "Large face-record factorization exceeds its FP64 rounding budget: " << name);
  };
  measure("original_dense", dense);
  measure("reference_factorized", InteriorFaceDGDiffusionRecord2DWarp<true>{
    record, shape_slots, shape_count, factors.reference_dshape.Read(),
    factors.normal_geometry.Read(), factors.face_slots.Read(), factors.slot_count,
    work.side_coefficient.Read()});
  MFEM_GPU_CHECK(cudaEventDestroy(start));
  MFEM_GPU_CHECK(cudaEventDestroy(stop));
#else
  (void)mesh_path;
  MFEM_ABORT("The large face-record benchmark must be compiled as native CUDA.");
#endif
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
  const bool benchmark = argc == 4 && std::string(argv[2]) == "--benchmark512";
  MFEM_VERIFY(argc <= 2 || benchmark,
              "Expected [cpu|cuda] or cuda --benchmark512 /path/to/sedov_taylor.msh.");
  if (benchmark)
  {
    MFEM_VERIFY(backend == "cuda", "--benchmark512 requires the CUDA backend.");
    BenchmarkFaceRecords512(argv[3]);
    return 0;
  }
  CheckDictionaryAdmission();
  CheckDictionaryGeometryRebuild();
  CheckProductionFactorizedDispatch();
  CheckProductionIdentityCopies();
  CheckGatherToBaseRounding();
  CheckChebyshevOwnedScratch();
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
  // Native p4/p5/p6 quadrature covers side-spanning 64-row groups and partial
  // output groups. P5 has a full first group and an eight-row tail.
  for (const int order : {4, 5, 6})
  {
    for (const int basis : {mfem::BasisType::GaussLegendre,
                            mfem::BasisType::GaussLobatto})
    { CheckCase(order, 2 * order + 1, basis, 0, cases, max_assembled_error); }
  }
  CheckCase(5, 11, mfem::BasisType::GaussLegendre, 1, cases, max_assembled_error);
  CheckCase(5, 11, mfem::BasisType::GaussLegendre, 2, cases, max_assembled_error);
  CheckCase(5, 11, mfem::BasisType::GaussLegendre, 3, cases, max_assembled_error);
  CheckCase(5, 11, mfem::BasisType::GaussLobatto, 4, cases, max_assembled_error);
  CheckCase(6, 13, mfem::BasisType::GaussLegendre, 1, cases, max_assembled_error);
  // P7 has 128 output rows: exactly two full 64-row groups.
  CheckCase(7, 15, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  // P10 has 242 output rows: several bounded row groups and a final tail.
  for (const int basis : {mfem::BasisType::GaussLegendre,
                          mfem::BasisType::GaussLobatto})
  { CheckCase(10, 21, basis, 0, cases, max_assembled_error); }
  // Q1 tests valid side-one sources for all inactive warp lanes while P5
  // exercises a partially filled second 64-row output group.
  MFEM_VERIFY(IntRules.Get(Geometry::SEGMENT, 0).GetNPoints() == 1,
              "Lower-bound shuffle fixture must contain one quadrature point.");
  CheckCase(5, 0, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  // Q16 fills all projection lanes; Q17 exercises the generic fallback.
  CheckCase(4, 31, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  // Fill all projection lanes while also exercising a partial output group.
  CheckCase(5, 31, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  CheckCase(4, 33, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  CheckCase(5, 33, mfem::BasisType::GaussLegendre, 0, cases, max_assembled_error);
  std::cout << std::setprecision(17) << "Exact face record/gather parity: backend=" << backend
            << " cases=" << cases << " max_assembled_error="
            << max_assembled_error
            << " factor_geometries=" << factorization_stats.prepared
            << " factor_cuda_applications=" << factorization_stats.cuda_applications
            << " factor_max_relative_l2=" << factorization_stats.max_relative_l2
            << " factor_max_relative_linf=" << factorization_stats.max_relative_linf << '\n';
  return 0;
}
