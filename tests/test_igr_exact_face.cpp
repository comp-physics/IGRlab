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
#include <sstream>
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

// Validity flags describe the authoritative copy, not whether a stale device
// allocation still exists. Inspect MFEM's existing test/debug inventory without
// requesting a device pointer (which would allocate the storage being tested).
std::string DeviceAllocationInventory()
{
  std::ostringstream out;
  mfem::mm.PrintPtrs(out);
  return out.str();
}

bool HasDeviceAllocation(const std::string& inventory, const void* host)
{
  if (!host) { return false; }
  std::ostringstream key;
  key << "\nkey " << host << ", ";
  const auto begin = inventory.find(key.str());
  if (begin == std::string::npos) { return false; } // Unregistered host-only data.
  const auto end = inventory.find('\n', begin + 1);
  const auto device = inventory.find(", d_ptr ", begin);
  MFEM_VERIFY(device != std::string::npos &&
                (end == std::string::npos || device < end),
              "Unexpected MFEM device-allocation inventory format.");
  void* pointer = nullptr;
  std::istringstream value(inventory.substr(device + 8));
  value >> pointer;
  MFEM_VERIFY(!value.fail(), "Could not inspect MFEM's existing device allocation.");
  return pointer != nullptr;
}

template <typename Buffer>
std::size_t DeviceResidentBytes(const std::string& inventory, const Buffer& buffer)
{
  // HostRead does not create a device allocation or change device validity.
  // Call only outside timed regions and after read-authority witnesses.
  return buffer.Size() && HasDeviceAllocation(inventory, buffer.HostRead())
    ? std::size_t(buffer.Capacity()) * sizeof(*buffer.HostRead()) : 0u;
}

std::size_t FaceScratchDeviceResidentBytes(
  const InteriorFaceDGDiffusionApplyScratch& work, const std::string& inventory)
{
  const auto& shape = work.shape_dictionary;
  const auto& factor = work.normal_factorization;
  std::size_t bytes = 0;
  const Vector* vectors[] = {
    &work.shape, &shape.values, &factor.reference_dshape, &factor.normal_geometry,
    &factor.tensor_endpoint, &factor.tensor_tangent, &work.weighted_normal_dshape,
    &work.penalty, &work.normal_dshape_base, &work.penalty_metric,
    &work.side_coefficient, &work.face_contribution};
  for (const auto* vector : vectors) { bytes += DeviceResidentBytes(inventory, *vector); }
  const Array<int>* arrays[] = {
    &shape.face_slots, &factor.face_slots, &factor.tensor_axes,
    &factor.tensor_face_slots, &factor.tensor_lex_to_native, &work.face_elements,
    &work.element_vdof_map, &work.element_face_offsets, &work.element_face_ids,
    &work.element_face_sides};
  for (const auto* array : arrays) { bytes += DeviceResidentBytes(inventory, *array); }
  return bytes;
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
  // Independent dense fixtures have not necessarily used the public apply,
  // which normally prepares this gather's element-to-face incidence data.
  ::igrlab::dg_face::EnsureInteriorFaceElementIncidence2D(info, work);
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
  int tensor_packs = 0, tensor_normal_axes = 0, reversed_tensor_sides = 0;
  int nonidentity_tensor_maps = 0;
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

// Reconstruct full shape and physical normal tables from the cached 1D data
// on the CPU, independently of the candidate's contraction schedule. Also
// prove that its local-index map and axes describe the actual MFEM elements.
void CheckTensorFacePack(const InteriorFaceDGDiffusionApplyInfo& info,
                         const InteriorFaceDGDiffusionApplyScratch& work,
                         const ParFiniteElementSpace& fes, ParMesh& mesh,
                         const FaceQuadratureSpace& fqs)
{
  const auto& pack = work.normal_factorization;
  if (!pack.TensorMatches(info)) { return; }
  const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(fes.GetFE(0));
  MFEM_VERIFY(tensor && info.dof == 36 && info.nq == 6,
              "Tensor-face fixture must use the admitted scalar p5/Q6 tensor element.");
  const auto& native_map = tensor->GetDofMap();
  MFEM_VERIFY(native_map.Size() == 0 || native_map.Size() == 36,
              "Scalar p5 tensor DOF map must be identity or a complete permutation.");
  const int* map = pack.tensor_lex_to_native.HostRead();
  bool used[36] = {};
  bool nonidentity = false;
  for (int lex = 0; lex < 36; ++lex)
  {
    const int native = native_map.Size() ? native_map[lex] : lex;
    MFEM_VERIFY(native >= 0 && native < 36 && !used[native] && map[lex] == native,
                "Tensor-face cache must preserve the actual lexical/native DOF permutation.");
    used[native] = true;
    nonidentity = nonidentity || native != lex;
  }
  factorization_stats.nonidentity_tensor_maps += nonidentity ? 1 : 0;
  ++factorization_stats.tensor_packs;
  const int* slots = pack.tensor_face_slots.HostRead();
  const auto A = Reshape(pack.tensor_axes.HostRead(), 2, pack.tensor_slot_count);
  const auto E = Reshape(pack.tensor_endpoint.HostRead(), 2, 6, 2, pack.tensor_slot_count);
  const auto T = Reshape(pack.tensor_tangent.HostRead(), 2, 6, 6, 2, pack.tensor_slot_count);
  const auto G = Reshape(pack.normal_geometry.HostRead(), 2, 2, info.nq, info.nf);
  Vector shape(work.shape.Size()), normal(work.normal_dshape_base.Size());
  auto S = Reshape(shape.HostWrite(), 2, info.nq, info.dof, info.nf);
  auto N = Reshape(normal.HostWrite(), 2, info.nq, info.dof, info.nf);
  for (int face = 0; face < info.nf; ++face)
  {
    const int slot = slots[face];
    MFEM_VERIFY(slot >= 0 && slot < pack.tensor_slot_count,
                "Tensor-face slot must address the independent 1D pack.");
    const int mesh_face = fqs.GetMeshFaceIndex(face);
    const auto face_info = mesh.GetFaceInformation(mesh_face);
    auto* transformation = mesh.GetFaceElementTransformations(mesh_face);
    const auto& rule = fqs.GetIntRule(face);
    real_t first_tangent[2];
    transformation->SetAllIntPoints(&rule.IntPoint(0));
    for (int side = 0; side < 2; ++side)
    {
      const int axis = ::igrlab::dg_face::ReferenceNormalComponent2D(
        face_info.element[side].local_face_id);
      MFEM_VERIFY(A(side, slot) == axis,
                  "Tensor-face normal axis must match the actual local face.");
      factorization_stats.tensor_normal_axes |= 1 << axis;
      const auto& ip = side ? transformation->GetElement2IntPoint()
                            : transformation->GetElement1IntPoint();
      first_tangent[side] = axis == 0 ? ip.y : ip.x;
    }
    transformation->SetAllIntPoints(&rule.IntPoint(info.nq - 1));
    for (int side = 0; side < 2; ++side)
    {
      const int axis = A(side, slot);
      const auto& ip = side ? transformation->GetElement2IntPoint()
                            : transformation->GetElement1IntPoint();
      const real_t last = axis == 0 ? ip.y : ip.x;
      if (last < first_tangent[side]) { ++factorization_stats.reversed_tensor_sides; }
      for (int q = 0; q < info.nq; ++q)
      {
        for (int j = 0; j < 6; ++j)
        {
          for (int i = 0; i < 6; ++i)
          {
            const int native = map[i + 6 * j];
            const int ni = axis == 0 ? i : j;
            const int ti = axis == 0 ? j : i;
            const real_t value = E(0, ni, side, slot) * T(0, ti, q, side, slot);
            const real_t dn = E(1, ni, side, slot) * T(0, ti, q, side, slot);
            const real_t dt = E(0, ni, side, slot) * T(1, ti, q, side, slot);
            S(side, q, native, face) = value;
            N(side, q, native, face) = axis == 0
              ? dn * G(0, side, q, face) + dt * G(1, side, q, face)
              : dt * G(0, side, q, face) + dn * G(1, side, q, face);
          }
        }
      }
    }
  }
  CheckFactorizationError(shape, work.shape, "Independent CPU tensor-product shape reconstruction");
  CheckFactorizationError(normal, work.normal_dshape_base,
                          "Independent CPU tensor-product physical normal reconstruction");
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
  // Reference blocks contain both derivative components and need more reuse
  // than scalar shape blocks. Retain the small tail fixtures, but give the
  // mandatory skew/curved factorization cases enough repeated face patterns.
  Mesh mesh = Mesh::MakeCartesian2D(factor_fixture ? 8 : (periodic ? 4 : 3),
                                    factor_fixture ? 8 : (periodic ? 3 : 2),
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
    MFEM_VERIFY(info.nf == fqs.GetNE(), "Single-rank fixture must pack every interior face.");
    MFEM_VERIFY(geometry >= 3 || info.nf == (geometry == 2 ? 24 : 7),
                "Original fixtures must retain whole batches and a three-face tail.");
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
    const Vector endpoint_before(work.normal_factorization.tensor_endpoint);
    const Vector tangent_before(work.normal_factorization.tensor_tangent);
    const auto dictionary_revision = work.shape_dictionary.geometry_revision;
    if (geometry >= 3)
    {
      MFEM_VERIFY(work.normal_factorization.TensorMatches(info) ==
                    (scaling != DGDiffusionFaceScaling::HarmonicWeighted),
                  "Skew/nonaffine p5 fixtures must admit supported tensor-face scalings only.");
    }
    if (work.normal_factorization.Matches(info)) { ++factorization_stats.prepared; }
    CheckTensorFacePack(info, work, fes, mesh, fqs);
    // Norm checks above may migrate the full shape table; restore host-only
    // authority for the existing dense-path dictionary-read witness.
    work.shape.HostReadWrite();
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
                      SameBits(geometry_before, work.normal_factorization.normal_geometry) &&
                      SameBits(endpoint_before, work.normal_factorization.tensor_endpoint) &&
                      SameBits(tangent_before, work.normal_factorization.tensor_tangent),
                    "Current qdata must not modify immutable reference, 1D or mapping factors.");
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
        work.shape.HostReadWrite();
        if (work.shape_dictionary.values.Size())
        { work.shape_dictionary.values.HostReadWrite(); }
        LaunchInteriorFaceDGDiffusionApply2D(info, work, x, ip_sigma, factorized);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
        if (::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA())
        {
          const bool admitted = work.normal_factorization.TensorMatches(info) &&
            factor_revision == info.connectivity_revision &&
            scaling != DGDiffusionFaceScaling::HarmonicWeighted;
          MFEM_VERIFY(work.weighted_normal_dshape.GetMemory().DeviceIsValid() != admitted,
                      "Public apply must select factorized storage only when admitted.");
          if (admitted)
          {
            MFEM_VERIFY(!work.shape.GetMemory().DeviceIsValid() &&
                          !work.shape_dictionary.values.GetMemory().DeviceIsValid(),
                        "Tensor-face dispatch must avoid both dense and cached full shape tables.");
            ++factorization_stats.cuda_applications;
          }
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
  // Seven faces may legitimately decline the two-component reference table.
  // Repetition makes this an admission fixture without relaxing its budget.
  Mesh serial = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL,
                                      true, 1.0, 1.0);
  ParMesh mesh(MPI_COMM_SELF, serial);
  DG_FECollection fec(5, 2, mfem::BasisType::GaussLegendre);
  ParFiniteElementSpace fes(&mesh, &fec);
  IntegrationRule rule(IntRules.Get(Geometry::SEGMENT, 11));
  FaceQuadratureSpace initial_fqs(mesh, rule, FaceType::Interior);
  Vector side_q(2 * rule.GetNPoints() * initial_fqs.GetNE());
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
  MFEM_VERIFY(work.shape_dictionary.Matches(info) && work.normal_factorization.TensorMatches(info),
              "Repeated rebuild fixture must admit shape and reference-derivative storage: "
              << "faces=" << info.nf << " shape_slots=" << work.shape_dictionary.slot_count
              << " reference_slots=" << work.normal_factorization.slot_count);
  MFEM_VERIFY(work.normal_factorization.coefficient_revision == 0,
              "An initial geometry build must retain dense normal dispatch.");
  CheckTensorFacePack(info, work, fes, mesh, initial_fqs);
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
  // Give the 1D packs a different independent numbering as well. Neither
  // shape slots nor the older reference-derivative slots may index these.
  const int tensor_slots = factors.tensor_slot_count;
  MFEM_VERIFY(tensor_slots > 1, "1D slot test requires distinct local face orientations.");
  const Vector original_endpoint(factors.tensor_endpoint), original_tangent(factors.tensor_tangent);
  const std::vector<int> original_axes(factors.tensor_axes.HostRead(),
                                       factors.tensor_axes.HostRead() + factors.tensor_axes.Size());
  real_t* endpoint = factors.tensor_endpoint.HostWrite();
  real_t* tangent = factors.tensor_tangent.HostWrite();
  int* axes = factors.tensor_axes.HostWrite();
  constexpr int endpoint_block = 2 * 6 * 2, tangent_block = 2 * 6 * 6 * 2;
  for (int slot = 0; slot < tensor_slots; ++slot)
  {
    const int shifted = (slot + 1) % tensor_slots;
    std::copy(original_endpoint.HostRead() + endpoint_block * slot,
              original_endpoint.HostRead() + endpoint_block * (slot + 1),
              endpoint + endpoint_block * shifted);
    std::copy(original_tangent.HostRead() + tangent_block * slot,
              original_tangent.HostRead() + tangent_block * (slot + 1),
              tangent + tangent_block * shifted);
    axes[2 * shifted] = original_axes[2 * slot];
    axes[2 * shifted + 1] = original_axes[2 * slot + 1];
  }
  int* tensor_face_slots = factors.tensor_face_slots.HostReadWrite();
  for (int face = 0; face < info.nf; ++face)
  { tensor_face_slots[face] = (tensor_face_slots[face] + 1) % tensor_slots; }
  CheckTensorFacePack(info, work, fes, mesh, initial_fqs);
  refresh(info);
  const Vector old_action = parity(info, true);
  side_q *= 1.19;
  refresh(info);
  const Vector new_action = parity(info, true);
  const auto tensor_revision = factors.tensor_geometry_revision;
  factors.tensor_geometry_revision = 0;
  MFEM_VERIFY(factors.Matches(info) && !factors.TensorMatches(info),
              "Tensor-only rejection must be tested while the reference cache remains valid.");
  parity(info, false);
  factors.tensor_geometry_revision = tensor_revision;
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
  MFEM_VERIFY(work.shape_dictionary.Matches(info) && work.normal_factorization.TensorMatches(info) &&
                !SameBits(old_shape, work.shape) &&
                !SameBits(old_geometry, work.normal_factorization.normal_geometry) &&
                info.connectivity_revision != first_revision,
              "Changed geometry/rule must rebuild the reference and mapping factors.");
  CheckDictionaryCopies(info, work);
  FaceQuadratureSpace changed_fqs(mesh, rule, FaceType::Interior);
  CheckTensorFacePack(info, work, fes, mesh, changed_fqs);
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
                work.normal_factorization.face_slots.Size() == 0 &&
                !work.normal_factorization.TensorMatches(info) &&
                work.normal_factorization.tensor_endpoint.Size() == 0 &&
                work.normal_factorization.tensor_tangent.Size() == 0 &&
                work.normal_factorization.tensor_face_slots.Size() == 0 &&
                work.normal_factorization.tensor_lex_to_native.Size() == 0,
              "Factor memory-budget decline must discard old reference and tensor storage.");
  parity(info, false);
  PrepareInteriorFaceShapeDictionary2D(info, work.shape, work.shape_dictionary, 1);
  parity(info, false);
}

// Discontinuous, spatially varying coefficients keep the two traces distinct;
// changing the stage also changes their relative weights rather than merely
// multiplying the whole face operator by one scalar.
class EvolvingFaceCoefficient final : public mfem::Coefficient
{
public:
  real_t Eval(mfem::ElementTransformation& transformation,
              const mfem::IntegrationPoint& ip) override
  {
    Vector point(2);
    transformation.Transform(ip, point);
    return 0.043 * (1.0 + 0.071 * stage +
      0.027 * std::sin(0.19 * (transformation.ElementNo + 1) * (stage + 1))) /
      (0.9 + 0.04 * point[0] + 0.03 * point[1] +
       0.003 * (transformation.ElementNo % 13));
  }
  int stage = 0;
};

void FillEvolvingFaceQdata(ParMesh& mesh, const FaceQuadratureSpace& fqs,
                          EvolvingFaceCoefficient& coefficient, Vector& side_q)
{
  const int nq = fqs.GetIntRule(0).GetNPoints();
  side_q.SetSize(2 * nq * fqs.GetNE());
  side_q.UseDevice(true);
  auto Q = Reshape(side_q.HostWrite(), 2, nq, fqs.GetNE());
  for (int face = 0; face < fqs.GetNE(); ++face)
  {
    auto* transformation = mesh.GetFaceElementTransformations(fqs.GetMeshFaceIndex(face));
    const auto& rule = fqs.GetIntRule(face);
    for (int q = 0; q < nq; ++q)
    {
      transformation->SetAllIntPoints(&rule.IntPoint(q));
      Q(0, q, face) = coefficient.Eval(*transformation->Elem1,
                                      transformation->GetElement1IntPoint());
      Q(1, q, face) = coefficient.Eval(*transformation->Elem2,
                                      transformation->GetElement2IntPoint());
    }
  }
}

void CheckAssembledFaceDiagonal(ParFiniteElementSpace& fes, ParMesh& mesh,
                                const FaceQuadratureSpace& fqs,
                                mfem::Coefficient& coefficient,
                                real_t ip_sigma, const Vector& actual)
{
  DGDiffusionIntegrator integrator(coefficient, ip_sigma, 2.75);
  Vector expected(actual.Size());
  expected = 0.125;
  real_t* diagonal = expected.HostReadWrite();
  DenseMatrix matrix;
  Array<int> left, right;
  for (int face = 0; face < fqs.GetNE(); ++face)
  {
    integrator.SetIntegrationRule(fqs.GetIntRule(face));
    auto* transformation = mesh.GetFaceElementTransformations(fqs.GetMeshFaceIndex(face));
    integrator.AssembleFaceMatrix(*fes.GetFE(transformation->Elem1No),
                                  *fes.GetFE(transformation->Elem2No),
                                  *transformation, matrix);
    fes.GetElementDofs(transformation->Elem1No, left);
    fes.GetElementDofs(transformation->Elem2No, right);
    for (int d = 0; d < left.Size(); ++d)
    {
      diagonal[FiniteElementSpace::DecodeDof(left[d])] += matrix(d, d);
      diagonal[FiniteElementSpace::DecodeDof(right[d])] +=
        matrix(left.Size() + d, left.Size() + d);
    }
  }
  CheckFactorizationError(actual, expected, "Compact face diagonal versus assembled MFEM");
}

void CheckCompactFaceQdata()
{
  Mesh serial = MakeMesh(4); // Rotated, curved elements and reversed tangents.
  ParMesh mesh(MPI_COMM_SELF, serial);
  DG_FECollection fec(5, 2, mfem::BasisType::GaussLobatto);
  ParFiniteElementSpace fes(&mesh, &fec);
  IntegrationRule rule(IntRules.Get(Geometry::SEGMENT, 11));
  FaceQuadratureSpace fqs(mesh, rule, FaceType::Interior);
  EvolvingFaceCoefficient coefficient;
  Vector side_q, x(fes.GetVSize());
  x.UseDevice(true);
  real_t* input = x.HostWrite();
  for (int i = 0; i < x.Size(); ++i)
  { input[i] = 0.01 * std::sin(0.071 * (i + 1)) + 0.001 * (i % 11); }
  bool native_cuda = false;
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  native_cuda = ::igrlab::igr_internal::OwnedScratchChebyshev::UsesOrdinaryCUDA();
#endif
  int compact_refreshes = 0, dense_fallbacks = 0, diagonal_checks = 0;
  for (const auto scaling : {DGDiffusionFaceScaling::Arithmetic,
                             DGDiffusionFaceScaling::Max,
                             DGDiffusionFaceScaling::Sum})
  {
    coefficient.stage = 0;
    FillEvolvingFaceQdata(mesh, fqs, coefficient, side_q);
    InteriorFaceDGDiffusionApplyScratch work, dense;
    work.compact_qdata = true;
    auto build = [&](InteriorFaceDGDiffusionApplyScratch& target,
                      bool factors, std::size_t budget = 256u * 1024u * 1024u)
    {
      return BuildInteriorFaceDGDiffusionApplyData2D(
        fes, fqs, side_q, 2.75, target.shape, target.weighted_normal_dshape,
        target.penalty, target.face_elements, target.element_vdof_map, scaling,
        &target.normal_dshape_base, &target.penalty_metric,
        factors ? &target.shape_dictionary : nullptr,
        factors ? &target.normal_factorization : nullptr, budget);
    };
    auto info = build(work, true);
    auto dense_info = build(dense, false);
    MFEM_VERIFY(work.normal_factorization.TensorMatches(info),
                "Compact qdata fixture requires independently admitted p5/Q6 tensor geometry.");
    const void* full_shape = work.shape.HostRead();
    const void* full_base = work.normal_dshape_base.HostRead();
    const void* full_normal = work.weighted_normal_dshape.HostRead();
    auto check_storage = [&]()
    {
      if (!native_cuda) { return; }
      const auto inventory = DeviceAllocationInventory();
      MFEM_VERIFY(!HasDeviceAllocation(inventory, full_shape) &&
                    !HasDeviceAllocation(inventory, full_base) &&
                    !HasDeviceAllocation(inventory, full_normal) &&
                    !work.shape.GetMemory().DeviceIsValid() &&
                    !work.normal_dshape_base.GetMemory().DeviceIsValid() &&
                    !work.weighted_normal_dshape.GetMemory().DeviceIsValid(),
                  "Compact refresh/apply/diagonal must free full S/base/N device allocations.");
      MFEM_VERIFY(work.side_coefficient.GetMemory().DeviceIsValid() &&
                    work.penalty.GetMemory().DeviceIsValid(),
                  "Compact refresh must publish current coefficients and penalties on device.");
      MFEM_VERIFY(work.dense_normal_generation != work.coefficient_generation,
                  "Compact apply and diagonal must not materialize the stale dense normal table.");
    };
    auto parity = [&](bool compact, bool assembled, DGDiffusionFaceScaling current_scaling)
    {
      for (const real_t ip_sigma : {-1.0, 1.0})
      {
        Vector actual(x.Size()), diagonal(x.Size());
        actual.UseDevice(true);
        actual = 0.03125;
        // Diagonal first also tests lazy fallback without an apply having
        // already repaired the old dense normal table.
        diagonal = 0.125;
        ::igrlab::dg_face::AddInteriorFaceDGDiffusionDiagonal2D(info, work, ip_sigma, diagonal);
        LaunchInteriorFaceDGDiffusionApply2D(info, work, x, ip_sigma, actual);
        if (compact) { check_storage(); }
        else if (work.coefficient_geometry_revision == info.connectivity_revision)
        {
          MFEM_VERIFY(work.dense_normal_generation == work.coefficient_generation,
                      "Dense fallback diagonal/apply must materialize the current Q generation.");
        }
        Vector dense_contribution;
        const Vector expected = OriginalRecordAndGather(dense_info, dense, x, ip_sigma,
                                                       dense_contribution);
        CheckFactorizationError(actual, expected, "Compact qdata versus independently rebuilt dense action");
        if (work.face_contribution.Size())
        {
          CheckFactorizationError(work.face_contribution, dense_contribution,
                                  "Compact qdata versus independent dense face records");
        }
        Vector dense_diagonal(x.Size());
        dense_diagonal = 0.125;
        ::igrlab::dg_face::AddInteriorFaceDGDiffusionDiagonal2D(
          dense_info, dense, ip_sigma, dense_diagonal);
        CheckFactorizationError(diagonal, dense_diagonal,
                                "Compact qdata versus independently rebuilt dense diagonal");
        if (assembled && current_scaling == DGDiffusionFaceScaling::Arithmetic)
        { CheckAssembledFaceDiagonal(fes, mesh, fqs, coefficient, ip_sigma, diagonal); }
        Vector repeat(x.Size());
        repeat.UseDevice(true);
        repeat = 0.03125;
        LaunchInteriorFaceDGDiffusionApply2D(info, work, x, ip_sigma, repeat);
        MFEM_VERIFY(SameBits(actual, repeat), "Repeated current-qdata apply changed bits.");
        ++diagonal_checks;
      }
      if (!compact && native_cuda) { ++dense_fallbacks; }
    };
    auto refresh = [&]()
    {
      const auto previous = work.coefficient_generation;
      // A compact update must not overwrite/read this invalid old dense cache.
      real_t* old_normal = work.weighted_normal_dshape.HostWrite();
      std::fill(old_normal, old_normal + work.weighted_normal_dshape.Size(),
                std::numeric_limits<real_t>::quiet_NaN());
      LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(info, work, side_q, 2.75, scaling);
      MFEM_VERIFY(work.coefficient_generation > previous,
                  "Every repeated coefficient refresh must invalidate the old dense generation.");
      if (native_cuda) { check_storage(); ++compact_refreshes; }
      dense_info = build(dense, false);
    };

    // The builder's initial dense values remain usable before current-Q storage exists.
    parity(false, true, scaling);
    if (native_cuda)
    {
      work.shape.Read(); work.normal_dshape_base.Read(); work.weighted_normal_dshape.Read();
      const auto inventory = DeviceAllocationInventory();
      MFEM_VERIFY(HasDeviceAllocation(inventory, full_shape) &&
                    HasDeviceAllocation(inventory, full_base) &&
                    HasDeviceAllocation(inventory, full_normal),
                  "Allocation-removal witness must start with all three full device tables.");
    }
    for (int stage = 1; stage <= 3; ++stage)
    {
      coefficient.stage = stage;
      FillEvolvingFaceQdata(mesh, fqs, coefficient, side_q);
      refresh();
      parity(native_cuda, true, scaling);
      if (stage == 2) // An unchanged Q is still a new, valid refresh generation.
      { refresh(); parity(native_cuda, false, scaling); }
    }
    auto& factors = work.normal_factorization;
    const auto tensor_revision = factors.tensor_geometry_revision;
    factors.tensor_geometry_revision = 0;
    parity(false, false, scaling); // First fallback must materialize the newest Q.
    factors.tensor_geometry_revision = tensor_revision;
    coefficient.stage = 4;
    FillEvolvingFaceQdata(mesh, fqs, coefficient, side_q);
    refresh();
    parity(native_cuda, true, scaling);
    const auto coefficient_revision = factors.coefficient_revision;
    factors.coefficient_revision = 0;
    parity(false, false, scaling); // A later fallback must not reuse that dense generation.
    factors.coefficient_revision = coefficient_revision;
    coefficient.stage = 5;
    FillEvolvingFaceQdata(mesh, fqs, coefficient, side_q);
    refresh();
    const Array<int> native_map(factors.tensor_lex_to_native);
    factors.tensor_lex_to_native.SetSize(0);
    parity(false, false, scaling); // Missing tensor storage with valid reference geometry.
    factors.tensor_lex_to_native = native_map;

    LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
      info, work, side_q, 2.75, DGDiffusionFaceScaling::HarmonicWeighted);
    dense_info = BuildInteriorFaceDGDiffusionApplyData2D(
      fes, fqs, side_q, 2.75, dense.shape, dense.weighted_normal_dshape,
      dense.penalty, dense.face_elements, dense.element_vdof_map,
      DGDiffusionFaceScaling::HarmonicWeighted);
    parity(false, false, DGDiffusionFaceScaling::HarmonicWeighted);
    refresh();
    parity(native_cuda, false, scaling);

    const auto old_revision = info.connectivity_revision;
    info = build(work, false); // Legacy same-size rebuild leaves old factor epochs stale.
    dense_info = build(dense, false);
    full_shape = work.shape.HostRead(); full_base = work.normal_dshape_base.HostRead();
    full_normal = work.weighted_normal_dshape.HostRead();
    MFEM_VERIFY(info.connectivity_revision != old_revision && !factors.TensorMatches(info),
                "Same-size rebuild must reject all earlier tensor/coefficient epochs.");
    parity(false, false, scaling);
    LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(info, work, side_q, 2.75, scaling);
    parity(false, false, scaling);
    info = build(work, true, 1);
    dense_info = build(dense, false);
    LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(info, work, side_q, 2.75, scaling);
    MFEM_VERIFY(!factors.TensorMatches(info), "Insufficient cache budget must force dense refresh.");
    parity(false, false, scaling);
    if (scaling == DGDiffusionFaceScaling::Sum)
    {
      // Geometry and quadrature change without changing any packed dimension.
      mesh.Transform([](const Vector& point, Vector& out)
      { out.SetSize(2); out[0] = 1.03 * point[0] + 0.07 * point[1];
        out[1] = 0.97 * point[1]; });
      rule.IntPoint(0).x += 0.001;
      FaceQuadratureSpace rebuilt_fqs(mesh, rule, FaceType::Interior);
      ++coefficient.stage;
      FillEvolvingFaceQdata(mesh, rebuilt_fqs, coefficient, side_q);
      const auto stale_revision = info.connectivity_revision;
      info = BuildInteriorFaceDGDiffusionApplyData2D(
        fes, rebuilt_fqs, side_q, 2.75, work.shape, work.weighted_normal_dshape,
        work.penalty, work.face_elements, work.element_vdof_map, scaling,
        &work.normal_dshape_base, &work.penalty_metric, &work.shape_dictionary,
        &work.normal_factorization);
      dense_info = BuildInteriorFaceDGDiffusionApplyData2D(
        fes, rebuilt_fqs, side_q, 2.75, dense.shape, dense.weighted_normal_dshape,
        dense.penalty, dense.face_elements, dense.element_vdof_map, scaling);
      full_shape = work.shape.HostRead(); full_base = work.normal_dshape_base.HostRead();
      full_normal = work.weighted_normal_dshape.HostRead();
      MFEM_VERIFY(info.connectivity_revision != stale_revision && factors.TensorMatches(info) &&
                    factors.coefficient_revision == 0,
                  "Geometry/rule rebuild must discard every earlier current-Q admission stamp.");
      parity(false, false, scaling);
      LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(info, work, side_q, 2.75, scaling);
      parity(native_cuda, false, scaling);
    }
  }

  // A real unsupported element order must refresh and apply through dense
  // storage even with the selector enabled and repeating reference geometry.
  DG_FECollection unsupported_fec(4, 2, mfem::BasisType::GaussLegendre);
  ParFiniteElementSpace unsupported_fes(&mesh, &unsupported_fec);
  FaceQuadratureSpace unsupported_fqs(mesh, rule, FaceType::Interior);
  FillEvolvingFaceQdata(mesh, unsupported_fqs, coefficient, side_q);
  InteriorFaceDGDiffusionApplyScratch unsupported, unsupported_dense;
  unsupported.compact_qdata = true;
  const auto unsupported_info = BuildInteriorFaceDGDiffusionApplyData2D(
    unsupported_fes, unsupported_fqs, side_q, 2.75, unsupported.shape,
    unsupported.weighted_normal_dshape, unsupported.penalty, unsupported.face_elements,
    unsupported.element_vdof_map, DGDiffusionFaceScaling::Arithmetic,
    &unsupported.normal_dshape_base, &unsupported.penalty_metric,
    &unsupported.shape_dictionary, &unsupported.normal_factorization);
  const auto unsupported_dense_info = BuildInteriorFaceDGDiffusionApplyData2D(
    unsupported_fes, unsupported_fqs, side_q, 2.75, unsupported_dense.shape,
    unsupported_dense.weighted_normal_dshape, unsupported_dense.penalty,
    unsupported_dense.face_elements, unsupported_dense.element_vdof_map);
  LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
    unsupported_info, unsupported, side_q, 2.75, DGDiffusionFaceScaling::Arithmetic);
  MFEM_VERIFY(!unsupported.normal_factorization.TensorMatches(unsupported_info) &&
                unsupported.dense_normal_generation == unsupported.coefficient_generation,
              "Unsupported p4 element order must retain the current dense refresh.");
  Vector unsupported_x(unsupported_fes.GetVSize()), unsupported_actual(unsupported_fes.GetVSize());
  unsupported_x.UseDevice(true); unsupported_actual.UseDevice(true);
  real_t* unsupported_input = unsupported_x.HostWrite();
  for (int i = 0; i < unsupported_x.Size(); ++i)
  { unsupported_input[i] = 0.01 * std::sin(0.083 * (i + 1)); }
  unsupported_actual = 0.03125;
  LaunchInteriorFaceDGDiffusionApply2D(
    unsupported_info, unsupported, unsupported_x, -1.0, unsupported_actual);
  Vector unsupported_contribution;
  const Vector unsupported_expected = OriginalRecordAndGather(
    unsupported_dense_info, unsupported_dense, unsupported_x, -1.0, unsupported_contribution);
  CheckFactorizationError(unsupported_actual, unsupported_expected,
                          "Unsupported compact-qdata selector dense fallback");
  Vector unsupported_diagonal(unsupported_x.Size());
  unsupported_diagonal = 0.125;
  ::igrlab::dg_face::AddInteriorFaceDGDiffusionDiagonal2D(
    unsupported_info, unsupported, -1.0, unsupported_diagonal);
  CheckAssembledFaceDiagonal(unsupported_fes, mesh, unsupported_fqs,
                             coefficient, -1.0, unsupported_diagonal);
  if (native_cuda)
  {
    MFEM_VERIFY(unsupported.weighted_normal_dshape.GetMemory().DeviceIsValid() &&
                  HasDeviceAllocation(DeviceAllocationInventory(),
                                      unsupported.weighted_normal_dshape.HostRead()),
                "Unsupported compact selector must actually use its current dense device table.");
  }
  std::cout << "ExactFaceCompactQdata native_cuda=" << native_cuda
            << " compact_refreshes=" << compact_refreshes
            << " dense_fallbacks=" << dense_fallbacks
            << " independent_diagonal_checks=" << diagonal_checks << '\n';
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
  Mesh serial = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL,
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
  Vector dense_actions[2], dense_diagonals[2];
  for (const int variant : {0, 1, 2})
  {
    // This switch is sampled when the production face operator is built.
    ScopedEnvironment selection("DG_IGR_IGR_FACTORIZED_FACE_NORMALS",
                                variant == 1 ? "true" : "false");
    ScopedEnvironment canonical_selection("IGRLAB_IGR_FACTORIZED_FACE_NORMALS",
                                          variant == 1 ? "true" : "false");
    ScopedEnvironment qdata_selection("DG_IGR_IGR_COMPACT_FACE_QDATA",
                                      variant == 2 ? "true" : "false");
    ScopedEnvironment canonical_qdata_selection("IGRLAB_IGR_COMPACT_FACE_QDATA",
                                                variant == 2 ? "true" : "false");
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
      Vector actual(input.Size()), diagonal(input.Size());
      actual.UseDevice(true);
      face->AssembleDiagonal(diagonal);
      face->Mult(input, actual);
      RequireFinite(actual);
      MFEM_VERIFY(actual.Normlinf() > 1.0e-6,
                  "Factorized-normal production fixture needs a nonzero face action.");
      if (!variant)
      { dense_actions[refresh] = actual; dense_diagonals[refresh] = diagonal; }
      else
      {
        CheckFactorizationError(actual, dense_actions[refresh],
                                "Production factorized face dispatch");
        CheckFactorizationError(diagonal, dense_diagonals[refresh],
                                "Production compact-qdata current diagonal");
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
// Opt-in fixed workload: preserve the record-only comparisons, then measure
// current dense-refresh/tensor apply against compact refresh/tensor apply.
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
  const bool tensor_admitted = factors.TensorMatches(info);
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
            << " tensor_admitted=" << tensor_admitted
            << " tensor_n=" << factors.tensor_n
            << " tensor_slots=" << factors.tensor_slot_count
            << " tensor_endpoint_bytes=" << std::size_t(factors.tensor_endpoint.Size()) * sizeof(real_t)
            << " tensor_tangent_bytes=" << std::size_t(factors.tensor_tangent.Size()) * sizeof(real_t)
            << " tensor_axes_bytes=" << std::size_t(factors.tensor_axes.Size()) * sizeof(int)
            << " tensor_slot_bytes=" << std::size_t(factors.tensor_face_slots.Size()) * sizeof(int)
            << " tensor_native_map_bytes=" << std::size_t(factors.tensor_lex_to_native.Size()) * sizeof(int)
            << " dense_shape_bytes=" << std::size_t(work.shape.Size()) * sizeof(real_t)
            << " dense_geometry_normal_bytes=" << std::size_t(work.normal_dshape_base.Size()) * sizeof(real_t)
            << " dense_weighted_normal_bytes=" << std::size_t(work.weighted_normal_dshape.Size()) * sizeof(real_t)
            << " factor_reference_bytes=" << std::size_t(factors.reference_dshape.Size()) * sizeof(real_t)
            << " factor_geometry_bytes=" << std::size_t(factors.normal_geometry.Size()) * sizeof(real_t)
            << " factor_slot_bytes=" << std::size_t(factors.face_slots.Size()) * sizeof(int)
            << " mesh_path=" << std::quoted(mesh_path) << " serial_refinements=0 parallel_refinements=4"
            << " warmups=" << warmups << " repeats=" << repeats
            << " scope=record_kernel_only" << std::endl;

  MFEM_VERIFY(factorized && tensor_admitted,
              "The prescribed large case must admit the tensor candidate before it is timed.");
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
  // No full S/N pointers are supplied: the true tensor record must consume
  // only its 1D basis, current coefficients and per-quadrature geometry.
  auto tensor_record = record;
  tensor_record.shape = nullptr;
  tensor_record.weighted_normal_dshape = nullptr;
  measure("tensor_1d", ::igrlab::dg_face::InteriorFaceDGDiffusionTensorRecord2DWarp{
    tensor_record, factors.tensor_endpoint.Read(), factors.tensor_tangent.Read(),
    factors.tensor_axes.Read(), factors.tensor_face_slots.Read(),
    factors.tensor_lex_to_native.Read(), factors.normal_geometry.Read(),
    work.side_coefficient.Read(), factors.tensor_slot_count});

  // Use one geometry pack for both variants so the large host S/base/N tables
  // and all mesh/packing choices are identical. The opt-in flag changes only
  // current qdata refresh and its resulting dense-device cache state.
  Vector combined_output(x.Size()), combined_reference(x.Size());
  combined_output.UseDevice(true); combined_reference.UseDevice(true);
  combined_output = 0.0; combined_reference = 0.0;
  const void* full_shape = work.shape.HostRead();
  const void* full_base = work.normal_dshape_base.HostRead();
  const void* full_normal = work.weighted_normal_dshape.HostRead();
  std::size_t baseline_resident_bytes = 0, baseline_dense_bytes = 0;
  std::size_t baseline_free_bytes = 0;
  double baseline_refresh_ms = 0.0, baseline_combined_ms = 0.0;
  for (const bool compact : {false, true})
  {
    work.compact_qdata = compact;
    auto update_and_apply = [&](bool apply)
    {
      LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
        info, work, side_q, 2.75, DGDiffusionFaceScaling::Arithmetic);
      if (apply)
      {
        combined_output = 0.03125;
        LaunchInteriorFaceDGDiffusionApply2D(info, work, x, -1.0, combined_output);
      }
    };
    auto time_refresh = [&](bool apply)
    {
      for (int repeat = 0; repeat < warmups; ++repeat) { update_and_apply(apply); }
      MFEM_GPU_CHECK(cudaGetLastError());
      MFEM_GPU_CHECK(cudaDeviceSynchronize());
      MFEM_GPU_CHECK(cudaEventRecord(start));
      for (int repeat = 0; repeat < repeats; ++repeat) { update_and_apply(apply); }
      MFEM_GPU_CHECK(cudaEventRecord(stop));
      MFEM_GPU_CHECK(cudaEventSynchronize(stop));
      MFEM_GPU_CHECK(cudaGetLastError());
      float elapsed_ms = 0.0f;
      MFEM_GPU_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      return double(elapsed_ms) / repeats;
    };
    const double refresh_ms = time_refresh(false);
    const double combined_ms = time_refresh(true);
    const bool tensor_apply_admitted =
      ::igrlab::dg_face::HasCurrentInteriorFaceDGDiffusionTensorData2D(info, work);
    const bool compact_refresh_admitted = work.compact_qdata && tensor_apply_admitted &&
      work.coefficient_geometry_revision == info.connectivity_revision &&
      work.coefficient_generation != 0 &&
      work.dense_normal_generation != work.coefficient_generation;
    MFEM_VERIFY(tensor_apply_admitted && compact_refresh_admitted == compact,
                "Large qdata benchmark must time its declared tensor refresh/apply admission.");
    const auto inventory = DeviceAllocationInventory();
    const std::size_t dense_shape_device_bytes = DeviceResidentBytes(inventory, work.shape);
    const std::size_t dense_base_device_bytes = DeviceResidentBytes(inventory, work.normal_dshape_base);
    const std::size_t dense_normal_device_bytes = DeviceResidentBytes(inventory, work.weighted_normal_dshape);
    const std::size_t resident_bytes = FaceScratchDeviceResidentBytes(work, inventory);
    std::size_t free_bytes = 0, total_bytes = 0;
    MFEM_GPU_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    MFEM_VERIFY(work.side_coefficient.GetMemory().DeviceIsValid() &&
                  work.penalty.GetMemory().DeviceIsValid() &&
                  factors.normal_geometry.GetMemory().DeviceIsValid() &&
                  factors.tensor_endpoint.GetMemory().DeviceIsValid() &&
                  factors.tensor_tangent.GetMemory().DeviceIsValid(),
                "Timed tensor update/apply must actually consume compact current device data.");
    if (compact)
    {
      MFEM_VERIFY(!HasDeviceAllocation(inventory, full_shape) &&
                    !HasDeviceAllocation(inventory, full_base) &&
                    !HasDeviceAllocation(inventory, full_normal) &&
                    dense_shape_device_bytes + dense_base_device_bytes + dense_normal_device_bytes == 0 &&
                    resident_bytes < baseline_resident_bytes &&
                    baseline_resident_bytes - resident_bytes >= baseline_dense_bytes,
                  "Compact large-case device residency must remove the actual dense S/base/N allocations.");
      CheckFactorizationError(combined_output, combined_reference,
                              "Large compact refresh/apply versus current dense-refresh/tensor apply");
    }
    else
    {
      MFEM_VERIFY(HasDeviceAllocation(inventory, full_base) &&
                    HasDeviceAllocation(inventory, full_normal) &&
                    work.dense_normal_generation == work.coefficient_generation,
                  "Baseline must retain and refresh the original full dense device normal tables.");
      combined_reference = combined_output;
      baseline_resident_bytes = resident_bytes;
      baseline_dense_bytes = dense_shape_device_bytes + dense_base_device_bytes + dense_normal_device_bytes;
      baseline_free_bytes = free_bytes;
      baseline_refresh_ms = refresh_ms;
      baseline_combined_ms = combined_ms;
    }
    CheckFactorizationError(work.face_contribution, reference,
                            "Large current-qdata face contributions versus original dense record");
    std::cout << "ExactFaceBenchmark512 variant="
              << (compact ? "compact_refresh_tensor_apply" : "dense_refresh_tensor_apply")
              << " scope=coefficient_refresh_and_face_apply"
              << " mean_refresh_ms=" << refresh_ms
              << " mean_refresh_apply_ms=" << combined_ms
              << " refresh_speedup=" << baseline_refresh_ms / refresh_ms
              << " refresh_apply_speedup=" << baseline_combined_ms / combined_ms
              << " tensor_apply_admitted=" << tensor_apply_admitted
              << " compact_refresh_admitted=" << compact_refresh_admitted
              << " coefficient_generation=" << work.coefficient_generation
              << " dense_normal_generation=" << work.dense_normal_generation
              << " scratch_device_resident_bytes=" << resident_bytes
              << " dense_shape_device_bytes=" << dense_shape_device_bytes
              << " dense_geometry_normal_device_bytes=" << dense_base_device_bytes
              << " dense_weighted_normal_device_bytes=" << dense_normal_device_bytes
              << " scratch_device_bytes_removed=" << baseline_resident_bytes - resident_bytes
              << " cuda_free_bytes=" << free_bytes
              << " cuda_total_bytes=" << total_bytes
              << " cuda_free_bytes_increase="
              << (free_bytes >= baseline_free_bytes ? free_bytes - baseline_free_bytes : 0u)
              << " bitwise_dense_refresh_tensor_reference=" << SameBits(combined_output, combined_reference)
              << " output_reset_included=true gather_included=true solves_included=false"
              << std::endl;
  }
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
  CheckCompactFaceQdata();
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
  MFEM_VERIFY(factorization_stats.tensor_packs > 0 &&
                factorization_stats.tensor_normal_axes == 3 &&
                factorization_stats.reversed_tensor_sides > 0,
              "Tensor-face correctness fixtures must cover both normal axes and reversed tangents.");
  std::cout << std::setprecision(17) << "Exact face record/gather parity: backend=" << backend
            << " cases=" << cases << " max_assembled_error="
            << max_assembled_error
            << " factor_geometries=" << factorization_stats.prepared
            << " tensor_cuda_applications=" << factorization_stats.cuda_applications
            << " tensor_packs=" << factorization_stats.tensor_packs
            << " tensor_normal_axes=" << factorization_stats.tensor_normal_axes
            << " reversed_tensor_sides=" << factorization_stats.reversed_tensor_sides
            << " nonidentity_tensor_maps=" << factorization_stats.nonidentity_tensor_maps
            << " factor_max_relative_l2=" << factorization_stats.max_relative_l2
            << " factor_max_relative_linf=" << factorization_stats.max_relative_linf << '\n';
  return 0;
}
