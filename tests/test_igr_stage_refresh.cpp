#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "physics/igr/IGR.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "../src/physics/igr/IGR_face_density_cuda.hpp"
#include "../src/physics/igr/IGR_stage_coefficients_cuda.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

// The oracle must use host MFEM finite-element evaluation. Current inputs are
// changed with compiled MFEM vector operations, which execute on the device.
#ifdef __CUDACC__
#error "Stage refresh oracle must remain an ordinary CXX translation unit."
#endif

namespace
{
using mfem::real_t;
namespace algebra = ::igrlab::frozen_scalar_igr_algebra;
namespace internal = ::igrlab::igr_internal;

class ScopedEnv
{
public:
  ScopedEnv(const char* name, const char* value) : name_(name)
  {
    if (const char* previous = std::getenv(name))
    { present_ = true; previous_ = previous; }
    setenv(name, value, 1);
  }
  ~ScopedEnv()
  {
    if (present_) { setenv(name_.c_str(), previous_.c_str(), 1); }
    else { unsetenv(name_.c_str()); }
  }
private:
  std::string name_, previous_;
  bool present_ = false;
};

struct Checks
{
  int failures = 0, cases = 0;

  void True(bool condition, const std::string& label)
  {
    if (!condition)
    {
      ++failures;
      std::cerr << "[FAIL] rank=" << mfem::Mpi::WorldRank()
                << " " << label << '\n';
    }
  }

  void Near(const mfem::Vector& actual, const mfem::Vector& expected,
            real_t relative_tolerance, const std::string& label)
  {
    bool finite = actual.Size() == expected.Size();
    real_t error = 0.0, scale = 0.0;
    const real_t* a = actual.HostRead();
    const real_t* b = expected.HostRead();
    for (int i = 0; finite && i < actual.Size(); ++i)
    {
      finite = std::isfinite(a[i]) && std::isfinite(b[i]);
      error = std::max(error, std::abs(a[i] - b[i]));
      scale = std::max(scale, std::abs(b[i]));
    }
    True(finite && error <= relative_tolerance * std::max(real_t(1e-14), scale),
         label + " relative_max_error=" +
           std::to_string(error / std::max(real_t(1e-14), scale)));
    ++cases;
  }
};

bool SameBits(const mfem::Vector& a, const mfem::Vector& b)
{
  return a.Size() == b.Size() && (a.Size() == 0 ||
    std::memcmp(a.HostRead(), b.HostRead(),
                std::size_t(a.Size()) * sizeof(real_t)) == 0);
}

mfem::Mesh MakeMesh(bool curved = true)
{
  auto mesh = mfem::Mesh::MakeCartesian2D(
    3, 2, mfem::Element::QUADRILATERAL, true, 1.2, 0.9);
  // Positive rotations retain a valid mesh while changing which reference
  // axis and endpoint represent the same physical interior face.
  for (int e : {0, 5})
  {
    int* v = mesh.GetElement(e)->GetVertices();
    const int first = v[0];
    v[0] = v[1]; v[1] = v[2]; v[2] = v[3]; v[3] = first;
  }
  mesh.FinalizeTopology();
  mesh.Finalize(false, false);
  if (curved)
  {
    mesh.SetCurvature(2, false, 2, mfem::Ordering::byVDIM);
    mesh.Transform([](const mfem::Vector& x, mfem::Vector& y)
    {
      y.SetSize(2);
      y[0] = x[0] + .08*x[0]*x[1] + .035*x[1]*(.9-x[1]);
      y[1] = x[1] + .06*x[0]*(1.2-x[0]) + .025*x[0]*x[1];
    });
  }
  return mesh;
}

void SetComponent(const mfem::ParFiniteElementSpace& fes,
                  const mfem::Array<int>& dofs, int component, int node,
                  real_t value, real_t* data)
{
  const int vdof = fes.DofToVDof(dofs[node], component);
  const int decoded = mfem::FiniteElementSpace::DecodeDof(vdof);
  data[decoded] = vdof >= 0 ? value : -value;
}

// density_case 0 is positive, 1 is below the floor with a nonzero gradient,
// and 2 gives the two sides of some faces different floor branches.
void FillState(mfem::ParFiniteElementSpace& fes, mfem::ParGridFunction& state,
               mfem::Vector& element_alpha, int density_case)
{
  state.UseDevice(true);
  real_t* data = state.HostWrite();
  mfem::Array<int> dofs;
  mfem::Vector x(2);
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementDofs(e, dofs);
    const auto& fe = *fes.GetFE(e);
    auto* tr = fes.GetMesh()->GetElementTransformation(e);
    const bool below = density_case == 1 || (density_case == 2 && e % 2 == 0);
    for (int j = 0; j < fe.GetDof(); ++j)
    {
      tr->Transform(fe.GetNodes().IntPoint(j), x);
      const real_t rho = below ? .043 + .011*x[0] + .006*x[1] + .001*e
                               : 1.0 + .18*x[0] + .11*x[1] + .017*e;
      const real_t mx = rho*(.2 - .7*x[1] + .13*x[0]*x[1]) + .015*x[0];
      const real_t my = rho*(-.1 + .9*x[0] - .05*x[1]*x[1]) + .012*x[1];
      SetComponent(fes, dofs, 0, j, rho, data);
      SetComponent(fes, dofs, 1, j, mx, data);
      SetComponent(fes, dofs, 2, j, my, data);
      SetComponent(fes, dofs, 3, j, 3.0 + .2*x[0]*x[1] + .03*e, data);
    }
  }
  element_alpha.SetSize(fes.GetNE());
  element_alpha.UseDevice(true);
  real_t* alpha = element_alpha.HostWrite();
  for (int e = 0; e < fes.GetNE(); ++e) { alpha[e] = .003*(1.0 + .09*e); }
}

mfem::Vector StateIncrement(mfem::ParFiniteElementSpace& fes, bool energy_only)
{
  mfem::Vector increment(fes.GetVSize());
  increment.UseDevice(true);
  real_t* data = increment.HostWrite();
  std::fill(data, data + increment.Size(), real_t(0.0));
  mfem::Array<int> dofs;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementDofs(e, dofs);
    const auto& fe = *fes.GetFE(e);
    for (int j = 0; j < fe.GetDof(); ++j)
    {
      const auto& ip = fe.GetNodes().IntPoint(j);
      if (!energy_only)
      {
        SetComponent(fes, dofs, 0, j, .008 + .002*ip.x + .001*e, data);
        SetComponent(fes, dofs, 1, j, .013*ip.x - .017*ip.y + .002*e, data);
        SetComponent(fes, dofs, 2, j, -.019*ip.x + .009*ip.y - .001*e, data);
      }
      // Energy is deliberately unrelated to rho and momentum. No gamma or
      // thermodynamic reconstruction belongs in these three coefficients.
      SetComponent(fes, dofs, 3, j, 19.0 + 7.0*ip.x - 3.0*ip.y + e, data);
    }
  }
  return increment;
}

struct VolumeValues
{
  mfem::Vector invrho, ck, src;
};

VolumeValues HostVolume(const mfem::ParGridFunction& state,
                        mfem::QuadratureSpace& mass_qs,
                        mfem::QuadratureSpace& diff_qs,
                        mfem::QuadratureSpace& rhs_qs,
                        real_t alpha, const mfem::Vector* element_alpha,
                        real_t rho_floor, real_t* floor_semantics_gap = nullptr)
{
  // Synchronize only after the implementation under test has consumed state.
  (void)state.HostRead();
  const real_t* local_alpha = element_alpha ? element_alpha->HostRead() : nullptr;
  VolumeValues result;
  result.invrho.SetSize(mass_qs.GetSize());
  result.ck.SetSize(diff_qs.GetSize());
  result.src.SetSize(rhs_qs.GetSize());
  mfem::Vector value(4);
  mfem::DenseMatrix gradient(4, 2), constant_floor_gradient(2, 2);
  int mass_offset = 0, diff_offset = 0, rhs_offset = 0;
  for (int e = 0; e < state.ParFESpace()->GetNE(); ++e)
  {
    auto* tr = state.ParFESpace()->GetMesh()->GetElementTransformation(e);
    const real_t alpha_e = local_alpha ? local_alpha[e] : alpha;
    const auto& mass_ir = mass_qs.GetIntRule(e);
    for (int q = 0; q < mass_ir.GetNPoints(); ++q)
    {
      const auto& ip = mass_ir.IntPoint(q);
      tr->SetIntPoint(&ip);
      state.GetVectorValue(*tr, ip, value);
      result.invrho[mass_offset++] = algebra::SafeInverseDensity(value[0], rho_floor);
    }
    const auto& diff_ir = diff_qs.GetIntRule(e);
    for (int q = 0; q < diff_ir.GetNPoints(); ++q)
    {
      const auto& ip = diff_ir.IntPoint(q);
      tr->SetIntPoint(&ip);
      state.GetVectorValue(*tr, ip, value);
      result.ck[diff_offset++] = algebra::AlphaOverDensity(alpha_e, value[0], rho_floor);
    }
    const auto& rhs_ir = rhs_qs.GetIntRule(e);
    for (int q = 0; q < rhs_ir.GetNPoints(); ++q)
    {
      const auto& ip = rhs_ir.IntPoint(q);
      tr->SetIntPoint(&ip);
      state.GetVectorValue(*tr, ip, value);
      state.GetVectorGradient(*tr, gradient);
      const real_t source = algebra::SourceFromConservativeStateGradient(
        2, value.GetData(), 1, gradient.GetData(), 1, gradient.Height(), alpha_e, rho_floor);
      result.src[rhs_offset++] = source;
      if (floor_semantics_gap && value[0] < rho_floor)
      {
        // This tempting alternative drops the density-gradient term when rho
        // is floored. The existing BrokenDG quotient algebra keeps that term.
        for (int d = 0; d < 2; ++d)
          for (int c = 0; c < 2; ++c)
          { constant_floor_gradient(c, d) = gradient(1+c, d) / rho_floor; }
        const real_t alternative = algebra::SourceFromVelocityGradient(
          2, constant_floor_gradient.GetData(), 1, 2, alpha_e);
        *floor_semantics_gap = std::max(*floor_semantics_gap, std::abs(source-alternative));
      }
    }
  }
  return result;
}

VolumeValues RunVolume(Checks& checks, mfem::ParFiniteElementSpace& fes,
                       const mfem::ParGridFunction& state,
                       mfem::QuadratureSpace& mass_qs,
                       mfem::QuadratureSpace& diff_qs,
                       mfem::QuadratureSpace& rhs_qs,
                       real_t alpha, const mfem::Vector* element_alpha,
                       std::shared_ptr<internal::FusedStageCoefficientCache>& cache,
                       const std::string& label, bool require_floor_gap = false)
{
  const auto* restriction = fes.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
  MFEM_VERIFY(restriction, "Stage refresh fixture needs the actual LEX restriction.");
  mfem::Vector restricted(restriction->Height());
  restricted.UseDevice(true);
  restriction->Mult(state, restricted);
  mfem::QuadratureFunction invrho(mass_qs), ck(diff_qs), src(rhs_qs);
  invrho.UseDevice(true); ck.UseDevice(true); src.UseDevice(true);
  invrho = std::numeric_limits<real_t>::quiet_NaN();
  ck = std::numeric_limits<real_t>::quiet_NaN();
  src = std::numeric_limits<real_t>::quiet_NaN();
  std::string reason;
  const bool used = internal::TryUpdateFusedStageCoefficientsCUDA(
    fes, mass_qs, diff_qs, rhs_qs, restricted, alpha, .1, element_alpha,
    invrho, ck, src, cache, &reason);
  checks.True(used && cache != nullptr, label + " selects fused volume: " + reason);
  if (!used) { return {}; }
  // Consume the same device-current native state before the host oracle reads
  // it. This also exercises switching the cache between restricted/native loads.
  mfem::QuadratureFunction direct_invrho(mass_qs), direct_ck(diff_qs), direct_src(rhs_qs);
  direct_invrho.UseDevice(true); direct_ck.UseDevice(true); direct_src.UseDevice(true);
  direct_invrho = -19.0; direct_ck = -23.0; direct_src = -29.0;
  const auto restricted_cache = cache;
  const bool direct_used = internal::TryUpdateFusedStageCoefficientsCUDA(
    fes, mass_qs, diff_qs, rhs_qs, state, alpha, .1, element_alpha,
    direct_invrho, direct_ck, direct_src, cache, &reason,
    internal::StageCoefficientStateLayout::Native);
  checks.True(direct_used && cache == restricted_cache,
              label + " selects direct native state with the same reference cache: " + reason);
  checks.True(SameBits(direct_invrho, invrho) && SameBits(direct_ck, ck) && SameBits(direct_src, src),
              label + " direct native loads retain restricted FP64 coefficient bits");
  real_t floor_gap = 0.0;
  const auto expected = HostVolume(state, mass_qs, diff_qs, rhs_qs, alpha,
                                    element_alpha, .1, &floor_gap);
  checks.Near(invrho, expected.invrho, 2e-12, label + " inverse density");
  checks.Near(ck, expected.ck, 2e-12, label + " diffusion coefficient");
  checks.Near(src, expected.src, 2e-11, label + " physical-gradient source");
  checks.Near(direct_invrho, expected.invrho, 2e-12, label + " direct inverse density host oracle");
  checks.Near(direct_ck, expected.ck, 2e-12, label + " direct diffusion host oracle");
  checks.Near(direct_src, expected.src, 2e-11, label + " direct physical-gradient source host oracle");
  if (require_floor_gap)
  { checks.True(floor_gap > 1e-6, "under-floor fixture distinguishes quotient-density gradient semantics"); }
  VolumeValues actual;
  actual.invrho = invrho; actual.ck = ck; actual.src = src;
  return actual;
}

void CheckVolumeRejections(Checks& checks, mfem::ParMesh& mesh,
                           mfem::ParFiniteElementSpace& fes,
                           const mfem::ParGridFunction& state,
                           mfem::QuadratureSpace& mass_qs,
                           mfem::QuadratureSpace& diff_qs,
                           mfem::QuadratureSpace& rhs_qs,
                           std::shared_ptr<internal::FusedStageCoefficientCache>& cache)
{
  auto reject = [&](const mfem::ParFiniteElementSpace& active_fes,
                    const mfem::ParGridFunction& active_state,
                    mfem::QuadratureSpace& active_mass,
                    mfem::QuadratureSpace& active_diff,
                    const std::string& label)
  {
    const auto* restriction = active_fes.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
    mfem::Vector restricted(restriction->Height());
    restricted.UseDevice(true);
    restriction->Mult(active_state, restricted);
    mfem::QuadratureFunction invrho(active_mass), ck(active_diff), src(rhs_qs);
    invrho.UseDevice(true); ck.UseDevice(true); src.UseDevice(true);
    invrho = -17.25; ck = 23.125; src = -39.0625;
    const mfem::Vector before_invrho(invrho), before_ck(ck), before_src(src);
    const auto cache_before = cache;
    std::string reason;
    for (const auto layout : {internal::StageCoefficientStateLayout::Restricted,
                              internal::StageCoefficientStateLayout::Native})
    {
      const mfem::Vector& input = layout == internal::StageCoefficientStateLayout::Native
        ? static_cast<const mfem::Vector&>(active_state) : restricted;
      const bool used = internal::TryUpdateFusedStageCoefficientsCUDA(
        active_fes, active_mass, active_diff, rhs_qs, input, .007, .1, nullptr,
        invrho, ck, src, cache, &reason, layout);
      checks.True(!used && !reason.empty(), label + " declines with a reason");
      checks.True(SameBits(invrho, before_invrho) && SameBits(ck, before_ck) &&
                    SameBits(src, before_src) && cache == cache_before,
                  label + " leaves outputs and valid cache untouched for either state layout");
    }
    ++checks.cases;
  };
  mfem::IntegrationRule changed_mass(mass_qs.GetIntRule(0));
  changed_mass.IntPoint(0).x += .0001;
  mfem::QuadratureSpace bad_mass(mesh, changed_mass);
  reject(fes, state, bad_mass, diff_qs, "noncanonical mass quadrature point");
  mfem::IntegrationRule changed_diff(diff_qs.GetIntRule(0));
  changed_diff.IntPoint(0).weight *= 1.001;
  mfem::QuadratureSpace bad_diff(mesh, changed_diff);
  reject(fes, state, mass_qs, bad_diff, "noncanonical diffusion quadrature weight");
  const auto& eight = mfem::IntRules.Get(mfem::Geometry::SQUARE, 14);
  mfem::QuadratureSpace wrong_size(mesh, eight);
  reject(fes, state, wrong_size, diff_qs, "unsupported mass quadrature size");
  mfem::L2_FECollection p4(4, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace p4_fes(&mesh, &p4, 4, fes.GetOrdering());
  mfem::ParGridFunction p4_state(&p4_fes);
  mfem::Vector unused_alpha;
  FillState(p4_fes, p4_state, unused_alpha, 0);
  reject(p4_fes, p4_state, mass_qs, diff_qs, "unsupported state polynomial order");
}

void CheckVolume(Checks& checks)
{
  // Two bounded cohorts cover both component layouts and both density-floor
  // branches. All have the physical Q2 geometry and p5 / 7,6,6-point rules.
  for (const auto ordering : {mfem::Ordering::byVDIM, mfem::Ordering::byNODES})
  {
    auto serial = MakeMesh();
    mfem::ParMesh mesh(MPI_COMM_SELF, serial);
    mfem::L2_FECollection fec(5, 2, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace fes(&mesh, &fec, 4, ordering);
    mfem::ParGridFunction state(&fes);
    mfem::Vector alpha;
    const bool under_floor = ordering == mfem::Ordering::byNODES;
    FillState(fes, state, alpha, under_floor ? 1 : 0);
    const auto& mass_ir = mfem::IntRules.Get(mfem::Geometry::SQUARE, 12);
    const auto& diff_ir = mfem::IntRules.Get(mfem::Geometry::SQUARE, 10);
    mfem::QuadratureSpace mass_qs(mesh, mass_ir), diff_qs(mesh, diff_ir), rhs_qs(mesh, diff_ir);
    checks.True(mass_ir.GetNPoints() == 49 && diff_ir.GetNPoints() == 36,
                "p5/Q2 fixture has 7-point mass and 6-point diffusion/RHS rules");
    std::shared_ptr<internal::FusedStageCoefficientCache> cache;
    const auto initial = RunVolume(checks, fes, state, mass_qs, diff_qs, rhs_qs,
                                   .007, nullptr, cache, "cold p5/Q2", under_floor);
    if (!under_floor && initial.src.Size())
    {
      real_t minimum = 0.0;
      const real_t* source = initial.src.HostRead();
      for (int i = 0; i < initial.src.Size(); ++i) { minimum = std::min(minimum, source[i]); }
      checks.True(minimum < -1e-4,
                  "rotation source is negative and distinguishes trace(G^2) from Frobenius norm");
    }
    auto increment = StateIncrement(fes, false);
    state.Add(.27, increment); alpha *= 1.17;
    checks.True(!state.GetMemory().HostIsValid() && !alpha.GetMemory().HostIsValid(),
                "warm volume fixture supplies device-current state and local alpha");
    const auto current = RunVolume(checks, fes, state, mass_qs, diff_qs, rhs_qs,
                                   .009, &alpha, cache, "warm changed device state/local alpha", under_floor);
    checks.True(!SameBits(initial.src, current.src) && !SameBits(initial.ck, current.ck),
                "warm volume refresh consumes changed state and per-element alpha");
    auto energy_increment = StateIncrement(fes, true);
    state.Add(1.0, energy_increment);
    const auto changed_energy = RunVolume(checks, fes, state, mass_qs, diff_qs, rhs_qs,
                                          .009, &alpha, cache, "energy-only update");
    checks.True(SameBits(current.invrho, changed_energy.invrho) &&
                  SameBits(current.ck, changed_energy.ck) && SameBits(current.src, changed_energy.src),
                "volume coefficients do not consume conservative energy or gamma");
    if (!under_floor)
    {
      const auto before_geometry = changed_energy.src;
      mesh.Transform([](const mfem::Vector& x, mfem::Vector& y)
      {
        y.SetSize(2);
        y[0] = x[0] + .018*x[0]*x[1];
        y[1] = x[1] + .013*x[0]*x[0];
      });
      mesh.NodesUpdated();
      const auto moved = RunVolume(checks, fes, state, mass_qs, diff_qs, rhs_qs,
                                    .009, &alpha, cache, "updated Q2 geometry");
      checks.True(!SameBits(before_geometry, moved.src),
                  "geometry update changes physical derivatives on unchanged state dofs");
      std::shared_ptr<internal::FusedStageCoefficientCache> fresh;
      const auto rebuilt = RunVolume(checks, fes, state, mass_qs, diff_qs, rhs_qs,
                                      .009, &alpha, fresh, "fresh updated-geometry cache");
      checks.Near(moved.src, rebuilt.src, 1e-14, "reused/fresh geometry cache source");
      CheckVolumeRejections(checks, mesh, fes, state, mass_qs, diff_qs, rhs_qs, cache);
      RunVolume(checks, fes, state, mass_qs, diff_qs, rhs_qs,
                 .009, &alpha, cache, "valid refresh after rejections");
      mfem::ParFiniteElementSpace replacement(&mesh, &fec, 4, ordering);
      mfem::ParGridFunction replacement_state(&replacement);
      FillState(replacement, replacement_state, alpha, 0);
      RunVolume(checks, replacement, replacement_state, mass_qs, diff_qs, rhs_qs,
                 .009, &alpha, cache, "replacement state finite-element space");
    }
  }
}

mfem::Vector HostFace(const mfem::ParFiniteElementSpace& fes,
                      const mfem::FaceQuadratureSpace& fqs,
                      const mfem::ParGridFunction& state, real_t alpha,
                      const mfem::Vector* element_alpha,
                      const ::igrlab::dg_face::InteriorFaceSideDensityProjectionInfo& info,
                      bool* mixed_reference_axes = nullptr)
{
  (void)state.HostRead();
  const real_t* local_alpha = element_alpha ? element_alpha->HostRead() : nullptr;
  mfem::Vector expected(2*info.nq*info.nf), value(4);
  auto& mesh = *fes.GetMesh();
  int packed = 0;
  for (int f = 0; f < fqs.GetNE(); ++f)
  {
    const int mesh_face = fqs.GetMeshFaceIndex(f);
    if (mesh.GetFaceInformation(mesh_face).IsShared()) { continue; }
    auto* tr = mesh.GetFaceElementTransformations(mesh_face);
    if (mixed_reference_axes)
    {
      std::array<real_t, 4> a{}, b{};
      mfem::IntegrationPoint endpoint;
      endpoint.Set1w(0.0, 1.0); tr->SetAllIntPoints(&endpoint);
      a = {tr->GetElement1IntPoint().x, tr->GetElement1IntPoint().y,
           tr->GetElement2IntPoint().x, tr->GetElement2IntPoint().y};
      endpoint.Set1w(1.0, 1.0); tr->SetAllIntPoints(&endpoint);
      b = {tr->GetElement1IntPoint().x, tr->GetElement1IntPoint().y,
           tr->GetElement2IntPoint().x, tr->GetElement2IntPoint().y};
      const int axis1 = std::abs(b[0]-a[0]) > std::abs(b[1]-a[1]) ? 0 : 1;
      const int axis2 = std::abs(b[2]-a[2]) > std::abs(b[3]-a[3]) ? 0 : 1;
      *mixed_reference_axes = *mixed_reference_axes || axis1 != axis2;
    }
    for (int q = 0; q < info.nq; ++q)
    {
      tr->SetAllIntPoints(&fqs.GetIntRule(f).IntPoint(q));
      for (int side = 0; side < 2; ++side)
      {
        const int e = side ? tr->Elem2No : tr->Elem1No;
        if (side) { state.GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), value); }
        else { state.GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), value); }
        expected[side + 2*(q+info.nq*packed)] =
          algebra::AlphaOverDensity(local_alpha ? local_alpha[e] : alpha, value[0], .1);
      }
    }
    ++packed;
  }
  MFEM_VERIFY(packed == info.nf, "Host face oracle did not visit every packed local face.");
  return expected;
}

mfem::Vector RunFace(Checks& checks, mfem::ParFiniteElementSpace& fes,
                     mfem::ParFiniteElementSpace& sfes,
                     const mfem::ParGridFunction& state,
                     const mfem::IntegrationRule& rule, const mfem::Vector* alpha,
                     internal::FaceDensityCUDAProjectionCache& cache,
                     bool expect_tensor, const std::string& label,
                     bool require_mixed_axes = false)
{
  const auto& fqs = cache.Prepare(fes, sfes, rule);
  checks.True(cache.UsesTensor() == expect_tensor,
              label + " execution selection: " + cache.TensorFallbackReason());
  if (expect_tensor)
  {
    checks.True(cache.work.shape.Size() == 0 && cache.work.element_vdof_map.Size() == 0,
                label + " retains no dense per-face shape or per-component vdof map");
  }
  const auto info = internal::ProjectInteriorFaceDensityCoefficientCUDA(
    fes, fqs, state, .007, alpha, .1, cache.side_q, cache.work, &cache.info);
  bool mixed_axes = false;
  const auto expected = HostFace(fes, fqs, state, .007, alpha, info, &mixed_axes);
  checks.True(info.nf > 0 && info.nq == rule.GetNPoints(), label + " has nonempty local-face data");
  if (require_mixed_axes)
  { checks.True(mixed_axes, "rotated Q2 elements exercise different reference tangent axes across a face"); }
  checks.Near(cache.side_q, expected, 3e-12, label + " independent host side alpha/rho");
  return mfem::Vector(cache.side_q);
}

void CheckFace(Checks& checks)
{
  for (const auto ordering : {mfem::Ordering::byVDIM, mfem::Ordering::byNODES})
  {
    auto serial = MakeMesh();
    mfem::ParMesh mesh(MPI_COMM_SELF, serial);
    mfem::L2_FECollection fec(5, 2, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace fes(&mesh, &fec, 4, ordering), sfes(&mesh, &fec);
    mfem::ParGridFunction state(&fes);
    mfem::Vector alpha;
    FillState(fes, state, alpha, 2);
    const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 10);
    checks.True(rule.GetNPoints() == 6, "p5 face cohort has six Gauss-Legendre points");
    internal::FaceDensityCUDAProjectionCache dense_cache, tensor_cache;
    mfem::Vector initial;
    for (int update = 0; update < 2; ++update)
    {
      if (update) { auto delta = StateIncrement(fes, false); state.Add(.39, delta); alpha *= 1.19; }
      const mfem::Vector* local = update ? &alpha : nullptr;
      mfem::Vector dense, tensor;
      if (update)
      { checks.True(!state.GetMemory().HostIsValid() && !alpha.GetMemory().HostIsValid(),
                    "warm tensor-face fixture supplies device-current state and local alpha"); }
      {
        ScopedEnv execution("IGRLAB_IGR_FACE_DENSITY_EXECUTION", "tensor");
        tensor = RunFace(checks, fes, sfes, state, rule, local, tensor_cache,
                         true, "tensor p5/Q2 face", true);
      }
      {
        ScopedEnv execution("IGRLAB_IGR_FACE_DENSITY_EXECUTION", "dense");
        dense = RunFace(checks, fes, sfes, state, rule, local, dense_cache,
                        false, "dense p5/Q2 face", true);
      }
      checks.Near(tensor, dense, 3e-12, "dense/tensor face coefficient parity");
      if (update)
      { checks.True(!SameBits(initial, tensor), "warm tensor face refresh consumes changed density and local alpha"); }
      else { initial = tensor; }
    }
    checks.True(tensor_cache.rebuilds == 1 && tensor_cache.hits == 1,
                "changed state and alpha reuse tensor face geometry");
    if (ordering == mfem::Ordering::byVDIM)
    {
      ScopedEnv execution("IGRLAB_IGR_FACE_DENSITY_EXECUTION", "tensor");
      checks.True(dense_cache.work.shape.Size() > 0 &&
                    dense_cache.work.element_vdof_map.Size() > 0,
                  "mode-transition fixture starts with populated dense face geometry");
      const auto transition = RunFace(checks, fes, sfes, state, rule, &alpha,
                                       dense_cache, true, "same-cache dense to tensor transition");
      checks.True(dense_cache.rebuilds == 2 &&
                    dense_cache.work.shape.Capacity() == 0 &&
                    dense_cache.work.element_vdof_map.Capacity() == 0,
                  "dense to tensor transition rebuilds and releases dense geometry allocations");
      checks.Near(transition, tensor_cache.side_q, 1e-14,
                   "transitioned tensor cache matches existing tensor cache");

      mfem::ParFiniteElementSpace replacement_fes(
        &mesh, &fec, 4, mfem::Ordering::byNODES), replacement_sfes(&mesh, &fec);
      mfem::ParGridFunction replacement_state(&replacement_fes);
      mfem::Vector replacement_alpha;
      FillState(replacement_fes, replacement_state, replacement_alpha, 2);
      auto delta = StateIncrement(replacement_fes, false);
      replacement_state.Add(.51, delta); replacement_alpha *= .93;
      checks.True(!replacement_state.GetMemory().HostIsValid() &&
                    !replacement_alpha.GetMemory().HostIsValid(),
                  "replacement-space cohort supplies device-current state and local alpha");
      const auto replaced = RunFace(checks, replacement_fes, replacement_sfes,
                                     replacement_state, rule, &replacement_alpha,
                                     dense_cache, true, "same-cache replacement vector/scalar spaces");
      checks.True(dense_cache.rebuilds == 3 && !SameBits(transition, replaced),
                  "replacement spaces and component ordering rebuild the current density map");
    }
    {
      ScopedEnv execution("IGRLAB_IGR_FACE_DENSITY_EXECUTION", "tensor");
      mesh.NodesUpdated();
      RunFace(checks, fes, sfes, state, rule, &alpha, tensor_cache,
               true, "tensor face after node sequence change");
      checks.True(tensor_cache.rebuilds == 2, "node sequence invalidates tensor face cache");
      mfem::IntegrationRule custom(rule);
      custom.IntPoint(0).x += .001;
      RunFace(checks, fes, sfes, state, custom, &alpha, tensor_cache,
               false, "noncanonical face rule dense fallback");
      checks.True(!tensor_cache.TensorFallbackReason().empty(),
                  "noncanonical face quadrature explains tensor fallback");
      RunFace(checks, fes, sfes, state, rule, &alpha, tensor_cache,
               true, "canonical tensor face after generic fallback");
      mfem::FaceQuadratureSpace uncached(mesh, rule, mfem::FaceType::Interior);
      ::igrlab::dg_face::InteriorFaceSideDensityProjectionScratch work;
      mfem::Vector actual;
      const auto info = internal::ProjectInteriorFaceDensityCoefficientCUDA(
        fes, uncached, state, .007, &alpha, .1, actual, work);
      checks.Near(actual, HostFace(fes, uncached, state, .007, &alpha, info),
                   3e-12, "uncached generic CUDA face fallback");
      checks.True(work.shape.Size() > 0 && work.element_vdof_map.Size() > 0,
                  "uncached request retains the generic dense implementation");
      mfem::L2_FECollection p4(4, 2, mfem::BasisType::GaussLegendre);
      mfem::ParFiniteElementSpace p4_fes(&mesh, &p4, 4, ordering), p4_sfes(&mesh, &p4);
      mfem::ParGridFunction p4_state(&p4_fes);
      FillState(p4_fes, p4_state, alpha, 2);
      internal::FaceDensityCUDAProjectionCache unsupported;
      RunFace(checks, p4_fes, p4_sfes, p4_state, rule, &alpha, unsupported,
               false, "p4 generic CUDA face fallback");
      checks.True(!unsupported.TensorFallbackReason().empty(),
                  "unsupported face order explains tensor fallback");
    }
  }
}

void CheckSharedFaceFallback(Checks& checks)
{
  if (mfem::Mpi::WorldSize() < 2) { return; }
  ScopedEnv execution("IGRLAB_IGR_FACE_DENSITY_EXECUTION", "tensor");
  auto serial = mfem::Mesh::MakeCartesian2D(
    4*mfem::Mpi::WorldSize(), 2, mfem::Element::QUADRILATERAL, true, 1.2, .9);
  serial.SetCurvature(2, false, 2, mfem::Ordering::byVDIM);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::L2_FECollection fec(5, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 4), sfes(&mesh, &fec);
  mfem::ParGridFunction state(&fes);
  mfem::Vector alpha;
  FillState(fes, state, alpha, 2);
  const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 10);
  internal::FaceDensityCUDAProjectionCache cache;
  checks.True(mesh.GetNSharedFaces() > 0, "MPI fallback cohort actually has shared faces");
  RunFace(checks, fes, sfes, state, rule, &alpha, cache,
           false, "shared-face generic CUDA fallback");
  checks.True(!cache.TensorFallbackReason().empty(), "shared-face tensor request has a fallback reason");
}

struct SolverSnapshot
{
  mfem::Vector rhs, sigma, action;
};

SolverSnapshot Snapshot(Checks& checks, const ::igrlab::IGRSolver& solver,
                         const std::string& label)
{
  const auto* op = solver.GetLinearOperator();
  MFEM_VERIFY(op, "Stage refresh integration fixture requires the current public operator.");
  SolverSnapshot result;
  result.rhs = solver.GetLinearRHS();
  solver.GetSigma().GetTrueDofs(result.sigma);
  mfem::Vector probe(op->Width());
  real_t* values = probe.HostWrite();
  for (int i = 0; i < probe.Size(); ++i) { values[i] = .2 + std::sin(.17*i) + .03*std::cos(.11*i); }
  result.action.SetSize(op->Height());
  op->Mult(probe, result.action);
  mfem::Vector action(op->Height()), residual(result.rhs);
  op->Mult(result.sigma, action);
  residual -= action;
  const real_t relative = residual.Norml2() / std::max(real_t(1e-300), result.rhs.Norml2());
  checks.True(result.rhs.Norml2() > 1e-8 && std::isfinite(relative) && relative <= 5e-10 &&
                solver.GetLastLinearConverged(),
              label + " independently satisfies current b-A*sigma certificate");
  checks.True(std::abs(relative-solver.GetLastLinearTrueResidualRelRHS()) <=
                2e-12*std::max(real_t(1.0), relative),
              label + " agrees with the public current-A residual report");
  ++checks.cases;
  return result;
}

std::string UpdateSolver(::igrlab::IGRSolver& solver,
                          const mfem::ParGridFunction& state, bool warm)
{
  std::ostringstream output;
  {
    struct RestoreOutput
    {
      std::streambuf* previous;
      ~RestoreOutput() { std::cout.rdbuf(previous); }
    } restore{std::cout.rdbuf(output.rdbuf())};
    solver.UpdateFromState(state, warm, ::igrlab::IGRSolvePurpose::ProductionStage, true);
  }
  std::cout << output.str();
  return output.str();
}

void CheckSolver(Checks& checks)
{
  ScopedEnv partial("IGRLAB_IGR_PARTIAL_ASSEMBLY", "1");
  ScopedEnv qf("IGRLAB_IGR_DEVICE_QF", "1");
  ScopedEnv exact("IGRLAB_IGR_EXACT_DG_FACE_OPERATOR", "1");
  ScopedEnv face_cuda("IGRLAB_IGR_FACE_DENSITY_CUDA", "1");
  ScopedEnv face_cache("IGRLAB_IGR_FACE_DENSITY_CACHE", "1");
  ScopedEnv source("IGRLAB_IGR_NODAL_SOURCE", "baseline");
  ScopedEnv krylov("IGRLAB_IGR_KRYLOV_BACKEND", "cuda-fp64");
  ScopedEnv residual_scale("IGRLAB_IGR_KRYLOV_RESIDUAL_SCALE", "rhs");
  ScopedEnv warm_guard("IGRLAB_IGR_WARM_START_GUARD", "0");
  ScopedEnv debug("IGRLAB_DEBUG_QF", "0");
  ScopedEnv audit("IGRLAB_IGR_INGESTION_AUDIT", "0");
  ScopedEnv source_order("IGRLAB_IGR_SOURCE_QUADRATURE_MIN_ORDER", "0");
  ScopedEnv operator_order("IGRLAB_IGR_OPERATOR_QUADRATURE_MIN_ORDER", "0");
  auto serial = MakeMesh();
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  mfem::L2_FECollection fec(5, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 4), sfes(&mesh, &fec);
  mfem::ParGridFunction state(&fes);
  mfem::Vector alpha;
  FillState(fes, state, alpha, 0);
  ::igrlab::IGRSolver baseline(sfes, fes, 2, 4, .0015);
  ::igrlab::IGRSolver fused(sfes, fes, 2, 4, .0015);
  for (auto* solver : {&baseline, &fused})
  {
    solver->SetRhoFloor(.1);
    solver->SetGradBackend(::igrlab::IGRSolver::GradBackend::BrokenDG);
    solver->SetAlphaLengthScalePolicy(::igrlab::IGRSolver::AlphaLengthScalePolicy::Local);
    solver->SetIPKind(::igrlab::IGRSolver::IPKind::SIPG);
    solver->SetExactDGFaceOperator(true);
    solver->SetDiagnosticNanScan(false);
    solver->SetSolveMode(::igrlab::IGRSolver::SolveMode::Krylov);
    solver->SetLinearSolver(::igrlab::IGRSolver::LinearSolverType::FGMRES);
    solver->SetPreconditioner(::igrlab::IGRSolver::PrecondType::None);
    solver->SetNonconvergencePolicy(::igrlab::IGRSolver::NonconvergencePolicy::Error);
    solver->SetLinearSolveParams(2e-12, 0.0, 1200, 0);
  }
  mfem::Vector previous_rhs, previous_sigma;
  for (int update = 0; update < 2; ++update)
  {
    if (update) { auto delta = StateIncrement(fes, false); state.Add(.21, delta); }
    SolverSnapshot reference, actual;
    baseline.SetAlpha(.0015*(1.0+.13*update));
    fused.SetAlpha(.0015*(1.0+.13*update));
    {
      ScopedEnv coefficients("IGRLAB_IGR_COEFFICIENT_EXECUTION", "baseline");
      ScopedEnv faces("IGRLAB_IGR_FACE_DENSITY_EXECUTION", "dense");
      const auto log = UpdateSolver(baseline, state, update != 0);
      if (!update && mfem::Mpi::WorldRank() == 0)
      { checks.True(log.find("requested=baseline selected=baseline") != std::string::npos,
                    "production baseline pair actually selects baseline coefficients"); }
      reference = Snapshot(checks, baseline, "baseline solve");
    }
    {
      ScopedEnv coefficients("IGRLAB_IGR_COEFFICIENT_EXECUTION", "fused");
      ScopedEnv faces("IGRLAB_IGR_FACE_DENSITY_EXECUTION", "tensor");
      ScopedEnv direct_state("IGRLAB_IGR_STAGE_STATE_DIRECT", "1");
      const auto log = UpdateSolver(fused, state, update != 0);
      if (!update && mfem::Mpi::WorldRank() == 0)
      {
        checks.True(log.find("requested=fused selected=fused") != std::string::npos,
                    "production integration pair actually selects fused coefficients");
        checks.True(log.find("state_requested=direct state_selected=direct") != std::string::npos,
                    "production integration pair actually skips the full state restriction");
        checks.True(log.find("requested=tensor selected=tensor") != std::string::npos,
                    "production integration pair actually selects tensor face density");
      }
      actual = Snapshot(checks, fused, "fused/tensor solve");
    }
    checks.Near(actual.rhs, reference.rhs, 2e-11, "production refreshed RHS parity");
    checks.Near(actual.action, reference.action, 2e-11, "production current operator action parity");
    checks.Near(actual.sigma, reference.sigma, 2e-9, "production sigma solve parity");
    if (update)
    {
      checks.True(!SameBits(previous_rhs, actual.rhs) && !SameBits(previous_sigma, actual.sigma),
                  "production warm refresh consumes current state and alpha policy");
    }
    previous_rhs = actual.rhs; previous_sigma = actual.sigma;
  }
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const std::string backend = argc > 1 ? argv[1] : "cuda";
  MFEM_VERIFY(backend == "cuda", "Stage refresh test requires the cuda backend.");
  mfem::Device device(backend.c_str());
  MFEM_VERIFY(mfem::Device::Allows(mfem::Backend::CUDA), "CUDA backend is not active.");
  Checks checks;
  CheckVolume(checks);
  CheckFace(checks);
  CheckSharedFaceFallback(checks);
  CheckSolver(checks);
  int failures = checks.failures;
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (mfem::Mpi::WorldRank() == 0)
  {
    std::cout << "IGR stage refresh parity: backend=" << backend
              << " ranks=" << mfem::Mpi::WorldSize()
              << " local_cases=" << checks.cases << " failures=" << failures << '\n';
  }
  return failures ? 1 : 0;
}
