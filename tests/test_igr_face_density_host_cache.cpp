#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "../src/physics/igr/IGR_face_density_host_cache.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Deliberately compile this fixture as CXX, including in CUDA builds: the
// production host projection's CXX branch is the exact arithmetic oracle.
#ifdef __CUDACC__
#error "Host face-density cache fixture must remain an ordinary CXX source."
#endif

namespace
{
using Cache = ::igrlab::igr_internal::HostFaceDensityPreparationCache;
using Info = ::igrlab::dg_face::InteriorFaceSideDensityProjectionInfo;
using mfem::real_t;

bool SameBits(const mfem::Vector& a, const mfem::Vector& b)
{
  return a.Size() == b.Size() && (a.Size() == 0 ||
    std::memcmp(a.HostRead(), b.HostRead(),
                static_cast<std::size_t>(a.Size()) * sizeof(real_t)) == 0);
}

bool SameScalar(real_t a, real_t b)
{
  return std::memcmp(&a, &b, sizeof(real_t)) == 0;
}

bool SameInfo(const Info& a, const Info& b)
{
  return a.dim == b.dim && a.dof == b.dof && a.vdim == b.vdim &&
    a.nq == b.nq && a.nf == b.nf && a.ne == b.ne &&
    a.shape_size == b.shape_size &&
    a.face_element_size == b.face_element_size &&
    a.element_vdof_map_size == b.element_vdof_map_size &&
    a.qf_size == b.qf_size;
}

void Check(bool condition, const std::string& label, int& failures)
{
  int all_passed = condition ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &all_passed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all_passed)
  {
    ++failures;
    if (mfem::Mpi::WorldRank() == 0)
    { std::cerr << "[FAIL] " << label << '\n'; }
  }
}

mfem::Mesh MakeMesh(int geometry)
{
  auto mesh = mfem::Mesh::MakeCartesian2D(
    4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 0.7);
  if (geometry == 2)
  {
    // At least three periodic cells per axis retain distinct mesh edges.
    const std::vector<mfem::Vector> shifts = {
      mfem::Vector({1.0, 0.0}), mfem::Vector({0.0, 0.7})};
    return mfem::Mesh::MakePeriodic(
      mesh, mesh.CreatePeriodicVertexMapping(shifts));
  }
  if (geometry == 1)
  {
    for (int e : {0, 15})
    {
      int* vertices = mesh.GetElement(e)->GetVertices();
      const int first = vertices[0];
      vertices[0] = vertices[1]; vertices[1] = vertices[2];
      vertices[2] = vertices[3]; vertices[3] = first;
    }
    mesh.FinalizeTopology();
    mesh.Finalize(false, false);
    mesh.SetCurvature(2, false, 2, mfem::Ordering::byVDIM);
    mesh.Transform([](const mfem::Vector& x, mfem::Vector& y)
    {
      y.SetSize(2);
      y[0] = x[0] + 0.08 * x[0] * x[1];
      y[1] = x[1] + 0.06 * x[0] * (1.0 - x[0]);
    });
  }
  return mesh;
}

void FillState(mfem::ParFiniteElementSpace& vfes,
               mfem::ParGridFunction& state, mfem::Vector& alpha,
               real_t constant_rho = -1.0)
{
  state.Update();
  state.UseDevice(true);
  real_t* values = state.HostWrite();
  mfem::Array<int> dofs;
  for (int e = 0; e < vfes.GetNE(); ++e)
  {
    vfes.GetElementDofs(e, dofs);
    for (int c = 0; c < vfes.GetVDim(); ++c)
    {
      for (int d = 0; d < dofs.Size(); ++d)
      {
        const int vdof = vfes.DofToVDof(dofs[d], c);
        const int decoded = mfem::FiniteElementSpace::DecodeDof(vdof);
        const real_t value = c == 0 ?
          (constant_rho >= 0.0 ? constant_rho :
           0.055 + 0.013 * mfem::Mpi::WorldRank() + 0.004 * e + 0.0007 * d) :
          0.4 + 0.1 * c + 0.003 * e + 0.0002 * d;
        values[decoded] = vdof >= 0 ? value : -value;
      }
    }
  }
  alpha.SetSize(vfes.GetNE());
  alpha.UseDevice(true);
  real_t* coefficients = alpha.HostWrite();
  for (int e = 0; e < vfes.GetNE(); ++e)
  { coefficients[e] = 0.003 * (1.0 + 0.09 * e + 0.02 * mfem::Mpi::WorldRank()); }
}

Info Original(const mfem::ParFiniteElementSpace& vfes,
              const mfem::FaceQuadratureSpace& fqs,
              const mfem::ParGridFunction& state, real_t scalar_alpha,
              const mfem::Vector* element_alpha, real_t rho_floor,
              mfem::Vector& result)
{
  if (element_alpha)
  {
    return ::igrlab::dg_face::ProjectInteriorFaceDensityCoefficient2D(
      vfes, fqs, state, *element_alpha, rho_floor, result);
  }
  return ::igrlab::dg_face::ProjectInteriorFaceDensityCoefficient2D(
    vfes, fqs, state, scalar_alpha, rho_floor, result);
}

mfem::Vector Compare(Cache& cache, mfem::ParFiniteElementSpace& vfes,
                     mfem::ParFiniteElementSpace& sfes,
                     const mfem::IntegrationRule& rule,
                     const mfem::ParGridFunction& state, real_t scalar_alpha,
                     const mfem::Vector* element_alpha, real_t floor,
                     bool expect_layout_change, int& failures, int& cases)
{
  bool changed = false;
  const auto* prepared = cache.Prepare(vfes, sfes, rule, &changed);
  Check(prepared != nullptr && changed == expect_layout_change,
        "supported layout admission and invalidation", failures);
  MFEM_VERIFY(prepared, "Supported host cache fixture declined preparation.");

  mfem::Vector actual;
  // Consume the cache first. In CUDA mode the current input may exist only
  // on the device after compiled MFEM vector operations below.
  const auto actual_info = cache.Project(
    vfes, state, scalar_alpha, element_alpha, floor, actual);
  mfem::FaceQuadratureSpace fresh(*vfes.GetMesh(), rule, mfem::FaceType::Interior);
  mfem::Vector expected;
  const auto expected_info = Original(
    vfes, fresh, state, scalar_alpha, element_alpha, floor, expected);
  Check(SameInfo(actual_info, expected_info) && SameBits(actual, expected),
        "cached/original host projection agrees in every output bit and dimension",
        failures);
  Check(actual_info.nf > 0 && actual_info.qf_size ==
          2 * actual_info.nq * actual_info.nf,
        "nonempty packed local-face outputs", failures);
  const auto& owned = prepared->GetIntRule(0);
  bool same_rule = owned.GetOrder() == rule.GetOrder() &&
                   owned.GetNPoints() == rule.GetNPoints();
  for (int q = 0; same_rule && q < rule.GetNPoints(); ++q)
  {
    same_rule = SameScalar(owned.IntPoint(q).x, rule.IntPoint(q).x) &&
                SameScalar(owned.IntPoint(q).weight, rule.IntPoint(q).weight);
  }
  Check(same_rule, "owned quadrature preserves exact point/weight bytes", failures);
  ++cases;
  return actual;
}

void CheckSpaces(int& failures, int& cases)
{
  for (int basis : {mfem::BasisType::GaussLegendre, mfem::BasisType::GaussLobatto})
  for (int order : {1, 3, 10})
  for (auto ordering : {mfem::Ordering::byNODES, mfem::Ordering::byVDIM})
  {
    Cache cache;
    auto serial = MakeMesh(order == 3 ? 1 : 0);
    mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
    mfem::L2_FECollection fec(order, 2, basis);
    mfem::ParFiniteElementSpace vfes(&mesh, &fec, 4, ordering);
    // Scalar and vector spaces need not share a collection or basis.
    mfem::L2_FECollection scalar_fec(order + 1, 2, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sfes(&mesh, &scalar_fec);
    mfem::ParGridFunction state(&vfes);
    mfem::Vector alpha;
    FillState(vfes, state, alpha);
    const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * order + 3);
    auto before = Compare(cache, vfes, sfes, rule, state, 0.007, nullptr,
                          0.1, false, failures, cases);
    Compare(cache, vfes, sfes, rule, state, 0.007, nullptr,
            0.1, false, failures, cases);
    Compare(cache, vfes, sfes, rule, state, 0.0, &alpha,
            0.1, false, failures, cases);

    // These library operations use the compiled device backend when selected,
    // unlike an ordinary CXX-TU forall lambda that would run on the host.
    state *= 1.23;
    alpha *= 1.11;
    const auto after = Compare(cache, vfes, sfes, rule, state, 0.009, nullptr,
                               0.04, false, failures, cases);
    Check(!SameBits(before, after), "warm cache consumes changed state/alpha/floor",
          failures);
    Compare(cache, vfes, sfes, rule, state, 0.0, &alpha,
            0.13, false, failures, cases);
    Compare(cache, vfes, sfes, rule, state, 0.0, &alpha,
            0.04, false, failures, cases);

    // Identical rule contents at another address do not change the layout.
    mfem::IntegrationRule custom(rule);
    Compare(cache, vfes, sfes, custom, state, 0.0, &alpha,
            0.1, false, failures, cases);
    custom.IntPoint(0).weight *= 1.001;
    Compare(cache, vfes, sfes, custom, state, 0.0, &alpha,
            0.1, true, failures, cases);
    custom.IntPoint(0).x += 0.001;
    Compare(cache, vfes, sfes, custom, state, 0.0, &alpha,
            0.1, true, failures, cases);
    custom.SetOrder(custom.GetOrder() + 1);
    Compare(cache, vfes, sfes, custom, state, 0.0, &alpha,
            0.1, true, failures, cases);

    mfem::IntegrationRule retained(custom);
    retained.IntPoint(0).x += 0.002;
    {
      mfem::IntegrationRule temporary(retained);
      Compare(cache, vfes, sfes, temporary, state, 0.0, &alpha,
              0.1, true, failures, cases);
    }
    // Project before supplying another rule: the destroyed input must not be
    // retained by the quadrature space or the prepared shape layout.
    mfem::Vector owned_result, owned_expected;
    const auto owned_info = cache.Project(vfes, state, 0.0, &alpha, 0.1, owned_result);
    mfem::FaceQuadratureSpace owned_fresh(mesh, retained, mfem::FaceType::Interior);
    const auto owned_expected_info = Original(
      vfes, owned_fresh, state, 0.0, &alpha, 0.1, owned_expected);
    Check(SameInfo(owned_info, owned_expected_info) &&
          SameBits(owned_result, owned_expected), "temporary rule lifetime", failures);

    mesh.NodesUpdated();
    Compare(cache, vfes, sfes, retained, state, 0.0, &alpha,
            0.1, true, failures, cases);
    mesh.Transform([](const mfem::Vector& x, mfem::Vector& y)
    { y.SetSize(2); y[0] = 1.03 * x[0]; y[1] = 0.97 * x[1]; });
    Compare(cache, vfes, sfes, retained, state, 0.0, &alpha,
            0.1, true, failures, cases);

    // Pointer identity must be checked independently from equal dimensions.
    mfem::ParFiniteElementSpace replaced_vfes(&mesh, &fec, 4, ordering);
    mfem::ParGridFunction replaced_state(&replaced_vfes);
    FillState(replaced_vfes, replaced_state, alpha);
    Compare(cache, replaced_vfes, sfes, retained, replaced_state, 0.0, &alpha,
            0.1, true, failures, cases);
    mfem::ParFiniteElementSpace replaced_sfes(&mesh, &scalar_fec);
    Compare(cache, replaced_vfes, replaced_sfes, retained, replaced_state, 0.0,
            &alpha, 0.1, true, failures, cases);
    mfem::L2_FECollection changed_fec(order + 1, 2,
      basis == mfem::BasisType::GaussLegendre ? mfem::BasisType::GaussLobatto :
                                             mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace changed_vfes(&mesh, &changed_fec, 4, ordering);
    mfem::ParGridFunction changed_state(&changed_vfes);
    FillState(changed_vfes, changed_state, alpha);
    Compare(cache, changed_vfes, replaced_sfes, retained, changed_state, 0.0,
            &alpha, 0.1, true, failures, cases);
  }

  // Changed topology/native maps and periodic face orientations.
  Cache cache;
  auto serial = MakeMesh(2);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::L2_FECollection fec(3, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(&mesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::ParFiniteElementSpace sfes(&mesh, &fec);
  mfem::ParGridFunction state(&vfes);
  mfem::Vector alpha;
  FillState(vfes, state, alpha);
  const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 9);
  Compare(cache, vfes, sfes, rule, state, 0.0, &alpha, 0.1, false, failures, cases);
  mesh.UniformRefinement();
  vfes.Update(false); sfes.Update(false);
  FillState(vfes, state, alpha);
  Compare(cache, vfes, sfes, rule, state, 0.0, &alpha, 0.1, true, failures, cases);
  Compare(cache, vfes, sfes, rule, state, 0.0, &alpha, 0.1, false, failures, cases);
}

void CheckFloorBoundaries(int& failures, int& cases)
{
  auto serial = MakeMesh(0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::L2_FECollection fec(1, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(&mesh, &fec, 4, mfem::Ordering::byVDIM);
  mfem::ParFiniteElementSpace sfes(&mesh, &fec);
  mfem::ParGridFunction state(&vfes);
  mfem::Vector alpha;
  mfem::IntegrationRule rule(2);
  rule.SetOrder(1);
  rule.IntPoint(0).x = 0.0; rule.IntPoint(0).weight = 0.5;
  rule.IntPoint(1).x = 1.0; rule.IntPoint(1).weight = 0.5;
  Cache cache;
  constexpr real_t floor = 0.125;
  for (real_t rho : {0.0, std::nextafter(floor, 0.0), floor,
                    std::nextafter(floor, 1.0), 0.25})
  {
    FillState(vfes, state, alpha, rho);
    alpha = 0.03125;
    for (bool local : {false, true})
    {
      const auto result = Compare(cache, vfes, sfes, rule, state, 0.03125,
                                  local ? &alpha : nullptr, floor,
                                  false, failures, cases);
      const real_t expected = 0.03125 / (rho < floor ? floor : rho);
      bool same = true;
      const real_t* values = result.HostRead();
      for (int i = 0; i < result.Size(); ++i)
      { same = same && SameScalar(values[i], expected); }
      Check(same, "endpoint density floor/equality/nextafter oracle", failures);
    }
  }
  // Rule matching uses exact bytes, even where equal numeric coordinates
  // have different signs. Restoring the original also invalidates.
  rule.IntPoint(0).x = -0.0;
  Compare(cache, vfes, sfes, rule, state, 0.03125, nullptr,
          floor, true, failures, cases);
  rule.IntPoint(0).x = 0.0;
  Compare(cache, vfes, sfes, rule, state, 0.03125, nullptr,
          floor, true, failures, cases);
}

void CheckFallback(int& failures, int& cases)
{
  for (int unsupported : {0, 1, 2})
  {
    const bool triangle = unsupported == 1;
    auto serial = triangle ? mfem::Mesh::MakeCartesian2D(
      4, 4, mfem::Element::TRIANGLE, true, 1.0, 0.7) : MakeMesh(0);
    mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
    mfem::L2_FECollection l2_fec(3, 2, mfem::BasisType::GaussLegendre);
    mfem::H1_FECollection h1_fec(3, 2);
    const mfem::FiniteElementCollection* fec = unsupported == 2 ?
      static_cast<const mfem::FiniteElementCollection*>(&h1_fec) : &l2_fec;
    mfem::ParFiniteElementSpace vfes(&mesh, fec, 4, mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace sfes(&mesh, fec);
    mfem::ParGridFunction state(&vfes);
    mfem::Vector alpha;
    FillState(vfes, state, alpha);
    const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 7);
    Cache cache(unsupported ? 64u * 1024u * 1024u : 0u);
    for (int repetition = 0; repetition < 2; ++repetition)
    {
      Check(cache.Prepare(vfes, sfes, rule) == nullptr,
            "unsupported geometry/collection or bounded cache declined",
            failures);
      // This is the production fallback decision: no prepared state is used.
      mfem::FaceQuadratureSpace fqs(mesh, rule, mfem::FaceType::Interior);
      mfem::Vector actual;
      const auto a = Original(vfes, fqs, state, 0.0, &alpha, 0.1, actual);
      // Independent finite-element value evaluation verifies that the generic
      // route remains usable, rather than comparing the same call to itself.
      const real_t* values = actual.HostRead();
      const real_t* coefficients = alpha.HostRead();
      mfem::Vector value(4);
      int packed = 0;
      bool same = true;
      for (int f = 0; f < fqs.GetNE(); ++f)
      {
        const int face = fqs.GetMeshFaceIndex(f);
        if (mesh.GetFaceInformation(face).IsShared()) { continue; }
        auto* tr = mesh.GetFaceElementTransformations(face);
        for (int q = 0; q < a.nq; ++q)
        {
          tr->SetAllIntPoints(&fqs.GetIntRule(f).IntPoint(q));
          for (int side = 0; side < 2; ++side)
          {
            const int element = side ? tr->Elem2No : tr->Elem1No;
            if (side) { state.GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), value); }
            else { state.GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), value); }
            const real_t expected = coefficients[element] /
              (value[0] < 0.1 ? 0.1 : value[0]);
            const real_t measured = values[side + 2 * (q + a.nq * packed)];
            same = same && std::isfinite(measured) && std::isfinite(expected) &&
              std::abs(measured - expected) <= 3e-13 * std::abs(expected);
          }
        }
        ++packed;
      }
      Check(same && packed == a.nf && a.nf > 0,
            "declined cache retains independently verified generic projection",
            failures);
      state *= 1.03; alpha *= 0.97;
      ++cases;
    }
  }
}
void CheckLargeAdmission(int& failures, int& cases)
{
  if (mfem::Mpi::WorldSize() != 1) { return; }
  auto box = mfem::Mesh::MakeCartesian2D(
    128, 128, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  const std::vector<mfem::Vector> shifts = {
    mfem::Vector({1.0, 0.0}), mfem::Vector({0.0, 1.0})};
  auto periodic = mfem::Mesh::MakePeriodic(
    box, box.CreatePeriodicVertexMapping(shifts));
  mfem::ParMesh mesh(MPI_COMM_WORLD, periodic);
  mfem::L2_FECollection fec(5, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(&mesh, &fec, 4);
  mfem::ParFiniteElementSpace sfes(&mesh, &fec);
  mfem::ParGridFunction state(&vfes);
  mfem::Vector alpha;
  FillState(vfes, state, alpha);
  const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 11);
  // This production-size case exceeds the former cap through geometry alone.
  Cache old_budget(64u * 1024u * 1024u);
  Check(old_budget.Prepare(vfes, sfes, rule) == nullptr,
        "p5/n128 exceeds the former face geometry budget", failures);
  Cache cache;
  auto before = Compare(cache, vfes, sfes, rule, state, 0.0, &alpha,
                        0.1, false, failures, cases);
  state *= 1.07;
  alpha *= 0.93;
  auto after = Compare(cache, vfes, sfes, rule, state, 0.0, &alpha,
                       0.08, false, failures, cases);
  Check(!SameBits(before, after),
        "large warm cache consumes fresh state and alpha", failures);
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  MFEM_VERIFY(backend == "cpu" || backend == "cuda", "Expected cpu or cuda.");
  mfem::Device device(backend.c_str());
  int failures = 0, cases = 0;
  CheckSpaces(failures, cases);
  CheckFloorBoundaries(failures, cases);
  CheckFallback(failures, cases);
  CheckLargeAdmission(failures, cases);
  if (mfem::Mpi::WorldRank() == 0)
  {
    std::cout << "Host face-density cache parity: backend=" << backend
              << " ranks=" << mfem::Mpi::WorldSize() << " cases=" << cases
              << " failures=" << failures << '\n';
  }
  return failures ? 1 : 0;
}
