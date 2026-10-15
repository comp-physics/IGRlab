#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "utils/diagnostics/euler_admissibility.hpp"
#if defined(IGRLAB_ADMISSIBILITY_CUDA_PROBE)
#include "test_euler_admissibility_cache_cuda.hpp"
#endif

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace
{
using Cache = mfem::igrlab::detail::EulerAdmissibilityCache;
using PointSet = mfem::igrlab::EulerAdmissibilityPointSet;
using Summary = mfem::igrlab::EulerAdmissibilitySummary;

bool SameDouble(double a, double b)
{
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

bool SameSummary(const Summary& a, const Summary& b)
{
  const auto& x = a.first_local_issue;
  const auto& y = b.first_local_issue;
  return SameDouble(a.global_min_rho, b.global_min_rho) &&
    SameDouble(a.global_min_pressure, b.global_min_pressure) &&
    a.global_bad_rho == b.global_bad_rho &&
    a.global_bad_pressure == b.global_bad_pressure &&
    a.first_bad_rank == b.first_bad_rank &&
    x.found == y.found && x.pressure_bad == y.pressure_bad &&
    x.face_point == y.face_point && x.element == y.element &&
    x.face == y.face && x.point_index == y.point_index &&
    x.scalar_dof == y.scalar_dof && SameDouble(x.rho, y.rho) &&
    SameDouble(x.pressure, y.pressure);
}

void Check(bool condition, const std::string& label, MPI_Comm comm, int& failures)
{
  int passed = condition ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_INT, MPI_MIN, comm);
  if (!passed)
  {
    ++failures;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) { std::cerr << "[FAIL] " << label << '\n'; }
  }
}

void SetConstant(mfem::ParFiniteElementSpace& fes, mfem::Vector& state,
                 double rho = 1.0, double energy = 2.5)
{
  state.SetSize(fes.GetVSize());
  double* data = state.HostWrite();
  const int ndofs = fes.GetNDofs();
  for (int c = 0; c < fes.GetVDim(); ++c)
  {
    for (int d = 0; d < ndofs; ++d)
    {
      data[d + c * ndofs] = c == 0 ? rho :
        (c == fes.GetVDim() - 1 ? energy : 0.0);
    }
  }
}

Summary Compare(Cache& cache, mfem::ParFiniteElementSpace& fes,
                const mfem::Vector& state, PointSet points, int& failures,
                int qorder = 0, const std::vector<int>& face_orders = {},
                double gamma = 1.4, double rho_min = 1.0e-10,
                double p_min = 1.0e-10)
{
  const int dim = fes.GetMesh()->Dimension();
  const auto expected = mfem::igrlab::ScanEulerAdmissibility(
    fes, state, dim, gamma, rho_min, p_min, qorder, face_orders, points);
  const auto cold = cache.Scan(
    fes, state, dim, gamma, rho_min, p_min, qorder, face_orders, points);
  const auto warm = cache.Scan(
    fes, state, dim, gamma, rho_min, p_min, qorder, face_orders, points);
  Check(SameSummary(expected, cold) && SameSummary(expected, warm),
        "cached scan preserves every original summary field and bit pattern: dim=" +
          std::to_string(dim) + " p=" + std::to_string(fes.GetFE(0)->GetOrder()) +
          " point_set=" + std::to_string(static_cast<int>(points)) +
          " qorder=" + std::to_string(qorder),
        fes.GetComm(), failures);
  return warm;
}

void CompareAll(Cache& cache, mfem::ParFiniteElementSpace& fes,
                const mfem::Vector& state, int& failures, int qorder = 0,
                const std::vector<int>& face_orders = {})
{
  for (const auto points : {PointSet::Dof, PointSet::Limiter, PointSet::Output})
  { Compare(cache, fes, state, points, failures, qorder, face_orders); }
}

mfem::Mesh MakeMesh(int dim, int ranks, bool triangles = false)
{
  if (dim == 1) { return mfem::Mesh::MakeCartesian1D(4 * ranks); }
  if (dim == 3)
  {
    return mfem::Mesh::MakeCartesian3D(2 * ranks, 1, 1,
                                      mfem::Element::HEXAHEDRON);
  }
  auto mesh = mfem::Mesh::MakeCartesian2D(
    3 * ranks, 2, triangles ? mfem::Element::TRIANGLE :
    mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  if (!triangles)
  {
    // Reversed local face orientations and a curved map test cached face
    // reference coordinates rather than a Cartesian shape-table shortcut.
    int* vertices = mesh.GetElement(0)->GetVertices();
    const int first = vertices[0];
    vertices[0] = vertices[1]; vertices[1] = vertices[2];
    vertices[2] = vertices[3]; vertices[3] = first;
    mesh.FinalizeTopology();
    mesh.Finalize(false, false);
    mesh.SetCurvature(2, false, 2, mfem::Ordering::byNODES);
    mesh.Transform([](const mfem::Vector& x, mfem::Vector& y)
    {
      y.SetSize(2);
      y[0] = x[0] + 0.05 * x[0] * x[1];
      y[1] = x[1] + 0.04 * x[0] * (1.0 - x[0]);
    });
  }
  return mesh;
}

void CheckSpaces(int& failures)
{
  const int ranks = mfem::Mpi::WorldSize();
  Cache cache;
  for (const int basis : {mfem::BasisType::GaussLegendre,
                          mfem::BasisType::GaussLobatto})
  {
    for (const int order : {1, 3, 5, 10})
    {
      auto mesh = MakeMesh(2, ranks);
      mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mfem::DG_FECollection fec(order, 2, basis);
      mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
      mfem::Vector state(fes.GetVSize());
      SetConstant(fes, state);
      // Different element/dof values prevent a wrong cached incidence or
      // projection from passing merely because all states are constant.
      const int ndofs = fes.GetNDofs();
      double* values = state.HostReadWrite();
      for (int d = 0; d < ndofs; ++d)
      {
        values[d] += 0.001 * (d + 1);
        values[d + ndofs] = 0.02 + 0.0003 * d;
        values[d + 2 * ndofs] = -0.015 - 0.0002 * d;
        values[d + 3 * ndofs] += 0.002 * d;
      }
      CompareAll(cache, fes, state, failures);
      CompareAll(cache, fes, state, failures, 2 * order + 3, {7, -1, 7, 10});
      CompareAll(cache, fes, state, failures, 2 * order + 2, {10, 7});
      // Reuse the same state object: its values must never enter the cache.
      SetConstant(fes, state, 0.125, 2.5);
      CompareAll(cache, fes, state, failures);
      pmesh.UniformRefinement();
      fes.Update(false);
      SetConstant(fes, state);
      CompareAll(cache, fes, state, failures);
    }
  }
  // These paths remain the original evaluator if the plan cannot support the
  // dimension/geometry. Switching back to quads must not reuse an old plan.
  for (const int dim : {1, 2, 3})
  {
    auto mesh = MakeMesh(dim, ranks, dim == 2);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::DG_FECollection fec(2, dim, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace fes(&pmesh, &fec, dim + 2,
                                    mfem::Ordering::byNODES);
    mfem::Vector state(fes.GetVSize());
    SetConstant(fes, state);
    CompareAll(cache, fes, state, failures);
  }
}

void CheckSampleContract(int& failures)
{
  Cache cache;
  for (const int basis : {mfem::BasisType::GaussLegendre,
                          mfem::BasisType::GaussLobatto})
  {
    auto mesh = MakeMesh(2, mfem::Mpi::WorldSize());
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::DG_FECollection fec(3, 2, basis);
    mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
    mfem::Vector state(fes.GetVSize());
    SetConstant(fes, state, 0.125);
    for (const auto points : {PointSet::Dof, PointSet::Limiter, PointSet::Output})
    {
      // Independent counts for p3/base8/face{7,7,-3,9,10}: Limiter's distinct
      // orders are8/7/9/10. Orders8/9 share their Gauss points, but both count.
      // Output's scan order10 gives face orders12/9/11 after
      // the existing raise+dedup policy. Geometrically repeated points in
      // different rules, or on opposite element sides, remain separate.
      int expected = 0;
      for (int e = 0; e < fes.GetNE(); ++e)
      {
        if (points == PointSet::Dof || basis == mfem::BasisType::GaussLobatto)
        { expected += 16; }
        if (points == PointSet::Dof) { continue; }
        expected += mfem::IntRules.Get(mfem::Geometry::SQUARE,
          points == PointSet::Output ? 10 : 8).GetNPoints();
        for (int f = 0; f < pmesh.GetNumFaces(); ++f)
        {
          int e1 = -1, e2 = -1;
          pmesh.GetFaceElements(f, &e1, &e2);
          if ((e1 != e && e2 != e) || !pmesh.GetFaceElementTransformations(f))
          { continue; }
          const std::vector<int> orders = points == PointSet::Output ?
            std::vector<int>{9, 11, 12} : std::vector<int>{8, 7, 9, 10};
          for (const int order : orders)
          { expected += mfem::IntRules.Get(mfem::Geometry::SEGMENT,
                                          order).GetNPoints(); }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &expected, 1, MPI_INT, MPI_SUM, fes.GetComm());
      const auto result = Compare(cache, fes, state, points, failures,
                                  8, {7, 7, -3, 9, 10}, 1.4, 0.25);
      Check(result.global_bad_rho == expected && result.global_bad_pressure == 0,
            "independent raw-node/volume/face duplicate sample counts",
            fes.GetComm(), failures);
      Check(result.first_local_issue.found &&
            result.first_local_issue.element == 0 &&
            result.first_local_issue.point_index == 0 &&
            result.first_local_issue.face == -1 &&
            result.first_local_issue.scalar_dof ==
              ((points == PointSet::Dof ||
                basis == mfem::BasisType::GaussLobatto) ? 0 : -1),
            "raw nodes precede volume rules in the first-issue contract",
            fes.GetComm(), failures);
    }
  }

  // A quadratic GL state is positive on the base even-point volume rule but
  // negative at the centre sampled by Output's richer odd-point rule.
  auto mesh = MakeMesh(2, mfem::Mpi::WorldSize());
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(2, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector state(fes.GetVSize());
  SetConstant(fes, state);
  mfem::Array<int> vdofs;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementVDofs(e, vdofs);
    mfem::Vector element(vdofs.Size());
    element = 0.0;
    const auto& nodes = fes.GetFE(e)->GetNodes();
    mfem::DenseMatrix values(element.GetData(), nodes.GetNPoints(), 4);
    for (int j = 0; j < nodes.GetNPoints(); ++j)
    {
      const auto& ip = nodes.IntPoint(j);
      values(j, 0) = 1.0;
      values(j, 3) = ((ip.x - 0.5) * (ip.x - 0.5) +
                     (ip.y - 0.5) * (ip.y - 0.5) - 0.02) / 0.4;
    }
    state.SetSubVector(vdofs, element);
  }
  const auto limiter = Compare(cache, fes, state, PointSet::Limiter,
                               failures, 6, {5});
  const auto output = Compare(cache, fes, state, PointSet::Output,
                              failures, 6, {5});
  const auto dof = Compare(cache, fes, state, PointSet::Dof,
                           failures, 6, {5});
  Check(limiter.global_bad_pressure == 0 && output.global_bad_pressure > 0 &&
        dof.global_bad_pressure > 0, "point-set-specific violation is retained",
        fes.GetComm(), failures);
  // Public rule data can change without any mesh/FES sequence change. The
  // centre is the only inadmissible point of this manufactured Output rule.
  // Restore the complete integration point before leaving this fixture.
  auto& changed_rule = const_cast<mfem::IntegrationRule&>(
    mfem::IntRules.Get(mfem::Geometry::SQUARE, 8));
  int centre = -1;
  for (int q = 0; q < changed_rule.GetNPoints(); ++q)
  {
    const auto& ip = changed_rule.IntPoint(q);
    if (ip.x == 0.5 && ip.y == 0.5) { centre = q; }
  }
  MFEM_VERIFY(centre >= 0, "Output rule must contain the manufactured centre.");
  const mfem::IntegrationPoint saved_point = changed_rule.IntPoint(centre);
  changed_rule.IntPoint(centre).x = 0.8;
  const auto changed_output = Compare(cache, fes, state, PointSet::Output,
                                      failures, 6, {5});
  changed_rule.IntPoint(centre) = saved_point;
  const auto restored_output = Compare(cache, fes, state, PointSet::Output,
                                       failures, 6, {5});
  Check(changed_output.global_bad_pressure == 0 &&
        restored_output.global_bad_pressure == output.global_bad_pressure,
        "reference-rule changes invalidate a warm plan without a mesh update",
        fes.GetComm(), failures);
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementVDofs(e, vdofs);
    mfem::Vector element(vdofs.Size());
    element = 0.0;
    const auto& nodes = fes.GetFE(e)->GetNodes();
    mfem::DenseMatrix values(element.GetData(), nodes.GetNPoints(), 4);
    for (int j = 0; j < nodes.GetNPoints(); ++j)
    {
      values(j, 0) = nodes.IntPoint(j).x;
      values(j, 3) = 2.5;
    }
    state.SetSubVector(vdofs, element);
  }
  const auto face_only = Compare(cache, fes, state, PointSet::Limiter,
                                 failures, 6, {5}, 1.4, 0.02);
  Check(face_only.global_bad_rho > 0 &&
        face_only.first_local_issue.face_point &&
        face_only.first_local_issue.face >= 0 &&
        face_only.first_local_issue.scalar_dof == -1,
        "face-only failure retains original ordered face metadata",
        fes.GetComm(), failures);
}

void CheckInvalidStates(int& failures)
{
  auto mesh = MakeMesh(2, mfem::Mpi::WorldSize());
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(3, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector state(fes.GetVSize());
  Cache cache;
  int global_dofs = fes.GetNDofs();
  MPI_Allreduce(MPI_IN_PLACE, &global_dofs, 1, MPI_INT, MPI_SUM, fes.GetComm());
  for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity()})
  {
    for (int component = 0; component < 4; ++component)
    {
      SetConstant(fes, state);
      double* data = state.HostReadWrite();
      for (int d = 0; d < fes.GetNDofs(); ++d)
      { data[d + component * fes.GetNDofs()] = value; }
      CompareAll(cache, fes, state, failures);
      const auto scan = Compare(cache, fes, state, PointSet::Dof, failures);
      Check(scan.global_bad_pressure == global_dofs &&
            scan.global_bad_rho == (component == 0 ? global_dofs : 0),
            "nonfinite components increment original thermodynamic counters",
            fes.GetComm(), failures);
    }
  }
  SetConstant(fes, state, 1.0, std::numeric_limits<double>::max());
  const auto overflow = Compare(cache, fes, state, PointSet::Dof, failures,
                                0, {}, 3.0);
  Check(overflow.global_bad_rho == 0 &&
        overflow.global_bad_pressure == global_dofs &&
        overflow.global_min_pressure == std::numeric_limits<double>::infinity(),
        "derived pressure overflow rejects and leaves no finite minimum",
        fes.GetComm(), failures);
  SetConstant(fes, state);
  for (int d = 0; d < fes.GetNDofs(); ++d)
  { state.HostReadWrite()[d + fes.GetNDofs()] = std::numeric_limits<double>::max(); }
  const auto kinetic_overflow = Compare(cache, fes, state, PointSet::Dof,
                                        failures);
  Check(kinetic_overflow.global_bad_rho == 0 &&
        kinetic_overflow.global_bad_pressure == global_dofs &&
        kinetic_overflow.global_min_pressure == std::numeric_limits<double>::infinity(),
        "finite momentum whose squared norm overflows is invalid",
        fes.GetComm(), failures);
  SetConstant(fes, state);
  state.HostReadWrite()[0] = std::numeric_limits<double>::quiet_NaN();
  const auto mixed = Compare(cache, fes, state, PointSet::Dof, failures);
  Check(mixed.global_min_rho == 1.0 && mixed.global_bad_rho == mfem::Mpi::WorldSize() &&
        mixed.global_bad_pressure == mfem::Mpi::WorldSize() &&
        std::isfinite(mixed.global_min_pressure),
        "finite samples determine minima even when other samples are nonfinite",
        fes.GetComm(), failures);
  SetConstant(fes, state, 0.125, 2.0);
  const auto equal = Compare(cache, fes, state, PointSet::Dof, failures,
                             0, {}, 1.5, 0.125, 1.0);
  Check(equal.global_bad_rho == 0 && equal.global_bad_pressure == 0,
        "density and pressure equality remain admissible", fes.GetComm(), failures);
  SetConstant(fes, state, std::nextafter(0.125, 0.0),
               std::nextafter(2.0, 0.0));
  const auto below = Compare(cache, fes, state, PointSet::Dof, failures,
                             0, {}, 1.5, 0.125, 1.0);
  Check(below.global_bad_rho == global_dofs &&
        below.global_bad_pressure == global_dofs &&
        !below.first_local_issue.pressure_bad,
        "strict threshold and density-first issue priority",
        fes.GetComm(), failures);
  // A nonpositive density still uses the original fixed pressure floor.
  SetConstant(fes, state, -0.125, 2.0);
  state.HostReadWrite()[fes.GetNDofs()] = 0.25;
  CompareAll(cache, fes, state, failures);
}

void CheckDeviceAuthority(int& failures)
{
  for (const int basis : {mfem::BasisType::GaussLegendre,
                          mfem::BasisType::GaussLobatto})
  {
    auto mesh = MakeMesh(2, mfem::Mpi::WorldSize());
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::DG_FECollection fec(5, 2, basis);
    mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
    mfem::Vector state(fes.GetVSize());
    state.UseDevice(true);
    SetConstant(fes, state);
    Cache cache;
    CompareAll(cache, fes, state, failures);
    const int ndofs = fes.GetNDofs();
    int global_dofs = ndofs;
    MPI_Allreduce(MPI_IN_PLACE, &global_dofs, 1, MPI_INT, MPI_SUM, fes.GetComm());
    for (const auto points : {PointSet::Dof, PointSet::Limiter, PointSet::Output})
    {
      for (int update = 0; update < 3; ++update)
      {
        const double rho = update == 0 ? 0.125 : 1.0;
        const double invalid = std::numeric_limits<double>::quiet_NaN();
        if (mfem::Device::Allows(mfem::Backend::CUDA))
        {
#if defined(IGRLAB_ADMISSIBILITY_CUDA_PROBE)
          admissibility_cuda_probe::FillDeviceState(state, ndofs, rho,
                                                    update == 2);
#else
          MFEM_ABORT("CUDA authority fixture requires its CUDA helper.");
#endif
        }
        else
        {
          double* data = state.HostWrite();
          for (int d = 0; d < ndofs; ++d)
          {
            data[d] = rho;
            data[d + ndofs] = update == 2 && d == 0 ? invalid : 0.0;
            data[d + 2 * ndofs] = -0.0;
            data[d + 3 * ndofs] = 2.5;
          }
        }
        if (mfem::Device::Allows(mfem::Backend::CUDA))
        {
          Check(!state.GetMemory().HostIsValid(),
                "candidate-first scan starts with fresh device authority", fes.GetComm(), failures);
        }
        // The candidate must perform the first synchronization. Calling the
        // original first would hide an invalid host pointer or stale gather.
        const auto scan = cache.Scan(fes, state, 2, 1.4, 0.25, 1.0e-10, 0, {}, points);
        const auto expected = mfem::igrlab::ScanEulerAdmissibility(
          fes, state, 2, 1.4, 0.25, 1.0e-10, 0, {}, points);
        Check(SameSummary(scan, expected),
              "candidate-first changed same-vector scan retains original bits/diagnostics",
              fes.GetComm(), failures);
        if (update == 0 && points == PointSet::Dof)
        {
          Check(scan.global_bad_rho == global_dofs && scan.global_min_rho == 0.125,
                "fresh device-authoritative Dof density retains exact count and minimum",
                fes.GetComm(), failures);
        }
        Check(update == 0 ? scan.global_bad_rho > 0 :
              (update == 1 ? scan.global_bad_rho == 0 && scan.global_bad_pressure == 0 :
                             scan.global_bad_pressure > 0),
              "fresh density/signed-zero/nonfinite updates reach every point set",
              fes.GetComm(), failures);
      }
    }
  }
}

void CheckSignedHostGather(int& failures)
{
  auto mesh = MakeMesh(2, mfem::Mpi::WorldSize());
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(3, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector state(fes.GetVSize());
  SetConstant(fes, state);
  state.HostReadWrite()[fes.GetNDofs()] = -0.0;
  state.HostReadWrite()[2 * fes.GetNDofs()] = std::numeric_limits<double>::quiet_NaN();
  mfem::Array<int> signs(state.Size());
  for (int i = 0; i < signs.Size(); ++i) { signs[i] = i; }
  std::vector<mfem::igrlab::detail::AdmissibilityElementMap> maps(fes.GetNE());
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    auto& map = maps[e];
    map.fe = fes.GetFE(e); map.dof = map.fe->GetDof();
    map.order = map.fe->GetOrder(); map.enforce_nodes = true;
    fes.GetElementVDofs(e, map.vdofs);
    for (int j = 0; j < map.vdofs.Size(); ++j)
    {
      const int index = map.vdofs[j];
      MFEM_VERIFY(index >= 0, "Directed signed fixture starts from ordinary L2 maps.");
      if (j % 2 == 0) { map.vdofs[j] = signs[index] = -1 - index; }
    }
  }
  // Exercise the existing private scan entry with directed encoded maps. The
  // reference uses MFEM's original signed gather to transform the input, then
  // its original scanner and GetRow; no production test hook is introduced.
  const auto actual = mfem::igrlab::detail::ScanEulerAdmissibilityImpl(
    fes, state, 2, 1.4, 1.0e-10, 1.0e-10, 0, {}, PointSet::Dof,
    &maps, nullptr, nullptr);
  mfem::Vector signed_state;
  state.GetSubVector(signs, signed_state);
  const auto expected = mfem::igrlab::ScanEulerAdmissibility(
    fes, signed_state, 2, 1.4, 1.0e-10, 1.0e-10, 0, {}, PointSet::Dof);
  Check(SameSummary(actual, expected) && actual.global_bad_rho > 0,
        "cached host gather preserves encoded signs, signed zero and nonfinite diagnostics",
        fes.GetComm(), failures);
}

void CheckCommunicators(int& failures)
{
  const int rank = mfem::Mpi::WorldRank();
  const int size = mfem::Mpi::WorldSize();
  auto mesh = MakeMesh(2, size);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(1, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector state(fes.GetVSize());
  SetConstant(fes, state);
  if (rank == size - 1) { state.HostReadWrite()[0] = -1.0; }
  Cache cache;
  const auto scan = Compare(cache, fes, state, PointSet::Dof, failures);
  Check(scan.global_bad_rho == 1 && scan.first_bad_rank == size - 1 &&
        scan.first_local_issue.found == (rank == size - 1),
        "minimum offending rank and local issue remain distinct",
        fes.GetComm(), failures);

  MPI_Comm subcomm = MPI_COMM_NULL;
  const int color = size > 1 ? rank % 2 : 1;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm);
  {
    int subsize = 1;
    MPI_Comm_size(subcomm, &subsize);
    auto submesh = MakeMesh(2, subsize);
    mfem::ParMesh parallel(subcomm, submesh);
    mfem::ParFiniteElementSpace subfes(&parallel, &fec, 4,
                                       mfem::Ordering::byNODES);
    mfem::Vector substate(subfes.GetVSize());
    SetConstant(subfes, substate, color == 0 ? -1.0 : 1.0);
    const auto subscan = Compare(cache, subfes, substate, PointSet::Limiter,
                                 failures);
    Check(color == 0 ? subscan.global_bad_rho > 0 && subscan.first_bad_rank == 0 :
          subscan.global_bad_rho == 0 && subscan.first_bad_rank == -1,
          "reductions use the finite-element-space subcommunicator",
          subcomm, failures);
  }
  MPI_Comm_free(&subcomm);
}

#if defined(IGRLAB_ADMISSIBILITY_CUDA_PROBE)
namespace probe = admissibility_cuda_probe;

class ScopedProductionCudaFlag
{
public:
  explicit ScopedProductionCudaFlag(const char* value)
  {
    const char* previous = std::getenv("IGRLAB_EULER_ADMISSIBILITY_CUDA");
    present_ = previous != nullptr;
    if (previous) { value_ = previous; }
    MFEM_VERIFY(setenv("IGRLAB_EULER_ADMISSIBILITY_CUDA", value, 1) == 0,
                "Cannot select the test-only device scan contract.");
  }
  ~ScopedProductionCudaFlag()
  {
    if (present_) { setenv("IGRLAB_EULER_ADMISSIBILITY_CUDA", value_.c_str(), 1); }
    else { unsetenv("IGRLAB_EULER_ADMISSIBILITY_CUDA"); }
  }
private:
  bool present_ = false;
  std::string value_;
};

bool ProbeEnvironmentSupported()
{
#if defined(__SSE2__) && !defined(__FAST_MATH__)
  const unsigned int csr = _mm_getcsr();
  return std::numeric_limits<double>::is_iec559 && sizeof(double) == 8 &&
    std::fegetround() == FE_TONEAREST && (csr & 0xe040u) == 0 &&
    (csr & 0x1f80u) == 0x1f80u;
#else
  return false;
#endif
}

struct ProbePlan
{
  std::vector<int> maps;
  std::vector<double> shapes;
  std::vector<probe::Sample> samples;
};

ProbePlan MakeProbePlan(mfem::ParFiniteElementSpace& fes, PointSet points,
                        int qorder = 0,
                        const std::vector<int>& face_orders = {})
{
  ProbePlan plan;
  const int order = fes.GetFE(0)->GetOrder();
  const int base = qorder > 0 ? qorder : 2 * order + 2;
  const int scan_order = points == PointSet::Output ?
    std::max(base + 2, 2 * order + 4) : base;
  auto face_qorders = mfem::igrlab::detail::BuildFaceQuadratureOrdersForAdmissibility(
    fes, scan_order, face_orders);
  if (points == PointSet::Output)
  {
    for (int& q : face_qorders) { q = std::max(q + 2, 2 * order + 3); }
    std::sort(face_qorders.begin(), face_qorders.end());
    face_qorders.erase(std::unique(face_qorders.begin(), face_qorders.end()),
                      face_qorders.end());
  }
  const auto& faces = mfem::igrlab::detail::GetElementFacesForAdmissibility(
    *fes.GetMesh());
  const auto* l2 = dynamic_cast<const mfem::L2_FECollection*>(fes.FEColl());
  MFEM_VERIFY(l2 && fes.GetVDim() == 4 && fes.GetMesh()->Dimension() == 2,
              "The test prototype expects a 2D Euler L2 space.");
  const bool nodes = l2->GetBasisType() == mfem::BasisType::GaussLobatto;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    const auto* fe = fes.GetFE(e);
    const int dof = fe->GetDof();
    const int map_offset = static_cast<int>(plan.maps.size());
    mfem::Array<int> vdofs;
    fes.GetElementVDofs(e, vdofs);
    plan.maps.insert(plan.maps.end(), vdofs.HostRead(),
                     vdofs.HostRead() + vdofs.Size());
    if (points == PointSet::Dof || nodes)
    {
      for (int j = 0; j < dof; ++j)
      { plan.samples.push_back({dof, map_offset, -1, j, e, -1, j}); }
    }
    if (points == PointSet::Dof) { continue; }
    const auto append = [&](const mfem::IntegrationPoint& point, int face, int q)
    {
      mfem::Vector shape(dof);
      fe->CalcShape(point, shape);
      const int offset = static_cast<int>(plan.shapes.size());
      plan.shapes.insert(plan.shapes.end(), shape.HostRead(),
                        shape.HostRead() + dof);
      plan.samples.push_back({dof, map_offset, offset, -1, e, face, q});
    };
    auto* transform = fes.GetElementTransformation(e);
    const auto& volume = mfem::IntRules.Get(transform->GetGeometryType(), scan_order);
    for (int q = 0; q < volume.GetNPoints(); ++q)
    { append(volume.IntPoint(q), -1, q); }
    for (int face : faces[e])
    {
      auto* ft = fes.GetMesh()->GetFaceElementTransformations(face);
      if (!ft) { continue; }
      int e1 = -1, e2 = -1;
      fes.GetMesh()->GetFaceElements(face, &e1, &e2);
      const bool first = e1 == e;
      if (!first && e2 != e) { continue; }
      for (int face_order : face_qorders)
      {
        const auto& rule = mfem::IntRules.Get(ft->GetGeometryType(), face_order);
        for (int q = 0; q < rule.GetNPoints(); ++q)
        {
          ft->SetAllIntPoints(&rule.IntPoint(q));
          append(first ? ft->GetElement1IntPoint() : ft->GetElement2IntPoint(),
                 face, q);
        }
      }
    }
  }
  return plan;
}

probe::Value OriginalSample(const ProbePlan& plan, const probe::Sample& sample,
                            const mfem::Vector& state, double gamma,
                            double rho_min, double p_min)
{
  mfem::Array<int> map(4 * sample.dof);
  std::copy_n(plan.maps.data() + sample.map_offset, map.Size(), map.HostWrite());
  mfem::Vector element;
  state.GetSubVector(map, element);
  mfem::DenseMatrix matrix(element.GetData(), sample.dof, 4);
  mfem::Vector value(4);
  if (sample.shape_offset < 0) { matrix.GetRow(sample.node, value); }
  else
  {
    // This calls the original compiled MFEM host pointer/pointer routine.
    matrix.MultTranspose(plan.shapes.data() + sample.shape_offset, value.HostWrite());
  }
  probe::Value out{};
  std::copy_n(value.HostRead(), 4, out.state);
  bool finite = true;
  for (int eq = 0; eq < 4; ++eq) { finite = finite && std::isfinite(value(eq)); }
  double momentum_squared = 0.0;
  for (int d = 0; d < 2; ++d)
  { momentum_squared += value(1 + d) * value(1 + d); }
  out.kinetic = 0.5 * momentum_squared / std::max(value(0), 1.0e-16);
  out.pressure = mfem::EulerPressureFromConservative(value, 2, gamma, 1.0e-16);
  out.bad_rho = !std::isfinite(value(0)) || value(0) < rho_min;
  out.bad_pressure = !finite || !std::isfinite(out.kinetic) ||
    !std::isfinite(out.pressure) || out.pressure < p_min;
  return out;
}

Summary ProbeGlobalSummary(const ProbePlan& plan, const probe::Result& result,
                           MPI_Comm comm)
{
  MFEM_VERIFY(!result.summary.declined && !result.environment_declined &&
                !result.sticky_flags_declined,
              "A declined prototype cannot publish partial summaries.");
  Summary out;
  double rho = result.summary.min_rho, pressure = result.summary.min_pressure;
  int bad_rho = result.summary.bad_rho, bad_pressure = result.summary.bad_pressure;
  MPI_Allreduce(&rho, &out.global_min_rho, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&pressure, &out.global_min_pressure, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&bad_rho, &out.global_bad_rho, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&bad_pressure, &out.global_bad_pressure, 1, MPI_INT, MPI_SUM, comm);
  if (result.summary.first_sample >= 0)
  {
    const int index = result.summary.first_sample;
    const auto& sample = plan.samples[index];
    const auto& value = result.values[index];
    out.first_local_issue = {true, !value.bad_rho && value.bad_pressure,
      sample.face >= 0, sample.element, sample.face, sample.point,
      sample.shape_offset < 0 ? sample.node : -1, value.state[0], value.pressure};
  }
  out.first_bad_rank = mfem::igrlab::detail::FindFirstBadRank(
    out.first_local_issue.found, comm);
  return out;
}

void CheckCudaArithmeticPrototype(int& failures)
{
  if (!mfem::Device::Allows(mfem::Backend::CUDA)) { return; }
  Check(ProbeEnvironmentSupported(),
        "native thread admits the bounded nearest-even prototype", MPI_COMM_WORLD,
        failures);
  if (!ProbeEnvironmentSupported()) { return; }
  int admitted_cases = 0, decline_cases = 0;
  for (int basis : {mfem::BasisType::GaussLegendre, mfem::BasisType::GaussLobatto})
  {
    for (int order : {4, 5}) // Odd/even original dot lengths, not a performance sweep.
    {
      auto mesh = MakeMesh(2, mfem::Mpi::WorldSize());
      mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mfem::DG_FECollection fec(order, 2, basis);
      mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
      mfem::Vector state(fes.GetVSize()); state.UseDevice(true);
      for (auto points : {PointSet::Dof, PointSet::Limiter, PointSet::Output})
      {
        // Distinct rule occurrences must not disappear when points coincide.
        const std::vector<int> face_orders{7, 7, -3, 9, 10};
        const auto plan = MakeProbePlan(fes, points, 0, face_orders);
        for (int update = 0; update < 3; ++update)
        {
          std::vector<double> input(state.Size());
          const int n = fes.GetNDofs();
          for (int j = 0; j < n; ++j)
          {
            input[j] = update == 1 ? -0.125 :
              (update == 2 ? 0.125 : 1.0 + 0.001 * (j + 1));
            input[j + n] = update == 2 ? -0.0 : 0.02 + 0.0003 * j;
            input[j + 2 * n] = update == 2 ? 0.0 : -0.015 - 0.0002 * j;
            input[j + 3 * n] = update == 2 ? 2.0 : 2.5 + 0.002 * j;
          }
          probe::WriteDeviceState(state, input);
          Check(!state.GetMemory().HostIsValid(),
                "prototype begins before original host synchronization", fes.GetComm(), failures);
          const double gamma = update == 2 ? 1.5 : 1.4;
          const double rho_min = update == 2 ? 0.125 : 1.0e-10;
          const double p_min = update == 2 ? 1.0 : 1.0e-10;
          const auto actual = probe::Evaluate(state, plan.maps, plan.shapes, plan.samples,
            gamma, rho_min, p_min, ProbeEnvironmentSupported());
          Check(!state.GetMemory().HostIsValid(),
                "test CUDA evaluator leaves conservative state device-authoritative",
                fes.GetComm(), failures);
          const auto expected = mfem::igrlab::ScanEulerAdmissibility(
            fes, state, 2, gamma, rho_min, p_min, 0, face_orders, points);
          Check(!actual.summary.declined, "ordinary p4/p5 proof cases are evaluated",
                fes.GetComm(), failures);
          if (actual.summary.declined) { ++decline_cases; continue; }
          bool same_values = actual.values.size() == plan.samples.size();
          for (std::size_t i = 0; i < actual.values.size(); ++i)
          {
            const auto reference = OriginalSample(plan, plan.samples[i], state,
                                                  gamma, rho_min, p_min);
            const auto& candidate = actual.values[i];
            same_values = same_values &&
              std::memcmp(reference.state, candidate.state, sizeof(reference.state)) == 0 &&
              SameDouble(reference.kinetic, candidate.kinetic) &&
              SameDouble(reference.pressure, candidate.pressure) &&
              reference.bad_rho == candidate.bad_rho &&
              reference.bad_pressure == candidate.bad_pressure;
          }
          Check(same_values, "CUDA samplewise original dot/EOS bits: p=" +
            std::to_string(order) + " update=" + std::to_string(update),
            fes.GetComm(), failures);
          Check(SameSummary(ProbeGlobalSummary(plan, actual, fes.GetComm()), expected),
                "ordered CUDA summary retains original counts/minima/first issue",
                fes.GetComm(), failures);
          ++admitted_cases;
        }

        // Mixed exceptional data must discard every provisional GPU summary.
        // The unchanged host routine then retains its complete issue/minimum
        // semantics; the ordinary host cache is an independent parity check.
        for (double exceptional : {std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::max()})
        {
          std::vector<double> input(state.Size(), 0.0);
          const int n = fes.GetNDofs();
          for (int j = 0; j < n; ++j)
          { input[j] = 1.0; input[j + 3 * n] = 2.5; }
          input[n] = exceptional;
          probe::WriteDeviceState(state, input);
          const auto rejected = probe::Evaluate(state, plan.maps, plan.shapes,
            plan.samples, 1.4, 1.0e-10, 1.0e-10, ProbeEnvironmentSupported());
          Check(rejected.summary.declined != 0,
                "mixed exceptional state declines the whole test call",
                fes.GetComm(), failures);
          const auto fallback = mfem::igrlab::ScanEulerAdmissibility(
            fes, state, 2, 1.4, 1.0e-10, 1.0e-10, 0, face_orders, points);
          Cache host_cache;
          const auto cached = host_cache.Scan(fes, state, 2, 1.4, 1.0e-10,
                                              1.0e-10, 0, face_orders, points);
          Check(SameSummary(fallback, cached),
                "discarded prototype uses the original complete host summary",
                fes.GetComm(), failures);
          ++decline_cases;
        }
      }
    }
  }

  // Directed arithmetic fixtures do not assume normal inputs imply normal
  // products, ordered partial sums, or EOS intermediates.
  ProbePlan directed;
  directed.maps = {0, 1, 2, 3, 4, 5, 6, 7};
  directed.shapes = {1.0, 1.0};
  directed.samples = {{2, 0, 0, -1, 0, -1, 0}};
  mfem::Vector state(8); state.UseDevice(true);
  const auto expect_decline = [&](std::vector<double> input,
                                   std::vector<double> shapes,
                                   const std::string& label)
  {
    probe::WriteDeviceState(state, input);
    const auto actual = probe::Evaluate(state, directed.maps, shapes, directed.samples,
                                        1.4, 1.0e-10, 1.0e-10, true);
    Check(actual.summary.declined != 0, label, MPI_COMM_WORLD, failures);
    ++decline_cases;
  };
  expect_decline({0x1p-1022, 1.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0},
                 {0.5, 0.0}, "normal operands/subnormal product decline");
  expect_decline({0x1p-1022, 1.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0},
                 {0x1p-1022, 0.0}, "normal operands/product rounded to zero decline");
  expect_decline({0x1.0000000000001p-1022, -0x1p-1022, 0.0, 0.0,
                  0.0, 0.0, 2.5, 0.0}, {1.0, 1.0},
                 "normal operands/subnormal ordered sum decline");
  expect_decline({std::numeric_limits<double>::max(), 1.0, 0.0, 0.0,
                  0.0, 0.0, 2.5, 0.0}, {2.0, 0.0},
                 "normal operands/overflow decline");
  expect_decline({1.0, 0.0, 0x1p-1022, 0.0, 0.0, 0.0, 2.5, 0.0},
                 {1.0, 0.0}, "normal momentum/EOS product rounded to zero decline");
  expect_decline({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0x1p-1022, 0.0},
                 {1.0, 0.0}, "normal energy/subnormal EOS pressure decline");
  expect_decline({std::numeric_limits<double>::max(), 0.0, 1.0, 0.0,
                  0.0, 0.0, 2.5, 0.0}, {1.0, 0.0},
                 "normal state/subnormal EOS division decline");
  for (double exceptional : {std::numeric_limits<double>::denorm_min(),
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()})
  {
    expect_decline({exceptional, 1.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0},
                   {1.0, 0.0}, "subnormal/nonfinite input decline");
  }
  // Unlike an FMA, the original -1 + round(a*b) is exactly +0.
  std::vector<double> cancellation{-1.0, 0x1.0000002p0, 0.0, 0.0,
                                    0.0, 0.0, 3.0, 0.0};
  directed.shapes = {1.0, 0x1.ffffffcp-1};
  probe::WriteDeviceState(state, cancellation);
  const auto separate = probe::Evaluate(state, directed.maps, directed.shapes,
    directed.samples, 1.4, 1.0e-10, 1.0e-10, true);
  const auto host_value = OriginalSample(directed, directed.samples[0], state,
                                          1.4, 1.0e-10, 1.0e-10);
  Check(!separate.summary.declined && SameDouble(separate.values[0].state[0],
        host_value.state[0]) && SameDouble(host_value.state[0], 0.0),
        "FMA-sensitive cancellation follows original separate products/adds",
        MPI_COMM_WORLD, failures);
  ++admitted_cases;

  ProbePlan signed_nodes = directed;
  signed_nodes.maps[0] = -1; // Encode negation of an input +0, not abs(index).
  signed_nodes.samples = {{2, 0, -1, 0, 0, -1, 0},
                           {2, 0, -1, 1, 0, -1, 1}};
  probe::WriteDeviceState(state, {0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 2.5, 2.5});
  const auto signed_result = probe::Evaluate(state, signed_nodes.maps,
    signed_nodes.shapes, signed_nodes.samples, 1.4, 1.0e-10, 1.0e-10, true);
  bool signed_same = !signed_result.summary.declined &&
    SameDouble(signed_result.summary.min_rho, -0.0) &&
    signed_result.summary.first_sample == 0 && signed_result.summary.bad_rho == 2;
  for (std::size_t i = 0; i < signed_nodes.samples.size(); ++i)
  {
    const auto expected = OriginalSample(signed_nodes, signed_nodes.samples[i],
                                          state, 1.4, 1.0e-10, 1.0e-10);
    const auto& actual = signed_result.values[i];
    signed_same = signed_same &&
      std::memcmp(expected.state, actual.state, sizeof(expected.state)) == 0 &&
      SameDouble(expected.pressure, actual.pressure);
  }
  Check(signed_same, "signed maps and first signed-zero minimum retain original bits",
        MPI_COMM_WORLD, failures);
  ++admitted_cases;

  const auto sticky = probe::Evaluate(state, directed.maps, directed.shapes,
    directed.samples, 1.4, 1.0e-10, 1.0e-10, true, true);
  Check(sticky.sticky_flags_declined && sticky.values.empty(),
        "arbitrary sticky exception equivalence is explicitly unsupported",
        MPI_COMM_WORLD, failures);
  ++decline_cases;
  std::fenv_t environment;
  MFEM_VERIFY(std::fegetenv(&environment) == 0,
              "Cannot save the native test floating environment.");
#if defined(__SSE2__)
  const unsigned int original_csr = _mm_getcsr();
#endif
  MFEM_VERIFY(std::fesetround(FE_DOWNWARD) == 0,
              "Cannot select the directed test rounding mode.");
  const auto rounded = probe::Evaluate(state, directed.maps, directed.shapes,
    directed.samples, 1.4, 1.0e-10, 1.0e-10, ProbeEnvironmentSupported());
  const bool rounding_declined = rounded.environment_declined && rounded.values.empty();
  std::fesetenv(&environment);
#if defined(__SSE2__)
  _mm_setcsr(original_csr | 0x8040u); // Test-only FTZ/DAZ change, restored below.
  const auto flushed = probe::Evaluate(state, directed.maps, directed.shapes,
    directed.samples, 1.4, 1.0e-10, 1.0e-10, ProbeEnvironmentSupported());
  const bool flushing_declined = flushed.environment_declined && flushed.values.empty();
  // Earlier exceptional host oracles may have set sticky status bits. Clear
  // those only in this temporary fixture before unmasking invalid traps.
  _mm_setcsr(original_csr & ~(0x3fu | 0x80u));
  const auto trapping = probe::Evaluate(state, directed.maps, directed.shapes,
    directed.samples, 1.4, 1.0e-10, 1.0e-10, ProbeEnvironmentSupported());
  const bool trapping_declined = trapping.environment_declined && trapping.values.empty();
  _mm_setcsr(original_csr);
  Check(flushing_declined && trapping_declined,
        "FTZ/DAZ and unmasked trap environments decline before CUDA arithmetic",
        MPI_COMM_WORLD, failures);
  decline_cases += 2;
#endif
  Check(rounding_declined && ProbeEnvironmentSupported(),
        "non-nearest environment declines and caller environment is restored",
        MPI_COMM_WORLD, failures);
  ++decline_cases;
  if (mfem::Mpi::Root())
  {
    std::cout << "[admissibility-cuda-prototype] scope=test-only"
              << " admitted_cases=" << admitted_cases
              << " decline_cases=" << decline_cases
              << " production_enabled=0 portable_arithmetic_claim=0"
              << " sticky_exception_equivalence=unsupported\n";
  }
}

void CheckProductionCudaBudget(int& failures)
{
  namespace detail = mfem::igrlab::detail;
  std::vector<detail::AdmissibilityElementMap> elements(1);
  elements[0].dof = 1;
  elements[0].vdofs.SetSize(4);
  for (int j = 0; j < 4; ++j) { elements[0].vdofs[j] = j; }
  std::vector<detail::AdmissibilityShapeTable> shapes(2);
  for (auto& shape : shapes)
  { shape.dof = 1; shape.points = 1; shape.values = {1.0}; }
  std::vector<detail::AdmissibilityElementSamples> samples(1);
  samples[0].volume_shape_table = 0;
  samples[0].face_rules.push_back({0, 0}); // Repeated table, distinct occurrence.
  constexpr std::size_t budget = 64u * 1024u * 1024u;
  for (const auto points : {PointSet::Dof, PointSet::Limiter})
  {
    std::size_t charged = 0;
    auto plan = detail::MakeAdmissibilityDevicePlan(
      elements, shapes, samples, points, 4, budget, charged);
    Check(plan && charged > 0 && charged <= budget,
          "CUDA plan reports a bounded peak construction charge", MPI_COMM_WORLD, failures);
    if (!plan || charged == 0 || charged > budget) { continue; }
    plan.reset();
    std::size_t offered_charge = 123;
    plan = detail::MakeAdmissibilityDevicePlan(
      elements, shapes, samples, points, 4, charged - 1, offered_charge);
    Check(!plan && offered_charge == 0,
          "CUDA plan declines one byte below its peak budget without charging",
          MPI_COMM_WORLD, failures);
    plan = detail::MakeAdmissibilityDevicePlan(
      elements, shapes, samples, points, 4, charged, offered_charge);
    Check(plan && offered_charge == charged,
          "CUDA plan accepts its exact peak budget", MPI_COMM_WORLD, failures);
    plan.reset();
    if (points == PointSet::Dof)
    {
      plan = detail::MakeAdmissibilityDevicePlan(
        elements, {}, {}, points, 4, charged, offered_charge);
      Check(plan && offered_charge == charged,
            "Dof plan peak budget excludes unrelated cached shape tables",
            MPI_COMM_WORLD, failures);
    }
  }
}

void CheckProductionCudaElementFold(int& failures)
{
  // Cross both a warp boundary and the final fold block's strided tail. An
  // earlier element's last node must beat a later element's first node, and
  // equal minima must retain the earlier element's sign of zero.
  auto mesh = mfem::Mesh::MakeCartesian2D(
    257, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(1, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector state(fes.GetVSize()); state.UseDevice(true);
  Cache cache;
  const int n = fes.GetNDofs();
  for (int component : {0, 3})
  {
    std::vector<double> input(state.Size(), 0.0);
    for (int j = 0; j < n; ++j)
    { input[j] = 1.0; input[j + 3 * n] = 2.5; }
    mfem::Array<int> dofs;
    for (int e : {127, 128, 256})
    {
      fes.GetElementDofs(e, dofs);
      const int local = e == 127 ? dofs.Size() - 1 : 0;
      const int encoded = dofs[local];
      const int decoded = encoded >= 0 ? encoded : -1 - encoded;
      const double value = e == 128 ? 0.0 : -0.0;
      input[decoded + component * n] = encoded >= 0 ? value : -value;
    }
    probe::WriteDeviceState(state, input);
    const auto actual = cache.Scan(
      fes, state, 2, 1.5, 0.1, 0.1, 0, {}, PointSet::Dof);
    Check(!state.GetMemory().HostIsValid(),
          "strided cross-element fold stays device resident",
          fes.GetComm(), failures);
    const auto expected = mfem::igrlab::ScanEulerAdmissibility(
      fes, state, 2, 1.5, 0.1, 0.1, 0, {}, PointSet::Dof);
    Check(SameSummary(actual, expected) &&
          actual.first_local_issue.element == 127 &&
          actual.first_local_issue.scalar_dof == fes.GetFE(127)->GetDof() - 1 &&
          SameDouble(component == 0 ? actual.global_min_rho :
                     actual.global_min_pressure, -0.0),
          "strided fold preserves element-first issues, signed minima and counts",
          fes.GetComm(), failures);
    // A declined value in the final partial stride must invalidate the whole
    // device result, even when an earlier element already supplies an issue.
    fes.GetElementDofs(256, dofs);
    const int encoded = dofs[dofs.Size() - 1];
    const int decoded = encoded >= 0 ? encoded : -1 - encoded;
    input[decoded + n] = std::numeric_limits<double>::denorm_min();
    probe::WriteDeviceState(state, input);
    const auto declined = cache.Scan(
      fes, state, 2, 1.5, 0.1, 0.1, 0, {}, PointSet::Dof);
    Check(state.GetMemory().HostIsValid(),
          "declined arithmetic in final fold stride triggers complete host fallback",
          fes.GetComm(), failures);
    const auto original = mfem::igrlab::ScanEulerAdmissibility(
      fes, state, 2, 1.5, 0.1, 0.1, 0, {}, PointSet::Dof);
    Check(SameSummary(declined, original),
          "strided declined fold retains original summary",
          fes.GetComm(), failures);
  }
}

void CheckProductionCudaScan(int& failures)
{
  if (!mfem::igrlab::detail::AdmissibilityOrdinaryCuda() ||
      mfem::Mpi::WorldSize() != 1) { return; }
  // The existing CXX/MFEM dot/EOS experiment is the prerequisite, rather than
  // trusting a CUDA implementation that merely agrees with another CUDA one.
  if (failures != 0 || !ProbeEnvironmentSupported()) { return; }
  ScopedProductionCudaFlag enabled("1");
  CheckProductionCudaBudget(failures);
  CheckProductionCudaElementFold(failures);
  int native_calls = 0;
  Cache cache;
  for (const int basis : {mfem::BasisType::GaussLegendre,
                          mfem::BasisType::GaussLobatto})
  {
    for (const int order : {4, 5})
    {
      auto mesh = MakeMesh(2, 1);
      mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mfem::DG_FECollection fec(order, 2, basis);
      mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
      mfem::Vector state(fes.GetVSize()); state.UseDevice(true);
      for (const auto points : {PointSet::Dof, PointSet::Limiter, PointSet::Output})
      {
        for (int update = 0; update < 3; ++update)
        {
          const int n = fes.GetNDofs();
          std::vector<double> input(state.Size());
          for (int j = 0; j < n; ++j)
          {
            input[j] = update == 1 ? -0.125 :
              (update == 2 ? 0.125 : 1.0 + 0.001 * (j + 1));
            input[j + n] = update == 2 ? -0.0 : 0.02 + 0.0003 * j;
            input[j + 2 * n] = update == 2 ? 0.0 : -0.015 - 0.0002 * j;
            input[j + 3 * n] = update == 2 ? 2.0 : 2.5 + 0.002 * j;
          }
          probe::WriteDeviceState(state, input);
          const double gamma = update == 2 ? 1.5 : 1.4;
          const double rho_min = update == 2 ? 0.125 : 1.0e-10;
          const double p_min = update == 2 ? 1.0 : 1.0e-10;
          const std::vector<int> face_orders{2 * order + 1, 2 * order + 1,
                                             -1, 2 * order + 3};
          const auto actual = cache.Scan(fes, state, 2, gamma, rho_min, p_min,
                                         0, face_orders, points);
          Check(!state.GetMemory().HostIsValid(),
                "production current-call CUDA summary avoids conservative HostRead",
                fes.GetComm(), failures);
          const auto expected = mfem::igrlab::ScanEulerAdmissibility(
            fes, state, 2, gamma, rho_min, p_min, 0, face_orders, points);
          Check(SameSummary(actual, expected),
                "production compact CUDA p4/p5 summary preserves original bits/counts/order",
                fes.GetComm(), failures);
          ++native_calls;
        }
      }
      // Unsupported arithmetic discards the whole provisional device result,
      // before executing the original host scan on this same fresh state.
      for (const double exceptional : {std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::denorm_min(),
          std::numeric_limits<double>::max(), std::numeric_limits<double>::min()})
      {
        const int n = fes.GetNDofs();
        std::vector<double> input(state.Size(), 0.0);
        for (int j = 0; j < n; ++j) { input[j] = 1.0; input[j + 3 * n] = 2.5; }
        input[n] = exceptional; // min^2 rounds to zero; max^2 overflows.
        for (const auto points : {PointSet::Dof, PointSet::Limiter, PointSet::Output})
        {
          probe::WriteDeviceState(state, input);
          const auto actual = cache.Scan(fes, state, 2, 1.4, 1.0e-10, 1.0e-10,
                                         0, {}, points);
          Check(state.GetMemory().HostIsValid(),
                "unsupported current arithmetic declines to complete original host scan",
                fes.GetComm(), failures);
          const auto expected = mfem::igrlab::ScanEulerAdmissibility(
            fes, state, 2, 1.4, 1.0e-10, 1.0e-10, 0, {}, points);
          Check(SameSummary(actual, expected),
                "whole-call production fallback retains original exceptional diagnostics",
                fes.GetComm(), failures);
        }
      }
      // First signed-zero minima must come from the original earliest sample.
      const int n = fes.GetNDofs();
      std::vector<double> zeros(state.Size(), 0.0);
      for (int j = 0; j < n; ++j)
      { zeros[j] = j % 2 == 0 ? -0.0 : 0.0; zeros[j + 3 * n] = 2.5; }
      probe::WriteDeviceState(state, zeros);
      const auto actual_zero = cache.Scan(fes, state, 2, 1.0, 0.1, 0.1, 0, {}, PointSet::Dof);
      Check(!state.GetMemory().HostIsValid(), "signed-zero Dof summary uses CUDA", fes.GetComm(), failures);
      const auto expected_zero = mfem::igrlab::ScanEulerAdmissibility(
        fes, state, 2, 1.0, 0.1, 0.1, 0, {}, PointSet::Dof);
      Check(SameSummary(actual_zero, expected_zero),
            "parallel compact min ties retain first signed zero and density-priority issue",
            fes.GetComm(), failures);
      ++native_calls;
    }
  }
  // Reuse the original independent occurrence, rule-mutation, face-only issue,
  // FES/mesh invalidation and generic-fallback fixtures under the new route.
  CheckSpaces(failures);
  CheckSampleContract(failures);
  CheckDeviceAuthority(failures);

  auto mesh = MakeMesh(2, 1);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(5, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector state(fes.GetVSize()); state.UseDevice(true);
  Cache guarded;
  std::fenv_t environment;
  MFEM_VERIFY(std::fegetenv(&environment) == 0, "Cannot save test floating environment.");
#if defined(__SSE2__)
  const unsigned int csr = _mm_getcsr();
#endif
  MFEM_VERIFY(std::fesetround(FE_DOWNWARD) == 0, "Cannot select test rounding.");
  probe::FillDeviceState(state, fes.GetNDofs(), 1.0, false);
  const auto rounded = guarded.Scan(fes, state, 2, 1.4, 1.0e-10, 1.0e-10,
                                    0, {}, PointSet::Dof);
  const bool host_rounding = state.GetMemory().HostIsValid();
  const auto original_rounded = mfem::igrlab::ScanEulerAdmissibility(
    fes, state, 2, 1.4, 1.0e-10, 1.0e-10, 0, {}, PointSet::Dof);
  const bool rounding_equal = SameSummary(rounded, original_rounded);
  Cache unsupported_shapes;
  probe::FillDeviceState(state, fes.GetNDofs(), 1.0, false);
  unsupported_shapes.Scan(fes, state, 2, 1.4, 1.0e-10, 1.0e-10,
                          0, {}, PointSet::Limiter);
  std::fesetenv(&environment);
  probe::FillDeviceState(state, fes.GetNDofs(), 1.0, false);
  const auto unqualified = unsupported_shapes.Scan(fes, state, 2, 1.4,
    1.0e-10, 1.0e-10, 0, {}, PointSet::Output);
  const bool unqualified_host = state.GetMemory().HostIsValid();
  Summary original_cached;
  {
    ScopedProductionCudaFlag disabled("0");
    original_cached = unsupported_shapes.Scan(fes, state, 2, 1.4,
      1.0e-10, 1.0e-10, 0, {}, PointSet::Output);
  }
  Check(unqualified_host && SameSummary(unqualified, original_cached),
        "restored rounding does not qualify shared tables built in unsupported context",
        fes.GetComm(), failures);
#if defined(__SSE2__)
  _mm_setcsr(csr | 0x8040u);
  probe::FillDeviceState(state, fes.GetNDofs(), 1.0, false);
  const auto flushed = guarded.Scan(fes, state, 2, 1.4, 1.0e-10, 1.0e-10,
                                    0, {}, PointSet::Dof);
  const bool host_flushing = state.GetMemory().HostIsValid();
  const auto original_flushed = mfem::igrlab::ScanEulerAdmissibility(
    fes, state, 2, 1.4, 1.0e-10, 1.0e-10, 0, {}, PointSet::Dof);
  const bool flushing_equal = SameSummary(flushed, original_flushed);
  _mm_setcsr(csr & ~(0x3fu | 0x80u));
  probe::FillDeviceState(state, fes.GetNDofs(), 1.0, false);
  const auto trapping = guarded.Scan(fes, state, 2, 1.4, 1.0e-10, 1.0e-10,
                                     0, {}, PointSet::Dof);
  const bool host_trapping = state.GetMemory().HostIsValid();
  const auto original_trapping = mfem::igrlab::ScanEulerAdmissibility(
    fes, state, 2, 1.4, 1.0e-10, 1.0e-10, 0, {}, PointSet::Dof);
  const bool trapping_equal = SameSummary(trapping, original_trapping);
  _mm_setcsr(csr);
  Check(host_flushing && flushing_equal && host_trapping && trapping_equal,
        "production FTZ/DAZ/trap contexts keep original host arithmetic",
        fes.GetComm(), failures);
#endif
  Check(host_rounding && rounding_equal && ProbeEnvironmentSupported(),
        "production non-nearest environment declines and caller context is restored",
        fes.GetComm(), failures);
  if (mfem::Mpi::Root())
  {
    std::cout << "[admissibility-cuda-production-proof] native_calls=" << native_calls
              << " state_host_readback=0 original_summary_bitwise=required"
              << " default_enabled=0 sticky_exception_equivalence=unsupported\n";
  }
}
#endif
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  MFEM_VERIFY(backend == "cpu" || backend == "cuda", "Expected cpu or cuda.");
  mfem::Device device(backend.c_str());
  int failures = 0;
#if defined(IGRLAB_ADMISSIBILITY_CUDA_PROBE)
  ScopedProductionCudaFlag original_host_path("0");
#endif
  CheckSpaces(failures);
  CheckSampleContract(failures);
  CheckInvalidStates(failures);
  CheckDeviceAuthority(failures);
  CheckSignedHostGather(failures);
  CheckCommunicators(failures);
#if defined(IGRLAB_ADMISSIBILITY_CUDA_PROBE)
  CheckCudaArithmeticPrototype(failures);
  CheckProductionCudaScan(failures);
#endif
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (mfem::Mpi::Root())
  {
    std::cout << "Euler admissibility cached/original parity: backend=" << backend
              << " ranks=" << mfem::Mpi::WorldSize()
              << " failures=" << failures << '\n';
  }
  return failures == 0 ? 0 : 1;
}
