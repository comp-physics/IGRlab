#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "utils/diagnostics/euler_admissibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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
  auto mesh = MakeMesh(2, mfem::Mpi::WorldSize());
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(3, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector state(fes.GetVSize());
  state.UseDevice(true);
  SetConstant(fes, state);
  Cache cache;
  CompareAll(cache, fes, state, failures);
  const int ndofs = fes.GetNDofs();
  double* data = state.ReadWrite();
  mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int d) { data[d] = 0.125; });
  const auto scan = Compare(cache, fes, state, PointSet::Dof, failures,
                            0, {}, 1.4, 0.25);
  int global_dofs = ndofs;
  MPI_Allreduce(MPI_IN_PLACE, &global_dofs, 1, MPI_INT, MPI_SUM, fes.GetComm());
  Check(scan.global_bad_rho == global_dofs && scan.global_min_rho == 0.125,
        "warm cache reads the current device-authoritative state",
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
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  MFEM_VERIFY(backend == "cpu" || backend == "cuda", "Expected cpu or cuda.");
  mfem::Device device(backend.c_str());
  int failures = 0;
  CheckSpaces(failures);
  CheckSampleContract(failures);
  CheckInvalidStates(failures);
  CheckDeviceAuthority(failures);
  CheckCommunicators(failures);
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (mfem::Mpi::Root())
  {
    std::cout << "Euler admissibility cached/original parity: backend=" << backend
              << " ranks=" << mfem::Mpi::WorldSize()
              << " failures=" << failures << '\n';
  }
  return failures == 0 ? 0 : 1;
}
