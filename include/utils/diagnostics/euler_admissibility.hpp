#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "mfem.hpp"
#include "physics/euler/eos.hpp"

/**
 * @file utils/diagnostics/euler_admissibility.hpp
 * @brief Helpers to scan Euler states for density/pressure admissibility.
 * @ingroup utils_config
 */

namespace mfem
{
namespace dg_igr
{

enum class EulerAdmissibilityPointSet
{
  Dof,
  Limiter,
  Output
};

struct EulerAdmissibilityIssue
{
  bool found = false;
  bool pressure_bad = false;
  bool face_point = false;
  int element = -1;
  int face = -1;
  int point_index = -1;
  int scalar_dof = -1;
  double rho = std::numeric_limits<double>::infinity();
  double pressure = std::numeric_limits<double>::infinity();
};

struct EulerAdmissibilitySummary
{
  double global_min_rho = std::numeric_limits<double>::infinity();
  double global_min_pressure = std::numeric_limits<double>::infinity();
  int global_bad_rho = 0;
  int global_bad_pressure = 0;
  int first_bad_rank = -1;
  EulerAdmissibilityIssue first_local_issue;
};

namespace detail
{

struct ElementFaceCacheEntry
{
  long mesh_sequence = -1;
  int ne = -1;
  std::vector<std::vector<int>> elem_faces;
};

inline const std::vector<std::vector<int>>& GetElementFacesForAdmissibility(
  mfem::Mesh& mesh)
{
  static std::unordered_map<const mfem::Mesh*, ElementFaceCacheEntry> cache;
  auto& entry = cache[&mesh];
  if (entry.mesh_sequence == mesh.GetSequence() && entry.ne == mesh.GetNE())
  {
    return entry.elem_faces;
  }

  entry.mesh_sequence = mesh.GetSequence();
  entry.ne = mesh.GetNE();
  entry.elem_faces.clear();
  entry.elem_faces.resize(entry.ne);

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1 = -1, e2 = -1;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e1 >= 0 && e1 < entry.ne)
    {
      entry.elem_faces[e1].push_back(f);
    }
    if (e2 >= 0 && e2 < entry.ne)
    {
      entry.elem_faces[e2].push_back(f);
    }
  }

  return entry.elem_faces;
}

inline std::vector<int>
BuildFaceQuadratureOrdersForAdmissibility(const mfem::ParFiniteElementSpace& vfes,
                                          int volume_qord,
                                          const std::vector<int>& configured_face_qords)
{
  std::vector<int> qords;
  qords.reserve(configured_face_qords.size() + 2);

  auto push_unique_positive = [&](int qord)
  {
    if (qord <= 0)
    {
      return;
    }
    if (std::find(qords.begin(), qords.end(), qord) == qords.end())
    {
      qords.push_back(qord);
    }
  };

  push_unique_positive(volume_qord);
  for (const int qord : configured_face_qords)
  {
    push_unique_positive(qord);
  }

  if (qords.size() == 1)
  {
    push_unique_positive(2 * vfes.GetFE(0)->GetOrder() + 1);
  }

  return qords;
}

inline int FindFirstBadRank(bool found, MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int first_bad_rank = found ? rank : std::numeric_limits<int>::max();
  MPI_Allreduce(MPI_IN_PLACE,
                &first_bad_rank,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_MIN,
                comm);
  if (first_bad_rank == std::numeric_limits<int>::max())
  {
    return -1;
  }
  return first_bad_rank;
}

} // namespace detail

inline EulerAdmissibilitySummary
ScanEulerAdmissibility(const mfem::ParFiniteElementSpace& vfes,
                       const mfem::Vector& v,
                       int dim,
                       double gamma,
                       double rho_min,
                       double p_min,
                       int quad_order,
                       const std::vector<int>& face_quad_orders,
                       EulerAdmissibilityPointSet point_set)
{
  EulerAdmissibilitySummary out;
  MFEM_VERIFY(vfes.GetOrdering() == mfem::Ordering::byNODES,
              "Euler admissibility scan assumes byNODES ordering.");

  const int num_eq = dim + 2;
  MFEM_VERIFY(vfes.GetVDim() == num_eq,
              "Euler admissibility scan expects Euler state layout.");

  mfem::Mesh& mesh = *vfes.GetMesh();
  const auto& elem_faces = detail::GetElementFacesForAdmissibility(mesh);
  const int NE = vfes.GetNE();

  const int base_qord =
    (quad_order > 0) ? quad_order : (2 * vfes.GetFE(0)->GetOrder() + 2);
  const int output_qord = std::max(base_qord + 2, 2 * vfes.GetFE(0)->GetOrder() + 4);
  const int scan_qord =
    (point_set == EulerAdmissibilityPointSet::Output) ? output_qord : base_qord;

  std::vector<int> scan_face_qords =
    detail::BuildFaceQuadratureOrdersForAdmissibility(vfes,
                                                      scan_qord,
                                                      face_quad_orders);
  if (point_set == EulerAdmissibilityPointSet::Output)
  {
    for (int& qord : scan_face_qords)
    {
      qord = std::max(qord + 2, 2 * vfes.GetFE(0)->GetOrder() + 3);
    }
    std::sort(scan_face_qords.begin(), scan_face_qords.end());
    scan_face_qords.erase(
      std::unique(scan_face_qords.begin(), scan_face_qords.end()),
      scan_face_qords.end());
  }

  double local_min_rho = std::numeric_limits<double>::infinity();
  double local_min_pressure = std::numeric_limits<double>::infinity();
  int local_bad_rho = 0;
  int local_bad_pressure = 0;

  mfem::Array<int> vdofs;
  mfem::Vector elvec;
  mfem::DenseMatrix elmat;
  mfem::Vector shape, state(num_eq);

  auto inspect_state = [&](const mfem::Vector& U,
                           int elem,
                           int face,
                           int point_index,
                           bool face_point,
                           int scalar_dof)
  {
    const double rho = U(0);
    bool conservative_finite = true;
    for (int eq = 0; eq < num_eq; ++eq)
    {
      conservative_finite = conservative_finite && std::isfinite(U(eq));
    }

    double momentum_squared = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      momentum_squared += U(1 + d) * U(1 + d);
    }
    const double kinetic =
      0.5 * momentum_squared / std::max(rho, 1.0e-16);
    const double pressure =
      mfem::EulerPressureFromConservative(U, dim, gamma, /*rho_floor=*/1e-16);

    const bool rho_finite = std::isfinite(rho);
    const bool kinetic_finite = std::isfinite(kinetic);
    const bool pressure_finite = std::isfinite(pressure);
    if (rho_finite)
    {
      local_min_rho = std::min(local_min_rho, rho);
    }
    if (pressure_finite)
    {
      local_min_pressure = std::min(local_min_pressure, pressure);
    }

    const bool bad_rho = !rho_finite || rho < rho_min;
    // The existing pressure counter is also the thermodynamic/state-invalid
    // counter.  In particular, do not let NaN comparisons make a nonfinite
    // conservative component or derived kinetic/pressure value look valid.
    const bool bad_pressure = !conservative_finite || !kinetic_finite ||
                              !pressure_finite || pressure < p_min;
    local_bad_rho += bad_rho ? 1 : 0;
    local_bad_pressure += bad_pressure ? 1 : 0;

    if (!out.first_local_issue.found && (bad_rho || bad_pressure))
    {
      out.first_local_issue.found = true;
      out.first_local_issue.pressure_bad = !bad_rho && bad_pressure;
      out.first_local_issue.face_point = face_point;
      out.first_local_issue.element = elem;
      out.first_local_issue.face = face;
      out.first_local_issue.point_index = point_index;
      out.first_local_issue.scalar_dof = scalar_dof;
      out.first_local_issue.rho = rho;
      out.first_local_issue.pressure = pressure;
    }
  };

  for (int i = 0; i < NE; ++i)
  {
    const mfem::FiniteElement* fe = vfes.GetFE(i);
    mfem::ElementTransformation* Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    const auto* fec_l2 = dynamic_cast<const mfem::L2_FECollection*>(vfes.FEColl());
    const bool enforce_nodes =
      (fec_l2 != nullptr && fec_l2->GetBasisType() == mfem::BasisType::GaussLobatto);

    vfes.GetElementVDofs(i, vdofs);
    v.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);
    shape.SetSize(dof);

    const bool do_dof_scan = (point_set == EulerAdmissibilityPointSet::Dof);
    if (do_dof_scan || enforce_nodes)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, state);
        inspect_state(state, i, -1, j, false, j);
      }
      if (do_dof_scan)
      {
        continue;
      }
    }

    const mfem::IntegrationRule& ir =
      mfem::IntRules.Get(Tr->GetGeometryType(), scan_qord);
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, state);
      inspect_state(state, i, -1, q, false, -1);
    }

    for (const int face : elem_faces[i])
    {
      mfem::FaceElementTransformations* FTr =
        mesh.GetFaceElementTransformations(face);
      if (FTr == nullptr)
      {
        continue;
      }

      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(face, &e1, &e2);
      const bool elem_is_first = (e1 == i);
      const bool elem_is_second = (e2 == i);
      if (!elem_is_first && !elem_is_second)
      {
        continue;
      }

      for (const int fqord : scan_face_qords)
      {
        const mfem::IntegrationRule& fir =
          mfem::IntRules.Get(FTr->GetGeometryType(), fqord);
        for (int q = 0; q < fir.GetNPoints(); ++q)
        {
          const mfem::IntegrationPoint& fip = fir.IntPoint(q);
          FTr->SetAllIntPoints(&fip);
          const mfem::IntegrationPoint& eip = elem_is_first
                                                ? FTr->GetElement1IntPoint()
                                                : FTr->GetElement2IntPoint();
          fe->CalcShape(eip, shape);
          elmat.MultTranspose(shape, state);
          inspect_state(state, i, face, q, true, -1);
        }
      }
    }
  }

  MPI_Allreduce(&local_min_rho,
                &out.global_min_rho,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN,
                vfes.GetComm());
  MPI_Allreduce(&local_min_pressure,
                &out.global_min_pressure,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN,
                vfes.GetComm());
  MPI_Allreduce(&local_bad_rho,
                &out.global_bad_rho,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM,
                vfes.GetComm());
  MPI_Allreduce(&local_bad_pressure,
                &out.global_bad_pressure,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM,
                vfes.GetComm());
  out.first_bad_rank =
    detail::FindFirstBadRank(out.first_local_issue.found, vfes.GetComm());

  return out;
}

} // namespace dg_igr
} // namespace mfem
