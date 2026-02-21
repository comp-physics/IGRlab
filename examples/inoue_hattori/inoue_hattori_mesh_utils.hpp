#pragma once

/**
 * @file examples/inoue_hattori/inoue_hattori_mesh_utils.hpp
 * @brief Shared periodic-y mesh + boundary marker helpers for Inoue-Hattori cases.
 */

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "sim/wiring/bc_wiring.hpp"

namespace dg_sim::inoue_hattori
{

struct PeriodicYMeshConfig
{
  int nx = 0;
  int ny = 0;
  double x_left = 0.0;
  double x_right = 0.0;
  double y_bottom = 0.0;
  double y_top = 0.0;
  const char* mesh_filename = "";
  const char* case_label = "";
};

inline std::string EnsurePeriodicYMeshFile(const char* case_source_file,
                                           const PeriodicYMeshConfig& cfg)
{
  namespace fs = std::filesystem;
  const fs::path case_dir = fs::path(case_source_file).parent_path();
  const fs::path mesh_path = case_dir / cfg.mesh_filename;
  if (!fs::exists(mesh_path))
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(cfg.nx,
                                                  cfg.ny,
                                                  mfem::Element::QUADRILATERAL,
                                                  /*gen_edges=*/true,
                                                  /*sx=*/cfg.x_right - cfg.x_left,
                                                  /*sy=*/cfg.y_top - cfg.y_bottom);

    for (int vi = 0; vi < mesh.GetNV(); ++vi)
    {
      double* v = mesh.GetVertex(vi);
      v[0] += cfg.x_left;
      v[1] += cfg.y_bottom;
    }

    mfem::Vector ty(2);
    ty = 0.0;
    ty(1) = cfg.y_top - cfg.y_bottom;
    std::vector<mfem::Vector> translations = {ty};

    mfem::Mesh periodic =
      mfem::Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(translations));

    std::ofstream out(mesh_path);
    MFEM_VERIFY(out.good(),
                "Failed to create periodic-y mesh at " << mesh_path);
    periodic.Print(out);
  }
  return mesh_path.string();
}

inline std::string EnsureNonPeriodicMeshFile(const char* case_source_file,
                                             const PeriodicYMeshConfig& cfg)
{
  namespace fs = std::filesystem;
  const fs::path case_dir = fs::path(case_source_file).parent_path();
  const fs::path mesh_path = case_dir / cfg.mesh_filename;
  if (!fs::exists(mesh_path))
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(cfg.nx,
                                                  cfg.ny,
                                                  mfem::Element::QUADRILATERAL,
                                                  /*gen_edges=*/true,
                                                  /*sx=*/cfg.x_right - cfg.x_left,
                                                  /*sy=*/cfg.y_top - cfg.y_bottom);

    for (int vi = 0; vi < mesh.GetNV(); ++vi)
    {
      double* v = mesh.GetVertex(vi);
      v[0] += cfg.x_left;
      v[1] += cfg.y_bottom;
    }

    std::ofstream out(mesh_path);
    MFEM_VERIFY(out.good(),
                "Failed to create non-periodic mesh at " << mesh_path);
    mesh.Print(out);
  }
  return mesh_path.string();
}

inline BoundaryMarkerMap BuildInletOutletMarkers(mfem::ParMesh& pmesh,
                                                 const PeriodicYMeshConfig& cfg)
{
  MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0,
              cfg.case_label << " mesh has no boundary attributes.");

  const int max_attr = pmesh.bdr_attributes.Max();
  std::vector<double> sum_x(max_attr + 1, 0.0);
  std::vector<double> count(max_attr + 1, 0.0);

  for (int be = 0; be < pmesh.GetNBE(); ++be)
  {
    const int attr = pmesh.GetBdrAttribute(be);
    mfem::ElementTransformation* Tr = pmesh.GetBdrElementTransformation(be);
    const mfem::Geometry::Type geom = pmesh.GetBdrElementBaseGeometry(be);
    const mfem::IntegrationPoint& ip = mfem::Geometries.GetCenter(geom);
    mfem::Vector x;
    Tr->Transform(ip, x);
    sum_x[attr] += x(0);
    count[attr] += 1.0;
  }

  MPI_Allreduce(MPI_IN_PLACE,
                sum_x.data(),
                static_cast<int>(sum_x.size()),
                mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM,
                pmesh.GetComm());
  MPI_Allreduce(MPI_IN_PLACE,
                count.data(),
                static_cast<int>(count.size()),
                mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM,
                pmesh.GetComm());

  mfem::Array<int> inlet(max_attr);
  mfem::Array<int> outlet(max_attr);
  inlet = 0;
  outlet = 0;

  const double tol = 1.0e-10 * (cfg.x_right - cfg.x_left);
  bool found_inlet = false;
  bool found_outlet = false;

  for (int attr = 1; attr <= max_attr; ++attr)
  {
    if (count[attr] <= 0.0)
    {
      continue;
    }

    const double x_avg = sum_x[attr] / count[attr];
    if (std::abs(x_avg - cfg.x_right) <= tol)
    {
      inlet[attr - 1] = 1;
      found_inlet = true;
    }
    else if (std::abs(x_avg - cfg.x_left) <= tol)
    {
      outlet[attr - 1] = 1;
      found_outlet = true;
    }
  }

  MFEM_VERIFY(found_inlet && found_outlet,
              "Failed to identify inlet/outlet boundary attributes for "
                << cfg.case_label << " periodic-y mesh.");

  BoundaryMarkerMap markers;
  markers.Add(BoundaryRole::Inlet, std::move(inlet));
  markers.Add(BoundaryRole::Outlet, std::move(outlet));
  return markers;
}

inline BoundaryMarkerMap
BuildInletOutletSlipMarkers(mfem::ParMesh& pmesh, const PeriodicYMeshConfig& cfg)
{
  MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0,
              cfg.case_label << " mesh has no boundary attributes.");

  const int max_attr = pmesh.bdr_attributes.Max();
  std::vector<double> sum_x(max_attr + 1, 0.0);
  std::vector<double> sum_y(max_attr + 1, 0.0);
  std::vector<double> count(max_attr + 1, 0.0);

  for (int be = 0; be < pmesh.GetNBE(); ++be)
  {
    const int attr = pmesh.GetBdrAttribute(be);
    mfem::ElementTransformation* Tr = pmesh.GetBdrElementTransformation(be);
    const mfem::Geometry::Type geom = pmesh.GetBdrElementBaseGeometry(be);
    const mfem::IntegrationPoint& ip = mfem::Geometries.GetCenter(geom);
    mfem::Vector x;
    Tr->Transform(ip, x);
    sum_x[attr] += x(0);
    sum_y[attr] += x(1);
    count[attr] += 1.0;
  }

  MPI_Allreduce(MPI_IN_PLACE,
                sum_x.data(),
                static_cast<int>(sum_x.size()),
                mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM,
                pmesh.GetComm());
  MPI_Allreduce(MPI_IN_PLACE,
                sum_y.data(),
                static_cast<int>(sum_y.size()),
                mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM,
                pmesh.GetComm());
  MPI_Allreduce(MPI_IN_PLACE,
                count.data(),
                static_cast<int>(count.size()),
                mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM,
                pmesh.GetComm());

  mfem::Array<int> inlet(max_attr);
  mfem::Array<int> outlet(max_attr);
  mfem::Array<int> slip(max_attr);
  inlet = 0;
  outlet = 0;
  slip = 0;

  const double xtol = 1.0e-10 * (cfg.x_right - cfg.x_left);
  const double ytol = 1.0e-10 * (cfg.y_top - cfg.y_bottom);
  bool found_inlet = false;
  bool found_outlet = false;
  bool found_bottom = false;
  bool found_top = false;

  for (int attr = 1; attr <= max_attr; ++attr)
  {
    if (count[attr] <= 0.0)
    {
      continue;
    }

    const double x_avg = sum_x[attr] / count[attr];
    const double y_avg = sum_y[attr] / count[attr];
    if (std::abs(x_avg - cfg.x_right) <= xtol)
    {
      inlet[attr - 1] = 1;
      found_inlet = true;
      continue;
    }
    if (std::abs(x_avg - cfg.x_left) <= xtol)
    {
      outlet[attr - 1] = 1;
      found_outlet = true;
      continue;
    }
    if (std::abs(y_avg - cfg.y_bottom) <= ytol)
    {
      slip[attr - 1] = 1;
      found_bottom = true;
      continue;
    }
    if (std::abs(y_avg - cfg.y_top) <= ytol)
    {
      slip[attr - 1] = 1;
      found_top = true;
      continue;
    }
  }

  MFEM_VERIFY(found_inlet && found_outlet && found_bottom && found_top,
              "Failed to identify inlet/outlet/slip attributes for "
                << cfg.case_label << " non-periodic mesh.");

  BoundaryMarkerMap markers;
  markers.Add(BoundaryRole::Inlet, std::move(inlet));
  markers.Add(BoundaryRole::Outlet, std::move(outlet));
  markers.Add(BoundaryRole::Slip, std::move(slip));
  return markers;
}

} // namespace dg_sim::inoue_hattori
