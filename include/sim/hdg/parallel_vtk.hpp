#pragma once

/**
 * @file sim/hdg/parallel_vtk.hpp
 * @brief Host-side ParaView publication seam for HDG interior fields.
 *
 * This adapter deliberately publishes only physical fields represented on the
 * HDG interior volume space.  In V1, trace/skeleton fields are not emitted.
 * Each field is an externally owned, host-readable mfem::ParGridFunction on
 * the supplied mfem::ParMesh; scalar fields stay scalar and vector fields keep
 * their MFEM component count and ordering.  The adapter neither gathers field
 * values to rank zero nor creates or repartitions a mesh.  Device callers must
 * stage their state into these host views before calling this API.
 */

#include <string>
#include <vector>

#include <mfem.hpp>

namespace dg_sim::hdg
{

/** A physical HDG interior field, owned by the caller. */
struct HDGInteriorFieldView
{
  /** Stable ParaView field name, for example "density" or "momentum". */
  std::string name;
  /** Host-readable field on the request mesh's interior volume space. */
  mfem::ParGridFunction& values;
};

/**
 * Output metadata.  @ref cycle is the MFEM publication identity and is
 * written through mfem::DataCollection::SetCycle.  @ref step is independent
 * solver-step provenance and may differ from cycle; V1 does not add a custom
 * sidecar or encode it into MFEM's native PVD/PVTU/VTU naming.  @ref time is
 * written through SetTime.
 */
struct HDGParaViewMetadata
{
  int cycle = 0; ///< Non-negative MFEM publication-cycle identity.
  int step = 0;  ///< Non-negative caller solver-step provenance.
  double time = 0.0;
};

/** Format and naming choices for one rank-local MFEM data collection. */
struct HDGParaViewOptions
{
  /** Parent directory; MFEM creates <prefix_path>/<collection_name>. */
  std::string prefix_path;
  /** MFEM DataCollection name and PVD/PVTU stem. */
  std::string collection_name;
  mfem::VTKFormat data_format = mfem::VTKFormat::BINARY;
  int levels_of_detail = 1;
  bool high_order_output = false;
};

/**
 * Externally owned mesh, physical interior fields, and output metadata.
 *
 * All field spaces must be parallel spaces attached to exactly @ref mesh.
 * Their data are used directly by MFEM's rank-local writer; ownership remains
 * with the caller for the whole call.
 */
struct HDGParaViewRequest
{
  mfem::ParMesh& mesh;
  std::vector<HDGInteriorFieldView> interior_fields;
  HDGParaViewMetadata metadata;
  HDGParaViewOptions options;
};

/**
 * Write MFEM ParaViewDataCollection PVD/PVTU/VTU output for physical HDG
 * interior fields.  Throws std::invalid_argument for an incompatible host view
 * or metadata, and std::runtime_error if MFEM reports a write failure.
 */
void WriteHDGParaViewOutput(const HDGParaViewRequest& request);

} // namespace dg_sim::hdg
