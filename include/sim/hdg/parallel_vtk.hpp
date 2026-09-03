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

#include <memory>
#include <optional>
#include <string>
#include <utility>
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
  /**
   * Trusted committed-boundary seed for appending an existing native series.
   * The cycle is the last durable output cycle; committed_time is the
   * checkpoint time and may be later than that cycle's publication time.
   */
  struct RestartSeed
  {
    int last_cycle = -1;
    double committed_time = 0.0;
  };
  std::optional<RestartSeed> restart;

  HDGParaViewOptions() = default;

  HDGParaViewOptions(
      std::string prefix,
      std::string collection,
      const mfem::VTKFormat format = mfem::VTKFormat::BINARY,
      const int detail = 1,
      const bool high_order = false,
      std::optional<RestartSeed> restart_seed = std::nullopt)
      : prefix_path(std::move(prefix)),
        collection_name(std::move(collection)),
        data_format(format),
        levels_of_detail(detail),
        high_order_output(high_order),
        restart(std::move(restart_seed))
  {
  }
};

/**
 * Externally owned mesh, physical interior fields, and output configuration.
 *
 * All field spaces must be parallel spaces attached to exactly @ref mesh.
 * Their data are used directly by MFEM's rank-local writer; ownership remains
 * with the caller for the whole call.
 */
struct HDGParaViewRequest
{
  mfem::ParMesh& mesh;
  std::vector<HDGInteriorFieldView> interior_fields;
  HDGParaViewOptions options;
};

/**
 * Caller-owned MFEM ParaView publication series for physical HDG interior
 * fields.  The series owns only its mfem::ParaViewDataCollection writer state;
 * the supplied mesh and fields remain externally owned and must outlive it.
 *
 * Keeping this object alive across Write() calls preserves native MFEM PVD
 * entries for each published cycle.  An explicit restart seed validates the
 * existing series and enables MFEM's native append mode; fresh-series behavior
 * remains the default.
 */
class HDGParaViewSeries
{
public:
  explicit HDGParaViewSeries(const HDGParaViewRequest& request);
  ~HDGParaViewSeries() = default;

  HDGParaViewSeries(const HDGParaViewSeries&) = delete;
  HDGParaViewSeries& operator=(const HDGParaViewSeries&) = delete;
  HDGParaViewSeries(HDGParaViewSeries&&) noexcept = default;
  HDGParaViewSeries& operator=(HDGParaViewSeries&&) noexcept = default;

  /**
   * Collectively append and validate one native MFEM PVD/PVTU/VTU
   * publication.  Every rank must call Write with the same metadata.  The
   * method returns only after every local VTU and the rank-zero PVTU/PVD are
   * present and structurally complete.  @ref cycle and @ref time are
   * serialized by MFEM; @ref step remains caller provenance only.
   */
  void Write(const HDGParaViewMetadata& metadata);

private:
  std::unique_ptr<mfem::ParaViewDataCollection> collection_;
  MPI_Comm communicator_ = MPI_COMM_NULL;
  int rank_ = -1;
  int ranks_ = 0;
  std::string collection_path_;
  std::string collection_name_;
  int last_cycle_ = -1;
  double committed_time_floor_ = 0.0;
  bool has_committed_time_floor_ = false;
};

} // namespace dg_sim::hdg
