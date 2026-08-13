#pragma once

/**
 * @file sim/wiring/euler_boundary_plan.hpp
 * @brief Semantic Euler boundary plan, independent of a numerical backend.
 */

#include <functional>
#include <utility>
#include <vector>

#include "mfem.hpp"

namespace dg_sim
{

/// @brief Semantic boundary category understood by Euler case descriptions.
enum class EulerBoundaryKind
{
  Periodic,
  Inlet,
  Outlet,
  Slip,
  NonReflecting
};

/// @brief Flow model used for an inlet entry.
enum class EulerInletModel
{
  Supersonic,
  Subsonic
};

/// @brief Flow model used for an outlet entry.
enum class EulerOutletModel
{
  Supersonic,
  SubsonicPressure,
  FarField
};

/// @brief Deferred physics data supplied by a case, not owned solution state.
using EulerBoundaryStateCallback = std::function<mfem::Vector()>;
using EulerBoundaryPressureCallback = std::function<double()>;

/// @brief One semantic Euler boundary instruction with its resolved marker.
struct EulerBoundaryPlanEntry
{
  EulerBoundaryKind kind = EulerBoundaryKind::Periodic;
  mfem::Array<int> marker;
  EulerInletModel inlet_model = EulerInletModel::Supersonic;
  EulerOutletModel outlet_model = EulerOutletModel::Supersonic;
  EulerBoundaryStateCallback state;
  EulerBoundaryPressureCallback pressure;
};

/// @brief Ordered, backend-neutral boundary instructions for one Euler case.
///
/// The plan has no numerical operator or solution ownership.  Its marker is
/// already resolved from mesh attributes; callbacks defer case physics data
/// until a backend lowers the plan.
class EulerBoundaryPlan
{
public:
  void AddPeriodic(mfem::Array<int> marker)
  {
    entries_.push_back({EulerBoundaryKind::Periodic, std::move(marker)});
  }

  void AddSupersonicInlet(mfem::Array<int> marker,
                          EulerBoundaryStateCallback state)
  {
    AddInlet(std::move(marker), EulerInletModel::Supersonic, std::move(state));
  }

  void AddSubsonicInlet(mfem::Array<int> marker,
                        EulerBoundaryStateCallback state)
  {
    AddInlet(std::move(marker), EulerInletModel::Subsonic, std::move(state));
  }

  void AddSupersonicOutlet(mfem::Array<int> marker)
  {
    EulerBoundaryPlanEntry entry;
    entry.kind = EulerBoundaryKind::Outlet;
    entry.marker = std::move(marker);
    entry.outlet_model = EulerOutletModel::Supersonic;
    entries_.push_back(std::move(entry));
  }

  void AddSubsonicOutletPressure(mfem::Array<int> marker,
                                 EulerBoundaryPressureCallback pressure)
  {
    EulerBoundaryPlanEntry entry;
    entry.kind = EulerBoundaryKind::Outlet;
    entry.marker = std::move(marker);
    entry.outlet_model = EulerOutletModel::SubsonicPressure;
    entry.pressure = std::move(pressure);
    entries_.push_back(std::move(entry));
  }

  void AddFarField(mfem::Array<int> marker, EulerBoundaryStateCallback state)
  {
    EulerBoundaryPlanEntry entry;
    entry.kind = EulerBoundaryKind::Outlet;
    entry.marker = std::move(marker);
    entry.outlet_model = EulerOutletModel::FarField;
    entry.state = std::move(state);
    entries_.push_back(std::move(entry));
  }

  void AddSlipWall(mfem::Array<int> marker)
  {
    entries_.push_back({EulerBoundaryKind::Slip, std::move(marker)});
  }

  void AddNonReflectingOutlet(mfem::Array<int> marker,
                              EulerBoundaryStateCallback state)
  {
    EulerBoundaryPlanEntry entry;
    entry.kind = EulerBoundaryKind::NonReflecting;
    entry.marker = std::move(marker);
    entry.state = std::move(state);
    entries_.push_back(std::move(entry));
  }

  const std::vector<EulerBoundaryPlanEntry>& Entries() const { return entries_; }
  bool Empty() const { return entries_.empty(); }

private:
  void AddInlet(mfem::Array<int> marker,
                EulerInletModel model,
                EulerBoundaryStateCallback state)
  {
    EulerBoundaryPlanEntry entry;
    entry.kind = EulerBoundaryKind::Inlet;
    entry.marker = std::move(marker);
    entry.inlet_model = model;
    entry.state = std::move(state);
    entries_.push_back(std::move(entry));
  }

  std::vector<EulerBoundaryPlanEntry> entries_;
};

} // namespace dg_sim
