#pragma once

/**
 * @file sim/wiring/bc_wiring.hpp
 * @brief Boundary marker helpers and BC wiring utilities for Euler DG.
 */

#include <initializer_list>
#include <map>
#include <memory>
#include <algorithm>
#include <vector>

#include "mfem.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "sim/wiring/euler_boundary_plan.hpp"
#include "utils/mesh/boundary_markers.hpp"

namespace dg_sim
{

/// @brief Semantic roles for boundary marker maps in examples.
enum class BoundaryRole
{
  Inlet,
  Outlet,
  Slip,
  Wall,
  Top,
  Bottom,
  BottomInlet,
  BottomWall,
  Cylinder
};

/// @brief Boundary role + attribute list pair.
struct BoundaryMarkerSpec
{
  BoundaryRole role;
  std::initializer_list<int> attrs;
};

/// @brief Container for boundary markers keyed by role.
class BoundaryMarkerMap
{
public:
  void Add(BoundaryRole role, mfem::Array<int> marker)
  {
    markers_[role] = std::move(marker);
  }

  bool Has(BoundaryRole role) const
  {
    return markers_.find(role) != markers_.end();
  }

  const mfem::Array<int>& At(BoundaryRole role) const
  {
    const auto it = markers_.find(role);
    MFEM_VERIFY(it != markers_.end(), "Missing boundary marker for role.");
    return it->second;
  }

private:
  std::map<BoundaryRole, mfem::Array<int>> markers_;
};

/// @brief Build a boundary marker map from role/attribute specifications.
inline BoundaryMarkerMap
BuildBoundaryMarkers(const mfem::ParMesh& pmesh,
                     std::initializer_list<BoundaryMarkerSpec> specs)
{
  BoundaryMarkerMap map;
  for (const auto& spec : specs)
  {
    mfem::Array<int> marker;
    mfem::MakeMarker(marker, pmesh, spec.attrs);
    map.Add(spec.role, std::move(marker));
  }
  return map;
}

inline void AddSupersonicInlet(mfem::DGHyperbolicConservationLaws& op,
                               const mfem::NumericalFlux& numerical_flux,
                               const mfem::Vector& inlet_state,
                               int int_order_offset,
                               const mfem::Array<int>& marker)
{
  op.AddBdrFaceIntegrator(
    std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
      numerical_flux,
      std::make_unique<mfem::SupersonicInletState>(inlet_state),
      int_order_offset),
    marker);
}

/// @brief Add a supersonic inlet BC using a builder that returns the inlet state.
template <typename StateBuilder>
inline void
AddSupersonicInletFromBuilder(mfem::DGHyperbolicConservationLaws& op,
                              const mfem::NumericalFlux& numerical_flux,
                              StateBuilder&& build_inlet_state,
                              int int_order_offset,
                              const mfem::Array<int>& marker)
{
  mfem::Vector inlet_state = build_inlet_state();
  AddSupersonicInlet(op, numerical_flux, inlet_state, int_order_offset, marker);
}

inline void AddSupersonicOutlet(mfem::DGHyperbolicConservationLaws& op,
                                const mfem::NumericalFlux& numerical_flux,
                                int int_order_offset,
                                const mfem::Array<int>& marker)
{
  op.AddBdrFaceIntegrator(std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
                            numerical_flux,
                            std::make_unique<mfem::SupersonicOutletState>(),
                            int_order_offset),
                          marker);
}

inline void AddSubsonicInlet(mfem::DGHyperbolicConservationLaws& op,
                             const mfem::NumericalFlux& numerical_flux,
                             const mfem::Vector& inlet_state,
                             int int_order_offset,
                             int dim,
                             double gamma,
                             const mfem::Array<int>& marker,
                             double density_floor =
                               mfem::kEulerBoundaryDensityFloor,
                             double pressure_floor =
                               mfem::kEulerBoundaryPressureFloor)
{
  op.AddBdrFaceIntegrator(std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
                            numerical_flux,
                            std::make_unique<mfem::SubsonicInletState>(inlet_state,
                                                                        dim,
                                                                        gamma,
                                                                        density_floor,
                                                                        pressure_floor),
                            int_order_offset),
                          marker);
}

inline void
AddSubsonicOutletPressure(mfem::DGHyperbolicConservationLaws& op,
                          const mfem::NumericalFlux& numerical_flux,
                          double pressure_out,
                          int int_order_offset,
                          int dim,
                          double gamma,
                          const mfem::Array<int>& marker,
                          double density_floor =
                            mfem::kEulerBoundaryDensityFloor,
                          double pressure_floor =
                            mfem::kEulerBoundaryPressureFloor)
{
  op.AddBdrFaceIntegrator(
    std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
      numerical_flux,
      std::make_unique<mfem::SubsonicOutletPressureState>(dim,
                                                          gamma,
                                                          pressure_out,
                                                          density_floor,
                                                          pressure_floor),
      int_order_offset),
    marker);
}

inline void AddNonReflectingOutlet(mfem::DGHyperbolicConservationLaws& op,
                                   const mfem::NumericalFlux& numerical_flux,
                                   const mfem::Vector& reference_state,
                                   int int_order_offset,
                                   int dim,
                                   double gamma,
                                   const mfem::Array<int>& marker,
                                   double density_floor =
                                     mfem::kEulerBoundaryDensityFloor,
                                   double pressure_floor =
                                     mfem::kEulerBoundaryPressureFloor)
{
  op.AddBdrFaceIntegrator(
    std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
      numerical_flux,
      std::make_unique<mfem::NonReflectingOutletState>(reference_state,
                                                       dim,
                                                       gamma,
                                                       density_floor,
                                                       pressure_floor),
      int_order_offset),
    marker);
}

inline void AddFarField(mfem::DGHyperbolicConservationLaws& op,
                        const mfem::NumericalFlux& numerical_flux,
                        const mfem::Vector& farfield_state,
                        int int_order_offset,
                        int dim,
                        double gamma,
                        const mfem::Array<int>& marker,
                        double density_floor =
                          mfem::kEulerBoundaryDensityFloor,
                        double pressure_floor =
                          mfem::kEulerBoundaryPressureFloor)
{
  op.AddBdrFaceIntegrator(
    std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
      numerical_flux,
      std::make_unique<mfem::FarFieldState>(farfield_state,
                                            dim,
                                            gamma,
                                            density_floor,
                                            pressure_floor),
      int_order_offset),
    marker);
}

inline void AddSlipWall(mfem::DGHyperbolicConservationLaws& op,
                        const mfem::NumericalFlux& numerical_flux,
                        int int_order_offset,
                        int dim,
                        const mfem::Array<int>& marker,
                        double density_floor =
                          mfem::kEulerBoundaryDensityFloor,
                        double pressure_floor =
                          mfem::kEulerBoundaryPressureFloor)
{
  op.AddBdrFaceIntegrator(std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
                            numerical_flux,
                            std::make_unique<mfem::SlipWallState>(dim,
                                                                  density_floor,
                                                                  pressure_floor),
                            int_order_offset),
                          marker);
}

/// @brief Lower a semantic plan to the existing Euler DG boundary integrators.
///
/// HDG lowering is intentionally not provided by this foundation.
inline void LowerDGEulerBoundaryPlan(
  mfem::DGHyperbolicConservationLaws& op,
  const mfem::NumericalFlux& numerical_flux,
  int int_order_offset,
  int dim,
  double gamma,
  const EulerBoundaryPlan& plan,
  double density_floor = mfem::kEulerBoundaryDensityFloor,
  double pressure_floor = mfem::kEulerBoundaryPressureFloor)
{
  for (const EulerBoundaryPlanEntry& entry : plan.Entries())
  {
    switch (entry.kind)
    {
    case EulerBoundaryKind::Periodic:
      // A periodic mesh has no physical boundary integrator to attach.
      break;
    case EulerBoundaryKind::Inlet:
      MFEM_VERIFY(entry.state, "Euler inlet plan entry requires a state callback.");
      if (entry.inlet_model == EulerInletModel::Supersonic)
      {
        AddSupersonicInlet(op,
                           numerical_flux,
                           entry.state(),
                           int_order_offset,
                           entry.marker);
      }
      else
      {
        AddSubsonicInlet(op,
                         numerical_flux,
                         entry.state(),
                         int_order_offset,
                         dim,
                         gamma,
                         entry.marker,
                         density_floor,
                         pressure_floor);
      }
      break;
    case EulerBoundaryKind::Outlet:
      switch (entry.outlet_model)
      {
      case EulerOutletModel::Supersonic:
        AddSupersonicOutlet(op, numerical_flux, int_order_offset, entry.marker);
        break;
      case EulerOutletModel::SubsonicPressure:
        MFEM_VERIFY(entry.pressure,
                    "Subsonic outlet plan entry requires a pressure callback.");
        AddSubsonicOutletPressure(op,
                                  numerical_flux,
                                  entry.pressure(),
                                  int_order_offset,
                                  dim,
                                  gamma,
                                  entry.marker,
                                  density_floor,
                                  pressure_floor);
        break;
      case EulerOutletModel::FarField:
        MFEM_VERIFY(entry.state,
                    "Far-field outlet plan entry requires a state callback.");
        AddFarField(op,
                    numerical_flux,
                    entry.state(),
                    int_order_offset,
                    dim,
                    gamma,
                    entry.marker,
                    density_floor,
                    pressure_floor);
        break;
      }
      break;
    case EulerBoundaryKind::Slip:
      AddSlipWall(op,
                  numerical_flux,
                  int_order_offset,
                  dim,
                  entry.marker,
                  density_floor,
                  pressure_floor);
      break;
    case EulerBoundaryKind::NonReflecting:
      MFEM_VERIFY(entry.state,
                  "Non-reflecting plan entry requires a state callback.");
      AddNonReflectingOutlet(op,
                             numerical_flux,
                             entry.state(),
                             int_order_offset,
                             dim,
                             gamma,
                             entry.marker,
                             density_floor,
                             pressure_floor);
      break;
    }
  }
}

/// @brief Convenience wrapper for adding Euler boundary integrators.
template <typename NumericalFlux> class EulerBoundaryWiring
{
public:
  EulerBoundaryWiring(mfem::DGHyperbolicConservationLaws& op,
                      const NumericalFlux& numerical_flux,
                      int int_order_offset,
                      int dim,
                      double gamma,
                      double density_floor = mfem::kEulerBoundaryDensityFloor,
                      double pressure_floor = mfem::kEulerBoundaryPressureFloor)
    : op_(op),
      numerical_flux_(numerical_flux),
      int_order_offset_(int_order_offset),
      dim_(dim),
      gamma_(gamma),
      density_floor_(density_floor),
      pressure_floor_(pressure_floor)
  {
  }

  void AddSlipWall(const BoundaryMarkerMap& markers,
                   BoundaryRole role = BoundaryRole::Slip) const
  {
    RecordBoundaryKind_(markers.At(role), true);
    dg_sim::AddSlipWall(op_,
                        numerical_flux_,
                        int_order_offset_,
                        dim_,
                        markers.At(role),
                        density_floor_,
                        pressure_floor_);
  }

  void AddSupersonicOutlet(const BoundaryMarkerMap& markers,
                           BoundaryRole role = BoundaryRole::Outlet) const
  {
    RecordBoundaryKind_(markers.At(role), false);
    dg_sim::AddSupersonicOutlet(op_,
                                numerical_flux_,
                                int_order_offset_,
                                markers.At(role));
  }

  void AddSupersonicInlet(const BoundaryMarkerMap& markers,
                          const mfem::Vector& inlet_state,
                          BoundaryRole role = BoundaryRole::Inlet) const
  {
    RecordBoundaryKind_(markers.At(role), false);
    dg_sim::AddSupersonicInlet(op_,
                               numerical_flux_,
                               inlet_state,
                               int_order_offset_,
                               markers.At(role));
  }

  void AddSubsonicInlet(const BoundaryMarkerMap& markers,
                        const mfem::Vector& inlet_state,
                        BoundaryRole role = BoundaryRole::Inlet) const
  {
    RecordBoundaryKind_(markers.At(role), false);
    dg_sim::AddSubsonicInlet(op_,
                             numerical_flux_,
                             inlet_state,
                             int_order_offset_,
                             dim_,
                             gamma_,
                             markers.At(role),
                             density_floor_,
                             pressure_floor_);
  }

  void AddSubsonicOutletPressure(const BoundaryMarkerMap& markers,
                                 double pressure_out,
                                 BoundaryRole role = BoundaryRole::Outlet) const
  {
    RecordBoundaryKind_(markers.At(role), false);
    dg_sim::AddSubsonicOutletPressure(op_,
                                      numerical_flux_,
                                      pressure_out,
                                      int_order_offset_,
                                      dim_,
                                      gamma_,
                                      markers.At(role),
                                      density_floor_,
                                      pressure_floor_);
  }

  void AddNonReflectingOutlet(const BoundaryMarkerMap& markers,
                              const mfem::Vector& reference_state,
                              BoundaryRole role = BoundaryRole::Outlet) const
  {
    RecordBoundaryKind_(markers.At(role), false);
    dg_sim::AddNonReflectingOutlet(op_,
                                   numerical_flux_,
                                   reference_state,
                                   int_order_offset_,
                                   dim_,
                                   gamma_,
                                   markers.At(role),
                                   density_floor_,
                                   pressure_floor_);
  }

  void AddFarField(const BoundaryMarkerMap& markers,
                   const mfem::Vector& farfield_state,
                   BoundaryRole role = BoundaryRole::Outlet) const
  {
    RecordBoundaryKind_(markers.At(role), false);
    dg_sim::AddFarField(op_,
                        numerical_flux_,
                        farfield_state,
                        int_order_offset_,
                        dim_,
                        gamma_,
                        markers.At(role),
                        density_floor_,
                        pressure_floor_);
  }

  template <typename StateBuilder>
  void
  AddSupersonicInletFromBuilder(const BoundaryMarkerMap& markers,
                                StateBuilder&& build_inlet_state,
                                BoundaryRole role = BoundaryRole::Inlet) const
  {
    RecordBoundaryKind_(markers.At(role), false);
    dg_sim::AddSupersonicInletFromBuilder(
      op_,
      numerical_flux_,
      std::forward<StateBuilder>(build_inlet_state),
      int_order_offset_,
      markers.At(role));
  }

  /// Same first-registered-attribute precedence as the Euler packed boundary
  /// selector. Other physical categories retain the source's own trace.
  mfem::Array<int> SlipWallMarker() const
  {
    mfem::Array<int> marker(static_cast<int>(boundary_kind_.size()));
    for (int i = 0; i < marker.Size(); ++i)
      marker[i] = boundary_kind_[i] == 2 ? 1 : 0;
    return marker;
  }

private:
  void RecordBoundaryKind_(const mfem::Array<int>& marker, bool slip) const
  {
    boundary_kind_.resize(std::max(boundary_kind_.size(),
                                  static_cast<std::size_t>(marker.Size())), 0);
    for (int i = 0; i < marker.Size(); ++i)
      if (marker[i] && !boundary_kind_[i]) boundary_kind_[i] = slip ? 2 : 1;
  }
  mutable std::vector<int> boundary_kind_;
  mfem::DGHyperbolicConservationLaws& op_;
  const NumericalFlux& numerical_flux_;
  int int_order_offset_ = 0;
  int dim_ = 0;
  double gamma_ = 1.4;
  double density_floor_ = mfem::kEulerBoundaryDensityFloor;
  double pressure_floor_ = mfem::kEulerBoundaryPressureFloor;
};

} // namespace dg_sim
