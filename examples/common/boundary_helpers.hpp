#pragma once

#include <initializer_list>
#include <map>

#include "mfem.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "utils/mesh/boundary_markers.hpp"

namespace dg_igr_examples
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
inline BoundaryMarkerMap BuildBoundaryMarkers(
    mfem::ParMesh& pmesh,
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
      new mfem::BdrHyperbolicFluxIntegrator(
          numerical_flux,
          std::make_unique<mfem::SupersonicInletState>(inlet_state),
          int_order_offset),
      marker);
}

/// @brief Add a supersonic inlet BC using a builder that returns the inlet state.
template <typename StateBuilder>
inline void AddSupersonicInletFromBuilder(mfem::DGHyperbolicConservationLaws& op,
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
  op.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
          numerical_flux,
          std::make_unique<mfem::SupersonicOutletState>(),
          int_order_offset),
      marker);
}

inline void AddSlipWall(mfem::DGHyperbolicConservationLaws& op,
                        const mfem::NumericalFlux& numerical_flux,
                        int int_order_offset,
                        int dim,
                        const mfem::Array<int>& marker)
{
  op.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(
          numerical_flux,
          std::make_unique<mfem::SlipWallState>(dim),
          int_order_offset),
      marker);
}

} // namespace dg_igr_examples
