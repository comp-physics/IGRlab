#pragma once

/**
 * @file sim/hdg/euler_output_fields.hpp
 * @brief Standard Euler volume fields for transactional HDG ParaView output.
 */

#include "numerics/hdg/generic/session_contract.hpp"
#include "sim/hdg/parallel_vtk.hpp"
#include "sim/state/state_views.hpp"

#include <mfem.hpp>

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace dg_sim::hdg
{

/**
 * Own the standard derived/IGR output fields while aliasing one conservative
 * byNODES L2 field.  Solver storage stays backend-private; only an explicit
 * committed or prepared export refreshes these host views.
 */
class HdgEulerParaViewFields
{
public:
  HdgEulerParaViewFields(mfem::ParMesh& mesh,
                         mfem::L2_FECollection& collection,
                         mfem::ParGridFunction& conservative,
                         const bool igr_active,
                         const double gamma,
                         const double density_floor)
    : scalar_space_(&mesh, &collection, 1, mfem::Ordering::byNODES),
      vector_space_(&mesh, &collection, 2, mfem::Ordering::byNODES),
      conservative_(conservative),
      state_views_(scalar_space_, vector_space_, conservative_, 2),
      velocity_(vector_space_, conservative_, 2, density_floor),
      pressure_(scalar_space_, conservative_, 2, gamma),
      sigma_(igr_active
          ? std::make_unique<mfem::ParGridFunction>(&scalar_space_)
          : nullptr),
      gamma_(gamma),
      density_floor_(density_floor)
  {
    MFEM_VERIFY(conservative_.ParFESpace() &&
                    conservative_.ParFESpace()->GetParMesh() == &mesh &&
                    conservative_.ParFESpace()->GetVDim() == 4 &&
                    conservative_.ParFESpace()->GetOrdering() ==
                        mfem::Ordering::byNODES &&
                    conservative_.Size() == 4 * scalar_space_.GetVSize(),
                "HDG Euler output requires a four-component byNODES L2 field.");
    MFEM_VERIFY(std::isfinite(gamma_) && gamma_ > 1.0 &&
                    std::isfinite(density_floor_) && density_floor_ > 0.0,
                "HDG Euler output requires valid thermodynamic floors.");
  }

  bool FreshHostOutput(
      dg::hdg::generic::HdgEulerHostInteriorFields& destination) noexcept
  {
    destination = {};
    try
    {
      const int scalar_size = scalar_space_.GetVSize();
      if (scalar_size < 0 || conservative_.Size() != 4 * scalar_size ||
          (sigma_ && sigma_->Size() != scalar_size))
      {
        return false;
      }
      double* const conservative = conservative_.HostWrite();
      double* const sigma = sigma_ ? sigma_->HostWrite() : nullptr;
      if ((scalar_size != 0 && conservative == nullptr) ||
          (sigma_ && scalar_size != 0 && sigma == nullptr))
      {
        return false;
      }
      destination = {conservative, sigma,
                     static_cast<std::size_t>(scalar_size)};
      return true;
    }
    catch (...) { return false; }
  }

  /** Refresh the standard velocity and pressure projections from exported U. */
  bool UpdateDerived() noexcept
  {
    try
    {
      velocity_.Update();
      pressure_.Update();
      return velocity_.velocity.CheckFinite() == 0 &&
          pressure_.pressure.CheckFinite() == 0;
    }
    catch (...) { return false; }
  }

  std::vector<HDGInteriorFieldView> Fields()
  {
    std::vector<HDGInteriorFieldView> fields;
    fields.reserve(sigma_ ? 6U : 5U);
    fields.push_back({"density", state_views_.density});
    fields.push_back({"momentum", state_views_.momentum});
    fields.push_back({"energy", state_views_.energy});
    fields.push_back({"velocity", velocity_.velocity});
    if (sigma_) fields.push_back({"sigma", *sigma_});
    fields.push_back({"pressure", pressure_.pressure});
    return fields;
  }

private:
  mfem::ParFiniteElementSpace scalar_space_;
  mfem::ParFiniteElementSpace vector_space_;
  mfem::ParGridFunction& conservative_;
  dg_sim::EulerStateViews state_views_;
  dg_sim::EulerVelocityView velocity_;
  dg_sim::EulerPressureView pressure_;
  std::unique_ptr<mfem::ParGridFunction> sigma_;
  double gamma_ = 1.4;
  double density_floor_ = 1.0e-16;
};

} // namespace dg_sim::hdg
