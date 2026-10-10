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
    : HdgEulerParaViewFields(mesh, collection, conservative, igr_active,
                             false, gamma, density_floor)
  {
  }

  HdgEulerParaViewFields(mfem::ParMesh& mesh,
                         mfem::L2_FECollection& collection,
                         mfem::ParGridFunction& conservative,
                         const bool igr_active,
                         const bool enable_zhang_shu_diagnostics,
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
      zhang_shu_collection_(enable_zhang_shu_diagnostics
          ? std::make_unique<mfem::L2_FECollection>(0, mesh.Dimension())
          : nullptr),
      zhang_shu_space_(zhang_shu_collection_
          ? std::make_unique<mfem::ParFiniteElementSpace>(
                &mesh, zhang_shu_collection_.get(), 1,
                mfem::Ordering::byNODES)
          : nullptr),
      zhang_shu_active_(zhang_shu_space_
          ? std::make_unique<mfem::ParGridFunction>(zhang_shu_space_.get())
          : nullptr),
      zhang_shu_theta_rho_(zhang_shu_space_
          ? std::make_unique<mfem::ParGridFunction>(zhang_shu_space_.get())
          : nullptr),
      zhang_shu_theta_p_(zhang_shu_space_
          ? std::make_unique<mfem::ParGridFunction>(zhang_shu_space_.get())
          : nullptr),
      zhang_shu_theta_rho_elements_(zhang_shu_space_
          ? static_cast<std::size_t>(mesh.GetNE()) : 0U, 1.0),
      zhang_shu_theta_p_elements_(zhang_shu_space_
          ? static_cast<std::size_t>(mesh.GetNE()) : 0U, 1.0),
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
    MFEM_VERIFY(!zhang_shu_space_ ||
                    (zhang_shu_space_->GetParMesh() == &mesh &&
                     zhang_shu_space_->GetVDim() == 1 &&
                     zhang_shu_space_->GetVSize() == mesh.GetNE()),
                "HDG Zhang-Shu output requires one discontinuous order-zero "
                "degree of freedom per local element.");
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
      const int element_count = scalar_space_.GetParMesh()->GetNE();
      if (scalar_size < 0 || conservative_.Size() != 4 * scalar_size ||
          element_count < 0 || (sigma_ && sigma_->Size() != scalar_size) ||
          (zhang_shu_space_ &&
           (zhang_shu_theta_rho_elements_.size() !=
                static_cast<std::size_t>(element_count) ||
            zhang_shu_theta_p_elements_.size() !=
                static_cast<std::size_t>(element_count))))
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
      destination = {
          conservative,
          sigma,
          static_cast<std::size_t>(scalar_size),
          zhang_shu_space_ && element_count != 0
              ? zhang_shu_theta_rho_elements_.data() : nullptr,
          zhang_shu_space_ && element_count != 0
              ? zhang_shu_theta_p_elements_.data() : nullptr,
          zhang_shu_space_
              ? static_cast<std::size_t>(element_count) : 0U};
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
          pressure_.pressure.CheckFinite() == 0 &&
          UpdateZhangShuDerived();
    }
    catch (...) { return false; }
  }

  std::vector<HDGInteriorFieldView> Fields()
  {
    std::vector<HDGInteriorFieldView> fields;
    fields.reserve((sigma_ ? 6U : 5U) +
                   (zhang_shu_space_ ? 3U : 0U));
    fields.push_back({"density", state_views_.density});
    fields.push_back({"momentum", state_views_.momentum});
    fields.push_back({"energy", state_views_.energy});
    fields.push_back({"velocity", velocity_.velocity});
    if (sigma_) fields.push_back({"sigma", *sigma_});
    fields.push_back({"pressure", pressure_.pressure});
    if (zhang_shu_space_)
    {
      fields.push_back({"zhang_shu_ever_active", *zhang_shu_active_});
      fields.push_back({"zhang_shu_theta_rho_min_cumulative",
                        *zhang_shu_theta_rho_});
      fields.push_back({"zhang_shu_theta_p_min_cumulative",
                        *zhang_shu_theta_p_});
    }
    return fields;
  }

private:
  bool UpdateZhangShuDerived() noexcept
  {
    if (!zhang_shu_space_) return true;
    try
    {
      mfem::ParMesh* const mesh = zhang_shu_space_->GetParMesh();
      const int element_count = mesh ? mesh->GetNE() : -1;
      const int field_size = zhang_shu_space_->GetVSize();
      if (!mesh || element_count < 0 || field_size < 0 ||
          field_size != element_count ||
          zhang_shu_active_->Size() != field_size ||
          zhang_shu_theta_rho_->Size() != field_size ||
          zhang_shu_theta_p_->Size() != field_size ||
          zhang_shu_theta_rho_elements_.size() !=
              static_cast<std::size_t>(element_count) ||
          zhang_shu_theta_p_elements_.size() !=
              static_cast<std::size_t>(element_count))
      {
        return false;
      }

      double* const active = zhang_shu_active_->HostWrite();
      double* const theta_rho = zhang_shu_theta_rho_->HostWrite();
      double* const theta_p = zhang_shu_theta_p_->HostWrite();
      if (field_size != 0 &&
          (!active || !theta_rho || !theta_p))
      {
        return false;
      }
      std::vector<unsigned char> assigned(
          static_cast<std::size_t>(field_size), 0U);
      mfem::Array<int> element_vdofs;
      for (int element = 0; element < element_count; ++element)
      {
        zhang_shu_space_->GetElementVDofs(element, element_vdofs);
        if (element_vdofs.Size() != 1) return false;
        const int encoded = element_vdofs[0];
        const double sign = encoded >= 0 ? 1.0 : -1.0;
        const int dof = encoded >= 0 ? encoded : -1 - encoded;
        if (dof < 0 || dof >= field_size ||
            assigned[static_cast<std::size_t>(dof)] != 0U)
        {
          return false;
        }
        const double element_theta_rho =
            zhang_shu_theta_rho_elements_[static_cast<std::size_t>(element)];
        const double element_theta_p =
            zhang_shu_theta_p_elements_[static_cast<std::size_t>(element)];
        if (!std::isfinite(element_theta_rho) ||
            !std::isfinite(element_theta_p) ||
            element_theta_rho < 0.0 || element_theta_rho > 1.0 ||
            element_theta_p < 0.0 || element_theta_p > 1.0)
        {
          return false;
        }
        active[dof] = sign *
            ((element_theta_rho < 1.0 || element_theta_p < 1.0) ? 1.0 : 0.0);
        theta_rho[dof] = sign * element_theta_rho;
        theta_p[dof] = sign * element_theta_p;
        assigned[static_cast<std::size_t>(dof)] = 1U;
      }
      return zhang_shu_active_->CheckFinite() == 0 &&
          zhang_shu_theta_rho_->CheckFinite() == 0 &&
          zhang_shu_theta_p_->CheckFinite() == 0;
    }
    catch (...) { return false; }
  }

  mfem::ParFiniteElementSpace scalar_space_;
  mfem::ParFiniteElementSpace vector_space_;
  mfem::ParGridFunction& conservative_;
  dg_sim::EulerStateViews state_views_;
  dg_sim::EulerVelocityView velocity_;
  dg_sim::EulerPressureView pressure_;
  std::unique_ptr<mfem::ParGridFunction> sigma_;
  std::unique_ptr<mfem::L2_FECollection> zhang_shu_collection_;
  std::unique_ptr<mfem::ParFiniteElementSpace> zhang_shu_space_;
  std::unique_ptr<mfem::ParGridFunction> zhang_shu_active_;
  std::unique_ptr<mfem::ParGridFunction> zhang_shu_theta_rho_;
  std::unique_ptr<mfem::ParGridFunction> zhang_shu_theta_p_;
  std::vector<double> zhang_shu_theta_rho_elements_;
  std::vector<double> zhang_shu_theta_p_elements_;
  double gamma_ = 1.4;
  double density_floor_ = 1.0e-16;
};

} // namespace dg_sim::hdg
