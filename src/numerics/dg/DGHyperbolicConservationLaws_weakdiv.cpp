#include "igrlab/compat.hpp"
#include "DGHyperbolicConservationLaws_impl.hpp"

namespace mfem
{

#include "DGHyperbolicConservationLaws_device_kernels.hpp"

void DGHyperbolicConservationLaws::BuildWeakDivApplyData_()
{
  weakdiv_apply_ready_ = false;
  face_data_dirty_ = true;
  weakdiv_elem_dof_ = 0;
  weakdiv_data_flat_.SetSize(0);
  weakdiv_shared_map_.reset();
  weakdiv_shared_row_sums_.SetSize(0);
  weakdiv_shared_column_sums_.SetSize(0);
  weakdiv_shared_mass_fractions_.SetSize(0);
  weakdiv_shared_moment_correction_.SetSize(0);
  weakdiv_shared_flux_.SetSize(0);
  weakdiv_volume_speed_.SetSize(0);
  weakdiv_q_sigma_telemetry_.SetSize(0);
  weakdiv_volume_speed_reduce_workspace_.SetSize(0);

  if (weakdiv.empty() || !invmass_apply_ready_)
  {
    return;
  }

  const int ne = vfes.GetNE();
  const int dof = invmass_elem_dof_;
  if (ne <= 0 || dof <= 0)
  {
    return;
  }

  for (int e = 0; e < ne; ++e)
  {
    MFEM_VERIFY(weakdiv[e].Height() == dof && weakdiv[e].Width() == dof * dim,
                "Invalid weak-divergence block size for element " << e << ".");
  }

  weakdiv_data_flat_.SetSize(ne * dof * dof * dim);
  real_t* weak_flat = weakdiv_data_flat_.HostWrite();
  for (int e = 0; e < ne; ++e)
  {
    const int eoff = e * dof * dof * dim;
    for (int i = 0; i < dof; ++i)
    {
      const int ioff = eoff + i * dof * dim;
      for (int j = 0; j < dof; ++j)
      {
        const int joff = ioff + j * dim;
        for (int d = 0; d < dim; ++d)
        {
          weak_flat[joff + d] = weakdiv[e](i, j * dim + d);
        }
      }
    }
  }

  weakdiv_elem_dof_ = dof;
  const auto* flux_igr = dynamic_cast<const FluxWithIGR*>(
    &formIntegrator->GetFluxFunction());
  if (flux_igr && flux_igr->GetSigmaField() &&
      flux_igr->GetInvariantDomainFlux())
  {
    dg_igr_detail::InitializeFrozenSigmaTelemetry(
      weakdiv_q_sigma_telemetry_, ne * dof);
  }
  weakdiv_apply_ready_ = true;

  // Sigma interpolation cache depends on the state-node layout; invalidate.
  weakdiv_sigma_ready_ = false;
  weakdiv_sigma_fes_ = nullptr;
  weakdiv_sigma_fes_sequence_ = -1;
  weakdiv_sigma_ldof_ = 0;
  weakdiv_sigma_elem_dof_map_.SetSize(0);
  weakdiv_sigma_interp_.SetSize(0);
}

bool DGHyperbolicConservationLaws::BuildWeakDivSigmaData_(
  const ParGridFunction& sigma) const
{
  const FiniteElementSpace* sfes = sigma.ParFESpace();
  if (!sfes || !weakdiv_apply_ready_)
  {
    return false;
  }

  const long seq = sfes->GetSequence();
  if (weakdiv_sigma_ready_ && weakdiv_sigma_fes_ == sfes &&
      weakdiv_sigma_fes_sequence_ == seq)
  {
    return true;
  }

  weakdiv_sigma_ready_ = false;
  weakdiv_sigma_fes_ = nullptr;
  weakdiv_sigma_fes_sequence_ = -1;
  weakdiv_sigma_ldof_ = 0;
  weakdiv_sigma_elem_dof_map_.SetSize(0);
  weakdiv_sigma_interp_.SetSize(0);

  const int ne = vfes.GetNE();
  const int dof = weakdiv_elem_dof_;
  if (sfes->GetNE() != ne || dof <= 0)
  {
    return false;
  }

  const int sigma_ldof0 = sfes->GetFE(0)->GetDof();
  if (sigma_ldof0 <= 0)
  {
    return false;
  }
  for (int e = 1; e < ne; ++e)
  {
    if (sfes->GetFE(e)->GetDof() != sigma_ldof0)
    {
      return false;
    }
  }

  weakdiv_sigma_elem_dof_map_.SetSize(ne * sigma_ldof0);
  weakdiv_sigma_interp_.SetSize(ne * dof * sigma_ldof0);
  // Topology/space updates rebuild these values on the host; discard any
  // device-valid views retained by a resize before writing the new metadata.
  int* const sigma_map = weakdiv_sigma_elem_dof_map_.HostWrite();
  real_t* const sigma_interp = weakdiv_sigma_interp_.HostWrite();

  Array<int> sigma_sdofs;
  Vector sigma_shape;
  for (int e = 0; e < ne; ++e)
  {
    const FiniteElement& state_fe = *vfes.GetFE(e);
    const FiniteElement& sigma_fe = *sfes->GetFE(e);
    const IntegrationRule& state_nodes = state_fe.GetNodes();
    if (state_nodes.Size() != dof)
    {
      return false;
    }

    sfes->GetElementDofs(e, sigma_sdofs);
    if (sigma_sdofs.Size() != sigma_ldof0)
    {
      return false;
    }
    const int moff = e * sigma_ldof0;
    for (int k = 0; k < sigma_ldof0; ++k)
    {
      // Scalar H1/L2 spaces are expected to have non-negative element dof ids.
      if (sigma_sdofs[k] < 0)
      {
        return false;
      }
      sigma_map[moff + k] = sigma_sdofs[k];
    }

    sigma_shape.SetSize(sigma_ldof0);
    const int ioff = e * dof * sigma_ldof0;
    for (int j = 0; j < dof; ++j)
    {
      sigma_fe.CalcShape(state_nodes.IntPoint(j), sigma_shape);
      const int joff = ioff + j * sigma_ldof0;
      for (int k = 0; k < sigma_ldof0; ++k)
      {
        sigma_interp[joff + k] = sigma_shape(k);
      }
    }
  }

  weakdiv_sigma_fes_ = sfes;
  weakdiv_sigma_fes_sequence_ = seq;
  weakdiv_sigma_ldof_ = sigma_ldof0;
  weakdiv_sigma_ready_ = true;
  return true;
}

real_t DGHyperbolicConservationLaws::AddPreassembledWeakDivergenceDeviceEuler_(
  const Vector& x_in,
  Vector& z_inout,
  bool use_sigma,
  const ParGridFunction* sigma_field,
  real_t sigma_rho_floor,
  real_t invariant_domain_pressure_floor,
  bool invariant_domain_flux,
  bool track_volume_speed) const
{
  MFEM_VERIFY(weakdiv_apply_ready_ && invmass_apply_ready_,
              "Packed weak-divergence device path requires packed element maps.");
  MFEM_VERIFY(weakdiv_elem_dof_ == invmass_elem_dof_,
              "Weak-divergence and inverse-mass dof layouts are inconsistent.");

  const int ne = vfes.GetNE();
  const int dof = weakdiv_elem_dof_;
  const int neq = num_equations;
  const int ldof = dof * neq;

  const auto shared_precision = ::igrlab::shared_reference::RequestedPrecision();
  const bool shared = shared_precision != ::igrlab::shared_reference::Precision::Off;
  if (shared && !weakdiv_shared_map_)
  {
    MFEM_VERIFY(dim == 2 && neq == 4 && dof <= 256 &&
                  Device::Allows(Backend::CUDA_MASK),
                "Shared DG weak divergence requires CUDA 2D Euler.");
    // Share only one identical affine geometry, never a representative cell.
    // Assembly at translated coordinates may differ by ordinary roundoff;
    // bound that difference against the actual FP64 operator norm explicitly.
    real_t reference_edges[4] = {0.0, 0.0, 0.0, 0.0};
    real_t reference_norm = 0.0, difference_norm = 0.0;
    for (int i = 0; i < dof; ++i)
    {
      real_t sum = 0.0;
      for (int j = 0; j < dof * dim; ++j) { sum += std::abs(weakdiv[0](i, j)); }
      reference_norm = std::max(reference_norm, sum);
    }
    for (int e = 0; e < ne; ++e)
    {
      const FiniteElement& fe = *vfes.GetFE(e);
      ElementTransformation* tr = vfes.GetElementTransformation(e);
      MFEM_VERIFY(fe.GetGeomType() == Geometry::SQUARE && tr->Order() == 1,
                  "Shared DG weak divergence requires affine quadrilaterals.");
      // Bilinear quads are affine iff their mixed corner difference is zero.
      // Evaluate corners (exact 0/1 shape values), rather than comparing
      // Jacobians recomputed at irrational high-order reference nodes.
      Vector corners[4] = {Vector(2), Vector(2), Vector(2), Vector(2)};
      for (int q = 0; q < 4; ++q)
      {
        IntegrationPoint corner;
        corner.Set2(q % 2, q / 2);
        tr->Transform(corner, corners[q]);
      }
      for (int d = 0; d < 2; ++d)
      {
        MFEM_VERIFY(corners[3][d] - corners[1][d] == corners[2][d] - corners[0][d],
                    "Shared DG weak divergence rejects non-affine geometry.");
        for (int direction = 0; direction < 2; ++direction)
        {
          const real_t edge = corners[direction + 1][d] - corners[0][d];
          if (e == 0) { reference_edges[d + 2 * direction] = edge; }
          MFEM_VERIFY(edge == reference_edges[d + 2 * direction],
                      "Shared DG weak divergence requires identical affine element geometry.");
        }
      }
      for (int i = 0; i < dof; ++i)
      {
        real_t difference = 0.0;
        for (int j = 0; j < dof * dim; ++j)
        {
          MFEM_VERIFY(std::isfinite(weakdiv[e](i, j)), "Nonfinite shared DG matrix.");
          difference += std::abs(weakdiv[e](i, j) - weakdiv[0](i, j));
        }
        difference_norm = std::max(difference_norm, difference);
      }
    }
    const real_t roundoff_limit = 64.0 * dof * std::numeric_limits<real_t>::epsilon();
    MFEM_VERIFY(reference_norm > 0.0 && difference_norm <= roundoff_limit * reference_norm,
                "Shared DG element matrices differ beyond the assembly-roundoff bound.");
    // For a nodal weak divergence, sum_i grad(phi_i)=0. Preserve the actual
    // assembled FP64 column sums, including their quadrature roundoff, rather
    // than assuming that every stored column sums to bitwise zero.
    weakdiv_shared_column_sums_.SetSize(dof * dim);
    weakdiv_shared_column_sums_.UseDevice(true);
    real_t* column_sums = weakdiv_shared_column_sums_.HostWrite();
    real_t maximum_column_sum = 0.0;
    for (int j = 0; j < dof * dim; ++j)
    {
      long double sum = 0.0L;
      for (int i = 0; i < dof; ++i) { sum += weakdiv[0](i, j); }
      column_sums[j] = static_cast<real_t>(sum);
      maximum_column_sum = std::max(maximum_column_sum, std::abs(column_sums[j]));
    }
    MFEM_VERIFY(maximum_column_sum <= roundoff_limit * reference_norm,
                "Shared DG matrix does not have the weak-divergence constant-test moment.");

    // The already assembled inverse must be exactly diagonal: its reciprocal
    // is M_ii and hence (M 1)_i. Use each cell's actual mass, not reconstructed
    // reference weights or a row-uniform weak correction. The resulting strong
    // correction is constant within the cell after the unchanged inverse mass.
    MFEM_VERIFY(invmass_data_flat_.Size() == ne * dof,
                "Shared DG moment preservation requires an exactly diagonal assembled mass.");
    weakdiv_shared_mass_fractions_.SetSize(ne * dof);
    weakdiv_shared_mass_fractions_.UseDevice(true);
    real_t* mass_fractions = weakdiv_shared_mass_fractions_.HostWrite();
    for (int e = 0; e < ne; ++e)
    {
      const auto* tensor = dynamic_cast<const TensorBasisElement*>(vfes.GetFE(e));
      MFEM_VERIFY(tensor && tensor->GetBasisType() == BasisType::GaussLegendre,
                  "Shared DG moment preservation requires the nodal Gauss-Legendre mass.");
      long double total_mass = 0.0L;
      for (int i = 0; i < dof; ++i)
      {
        const real_t inverse = invmass[e](i, i);
        const real_t mass = 1.0 / inverse;
        MFEM_VERIFY(std::isfinite(inverse) && inverse > 0.0 &&
                      std::isfinite(mass) && mass > 0.0,
                    "Shared DG moment preservation requires finite positive mass weights.");
        mass_fractions[e * dof + i] = mass;
        total_mass += mass;
      }
      MFEM_VERIFY(std::isfinite(static_cast<real_t>(total_mass)) && total_mass > 0.0L,
                  "Invalid shared DG cell mass.");
      for (int i = 0; i < dof; ++i)
      {
        mass_fractions[e * dof + i] =
          static_cast<real_t>(mass_fractions[e * dof + i] / total_mass);
      }
    }
    weakdiv_shared_moment_correction_.SetSize(ne * neq);
    weakdiv_shared_moment_correction_.UseDevice(true);
    weakdiv_shared_column_sums_.Read();
    weakdiv_shared_mass_fractions_.Read();
    weakdiv_shared_moment_correction_.Write();
    std::cout << "[shared-reference-map] event=operator-equivalence method=dg-weakdiv"
              << " geometry=identical-affine-parallelograms matrix_relative_inf_error="
              << difference_norm / reference_norm << " limit=" << roundoff_limit
              << std::endl;
    std::cout << "[shared-reference-map] event=moment-policy method=dg-weakdiv"
              << " target=original-fp64-weak-moment distribution=assembled-M1"
              << " precision=fp64 correction=centered-product"
              << " matrix_column_sum_relative=" << maximum_column_sum / reference_norm
              << " additional_workspace_bytes="
              << (std::size_t(dof) * dim + std::size_t(ne) * dof +
                  std::size_t(ne) * neq) * sizeof(real_t)
              << std::endl;
    weakdiv_shared_row_sums_.SetSize(dof, dim);
    weakdiv_shared_row_sums_.GetMemory().UseDevice(true);
    real_t* sums = weakdiv_shared_row_sums_.HostWrite();
    for (int i = 0; i < dof; ++i)
    {
      for (int d = 0; d < dim; ++d)
      {
        sums[i + dof * d] = 0.0;
        for (int j = 0; j < dof; ++j)
        {
          sums[i + dof * d] += weakdiv[0](i, j * dim + d);
        }
      }
    }
    weakdiv_shared_map_ = std::make_unique<::igrlab::shared_reference::Map>();
    weakdiv_shared_map_->Configure(weakdiv[0], ne * neq, shared_precision, "dg-weakdiv");
    weakdiv_shared_flux_.SetSize(ne * neq * dof * dim);
    weakdiv_shared_flux_.UseDevice(true);
    weakdiv_shared_flux_.Write();
  }
  MFEM_VERIFY(!shared || weakdiv_shared_map_->GetPrecision() == shared_precision,
              "Shared reference precision cannot change on a live DG operator.");

  const int* sigma_map = nullptr;
  const real_t* sigma_interp = nullptr;
  const real_t* sigma_vals = nullptr;
  int sigma_ldof = 0;
  if (use_sigma)
  {
    MFEM_VERIFY(sigma_field != nullptr,
                "Device weak-divergence sigma path requires a sigma field.");
    MFEM_VERIFY(weakdiv_sigma_ready_,
                "Device weak-divergence sigma interpolation is not ready.");
    sigma_map = weakdiv_sigma_elem_dof_map_.Read();
    sigma_interp = weakdiv_sigma_interp_.Read();
    sigma_vals = sigma_field->Read();
    sigma_ldof = weakdiv_sigma_ldof_;
  }

  if (invariant_domain_flux &&
      weakdiv_q_sigma_telemetry_.Size() !=
        ne * dof * dg_igr_detail::kFrozenSigmaTelemetryWidth)
  {
    dg_igr_detail::InitializeFrozenSigmaTelemetry(
      weakdiv_q_sigma_telemetry_, ne * dof);
  }
  MFEM_VERIFY(!track_volume_speed || invariant_domain_flux,
              "Volume-node speed tracking is restricted to exact "
              "invariant-domain Euler+IGR fluxes.");

#ifdef DG_IGR_HAS_ACCEL_FORALL
  if (track_volume_speed && weakdiv_volume_speed_.Size() != ne)
  {
    weakdiv_volume_speed_.SetSize(ne);
  }
  PackedWeakDivEulerKernel kernel;
  kernel.dof = dof;
  kernel.neq = neq;
  kernel.dim = dim;
  kernel.ldof = ldof;
  kernel.sigma_ldof = sigma_ldof;
  kernel.gamma = dgsem_gamma_;
  kernel.sigma_rho_floor = sigma_rho_floor;
  kernel.invariant_domain_pressure_floor = invariant_domain_pressure_floor;
  kernel.use_sigma = use_sigma;
  kernel.invariant_domain_flux = invariant_domain_flux;
  kernel.track_volume_speed = track_volume_speed;
  kernel.state_map = invmass_elem_vdof_map_.Read();
  kernel.weakdiv = shared ? nullptr : weakdiv_data_flat_.Read();
  kernel.x = x_in.Read();
  kernel.z = shared ? nullptr : z_inout.ReadWrite();
  kernel.volume_speed =
    track_volume_speed ? weakdiv_volume_speed_.Write() : nullptr;
  kernel.q_sigma_telemetry =
    invariant_domain_flux ? weakdiv_q_sigma_telemetry_.ReadWrite() : nullptr;
  kernel.sigma_map = sigma_map;
  kernel.sigma_interp = sigma_interp;
  kernel.sigma_vals = sigma_vals;
  kernel.packed_flux = shared ? weakdiv_shared_flux_.Write() : nullptr;
  mfem::forall(ne, kernel);
  if (shared)
  {
    const int k = dof * dim, columns = ne * neq;
    const real_t* flux = weakdiv_shared_flux_.Read();
    real_t* input = weakdiv_shared_map_->Input().Write();
    mfem::forall(k * columns,
      PackedSharedWeakDivCenterKernel{k, dim, flux, input});
    weakdiv_shared_map_->Apply();
    const real_t* output = weakdiv_shared_map_->Output().Read();
    real_t* correction = weakdiv_shared_moment_correction_.Write();
    mfem::forall(columns, PackedSharedWeakDivMomentKernel{
      dof, k, weakdiv_shared_map_->Input().Read(), output,
      weakdiv_shared_column_sums_.Read(), correction});
    const real_t* row_sums = weakdiv_shared_row_sums_.Read();
    const int* map = invmass_elem_vdof_map_.Read();
    real_t* z_out = z_inout.ReadWrite();
    mfem::forall(dof * columns,
      PackedSharedWeakDivRestoreKernel{dof, dim, neq, flux, output, row_sums,
        weakdiv_shared_mass_fractions_.Read(), weakdiv_shared_moment_correction_.Read(),
        map, z_out});
  }
#else
  auto state_map = invmass_elem_vdof_map_.HostRead();
  auto weak_flat = weakdiv_data_flat_.HostRead();
  auto xh = x_in.HostRead();
  auto zh = z_inout.HostReadWrite();
  auto sigma_map_h = weakdiv_sigma_elem_dof_map_.HostRead();
  auto sigma_interp_h = weakdiv_sigma_interp_.HostRead();
  auto sigma_vals_h =
    use_sigma ? sigma_field->HostRead() : static_cast<const real_t*>(nullptr);
  if (track_volume_speed && weakdiv_volume_speed_.Size() != ne)
  {
    weakdiv_volume_speed_.SetSize(ne);
  }
  real_t* const volume_speed_h =
    track_volume_speed ? weakdiv_volume_speed_.HostWrite() : nullptr;
  double* const q_sigma_telemetry_h =
    invariant_domain_flux ? weakdiv_q_sigma_telemetry_.HostReadWrite()
                          : nullptr;

  const int energy_eq = 1 + dim;
  const real_t gm1 = dgsem_gamma_ - 1.0;
  for (int e = 0; e < ne; ++e)
  {
    const int state_off = e * ldof;
    const int weak_off = e * dof * dof * dim;
    const int sigma_map_off = e * sigma_ldof;
    const int sigma_interp_off = e * dof * sigma_ldof;
    real_t element_max_speed = 0.0;

    for (int j = 0; j < dof; ++j)
    {
      const real_t rho = xh[state_map[state_off + j * neq + 0]];
      const real_t inv_rho = 1.0 / rho;
      real_t mom[3] = {0.0, 0.0, 0.0};
      real_t mom2 = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        mom[d] = xh[state_map[state_off + j * neq + (1 + d)]];
        mom2 += mom[d] * mom[d];
      }
      const real_t energy = xh[state_map[state_off + j * neq + energy_eq]];
      const real_t kinetic = 0.5 * mom2 * inv_rho;
      const real_t pressure = gm1 * (energy - kinetic);
      const real_t H = (energy + pressure) * inv_rho;

      real_t sigma_j = 0.0;
      if (use_sigma)
      {
        const int interp_row_off = sigma_interp_off + j * sigma_ldof;
        for (int k = 0; k < sigma_ldof; ++k)
        {
          const int sk = sigma_map_h[sigma_map_off + k];
          sigma_j += sigma_interp_h[interp_row_off + k] * sigma_vals_h[sk];
        }
      }
      const real_t inv_rho_sigma =
        (use_sigma ? 1.0 / std::max(rho, sigma_rho_floor) : 0.0);
      if (invariant_domain_flux)
      {
        constexpr int telemetry_width =
          dg_igr_detail::kFrozenSigmaTelemetryWidth;
        double* const slot =
          q_sigma_telemetry_h + (e * dof + j) * telemetry_width;
        const double q_sigma = static_cast<double>(dgsem_gamma_) * pressure +
          (static_cast<double>(dgsem_gamma_) - 1.0) * sigma_j;
        const double ratio =
          q_sigma / (static_cast<double>(dgsem_gamma_) * pressure);
        slot[0] += 1.0;
        if (!std::isfinite(q_sigma) || !std::isfinite(ratio))
        {
          slot[2] += 1.0;
        }
        if (std::isfinite(q_sigma))
        {
          slot[1] += q_sigma <= 0.0 ? 1.0 : 0.0;
          slot[3] = std::min(slot[3], q_sigma);
        }
        if (std::isfinite(ratio))
        {
          slot[4] = std::min(slot[4], ratio);
        }
      }
      if (track_volume_speed)
      {
        const real_t transport_speed = std::sqrt(mom2) * inv_rho;
        const real_t base_speed =
          transport_speed + std::sqrt(dgsem_gamma_ * pressure * inv_rho);
        const real_t nodal_speed =
          IGRFluxInvariantDomainSpeed(base_speed,
                                      rho,
                                      pressure,
                                      transport_speed,
                                      sigma_j,
                                      dgsem_gamma_,
                                      sigma_rho_floor,
                                      invariant_domain_pressure_floor);
        element_max_speed = std::max(element_max_speed, nodal_speed);
      }

      for (int i = 0; i < dof; ++i)
      {
        const int wbase = weak_off + i * dof * dim + j * dim;

        real_t sum_rho = 0.0;
        for (int d = 0; d < dim; ++d)
        {
          sum_rho += weak_flat[wbase + d] * mom[d];
        }
        zh[state_map[state_off + i * neq + 0]] += sum_rho;

        for (int m = 0; m < dim; ++m)
        {
          real_t sum_m = 0.0;
          for (int d = 0; d < dim; ++d)
          {
            real_t flux_md = mom[m] * mom[d] * inv_rho;
            if (m == d)
            {
              flux_md += pressure;
              if (use_sigma)
              {
                flux_md += ::igrlab::frozen_scalar_igr_algebra::
                  TensorSigmaMomentumDiagonalTerm(sigma_j);
              }
            }
            sum_m += weak_flat[wbase + d] * flux_md;
          }
          zh[state_map[state_off + i * neq + (1 + m)]] += sum_m;
        }

        real_t sum_e = 0.0;
        for (int d = 0; d < dim; ++d)
        {
          real_t flux_ed = mom[d] * H;
          if (use_sigma)
          {
            flux_ed += ::igrlab::frozen_scalar_igr_algebra::
              TensorSigmaEnergyTermFromInverseDensity(
                sigma_j, mom[d], inv_rho_sigma);
          }
          sum_e += weak_flat[wbase + d] * flux_ed;
        }
        zh[state_map[state_off + i * neq + energy_eq]] += sum_e;
      }
    }
    if (track_volume_speed)
    {
      volume_speed_h[e] = element_max_speed;
    }
  }
#endif

  if (Device::Allows(Backend::DEVICE_MASK) && !weakdiv_volume_path_logged_ &&
      mfem::Mpi::Root())
  {
    std::cout << "[dg-device] event=weakdiv-volume-path"
              << " selected=" << (shared ? "shared-reference-" : "packed-mfem-forall-")
              << (use_sigma ? "euler-igr" : "euler")
              << " backend=device\n";
    weakdiv_volume_path_logged_ = true;
  }
  return track_volume_speed
           ? ReduceFaceMaxSpeed_(weakdiv_volume_speed_,
                                 weakdiv_volume_speed_reduce_workspace_)
           : 0.0;
}

real_t DGHyperbolicConservationLaws::AddPreassembledWeakDivergence_(
  const Vector& x_in,
  Vector& z_inout) const
{
  if (weakdiv.empty())
  {
    return 0.0;
  }

  const FluxFunction& fluxFunction = formIntegrator->GetFluxFunction();

  if (Device::Allows(Backend::DEVICE_MASK))
  {
    MFEM_VERIFY(weakdiv_apply_ready_ && invmass_apply_ready_,
                "DG device-closure requires packed weak-divergence and map "
                "data (no host fallback).");

    const auto* flux_igr = dynamic_cast<const FluxWithIGR*>(&fluxFunction);
    const bool euler_base =
      (dynamic_cast<const EulerFlux*>(&fluxFunction) != nullptr) ||
      (flux_igr && flux_igr->HasEulerBase());
    MFEM_VERIFY(euler_base,
                "DG device-closure currently supports Euler/Euler+IGR weak-"
                "divergence only.");

    bool use_sigma = false;
    bool invariant_domain_flux = false;
    const ParGridFunction* sigma_field = nullptr;
    real_t sigma_rho_floor = 1.0e-12;
    real_t invariant_domain_pressure_floor =
      igrlab::LimiterDefaults::kPMin;
    if (flux_igr && flux_igr->GetSigmaField())
    {
      sigma_field = flux_igr->GetSigmaField();
      MFEM_VERIFY(BuildWeakDivSigmaData_(*sigma_field),
                  "DG device-closure requires weak-div sigma interpolation "
                  "metadata on device.");
      use_sigma = true;
      sigma_rho_floor = flux_igr->GetVelocityRhoFloor();
      invariant_domain_pressure_floor =
        flux_igr->GetInvariantDomainPressureFloor();
      invariant_domain_flux = flux_igr->GetInvariantDomainFlux();
    }
    MFEM_VERIFY(!invariant_domain_flux ||
                  flux_igr->GetGamma() == dgsem_gamma_,
                "Packed DG invariant-domain volume flux requires the "
                "operator and FluxWithIGR to use one gamma.");

    const bool track_volume_speed =
      invariant_domain_volume_speed_tracking_ && use_sigma &&
      invariant_domain_flux;
    return AddPreassembledWeakDivergenceDeviceEuler_(
      x_in,
      z_inout,
      use_sigma,
      sigma_field,
      sigma_rho_floor,
      invariant_domain_pressure_floor,
      invariant_domain_flux,
      track_volume_speed);
  }

  const auto* host_flux_igr =
    dynamic_cast<const FluxWithIGR*>(&fluxFunction);
  const bool host_exact_mode =
    host_flux_igr && host_flux_igr->GetSigmaField() &&
    host_flux_igr->GetInvariantDomainFlux();
  const bool track_host_volume_speed =
    invariant_domain_volume_speed_tracking_ && host_exact_mode;
  int host_volume_sample_slots = 0;
  if (host_exact_mode)
  {
    for (int e = 0; e < vfes.GetNE(); ++e)
    {
      host_volume_sample_slots += vfes.GetFE(e)->GetDof();
    }
    if (weakdiv_q_sigma_telemetry_.Size() !=
        host_volume_sample_slots *
          dg_igr_detail::kFrozenSigmaTelemetryWidth)
    {
      dg_igr_detail::InitializeFrozenSigmaTelemetry(
        weakdiv_q_sigma_telemetry_, host_volume_sample_slots);
    }
  }
  double* const host_volume_telemetry =
    host_exact_mode ? weakdiv_q_sigma_telemetry_.HostReadWrite() : nullptr;

  // FluxWithIGR::ComputeFlux returns its invariant-domain speed and therefore
  // performs the additional characteristic square roots unconditionally.
  // Reproduce its tensor flux directly so fixed-step host runs retain q_sigma
  // telemetry without paying for a volume speed that no caller consumes.
  const auto evaluate_host_volume_flux =
    [&](const Vector& state,
        ElementTransformation& transformation,
        DenseMatrix& flux,
        int telemetry_slot) -> real_t
  {
    if (!host_exact_mode)
    {
      fluxFunction.ComputeFlux(state, transformation, flux);
      return 0.0;
    }

    const real_t base_speed = host_flux_igr->GetBaseFlux().ComputeFlux(
      state, transformation, flux);
    const real_t sigma = host_flux_igr->GetSigmaAtPoint(transformation);
    const real_t rho_floor = host_flux_igr->GetVelocityRhoFloor();
    ::igrlab::frozen_scalar_igr_algebra::AddTensorSigmaFlux(
      dim,
      state.GetData(),
      1,
      sigma,
      rho_floor,
      flux.GetData(),
      1,
      flux.Height());

    const real_t density = state(0);
    const real_t safe_density = std::max(density, rho_floor);
    real_t momentum_sq = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      momentum_sq += state(1 + d) * state(1 + d);
    }
    const real_t gamma = host_flux_igr->GetGamma();
    const real_t pressure =
      (gamma - 1.0) *
      (state(1 + dim) - 0.5 * momentum_sq / safe_density);

    constexpr int telemetry_width =
      dg_igr_detail::kFrozenSigmaTelemetryWidth;
    double* const slot =
      host_volume_telemetry + telemetry_slot * telemetry_width;
    const double q_sigma = static_cast<double>(gamma) * pressure +
      (static_cast<double>(gamma) - 1.0) * sigma;
    const double ratio = q_sigma / (static_cast<double>(gamma) * pressure);
    slot[0] += 1.0;
    if (!std::isfinite(q_sigma) || !std::isfinite(ratio))
    {
      slot[2] += 1.0;
    }
    if (std::isfinite(q_sigma))
    {
      slot[1] += q_sigma <= 0.0 ? 1.0 : 0.0;
      slot[3] = std::min(slot[3], q_sigma);
    }
    if (std::isfinite(ratio))
    {
      slot[4] = std::min(slot[4], ratio);
    }

    if (!track_host_volume_speed)
    {
      return 0.0;
    }
    const real_t transport_speed =
      std::sqrt(momentum_sq) / safe_density;
    return IGRFluxInvariantDomainSpeed(
      base_speed,
      density,
      pressure,
      transport_speed,
      sigma,
      gamma,
      rho_floor,
      host_flux_igr->GetInvariantDomainPressureFloor());
  };

  // Fast host path: use the packed node-major/equation-minor element map to
  // avoid per-element GetSubVector/SetSubVector staging.
  if (invmass_apply_ready_)
  {
    const int ne = vfes.GetNE();
    const int dof = invmass_elem_dof_;
    const int neq = num_equations;
    const int ldof = dof * neq;

    auto map = invmass_elem_vdof_map_.HostRead();
    auto xh = x_in.HostRead();
    auto zh = z_inout.HostReadWrite();

    Vector state(neq);
    DenseMatrix flux(neq, dim * dof);
    real_t max_volume_speed = 0.0;
    for (int e = 0; e < ne; ++e)
    {
      ElementTransformation* Tr = vfes.GetElementTransformation(e);
      const FiniteElement& fe = *vfes.GetFE(e);
      const IntegrationRule& nodes = fe.GetNodes();
      MFEM_ASSERT(nodes.Size() == dof,
                  "FE is not nodal or GetNodes() != dof; "
                  "need to evaluate at an appropriate rule.");

      const int voff = e * ldof;
      for (int j = 0; j < dof; ++j)
      {
        for (int eq = 0; eq < neq; ++eq)
        {
          const int xj = map[voff + j * neq + eq];
          state(eq) = xh[xj];
        }

        const IntegrationPoint& ip = nodes.IntPoint(j);
        Tr->SetIntPoint(&ip);
        DenseMatrix flux_j(flux.GetData() + j * neq * dim, neq, dim);
        const real_t node_speed =
          evaluate_host_volume_flux(state, *Tr, flux_j, e * dof + j);
        if (track_host_volume_speed)
        {
          max_volume_speed = std::max(max_volume_speed, node_speed);
        }
      }

      for (int i = 0; i < dof; ++i)
      {
        for (int eq = 0; eq < neq; ++eq)
        {
          real_t sum = 0.0;
          for (int j = 0; j < dof; ++j)
          {
            const int col_base = j * dim;
            for (int d = 0; d < dim; ++d)
            {
              const int col = col_base + d;
              sum += weakdiv[e](i, col) * flux(eq, col);
            }
          }
          const int zi = map[voff + i * neq + eq];
          zh[zi] += sum;
        }
      }
    }

    if (Device::Allows(Backend::DEVICE_MASK) && !weakdiv_volume_path_logged_ &&
        mfem::Mpi::Root())
    {
      std::cout << "[dg-device] event=weakdiv-volume-path"
                << " selected=host-packed-map"
                << " backend=device\n";
      weakdiv_volume_path_logged_ = true;
    }
    return max_volume_speed;
  }

  // Fallback for mixed element spaces where packed maps are unavailable.
  Vector state;
  DenseMatrix flux;
  DenseMatrix Xmat;
  DenseMatrix Zmat;
  Array<int> vdofs;
  Vector xval, zval;
  real_t max_volume_speed = 0.0;
  int telemetry_slot_offset = 0;

  for (int i = 0; i < vfes.GetNE(); ++i)
  {
    ElementTransformation* Tr = vfes.GetElementTransformation(i);
    const FiniteElement& fe = *vfes.GetFE(i);
    const int dof = fe.GetDof();
    vfes.GetElementVDofs(i, vdofs);

    x_in.GetSubVector(vdofs, xval);
    Xmat.UseExternalData(xval.GetData(), dof, num_equations);

    flux.SetSize(num_equations, dim * dof);
    const IntegrationRule& nodes = fe.GetNodes();
    MFEM_ASSERT(nodes.Size() == dof,
                "FE is not nodal or GetNodes() != dof; "
                "need to evaluate at an appropriate rule.");

    for (int j = 0; j < dof; ++j)
    {
      const IntegrationPoint& ip = nodes.IntPoint(j);
      Tr->SetIntPoint(&ip);
      Xmat.GetRow(j, state);
      DenseMatrix flux_j(flux.GetData() + j * num_equations * dim,
                         num_equations,
                         dim);
      const real_t node_speed = evaluate_host_volume_flux(
        state, *Tr, flux_j, telemetry_slot_offset + j);
      if (track_host_volume_speed)
      {
        max_volume_speed = std::max(max_volume_speed, node_speed);
      }
    }

    z_inout.GetSubVector(vdofs, zval);
    Zmat.UseExternalData(zval.GetData(), dof, num_equations);
    mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, Zmat);
    z_inout.SetSubVector(vdofs, zval);
    telemetry_slot_offset += dof;
  }

  if (Device::Allows(Backend::DEVICE_MASK) && !weakdiv_volume_path_logged_ &&
      mfem::Mpi::Root())
  {
    std::cout << "[dg-device] event=weakdiv-volume-path"
              << " selected=host-subvector-fallback"
              << " backend=device\n";
    weakdiv_volume_path_logged_ = true;
  }
  return max_volume_speed;
}

// -----------------------------------------------------------------------------
// Preassemble weak divergence matrices
// -----------------------------------------------------------------------------
//
// We need a matrix that maps nodal flux values to test-function contributions:
//
//   weakdiv[i] (dof × (dof*dim)) with columns ordered as
//      [node0_x, node0_y, (node0_z)], [node1_x, node1_y, (node1_z)], ...
//
// We get this by assembling the transpose of GradientIntegrator:
//   TransposeIntegrator(GradientIntegrator()) gives (ϕ_i, ∇ψ_j)_T.
//
// Then we reorder the columns to ByDim-per-node layout to match how we pack
// the nodal flux tensor F(U_j) ∈ ℝ^{num_eq×dim} for each node j.
//
void DGHyperbolicConservationLaws::ComputeWeakDivergence()
{
  TransposeIntegrator weak_div(new GradientIntegrator());
  DenseMatrix tmp;

  weakdiv.resize(vfes.GetNE());

  for (int i = 0; i < vfes.GetNE(); i++)
  {
    const FiniteElement& fe = *vfes.GetFE(i);
    int dof = fe.GetDof();

    tmp.SetSize(dof, dof * dim);
    weak_div.AssembleElementMatrix2(fe,
                                    fe,
                                    *vfes.GetElementTransformation(i),
                                    tmp);

    weakdiv[i].SetSize(dof, dof * dim);
    for (int j = 0; j < dof; j++)
    {
      for (int d = 0; d < dim; d++)
      {
        // Copy column from the assembled block [all dofs for dimension d]
        // into the “ByDim per node” layout at (node j, dimension d).
        weakdiv[i].SetCol(j * dim + d, tmp.GetColumn(d * dof + j));
      }
    }
  }
}

} // namespace mfem
