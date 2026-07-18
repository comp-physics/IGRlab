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
      weakdiv_sigma_elem_dof_map_[moff + k] = sigma_sdofs[k];
    }

    sigma_shape.SetSize(sigma_ldof0);
    const int ioff = e * dof * sigma_ldof0;
    for (int j = 0; j < dof; ++j)
    {
      sigma_fe.CalcShape(state_nodes.IntPoint(j), sigma_shape);
      const int joff = ioff + j * sigma_ldof0;
      for (int k = 0; k < sigma_ldof0; ++k)
      {
        weakdiv_sigma_interp_(joff + k) = sigma_shape(k);
      }
    }
  }

  weakdiv_sigma_fes_ = sfes;
  weakdiv_sigma_fes_sequence_ = seq;
  weakdiv_sigma_ldof_ = sigma_ldof0;
  weakdiv_sigma_ready_ = true;
  return true;
}

void DGHyperbolicConservationLaws::AddPreassembledWeakDivergenceDeviceEuler_(
  const Vector& x_in,
  Vector& z_inout,
  bool use_sigma,
  const ParGridFunction* sigma_field,
  real_t sigma_rho_floor) const
{
  MFEM_VERIFY(weakdiv_apply_ready_ && invmass_apply_ready_,
              "Packed weak-divergence device path requires packed element maps.");
  MFEM_VERIFY(weakdiv_elem_dof_ == invmass_elem_dof_,
              "Weak-divergence and inverse-mass dof layouts are inconsistent.");

  const int ne = vfes.GetNE();
  const int dof = weakdiv_elem_dof_;
  const int neq = num_equations;
  const int ldof = dof * neq;

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

#ifdef DG_IGR_HAS_ACCEL_FORALL
  PackedWeakDivEulerKernel kernel;
  kernel.dof = dof;
  kernel.neq = neq;
  kernel.dim = dim;
  kernel.ldof = ldof;
  kernel.sigma_ldof = sigma_ldof;
  kernel.gamma = dgsem_gamma_;
  kernel.sigma_rho_floor = sigma_rho_floor;
  kernel.use_sigma = use_sigma;
  kernel.state_map = invmass_elem_vdof_map_.Read();
  kernel.weakdiv = weakdiv_data_flat_.Read();
  kernel.x = x_in.Read();
  kernel.z = z_inout.ReadWrite();
  kernel.sigma_map = sigma_map;
  kernel.sigma_interp = sigma_interp;
  kernel.sigma_vals = sigma_vals;
  mfem::forall(ne, kernel);
#else
  auto state_map = invmass_elem_vdof_map_.HostRead();
  auto weak_flat = weakdiv_data_flat_.HostRead();
  auto xh = x_in.HostRead();
  auto zh = z_inout.HostReadWrite();
  auto sigma_map_h = weakdiv_sigma_elem_dof_map_.HostRead();
  auto sigma_interp_h = weakdiv_sigma_interp_.HostRead();
  auto sigma_vals_h =
    use_sigma ? sigma_field->HostRead() : static_cast<const real_t*>(nullptr);

  const int energy_eq = 1 + dim;
  const real_t gm1 = dgsem_gamma_ - 1.0;
  for (int e = 0; e < ne; ++e)
  {
    const int state_off = e * ldof;
    const int weak_off = e * dof * dof * dim;
    const int sigma_map_off = e * sigma_ldof;
    const int sigma_interp_off = e * dof * sigma_ldof;

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
                flux_md += sigma_j;
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
            flux_ed += sigma_j * mom[d] * inv_rho_sigma;
          }
          sum_e += weak_flat[wbase + d] * flux_ed;
        }
        zh[state_map[state_off + i * neq + energy_eq]] += sum_e;
      }
    }
  }
#endif

  if (Device::Allows(Backend::DEVICE_MASK) && !weakdiv_volume_path_logged_ &&
      mfem::Mpi::Root())
  {
    std::cout << "[dg-device] event=weakdiv-volume-path"
              << " selected=packed-mfem-forall-"
              << (use_sigma ? "euler-igr" : "euler")
              << " backend=device\n";
    weakdiv_volume_path_logged_ = true;
  }
}

void DGHyperbolicConservationLaws::AddPreassembledWeakDivergence_(
  const Vector& x_in,
  Vector& z_inout) const
{
  if (weakdiv.empty())
  {
    return;
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
    const ParGridFunction* sigma_field = nullptr;
    real_t sigma_rho_floor = 1.0e-12;
    if (flux_igr && flux_igr->GetSigmaField())
    {
      sigma_field = flux_igr->GetSigmaField();
      MFEM_VERIFY(BuildWeakDivSigmaData_(*sigma_field),
                  "DG device-closure requires weak-div sigma interpolation "
                  "metadata on device.");
      use_sigma = true;
      sigma_rho_floor = flux_igr->GetVelocityRhoFloor();
    }

    AddPreassembledWeakDivergenceDeviceEuler_(x_in,
                                              z_inout,
                                              use_sigma,
                                              sigma_field,
                                              sigma_rho_floor);
    return;
  }

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
        fluxFunction.ComputeFlux(state, *Tr, flux_j);
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
    return;
  }

  // Fallback for mixed element spaces where packed maps are unavailable.
  Vector state;
  DenseMatrix flux;
  DenseMatrix Xmat;
  DenseMatrix Zmat;
  Array<int> vdofs;
  Vector xval, zval;

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
      fluxFunction.ComputeFlux(state, *Tr, flux_j);
    }

    z_inout.GetSubVector(vdofs, zval);
    Zmat.UseExternalData(zval.GetData(), dof, num_equations);
    mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, Zmat);
    z_inout.SetSubVector(vdofs, zval);
  }

  if (Device::Allows(Backend::DEVICE_MASK) && !weakdiv_volume_path_logged_ &&
      mfem::Mpi::Root())
  {
    std::cout << "[dg-device] event=weakdiv-volume-path"
              << " selected=host-subvector-fallback"
              << " backend=device\n";
    weakdiv_volume_path_logged_ = true;
  }
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
