#include "DGHyperbolicConservationLaws_impl.hpp"

namespace mfem
{

#include "DGHyperbolicConservationLaws_device_kernels.hpp"

// -----------------------------------------------------------------------------
// Compute inverse mass matrices
// -----------------------------------------------------------------------------
//
// For each element T, assemble M_T^{-1} by applying InverseIntegrator(MassIntegrator).
// We store the dense inverse per element, so we can do:
//     y_T = M_T^{-1} z_T
// cheaply in Mult().
//
void DGHyperbolicConservationLaws::ComputeInvMass()
{
  InverseIntegrator inv_mass(new MassIntegrator());
  invmass.resize(vfes.GetNE());
  static IntegrationRules lgl_rules(0, Quadrature1D::GaussLobatto);

  for (int i = 0; i < vfes.GetNE(); i++)
  {
    const FiniteElement& fe = *vfes.GetFE(i);
    const int dof = fe.GetDof();
    if (const auto* tbe = dynamic_cast<const TensorBasisElement*>(&fe))
    {
      if (tbe->GetBasisType() == BasisType::GaussLobatto)
      {
        const int order = std::max(1, 2 * fe.GetOrder() - 1);
        inv_mass.SetIntRule(&lgl_rules.Get(fe.GetGeomType(), order));
      }
      else
      {
        inv_mass.SetIntRule(nullptr);
      }
    }
    else
    {
      inv_mass.SetIntRule(nullptr);
    }
    invmass[i].SetSize(dof);
    inv_mass.AssembleElementMatrix(fe,
                                   *vfes.GetElementTransformation(i),
                                   invmass[i]);
  }
}

void DGHyperbolicConservationLaws::BuildInvMassApplyData_()
{
  invmass_apply_ready_ = false;
  face_data_dirty_ = true;
  invmass_elem_dof_ = 0;
  invmass_data_flat_.SetSize(0);
  invmass_elem_vdof_map_.SetSize(0);

  const int ne = vfes.GetNE();
  if (ne <= 0 || invmass.empty())
  {
    return;
  }

  const int dof = vfes.GetFE(0)->GetDof();
  if (dof <= 0)
  {
    return;
  }

  // Packed kernel currently assumes constant dof count across elements.
  for (int e = 0; e < ne; ++e)
  {
    const int edof = vfes.GetFE(e)->GetDof();
    if (edof != dof)
    {
      return;
    }
    MFEM_VERIFY(invmass[e].Height() == dof && invmass[e].Width() == dof,
                "Invalid inverse-mass block size for element " << e << ".");
  }

  const int ldof = dof * num_equations;
  invmass_elem_vdof_map_.SetSize(ne * ldof);

  Array<int> sdofs;
  for (int e = 0; e < ne; ++e)
  {
    vfes.GetElementDofs(e, sdofs);
    MFEM_VERIFY(sdofs.Size() == dof,
                "Scalar element dof count mismatch for element " << e << ".");

    const int e_off = e * ldof;
    for (int i = 0; i < dof; ++i)
    {
      for (int eq = 0; eq < num_equations; ++eq)
      {
        int vdof = vfes.DofToVDof(sdofs[i], eq);
        if (vdof < 0)
        {
          vdof = -1 - vdof;
        }
        invmass_elem_vdof_map_[e_off + i * num_equations + eq] = vdof;
      }
    }
  }

  invmass_data_flat_.SetSize(ne * dof * dof);
  real_t* minv_flat = invmass_data_flat_.HostWrite();
  for (int e = 0; e < ne; ++e)
  {
    const int m_off = e * dof * dof;
    for (int c = 0; c < dof; ++c)
    {
      for (int r = 0; r < dof; ++r)
      {
        minv_flat[m_off + c * dof + r] = invmass[e](r, c);
      }
    }
  }

  invmass_elem_dof_ = dof;
  invmass_apply_ready_ = true;
}

void DGHyperbolicConservationLaws::BuildNativeMassInverseSolver_()
{
  mass_inv_solver_ready_ = false;
  mass_inv_solver_.reset();
  mass_inv_prec_.reset();
  mass_form_.reset();
  mass_inv_solver_fallback_warned_ = false;
#ifdef MFEM_USE_MPI
  par_mass_form_.reset();
#endif

  if (!::dg_igr::EnvFlagEnabledDefault("DG_IGR_DG_NATIVE_MASS_INV", false))
  {
    return;
  }

  // Only build this path for accelerated backends; CPU keeps the exact local
  // dense inverse-mass apply.
  if (!Device::Allows(Backend::DEVICE_MASK))
  {
    return;
  }

  Array<int> ess_tdof_list;

#ifdef MFEM_USE_MPI
  if (auto* pvfes = dynamic_cast<ParFiniteElementSpace*>(&vfes))
  {
    par_mass_form_ = std::make_unique<ParBilinearForm>(pvfes);
    par_mass_form_->AddDomainIntegrator(new VectorMassIntegrator());
    par_mass_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    par_mass_form_->Assemble();

    mass_inv_prec_ =
      std::make_unique<OperatorJacobiSmoother>(*par_mass_form_, ess_tdof_list);
    mass_inv_solver_ = std::make_unique<CGSolver>(pvfes->GetComm());
    mass_inv_solver_->SetRelTol(1.0e-12);
    mass_inv_solver_->SetAbsTol(0.0);
    mass_inv_solver_->SetMaxIter(120);
    mass_inv_solver_->SetPrintLevel(-1);
    mass_inv_solver_->iterative_mode = false;
    mass_inv_solver_->SetPreconditioner(*mass_inv_prec_);
    mass_inv_solver_->SetOperator(*par_mass_form_);

    mass_inv_solver_ready_ = true;
    return;
  }
#endif

  mass_form_ = std::make_unique<BilinearForm>(&vfes);
  mass_form_->AddDomainIntegrator(new VectorMassIntegrator());
  mass_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  mass_form_->Assemble();

  mass_inv_prec_ =
    std::make_unique<OperatorJacobiSmoother>(*mass_form_, ess_tdof_list);
  mass_inv_solver_ = std::make_unique<CGSolver>();
  mass_inv_solver_->SetRelTol(1.0e-12);
  mass_inv_solver_->SetAbsTol(0.0);
  mass_inv_solver_->SetMaxIter(120);
  mass_inv_solver_->SetPrintLevel(-1);
  mass_inv_solver_->iterative_mode = false;
  mass_inv_solver_->SetPreconditioner(*mass_inv_prec_);
  mass_inv_solver_->SetOperator(*mass_form_);

  mass_inv_solver_ready_ = true;
}

bool DGHyperbolicConservationLaws::UseNativeMassInverseSolver_() const
{
  return mass_inv_solver_ready_ && (mass_inv_solver_ != nullptr);
}

bool DGHyperbolicConservationLaws::UsePackedInverseMassDevice_() const
{
  return invmass_apply_ready_ && Device::Allows(Backend::DEVICE_MASK) &&
         ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DG_PACKED_INVMASS_DEVICE",
                                         true);
}

void DGHyperbolicConservationLaws::ApplyInverseMassPacked_(
  const Vector& z_in,
  Vector& y_out) const
{
  if (!invmass_apply_ready_)
  {
    MFEM_VERIFY(!Device::Allows(Backend::DEVICE_MASK),
                "DG device-closure requires packed inverse-mass metadata.");
    ApplyInverseMassHost_(z_in, y_out);
    return;
  }

  const int ne = vfes.GetNE();
  const int dof = invmass_elem_dof_;
  const int neq = num_equations;
  if (ne <= 0 || dof <= 0 || neq <= 0)
  {
    y_out = 0.0;
    return;
  }

  const int ldof = dof * neq;
  auto map = invmass_elem_vdof_map_.Read();
  auto minv = invmass_data_flat_.Read();
  auto zin = z_in.Read();
  auto yout = y_out.Write();

#ifdef DG_IGR_HAS_ACCEL_FORALL
  // invmass_data_flat_ stores each local inverse mass in column-major order:
  // entry (row=i, col=j) is minv[moff + j*dof + i].
  PackedInverseMassKernel kernel;
  kernel.dof = dof;
  kernel.neq = neq;
  kernel.ldof = ldof;
  kernel.map = map;
  kernel.minv = minv;
  kernel.zin = zin;
  kernel.yout = yout;
  mfem::forall(ne, kernel);
#else
  for (int e = 0; e < ne; ++e)
  {
    const int voff = e * ldof;
    const int moff = e * dof * dof;
    for (int eq = 0; eq < neq; ++eq)
    {
      for (int i = 0; i < dof; ++i)
      {
        real_t sum = 0.0;
        for (int j = 0; j < dof; ++j)
        {
          const int zj = map[voff + j * neq + eq];
          sum += minv[moff + j * dof + i] * zin[zj];
        }
        const int yi = map[voff + i * neq + eq];
        yout[yi] = sum;
      }
    }
  }
#endif
}

void DGHyperbolicConservationLaws::ApplyInverseMassHost_(
  const Vector& z_in,
  Vector& y_out) const
{
  if (invmass_apply_ready_)
  {
    const int ne = vfes.GetNE();
    const int dof = invmass_elem_dof_;
    const int neq = num_equations;
    const int ldof = dof * neq;

    auto map = invmass_elem_vdof_map_.HostRead();
    auto minv = invmass_data_flat_.HostRead();
    auto zin = z_in.HostRead();
    auto yout = y_out.HostWrite();

    for (int e = 0; e < ne; ++e)
    {
      const int voff = e * ldof;
      const int moff = e * dof * dof;
      for (int eq = 0; eq < neq; ++eq)
      {
        for (int i = 0; i < dof; ++i)
        {
          real_t sum = 0.0;
          for (int j = 0; j < dof; ++j)
          {
            const int zj = map[voff + j * neq + eq];
            sum += minv[moff + j * dof + i] * zin[zj];
          }
          const int yi = map[voff + i * neq + eq];
          yout[yi] = sum;
        }
      }
    }
    return;
  }

  Array<int> vdofs;
  Vector zval;
  DenseMatrix Zmat, Ymat;

  for (int e = 0; e < vfes.GetNE(); ++e)
  {
    const int dof = vfes.GetFE(e)->GetDof();
    vfes.GetElementVDofs(e, vdofs);
    z_in.GetSubVector(vdofs, zval);
    Zmat.UseExternalData(zval.GetData(), dof, num_equations);
    Ymat.SetSize(dof, num_equations);
    mfem::Mult(invmass[e], Zmat, Ymat);
    y_out.SetSubVector(vdofs, Ymat.GetData());
  }
}

void DGHyperbolicConservationLaws::ApplyInverseMass_(const Vector& z_in,
                                                     Vector& y_out) const
{
  if (Device::Allows(Backend::DEVICE_MASK))
  {
    MFEM_VERIFY(invmass_apply_ready_,
                "DG device-closure requires packed inverse-mass apply data.");
    ApplyInverseMassPacked_(z_in, y_out);
    if (!invmass_path_logged_ && mfem::Mpi::Root())
    {
      std::cout << "[dg-device] event=invmass-path"
                << " selected=packed-mfem-forall"
                << " backend=device\n";
      invmass_path_logged_ = true;
    }
    return;
  }

  if (UseNativeMassInverseSolver_())
  {
    mass_inv_solver_->Mult(z_in, y_out);
    const bool converged = mass_inv_solver_->GetConverged();
    const real_t final_norm = mass_inv_solver_->GetFinalNorm();
    if (converged && std::isfinite(final_norm))
    {
      return;
    }
    if (!mass_inv_solver_fallback_warned_ && mfem::Mpi::Root())
    {
      std::cout << "[dg-device] event=invmass-fallback"
                << " reason=nonconverged-native-solve"
                << " iterations=" << mass_inv_solver_->GetNumIterations()
                << " final_norm=" << final_norm << "\n";
      mass_inv_solver_fallback_warned_ = true;
    }
    if (!invmass_path_logged_ && mfem::Mpi::Root())
    {
      std::cout << "[dg-device] event=invmass-path"
                << " selected=mfem-native-cg"
                << " backend=device\n";
      invmass_path_logged_ = true;
    }
  }

  ApplyInverseMassHost_(z_in, y_out);
  if (!invmass_path_logged_ && mfem::Mpi::Root())
  {
    std::cout << "[dg-device] event=invmass-path"
              << " selected=host-fallback"
              << " backend="
              << (Device::Allows(Backend::DEVICE_MASK) ? "device" : "cpu")
              << "\n";
    invmass_path_logged_ = true;
  }
}

} // namespace mfem
