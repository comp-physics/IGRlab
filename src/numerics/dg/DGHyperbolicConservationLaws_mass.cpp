#include "igrlab/compat.hpp"
#include "DGHyperbolicConservationLaws_impl.hpp"
#include "utils/profiling/nvtx.hpp"

namespace mfem
{

#include "DGHyperbolicConservationLaws_device_kernels.hpp"
#include "DGHyperbolicConservationLaws_mass_reuse_kernels.hpp"

namespace
{

bool UsePackedInverseMassRowReuse_(bool requested, bool dense_rows,
                                  bool diagonal, int components)
{
  return requested && dense_rows && !diagonal && sizeof(real_t) == sizeof(double) &&
         components >= 2 && components <= 5;
}

bool AssembleCollocatedInverseMass(const FiniteElement& fe,
                                  ElementTransformation& transformation,
                                  const IntegrationRule& rule,
                                  Vector& shape,
                                  DenseMatrix& inverse)
{
  const int dof = fe.GetDof();
  if (rule.GetNPoints() != dof)
  {
    return false;
  }

  // Admit only exact nodal collocation under the same physical shapes and
  // quadrature as MassIntegrator. No near-zero tolerance or lumping is used.
  inverse.SetSize(dof);
  inverse = 0.0;
  shape.SetSize(dof);
  for (int q = 0; q < rule.GetNPoints(); ++q)
  {
    const IntegrationPoint& point = rule.IntPoint(q);
    transformation.SetIntPoint(&point);
    fe.CalcPhysShape(transformation, shape);
    int node = -1;
    for (int i = 0; i < dof; ++i)
    {
      if (shape(i) == 0.0)
      {
        continue;
      }
      if (shape(i) != 1.0 || node >= 0)
      {
        return false;
      }
      node = i;
    }
    if (node < 0 || inverse(node, node) != 0.0)
    {
      return false;
    }
    const real_t weight = transformation.Weight() * point.weight;
    if (!std::isfinite(weight) || weight <= 0.0)
    {
      return false;
    }
    inverse(node, node) = weight;
  }
  for (int i = 0; i < dof; ++i)
  {
    const real_t reciprocal = 1.0 / inverse(i, i);
    if (!std::isfinite(reciprocal))
    {
      return false;
    }
    inverse(i, i) = reciprocal;
  }
  return true;
}

struct PackedDiagonalInverseMassKernel
{
  int neq = 0;
  const int* map = nullptr;
  const real_t* diagonal = nullptr;
  const real_t* zin = nullptr;
  real_t* yout = nullptr;

  MFEM_HOST_DEVICE void operator()(int entry) const
  {
    const int vdof = map[entry];
    yout[vdof] = diagonal[entry / neq] * zin[vdof];
  }
};

} // namespace

// -----------------------------------------------------------------------------
// Compute inverse mass matrices
// -----------------------------------------------------------------------------
//
// Assemble exact nodally collocated masses directly; otherwise retain
// InverseIntegrator(MassIntegrator). Store the dense inverse per element for:
//     y_T = M_T^{-1} z_T
// cheaply in Mult().
//
void DGHyperbolicConservationLaws::ComputeInvMass()
{
  ::igrlab::NVTXRange range("DG/setup-inverse-mass");
  InverseIntegrator inv_mass(new MassIntegrator());
  invmass.resize(vfes.GetNE());
  static IntegrationRules lgl_rules(0, Quadrature1D::GaussLobatto);
  Vector shape;

  for (int i = 0; i < vfes.GetNE(); i++)
  {
    const FiniteElement& fe = *vfes.GetFE(i);
    const int dof = fe.GetDof();
    ElementTransformation& transformation = *vfes.GetElementTransformation(i);
    const IntegrationRule* collocated_rule = nullptr;
    if (const auto* tbe = dynamic_cast<const TensorBasisElement*>(&fe))
    {
      if (tbe->GetBasisType() == BasisType::GaussLobatto)
      {
        const int order = std::max(1, 2 * fe.GetOrder() - 1);
        collocated_rule = &lgl_rules.Get(fe.GetGeomType(), order);
        inv_mass.SetIntRule(collocated_rule);
      }
      else
      {
        inv_mass.SetIntRule(nullptr);
        if (tbe->GetBasisType() == BasisType::GaussLegendre)
        {
          collocated_rule = &MassIntegrator::GetRule(fe, fe, transformation);
        }
      }
    }
    else
    {
      inv_mass.SetIntRule(nullptr);
    }
    invmass[i].SetSize(dof);
    if (!collocated_rule ||
        !AssembleCollocatedInverseMass(fe, transformation, *collocated_rule,
                                      shape, invmass[i]))
    {
      inv_mass.AssembleElementMatrix(fe, transformation, invmass[i]);
    }
  }
}

void DGHyperbolicConservationLaws::BuildInvMassApplyData_()
{
  // Reject malformed execution policy even when this backend keeps the
  // original element algorithm or the mass turns out to be diagonal.
  (void)UsePackedDenseRows_(vfes, 0, num_equations);
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
  // This map may already have a device copy from the previous topology.
  // Rebuilding its host entries must invalidate that copy, including after
  // SetSize grows device-valid storage.
  int* const vdof_map = invmass_elem_vdof_map_.HostWrite();

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
        vdof_map[e_off + i * num_equations + eq] = vdof;
      }
    }
  }

  // Inspect the assembled inverses themselves: compress only exact diagonal
  // blocks, without a tolerance or a change to mass quadrature/inversion.
  // Any nonzero off-diagonal entry retains the existing dense apply.
  bool diagonal = true;
  for (int e = 0; diagonal && e < ne; ++e)
  {
    for (int c = 0; diagonal && c < dof; ++c)
    {
      for (int r = 0; r < dof; ++r)
      {
        if (r != c && invmass[e](r, c) != 0.0)
        {
          diagonal = false;
          break;
        }
      }
    }
  }

  invmass_data_flat_.SetSize(ne * dof * (diagonal ? 1 : dof));
  real_t* minv_flat = invmass_data_flat_.HostWrite();
  for (int e = 0; e < ne; ++e)
  {
    if (diagonal)
    {
      for (int i = 0; i < dof; ++i)
      {
        minv_flat[e * dof + i] = invmass[e](i, i);
      }
      continue;
    }
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

  if (!::igrlab::EnvFlagEnabledDefault("DG_IGR_DG_NATIVE_MASS_INV", false))
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
         ::igrlab::EnvFlagEnabledDefault("DG_IGR_DG_PACKED_INVMASS_DEVICE",
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
  if (PackedPointersOverlap_(zin, z_in.Size(), yout, y_out.Size()))
  {
    // Dense row outputs must not overwrite inputs still needed by another
    // output. Offset aliases also require a snapshot for diagonal masses.
    Vector input_copy(z_in.Size());
    input_copy.UseDevice(true);
    input_copy = z_in;
    ApplyInverseMassPacked_(input_copy, y_out);
    return;
  }
  if (invmass_data_flat_.Size() == ne * dof)
  {
    PackedDiagonalInverseMassKernel kernel;
    kernel.neq = neq;
    kernel.map = map;
    kernel.diagonal = minv;
    kernel.zin = zin;
    kernel.yout = yout;
#ifdef DG_IGR_HAS_ACCEL_FORALL
    mfem::forall(ne * ldof, kernel);
#else
    for (int entry = 0; entry < ne * ldof; ++entry)
    {
      kernel(entry);
    }
#endif
    return;
  }

#ifdef DG_IGR_HAS_ACCEL_FORALL
  // invmass_data_flat_ stores each local inverse mass in column-major order:
  // entry (row=i, col=j) is minv[moff + j*dof + i].
  const bool dense_rows = UsePackedDenseRows_(vfes, dof, neq);
  const bool row_reuse = UsePackedInverseMassRowReuse_(
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_DG_INVMASS_ROW_REUSE", false),
    dense_rows, false, neq);
  if (row_reuse)
  {
    switch (neq)
    {
      case 2:
        mfem::forall(ne * dof, PackedInverseMassRowReuseKernel<2>{
          dof, map, minv, zin, yout});
        break;
      case 3:
        mfem::forall(ne * dof, PackedInverseMassRowReuseKernel<3>{
          dof, map, minv, zin, yout});
        break;
      case 4:
        mfem::forall(ne * dof, PackedInverseMassRowReuseKernel<4>{
          dof, map, minv, zin, yout});
        break;
      case 5:
        mfem::forall(ne * dof, PackedInverseMassRowReuseKernel<5>{
          dof, map, minv, zin, yout});
        break;
      default:
        MFEM_ABORT("Unsupported inverse-mass row-reuse component count.");
    }
  }
  else if (dense_rows)
  {
    mfem::forall(ne * ldof, PackedInverseMassRowsKernel{
      dof, neq, map, minv, zin, yout});
  }
  else
  {
    PackedInverseMassKernel kernel;
    kernel.dof = dof;
    kernel.neq = neq;
    kernel.ldof = ldof;
    kernel.map = map;
    kernel.minv = minv;
    kernel.zin = zin;
    kernel.yout = yout;
    mfem::forall(ne, kernel);
  }
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
  const real_t* input = z_in.HostRead();
  real_t* output = y_out.HostWrite();
  if (PackedPointersOverlap_(input, z_in.Size(), output, y_out.Size()))
  {
    Vector input_copy(z_in);
    ApplyInverseMassHost_(input_copy, y_out);
    return;
  }
  if (invmass_apply_ready_)
  {
    const int ne = vfes.GetNE();
    const int dof = invmass_elem_dof_;
    const int neq = num_equations;
    const int ldof = dof * neq;

    auto map = invmass_elem_vdof_map_.HostRead();
    auto minv = invmass_data_flat_.HostRead();
    const real_t* zin = input;
    real_t* yout = output;

    if (invmass_data_flat_.Size() == ne * dof)
    {
      for (int entry = 0; entry < ne * ldof; ++entry)
      {
        const int vdof = map[entry];
        yout[vdof] = minv[entry / neq] * zin[vdof];
      }
      return;
    }

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
  const bool dense_rows =
    UsePackedDenseRows_(vfes, invmass_elem_dof_, num_equations);
  const bool row_reuse_requested = ::igrlab::EnvFlagEnabledDefault(
    "DG_IGR_DG_INVMASS_ROW_REUSE", false);
  const bool diagonal =
    invmass_data_flat_.Size() == vfes.GetNE() * invmass_elem_dof_;
  const bool row_reuse = UsePackedInverseMassRowReuse_(
    row_reuse_requested, dense_rows, diagonal, num_equations);
  if (Device::Allows(Backend::DEVICE_MASK))
  {
    MFEM_VERIFY(invmass_apply_ready_,
                "DG device-closure requires packed inverse-mass apply data.");
    ApplyInverseMassPacked_(z_in, y_out);
    if (!invmass_path_logged_ && mfem::Mpi::Root())
    {
      std::cout << "[dg-device] event=invmass-path"
                << " selected=packed-mfem-forall"
                << " dense_execution="
                << (diagonal ? "diagonal" : (dense_rows ? "rows" : "element"))
                << " backend=device"
                << " invmass_row_reuse_requested=" << int(row_reuse_requested)
                << " invmass_row_reuse=" << int(row_reuse)
                << '\n';
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
