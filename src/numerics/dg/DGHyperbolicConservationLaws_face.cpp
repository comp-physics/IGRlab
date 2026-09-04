#include "DGHyperbolicConservationLaws_impl.hpp"

namespace mfem
{

#include "DGHyperbolicConservationLaws_device_kernels.hpp"

const BdrHyperbolicFluxIntegrator*
DGHyperbolicConservationLaws::FindBdrIntegrator_(int attr) const
{
  if (attr <= 0)
  {
    return nullptr;
  }
  for (size_t i = 0; i < ext_bdr_int_.size(); ++i)
  {
    const auto* marker = ext_bdr_markers_[i].get();
    if (!marker || attr > marker->Size() || (*marker)[attr - 1] == 0)
    {
      continue;
    }
    return dynamic_cast<const BdrHyperbolicFluxIntegrator*>(ext_bdr_int_[i].get());
  }
  return nullptr;
}

bool DGHyperbolicConservationLaws::BuildBoundaryPolicyData_()
{
  face_bdr_policy_ready_ = false;
  face_bdr_max_attr_ = 0;
  face_bdr_type_by_attr_.SetSize(0);
  face_bdr_gamma_by_attr_.SetSize(0);
  face_bdr_pressure_by_attr_.SetSize(0);
  face_bdr_state_by_attr_.SetSize(0);

  Mesh* mesh = vfes.GetMesh();
  if (!mesh || face_bdr_attr_.Size() == 0)
  {
    face_bdr_policy_ready_ = true;
    return true;
  }
  if (dim > 2)
  {
    return false;
  }

  const FluxFunction& physical_flux = formIntegrator->GetFluxFunction();
  const auto* igr_flux = dynamic_cast<const FluxWithIGR*>(&physical_flux);
  const bool exact_igr_flux =
    igr_flux && igr_flux->GetInvariantDomainFlux();
  const real_t exact_gamma = exact_igr_flux ? igr_flux->GetGamma() : 0.0;
  const real_t exact_density_floor =
    exact_igr_flux ? igr_flux->GetVelocityRhoFloor() : 0.0;
  const real_t exact_pressure_floor =
    exact_igr_flux ? igr_flux->GetInvariantDomainPressureFloor() : 0.0;
  if (exact_igr_flux)
  {
    MFEM_VERIFY(std::isfinite(exact_gamma) && exact_gamma > 1.0 &&
                  exact_gamma == dgsem_gamma_,
                "Packed DG invariant-domain flux requires the operator and "
                "FluxWithIGR to use one finite gamma greater than one.");
    MFEM_VERIFY(std::isfinite(exact_density_floor) &&
                  exact_density_floor > 0.0 &&
                  std::isfinite(exact_pressure_floor) &&
                  exact_pressure_floor > 0.0,
                "Packed DG invariant-domain flux requires finite positive "
                "density and pressure floors.");
  }

  const auto verify_exact_gamma = [&](real_t boundary_gamma,
                                      const char* boundary_name)
  {
    MFEM_VERIFY(!exact_igr_flux || boundary_gamma == exact_gamma,
                "Packed DG invariant-domain " << boundary_name
                                               << " gamma differs from the "
                                                  "FluxWithIGR gamma.");
  };
  const auto verify_exact_floors = [&](real_t boundary_density_floor,
                                       real_t boundary_pressure_floor,
                                       const char* boundary_name)
  {
    MFEM_VERIFY(!exact_igr_flux ||
                  (boundary_density_floor == exact_density_floor &&
                   boundary_pressure_floor == exact_pressure_floor),
                "Packed DG invariant-domain " << boundary_name
                                               << " floors differ from the "
                                                  "FluxWithIGR admitted-state "
                                                  "contract.");
  };
  const auto verify_exact_state = [&](const Vector& state,
                                      const char* state_name)
  {
    if (!exact_igr_flux)
    {
      return;
    }
    bool finite = state.Size() == num_equations;
    for (int eq = 0; finite && eq < state.Size(); ++eq)
    {
      finite = std::isfinite(state(eq));
    }
    real_t pressure = -1.0;
    if (finite && state(0) >= exact_density_floor)
    {
      real_t momentum_squared = 0.0;
      for (int d = 0; d < dim; ++d)
      {
        momentum_squared += state(1 + d) * state(1 + d);
      }
      pressure = (exact_gamma - 1.0) *
                 (state(1 + dim) -
                  0.5 * momentum_squared / state(0));
    }
    MFEM_VERIFY(finite && state(0) >= exact_density_floor &&
                  std::isfinite(pressure) &&
                  pressure >= exact_pressure_floor,
                "Packed DG invariant-domain " << state_name
                                               << " violates the configured "
                                                  "density/pressure floors.");
  };

  face_bdr_max_attr_ = mesh->bdr_attributes.Max();
  if (face_bdr_max_attr_ <= 0)
  {
    face_bdr_policy_ready_ = true;
    return true;
  }

  face_bdr_type_by_attr_.SetSize(face_bdr_max_attr_ + 1);
  face_bdr_type_by_attr_ = static_cast<int>(DeviceBoundaryType::unsupported);
  face_bdr_gamma_by_attr_.SetSize(face_bdr_max_attr_ + 1);
  face_bdr_gamma_by_attr_ = 0.0;
  face_bdr_pressure_by_attr_.SetSize(face_bdr_max_attr_ + 1);
  face_bdr_pressure_by_attr_ = 0.0;
  face_bdr_state_by_attr_.SetSize((face_bdr_max_attr_ + 1) * num_equations);
  face_bdr_state_by_attr_ = 0.0;

  auto state_out = face_bdr_state_by_attr_.HostWrite();
  auto set_state = [&](int attr, const Vector& s)
  {
    MFEM_VERIFY(s.Size() == num_equations,
                "Boundary policy state has unexpected size.");
    const int off = attr * num_equations;
    for (int eq = 0; eq < num_equations; ++eq)
    {
      state_out[off + eq] = s(eq);
    }
  };

  for (int attr = 1; attr <= face_bdr_max_attr_; ++attr)
  {
    const auto* bdr = FindBdrIntegrator_(attr);
    if (!bdr)
    {
      continue;
    }
    const BoundaryState& bc = bdr->GetBoundaryState();

    if (auto* inlet = dynamic_cast<const SupersonicInletState*>(&bc))
    {
      verify_exact_state(inlet->GetInletState(),
                         "supersonic-inlet reference state");
      face_bdr_type_by_attr_[attr] =
        static_cast<int>(DeviceBoundaryType::supersonic_inlet);
      face_bdr_gamma_by_attr_(attr) = dgsem_gamma_;
      set_state(attr, inlet->GetInletState());
      continue;
    }
    if (dynamic_cast<const SupersonicOutletState*>(&bc))
    {
      face_bdr_type_by_attr_[attr] =
        static_cast<int>(DeviceBoundaryType::supersonic_outlet);
      face_bdr_gamma_by_attr_(attr) = dgsem_gamma_;
      continue;
    }
    if (auto* outlet_p = dynamic_cast<const SubsonicOutletPressureState*>(&bc))
    {
      verify_exact_gamma(outlet_p->GetGamma(),
                         "subsonic-outlet-pressure");
      verify_exact_floors(outlet_p->GetDensityFloor(),
                          outlet_p->GetPressureFloor(),
                          "subsonic-outlet-pressure");
      MFEM_VERIFY(!exact_igr_flux || outlet_p->GetDimension() == dim,
                  "Packed DG invariant-domain subsonic-outlet dimension "
                  "differs from the operator dimension.");
      MFEM_VERIFY(!exact_igr_flux ||
                    (std::isfinite(outlet_p->GetOutletPressure()) &&
                     outlet_p->GetOutletPressure() >= exact_pressure_floor),
                  "Packed DG invariant-domain outlet pressure violates the "
                  "configured pressure floor.");
      face_bdr_type_by_attr_[attr] =
        static_cast<int>(DeviceBoundaryType::subsonic_outlet_pressure);
      face_bdr_gamma_by_attr_(attr) = outlet_p->GetGamma();
      face_bdr_pressure_by_attr_(attr) = outlet_p->GetOutletPressure();
      continue;
    }
    if (auto* outlet_nr = dynamic_cast<const NonReflectingOutletState*>(&bc))
    {
      verify_exact_gamma(outlet_nr->GetGamma(),
                         "nonreflecting-outlet");
      verify_exact_floors(outlet_nr->GetDensityFloor(),
                          outlet_nr->GetPressureFloor(),
                          "nonreflecting-outlet");
      MFEM_VERIFY(!exact_igr_flux || outlet_nr->GetDimension() == dim,
                  "Packed DG invariant-domain nonreflecting-outlet dimension "
                  "differs from the operator dimension.");
      verify_exact_state(outlet_nr->GetReferenceState(),
                         "nonreflecting-outlet reference state");
      face_bdr_type_by_attr_[attr] =
        static_cast<int>(DeviceBoundaryType::nonreflecting_outlet);
      face_bdr_gamma_by_attr_(attr) = outlet_nr->GetGamma();
      set_state(attr, outlet_nr->GetReferenceState());
      continue;
    }
    if (auto* inlet_sub = dynamic_cast<const SubsonicInletState*>(&bc))
    {
      verify_exact_gamma(inlet_sub->GetGamma(), "subsonic-inlet");
      verify_exact_floors(inlet_sub->GetDensityFloor(),
                          inlet_sub->GetPressureFloor(),
                          "subsonic-inlet");
      MFEM_VERIFY(!exact_igr_flux || inlet_sub->GetDimension() == dim,
                  "Packed DG invariant-domain subsonic-inlet dimension "
                  "differs from the operator dimension.");
      verify_exact_state(inlet_sub->GetInletState(),
                         "subsonic-inlet reference state");
      face_bdr_type_by_attr_[attr] =
        static_cast<int>(DeviceBoundaryType::subsonic_inlet);
      face_bdr_gamma_by_attr_(attr) = inlet_sub->GetGamma();
      set_state(attr, inlet_sub->GetInletState());
      continue;
    }
    if (auto* far = dynamic_cast<const FarFieldState*>(&bc))
    {
      verify_exact_gamma(far->GetGamma(), "far-field");
      verify_exact_floors(far->GetDensityFloor(),
                          far->GetPressureFloor(),
                          "far-field");
      MFEM_VERIFY(!exact_igr_flux || far->GetDimension() == dim,
                  "Packed DG invariant-domain far-field dimension differs "
                  "from the operator dimension.");
      verify_exact_state(far->GetFarFieldState(),
                         "far-field reference state");
      face_bdr_type_by_attr_[attr] =
        static_cast<int>(DeviceBoundaryType::far_field);
      face_bdr_gamma_by_attr_(attr) = far->GetGamma();
      set_state(attr, far->GetFarFieldState());
      continue;
    }
    if (auto* slip = dynamic_cast<const SlipWallState*>(&bc))
    {
      verify_exact_floors(slip->GetDensityFloor(),
                          slip->GetPressureFloor(),
                          "slip-wall");
      MFEM_VERIFY(!exact_igr_flux || slip->GetDimension() == dim,
                  "Packed DG invariant-domain slip-wall dimension differs "
                  "from the operator dimension.");
      face_bdr_type_by_attr_[attr] =
        static_cast<int>(DeviceBoundaryType::slip_wall);
      face_bdr_gamma_by_attr_(attr) = dgsem_gamma_;
      continue;
    }
    return false;
  }

  auto bdr_attr = face_bdr_attr_.HostRead();
  auto bdr_type = face_bdr_type_by_attr_.HostRead();
  for (int i = 0; i < face_bdr_attr_.Size(); ++i)
  {
    const int attr = bdr_attr[i];
    if (attr <= 0 || attr > face_bdr_max_attr_ ||
        bdr_type[attr] == static_cast<int>(DeviceBoundaryType::unsupported))
    {
      return false;
    }
  }

  face_bdr_policy_ready_ = true;
  return true;
}

void DGHyperbolicConservationLaws::BuildFaceApplyData_()
{
  face_apply_ready_ = false;
  face_data_dirty_ = false;
  face_elem_dof_ = 0;
  face_int_nq_ = 0;
  face_bdr_nq_ = 0;
  face_shared_nq_ = 0;
  face_int_e1_.SetSize(0);
  face_int_e2_.SetSize(0);
  face_int_mesh_id_.SetSize(0);
  face_int_shape1_.SetSize(0);
  face_int_shape2_.SetSize(0);
  face_int_normals_.SetSize(0);
  face_int_weights_.SetSize(0);
  face_int_ip1_ref_.SetSize(0);
  face_int_ip2_ref_.SetSize(0);
  face_int_speed_.SetSize(0);
  face_int_speed_q_.SetSize(0);
  face_int_q_sigma_telemetry_.SetSize(0);
  face_bdr_e1_.SetSize(0);
  face_bdr_attr_.SetSize(0);
  face_bdr_mesh_id_.SetSize(0);
  face_bdr_shape1_.SetSize(0);
  face_bdr_normals_.SetSize(0);
  face_bdr_weights_.SetSize(0);
  face_bdr_ip1_ref_.SetSize(0);
  face_bdr_speed_.SetSize(0);
  face_bdr_speed_q_.SetSize(0);
  face_bdr_q_sigma_telemetry_.SetSize(0);
  face_shared_e1_.SetSize(0);
  face_shared_nbr_vdof_map_.SetSize(0);
  face_shared_shape1_.SetSize(0);
  face_shared_shape2_.SetSize(0);
  face_shared_normals_.SetSize(0);
  face_shared_weights_.SetSize(0);
  face_shared_ip1_ref_.SetSize(0);
  face_shared_ip2_ref_.SetSize(0);
  face_shared_speed_.SetSize(0);
  face_shared_speed_q_.SetSize(0);
  face_shared_q_sigma_telemetry_.SetSize(0);
  face_int_flux_record_.SetSize(0);
  face_bdr_flux_record_.SetSize(0);
  face_shared_flux_record_.SetSize(0);
  fused_elem_face_offsets_.SetSize(0);
  fused_elem_face_kind_.SetSize(0);
  fused_elem_face_id_.SetSize(0);
  fused_elem_face_side_.SetSize(0);
  fused_elem_incidence_ready_ = false;
  fused_wide_panel_layout_ready_ = false;
  face_bdr_policy_ready_ = false;
  face_sigma_ready_ = false;
  face_sigma_fes_ = nullptr;
  face_sigma_fes_sequence_ = -1;
  face_sigma_ldof_ = 0;
  face_sigma_elem_dof_map_.SetSize(0);
  face_int_sigma_shape1_.SetSize(0);
  face_int_sigma_shape2_.SetSize(0);
  face_bdr_sigma_shape1_.SetSize(0);
  face_shared_sigma_shape1_.SetSize(0);
  face_shared_sigma_shape2_.SetSize(0);
  face_shared_sigma_nbr_dof_map_.SetSize(0);

  if (!invmass_apply_ready_ || weakdiv.empty() || !num_flux_)
  {
    return;
  }
  if (dim > 2)
  {
    return;
  }

  const FluxFunction& flux_fn = formIntegrator->GetFluxFunction();
  const auto* flux_igr = dynamic_cast<const FluxWithIGR*>(&flux_fn);
  const bool euler_base =
    (dynamic_cast<const EulerFlux*>(&flux_fn) != nullptr) ||
    (flux_igr && flux_igr->HasEulerBase());
  if (!euler_base)
  {
    return;
  }

  bool flux_supported =
    dynamic_cast<const RusanovFluxWithSides*>(num_flux_) != nullptr ||
    dynamic_cast<const HdgTraceCenteredFlux*>(num_flux_) != nullptr;
  if (!flux_supported)
  {
    const auto* dgsem_flux = dynamic_cast<const DGSEMSurfaceFlux*>(num_flux_);
    flux_supported = dgsem_flux &&
      (dgsem_flux->GetSurfaceFluxType() == DGSEMSurfaceFluxType::Rusanov ||
       dgsem_flux->GetSurfaceFluxType() ==
         DGSEMSurfaceFluxType::HdgTraceCentered);
  }
  if (!flux_supported)
  {
    return;
  }

  Mesh* mesh = vfes.GetMesh();
  if (!mesh)
  {
    return;
  }

  const int dof = invmass_elem_dof_;
  if (dof <= 0)
  {
    return;
  }
  face_elem_dof_ = dof;
  const int ldof = dof * num_equations;

#ifdef MFEM_USE_MPI
  auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&vfes);
  ParMesh* pmesh = pfes ? pfes->GetParMesh() : nullptr;
#else
  auto* pfes = static_cast<ParFiniteElementSpace*>(nullptr);
  ParMesh* pmesh = nullptr;
#endif

  std::vector<int> int_e1;
  std::vector<int> int_e2;
  std::vector<int> int_mesh_id;
  std::vector<real_t> int_shape1;
  std::vector<real_t> int_shape2;
  std::vector<real_t> int_normals;
  std::vector<real_t> int_weights;
  std::vector<real_t> int_ip1_ref;
  std::vector<real_t> int_ip2_ref;

  std::vector<int> bdr_e1;
  std::vector<int> bdr_attr;
  std::vector<int> bdr_mesh_id;
  std::vector<real_t> bdr_shape1;
  std::vector<real_t> bdr_normals;
  std::vector<real_t> bdr_weights;
  std::vector<real_t> bdr_ip1_ref;

  std::vector<int> shared_e1;
  std::vector<int> shared_nbr_vdof_map;
  std::vector<real_t> shared_shape1;
  std::vector<real_t> shared_shape2;
  std::vector<real_t> shared_normals;
  std::vector<real_t> shared_weights;
  std::vector<real_t> shared_ip1_ref;
  std::vector<real_t> shared_ip2_ref;

  int face_int_nq = -1;
  int face_bdr_nq = -1;
  int face_shared_nq = -1;
  const IntegrationRule* fixed_ir = formIntegrator->GetIntegrationRule();

  std::vector<char> is_shared_face;
#ifdef MFEM_USE_MPI
  if (pmesh && pmesh->GetNSharedFaces() > 0)
  {
    is_shared_face.assign(mesh->GetNumFaces(), 0);
    for (int sf = 0; sf < pmesh->GetNSharedFaces(); ++sf)
    {
      const int f = pmesh->GetSharedFace(sf);
      if (f >= 0 && f < static_cast<int>(is_shared_face.size()))
      {
        is_shared_face[f] = 1;
      }
    }
  }
#endif

  Vector shape1(dof), shape2(dof), normal(dim);
  for (int f = 0; f < mesh->GetNumFaces(); ++f)
  {
    if (!is_shared_face.empty() && is_shared_face[f])
    {
      continue;
    }
    FaceElementTransformations* Tr = mesh->GetFaceElementTransformations(f);
    if (!Tr || Tr->Elem2No < 0)
    {
      continue;
    }
    const int e1 = Tr->Elem1No;
    const int e2 = Tr->Elem2No;
    const FiniteElement& fe1 = *vfes.GetFE(e1);
    const FiniteElement& fe2 = *vfes.GetFE(e2);
    if (fe1.GetDof() != dof || fe2.GetDof() != dof)
    {
      return;
    }

    const IntegrationRule* ir = fixed_ir;
    if (!ir)
    {
      const int order =
        2 * std::max(fe1.GetOrder(), fe2.GetOrder()) + face_int_order_offset_;
      ir = &IntRules.Get(Tr->GetGeometryType(), order);
    }
    if (!ir)
    {
      return;
    }
    if (face_int_nq < 0)
    {
      face_int_nq = ir->GetNPoints();
    }
    else if (face_int_nq != ir->GetNPoints())
    {
      return;
    }

    int_e1.push_back(e1);
    int_e2.push_back(e2);
    int_mesh_id.push_back(f);
    for (int q = 0; q < ir->GetNPoints(); ++q)
    {
      const IntegrationPoint& ip = ir->IntPoint(q);
      Tr->SetAllIntPoints(&ip);
      fe1.CalcShape(Tr->GetElement1IntPoint(), shape1);
      fe2.CalcShape(Tr->GetElement2IntPoint(), shape2);

      if (dim == 1)
      {
        normal(0) = (Tr->GetElement1IntPoint().x - 0.5) * 2.0;
      }
      else
      {
        CalcOrtho(Tr->Jacobian(), normal);
      }

      int_weights.push_back(ip.weight);
      for (int i = 0; i < dof; ++i)
      {
        int_shape1.push_back(shape1(i));
        int_shape2.push_back(shape2(i));
      }
      for (int d = 0; d < dim; ++d)
      {
        int_normals.push_back(normal(d));
      }

      const IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
      int_ip1_ref.push_back(ip1.x);
      if (dim > 1)
      {
        int_ip1_ref.push_back(ip1.y);
      }

      const IntegrationPoint& ip2 = Tr->GetElement2IntPoint();
      int_ip2_ref.push_back(ip2.x);
      if (dim > 1)
      {
        int_ip2_ref.push_back(ip2.y);
      }
    }
  }

#ifdef MFEM_USE_MPI
  if (pmesh && pmesh->GetNSharedFaces() > 0)
  {
    if (vfes.GetOrdering() != Ordering::byNODES)
    {
      return;
    }
    pfes->ExchangeFaceNbrData();
    const int face_nbr_vsize = pfes->GetFaceNbrVSize();

    Array<int> nbr_vdofs;
    for (int sf = 0; sf < pmesh->GetNSharedFaces(); ++sf)
    {
      FaceElementTransformations* Tr =
        pmesh->GetSharedFaceTransformations(sf, /*use_local_face_nr=*/true);
      if (!Tr)
      {
        continue;
      }

      const int e1 = Tr->Elem1No;
      const int e2 = Tr->Elem2No - pmesh->GetNE();
      if (e2 < 0)
      {
        return;
      }

      const FiniteElement& fe1 = *vfes.GetFE(e1);
      const FiniteElement* fe2_ptr = pfes->GetFaceNbrFE(e2);
      if (!fe2_ptr)
      {
        return;
      }
      const FiniteElement& fe2 = *fe2_ptr;
      if (fe1.GetDof() != dof || fe2.GetDof() != dof)
      {
        return;
      }

      const IntegrationRule* ir = fixed_ir;
      if (!ir)
      {
        const int order =
          2 * std::max(fe1.GetOrder(), fe2.GetOrder()) + face_int_order_offset_;
        ir = &IntRules.Get(Tr->GetGeometryType(), order);
      }
      if (!ir)
      {
        return;
      }
      if (face_shared_nq < 0)
      {
        face_shared_nq = ir->GetNPoints();
      }
      else if (face_shared_nq != ir->GetNPoints())
      {
        return;
      }

      if (pfes->GetFaceNbrElementVDofs(e2, nbr_vdofs) != nullptr)
      {
        return;
      }
      if (nbr_vdofs.Size() != ldof)
      {
        return;
      }

      shared_e1.push_back(e1);
      // Reorder neighbor vdofs into node-major (i,eq) layout expected by kernels.
      // pfes->GetFaceNbrElementVDofs returns byNODES ordering (component blocks).
      for (int i = 0; i < dof; ++i)
      {
        for (int eq = 0; eq < num_equations; ++eq)
        {
          const int idx = eq * dof + i; // byNODES: component-major
          const int vd = nbr_vdofs[idx];
          int vd_abs = vd;
          if (vd_abs < 0)
          {
            vd_abs = -1 - vd_abs;
          }
          if (vd_abs < 0 || vd_abs >= face_nbr_vsize)
          {
            return;
          }
          shared_nbr_vdof_map.push_back(vd);
        }
      }

      for (int q = 0; q < ir->GetNPoints(); ++q)
      {
        const IntegrationPoint& ip = ir->IntPoint(q);
        Tr->SetAllIntPoints(&ip);
        fe1.CalcShape(Tr->GetElement1IntPoint(), shape1);
        fe2.CalcShape(Tr->GetElement2IntPoint(), shape2);

        if (dim == 1)
        {
          normal(0) = (Tr->GetElement1IntPoint().x - 0.5) * 2.0;
        }
        else
        {
          CalcOrtho(Tr->Jacobian(), normal);
        }

        shared_weights.push_back(ip.weight);
        for (int i = 0; i < dof; ++i)
        {
          shared_shape1.push_back(shape1(i));
          shared_shape2.push_back(shape2(i));
        }
        for (int d = 0; d < dim; ++d)
        {
          shared_normals.push_back(normal(d));
        }

        const IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
        shared_ip1_ref.push_back(ip1.x);
        if (dim > 1)
        {
          shared_ip1_ref.push_back(ip1.y);
        }

        const IntegrationPoint& ip2 = Tr->GetElement2IntPoint();
        shared_ip2_ref.push_back(ip2.x);
        if (dim > 1)
        {
          shared_ip2_ref.push_back(ip2.y);
        }
      }
    }
  }
#endif

  for (int b = 0; b < mesh->GetNBE(); ++b)
  {
    FaceElementTransformations* Tr = mesh->GetBdrFaceTransformations(b);
    if (!Tr)
    {
      continue;
    }
    const int e1 = Tr->Elem1No;
    const FiniteElement& fe1 = *vfes.GetFE(e1);
    if (fe1.GetDof() != dof)
    {
      return;
    }
    const int attr = Tr->Attribute;
    const auto* bdr_int = FindBdrIntegrator_(attr);
    if (!bdr_int)
    {
      return;
    }

    const IntegrationRule* ir = bdr_int->GetIntegrationRule();
    if (!ir)
    {
      const int order = 2 * fe1.GetOrder() + face_int_order_offset_;
      ir = &IntRules.Get(Tr->GetGeometryType(), order);
    }
    if (!ir)
    {
      return;
    }
    if (face_bdr_nq < 0)
    {
      face_bdr_nq = ir->GetNPoints();
    }
    else if (face_bdr_nq != ir->GetNPoints())
    {
      return;
    }

    bdr_e1.push_back(e1);
    bdr_attr.push_back(attr);
    bdr_mesh_id.push_back(b);
    for (int q = 0; q < ir->GetNPoints(); ++q)
    {
      const IntegrationPoint& ip = ir->IntPoint(q);
      Tr->SetAllIntPoints(&ip);
      fe1.CalcShape(Tr->GetElement1IntPoint(), shape1);

      if (dim == 1)
      {
        normal(0) = (Tr->GetElement1IntPoint().x - 0.5) * 2.0;
      }
      else
      {
        CalcOrtho(Tr->Jacobian(), normal);
      }

      bdr_weights.push_back(ip.weight);
      for (int i = 0; i < dof; ++i)
      {
        bdr_shape1.push_back(shape1(i));
      }
      for (int d = 0; d < dim; ++d)
      {
        bdr_normals.push_back(normal(d));
      }

      const IntegrationPoint& ip1 = Tr->GetElement1IntPoint();
      bdr_ip1_ref.push_back(ip1.x);
      if (dim > 1)
      {
        bdr_ip1_ref.push_back(ip1.y);
      }
    }
  }

  face_int_nq_ = std::max(face_int_nq, 0);
  face_bdr_nq_ = std::max(face_bdr_nq, 0);
  face_shared_nq_ = std::max(face_shared_nq, 0);

  face_int_e1_.SetSize(static_cast<int>(int_e1.size()));
  face_int_e2_.SetSize(static_cast<int>(int_e2.size()));
  face_int_mesh_id_.SetSize(static_cast<int>(int_mesh_id.size()));
  for (int i = 0; i < face_int_e1_.Size(); ++i)
  {
    face_int_e1_[i] = int_e1[i];
    face_int_e2_[i] = int_e2[i];
    face_int_mesh_id_[i] = int_mesh_id[i];
  }
  face_int_shape1_.SetSize(static_cast<int>(int_shape1.size()));
  face_int_shape2_.SetSize(static_cast<int>(int_shape2.size()));
  face_int_normals_.SetSize(static_cast<int>(int_normals.size()));
  face_int_weights_.SetSize(static_cast<int>(int_weights.size()));
  face_int_ip1_ref_.SetSize(static_cast<int>(int_ip1_ref.size()));
  face_int_ip2_ref_.SetSize(static_cast<int>(int_ip2_ref.size()));
  std::copy(int_shape1.begin(), int_shape1.end(), face_int_shape1_.HostWrite());
  std::copy(int_shape2.begin(), int_shape2.end(), face_int_shape2_.HostWrite());
  std::copy(int_normals.begin(),
            int_normals.end(),
            face_int_normals_.HostWrite());
  std::copy(int_weights.begin(),
            int_weights.end(),
            face_int_weights_.HostWrite());
  std::copy(int_ip1_ref.begin(),
            int_ip1_ref.end(),
            face_int_ip1_ref_.HostWrite());
  std::copy(int_ip2_ref.begin(),
            int_ip2_ref.end(),
            face_int_ip2_ref_.HostWrite());
  face_int_speed_.SetSize(face_int_e1_.Size());
  face_int_speed_ = 0.0;
  face_int_speed_q_.SetSize(face_int_e1_.Size() * std::max(face_int_nq_, 0));
  face_int_speed_q_ = 0.0;
  if (flux_igr && flux_igr->GetSigmaField() &&
      flux_igr->GetInvariantDomainFlux())
  {
    dg_igr_detail::InitializeFrozenSigmaTelemetry(
      face_int_q_sigma_telemetry_, face_int_e1_.Size() * face_int_nq_);
  }

  face_bdr_e1_.SetSize(static_cast<int>(bdr_e1.size()));
  face_bdr_attr_.SetSize(static_cast<int>(bdr_attr.size()));
  face_bdr_mesh_id_.SetSize(static_cast<int>(bdr_mesh_id.size()));
  for (int i = 0; i < face_bdr_e1_.Size(); ++i)
  {
    face_bdr_e1_[i] = bdr_e1[i];
    face_bdr_attr_[i] = bdr_attr[i];
    face_bdr_mesh_id_[i] = bdr_mesh_id[i];
  }
  face_bdr_shape1_.SetSize(static_cast<int>(bdr_shape1.size()));
  face_bdr_normals_.SetSize(static_cast<int>(bdr_normals.size()));
  face_bdr_weights_.SetSize(static_cast<int>(bdr_weights.size()));
  face_bdr_ip1_ref_.SetSize(static_cast<int>(bdr_ip1_ref.size()));
  std::copy(bdr_shape1.begin(), bdr_shape1.end(), face_bdr_shape1_.HostWrite());
  std::copy(bdr_normals.begin(),
            bdr_normals.end(),
            face_bdr_normals_.HostWrite());
  std::copy(bdr_weights.begin(),
            bdr_weights.end(),
            face_bdr_weights_.HostWrite());
  std::copy(bdr_ip1_ref.begin(),
            bdr_ip1_ref.end(),
            face_bdr_ip1_ref_.HostWrite());
  face_bdr_speed_.SetSize(face_bdr_e1_.Size());
  face_bdr_speed_ = 0.0;
  face_bdr_speed_q_.SetSize(face_bdr_e1_.Size() * std::max(face_bdr_nq_, 0));
  face_bdr_speed_q_ = 0.0;
  if (flux_igr && flux_igr->GetSigmaField() &&
      flux_igr->GetInvariantDomainFlux())
  {
    dg_igr_detail::InitializeFrozenSigmaTelemetry(
      face_bdr_q_sigma_telemetry_, face_bdr_e1_.Size() * face_bdr_nq_);
  }

  face_shared_e1_.SetSize(static_cast<int>(shared_e1.size()));
  face_shared_nbr_vdof_map_.SetSize(static_cast<int>(shared_nbr_vdof_map.size()));
  for (int i = 0; i < face_shared_e1_.Size(); ++i)
  {
    face_shared_e1_[i] = shared_e1[i];
  }
  for (int i = 0; i < face_shared_nbr_vdof_map_.Size(); ++i)
  {
    face_shared_nbr_vdof_map_[i] = shared_nbr_vdof_map[i];
  }
  face_shared_shape1_.SetSize(static_cast<int>(shared_shape1.size()));
  face_shared_shape2_.SetSize(static_cast<int>(shared_shape2.size()));
  face_shared_normals_.SetSize(static_cast<int>(shared_normals.size()));
  face_shared_weights_.SetSize(static_cast<int>(shared_weights.size()));
  face_shared_ip1_ref_.SetSize(static_cast<int>(shared_ip1_ref.size()));
  face_shared_ip2_ref_.SetSize(static_cast<int>(shared_ip2_ref.size()));
  std::copy(shared_shape1.begin(),
            shared_shape1.end(),
            face_shared_shape1_.HostWrite());
  std::copy(shared_shape2.begin(),
            shared_shape2.end(),
            face_shared_shape2_.HostWrite());
  std::copy(shared_normals.begin(),
            shared_normals.end(),
            face_shared_normals_.HostWrite());
  std::copy(shared_weights.begin(),
            shared_weights.end(),
            face_shared_weights_.HostWrite());
  std::copy(shared_ip1_ref.begin(),
            shared_ip1_ref.end(),
            face_shared_ip1_ref_.HostWrite());
  std::copy(shared_ip2_ref.begin(),
            shared_ip2_ref.end(),
            face_shared_ip2_ref_.HostWrite());
  face_shared_speed_.SetSize(face_shared_e1_.Size());
  face_shared_speed_ = 0.0;
  face_shared_speed_q_.SetSize(face_shared_e1_.Size() * std::max(face_shared_nq_, 0));
  face_shared_speed_q_ = 0.0;
  if (flux_igr && flux_igr->GetSigmaField() &&
      flux_igr->GetInvariantDomainFlux())
  {
    dg_igr_detail::InitializeFrozenSigmaTelemetry(
      face_shared_q_sigma_telemetry_,
      face_shared_e1_.Size() * face_shared_nq_);
  }

  if (!BuildBoundaryPolicyData_())
  {
    return;
  }

  face_apply_ready_ = true;
  // Failure here only disables the opt-in research screen. The established
  // packed face path remains available for every tuple it supported before.
  BuildFusedElementIncidence_();
}

bool DGHyperbolicConservationLaws::BuildFusedElementIncidence_()
{
  fused_elem_incidence_ready_ = false;
  fused_elem_face_offsets_.SetSize(0);
  fused_elem_face_kind_.SetSize(0);
  fused_elem_face_id_.SetSize(0);
  fused_elem_face_side_.SetSize(0);
  face_int_flux_record_.SetSize(0);
  face_bdr_flux_record_.SetSize(0);
  face_shared_flux_record_.SetSize(0);
  fused_wide_panel_layout_ready_ = false;

  const int ne = vfes.GetNE();
  const int dof = face_elem_dof_;
  if (!face_apply_ready_ || ne <= 0 || dim != 2 || num_equations != 4 ||
      dof <= 0 || dof > 9)
  {
    return false;
  }
  for (int e = 0; e < ne; ++e)
  {
    const int order = vfes.GetFE(e)->GetOrder();
    if (order < 1 || order > 2 ||
        vfes.GetFE(e)->GetGeomType() != Geometry::SQUARE ||
        vfes.GetFE(e)->GetDof() != dof)
    {
      return false;
    }
  }

  // MFEM's byNODES vector ordering is component-major. For a uniform L2
  // space whose scalar element dofs are consecutive, the local L-vector is
  // already a column-major K x (elements*equations) panel. Record this as a
  // follow-on opportunity; the fused screen itself continues to use maps.
  fused_wide_panel_layout_ready_ =
    vfes.GetOrdering() == Ordering::byNODES && vfes.GetNDofs() == ne * dof;
  Array<int> element_dofs;
  for (int e = 0; fused_wide_panel_layout_ready_ && e < ne; ++e)
  {
    vfes.GetElementDofs(e, element_dofs);
    if (element_dofs.Size() != dof)
    {
      fused_wide_panel_layout_ready_ = false;
      break;
    }
    for (int i = 0; i < dof; ++i)
    {
      if (element_dofs[i] != e * dof + i)
      {
        fused_wide_panel_layout_ready_ = false;
        break;
      }
    }
  }

  std::vector<int> counts(ne, 0);
  for (int f = 0; f < face_int_e1_.Size(); ++f)
  {
    ++counts[face_int_e1_[f]];
    ++counts[face_int_e2_[f]];
  }
  for (int f = 0; f < face_bdr_e1_.Size(); ++f)
  {
    ++counts[face_bdr_e1_[f]];
  }
  for (int f = 0; f < face_shared_e1_.Size(); ++f)
  {
    ++counts[face_shared_e1_[f]];
  }
  for (int e = 0; e < ne; ++e)
  {
    // The screen is intentionally bounded to conforming quadrilateral meshes.
    // Refuse a partial incidence graph instead of silently dropping a face.
    if (counts[e] != 4)
    {
      return false;
    }
  }

  fused_elem_face_offsets_.SetSize(ne + 1);
  fused_elem_face_offsets_[0] = 0;
  for (int e = 0; e < ne; ++e)
  {
    fused_elem_face_offsets_[e + 1] =
      fused_elem_face_offsets_[e] + counts[e];
  }
  const int incidence_count = fused_elem_face_offsets_[ne];
  fused_elem_face_kind_.SetSize(incidence_count);
  fused_elem_face_id_.SetSize(incidence_count);
  fused_elem_face_side_.SetSize(incidence_count);

  std::vector<int> cursor(ne);
  for (int e = 0; e < ne; ++e)
  {
    cursor[e] = fused_elem_face_offsets_[e];
  }
  const auto append = [&](int element, int kind, int face, int side)
  {
    const int slot = cursor[element]++;
    fused_elem_face_kind_[slot] = kind;
    fused_elem_face_id_[slot] = face;
    fused_elem_face_side_[slot] = side;
  };
  for (int f = 0; f < face_int_e1_.Size(); ++f)
  {
    append(face_int_e1_[f], kFusedInteriorFace_, f, 0);
    append(face_int_e2_[f], kFusedInteriorFace_, f, 1);
  }
  for (int f = 0; f < face_bdr_e1_.Size(); ++f)
  {
    append(face_bdr_e1_[f], kFusedBoundaryFace_, f, 0);
  }
  for (int f = 0; f < face_shared_e1_.Size(); ++f)
  {
    append(face_shared_e1_[f], kFusedSharedFace_, f, 0);
  }

  face_int_flux_record_.SetSize(face_int_e1_.Size() * face_int_nq_ *
                                num_equations);
  face_bdr_flux_record_.SetSize(face_bdr_e1_.Size() * face_bdr_nq_ *
                                num_equations);
  face_shared_flux_record_.SetSize(face_shared_e1_.Size() * face_shared_nq_ *
                                   num_equations);
  fused_elem_incidence_ready_ = true;
  return true;
}

bool DGHyperbolicConservationLaws::BuildFaceSigmaData_(
  const ParGridFunction& sigma) const
{
  const FiniteElementSpace* sfes = sigma.ParFESpace();
  if (!sfes || !face_apply_ready_)
  {
    return false;
  }

  const long seq = sfes->GetSequence();
  if (face_sigma_ready_ && face_sigma_fes_ == sfes &&
      face_sigma_fes_sequence_ == seq)
  {
    return true;
  }

  face_sigma_ready_ = false;
  face_sigma_fes_ = nullptr;
  face_sigma_fes_sequence_ = -1;
  face_sigma_ldof_ = 0;
  face_sigma_elem_dof_map_.SetSize(0);
  face_int_sigma_shape1_.SetSize(0);
  face_int_sigma_shape2_.SetSize(0);
  face_bdr_sigma_shape1_.SetSize(0);

  const int ne = vfes.GetNE();
  if (sfes->GetNE() != ne)
  {
    return false;
  }

  const int sigma_ldof = sfes->GetFE(0)->GetDof();
  if (sigma_ldof <= 0)
  {
    return false;
  }
  for (int e = 1; e < ne; ++e)
  {
    if (sfes->GetFE(e)->GetDof() != sigma_ldof)
    {
      return false;
    }
  }

  face_sigma_elem_dof_map_.SetSize(ne * sigma_ldof);
  Array<int> sigma_sdofs;
  for (int e = 0; e < ne; ++e)
  {
    sfes->GetElementDofs(e, sigma_sdofs);
    if (sigma_sdofs.Size() != sigma_ldof)
    {
      return false;
    }
    const int off = e * sigma_ldof;
    for (int k = 0; k < sigma_ldof; ++k)
    {
      if (sigma_sdofs[k] < 0)
      {
        return false;
      }
      face_sigma_elem_dof_map_[off + k] = sigma_sdofs[k];
    }
  }

  face_int_sigma_shape1_.SetSize(face_int_e1_.Size() * face_int_nq_ * sigma_ldof);
  face_int_sigma_shape2_.SetSize(face_int_e2_.Size() * face_int_nq_ * sigma_ldof);
  face_bdr_sigma_shape1_.SetSize(face_bdr_e1_.Size() * face_bdr_nq_ * sigma_ldof);
  face_shared_sigma_shape1_.SetSize(face_shared_e1_.Size() * face_shared_nq_ *
                                    sigma_ldof);
  face_shared_sigma_shape2_.SetSize(face_shared_e1_.Size() * face_shared_nq_ *
                                    sigma_ldof);
  face_shared_sigma_nbr_dof_map_.SetSize(face_shared_e1_.Size() * sigma_ldof);
  auto int_shape1 = face_int_sigma_shape1_.HostWrite();
  auto int_shape2 = face_int_sigma_shape2_.HostWrite();
  auto bdr_shape1 = face_bdr_sigma_shape1_.HostWrite();
  auto shared_shape1 = face_shared_sigma_shape1_.HostWrite();
  auto shared_shape2 = face_shared_sigma_shape2_.HostWrite();
  Vector sigma_shape(sigma_ldof);

#ifdef MFEM_USE_MPI
  if (face_shared_e1_.Size() > 0)
  {
    const auto* psfes = dynamic_cast<const ParFiniteElementSpace*>(sfes);
    if (!psfes)
    {
      return false;
    }
    if (psfes->GetOrdering() != Ordering::byNODES)
    {
      return false;
    }
    const_cast<ParFiniteElementSpace*>(psfes)->ExchangeFaceNbrData();
    const int face_nbr_vsize = psfes->GetFaceNbrVSize();

    ParMesh* pmesh = const_cast<ParMesh*>(psfes->GetParMesh());
    if (!pmesh)
    {
      return false;
    }

    Array<int> sigma_nbr_vdofs;
    for (int sf = 0; sf < face_shared_e1_.Size(); ++sf)
    {
      FaceElementTransformations* Tr =
        pmesh->GetSharedFaceTransformations(sf, /*use_local_face_nr=*/true);
      if (!Tr)
      {
        return false;
      }

      const int e2 = Tr->Elem2No - pmesh->GetNE();
      if (e2 < 0)
      {
        return false;
      }

      const FiniteElement& sigma_fe_1 = *sfes->GetFE(face_shared_e1_[sf]);
      const FiniteElement* sigma_fe_2_ptr = psfes->GetFaceNbrFE(e2);
      if (!sigma_fe_2_ptr)
      {
        return false;
      }
      const FiniteElement& sigma_fe_2 = *sigma_fe_2_ptr;
      if (sigma_fe_2.GetDof() != sigma_ldof)
      {
        return false;
      }

      if (psfes->GetFaceNbrElementVDofs(e2, sigma_nbr_vdofs) != nullptr)
      {
        return false;
      }
      if (sigma_nbr_vdofs.Size() != sigma_ldof)
      {
        return false;
      }
      for (int k = 0; k < sigma_ldof; ++k)
      {
        const int d = sigma_nbr_vdofs[k];
        int d_abs = d;
        if (d_abs < 0)
        {
          d_abs = -1 - d_abs;
        }
        if (d_abs < 0 || d_abs >= face_nbr_vsize)
        {
          return false;
        }
        face_shared_sigma_nbr_dof_map_[sf * sigma_ldof + k] = d;
      }

      for (int q = 0; q < face_shared_nq_; ++q)
      {
        const int ip_off = (sf * face_shared_nq_ + q) * dim;
        const int s_off = (sf * face_shared_nq_ + q) * sigma_ldof;

        IntegrationPoint ip1;
        ip1.x = face_shared_ip1_ref_(ip_off);
        if (dim > 1)
        {
          ip1.y = face_shared_ip1_ref_(ip_off + 1);
        }
        sigma_fe_1.CalcShape(ip1, sigma_shape);
        for (int k = 0; k < sigma_ldof; ++k)
        {
          shared_shape1[s_off + k] = sigma_shape(k);
        }

        IntegrationPoint ip2;
        ip2.x = face_shared_ip2_ref_(ip_off);
        if (dim > 1)
        {
          ip2.y = face_shared_ip2_ref_(ip_off + 1);
        }
        sigma_fe_2.CalcShape(ip2, sigma_shape);
        for (int k = 0; k < sigma_ldof; ++k)
        {
          shared_shape2[s_off + k] = sigma_shape(k);
        }
      }
    }
  }
#endif

  for (int f = 0; f < face_int_e1_.Size(); ++f)
  {
    const FiniteElement& sigma_fe_1 = *sfes->GetFE(face_int_e1_[f]);
    const FiniteElement& sigma_fe_2 = *sfes->GetFE(face_int_e2_[f]);
    for (int q = 0; q < face_int_nq_; ++q)
    {
      IntegrationPoint ip1;
      const int ip_off = (f * face_int_nq_ + q) * dim;
      ip1.x = face_int_ip1_ref_(ip_off);
      if (dim > 1)
      {
        ip1.y = face_int_ip1_ref_(ip_off + 1);
      }
      sigma_fe_1.CalcShape(ip1, sigma_shape);
      const int s_off = (f * face_int_nq_ + q) * sigma_ldof;
      for (int k = 0; k < sigma_ldof; ++k)
      {
        int_shape1[s_off + k] = sigma_shape(k);
      }

      IntegrationPoint ip2;
      ip2.x = face_int_ip2_ref_(ip_off);
      if (dim > 1)
      {
        ip2.y = face_int_ip2_ref_(ip_off + 1);
      }
      sigma_fe_2.CalcShape(ip2, sigma_shape);
      for (int k = 0; k < sigma_ldof; ++k)
      {
        int_shape2[s_off + k] = sigma_shape(k);
      }
    }
  }

  for (int f = 0; f < face_bdr_e1_.Size(); ++f)
  {
    const FiniteElement& sigma_fe_1 = *sfes->GetFE(face_bdr_e1_[f]);
    for (int q = 0; q < face_bdr_nq_; ++q)
    {
      const int ip_off = (f * face_bdr_nq_ + q) * dim;
      IntegrationPoint ip1;
      ip1.x = face_bdr_ip1_ref_(ip_off);
      if (dim > 1)
      {
        ip1.y = face_bdr_ip1_ref_(ip_off + 1);
      }
      sigma_fe_1.CalcShape(ip1, sigma_shape);
      const int s_off = (f * face_bdr_nq_ + q) * sigma_ldof;
      for (int k = 0; k < sigma_ldof; ++k)
      {
        bdr_shape1[s_off + k] = sigma_shape(k);
      }
    }
  }

  face_sigma_ldof_ = sigma_ldof;
  face_sigma_fes_ = sfes;
  face_sigma_fes_sequence_ = seq;
  face_sigma_ready_ = true;
  return true;
}

real_t DGHyperbolicConservationLaws::ReduceFaceMaxSpeed_(
  const Vector& face_speed,
  Array<real_t>& workspace) const
{
  if (face_speed.Size() == 0)
  {
    return 0.0;
  }
  real_t max_speed = 0.0;
#ifdef DG_IGR_HAS_ACCEL_FORALL
  const bool use_dev = Device::Allows(Backend::DEVICE_MASK);
  MaxSpeedReductionBody body;
  body.speed = face_speed.Read();
  mfem::reduce(face_speed.Size(),
               max_speed,
               body,
               mfem::MaxReducer<real_t>(),
               use_dev,
               workspace);
#else
  auto speed = face_speed.HostRead();
  for (int i = 0; i < face_speed.Size(); ++i)
  {
    max_speed = std::max(max_speed, speed[i]);
  }
#endif
  return max_speed;
}

bool DGHyperbolicConservationLaws::UseDeviceFaceResidual_() const
{
  if (!Device::Allows(Backend::DEVICE_MASK))
  {
    return false;
  }
  if (vol_mode_ != VolumeMode::WeakDivergence || weakdiv.empty() ||
      !invmass_apply_ready_)
  {
    return false;
  }
  if (face_data_dirty_)
  {
    const_cast<DGHyperbolicConservationLaws*>(this)->BuildFaceApplyData_();
  }
  return face_apply_ready_;
}

void DGHyperbolicConservationLaws::LogFusedElementResidualSelection_(
  bool selected,
  const char* reason) const
{
  if (fused_residual_selection_logged_)
  {
    return;
  }
  if (!mfem::Mpi::Root())
  {
    fused_residual_selection_logged_ = true;
    return;
  }
  const std::int64_t weighted_flux_values =
    static_cast<std::int64_t>(face_int_e1_.Size()) * face_int_nq_ *
      num_equations +
    static_cast<std::int64_t>(face_bdr_e1_.Size()) * face_bdr_nq_ *
      num_equations +
    static_cast<std::int64_t>(face_shared_e1_.Size()) * face_shared_nq_ *
      num_equations;
  const std::int64_t eliminated_atomic_updates =
    (2LL * face_int_e1_.Size() * face_int_nq_ +
     1LL * face_bdr_e1_.Size() * face_bdr_nq_ +
     1LL * face_shared_e1_.Size() * face_shared_nq_) *
    face_elem_dof_ * num_equations;
  const int order = vfes.GetNE() > 0 ? vfes.GetFE(0)->GetOrder() : -1;
  std::cout << "[dg-fused-screen] requested=1"
            << " selected="
            << (selected ? "face-record-element-owned" : "legacy-packed")
            << " reason=" << reason
            << " order=" << order
            << " dof=" << face_elem_dof_
            << " elements=" << vfes.GetNE()
            << " incidences=" << fused_elem_face_kind_.Size()
            << " weighted_flux_values=" << weighted_flux_values
            << " eliminated_residual_atomics=" << eliminated_atomic_updates
            << " lift_order=deterministic-csr"
            << " parity_contract=paired-fp64-run"
            << " wide_panel_layout="
            << (fused_wide_panel_layout_ready_ ? "zero-copy" : "mapped")
            << " p3_tensor_core=disabled"
            << " backend="
            << (Device::Allows(Backend::CUDA_MASK) ? "cuda" : "non-cuda")
            << '\n';
  fused_residual_selection_logged_ = true;
}

bool DGHyperbolicConservationLaws::UseFusedElementResidualScreen_() const
{
  if (!::dg_igr::EnvFlagEnabledDefault(
        "DG_IGR_DG_FUSED_ELEMENT_RESIDUAL", false))
  {
    return false;
  }
  if (!Device::Allows(Backend::CUDA_MASK))
  {
    LogFusedElementResidualSelection_(false, "requires-cuda");
    return false;
  }
  if (vol_mode_ != VolumeMode::WeakDivergence || weakdiv.empty() ||
      !weakdiv_apply_ready_ || !invmass_apply_ready_)
  {
    LogFusedElementResidualSelection_(false, "requires-packed-weak-dg");
    return false;
  }
  if (face_data_dirty_)
  {
    const_cast<DGHyperbolicConservationLaws*>(this)->BuildFaceApplyData_();
  }
  if (!face_apply_ready_)
  {
    LogFusedElementResidualSelection_(false, "packed-face-unavailable");
    return false;
  }
  if (!fused_elem_incidence_ready_)
  {
    LogFusedElementResidualSelection_(false,
                                      "requires-conforming-p1-p2-quads");
    return false;
  }
  const int expected_int =
    face_int_e1_.Size() * face_int_nq_ * num_equations;
  const int expected_bdr =
    face_bdr_e1_.Size() * face_bdr_nq_ * num_equations;
  const int expected_shared =
    face_shared_e1_.Size() * face_shared_nq_ * num_equations;
  if (face_int_flux_record_.Size() != expected_int ||
      face_bdr_flux_record_.Size() != expected_bdr ||
      face_shared_flux_record_.Size() != expected_shared)
  {
    LogFusedElementResidualSelection_(false, "flux-record-size-mismatch");
    return false;
  }
  return true;
}

bool DGHyperbolicConservationLaws::AddDeviceFaceContributionsEulerRusanov_(
  const Vector& x_in,
  Vector& z_inout,
  real_t& max_face_speed,
  bool capture_boundary_flux,
  bool record_flux_only) const
{
#ifndef DG_IGR_HAS_ACCEL_FORALL
  (void)x_in;
  (void)z_inout;
  (void)capture_boundary_flux;
  (void)record_flux_only;
  max_face_speed = 0.0;
  return false;
#else
  if (!face_apply_ready_)
  {
    max_face_speed = 0.0;
    return false;
  }

  const FluxFunction& flux_fn = formIntegrator->GetFluxFunction();
  const auto* flux_igr = dynamic_cast<const FluxWithIGR*>(&flux_fn);
  const auto* dgsem_surface_flux =
    dynamic_cast<const DGSEMSurfaceFlux*>(num_flux_);
  const bool trace_centered =
    dynamic_cast<const HdgTraceCenteredFlux*>(num_flux_) != nullptr ||
    (dgsem_surface_flux &&
     dgsem_surface_flux->GetSurfaceFluxType() ==
       DGSEMSurfaceFluxType::HdgTraceCentered);

  bool use_sigma = false;
  bool invariant_domain_flux = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  const ParGridFunction* sigma_field = nullptr;
  real_t sigma_rho_floor = 1.0e-12;
  real_t invariant_domain_pressure_floor =
    dg_igr::LimiterDefaults::kPMin;
  if (flux_igr && flux_igr->GetSigmaField())
  {
    sigma_field = flux_igr->GetSigmaField();
    if (!BuildFaceSigmaData_(*sigma_field))
    {
      max_face_speed = 0.0;
      return false;
    }
    sigma_rho_floor = flux_igr->GetVelocityRhoFloor();
    invariant_domain_pressure_floor =
      flux_igr->GetInvariantDomainPressureFloor();
    invariant_domain_flux = flux_igr->GetInvariantDomainFlux();
    sigma_aware_cfl = flux_igr->GetSigmaAwareCFL();
    sigma_aware_rusanov = flux_igr->GetSigmaAwareRusanov();
    use_sigma = true;
  }
  MFEM_VERIFY(!invariant_domain_flux || flux_igr->GetGamma() == dgsem_gamma_,
              "Packed DG invariant-domain flux requires the operator, "
              "FluxWithIGR, and Euler base flux to use one gamma.");

  if (record_flux_only && !fused_elem_incidence_ready_)
  {
    max_face_speed = 0.0;
    return false;
  }
  if (!record_flux_only)
  {
    z_inout = 0.0;
  }

  const int dof = face_elem_dof_;
  const int neq = num_equations;
  const int ldof = dof * neq;
  const bool use_q_parallel = record_flux_only ||
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DG_FACE_Q_PARALLEL", true);

  const int* sigma_map = nullptr;
  const real_t* sigma_vals = nullptr;
  const real_t* int_sigma_shape1 = nullptr;
  const real_t* int_sigma_shape2 = nullptr;
  const real_t* bdr_sigma_shape1 = nullptr;
  const real_t* shared_sigma_shape1 = nullptr;
  const real_t* shared_sigma_shape2 = nullptr;
  const int* shared_sigma_nbr_dof_map = nullptr;
  const real_t* x_nbr_vals = nullptr;
  const real_t* sigma_nbr_vals = nullptr;
  real_t* boundary_flux_integral =
    capture_boundary_flux ? physical_boundary_flux_integral_.Write()
                          : nullptr;
  int sigma_ldof = 0;
  if (use_sigma)
  {
    sigma_map = face_sigma_elem_dof_map_.Read();
    sigma_vals = sigma_field->Read();
    int_sigma_shape1 = face_int_sigma_shape1_.Read();
    int_sigma_shape2 = face_int_sigma_shape2_.Read();
    bdr_sigma_shape1 = face_bdr_sigma_shape1_.Read();
    shared_sigma_shape1 = face_shared_sigma_shape1_.Read();
    shared_sigma_shape2 = face_shared_sigma_shape2_.Read();
    shared_sigma_nbr_dof_map = face_shared_sigma_nbr_dof_map_.Read();
    sigma_ldof = face_sigma_ldof_;
  }

  const int n_int = face_int_e1_.Size();
  MFEM_VERIFY(!invariant_domain_flux ||
                face_int_q_sigma_telemetry_.Size() ==
                  n_int * face_int_nq_ *
                    dg_igr_detail::kFrozenSigmaTelemetryWidth,
              "Packed invariant-domain interior telemetry is not initialized.");
  double* const int_q_sigma_telemetry =
    invariant_domain_flux ? face_int_q_sigma_telemetry_.ReadWrite() : nullptr;
  if (n_int > 0)
  {
    if (use_q_parallel)
    {
      PackedInteriorFaceEulerRusanovQKernel int_kernel;
      int_kernel.dof = dof;
      int_kernel.neq = neq;
      int_kernel.dim = dim;
      int_kernel.nq = face_int_nq_;
      int_kernel.ldof = ldof;
      int_kernel.sigma_ldof = sigma_ldof;
      int_kernel.gamma = dgsem_gamma_;
      int_kernel.sigma_rho_floor = sigma_rho_floor;
      int_kernel.invariant_domain_pressure_floor =
        invariant_domain_pressure_floor;
      int_kernel.use_sigma = use_sigma;
      int_kernel.invariant_domain_flux = invariant_domain_flux;
      int_kernel.sigma_aware_cfl = sigma_aware_cfl;
      int_kernel.sigma_aware_rusanov = sigma_aware_rusanov;
      int_kernel.trace_centered = trace_centered;
      int_kernel.state_map = invmass_elem_vdof_map_.Read();
      int_kernel.face_e1 = face_int_e1_.Read();
      int_kernel.face_e2 = face_int_e2_.Read();
      int_kernel.shape1 = face_int_shape1_.Read();
      int_kernel.shape2 = face_int_shape2_.Read();
      int_kernel.normals = face_int_normals_.Read();
      int_kernel.weights = face_int_weights_.Read();
      int_kernel.x = x_in.Read();
      int_kernel.z = record_flux_only ? nullptr : z_inout.ReadWrite();
      int_kernel.flux_record =
        record_flux_only ? face_int_flux_record_.Write() : nullptr;
      int_kernel.face_speed_q = face_int_speed_q_.Write();
      int_kernel.q_sigma_telemetry = int_q_sigma_telemetry;
      int_kernel.sigma_map = sigma_map;
      int_kernel.sigma_vals = sigma_vals;
      int_kernel.sigma_shape1 = int_sigma_shape1;
      int_kernel.sigma_shape2 = int_sigma_shape2;
      mfem::forall(n_int * face_int_nq_, int_kernel);

      FaceSpeedQReduceKernel int_speed_reduce;
      int_speed_reduce.nq = face_int_nq_;
      int_speed_reduce.face_speed_q = face_int_speed_q_.Read();
      int_speed_reduce.face_speed = face_int_speed_.Write();
      mfem::forall(n_int, int_speed_reduce);
    }
    else
    {
      PackedInteriorFaceEulerRusanovKernel int_kernel;
      int_kernel.dof = dof;
      int_kernel.neq = neq;
      int_kernel.dim = dim;
      int_kernel.nq = face_int_nq_;
      int_kernel.ldof = ldof;
      int_kernel.sigma_ldof = sigma_ldof;
      int_kernel.gamma = dgsem_gamma_;
      int_kernel.sigma_rho_floor = sigma_rho_floor;
      int_kernel.invariant_domain_pressure_floor =
        invariant_domain_pressure_floor;
      int_kernel.use_sigma = use_sigma;
      int_kernel.invariant_domain_flux = invariant_domain_flux;
      int_kernel.sigma_aware_cfl = sigma_aware_cfl;
      int_kernel.sigma_aware_rusanov = sigma_aware_rusanov;
      int_kernel.trace_centered = trace_centered;
      int_kernel.state_map = invmass_elem_vdof_map_.Read();
      int_kernel.face_e1 = face_int_e1_.Read();
      int_kernel.face_e2 = face_int_e2_.Read();
      int_kernel.shape1 = face_int_shape1_.Read();
      int_kernel.shape2 = face_int_shape2_.Read();
      int_kernel.normals = face_int_normals_.Read();
      int_kernel.weights = face_int_weights_.Read();
      int_kernel.x = x_in.Read();
      int_kernel.z = z_inout.ReadWrite();
      int_kernel.face_speed = face_int_speed_.Write();
      int_kernel.q_sigma_telemetry = int_q_sigma_telemetry;
      int_kernel.sigma_map = sigma_map;
      int_kernel.sigma_vals = sigma_vals;
      int_kernel.sigma_shape1 = int_sigma_shape1;
      int_kernel.sigma_shape2 = int_sigma_shape2;
      mfem::forall(n_int, int_kernel);
    }
  }

  const int n_bdr = face_bdr_e1_.Size();
  MFEM_VERIFY(!invariant_domain_flux ||
                face_bdr_q_sigma_telemetry_.Size() ==
                  n_bdr * face_bdr_nq_ *
                    dg_igr_detail::kFrozenSigmaTelemetryWidth,
              "Packed invariant-domain boundary telemetry is not initialized.");
  double* const bdr_q_sigma_telemetry =
    invariant_domain_flux ? face_bdr_q_sigma_telemetry_.ReadWrite() : nullptr;
  if (n_bdr > 0)
  {
    if (!face_bdr_policy_ready_)
    {
      max_face_speed = 0.0;
      return false;
    }

    if (use_q_parallel)
    {
      PackedBoundaryFaceEulerRusanovQKernel bdr_kernel;
      bdr_kernel.dof = dof;
      bdr_kernel.neq = neq;
      bdr_kernel.dim = dim;
      bdr_kernel.nq = face_bdr_nq_;
      bdr_kernel.ldof = ldof;
      bdr_kernel.sigma_ldof = sigma_ldof;
      bdr_kernel.gamma_default = dgsem_gamma_;
      bdr_kernel.sigma_rho_floor = sigma_rho_floor;
      bdr_kernel.invariant_domain_pressure_floor =
        invariant_domain_pressure_floor;
      bdr_kernel.use_sigma = use_sigma;
      bdr_kernel.invariant_domain_flux = invariant_domain_flux;
      bdr_kernel.sigma_aware_cfl = sigma_aware_cfl;
      bdr_kernel.sigma_aware_rusanov = sigma_aware_rusanov;
      bdr_kernel.trace_centered = trace_centered;
      bdr_kernel.bdr_max_attr = face_bdr_max_attr_;
      bdr_kernel.state_map = invmass_elem_vdof_map_.Read();
      bdr_kernel.face_e1 = face_bdr_e1_.Read();
      bdr_kernel.face_attr = face_bdr_attr_.Read();
      bdr_kernel.shape1 = face_bdr_shape1_.Read();
      bdr_kernel.normals = face_bdr_normals_.Read();
      bdr_kernel.weights = face_bdr_weights_.Read();
      bdr_kernel.x = x_in.Read();
      bdr_kernel.z = record_flux_only ? nullptr : z_inout.ReadWrite();
      bdr_kernel.flux_record =
        record_flux_only ? face_bdr_flux_record_.Write() : nullptr;
      bdr_kernel.face_speed_q = face_bdr_speed_q_.Write();
      bdr_kernel.q_sigma_telemetry = bdr_q_sigma_telemetry;
      bdr_kernel.boundary_flux_integral = boundary_flux_integral;
      bdr_kernel.bdr_type_by_attr = face_bdr_type_by_attr_.Read();
      bdr_kernel.bdr_gamma_by_attr = face_bdr_gamma_by_attr_.Read();
      bdr_kernel.bdr_pressure_by_attr = face_bdr_pressure_by_attr_.Read();
      bdr_kernel.bdr_state_by_attr = face_bdr_state_by_attr_.Read();
      bdr_kernel.sigma_map = sigma_map;
      bdr_kernel.sigma_vals = sigma_vals;
      bdr_kernel.sigma_shape1 = bdr_sigma_shape1;
      mfem::forall(n_bdr * face_bdr_nq_, bdr_kernel);

      FaceSpeedQReduceKernel bdr_speed_reduce;
      bdr_speed_reduce.nq = face_bdr_nq_;
      bdr_speed_reduce.face_speed_q = face_bdr_speed_q_.Read();
      bdr_speed_reduce.face_speed = face_bdr_speed_.Write();
      mfem::forall(n_bdr, bdr_speed_reduce);
    }
    else
    {
      PackedBoundaryFaceEulerRusanovKernel bdr_kernel;
      bdr_kernel.dof = dof;
      bdr_kernel.neq = neq;
      bdr_kernel.dim = dim;
      bdr_kernel.nq = face_bdr_nq_;
      bdr_kernel.ldof = ldof;
      bdr_kernel.sigma_ldof = sigma_ldof;
      bdr_kernel.gamma_default = dgsem_gamma_;
      bdr_kernel.sigma_rho_floor = sigma_rho_floor;
      bdr_kernel.invariant_domain_pressure_floor =
        invariant_domain_pressure_floor;
      bdr_kernel.use_sigma = use_sigma;
      bdr_kernel.invariant_domain_flux = invariant_domain_flux;
      bdr_kernel.sigma_aware_cfl = sigma_aware_cfl;
      bdr_kernel.sigma_aware_rusanov = sigma_aware_rusanov;
      bdr_kernel.trace_centered = trace_centered;
      bdr_kernel.bdr_max_attr = face_bdr_max_attr_;
      bdr_kernel.state_map = invmass_elem_vdof_map_.Read();
      bdr_kernel.face_e1 = face_bdr_e1_.Read();
      bdr_kernel.face_attr = face_bdr_attr_.Read();
      bdr_kernel.shape1 = face_bdr_shape1_.Read();
      bdr_kernel.normals = face_bdr_normals_.Read();
      bdr_kernel.weights = face_bdr_weights_.Read();
      bdr_kernel.x = x_in.Read();
      bdr_kernel.z = z_inout.ReadWrite();
      bdr_kernel.face_speed = face_bdr_speed_.Write();
      bdr_kernel.q_sigma_telemetry = bdr_q_sigma_telemetry;
      bdr_kernel.boundary_flux_integral = boundary_flux_integral;
      bdr_kernel.bdr_type_by_attr = face_bdr_type_by_attr_.Read();
      bdr_kernel.bdr_gamma_by_attr = face_bdr_gamma_by_attr_.Read();
      bdr_kernel.bdr_pressure_by_attr = face_bdr_pressure_by_attr_.Read();
      bdr_kernel.bdr_state_by_attr = face_bdr_state_by_attr_.Read();
      bdr_kernel.sigma_map = sigma_map;
      bdr_kernel.sigma_vals = sigma_vals;
      bdr_kernel.sigma_shape1 = bdr_sigma_shape1;
      mfem::forall(n_bdr, bdr_kernel);
    }
  }

  const int n_shared = face_shared_e1_.Size();
  MFEM_VERIFY(!invariant_domain_flux ||
                face_shared_q_sigma_telemetry_.Size() ==
                  n_shared * face_shared_nq_ *
                    dg_igr_detail::kFrozenSigmaTelemetryWidth,
              "Packed invariant-domain shared telemetry is not initialized.");
  double* const shared_q_sigma_telemetry =
    invariant_domain_flux ? face_shared_q_sigma_telemetry_.ReadWrite()
                          : nullptr;
  if (n_shared > 0)
  {
#ifdef MFEM_USE_MPI
    auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&vfes);
    if (!pfes)
    {
      max_face_speed = 0.0;
      return false;
    }
    MFEM_VERIFY(x_in.Size() == pfes->GetVSize(),
                "DG shared-face device assembly expects local L-vector size.");

    if (!face_state_face_nbr_alias_ ||
        face_state_face_nbr_alias_->ParFESpace() != pfes)
    {
      face_state_face_nbr_alias_ = std::make_unique<ParGridFunction>(pfes);
    }
    face_state_face_nbr_alias_->MakeRef(pfes, const_cast<Vector&>(x_in), 0);
    face_state_face_nbr_alias_->ExchangeFaceNbrData();
    const Vector& x_nbr_data = face_state_face_nbr_alias_->FaceNbrData();
    if (x_nbr_data.Size() == 0)
    {
      max_face_speed = 0.0;
      return false;
    }
    x_nbr_vals = x_nbr_data.Read();

    if (use_sigma)
    {
      const auto* sfes = dynamic_cast<const ParFiniteElementSpace*>(sigma_field->ParFESpace());
      if (!sfes)
      {
        max_face_speed = 0.0;
        return false;
      }
      const_cast<ParGridFunction*>(sigma_field)->ExchangeFaceNbrData();
      const Vector& sigma_nbr_data =
        const_cast<ParGridFunction*>(sigma_field)->FaceNbrData();
      if (sigma_nbr_data.Size() == 0)
      {
        max_face_speed = 0.0;
        return false;
      }
      sigma_nbr_vals = sigma_nbr_data.Read();
    }

    if (use_q_parallel)
    {
      PackedSharedFaceEulerRusanovQKernel shared_kernel;
      shared_kernel.dof = dof;
      shared_kernel.neq = neq;
      shared_kernel.dim = dim;
      shared_kernel.nq = face_shared_nq_;
      shared_kernel.ldof = ldof;
      shared_kernel.sigma_ldof = sigma_ldof;
      shared_kernel.gamma = dgsem_gamma_;
      shared_kernel.sigma_rho_floor = sigma_rho_floor;
      shared_kernel.invariant_domain_pressure_floor =
        invariant_domain_pressure_floor;
      shared_kernel.use_sigma = use_sigma;
      shared_kernel.invariant_domain_flux = invariant_domain_flux;
      shared_kernel.sigma_aware_cfl = sigma_aware_cfl;
      shared_kernel.sigma_aware_rusanov = sigma_aware_rusanov;
      shared_kernel.trace_centered = trace_centered;
      shared_kernel.state_map = invmass_elem_vdof_map_.Read();
      shared_kernel.face_e1 = face_shared_e1_.Read();
      shared_kernel.face_nbr_vdof_map = face_shared_nbr_vdof_map_.Read();
      shared_kernel.shape1 = face_shared_shape1_.Read();
      shared_kernel.shape2 = face_shared_shape2_.Read();
      shared_kernel.normals = face_shared_normals_.Read();
      shared_kernel.weights = face_shared_weights_.Read();
      shared_kernel.x = x_in.Read();
      shared_kernel.x_nbr = x_nbr_vals;
      shared_kernel.z = record_flux_only ? nullptr : z_inout.ReadWrite();
      shared_kernel.flux_record =
        record_flux_only ? face_shared_flux_record_.Write() : nullptr;
      shared_kernel.face_speed_q = face_shared_speed_q_.Write();
      shared_kernel.q_sigma_telemetry = shared_q_sigma_telemetry;
      shared_kernel.sigma_map = sigma_map;
      shared_kernel.sigma_vals = sigma_vals;
      shared_kernel.sigma_nbr_dof_map = shared_sigma_nbr_dof_map;
      shared_kernel.sigma_nbr_vals = sigma_nbr_vals;
      shared_kernel.sigma_shape1 = shared_sigma_shape1;
      shared_kernel.sigma_shape2 = shared_sigma_shape2;
      mfem::forall(n_shared * face_shared_nq_, shared_kernel);

      FaceSpeedQReduceKernel shared_speed_reduce;
      shared_speed_reduce.nq = face_shared_nq_;
      shared_speed_reduce.face_speed_q = face_shared_speed_q_.Read();
      shared_speed_reduce.face_speed = face_shared_speed_.Write();
      mfem::forall(n_shared, shared_speed_reduce);
    }
    else
    {
      PackedSharedFaceEulerRusanovKernel shared_kernel;
      shared_kernel.dof = dof;
      shared_kernel.neq = neq;
      shared_kernel.dim = dim;
      shared_kernel.nq = face_shared_nq_;
      shared_kernel.ldof = ldof;
      shared_kernel.sigma_ldof = sigma_ldof;
      shared_kernel.gamma = dgsem_gamma_;
      shared_kernel.sigma_rho_floor = sigma_rho_floor;
      shared_kernel.invariant_domain_pressure_floor =
        invariant_domain_pressure_floor;
      shared_kernel.use_sigma = use_sigma;
      shared_kernel.invariant_domain_flux = invariant_domain_flux;
      shared_kernel.sigma_aware_cfl = sigma_aware_cfl;
      shared_kernel.sigma_aware_rusanov = sigma_aware_rusanov;
      shared_kernel.trace_centered = trace_centered;
      shared_kernel.state_map = invmass_elem_vdof_map_.Read();
      shared_kernel.face_e1 = face_shared_e1_.Read();
      shared_kernel.face_nbr_vdof_map = face_shared_nbr_vdof_map_.Read();
      shared_kernel.shape1 = face_shared_shape1_.Read();
      shared_kernel.shape2 = face_shared_shape2_.Read();
      shared_kernel.normals = face_shared_normals_.Read();
      shared_kernel.weights = face_shared_weights_.Read();
      shared_kernel.x = x_in.Read();
      shared_kernel.x_nbr = x_nbr_vals;
      shared_kernel.z = z_inout.ReadWrite();
      shared_kernel.face_speed = face_shared_speed_.Write();
      shared_kernel.q_sigma_telemetry = shared_q_sigma_telemetry;
      shared_kernel.sigma_map = sigma_map;
      shared_kernel.sigma_vals = sigma_vals;
      shared_kernel.sigma_nbr_dof_map = shared_sigma_nbr_dof_map;
      shared_kernel.sigma_nbr_vals = sigma_nbr_vals;
      shared_kernel.sigma_shape1 = shared_sigma_shape1;
      shared_kernel.sigma_shape2 = shared_sigma_shape2;
      mfem::forall(n_shared, shared_kernel);
    }
#else
    max_face_speed = 0.0;
    return false;
#endif
  }

  const real_t max_int =
    ReduceFaceMaxSpeed_(face_int_speed_, face_int_speed_reduce_workspace_);
  const real_t max_bdr =
    ReduceFaceMaxSpeed_(face_bdr_speed_, face_bdr_speed_reduce_workspace_);
  const real_t max_shared =
    ReduceFaceMaxSpeed_(face_shared_speed_, face_shared_speed_reduce_workspace_);
  max_face_speed = std::max(max_shared, std::max(max_int, max_bdr));
  return true;
#endif
}

} // namespace mfem
