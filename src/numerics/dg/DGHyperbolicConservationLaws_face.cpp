#include "igrlab/compat.hpp"
#include "DGHyperbolicConservationLaws_impl.hpp"
#include "DGEntropyFaceKernels.hpp"
#include <typeinfo>

namespace mfem
{

#include "DGHyperbolicConservationLaws_device_kernels.hpp"
#include "DGTensorFaceKernels.hpp"

void DGHyperbolicConservationLaws::SetP0FaceReferenceXSlopes(const Vector* slopes)
{
  MFEM_VERIFY(!slopes || slopes->Size() == vfes.GetVSize(),
              "P0 MUSCL slope vector has an incompatible native layout.");
  if (slopes && slopes != p0_face_reference_x_slopes_)
  {
    MFEM_VERIFY(Device::Allows(Backend::CUDA_MASK) &&
                  vol_mode_ == VolumeMode::WeakDivergence && dim == 2 &&
                  num_equations == 4 && slopes->Size() == vfes.GetVSize() &&
                  vfes.GetNE() > 0,
                "P0 MUSCL traces require 2D four-field CUDA weak-DG transport.");
    for (int e = 0; e < vfes.GetNE(); ++e)
      MFEM_VERIFY(vfes.GetFE(e)->GetOrder() == 0 && vfes.GetFE(e)->GetDof() == 1 &&
                    vfes.GetFE(e)->GetGeomType() == Geometry::SQUARE,
                  "P0 MUSCL traces require quadrilateral cell-mean test functions.");
#ifdef MFEM_USE_MPI
    const auto* pfes = dynamic_cast<const ParFiniteElementSpace*>(&vfes);
    int ranks = 0;
    if (pfes) MPI_Comm_size(pfes->GetComm(), &ranks);
    MFEM_VERIFY(pfes && ranks == 1 && pfes->GetParMesh()->GetNSharedFaces() == 0,
                "P0 MUSCL traces support rank one without shared faces.");
#endif
  }
  p0_face_reference_x_slopes_ = slopes;
}

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

bool DGHyperbolicConservationLaws::BuildTensorFaceApplyData_()
{
  using dg_igr_detail::tensor_face::CheckedSize;
  const FluxFunction& physical = formIntegrator->GetFluxFunction();
  const auto* igr = dynamic_cast<const FluxWithIGR*>(&physical);
  const auto* sigma = igr ? igr->GetSigmaField() : nullptr;
  face_tensor_data_ = std::make_shared<dg_igr_detail::TensorFaceData>();
  auto& data = *face_tensor_data_;
  data.key = dg_igr_detail::MakeTensorFaceKey(vfes, sigma, num_flux_,
    p0_face_reference_x_slopes_, formIntegrator->GetIntegrationRule(),
    face_int_order_offset_, dgsem_gamma_, true, igr);
  const auto fallback = [](const char* reason)
  {
    if (mfem::Mpi::Root())
      std::cout << "[dg-tensor-face] event=fallback reason=" << reason
                << " row_reuse_requested="
                << (dg_igr_detail::tensor_face::RowReuseRequested() ? 1 : 0)
                << " row_reuse=0" << std::endl;
    return false;
  };
#ifndef DG_IGR_HAS_ACCEL_FORALL
  return fallback("backend");
#else
  const unsigned long native = static_cast<unsigned long>(Backend::CPU) |
                               static_cast<unsigned long>(Backend::CUDA);
  Mesh* mesh = vfes.GetMesh();
  if (!Device::Allows(Backend::CUDA) || Device::Allows(~native) ||
      sizeof(real_t) != sizeof(double)) { return fallback("backend"); }
  if (!mesh || !mesh->Conforming() || mesh->Dimension() != 2 ||
      mesh->SpaceDimension() != 2 || dim != 2 || num_equations != 4 ||
      vfes.GetVDim() != 4 || !invmass_apply_ready_ || !HasPreassembledWeakDivergence_() ||
      vol_mode_ != VolumeMode::WeakDivergence || vfes.GetNE() <= 0 ||
      !dynamic_cast<const L2_FECollection*>(vfes.FEColl()))
    { return fallback("state-layout"); }
#ifdef MFEM_USE_MPI
  if (const auto* parallel = dynamic_cast<const ParFiniteElementSpace*>(&vfes))
  {
    int ranks = 0;
    MPI_Comm_size(parallel->GetComm(), &ranks);
    if (ranks != 1 || parallel->GetParMesh()->GetNSharedFaces() != 0)
      { return fallback("mpi"); }
  }
#endif
  if (!standard_face_integrator_ || !num_flux_ ||
      num_flux_ != standard_face_flux_ ||
      face_int_order_offset_ != standard_face_int_order_offset_ ||
      p0_face_reference_x_slopes_ || &physical != &num_flux_->GetFluxFunction())
    { return fallback("flux-policy"); }
  const bool rusanov = typeid(*num_flux_) == typeid(RusanovFluxWithSides) ||
    (typeid(*num_flux_) == typeid(DGSEMSurfaceFlux) &&
      static_cast<const DGSEMSurfaceFlux*>(num_flux_)->GetSurfaceFluxType() ==
        DGSEMSurfaceFluxType::Rusanov);
  const FluxFunction* base = &physical;
  if (igr)
  {
    if (typeid(physical) != typeid(FluxWithIGR) || !igr->HasEulerBase() ||
        igr->GetGamma() != dgsem_gamma_) { return fallback("flux-policy"); }
    base = &igr->GetBaseFlux();
  }
  if (!rusanov || typeid(*base) != typeid(EulerFlux) || base->dim != 2 ||
      !(dgsem_gamma_ > 1.0 && dgsem_gamma_ <= 2.0))
    { return fallback("flux-policy"); }
  // The exact Euler type exposes gamma through this stationary unit-state
  // flux. Subtraction is exact for the admitted gamma interval.
  real_t probe_data[4] = {1.0, 0.0, 0.0, 1.0}, probe_n[2] = {1.0, 0.0};
  real_t probe_flux[4];
  Vector probe(probe_data, 4), normal_probe(probe_n, 2), flux_probe(probe_flux, 4);
  FaceElementTransformations probe_tr;
  base->ComputeFluxDotN(probe, normal_probe, probe_tr, flux_probe);
  if (probe_flux[1] != dgsem_gamma_ - 1.0) { return fallback("flux-policy"); }

  const auto& reference = *vfes.GetFE(0);
  const int order = reference.GetOrder(), n = order + 1, dof = n * n;
  const auto* tensor = dynamic_cast<const TensorBasisElement*>(&reference);
  if (!tensor || tensor->GetBasisType() != BasisType::GaussLegendre ||
      reference.GetMapType() != FiniteElement::VALUE ||
      reference.GetRangeType() != FiniteElement::SCALAR ||
      reference.GetGeomType() != Geometry::SQUARE || n < 2 || n > 16 ||
      dof != invmass_elem_dof_ || reference.GetDof() != dof ||
      reference.GetNodes().Size() != dof)
    { return fallback("state-layout"); }
  data.n = n;
  data.elements = vfes.GetNE();
  data.lex_to_local.SetSize(dof);
  auto* local = data.lex_to_local.HostWrite();
  const auto& permutation = tensor->GetDofMap();
  if (permutation.Size() != 0 && permutation.Size() != dof)
    { return fallback("state-layout"); }
  const auto& line_rule = IntRules.Get(Geometry::SEGMENT, 2 * order + 1);
  if (line_rule.GetNPoints() != n) { return fallback("quadrature"); }
  std::vector<char> used(static_cast<std::size_t>(dof), 0);
  for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
  {
    const int lex = i + n * j, index = permutation.Size() ? permutation[lex] : lex;
    if (index < 0 || index >= dof || used[static_cast<std::size_t>(index)])
      { return fallback("state-layout"); }
    used[static_cast<std::size_t>(index)] = 1;
    local[lex] = index;
    const auto& point = reference.GetNodes().IntPoint(index);
    if (point.x != line_rule.IntPoint(i).x || point.y != line_rule.IntPoint(j).x)
      { return fallback("state-layout"); }
  }
  const auto positive_jacobian = [](ElementTransformation& transform,
                                    const IntegrationPoint& point)
  {
    transform.SetIntPoint(&point);
    const auto& jacobian = transform.Jacobian();
    if (jacobian.Height() != 2 || jacobian.Width() != 2) { return false; }
    const real_t a = jacobian(0, 0), b = jacobian(0, 1);
    const real_t c = jacobian(1, 0), d = jacobian(1, 1);
    const real_t determinant = a * d - b * c;
    return std::isfinite(a) && std::isfinite(b) && std::isfinite(c) &&
      std::isfinite(d) && std::isfinite(determinant) && determinant > 0.0 &&
      std::isfinite(transform.Weight()) && transform.Weight() > 0.0;
  };
  int maximum_geometry_order = 1;
  Array<int> element_dofs;
  for (int e = 0; e < data.elements; ++e)
  {
    const auto& fe = *vfes.GetFE(e);
    if (&fe != &reference || fe.GetDof() != dof ||
        vfes.GetElementDofs(e, element_dofs) || element_dofs.Size() != dof)
      { return fallback("state-layout"); }
    for (int i = 0; i < dof; ++i)
      if (element_dofs[i] < 0 || element_dofs[i] >= vfes.GetNDofs())
        { return fallback("state-layout"); }
    auto* transform = vfes.GetElementTransformation(e);
    if (!transform) { return fallback("geometry"); }
    if (transform->Order() == 1)
    {
      Vector corners[4] = {Vector(2), Vector(2), Vector(2), Vector(2)};
      for (int c = 0; c < 4; ++c)
      {
        IntegrationPoint point;
        point.Set2(c % 2, c / 2);
        transform->Transform(point, corners[c]);
      }
      for (int d = 0; d < 2; ++d)
        if (!std::isfinite(corners[0][d]) || !std::isfinite(corners[1][d]) ||
            !std::isfinite(corners[2][d]) || !std::isfinite(corners[3][d]) ||
            corners[3][d] - corners[1][d] != corners[2][d] - corners[0][d])
          { return fallback("geometry"); }
    }
    else
    {
      const auto* isoparametric = dynamic_cast<const IsoparametricTransformation*>(transform);
      const auto* geometry_fe = isoparametric ? isoparametric->GetFE() : nullptr;
      if (transform->Order() != 2 || mesh->NURBSext || !geometry_fe ||
          (typeid(*geometry_fe) != typeid(H1_QuadrilateralElement) &&
           typeid(*geometry_fe) != typeid(L2_QuadrilateralElement)) ||
          geometry_fe->GetGeomType() != Geometry::SQUARE ||
          geometry_fe->GetMapType() != FiniteElement::VALUE ||
          geometry_fe->GetRangeType() != FiniteElement::SCALAR ||
          geometry_fe->GetDof() != 9 || geometry_fe->GetNodes().Size() != 9)
        { return fallback("geometry"); }
      const auto& coordinates = isoparametric->GetPointMat();
      if (coordinates.Height() != 2 || coordinates.Width() != 9)
        { return fallback("geometry"); }
      for (int j = 0; j < 9; ++j) for (int d = 0; d < 2; ++d)
        if (!std::isfinite(coordinates(d, j))) { return fallback("geometry"); }
      // These setup samples reject invalid metrics where observed; they do not
      // certify positivity between samples of the polynomial Q2 map.
      const IntegrationRule* samples[] = {
        Geometries.GetVertices(Geometry::SQUARE), &geometry_fe->GetNodes(),
        &fe.GetNodes(), &MassIntegrator::GetRule(fe, fe, *transform),
        &GradientIntegrator::GetRule(fe, fe, *transform),
        &DiffusionIntegrator::GetRule(fe, fe)};
      for (int sample = 0; sample < 6; ++sample)
      {
        bool duplicate = false;
        for (int previous = 0; previous < sample; ++previous)
          duplicate = duplicate || samples[previous] == samples[sample];
        if (duplicate) { continue; }
        for (int q = 0; q < samples[sample]->GetNPoints(); ++q)
          if (!positive_jacobian(*transform, samples[sample]->IntPoint(q)))
            { return fallback("geometry"); }
      }
      maximum_geometry_order = 2;
    }
  }

  // Matching sigma coefficients may have a different local-node permutation.
  // Build that permutation from actual coordinates; never infer it from p/dof.
  if (sigma)
  {
    const auto* scalar = sigma->ParFESpace();
    if (!scalar || scalar->GetMesh() != mesh || scalar->GetNE() != data.elements ||
        scalar->GetVDim() != 1 || !dynamic_cast<const L2_FECollection*>(scalar->FEColl()))
      { return fallback("sigma-layout"); }
    data.sigma_map.SetSize(CheckedSize(std::size_t(data.elements) * dof));
    auto* scalar_map = data.sigma_map.HostWrite();
    const FiniteElement* previous = nullptr;
    std::vector<int> scalar_permutation(static_cast<std::size_t>(dof), -1);
    for (int e = 0; e < data.elements; ++e)
    {
      const auto& fe = *scalar->GetFE(e);
      const auto* scalar_tensor = dynamic_cast<const TensorBasisElement*>(&fe);
      if (!scalar_tensor || scalar_tensor->GetBasisType() != BasisType::GaussLegendre ||
          fe.GetMapType() != FiniteElement::VALUE ||
          fe.GetRangeType() != FiniteElement::SCALAR ||
          fe.GetGeomType() != Geometry::SQUARE || fe.GetOrder() != order ||
          fe.GetDof() != dof || fe.GetNodes().Size() != dof ||
          scalar->GetElementDofs(e, element_dofs) || element_dofs.Size() != dof)
        { return fallback("sigma-layout"); }
      if (previous != &fe)
      {
        std::fill(used.begin(), used.end(), 0);
        for (int j = 0; j < dof; ++j)
        {
          const auto& target = reference.GetNodes().IntPoint(j);
          int match = -1;
          for (int k = 0; k < dof; ++k)
          {
            const auto& point = fe.GetNodes().IntPoint(k);
            if (point.x == target.x && point.y == target.y) { match = k; break; }
          }
          if (match < 0 || used[static_cast<std::size_t>(match)])
            { return fallback("sigma-layout"); }
          used[static_cast<std::size_t>(match)] = 1;
          scalar_permutation[static_cast<std::size_t>(j)] = match;
        }
        previous = &fe;
      }
      for (int j = 0; j < dof; ++j)
      {
        const int global = element_dofs[scalar_permutation[static_cast<std::size_t>(j)]];
        if (global < 0 || global >= sigma->Size()) { return fallback("sigma-layout"); }
        scalar_map[e * dof + j] = global;
      }
    }
  }

  L2_FECollection line_collection(order, 1, BasisType::GaussLegendre);
  const auto& line = *line_collection.FiniteElementForGeometry(Geometry::SEGMENT);
  data.endpoints.SetSize(2 * n);
  auto* endpoint = data.endpoints.HostWrite();
  Vector line_shape(n), shape(dof), normal(2);
  for (int end = 0; end < 2; ++end)
  {
    IntegrationPoint point;
    point.x = end;
    line.CalcShape(point, line_shape);
    for (int k = 0; k < n; ++k) { endpoint[end * n + k] = line_shape[k]; }
  }
  data.tangents.SetSize(8 * n * n);
  data.tangents = 0.0;
  auto* tangent = data.tangents.HostWrite();
  std::array<bool, 8> reference_ready{};
  std::vector<real_t> tangent_points(static_cast<std::size_t>(8 * n));
  data.element_faces.SetSize(CheckedSize(std::size_t(data.elements) * 4));
  data.side_maps.SetSize(data.element_faces.Size());
  data.element_faces = -1;
  data.side_maps = -1;
  auto* incidence_faces = data.element_faces.HostWrite();
  auto* side_maps = data.side_maps.HostWrite();
  std::vector<int> interior_e1, interior_e2, interior_mesh, boundary_e, boundary_attr;
  std::vector<int> boundary_mesh, interior_side1, interior_side2, boundary_side;
  std::vector<real_t> interior_normal, boundary_normal, interior_weights, boundary_weights;

  const auto rule_supported = [&](const IntegrationRule& rule)
  {
    if (rule.GetNPoints() != n) { return false; }
    for (int q = 0; q < n; ++q)
      if (rule.IntPoint(q).x != line_rule.IntPoint(q).x ||
          rule.IntPoint(q).weight != line_rule.IntPoint(q).weight) { return false; }
    return true;
  };
  const auto map_side = [&](FaceElementTransformations& transform, bool second,
                            const IntegrationRule& rule, int& incidence) -> bool
  {
    transform.SetAllIntPoints(&rule.IntPoint(0));
    const auto& first = second ? transform.GetElement2IntPoint() :
                                 transform.GetElement1IntPoint();
    const int side = first.x == 0.0 ? 0 : first.x == 1.0 ? 1 :
                     first.y == 0.0 ? 2 : first.y == 1.0 ? 3 : -1;
    if (side < 0) { return false; }
    const real_t first_t = side < 2 ? first.y : first.x;
    const bool reverse = std::abs(first_t - line_rule.IntPoint(n - 1).x) <
                         std::abs(first_t - line_rule.IntPoint(0).x);
    const int signature = 2 * side + static_cast<int>(reverse);
    const int e = second ? transform.Elem2No : transform.Elem1No;
    if (e < 0 || e >= data.elements) { return false; }
    incidence = 4 * e + side;
    if (incidence_faces[incidence] != -1) { return false; }
    for (int q = 0; q < n; ++q)
    {
      transform.SetAllIntPoints(&rule.IntPoint(q));
      const auto& point = second ? transform.GetElement2IntPoint() :
                                  transform.GetElement1IntPoint();
      const real_t fixed = side < 2 ? point.x : point.y;
      const real_t t = side < 2 ? point.y : point.x;
      const real_t expected = reverse ? 1.0 - rule.IntPoint(q).x : rule.IntPoint(q).x;
      if (fixed != real_t(side % 2) || !std::isfinite(t) ||
          std::abs(t - expected) > 32 * std::numeric_limits<real_t>::epsilon())
        { return false; }
      if (reference_ready[static_cast<std::size_t>(signature)])
      {
        if (t != tangent_points[static_cast<std::size_t>(signature * n + q)])
          { return false; }
      }
      else
      {
        tangent_points[static_cast<std::size_t>(signature * n + q)] = t;
        IntegrationPoint line_point;
        line_point.x = t;
        line.CalcShape(line_point, line_shape);
        for (int k = 0; k < n; ++k)
          tangent[(signature * n + q) * n + k] = line_shape[k];
        // Prove the reference tensor factorization against the original FE.
        reference.CalcShape(point, shape);
        for (int lex = 0; lex < dof; ++lex)
        {
          const int i = lex % n, j = lex / n;
          const real_t reconstructed = endpoint[(side % 2) * n + (side < 2 ? i : j)] *
            line_shape[side < 2 ? j : i];
          if (!std::isfinite(reconstructed) || !std::isfinite(shape[local[lex]]) ||
              std::abs(shape[local[lex]] - reconstructed) >
                64 * dof * std::numeric_limits<real_t>::epsilon() *
                  std::max(real_t(1.0), std::abs(shape[local[lex]])))
            { return false; }
        }
      }
    }
    reference_ready[static_cast<std::size_t>(signature)] = true;
    side_maps[incidence] = signature;
    return true;
  };
  const auto append_metrics = [&](FaceElementTransformations& transform,
    const IntegrationRule& rule, std::vector<real_t>& normals,
    std::vector<real_t>& weights)
  {
    for (int q = 0; q < n; ++q)
    {
      transform.SetAllIntPoints(&rule.IntPoint(q));
      if (maximum_geometry_order == 2 &&
          (!transform.Elem1 ||
           !positive_jacobian(*transform.Elem1, transform.GetElement1IntPoint()) ||
           (transform.Elem2No >= 0 && (!transform.Elem2 ||
            !positive_jacobian(*transform.Elem2, transform.GetElement2IntPoint())))))
        { return false; }
      CalcOrtho(transform.Jacobian(), normal);
      if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) ||
          !(normal[0] * normal[0] + normal[1] * normal[1] > 0.0)) { return false; }
      normals.push_back(normal[0]); normals.push_back(normal[1]);
      weights.push_back(rule.IntPoint(q).weight);
    }
    return true;
  };
  for (int f = 0; f < mesh->GetNumFaces(); ++f)
  {
    auto* transform = mesh->GetFaceElementTransformations(f);
    if (!transform || transform->Elem2No < 0) { continue; }
    const auto* rule = formIntegrator->GetIntegrationRule();
    if (!rule) { rule = &IntRules.Get(transform->GetGeometryType(), 2 * order + face_int_order_offset_); }
    if (transform->GetGeometryType() != Geometry::SEGMENT || !rule_supported(*rule))
      { return fallback("quadrature"); }
    int left = -1, right = -1;
    if (!map_side(*transform, false, *rule, left) ||
        !map_side(*transform, true, *rule, right)) { return fallback("orientation"); }
    const int packed = static_cast<int>(interior_e1.size());
    incidence_faces[left] = 2 * packed;
    incidence_faces[right] = 2 * packed + 1;
    interior_e1.push_back(transform->Elem1No); interior_e2.push_back(transform->Elem2No);
    interior_mesh.push_back(f); interior_side1.push_back(left); interior_side2.push_back(right);
    if (!append_metrics(*transform, *rule, interior_normal, interior_weights))
      { return fallback("geometry"); }
  }
  data.interior_faces = static_cast<int>(interior_e1.size());
  for (int b = 0; b < mesh->GetNBE(); ++b)
  {
    auto* transform = mesh->GetBdrFaceTransformations(b);
    // MakePeriodic retains boundary element records for identified seams.
    // MFEM returns null for those now-interior faces, already packed above.
    // Match the legacy boundary loop and count only physical boundaries.
    if (!transform) { continue; }
    const auto* integrator = FindBdrIntegrator_(transform->Attribute);
    if (!integrator || typeid(*integrator) != typeid(BdrHyperbolicFluxIntegrator) ||
        &integrator->GetNumericalFlux() != num_flux_)
      { return fallback("boundary-policy"); }
    const auto* rule = integrator->GetIntegrationRule();
    if (!rule) { rule = &IntRules.Get(transform->GetGeometryType(), 2 * order + face_int_order_offset_); }
    if (transform->GetGeometryType() != Geometry::SEGMENT || !rule_supported(*rule))
      { return fallback("quadrature"); }
    int side = -1;
    if (!map_side(*transform, false, *rule, side)) { return fallback("orientation"); }
    const int packed = static_cast<int>(boundary_e.size());
    incidence_faces[side] = 2 * (data.interior_faces + packed);
    boundary_e.push_back(transform->Elem1No); boundary_attr.push_back(transform->Attribute);
    boundary_mesh.push_back(b); boundary_side.push_back(side);
    if (!append_metrics(*transform, *rule, boundary_normal, boundary_weights))
      { return fallback("geometry"); }
  }
  for (int incidence = 0; incidence < data.element_faces.Size(); ++incidence)
    if (incidence_faces[incidence] < 0 || side_maps[incidence] < 0)
      { return fallback("topology"); }
  data.boundary_faces = static_cast<int>(boundary_e.size());
  data.tangent_sparse.SetSize(8);
  data.tangent_permutation.SetSize(8 * n);
  data.tangent_inverse.SetSize(8 * n);
  auto* sparse_maps = data.tangent_sparse.HostWrite();
  auto* tangent_permutation = data.tangent_permutation.HostWrite();
  auto* tangent_inverse = data.tangent_inverse.HostWrite();
  int sparse_count = 0, reference_count = 0;
  for (int map = 0; map < 8; ++map)
  {
    bool sparse = reference_ready[static_cast<std::size_t>(map)];
    if (sparse) { ++reference_count; }
    for (int q = 0; q < n; ++q)
    {
      const int pivot = map % 2 ? n - 1 - q : q;
      tangent_permutation[map * n + q] = pivot;
      tangent_inverse[map * n + pivot] = q;
      for (int t = 0; t < n; ++t)
      {
        const real_t value = tangent[(map * n + q) * n + t];
        if ((t != pivot && value != 0.0) ||
            (t == pivot && (!std::isfinite(value) || value == 0.0))) { sparse = false; }
      }
    }
    sparse_maps[map] = sparse ? 1 : 0;
    if (sparse) { ++sparse_count; }
  }
  const auto copy_array = [](const std::vector<int>& input, Array<int>& output)
  {
    output.SetSize(dg_igr_detail::tensor_face::CheckedSize(input.size()));
    std::copy(input.begin(), input.end(), output.HostWrite());
  };
  const auto copy_vector = [](const std::vector<real_t>& input, Vector& output)
  {
    output.SetSize(dg_igr_detail::tensor_face::CheckedSize(input.size()));
    std::copy(input.begin(), input.end(), output.HostWrite());
  };
  copy_array(interior_e1, face_int_e1_); copy_array(interior_e2, face_int_e2_);
  copy_array(interior_mesh, face_int_mesh_id_);
  copy_array(boundary_e, face_bdr_e1_); copy_array(boundary_attr, face_bdr_attr_);
  copy_array(boundary_mesh, face_bdr_mesh_id_);
  if (!BuildBoundaryPolicyData_()) { return fallback("boundary-policy"); }
  copy_array(interior_side1, data.interior_side1);
  copy_array(interior_side2, data.interior_side2); copy_array(boundary_side, data.boundary_side);
  copy_vector(interior_normal, face_int_normals_); copy_vector(interior_weights, face_int_weights_);
  copy_vector(boundary_normal, face_bdr_normals_); copy_vector(boundary_weights, face_bdr_weights_);
  face_elem_dof_ = dof;
  face_int_nq_ = data.interior_faces ? n : 0;
  face_bdr_nq_ = data.boundary_faces ? n : 0;
  face_int_speed_.SetSize(data.interior_faces);
  face_bdr_speed_.SetSize(data.boundary_faces);
  face_int_speed_q_.SetSize(CheckedSize(std::size_t(data.interior_faces) * n));
  face_bdr_speed_q_.SetSize(CheckedSize(std::size_t(data.boundary_faces) * n));
  data.boundary_rejected.SetSize(face_bdr_speed_q_.Size());
  if (igr && sigma && igr->GetInvariantDomainFlux())
  {
    dg_igr_detail::InitializeFrozenSigmaTelemetry(face_int_q_sigma_telemetry_, face_int_speed_q_.Size());
    dg_igr_detail::InitializeFrozenSigmaTelemetry(face_bdr_q_sigma_telemetry_, face_bdr_speed_q_.Size());
  }
  data.traces.SetSize(CheckedSize(std::size_t(sigma ? 5 : 4) * 4 * data.elements * n));
  data.flux.SetSize(CheckedSize(std::size_t(4) *
    (data.interior_faces + data.boundary_faces) * n));
  // Release any former dense policy allocation when a live operator opts in.
  face_int_shape1_.Destroy(); face_int_shape2_.Destroy(); face_bdr_shape1_.Destroy();
  face_shared_shape1_.Destroy(); face_shared_shape2_.Destroy();
  face_int_sigma_shape1_.Destroy(); face_int_sigma_shape2_.Destroy();
  face_bdr_sigma_shape1_.Destroy(); face_shared_sigma_shape1_.Destroy();
  face_shared_sigma_shape2_.Destroy();
  face_int_ip1_ref_.Destroy(); face_int_ip2_ref_.Destroy(); face_bdr_ip1_ref_.Destroy();
  face_shared_ip1_ref_.Destroy(); face_shared_ip2_ref_.Destroy();
  data.ready = true;
  if (mfem::Mpi::Root())
    std::cout << "[dg-tensor-face] event=setup selected=fp64 n1d=" << n
              << " elements=" << data.elements << " interior_faces=" << data.interior_faces
              << " boundary_faces=" << data.boundary_faces
              << " geometry_order_max=" << maximum_geometry_order
              << " jacobian_validation=" << (maximum_geometry_order == 2 ? "sampled-q2" : "affine-q1")
              << " reference_maps=" << reference_count << " sparse_tangent_maps=" << sparse_count
              << " dense_shape_entries=0 trace_bytes=" << std::size_t(data.traces.Size()) * sizeof(real_t)
              << " flux_bytes=" << std::size_t(data.flux.Size()) * sizeof(real_t)
              << " state=fp64 nonlinear_flux=fp64 lift=element-owned residual_atomics=0"
              << " row_reuse_requested="
              << (dg_igr_detail::tensor_face::RowReuseRequested() ? 1 : 0)
              << " row_reuse="
              << (dg_igr_detail::tensor_face::RowReuseRequested() ? 1 : 0)
              << std::endl;
  return true;
#endif
}

void DGHyperbolicConservationLaws::BuildFaceApplyData_()
{
  face_tensor_requested_ = dg_igr_detail::tensor_face::Requested();
  face_tensor_data_.reset();
  face_entropy_ready_ = false;
  face_built_entropy_policy_ = SupportsPackedEntropyFacePolicy_();
  face_entropy_element_offsets_.SetSize(0);
  face_entropy_element_sides_.SetSize(0);
  face_entropy_flux_.SetSize(0);
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

  // Admission precedes all per-face dense shape construction/allocation.
  if (face_tensor_requested_ && BuildTensorFaceApplyData_())
  {
    face_apply_ready_ = true;
    return;
  }

  if (!invmass_apply_ready_ || !HasPreassembledWeakDivergence_() || !num_flux_)
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

  bool flux_supported = (dynamic_cast<const RusanovFluxWithSides*>(num_flux_) != nullptr);
  if (!flux_supported)
  {
    const auto* dgsem_flux = dynamic_cast<const DGSEMSurfaceFlux*>(num_flux_);
    flux_supported =
      (dgsem_flux && dgsem_flux->GetSurfaceFluxType() == DGSEMSurfaceFluxType::Rusanov);
  }
  flux_supported = flux_supported || face_built_entropy_policy_;
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

  if (face_built_entropy_policy_)
  {
    // The packed kernels use the actual DOF and quadrature counts, so all
    // positive uniform orders share this path. Retain the qualified rank-one,
    // periodic affine quadrilateral topology and the integrator's sampled
    // quadrature/normal data; unsupported layouts keep the host fallback.
    int ranks = 0;
#ifdef MFEM_USE_MPI
    if (pfes) { MPI_Comm_size(pfes->GetComm(), &ranks); }
#endif
    if (!pfes || ranks != 1 || !pmesh || pmesh->GetNSharedFaces() != 0 ||
        face_bdr_e1_.Size() != 0 || face_shared_e1_.Size() != 0 ||
        face_int_e1_.Size() == 0 ||
        !dynamic_cast<const L2_FECollection*>(vfes.FEColl())) { return; }
    const int order = vfes.GetFE(0)->GetOrder();
    if (order < 1) { return; }
    for (int e = 0; e < vfes.GetNE(); ++e)
    {
      const auto& fe = *vfes.GetFE(e);
      const auto* tensor = dynamic_cast<const TensorBasisElement*>(&fe);
      if (fe.GetOrder() != order || fe.GetDof() != dof ||
          fe.GetGeomType() != Geometry::SQUARE ||
          fe.GetMapType() != FiniteElement::VALUE || !tensor ||
          (tensor->GetBasisType() != BasisType::GaussLegendre &&
           tensor->GetBasisType() != BasisType::GaussLobatto)) { return; }
      auto* tr = vfes.GetElementTransformation(e);
      const auto& nodes = fe.GetNodes();
      tr->SetIntPoint(&nodes.IntPoint(0));
      DenseMatrix first(tr->Jacobian());
      for (int q = 1; q < nodes.GetNPoints(); ++q)
      {
        tr->SetIntPoint(&nodes.IntPoint(q));
        const auto& jac = tr->Jacobian();
        for (int j = 0; j < 2; ++j)
          for (int i = 0; i < 2; ++i)
            if (std::abs(jac(i, j) - first(i, j)) >
                64 * std::numeric_limits<real_t>::epsilon() *
                  std::max(real_t(1.0), std::abs(first(i, j)))) { return; }
      }
    }
    const int ne = vfes.GetNE();
    face_entropy_element_offsets_.SetSize(ne + 1);
    face_entropy_element_offsets_ = 0;
    for (int f = 0; f < face_int_e1_.Size(); ++f)
    {
      ++face_entropy_element_offsets_[face_int_e1_[f] + 1];
      ++face_entropy_element_offsets_[face_int_e2_[f] + 1];
    }
    for (int e = 0; e < ne; ++e)
      face_entropy_element_offsets_[e + 1] += face_entropy_element_offsets_[e];
    face_entropy_element_sides_.SetSize(2 * face_int_e1_.Size());
    Array<int> next(face_entropy_element_offsets_);
    for (int f = 0; f < face_int_e1_.Size(); ++f)
    {
      face_entropy_element_sides_[next[face_int_e1_[f]]++] = 2 * f;
      face_entropy_element_sides_[next[face_int_e2_[f]]++] = 2 * f + 1;
    }
    face_entropy_flux_.SetSize(4 * face_int_e1_.Size() * face_int_nq_);
    face_entropy_ready_ = true;
  }
  face_apply_ready_ = true;
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
  for (int e = 0; e < ne; ++e)
  {
    if (sfes->GetFE(e)->GetDof() != sigma_ldof ||
        (face_entropy_ready_ && sfes->GetFE(e)->GetMapType() != FiniteElement::VALUE))
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

bool DGHyperbolicConservationLaws::SupportsPackedEntropyFacePolicy_() const
{
  if (!standard_face_integrator_ || dim != 2 || num_equations != 4 ||
      num_flux_ != standard_face_flux_ ||
      face_int_order_offset_ != standard_face_int_order_offset_ || !num_flux_ || typeid(*num_flux_) != typeid(DGSEMSurfaceFlux)) { return false; }
  const auto& surface = static_cast<const DGSEMSurfaceFlux&>(*num_flux_);
  if (surface.GetSurfaceFluxType() != DGSEMSurfaceFluxType::EntropyStableWinters ||
      surface.GetEntropyFluxType() != dgsem::EntropyFluxType::Ranocha ||
      surface.GetGamma() != dgsem_gamma_ || !(dgsem_gamma_ > 1.0 && dgsem_gamma_ <= 2.0))
    { return false; }
  const FluxFunction& physical = formIntegrator->GetFluxFunction();
  if (&physical != &num_flux_->GetFluxFunction()) { return false; }
  const FluxFunction* base = &physical;
  if (typeid(physical) == typeid(FluxWithIGR))
  {
    const auto& igr = static_cast<const FluxWithIGR&>(physical);
    if (!igr.HasEulerBase() || igr.GetGamma() != dgsem_gamma_ ||
        igr.GetSigmaAwareCFL() || igr.GetSigmaAwareRusanov() ||
        igr.GetFrozenSigmaSpectralSpeed() || igr.GetInvariantDomainFlux()) { return false; }
    base = &igr.GetBaseFlux();
    const auto* sigma = igr.GetSigmaField();
    if (sigma && (!sigma->ParFESpace() || sigma->ParFESpace()->GetVDim() != 1 ||
                  sigma->ParFESpace()->GetMesh() != vfes.GetMesh())) { return false; }
  }
  if (typeid(*base) != typeid(EulerFlux) || base->dim != 2) { return false; }
  // MFEM keeps Euler's gamma private. For this exact type, the stationary
  // unit-density/unit-energy flux is gamma-1. On (1,2], subtraction is exact.
  real_t state_data[4] = {1.0, 0.0, 0.0, 1.0}, normal_data[2] = {1.0, 0.0};
  real_t flux_data[4];
  Vector state(state_data, 4), normal(normal_data, 2), flux(flux_data, 4);
  FaceElementTransformations tr;
  base->ComputeFluxDotN(state, normal, tr, flux);
  return flux(1) == dgsem_gamma_ - 1.0;
}

bool DGHyperbolicConservationLaws::UseDeviceFaceResidual_() const
{
  if (!Device::Allows(Backend::DEVICE_MASK) ||
      vol_mode_ != VolumeMode::WeakDivergence || !HasPreassembledWeakDivergence_() ||
      !invmass_apply_ready_) { return false; }
  const bool entropy = SupportsPackedEntropyFacePolicy_();
  const unsigned long native = static_cast<unsigned long>(Backend::CPU) |
                               static_cast<unsigned long>(Backend::CUDA);
  if (entropy && (!Device::Allows(Backend::CUDA) || Device::Allows(~native)))
    { return false; }
  const auto* surface = dynamic_cast<const DGSEMSurfaceFlux*>(num_flux_);
  const bool rusanov = dynamic_cast<const RusanovFluxWithSides*>(num_flux_) ||
    (surface && surface->GetSurfaceFluxType() == DGSEMSurfaceFluxType::Rusanov);
  if (!entropy && !rusanov) { return false; }
  const bool tensor_requested = dg_igr_detail::tensor_face::Requested();
  bool tensor_stale = tensor_requested != face_tensor_requested_;
  if (tensor_requested)
  {
    const auto* igr = dynamic_cast<const FluxWithIGR*>(&formIntegrator->GetFluxFunction());
    const auto* sigma = igr ? igr->GetSigmaField() : nullptr;
    const auto key = dg_igr_detail::MakeTensorFaceKey(vfes, sigma, num_flux_,
      p0_face_reference_x_slopes_, formIntegrator->GetIntegrationRule(),
      face_int_order_offset_, dgsem_gamma_, true, igr);
    tensor_stale = tensor_stale || !face_tensor_data_ ||
      !(face_tensor_data_->key == key);
  }
  if (face_data_dirty_ || tensor_stale || (entropy && !face_built_entropy_policy_))
    const_cast<DGHyperbolicConservationLaws*>(this)->BuildFaceApplyData_();
  if (entropy)
  {
    if (!Device::Allows(Backend::CUDA_MASK) || !face_apply_ready_ || !face_entropy_ready_)
      { return false; }
    const auto* igr = dynamic_cast<const FluxWithIGR*>(&formIntegrator->GetFluxFunction());
    const auto* sigma = igr ? igr->GetSigmaField() : nullptr;
    return !sigma || BuildFaceSigmaData_(*sigma);
  }
  // A mutable flux policy can safely return to the independent host route.
  return face_apply_ready_ && !face_entropy_ready_;
}

bool DGHyperbolicConservationLaws::AddTensorFaceContributions_(
  const Vector& x_in, Vector& z_out, real_t& max_face_speed,
  bool capture_boundary_flux) const
{
#ifndef DG_IGR_HAS_ACCEL_FORALL
  (void)x_in; (void)z_out; (void)capture_boundary_flux;
  max_face_speed = 0.0;
  return false;
#else
  if (!face_tensor_data_ || !face_tensor_data_->ready) { return false; }
  auto& data = *face_tensor_data_;
  const auto* igr = dynamic_cast<const FluxWithIGR*>(&formIntegrator->GetFluxFunction());
  const auto* sigma = igr ? igr->GetSigmaField() : nullptr;
  const bool use_sigma = sigma != nullptr;
  const bool invariant = use_sigma && igr->GetInvariantDomainFlux();
  MFEM_VERIFY(!use_sigma || data.sigma_map.Size() == data.elements * data.n * data.n,
              "Tensor DG faces require their current matching scalar map.");
  if (invariant)
  {
    if (face_int_q_sigma_telemetry_.Size() !=
        face_int_speed_q_.Size() * dg_igr_detail::kFrozenSigmaTelemetryWidth)
      dg_igr_detail::InitializeFrozenSigmaTelemetry(
        face_int_q_sigma_telemetry_, face_int_speed_q_.Size());
    if (face_bdr_q_sigma_telemetry_.Size() !=
        face_bdr_speed_q_.Size() * dg_igr_detail::kFrozenSigmaTelemetryWidth)
      dg_igr_detail::InitializeFrozenSigmaTelemetry(
        face_bdr_q_sigma_telemetry_, face_bdr_speed_q_.Size());
  }
  dg_igr_detail::tensor_face::TraceKernel trace;
  trace.n = data.n; trace.elements = data.elements; trace.use_sigma = use_sigma;
  trace.local = data.lex_to_local.Read(); trace.state_map = invmass_elem_vdof_map_.Read();
  trace.sigma_map = use_sigma ? data.sigma_map.Read() : nullptr;
  trace.endpoints = data.endpoints.Read(); trace.state = x_in.Read();
  trace.sigma = use_sigma ? sigma->Read() : nullptr;
  trace.traces = data.traces.Write();
  const bool row_reuse = dg_igr_detail::tensor_face::RowReuseRequested();
  if (row_reuse)
  {
    dg_igr_detail::tensor_face::TraceRowReuseKernel reuse;
    static_cast<dg_igr_detail::tensor_face::TraceKernel&>(reuse) = trace;
    mfem::forall(data.elements * 4 * data.n, reuse);
  }
  else
  {
    mfem::forall(data.elements * 4 * data.n, trace);
  }

  dg_igr_detail::tensor_face::FluxKernel flux;
  flux.n = data.n; flux.elements = data.elements;
  flux.interior_faces = data.interior_faces; flux.boundary_faces = data.boundary_faces;
  flux.max_attr = face_bdr_max_attr_; flux.use_sigma = use_sigma;
  flux.invariant_domain = invariant;
  flux.sigma_cfl = use_sigma && igr->GetSigmaAwareCFL();
  flux.sigma_rusanov = use_sigma && igr->GetSigmaAwareRusanov();
  flux.gamma = dgsem_gamma_;
  flux.rho_floor = use_sigma ? igr->GetVelocityRhoFloor() : 1.0e-12;
  flux.pressure_floor = use_sigma ? igr->GetInvariantDomainPressureFloor() :
                                     igrlab::LimiterDefaults::kPMin;
  flux.side1 = data.interior_side1.Read(); flux.side2 = data.interior_side2.Read();
  flux.boundary_side = data.boundary_side.Read(); flux.side_maps = data.side_maps.Read();
  flux.tangent_sparse = data.tangent_sparse.Read();
  flux.tangent_permutation = data.tangent_permutation.Read();
  flux.boundary_attr = face_bdr_attr_.Read(); flux.boundary_type = face_bdr_type_by_attr_.Read();
  flux.tangents = data.tangents.Read(); flux.traces = data.traces.Read();
  flux.interior_normal = face_int_normals_.Read(); flux.boundary_normal = face_bdr_normals_.Read();
  flux.boundary_gamma = face_bdr_gamma_by_attr_.Read();
  flux.boundary_pressure = face_bdr_pressure_by_attr_.Read();
  flux.boundary_state = face_bdr_state_by_attr_.Read();
  flux.flux = data.flux.Write(); flux.interior_speed = face_int_speed_q_.Write();
  flux.boundary_speed = face_bdr_speed_q_.Write();
  flux.boundary_rejected = data.boundary_rejected.Write();
  flux.interior_telemetry = invariant ? face_int_q_sigma_telemetry_.ReadWrite() : nullptr;
  flux.boundary_telemetry = invariant ? face_bdr_q_sigma_telemetry_.ReadWrite() : nullptr;
  mfem::forall((data.interior_faces + data.boundary_faces) * data.n, flux);

  dg_igr_detail::tensor_face::LiftKernel lift;
  lift.n = data.n; lift.interior_faces = data.interior_faces;
  lift.boundary_faces = data.boundary_faces;
  lift.local = data.lex_to_local.Read(); lift.state_map = invmass_elem_vdof_map_.Read();
  lift.element_faces = data.element_faces.Read(); lift.side_maps = data.side_maps.Read();
  lift.tangent_sparse = data.tangent_sparse.Read(); lift.tangent_inverse = data.tangent_inverse.Read();
  lift.boundary_rejected = data.boundary_rejected.Read();
  lift.endpoints = data.endpoints.Read(); lift.tangents = data.tangents.Read();
  lift.interior_weights = face_int_weights_.Read(); lift.boundary_weights = face_bdr_weights_.Read();
  lift.flux = data.flux.Read(); lift.residual = z_out.Write();
  if (row_reuse)
  {
    dg_igr_detail::tensor_face::LiftRowReuseKernel reuse;
    static_cast<dg_igr_detail::tensor_face::LiftKernel&>(reuse) = lift;
    mfem::forall(data.elements * data.n * data.n, reuse);
  }
  else
  {
    mfem::forall(data.elements * data.n * data.n * 4, lift);
  }
  if (capture_boundary_flux)
    mfem::forall(4, dg_igr_detail::tensor_face::BoundaryIntegralKernel{
      data.n, data.interior_faces, data.boundary_faces,
      face_bdr_weights_.Read(), data.flux.Read(), physical_boundary_flux_integral_.Write()});
  FaceSpeedQReduceKernel reduce_interior;
  reduce_interior.nq = data.n;
  reduce_interior.face_speed_q = face_int_speed_q_.Read();
  reduce_interior.face_speed = face_int_speed_.Write();
  mfem::forall(data.interior_faces, reduce_interior);
  FaceSpeedQReduceKernel reduce_boundary;
  reduce_boundary.nq = data.n;
  reduce_boundary.face_speed_q = face_bdr_speed_q_.Read();
  reduce_boundary.face_speed = face_bdr_speed_.Write();
  mfem::forall(data.boundary_faces, reduce_boundary);
  const real_t interior_max = ReduceFaceMaxSpeed_(face_int_speed_, face_int_speed_reduce_workspace_);
  const real_t boundary_max = ReduceFaceMaxSpeed_(face_bdr_speed_, face_bdr_speed_reduce_workspace_);
  max_face_speed = std::max(interior_max, boundary_max);
  return true;
#endif
}

bool DGHyperbolicConservationLaws::AddDeviceFaceContributionsEntropyStable_(
  const Vector& x_in, Vector& z_out, real_t& max_face_speed) const
{
#ifndef DG_IGR_HAS_ACCEL_FORALL
  (void)x_in; (void)z_out; max_face_speed = 0.0;
  return false;
#else
  if (!face_entropy_ready_ || !SupportsPackedEntropyFacePolicy_()) { return false; }
  const auto* igr = dynamic_cast<const FluxWithIGR*>(&formIntegrator->GetFluxFunction());
  const auto* sigma = igr ? igr->GetSigmaField() : nullptr;
  if (sigma && !BuildFaceSigmaData_(*sigma)) { return false; }
  dg_igr_detail::packed_entropy::FaceKernel face;
  face.dof = face_elem_dof_; face.nq = face_int_nq_;
  face.sigma_dof = sigma ? face_sigma_ldof_ : 0;
  face.gamma = dgsem_gamma_;
  face.e1 = face_int_e1_.Read(); face.e2 = face_int_e2_.Read();
  face.state_map = invmass_elem_vdof_map_.Read();
  face.sigma_map = sigma ? face_sigma_elem_dof_map_.Read() : nullptr;
  face.x = x_in.Read(); face.sigma = sigma ? sigma->Read() : nullptr;
  face.shape1 = face_int_shape1_.Read(); face.shape2 = face_int_shape2_.Read();
  face.sigma_shape1 = sigma ? face_int_sigma_shape1_.Read() : nullptr;
  face.sigma_shape2 = sigma ? face_int_sigma_shape2_.Read() : nullptr;
  face.normal = face_int_normals_.Read();
  face.flux = face_entropy_flux_.Write(); face.speed = face_int_speed_q_.Write();
  mfem::forall(face_int_e1_.Size() * face_int_nq_, face);
  dg_igr_detail::packed_entropy::GatherKernel gather;
  gather.dof = face_elem_dof_; gather.nq = face_int_nq_;
  gather.offsets = face_entropy_element_offsets_.Read();
  gather.sides = face_entropy_element_sides_.Read();
  gather.state_map = invmass_elem_vdof_map_.Read();
  gather.shape1 = face_int_shape1_.Read(); gather.shape2 = face_int_shape2_.Read();
  gather.weights = face_int_weights_.Read(); gather.flux = face_entropy_flux_.Read();
  gather.z = z_out.Write();
  mfem::forall(vfes.GetNE() * face_elem_dof_ * num_equations, gather);
  max_face_speed = ReduceFaceMaxSpeed_(face_int_speed_q_, face_int_speed_reduce_workspace_);
  MFEM_VERIFY(std::isfinite(max_face_speed),
              "Packed ES-Winters face flux encountered an inadmissible/nonfinite state.");
  return true;
#endif
}

bool DGHyperbolicConservationLaws::AddDeviceFaceContributionsEulerRusanov_(
  const Vector& x_in,
  Vector& z_inout,
  real_t& max_face_speed,
  bool capture_boundary_flux) const
{
#ifndef DG_IGR_HAS_ACCEL_FORALL
  (void)x_in;
  (void)z_inout;
  (void)capture_boundary_flux;
  max_face_speed = 0.0;
  return false;
#else
  if (!face_apply_ready_)
  {
    max_face_speed = 0.0;
    return false;
  }
  if (face_tensor_data_ && face_tensor_data_->ready)
    return AddTensorFaceContributions_(x_in, z_inout, max_face_speed, capture_boundary_flux);

  const FluxFunction& flux_fn = formIntegrator->GetFluxFunction();
  const auto* flux_igr = dynamic_cast<const FluxWithIGR*>(&flux_fn);

  bool use_sigma = false;
  bool invariant_domain_flux = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
  const ParGridFunction* sigma_field = nullptr;
  real_t sigma_rho_floor = 1.0e-12;
  real_t invariant_domain_pressure_floor =
    igrlab::LimiterDefaults::kPMin;
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

  z_inout = 0.0;

  const int dof = face_elem_dof_;
  const int neq = num_equations;
  const int ldof = dof * neq;
  const real_t* const transport_slopes =
    p0_face_reference_x_slopes_ ? p0_face_reference_x_slopes_->Read() : nullptr;
  MFEM_VERIFY(!transport_slopes || (dof == 1 && face_shared_e1_.Size() == 0),
              "P0 MUSCL packed faces cannot use higher-order tests or shared faces.");
  const bool use_q_parallel =
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_DG_FACE_Q_PARALLEL", true);

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
      int_kernel.state_map = invmass_elem_vdof_map_.Read();
      int_kernel.face_e1 = face_int_e1_.Read();
      int_kernel.face_e2 = face_int_e2_.Read();
      int_kernel.shape1 = face_int_shape1_.Read();
      int_kernel.shape2 = face_int_shape2_.Read();
      int_kernel.normals = face_int_normals_.Read();
      int_kernel.weights = face_int_weights_.Read();
      int_kernel.x = x_in.Read();
      int_kernel.reference_x_slopes = transport_slopes;
      int_kernel.ip1_ref = transport_slopes ? face_int_ip1_ref_.Read() : nullptr;
      int_kernel.ip2_ref = transport_slopes ? face_int_ip2_ref_.Read() : nullptr;
      int_kernel.z = z_inout.ReadWrite();
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
      int_kernel.state_map = invmass_elem_vdof_map_.Read();
      int_kernel.face_e1 = face_int_e1_.Read();
      int_kernel.face_e2 = face_int_e2_.Read();
      int_kernel.shape1 = face_int_shape1_.Read();
      int_kernel.shape2 = face_int_shape2_.Read();
      int_kernel.normals = face_int_normals_.Read();
      int_kernel.weights = face_int_weights_.Read();
      int_kernel.x = x_in.Read();
      int_kernel.reference_x_slopes = transport_slopes;
      int_kernel.ip1_ref = transport_slopes ? face_int_ip1_ref_.Read() : nullptr;
      int_kernel.ip2_ref = transport_slopes ? face_int_ip2_ref_.Read() : nullptr;
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
      bdr_kernel.bdr_max_attr = face_bdr_max_attr_;
      bdr_kernel.state_map = invmass_elem_vdof_map_.Read();
      bdr_kernel.face_e1 = face_bdr_e1_.Read();
      bdr_kernel.face_attr = face_bdr_attr_.Read();
      bdr_kernel.shape1 = face_bdr_shape1_.Read();
      bdr_kernel.normals = face_bdr_normals_.Read();
      bdr_kernel.weights = face_bdr_weights_.Read();
      bdr_kernel.x = x_in.Read();
      bdr_kernel.reference_x_slopes = transport_slopes;
      bdr_kernel.ip1_ref = transport_slopes ? face_bdr_ip1_ref_.Read() : nullptr;
      bdr_kernel.z = z_inout.ReadWrite();
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
      bdr_kernel.bdr_max_attr = face_bdr_max_attr_;
      bdr_kernel.state_map = invmass_elem_vdof_map_.Read();
      bdr_kernel.face_e1 = face_bdr_e1_.Read();
      bdr_kernel.face_attr = face_bdr_attr_.Read();
      bdr_kernel.shape1 = face_bdr_shape1_.Read();
      bdr_kernel.normals = face_bdr_normals_.Read();
      bdr_kernel.weights = face_bdr_weights_.Read();
      bdr_kernel.x = x_in.Read();
      bdr_kernel.reference_x_slopes = transport_slopes;
      bdr_kernel.ip1_ref = transport_slopes ? face_bdr_ip1_ref_.Read() : nullptr;
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
