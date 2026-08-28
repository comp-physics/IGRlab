/* Included inside dg::hdg::runtime::cuda_w8a. */

namespace detail
{

inline int CudaDecodeSigned(const int encoded, double& sign) noexcept
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
}

inline bool CudaEvaluateStaticExteriorStateHost(
    const CudaStaticBoundaryPolicy& policy,
    const std::array<double, conserved_components>& interior,
    const std::array<double, dimension>& normal,
    const double gamma,
    std::array<double, conserved_components>& exterior) noexcept
{
  try
  {
    mfem::EulerExteriorStateKind kind{};
    bool reference_required = false;
    switch (policy.kind)
    {
    case CudaStaticBoundaryPolicyKind::supersonic_inlet:
      kind = mfem::EulerExteriorStateKind::SupersonicInlet;
      reference_required = true;
      break;
    case CudaStaticBoundaryPolicyKind::subsonic_inlet:
      kind = mfem::EulerExteriorStateKind::SubsonicInlet;
      reference_required = true;
      break;
    case CudaStaticBoundaryPolicyKind::supersonic_outlet:
      kind = mfem::EulerExteriorStateKind::SupersonicOutlet;
      break;
    case CudaStaticBoundaryPolicyKind::subsonic_outlet_pressure:
      kind = mfem::EulerExteriorStateKind::SubsonicOutletPressure;
      break;
    case CudaStaticBoundaryPolicyKind::far_field:
      kind = mfem::EulerExteriorStateKind::FarField;
      reference_required = true;
      break;
    case CudaStaticBoundaryPolicyKind::nonreflecting_outlet:
      kind = mfem::EulerExteriorStateKind::NonReflectingOutlet;
      reference_required = true;
      break;
    case CudaStaticBoundaryPolicyKind::slip_wall:
      kind = mfem::EulerExteriorStateKind::SlipWall;
      break;
    case CudaStaticBoundaryPolicyKind::none:
    case CudaStaticBoundaryPolicyKind::subsonic_outlet:
    case CudaStaticBoundaryPolicyKind::nonreflecting:
      return false;
    }
    mfem::Vector interior_view(
        const_cast<double*>(interior.data()), conserved_components);
    mfem::Vector normal_view(const_cast<double*>(normal.data()), dimension);
    mfem::Vector exterior_view(exterior.data(), conserved_components);
    mfem::Vector reference_view(
        const_cast<double*>(policy.reference_state.data()),
        conserved_components);
    mfem::EvaluateEulerExteriorState(
        kind, interior_view, normal_view, dimension, gamma,
        reference_required ? &reference_view : nullptr,
        policy.pressure_out, exterior_view);
    return IsCudaPhysicalEulerState(exterior);
  }
  catch (...) { return false; }
}

inline void CudaManufacturedState(const double x, const double y,
                                  double* state) noexcept
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  const double rho = 1.0 + 0.1 * sx * sy;
  const double u = 0.35 + 0.05 * cx * sy;
  const double v = -0.20 + 0.05 * sx * cy;
  const double pressure = 1.0 + 0.1 * cx * cy;
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = pressure / 0.4 + 0.5 * rho * (u * u + v * v);
}

inline void CudaPerturbedState(const double x, const double y,
                               double* state) noexcept
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  double exact[conserved_components];
  CudaManufacturedState(x, y, exact);
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  const double rho0 = exact[0];
  const double u0 = exact[1] / rho0;
  const double v0 = exact[2] / rho0;
  const double p0 = 0.4 *
      (exact[3] - 0.5 * (exact[1] * exact[1] + exact[2] * exact[2]) / rho0);
  const double rho = rho0 * (1.0 + 0.04 * sx * cy);
  const double u = u0 + 0.03 * cx * sy;
  const double v = v0 - 0.03 * sx * cy;
  const double pressure = p0 * (1.0 + 0.04 *
      std::sin(2.0 * pi * (x + y)));
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = pressure / 0.4 + 0.5 * rho * (u * u + v * v);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BuildHostGeometryAndMaps(const CudaConfig& config) noexcept
{
  const bool explicit_initial_true_trace = shape_.borrowed_host_inputs &&
      config.borrowed_initial_true_trace.coefficients != nullptr;
  std::vector<double> local_exact_trace(
      static_cast<std::size_t>(shape_.local_trace_count), 0.0);
  std::vector<double> local_perturbed_trace(
      static_cast<std::size_t>(shape_.local_trace_count), 0.0);
  std::vector<unsigned char> assigned(
      static_cast<std::size_t>(shape_.local_trace_count), 0);
  std::vector<unsigned char> shared_faces(
      static_cast<std::size_t>(mesh_->GetNumFaces()), 0);
  if (shape_.borrowed_host_inputs)
  {
    for (int index = 0; index < mesh_->GetNSharedFaces(); ++index)
    {
      const int face = mesh_->GetSharedFace(index);
      if (face < 0 || face >= static_cast<int>(shared_faces.size())) return false;
      shared_faces[static_cast<std::size_t>(face)] = 1;
    }
    if (shape_.borrowed_static_boundary_policies)
    {
      std::fill(host_boundary_policy_kind_.begin(),
                host_boundary_policy_kind_.end(), 0);
      std::fill(host_boundary_reference_.begin(), host_boundary_reference_.end(),
                0.0);
      std::fill(host_boundary_pressure_.begin(), host_boundary_pressure_.end(),
                0.0);
    }
  }
  for (int element = 0; element < shape_.elements; ++element)
  {
    const mfem::FiniteElement* fe = interior_space_->GetFE(element);
    mfem::ElementTransformation* transformation =
        mesh_->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr) { return false; }
    const double element_hmin = mesh_->GetElementSize(element, 1);
    if (!std::isfinite(element_hmin) || !(element_hmin > 0.0)) return false;
    local_hmin_ = std::min(local_hmin_, element_hmin);
    mfem::Array<int> interior_vdofs;
    interior_space_->GetElementVDofs(element, interior_vdofs);
    if (interior_vdofs.Size() != shape_.interior_dofs) { return false; }
    for (int local = 0; local < shape_.interior_dofs; ++local)
      signed_element_interior_[static_cast<std::size_t>(element) *
          shape_.interior_dofs + local] = interior_vdofs[local];
    mfem::IntegrationPoint center;
    center.x = center.y = 0.5;
    mfem::Vector center_physical(dimension);
    transformation->Transform(center, center_physical);
    for (int direction = 0; direction < dimension; ++direction)
    {
      double periodic = std::fmod(center_physical[direction], 1.0);
      if (periodic < 0.0) { periodic += 1.0; }
      if (std::abs(periodic) <= 1.0e-11 ||
          std::abs(periodic - 1.0) <= 1.0e-11)
        periodic = 0.0;
      element_coordinate_keys_[static_cast<std::size_t>(element) * dimension +
                               direction] =
          static_cast<std::int64_t>(std::llround(periodic * 1.0e12));
    }
    const mfem::IntegrationRule& nodes = fe->GetNodes();
    if (nodes.GetNPoints() != shape_.scalar_interior_dofs) { return false; }
    for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
    {
      mfem::Vector physical(dimension);
      transformation->Transform(nodes.IntPoint(scalar), physical);
      double exact[conserved_components], perturbed[conserved_components];
      CudaManufacturedState(physical[0], physical[1], exact);
      CudaPerturbedState(physical[0], physical[1], perturbed);
      for (int component = 0; component < conserved_components; ++component)
      {
        const int local = scalar + shape_.scalar_interior_dofs * component;
        const std::size_t index = static_cast<std::size_t>(element) *
                                      shape_.interior_dofs + local;
        host_exact_interior_[index] = exact[component];
        host_perturbed_interior_[index] = perturbed[component];
      }
    }
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        fe->GetGeomType(), shape_.volume_assembly_order);
    mfem::Vector shape(shape_.scalar_interior_dofs), physical(dimension);
    mfem::DenseMatrix derivative(shape_.scalar_interior_dofs, dimension);
    for (int q = 0; q < shape_.volume_points; ++q)
    {
      const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      fe->CalcPhysDShape(*transformation, derivative);
      const std::size_t quadrature =
          static_cast<std::size_t>(element) * shape_.volume_points + q;
      host_volume_weight_[quadrature] =
          point.weight * transformation->Weight();
      for (int direction = 0; direction < dimension; ++direction)
        host_volume_position_[quadrature * dimension + direction] =
            physical[direction];
      for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
      {
        host_volume_shape_[quadrature * shape_.scalar_interior_dofs + scalar] =
            shape[scalar];
        for (int direction = 0; direction < dimension; ++direction)
          host_volume_derivative_[
              (quadrature * shape_.scalar_interior_dofs + scalar) * dimension +
              direction] = derivative(scalar, direction);
      }
    }

    mfem::Array<int> faces, orientations;
    mesh_->GetElementEdges(element, faces, orientations);
    if (faces.Size() != faces_per_element) { return false; }
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces_per_element; ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* ft =
          mesh_->GetFaceElementTransformations(face);
      const mfem::FiniteElement* trace_fe = trace_space_->GetFaceElement(face);
      mfem::Array<int> vdofs;
      trace_space_->GetFaceVDofs(face, vdofs);
      if (ft == nullptr || ft->Face == nullptr || trace_fe == nullptr ||
          vdofs.Size() != shape_.trace_components * shape_.scalar_face_dofs)
        return false;
      const bool element_one = ft->Elem1No == element;
      if (!element_one && ft->Elem2No != element) { return false; }
      if (shape_.borrowed_static_boundary_policies)
      {
        const int boundary_element = mesh_->GetFaceToBdrElMap()[face];
        const bool physical = shared_faces[static_cast<std::size_t>(face)] == 0 &&
            ft->Elem2No < 0 && boundary_element >= 0;
        if (physical)
        {
          const int attribute = mesh_->GetBdrAttribute(boundary_element);
          const CudaStaticBoundaryPolicy* policy = nullptr;
          for (std::size_t index = 0;
               index < config.borrowed_boundary_policies.count; ++index)
            if (config.borrowed_boundary_policies.policies[index].attribute == attribute)
              policy = config.borrowed_boundary_policies.policies + index;
          if (policy == nullptr || !IsCudaAdmittedStaticBoundaryPolicy(policy->kind))
            return false;
          const std::size_t face_index = static_cast<std::size_t>(element) *
              faces_per_element + local_face;
          host_boundary_policy_kind_[face_index] =
              static_cast<int>(policy->kind);
          host_boundary_pressure_[face_index] = policy->pressure_out;
          for (int component = 0; component < conserved_components; ++component)
            host_boundary_reference_[face_index * conserved_components + component] =
                policy->reference_state[static_cast<std::size_t>(component)];
        }
      }
      const mfem::IntegrationRule& face_nodes = trace_fe->GetNodes();
      for (int scalar = 0; scalar < shape_.scalar_face_dofs; ++scalar)
      {
        mfem::Vector face_physical(dimension);
        ft->Face->Transform(face_nodes.IntPoint(scalar), face_physical);
        double exact[conserved_components]{}, perturbed[conserved_components]{};
        if (!explicit_initial_true_trace)
        {
          CudaManufacturedState(face_physical[0], face_physical[1], exact);
          CudaPerturbedState(face_physical[0], face_physical[1], perturbed);
        }
        for (int component = 0; component < shape_.trace_components; ++component)
        {
          const int local = scalar + shape_.scalar_face_dofs * component;
          double sign = 1.0;
          const int dof = CudaDecodeSigned(vdofs[local], sign);
          if (dof < 0 || dof >= shape_.local_trace_count) { return false; }
          const std::size_t map_index = static_cast<std::size_t>(element) *
                  shape_.trace_dofs + trace_offset + local;
          signed_element_trace_[map_index] = vdofs[local];
          const double exact_value = component < conserved_components
              ? sign * exact[component] : 0.0;
          const double perturbed_value = component < conserved_components
              ? sign * perturbed[component] : 0.0;
          if (assigned[static_cast<std::size_t>(dof)] != 0 &&
              (std::abs(local_exact_trace[static_cast<std::size_t>(dof)] -
                        exact_value) > 1.0e-12 ||
               std::abs(local_perturbed_trace[static_cast<std::size_t>(dof)] -
                        perturbed_value) > 1.0e-12))
            return false;
          local_exact_trace[static_cast<std::size_t>(dof)] = exact_value;
          local_perturbed_trace[static_cast<std::size_t>(dof)] = perturbed_value;
          assigned[static_cast<std::size_t>(dof)] = 1;
        }
      }
      const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
          trace_fe->GetGeomType(), shape_.face_assembly_order);
      if (shape_.igr_active)
      {
        double face_measure = 0.0;
        for (int measure_q = 0; measure_q < face_rule.GetNPoints(); ++measure_q)
        {
          const mfem::IntegrationPoint& point = face_rule.IntPoint(measure_q);
          ft->SetAllIntPoints(&point);
          const double contribution = point.weight * ft->Face->Weight();
          if (!(contribution > 0.0) || !std::isfinite(contribution))
            return false;
          face_measure += contribution;
        }
        const double element_volume = mesh_->GetElementVolume(element);
        const double h = element_volume / face_measure;
        if (!(element_volume > 0.0) || !std::isfinite(element_volume) ||
            !(face_measure > 0.0) || !std::isfinite(face_measure) ||
            !(h > 0.0) || !std::isfinite(h))
          return false;
        host_face_h_[static_cast<std::size_t>(element) * faces_per_element +
                     local_face] = h;
      }
      mfem::Vector interior_shape(shape_.scalar_interior_dofs);
      mfem::Vector trace_shape(shape_.scalar_face_dofs);
      for (int q = 0; q < shape_.face_points; ++q)
      {
        const mfem::IntegrationPoint& point = face_rule.IntPoint(q);
        ft->SetAllIntPoints(&point);
        const mfem::IntegrationPoint& element_point = element_one
            ? ft->GetElement1IntPoint() : ft->GetElement2IntPoint();
        mfem::Vector scaled_normal(dimension);
        mfem::CalcOrtho(ft->Face->Jacobian(), scaled_normal);
        const double normal_norm = scaled_normal.Norml2();
        // Card requirement: physical face measure comes from Weight exactly
        // once; CalcOrtho supplies direction only.
        const double measure = ft->Face->Weight();
        if (!(normal_norm > 0.0) || !(measure > 0.0) ||
            !std::isfinite(measure))
          return false;
        fe->CalcShape(element_point, interior_shape);
        trace_fe->CalcShape(point, trace_shape);
        const std::size_t quadrature =
            (static_cast<std::size_t>(element) * faces_per_element +
             local_face) * shape_.face_points + q;
        host_face_weight_[quadrature] = point.weight * measure;
        const double side = element_one ? 1.0 : -1.0;
        for (int direction = 0; direction < dimension; ++direction)
          host_face_normal_[quadrature * dimension + direction] =
              side * scaled_normal[direction] / normal_norm;
        for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
          host_face_interior_shape_[
              quadrature * shape_.scalar_interior_dofs + scalar] =
              interior_shape[scalar];
        for (int scalar = 0; scalar < shape_.scalar_face_dofs; ++scalar)
          host_face_trace_shape_[quadrature * shape_.scalar_face_dofs + scalar] =
              trace_shape[scalar];
      }
      trace_offset += vdofs.Size();
    }
    if (trace_offset != shape_.trace_dofs) { return false; }
  }
  for (int local = 0; local < shape_.local_trace_count; ++local)
    if (assigned[static_cast<std::size_t>(local)] == 0) { return false; }

  std::fill(local_true_index_.begin(), local_true_index_.end(), -1);
  std::fill(owned_global_true_.begin(), owned_global_true_.end(), -1);
  for (const dg::hdg::trace::TraceMapEntry& entry :
       exchange_plan_.map.dofs.entries)
  {
    if (entry.local_dof < 0 || entry.local_dof >= shape_.local_trace_count ||
        entry.global_true_dof < 0)
      return false;
    local_global_true_[static_cast<std::size_t>(entry.local_dof)] =
        static_cast<std::int64_t>(entry.global_true_dof);
    local_owner_[static_cast<std::size_t>(entry.local_dof)] = entry.owner_rank;
    local_true_index_[static_cast<std::size_t>(entry.local_dof)] =
        entry.local_true_dof;
    if (entry.ownership == dg::hdg::Ownership::owned)
    {
      if (entry.local_true_dof < 0 ||
          entry.local_true_dof >= shape_.true_trace_count)
        return false;
      owned_global_true_[static_cast<std::size_t>(entry.local_true_dof)] =
          static_cast<std::int64_t>(entry.global_true_dof);
      if (!explicit_initial_true_trace)
      {
        host_exact_true_trace_[static_cast<std::size_t>(entry.local_true_dof)] =
            local_exact_trace[static_cast<std::size_t>(entry.local_dof)];
        host_perturbed_true_trace_[
            static_cast<std::size_t>(entry.local_true_dof)] =
            local_perturbed_trace[static_cast<std::size_t>(entry.local_dof)];
      }
    }
  }
  signed_owner_ghost_valid_ = exchange_plan_.map.dofs.IsConsistent() &&
      std::find(owned_global_true_.begin(),
                owned_global_true_.begin() + shape_.true_trace_count, -1) ==
          owned_global_true_.begin() + shape_.true_trace_count;
  return signed_owner_ghost_valid_;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
InitializeBorrowedHostFields(const CudaConfig& config) noexcept
{
  borrowed_host_inputs_ = config.use_borrowed_host_inputs;
  shape_.borrowed_static_source = config.use_borrowed_host_inputs;
  if (!config.use_borrowed_host_inputs) { return true; }

  const std::size_t scalar_count =
      static_cast<std::size_t>(interior_space_->GetVSize() /
                               shape_.interior_components);
  const std::size_t coefficient_count =
      scalar_count * static_cast<std::size_t>(conserved_components);
  const std::size_t true_trace_scalar_count =
      static_cast<std::size_t>(shape_.true_trace_count /
                               shape_.trace_components);
  const bool initial_trace_empty =
      config.borrowed_initial_true_trace.coefficients == nullptr &&
      config.borrowed_initial_true_trace.scalar_count == 0 &&
      config.borrowed_initial_true_trace.components == conserved_components;
  const bool source_empty = config.borrowed_source.coefficients == nullptr &&
      config.borrowed_source.scalar_count == 0 &&
      config.borrowed_source.components == conserved_components;
  bool local_valid = config.borrowed_initial.coefficients != nullptr &&
      config.borrowed_initial.scalar_count == scalar_count &&
      config.borrowed_initial.components == conserved_components &&
      scalar_count * static_cast<std::size_t>(shape_.interior_components) ==
          static_cast<std::size_t>(shape_.interior_count) &&
      (initial_trace_empty ||
       (config.borrowed_initial_true_trace.coefficients != nullptr &&
        config.borrowed_initial_true_trace.scalar_count ==
            true_trace_scalar_count &&
        config.borrowed_initial_true_trace.components ==
            conserved_components)) &&
      (source_empty ||
       (config.borrowed_source.coefficients != nullptr &&
        config.borrowed_source.scalar_count == scalar_count &&
        config.borrowed_source.components == conserved_components));
  if (local_valid)
  {
    for (std::size_t i = 0; i < coefficient_count; ++i)
      local_valid = std::isfinite(config.borrowed_initial.coefficients[i]) &&
          (source_empty ||
           std::isfinite(config.borrowed_source.coefficients[i]));
    const std::size_t initial_trace_count = true_trace_scalar_count *
        static_cast<std::size_t>(conserved_components);
    for (std::size_t i = 0;
         local_valid && !initial_trace_empty && i < initial_trace_count; ++i)
      local_valid =
          std::isfinite(config.borrowed_initial_true_trace.coefficients[i]);
  }
  // All ranks reject before any input is copied or a trace projection starts.
  if (!CollectiveContinue(local_valid)) { return false; }

  std::vector<double> initial(coefficient_count, 0.0);
  std::vector<double> static_source(coefficient_count, 0.0);
  std::copy_n(config.borrowed_initial.coefficients, coefficient_count,
              initial.data());
  std::fill(host_static_source_.begin(), host_static_source_.end(), 0.0);
  if (!source_empty)
    std::copy_n(config.borrowed_source.coefficients, coefficient_count,
                static_source.data());

  bool scatter_ready = true;
  for (int element = 0; scatter_ready && element < shape_.elements; ++element)
    for (int component = 0;
         scatter_ready && component < shape_.interior_components; ++component)
      for (int scalar = 0;
           scatter_ready && scalar < shape_.scalar_interior_dofs; ++scalar)
    {
      const int local = scalar + shape_.scalar_interior_dofs * component;
      double sign = 1.0;
      const int dof = CudaDecodeSigned(signed_element_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local],
          sign);
      const bool public_component = component < conserved_components;
      if (dof < 0 || dof >= shape_.interior_count ||
          (public_component &&
           static_cast<std::size_t>(dof) >= coefficient_count))
      {
        scatter_ready = false;
        continue;
      }
      const std::size_t destination = static_cast<std::size_t>(element) *
          shape_.interior_dofs + local;
      host_exact_interior_[destination] = public_component
          ? sign * initial[static_cast<std::size_t>(dof)] : 0.0;
      // Generic data mode starts the physical current state at the supplied
      // committed state; the manufactured perturbed starting state remains
      // exclusively on the legacy path.
      host_perturbed_interior_[destination] = host_exact_interior_[destination];
      if (public_component && !source_empty)
        host_static_source_[destination] = sign *
            static_source[static_cast<std::size_t>(dof)];
    }
  if (!CollectiveContinue(scatter_ready)) { return false; }

  if (!initial_trace_empty)
  {
    // An explicit public true trace is already in the generic trace ABI.
    // Copy it directly and zero only the private IGR trace component; do not
    // derive a compatibility trace or invoke physical boundary callbacks.
    std::fill(host_exact_true_trace_.begin(), host_exact_true_trace_.end(), 0.0);
    std::fill(host_perturbed_true_trace_.begin(),
              host_perturbed_true_trace_.end(), 0.0);
    const std::size_t public_count = true_trace_scalar_count *
        static_cast<std::size_t>(conserved_components);
    std::copy_n(config.borrowed_initial_true_trace.coefficients, public_count,
                host_exact_true_trace_.data());
    std::copy_n(config.borrowed_initial_true_trace.coefficients, public_count,
                host_perturbed_true_trace_.data());
    return CollectiveContinue(true);
  }

  std::optional<mfem::ParGridFunction> local_trace;
  std::vector<unsigned char> local_trace_assigned(
      static_cast<std::size_t>(shape_.local_trace_count), 0);
  bool restriction_ready = false;
  try
  {
    local_trace.emplace(trace_space_.operator->());
    if (local_trace->Size() < 0 ||
        local_trace->Size() != shape_.local_trace_count)
      throw 0;
    *local_trace = 0.0;
    // The compatibility empty-trace route has the same public semantics as
    // the CPU runtime: restrict the supplied committed interior state onto
    // every trace face.  Boundary policies are residual closures, not an
    // alternate initial-trace state.  Explicit supplied traces continue to
    // bypass this projection above.
    for (int face = 0; face < mesh_->GetNumFaces(); ++face)
    {
      mfem::FaceElementTransformations* transformations =
          mesh_->GetFaceElementTransformations(face);
      const mfem::FiniteElement* trace_element =
          trace_space_->GetFaceElement(face);
      if (transformations == nullptr || trace_element == nullptr ||
          transformations->Elem1No < 0 || transformations->Elem1No >=
              shape_.elements)
        throw 0;
      const mfem::FiniteElement* interior_element =
          interior_space_->GetFE(transformations->Elem1No);
      mfem::Array<int> interior_vdofs, trace_vdofs;
      interior_space_->GetElementVDofs(transformations->Elem1No,
                                       interior_vdofs);
      trace_space_->GetFaceVDofs(face, trace_vdofs);
      if (interior_element == nullptr ||
          interior_vdofs.Size() != shape_.interior_components *
              interior_element->GetDof() ||
          trace_vdofs.Size() != shape_.trace_components *
              trace_element->GetDof())
        throw 0;
      mfem::Vector shape(interior_element->GetDof());
      const mfem::IntegrationRule& nodes = trace_element->GetNodes();
      for (int scalar = 0; scalar < trace_element->GetDof(); ++scalar)
      {
        const mfem::IntegrationPoint& point = nodes.IntPoint(scalar);
        transformations->SetAllIntPoints(&point);
        interior_element->CalcShape(transformations->GetElement1IntPoint(),
                                    shape);
        std::array<double, conserved_components> restricted{};
        for (int component = 0; component < conserved_components; ++component)
        {
          for (int interior_scalar = 0;
               interior_scalar < interior_element->GetDof(); ++interior_scalar)
          {
            double sign = 1.0;
            const int dof = CudaDecodeSigned(interior_vdofs[
                interior_scalar + interior_element->GetDof() * component], sign);
            if (dof < 0 || static_cast<std::size_t>(dof) >= coefficient_count)
              throw 0;
            restricted[static_cast<std::size_t>(component)] +=
                shape[interior_scalar] * sign *
                initial[static_cast<std::size_t>(dof)];
          }
        }
        for (int component = 0; component < shape_.trace_components; ++component)
        {
          double sign = 1.0;
          const int dof = CudaDecodeSigned(trace_vdofs[
              scalar + trace_element->GetDof() * component], sign);
          if (dof < 0 || dof >= local_trace->Size()) { throw 0; }
          (*local_trace)[dof] = component < conserved_components
              ? sign * restricted[static_cast<std::size_t>(component)] : 0.0;
          local_trace_assigned[static_cast<std::size_t>(dof)] = 1;
        }
      }
    }
    restriction_ready = std::find(local_trace_assigned.begin(),
        local_trace_assigned.begin() + local_trace->Size(), 0) ==
            local_trace_assigned.begin() + local_trace->Size();
  }
  catch (...) { restriction_ready = false; }
  // No rank seeds true trace data until every rank has completed the local
  // restriction and map validation.
  if (!CollectiveContinue(restriction_ready)) { return false; }

  bool trace_seeded = false;
  try
  {
    if (!exchange_plan_.map.dofs.IsConsistent() ||
        shape_.true_trace_count < 0 ||
        shape_.true_trace_count < 0)
      throw 0;
    std::vector<unsigned char> true_assigned(
        static_cast<std::size_t>(shape_.true_trace_count), 0);
    for (const dg::hdg::trace::TraceMapEntry& entry :
         exchange_plan_.map.dofs.entries)
    {
      if (entry.ownership != dg::hdg::Ownership::owned) continue;
      if (entry.local_dof < 0 || entry.local_dof >= local_trace->Size() ||
          entry.local_true_dof < 0 ||
          entry.local_true_dof >= shape_.true_trace_count ||
          true_assigned[static_cast<std::size_t>(entry.local_true_dof)] != 0)
        throw 0;
      const double value = (*local_trace)[entry.local_dof];
      if (!std::isfinite(value)) { throw 0; }
      const std::size_t local_true =
          static_cast<std::size_t>(entry.local_true_dof);
      host_exact_true_trace_[local_true] = value;
      host_perturbed_true_trace_[local_true] = value;
      true_assigned[local_true] = 1;
    }
    trace_seeded = std::find(true_assigned.begin(),
        true_assigned.begin() + shape_.true_trace_count, 0) ==
            true_assigned.begin() + shape_.true_trace_count;
  }
  catch (...) { trace_seeded = false; }
  return CollectiveContinue(trace_seeded);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
ExportInterior(const CudaInteriorState& source,
               CudaMutableHostL2Output destination) noexcept
{
  const std::size_t scalar_count =
      static_cast<std::size_t>(interior_space_->GetVSize() /
                               shape_.interior_components);
  const std::size_t coefficient_count =
      scalar_count * static_cast<std::size_t>(conserved_components);
  bool local_valid = initialized_ && borrowed_host_inputs_ &&
      source.coefficients != nullptr && source.count ==
          static_cast<std::size_t>(shape_.interior_count) &&
      destination.coefficients != nullptr &&
      destination.scalar_count == scalar_count &&
      destination.components == conserved_components;
  // Validate the complete reverse scatter before the D2H transfer.  A bad
  // destination or map therefore cannot receive a prefix of a failed export.
  for (int element = 0; local_valid && element < shape_.elements; ++element)
    for (int component = 0;
         local_valid && component < conserved_components; ++component)
      for (int scalar = 0;
           local_valid && scalar < shape_.scalar_interior_dofs; ++scalar)
    {
      const int local = scalar + shape_.scalar_interior_dofs * component;
      double sign = 1.0;
      const int dof = CudaDecodeSigned(signed_element_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local],
          sign);
      local_valid = dof >= 0 &&
          static_cast<std::size_t>(dof) < coefficient_count;
    }
  if (!CollectiveContinue(local_valid)) { return false; }
  const bool physical_layout_contiguous =
      shape_.interior_components == conserved_components;
  const bool packed = physical_layout_contiguous ||
      CudaLaunchPackInteriorOutput(
          shape_, source.coefficients, view_.output_interior,
          conserved_components);
  const double* const device_export = physical_layout_contiguous
      ? source.coefficients : view_.output_interior;
  const bool copied = packed &&
      TransportPolicy::DeviceToHost(
          host_export_interior_.data(), device_export,
          coefficient_count * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export) &&
      TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
  if (!CollectiveContinue(copied)) { return false; }
  for (int element = 0; element < shape_.elements; ++element)
    for (int component = 0; component < conserved_components; ++component)
      for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
    {
      const int local = scalar + shape_.scalar_interior_dofs * component;
      double sign = 1.0;
      const int dof = CudaDecodeSigned(signed_element_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local],
          sign);
      destination.coefficients[dof] = sign * host_export_interior_[
          static_cast<std::size_t>(element) * conserved_components *
              shape_.scalar_interior_dofs + local];
    }
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
ExportFields(const CudaInteriorState& source,
             const generic::HdgEulerHostInteriorFields destination) noexcept
{
  const std::size_t scalar_count =
      static_cast<std::size_t>(interior_space_->GetVSize() /
                               shape_.interior_components);
  const bool sigma_requested = destination.sigma_coefficients != nullptr;
  const int output_components = sigma_requested
      ? igr_published_components : conserved_components;
  const std::size_t coefficient_count = scalar_count *
      static_cast<std::size_t>(output_components);
  bool local_valid = initialized_ && borrowed_host_inputs_ &&
      source.coefficients != nullptr && source.count ==
          static_cast<std::size_t>(shape_.interior_count) &&
      destination.scalar_count == scalar_count &&
      (destination.conservative_coefficients != nullptr || scalar_count == 0) &&
      (!sigma_requested || shape_.igr_active);
  // Validate the complete reverse scatter before launching or transferring,
  // so a rejected request cannot partially overwrite either caller field.
  for (int element = 0; local_valid && element < shape_.elements; ++element)
    for (int component = 0;
         local_valid && component < output_components; ++component)
      for (int scalar = 0;
           local_valid && scalar < shape_.scalar_interior_dofs; ++scalar)
    {
      const int local = scalar + shape_.scalar_interior_dofs * component;
      double sign = 1.0;
      const int dof = CudaDecodeSigned(signed_element_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local],
          sign);
      const std::size_t lower = static_cast<std::size_t>(component) *
          scalar_count;
      const std::size_t upper = lower + scalar_count;
      local_valid = dof >= 0 && static_cast<std::size_t>(dof) >= lower &&
          static_cast<std::size_t>(dof) < upper;
    }
  if (!CollectiveContinue(local_valid)) { return false; }
  const bool layout_contiguous =
      shape_.interior_components == output_components;
  const bool packed = layout_contiguous ||
      CudaLaunchPackInteriorOutput(
          shape_, source.coefficients, view_.output_interior,
          output_components);
  const double* const device_export = layout_contiguous
      ? source.coefficients : view_.output_interior;
  const bool copied = packed &&
      TransportPolicy::DeviceToHost(
          host_export_interior_.data(), device_export,
          coefficient_count * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export) &&
      TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
  if (!CollectiveContinue(copied)) { return false; }
  for (int element = 0; element < shape_.elements; ++element)
    for (int component = 0; component < output_components; ++component)
      for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
    {
      const int local = scalar + shape_.scalar_interior_dofs * component;
      double sign = 1.0;
      const int dof = CudaDecodeSigned(signed_element_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local],
          sign);
      const double value = sign * host_export_interior_[
          static_cast<std::size_t>(element) * output_components *
              shape_.scalar_interior_dofs + local];
      if (component < conserved_components)
      {
        destination.conservative_coefficients[dof] = value;
      }
      else
      {
        destination.sigma_coefficients[
            dof - static_cast<int>(conserved_components * scalar_count)] =
                value;
      }
    }
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BuildLocalTraceGraph() noexcept
{
  try
  {
    local_trace_matrix_.emplace(shape_.local_trace_count);
    for (int element = 0; element < shape_.elements; ++element)
      for (int row = 0; row < shape_.trace_dofs; ++row)
      {
        double row_sign = 1.0;
        const int local_row = CudaDecodeSigned(
            signed_element_trace_[static_cast<std::size_t>(element) *
                                      shape_.trace_dofs + row],
            row_sign);
        for (int column = 0; column < shape_.trace_dofs; ++column)
        {
          double column_sign = 1.0;
          const int local_column = CudaDecodeSigned(
              signed_element_trace_[static_cast<std::size_t>(element) *
                                        shape_.trace_dofs + column],
              column_sign);
          local_trace_matrix_->Add(local_row, local_column,
                                   row_sign * column_sign);
        }
      }
    local_trace_matrix_->Finalize();
    double* values = local_trace_matrix_->HostWriteData();
    const int* offsets = local_trace_matrix_->HostReadI();
    const int* columns = local_trace_matrix_->HostReadJ();
    for (int row = 0; row < shape_.local_trace_count; ++row)
      for (int entry = offsets[row]; entry < offsets[row + 1]; ++entry)
        values[entry] = columns[entry] == row ? 64.0 : 1.0e-3;
    local_trace_matrix_->MoveDiagonalFirst();
  }
  catch (...)
  {
    return false;
  }
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BindArenaViews() noexcept
{
  double* d = double_arena_;
  auto take_d = [&](const std::size_t count)
  {
    double* result = d;
    d += count;
    return result;
  };
  const bool generic = borrowed_topology_ && shape_.borrowed_host_inputs;
  const std::size_t max_i = generic ? generic_arena_layout_.interior :
      static_cast<std::size_t>(maximum_rank_elements) * maximum_interior_dofs_per_element;
  const std::size_t max_lt = generic ? generic_arena_layout_.local_trace :
      maximum_local_trace_dofs;
  const std::size_t max_tt = generic ? generic_arena_layout_.true_trace :
      maximum_true_trace_dofs;
  const std::size_t max_er = generic ? generic_arena_layout_.element_trace :
      static_cast<std::size_t>(maximum_rank_elements) *
      maximum_trace_dofs_per_element;
  const std::size_t max_a = generic ? generic_arena_layout_.element_interior2 :
      static_cast<std::size_t>(maximum_rank_elements) *
      maximum_interior_dofs_per_element * maximum_interior_dofs_per_element;
  const std::size_t max_b = generic ? generic_arena_layout_.element_coupling :
      static_cast<std::size_t>(maximum_rank_elements) *
      maximum_interior_dofs_per_element * maximum_trace_dofs_per_element;
  const std::size_t max_d = generic ? generic_arena_layout_.local_matrix :
      maximum_local_matrix_entries;
  const std::size_t max_exchange = generic ? max_d : maximum_exchange_entries;
  view_.slot_interior[0] = take_d(max_i);
  view_.slot_interior[1] = take_d(max_i);
  view_.old_interior = take_d(max_i);
  view_.work_interior = take_d(max_i);
  view_.static_source = take_d(max_i);
  view_.initial_current_interior = take_d(max_i);
  view_.base_interior = take_d(max_i);
  view_.output_interior = take_d(max_i);
  view_.delta_interior = take_d(max_i);
  view_.recovered_interior = take_d(max_i);
  view_.work_true_trace = take_d(max_tt);
  view_.initial_current_true_trace = take_d(max_tt);
  view_.base_true_trace = take_d(max_tt);
  view_.trial_true_trace = take_d(max_tt);
  view_.distributed_local_trace = take_d(max_lt);
  view_.current_local_trace = take_d(max_er);
  view_.base_local_trace = take_d(max_er);
  view_.trace_increment_local = take_d(max_er);
  view_.residual_r = take_d(max_i);
  view_.residual_g = take_d(max_er);
  view_.a = take_d(max_a);
  view_.b = take_d(max_b);
  view_.c = take_d(max_b);
  view_.d = take_d(max_d);
  view_.f = take_d(max_i);
  view_.g = take_d(max_er);
  view_.lu = take_d(max_a);
  view_.pivots = take_d(max_i);
  view_.solve_workspace = take_d(generic ? generic_arena_layout_.solve_workspace :
      static_cast<std::size_t>(maximum_rank_elements) *
      maximum_interior_dofs_per_element * (maximum_trace_dofs_per_element + 1));
  view_.schur = take_d(max_d);
  view_.condensed_rhs = take_d(max_er);
  view_.true_residual = take_d(max_tt);
  view_.remote_matrix_send = take_d(max_exchange);
  view_.remote_matrix_receive = take_d(max_exchange);
  view_.remote_vector_send = take_d(max_er);
  view_.remote_vector_receive = take_d(max_er);
  view_.volume_shape = take_d(generic ? generic_arena_layout_.volume_shape :
      static_cast<std::size_t>(maximum_rank_elements) * maximum_volume_points * maximum_scalar_dofs_per_element);
  view_.volume_derivative = take_d(generic ? generic_arena_layout_.volume_derivative :
      static_cast<std::size_t>(maximum_rank_elements) * maximum_volume_points * maximum_scalar_dofs_per_element * dimension);
  view_.volume_weight = take_d(generic ? generic_arena_layout_.volume_weight :
      static_cast<std::size_t>(maximum_rank_elements) * maximum_volume_points);
  view_.volume_position = take_d(generic ? generic_arena_layout_.volume_position :
      static_cast<std::size_t>(maximum_rank_elements) * maximum_volume_points * dimension);
  view_.face_interior_shape = take_d(generic ? generic_arena_layout_.face_interior :
      static_cast<std::size_t>(maximum_rank_elements) * faces_per_element * maximum_face_points * maximum_scalar_dofs_per_element);
  view_.face_trace_shape = take_d(generic ? generic_arena_layout_.face_trace :
      static_cast<std::size_t>(maximum_rank_elements) * faces_per_element * maximum_face_points * maximum_scalar_dofs_per_face);
  view_.face_weight = take_d(generic ? generic_arena_layout_.face_weight :
      static_cast<std::size_t>(maximum_rank_elements) * faces_per_element * maximum_face_points);
  view_.face_normal = take_d(generic ? generic_arena_layout_.face_normal :
      static_cast<std::size_t>(maximum_rank_elements) * faces_per_element * maximum_face_points * dimension);
  if (shape_.igr_active)
    view_.face_h = generic && shape_.igr_active
        ? take_d(generic_arena_layout_.face_h) : nullptr;
  view_.status = take_d(generic ? generic_arena_layout_.elements : maximum_rank_elements);
  view_.scalar = take_d(generic ? euler_conservation_result_scalars :
      legacy_scalar_workspace_scalars);
  // Append the borrowed-only stabilization slice after every frozen W8A
  // allocation.  False mode therefore preserves the exact legacy arena
  // partition, offsets, and resource evidence tokens.
  if (shape_.borrowed_host_inputs)
    view_.borrowed_lagged_llf_tau = take_d(generic ? generic_arena_layout_.face_weight :
        static_cast<std::size_t>(maximum_rank_elements) *
        faces_per_element * maximum_face_points);
  if (shape_.borrowed_static_boundary_policies)
  {
    view_.boundary_reference = take_d(generic ?
        generic_arena_layout_.boundary_reference : 0);
    view_.boundary_pressure = take_d(generic ?
        generic_arena_layout_.boundary_pressure : 0);
  }
  if (trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64)
  {
    native_ilu_inverse_diagonal_ =
        generic_arena_layout_.native_ilu_inverse_diagonal == 0
        ? nullptr
        : take_d(generic_arena_layout_.native_ilu_inverse_diagonal);
    native_trace_workspace_.basis =
        take_d(native_trace_requirements_.basis_doubles);
    native_trace_workspace_.basis_doubles =
        native_trace_requirements_.basis_doubles;
    native_trace_workspace_.vectors =
        take_d(native_trace_requirements_.vector_doubles);
    native_trace_workspace_.vector_doubles =
        native_trace_requirements_.vector_doubles;
    native_trace_workspace_.reduction =
        take_d(native_trace_requirements_.reduction_doubles);
    native_trace_workspace_.reduction_doubles =
        native_trace_requirements_.reduction_doubles;
    native_trace_workspace_.small =
        take_d(native_trace_requirements_.small_doubles);
    native_trace_workspace_.small_doubles =
        native_trace_requirements_.small_doubles;
    native_trace_workspace_.device_control =
        take_d(native_trace_requirements_.control_double_slots);
  }
  if (static_cast<std::size_t>(d - double_arena_) >
      double_arena_capacity_)
    return false;

  int* i = integer_arena_;
  // The adjacent hash and ticket slices also provide one contiguous 64-byte
  // result for the paired accepted-state hash after the last execution ticket.
  view_.hash_words = reinterpret_cast<std::uint64_t*>(i);
  i += 8;
  view_.ticket_words = i;
  i += 8;
  view_.status_flag = i;
  i += 1;
  view_.signed_element_trace = TakeArena(i, max_er);
  if (shape_.borrowed_static_boundary_policies)
    view_.boundary_policy_kind = TakeArena(i, generic ?
        generic_arena_layout_.element_faces : 0);
  view_.matrix_owned_source = TakeArena(i, max_d);
  view_.matrix_owned_section = TakeArena(i, max_d);
  view_.matrix_owned_target = TakeArena(i, max_d);
  view_.matrix_send_source = TakeArena(i, max_exchange);
  view_.matrix_receive_section = TakeArena(i, max_exchange);
  view_.matrix_receive_target = TakeArena(i, max_exchange);
  view_.vector_owned_source = TakeArena(i, max_er);
  view_.vector_owned_target = TakeArena(i, max_er);
  view_.vector_send_source = TakeArena(i, max_er);
  view_.vector_receive_target = TakeArena(i, max_er);
  if (trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64)
  {
    native_trace_workspace_.permutation_marks =
        native_trace_requirements_.permutation_integers == 0
        ? nullptr
        : TakeArena(i, native_trace_requirements_.permutation_integers);
    native_trace_workspace_.permutation_integers =
        native_trace_requirements_.permutation_integers;
  }
  if (static_cast<std::size_t>(i - integer_arena_) >
      integer_arena_capacity_)
    return false;

  double_memory_.Wrap(double_arena_,
                      static_cast<int>(double_arena_capacity_),
                      mfem::MemoryType::DEVICE_UMPIRE, false);
  integer_memory_.Wrap(integer_arena_,
                       static_cast<int>(integer_arena_capacity_),
                       mfem::MemoryType::DEVICE_UMPIRE, false);
  double_arena_vector_.NewMemoryAndSize(
      double_memory_, static_cast<int>(double_arena_capacity_), false);
  double_arena_vector_.UseDevice(true);
  return double_memory_.DeviceIsValid() && integer_memory_.DeviceIsValid();
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
UploadTopologyAndState() noexcept
{
  const auto h2d = [](void* destination, const void* source,
                      const std::size_t bytes, CudaTransferSite site)
  {
    return TransportPolicy::HostToDevice(destination, source, bytes,
        CudaTransferCategory::topology, site);
  };
  const std::size_t interior_bytes =
      static_cast<std::size_t>(shape_.interior_count) * sizeof(double);
  const bool static_source_uploaded = !shape_.borrowed_static_source ||
      TransportPolicy::HostToDevice(
          view_.static_source, host_static_source_.data(), interior_bytes,
          CudaTransferCategory::initial_state,
          CudaTransferSite::initial_interior_upload);
  if (!TransportPolicy::HostToDevice(
          view_.slot_interior[0], host_exact_interior_.data(), interior_bytes,
          CudaTransferCategory::initial_state,
          CudaTransferSite::initial_interior_upload) ||
      !TransportPolicy::HostToDevice(
          view_.slot_interior[1], host_exact_interior_.data(), interior_bytes,
          CudaTransferCategory::initial_state,
          CudaTransferSite::initial_interior_upload) ||
      !TransportPolicy::HostToDevice(
          view_.initial_current_interior, host_perturbed_interior_.data(),
          interior_bytes, CudaTransferCategory::initial_state,
          CudaTransferSite::initial_interior_upload) ||
      !static_source_uploaded ||
      !TransportPolicy::HostToDevice(
          view_.initial_current_true_trace,
          host_perturbed_true_trace_.data(),
          static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double),
          CudaTransferCategory::initial_state,
          CudaTransferSite::initial_trace_upload) ||
      !h2d(view_.signed_element_trace, signed_element_trace_.data(),
           static_cast<std::size_t>(shape_.elements) *
               shape_.trace_dofs * sizeof(int),
           CudaTransferSite::map_upload) ||
      !h2d(view_.volume_shape, host_volume_shape_.data(),
           static_cast<std::size_t>(shape_.elements) * shape_.volume_points *
               shape_.scalar_interior_dofs * sizeof(double),
           CudaTransferSite::quadrature_upload) ||
      !h2d(view_.volume_derivative, host_volume_derivative_.data(),
           static_cast<std::size_t>(shape_.elements) * shape_.volume_points *
               shape_.scalar_interior_dofs * dimension * sizeof(double),
           CudaTransferSite::quadrature_upload) ||
      !h2d(view_.volume_weight, host_volume_weight_.data(),
           static_cast<std::size_t>(shape_.elements) * shape_.volume_points *
               sizeof(double), CudaTransferSite::quadrature_upload) ||
      !h2d(view_.volume_position, host_volume_position_.data(),
           static_cast<std::size_t>(shape_.elements) * shape_.volume_points *
               dimension * sizeof(double),
           CudaTransferSite::quadrature_upload) ||
      !h2d(view_.face_interior_shape, host_face_interior_shape_.data(),
           static_cast<std::size_t>(shape_.elements) * faces_per_element *
               shape_.face_points * shape_.scalar_interior_dofs *
                   sizeof(double),
           CudaTransferSite::quadrature_upload) ||
      !h2d(view_.face_trace_shape, host_face_trace_shape_.data(),
           static_cast<std::size_t>(shape_.elements) * faces_per_element *
               shape_.face_points * shape_.scalar_face_dofs * sizeof(double),
           CudaTransferSite::quadrature_upload) ||
      !h2d(view_.face_weight, host_face_weight_.data(),
           static_cast<std::size_t>(shape_.elements) * faces_per_element *
               shape_.face_points * sizeof(double),
           CudaTransferSite::quadrature_upload) ||
      !h2d(view_.face_normal, host_face_normal_.data(),
           static_cast<std::size_t>(shape_.elements) * faces_per_element *
               shape_.face_points * dimension * sizeof(double),
           CudaTransferSite::quadrature_upload))
    return false;
  if (shape_.igr_active &&
      !h2d(view_.face_h, host_face_h_.data(),
           static_cast<std::size_t>(shape_.elements) * faces_per_element *
               sizeof(double), CudaTransferSite::quadrature_upload))
    return false;
  if (shape_.borrowed_static_boundary_policies &&
      (!h2d(view_.boundary_reference, host_boundary_reference_.data(),
            static_cast<std::size_t>(shape_.elements) * faces_per_element *
                conserved_components * sizeof(double),
            CudaTransferSite::quadrature_upload) ||
       !h2d(view_.boundary_pressure, host_boundary_pressure_.data(),
            static_cast<std::size_t>(shape_.elements) * faces_per_element *
                sizeof(double), CudaTransferSite::quadrature_upload) ||
       !h2d(view_.boundary_policy_kind, host_boundary_policy_kind_.data(),
            static_cast<std::size_t>(shape_.elements) * faces_per_element *
                sizeof(int), CudaTransferSite::map_upload)))
    return false;
  return TransportPolicy::Synchronize(CudaTransferSite::topology_upload);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BuildPersistentTrueSystem() noexcept
{
  const CudaTracePolicyDescriptor& trace_policy = trace_policy_descriptor_;
  if (!CudaTracePolicyDescriptorIsExact(trace_policy) ||
      trace_policy.igr_active != shape_.igr_active)
    return false;
  const CudaTracePreconditionerKind selected_preconditioner =
      trace_policy.trace_preconditioner;
  const bool use_ilu =
      CudaTracePreconditionerIsIlu(selected_preconditioner);
  const bool use_native = trace_solver_kind_ ==
      CudaTraceSolverKind::native_rank1_fp64;
  if (!IsKnownCudaTraceSolverKind(trace_solver_kind_) ||
      (use_native && (ranks_ != 1 || !borrowed_topology_ ||
                      !shape_.borrowed_host_inputs)))
    return false;
  initialization_tensor_core_evidence_ = {};
  initialization_tensor_core_evidence_.selected_engine = cuda_engine_;
  initialization_tensor_core_evidence_.trace_policy = trace_policy.selection;
  initialization_tensor_core_evidence_.trace_preconditioner =
      selected_preconditioner;
  initialization_tensor_core_evidence_.native_rank1_trace_solver_configured =
      use_native;
  initialization_tensor_core_evidence_.
      hypre_host_staged_rank2_trace_solver_configured =
          trace_solver_kind_ ==
              CudaTraceSolverKind::hypre_host_staged_rank2;
  initialization_tensor_core_evidence_.trace_solver_accounting_complete =
      true;
  // Native IGR keeps one allocator-prime ILU factor alive while the
  // production ILU object materializes its own fixed-shape factor below.
  // This bounded overlap establishes the recurring HYPRE/Umpire replacement
  // footprint before the allocation baseline.  It is initialization work,
  // not an accepted-step allocation allowance.
  std::optional<mfem::HypreILU> native_allocator_prime_ilu;
  // Aggregate work evidence is qualification-only.  The independently
  // verified HYPRE configuration below remains the setup authority if
  // bookkeeping cannot be initialized.
  (void)CudaInitializeIluEvidence(
      initialization_tensor_core_evidence_, selected_preconditioner,
      shape_.igr_active);
  bool local_objects_ready = false;
  try
  {
    mfem::ParBilinearForm form(&*trace_space_);
    true_matrix_.reset(form.ParallelAssemble(&*local_trace_matrix_));
    local_objects_ready = true_matrix_ &&
        CudaGetTrueMatrixStructure(*true_matrix_, owned_global_true_.data(),
                                   shape_.true_trace_count, true_structure_,
                                   &TransportPolicy::DeviceToHost,
                                   &TransportPolicy::Synchronize) &&
        CudaGetTrueMatrixDeviceView(*true_matrix_, true_device_);
    if (local_objects_ready && use_native)
    {
      bool diagonal_first = true_structure_.rows == shape_.true_trace_count &&
          true_device_.rows == shape_.true_trace_count &&
          true_device_.diag_columns_count == shape_.true_trace_count &&
          true_structure_.offd_nonzeros == 0 &&
          true_device_.offd_nonzeros == 0 &&
          true_structure_.diag_offsets.size() ==
              static_cast<std::size_t>(shape_.true_trace_count) + 1 &&
          true_structure_.diag_columns.size() ==
              static_cast<std::size_t>(true_structure_.diag_nonzeros);
      for (int row = 0; diagonal_first && row < shape_.true_trace_count;
           ++row)
      {
        const int begin = true_structure_.diag_offsets[
            static_cast<std::size_t>(row)];
        const int end = true_structure_.diag_offsets[
            static_cast<std::size_t>(row + 1)];
        diagonal_first = begin >= 0 && begin < end &&
            end <= true_structure_.diag_nonzeros &&
            true_structure_.diag_columns[
                static_cast<std::size_t>(begin)] == row;
      }
      local_objects_ready = diagonal_first;
    }
    if (local_objects_ready && oracle_capture_requested_)
    {
      local_objects_ready = true_structure_.diag_nonzeros >= 0 &&
          true_structure_.offd_nonzeros >= 0 &&
          static_cast<std::size_t>(true_structure_.diag_nonzeros) <=
              maximum_true_matrix_entries &&
          static_cast<std::size_t>(true_structure_.offd_nonzeros) <=
              maximum_true_matrix_entries;
      if (local_objects_ready)
      {
        host_true_diag_values_.assign(
            static_cast<std::size_t>(true_structure_.diag_nonzeros), 0.0);
        host_true_offd_values_.assign(
            static_cast<std::size_t>(true_structure_.offd_nonzeros), 0.0);
      }
    }
    if (local_objects_ready)
    {
      trace_slots_[0].emplace(trace_space_.operator->());
      trace_slots_[1].emplace(trace_space_.operator->());
      hypre_rhs_.emplace(*true_matrix_);
      hypre_solution_.emplace(*true_matrix_);
      hypre_action_.emplace(*true_matrix_);
      if (use_ilu)
      {
        if (use_native)
          native_allocator_prime_ilu.emplace();
        else
          ilu_.emplace();
      }
      else if (!use_native)
        diagonal_.emplace();
      if (!use_native)
        gmres_.emplace(*true_matrix_);
      if (!use_native && use_ilu)
      {
        CudaHypreSetupState initial_setup{};
        local_objects_ready = CudaConfigurePersistentHypre(
            *true_matrix_, *ilu_, *gmres_, ranks_, trace_policy,
            final_coherent_linear_relative_tolerance_,
            maximum_linear_iterations_,
            krylov_dimension_, 0, initial_setup) &&
            initial_setup.ConfigurationExactForGeneration(
                0, trace_policy);
      }
      else if (!use_native)
      {
        CudaDiagScaleSetupState initial_setup{};
        local_objects_ready = CudaConfigurePersistentDiagonal(
            *true_matrix_, *diagonal_, *gmres_,
            trace_policy,
            final_coherent_linear_relative_tolerance_,
            maximum_linear_iterations_,
            krylov_dimension_, 0, initial_setup) &&
            initial_setup.ConfigurationExactForGeneration(
                0, trace_policy);
      }
    }
    if (local_objects_ready)
    {
      for (auto& slot : trace_slots_)
      {
        slot->UseDevice(true);
        slot->HypreReadWrite();
        double* data = static_cast<mfem::Vector&>(*slot).ReadWrite();
        local_objects_ready = local_objects_ready &&
            TransportPolicy::HostToDevice(
                data, host_exact_true_trace_.data(),
                static_cast<std::size_t>(shape_.true_trace_count) *
                    sizeof(double),
                CudaTransferCategory::initial_state,
                CudaTransferSite::initial_trace_upload);
      }
    }
    if (local_objects_ready)
    {
      hypre_rhs_->UseDevice(true);
      hypre_solution_->UseDevice(true);
      hypre_action_->UseDevice(true);
      hypre_rhs_->HypreReadWrite();
      hypre_solution_->HypreReadWrite();
      hypre_action_->HypreReadWrite();
      view_.true_rhs_scratch =
          static_cast<mfem::Vector&>(*hypre_rhs_).ReadWrite();
      view_.true_solution =
          static_cast<mfem::Vector&>(*hypre_solution_).ReadWrite();
      owned_stage_.SetSize(shape_.true_trace_count);
      local_stage_.SetSize(shape_.local_trace_count);
    }
  }
  catch (...)
  {
    local_objects_ready = false;
  }
  if (!CollectiveContinue(local_objects_ready)) { return false; }
  const bool staging_ready = BuildSparseStagingPlan() &&
      TransportPolicy::Synchronize(CudaTransferSite::initial_trace_upload);
  if (!CollectiveContinue(staging_ready))
    return false;
  // Materialize the exact Krylov/factor allocation footprint before the
  // allocation baseline. The topology matrix is deliberately diagonally
  // dominant here; each real generation overwrites only its already-fixed
  // numerical values. Ordinary and rank-two warm objects are retired below.
  // Native IGR instead keeps this setup-only factor live until the production
  // factor has been built, so the pool baseline includes the exact bounded
  // replacement overlap observed on a later fixed-shape refresh. A
  // deterministic nonzero RHS forces every executed warm solve to
  // materialize its persistent workspace instead of accepting an identity
  // no-op.
  *hypre_rhs_ = 1.0;
  *hypre_solution_ = 0.0;
  CudaSolveDiagnostics warm{};
  bool warm_solved = false;
  bool native_allocator_prime_ready = false;
  if (use_native)
  {
    CudaNativeIlu0FactorView warm_factor{};
    CudaIluAttemptFactorState warm_factor_state{};
    const AttemptIdentity warm_attempt{};
    bool warm_factor_ready = !use_ilu;
    if (use_ilu)
    {
      warm_factor_ready = native_allocator_prime_ilu &&
          CudaBeginIluFactorAttempt(
              warm_factor_state, warm_attempt,
              trace_policy.ilu_factor_reuse_policy,
              reinterpret_cast<std::uintptr_t>(true_matrix_.get()),
              reinterpret_cast<std::uintptr_t>(true_device_.diag_values),
              reinterpret_cast<std::uintptr_t>(true_device_.offd_values),
              true_device_.diag_nonzeros, true_device_.offd_nonzeros) &&
          CudaPrepareNativeRankOneIluFactor(
              *true_matrix_, *hypre_rhs_, *hypre_solution_,
              *native_allocator_prime_ilu,
              trace_policy, warm_attempt, 0,
              native_ilu_inverse_diagonal_,
              generic_arena_layout_.native_ilu_inverse_diagonal,
              warm_factor_state, warm_factor, warm);
      native_allocator_prime_ready = warm_factor_ready &&
          warm.ilu_factor_setups == 1 && warm.ilu_factor_reuses == 0;
    }
    warm_solved = use_ilu ? native_allocator_prime_ready :
        warm_factor_ready &&
            LaunchAndObserveNativeRankOneTraceSolve(
                warm_factor, 0, final_coherent_linear_relative_tolerance_,
                warm);
  }
  else
  {
    warm_solved = use_ilu
        ? CudaRefreshFactorAndSolve(
              *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_, *gmres_,
              ranks_, communicator_, trace_policy,
              final_coherent_linear_relative_tolerance_,
              maximum_linear_iterations_,
              krylov_dimension_, 0, warm)
        : CudaRefreshDiagonalAndSolve(
              *true_matrix_, *hypre_rhs_, *hypre_solution_, *diagonal_,
              *gmres_, ranks_, communicator_, trace_policy,
              final_coherent_linear_relative_tolerance_,
              maximum_linear_iterations_, krylov_dimension_, 0, warm);
  }
  if (use_ilu && warm_solved && !use_native)
    (void)CudaAccumulateIluSolveEvidence(
        initialization_tensor_core_evidence_, warm);
  if (warm_solved && !(use_native && use_ilu))
  {
    if (use_native)
      initialization_tensor_core_evidence_.
          native_rank1_trace_solve_calls = 1;
    else
      initialization_tensor_core_evidence_.hypre_trace_solve_calls = 1;
  }
  const bool warmed = use_native && use_ilu
      ? native_allocator_prime_ready
      : warm_solved && warm.device_policy &&
          warm.configured_preconditioner && warm.finite &&
          warm.matrix_generation == 0 &&
          warm.trace_solver == trace_solver_kind_ &&
          warm.configured_native_solver == use_native;
  if (!CollectiveContinue(warmed)) { return false; }

  bool warm_state_zeroed = false;
  try
  {
    *hypre_rhs_ = 0.0;
    *hypre_solution_ = 0.0;
    warm_state_zeroed = true;
  }
  catch (...) { warm_state_zeroed = false; }
  ReportCudaWarmLifetimeMarker(
      rank_, "warm-synchronization", "before", false);
  const bool warm_synchronized = warm_state_zeroed &&
      TransportPolicy::Synchronize(CudaTransferSite::initial_trace_upload);
  ReportCudaWarmLifetimeMarker(
      rank_, "warm-synchronization", "after", warm_synchronized);
  if (!CollectiveContinue(warm_synchronized)) { return false; }

  ReportCudaWarmLifetimeMarker(
      rank_, "gmres-retirement", "before", false);
  gmres_.reset();
  ReportCudaWarmLifetimeMarker(
      rank_, "gmres-retirement", "after", !gmres_);
  ReportCudaWarmLifetimeMarker(
      rank_, "trace-preconditioner-retirement", "before", false);
  ilu_.reset();
  diagonal_.reset();
  trace_preconditioner_ = CudaTracePreconditionerKind::unconfigured;
  ReportCudaWarmLifetimeMarker(
      rank_, "trace-preconditioner-retirement", "after",
      !ilu_ && !diagonal_);

  bool persistent_configured = false;
  ReportCudaWarmLifetimeMarker(
      rank_, "persistent-recreation-configuration", "before", false);
  try
  {
    if (use_native)
    {
      if (use_ilu)
        ilu_.emplace();
      persistent_configured = !gmres_ && !diagonal_ &&
          (use_ilu == static_cast<bool>(ilu_)) &&
          native_trace_requirements_.valid &&
          native_trace_support_.supported &&
          native_trace_workspace_.basis != nullptr &&
          native_trace_workspace_.vectors != nullptr &&
          native_trace_workspace_.reduction != nullptr &&
          native_trace_workspace_.small != nullptr &&
          native_trace_workspace_.device_control != nullptr &&
          native_trace_workspace_.basis_doubles ==
              native_trace_requirements_.basis_doubles &&
          native_trace_workspace_.vector_doubles ==
              native_trace_requirements_.vector_doubles &&
          native_trace_workspace_.reduction_doubles ==
              native_trace_requirements_.reduction_doubles &&
          native_trace_workspace_.small_doubles ==
              native_trace_requirements_.small_doubles &&
          native_trace_workspace_.permutation_integers ==
              native_trace_requirements_.permutation_integers &&
          (!use_ilu ||
           (native_ilu_inverse_diagonal_ != nullptr &&
            generic_arena_layout_.native_ilu_inverse_diagonal ==
                static_cast<std::size_t>(shape_.true_trace_count) &&
            native_trace_workspace_.permutation_marks != nullptr)) &&
          (use_ilu ||
           (native_ilu_inverse_diagonal_ == nullptr &&
            native_trace_workspace_.permutation_marks == nullptr));
      if (persistent_configured && use_ilu)
      {
        // Build and validate the production factor while the allocator-prime
        // factor is still live.  The persistent object remains factored; the
        // prime object is destroyed only after this synchronized setup.
        *hypre_rhs_ = 1.0;
        *hypre_solution_ = 0.0;
        CudaNativeIlu0FactorView persistent_warm_factor{};
        CudaIluAttemptFactorState persistent_warm_factor_state{};
        const AttemptIdentity persistent_warm_attempt{};
        CudaSolveDiagnostics persistent_warm{};
        persistent_configured = CudaBeginIluFactorAttempt(
            persistent_warm_factor_state, persistent_warm_attempt,
            trace_policy.ilu_factor_reuse_policy,
            reinterpret_cast<std::uintptr_t>(true_matrix_.get()),
            reinterpret_cast<std::uintptr_t>(true_device_.diag_values),
            reinterpret_cast<std::uintptr_t>(true_device_.offd_values),
            true_device_.diag_nonzeros, true_device_.offd_nonzeros) &&
            CudaPrepareNativeRankOneIluFactor(
                *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_,
                trace_policy, persistent_warm_attempt, 0,
                native_ilu_inverse_diagonal_,
                generic_arena_layout_.native_ilu_inverse_diagonal,
                persistent_warm_factor_state, persistent_warm_factor,
                persistent_warm) &&
            LaunchAndObserveNativeRankOneTraceSolve(
                persistent_warm_factor, 0,
                final_coherent_linear_relative_tolerance_,
                persistent_warm) &&
            persistent_warm.device_policy &&
            persistent_warm.configured_preconditioner &&
            persistent_warm.finite &&
            persistent_warm.matrix_generation == 0 &&
            persistent_warm.trace_solver == trace_solver_kind_ &&
            persistent_warm.configured_native_solver;
        if (persistent_configured)
        {
          const bool persistent_warm_accounted =
              CudaAccumulateIluSolveEvidence(
                  initialization_tensor_core_evidence_, persistent_warm) &&
              initialization_tensor_core_evidence_.ilu_factor_setups !=
                  std::numeric_limits<std::uint64_t>::max();
          if (persistent_warm_accounted)
          {
            // The allocator-prime factor was a setup-only authority.  Count
            // it explicitly; only the persistent factor ran a trace solve.
            ++initialization_tensor_core_evidence_.ilu_factor_setups;
          }
          else
            initialization_tensor_core_evidence_.ilu_accounting_complete =
                false;
          initialization_tensor_core_evidence_.
              native_rank1_trace_solve_calls = 1;
        }
        bool persistent_warm_zeroed = false;
        try
        {
          *hypre_rhs_ = 0.0;
          *hypre_solution_ = 0.0;
          persistent_warm_zeroed = true;
        }
        catch (...) { persistent_warm_zeroed = false; }
        persistent_configured = persistent_configured &&
            persistent_warm_zeroed &&
            TransportPolicy::Synchronize(
                CudaTransferSite::initial_trace_upload);
      }
    }
    else
    {
      if (use_ilu)
        ilu_.emplace();
      else
        diagonal_.emplace();
      gmres_.emplace(*true_matrix_);
      if (use_ilu)
      {
        CudaHypreSetupState persistent_setup{};
        persistent_configured = CudaConfigurePersistentHypre(
            *true_matrix_, *ilu_, *gmres_, ranks_, trace_policy,
            final_coherent_linear_relative_tolerance_,
            maximum_linear_iterations_,
            krylov_dimension_, 0, persistent_setup) &&
            persistent_setup.ConfigurationExactForGeneration(
                0, trace_policy);
      }
      else
      {
        CudaDiagScaleSetupState persistent_setup{};
        persistent_configured = CudaConfigurePersistentDiagonal(
            *true_matrix_, *diagonal_, *gmres_, trace_policy,
            final_coherent_linear_relative_tolerance_,
            maximum_linear_iterations_, krylov_dimension_, 0,
            persistent_setup) &&
            persistent_setup.ConfigurationExactForGeneration(
                0, trace_policy);
      }
    }
  }
  catch (...) { persistent_configured = false; }
  if (persistent_configured)
  {
    trace_preconditioner_ = selected_preconditioner;
    initialization_tensor_core_evidence_.trace_preconditioner_configured =
        true;
  }
  ReportCudaWarmLifetimeMarker(
      rank_, "persistent-recreation-configuration", "after",
      persistent_configured);
  return CollectiveContinue(persistent_configured);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BuildSparseStagingPlan() noexcept
{
  const auto find_row = [&](const std::int64_t global)
  {
    for (int row = 0; row < shape_.true_trace_count; ++row)
      if (owned_global_true_[static_cast<std::size_t>(row)] == global)
        return row;
    return -1;
  };
  const auto find_target = [&](const std::int64_t row_global,
                               const std::int64_t column_global,
                               int& section, int& target)
  {
    const int row = find_row(row_global);
    if (row < 0) { return false; }
    for (int entry = true_structure_.diag_offsets[static_cast<std::size_t>(row)];
         entry < true_structure_.diag_offsets[static_cast<std::size_t>(row + 1)];
         ++entry)
    {
      const int column = true_structure_.diag_columns[static_cast<std::size_t>(entry)];
      if (column >= 0 && column < shape_.true_trace_count &&
          owned_global_true_[static_cast<std::size_t>(column)] == column_global)
      {
        section = 0;
        target = entry;
        return true;
      }
    }
    for (int entry = true_structure_.offd_offsets[static_cast<std::size_t>(row)];
         entry < true_structure_.offd_offsets[static_cast<std::size_t>(row + 1)];
         ++entry)
      if (true_structure_.offd_columns[static_cast<std::size_t>(entry)] ==
          column_global)
      {
        section = 1;
        target = entry;
        return true;
      }
    return false;
  };

  bool local_plan_valid = true;
  for (int element = 0; element < shape_.elements; ++element)
    for (int row = 0; row < shape_.trace_dofs; ++row)
    {
      double row_sign = 1.0;
      const int row_local = CudaDecodeSigned(
          signed_element_trace_[static_cast<std::size_t>(element) *
                                    shape_.trace_dofs + row], row_sign);
      const std::int64_t row_global =
          local_global_true_[static_cast<std::size_t>(row_local)];
      const int row_owner = local_owner_[static_cast<std::size_t>(row_local)];
      const int vector_source = element * shape_.trace_dofs + row;
      if (row_owner == rank_)
      {
        const int true_row = find_row(row_global);
        if (true_row < 0)
        {
          local_plan_valid = false;
        }
        else
        {
          host_vector_owned_source_[static_cast<std::size_t>(
              shape_.vector_owned_count)] = vector_source;
          host_vector_owned_target_[static_cast<std::size_t>(
              shape_.vector_owned_count)] = true_row;
          ++shape_.vector_owned_count;
        }
      }
      else
      {
        host_vector_send_source_[static_cast<std::size_t>(
            shape_.vector_send_count)] = vector_source;
        vector_send_keys_[static_cast<std::size_t>(shape_.vector_send_count)] =
            row_global;
        ++shape_.vector_send_count;
      }
      for (int column = 0; column < shape_.trace_dofs; ++column)
      {
        double column_sign = 1.0;
        const int column_local = CudaDecodeSigned(
            signed_element_trace_[static_cast<std::size_t>(element) *
                                      shape_.trace_dofs + column],
            column_sign);
        const std::int64_t column_global =
            local_global_true_[static_cast<std::size_t>(column_local)];
        const int source = (element * shape_.trace_dofs + row) *
                               shape_.trace_dofs + column;
        if (row_owner == rank_)
        {
          int section = -1, target = -1;
          if (!find_target(row_global, column_global, section, target))
          {
            local_plan_valid = false;
          }
          else
          {
            host_matrix_owned_source_[static_cast<std::size_t>(
                shape_.matrix_owned_count)] = source;
            host_matrix_owned_section_[static_cast<std::size_t>(
                shape_.matrix_owned_count)] = section;
            host_matrix_owned_target_[static_cast<std::size_t>(
                shape_.matrix_owned_count)] = target;
            ++shape_.matrix_owned_count;
          }
        }
        else
        {
          host_matrix_send_source_[static_cast<std::size_t>(
              shape_.matrix_send_count)] = source;
          matrix_send_keys_[2 * static_cast<std::size_t>(shape_.matrix_send_count)] =
              row_global;
          matrix_send_keys_[2 * static_cast<std::size_t>(shape_.matrix_send_count) + 1] =
              column_global;
          ++shape_.matrix_send_count;
        }
      }
    }

  if (!CollectiveContinue(local_plan_valid)) { return false; }
  if (ranks_ == 2)
  {
    const int peer = 1 - rank_;
    const bool matrix_count_ok = MPI_Sendrecv(
                     &shape_.matrix_send_count, 1, MPI_INT, peer, 4101,
                     &shape_.matrix_receive_count, 1, MPI_INT, peer, 4101,
                     communicator_, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    const bool matrix_count_recorded = TransportPolicy::RecordMpiStaging(
            2 * sizeof(int), CudaTransferCategory::topology,
            CudaTransferSite::map_upload, &shape_.matrix_send_count,
            &shape_.matrix_receive_count, sizeof(int), sizeof(int),
            matrix_count_ok);
    if (!CollectiveContinue(matrix_count_ok && matrix_count_recorded))
      return false;
    const bool vector_count_ok = MPI_Sendrecv(
                     &shape_.vector_send_count, 1, MPI_INT, peer, 4102,
                     &shape_.vector_receive_count, 1, MPI_INT, peer, 4102,
                     communicator_, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    const bool vector_count_recorded = TransportPolicy::RecordMpiStaging(
            2 * sizeof(int), CudaTransferCategory::topology,
            CudaTransferSite::map_upload, &shape_.vector_send_count,
            &shape_.vector_receive_count, sizeof(int), sizeof(int),
            vector_count_ok);
    if (!CollectiveContinue(vector_count_ok && vector_count_recorded))
      return false;
    const bool counts_valid = shape_.matrix_receive_count >= 0 &&
        shape_.vector_receive_count >= 0 &&
        static_cast<std::size_t>(shape_.matrix_receive_count) <=
            host_matrix_receive_section_.size() &&
        static_cast<std::size_t>(shape_.vector_receive_count) <=
            host_vector_receive_target_.size();
    if (!CollectiveContinue(counts_valid))
      return false;
    const bool matrix_keys_ok = MPI_Sendrecv(
                     matrix_send_keys_.data(), 2 * shape_.matrix_send_count,
                     MPI_INT64_T, peer, 4103,
                     matrix_receive_keys_.data(),
                     2 * shape_.matrix_receive_count, MPI_INT64_T,
                     peer, 4103, communicator_,
                     MPI_STATUS_IGNORE) == MPI_SUCCESS;
    const std::size_t matrix_key_bytes = 2 * static_cast<std::size_t>(
        shape_.matrix_send_count + shape_.matrix_receive_count) *
        sizeof(std::int64_t);
    const bool matrix_keys_recorded = TransportPolicy::RecordMpiStaging(
            matrix_key_bytes, CudaTransferCategory::topology,
            CudaTransferSite::map_upload, matrix_send_keys_.data(),
            matrix_receive_keys_.data(),
            matrix_send_keys_.size() * sizeof(std::int64_t),
            matrix_receive_keys_.size() * sizeof(std::int64_t),
            matrix_keys_ok);
    if (!CollectiveContinue(matrix_keys_ok && matrix_keys_recorded))
      return false;
    const bool vector_keys_ok = MPI_Sendrecv(
                     vector_send_keys_.data(), shape_.vector_send_count,
                     MPI_INT64_T, peer, 4104,
                     vector_receive_keys_.data(), shape_.vector_receive_count,
                     MPI_INT64_T, peer, 4104, communicator_,
                     MPI_STATUS_IGNORE) == MPI_SUCCESS;
    const std::size_t vector_key_bytes = static_cast<std::size_t>(
        shape_.vector_send_count + shape_.vector_receive_count) *
        sizeof(std::int64_t);
    const bool vector_keys_recorded = TransportPolicy::RecordMpiStaging(
            vector_key_bytes, CudaTransferCategory::topology,
            CudaTransferSite::map_upload, vector_send_keys_.data(),
            vector_receive_keys_.data(),
            vector_send_keys_.size() * sizeof(std::int64_t),
            vector_receive_keys_.size() * sizeof(std::int64_t),
            vector_keys_ok);
    if (!CollectiveContinue(vector_keys_ok && vector_keys_recorded))
      return false;
    bool local_receive_valid = true;
    for (int entry = 0; entry < shape_.matrix_receive_count; ++entry)
      local_receive_valid = find_target(
          matrix_receive_keys_[2 * static_cast<std::size_t>(entry)],
          matrix_receive_keys_[2 * static_cast<std::size_t>(entry) + 1],
          host_matrix_receive_section_[static_cast<std::size_t>(entry)],
          host_matrix_receive_target_[static_cast<std::size_t>(entry)]) &&
          local_receive_valid;
    for (int entry = 0; entry < shape_.vector_receive_count; ++entry)
    {
      const int target = find_row(
          vector_receive_keys_[static_cast<std::size_t>(entry)]);
      local_receive_valid = target >= 0 && local_receive_valid;
      if (target >= 0)
        host_vector_receive_target_[static_cast<std::size_t>(entry)] = target;
    }
    if (!CollectiveContinue(local_receive_valid)) { return false; }
  }

  const auto upload = [](int* destination, const int* source, int count)
  {
    return TransportPolicy::HostToDevice(
        destination, source, static_cast<std::size_t>(count) * sizeof(int),
        CudaTransferCategory::topology, CudaTransferSite::map_upload);
  };
  const bool uploaded = upload(
                view_.matrix_owned_source, host_matrix_owned_source_.data(),
                shape_.matrix_owned_count) &&
      upload(view_.matrix_owned_section, host_matrix_owned_section_.data(),
             shape_.matrix_owned_count) &&
      upload(view_.matrix_owned_target, host_matrix_owned_target_.data(),
             shape_.matrix_owned_count) &&
      upload(view_.matrix_send_source, host_matrix_send_source_.data(),
             shape_.matrix_send_count) &&
      upload(view_.matrix_receive_section,
             host_matrix_receive_section_.data(),
             shape_.matrix_receive_count) &&
      upload(view_.matrix_receive_target, host_matrix_receive_target_.data(),
             shape_.matrix_receive_count) &&
      upload(view_.vector_owned_source, host_vector_owned_source_.data(),
             shape_.vector_owned_count) &&
      upload(view_.vector_owned_target, host_vector_owned_target_.data(),
             shape_.vector_owned_count) &&
      upload(view_.vector_send_source, host_vector_send_source_.data(),
             shape_.vector_send_count) &&
      upload(view_.vector_receive_target, host_vector_receive_target_.data(),
             shape_.vector_receive_count);
  return CollectiveContinue(uploaded);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BindStateSlots() noexcept
{
  double* trace0 = static_cast<mfem::Vector&>(*trace_slots_[0]).ReadWrite();
  double* trace1 = static_cast<mfem::Vector&>(*trace_slots_[1]).ReadWrite();
  committed_slot_.interior = {view_.slot_interior[0],
                              static_cast<std::size_t>(shape_.interior_count),
                              {}, {}, 0,
                              reinterpret_cast<std::uintptr_t>(
                                  view_.slot_interior[0]), true};
  candidate_slot_.interior = {view_.slot_interior[1],
                              static_cast<std::size_t>(shape_.interior_count),
                              {}, {}, 0,
                              reinterpret_cast<std::uintptr_t>(
                                  view_.slot_interior[1]), true};
  committed_slot_.trace = {trace0,
                           static_cast<std::size_t>(shape_.true_trace_count),
                           {}, {}, 0, reinterpret_cast<std::uintptr_t>(trace0),
                           reinterpret_cast<std::uintptr_t>(&*trace_slots_[0]),
                           true, true};
  candidate_slot_.trace = {trace1,
                           static_cast<std::size_t>(shape_.true_trace_count),
                           {}, {}, 0, reinterpret_cast<std::uintptr_t>(trace1),
                           reinterpret_cast<std::uintptr_t>(&*trace_slots_[1]),
                           true, true};
  workspace_slot_.interior = {view_.work_interior,
                              static_cast<std::size_t>(shape_.interior_count),
                              {}, {}, 0,
                              reinterpret_cast<std::uintptr_t>(
                                  view_.work_interior), true};
  workspace_slot_.trace = {
      view_.work_true_trace,
      static_cast<std::size_t>(shape_.true_trace_count), {}, {}, 0,
      reinterpret_cast<std::uintptr_t>(view_.work_true_trace), 0, false, true};
  return UpdateStateHashes(committed_slot_) &&
      UpdateStateHashes(candidate_slot_);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
VerifyResidencyAndTicket(CudaRunEvidence&) noexcept
{
  hypre_policy_valid_ = CudaHypreDevicePolicyValid(
      *true_matrix_, *hypre_rhs_, *hypre_solution_);
  device_resident_ = double_memory_.DeviceIsValid() &&
      integer_memory_.DeviceIsValid() &&
      trace_slots_[0]->GetMemory().DeviceIsValid() &&
      trace_slots_[1]->GetMemory().DeviceIsValid() && hypre_policy_valid_;
  return device_resident_ && Ticket(CudaFailurePhase::setup);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
Ticket(const CudaFailurePhase phase) noexcept
{
  TransportPolicy::SetPhase(phase, current_generation_);
  if (!CudaLaunchTicket(view_.ticket_words, phase, current_attempt_.sequence,
                        current_generation_, device_))
    return false;
  std::array<int, 8> words{};
  if (!TransportPolicy::DeviceToHost(
          words.data(), view_.ticket_words, words.size() * sizeof(int),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status))
    return false;
  const bool valid = CudaTicketWordsValid(
      words, phase, current_attempt_.sequence, current_generation_, device_);
  if (valid && ticket_count_ < tickets_.size())
    tickets_[ticket_count_++] = {phase, current_attempt_.sequence,
                                 current_generation_, words[1], words[2],
                                 true};
  return valid;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
StatusPassed() noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_reduction);
  if (!CudaLaunchStatusReduction(view_.status, shape_.elements,
                                 view_.status_flag))
    return false;
  int failure = 1;
  if (!TransportPolicy::DeviceToHost(
          &failure, view_.status_flag, sizeof(failure),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status))
    return false;
  return failure == 0;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
DistributeCurrentTrace(const double* true_values) noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_packing);
  if (ranks_ == 1)
    return CollectiveContinue(CudaLaunchGatherLocalTrace(
        view_.signed_element_trace,
        shape_.elements * shape_.trace_dofs, true_values,
        view_.current_local_trace));
  double* owned_host = owned_stage_.HostWrite();
  const bool staged_out = TransportPolicy::DeviceToHost(
          owned_host, true_values,
          static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double),
          CudaTransferCategory::host_staged_trace_exchange,
          CudaTransferSite::trace_solution_stage_out) &&
      TransportPolicy::Synchronize(
          CudaTransferSite::trace_solution_stage_out);
  if (!CollectiveContinue(staged_out))
    return false;
  const bool distributed =
      exchange_plan_.Distribute(*trace_space_, owned_stage_, local_stage_);
  const std::size_t local_bytes =
      static_cast<std::size_t>(shape_.local_trace_count) * sizeof(double);
  const std::size_t owned_bytes =
      static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double);
  const bool staging_recorded = TransportPolicy::RecordMpiStaging(
      host_exchange_bytes_,
      CudaTransferCategory::host_staged_trace_exchange,
      CudaTransferSite::trace_ghost_stage_in, owned_host,
      local_stage_.HostRead(), owned_bytes, local_bytes, distributed);
  if (!CollectiveContinue(distributed && staging_recorded))
    return false;
  const bool local_ready = TransportPolicy::HostToDevice(
             view_.distributed_local_trace, local_stage_.HostRead(), local_bytes,
             CudaTransferCategory::host_staged_trace_exchange,
             CudaTransferSite::trace_ghost_stage_in) &&
      CudaLaunchGatherLocalTrace(
          view_.signed_element_trace,
          shape_.elements * shape_.trace_dofs,
          view_.distributed_local_trace, view_.current_local_trace) &&
      TransportPolicy::Synchronize(CudaTransferSite::trace_ghost_stage_in);
  return CollectiveContinue(local_ready);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
DistributeTraceIncrement() noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_packing);
  const double* solution =
      static_cast<const mfem::Vector&>(*hypre_solution_).Read();
  if (ranks_ == 1)
    return CollectiveContinue(CudaLaunchGatherLocalTrace(
        view_.signed_element_trace,
        shape_.elements * shape_.trace_dofs, solution,
        view_.trace_increment_local));
  double* owned_host = owned_stage_.HostWrite();
  const bool staged_out = TransportPolicy::DeviceToHost(
          owned_host, solution,
          static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double),
          CudaTransferCategory::host_staged_trace_exchange,
          CudaTransferSite::trace_solution_stage_out) &&
      TransportPolicy::Synchronize(
          CudaTransferSite::trace_solution_stage_out);
  if (!CollectiveContinue(staged_out))
    return false;
  const bool distributed =
      exchange_plan_.Distribute(*trace_space_, owned_stage_, local_stage_);
  // local_stage is in local physical signed coefficients. Gather the element
  // side orientation in a second device kernel from the local array.
  const std::size_t local_bytes =
      static_cast<std::size_t>(shape_.local_trace_count) * sizeof(double);
  const std::size_t owned_bytes =
      static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double);
  const bool staging_recorded = TransportPolicy::RecordMpiStaging(
      host_exchange_bytes_,
      CudaTransferCategory::host_staged_trace_exchange,
      CudaTransferSite::trace_ghost_stage_in, owned_host,
      local_stage_.HostRead(), owned_bytes, local_bytes, distributed);
  if (!CollectiveContinue(distributed && staging_recorded)) { return false; }
  const bool local_ready = TransportPolicy::HostToDevice(
          view_.distributed_local_trace, local_stage_.HostRead(), local_bytes,
          CudaTransferCategory::host_staged_trace_exchange,
          CudaTransferSite::trace_ghost_stage_in) &&
      CudaLaunchGatherLocalTrace(
          view_.signed_element_trace,
          shape_.elements * shape_.trace_dofs,
          view_.distributed_local_trace, view_.trace_increment_local) &&
      TransportPolicy::Synchronize(CudaTransferSite::trace_ghost_stage_in);
  return CollectiveContinue(local_ready);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleTrueVector(const double* element_values, double* true_values,
                   const bool completion_required) noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_trace_assembly);
  const bool local_gather = CudaLaunchAssembleOwnedVector(
      view_, shape_, element_values, true_values) &&
      CudaLaunchGatherRemoteVector(view_, shape_, element_values);
  if (!CollectiveContinue(local_gather))
    return false;
  if (ranks_ == 2)
  {
    const int peer = 1 - rank_;
    const std::size_t send_bytes =
        static_cast<std::size_t>(shape_.vector_send_count) * sizeof(double);
    const std::size_t receive_bytes =
        static_cast<std::size_t>(shape_.vector_receive_count) * sizeof(double);
    const bool staged_out = TransportPolicy::DeviceToHost(
            vector_send_host_.data(), view_.remote_vector_send, send_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::rhs_remote_gather) &&
        TransportPolicy::Synchronize(CudaTransferSite::rhs_remote_gather);
    if (!CollectiveContinue(staged_out)) { return false; }
    const bool exchanged = MPI_Sendrecv(
                     vector_send_host_.data(), shape_.vector_send_count,
                     MPI_DOUBLE, peer, 4201, vector_receive_host_.data(),
                     shape_.vector_receive_count, MPI_DOUBLE, peer, 4201,
                     communicator_, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    const bool exchange_recorded = TransportPolicy::RecordMpiStaging(
            send_bytes + receive_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::rhs_remote_gather,
            vector_send_host_.data(), vector_receive_host_.data(),
            vector_send_host_.size() * sizeof(double),
            vector_receive_host_.size() * sizeof(double), exchanged);
    if (!CollectiveContinue(exchanged && exchange_recorded))
      return false;
    const bool scattered = TransportPolicy::HostToDevice(
            view_.remote_vector_receive, vector_receive_host_.data(),
            receive_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::rhs_remote_scatter) &&
        CudaLaunchScatterRemoteVector(view_, shape_, true_values);
    if (!CollectiveContinue(scattered))
      return false;
  }
  if (!completion_required)
    return true;
  return CollectiveContinue(
      TransportPolicy::Synchronize(CudaTransferSite::rhs_remote_scatter));
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleTrueSystem() noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_trace_assembly);
  const bool local_gather = CudaLaunchZeroTrueSystem(
      true_device_, view_.true_rhs_scratch, shape_.true_trace_count) &&
      CudaLaunchAssembleOwnedMatrix(view_, true_device_, shape_) &&
      CudaLaunchGatherRemoteMatrix(view_, shape_);
  if (!CollectiveContinue(local_gather))
    return false;
  if (ranks_ == 2)
  {
    const int peer = 1 - rank_;
    const std::size_t send_bytes =
        static_cast<std::size_t>(shape_.matrix_send_count) * sizeof(double);
    const std::size_t receive_bytes =
        static_cast<std::size_t>(shape_.matrix_receive_count) * sizeof(double);
    const bool staged_out = TransportPolicy::DeviceToHost(
            matrix_send_host_.data(), view_.remote_matrix_send, send_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::matrix_remote_gather) &&
        TransportPolicy::Synchronize(CudaTransferSite::matrix_remote_gather);
    if (!CollectiveContinue(staged_out)) { return false; }
    const bool exchanged = MPI_Sendrecv(
                     matrix_send_host_.data(), shape_.matrix_send_count,
                     MPI_DOUBLE, peer, 4202, matrix_receive_host_.data(),
                     shape_.matrix_receive_count, MPI_DOUBLE, peer, 4202,
                     communicator_, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    const bool exchange_recorded = TransportPolicy::RecordMpiStaging(
            send_bytes + receive_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::matrix_remote_gather,
            matrix_send_host_.data(), matrix_receive_host_.data(),
            matrix_send_host_.size() * sizeof(double),
            matrix_receive_host_.size() * sizeof(double), exchanged);
    if (!CollectiveContinue(exchanged && exchange_recorded))
      return false;
    const bool scattered = TransportPolicy::HostToDevice(
            view_.remote_matrix_receive, matrix_receive_host_.data(),
            receive_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::matrix_remote_scatter) &&
        CudaLaunchScatterRemoteMatrix(view_, true_device_, shape_);
    if (!CollectiveContinue(scattered))
      return false;
  }
  if (!AssembleTrueVector(
          view_.condensed_rhs, view_.true_rhs_scratch, true))
    return false;
  true_matrix_->HypreRead();
  hypre_rhs_->HypreRead();
  return CollectiveContinue(CudaHypreDevicePolicyValid(
      *true_matrix_, *hypre_rhs_, *hypre_solution_));
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
LaunchAndObserveNativeRankOneTraceSolve(
    const CudaNativeIlu0FactorView& factor,
    const std::uint64_t matrix_generation,
    const double requested_relative_tolerance,
    CudaSolveDiagnostics& diagnostics) noexcept
{
  if (trace_solver_kind_ != CudaTraceSolverKind::native_rank1_fp64 ||
      ranks_ != 1 || !true_matrix_ || !hypre_rhs_ || !hypre_solution_ ||
      !native_trace_requirements_.valid ||
      !native_trace_support_.supported ||
      native_trace_requirements_.control_double_slots == 0 ||
      native_trace_requirements_.control_double_slots >
          native_trace_host_control_.size())
    return false;
  const bool use_ilu = shape_.igr_active;
  const bool factor_exact = use_ilu
      ? factor.offsets != nullptr && factor.columns != nullptr &&
          factor.values != nullptr && factor.permutation != nullptr &&
          factor.inverse_diagonal == native_ilu_inverse_diagonal_ &&
          factor.rows == shape_.true_trace_count &&
          factor.nonzeros >= factor.rows &&
          factor.lower_jacobi_iterations ==
              trace_policy_descriptor_.ilu_lower_jacobi_iterations &&
          factor.upper_jacobi_iterations ==
              trace_policy_descriptor_.ilu_upper_jacobi_iterations
      : factor.offsets == nullptr && factor.columns == nullptr &&
          factor.values == nullptr && factor.permutation == nullptr &&
          factor.inverse_diagonal == nullptr && factor.rows == 0 &&
          factor.nonzeros == 0;
  CudaTrueMatrixDeviceView current_matrix{};
  const bool matrix_exact = factor_exact &&
      CudaGetTrueMatrixDeviceView(*true_matrix_, current_matrix) &&
      current_matrix.rows == shape_.true_trace_count &&
      current_matrix.diag_columns_count == shape_.true_trace_count &&
      current_matrix.offd_nonzeros == 0 &&
      current_matrix.diag_offsets == true_device_.diag_offsets &&
      current_matrix.diag_columns == true_device_.diag_columns &&
      current_matrix.diag_values == true_device_.diag_values &&
      current_matrix.diag_nonzeros == true_device_.diag_nonzeros;
  if (!matrix_exact) return false;
  // HYPRE may issue factor/extracted-diagonal work on its configured compute
  // stream.  The one first-factor-per-attempt refresh must complete before
  // the application-owned cooperative stream borrows those pointers.
  if (use_ilu && diagnostics.ilu_factor_setups == 1 &&
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status))
    return false;

  bool vectors_ready = false;
  try
  {
    *hypre_solution_ = 0.0;
    hypre_rhs_->HypreReadWrite();
    hypre_solution_->HypreReadWrite();
    vectors_ready = CudaHypreDevicePolicyValid(
        *true_matrix_, *hypre_rhs_, *hypre_solution_) &&
        view_.true_rhs_scratch ==
            static_cast<mfem::Vector&>(*hypre_rhs_).ReadWrite() &&
        view_.true_solution ==
            static_cast<mfem::Vector&>(*hypre_solution_).ReadWrite() &&
        HYPRE_GetError() == 0;
  }
  catch (...) { vectors_ready = false; }
  if (!vectors_ready) return false;

  native_trace_host_control_.fill(0.0);
  const bool launched = CudaLaunchNativeRankOneTraceSolver(
      current_matrix, view_.true_rhs_scratch, view_.true_solution,
      use_ilu, factor, native_trace_workspace_, native_trace_support_,
      requested_relative_tolerance, maximum_linear_iterations_,
      krylov_dimension_, nullptr);
  const std::size_t control_bytes =
      native_trace_requirements_.control_double_slots * sizeof(double);
  const bool observed = launched && TransportPolicy::DeviceToHost(
      native_trace_host_control_.data(),
      native_trace_workspace_.device_control, control_bytes,
      CudaTransferCategory::scalar_diagnostic,
      CudaTransferSite::scalar_status, control_bytes,
      native_trace_host_control_.size() * sizeof(double)) &&
      TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
  CudaNativeTraceSolveObservation observation{};
  const bool decoded = observed && CudaDecodeNativeRankOneTraceControl(
      native_trace_host_control_.data(),
      native_trace_requirements_.control_double_slots,
      requested_relative_tolerance, maximum_linear_iterations_,
      krylov_dimension_, observation);

  diagnostics.matrix_generation = matrix_generation;
  diagnostics.trace_solver = CudaTraceSolverKind::native_rank1_fp64;
  diagnostics.trace_preconditioner =
      trace_policy_descriptor_.trace_preconditioner;
  diagnostics.igr_active = shape_.igr_active;
  diagnostics.krylov_dimension = krylov_dimension_;
  diagnostics.iterations = observation.iterations;
  diagnostics.reported_residual = observation.relative_residual_norm;
  diagnostics.device_policy = decoded && CudaHypreDevicePolicyValid(
      *true_matrix_, *hypre_rhs_, *hypre_solution_);
  diagnostics.configured_native_solver = decoded &&
      observation.ready_for_true_residual;
  diagnostics.configured_ilu = use_ilu && factor_exact;
  diagnostics.configured_diagonal = !use_ilu && factor_exact;
  diagnostics.configured_preconditioner =
      diagnostics.configured_native_solver && factor_exact;
  diagnostics.finite = decoded && observation.finite;
  diagnostics.reported_tolerance_met = diagnostics.finite &&
      diagnostics.reported_residual <= requested_relative_tolerance;

  bool native_accounting_ready = true;
  if (use_ilu)
  {
    diagnostics.ilu_type = trace_policy_descriptor_.ilu_type;
    diagnostics.ilu_level_of_fill =
        trace_policy_descriptor_.ilu_level_of_fill;
    diagnostics.ilu_local_reordering =
        trace_policy_descriptor_.ilu_local_reordering;
    diagnostics.ilu_application_iterations =
        trace_policy_descriptor_.ilu_application_iterations;
    diagnostics.ilu_triangular_solve_kind =
        trace_policy_descriptor_.ilu_triangular_solve_kind;
    diagnostics.ilu_lower_jacobi_iterations =
        trace_policy_descriptor_.ilu_lower_jacobi_iterations;
    diagnostics.ilu_upper_jacobi_iterations =
        trace_policy_descriptor_.ilu_upper_jacobi_iterations;
    diagnostics.ilu_accounting_kind =
        CudaIluAccountingKind::derived_bounds_v1;
    diagnostics.ilu_factor_reuse_policy =
        trace_policy_descriptor_.ilu_factor_reuse_policy;
    const std::uint64_t jacobi_updates =
        static_cast<std::uint64_t>(
            diagnostics.ilu_lower_jacobi_iterations - 1) +
        static_cast<std::uint64_t>(
            diagnostics.ilu_upper_jacobi_iterations - 1);
    const CudaDerivedIluAccountingBoundsV1 bounds =
        observation.iterations >= 0
        ? CudaDeriveIluAccountingBoundsV1(
              static_cast<std::uint64_t>(observation.iterations),
              static_cast<std::uint64_t>(krylov_dimension_),
              static_cast<std::uint64_t>(
                  diagnostics.ilu_application_iterations),
              jacobi_updates)
        : CudaDerivedIluAccountingBoundsV1{};
    diagnostics.ilu_preconditioner_callbacks_lower_bound =
        bounds.ilu_preconditioner_callbacks_lower_bound;
    diagnostics.ilu_preconditioner_callbacks_upper_bound =
        bounds.ilu_preconditioner_callbacks_upper_bound;
    diagnostics.ilu_inner_iterations_lower_bound =
        bounds.ilu_inner_iterations_lower_bound;
    diagnostics.ilu_inner_iterations_upper_bound =
        bounds.ilu_inner_iterations_upper_bound;
    diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound =
        bounds.ilu_jacobi_factor_spmv_updates_lower_bound;
    diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound =
        bounds.ilu_jacobi_factor_spmv_updates_upper_bound;
    const std::uint64_t applications =
        observation.preconditioner_applications >= 0
        ? static_cast<std::uint64_t>(
              observation.preconditioner_applications)
        : std::numeric_limits<std::uint64_t>::max();
    native_accounting_ready = bounds.ilu_accounting_complete &&
        applications >= bounds.ilu_preconditioner_callbacks_lower_bound &&
        applications <= bounds.ilu_preconditioner_callbacks_upper_bound;
    diagnostics.ilu_accounting_complete = native_accounting_ready;
  }
  return native_accounting_ready && diagnostics.finite &&
      CudaTraceSolveReadyForTrueResidualVerification(
          diagnostics, maximum_linear_iterations_);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
SolveTrueSystemAttempt(double& true_relative_residual,
                const std::uint64_t matrix_generation,
                const double requested_relative_tolerance,
                const bool same_matrix_retry,
                int& linear_iterations,
                CudaTensorCoreEvidence& tensor_core_evidence) noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_trace_solve);
  linear_iterations = 0;
  CudaSolveDiagnostics diagnostics{};
  const bool use_ilu = CudaTracePreconditionerIsIlu(
      trace_preconditioner_);
  bool solved = false;
  if (trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64)
  {
    CudaNativeIlu0FactorView factor{};
    const bool factor_ready = !use_ilu ||
        CudaPrepareNativeRankOneIluFactor(
            *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_,
            trace_policy_descriptor_, current_attempt_, matrix_generation,
            native_ilu_inverse_diagonal_,
            generic_arena_layout_.native_ilu_inverse_diagonal,
            ilu_factor_state_, factor, diagnostics, same_matrix_retry);
    solved = factor_ready && LaunchAndObserveNativeRankOneTraceSolve(
        factor, matrix_generation, requested_relative_tolerance,
        diagnostics);
    if (solved && use_ilu)
    {
      const bool state_generation_ready = ilu_factor_state_.factor_ready &&
          ilu_factor_state_.factor_generation <= matrix_generation &&
          (same_matrix_retry
               ? matrix_generation == ilu_factor_state_.last_solve_generation
               : matrix_generation > ilu_factor_state_.last_solve_generation);
      if (state_generation_ready)
        ilu_factor_state_.last_solve_generation = matrix_generation;
      else
        solved = false;
    }
  }
  else if (trace_solver_kind_ == CudaTraceSolverKind::hypre_parcsr_gmres ||
           trace_solver_kind_ ==
               CudaTraceSolverKind::hypre_host_staged_rank2)
  {
    if (use_ilu)
    {
      const bool ordinary_rank1_refresh = ranks_ == 1 &&
          CudaOrdinaryRankOneRefreshPolicyIsExact(
              trace_policy_descriptor_);
      if (ordinary_rank1_refresh)
      {
        const bool attempt_ready =
            CudaOrdinaryRankOneRefreshAttemptIsExact(
                ilu_factor_state_, current_attempt_,
                reinterpret_cast<std::uintptr_t>(true_matrix_.get()),
                reinterpret_cast<std::uintptr_t>(true_device_.diag_values),
                reinterpret_cast<std::uintptr_t>(true_device_.offd_values),
                true_device_.diag_nonzeros, true_device_.offd_nonzeros);
        solved = attempt_ready && CudaOrdinaryRankOneRefreshFactorAndSolve(
            *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_, *gmres_,
            ranks_, communicator_, trace_policy_descriptor_,
            requested_relative_tolerance, maximum_linear_iterations_,
            krylov_dimension_, matrix_generation, diagnostics);
      }
      else
      {
        solved = CudaSolveWithIluFactorPolicy(
            *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_, *gmres_,
            ranks_, communicator_, trace_policy_descriptor_,
            current_attempt_, requested_relative_tolerance,
            maximum_linear_iterations_, krylov_dimension_,
            matrix_generation, ilu_factor_state_, diagnostics,
            same_matrix_retry);
      }
    }
    else
    {
      solved = CudaRefreshDiagonalAndSolve(
          *true_matrix_, *hypre_rhs_, *hypre_solution_, *diagonal_,
          *gmres_, ranks_, communicator_, trace_policy_descriptor_,
          requested_relative_tolerance, maximum_linear_iterations_,
          krylov_dimension_, matrix_generation, diagnostics);
    }
  }
  if (use_ilu && solved)
    (void)CudaAccumulateIluSolveEvidence(
        tensor_core_evidence, diagnostics);
  if (solved)
  {
    std::uint64_t* const calls = trace_solver_kind_ ==
            CudaTraceSolverKind::native_rank1_fp64
        ? &tensor_core_evidence.native_rank1_trace_solve_calls
        : &tensor_core_evidence.hypre_trace_solve_calls;
    if (*calls == std::numeric_limits<std::uint64_t>::max())
    {
      tensor_core_evidence.trace_solver_accounting_complete = false;
      solved = false;
    }
    else
      ++*calls;
  }
  // Policy/factor/configuration mismatches already fail `solved`.  Aggregate
  // accounting remains qualification evidence only; the FP64 current-matrix
  // true residual below is the numerical acceptance authority.
  const bool local_solve = solved &&
      diagnostics.matrix_generation == matrix_generation &&
      diagnostics.trace_solver == trace_solver_kind_ &&
      CudaTraceSolveReadyForTrueResidualVerification(
          diagnostics, maximum_linear_iterations_);
  linear_iterations = diagnostics.iterations;
  if (!CollectiveContinue(local_solve))
    return false;
  true_matrix_->Mult(*hypre_solution_, *hypre_action_);
  *hypre_action_ -= *hypre_rhs_;
  const double residual_squared = mfem::InnerProduct(
      communicator_, *hypre_action_, *hypre_action_);
  const double rhs_squared = mfem::InnerProduct(
      communicator_, *hypre_rhs_, *hypre_rhs_);
  true_relative_residual = std::sqrt(residual_squared) /
      std::max(std::sqrt(rhs_squared), 1.0e-300);
  return std::isfinite(true_relative_residual);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
SolveTrueSystem(const generic::EulerHdgLinearSolveRole role,
                double& true_relative_residual,
                const std::uint64_t matrix_generation,
                int& linear_iterations,
                int& linear_solve_attempts,
                int& true_residual_tightenings,
                CudaTensorCoreEvidence& tensor_core_evidence) noexcept
{
  linear_iterations = 0;
  linear_solve_attempts = 0;
  true_residual_tightenings = 0;
  const double requested_tolerance =
      role == generic::EulerHdgLinearSolveRole::newton_correction
      ? newton_correction_linear_relative_tolerance_
      : final_coherent_linear_relative_tolerance_;
  const auto invoke = [&](const double tolerance,
                          const bool same_matrix_retry) noexcept
  {
    int invocation_iterations = 0;
    const bool solved = SolveTrueSystemAttempt(
        true_relative_residual, matrix_generation, tolerance,
        same_matrix_retry, invocation_iterations, tensor_core_evidence);
    const bool bounded = invocation_iterations >= 0 &&
        linear_iterations <= std::numeric_limits<int>::max() -
            invocation_iterations &&
        linear_solve_attempts != std::numeric_limits<int>::max();
    if (bounded)
    {
      linear_iterations += invocation_iterations;
      ++linear_solve_attempts;
    }
    return solved && bounded;
  };

  if (!invoke(requested_tolerance, false)) return false;
  if (true_relative_residual <= true_residual_relative_tolerance_)
    return true;
  const bool tighten =
      true_residual_tightening_linear_relative_tolerance_ <
          requested_tolerance;
  if (!CollectiveContinue(tighten)) return false;
  ++true_residual_tightenings;
  return invoke(true_residual_tightening_linear_relative_tolerance_, true) &&
      true_relative_residual <= true_residual_relative_tolerance_;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleAndNorm(state_slot&, const double dt, double& norm) noexcept
{
  TransportPolicy::SetPhase(CudaFailurePhase::assembly, current_generation_);
  bool assembly_launched = false;
  {
    ::dg_igr::NVTXRange profile_region(
        ::dg_igr::observability::RegionId::hdg_local_assembly);
    assembly_launched = CudaLaunchAssembly(
        view_, shape_, dt, cuda_engine_);
  }
  const bool local_assembly = assembly_launched && StatusPassed();
  if (!CollectiveContinue(local_assembly) ||
      !AssembleTrueVector(view_.residual_g, view_.true_residual, false))
    return false;
  // StatusPassed has consumed and synchronized every per-element assembly
  // status.  Later condensation/admissibility/assembly phases overwrite the
  // whole array before its next status use, so it is safe norm scratch here.
  {
    ::dg_igr::NVTXRange profile_region(
        ::dg_igr::observability::RegionId::hdg_reduction);
    const bool norm_launched = CudaLaunchLocalSquaredNorm(
        view_.residual_r, shape_.interior_count, view_.true_residual,
        shape_.true_trace_count, view_.status, shape_.elements, view_.scalar);
    if (!CollectiveContinue(norm_launched))
      return false;
    double local_squared = 0.0;
    const bool scalar_ready = TransportPolicy::DeviceToHost(
            &local_squared, view_.scalar, sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!CollectiveContinue(scalar_ready))
      return false;
    double global_squared = local_squared;
    if (ranks_ > 1 && MPI_Allreduce(&local_squared, &global_squared, 1,
                                    MPI_DOUBLE, MPI_SUM,
                                    communicator_) != MPI_SUCCESS)
      return false;
    norm = std::sqrt(global_squared);
    return std::isfinite(norm);
  }
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
CopyBase(state_slot& workspace) noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_packing);
  const bool interior = TransportPolicy::DeviceToDevice(
             view_.base_interior, workspace.interior.coefficients,
             workspace.interior.count * sizeof(double),
             CudaTransferCategory::attempt_history,
             CudaTransferSite::nonlinear_base_interior);
  const bool trace = TransportPolicy::DeviceToDevice(
          view_.base_true_trace, workspace.trace.coefficients,
          workspace.trace.count * sizeof(double),
          CudaTransferCategory::attempt_history,
          CudaTransferSite::nonlinear_base_trace);
  const bool local_trace = TransportPolicy::DeviceToDevice(
          view_.base_local_trace, view_.current_local_trace,
          static_cast<std::size_t>(shape_.elements) *
              shape_.trace_dofs * sizeof(double),
          CudaTransferCategory::attempt_history,
          CudaTransferSite::nonlinear_base_local_trace);
  return interior && trace && local_trace;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
RestoreBase(state_slot& workspace) noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_packing);
  const bool interior = TransportPolicy::DeviceToDevice(
      workspace.interior.coefficients, view_.base_interior,
      workspace.interior.count * sizeof(double),
      CudaTransferCategory::attempt_history,
      CudaTransferSite::nonlinear_restore_interior);
  const bool trace = TransportPolicy::DeviceToDevice(
      workspace.trace.coefficients, view_.base_true_trace,
      workspace.trace.count * sizeof(double),
      CudaTransferCategory::attempt_history,
      CudaTransferSite::nonlinear_restore_trace);
  const bool local_trace = TransportPolicy::DeviceToDevice(
      view_.current_local_trace, view_.base_local_trace,
      static_cast<std::size_t>(shape_.elements) *
          shape_.trace_dofs * sizeof(double),
      CudaTransferCategory::attempt_history,
      CudaTransferSite::nonlinear_restore_local_trace);
  const bool synchronized = TransportPolicy::Synchronize(
      CudaTransferSite::nonlinear_restore_local_trace);
  return interior && trace && local_trace && synchronized;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
HashState(const double* values, const int count, StateHash& hash) noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_validation);
  if (!CudaLaunchHash(values, count, view_.hash_words))
    return false;
  std::array<std::uint64_t, 4> words{};
  if (!TransportPolicy::DeviceToHost(
          words.data(), view_.hash_words, sizeof(words),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::state_hash) ||
      !TransportPolicy::Synchronize(CudaTransferSite::state_hash))
    return false;
  std::memcpy(hash.bytes.data(), words.data(), sizeof(words));
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
UpdateStateHashes(state_slot& workspace) noexcept
{
  ::dg_igr::NVTXRange profile_region(
      ::dg_igr::observability::RegionId::hdg_validation);
  constexpr std::size_t words_per_hash = 4;
  std::array<std::uint64_t, 2 * words_per_hash> words{};
  const bool local_ready = CudaLaunchHashPair(
          workspace.interior.coefficients, shape_.interior_count,
          workspace.trace.coefficients, shape_.true_trace_count,
          view_.hash_words) &&
      TransportPolicy::DeviceToHost(
          words.data(), view_.hash_words, sizeof(words),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::state_hash) &&
      TransportPolicy::Synchronize(CudaTransferSite::state_hash);
  if (!CollectiveContinue(local_ready)) { return false; }
  static_assert(words_per_hash * sizeof(std::uint64_t) == 32,
                "StateHash must remain exactly 32 bytes");
  std::memcpy(workspace.interior.local_hash.bytes.data(), words.data(),
              words_per_hash * sizeof(std::uint64_t));
  std::memcpy(workspace.trace.local_hash.bytes.data(),
              words.data() + words_per_hash,
              words_per_hash * sizeof(std::uint64_t));
  return AggregateHash(workspace.interior.local_hash,
                       workspace.interior.hash) &&
      AggregateHash(workspace.trace.local_hash, workspace.trace.hash);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AggregateHash(const StateHash& local, StateHash& aggregate) const noexcept
{
  std::array<unsigned char, 2 * 32> pieces{};
  if (ranks_ < 1 || ranks_ > 2 ||
      MPI_Allgather(local.bytes.data(), 32, MPI_UNSIGNED_CHAR,
                    pieces.data(), 32, MPI_UNSIGNED_CHAR,
                    communicator_) != MPI_SUCCESS)
    return false;
  aggregate = HashHostBytes(
      pieces.data(), static_cast<std::size_t>(ranks_) * 32);
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
std::size_t CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
WorkspaceBytes() const noexcept
{
  if (!(borrowed_topology_ && shape_.borrowed_host_inputs))
  {
    return (persistent_double_capacity -
            2 * maximum_rank_elements * maximum_interior_dofs_per_element -
            2 * maximum_true_trace_dofs) * sizeof(double) +
        persistent_integer_capacity * sizeof(int);
  }
  const std::size_t committed_slots = 2 * static_cast<std::size_t>(
      shape_.interior_count) + 2 * static_cast<std::size_t>(
      shape_.true_trace_count);
  return (double_arena_capacity_ - committed_slots) * sizeof(double) +
      integer_arena_capacity_ * sizeof(int);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
OraclePayloadFitsCapacity() const noexcept
{
  const auto nonnegative = [](const int value) { return value >= 0; };
  if (!nonnegative(shape_.elements) || !nonnegative(shape_.interior_count) ||
      !nonnegative(shape_.local_trace_count) ||
      !nonnegative(shape_.true_trace_count) ||
      !nonnegative(shape_.trace_dofs) ||
      !nonnegative(shape_.volume_points) || !nonnegative(shape_.face_points) ||
      !nonnegative(shape_.scalar_interior_dofs) ||
      !nonnegative(shape_.scalar_face_dofs) ||
      !nonnegative(true_structure_.diag_nonzeros) ||
      !nonnegative(true_structure_.offd_nonzeros))
    return false;

  // Check dimensions before forming payload products.  These fixed limits
  // bound every CudaOraclePod array and keep the following arithmetic exact.
  if (shape_.elements > maximum_rank_elements ||
      shape_.trace_dofs > maximum_trace_dofs_per_element ||
      shape_.volume_points > maximum_volume_points ||
      shape_.face_points > maximum_face_points ||
      shape_.scalar_interior_dofs > maximum_scalar_dofs_per_element ||
      shape_.scalar_face_dofs > maximum_scalar_dofs_per_face ||
      shape_.interior_count >
          maximum_rank_elements * maximum_interior_dofs_per_element ||
      shape_.local_trace_count > maximum_local_trace_dofs ||
      shape_.true_trace_count > maximum_true_trace_dofs)
    return false;

  const std::size_t elements = static_cast<std::size_t>(shape_.elements);
  const std::size_t interior = static_cast<std::size_t>(shape_.interior_count);
  const std::size_t local_trace =
      static_cast<std::size_t>(shape_.local_trace_count);
  const std::size_t true_trace =
      static_cast<std::size_t>(shape_.true_trace_count);
  const std::size_t element_trace = elements *
      static_cast<std::size_t>(shape_.trace_dofs);
  const std::size_t csr_nonzeros = static_cast<std::size_t>(
      true_structure_.diag_nonzeros) + static_cast<std::size_t>(
      true_structure_.offd_nonzeros);
  const auto fits = [](const std::size_t available, const std::size_t required,
                       const std::size_t capacity)
  { return required <= available && required <= capacity; };

  return fits(signed_element_trace_.size(), element_trace,
              maximum_element_trace_dofs) &&
      fits(local_global_true_.size(), local_trace, maximum_local_trace_dofs) &&
      fits(owned_global_true_.size(), true_trace, maximum_true_trace_dofs) &&
      fits(element_coordinate_keys_.size(), elements * dimension,
           maximum_rank_elements * dimension) &&
      fits(host_volume_shape_.size(), elements * shape_.volume_points *
           shape_.scalar_interior_dofs, maximum_rank_elements *
           maximum_volume_points * maximum_scalar_dofs_per_element) &&
      fits(host_volume_derivative_.size(), elements * shape_.volume_points *
           shape_.scalar_interior_dofs * dimension, maximum_rank_elements *
           maximum_volume_points * maximum_scalar_dofs_per_element * dimension) &&
      fits(host_volume_weight_.size(), elements * shape_.volume_points,
           maximum_rank_elements * maximum_volume_points) &&
      fits(host_volume_position_.size(), elements * shape_.volume_points * dimension,
           maximum_rank_elements * maximum_volume_points * dimension) &&
      fits(host_face_interior_shape_.size(), elements * faces_per_element *
           shape_.face_points * shape_.scalar_interior_dofs,
           maximum_rank_elements * faces_per_element * maximum_face_points *
           maximum_scalar_dofs_per_element) &&
      fits(host_face_trace_shape_.size(), elements * faces_per_element *
           shape_.face_points * shape_.scalar_face_dofs,
           maximum_rank_elements * faces_per_element * maximum_face_points *
           maximum_scalar_dofs_per_face) &&
      fits(host_face_weight_.size(), elements * faces_per_element * shape_.face_points,
           maximum_rank_elements * faces_per_element * maximum_face_points) &&
      fits(host_face_normal_.size(), elements * faces_per_element *
           shape_.face_points * dimension, maximum_rank_elements *
           faces_per_element * maximum_face_points * dimension) &&
      interior <= maximum_rank_elements * maximum_interior_dofs_per_element &&
      csr_nonzeros <= maximum_true_matrix_entries &&
      host_true_diag_values_.size() >= static_cast<std::size_t>(
          true_structure_.diag_nonzeros) &&
      host_true_offd_values_.size() >= static_cast<std::size_t>(
          true_structure_.offd_nonzeros) &&
      true_structure_.diag_offsets.size() >= true_trace + 1 &&
      true_structure_.offd_offsets.size() >= true_trace + 1 &&
      true_structure_.diag_columns.size() >= static_cast<std::size_t>(
          true_structure_.diag_nonzeros) &&
      true_structure_.offd_columns.size() >= static_cast<std::size_t>(
          true_structure_.offd_nonzeros);
}

template <typename AllocatorPolicy, typename TransportPolicy>
void CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
CaptureOracleMetadata(const AttemptIdentity& attempt,
                      const std::uint64_t generation) noexcept
{
  oracle_ = {};
  oracle_captured_ = false;
  // Dynamic borrowed storage is not a complete CudaOraclePod.  Refuse the
  // fixed export rather than truncating bytes or writing beyond the ABI.
  if (!OraclePayloadFitsCapacity()) return;
  oracle_.rank = rank_;
  oracle_.ranks = ranks_;
  oracle_.elements = shape_.elements;
  oracle_.interior_count = shape_.interior_count;
  oracle_.local_trace_count = shape_.local_trace_count;
  oracle_.true_trace_count = shape_.true_trace_count;
  oracle_.global_true_trace_count = shape_.global_true_trace_count;
  oracle_.volume_points = shape_.volume_points;
  oracle_.face_points = shape_.face_points;
  oracle_.scalar_interior_dofs = shape_.scalar_interior_dofs;
  oracle_.scalar_face_dofs = shape_.scalar_face_dofs;
  oracle_.interior_dofs = shape_.interior_dofs;
  oracle_.trace_dofs = shape_.trace_dofs;
  oracle_.attempt_sequence = attempt.sequence;
  oracle_.product_generation = generation;
  std::copy_n(signed_element_trace_.begin(),
              shape_.elements * shape_.trace_dofs,
              oracle_.signed_element_trace.begin());
  std::copy_n(local_global_true_.begin(), shape_.local_trace_count,
              oracle_.local_to_global_true.begin());
  std::copy_n(owned_global_true_.begin(), shape_.true_trace_count,
              oracle_.owned_global_true.begin());
  std::copy_n(element_coordinate_keys_.begin(), shape_.elements * dimension,
              oracle_.element_coordinate_keys.begin());
  std::copy(host_volume_shape_.begin(), host_volume_shape_.end(),
            oracle_.volume_shape.begin());
  std::copy(host_volume_derivative_.begin(), host_volume_derivative_.end(),
            oracle_.volume_derivative.begin());
  std::copy(host_volume_weight_.begin(), host_volume_weight_.end(),
            oracle_.volume_weight.begin());
  std::copy(host_volume_position_.begin(), host_volume_position_.end(),
            oracle_.volume_position.begin());
  std::copy(host_face_interior_shape_.begin(),
            host_face_interior_shape_.end(),
            oracle_.face_interior_shape.begin());
  std::copy(host_face_trace_shape_.begin(), host_face_trace_shape_.end(),
            oracle_.face_trace_shape.begin());
  std::copy(host_face_weight_.begin(), host_face_weight_.end(),
            oracle_.face_weight.begin());
  std::copy(host_face_normal_.begin(), host_face_normal_.end(),
            oracle_.face_normal.begin());
  oracle_captured_ = true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
ExportTrueSystem(CudaOraclePod& pod) noexcept
{
  if (!OraclePayloadFitsCapacity() ||
      !TransportPolicy::DeviceToHost(
          host_true_diag_values_.data(), true_device_.diag_values,
          static_cast<std::size_t>(true_structure_.diag_nonzeros) *
              sizeof(double),
          CudaTransferCategory::oracle_export,
          CudaTransferSite::oracle_export) ||
      !TransportPolicy::DeviceToHost(
          host_true_offd_values_.data(), true_device_.offd_values,
          static_cast<std::size_t>(true_structure_.offd_nonzeros) *
              sizeof(double),
          CudaTransferCategory::oracle_export,
          CudaTransferSite::oracle_export) ||
      !TransportPolicy::Synchronize(CudaTransferSite::oracle_export))
    return false;
  int cursor = 0;
  pod.true_csr_offsets[0] = 0;
  for (int row = 0; row < shape_.true_trace_count; ++row)
  {
    for (int entry = true_structure_.diag_offsets[static_cast<std::size_t>(row)];
         entry < true_structure_.diag_offsets[static_cast<std::size_t>(row + 1)];
         ++entry)
    {
      const int local_column =
          true_structure_.diag_columns[static_cast<std::size_t>(entry)];
      pod.true_csr_columns[static_cast<std::size_t>(cursor)] =
          owned_global_true_[static_cast<std::size_t>(local_column)];
      pod.true_csr_values[static_cast<std::size_t>(cursor++)] =
          host_true_diag_values_[static_cast<std::size_t>(entry)];
    }
    for (int entry = true_structure_.offd_offsets[static_cast<std::size_t>(row)];
         entry < true_structure_.offd_offsets[static_cast<std::size_t>(row + 1)];
         ++entry)
    {
      pod.true_csr_columns[static_cast<std::size_t>(cursor)] =
          true_structure_.offd_columns[static_cast<std::size_t>(entry)];
      pod.true_csr_values[static_cast<std::size_t>(cursor++)] =
          host_true_offd_values_[static_cast<std::size_t>(entry)];
    }
    pod.true_csr_offsets[static_cast<std::size_t>(row + 1)] = cursor;
  }
  pod.true_csr_rows = shape_.true_trace_count;
  pod.true_csr_nonzeros = cursor;
  return ExportArray(pod.true_rhs.data(),
                     static_cast<const mfem::Vector&>(*hypre_rhs_).Read(),
                     shape_.true_trace_count) &&
      ExportArray(pod.true_solution.data(),
                  static_cast<const mfem::Vector&>(*hypre_solution_).Read(),
                  shape_.true_trace_count);
}

} // namespace detail
