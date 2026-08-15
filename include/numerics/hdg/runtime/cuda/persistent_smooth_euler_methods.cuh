/* Included inside dg::hdg::runtime::cuda_w8a. */

namespace detail
{

inline int CudaDecodeSigned(const int encoded, double& sign) noexcept
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
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
BuildHostGeometryAndMaps() noexcept
{
  std::vector<double> local_exact_trace(
      static_cast<std::size_t>(shape_.local_trace_count), 0.0);
  std::vector<double> local_perturbed_trace(
      static_cast<std::size_t>(shape_.local_trace_count), 0.0);
  std::vector<unsigned char> assigned(
      static_cast<std::size_t>(shape_.local_trace_count), 0);
  for (int element = 0; element < shape_.elements; ++element)
  {
    const mfem::FiniteElement* fe = interior_space_->GetFE(element);
    mfem::ElementTransformation* transformation =
        mesh_->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr) { return false; }
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
        fe->GetGeomType(), shape_.assembly_order);
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
          vdofs.Size() != conserved_components * shape_.scalar_face_dofs)
        return false;
      const bool element_one = ft->Elem1No == element;
      if (!element_one && ft->Elem2No != element) { return false; }
      const mfem::IntegrationRule& face_nodes = trace_fe->GetNodes();
      for (int scalar = 0; scalar < shape_.scalar_face_dofs; ++scalar)
      {
        mfem::Vector face_physical(dimension);
        ft->Face->Transform(face_nodes.IntPoint(scalar), face_physical);
        double exact[conserved_components], perturbed[conserved_components];
        CudaManufacturedState(face_physical[0], face_physical[1], exact);
        CudaPerturbedState(face_physical[0], face_physical[1], perturbed);
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = scalar + shape_.scalar_face_dofs * component;
          double sign = 1.0;
          const int dof = CudaDecodeSigned(vdofs[local], sign);
          if (dof < 0 || dof >= shape_.local_trace_count) { return false; }
          const std::size_t map_index = static_cast<std::size_t>(element) *
                  shape_.trace_dofs + trace_offset + local;
          signed_element_trace_[map_index] = vdofs[local];
          const double exact_value = sign * exact[component];
          const double perturbed_value = sign * perturbed[component];
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
          trace_fe->GetGeomType(), shape_.assembly_order);
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
      host_exact_true_trace_[static_cast<std::size_t>(entry.local_true_dof)] =
          local_exact_trace[static_cast<std::size_t>(entry.local_dof)];
      host_perturbed_true_trace_[static_cast<std::size_t>(entry.local_true_dof)] =
          local_perturbed_trace[static_cast<std::size_t>(entry.local_dof)];
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
                               conserved_components);
  const std::size_t coefficient_count =
      scalar_count * static_cast<std::size_t>(conserved_components);
  const bool source_empty = config.borrowed_source.coefficients == nullptr &&
      config.borrowed_source.scalar_count == 0 &&
      config.borrowed_source.components == conserved_components;
  bool local_valid = config.borrowed_initial.coefficients != nullptr &&
      config.borrowed_initial.scalar_count == scalar_count &&
      config.borrowed_initial.components == conserved_components &&
      coefficient_count == static_cast<std::size_t>(shape_.interior_count) &&
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
    for (int local = 0; scatter_ready && local < shape_.interior_dofs; ++local)
    {
      double sign = 1.0;
      const int dof = CudaDecodeSigned(signed_element_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local],
          sign);
      if (dof < 0 || static_cast<std::size_t>(dof) >= coefficient_count)
      {
        scatter_ready = false;
        continue;
      }
      const std::size_t destination = static_cast<std::size_t>(element) *
          shape_.interior_dofs + local;
      host_exact_interior_[destination] = sign * initial[static_cast<std::size_t>(dof)];
      // Generic data mode starts the physical current state at the supplied
      // committed state; the manufactured perturbed starting state remains
      // exclusively on the legacy path.
      host_perturbed_interior_[destination] = host_exact_interior_[destination];
      if (!source_empty)
        host_static_source_[destination] = sign *
            static_source[static_cast<std::size_t>(dof)];
    }
  if (!CollectiveContinue(scatter_ready)) { return false; }

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
          interior_vdofs.Size() != conserved_components *
              interior_element->GetDof() ||
          trace_vdofs.Size() != conserved_components * trace_element->GetDof())
        throw 0;
      mfem::Vector shape(interior_element->GetDof());
      const mfem::IntegrationRule& nodes = trace_element->GetNodes();
      for (int scalar = 0; scalar < trace_element->GetDof(); ++scalar)
      {
        const mfem::IntegrationPoint& point = nodes.IntPoint(scalar);
        transformations->SetAllIntPoints(&point);
        interior_element->CalcShape(transformations->GetElement1IntPoint(),
                                    shape);
        for (int component = 0; component < conserved_components; ++component)
        {
          double restricted = 0.0;
          for (int interior_scalar = 0;
               interior_scalar < interior_element->GetDof(); ++interior_scalar)
          {
            double sign = 1.0;
            const int dof = CudaDecodeSigned(interior_vdofs[
                interior_scalar + interior_element->GetDof() * component], sign);
            if (dof < 0 || static_cast<std::size_t>(dof) >= coefficient_count)
              throw 0;
            restricted += shape[interior_scalar] * sign *
                initial[static_cast<std::size_t>(dof)];
          }
          double sign = 1.0;
          const int dof = CudaDecodeSigned(trace_vdofs[
              scalar + trace_element->GetDof() * component], sign);
          if (dof < 0 || dof >= local_trace->Size()) { throw 0; }
          (*local_trace)[dof] = sign * restricted;
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
                               conserved_components);
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
    for (int local = 0; local_valid && local < shape_.interior_dofs; ++local)
    {
      double sign = 1.0;
      const int dof = CudaDecodeSigned(signed_element_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local],
          sign);
      local_valid = dof >= 0 &&
          static_cast<std::size_t>(dof) < coefficient_count;
    }
  if (!CollectiveContinue(local_valid)) { return false; }
  const bool copied = TransportPolicy::DeviceToHost(
      host_export_interior_.data(), source.coefficients,
      static_cast<std::size_t>(shape_.interior_count) * sizeof(double),
      CudaTransferCategory::generic_host_bridge,
      CudaTransferSite::generic_host_export) &&
      TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
  if (!CollectiveContinue(copied)) { return false; }
  for (int element = 0; element < shape_.elements; ++element)
    for (int local = 0; local < shape_.interior_dofs; ++local)
    {
      double sign = 1.0;
      const int dof = CudaDecodeSigned(signed_element_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local],
          sign);
      destination.coefficients[dof] = sign * host_export_interior_[
          static_cast<std::size_t>(element) * shape_.interior_dofs + local];
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
  view_.trial_interior = take_d(max_i);
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
  view_.status = take_d(generic ? generic_arena_layout_.elements : maximum_rank_elements);
  view_.scalar = take_d(8);
  // Append the borrowed-only stabilization slice after every frozen W8A
  // allocation.  False mode therefore preserves the exact legacy arena
  // partition, offsets, and resource evidence tokens.
  if (shape_.borrowed_host_inputs)
    view_.borrowed_lagged_llf_tau = take_d(generic ? generic_arena_layout_.face_weight :
        static_cast<std::size_t>(maximum_rank_elements) *
        faces_per_element * maximum_face_points);
  if (static_cast<std::size_t>(d - double_arena_) >
      double_arena_capacity_)
    return false;

  int* i = integer_arena_;
  view_.hash_words = reinterpret_cast<std::uint64_t*>(i);
  i += 8;
  view_.ticket_words = i;
  i += 8;
  view_.signed_element_trace = TakeArena(i, max_er);
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
  return TransportPolicy::Synchronize(CudaTransferSite::topology_upload);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BuildPersistentTrueSystem() noexcept
{
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
      ilu_.emplace();
      gmres_.emplace(*true_matrix_);
      CudaHypreSetupOrderEvidence initial_setup{};
      local_objects_ready = CudaConfigurePersistentHypre(
          *true_matrix_, *ilu_, *gmres_, ranks_, 0, initial_setup) &&
          initial_setup.ConfigurationExactForGeneration(0);
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
  // numerical values. The warmed solver objects are retired below so their
  // released pool blocks back the subsequently recreated persistent pair.
  // A deterministic nonzero RHS forces the setup path to materialize the
  // persistent solver workspace instead of accepting an identity no-op.
  *hypre_rhs_ = 1.0;
  *hypre_solution_ = 0.0;
  CudaSolveDiagnostics warm{};
  const bool warm_solved = CudaRefreshFactorAndSolve(
      *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_, *gmres_, ranks_,
      0, warm);
  const bool warmed = warm_solved && warm.device_policy &&
      warm.configured_ilu0 &&
      warm.matrix_generation == 0;
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
      rank_, "ilu-retirement", "before", false);
  ilu_.reset();
  ReportCudaWarmLifetimeMarker(
      rank_, "ilu-retirement", "after", !ilu_);

  bool persistent_configured = false;
  ReportCudaWarmLifetimeMarker(
      rank_, "persistent-recreation-configuration", "before", false);
  try
  {
    ilu_.emplace();
    gmres_.emplace(*true_matrix_);
    CudaHypreSetupOrderEvidence persistent_setup{};
    persistent_configured = CudaConfigurePersistentHypre(
        *true_matrix_, *ilu_, *gmres_, ranks_, 0, persistent_setup) &&
        persistent_setup.ConfigurationExactForGeneration(0);
  }
  catch (...) { persistent_configured = false; }
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
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS;
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
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS;
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
                     peer, 4103, MPI_COMM_WORLD,
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
                     MPI_INT64_T, peer, 4104, MPI_COMM_WORLD,
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
                        current_generation_, device_) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status))
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
  if (!TransportPolicy::DeviceToHost(
          host_status_.data(), view_.status,
          static_cast<std::size_t>(shape_.elements) * sizeof(double),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status))
    return false;
  for (int element = 0; element < shape_.elements; ++element)
    if (host_status_[static_cast<std::size_t>(element)] != 1.0) { return false; }
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
DistributeCurrentTrace(const double* true_values) noexcept
{
  if (ranks_ == 1)
    return CollectiveContinue(CudaLaunchGatherLocalTrace(
        view_.signed_element_trace,
        shape_.elements * shape_.trace_dofs, true_values,
        view_.current_local_trace) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status));
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
  const double* solution =
      static_cast<const mfem::Vector&>(*hypre_solution_).Read();
  if (ranks_ == 1)
    return CollectiveContinue(CudaLaunchGatherLocalTrace(
        view_.signed_element_trace,
        shape_.elements * shape_.trace_dofs, solution,
        view_.trace_increment_local) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status));
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
AssembleTrueVector(const double* element_values, double* true_values) noexcept
{
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
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS;
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
  return CollectiveContinue(
      TransportPolicy::Synchronize(CudaTransferSite::rhs_remote_scatter));
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleTrueSystem() noexcept
{
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
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS;
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
  if (!AssembleTrueVector(view_.condensed_rhs, view_.true_rhs_scratch))
    return false;
  true_matrix_->HypreRead();
  hypre_rhs_->HypreRead();
  return CollectiveContinue(CudaHypreDevicePolicyValid(
      *true_matrix_, *hypre_rhs_, *hypre_solution_));
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
SolveTrueSystem(double& true_relative_residual,
                const std::uint64_t matrix_generation) noexcept
{
  CudaSolveDiagnostics diagnostics{};
  const bool local_solve = CudaRefreshFactorAndSolve(
          *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_, *gmres_,
          ranks_, matrix_generation, diagnostics) &&
      diagnostics.device_policy && diagnostics.configured_ilu0 &&
      diagnostics.matrix_generation == matrix_generation &&
      diagnostics.converged && diagnostics.finite;
  if (!CollectiveContinue(local_solve))
    return false;
  true_matrix_->Mult(*hypre_solution_, *hypre_action_);
  *hypre_action_ -= *hypre_rhs_;
  const double residual_squared = mfem::InnerProduct(
      MPI_COMM_WORLD, *hypre_action_, *hypre_action_);
  const double rhs_squared = mfem::InnerProduct(
      MPI_COMM_WORLD, *hypre_rhs_, *hypre_rhs_);
  true_relative_residual = std::sqrt(residual_squared) /
      std::max(std::sqrt(rhs_squared), 1.0e-300);
  return std::isfinite(true_relative_residual) &&
      true_relative_residual <= 1.0e-10;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleAndNorm(state_slot&, double& norm) noexcept
{
  TransportPolicy::SetPhase(CudaFailurePhase::assembly, current_generation_);
  const bool local_assembly = CudaLaunchAssembly(view_, shape_, fixed_dt) &&
      TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
      StatusPassed();
  if (!CollectiveContinue(local_assembly) ||
      !AssembleTrueVector(view_.residual_g, view_.true_residual))
    return false;
  const bool norm_launched = CudaLaunchLocalSquaredNorm(
      view_.residual_r, shape_.interior_count, view_.true_residual,
      shape_.true_trace_count, view_.scalar);
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
                                  MPI_COMM_WORLD) != MPI_SUCCESS)
    return false;
  norm = std::sqrt(global_squared);
  return std::isfinite(norm);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
CopyBase(state_slot& workspace) noexcept
{
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
  const bool synchronized = TransportPolicy::Synchronize(
      CudaTransferSite::nonlinear_base_local_trace);
  return interior && trace && local_trace && synchronized;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
RestoreBase(state_slot& workspace) noexcept
{
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
SwapTrialIntoWorkspace(state_slot& workspace) noexcept
{
  const bool interior = TransportPolicy::DeviceToDevice(
      workspace.interior.coefficients, view_.trial_interior,
      workspace.interior.count * sizeof(double),
      CudaTransferCategory::attempt_history,
      CudaTransferSite::nonlinear_accept_interior);
  const bool trace = TransportPolicy::DeviceToDevice(
      workspace.trace.coefficients, view_.trial_true_trace,
      workspace.trace.count * sizeof(double),
      CudaTransferCategory::attempt_history,
      CudaTransferSite::nonlinear_accept_trace);
  const bool synchronized =
      TransportPolicy::Synchronize(CudaTransferSite::nonlinear_accept_trace);
  return interior && trace && synchronized;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
HashState(const double* values, const int count, StateHash& hash) noexcept
{
  if (!CudaLaunchHash(values, count, view_.hash_words) ||
      !TransportPolicy::Synchronize(CudaTransferSite::state_hash))
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
  const bool local_ready = HashState(
      workspace.interior.coefficients, shape_.interior_count,
      workspace.interior.local_hash) &&
      HashState(workspace.trace.coefficients, shape_.true_trace_count,
                workspace.trace.local_hash);
  if (!CollectiveContinue(local_ready)) { return false; }
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
                    MPI_COMM_WORLD) != MPI_SUCCESS)
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
