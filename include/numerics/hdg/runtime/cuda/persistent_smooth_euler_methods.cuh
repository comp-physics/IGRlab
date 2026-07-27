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
  std::array<double, maximum_local_trace_dofs> local_exact_trace{};
  std::array<double, maximum_local_trace_dofs> local_perturbed_trace{};
  std::array<unsigned char, maximum_local_trace_dofs> assigned{};
  for (int element = 0; element < shape_.elements; ++element)
  {
    const mfem::FiniteElement* fe = interior_space_->GetFE(element);
    mfem::ElementTransformation* transformation =
        mesh_->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr) { return false; }
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
    if (nodes.GetNPoints() != scalar_dofs_per_element) { return false; }
    for (int scalar = 0; scalar < scalar_dofs_per_element; ++scalar)
    {
      mfem::Vector physical(dimension);
      transformation->Transform(nodes.IntPoint(scalar), physical);
      double exact[conserved_components], perturbed[conserved_components];
      CudaManufacturedState(physical[0], physical[1], exact);
      CudaPerturbedState(physical[0], physical[1], perturbed);
      for (int component = 0; component < conserved_components; ++component)
      {
        const int local = scalar + scalar_dofs_per_element * component;
        const std::size_t index = static_cast<std::size_t>(element) *
                                      interior_dofs_per_element + local;
        host_exact_interior_[index] = exact[component];
        host_perturbed_interior_[index] = perturbed[component];
      }
    }
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        fe->GetGeomType(), assembly_quadrature_order);
    mfem::Vector shape(scalar_dofs_per_element), physical(dimension);
    mfem::DenseMatrix derivative(scalar_dofs_per_element, dimension);
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
      for (int scalar = 0; scalar < scalar_dofs_per_element; ++scalar)
      {
        host_volume_shape_[quadrature * scalar_dofs_per_element + scalar] =
            shape[scalar];
        for (int direction = 0; direction < dimension; ++direction)
          host_volume_derivative_[
              (quadrature * scalar_dofs_per_element + scalar) * dimension +
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
          vdofs.Size() != conserved_components * scalar_dofs_per_face)
        return false;
      const bool element_one = ft->Elem1No == element;
      if (!element_one && ft->Elem2No != element) { return false; }
      const mfem::IntegrationRule& face_nodes = trace_fe->GetNodes();
      for (int scalar = 0; scalar < scalar_dofs_per_face; ++scalar)
      {
        mfem::Vector face_physical(dimension);
        ft->Face->Transform(face_nodes.IntPoint(scalar), face_physical);
        double exact[conserved_components], perturbed[conserved_components];
        CudaManufacturedState(face_physical[0], face_physical[1], exact);
        CudaPerturbedState(face_physical[0], face_physical[1], perturbed);
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = scalar + scalar_dofs_per_face * component;
          double sign = 1.0;
          const int dof = CudaDecodeSigned(vdofs[local], sign);
          if (dof < 0 || dof >= shape_.local_trace_count) { return false; }
          const std::size_t map_index = static_cast<std::size_t>(element) *
                  trace_dofs_per_element + trace_offset + local;
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
          trace_fe->GetGeomType(), assembly_quadrature_order);
      mfem::Vector interior_shape(scalar_dofs_per_element);
      mfem::Vector trace_shape(scalar_dofs_per_face);
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
        for (int scalar = 0; scalar < scalar_dofs_per_element; ++scalar)
          host_face_interior_shape_[
              quadrature * scalar_dofs_per_element + scalar] =
              interior_shape[scalar];
        for (int scalar = 0; scalar < scalar_dofs_per_face; ++scalar)
          host_face_trace_shape_[quadrature * scalar_dofs_per_face + scalar] =
              trace_shape[scalar];
      }
      trace_offset += vdofs.Size();
    }
    if (trace_offset != trace_dofs_per_element) { return false; }
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
BuildLocalTraceGraph() noexcept
{
  try
  {
    local_trace_matrix_.emplace(shape_.local_trace_count);
    for (int element = 0; element < shape_.elements; ++element)
      for (int row = 0; row < trace_dofs_per_element; ++row)
      {
        double row_sign = 1.0;
        const int local_row = CudaDecodeSigned(
            signed_element_trace_[static_cast<std::size_t>(element) *
                                      trace_dofs_per_element + row],
            row_sign);
        for (int column = 0; column < trace_dofs_per_element; ++column)
        {
          double column_sign = 1.0;
          const int local_column = CudaDecodeSigned(
              signed_element_trace_[static_cast<std::size_t>(element) *
                                        trace_dofs_per_element + column],
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
  constexpr std::size_t max_i =
      maximum_rank_elements * interior_dofs_per_element;
  constexpr std::size_t max_lt = maximum_local_trace_dofs;
  constexpr std::size_t max_tt = maximum_true_trace_dofs;
  constexpr std::size_t max_er =
      maximum_rank_elements * trace_dofs_per_element;
  constexpr std::size_t max_a = maximum_rank_elements *
      interior_dofs_per_element * interior_dofs_per_element;
  constexpr std::size_t max_b = maximum_rank_elements *
      interior_dofs_per_element * trace_dofs_per_element;
  constexpr std::size_t max_d = maximum_local_matrix_entries;
  view_.slot_interior[0] = take_d(max_i);
  view_.slot_interior[1] = take_d(max_i);
  view_.old_interior = take_d(max_i);
  view_.work_interior = take_d(max_i);
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
  view_.solve_workspace = take_d(maximum_rank_elements *
      interior_dofs_per_element * (trace_dofs_per_element + 1));
  view_.schur = take_d(max_d);
  view_.condensed_rhs = take_d(max_er);
  view_.true_residual = take_d(max_tt);
  view_.remote_matrix_send = take_d(maximum_exchange_entries);
  view_.remote_matrix_receive = take_d(maximum_exchange_entries);
  view_.remote_vector_send = take_d(max_er);
  view_.remote_vector_receive = take_d(max_er);
  view_.volume_shape = take_d(maximum_rank_elements * maximum_volume_points *
                              scalar_dofs_per_element);
  view_.volume_derivative = take_d(maximum_rank_elements *
      maximum_volume_points * scalar_dofs_per_element * dimension);
  view_.volume_weight = take_d(maximum_rank_elements * maximum_volume_points);
  view_.volume_position = take_d(maximum_rank_elements *
                                 maximum_volume_points * dimension);
  view_.face_interior_shape = take_d(maximum_rank_elements *
      faces_per_element * maximum_face_points * scalar_dofs_per_element);
  view_.face_trace_shape = take_d(maximum_rank_elements * faces_per_element *
                                  maximum_face_points * scalar_dofs_per_face);
  view_.face_weight = take_d(maximum_rank_elements * faces_per_element *
                             maximum_face_points);
  view_.face_normal = take_d(maximum_rank_elements * faces_per_element *
                             maximum_face_points * dimension);
  view_.status = take_d(maximum_rank_elements);
  view_.scalar = take_d(8);
  if (static_cast<std::size_t>(d - double_arena_) >
      persistent_double_capacity)
    return false;

  int* i = integer_arena_;
  view_.hash_words = reinterpret_cast<std::uint64_t*>(i);
  i += 8;
  view_.ticket_words = i;
  i += 8;
  view_.signed_element_trace = TakeArena(i,
      maximum_rank_elements * trace_dofs_per_element);
  view_.matrix_owned_source = TakeArena(i, maximum_local_matrix_entries);
  view_.matrix_owned_section = TakeArena(i, maximum_local_matrix_entries);
  view_.matrix_owned_target = TakeArena(i, maximum_local_matrix_entries);
  view_.matrix_send_source = TakeArena(i, maximum_exchange_entries);
  view_.matrix_receive_section = TakeArena(i, maximum_exchange_entries);
  view_.matrix_receive_target = TakeArena(i, maximum_exchange_entries);
  view_.vector_owned_source = TakeArena(i, max_er);
  view_.vector_owned_target = TakeArena(i, max_er);
  view_.vector_send_source = TakeArena(i, max_er);
  view_.vector_receive_target = TakeArena(i, max_er);
  if (static_cast<std::size_t>(i - integer_arena_) >
      persistent_integer_capacity)
    return false;

  double_memory_.Wrap(double_arena_,
                      static_cast<int>(persistent_double_capacity),
                      mfem::MemoryType::DEVICE_UMPIRE, false);
  integer_memory_.Wrap(integer_arena_,
                       static_cast<int>(persistent_integer_capacity),
                       mfem::MemoryType::DEVICE_UMPIRE, false);
  double_arena_vector_.NewMemoryAndSize(
      double_memory_, static_cast<int>(persistent_double_capacity), false);
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
      !TransportPolicy::HostToDevice(
          view_.initial_current_true_trace,
          host_perturbed_true_trace_.data(),
          static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double),
          CudaTransferCategory::initial_state,
          CudaTransferSite::initial_trace_upload) ||
      !h2d(view_.signed_element_trace, signed_element_trace_.data(),
           static_cast<std::size_t>(shape_.elements) *
               trace_dofs_per_element * sizeof(int),
           CudaTransferSite::map_upload) ||
      !h2d(view_.volume_shape, host_volume_shape_.data(),
           static_cast<std::size_t>(shape_.elements) * shape_.volume_points *
               scalar_dofs_per_element * sizeof(double),
           CudaTransferSite::quadrature_upload) ||
      !h2d(view_.volume_derivative, host_volume_derivative_.data(),
           static_cast<std::size_t>(shape_.elements) * shape_.volume_points *
               scalar_dofs_per_element * dimension * sizeof(double),
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
               shape_.face_points * scalar_dofs_per_element * sizeof(double),
           CudaTransferSite::quadrature_upload) ||
      !h2d(view_.face_trace_shape, host_face_trace_shape_.data(),
           static_cast<std::size_t>(shape_.elements) * faces_per_element *
               shape_.face_points * scalar_dofs_per_face * sizeof(double),
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
  try
  {
    mfem::ParBilinearForm form(&*trace_space_);
    true_matrix_.reset(form.ParallelAssemble(&*local_trace_matrix_));
    if (!true_matrix_ ||
        !CudaGetTrueMatrixStructure(*true_matrix_, owned_global_true_.data(),
                                    shape_.true_trace_count, true_structure_,
                                    &TransportPolicy::DeviceToHost,
                                    &TransportPolicy::Synchronize) ||
        !CudaGetTrueMatrixDeviceView(*true_matrix_, true_device_))
      return false;
    trace_slots_[0].emplace(trace_space_.operator->());
    trace_slots_[1].emplace(trace_space_.operator->());
    hypre_rhs_.emplace(*true_matrix_);
    hypre_solution_.emplace(*true_matrix_);
    hypre_action_.emplace(*true_matrix_);
    ilu_.emplace();
    gmres_.emplace(*true_matrix_);
    if (!CudaConfigurePersistentHypre(*true_matrix_, *ilu_, *gmres_, ranks_))
      return false;
    for (auto& slot : trace_slots_)
    {
      slot->UseDevice(true);
      slot->HypreReadWrite();
      double* data = static_cast<mfem::Vector&>(*slot).ReadWrite();
      if (!TransportPolicy::HostToDevice(
              data, host_exact_true_trace_.data(),
              static_cast<std::size_t>(shape_.true_trace_count) *
                  sizeof(double),
              CudaTransferCategory::initial_state,
              CudaTransferSite::initial_trace_upload))
        return false;
    }
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
  catch (...)
  {
    return false;
  }
  if (!BuildSparseStagingPlan() ||
      !TransportPolicy::Synchronize(CudaTransferSite::initial_trace_upload))
    return false;
  // Warm the persistent Krylov/factor workspaces before the allocation
  // baseline. The topology matrix is deliberately diagonally dominant here;
  // each real generation overwrites only its already-fixed numerical values.
  *hypre_rhs_ = 0.0;
  *hypre_solution_ = 0.0;
  CudaSolveDiagnostics warm{};
  return CudaRefreshFactorAndSolve(
      *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_, *gmres_, ranks_,
      warm) && warm.device_policy && warm.configured_ilu0;
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

  for (int element = 0; element < shape_.elements; ++element)
    for (int row = 0; row < trace_dofs_per_element; ++row)
    {
      double row_sign = 1.0;
      const int row_local = CudaDecodeSigned(
          signed_element_trace_[static_cast<std::size_t>(element) *
                                    trace_dofs_per_element + row], row_sign);
      const std::int64_t row_global =
          local_global_true_[static_cast<std::size_t>(row_local)];
      const int row_owner = local_owner_[static_cast<std::size_t>(row_local)];
      const int vector_source = element * trace_dofs_per_element + row;
      if (row_owner == rank_)
      {
        const int true_row = find_row(row_global);
        if (true_row < 0) { return false; }
        host_vector_owned_source_[static_cast<std::size_t>(
            shape_.vector_owned_count)] = vector_source;
        host_vector_owned_target_[static_cast<std::size_t>(
            shape_.vector_owned_count)] = true_row;
        ++shape_.vector_owned_count;
      }
      else
      {
        host_vector_send_source_[static_cast<std::size_t>(
            shape_.vector_send_count)] = vector_source;
        vector_send_keys_[static_cast<std::size_t>(shape_.vector_send_count)] =
            row_global;
        ++shape_.vector_send_count;
      }
      for (int column = 0; column < trace_dofs_per_element; ++column)
      {
        double column_sign = 1.0;
        const int column_local = CudaDecodeSigned(
            signed_element_trace_[static_cast<std::size_t>(element) *
                                      trace_dofs_per_element + column],
            column_sign);
        const std::int64_t column_global =
            local_global_true_[static_cast<std::size_t>(column_local)];
        const int source = (element * trace_dofs_per_element + row) *
                               trace_dofs_per_element + column;
        if (row_owner == rank_)
        {
          int section = -1, target = -1;
          if (!find_target(row_global, column_global, section, target))
            return false;
          host_matrix_owned_source_[static_cast<std::size_t>(
              shape_.matrix_owned_count)] = source;
          host_matrix_owned_section_[static_cast<std::size_t>(
              shape_.matrix_owned_count)] = section;
          host_matrix_owned_target_[static_cast<std::size_t>(
              shape_.matrix_owned_count)] = target;
          ++shape_.matrix_owned_count;
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

  if (ranks_ == 2)
  {
    const int peer = 1 - rank_;
    if (MPI_Sendrecv(&shape_.matrix_send_count, 1, MPI_INT, peer, 4101,
                     &shape_.matrix_receive_count, 1, MPI_INT, peer, 4101,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS ||
        MPI_Sendrecv(&shape_.vector_send_count, 1, MPI_INT, peer, 4102,
                     &shape_.vector_receive_count, 1, MPI_INT, peer, 4102,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS ||
        shape_.matrix_receive_count > maximum_exchange_entries ||
        shape_.vector_receive_count >
            maximum_rank_elements * trace_dofs_per_element)
      return false;
    if (MPI_Sendrecv(matrix_send_keys_.data(), 2 * shape_.matrix_send_count,
                     MPI_INT64_T, peer, 4103,
                     matrix_receive_keys_.data(),
                     2 * shape_.matrix_receive_count, MPI_INT64_T,
                     peer, 4103, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE) != MPI_SUCCESS ||
        MPI_Sendrecv(vector_send_keys_.data(), shape_.vector_send_count,
                     MPI_INT64_T, peer, 4104,
                     vector_receive_keys_.data(), shape_.vector_receive_count,
                     MPI_INT64_T, peer, 4104, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE) != MPI_SUCCESS)
      return false;
    for (int entry = 0; entry < shape_.matrix_receive_count; ++entry)
      if (!find_target(matrix_receive_keys_[2 * static_cast<std::size_t>(entry)],
                       matrix_receive_keys_[2 * static_cast<std::size_t>(entry) + 1],
                       host_matrix_receive_section_[static_cast<std::size_t>(entry)],
                       host_matrix_receive_target_[static_cast<std::size_t>(entry)]))
        return false;
    for (int entry = 0; entry < shape_.vector_receive_count; ++entry)
    {
      const int target = find_row(
          vector_receive_keys_[static_cast<std::size_t>(entry)]);
      if (target < 0) { return false; }
      host_vector_receive_target_[static_cast<std::size_t>(entry)] = target;
    }
    TransportPolicy::RecordMpiStaging(
        4 * sizeof(int) +
            2 * static_cast<std::size_t>(shape_.matrix_send_count +
                                         shape_.matrix_receive_count) *
                sizeof(std::int64_t) +
            static_cast<std::size_t>(shape_.vector_send_count +
                                     shape_.vector_receive_count) *
                sizeof(std::int64_t),
        CudaTransferSite::map_upload);
  }

  const auto upload = [](int* destination, const int* source, int count)
  {
    return TransportPolicy::HostToDevice(
        destination, source, static_cast<std::size_t>(count) * sizeof(int),
        CudaTransferCategory::topology, CudaTransferSite::map_upload);
  };
  return upload(view_.matrix_owned_source, host_matrix_owned_source_.data(),
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
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BindStateSlots() noexcept
{
  double* trace0 = static_cast<mfem::Vector&>(*trace_slots_[0]).ReadWrite();
  double* trace1 = static_cast<mfem::Vector&>(*trace_slots_[1]).ReadWrite();
  committed_slot_.interior = {view_.slot_interior[0],
                              static_cast<std::size_t>(shape_.interior_count),
                              {}, 0,
                              reinterpret_cast<std::uintptr_t>(
                                  view_.slot_interior[0]), true};
  candidate_slot_.interior = {view_.slot_interior[1],
                              static_cast<std::size_t>(shape_.interior_count),
                              {}, 0,
                              reinterpret_cast<std::uintptr_t>(
                                  view_.slot_interior[1]), true};
  committed_slot_.trace = {trace0,
                           static_cast<std::size_t>(shape_.true_trace_count),
                           {}, 0, reinterpret_cast<std::uintptr_t>(trace0),
                           reinterpret_cast<std::uintptr_t>(&*trace_slots_[0]),
                           true, true};
  candidate_slot_.trace = {trace1,
                           static_cast<std::size_t>(shape_.true_trace_count),
                           {}, 0, reinterpret_cast<std::uintptr_t>(trace1),
                           reinterpret_cast<std::uintptr_t>(&*trace_slots_[1]),
                           true, true};
  workspace_slot_.interior = {view_.work_interior,
                              static_cast<std::size_t>(shape_.interior_count),
                              {}, 0,
                              reinterpret_cast<std::uintptr_t>(
                                  view_.work_interior), true};
  workspace_slot_.trace = {
      view_.work_true_trace,
      static_cast<std::size_t>(shape_.true_trace_count), {}, 0,
      reinterpret_cast<std::uintptr_t>(view_.work_true_trace), 0, false, true};
  return HashState(committed_slot_.interior.coefficients,
                   shape_.interior_count, committed_slot_.interior.hash) &&
      HashState(candidate_slot_.interior.coefficients, shape_.interior_count,
                candidate_slot_.interior.hash) &&
      HashState(committed_slot_.trace.coefficients, shape_.true_trace_count,
                committed_slot_.trace.hash) &&
      HashState(candidate_slot_.trace.coefficients, shape_.true_trace_count,
                candidate_slot_.trace.hash);
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
  const bool valid = words[0] == 1 && words[1] == 1200 &&
      words[2] == device_ && words[3] == static_cast<int>(phase) &&
      static_cast<std::uint32_t>(words[4]) ==
          static_cast<std::uint32_t>(current_attempt_.sequence) &&
      static_cast<std::uint32_t>(words[6]) ==
          static_cast<std::uint32_t>(current_generation_);
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
  std::array<double, maximum_rank_elements> status{};
  if (!TransportPolicy::DeviceToHost(
          status.data(), view_.status,
          static_cast<std::size_t>(shape_.elements) * sizeof(double),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status))
    return false;
  for (int element = 0; element < shape_.elements; ++element)
    if (status[static_cast<std::size_t>(element)] != 1.0) { return false; }
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
DistributeCurrentTrace(const double* true_values) noexcept
{
  if (ranks_ == 1)
    return CudaLaunchGatherLocalTrace(
        view_.signed_element_trace,
        shape_.elements * trace_dofs_per_element, true_values,
        view_.current_local_trace) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
  double* owned_host = owned_stage_.HostWrite();
  if (!TransportPolicy::DeviceToHost(
          owned_host, true_values,
          static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double),
          CudaTransferCategory::host_staged_trace_exchange,
          CudaTransferSite::trace_solution_stage_out) ||
      !TransportPolicy::Synchronize(
          CudaTransferSite::trace_solution_stage_out) ||
      !exchange_plan_.Distribute(*trace_space_, owned_stage_, local_stage_))
    return false;
  const std::size_t local_bytes =
      static_cast<std::size_t>(shape_.local_trace_count) * sizeof(double);
  TransportPolicy::RecordMpiStaging(
      static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double) +
          local_bytes,
      CudaTransferSite::trace_ghost_stage_in);
  return TransportPolicy::HostToDevice(
             view_.distributed_local_trace, local_stage_.HostRead(), local_bytes,
             CudaTransferCategory::host_staged_trace_exchange,
             CudaTransferSite::trace_ghost_stage_in) &&
      CudaLaunchGatherLocalTrace(
          view_.signed_element_trace,
          shape_.elements * trace_dofs_per_element,
          view_.distributed_local_trace, view_.current_local_trace) &&
      TransportPolicy::Synchronize(CudaTransferSite::trace_ghost_stage_in);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
DistributeTraceIncrement() noexcept
{
  const double* solution =
      static_cast<const mfem::Vector&>(*hypre_solution_).Read();
  if (ranks_ == 1)
    return CudaLaunchGatherLocalTrace(
        view_.signed_element_trace,
        shape_.elements * trace_dofs_per_element, solution,
        view_.trace_increment_local) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
  double* owned_host = owned_stage_.HostWrite();
  if (!TransportPolicy::DeviceToHost(
          owned_host, solution,
          static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double),
          CudaTransferCategory::host_staged_trace_exchange,
          CudaTransferSite::trace_solution_stage_out) ||
      !TransportPolicy::Synchronize(
          CudaTransferSite::trace_solution_stage_out) ||
      !exchange_plan_.Distribute(*trace_space_, owned_stage_, local_stage_))
    return false;
  // local_stage is in local physical signed coefficients. Gather the element
  // side orientation in a second device kernel from the local array.
  const std::size_t local_bytes =
      static_cast<std::size_t>(shape_.local_trace_count) * sizeof(double);
  if (!TransportPolicy::HostToDevice(
          view_.distributed_local_trace, local_stage_.HostRead(), local_bytes,
          CudaTransferCategory::host_staged_trace_exchange,
          CudaTransferSite::trace_ghost_stage_in) ||
      !CudaLaunchGatherLocalTrace(
          view_.signed_element_trace,
          shape_.elements * trace_dofs_per_element,
          view_.distributed_local_trace, view_.trace_increment_local))
    return false;
  TransportPolicy::RecordMpiStaging(
      static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double) +
          local_bytes,
      CudaTransferSite::trace_ghost_stage_in);
  return TransportPolicy::Synchronize(CudaTransferSite::trace_ghost_stage_in);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleTrueVector(const double* element_values, double* true_values) noexcept
{
  if (!CudaLaunchAssembleOwnedVector(view_, shape_, element_values,
                                     true_values) ||
      !CudaLaunchGatherRemoteVector(view_, shape_, element_values))
    return false;
  if (ranks_ == 2)
  {
    const int peer = 1 - rank_;
    const std::size_t send_bytes =
        static_cast<std::size_t>(shape_.vector_send_count) * sizeof(double);
    const std::size_t receive_bytes =
        static_cast<std::size_t>(shape_.vector_receive_count) * sizeof(double);
    if (!TransportPolicy::DeviceToHost(
            vector_send_host_.data(), view_.remote_vector_send, send_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::rhs_remote_gather) ||
        !TransportPolicy::Synchronize(CudaTransferSite::rhs_remote_gather) ||
        MPI_Sendrecv(vector_send_host_.data(), shape_.vector_send_count,
                     MPI_DOUBLE, peer, 4201, vector_receive_host_.data(),
                     shape_.vector_receive_count, MPI_DOUBLE, peer, 4201,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS ||
        !TransportPolicy::HostToDevice(
            view_.remote_vector_receive, vector_receive_host_.data(),
            receive_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::rhs_remote_scatter) ||
        !CudaLaunchScatterRemoteVector(view_, shape_, true_values))
      return false;
    TransportPolicy::RecordMpiStaging(send_bytes + receive_bytes,
                                      CudaTransferSite::rhs_remote_scatter);
  }
  return TransportPolicy::Synchronize(CudaTransferSite::rhs_remote_scatter);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleTrueSystem() noexcept
{
  if (!CudaLaunchZeroTrueSystem(true_device_, view_.true_rhs_scratch,
                                shape_.true_trace_count) ||
      !CudaLaunchAssembleOwnedMatrix(view_, true_device_, shape_) ||
      !CudaLaunchGatherRemoteMatrix(view_, shape_))
    return false;
  if (ranks_ == 2)
  {
    const int peer = 1 - rank_;
    const std::size_t send_bytes =
        static_cast<std::size_t>(shape_.matrix_send_count) * sizeof(double);
    const std::size_t receive_bytes =
        static_cast<std::size_t>(shape_.matrix_receive_count) * sizeof(double);
    if (!TransportPolicy::DeviceToHost(
            matrix_send_host_.data(), view_.remote_matrix_send, send_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::matrix_remote_gather) ||
        !TransportPolicy::Synchronize(CudaTransferSite::matrix_remote_gather) ||
        MPI_Sendrecv(matrix_send_host_.data(), shape_.matrix_send_count,
                     MPI_DOUBLE, peer, 4202, matrix_receive_host_.data(),
                     shape_.matrix_receive_count, MPI_DOUBLE, peer, 4202,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS ||
        !TransportPolicy::HostToDevice(
            view_.remote_matrix_receive, matrix_receive_host_.data(),
            receive_bytes,
            CudaTransferCategory::host_staged_trace_exchange,
            CudaTransferSite::matrix_remote_scatter) ||
        !CudaLaunchScatterRemoteMatrix(view_, true_device_, shape_))
      return false;
    TransportPolicy::RecordMpiStaging(send_bytes + receive_bytes,
                                      CudaTransferSite::matrix_remote_scatter);
  }
  if (!AssembleTrueVector(view_.condensed_rhs, view_.true_rhs_scratch))
    return false;
  true_matrix_->HypreRead();
  hypre_rhs_->HypreRead();
  return CudaHypreDevicePolicyValid(*true_matrix_, *hypre_rhs_,
                                    *hypre_solution_);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
SolveTrueSystem(double& true_relative_residual) noexcept
{
  CudaSolveDiagnostics diagnostics{};
  if (!CudaRefreshFactorAndSolve(*true_matrix_, *hypre_rhs_,
                                 *hypre_solution_, *ilu_, *gmres_, ranks_,
                                 diagnostics) ||
      !diagnostics.device_policy || !diagnostics.configured_ilu0 ||
      !diagnostics.converged || !diagnostics.finite)
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
  if (!CudaLaunchAssembly(view_, shape_, fixed_dt) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status) ||
      !StatusPassed() ||
      !AssembleTrueVector(view_.residual_g, view_.true_residual) ||
      !CudaLaunchLocalSquaredNorm(view_.residual_r, shape_.interior_count,
                                  view_.true_residual,
                                  shape_.true_trace_count, view_.scalar))
    return false;
  double local_squared = 0.0;
  if (!TransportPolicy::DeviceToHost(
          &local_squared, view_.scalar, sizeof(double),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status))
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
  return TransportPolicy::DeviceToDevice(
             view_.base_interior, workspace.interior.coefficients,
             workspace.interior.count * sizeof(double),
             CudaTransferCategory::state_prepare,
             CudaTransferSite::candidate_prepare_interior) &&
      TransportPolicy::DeviceToDevice(
          view_.base_true_trace, workspace.trace.coefficients,
          workspace.trace.count * sizeof(double),
          CudaTransferCategory::state_prepare,
          CudaTransferSite::candidate_prepare_trace) &&
      TransportPolicy::DeviceToDevice(
          view_.base_local_trace, view_.current_local_trace,
          static_cast<std::size_t>(shape_.elements) *
              trace_dofs_per_element * sizeof(double),
          CudaTransferCategory::state_prepare,
          CudaTransferSite::candidate_prepare_trace);
}

template <typename AllocatorPolicy, typename TransportPolicy>
void CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
RestoreBase(state_slot& workspace) noexcept
{
  (void)TransportPolicy::DeviceToDevice(
      workspace.interior.coefficients, view_.base_interior,
      workspace.interior.count * sizeof(double),
      CudaTransferCategory::state_prepare,
      CudaTransferSite::candidate_prepare_interior);
  (void)TransportPolicy::DeviceToDevice(
      workspace.trace.coefficients, view_.base_true_trace,
      workspace.trace.count * sizeof(double),
      CudaTransferCategory::state_prepare,
      CudaTransferSite::candidate_prepare_trace);
  (void)TransportPolicy::DeviceToDevice(
      view_.current_local_trace, view_.base_local_trace,
      static_cast<std::size_t>(shape_.elements) *
          trace_dofs_per_element * sizeof(double),
      CudaTransferCategory::state_prepare,
      CudaTransferSite::candidate_prepare_trace);
}

template <typename AllocatorPolicy, typename TransportPolicy>
void CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
SwapTrialIntoWorkspace(state_slot& workspace) noexcept
{
  (void)TransportPolicy::DeviceToDevice(
      workspace.interior.coefficients, view_.trial_interior,
      workspace.interior.count * sizeof(double),
      CudaTransferCategory::state_store,
      CudaTransferSite::candidate_store_interior);
  (void)TransportPolicy::DeviceToDevice(
      workspace.trace.coefficients, view_.trial_true_trace,
      workspace.trace.count * sizeof(double),
      CudaTransferCategory::state_store,
      CudaTransferSite::candidate_store_trace);
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
  return HashState(workspace.interior.coefficients, shape_.interior_count,
                   workspace.interior.hash) &&
      HashState(workspace.trace.coefficients, shape_.true_trace_count,
                workspace.trace.hash);
}

template <typename AllocatorPolicy, typename TransportPolicy>
std::size_t CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
WorkspaceBytes() const noexcept
{
  return (persistent_double_capacity -
          2 * maximum_rank_elements * interior_dofs_per_element -
          2 * maximum_true_trace_dofs) * sizeof(double) +
      persistent_integer_capacity * sizeof(int);
}

template <typename AllocatorPolicy, typename TransportPolicy>
void CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
CaptureOracleMetadata(const AttemptIdentity& attempt,
                      const std::uint64_t generation) noexcept
{
  oracle_ = {};
  oracle_.rank = rank_;
  oracle_.ranks = ranks_;
  oracle_.elements = shape_.elements;
  oracle_.interior_count = shape_.interior_count;
  oracle_.local_trace_count = shape_.local_trace_count;
  oracle_.true_trace_count = shape_.true_trace_count;
  oracle_.global_true_trace_count = shape_.global_true_trace_count;
  oracle_.volume_points = shape_.volume_points;
  oracle_.face_points = shape_.face_points;
  oracle_.attempt_sequence = attempt.sequence;
  oracle_.product_generation = generation;
  std::copy_n(signed_element_trace_.begin(),
              shape_.elements * trace_dofs_per_element,
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
  std::array<double, maximum_true_matrix_entries> diag{};
  std::array<double, maximum_true_matrix_entries> offd{};
  if (true_structure_.diag_nonzeros + true_structure_.offd_nonzeros >
          maximum_true_matrix_entries ||
      !TransportPolicy::DeviceToHost(
          diag.data(), true_device_.diag_values,
          static_cast<std::size_t>(true_structure_.diag_nonzeros) *
              sizeof(double),
          CudaTransferCategory::oracle_export,
          CudaTransferSite::oracle_export) ||
      !TransportPolicy::DeviceToHost(
          offd.data(), true_device_.offd_values,
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
          diag[static_cast<std::size_t>(entry)];
    }
    for (int entry = true_structure_.offd_offsets[static_cast<std::size_t>(row)];
         entry < true_structure_.offd_offsets[static_cast<std::size_t>(row + 1)];
         ++entry)
    {
      pod.true_csr_columns[static_cast<std::size_t>(cursor)] =
          true_structure_.offd_columns[static_cast<std::size_t>(entry)];
      pod.true_csr_values[static_cast<std::size_t>(cursor++)] =
          offd[static_cast<std::size_t>(entry)];
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
