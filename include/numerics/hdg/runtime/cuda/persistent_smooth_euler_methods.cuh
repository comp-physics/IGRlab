#include "igrlab/compat.hpp"
/* Included inside dg::hdg::runtime::cuda_w8a. */

namespace detail
{

// Explicit research SIMT contractions preserve the tensor engine's dataflow
// and solver policy but issue no volume MMA instructions. Keep the default
// accounting contract unchanged for all other routes.
inline bool CudaAccumulateResearchVolumeWork(
    const bool tensor_selected, const bool borrowed_host_inputs,
    const std::uint64_t invocation_mma_tiles,
    std::uint64_t& volume_assembly_launches,
    std::uint64_t& volume_mma_tiles) noexcept
{
  if (!CudaResearchVolumeUsesSimt(tensor_selected && borrowed_host_inputs))
    return AccumulateTensorVolumeWork(
        tensor_selected, invocation_mma_tiles,
        volume_assembly_launches, volume_mma_tiles);
  if (invocation_mma_tiles != 0 || volume_mma_tiles != 0 ||
      volume_assembly_launches == std::numeric_limits<std::uint64_t>::max())
    return false;
  ++volume_assembly_launches;
  return true;
}

// Qualification of a supplied RODAS3 entry state uses the same normalized
// algebraic correction as the CPU leaf. Keep the reduction private to this
// implementation: it is one scalar gate, not retained diagnostic history.
static __global__ void CudaRodasNormalizedEntryCorrectionSquaredKernel(
    const CudaDeviceView view, const CudaMeshShape shape,
    double* const result)
{
  __shared__ double partial[256];
  const int thread = threadIdx.x;
  double sum = 0.0;
  const int auxiliary_components =
      shape.interior_components - conserved_components;
  const int auxiliary_offset =
      conserved_components * shape.scalar_interior_dofs;
  for (int element = 0; element < shape.elements; ++element)
    for (int local = thread;
         local < auxiliary_components * shape.scalar_interior_dofs;
         local += blockDim.x)
    {
      const int index = element * shape.interior_dofs +
          auxiliary_offset + local;
      const double before = view.rodas_combination_interior[index];
      const double after = view.work_interior[index];
      const double scale = fmax(1.0, fmax(fabs(before), fabs(after)));
      const double correction = (after - before) / scale;
      sum += correction * correction;
    }
  for (int index = thread; index < shape.true_trace_count;
       index += blockDim.x)
  {
    const double before = view.rodas_combination_true_trace[index];
    const double after = view.work_true_trace[index];
    const double scale = fmax(1.0, fmax(fabs(before), fabs(after)));
    const double correction = (after - before) / scale;
    sum += correction * correction;
  }
  partial[thread] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
  {
    if (thread < offset) partial[thread] += partial[thread + offset];
    __syncthreads();
  }
  if (thread == 0) result[0] = partial[0];
}

inline bool CudaLaunchRodasNormalizedEntryCorrectionSquared(
    const CudaDeviceView& view, const CudaMeshShape& shape,
    double* const result) noexcept
{
  if (!shape.borrowed_host_inputs || shape.elements <= 0 ||
      shape.scalar_interior_dofs <= 0 ||
      shape.interior_components < conserved_components ||
      shape.true_trace_count <= 0 ||
      view.rodas_combination_interior == nullptr ||
      view.rodas_combination_true_trace == nullptr ||
      view.work_interior == nullptr || view.work_true_trace == nullptr ||
      result == nullptr)
    return false;
  CudaRodasNormalizedEntryCorrectionSquaredKernel<<<1, 256>>>(
      view, shape, result);
  return cudaPeekAtLastError() == cudaSuccess;
}

inline int CudaDecodeSigned(const int encoded, double& sign) noexcept
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
}

// Temporary host setup scratch, never retained by an engine. CSR row ranges
// are disjoint, so an index outside the current row is a stale lookup entry;
// no O(number of true rows) clear is needed when switching rows.
class CudaSparseOwnedRowLookup
{
public:
  static constexpr std::size_t maximum_bytes = 16 * 1024 * 1024;

  bool Initialize(const int columns) noexcept
  {
    targets_.clear();
    begin_ = end_ = -1;
    if (columns <= 0 || static_cast<std::size_t>(columns) >
                            maximum_bytes / sizeof(int))
      return false;
    try
    {
      targets_.assign(static_cast<std::size_t>(columns), -1);
      return true;
    }
    catch (...)
    {
      return false;
    }
  }

  bool Admits(const int column) const noexcept
  {
    return column >= 0 &&
        static_cast<std::size_t>(column) < targets_.size();
  }

  int Find(const std::vector<int>& columns, const int begin,
           const int end, const int column) noexcept
  {
    if (!Admits(column)) return -1;
    if (begin != begin_ || end != end_)
    {
      // Last write wins: walking backwards retains the first CSR occurrence,
      // including a diagonal-first entry and any unsorted duplicate columns.
      for (int entry = end; entry > begin;)
      {
        --entry;
        const int owned_column = columns[static_cast<std::size_t>(entry)];
        if (Admits(owned_column))
          targets_[static_cast<std::size_t>(owned_column)] = entry;
      }
      begin_ = begin;
      end_ = end;
    }
    const int target = targets_[static_cast<std::size_t>(column)];
    return target >= begin && target < end ? target : -1;
  }

private:
  std::vector<int> targets_{};
  int begin_ = -1;
  int end_ = -1;
};

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
  // Borrowed input initialization below replaces every interior and owned
  // true-trace coefficient before either state slot is uploaded. Its trace
  // restriction depends on the supplied interior, never the manufactured
  // seed. Keep the topology/map checks here without evaluating unused data.
  const bool manufactured_initial_state = !shape_.borrowed_host_inputs;
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
  if (positivity_limiter_enabled_)
  {
    const mfem::FiniteElement* const reference_element =
        interior_space_->GetFE(0);
    const mfem::IntegrationRule* const vertices = reference_element
        ? mfem::Geometries.GetVertices(reference_element->GetGeomType())
        : nullptr;
    if (reference_element == nullptr || vertices == nullptr ||
        vertices->GetNPoints() != 4 ||
        host_limiter_vertex_shape_.size() !=
            static_cast<std::size_t>(4 * shape_.scalar_interior_dofs))
      return false;
    mfem::Vector vertex_shape(shape_.scalar_interior_dofs);
    for (int vertex = 0; vertex < 4; ++vertex)
    {
      reference_element->CalcShape(vertices->IntPoint(vertex), vertex_shape);
      for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
      {
        const double value = vertex_shape[scalar];
        if (!std::isfinite(value)) return false;
        host_limiter_vertex_shape_[
            static_cast<std::size_t>(vertex) *
                shape_.scalar_interior_dofs + scalar] = value;
      }
    }
  }
  const mfem::FiniteElement* reference_shape_element = nullptr;
  std::vector<double> reference_volume_shape(
      static_cast<std::size_t>(shape_.volume_points) *
          shape_.scalar_interior_dofs);
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
    if (manufactured_initial_state)
    {
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
    }
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        fe->GetGeomType(), shape_.volume_assembly_order);
    const bool rebuild_reference_shape = reference_shape_element != fe;
    mfem::Vector shape(shape_.scalar_interior_dofs), physical(dimension);
    mfem::DenseMatrix derivative(shape_.scalar_interior_dofs, dimension);
    for (int q = 0; q < shape_.volume_points; ++q)
    {
      const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      if (rebuild_reference_shape)
      {
        fe->CalcShape(point, shape);
        for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
          reference_volume_shape[
              static_cast<std::size_t>(q) * shape_.scalar_interior_dofs +
              scalar] = shape[scalar];
      }
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
            reference_volume_shape[
                static_cast<std::size_t>(q) * shape_.scalar_interior_dofs +
                scalar];
        for (int direction = 0; direction < dimension; ++direction)
          host_volume_derivative_[
              (quadrature * shape_.scalar_interior_dofs + scalar) * dimension +
              direction] = derivative(scalar, direction);
      }
    }
    reference_shape_element = fe;

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
        double exact[conserved_components]{}, perturbed[conserved_components]{};
        if (manufactured_initial_state)
        {
          mfem::Vector face_physical(dimension);
          ft->Face->Transform(face_nodes.IntPoint(scalar), face_physical);
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
      if (manufactured_initial_state)
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
SeedCandidateLimiterDiagnostics() noexcept
{
  const std::size_t elements = static_cast<std::size_t>(shape_.elements);
  const std::size_t value_count = 2 * elements;
  const bool local_valid = initialized_ && positivity_limiter_enabled_ &&
      elements > 0 && committed_limiter_diagnostics_ >= 0 &&
      committed_limiter_diagnostics_ < 2 &&
      candidate_limiter_diagnostics_ >= 0 &&
      candidate_limiter_diagnostics_ < 2 &&
      committed_limiter_diagnostics_ != candidate_limiter_diagnostics_ &&
      view_.positivity_theta_rho[committed_limiter_diagnostics_] != nullptr &&
      view_.positivity_theta_p[committed_limiter_diagnostics_] ==
          view_.positivity_theta_rho[committed_limiter_diagnostics_] +
              shape_.elements &&
      view_.positivity_theta_rho[candidate_limiter_diagnostics_] != nullptr &&
      view_.positivity_theta_p[candidate_limiter_diagnostics_] ==
          view_.positivity_theta_rho[candidate_limiter_diagnostics_] +
              shape_.elements;
  if (!CollectiveContinue(local_valid)) return false;
  const bool copied = TransportPolicy::DeviceToDevice(
      view_.positivity_theta_rho[candidate_limiter_diagnostics_],
      view_.positivity_theta_rho[committed_limiter_diagnostics_],
      value_count * sizeof(double), CudaTransferCategory::attempt_history,
      CudaTransferSite::attempt_seed_interior);
  return CollectiveContinue(copied);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
ExportFields(const CudaInteriorState& source,
             const generic::HdgEulerHostInteriorFields destination,
             const bool prepared_diagnostics) noexcept
{
  const std::size_t scalar_count =
      static_cast<std::size_t>(interior_space_->GetVSize() /
                               shape_.interior_components);
  const bool sigma_requested = destination.sigma_coefficients != nullptr;
  const int output_components = sigma_requested
      ? igr_published_components : conserved_components;
  const std::size_t coefficient_count = scalar_count *
      static_cast<std::size_t>(output_components);
  const bool any_limiter_diagnostics =
      destination.zhang_shu_theta_rho != nullptr ||
      destination.zhang_shu_theta_p != nullptr ||
      destination.zhang_shu_element_count != 0;
  const bool limiter_diagnostics_requested =
      destination.zhang_shu_theta_rho != nullptr &&
      destination.zhang_shu_theta_p != nullptr &&
      destination.zhang_shu_element_count ==
          static_cast<std::size_t>(shape_.elements);
  const int diagnostic_slot = prepared_diagnostics
      ? candidate_limiter_diagnostics_
      : committed_limiter_diagnostics_;
  const bool limiter_diagnostics_ready =
      !any_limiter_diagnostics ||
      (limiter_diagnostics_requested && positivity_limiter_enabled_ &&
       diagnostic_slot >= 0 && diagnostic_slot < 2 &&
       view_.positivity_theta_rho[diagnostic_slot] != nullptr &&
       view_.positivity_theta_p[diagnostic_slot] ==
           view_.positivity_theta_rho[diagnostic_slot] + shape_.elements &&
       host_export_positivity_.size() ==
           2 * destination.zhang_shu_element_count);
  bool local_valid = initialized_ && borrowed_host_inputs_ &&
      source.coefficients != nullptr && source.count ==
          static_cast<std::size_t>(shape_.interior_count) &&
      destination.scalar_count == scalar_count &&
      (destination.conservative_coefficients != nullptr || scalar_count == 0) &&
      (!sigma_requested || shape_.igr_active) &&
      limiter_diagnostics_ready;
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
      (!limiter_diagnostics_requested ||
       TransportPolicy::DeviceToHost(
           host_export_positivity_.data(),
           view_.positivity_theta_rho[diagnostic_slot],
           2 * destination.zhang_shu_element_count * sizeof(double),
           CudaTransferCategory::generic_host_bridge,
           CudaTransferSite::generic_host_export)) &&
      TransportPolicy::Synchronize(CudaTransferSite::generic_host_export);
  if (!CollectiveContinue(copied)) { return false; }
  bool limiter_values_valid = true;
  if (limiter_diagnostics_requested)
  {
    const std::size_t elements = destination.zhang_shu_element_count;
    for (std::size_t element = 0;
         element < elements && limiter_values_valid; ++element)
    {
      const double theta_rho = host_export_positivity_[element];
      const double theta_p = host_export_positivity_[elements + element];
      limiter_values_valid = std::isfinite(theta_rho) &&
          theta_rho >= 0.0 && theta_rho <= 1.0 &&
          std::isfinite(theta_p) && theta_p >= 0.0 && theta_p <= 1.0;
    }
  }
  if (!CollectiveContinue(limiter_values_valid)) { return false; }
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
  if (limiter_diagnostics_requested)
  {
    const std::size_t elements = destination.zhang_shu_element_count;
    for (std::size_t element = 0; element < elements; ++element)
    {
      const double theta_rho = host_export_positivity_[element];
      const double theta_p = host_export_positivity_[elements + element];
      destination.zhang_shu_theta_rho[element] = theta_rho;
      destination.zhang_shu_theta_p[element] = theta_p;
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
    TraceCsrGraphPlan graph = BuildTraceCsrGraphPlan({
        shape_.elements, shape_.trace_dofs, shape_.local_trace_count,
        signed_element_trace_.data(),
        static_cast<std::size_t>(shape_.elements) * shape_.trace_dofs});
    if (!graph.valid ||
        graph.offsets.size() !=
            static_cast<std::size_t>(shape_.local_trace_count) + 1 ||
        graph.columns.empty())
      return false;

    // Wrap the topology without ownership, let MFEM allocate the value array,
    // then deep-copy the complete CSR object into MFEM-managed storage. Raw
    // new[] graph arrays cannot be transferred into a DEVICE_UMPIRE build:
    // its MemoryManager must remain the allocation and teardown authority.
    mfem::SparseMatrix planned_graph(
        graph.offsets.data(), graph.columns.data(), nullptr,
        shape_.local_trace_count, shape_.local_trace_count,
        false, false, graph.legacy_columns_sorted);
    double* values = planned_graph.HostWriteData();
    for (int row = 0; row < shape_.local_trace_count; ++row)
      for (int entry = graph.offsets[static_cast<std::size_t>(row)];
           entry < graph.offsets[static_cast<std::size_t>(row) + 1]; ++entry)
        values[entry] =
            graph.columns[static_cast<std::size_t>(entry)] == row
            ? 64.0 : 1.0e-3;
    local_trace_matrix_.emplace(planned_graph, true);
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
  if (partitioned_backward_euler_)
    view_.partitioned_base_interior = take_d(
        generic ? generic_arena_layout_.partitioned_base_interior : 0);
  if (rodas3_)
  {
    for (double*& stage : view_.rodas_stage_interior)
      stage = take_d(max_i);
    view_.rodas_combination_interior = take_d(max_i);
    // The initial scalar bootstrap completes before any RODAS stage exists.
    // Reuse its principal-combination vector for the frozen U4 snapshot so
    // making the private t=0 IGR state coherent adds no persistent arena.
    if (shape_.igr_active)
      view_.partitioned_base_interior =
          view_.rodas_combination_interior;
    view_.rodas_residual_f = take_d(max_i);
    for (double*& stage : view_.rodas_stage_true_trace)
      stage = take_d(max_tt);
    if (ranks_ == 2)
      for (double*& stage : view_.rodas_stage_local_trace)
        stage = take_d(max_lt);
    view_.rodas_combination_true_trace = take_d(max_tt);
    view_.rodas_residual_g = take_d(max_er);
    view_.rodas_operator_b = take_d(max_b);
    view_.rodas_operator_c = take_d(max_b);
  }
  view_.output_interior = take_d(max_i);
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
  if (shape_.local_solve != CudaLocalSolveKind::current)
  {
    view_.local_solve_workspace = take_d(generic_arena_layout_.local_solve_workspace);
    view_.local_solve_work = take_d(generic_arena_layout_.local_solve_work);
    if (shape_.local_solve_backend == CudaLocalSolveBackend::cublas_batched)
      view_.local_solve_vendor_workspace = take_d(
          generic_arena_layout_.local_solve_vendor_workspace);
  }
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
  if (positivity_limiter_enabled_)
    view_.limiter_vertex_shape = take_d(
        generic_arena_layout_.limiter_vertex_shape);
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
  if (positivity_limiter_enabled_)
  {
    for (int slot = 0; slot < 2; ++slot)
    {
      view_.positivity_theta_rho[slot] =
          take_d(generic_arena_layout_.elements);
      view_.positivity_theta_p[slot] =
          take_d(generic_arena_layout_.elements);
    }
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

  if (CudaTracePreconditionerIsNativeElementSchwarz(
          trace_policy_descriptor_.trace_preconditioner))
  {
    // Rank-one has no remote matrix/vector traffic. Repartition those already
    // persistent slices as the Schwarz principal matrix, LU, maps, pivots,
    // and two element-vector work arrays without growing either arena.
    if (ranks_ != 1 || shape_.matrix_send_count != 0 ||
        shape_.matrix_receive_count != 0 || shape_.vector_send_count != 0 ||
        shape_.vector_receive_count != 0)
      return false;
    view_.element_schwarz_patch_lu = view_.remote_matrix_send;
    view_.element_schwarz_patch_matrix = view_.remote_matrix_receive;
    view_.element_schwarz_rhs = view_.remote_vector_send;
    view_.element_schwarz_solution = view_.remote_vector_receive;
    view_.element_schwarz_first_source = view_.matrix_send_source;
    view_.element_schwarz_second_source = view_.matrix_receive_target;
    view_.element_schwarz_true_incidence_slots =
        view_.matrix_receive_section;
    view_.element_schwarz_signed_true_dof = view_.vector_send_source;
    view_.element_schwarz_pivots = view_.vector_receive_target;
  }

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
  if (positivity_limiter_enabled_ &&
      !h2d(view_.limiter_vertex_shape,
           host_limiter_vertex_shape_.data(),
           static_cast<std::size_t>(4 * shape_.scalar_interior_dofs) *
               sizeof(double), CudaTransferSite::quadrature_upload))
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
  ::igrlab::NVTXRange setup_profile_region("HDG persistent true-system setup");
  const CudaTracePolicyDescriptor& trace_policy = trace_policy_descriptor_;
  if (!CudaTracePolicyDescriptorIsExact(trace_policy) ||
      trace_policy.igr_active != shape_.igr_active)
    return false;
  const CudaTracePreconditionerKind selected_preconditioner =
      trace_policy.trace_preconditioner;
  const bool use_ilu =
      CudaTracePreconditionerIsIlu(selected_preconditioner);
  const bool use_element_schwarz =
      CudaTracePreconditionerIsNativeElementSchwarz(
          selected_preconditioner);
  const bool use_matrix_free =
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          trace_policy.resolved_selection);
  const bool use_native = trace_solver_kind_ ==
      CudaTraceSolverKind::native_rank1_fp64;
  const bool use_mgr = trace_solver_kind_ ==
      CudaTraceSolverKind::hypre_fgmres_mgr_rank1;
  const int expected_element_schwarz_rows =
      NativeElementSchwarzPatchRows(shape_.order, shape_.igr_active);
  const int expected_trace_components = shape_.igr_active
      ? igr_trace_components : conserved_components;
  if (!IsKnownCudaTraceSolverKind(trace_solver_kind_) ||
      (use_native && (ranks_ != 1 || !borrowed_topology_ ||
                      !shape_.borrowed_host_inputs)) ||
      (use_mgr &&
       (ranks_ != 1 || !borrowed_topology_ ||
        !shape_.borrowed_host_inputs || !shape_.igr_active ||
        selected_preconditioner !=
            CudaTracePreconditionerKind::hypre_mgr_field_split)) ||
      (!use_mgr && selected_preconditioner ==
          CudaTracePreconditionerKind::hypre_mgr_field_split) ||
      (use_element_schwarz &&
       (!use_native || ranks_ != 1 ||
        shape_.trace_components != expected_trace_components ||
        shape_.trace_dofs != expected_element_schwarz_rows)) ||
      (use_matrix_free && !use_element_schwarz))
    return false;
  initialization_tensor_core_evidence_ = {};
  initialization_tensor_core_evidence_.selected_engine = cuda_engine_;
  initialization_tensor_core_evidence_.requested_trace_policy =
      trace_policy.requested_selection;
  initialization_tensor_core_evidence_.resolved_trace_policy =
      trace_policy.resolved_selection;
  initialization_tensor_core_evidence_.trace_policy_rank_count =
      trace_policy.rank_count;
  initialization_tensor_core_evidence_.trace_policy_order =
      trace_policy.polynomial_order;
  initialization_tensor_core_evidence_.trace_preconditioner =
      selected_preconditioner;
  initialization_tensor_core_evidence_.native_rank1_trace_solver_configured =
      use_native;
  initialization_tensor_core_evidence_.
      hypre_mgr_rank1_trace_solver_configured = use_mgr;
  initialization_tensor_core_evidence_.
      hypre_host_staged_rank2_trace_solver_configured =
          trace_solver_kind_ ==
              CudaTraceSolverKind::hypre_host_staged_rank2;
  initialization_tensor_core_evidence_.trace_solver_accounting_complete =
      true;
  const bool robust =
      CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
          trace_policy.resolved_selection) &&
      CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
          selected_preconditioner);
  initialization_tensor_core_evidence_.robust_lower_specialization_supported =
      robust && native_trace_support_.supported &&
      CudaNativeTraceCooperativeSupportMatchesPreconditioner(
          native_trace_support_, CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64);
  initialization_tensor_core_evidence_.robust_coupled_specialization_supported =
      robust && native_trace_robust_coupled_support_.supported &&
      CudaNativeTraceCooperativeSupportMatchesPreconditioner(
          native_trace_robust_coupled_support_,
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_coupled_sigma_jacobi10_fp64);
  initialization_tensor_core_evidence_.robust_branch_accounting_complete =
      !robust ||
      (initialization_tensor_core_evidence_.
           robust_lower_specialization_supported &&
       initialization_tensor_core_evidence_.
           robust_coupled_specialization_supported);
  initialization_tensor_core_evidence_.native_element_schwarz_configured =
      false;
  initialization_tensor_core_evidence_.
      native_matrix_free_trace_operator_configured = false;
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
    if (local_objects_ready && use_mgr)
    {
      const int scalar_trace_dofs =
          shape_.true_trace_count / igr_trace_components;
      const int scalar_local_trace_dofs =
          shape_.local_trace_count / igr_trace_components;
      std::vector<unsigned char> true_marker_seen(
          static_cast<std::size_t>(shape_.true_trace_count), 0);
      bool component_ordering_exact = scalar_trace_dofs > 0 &&
          scalar_local_trace_dofs > 0 && trace_space_.has_value() &&
          trace_space_->GetOrdering() == mfem::Ordering::byNODES &&
          trace_space_->GetNDofs() == scalar_local_trace_dofs &&
          scalar_trace_dofs * igr_trace_components ==
              shape_.true_trace_count &&
          scalar_local_trace_dofs * igr_trace_components ==
              shape_.local_trace_count;
      for (const dg::hdg::trace::TraceMapEntry& entry :
           exchange_plan_.map.dofs.entries)
      {
        if (!component_ordering_exact) break;
        const bool entry_exact = entry.local_dof >= 0 &&
            entry.local_dof < shape_.local_trace_count &&
            entry.local_true_dof >= 0 &&
            entry.local_true_dof < shape_.true_trace_count &&
            entry.ownership == dg::hdg::Ownership::owned &&
            entry.local_dof / scalar_local_trace_dofs ==
                entry.local_true_dof / scalar_trace_dofs;
        if (!entry_exact)
        {
          component_ordering_exact = false;
          break;
        }
        true_marker_seen[static_cast<std::size_t>(
            entry.local_true_dof)] = 1;
      }
      component_ordering_exact = component_ordering_exact &&
          std::find(true_marker_seen.begin(), true_marker_seen.end(), 0) ==
              true_marker_seen.end();
      equilibrated_true_matrix_ =
          std::make_unique<mfem::HypreParMatrix>(*true_matrix_);
      local_objects_ready =
          shape_.trace_components == igr_trace_components &&
          component_ordering_exact &&
          scalar_trace_dofs > 0 &&
          scalar_trace_dofs * igr_trace_components ==
              shape_.true_trace_count &&
          true_device_.rows == shape_.true_trace_count &&
          true_device_.diag_columns_count == shape_.true_trace_count &&
          true_device_.offd_nonzeros == 0 &&
          equilibrated_true_matrix_ &&
          CudaGetTrueMatrixDeviceView(
              *equilibrated_true_matrix_, equilibrated_true_device_) &&
          equilibrated_true_device_.rows == true_device_.rows &&
          equilibrated_true_device_.diag_columns_count ==
              true_device_.diag_columns_count &&
          equilibrated_true_device_.diag_nonzeros ==
              true_device_.diag_nonzeros &&
          equilibrated_true_device_.offd_nonzeros == 0 &&
          equilibrated_true_device_.diag_values != true_device_.diag_values;
      if (local_objects_ready)
      {
        equilibration_left_.SetSize(shape_.true_trace_count);
        equilibration_right_.SetSize(shape_.true_trace_count);
        equilibration_row_norm_.SetSize(shape_.true_trace_count);
        equilibration_column_norm_.SetSize(shape_.true_trace_count);
        equilibration_control_.SetSize(
            static_cast<int>(cuda_rank1_equilibration_control_doubles));
        equilibration_left_.UseDevice(true);
        equilibration_right_.UseDevice(true);
        equilibration_row_norm_.UseDevice(true);
        equilibration_column_norm_.UseDevice(true);
        equilibration_control_.UseDevice(true);
        equilibration_left_device_ = equilibration_left_.Write();
        equilibration_right_device_ = equilibration_right_.Write();
        equilibration_row_norm_device_ = equilibration_row_norm_.Write();
        equilibration_column_norm_device_ =
            equilibration_column_norm_.Write();
        equilibration_control_device_ = equilibration_control_.Write();
        local_objects_ready = equilibration_left_device_ != nullptr &&
            equilibration_right_device_ != nullptr &&
            equilibration_row_norm_device_ != nullptr &&
            equilibration_column_norm_device_ != nullptr &&
            equilibration_control_device_ != nullptr &&
            CudaConfigureRankOneMgrState(
                rank_one_mgr_state_, communicator_,
                shape_.true_trace_count, scalar_trace_dofs,
                final_coherent_linear_relative_tolerance_,
                maximum_linear_iterations_,
                krylov_dimension_);
        for (int row = 0; local_objects_ready &&
             row < shape_.true_trace_count; ++row)
          local_objects_ready =
              rank_one_mgr_state_.point_markers[
                  static_cast<std::size_t>(row)] ==
              static_cast<HYPRE_Int>(row / scalar_trace_dofs);
      }
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
      if (use_mgr)
        hypre_equilibrated_rhs_.emplace(*equilibrated_true_matrix_);
      if (use_ilu)
      {
        if (use_native)
          native_allocator_prime_ilu.emplace();
        else
          ilu_.emplace();
      }
      else if (!use_native && !use_mgr)
        diagonal_.emplace();
      if (!use_native && !use_mgr)
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
      else if (!use_native && !use_mgr)
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
      if (use_mgr) hypre_equilibrated_rhs_->UseDevice(true);
      hypre_rhs_->HypreReadWrite();
      hypre_solution_->HypreReadWrite();
      hypre_action_->HypreReadWrite();
      if (use_mgr) hypre_equilibrated_rhs_->HypreReadWrite();
      view_.true_rhs_scratch =
          static_cast<mfem::Vector&>(*hypre_rhs_).ReadWrite();
      view_.true_solution =
          static_cast<mfem::Vector&>(*hypre_solution_).ReadWrite();
      if (use_mgr)
        equilibrated_rhs_device_ =
            static_cast<mfem::Vector&>(
                *hypre_equilibrated_rhs_).ReadWrite();
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
  if (borrowed_topology_ && shape_.borrowed_host_inputs)
  {
    // Uploads are complete. Runtime assembly reads device maps; Schwarz
    // retains only its scalar shape/multiplicity metadata on the host.
    std::vector<int>{}.swap(host_matrix_owned_source_);
    std::vector<int>{}.swap(host_matrix_owned_section_);
    std::vector<int>{}.swap(host_matrix_owned_target_);
    std::vector<int>{}.swap(element_schwarz_plan_.signed_element_true_dofs);
    std::vector<int>{}.swap(element_schwarz_plan_.first_element_sources);
    std::vector<int>{}.swap(element_schwarz_plan_.second_element_sources);
    std::vector<int>{}.swap(element_schwarz_plan_.true_multiplicity);
    std::vector<int>{}.swap(element_schwarz_plan_.true_incidence_slots);
  }
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
        use_element_schwarz ? true : warm_factor_ready &&
            LaunchAndObserveNativeRankOneTraceSolve(
                warm_factor, {}, 0,
                CudaNativeTraceActiveSubspace::full,
                final_coherent_linear_relative_tolerance_,
                warm);
  }
  else if (use_mgr)
  {
    CudaEquilibrationObservation equilibration{};
    warm_solved = BuildAndObserveRankOneEquilibratedSystem(
                      equilibration) &&
        CudaSolveRankOneMgr(
            rank_one_mgr_state_, *equilibrated_true_matrix_,
            *hypre_equilibrated_rhs_, *hypre_solution_, trace_policy,
            final_coherent_linear_relative_tolerance_,
            maximum_linear_iterations_,
            krylov_dimension_, 0, warm) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
        CudaLaunchUnscaleRankOneSolution(
            view_.true_solution, equilibration_right_device_,
            shape_.true_trace_count, nullptr) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (warm_solved)
    {
      initialization_tensor_core_evidence_.equilibration_builds = 1;
      initialization_tensor_core_evidence_.equilibration_passes =
          static_cast<std::uint64_t>(equilibration.passes);
      initialization_tensor_core_evidence_.minimum_left_scale =
          equilibration.minimum_left_scale;
      initialization_tensor_core_evidence_.maximum_left_scale =
          equilibration.maximum_left_scale;
      initialization_tensor_core_evidence_.minimum_right_scale =
          equilibration.minimum_right_scale;
      initialization_tensor_core_evidence_.maximum_right_scale =
          equilibration.maximum_right_scale;
      initialization_tensor_core_evidence_.mgr_setup_calls =
          rank_one_mgr_state_.setup_calls;
      initialization_tensor_core_evidence_.
          mgr_preconditioner_applications =
              rank_one_mgr_state_.preconditioner_applications;
      initialization_tensor_core_evidence_.mgr_f_ilu_setups =
          initialization_tensor_core_evidence_.mgr_setup_calls;
      initialization_tensor_core_evidence_.mgr_f_ilu_applications =
          initialization_tensor_core_evidence_.
              mgr_preconditioner_applications;
    }
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
  if (warm_solved && !(use_native && (use_ilu || use_element_schwarz)))
  {
    if (use_native)
      initialization_tensor_core_evidence_.
          native_rank1_trace_solve_calls = 1;
    else
      initialization_tensor_core_evidence_.hypre_trace_solve_calls = 1;
  }
  const bool warmed = use_native && (use_ilu || use_element_schwarz)
      ? (use_ilu ? native_allocator_prime_ready :
          element_schwarz_plan_.valid)
      : warm_solved && warm.device_policy &&
          warm.configured_preconditioner && warm.finite &&
          warm.matrix_generation == 0 &&
          warm.trace_solver == trace_solver_kind_ &&
          warm.configured_native_solver == use_native &&
          warm.configured_mgr == use_mgr;
  if (!CollectiveContinue(warmed)) { return false; }

  bool warm_state_zeroed = false;
  try
  {
    *hypre_rhs_ = 0.0;
    *hypre_solution_ = 0.0;
    *hypre_action_ = 0.0;
    if (use_mgr) *hypre_equilibrated_rhs_ = 0.0;
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
      const bool native_specific_ready = use_ilu
          ? native_ilu_inverse_diagonal_ != nullptr &&
              generic_arena_layout_.native_ilu_inverse_diagonal ==
                  static_cast<std::size_t>(shape_.true_trace_count) &&
              native_trace_workspace_.permutation_marks != nullptr
          : use_element_schwarz
              ? native_ilu_inverse_diagonal_ == nullptr &&
                  (native_trace_requirements_.permutation_integers == 0
                       ? native_trace_workspace_.permutation_marks == nullptr
                       : native_trace_workspace_.permutation_marks !=
                             nullptr) &&
                  element_schwarz_plan_.valid &&
                  view_.element_schwarz_patch_matrix != nullptr &&
                  view_.element_schwarz_patch_lu != nullptr &&
                  view_.element_schwarz_rhs != nullptr &&
                  view_.element_schwarz_solution != nullptr &&
                  view_.element_schwarz_first_source != nullptr &&
                  view_.element_schwarz_second_source != nullptr &&
                  view_.element_schwarz_signed_true_dof != nullptr &&
                  view_.element_schwarz_true_incidence_slots != nullptr &&
                  view_.element_schwarz_pivots != nullptr
              : native_ilu_inverse_diagonal_ == nullptr &&
                  native_trace_workspace_.permutation_marks == nullptr;
      persistent_configured = !gmres_ && !diagonal_ &&
          (use_ilu == static_cast<bool>(ilu_)) &&
          native_trace_requirements_.valid &&
          native_trace_support_.supported &&
          (!robust ||
           (native_trace_robust_lower_requirements_.valid &&
            native_trace_robust_coupled_requirements_.valid &&
            native_trace_robust_coupled_support_.supported &&
            CudaNativeTraceCooperativeSupportMatchesPreconditioner(
                native_trace_support_, CudaTracePreconditionerKind::
                    native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64) &&
            CudaNativeTraceCooperativeSupportMatchesPreconditioner(
                native_trace_robust_coupled_support_,
                CudaTracePreconditionerKind::
                    native_element_schwarz_warp_coupled_sigma_jacobi10_fp64))) &&
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
          native_specific_ready;
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
                persistent_warm_factor, {}, 0,
                CudaNativeTraceActiveSubspace::full,
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
    else if (use_mgr)
    {
      persistent_configured = !gmres_ && !ilu_ && !diagonal_ &&
          equilibrated_true_matrix_ && hypre_equilibrated_rhs_ &&
          rank_one_mgr_state_.configured &&
          rank_one_mgr_state_.fgmres != nullptr &&
          rank_one_mgr_state_.mgr != nullptr &&
          rank_one_mgr_state_.f_ilu != nullptr &&
          rank_one_mgr_state_.coarse_amg != nullptr &&
          rank_one_mgr_state_.point_markers.size() ==
              static_cast<std::size_t>(shape_.true_trace_count) &&
          equilibration_left_device_ != nullptr &&
          equilibration_right_device_ != nullptr &&
          equilibration_row_norm_device_ != nullptr &&
          equilibration_column_norm_device_ != nullptr &&
          equilibration_control_device_ != nullptr &&
          equilibrated_rhs_device_ != nullptr &&
          equilibrated_true_device_.rows == shape_.true_trace_count &&
          equilibrated_true_device_.offd_nonzeros == 0;
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
    initialization_tensor_core_evidence_.native_element_schwarz_configured =
        use_element_schwarz;
    initialization_tensor_core_evidence_.
        native_matrix_free_trace_operator_configured = use_matrix_free;
    initialization_tensor_core_evidence_.element_schwarz_patch_rows =
        use_element_schwarz ? element_schwarz_plan_.patch_rows : 0;
    initialization_tensor_core_evidence_.
        element_schwarz_minimum_multiplicity = use_element_schwarz
            ? element_schwarz_plan_.minimum_multiplicity : 0;
    initialization_tensor_core_evidence_.
        element_schwarz_maximum_multiplicity = use_element_schwarz
            ? element_schwarz_plan_.maximum_multiplicity : 0;
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
  ::igrlab::NVTXRange staging_profile_region("HDG sparse staging plan");
  // The generic true graph is persistent, so resolve its topology once.  The
  // old staging builder linearly scanned every owned true row for every dense
  // element-Schur entry.  That made setup O(E * n_t^2 * N_true), even for the
  // native rank-one matrix-free operator.  Keep the exact CSR staging needed
  // by the independent residual and robust native checks, but resolve local
  // rows directly. A bounded temporary row lookup avoids a CSR search for
  // every local element-Schur column; the original searches remain available
  // if the scratch allocation is unavailable or a column is outside it.
  std::vector<std::pair<std::int64_t, int>> owned_row_lookup;
  std::vector<unsigned char> diag_tail_sorted;
  std::vector<unsigned char> offd_sorted;
  bool lookup_ready = shape_.true_trace_count > 0 &&
      owned_global_true_.size() >=
          static_cast<std::size_t>(shape_.true_trace_count);
  try
  {
    if (lookup_ready)
    {
      owned_row_lookup.reserve(
          static_cast<std::size_t>(shape_.true_trace_count));
      for (int row = 0; row < shape_.true_trace_count; ++row)
        owned_row_lookup.emplace_back(
            owned_global_true_[static_cast<std::size_t>(row)], row);
      std::sort(owned_row_lookup.begin(), owned_row_lookup.end());
      diag_tail_sorted.assign(
          static_cast<std::size_t>(shape_.true_trace_count), 0);
      offd_sorted.assign(
          static_cast<std::size_t>(shape_.true_trace_count), 0);
    }
  }
  catch (...)
  {
    lookup_ready = false;
  }
  for (std::size_t index = 1;
       lookup_ready && index < owned_row_lookup.size(); ++index)
    if (owned_row_lookup[index - 1].first == owned_row_lookup[index].first)
      lookup_ready = false;
  if (!CollectiveContinue(lookup_ready)) return false;

  bool structure_searchable =
      true_structure_.diag_nonzeros >= 0 &&
      true_structure_.offd_nonzeros >= 0 &&
      true_structure_.diag_offsets.size() ==
          static_cast<std::size_t>(shape_.true_trace_count) + 1 &&
      true_structure_.offd_offsets.size() ==
          static_cast<std::size_t>(shape_.true_trace_count) + 1 &&
      true_structure_.diag_columns.size() ==
          static_cast<std::size_t>(true_structure_.diag_nonzeros) &&
      true_structure_.offd_columns.size() ==
          static_cast<std::size_t>(true_structure_.offd_nonzeros);
  for (int row = 0; structure_searchable &&
       row < shape_.true_trace_count; ++row)
  {
    const int diag_begin = true_structure_.diag_offsets[
        static_cast<std::size_t>(row)];
    const int diag_end = true_structure_.diag_offsets[
        static_cast<std::size_t>(row + 1)];
    const int offd_begin = true_structure_.offd_offsets[
        static_cast<std::size_t>(row)];
    const int offd_end = true_structure_.offd_offsets[
        static_cast<std::size_t>(row + 1)];
    structure_searchable = diag_begin >= 0 && diag_begin <= diag_end &&
        diag_end <= true_structure_.diag_nonzeros && offd_begin >= 0 &&
        offd_begin <= offd_end && offd_end <= true_structure_.offd_nonzeros;
    if (!structure_searchable) break;
    const int diag_tail_begin = std::min(diag_begin + 1, diag_end);
    diag_tail_sorted[static_cast<std::size_t>(row)] =
        static_cast<unsigned char>(std::is_sorted(
            true_structure_.diag_columns.begin() + diag_tail_begin,
            true_structure_.diag_columns.begin() + diag_end));
    offd_sorted[static_cast<std::size_t>(row)] =
        static_cast<unsigned char>(std::is_sorted(
            true_structure_.offd_columns.begin() + offd_begin,
            true_structure_.offd_columns.begin() + offd_end));
  }
  if (!CollectiveContinue(structure_searchable)) return false;

  CudaSparseOwnedRowLookup owned_row_targets;
  (void)owned_row_targets.Initialize(shape_.true_trace_count);

  const auto find_row = [&](const std::int64_t global)
  {
    const auto found = std::lower_bound(
        owned_row_lookup.begin(), owned_row_lookup.end(), global,
        [](const std::pair<std::int64_t, int>& entry,
           const std::int64_t value) { return entry.first < value; });
    return found != owned_row_lookup.end() && found->first == global
        ? found->second : -1;
  };
  const auto find_target = [&](const int row,
                               const int owned_column,
                               const std::int64_t column_global,
                               int& section, int& target)
  {
    if (row < 0 || row >= shape_.true_trace_count) return false;
    if (owned_column >= 0)
    {
      const int begin = true_structure_.diag_offsets[
          static_cast<std::size_t>(row)];
      const int end = true_structure_.diag_offsets[
          static_cast<std::size_t>(row + 1)];
      if (owned_row_targets.Admits(owned_column))
      {
        const int found = owned_row_targets.Find(
            true_structure_.diag_columns, begin, end, owned_column);
        if (found < 0) return false;
        section = 0;
        target = found;
        return true;
      }
      if (begin < end && true_structure_.diag_columns[
              static_cast<std::size_t>(begin)] == owned_column)
      {
        section = 0;
        target = begin;
        return true;
      }
      const int tail_begin = std::min(begin + 1, end);
      if (diag_tail_sorted[static_cast<std::size_t>(row)] != 0)
      {
        const auto first = true_structure_.diag_columns.begin() + tail_begin;
        const auto last = true_structure_.diag_columns.begin() + end;
        const auto found = std::lower_bound(first, last, owned_column);
        if (found != last && *found == owned_column)
        {
          section = 0;
          target = static_cast<int>(
              found - true_structure_.diag_columns.begin());
          return true;
        }
      }
      else
        for (int entry = tail_begin; entry < end; ++entry)
          if (true_structure_.diag_columns[
                  static_cast<std::size_t>(entry)] == owned_column)
          {
            section = 0;
            target = entry;
            return true;
          }
      return false;
    }

    const int begin = true_structure_.offd_offsets[
        static_cast<std::size_t>(row)];
    const int end = true_structure_.offd_offsets[
        static_cast<std::size_t>(row + 1)];
    if (offd_sorted[static_cast<std::size_t>(row)] != 0)
    {
      const auto first = true_structure_.offd_columns.begin() + begin;
      const auto last = true_structure_.offd_columns.begin() + end;
      const auto found = std::lower_bound(first, last, column_global);
      if (found != last && *found == column_global)
      {
        section = 1;
        target = static_cast<int>(
            found - true_structure_.offd_columns.begin());
        return true;
      }
    }
    else
      for (int entry = begin; entry < end; ++entry)
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
  {
    ::igrlab::NVTXRange mapping_profile_region("HDG sparse staging local maps");
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
        const int true_row = row_owner == rank_
            ? local_true_index_[static_cast<std::size_t>(row_local)] : -1;
        const int vector_source = element * shape_.trace_dofs + row;
        if (row_owner == rank_)
        {
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
          const int column_owner =
              local_owner_[static_cast<std::size_t>(column_local)];
          const int owned_column = column_owner == rank_
              ? local_true_index_[static_cast<std::size_t>(column_local)] : -1;
          const int source = (element * shape_.trace_dofs + row) *
                                 shape_.trace_dofs + column;
          if (row_owner == rank_)
          {
            int section = -1, target = -1;
            if (!find_target(
                    true_row, owned_column, column_global, section, target))
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
    {
      const std::int64_t row_global =
          matrix_receive_keys_[2 * static_cast<std::size_t>(entry)];
      const std::int64_t column_global =
          matrix_receive_keys_[2 * static_cast<std::size_t>(entry) + 1];
      local_receive_valid = find_target(
          find_row(row_global), find_row(column_global), column_global,
          host_matrix_receive_section_[static_cast<std::size_t>(entry)],
          host_matrix_receive_target_[static_cast<std::size_t>(entry)]) &&
          local_receive_valid;
    }
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

  const bool use_element_schwarz =
      CudaTracePreconditionerIsNativeElementSchwarz(
          trace_policy_descriptor_.trace_preconditioner);
  if (use_element_schwarz)
  {
    ::igrlab::NVTXRange schwarz_profile_region("HDG sparse staging Schwarz plan");
    const int expected_patch_rows =
        NativeElementSchwarzPatchRows(shape_.order, shape_.igr_active);
    const bool direct_rank_one = ranks_ == 1 &&
        CudaTracePolicyUsesNativeElementMatrixFreeOperator(
            trace_policy_descriptor_.resolved_selection);
    element_schwarz_plan_ = direct_rank_one
        ? BuildDirectNativeElementSchwarzPlan({
              shape_.elements, shape_.trace_dofs,
              shape_.local_trace_count, shape_.true_trace_count,
              signed_element_trace_.data(),
              static_cast<std::size_t>(shape_.elements) * shape_.trace_dofs,
              local_true_index_.data(), local_true_index_.size()})
        : BuildNativeElementSchwarzPlan({
              shape_.elements, shape_.trace_dofs,
              shape_.local_trace_count, shape_.true_trace_count,
              true_structure_.diag_nonzeros,
              signed_element_trace_.data(),
              static_cast<std::size_t>(shape_.elements) * shape_.trace_dofs,
              local_true_index_.data(), local_true_index_.size(),
              host_matrix_owned_source_.data(),
              host_matrix_owned_section_.data(),
              host_matrix_owned_target_.data(),
              static_cast<std::size_t>(shape_.matrix_owned_count)});
    const std::size_t patch_entries =
        static_cast<std::size_t>(shape_.elements) * shape_.trace_dofs *
        shape_.trace_dofs;
    const std::size_t element_rows =
        static_cast<std::size_t>(shape_.elements) * shape_.trace_dofs;
    const int expected_trace_components = shape_.igr_active
        ? igr_trace_components : conserved_components;
    const bool plan_exact = ranks_ == 1 &&
        shape_.trace_components == expected_trace_components &&
        shape_.trace_dofs == expected_patch_rows &&
        shape_.matrix_owned_count == static_cast<int>(patch_entries) &&
        shape_.matrix_send_count == 0 && shape_.matrix_receive_count == 0 &&
        shape_.vector_send_count == 0 && shape_.vector_receive_count == 0 &&
        element_schwarz_plan_.valid &&
        element_schwarz_plan_.signed_element_true_dofs.size() ==
            element_rows &&
        element_schwarz_plan_.first_element_sources.size() == patch_entries &&
        element_schwarz_plan_.second_element_sources.size() == patch_entries &&
        element_schwarz_plan_.true_incidence_slots.size() ==
            2 * static_cast<std::size_t>(shape_.true_trace_count);
    if (!CollectiveContinue(plan_exact)) return false;
  }
  else
    element_schwarz_plan_ = {};

  const auto upload = [](int* destination, const int* source, int count)
  {
    return count == 0 || TransportPolicy::HostToDevice(
        destination, source, static_cast<std::size_t>(count) * sizeof(int),
        CudaTransferCategory::topology, CudaTransferSite::map_upload);
  };
  ::igrlab::NVTXRange upload_profile_region("HDG sparse staging upload");
  bool uploaded = upload(
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
  if (uploaded && use_element_schwarz)
  {
    const int patch_entries = shape_.elements * shape_.trace_dofs *
        shape_.trace_dofs;
    const int element_rows = shape_.elements * shape_.trace_dofs;
    uploaded = upload(
                   view_.element_schwarz_first_source,
                   element_schwarz_plan_.first_element_sources.data(),
                   patch_entries) &&
        upload(view_.element_schwarz_second_source,
               element_schwarz_plan_.second_element_sources.data(),
               patch_entries) &&
        upload(view_.element_schwarz_signed_true_dof,
               element_schwarz_plan_.signed_element_true_dofs.data(),
               element_rows) &&
        upload(view_.element_schwarz_true_incidence_slots,
               element_schwarz_plan_.true_incidence_slots.data(),
               2 * shape_.true_trace_count);
  }
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
  if (!UpdateStateHashes(committed_slot_)) { return false; }
  // UploadTopologyAndState and BuildPersistentTrueSystem have completed and
  // synchronized successful uploads of the same host interior/trace bytes to
  // both slots. Only their storage identities differ. As with a checked state
  // copy, transfer the hash metadata; all later mutable-state audits remain.
  candidate_slot_.interior.local_hash = committed_slot_.interior.local_hash;
  candidate_slot_.interior.hash = committed_slot_.interior.hash;
  candidate_slot_.trace.local_hash = committed_slot_.trace.local_hash;
  candidate_slot_.trace.hash = committed_slot_.trace.hash;
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
VerifyResidencyAndTicket(CudaRunEvidence&) noexcept
{
  hypre_policy_valid_ = CudaHypreDevicePolicyValid(
      *true_matrix_, *hypre_rhs_, *hypre_solution_) &&
      (trace_solver_kind_ !=
               CudaTraceSolverKind::hypre_fgmres_mgr_rank1 ||
       (equilibrated_true_matrix_ && hypre_equilibrated_rhs_ &&
        CudaHypreDevicePolicyValid(
            *equilibrated_true_matrix_, *hypre_equilibrated_rhs_,
            *hypre_solution_)));
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
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_reduction);
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
ObserveLocalSolveWork() noexcept
{
  if (shape_.local_solve == CudaLocalSolveKind::current) return true;
  double counts[3]{};
  if (!TransportPolicy::DeviceToHost(counts, view_.scalar, sizeof(counts),
          CudaTransferCategory::scalar_diagnostic, CudaTransferSite::scalar_status) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status)) return false;
  for (double value : counts) if (!std::isfinite(value) || value < 0.0) return false;
  local_solve_elements_ += shape_.elements;
  local_solve_corrections_ += counts[0];
  local_solve_fallbacks_ += counts[1];
  local_solve_failures_ += shape_.elements - counts[2];
  return counts[2] == shape_.elements;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
OriginalCoupledResidualPassed() noexcept
{
  if (shape_.local_solve == CudaLocalSolveKind::current) return true;
  // Assemble original trace RHS and action separately. Subtracting the tiny
  // coherent correction from each large, cancelling face RHS first loses the
  // very digits this relative residual is intended to certify.
  if (!CudaLaunchRefinedOriginalCoupledSquaredNorms(view_, shape_)) return false;
  double norms[2]{};
  if (!TransportPolicy::DeviceToHost(norms, view_.scalar, sizeof(norms),
          CudaTransferCategory::scalar_diagnostic, CudaTransferSite::scalar_status) ||
      !TransportPolicy::Synchronize(CudaTransferSite::scalar_status) ||
      !std::isfinite(norms[0]) || !std::isfinite(norms[1]) ||
      norms[0] < 0.0 || norms[1] < 0.0) return false;
  const double relative = norms[1] > 0.0
      ? std::sqrt(norms[0]) / std::sqrt(norms[1])
      : norms[0] == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  local_solve_maximum_coupled_residual_ = std::max(
      local_solve_maximum_coupled_residual_, relative);
  return std::isfinite(relative) && relative <= true_residual_relative_tolerance_;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
DistributeCurrentTrace(const double* true_values) noexcept
{
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_packing);
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
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_packing);
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
DistributeRodasStageIncrement(const std::size_t stage) noexcept
{
  using Coefficients = ::dg::hdg::time::Rodas3Coefficients;
  if (stage >= Coefficients::stage_count ||
      view_.rodas_stage_true_trace[stage] == nullptr)
    return false;
  // Rank one has no ghosts and the conservation kernel deliberately keeps the
  // established owned=true-vector path without any added staging work.
  if (ranks_ == 1) return true;
  if (ranks_ != 2 || view_.rodas_stage_local_trace[stage] == nullptr)
    return false;

  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_packing);
  double* owned_host = owned_stage_.HostWrite();
  const std::size_t owned_bytes =
      static_cast<std::size_t>(shape_.true_trace_count) * sizeof(double);
  const std::size_t local_bytes =
      static_cast<std::size_t>(shape_.local_trace_count) * sizeof(double);
  const bool staged_out = TransportPolicy::DeviceToHost(
          owned_host, view_.rodas_stage_true_trace[stage], owned_bytes,
          CudaTransferCategory::host_staged_trace_exchange,
          CudaTransferSite::trace_solution_stage_out) &&
      TransportPolicy::Synchronize(
          CudaTransferSite::trace_solution_stage_out);
  if (!CollectiveContinue(staged_out)) return false;
  const bool distributed =
      exchange_plan_.Distribute(*trace_space_, owned_stage_, local_stage_);
  const bool staging_recorded = TransportPolicy::RecordMpiStaging(
      host_exchange_bytes_,
      CudaTransferCategory::host_staged_trace_exchange,
      CudaTransferSite::trace_ghost_stage_in, owned_host,
      local_stage_.HostRead(), owned_bytes, local_bytes, distributed);
  if (!CollectiveContinue(distributed && staging_recorded)) return false;
  const bool local_ready = TransportPolicy::HostToDevice(
          view_.rodas_stage_local_trace[stage], local_stage_.HostRead(),
          local_bytes, CudaTransferCategory::host_staged_trace_exchange,
          CudaTransferSite::trace_ghost_stage_in) &&
      TransportPolicy::Synchronize(CudaTransferSite::trace_ghost_stage_in);
  return CollectiveContinue(local_ready);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleTrueVector(const double* element_values, double* true_values,
                   const bool completion_required) noexcept
{
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_trace_assembly);
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
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_trace_assembly);
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
  const bool matrix_free_operator =
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          trace_policy_descriptor_.resolved_selection);
  if (shape_.local_solve != CudaLocalSolveKind::current)
  {
    // Local g-C*A^{-1}f can lose a near-converged RHS before opposing
    // element contributions are assembled. Preserve the original global
    // equation by assembling its two terms independently in the opt-in path.
    if (!CollectiveContinue(ranks_ == 1 &&
            CudaLaunchRefinedTrueRhs(view_, shape_)) ||
        !CollectiveContinue(TransportPolicy::Synchronize(
            CudaTransferSite::rhs_remote_scatter))) return false;
  }
  else if (matrix_free_operator)
  {
    const bool exact_rank_one_rhs = ranks_ == 1 &&
        element_schwarz_plan_.valid &&
        CudaLaunchAssembleNativeElementVector(
            view_, shape_, view_.condensed_rhs, view_.true_rhs_scratch);
    if (!CollectiveContinue(exact_rank_one_rhs) ||
        !CollectiveContinue(TransportPolicy::Synchronize(
            CudaTransferSite::rhs_remote_scatter)))
      return false;
  }
  else if (!AssembleTrueVector(
               view_.condensed_rhs, view_.true_rhs_scratch, true))
    return false;
  true_matrix_->HypreRead();
  hypre_rhs_->HypreRead();
  const bool ready = CudaHypreDevicePolicyValid(
      *true_matrix_, *hypre_rhs_, *hypre_solution_) &&
      true_rhs_generation_ != std::numeric_limits<std::uint64_t>::max();
  if (!CollectiveContinue(ready)) return false;
  ++true_rhs_generation_;
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
RefreshRetainedTrueRhs() noexcept
{
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_trace_assembly);
  const bool common_admission = (ranks_ == 1 || ranks_ == 2) &&
      true_matrix_ && hypre_rhs_ && hypre_solution_;
  const bool native_admitted = common_admission &&
      trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64 &&
      element_schwarz_plan_.valid &&
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          trace_policy_descriptor_.resolved_selection);
  const bool ordinary_admitted = common_admission &&
      ranks_ == 1 &&
      trace_solver_kind_ == CudaTraceSolverKind::hypre_parcsr_gmres &&
      CudaTracePreconditionerIsIlu(trace_preconditioner_) &&
      CudaOrdinaryRankOneRefreshPolicyIsExact(trace_policy_descriptor_);
  const bool host_staged_rank_two_admitted = common_admission &&
      ranks_ == 2 && host_staged_rank2_ &&
      trace_solver_kind_ == CudaTraceSolverKind::hypre_host_staged_rank2 &&
      CudaHostStagedRankTwoPolicyIsExact(trace_policy_descriptor_);
  bool refreshed = false;
  if (native_admitted)
  {
    refreshed = CudaLaunchAssembleNativeElementVector(
            view_, shape_, view_.condensed_rhs,
            view_.true_rhs_scratch) &&
        TransportPolicy::Synchronize(
            CudaTransferSite::rhs_remote_scatter);
  }
  else if (ordinary_admitted || host_staged_rank_two_admitted)
  {
    // AssembleTrueVector writes only the retained HYPRE RHS storage. It does
    // not touch true_matrix_ or any element Schur/LU data. Rank two performs
    // its established host-staged remote-vector gather inside this call.
    refreshed = AssembleTrueVector(
        view_.condensed_rhs, view_.true_rhs_scratch, true);
  }
  if (!CollectiveContinue(refreshed))
    return false;
  try
  {
    // HypreRead publishes the retained ParCSR operator and refreshed vector.
    // No matrix value, transpose cache, or local condensation factor is
    // rebuilt on any admitted RHS-only path.
    true_matrix_->HypreRead();
    hypre_rhs_->HypreRead();
  }
  catch (...) { return false; }
  const bool ready = CudaHypreDevicePolicyValid(
      *true_matrix_, *hypre_rhs_, *hypre_solution_) &&
      true_rhs_generation_ != std::numeric_limits<std::uint64_t>::max();
  if (!CollectiveContinue(ready)) return false;
  ++true_rhs_generation_;
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BuildAndObserveRankOneEquilibratedSystem(
    CudaEquilibrationObservation& observation) noexcept
{
  observation = {};
  if (trace_solver_kind_ !=
          CudaTraceSolverKind::hypre_fgmres_mgr_rank1 ||
      ranks_ != 1 || !shape_.igr_active || !true_matrix_ ||
      !equilibrated_true_matrix_ || !hypre_rhs_ || !hypre_solution_ ||
      !hypre_equilibrated_rhs_ || !rank_one_mgr_state_.configured ||
      view_.true_rhs_scratch == nullptr ||
      equilibrated_rhs_device_ == nullptr ||
      equilibration_left_device_ == nullptr ||
      equilibration_right_device_ == nullptr ||
      equilibration_row_norm_device_ == nullptr ||
      equilibration_column_norm_device_ == nullptr ||
      equilibration_control_device_ == nullptr)
    return false;

  CudaTrueMatrixDeviceView original{};
  CudaTrueMatrixDeviceView equilibrated{};
  bool vectors_exact = false;
  try
  {
    hypre_rhs_->HypreRead();
    hypre_equilibrated_rhs_->HypreReadWrite();
    vectors_exact = view_.true_rhs_scratch ==
            static_cast<const mfem::Vector&>(*hypre_rhs_).Read() &&
        equilibrated_rhs_device_ ==
            static_cast<mfem::Vector&>(
                *hypre_equilibrated_rhs_).ReadWrite();
  }
  catch (...) { vectors_exact = false; }
  const bool matrices_exact = vectors_exact &&
      CudaGetTrueMatrixDeviceView(*true_matrix_, original) &&
      CudaGetTrueMatrixDeviceView(*equilibrated_true_matrix_, equilibrated) &&
      original.rows == shape_.true_trace_count &&
      original.rows == original.diag_columns_count &&
      original.offd_nonzeros == 0 &&
      original.diag_offsets == true_device_.diag_offsets &&
      original.diag_columns == true_device_.diag_columns &&
      original.diag_values == true_device_.diag_values &&
      original.diag_nonzeros == true_device_.diag_nonzeros &&
      equilibrated.rows == original.rows &&
      equilibrated.diag_columns_count == original.diag_columns_count &&
      equilibrated.diag_nonzeros == original.diag_nonzeros &&
      equilibrated.offd_nonzeros == 0 &&
      equilibrated.diag_offsets == equilibrated_true_device_.diag_offsets &&
      equilibrated.diag_columns == equilibrated_true_device_.diag_columns &&
      equilibrated.diag_values == equilibrated_true_device_.diag_values;
  if (!matrices_exact) return false;

  rank_one_equilibration_host_control_.fill(0.0);
  int status = -1;
  const bool launched = CudaLaunchBuildRankOneEquilibratedSystem(
      original, equilibrated, view_.true_rhs_scratch,
      equilibrated_rhs_device_, equilibration_left_device_,
      equilibration_right_device_, equilibration_row_norm_device_,
      equilibration_column_norm_device_, equilibration_control_device_,
      view_.status_flag, cuda_rank1_equilibration_passes, nullptr);
  const std::size_t control_bytes =
      cuda_rank1_equilibration_control_doubles * sizeof(double);
  const bool observed = launched && TransportPolicy::DeviceToHost(
      rank_one_equilibration_host_control_.data(),
      equilibration_control_device_, control_bytes,
      CudaTransferCategory::scalar_diagnostic,
      CudaTransferSite::scalar_status, control_bytes,
      rank_one_equilibration_host_control_.size() * sizeof(double)) &&
      TransportPolicy::DeviceToHost(
          &status, view_.status_flag, sizeof(status),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) &&
      TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
  if (!observed || status != 0) return false;

  observation.passes = cuda_rank1_equilibration_passes;
  observation.minimum_left_scale =
      rank_one_equilibration_host_control_[0];
  observation.maximum_left_scale =
      rank_one_equilibration_host_control_[1];
  observation.minimum_right_scale =
      rank_one_equilibration_host_control_[2];
  observation.maximum_right_scale =
      rank_one_equilibration_host_control_[3];
  observation.finite =
      std::isfinite(observation.minimum_left_scale) &&
      std::isfinite(observation.maximum_left_scale) &&
      std::isfinite(observation.minimum_right_scale) &&
      std::isfinite(observation.maximum_right_scale) &&
      observation.minimum_left_scale >=
          cuda_rank1_equilibration_minimum_scale &&
      observation.maximum_left_scale <=
          cuda_rank1_equilibration_maximum_scale &&
      observation.minimum_right_scale >=
          cuda_rank1_equilibration_minimum_scale &&
      observation.maximum_right_scale <=
          cuda_rank1_equilibration_maximum_scale &&
      observation.minimum_left_scale <=
          observation.maximum_left_scale &&
      observation.minimum_right_scale <=
          observation.maximum_right_scale;
  if (!observation.finite) return false;
  // MGR restriction construction may have materialized transpose caches on
  // the previous generation. Numerical values have just changed in place,
  // so invalidate those caches before the new setup is allowed to inspect A.
  equilibrated_true_matrix_->ResetTranspose();
  equilibrated_true_matrix_->HypreRead();
  hypre_equilibrated_rhs_->HypreRead();
  return CudaHypreDevicePolicyValid(
      *equilibrated_true_matrix_, *hypre_equilibrated_rhs_,
      *hypre_solution_);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
LaunchAndObserveNativeRankOneTraceSolve(
    const CudaNativeIlu0FactorView& factor,
    const CudaNativeElementSchwarzView& element_schwarz,
    const std::uint64_t matrix_generation,
    const CudaNativeTraceActiveSubspace active_subspace,
    const double requested_relative_tolerance,
    CudaSolveDiagnostics& diagnostics,
    CudaNativeTraceSolveObservation* const solve_observation) noexcept
{
  const bool robust =
      CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
          trace_policy_descriptor_.resolved_selection) &&
      CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
          trace_policy_descriptor_.trace_preconditioner);
  const CudaTracePreconditionerKind effective_preconditioner = robust
      ? element_schwarz_attempt_state_.factor_preconditioner
      : trace_policy_descriptor_.trace_preconditioner;
  const bool effective_lower = effective_preconditioner ==
      CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64;
  const bool effective_coupled = effective_preconditioner ==
      CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_sigma_jacobi10_fp64;
  const bool projected_subspace =
      CudaNativeTraceActiveSubspaceIsProjected(active_subspace);
  const bool projected_context = RequiresInitialScalarClosure() &&
      cuda_engine_ == generic::CudaEngineKind::tensor_core &&
      ranks_ == 1 && shape_.igr_active &&
      shape_.trace_components == igr_trace_components &&
      trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64 &&
      robust && effective_lower;
  const CudaNativeTraceCooperativeSupport& effective_support =
      robust && effective_coupled
      ? native_trace_robust_coupled_support_ : native_trace_support_;
  if (!IsKnownCudaNativeTraceActiveSubspace(active_subspace) ||
      (projected_subspace && !projected_context) ||
      trace_solver_kind_ != CudaTraceSolverKind::native_rank1_fp64 ||
      ranks_ != 1 || !true_matrix_ || !hypre_rhs_ || !hypre_solution_ ||
      !native_trace_requirements_.valid ||
      !effective_support.supported ||
      (robust && !effective_lower && !effective_coupled) ||
      native_trace_requirements_.control_double_slots == 0 ||
      native_trace_requirements_.control_double_slots >
          native_trace_host_control_.size())
    return false;
  const bool use_ilu = CudaTracePreconditionerIsIlu(
      effective_preconditioner);
  const bool use_element_schwarz =
      CudaTracePreconditionerIsNativeElementSchwarz(
          effective_preconditioner);
  const bool use_matrix_free =
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          trace_policy_descriptor_.resolved_selection);
  const bool use_matrix_free_equilibrated_field_split =
      robust ? effective_lower :
          CudaTracePolicyUsesNativeElementMatrixFreeEquilibratedFieldSplit(
              trace_policy_descriptor_.resolved_selection);
  const bool use_matrix_free_warp_equilibrated_field_split =
      robust ? effective_lower :
          CudaTracePolicyUsesNativeElementMatrixFreeWarpEquilibratedFieldSplit(
              trace_policy_descriptor_.resolved_selection);
  const bool use_equilibrated_field_split =
      CudaTracePreconditionerIsNativeElementSchwarzEquilibratedFieldSplit(
          effective_preconditioner);
  const bool use_warp_equilibrated_field_split =
      CudaTracePreconditionerIsNativeElementSchwarzWarpEquilibratedFieldSplit(
          effective_preconditioner);
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
  const bool element_schwarz_exact = use_element_schwarz
      ? element_schwarz.patch_matrix ==
            view_.element_schwarz_patch_matrix &&
          element_schwarz.patch_lu == view_.element_schwarz_patch_lu &&
          element_schwarz.pivots == view_.element_schwarz_pivots &&
          element_schwarz.signed_element_true_dofs ==
              view_.element_schwarz_signed_true_dof &&
          element_schwarz.true_incidence_slots ==
              view_.element_schwarz_true_incidence_slots &&
          element_schwarz.element_rhs == view_.element_schwarz_rhs &&
          element_schwarz.element_solution ==
              view_.element_schwarz_solution &&
          element_schwarz.elements == shape_.elements &&
          element_schwarz.patch_rows == element_schwarz_plan_.patch_rows &&
          element_schwarz.true_rows == shape_.true_trace_count &&
          element_schwarz.minimum_multiplicity ==
              element_schwarz_plan_.minimum_multiplicity &&
          element_schwarz.maximum_multiplicity ==
              element_schwarz_plan_.maximum_multiplicity &&
          element_schwarz_plan_.valid
      : element_schwarz.patch_matrix == nullptr &&
          element_schwarz.patch_lu == nullptr &&
          element_schwarz.pivots == nullptr &&
          element_schwarz.signed_element_true_dofs == nullptr &&
          element_schwarz.true_incidence_slots == nullptr &&
          element_schwarz.element_rhs == nullptr &&
          element_schwarz.element_solution == nullptr &&
          element_schwarz.elements == 0 && element_schwarz.patch_rows == 0 &&
          element_schwarz.true_rows == 0;
  CudaNativeTraceOperatorView trace_operator{};
  trace_operator.kind = use_matrix_free_warp_equilibrated_field_split
      ? CudaNativeTraceOperatorKind::
            element_matrix_free_warp_equilibrated_field_split_fp64
      : use_matrix_free_equilibrated_field_split
          ? CudaNativeTraceOperatorKind::
                element_matrix_free_equilibrated_field_split_fp64
      : use_matrix_free
          ? CudaNativeTraceOperatorKind::element_matrix_free_fp64
          : CudaNativeTraceOperatorKind::assembled_csr_fp64;
  trace_operator.element_schur = use_matrix_free ? view_.schur : nullptr;
  const bool operator_exact = use_matrix_free
      ? use_element_schwarz && element_schwarz_exact && ranks_ == 1 &&
          trace_operator.element_schur == view_.schur &&
          element_schwarz.patch_rows == shape_.trace_dofs &&
          use_matrix_free_equilibrated_field_split ==
              use_equilibrated_field_split &&
          use_matrix_free_warp_equilibrated_field_split ==
              use_warp_equilibrated_field_split
      : trace_operator.element_schur == nullptr &&
          !use_matrix_free_equilibrated_field_split &&
          !use_matrix_free_warp_equilibrated_field_split;
  CudaTrueMatrixDeviceView current_matrix{};
  const bool matrix_exact = factor_exact && element_schwarz_exact &&
      operator_exact &&
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
      effective_preconditioner, factor,
      element_schwarz, trace_operator, active_subspace,
      native_trace_workspace_,
      effective_support,
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
  if (solve_observation != nullptr) *solve_observation = observation;

  diagnostics.matrix_generation = matrix_generation;
  diagnostics.trace_solver = CudaTraceSolverKind::native_rank1_fp64;
  diagnostics.trace_preconditioner = effective_preconditioner;
  diagnostics.igr_active = shape_.igr_active;
  diagnostics.krylov_dimension = krylov_dimension_;
  diagnostics.iterations = observation.iterations;
  diagnostics.reported_residual = observation.relative_residual_norm;
  diagnostics.device_policy = decoded && CudaHypreDevicePolicyValid(
      *true_matrix_, *hypre_rhs_, *hypre_solution_);
  diagnostics.configured_native_solver = decoded &&
      observation.ready_for_true_residual;
  diagnostics.configured_ilu = use_ilu && factor_exact;
  diagnostics.configured_diagonal =
      !use_ilu && !use_element_schwarz && factor_exact;
  diagnostics.configured_element_schwarz =
      use_element_schwarz && element_schwarz_exact;
  diagnostics.element_schwarz_preconditioner_applications =
      use_element_schwarz && observation.preconditioner_applications >= 0
      ? static_cast<std::uint64_t>(observation.preconditioner_applications)
      : 0;
  diagnostics.native_matrix_free_trace_operator_applications =
      use_matrix_free && observation.operator_applications >= 0
      ? static_cast<std::uint64_t>(observation.operator_applications) : 0;
  diagnostics.configured_preconditioner =
      diagnostics.configured_native_solver && factor_exact &&
      element_schwarz_exact;
  diagnostics.finite = decoded && observation.finite;
  diagnostics.reported_tolerance_met = diagnostics.finite &&
      diagnostics.reported_residual <= requested_relative_tolerance;

  const bool operator_accounting_ready = use_matrix_free
      ? observation.operator_applications >= 0 &&
          observation.actual_residual_checks >= 1 &&
          observation.operator_applications == observation.iterations +
              observation.actual_residual_checks - 1
      : observation.operator_applications == 0;
  bool native_accounting_ready = operator_accounting_ready;
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
                const CudaNativeTraceActiveSubspace active_subspace,
                const double requested_relative_tolerance,
                const bool same_matrix_retry,
                const bool same_operator_new_rhs,
                int& linear_iterations,
                CudaTensorCoreEvidence& tensor_core_evidence) noexcept
{
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_trace_solve);
  linear_iterations = 0;
  CudaSolveDiagnostics diagnostics{};
  const bool use_ilu = CudaTracePreconditionerIsIlu(
      trace_preconditioner_);
  const bool use_element_schwarz =
      CudaTracePreconditionerIsNativeElementSchwarz(
          trace_preconditioner_);
  const bool robust =
      CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
          trace_policy_descriptor_.resolved_selection) &&
      CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
          trace_preconditioner_);
  const bool projected_subspace =
      CudaNativeTraceActiveSubspaceIsProjected(active_subspace);
  const bool retained_operator = same_matrix_retry ||
      same_operator_new_rhs;
  const CudaTraceRhsTransition rhs_transition = same_matrix_retry
      ? CudaTraceRhsTransition::same_operator_same_rhs_retry
      : same_operator_new_rhs
          ? CudaTraceRhsTransition::same_operator_new_rhs
          : CudaTraceRhsTransition::new_operator;
  const bool rhs_transition_exact = !use_element_schwarz ||
      CudaElementSchwarzRhsTransitionIsExact(
          element_schwarz_attempt_state_, true_rhs_generation_,
          rhs_transition);
  const bool native_retained_rhs = !same_matrix_retry &&
      trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64 &&
      ranks_ == 1 && use_element_schwarz &&
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          trace_policy_descriptor_.resolved_selection) &&
      element_schwarz_attempt_state_.factor_ready &&
      element_schwarz_attempt_state_.factor_generation ==
          matrix_generation &&
      element_schwarz_attempt_state_.last_solve_generation ==
          matrix_generation &&
      true_rhs_generation_ >
          element_schwarz_attempt_state_.last_rhs_generation;
  const bool ordinary_retained_rhs = !same_matrix_retry &&
      trace_solver_kind_ == CudaTraceSolverKind::hypre_parcsr_gmres &&
      ranks_ == 1 && use_ilu &&
      CudaOrdinaryRankOneRefreshPolicyIsExact(
          trace_policy_descriptor_) &&
      CudaOrdinaryRankOneRefreshAttemptIsExact(
          ilu_factor_state_, current_attempt_,
          reinterpret_cast<std::uintptr_t>(true_matrix_.get()),
          reinterpret_cast<std::uintptr_t>(true_device_.diag_values),
          reinterpret_cast<std::uintptr_t>(true_device_.offd_values),
          true_device_.diag_nonzeros, true_device_.offd_nonzeros) &&
      true_rhs_generation_ > 0;
  const bool host_staged_rank_two_retained_rhs = !same_matrix_retry &&
      trace_solver_kind_ == CudaTraceSolverKind::hypre_host_staged_rank2 &&
      ranks_ == 2 && host_staged_rank2_ &&
      CudaHostStagedRankTwoPolicyIsExact(trace_policy_descriptor_) &&
      true_rhs_generation_ > 0;
  const bool same_operator_rhs_admitted = !same_operator_new_rhs ||
      native_retained_rhs || ordinary_retained_rhs ||
      host_staged_rank_two_retained_rhs;
  if (!IsKnownCudaNativeTraceActiveSubspace(active_subspace) ||
      (projected_subspace &&
       trace_solver_kind_ != CudaTraceSolverKind::native_rank1_fp64) ||
      !rhs_transition_exact || !same_operator_rhs_admitted)
    return false;
  std::uint64_t robust_lower_setup_delta = 0;
  std::uint64_t robust_coupled_setup_delta = 0;
  std::uint64_t robust_lower_solve_delta = 0;
  std::uint64_t robust_lower_rejected_solve_delta = 0;
  std::uint64_t robust_coupled_solve_delta = 0;
  bool solved = false;
  if (trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64)
  {
    CudaNativeIlu0FactorView factor{};
    CudaNativeElementSchwarzView element_schwarz{};
    if (use_element_schwarz)
    {
      element_schwarz.patch_matrix = view_.element_schwarz_patch_matrix;
      element_schwarz.patch_lu = view_.element_schwarz_patch_lu;
      element_schwarz.pivots = view_.element_schwarz_pivots;
      element_schwarz.signed_element_true_dofs =
          view_.element_schwarz_signed_true_dof;
      element_schwarz.true_incidence_slots =
          view_.element_schwarz_true_incidence_slots;
      element_schwarz.element_rhs = view_.element_schwarz_rhs;
      element_schwarz.element_solution = view_.element_schwarz_solution;
      element_schwarz.elements = shape_.elements;
      element_schwarz.patch_rows = element_schwarz_plan_.patch_rows;
      element_schwarz.true_rows = shape_.true_trace_count;
      element_schwarz.minimum_multiplicity =
          element_schwarz_plan_.minimum_multiplicity;
      element_schwarz.maximum_multiplicity =
          element_schwarz_plan_.maximum_multiplicity;
    }
    const bool ilu_factor_ready = !use_ilu ||
        CudaPrepareNativeRankOneIluFactor(
            *true_matrix_, *hypre_rhs_, *hypre_solution_, *ilu_,
            trace_policy_descriptor_, current_attempt_, matrix_generation,
            native_ilu_inverse_diagonal_,
            generic_arena_layout_.native_ilu_inverse_diagonal,
            ilu_factor_state_, factor, diagnostics, same_matrix_retry);
    bool schwarz_factor_ready = !use_element_schwarz;
    if (use_element_schwarz)
    {
      const bool transition_ready = element_schwarz_plan_.valid &&
          CudaElementSchwarzFactorTransitionIsExact(
              element_schwarz_attempt_state_, current_attempt_,
              matrix_generation, retained_operator);
      if (retained_operator)
      {
        const CudaTracePreconditionerKind retained =
            element_schwarz_attempt_state_.factor_preconditioner;
        const bool retained_exact = !robust ||
            retained == CudaTracePreconditionerKind::
                native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64 ||
            retained == CudaTracePreconditionerKind::
                native_element_schwarz_warp_coupled_sigma_jacobi10_fp64;
        schwarz_factor_ready = transition_ready && retained_exact;
      }
      else
      {
        CudaTracePreconditionerKind effective_preconditioner =
            trace_preconditioner_;
        if (transition_ready && robust)
        {
          constexpr CudaTracePreconditionerKind lower =
              CudaTracePreconditionerKind::
                  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64;
          constexpr CudaTracePreconditionerKind coupled =
              CudaTracePreconditionerKind::
                  native_element_schwarz_warp_coupled_sigma_jacobi10_fp64;
          const auto observe_setup = [&](
              const CudaTracePreconditionerKind kind,
              CudaNativeElementSchwarzSetupStatus& status) noexcept
          {
            status = CudaNativeElementSchwarzSetupStatus::unobserved;
            int host_status = static_cast<int>(
                CudaNativeElementSchwarzSetupStatus::invalid_or_nonfinite);
            const bool observed =
                CudaLaunchSetupNativeElementSchwarz(view_, shape_, kind) &&
                CudaLaunchClassifyNativeElementSchwarzSetup(
                    view_, shape_, view_.status_flag) &&
                TransportPolicy::DeviceToHost(
                    &host_status, view_.status_flag, sizeof(host_status),
                    CudaTransferCategory::scalar_diagnostic,
                    CudaTransferSite::scalar_status) &&
                TransportPolicy::Synchronize(
                    CudaTransferSite::scalar_status);
            if (!observed ||
                host_status < static_cast<int>(
                    CudaNativeElementSchwarzSetupStatus::ready) ||
                host_status > static_cast<int>(
                    CudaNativeElementSchwarzSetupStatus::
                        invalid_or_nonfinite))
            {
              tensor_core_evidence.robust_branch_accounting_complete = false;
              tensor_core_evidence.trace_solver_accounting_complete = false;
              return false;
            }
            status = static_cast<CudaNativeElementSchwarzSetupStatus>(
                host_status);
            return true;
          };
          CudaNativeElementSchwarzSetupStatus lower_status{};
          const bool lower_observed = observe_setup(lower, lower_status);
          const CudaRobustElementSchwarzSetupAction lower_action =
              lower_observed
              ? CudaRobustElementSchwarzSetupActionFor(lower_status)
              : CudaRobustElementSchwarzSetupAction::fail_closed;
          if (lower_observed)
          {
            robust_lower_setup_delta = 1;
            diagnostics.element_schwarz_equilibration_builds = 1;
            diagnostics.element_schwarz_equilibration_passes =
                native_element_schwarz_equilibration_passes;
          }
          if (lower_action ==
              CudaRobustElementSchwarzSetupAction::use_lower)
          {
            effective_preconditioner = lower;
            schwarz_factor_ready = true;
          }
          else if (!projected_subspace && lower_action ==
                   CudaRobustElementSchwarzSetupAction::try_coupled)
          {
            CudaNativeElementSchwarzSetupStatus coupled_status{};
            const bool coupled_observed =
                observe_setup(coupled, coupled_status);
            if (coupled_observed) robust_coupled_setup_delta = 1;
            schwarz_factor_ready = coupled_observed &&
                coupled_status ==
                    CudaNativeElementSchwarzSetupStatus::ready;
            if (schwarz_factor_ready) effective_preconditioner = coupled;
          }
        }
        else if (transition_ready)
        {
          schwarz_factor_ready =
              CudaLaunchSetupNativeElementSchwarz(
                  view_, shape_, trace_preconditioner_) &&
              StatusPassed();
        }
        if (schwarz_factor_ready)
        {
          element_schwarz_attempt_state_.factor_ready = true;
          element_schwarz_attempt_state_.factor_generation =
              matrix_generation;
          element_schwarz_attempt_state_.factor_preconditioner =
              effective_preconditioner;
          diagnostics.element_schwarz_patch_setups = 1;
          if (!robust &&
              CudaTracePreconditionerIsNativeElementSchwarzEquilibratedFieldSplit(
                  trace_preconditioner_))
          {
            diagnostics.element_schwarz_equilibration_builds = 1;
            diagnostics.element_schwarz_equilibration_passes =
                native_element_schwarz_equilibration_passes;
          }
        }
      }
    }
    CudaNativeTraceSolveObservation native_observation{};
    solved = ilu_factor_ready && schwarz_factor_ready &&
        LaunchAndObserveNativeRankOneTraceSolve(
            factor, element_schwarz, matrix_generation,
            active_subspace,
            requested_relative_tolerance, diagnostics,
            &native_observation);
    constexpr std::uint32_t invalid_preconditioner_failure_mask = 1U << 2;
    const bool lower_application_structurally_rejected =
        robust && !projected_subspace && !retained_operator && !solved &&
        diagnostics.device_policy &&
        schwarz_factor_ready &&
        element_schwarz_attempt_state_.factor_preconditioner ==
            CudaTracePreconditionerKind::
                native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64 &&
        native_observation.status ==
            native_trace_solve_status_invalid_preconditioner &&
        native_observation.failure_mask ==
            invalid_preconditioner_failure_mask &&
        native_observation.iterations == 0 &&
        native_observation.restart_cycles == 1 &&
        native_observation.preconditioner_applications == 1 &&
        native_observation.operator_applications == 0 &&
        native_observation.actual_residual_checks == 1 &&
        native_observation.finite &&
        !native_observation.ready_for_true_residual;
    if (lower_application_structurally_rejected)
    {
      // The lower field split passed factor construction but its first actual
      // application failed the retained-patch backward-error check before any
      // operator or Krylov work.  This exact terminal tuple completes robust
      // applicability classification; every other native failure remains
      // fail-closed.  Count the rejected physical call as recurring work.
      robust_lower_solve_delta = 1;
      robust_lower_rejected_solve_delta = 1;
      const std::uint64_t lower_preconditioner_applications =
          diagnostics.element_schwarz_preconditioner_applications;
      const std::uint64_t lower_operator_applications =
          diagnostics.native_matrix_free_trace_operator_applications;
      constexpr CudaTracePreconditionerKind coupled =
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_coupled_sigma_jacobi10_fp64;
      element_schwarz_attempt_state_.factor_ready = false;
      element_schwarz_attempt_state_.factor_preconditioner =
          CudaTracePreconditionerKind::unconfigured;
      CudaNativeElementSchwarzSetupStatus coupled_status =
          CudaNativeElementSchwarzSetupStatus::unobserved;
      int host_status = static_cast<int>(
          CudaNativeElementSchwarzSetupStatus::invalid_or_nonfinite);
      const bool coupled_observed =
          CudaLaunchSetupNativeElementSchwarz(view_, shape_, coupled) &&
          CudaLaunchClassifyNativeElementSchwarzSetup(
              view_, shape_, view_.status_flag) &&
          TransportPolicy::DeviceToHost(
              &host_status, view_.status_flag, sizeof(host_status),
              CudaTransferCategory::scalar_diagnostic,
              CudaTransferSite::scalar_status) &&
          TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
          host_status >= static_cast<int>(
              CudaNativeElementSchwarzSetupStatus::ready) &&
          host_status <= static_cast<int>(
              CudaNativeElementSchwarzSetupStatus::invalid_or_nonfinite);
      if (coupled_observed)
      {
        coupled_status = static_cast<CudaNativeElementSchwarzSetupStatus>(
            host_status);
        robust_coupled_setup_delta = 1;
      }
      else
      {
        tensor_core_evidence.robust_branch_accounting_complete = false;
        tensor_core_evidence.trace_solver_accounting_complete = false;
      }
      const bool coupled_ready = coupled_observed &&
          coupled_status == CudaNativeElementSchwarzSetupStatus::ready;
      if (coupled_ready)
      {
        element_schwarz_attempt_state_.factor_ready = true;
        element_schwarz_attempt_state_.factor_generation =
            matrix_generation;
        element_schwarz_attempt_state_.factor_preconditioner = coupled;
        CudaSolveDiagnostics coupled_diagnostics{};
        solved = LaunchAndObserveNativeRankOneTraceSolve(
            factor, element_schwarz, matrix_generation,
            active_subspace,
            requested_relative_tolerance, coupled_diagnostics);
        const bool work_bounded =
            lower_preconditioner_applications <=
                std::numeric_limits<std::uint64_t>::max() -
                    coupled_diagnostics.
                        element_schwarz_preconditioner_applications &&
            lower_operator_applications <=
                std::numeric_limits<std::uint64_t>::max() -
                    coupled_diagnostics.
                        native_matrix_free_trace_operator_applications;
        if (work_bounded)
        {
          coupled_diagnostics.element_schwarz_preconditioner_applications +=
              lower_preconditioner_applications;
          coupled_diagnostics.native_matrix_free_trace_operator_applications +=
              lower_operator_applications;
          coupled_diagnostics.element_schwarz_patch_setups =
              robust_lower_setup_delta + robust_coupled_setup_delta;
          coupled_diagnostics.element_schwarz_equilibration_builds =
              diagnostics.element_schwarz_equilibration_builds;
          coupled_diagnostics.element_schwarz_equilibration_passes =
              diagnostics.element_schwarz_equilibration_passes;
          diagnostics = coupled_diagnostics;
        }
        else
        {
          tensor_core_evidence.robust_branch_accounting_complete = false;
          tensor_core_evidence.trace_solver_accounting_complete = false;
          solved = false;
        }
      }
      else
      {
        element_schwarz_attempt_state_.factor_ready = false;
        solved = false;
      }
    }
    if (robust)
      diagnostics.element_schwarz_patch_setups =
          robust_lower_setup_delta + robust_coupled_setup_delta;
    if (solved && use_ilu)
    {
      const bool state_generation_ready = ilu_factor_state_.factor_ready &&
          ilu_factor_state_.factor_generation <= matrix_generation &&
          (retained_operator
               ? matrix_generation == ilu_factor_state_.last_solve_generation
               : matrix_generation > ilu_factor_state_.last_solve_generation);
      if (state_generation_ready)
        ilu_factor_state_.last_solve_generation = matrix_generation;
      else
        solved = false;
    }
    if (solved && use_element_schwarz)
    {
      const bool state_generation_ready =
          element_schwarz_attempt_state_.factor_ready &&
          element_schwarz_attempt_state_.factor_generation ==
              matrix_generation &&
          (retained_operator
               ? matrix_generation ==
                   element_schwarz_attempt_state_.last_solve_generation
               : matrix_generation >
                   element_schwarz_attempt_state_.last_solve_generation) &&
          (same_matrix_retry
               ? true_rhs_generation_ ==
                   element_schwarz_attempt_state_.last_rhs_generation
               : true_rhs_generation_ >
                   element_schwarz_attempt_state_.last_rhs_generation);
      if (state_generation_ready)
      {
        element_schwarz_attempt_state_.last_solve_generation =
            matrix_generation;
        element_schwarz_attempt_state_.last_rhs_generation =
            true_rhs_generation_;
      }
      else
      {
        if (robust)
        {
          tensor_core_evidence.robust_branch_accounting_complete = false;
          tensor_core_evidence.trace_solver_accounting_complete = false;
        }
        solved = false;
      }
    }
    if (solved && robust)
    {
      if (element_schwarz_attempt_state_.factor_preconditioner ==
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64)
        robust_lower_solve_delta = 1;
      else if (element_schwarz_attempt_state_.factor_preconditioner ==
               CudaTracePreconditionerKind::
                   native_element_schwarz_warp_coupled_sigma_jacobi10_fp64)
        robust_coupled_solve_delta = 1;
      else
      {
        tensor_core_evidence.robust_branch_accounting_complete = false;
        tensor_core_evidence.trace_solver_accounting_complete = false;
        solved = false;
      }
    }
  }
  else if (trace_solver_kind_ ==
           CudaTraceSolverKind::hypre_fgmres_mgr_rank1)
  {
    CudaEquilibrationObservation equilibration{};
    const bool equilibration_ready = same_matrix_retry ||
        BuildAndObserveRankOneEquilibratedSystem(equilibration);
    solved = equilibration_ready &&
        CudaSolveRankOneMgr(
            rank_one_mgr_state_, *equilibrated_true_matrix_,
            *hypre_equilibrated_rhs_, *hypre_solution_,
            trace_policy_descriptor_, requested_relative_tolerance,
            maximum_linear_iterations_, krylov_dimension_,
            matrix_generation, diagnostics, same_matrix_retry) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status) &&
        CudaLaunchUnscaleRankOneSolution(
            view_.true_solution, equilibration_right_device_,
            shape_.true_trace_count, nullptr) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    const auto checked_add = [](std::uint64_t& target,
                                const std::uint64_t value) noexcept
    {
      if (value > std::numeric_limits<std::uint64_t>::max() - target)
        return false;
      target += value;
      return true;
    };
    if (solved)
    {
      const bool new_equilibration = !same_matrix_retry;
      const bool first_equilibration = new_equilibration &&
          tensor_core_evidence.equilibration_builds == 0;
      const bool accounted =
          (!new_equilibration ||
           (checked_add(tensor_core_evidence.equilibration_builds, 1) &&
            checked_add(tensor_core_evidence.equilibration_passes,
                        static_cast<std::uint64_t>(
                            equilibration.passes)))) &&
          checked_add(tensor_core_evidence.mgr_setup_calls,
                      diagnostics.mgr_setup_calls) &&
          checked_add(
              tensor_core_evidence.mgr_preconditioner_applications,
              diagnostics.mgr_preconditioner_applications) &&
          checked_add(tensor_core_evidence.mgr_f_ilu_setups,
                      diagnostics.mgr_f_ilu_setups) &&
          checked_add(tensor_core_evidence.mgr_f_ilu_applications,
                      diagnostics.mgr_f_ilu_applications);
      if (!accounted)
      {
        tensor_core_evidence.trace_solver_accounting_complete = false;
        solved = false;
      }
      else
      {
        if (first_equilibration)
        {
          tensor_core_evidence.minimum_left_scale =
              equilibration.minimum_left_scale;
          tensor_core_evidence.maximum_left_scale =
              equilibration.maximum_left_scale;
          tensor_core_evidence.minimum_right_scale =
              equilibration.minimum_right_scale;
          tensor_core_evidence.maximum_right_scale =
              equilibration.maximum_right_scale;
        }
        else if (new_equilibration)
        {
          tensor_core_evidence.minimum_left_scale = std::min(
              tensor_core_evidence.minimum_left_scale,
              equilibration.minimum_left_scale);
          tensor_core_evidence.maximum_left_scale = std::max(
              tensor_core_evidence.maximum_left_scale,
              equilibration.maximum_left_scale);
          tensor_core_evidence.minimum_right_scale = std::min(
              tensor_core_evidence.minimum_right_scale,
              equilibration.minimum_right_scale);
          tensor_core_evidence.maximum_right_scale = std::max(
              tensor_core_evidence.maximum_right_scale,
              equilibration.maximum_right_scale);
        }
      }
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
            retained_operator);
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
  if (solved && robust)
  {
    const bool setup_deltas_exact = retained_operator
        ? robust_lower_setup_delta == 0 &&
            robust_coupled_setup_delta == 0 &&
            robust_lower_rejected_solve_delta == 0
        : robust_lower_setup_delta == 1 &&
            ((robust_coupled_setup_delta == 0 &&
              robust_lower_solve_delta == 1 &&
              robust_lower_rejected_solve_delta == 0 &&
              robust_coupled_solve_delta == 0) ||
             (robust_coupled_setup_delta == 1 &&
              robust_coupled_solve_delta == 1 &&
              robust_lower_solve_delta ==
                  robust_lower_rejected_solve_delta &&
              robust_lower_rejected_solve_delta <= 1));
    const bool solve_deltas_exact =
        robust_lower_rejected_solve_delta <=
            robust_lower_solve_delta &&
        robust_lower_solve_delta - robust_lower_rejected_solve_delta +
                robust_coupled_solve_delta ==
            1 &&
        robust_lower_solve_delta + robust_coupled_solve_delta ==
            1 + robust_lower_rejected_solve_delta;
    const bool deltas_bounded =
        robust_lower_setup_delta <=
            std::numeric_limits<std::uint64_t>::max() -
                tensor_core_evidence.robust_lower_patch_setups &&
        robust_coupled_setup_delta <=
            std::numeric_limits<std::uint64_t>::max() -
                tensor_core_evidence.robust_coupled_patch_setups &&
        robust_lower_solve_delta <=
            std::numeric_limits<std::uint64_t>::max() -
                tensor_core_evidence.robust_lower_trace_solve_calls &&
        robust_coupled_solve_delta <=
            std::numeric_limits<std::uint64_t>::max() -
                tensor_core_evidence.robust_coupled_trace_solve_calls;
    if (!tensor_core_evidence.robust_branch_accounting_complete ||
        !setup_deltas_exact || !solve_deltas_exact || !deltas_bounded)
    {
      tensor_core_evidence.robust_branch_accounting_complete = false;
      tensor_core_evidence.trace_solver_accounting_complete = false;
      solved = false;
    }
  }
  if (use_ilu && solved)
    (void)CudaAccumulateIluSolveEvidence(
        tensor_core_evidence, diagnostics);
  if (use_element_schwarz && solved)
  {
    const auto checked_add = [](std::uint64_t& target,
                                const std::uint64_t value) noexcept
    {
      if (value > std::numeric_limits<std::uint64_t>::max() - target)
        return false;
      target += value;
      return true;
    };
    const bool accounted =
        tensor_core_evidence.native_element_schwarz_configured &&
        tensor_core_evidence.element_schwarz_patch_rows ==
            element_schwarz_plan_.patch_rows &&
        tensor_core_evidence.element_schwarz_minimum_multiplicity ==
            element_schwarz_plan_.minimum_multiplicity &&
        tensor_core_evidence.element_schwarz_maximum_multiplicity ==
            element_schwarz_plan_.maximum_multiplicity &&
        checked_add(tensor_core_evidence.element_schwarz_patch_setups,
                    diagnostics.element_schwarz_patch_setups) &&
        checked_add(
            tensor_core_evidence.element_schwarz_preconditioner_applications,
            diagnostics.element_schwarz_preconditioner_applications) &&
        checked_add(
            tensor_core_evidence.
                native_matrix_free_trace_operator_applications,
            diagnostics.native_matrix_free_trace_operator_applications) &&
        checked_add(tensor_core_evidence.equilibration_builds,
                    diagnostics.element_schwarz_equilibration_builds) &&
        checked_add(tensor_core_evidence.equilibration_passes,
                    diagnostics.element_schwarz_equilibration_passes);
    if (!accounted)
    {
      tensor_core_evidence.trace_solver_accounting_complete = false;
      solved = false;
    }
  }
  if (solved)
  {
    std::uint64_t* const calls = trace_solver_kind_ ==
            CudaTraceSolverKind::native_rank1_fp64
        ? &tensor_core_evidence.native_rank1_trace_solve_calls
        : &tensor_core_evidence.hypre_trace_solve_calls;
    const std::uint64_t call_delta = robust
        ? robust_lower_solve_delta + robust_coupled_solve_delta : 1;
    if (call_delta == 0 ||
        call_delta > std::numeric_limits<std::uint64_t>::max() - *calls)
    {
      tensor_core_evidence.trace_solver_accounting_complete = false;
      solved = false;
    }
    else
      *calls += call_delta;
  }
  if (!solved && robust)
  {
    // Any failed robust invocation may have launched setup or solve work that
    // cannot be represented by the transactional branch counters.  Keep every
    // branch counter unchanged and fail all aggregate accounting closed instead
    // of leaving a plausible partial census.
    tensor_core_evidence.robust_branch_accounting_complete = false;
    tensor_core_evidence.trace_solver_accounting_complete = false;
  }
  if (solved && robust)
  {
    // These deltas were fully validated before any generic Schwarz evidence
    // was mutated.  Commit them only after both that evidence and the global
    // native-solve call count have succeeded, so branch accounting never
    // exposes a partially completed invocation.
    tensor_core_evidence.robust_lower_patch_setups +=
        robust_lower_setup_delta;
    tensor_core_evidence.robust_coupled_patch_setups +=
        robust_coupled_setup_delta;
    tensor_core_evidence.robust_lower_trace_solve_calls +=
        robust_lower_solve_delta;
    tensor_core_evidence.robust_coupled_trace_solve_calls +=
        robust_coupled_solve_delta;
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
                const CudaNativeTraceActiveSubspace active_subspace,
                double& true_relative_residual,
                const std::uint64_t matrix_generation,
                int& linear_iterations,
                int& linear_solve_attempts,
                int& true_residual_tightenings,
                CudaTensorCoreEvidence& tensor_core_evidence,
                const bool same_operator_new_rhs) noexcept
{
  linear_iterations = 0;
  linear_solve_attempts = 0;
  true_residual_tightenings = 0;
  const double requested_tolerance =
      role == generic::EulerHdgLinearSolveRole::newton_correction
      ? newton_correction_linear_relative_tolerance_
      : final_coherent_linear_relative_tolerance_;
  const auto invoke = [&](const double tolerance,
                          const bool same_matrix_retry,
                          const bool retained_new_rhs) noexcept
  {
    int invocation_iterations = 0;
    const bool solved = SolveTrueSystemAttempt(
        true_relative_residual, matrix_generation, active_subspace, tolerance,
        same_matrix_retry, retained_new_rhs, invocation_iterations,
        tensor_core_evidence);
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

  if (!invoke(requested_tolerance, false, same_operator_new_rhs)) return false;
  if (true_relative_residual <= true_residual_relative_tolerance_)
    return true;
  const bool tighten =
      true_residual_tightening_linear_relative_tolerance_ <
          requested_tolerance;
  if (!CollectiveContinue(tighten)) return false;
  ++true_residual_tightenings;
  return invoke(true_residual_tightening_linear_relative_tolerance_, true,
                false) &&
      true_relative_residual <= true_residual_relative_tolerance_;
}

template <typename AllocatorPolicy, typename TransportPolicy>
auto CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
ExecuteRodas3(const CudaConfig& config, const AttemptIdentity& attempt,
              state_slot& workspace) noexcept -> CudaNumericalResult
{
  using Coefficients = ::dg::hdg::time::Rodas3Coefficients;
  ::igrlab::NVTXRange accepted_step_profile_region(
      ::igrlab::observability::RegionId::hdg_accepted_step);
  CudaIluFactorAttemptEndGuard end_factor_attempt(ilu_factor_state_);
  CudaElementSchwarzAttemptEndGuard end_schwarz_attempt(
      element_schwarz_attempt_state_);
  CudaNumericalResult result{};
  result.tensor_core.selected_engine = cuda_engine_;
  result.tensor_core.requested_trace_policy =
      trace_policy_descriptor_.requested_selection;
  result.tensor_core.resolved_trace_policy =
      trace_policy_descriptor_.resolved_selection;
  result.tensor_core.trace_policy_rank_count =
      trace_policy_descriptor_.rank_count;
  result.tensor_core.trace_policy_order =
      trace_policy_descriptor_.polynomial_order;
  result.tensor_core.trace_preconditioner = trace_preconditioner_;
  result.tensor_core.trace_preconditioner_configured =
      CudaTracePolicyDescriptorIsExact(trace_policy_descriptor_) &&
      trace_preconditioner_ == trace_policy_descriptor_.trace_preconditioner;
  result.tensor_core.native_rank1_trace_solver_configured =
      trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64;
  result.tensor_core.hypre_host_staged_rank2_trace_solver_configured =
      trace_solver_kind_ == CudaTraceSolverKind::hypre_host_staged_rank2;
  result.tensor_core.native_element_schwarz_configured =
      CudaTracePreconditionerIsNativeElementSchwarz(trace_preconditioner_) &&
      element_schwarz_plan_.valid;
  result.tensor_core.native_matrix_free_trace_operator_configured =
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          trace_policy_descriptor_.resolved_selection) &&
      result.tensor_core.native_element_schwarz_configured;
  const bool robust_trace_policy =
      CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
          trace_policy_descriptor_.resolved_selection) &&
      CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
          trace_preconditioner_);
  result.tensor_core.robust_lower_specialization_supported =
      robust_trace_policy && native_trace_support_.supported;
  result.tensor_core.robust_coupled_specialization_supported =
      robust_trace_policy && native_trace_robust_coupled_support_.supported;
  result.tensor_core.robust_branch_accounting_complete =
      !robust_trace_policy ||
      (result.tensor_core.robust_lower_specialization_supported &&
       result.tensor_core.robust_coupled_specialization_supported);
  result.tensor_core.element_schwarz_patch_rows =
      element_schwarz_plan_.patch_rows;
  result.tensor_core.element_schwarz_minimum_multiplicity =
      element_schwarz_plan_.minimum_multiplicity;
  result.tensor_core.element_schwarz_maximum_multiplicity =
      element_schwarz_plan_.maximum_multiplicity;
  result.tensor_core.trace_solver_accounting_complete = true;
  result.tensor_core.hardware_supported =
      tensor_core_hardware_supported_;
  // Every admitted RODAS3 route is fixed before device work and has no
  // fallback: rank-one tensor-native, rank-one ordinary ParCSR, or the exact
  // engine-default rank-two host-staged ParCSR leaf.
  result.tensor_core.fallback_accounting_complete = true;
  const bool reduced_precision_volume =
      detail::CudaResearchVolumeUsesReducedPrecision(
          shape_.borrowed_host_inputs &&
          cuda_engine_ == generic::CudaEngineKind::tensor_core);
  result.precision_work.all_fp64_policy = !reduced_precision_volume;
  result.precision_work.accounting_complete = !reduced_precision_volume;
  (void)CudaInitializeIluEvidence(
      result.tensor_core, trace_preconditioner_, shape_.igr_active);

  accepted_allocation_events_ = AllocatorPolicy::AllocationEvents();
  const bool tensor_rank_one =
      cuda_engine_ == generic::CudaEngineKind::tensor_core &&
      trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64 &&
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          trace_policy_descriptor_.resolved_selection) &&
      CudaTracePreconditionerIsNativeElementSchwarz(trace_preconditioner_) &&
      element_schwarz_plan_.valid;
  const bool ordinary_rank_one =
      cuda_engine_ == generic::CudaEngineKind::ordinary &&
      trace_solver_kind_ == CudaTraceSolverKind::hypre_parcsr_gmres &&
      CudaTracePreconditionerIsIlu(trace_preconditioner_) &&
      CudaOrdinaryRankOneRefreshPolicyIsExact(trace_policy_descriptor_);
  const bool host_staged_rank_two =
      ranks_ == 2 && host_staged_rank2_ &&
      trace_solver_kind_ == CudaTraceSolverKind::hypre_host_staged_rank2 &&
      CudaHostStagedRankTwoPolicyIsExact(trace_policy_descriptor_);
  const bool ilu_common_shift =
      CudaTracePreconditionerIsIlu(trace_preconditioner_);
  const bool diagonal_common_shift = trace_preconditioner_ ==
      CudaTracePreconditionerKind::hypre_diag_scale;
  const double algebraic_residual_allowance = std::max(
      nonlinear_relative_tolerance_,
      1024.0 * std::numeric_limits<double>::epsilon());
  const bool positivity_enabled =
      config.numerical_policy.positivity_limiter.kind ==
          generic::EulerHdgPositivityLimiterKind::zhang_shu;
  const bool admitted = initialized_ && rodas3_ &&
      time_treatment_ == generic::EulerHdgTimeTreatment::rodas3 &&
      (ranks_ == 1 || ranks_ == 2) && shape_.borrowed_host_inputs &&
      (tensor_rank_one || ordinary_rank_one || host_staged_rank_two) &&
      !config.capture_oracle &&
      std::isfinite(attempt.dt) && attempt.dt > 0.0 &&
      workspace.interior.coefficients == view_.work_interior &&
      workspace.trace.coefficients == view_.work_true_trace &&
      view_.rodas_residual_f != nullptr &&
      view_.rodas_residual_g != nullptr &&
      view_.rodas_operator_b != nullptr &&
      view_.rodas_operator_c != nullptr &&
      view_.rodas_combination_interior != nullptr &&
      view_.rodas_combination_true_trace != nullptr &&
      (ranks_ == 1 ||
       std::all_of(std::begin(view_.rodas_stage_local_trace),
                   std::end(view_.rodas_stage_local_trace),
                   [](const double* value) { return value != nullptr; }));
  if (!CollectiveContinue(admitted))
  {
    result.phase = CudaFailurePhase::setup;
    result.reason = FailureReason::backend_failure;
    return result;
  }

  const bool numerical_injection_phase =
      config.injection.phase == CudaFailurePhase::assembly ||
      config.injection.phase == CudaFailurePhase::condensation ||
      config.injection.phase == CudaFailurePhase::trace_solve ||
      config.injection.phase == CudaFailurePhase::recovery ||
      config.injection.phase ==
          CudaFailurePhase::after_recovery_before_commit ||
      config.injection.phase == CudaFailurePhase::diagnostics;
  const bool injection_config_valid = !config.injection.enabled ||
      (numerical_injection_phase && config.injection.target_step != 0 &&
       config.injection.source_rank >= 0 &&
       config.injection.source_rank < ranks_);
  if (!CollectiveContinue(injection_config_valid))
  {
    result.reason = FailureReason::collective_protocol;
    return result;
  }
  const bool injection_active = config.injection.enabled &&
      config.injection.target_step == attempt.committed_step + 1 &&
      attempt.retry_index == 0;
  const auto inject_at = [&](const CudaFailurePhase phase) noexcept
  {
    result.phase = phase;
    if (!injection_active || config.injection.phase != phase) return false;
    const int local_claim = rank_ == config.injection.source_rank ? 1 : 0;
    int claims = 0;
    if (MPI_Allreduce(&local_claim, &claims, 1, MPI_INT, MPI_SUM,
                      communicator_) != MPI_SUCCESS || claims != 1)
    {
      result.reason = FailureReason::collective_protocol;
      return true;
    }
    result.injected = true;
    result.injection_source_rank = config.injection.source_rank;
    result.reason = FailureReason::backend_failure;
    return true;
  };
  const auto enter = [&](const CudaFailurePhase phase) noexcept
  {
    return !inject_at(phase) &&
        CollectiveContinue(Ticket(phase));
  };
  const auto add_volume_work = [&](const std::uint64_t mma_tiles) noexcept
  {
    if (result.tensor_core.volume_assembly_launches ==
            std::numeric_limits<std::uint64_t>::max() ||
        mma_tiles > std::numeric_limits<std::uint64_t>::max() -
            result.tensor_core.volume_mma_tiles)
      return false;
    ++result.tensor_core.volume_assembly_launches;
    result.tensor_core.volume_mma_tiles += mma_tiles;
    ++result.complete_assemblies;
    return true;
  };
  const auto assemble_spatial = [&](const bool algebraic_mask,
                                    double* residual_norm) noexcept
  {
    const std::size_t interior_bytes = workspace.interior.count *
        sizeof(double);
    const bool staged = TransportPolicy::DeviceToDevice(
        view_.old_interior, workspace.interior.coefficients, interior_bytes,
        CudaTransferCategory::attempt_history,
        CudaTransferSite::attempt_seed_interior);
    if (!CollectiveContinue(staged)) return false;
    std::uint64_t mma_tiles = 0;
    bool launched = false;
    {
      ::igrlab::NVTXRange profile_region(
          ::igrlab::observability::RegionId::hdg_local_assembly);
      launched = CudaLaunchAssembly(
          view_, shape_, Coefficients::ShiftScale(attempt.dt), cuda_engine_,
          mma_tiles, true);
    }
    if (!CollectiveContinue(launched && StatusPassed()) ||
        !CollectiveContinue(add_volume_work(mma_tiles)))
      return false;
    if (algebraic_mask &&
        !CollectiveContinue(CudaLaunchRodasAlgebraicMask(view_, shape_) &&
                            StatusPassed()))
      return false;
    if (residual_norm == nullptr) return true;
    if (!AssembleTrueVector(
            view_.residual_g, view_.true_residual, false) ||
        !CollectiveContinue(CudaLaunchLocalSquaredNorm(
            view_.residual_r, shape_.interior_count, view_.true_residual,
            shape_.true_trace_count, view_.status, shape_.elements,
            view_.scalar)))
      return false;
    double local_squared = 0.0;
    const bool copied = TransportPolicy::DeviceToHost(
            &local_squared, view_.scalar, sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!CollectiveContinue(copied)) return false;
    double global_squared = 0.0;
    const bool reduced = MPI_Allreduce(
        &local_squared, &global_squared, 1, MPI_DOUBLE, MPI_SUM,
        communicator_) == MPI_SUCCESS;
    *residual_norm = std::sqrt(global_squared);
    return CollectiveContinue(reduced && std::isfinite(local_squared) &&
                              local_squared >= 0.0 &&
                              std::isfinite(*residual_norm));
  };
  const auto condense_current = [&]() noexcept
  {
    std::uint64_t mma_tiles = 0;
    bool condensed = false;
    {
      ::igrlab::NVTXRange profile_region(
          ::igrlab::observability::RegionId::hdg_condensation);
      condensed = CopyBase(workspace) &&
          CudaLaunchCondensation(view_, shape_, cuda_engine_, mma_tiles) &&
          StatusPassed();
    }
    if (!CollectiveContinue(condensed)) return false;
    if (result.tensor_core.condensation_launches ==
            std::numeric_limits<std::uint64_t>::max() ||
        mma_tiles > std::numeric_limits<std::uint64_t>::max() -
            result.tensor_core.mma_tiles)
      return false;
    ++result.tensor_core.condensation_launches;
    result.tensor_core.mma_tiles += mma_tiles;
    ++result.complete_condensations;
    return AssembleTrueSystem();
  };
  const auto record_linear_work = [&](const int iterations,
                                      const int attempts,
                                      const int tightenings) noexcept
  {
    if (iterations < 0 || attempts < 0 || tightenings < 0 ||
        iterations > std::numeric_limits<int>::max() -
            result.linear_iterations ||
        iterations > std::numeric_limits<int>::max() -
            result.final_coherent_linear_iterations ||
        attempts > std::numeric_limits<int>::max() -
            result.linear_solve_attempts ||
        tightenings > std::numeric_limits<int>::max() -
            result.true_residual_tightenings ||
        result.trace_solves == std::numeric_limits<int>::max())
      return false;
    result.linear_iterations += iterations;
    result.final_coherent_linear_iterations += iterations;
    result.linear_solve_attempts += attempts;
    result.true_residual_tightenings += tightenings;
    ++result.trace_solves;
    return true;
  };
  double maximum_true_residual = 0.0;
  const auto solve_and_recover = [&](const std::uint64_t generation,
                                     const bool retained_rhs,
                                     const bool algebraic_projection) noexcept
  {
    int iterations = 0, attempts = 0, tightenings = 0;
    double true_residual = std::numeric_limits<double>::infinity();
    const bool solved = SolveTrueSystem(
        generic::EulerHdgLinearSolveRole::final_coherent,
        CudaNativeTraceActiveSubspace::full, true_residual, generation,
        iterations, attempts, tightenings, result.tensor_core, retained_rhs);
    if (!CollectiveContinue(solved &&
                            record_linear_work(iterations, attempts,
                                               tightenings)))
      return false;
    maximum_true_residual = std::max(maximum_true_residual, true_residual);
    if (!DistributeTraceIncrement()) return false;
    bool recovered = false;
    {
      ::igrlab::NVTXRange profile_region(
          ::igrlab::observability::RegionId::hdg_interior_recovery);
      recovered = algebraic_projection
          ? CudaLaunchRecover(view_, shape_, cuda_engine_)
          : CudaLaunchRecoverFromRetainedLu(view_, shape_);
    }
    if (!CollectiveContinue(recovered && StatusPassed())) return false;
    ++result.complete_recoveries;
    return true;
  };
  constexpr unsigned rodas_admissibility_unknown_failure = 1U << 31U;
  const auto classify_admissibility = [&](const char* const site,
                                           const int stage,
                                           unsigned& global_mask) noexcept
  {
    global_mask = rodas_admissibility_unknown_failure;
    const bool launched = CudaLaunchAdmissibility(view_, shape_);
    const bool local_admissible = launched && StatusPassed();
    if (CollectiveContinue(local_admissible))
    {
      global_mask = 0U;
      return true;
    }

    // A failed attempt has no publishable state.  Copy only its bounded
    // per-element classification so the lifecycle can distinguish a finite
    // Uhat-only DAE inconsistency from a differential or nonfinite failure.
    // The existing export buffer is deliberately reused: this slow path
    // performs no allocation and retains no stage/event history.
    unsigned local_mask = 0U;
    bool local_probe_ready = launched;
    if (!local_admissible && launched)
    {
      local_probe_ready =
          host_export_interior_.size() >=
              static_cast<std::size_t>(shape_.elements) &&
          TransportPolicy::DeviceToHost(
              host_export_interior_.data(), view_.status,
              static_cast<std::size_t>(shape_.elements) * sizeof(double),
              CudaTransferCategory::scalar_diagnostic,
              CudaTransferSite::scalar_status) &&
          TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
      bool classified = false;
      if (local_probe_ready)
      {
        for (int element = 0; element < shape_.elements; ++element)
        {
          const double value = host_export_interior_[
              static_cast<std::size_t>(element)];
          if (value == 1.0) continue;
          classified = true;
          if (std::isfinite(value) && value < 0.0 &&
              -value <= static_cast<double>(
                  std::numeric_limits<unsigned>::max()))
            local_mask |= static_cast<unsigned>(-value);
          else
            local_mask |= rodas_admissibility_unknown_failure;
        }
        if (!classified)
          local_mask |= rodas_admissibility_unknown_failure;
      }
    }
    if (!CollectiveContinue(local_probe_ready)) return false;
    unsigned reduced_mask = 0U;
    const bool reduced = MPI_Allreduce(
        &local_mask, &reduced_mask, 1, MPI_UNSIGNED, MPI_BOR,
        communicator_) == MPI_SUCCESS;
    if (!CollectiveContinue(reduced)) return false;
    global_mask = reduced_mask;
    (void)site;
    (void)stage;
    return (global_mask & rodas_admissibility_unknown_failure) == 0U;
  };
  const auto admissible = [&](const char* const site,
                              const int stage) noexcept
  {
    unsigned mask = 0U;
    return classify_admissibility(site, stage, mask) && mask == 0U;
  };
  const auto limit_rodas_state = [&](const char* const site,
                                     const int stage, bool& active,
                                     double& correction_norm) noexcept
  {
    active = false;
    correction_norm = 0.0;
    const auto& policy = config.numerical_policy.positivity_limiter;
    if (policy.kind == generic::EulerHdgPositivityLimiterKind::none)
      return true;
    const bool seed_diagnostics = result.positivity_limiter_calls == 0;
    if (result.positivity_limiter_calls ==
        std::numeric_limits<std::uint64_t>::max())
      return false;
    if (seed_diagnostics && !SeedCandidateLimiterDiagnostics()) return false;
    ++result.positivity_limiter_calls;
    const bool launched = CudaLaunchZhangShuU4(
        view_, shape_, policy, candidate_limiter_diagnostics_,
        view_.status_flag, view_.scalar);
    int local_classification = 2;
    double local_correction_squared =
        std::numeric_limits<double>::infinity();
    const bool copied = launched && TransportPolicy::DeviceToHost(
            &local_classification, view_.status_flag,
            sizeof(local_classification),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::DeviceToHost(
            &local_correction_squared, view_.scalar,
            sizeof(local_correction_squared),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!CollectiveContinue(copied)) return false;
    int global_classification = 2;
    double global_correction_squared = 0.0;
    const std::uint64_t local_count =
        static_cast<std::uint64_t>(shape_.elements) *
        conserved_components * shape_.scalar_interior_dofs;
    std::uint64_t global_count = 0;
    const bool reduced = MPI_Allreduce(
            &local_classification, &global_classification, 1, MPI_INT,
            MPI_MAX, communicator_) == MPI_SUCCESS &&
        MPI_Allreduce(&local_correction_squared,
                      &global_correction_squared, 1, MPI_DOUBLE, MPI_SUM,
                      communicator_) == MPI_SUCCESS &&
        MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T,
                      MPI_SUM, communicator_) == MPI_SUCCESS;
    const bool classification_valid = reduced &&
        global_classification >= 0 && global_classification <= 3;
    if (classification_valid && global_classification == 3)
      result.positivity_limiter_inadmissible_mean_rejections = 1;
    const bool ready = classification_valid && global_classification <= 1 &&
        global_count > 0 && std::isfinite(local_correction_squared) &&
        local_correction_squared >= 0.0 &&
        std::isfinite(global_correction_squared) &&
        global_correction_squared >= 0.0;
    if (!CollectiveContinue(ready)) return false;
    active = global_classification == 1;
    correction_norm = std::sqrt(
        global_correction_squared / static_cast<double>(global_count));
    if (!CollectiveContinue(std::isfinite(correction_norm))) return false;
    result.maximum_positivity_limiter_correction_norm = std::max(
        result.maximum_positivity_limiter_correction_norm,
        correction_norm);
    if (active)
    {
      if (result.positivity_limiter_active_calls ==
          std::numeric_limits<std::uint64_t>::max())
        return false;
      ++result.positivity_limiter_active_calls;
    }
    return true;
  };
  std::array<double, igr_trace_components>
      admitted_stage_boundary_balance{};
  std::array<double, igr_trace_components>
      admitted_stage_boundary_scale{};
  const auto capture_admitted_stage_conservation =
      [&](const std::size_t stage) noexcept
  {
    if (!positivity_enabled) return true;
    std::array<double, euler_conservation_result_scalars> local{};
    const bool launched = CudaLaunchRodasAdmittedStageConservation(
        view_, shape_, attempt.dt, stage, view_.scalar);
    if (!CollectiveContinue(launched)) return false;
    const bool copied = TransportPolicy::DeviceToHost(
            local.data(), view_.scalar, sizeof(local),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!CollectiveContinue(copied)) return false;
    bool local_valid = copied &&
        local[2 * igr_trace_components + 1] == 1.0 &&
        local[2 * igr_trace_components] == 0.0;
    for (int component = 0;
         component < igr_trace_components && local_valid; ++component)
    {
      const double balance = local[static_cast<std::size_t>(component)];
      const double scale = local[static_cast<std::size_t>(
          igr_trace_components + component)];
      local_valid = std::isfinite(balance) && std::isfinite(scale) &&
          scale >= 0.0;
      if (!local_valid) break;
      admitted_stage_boundary_balance[component] += balance;
      admitted_stage_boundary_scale[component] += scale;
      local_valid = std::isfinite(
              admitted_stage_boundary_balance[component]) &&
          std::isfinite(admitted_stage_boundary_scale[component]) &&
          admitted_stage_boundary_scale[component] >= 0.0;
    }
    return CollectiveContinue(local_valid);
  };
  const std::size_t interior_bytes = workspace.interior.count *
      sizeof(double);
  const std::size_t true_trace_bytes = workspace.trace.count *
      sizeof(double);
  const auto snapshot_algebraic_reference = [&]() noexcept
  {
    return TransportPolicy::DeviceToDevice(
            view_.rodas_combination_interior,
            workspace.interior.coefficients, interior_bytes,
            CudaTransferCategory::attempt_history,
            CudaTransferSite::nonlinear_base_interior) &&
        TransportPolicy::DeviceToDevice(
            view_.rodas_combination_true_trace,
            workspace.trace.coefficients, true_trace_bytes,
            CudaTransferCategory::attempt_history,
            CudaTransferSite::nonlinear_base_trace) &&
        TransportPolicy::Synchronize(
            CudaTransferSite::nonlinear_base_trace);
  };
  const auto project_algebraic = [&](double& initial_norm,
                                     double& final_norm) noexcept
  {
    initial_norm = std::numeric_limits<double>::infinity();
    final_norm = std::numeric_limits<double>::infinity();
    constexpr int maximum_updates = 6;
    for (int update = 0; update <= maximum_updates; ++update)
    {
      if (!enter(CudaFailurePhase::assembly) ||
          !assemble_spatial(true, &final_norm))
        return false;
      if (update == 0) initial_norm = final_norm;
      // Algebraic consistency is an absolute acceptance contract.  Scaling
      // the stopping threshold by the entry inconsistency could otherwise
      // admit a trace/IGR state that merely made a large relative reduction
      // while remaining outside the qualified index-one constraint surface.
      if (final_norm <= algebraic_residual_allowance)
        return true;
      if (update == maximum_updates) return false;
      if (!enter(CudaFailurePhase::condensation) || !condense_current())
        return false;
      const std::uint64_t generation = ++linear_generation_;
      if (!enter(CudaFailurePhase::trace_solve)) return false;
      if (!enter(CudaFailurePhase::recovery) ||
          !solve_and_recover(generation, false, true) ||
          !CollectiveContinue(
              CudaLaunchRodasFreezeDifferentialIncrement(view_, shape_) &&
              CudaLaunchTrial(view_, shape_, 1.0) &&
              CudaLaunchRodasRestoreDifferentialState(view_, shape_) &&
              DistributeCurrentTrace(workspace.trace.coefficients)))
        return false;
      ++result.nonlinear_updates;
    }
    return false;
  };

  bool entry_limited = false;
  double entry_limiter_correction = 0.0;
  if (!limit_rodas_state(
          "entry", -1, entry_limited, entry_limiter_correction))
  {
    result.phase = CudaFailurePhase::assembly;
    result.reason = FailureReason::admissibility_gate;
    return result;
  }
  (void)entry_limiter_correction;
  const bool entry_snapshot = TransportPolicy::DeviceToDevice(
          view_.rodas_combination_interior,
          workspace.interior.coefficients, interior_bytes,
          CudaTransferCategory::attempt_history,
          CudaTransferSite::nonlinear_base_interior) &&
      TransportPolicy::DeviceToDevice(
          view_.rodas_combination_true_trace,
          workspace.trace.coefficients, true_trace_bytes,
          CudaTransferCategory::attempt_history,
          CudaTransferSite::nonlinear_base_trace) &&
      TransportPolicy::Synchronize(
          CudaTransferSite::nonlinear_base_trace);
  if (!CollectiveContinue(entry_snapshot))
  {
    result.reason = FailureReason::backend_failure;
    return result;
  }

  double initial_algebraic_norm = 0.0;
  double projected_algebraic_norm = 0.0;
  if (!project_algebraic(initial_algebraic_norm,
                         projected_algebraic_norm))
  {
    if (result.reason == FailureReason::none)
      result.reason = FailureReason::recovery_gate;
    return result;
  }
  if (entry_limited)
    ++result.positivity_algebraic_projections;
  result.initial_residual = initial_algebraic_norm;
  const bool entry_correction_launched =
      CudaLaunchRodasNormalizedEntryCorrectionSquared(
          view_, shape_, view_.scalar);
  double local_entry_correction_squared = 0.0;
  const bool entry_correction_copied = entry_correction_launched &&
      TransportPolicy::DeviceToHost(
          &local_entry_correction_squared, view_.scalar, sizeof(double),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) &&
      TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
  if (!CollectiveContinue(entry_correction_copied))
  {
    result.phase = CudaFailurePhase::recovery;
    result.reason = FailureReason::recovery_gate;
    return result;
  }
  double global_entry_correction_squared = 0.0;
  const std::uint64_t local_entry_algebraic_count =
      static_cast<std::uint64_t>(shape_.elements) *
          (shape_.interior_components - conserved_components) *
          shape_.scalar_interior_dofs +
      static_cast<std::uint64_t>(shape_.true_trace_count);
  std::uint64_t global_entry_algebraic_count = 0;
  const bool entry_correction_reduced = MPI_Allreduce(
          &local_entry_correction_squared,
          &global_entry_correction_squared, 1, MPI_DOUBLE,
          MPI_SUM, communicator_) == MPI_SUCCESS &&
      MPI_Allreduce(&local_entry_algebraic_count,
                    &global_entry_algebraic_count, 1, MPI_UINT64_T,
                    MPI_SUM, communicator_) == MPI_SUCCESS;
  const double normalized_entry_correction =
      global_entry_algebraic_count > 0
      ? std::sqrt(global_entry_correction_squared /
                  static_cast<double>(global_entry_algebraic_count))
      : std::numeric_limits<double>::infinity();
  constexpr double maximum_entry_algebraic_correction_norm = 0.25;
  if (!CollectiveContinue(
          entry_correction_reduced && global_entry_algebraic_count > 0 &&
          std::isfinite(local_entry_correction_squared) &&
          local_entry_correction_squared >= 0.0 &&
          std::isfinite(normalized_entry_correction) &&
          normalized_entry_correction <=
              maximum_entry_algebraic_correction_norm))
  {
    result.phase = CudaFailurePhase::recovery;
    result.reason = FailureReason::recovery_gate;
    return result;
  }
  if (!CollectiveContinue(CopyBase(workspace)) ||
      !admissible("entry_projection", -1))
  {
    result.reason = FailureReason::admissibility_gate;
    return result;
  }

  if (!enter(CudaFailurePhase::assembly) ||
      !assemble_spatial(false, nullptr))
  {
    result.reason = FailureReason::finite_gate;
    return result;
  }
  const std::size_t element_trace_bytes =
      static_cast<std::size_t>(shape_.elements) * shape_.trace_dofs *
      sizeof(double);
  const std::size_t coupling_bytes =
      static_cast<std::size_t>(shape_.elements) * shape_.interior_dofs *
      shape_.trace_dofs * sizeof(double);
  const bool common_saved = TransportPolicy::DeviceToDevice(
          view_.rodas_residual_f, view_.f, interior_bytes,
          CudaTransferCategory::attempt_history,
          CudaTransferSite::nonlinear_base_interior) &&
      TransportPolicy::DeviceToDevice(
          view_.rodas_residual_g, view_.g, element_trace_bytes,
          CudaTransferCategory::attempt_history,
          CudaTransferSite::nonlinear_base_trace) &&
      TransportPolicy::DeviceToDevice(
          view_.rodas_operator_b, view_.b, coupling_bytes,
          CudaTransferCategory::attempt_history,
          CudaTransferSite::nonlinear_base_interior) &&
      TransportPolicy::DeviceToDevice(
          view_.rodas_operator_c, view_.c, coupling_bytes,
          CudaTransferCategory::attempt_history,
          CudaTransferSite::nonlinear_base_interior) &&
      TransportPolicy::Synchronize(
          CudaTransferSite::nonlinear_base_interior);
  if (!CollectiveContinue(common_saved))
  {
    result.reason = FailureReason::backend_failure;
    return result;
  }

  std::uint64_t common_generation = 0;
  bool final_stage_reconstructed = false;
  for (std::size_t stage = 0; stage < Coefficients::stage_count; ++stage)
  {
    bool stage_limited = false;
    double stage_limiter_correction = 0.0;
    if (!CollectiveContinue(CudaLaunchRodasLoadStage(view_, shape_, stage) &&
                            DistributeCurrentTrace(
                                workspace.trace.coefficients)) ||
        !limit_rodas_state("stage", static_cast<int>(stage),
                           stage_limited, stage_limiter_correction))
    {
      result.phase = CudaFailurePhase::assembly;
      result.reason = FailureReason::admissibility_gate;
      return result;
    }
    (void)stage_limiter_correction;
    unsigned stage_admissibility_mask = 0U;
    if (!classify_admissibility(
            "stage", static_cast<int>(stage),
            stage_admissibility_mask))
    {
      result.phase = CudaFailurePhase::assembly;
      result.reason = FailureReason::backend_failure;
      return result;
    }

    const bool stage_reconstructed = stage_limited;
    if (stage_admissibility_mask != 0U && !stage_reconstructed)
    {
      result.phase = CudaFailurePhase::assembly;
      result.reason = FailureReason::admissibility_gate;
      return result;
    }
    if (stage + 1 == Coefficients::stage_count)
      final_stage_reconstructed = stage_reconstructed;
    if (stage_reconstructed)
    {
      // Stages zero and one are the common-shift base surfaces.  Only an
      // explicitly selected, active U4 positivity map may replace a later
      // causal stage and trigger exact fixed-U4 algebraic reconstruction.
      if (!CollectiveContinue(stage >= 2 && common_generation != 0))
      {
        result.phase = CudaFailurePhase::assembly;
        result.reason = FailureReason::admissibility_gate;
        return result;
      }

      // The algebraic projector owns base_*, local factors, the assembled
      // skeleton operator, and solver setup state. Preserve the physical step
      // base in disposable RODAS scratch before seeding the projection from
      // that last coherent algebraic state.
      const bool base_saved = TransportPolicy::DeviceToDevice(
              view_.rodas_residual_f, view_.base_interior, interior_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              view_.rodas_combination_true_trace, view_.base_true_trace,
              true_trace_bytes, CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_trace) &&
          CudaLaunchRodasSeedAlgebraicProjection(view_, shape_) &&
          TransportPolicy::Synchronize(
              CudaTransferSite::nonlinear_base_trace);
      if (!CollectiveContinue(base_saved) ||
          !DistributeCurrentTrace(workspace.trace.coefficients))
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }

      double stage_projection_initial = 0.0;
      double stage_projection_final = 0.0;
      if (!project_algebraic(stage_projection_initial,
                             stage_projection_final) ||
          !CollectiveContinue(
              stage_projection_final <= algebraic_residual_allowance))
      {
        result.reason = FailureReason::recovery_gate;
        return result;
      }
      ++result.positivity_algebraic_projections;

      // Keep the coherent filtered stage while restoring the exact step base
      // and rebuilding the one common-shift operator that the projector has
      // overwritten. output_interior/trial_true_trace are otherwise dormant
      // during Prepare and require no new persistent allocation.
      const bool stage_saved_and_base_restored =
          TransportPolicy::DeviceToDevice(
              view_.output_interior, workspace.interior.coefficients,
              interior_bytes, CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              view_.trial_true_trace, workspace.trace.coefficients,
              true_trace_bytes, CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_trace) &&
          TransportPolicy::DeviceToDevice(
              view_.base_interior, view_.rodas_residual_f, interior_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              workspace.interior.coefficients, view_.rodas_residual_f,
              interior_bytes, CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              view_.base_true_trace, view_.rodas_combination_true_trace,
              true_trace_bytes, CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_trace) &&
          TransportPolicy::DeviceToDevice(
              workspace.trace.coefficients,
              view_.rodas_combination_true_trace, true_trace_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_trace) &&
          TransportPolicy::Synchronize(
              CudaTransferSite::nonlinear_base_trace);
      if (!CollectiveContinue(stage_saved_and_base_restored) ||
          !DistributeCurrentTrace(workspace.trace.coefficients))
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
      const bool local_base_trace_restored =
          TransportPolicy::DeviceToDevice(
              view_.base_local_trace, view_.current_local_trace,
              element_trace_bytes, CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_local_trace);
      if (!CollectiveContinue(local_base_trace_restored) ||
          !enter(CudaFailurePhase::assembly) ||
          !assemble_spatial(false, nullptr))
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
      const bool common_resaved = TransportPolicy::DeviceToDevice(
              view_.rodas_residual_f, view_.f, interior_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              view_.rodas_residual_g, view_.g, element_trace_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_trace) &&
          TransportPolicy::DeviceToDevice(
              view_.rodas_operator_b, view_.b, coupling_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              view_.rodas_operator_c, view_.c, coupling_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          CudaLaunchRodasBuildStageRightHandSide(
              view_, shape_, 0, attempt.dt) && StatusPassed();
      if (!CollectiveContinue(common_resaved) ||
          !enter(CudaFailurePhase::condensation))
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
      std::uint64_t rebuild_tiles = 0;
      bool rebuilt = false;
      {
        ::igrlab::NVTXRange profile_region(
            ::igrlab::observability::RegionId::hdg_condensation);
        rebuilt = CudaLaunchCondensation(
            view_, shape_, cuda_engine_, rebuild_tiles) && StatusPassed();
      }
      if (!CollectiveContinue(rebuilt) ||
          result.tensor_core.condensation_launches ==
              std::numeric_limits<std::uint64_t>::max() ||
          rebuild_tiles > std::numeric_limits<std::uint64_t>::max() -
              result.tensor_core.mma_tiles ||
          !AssembleTrueSystem())
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
      ++result.tensor_core.condensation_launches;
      result.tensor_core.mma_tiles += rebuild_tiles;
      ++result.complete_condensations;
      ++result.common_shift_operator_builds;
      ++result.common_shift_operator_rebuilds;
      common_generation = ++linear_generation_;

      const bool stage_restored = TransportPolicy::DeviceToDevice(
              workspace.interior.coefficients, view_.output_interior,
              interior_bytes, CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              workspace.trace.coefficients, view_.trial_true_trace,
              true_trace_bytes, CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_trace) &&
          TransportPolicy::Synchronize(
              CudaTransferSite::nonlinear_base_trace);
      if (!CollectiveContinue(stage_restored) ||
          !DistributeCurrentTrace(workspace.trace.coefficients))
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
    }
    if (stage_reconstructed &&
        !admissible("stage_projection", static_cast<int>(stage)))
    {
      result.phase = CudaFailurePhase::assembly;
      result.reason = FailureReason::admissibility_gate;
      return result;
    }
    if (positivity_enabled && stage + 1 == Coefficients::stage_count)
    {
      // RODAS3's embedded endpoint is exactly Y_4.  Capture only its admitted
      // U4 lanes after the stage limiter/reconstruction and before the
      // principal candidate overwrites work_interior.  output_interior is
      // lifecycle-dead from the end of stage reconstruction until Execute
      // returns; the captured lanes are moved to protected RODAS storage
      // before a prepared candidate can be exported.
      if (!CollectiveContinue(CudaLaunchRodasCopyDifferentialState(
              shape_, workspace.interior.coefficients,
              view_.output_interior)))
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
    }
    if (stage >= 2)
    {
      if (!enter(CudaFailurePhase::assembly) ||
          !assemble_spatial(false, nullptr))
      {
        result.reason = FailureReason::finite_gate;
        return result;
      }
      const bool residual_and_operator_ready =
          TransportPolicy::DeviceToDevice(
              view_.rodas_residual_f, view_.f, interior_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              view_.rodas_residual_g, view_.g, element_trace_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_trace) &&
          TransportPolicy::DeviceToDevice(
              view_.b, view_.rodas_operator_b, coupling_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior) &&
          TransportPolicy::DeviceToDevice(
              view_.c, view_.rodas_operator_c, coupling_bytes,
              CudaTransferCategory::attempt_history,
              CudaTransferSite::nonlinear_base_interior);
      if (!CollectiveContinue(residual_and_operator_ready))
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
    }
    if (!CollectiveContinue(CudaLaunchRodasBuildStageRightHandSide(
            view_, shape_, stage, attempt.dt) && StatusPassed()))
    {
      result.reason = FailureReason::finite_gate;
      return result;
    }
    if (!enter(CudaFailurePhase::condensation)) return result;
    if (stage == 0)
    {
      std::uint64_t mma_tiles = 0;
      bool condensed = false;
      {
        ::igrlab::NVTXRange profile_region(
            ::igrlab::observability::RegionId::hdg_condensation);
        condensed = CudaLaunchCondensation(
            view_, shape_, cuda_engine_, mma_tiles) && StatusPassed();
      }
      if (!CollectiveContinue(condensed) ||
          result.tensor_core.condensation_launches ==
              std::numeric_limits<std::uint64_t>::max() ||
          mma_tiles > std::numeric_limits<std::uint64_t>::max() -
              result.tensor_core.mma_tiles ||
          !AssembleTrueSystem())
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
      ++result.tensor_core.condensation_launches;
      result.tensor_core.mma_tiles += mma_tiles;
      ++result.complete_condensations;
      ++result.common_shift_operator_builds;
      common_generation = ++linear_generation_;
    }
    else
    {
      const bool refreshed = CudaLaunchCondenseRetainedLuRhs(
              view_, shape_) && StatusPassed() &&
          RefreshRetainedTrueRhs();
      if (!CollectiveContinue(refreshed))
      {
        result.reason = FailureReason::backend_failure;
        return result;
      }
      ++result.common_shift_operator_reuses;
      ++result.condensed_rhs_refreshes;
    }
    const std::uint64_t setups_before = tensor_rank_one
        ? result.tensor_core.element_schwarz_patch_setups
        : ilu_common_shift ? result.tensor_core.ilu_factor_setups : 0;
    const std::uint64_t reuses_before =
        ilu_common_shift ? result.tensor_core.ilu_factor_reuses : 0;
    // A stage algebraic reconstruction invalidates and rebuilds the
    // common-shift operator.  Its first subsequent solve is therefore a new
    // operator transition even when this is not tableau stage zero.  Calling
    // it a retained-RHS transition would (correctly) fail the generation
    // contract before the new Schwarz/ILU setup can be constructed.
    const bool same_operator_new_rhs = stage != 0 && !stage_reconstructed;
    if (!enter(CudaFailurePhase::trace_solve) ||
        !enter(CudaFailurePhase::recovery) ||
        !solve_and_recover(common_generation, same_operator_new_rhs, false) ||
        !CollectiveContinue(CudaLaunchRodasCaptureStageIncrement(
                                view_, shape_, stage) &&
                            StatusPassed() &&
                            DistributeRodasStageIncrement(stage)))
    {
      if (result.reason == FailureReason::none)
        result.reason = FailureReason::recovery_gate;
      return result;
    }
    if (!capture_admitted_stage_conservation(stage))
    {
      result.reason = FailureReason::backend_failure;
      return result;
    }
    const std::uint64_t setups_after = tensor_rank_one
        ? result.tensor_core.element_schwarz_patch_setups
        : ilu_common_shift ? result.tensor_core.ilu_factor_setups : 0;
    const std::uint64_t reuses_after =
        ilu_common_shift ? result.tensor_core.ilu_factor_reuses : 0;
    if (!CollectiveContinue(setups_after >= setups_before &&
                            reuses_after >= reuses_before))
    {
      result.reason = FailureReason::telemetry_gate;
      return result;
    }
    const std::uint64_t setup_delta = setups_after - setups_before;
    const std::uint64_t reuse_delta = reuses_after - reuses_before;
    if (tensor_rank_one)
    {
      if (stage == 0 || stage_reconstructed)
      {
        if (!CollectiveContinue(setup_delta > 0 && reuse_delta == 0))
        {
          result.reason = FailureReason::telemetry_gate;
          return result;
        }
        result.common_shift_preconditioner_setups += setup_delta;
      }
      else
      {
        if (!CollectiveContinue(setup_delta == 0 && reuse_delta == 0))
        {
          result.reason = FailureReason::telemetry_gate;
          return result;
        }
        ++result.common_shift_preconditioner_reuses;
      }
    }
    else if (ilu_common_shift)
    {
      // The selected ILU lifetime remains authoritative: ordinary CUDA
      // refreshes every causal solve, whereas tensor IGR retains the first
      // factor for the complete attempt (including the algebraic projection
      // that precedes these stages).  Keep the common-shift census logical:
      // stage zero owns its one setup and stages one through three reuse it,
      // while tensor_core retains the exact physical setup/reuse totals.
      if (!CollectiveContinue(setup_delta + reuse_delta == 1))
      {
        result.reason = FailureReason::telemetry_gate;
        return result;
      }
      if (trace_policy_descriptor_.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt)
      {
        if (stage == 0)
          ++result.common_shift_preconditioner_setups;
        else
          ++result.common_shift_preconditioner_reuses;
      }
      else
      {
        result.common_shift_preconditioner_setups += setup_delta;
        result.common_shift_preconditioner_reuses += reuse_delta;
      }
    }
    else
    {
      // Conventional tensor rank two uses its established HYPRE diagonal
      // scale. CudaRefreshDiagonalAndSolve rebuilds that scale once per solve;
      // it intentionally has no ILU counter, so record the physical setup here.
      if (!CollectiveContinue(host_staged_rank_two &&
                              diagonal_common_shift &&
                              setup_delta == 0 && reuse_delta == 0))
      {
        result.reason = FailureReason::telemetry_gate;
        return result;
      }
      ++result.common_shift_preconditioner_setups;
    }
    ++result.common_shift_trace_solves;
  }

  const auto preserve_endpoint_base = [&]() noexcept
  {
    // The projector reuses base_* for its Newton snapshots.  Preserve both
    // forms of the coherent step-entry state that remain live after the
    // causal stages: U4 for estimation/conservation and the element-local
    // trace for any subsequent reconstruction work.
    return TransportPolicy::DeviceToDevice(
            view_.rodas_residual_f, view_.base_interior, interior_bytes,
            CudaTransferCategory::attempt_history,
            CudaTransferSite::nonlinear_base_interior) &&
        TransportPolicy::DeviceToDevice(
            view_.rodas_residual_g, view_.base_local_trace,
            element_trace_bytes,
            CudaTransferCategory::attempt_history,
            CudaTransferSite::nonlinear_base_local_trace) &&
        TransportPolicy::Synchronize(
            CudaTransferSite::nonlinear_base_local_trace);
  };

  if (!CollectiveContinue(CudaLaunchRodasFormCandidate(
                              view_, shape_, final_stage_reconstructed) &&
                          DistributeCurrentTrace(
                              workspace.trace.coefficients)))
  {
    result.reason = FailureReason::admissibility_gate;
    return result;
  }
  bool endpoint_limited = false;
  bool endpoint_projected = false;
  double endpoint_limiter_correction = 0.0;
  double final_projection_initial = 0.0;
  if (!limit_rodas_state("endpoint", -1, endpoint_limited,
                         endpoint_limiter_correction))
  {
    result.reason = FailureReason::admissibility_gate;
    return result;
  }
  (void)endpoint_limiter_correction;
  if (!CollectiveContinue(preserve_endpoint_base()))
  {
    result.reason = FailureReason::backend_failure;
    return result;
  }
  // The endpoint predictor is the independent algebraic-correction
  // reference, including after a U4-only positivity map.  Snapshot it before
  // any fixed-U4 Newton update so the tangent lift cannot evade the temporal
  // error allowance.
  if (!CollectiveContinue(snapshot_algebraic_reference()))
  {
    result.reason = FailureReason::backend_failure;
    return result;
  }
  if (endpoint_limited)
  {
    // A Zhang--Shu map owns U4 only.  The affine endpoint's auxiliary fields
    // and trace are merely a predictor after U4 is filtered.  Keep those
    // algebraic values as the Newton initial guess and independently bound
    // the complete reconstruction needed to reach exact nonlinear closure.
    if (!project_algebraic(final_projection_initial,
                           result.final_algebraic_residual))
    {
      result.reason = FailureReason::recovery_gate;
      return result;
    }
    ++result.positivity_algebraic_projections;
    endpoint_projected = true;

    // project_algebraic uses base_* as private iteration storage. Restore
    // only the differential entry lanes needed by the filtered-pair scale;
    // the protected preprojection reference remains in rodas_combination_*.
    if (!CollectiveContinue(CudaLaunchRodasCopyDifferentialState(
            shape_, view_.rodas_residual_f, view_.base_interior)))
    {
      result.reason = FailureReason::backend_failure;
      return result;
    }
  }
  if (!admissible("endpoint", -1))
  {
    result.reason = FailureReason::admissibility_gate;
    return result;
  }
  if (positivity_enabled)
  {
    const bool launched = CudaLaunchRodasFilteredPairErrorSquared(
        view_, shape_, view_.scalar, view_.scalar + 1);
    std::array<double, 2> local{{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()}};
    const bool error_copied = launched && TransportPolicy::DeviceToHost(
            local.data(), view_.scalar, sizeof(local),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    // Converge rank-local device/transport success before any rank enters the
    // estimator reductions; otherwise a one-rank copy failure can deadlock.
    if (!CollectiveContinue(error_copied))
    {
      result.reason = FailureReason::finite_gate;
      return result;
    }
    std::array<double, 2> global{};
    const bool error_reduced = MPI_Allreduce(
        local.data(), global.data(), static_cast<int>(global.size()),
        MPI_DOUBLE, MPI_SUM, communicator_) == MPI_SUCCESS;
    const bool error_ready = error_reduced && std::isfinite(local[0]) &&
        local[0] >= 0.0 && std::isfinite(local[1]) && local[1] > 0.0 &&
        std::isfinite(global[0]) && global[0] >= 0.0 &&
        std::isfinite(global[1]) && global[1] > 0.0;
    result.temporal_error_norm = error_ready
        ? std::sqrt(global[0] /
              (static_cast<double>(conserved_components) * global[1]))
        : std::numeric_limits<double>::infinity();
    if (!CollectiveContinue(error_ready &&
                            std::isfinite(result.temporal_error_norm)))
    {
      result.reason = FailureReason::finite_gate;
      return result;
    }
  }
  else
  {
    // Preserve the existing limiter-off launch, transfer, and two-reduction
    // estimator path exactly.  Only collective convergence is inserted
    // before MPI so a local transport failure cannot strand another rank.
    const bool launched = CudaLaunchRodasTemporalErrorSquared(
        view_, shape_, view_.scalar);
    double local_error_squared = 0.0;
    const bool error_copied = launched && TransportPolicy::DeviceToHost(
            &local_error_squared, view_.scalar, sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!CollectiveContinue(error_copied))
    {
      result.reason = FailureReason::finite_gate;
      return result;
    }
    double global_error_squared = 0.0;
    const std::uint64_t local_differential_count =
        static_cast<std::uint64_t>(shape_.elements) *
        conserved_components * shape_.scalar_interior_dofs;
    std::uint64_t global_differential_count = 0;
    const bool error_reduced = MPI_Allreduce(
            &local_error_squared, &global_error_squared, 1,
            MPI_DOUBLE, MPI_SUM, communicator_) == MPI_SUCCESS &&
        MPI_Allreduce(&local_differential_count,
                      &global_differential_count, 1, MPI_UINT64_T,
                      MPI_SUM, communicator_) == MPI_SUCCESS;
    result.temporal_error_norm = global_differential_count > 0
        ? std::sqrt(global_error_squared /
                    static_cast<double>(global_differential_count))
        : std::numeric_limits<double>::infinity();
    if (!CollectiveContinue(error_reduced &&
                            global_differential_count > 0 &&
                            std::isfinite(result.temporal_error_norm)))
    {
      result.reason = FailureReason::finite_gate;
      return result;
    }
  }

  // Preserve the original limiter-inactive ordering and operation census.
  // The active endpoint has already taken this exact snapshot and completed
  // its projection before the filtered embedded estimate was formed.
  if (!endpoint_projected)
  {
    if (!project_algebraic(final_projection_initial,
                           result.final_algebraic_residual))
    {
      result.reason = FailureReason::recovery_gate;
      return result;
    }
  }
  if (!CollectiveContinue(CudaLaunchRodasAlgebraicCorrectionSquared(
          view_, shape_, view_.scalar)))
  {
    result.reason = FailureReason::recovery_gate;
    return result;
  }
  double local_correction_squared = 0.0;
  const bool correction_copied = TransportPolicy::DeviceToHost(
          &local_correction_squared, view_.scalar, sizeof(double),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) &&
      TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
  if (!CollectiveContinue(correction_copied))
  {
    result.reason = FailureReason::recovery_gate;
    return result;
  }
  double global_correction_squared = 0.0;
  std::uint64_t local_algebraic_count =
      static_cast<std::uint64_t>(shape_.elements) *
          (shape_.interior_components - conserved_components) *
          shape_.scalar_interior_dofs +
      static_cast<std::uint64_t>(shape_.true_trace_count);
  std::uint64_t global_algebraic_count = 0;
  const bool correction_reduced = MPI_Allreduce(
          &local_correction_squared, &global_correction_squared, 1,
          MPI_DOUBLE, MPI_SUM, communicator_) == MPI_SUCCESS &&
      MPI_Allreduce(&local_algebraic_count, &global_algebraic_count, 1,
                    MPI_UINT64_T, MPI_SUM, communicator_) == MPI_SUCCESS;
  result.final_algebraic_correction_norm = global_algebraic_count > 0
      ? std::sqrt(global_correction_squared /
                  static_cast<double>(global_algebraic_count))
      : std::numeric_limits<double>::infinity();
  if (!CollectiveContinue(correction_reduced && global_algebraic_count > 0 &&
                          std::isfinite(
                              result.final_algebraic_correction_norm)) ||
      !admissible("final_projection", -1))
  {
    result.reason = FailureReason::recovery_gate;
    return result;
  }

  if (!enter(CudaFailurePhase::after_recovery_before_commit)) return result;
  result.phase = CudaFailurePhase::diagnostics;
  if (inject_at(CudaFailurePhase::diagnostics)) return result;
  const bool conservation_ready = RodasConservationBalance(
      attempt.dt, result.maximum_conservation_error,
      result.maximum_conservation_relative_error,
      positivity_enabled ? admitted_stage_boundary_balance.data() : nullptr,
      positivity_enabled ? admitted_stage_boundary_scale.data() : nullptr);
  if (positivity_enabled &&
      !CollectiveContinue(CudaLaunchRodasCopyDifferentialState(
          shape_, view_.output_interior,
          view_.rodas_combination_interior)))
  {
    result.reason = FailureReason::backend_failure;
    return result;
  }
  // The conservation witness has now consumed the principal preprojection
  // U4 snapshot.  Its leading lanes are dead, so they protect the admitted
  // embedded U4 through Prepare -> Assess even if an intervening output
  // export repacks output_interior.  Auxiliary lanes remain untouched for the
  // prepared algebraic-correction gate.
  result.conservation = CollectiveContinue(
      conservation_ready &&
      result.maximum_conservation_relative_error <=
          config.numerical_policy.conservation_relative_tolerance);
  result.finite = std::isfinite(result.temporal_error_norm) &&
      std::isfinite(result.final_algebraic_residual) &&
      std::isfinite(result.final_algebraic_correction_norm);
  result.admissible = true;
  result.recovery = result.finite &&
      result.final_algebraic_residual <= algebraic_residual_allowance;
  result.true_relative_residual = maximum_true_residual;
  result.true_residual = std::isfinite(maximum_true_residual) &&
      maximum_true_residual <= true_residual_relative_tolerance_;
  if (config.force_private_scientific_gate_failure)
    result.true_residual = false;
  result.final_residual = result.final_algebraic_residual;
  result.candidate_ready = result.finite && result.admissible &&
      result.recovery;
  const bool retained_ilu_census = ilu_common_shift &&
      trace_policy_descriptor_.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
      result.common_shift_preconditioner_setups == 1 &&
      result.common_shift_preconditioner_reuses == 3;
  const bool refresh_preconditioner_census =
      ((ilu_common_shift &&
        trace_policy_descriptor_.ilu_factor_reuse_policy ==
            CudaIluFactorReusePolicy::refresh_each_trace_solve) ||
       diagonal_common_shift) &&
      result.common_shift_preconditioner_setups == 4 &&
      result.common_shift_preconditioner_reuses == 0;
  const bool preconditioner_census = tensor_rank_one
      ? result.common_shift_preconditioner_setups >=
              1 + result.common_shift_operator_rebuilds &&
          result.common_shift_preconditioner_reuses +
                  result.common_shift_operator_rebuilds ==
              Coefficients::stage_count - 1
      : (ordinary_rank_one || host_staged_rank_two) &&
          (retained_ilu_census || refresh_preconditioner_census);
  const bool architecture_census =
      result.common_shift_operator_builds ==
          1 + result.common_shift_operator_rebuilds &&
      result.common_shift_operator_reuses == 3 &&
      result.condensed_rhs_refreshes == 3 &&
      result.common_shift_trace_solves == 4 &&
      preconditioner_census;
  const std::uint64_t current_allocation_events =
      AllocatorPolicy::AllocationEvents();
  bool allocation_plateau_exact =
      current_allocation_events == accepted_allocation_events_;
  if (!allocation_plateau_sampled_)
  {
    // Rank-two HYPRE may materialize two opaque pool high-water transitions
    // during the first complete RODAS3 attempt (algebraic projection plus
    // common-shift work). Freeze that bounded first-use footprint; every
    // later attempt remains exact at the accepted count.
    const std::uint64_t first_attempt_high_water_allowance =
        host_staged_rank_two ? 2U : 1U;
    allocation_plateau_exact =
        current_allocation_events == accepted_allocation_events_ ||
        (accepted_allocation_events_ <=
             std::numeric_limits<std::uint64_t>::max() -
                 first_attempt_high_water_allowance &&
         current_allocation_events > accepted_allocation_events_ &&
         current_allocation_events <= accepted_allocation_events_ +
             first_attempt_high_water_allowance);
    if (allocation_plateau_exact)
      accepted_allocation_events_ = current_allocation_events;
    allocation_plateau_sampled_ = true;
  }
  const bool solver_calls_exact = tensor_rank_one
      ? result.tensor_core.native_rank1_trace_solve_calls >=
            static_cast<std::uint64_t>(result.linear_solve_attempts) &&
          result.tensor_core.hypre_trace_solve_calls == 0
      : (ordinary_rank_one || host_staged_rank_two) &&
          result.tensor_core.hypre_trace_solve_calls >=
              static_cast<std::uint64_t>(result.linear_solve_attempts) &&
          result.tensor_core.native_rank1_trace_solve_calls == 0;
  result.tensor_core.trace_solver_accounting_complete =
      result.tensor_core.trace_solver_accounting_complete &&
      result.linear_solve_attempts ==
          result.trace_solves + result.true_residual_tightenings &&
      solver_calls_exact;
  result.telemetry = architecture_census && DeviceResident() && NoFallback() &&
      result.tensor_core.trace_solver_accounting_complete &&
      AllocatorPolicy::W8OwnedAllocationCalls() ==
          baseline_w8_owned_allocation_calls_ &&
      AllocatorPolicy::DeallocationEvents() ==
          baseline_deallocation_events_ && allocation_plateau_exact;
  result.success = result.candidate_ready && result.true_residual &&
      result.conservation && result.telemetry;
  result.reason = result.success ? FailureReason::none :
      (!result.true_residual ? FailureReason::true_residual_gate :
       !result.conservation ? FailureReason::success_gate :
       !result.telemetry ? FailureReason::telemetry_gate :
                           FailureReason::success_gate);
  workspace.interior.generation = current_generation_;
  workspace.trace.generation = current_generation_;
  const std::uint64_t product_generation = ++linear_generation_;
  result.products = {product_generation, product_generation,
                     product_generation, product_generation, 0};
  if (!UpdateStateHashes(workspace))
  {
    result.candidate_ready = false;
    result.success = false;
    result.reason = FailureReason::backend_failure;
  }
  return result;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssemblePartitionedScalarClosure(
    state_slot&, const double dt, double& norm,
    CudaTensorCoreEvidence& tensor_core_evidence) noexcept
{
  TransportPolicy::SetPhase(CudaFailurePhase::assembly, current_generation_);
  std::uint64_t invocation_volume_mma_tiles = 0;
  bool launched = false;
  {
    ::igrlab::NVTXRange profile_region(
        ::igrlab::observability::RegionId::hdg_local_assembly);
    launched = CudaLaunchAssembly(
        view_, shape_, dt, cuda_engine_, invocation_volume_mma_tiles);
  }
  if (!CollectiveContinue(launched && StatusPassed())) return false;
  std::uint64_t volume_launches = tensor_core_evidence.volume_assembly_launches;
  std::uint64_t volume_tiles = tensor_core_evidence.volume_mma_tiles;
  const bool tensor_selected =
      cuda_engine_ == generic::CudaEngineKind::tensor_core;
  const bool accounted =
      (cuda_engine_ == generic::CudaEngineKind::ordinary || tensor_selected) &&
      CudaAccumulateResearchVolumeWork(
          tensor_selected, shape_.borrowed_host_inputs,
          invocation_volume_mma_tiles,
          volume_launches, volume_tiles);
  if (!CollectiveContinue(accounted)) return false;
  tensor_core_evidence.volume_assembly_launches = volume_launches;
  tensor_core_evidence.volume_mma_tiles = volume_tiles;
  if (!CollectiveContinue(
          CudaLaunchPartitionedScalarClosureMask(view_, shape_) &&
          StatusPassed()) ||
      !AssembleTrueVector(view_.residual_g, view_.true_residual, false))
    return false;
  const bool norm_launched = CudaLaunchLocalSquaredNorm(
      view_.residual_r, shape_.interior_count, view_.true_residual,
      shape_.true_trace_count, view_.status, shape_.elements, view_.scalar);
  if (!CollectiveContinue(norm_launched)) return false;
  double local_squared = 0.0;
  const bool scalar_ready = TransportPolicy::DeviceToHost(
          &local_squared, view_.scalar, sizeof(double),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) &&
      TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
  if (!CollectiveContinue(scalar_ready)) return false;
  double global_squared = 0.0;
  const bool reduced = MPI_Allreduce(
      &local_squared, &global_squared, 1, MPI_DOUBLE, MPI_SUM,
      communicator_) == MPI_SUCCESS;
  if (!CollectiveContinue(reduced && std::isfinite(local_squared) &&
                          local_squared >= 0.0 &&
                          std::isfinite(global_squared) &&
                          global_squared >= 0.0))
    return false;
  norm = std::sqrt(global_squared);
  return CollectiveContinue(std::isfinite(norm));
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
PartitionedAuxiliaryNorm(state_slot& workspace, double& norm) noexcept
{
  norm = std::numeric_limits<double>::quiet_NaN();
  // Assembly status has already been consumed before this helper is called,
  // so the per-element status array can serve as bounded reduction scratch.
  const bool launched = CudaLaunchPartitionedAuxiliarySquaredNorm(
      shape_, workspace.interior.coefficients, workspace.trace.coefficients,
      view_.status, shape_.elements, view_.scalar);
  if (!CollectiveContinue(launched)) return false;
  double local_squared = 0.0;
  const bool scalar_ready = TransportPolicy::DeviceToHost(
          &local_squared, view_.scalar, sizeof(double),
          CudaTransferCategory::scalar_diagnostic,
          CudaTransferSite::scalar_status) &&
      TransportPolicy::Synchronize(CudaTransferSite::scalar_status);
  if (!CollectiveContinue(scalar_ready)) return false;
  double global_squared = 0.0;
  const bool reduced = MPI_Allreduce(
      &local_squared, &global_squared, 1, MPI_DOUBLE, MPI_SUM,
      communicator_) == MPI_SUCCESS;
  if (!CollectiveContinue(reduced && std::isfinite(local_squared) &&
                          local_squared >= 0.0 &&
                          std::isfinite(global_squared) &&
                          global_squared >= 0.0))
    return false;
  norm = std::sqrt(global_squared);
  return CollectiveContinue(std::isfinite(norm));
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
BootstrapInitialScalarClosure(const CudaConfig& config) noexcept
{
  if (!RequiresInitialScalarClosure()) return true;
  state_slot& workspace = workspace_slot_;
  const AttemptIdentity bootstrap_attempt{0, 0, 0, 0, config.dt};
  bool projected = false;
  {
    CudaIluFactorAttemptEndGuard end_factor_attempt(ilu_factor_state_);
    CudaElementSchwarzAttemptEndGuard end_schwarz_attempt(
        element_schwarz_attempt_state_);
    if (!PrepareAttempt(bootstrap_attempt, workspace, true))
    {
      ReportCudaSetupFailure(rank_, "initial-scalar-bootstrap-prepare");
      return false;
    }
    const bool partitioned_snapshot =
        TransportPolicy::DeviceToDevice(
            view_.partitioned_base_interior,
            workspace.interior.coefficients,
            workspace.interior.count * sizeof(double),
            CudaTransferCategory::attempt_history,
            CudaTransferSite::nonlinear_base_interior) &&
        TransportPolicy::DeviceToDevice(
            view_.trial_true_trace, workspace.trace.coefficients,
            workspace.trace.count * sizeof(double),
            CudaTransferCategory::attempt_history,
            CudaTransferSite::nonlinear_base_trace) &&
        TransportPolicy::Synchronize(
            CudaTransferSite::nonlinear_base_trace);
    if (!CollectiveContinue(partitioned_snapshot))
    {
      ReportCudaSetupFailure(rank_, "initial-scalar-bootstrap-snapshot");
      return false;
    }

    CudaNumericalResult result{};
    result.finite = true;
    result.tensor_core = initialization_tensor_core_evidence_;
    result.tensor_core.selected_engine = cuda_engine_;
    result.tensor_core.requested_trace_policy =
        trace_policy_descriptor_.requested_selection;
    result.tensor_core.resolved_trace_policy =
        trace_policy_descriptor_.resolved_selection;
    result.tensor_core.trace_policy_rank_count =
        trace_policy_descriptor_.rank_count;
    result.tensor_core.trace_policy_order =
        trace_policy_descriptor_.polynomial_order;
    result.tensor_core.trace_preconditioner = trace_preconditioner_;
    std::uint64_t product_generation = 0;
    projected = SolvePartitionedScalarClosure(
        workspace, generic::EulerHdgLinearSolveRole::final_coherent,
        result, product_generation);
    if (!projected)
    {
      ReportCudaSetupFailure(
          rank_, result.phase == CudaFailurePhase::assembly
              ? "initial-scalar-bootstrap-closure-assembly"
              : result.phase == CudaFailurePhase::condensation
                  ? "initial-scalar-bootstrap-closure-condensation"
                  : result.phase == CudaFailurePhase::trace_solve
                      ? "initial-scalar-bootstrap-closure-trace"
                      : "initial-scalar-bootstrap-closure-recovery");
    }
    projected = projected && result.finite && result.recovery;
    if (projected)
      projected = CudaLaunchPartitionedBackwardEulerStateAudit(
                      view_, shape_, false) && StatusPassed();
    if (!projected && result.recovery)
      ReportCudaSetupFailure(rank_, "initial-scalar-bootstrap-state-audit");
  }
  if (!CollectiveContinue(projected)) return false;

  const auto copy_interior = [&](double* destination) noexcept
  {
    return TransportPolicy::DeviceToDevice(
        destination, workspace.interior.coefficients,
        workspace.interior.count * sizeof(double),
        CudaTransferCategory::attempt_history,
        CudaTransferSite::attempt_seed_interior);
  };
  const auto copy_trace = [&](double* destination) noexcept
  {
    return TransportPolicy::DeviceToDevice(
        destination, workspace.trace.coefficients,
        workspace.trace.count * sizeof(double),
        CudaTransferCategory::attempt_history,
        CudaTransferSite::attempt_seed_trace);
  };
  const bool copied = copy_interior(committed_slot_.interior.coefficients) &&
      copy_interior(candidate_slot_.interior.coefficients) &&
      copy_interior(view_.initial_current_interior) &&
      copy_trace(committed_slot_.trace.coefficients) &&
      copy_trace(candidate_slot_.trace.coefficients) &&
      copy_trace(view_.initial_current_true_trace) &&
      TransportPolicy::Synchronize(CudaTransferSite::attempt_seed_trace);
  if (!CollectiveContinue(copied)) return false;
  committed_slot_.interior.generation = 0;
  committed_slot_.trace.generation = 0;
  candidate_slot_.interior.generation = 0;
  candidate_slot_.trace.generation = 0;
  if (!UpdateStateHashes(committed_slot_) ||
      !UpdateStateHashes(candidate_slot_) ||
      committed_slot_.interior.hash != candidate_slot_.interior.hash ||
      committed_slot_.trace.hash != candidate_slot_.trace.hash)
    return false;
  MarkCommittedScalarClosureCoherent();
  current_attempt_ = {};
  current_generation_ = 0;
  ticket_count_ = 0;
  return CommittedScalarClosureCoherent();
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
ValidatePartitionedScalarCheckpoint(
    const CudaSerializedCheckpointV1& checkpoint, state_slot& workspace,
    const double dt) noexcept
{
  if (!partitioned_backward_euler_) return true;
  const int true_scalar_count = shape_.true_trace_count /
      shape_.trace_components;
  bool finite = shape_.interior_components == igr_interior_components &&
      shape_.trace_components == igr_trace_components &&
      true_scalar_count > 0 && std::isfinite(dt) && dt > 0.0;
  for (int element = 0; finite && element < shape_.elements; ++element)
    for (int component = conserved_components;
         finite && component < shape_.interior_components; ++component)
      for (int scalar = 0; scalar < shape_.scalar_interior_dofs; ++scalar)
      {
        const std::size_t index = static_cast<std::size_t>(element) *
            shape_.interior_dofs + component * shape_.scalar_interior_dofs +
            scalar;
        const double value = checkpoint.interior[index];
        finite = std::isfinite(value);
      }
  for (int component = conserved_components;
       finite && component < shape_.trace_components; ++component)
    for (int scalar = 0; scalar < true_scalar_count; ++scalar)
    {
      const double value = checkpoint.owned_true_trace[
          component * true_scalar_count + scalar];
      finite = std::isfinite(value);
    }
  if (!CollectiveContinue(finite)) return false;

  const bool staged = TransportPolicy::DeviceToDevice(
          view_.old_interior, workspace.interior.coefficients,
          workspace.interior.count * sizeof(double),
          CudaTransferCategory::attempt_history,
          CudaTransferSite::attempt_seed_interior) &&
      DistributeCurrentTrace(workspace.trace.coefficients);
  if (!CollectiveContinue(staged)) return false;
  CudaTensorCoreEvidence scratch = initialization_tensor_core_evidence_;
  double global_residual = std::numeric_limits<double>::infinity();
  if (!AssemblePartitionedScalarClosure(
          workspace, dt, global_residual, scratch))
    return false;
  double global_auxiliary_norm = std::numeric_limits<double>::quiet_NaN();
  if (!PartitionedAuxiliaryNorm(workspace, global_auxiliary_norm))
    return false;
  const double relative_residual = global_residual /
      std::max(global_auxiliary_norm, 1.0);
  return CollectiveContinue(std::isfinite(relative_residual) &&
      relative_residual <= true_residual_relative_tolerance_);
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
SolvePartitionedScalarClosure(
    state_slot& workspace, const generic::EulerHdgLinearSolveRole role,
    CudaNumericalResult& result,
    std::uint64_t& product_generation) noexcept
{
  double closure_norm = std::numeric_limits<double>::infinity();
  if (!AssemblePartitionedScalarClosure(
          workspace, current_attempt_.dt, closure_norm, result.tensor_core))
  {
    result.phase = CudaFailurePhase::assembly;
    result.reason = FailureReason::finite_gate;
    return false;
  }
  if (result.trace_solves == 0) result.initial_residual = closure_norm;
  result.finite = result.finite && std::isfinite(closure_norm);
  if (!result.finite)
  {
    result.reason = FailureReason::finite_gate;
    return false;
  }
  result.phase = CudaFailurePhase::condensation;
  std::uint64_t condensation_tiles = 0;
  bool condensed = false;
  {
    ::igrlab::NVTXRange profile_region(
        ::igrlab::observability::RegionId::hdg_condensation);
    condensed = CopyBase(workspace) && Ticket(CudaFailurePhase::condensation) &&
        CudaLaunchCondensation(
            view_, shape_, cuda_engine_, condensation_tiles) &&
        ObserveLocalSolveWork() && StatusPassed();
  }
  if (cuda_engine_ == generic::CudaEngineKind::tensor_core && condensed)
  {
    if (result.tensor_core.condensation_launches ==
            std::numeric_limits<std::uint64_t>::max() ||
        condensation_tiles > std::numeric_limits<std::uint64_t>::max() -
            result.tensor_core.mma_tiles)
      condensed = false;
    else
    {
      ++result.tensor_core.condensation_launches;
      result.tensor_core.mma_tiles += condensation_tiles;
    }
  }
  if (!CollectiveContinue(condensed))
  {
    result.reason = FailureReason::backend_failure;
    return false;
  }
  result.phase = CudaFailurePhase::trace_solve;
  if (!AssembleTrueSystem() ||
      !CollectiveContinue(Ticket(CudaFailurePhase::trace_solve)))
  {
    result.reason = FailureReason::true_residual_gate;
    return false;
  }
  product_generation = ++linear_generation_;
  int iterations = 0;
  int solve_attempts = 0;
  int tightenings = 0;
  double true_relative_residual = std::numeric_limits<double>::infinity();
  const bool solved = SolveTrueSystem(
      role,
      ResolveCudaNativeTraceActiveSubspace(
          RequiresInitialScalarClosure(), cuda_engine_, ranks_,
          trace_solver_kind_, true),
      true_relative_residual, product_generation, iterations,
      solve_attempts, tightenings, result.tensor_core);
  int& role_iterations =
      role == generic::EulerHdgLinearSolveRole::newton_correction
      ? result.newton_correction_linear_iterations
      : result.final_coherent_linear_iterations;
  const bool bounded = iterations >= 0 && solve_attempts >= 0 &&
      tightenings >= 0 &&
      iterations <= std::numeric_limits<int>::max() -
          result.linear_iterations &&
      iterations <= std::numeric_limits<int>::max() - role_iterations &&
      solve_attempts <= std::numeric_limits<int>::max() -
          result.linear_solve_attempts &&
      tightenings <= std::numeric_limits<int>::max() -
          result.true_residual_tightenings &&
      result.scalar_closure_solves != std::numeric_limits<int>::max() &&
      iterations <= std::numeric_limits<int>::max() -
          result.scalar_closure_linear_iterations &&
      solve_attempts <= std::numeric_limits<int>::max() -
          result.scalar_closure_linear_solve_attempts &&
      tightenings <= std::numeric_limits<int>::max() -
          result.scalar_closure_true_residual_tightenings &&
      result.trace_solves != std::numeric_limits<int>::max();
  if (!CollectiveContinue(solved && bounded))
  {
    result.reason = FailureReason::true_residual_gate;
    return false;
  }
  result.linear_iterations += iterations;
  role_iterations += iterations;
  result.linear_solve_attempts += solve_attempts;
  result.true_residual_tightenings += tightenings;
  ++result.trace_solves;
  ++result.scalar_closure_solves;
  result.scalar_closure_linear_iterations += iterations;
  result.scalar_closure_linear_solve_attempts += solve_attempts;
  result.scalar_closure_true_residual_tightenings += tightenings;
  result.true_relative_residual = true_relative_residual;
  const bool distributed = DistributeTraceIncrement();
  result.phase = CudaFailurePhase::recovery;
  bool recovered = false;
  if (distributed && Ticket(CudaFailurePhase::recovery))
  {
    ::igrlab::NVTXRange profile_region(
        ::igrlab::observability::RegionId::hdg_interior_recovery);
    recovered = CudaLaunchRecover(view_, shape_, cuda_engine_) &&
        StatusPassed() && OriginalCoupledResidualPassed();
  }
  const bool frozen_euler_increments = recovered &&
      CudaLaunchPartitionedScalarClosureFreezeEulerIncrements(view_, shape_);
  if (!CollectiveContinue(frozen_euler_increments &&
                          CudaLaunchTrial(view_, shape_, 1.0) &&
                          CudaLaunchPartitionedScalarClosureRestoreEuler(
                              view_, shape_) &&
                          DistributeCurrentTrace(
                              workspace.trace.coefficients)))
  {
    result.reason = FailureReason::recovery_gate;
    return false;
  }
  // Recovery updates local sigma/z as well as the owned sigmahat trace.  A
  // converged condensed trace residual alone does not certify those recovered
  // local equations, so reassemble the exact candidate and gate the complete
  // scalar constraint before it can become a committed witness.
  double final_closure_norm = std::numeric_limits<double>::infinity();
  if (!AssemblePartitionedScalarClosure(
          workspace, current_attempt_.dt, final_closure_norm,
          result.tensor_core))
  {
    result.phase = CudaFailurePhase::assembly;
    result.reason = FailureReason::finite_gate;
    result.finite = false;
    return false;
  }
  double auxiliary_norm = std::numeric_limits<double>::quiet_NaN();
  if (!PartitionedAuxiliaryNorm(workspace, auxiliary_norm))
  {
    result.phase = CudaFailurePhase::recovery;
    result.reason = FailureReason::finite_gate;
    result.finite = false;
    return false;
  }
  const double closure_tolerance = true_residual_relative_tolerance_ *
      std::max(auxiliary_norm, 1.0);
  const double final_closure_relative_residual = final_closure_norm /
      std::max(auxiliary_norm, 1.0);
  result.finite = result.finite && std::isfinite(final_closure_norm) &&
      std::isfinite(closure_tolerance) &&
      std::isfinite(final_closure_relative_residual);
  if (!CollectiveContinue(result.finite &&
                          final_closure_norm <= closure_tolerance))
  {
    result.phase = CudaFailurePhase::recovery;
    result.reason = result.finite ? FailureReason::recovery_gate
                                  : FailureReason::finite_gate;
    return false;
  }
  result.true_relative_residual = std::max(
      result.true_relative_residual, final_closure_relative_residual);
  result.recovery = true;
  return true;
}

template <typename AllocatorPolicy, typename TransportPolicy>
bool CudaPersistentNumerics<AllocatorPolicy, TransportPolicy>::
AssembleAndNorm(state_slot&, const double dt, double& norm,
                CudaTensorCoreEvidence& tensor_core_evidence) noexcept
{
  TransportPolicy::SetPhase(CudaFailurePhase::assembly, current_generation_);
  bool assembly_launched = false;
  std::uint64_t invocation_volume_mma_tiles = 0;
  {
    ::igrlab::NVTXRange profile_region(
        ::igrlab::observability::RegionId::hdg_local_assembly);
    assembly_launched = CudaLaunchAssembly(
        view_, shape_, dt, cuda_engine_, invocation_volume_mma_tiles);
  }
  const bool local_assembly = assembly_launched && StatusPassed();
  if (!CollectiveContinue(local_assembly)) return false;
  if (partitioned_backward_euler_ &&
      !CollectiveContinue(
          CudaLaunchPartitionedBackwardEulerMask(view_, shape_) &&
          StatusPassed()))
    return false;

  std::uint64_t updated_volume_assembly_launches =
      tensor_core_evidence.volume_assembly_launches;
  std::uint64_t updated_volume_mma_tiles =
      tensor_core_evidence.volume_mma_tiles;
  const bool known_engine =
      cuda_engine_ == generic::CudaEngineKind::ordinary ||
      cuda_engine_ == generic::CudaEngineKind::tensor_core;
  const bool tensor_selected =
      cuda_engine_ == generic::CudaEngineKind::tensor_core;
  const bool volume_work_accounted = known_engine &&
      CudaAccumulateResearchVolumeWork(
          tensor_selected, shape_.borrowed_host_inputs,
          invocation_volume_mma_tiles,
          updated_volume_assembly_launches, updated_volume_mma_tiles);
  if (!CollectiveContinue(volume_work_accounted)) return false;
  tensor_core_evidence.volume_assembly_launches =
      updated_volume_assembly_launches;
  tensor_core_evidence.volume_mma_tiles = updated_volume_mma_tiles;

  if (!AssembleTrueVector(
          view_.residual_g, view_.true_residual, false))
    return false;
  // StatusPassed has consumed and synchronized every per-element assembly
  // status.  Later condensation/admissibility/assembly phases overwrite the
  // whole array before its next status use, so it is safe norm scratch here.
  {
    ::igrlab::NVTXRange profile_region(
        ::igrlab::observability::RegionId::hdg_reduction);
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
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_packing);
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
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_packing);
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
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_validation);
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
  ::igrlab::NVTXRange profile_region(
      ::igrlab::observability::RegionId::hdg_validation);
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
