#include "utils/diagnostics/euler_admissibility.hpp"

#include <climits>
#include <limits>
#include <new>

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
#include "mfem/general/cuda.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#endif

namespace mfem
{
namespace igrlab
{
namespace detail
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
namespace
{
constexpr int scan_threads = 128;

struct DeviceElement
{
  int map_offset, dof, rule_begin, rule_count, nodes, points;
};
struct DeviceRule
{
  int shape_offset, points, face, first;
};
// One record per element, then one record for the whole call. Reusable bounded
// workspace is overwritten completely; no prior result is an input. No sample
// state values or per-point results are transferred back to the host.
struct DeviceSummary
{
  double min_rho, min_pressure, first_rho, first_pressure;
  int bad_rho, bad_pressure, declined, min_rho_at, min_pressure_at;
  int first_at, element, face, point, scalar_dof, pressure_bad;
};

__device__ bool NormalOrZero(double x)
{
  const unsigned long long bits = __double_as_longlong(x);
  const unsigned long long exponent = bits & 0x7ff0000000000000ULL;
  const unsigned long long fraction = bits & 0x000fffffffffffffULL;
  return exponent != 0x7ff0000000000000ULL &&
         (exponent != 0 || fraction == 0);
}

__device__ double Multiply(double a, double b, int& declined)
{
  const double result = __dmul_rn(a, b);
  if (!NormalOrZero(result) || (result == 0.0 && a != 0.0 && b != 0.0))
  { declined = 1; }
  return result;
}
__device__ double Add(double a, double b, int& declined)
{
  const double result = __dadd_rn(a, b);
  if (!NormalOrZero(result)) { declined = 1; }
  return result;
}
__device__ double Subtract(double a, double b, int& declined)
{
  const double result = __dsub_rn(a, b);
  if (!NormalOrZero(result)) { declined = 1; }
  return result;
}
__device__ double Divide(double a, double b, int& declined)
{
  const double result = __ddiv_rn(a, b);
  if (!NormalOrZero(result) || (result == 0.0 && a != 0.0))
  { declined = 1; }
  return result;
}

__device__ DeviceSummary EmptySummary()
{
  DeviceSummary s{};
  s.min_rho = s.min_pressure = s.first_rho = s.first_pressure = CUDART_INF;
  s.min_rho_at = s.min_pressure_at = s.first_at = INT_MAX;
  s.element = s.face = s.point = s.scalar_dof = -1;
  return s;
}

__device__ void MergeSummary(DeviceSummary& a, const DeviceSummary& b)
{
  // Occurrence index is a tie breaker, not a changed FP reduction. It retains
  // the original first value on equality, including the sign of zero.
  if (b.min_rho < a.min_rho ||
      (b.min_rho == a.min_rho && b.min_rho_at < a.min_rho_at))
  { a.min_rho = b.min_rho; a.min_rho_at = b.min_rho_at; }
  if (b.min_pressure < a.min_pressure ||
      (b.min_pressure == a.min_pressure && b.min_pressure_at < a.min_pressure_at))
  { a.min_pressure = b.min_pressure; a.min_pressure_at = b.min_pressure_at; }
  a.bad_rho += b.bad_rho;
  a.bad_pressure += b.bad_pressure;
  a.declined |= b.declined;
  if (b.first_at < a.first_at)
  {
    a.first_at = b.first_at; a.element = b.element; a.face = b.face;
    a.point = b.point; a.scalar_dof = b.scalar_dof;
    a.pressure_bad = b.pressure_bad;
    a.first_rho = b.first_rho; a.first_pressure = b.first_pressure;
  }
}

__global__ void EulerAdmissibilityElementKernel(
  const mfem::real_t* state, const int* maps, const double* shapes,
  const DeviceElement* elements, const DeviceRule* rules,
  double gamma, double rho_min, double p_min, DeviceSummary* output)
{
  const int e = blockIdx.x;
  const DeviceElement element = elements[e];
  extern __shared__ double current[];
  __shared__ DeviceSummary partial[scan_threads];
  // Each current conservative value is read once from the true device state.
  for (int j = threadIdx.x; j < 4 * element.dof; j += blockDim.x)
  {
    const int index = maps[element.map_offset + j];
    current[j] = index >= 0 ? state[index] : -state[-index - 1];
  }
  __syncthreads();
  DeviceSummary local = EmptySummary();
  for (int occurrence = threadIdx.x; occurrence < element.points;
       occurrence += blockDim.x)
  {
    double u[4];
    int declined = 0, face = -1, point = occurrence, scalar = -1;
    if (occurrence < element.nodes)
    {
      scalar = occurrence;
      for (int eq = 0; eq < 4; ++eq)
      {
        u[eq] = current[scalar + eq * element.dof];
        if (!NormalOrZero(u[eq])) { declined = 1; }
      }
    }
    else
    {
      DeviceRule rule{};
      // Rules are the original volume, then ordered face/rule occurrences.
      // Duplicate incidences remain separate even when their shapes are shared.
      for (int r = 0; r < element.rule_count; ++r)
      {
        const DeviceRule candidate = rules[element.rule_begin + r];
        if (occurrence >= candidate.first &&
            occurrence - candidate.first < candidate.points)
        { rule = candidate; break; }
      }
      face = rule.face; point = occurrence - rule.first;
      const double* shape = shapes + rule.shape_offset + point * element.dof;
      for (int eq = 0; eq < 4; ++eq)
      {
        double accumulator = 0.0;
        int j = 0;
        // The native MFEM pointer overload has ascending paired additions and
        // a product-first odd tail. Never contract these operations into FMA.
        for (; j + 1 < element.dof; j += 2)
        {
          for (int lane = 0; lane < 2; ++lane)
          {
            const double value = current[j + lane + eq * element.dof];
            if (!NormalOrZero(value)) { declined = 1; }
            accumulator = Add(accumulator,
              Multiply(value, shape[j + lane], declined), declined);
          }
        }
        if (j < element.dof)
        {
          const double value = current[j + eq * element.dof];
          if (!NormalOrZero(value)) { declined = 1; }
          accumulator = Add(Multiply(value, shape[j], declined),
                            accumulator, declined);
        }
        u[eq] = accumulator;
      }
    }
    if (declined) { local.declined = 1; continue; }
    const double rho = u[0], safe_rho = rho < 1.0e-16 ? 1.0e-16 : rho;
    double mom2 = 0.0;
    for (int d = 0; d < 2; ++d)
    { mom2 = Add(mom2, Multiply(u[1 + d], u[1 + d], declined), declined); }
    const double kinetic = Divide(Multiply(0.5, mom2, declined), safe_rho, declined);
    // Preserve the separate original EOS momentum loop and its scalar order.
    double eos_mom2 = 0.0;
    for (int d = 0; d < 2; ++d)
    { eos_mom2 = Add(eos_mom2, Multiply(u[1 + d], u[1 + d], declined), declined); }
    const double eos_k = Divide(Multiply(0.5, eos_mom2, declined), safe_rho, declined);
    const double pressure = Multiply(Subtract(gamma, 1.0, declined),
      Subtract(u[3], eos_k, declined), declined);
    if (!NormalOrZero(kinetic) || declined)
    { local.declined = 1; continue; }
    if (rho < local.min_rho)
    { local.min_rho = rho; local.min_rho_at = occurrence; }
    if (pressure < local.min_pressure)
    { local.min_pressure = pressure; local.min_pressure_at = occurrence; }
    const bool bad_rho = rho < rho_min, bad_pressure = pressure < p_min;
    local.bad_rho += bad_rho;
    local.bad_pressure += bad_pressure;
    if (local.first_at == INT_MAX && (bad_rho || bad_pressure))
    {
      local.first_at = occurrence; local.element = e; local.face = face;
      local.point = point; local.scalar_dof = scalar;
      local.pressure_bad = !bad_rho && bad_pressure;
      local.first_rho = rho; local.first_pressure = pressure;
    }
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  for (int stride = scan_threads / 2; stride > 0; stride /= 2)
  {
    if (threadIdx.x < stride)
    { MergeSummary(partial[threadIdx.x], partial[threadIdx.x + stride]); }
    __syncthreads();
  }
  if (threadIdx.x == 0) { output[e] = partial[0]; }
}

__global__ void EulerAdmissibilityFoldKernel(
  const DeviceSummary* elements, int count, DeviceSummary* output)
{
  DeviceSummary result = EmptySummary();
  // This compact NE fold retains the original ascending-element min ties and
  // first diagnostic; all expensive sample interpolation is already parallel.
  for (int e = 0; e < count; ++e)
  {
    const DeviceSummary value = elements[e];
    if (value.min_rho < result.min_rho) { result.min_rho = value.min_rho; }
    if (value.min_pressure < result.min_pressure)
    { result.min_pressure = value.min_pressure; }
    result.bad_rho += value.bad_rho;
    result.bad_pressure += value.bad_pressure;
    result.declined |= value.declined;
    if (result.first_at == INT_MAX && value.first_at != INT_MAX)
    {
      result.first_at = value.first_at; result.element = e;
      result.face = value.face; result.point = value.point;
      result.scalar_dof = value.scalar_dof;
      result.pressure_bad = value.pressure_bad;
      result.first_rho = value.first_rho; result.first_pressure = value.first_pressure;
    }
  }
  *output = result;
}
} // namespace

struct AdmissibilityDevicePlan
{
  mfem::Array<int> maps;
  mfem::Array<double> shapes;
  mfem::Array<DeviceElement> elements;
  mfem::Array<DeviceRule> rules;
  mutable mfem::Array<DeviceSummary> per_element, final;
  int state_size = 0, max_dof = 0;
};

std::shared_ptr<AdmissibilityDevicePlan> MakeAdmissibilityDevicePlan(
  const std::vector<AdmissibilityElementMap>& elements,
  const std::vector<AdmissibilityShapeTable>& shapes,
  const std::vector<AdmissibilityElementSamples>& samples,
  EulerAdmissibilityPointSet point_set, int state_size,
  std::size_t available_bytes, std::size_t& charged_bytes)
{
  charged_bytes = 0;
  if (!AdmissibilityOrdinaryCuda() ||
      sizeof(mfem::real_t) != sizeof(double) || elements.empty() ||
      elements.size() > INT_MAX || state_size <= 0) { return {}; }
  const bool dof_only = point_set == EulerAdmissibilityPointSet::Dof;
  if (!dof_only && samples.size() != elements.size()) { return {}; }
  // Preflight without allocating: all later arrays have fixed upper bounds.
  // Counting every shared shape table is conservative when this point set uses
  // only a subset. Dof plans need neither shape values nor the offsets table.
  std::size_t map_count = 0, shape_count = 0, rule_count = 0;
  const std::size_t offset_count = dof_only ? 0 : shapes.size();
  if (offset_count > INT_MAX) { return {}; }
  int max_dof = 0;
  for (std::size_t e = 0; e < elements.size(); ++e)
  {
    const auto& source = elements[e];
    if (source.dof <= 0 || source.dof > INT_MAX / 4 ||
        source.vdofs.Size() != 4 * source.dof ||
        map_count > static_cast<std::size_t>(INT_MAX - 4 * source.dof))
    { return {}; }
    map_count += 4 * source.dof;
    max_dof = std::max(max_dof, source.dof);
    if (!dof_only)
    {
      if (rule_count >= INT_MAX ||
          samples[e].face_rules.size() > static_cast<std::size_t>(INT_MAX) - rule_count - 1)
      { return {}; }
      rule_count += 1 + samples[e].face_rules.size();
    }
  }
  if (!dof_only)
  {
    for (const auto& shape : shapes)
    {
      if (shape.values.size() > static_cast<std::size_t>(INT_MAX) - shape_count)
      { return {}; }
      shape_count += shape.values.size();
    }
  }
  std::size_t bytes = 0;
  const auto charge = [&](std::size_t count, std::size_t item_bytes) -> bool
  {
    if (count > (available_bytes - bytes) / item_bytes) { return false; }
    bytes += count * item_bytes;
    return true;
  };
  // Three simultaneous immutable payloads: exact packing, MFEM host, CUDA.
  // Both scratch mirrors can also be live. Include offsets and a conservative
  // allowance for the fixed number of allocation/control records (at most 18
  // payload allocations); allocator/driver-wide pools are outside this cache.
  if (!charge(1, sizeof(AdmissibilityDevicePlan) + 32u * 256u) ||
      !charge(map_count, 3 * sizeof(int)) ||
      !charge(shape_count, 3 * sizeof(double)) ||
      !charge(elements.size(), 3 * sizeof(DeviceElement)) ||
      !charge(rule_count, 3 * sizeof(DeviceRule)) ||
      !charge(elements.size() + 1, 2 * sizeof(DeviceSummary)) ||
      !charge(offset_count, sizeof(int))) { return {}; }
  const std::size_t mirror_bytes = map_count * sizeof(int) +
    shape_count * sizeof(double) + rule_count * sizeof(DeviceRule) +
    elements.size() * sizeof(DeviceElement);
  const std::size_t scratch_bytes = (elements.size() + 1) * sizeof(DeviceSummary);
  int device = 0, max_shared = 0;
  std::size_t free_bytes = 0, total_bytes = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device)
        != cudaSuccess ||
      4u * max_dof * sizeof(double) + scan_threads * sizeof(DeviceSummary) >
        static_cast<std::size_t>(max_shared) ||
      cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      mirror_bytes + scratch_bytes > free_bytes / 2) { return {}; }
  try
  {
    // Exact-size arrays avoid vector capacity slack and old/new allocation
    // overlap entirely. Their complete footprint was charged before allocation.
    auto packed_maps = std::make_unique<int[]>(map_count);
    auto packed_shapes = std::make_unique<double[]>(shape_count);
    auto packed_elements = std::make_unique<DeviceElement[]>(elements.size());
    auto packed_rules = std::make_unique<DeviceRule[]>(rule_count);
    auto offsets = std::make_unique<int[]>(offset_count);
    std::fill_n(offsets.get(), offset_count, -1);
    std::size_t maps_used = 0, shapes_used = 0, rules_used = 0;
    std::size_t occurrences = 0;
    const auto table_offset = [&](int table, int dof) -> int
    {
      if (table < 0 || static_cast<std::size_t>(table) >= shapes.size())
      { return -1; }
      const auto& shape = shapes[table];
      if (shape.dof != dof || shape.points <= 0 ||
          shape.values.size() != static_cast<std::size_t>(dof) * shape.points)
      { return -1; }
      if (offsets[table] >= 0) { return offsets[table]; }
      if (shape.values.size() > shape_count - shapes_used) { return -1; }
      for (const auto value : shape.values)
      { if (!std::isfinite(value) || std::fpclassify(value) == FP_SUBNORMAL) { return -1; } }
      offsets[table] = static_cast<int>(shapes_used);
      std::copy(shape.values.begin(), shape.values.end(), packed_shapes.get() + shapes_used);
      shapes_used += shape.values.size();
      return offsets[table];
    };
    for (std::size_t e = 0; e < elements.size(); ++e)
    {
      const auto& source = elements[e];
      DeviceElement element{static_cast<int>(maps_used), source.dof,
        static_cast<int>(rules_used), 0,
        (dof_only || source.enforce_nodes) ? source.dof : 0, 0};
      element.points = element.nodes;
      const int* map = source.vdofs.HostRead();
      for (int j = 0; j < source.vdofs.Size(); ++j)
      {
        const long long decoded = map[j] >= 0 ? map[j] : -1LL - map[j];
        if (decoded >= state_size) { return {}; }
        packed_maps[maps_used++] = map[j];
      }
      const auto add_rule = [&](int table, int face) -> bool
      {
        const int offset = table_offset(table, source.dof);
        if (offset < 0 || rules_used >= rule_count ||
            shapes[table].points > INT_MAX - element.points) { return false; }
        packed_rules[rules_used++] = {offset, shapes[table].points, face, element.points};
        element.points += shapes[table].points;
        ++element.rule_count;
        return true;
      };
      if (!dof_only)
      {
        if (!add_rule(samples[e].volume_shape_table, -1)) { return {}; }
        for (const auto& rule : samples[e].face_rules)
        { if (!add_rule(rule.shape_table, rule.face)) { return {}; } }
      }
      occurrences += static_cast<std::size_t>(element.points);
      if (occurrences > INT_MAX) { return {}; } // Original counters use int.
      packed_elements[e] = element;
    }
    if (maps_used != map_count || rules_used != rule_count) { return {}; }
    auto plan = std::make_shared<AdmissibilityDevicePlan>();
    plan->maps.SetSize(static_cast<int>(maps_used));
    plan->shapes.SetSize(static_cast<int>(shapes_used));
    plan->elements.SetSize(static_cast<int>(elements.size()));
    plan->rules.SetSize(static_cast<int>(rules_used));
    plan->per_element.SetSize(static_cast<int>(elements.size()));
    plan->final.SetSize(1);
    // Fresh MFEM arrays allocate the requested capacity exactly. Retain the
    // capacity check so a changed allocation policy cannot upload excess data.
    if (plan->maps.Capacity() != static_cast<int>(maps_used) ||
        plan->shapes.Capacity() != static_cast<int>(shapes_used) ||
        plan->elements.Capacity() != static_cast<int>(elements.size()) ||
        plan->rules.Capacity() != static_cast<int>(rules_used) ||
        plan->per_element.Capacity() != static_cast<int>(elements.size()) ||
        plan->final.Capacity() != 1) { return {}; }
    plan->maps.GetMemory().UseDevice(true); plan->shapes.GetMemory().UseDevice(true);
    plan->elements.GetMemory().UseDevice(true); plan->rules.GetMemory().UseDevice(true);
    plan->per_element.GetMemory().UseDevice(true); plan->final.GetMemory().UseDevice(true);
    std::copy_n(packed_maps.get(), maps_used, plan->maps.HostWrite());
    if (shapes_used)
    { std::copy_n(packed_shapes.get(), shapes_used, plan->shapes.HostWrite()); }
    std::copy_n(packed_elements.get(), elements.size(), plan->elements.HostWrite());
    if (rules_used)
    { std::copy_n(packed_rules.get(), rules_used, plan->rules.HostWrite()); }
    plan->state_size = state_size; plan->max_dof = max_dof;
    // Allocate/upload once with the plan, never at a hot scan boundary.
    plan->maps.Read(true); plan->elements.Read(true);
    if (plan->shapes.Size()) { plan->shapes.Read(true); }
    if (plan->rules.Size()) { plan->rules.Read(true); }
    plan->per_element.Write(true); plan->final.Write(true);
    charged_bytes = bytes;
    return plan;
  }
  catch (const std::bad_alloc&) { return {}; }
}

bool EvaluateAdmissibilityDevice(const AdmissibilityDevicePlan& plan,
  const mfem::Vector& state, double gamma, double rho_min, double p_min,
  EulerAdmissibilitySummary& local)
{
  if (!AdmissibilityOrdinaryCuda() || state.Size() != plan.state_size ||
      !state.GetMemory().DeviceIsValid() ||
      !std::isfinite(gamma) || !std::isfinite(rho_min) || !std::isfinite(p_min) ||
      std::fpclassify(gamma) == FP_SUBNORMAL ||
      std::fpclassify(rho_min) == FP_SUBNORMAL ||
      std::fpclassify(p_min) == FP_SUBNORMAL) { return false; }
    EulerAdmissibilityElementKernel<<<plan.elements.Size(), scan_threads,
                                    4u * plan.max_dof * sizeof(double)>>>(
      state.Read(true), plan.maps.Read(true),
      plan.shapes.Size() ? plan.shapes.Read(true) : nullptr,
      plan.elements.Read(true), plan.rules.Size() ? plan.rules.Read(true) : nullptr,
      gamma, rho_min, p_min, plan.per_element.Write(true));
    MFEM_GPU_CHECK(cudaGetLastError());
    EulerAdmissibilityFoldKernel<<<1, 1>>>(
      plan.per_element.Read(true), plan.per_element.Size(), plan.final.Write(true));
    MFEM_GPU_CHECK(cudaGetLastError());
    const DeviceSummary result = plan.final.HostRead()[0];
    if (result.declined) { return false; }
    local = EulerAdmissibilitySummary{};
    local.global_min_rho = result.min_rho;
    local.global_min_pressure = result.min_pressure;
    local.global_bad_rho = result.bad_rho;
    local.global_bad_pressure = result.bad_pressure;
    if (result.first_at != INT_MAX)
    {
      auto& issue = local.first_local_issue;
      issue.found = true; issue.pressure_bad = result.pressure_bad;
      issue.face_point = result.face >= 0; issue.element = result.element;
      issue.face = result.face; issue.point_index = result.point;
      issue.scalar_dof = result.scalar_dof;
      issue.rho = result.first_rho; issue.pressure = result.first_pressure;
    }
    return true;
}
#else
// The canonical source remains usable without native CUDA compilation. No
// CXX-only or alternate backend may accidentally execute CUDA expressions.
struct AdmissibilityDevicePlan {};
std::shared_ptr<AdmissibilityDevicePlan> MakeAdmissibilityDevicePlan(
  const std::vector<AdmissibilityElementMap>&,
  const std::vector<AdmissibilityShapeTable>&,
  const std::vector<AdmissibilityElementSamples>&,
  EulerAdmissibilityPointSet, int, std::size_t, std::size_t& charged_bytes)
{ charged_bytes = 0; return {}; }
bool EvaluateAdmissibilityDevice(const AdmissibilityDevicePlan&,
  const mfem::Vector&, double, double, double, EulerAdmissibilitySummary&)
{ return false; }
#endif
} // namespace detail
} // namespace igrlab
} // namespace mfem
