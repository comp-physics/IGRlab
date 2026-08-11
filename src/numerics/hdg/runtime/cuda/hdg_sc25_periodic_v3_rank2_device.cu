#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr_rank2_impl.cuh"
#include "numerics/hdg/runtime/sha256.hpp"

#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
#include <_hypre_parcsr_ls.h>
#include <_hypre_utilities.h>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace scalar = dg::hdg::runtime::cuda_w8b::sc25;
namespace scalar_detail = dg::hdg::runtime::cuda_w8b::sc25::detail;

namespace
{

constexpr int ns = scalar::volume_nodes;
constexpr int face_nodes = scalar::face_nodes;
constexpr int nf = 4 * scalar::face_nodes;
constexpr int ni = scalar::local_interior_unknowns;
constexpr int nt = scalar::local_trace_unknowns;
constexpr double gamma_value = 1.4;
constexpr double alpha_value = 0.1;
constexpr double state_tau = 2.5;
constexpr double pi_value = 3.141592653589793238462643383279502884;
constexpr double local_relative_tolerance = 2.0e-12;
constexpr double local_absolute_tolerance = 2.0e-13;
constexpr std::uint64_t structural_host_numerical_fallback_paths = 0;
static_assert(structural_host_numerical_fallback_paths == 0,
              "the W8B numerical path has no host fallback branch");

std::atomic<std::uint64_t> next_owner_incarnation{1};

bool ClaimOwnerIncarnation(std::uint64_t& output) noexcept
{
  output = 0;
  std::uint64_t candidate = next_owner_incarnation.load(
      std::memory_order_relaxed);
  while (candidate != 0 &&
         candidate != std::numeric_limits<std::uint64_t>::max())
  {
    if (next_owner_incarnation.compare_exchange_weak(
            candidate, candidate + 1, std::memory_order_relaxed,
            std::memory_order_relaxed))
    {
      output = candidate;
      return true;
    }
  }
  return false;
}

bool HypreCudaBackendAdmissionValid() noexcept
{
  // A stale HYPRE error is backend evidence.  Capture it before restoring the
  // process-global flag to a clean state, and do not initialize through it.
  const HYPRE_Int incoming_error = HYPRE_GetError();
  (void)HYPRE_ClearAllErrors();
  if (incoming_error != 0) { return false; }

  bool init_returned = false;
  try
  {
    mfem::Hypre::Init();
    init_returned = true;
  }
  catch (...) {}
  const HYPRE_Int init_error = HYPRE_GetError();
  (void)HYPRE_ClearAllErrors();
  if (!init_returned || init_error != 0) { return false; }

  HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_UNDEFINED;
  HYPRE_ExecutionPolicy execution_policy = HYPRE_EXEC_UNDEFINED;
  const bool hypre_gpu = mfem::HypreUsingGPU();
  const bool memory_query_valid =
      HYPRE_GetMemoryLocation(&memory_location) == 0;
  const bool execution_query_valid =
      HYPRE_GetExecutionPolicy(&execution_policy) == 0;
  const HYPRE_Int policy_error = HYPRE_GetError();
  (void)HYPRE_ClearAllErrors();
  return hypre_gpu && memory_query_valid && execution_query_valid &&
      memory_location == HYPRE_MEMORY_DEVICE &&
      execution_policy == HYPRE_EXEC_DEVICE && policy_error == 0;
}

bool HypreTrueSystemDevicePolicyValid(
    mfem::HypreParMatrix& matrix, mfem::HypreParVector& rhs,
    mfem::HypreParVector& solution) noexcept
{
  hypre_ParCSRMatrix* raw_matrix = matrix;
  hypre_ParVector* raw_rhs = rhs;
  hypre_ParVector* raw_solution = solution;
  if (raw_matrix == nullptr || raw_rhs == nullptr ||
      raw_solution == nullptr ||
      hypre_ParVectorLocalVector(raw_rhs) == nullptr ||
      hypre_ParVectorLocalVector(raw_solution) == nullptr)
  {
    return false;
  }
  const HYPRE_MemoryLocation matrix_location =
      hypre_ParCSRMatrixMemoryLocation(raw_matrix);
  const HYPRE_MemoryLocation rhs_location =
      hypre_ParVectorMemoryLocation(raw_rhs);
  const HYPRE_MemoryLocation solution_location =
      hypre_ParVectorMemoryLocation(raw_solution);
  return ::dg::hdg::runtime::cuda_w8a::UmpireDeviceAllocatorPolicy::
             DevicePoolPolicyValid() &&
      matrix_location == HYPRE_MEMORY_DEVICE &&
      rhs_location == HYPRE_MEMORY_DEVICE &&
      solution_location == HYPRE_MEMORY_DEVICE &&
      hypre_GetExecPolicy1(matrix_location) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy2(matrix_location, rhs_location) ==
          HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy2(matrix_location, solution_location) ==
          HYPRE_EXEC_DEVICE;
}

__device__ inline bool DeviceFinite(const double value)
{
  return isfinite(value) != 0;
}

__device__ inline int InteriorIndex(const int element,
                                    const int component,
                                    const int scalar_node)
{
  if (component < scalar::state_components)
  {
    return (element * scalar::state_components + component) * ns +
        scalar_node;
  }
  if (component == scalar::state_components)
  {
    return scalar::persistent_u_unknowns + element * ns +
        scalar_node;
  }
  return scalar::persistent_u_unknowns +
      scalar::persistent_sigma_unknowns +
      (element * scalar::flux_components +
       component - scalar::state_components - 1) * ns + scalar_node;
}

__device__ inline double* CurrentInterior(scalar_detail::DeviceView& view)
{
  return view.selected_current_slot == 2 ? view.trial_interior :
      view.slot_interior[view.selected_current_slot];
}

__device__ inline double* CurrentTrace(scalar_detail::DeviceView& view)
{
  return view.selected_current_slot == 2 ? view.trial_trace :
      view.slot_trace[view.selected_current_slot];
}

__device__ inline int LocalIndex(const int component, const int scalar_node)
{
  return component * ns + scalar_node;
}

__device__ inline int TraceIndex(const int component,
                                 const int face_scalar)
{
  return component * nf + face_scalar;
}

__device__ inline double InteriorValue(const double* state,
                                       const int element,
                                       const int component,
                                       const int scalar_node)
{
  return state[InteriorIndex(element, component, scalar_node)];
}

__device__ inline double TraceValue(const scalar_detail::DeviceView& view,
                                    const double* state,
                                    const int element,
                                    const int local_row)
{
  const int map = element * nt + local_row;
  return static_cast<double>(view.local_trace_sign[map]) *
      state[view.local_trace_to_true[map]];
}

__device__ inline double EvaluateInterior(const double* state,
                                          const int element,
                                          const int component,
                                          const double* shape)
{
  double value = 0.0;
  for (int a = 0; a < ns; ++a)
  {
    value += InteriorValue(state, element, component, a) * shape[a];
  }
  return value;
}

__device__ inline double EvaluateDerivative(const double* state,
                                            const int element,
                                            const int component,
                                            const double* derivative,
                                            const int direction)
{
  double value = 0.0;
  for (int a = 0; a < ns; ++a)
  {
    value += InteriorValue(state, element, component, a) *
        derivative[a * scalar::dimension + direction];
  }
  return value;
}

__device__ inline bool Primitive(const double u[4], double& pressure,
                                 double velocity[2])
{
  if (!(u[0] > 0.0) || !DeviceFinite(u[0]) || !DeviceFinite(u[1]) ||
      !DeviceFinite(u[2]) || !DeviceFinite(u[3]))
  {
    return false;
  }
  velocity[0] = u[1] / u[0];
  velocity[1] = u[2] / u[0];
  pressure = (gamma_value - 1.0) *
      (u[3] - 0.5 * (u[1] * u[1] + u[2] * u[2]) / u[0]);
  return pressure > 0.0 && DeviceFinite(pressure) &&
      DeviceFinite(velocity[0]) && DeviceFinite(velocity[1]);
}

__device__ inline bool Admissible(const double u[4])
{
  double pressure = 0.0;
  double velocity[2]{};
  if (!Primitive(u, pressure, velocity) || u[0] < 1.0e-8 ||
      pressure < 1.0e-8)
  {
    return false;
  }
  const double speed = hypot(velocity[0], velocity[1]);
  const double sound = sqrt(gamma_value * pressure / u[0]);
  return DeviceFinite(speed) && DeviceFinite(sound) && speed + sound < 2.5;
}

__device__ inline bool FiniteConservativeState(const double u[4])
{
  return DeviceFinite(u[0]) && DeviceFinite(u[1]) &&
      DeviceFinite(u[2]) && DeviceFinite(u[3]);
}

__device__ inline bool EulerSigmaFlux(const double u[4], const double sigma,
                                      const double normal[2],
                                      double flux[4], double jacobian[4][5])
{
  double pressure = 0.0;
  double velocity[2]{};
  if (!Primitive(u, pressure, velocity) || !DeviceFinite(sigma))
  {
    return false;
  }
  const double rho = u[0];
  const double mx = u[1];
  const double my = u[2];
  const double energy = u[3];
  const double un = velocity[0] * normal[0] + velocity[1] * normal[1];
  flux[0] = mx * normal[0] + my * normal[1];
  flux[1] = mx * un + (pressure + sigma) * normal[0];
  flux[2] = my * un + (pressure + sigma) * normal[1];
  flux[3] = (energy + pressure + sigma) * un;
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 5; ++column)
    {
      jacobian[row][column] = 0.0;
    }
  const double gm1 = gamma_value - 1.0;
  const double momentum_square = mx * mx + my * my;
  const double pressure_derivative[4]{
      0.5 * gm1 * momentum_square / (rho * rho),
      -gm1 * mx / rho, -gm1 * my / rho, gm1};
  const double normal_velocity_derivative[4]{
      -un / rho, normal[0] / rho, normal[1] / rho, 0.0};
  jacobian[0][1] = normal[0];
  jacobian[0][2] = normal[1];
  for (int column = 0; column < 4; ++column)
  {
    jacobian[1][column] = (column == 1 ? un : 0.0) +
        mx * normal_velocity_derivative[column] +
        pressure_derivative[column] * normal[0];
    jacobian[2][column] = (column == 2 ? un : 0.0) +
        my * normal_velocity_derivative[column] +
        pressure_derivative[column] * normal[1];
    jacobian[3][column] =
        ((column == 3 ? 1.0 : 0.0) + pressure_derivative[column]) * un +
        (energy + pressure + sigma) * normal_velocity_derivative[column];
  }
  jacobian[1][4] = normal[0];
  jacobian[2][4] = normal[1];
  jacobian[3][4] = un;
  bool finite = DeviceFinite(flux[0]) && DeviceFinite(flux[1]) &&
      DeviceFinite(flux[2]) && DeviceFinite(flux[3]);
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 5; ++column)
    {
      finite = finite && DeviceFinite(jacobian[row][column]);
    }
  return finite;
}

__device__ inline void SmoothState(const double x, const double y,
                                   double value[4])
{
  const double rho = 1.0 + 0.05 * sin(2.0 * pi_value * x) *
      sin(2.0 * pi_value * y);
  const double u = 0.1 * sin(2.0 * pi_value * x);
  const double v = 0.1 * sin(2.0 * pi_value * y);
  value[0] = rho;
  value[1] = rho * u;
  value[2] = rho * v;
  value[3] = 1.0 / (gamma_value - 1.0) +
      0.5 * rho * (u * u + v * v);
}

__device__ inline void SmoothTransitionState(const double x, double value[4])
{
  const double kappa = atanh(0.8) / sin(pi_value / 8.0);
  const double chi = 0.5 * (1.0 - tanh(kappa * cos(2.0 * pi_value * x)));
  const double rho = 0.125 + 0.875 * chi;
  const double pressure = 0.1 + 0.9 * chi;
  value[0] = rho;
  value[1] = 0.0;
  value[2] = 0.0;
  value[3] = pressure / (gamma_value - 1.0);
}

template <int N>
__device__ bool SmallFullPivotSolve(double matrix[N * N],
                                    double rhs[N], double solution[N])
{
  int columns[N];
  for (int i = 0; i < N; ++i) { columns[i] = i; }
  for (int k = 0; k < N; ++k)
  {
    int pivot_row = k;
    int pivot_column = k;
    double pivot = 0.0;
    double scale = 0.0;
    for (int row = k; row < N; ++row)
      for (int column = k; column < N; ++column)
      {
        const double candidate = fabs(matrix[row * N + column]);
        scale = fmax(scale, candidate);
        if (candidate > pivot)
        {
          pivot = candidate;
          pivot_row = row;
          pivot_column = column;
        }
      }
    if (!(scale > 0.0) || pivot <= 64.0 * DBL_EPSILON * scale)
    {
      return false;
    }
    if (pivot_row != k)
    {
      for (int column = 0; column < N; ++column)
      {
        const double temporary = matrix[k * N + column];
        matrix[k * N + column] = matrix[pivot_row * N + column];
        matrix[pivot_row * N + column] = temporary;
      }
      const double temporary = rhs[k];
      rhs[k] = rhs[pivot_row];
      rhs[pivot_row] = temporary;
    }
    if (pivot_column != k)
    {
      for (int row = 0; row < N; ++row)
      {
        const double temporary = matrix[row * N + k];
        matrix[row * N + k] = matrix[row * N + pivot_column];
        matrix[row * N + pivot_column] = temporary;
      }
      const int temporary = columns[k];
      columns[k] = columns[pivot_column];
      columns[pivot_column] = temporary;
    }
    for (int row = k + 1; row < N; ++row)
    {
      const double multiplier = matrix[row * N + k] / matrix[k * N + k];
      matrix[row * N + k] = multiplier;
      rhs[row] -= multiplier * rhs[k];
      for (int column = k + 1; column < N; ++column)
      {
        matrix[row * N + column] -= multiplier * matrix[k * N + column];
      }
    }
  }
  double permuted[N]{};
  for (int row = N - 1; row >= 0; --row)
  {
    double value = rhs[row];
    for (int column = row + 1; column < N; ++column)
    {
      value -= matrix[row * N + column] * permuted[column];
    }
    permuted[row] = value / matrix[row * N + row];
  }
  for (int position = 0; position < N; ++position)
  {
    solution[columns[position]] = permuted[position];
  }
  return true;
}

__device__ bool FactorPartialPivot(double* matrix, int* row_pivots)
{
  for (int k = 0; k < ni; ++k)
  {
    int pivot_row = k;
    double maximum = fabs(matrix[k * ni + k]);
    for (int row = k + 1; row < ni; ++row)
    {
      const double candidate = fabs(matrix[row * ni + k]);
      if (candidate > maximum)
      {
        maximum = candidate;
        pivot_row = row;
      }
    }
    if (!DeviceFinite(maximum) || !(maximum > 0.0))
    {
      return false;
    }
    row_pivots[k] = pivot_row;
    if (pivot_row != k)
    {
      for (int column = 0; column < ni; ++column)
      {
        const double temporary = matrix[k * ni + column];
        matrix[k * ni + column] = matrix[pivot_row * ni + column];
        matrix[pivot_row * ni + column] = temporary;
      }
    }
    for (int row = k + 1; row < ni; ++row)
    {
      matrix[row * ni + k] /= matrix[k * ni + k];
      for (int column = k + 1; column < ni; ++column)
      {
        matrix[row * ni + column] -=
            matrix[row * ni + k] * matrix[k * ni + column];
      }
    }
  }
  for (int entry = 0; entry < ni * ni; ++entry)
  {
    if (!DeviceFinite(matrix[entry])) { return false; }
  }
  return true;
}

__device__ bool SolvePartialPivot(const double* lu, const int* row_pivots,
                                  const double* rhs, double* solution)
{
  double work[ni];
  for (int row = 0; row < ni; ++row)
  {
    if (!DeviceFinite(rhs[row])) { return false; }
    work[row] = rhs[row];
  }
  for (int k = 0; k < ni; ++k)
  {
    const int pivot = row_pivots[k];
    if (pivot < k || pivot >= ni) { return false; }
    if (pivot != k)
    {
      const double temporary = work[k];
      work[k] = work[pivot];
      work[pivot] = temporary;
    }
  }
  for (int row = 0; row < ni; ++row)
    for (int column = 0; column < row; ++column)
    {
      work[row] -= lu[row * ni + column] * work[column];
    }
  for (int row = ni - 1; row >= 0; --row)
  {
    for (int column = row + 1; column < ni; ++column)
    {
      work[row] -= lu[row * ni + column] * work[column];
    }
    if (!DeviceFinite(lu[row * ni + row]) || lu[row * ni + row] == 0.0)
    {
      return false;
    }
    work[row] /= lu[row * ni + row];
  }
  for (int position = 0; position < ni; ++position)
  {
    if (!DeviceFinite(work[position])) { return false; }
    solution[position] = work[position];
  }
  return true;
}

__device__ double ApplicationResidual(const double* matrix,
                                      const double* solution,
                                      const double* rhs)
{
  double square = 0.0;
  for (int row = 0; row < ni; ++row)
  {
    double value = -rhs[row];
    for (int column = 0; column < ni; ++column)
    {
      value += matrix[row * ni + column] * solution[column];
    }
    square += value * value;
  }
  return sqrt(square);
}

__device__ void PartialPivotContract(scalar_detail::DeviceView& view)
{
  for (int item = 0; item < 16; ++item) view.reduction[item] = 0.0;
  double matrix[ni * ni]{};
  int pivots[ni]{};
  double rhs[ni]{};
  double solution[ni]{};
  for (int row = 0; row < ni; ++row)
  {
    matrix[row * ni + row] = 1.0;
    const double expected = static_cast<double>(row + 1) / 32.0;
    rhs[row] = expected;
  }
  matrix[0] = 1.0;
  matrix[1] = 100.0;
  matrix[ni] = 2.0;
  rhs[0] = 1.0 / 32.0 + 100.0 * 2.0 / 32.0;
  rhs[1] = 2.0 / 32.0 + 2.0 / 32.0;
  bool row_contract = FactorPartialPivot(matrix, pivots) && pivots[0] == 1 &&
      SolvePartialPivot(matrix, pivots, rhs, solution);
  for (int row = 0; row < ni; ++row)
  {
    row_contract = row_contract &&
        fabs(solution[row] - static_cast<double>(row + 1) / 32.0) <= 1.0e-12;
  }

  for (double& entry : matrix) { entry = 0.0; }
  for (int row = 0; row < ni; ++row) matrix[row * ni + row] = 1.0;
  matrix[0] = 1.0e-300;
  matrix[1] = 1.0;
  matrix[ni] = 2.0e-300;
  const bool small_contract =
      2.0e-300 <= 64.0 * DBL_EPSILON &&
      FactorPartialPivot(matrix, pivots) && pivots[0] == 1 &&
      matrix[0] == 2.0e-300;
  view.reduction[12] = row_contract ? 1.0 : 0.0;
  view.reduction[13] = small_contract ? 1.0 : 0.0;
}

__device__ double VectorNorm(const double* values, const int count)
{
  double square = 0.0;
  for (int i = 0; i < count; ++i) { square += values[i] * values[i]; }
  return sqrt(square);
}

__device__ void RecordArchitecture(scalar_detail::DeviceView& view)
{
#if defined(__CUDA_ARCH__)
  *view.compiled_architecture = __CUDA_ARCH__;
  if (__CUDA_ARCH__ != scalar::required_native_architecture)
  {
    *view.status |= scalar_detail::device_wrong_architecture;
  }
#else
  *view.compiled_architecture = 0;
  *view.status |= scalar_detail::device_wrong_architecture;
#endif
}

__device__ void InitializeProjection(scalar_detail::DeviceView& view)
{
  double* interior = view.slot_interior[0];
  double* trace = view.slot_trace[0];
  const int volume_qpoints = view.smooth_transition != 0
      ? view.projection_volume_quadrature_points
      : view.volume_quadrature_points;
  const int face_qpoints = view.smooth_transition != 0
      ? view.projection_face_quadrature_points
      : view.face_quadrature_points;
  const double* volume_weight = view.smooth_transition != 0
      ? view.projection_volume_weight : view.volume_weight;
  const double* volume_shape = view.smooth_transition != 0
      ? view.projection_volume_shape : view.volume_shape;
  const double* volume_x = view.smooth_transition != 0
      ? view.projection_volume_x : view.volume_x;
  const double* face_weight = view.smooth_transition != 0
      ? view.projection_face_weight : view.face_weight;
  const double* face_oriented_shape = view.smooth_transition != 0
      ? view.projection_face_oriented_shape : view.face_oriented_shape;
  const double* face_x = view.smooth_transition != 0
      ? view.projection_face_x : view.face_x;
  for (int i = 0; i < scalar::persistent_interior_unknowns; ++i)
  {
    interior[i] = 0.0;
  }
  for (int i = 0; i < view.true_trace_unknowns; ++i) { trace[i] = 0.0; }
  for (int element = 0; element < view.elements; ++element)
  {
    double mass[ns * ns]{};
    double rhs[scalar::state_components][ns]{};
    for (int q = 0; q < volume_qpoints; ++q)
    {
      const int qindex = element * volume_qpoints + q;
      const double* shape = volume_shape + qindex * ns;
      double state[4]{};
      if (view.smooth_transition != 0)
        SmoothTransitionState(volume_x[qindex], state);
      else
        SmoothState(view.volume_x[qindex], view.volume_y[qindex], state);
      const double weight = volume_weight[qindex];
      for (int row = 0; row < ns; ++row)
      {
        for (int column = 0; column < ns; ++column)
        {
          mass[row * ns + column] +=
              weight * shape[row] * shape[column];
        }
        for (int component = 0; component < scalar::state_components;
             ++component)
        {
          rhs[component][row] += weight * shape[row] * state[component];
        }
      }
    }
    for (int component = 0; component < scalar::state_components;
         ++component)
    {
      double matrix[ns * ns];
      double right[ns];
      double solution[ns]{};
      for (int i = 0; i < ns * ns; ++i) { matrix[i] = mass[i]; }
      for (int i = 0; i < ns; ++i) { right[i] = rhs[component][i]; }
      if (!SmallFullPivotSolve<ns>(matrix, right, solution))
      {
        *view.status |= scalar_detail::device_singular_local_block;
        return;
      }
      for (int a = 0; a < ns; ++a)
      {
        interior[InteriorIndex(element, component, a)] = solution[a];
      }
    }
  }
  // Each periodic face occurs twice in the element-face traversal.  The
  // sequential writes are identical in canonical true-dof orientation.
  for (int element = 0; element < view.elements; ++element)
    for (int local_face = 0; local_face < 4; ++local_face)
    {
      double mass[face_nodes * face_nodes]{};
      double rhs[scalar::state_components][face_nodes]{};
      for (int q = 0; q < face_qpoints; ++q)
      {
        const int qindex =
            (element * 4 + local_face) * face_qpoints + q;
        const double* shape = face_oriented_shape + qindex * face_nodes;
        double state[4]{};
        if (view.smooth_transition != 0)
          SmoothTransitionState(face_x[qindex], state);
        else
          SmoothState(view.face_x[qindex], view.face_y[qindex], state);
        const double weight = face_weight[qindex];
        for (int row = 0; row < face_nodes; ++row)
        {
          for (int column = 0; column < face_nodes; ++column)
          {
            mass[row * face_nodes + column] +=
                weight * shape[row] * shape[column];
          }
          for (int component = 0; component < scalar::state_components;
               ++component)
          {
            rhs[component][row] +=
                weight * shape[row] * state[component];
          }
        }
      }
      for (int component = 0; component < scalar::state_components;
           ++component)
      {
        double matrix[face_nodes * face_nodes];
        double right[face_nodes];
        double solution[face_nodes]{};
        for (int i = 0; i < face_nodes * face_nodes; ++i)
        {
          matrix[i] = mass[i];
        }
        for (int i = 0; i < face_nodes; ++i) { right[i] = rhs[component][i]; }
        if (!SmallFullPivotSolve<face_nodes>(matrix, right, solution))
        {
          *view.status |= scalar_detail::device_singular_local_block;
          return;
        }
        for (int a = 0; a < face_nodes; ++a)
        {
          const int local_row =
              TraceIndex(component, local_face * face_nodes + a);
          const int map = element * nt + local_row;
          trace[view.local_trace_to_true[map]] =
              static_cast<double>(view.local_trace_sign[map]) * solution[a];
        }
      }
    }
  const std::size_t interior_count = scalar::persistent_interior_unknowns;
  for (std::size_t i = 0; i < interior_count; ++i)
  {
    view.slot_interior[1][i] = view.slot_interior[0][i];
  }
  for (int i = 0; i < view.true_trace_unknowns; ++i)
  {
    view.slot_trace[1][i] = view.slot_trace[0][i];
  }
}

__device__ void InitializeFixture(scalar_detail::DeviceView& view)
{
  for (int element = 0; element < view.elements; ++element)
  {
    const int n = 3;
    const int ex = element % n;
    const int ey = element / n;
    const double parity = ((ex + ey) % 2 == 0) ? 1.0 : -1.0;
    for (int a = 0; a < ns; ++a)
    {
      const int node = element * ns + a;
      const double xi = 2.0 * (view.volume_node_x[node] * n - ex) - 1.0;
      const double eta = 2.0 * (view.volume_node_y[node] * n - ey) - 1.0;
      const double old_values[7]{
          1.0 + 0.008 * parity * xi + 0.004 * eta,
          0.09 + 0.006 * eta,
          -0.04 + 0.004 * xi,
          2.505 + 0.001 * parity * xi,
          0.018 + 0.003 * xi - 0.001 * eta,
          0.002 + 0.001 * eta,
          -0.001 + 0.0005 * xi};
      const double current_values[7]{
          1.0 + 0.010 * parity * xi + 0.005 * eta,
          0.10 + 0.010 * eta,
          -0.05 + 0.005 * xi,
          2.510 + 0.002 * parity * xi,
          0.020 + 0.004 * xi - 0.002 * eta,
          0.003 + 0.001 * eta,
          -0.002 + 0.001 * xi};
      for (int component = 0; component < 7; ++component)
      {
        view.slot_interior[0][InteriorIndex(element, component, a)] =
            old_values[component];
        view.slot_interior[1][InteriorIndex(element, component, a)] =
            current_values[component];
      }
    }
  }
  for (int i = 0; i < view.true_trace_unknowns; ++i)
  {
    view.slot_trace[0][i] = nan("");
    view.slot_trace[1][i] = nan("");
  }
  for (int element = 0; element < view.elements; ++element)
    for (int local_face = 0; local_face < 4; ++local_face)
      for (int a = 0; a < face_nodes; ++a)
      {
        const double s = view.trace_node_coordinate[
            (element * 4 + local_face) * face_nodes + a];
        for (int component = 0; component < scalar::trace_components;
             ++component)
        {
          double old_value = 0.0;
          double value = 0.0;
          if (component == 0)
          {
            old_value = 1.0 + 0.001 * s;
            value = 1.0 + 0.002 * s;
          }
          else if (component == 1)
          {
            old_value = 0.09 + 0.002 * s;
            value = 0.095 + 0.003 * s;
          }
          else if (component == 2)
          {
            old_value = -0.04 + 0.001 * s;
            value = -0.045 + 0.002 * s;
          }
          else if (component == 3)
          {
            old_value = 2.505 + 0.0005 * s;
            value = 2.508 + 0.001 * s;
          }
          else
          {
            old_value = 0.018 + 0.001 * s;
            value = 0.020 + 0.003 * s;
          }
          const int local_row = TraceIndex(
              component, local_face * face_nodes + a);
          const int map = element * nt + local_row;
          const int true_dof = view.local_trace_to_true[map];
          const double sign = static_cast<double>(view.local_trace_sign[map]);
          view.slot_trace[0][true_dof] = sign * old_value;
          view.slot_trace[1][true_dof] = sign * value;
        }
      }
}

__device__ void ZeroAssembly(scalar_detail::DeviceView& view)
{
  const std::size_t elements = static_cast<std::size_t>(view.elements);
  for (std::size_t i = 0; i < elements * ni * ni; ++i) { view.a[i] = 0.0; }
  for (std::size_t i = 0; i < elements * ni * nt; ++i) { view.b[i] = 0.0; }
  for (std::size_t i = 0; i < elements * nt * ni; ++i) { view.c[i] = 0.0; }
  for (std::size_t i = 0; i < elements * nt * nt; ++i) { view.d[i] = 0.0; }
  for (std::size_t i = 0; i < elements * ni; ++i)
  {
    view.local_residual[i] = 0.0;
  }
  for (std::size_t i = 0; i < elements * nt; ++i)
  {
    view.trace_residual[i] = 0.0;
  }
  for (int i = 0; i < view.full_unknowns; ++i) { view.full_residual[i] = 0.0; }
  if (view.assemble_dense_jacobian != 0)
  {
    const std::size_t full = static_cast<std::size_t>(view.full_unknowns);
    for (std::size_t i = 0; i < full * full; ++i)
    {
      view.full_jacobian[i] = 0.0;
    }
  }
  for (int i = 0; i < 16; ++i) { view.reduction[i] = 0.0; }
  view.reduction[1] = DBL_MAX;
  view.reduction[2] = DBL_MAX;
  view.reduction[3] = DBL_MAX;
}

__device__ void AssembleSystem(scalar_detail::DeviceView& view)
{
  ZeroAssembly(view);
  const double* old_state = view.slot_interior[view.committed_slot];
  const double* old_trace = view.slot_trace[view.committed_slot];
  double* current_state = CurrentInterior(view);
  double* current_trace = CurrentTrace(view);
  const int global_interior = view.elements * ni;
  for (int element = 0; element < view.elements; ++element)
  {
    double* block_a = view.a + static_cast<std::size_t>(element) * ni * ni;
    double* block_b = view.b + static_cast<std::size_t>(element) * ni * nt;
    double* block_c = view.c + static_cast<std::size_t>(element) * nt * ni;
    double* block_d = view.d + static_cast<std::size_t>(element) * nt * nt;
    double* local_r = view.local_residual + element * ni;
    double* trace_r = view.trace_residual + element * nt;
    for (int qpoint = 0; qpoint < view.volume_quadrature_points; ++qpoint)
    {
      const int qindex = element * view.volume_quadrature_points + qpoint;
      const double* shape = view.volume_shape + qindex * ns;
      const double* derivative =
          view.volume_derivative + qindex * ns * scalar::dimension;
      double conservative[4]{};
      double previous[4]{};
      double conservative_derivative[4][2]{};
      for (int component = 0; component < 4; ++component)
      {
        conservative[component] = EvaluateInterior(
            current_state, element, component, shape);
        previous[component] = EvaluateInterior(
            old_state, element, component, shape);
        for (int direction = 0; direction < 2; ++direction)
        {
          conservative_derivative[component][direction] = EvaluateDerivative(
              current_state, element, component, derivative, direction);
        }
      }
      if (!FiniteConservativeState(conservative) ||
          !FiniteConservativeState(previous))
      {
        *view.status |= scalar_detail::device_nonfinite;
        return;
      }
      if (!Admissible(conservative) || !Admissible(previous))
      {
        *view.status |= scalar_detail::device_admissibility;
        return;
      }
      const double sigma = EvaluateInterior(
          current_state, element, 4, shape);
      const double z[2]{EvaluateInterior(current_state, element, 5, shape),
                        EvaluateInterior(current_state, element, 6, shape)};
      double flux[2][4]{};
      double flux_jacobian[2][4][5]{};
      const double nx[2]{1.0, 0.0};
      const double ny[2]{0.0, 1.0};
      if (!EulerSigmaFlux(conservative, sigma, nx, flux[0],
                          flux_jacobian[0]) ||
          !EulerSigmaFlux(conservative, sigma, ny, flux[1],
                          flux_jacobian[1]))
      {
        *view.status |= scalar_detail::device_nonfinite;
        return;
      }
      const double rho = conservative[0];
      const double mx = conservative[1];
      const double my = conservative[2];
      double velocity_gradient[2][2]{};
      for (int direction = 0; direction < 2; ++direction)
      {
        velocity_gradient[0][direction] =
            conservative_derivative[1][direction] / rho -
            mx * conservative_derivative[0][direction] / (rho * rho);
        velocity_gradient[1][direction] =
            conservative_derivative[2][direction] / rho -
            my * conservative_derivative[0][direction] / (rho * rho);
      }
      const double trace_gradient =
          velocity_gradient[0][0] + velocity_gradient[1][1];
      const double invariant =
          velocity_gradient[0][0] * velocity_gradient[0][0] +
          2.0 * velocity_gradient[0][1] * velocity_gradient[1][0] +
          velocity_gradient[1][1] * velocity_gradient[1][1] +
          trace_gradient * trace_gradient;
      double dq_dgradient[2][2]{};
      for (int velocity = 0; velocity < 2; ++velocity)
        for (int direction = 0; direction < 2; ++direction)
        {
          dq_dgradient[velocity][direction] =
              2.0 * velocity_gradient[direction][velocity] +
              (velocity == direction ? 2.0 * trace_gradient : 0.0);
        }
      const double weight = view.volume_weight[qindex];
      if (!DeviceFinite(weight) || !(weight > 0.0))
      {
        *view.status |= scalar_detail::device_nonfinite;
        return;
      }
      for (int row_scalar = 0; row_scalar < ns; ++row_scalar)
      {
        for (int equation = 0; equation < 4; ++equation)
        {
          const int row = LocalIndex(equation, row_scalar);
          local_r[row] += weight *
              (shape[row_scalar] *
                   (conservative[equation] - previous[equation]) / view.dt -
               derivative[row_scalar * 2] * flux[0][equation] -
               derivative[row_scalar * 2 + 1] * flux[1][equation]);
          for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
            for (int variable = 0; variable < 5; ++variable)
            {
              const int column = LocalIndex(variable, column_scalar);
              double value = variable == equation ?
                  shape[row_scalar] * shape[column_scalar] / view.dt : 0.0;
              value -= derivative[row_scalar * 2] *
                  flux_jacobian[0][equation][variable] *
                  shape[column_scalar];
              value -= derivative[row_scalar * 2 + 1] *
                  flux_jacobian[1][equation][variable] *
                  shape[column_scalar];
              block_a[row * ni + column] += weight * value;
            }
        }
        for (int direction = 0; direction < 2; ++direction)
        {
          const int row = LocalIndex(4 + direction, row_scalar);
          local_r[row] += weight *
              (shape[row_scalar] * rho * z[direction] +
               derivative[row_scalar * 2 + direction] * sigma);
          for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
          {
            block_a[row * ni + LocalIndex(0, column_scalar)] +=
                weight * shape[row_scalar] * z[direction] *
                shape[column_scalar];
            block_a[row * ni + LocalIndex(4, column_scalar)] +=
                weight * derivative[row_scalar * 2 + direction] *
                shape[column_scalar];
            block_a[row * ni + LocalIndex(5 + direction, column_scalar)] +=
                weight * shape[row_scalar] * rho * shape[column_scalar];
          }
        }
        const int row_sigma = LocalIndex(6, row_scalar);
        local_r[row_sigma] += weight *
            (shape[row_scalar] * sigma / rho +
             alpha_value * derivative[row_scalar * 2] * z[0] +
             alpha_value * derivative[row_scalar * 2 + 1] * z[1] -
             alpha_value * shape[row_scalar] * invariant);
        for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
        {
          block_a[row_sigma * ni + LocalIndex(4, column_scalar)] +=
              weight * shape[row_scalar] * shape[column_scalar] / rho;
          for (int direction = 0; direction < 2; ++direction)
          {
            block_a[row_sigma * ni +
                    LocalIndex(5 + direction, column_scalar)] +=
                alpha_value * weight *
                derivative[row_scalar * 2 + direction] *
                shape[column_scalar];
          }
          for (int variable = 0; variable < 4; ++variable)
          {
            double dq = 0.0;
            if (variable < 3)
            {
              for (int velocity = 0; velocity < 2; ++velocity)
                for (int direction = 0; direction < 2; ++direction)
                {
                  const double delta_rho =
                      variable == 0 ? shape[column_scalar] : 0.0;
                  const double delta_drho = variable == 0 ?
                      derivative[column_scalar * 2 + direction] : 0.0;
                  const double delta_m = variable == 1 + velocity ?
                      shape[column_scalar] : 0.0;
                  const double delta_dm = variable == 1 + velocity ?
                      derivative[column_scalar * 2 + direction] : 0.0;
                  const double momentum = velocity == 0 ? mx : my;
                  const double momentum_derivative =
                      conservative_derivative[1 + velocity][direction];
                  const double rho_derivative =
                      conservative_derivative[0][direction];
                  const double delta_gradient =
                      delta_dm / rho -
                      momentum_derivative * delta_rho / (rho * rho) -
                      delta_m * rho_derivative / (rho * rho) -
                      momentum * delta_drho / (rho * rho) +
                      2.0 * momentum * rho_derivative * delta_rho /
                          (rho * rho * rho);
                  dq += dq_dgradient[velocity][direction] * delta_gradient;
                }
            }
            double value = -alpha_value * shape[row_scalar] * dq;
            if (variable == 0)
            {
              value -= shape[row_scalar] * sigma * shape[column_scalar] /
                  (rho * rho);
            }
            block_a[row_sigma * ni + LocalIndex(variable, column_scalar)] +=
                weight * value;
          }
        }
      }
    }

    for (int local_face = 0; local_face < 4; ++local_face)
      for (int qpoint = 0; qpoint < view.face_quadrature_points; ++qpoint)
      {
        const int qindex =
            (element * 4 + local_face) * view.face_quadrature_points + qpoint;
        const double* interior_shape =
            view.face_interior_shape + qindex * ns;
        const double* oriented_shape =
            view.face_oriented_shape + qindex * face_nodes;
        const double* normal = view.face_normal + qindex * 2;
        double conservative[4]{};
        double previous[4]{};
        double trace_state[4]{};
        double previous_trace[4]{};
        for (int component = 0; component < 4; ++component)
        {
          conservative[component] = EvaluateInterior(
              current_state, element, component, interior_shape);
          previous[component] = EvaluateInterior(
              old_state, element, component, interior_shape);
          for (int a = 0; a < face_nodes; ++a)
          {
            const int local_row = TraceIndex(
                component, local_face * face_nodes + a);
            trace_state[component] += TraceValue(
                view, current_trace, element, local_row) * oriented_shape[a];
            previous_trace[component] += TraceValue(
                view, old_trace, element, local_row) * oriented_shape[a];
          }
        }
        if (!FiniteConservativeState(conservative) ||
            !FiniteConservativeState(previous) ||
            !FiniteConservativeState(trace_state) ||
            !FiniteConservativeState(previous_trace))
        {
          *view.status |= scalar_detail::device_nonfinite;
          return;
        }
        if (!Admissible(conservative) || !Admissible(previous) ||
            !Admissible(trace_state) || !Admissible(previous_trace))
        {
          *view.status |= scalar_detail::device_admissibility;
          return;
        }
        const double sigma = EvaluateInterior(
            current_state, element, 4, interior_shape);
        const double z[2]{
            EvaluateInterior(current_state, element, 5, interior_shape),
            EvaluateInterior(current_state, element, 6, interior_shape)};
        double sigmahat = 0.0;
        for (int a = 0; a < face_nodes; ++a)
        {
          sigmahat += TraceValue(view, current_trace, element,
              TraceIndex(4, local_face * face_nodes + a)) * oriented_shape[a];
        }
        double physical_flux[4]{};
        double physical_jacobian[4][5]{};
        if (!EulerSigmaFlux(trace_state, sigmahat, normal, physical_flux,
                            physical_jacobian))
        {
          *view.status |= scalar_detail::device_nonfinite;
          return;
        }
        double numerical_flux[4]{};
        for (int equation = 0; equation < 4; ++equation)
        {
          numerical_flux[equation] = physical_flux[equation] +
              state_tau * (conservative[equation] - trace_state[equation]);
        }
        const double rhohat = trace_state[0];
        const double tau = 4.0 /
            (rhohat * view.face_height[element * 4 + local_face]);
        const double jump = sigma - sigmahat;
        const double auxiliary_flux = z[0] * normal[0] + z[1] * normal[1] -
            tau * jump;
        const double weight = view.face_weight[qindex];
        if (!(tau > 0.0) || !DeviceFinite(tau) || !DeviceFinite(weight) ||
            !(weight > 0.0))
        {
          *view.status |= scalar_detail::device_nonfinite;
          return;
        }
        for (int row_scalar = 0; row_scalar < ns; ++row_scalar)
        {
          for (int equation = 0; equation < 4; ++equation)
          {
            const int row = LocalIndex(equation, row_scalar);
            local_r[row] += weight * interior_shape[row_scalar] *
                numerical_flux[equation];
            for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
            {
              block_a[row * ni + LocalIndex(equation, column_scalar)] +=
                  weight * interior_shape[row_scalar] * state_tau *
                  interior_shape[column_scalar];
            }
            for (int trace_variable = 0; trace_variable < 5;
                 ++trace_variable)
              for (int column_scalar = 0; column_scalar < face_nodes;
                   ++column_scalar)
              {
                const int column = TraceIndex(
                    trace_variable, local_face * face_nodes + column_scalar);
                const double derivative_value =
                    physical_jacobian[equation][trace_variable] -
                    (trace_variable == equation ? state_tau : 0.0);
                block_b[row * nt + column] += weight *
                    interior_shape[row_scalar] * derivative_value *
                    oriented_shape[column_scalar];
              }
          }
          for (int direction = 0; direction < 2; ++direction)
          {
            const int row = LocalIndex(4 + direction, row_scalar);
            local_r[row] -= weight * interior_shape[row_scalar] *
                normal[direction] * sigmahat;
            for (int column_scalar = 0; column_scalar < face_nodes;
                 ++column_scalar)
            {
              const int column = TraceIndex(
                  4, local_face * face_nodes + column_scalar);
              block_b[row * nt + column] -= weight *
                  interior_shape[row_scalar] * normal[direction] *
                  oriented_shape[column_scalar];
            }
          }
          const int row_sigma = LocalIndex(6, row_scalar);
          local_r[row_sigma] -= alpha_value * weight *
              interior_shape[row_scalar] * auxiliary_flux;
          for (int column_scalar = 0; column_scalar < ns; ++column_scalar)
          {
            block_a[row_sigma * ni + LocalIndex(4, column_scalar)] +=
                alpha_value * weight * interior_shape[row_scalar] * tau *
                interior_shape[column_scalar];
            for (int direction = 0; direction < 2; ++direction)
            {
              block_a[row_sigma * ni +
                      LocalIndex(5 + direction, column_scalar)] -=
                  alpha_value * weight * interior_shape[row_scalar] *
                  normal[direction] * interior_shape[column_scalar];
            }
          }
          for (int column_scalar = 0; column_scalar < face_nodes;
               ++column_scalar)
          {
            block_b[row_sigma * nt + TraceIndex(
                4, local_face * face_nodes + column_scalar)] -=
                alpha_value * weight * interior_shape[row_scalar] * tau *
                oriented_shape[column_scalar];
            block_b[row_sigma * nt + TraceIndex(
                0, local_face * face_nodes + column_scalar)] -=
                alpha_value * weight * interior_shape[row_scalar] * tau *
                jump / rhohat * oriented_shape[column_scalar];
          }
        }
        for (int row_component = 0; row_component < 5; ++row_component)
          for (int row_scalar = 0; row_scalar < face_nodes; ++row_scalar)
          {
            const int row = TraceIndex(
                row_component, local_face * face_nodes + row_scalar);
            if (row_component < 4)
            {
              trace_r[row] += weight * oriented_shape[row_scalar] *
                  numerical_flux[row_component];
              for (int column_scalar = 0; column_scalar < ns;
                   ++column_scalar)
              {
                block_c[row * ni +
                        LocalIndex(row_component, column_scalar)] +=
                    weight * oriented_shape[row_scalar] * state_tau *
                    interior_shape[column_scalar];
              }
              for (int trace_variable = 0; trace_variable < 5;
                   ++trace_variable)
                for (int column_scalar = 0; column_scalar < face_nodes;
                     ++column_scalar)
                {
                  const int column = TraceIndex(
                      trace_variable, local_face * face_nodes + column_scalar);
                  const double derivative_value =
                      physical_jacobian[row_component][trace_variable] -
                      (trace_variable == row_component ? state_tau : 0.0);
                  block_d[row * nt + column] += weight *
                      oriented_shape[row_scalar] * derivative_value *
                      oriented_shape[column_scalar];
                }
            }
            else
            {
              trace_r[row] += alpha_value * weight *
                  oriented_shape[row_scalar] * auxiliary_flux;
              for (int column_scalar = 0; column_scalar < ns;
                   ++column_scalar)
              {
                block_c[row * ni + LocalIndex(4, column_scalar)] -=
                    alpha_value * weight * oriented_shape[row_scalar] * tau *
                    interior_shape[column_scalar];
                for (int direction = 0; direction < 2; ++direction)
                {
                  block_c[row * ni +
                          LocalIndex(5 + direction, column_scalar)] +=
                      alpha_value * weight * oriented_shape[row_scalar] *
                      normal[direction] * interior_shape[column_scalar];
                }
              }
              for (int column_scalar = 0; column_scalar < face_nodes;
                   ++column_scalar)
              {
                block_d[row * nt + TraceIndex(
                    4, local_face * face_nodes + column_scalar)] +=
                    alpha_value * weight * oriented_shape[row_scalar] * tau *
                    oriented_shape[column_scalar];
                block_d[row * nt + TraceIndex(
                    0, local_face * face_nodes + column_scalar)] +=
                    alpha_value * weight * oriented_shape[row_scalar] * tau *
                    jump / rhohat * oriented_shape[column_scalar];
              }
            }
          }
      }

    const int local_base = element * ni;
    for (int row = 0; row < ni; ++row)
    {
      view.full_residual[local_base + row] = local_r[row];
      for (int column = 0;
           view.assemble_dense_jacobian != 0 && column < ni; ++column)
      {
        view.full_jacobian[
            static_cast<std::size_t>(local_base + row) * view.full_unknowns +
            local_base + column] = block_a[row * ni + column];
      }
      for (int column = 0; column < nt; ++column)
      {
        const int map = element * nt + column;
        const int true_column = global_interior +
            view.local_trace_to_true[map];
        if (view.assemble_dense_jacobian != 0)
        {
          view.full_jacobian[
              static_cast<std::size_t>(local_base + row) *
                  view.full_unknowns + true_column] +=
              static_cast<double>(view.local_trace_sign[map]) *
              block_b[row * nt + column];
        }
      }
    }
    for (int row = 0; row < nt; ++row)
    {
      const int row_map = element * nt + row;
      const int true_row = global_interior +
          view.local_trace_to_true[row_map];
      const double row_sign =
          static_cast<double>(view.local_trace_sign[row_map]);
      view.full_residual[true_row] += row_sign * trace_r[row];
      for (int column = 0;
           view.assemble_dense_jacobian != 0 && column < ni; ++column)
      {
        view.full_jacobian[
            static_cast<std::size_t>(true_row) * view.full_unknowns +
            local_base + column] += row_sign * block_c[row * ni + column];
      }
      for (int column = 0; column < nt; ++column)
      {
        const int column_map = element * nt + column;
        const int true_column = global_interior +
            view.local_trace_to_true[column_map];
        if (view.assemble_dense_jacobian != 0)
        {
          view.full_jacobian[
              static_cast<std::size_t>(true_row) * view.full_unknowns +
              true_column] += row_sign *
              static_cast<double>(view.local_trace_sign[column_map]) *
              block_d[row * nt + column];
        }
      }
    }
  }
  view.reduction[0] = VectorNorm(view.full_residual, view.full_unknowns);
  double auxiliary_residual_squared = 0.0;
  for (int element = 0; element < view.elements; ++element)
  {
    const int local_base = element * ni;
    for (int row = 4 * ns; row < ni; ++row)
    {
      const double residual = view.full_residual[local_base + row];
      auxiliary_residual_squared += residual * residual;
    }
  }
  const int auxiliary_trace_begin = global_interior +
      scalar::state_components * view.scalar_true_trace_nodes;
  for (int row = auxiliary_trace_begin; row < view.full_unknowns; ++row)
  {
    const double residual = view.full_residual[row];
    auxiliary_residual_squared += residual * residual;
  }
  view.reduction[5] = sqrt(auxiliary_residual_squared);
  if (!DeviceFinite(view.reduction[0]) || !DeviceFinite(view.reduction[5]))
  {
    *view.status |= scalar_detail::device_nonfinite;
  }
}

__device__ void ApplyAuxiliaryIdentityRows(scalar_detail::DeviceView& view)
{
  constexpr int physical_interior = scalar::state_components * ns;
  constexpr int physical_trace = scalar::state_components * nf;
  for (int element = 0; element < view.elements; ++element)
  {
    double* a = view.a + static_cast<std::size_t>(element) * ni * ni;
    double* b = view.b + static_cast<std::size_t>(element) * ni * nt;
    double* c = view.c + static_cast<std::size_t>(element) * nt * ni;
    double* d = view.d + static_cast<std::size_t>(element) * nt * nt;
    double* local_r = view.local_residual + element * ni;
    double* trace_r = view.trace_residual + element * nt;
    for (int row = 0; row < physical_interior; ++row)
    {
      for (int column = 0; column < ni; ++column)
      {
        a[row * ni + column] = row == column ? 1.0 : 0.0;
      }
      for (int column = 0; column < nt; ++column)
      {
        b[row * nt + column] = 0.0;
      }
      local_r[row] = 0.0;
    }
    for (int row = physical_interior; row < ni; ++row)
    {
      for (int column = 0; column < physical_interior; ++column)
      {
        a[row * ni + column] = 0.0;
      }
      for (int column = 0; column < physical_trace; ++column)
      {
        b[row * nt + column] = 0.0;
      }
    }
    for (int row = 0; row < physical_trace; ++row)
    {
      for (int column = 0; column < ni; ++column)
      {
        c[row * ni + column] = 0.0;
      }
      for (int column = 0; column < nt; ++column)
      {
        d[row * nt + column] = row == column ? 1.0 : 0.0;
      }
      trace_r[row] = 0.0;
    }
    for (int row = physical_trace; row < nt; ++row)
    {
      for (int column = 0; column < physical_interior; ++column)
      {
        c[row * ni + column] = 0.0;
      }
      for (int column = 0; column < physical_trace; ++column)
      {
        d[row * nt + column] = 0.0;
      }
    }
  }
}

__device__ void CondenseAndScatter(scalar_detail::DeviceView& view)
{
  for (int i = 0; i < view.true_matrix_diag_nonzeros; ++i)
  {
    view.true_matrix_data[i] = 0.0;
  }
  for (int i = 0; i < view.true_matrix_offd_nonzeros; ++i)
  {
    view.true_matrix_offd_data[i] = 0.0;
  }
  for (int i = 0; i < view.owned_true_trace_unknowns; ++i)
  {
    view.true_rhs[i] = 0.0;
  }
  double maximum_residual = 0.0;
  double paired_residual = 0.0;
  double paired_rhs_norm = 0.0;
  double maximum_contract_ratio = -1.0;
  const auto record_application = [&](const double residual,
                                      const double rhs_norm)
  {
    const double ratio = residual / (1.0e-12 + 1.0e-11 * rhs_norm);
    if (ratio > maximum_contract_ratio)
    {
      maximum_contract_ratio = ratio;
      paired_residual = residual;
      paired_rhs_norm = rhs_norm;
    }
  };
  for (int element = 0; element < view.elements; ++element)
  {
    const double* a = view.a + static_cast<std::size_t>(element) * ni * ni;
    const double* b = view.b + static_cast<std::size_t>(element) * ni * nt;
    const double* c = view.c + static_cast<std::size_t>(element) * nt * ni;
    const double* d = view.d + static_cast<std::size_t>(element) * nt * nt;
    const double* local_r = view.local_residual + element * ni;
    const double* trace_r = view.trace_residual + element * nt;
    double* lu = view.lu + static_cast<std::size_t>(element) * ni * ni;
    double* inverse_r = view.inverse_residual + element * ni;
    double* inverse_b = view.inverse_b +
        static_cast<std::size_t>(element) * ni * nt;
    double* schur = view.schur + static_cast<std::size_t>(element) * nt * nt;
    double* rhs = view.condensed_rhs + element * nt;
    int* rows = view.row_pivots + element * ni;
    for (int i = 0; i < ni * ni; ++i) { lu[i] = a[i]; }
    if (!FactorPartialPivot(lu, rows) ||
        !SolvePartialPivot(lu, rows, local_r, inverse_r))
    {
      *view.status |= scalar_detail::device_singular_local_block;
      return;
    }
    double residual = ApplicationResidual(a, inverse_r, local_r);
    const double local_r_norm = VectorNorm(local_r, ni);
    maximum_residual = fmax(maximum_residual, residual);
    record_application(residual, local_r_norm);
    if (residual > local_absolute_tolerance +
            local_relative_tolerance * local_r_norm)
    {
      *view.status |= scalar_detail::device_local_application;
      return;
    }
    double column_rhs[ni];
    double column_solution[ni];
    for (int trace_column = 0; trace_column < nt; ++trace_column)
    {
      for (int row = 0; row < ni; ++row)
      {
        column_rhs[row] = b[row * nt + trace_column];
      }
      if (!SolvePartialPivot(lu, rows, column_rhs, column_solution))
      {
        *view.status |= scalar_detail::device_singular_local_block;
        return;
      }
      residual = ApplicationResidual(a, column_solution, column_rhs);
      const double column_rhs_norm = VectorNorm(column_rhs, ni);
      maximum_residual = fmax(maximum_residual, residual);
      record_application(residual, column_rhs_norm);
      if (residual > local_absolute_tolerance +
              local_relative_tolerance * column_rhs_norm)
      {
        *view.status |= scalar_detail::device_local_application;
        return;
      }
      for (int row = 0; row < ni; ++row)
      {
        inverse_b[row * nt + trace_column] = column_solution[row];
      }
    }
    for (int row = 0; row < nt; ++row)
    {
      double c_inverse_r = 0.0;
      for (int inner = 0; inner < ni; ++inner)
      {
        c_inverse_r += c[row * ni + inner] * inverse_r[inner];
      }
      rhs[row] = -trace_r[row] + c_inverse_r;
      for (int column = 0; column < nt; ++column)
      {
        double value = d[row * nt + column];
        for (int inner = 0; inner < ni; ++inner)
        {
          value -= c[row * ni + inner] * inverse_b[inner * nt + column];
        }
        schur[row * nt + column] = value;
      }
    }
  }
  const auto matrix_value = [&](const int source)
  {
    const int element = source / (nt * nt);
    const int local = source % (nt * nt);
    const int row = local / nt;
    const int column = local % nt;
    const int row_map = element * nt + row;
    const int column_map = element * nt + column;
    return static_cast<double>(view.local_trace_sign[row_map] *
                               view.local_trace_sign[column_map]) *
        view.schur[source];
  };
  const auto vector_value = [&](const int source)
  {
    const int element = source / nt;
    const int row = source % nt;
    return static_cast<double>(view.local_trace_sign[source]) *
        view.condensed_rhs[element * nt + row];
  };
  for (int item = 0; item < view.matrix_owned_count; ++item)
  {
    const int section = view.matrix_owned_section[item];
    const int target = view.matrix_owned_target[item];
    const double value = matrix_value(view.matrix_owned_source[item]);
    if (section == 0 && target >= 0 &&
        target < view.true_matrix_diag_nonzeros)
      view.true_matrix_data[target] += value;
    else if (section == 1 && target >= 0 &&
             target < view.true_matrix_offd_nonzeros)
      view.true_matrix_offd_data[target] += value;
    else
      *view.status |= scalar_detail::device_invalid_layout;
  }
  for (int item = 0; item < view.matrix_send_count; ++item)
    view.remote_matrix_send[item] =
        matrix_value(view.matrix_send_source[item]);
  for (int item = 0; item < view.vector_owned_count; ++item)
  {
    const int target = view.vector_owned_target[item];
    if (target < 0 || target >= view.owned_true_trace_unknowns)
      *view.status |= scalar_detail::device_invalid_layout;
    else
      view.true_rhs[target] +=
          vector_value(view.vector_owned_source[item]);
  }
  for (int item = 0; item < view.vector_send_count; ++item)
    view.remote_vector_send[item] =
        vector_value(view.vector_send_source[item]);
  view.reduction[4] = maximum_residual;
  view.reduction[6] = paired_residual;
  view.reduction[7] = paired_rhs_norm;
}

__device__ void ScatterRemoteSystem(scalar_detail::DeviceView& view)
{
  for (int item = 0; item < view.matrix_receive_count; ++item)
  {
    const int section = view.matrix_receive_section[item];
    const int target = view.matrix_receive_target[item];
    if (section == 0 && target >= 0 &&
        target < view.true_matrix_diag_nonzeros)
      view.true_matrix_data[target] += view.remote_matrix_receive[item];
    else if (section == 1 && target >= 0 &&
             target < view.true_matrix_offd_nonzeros)
      view.true_matrix_offd_data[target] += view.remote_matrix_receive[item];
    else
      *view.status |= scalar_detail::device_invalid_layout;
  }
  for (int item = 0; item < view.vector_receive_count; ++item)
  {
    const int target = view.vector_receive_target[item];
    if (target < 0 || target >= view.owned_true_trace_unknowns)
      *view.status |= scalar_detail::device_invalid_layout;
    else
      view.true_rhs[target] += view.remote_vector_receive[item];
  }
}

__device__ void FixedJacobianAction(scalar_detail::DeviceView& view)
{
  const int local_full = scalar::persistent_interior_unknowns +
      view.owned_true_trace_unknowns;
  for (int item = 0; item < local_full; ++item)
  {
    view.evidence_action_input[item] = 0.0;
    view.evidence_action_result[item] = 0.0;
  }
  for (int element = 0; element < view.elements; ++element)
  {
    double center_x = 0.0;
    double center_y = 0.0;
    for (int node = 0; node < ns; ++node)
    {
      center_x += view.volume_node_x[element * ns + node];
      center_y += view.volume_node_y[element * ns + node];
    }
    center_x /= static_cast<double>(ns);
    center_y /= static_cast<double>(ns);
    int ex = static_cast<int>(floor(8.0 * center_x));
    int ey = static_cast<int>(floor(8.0 * center_y));
    ex = ex < 0 ? 0 : (ex > 7 ? 7 : ex);
    ey = ey < 0 ? 0 : (ey > 7 ? 7 : ey);
    const int global_element = ey * 8 + ex;
    const double* a = view.a + static_cast<std::size_t>(element) * ni * ni;
    const double* b = view.b + static_cast<std::size_t>(element) * ni * nt;
    const double* c = view.c + static_cast<std::size_t>(element) * nt * ni;
    const double* d = view.d + static_cast<std::size_t>(element) * nt * nt;
    double trace_action[nt];
    for (int column = 0; column < nt; ++column)
    {
      const int map = element * nt + column;
      const double global_value = -0.015625 +
          static_cast<double>(view.local_trace_to_true[map] + 1) / 131072.0;
      trace_action[column] =
          static_cast<double>(view.local_trace_sign[map]) * global_value;
    }
    for (int row = 0; row < ni; ++row)
    {
      const int packed = element * ni + row;
      view.evidence_action_input[packed] = 0.03125 +
          static_cast<double>(global_element * ni + row + 1) / 65536.0;
      double value = 0.0;
      for (int column = 0; column < ni; ++column)
      {
        const double input = 0.03125 +
            static_cast<double>(global_element * ni + column + 1) / 65536.0;
        value += a[row * ni + column] * input;
      }
      for (int column = 0; column < nt; ++column)
      {
        value += b[row * nt + column] * trace_action[column];
      }
      view.evidence_action_result[packed] = value;
    }
    for (int row = 0; row < nt; ++row)
    {
      double value = 0.0;
      for (int column = 0; column < ni; ++column)
      {
        const double input = 0.03125 +
            static_cast<double>(global_element * ni + column + 1) / 65536.0;
        value += c[row * ni + column] * input;
      }
      for (int column = 0; column < nt; ++column)
      {
        value += d[row * nt + column] * trace_action[column];
      }
      view.evidence_trace_action[element * nt + row] = value;
    }
  }
  for (int row = 0; row < view.owned_true_trace_unknowns; ++row)
  {
    view.evidence_action_input[scalar::persistent_interior_unknowns + row] =
        -0.015625 + static_cast<double>(view.owned_true_begin + row + 1) /
            131072.0;
  }
  for (int item = 0; item < view.vector_owned_count; ++item)
  {
    const int source = view.vector_owned_source[item];
    const int target = view.vector_owned_target[item];
    view.evidence_action_result[
        scalar::persistent_interior_unknowns + target] +=
        static_cast<double>(view.local_trace_sign[source]) *
        view.evidence_trace_action[source];
  }
  for (int item = 0; item < view.vector_send_count; ++item)
  {
    const int source = view.vector_send_source[item];
    view.remote_vector_send[item] =
        static_cast<double>(view.local_trace_sign[source]) *
        view.evidence_trace_action[source];
  }
}

__device__ void ScatterRemoteFixedAction(scalar_detail::DeviceView& view)
{
  for (int item = 0; item < view.vector_receive_count; ++item)
  {
    view.evidence_action_result[
        scalar::persistent_interior_unknowns +
        view.vector_receive_target[item]] += view.remote_vector_receive[item];
  }
}

__device__ void GatherTraceResidual(scalar_detail::DeviceView& view)
{
  for (int row = 0; row < view.owned_true_trace_unknowns; ++row)
  {
    view.true_rhs[row] = 0.0;
  }
  for (int item = 0; item < view.vector_owned_count; ++item)
  {
    const int source = view.vector_owned_source[item];
    const int target = view.vector_owned_target[item];
    const int map = source;
    view.true_rhs[target] += static_cast<double>(view.local_trace_sign[map]) *
        view.trace_residual[source];
  }
  for (int item = 0; item < view.vector_send_count; ++item)
  {
    const int source = view.vector_send_source[item];
    view.remote_vector_send[item] =
        static_cast<double>(view.local_trace_sign[source]) *
        view.trace_residual[source];
  }
}

__device__ void ScatterRemoteTraceResidual(scalar_detail::DeviceView& view)
{
  for (int item = 0; item < view.vector_receive_count; ++item)
  {
    view.true_rhs[view.vector_receive_target[item]] +=
        view.remote_vector_receive[item];
  }
  double full_square = 0.0;
  double auxiliary_square = 0.0;
  for (int item = 0; item < view.elements * ni; ++item)
  {
    const double residual = view.local_residual[item];
    full_square += residual * residual;
    if (item % ni >= scalar::state_components * ns)
    {
      auxiliary_square += residual * residual;
    }
  }
  const int local_scalar =
      view.owned_true_trace_unknowns / scalar::trace_components;
  for (int item = 0; item < view.owned_true_trace_unknowns; ++item)
  {
    const double residual = view.true_rhs[item];
    full_square += residual * residual;
    if (item >= scalar::state_components * local_scalar)
    {
      auxiliary_square += residual * residual;
    }
  }
  view.reduction[0] = full_square;
  view.reduction[5] = auxiliary_square;
  if (!DeviceFinite(full_square) || !DeviceFinite(auxiliary_square))
  {
    *view.status |= scalar_detail::device_nonfinite;
  }
}

__device__ void RecoverDirection(scalar_detail::DeviceView& view)
{
  for (int i = 0; i < view.true_trace_unknowns; ++i)
  {
    view.direction_trace[i] = view.true_solution[i];
  }
  double maximum_residual = view.reduction[4];
  double paired_residual = view.reduction[6];
  double paired_rhs_norm = view.reduction[7];
  double maximum_contract_ratio = paired_residual /
      (1.0e-12 + 1.0e-11 * paired_rhs_norm);
  for (int element = 0; element < view.elements; ++element)
  {
    const double* a = view.a + static_cast<std::size_t>(element) * ni * ni;
    const double* b = view.b + static_cast<std::size_t>(element) * ni * nt;
    const double* local_r = view.local_residual + element * ni;
    const double* lu = view.lu + static_cast<std::size_t>(element) * ni * ni;
    const int* rows = view.row_pivots + element * ni;
    double* recovery_rhs = view.recovery_rhs + element * ni;
    double* recovered = view.recovered + element * ni;
    for (int row = 0; row < ni; ++row)
    {
      recovery_rhs[row] = local_r[row];
      for (int column = 0; column < nt; ++column)
      {
        const int map = element * nt + column;
        recovery_rhs[row] += b[row * nt + column] *
            static_cast<double>(view.local_trace_sign[map]) *
            view.true_solution[view.local_trace_to_true[map]];
      }
    }
    if (!SolvePartialPivot(lu, rows, recovery_rhs, recovered))
    {
      *view.status |= scalar_detail::device_singular_local_block;
      return;
    }
    const double residual = ApplicationResidual(a, recovered, recovery_rhs);
    const double recovery_rhs_norm = VectorNorm(recovery_rhs, ni);
    maximum_residual = fmax(maximum_residual, residual);
    const double ratio = residual /
        (1.0e-12 + 1.0e-11 * recovery_rhs_norm);
    if (ratio > maximum_contract_ratio)
    {
      maximum_contract_ratio = ratio;
      paired_residual = residual;
      paired_rhs_norm = recovery_rhs_norm;
    }
    if (residual > local_absolute_tolerance +
            local_relative_tolerance * recovery_rhs_norm)
    {
      *view.status |= scalar_detail::device_local_application;
      return;
    }
    for (int component = 0; component < 7; ++component)
      for (int aindex = 0; aindex < ns; ++aindex)
      {
        view.direction_interior[InteriorIndex(element, component, aindex)] =
            -recovered[LocalIndex(component, aindex)];
      }
  }
  view.reduction[4] = maximum_residual;
  view.reduction[6] = paired_residual;
  view.reduction[7] = paired_rhs_norm;
}

__device__ void ApplyTrial(scalar_detail::DeviceView& view)
{
  const double* entry = view.slot_interior[view.selected_current_slot];
  const double* entry_trace = view.slot_trace[view.selected_current_slot];
  const int interior_count = scalar::persistent_interior_unknowns;
  for (int i = 0; i < interior_count; ++i)
  {
    view.trial_interior[i] = entry[i] +
        view.trial_theta * view.direction_interior[i];
  }
  for (int i = 0; i < view.true_trace_unknowns; ++i)
  {
    view.trial_trace[i] = entry_trace[i] +
        view.trial_theta * view.direction_trace[i];
  }
}

__device__ void AcceptTrial(scalar_detail::DeviceView& view)
{
  double* destination = view.slot_interior[view.selected_current_slot];
  double* destination_trace = view.slot_trace[view.selected_current_slot];
  const int interior_count = scalar::persistent_interior_unknowns;
  for (int i = 0; i < interior_count; ++i)
  {
    destination[i] = view.trial_interior[i];
  }
  for (int i = 0; i < view.true_trace_unknowns; ++i)
  {
    destination_trace[i] = view.trial_trace[i];
  }
}

__device__ void PackCheckpoint(scalar_detail::DeviceView& view)
{
  const double* interior = view.slot_interior[view.selected_current_slot];
  const double* trace = view.slot_trace[view.selected_current_slot];
  const int scalar_nodes = view.elements * ns;
  for (int element = 0; element < view.elements; ++element)
    for (int a = 0; a < ns; ++a)
    {
      const int scalar_node = element * ns + a;
      for (int component = 0; component < 7; ++component)
      {
        view.checkpoint_interior_slabs[component * scalar_nodes + scalar_node] =
            interior[InteriorIndex(element, component, a)];
      }
    }
  for (int component = 0; component < 5; ++component)
    for (int a = 0;
         a < view.owned_true_trace_unknowns / scalar::trace_components; ++a)
    {
      const int local_scalar =
          view.owned_true_trace_unknowns / scalar::trace_components;
      view.checkpoint_trace_slabs[
          component * local_scalar + a] =
          trace[view.owned_true_begin + component * local_scalar + a];
    }
}

__device__ void UnpackCheckpoint(scalar_detail::DeviceView& view)
{
  double* interior = view.slot_interior[view.selected_current_slot];
  double* trace = view.slot_trace[view.selected_current_slot];
  const int scalar_nodes = view.elements * ns;
  for (int element = 0; element < view.elements; ++element)
    for (int a = 0; a < ns; ++a)
    {
      const int scalar_node = element * ns + a;
      for (int component = 0; component < 7; ++component)
      {
        interior[InteriorIndex(element, component, a)] =
            view.checkpoint_interior_slabs[
                component * scalar_nodes + scalar_node];
      }
    }
  for (int component = 0; component < 5; ++component)
    for (int a = 0;
         a < view.owned_true_trace_unknowns / scalar::trace_components; ++a)
    {
      const int local_scalar =
          view.owned_true_trace_unknowns / scalar::trace_components;
      trace[view.owned_true_begin + component * local_scalar + a] =
          view.checkpoint_trace_slabs[
              component * local_scalar + a];
    }
}

__device__ void ReduceState(scalar_detail::DeviceView& view)
{
  const double* interior = view.slot_interior[view.selected_current_slot];
  const double* trace = view.slot_trace[view.selected_current_slot];
  for (int i = 0; i < 16; ++i) { view.reduction[i] = 0.0; }
  view.reduction[1] = DBL_MAX;
  view.reduction[2] = DBL_MAX;
  view.reduction[3] = DBL_MAX;
  for (int element = 0; element < view.elements; ++element)
    for (int qpoint = 0; qpoint < view.volume_quadrature_points; ++qpoint)
    {
      const int qindex = element * view.volume_quadrature_points + qpoint;
      const double* shape = view.volume_shape + qindex * ns;
      double state[4]{};
      for (int component = 0; component < 4; ++component)
      {
        state[component] = EvaluateInterior(
            interior, element, component, shape);
        view.reduction[8 + component] +=
            view.volume_weight[qindex] * state[component];
      }
      double pressure = 0.0;
      double velocity[2]{};
      if (!FiniteConservativeState(state))
      {
        *view.status |= scalar_detail::device_nonfinite;
        return;
      }
      if (!Primitive(state, pressure, velocity))
      {
        *view.status |= scalar_detail::device_admissibility;
        return;
      }
      const double margin = 2.5 - hypot(velocity[0], velocity[1]) -
          sqrt(gamma_value * pressure / state[0]);
      if (!DeviceFinite(margin))
      {
        *view.status |= scalar_detail::device_nonfinite;
        return;
      }
      view.reduction[1] = fmin(view.reduction[1], state[0]);
      view.reduction[2] = fmin(view.reduction[2], pressure);
      view.reduction[3] = fmin(view.reduction[3], margin);
    }
  const int local_scalar =
      view.owned_true_trace_unknowns / scalar::trace_components;
  for (int a = 0; a < local_scalar; ++a)
  {
    double state[4]{};
    for (int component = 0; component < 4; ++component)
    {
      state[component] = trace[view.owned_true_begin +
          component * local_scalar + a];
    }
    double pressure = 0.0;
    double velocity[2]{};
    if (!FiniteConservativeState(state))
    {
      *view.status |= scalar_detail::device_nonfinite;
      return;
    }
    if (!Primitive(state, pressure, velocity))
    {
      *view.status |= scalar_detail::device_admissibility;
      return;
    }
    const double margin = 2.5 - hypot(velocity[0], velocity[1]) -
        sqrt(gamma_value * pressure / state[0]);
    if (!DeviceFinite(margin))
    {
      *view.status |= scalar_detail::device_nonfinite;
      return;
    }
    view.reduction[1] = fmin(view.reduction[1], state[0]);
    view.reduction[2] = fmin(view.reduction[2], pressure);
    view.reduction[3] = fmin(view.reduction[3], margin);
  }
}

__device__ inline std::uint32_t RotateRight(const std::uint32_t value,
                                            const unsigned int shift)
{
  return (value >> shift) | (value << (32U - shift));
}

__device__ void DeviceSha256Two(const std::uint8_t* first,
                                const std::size_t first_length,
                                const std::uint8_t* second,
                                const std::size_t second_length,
                                std::uint8_t* digest)
{
  constexpr std::uint32_t initial[8]{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  constexpr std::uint32_t constants[64]{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::uint32_t state[8];
  for (int i = 0; i < 8; ++i) { state[i] = initial[i]; }
  const std::size_t length = first_length + second_length;
  const std::size_t blocks = (length + 9U + 63U) / 64U;
  const std::uint64_t bit_length = static_cast<std::uint64_t>(length) * 8U;
  for (std::size_t block = 0; block < blocks; ++block)
  {
    std::uint32_t words[64]{};
    for (int word = 0; word < 16; ++word)
      for (int byte = 0; byte < 4; ++byte)
      {
        const std::size_t position = block * 64U + word * 4U + byte;
        std::uint8_t value = 0;
        if (position < first_length) { value = first[position]; }
        else if (position < length)
        {
          value = second[position - first_length];
        }
        else if (position == length) { value = 0x80U; }
        else if (position >= blocks * 64U - 8U)
        {
          const unsigned int shift = static_cast<unsigned int>(
              (blocks * 64U - 1U - position) * 8U);
          value = static_cast<std::uint8_t>(bit_length >> shift);
        }
        words[word] |= static_cast<std::uint32_t>(value) <<
            static_cast<unsigned int>((3 - byte) * 8);
      }
    for (int word = 16; word < 64; ++word)
    {
      const std::uint32_t s0 = RotateRight(words[word - 15], 7) ^
          RotateRight(words[word - 15], 18) ^ (words[word - 15] >> 3);
      const std::uint32_t s1 = RotateRight(words[word - 2], 17) ^
          RotateRight(words[word - 2], 19) ^ (words[word - 2] >> 10);
      words[word] = words[word - 16] + s0 + words[word - 7] + s1;
    }
    std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int round = 0; round < 64; ++round)
    {
      const std::uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^
          RotateRight(e, 25);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + sum1 + choice + constants[round] + words[round];
      const std::uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^
          RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g; g = f; f = e; e = d + temporary1;
      d = c; c = b; b = a; a = temporary1 + temporary2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }
  for (int word = 0; word < 8; ++word)
    for (int byte = 0; byte < 4; ++byte)
    {
      digest[word * 4 + byte] = static_cast<std::uint8_t>(
          state[word] >> static_cast<unsigned int>((3 - byte) * 8));
    }
}

__device__ void DeviceSha256(const std::uint8_t* bytes,
                             const std::size_t length,
                             std::uint8_t* digest)
{
  DeviceSha256Two(bytes, length, nullptr, 0, digest);
}

__device__ void HashSelectedState(scalar_detail::DeviceView& view)
{
  const auto* interior = reinterpret_cast<const std::uint8_t*>(
      view.slot_interior[view.selected_current_slot]);
  const auto* trace = reinterpret_cast<const std::uint8_t*>(
      view.slot_trace[view.selected_current_slot] + view.owned_true_begin);
  const std::size_t u_bytes = scalar::persistent_u_unknowns *
      sizeof(double);
  const std::size_t sigma_bytes =
      scalar::persistent_sigma_unknowns * sizeof(double);
  const std::size_t z_bytes = scalar::persistent_z_unknowns *
      sizeof(double);
  const std::size_t local_scalar =
      static_cast<std::size_t>(view.owned_true_trace_unknowns) /
      scalar::trace_components;
  const std::size_t uhat_bytes = scalar::state_components * local_scalar *
      sizeof(double);
  const std::size_t sigmahat_bytes = local_scalar * sizeof(double);
  DeviceSha256(interior, u_bytes, view.state_hash_bytes + 0 * 32);
  DeviceSha256(interior + u_bytes, sigma_bytes,
               view.state_hash_bytes + 1 * 32);
  DeviceSha256(interior + u_bytes + sigma_bytes, z_bytes,
               view.state_hash_bytes + 2 * 32);
  DeviceSha256(trace, uhat_bytes, view.state_hash_bytes + 3 * 32);
  DeviceSha256(trace + uhat_bytes, sigmahat_bytes,
               view.state_hash_bytes + 4 * 32);
  DeviceSha256(interior, u_bytes + sigma_bytes + z_bytes,
               view.state_hash_bytes + 5 * 32);
  DeviceSha256(trace, uhat_bytes + sigmahat_bytes,
               view.state_hash_bytes + 6 * 32);
  DeviceSha256Two(interior, u_bytes + sigma_bytes + z_bytes, trace,
                  uhat_bytes + sigmahat_bytes,
                  view.state_hash_bytes + 7 * 32);
}

__device__ void HashUploadedTopology(scalar_detail::DeviceView& view)
{
  DeviceSha256(reinterpret_cast<const std::uint8_t*>(
                   view.local_trace_to_true),
               static_cast<std::size_t>(view.topology_integer_count) *
                   sizeof(int),
               view.state_hash_bytes + 8 * 32);
  DeviceSha256(reinterpret_cast<const std::uint8_t*>(view.volume_weight),
               static_cast<std::size_t>(view.quadrature_double_count) *
                   sizeof(double),
               view.state_hash_bytes + 9 * 32);
}

__device__ void CopySlot(scalar_detail::DeviceView& view)
{
  const int source = view.selected_current_slot;
  const int destination = 1 - source;
  for (int i = 0; i < scalar::persistent_interior_unknowns; ++i)
  {
    view.slot_interior[destination][i] = view.slot_interior[source][i];
  }
  for (int i = 0; i < view.true_trace_unknowns; ++i)
  {
    view.slot_trace[destination][i] = view.slot_trace[source][i];
  }
}

__device__ void InstallDistributedTrace(scalar_detail::DeviceView& view)
{
  double* destination = view.slot_trace[view.selected_current_slot];
  for (int item = 0; item < view.true_trace_unknowns; ++item)
  {
    destination[item] = view.distributed_global_trace[item];
  }
}

} // namespace

extern "C" __global__ void HdgSc25Rank2AssembleLocalFp64Kernel(
    scalar_detail::DeviceView view)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  RecordArchitecture(view);
  const auto operation =
      static_cast<scalar_detail::KernelOperation>(view.operation);
  if (operation == scalar_detail::KernelOperation::initialize_projection)
  {
    InitializeProjection(view);
  }
  else if (operation == scalar_detail::KernelOperation::initialize_fixture)
  {
    InitializeFixture(view);
  }
  else if (operation == scalar_detail::KernelOperation::clear_assembly)
  {
    ZeroAssembly(view);
  }
  else if (operation == scalar_detail::KernelOperation::assemble)
  {
    AssembleSystem(view);
  }
  else if (operation == scalar_detail::KernelOperation::auxiliary_identity_rows)
  {
    ApplyAuxiliaryIdentityRows(view);
  }
}

extern "C" __global__ void HdgSc25Rank2CondenseFp64Kernel(
    scalar_detail::DeviceView view)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  RecordArchitecture(view);
  const auto operation =
      static_cast<scalar_detail::KernelOperation>(view.operation);
  if (operation == scalar_detail::KernelOperation::condense_and_scatter)
    CondenseAndScatter(view);
  else if (operation == scalar_detail::KernelOperation::scatter_remote_system)
    ScatterRemoteSystem(view);
  else if (operation == scalar_detail::KernelOperation::gather_trace_residual)
    GatherTraceResidual(view);
  else if (operation == scalar_detail::KernelOperation::fixed_jacobian_action)
    FixedJacobianAction(view);
  else if (operation ==
           scalar_detail::KernelOperation::scatter_remote_fixed_action)
    ScatterRemoteFixedAction(view);
  else if (operation == scalar_detail::KernelOperation::partial_pivot_contract)
    PartialPivotContract(view);
  else if (operation ==
           scalar_detail::KernelOperation::scatter_remote_trace_residual)
    ScatterRemoteTraceResidual(view);
  else
    *view.status |= scalar_detail::device_invalid_layout;
}

extern "C" __global__ void HdgSc25Rank2RecoverFp64Kernel(
    scalar_detail::DeviceView view)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  RecordArchitecture(view);
  const auto operation =
      static_cast<scalar_detail::KernelOperation>(view.operation);
  if (operation == scalar_detail::KernelOperation::recover_direction)
  {
    RecoverDirection(view);
  }
  else if (operation == scalar_detail::KernelOperation::apply_trial)
  {
    ApplyTrial(view);
  }
  else if (operation == scalar_detail::KernelOperation::accept_trial)
  {
    AcceptTrial(view);
  }
  else if (operation == scalar_detail::KernelOperation::pack_checkpoint)
  {
    PackCheckpoint(view);
  }
  else if (operation == scalar_detail::KernelOperation::unpack_checkpoint)
  {
    UnpackCheckpoint(view);
  }
  else if (operation == scalar_detail::KernelOperation::reduce_state)
  {
    ReduceState(view);
  }
  else if (operation == scalar_detail::KernelOperation::hash_state)
  {
    HashSelectedState(view);
  }
  else if (operation == scalar_detail::KernelOperation::hash_topology)
  {
    HashUploadedTopology(view);
  }
  else if (operation == scalar_detail::KernelOperation::copy_slot)
  {
    CopySlot(view);
  }
  else if (operation ==
           scalar_detail::KernelOperation::install_distributed_trace)
  {
    InstallDistributedTrace(view);
  }
}

namespace dg::hdg::runtime::cuda_w8b::sc25
{
namespace
{

int DecodeSignedDof(const int encoded, int& sign) noexcept
{
  sign = encoded < 0 ? -1 : 1;
  return encoded < 0 ? -1 - encoded : encoded;
}

int EncodeSignedDof(const int dof, const int sign) noexcept
{
  return sign < 0 ? -1 - dof : dof;
}

StateHash HashBytes(const void* data, const std::size_t bytes) noexcept
{
  Sha256 hash;
  Sha256Digest digest{};
  if ((bytes != 0 && data == nullptr) || !hash.Update(data, bytes) ||
      !hash.Finalize(digest))
  {
    return {};
  }
  return {digest};
}

template <typename... Arrays>
StateHash HashArrays(const Arrays&... arrays) noexcept
{
  Sha256 hash;
  Sha256Digest digest{};
  const bool updated =
      (hash.Update(arrays.data(), arrays.size() * sizeof(*arrays.data())) &&
       ...);
  return updated && hash.Finalize(digest) ? StateHash{digest} : StateHash{};
}

StateHashes HashPublishedState(const PublishedState& state) noexcept
{
  return {HashArrays(state.u), HashArrays(state.sigma), HashArrays(state.z),
          HashArrays(state.uhat), HashArrays(state.sigmahat),
          HashArrays(state.u, state.sigma, state.z),
          HashArrays(state.uhat, state.sigmahat),
          HashArrays(state.u, state.sigma, state.z, state.uhat,
                     state.sigmahat)};
}

bool SameStateHashes(const StateHashes& left,
                     const StateHashes& right) noexcept
{
  return left.u == right.u && left.sigma == right.sigma &&
      left.z == right.z && left.uhat == right.uhat &&
      left.sigmahat == right.sigmahat &&
      left.interior == right.interior && left.trace == right.trace &&
      left.shard == right.shard;
}

void ConfigureIlu(mfem::HypreILU& ilu)
{
  ilu.SetType(0);
  ilu.SetLevelOfFill(0);
  ilu.SetMaxIter(1);
  ilu.SetTol(0.0);
  ilu.SetLocalReordering(1);
  ilu.SetPrintLevel(0);
}

std::size_t UmpireHighWater(const char* name) noexcept
{
  if (name == nullptr || name[0] == '\0') { return 0; }
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    return manager.isAllocator(name)
        ? manager.getAllocator(name).getHighWatermark() : 0;
  }
  catch (...) { return 0; }
}

bool SameCoordinates(const StateCoordinates& left,
                     const StateCoordinates& right) noexcept
{
  return left.time == right.time && left.step == right.step &&
      left.generation == right.generation && left.cursor == right.cursor &&
      left.dt == right.dt;
}

int HostInteriorIndex(const int element, const int component,
                      const int scalar_node) noexcept
{
  if (component < state_components)
  {
    return (element * state_components + component) * volume_nodes +
        scalar_node;
  }
  if (component == state_components)
  {
    return persistent_u_unknowns + element * volume_nodes +
        scalar_node;
  }
  return persistent_u_unknowns +
      persistent_sigma_unknowns +
      (element * flux_components + component - state_components - 1) *
          volume_nodes + scalar_node;
}

bool ObservedCudaDevicePointer(const void* pointer) noexcept
{
  if (pointer == nullptr) { return false; }
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice;
}

bool CollectiveAll(MPI_Comm communicator, const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && output == 1;
}

bool CollectiveAny(MPI_Comm communicator, const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MAX,
                       communicator) == MPI_SUCCESS && output == 1;
}

struct LocalPhaseFailure final
{};

} // namespace

struct PersistentSc25ScalarIgrCudaRank2::Implementation final
{
  struct OwnedCommunicator final
  {
    MPI_Comm value = MPI_COMM_NULL;
    ~OwnedCommunicator() noexcept
    {
      if (value == MPI_COMM_NULL) { return; }
      int finalized = 0;
      MPI_Finalized(&finalized);
      if (finalized == 0) { MPI_Comm_free(&value); }
    }
  } communicator;

  ScalarCudaConfig config{};
  ScalarCudaStatus status = ScalarCudaStatus::uninitialized;
  ScalarCudaStatus first_failure = ScalarCudaStatus::uninitialized;
  int rank = -1;
  int ranks = 0;

  bool smooth_transition() const noexcept
  {
    return config.lifecycle == ScalarCudaLifecycle::w8c_smooth_transition_v2;
  }

  int required_steps() const noexcept
  {
    return smooth_transition() ? smooth_transition_required_steps
                               : persistent_required_steps;
  }

  double time_step() const noexcept
  {
    return smooth_transition() ? smooth_transition_dt : fixed_dt;
  }

  std::string_view fixture_id() const noexcept
  {
    return smooth_transition() ? cuda_rank2_smooth_transition_fixture_id
                               : cuda_rank2_fixture_id;
  }
  int device = -1;
  int elements = 0;
  int scalar_true_trace_nodes = 0;
  int true_trace_unknowns = 0;
  int owned_scalar_trace_nodes = 0;
  int owned_true_trace_unknowns = 0;
  int owned_true_begin = 0;
  int full_unknowns = 0;
  int volume_qpoints = 0;
  int face_qpoints = 0;
  int projection_volume_qpoints = 0;
  int projection_face_qpoints = 0;
  bool same_device_uuid = false;
  StateHash mesh_hash{};
  StateHash partition_hash{};
  StateHash topology_hash{};
  StateHash topology_reference_hash{};
  StateHash quadrature_reference_hash{};
  StateHash observed_device_topology_hash{};
  StateHash observed_device_quadrature_hash{};

  std::unique_ptr<cpu_sc25::PersistentSc25ScalarIgrCpuFoundation>
      topology_foundation;
  const mfem::L2_FECollection* volume_collection = nullptr;
  const mfem::DG_Interface_FECollection* trace_collection = nullptr;
  const mfem::ParMesh* mesh = nullptr;
  std::unique_ptr<mfem::ParFiniteElementSpace> volume_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> carrier_space;
  trace::DGInterfaceTraceMap carrier_map;
  std::unique_ptr<mfem::SparseMatrix> local_pattern;
  std::unique_ptr<mfem::HypreParMatrix> true_matrix;
  std::unique_ptr<mfem::HypreParVector> true_rhs;
  std::unique_ptr<mfem::HypreParVector> true_solution;
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;
  mfem::Vector true_residual;
  mfem::SparseMatrix diagonal_view;
  mfem::SparseMatrix offdiagonal_view;
  HYPRE_BigInt* offdiagonal_column_map = nullptr;

  std::vector<int> signed_trace_vdofs;
  std::vector<int> local_trace_to_true;
  std::vector<int> local_trace_sign;
  std::vector<int> local_trace_owner;
  std::vector<int> element_to_true_csr;
  std::vector<int> true_row_offsets;
  std::vector<int> true_columns;
  std::vector<int> matrix_owned_source;
  std::vector<int> matrix_owned_section;
  std::vector<int> matrix_owned_target;
  std::vector<int> matrix_send_source;
  std::vector<int> matrix_receive_section;
  std::vector<int> matrix_receive_target;
  std::vector<int> vector_owned_source;
  std::vector<int> vector_owned_target;
  std::vector<int> vector_send_source;
  std::vector<int> vector_receive_target;
  std::vector<std::int64_t> matrix_send_keys;
  std::vector<std::int64_t> matrix_receive_keys;
  std::vector<std::int64_t> vector_send_keys;
  std::vector<std::int64_t> vector_receive_keys;
  std::vector<double> matrix_send_host;
  std::vector<double> matrix_receive_host;
  std::vector<double> vector_send_host;
  std::vector<double> vector_receive_host;
  std::vector<double> owned_trace_host;
  std::vector<double> global_trace_host;

  std::vector<double> volume_weight;
  std::vector<double> volume_shape;
  std::vector<double> volume_derivative;
  std::vector<double> volume_x;
  std::vector<double> volume_y;
  std::vector<double> volume_node_x;
  std::vector<double> volume_node_y;
  std::vector<double> face_weight;
  std::vector<double> face_height;
  std::vector<double> face_interior_shape;
  std::vector<double> face_oriented_shape;
  std::vector<double> face_normal;
  std::vector<double> face_x;
  std::vector<double> face_y;
  std::vector<double> projection_volume_weight;
  std::vector<double> projection_volume_shape;
  std::vector<double> projection_volume_x;
  std::vector<double> projection_face_weight;
  std::vector<double> projection_face_oriented_shape;
  std::vector<double> projection_face_x;
  std::vector<double> trace_node_coordinate;

  void* double_arena = nullptr;
  void* integer_arena = nullptr;
  std::size_t double_arena_bytes = 0;
  std::size_t integer_arena_bytes = 0;
  detail::DeviceView view{};

  int committed_slot = 0;
  StateCoordinates coordinates{};
  std::uint64_t incarnation = 0;
  std::uint64_t next_transaction_serial = 1;
  std::uint64_t active_transaction_serial = 0;
  int active_candidate_slot = -1;
  bool active_installed = false;
  bool initialized = false;
  bool baseline_frozen = false;
  std::uint64_t topology_generation = 1;
  std::uint64_t storage_generation = 1;
  std::uint64_t factor_generation = 0;
  std::uint64_t solver_generation = 0;

  ScalarCudaEvidence evidence{};
  ScalarCudaAttemptEvidence pending_attempt{};
  ScalarCudaStepEvidence pending_step{};
  ScalarCudaSnapshotEvidence capture_snapshot{};
  ScalarCudaSnapshotEvidence pending_snapshot{};
  PublishedState checkpoint_workspace{};
  bool pending_attempt_active = false;
  int prepare_retry_index = 0;
  int last_device_status = detail::device_success;
  ScalarCudaEvidence transaction_prior_evidence{};
  ScalarCudaStatus transaction_prior_status = ScalarCudaStatus::uninitialized;
  ScalarCudaStatus transaction_prior_first_failure =
      ScalarCudaStatus::uninitialized;
  bool transaction_installed_after_retry = false;

  ~Implementation() noexcept
  {
    gmres.reset();
    ilu.reset();
    true_solution.reset();
    true_rhs.reset();
    true_matrix.reset();
    local_pattern.reset();
    if (integer_arena != nullptr)
    {
      cuda_w8a::UmpireDeviceAllocatorPolicy::Deallocate(
          integer_arena,
          cuda_w8a::CudaAllocationSite::persistent_integer_arena);
      integer_arena = nullptr;
    }
    if (double_arena != nullptr)
    {
      cuda_w8a::UmpireDeviceAllocatorPolicy::Deallocate(
          double_arena,
          cuda_w8a::CudaAllocationSite::persistent_double_arena);
      double_arena = nullptr;
    }
  }

  bool BuildMeshTopologyAndQuadrature() noexcept
  {
    const int n = config.elements_per_direction;
    bool local_valid = true;
    try
    {
      cpu_sc25::FoundationConfig foundation_config{};
      foundation_config.polynomial_degree = 1;
      foundation_config.elements_per_direction = n;
      const cpu_sc25::FoundationStatus foundation_status =
          cpu_sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
              communicator.value, cpu_sc25::CpuMpiRank2Tag{},
              foundation_config, topology_foundation);
      local_valid = foundation_status == cpu_sc25::FoundationStatus::ready &&
          topology_foundation != nullptr;
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    local_valid = true;
    try
    {
      mesh = topology_foundation->mesh();
      volume_collection = topology_foundation->volume_collection();
      trace_collection = topology_foundation->trace_collection();
      local_valid = mesh != nullptr && volume_collection != nullptr &&
          trace_collection != nullptr;
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    local_valid = true;
    try
    {
      volume_space = std::make_unique<mfem::ParFiniteElementSpace>(
          const_cast<mfem::ParMesh*>(mesh), volume_collection, 1,
          mfem::Ordering::byNODES);
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    local_valid = true;
    try
    {
      carrier_space = std::make_unique<mfem::ParFiniteElementSpace>(
          const_cast<mfem::ParMesh*>(mesh), trace_collection,
          trace_components, mfem::Ordering::byNODES);
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    local_valid = true;
    try
    {
      local_valid = ::dg::hdg::trace::BuildDGInterfaceTraceMap(
          *carrier_space, carrier_map);
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    try
    {
      elements = mesh->GetNE();
      scalar_true_trace_nodes = persistent_scalar_trace_nodes;
      true_trace_unknowns = persistent_trace_unknowns;
      owned_true_trace_unknowns = carrier_space->GetTrueVSize();
      owned_scalar_trace_nodes = owned_true_trace_unknowns / trace_components;
      owned_true_begin = static_cast<int>(carrier_space->GetMyTDofOffset());
      full_unknowns = elements * local_interior_unknowns +
          true_trace_unknowns;
      const int expected_owned = rank == 0 ?
          cpu_sc25::persistent_rank2_rank0_owned_trace_unknowns :
          cpu_sc25::persistent_rank2_rank1_owned_trace_unknowns;
      const int expected_scalar = rank == 0 ?
          cpu_sc25::persistent_rank2_rank0_scalar_true_dofs :
          cpu_sc25::persistent_rank2_rank1_scalar_true_dofs;
      const int expected_ghost = rank == 0 ?
          cpu_sc25::persistent_rank2_rank0_ghost_trace_unknowns :
          cpu_sc25::persistent_rank2_rank1_ghost_trace_unknowns;
      if (elements != persistent_elements ||
          mesh->GetNumFaces() != cpu_sc25::persistent_rank2_local_faces ||
          volume_space->GetVSize() != elements * volume_nodes ||
          carrier_space->GlobalTrueVSize() != true_trace_unknowns ||
          owned_true_trace_unknowns != expected_owned ||
          owned_scalar_trace_nodes != expected_scalar ||
          static_cast<int>(carrier_map.GhostCount()) != expected_ghost ||
          n != cpu_sc25::persistent_elements_per_direction)
      {
        throw LocalPhaseFailure{};
      }

      signed_trace_vdofs.assign(
          static_cast<std::size_t>(elements) * local_trace_unknowns, 0);
      local_trace_to_true.assign(signed_trace_vdofs.size(), -1);
      local_trace_sign.assign(signed_trace_vdofs.size(), 0);
      local_trace_owner.assign(signed_trace_vdofs.size(), -1);
      trace_node_coordinate.assign(
          static_cast<std::size_t>(elements) * 4 * face_nodes, 0.0);
      volume_node_x.assign(static_cast<std::size_t>(elements) * volume_nodes,
                           0.0);
      volume_node_y.assign(volume_node_x.size(), 0.0);

      for (int element = 0; element < elements; ++element)
      {
        const mfem::FiniteElement* volume = volume_space->GetFE(element);
        mfem::IsoparametricTransformation transformation_storage;
        mesh->GetElementTransformation(element, &transformation_storage);
        mfem::ElementTransformation* transformation =
            &transformation_storage;
        if (volume == nullptr || transformation == nullptr ||
            volume->GetDof() != volume_nodes)
        {
          throw LocalPhaseFailure{};
        }
        mfem::Vector point(dimension);
        const mfem::IntegrationRule& nodes = volume->GetNodes();
        for (int a = 0; a < volume_nodes; ++a)
        {
          transformation->Transform(nodes.IntPoint(a), point);
          volume_node_x[static_cast<std::size_t>(element) * volume_nodes + a] =
              point[0];
          volume_node_y[static_cast<std::size_t>(element) * volume_nodes + a] =
              point[1];
        }
        mfem::Array<int> faces, orientations;
        mesh->GetElementEdges(element, faces, orientations);
        if (faces.Size() != 4 || orientations.Size() != 4)
          throw LocalPhaseFailure{};
        for (int local_face = 0; local_face < 4; ++local_face)
        {
          const int face = faces[local_face];
          const mfem::FiniteElement* face_element =
              carrier_space->GetFaceElement(face);
          mfem::Array<int> face_vdofs;
          carrier_space->GetFaceVDofs(face, face_vdofs);
          if (face_element == nullptr || face_element->GetDof() != face_nodes ||
              face_vdofs.Size() != trace_components * face_nodes)
          {
            throw LocalPhaseFailure{};
          }
          const int* permutation = trace_collection->DofOrderForOrientation(
              mfem::Geometry::SEGMENT, orientations[local_face]);
          if (permutation == nullptr) { throw LocalPhaseFailure{}; }
          const mfem::IntegrationRule& face_nodes_rule =
              face_element->GetNodes();
          for (int component = 0; component < trace_components; ++component)
            for (int oriented = 0; oriented < face_nodes; ++oriented)
            {
              int orientation_sign = 0;
              const int canonical = DecodeSignedDof(
                  permutation[oriented], orientation_sign);
              int carrier_sign = 0;
              const int local_dof = DecodeSignedDof(
                  face_vdofs[component * face_nodes + canonical],
                  carrier_sign);
              if (local_dof < 0 ||
                  static_cast<std::size_t>(local_dof) >=
                      carrier_map.dofs.entries.size())
              {
                throw LocalPhaseFailure{};
              }
              const ::dg::hdg::trace::TraceMapEntry& trace_entry =
                  carrier_map.dofs.entries[
                      static_cast<std::size_t>(local_dof)];
              const int true_dof =
                  static_cast<int>(trace_entry.global_true_dof);
              const int sign = orientation_sign * carrier_sign;
              const int local_row = component * nf +
                  local_face * face_nodes + oriented;
              const std::size_t map =
                  static_cast<std::size_t>(element) * local_trace_unknowns +
                  local_row;
              if (canonical < 0 || canonical >= face_nodes || local_dof < 0 ||
                  true_dof < 0 || true_dof >= true_trace_unknowns ||
                  trace_entry.owner_rank < 0 || trace_entry.owner_rank > 1 ||
                  sign == 0)
              {
                throw LocalPhaseFailure{};
              }
              signed_trace_vdofs[map] = EncodeSignedDof(local_dof, sign);
              local_trace_to_true[map] = true_dof;
              local_trace_sign[map] = sign;
              local_trace_owner[map] = trace_entry.owner_rank;
              if (component == 0)
              {
                trace_node_coordinate[
                    (static_cast<std::size_t>(element) * 4 + local_face) *
                        face_nodes + oriented] =
                    2.0 * face_nodes_rule.IntPoint(canonical).x - 1.0;
              }
            }
        }
      }
      mesh_hash = HashArrays(volume_node_x, volume_node_y);
      const std::array<int, 3> partition_identity{{rank, ranks, elements}};
      partition_hash = HashBytes(partition_identity.data(),
                                 sizeof(partition_identity));
      topology_hash = HashArrays(local_trace_to_true, local_trace_sign,
                                 local_trace_owner);
      if (mesh_hash == StateHash{} || partition_hash == StateHash{} ||
          topology_hash == StateHash{})
      {
        throw LocalPhaseFailure{};
      }

      const mfem::FiniteElement* first_volume = volume_space->GetFE(0);
      const mfem::FiniteElement* first_trace = carrier_space->GetFaceElement(0);
      if (first_volume == nullptr || first_trace == nullptr)
        throw LocalPhaseFailure{};
      volume_qpoints = mfem::IntRules.Get(
          first_volume->GetGeomType(),
          cpu_sc25::coupled_fixture_quadrature_order).GetNPoints();
      face_qpoints = mfem::IntRules.Get(
          first_trace->GetGeomType(),
          cpu_sc25::coupled_fixture_quadrature_order).GetNPoints();
      const std::size_t volume_points =
          static_cast<std::size_t>(elements) * volume_qpoints;
      const std::size_t face_points =
          static_cast<std::size_t>(elements) * 4 * face_qpoints;
      volume_weight.assign(volume_points, 0.0);
      volume_shape.assign(volume_points * volume_nodes, 0.0);
      volume_derivative.assign(volume_points * volume_nodes * dimension, 0.0);
      volume_x.assign(volume_points, 0.0);
      volume_y.assign(volume_points, 0.0);
      face_weight.assign(face_points, 0.0);
      face_height.assign(static_cast<std::size_t>(elements) * 4, 0.0);
      face_interior_shape.assign(face_points * volume_nodes, 0.0);
      face_oriented_shape.assign(face_points * face_nodes, 0.0);
      face_normal.assign(face_points * dimension, 0.0);
      face_x.assign(face_points, 0.0);
      face_y.assign(face_points, 0.0);

      mfem::Vector volume_shape_value(volume_nodes);
      mfem::DenseMatrix volume_derivative_value(volume_nodes, dimension);
      mfem::Vector point(dimension);
      for (int element = 0; element < elements; ++element)
      {
        const mfem::FiniteElement* finite_element = volume_space->GetFE(element);
        mfem::IsoparametricTransformation transformation_storage;
        mesh->GetElementTransformation(element, &transformation_storage);
        mfem::ElementTransformation* transformation =
            &transformation_storage;
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(
            finite_element->GetGeomType(),
            cpu_sc25::coupled_fixture_quadrature_order);
        if (rule.GetNPoints() != volume_qpoints)
          throw LocalPhaseFailure{};
        double element_measure = 0.0;
        for (int q = 0; q < volume_qpoints; ++q)
        {
          const mfem::IntegrationPoint& ip = rule.IntPoint(q);
          transformation->SetIntPoint(&ip);
          finite_element->CalcShape(ip, volume_shape_value);
          finite_element->CalcPhysDShape(*transformation,
                                         volume_derivative_value);
          transformation->Transform(ip, point);
          const int qindex = element * volume_qpoints + q;
          const double weight = ip.weight * transformation->Weight();
          volume_weight[qindex] = weight;
          element_measure += weight;
          volume_x[qindex] = point[0];
          volume_y[qindex] = point[1];
          for (int a = 0; a < volume_nodes; ++a)
          {
            volume_shape[static_cast<std::size_t>(qindex) * volume_nodes + a] =
                volume_shape_value[a];
            for (int direction = 0; direction < dimension; ++direction)
            {
              volume_derivative[
                  (static_cast<std::size_t>(qindex) * volume_nodes + a) *
                      dimension + direction] =
                  volume_derivative_value(a, direction);
            }
          }
        }

        mfem::Array<int> faces, orientations;
        mesh->GetElementEdges(element, faces, orientations);
        for (int local_face = 0; local_face < 4; ++local_face)
        {
          const int face = faces[local_face];
          mfem::FaceElementTransformations face_transformation_storage;
          mfem::IsoparametricTransformation element_one_storage;
          mfem::IsoparametricTransformation element_two_storage;
          mesh->GetFaceElementTransformations(
              face, face_transformation_storage, element_one_storage,
              element_two_storage);
          mfem::FaceElementTransformations* face_transformation =
              &face_transformation_storage;
          const mfem::FiniteElement* trace_element =
              carrier_space->GetFaceElement(face);
          if (face_transformation == nullptr ||
              face_transformation->Face == nullptr || trace_element == nullptr)
          {
            throw LocalPhaseFailure{};
          }
          const bool element_one = face_transformation->Elem1No == element;
          if (!element_one && face_transformation->Elem2No != element)
          {
            throw LocalPhaseFailure{};
          }
          const int* permutation = trace_collection->DofOrderForOrientation(
              mfem::Geometry::SEGMENT, orientations[local_face]);
          const mfem::IntegrationRule& rule = mfem::IntRules.Get(
              trace_element->GetGeomType(),
              cpu_sc25::coupled_fixture_quadrature_order);
          if (permutation == nullptr || rule.GetNPoints() != face_qpoints)
          {
            throw LocalPhaseFailure{};
          }
          double face_measure = 0.0;
          for (int q = 0; q < face_qpoints; ++q)
          {
            const mfem::IntegrationPoint& ip = rule.IntPoint(q);
            face_transformation->Face->SetIntPoint(&ip);
            face_measure += ip.weight * face_transformation->Face->Weight();
          }
          if (!(face_measure > 0.0) || !(element_measure > 0.0))
          {
            throw LocalPhaseFailure{};
          }
          face_height[element * 4 + local_face] =
              element_measure / face_measure;
          mfem::Vector interior_shape(volume_nodes), trace_shape(face_nodes);
          mfem::Vector scaled_normal(dimension), physical_point(dimension);
          for (int q = 0; q < face_qpoints; ++q)
          {
            const mfem::IntegrationPoint& ip = rule.IntPoint(q);
            face_transformation->SetAllIntPoints(&ip);
            const mfem::IntegrationPoint& element_ip = element_one
                ? face_transformation->GetElement1IntPoint()
                : face_transformation->GetElement2IntPoint();
            finite_element->CalcShape(element_ip, interior_shape);
            trace_element->CalcShape(ip, trace_shape);
            face_transformation->Face->Transform(ip, physical_point);
            mfem::CalcOrtho(face_transformation->Face->Jacobian(),
                            scaled_normal);
            const double normal_norm = scaled_normal.Norml2();
            const double side = element_one ? 1.0 : -1.0;
            const int qindex =
                (element * 4 + local_face) * face_qpoints + q;
            face_weight[qindex] =
                ip.weight * face_transformation->Face->Weight();
            face_x[qindex] = physical_point[0];
            face_y[qindex] = physical_point[1];
            for (int a = 0; a < volume_nodes; ++a)
            {
              face_interior_shape[
                  static_cast<std::size_t>(qindex) * volume_nodes + a] =
                  interior_shape[a];
            }
            for (int a = 0; a < face_nodes; ++a)
            {
              int sign = 0;
              const int canonical = DecodeSignedDof(permutation[a], sign);
              face_oriented_shape[
                  static_cast<std::size_t>(qindex) * face_nodes + a] =
                  sign * trace_shape[canonical];
            }
            for (int direction = 0; direction < dimension; ++direction)
            {
              face_normal[static_cast<std::size_t>(qindex) * dimension +
                          direction] =
                  side * scaled_normal[direction] / normal_norm;
            }
          }
        }
      }
      if (smooth_transition())
      {
        const int projection_order =
            cpu_sc25::coupled_fixture_repeat_quadrature_order;
        projection_volume_qpoints = mfem::IntRules.Get(
            first_volume->GetGeomType(), projection_order).GetNPoints();
        projection_face_qpoints = mfem::IntRules.Get(
            first_trace->GetGeomType(), projection_order).GetNPoints();
        if (projection_volume_qpoints <= volume_qpoints ||
            projection_face_qpoints <= face_qpoints)
        {
          throw LocalPhaseFailure{};
        }
        const std::size_t projection_volume_points =
            static_cast<std::size_t>(elements) * projection_volume_qpoints;
        const std::size_t projection_face_points =
            static_cast<std::size_t>(elements) * 4 * projection_face_qpoints;
        projection_volume_weight.assign(projection_volume_points, 0.0);
        projection_volume_shape.assign(
            projection_volume_points * volume_nodes, 0.0);
        projection_volume_x.assign(projection_volume_points, 0.0);
        projection_face_weight.assign(projection_face_points, 0.0);
        projection_face_oriented_shape.assign(
            projection_face_points * face_nodes, 0.0);
        projection_face_x.assign(projection_face_points, 0.0);

        mfem::Vector projection_volume_shape_value(volume_nodes);
        mfem::Vector projection_point(dimension);
        for (int element = 0; element < elements; ++element)
        {
          const mfem::FiniteElement* finite_element =
              volume_space->GetFE(element);
          mfem::IsoparametricTransformation transformation_storage;
          mesh->GetElementTransformation(element, &transformation_storage);
          mfem::ElementTransformation* transformation =
              &transformation_storage;
          if (finite_element == nullptr) { throw LocalPhaseFailure{}; }
          const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
              finite_element->GetGeomType(), projection_order);
          if (volume_rule.GetNPoints() != projection_volume_qpoints)
          {
            throw LocalPhaseFailure{};
          }
          for (int q = 0; q < projection_volume_qpoints; ++q)
          {
            const mfem::IntegrationPoint& ip = volume_rule.IntPoint(q);
            transformation->SetIntPoint(&ip);
            finite_element->CalcShape(ip, projection_volume_shape_value);
            transformation->Transform(ip, projection_point);
            const int qindex = element * projection_volume_qpoints + q;
            projection_volume_weight[qindex] =
                ip.weight * transformation->Weight();
            projection_volume_x[qindex] = projection_point[0];
            for (int a = 0; a < volume_nodes; ++a)
            {
              projection_volume_shape[
                  static_cast<std::size_t>(qindex) * volume_nodes + a] =
                  projection_volume_shape_value[a];
            }
          }

          mfem::Array<int> faces, orientations;
          mesh->GetElementEdges(element, faces, orientations);
          for (int local_face = 0; local_face < 4; ++local_face)
          {
            const int face = faces[local_face];
            mfem::FaceElementTransformations face_transformation_storage;
            mfem::IsoparametricTransformation element_one_storage;
            mfem::IsoparametricTransformation element_two_storage;
            mesh->GetFaceElementTransformations(
                face, face_transformation_storage, element_one_storage,
                element_two_storage);
            mfem::FaceElementTransformations* face_transformation =
                &face_transformation_storage;
            const mfem::FiniteElement* trace_element =
                carrier_space->GetFaceElement(face);
            const int* permutation = trace_collection->DofOrderForOrientation(
                mfem::Geometry::SEGMENT, orientations[local_face]);
            if (trace_element == nullptr) { throw LocalPhaseFailure{}; }
            const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
                trace_element->GetGeomType(), projection_order);
            if (face_transformation == nullptr ||
                face_transformation->Face == nullptr || permutation == nullptr ||
                face_rule.GetNPoints() != projection_face_qpoints)
            {
              throw LocalPhaseFailure{};
            }
            mfem::Vector trace_shape(face_nodes), physical_point(dimension);
            for (int q = 0; q < projection_face_qpoints; ++q)
            {
              const mfem::IntegrationPoint& ip = face_rule.IntPoint(q);
              face_transformation->Face->SetIntPoint(&ip);
              trace_element->CalcShape(ip, trace_shape);
              face_transformation->Face->Transform(ip, physical_point);
              const int qindex =
                  (element * 4 + local_face) * projection_face_qpoints + q;
              projection_face_weight[qindex] =
                  ip.weight * face_transformation->Face->Weight();
              projection_face_x[qindex] = physical_point[0];
              for (int a = 0; a < face_nodes; ++a)
              {
                int sign = 0;
                const int canonical = DecodeSignedDof(permutation[a], sign);
                if (canonical < 0 || canonical >= face_nodes || sign == 0)
                {
                  throw LocalPhaseFailure{};
                }
                projection_face_oriented_shape[
                    static_cast<std::size_t>(qindex) * face_nodes + a] =
                    sign * trace_shape[canonical];
              }
            }
          }
        }
      }
      return CollectiveAll(communicator.value, true);
    }
    catch (...) { return CollectiveAll(communicator.value, false); }
  }

  bool BuildPersistentTrueSystem() noexcept
  {
    bool local_valid = true;
    try
    {
      local_pattern =
          std::make_unique<mfem::SparseMatrix>(carrier_space->GetVSize());
      mfem::DenseMatrix seed(local_trace_unknowns);
      seed = 1.0;
      for (int element = 0; element < elements; ++element)
      {
        mfem::Array<int> signed_dofs(local_trace_unknowns);
        for (int row = 0; row < local_trace_unknowns; ++row)
        {
          signed_dofs[row] = signed_trace_vdofs[
              static_cast<std::size_t>(element) * local_trace_unknowns + row];
        }
        local_pattern->AddSubMatrix(signed_dofs, signed_dofs, seed, 1);
      }
      local_pattern->Finalize();
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    local_valid = true;
    try
    {
      mfem::ParBilinearForm form(carrier_space.get());
      true_matrix.reset(form.ParallelAssemble(local_pattern.get()));
      local_valid = true_matrix &&
          true_matrix->Height() == owned_true_trace_unknowns &&
          true_matrix->Width() == owned_true_trace_unknowns &&
          true_matrix->GetGlobalNumRows() == true_trace_unknowns;
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    local_valid = true;
    try
    {
      true_matrix->GetDiag(diagonal_view);
      true_matrix->GetOffd(offdiagonal_view, offdiagonal_column_map);
      const int* diag_i = diagonal_view.GetI();
      const int* diag_j = diagonal_view.GetJ();
      const int* offd_i = offdiagonal_view.GetI();
      const int* offd_j = offdiagonal_view.GetJ();
      const int offd_columns = offdiagonal_view.Width();
      if (diag_i == nullptr || diag_j == nullptr || offd_i == nullptr ||
          (offd_columns != 0 &&
           (offd_j == nullptr || offdiagonal_column_map == nullptr)))
      {
        throw LocalPhaseFailure{};
      }

      const std::int64_t begin = owned_true_begin;
      const std::int64_t end = begin + owned_true_trace_unknowns;
      const auto find_slot = [&](const std::int64_t global_row,
                                 const std::int64_t global_column,
                                 int& section, int& target)
      {
        if (global_row < begin || global_row >= end) { return false; }
        const int row = static_cast<int>(global_row - begin);
        if (global_column >= begin && global_column < end)
        {
          const int column = static_cast<int>(global_column - begin);
          for (int entry = diag_i[row]; entry < diag_i[row + 1]; ++entry)
          {
            if (diag_j[entry] == column)
            {
              section = 0;
              target = entry;
              return true;
            }
          }
        }
        else
        {
          int column = -1;
          for (int item = 0; item < offd_columns; ++item)
          {
            if (offdiagonal_column_map[item] == global_column)
            {
              column = item;
              break;
            }
          }
          for (int entry = offd_i[row];
               column >= 0 && entry < offd_i[row + 1]; ++entry)
          {
            if (offd_j[entry] == column)
            {
              section = 1;
              target = entry;
              return true;
            }
          }
        }
        return false;
      };

      matrix_owned_source.clear();
      matrix_owned_section.clear();
      matrix_owned_target.clear();
      matrix_send_source.clear();
      matrix_send_keys.clear();
      vector_owned_source.clear();
      vector_owned_target.clear();
      vector_send_source.clear();
      vector_send_keys.clear();
      for (int element = 0; element < elements; ++element)
        for (int row = 0; row < local_trace_unknowns; ++row)
        {
          const std::size_t row_map =
              static_cast<std::size_t>(element) * local_trace_unknowns + row;
          const int global_row = local_trace_to_true[row_map];
          const int owner = local_trace_owner[row_map];
          const int vector_source = element * local_trace_unknowns + row;
          if (owner == rank)
          {
            vector_owned_source.push_back(vector_source);
            vector_owned_target.push_back(global_row - owned_true_begin);
          }
          else
          {
            vector_send_source.push_back(vector_source);
            vector_send_keys.push_back(global_row);
          }
          for (int column = 0; column < local_trace_unknowns; ++column)
          {
            const int global_column = local_trace_to_true[
                static_cast<std::size_t>(element) * local_trace_unknowns +
                column];
            const int source =
                (element * local_trace_unknowns + row) *
                    local_trace_unknowns + column;
            if (owner == rank)
            {
              int section = -1;
              int target = -1;
              if (!find_slot(global_row, global_column, section, target))
              {
                local_valid = false;
              }
              matrix_owned_source.push_back(source);
              matrix_owned_section.push_back(section);
              matrix_owned_target.push_back(target);
            }
            else
            {
              matrix_send_source.push_back(source);
              matrix_send_keys.push_back(global_row);
              matrix_send_keys.push_back(global_column);
            }
          }
        }
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    const int peer = 1 - rank;
    int matrix_receive_count = 0;
    int vector_receive_count = 0;
    int matrix_send_count = static_cast<int>(matrix_send_source.size());
    int vector_send_count = static_cast<int>(vector_send_source.size());
    const bool matrix_count_exchange = MPI_Sendrecv(
        &matrix_send_count, 1, MPI_INT, peer, 5101,
        &matrix_receive_count, 1, MPI_INT, peer, 5101,
        communicator.value, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, matrix_count_exchange))
      return false;
    const bool vector_count_exchange = MPI_Sendrecv(
        &vector_send_count, 1, MPI_INT, peer, 5102,
        &vector_receive_count, 1, MPI_INT, peer, 5102,
        communicator.value, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, vector_count_exchange))
      return false;
    local_valid = true;
    try
    {
      matrix_receive_keys.assign(2U * matrix_receive_count, -1);
      vector_receive_keys.assign(vector_receive_count, -1);
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }
    const bool matrix_key_exchange = MPI_Sendrecv(
        matrix_send_keys.data(), 2 * matrix_send_count, MPI_INT64_T, peer,
        5103, matrix_receive_keys.data(), 2 * matrix_receive_count,
        MPI_INT64_T, peer, 5103, communicator.value,
        MPI_STATUS_IGNORE) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, matrix_key_exchange))
      return false;
    const bool vector_key_exchange = MPI_Sendrecv(
        vector_send_keys.data(), vector_send_count, MPI_INT64_T, peer, 5104,
        vector_receive_keys.data(), vector_receive_count, MPI_INT64_T, peer,
        5104, communicator.value, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, vector_key_exchange))
      return false;

    local_valid = true;
    try
    {
      const std::int64_t begin = owned_true_begin;
      const std::int64_t end = begin + owned_true_trace_unknowns;
      const int* diag_i = diagonal_view.GetI();
      const int* diag_j = diagonal_view.GetJ();
      const int* offd_i = offdiagonal_view.GetI();
      const int* offd_j = offdiagonal_view.GetJ();
      const int offd_columns = offdiagonal_view.Width();
      const auto find_slot = [&](const std::int64_t global_row,
                                 const std::int64_t global_column,
                                 int& section, int& target)
      {
        if (global_row < begin || global_row >= end) { return false; }
        const int row = static_cast<int>(global_row - begin);
        if (global_column >= begin && global_column < end)
        {
          const int column = static_cast<int>(global_column - begin);
          for (int entry = diag_i[row]; entry < diag_i[row + 1]; ++entry)
            if (diag_j[entry] == column)
            { section = 0; target = entry; return true; }
        }
        else
        {
          int column = -1;
          for (int item = 0; item < offd_columns; ++item)
            if (offdiagonal_column_map[item] == global_column)
            { column = item; break; }
          for (int entry = offd_i[row];
               column >= 0 && entry < offd_i[row + 1]; ++entry)
            if (offd_j[entry] == column)
            { section = 1; target = entry; return true; }
        }
        return false;
      };
      matrix_receive_section.resize(matrix_receive_count);
      matrix_receive_target.resize(matrix_receive_count);
      for (int item = 0; item < matrix_receive_count; ++item)
      {
        if (!find_slot(matrix_receive_keys[2U * item],
                       matrix_receive_keys[2U * item + 1],
                       matrix_receive_section[item],
                       matrix_receive_target[item]))
        {
          local_valid = false;
        }
      }
      vector_receive_target.resize(vector_receive_count);
      for (int item = 0; item < vector_receive_count; ++item)
      {
        const std::int64_t global = vector_receive_keys[item];
        if (global < begin || global >= end) { local_valid = false; }
        vector_receive_target[item] = static_cast<int>(global - begin);
      }
      matrix_send_host.assign(matrix_send_count, 0.0);
      matrix_receive_host.assign(matrix_receive_count, 0.0);
      vector_send_host.assign(vector_send_count, 0.0);
      vector_receive_host.assign(vector_receive_count, 0.0);
      owned_trace_host.assign(owned_true_trace_unknowns, 0.0);
      global_trace_host.assign(true_trace_unknowns, 0.0);

      true_rhs = std::make_unique<mfem::HypreParVector>(carrier_space.get());
      true_solution =
          std::make_unique<mfem::HypreParVector>(carrier_space.get());
      *true_rhs = 0.0;
      *true_solution = 0.0;
      true_residual.SetSize(owned_true_trace_unknowns);
      true_residual.UseDevice(true);
      ilu = std::make_unique<mfem::HypreILU>();
      gmres = std::make_unique<mfem::HypreGMRES>(communicator.value);
      ConfigureIlu(*ilu);
      gmres->SetTol(relative_trace_tolerance);
      gmres->SetAbsTol(absolute_trace_tolerance);
      gmres->SetMaxIter(trace_maximum_iterations);
      gmres->SetKDim(trace_restart_dimension);
      gmres->SetLogging(1);
      gmres->SetPrintLevel(0);
      gmres->SetZeroInitialIterate();
      gmres->SetPreconditioner(*ilu);
      gmres->SetOperator(*true_matrix);
    }
    catch (...) { local_valid = false; }
    return CollectiveAll(communicator.value, local_valid);
  }

  template <typename Value>
  bool Upload(Value* destination, const std::vector<Value>& source,
              const cuda_w8a::CudaTransferSite site) noexcept
  {
    const std::size_t bytes = source.size() * sizeof(Value);
    if (bytes == 0) { return true; }
    return cuda_w8a::CudaTransportPolicy::HostToDevice(
        destination, source.data(), bytes,
        cuda_w8a::CudaTransferCategory::topology, site, bytes, bytes);
  }

  bool BindPersistentArenas() noexcept
  {
    topology_reference_hash = HashArrays(
        local_trace_to_true, local_trace_sign, local_trace_owner,
        matrix_owned_source, matrix_owned_section, matrix_owned_target,
        matrix_send_source, matrix_receive_section, matrix_receive_target,
        vector_owned_source, vector_owned_target, vector_send_source,
        vector_receive_target);
    quadrature_reference_hash = HashArrays(
        volume_weight, volume_shape, volume_derivative, volume_x, volume_y,
        face_weight, face_height, face_interior_shape, face_oriented_shape,
        face_normal, face_x, face_y, volume_node_x, volume_node_y,
        projection_volume_weight, projection_volume_shape, projection_volume_x,
        projection_face_weight, projection_face_oriented_shape,
        projection_face_x, trace_node_coordinate);
    if (topology_reference_hash == StateHash{} ||
        quadrature_reference_hash == StateHash{})
    {
      return false;
    }
    const std::size_t element_count = static_cast<std::size_t>(elements);
    const std::size_t full = static_cast<std::size_t>(full_unknowns);
    const std::size_t dense_entries = 1;
    std::size_t double_count = 0;
    const auto reserve_double = [&](const std::size_t count)
    { double_count += count; };
    reserve_double(2 * persistent_interior_unknowns);
    reserve_double(2 * persistent_trace_unknowns);
    reserve_double(2 * persistent_interior_unknowns);
    reserve_double(2 * persistent_trace_unknowns);
    reserve_double(element_count * ni * ni);
    reserve_double(element_count * ni * nt);
    reserve_double(element_count * nt * ni);
    reserve_double(element_count * nt * nt);
    reserve_double(element_count * ni);
    reserve_double(element_count * nt);
    reserve_double(element_count * ni * ni);
    reserve_double(element_count * ni);
    reserve_double(element_count * ni * nt);
    reserve_double(element_count * nt * nt);
    reserve_double(element_count * nt);
    reserve_double(element_count * ni);
    reserve_double(element_count * ni);
    reserve_double(full);
    reserve_double(dense_entries);
    reserve_double(16);
    reserve_double(persistent_interior_unknowns);
    reserve_double(persistent_trace_unknowns);
    reserve_double(persistent_trace_unknowns);
    reserve_double(matrix_send_host.size());
    reserve_double(matrix_receive_host.size());
    reserve_double(vector_send_host.size());
    reserve_double(vector_receive_host.size());
    reserve_double(full);
    reserve_double(full);
    reserve_double(element_count * nt);
    reserve_double(10 * 32 / sizeof(double));
    std::size_t quadrature_double_count = 0;
    const auto reserve_vector = [&](const auto& values)
    {
      quadrature_double_count += values.size();
      reserve_double(values.size());
    };
    reserve_vector(volume_weight);
    reserve_vector(volume_shape);
    reserve_vector(volume_derivative);
    reserve_vector(volume_x);
    reserve_vector(volume_y);
    reserve_vector(face_weight);
    reserve_vector(face_height);
    reserve_vector(face_interior_shape);
    reserve_vector(face_oriented_shape);
    reserve_vector(face_normal);
    reserve_vector(face_x);
    reserve_vector(face_y);
    reserve_vector(volume_node_x);
    reserve_vector(volume_node_y);
    reserve_vector(projection_volume_weight);
    reserve_vector(projection_volume_shape);
    reserve_vector(projection_volume_x);
    reserve_vector(projection_face_weight);
    reserve_vector(projection_face_oriented_shape);
    reserve_vector(projection_face_x);
    reserve_vector(trace_node_coordinate);

    std::size_t integer_count = 0;
    const auto reserve_integer = [&](const std::size_t count)
    { integer_count += count; };
    reserve_integer(local_trace_to_true.size());
    reserve_integer(local_trace_sign.size());
    reserve_integer(local_trace_owner.size());
    reserve_integer(matrix_owned_source.size());
    reserve_integer(matrix_owned_section.size());
    reserve_integer(matrix_owned_target.size());
    reserve_integer(matrix_send_source.size());
    reserve_integer(matrix_receive_section.size());
    reserve_integer(matrix_receive_target.size());
    reserve_integer(vector_owned_source.size());
    reserve_integer(vector_owned_target.size());
    reserve_integer(vector_send_source.size());
    reserve_integer(vector_receive_target.size());
    reserve_integer(element_count * ni);
    reserve_integer(1);
    reserve_integer(1);

    double_arena_bytes = double_count * sizeof(double);
    integer_arena_bytes = integer_count * sizeof(int);
    void* doubles = nullptr;
    void* integers = nullptr;
    const bool doubles_allocated =
        cuda_w8a::UmpireDeviceAllocatorPolicy::Allocate(
            double_arena_bytes, alignof(double),
            cuda_w8a::CudaAllocationSite::persistent_double_arena, doubles);
    const bool integers_allocated =
        cuda_w8a::UmpireDeviceAllocatorPolicy::Allocate(
            integer_arena_bytes, alignof(int),
            cuda_w8a::CudaAllocationSite::persistent_integer_arena, integers);
    if (!doubles_allocated || !integers_allocated)
    {
      if (integers != nullptr)
      {
        cuda_w8a::UmpireDeviceAllocatorPolicy::Deallocate(
            integers, cuda_w8a::CudaAllocationSite::persistent_integer_arena);
      }
      if (doubles != nullptr)
      {
        cuda_w8a::UmpireDeviceAllocatorPolicy::Deallocate(
            doubles, cuda_w8a::CudaAllocationSite::persistent_double_arena);
      }
      return false;
    }
    double_arena = doubles;
    integer_arena = integers;
    double* double_cursor = static_cast<double*>(double_arena);
    int* integer_cursor = static_cast<int*>(integer_arena);
    auto take_double = [&](const std::size_t count)
    {
      double* result = double_cursor;
      double_cursor += count;
      return result;
    };
    auto take_integer = [&](const std::size_t count)
    {
      int* result = integer_cursor;
      integer_cursor += count;
      return result;
    };

    view.rank = rank;
    view.ranks = ranks;
    view.elements = elements;
    view.scalar_true_trace_nodes = scalar_true_trace_nodes;
    view.full_unknowns = full_unknowns;
    view.true_trace_unknowns = true_trace_unknowns;
    view.volume_quadrature_points = volume_qpoints;
    view.face_quadrature_points = face_qpoints;
    view.projection_volume_quadrature_points = projection_volume_qpoints;
    view.projection_face_quadrature_points = projection_face_qpoints;
    view.true_matrix_diag_nonzeros = diagonal_view.NumNonZeroElems();
    view.true_matrix_offd_nonzeros = offdiagonal_view.NumNonZeroElems();
    view.true_matrix_nonzeros = view.true_matrix_diag_nonzeros +
        view.true_matrix_offd_nonzeros;
    view.owned_true_trace_unknowns = owned_true_trace_unknowns;
    view.owned_true_begin = owned_true_begin;
    view.matrix_owned_count = static_cast<int>(matrix_owned_source.size());
    view.matrix_send_count = static_cast<int>(matrix_send_source.size());
    view.matrix_receive_count =
        static_cast<int>(matrix_receive_target.size());
    view.vector_owned_count = static_cast<int>(vector_owned_source.size());
    view.vector_send_count = static_cast<int>(vector_send_source.size());
    view.vector_receive_count =
        static_cast<int>(vector_receive_target.size());
    view.topology_integer_count = static_cast<int>(
        local_trace_to_true.size() + local_trace_sign.size() +
        local_trace_owner.size() + matrix_owned_source.size() +
        matrix_owned_section.size() + matrix_owned_target.size() +
        matrix_send_source.size() + matrix_receive_section.size() +
        matrix_receive_target.size() + vector_owned_source.size() +
        vector_owned_target.size() + vector_send_source.size() +
        vector_receive_target.size());
    view.quadrature_double_count =
        static_cast<int>(quadrature_double_count);
    view.assemble_dense_jacobian = 0;
    view.dt = time_step();
    view.smooth_transition = smooth_transition() ? 1 : 0;
    view.slot_interior[0] = take_double(persistent_interior_unknowns);
    view.slot_interior[1] = take_double(persistent_interior_unknowns);
    view.slot_trace[0] = take_double(persistent_trace_unknowns);
    view.slot_trace[1] = take_double(persistent_trace_unknowns);
    view.trial_interior = take_double(persistent_interior_unknowns);
    view.direction_interior = take_double(persistent_interior_unknowns);
    view.trial_trace = take_double(persistent_trace_unknowns);
    view.direction_trace = take_double(persistent_trace_unknowns);
    view.a = take_double(element_count * ni * ni);
    view.b = take_double(element_count * ni * nt);
    view.c = take_double(element_count * nt * ni);
    view.d = take_double(element_count * nt * nt);
    view.local_residual = take_double(element_count * ni);
    view.trace_residual = take_double(element_count * nt);
    view.lu = take_double(element_count * ni * ni);
    view.inverse_residual = take_double(element_count * ni);
    view.inverse_b = take_double(element_count * ni * nt);
    view.schur = take_double(element_count * nt * nt);
    view.condensed_rhs = take_double(element_count * nt);
    view.recovery_rhs = take_double(element_count * ni);
    view.recovered = take_double(element_count * ni);
    view.full_residual = take_double(full);
    view.full_jacobian = take_double(dense_entries);
    view.reduction = take_double(16);
    view.checkpoint_interior_slabs =
        take_double(persistent_interior_unknowns);
    view.checkpoint_trace_slabs = take_double(persistent_trace_unknowns);
    view.distributed_global_trace = take_double(persistent_trace_unknowns);
    view.remote_matrix_send = take_double(matrix_send_host.size());
    view.remote_matrix_receive = take_double(matrix_receive_host.size());
    view.remote_vector_send = take_double(vector_send_host.size());
    view.remote_vector_receive = take_double(vector_receive_host.size());
    view.evidence_action_input = take_double(full);
    view.evidence_action_result = take_double(full);
    view.evidence_trace_action = take_double(element_count * nt);
    view.state_hash_bytes = reinterpret_cast<std::uint8_t*>(
        take_double(10 * 32 / sizeof(double)));
    view.volume_weight = take_double(volume_weight.size());
    view.volume_shape = take_double(volume_shape.size());
    view.volume_derivative = take_double(volume_derivative.size());
    view.volume_x = take_double(volume_x.size());
    view.volume_y = take_double(volume_y.size());
    view.face_weight = take_double(face_weight.size());
    view.face_height = take_double(face_height.size());
    view.face_interior_shape = take_double(face_interior_shape.size());
    view.face_oriented_shape = take_double(face_oriented_shape.size());
    view.face_normal = take_double(face_normal.size());
    view.face_x = take_double(face_x.size());
    view.face_y = take_double(face_y.size());
    view.volume_node_x = take_double(volume_node_x.size());
    view.volume_node_y = take_double(volume_node_y.size());
    view.projection_volume_weight =
        take_double(projection_volume_weight.size());
    view.projection_volume_shape =
        take_double(projection_volume_shape.size());
    view.projection_volume_x = take_double(projection_volume_x.size());
    view.projection_face_weight = take_double(projection_face_weight.size());
    view.projection_face_oriented_shape =
        take_double(projection_face_oriented_shape.size());
    view.projection_face_x = take_double(projection_face_x.size());
    view.trace_node_coordinate = take_double(trace_node_coordinate.size());
    view.local_trace_to_true = take_integer(local_trace_to_true.size());
    view.local_trace_sign = take_integer(local_trace_sign.size());
    view.local_trace_owner = take_integer(local_trace_owner.size());
    view.matrix_owned_source = take_integer(matrix_owned_source.size());
    view.matrix_owned_section = take_integer(matrix_owned_section.size());
    view.matrix_owned_target = take_integer(matrix_owned_target.size());
    view.matrix_send_source = take_integer(matrix_send_source.size());
    view.matrix_receive_section =
        take_integer(matrix_receive_section.size());
    view.matrix_receive_target =
        take_integer(matrix_receive_target.size());
    view.vector_owned_source = take_integer(vector_owned_source.size());
    view.vector_owned_target = take_integer(vector_owned_target.size());
    view.vector_send_source = take_integer(vector_send_source.size());
    view.vector_receive_target =
        take_integer(vector_receive_target.size());
    view.row_pivots = take_integer(element_count * ni);
    view.status = take_integer(1);
    view.compiled_architecture = take_integer(1);
    if (double_cursor != static_cast<double*>(double_arena) + double_count ||
        integer_cursor != static_cast<int*>(integer_arena) + integer_count)
    {
      return false;
    }
    true_matrix->HypreReadWrite();
    hypre_ParCSRMatrix* raw = *true_matrix;
    auto* diag = hypre_ParCSRMatrixDiag(raw);
    auto* offd = hypre_ParCSRMatrixOffd(raw);
    view.true_matrix_data = hypre_CSRMatrixData(diag);
    view.true_matrix_offd_data = hypre_CSRMatrixData(offd);
    view.true_rhs = true_rhs->Write();
    view.true_solution = true_solution->Read();
    bool uploads = true;
    const auto upload = [&](auto* destination, const auto& source,
                            const cuda_w8a::CudaTransferSite site)
    {
      const bool uploaded = Upload(destination, source, site);
      uploads = uploaded && uploads;
    };
    upload(view.local_trace_to_true, local_trace_to_true,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.local_trace_sign, local_trace_sign,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.local_trace_owner, local_trace_owner,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.matrix_owned_source, matrix_owned_source,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.matrix_owned_section, matrix_owned_section,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.matrix_owned_target, matrix_owned_target,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.matrix_send_source, matrix_send_source,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.matrix_receive_section, matrix_receive_section,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.matrix_receive_target, matrix_receive_target,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.vector_owned_source, vector_owned_source,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.vector_owned_target, vector_owned_target,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.vector_send_source, vector_send_source,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.vector_receive_target, vector_receive_target,
           cuda_w8a::CudaTransferSite::map_upload);
    upload(view.volume_weight, volume_weight,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.volume_shape, volume_shape,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.volume_derivative, volume_derivative,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.volume_x, volume_x,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.volume_y, volume_y,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.face_weight, face_weight,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.face_height, face_height,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.face_interior_shape, face_interior_shape,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.face_oriented_shape, face_oriented_shape,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.face_normal, face_normal,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.face_x, face_x,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.face_y, face_y,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.volume_node_x, volume_node_x,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.volume_node_y, volume_node_y,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.projection_volume_weight, projection_volume_weight,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.projection_volume_shape, projection_volume_shape,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.projection_volume_x, projection_volume_x,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.projection_face_weight, projection_face_weight,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.projection_face_oriented_shape,
           projection_face_oriented_shape,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.projection_face_x, projection_face_x,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    upload(view.trace_node_coordinate, trace_node_coordinate,
           cuda_w8a::CudaTransferSite::quadrature_upload);
    const bool uploads_synchronized =
        cuda_w8a::CudaTransportPolicy::Synchronize(
            cuda_w8a::CudaTransferSite::quadrature_upload);
    uploads = uploads_synchronized && uploads;
    return uploads && view.true_matrix_data != nullptr &&
        (view.true_matrix_offd_nonzeros == 0 ||
         view.true_matrix_offd_data != nullptr) &&
        view.true_rhs != nullptr && view.true_solution != nullptr;
  }

  bool ReadKernelStatus() noexcept
  {
    int device_status = detail::device_invalid_layout;
    int architecture = 0;
    bool copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
        &device_status, view.status, sizeof(device_status),
        cuda_w8a::CudaTransferCategory::scalar_diagnostic,
        cuda_w8a::CudaTransferSite::scalar_status);
    const bool architecture_copied =
        cuda_w8a::CudaTransportPolicy::DeviceToHost(
            &architecture, view.compiled_architecture, sizeof(architecture),
            cuda_w8a::CudaTransferCategory::scalar_diagnostic,
            cuda_w8a::CudaTransferSite::scalar_status);
    copied = copied && architecture_copied;
    const bool synchronized =
        cuda_w8a::CudaTransportPolicy::Synchronize(
            cuda_w8a::CudaTransferSite::scalar_status);
    copied = copied && synchronized;
    evidence.launches.last_compiled_architecture = architecture;
    last_device_status = copied ? device_status : detail::device_invalid_layout;
    evidence.launches.all_launches_completed =
        evidence.launches.all_launches_completed && copied &&
        device_status == detail::device_success &&
        architecture == required_native_architecture;
    return copied && device_status == detail::device_success &&
        architecture == required_native_architecture;
  }

  bool PrepareKernelStatus(const cuda_w8a::CudaFailurePhase phase) noexcept
  {
    view.committed_slot = committed_slot;
    const int clear = detail::device_success;
    const int architecture = 0;
    const std::uint64_t operation_generation = pending_attempt_active
        ? pending_attempt.candidate_coordinates.generation
        : coordinates.generation;
    cuda_w8a::CudaTransportPolicy::SetPhase(phase, operation_generation);
    bool prepared = cuda_w8a::CudaTransportPolicy::HostToDevice(
               view.status, &clear, sizeof(clear),
               cuda_w8a::CudaTransferCategory::scalar_diagnostic,
               cuda_w8a::CudaTransferSite::scalar_status);
    const bool architecture_prepared =
        cuda_w8a::CudaTransportPolicy::HostToDevice(
            view.compiled_architecture, &architecture, sizeof(architecture),
            cuda_w8a::CudaTransferCategory::scalar_diagnostic,
            cuda_w8a::CudaTransferSite::scalar_status);
    prepared = prepared && architecture_prepared;
    return prepared;
  }

  bool ReconcileKernelOutcome(const bool local_success) noexcept
  {
    int local_status = last_device_status;
    if (!local_success && local_status == detail::device_success)
      local_status = detail::device_invalid_layout;
    std::array<int, 2> rank_status{{detail::device_invalid_layout,
                                    detail::device_invalid_layout}};
    const bool gathered = MPI_Allgather(
        &local_status, 1, MPI_INT, rank_status.data(), 1, MPI_INT,
        communicator.value) == MPI_SUCCESS;
    if (!gathered)
    {
      last_device_status = detail::device_invalid_layout;
      evidence.launches.all_launches_completed = false;
      return false;
    }
    last_device_status = detail::device_success;
    for (const int status : rank_status)
    {
      if (status != detail::device_success)
      {
        last_device_status = status;
        break;
      }
    }
    const bool collective_success =
        last_device_status == detail::device_success;
    evidence.launches.all_launches_completed =
        evidence.launches.all_launches_completed && collective_success;
    return collective_success;
  }

  ScalarCudaFailureCause LastDeviceFailureCause() const noexcept
  {
    if ((last_device_status & detail::device_invalid_layout) != 0)
      return ScalarCudaFailureCause::device_invalid_layout;
    if ((last_device_status & detail::device_nonfinite) != 0)
      return ScalarCudaFailureCause::device_nonfinite;
    if ((last_device_status & detail::device_singular_local_block) != 0)
      return ScalarCudaFailureCause::device_local_singularity;
    if ((last_device_status & detail::device_local_application) != 0)
      return ScalarCudaFailureCause::device_local_application;
    if ((last_device_status & detail::device_admissibility) != 0)
      return ScalarCudaFailureCause::device_state_admissibility;
    return ScalarCudaFailureCause::transfer;
  }

  bool LastDeviceFailureIsAdmissibility() const noexcept
  {
    return last_device_status == detail::device_admissibility;
  }

  ScalarCudaStatus LastAssemblyFailureStatus() const noexcept
  {
    return LastDeviceFailureIsAdmissibility()
        ? ScalarCudaStatus::admissibility_failure
        : ScalarCudaStatus::assembly_failure;
  }

  bool AssembleCollectiveResidual() noexcept
  {
    view.operation = static_cast<int>(
        detail::KernelOperation::gather_trace_residual);
    view.true_rhs = true_rhs->Write();
    const bool prepared =
        PrepareKernelStatus(cuda_w8a::CudaFailurePhase::diagnostics);
    if (!CollectiveAll(communicator.value, prepared)) { return false; }
    HdgSc25Rank2CondenseFp64Kernel<<<1, 1>>>(view);
    const bool kernel_launched = cudaGetLastError() == cudaSuccess;
    const bool status_read = ReadKernelStatus();
    if (!ReconcileKernelOutcome(kernel_launched && status_read)) return false;
    const std::size_t send_bytes = vector_send_host.size() * sizeof(double);
    const std::size_t receive_bytes =
        vector_receive_host.size() * sizeof(double);
    bool staged_out = send_bytes == 0 ||
        cuda_w8a::CudaTransportPolicy::DeviceToHost(
             vector_send_host.data(), view.remote_vector_send, send_bytes,
             cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
             cuda_w8a::CudaTransferSite::rhs_remote_gather,
             send_bytes, send_bytes);
    const bool synchronized = cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::rhs_remote_gather);
    staged_out = staged_out && synchronized;
    if (!CollectiveAll(communicator.value, staged_out)) { return false; }
    const int peer = 1 - rank;
    const bool exchanged = MPI_Sendrecv(
        vector_send_host.data(), static_cast<int>(vector_send_host.size()),
        MPI_DOUBLE, peer, 5199, vector_receive_host.data(),
        static_cast<int>(vector_receive_host.size()), MPI_DOUBLE, peer, 5199,
        communicator.value, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, exchanged)) { return false; }
    const bool recorded = cuda_w8a::CudaTransportPolicy::RecordMpiStaging(
        send_bytes + receive_bytes,
        cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
        cuda_w8a::CudaTransferSite::rhs_remote_gather,
        vector_send_host.data(), vector_receive_host.data(),
        send_bytes, receive_bytes, exchanged);
    if (!CollectiveAll(communicator.value, recorded)) { return false; }
    const bool staged_in = receive_bytes == 0 ||
        cuda_w8a::CudaTransportPolicy::HostToDevice(
            view.remote_vector_receive, vector_receive_host.data(),
            receive_bytes,
            cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
            cuda_w8a::CudaTransferSite::rhs_remote_scatter,
            receive_bytes, receive_bytes);
    if (!CollectiveAll(communicator.value, staged_in)) { return false; }
    view.operation = static_cast<int>(
        detail::KernelOperation::scatter_remote_trace_residual);
    const bool scatter_prepared =
        PrepareKernelStatus(cuda_w8a::CudaFailurePhase::diagnostics);
    if (!CollectiveAll(communicator.value, scatter_prepared)) { return false; }
    HdgSc25Rank2CondenseFp64Kernel<<<1, 1>>>(view);
    const bool scatter_launched = cudaGetLastError() == cudaSuccess;
    const bool scatter_status_read = ReadKernelStatus();
    return ReconcileKernelOutcome(scatter_launched && scatter_status_read);
  }

  bool LaunchAssemble(const detail::KernelOperation operation,
                      const cuda_w8a::CudaFailurePhase phase) noexcept
  {
    view.operation = static_cast<int>(operation);
    const bool prepared = PrepareKernelStatus(phase);
    if (!CollectiveAll(communicator.value, prepared)) { return false; }
    HdgSc25Rank2AssembleLocalFp64Kernel<<<1, 1>>>(view);
    ++evidence.launches.assemble_launches;
    const bool kernel_launched = cudaGetLastError() == cudaSuccess;
    const bool status_read = ReadKernelStatus();
    if (!ReconcileKernelOutcome(kernel_launched && status_read)) return false;
    if (operation == detail::KernelOperation::assemble)
    {
      return AssembleCollectiveResidual();
    }
    return true;
  }

  bool CaptureOperatorEvidence(ScalarCudaSnapshotEvidence& output) noexcept
  {
    output = {};
    output.full_vector_size = persistent_interior_unknowns +
        owned_true_trace_unknowns;
    output.owned_trace_size = owned_true_trace_unknowns;
    view.operation =
        static_cast<int>(detail::KernelOperation::fixed_jacobian_action);
    const bool prepared =
        PrepareKernelStatus(cuda_w8a::CudaFailurePhase::diagnostics);
    if (!CollectiveAll(communicator.value, prepared)) { return false; }
    HdgSc25Rank2CondenseFp64Kernel<<<1, 1>>>(view);
    ++evidence.launches.condense_launches;
    const bool launched = cudaGetLastError() == cudaSuccess;
    const bool status_read = ReadKernelStatus();
    if (!ReconcileKernelOutcome(launched && status_read)) { return false; }

    const std::size_t send_bytes = vector_send_host.size() * sizeof(double);
    const std::size_t receive_bytes =
        vector_receive_host.size() * sizeof(double);
    bool staged_out = send_bytes == 0 ||
        cuda_w8a::CudaTransportPolicy::DeviceToHost(
            vector_send_host.data(), view.remote_vector_send, send_bytes,
            cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
            cuda_w8a::CudaTransferSite::rhs_remote_gather,
            send_bytes, send_bytes);
    staged_out = staged_out && cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::rhs_remote_gather);
    if (!CollectiveAll(communicator.value, staged_out)) { return false; }
    const int peer = 1 - rank;
    const bool exchanged = MPI_Sendrecv(
        vector_send_host.data(), static_cast<int>(vector_send_host.size()),
        MPI_DOUBLE, peer, 5203, vector_receive_host.data(),
        static_cast<int>(vector_receive_host.size()), MPI_DOUBLE, peer, 5203,
        communicator.value, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, exchanged)) { return false; }
    const bool recorded = cuda_w8a::CudaTransportPolicy::RecordMpiStaging(
        send_bytes + receive_bytes,
        cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
        cuda_w8a::CudaTransferSite::rhs_remote_gather,
        vector_send_host.data(), vector_receive_host.data(),
        send_bytes, receive_bytes, exchanged);
    if (!CollectiveAll(communicator.value, recorded)) { return false; }
    const bool staged_in = receive_bytes == 0 ||
        cuda_w8a::CudaTransportPolicy::HostToDevice(
            view.remote_vector_receive, vector_receive_host.data(),
            receive_bytes,
            cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
            cuda_w8a::CudaTransferSite::rhs_remote_scatter,
            receive_bytes, receive_bytes);
    if (!CollectiveAll(communicator.value, staged_in)) { return false; }
    view.operation = static_cast<int>(
        detail::KernelOperation::scatter_remote_fixed_action);
    const bool scatter_prepared =
        PrepareKernelStatus(cuda_w8a::CudaFailurePhase::diagnostics);
    if (!CollectiveAll(communicator.value, scatter_prepared)) { return false; }
    HdgSc25Rank2CondenseFp64Kernel<<<1, 1>>>(view);
    ++evidence.launches.remote_system_exchange_launches;
    const bool scatter_launched = cudaGetLastError() == cudaSuccess;
    const bool scatter_status_read = ReadKernelStatus();
    if (!ReconcileKernelOutcome(scatter_launched && scatter_status_read))
      return false;

    const std::size_t interior_bytes =
        static_cast<std::size_t>(persistent_interior_unknowns) *
        sizeof(double);
    const std::size_t owned_bytes =
        static_cast<std::size_t>(owned_true_trace_unknowns) * sizeof(double);
    const std::size_t full_bytes = interior_bytes + owned_bytes;
    const double* entry_interior =
        view.slot_interior[view.selected_current_slot];
    const double* entry_trace =
        view.slot_trace[view.selected_current_slot] + owned_true_begin;
    const auto copy = [](double* destination, const double* source,
                         const std::size_t bytes)
    {
      return bytes == 0 || cuda_w8a::CudaTransportPolicy::DeviceToHost(
          destination, source, bytes,
          cuda_w8a::CudaTransferCategory::oracle_export,
          cuda_w8a::CudaTransferSite::oracle_export, bytes, bytes);
    };
    bool copied =
        copy(output.nonlinear_residual.data(), view.local_residual,
             interior_bytes) &&
        copy(output.nonlinear_residual.data() +
                 persistent_interior_unknowns,
             view.true_rhs, owned_bytes) &&
        copy(output.jacobian_action_input.data(), view.evidence_action_input,
             full_bytes) &&
        copy(output.jacobian_action_result.data(), view.evidence_action_result,
             full_bytes) &&
        copy(output.entry_interior.data(), entry_interior, interior_bytes) &&
        copy(output.entry_trace.data(), entry_trace, owned_bytes);
    copied = copied && cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::oracle_export);
    return CollectiveAll(communicator.value, copied);
  }

  bool RunPartialPivotContract() noexcept
  {
    view.operation =
        static_cast<int>(detail::KernelOperation::partial_pivot_contract);
    const bool prepared =
        PrepareKernelStatus(cuda_w8a::CudaFailurePhase::diagnostics);
    if (!CollectiveAll(communicator.value, prepared)) { return false; }
    HdgSc25Rank2CondenseFp64Kernel<<<1, 1>>>(view);
    ++evidence.launches.condense_launches;
    const bool launched = cudaGetLastError() == cudaSuccess;
    const bool status_read = ReadKernelStatus();
    if (!ReconcileKernelOutcome(launched && status_read)) { return false; }
    std::array<double, 16> metrics{};
    if (!CopyReduction(metrics)) { return false; }
    evidence.row_partial_pivot_contract = metrics[12] == 1.0;
    evidence.nonzero_small_pivot_contract = metrics[13] == 1.0;
    return CollectiveAll(communicator.value,
                         evidence.row_partial_pivot_contract &&
                             evidence.nonzero_small_pivot_contract);
  }

  bool CaptureCondensedRightHandSide(
      ScalarCudaSnapshotEvidence& output) noexcept
  {
    const std::size_t bytes =
        static_cast<std::size_t>(owned_true_trace_unknowns) * sizeof(double);
    bool copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
        output.condensed_right_hand_side.data(), view.true_rhs, bytes,
        cuda_w8a::CudaTransferCategory::oracle_export,
        cuda_w8a::CudaTransferSite::oracle_export, bytes, bytes);
    copied = copied && cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::oracle_export);
    return CollectiveAll(communicator.value, copied);
  }

  bool CaptureTrueTraceSolution(
      ScalarCudaSnapshotEvidence& output) noexcept
  {
    const std::size_t bytes =
        static_cast<std::size_t>(owned_true_trace_unknowns) * sizeof(double);
    bool copied = true;
    try
    {
      copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
          output.true_trace_solution.data(), true_solution->Read(), bytes,
          cuda_w8a::CudaTransferCategory::oracle_export,
          cuda_w8a::CudaTransferSite::oracle_export, bytes, bytes);
    }
    catch (...) { copied = false; }
    copied = copied && cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::oracle_export);
    return CollectiveAll(communicator.value, copied);
  }

  bool CaptureRecoveredIncrement(
      ScalarCudaSnapshotEvidence& output) noexcept
  {
    const std::size_t interior_bytes =
        static_cast<std::size_t>(persistent_interior_unknowns) *
        sizeof(double);
    const std::size_t owned_bytes =
        static_cast<std::size_t>(owned_true_trace_unknowns) * sizeof(double);
    bool copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
        output.recovered_interior.data(), view.direction_interior,
        interior_bytes, cuda_w8a::CudaTransferCategory::oracle_export,
        cuda_w8a::CudaTransferSite::oracle_export,
        interior_bytes, interior_bytes);
    copied = copied && cuda_w8a::CudaTransportPolicy::DeviceToHost(
        output.recovered_trace.data(),
        view.direction_trace + owned_true_begin, owned_bytes,
        cuda_w8a::CudaTransferCategory::oracle_export,
        cuda_w8a::CudaTransferSite::oracle_export, owned_bytes, owned_bytes);
    copied = copied && cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::oracle_export);
    return CollectiveAll(communicator.value, copied);
  }

  bool LaunchCondense() noexcept
  {
    view.operation = static_cast<int>(detail::KernelOperation::condense_and_scatter);
    bool pointers_ready = true;
    try
    {
      true_matrix->HypreReadWrite();
      hypre_ParCSRMatrix* raw = *true_matrix;
      auto* diag = hypre_ParCSRMatrixDiag(raw);
      auto* offd = hypre_ParCSRMatrixOffd(raw);
      view.true_matrix_data = hypre_CSRMatrixData(diag);
      view.true_matrix_offd_data = hypre_CSRMatrixData(offd);
      view.true_rhs = true_rhs->Write();
      pointers_ready = view.true_matrix_data != nullptr &&
          (view.true_matrix_offd_nonzeros == 0 ||
           view.true_matrix_offd_data != nullptr) && view.true_rhs != nullptr;
    }
    catch (...) { pointers_ready = false; }
    if (!CollectiveAll(communicator.value, pointers_ready)) { return false; }
    const bool prepared =
        PrepareKernelStatus(cuda_w8a::CudaFailurePhase::condensation);
    if (!CollectiveAll(communicator.value, prepared)) { return false; }
    HdgSc25Rank2CondenseFp64Kernel<<<1, 1>>>(view);
    ++evidence.launches.condense_launches;
    const bool kernel_launched = cudaGetLastError() == cudaSuccess;
    const bool status_read = ReadKernelStatus();
    if (!ReconcileKernelOutcome(kernel_launched && status_read)) return false;

    const std::size_t matrix_send_bytes =
        matrix_send_host.size() * sizeof(double);
    const std::size_t matrix_receive_bytes =
        matrix_receive_host.size() * sizeof(double);
    const std::size_t vector_send_bytes =
        vector_send_host.size() * sizeof(double);
    const std::size_t vector_receive_bytes =
        vector_receive_host.size() * sizeof(double);
    const bool matrix_staged_out = matrix_send_bytes == 0 ||
         cuda_w8a::CudaTransportPolicy::DeviceToHost(
             matrix_send_host.data(), view.remote_matrix_send,
             matrix_send_bytes,
             cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
             cuda_w8a::CudaTransferSite::matrix_remote_gather,
             matrix_send_bytes, matrix_send_bytes);
    if (!CollectiveAll(communicator.value, matrix_staged_out)) { return false; }
    const bool vector_staged_out = vector_send_bytes == 0 ||
         cuda_w8a::CudaTransportPolicy::DeviceToHost(
             vector_send_host.data(), view.remote_vector_send,
             vector_send_bytes,
             cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
             cuda_w8a::CudaTransferSite::rhs_remote_gather,
             vector_send_bytes, vector_send_bytes);
    if (!CollectiveAll(communicator.value, vector_staged_out)) { return false; }
    const bool staged_out_synchronized =
        cuda_w8a::CudaTransportPolicy::Synchronize(
            cuda_w8a::CudaTransferSite::rhs_remote_gather);
    if (!CollectiveAll(communicator.value, staged_out_synchronized))
      return false;
    const int peer = 1 - rank;
    const bool matrix_exchanged = MPI_Sendrecv(
        matrix_send_host.data(), static_cast<int>(matrix_send_host.size()),
        MPI_DOUBLE, peer, 5201, matrix_receive_host.data(),
        static_cast<int>(matrix_receive_host.size()), MPI_DOUBLE, peer, 5201,
        communicator.value, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, matrix_exchanged)) { return false; }
    const bool matrix_exchange_recorded =
        cuda_w8a::CudaTransportPolicy::RecordMpiStaging(
            matrix_send_bytes + matrix_receive_bytes,
            cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
            cuda_w8a::CudaTransferSite::matrix_remote_gather,
            matrix_send_host.data(), matrix_receive_host.data(),
            matrix_send_bytes, matrix_receive_bytes, matrix_exchanged);
    if (!CollectiveAll(communicator.value, matrix_exchange_recorded))
      return false;
    const bool vector_exchanged = MPI_Sendrecv(
        vector_send_host.data(), static_cast<int>(vector_send_host.size()),
        MPI_DOUBLE, peer, 5202, vector_receive_host.data(),
        static_cast<int>(vector_receive_host.size()), MPI_DOUBLE, peer, 5202,
        communicator.value, MPI_STATUS_IGNORE) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, vector_exchanged)) { return false; }
    const bool vector_exchange_recorded =
        cuda_w8a::CudaTransportPolicy::RecordMpiStaging(
            vector_send_bytes + vector_receive_bytes,
            cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
            cuda_w8a::CudaTransferSite::rhs_remote_gather,
            vector_send_host.data(), vector_receive_host.data(),
            vector_send_bytes, vector_receive_bytes, vector_exchanged);
    if (!CollectiveAll(communicator.value, vector_exchange_recorded))
      return false;
    const bool matrix_staged_in = matrix_receive_bytes == 0 ||
         cuda_w8a::CudaTransportPolicy::HostToDevice(
             view.remote_matrix_receive, matrix_receive_host.data(),
             matrix_receive_bytes,
             cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
             cuda_w8a::CudaTransferSite::matrix_remote_scatter,
             matrix_receive_bytes, matrix_receive_bytes);
    if (!CollectiveAll(communicator.value, matrix_staged_in)) { return false; }
    const bool vector_staged_in = vector_receive_bytes == 0 ||
         cuda_w8a::CudaTransportPolicy::HostToDevice(
             view.remote_vector_receive, vector_receive_host.data(),
             vector_receive_bytes,
             cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
             cuda_w8a::CudaTransferSite::rhs_remote_scatter,
             vector_receive_bytes, vector_receive_bytes);
    if (!CollectiveAll(communicator.value, vector_staged_in)) { return false; }
    view.operation =
        static_cast<int>(detail::KernelOperation::scatter_remote_system);
    const bool scatter_prepared =
        PrepareKernelStatus(cuda_w8a::CudaFailurePhase::condensation);
    if (!CollectiveAll(communicator.value, scatter_prepared)) { return false; }
    HdgSc25Rank2CondenseFp64Kernel<<<1, 1>>>(view);
    ++evidence.launches.remote_system_exchange_launches;
    const bool scatter_launched = cudaGetLastError() == cudaSuccess;
    const bool scatter_status_read = ReadKernelStatus();
    if (!ReconcileKernelOutcome(scatter_launched && scatter_status_read))
      return false;
    bool published = true;
    try
    {
      true_matrix->HypreRead();
      true_rhs->HypreRead();
    }
    catch (...) { published = false; }
    if (!CollectiveAll(communicator.value, published)) { return false; }
    ++factor_generation;
    return true;
  }

  bool StageOwnedTrace(const double* owned_device) noexcept
  {
    const std::size_t owned_bytes =
        static_cast<std::size_t>(owned_true_trace_unknowns) * sizeof(double);
    bool staged_out = cuda_w8a::CudaTransportPolicy::DeviceToHost(
            owned_trace_host.data(), owned_device, owned_bytes,
            cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
            cuda_w8a::CudaTransferSite::trace_solution_stage_out,
            owned_bytes, owned_bytes);
    const bool stage_out_synchronized =
        cuda_w8a::CudaTransportPolicy::Synchronize(
            cuda_w8a::CudaTransferSite::trace_solution_stage_out);
    staged_out = staged_out && stage_out_synchronized;
    if (!CollectiveAll(communicator.value, staged_out)) { return false; }
    const std::array<int, 2> counts{{
        cpu_sc25::persistent_rank2_rank0_owned_trace_unknowns,
        cpu_sc25::persistent_rank2_rank1_owned_trace_unknowns}};
    const std::array<int, 2> displacements{{0, counts[0]}};
    const bool gathered = MPI_Allgatherv(
        owned_trace_host.data(), owned_true_trace_unknowns, MPI_DOUBLE,
        global_trace_host.data(), counts.data(), displacements.data(),
        MPI_DOUBLE, communicator.value) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, gathered)) { return false; }
    const bool gather_recorded =
        cuda_w8a::CudaTransportPolicy::RecordMpiStaging(
            global_trace_host.size() * sizeof(double),
            cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
            cuda_w8a::CudaTransferSite::trace_ghost_stage_in,
            owned_trace_host.data(), global_trace_host.data(), owned_bytes,
            global_trace_host.size() * sizeof(double), gathered);
    if (!CollectiveAll(communicator.value, gather_recorded)) { return false; }
    const std::size_t global_bytes =
        global_trace_host.size() * sizeof(double);
    bool staged_in = cuda_w8a::CudaTransportPolicy::HostToDevice(
            view.distributed_global_trace, global_trace_host.data(),
            global_bytes,
            cuda_w8a::CudaTransferCategory::host_staged_trace_exchange,
            cuda_w8a::CudaTransferSite::trace_ghost_stage_in,
            global_bytes, global_bytes);
    const bool stage_in_synchronized =
        cuda_w8a::CudaTransportPolicy::Synchronize(
            cuda_w8a::CudaTransferSite::trace_ghost_stage_in);
    staged_in = staged_in && stage_in_synchronized;
    if (!CollectiveAll(communicator.value, staged_in)) { return false; }
    ++evidence.launches.owned_trace_distribution_launches;
    return true;
  }

  bool LaunchRecover(const detail::KernelOperation operation,
                     const cuda_w8a::CudaFailurePhase phase) noexcept
  {
    view.operation = static_cast<int>(operation);
    if (operation == detail::KernelOperation::recover_direction)
    {
      const double* owned_device = true_solution->Read();
      if (!StageOwnedTrace(owned_device)) { return false; }
      view.true_solution = view.distributed_global_trace;
    }
    const bool prepared = PrepareKernelStatus(phase);
    if (!CollectiveAll(communicator.value, prepared)) { return false; }
    HdgSc25Rank2RecoverFp64Kernel<<<1, 1>>>(view);
    ++evidence.launches.recover_launches;
    const bool kernel_launched = cudaGetLastError() == cudaSuccess;
    const bool status_read = ReadKernelStatus();
    return ReconcileKernelOutcome(kernel_launched && status_read);
  }

  bool CopyReduction(std::array<double, 16>& output) noexcept
  {
    bool copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
               output.data(), view.reduction,
               output.size() * sizeof(double),
               cuda_w8a::CudaTransferCategory::scalar_diagnostic,
               cuda_w8a::CudaTransferSite::scalar_status);
    const bool synchronized = cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::scalar_status);
    copied = copied && synchronized;
    if (!CollectiveAll(communicator.value, copied)) { return false; }
    std::array<double, 4> application_pairs{};
    const bool pairs_gathered = MPI_Allgather(
        output.data() + 6, 2, MPI_DOUBLE, application_pairs.data(), 2,
        MPI_DOUBLE, communicator.value) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, pairs_gathered)) { return false; }
    std::array<double, 16> sum{};
    std::array<double, 16> minimum{};
    std::array<double, 16> maximum{};
    const bool summed = MPI_Allreduce(output.data(), sum.data(), 16,
                                     MPI_DOUBLE, MPI_SUM,
                                     communicator.value) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, summed)) { return false; }
    const bool minimized = MPI_Allreduce(output.data(), minimum.data(), 16,
                                        MPI_DOUBLE, MPI_MIN,
                                        communicator.value) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, minimized)) { return false; }
    const bool maximized = MPI_Allreduce(output.data(), maximum.data(), 16,
                                        MPI_DOUBLE, MPI_MAX,
                                        communicator.value) == MPI_SUCCESS;
    if (!CollectiveAll(communicator.value, maximized)) { return false; }
    output[0] = std::sqrt(std::max(0.0, sum[0]));
    output[1] = minimum[1];
    output[2] = minimum[2];
    output[3] = minimum[3];
    output[4] = maximum[4];
    output[5] = std::sqrt(std::max(0.0, sum[5]));
    int selected_pair = 0;
    double selected_ratio = -1.0;
    for (int item = 0; item < 2; ++item)
    {
      const double residual = application_pairs[2 * item];
      const double rhs_norm = application_pairs[2 * item + 1];
      const double ratio = residual / (1.0e-12 + 1.0e-11 * rhs_norm);
      if (ratio > selected_ratio)
      {
        selected_ratio = ratio;
        selected_pair = item;
      }
    }
    output[6] = application_pairs[2 * selected_pair];
    output[7] = application_pairs[2 * selected_pair + 1];
    for (int component = 0; component < state_components; ++component)
    {
      output[8 + component] = sum[8 + component];
    }
    return true;
  }

  bool ComputeStateHashes(const int slot, StateHashes& output) noexcept
  {
    output = {};
    if (!CollectiveAll(communicator.value, slot >= 0 && slot <= 1))
      return false;
    view.selected_current_slot = slot;
    if (!LaunchRecover(detail::KernelOperation::hash_state,
                       cuda_w8a::CudaFailurePhase::diagnostics))
    {
      return false;
    }
    std::array<std::uint8_t, 8 * 32> bytes{};
    bool copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
            bytes.data(), view.state_hash_bytes, bytes.size(),
            cuda_w8a::CudaTransferCategory::scalar_diagnostic,
            cuda_w8a::CudaTransferSite::state_hash,
            bytes.size(), bytes.size());
    const bool synchronized = cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::state_hash);
    copied = copied && synchronized;
    if (!CollectiveAll(communicator.value, copied)) { return false; }
    const auto copy = [&bytes](StateHash& hash, const int index)
    {
      std::copy_n(bytes.begin() + index * 32, 32, hash.bytes.begin());
    };
    copy(output.u, 0);
    copy(output.sigma, 1);
    copy(output.z, 2);
    copy(output.uhat, 3);
    copy(output.sigmahat, 4);
    copy(output.interior, 5);
    copy(output.trace, 6);
    copy(output.shard, 7);
    const bool valid = output.interior != StateHash{} &&
        output.trace != StateHash{} && output.shard != StateHash{};
    if (!CollectiveAll(communicator.value, valid))
    {
      output = {};
      return false;
    }
    return true;
  }

  bool ComputeUploadedTopologyHashes() noexcept
  {
    if (!LaunchRecover(detail::KernelOperation::hash_topology,
                       cuda_w8a::CudaFailurePhase::diagnostics))
    {
      return false;
    }
    std::array<std::uint8_t, 2 * 32> bytes{};
    bool copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
            bytes.data(), view.state_hash_bytes + 8 * 32, bytes.size(),
            cuda_w8a::CudaTransferCategory::scalar_diagnostic,
            cuda_w8a::CudaTransferSite::state_hash,
            bytes.size(), bytes.size());
    const bool synchronized = cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::state_hash);
    copied = copied && synchronized;
    if (!CollectiveAll(communicator.value, copied)) { return false; }
    std::copy_n(bytes.begin(), 32,
                observed_device_topology_hash.bytes.begin());
    std::copy_n(bytes.begin() + 32, 32,
                observed_device_quadrature_hash.bytes.begin());
    const bool valid =
        observed_device_topology_hash == topology_reference_hash &&
        observed_device_quadrature_hash == quadrature_reference_hash;
    return CollectiveAll(communicator.value, valid);
  }

  bool SolveTrueSystem(int& iterations, double& residual_norm,
                       double& rhs_norm,
                       const bool admit_exact_homogeneous = false) noexcept
  {
    iterations = -1;
    residual_norm = std::numeric_limits<double>::infinity();
    rhs_norm = std::numeric_limits<double>::infinity();
    bool local_valid = true;
    try
    {
      *true_solution = 0.0;
      ConfigureIlu(*ilu);
      HYPRE_ClearAllErrors();
      true_rhs->HypreRead();
      true_solution->HypreReadWrite();
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    int setup_status = -1;
    try
    {
      setup_status = HYPRE_ParCSRGMRESSetup(
          static_cast<HYPRE_Solver>(*gmres), *true_matrix, *true_rhs,
          *true_solution);
    }
    catch (...) { setup_status = -1; }
    if (!CollectiveAll(communicator.value, setup_status == 0)) { return false; }

    int solve_status = -1;
    try
    {
      solve_status = HYPRE_ParCSRGMRESSolve(
          static_cast<HYPRE_Solver>(*gmres), *true_matrix, *true_rhs,
          *true_solution);
    }
    catch (...) { solve_status = -1; }
    if (!CollectiveAll(communicator.value, solve_status == 0)) { return false; }

    HYPRE_Int converged = 0;
    int query_status = -1;
    local_valid = true;
    try
    {
      query_status = HYPRE_GMRESGetConverged(
          static_cast<HYPRE_Solver>(*gmres), &converged);
      gmres->GetNumIterations(iterations);
    }
    catch (...) { local_valid = false; }
    local_valid = local_valid && query_status == 0 && iterations >= 0 &&
        iterations <= trace_maximum_iterations &&
        (converged == 1 || admit_exact_homogeneous);
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    local_valid = true;
    try
    {
      true_matrix->Mult(*true_solution, true_residual);
      true_residual -= *true_rhs;
    }
    catch (...) { local_valid = false; }
    if (!CollectiveAll(communicator.value, local_valid)) { return false; }

    try
    {
      residual_norm = std::sqrt(mfem::InnerProduct(
          communicator.value, true_residual, true_residual));
    }
    catch (...) { residual_norm = std::numeric_limits<double>::infinity(); }
    if (!CollectiveAll(communicator.value, std::isfinite(residual_norm)))
      return false;
    try
    {
      rhs_norm = std::sqrt(mfem::InnerProduct(
          communicator.value, *true_rhs, *true_rhs));
    }
    catch (...) { rhs_norm = std::numeric_limits<double>::infinity(); }
    const bool converged_or_exact_homogeneous = converged == 1 ||
        (admit_exact_homogeneous && iterations == 0 && rhs_norm == 0.0 &&
         residual_norm == 0.0);
    const bool valid = CollectiveAll(
        communicator.value,
        converged_or_exact_homogeneous && std::isfinite(rhs_norm) &&
            residual_norm <= absolute_trace_tolerance +
                relative_trace_tolerance * rhs_norm);
    if (valid) { ++solver_generation; }
    return valid;
  }

  bool SampleHighWater(const int sample) noexcept
  {
    if (sample < 0 || sample > warm_plateau_cycles) { return false; }
    const char* mfem_name = mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    const char* hypre_name = hypre_HandleUmpireDevicePoolName(hypre_handle());
    evidence.allocations.mfem_high_water[static_cast<std::size_t>(sample)] =
        UmpireHighWater(mfem_name);
    evidence.allocations.hypre_high_water[static_cast<std::size_t>(sample)] =
        UmpireHighWater(hypre_name);
    return evidence.allocations.mfem_high_water[static_cast<std::size_t>(sample)]
               != 0 &&
        evidence.allocations.hypre_high_water[static_cast<std::size_t>(sample)]
               != 0;
  }

  bool FreezeBaseline() noexcept
  {
    bool plateau = true;
    for (int sample = 2; sample <= warm_plateau_cycles; ++sample)
    {
      plateau = plateau &&
          evidence.allocations.mfem_high_water[sample] ==
              evidence.allocations.mfem_high_water[1] &&
          evidence.allocations.hypre_high_water[sample] ==
              evidence.allocations.hypre_high_water[1];
    }
    evidence.allocations.warm_plateau = plateau;
    evidence.allocations.double_arena =
        reinterpret_cast<std::uintptr_t>(double_arena);
    evidence.allocations.integer_arena =
        reinterpret_cast<std::uintptr_t>(integer_arena);
    evidence.allocations.true_matrix =
        reinterpret_cast<std::uintptr_t>(true_matrix.get());
    evidence.allocations.gmres = reinterpret_cast<std::uintptr_t>(
        static_cast<HYPRE_Solver>(*gmres));
    evidence.allocations.ilu = reinterpret_cast<std::uintptr_t>(
        static_cast<HYPRE_Solver>(*ilu));
    evidence.allocations.baseline_observed_events =
        cuda_w8a::UmpireDeviceAllocatorPolicy::AllocationEvents();
    evidence.allocations.baseline_owned_calls =
        cuda_w8a::UmpireDeviceAllocatorPolicy::W8OwnedAllocationCalls();
    evidence.allocations.baseline_deallocations =
        cuda_w8a::UmpireDeviceAllocatorPolicy::DeallocationEvents();
    evidence.allocations.baseline_mfem_high_water =
        evidence.allocations.mfem_high_water[warm_plateau_cycles];
    evidence.allocations.baseline_hypre_high_water =
        evidence.allocations.hypre_high_water[warm_plateau_cycles];
    baseline_frozen = plateau;
    return plateau;
  }

  bool ObserveResidency() noexcept
  {
    ScalarCudaResidencyEvidence observed{};
    observed.mfem_cuda_device = mfem::Device::IsConfigured() &&
        mfem::Device::Allows(mfem::Backend::CUDA_MASK) &&
        !mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE);
    observed.umpire_device_pool =
        cuda_w8a::UmpireDeviceAllocatorPolicy::DevicePoolPolicyValid();
    observed.state_device_resident =
        ObservedCudaDevicePointer(view.slot_interior[0]) &&
        ObservedCudaDevicePointer(view.slot_interior[1]) &&
        ObservedCudaDevicePointer(view.slot_trace[0]) &&
        ObservedCudaDevicePointer(view.slot_trace[1]) &&
        ObservedCudaDevicePointer(view.trial_interior) &&
        ObservedCudaDevicePointer(view.trial_trace);
    observed.topology_device_resident =
        ObservedCudaDevicePointer(view.local_trace_to_true) &&
        ObservedCudaDevicePointer(view.local_trace_sign) &&
        ObservedCudaDevicePointer(view.local_trace_owner) &&
        ObservedCudaDevicePointer(view.matrix_owned_source) &&
        ObservedCudaDevicePointer(view.matrix_receive_target) &&
        ObservedCudaDevicePointer(view.vector_owned_source) &&
        ObservedCudaDevicePointer(view.vector_receive_target);
    observed.workspace_device_resident =
        ObservedCudaDevicePointer(view.a) &&
        ObservedCudaDevicePointer(view.b) &&
        ObservedCudaDevicePointer(view.c) &&
        ObservedCudaDevicePointer(view.d) &&
        ObservedCudaDevicePointer(view.lu) &&
        ObservedCudaDevicePointer(view.schur) &&
        ObservedCudaDevicePointer(view.recovered) &&
        ObservedCudaDevicePointer(view.volume_weight) &&
        ObservedCudaDevicePointer(view.face_weight) &&
        (!smooth_transition() ||
         (ObservedCudaDevicePointer(view.projection_volume_weight) &&
          ObservedCudaDevicePointer(view.projection_volume_shape) &&
          ObservedCudaDevicePointer(view.projection_volume_x) &&
          ObservedCudaDevicePointer(view.projection_face_weight) &&
          ObservedCudaDevicePointer(view.projection_face_oriented_shape) &&
          ObservedCudaDevicePointer(view.projection_face_x))) &&
        ObservedCudaDevicePointer(view.state_hash_bytes);
    bool hypre_observed = false;
    if (true_matrix && true_rhs && true_solution)
    {
      hypre_ParCSRMatrix* matrix = *true_matrix;
      hypre_ParVector* rhs = *true_rhs;
      hypre_ParVector* solution = *true_solution;
      const HYPRE_MemoryLocation matrix_location =
          hypre_ParCSRMatrixMemoryLocation(matrix);
      const HYPRE_MemoryLocation rhs_location =
          hypre_ParVectorMemoryLocation(rhs);
      const HYPRE_MemoryLocation solution_location =
          hypre_ParVectorMemoryLocation(solution);
      hypre_observed = matrix_location == HYPRE_MEMORY_DEVICE &&
          rhs_location == HYPRE_MEMORY_DEVICE &&
          solution_location == HYPRE_MEMORY_DEVICE &&
          hypre_GetExecPolicy1(matrix_location) == HYPRE_EXEC_DEVICE &&
          hypre_GetExecPolicy1(rhs_location) == HYPRE_EXEC_DEVICE &&
          hypre_GetExecPolicy1(solution_location) == HYPRE_EXEC_DEVICE &&
          ObservedCudaDevicePointer(view.true_matrix_data) &&
          ObservedCudaDevicePointer(view.true_rhs) &&
          ObservedCudaDevicePointer(view.true_solution);
    }
    observed.hypre_cuda = hypre_observed;
    observed.trace_solve_device_resident = hypre_observed;
    observed.host_numerical_fallback =
        structural_host_numerical_fallback_paths != 0;
    observed.host_staged_rank2 = config.require_host_staged_rank2;
    observed.same_device_uuid = same_device_uuid;
    evidence.launches.fallback_calls =
        structural_host_numerical_fallback_paths;
    evidence.residency = observed;
    return observed.mfem_cuda_device && observed.hypre_cuda &&
        observed.umpire_device_pool && observed.state_device_resident &&
        observed.topology_device_resident &&
        observed.workspace_device_resident &&
        observed.trace_solve_device_resident &&
        observed.host_staged_rank2 && observed.same_device_uuid &&
        !observed.host_numerical_fallback;
  }

  ScalarCudaReuseEvidence ReuseEvidence() const noexcept
  {
    ScalarCudaReuseEvidence result{};
    result.owner = reinterpret_cast<std::uintptr_t>(this);
    result.topology = reinterpret_cast<std::uintptr_t>(
        view.local_trace_to_true);
    result.workspace = reinterpret_cast<std::uintptr_t>(view.a);
    result.interior_slots = {{
        reinterpret_cast<std::uintptr_t>(view.slot_interior[0]),
        reinterpret_cast<std::uintptr_t>(view.slot_interior[1])}};
    result.trace_slots = {{
        reinterpret_cast<std::uintptr_t>(view.slot_trace[0]),
        reinterpret_cast<std::uintptr_t>(view.slot_trace[1])}};
    result.topology_generation = topology_generation;
    result.storage_generation = storage_generation;
    result.factor_generation = factor_generation;
    result.solver_generation = solver_generation;
    const std::array<std::uintptr_t, 7> identities{{
        result.owner, result.topology, result.workspace,
        result.interior_slots[0], result.interior_slots[1],
        result.trace_slots[0], result.trace_slots[1]}};
    Sha256 identity;
    Sha256Digest digest{};
    const bool identity_valid = identity.Update(
        topology_hash.bytes.data(), topology_hash.bytes.size()) &&
        identity.Update(identities.data(), sizeof(identities)) &&
        identity.Update(&result.topology_generation,
                        sizeof(result.topology_generation)) &&
        identity.Update(&result.storage_generation,
                        sizeof(result.storage_generation)) &&
        identity.Finalize(digest);
    result.identity_hash = identity_valid ? StateHash{digest} : StateHash{};
    result.host_topology_hash = topology_reference_hash;
    result.device_topology_hash = observed_device_topology_hash;
    result.host_quadrature_hash = quadrature_reference_hash;
    result.device_quadrature_hash = observed_device_quadrature_hash;
    result.device_topology_matches =
        result.host_topology_hash != StateHash{} &&
        result.host_topology_hash == result.device_topology_hash &&
        result.host_quadrature_hash != StateHash{} &&
        result.host_quadrature_hash == result.device_quadrature_hash;
    result.unchanged = baseline_frozen &&
        evidence.reuse.owner == result.owner &&
        evidence.reuse.topology == result.topology &&
        evidence.reuse.workspace == result.workspace &&
        evidence.reuse.interior_slots == result.interior_slots &&
        evidence.reuse.trace_slots == result.trace_slots &&
        evidence.reuse.identity_hash == result.identity_hash &&
        result.device_topology_matches;
    return result;
  }

  bool CopySlotState(const int slot, PublishedState& output) noexcept
  {
    output = {};
    if (slot < 0 || slot > 1 || elements != persistent_elements ||
        true_trace_unknowns != persistent_trace_unknowns)
    {
      return false;
    }
    std::array<double, persistent_interior_unknowns> interior{};
    std::vector<double> trace;
    try
    {
      trace.assign(static_cast<std::size_t>(owned_true_trace_unknowns), 0.0);
      output.u.assign(persistent_u_unknowns, 0.0);
      output.sigma.assign(persistent_sigma_unknowns, 0.0);
      output.z.assign(persistent_z_unknowns, 0.0);
      output.uhat.assign(
          static_cast<std::size_t>(state_components) *
              owned_scalar_trace_nodes,
          0.0);
      output.sigmahat.assign(owned_scalar_trace_nodes, 0.0);
    }
    catch (...) { output = {}; return false; }
    const std::size_t interior_bytes = interior.size() * sizeof(double);
    const std::size_t trace_bytes = trace.size() * sizeof(double);
    const bool interior_copied =
        cuda_w8a::CudaTransportPolicy::DeviceToHost(
            interior.data(), view.slot_interior[slot], interior_bytes,
            cuda_w8a::CudaTransferCategory::oracle_export,
            cuda_w8a::CudaTransferSite::oracle_export,
            interior_bytes, interior_bytes);
    const bool trace_copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
            trace.data(), view.slot_trace[slot] + owned_true_begin,
            trace_bytes,
            cuda_w8a::CudaTransferCategory::oracle_export,
            cuda_w8a::CudaTransferSite::oracle_export,
            trace_bytes, trace_bytes);
    const bool synchronized = cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::oracle_export);
    if (!interior_copied || !trace_copied || !synchronized)
    {
      output = {};
      return false;
    }
    std::copy_n(interior.begin(), output.u.size(), output.u.begin());
    std::copy_n(interior.begin() + output.u.size(), output.sigma.size(),
                output.sigma.begin());
    std::copy_n(interior.begin() + output.u.size() + output.sigma.size(),
                output.z.size(), output.z.begin());
    std::copy_n(trace.begin(), output.uhat.size(), output.uhat.begin());
    std::copy_n(trace.begin() + output.uhat.size(), output.sigmahat.size(),
                output.sigmahat.begin());
    return true;
  }

  static StateHash CheckpointManifest(
      const ScalarCudaCheckpoint& checkpoint) noexcept
  {
    Sha256 hash;
    Sha256Digest digest{};
    const auto text = [&hash](const std::string_view value)
    {
      const std::uint64_t size = value.size();
      return hash.Update(&size, sizeof(size)) &&
          hash.Update(value.data(), value.size());
    };
    const CheckpointDescriptor& descriptor = checkpoint.descriptor;
    const bool valid = text(checkpoint.projection.view()) &&
        text(checkpoint.tuple.view()) && text(checkpoint.scenario.view()) &&
        text(checkpoint.layout.view()) &&
        hash.Update(&checkpoint.shard_rank,
                    sizeof(checkpoint.shard_rank)) &&
        hash.Update(&checkpoint.owned_scalar_trace_nodes,
                    sizeof(checkpoint.owned_scalar_trace_nodes)) &&
        hash.Update(&descriptor.schema_version,
                    sizeof(descriptor.schema_version)) &&
        hash.Update(&descriptor.successful_commit,
                    sizeof(descriptor.successful_commit)) &&
        hash.Update(&descriptor.restart_mode,
                    sizeof(descriptor.restart_mode)) &&
        hash.Update(&descriptor.tuple, sizeof(descriptor.tuple)) &&
        hash.Update(&descriptor.mfem_pin, sizeof(descriptor.mfem_pin)) &&
        hash.Update(descriptor.mesh_hash.bytes.data(),
                    descriptor.mesh_hash.bytes.size()) &&
        hash.Update(descriptor.partition_hash.bytes.data(),
                    descriptor.partition_hash.bytes.size()) &&
        hash.Update(descriptor.topology_hash.bytes.data(),
                    descriptor.topology_hash.bytes.size()) &&
        hash.Update(&descriptor.rank_count, sizeof(descriptor.rank_count)) &&
        hash.Update(&descriptor.polynomial_order,
                    sizeof(descriptor.polynomial_order)) &&
        hash.Update(&descriptor.layout, sizeof(descriptor.layout)) &&
        hash.Update(&descriptor.precision, sizeof(descriptor.precision)) &&
        hash.Update(&descriptor.cursor, sizeof(descriptor.cursor)) &&
        hash.Update(&descriptor.dt, sizeof(descriptor.dt)) &&
        hash.Update(&descriptor.coordinates.time,
                    sizeof(descriptor.coordinates.time)) &&
        hash.Update(&descriptor.coordinates.step,
                    sizeof(descriptor.coordinates.step)) &&
        hash.Update(&descriptor.coordinates.generation,
                    sizeof(descriptor.coordinates.generation)) &&
        hash.Update(&descriptor.coordinates.cursor,
                    sizeof(descriptor.coordinates.cursor)) &&
        hash.Update(&descriptor.coordinates.dt,
                    sizeof(descriptor.coordinates.dt)) &&
        hash.Update(&descriptor.coordinate_fingerprint,
                    sizeof(descriptor.coordinate_fingerprint)) &&
        hash.Update(descriptor.interior_hash.bytes.data(),
                    descriptor.interior_hash.bytes.size()) &&
        hash.Update(descriptor.trace_hash.bytes.data(),
                    descriptor.trace_hash.bytes.size()) &&
        hash.Update(checkpoint.raw_field_hashes.u.bytes.data(),
                    checkpoint.raw_field_hashes.u.bytes.size()) &&
        hash.Update(checkpoint.raw_field_hashes.sigma.bytes.data(),
                    checkpoint.raw_field_hashes.sigma.bytes.size()) &&
        hash.Update(checkpoint.raw_field_hashes.z.bytes.data(),
                    checkpoint.raw_field_hashes.z.bytes.size()) &&
        hash.Update(checkpoint.raw_field_hashes.uhat.bytes.data(),
                    checkpoint.raw_field_hashes.uhat.bytes.size()) &&
        hash.Update(checkpoint.raw_field_hashes.sigmahat.bytes.data(),
                    checkpoint.raw_field_hashes.sigmahat.bytes.size()) &&
        hash.Update(checkpoint.raw_field_hashes.shard.bytes.data(),
                    checkpoint.raw_field_hashes.shard.bytes.size()) &&
        hash.Update(checkpoint.interior_component_slabs.data(),
                    checkpoint.interior_component_slabs.size() *
                        sizeof(double)) &&
        hash.Update(checkpoint.owned_trace_component_slabs.data(),
                    checkpoint.owned_trace_component_slabs.size() *
                        sizeof(double));
    return valid && hash.Finalize(digest) ? StateHash{digest} : StateHash{};
  }

  static bool StateFromCheckpoint(const ScalarCudaCheckpoint& checkpoint,
                                  PublishedState& state,
                                  const bool require_preallocated = false)
      noexcept
  {
    const int scalar_nodes = persistent_elements * volume_nodes;
    if (checkpoint.shard_rank < 0 || checkpoint.shard_rank >= 2 ||
        checkpoint.owned_scalar_trace_nodes <= 0 ||
        checkpoint.interior_component_slabs.size() !=
            persistent_interior_unknowns ||
        checkpoint.owned_trace_component_slabs.size() !=
            static_cast<std::size_t>(trace_components) *
                checkpoint.owned_scalar_trace_nodes)
    {
      return false;
    }
    const bool prepared = state.u.size() == persistent_u_unknowns &&
        state.sigma.size() == persistent_sigma_unknowns &&
        state.z.size() == persistent_z_unknowns &&
        state.uhat.size() == static_cast<std::size_t>(state_components) *
            checkpoint.owned_scalar_trace_nodes &&
        state.sigmahat.size() ==
            static_cast<std::size_t>(checkpoint.owned_scalar_trace_nodes);
    if (!prepared && require_preallocated) { return false; }
    if (!prepared)
    {
      try
      {
        state.u.resize(persistent_u_unknowns);
        state.sigma.resize(persistent_sigma_unknowns);
        state.z.resize(persistent_z_unknowns);
        state.uhat.resize(static_cast<std::size_t>(state_components) *
                          checkpoint.owned_scalar_trace_nodes);
        state.sigmahat.resize(checkpoint.owned_scalar_trace_nodes);
      }
      catch (...) { state = {}; return false; }
    }
    for (int element = 0; element < persistent_elements; ++element)
      for (int a = 0; a < volume_nodes; ++a)
      {
        const int scalar_node = element * volume_nodes + a;
        for (int component = 0; component < state_components; ++component)
        {
          state.u[(element * state_components + component) * volume_nodes +
                  a] = checkpoint.interior_component_slabs[
              component * scalar_nodes + scalar_node];
        }
        state.sigma[scalar_node] = checkpoint.interior_component_slabs[
            state_components * scalar_nodes + scalar_node];
        for (int component = 0; component < flux_components; ++component)
        {
          state.z[(element * flux_components + component) * volume_nodes +
                  a] = checkpoint.interior_component_slabs[
              (state_components + 1 + component) *
                  scalar_nodes + scalar_node];
        }
      }
    std::copy_n(checkpoint.owned_trace_component_slabs.begin(),
                state.uhat.size(),
                state.uhat.begin());
    std::copy_n(checkpoint.owned_trace_component_slabs.begin() +
                    state.uhat.size(),
                state.sigmahat.size(), state.sigmahat.begin());
    return true;
  }

  StateHash CombinedCheckpointManifest(const StateHash& shard,
                                       bool& valid) const noexcept
  {
    std::array<std::uint8_t, 64> rank_ordered{};
    valid = MPI_Allgather(shard.bytes.data(), 32, MPI_UINT8_T,
                          rank_ordered.data(), 32, MPI_UINT8_T,
                          communicator.value) == MPI_SUCCESS;
    if (!valid) { return {}; }
    Sha256 hash;
    Sha256Digest digest{};
    const std::string_view identity =
        "HDG-SC25-CUDA-R2-COMBINED-CHECKPOINT-v1";
    const std::uint64_t rank_count = 2;
    valid = hash.Update(identity.data(), identity.size()) &&
        hash.Update(&rank_count, sizeof(rank_count)) &&
        hash.Update(rank_ordered.data(), rank_ordered.size()) &&
        hash.Finalize(digest);
    return valid ? StateHash{digest} : StateHash{};
  }

  bool CaptureCheckpointSlot(const int slot,
                             const StateCoordinates& checkpoint_coordinates,
                             ScalarCudaCheckpoint& output,
                             const bool require_preallocated) noexcept
  {
    const bool valid_entry = initialized && slot >= 0 && slot <= 1 &&
        elements == persistent_elements && checkpoint_coordinates.step != 0 &&
        checkpoint_coordinates.step == checkpoint_coordinates.generation &&
        checkpoint_coordinates.step == checkpoint_coordinates.cursor &&
        checkpoint_coordinates.time ==
            static_cast<double>(checkpoint_coordinates.step) * time_step() &&
        checkpoint_coordinates.dt == time_step();
    if (!CollectiveAll(communicator.value, valid_entry))
    {
      return false;
    }
    view.selected_current_slot = slot;
    bool allocated = output.interior_component_slabs.size() ==
            persistent_interior_unknowns &&
        output.owned_trace_component_slabs.size() ==
            static_cast<std::size_t>(owned_true_trace_unknowns);
    if (!allocated && !require_preallocated)
    {
      try
      {
        output.interior_component_slabs.resize(
            persistent_interior_unknowns);
        output.owned_trace_component_slabs.resize(
            static_cast<std::size_t>(owned_true_trace_unknowns));
      }
      catch (...) {}
      allocated = output.interior_component_slabs.size() ==
              persistent_interior_unknowns &&
          output.owned_trace_component_slabs.size() ==
              static_cast<std::size_t>(owned_true_trace_unknowns);
    }
    if (!CollectiveAll(communicator.value, allocated))
    {
      return false;
    }
    if (!LaunchRecover(detail::KernelOperation::pack_checkpoint,
                       cuda_w8a::CudaFailurePhase::terminal_evidence))
    {
      return false;
    }
    const std::size_t interior_bytes =
        output.interior_component_slabs.size() * sizeof(double);
    const std::size_t trace_bytes =
        output.owned_trace_component_slabs.size() * sizeof(double);
    const bool interior_copied =
        cuda_w8a::CudaTransportPolicy::DeviceToHost(
            output.interior_component_slabs.data(),
            view.checkpoint_interior_slabs, interior_bytes,
            cuda_w8a::CudaTransferCategory::checkpoint_candidate_staging,
            cuda_w8a::CudaTransferSite::checkpoint_stage_interior,
            interior_bytes, interior_bytes);
    if (!CollectiveAll(communicator.value, interior_copied))
    {
      return false;
    }
    const bool trace_copied = cuda_w8a::CudaTransportPolicy::DeviceToHost(
            output.owned_trace_component_slabs.data(),
            view.checkpoint_trace_slabs, trace_bytes,
            cuda_w8a::CudaTransferCategory::checkpoint_candidate_staging,
            cuda_w8a::CudaTransferSite::checkpoint_stage_trace,
            trace_bytes, trace_bytes);
    if (!CollectiveAll(communicator.value, trace_copied))
    {
      return false;
    }
    const bool synchronized = cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::checkpoint_stage_trace);
    if (!CollectiveAll(communicator.value, synchronized))
    {
      return false;
    }
    output.shard_rank = rank;
    output.owned_scalar_trace_nodes = owned_scalar_trace_nodes;
    if (!CollectiveAll(communicator.value,
                       StateFromCheckpoint(output, checkpoint_workspace,
                                           true)))
    {
      return false;
    }
    output.projection = StableIdentity::Copy(projection_id);
    output.tuple = StableIdentity::Copy(cuda_rank2_tuple_id);
    output.scenario = StableIdentity::Copy(fixture_id());
    output.layout = StableIdentity::Copy(checkpoint_layout_id);
    output.raw_field_hashes = HashPublishedState(checkpoint_workspace);
    output.descriptor.schema_version = checkpoint_schema_version;
    output.descriptor.successful_commit = true;
    output.descriptor.restart_mode =
        RestartMode::same_backend_rank_partition;
    output.descriptor.tuple = CudaRank2Selection::Descriptor();
    output.descriptor.mfem_pin = StableIdentity::Copy(mfem_pin);
    output.descriptor.mesh_hash = mesh_hash;
    output.descriptor.partition_hash = partition_hash;
    output.descriptor.topology_hash = topology_hash;
    output.descriptor.rank_count = 2;
    output.descriptor.polynomial_order = 1;
    output.descriptor.layout = StableIdentity::Copy(interior_layout_id);
    output.descriptor.precision = StableIdentity::Copy("Fp64");
    output.descriptor.cursor = checkpoint_coordinates.cursor;
    output.descriptor.dt = checkpoint_coordinates.dt;
    output.descriptor.coordinates = checkpoint_coordinates;
    output.descriptor.coordinate_fingerprint =
        ::dg::hdg::runtime::detail::CoordinateFingerprint(
            checkpoint_coordinates);
    output.descriptor.interior_hash = output.raw_field_hashes.interior;
    output.descriptor.trace_hash = output.raw_field_hashes.trace;
    output.shard_manifest_hash = CheckpointManifest(output);
    if (!CollectiveAll(communicator.value,
                       output.shard_manifest_hash != StateHash{}))
    {
      return false;
    }
    bool combined_valid = false;
    output.canonical_manifest_hash = CombinedCheckpointManifest(
        output.shard_manifest_hash, combined_valid);
    const bool valid_manifest = CollectiveAll(
        communicator.value,
        combined_valid && output.canonical_manifest_hash != StateHash{});
    return valid_manifest;
  }

  bool CaptureCheckpoint(ScalarCudaCheckpoint& output) noexcept
  {
    const bool entry = active_transaction_serial == 0;
    if (!CollectiveAll(communicator.value, entry))
    {
      output = {};
      return false;
    }
    const bool captured = CaptureCheckpointSlot(
        committed_slot, coordinates, output, false);
    if (!captured) { output = {}; }
    return captured;
  }

  bool WriteProspectiveEvidence(ScalarCudaEvidence& output) const noexcept
  {
    if (!pending_attempt_active || active_transaction_serial == 0 ||
        !active_installed || evidence.attempt_count < 0 ||
        evidence.attempt_count >= static_cast<int>(evidence.attempts.size()) ||
        evidence.step_evidence_count < 0 ||
        evidence.step_evidence_count >=
            static_cast<int>(evidence.steps.size()) ||
        evidence.snapshot_count < 0 ||
        evidence.snapshot_count >=
            static_cast<int>(evidence.snapshots.size()) ||
        !pending_snapshot.valid)
    {
      return false;
    }
    output = evidence;
    ScalarCudaAttemptEvidence accepted_attempt = pending_attempt;
    accepted_attempt.status = transaction_installed_after_retry
        ? ScalarCudaStatus::accepted_after_one_retry
        : ScalarCudaStatus::accepted;
    accepted_attempt.failure_cause = ScalarCudaFailureCause::none;
    ScalarCudaStepEvidence accepted_step = pending_step;
    accepted_step.attempt_index = output.attempt_count;
    ScalarCudaSnapshotEvidence accepted_snapshot = pending_snapshot;
    accepted_snapshot.status = accepted_attempt.status;
    output.attempts[static_cast<std::size_t>(output.attempt_count++)] =
        accepted_attempt;
    output.steps[static_cast<std::size_t>(output.step_evidence_count++)] =
        accepted_step;
    output.snapshots[static_cast<std::size_t>(output.snapshot_count++)] =
        accepted_snapshot;
    ++output.accepted_steps;
    ++output.attempted_steps;
    output.coordinates = pending_attempt.candidate_coordinates;
    output.status = accepted_attempt.status;
    return true;
  }

  void RestorePriorLogicalEvidence() noexcept
  {
    RefreshLedgers();
    const ScalarCudaResidencyEvidence residency = evidence.residency;
    const ScalarCudaAllocationEvidence allocations = evidence.allocations;
    const ScalarCudaReuseEvidence reuse = evidence.reuse;
    const ScalarCudaLaunchLedger launches = evidence.launches;
    const cuda_w8a::CudaTransferLedgerV1 transfers = evidence.transfers;
    const cuda_w8a::CudaAllocationLedgerV1 allocation_ledger =
        evidence.allocation_ledger;
    evidence = transaction_prior_evidence;
    evidence.residency = residency;
    evidence.allocations = allocations;
    evidence.reuse = reuse;
    evidence.launches = launches;
    evidence.transfers = transfers;
    evidence.allocation_ledger = allocation_ledger;
  }

  void RefreshExposedPriorEvidence() noexcept
  {
    RefreshLedgers();
    transaction_prior_evidence.residency = evidence.residency;
    transaction_prior_evidence.allocations = evidence.allocations;
    transaction_prior_evidence.reuse = evidence.reuse;
    transaction_prior_evidence.launches = evidence.launches;
    transaction_prior_evidence.transfers = evidence.transfers;
    transaction_prior_evidence.allocation_ledger =
        evidence.allocation_ledger;
  }

  void RefreshLedgers() noexcept
  {
    ObserveResidency();
    evidence.transfers = cuda_w8a::CudaTransportPolicy::Ledger();
    evidence.allocation_ledger =
        cuda_w8a::UmpireDeviceAllocatorPolicy::Ledger();
    evidence.allocations.current_observed_events =
        cuda_w8a::UmpireDeviceAllocatorPolicy::AllocationEvents();
    evidence.allocations.current_owned_calls =
        cuda_w8a::UmpireDeviceAllocatorPolicy::W8OwnedAllocationCalls();
    evidence.allocations.current_deallocations =
        cuda_w8a::UmpireDeviceAllocatorPolicy::DeallocationEvents();
    evidence.allocations.current_mfem_high_water = UmpireHighWater(
        mfem::MemoryManager::GetUmpireDeviceAllocatorName());
    evidence.allocations.current_hypre_high_water = UmpireHighWater(
        hypre_HandleUmpireDevicePoolName(hypre_handle()));
    evidence.allocations.zero_post_baseline_observed_growth =
        baseline_frozen &&
        evidence.allocations.current_observed_events ==
            evidence.allocations.baseline_observed_events &&
        evidence.allocations.current_mfem_high_water ==
            evidence.allocations.baseline_mfem_high_water &&
        evidence.allocations.current_hypre_high_water ==
            evidence.allocations.baseline_hypre_high_water;
    evidence.allocations.zero_post_baseline_owned_allocation =
        baseline_frozen &&
        evidence.allocations.current_owned_calls ==
            evidence.allocations.baseline_owned_calls;
    evidence.allocations.zero_post_baseline_owned_deallocation =
        baseline_frozen &&
        evidence.allocations.current_deallocations ==
            evidence.allocations.baseline_deallocations;
    ScalarCudaReuseEvidence current = ReuseEvidence();
    if (evidence.reuse.owner != 0)
    {
      current.unchanged = current.unchanged &&
          evidence.allocations.zero_post_baseline_owned_allocation &&
          evidence.allocations.zero_post_baseline_owned_deallocation;
    }
    evidence.reuse = current;
  }

  void RecordFailedAttempt(const ScalarCudaStatus failure,
                           const bool terminal) noexcept
  {
    pending_attempt.status = failure;
    if (pending_attempt.failure_cause == ScalarCudaFailureCause::none)
    {
      pending_attempt.failure_cause = failure ==
              ScalarCudaStatus::publication_failure
          ? ScalarCudaFailureCause::publication
          : ScalarCudaFailureCause::transfer;
    }
    if (evidence.attempt_count < static_cast<int>(evidence.attempts.size()))
    {
      evidence.attempts[static_cast<std::size_t>(evidence.attempt_count++)] =
          pending_attempt;
    }
    ++evidence.attempted_steps;
    if (first_failure == ScalarCudaStatus::uninitialized)
    {
      first_failure = failure;
      evidence.first_failure = failure;
    }
    pending_attempt_active = false;
    if (terminal)
    {
      status = failure;
      evidence.status = failure;
      active_transaction_serial = 0;
      active_candidate_slot = -1;
      active_installed = false;
      prepare_retry_index = 0;
    }
    else
    {
      status = ScalarCudaStatus::ready;
      evidence.status = ScalarCudaStatus::ready;
    }
    RefreshLedgers();
  }
};

PersistentSc25ScalarIgrCudaRank2::PersistentSc25ScalarIgrCudaRank2(
    std::unique_ptr<Implementation> implementation) noexcept
    : implementation_(std::move(implementation))
{}

PersistentSc25ScalarIgrCudaRank2::~PersistentSc25ScalarIgrCudaRank2()
    noexcept = default;

PersistentSc25ScalarIgrCudaRank2::PersistentSc25ScalarIgrCudaRank2(
    PersistentSc25ScalarIgrCudaRank2&&) noexcept = default;

ScalarCudaStatus PersistentSc25ScalarIgrCudaRank2::Create(
    MPI_Comm source_communicator, const ScalarCudaConfig& config,
    std::unique_ptr<PersistentSc25ScalarIgrCudaRank2>& output) noexcept
{
  output.reset();
  if (source_communicator == MPI_COMM_NULL)
  {
    return ScalarCudaStatus::unsupported_communicator;
  }
  std::unique_ptr<Implementation> implementation;
  try
  {
    implementation = std::make_unique<Implementation>();
  }
  catch (...) {}
  if (!CollectiveAll(source_communicator, implementation != nullptr))
    return ScalarCudaStatus::allocation_failure;
  implementation->config = config;
  const bool duplicated = MPI_Comm_dup(
      source_communicator, &implementation->communicator.value) == MPI_SUCCESS;
  if (!CollectiveAll(source_communicator, duplicated))
  {
    return ScalarCudaStatus::unsupported_communicator;
  }
  const bool rank_queried = MPI_Comm_rank(implementation->communicator.value,
                                         &implementation->rank) == MPI_SUCCESS;
  if (!CollectiveAll(implementation->communicator.value, rank_queried))
    return ScalarCudaStatus::unsupported_communicator;
  const bool size_queried = MPI_Comm_size(implementation->communicator.value,
                                         &implementation->ranks) == MPI_SUCCESS;
  if (!CollectiveAll(implementation->communicator.value, size_queried))
    return ScalarCudaStatus::unsupported_communicator;
  const bool valid_config = config.polynomial_degree == 1 &&
      config.elements_per_direction ==
          cpu_sc25::persistent_elements_per_direction &&
      config.requested_device == 0 && config.require_host_staged_rank2 &&
      implementation->ranks == 2 && implementation->rank >= 0 &&
      implementation->rank < 2;
  if (!CollectiveAll(implementation->communicator.value, valid_config))
  {
    return implementation->ranks == 2 ?
        ScalarCudaStatus::invalid_configuration :
        ScalarCudaStatus::unsupported_communicator;
  }
  const bool claimed = ClaimOwnerIncarnation(implementation->incarnation);
  if (!CollectiveAll(implementation->communicator.value, claimed))
  {
    return ScalarCudaStatus::publication_failure;
  }
  std::array<unsigned char, 16> local_uuid{};
  std::array<unsigned char, 32> gathered_uuid{};
  bool local_device = cudaSetDevice(config.requested_device) == cudaSuccess;
  const bool device_queried =
      cudaGetDevice(&implementation->device) == cudaSuccess;
  local_device = local_device && device_queried &&
      implementation->device == config.requested_device;
  cudaDeviceProp device_properties{};
  const bool uuid_queried = cudaGetDeviceProperties(
      &device_properties, config.requested_device) == cudaSuccess;
  if (uuid_queried)
  {
    std::memcpy(local_uuid.data(), device_properties.uuid.bytes,
                local_uuid.size());
  }
  local_device = local_device && uuid_queried;
  const bool mfem_cuda = mfem::Device::IsConfigured() &&
      mfem::Device::Allows(mfem::Backend::CUDA_MASK) &&
      !mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE);
  local_device = local_device && mfem_cuda;
  const bool hypre_cuda = HypreCudaBackendAdmissionValid();
  local_device = local_device && hypre_cuda;
  if (!CollectiveAll(implementation->communicator.value, local_device))
  {
    return ScalarCudaStatus::unsupported_device;
  }
  const bool uuid_gathered = MPI_Allgather(
      local_uuid.data(), static_cast<int>(local_uuid.size()),
      MPI_UNSIGNED_CHAR, gathered_uuid.data(),
      static_cast<int>(local_uuid.size()), MPI_UNSIGNED_CHAR,
      implementation->communicator.value) == MPI_SUCCESS;
  implementation->same_device_uuid = uuid_gathered &&
      std::equal(gathered_uuid.begin(),
                 gathered_uuid.begin() + local_uuid.size(),
                 gathered_uuid.begin() + local_uuid.size());
  if (!CollectiveAll(implementation->communicator.value,
                     implementation->same_device_uuid))
  {
    return ScalarCudaStatus::unsupported_device;
  }
  cuda_w8a::CudaPolicyContext context{};
  context.rank = implementation->rank;
  context.phase = cuda_w8a::CudaFailurePhase::setup;
  cuda_w8a::CudaTransportPolicy::ResetLedger();
  cuda_w8a::CudaTransportPolicy::BeginAttempt(context);
  cuda_w8a::UmpireDeviceAllocatorPolicy::ResetLedger();
  if (!CollectiveAll(
          implementation->communicator.value,
          cuda_w8a::UmpireDeviceAllocatorPolicy::PrepareDeviceArenaAllocator()))
  {
    return ScalarCudaStatus::allocation_failure;
  }
  if (!implementation->BuildMeshTopologyAndQuadrature())
  {
    return ScalarCudaStatus::mfem_topology_failure;
  }
  if (!implementation->BuildPersistentTrueSystem())
  {
    return ScalarCudaStatus::mfem_topology_failure;
  }
  if (!CollectiveAll(
          implementation->communicator.value,
          HypreTrueSystemDevicePolicyValid(
              *implementation->true_matrix, *implementation->true_rhs,
              *implementation->true_solution)))
  {
    return ScalarCudaStatus::unsupported_device;
  }
  if (!CollectiveAll(implementation->communicator.value,
                     implementation->BindPersistentArenas()))
  {
    return ScalarCudaStatus::allocation_failure;
  }
  if (!implementation->ComputeUploadedTopologyHashes())
  {
    return ScalarCudaStatus::publication_failure;
  }
  implementation->status = ScalarCudaStatus::ready;
  implementation->evidence.status = ScalarCudaStatus::ready;
  if (!CollectiveAll(implementation->communicator.value,
                     implementation->ObserveResidency()))
  {
    return ScalarCudaStatus::unsupported_device;
  }
  implementation->evidence.trace_restart = trace_restart_dimension;
  implementation->evidence.trace_maximum_iterations =
      trace_maximum_iterations;
  implementation->evidence.trace_relative_tolerance =
      relative_trace_tolerance;
  implementation->evidence.trace_absolute_tolerance =
      absolute_trace_tolerance;
  implementation->evidence.ilu_type = 0;
  implementation->evidence.ilu_level_of_fill = 0;
  implementation->evidence.ilu_local_reordering = 1;
  implementation->evidence.census = {
      implementation->rank,
      implementation->ranks,
      implementation->elements,
      persistent_interior_unknowns,
      cpu_sc25::persistent_rank2_local_faces,
      implementation->owned_scalar_trace_nodes / face_nodes,
      implementation->owned_scalar_trace_nodes,
      implementation->owned_true_trace_unknowns,
      implementation->rank == 0 ?
          cpu_sc25::persistent_rank2_rank0_ghost_trace_unknowns :
          cpu_sc25::persistent_rank2_rank1_ghost_trace_unknowns,
      persistent_scalar_trace_nodes / face_nodes,
      persistent_trace_unknowns};
  const MPI_Comm admission_communicator = implementation->communicator.value;
  std::unique_ptr<PersistentSc25ScalarIgrCudaRank2> result(
      new (std::nothrow) PersistentSc25ScalarIgrCudaRank2(
          std::move(implementation)));
  if (!CollectiveAll(admission_communicator, result != nullptr))
    return ScalarCudaStatus::allocation_failure;
  output = std::move(result);
  return ScalarCudaStatus::ready;
}

ScalarCudaStatus PersistentSc25ScalarIgrCudaRank2::status() const noexcept
{
  return implementation_ ? implementation_->status
                         : ScalarCudaStatus::uninitialized;
}

const ScalarCudaConfig& PersistentSc25ScalarIgrCudaRank2::config()
    const noexcept
{
  static const ScalarCudaConfig empty{};
  return implementation_ ? implementation_->config : empty;
}

ScalarCudaStatus
PersistentSc25ScalarIgrCudaRank2::InitializePersistentTrajectory() noexcept
{
  if (!implementation_)
  {
    return ScalarCudaStatus::invalid_configuration;
  }
  Implementation& value = *implementation_;
  const bool valid_entry = !value.initialized &&
      value.config.elements_per_direction == 8 &&
      value.active_transaction_serial == 0;
  if (!CollectiveAll(value.communicator.value, valid_entry))
    return ScalarCudaStatus::invalid_configuration;
  bool checkpoint_workspace_ready = true;
  try
  {
    value.checkpoint_workspace.u.resize(persistent_u_unknowns);
    value.checkpoint_workspace.sigma.resize(persistent_sigma_unknowns);
    value.checkpoint_workspace.z.resize(persistent_z_unknowns);
    value.checkpoint_workspace.uhat.resize(
        static_cast<std::size_t>(state_components) *
        value.owned_scalar_trace_nodes);
    value.checkpoint_workspace.sigmahat.resize(
        value.owned_scalar_trace_nodes);
  }
  catch (...) { checkpoint_workspace_ready = false; }
  if (!CollectiveAll(value.communicator.value, checkpoint_workspace_ready))
    return ScalarCudaStatus::allocation_failure;
  value.committed_slot = 0;
  value.view.committed_slot = 0;
  value.view.selected_current_slot = 0;
  value.view.dt = value.time_step();
  if (!value.RunPartialPivotContract())
  {
    value.status = ScalarCudaStatus::local_factor_failure;
    return value.status;
  }
  if (!value.LaunchAssemble(detail::KernelOperation::initialize_projection,
                            cuda_w8a::CudaFailurePhase::setup))
  {
    value.status = ScalarCudaStatus::assembly_failure;
    return value.status;
  }
  if (!value.LaunchAssemble(detail::KernelOperation::assemble,
                            cuda_w8a::CudaFailurePhase::assembly))
  {
    value.status = ScalarCudaStatus::assembly_failure;
    return value.status;
  }
  if (!value.LaunchAssemble(detail::KernelOperation::auxiliary_identity_rows,
                            cuda_w8a::CudaFailurePhase::assembly))
  {
    value.status = ScalarCudaStatus::assembly_failure;
    return value.status;
  }
  if (!value.AssembleCollectiveResidual())
  {
    value.status = ScalarCudaStatus::assembly_failure;
    return value.status;
  }
  ScalarCudaSnapshotEvidence& initialization_snapshot =
      value.evidence.snapshots[0];
  if (!value.CaptureOperatorEvidence(initialization_snapshot))
  {
    value.status = ScalarCudaStatus::transfer_failure;
    return value.status;
  }
  std::array<double, 16> initialization_operator_metrics{};
  if (!value.CopyReduction(initialization_operator_metrics))
  {
    value.status = ScalarCudaStatus::transfer_failure;
    return value.status;
  }
  initialization_snapshot.initialization = true;
  initialization_snapshot.iteration = 0;
  initialization_snapshot.entry_residual_norm =
      initialization_operator_metrics[0];
  if (!value.LaunchCondense())
  {
    value.status = ScalarCudaStatus::assembly_failure;
    return value.status;
  }
  if (!value.CaptureCondensedRightHandSide(initialization_snapshot))
  {
    value.status = ScalarCudaStatus::transfer_failure;
    return value.status;
  }
  int iterations = -1;
  double residual = 0.0;
  double rhs = 0.0;
  if (!value.SolveTrueSystem(iterations, residual, rhs,
                             value.smooth_transition()))
  {
    value.status = ScalarCudaStatus::trace_linear_failure;
    return value.status;
  }
  if (!value.CaptureTrueTraceSolution(initialization_snapshot))
  {
    value.status = ScalarCudaStatus::transfer_failure;
    return value.status;
  }
  if (!value.LaunchRecover(detail::KernelOperation::recover_direction,
                           cuda_w8a::CudaFailurePhase::recovery))
  {
    value.status = ScalarCudaStatus::trace_linear_failure;
    return value.status;
  }
  if (!value.CaptureRecoveredIncrement(initialization_snapshot))
  {
    value.status = ScalarCudaStatus::transfer_failure;
    return value.status;
  }
  std::array<double, 16> initialization_solve_metrics{};
  if (!value.CopyReduction(initialization_solve_metrics))
  {
    value.status = ScalarCudaStatus::transfer_failure;
    return value.status;
  }
  value.view.selected_current_slot = 0;
  value.view.trial_theta = 1.0;
  if (!value.LaunchRecover(detail::KernelOperation::apply_trial,
                           cuda_w8a::CudaFailurePhase::recovery))
  {
    value.status = ScalarCudaStatus::admissibility_failure;
    return value.status;
  }
  if (!value.LaunchRecover(detail::KernelOperation::accept_trial,
                           cuda_w8a::CudaFailurePhase::recovery))
  {
    value.status = ScalarCudaStatus::admissibility_failure;
    return value.status;
  }
  // The persistent rank-two trace is stored redundantly so each rank can
  // evaluate its local faces without a hot-path ghost exchange.  Establish
  // that redundancy from the owning true-dof slices once, before the warm
  // plateau: independently projected ghost copies can otherwise retain
  // rank-local last-bit differences that a checkpoint restore canonicalizes.
  if (!value.StageOwnedTrace(
          value.view.slot_trace[0] + value.owned_true_begin))
  {
    value.status = ScalarCudaStatus::transfer_failure;
    return value.status;
  }
  value.view.selected_current_slot = 0;
  if (!value.LaunchRecover(
          detail::KernelOperation::install_distributed_trace,
          cuda_w8a::CudaFailurePhase::setup))
  {
    value.status = ScalarCudaStatus::transfer_failure;
    return value.status;
  }
  if (!value.LaunchRecover(detail::KernelOperation::copy_slot,
                           cuda_w8a::CudaFailurePhase::setup))
  {
    value.status = ScalarCudaStatus::admissibility_failure;
    return value.status;
  }
  if (!value.LaunchRecover(detail::KernelOperation::reduce_state,
                           cuda_w8a::CudaFailurePhase::diagnostics))
  {
    value.status = ScalarCudaStatus::admissibility_failure;
    return value.status;
  }
  std::array<double, 16> initialization_metrics{};
  if (!value.CopyReduction(initialization_metrics))
  {
    value.status = ScalarCudaStatus::admissibility_failure;
    return value.status;
  }
  const bool initialization_admissible = initialization_metrics[1] > 0.0 &&
      initialization_metrics[2] > 0.0 &&
      initialization_metrics[3] > 0.0;
  if (!CollectiveAll(value.communicator.value, initialization_admissible))
  {
    value.status = ScalarCudaStatus::admissibility_failure;
    return value.status;
  }
  initialization_snapshot.valid = true;
  initialization_snapshot.status = ScalarCudaStatus::ready;
  initialization_snapshot.coordinates = {0.0, 0, 0, 0, value.time_step()};
  initialization_snapshot.trace_iterations = iterations;
  initialization_snapshot.trace_true_residual_norm = residual;
  initialization_snapshot.trace_right_hand_side_norm = rhs;
  initialization_snapshot.maximum_local_application_residual =
      initialization_solve_metrics[4];
  initialization_snapshot.paired_local_application_residual =
      initialization_solve_metrics[6];
  initialization_snapshot.paired_local_right_hand_side_norm =
      initialization_solve_metrics[7];
  for (int component = 0; component < state_components; ++component)
  {
    initialization_snapshot.committed_integrals[component] =
        initialization_metrics[8 + component];
    initialization_snapshot.conserved_integrals[component] =
        initialization_metrics[8 + component];
  }
  value.evidence.snapshot_count = 1;
  if (!CollectiveAll(value.communicator.value, value.SampleHighWater(0)))
  {
    value.status = ScalarCudaStatus::allocation_failure;
    return value.status;
  }

  // Decision-0013 permits opaque MFEM/HYPRE warm-up growth.  The complete
  // matrix/factor/solve/recovery path is exercised repeatedly, without state
  // publication, and samples 1..20 must be an exact high-water plateau.
  for (int cycle = 1; cycle <= warm_plateau_cycles; ++cycle)
  {
    value.view.selected_current_slot = value.committed_slot;
    if (!value.LaunchAssemble(detail::KernelOperation::assemble,
                              cuda_w8a::CudaFailurePhase::assembly))
    {
      value.status = ScalarCudaStatus::trace_linear_failure;
      return value.status;
    }
    if (!value.LaunchCondense())
    {
      value.status = ScalarCudaStatus::trace_linear_failure;
      return value.status;
    }
    if (!value.SolveTrueSystem(iterations, residual, rhs))
    {
      value.status = ScalarCudaStatus::trace_linear_failure;
      return value.status;
    }
    if (!value.LaunchRecover(detail::KernelOperation::recover_direction,
                             cuda_w8a::CudaFailurePhase::recovery))
    {
      value.status = ScalarCudaStatus::trace_linear_failure;
      return value.status;
    }
    if (!CollectiveAll(value.communicator.value,
                       value.SampleHighWater(cycle)))
    {
      value.status = ScalarCudaStatus::allocation_failure;
      return value.status;
    }
  }
  if (!CollectiveAll(value.communicator.value, value.FreezeBaseline()))
  {
    value.status = ScalarCudaStatus::allocation_failure;
    return value.status;
  }
  value.coordinates = {0.0, 0, 0, 0, value.time_step()};
  value.evidence.coordinates = value.coordinates;
  value.evidence.reuse = value.ReuseEvidence();
  value.evidence.reuse.unchanged = true;
  value.initialized = true;
  value.status = ScalarCudaStatus::ready;
  value.evidence.status = ScalarCudaStatus::ready;
  value.RefreshLedgers();
  const bool ready_to_publish = value.ObserveResidency() &&
      value.evidence.allocations.zero_post_baseline_observed_growth &&
      value.evidence.allocations.zero_post_baseline_owned_allocation &&
      value.evidence.allocations.zero_post_baseline_owned_deallocation &&
      value.evidence.reuse.unchanged;
  if (!CollectiveAll(value.communicator.value, ready_to_publish))
  {
    value.status = ScalarCudaStatus::publication_failure;
    value.evidence.status = value.status;
  }
  return value.status;
}

namespace
{

bool IsInjectedFailure(const ScalarCudaStatus status) noexcept
{
  return status == ScalarCudaStatus::injected_local_factor_failure ||
      status == ScalarCudaStatus::injected_trace_linear_failure ||
      status == ScalarCudaStatus::injected_admissibility_failure ||
      status == ScalarCudaStatus::injected_newton_iteration_limit;
}

} // namespace

ScalarCudaStatus
PersistentSc25ScalarIgrCudaRank2::PreparePersistentTrajectoryTransaction(
    Transaction& output, const FailureInjection injection) noexcept
{
  const ScalarCudaStatus result =
      PreparePersistentTrajectoryTransactionImpl(output, injection);
  if (result != ScalarCudaStatus::accepted && implementation_ &&
      implementation_->pending_attempt_active &&
      !implementation_->active_installed)
  {
    const bool retryable = IsInjectedFailure(result);
    implementation_->RecordFailedAttempt(result, !retryable);
    if (retryable)
    {
      implementation_->prepare_retry_index = 1;
      ++implementation_->evidence.retry_attempts;
    }
  }
  return result;
}

ScalarCudaStatus PersistentSc25ScalarIgrCudaRank2::
    PreparePersistentTrajectoryTransactionImpl(
        Transaction& output, const FailureInjection injection) noexcept
{
  if (implementation_ &&
      !CollectiveAll(implementation_->communicator.value, !output.valid_))
    return ScalarCudaStatus::publication_failure;
  output.incarnation_ = 0;
  output.serial_ = 0;
  output.candidate_slot_ = -1;
  output.target_ = {};
  output.valid_ = false;
  output.installed_ = false;
  if (!implementation_) { return ScalarCudaStatus::publication_failure; }
  const bool prepared_state = implementation_->initialized &&
      implementation_->status == ScalarCudaStatus::ready &&
      implementation_->active_transaction_serial == 0 &&
      implementation_->coordinates.step < implementation_->required_steps() &&
      implementation_->next_transaction_serial != 0 &&
      implementation_->next_transaction_serial !=
          std::numeric_limits<std::uint64_t>::max();
  if (!CollectiveAll(implementation_->communicator.value, prepared_state))
  {
    return ScalarCudaStatus::publication_failure;
  }
  Implementation& value = *implementation_;
  value.transaction_prior_evidence = value.evidence;
  value.transaction_prior_status = value.status;
  value.transaction_prior_first_failure = value.first_failure;
  const int candidate_slot = 1 - value.committed_slot;
  value.pending_attempt = {};
  value.pending_step = {};
  value.pending_snapshot = {};
  value.pending_attempt_active = true;
  value.pending_attempt.injection =
      value.rank == 1 ? injection : FailureInjection::none;
  value.pending_attempt.target_step =
      static_cast<int>(value.coordinates.step + 1);
  value.pending_attempt.retry_index = value.prepare_retry_index;
  value.pending_attempt.dt = value.time_step();
  value.pending_attempt.committed_coordinates = value.coordinates;
  const std::uint64_t target_step = value.coordinates.step + 1;
  value.pending_attempt.candidate_coordinates = {
      static_cast<double>(target_step) * value.time_step(),
      target_step,
      value.coordinates.generation + 1,
      value.coordinates.cursor + 1,
      value.time_step()};

  cuda_w8a::CudaPolicyContext context{};
  context.attempt_sequence = value.coordinates.cursor + 1;
  context.target_step = value.coordinates.step + 1;
  context.generation = value.coordinates.generation + 1;
  context.rank = value.rank;
  context.phase = cuda_w8a::CudaFailurePhase::setup;
  cuda_w8a::CudaTransportPolicy::BeginAttempt(context);
  value.view.selected_current_slot = value.committed_slot;
  if (!value.ComputeStateHashes(value.committed_slot,
                                value.pending_attempt.entry_hashes))
  {
    value.pending_attempt.status = ScalarCudaStatus::transfer_failure;
    value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
    return value.pending_attempt.status;
  }
  if (!value.LaunchRecover(detail::KernelOperation::copy_slot,
                           cuda_w8a::CudaFailurePhase::setup))
  {
    value.ComputeStateHashes(candidate_slot,
                             value.pending_attempt.candidate_hashes);
    value.pending_attempt.status = ScalarCudaStatus::transfer_failure;
    value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
    return value.pending_attempt.status;
  }
  value.view.selected_current_slot = value.committed_slot;
  if (!value.LaunchRecover(detail::KernelOperation::reduce_state,
                           cuda_w8a::CudaFailurePhase::diagnostics))
  {
    value.ComputeStateHashes(candidate_slot,
                             value.pending_attempt.candidate_hashes);
    value.pending_attempt.status = value.LastAssemblyFailureStatus();
    value.pending_attempt.failure_cause = value.LastDeviceFailureCause();
    return value.pending_attempt.status;
  }
  std::array<double, 16> entry_metrics{};
  if (!value.CopyReduction(entry_metrics))
  {
    value.ComputeStateHashes(candidate_slot,
                             value.pending_attempt.candidate_hashes);
    value.pending_attempt.status = ScalarCudaStatus::transfer_failure;
    value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
    return value.pending_attempt.status;
  }
  std::array<double, state_components> committed_integrals{};
  for (int component = 0; component < state_components; ++component)
  {
    committed_integrals[component] = entry_metrics[8 + component];
  }
  const bool inject_newton_limit = CollectiveAny(
      value.communicator.value,
      value.rank == 1 &&
          injection == FailureInjection::newton_iteration_limit);
  const bool inject_local_factor = CollectiveAny(
      value.communicator.value,
      value.rank == 1 && injection == FailureInjection::local_factor);
  const bool inject_trace_linear = CollectiveAny(
      value.communicator.value,
      value.rank == 1 && injection == FailureInjection::trace_linear);
  const bool inject_admissibility = CollectiveAny(
      value.communicator.value,
      value.rank == 1 && injection == FailureInjection::admissibility);

  bool converged = false;
  ScalarCudaStatus failure = ScalarCudaStatus::uninitialized;
  double entry_norm = 0.0;
  double final_norm = std::numeric_limits<double>::infinity();
  double nonlinear_bound = std::numeric_limits<double>::infinity();
  for (int iteration = 0;
       failure == ScalarCudaStatus::uninitialized &&
           iteration < newton_maximum_iterations;
       ++iteration)
  {
    value.view.selected_current_slot = candidate_slot;
    if (!value.LaunchAssemble(detail::KernelOperation::assemble,
                              cuda_w8a::CudaFailurePhase::assembly))
    {
      failure = value.LastAssemblyFailureStatus();
      value.pending_attempt.failure_cause = value.LastDeviceFailureCause();
      break;
    }
    std::array<double, 16> assembly_metrics{};
    if (!value.CopyReduction(assembly_metrics))
    {
      failure = ScalarCudaStatus::transfer_failure;
      value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
      break;
    }
    const double residual_norm = assembly_metrics[0];
    value.pending_attempt.attempted_newton_iterations = iteration + 1;
    if (iteration == 0)
    {
      entry_norm = residual_norm;
      nonlinear_bound = 1.0e-12 + 1.0e-10 * entry_norm;
    }
    final_norm = residual_norm;
    if (std::isfinite(residual_norm) && residual_norm <= nonlinear_bound)
    {
      converged = true;
      value.pending_attempt.newton_iterations = iteration;
      break;
    }
    if (iteration == 0 && inject_newton_limit)
    {
      value.pending_attempt.attempted_newton_iterations =
          newton_maximum_iterations;
      failure = ScalarCudaStatus::injected_newton_iteration_limit;
      value.pending_attempt.failure_cause =
          ScalarCudaFailureCause::injected_newton_limit;
      break;
    }
    if (iteration == 0 && inject_local_factor)
    {
      failure = ScalarCudaStatus::injected_local_factor_failure;
      value.pending_attempt.failure_cause =
          ScalarCudaFailureCause::injected_local_factor;
      break;
    }
    if (!value.CaptureOperatorEvidence(value.capture_snapshot))
    {
      failure = ScalarCudaStatus::transfer_failure;
      value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
      break;
    }
    value.capture_snapshot.entry_residual_norm = residual_norm;
    value.capture_snapshot.iteration = iteration;
    if (!value.LaunchCondense())
    {
      failure = ScalarCudaStatus::local_factor_failure;
      value.pending_attempt.failure_cause = value.LastDeviceFailureCause();
      break;
    }
    if (!value.CaptureCondensedRightHandSide(value.capture_snapshot))
    {
      failure = ScalarCudaStatus::transfer_failure;
      value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
      break;
    }
    if (iteration == 0 && inject_trace_linear)
    {
      failure = ScalarCudaStatus::injected_trace_linear_failure;
      value.pending_attempt.failure_cause =
          ScalarCudaFailureCause::injected_trace_linear;
      break;
    }
    int trace_iterations = -1;
    double trace_residual = 0.0;
    double trace_rhs = 0.0;
    if (!value.SolveTrueSystem(trace_iterations, trace_residual, trace_rhs))
    {
      failure = ScalarCudaStatus::trace_linear_failure;
      value.pending_attempt.failure_cause =
          ScalarCudaFailureCause::trace_nonconvergence;
      break;
    }
    if (!value.CaptureTrueTraceSolution(value.capture_snapshot))
    {
      failure = ScalarCudaStatus::transfer_failure;
      value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
      break;
    }
    if (!value.LaunchRecover(detail::KernelOperation::recover_direction,
                             cuda_w8a::CudaFailurePhase::recovery))
    {
      failure = ScalarCudaStatus::local_factor_failure;
      value.pending_attempt.failure_cause = value.LastDeviceFailureCause();
      break;
    }
    if (!value.CaptureRecoveredIncrement(value.capture_snapshot))
    {
      failure = ScalarCudaStatus::transfer_failure;
      value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
      break;
    }
    std::array<double, 16> solve_metrics{};
    if (!value.CopyReduction(solve_metrics))
    {
      failure = ScalarCudaStatus::transfer_failure;
      value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
      break;
    }
    ScalarCudaNewtonEvidence& newton =
        value.pending_attempt.newton[static_cast<std::size_t>(iteration)];
    newton.iteration = iteration;
    newton.entry_residual_norm = residual_norm;
    newton.trace_iterations = trace_iterations;
    newton.trace_true_residual_norm = trace_residual;
    newton.trace_right_hand_side_norm = trace_rhs;
    newton.maximum_local_application_residual = solve_metrics[4];
    value.capture_snapshot.trace_iterations = trace_iterations;
    value.capture_snapshot.trace_true_residual_norm = trace_residual;
    value.capture_snapshot.trace_right_hand_side_norm = trace_rhs;
    value.capture_snapshot.maximum_local_application_residual =
        solve_metrics[4];
    value.capture_snapshot.paired_local_application_residual =
        solve_metrics[6];
    value.capture_snapshot.paired_local_right_hand_side_norm =
        solve_metrics[7];
    value.capture_snapshot.valid = true;
    value.pending_snapshot = value.capture_snapshot;
    newton.trial_residual_norms.fill(
        std::numeric_limits<double>::infinity());
    bool accepted = false;
    for (int line = 0; line < line_search_trials; ++line)
    {
      const double theta = std::ldexp(1.0, -line);
      if (inject_admissibility)
      {
        newton.trial_admissible[static_cast<std::size_t>(line)] = 0;
        value.pending_attempt.failure_cause =
            ScalarCudaFailureCause::injected_admissibility;
        continue;
      }
      value.view.selected_current_slot = candidate_slot;
      value.view.trial_theta = theta;
      if (!value.LaunchRecover(detail::KernelOperation::apply_trial,
                               cuda_w8a::CudaFailurePhase::diagnostics))
      {
        failure = ScalarCudaStatus::assembly_failure;
        value.pending_attempt.failure_cause = value.LastDeviceFailureCause();
        break;
      }
      value.view.selected_current_slot = 2;
      if (!value.LaunchAssemble(detail::KernelOperation::assemble,
                                cuda_w8a::CudaFailurePhase::diagnostics))
      {
        if (value.LastDeviceFailureIsAdmissibility())
        {
          newton.trial_admissible[static_cast<std::size_t>(line)] = 0;
          continue;
        }
        failure = value.LastAssemblyFailureStatus();
        value.pending_attempt.failure_cause =
            value.LastDeviceFailureCause();
        break;
      }
      std::array<double, 16> trial_metrics{};
      if (!value.CopyReduction(trial_metrics))
      {
        failure = ScalarCudaStatus::transfer_failure;
        value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
        break;
      }
      const double trial_norm = trial_metrics[0];
      newton.trial_residual_norms[static_cast<std::size_t>(line)] = trial_norm;
      newton.trial_admissible[static_cast<std::size_t>(line)] = 1;
      const bool armijo = std::isfinite(trial_norm) &&
          trial_norm <= (1.0 - armijo_coefficient * theta) * residual_norm;
      newton.trial_armijo[static_cast<std::size_t>(line)] = armijo ? 1 : 0;
      if (armijo)
      {
        value.view.selected_current_slot = candidate_slot;
        if (!value.LaunchRecover(detail::KernelOperation::accept_trial,
                                 cuda_w8a::CudaFailurePhase::diagnostics))
        {
          failure = value.LastAssemblyFailureStatus();
          value.pending_attempt.failure_cause =
              value.LastDeviceFailureCause();
          break;
        }
        newton.accepted_line_search_index = line;
        newton.accepted_residual_norm = trial_norm;
        accepted = true;
        break;
      }
    }
    value.pending_attempt.newton_iterations = iteration + 1;
    if (failure != ScalarCudaStatus::uninitialized) { break; }
    if (!accepted)
    {
      failure = inject_admissibility
          ? ScalarCudaStatus::injected_admissibility_failure
          : ScalarCudaStatus::admissibility_failure;
      if (value.pending_attempt.failure_cause == ScalarCudaFailureCause::none)
      {
        value.pending_attempt.failure_cause =
            inject_admissibility
            ? ScalarCudaFailureCause::injected_admissibility
            : ScalarCudaFailureCause::device_state_admissibility;
      }
      break;
    }
  }
  if (failure == ScalarCudaStatus::uninitialized && !converged)
  {
    failure = ScalarCudaStatus::newton_iteration_limit;
    value.pending_attempt.failure_cause =
        ScalarCudaFailureCause::nonlinear_nonconvergence;
  }
  value.pending_attempt.entry_residual_norm = entry_norm;
  value.pending_attempt.final_residual_norm = final_norm;
  value.pending_attempt.nonlinear_residual_bound = nonlinear_bound;
  if (failure != ScalarCudaStatus::uninitialized)
  {
    if (!value.ComputeStateHashes(candidate_slot,
                                  value.pending_attempt.candidate_hashes))
    {
      failure = ScalarCudaStatus::transfer_failure;
      value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
    }
    value.pending_attempt.status = failure;
    return failure;
  }

  value.view.selected_current_slot = candidate_slot;
  if (!value.LaunchRecover(detail::KernelOperation::reduce_state,
                           cuda_w8a::CudaFailurePhase::diagnostics))
  {
    value.ComputeStateHashes(candidate_slot,
                             value.pending_attempt.candidate_hashes);
    value.pending_attempt.status = value.LastAssemblyFailureStatus();
    value.pending_attempt.failure_cause = value.LastDeviceFailureCause();
    return value.pending_attempt.status;
  }
  std::array<double, 16> candidate_metrics{};
  if (!value.CopyReduction(candidate_metrics))
  {
    value.ComputeStateHashes(candidate_slot,
                             value.pending_attempt.candidate_hashes);
    value.pending_attempt.status = ScalarCudaStatus::transfer_failure;
    value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
    return value.pending_attempt.status;
  }
  value.pending_attempt.minimum_density = candidate_metrics[1];
  value.pending_attempt.minimum_pressure = candidate_metrics[2];
  value.pending_attempt.minimum_speed_margin = candidate_metrics[3];
  if (!(candidate_metrics[1] > 0.0) || !(candidate_metrics[2] > 0.0) ||
      !(candidate_metrics[3] > 0.0))
  {
    value.ComputeStateHashes(candidate_slot,
                             value.pending_attempt.candidate_hashes);
    value.pending_attempt.status = ScalarCudaStatus::admissibility_failure;
    value.pending_attempt.failure_cause =
        ScalarCudaFailureCause::device_state_admissibility;
    return value.pending_attempt.status;
  }
  if (!value.ComputeStateHashes(candidate_slot,
                                value.pending_attempt.candidate_hashes))
  {
    value.pending_attempt.status = ScalarCudaStatus::transfer_failure;
    value.pending_attempt.failure_cause = ScalarCudaFailureCause::transfer;
    return value.pending_attempt.status;
  }
  std::array<double, state_components> accepted_integrals{};
  double maximum_conservation = 0.0;
  for (int component = 0; component < state_components; ++component)
  {
    accepted_integrals[component] = candidate_metrics[8 + component];
    maximum_conservation = std::max(maximum_conservation,
        std::abs(accepted_integrals[component] -
                 committed_integrals[component]) /
            std::max(1.0, std::abs(committed_integrals[component])));
  }
  value.pending_attempt.maximum_conservation_error = maximum_conservation;
  if (maximum_conservation > 2.0e-10)
  {
    value.pending_attempt.status = ScalarCudaStatus::conservation_failure;
    value.pending_attempt.failure_cause = ScalarCudaFailureCause::conservation;
    return value.pending_attempt.status;
  }
  const bool topology_matches = value.ComputeUploadedTopologyHashes();
  value.RefreshLedgers();
  const bool publishable = topology_matches &&
      value.evidence.allocations.zero_post_baseline_observed_growth &&
      value.evidence.allocations.zero_post_baseline_owned_allocation &&
      value.evidence.allocations.zero_post_baseline_owned_deallocation &&
      value.ObserveResidency() && value.evidence.reuse.unchanged &&
      value.evidence.launches.fallback_calls == 0;
  if (!CollectiveAll(value.communicator.value, publishable))
  {
    value.pending_attempt.status = ScalarCudaStatus::publication_failure;
    value.pending_attempt.failure_cause = ScalarCudaFailureCause::publication;
    return value.pending_attempt.status;
  }

  value.pending_attempt.status = ScalarCudaStatus::accepted;
  value.pending_attempt.failure_cause = ScalarCudaFailureCause::none;
  value.pending_step.accepted = true;
  value.pending_step.step = static_cast<int>(value.coordinates.step + 1);
  value.pending_step.time = static_cast<double>(target_step) * value.time_step();
  value.pending_step.generation = value.coordinates.generation + 1;
  value.pending_step.committed_integrals = committed_integrals;
  value.pending_step.accepted_integrals = accepted_integrals;
  value.pending_step.accepted_hashes = value.pending_attempt.candidate_hashes;
  value.pending_snapshot.initialization = false;
  value.pending_snapshot.status = value.prepare_retry_index == 1
      ? ScalarCudaStatus::accepted_after_one_retry
      : ScalarCudaStatus::accepted;
  value.pending_snapshot.coordinates =
      value.pending_attempt.candidate_coordinates;
  value.pending_snapshot.target_step = value.pending_attempt.target_step;
  value.pending_snapshot.retry_index = value.pending_attempt.retry_index;
  value.pending_snapshot.attempted_newton_iterations =
      value.pending_attempt.attempted_newton_iterations;
  value.pending_snapshot.newton_iterations =
      value.pending_attempt.newton_iterations;
  value.pending_snapshot.committed_integrals = committed_integrals;
  value.pending_snapshot.conserved_integrals = accepted_integrals;
  if (!CollectiveAll(value.communicator.value, value.pending_snapshot.valid))
  {
    value.pending_attempt.status = ScalarCudaStatus::publication_failure;
    value.pending_attempt.failure_cause = ScalarCudaFailureCause::publication;
    return value.pending_attempt.status;
  }
  const std::uint64_t serial = value.next_transaction_serial++;
  value.active_transaction_serial = serial;
  value.active_candidate_slot = candidate_slot;
  value.active_installed = false;
  output.incarnation_ = value.incarnation;
  output.serial_ = serial;
  output.candidate_slot_ = candidate_slot;
  output.target_ = value.pending_attempt.candidate_coordinates;
  output.valid_ = true;
  output.installed_ = false;
  value.status = ScalarCudaStatus::transaction_prepared;
  value.evidence.status = ScalarCudaStatus::transaction_prepared;
  return ScalarCudaStatus::accepted;
}

bool PersistentSc25ScalarIgrCudaRank2::
    InstallPersistentTrajectoryTransaction(Transaction& token) noexcept
{
  if (!implementation_) { return false; }
  const Implementation& value = *implementation_;
  const bool locally_valid = token.valid_ &&
      token.incarnation_ == value.incarnation && token.serial_ != 0 &&
      token.serial_ == value.active_transaction_serial &&
      token.candidate_slot_ == value.active_candidate_slot &&
      !token.installed_ && !value.active_installed &&
      SameCoordinates(token.target_,
                      value.pending_attempt.candidate_coordinates);
  if (!CollectiveAll(value.communicator.value, locally_valid))
  {
    return false;
  }
  Implementation& installed = *implementation_;
  installed.transaction_installed_after_retry =
      installed.pending_attempt.retry_index == 1;
  installed.active_installed = true;
  token.installed_ = true;
  return true;
}

bool PersistentSc25ScalarIgrCudaRank2::
    StageInstalledTrajectoryTransaction(
        Transaction& token, ScalarCudaCheckpoint& checkpoint,
        ScalarCudaEvidence& prospective_evidence) noexcept
{
  if (!implementation_) { return false; }
  Implementation& value = *implementation_;
  const bool locally_valid = token.valid_ &&
      token.incarnation_ == value.incarnation && token.serial_ != 0 &&
      token.serial_ == value.active_transaction_serial &&
      token.candidate_slot_ == value.active_candidate_slot &&
      token.installed_ && value.active_installed &&
      SameCoordinates(token.target_,
                      value.pending_attempt.candidate_coordinates) &&
      checkpoint.interior_component_slabs.size() ==
          persistent_interior_unknowns &&
      checkpoint.owned_trace_component_slabs.size() ==
          static_cast<std::size_t>(value.owned_true_trace_unknowns) &&
      value.evidence.attempt_count >= 0 &&
      value.evidence.attempt_count <
          static_cast<int>(value.evidence.attempts.size()) &&
      value.evidence.step_evidence_count >= 0 &&
      value.evidence.step_evidence_count <
          static_cast<int>(value.evidence.steps.size()) &&
      value.evidence.snapshot_count >= 0 &&
      value.evidence.snapshot_count <
          static_cast<int>(value.evidence.snapshots.size()) &&
      value.pending_snapshot.valid;
  if (!CollectiveAll(value.communicator.value, locally_valid))
  {
    return false;
  }
  if (!value.CaptureCheckpointSlot(token.candidate_slot_, token.target_,
                                   checkpoint, true))
  {
    return false;
  }
  value.RefreshLedgers();
  return value.WriteProspectiveEvidence(prospective_evidence);
}

bool PersistentSc25ScalarIgrCudaRank2::
    RevertPersistentTrajectoryTransaction(Transaction&& token) noexcept
{
  if (!implementation_) { return false; }
  const Implementation& observed = *implementation_;
  const bool locally_valid = token.valid_ &&
      token.incarnation_ == observed.incarnation && token.serial_ != 0 &&
      token.serial_ == observed.active_transaction_serial &&
      token.candidate_slot_ == observed.active_candidate_slot &&
      token.installed_ == observed.active_installed &&
      SameCoordinates(token.target_,
                      observed.pending_attempt.candidate_coordinates);
  if (!CollectiveAll(observed.communicator.value, locally_valid))
  {
    return false;
  }
  Implementation& reverted = *implementation_;
  // Candidate execution remains physically observable, but no logical
  // trajectory field survives a successful rollback.
  reverted.RestorePriorLogicalEvidence();
  reverted.status = reverted.transaction_prior_status;
  reverted.first_failure = reverted.transaction_prior_first_failure;
  reverted.pending_attempt_active = false;
  reverted.transaction_installed_after_retry = false;
  reverted.active_transaction_serial = 0;
  reverted.active_candidate_slot = -1;
  reverted.active_installed = false;
  token.incarnation_ = 0;
  token.serial_ = 0;
  token.candidate_slot_ = -1;
  token.target_ = {};
  token.valid_ = false;
  token.installed_ = false;
  return true;
}

bool PersistentSc25ScalarIgrCudaRank2::
    FinalizePersistentTrajectoryTransaction(Transaction&& token) noexcept
{
  if (!implementation_) { return false; }
  const Implementation& observed = *implementation_;
  const bool locally_valid = token.valid_ &&
      token.incarnation_ == observed.incarnation && token.serial_ != 0 &&
      token.serial_ == observed.active_transaction_serial &&
      token.candidate_slot_ == observed.active_candidate_slot &&
      token.installed_ && observed.active_installed &&
      observed.evidence.attempt_count >= 0 &&
      observed.evidence.attempt_count <
          static_cast<int>(observed.evidence.attempts.size()) &&
      observed.evidence.step_evidence_count >= 0 &&
      observed.evidence.step_evidence_count <
          static_cast<int>(observed.evidence.steps.size()) &&
      observed.evidence.snapshot_count >= 0 &&
      observed.evidence.snapshot_count <
          static_cast<int>(observed.evidence.snapshots.size()) &&
      observed.pending_snapshot.valid &&
      SameCoordinates(token.target_,
                      observed.pending_attempt.candidate_coordinates);
  if (!CollectiveAll(observed.communicator.value, locally_valid))
  {
    return false;
  }
  Implementation& value = *implementation_;
  value.pending_attempt.status = value.transaction_installed_after_retry
      ? ScalarCudaStatus::accepted_after_one_retry
      : ScalarCudaStatus::accepted;
  value.pending_step.attempt_index = value.evidence.attempt_count;
  value.evidence.attempts[
      static_cast<std::size_t>(value.evidence.attempt_count++)] =
      value.pending_attempt;
  value.evidence.steps[
      static_cast<std::size_t>(value.evidence.step_evidence_count++)] =
      value.pending_step;
  const int snapshot_index = value.evidence.snapshot_count;
  value.pending_snapshot.status = value.pending_attempt.status;
  value.evidence.snapshots[static_cast<std::size_t>(snapshot_index)] =
      value.pending_snapshot;
  ++value.evidence.snapshot_count;
  ++value.evidence.accepted_steps;
  ++value.evidence.attempted_steps;
  value.evidence.coordinates = token.target_;
  value.evidence.status = value.pending_attempt.status;
  value.committed_slot = token.candidate_slot_;
  value.coordinates = token.target_;
  value.pending_attempt_active = false;
  value.prepare_retry_index = 0;
  value.transaction_installed_after_retry = false;
  value.active_transaction_serial = 0;
  value.active_candidate_slot = -1;
  value.active_installed = false;
  token.incarnation_ = 0;
  token.serial_ = 0;
  token.candidate_slot_ = -1;
  token.target_ = {};
  token.valid_ = false;
  token.installed_ = false;
  value.RefreshLedgers();
  // A committed leaf remains ready for the next transaction; the accepted
  // result is retained in evidence and in the return from Advance.
  value.status = ScalarCudaStatus::ready;
  return true;
}

ScalarCudaStatus PersistentSc25ScalarIgrCudaRank2::AdvancePersistentTrajectory(
    const FailureInjection injection) noexcept
{
  Transaction transaction;
  ScalarCudaStatus result = PreparePersistentTrajectoryTransactionImpl(
      transaction, injection);
  if (result == ScalarCudaStatus::accepted)
  {
    if (!InstallPersistentTrajectoryTransaction(transaction))
    {
      return ScalarCudaStatus::publication_failure;
    }
    if (!FinalizePersistentTrajectoryTransaction(std::move(transaction)))
    {
      return ScalarCudaStatus::publication_failure;
    }
    return ScalarCudaStatus::accepted;
  }
  if (!IsInjectedFailure(result) || !implementation_)
  {
    if (implementation_ && implementation_->pending_attempt_active)
    {
      implementation_->RecordFailedAttempt(result, true);
    }
    return result;
  }
  Implementation& value = *implementation_;
  value.RecordFailedAttempt(result, false);
  value.prepare_retry_index = 1;
  ++value.evidence.retry_attempts;
  Transaction retry;
  const ScalarCudaStatus retry_result =
      PreparePersistentTrajectoryTransactionImpl(
          retry, FailureInjection::none);
  if (retry_result != ScalarCudaStatus::accepted)
  {
    if (value.pending_attempt_active)
    {
      value.RecordFailedAttempt(retry_result, true);
    }
    value.prepare_retry_index = 0;
    value.status = ScalarCudaStatus::retry_exhausted;
    value.evidence.status = value.status;
    return value.status;
  }
  if (!InstallPersistentTrajectoryTransaction(retry))
  {
    value.prepare_retry_index = 0;
    value.status = ScalarCudaStatus::retry_exhausted;
    value.evidence.status = value.status;
    return value.status;
  }
  if (!FinalizePersistentTrajectoryTransaction(std::move(retry)))
  {
    value.prepare_retry_index = 0;
    value.status = ScalarCudaStatus::retry_exhausted;
    value.evidence.status = value.status;
    return value.status;
  }
  return ScalarCudaStatus::accepted_after_one_retry;
}

bool PersistentSc25ScalarIgrCudaRank2::CopyPublishedTrajectoryState(
    PublishedState& output) const noexcept
{
  if (!implementation_)
  {
    output = {};
    return false;
  }
  Implementation& value = *implementation_;
  if (!CollectiveAll(value.communicator.value, value.initialized))
  {
    output = {};
    return false;
  }
  const bool copied = value.CopySlotState(value.committed_slot, output);
  if (!CollectiveAll(value.communicator.value, copied))
  {
    output = {};
    return false;
  }
  value.RefreshLedgers();
  return CollectiveAll(value.communicator.value, true);
}

bool PersistentSc25ScalarIgrCudaRank2::CaptureTrajectoryCheckpoint(
    ScalarCudaCheckpoint& output) const noexcept
{
  if (!implementation_)
  {
    output = {};
    return false;
  }
  const bool captured = implementation_->CaptureCheckpoint(output);
  implementation_->RefreshLedgers();
  const bool refreshed = CollectiveAll(
      implementation_->communicator.value, true);
  return captured && refreshed;
}

ScalarCudaStatus PersistentSc25ScalarIgrCudaRank2::
    RestoreTrajectoryCheckpoint(const ScalarCudaCheckpoint& checkpoint)
    noexcept
{
  if (!implementation_) { return ScalarCudaStatus::restart_identity_failure; }
  Implementation& value = *implementation_;
  if (!CollectiveAll(value.communicator.value,
                     value.initialized &&
                         value.active_transaction_serial == 0))
  {
    return ScalarCudaStatus::restart_identity_failure;
  }
  const CheckpointDescriptor& descriptor = checkpoint.descriptor;
  const bool identity = checkpoint.projection == StableIdentity::Copy(
                            projection_id) &&
      checkpoint.tuple == StableIdentity::Copy(cuda_rank2_tuple_id) &&
      checkpoint.scenario == StableIdentity::Copy(value.fixture_id()) &&
      checkpoint.layout == StableIdentity::Copy(checkpoint_layout_id) &&
      descriptor.schema_version == checkpoint_schema_version &&
      descriptor.successful_commit &&
      descriptor.restart_mode == RestartMode::same_backend_rank_partition &&
      descriptor.tuple == CudaRank2Selection::Descriptor() &&
      descriptor.mfem_pin == StableIdentity::Copy(mfem_pin) &&
      descriptor.mesh_hash == value.mesh_hash &&
      descriptor.partition_hash == value.partition_hash &&
      descriptor.topology_hash == value.topology_hash &&
      descriptor.rank_count == 2 && descriptor.polynomial_order == 1 &&
      checkpoint.shard_rank == value.rank &&
      checkpoint.owned_scalar_trace_nodes ==
          value.owned_scalar_trace_nodes &&
      descriptor.layout == StableIdentity::Copy(interior_layout_id) &&
      descriptor.precision == StableIdentity::Copy("Fp64") &&
      descriptor.cursor == descriptor.coordinates.cursor &&
      descriptor.dt == descriptor.coordinates.dt &&
      descriptor.coordinate_fingerprint ==
          ::dg::hdg::runtime::detail::CoordinateFingerprint(
              descriptor.coordinates) &&
      descriptor.coordinates.step != 0 &&
      descriptor.coordinates.step <= value.required_steps() &&
      descriptor.coordinates.step == descriptor.coordinates.generation &&
      descriptor.coordinates.step == descriptor.coordinates.cursor &&
      descriptor.coordinates.time ==
          static_cast<double>(descriptor.coordinates.step) * value.time_step() &&
      descriptor.coordinates.dt == value.time_step();
  if (!CollectiveAll(value.communicator.value, identity))
    return ScalarCudaStatus::restart_identity_failure;
  PublishedState host_state;
  const bool decoded = Implementation::StateFromCheckpoint(
      checkpoint, host_state);
  const StateHashes hashes = HashPublishedState(host_state);
  const StateHash shard_manifest =
      Implementation::CheckpointManifest(checkpoint);
  const bool shard_hashes_valid = decoded &&
      SameStateHashes(hashes, checkpoint.raw_field_hashes) &&
      descriptor.interior_hash == hashes.interior &&
      descriptor.trace_hash == hashes.trace &&
      checkpoint.shard_manifest_hash == shard_manifest &&
      shard_manifest != StateHash{};
  if (!CollectiveAll(value.communicator.value, shard_hashes_valid))
    return ScalarCudaStatus::restart_hash_failure;
  bool combined_valid = false;
  const StateHash combined_manifest = value.CombinedCheckpointManifest(
      shard_manifest, combined_valid);
  const bool all_hashes_valid = combined_valid &&
      checkpoint.canonical_manifest_hash == combined_manifest;
  if (!CollectiveAll(value.communicator.value, all_hashes_valid))
  {
    return ScalarCudaStatus::restart_hash_failure;
  }
  const auto finite_array = [](const auto& array)
  {
    return std::all_of(array.begin(), array.end(),
                       [](const double item) { return std::isfinite(item); });
  };
  if (!CollectiveAll(value.communicator.value,
          finite_array(host_state.u) && finite_array(host_state.sigma) &&
          finite_array(host_state.z) && finite_array(host_state.uhat) &&
          finite_array(host_state.sigmahat)))
  {
    return ScalarCudaStatus::restart_admissibility_failure;
  }
  const std::size_t interior_bytes =
      checkpoint.interior_component_slabs.size() * sizeof(double);
  const std::size_t trace_bytes =
      checkpoint.owned_trace_component_slabs.size() * sizeof(double);
  const bool interior_imported = cuda_w8a::CudaTransportPolicy::HostToDevice(
          value.view.checkpoint_interior_slabs,
          checkpoint.interior_component_slabs.data(), interior_bytes,
          cuda_w8a::CudaTransferCategory::checkpoint_restore_import,
          cuda_w8a::CudaTransferSite::checkpoint_import_interior,
          interior_bytes, interior_bytes);
  if (!CollectiveAll(value.communicator.value, interior_imported))
    return ScalarCudaStatus::transfer_failure;
  const bool trace_imported = cuda_w8a::CudaTransportPolicy::HostToDevice(
          value.view.checkpoint_trace_slabs,
          checkpoint.owned_trace_component_slabs.data(), trace_bytes,
          cuda_w8a::CudaTransferCategory::checkpoint_restore_import,
          cuda_w8a::CudaTransferSite::checkpoint_import_trace,
          trace_bytes, trace_bytes);
  if (!CollectiveAll(value.communicator.value, trace_imported))
  {
    return ScalarCudaStatus::transfer_failure;
  }
  const int candidate = 1 - value.committed_slot;
  value.view.selected_current_slot = candidate;
  if (!value.LaunchRecover(detail::KernelOperation::unpack_checkpoint,
                           cuda_w8a::CudaFailurePhase::rollback_audit))
  {
    return ScalarCudaStatus::restart_admissibility_failure;
  }
  if (!value.StageOwnedTrace(value.view.slot_trace[candidate] +
                             value.owned_true_begin))
  {
    return ScalarCudaStatus::restart_admissibility_failure;
  }
  if (!value.LaunchRecover(detail::KernelOperation::install_distributed_trace,
                           cuda_w8a::CudaFailurePhase::rollback_audit))
  {
    return ScalarCudaStatus::restart_admissibility_failure;
  }
  if (!value.LaunchRecover(detail::KernelOperation::reduce_state,
                           cuda_w8a::CudaFailurePhase::diagnostics))
  {
    return ScalarCudaStatus::restart_admissibility_failure;
  }
  std::array<double, 16> metrics{};
  if (!value.CopyReduction(metrics))
  {
    return ScalarCudaStatus::transfer_failure;
  }
  const bool admissible = metrics[1] > 0.0 && metrics[2] > 0.0 &&
      metrics[3] > 0.0;
  if (!CollectiveAll(value.communicator.value, admissible))
  {
    return ScalarCudaStatus::restart_admissibility_failure;
  }
  StateHashes materialized_hashes{};
  if (!value.ComputeStateHashes(candidate, materialized_hashes))
  {
    return ScalarCudaStatus::transfer_failure;
  }
  if (!CollectiveAll(value.communicator.value,
                     SameStateHashes(materialized_hashes,
                                     checkpoint.raw_field_hashes)))
  {
    return ScalarCudaStatus::restart_hash_failure;
  }
  const int saved_committed = value.committed_slot;
  value.committed_slot = candidate;
  value.view.selected_current_slot = candidate;
  const bool assembled = value.LaunchAssemble(
      detail::KernelOperation::assemble,
      cuda_w8a::CudaFailurePhase::rollback_audit);
  value.committed_slot = saved_committed;
  if (!assembled)
  {
    return ScalarCudaStatus::restart_assembly_failure;
  }
  if (!value.CopyReduction(metrics))
  {
    return ScalarCudaStatus::transfer_failure;
  }
  if (!CollectiveAll(value.communicator.value, metrics[5] <= 1.0e-10))
  {
    return ScalarCudaStatus::restart_auxiliary_residual_failure;
  }
  const bool topology_matches = value.ComputeUploadedTopologyHashes();
  value.RefreshLedgers();
  const bool reusable = topology_matches &&
      value.evidence.allocations.zero_post_baseline_observed_growth &&
      value.evidence.allocations.zero_post_baseline_owned_allocation &&
      value.evidence.allocations.zero_post_baseline_owned_deallocation &&
      value.ObserveResidency() && value.evidence.reuse.unchanged;
  if (!CollectiveAll(value.communicator.value, reusable))
  {
    return ScalarCudaStatus::restart_auxiliary_residual_failure;
  }
  std::uint64_t restored_incarnation = 0;
  if (!CollectiveAll(value.communicator.value,
                     ClaimOwnerIncarnation(restored_incarnation)))
  {
    return ScalarCudaStatus::publication_failure;
  }
  value.committed_slot = candidate;
  value.view.committed_slot = candidate;
  value.view.selected_current_slot = candidate;
  value.coordinates = descriptor.coordinates;
  value.incarnation = restored_incarnation;
  value.status = ScalarCudaStatus::ready;
  value.first_failure = ScalarCudaStatus::uninitialized;
  value.evidence.status = ScalarCudaStatus::ready;
  value.evidence.first_failure = ScalarCudaStatus::uninitialized;
  value.evidence.accepted_steps =
      static_cast<int>(descriptor.coordinates.step);
  value.evidence.attempted_steps = 0;
  value.evidence.retry_attempts = 0;
  value.evidence.attempt_count = 0;
  value.evidence.step_evidence_count = 0;
  value.evidence.coordinates = value.coordinates;
  value.evidence.attempts = {};
  value.evidence.steps = {};
  value.evidence.snapshot_count = 0;
  value.evidence.snapshots = {};
  value.pending_attempt = {};
  value.pending_step = {};
  value.capture_snapshot = {};
  value.pending_snapshot = {};
  value.pending_attempt_active = false;
  value.prepare_retry_index = 0;
  value.last_device_status = detail::device_success;
  value.transaction_prior_evidence = {};
  value.transaction_prior_status = ScalarCudaStatus::uninitialized;
  value.transaction_prior_first_failure = ScalarCudaStatus::uninitialized;
  value.transaction_installed_after_retry = false;
  value.active_transaction_serial = 0;
  value.active_candidate_slot = -1;
  value.active_installed = false;
  return CollectiveAll(value.communicator.value, true)
      ? ScalarCudaStatus::ready
      : ScalarCudaStatus::publication_failure;
}


const ScalarCudaEvidence&
PersistentSc25ScalarIgrCudaRank2::trajectory_evidence() const noexcept
{
  static const ScalarCudaEvidence empty{};
  if (!implementation_) { return empty; }
  if (implementation_->active_transaction_serial != 0)
  {
    implementation_->RefreshExposedPriorEvidence();
    return implementation_->transaction_prior_evidence;
  }
  implementation_->RefreshLedgers();
  return implementation_->evidence;
}

ScalarCudaReuseEvidence
PersistentSc25ScalarIgrCudaRank2::reuse_evidence() const noexcept
{
  if (!implementation_) { return {}; }
  if (implementation_->active_transaction_serial != 0)
  {
    implementation_->RefreshExposedPriorEvidence();
    return implementation_->transaction_prior_evidence.reuse;
  }
  implementation_->RefreshLedgers();
  return implementation_->evidence.reuse;
}

cuda_w8a::CudaTransferLedgerV1
PersistentSc25ScalarIgrCudaRank2::transfer_ledger() const noexcept
{
  return cuda_w8a::CudaTransportPolicy::Ledger();
}

cuda_w8a::CudaAllocationLedgerV1
PersistentSc25ScalarIgrCudaRank2::allocation_ledger() const noexcept
{
  return cuda_w8a::UmpireDeviceAllocatorPolicy::Ledger();
}

} // namespace dg::hdg::runtime::cuda_w8b::sc25
