#include "numerics/hdg/runtime/common/traced_scalar_igr_p1_local_algebra.hpp"
#include "numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.hpp"

#include <cuda_runtime.h>
#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
namespace algebra = dg::hdg::runtime::common::traced_scalar_igr_p1_local_algebra;
namespace cpu = dg::hdg::runtime::cpu::traced_scalar_igr_p1;

struct Quadrature final
{
  std::vector<double> volume_xyw;
  std::vector<double> segment_xw;
};

struct DeviceResult final
{
  algebra::Status status = algebra::Status::invalid_parameter;
  std::vector<double> interior_residual, trace_residual, A, B, C, D, f, g;
};

__global__ void AssembleOne(const algebra::Input input, algebra::Output output,
                            algebra::Status* status)
{
  *status = algebra::Assemble(input, output);
}

__global__ void MassFluxJacobianWitness(double* values)
{
  const double state[4] = {1.2, .3, -.2, 3.1};
  const double normal[2] = {.6, -.8};
  const algebra::Parameters parameters{1.4, .7, 1.3, .09, 1.e-12};
  double flux[4] = {}, jacobian[20] = {};
  if (!algebra::EulerSigma(state, .4, normal, parameters, flux, jacobian))
  {
    values[0] = 0.; values[1] = 0.; return;
  }
  values[0] = jacobian[1];
  values[1] = jacobian[2];
}

bool CudaOk(const cudaError_t error) { return error == cudaSuccess; }

bool Near(const double left, const double right, const double tolerance = 3.e-10)
{ return std::abs(left - right) <= tolerance * std::max({1., std::abs(left), std::abs(right)}); }

bool Same(const std::vector<double>& left, const std::vector<double>& right)
{
  return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
    [](const double a, const double b) { return Near(a, b); });
}

bool SameStatus(const cpu::Status cpu_status, const algebra::Status device_status)
{
  switch (cpu_status)
  {
    case cpu::Status::success: return device_status == algebra::Status::success;
    case cpu::Status::invalid_parameter: return device_status == algebra::Status::invalid_parameter;
    case cpu::Status::invalid_geometry: return device_status == algebra::Status::invalid_geometry;
    case cpu::Status::invalid_layout: return device_status == algebra::Status::invalid_layout;
    case cpu::Status::inadmissible_state: return device_status == algebra::Status::inadmissible_state;
    case cpu::Status::nonfinite_result: return device_status == algebra::Status::nonfinite_result;
  }
  return false;
}

Quadrature MakeQuadrature(const int order)
{
  const mfem::IntegrationRule& volume = mfem::IntRules.Get(mfem::Geometry::SQUARE, order);
  const mfem::IntegrationRule& segment = mfem::IntRules.Get(mfem::Geometry::SEGMENT, order);
  Quadrature result;
  result.volume_xyw.resize(static_cast<std::size_t>(volume.GetNPoints()) * 3);
  result.segment_xw.resize(static_cast<std::size_t>(segment.GetNPoints()) * 2);
  for (int point = 0; point < volume.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = volume.IntPoint(point);
    result.volume_xyw[3 * point] = ip.x;
    result.volume_xyw[3 * point + 1] = ip.y;
    result.volume_xyw[3 * point + 2] = ip.weight;
  }
  for (int point = 0; point < segment.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = segment.IntPoint(point);
    result.segment_xw[2 * point] = ip.x;
    result.segment_xw[2 * point + 1] = ip.weight;
  }
  return result;
}

template <typename T> bool Upload(T*& device, const T* host, const std::size_t count)
{
  device = nullptr;
  return CudaOk(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(T))) &&
    CudaOk(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> bool Download(T* host, const T* device, const std::size_t count)
{ return CudaOk(cudaMemcpy(host, device, count * sizeof(T), cudaMemcpyDeviceToHost)); }

template <typename T> bool Allocate(T*& device, const std::size_t count)
{ return CudaOk(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(T))); }

template <typename T> void Release(T*& device)
{ if (device) { cudaFree(device); device = nullptr; } }

bool CheckMassFluxJacobianWitness()
{
  double* device_values = nullptr;
  double values[2] = {};
  bool ok = Allocate(device_values, 2);
  if (ok)
  {
    MassFluxJacobianWitness<<<1, 1>>>(device_values);
    ok = CudaOk(cudaGetLastError()) && CudaOk(cudaDeviceSynchronize()) &&
      Download(values, device_values, 2) && Near(values[0], .6) && Near(values[1], -.8);
  }
  Release(device_values);
  return ok;
}

bool DeviceAssemble(const std::array<double, 8>& vertices, const algebra::Parameters& parameters,
                    const std::vector<double>& interior, const std::vector<double>& previous_interior,
                    const std::vector<double>& trace, const std::vector<double>& previous_trace,
                    const Quadrature& quadrature, const double* rhs, const int rhs_points,
                    DeviceResult& result)
{
  double *d_vertices = nullptr, *d_interior = nullptr, *d_previous_interior = nullptr;
  double *d_trace = nullptr, *d_previous_trace = nullptr, *d_volume = nullptr, *d_segment = nullptr;
  double *d_rhs = nullptr, *d_ri = nullptr, *d_rt = nullptr, *d_A = nullptr, *d_B = nullptr;
  double *d_C = nullptr, *d_D = nullptr, *d_f = nullptr, *d_g = nullptr;
  algebra::Status* d_status = nullptr;
  const bool allocated =
    Upload(d_vertices, vertices.data(), vertices.size()) &&
    Upload(d_interior, interior.data(), interior.size()) &&
    Upload(d_previous_interior, previous_interior.data(), previous_interior.size()) &&
    Upload(d_trace, trace.data(), trace.size()) &&
    Upload(d_previous_trace, previous_trace.data(), previous_trace.size()) &&
    Upload(d_volume, quadrature.volume_xyw.data(), quadrature.volume_xyw.size()) &&
    Upload(d_segment, quadrature.segment_xw.data(), quadrature.segment_xw.size()) &&
    (!rhs || Upload(d_rhs, rhs, static_cast<std::size_t>(rhs_points) * 4)) &&
    Allocate(d_ri, algebra::interior_unknowns) &&
    Allocate(d_rt, algebra::trace_unknowns) &&
    Allocate(d_A, algebra::interior_unknowns * algebra::interior_unknowns) &&
    Allocate(d_B, algebra::interior_unknowns * algebra::trace_unknowns) &&
    Allocate(d_C, algebra::trace_unknowns * algebra::interior_unknowns) &&
    Allocate(d_D, algebra::trace_unknowns * algebra::trace_unknowns) &&
    Allocate(d_f, algebra::interior_unknowns) &&
    Allocate(d_g, algebra::trace_unknowns) &&
    Allocate(d_status, 1);
  bool ok = allocated;
  if (ok)
  {
    const algebra::Input input{d_vertices, parameters, d_interior, d_previous_interior,
                               d_trace, d_previous_trace, d_rhs, rhs_points,
                               {d_volume, static_cast<int>(quadrature.volume_xyw.size() / 3),
                                d_segment, static_cast<int>(quadrature.segment_xw.size() / 2)}};
    const algebra::Output output{d_ri, d_rt, d_A, d_B, d_C, d_D, d_f, d_g};
    AssembleOne<<<1, 1>>>(input, output, d_status);
    ok = CudaOk(cudaGetLastError()) && CudaOk(cudaDeviceSynchronize()) &&
      Download(&result.status, d_status, 1);
    if (ok && result.status == algebra::Status::success)
    {
      result.interior_residual.resize(algebra::interior_unknowns);
      result.trace_residual.resize(algebra::trace_unknowns);
      result.A.resize(algebra::interior_unknowns * algebra::interior_unknowns);
      result.B.resize(algebra::interior_unknowns * algebra::trace_unknowns);
      result.C.resize(algebra::trace_unknowns * algebra::interior_unknowns);
      result.D.resize(algebra::trace_unknowns * algebra::trace_unknowns);
      result.f.resize(algebra::interior_unknowns); result.g.resize(algebra::trace_unknowns);
      ok = Download(result.interior_residual.data(), d_ri, result.interior_residual.size()) &&
        Download(result.trace_residual.data(), d_rt, result.trace_residual.size()) &&
        Download(result.A.data(), d_A, result.A.size()) && Download(result.B.data(), d_B, result.B.size()) &&
        Download(result.C.data(), d_C, result.C.size()) && Download(result.D.data(), d_D, result.D.size()) &&
        Download(result.f.data(), d_f, result.f.size()) && Download(result.g.data(), d_g, result.g.size());
    }
  }
  Release(d_vertices); Release(d_interior); Release(d_previous_interior); Release(d_trace);
  Release(d_previous_trace); Release(d_volume); Release(d_segment); Release(d_rhs); Release(d_ri);
  Release(d_rt); Release(d_A); Release(d_B); Release(d_C); Release(d_D); Release(d_f); Release(d_g);
  Release(d_status);
  return ok;
}

void FillState(std::vector<double>& interior, std::vector<double>& previous_interior,
               std::vector<double>& trace, std::vector<double>& previous_trace)
{
  interior.resize(28); previous_interior.resize(28); trace.resize(40); previous_trace.resize(40);
  for (int component = 0; component < 4; ++component)
    for (int q = 0; q < 4; ++q)
    {
      interior[component * 4 + q] = component == 0 ? 1.0 + .02 * q :
        component == 1 ? .15 + .01 * q : component == 2 ? -.05 + .008 * q : 2.7 + .04 * q;
      previous_interior[component * 4 + q] = interior[component * 4 + q] - .003 * (component + 1);
    }
  for (int q = 0; q < 4; ++q)
  {
    interior[16 + q] = .9 + .03 * q; previous_interior[16 + q] = interior[16 + q] - .01;
    interior[20 + q] = .02 + .01 * q; previous_interior[20 + q] = interior[20 + q];
    interior[24 + q] = -.01 + .004 * q; previous_interior[24 + q] = interior[24 + q];
  }
  for (int component = 0; component < 5; ++component)
    for (int q = 0; q < 8; ++q)
    {
      trace[component * 8 + q] = (component == 0 ? 1.01 : component == 1 ? .14 :
        component == 2 ? -.04 : component == 3 ? 2.71 : .92) + .002 * q;
      previous_trace[component * 8 + q] = trace[component * 8 + q] - .001;
    }
}

bool CheckCase(const int order, const bool nonzero_source)
{
  cpu::AffineQuad geometry{{{{.2, -.1}, {2.2, .2}, {2.6, 1.5}, {.6, 1.2}}}};
  std::array<double, 8> vertices{};
  for (int vertex = 0; vertex < 4; ++vertex)
    for (int direction = 0; direction < 2; ++direction)
      vertices[2 * vertex + direction] = geometry.vertices[vertex][direction];
  cpu::Parameters parameters{};
  parameters.alpha = .7; parameters.beta = 1.3; parameters.dt = .09; parameters.quadrature_order = order;
  std::vector<double> interior, previous_interior, trace, previous_trace;
  FillState(interior, previous_interior, trace, previous_trace);
  const Quadrature quadrature = MakeQuadrature(order);
  std::vector<double> rhs(quadrature.volume_xyw.size() / 3 * 4, 0.);
  if (nonzero_source)
    for (std::size_t point = 0; point < rhs.size() / 4; ++point) rhs[4 * point + 3] = .03;
  const cpu::ElementState state{interior.data(), previous_interior.data(), trace.data(),
                                previous_trace.data(),
                                {nonzero_source ? rhs.data() : nullptr, nonzero_source ? rhs.size() / 4 : 0}};
  cpu::LocalOperator host;
  if (cpu::Assemble(geometry, parameters, state, host) != cpu::Status::success) return false;
  DeviceResult device;
  const algebra::Parameters raw_parameters{parameters.gamma, parameters.alpha, parameters.beta,
                                           parameters.dt, parameters.density_floor};
  if (!DeviceAssemble(vertices, raw_parameters, interior, previous_interior, trace, previous_trace,
                      quadrature, nonzero_source ? rhs.data() : nullptr,
                      nonzero_source ? static_cast<int>(rhs.size() / 4) : 0, device) ||
      device.status != algebra::Status::success) return false;
  return Same(host.interior_residual, device.interior_residual) &&
    Same(host.trace_residual, device.trace_residual) && Same(host.A, device.A) &&
    Same(host.B, device.B) && Same(host.C, device.C) && Same(host.D, device.D) &&
    Same(host.f, device.f) && Same(host.g, device.g);
}

bool CheckSourceDifferential()
{
  cpu::AffineQuad geometry{{{{.2, -.1}, {2.2, .2}, {2.6, 1.5}, {.6, 1.2}}}};
  std::array<double, 8> vertices{};
  for (int vertex = 0; vertex < 4; ++vertex)
    for (int direction = 0; direction < 2; ++direction)
      vertices[2 * vertex + direction] = geometry.vertices[vertex][direction];
  cpu::Parameters parameters{};
  parameters.alpha = .7; parameters.beta = 1.3; parameters.dt = .09; parameters.quadrature_order = 12;
  std::vector<double> interior, previous_interior, trace, previous_trace;
  FillState(interior, previous_interior, trace, previous_trace);
  const Quadrature quadrature = MakeQuadrature(parameters.quadrature_order);
  std::vector<double> rhs(quadrature.volume_xyw.size() / 3 * 4, 0.);
  for (std::size_t point = 0; point < rhs.size() / 4; ++point) rhs[4 * point + 3] = .03;
  const cpu::ElementState zero{interior.data(), previous_interior.data(), trace.data(), previous_trace.data(), {}};
  const cpu::ElementState sourced{interior.data(), previous_interior.data(), trace.data(), previous_trace.data(),
                                  {rhs.data(), rhs.size() / 4}};
  cpu::LocalOperator host_zero, host_sourced;
  if (cpu::Assemble(geometry, parameters, zero, host_zero) != cpu::Status::success ||
      cpu::Assemble(geometry, parameters, sourced, host_sourced) != cpu::Status::success ||
      !Same(host_zero.A, host_sourced.A) || !Same(host_zero.B, host_sourced.B) ||
      !Same(host_zero.C, host_sourced.C) || !Same(host_zero.D, host_sourced.D)) return false;
  bool changed_u = false;
  for (int row = 0; row < algebra::interior_unknowns; ++row)
  {
    if (row < 16) changed_u = changed_u || !Near(host_zero.interior_residual[row], host_sourced.interior_residual[row]);
    else if (!Near(host_zero.interior_residual[row], host_sourced.interior_residual[row])) return false;
  }
  if (!changed_u || !Same(host_zero.trace_residual, host_sourced.trace_residual)) return false;
  const algebra::Parameters raw_parameters{parameters.gamma, parameters.alpha, parameters.beta,
                                           parameters.dt, parameters.density_floor};
  DeviceResult device_zero, device_sourced;
  if (!DeviceAssemble(vertices, raw_parameters, interior, previous_interior, trace, previous_trace,
                      quadrature, nullptr, 0, device_zero) ||
      !DeviceAssemble(vertices, raw_parameters, interior, previous_interior, trace, previous_trace,
                      quadrature, rhs.data(), static_cast<int>(rhs.size() / 4), device_sourced) ||
      device_zero.status != algebra::Status::success || device_sourced.status != algebra::Status::success)
    return false;
  return Same(host_zero.interior_residual, device_zero.interior_residual) &&
    Same(host_sourced.interior_residual, device_sourced.interior_residual) &&
    Same(host_zero.trace_residual, device_zero.trace_residual) &&
    Same(host_sourced.trace_residual, device_sourced.trace_residual) &&
    Same(host_zero.f, device_zero.f) && Same(host_sourced.f, device_sourced.f) &&
    Same(host_zero.g, device_zero.g) && Same(host_sourced.g, device_sourced.g) &&
    Same(device_zero.A, device_sourced.A) && Same(device_zero.B, device_sourced.B) &&
    Same(device_zero.C, device_sourced.C) && Same(device_zero.D, device_sourced.D);
}

bool CheckStatuses()
{
  cpu::AffineQuad geometry{{{{.2, -.1}, {2.2, .2}, {2.6, 1.5}, {.6, 1.2}}}};
  std::array<double, 8> vertices{};
  for (int vertex = 0; vertex < 4; ++vertex)
    for (int direction = 0; direction < 2; ++direction) vertices[2 * vertex + direction] = geometry.vertices[vertex][direction];
  cpu::Parameters parameters{}; parameters.alpha = .7; parameters.beta = 1.3; parameters.dt = .09;
  const Quadrature quadrature = MakeQuadrature(parameters.quadrature_order);
  std::vector<double> interior, previous_interior, trace, previous_trace;
  FillState(interior, previous_interior, trace, previous_trace);
  std::vector<double> rhs(quadrature.volume_xyw.size() / 3 * 4, 0.);
  const algebra::Parameters raw_parameters{parameters.gamma, parameters.alpha, parameters.beta,
                                           parameters.dt, parameters.density_floor};
  const auto compare = [&](const cpu::AffineQuad& cpu_geometry, const cpu::Parameters& cpu_parameters,
                           const std::vector<double>& current, const std::vector<double>& previous,
                           const std::vector<double>& current_trace, const std::vector<double>& previous_trace_input,
                           const cpu::ElementState& state, std::array<double, 8> raw_vertices,
                           algebra::Parameters raw, const double* raw_rhs, const int rhs_points)
  {
    cpu::LocalOperator host;
    DeviceResult device;
    return DeviceAssemble(raw_vertices, raw, current, previous, current_trace, previous_trace_input,
                          quadrature, raw_rhs, rhs_points, device) &&
      SameStatus(cpu::Assemble(cpu_geometry, cpu_parameters, state, host), device.status);
  };
  cpu::AffineQuad reversed = geometry; std::swap(reversed.vertices[1], reversed.vertices[3]);
  std::array<double, 8> reversed_vertices = vertices;
  for (int direction = 0; direction < 2; ++direction)
    std::swap(reversed_vertices[2 + direction], reversed_vertices[6 + direction]);
  const cpu::ElementState regular{interior.data(), previous_interior.data(), trace.data(), previous_trace.data(), {}};
  if (!compare(reversed, parameters, interior, previous_interior, trace, previous_trace, regular,
               reversed_vertices, raw_parameters, nullptr, 0)) return false;
  cpu::AffineQuad non_affine = geometry; non_affine.vertices[2][0] += .05;
  std::array<double, 8> non_affine_vertices = vertices; non_affine_vertices[4] += .05;
  if (!compare(non_affine, parameters, interior, previous_interior, trace, previous_trace, regular,
               non_affine_vertices, raw_parameters, nullptr, 0)) return false;
  cpu::AffineQuad nan_geometry = geometry;
  nan_geometry.vertices[2][0] = std::numeric_limits<double>::quiet_NaN();
  std::array<double, 8> nan_vertices = vertices;
  nan_vertices[4] = std::numeric_limits<double>::quiet_NaN();
  if (!compare(nan_geometry, parameters, interior, previous_interior, trace, previous_trace, regular,
               nan_vertices, raw_parameters, nullptr, 0)) return false;
  cpu::Parameters bad_parameters = parameters; bad_parameters.gamma = 1.;
  algebra::Parameters bad_raw = raw_parameters; bad_raw.gamma = 1.;
  if (!compare(geometry, bad_parameters, interior, previous_interior, trace, previous_trace, regular,
               vertices, bad_raw, nullptr, 0)) return false;
  const cpu::ElementState bad_source{interior.data(), previous_interior.data(), trace.data(), previous_trace.data(),
                                     {rhs.data(), rhs.size() / 4 - 1}};
  if (!compare(geometry, parameters, interior, previous_interior, trace, previous_trace, bad_source,
               vertices, raw_parameters, rhs.data(), static_cast<int>(rhs.size() / 4 - 1))) return false;
  std::vector<double> nonfinite_rhs = rhs;
  nonfinite_rhs[3] = std::numeric_limits<double>::quiet_NaN();
  const cpu::ElementState nonfinite_source{interior.data(), previous_interior.data(), trace.data(), previous_trace.data(),
                                           {nonfinite_rhs.data(), nonfinite_rhs.size() / 4}};
  if (!compare(geometry, parameters, interior, previous_interior, trace, previous_trace, nonfinite_source,
               vertices, raw_parameters, nonfinite_rhs.data(), static_cast<int>(nonfinite_rhs.size() / 4))) return false;
  std::vector<double> inadmissible = interior; inadmissible[0] = 0.;
  const cpu::ElementState bad_state{inadmissible.data(), previous_interior.data(), trace.data(), previous_trace.data(), {}};
  if (!compare(geometry, parameters, inadmissible, previous_interior, trace, previous_trace, bad_state,
               vertices, raw_parameters, nullptr, 0)) return false;
  std::vector<double> previous_density = previous_interior; previous_density[0] = 0.;
  const cpu::ElementState bad_previous_density{interior.data(), previous_density.data(), trace.data(), previous_trace.data(), {}};
  if (!compare(geometry, parameters, interior, previous_density, trace, previous_trace, bad_previous_density,
               vertices, raw_parameters, nullptr, 0)) return false;
  std::vector<double> previous_pressure = previous_interior;
  for (int q = 0; q < 4; ++q) previous_pressure[12 + q] = .001;
  const cpu::ElementState bad_previous_pressure{interior.data(), previous_pressure.data(), trace.data(), previous_trace.data(), {}};
  if (!compare(geometry, parameters, interior, previous_pressure, trace, previous_trace, bad_previous_pressure,
               vertices, raw_parameters, nullptr, 0)) return false;
  std::vector<double> trace_density = trace;
  for (int q = 0; q < 8; ++q) trace_density[q] = 0.;
  const cpu::ElementState bad_trace_density{interior.data(), previous_interior.data(), trace_density.data(), previous_trace.data(), {}};
  if (!compare(geometry, parameters, interior, previous_interior, trace_density, previous_trace, bad_trace_density,
               vertices, raw_parameters, nullptr, 0)) return false;
  std::vector<double> previous_trace_density = previous_trace;
  for (int q = 0; q < 8; ++q) previous_trace_density[q] = 0.;
  const cpu::ElementState bad_previous_trace_density{interior.data(), previous_interior.data(), trace.data(),
                                                     previous_trace_density.data(), {}};
  if (!compare(geometry, parameters, interior, previous_interior, trace, previous_trace_density,
               bad_previous_trace_density, vertices, raw_parameters, nullptr, 0)) return false;
  std::vector<double> trace_pressure = trace;
  for (int q = 0; q < 8; ++q) trace_pressure[24 + q] = .001;
  const cpu::ElementState bad_trace_pressure{interior.data(), previous_interior.data(), trace_pressure.data(), previous_trace.data(), {}};
  if (!compare(geometry, parameters, interior, previous_interior, trace_pressure, previous_trace, bad_trace_pressure,
               vertices, raw_parameters, nullptr, 0)) return false;
  std::vector<double> previous_trace_pressure = previous_trace;
  for (int q = 0; q < 8; ++q) previous_trace_pressure[24 + q] = .001;
  const cpu::ElementState bad_previous_trace_pressure{interior.data(), previous_interior.data(), trace.data(),
                                                      previous_trace_pressure.data(), {}};
  return compare(geometry, parameters, interior, previous_interior, trace, previous_trace_pressure,
                 bad_previous_trace_pressure, vertices, raw_parameters, nullptr, 0);
}
} // namespace

int main()
{
  return CheckMassFluxJacobianWitness() && CheckCase(12, true) && CheckCase(14, false) && CheckSourceDifferential() &&
    CheckStatuses() ? 0 : 1;
}
