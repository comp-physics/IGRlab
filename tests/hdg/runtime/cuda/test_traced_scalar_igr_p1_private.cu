#include "numerics/hdg/runtime/common/traced_scalar_igr_p1_local_algebra.hpp"
#include "numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.hpp"
#include "numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private.cuh"

#include <cuda_runtime.h>
#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace
{
namespace algebra = dg::hdg::runtime::common::traced_scalar_igr_p1_local_algebra;
namespace cpu = dg::hdg::runtime::cpu::traced_scalar_igr_p1;
namespace packed = dg::hdg::runtime::cuda::traced_scalar_igr_p1_private;

constexpr double output_sentinel = -991.25;

struct Quadrature final
{
  std::vector<double> volume_xyw;
  std::vector<double> segment_xw;
};

struct HostBatch final
{
  int elements = 0;
  Quadrature quadrature;
  std::vector<double> vertices, interior, previous_interior, trace, previous_trace, rhs;
  std::vector<algebra::Parameters> parameters;
  std::vector<int> rhs_points;
};

struct HostOutputs final
{
  std::vector<algebra::Status> statuses;
  std::vector<double> ri, rt, A, B, C, D, f, g;
};

struct DeviceStorage final
{
  double *vertices = nullptr, *interior = nullptr, *previous_interior = nullptr;
  double *trace = nullptr, *previous_trace = nullptr, *rhs = nullptr, *volume = nullptr;
  double *segment = nullptr, *ri = nullptr, *rt = nullptr, *A = nullptr, *B = nullptr;
  double *C = nullptr, *D = nullptr, *f = nullptr, *g = nullptr;
  algebra::Parameters* parameters = nullptr;
  int* rhs_points = nullptr;
  algebra::Status* statuses = nullptr;
};

bool CudaOk(const cudaError_t error) { return error == cudaSuccess; }

template <typename T> bool Upload(T*& device, const std::vector<T>& host)
{
  device = nullptr;
  return CudaOk(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T))) &&
    CudaOk(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> bool Download(std::vector<T>& host, const T* device)
{
  return CudaOk(cudaMemcpy(host.data(), device, host.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T> void Release(T*& device)
{
  if (device) { cudaFree(device); device = nullptr; }
}

void Release(DeviceStorage& storage)
{
  Release(storage.vertices); Release(storage.interior); Release(storage.previous_interior);
  Release(storage.trace); Release(storage.previous_trace); Release(storage.rhs); Release(storage.volume);
  Release(storage.segment); Release(storage.ri); Release(storage.rt); Release(storage.A); Release(storage.B);
  Release(storage.C); Release(storage.D); Release(storage.f); Release(storage.g);
  Release(storage.parameters); Release(storage.rhs_points); Release(storage.statuses);
}

Quadrature MakeQuadrature()
{
  constexpr int order = 8;
  const mfem::IntegrationRule& volume = mfem::IntRules.Get(mfem::Geometry::SQUARE, order);
  const mfem::IntegrationRule& segment = mfem::IntRules.Get(mfem::Geometry::SEGMENT, order);
  Quadrature tables;
  tables.volume_xyw.resize(static_cast<std::size_t>(volume.GetNPoints()) * 3);
  tables.segment_xw.resize(static_cast<std::size_t>(segment.GetNPoints()) * 2);
  for (int point = 0; point < volume.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = volume.IntPoint(point);
    tables.volume_xyw[3 * point] = ip.x;
    tables.volume_xyw[3 * point + 1] = ip.y;
    tables.volume_xyw[3 * point + 2] = ip.weight;
  }
  for (int point = 0; point < segment.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = segment.IntPoint(point);
    tables.segment_xw[2 * point] = ip.x;
    tables.segment_xw[2 * point + 1] = ip.weight;
  }
  return tables;
}

HostBatch MakeBatch(const int elements)
{
  HostBatch batch;
  batch.elements = elements;
  batch.quadrature = MakeQuadrature();
  const int volume_points = static_cast<int>(batch.quadrature.volume_xyw.size() / 3);
  batch.vertices.resize(static_cast<std::size_t>(elements) * packed::vertices_per_element);
  batch.parameters.resize(elements);
  batch.interior.resize(static_cast<std::size_t>(elements) * algebra::interior_unknowns);
  batch.previous_interior.resize(static_cast<std::size_t>(elements) * algebra::interior_unknowns);
  batch.trace.resize(static_cast<std::size_t>(elements) * algebra::trace_unknowns);
  batch.previous_trace.resize(static_cast<std::size_t>(elements) * algebra::trace_unknowns);
  batch.rhs.resize(static_cast<std::size_t>(elements) * volume_points * 4);
  batch.rhs_points.assign(elements, volume_points);
  for (int element = 0; element < elements; ++element)
  {
    const std::size_t v = static_cast<std::size_t>(element) * packed::vertices_per_element;
    const double x = .2 + 2.7 * element;
    const double y = -.1 + .15 * element;
    const double vertices[8] = {x, y, x + 2., y + .3, x + 2.4, y + 1.6, x + .4, y + 1.3};
    std::copy(vertices, vertices + 8, batch.vertices.begin() + v);
    batch.parameters[element] = {1.4, .7 + .03 * element, 1.3, .09, 1.e-12};
    const std::size_t i = static_cast<std::size_t>(element) * algebra::interior_unknowns;
    const std::size_t t = static_cast<std::size_t>(element) * algebra::trace_unknowns;
    for (int component = 0; component < 4; ++component)
      for (int q = 0; q < 4; ++q)
      {
        batch.interior[i + 4 * component + q] = component == 0 ? 1. + .02 * q + .01 * element :
          component == 1 ? .15 + .01 * q : component == 2 ? -.05 + .008 * q : 2.7 + .04 * q;
        batch.previous_interior[i + 4 * component + q] =
          batch.interior[i + 4 * component + q] - .003 * (component + 1);
      }
    for (int q = 0; q < 4; ++q)
    {
      batch.interior[i + 16 + q] = .9 + .03 * q;
      batch.previous_interior[i + 16 + q] = batch.interior[i + 16 + q] - .01;
      batch.interior[i + 20 + q] = .02 + .01 * q;
      batch.previous_interior[i + 20 + q] = batch.interior[i + 20 + q];
      batch.interior[i + 24 + q] = -.01 + .004 * q;
      batch.previous_interior[i + 24 + q] = batch.interior[i + 24 + q];
    }
    for (int component = 0; component < 5; ++component)
      for (int q = 0; q < 8; ++q)
      {
        batch.trace[t + 8 * component + q] = (component == 0 ? 1.01 : component == 1 ? .14 :
          component == 2 ? -.04 : component == 3 ? 2.71 : .92) + .002 * q + .001 * element;
        batch.previous_trace[t + 8 * component + q] = batch.trace[t + 8 * component + q] - .001;
      }
    for (int point = 0; point < volume_points; ++point)
      for (int component = 0; component < 4; ++component)
        batch.rhs[(static_cast<std::size_t>(element) * volume_points + point) * 4 + component] =
          .0005 * (element + 1) * (point + 1) * (component + 1);
  }
  return batch;
}

HostOutputs SentinelOutputs(const int elements)
{
  HostOutputs output;
  output.statuses.assign(elements, algebra::Status::invalid_parameter);
  output.ri.assign(static_cast<std::size_t>(elements) * algebra::interior_unknowns, output_sentinel);
  output.rt.assign(static_cast<std::size_t>(elements) * algebra::trace_unknowns, output_sentinel);
  output.A.assign(static_cast<std::size_t>(elements) * algebra::interior_unknowns * algebra::interior_unknowns, output_sentinel);
  output.B.assign(static_cast<std::size_t>(elements) * algebra::interior_unknowns * algebra::trace_unknowns, output_sentinel);
  output.C.assign(static_cast<std::size_t>(elements) * algebra::trace_unknowns * algebra::interior_unknowns, output_sentinel);
  output.D.assign(static_cast<std::size_t>(elements) * algebra::trace_unknowns * algebra::trace_unknowns, output_sentinel);
  output.f.assign(static_cast<std::size_t>(elements) * algebra::interior_unknowns, output_sentinel);
  output.g.assign(static_cast<std::size_t>(elements) * algebra::trace_unknowns, output_sentinel);
  return output;
}

bool UploadBatch(const HostBatch& batch, HostOutputs& output, DeviceStorage& storage)
{
  return Upload(storage.vertices, batch.vertices) && Upload(storage.parameters, batch.parameters) &&
    Upload(storage.interior, batch.interior) && Upload(storage.previous_interior, batch.previous_interior) &&
    Upload(storage.trace, batch.trace) && Upload(storage.previous_trace, batch.previous_trace) &&
    Upload(storage.rhs, batch.rhs) && Upload(storage.rhs_points, batch.rhs_points) &&
    Upload(storage.volume, batch.quadrature.volume_xyw) && Upload(storage.segment, batch.quadrature.segment_xw) &&
    Upload(storage.statuses, output.statuses) && Upload(storage.ri, output.ri) && Upload(storage.rt, output.rt) &&
    Upload(storage.A, output.A) && Upload(storage.B, output.B) && Upload(storage.C, output.C) &&
    Upload(storage.D, output.D) && Upload(storage.f, output.f) && Upload(storage.g, output.g);
}

bool Run(const HostBatch& batch, HostOutputs& output)
{
  DeviceStorage storage;
  bool ok = UploadBatch(batch, output, storage);
  if (ok)
  {
    const packed::BatchInput input{batch.elements, storage.vertices, storage.parameters,
      storage.interior, storage.previous_interior, storage.trace, storage.previous_trace,
      storage.rhs, storage.rhs_points, storage.volume,
      static_cast<int>(batch.quadrature.volume_xyw.size() / 3), storage.segment,
      static_cast<int>(batch.quadrature.segment_xw.size() / 2)};
    const packed::BatchOutput device_output{storage.statuses, storage.ri, storage.rt, storage.A,
      storage.B, storage.C, storage.D, storage.f, storage.g};
    ok = CudaOk(packed::Launch(input, device_output)) && CudaOk(cudaDeviceSynchronize()) &&
      Download(output.statuses, storage.statuses) && Download(output.ri, storage.ri) &&
      Download(output.rt, storage.rt) && Download(output.A, storage.A) && Download(output.B, storage.B) &&
      Download(output.C, storage.C) && Download(output.D, storage.D) && Download(output.f, storage.f) &&
      Download(output.g, storage.g);
  }
  Release(storage);
  return ok;
}

bool Near(const double left, const double right)
{
  return std::abs(left - right) <= 4.e-10 * std::max({1., std::abs(left), std::abs(right)});
}

bool Same(const double* left, const double* right, const std::size_t count)
{
  for (std::size_t i = 0; i < count; ++i) if (!Near(left[i], right[i])) return false;
  return true;
}

bool MatchesCpu(const HostBatch& batch, const HostOutputs& output)
{
  const int volume_points = static_cast<int>(batch.quadrature.volume_xyw.size() / 3);
  for (int element = 0; element < batch.elements; ++element)
  {
    cpu::AffineQuad quad{};
    const std::size_t vertex = static_cast<std::size_t>(element) * packed::vertices_per_element;
    for (int point = 0; point < 4; ++point)
      for (int direction = 0; direction < 2; ++direction)
        quad.vertices[point][direction] = batch.vertices[vertex + 2 * point + direction];
    const algebra::Parameters raw = batch.parameters[element];
    const cpu::Parameters parameters{raw.gamma, raw.alpha, raw.beta, raw.dt, raw.density_floor, 8};
    const std::size_t i = static_cast<std::size_t>(element) * algebra::interior_unknowns;
    const std::size_t t = static_cast<std::size_t>(element) * algebra::trace_unknowns;
    const std::size_t rhs = static_cast<std::size_t>(element) * volume_points * 4;
    const cpu::ElementState state{batch.interior.data() + i, batch.previous_interior.data() + i,
      batch.trace.data() + t, batch.previous_trace.data() + t,
      {batch.rhs.data() + rhs, static_cast<std::size_t>(batch.rhs_points[element])}};
    cpu::LocalOperator expected;
    if (cpu::Assemble(quad, parameters, state, expected) != cpu::Status::success ||
        output.statuses[element] != algebra::Status::success ||
        !Same(expected.interior_residual.data(), output.ri.data() + i, algebra::interior_unknowns) ||
        !Same(expected.trace_residual.data(), output.rt.data() + t, algebra::trace_unknowns) ||
        !Same(expected.A.data(), output.A.data() + static_cast<std::size_t>(element) * 28 * 28, 28 * 28) ||
        !Same(expected.B.data(), output.B.data() + static_cast<std::size_t>(element) * 28 * 40, 28 * 40) ||
        !Same(expected.C.data(), output.C.data() + static_cast<std::size_t>(element) * 40 * 28, 40 * 28) ||
        !Same(expected.D.data(), output.D.data() + static_cast<std::size_t>(element) * 40 * 40, 40 * 40) ||
        !Same(expected.f.data(), output.f.data() + i, algebra::interior_unknowns) ||
        !Same(expected.g.data(), output.g.data() + t, algebra::trace_unknowns)) return false;
  }
  return true;
}

bool SliceIs(const std::vector<double>& values, const std::size_t offset,
             const std::size_t count, const double expected)
{
  return std::all_of(values.begin() + static_cast<std::ptrdiff_t>(offset),
                     values.begin() + static_cast<std::ptrdiff_t>(offset + count),
                     [expected](const double value) { return value == expected; });
}

bool ElementIs(const HostOutputs& output, const int element, const double expected)
{
  return SliceIs(output.ri, static_cast<std::size_t>(element) * 28, 28, expected) &&
    SliceIs(output.rt, static_cast<std::size_t>(element) * 40, 40, expected) &&
    SliceIs(output.A, static_cast<std::size_t>(element) * 28 * 28, 28 * 28, expected) &&
    SliceIs(output.B, static_cast<std::size_t>(element) * 28 * 40, 28 * 40, expected) &&
    SliceIs(output.C, static_cast<std::size_t>(element) * 40 * 28, 40 * 28, expected) &&
    SliceIs(output.D, static_cast<std::size_t>(element) * 40 * 40, 40 * 40, expected) &&
    SliceIs(output.f, static_cast<std::size_t>(element) * 28, 28, expected) &&
    SliceIs(output.g, static_cast<std::size_t>(element) * 40, 40, expected);
}

bool CheckPackedCpuAgreement()
{
  const HostBatch batch = MakeBatch(3);
  HostOutputs output = SentinelOutputs(batch.elements);
  return Run(batch, output) && MatchesCpu(batch, output);
}

bool CheckStatusIsolationAndPreservation()
{
  HostBatch batch = MakeBatch(4);
  batch.vertices[static_cast<std::size_t>(1) * 8 + 4] += .25; // non-affine geometry
  for (int q = 0; q < 4; ++q)
    batch.interior[static_cast<std::size_t>(2) * algebra::interior_unknowns + q] = 0.; // density
  batch.rhs_points[3] -= 1; // source table/layout mismatch
  HostOutputs output = SentinelOutputs(batch.elements);
  if (!Run(batch, output) || output.statuses[0] != algebra::Status::success ||
      output.statuses[1] != algebra::Status::invalid_geometry ||
      output.statuses[2] != algebra::Status::inadmissible_state ||
      output.statuses[3] != algebra::Status::invalid_layout) return false;
  // Geometry and source-layout rejection happen before common algebra clears
  // output, while inadmissible state clears its private element only.
  return ElementIs(output, 1, output_sentinel) && ElementIs(output, 2, 0.) &&
    ElementIs(output, 3, output_sentinel) && !ElementIs(output, 0, output_sentinel);
}
} // namespace

int main()
{
  if (!CheckPackedCpuAgreement())
  {
    std::cerr << "traced_scalar_igr_p1_private packed CPU agreement failed\n";
    return 1;
  }
  if (!CheckStatusIsolationAndPreservation())
  {
    std::cerr << "traced_scalar_igr_p1_private status isolation failed\n";
    return 1;
  }
  std::cout << "traced_scalar_igr_p1_private PASS elements=3 source=per-element\n";
  return 0;
}
