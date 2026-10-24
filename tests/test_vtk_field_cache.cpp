#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "io/vtk_output.hpp"
#include "../src/io/vtk_field_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{
namespace cache = mfem::io::detail;

struct Environment
{
  std::string name, value;
  bool present = false;
  Environment(const char* key, const char* selected) : name(key)
  {
    if (const char* old = std::getenv(key)) { present = true; value = old; }
    setenv(key, selected, 1);
  }
  ~Environment()
  {
    if (present) { setenv(name.c_str(), value.c_str(), 1); }
    else { unsetenv(name.c_str()); }
  }
};

void Same(mfem::real_t actual, mfem::real_t expected)
{
  MFEM_VERIFY(std::isfinite(actual) && std::isfinite(expected) &&
              std::memcmp(&actual, &expected, sizeof(actual)) == 0,
              "Cached VTK evaluation differs from the native GridFunction writer oracle.");
}

mfem::Mesh Mesh(int geometry_order, int dimension, int elements = 2)
{
  auto mesh = dimension == 2
    ? mfem::Mesh::MakeCartesian2D(elements, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0)
    : mfem::Mesh::MakeCartesian3D(elements, 1, 1, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0);
  mesh.SetCurvature(geometry_order, false, dimension, mfem::Ordering::byVDIM);
  if (geometry_order == 2)
  {
    mesh.Transform([](const mfem::Vector& x, mfem::Vector& y)
    {
      y = x;
      y[0] += .04*x[0]*x[1]; y[1] += .03*x[0]*x[0];
      if (x.Size() == 3) { y[2] += .02*x[0]*x[2]; }
    });
  }
  return mesh;
}

void Fill(mfem::ParGridFunction& field, int frame)
{
  const auto& fes = *field.ParFESpace();
  auto* values = field.HostWrite();
  for (int c = 0; c < fes.GetVDim(); ++c)
    for (int i = 0; i < fes.GetNDofs(); ++i)
    { values[fes.DofToVDof(i, c)] = .17*(c+1)+.07*frame+std::sin(.13*i+.11*c+.03*frame); }
}

void DeviceUpdate(mfem::ParGridFunction& field)
{
  field.UseDevice(true);
  field *= 1.0001; // The compiled MFEM operation executes on the selected backend.
  if (mfem::Device::Allows(mfem::Backend::CUDA))
  { MFEM_VERIFY(!field.GetMemory().HostIsValid(), "Expected device-current VTK fixture input."); }
}

void CheckInterpolation(int geometry_order, int order, int dimension,
                         mfem::Ordering::Type ordering, bool continuous,
                         int basis = mfem::BasisType::GaussLegendre)
{
  auto serial = Mesh(geometry_order, dimension);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  std::unique_ptr<mfem::FiniteElementCollection> fec;
  if (continuous) { fec = std::make_unique<mfem::H1_FECollection>(order, dimension); }
  else { fec = std::make_unique<mfem::L2_FECollection>(order, dimension, basis); }
  for (int components : {1, dimension})
  {
    mfem::ParFiniteElementSpace fes(&mesh, fec.get(), components, ordering);
    mfem::ParGridFunction field(&fes);
    cache::VTKReferenceShapes shapes;
    cache::VTKFieldEvaluation evaluation(field, shapes);
    cache::CachedVTKScalar scalar(evaluation);
    cache::CachedVTKVector vector(components, evaluation);
    MFEM_VERIFY(cache::CanCacheVTKField(field, &mesh), "Expected scalar VALUE finite-element cache admission.");
    for (int frame = 0; frame < 3; ++frame)
    {
      Fill(field, frame);
      DeviceUpdate(field);
      evaluation.Reset(); shapes.Reset();
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
        const auto& points = mfem::GlobGeometryRefiner.Refine(
          mesh.GetElementBaseGeometry(e), std::max(1, order), 1)->RefPts;
        auto& transformation = *mesh.GetElementTransformation(e);
        mfem::Vector expected, actual;
        mfem::DenseMatrix vector_expected, positions;
        // Cache evaluation precedes the oracle's host read, including on CUDA.
        if (components == 1) { (void)scalar.Eval(transformation, points.IntPoint(0)); }
        else { vector.Eval(actual, transformation, points.IntPoint(0)); }
        if (components == 1) { field.GetValues(e, points, expected, positions); }
        else { field.GetVectorValues(e, points, vector_expected, positions); }
        for (int q = 0; q < points.GetNPoints(); ++q)
        {
          if (components == 1) { Same(scalar.Eval(transformation, points.IntPoint(q)), expected[q]); }
          else
          {
            vector.Eval(actual, transformation, points.IntPoint(q));
            for (int c = 0; c < components; ++c) { Same(actual[c], vector_expected(c, q)); }
          }
        }
        // Same IP ordinal, different actual coordinates: an index-only key is invalid.
        if (components == 1)
        {
          for (double x : {.23, .67})
          {
            mfem::IntegrationPoint point;
            point.Set3(x, .31, dimension == 3 ? .2+x : 0.0);
            point.index = 0;
            transformation.SetIntPoint(&point);
            Same(scalar.Eval(transformation, point), field.GetValue(transformation, point));
          }
        }
      }
    }
  }
}

std::map<std::string, std::vector<double>> Arrays(const std::filesystem::path& path)
{
  std::ifstream input(path);
  MFEM_VERIFY(input, "Missing native VTU from cache writer fixture.");
  const std::string text((std::istreambuf_iterator<char>(input)), {});
  std::map<std::string, std::vector<double>> arrays;
  std::size_t first = 0;
  while ((first = text.find("<DataArray", first)) != std::string::npos)
  {
    const auto begin = text.find('>', first), end = text.find("</DataArray>", begin);
    MFEM_VERIFY(begin != std::string::npos && end != std::string::npos, "Incomplete VTU array.");
    const std::string tag = text.substr(first, begin-first);
    const auto name_at = tag.find("Name=\"");
    if (name_at != std::string::npos)
    {
      const auto start = name_at+6, stop = tag.find('"', start);
      std::istringstream values(text.substr(begin+1, end-begin-1));
      double value;
      auto& array = arrays[tag.substr(start, stop-start)];
      while (values >> value) { array.push_back(value); }
    }
    first = end+12;
  }
  return arrays;
}

void CheckWriter(const std::filesystem::path& prefix, bool direct_pressure,
                  mfem::Ordering::Type ordering,
                  int basis = mfem::BasisType::GaussLegendre)
{
  Environment pressure_policy("IGRLAB_VTK_PRESSURE_COEFFICIENT", direct_pressure ? "1" : "0");
  auto serial = Mesh(2, 2, 1); // The same element number is reused across saves.
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  mfem::L2_FECollection fec(5, 2, basis), zero(0, 2);
  mfem::ParFiniteElementSpace scalar_fes(&mesh, &fec), vector_fes(&mesh, &fec, 2, ordering),
    conservative_fes(&mesh, &fec, 4, ordering), zero_fes(&mesh, &zero);
  mfem::ParGridFunction density(&scalar_fes), momentum(&vector_fes), energy(&scalar_fes),
    velocity(&vector_fes), sigma(&scalar_fes), pressure(&scalar_fes), conservative(&conservative_fes),
    replacement(&conservative_fes),
    rank(&zero_fes), active(&zero_fes);
  rank = 0.0; active = 1.0;
  mfem::PressureCoefficient coefficient(conservative, 2, 1.4);
  mfem::io::EulerVTKFields fields;
  fields.density = &density; fields.momentum = &momentum; fields.energy = &energy;
  fields.velocity = &velocity; fields.sigma = &sigma; fields.pressure = &pressure;
  fields.pressure_coefficient = &coefficient; fields.mpi_rank = &rank; fields.zhang_shu_active = &active;
  mfem::ParaViewDataCollection native("native", &mesh);
  Environment cached_policy("IGRLAB_VTK_CACHED_EVALUATION", "1");
  auto cached = mfem::io::MakeEulerVTKDataCollection("cached", &mesh);
  for (auto* collection : {&native, cached.get()})
  {
    mfem::io::RegisterEulerFields(*collection, fields);
    collection->SetPrefixPath(prefix.string()); collection->SetDataFormat(mfem::VTKFormat::ASCII);
    collection->SetPrecision(17); collection->SetLevelsOfDetail(5); collection->SetHighOrderOutput(true);
  }
  MFEM_VERIFY(native.HasField("density") && !cached->HasField("density") &&
              cached->GetVCoeffFieldMap().count("momentum") == 1,
              "Only the explicit Euler factory may replace native GridFunction registration.");
  for (int frame = 0; frame < 3; ++frame)
  {
    for (auto* field : {&density, &momentum, &energy, &velocity, &sigma, &pressure})
    { Fill(*field, frame); DeviceUpdate(*field); }
    auto* state = conservative.HostWrite();
    for (int i = 0; i < conservative_fes.GetNDofs(); ++i)
    {
      state[conservative_fes.DofToVDof(i, 0)] = frame == 2 ? 1e-18 : 1.0+.002*i+.01*frame;
      state[conservative_fes.DofToVDof(i, 1)] = .02+.001*i;
      state[conservative_fes.DofToVDof(i, 2)] = -.01+.0002*i;
      state[conservative_fes.DofToVDof(i, 3)] = 3.0+.003*i+.1*frame;
    }
    if (frame == 1)
    { replacement = conservative; replacement *= 1.07; coefficient.SetState(replacement); }
    else { coefficient.SetState(conservative); }
    DeviceUpdate(frame == 1 ? replacement : conservative);
    coefficient.SetDimGamma(2, 1.4+.02*frame);
    for (auto* collection : {cached.get(), &native})
    { collection->SetCycle(frame); collection->SetTime(.1*frame); collection->Save(); }
    std::ostringstream cycle; cycle << "Cycle" << std::setfill('0') << std::setw(6) << frame;
    const auto expected = Arrays(prefix/"native"/cycle.str()/"proc000000.vtu");
    const auto actual = Arrays(prefix/"cached"/cycle.str()/"proc000000.vtu");
    MFEM_VERIFY(expected == actual && actual.count("sigma") && actual.count("pressure") &&
                actual.count("mpi_rank") && actual.count("zhang_shu_active") && !actual.count("div_h1"),
                "Cached native writer changed field values, component order, optional fields, or frame freshness.");
  }
  // Unknown coefficient policies retain their own native Eval implementation.
  mfem::ConstantCoefficient unknown(7.25);
  fields.pressure_coefficient = &unknown;
  if (direct_pressure)
  {
    mfem::io::RegisterEulerFields(*cached, fields);
    MFEM_VERIFY(cached->GetCoeffFieldMap().at("pressure") == &unknown,
                "Unknown pressure coefficient was replaced by a different policy.");
  }
  mfem::ND_FECollection nd_fec(1, 2);
  mfem::ParFiniteElementSpace nd_fes(&mesh, &nd_fec);
  mfem::ParGridFunction nd_field(&nd_fes);
  cached->RegisterField("unsupported", &nd_field);
  MFEM_VERIFY(cached->HasField("unsupported"), "Vector finite elements must retain the native writer path.");
  cached->DeregisterField("unsupported");
  mfem::L2_FECollection integral_fec(2, 2, mfem::BasisType::GaussLegendre,
                                    mfem::FiniteElement::INTEGRAL);
  mfem::ParFiniteElementSpace integral_fes(&mesh, &integral_fec);
  mfem::ParGridFunction integral_field(&integral_fes);
  cached->RegisterField("integral", &integral_field);
  MFEM_VERIFY(cached->HasField("integral"), "Integral-map fields must retain the native writer path.");
  cached->DeregisterField("integral");
  cached.reset(); // Borrowed mesh, GFs and original coefficients remain alive.
  density = 2.0;
  MFEM_VERIFY(density.Norml2() > 0, "Writer destruction deleted caller-owned data.");
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv); mfem::Hypre::Init();
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  MFEM_VERIFY(backend == "cpu" || backend == "cuda", "Expected cpu or cuda.");
  mfem::Device device(backend.c_str());
  for (int geometry : {1, 2})
    for (auto ordering : {mfem::Ordering::byNODES, mfem::Ordering::byVDIM})
      for (int order : {0, 1, 3, 5})
      {
        CheckInterpolation(geometry, order, 2, ordering, false);
        if (order)
        {
          CheckInterpolation(geometry, order, 2, ordering, true);
          CheckInterpolation(geometry, order, 2, ordering, false, mfem::BasisType::GaussLobatto);
        }
      }
  CheckInterpolation(2, 2, 3, mfem::Ordering::byVDIM, false);
  const auto token = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::path("/tmp")/ ("igrlab-vtk-cache-"+std::to_string(token));
  MFEM_VERIFY(std::filesystem::create_directory(root), "Use a new private temporary writer-test directory.");
  for (bool direct : {false, true})
    for (auto ordering : {mfem::Ordering::byNODES, mfem::Ordering::byVDIM})
    {
      const auto output = root/(std::string(direct ? "direct-" : "projected-")+std::to_string(int(ordering)));
      CheckWriter(output, direct, ordering);
      CheckWriter(output.string()+"-gll", direct, ordering, mfem::BasisType::GaussLobatto);
    }
  std::filesystem::remove_all(root);
  std::cout << "VTK cached interpolation: native scalar/vector/pressure oracles and repeated Save passed.\n";
  return 0;
}
