#include "numerics/hdg/euler/smooth_euler_device_full.hpp"

#include <mfem.hpp>
#include <umpire/Umpire.hpp>
#include <umpire/json/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace
{
using dg::hdg::euler::FullDeviceEvidence;
using dg::hdg::euler::FullDeviceMeshEvidence;
using dg::hdg::euler::FullDevicePhase;
using dg::hdg::euler::FullDeviceRequest;
using dg::hdg::euler::FullDeviceStatus;
using dg::hdg::euler::FullDeviceTestSnapshot;
using dg::hdg::euler::conserved_components;
using dg::hdg::euler::backward_euler_dt;
using dg::hdg::euler::dimension;
using dg::hdg::euler::experiment_epsilon;
using dg::hdg::euler::fixed_scalar_tau;
using dg::hdg::euler::gamma;

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr const char* umpire_terminal_marker =
    "DG_IGR_UMPIRE_AUDIT_TERMINAL";

using Json = nlohmann::json;

std::string ShellQuote(const std::string& value)
{
  std::string result = "'";
  for (const char character : value)
  {
    if (character == '\'') { result += "'\\''"; }
    else { result += character; }
  }
  result += '\'';
  return result;
}

struct AllocationIntervalAudit
{
  std::size_t logical_allocate_events = 0;
  std::size_t logical_deallocate_events = 0;
};

struct UmpireLogAudit
{
  std::size_t zero_intervals = 0;
  std::size_t churn_intervals = 0;
  std::size_t churn_logical_allocate_events = 0;
  std::size_t churn_logical_deallocate_events = 0;
  std::size_t mapped_allocators = 0;
};

bool JsonString(const Json& event, const char* section, const char* key,
                std::string& result)
{
  if (!event.is_object() || !event.contains(section) ||
      !event[section].is_object() || !event[section].contains(key))
  {
    return false;
  }
  const Json& value = event[section][key];
  if (value.is_string())
  {
    result = value.get<std::string>();
    return true;
  }
  if (value.is_number_unsigned())
  {
    result = std::to_string(value.get<std::uint64_t>());
    return true;
  }
  if (value.is_number_integer())
  {
    result = std::to_string(value.get<std::int64_t>());
    return true;
  }
  return false;
}

bool AllocatorRef(const Json& event, std::string& result)
{
  // Umpire releases encode pointer-shaped refs in string_args; accept the
  // numeric_args representation as well, but never search unrelated fields.
  return JsonString(event, "numeric_args", "allocator_ref", result) ||
         JsonString(event, "string_args", "allocator_ref", result);
}

bool EventName(const Json& event, std::string& result)
{
  if (!event.is_object() || !event.contains("name") ||
      !event["name"].is_string())
  {
    return false;
  }
  result = event["name"].get<std::string>();
  return true;
}

enum class IntervalKind : unsigned char { none, zero, churn };

bool MarkerIdentity(const std::string& marker, const std::string& prefix,
                    std::string& identity)
{
  if (marker.rfind(prefix, 0) != 0) { return false; }
  identity = marker.substr(prefix.size());
  return !identity.empty();
}

bool AuditUmpireLog(const std::filesystem::path& path,
                    UmpireLogAudit& audit)
{
  audit = {};
  std::ifstream binary(path, std::ios::binary);
  if (!binary) { return false; }
  binary.seekg(0, std::ios::end);
  const std::streamoff bytes = binary.tellg();
  if (bytes <= 0) { return false; }
  binary.seekg(-1, std::ios::end);
  char final_character = '\0';
  binary.get(final_character);
  if (final_character != '\n') { return false; }
  binary.close();

  std::ifstream input(path);
  if (!input) { return false; }
  std::unordered_map<std::string, std::string> allocator_names;
  std::unordered_map<std::string, AllocationIntervalAudit> zero_intervals;
  std::unordered_map<std::string, AllocationIntervalAudit> churn_intervals;
  std::unordered_set<std::string> completed;
  std::unordered_set<std::string> identities;
  std::string active;
  IntervalKind active_kind = IntervalKind::none;
  std::size_t terminal_markers = 0;
  bool terminal_seen = false;
  bool mfem_device_pool_mapped = false;
  bool hypre_device_pool_mapped = false;
  std::string line;
  while (std::getline(input, line))
  {
    if (line.empty()) { continue; }
    Json event = Json::parse(line, nullptr, false);
    if (event.is_discarded()) { return false; }
    std::string event_name;
    if (!EventName(event, event_name)) { return false; }
    if (event_name == "make_allocator")
    {
      std::string allocator_ref;
      std::string allocator_name;
      if (!AllocatorRef(event, allocator_ref) ||
          !JsonString(event, "tags", "allocator_name", allocator_name) ||
          allocator_ref.empty())
      {
        return false;
      }
      allocator_names[allocator_ref] = allocator_name;
      mfem_device_pool_mapped = mfem_device_pool_mapped ||
          allocator_name == "MFEM_DEVICE";
      hypre_device_pool_mapped = hypre_device_pool_mapped ||
          allocator_name == "HYPRE_DEVICE_POOL";
    }
    if (event_name == "event")
    {
      std::string marker;
      if (!JsonString(event, "string_args", "name", marker)) { return false; }
      if (terminal_seen) { return false; }
      if (marker == umpire_terminal_marker)
      {
        if (!active.empty() || active_kind != IntervalKind::none)
          return false;
        ++terminal_markers;
        terminal_seen = true;
        continue;
      }
      std::string identity;
      if (MarkerIdentity(marker, "DG_IGR_ZERO_ALLOC_BEGIN_", identity) ||
          MarkerIdentity(marker, "DG_IGR_ALLOC_CHURN_BEGIN_", identity))
      {
        const bool zero =
            marker.rfind("DG_IGR_ZERO_ALLOC_BEGIN_", 0) == 0;
        if (!active.empty() || active_kind != IntervalKind::none ||
            !identities.insert(identity).second)
        {
          return false;
        }
        active = identity;
        active_kind = zero ? IntervalKind::zero : IntervalKind::churn;
        if (zero) { zero_intervals.emplace(active, AllocationIntervalAudit{}); }
        else { churn_intervals.emplace(active, AllocationIntervalAudit{}); }
        continue;
      }
      if (MarkerIdentity(marker, "DG_IGR_ZERO_ALLOC_END_", identity) ||
          MarkerIdentity(marker, "DG_IGR_ALLOC_CHURN_END_", identity))
      {
        const bool zero = marker.rfind("DG_IGR_ZERO_ALLOC_END_", 0) == 0;
        const IntervalKind expected =
            zero ? IntervalKind::zero : IntervalKind::churn;
        if (active != identity || active_kind != expected ||
            !completed.insert(active).second)
        {
          return false;
        }
        active.clear();
        active_kind = IntervalKind::none;
        continue;
      }
      return false;
    }

    const bool allocate = event_name == "allocate" ||
        event_name == "named_allocate" ||
        event_name == "allocate_resource";
    const bool deallocate = event_name == "deallocate" ||
        event_name == "deallocate_resource";
    if (!allocate && !deallocate) { continue; }
    if (active_kind == IntervalKind::none) { continue; }
    std::string allocator_ref;
    if (!AllocatorRef(event, allocator_ref) ||
        allocator_names.count(allocator_ref) == 0)
    {
      return false;
    }
    AllocationIntervalAudit* interval = active_kind == IntervalKind::zero ?
        &zero_intervals.at(active) : &churn_intervals.at(active);
    if (allocate) { ++interval->logical_allocate_events; }
    if (deallocate) { ++interval->logical_deallocate_events; }
  }
  if (!input.eof() || !active.empty() ||
      active_kind != IntervalKind::none || terminal_markers != 1 ||
      !terminal_seen || !mfem_device_pool_mapped ||
      !hypre_device_pool_mapped || allocator_names.empty() ||
      zero_intervals.empty() || churn_intervals.empty() ||
      completed.size() != zero_intervals.size() + churn_intervals.size())
  {
    return false;
  }
  for (const auto& item : zero_intervals)
  {
    if (item.second.logical_allocate_events != 0 ||
        item.second.logical_deallocate_events != 0)
    {
      return false;
    }
  }
  for (const auto& item : churn_intervals)
  {
    audit.churn_logical_allocate_events +=
        item.second.logical_allocate_events;
    audit.churn_logical_deallocate_events +=
        item.second.logical_deallocate_events;
  }
  audit.zero_intervals = zero_intervals.size();
  audit.churn_intervals = churn_intervals.size();
  audit.mapped_allocators = allocator_names.size();
  return audit.churn_logical_allocate_events +
             audit.churn_logical_deallocate_events >
         0;
}

int RunUmpireAuditDriver(const char* executable)
{
  namespace fs = std::filesystem;
  const fs::path directory_template = fs::temp_directory_path() /
      "dg_igr_hdg_euler_full_umpire_XXXXXX";
  const std::string directory_template_string = directory_template.string();
  std::vector<char> writable_template(directory_template_string.begin(),
                                      directory_template_string.end());
  writable_template.push_back('\0');
  char* created = ::mkdtemp(writable_template.data());
  if (created == nullptr) { return 70; }
  const fs::path output_directory(created);
  const std::string basename = "hdg_euler_full";
  const std::string command =
      "UMPIRE_EVENTS=On UMPIRE_OUTPUT_DIR=" +
      ShellQuote(output_directory.string()) + " UMPIRE_OUTPUT_BASENAME=" +
      ShellQuote(basename) + " " + ShellQuote(executable) +
      " --umpire-audit-child";
  const int child_status = std::system(command.c_str());
  if (child_status != 0)
  {
    std::cerr << "HDG_EULER_DEVICE_FULL_UMPIRE_AUDIT_FAILURE "
              << "preserved_log_dir=" << output_directory << '\n';
    return 71;
  }
  std::vector<fs::path> logs;
  for (const auto& entry : fs::directory_iterator(output_directory))
  {
    const std::string filename = entry.path().filename().string();
    if (entry.is_regular_file() &&
        filename.rfind(basename + ".", 0) == 0 &&
        entry.path().extension() == ".stats")
    {
      logs.push_back(entry.path());
    }
  }
  if (logs.size() != 1)
  {
    std::cerr << "HDG_EULER_DEVICE_FULL_UMPIRE_AUDIT_FAILURE "
              << "log_count=" << logs.size()
              << " preserved_log_dir=" << output_directory << '\n';
    for (const auto& log : logs)
      std::cerr << "preserved_log=" << log << '\n';
    return 72;
  }
  UmpireLogAudit audit;
  if (!AuditUmpireLog(logs.front(), audit))
  {
    std::cerr << "HDG_EULER_DEVICE_FULL_UMPIRE_AUDIT_FAILURE "
              << "preserved_log=" << logs.front() << '\n';
    return 72;
  }
  std::cout << "HDG_EULER_DEVICE_FULL_UMPIRE_AUDIT_SUCCESS log="
            << logs.front() << " allocator_scope=all_mapped"
            << " mapped_allocators=" << audit.mapped_allocators
            << " zero_intervals=" << audit.zero_intervals
            << " churn_intervals=" << audit.churn_intervals
            << " churn_logical_allocate_events="
            << audit.churn_logical_allocate_events
            << " churn_logical_deallocate_events="
            << audit.churn_logical_deallocate_events << '\n';
  return 0;
}

int Decode(const int encoded, double& sign)
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
}

int ByNodes(const int scalar, const int scalar_dofs, const int component)
{
  return scalar + scalar_dofs * component;
}

bool BitwiseEqual(const std::vector<double>& first,
                  const std::vector<double>& second)
{
  return first.size() == second.size() &&
         (first.empty() ||
          std::memcmp(first.data(), second.data(),
                      first.size() * sizeof(double)) == 0);
}

bool DerivedSnapshotEmpty(const FullDeviceTestSnapshot& snapshot)
{
  return snapshot.polynomial_order == 0 &&
         snapshot.elements_per_direction == 0 && snapshot.elements == 0 &&
         snapshot.interior_dofs == 0 && snapshot.trace_dofs == 0 &&
         snapshot.global_trace_dofs == 0 &&
         snapshot.assembly_quadrature_order == 0 &&
         snapshot.signed_trace_dofs.empty() && snapshot.old_interior.empty() &&
         snapshot.initial_interior.empty() && snapshot.initial_trace.empty() &&
         snapshot.final_interior.empty() && snapshot.final_trace.empty() &&
         snapshot.first_a.empty() && snapshot.first_b.empty() &&
         snapshot.first_c.empty() && snapshot.first_d.empty() &&
         snapshot.first_f.empty() && snapshot.first_g.empty() &&
         snapshot.first_schur.empty() &&
         snapshot.first_condensed_rhs.empty() &&
         snapshot.first_local_trace_increment.empty() &&
         snapshot.first_recovered_increment.empty() &&
         snapshot.global_a.empty() && snapshot.global_b.empty() &&
         snapshot.global_c.empty() && snapshot.global_d.empty() &&
         snapshot.global_f.empty() && snapshot.global_g.empty() &&
         snapshot.global_schur.empty() &&
         snapshot.global_condensed_rhs.empty() &&
         snapshot.global_trace_increment.empty() &&
         snapshot.global_local_trace_increment.empty() &&
         snapshot.global_recovered_increment.empty() &&
         snapshot.actual_true_csr_offsets.empty() &&
         snapshot.actual_true_csr_columns.empty() &&
         snapshot.actual_true_csr_values.empty() &&
         snapshot.actual_true_rhs.empty() &&
         snapshot.final_residual_r.empty() &&
         snapshot.final_residual_g.empty() &&
         snapshot.last_base_interior.empty() &&
         snapshot.last_base_trace.empty() &&
         snapshot.last_delta_interior.empty() &&
         snapshot.last_delta_trace.empty() &&
         snapshot.last_accepted_alpha == 0.0 &&
         !snapshot.global_linear_oracle_staged;
}

void ExactState(const double x, const double y,
                std::array<double, conserved_components>& state)
{
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  const double rho = 1.0 + 0.1 * sx * sy;
  const double u = 0.35 + 0.05 * cx * sy;
  const double v = -0.20 + 0.05 * sx * cy;
  const double pressure = 1.0 + 0.1 * cx * cy;
  state = {{rho, rho * u, rho * v,
            pressure / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)}};
}

bool Flux(const std::array<double, conserved_components>& state,
          double nx, double ny,
          std::array<double, conserved_components>& flux);

bool FluxJacobian(
    const std::array<double, conserved_components>& state, const double nx,
    const double ny, std::array<double, conserved_components>& flux,
    std::array<double, conserved_components * conserved_components>& jacobian)
{
  if (!Flux(state, nx, ny, flux)) { return false; }
  const double rho = state[0];
  const double mx = state[1];
  const double my = state[2];
  const double energy = state[3];
  const double q = mx * nx + my * ny;
  const double kinetic = 0.5 * (mx * mx + my * my) / rho;
  const double pressure = (gamma - 1.0) * (energy - kinetic);
  const double pressure_rho =
      (gamma - 1.0) * 0.5 * (mx * mx + my * my) / (rho * rho);
  const double pressure_mx = -(gamma - 1.0) * mx / rho;
  const double pressure_my = -(gamma - 1.0) * my / rho;
  const double pressure_energy = gamma - 1.0;
  const double enthalpy = energy + pressure;
  jacobian.fill(0.0);
  jacobian[1] = nx;
  jacobian[2] = ny;
  jacobian[4] = -mx * q / (rho * rho) + pressure_rho * nx;
  jacobian[5] = q / rho + mx * nx / rho + pressure_mx * nx;
  jacobian[6] = mx * ny / rho + pressure_my * nx;
  jacobian[7] = pressure_energy * nx;
  jacobian[8] = -my * q / (rho * rho) + pressure_rho * ny;
  jacobian[9] = my * nx / rho + pressure_mx * ny;
  jacobian[10] = q / rho + my * ny / rho + pressure_my * ny;
  jacobian[11] = pressure_energy * ny;
  jacobian[12] = pressure_rho * q / rho - enthalpy * q / (rho * rho);
  jacobian[13] = pressure_mx * q / rho + enthalpy * nx / rho;
  jacobian[14] = pressure_my * q / rho + enthalpy * ny / rho;
  jacobian[15] = (1.0 + pressure_energy) * q / rho;
  return std::all_of(jacobian.begin(), jacobian.end(),
                     [](const double value) { return std::isfinite(value); });
}

bool ManufacturedSourceIndependent(
    const double x, const double y,
    std::array<double, conserved_components>& source)
{
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  std::array<double, conserved_components> state{};
  ExactState(x, y, state);
  const double rho = state[0];
  const double u = state[1] / rho;
  const double v = state[2] / rho;
  const double rho_x = 0.2 * pi * cx * sy;
  const double rho_y = 0.2 * pi * sx * cy;
  const double u_x = -0.1 * pi * sx * sy;
  const double u_y = 0.1 * pi * cx * cy;
  const double v_x = 0.1 * pi * cx * cy;
  const double v_y = -0.1 * pi * sx * sy;
  const double pressure_x = -0.2 * pi * sx * cy;
  const double pressure_y = -0.2 * pi * cx * sy;
  const std::array<double, conserved_components> derivative_x{{
      rho_x, rho_x * u + rho * u_x, rho_x * v + rho * v_x,
      pressure_x / (gamma - 1.0) +
          0.5 * rho_x * (u * u + v * v) + rho * (u * u_x + v * v_x)}};
  const std::array<double, conserved_components> derivative_y{{
      rho_y, rho_y * u + rho * u_y, rho_y * v + rho * v_y,
      pressure_y / (gamma - 1.0) +
          0.5 * rho_y * (u * u + v * v) + rho * (u * u_y + v * v_y)}};
  std::array<double, conserved_components> flux{};
  std::array<double, conserved_components * conserved_components> jacobian_x{};
  std::array<double, conserved_components * conserved_components> jacobian_y{};
  if (!FluxJacobian(state, 1.0, 0.0, flux, jacobian_x) ||
      !FluxJacobian(state, 0.0, 1.0, flux, jacobian_y))
  {
    return false;
  }
  source.fill(0.0);
  for (int row = 0; row < conserved_components; ++row)
    for (int column = 0; column < conserved_components; ++column)
      source[row] += jacobian_x[row * conserved_components + column] *
                         derivative_x[column] +
                     jacobian_y[row * conserved_components + column] *
                         derivative_y[column];
  return std::all_of(source.begin(), source.end(),
                     [](const double value) { return std::isfinite(value); });
}

struct ErrorIntegral
{
  long double error = 0.0L;
  long double norm = 0.0L;
  long double measure = 0.0L;
  int faces = 0;

  double Relative() const
  {
    return std::sqrt(static_cast<double>(error / norm));
  }
};

struct OracleMesh
{
  std::unique_ptr<mfem::Mesh> serial;
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> interior_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
};

bool BuildOracleMesh(const int p, const int N, OracleMesh& result)
{
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        N, N, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    result.serial = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations)));
    result.mesh =
        std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *result.serial);
    result.interior_collection = std::make_unique<mfem::L2_FECollection>(
        p, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    result.trace_collection =
        std::make_unique<mfem::DG_Interface_FECollection>(
            p, dimension, mfem::FiniteElement::VALUE,
            mfem::BasisType::GaussLegendre);
    result.interior_space = std::make_unique<mfem::ParFiniteElementSpace>(
        result.mesh.get(), result.interior_collection.get(),
        conserved_components, mfem::Ordering::byNODES);
    result.trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
        result.mesh.get(), result.trace_collection.get(), conserved_components,
        mfem::Ordering::byNODES);
  }
  catch (const std::exception&)
  {
    return false;
  }
  return result.mesh->GetNE() == N * N &&
         result.interior_space->GetOrdering() == mfem::Ordering::byNODES &&
         result.trace_space->GetOrdering() == mfem::Ordering::byNODES;
}

bool Interpolate(const std::vector<double>& coefficients, const int offset,
                 const int scalar_dofs, const mfem::Vector& shape,
                 std::array<double, conserved_components>& state)
{
  state = {{0.0, 0.0, 0.0, 0.0}};
  for (int scalar = 0; scalar < scalar_dofs; ++scalar)
    for (int component = 0; component < conserved_components; ++component)
    {
      const int entry = offset + ByNodes(scalar, scalar_dofs, component);
      if (entry < 0 || entry >= static_cast<int>(coefficients.size()))
      {
        return false;
      }
      state[component] += shape[scalar] * coefficients[entry];
    }
  return std::all_of(state.begin(), state.end(),
                     [](const double value) { return std::isfinite(value); });
}

bool IndependentErrors(const FullDeviceTestSnapshot& snapshot,
                       ErrorIntegral& volume, ErrorIntegral& trace,
                       bool& signed_map_exact)
{
  OracleMesh oracle;
  if (!BuildOracleMesh(snapshot.polynomial_order,
                       snapshot.elements_per_direction, oracle))
  {
    return false;
  }
  const int qO = 5 * snapshot.polynomial_order + 16;
  const int elements = oracle.mesh->GetNE();
  for (int element = 0; element < elements; ++element)
  {
    const mfem::FiniteElement* fe = oracle.interior_space->GetFE(element);
    mfem::ElementTransformation* transformation =
        oracle.mesh->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr) { return false; }
    const int scalar_dofs = fe->GetDof();
    if (snapshot.interior_dofs != conserved_components * scalar_dofs)
    {
      return false;
    }
    mfem::Vector shape(scalar_dofs), physical(dimension);
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), qO);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      std::array<double, conserved_components> numerical{}, exact{};
      if (!Interpolate(snapshot.final_interior,
                       element * snapshot.interior_dofs, scalar_dofs, shape,
                       numerical))
      {
        return false;
      }
      ExactState(physical[0], physical[1], exact);
      const double weight = point.weight * transformation->Weight();
      if (!(weight > 0.0) || !std::isfinite(weight)) { return false; }
      volume.measure += weight;
      for (int component = 0; component < conserved_components; ++component)
      {
        const long double difference = numerical[component] - exact[component];
        volume.error += weight * difference * difference;
        volume.norm += weight * static_cast<long double>(exact[component]) *
                       exact[component];
      }
    }
  }

  signed_map_exact = true;
  std::size_t element_map_offset = 0;
  for (int element = 0; element < elements; ++element)
  {
    mfem::Array<int> faces, orientations;
    oracle.mesh->GetElementEdges(element, faces, orientations);
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      mfem::Array<int> vdofs;
      oracle.trace_space->GetFaceVDofs(faces[local_face], vdofs);
      for (int local = 0; local < vdofs.Size(); ++local)
      {
        if (element_map_offset >= snapshot.signed_trace_dofs.size() ||
            snapshot.signed_trace_dofs[element_map_offset++] != vdofs[local])
        {
          signed_map_exact = false;
        }
      }
    }
  }
  signed_map_exact = signed_map_exact &&
                     element_map_offset == snapshot.signed_trace_dofs.size();

  for (int face = 0; face < oracle.mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* fe = oracle.trace_space->GetFaceElement(face);
    mfem::ElementTransformation* transformation =
        oracle.mesh->GetFaceTransformation(face);
    mfem::Array<int> vdofs;
    oracle.trace_space->GetFaceVDofs(face, vdofs);
    if (fe == nullptr || transformation == nullptr ||
        vdofs.Size() != conserved_components * fe->GetDof())
    {
      return false;
    }
    std::vector<double> local(vdofs.Size());
    for (int entry = 0; entry < vdofs.Size(); ++entry)
    {
      double sign = 1.0;
      const int global = Decode(vdofs[entry], sign);
      if (global < 0 || global >= static_cast<int>(snapshot.final_trace.size()))
      {
        return false;
      }
      local[entry] = sign * snapshot.final_trace[global];
    }
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), qO);
    mfem::Vector shape(fe->GetDof()), physical(dimension);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      std::array<double, conserved_components> numerical{}, exact{};
      if (!Interpolate(local, 0, fe->GetDof(), shape, numerical))
      {
        return false;
      }
      ExactState(physical[0], physical[1], exact);
      const double weight = point.weight * transformation->Weight();
      if (!(weight > 0.0) || !std::isfinite(weight)) { return false; }
      trace.measure += weight;
      for (int component = 0; component < conserved_components; ++component)
      {
        const long double difference = numerical[component] - exact[component];
        trace.error += weight * difference * difference;
        trace.norm += weight * static_cast<long double>(exact[component]) *
                      exact[component];
      }
    }
    ++trace.faces;
  }
  volume.faces = 0;
  return volume.error > 0.0L && volume.norm > 0.0L &&
         trace.error > 0.0L && trace.norm > 0.0L;
}

bool IndependentOrientationProbe(const int p, const int N)
{
  OracleMesh oracle;
  if (!BuildOracleMesh(p, N, oracle)) { return false; }
  std::vector<std::array<int, 2>> face_orientations(
      oracle.mesh->GetNumFaces(), std::array<int, 2>{{-1, -1}});
  std::vector<int> face_counts(oracle.mesh->GetNumFaces(), 0);
  int negative_vdofs = 0;
  for (int element = 0; element < oracle.mesh->GetNE(); ++element)
  {
    mfem::Array<int> faces, orientations;
    oracle.mesh->GetElementEdges(element, faces, orientations);
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      if (face < 0 || face >= static_cast<int>(face_counts.size()) ||
          face_counts[face] >= 2)
      {
        return false;
      }
      face_orientations[face][face_counts[face]++] =
          orientations[local_face];
      mfem::Array<int> vdofs;
      oracle.trace_space->GetFaceVDofs(face, vdofs);
      for (int local = 0; local < vdofs.Size(); ++local)
        negative_vdofs += vdofs[local] < 0 ? 1 : 0;
    }
  }
  int reversed_face = -1;
  for (int face = 0; face < static_cast<int>(face_counts.size()); ++face)
  {
    if (face_counts[face] == 2 &&
        face_orientations[face][0] * face_orientations[face][1] == -1)
    {
      reversed_face = face;
      break;
    }
  }
  if (negative_vdofs != 0 || reversed_face < 0) { return false; }
  const mfem::FiniteElement* trace_fe =
      oracle.trace_space->GetFaceElement(reversed_face);
  if (trace_fe == nullptr || trace_fe->GetDof() <= 1) { return false; }
  const int dofs = trace_fe->GetDof();
  const int* reversed = oracle.trace_collection->DofOrderForOrientation(
      mfem::Geometry::SEGMENT, -1);
  if (reversed == nullptr) { return false; }
  for (int scalar = 0; scalar < dofs; ++scalar)
    if (reversed[scalar] != dofs - 1 - scalar || reversed[scalar] < 0)
      return false;
  const mfem::IntegrationRule& rule =
      mfem::IntRules.Get(trace_fe->GetGeomType(), 2 * p + 10);
  mfem::Vector forward_shape(dofs), reverse_shape(dofs);
  long double forward_action = 0.0L;
  long double reverse_action = 0.0L;
  for (int q = 0; q < rule.GetNPoints(); ++q)
  {
    const auto& point = rule.IntPoint(q);
    mfem::IntegrationPoint reverse_point(point);
    reverse_point.x = 1.0 - point.x;
    trace_fe->CalcShape(point, forward_shape);
    trace_fe->CalcShape(reverse_point, reverse_shape);
    double forward_value = 0.0;
    double reverse_value = 0.0;
    for (int scalar = 0; scalar < dofs; ++scalar)
    {
      forward_value += forward_shape[scalar] * (1.0 + 0.25 * scalar);
      reverse_value += reverse_shape[scalar] *
                       (1.0 + 0.25 * reversed[scalar]);
    }
    forward_action += point.weight * forward_value * forward_value;
    reverse_action += point.weight * reverse_value * reverse_value;
  }
  return std::abs(static_cast<double>(forward_action - reverse_action)) <=
         256.0 * experiment_epsilon *
             std::max(1.0, std::abs(static_cast<double>(forward_action)));
}

bool IndependentFinalPdeResidual(const FullDeviceMeshEvidence& evidence,
                                 double& residual,
                                 double& relative_error,
                                 bool& independent_success)
{
  const auto& snapshot = evidence.test_snapshot;
  OracleMesh oracle;
  if (!BuildOracleMesh(snapshot.polynomial_order,
                       snapshot.elements_per_direction, oracle))
  {
    return false;
  }
  const int elements = oracle.mesh->GetNE();
  const int ni = snapshot.interior_dofs;
  const int nt = snapshot.trace_dofs;
  const int ng = snapshot.global_trace_dofs;
  if (elements != snapshot.elements || ni <= 0 || nt <= 0 || ng <= 0 ||
      snapshot.old_interior.size() !=
          static_cast<std::size_t>(elements) * ni ||
      snapshot.final_interior.size() !=
          static_cast<std::size_t>(elements) * ni ||
      snapshot.final_trace.size() != static_cast<std::size_t>(ng) ||
      snapshot.signed_trace_dofs.size() !=
          static_cast<std::size_t>(elements) * nt)
  {
    return false;
  }
  std::vector<double> interior_residual(
      static_cast<std::size_t>(elements) * ni, 0.0);
  std::vector<double> local_trace_residual(
      static_cast<std::size_t>(elements) * nt, 0.0);
  const int qO = 5 * snapshot.polynomial_order + 16;
  bool admissible = true;
  for (int element = 0; element < elements; ++element)
  {
    const mfem::FiniteElement* fe = oracle.interior_space->GetFE(element);
    mfem::ElementTransformation* transformation =
        oracle.mesh->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr ||
        conserved_components * fe->GetDof() != ni)
    {
      return false;
    }
    const int scalar_dofs = fe->GetDof();
    const int element_offset = element * ni;
    mfem::Vector shape(scalar_dofs), physical(dimension);
    mfem::DenseMatrix derivative(scalar_dofs, dimension);
    const mfem::IntegrationRule& volume_rule =
        mfem::IntRules.Get(fe->GetGeomType(), qO);
    for (int q = 0; q < volume_rule.GetNPoints(); ++q)
    {
      const auto& point = volume_rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      fe->CalcPhysDShape(*transformation, derivative);
      std::array<double, conserved_components> state{}, old_state{};
      if (!Interpolate(snapshot.final_interior, element_offset, scalar_dofs,
                       shape, state) ||
          !Interpolate(snapshot.old_interior, element_offset, scalar_dofs,
                       shape, old_state))
      {
        return false;
      }
      const double pressure = (gamma - 1.0) *
          (state[3] - 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                          state[0]);
      admissible = admissible && state[0] >= 1.0e-8 && pressure >= 1.0e-8;
      std::array<double, conserved_components> source{}, flux_x{}, flux_y{};
      if (!ManufacturedSourceIndependent(physical[0], physical[1], source) ||
          !Flux(state, 1.0, 0.0, flux_x) ||
          !Flux(state, 0.0, 1.0, flux_y))
      {
        return false;
      }
      const double weight = point.weight * transformation->Weight();
      if (!(weight > 0.0) || !std::isfinite(weight)) { return false; }
      for (int row = 0; row < scalar_dofs; ++row)
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = ByNodes(row, scalar_dofs, component);
          interior_residual[element_offset + local] += weight *
              (shape[row] * (state[component] - old_state[component]) /
                   backward_euler_dt -
               derivative(row, 0) * flux_x[component] -
               derivative(row, 1) * flux_y[component] -
               shape[row] * source[component]);
        }
    }

    mfem::Array<int> faces, orientations;
    oracle.mesh->GetElementEdges(element, faces, orientations);
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      const mfem::FiniteElement* trace_fe =
          oracle.trace_space->GetFaceElement(face);
      mfem::FaceElementTransformations* face_transformation =
          oracle.mesh->GetFaceElementTransformations(face);
      if (trace_fe == nullptr || face_transformation == nullptr ||
          face_transformation->Face == nullptr)
      {
        return false;
      }
      const int face_scalar_dofs = trace_fe->GetDof();
      const int face_vdofs = conserved_components * face_scalar_dofs;
      const bool first = face_transformation->Elem1No == element;
      if (!first && face_transformation->Elem2No != element) { return false; }
      mfem::Vector interior_shape(scalar_dofs);
      mfem::Vector trace_shape(face_scalar_dofs);
      const mfem::IntegrationRule& face_rule =
          mfem::IntRules.Get(trace_fe->GetGeomType(), qO);
      for (int q = 0; q < face_rule.GetNPoints(); ++q)
      {
        const auto& point = face_rule.IntPoint(q);
        face_transformation->SetAllIntPoints(&point);
        const auto& element_point = first ?
            face_transformation->GetElement1IntPoint() :
            face_transformation->GetElement2IntPoint();
        fe->CalcShape(element_point, interior_shape);
        trace_fe->CalcShape(point, trace_shape);
        std::array<double, conserved_components> state{}, trace_state{};
        if (!Interpolate(snapshot.final_interior, element_offset, scalar_dofs,
                         interior_shape, state))
        {
          return false;
        }
        for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
          for (int component = 0; component < conserved_components; ++component)
          {
            const int local = trace_offset +
                ByNodes(scalar, face_scalar_dofs, component);
            double sign = 1.0;
            const int global = Decode(
                snapshot.signed_trace_dofs[
                    static_cast<std::size_t>(element) * nt + local], sign);
            if (global < 0 || global >= ng) { return false; }
            trace_state[component] +=
                trace_shape[scalar] * sign * snapshot.final_trace[global];
          }
        const double trace_pressure = (gamma - 1.0) *
            (trace_state[3] -
             0.5 * (trace_state[1] * trace_state[1] +
                    trace_state[2] * trace_state[2]) /
                 trace_state[0]);
        admissible = admissible && trace_state[0] >= 1.0e-8 &&
                     trace_pressure >= 1.0e-8;
        mfem::Vector scaled_normal(dimension);
        mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled_normal);
        const double measure = scaled_normal.Norml2();
        if (!(measure > 0.0) || !std::isfinite(measure)) { return false; }
        const double side = first ? 1.0 : -1.0;
        std::array<double, conserved_components> trace_flux{};
        if (!Flux(trace_state, side * scaled_normal[0] / measure,
                  side * scaled_normal[1] / measure, trace_flux))
        {
          return false;
        }
        const double weight = point.weight * measure;
        for (int component = 0; component < conserved_components; ++component)
        {
          const double numerical_flux = trace_flux[component] +
              fixed_scalar_tau * (state[component] - trace_state[component]);
          for (int row = 0; row < scalar_dofs; ++row)
            interior_residual[element_offset +
                ByNodes(row, scalar_dofs, component)] +=
                weight * interior_shape[row] * numerical_flux;
          for (int row = 0; row < face_scalar_dofs; ++row)
            local_trace_residual[static_cast<std::size_t>(element) * nt +
                trace_offset + ByNodes(row, face_scalar_dofs, component)] +=
                weight * trace_shape[row] * numerical_flux;
        }
      }
      trace_offset += face_vdofs;
    }
    if (trace_offset != nt) { return false; }
  }
  std::vector<double> true_trace_residual(ng, 0.0);
  for (int entry = 0; entry < elements * nt; ++entry)
  {
    double sign = 1.0;
    const int global = Decode(snapshot.signed_trace_dofs[entry], sign);
    if (global < 0 || global >= ng) { return false; }
    true_trace_residual[global] += sign * local_trace_residual[entry];
  }
  long double norm_squared = 0.0L;
  for (const double value : interior_residual)
    norm_squared += static_cast<long double>(value) * value;
  for (const double value : true_trace_residual)
    norm_squared += static_cast<long double>(value) * value;
  residual = std::sqrt(static_cast<double>(norm_squared));
  const double comparison_scale = 1.0 + std::abs(evidence.initial_residual);
  relative_error = std::abs(residual - evidence.final_residual) /
      comparison_scale;
  const double target = 1.0e-10 * (1.0 + evidence.initial_residual);
  independent_success = admissible && std::isfinite(residual) &&
                        residual <= target;
  return std::isfinite(relative_error);
}

bool FactorAndSolve(std::vector<double> matrix, const int rows,
                    std::vector<double>& right_hand_sides, const int columns)
{
  for (int pivot = 0; pivot < rows; ++pivot)
  {
    int best = pivot;
    for (int row = pivot + 1; row < rows; ++row)
      if (std::abs(matrix[row * rows + pivot]) >
          std::abs(matrix[best * rows + pivot]))
      {
        best = row;
      }
    if (!(std::abs(matrix[best * rows + pivot]) > 1.0e-14)) { return false; }
    if (best != pivot)
      for (int column = 0; column < rows; ++column)
        std::swap(matrix[pivot * rows + column],
                  matrix[best * rows + column]);
    if (best != pivot)
      for (int column = 0; column < columns; ++column)
        std::swap(right_hand_sides[pivot * columns + column],
                  right_hand_sides[best * columns + column]);
    for (int row = pivot + 1; row < rows; ++row)
    {
      const double multiplier = matrix[row * rows + pivot] /
                                matrix[pivot * rows + pivot];
      matrix[row * rows + pivot] = 0.0;
      for (int column = pivot + 1; column < rows; ++column)
        matrix[row * rows + column] -=
            multiplier * matrix[pivot * rows + column];
      for (int column = 0; column < columns; ++column)
        right_hand_sides[row * columns + column] -=
            multiplier * right_hand_sides[pivot * columns + column];
    }
  }
  for (int row = rows - 1; row >= 0; --row)
    for (int rhs = 0; rhs < columns; ++rhs)
    {
      double value = right_hand_sides[row * columns + rhs];
      for (int column = row + 1; column < rows; ++column)
        value -= matrix[row * rows + column] *
                 right_hand_sides[column * columns + rhs];
      right_hand_sides[row * columns + rhs] =
          value / matrix[row * rows + row];
    }
  return true;
}

bool IndependentFirstLinearOracle(const FullDeviceTestSnapshot& snapshot,
                                  double& condensed_error,
                                  double& recovery_error)
{
  const int ni = snapshot.interior_dofs;
  const int nt = snapshot.trace_dofs;
  if (ni <= 0 || nt <= 0 ||
      snapshot.first_a.size() != static_cast<std::size_t>(ni * ni) ||
      snapshot.first_b.size() != static_cast<std::size_t>(ni * nt) ||
      snapshot.first_c.size() != static_cast<std::size_t>(nt * ni) ||
      snapshot.first_d.size() != static_cast<std::size_t>(nt * nt) ||
      snapshot.first_f.size() != static_cast<std::size_t>(ni) ||
      snapshot.first_g.size() != static_cast<std::size_t>(nt) ||
      snapshot.first_schur.size() != static_cast<std::size_t>(nt * nt) ||
      snapshot.first_condensed_rhs.size() != static_cast<std::size_t>(nt) ||
      snapshot.first_local_trace_increment.size() !=
          static_cast<std::size_t>(nt) ||
      snapshot.first_recovered_increment.size() !=
          static_cast<std::size_t>(ni))
  {
    return false;
  }
  std::vector<double> solved(static_cast<std::size_t>(ni) * (nt + 1));
  for (int row = 0; row < ni; ++row)
  {
    std::copy_n(snapshot.first_b.data() + row * nt, nt,
                solved.data() + row * (nt + 1));
    solved[row * (nt + 1) + nt] = snapshot.first_f[row];
  }
  if (!FactorAndSolve(snapshot.first_a, ni, solved, nt + 1)) { return false; }
  condensed_error = 0.0;
  for (int row = 0; row < nt; ++row)
  {
    double rhs = snapshot.first_g[row];
    for (int interior = 0; interior < ni; ++interior)
      rhs -= snapshot.first_c[row * ni + interior] *
             solved[interior * (nt + 1) + nt];
    condensed_error = std::max(
        condensed_error, std::abs(rhs - snapshot.first_condensed_rhs[row]));
    for (int column = 0; column < nt; ++column)
    {
      double value = snapshot.first_d[row * nt + column];
      for (int interior = 0; interior < ni; ++interior)
        value -= snapshot.first_c[row * ni + interior] *
                 solved[interior * (nt + 1) + column];
      condensed_error = std::max(
          condensed_error,
          std::abs(value - snapshot.first_schur[row * nt + column]));
    }
  }
  recovery_error = 0.0;
  for (int row = 0; row < ni; ++row)
  {
    double residual = -snapshot.first_f[row];
    for (int column = 0; column < ni; ++column)
      residual += snapshot.first_a[row * ni + column] *
                  snapshot.first_recovered_increment[column];
    for (int column = 0; column < nt; ++column)
      residual += snapshot.first_b[row * nt + column] *
                  snapshot.first_local_trace_increment[column];
    recovery_error = std::max(recovery_error, std::abs(residual));
  }
  return std::isfinite(condensed_error) && std::isfinite(recovery_error);
}

struct GlobalLinearOracleResult
{
  double condensation_error = std::numeric_limits<double>::infinity();
  double rhs_error = std::numeric_limits<double>::infinity();
  double true_matrix_error = std::numeric_limits<double>::infinity();
  double true_rhs_error = std::numeric_limits<double>::infinity();
  double action_error = std::numeric_limits<double>::infinity();
  double trace_error = std::numeric_limits<double>::infinity();
  double recovery_error = std::numeric_limits<double>::infinity();
  double state_error = std::numeric_limits<double>::infinity();
  double final_residual = std::numeric_limits<double>::infinity();
  double final_residual_error = std::numeric_limits<double>::infinity();
  bool status_matches = false;
};

bool IndependentGlobalLinearOracle(const FullDeviceMeshEvidence& evidence,
                                   GlobalLinearOracleResult& result)
{
  const auto& snapshot = evidence.test_snapshot;
  const int elements = snapshot.elements;
  const int ni = snapshot.interior_dofs;
  const int nt = snapshot.trace_dofs;
  const int ng = snapshot.global_trace_dofs;
  const auto sized = [](const std::vector<double>& values,
                        const std::size_t expected)
  { return values.size() == expected; };
  if (!snapshot.global_linear_oracle_staged || elements != 9 || ni <= 0 ||
      nt <= 0 || ng <= 0 ||
      snapshot.signed_trace_dofs.size() !=
          static_cast<std::size_t>(elements * nt) ||
      !sized(snapshot.global_a,
             static_cast<std::size_t>(elements) * ni * ni) ||
      !sized(snapshot.global_b,
             static_cast<std::size_t>(elements) * ni * nt) ||
      !sized(snapshot.global_c,
             static_cast<std::size_t>(elements) * nt * ni) ||
      !sized(snapshot.global_d,
             static_cast<std::size_t>(elements) * nt * nt) ||
      !sized(snapshot.global_f, static_cast<std::size_t>(elements) * ni) ||
      !sized(snapshot.global_g, static_cast<std::size_t>(elements) * nt) ||
      !sized(snapshot.global_schur,
             static_cast<std::size_t>(elements) * nt * nt) ||
      !sized(snapshot.global_condensed_rhs,
             static_cast<std::size_t>(elements) * nt) ||
      !sized(snapshot.global_trace_increment, ng) ||
      !sized(snapshot.global_local_trace_increment,
             static_cast<std::size_t>(elements) * nt) ||
      !sized(snapshot.global_recovered_increment,
             static_cast<std::size_t>(elements) * ni) ||
      snapshot.actual_true_csr_offsets.size() !=
          static_cast<std::size_t>(ng + 1) ||
      snapshot.actual_true_csr_columns.size() !=
          snapshot.actual_true_csr_values.size() ||
      !sized(snapshot.actual_true_rhs, ng))
  {
    return false;
  }

  std::vector<double> global_matrix(static_cast<std::size_t>(ng) * ng, 0.0);
  std::vector<double> global_rhs(ng, 0.0);
  result.condensation_error = 0.0;
  result.rhs_error = 0.0;
  for (int element = 0; element < elements; ++element)
  {
    const double* a = snapshot.global_a.data() +
                      static_cast<std::size_t>(element) * ni * ni;
    const double* b = snapshot.global_b.data() +
                      static_cast<std::size_t>(element) * ni * nt;
    const double* c = snapshot.global_c.data() +
                      static_cast<std::size_t>(element) * nt * ni;
    const double* d = snapshot.global_d.data() +
                      static_cast<std::size_t>(element) * nt * nt;
    const double* f = snapshot.global_f.data() +
                      static_cast<std::size_t>(element) * ni;
    const double* g = snapshot.global_g.data() +
                      static_cast<std::size_t>(element) * nt;
    std::vector<double> matrix(a, a + ni * ni);
    std::vector<double> solved(static_cast<std::size_t>(ni) * (nt + 1));
    for (int row = 0; row < ni; ++row)
    {
      std::copy_n(b + row * nt, nt, solved.data() + row * (nt + 1));
      solved[row * (nt + 1) + nt] = f[row];
    }
    if (!FactorAndSolve(matrix, ni, solved, nt + 1)) { return false; }
    for (int row = 0; row < nt; ++row)
    {
      double local_rhs = g[row];
      for (int interior = 0; interior < ni; ++interior)
        local_rhs -= c[row * ni + interior] *
                     solved[interior * (nt + 1) + nt];
      const std::size_t local_row =
          static_cast<std::size_t>(element) * nt + row;
      result.rhs_error = std::max(
          result.rhs_error,
          std::abs(local_rhs - snapshot.global_condensed_rhs[local_row]));
      double row_sign = 1.0;
      const int global_row =
          Decode(snapshot.signed_trace_dofs[local_row], row_sign);
      global_rhs[global_row] += row_sign * local_rhs;
      for (int column = 0; column < nt; ++column)
      {
        double local_value = d[row * nt + column];
        for (int interior = 0; interior < ni; ++interior)
          local_value -= c[row * ni + interior] *
                         solved[interior * (nt + 1) + column];
        const std::size_t local_entry =
            (static_cast<std::size_t>(element) * nt + row) * nt + column;
        result.condensation_error = std::max(
            result.condensation_error,
            std::abs(local_value - snapshot.global_schur[local_entry]));
        double column_sign = 1.0;
        const int global_column = Decode(
            snapshot.signed_trace_dofs[static_cast<std::size_t>(element) * nt +
                                       column],
            column_sign);
        global_matrix[static_cast<std::size_t>(global_row) * ng +
                      global_column] +=
            row_sign * column_sign * local_value;
      }
    }
  }

  std::vector<double> actual_matrix(static_cast<std::size_t>(ng) * ng, 0.0);
  result.true_matrix_error = 0.0;
  result.true_rhs_error = 0.0;
  if (snapshot.actual_true_csr_offsets.front() != 0 ||
      snapshot.actual_true_csr_offsets.back() !=
          static_cast<int>(snapshot.actual_true_csr_values.size()))
  {
    return false;
  }
  for (int row = 0; row < ng; ++row)
  {
    const int begin = snapshot.actual_true_csr_offsets[row];
    const int end = snapshot.actual_true_csr_offsets[row + 1];
    if (begin < 0 || end < begin ||
        end > static_cast<int>(snapshot.actual_true_csr_values.size()))
    {
      return false;
    }
    for (int entry = begin; entry < end; ++entry)
    {
      const int column = snapshot.actual_true_csr_columns[entry];
      if (column < 0 || column >= ng) { return false; }
      actual_matrix[static_cast<std::size_t>(row) * ng + column] +=
          snapshot.actual_true_csr_values[entry];
    }
    result.true_rhs_error = std::max(
        result.true_rhs_error,
        std::abs(snapshot.actual_true_rhs[row] - global_rhs[row]));
  }
  for (std::size_t entry = 0; entry < actual_matrix.size(); ++entry)
    result.true_matrix_error = std::max(
        result.true_matrix_error,
        std::abs(actual_matrix[entry] - global_matrix[entry]));

  long double rhs_norm_squared = 0.0L;
  long double action_norm_squared = 0.0L;
  for (int row = 0; row < ng; ++row)
  {
    double action = -global_rhs[row];
    for (int column = 0; column < ng; ++column)
      action += global_matrix[static_cast<std::size_t>(row) * ng + column] *
                snapshot.global_trace_increment[column];
    rhs_norm_squared += static_cast<long double>(global_rhs[row]) *
                        global_rhs[row];
    action_norm_squared += static_cast<long double>(action) * action;
  }
  result.action_error = std::sqrt(static_cast<double>(action_norm_squared)) /
                        std::max(std::sqrt(static_cast<double>(rhs_norm_squared)),
                                 1.0e-300);

  std::vector<double> independent_trace = global_rhs;
  if (!FactorAndSolve(global_matrix, ng, independent_trace, 1)) { return false; }
  result.trace_error = 0.0;
  result.recovery_error = 0.0;
  for (int entry = 0; entry < ng; ++entry)
    result.trace_error = std::max(
        result.trace_error,
        std::abs(independent_trace[entry] -
                 snapshot.global_trace_increment[entry]));
  for (int element = 0; element < elements; ++element)
  {
    const double* a = snapshot.global_a.data() +
                      static_cast<std::size_t>(element) * ni * ni;
    const double* b = snapshot.global_b.data() +
                      static_cast<std::size_t>(element) * ni * nt;
    const double* f = snapshot.global_f.data() +
                      static_cast<std::size_t>(element) * ni;
    std::vector<double> local_trace(nt);
    for (int column = 0; column < nt; ++column)
    {
      double sign = 1.0;
      const int global = Decode(
          snapshot.signed_trace_dofs[static_cast<std::size_t>(element) * nt +
                                     column],
          sign);
      local_trace[column] = sign * independent_trace[global];
      result.trace_error = std::max(
          result.trace_error,
          std::abs(local_trace[column] -
                   snapshot.global_local_trace_increment[
                       static_cast<std::size_t>(element) * nt + column]));
    }
    std::vector<double> recovered(ni);
    for (int row = 0; row < ni; ++row)
    {
      recovered[row] = f[row];
      for (int column = 0; column < nt; ++column)
        recovered[row] -= b[row * nt + column] * local_trace[column];
    }
    if (!FactorAndSolve(std::vector<double>(a, a + ni * ni), ni, recovered, 1))
      return false;
    for (int row = 0; row < ni; ++row)
      result.recovery_error = std::max(
          result.recovery_error,
          std::abs(recovered[row] -
                   snapshot.global_recovered_increment[
                       static_cast<std::size_t>(element) * ni + row]));
  }

  if (!sized(snapshot.last_base_interior,
             static_cast<std::size_t>(elements) * ni) ||
      !sized(snapshot.last_delta_interior,
             static_cast<std::size_t>(elements) * ni) ||
      !sized(snapshot.last_base_trace, ng) ||
      !sized(snapshot.last_delta_trace, ng) ||
      !sized(snapshot.final_interior,
             static_cast<std::size_t>(elements) * ni) ||
      !sized(snapshot.final_trace, ng) ||
      !(snapshot.last_accepted_alpha > 0.0))
  {
    return false;
  }
  bool independent_success = false;
  if (!IndependentFinalPdeResidual(evidence, result.final_residual,
                                   result.final_residual_error,
                                   independent_success))
  {
    return false;
  }
  result.state_error = 0.0;
  for (std::size_t entry = 0; entry < snapshot.final_interior.size(); ++entry)
    result.state_error = std::max(
        result.state_error,
        std::abs(snapshot.final_interior[entry] -
                 (snapshot.last_base_interior[entry] +
                  snapshot.last_accepted_alpha *
                      snapshot.last_delta_interior[entry])));
  for (std::size_t entry = 0; entry < snapshot.final_trace.size(); ++entry)
    result.state_error = std::max(
        result.state_error,
        std::abs(snapshot.final_trace[entry] -
                 (snapshot.last_base_trace[entry] +
                  snapshot.last_accepted_alpha *
                      snapshot.last_delta_trace[entry])));
  const double target = 1.0e-10 * (1.0 + evidence.initial_residual);
  result.status_matches = independent_success &&
                          evidence.status == FullDeviceStatus::success &&
                          evidence.pass && evidence.nonlinear_iterations >= 2 &&
                          result.final_residual <= target;
  return std::isfinite(result.condensation_error) &&
         std::isfinite(result.rhs_error) &&
         std::isfinite(result.true_matrix_error) &&
         std::isfinite(result.true_rhs_error) &&
         std::isfinite(result.action_error) &&
         std::isfinite(result.trace_error) &&
         std::isfinite(result.recovery_error) &&
         std::isfinite(result.state_error) &&
         std::isfinite(result.final_residual_error);
}

bool Flux(const std::array<double, conserved_components>& state,
          const double nx, const double ny,
          std::array<double, conserved_components>& flux)
{
  if (!(state[0] > 0.0)) { return false; }
  const double pressure = (gamma - 1.0) *
      (state[3] - 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                       state[0]);
  if (!(pressure > 0.0)) { return false; }
  const double q = state[1] * nx + state[2] * ny;
  flux = {{q, state[1] * q / state[0] + pressure * nx,
           state[2] * q / state[0] + pressure * ny,
           (state[3] + pressure) * q / state[0]}};
  return true;
}

bool IndependentPhysicsProbe()
{
  const std::array<double, conserved_components> state{{1.2, 0.48, -0.3,
                                                         2.3835}};
  const std::array<double, 8> normals{{1.0, 0.0, -1.0, 0.0,
                                       0.0, 1.0, 0.0, -1.0}};
  std::array<double, conserved_components> sum{{0.0, 0.0, 0.0, 0.0}};
  for (int face = 0; face < 4; ++face)
  {
    std::array<double, conserved_components> value{};
    if (!Flux(state, normals[2 * face], normals[2 * face + 1], value))
      return false;
    for (int component = 0; component < conserved_components; ++component)
      sum[component] += value[component];
  }
  return *std::max_element(sum.begin(), sum.end(),
                           [](const double first, const double second)
                           { return std::abs(first) < std::abs(second); }) <=
         5.0e-12;
}

bool ValidateCase(const FullDeviceMeshEvidence& value)
{
  if (!value.pass || value.status != FullDeviceStatus::success ||
      value.polynomial_order < 1 || value.polynomial_order > 4 ||
      (value.elements_per_direction != 3 &&
       value.elements_per_direction != 6 &&
       value.elements_per_direction != 12) ||
      value.nonlinear_iterations < 2 || !(value.final_residual < value.initial_residual) ||
      value.max_physics_finite_difference_error > 2.0e-6 ||
      value.constant_state_residual > 5.0e-12 ||
      value.local_equation_error > 5.0e-11 ||
      value.condensed_equation_error > 5.0e-11 ||
      value.schur_action_error > 5.0e-9 ||
      value.trace_solution_error > 5.0e-9 ||
      value.recovery_error > 5.0e-9 || value.state_update_error > 5.0e-9 ||
      value.true_relative_residual > 1.0e-10 || !value.bynodes_vdim4 ||
      value.negative_trace_vdof_count != 0 ||
      !value.reversed_trace_permutation_observed ||
      !value.shared_face_orientation_action_verified ||
      !value.exact_ptslp_assembly ||
      !value.hypre_matrix_device ||
      !value.hypre_rhs_device || !value.hypre_solution_device ||
      !value.hypre_policy_before_solve_device ||
      !value.hypre_policy_after_solve_device ||
      !value.hypre_gmres || !value.hypre_ilu_type_zero_fill_zero ||
      value.effective_hypre_ilu_type != 0 ||
      value.effective_hypre_ilu_level_of_fill != 0 ||
      value.telemetry.hypre_ilu_configuration_checks <
          static_cast<std::size_t>(value.nonlinear_iterations + 1) ||
      value.telemetry.hypre_setup_refresh_events <
          static_cast<std::size_t>(value.nonlinear_iterations - 1) ||
      value.telemetry.hypre_shape_warmup_setup_events == 0 ||
      value.telemetry.hypre_persistent_bytes == 0 ||
      value.telemetry.whole_transaction_zero_allocation_claimed ||
      !value.telemetry.hypre_setup_refresh_allocation_churn_reported ||
      !value.telemetry.external_umpire_event_audit_required ||
      value.telemetry.external_zero_allocation_intervals_marked == 0 ||
      value.telemetry.external_allocation_churn_intervals_marked == 0 ||
      value.telemetry.allocator_snapshot_events <
          static_cast<std::size_t>(2 * value.nonlinear_iterations) ||
      value.telemetry.mfem_allocator_high_watermark_bytes == 0 ||
      value.telemetry.hypre_allocator_high_watermark_bytes == 0 ||
      value.telemetry.managed_device_objects_checked == 0 ||
      value.telemetry.managed_device_objects_checked !=
          value.telemetry.managed_device_objects_valid ||
      value.telemetry.backend_selection_checks == 0 ||
      !value.telemetry.allocator_endpoint_census_complete ||
      !value.telemetry.app_forall_fallback_census_complete ||
      value.telemetry.mfem_internal_backend_observable ||
      value.telemetry.hypre_internal_backend_observable ||
      value.telemetry.fallback_census_complete ||
      value.telemetry.app_forall_submissions == 0 ||
      value.telemetry.app_forall_submissions !=
          value.telemetry.app_forall_device_tickets ||
      value.telemetry.app_forall_host_fallbacks != 0 ||
      value.telemetry.app_forall_rejected_submissions != 0 ||
      value.telemetry.external_umpire_event_results_available_in_process ||
      !value.telemetry
           .marked_zero_allocation_scope_excludes_trace_assembly_solve ||
      !value.telemetry.cuda_kernel_execution_verified ||
      !value.allocation_observability_reported ||
      !value.app_forall_fallback_census_complete ||
      value.mfem_internal_backend_observable ||
      value.hypre_internal_backend_observable ||
      value.fallback_census_complete ||
      !value.phase_audit_append_only ||
      !value.managed_residency_verified ||
      !value.cuda_kernel_execution_verified ||
      value.telemetry.fallback_events != 0 ||
      value.telemetry.dense_host_trace_inverse_events != 0 ||
      value.telemetry.solver_substitution_events != 0 ||
      value.telemetry.test_only_device_to_host_bytes == 0)
  {
    std::cerr << "HDG_EULER_DEVICE_FULL_VALIDATE_CASE core_contract"
              << " ilu_checks="
              << value.telemetry.hypre_ilu_configuration_checks
              << " refresh=" << value.telemetry.hypre_setup_refresh_events
              << " warmup="
              << value.telemetry.hypre_shape_warmup_setup_events
              << " hypre_bytes=" << value.telemetry.hypre_persistent_bytes
              << " intervals="
              << value.telemetry.external_zero_allocation_intervals_marked
              << "/"
              << value.telemetry.external_allocation_churn_intervals_marked
              << " allocator_snapshots="
              << value.telemetry.allocator_snapshot_events
              << " high_water="
              << value.telemetry.mfem_allocator_high_watermark_bytes << "/"
              << value.telemetry.hypre_allocator_high_watermark_bytes
              << " managed="
              << value.telemetry.managed_device_objects_valid << "/"
              << value.telemetry.managed_device_objects_checked
              << " backend_checks="
              << value.telemetry.backend_selection_checks
              << " endpoint="
              << value.telemetry.allocator_endpoint_census_complete
              << " test_d2h="
              << value.telemetry.test_only_device_to_host_bytes << '\n';
    return false;
  }
  for (std::size_t phase = 0; phase < value.phases.size(); ++phase)
  {
    const auto& record = value.phases[phase];
    if (!record.reached || !record.device_resident ||
        record.status != FullDeviceStatus::success ||
        record.phase != static_cast<FullDevicePhase>(phase) ||
        !record.contemporaneous_audit ||
        !record.app_forall_census_complete ||
        (record.app_forall_submissions == 0 &&
         record.phase != FullDevicePhase::trace_solve) ||
        record.app_forall_submissions != record.app_forall_device_tickets ||
        record.app_forall_host_fallbacks != 0 ||
        record.managed_device_objects_checked == 0 ||
        record.managed_device_objects_checked !=
            record.managed_device_objects_valid ||
        record.mfem_internal_backend_observable ||
        record.hypre_internal_backend_observable)
    {
      std::cerr << "HDG_EULER_DEVICE_FULL_VALIDATE_CASE phase"
                << " index=" << phase
                << " reached=" << record.reached
                << " resident=" << record.device_resident
                << " status=" << static_cast<int>(record.status)
                << " phase=" << static_cast<int>(record.phase)
                << " contemporary=" << record.contemporaneous_audit
                << " census=" << record.app_forall_census_complete
                << " submissions=" << record.app_forall_submissions
                << " tickets=" << record.app_forall_device_tickets
                << " host=" << record.app_forall_host_fallbacks
                << " managed=" << record.managed_device_objects_valid << "/"
                << record.managed_device_objects_checked << '\n';
      return false;
    }
  }
  if (value.phase_audit_log.size() < value.phases.size() ||
      std::any_of(value.phase_audit_log.begin(), value.phase_audit_log.end(),
                  [](const auto& record)
                  {
                    return !record.reached ||
                           !record.contemporaneous_audit ||
                           !record.device_resident ||
                           record.status != FullDeviceStatus::success;
                  }))
  {
    std::cerr << "HDG_EULER_DEVICE_FULL_VALIDATE_CASE phase_log"
              << " records=" << value.phase_audit_log.size()
              << " expected_at_least=" << value.phases.size() << '\n';
    return false;
  }
  ErrorIntegral volume, trace;
  bool signed_map_exact = false;
  const bool independent_errors =
      IndependentErrors(value.test_snapshot, volume, trace, signed_map_exact);
  const bool orientation = IndependentOrientationProbe(
      value.polynomial_order, value.elements_per_direction);
  const bool negative_snapshot = std::any_of(
      value.test_snapshot.signed_trace_dofs.begin(),
      value.test_snapshot.signed_trace_dofs.end(),
      [](const int dof) { return dof < 0; });
  if (!independent_errors || !signed_map_exact || !orientation ||
      negative_snapshot)
  {
    std::cerr << "HDG_EULER_DEVICE_FULL_VALIDATE_CASE reconstruction"
              << " errors=" << independent_errors
              << " signed=" << signed_map_exact
              << " orientation=" << orientation
              << " negative=" << negative_snapshot << '\n';
    return false;
  }
  const double expected_trace_measure = 2.0 * value.elements_per_direction;
  const double oracle_volume = volume.Relative();
  const double oracle_trace = trace.Relative();
  const auto agrees = [](const double first, const double second)
  {
    return std::abs(first - second) /
               std::max({std::abs(first), std::abs(second), 1.0e-300}) <=
           1.0e-8;
  };
  if (!std::isfinite(oracle_volume) || !std::isfinite(oracle_trace) ||
      oracle_volume < 1.0e-10 || oracle_trace < 1.0e-10 ||
      std::abs(static_cast<double>(volume.measure) - 1.0) >
          256.0 * experiment_epsilon ||
      std::abs(static_cast<double>(trace.measure) - expected_trace_measure) >
          256.0 * experiment_epsilon * expected_trace_measure ||
      trace.faces != 2 * value.elements_per_direction *
                         value.elements_per_direction ||
      !agrees(oracle_volume, value.volume_relative_l2_qe4) ||
      !agrees(oracle_trace, value.trace_relative_l2_qe4))
  {
    std::cerr << "HDG_EULER_DEVICE_FULL_VALIDATE_CASE error_integrals"
              << " volume=" << oracle_volume << "/"
              << value.volume_relative_l2_qe4
              << " trace=" << oracle_trace << "/"
              << value.trace_relative_l2_qe4
              << " measures=" << static_cast<double>(volume.measure) << "/"
              << static_cast<double>(trace.measure)
              << " faces=" << trace.faces << '\n';
    return false;
  }
  if (value.elements_per_direction == 3)
  {
    double independent_residual = std::numeric_limits<double>::infinity();
    double residual_error = std::numeric_limits<double>::infinity();
    bool independent_success = false;
    if (!IndependentFinalPdeResidual(value, independent_residual,
                                     residual_error, independent_success) ||
        !independent_success || residual_error > 5.0e-9)
    {
      std::cerr << "HDG_EULER_DEVICE_FULL_VALIDATE_CASE final_pde"
                << " residual=" << independent_residual
                << " error=" << residual_error
                << " success=" << independent_success
                << " production_nonlinear=" << value.final_residual
                << " production_linear_true="
                << value.true_relative_residual
                << " initial=" << value.initial_residual
                << " target="
                << 1.0e-10 * (1.0 + value.initial_residual)
                << " comparison_scale="
                << 1.0 + std::abs(value.initial_residual)
                << " assembly_q="
                << value.test_snapshot.assembly_quadrature_order
                << " independent_q="
                << 5 * value.polynomial_order + 16
                << " committed_generation=" << value.committed_generation
                << '\n';
      return false;
    }
    double condensed = 0.0, recovery = 0.0;
    if (!IndependentFirstLinearOracle(value.test_snapshot, condensed,
                                      recovery) ||
        condensed > 5.0e-11 || recovery > 5.0e-9)
    {
      std::cerr << "HDG_EULER_DEVICE_FULL_VALIDATE_CASE local_oracle"
                << " condensed=" << condensed
                << " recovery=" << recovery << '\n';
      return false;
    }
  }
  return true;
}

void PrintFailure(const FullDeviceEvidence& evidence, const std::string& stage)
{
  std::cerr << "HDG_EULER_DEVICE_FULL_FAILURE stage=" << stage
            << " status=" << static_cast<int>(evidence.status)
            << " cases=" << evidence.cases.size() << '\n';
  for (const auto& value : evidence.cases)
  {
    std::cerr << "  p=" << value.polynomial_order
              << " N=" << value.elements_per_direction
              << " status=" << static_cast<int>(value.status)
              << " nonlinear=" << value.nonlinear_iterations
              << " linear=" << value.linear_iterations
              << " residual=" << value.final_residual
              << " true_residual=" << value.true_relative_residual
              << " action=" << value.schur_action_error
              << " trace_solution=" << value.trace_solution_error
              << " local=" << value.local_equation_error
              << " condensed=" << value.condensed_equation_error
              << " recovery=" << value.recovery_error
              << " state=" << value.state_update_error
              << " physics=" << value.max_physics_finite_difference_error
              << "/" << value.constant_state_residual
              << " layout=" << value.bynodes_vdim4
              << "/" << value.negative_trace_vdof_count
              << "/" << value.reversed_trace_permutation_observed
              << "/" << value.shared_face_orientation_action_verified
              << "/" << value.exact_ptslp_assembly
              << " min_diag=" << value.minimum_trace_diagonal_magnitude
              << " max_entry=" << value.maximum_trace_matrix_entry
              << " steady_alloc="
              << value.telemetry.hypre_refresh_endpoint_growth_events
              << "/" << value.telemetry.hypre_refresh_endpoint_growth_bytes
              << " allocator_snapshots="
              << value.telemetry.allocator_snapshot_events
              << " managed=" << value.telemetry.managed_device_objects_valid
              << "/" << value.telemetry.managed_device_objects_checked
              << " pass=" << value.pass
              << " allocation_report="
              << value.allocation_observability_reported
              << " app_census="
              << value.app_forall_fallback_census_complete
              << " aggregate_census=" << value.fallback_census_complete
              << " kernel=" << value.cuda_kernel_execution_verified
              << " hypre_policy=" << value.hypre_policy_before_solve_device
              << "/" << value.hypre_policy_after_solve_device
              << " hypre_path=" << value.hypre_matrix_device << "/"
              << value.hypre_rhs_device << "/"
              << value.hypre_solution_device << "/" << value.hypre_gmres
              << "/" << value.hypre_ilu_type_zero_fill_zero
              << " internal_observable="
              << value.mfem_internal_backend_observable << "/"
              << value.hypre_internal_backend_observable
              << " app_tickets=" << value.telemetry.app_forall_submissions
              << "/" << value.telemetry.app_forall_device_tickets
              << "/" << value.telemetry.app_forall_host_fallbacks
              << " app_rejected="
              << value.telemetry.app_forall_rejected_submissions
              << " phase_log=" << value.phase_audit_log.size()
              << " append=" << value.phase_audit_append_only
              << " fallback=" << value.telemetry.fallback_events
              << " dense="
              << value.telemetry.dense_host_trace_inverse_events
              << " substitution="
              << value.telemetry.solver_substitution_events
              << " rollback=" << value.rollback_restored
              << " volume=" << value.volume_relative_l2_qe4
              << " trace=" << value.trace_relative_l2_qe4 << '\n';
    for (const auto& phase : value.phases)
    {
      std::cerr << "    phase=" << static_cast<int>(phase.phase)
                << " reached=" << phase.reached
                << " status=" << static_cast<int>(phase.status)
                << " resident=" << phase.device_resident
                << " current=" << phase.contemporaneous_audit
                << " app=" << phase.app_forall_submissions << "/"
                << phase.app_forall_device_tickets << "/"
                << phase.app_forall_host_fallbacks
                << " managed=" << phase.managed_device_objects_valid << "/"
                << phase.managed_device_objects_checked << '\n';
    }
  }
}
}  // namespace

int main(int argc, char* argv[])
{
  if (argc == 2 && std::string(argv[1]) == "--umpire-event-audit-driver")
    return RunUmpireAuditDriver(argv[0]);
  const bool umpire_audit_child =
      argc == 2 && std::string(argv[1]) == "--umpire-audit-child";
  if (argc != 1 && !umpire_audit_child) { return 73; }
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  if (!IndependentPhysicsProbe()) { return 1; }

  FullDeviceEvidence disabled;
  if (dg::hdg::euler::RunSmoothEulerDeviceFullExperiment({}, disabled) ||
      disabled.status != FullDeviceStatus::default_disabled ||
      !disabled.cases.empty() || disabled.pass)
  {
    PrintFailure(disabled, "default_off");
    return 2;
  }

  FullDeviceRequest inconsistent_request;
  inconsistent_request.enabled = true;
  inconsistent_request.run_complete_ladder = false;
  inconsistent_request.enable_global_linear_oracle_staging = true;
  FullDeviceEvidence inconsistent;
  if (dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(
          inconsistent_request, inconsistent) ||
      inconsistent.status != FullDeviceStatus::setup_failure ||
      !inconsistent.cases.empty() || inconsistent.pass)
  {
    PrintFailure(inconsistent, "oracle_request_contract");
    return 10;
  }

  FullDeviceRequest smoke_request;
  smoke_request.enabled = true;
  smoke_request.run_complete_ladder = false;
  smoke_request.enable_test_oracle_staging = true;
  smoke_request.enable_global_linear_oracle_staging = true;
  FullDeviceEvidence smoke;
  if (!dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(smoke_request,
                                                           smoke) ||
      !smoke.pass || smoke.status != FullDeviceStatus::success ||
      smoke.cases.size() != 1 || smoke.cases[0].polynomial_order != 1 ||
      smoke.cases[0].elements_per_direction != 3 ||
      !ValidateCase(smoke.cases[0]))
  {
    PrintFailure(smoke, "p1_n3_smoke");
    return 3;
  }
  GlobalLinearOracleResult global_oracle;
  if (!IndependentGlobalLinearOracle(smoke.cases[0], global_oracle) ||
      !global_oracle.status_matches ||
      global_oracle.condensation_error > 5.0e-11 ||
      global_oracle.rhs_error > 5.0e-11 ||
      global_oracle.true_matrix_error > 5.0e-11 ||
      global_oracle.true_rhs_error > 5.0e-11 ||
      global_oracle.action_error > 1.0e-10 ||
      global_oracle.trace_error > 5.0e-9 ||
      global_oracle.recovery_error > 5.0e-9 ||
      global_oracle.state_error > 5.0e-9 ||
      global_oracle.final_residual_error > 5.0e-9)
  {
    PrintFailure(smoke, "p1_n3_global_oracle");
    return 8;
  }

  FullDeviceRequest request;
  request.enabled = true;
  request.run_complete_ladder = true;
  request.enable_test_oracle_staging = true;
  request.enable_global_linear_oracle_staging = true;
  FullDeviceEvidence evidence;
  if (!dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(request, evidence) ||
      !evidence.pass || evidence.status != FullDeviceStatus::success ||
      !evidence.real_mfem_cuda_path || !evidence.real_hypre_device_path ||
      evidence.cases.size() != 12)
  {
    PrintFailure(evidence, "campaign");
    return 4;
  }
  for (const auto& value : evidence.cases)
  {
    if (!ValidateCase(value))
    {
      PrintFailure(evidence, "case_validation");
      return 5;
    }
  }
  if (std::count_if(evidence.cases.begin(), evidence.cases.end(),
                    [](const FullDeviceMeshEvidence& value)
                    {
                      return value.test_snapshot.global_linear_oracle_staged;
                    }) != 1 ||
      !evidence.cases.front().test_snapshot.global_linear_oracle_staged)
  {
    PrintFailure(evidence, "bounded_global_staging");
    return 11;
  }
  for (int p = 1; p <= 4; ++p)
  {
    const auto& convergence = evidence.convergence[p - 1];
    if (!convergence.pass || convergence.polynomial_order != p ||
        convergence.volume_rates[0] < p + 0.25 ||
        convergence.volume_rates[1] < p + 0.25 ||
        convergence.trace_rates[0] < p + 0.45 ||
        convergence.trace_rates[1] < p + 0.45 ||
        convergence.trace_three_level_slope < p + 0.50)
    {
      PrintFailure(evidence, "convergence");
      return 6;
    }
  }

  FullDeviceRequest rollback_request;
  rollback_request.enabled = true;
  rollback_request.run_complete_ladder = false;
  rollback_request.enable_test_oracle_staging = true;
  rollback_request.inject_failure_after_first_update = true;
  rollback_request.inject_nonfinite_before_rollback = true;
  FullDeviceEvidence rollback;
  if (dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(rollback_request,
                                                          rollback) ||
      rollback.status != FullDeviceStatus::injected_failure ||
      rollback.cases.size() != 1 || rollback.cases[0].pass)
  {
    PrintFailure(rollback, "rollback");
    return 7;
  }
  const auto& rolled = rollback.cases[0];
  const auto& rollback_snapshot = rolled.test_snapshot;
  const bool interior_changed = !BitwiseEqual(
      rollback_snapshot.rollback_entry_interior,
      rollback_snapshot.rollback_changed_interior);
  const bool trace_changed = !BitwiseEqual(
      rollback_snapshot.rollback_entry_trace,
      rollback_snapshot.rollback_changed_trace);
  const bool nonfinite_changed = std::any_of(
      rollback_snapshot.rollback_changed_interior.begin(),
      rollback_snapshot.rollback_changed_interior.end(),
      [](const double value) { return !std::isfinite(value); });
  const auto& failed_phase =
      rolled.phases[static_cast<std::size_t>(FullDevicePhase::state_update)];
  if (!rolled.rollback_restored || !rolled.rollback_had_uncommitted_change ||
      !rolled.rollback_nonfinite_change_observed || !nonfinite_changed ||
      !rolled.rollback_rejected_post_failure_consumption ||
      rolled.invalidated_product_count != 5 ||
      rolled.rejected_consumer_count != 6 ||
      rolled.nonlinear_iterations != 0 || rolled.linear_iterations != 0 ||
      rolled.committed_generation != 0 ||
      rolled.discarded_nonlinear_iterations != 1 ||
      rolled.discarded_linear_iterations <= 0 ||
      rolled.discarded_attempt_generation == 0 ||
      !(rolled.discarded_accepted_residual >
        1.0e-10 * (1.0 + rolled.initial_residual)) ||
      !interior_changed || !trace_changed ||
      !BitwiseEqual(rollback_snapshot.rollback_entry_interior,
                    rollback_snapshot.rollback_restored_interior) ||
      !BitwiseEqual(rollback_snapshot.rollback_entry_trace,
                    rollback_snapshot.rollback_restored_trace) ||
      !DerivedSnapshotEmpty(rollback_snapshot) || !failed_phase.reached ||
      failed_phase.status != FullDeviceStatus::injected_failure ||
      failed_phase.device_resident ||
      failed_phase.generation == 0 ||
      !rolled.phase_audit_append_only ||
      !rolled.failure_phase_audit_valid ||
      rolled.failure_phase != FullDevicePhase::state_update ||
      rolled.failure_phase_status != FullDeviceStatus::injected_failure ||
      rolled.failure_phase_generation !=
          rolled.discarded_attempt_generation ||
      rolled.phase_audit_log.empty() ||
      rolled.phase_audit_log.back().phase !=
          FullDevicePhase::state_update ||
      rolled.phase_audit_log.back().status !=
          FullDeviceStatus::injected_failure ||
      rolled.phase_audit_log.back().device_resident ||
      std::any_of(rolled.phase_audit_log.begin(),
                  rolled.phase_audit_log.end() - 1,
                  [](const auto& record)
                  {
                    return !record.reached ||
                           !record.contemporaneous_audit ||
                           !record.device_resident ||
                           record.status != FullDeviceStatus::success;
                  }) ||
      !rolled.phases[static_cast<std::size_t>(FullDevicePhase::trace_solve)]
           .reached ||
      !rolled.phases[static_cast<std::size_t>(FullDevicePhase::recovery)]
           .reached)
  {
    PrintFailure(rollback, "rollback_transaction");
    return 9;
  }

  FullDeviceRequest late_failure_request;
  late_failure_request.enabled = true;
  late_failure_request.run_complete_ladder = true;
  late_failure_request.enable_test_oracle_staging = true;
  late_failure_request.inject_failure_after_first_update = true;
  late_failure_request.injected_failure_polynomial_order = 2;
  late_failure_request.injected_failure_elements_per_direction = 3;
  FullDeviceEvidence late_failure;
  if (dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(
          late_failure_request, late_failure) ||
      late_failure.status != FullDeviceStatus::injected_failure ||
      late_failure.cases.size() != 1 ||
      late_failure.cases.front().polynomial_order != 2 ||
      late_failure.cases.front().elements_per_direction != 3 ||
      !DerivedSnapshotEmpty(late_failure.cases.front().test_snapshot) ||
      std::any_of(late_failure.convergence.begin(),
                  late_failure.convergence.end(),
                  [](const auto& item) { return item.pass; }))
  {
    PrintFailure(late_failure, "prior_campaign_sanitization");
    return 12;
  }

  std::cout << "HDG_EULER_DEVICE_FULL_SUCCESS cases=" << evidence.cases.size()
            << " rollback=restored\n";
  if (umpire_audit_child) { umpire::mark_event(umpire_terminal_marker); }
  return 0;
}
