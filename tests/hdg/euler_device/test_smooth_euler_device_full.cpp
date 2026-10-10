#include "numerics/hdg/euler/smooth_euler_device_full.hpp"

#include <HYPRE_utilities.h>
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
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace
{
using dg::hdg::euler::FullDeviceEvidence;
using dg::hdg::euler::FullDeviceFailurePoint;
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

bool GlobalAll(const bool local)
{
  const int local_value = local ? 1 : 0;
  int global_value = 0;
  MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  return global_value == 1;
}

double GlobalMaximum(const double local)
{
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return global;
}

bool DerivedSnapshotEmpty(const FullDeviceTestSnapshot& snapshot)
{
  return snapshot.polynomial_order == 0 &&
         snapshot.elements_per_direction == 0 && snapshot.elements == 0 &&
         snapshot.interior_dofs == 0 && snapshot.trace_dofs == 0 &&
         snapshot.global_trace_dofs == 0 &&
         snapshot.assembly_quadrature_order == 0 &&
         snapshot.signed_trace_dofs.empty() &&
         snapshot.signed_global_trace_dofs.empty() &&
         snapshot.local_trace_global_true_dofs.empty() &&
         snapshot.owned_true_global_dofs.empty() &&
         snapshot.element_coordinate_keys.empty() &&
         snapshot.ghost_probe_owned.empty() &&
         snapshot.ghost_probe_distributed.empty() &&
         snapshot.old_interior.empty() &&
         snapshot.initial_interior.empty() && snapshot.initial_trace.empty() &&
         snapshot.final_interior.empty() && snapshot.final_trace.empty() &&
         snapshot.first_a.empty() && snapshot.first_b.empty() &&
         snapshot.first_c.empty() && snapshot.first_d.empty() &&
         snapshot.first_f.empty() && snapshot.first_g.empty() &&
         snapshot.first_schur.empty() &&
         snapshot.first_condensed_rhs.empty() &&
         snapshot.first_local_trace_increment.empty() &&
         snapshot.first_recovered_increment.empty() &&
         snapshot.global_residual_r.empty() &&
         snapshot.global_residual_g.empty() &&
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
         snapshot.actual_true_solution.empty() &&
         snapshot.actual_true_basis_actions.empty() &&
         snapshot.final_residual_r.empty() &&
         snapshot.final_residual_g.empty() &&
         snapshot.last_base_interior.empty() &&
         snapshot.last_base_trace.empty() &&
         snapshot.last_delta_interior.empty() &&
         snapshot.last_delta_trace.empty() &&
         snapshot.refinement_corrections.empty() &&
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
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int local_failure_stage = 0;
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
    std::vector<int> partition(result.serial->GetNE());
    for (int element = 0; element < result.serial->GetNE(); ++element)
      partition[static_cast<std::size_t>(element)] =
          element * ranks / result.serial->GetNE();
    result.mesh = std::make_unique<mfem::ParMesh>(
        MPI_COMM_WORLD, *result.serial, partition.data());
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
    local_failure_stage = 1;
  }
  if (local_failure_stage == 0 &&
      (!result.serial || !result.mesh || !result.interior_collection ||
       !result.trace_collection || !result.interior_space ||
       !result.trace_space))
  {
    local_failure_stage = 2;
  }
  std::vector<int> failure_stages(static_cast<std::size_t>(ranks), 0);
  MPI_Allgather(&local_failure_stage, 1, MPI_INT, failure_stages.data(), 1,
                MPI_INT, MPI_COMM_WORLD);
  const int maximum_failure_stage = *std::max_element(
      failure_stages.begin(), failure_stages.end());
  if (maximum_failure_stage != 0)
  {
    int minimum_failure_stage = std::numeric_limits<int>::max();
    for (const int stage : failure_stages)
      if (stage != 0)
        minimum_failure_stage = std::min(minimum_failure_stage, stage);
    if (rank == 0)
    {
      std::cerr << "HDG_EULER_DEVICE_FULL_ORACLE_MESH_COLLECTIVE_FAILURE"
                << " failure_stage_min=" << minimum_failure_stage
                << " failure_stage_max=" << maximum_failure_stage
                << " rank_stages=";
      for (int peer = 0; peer < ranks; ++peer)
        std::cerr << (peer == 0 ? "" : ",")
                  << failure_stages[static_cast<std::size_t>(peer)];
      std::cerr << '\n';
    }
    return false;
  }
  const int local_elements = result.mesh->GetNE();
  int global_elements = 0;
  MPI_Allreduce(&local_elements, &global_elements, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  return local_elements > 0 && global_elements == N * N &&
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
                    std::vector<double>& right_hand_sides, const int columns,
                    int* const failed_pivot = nullptr)
{
  if (failed_pivot != nullptr) { *failed_pivot = -1; }
  for (int pivot = 0; pivot < rows; ++pivot)
  {
    int best = pivot;
    for (int row = pivot + 1; row < rows; ++row)
      if (std::abs(matrix[row * rows + pivot]) >
          std::abs(matrix[best * rows + pivot]))
      {
        best = row;
      }
    if (!(std::abs(matrix[best * rows + pivot]) > 1.0e-14))
    {
      if (failed_pivot != nullptr) { *failed_pivot = pivot; }
      return false;
    }
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

struct IndependentLinearization
{
  int elements = 0;
  int ni = 0;
  int nt = 0;
  int ng = 0;
  int quadrature_order = 0;
  bool admissible = false;
  std::vector<double> residual_r;
  std::vector<double> residual_g;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> d;
  std::vector<double> f;
  std::vector<double> g;
};

bool AdmissibleIndependentState(
    const std::array<double, conserved_components>& state)
{
  if (!(state[0] >= 1.0e-8)) { return false; }
  const double pressure = (gamma - 1.0) *
      (state[3] - 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                      state[0]);
  return pressure >= 1.0e-8 && std::isfinite(pressure);
}

bool FiniteVector(const std::vector<double>& values)
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return std::isfinite(value); });
}

double MaximumDifference(const std::vector<double>& first,
                         const std::vector<double>& second)
{
  if (first.size() != second.size())
    return std::numeric_limits<double>::infinity();
  double error = 0.0;
  for (std::size_t entry = 0; entry < first.size(); ++entry)
    error = std::max(error, std::abs(first[entry] - second[entry]));
  return error;
}

bool IndependentAssembleLinearization(
    const FullDeviceTestSnapshot& snapshot,
    const std::vector<double>& interior_coefficients,
    const std::vector<double>& trace_coefficients,
    IndependentLinearization& result)
{
  OracleMesh oracle;
  if (!BuildOracleMesh(snapshot.polynomial_order,
                       snapshot.elements_per_direction, oracle))
  {
    return false;
  }
  result = {};
  result.elements = oracle.mesh->GetNE();
  result.ni = snapshot.interior_dofs;
  result.nt = snapshot.trace_dofs;
  result.ng = snapshot.global_trace_dofs;
  result.quadrature_order = 5 * snapshot.polynomial_order + 16;
  const std::size_t total_interior =
      static_cast<std::size_t>(result.elements) * result.ni;
  const std::size_t total_trace =
      static_cast<std::size_t>(result.elements) * result.nt;
  const std::size_t local_trace_size = trace_coefficients.size();
  if (result.elements != snapshot.elements ||
      result.ni <= 0 || result.nt <= 0 || result.ng <= 0 ||
      snapshot.assembly_quadrature_order != result.quadrature_order ||
      snapshot.old_interior.size() != total_interior ||
      interior_coefficients.size() != total_interior ||
      local_trace_size == 0 ||
      snapshot.signed_trace_dofs.size() != total_trace)
  {
    return false;
  }
  result.residual_r.assign(total_interior, 0.0);
  result.residual_g.assign(total_trace, 0.0);
  result.a.assign(total_interior * result.ni, 0.0);
  result.b.assign(total_interior * result.nt, 0.0);
  result.c.assign(total_trace * result.ni, 0.0);
  result.d.assign(total_trace * result.nt, 0.0);
  result.f.assign(total_interior, 0.0);
  result.g.assign(total_trace, 0.0);
  result.admissible = true;

  // Never assume that MFEM preserves the serial or production element order
  // after partitioning.  Match each independently rebuilt local element to
  // the production snapshot by its periodic physical center.
  constexpr double coordinate_scale = 1.0e12;
  using ElementKey = std::pair<std::int64_t, std::int64_t>;
  std::map<ElementKey, int> snapshot_element_for_key;
  if (snapshot.element_coordinate_keys.size() !=
      static_cast<std::size_t>(result.elements) * dimension)
  {
    return false;
  }
  for (int element = 0; element < result.elements; ++element)
  {
    const ElementKey key{
        snapshot.element_coordinate_keys[
            static_cast<std::size_t>(element) * dimension],
        snapshot.element_coordinate_keys[
            static_cast<std::size_t>(element) * dimension + 1]};
    if (!snapshot_element_for_key.emplace(key, element).second) { return false; }
  }

  for (int oracle_element = 0; oracle_element < result.elements;
       ++oracle_element)
  {
    const mfem::FiniteElement* fe =
        oracle.interior_space->GetFE(oracle_element);
    mfem::ElementTransformation* transformation =
        oracle.mesh->GetElementTransformation(oracle_element);
    if (fe == nullptr || transformation == nullptr ||
        conserved_components * fe->GetDof() != result.ni)
    {
      return false;
    }
    mfem::IntegrationPoint center;
    center.x = 0.5;
    center.y = 0.5;
    mfem::Vector center_physical(dimension);
    transformation->Transform(center, center_physical);
    auto coordinate_bin = [](const double coordinate)
    {
      double periodic = std::fmod(coordinate, 1.0);
      if (periodic < 0.0) { periodic += 1.0; }
      if (std::abs(periodic) <= 1.0e-11 ||
          std::abs(periodic - 1.0) <= 1.0e-11)
      {
        periodic = 0.0;
      }
      return static_cast<std::int64_t>(
          std::llround(periodic * coordinate_scale));
    };
    const ElementKey key{coordinate_bin(center_physical[0]),
                         coordinate_bin(center_physical[1])};
    const auto matched = snapshot_element_for_key.find(key);
    if (matched == snapshot_element_for_key.end()) { return false; }
    const int element = matched->second;
    const int scalar_dofs = fe->GetDof();
    const std::size_t interior_offset =
        static_cast<std::size_t>(element) * result.ni;
    const std::size_t trace_offset =
        static_cast<std::size_t>(element) * result.nt;
    const std::size_t a_offset = interior_offset * result.ni;
    const std::size_t b_offset = interior_offset * result.nt;
    const std::size_t c_offset = trace_offset * result.ni;
    const std::size_t d_offset = trace_offset * result.nt;
    mfem::Vector shape(scalar_dofs), physical(dimension);
    mfem::DenseMatrix derivative(scalar_dofs, dimension);
    const mfem::IntegrationRule& volume_rule =
        mfem::IntRules.Get(fe->GetGeomType(), result.quadrature_order);
    for (int q = 0; q < volume_rule.GetNPoints(); ++q)
    {
      const auto& point = volume_rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      fe->CalcPhysDShape(*transformation, derivative);
      std::array<double, conserved_components> state{}, old_state{};
      std::array<double, conserved_components> source{}, flux_x{}, flux_y{};
      std::array<double, conserved_components * conserved_components>
          jacobian_x{}, jacobian_y{};
      if (!Interpolate(interior_coefficients,
                       static_cast<int>(interior_offset), scalar_dofs, shape,
                       state) ||
          !Interpolate(snapshot.old_interior,
                       static_cast<int>(interior_offset), scalar_dofs, shape,
                       old_state) ||
          !ManufacturedSourceIndependent(physical[0], physical[1], source) ||
          !FluxJacobian(state, 1.0, 0.0, flux_x, jacobian_x) ||
          !FluxJacobian(state, 0.0, 1.0, flux_y, jacobian_y))
      {
        return false;
      }
      result.admissible = result.admissible &&
                          AdmissibleIndependentState(state);
      const double weight = point.weight * transformation->Weight();
      if (!(weight > 0.0) || !std::isfinite(weight)) { return false; }
      for (int row = 0; row < scalar_dofs; ++row)
        for (int row_component = 0;
             row_component < conserved_components; ++row_component)
        {
          const int local_row = ByNodes(row, scalar_dofs, row_component);
          result.residual_r[interior_offset + local_row] += weight *
              (shape[row] * (state[row_component] - old_state[row_component]) /
                   backward_euler_dt -
               derivative(row, 0) * flux_x[row_component] -
               derivative(row, 1) * flux_y[row_component] -
               shape[row] * source[row_component]);
          for (int column = 0; column < scalar_dofs; ++column)
            for (int column_component = 0;
                 column_component < conserved_components; ++column_component)
            {
              const int local_column =
                  ByNodes(column, scalar_dofs, column_component);
              const double mass = row_component == column_component ?
                  shape[row] * shape[column] / backward_euler_dt : 0.0;
              const double volume =
                  -(derivative(row, 0) *
                        jacobian_x[row_component * conserved_components +
                                   column_component] +
                    derivative(row, 1) *
                        jacobian_y[row_component * conserved_components +
                                   column_component]) *
                  shape[column];
              result.a[a_offset + static_cast<std::size_t>(local_row) *
                                      result.ni + local_column] +=
                  weight * (mass + volume);
            }
        }
    }

    mfem::Array<int> faces, orientations;
    oracle.mesh->GetElementEdges(oracle_element, faces, orientations);
    if (faces.Size() != 4) { return false; }
    int local_trace_offset = 0;
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
      const bool first = face_transformation->Elem1No == oracle_element;
      if (!first && face_transformation->Elem2No != oracle_element)
      {
        return false;
      }
      mfem::Vector interior_shape(scalar_dofs);
      mfem::Vector trace_shape(face_scalar_dofs);
      const mfem::IntegrationRule& face_rule =
          mfem::IntRules.Get(trace_fe->GetGeomType(), result.quadrature_order);
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
        if (!Interpolate(interior_coefficients,
                         static_cast<int>(interior_offset), scalar_dofs,
                         interior_shape, state))
        {
          return false;
        }
        for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
          for (int component = 0; component < conserved_components;
               ++component)
          {
            const int local = local_trace_offset +
                ByNodes(scalar, face_scalar_dofs, component);
            double sign = 1.0;
            const int global = Decode(
                snapshot.signed_trace_dofs[trace_offset + local], sign);
            if (global < 0 ||
                static_cast<std::size_t>(global) >= local_trace_size)
            {
              return false;
            }
            trace_state[component] +=
                trace_shape[scalar] * sign * trace_coefficients[global];
          }
        result.admissible = result.admissible &&
                            AdmissibleIndependentState(state) &&
                            AdmissibleIndependentState(trace_state);
        mfem::Vector scaled_normal(dimension);
        mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled_normal);
        const double measure = scaled_normal.Norml2();
        if (!(measure > 0.0) || !std::isfinite(measure)) { return false; }
        const double side = first ? 1.0 : -1.0;
        const double nx = side * scaled_normal[0] / measure;
        const double ny = side * scaled_normal[1] / measure;
        const double weight = point.weight * measure;
        std::array<double, conserved_components> trace_flux{};
        std::array<double, conserved_components * conserved_components>
            trace_jacobian{};
        if (!FluxJacobian(trace_state, nx, ny, trace_flux,
                          trace_jacobian) ||
            !(weight > 0.0) || !std::isfinite(weight))
        {
          return false;
        }
        for (int row = 0; row < scalar_dofs; ++row)
        {
          for (int component = 0; component < conserved_components;
               ++component)
          {
            const int local_row = ByNodes(row, scalar_dofs, component);
            result.residual_r[interior_offset + local_row] +=
                weight * interior_shape[row] *
                (trace_flux[component] + fixed_scalar_tau *
                    (state[component] - trace_state[component]));
          }
          for (int column = 0; column < scalar_dofs; ++column)
            for (int component = 0; component < conserved_components;
                 ++component)
            {
              const int local_row = ByNodes(row, scalar_dofs, component);
              const int local_column =
                  ByNodes(column, scalar_dofs, component);
              result.a[a_offset + static_cast<std::size_t>(local_row) *
                                      result.ni + local_column] +=
                  weight * interior_shape[row] * fixed_scalar_tau *
                  interior_shape[column];
            }
          for (int column = 0; column < face_scalar_dofs; ++column)
            for (int row_component = 0;
                 row_component < conserved_components; ++row_component)
              for (int column_component = 0;
                   column_component < conserved_components;
                   ++column_component)
              {
                const int local_row =
                    ByNodes(row, scalar_dofs, row_component);
                const int local_column = local_trace_offset +
                    ByNodes(column, face_scalar_dofs, column_component);
                result.b[b_offset + static_cast<std::size_t>(local_row) *
                                        result.nt + local_column] +=
                    weight * interior_shape[row] * trace_shape[column] *
                    (trace_jacobian[row_component * conserved_components +
                                    column_component] -
                     (row_component == column_component ? fixed_scalar_tau :
                                                          0.0));
              }
        }
        for (int row = 0; row < face_scalar_dofs; ++row)
        {
          for (int component = 0; component < conserved_components;
               ++component)
          {
            const int local_row = local_trace_offset +
                ByNodes(row, face_scalar_dofs, component);
            result.residual_g[trace_offset + local_row] +=
                weight * trace_shape[row] *
                (trace_flux[component] + fixed_scalar_tau *
                    (state[component] - trace_state[component]));
          }
          for (int column = 0; column < scalar_dofs; ++column)
            for (int component = 0; component < conserved_components;
                 ++component)
            {
              const int local_row = local_trace_offset +
                  ByNodes(row, face_scalar_dofs, component);
              const int local_column =
                  ByNodes(column, scalar_dofs, component);
              result.c[c_offset + static_cast<std::size_t>(local_row) *
                                      result.ni + local_column] +=
                  weight * trace_shape[row] * fixed_scalar_tau *
                  interior_shape[column];
            }
          for (int column = 0; column < face_scalar_dofs; ++column)
            for (int row_component = 0;
                 row_component < conserved_components; ++row_component)
              for (int column_component = 0;
                   column_component < conserved_components;
                   ++column_component)
              {
                const int local_row = local_trace_offset +
                    ByNodes(row, face_scalar_dofs, row_component);
                const int local_column = local_trace_offset +
                    ByNodes(column, face_scalar_dofs, column_component);
                result.d[d_offset + static_cast<std::size_t>(local_row) *
                                        result.nt + local_column] +=
                    weight * trace_shape[row] * trace_shape[column] *
                    (trace_jacobian[row_component * conserved_components +
                                    column_component] -
                     (row_component == column_component ? fixed_scalar_tau :
                                                          0.0));
              }
        }
      }
      local_trace_offset += face_vdofs;
    }
    if (local_trace_offset != result.nt) { return false; }
    for (int row = 0; row < result.ni; ++row)
      result.f[interior_offset + row] =
          -result.residual_r[interior_offset + row];
    for (int row = 0; row < result.nt; ++row)
      result.g[trace_offset + row] = -result.residual_g[trace_offset + row];
  }
  return result.admissible && FiniteVector(result.residual_r) &&
         FiniteVector(result.residual_g) && FiniteVector(result.a) &&
         FiniteVector(result.b) && FiniteVector(result.c) &&
         FiniteVector(result.d) && FiniteVector(result.f) &&
         FiniteVector(result.g);
}

struct IndependentCondensedSystem
{
  std::vector<double> schur;
  std::vector<double> condensed_rhs;
  std::vector<double> true_matrix;
  std::vector<double> true_rhs;
  std::vector<double> trace_increment;
  std::vector<double> local_trace_increment;
  std::vector<double> recovered_increment;
};

enum class IndependentCondenseFailureStage : int
{
  none = 0,
  invalid_dimensions = 1,
  invalid_storage = 2,
  invalid_global_map = 3,
  injected_preparation = 4,
  element_factorization = 5,
  decoded_global_row = 6,
  decoded_global_column = 7,
  preparation_exception = 8,
  collective_signature = 9,
  global_factorization = 10,
  recovery_map = 11,
  recovery_factorization = 12,
  nonfinite_result = 13,
  post_reduction_exception = 14
};

struct IndependentCondenseDiagnostic
{
  int local_failure_stage = 0;
  int local_element = -1;
  int local_index = -1;
  int local_decoded_value = -1;
  int local_pivot = -1;
  int minimum_failure_stage = 0;
  int maximum_failure_stage = 0;
};

bool CondenseCollectiveReady(
    const IndependentCondenseDiagnostic& local, const int ni, const int nt,
    const int ng, const bool distributed_reduction, const char* const phase,
    IndependentCondenseDiagnostic* const diagnostic)
{
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::array<int, 9> local_values{{
      local.local_failure_stage, local.local_element, local.local_index,
      local.local_decoded_value, local.local_pivot, ni, nt, ng,
      distributed_reduction ? 1 : 0}};
  std::vector<int> gathered(static_cast<std::size_t>(ranks) *
                            local_values.size());
  MPI_Allgather(local_values.data(), static_cast<int>(local_values.size()),
                MPI_INT, gathered.data(), static_cast<int>(local_values.size()),
                MPI_INT, MPI_COMM_WORLD);

  int minimum_failure_stage = std::numeric_limits<int>::max();
  int maximum_failure_stage = 0;
  for (int peer = 0; peer < ranks; ++peer)
  {
    const int stage = gathered[static_cast<std::size_t>(peer) *
                               local_values.size()];
    if (stage != 0)
    {
      minimum_failure_stage = std::min(minimum_failure_stage, stage);
      maximum_failure_stage = std::max(maximum_failure_stage, stage);
    }
  }
  bool signature_matches = true;
  if (maximum_failure_stage == 0)
  {
    for (int peer = 1; peer < ranks; ++peer)
      for (std::size_t entry = 5; entry < local_values.size(); ++entry)
        signature_matches = signature_matches &&
            gathered[static_cast<std::size_t>(peer) * local_values.size() +
                     entry] == gathered[entry];
    if (!signature_matches)
    {
      minimum_failure_stage = static_cast<int>(
          IndependentCondenseFailureStage::collective_signature);
      maximum_failure_stage = minimum_failure_stage;
    }
  }
  const bool ready = maximum_failure_stage == 0 && signature_matches;
  if (diagnostic != nullptr)
  {
    *diagnostic = local;
    diagnostic->minimum_failure_stage = ready ? 0 : minimum_failure_stage;
    diagnostic->maximum_failure_stage = ready ? 0 : maximum_failure_stage;
  }
  if (!ready && rank == 0)
  {
    std::cerr << "HDG_EULER_DEVICE_FULL_MPI_ORACLE_CONDENSE_FAILURE"
              << " phase=" << phase
              << " failure_stage_min=" << minimum_failure_stage
              << " failure_stage_max=" << maximum_failure_stage
              << " rank_details=";
    for (int peer = 0; peer < ranks; ++peer)
    {
      const std::size_t offset =
          static_cast<std::size_t>(peer) * local_values.size();
      std::cerr << (peer == 0 ? "" : ";") << peer << ":"
                << gathered[offset] << "/e" << gathered[offset + 1]
                << "/i" << gathered[offset + 2] << "/v"
                << gathered[offset + 3] << "/p" << gathered[offset + 4]
                << "/shape" << gathered[offset + 5] << ","
                << gathered[offset + 6] << "," << gathered[offset + 7]
                << "," << gathered[offset + 8];
    }
    std::cerr << '\n';
  }
  return ready;
}

bool IndependentCondenseSolve(const FullDeviceTestSnapshot& snapshot,
                              const IndependentLinearization& linearization,
                              IndependentCondensedSystem& result,
                              const int injected_preparation_failure_rank = -1,
                              IndependentCondenseDiagnostic* const diagnostic =
                                  nullptr)
{
  const int elements = linearization.elements;
  const int ni = linearization.ni;
  const int nt = linearization.nt;
  const int ng = linearization.ng;
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  IndependentCondenseDiagnostic local;
  const auto fail = [&local](const IndependentCondenseFailureStage stage,
                             const int element, const int index,
                             const int decoded_value, const int pivot)
  {
    if (local.local_failure_stage != 0) { return; }
    local.local_failure_stage = static_cast<int>(stage);
    local.local_element = element;
    local.local_index = index;
    local.local_decoded_value = decoded_value;
    local.local_pivot = pivot;
  };
  if (elements <= 0 || ni <= 0 || nt <= 0 || ng <= 0 ||
      static_cast<long long>(ng) * ng > std::numeric_limits<int>::max())
  {
    fail(IndependentCondenseFailureStage::invalid_dimensions, -1, -1, -1,
         -1);
  }
  if (rank == injected_preparation_failure_rank)
  {
    fail(IndependentCondenseFailureStage::injected_preparation, -1, -1,
         rank, -1);
  }
  result = {};
  const std::vector<int>& global_map =
      snapshot.signed_global_trace_dofs.empty() ?
          snapshot.signed_trace_dofs : snapshot.signed_global_trace_dofs;
  const bool distributed_reduction =
      snapshot.signed_global_trace_dofs.size() == global_map.size();
  std::vector<double> global_matrix;
  std::vector<double> global_rhs;
  try
  {
    if (local.local_failure_stage == 0)
    {
      const std::size_t local_interior =
          static_cast<std::size_t>(elements) * ni;
      const std::size_t local_trace =
          static_cast<std::size_t>(elements) * nt;
      const bool storage_valid =
          linearization.a.size() == local_interior * ni &&
          linearization.b.size() == local_interior * nt &&
          linearization.c.size() == local_trace * ni &&
          linearization.d.size() == local_trace * nt &&
          linearization.f.size() == local_interior &&
          linearization.g.size() == local_trace;
      if (!storage_valid)
        fail(IndependentCondenseFailureStage::invalid_storage, -1, -1, -1,
             -1);
      else if (global_map.size() != local_trace)
        fail(IndependentCondenseFailureStage::invalid_global_map, -1,
             static_cast<int>(global_map.size()),
             static_cast<int>(local_trace), -1);
    }
    if (local.local_failure_stage == 0)
    {
      result.schur.assign(static_cast<std::size_t>(elements) * nt * nt, 0.0);
      result.condensed_rhs.assign(static_cast<std::size_t>(elements) * nt,
                                  0.0);
      result.true_matrix.assign(static_cast<std::size_t>(ng) * ng, 0.0);
      result.true_rhs.assign(ng, 0.0);
      result.local_trace_increment.assign(
          static_cast<std::size_t>(elements) * nt, 0.0);
      result.recovered_increment.assign(
          static_cast<std::size_t>(elements) * ni, 0.0);
      if (distributed_reduction)
      {
        global_matrix.assign(result.true_matrix.size(), 0.0);
        global_rhs.assign(result.true_rhs.size(), 0.0);
      }
    }
    for (int element = 0;
         element < elements && local.local_failure_stage == 0; ++element)
    {
      const double* a = linearization.a.data() +
                        static_cast<std::size_t>(element) * ni * ni;
      const double* b = linearization.b.data() +
                        static_cast<std::size_t>(element) * ni * nt;
      const double* c = linearization.c.data() +
                        static_cast<std::size_t>(element) * nt * ni;
      const double* d = linearization.d.data() +
                        static_cast<std::size_t>(element) * nt * nt;
      const double* f = linearization.f.data() +
                        static_cast<std::size_t>(element) * ni;
      const double* g = linearization.g.data() +
                        static_cast<std::size_t>(element) * nt;
      std::vector<double> solved(static_cast<std::size_t>(ni) * (nt + 1));
      for (int row = 0; row < ni; ++row)
      {
        std::copy_n(b + row * nt, nt, solved.data() + row * (nt + 1));
        solved[row * (nt + 1) + nt] = f[row];
      }
      int failed_pivot = -1;
      if (!FactorAndSolve(std::vector<double>(a, a + ni * ni), ni, solved,
                          nt + 1, &failed_pivot))
      {
        fail(IndependentCondenseFailureStage::element_factorization, element,
             -1, -1, failed_pivot);
        break;
      }
      for (int row = 0; row < nt && local.local_failure_stage == 0; ++row)
      {
        const std::size_t local_row =
            static_cast<std::size_t>(element) * nt + row;
        double local_rhs = g[row];
        for (int interior = 0; interior < ni; ++interior)
          local_rhs -= c[row * ni + interior] *
                       solved[interior * (nt + 1) + nt];
        result.condensed_rhs[local_row] = local_rhs;
        double row_sign = 1.0;
        const int global_row = Decode(global_map[local_row], row_sign);
        if (global_row < 0 || global_row >= ng)
        {
          fail(IndependentCondenseFailureStage::decoded_global_row, element,
               static_cast<int>(local_row), global_row, -1);
          break;
        }
        result.true_rhs[global_row] += row_sign * local_rhs;
        for (int column = 0; column < nt; ++column)
        {
          double value = d[row * nt + column];
          for (int interior = 0; interior < ni; ++interior)
            value -= c[row * ni + interior] *
                     solved[interior * (nt + 1) + column];
          result.schur[(static_cast<std::size_t>(element) * nt + row) * nt +
                       column] = value;
          double column_sign = 1.0;
          const std::size_t local_column =
              static_cast<std::size_t>(element) * nt + column;
          const int global_column =
              Decode(global_map[local_column], column_sign);
          if (global_column < 0 || global_column >= ng)
          {
            fail(IndependentCondenseFailureStage::decoded_global_column,
                 element, static_cast<int>(local_column), global_column, -1);
            break;
          }
          result.true_matrix[static_cast<std::size_t>(global_row) * ng +
                             global_column] +=
              row_sign * column_sign * value;
        }
      }
    }
  }
  catch (const std::exception&)
  {
    fail(IndependentCondenseFailureStage::preparation_exception, -1, -1, -1,
         -1);
  }

  if (!CondenseCollectiveReady(local, ni, nt, ng, distributed_reduction,
                               "prepare", diagnostic))
    return false;
  if (distributed_reduction)
  {
    MPI_Allreduce(result.true_matrix.data(), global_matrix.data(),
                  static_cast<int>(result.true_matrix.size()), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(result.true_rhs.data(), global_rhs.data(),
                  static_cast<int>(result.true_rhs.size()), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    result.true_matrix = std::move(global_matrix);
    result.true_rhs = std::move(global_rhs);
  }
  try
  {
    result.trace_increment = result.true_rhs;
    int failed_pivot = -1;
    if (!FactorAndSolve(result.true_matrix, ng, result.trace_increment, 1,
                        &failed_pivot))
    {
      fail(IndependentCondenseFailureStage::global_factorization, -1, -1, -1,
           failed_pivot);
    }
    for (int element = 0;
         element < elements && local.local_failure_stage == 0; ++element)
    {
      const double* a = linearization.a.data() +
                        static_cast<std::size_t>(element) * ni * ni;
      const double* b = linearization.b.data() +
                        static_cast<std::size_t>(element) * ni * nt;
      const double* f = linearization.f.data() +
                        static_cast<std::size_t>(element) * ni;
      std::vector<double> recovered(ni);
      for (int column = 0; column < nt; ++column)
      {
        double sign = 1.0;
        const std::size_t local_column =
            static_cast<std::size_t>(element) * nt + column;
        const int global = Decode(global_map[local_column], sign);
        if (global < 0 || global >= ng)
        {
          fail(IndependentCondenseFailureStage::recovery_map, element,
               static_cast<int>(local_column), global, -1);
          break;
        }
        result.local_trace_increment[local_column] =
            sign * result.trace_increment[global];
      }
      if (local.local_failure_stage != 0) { break; }
      for (int row = 0; row < ni; ++row)
      {
        recovered[row] = f[row];
        for (int column = 0; column < nt; ++column)
          recovered[row] -=
              b[row * nt + column] *
              result.local_trace_increment[
                  static_cast<std::size_t>(element) * nt + column];
      }
      failed_pivot = -1;
      if (!FactorAndSolve(std::vector<double>(a, a + ni * ni), ni, recovered,
                          1, &failed_pivot))
      {
        fail(IndependentCondenseFailureStage::recovery_factorization, element,
             -1, -1, failed_pivot);
        break;
      }
      std::copy(recovered.begin(), recovered.end(),
                result.recovered_increment.begin() +
                    static_cast<std::size_t>(element) * ni);
    }
    if (local.local_failure_stage == 0 &&
        !(FiniteVector(result.schur) &&
          FiniteVector(result.condensed_rhs) &&
          FiniteVector(result.true_matrix) && FiniteVector(result.true_rhs) &&
          FiniteVector(result.trace_increment) &&
          FiniteVector(result.local_trace_increment) &&
          FiniteVector(result.recovered_increment)))
      fail(IndependentCondenseFailureStage::nonfinite_result, -1, -1, -1,
           -1);
  }
  catch (const std::exception&)
  {
    fail(IndependentCondenseFailureStage::post_reduction_exception, -1, -1,
         -1, -1);
  }
  return CondenseCollectiveReady(local, ni, nt, ng, distributed_reduction,
                                 "post_reduction", diagnostic);
}

struct GlobalLinearOracleResult
{
  double residual_r_error = std::numeric_limits<double>::infinity();
  double residual_g_error = std::numeric_limits<double>::infinity();
  double a_error = std::numeric_limits<double>::infinity();
  double b_error = std::numeric_limits<double>::infinity();
  double c_error = std::numeric_limits<double>::infinity();
  double d_error = std::numeric_limits<double>::infinity();
  double f_error = std::numeric_limits<double>::infinity();
  double g_error = std::numeric_limits<double>::infinity();
  double condensation_error = std::numeric_limits<double>::infinity();
  double rhs_error = std::numeric_limits<double>::infinity();
  double true_matrix_error = std::numeric_limits<double>::infinity();
  double true_rhs_error = std::numeric_limits<double>::infinity();
  double action_error = std::numeric_limits<double>::infinity();
  double trace_error = std::numeric_limits<double>::infinity();
  double recovery_error = std::numeric_limits<double>::infinity();
  double last_trace_error = std::numeric_limits<double>::infinity();
  double last_recovery_error = std::numeric_limits<double>::infinity();
  double state_error = std::numeric_limits<double>::infinity();
  double final_residual = std::numeric_limits<double>::infinity();
  double final_residual_error = std::numeric_limits<double>::infinity();
  bool admissible = false;
  bool generation_matches = false;
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
  const std::size_t total_interior =
      static_cast<std::size_t>(elements) * ni;
  const std::size_t total_trace = static_cast<std::size_t>(elements) * nt;
  if (!snapshot.global_linear_oracle_staged || elements != 9 || ni <= 0 ||
      nt <= 0 || ng <= 0 ||
      snapshot.signed_trace_dofs.size() != total_trace ||
      !sized(snapshot.global_residual_r, total_interior) ||
      !sized(snapshot.global_residual_g, total_trace) ||
      !sized(snapshot.global_a, total_interior * ni) ||
      !sized(snapshot.global_b, total_interior * nt) ||
      !sized(snapshot.global_c, total_trace * ni) ||
      !sized(snapshot.global_d, total_trace * nt) ||
      !sized(snapshot.global_f, total_interior) ||
      !sized(snapshot.global_g, total_trace) ||
      !sized(snapshot.global_schur, total_trace * nt) ||
      !sized(snapshot.global_condensed_rhs, total_trace) ||
      !sized(snapshot.global_trace_increment, ng) ||
      !sized(snapshot.global_local_trace_increment, total_trace) ||
      !sized(snapshot.global_recovered_increment, total_interior) ||
      snapshot.actual_true_csr_offsets.size() !=
          static_cast<std::size_t>(ng + 1) ||
      snapshot.actual_true_csr_columns.size() !=
          snapshot.actual_true_csr_values.size() ||
      !sized(snapshot.actual_true_rhs, ng) ||
      !sized(snapshot.last_base_interior, total_interior) ||
      !sized(snapshot.last_delta_interior, total_interior) ||
      !sized(snapshot.last_base_trace, ng) ||
      !sized(snapshot.last_delta_trace, ng) ||
      !sized(snapshot.final_interior, total_interior) ||
      !sized(snapshot.final_trace, ng) ||
      !(snapshot.last_accepted_alpha > 0.0))
  {
    return false;
  }

  IndependentLinearization first_linearization;
  IndependentCondensedSystem first_system;
  if (!IndependentAssembleLinearization(
          snapshot, snapshot.initial_interior, snapshot.initial_trace,
          first_linearization) ||
      !IndependentCondenseSolve(snapshot, first_linearization, first_system))
  {
    return false;
  }
  result.residual_r_error = MaximumDifference(
      first_linearization.residual_r, snapshot.global_residual_r);
  result.residual_g_error = MaximumDifference(
      first_linearization.residual_g, snapshot.global_residual_g);
  result.a_error = MaximumDifference(first_linearization.a, snapshot.global_a);
  result.b_error = MaximumDifference(first_linearization.b, snapshot.global_b);
  result.c_error = MaximumDifference(first_linearization.c, snapshot.global_c);
  result.d_error = MaximumDifference(first_linearization.d, snapshot.global_d);
  result.f_error = MaximumDifference(first_linearization.f, snapshot.global_f);
  result.g_error = MaximumDifference(first_linearization.g, snapshot.global_g);
  result.condensation_error =
      MaximumDifference(first_system.schur, snapshot.global_schur);
  result.rhs_error = MaximumDifference(
      first_system.condensed_rhs, snapshot.global_condensed_rhs);

  std::vector<double> actual_matrix(static_cast<std::size_t>(ng) * ng, 0.0);
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
  }
  result.true_matrix_error =
      MaximumDifference(actual_matrix, first_system.true_matrix);
  result.true_rhs_error =
      MaximumDifference(snapshot.actual_true_rhs, first_system.true_rhs);
  result.trace_error = MaximumDifference(
      first_system.trace_increment, snapshot.global_trace_increment);
  result.trace_error = std::max(
      result.trace_error,
      MaximumDifference(first_system.local_trace_increment,
                        snapshot.global_local_trace_increment));
  result.recovery_error = MaximumDifference(
      first_system.recovered_increment, snapshot.global_recovered_increment);
  long double rhs_norm_squared = 0.0L;
  long double action_norm_squared = 0.0L;
  for (int row = 0; row < ng; ++row)
  {
    double action = -first_system.true_rhs[row];
    for (int column = 0; column < ng; ++column)
      action += first_system.true_matrix[
                    static_cast<std::size_t>(row) * ng + column] *
                snapshot.global_trace_increment[column];
    rhs_norm_squared +=
        static_cast<long double>(first_system.true_rhs[row]) *
        first_system.true_rhs[row];
    action_norm_squared += static_cast<long double>(action) * action;
  }
  result.action_error = std::sqrt(static_cast<double>(action_norm_squared)) /
      std::max(std::sqrt(static_cast<double>(rhs_norm_squared)), 1.0e-300);

  IndependentLinearization last_linearization;
  IndependentCondensedSystem last_system;
  if (!IndependentAssembleLinearization(
          snapshot, snapshot.last_base_interior, snapshot.last_base_trace,
          last_linearization) ||
      !IndependentCondenseSolve(snapshot, last_linearization, last_system))
  {
    return false;
  }
  result.last_trace_error = MaximumDifference(
      last_system.trace_increment, snapshot.last_delta_trace);
  result.last_recovery_error = MaximumDifference(
      last_system.recovered_increment, snapshot.last_delta_interior);
  result.state_error = 0.0;
  for (std::size_t entry = 0; entry < snapshot.final_interior.size(); ++entry)
    result.state_error = std::max(
        result.state_error,
        std::abs(snapshot.final_interior[entry] -
                 (snapshot.last_base_interior[entry] +
                  snapshot.last_accepted_alpha *
                      last_system.recovered_increment[entry])));
  for (std::size_t entry = 0; entry < snapshot.final_trace.size(); ++entry)
    result.state_error = std::max(
        result.state_error,
        std::abs(snapshot.final_trace[entry] -
                 (snapshot.last_base_trace[entry] +
                  snapshot.last_accepted_alpha *
                      last_system.trace_increment[entry])));

  bool independent_success = false;
  if (!IndependentFinalPdeResidual(evidence, result.final_residual,
                                   result.final_residual_error,
                                   independent_success))
  {
    return false;
  }
  result.admissible = first_linearization.admissible &&
                      last_linearization.admissible && independent_success;
  result.generation_matches = evidence.nonlinear_iterations >= 2 &&
      evidence.committed_generation ==
          static_cast<std::uint64_t>(evidence.nonlinear_iterations);
  const double target = 1.0e-10 * (1.0 + evidence.initial_residual);
  result.status_matches = result.admissible && result.generation_matches &&
                          evidence.status == FullDeviceStatus::success &&
                          evidence.pass && result.final_residual <= target;
  return std::isfinite(result.residual_r_error) &&
         std::isfinite(result.residual_g_error) &&
         std::isfinite(result.a_error) && std::isfinite(result.b_error) &&
         std::isfinite(result.c_error) && std::isfinite(result.d_error) &&
         std::isfinite(result.f_error) && std::isfinite(result.g_error) &&
         std::isfinite(result.condensation_error) &&
         std::isfinite(result.rhs_error) &&
         std::isfinite(result.true_matrix_error) &&
         std::isfinite(result.true_rhs_error) &&
         std::isfinite(result.action_error) &&
         std::isfinite(result.trace_error) &&
         std::isfinite(result.recovery_error) &&
         std::isfinite(result.last_trace_error) &&
         std::isfinite(result.last_recovery_error) &&
         std::isfinite(result.state_error) &&
         std::isfinite(result.final_residual_error);
}

bool GlobalOracleAccepted(const GlobalLinearOracleResult& result)
{
  return result.status_matches && result.admissible &&
         result.generation_matches &&
         result.residual_r_error <= 5.0e-11 &&
         result.residual_g_error <= 5.0e-11 &&
         result.a_error <= 5.0e-11 && result.b_error <= 5.0e-11 &&
         result.c_error <= 5.0e-11 && result.d_error <= 5.0e-11 &&
         result.f_error <= 5.0e-11 && result.g_error <= 5.0e-11 &&
         result.condensation_error <= 5.0e-11 &&
         result.rhs_error <= 5.0e-11 &&
         result.true_matrix_error <= 5.0e-11 &&
         result.true_rhs_error <= 5.0e-11 &&
         result.action_error <= 1.0e-10 &&
         result.trace_error <= 5.0e-9 &&
         result.recovery_error <= 5.0e-9 &&
         result.last_trace_error <= 5.0e-9 &&
         result.last_recovery_error <= 5.0e-9 &&
         result.state_error <= 5.0e-9 &&
         result.final_residual_error <= 5.0e-9;
}

void PrintGlobalOracleFailure(const int p,
                              const GlobalLinearOracleResult& result)
{
  std::cerr << "HDG_EULER_DEVICE_FULL_GLOBAL_ORACLE_FAILURE"
            << " p=" << p
            << " R/G=" << result.residual_r_error << "/"
            << result.residual_g_error
            << " A/B/C/D=" << result.a_error << "/" << result.b_error
            << "/" << result.c_error << "/" << result.d_error
            << " f/g=" << result.f_error << "/" << result.g_error
            << " condensed=" << result.condensation_error << "/"
            << result.rhs_error
            << " true=" << result.true_matrix_error << "/"
            << result.true_rhs_error
            << " action=" << result.action_error
            << " trace/recovery=" << result.trace_error << "/"
            << result.recovery_error
            << " last=" << result.last_trace_error << "/"
            << result.last_recovery_error
            << " state=" << result.state_error
            << " final=" << result.final_residual << "/"
            << result.final_residual_error
            << " admissible=" << result.admissible
            << " generation=" << result.generation_matches
            << " status=" << result.status_matches << '\n';
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
  return std::abs(*std::max_element(
             sum.begin(), sum.end(),
             [](const double first, const double second)
             { return std::abs(first) < std::abs(second); })) <=
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
      value.effective_hypre_ilu_local_reordering != 1 ||
      value.trace_solve_failure_code != 0 || value.last_hypre_error != 0 ||
      value.last_hypre_converged != 1 || value.last_hypre_iterations <= 0 ||
      !std::isfinite(value.last_hypre_reported_residual) ||
      !value.test_snapshot.refinement_corrections.empty() ||
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
              << " discarded_nonlinear="
              << value.discarded_nonlinear_iterations
              << " linear=" << value.linear_iterations
              << " discarded_linear=" << value.discarded_linear_iterations
              << " rejected_trials=" << value.rejected_trials
              << " discarded_residual="
              << value.discarded_accepted_residual
              << " refinement_records="
              << value.test_snapshot.refinement_corrections.size()
              << " solve_diag=" << value.trace_solve_failure_code
              << "/" << value.last_hypre_error
              << "/" << value.last_hypre_converged
              << "/" << value.last_hypre_iterations
              << "/" << value.last_hypre_krylov_dimension
              << "/" << value.last_hypre_max_iterations
              << "/" << value.last_hypre_global_rows
              << "/" << value.last_hypre_reported_residual
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

bool ValidateRollbackEvidence(const FullDeviceEvidence& rollback,
                              const int polynomial_order)
{
  if (rollback.status != FullDeviceStatus::injected_failure ||
      rollback.pass || rollback.cases.size() != 1)
  {
    return false;
  }
  const auto& rolled = rollback.cases.front();
  const auto& snapshot = rolled.test_snapshot;
  if (rolled.polynomial_order != polynomial_order ||
      rolled.elements_per_direction != 3 || rolled.pass ||
      rolled.status != FullDeviceStatus::injected_failure)
  {
    return false;
  }
  const bool interior_changed = !BitwiseEqual(
      snapshot.rollback_entry_interior, snapshot.rollback_changed_interior);
  const bool trace_changed = !BitwiseEqual(
      snapshot.rollback_entry_trace, snapshot.rollback_changed_trace);
  const bool nonfinite_changed = std::any_of(
      snapshot.rollback_changed_interior.begin(),
      snapshot.rollback_changed_interior.end(),
      [](const double value) { return !std::isfinite(value); });
  const auto& failed_phase =
      rolled.phases[static_cast<std::size_t>(FullDevicePhase::state_update)];
  return rolled.rollback_restored &&
         rolled.rollback_had_uncommitted_change &&
         rolled.rollback_nonfinite_change_observed && nonfinite_changed &&
         rolled.rollback_rejected_post_failure_consumption &&
         rolled.invalidated_product_count == 5 &&
         rolled.rejected_consumer_count == 6 &&
         rolled.nonlinear_iterations == 0 && rolled.linear_iterations == 0 &&
         rolled.committed_generation == 0 &&
         rolled.discarded_nonlinear_iterations == 1 &&
         rolled.discarded_linear_iterations > 0 &&
         rolled.discarded_attempt_generation > 0 &&
         rolled.final_residual == rolled.initial_residual &&
         std::isinf(rolled.true_relative_residual) &&
         rolled.discarded_accepted_residual >
             1.0e-10 * (1.0 + rolled.initial_residual) &&
         interior_changed && trace_changed &&
         BitwiseEqual(snapshot.rollback_entry_interior,
                      snapshot.rollback_restored_interior) &&
         BitwiseEqual(snapshot.rollback_entry_trace,
                      snapshot.rollback_restored_trace) &&
         DerivedSnapshotEmpty(snapshot) && failed_phase.reached &&
         failed_phase.status == FullDeviceStatus::injected_failure &&
         failed_phase.contemporaneous_audit && !failed_phase.device_resident &&
         failed_phase.generation > 0 && rolled.phase_audit_append_only &&
         rolled.failure_phase_audit_valid &&
         rolled.failure_phase == FullDevicePhase::state_update &&
         rolled.failure_phase_status == FullDeviceStatus::injected_failure &&
         rolled.failure_phase_generation ==
             rolled.discarded_attempt_generation &&
         !rolled.phase_audit_log.empty() &&
         rolled.phase_audit_log.back().phase ==
             FullDevicePhase::state_update &&
         rolled.phase_audit_log.back().status ==
             FullDeviceStatus::injected_failure &&
         rolled.phase_audit_log.back().contemporaneous_audit &&
         !rolled.phase_audit_log.back().device_resident &&
         std::all_of(rolled.phase_audit_log.begin(),
                     rolled.phase_audit_log.end() - 1,
                     [](const auto& record)
                     {
                       return record.reached &&
                              record.contemporaneous_audit &&
                              record.device_resident &&
                              record.status == FullDeviceStatus::success;
                     }) &&
         rolled.phases[static_cast<std::size_t>(FullDevicePhase::trace_solve)]
             .reached &&
         rolled.phases[static_cast<std::size_t>(FullDevicePhase::recovery)]
             .reached &&
         rolled.managed_residency_verified &&
         rolled.allocation_observability_reported &&
         rolled.app_forall_fallback_census_complete &&
         !rolled.fallback_census_complete &&
         rolled.telemetry.managed_device_objects_checked > 0 &&
         rolled.telemetry.managed_device_objects_checked ==
             rolled.telemetry.managed_device_objects_valid &&
         rolled.telemetry.phase_residency_audit_events > 0 &&
         rolled.telemetry.synchronization_events > 0 &&
         rolled.telemetry.fallback_events == 0 &&
         rolled.telemetry.dense_host_trace_inverse_events == 0 &&
         rolled.telemetry.solver_substitution_events == 0 &&
         std::none_of(rollback.convergence.begin(), rollback.convergence.end(),
                      [](const auto& item) { return item.pass; });
}

struct MpiDeviceOracleResult
{
  double residual_error = std::numeric_limits<double>::infinity();
  double jacobian_error = std::numeric_limits<double>::infinity();
  double condensation_error = std::numeric_limits<double>::infinity();
  double true_matrix_error = std::numeric_limits<double>::infinity();
  double true_rhs_error = std::numeric_limits<double>::infinity();
  double trace_solution_error = std::numeric_limits<double>::infinity();
  double recovery_error = std::numeric_limits<double>::infinity();
  double state_error = std::numeric_limits<double>::infinity();
  bool coordinate_coverage = false;
  bool ghost_truth = false;
  bool offdiag_action = false;
  bool status_truth = false;
  bool pass = false;
  int condense_local_failure_stage = 0;
  int condense_failure_stage_min = 0;
  int condense_failure_stage_max = 0;
};

bool CoordinateCoverage(const FullDeviceTestSnapshot& snapshot)
{
  const int local_key_count = static_cast<int>(
      snapshot.element_coordinate_keys.size());
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  std::vector<int> counts(static_cast<std::size_t>(ranks), 0);
  MPI_Allgather(&local_key_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  std::vector<int> displacements(static_cast<std::size_t>(ranks), 0);
  int global_key_count = 0;
  for (int rank = 0; rank < ranks; ++rank)
  {
    displacements[static_cast<std::size_t>(rank)] = global_key_count;
    global_key_count += counts[static_cast<std::size_t>(rank)];
  }
  std::vector<std::int64_t> global_keys(
      static_cast<std::size_t>(global_key_count));
  MPI_Allgatherv(snapshot.element_coordinate_keys.data(), local_key_count,
                 MPI_INT64_T, global_keys.data(), counts.data(),
                 displacements.data(), MPI_INT64_T, MPI_COMM_WORLD);
  bool local_ok = local_key_count == 2 * snapshot.elements &&
      global_key_count == 2 * 9;
  std::set<std::pair<std::int64_t, std::int64_t>> partition_keys;
  for (int entry = 0; entry + 1 < global_key_count; entry += 2)
    local_ok = partition_keys.emplace(global_keys[entry],
                                      global_keys[entry + 1]).second &&
        local_ok;
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    mfem::Mesh serial = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    std::set<std::pair<std::int64_t, std::int64_t>> serial_keys;
    for (int element = 0; element < serial.GetNE(); ++element)
    {
      mfem::ElementTransformation* transformation =
          serial.GetElementTransformation(element);
      if (transformation == nullptr)
      {
        local_ok = false;
        break;
      }
      mfem::IntegrationPoint center;
      center.x = 0.5;
      center.y = 0.5;
      mfem::Vector physical(dimension);
      transformation->Transform(center, physical);
      serial_keys.emplace(
          static_cast<std::int64_t>(std::llround(physical[0] * 1.0e12)),
          static_cast<std::int64_t>(std::llround(physical[1] * 1.0e12)));
    }
    local_ok = local_ok && serial_keys == partition_keys &&
        serial_keys.size() == 9;
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  return GlobalAll(local_ok);
}

bool ValidateMpiDeviceCase(const FullDeviceMeshEvidence& value,
                           MpiDeviceOracleResult& result,
                           const int injected_condense_failure_rank = -1)
{
  result = {};
  const FullDeviceTestSnapshot& snapshot = value.test_snapshot;
  const int elements = snapshot.elements;
  const int ni = snapshot.interior_dofs;
  const int nt = snapshot.trace_dofs;
  const int ng = snapshot.global_trace_dofs;
  const int owned = static_cast<int>(snapshot.owned_true_global_dofs.size());
  const int local_trace =
      static_cast<int>(snapshot.local_trace_global_true_dofs.size());
  const std::size_t total_interior =
      static_cast<std::size_t>(elements) * ni;
  const std::size_t total_trace =
      static_cast<std::size_t>(elements) * nt;
  const auto sized = [](const auto& values, const std::size_t expected)
  { return values.size() == expected; };
  const auto& corrections = snapshot.refinement_corrections;
  bool refinement_records_valid =
      corrections.size() <=
      static_cast<std::size_t>(2 * value.nonlinear_iterations);
  std::uint64_t previous_generation = 0;
  int previous_index = 0;
  double previous_candidate_residual =
      std::numeric_limits<double>::infinity();
  for (std::size_t entry = 0; entry < corrections.size(); ++entry)
  {
    const auto& record = corrections[entry];
    const bool clean = record.error == 0 && record.convergence_query == 0 &&
        record.converged == 1;
    const bool pure_warning = record.error == HYPRE_ERROR_CONV &&
        (record.convergence_query == 0 ||
         record.convergence_query == HYPRE_ERROR_CONV) &&
        record.converged == 0;
    const double expected_reduction = record.direct_candidate_true_residual /
        std::max(record.previous_true_residual, 1.0e-300);
    const bool same_generation =
        entry != 0 && record.generation == previous_generation;
    refinement_records_valid = refinement_records_valid &&
        record.generation > 0 && record.correction_index >= 1 &&
        record.correction_index <= 2 && (clean != pure_warning) &&
        record.iterations > 0 && std::isfinite(record.reported_residual) &&
        std::isfinite(record.previous_true_residual) &&
        std::isfinite(record.correction_true_residual) &&
        std::isfinite(record.derived_candidate_true_residual) &&
        std::isfinite(record.direct_candidate_true_residual) &&
        std::isfinite(record.reduction_factor) &&
        std::isfinite(record.derived_direct_agreement) &&
        record.correction_true_residual < 1.0 &&
        record.derived_candidate_true_residual <
            record.previous_true_residual &&
        record.direct_candidate_true_residual <
            record.previous_true_residual &&
        record.reduction_factor < 1.0 &&
        record.derived_direct_agreement <= 1.0e-10 &&
        std::abs(record.reduction_factor - expected_reduction) <=
            1.0e-12 * std::max(1.0, std::abs(expected_reduction)) &&
        (record.published ==
         (record.direct_candidate_true_residual <= 1.0e-10));
    if (same_generation)
    {
      refinement_records_valid = refinement_records_valid &&
          record.correction_index == previous_index + 1 &&
          std::abs(record.previous_true_residual -
                   previous_candidate_residual) <=
              1.0e-12 *
                  std::max(1.0, std::abs(previous_candidate_residual));
    }
    else
    {
      refinement_records_valid = refinement_records_valid &&
          record.generation > previous_generation &&
          record.correction_index == 1 &&
          (entry == 0 || corrections[entry - 1].published);
    }
    if (!record.published)
    {
      refinement_records_valid = refinement_records_valid &&
          entry + 1 < corrections.size() &&
          corrections[entry + 1].generation == record.generation;
    }
    previous_generation = record.generation;
    previous_index = record.correction_index;
    previous_candidate_residual = record.direct_candidate_true_residual;
  }
  refinement_records_valid = refinement_records_valid &&
      (corrections.empty() || corrections.back().published);
  const bool clean_inner_terminal = value.last_hypre_error == 0 &&
      value.last_hypre_converged == 1;
  const bool exact_bounded_warning =
      value.last_hypre_error == HYPRE_ERROR_CONV &&
      value.last_hypre_converged == 0;
  bool local_ok = value.rank >= 0 && value.rank < 2 && value.ranks == 2 &&
      value.elements_per_direction == 3 && elements > 0 && elements < 9 &&
      value.global_elements == 9 && ni > 0 && nt > 0 &&
      ng == 2 * 3 * 3 * (value.polynomial_order + 1) * conserved_components &&
      value.global_true_trace_unknowns == ng && owned > 0 &&
      local_trace >= owned && value.global_ghost_trace_unknowns > 0 &&
      value.global_unique_trace_faces == 18 &&
      value.effective_hypre_ilu_type == 0 &&
      value.effective_hypre_ilu_level_of_fill == 0 &&
      value.effective_hypre_ilu_local_reordering == 0 &&
      value.trace_solve_failure_code == 0 &&
      (clean_inner_terminal || exact_bounded_warning) &&
      value.last_hypre_iterations > 0 &&
      value.last_hypre_global_rows == ng &&
      value.last_hypre_krylov_dimension == ng &&
      value.last_hypre_max_iterations == std::max(200, 2 * ng) &&
      std::isfinite(value.last_hypre_reported_residual) &&
      refinement_records_valid &&
      snapshot.global_linear_oracle_staged &&
      sized(snapshot.old_interior, total_interior) &&
      sized(snapshot.initial_interior, total_interior) &&
      sized(snapshot.initial_trace, local_trace) &&
      sized(snapshot.signed_trace_dofs, total_trace) &&
      sized(snapshot.signed_global_trace_dofs, total_trace) &&
      sized(snapshot.global_residual_r, total_interior) &&
      sized(snapshot.global_residual_g, total_trace) &&
      sized(snapshot.global_a, total_interior * ni) &&
      sized(snapshot.global_b, total_interior * nt) &&
      sized(snapshot.global_c, total_trace * ni) &&
      sized(snapshot.global_d, total_trace * nt) &&
      sized(snapshot.global_f, total_interior) &&
      sized(snapshot.global_g, total_trace) &&
      sized(snapshot.global_schur, total_trace * nt) &&
      sized(snapshot.global_condensed_rhs, total_trace) &&
      sized(snapshot.global_trace_increment, local_trace) &&
      sized(snapshot.global_local_trace_increment, total_trace) &&
      sized(snapshot.global_recovered_increment, total_interior) &&
      sized(snapshot.actual_true_rhs, owned) &&
      sized(snapshot.actual_true_solution, owned) &&
      sized(snapshot.actual_true_basis_actions,
            static_cast<std::size_t>(ng) * owned) &&
      sized(snapshot.final_interior, total_interior) &&
      sized(snapshot.final_trace, local_trace) &&
      sized(snapshot.last_base_interior, total_interior) &&
      sized(snapshot.last_base_trace, local_trace) &&
      sized(snapshot.last_delta_interior, total_interior) &&
      sized(snapshot.last_delta_trace, local_trace) &&
      snapshot.last_accepted_alpha > 0.0;
  if (!GlobalAll(local_ok)) { return false; }

  IndependentLinearization first_linearization;
  local_ok = IndependentAssembleLinearization(
      snapshot, snapshot.initial_interior, snapshot.initial_trace,
      first_linearization);
  if (!GlobalAll(local_ok)) { return false; }
  IndependentCondensedSystem first_system;
  IndependentCondenseDiagnostic first_condense_diagnostic;
  local_ok = IndependentCondenseSolve(
      snapshot, first_linearization, first_system,
      injected_condense_failure_rank, &first_condense_diagnostic);
  result.condense_local_failure_stage =
      first_condense_diagnostic.local_failure_stage;
  result.condense_failure_stage_min =
      first_condense_diagnostic.minimum_failure_stage;
  result.condense_failure_stage_max =
      first_condense_diagnostic.maximum_failure_stage;
  if (!GlobalAll(local_ok)) { return false; }

  result.residual_error = std::max(
      MaximumDifference(first_linearization.residual_r,
                        snapshot.global_residual_r),
      MaximumDifference(first_linearization.residual_g,
                        snapshot.global_residual_g));
  result.jacobian_error = std::max({
      MaximumDifference(first_linearization.a, snapshot.global_a),
      MaximumDifference(first_linearization.b, snapshot.global_b),
      MaximumDifference(first_linearization.c, snapshot.global_c),
      MaximumDifference(first_linearization.d, snapshot.global_d),
      MaximumDifference(first_linearization.f, snapshot.global_f),
      MaximumDifference(first_linearization.g, snapshot.global_g)});
  result.condensation_error = std::max(
      MaximumDifference(first_system.schur, snapshot.global_schur),
      MaximumDifference(first_system.condensed_rhs,
                        snapshot.global_condensed_rhs));
  result.recovery_error = std::max(
      MaximumDifference(first_system.local_trace_increment,
                        snapshot.global_local_trace_increment),
      MaximumDifference(first_system.recovered_increment,
                        snapshot.global_recovered_increment));

  std::set<std::int64_t> owned_ids(
      snapshot.owned_true_global_dofs.begin(),
      snapshot.owned_true_global_dofs.end());
  bool local_offdiag = false;
  result.true_matrix_error = 0.0;
  result.true_rhs_error = 0.0;
  result.trace_solution_error = 0.0;
  for (int row = 0; row < owned; ++row)
  {
    const std::int64_t global_row_i64 =
        snapshot.owned_true_global_dofs[static_cast<std::size_t>(row)];
    if (global_row_i64 < 0 || global_row_i64 >= ng)
    {
      local_ok = false;
      continue;
    }
    const int global_row = static_cast<int>(global_row_i64);
    result.true_rhs_error = std::max(
        result.true_rhs_error,
        std::abs(snapshot.actual_true_rhs[static_cast<std::size_t>(row)] -
                 first_system.true_rhs[static_cast<std::size_t>(global_row)]));
    result.trace_solution_error = std::max(
        result.trace_solution_error,
        std::abs(snapshot.actual_true_solution[static_cast<std::size_t>(row)] -
                 first_system.trace_increment[
                     static_cast<std::size_t>(global_row)]));
    for (int column = 0; column < ng; ++column)
    {
      const double expected = first_system.true_matrix[
          static_cast<std::size_t>(global_row) * ng + column];
      const double actual = snapshot.actual_true_basis_actions[
          static_cast<std::size_t>(column) * owned + row];
      result.true_matrix_error = std::max(
          result.true_matrix_error, std::abs(actual - expected));
      local_offdiag = local_offdiag ||
          (owned_ids.count(column) == 0 && std::abs(expected) > 1.0e-14);
    }
  }
  if (!GlobalAll(local_ok)) { return false; }
  int local_offdiag_value = local_offdiag ? 1 : 0;
  int global_offdiag_value = 0;
  MPI_Allreduce(&local_offdiag_value, &global_offdiag_value, 1, MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);
  result.offdiag_action = global_offdiag_value == 1;

  // Check the distributed first solution through the signed element maps,
  // independently of the owner-only true-vector comparison above.
  double distributed_trace_error = 0.0;
  for (std::size_t entry = 0; entry < total_trace; ++entry)
  {
    double sign = 1.0;
    const int local = Decode(snapshot.signed_trace_dofs[entry], sign);
    if (local < 0 || local >= local_trace)
    {
      local_ok = false;
      continue;
    }
    distributed_trace_error = std::max(
        distributed_trace_error,
        std::abs(sign * snapshot.global_trace_increment[
                            static_cast<std::size_t>(local)] -
                 first_system.local_trace_increment[entry]));
  }

  IndependentLinearization last_linearization;
  if (!GlobalAll(local_ok)) { return false; }
  local_ok = IndependentAssembleLinearization(
      snapshot, snapshot.last_base_interior, snapshot.last_base_trace,
      last_linearization);
  if (!GlobalAll(local_ok)) { return false; }
  IndependentCondensedSystem last_system;
  IndependentCondenseDiagnostic last_condense_diagnostic;
  local_ok = IndependentCondenseSolve(snapshot, last_linearization,
                                      last_system, -1,
                                      &last_condense_diagnostic);
  if (!local_ok)
  {
    result.condense_local_failure_stage =
        last_condense_diagnostic.local_failure_stage;
    result.condense_failure_stage_min =
        last_condense_diagnostic.minimum_failure_stage;
    result.condense_failure_stage_max =
        last_condense_diagnostic.maximum_failure_stage;
  }
  if (!GlobalAll(local_ok)) { return false; }
  result.recovery_error = std::max(
      result.recovery_error,
      MaximumDifference(last_system.recovered_increment,
                        snapshot.last_delta_interior));
  result.state_error = 0.0;
  for (std::size_t entry = 0; entry < total_interior; ++entry)
    result.state_error = std::max(
        result.state_error,
        std::abs(snapshot.final_interior[entry] -
                 (snapshot.last_base_interior[entry] +
                  snapshot.last_accepted_alpha *
                      last_system.recovered_increment[entry])));
  for (std::size_t entry = 0; entry < snapshot.final_trace.size(); ++entry)
    result.state_error = std::max(
        result.state_error,
        std::abs(snapshot.final_trace[entry] -
                 (snapshot.last_base_trace[entry] +
                  snapshot.last_accepted_alpha *
                      snapshot.last_delta_trace[entry])));
  for (std::size_t entry = 0; entry < total_trace; ++entry)
  {
    double sign = 1.0;
    const int local = Decode(snapshot.signed_trace_dofs[entry], sign);
    if (local < 0 || local >= local_trace)
    {
      local_ok = false;
      continue;
    }
    result.state_error = std::max(
        result.state_error,
        std::abs(sign * snapshot.last_delta_trace[
                            static_cast<std::size_t>(local)] -
                 last_system.local_trace_increment[entry]));
  }
  result.trace_solution_error = std::max(
      result.trace_solution_error, distributed_trace_error);

  bool local_ghost = false;
  bool local_ghost_truth =
      sized(snapshot.ghost_probe_owned, owned) &&
      sized(snapshot.ghost_probe_distributed, local_trace);
  for (int row = 0; row < owned && local_ghost_truth; ++row)
  {
    const double expected = 0.25 + std::sin(
        0.01 * static_cast<double>(
            snapshot.owned_true_global_dofs[static_cast<std::size_t>(row)] +
            1));
    local_ghost_truth = std::abs(
        snapshot.ghost_probe_owned[static_cast<std::size_t>(row)] -
        expected) <= 1.0e-14;
  }
  for (int local = 0; local < local_trace && local_ghost_truth; ++local)
  {
    const std::int64_t global =
        snapshot.local_trace_global_true_dofs[static_cast<std::size_t>(local)];
    const double expected =
        0.25 + std::sin(0.01 * static_cast<double>(global + 1));
    local_ghost_truth = global >= 0 && global < ng &&
        std::abs(snapshot.ghost_probe_distributed[
                     static_cast<std::size_t>(local)] - expected) <= 1.0e-14;
    local_ghost = local_ghost || owned_ids.count(global) == 0;
  }
  for (std::size_t entry = 0;
       entry < total_trace && local_ghost_truth; ++entry)
  {
    double local_sign = 1.0;
    double global_sign = 1.0;
    const int local = Decode(snapshot.signed_trace_dofs[entry], local_sign);
    const int global =
        Decode(snapshot.signed_global_trace_dofs[entry], global_sign);
    local_ghost_truth = local >= 0 && local < local_trace && global >= 0 &&
        global < ng && local_sign == global_sign &&
        snapshot.local_trace_global_true_dofs[
            static_cast<std::size_t>(local)] == global;
  }
  int local_ghost_value = local_ghost ? 1 : 0;
  int global_ghost_value = 0;
  MPI_Allreduce(&local_ghost_value, &global_ghost_value, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  result.ghost_truth = GlobalAll(local_ghost_truth) &&
      global_ghost_value == 1;
  result.coordinate_coverage = CoordinateCoverage(snapshot);

  result.residual_error = GlobalMaximum(result.residual_error);
  result.jacobian_error = GlobalMaximum(result.jacobian_error);
  result.condensation_error = GlobalMaximum(result.condensation_error);
  result.true_matrix_error = GlobalMaximum(result.true_matrix_error);
  result.true_rhs_error = GlobalMaximum(result.true_rhs_error);
  result.trace_solution_error = GlobalMaximum(result.trace_solution_error);
  result.recovery_error = GlobalMaximum(result.recovery_error);
  result.state_error = GlobalMaximum(result.state_error);
  int minimum_status = static_cast<int>(value.status);
  int maximum_status = minimum_status;
  MPI_Allreduce(MPI_IN_PLACE, &minimum_status, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum_status, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  result.status_truth = minimum_status == maximum_status && value.pass &&
      value.status == FullDeviceStatus::success &&
      value.nonlinear_iterations >= 2 &&
      value.committed_generation ==
          static_cast<std::uint64_t>(value.nonlinear_iterations) &&
      std::isfinite(value.true_relative_residual) &&
      value.true_relative_residual <= 1.0e-10;
  local_ok = local_ok && result.coordinate_coverage && result.ghost_truth &&
      result.offdiag_action && result.status_truth &&
      result.residual_error <= 5.0e-11 &&
      result.jacobian_error <= 5.0e-11 &&
      result.condensation_error <= 5.0e-11 &&
      result.true_matrix_error <= 5.0e-11 &&
      result.true_rhs_error <= 5.0e-11 &&
      result.trace_solution_error <= 5.0e-9 &&
      result.recovery_error <= 5.0e-9 && result.state_error <= 5.0e-9 &&
      value.real_partition_verified &&
      value.mfem_true_dof_ownership_verified &&
      value.cross_rank_ghost_exchange_verified &&
      value.cross_rank_ghost_orientation_verified &&
      value.host_staged_mpi_exchange && value.collective_failure_atomic &&
      value.global_unique_trace_faces == 18 &&
      value.global_shared_faces > 0 && value.global_ghost_trace_unknowns > 0 &&
      value.exact_ptslp_assembly && value.hypre_matrix_device &&
      value.hypre_rhs_device && value.hypre_solution_device &&
      value.hypre_gmres && value.hypre_ilu_type_zero_fill_zero &&
      value.effective_hypre_ilu_type == 0 &&
      value.effective_hypre_ilu_level_of_fill == 0 &&
      value.telemetry.host_staged_mpi_exchange_events > 0 &&
      value.telemetry.host_staged_mpi_device_to_host_bytes > 0 &&
      value.telemetry.host_staged_mpi_host_to_device_bytes > 0 &&
      value.telemetry.mpi_collective_events > 0 &&
      value.telemetry.fallback_events == 0 &&
      value.telemetry.dense_host_trace_inverse_events == 0 &&
      value.telemetry.solver_substitution_events == 0 &&
      value.telemetry.app_forall_host_fallbacks == 0;
  result.pass = GlobalAll(local_ok);
  return result.pass;
}

FullDevicePhase InjectedFailurePhase(const FullDeviceFailurePoint point)
{
  switch (point)
  {
    case FullDeviceFailurePoint::local_assembly:
      return FullDevicePhase::local_assembly;
    case FullDeviceFailurePoint::condensation:
      return FullDevicePhase::condensation;
    case FullDeviceFailurePoint::trace_assembly:
      return FullDevicePhase::trace_assembly;
    case FullDeviceFailurePoint::trace_solve:
    case FullDeviceFailurePoint::distribution:
      return FullDevicePhase::trace_solve;
    case FullDeviceFailurePoint::recovery:
      return FullDevicePhase::recovery;
    case FullDeviceFailurePoint::state_update:
      return FullDevicePhase::state_update;
    case FullDeviceFailurePoint::none:
      return FullDevicePhase::setup;
  }
  return FullDevicePhase::setup;
}

bool ValidateMpiRollback(const FullDeviceEvidence& rollback, const int p,
                         const FullDeviceFailurePoint point)
{
  bool local_ok = rollback.status == FullDeviceStatus::injected_failure &&
      !rollback.pass && rollback.cases.size() == 1;
  if (!GlobalAll(local_ok)) { return false; }
  const FullDeviceMeshEvidence& rolled = rollback.cases.front();
  const FullDeviceTestSnapshot& snapshot = rolled.test_snapshot;
  const FullDevicePhase failure_phase = InjectedFailurePhase(point);
  const bool entry_restored =
      BitwiseEqual(snapshot.rollback_entry_interior,
                   snapshot.rollback_restored_interior) &&
      BitwiseEqual(snapshot.rollback_entry_trace,
                   snapshot.rollback_restored_trace);
  const bool changed =
      !BitwiseEqual(snapshot.rollback_entry_interior,
                    snapshot.rollback_changed_interior) ||
      !BitwiseEqual(snapshot.rollback_entry_trace,
                    snapshot.rollback_changed_trace);
  int minimum_status = static_cast<int>(rolled.status);
  int maximum_status = minimum_status;
  unsigned long long minimum_generation = rolled.committed_generation;
  unsigned long long maximum_generation = minimum_generation;
  MPI_Allreduce(MPI_IN_PLACE, &minimum_status, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum_status, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minimum_generation, 1,
                MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum_generation, 1,
                MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
  const auto& failed =
      rolled.phases[static_cast<std::size_t>(failure_phase)];
  local_ok = minimum_status == maximum_status &&
      minimum_generation == maximum_generation &&
      rolled.polynomial_order == p && rolled.elements_per_direction == 3 &&
      rolled.ranks == 2 && rolled.status == FullDeviceStatus::injected_failure &&
      !rolled.pass && rolled.rollback_restored && entry_restored &&
      rolled.rollback_had_uncommitted_change ==
          (point == FullDeviceFailurePoint::state_update) &&
      rolled.rollback_nonfinite_change_observed ==
          (point == FullDeviceFailurePoint::state_update) &&
      changed == (point == FullDeviceFailurePoint::state_update) &&
      rolled.rollback_rejected_post_failure_consumption &&
      rolled.invalidated_product_count == 5 &&
      rolled.rejected_consumer_count == 6 &&
      rolled.nonlinear_iterations == 0 && rolled.linear_iterations == 0 &&
      rolled.committed_generation == 0 &&
      rolled.discarded_attempt_generation > 0 &&
      rolled.final_residual == rolled.initial_residual &&
      std::isinf(rolled.true_relative_residual) &&
      DerivedSnapshotEmpty(snapshot) && failed.reached &&
      failed.status == FullDeviceStatus::injected_failure &&
      failed.contemporaneous_audit && !failed.device_resident &&
      rolled.failure_phase_audit_valid &&
      rolled.failure_phase == failure_phase &&
      rolled.failure_phase_status == FullDeviceStatus::injected_failure &&
      rolled.failure_phase_generation == rolled.discarded_attempt_generation &&
      rolled.phase_audit_append_only && rolled.managed_residency_verified &&
      rolled.telemetry.fallback_events == 0 &&
      rolled.telemetry.dense_host_trace_inverse_events == 0 &&
      rolled.telemetry.solver_substitution_events == 0;
  return GlobalAll(local_ok);
}

FullDeviceRequest MpiDeviceRequest()
{
  FullDeviceRequest request;
  request.enabled = true;
  request.run_complete_ladder = false;
  request.enable_test_oracle_staging = true;
  request.enable_global_linear_oracle_staging = true;
  request.run_mpi_device_rank_bounded = true;
  request.enable_host_staged_mpi_exchange = true;
  return request;
}

bool RunMpiRollbackCampaign(const FullDeviceRequest& request,
                            int& failure_transactions)
{
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::array<FullDeviceFailurePoint, 7> failure_points{{
      FullDeviceFailurePoint::local_assembly,
      FullDeviceFailurePoint::condensation,
      FullDeviceFailurePoint::trace_assembly,
      FullDeviceFailurePoint::trace_solve,
      FullDeviceFailurePoint::distribution,
      FullDeviceFailurePoint::recovery,
      FullDeviceFailurePoint::state_update}};
  bool campaign_ok = true;
  failure_transactions = 0;
  for (const FullDeviceFailurePoint point : failure_points)
  {
    const int last_p = point == FullDeviceFailurePoint::state_update ? 4 : 1;
    for (int p = 1; p <= last_p; ++p)
    {
      FullDeviceRequest failure_request = request;
      failure_request.test_only_polynomial_order = p;
      failure_request.injected_failure_point = point;
      failure_request.injected_failure_rank =
          (failure_transactions + p) % ranks;
      failure_request.inject_nonfinite_before_rollback =
          point == FullDeviceFailurePoint::state_update;
      FullDeviceEvidence rollback;
      const bool unexpectedly_passed =
          dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(
              failure_request, rollback);
      const bool expected_failure = GlobalAll(!unexpectedly_passed);
      const bool rollback_valid =
          expected_failure && ValidateMpiRollback(rollback, p, point);
      const bool transaction_ok = GlobalAll(rollback_valid);
      if (!transaction_ok && rank == 0)
      {
        PrintFailure(rollback, expected_failure ?
            "mpi_rank2_rollback" : "mpi_rank2_rollback_pass");
      }
      campaign_ok = campaign_ok && transaction_ok;
      ++failure_transactions;
    }
  }
  return GlobalAll(campaign_ok);
}

int RunMpiDeviceRank2Production(const int p)
{
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (!GlobalAll(ranks == 2 && p >= 1 && p <= 4)) { return 90; }
  FullDeviceRequest request = MpiDeviceRequest();
  request.test_only_polynomial_order = p;
  FullDeviceEvidence evidence;
  const bool production =
      dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(request, evidence);
  if (!GlobalAll(production && evidence.pass &&
                 evidence.status == FullDeviceStatus::success &&
                 evidence.cases.size() == 1 && evidence.ranks == 2 &&
                 evidence.one_physical_gpu_shared))
  {
    if (rank == 0) { PrintFailure(evidence, "mpi_rank2_split_production"); }
    return 100 + p;
  }
  MpiDeviceOracleResult oracle;
  if (!ValidateMpiDeviceCase(evidence.cases.front(), oracle))
  {
    if (rank == 0)
    {
      std::cerr << "HDG_EULER_DEVICE_FULL_MPI_SPLIT_ORACLE_FAILURE"
                << " p=" << p
                << " residual=" << oracle.residual_error
                << " jacobian=" << oracle.jacobian_error
                << " condensation=" << oracle.condensation_error
                << " true=" << oracle.true_matrix_error << "/"
                << oracle.true_rhs_error
                << " trace=" << oracle.trace_solution_error
                << " recovery=" << oracle.recovery_error
                << " state=" << oracle.state_error
                << " coordinate=" << oracle.coordinate_coverage
                << " ghost=" << oracle.ghost_truth
                << " offdiag=" << oracle.offdiag_action
                << " status=" << oracle.status_truth << '\n';
    }
    return 110 + p;
  }
  const FullDeviceMeshEvidence& value = evidence.cases.front();
  if (rank == 0)
  {
    std::cout << "HDG_EULER_DEVICE_FULL_MPI_RANK2_PRODUCTION_SUCCESS"
              << " p=" << p << " N=3 cases=1 ranks=2"
              << " nonlinear=" << value.nonlinear_iterations
              << " linear=" << value.linear_iterations
              << " refinement_records="
              << value.test_snapshot.refinement_corrections.size()
              << " true_residual=" << value.true_relative_residual
              << " oracle_max="
              << std::max({oracle.residual_error, oracle.jacobian_error,
                           oracle.condensation_error,
                           oracle.true_matrix_error, oracle.true_rhs_error,
                           oracle.trace_solution_error, oracle.recovery_error,
                           oracle.state_error})
              << " fallback=0 dense_host_inverse=0 solver_substitution=0\n";
  }
  return 0;
}

int RunMpiDeviceRank2Rollbacks()
{
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (!GlobalAll(ranks == 2)) { return 120; }
  int failure_transactions = 0;
  if (!RunMpiRollbackCampaign(MpiDeviceRequest(), failure_transactions))
    return 121;
  if (rank == 0)
  {
    std::cout << "HDG_EULER_DEVICE_FULL_MPI_RANK2_ROLLBACK_SUCCESS"
              << " ranks=2 transactions=" << failure_transactions
              << " fallback=0 dense_host_inverse=0 solver_substitution=0\n";
  }
  return 0;
}

int RunMpiDeviceRank2()
{
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (!GlobalAll(ranks == 2)) { return 80; }
  FullDeviceEvidence disabled;
  if (dg::hdg::euler::RunSmoothEulerDeviceFullExperiment({}, disabled) ||
      !GlobalAll(disabled.status == FullDeviceStatus::default_disabled &&
                 disabled.cases.empty() && !disabled.pass))
  {
    if (rank == 0) { PrintFailure(disabled, "mpi_rank2_default_off"); }
    return 81;
  }

  FullDeviceRequest request = MpiDeviceRequest();
  FullDeviceEvidence evidence;
  const bool production =
      dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(request, evidence);
  if (!GlobalAll(production && evidence.pass &&
                 evidence.status == FullDeviceStatus::success &&
                 evidence.cases.size() == 4 && evidence.ranks == 2 &&
                 evidence.one_physical_gpu_shared))
  {
    if (rank == 0) { PrintFailure(evidence, "mpi_rank2_production"); }
    return 82;
  }
  double maximum_oracle_error = 0.0;
  std::array<int, 3> p1_local_topology{{0, 0, 0}};
  int p1_global_rows = 0;
  int p1_global_shared_faces = 0;
  std::size_t p1_global_ghosts = 0;
  double maximum_true_residual = 0.0;
  bool oracle_preflight_collective_failure_verified = false;
  for (int p = 1; p <= 4; ++p)
  {
    const auto found = std::find_if(
        evidence.cases.begin(), evidence.cases.end(),
        [p](const FullDeviceMeshEvidence& value)
        { return value.polynomial_order == p; });
    bool found_case = found != evidence.cases.end();
    if (!GlobalAll(found_case)) { return 83; }
    if (p == 1)
    {
      MpiDeviceOracleResult injected_oracle;
      const bool unexpectedly_admitted =
          ValidateMpiDeviceCase(*found, injected_oracle, 1);
      const int expected_stage = static_cast<int>(
          IndependentCondenseFailureStage::injected_preparation);
      const bool local_injection_observed =
          injected_oracle.condense_local_failure_stage ==
              (rank == 1 ? expected_stage : 0) &&
          injected_oracle.condense_failure_stage_min == expected_stage &&
          injected_oracle.condense_failure_stage_max == expected_stage;
      oracle_preflight_collective_failure_verified =
          GlobalAll(!unexpectedly_admitted && local_injection_observed);
      if (!oracle_preflight_collective_failure_verified)
      {
        if (rank == 0)
        {
          std::cerr
              << "HDG_EULER_DEVICE_FULL_MPI_ORACLE_PREFLIGHT_FAILURE"
              << " local_stage="
              << injected_oracle.condense_local_failure_stage
              << " stage_min="
              << injected_oracle.condense_failure_stage_min
              << " stage_max="
              << injected_oracle.condense_failure_stage_max << '\n';
        }
        return 86;
      }
      if (rank == 0)
      {
        std::cout
            << "HDG_EULER_DEVICE_FULL_MPI_ORACLE_PREFLIGHT_COLLECTIVE_SUCCESS"
            << " injected_rank=1 failure_stage=" << expected_stage
            << " failure_stage_min="
            << injected_oracle.condense_failure_stage_min
            << " failure_stage_max="
            << injected_oracle.condense_failure_stage_max << '\n';
      }
    }
    MpiDeviceOracleResult oracle;
    if (!ValidateMpiDeviceCase(*found, oracle))
    {
      if (rank == 0)
      {
        std::cerr << "HDG_EULER_DEVICE_FULL_MPI_ORACLE_FAILURE p=" << p
                  << " residual=" << oracle.residual_error
                  << " jacobian=" << oracle.jacobian_error
                  << " condensation=" << oracle.condensation_error
                  << " true=" << oracle.true_matrix_error << "/"
                  << oracle.true_rhs_error
                  << " trace=" << oracle.trace_solution_error
                  << " recovery=" << oracle.recovery_error
                  << " state=" << oracle.state_error
                  << " coordinate=" << oracle.coordinate_coverage
                  << " ghost=" << oracle.ghost_truth
                  << " offdiag=" << oracle.offdiag_action
                  << " status=" << oracle.status_truth << '\n';
      }
      return 84;
    }
    maximum_oracle_error = std::max({
        maximum_oracle_error, oracle.residual_error, oracle.jacobian_error,
        oracle.condensation_error, oracle.true_matrix_error,
        oracle.true_rhs_error, oracle.trace_solution_error,
        oracle.recovery_error, oracle.state_error});
    maximum_true_residual =
        std::max(maximum_true_residual, found->true_relative_residual);
    if (p == 1)
    {
      p1_local_topology = {{found->local_trace_unknowns,
                            found->true_trace_unknowns,
                            static_cast<int>(found->ghost_trace_unknowns)}};
      p1_global_rows = static_cast<int>(found->global_true_trace_unknowns);
      p1_global_shared_faces = found->global_shared_faces;
      p1_global_ghosts = found->global_ghost_trace_unknowns;
    }
  }

  int failure_transactions = 0;
  if (!RunMpiRollbackCampaign(request, failure_transactions))
  {
    return 85;
  }

  std::array<int, 6> p1_topology_by_rank{{0, 0, 0, 0, 0, 0}};
  MPI_Gather(p1_local_topology.data(), 3, MPI_INT,
             p1_topology_by_rank.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  unsigned long long local_telemetry[4] = {0, 0, 0, 0};
  for (const FullDeviceMeshEvidence& value : evidence.cases)
  {
    local_telemetry[0] += value.telemetry.host_staged_mpi_exchange_events;
    local_telemetry[1] +=
        value.telemetry.host_staged_mpi_device_to_host_bytes;
    local_telemetry[2] +=
        value.telemetry.host_staged_mpi_host_to_device_bytes;
    local_telemetry[3] += value.telemetry.mpi_collective_events;
  }
  unsigned long long global_telemetry[4] = {0, 0, 0, 0};
  MPI_Allreduce(local_telemetry, global_telemetry, 4,
                MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
  {
    std::cout << "HDG_EULER_DEVICE_FULL_MPI_RANK2_SUCCESS"
              << " ranks=2 cases=4 p=1-4 N=3 global_elements=9"
              << " shared_gpu=" << evidence.one_physical_gpu_shared
              << " p1_global_rows=" << p1_global_rows
              << " p1_global_shared_faces=" << p1_global_shared_faces
              << " p1_global_ghosts=" << p1_global_ghosts
              << " p1_rank0_vsize/true/ghost=" << p1_topology_by_rank[0]
              << "/" << p1_topology_by_rank[1] << "/"
              << p1_topology_by_rank[2]
              << " p1_rank1_vsize/true/ghost=" << p1_topology_by_rank[3]
              << "/" << p1_topology_by_rank[4] << "/"
              << p1_topology_by_rank[5]
              << " oracle_max=" << maximum_oracle_error
              << " true_residual_max=" << maximum_true_residual
              << " host_staging_events_global=" << global_telemetry[0]
              << " staged_d2h_global=" << global_telemetry[1]
              << " staged_h2d_global=" << global_telemetry[2]
              << " collective_events_global=" << global_telemetry[3]
              << " rollback_transactions=" << failure_transactions
              << " oracle_preflight_collective_failure="
              << oracle_preflight_collective_failure_verified
              << " fallback=0 dense_host_inverse=0 solver_substitution=0"
              << '\n';
  }
  return 0;
}
}  // namespace

int main(int argc, char* argv[])
{
  if (argc == 2 && std::string(argv[1]) == "--umpire-event-audit-driver")
    return RunUmpireAuditDriver(argv[0]);
  const bool umpire_audit_child =
      argc == 2 && std::string(argv[1]) == "--umpire-audit-child";
  const bool mpi_device_rank2 =
      argc == 2 && std::string(argv[1]) == "--mpi-device-rank2";
  const bool mpi_device_rank2_rollbacks =
      argc == 2 && std::string(argv[1]) == "--mpi-device-rank2-rollbacks";
  int mpi_device_rank2_production_p = 0;
  if (argc == 2)
  {
    const std::string mode(argv[1]);
    for (int p = 1; p <= 4; ++p)
      if (mode == "--mpi-device-rank2-production-p" + std::to_string(p))
        mpi_device_rank2_production_p = p;
  }
  const bool mpi_device_split = mpi_device_rank2_rollbacks ||
      mpi_device_rank2_production_p != 0;
  if (argc != 1 && !umpire_audit_child && !mpi_device_rank2 &&
      !mpi_device_split)
    return 73;
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  if (!GlobalAll(IndependentPhysicsProbe())) { return 1; }
  if (mpi_device_rank2_production_p != 0)
    return RunMpiDeviceRank2Production(mpi_device_rank2_production_p);
  if (mpi_device_rank2_rollbacks) { return RunMpiDeviceRank2Rollbacks(); }
  if (mpi_device_rank2) { return RunMpiDeviceRank2(); }

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
      !GlobalOracleAccepted(global_oracle))
  {
    PrintGlobalOracleFailure(1, global_oracle);
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
                    }) != 4 ||
      !std::all_of(evidence.cases.begin(), evidence.cases.end(),
                   [](const FullDeviceMeshEvidence& value)
                   {
                     return value.test_snapshot.global_linear_oracle_staged ==
                            (value.elements_per_direction == 3);
                   }))
  {
    PrintFailure(evidence, "bounded_global_staging");
    return 11;
  }
  std::size_t global_oracle_count = 0;
  for (int p = 1; p <= 4; ++p)
  {
    const auto found = std::find_if(
        evidence.cases.begin(), evidence.cases.end(),
        [p](const FullDeviceMeshEvidence& value)
        {
          return value.polynomial_order == p &&
                 value.elements_per_direction == 3;
        });
    GlobalLinearOracleResult oracle;
    if (found == evidence.cases.end() ||
        !IndependentGlobalLinearOracle(*found, oracle) ||
        !GlobalOracleAccepted(oracle))
    {
      PrintGlobalOracleFailure(p, oracle);
      PrintFailure(evidence, "p1_p4_n3_global_oracle");
      return 13;
    }
    ++global_oracle_count;
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

  std::size_t rollback_count = 0;
  for (int p = 1; p <= 4; ++p)
  {
    FullDeviceRequest rollback_request;
    rollback_request.enabled = true;
    rollback_request.run_complete_ladder = true;
    rollback_request.enable_test_oracle_staging = true;
    rollback_request.inject_failure_after_first_update = true;
    rollback_request.injected_failure_polynomial_order = p;
    rollback_request.injected_failure_elements_per_direction = 3;
    rollback_request.inject_nonfinite_before_rollback = true;
    FullDeviceEvidence rollback;
    if (dg::hdg::euler::RunSmoothEulerDeviceFullExperiment(rollback_request,
                                                            rollback) ||
        !ValidateRollbackEvidence(rollback, p))
    {
      PrintFailure(rollback, "p1_p4_n3_rollback_transaction");
      return p == 1 ? 7 : 9;
    }
    ++rollback_count;
  }

  std::cout << "HDG_EULER_DEVICE_FULL_SUCCESS cases=" << evidence.cases.size()
            << " global_oracles=" << global_oracle_count
            << " rollbacks=" << rollback_count << " rollback=restored\n";
  if (umpire_audit_child) { umpire::mark_event(umpire_terminal_marker); }
  return 0;
}
