#include "independent_oracle.hpp"
#include "sc25_scalar_igr_independent_oracle.hpp"

#include "hdg_cpu_serial_lifecycle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace dg::hdg::runtime::cpu::sc25::w8c_internal
{
[[gnu::visibility("hidden")]] void RequestSmoothTransitionInitialization(
    const PersistentSc25ScalarIgrCpuFoundation& owner) noexcept;
}

namespace
{

using namespace dg::hdg::runtime;
namespace cpu = dg::hdg::runtime::cpu;
namespace sc25 = dg::hdg::runtime::cpu::sc25;
using SmoothCase = cpu::w8c::SmoothTransitionV2Case;

struct SmoothSelection : CpuSerialRuntimeSelection
{
  static constexpr std::string_view experiment_id = SmoothCase::experiment_id;
};

struct Checks final
{
  void Require(const bool condition, const char* message)
  {
    if (condition) return;
    ++failures;
    std::cerr << message << '\n';
  }
  int failures = 0;
};

bool Close(const double actual, const double expected) noexcept
{
  return std::abs(actual - expected) <=
      2048.0 * std::numeric_limits<double>::epsilon() *
          std::max({1.0, std::abs(actual), std::abs(expected)});
}

bool CloseArrays(const double* actual, const std::vector<double>& expected)
    noexcept
{
  if (!actual && !expected.empty()) return false;
  for (std::size_t index = 0; index < expected.size(); ++index)
    if (!Close(actual[index], expected[index])) return false;
  return true;
}

std::vector<double> ScalarUComponentMajor(
    const sc25::PersistentTrajectoryState& state)
{
  constexpr std::size_t elements = 64;
  constexpr std::size_t scalar_dofs = 4;
  std::vector<double> packed(state.u.size());
  for (std::size_t element = 0; element < elements; ++element)
    for (std::size_t component = 0; component < 4; ++component)
      for (std::size_t scalar = 0; scalar < scalar_dofs; ++scalar)
        packed[component * elements * scalar_dofs +
               element * scalar_dofs + scalar] =
            state.u[element * 4 * scalar_dofs +
                    component * scalar_dofs + scalar];
  return packed;
}

template <typename Range>
bool PositiveZero(const Range& values)
{
  return std::all_of(values.begin(), values.end(), [](const double value)
  {
    return value == 0.0 && !std::signbit(value);
  });
}

// This is deliberately a test-only postprocessor.  It reads the ordinary FE
// VTK state files that the application publishes; it does not add a solver
// diagnostic type or a second solver/reference lifecycle.
struct VtkPoint final
{
  double x = 0.0;
  double y = 0.0;
};

struct VtkState final
{
  std::vector<VtkPoint> points;
  std::vector<std::vector<std::size_t>> cells;
  std::array<std::vector<double>, 4> conservative;
};

struct CellSample final
{
  // Corner ordering is (xlo,ylo), (xhi,ylo), (xlo,yhi), (xhi,yhi).
  std::array<std::array<double, 4>, 4> conservative{};
};

struct Regularity final
{
  double rho_g = 0.0;
  double pressure_g = 0.0;
};

constexpr std::array<double, 18> q18_nodes{{
    -0.9915651684209309, -0.9558239495713977, -0.8926024664975557,
    -0.8037049589725231, -0.6916870430603532, -0.5597708310739475,
    -0.4117511614628426, -0.2582677505190968, -0.0950125098376374,
     0.0950125098376374,  0.2582677505190968,  0.4117511614628426,
     0.5597708310739475,  0.6916870430603532,  0.8037049589725231,
     0.8926024664975557,  0.9558239495713977,  0.9915651684209309}};

constexpr std::array<double, 18> q18_weights{{
    0.0216160135264833, 0.0497145488949698, 0.0764257302548891,
    0.1009420441062872, 0.1245092153167992, 0.1477391049013385,
    0.1691423829631436, 0.1890093628079284, 0.1984314853271116,
    0.1984314853271116, 0.1890093628079284, 0.1691423829631436,
    0.1477391049013385, 0.1245092153167992, 0.1009420441062872,
    0.0764257302548891, 0.0497145488949698, 0.0216160135264833}};

bool ReadOrdinaryVtk(const std::string& path, const bool scalar,
                     VtkState& result, std::string& error)
{
  std::ifstream input(path);
  if (!input) { error = "cannot open " + path; return false; }
  std::vector<std::vector<double>> fields(4);
  const std::string prefix = scalar ? "U" : "conservative";
  std::string token;
  std::size_t point_data = 0;
  while (input >> token)
  {
    if (token == "POINTS")
    {
      std::size_t count = 0;
      std::string type;
      if (!(input >> count >> type)) { error = "malformed POINTS in " + path; return false; }
      result.points.resize(count);
      for (auto& point : result.points)
      {
        double ignored_z = 0.0;
        if (!(input >> point.x >> point.y >> ignored_z))
        { error = "truncated POINTS in " + path; return false; }
      }
    }
    else if (token == "CELLS")
    {
      std::size_t count = 0;
      std::size_t ignored_size = 0;
      if (!(input >> count >> ignored_size)) { error = "malformed CELLS in " + path; return false; }
      result.cells.resize(count);
      for (auto& cell : result.cells)
      {
        std::size_t cell_size = 0;
        if (!(input >> cell_size) || cell_size != 4)
        { error = "non-quad VTK cell in " + path; return false; }
        cell.resize(cell_size);
        for (auto& index : cell)
          if (!(input >> index) || index >= result.points.size())
          { error = "invalid VTK cell index in " + path; return false; }
      }
    }
    else if (token == "POINT_DATA")
    {
      if (!(input >> point_data)) { error = "malformed POINT_DATA in " + path; return false; }
    }
    else if (token == "SCALARS")
    {
      std::string name;
      std::string type;
      std::size_t components = 0;
      std::string lookup;
      std::string lookup_name;
      if (!(input >> name >> type >> components >> lookup >> lookup_name) ||
          lookup != "LOOKUP_TABLE" || point_data != result.points.size())
      { error = "malformed scalar field in " + path; return false; }
      std::vector<double> values(point_data * components);
      for (double& value : values)
        if (!(input >> value) || !std::isfinite(value))
        { error = "invalid scalar value in " + path; return false; }
      for (std::size_t component = 0; component < fields.size(); ++component)
        if (name == prefix + std::to_string(component) && components == 1)
          fields[component] = std::move(values);
    }
    else if (token == "VECTORS")
    {
      std::string name;
      std::string type;
      if (!(input >> name >> type)) { error = "malformed VECTORS in " + path; return false; }
      double ignored = 0.0;
      for (std::size_t index = 0; index < 3 * point_data; ++index)
        if (!(input >> ignored)) { error = "truncated VECTORS in " + path; return false; }
    }
  }
  if (result.points.empty() || result.cells.empty())
  { error = "missing mesh in " + path; return false; }
  for (std::size_t component = 0; component < fields.size(); ++component)
    if (fields[component].size() != result.points.size())
    { error = "missing conservative component in " + path; return false; }
  result.conservative = std::move(fields);
  return true;
}

double Bilinear(const std::array<double, 4>& values, const double x,
                const double y) noexcept
{
  return (1.0 - x) * ((1.0 - y) * values[0] + y * values[2]) +
      x * ((1.0 - y) * values[1] + y * values[3]);
}

double BilinearDx(const std::array<double, 4>& values, const double y,
                  const double h) noexcept
{
  return ((1.0 - y) * (values[1] - values[0]) +
          y * (values[3] - values[2])) / h;
}

struct ConservativeEvaluation final
{
  std::array<double, 4> value{};
  std::array<double, 4> dx{};
};

bool EvaluateConservativeQ1(const CellSample& cell, const double x,
                            const double y, const double h,
                            ConservativeEvaluation& result,
                            std::string& error)
{
  for (std::size_t component = 0; component < result.value.size(); ++component)
  {
    result.value[component] = Bilinear(cell.conservative[component], x, y);
    result.dx[component] = BilinearDx(cell.conservative[component], y, h);
    if (!std::isfinite(result.value[component]) || !std::isfinite(result.dx[component]))
    { error = "nonfinite Q1 conservative reconstruction"; return false; }
  }
  const double rho = result.value[0];
  const double mx = result.value[1];
  const double my = result.value[2];
  const double energy = result.value[3];
  const double pressure = 0.4 *
      (energy - 0.5 * (mx * mx + my * my) / rho);
  if (!(rho > 0.0) || !(pressure > 0.0) || !std::isfinite(pressure))
  { error = "nonphysical Q1 conservative reconstruction"; return false; }
  return true;
}

bool EvaluateField(const CellSample& cell, const double x, const double y,
                   const double h, const bool pressure, double& value,
                   double& dx, std::string& error)
{
  ConservativeEvaluation conservative;
  if (!EvaluateConservativeQ1(cell, x, y, h, conservative, error)) return false;
  const double rho = conservative.value[0];
  if (!pressure)
  {
    value = rho;
    dx = conservative.dx[0];
    return true;
  }
  const double mx = conservative.value[1];
  const double my = conservative.value[2];
  const double energy = conservative.value[3];
  const double drho = conservative.dx[0];
  const double dmx = conservative.dx[1];
  const double dmy = conservative.dx[2];
  const double denergy = conservative.dx[3];
  value = 0.4 * (energy - 0.5 * (mx * mx + my * my) / rho);
  dx = 0.4 * (denergy - (mx * dmx + my * dmy) / rho +
              0.5 * (mx * mx + my * my) * drho / (rho * rho));
  if (!std::isfinite(value) || !std::isfinite(dx))
  { error = "nonfinite pressure reconstruction"; return false; }
  return true;
}

bool BuildCellSamples(const VtkState& state, const int n,
                      std::vector<CellSample>& samples, std::string& error)
{
  if (n <= 0 || state.cells.size() != static_cast<std::size_t>(n * n))
  { error = "unexpected VTK element census"; return false; }
  samples.assign(static_cast<std::size_t>(n * n), CellSample{});
  std::vector<bool> assigned(samples.size(), false);
  const double h = 1.0 / static_cast<double>(n);
  for (const auto& cell : state.cells)
  {
    double xlo = std::numeric_limits<double>::infinity();
    double xhi = -std::numeric_limits<double>::infinity();
    double ylo = std::numeric_limits<double>::infinity();
    double yhi = -std::numeric_limits<double>::infinity();
    for (const std::size_t point : cell)
    {
      xlo = std::min(xlo, state.points[point].x);
      xhi = std::max(xhi, state.points[point].x);
      ylo = std::min(ylo, state.points[point].y);
      yhi = std::max(yhi, state.points[point].y);
    }
    const int ix = static_cast<int>(std::lround(xlo / h));
    const int iy = static_cast<int>(std::lround(ylo / h));
    if (ix < 0 || ix >= n || iy < 0 || iy >= n ||
        !Close(xhi - xlo, h) || !Close(yhi - ylo, h))
    { error = "VTK mesh is not the expected periodic Cartesian grid"; return false; }
    const std::size_t target = static_cast<std::size_t>(ix * n + iy);
    if (assigned[target]) { error = "duplicate VTK cell"; return false; }
    assigned[target] = true;
    std::array<bool, 4> corners{{false, false, false, false}};
    for (const std::size_t point : cell)
    {
      const bool high_x = Close(state.points[point].x, xhi);
      const bool high_y = Close(state.points[point].y, yhi);
      const std::size_t corner = (high_x ? 1U : 0U) + (high_y ? 2U : 0U);
      if (corners[corner]) { error = "duplicate VTK cell corner"; return false; }
      corners[corner] = true;
      for (std::size_t component = 0;
           component < samples[target].conservative.size(); ++component)
        samples[target].conservative[component][corner] =
            state.conservative[component][point];
    }
    if (!std::all_of(corners.begin(), corners.end(), [](bool value) { return value; }))
    { error = "incomplete VTK cell corners"; return false; }
    for (const double local_x : std::array<double, 2>{{0.0, 1.0}})
      for (const double local_y : std::array<double, 2>{{0.0, 1.0}})
      {
        ConservativeEvaluation evaluation;
        if (!EvaluateConservativeQ1(samples[target], local_x, local_y, h,
                                    evaluation, error)) return false;
      }
  }
  if (!std::all_of(assigned.begin(), assigned.end(), [](bool value) { return value; }))
  { error = "missing VTK cell"; return false; }
  return true;
}

bool YAverageValue(const std::vector<CellSample>& cells, const int n,
                   const int ix, const bool pressure, const double local_x,
                   double& average, std::string& error)
{
  average = 0.0;
  const double h = 1.0 / static_cast<double>(n);
  for (int iy = 0; iy < n; ++iy)
  {
    const CellSample& cell = cells[static_cast<std::size_t>(ix * n + iy)];
    for (std::size_t q = 0; q < q18_nodes.size(); ++q)
    {
      const double local_y = 0.5 * (1.0 + q18_nodes[q]);
      double value = 0.0;
      double ignored_dx = 0.0;
      if (!EvaluateField(cell, local_x, local_y, h, pressure, value,
                         ignored_dx, error)) return false;
      average += q18_weights[q] * value / (2.0 * static_cast<double>(n));
    }
  }
  return std::isfinite(average);
}

bool YAverageDx(const std::vector<CellSample>& cells, const int n,
                const int ix, const bool pressure, const double local_x,
                double& average, std::string& error)
{
  average = 0.0;
  const double h = 1.0 / static_cast<double>(n);
  for (int iy = 0; iy < n; ++iy)
  {
    const CellSample& cell = cells[static_cast<std::size_t>(ix * n + iy)];
    for (std::size_t q = 0; q < q18_nodes.size(); ++q)
    {
      const double local_y = 0.5 * (1.0 + q18_nodes[q]);
      double ignored_value = 0.0;
      double dx = 0.0;
      if (!EvaluateField(cell, local_x, local_y, h, pressure, ignored_value,
                         dx, error)) return false;
      average += q18_weights[q] * dx / (2.0 * static_cast<double>(n));
    }
  }
  return std::isfinite(average);
}

bool EvaluateQ18Regularity(const std::vector<CellSample>& cells, const int n,
                           Regularity& result, std::string& error)
{
  const double h = 1.0 / static_cast<double>(n);
  result = {};
  for (const bool pressure : std::array<bool, 2>{{false, true}})
  {
    double broken_dx = 0.0;
    double jumps = 0.0;
    for (int ix = 0; ix < n; ++ix)
      for (const double node : q18_nodes)
      {
        double average_dx = 0.0;
        if (!YAverageDx(cells, n, ix, pressure, 0.5 * (1.0 + node),
                        average_dx, error)) return false;
        broken_dx = std::max(broken_dx, std::abs(average_dx));
      }
    // The periodic Cartesian mesh has one x-face per x element, including the
    // periodic seam.  Both incident elements are sampled one-sided.
    for (int face = 0; face < n; ++face)
    {
      const int left = (face + n - 1) % n;
      double left_value = 0.0;
      double right_value = 0.0;
      if (!YAverageValue(cells, n, left, pressure, 1.0, left_value, error) ||
          !YAverageValue(cells, n, face, pressure, 0.0, right_value, error))
        return false;
      jumps = std::max(jumps, std::abs(left_value - right_value) / h);
    }
    if (pressure) result.pressure_g = std::max(broken_dx, jumps);
    else result.rho_g = std::max(broken_dx, jumps);
  }
  return std::isfinite(result.rho_g) && std::isfinite(result.pressure_g);
}

std::string StepFileName(const std::uint64_t step)
{
  std::string digits = std::to_string(step);
  return "state-step-" + std::string(6U - std::min<std::size_t>(6U, digits.size()), '0') +
      digits + ".vtk";
}

bool EvaluateMatrixSeries(const std::string& directory, const int n,
                          const bool scalar,
                          const std::array<std::uint64_t, 9>& steps,
                          std::vector<Regularity>& series, std::string& error)
{
  series.clear();
  series.reserve(steps.size());
  for (const std::uint64_t step : steps)
  {
    VtkState state;
    std::vector<CellSample> cells;
    if (!ReadOrdinaryVtk(directory + "/" + StepFileName(step), scalar, state, error) ||
        !BuildCellSamples(state, n, cells, error)) return false;
    Regularity regularity;
    if (!EvaluateQ18Regularity(cells, n, regularity, error)) return false;
    series.push_back(regularity);
  }
  return true;
}

int RunQ18MatrixComparator(const int argc, char** argv)
{
  if (argc != 8)
  {
    std::cerr << "q18 matrix comparator requires six output directories\n";
    return 2;
  }
  struct Row final
  {
    std::string_view id;
    int n;
    std::array<std::uint64_t, 9> steps;
  };
  const std::array<Row, 3> rows{{
      {"n16dt11", 16, {{0, 16, 32, 48, 64, 80, 96, 112, 128}}},
      {"n16dt12", 16, {{0, 32, 64, 96, 128, 160, 192, 224, 256}}},
      {"n32dt12", 32, {{0, 32, 64, 96, 128, 160, 192, 224, 256}}}}};
  std::array<double, 3> s_none{};
  std::array<double, 3> s_scalar{};
  std::array<double, 3> delta_s{};
  bool statistics_defined = true;
  for (std::size_t row = 0; row < rows.size(); ++row)
  {
    std::vector<Regularity> conventional;
    std::vector<Regularity> scalar;
    std::string error;
    if (!EvaluateMatrixSeries(argv[2 + 2 * row], rows[row].n, false,
                              rows[row].steps, conventional, error) ||
        !EvaluateMatrixSeries(argv[3 + 2 * row], rows[row].n, true,
                              rows[row].steps, scalar, error))
    {
      std::cerr << "q18 matrix row " << row << " invalid: " << error << '\n';
      return 1;
    }
    const auto row_s = [](const std::vector<Regularity>& values,
                          double& result)
    {
      const double initial = values.front().pressure_g;
      if (!(initial > 0.0) || !std::isfinite(initial)) return false;
      result = 0.0;
      for (std::size_t state = 1; state < values.size(); ++state)
        result = std::max(result, values[state].pressure_g / initial);
      return std::isfinite(result);
    };
    const bool row_defined = row_s(conventional, s_none[row]) &&
        row_s(scalar, s_scalar[row]);
    statistics_defined = statistics_defined && row_defined;
    delta_s[row] = row_defined ? s_scalar[row] - s_none[row]
                               : std::numeric_limits<double>::quiet_NaN();
    std::cout << "q18-row id=" << rows[row].id
              << " n=" << rows[row].n
              << " S-none=" << s_none[row]
              << " S-scalar=" << s_scalar[row]
              << " Delta_S=" << delta_s[row] << '\n';
  }
  constexpr double epsilon = 0x1p-40;
  const double eta_delta = statistics_defined
      ? std::max(std::abs(delta_s[0] - delta_s[1]),
                 std::abs(delta_s[1] - delta_s[2]))
      : std::numeric_limits<double>::quiet_NaN();
  const double eta_none = statistics_defined
      ? std::max(std::abs(s_none[0] - s_none[1]),
                 std::abs(s_none[1] - s_none[2]))
      : std::numeric_limits<double>::quiet_NaN();
  const bool control = statistics_defined &&
      s_none[1] - 1.0 > 2.0 * eta_none + epsilon;
  const bool same_strict_sign = statistics_defined &&
      ((delta_s[1] > 0.0 && delta_s[2] > 0.0) ||
       (delta_s[1] < 0.0 && delta_s[2] < 0.0));
  const bool resolution = same_strict_sign &&
      std::abs(delta_s[1]) > 2.0 * eta_delta + epsilon;
  const char* disposition = control && resolution && delta_s[1] < 0.0
      ? "positive-bounded"
      : control && resolution && delta_s[1] > 0.0
      ? "negative-characterized"
      : "inconclusive";
  std::cout << "q18-summary eta_Delta=" << eta_delta
            << " eta_none=" << eta_none
            << " control=" << (control ? "excited" : "not-excited")
            << " resolution=" << (resolution ? "resolved" : "unresolved")
            << '\n';
  std::cout << "q18-matrix-disposition=" << disposition << '\n';
  return 0;
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  if (argc >= 2 && std::string_view(argv[1]) == "--matrix")
  {
    const int status = RunQ18MatrixComparator(argc, argv);
    MPI_Finalize();
    return status;
  }
  Checks checks;
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  checks.Require(ranks == 1,
                 "smooth-transition component test requires CPU serial");

  checks.Require(SmoothCase::case_id == "periodic-smooth-transition-v2" &&
                     SmoothCase::scenario_id == "W8C-V2-N8-DT10" &&
                     SmoothCase::mesh_n == 8 &&
                     SmoothCase::polynomial_order == 1 &&
                     SmoothCase::projection_q == 14 &&
                     SmoothCase::observable_q_o == 18 &&
                     SmoothCase::accepted_steps == 64 &&
                     SmoothCase::output_steps ==
                         std::array<std::uint64_t, 3>{{0, 32, 64}},
                 "smooth-transition descriptor must remain frozen and private");

  std::vector<double> oracle_interior;
  std::vector<double> oracle_trace;
  dg_igr_hdg_cpu_test::SmoothTransitionProjectionEvidence oracle_evidence;
  const bool oracle_ready =
      dg_igr_hdg_cpu_test::BuildSmoothTransitionV2ProjectionOracle(
          MPI_COMM_WORLD, oracle_interior, oracle_trace, oracle_evidence);
  checks.Require(oracle_ready &&
                     Close(oracle_evidence.density_integral, 0.5625) &&
                     Close(oracle_evidence.x_momentum_integral, 0.0) &&
                     Close(oracle_evidence.y_momentum_integral, 0.0) &&
                     Close(oracle_evidence.energy_integral, 1.375) &&
                     oracle_evidence.maximum_reflection_defect <= 1.0e-14 &&
                     oracle_evidence.maximum_half_period_complement_defect <=
                         1.0e-14 &&
                     oracle_evidence.maximum_y_invariance_defect == 0.0 &&
                     oracle_evidence.signed_permuted_trace_entries == 2048 &&
                     oracle_evidence.permuted_trace_entries == 1024 &&
                     oracle_evidence.doubly_covered_trace_dofs == 1024 &&
                     oracle_evidence.maximum_signed_trace_coefficient_defect <=
                         1.0e-14 &&
                     oracle_evidence.maximum_projected_trace_value_defect <=
                         1.0e-14,
                 "independent q14/q18 oracle must recover projections, ownership, means, and symmetries");

  cpu::CpuPersistentConfig conventional_config = cpu::MakeCpuSmokeConfig();
  conventional_config.experiment = StableIdentity::Copy(
      SmoothCase::experiment_id);
  conventional_config.source = StableIdentity::Copy(
      SmoothCase::conventional_source_id);
  conventional_config.polynomial_order = SmoothCase::polynomial_order;
  conventional_config.elements_per_direction =
      static_cast<int>(SmoothCase::mesh_n);
  conventional_config.assembly_quadrature_order = SmoothCase::conventional_q_a;
  conventional_config.dt = SmoothCase::dt;
  cpu::PersistentSmoothEulerCpuAdapter<SmoothSelection> conventional;
  const bool conventional_ready = conventional.Initialize(conventional_config);
  const cpu::CpuStateSnapshot conventional_state = conventional.Snapshot();
  checks.Require(conventional_ready &&
                     conventional_state.interior.size == oracle_interior.size() &&
                     conventional_state.trace.size == oracle_trace.size() &&
                     CloseArrays(conventional_state.interior.data,
                                 oracle_interior) &&
                     CloseArrays(conventional_state.trace.data, oracle_trace),
                 "conventional smooth U/Uhat must match independent L2 oracles");

  sc25::FoundationConfig foundation;
  foundation.polynomial_degree = SmoothCase::polynomial_order;
  foundation.elements_per_direction = static_cast<int>(SmoothCase::mesh_n);
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> scalar;
  const bool scalar_created =
      sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, sc25::CpuSerialTag{}, foundation, scalar) ==
          sc25::FoundationStatus::ready && scalar;
  if (scalar_created)
    sc25::w8c_internal::RequestSmoothTransitionInitialization(*scalar);
  const sc25::TrajectoryStatus scalar_status = scalar_created
      ? scalar->InitializePersistentTrajectory()
      : sc25::TrajectoryStatus::uninitialized;
  const bool scalar_ready = scalar_status == sc25::TrajectoryStatus::ready;
  if (!scalar_ready)
    std::cerr << "smooth scalar initialization status="
              << static_cast<int>(scalar_status) << '\n';
  sc25::PersistentTrajectoryState scalar_state;
  const bool scalar_copied = scalar_ready &&
      scalar->CopyPublishedTrajectoryState(scalar_state);
  const std::vector<double> scalar_u = scalar_copied
      ? ScalarUComponentMajor(scalar_state) : std::vector<double>{};
  checks.Require(scalar_copied && scalar_u.size() == oracle_interior.size() &&
                     CloseArrays(scalar_u.data(), oracle_interior) &&
                     CloseArrays(scalar_state.uhat.data(), oracle_trace),
                 "scalar smooth U/Uhat must match independent L2 oracles");
  checks.Require(scalar_copied && conventional_ready &&
                     scalar_u.size() == conventional_state.interior.size &&
                     scalar_state.uhat.size() == conventional_state.trace.size &&
                     std::memcmp(scalar_u.data(),
                                 conventional_state.interior.data,
                                 scalar_u.size() * sizeof(double)) == 0 &&
                     std::memcmp(scalar_state.uhat.data(),
                                 conventional_state.trace.data,
                                 scalar_state.uhat.size() * sizeof(double)) == 0,
                 "none and scalar policies must start from byte-identical U/Uhat");
  sc25::AuxiliaryResidualMetrics independent_aux_q12;
  sc25::AuxiliaryResidualMetrics independent_aux_q14;
  const bool independent_auxiliary_ready = scalar_copied &&
      dg_igr_hdg_sc25_cpu_test::EvaluateIndependentAuxiliaryResidualOracle(
          *scalar, scalar_state, 12, independent_aux_q12) &&
      dg_igr_hdg_sc25_cpu_test::EvaluateIndependentAuxiliaryResidualOracle(
          *scalar, scalar_state, 14, independent_aux_q14);
  checks.Require(scalar_copied && PositiveZero(scalar_state.sigma) &&
                     PositiveZero(scalar_state.z) &&
                     PositiveZero(scalar_state.sigmahat),
                 "smooth auxiliary fields must be homogeneous positive zero");
  checks.Require(independent_auxiliary_ready &&
                     independent_aux_q12.normalized_maximum <= 1.0e-10 &&
                     independent_aux_q14.normalized_maximum <= 1.0e-10,
                 "complete independent q12/q14 auxiliary weak residual must be homogeneous");
  const sc25::AuxiliaryInitializationEvidence* auxiliary = scalar_ready
      ? &scalar->trajectory_evidence().initialization : nullptr;
  checks.Require(auxiliary && auxiliary->quadrature_order == 12 &&
                     auxiliary->trace_restart_dimension == 50 &&
                     auxiliary->trace_maximum_iterations == 200 &&
                     auxiliary->trace_iterations >= 0 &&
                     auxiliary->trace_true_residual_norm == 0.0 &&
                     auxiliary->trace_right_hand_side_norm == 0.0 &&
                     auxiliary->residual.normalized_maximum <= 1.0e-10 &&
                     independent_auxiliary_ready &&
                     std::abs(auxiliary->residual.normalized_maximum -
                              independent_aux_q12.normalized_maximum) <=
                         2.0e-11,
                 "smooth scalar initialization must complete the full auxiliary solve");

  MPI_Finalize();
  return checks.failures == 0 ? 0 : 1;
}
