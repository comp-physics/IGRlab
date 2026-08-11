#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#include "sc25_scalar_igr_rank2_manifest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::app;
namespace cpu_manifest = dg_igr_hdg_sc25_rank2_manifest;
namespace cpu_sc25 = cpu_manifest::cpu;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "CLI test must compile as strict C++17");
static_assert(!std::is_polymorphic<ParsedCommandLine>::value &&
                  !std::is_polymorphic<ApplicationOptionsView>::value,
              "CLI and configuration use static values");

std::vector<std::string> ScalarArguments(const std::string& backend,
                                         const std::string& device);
ParsedCommandLine Parse(std::vector<std::string> values);

bool Check(const bool condition, const char* name)
{
  if (!condition) std::cerr << "FAILED: " << name << '\n';
  return condition;
}

struct ScalarVtkField final
{
  std::size_t points = 0;
  std::size_t cells = 0;
  std::vector<std::uint64_t> connectivity;
  std::vector<int> cell_types;
  std::vector<int> materials;
  std::vector<int> material_ids;
  std::vector<double> coordinates;
  std::array<std::vector<double>, 7> fields;
};

bool ParseScalarVtk(std::istream& input, ScalarVtkField& field)
{
  std::vector<std::string> tokens;
  for (std::string token; input >> token;) tokens.push_back(std::move(token));
  const auto find = [](const std::vector<std::string>& values,
                       const std::string& token, const std::size_t begin)
  {
    for (std::size_t index = begin; index < values.size(); ++index)
      if (values[index] == token) return index;
    return values.size();
  };
  try
  {
    const auto expect = [&tokens](const std::size_t index,
                                  const std::string& value)
    {
      return index < tokens.size() && tokens[index] == value;
    };
    std::size_t cursor = find(tokens, "FIELD", 0);
    if (cursor == tokens.size() || find(tokens, "FIELD", cursor + 1) !=
            tokens.size() ||
        !expect(cursor + 1, "FieldData") || !expect(cursor + 2, "1") ||
        !expect(cursor + 3, "MaterialIds") || !expect(cursor + 4, "1") ||
        !expect(cursor + 5, "1") || !expect(cursor + 6, "int") ||
        cursor + 7 >= tokens.size())
      return false;
    field.material_ids = {std::stoi(tokens[cursor + 7])};
    cursor += 8;

    if (!expect(cursor, "POINTS") || cursor + 3 > tokens.size()) return false;
    field.points = static_cast<std::size_t>(std::stoull(tokens[cursor + 1]));
    if (!expect(cursor + 2, "double")) return false;
    cursor += 3;
    field.coordinates.resize(3 * field.points);
    for (std::size_t index = 0; index < field.coordinates.size(); ++index)
    {
      if (cursor >= tokens.size()) return false;
      field.coordinates[index] = std::stod(tokens[cursor++]);
    }

    if (!expect(cursor, "CELLS") || cursor + 3 > tokens.size()) return false;
    field.cells = static_cast<std::size_t>(std::stoull(tokens[cursor + 1]));
    const std::size_t cell_words =
        static_cast<std::size_t>(std::stoull(tokens[cursor + 2]));
    if (cell_words != 5 * field.cells) return false;
    cursor += 3;
    field.connectivity.reserve(4 * field.cells);
    for (std::size_t cell = 0; cell < field.cells; ++cell)
    {
      if (cursor >= tokens.size()) return false;
      const std::size_t vertices =
          static_cast<std::size_t>(std::stoull(tokens[cursor++]));
      if (vertices != 4 || cursor + vertices > tokens.size()) return false;
      for (std::size_t vertex = 0; vertex < vertices; ++vertex)
      {
        const std::uint64_t point =
            static_cast<std::uint64_t>(std::stoull(tokens[cursor++]));
        if (point >= field.points) return false;
        field.connectivity.push_back(point);
      }
    }
    if (!expect(cursor, "CELL_TYPES") || cursor + 2 > tokens.size() ||
        static_cast<std::size_t>(std::stoull(tokens[cursor + 1])) !=
            field.cells)
      return false;
    cursor += 2;
    field.cell_types.resize(field.cells);
    for (std::size_t cell = 0; cell < field.cells; ++cell)
    {
      if (cursor >= tokens.size()) return false;
      field.cell_types[cell] = std::stoi(tokens[cursor++]);
    }

    if (!expect(cursor, "CELL_DATA") || cursor + 2 > tokens.size() ||
        static_cast<std::size_t>(std::stoull(tokens[cursor + 1])) !=
            field.cells ||
        !expect(cursor + 2, "SCALARS") || !expect(cursor + 3, "material") ||
        !expect(cursor + 4, "int") || !expect(cursor + 5, "LOOKUP_TABLE") ||
        !expect(cursor + 6, "default"))
      return false;
    cursor += 7;
    field.materials.resize(field.cells);
    for (std::size_t cell = 0; cell < field.cells; ++cell)
    {
      if (cursor >= tokens.size()) return false;
      field.materials[cell] = std::stoi(tokens[cursor++]);
    }

    // MFEM's coloring values are process-local diagnostics.  Require their
    // exact section/type/count but intentionally ignore only the values.
    if (!expect(cursor, "SCALARS") ||
        !expect(cursor + 1, "element_coloring") ||
        !expect(cursor + 2, "int") ||
        !expect(cursor + 3, "LOOKUP_TABLE") ||
        !expect(cursor + 4, "default"))
      return false;
    cursor += 5;
    for (std::size_t cell = 0; cell < field.cells; ++cell)
    {
      if (cursor >= tokens.size()) return false;
      (void)std::stoi(tokens[cursor++]);
    }

    if (!expect(cursor, "POINT_DATA") || cursor + 2 > tokens.size() ||
        static_cast<std::size_t>(std::stoull(tokens[cursor + 1])) !=
            field.points)
      return false;
    cursor += 2;
    constexpr std::array<const char*, 5> names{{"U0", "U1", "U2", "U3",
                                                "Sigma"}};
    for (std::size_t component = 0; component < names.size(); ++component)
    {
      if (!expect(cursor, "SCALARS") ||
          !expect(cursor + 1, names[component]) ||
          !expect(cursor + 2, "double") || !expect(cursor + 3, "1") ||
          !expect(cursor + 4, "LOOKUP_TABLE") ||
          !expect(cursor + 5, "default"))
        return false;
      cursor += 6;
      field.fields[component].resize(field.points);
      for (std::size_t index = 0; index < field.points; ++index)
      {
        if (cursor >= tokens.size()) return false;
        field.fields[component][index] = std::stod(tokens[cursor++]);
      }
    }
    if (!expect(cursor, "VECTORS") || !expect(cursor + 1, "z") ||
        !expect(cursor + 2, "double"))
      return false;
    cursor += 3;
    field.fields[5].resize(field.points);
    field.fields[6].resize(field.points);
    for (std::size_t index = 0; index < field.points; ++index)
    {
      if (cursor + 3 > tokens.size()) return false;
      field.fields[5][index] = std::stod(tokens[cursor++]);
      field.fields[6][index] = std::stod(tokens[cursor++]);
      if (std::stod(tokens[cursor++]) != 0.0) return false;
    }
    if (cursor != tokens.size()) return false;
  }
  catch (...) { return false; }
  return field.points > 0 && field.cells > 0;
}

bool ParseScalarVtk(const char* path, ScalarVtkField& field)
{
  std::ifstream input(path);
  return input && ParseScalarVtk(input, field);
}

bool ParseScalarVtk(const std::vector<std::uint8_t>& bytes,
                    ScalarVtkField& field)
{
  const std::string text(bytes.begin(), bytes.end());
  std::istringstream input{text};
  return ParseScalarVtk(input, field);
}

bool CompareScalarVtk(const char* left_path, const char* right_path,
                      const double tolerance,
                      const bool unit_norm_floor = false)
{
  ScalarVtkField left, right;
  if (!ParseScalarVtk(left_path, left) || !ParseScalarVtk(right_path, right) ||
      left.points != right.points || left.cells != right.cells ||
      left.coordinates.size() != right.coordinates.size() ||
      left.connectivity != right.connectivity ||
      left.cell_types != right.cell_types ||
      left.materials != right.materials ||
      left.material_ids != right.material_ids)
    return false;
  for (std::size_t index = 0; index < left.coordinates.size(); ++index)
    if (left.coordinates[index] != right.coordinates[index]) return false;
  for (std::size_t component = 0; component < left.fields.size(); ++component)
  {
    long double difference_squared = 0.0L;
    long double reference_squared = 0.0L;
    for (std::size_t index = 0; index < left.points; ++index)
    {
      const double a = left.fields[component][index];
      const double b = right.fields[component][index];
      if (!std::isfinite(a) || !std::isfinite(b)) return false;
      const long double difference = static_cast<long double>(a) - b;
      difference_squared += difference * difference;
      reference_squared += static_cast<long double>(b) * b;
    }
    const long double difference_norm = std::sqrt(difference_squared);
    const long double reference_norm = std::sqrt(reference_squared);
    const long double error = unit_norm_floor
        ? difference_norm / std::max(1.0L, reference_norm)
        : (reference_norm == 0.0L
            ? difference_norm : difference_norm / reference_norm);
    if (!std::isfinite(error) || error > tolerance) return false;
  }
  return true;
}

bool ScalarNear(const double actual, const double expected,
                const double tolerance)
{
  return std::isfinite(actual) && std::isfinite(expected) &&
      std::abs(actual - expected) <=
          tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

std::array<double, 4> SmoothConservativeOracle(const double x,
                                                const double y)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double gamma = 1.4;
  const double rho = 1.0 + 0.05 * std::sin(2.0 * pi * x) *
      std::sin(2.0 * pi * y);
  const double u = 0.1 * std::sin(2.0 * pi * x);
  const double v = 0.1 * std::sin(2.0 * pi * y);
  return {{rho, rho * u, rho * v,
           1.0 / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)}};
}

bool ValidateScalarInitialVtk(const char* path, const double tolerance)
{
  ScalarVtkField field;
  if (!ParseScalarVtk(path, field) || field.points != 256 ||
      field.cells != 64 || field.connectivity.size() != 256 ||
      field.cell_types.size() != 64 || field.materials.size() != 64 ||
      field.material_ids != std::vector<int>{1})
    return false;

  constexpr int n = 8;
  for (std::size_t cell = 0; cell < field.cells; ++cell)
  {
    const int ex = static_cast<int>(cell % n);
    const int ey = static_cast<int>(cell / n);
    constexpr std::array<int, 4> vtk_order{{0, 1, 3, 2}};
    for (int vertex = 0; vertex < 4; ++vertex)
    {
      const std::size_t point = 4 * cell + static_cast<std::size_t>(vertex);
      if (field.connectivity[4 * cell + static_cast<std::size_t>(vertex)] !=
              4 * cell + static_cast<std::size_t>(vtk_order[vertex]) ||
          field.cell_types[cell] != 9 || field.materials[cell] != 1)
        return false;
      const int dx = vertex == 1 || vertex == 3 ? 1 : 0;
      const int dy = vertex >= 2 ? 1 : 0;
      if (!ScalarNear(field.coordinates[3 * point],
                      static_cast<double>(ex + dx) / n, 1.0e-15) ||
          !ScalarNear(field.coordinates[3 * point + 1],
                      static_cast<double>(ey + dy) / n, 1.0e-15) ||
          field.coordinates[3 * point + 2] != 0.0)
        return false;
    }
  }

  // Independent seven-point Gauss-Legendre projection of the analytic
  // conservative initial state onto each p=1 L2 element, followed by exact
  // evaluation at the four VTK corners.  This shares neither PackScalarIgrState
  // nor MakeScalarIgrVtk indexing with the runtime.
  constexpr std::array<std::array<double, 2>, 7> quadrature{{
      {{0.025446043828620757, 0.064742483084435323}},
      {{0.12923440720030277, 0.13985269574463829}},
      {{0.29707742431130141, 0.19091502525255916}},
      {{0.5, 0.20897959183673448}},
      {{0.70292257568869854, 0.19091502525255916}},
      {{0.87076559279969723, 0.13985269574463829}},
      {{0.97455395617137919, 0.064742483084435323}}}};
  constexpr double root_three = 1.73205080756887729352744634150587236;
  const double node0 = 0.5 * (1.0 - 1.0 / root_three);
  const double node1 = 0.5 * (1.0 + 1.0 / root_three);
  const auto basis = [node0, node1](const int node, const double x)
  {
    return node == 0 ? (node1 - x) / (node1 - node0)
                     : (x - node0) / (node1 - node0);
  };
  for (int ey = 0; ey < n; ++ey)
    for (int ex = 0; ex < n; ++ex)
    {
      const std::size_t cell = static_cast<std::size_t>(ey * n + ex);
      std::array<std::array<double, 4>, 4> coefficients{};
      for (int component = 0; component < 4; ++component)
        for (int jy = 0; jy < 2; ++jy)
          for (int ix = 0; ix < 2; ++ix)
            for (const auto& qy : quadrature)
              for (const auto& qx : quadrature)
                coefficients[static_cast<std::size_t>(component)]
                            [static_cast<std::size_t>(2 * jy + ix)] +=
                    4.0 * qx[1] * qy[1] * basis(ix, qx[0]) *
                    basis(jy, qy[0]) *
                    SmoothConservativeOracle(
                        (static_cast<double>(ex) + qx[0]) / n,
                        (static_cast<double>(ey) + qy[0]) / n)
                        [static_cast<std::size_t>(component)];
      for (int vertex = 0; vertex < 4; ++vertex)
      {
        const int vx = vertex == 1 || vertex == 3 ? 1 : 0;
        const int vy = vertex >= 2 ? 1 : 0;
        const std::size_t point = 4 * cell + static_cast<std::size_t>(vertex);
        for (int component = 0; component < 4; ++component)
        {
          double expected = 0.0;
          for (int jy = 0; jy < 2; ++jy)
            for (int ix = 0; ix < 2; ++ix)
              expected += coefficients[static_cast<std::size_t>(component)]
                                      [static_cast<std::size_t>(2 * jy + ix)] *
                  basis(ix, static_cast<double>(vx)) *
                  basis(jy, static_cast<double>(vy));
          if (!ScalarNear(field.fields[static_cast<std::size_t>(component)]
                              [point], expected, tolerance))
            return false;
        }
      }
    }

  // Independent formulation probes protect the Sigma/z field association;
  // broad finite/range checks make a U or component block shift fail loudly.
  for (const double sigma : field.fields[4])
    if (!std::isfinite(sigma) || sigma < 0.07 || sigma > 0.10) return false;
  for (std::size_t component = 5; component < 7; ++component)
    for (const double value : field.fields[component])
      if (!std::isfinite(value) || std::abs(value) > 0.08) return false;
  return ScalarNear(field.fields[4][0], 0.08968703595877049, tolerance) &&
      ScalarNear(field.fields[5][4], -0.06848392124134046, tolerance) &&
      ScalarNear(field.fields[6][4], -0.00973388369071104, tolerance);
}

struct CliScalarResourceRecord final
{
  StateHash mesh{};
  StateHash partition{};
  StateHash topology{};
  std::array<std::uint8_t, 16> gpu_uuid{};
  CheckpointDescriptor private_descriptor{};
  cpu_sc25::Rank2TrajectoryStateHashes raw_field_hashes{};
  StateHash shard_manifest_hash{};
  StateHash canonical_manifest_hash{};
};

struct CliScalarResourceSidecar final
{
  std::array<char, 16> magic{{'D','G','I','G','R','-','S','C','2','5','-','R','E','S','1',0}};
  std::uint32_t version = 1;
  std::uint32_t rank_count = 0;
  std::array<CliScalarResourceRecord, 2> ranks{};
  StateHash checksum{};
};

static_assert(std::is_trivially_copyable<CliScalarResourceSidecar>::value,
              "CLI sidecar decoder requires the fixed application layout");

bool ReadCurrentArtifact(const std::filesystem::path& path,
                         const std::size_t maximum,
                         std::vector<std::uint8_t>& bytes)
{
  std::string diagnostic;
  return ::dg::hdg::runtime::app::detail::ReadCurrentGenerationArtifact(
             path, maximum, bytes, diagnostic) == DurableIoStatus::ok;
}

bool ReadDurableArtifact(const std::filesystem::path& path,
                         const std::size_t maximum,
                         std::vector<std::uint8_t>& bytes)
{
  std::string detail;
  return ReadDurableFile(path, maximum, bytes, detail) == DurableIoStatus::ok;
}

bool DigestIs(const std::vector<std::uint8_t>& bytes,
              const std::string_view expected)
{
  cpu_manifest::Digest digest{};
  return cpu_manifest::ParseHexDigest(expected, digest) &&
      cpu_manifest::DigestEqual(
          ComputeSha256(bytes.data(), bytes.size()), digest);
}

bool LoadCpuManifests(
    const std::string_view directory_path, const std::string_view run_id,
    const std::array<std::string_view, 3>& expected_hashes,
    std::array<cpu_manifest::CpuReferenceManifest, 2>& manifests)
{
  cpu_manifest::DirectoryHandle directory;
  cpu_manifest::ManifestSetIdentity before{}, after{};
  if (!directory.Open(directory_path) ||
      !cpu_manifest::CaptureManifestSetIdentity(directory, before))
    return false;
  std::vector<std::uint8_t> complete_bytes;
  std::array<std::vector<std::uint8_t>, 2> shard_bytes;
  if (!cpu_manifest::ReadFileAt(
          directory, cpu_manifest::complete_filename, complete_bytes) ||
      !cpu_manifest::ReadFileAt(directory, "rank-0.c3m", shard_bytes[0]) ||
      !cpu_manifest::ReadFileAt(directory, "rank-1.c3m", shard_bytes[1]) ||
      !DigestIs(complete_bytes, expected_hashes[0]) ||
      !DigestIs(shard_bytes[0], expected_hashes[1]) ||
      !DigestIs(shard_bytes[1], expected_hashes[2]))
    return false;
  cpu_manifest::CompleteManifest complete{};
  if (!cpu_manifest::DecodeComplete(
          complete_bytes, run_id, directory.identity(), complete))
    return false;
  for (std::size_t rank = 0; rank < manifests.size(); ++rank)
  {
    const auto& entry = complete.shards[rank];
    if (entry.filename != cpu_manifest::RankFilename(static_cast<int>(rank)) ||
        entry.bytes != shard_bytes[rank].size() ||
        !cpu_manifest::DigestEqual(
            entry.sha256,
            ComputeSha256(shard_bytes[rank].data(), shard_bytes[rank].size())) ||
        !cpu_manifest::DecodeCpuManifest(
            shard_bytes[rank], run_id, complete.executable_sha256,
            static_cast<int>(rank), directory.identity(), manifests[rank]))
      return false;
  }
  return cpu_manifest::CaptureManifestSetIdentity(directory, after) &&
      before == after;
}

bool HandoffPathNegativeControls()
{
  const std::string suffix =
      std::to_string(static_cast<unsigned long long>(::getpid()));
  const std::filesystem::path root =
      std::filesystem::path{"/tmp"} / ("dg-igr-hdg-handoff-negative-" + suffix);
  const std::filesystem::path alias = root.string() + "-alias";
  std::error_code error;
  std::filesystem::remove_all(root, error);
  error.clear();
  std::filesystem::remove(alias, error);
  error.clear();
  if (!std::filesystem::create_directory(root, error) || error ||
      ::chmod(root.c_str(), S_IRWXU) != 0)
    return false;

  bool valid = false;
  const std::filesystem::path regular = root / "regular";
  const std::filesystem::path manifest = root / "rank-0.c3m";
  {
    std::ofstream output(regular, std::ios::binary);
    output << "not-a-manifest";
  }
  if (::chmod(regular.c_str(), S_IRUSR | S_IWUSR) == 0)
  {
    std::filesystem::create_symlink(regular.filename(), manifest, error);
    cpu_manifest::DirectoryHandle directory;
    std::vector<std::uint8_t> bytes;
    valid = !error && directory.Open(root.string()) &&
        !cpu_manifest::ReadFileAt(directory, "rank-0.c3m", bytes);
    std::filesystem::remove(manifest, error);
    error.clear();
    valid = valid && ::mkfifo(manifest.c_str(), S_IRUSR | S_IWUSR) == 0 &&
        !cpu_manifest::ReadFileAt(directory, "rank-0.c3m", bytes);
    std::filesystem::remove(manifest, error);
    error.clear();
    std::filesystem::create_directory_symlink(root, alias, error);
    cpu_manifest::DirectoryHandle symlinked_directory;
    valid = valid && !error && !symlinked_directory.Open(alias.string());
  }
  std::filesystem::remove(alias, error);
  error.clear();
  std::filesystem::remove_all(root, error);
  return valid && !error;
}

bool RelativeL2Within(const std::vector<double>& actual,
                      const std::vector<double>& reference,
                      const double tolerance)
{
  if (actual.size() != reference.size() || actual.empty()) return false;
  long double difference_squared = 0.0L;
  long double reference_squared = 0.0L;
  for (std::size_t index = 0; index < actual.size(); ++index)
  {
    if (!std::isfinite(actual[index]) || !std::isfinite(reference[index]))
      return false;
    const long double difference =
        static_cast<long double>(actual[index]) - reference[index];
    difference_squared += difference * difference;
    reference_squared += static_cast<long double>(reference[index]) *
        reference[index];
  }
  const long double difference_norm = std::sqrt(difference_squared);
  const long double reference_norm = std::sqrt(reference_squared);
  const long double error = reference_norm == 0.0L
      ? difference_norm : difference_norm / reference_norm;
  return std::isfinite(error) && error <= tolerance;
}

double ManifestInteriorCoefficient(
    const cpu_sc25::PersistentRank2TrajectoryState& state,
    const int component, const int dof)
{
  const int element = dof / 4;
  const int node = dof % 4;
  if (component < 4)
    return state.u[static_cast<std::size_t>(element * 16 + component * 4 + node)];
  if (component == 4)
    return state.sigma[static_cast<std::size_t>(element * 4 + node)];
  return state.z[static_cast<std::size_t>(
      element * 8 + (component - 5) * 4 + node)];
}

bool CompareScalarVtkToCpuManifest(
    const std::filesystem::path& path,
    const std::array<cpu_manifest::CpuReferenceManifest, 2>& manifests,
    const int step, const double tolerance,
    StateHash& artifact_hash)
{
  if (step < 0 || step > cpu_sc25::persistent_required_steps) return false;
  std::vector<std::uint8_t> bytes;
  ScalarVtkField field;
  if (!ReadDurableArtifact(path, 32U * 1024U * 1024U, bytes) ||
      !ParseScalarVtk(bytes, field) || field.points != 256 ||
      field.cells != 64 || field.connectivity.size() != 256)
    return false;
  artifact_hash.bytes = ComputeSha256(bytes.data(), bytes.size());
  if (artifact_hash == StateHash{}) return false;

  constexpr double root_three = 1.73205080756887729352744634150587236;
  const double node0 = 0.5 * (1.0 - 1.0 / root_three);
  const double node1 = 0.5 * (1.0 + 1.0 / root_three);
  const auto basis = [node0, node1](const int node, const double x)
  {
    return node == 0 ? (node1 - x) / (node1 - node0)
                     : (x - node0) / (node1 - node0);
  };
  std::array<std::vector<double>, 7> expected;
  for (auto& values : expected) values.resize(field.points);
  for (int ey = 0; ey < 8; ++ey)
    for (int ex = 0; ex < 8; ++ex)
    {
      const int rank = ex < 4 ? 0 : 1;
      const int local_element = ey * 4 + ex % 4;
      const auto& record = manifests[static_cast<std::size_t>(rank)]
                               .steps[static_cast<std::size_t>(step)];
      if (record.summary.coordinates.step != static_cast<std::uint64_t>(step) ||
          record.summary.coordinates.generation !=
              static_cast<std::uint64_t>(step) ||
          record.summary.coordinates.cursor != static_cast<std::uint64_t>(step))
        return false;
      for (int vertex = 0; vertex < 4; ++vertex)
      {
        const int vx = vertex == 1 || vertex == 3 ? 1 : 0;
        const int vy = vertex >= 2 ? 1 : 0;
        const std::size_t point = static_cast<std::size_t>(4 * (ey * 8 + ex) + vertex);
        if (field.connectivity[point] !=
            static_cast<std::uint64_t>(4 * (ey * 8 + ex) +
                std::array<int, 4>{{0, 1, 3, 2}}[static_cast<std::size_t>(vertex)]))
          return false;
        for (int component = 0; component < 7; ++component)
        {
          double value = 0.0;
          for (int jy = 0; jy < 2; ++jy)
            for (int ix = 0; ix < 2; ++ix)
              value += ManifestInteriorCoefficient(
                           record.state, component,
                           local_element * 4 + 2 * jy + ix) *
                  basis(ix, static_cast<double>(vx)) *
                  basis(jy, static_cast<double>(vy));
          expected[static_cast<std::size_t>(component)][point] = value;
        }
      }
    }
  for (std::size_t component = 0; component < field.fields.size(); ++component)
    if (!RelativeL2Within(field.fields[component], expected[component], tolerance))
      return false;
  return true;
}

StateHash HashRanges(const std::initializer_list<std::pair<const double*,
                                                           std::size_t>>& ranges)
{
  Sha256 hash;
  Sha256Digest digest{};
  for (const auto& range : ranges)
    if (!hash.Update(range.first, range.second * sizeof(double))) return {};
  return hash.Finalize(digest) ? StateHash{digest} : StateHash{};
}

bool ValidCliSidecar(const CliScalarResourceSidecar& sidecar)
{
  const CliScalarResourceSidecar expected{};
  if (sidecar.magic != expected.magic || sidecar.version != 1 ||
      sidecar.rank_count != 2 || sidecar.checksum == StateHash{} ||
      sidecar.checksum.bytes != ComputeSha256(
          &sidecar, offsetof(CliScalarResourceSidecar, checksum)))
    return false;
  for (const auto& record : sidecar.ranks)
  {
    if (record.mesh == StateHash{} || record.partition == StateHash{} ||
        record.topology == StateHash{} ||
        std::all_of(record.gpu_uuid.begin(), record.gpu_uuid.end(),
                    [](const std::uint8_t byte) { return byte == 0; }) ||
        record.shard_manifest_hash == StateHash{} ||
        record.canonical_manifest_hash == StateHash{} ||
        record.private_descriptor.rank_count != 2 ||
        record.private_descriptor.coordinates.step != 10 ||
        record.private_descriptor.coordinates.generation != 10 ||
        record.private_descriptor.coordinates.cursor != 10 ||
        record.private_descriptor.coordinates.time !=
            10 * cpu_sc25::persistent_dt ||
        record.private_descriptor.coordinates.dt != cpu_sc25::persistent_dt)
      return false;
  }
  return sidecar.ranks[0].gpu_uuid == sidecar.ranks[1].gpu_uuid &&
      sidecar.ranks[0].canonical_manifest_hash ==
          sidecar.ranks[1].canonical_manifest_hash;
}

bool ValidCliSidecarForConfig(const CliScalarResourceSidecar& sidecar)
{
  if (!ValidCliSidecar(sidecar)) return false;
  const RuntimeTupleDescriptor private_tuple{
      StableIdentity::Copy(cpu_sc25::projection_id),
      StableIdentity::Copy(cpu_sc25::object_id),
      StableIdentity::Copy("DualConservativeScalarTraceV1"),
      StableIdentity::Copy("W7Tau25.DensityHeightP2.v3"),
      StableIdentity::Copy("BackwardEulerMonolithicV1"),
      StableIdentity::Copy("CudaRank2OneGpuHostStaged"),
      StableIdentity::Copy("Fp64")};
  for (const auto& record : sidecar.ranks)
  {
    const auto& descriptor = record.private_descriptor;
    const auto& hashes = record.raw_field_hashes;
    if (descriptor.schema_version != checkpoint_schema_version ||
        !descriptor.successful_commit ||
        descriptor.restart_mode !=
            RestartMode::same_backend_rank_partition ||
        descriptor.tuple != private_tuple ||
        descriptor.mfem_pin != StableIdentity::Copy(cpu_manifest::mfem_pin) ||
        descriptor.mesh_hash != record.mesh ||
        descriptor.partition_hash != record.partition ||
        descriptor.topology_hash != record.topology ||
        descriptor.polynomial_order != 1 ||
        descriptor.layout != StableIdentity::Copy(
            "device.field-blocked.U[e][c][a].Sigma[e][a].z[e][c][a].v3") ||
        descriptor.precision != StableIdentity::Copy("Fp64") ||
        descriptor.coordinate_fingerprint !=
            ::dg::hdg::runtime::detail::CoordinateFingerprint(
                descriptor.coordinates) ||
        hashes.u == StateHash{} || hashes.sigma == StateHash{} ||
        hashes.z == StateHash{} || hashes.uhat == StateHash{} ||
        hashes.sigmahat == StateHash{} || hashes.interior == StateHash{} ||
        hashes.trace == StateHash{} || hashes.shard == StateHash{})
      return false;
  }
  return true;
}

PersistentResourceDescriptor CliScalarResources(
    const CliScalarResourceSidecar& sidecar,
    const ValidatedApplicationConfig& config)
{
  PersistentResourceDescriptor resources{};
  resources.mfem_pin = StableIdentity::Copy(cpu_manifest::mfem_pin);
  std::array<StateHash, 2> ordered{};
  for (std::size_t rank = 0; rank < ordered.size(); ++rank)
    ordered[rank] = sidecar.ranks[rank].mesh;
  resources.mesh_hash.bytes = ComputeSha256(ordered.data(), sizeof(ordered));
  for (std::size_t rank = 0; rank < ordered.size(); ++rank)
    ordered[rank] = sidecar.ranks[rank].partition;
  resources.partition_hash.bytes = ComputeSha256(ordered.data(), sizeof(ordered));
  for (std::size_t rank = 0; rank < ordered.size(); ++rank)
    ordered[rank] = sidecar.ranks[rank].topology;
  resources.topology_hash.bytes = ComputeSha256(ordered.data(), sizeof(ordered));
  resources.rank_count = 2;
  resources.polynomial_order = config.polynomial_order();
  resources.layout = StableIdentity::Copy(config.layout_id().view());
  resources.reuse = {true, true, true, false, 1, 1, 1, 1};
  return resources;
}

CheckpointRuntimeIdentityV1 CliScalarRuntimeIdentity(
    const CliScalarResourceSidecar& sidecar,
    const PersistentResourceDescriptor& resources)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string uuid = "GPU-";
  uuid.resize(36);
  for (std::size_t index = 0; index < sidecar.ranks[0].gpu_uuid.size(); ++index)
  {
    uuid[4 + 2 * index] = digits[sidecar.ranks[0].gpu_uuid[index] >> 4U];
    uuid[5 + 2 * index] = digits[sidecar.ranks[0].gpu_uuid[index] & 0xfU];
  }
  CheckpointRuntimeIdentityV1 identity{};
  identity.physical_gpu_uuid = StableIdentity::Copy(uuid);
  identity.ownership_hash = resources.partition_hash;
  return identity;
}

std::optional<ValidatedApplicationConfig> ScalarRank2ReaderConfig(
    const std::string& output_directory, const std::string& checkpoint)
{
  std::vector<std::string> arguments = ScalarArguments("cuda", "0");
  for (std::size_t index = 1; index + 1 < arguments.size(); index += 2)
  {
    if (arguments[index] == "--output-dir") arguments[index + 1] = output_directory;
    if (arguments[index] == "--checkpoint-out") arguments[index + 1] = "none";
    if (arguments[index] == "--checkpoint-step") arguments[index + 1] = "none";
    if (arguments[index] == "--stop-after-checkpoint") arguments[index + 1] = "false";
    if (arguments[index] == "--restart-from") arguments[index + 1] = checkpoint;
    if (arguments[index] == "--max-retries") arguments[index + 1] = "0";
    if (arguments[index] == "--inject-failure") arguments[index + 1] = "off";
  }
  const ParsedCommandLine parsed = Parse(std::move(arguments));
  ApplicationOptionsView options = parsed.options;
  options.backend_id = CudaRank2OneGpuHostStagedBackendTag::id;
  const ApplicationValidation validation = ValidateApplicationOptions(options);
  return validation.accepted()
      ? std::optional<ValidatedApplicationConfig>{*validation.config()}
      : std::nullopt;
}

bool CheckDecodedRankAgainstManifest(
    const std::vector<double>& interior, const std::vector<double>& trace,
    const cpu_manifest::CpuReferenceManifest& manifest, const int rank,
    const double tolerance)
{
  const auto& state = manifest.steps[10].state;
  if (interior.size() != 896 || trace.size() != (rank == 0 ? 720U : 560U))
    return false;
  for (int component = 0; component < 7; ++component)
  {
    std::vector<double> actual(128), reference(128);
    for (int dof = 0; dof < 128; ++dof)
    {
      actual[static_cast<std::size_t>(dof)] =
          interior[static_cast<std::size_t>(component * 128 + dof)];
      reference[static_cast<std::size_t>(dof)] =
          ManifestInteriorCoefficient(state, component, dof);
    }
    if (!RelativeL2Within(actual, reference, tolerance)) return false;
  }
  const std::size_t nodes = rank == 0 ? 144U : 112U;
  for (std::size_t component = 0; component < 5; ++component)
  {
    std::vector<double> actual(nodes), reference(nodes);
    for (std::size_t dof = 0; dof < nodes; ++dof)
    {
      actual[dof] = trace[component * nodes + dof];
      reference[dof] = component < 4
          ? state.uhat[component * nodes + dof]
          : state.sigmahat[dof];
    }
    if (!RelativeL2Within(actual, reference, tolerance)) return false;
  }
  return true;
}

bool CheckDecodedRankHashes(
    const std::vector<double>& interior, const std::vector<double>& trace,
    const CliScalarResourceRecord& record)
{
  std::vector<double> fields;
  if (!::dg::hdg::runtime::app::detail::
           ScalarIgrComponentMajorToFieldBlocksV1(interior, 4, fields) ||
      fields.size() != 896)
    return false;
  const double* const u = fields.data();
  const double* const sigma = u + 512;
  const double* const z = sigma + 128;
  const std::size_t trace_nodes = trace.size() / 5;
  const double* const uhat = trace.data();
  const double* const sigmahat = uhat + 4 * trace_nodes;
  const auto& hashes = record.raw_field_hashes;
  return hashes.u == HashRanges({{u, 512}}) &&
      hashes.sigma == HashRanges({{sigma, 128}}) &&
      hashes.z == HashRanges({{z, 256}}) &&
      hashes.uhat == HashRanges({{uhat, 4 * trace_nodes}}) &&
      hashes.sigmahat == HashRanges({{sigmahat, trace_nodes}}) &&
      hashes.interior == HashRanges({{u, 512}, {sigma, 128}, {z, 256}}) &&
      hashes.trace == HashRanges({{uhat, 4 * trace_nodes},
                                  {sigmahat, trace_nodes}}) &&
      hashes.shard == HashRanges({{u, 512}, {sigma, 128}, {z, 256},
                                  {uhat, 4 * trace_nodes},
                                  {sigmahat, trace_nodes}}) &&
      record.private_descriptor.interior_hash == hashes.interior &&
      record.private_descriptor.trace_hash == hashes.trace;
}

bool ValidateScalarRank2Artifacts(
    const std::array<std::string_view, 3>& handoff,
    const std::string_view run_id,
    const std::array<std::string_view, 3>& expected_hashes,
    const std::filesystem::path& output_directory,
    const std::filesystem::path& checkpoint)
{
  std::array<cpu_manifest::CpuReferenceManifest, 2> manifests{};
  if (!LoadCpuManifests(handoff[0], run_id, expected_hashes, manifests))
    return false;
  StateHash vtk0{}, vtk10{};
  if (!CompareScalarVtkToCpuManifest(
          output_directory / "state-step-000000.vtk", manifests, 0, 1.0e-10,
          vtk0) ||
      !CompareScalarVtkToCpuManifest(
          output_directory / "state-step-000010.vtk", manifests, 10, 1.0e-10,
          vtk10))
    return false;

  if (checkpoint == "none")
  {
    StateHash vtk20{};
    std::vector<std::uint8_t> metadata_bytes;
    if (!CompareScalarVtkToCpuManifest(
            output_directory / "state-step-000020.vtk", manifests, 20,
            1.0e-10, vtk20) ||
        !ReadCurrentArtifact(output_directory / "run-metadata.json",
                             1024U * 1024U, metadata_bytes))
      return false;
    const StateHash metadata_hash{
        ComputeSha256(metadata_bytes.data(), metadata_bytes.size())};
    if (metadata_hash == StateHash{}) return false;
    std::cout << "SCALAR_RANK2_TRAJECTORY_BINDING_PASS"
              << " vtk0_sha256=" << cpu_manifest::Hex(vtk0.bytes)
              << " vtk10_sha256=" << cpu_manifest::Hex(vtk10.bytes)
              << " vtk20_sha256=" << cpu_manifest::Hex(vtk20.bytes)
              << " metadata_sha256="
              << cpu_manifest::Hex(metadata_hash.bytes) << '\n';
    return true;
  }

  std::vector<std::uint8_t> checkpoint_bytes, sidecar_bytes, metadata_bytes;
  if (!ReadCurrentArtifact(checkpoint, 64U * 1024U * 1024U,
                           checkpoint_bytes) ||
      !ReadCurrentArtifact(output_directory / "checkpoint-runtime-identity.v1",
                           sizeof(CliScalarResourceSidecar), sidecar_bytes) ||
      sidecar_bytes.size() != sizeof(CliScalarResourceSidecar) ||
      !ReadCurrentArtifact(output_directory / "run-metadata-checkpoint.json",
                           1024U * 1024U, metadata_bytes))
    return false;
  CliScalarResourceSidecar sidecar{};
  std::memcpy(&sidecar, sidecar_bytes.data(), sizeof(sidecar));
  const auto config = ScalarRank2ReaderConfig(
      output_directory.string(), checkpoint.string());
  if (!config || !ValidCliSidecarForConfig(sidecar)) return false;
  const PersistentResourceDescriptor resources =
      CliScalarResources(sidecar, *config);
  const CheckpointRuntimeIdentityV1 runtime =
      CliScalarRuntimeIdentity(sidecar, resources);
  ScenarioCoefficientExtentsV1 extents{};
  if (!DeriveScenarioCoefficientExtentsV1(*config, 2, extents) ||
      extents.interior_component_count != 7 ||
      extents.trace_component_count != 5)
    return false;
  std::array<std::vector<double>, 2> interiors, traces;
  std::array<RankStateBufferV1, 2> buffers{};
  for (std::size_t rank = 0; rank < buffers.size(); ++rank)
  {
    const auto& shape = extents.ranks[rank];
    interiors[rank].resize(shape.interior_coefficient_count);
    traces[rank].resize(shape.trace_coefficient_count);
    buffers[rank] = {
        static_cast<std::uint32_t>(rank),
        {interiors[rank].data(), interiors[rank].size(), interiors[rank].size(),
         7, shape.interior_dofs_per_component,
         CoefficientOrderingV1::component_major},
        {traces[rank].data(), traces[rank].size(), traces[rank].size(), 5,
         shape.trace_dofs_per_component,
         CoefficientOrderingV1::component_major}};
  }
  DecodedCheckpointMetadataV1 metadata{};
  if (DecodeCheckpointV1(
          {checkpoint_bytes.data(), checkpoint_bytes.size()}, *config,
          resources, runtime, buffers.data(), buffers.size(), metadata) !=
          CheckpointCodecStatus::ok ||
      metadata.rank_count != 2 || metadata.descriptor.coordinates.step != 10 ||
      metadata.descriptor.coordinates.generation != 10 ||
      metadata.descriptor.coordinates.cursor != 10 ||
      metadata.aggregate_interior_coefficients != 1792 ||
      metadata.aggregate_true_trace_coefficients != 1280)
    return false;
  for (std::size_t rank = 0; rank < buffers.size(); ++rank)
    if (buffers[rank].rank != rank ||
        buffers[rank].interior.ordering !=
            CoefficientOrderingV1::component_major ||
        buffers[rank].owned_true_trace.ordering !=
            CoefficientOrderingV1::component_major ||
        !CheckDecodedRankAgainstManifest(interiors[rank], traces[rank],
                                         manifests[rank],
                                         static_cast<int>(rank), 1.0e-10) ||
        !CheckDecodedRankHashes(interiors[rank], traces[rank],
                                sidecar.ranks[rank]))
      return false;
  const StateHash checkpoint_hash{
      ComputeSha256(checkpoint_bytes.data(), checkpoint_bytes.size())};
  const StateHash sidecar_hash{
      ComputeSha256(sidecar_bytes.data(), sidecar_bytes.size())};
  const StateHash metadata_hash{
      ComputeSha256(metadata_bytes.data(), metadata_bytes.size())};
  if (checkpoint_hash == StateHash{} || sidecar_hash == StateHash{} ||
      metadata_hash == StateHash{})
    return false;
  std::cout << "SCALAR_RANK2_ARTIFACT_BINDING_PASS"
            << " vtk0_sha256=" << cpu_manifest::Hex(vtk0.bytes)
            << " vtk10_sha256=" << cpu_manifest::Hex(vtk10.bytes)
            << " checkpoint_sha256=" << cpu_manifest::Hex(checkpoint_hash.bytes)
            << " sidecar_sha256=" << cpu_manifest::Hex(sidecar_hash.bytes)
            << " metadata_sha256=" << cpu_manifest::Hex(metadata_hash.bytes)
            << '\n';
  return true;
}

bool MakeScalarSidecarNegative(const std::filesystem::path& source,
                               const std::filesystem::path& destination,
                               const std::string_view kind)
{
  std::vector<std::uint8_t> checkpoint, sidecar_bytes;
  if (!ReadCurrentArtifact(source / "restart-checkpoint.v2",
                           64U * 1024U * 1024U, checkpoint) ||
      !ReadCurrentArtifact(source / "checkpoint-runtime-identity.v1",
                           sizeof(CliScalarResourceSidecar), sidecar_bytes) ||
      sidecar_bytes.size() != sizeof(CliScalarResourceSidecar))
    return false;
  CliScalarResourceSidecar sidecar{};
  std::memcpy(&sidecar, sidecar_bytes.data(), sizeof(sidecar));
  if (!ValidCliSidecar(sidecar)) return false;
  if (kind == "count-zero") sidecar.rank_count = 0;
  else if (kind == "count-three") sidecar.rank_count = 3;
  else if (kind == "count-maximum")
    sidecar.rank_count = std::numeric_limits<std::uint32_t>::max();
  else if (kind == "checksum") sidecar.checksum.bytes[0] ^= 1U;
  else if (kind == "shape") sidecar.ranks[0].private_descriptor.rank_count = 1;
  else return false;
  if (kind != "checksum")
    sidecar.checksum.bytes = ComputeSha256(
        &sidecar, offsetof(CliScalarResourceSidecar, checksum));
  const auto* const first = reinterpret_cast<const std::uint8_t*>(&sidecar);
  sidecar_bytes.assign(first, first + sizeof(sidecar));
  std::string detail;
  DurableGenerationTransactionV1 transaction;
  return CreateDurableDirectory(destination, detail) == DurableIoStatus::ok &&
      transaction.Begin(destination, 10, detail) == DurableIoStatus::ok &&
      transaction.Stage("restart-checkpoint.v2", checkpoint.data(),
                        checkpoint.size(), detail) == DurableIoStatus::ok &&
      transaction.Stage("checkpoint-runtime-identity.v1", sidecar_bytes.data(),
                        sidecar_bytes.size(), detail) == DurableIoStatus::ok &&
      transaction.Publish(detail) == DurableIoStatus::ok;
}

std::vector<std::string> SmokeArguments()
{
  return {"dg_igr_hdg_runtime",
      "--card", "exp.hdg.euler.conventional.smooth_persistent.w8a.v1",
      "--case", "conventional-smooth-persistent-v1",
      "--space", "l2", "--mesh-family", "periodic-cartesian-quad",
      "--mesh-n", "3", "--order", "1", "--dt", "0.05",
      "--steps", "3", "--final-time", "0.15", "--backend", "cpu",
      "--device-id", "none", "--output-dir", "w8a-test-output",
      "--output-steps", "0,1,3", "--checkpoint-out",
      "w8a-test-checkpoint", "--checkpoint-step", "1",
      "--stop-after-checkpoint", "false", "--restart-from", "none",
      "--max-retries", "0", "--inject-failure", "off", "--seed",
      "0x573841534d4f4b31"};
}

std::vector<std::string> Rank2ReaderArguments(const std::string& backend,
                                              const std::string& device)
{
  return {"dg_igr_hdg_runtime",
      "--card", "exp.hdg.euler.conventional.smooth_persistent.w8a.v1",
      "--case", "conventional-smooth-persistent-v1",
      "--space", "l2", "--mesh-family", "periodic-cartesian-quad",
      "--mesh-n", "6", "--order", "2", "--dt", "0.05",
      "--steps", "20", "--final-time", "1.0", "--backend", backend,
      "--device-id", device, "--output-dir", "w8a-rank2-reader",
      "--output-steps", "0,10,20", "--checkpoint-out", "none",
      "--checkpoint-step", "none", "--stop-after-checkpoint", "false",
      "--restart-from", "w8a-rank2-checkpoint", "--max-retries", "1",
      "--inject-failure", "AfterRecoveryBeforeCommit:14:1", "--seed",
      "0x57384146554e4331"};
}

std::vector<std::string> ScalarArguments(const std::string& backend,
                                         const std::string& device)
{
  return {"dg_igr_hdg_runtime",
      "--card", "exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3",
      "--case", "sc25-periodic-smooth-persistent-v3",
      "--space", "l2", "--mesh-family", "periodic-cartesian-quad",
      "--mesh-n", "8", "--order", "1", "--dt", "0.000244140625",
      "--steps", "20", "--final-time", "0.0048828125", "--backend", backend,
      "--device-id", device, "--output-dir", "w8b-scalar-output",
      "--output-steps", "0,10,20", "--checkpoint-out",
      "w8b-scalar-checkpoint", "--checkpoint-step", "10",
      "--stop-after-checkpoint", "false", "--restart-from", "none",
      "--max-retries", "1", "--inject-failure",
      "AfterRecoveryBeforeCommit:7:0", "--seed",
      "0x5738425343323553"};
}

std::vector<std::string> SmoothTransitionArguments(const bool scalar,
                                                   const std::string& backend,
                                                   const std::string& device)
{
  return {"dg_igr_hdg_runtime",
      "--card", scalar
          ? "exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3"
          : "exp.hdg.euler.conventional.smooth_persistent.w8a.v1",
      "--case", "periodic-smooth-transition-v2",
      "--space", "l2", "--mesh-family", "periodic-cartesian-quad",
      "--mesh-n", "8", "--order", "1", "--dt", "0.0009765625",
      "--steps", "64", "--final-time", "0.0625", "--backend", backend,
      "--device-id", device, "--output-dir", "w8c-smooth-output",
      "--output-steps", "0,32,64", "--checkpoint-out",
      "w8c-smooth-checkpoint", "--checkpoint-step", "32",
      "--stop-after-checkpoint", "false", "--restart-from", "none",
      "--max-retries", "0", "--inject-failure", "off", "--seed",
      "0x573843534d4f5632"};
}

void SetOption(std::vector<std::string>& arguments,
               const std::string& option, const std::string& value)
{
  const auto found = std::find(arguments.begin(), arguments.end(), option);
  if (found != arguments.end() && found + 1 != arguments.end())
    *(found + 1) = value;
}

ParsedCommandLine Parse(std::vector<std::string> values)
{
  std::vector<const char*> argv;
  for (const std::string& value : values) argv.push_back(value.c_str());
  return ParseCommandLine(static_cast<int>(argv.size()), argv.data());
}

bool DirectAndNegativeControls()
{
  bool passed = true;
  const ParsedCommandLine valid = Parse(SmokeArguments());
  passed = Check(valid.status == CommandLineStatus::run &&
                     valid.options.card_id == smooth_periodic_euler_v1_card_id &&
                     valid.options.case_id == smooth_periodic_euler_v1_case_id &&
                     valid.options.backend_id == CpuSerialBackendTag::id &&
                     valid.options.device_id == -1 &&
                     valid.options.output_steps ==
                         smooth_periodic_euler_v1_smoke_output_steps,
                 "complete_direct_smoke") && passed;

  const ParsedCommandLine cpu_rank2 =
      Parse(Rank2ReaderArguments("cpu", "none"));
  const ParsedCommandLine cuda_rank2 =
      Parse(Rank2ReaderArguments("cuda", "0"));
  passed = Check(cpu_rank2.status == CommandLineStatus::run &&
                     cpu_rank2.options.backend_id == CpuRank2BackendTag::id &&
                     cpu_rank2.options.injection.source_rank == 1,
                 "rank2_reader_cpu_backend_fallback") && passed;
#if DG_IGR_HDG_APP_HAS_CUDA_RANK2
  passed = Check(cuda_rank2.status == CommandLineStatus::run &&
                     cuda_rank2.options.backend_id ==
                         CudaRank2OneGpuHostStagedBackendTag::id,
                 "rank2_reader_cuda_backend_fallback") && passed;
#else
  passed = Check(cuda_rank2.status == CommandLineStatus::rejected &&
                     cuda_rank2.rejection ==
                         ApplicationRejection::unbuilt_backend,
                 "rank2_reader_cuda_unbuilt_backend") && passed;
#endif

  auto h1 = SmokeArguments();
  h1[6] = "h1";
  const ParsedCommandLine unsupported = Parse(h1);
  passed = Check(unsupported.status == CommandLineStatus::rejected &&
                     unsupported.rejection == ApplicationRejection::unsupported_h1 &&
                     ExitForRejection(unsupported.rejection) ==
                         ApplicationExit::unsupported_discretization,
                 "h1_exact_rejection") && passed;

  auto invalid_pair = SmokeArguments();
  invalid_pair[10] = "6";
  const ParsedCommandLine invalid = Parse(invalid_pair);
  passed = Check(invalid.status == CommandLineStatus::rejected &&
                     invalid.rejection ==
                         ApplicationRejection::invalid_polynomial_mesh_pair &&
                     ExitForRejection(invalid.rejection) ==
                         ApplicationExit::configuration_error,
                 "unregistered_polynomial_mesh_pair") && passed;

  auto duplicate = SmokeArguments();
  duplicate[1] = "--case";
  passed = Check(Parse(duplicate).status == CommandLineStatus::usage_error,
                 "duplicate_missing_census") && passed;

  auto conventional_scalar_case = SmokeArguments();
  SetOption(conventional_scalar_case, "--case",
            std::string{scalar_igr_sc25_v3_case_id});
  const ParsedCommandLine crossed_conventional =
      Parse(conventional_scalar_case);
  passed = Check(
               crossed_conventional.status == CommandLineStatus::rejected &&
                   crossed_conventional.rejection ==
                       ApplicationRejection::unknown_case &&
                   ExitForRejection(crossed_conventional.rejection) ==
                       ApplicationExit::configuration_error,
               "conventional_card_scalar_case_cli_rejected") && passed;

  auto scalar_conventional_case = ScalarArguments("cpu", "none");
  SetOption(scalar_conventional_case, "--case",
            std::string{smooth_periodic_euler_v1_case_id});
  const ParsedCommandLine crossed_scalar = Parse(scalar_conventional_case);
  passed = Check(crossed_scalar.status == CommandLineStatus::rejected &&
                     crossed_scalar.rejection ==
                         ApplicationRejection::unknown_case &&
                     ExitForRejection(crossed_scalar.rejection) ==
                         ApplicationExit::configuration_error,
                 "scalar_card_conventional_case_cli_rejected") && passed;

  auto scalar_short = ScalarArguments("cpu", "none");
  SetOption(scalar_short, "--steps", "3");
  const ParsedCommandLine scalar_short_result = Parse(scalar_short);
  passed = Check(
               scalar_short_result.status == CommandLineStatus::rejected &&
                   scalar_short_result.rejection ==
                       ApplicationRejection::unknown_scenario &&
                   ExitForRejection(scalar_short_result.rejection) ==
                       ApplicationExit::configuration_error,
               "scalar_card_three_steps_cli_rejected") && passed;

  const ParsedCommandLine scalar_cpu = Parse(ScalarArguments("cpu", "none"));
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
  passed = Check(scalar_cpu.status == CommandLineStatus::run &&
                     scalar_cpu.options.scenario_id ==
                         scalar_igr_sc25_v3_scenario_id &&
                     scalar_cpu.options.formulation_id ==
                         FullyHybridizedScalarSc25V3Tag::id,
                 "scalar_igr_static_cli") && passed;
#else
  passed = Check(scalar_cpu.status == CommandLineStatus::rejected &&
                     scalar_cpu.rejection == ApplicationRejection::unbuilt_backend,
                 "scalar_igr_default_build_unavailable") && passed;
#endif
  const ParsedCommandLine scalar_cuda = Parse(ScalarArguments("cuda", "0"));
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1
  passed = Check(scalar_cuda.status == CommandLineStatus::run &&
                     scalar_cuda.options.backend_id ==
                         CudaRank1OneGpuBackendTag::id,
                 "scalar_igr_cuda_static_cli") && passed;
#else
  passed = Check(scalar_cuda.status == CommandLineStatus::rejected &&
                     scalar_cuda.rejection == ApplicationRejection::unbuilt_backend,
                 "scalar_igr_cuda_explicit_unbuilt") && passed;
#endif
  auto scalar_cuda_fault_free_arguments = ScalarArguments("cuda", "0");
  SetOption(scalar_cuda_fault_free_arguments, "--max-retries", "0");
  SetOption(scalar_cuda_fault_free_arguments, "--inject-failure", "off");
  const ParsedCommandLine scalar_cuda_fault_free =
      Parse(scalar_cuda_fault_free_arguments);
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1
  passed = Check(scalar_cuda_fault_free.status == CommandLineStatus::run &&
                     scalar_cuda_fault_free.options.injection.mode ==
                         FailureInjectionModeV1::off &&
                     scalar_cuda_fault_free.options.retry_limit == 0,
                 "scalar_cuda_fault_free_cli_contract") && passed;
  for (const std::string phase :
       {"CheckpointStaging", "OutputStaging",
        "AfterRecoveryBeforeCommit"})
  {
    auto scheduled_arguments = ScalarArguments("cuda", "0");
    SetOption(scheduled_arguments, "--inject-failure",
              phase + ":10:0");
    const ParsedCommandLine scheduled = Parse(scheduled_arguments);
    passed = Check(scheduled.status == CommandLineStatus::run &&
                       scheduled.options.backend_id ==
                           CudaRank1OneGpuBackendTag::id &&
                       scheduled.options.injection.target_step == 10 &&
                       scheduled.options.injection.source_rank == 0,
                   "scalar_cuda_rank1_step10_cli_contract") && passed;
  }
  auto terminal_arguments = ScalarArguments("cuda", "0");
  SetOption(terminal_arguments, "--max-retries", "0");
  SetOption(terminal_arguments, "--inject-failure",
            "AfterRecoveryBeforeCommit:20:0");
  const ParsedCommandLine terminal = Parse(terminal_arguments);
  auto terminal_writer_arguments = terminal_arguments;
  SetOption(terminal_writer_arguments, "--stop-after-checkpoint", "true");
  const ParsedCommandLine terminal_writer = Parse(terminal_writer_arguments);
  passed = Check(terminal.status == CommandLineStatus::run &&
                     terminal.options.backend_id ==
                         CudaRank1OneGpuBackendTag::id &&
                     terminal.options.injection.target_step == 20 &&
                     terminal.options.retry_limit == 0 &&
                     terminal_writer.status == CommandLineStatus::rejected &&
                     terminal_writer.rejection ==
                         ApplicationRejection::invalid_run_role,
                 "scalar_cuda_rank1_step20_cli_contract") && passed;
#endif
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2
  for (const std::string phase :
       {"CheckpointStaging", "OutputStaging",
        "AfterRecoveryBeforeCommit"})
  {
    auto scheduled_arguments = ScalarArguments("cuda", "0");
    SetOption(scheduled_arguments, "--inject-failure",
              phase + ":10:1");
    const ParsedCommandLine scheduled = Parse(scheduled_arguments);
    passed = Check(scheduled.status == CommandLineStatus::run &&
                       scheduled.options.backend_id ==
                           CudaRank2OneGpuHostStagedBackendTag::id &&
                       scheduled.options.injection.target_step == 10 &&
                       scheduled.options.injection.source_rank == 1,
                   "scalar_cuda_rank2_step10_cli_fallback") && passed;
  }
  auto rank2_terminal_arguments = ScalarArguments("cuda", "0");
  SetOption(rank2_terminal_arguments, "--max-retries", "0");
  SetOption(rank2_terminal_arguments, "--inject-failure",
            "AfterRecoveryBeforeCommit:20:1");
  const ParsedCommandLine rank2_terminal = Parse(rank2_terminal_arguments);
  auto rank2_terminal_writer_arguments = rank2_terminal_arguments;
  SetOption(rank2_terminal_writer_arguments, "--stop-after-checkpoint",
            "true");
  const ParsedCommandLine rank2_terminal_writer =
      Parse(rank2_terminal_writer_arguments);
  passed = Check(rank2_terminal.status == CommandLineStatus::run &&
                     rank2_terminal.options.backend_id ==
                         CudaRank2OneGpuHostStagedBackendTag::id &&
                     rank2_terminal.options.injection.target_step == 20 &&
                     rank2_terminal.options.retry_limit == 0 &&
                     rank2_terminal_writer.status ==
                         CommandLineStatus::rejected &&
                     rank2_terminal_writer.rejection ==
                         ApplicationRejection::invalid_run_role,
                 "scalar_cuda_rank2_step20_cli_fallback") && passed;
#endif
  const ParsedCommandLine smooth_conventional =
      Parse(SmoothTransitionArguments(false, "cpu", "none"));
  passed = Check(
               smooth_conventional.status == CommandLineStatus::run &&
                   smooth_conventional.options.case_id ==
                       "periodic-smooth-transition-v2" &&
                   smooth_conventional.options.scenario_id ==
                       "W8C-V2-N8-DT10" &&
                   smooth_conventional.options.source_id ==
                       "source.euler.periodic-smooth-transition.v2" &&
                   smooth_conventional.options.output_steps ==
                       std::array<std::uint64_t, 3>{{0, 32, 64}} &&
                   smooth_conventional.options.retry_limit == 0 &&
                   smooth_conventional.options.injection.mode ==
                       FailureInjectionModeV1::off,
               "smooth_transition_conventional_cli") && passed;

  const ParsedCommandLine smooth_scalar =
      Parse(SmoothTransitionArguments(true, "cpu", "none"));
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
  passed = Check(
               smooth_scalar.status == CommandLineStatus::run &&
                   smooth_scalar.options.case_id ==
                       "periodic-smooth-transition-v2" &&
                   smooth_scalar.options.source_id ==
                       "source.sc25.periodic-smooth-transition.v3" &&
                   smooth_scalar.options.formulation_id ==
                       FullyHybridizedScalarSc25V3Tag::id &&
                   smooth_scalar.options.q_a == 12,
               "smooth_transition_scalar_cli") && passed;
#else
  passed = Check(
               smooth_scalar.status == CommandLineStatus::rejected &&
                   smooth_scalar.rejection ==
                       ApplicationRejection::unbuilt_backend,
               "smooth_transition_scalar_unbuilt_cli") && passed;
#endif

  auto smooth_reader = SmoothTransitionArguments(false, "cpu", "none");
  SetOption(smooth_reader, "--checkpoint-out", "none");
  SetOption(smooth_reader, "--checkpoint-step", "none");
  SetOption(smooth_reader, "--restart-from", "w8c-smooth-checkpoint");
  const ParsedCommandLine smooth_reader_result = Parse(smooth_reader);
  passed = Check(
               smooth_reader_result.status == CommandLineStatus::run &&
                   smooth_reader_result.options.run_role ==
                       ApplicationRunRoleV1::split_reader &&
                   smooth_reader_result.options.retry_limit == 0 &&
                   smooth_reader_result.options.injection.mode ==
                       FailureInjectionModeV1::off,
               "smooth_transition_split_reader_cli") && passed;

  auto smooth_bad_case = SmoothTransitionArguments(false, "cpu", "none");
  SetOption(smooth_bad_case, "--case", "periodic-smooth-transition-v3");
  passed = Check(Parse(smooth_bad_case).rejection ==
                     ApplicationRejection::unknown_case,
                 "smooth_transition_unknown_case_cli") && passed;

  auto smooth_bad_dt = SmoothTransitionArguments(false, "cpu", "none");
  SetOption(smooth_bad_dt, "--dt", "0.001");
  passed = Check(Parse(smooth_bad_dt).rejection ==
                     ApplicationRejection::invalid_dt,
                 "smooth_transition_time_schedule_cli") && passed;

  const ParsedCommandLine smooth_cuda =
      Parse(SmoothTransitionArguments(false, "cuda", "0"));
  passed = Check(smooth_cuda.status == CommandLineStatus::rejected &&
#if DG_IGR_HDG_APP_HAS_CUDA_RANK1
                     smooth_cuda.rejection ==
                         ApplicationRejection::unknown_backend,
                 "smooth_transition_cuda_case_is_cpu_serial_only") && passed;
#else
                     smooth_cuda.rejection ==
                         ApplicationRejection::unbuilt_backend,
                 "smooth_transition_cuda_rejected") && passed;
#endif

  auto scalar_h1_args = ScalarArguments("cpu", "none");
  scalar_h1_args[6] = "h1";
  const ParsedCommandLine scalar_h1 = Parse(scalar_h1_args);
  passed = Check(scalar_h1.status == CommandLineStatus::rejected &&
                     scalar_h1.rejection == ApplicationRejection::unsupported_h1,
                 "scalar_igr_h1_pre_resource_rejection") && passed;
  return passed;
}

bool HelpAndConfig()
{
  bool passed = true;
  const char* help_argv[] = {"dg_igr_hdg_runtime", "--help"};
  const ParsedCommandLine help = ParseCommandLine(2, help_argv);
  std::ostringstream text;
  PrintCommandLineHelp(text);
  passed = Check(help.status == CommandLineStatus::help &&
                     text.str().find("--config") != std::string::npos &&
                     text.str().find("periodic-smooth-transition-v2") !=
                         std::string::npos &&
                     text.str().find("--validation-capture") == std::string::npos &&
                     text.str().find("--evidence-dir") == std::string::npos &&
                     text.str().find("AfterRecoveryBeforeCommit") !=
                         std::string::npos,
                 "help_complete") && passed;

  const std::filesystem::path path = std::filesystem::path{"/tmp"} /
      ("dg-igr-hdg-cli-" + std::to_string(static_cast<long long>(::getpid())) +
       ".cfg");
  {
    std::ofstream output(path);
    const auto args = SmokeArguments();
    for (std::size_t index = 1; index < args.size(); index += 2)
      output << args[index].substr(2) << '=' << args[index + 1] << '\n';
  }
  const std::string path_text = path.string();
  const char* config_argv[] = {"dg_igr_hdg_runtime", "--config",
                               path_text.c_str()};
  const ParsedCommandLine config = ParseCommandLine(3, config_argv);
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  passed = Check(config.status == CommandLineStatus::run &&
                     config.options.seed ==
                         smooth_periodic_euler_v1_smoke_seed,
                 "dependency_free_config") && passed;
  return passed;
}
} // namespace

int main(const int argc, const char* const* argv)
{
  if (argc == 9 &&
      std::string{argv[1]} == "--validate-scalar-rank2-artifacts")
  {
    const bool valid = ValidateScalarRank2Artifacts(
        {{argv[2], {}, {}}}, argv[3], {{argv[4], argv[5], argv[6]}},
        argv[7], argv[8]);
    return valid ? 0 : 1;
  }
  if (argc == 5 &&
      std::string{argv[1]} == "--make-scalar-sidecar-negative")
  {
    return MakeScalarSidecarNegative(argv[2], argv[3], argv[4]) ? 0 : 1;
  }
  if (argc == 5 && std::string{argv[1]} == "--compare-scalar-vtk")
  {
    double tolerance = 0.0;
    try { tolerance = std::stod(argv[4]); }
    catch (...) { return 64; }
    const bool compared = tolerance > 0.0 &&
        CompareScalarVtk(argv[2], argv[3], tolerance);
    if (compared) std::cout << "HDG scalar-IGR VTK parity PASS\n";
    return compared ? 0 : 1;
  }
  if (argc == 5 &&
      std::string{argv[1]} == "--compare-scalar-vtk-w8c-rank2")
  {
    double tolerance = 0.0;
    try { tolerance = std::stod(argv[4]); }
    catch (...) { return 64; }
    const bool compared = tolerance > 0.0 &&
        CompareScalarVtk(argv[2], argv[3], tolerance, true);
    if (compared) std::cout << "HDG scalar-IGR W8C rank-two VTK parity PASS\n";
    return compared ? 0 : 1;
  }
  if (argc == 4 && std::string{argv[1]} == "--validate-scalar-initial-vtk")
  {
    double tolerance = 0.0;
    try { tolerance = std::stod(argv[3]); }
    catch (...) { return 64; }
    const bool valid = tolerance > 0.0 &&
        ValidateScalarInitialVtk(argv[2], tolerance);
    if (valid) std::cout << "HDG scalar-IGR VTK oracle PASS\n";
    return valid ? 0 : 1;
  }
  if (argc != 1) return 64;
  bool passed = true;
  passed = DirectAndNegativeControls() && passed;
  passed = HelpAndConfig() && passed;
  passed = Check(HandoffPathNegativeControls(),
                 "handoff_path_negative_controls") && passed;
  if (passed) std::cout << "HDG runtime CLI contract PASS\n";
  return passed ? 0 : 1;
}
