#include "igrlab/compat.hpp"
#include "sim/io/vtk_restart.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <zlib.h>

#include "picojson.h"
#include "mfem/general/binaryio.hpp"
#include "tinyxml2.h"
#include "mfem/mesh/vtk.hpp"

#include "sim/io/vtk_helpers.hpp"
#include "utils/diagnostics/euler_admissibility.hpp"
#include "utils/diagnostics/non_finite.hpp"

namespace dg_sim
{
namespace
{

namespace fs = std::filesystem;
using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;

constexpr int legacy_manifest_schema = 1;
constexpr int typed_integrator_manifest_schema = 2;
constexpr int static_source_manifest_schema = 3;
constexpr int transport_manifest_schema = 4;
constexpr int current_manifest_schema = transport_manifest_schema;

bool IsSupportedManifestSchema(int schema)
{
  return schema == legacy_manifest_schema ||
         schema == typed_integrator_manifest_schema ||
         schema == static_source_manifest_schema ||
         schema == current_manifest_schema;
}

struct PVDEntry
{
  int cycle = -1;
  double time = 0.0;
  fs::path pvtu;
};

struct Manifest
{
  int schema = 0;
  bool complete = false;
  int cycle = -1;
  double time = 0.0;
  long step = -1;
  int ranks = -1;
  int dim = -1;
  int equations = -1;
  int order = -1;
  int serial_refinement = -1;
  int parallel_refinement = -1;
  bool igr_enabled = false;
  bool exact_dg_face = false;
  bool igr_invariant_domain_flux = false;
  bool zhang_shu_limiter = false;
  bool zhang_shu_vtk_points = false;
  double gamma = 0.0;
  double rho_floor = 0.0;
  double pressure_floor = 0.0;
  double c_alpha = 0.0;
  int sigma_update_interval = 1;
  std::string case_name;
  std::string dg_backend;
  std::string time_integrator;
  std::string dgsem_stabilization;
  std::string dgsem_volume_flux;
  std::string dgsem_volume_dissipation;
  std::string dgsem_surface_flux;
  std::string dgsem_entropy_jacobian;
  std::string basis;
  std::string ordering;
  std::string data_type;
  std::string mesh_fingerprint;
  std::string zhang_shu_points;
  std::string igr_space;
  std::string igr_grad;
  std::string alpha_length_policy;
  std::string ip_kind;
  std::string sigma_cadence;
  std::string sigma_restart_policy;
  bool static_source_present = false;
  std::string static_source_fingerprint;
};

struct PieceData
{
  int points_count = 0;
  int cells_count = 0;
  std::vector<double> points;
  std::vector<std::int32_t> connectivity;
  std::vector<std::int32_t> offsets;
  std::vector<std::uint8_t> types;
  std::vector<std::int32_t> attributes;
  std::vector<double> density;
  std::vector<double> momentum;
  std::vector<double> energy;
};

[[noreturn]] void Fail(const std::string& message)
{
  throw std::runtime_error(message);
}

std::string RequiredAttribute(const XMLElement* element, const char* name)
{
  if (element == nullptr || element->Attribute(name) == nullptr)
  {
    Fail(std::string("missing XML attribute '") + name + "'");
  }
  return element->Attribute(name);
}

int RequiredIntAttribute(const XMLElement* element, const char* name)
{
  int value = 0;
  if (element == nullptr || element->QueryIntAttribute(name, &value) !=
                              tinyxml2::XML_SUCCESS)
  {
    Fail(std::string("missing or invalid integer XML attribute '") + name + "'");
  }
  return value;
}

std::string StripWhitespace(const char* text)
{
  std::string out;
  if (text == nullptr) { return out; }
  for (const unsigned char c : std::string(text))
  {
    if (!std::isspace(c)) { out.push_back(static_cast<char>(c)); }
  }
  return out;
}

template <typename T>
T ByteSwap(T value)
{
  std::array<unsigned char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  std::reverse(bytes.begin(), bytes.end());
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

bool FileByteSwapRequired(const std::string& byte_order)
{
  const std::uint16_t marker = 1;
  unsigned char first_byte = 0;
  std::memcpy(&first_byte, &marker, sizeof(first_byte));
  const bool host_little = first_byte == 1;
  if (byte_order == "LittleEndian") { return !host_little; }
  if (byte_order == "BigEndian") { return host_little; }
  Fail("unsupported VTK byte_order '" + byte_order + "'");
}

template <typename Header>
std::uint64_t ReadHeader(const char* data, bool swap)
{
  Header value{};
  std::memcpy(&value, data, sizeof(Header));
  if (swap) { value = ByteSwap(value); }
  return static_cast<std::uint64_t>(value);
}

std::vector<char> DecodeBase64(const std::string& encoded)
{
  std::vector<char> decoded;
  try
  {
    mfem::bin_io::DecodeBase64(encoded.data(), encoded.size(), decoded);
  }
  catch (const std::exception& ex)
  {
    Fail(std::string("invalid base64 payload: ") + ex.what());
  }
  return decoded;
}

template <typename Header>
std::vector<char> DecodeBinaryPayload(const std::string& encoded,
                                      bool compressed,
                                      bool swap,
                                      std::size_t expected_bytes)
{
  constexpr std::size_t hs = sizeof(Header);
  if (!compressed)
  {
    std::vector<char> decoded = DecodeBase64(encoded);
    if (decoded.size() < hs) { Fail("truncated VTK binary header"); }
    const std::uint64_t count = ReadHeader<Header>(decoded.data(), swap);
    if (count != expected_bytes || decoded.size() != hs + expected_bytes)
    {
      Fail("VTK binary byte count does not match the declared array length");
    }
    return {decoded.begin() + static_cast<std::ptrdiff_t>(hs), decoded.end()};
  }

  const std::size_t first_header_chars = mfem::bin_io::NumBase64Chars(hs);
  if (encoded.size() < first_header_chars) { Fail("truncated compressed VTK header"); }
  const std::vector<char> first = DecodeBase64(encoded.substr(0, first_header_chars));
  if (first.size() < hs) { Fail("invalid compressed VTK block header"); }
  const std::uint64_t nblocks64 = ReadHeader<Header>(first.data(), swap);
  if (nblocks64 == 0 || nblocks64 > 1000000)
  {
    Fail("invalid compressed VTK block count");
  }
  const std::size_t nblocks = static_cast<std::size_t>(nblocks64);
  const std::size_t header_bytes = (nblocks + 3) * hs;
  const std::size_t header_chars = mfem::bin_io::NumBase64Chars(header_bytes);
  if (encoded.size() < header_chars) { Fail("truncated compressed VTK header"); }
  const std::vector<char> header = DecodeBase64(encoded.substr(0, header_chars));
  if (header.size() != header_bytes) { Fail("invalid compressed VTK header size"); }

  const auto h = [&](std::size_t i)
  { return ReadHeader<Header>(header.data() + i * hs, swap); };
  const std::uint64_t block_bytes = h(1);
  const std::uint64_t last_bytes = h(2);
  if ((nblocks - 1) * block_bytes + last_bytes != expected_bytes)
  {
    Fail("compressed VTK uncompressed size does not match array length");
  }

  const std::vector<char> compressed_data = DecodeBase64(encoded.substr(header_chars));
  std::size_t compressed_total = 0;
  for (std::size_t i = 0; i < nblocks; ++i)
  {
    const std::uint64_t size = h(i + 3);
    if (size > std::numeric_limits<std::size_t>::max() - compressed_total)
    {
      Fail("compressed VTK payload size overflow");
    }
    compressed_total += static_cast<std::size_t>(size);
  }
  if (compressed_total != compressed_data.size())
  {
    Fail("truncated or trailing compressed VTK payload");
  }

  std::vector<char> out(expected_bytes);
  std::size_t src_offset = 0;
  std::size_t dst_offset = 0;
  for (std::size_t i = 0; i < nblocks; ++i)
  {
    const uLong src_size = static_cast<uLong>(h(i + 3));
    uLongf dst_size = static_cast<uLongf>((i + 1 == nblocks) ? last_bytes
                                                              : block_bytes);
    const int rc = uncompress(reinterpret_cast<Bytef*>(out.data() + dst_offset),
                              &dst_size,
                              reinterpret_cast<const Bytef*>(
                                compressed_data.data() + src_offset),
                              src_size);
    if (rc != Z_OK || dst_size != ((i + 1 == nblocks) ? last_bytes : block_bytes))
    {
      Fail("zlib failed while decoding a VTK data block");
    }
    src_offset += src_size;
    dst_offset += dst_size;
  }
  return out;
}

template <typename T>
std::vector<T> ReadDataArray(const XMLElement* array,
                             const std::string& expected_type,
                             std::size_t count,
                             const std::string& byte_order,
                             const std::string& header_type,
                             bool compressed)
{
  if (array == nullptr) { Fail("missing VTK DataArray"); }
  if (RequiredAttribute(array, "type") != expected_type)
  {
    Fail("VTK DataArray type mismatch: expected " + expected_type);
  }
  const std::string format = RequiredAttribute(array, "format");
  std::vector<T> out(count);
  if (format == "ascii")
  {
    std::istringstream input(array->GetText() ? array->GetText() : "");
    for (std::size_t i = 0; i < count; ++i)
    {
      if constexpr (std::is_same_v<T, std::uint8_t>)
      {
        unsigned int value = 0;
        if (!(input >> value) || value > 255) { Fail("invalid ASCII UInt8 array"); }
        out[i] = static_cast<std::uint8_t>(value);
      }
      else if (!(input >> out[i]))
      {
        Fail("truncated ASCII VTK DataArray");
      }
    }
    std::string extra;
    if (input >> extra) { Fail("trailing values in ASCII VTK DataArray"); }
    return out;
  }
  if (format != "binary") { Fail("only inline ASCII or binary VTK arrays are supported"); }

  const bool swap = FileByteSwapRequired(byte_order);
  const std::string encoded = StripWhitespace(array->GetText());
  std::vector<char> bytes;
  if (header_type == "UInt64")
  {
    bytes = DecodeBinaryPayload<std::uint64_t>(encoded, compressed, swap,
                                               count * sizeof(T));
  }
  else if (header_type.empty() || header_type == "UInt32")
  {
    bytes = DecodeBinaryPayload<std::uint32_t>(encoded, compressed, swap,
                                               count * sizeof(T));
  }
  else
  {
    Fail("unsupported VTK header_type '" + header_type + "'");
  }
  std::memcpy(out.data(), bytes.data(), bytes.size());
  if (swap)
  {
    for (T& value : out) { value = ByteSwap(value); }
  }
  return out;
}

const XMLElement* FindDataArray(const XMLElement* parent, const std::string& name)
{
  if (parent == nullptr) { return nullptr; }
  for (const char* tag : {"DataArray", "PDataArray"})
  {
    for (const XMLElement* array = parent->FirstChildElement(tag); array;
         array = array->NextSiblingElement(tag))
    {
      const char* found = array->Attribute("Name");
      if (found != nullptr && name == found) { return array; }
    }
  }
  return nullptr;
}

int Components(const XMLElement* array)
{
  int components = 1;
  if (array != nullptr && array->Attribute("NumberOfComponents") != nullptr &&
      array->QueryIntAttribute("NumberOfComponents", &components) !=
        tinyxml2::XML_SUCCESS)
  {
    Fail("invalid NumberOfComponents");
  }
  return components;
}

} // namespace

namespace
{

struct PVTUInfo
{
  fs::path piece;
  std::string byte_order;
};

fs::path ResolvePVDPath(const std::string& input);
std::vector<PVDEntry> ReadPVD(const fs::path& pvd);
PVDEntry SelectEntry(const std::vector<PVDEntry>& entries,
                     int requested_cycle,
                     long legacy_step);
Manifest ReadManifest(const fs::path& path);
std::string FingerprintString(std::uint64_t value);
std::string BackendName(const RunConfig& run_cfg);
std::string BasisName(const RunConfig& run_cfg);
std::string IGRSpaceName(const RunConfig& run_cfg);
std::string IGRGradName(const RunConfig& run_cfg);
std::string SigmaRestartPolicy(const RunConfig& run_cfg);
void ValidateManifest(const Manifest& manifest,
                      const PVDEntry& entry,
                      const RunConfig& run_cfg,
                      const ExampleContext& ctx,
                      std::uint64_t fingerprint,
                      bool igr_enabled,
                      const mfem::ParGridFunction* static_source,
                      std::uint64_t static_source_fingerprint);
PVTUInfo ReadPVTU(const PVDEntry& entry, int dim, int ranks, int rank);
PieceData ReadPiece(const PVTUInfo& info, int dim);
void CollectiveFailure(const std::string& local_error, MPI_Comm comm);

double InfinityNorm(const mfem::DenseMatrix& matrix)
{
  double norm = 0.0;
  for (int row = 0; row < matrix.Height(); ++row)
  {
    double sum = 0.0;
    for (int col = 0; col < matrix.Width(); ++col)
    {
      sum += std::abs(matrix(row, col));
    }
    norm = std::max(norm, sum);
  }
  return norm;
}

struct Interpolation
{
  mfem::DenseMatrix samples;
  mfem::DenseMatrix inverse;
  double condition_inf = 0.0;
};

Interpolation BuildInterpolation(const mfem::FiniteElement& fe,
                                 mfem::Geometry::Type geometry,
                                 int level)
{
  const mfem::RefinedGeometry* refined =
    mfem::GlobGeometryRefiner.Refine(geometry, level, 1);
  const int npoints = refined->RefPts.GetNPoints();
  const int dofs = fe.GetDof();
  if (npoints != dofs)
  {
    Fail("VTK level does not define a square interpolation map for the active FE");
  }
  Interpolation result;
  result.samples.SetSize(npoints, dofs);
  mfem::Vector shape(dofs);
  for (int q = 0; q < npoints; ++q)
  {
    fe.CalcShape(refined->RefPts.IntPoint(q), shape);
    for (int j = 0; j < dofs; ++j) { result.samples(q, j) = shape(j); }
  }
  mfem::DenseMatrixInverse factor(result.samples);
  factor.GetInverseMatrix(result.inverse);
  result.condition_inf =
    InfinityNorm(result.samples) * InfinityNorm(result.inverse);
  if (!std::isfinite(result.condition_inf) || result.condition_inf > 1.0e8)
  {
    Fail("VTK-to-FE interpolation map is singular or too ill-conditioned");
  }
  return result;
}

struct ReconstructionMetrics
{
  double residual_max = 0.0;
  double residual_sq = 0.0;
  double source_sq = 0.0;
  double point_max = 0.0;
  double condition_inf = 0.0;
};

double FieldValue(const PieceData& piece,
                  int dim,
                  int component,
                  std::size_t point)
{
  if (component == 0) { return piece.density[point]; }
  if (component == dim + 1) { return piece.energy[point]; }
  return piece.momentum[point * dim + component - 1];
}

ReconstructionMetrics ValidateAndReconstruct(const PieceData& piece,
                                             const RunConfig& run_cfg,
                                             const ExampleContext& ctx,
                                             mfem::ParGridFunction& state)
{
  const auto* dg =
    dynamic_cast<const mfem::L2_FECollection*>(ctx.spaces.vfes.FEColl());
  if (dg == nullptr)
  {
    Fail("VTK restart requires an L2_FECollection/DG_FECollection state space");
  }
  if (ctx.spaces.vfes.GetOrdering() != mfem::Ordering::byNODES)
  {
    Fail("VTK restart requires byNODES state ordering");
  }
  if (run_cfg.dg_backend == DGBackend::DG)
  {
    if (dg->GetBasisType() != mfem::BasisType::GaussLegendre)
    {
      Fail("VTK restart backend/basis mismatch: DG requires GaussLegendre");
    }
    if (run_cfg.opts.order < 1 || run_cfg.opts.order > 3)
    {
      Fail("VTK restart DG backend supports polynomial orders 1 through 3");
    }
  }
  else if (run_cfg.dg_backend == DGBackend::DGSEM)
  {
    if (dg->GetBasisType() != mfem::BasisType::GaussLobatto)
    {
      Fail("VTK restart backend/basis mismatch: DGSEM requires GaussLobatto");
    }
    if (run_cfg.opts.order < 1 || run_cfg.opts.order > 4)
    {
      Fail("VTK restart DGSEM backend supports polynomial orders 1 through 4");
    }
  }
  if (piece.cells_count != ctx.Mesh().GetNE())
  {
    Fail("VTU cell count does not match the active local mesh");
  }

  std::size_t expected_points = 0;
  for (int e = 0; e < ctx.Mesh().GetNE(); ++e)
  {
    const auto geometry = ctx.Mesh().GetElementBaseGeometry(e);
    expected_points += mfem::GlobGeometryRefiner
                         .Refine(geometry, run_cfg.opts.order, 1)
                         ->RefPts.GetNPoints();
  }
  if (expected_points != static_cast<std::size_t>(piece.points_count) ||
      piece.connectivity.size() != expected_points)
  {
    Fail("VTU point/connectivity count does not match the active mesh/order");
  }

  for (const auto* values : {&piece.points, &piece.density, &piece.momentum,
                             &piece.energy})
  {
    if (!std::all_of(values->begin(), values->end(),
                     [](double value) { return std::isfinite(value); }))
    {
      Fail("VTU contains a non-finite coordinate or conservative value");
    }
  }

  std::map<std::pair<int, int>, Interpolation> interpolations;
  ReconstructionMetrics metrics;
  std::size_t point_offset = 0;
  mfem::Array<int> dofs;
  std::vector<double> planned_state(static_cast<std::size_t>(state.Size()));
  std::vector<unsigned char> planned_dofs(
    static_cast<std::size_t>(state.Size()), 0);

  for (int e = 0; e < ctx.Mesh().GetNE(); ++e)
  {
    const auto geometry = ctx.Mesh().GetElementBaseGeometry(e);
    const mfem::FiniteElement* fe = ctx.spaces.vfes.GetFE(e);
    const mfem::RefinedGeometry* refined =
      mfem::GlobGeometryRefiner.Refine(geometry, run_cfg.opts.order, 1);
    const int npoints = refined->RefPts.GetNPoints();
    const auto key = std::make_pair(static_cast<int>(geometry), fe->GetDof());
    auto found = interpolations.find(key);
    if (found == interpolations.end())
    {
      found = interpolations
                .emplace(key,
                         BuildInterpolation(*fe, geometry, run_cfg.opts.order))
                .first;
    }
    const Interpolation& interpolation = found->second;
    metrics.condition_inf =
      std::max(metrics.condition_inf, interpolation.condition_inf);

    mfem::Array<int> local_connectivity;
    mfem::CreateVTKElementConnectivity(local_connectivity,
                                       geometry,
                                       run_cfg.opts.order);
    if (local_connectivity.Size() != npoints)
    {
      Fail("unexpected high-order VTK connectivity size");
    }
    for (int q = 0; q < npoints; ++q)
    {
      const std::int32_t expected =
        static_cast<std::int32_t>(point_offset + local_connectivity[q]);
      if (piece.connectivity[point_offset + q] != expected)
      {
        Fail("VTU connectivity does not match the active local element ordering");
      }
    }
    if (piece.offsets[e] != static_cast<std::int32_t>(point_offset + npoints) ||
        piece.types[e] != mfem::VTKGeometry::HighOrderMap[geometry] ||
        piece.attributes[e] != ctx.Mesh().GetAttribute(e))
    {
      Fail("VTU cell offset/type/attribute is incompatible with the active mesh");
    }

    mfem::DenseMatrix physical;
    const_cast<mfem::ParMesh&>(ctx.Mesh())
      .GetElementTransformation(e)
      ->Transform(refined->RefPts, physical);
    for (int q = 0; q < npoints; ++q)
    {
      for (int d = 0; d < 3; ++d)
      {
        const double expected = d < physical.Height() ? physical(d, q) : 0.0;
        metrics.point_max =
          std::max(metrics.point_max,
                   std::abs(piece.points[3 * (point_offset + q) + d] - expected));
      }
    }

    ctx.spaces.vfes.GetElementDofs(e, dofs);
    if (dofs.Size() != fe->GetDof()) { Fail("unexpected active element dof count"); }
    mfem::Vector values(npoints), coefficients(fe->GetDof()), replay(npoints);
    for (int component = 0; component < ctx.num_equations; ++component)
    {
      for (int q = 0; q < npoints; ++q)
      {
        values(q) = FieldValue(piece, ctx.dim, component, point_offset + q);
      }
      interpolation.inverse.Mult(values, coefficients);
      interpolation.samples.Mult(coefficients, replay);
      for (int q = 0; q < npoints; ++q)
      {
        const double residual = replay(q) - values(q);
        metrics.residual_max = std::max(metrics.residual_max, std::abs(residual));
        metrics.residual_sq += residual * residual;
        metrics.source_sq += values(q) * values(q);
      }
      for (int j = 0; j < dofs.Size(); ++j)
      {
        const int scalar_dof = dofs[j] >= 0 ? dofs[j] : -1 - dofs[j];
        const double sign = dofs[j] >= 0 ? 1.0 : -1.0;
        const int vdof = ctx.spaces.vfes.DofToVDof(scalar_dof, component);
        if (vdof < 0 || vdof >= state.Size())
        {
          Fail("reconstructed VTK coefficient maps outside the active state");
        }
        const std::size_t planned = static_cast<std::size_t>(vdof);
        if (planned_dofs[planned] != 0)
        {
          Fail("reconstructed VTK coefficient maps to a duplicate state DOF");
        }
        planned_state[planned] = sign * coefficients(j);
        planned_dofs[planned] = 1;
      }
    }
    point_offset += npoints;
  }
  if (point_offset != expected_points) { Fail("internal VTK point accounting error"); }

  const double coordinate_scale = [&]
  {
    double scale = 1.0;
    for (double value : piece.points) { scale = std::max(scale, std::abs(value)); }
    return scale;
  }();
  if (metrics.point_max > 2.0e-12 * coordinate_scale)
  {
    Fail("VTU physical points do not match the active mesh");
  }
  const double rel = std::sqrt(metrics.residual_sq /
                               std::max(metrics.source_sq,
                                        std::numeric_limits<double>::min()));
  if (metrics.residual_max > 5.0e-11 || rel > 5.0e-12)
  {
    Fail("VTK conservative fields are not representable in the active L2 space");
  }
  if (!std::all_of(planned_dofs.begin(),
                   planned_dofs.end(),
                   [](unsigned char planned) { return planned == 1; }))
  {
    Fail("VTK reconstruction did not define every active state DOF exactly once");
  }

  double* destination = state.HostWrite();
  std::copy(planned_state.begin(), planned_state.end(), destination);
  return metrics;
}

void HashBytes(std::uint64_t& hash, const void* data, std::size_t size)
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i)
  {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
}

template <typename T>
void HashValue(std::uint64_t& hash, const T& value)
{
  HashBytes(hash, &value, sizeof(value));
}

void HashString(std::uint64_t& hash, const std::string& value)
{
  HashBytes(hash, value.data(), value.size());
  const unsigned char terminator = 0;
  HashBytes(hash, &terminator, 1);
}

void HashCanonicalUInt64(std::uint64_t& hash, std::uint64_t value)
{
  std::array<unsigned char, sizeof(value)> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
  {
    bytes[i] = static_cast<unsigned char>(value & 0xffU);
    value >>= 8U;
  }
  HashBytes(hash, bytes.data(), bytes.size());
}

void HashCanonicalDouble(std::uint64_t& hash, double value)
{
  // Positive and negative zero are the same projected source.  Normalize
  // them so the identity represents numerical values rather than incidental
  // sign bits introduced by a coefficient implementation.
  if (value == 0.0) { value = 0.0; }
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value),
                "Restart field fingerprints require IEEE-sized doubles.");
  std::memcpy(&bits, &value, sizeof(bits));
  HashCanonicalUInt64(hash, bits);
}

} // namespace

std::uint64_t ComputeVTKRestartMeshFingerprint(const RunConfig& run_cfg,
                                               const ExampleContext& ctx)
{
  const MPI_Comm comm = ctx.Mesh().GetComm();
  int comm_size = 0;
  MFEM_VERIFY(MPI_Comm_size(comm, &comm_size) == MPI_SUCCESS && comm_size > 0,
              "VTK restart requires a live mesh communicator.");
  std::uint64_t local = 1469598103934665603ULL;
  HashString(local, run_cfg.case_config.case_name);
  HashString(local, BackendName(run_cfg));
  HashString(local, BasisName(run_cfg));
  HashValue(local, run_cfg.opts.order);
  HashValue(local, run_cfg.opts.ser_ref_levels);
  HashValue(local, run_cfg.opts.par_ref_levels);
  HashValue(local, ctx.dim);
  const int ne = ctx.Mesh().GetNE();
  HashValue(local, ne);
  for (int e = 0; e < ne; ++e)
  {
    const auto geometry = ctx.Mesh().GetElementBaseGeometry(e);
    const int geometry_int = static_cast<int>(geometry);
    HashValue(local, geometry_int);
    HashValue(local, ctx.Mesh().GetAttribute(e));
    const mfem::RefinedGeometry* refined =
      mfem::GlobGeometryRefiner.Refine(geometry, run_cfg.opts.order, 1);
    mfem::DenseMatrix physical;
    const_cast<mfem::ParMesh&>(ctx.Mesh())
      .GetElementTransformation(e)
      ->Transform(refined->RefPts, physical);
    for (int q = 0; q < physical.Width(); ++q)
    {
      for (int d = 0; d < physical.Height(); ++d)
      {
        HashValue(local, physical(d, q));
      }
    }
  }
  std::vector<std::uint64_t> rank_hashes(static_cast<std::size_t>(comm_size));
  MPI_Allgather(&local,
                1,
                mfem::MPITypeMap<std::uint64_t>::mpi_type,
                rank_hashes.data(),
                1,
                mfem::MPITypeMap<std::uint64_t>::mpi_type,
                comm);
  std::uint64_t global = 1469598103934665603ULL;
  for (const auto value : rank_hashes) { HashValue(global, value); }
  return global;
}

std::uint64_t ComputeVTKRestartFieldFingerprint(
  const mfem::ParGridFunction& field,
  const ExampleContext& ctx)
{
  const auto* const space = field.ParFESpace();
  MFEM_VERIFY(space != nullptr && space->GetParMesh() != nullptr,
              "VTK restart field fingerprints require a parallel FE space.");
  MFEM_VERIFY(space == &ctx.spaces.vfes,
              "The VTK restart static source must use the active Euler space.");
  MFEM_VERIFY(field.Size() == space->GetVSize(),
              "The VTK restart static source has an invalid local size.");

  const MPI_Comm comm = ctx.Mesh().GetComm();
  int comm_rank = -1;
  int comm_size = 0;
  int communicator_relation = MPI_UNEQUAL;
  MFEM_VERIFY(MPI_Comm_rank(comm, &comm_rank) == MPI_SUCCESS &&
                MPI_Comm_size(comm, &comm_size) == MPI_SUCCESS &&
                comm_rank >= 0 && comm_size > 0 &&
                MPI_Comm_compare(space->GetParMesh()->GetComm(),
                                 comm,
                                 &communicator_relation) == MPI_SUCCESS &&
                (communicator_relation == MPI_IDENT ||
                 communicator_relation == MPI_CONGRUENT),
              "The VTK restart static source must use the mesh communicator.");

  std::uint64_t local = 1469598103934665603ULL;
  HashString(local, "dg-igr-vtk-restart-field-v1");
  HashCanonicalUInt64(local, static_cast<std::uint64_t>(comm_size));
  HashCanonicalUInt64(local, static_cast<std::uint64_t>(comm_rank));
  HashCanonicalUInt64(local, static_cast<std::uint64_t>(field.Size()));
  HashCanonicalUInt64(local, static_cast<std::uint64_t>(space->GetVSize()));
  HashCanonicalUInt64(local, static_cast<std::uint64_t>(space->GetTrueVSize()));
  HashCanonicalUInt64(local, static_cast<std::uint64_t>(space->GetVDim()));
  HashCanonicalUInt64(
    local, static_cast<std::uint64_t>(space->GetOrdering()));
  HashCanonicalUInt64(local, static_cast<std::uint64_t>(space->GetNE()));
  const char* const collection_name =
    space->FEColl() == nullptr ? nullptr : space->FEColl()->Name();
  HashString(local, collection_name == nullptr ? "" : collection_name);
  const mfem::real_t* const values = field.HostRead();
  for (int i = 0; i < field.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(values[i]),
                "The projected VTK restart static source must be finite.");
    HashCanonicalDouble(local, static_cast<double>(values[i]));
  }

  std::vector<std::uint64_t> rank_hashes(static_cast<std::size_t>(comm_size));
  MFEM_VERIFY(
    MPI_Allgather(&local,
                  1,
                  mfem::MPITypeMap<std::uint64_t>::mpi_type,
                  rank_hashes.data(),
                  1,
                  mfem::MPITypeMap<std::uint64_t>::mpi_type,
                  comm) == MPI_SUCCESS,
    "Unable to collect VTK restart static-source fingerprints.");
  std::uint64_t global = 1469598103934665603ULL;
  HashString(global, "dg-igr-vtk-restart-distributed-field-v1");
  HashCanonicalUInt64(global, static_cast<std::uint64_t>(comm_size));
  for (std::size_t rank = 0; rank < rank_hashes.size(); ++rank)
  {
    HashCanonicalUInt64(global, static_cast<std::uint64_t>(rank));
    HashCanonicalUInt64(global, rank_hashes[rank]);
  }
  return global;
}

VTKRestartInfo LoadEulerVTKRestart(const VTKRestartRequest& request,
                                   const RunConfig& run_cfg,
                                   const ExampleContext& ctx,
                                   mfem::ParGridFunction& state,
                                   bool igr_enabled,
                                   const mfem::ParGridFunction* static_source)
{
  const auto start = std::chrono::steady_clock::now();
  VTKRestartInfo info;
  PVDEntry entry;
  PVTUInfo pvtu;
  PieceData piece;
  Manifest manifest;
  bool has_manifest = false;
  std::string error;
  std::unique_ptr<mfem::ParGridFunction> candidate;
  const MPI_Comm comm = ctx.Mesh().GetComm();
  int comm_rank = 0;
  int comm_size = 0;
  MFEM_VERIFY(MPI_Comm_rank(comm, &comm_rank) == MPI_SUCCESS &&
                MPI_Comm_size(comm, &comm_size) == MPI_SUCCESS &&
                comm_size > 0,
              "VTK restart requires a live mesh communicator.");
  try
  {
    if (igr_enabled)
    {
      const auto cadence = ResolveEffectiveIGRSigmaUpdateCadence(run_cfg);
      if (cadence != igrlab::IGRSigmaUpdateCadence::EveryStage &&
          cadence !=
            igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep)
      {
        Fail("production VTK restart supports only every-stage or "
             "extrapolated-every-step IGR sigma updates");
      }
      // Only conservative state is serialized. RunEulerCase passes a cold
      // start to PrepareSigmaForOutput after this load, which exactly
      // reconstructs the accepted sigma anchor and deliberately clears the
      // extrapolation predecessor/history before the first resumed step.
      if (IsL2IGRSpaceName(run_cfg.igr_opts.space) &&
          IsBrokenDGIGRGradName(run_cfg.igr_opts.grad) &&
          !run_cfg.igr_opts.exact_dg_face_operator)
      {
        Fail("production L2 DG + IGR VTK restart requires exact DG faces");
      }
    }
    const fs::path pvd = ResolvePVDPath(request.path);
    entry = SelectEntry(ReadPVD(pvd), request.cycle, request.legacy_step);
    info.pvd_path = pvd.string();
    info.dataset_dir = pvd.parent_path().string();
    info.cycle_dir = entry.pvtu.parent_path().string();
    const fs::path manifest_path = entry.pvtu.parent_path() / "restart.json";
    has_manifest = fs::is_regular_file(manifest_path);
    if (has_manifest) { manifest = ReadManifest(manifest_path); }
    pvtu = ReadPVTU(entry, ctx.dim, comm_size, comm_rank);
    piece = ReadPiece(pvtu, ctx.dim);
  }
  catch (const std::exception& ex)
  {
    error = ex.what();
  }
  CollectiveFailure(error, comm);

  const std::uint64_t fingerprint =
    ComputeVTKRestartMeshFingerprint(run_cfg, ctx);
  int static_source_presence[2] = {static_source == nullptr ? 0 : 1,
                                   static_source == nullptr ? 0 : 1};
  MFEM_VERIFY(MPI_Allreduce(MPI_IN_PLACE,
                            &static_source_presence[0],
                            1,
                            MPI_INT,
                            MPI_MIN,
                            comm) == MPI_SUCCESS &&
                MPI_Allreduce(MPI_IN_PLACE,
                              &static_source_presence[1],
                              1,
                              MPI_INT,
                              MPI_MAX,
                              comm) == MPI_SUCCESS &&
                static_source_presence[0] == static_source_presence[1],
              "VTK restart static-source presence must agree on every rank.");
  const std::uint64_t static_source_fingerprint =
    static_source == nullptr
      ? 0
      : ComputeVTKRestartFieldFingerprint(*static_source, ctx);
  ReconstructionMetrics local_metrics;
  error.clear();
  try
  {
    if (has_manifest)
    {
      ValidateManifest(
        manifest,
        entry,
        run_cfg,
        ctx,
        fingerprint,
        igr_enabled,
        static_source,
        static_source_fingerprint);
      if (request.legacy_step >= 0 && request.legacy_step != manifest.step)
      {
        Fail("--restart-step conflicts with the selected cycle manifest");
      }
      info.step = manifest.step;
      info.legacy = false;
    }
    else
    {
      if (static_source != nullptr)
      {
        Fail("legacy VTK restart data without a manifest cannot certify the "
             "active static source");
      }
      if (run_cfg.dg_backend == DGBackend::DGSEM)
      {
        Fail("DGSEM VTK restart requires a complete restart manifest");
      }
      if (request.legacy_step < 0)
      {
        Fail("legacy VTK cycle has no restart manifest; provide --restart-step");
      }
      info.step = request.legacy_step;
      info.legacy = true;
    }
    candidate = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(&ctx.spaces.vfes));
    local_metrics =
      ValidateAndReconstruct(piece, run_cfg, ctx, *candidate);
  }
  catch (const std::exception& ex)
  {
    error = ex.what();
  }
  CollectiveFailure(error, comm);

  info.cycle = entry.cycle;
  info.time = entry.time;
  MPI_Allreduce(&local_metrics.residual_max,
                &info.reconstruction_max_abs,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                comm);
  double sums[2] = {local_metrics.residual_sq, local_metrics.source_sq};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
  info.reconstruction_rel_l2 =
    std::sqrt(sums[0] / std::max(sums[1], std::numeric_limits<double>::min()));
  MPI_Allreduce(&local_metrics.point_max,
                &info.mesh_point_max_abs,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                comm);
  MPI_Allreduce(&local_metrics.condition_inf,
                &info.interpolation_condition_inf,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                comm);

  MFEM_VERIFY(candidate, "collective VTK restart candidate is unavailable");
  const mfem::igrlab::NonFiniteSummary finite =
    mfem::igrlab::ScanNonFinite(*candidate);
  if (finite.global_non_finite != 0)
  {
    MFEM_ABORT("VTK restart reconstructed a non-finite conservative state.");
  }
  const int face_qord =
    2 * run_cfg.opts.order + run_cfg.case_config.int_order_offset;
  const double rho_floor = run_cfg.case_config.rk3_config.rho_min;
  const double pressure_floor = run_cfg.case_config.rk3_config.p_min;
  const double rho_roundoff =
    64.0 * std::numeric_limits<double>::epsilon() *
    std::max(1.0, std::abs(rho_floor));
  const double pressure_roundoff =
    64.0 * std::numeric_limits<double>::epsilon() *
    std::max(1.0, std::abs(pressure_floor));
  auto admissibility = mfem::igrlab::ScanEulerAdmissibility(
    ctx.spaces.vfes,
    *candidate,
    ctx.dim,
    run_cfg.case_config.gamma,
    rho_floor - rho_roundoff,
    pressure_floor - pressure_roundoff,
    run_cfg.case_config.rk3_config.quad_order,
    {face_qord},
    mfem::igrlab::EulerAdmissibilityPointSet::Limiter);
  const auto nodal_admissibility = mfem::igrlab::ScanEulerAdmissibility(
    ctx.spaces.vfes,
    *candidate,
    ctx.dim,
    run_cfg.case_config.gamma,
    rho_floor - rho_roundoff,
    pressure_floor - pressure_roundoff,
    run_cfg.case_config.rk3_config.quad_order,
    {face_qord},
    mfem::igrlab::EulerAdmissibilityPointSet::Dof);
  admissibility.global_min_rho =
    std::min(admissibility.global_min_rho,
             nodal_admissibility.global_min_rho);
  admissibility.global_min_pressure =
    std::min(admissibility.global_min_pressure,
             nodal_admissibility.global_min_pressure);
  admissibility.global_bad_rho += nodal_admissibility.global_bad_rho;
  admissibility.global_bad_pressure += nodal_admissibility.global_bad_pressure;
  if (admissibility.first_bad_rank < 0 ||
      (nodal_admissibility.first_bad_rank >= 0 &&
       nodal_admissibility.first_bad_rank < admissibility.first_bad_rank))
  {
    admissibility.first_bad_rank = nodal_admissibility.first_bad_rank;
  }
  info.density_min = admissibility.global_min_rho;
  info.pressure_min = admissibility.global_min_pressure;
  if (admissibility.global_bad_rho != 0 || admissibility.global_bad_pressure != 0)
  {
    MFEM_ABORT("VTK restart state is not consumer-point admissible: bad_rho="
               << admissibility.global_bad_rho
               << " bad_pressure=" << admissibility.global_bad_pressure
               << " min_rho=" << admissibility.global_min_rho
               << " min_pressure=" << admissibility.global_min_pressure
               << " rho_floor=" << rho_floor
               << " pressure_floor=" << pressure_floor
               << " rho_roundoff=" << rho_roundoff
               << " pressure_roundoff=" << pressure_roundoff
               << " reconstruction_max_abs=" << info.reconstruction_max_abs
               << " reconstruction_rel_l2=" << info.reconstruction_rel_l2
               << " first_bad_rank=" << admissibility.first_bad_rank << ".");
  }
  state = *candidate;
  info.load_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  return info;
}

void WriteVTKRestartManifest(const std::string& dataset_dir,
                             const std::string& dataset_name,
                             int cycle,
                             mfem::real_t time,
                             long step,
                             const RunConfig& run_cfg,
                             const ExampleContext& ctx,
                             bool sigma_saved,
                             int vtk_compression_level,
                             const mfem::ParGridFunction* static_source)
{
  MFEM_VERIFY(cycle >= 0 && step >= 0 && std::isfinite(time),
              "Invalid VTK restart manifest state.");
  const MPI_Comm comm = ctx.Mesh().GetComm();
  int comm_rank = 0;
  int comm_size = 0;
  MFEM_VERIFY(MPI_Comm_rank(comm, &comm_rank) == MPI_SUCCESS &&
                MPI_Comm_size(comm, &comm_size) == MPI_SUCCESS &&
                comm_size > 0,
              "VTK restart manifest requires a live mesh communicator.");
  const std::uint64_t fingerprint =
    ComputeVTKRestartMeshFingerprint(run_cfg, ctx);
  int static_source_presence[2] = {static_source == nullptr ? 0 : 1,
                                   static_source == nullptr ? 0 : 1};
  MFEM_VERIFY(MPI_Allreduce(MPI_IN_PLACE,
                            &static_source_presence[0],
                            1,
                            MPI_INT,
                            MPI_MIN,
                            comm) == MPI_SUCCESS &&
                MPI_Allreduce(MPI_IN_PLACE,
                              &static_source_presence[1],
                              1,
                              MPI_INT,
                              MPI_MAX,
                              comm) == MPI_SUCCESS &&
                static_source_presence[0] == static_source_presence[1],
              "VTK restart static-source presence must agree on every rank.");
  const std::uint64_t static_source_fingerprint =
    static_source == nullptr
      ? 0
      : ComputeVTKRestartFieldFingerprint(*static_source, ctx);
  long long local_elements = ctx.Mesh().GetNE();
  long long global_elements = 0;
  std::vector<long long> local_element_counts(
    static_cast<std::size_t>(comm_size));
  MPI_Allgather(&local_elements,
                1,
                MPI_LONG_LONG,
                local_element_counts.data(),
                1,
                MPI_LONG_LONG,
                comm);
  MPI_Allreduce(&local_elements,
                &global_elements,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                comm);
  MPI_Barrier(comm);
  if (comm_rank != 0)
  {
    // The root publishes the completion marker only after every rank has
    // closed its piece.  Do not let peers leave this collective operation
    // until that marker is visible.
    MPI_Barrier(comm);
    return;
  }

  picojson::object object;
  object["schema_version"] =
    picojson::value(static_cast<double>(current_manifest_schema));
  object["complete"] = picojson::value(true);
  object["dataset"] = picojson::value(dataset_name);
  object["case"] = picojson::value(run_cfg.case_config.case_name);
  object["cycle"] = picojson::value(static_cast<double>(cycle));
  object["time"] = picojson::value(static_cast<double>(time));
  object["step"] = picojson::value(static_cast<double>(step));
  object["dt_save"] = picojson::value(static_cast<double>(run_cfg.opts.dt_save));
  object["next_cycle"] = picojson::value(static_cast<double>(cycle + 1));
  object["next_save_time"] =
    picojson::value(static_cast<double>(time + run_cfg.opts.dt_save));
  object["ranks"] = picojson::value(static_cast<double>(comm_size));
  object["piece_pattern"] = picojson::value(std::string("proc%06d.vtu"));
  object["local_elements_rank0"] = picojson::value(static_cast<double>(local_elements));
  picojson::array local_elements_json;
  for (const long long count : local_element_counts)
  {
    local_elements_json.emplace_back(static_cast<double>(count));
  }
  object["local_elements"] = picojson::value(local_elements_json);
  object["global_elements"] = picojson::value(static_cast<double>(global_elements));
  object["dimension"] = picojson::value(static_cast<double>(ctx.dim));
  object["equations"] = picojson::value(static_cast<double>(ctx.num_equations));
  object["order"] = picojson::value(static_cast<double>(run_cfg.opts.order));
  object["serial_refinement"] =
    picojson::value(static_cast<double>(run_cfg.opts.ser_ref_levels));
  object["parallel_refinement"] =
    picojson::value(static_cast<double>(run_cfg.opts.par_ref_levels));
  object["dg_backend"] = picojson::value(BackendName(run_cfg));
  object["time_integrator"] = picojson::value(
    std::string(TimeIntegratorToString(run_cfg.time_integrator)));
  object["dgsem_stabilization"] = picojson::value(
    std::string(DGSEMStabilizationToString(run_cfg.dgsem_stab)));
  object["dgsem_volume_flux"] = picojson::value(
    std::string(DGSEMVolumeFluxToString(run_cfg.dgsem_volume_flux)));
  object["dgsem_volume_dissipation"] = picojson::value(std::string(
    DGSEMVolumeDissipationToString(run_cfg.dgsem_volume_dissipation)));
  object["dgsem_surface_flux"] = picojson::value(
    std::string(DGSEMSurfaceFluxToString(run_cfg.dgsem_surface_flux)));
  object["dgsem_entropy_jacobian"] = picojson::value(std::string(
    DGSEMEntropyJacobianToString(run_cfg.dgsem_entropy_jacobian)));
  object["basis"] = picojson::value(BasisName(run_cfg));
  object["ordering"] = picojson::value(std::string("byNODES"));
  const auto vtk_config = MakeCaseVTKConfig(run_cfg.case_config);
  const bool float32 = vtk_config.format == mfem::VTKFormat::BINARY32;
  const char* vtk_format = vtk_config.format == mfem::VTKFormat::ASCII
                             ? "ascii"
                             : (float32 ? "binary32" : "binary");
  object["data_type"] =
    picojson::value(std::string(float32 ? "Float32" : "Float64"));
  object["vtk_format"] = picojson::value(std::string(vtk_format));
  object["compression"] = picojson::value(std::string(
    vtk_compression_level == 0 ? "none" : "vtkZLibDataCompressor"));
  object["compression_level"] =
    picojson::value(static_cast<double>(vtk_compression_level));
  object["high_order"] = picojson::value(vtk_config.high_order_output);
  object["level_of_detail"] =
    picojson::value(static_cast<double>(run_cfg.opts.order));
  object["mesh_fingerprint"] = picojson::value(FingerprintString(fingerprint));
  object["static_source_present"] = picojson::value(static_source != nullptr);
  object["static_source_fingerprint"] = picojson::value(
    static_source == nullptr ? std::string("none")
                             : FingerprintString(static_source_fingerprint));
  object["required_fields"] =
    picojson::value(std::string("density:1,momentum:") +
                    std::to_string(ctx.dim) + ",energy:1");
  object["gamma"] = picojson::value(run_cfg.case_config.gamma);
  object["rho_floor"] =
    picojson::value(run_cfg.case_config.rk3_config.rho_min);
  object["pressure_floor"] =
    picojson::value(run_cfg.case_config.rk3_config.p_min);
  object["zhang_shu_limiter"] =
    picojson::value(run_cfg.case_config.limiter_pipeline.zhang_shu);
  object["zhang_shu_points"] = picojson::value(run_cfg.opts.zhang_shu_points);
  object["zhang_shu_vtk_points"] =
    picojson::value(run_cfg.opts.zhang_shu_vtk_points);
  object["igr_enabled"] = picojson::value(sigma_saved);
  object["igr_space"] = picojson::value(IGRSpaceName(run_cfg));
  object["igr_grad"] = picojson::value(IGRGradName(run_cfg));
  object["c_alpha"] = picojson::value(
    run_cfg.igr_opts.c_alpha_user > 0.0 ? run_cfg.igr_opts.c_alpha_user : 1.0);
  object["alpha_length_policy"] = picojson::value(std::string(
    IGRAlphaLengthScalePolicyToString(ResolveIGRAlphaLengthScalePolicy(run_cfg))));
  object["ip_kind"] = picojson::value(std::string(
    IGRIPKindToString(ParseIGRIPKind(run_cfg.igr_opts.ip_kind))));
  object["exact_dg_face"] =
    picojson::value(run_cfg.igr_opts.exact_dg_face_operator);
  object["igr_invariant_domain_flux"] =
    picojson::value(run_cfg.igr_opts.igr_invariant_domain_flux);
  const auto sigma_cadence = ResolveEffectiveIGRSigmaUpdateCadence(run_cfg);
  object["sigma_cadence"] = picojson::value(
    std::string(IGRSigmaUpdateCadenceToString(sigma_cadence)));
  object["sigma_update_interval"] = picojson::value(static_cast<double>(
    ResolveEffectiveIGRSigmaUpdateInterval(run_cfg)));
  object["sigma_saved"] = picojson::value(sigma_saved);
  object["sigma_restart_policy"] =
    picojson::value(SigmaRestartPolicy(run_cfg));

  std::ostringstream cycle_name;
  cycle_name << "Cycle" << std::setfill('0') << std::setw(6) << cycle;
  const fs::path cycle_dir = fs::path(dataset_dir) / cycle_name.str();
  const fs::path final_path = cycle_dir / "restart.json";
  const fs::path temp_path = cycle_dir / "restart.json.tmp";
  std::ofstream output(temp_path, std::ios::trunc);
  if (!output) { MFEM_ABORT("Unable to write restart manifest " << temp_path); }
  output << picojson::value(object).serialize(true) << '\n';
  output.close();
  if (!output) { MFEM_ABORT("Unable to finish restart manifest " << temp_path); }
  std::error_code ec;
  fs::rename(temp_path, final_path, ec);
  if (ec)
  {
    MFEM_ABORT("Unable to atomically publish restart manifest "
               << final_path << ": " << ec.message());
  }
  MPI_Barrier(comm);
}

} // namespace dg_sim

namespace dg_sim
{
namespace
{

void LoadXML(const fs::path& path, XMLDocument& document)
{
  const auto rc = document.LoadFile(path.string().c_str());
  if (rc != tinyxml2::XML_SUCCESS)
  {
    Fail("unable to parse XML file '" + path.string() + "': " +
         document.ErrorStr());
  }
}

fs::path ResolvePVDPath(const std::string& input)
{
  const fs::path path = fs::absolute(input);
  if (fs::is_regular_file(path))
  {
    if (path.extension() != ".pvd") { Fail("--restart-vtk file must have .pvd extension"); }
    return path;
  }
  if (!fs::is_directory(path)) { Fail("restart VTK path does not exist: " + path.string()); }
  std::vector<fs::path> candidates;
  for (const auto& item : fs::directory_iterator(path))
  {
    if (item.is_regular_file() && item.path().extension() == ".pvd")
    {
      candidates.push_back(item.path());
    }
  }
  if (candidates.size() != 1)
  {
    Fail("restart dataset directory must contain exactly one .pvd file");
  }
  return candidates.front();
}

int CycleFromPVTUPath(const fs::path& path)
{
  const std::string name = path.parent_path().filename().string();
  constexpr const char* prefix = "Cycle";
  if (name.rfind(prefix, 0) != 0 || name.size() == std::strlen(prefix))
  {
    Fail("PVD entry does not use an MFEM Cycle directory: " + name);
  }
  const std::string digits = name.substr(std::strlen(prefix));
  if (!std::all_of(digits.begin(), digits.end(),
                   [](unsigned char c) { return std::isdigit(c); }))
  {
    Fail("invalid cycle directory name: " + name);
  }
  return std::stoi(digits);
}

std::vector<PVDEntry> ReadPVD(const fs::path& pvd)
{
  XMLDocument document;
  LoadXML(pvd, document);
  const XMLElement* root = document.FirstChildElement("VTKFile");
  if (root == nullptr || RequiredAttribute(root, "type") != "Collection")
  {
    Fail("restart PVD is not a VTK Collection");
  }
  const XMLElement* collection = root->FirstChildElement("Collection");
  if (collection == nullptr) { Fail("restart PVD has no Collection element"); }

  std::vector<PVDEntry> entries;
  int previous_cycle = -1;
  double previous_time = -std::numeric_limits<double>::infinity();
  for (const XMLElement* item = collection->FirstChildElement("DataSet"); item;
       item = item->NextSiblingElement("DataSet"))
  {
    double time = 0.0;
    if (item->QueryDoubleAttribute("timestep", &time) !=
          tinyxml2::XML_SUCCESS ||
        !std::isfinite(time))
    {
      Fail("PVD contains an invalid timestep");
    }
    const fs::path relative = RequiredAttribute(item, "file");
    const fs::path pvtu = pvd.parent_path() / relative;
    const int cycle = CycleFromPVTUPath(relative);
    if (cycle <= previous_cycle || time <= previous_time)
    {
      Fail("PVD cycle and timestep entries must be strictly increasing");
    }
    if (!fs::is_regular_file(pvtu))
    {
      Fail("PVD references a missing PVTU file: " + pvtu.string());
    }
    entries.push_back({cycle, time, pvtu});
    previous_cycle = cycle;
    previous_time = time;
  }
  if (entries.empty()) { Fail("PVD contains no restart cycles"); }
  return entries;
}

PVDEntry SelectEntry(const std::vector<PVDEntry>& entries,
                     int requested_cycle,
                     long legacy_step)
{
  if (requested_cycle < 0)
  {
    if (legacy_step >= 0) { return entries.back(); }
    for (auto entry = entries.rbegin(); entry != entries.rend(); ++entry)
    {
      const fs::path manifest_path = entry->pvtu.parent_path() / "restart.json";
      if (!fs::is_regular_file(manifest_path)) { continue; }
      try
      {
        const Manifest manifest = ReadManifest(manifest_path);
        if (IsSupportedManifestSchema(manifest.schema) && manifest.complete)
        {
          return *entry;
        }
      }
      catch (const std::exception&)
      {
        // An interrupted or corrupt newest cycle does not hide an earlier
        // complete checkpoint. Explicit selection still diagnoses it.
      }
    }
    return entries.back();
  }
  const auto found = std::find_if(entries.begin(), entries.end(),
                                  [requested_cycle](const PVDEntry& item)
                                  { return item.cycle == requested_cycle; });
  if (found == entries.end())
  {
    Fail("requested restart cycle " + std::to_string(requested_cycle) +
         " is not present in the PVD");
  }
  return *found;
}

const picojson::value& RequiredJSON(const picojson::object& object,
                                    const std::string& key)
{
  const auto found = object.find(key);
  if (found == object.end()) { Fail("restart manifest is missing '" + key + "'"); }
  return found->second;
}

long JSONLong(const picojson::object& object, const std::string& key)
{
  const auto& value = RequiredJSON(object, key);
  if (!value.is<double>() || !std::isfinite(value.get<double>()))
  {
    Fail("restart manifest field '" + key + "' is not numeric");
  }
  const double number = value.get<double>();
  if (std::floor(number) != number || number < 0.0 ||
      number > static_cast<double>(std::numeric_limits<long>::max()))
  {
    Fail("restart manifest field '" + key + "' is not a valid integer");
  }
  return static_cast<long>(number);
}

std::string JSONString(const picojson::object& object, const std::string& key)
{
  const auto& value = RequiredJSON(object, key);
  if (!value.is<std::string>())
  {
    Fail("restart manifest field '" + key + "' is not a string");
  }
  return value.get<std::string>();
}

double JSONDouble(const picojson::object& object, const std::string& key)
{
  const auto& value = RequiredJSON(object, key);
  if (!value.is<double>() || !std::isfinite(value.get<double>()))
  {
    Fail("restart manifest field '" + key + "' is not finite numeric data");
  }
  return value.get<double>();
}

bool JSONBool(const picojson::object& object, const std::string& key)
{
  const auto& value = RequiredJSON(object, key);
  if (!value.is<bool>())
  {
    Fail("restart manifest field '" + key + "' is not boolean");
  }
  return value.get<bool>();
}

Manifest ReadManifest(const fs::path& path)
{
  std::ifstream input(path);
  if (!input) { Fail("unable to open restart manifest: " + path.string()); }
  picojson::value value;
  const std::string error = picojson::parse(value, input);
  if (!error.empty() || !value.is<picojson::object>())
  {
    Fail("invalid restart manifest '" + path.string() + "': " + error);
  }
  const auto& object = value.get<picojson::object>();
  Manifest result;
  result.schema = static_cast<int>(JSONLong(object, "schema_version"));
  const auto& complete = RequiredJSON(object, "complete");
  if (!complete.is<bool>()) { Fail("restart manifest 'complete' is not boolean"); }
  result.complete = complete.get<bool>();
  result.cycle = static_cast<int>(JSONLong(object, "cycle"));
  const auto& time = RequiredJSON(object, "time");
  if (!time.is<double>() || !std::isfinite(time.get<double>()))
  {
    Fail("restart manifest time is invalid");
  }
  result.time = time.get<double>();
  result.step = JSONLong(object, "step");
  result.ranks = static_cast<int>(JSONLong(object, "ranks"));
  result.dim = static_cast<int>(JSONLong(object, "dimension"));
  result.equations = static_cast<int>(JSONLong(object, "equations"));
  result.order = static_cast<int>(JSONLong(object, "order"));
  result.serial_refinement =
    static_cast<int>(JSONLong(object, "serial_refinement"));
  result.parallel_refinement =
    static_cast<int>(JSONLong(object, "parallel_refinement"));
  result.igr_enabled = JSONBool(object, "igr_enabled");
  result.exact_dg_face = JSONBool(object, "exact_dg_face");
  result.gamma = JSONDouble(object, "gamma");
  result.c_alpha = JSONDouble(object, "c_alpha");
  result.sigma_update_interval =
    static_cast<int>(JSONLong(object, "sigma_update_interval"));
  result.case_name = JSONString(object, "case");
  result.dg_backend = JSONString(object, "dg_backend");
  result.basis = JSONString(object, "basis");
  result.ordering = JSONString(object, "ordering");
  result.data_type = JSONString(object, "data_type");
  result.mesh_fingerprint = JSONString(object, "mesh_fingerprint");
  result.zhang_shu_points = JSONString(object, "zhang_shu_points");
  result.igr_space = JSONString(object, "igr_space");
  result.igr_grad = JSONString(object, "igr_grad");
  result.alpha_length_policy = JSONString(object, "alpha_length_policy");
  result.ip_kind = JSONString(object, "ip_kind");
  result.sigma_cadence = JSONString(object, "sigma_cadence");
  if (result.schema >= typed_integrator_manifest_schema)
  {
    result.time_integrator = JSONString(object, "time_integrator");
    result.igr_invariant_domain_flux =
      JSONBool(object, "igr_invariant_domain_flux");
    result.zhang_shu_limiter = JSONBool(object, "zhang_shu_limiter");
    result.zhang_shu_vtk_points =
      JSONBool(object, "zhang_shu_vtk_points");
    result.rho_floor = JSONDouble(object, "rho_floor");
    result.pressure_floor = JSONDouble(object, "pressure_floor");
  }
  else if (result.schema == legacy_manifest_schema)
  {
    // Schema 1 predates typed SSPRK(10,4) and the invariant-domain flux.
    // Retain its established compatibility only under that legacy method.
    result.time_integrator = "rk3ssp";
    result.igr_invariant_domain_flux = false;
  }
  if (result.schema >= static_source_manifest_schema)
  {
    result.static_source_present =
      JSONBool(object, "static_source_present");
    result.static_source_fingerprint =
      JSONString(object, "static_source_fingerprint");
  }
  else
  {
    // Schemas 1 and 2 predate projected static-source identity.  They remain
    // usable only when the active run is source-free.
    result.static_source_present = false;
    result.static_source_fingerprint = "none";
  }
  if (result.schema >= transport_manifest_schema)
  {
    result.dgsem_stabilization =
      JSONString(object, "dgsem_stabilization");
    result.dgsem_volume_flux = JSONString(object, "dgsem_volume_flux");
    result.dgsem_volume_dissipation =
      JSONString(object, "dgsem_volume_dissipation");
    result.dgsem_surface_flux = JSONString(object, "dgsem_surface_flux");
    result.dgsem_entropy_jacobian =
      JSONString(object, "dgsem_entropy_jacobian");
    result.sigma_restart_policy =
      JSONString(object, "sigma_restart_policy");
  }
  return result;
}

std::string FingerprintString(std::uint64_t value)
{
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << value;
  return out.str();
}

std::string BackendName(const RunConfig& run_cfg)
{
  return run_cfg.dg_backend == DGBackend::DG ? "dg" : "dgsem";
}

std::string BasisName(const RunConfig& run_cfg)
{
  return run_cfg.dg_backend == DGBackend::DG ? "GaussLegendre" : "GaussLobatto";
}

std::string IGRSpaceName(const RunConfig& run_cfg)
{
  return IsL2IGRSpaceName(run_cfg.igr_opts.space) ? "l2" : "h1";
}

std::string IGRGradName(const RunConfig& run_cfg)
{
  return IsBrokenDGIGRGradName(run_cfg.igr_opts.grad) ? "brokenl2" : "h1";
}

std::string SigmaRestartPolicy(const RunConfig& run_cfg)
{
  return ResolveEffectiveIGRSigmaUpdateCadence(run_cfg) ==
           igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep
           ? "cold-recompute-and-reset-extrapolation-history"
           : "cold-recompute-from-conservative-state";
}

void ValidateManifest(const Manifest& manifest,
                      const PVDEntry& entry,
                      const RunConfig& run_cfg,
                      const ExampleContext& ctx,
                      std::uint64_t fingerprint,
                      bool igr_enabled,
                      const mfem::ParGridFunction* static_source,
                      std::uint64_t static_source_fingerprint)
{
  int comm_size = 0;
  MFEM_VERIFY(MPI_Comm_size(ctx.Mesh().GetComm(), &comm_size) == MPI_SUCCESS &&
                comm_size > 0,
              "VTK restart validation requires a live mesh communicator.");
  if (!manifest.complete)
  {
    Fail("restart manifest is incomplete");
  }
  if (!IsSupportedManifestSchema(manifest.schema))
  {
    Fail("restart manifest uses an unsupported schema");
  }
  if (manifest.schema == legacy_manifest_schema &&
      (run_cfg.time_integrator != TimeIntegrator::RK3SSP ||
       run_cfg.igr_opts.igr_invariant_domain_flux ||
       run_cfg.opts.zhang_shu_vtk_points))
  {
    Fail("restart manifest schema 1 is compatible only with rk3ssp and "
         "invariant-domain flux and VTK-point limiter enforcement disabled");
  }
  if (manifest.schema < static_source_manifest_schema &&
      static_source != nullptr)
  {
    Fail("restart manifest schema " + std::to_string(manifest.schema) +
         " cannot certify the active static source");
  }
  if (manifest.schema >= static_source_manifest_schema)
  {
    const bool active_static_source = static_source != nullptr;
    const std::string active_static_source_fingerprint =
      active_static_source ? FingerprintString(static_source_fingerprint)
                           : std::string("none");
    if (manifest.static_source_present != active_static_source ||
        manifest.static_source_fingerprint !=
          active_static_source_fingerprint)
    {
      Fail("restart manifest static source is incompatible with the active "
           "projected source");
    }
  }
  const auto same_time = std::abs(manifest.time - entry.time) <=
                         32.0 * std::numeric_limits<double>::epsilon() *
                           std::max(1.0, std::abs(entry.time));
  if (manifest.cycle != entry.cycle || !same_time)
  {
    Fail("restart manifest cycle/time does not match the PVD entry");
  }
  const double active_c_alpha = run_cfg.igr_opts.c_alpha_user > 0.0
                                  ? run_cfg.igr_opts.c_alpha_user
                                  : 1.0;
  const std::string active_alpha_policy = IGRAlphaLengthScalePolicyToString(
    ResolveIGRAlphaLengthScalePolicy(run_cfg));
  const auto active_sigma_cadence =
    ResolveEffectiveIGRSigmaUpdateCadence(run_cfg);
  const int active_sigma_interval =
    ResolveEffectiveIGRSigmaUpdateInterval(run_cfg);
  const bool active_nondefault_dgsem_transport =
    run_cfg.dg_backend == DGBackend::DGSEM &&
    (run_cfg.dgsem_stab != DGSEMStabilization::None ||
     run_cfg.dgsem_volume_flux != DGSEMVolumeFlux::Central ||
     run_cfg.dgsem_volume_dissipation != DGSEMVolumeDissipation::None ||
     run_cfg.dgsem_surface_flux != DGSEMSurfaceFlux::Rusanov ||
     run_cfg.dgsem_entropy_jacobian !=
       DGSEMEntropyJacobian::FiniteDifference);
  if (manifest.schema < transport_manifest_schema &&
      active_nondefault_dgsem_transport)
  {
    Fail("restart manifest schema " + std::to_string(manifest.schema) +
         " cannot certify the active non-default DGSEM transport tuple");
  }
  if (manifest.schema < transport_manifest_schema && igr_enabled &&
      active_sigma_cadence ==
        igrlab::IGRSigmaUpdateCadence::ExtrapolatedEveryStep)
  {
    Fail("restart manifest schema " + std::to_string(manifest.schema) +
         " cannot certify extrapolation-history reset semantics");
  }
  const auto close = [](double a, double b)
  {
    return std::abs(a - b) <= 64.0 * std::numeric_limits<double>::epsilon() *
                               std::max({1.0, std::abs(a), std::abs(b)});
  };
  const bool current_method_mismatch =
    manifest.schema >= typed_integrator_manifest_schema &&
    (manifest.time_integrator !=
       TimeIntegratorToString(run_cfg.time_integrator) ||
     manifest.igr_invariant_domain_flux !=
       run_cfg.igr_opts.igr_invariant_domain_flux ||
     manifest.zhang_shu_limiter !=
       run_cfg.case_config.limiter_pipeline.zhang_shu ||
     manifest.zhang_shu_vtk_points != run_cfg.opts.zhang_shu_vtk_points ||
     manifest.rho_floor != run_cfg.case_config.rk3_config.rho_min ||
     manifest.pressure_floor != run_cfg.case_config.rk3_config.p_min);
  const bool dgsem_transport_mismatch =
    manifest.schema >= transport_manifest_schema &&
    run_cfg.dg_backend == DGBackend::DGSEM &&
    (manifest.dgsem_stabilization !=
       DGSEMStabilizationToString(run_cfg.dgsem_stab) ||
     manifest.dgsem_volume_flux !=
       DGSEMVolumeFluxToString(run_cfg.dgsem_volume_flux) ||
     manifest.dgsem_volume_dissipation !=
       DGSEMVolumeDissipationToString(run_cfg.dgsem_volume_dissipation) ||
     manifest.dgsem_surface_flux !=
       DGSEMSurfaceFluxToString(run_cfg.dgsem_surface_flux) ||
     manifest.dgsem_entropy_jacobian !=
       DGSEMEntropyJacobianToString(run_cfg.dgsem_entropy_jacobian));
  const bool sigma_restart_policy_mismatch =
    manifest.schema >= transport_manifest_schema && igr_enabled &&
    manifest.sigma_restart_policy != SigmaRestartPolicy(run_cfg);
  if (manifest.ranks != comm_size || manifest.dim != ctx.dim ||
      manifest.equations != ctx.num_equations ||
      manifest.order != run_cfg.opts.order ||
      manifest.serial_refinement != run_cfg.opts.ser_ref_levels ||
      manifest.parallel_refinement != run_cfg.opts.par_ref_levels ||
      manifest.case_name != run_cfg.case_config.case_name ||
      manifest.dg_backend != BackendName(run_cfg) ||
      manifest.basis != BasisName(run_cfg) || manifest.ordering != "byNODES" ||
      manifest.data_type != "Float64" ||
      manifest.mesh_fingerprint != FingerprintString(fingerprint) ||
      !close(manifest.gamma, run_cfg.case_config.gamma) ||
      manifest.zhang_shu_points != run_cfg.opts.zhang_shu_points ||
      manifest.igr_enabled != igr_enabled ||
      current_method_mismatch ||
      dgsem_transport_mismatch || sigma_restart_policy_mismatch ||
      (igr_enabled &&
       (manifest.igr_space != IGRSpaceName(run_cfg) ||
        manifest.igr_grad != IGRGradName(run_cfg) ||
        !close(manifest.c_alpha, active_c_alpha) ||
        manifest.alpha_length_policy != active_alpha_policy ||
        manifest.ip_kind !=
          IGRIPKindToString(ParseIGRIPKind(run_cfg.igr_opts.ip_kind)) ||
        manifest.exact_dg_face != run_cfg.igr_opts.exact_dg_face_operator ||
        manifest.sigma_cadence !=
          IGRSigmaUpdateCadenceToString(active_sigma_cadence) ||
        manifest.sigma_update_interval != active_sigma_interval)))
  {
    Fail("restart manifest is incompatible with the active mesh/discretization");
  }
}

PVTUInfo ReadPVTU(const PVDEntry& entry, int dim, int ranks, int rank)
{
  XMLDocument document;
  LoadXML(entry.pvtu, document);
  const XMLElement* root = document.FirstChildElement("VTKFile");
  if (root == nullptr || RequiredAttribute(root, "type") != "PUnstructuredGrid")
  {
    Fail("selected restart file is not a PUnstructuredGrid");
  }
  const std::string byte_order = RequiredAttribute(root, "byte_order");
  FileByteSwapRequired(byte_order);
  const XMLElement* grid = root->FirstChildElement("PUnstructuredGrid");
  if (grid == nullptr) { Fail("PVTU has no PUnstructuredGrid element"); }

  const XMLElement* point_data = grid->FirstChildElement("PPointData");
  for (const auto& required :
       std::array<std::pair<const char*, int>, 3>{{{"density", 1},
                                                   {"momentum", dim},
                                                   {"energy", 1}}})
  {
    const XMLElement* array = FindDataArray(point_data, required.first);
    if (array == nullptr || RequiredAttribute(array, "type") != "Float64" ||
        Components(array) != required.second)
    {
      Fail(std::string("PVTU required field declaration is incompatible: ") +
           required.first);
    }
  }

  std::vector<fs::path> pieces;
  for (const XMLElement* piece = grid->FirstChildElement("Piece"); piece;
       piece = piece->NextSiblingElement("Piece"))
  {
    pieces.push_back(entry.pvtu.parent_path() /
                     RequiredAttribute(piece, "Source"));
  }
  if (static_cast<int>(pieces.size()) != ranks)
  {
    Fail("PVTU piece count does not match the active MPI rank count");
  }
  const std::string expected = "proc" + [&]
  {
    std::ostringstream value;
    value << std::setfill('0') << std::setw(6) << rank << ".vtu";
    return value.str();
  }();
  if (pieces[rank].filename() != expected)
  {
    Fail("PVTU piece ordering/naming is incompatible with same-rank restart");
  }
  if (!fs::is_regular_file(pieces[rank]))
  {
    Fail("missing restart rank piece: " + pieces[rank].string());
  }
  return {pieces[rank], byte_order};
}

PieceData ReadPiece(const PVTUInfo& info, int dim)
{
  XMLDocument document;
  LoadXML(info.piece, document);
  const XMLElement* root = document.FirstChildElement("VTKFile");
  if (root == nullptr || RequiredAttribute(root, "type") != "UnstructuredGrid")
  {
    Fail("rank piece is not an UnstructuredGrid");
  }
  const std::string byte_order = RequiredAttribute(root, "byte_order");
  if (byte_order != info.byte_order) { Fail("PVTU/VTU byte_order mismatch"); }
  const std::string header_type =
    root->Attribute("header_type") ? root->Attribute("header_type") : "UInt32";
  const char* compressor_attr = root->Attribute("compressor");
  const bool compressed = compressor_attr != nullptr;
  if (compressed && std::string(compressor_attr) != "vtkZLibDataCompressor")
  {
    Fail("unsupported VTK compressor");
  }
  const XMLElement* grid = root->FirstChildElement("UnstructuredGrid");
  const XMLElement* piece = grid ? grid->FirstChildElement("Piece") : nullptr;
  if (piece == nullptr) { Fail("VTU has no Piece element"); }

  PieceData data;
  data.points_count = RequiredIntAttribute(piece, "NumberOfPoints");
  data.cells_count = RequiredIntAttribute(piece, "NumberOfCells");
  if (data.points_count < 0 || data.cells_count < 0) { Fail("negative VTU sizes"); }

  const XMLElement* points = piece->FirstChildElement("Points");
  const XMLElement* points_array = points ? points->FirstChildElement("DataArray") : nullptr;
  if (Components(points_array) != 3) { Fail("VTU Points must have three components"); }
  data.points = ReadDataArray<double>(points_array,
                                      "Float64",
                                      3ULL * data.points_count,
                                      byte_order,
                                      header_type,
                                      compressed);

  const XMLElement* cells = piece->FirstChildElement("Cells");
  data.connectivity = ReadDataArray<std::int32_t>(FindDataArray(cells, "connectivity"),
                                                  "Int32",
                                                  data.points_count,
                                                  byte_order,
                                                  header_type,
                                                  compressed);
  data.offsets = ReadDataArray<std::int32_t>(FindDataArray(cells, "offsets"),
                                             "Int32",
                                             data.cells_count,
                                             byte_order,
                                             header_type,
                                             compressed);
  data.types = ReadDataArray<std::uint8_t>(FindDataArray(cells, "types"),
                                           "UInt8",
                                           data.cells_count,
                                           byte_order,
                                           header_type,
                                           compressed);

  const XMLElement* cell_data = piece->FirstChildElement("CellData");
  data.attributes = ReadDataArray<std::int32_t>(FindDataArray(cell_data, "attribute"),
                                                "Int32",
                                                data.cells_count,
                                                byte_order,
                                                header_type,
                                                compressed);
  const XMLElement* point_data = piece->FirstChildElement("PointData");
  const XMLElement* density = FindDataArray(point_data, "density");
  const XMLElement* momentum = FindDataArray(point_data, "momentum");
  const XMLElement* energy = FindDataArray(point_data, "energy");
  if (Components(density) != 1 || Components(momentum) != dim ||
      Components(energy) != 1)
  {
    Fail("VTU conservative field component count mismatch");
  }
  data.density = ReadDataArray<double>(density, "Float64", data.points_count,
                                       byte_order, header_type, compressed);
  data.momentum = ReadDataArray<double>(momentum, "Float64",
                                        static_cast<std::size_t>(dim) * data.points_count,
                                        byte_order, header_type, compressed);
  data.energy = ReadDataArray<double>(energy, "Float64", data.points_count,
                                      byte_order, header_type, compressed);
  return data;
}

void CollectiveFailure(const std::string& local_error, MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int bad_rank = local_error.empty() ? std::numeric_limits<int>::max() : rank;
  MPI_Allreduce(MPI_IN_PLACE, &bad_rank, 1, MPI_INT, MPI_MIN, comm);
  if (bad_rank != std::numeric_limits<int>::max())
  {
    if (!local_error.empty())
    {
      std::cerr << "[vtk-restart] rank=" << rank << " error=" << local_error << '\n';
    }
    MFEM_ABORT("VTK restart validation failed on rank " << bad_rank << ".");
  }
}

} // namespace

} // namespace dg_sim
