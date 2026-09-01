#include "sim/hdg/parallel_vtk.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace dg_sim::hdg
{
namespace
{

bool IsParaViewFieldName(const std::string& name)
{
  const auto identifier_character = [](const unsigned char value)
  { return std::isalnum(value) != 0 || value == '_'; };
  return !name.empty() &&
      (std::isalpha(static_cast<unsigned char>(name.front())) != 0 ||
       name.front() == '_') &&
      std::all_of(name.begin() + 1, name.end(), identifier_character);
}

void ValidateRequest(const HDGParaViewRequest& request)
{
  if (request.options.prefix_path.empty())
  {
    throw std::invalid_argument("HDG ParaView prefix_path must not be empty");
  }
  if (request.options.collection_name.empty())
  {
    throw std::invalid_argument("HDG ParaView collection_name must not be empty");
  }
  if (request.options.levels_of_detail < 0)
  {
    throw std::invalid_argument(
        "HDG ParaView levels_of_detail must be non-negative");
  }
  if (request.interior_fields.empty())
  {
    throw std::invalid_argument(
        "HDG ParaView requires at least one physical interior field");
  }
  if (request.options.restart &&
      (request.options.restart->last_cycle < 0 ||
       !std::isfinite(request.options.restart->committed_time)))
  {
    throw std::invalid_argument(
        "HDG ParaView restart seed must have a non-negative cycle and "
        "finite committed time");
  }

  std::unordered_set<std::string> names;
  for (const HDGInteriorFieldView& view : request.interior_fields)
  {
    if (!IsParaViewFieldName(view.name) || !names.insert(view.name).second)
    {
      throw std::invalid_argument(
          "HDG ParaView field names must be unique identifier-like strings");
    }

    mfem::ParFiniteElementSpace* const space = view.values.ParFESpace();
    if (space == nullptr || space->GetParMesh() != &request.mesh)
    {
      throw std::invalid_argument(
          "HDG ParaView field must use the supplied externally owned ParMesh");
    }
  }
}

void ValidateMetadata(const HDGParaViewMetadata& metadata)
{
  if (metadata.cycle < 0 || metadata.step < 0)
  {
    throw std::invalid_argument(
        "HDG ParaView cycle and step must be non-negative");
  }
  if (!std::isfinite(metadata.time))
  {
    throw std::invalid_argument("HDG ParaView time must be finite");
  }
}

bool CollectiveAll(const MPI_Comm communicator, const bool local) noexcept
{
  const int input = local ? 1 : 0;
  int output = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
          MPI_SUCCESS &&
      output == 1;
}

bool ExactMetadataConsensus(const MPI_Comm communicator,
                            const HDGParaViewMetadata& metadata) noexcept
{
  int minimum_cycle = 0;
  int maximum_cycle = 0;
  int minimum_step = 0;
  int maximum_step = 0;
  std::uint64_t time_bits = 0;
  std::uint64_t minimum_time_bits = 0;
  std::uint64_t maximum_time_bits = 0;
  static_assert(sizeof(time_bits) == sizeof(metadata.time),
                "HDG ParaView metadata requires IEEE binary64 doubles");
  std::memcpy(&time_bits, &metadata.time, sizeof(time_bits));
  bool reduced = true;
  reduced =
      (MPI_Allreduce(&metadata.cycle, &minimum_cycle, 1, MPI_INT, MPI_MIN,
                     communicator) == MPI_SUCCESS) && reduced;
  reduced =
      (MPI_Allreduce(&metadata.cycle, &maximum_cycle, 1, MPI_INT, MPI_MAX,
                     communicator) == MPI_SUCCESS) && reduced;
  reduced =
      (MPI_Allreduce(&metadata.step, &minimum_step, 1, MPI_INT, MPI_MIN,
                     communicator) == MPI_SUCCESS) && reduced;
  reduced =
      (MPI_Allreduce(&metadata.step, &maximum_step, 1, MPI_INT, MPI_MAX,
                     communicator) == MPI_SUCCESS) && reduced;
  reduced =
      (MPI_Allreduce(&time_bits, &minimum_time_bits, 1, MPI_UINT64_T,
                     MPI_MIN, communicator) == MPI_SUCCESS) && reduced;
  reduced =
      (MPI_Allreduce(&time_bits, &maximum_time_bits, 1, MPI_UINT64_T,
                     MPI_MAX, communicator) == MPI_SUCCESS) && reduced;
  return CollectiveAll(
      communicator,
      reduced && minimum_cycle == maximum_cycle &&
          minimum_step == maximum_step &&
          minimum_time_bits == maximum_time_bits);
}

std::string Padded(const int value)
{
  std::ostringstream stream;
  stream << std::setw(6) << std::setfill('0') << value;
  return stream.str();
}

bool StructurallyCompleteXml(const std::filesystem::path& path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  if (size <= 0) return false;
  constexpr std::streamoff tail_capacity = 512;
  const std::streamoff tail_size = std::min(size, tail_capacity);
  input.seekg(size - tail_size);
  std::string tail(static_cast<std::size_t>(tail_size), '\0');
  input.read(tail.data(), tail_size);
  return input && tail.find("</VTKFile>") != std::string::npos;
}

bool TextContains(const std::filesystem::path& path,
                  const std::string& token)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  return !input.bad() && text.find(token) != std::string::npos;
}

struct HistoricalPublication
{
  int cycle = -1;
  double time = 0.0;
};

bool ValidateRestartHistory(
    const std::filesystem::path& root,
    const std::string& collection_name,
    const HDGParaViewOptions::RestartSeed& seed,
    const int rank,
    const int ranks)
{
  const std::filesystem::path pvd = root / (collection_name + ".pvd");
  if (!StructurallyCompleteXml(pvd)) return false;
  std::ifstream input(pvd);
  if (!input) return false;
  const std::regex entry(
      "timestep=\"([^\"]+)\".*file=\"Cycle([0-9]+)/data\\.pvtu\"");
  std::vector<HistoricalPublication> publications;
  std::string line;
  try
  {
    std::smatch match;
    while (std::getline(input, line))
    {
      if (!std::regex_search(line, match, entry)) continue;
      if (match.size() != 3) return false;
      const double time = std::stod(match[1].str());
      const int cycle = std::stoi(match[2].str());
      if (!std::isfinite(time) || cycle < 0) return false;
      if (!publications.empty() &&
          (cycle != publications.back().cycle + 1 ||
           !(time > publications.back().time)))
        return false;
      publications.push_back({cycle, time});
    }
  }
  catch (...)
  {
    return false;
  }
  if (input.bad() || publications.empty() ||
      publications.front().cycle > 1 ||
      publications.back().cycle != seed.last_cycle ||
      publications.back().time > seed.committed_time)
    return false;

  for (const HistoricalPublication& publication : publications)
  {
    const std::filesystem::path cycle =
        root / ("Cycle" + Padded(publication.cycle));
    const std::filesystem::path local_vtu =
        cycle / ("proc" + Padded(rank) + ".vtu");
    if (!StructurallyCompleteXml(local_vtu)) return false;
    if (rank == 0)
    {
      const std::filesystem::path pvtu = cycle / "data.pvtu";
      if (!StructurallyCompleteXml(pvtu)) return false;
      for (int source_rank = 0; source_rank < ranks; ++source_rank)
      {
        if (!TextContains(pvtu,
                          "proc" + Padded(source_rank) + ".vtu"))
          return false;
      }
    }
  }
  return true;
}

} // namespace

HDGParaViewSeries::HDGParaViewSeries(const HDGParaViewRequest& request)
{
  ValidateRequest(request);

  communicator_ = request.mesh.GetComm();
  if (communicator_ == MPI_COMM_NULL ||
      MPI_Comm_rank(communicator_, &rank_) != MPI_SUCCESS ||
      MPI_Comm_size(communicator_, &ranks_) != MPI_SUCCESS ||
      rank_ < 0 || ranks_ <= 0)
  {
    throw std::invalid_argument(
        "HDG ParaView requires a live mesh communicator");
  }
  collection_name_ = request.options.collection_name;
  collection_path_ =
      (std::filesystem::path(request.options.prefix_path) /
       collection_name_).string();

  if (request.options.restart)
  {
    const bool history_valid = ValidateRestartHistory(
        std::filesystem::path(collection_path_), collection_name_,
        *request.options.restart, rank_, ranks_);
    if (!CollectiveAll(communicator_, history_valid))
    {
      throw std::invalid_argument(
          "HDG ParaView restart history is missing, stale, or malformed");
    }
    last_cycle_ = request.options.restart->last_cycle;
    committed_time_floor_ = request.options.restart->committed_time;
    has_committed_time_floor_ = true;
  }

  // Passing the caller's ParMesh directly preserves its rank-local geometry
  // and partition.  ParaViewDataCollection writes one VTU per rank and its
  // rank-zero PVTU/PVD coordination; this adapter performs no gather.
  collection_ = std::make_unique<mfem::ParaViewDataCollection>(
      request.options.collection_name, &request.mesh);
  collection_->SetOwnData(false);
  collection_->SetPrefixPath(request.options.prefix_path);
  collection_->SetLevelsOfDetail(request.options.levels_of_detail);
  collection_->SetHighOrderOutput(request.options.high_order_output);
  collection_->SetDataFormat(request.options.data_format);
  collection_->UseRestartMode(request.options.restart.has_value());

  for (const HDGInteriorFieldView& view : request.interior_fields)
  {
    collection_->RegisterField(view.name, &view.values);
  }
}

void HDGParaViewSeries::Write(const HDGParaViewMetadata& metadata)
{
  std::string cycle;
  std::filesystem::path root;
  std::filesystem::path cycle_path;
  std::filesystem::path local_vtu;
  std::filesystem::path pvtu;
  std::filesystem::path pvd;
  bool metadata_ready = true;
  try
  {
    ValidateMetadata(metadata);
    metadata_ready = metadata.cycle > last_cycle_ &&
        (!has_committed_time_floor_ ||
         metadata.time > committed_time_floor_);
    cycle = "Cycle" + Padded(metadata.cycle);
    root = std::filesystem::path(collection_path_);
    cycle_path = root / cycle;
    local_vtu = cycle_path / ("proc" + Padded(rank_) + ".vtu");
    pvtu = cycle_path / "data.pvtu";
    pvd = root / (collection_name_ + ".pvd");
  }
  catch (...)
  {
    metadata_ready = false;
  }
  if (!CollectiveAll(communicator_, metadata_ready))
  {
    throw std::invalid_argument(
        "HDG ParaView metadata is invalid or cycles did not increase");
  }
  if (!ExactMetadataConsensus(communicator_, metadata))
  {
    throw std::invalid_argument(
        "HDG ParaView metadata differs between MPI ranks");
  }

  bool targets_ready = true;
  try
  {
    std::error_code error;
    std::filesystem::remove(local_vtu, error);
    targets_ready = !error && !std::filesystem::exists(local_vtu);
    if (rank_ == 0)
    {
      error.clear();
      std::filesystem::remove(pvtu, error);
      targets_ready = targets_ready && !error &&
          !std::filesystem::exists(pvtu);
      if (last_cycle_ < 0)
      {
        error.clear();
        std::filesystem::remove(pvd, error);
        targets_ready = targets_ready && !error &&
            !std::filesystem::exists(pvd);
      }
    }
  }
  catch (...)
  {
    targets_ready = false;
  }
  if (!CollectiveAll(communicator_, targets_ready))
  {
    throw std::runtime_error(
        "HDG ParaView could not prepare fresh publication targets");
  }

  collection_->SetCycle(metadata.cycle);
  collection_->SetTime(metadata.time);

  bool local_saved = true;
  try
  {
    collection_->Save();
    local_saved =
        collection_->Error() == mfem::DataCollection::No_Error;
  }
  catch (...)
  {
    local_saved = false;
  }
  if (!CollectiveAll(communicator_, local_saved))
  {
    throw std::runtime_error("MFEM failed to write HDG ParaView output");
  }

  bool local_complete = false;
  try
  {
    local_complete = StructurallyCompleteXml(local_vtu);
    if (local_complete && rank_ == 0)
    {
      local_complete = StructurallyCompleteXml(pvtu) &&
          StructurallyCompleteXml(pvd) &&
          TextContains(pvd, cycle + "/data.pvtu");
      for (int rank = 0; local_complete && rank < ranks_; ++rank)
      {
        local_complete =
            TextContains(pvtu, "proc" + Padded(rank) + ".vtu");
      }
    }
  }
  catch (...)
  {
    local_complete = false;
  }
  if (!CollectiveAll(communicator_, local_complete))
  {
    throw std::runtime_error(
        "HDG ParaView publication is incomplete on at least one rank");
  }
  last_cycle_ = metadata.cycle;
  committed_time_floor_ = metadata.time;
  has_committed_time_floor_ = true;
}

} // namespace dg_sim::hdg
