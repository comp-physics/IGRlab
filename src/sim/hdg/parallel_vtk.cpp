#include "sim/hdg/parallel_vtk.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
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

} // namespace

HDGParaViewSeries::HDGParaViewSeries(const HDGParaViewRequest& request)
{
  ValidateRequest(request);

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

  for (const HDGInteriorFieldView& view : request.interior_fields)
  {
    collection_->RegisterField(view.name, &view.values);
  }
}

void HDGParaViewSeries::Write(const HDGParaViewMetadata& metadata)
{
  ValidateMetadata(metadata);
  collection_->SetCycle(metadata.cycle);
  collection_->SetTime(metadata.time);
  collection_->Save();
  if (collection_->Error() != mfem::DataCollection::No_Error)
  {
    throw std::runtime_error("MFEM failed to write HDG ParaView output");
  }
}

} // namespace dg_sim::hdg
