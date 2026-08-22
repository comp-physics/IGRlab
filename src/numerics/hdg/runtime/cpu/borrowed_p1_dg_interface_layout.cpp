#include "numerics/hdg/runtime/cpu/borrowed_p1_dg_interface_layout.hpp"

#include "numerics/hdg/trace/trace_dg_mpi.hpp"

#include <array>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cpu
{
namespace
{

int DecodeSignedVDof(const int signed_vdof) noexcept
{
  if (signed_vdof == std::numeric_limits<int>::min()) { return -1; }
  return signed_vdof < 0 ? -1 - signed_vdof : signed_vdof;
}

int SignedVDofSign(const int signed_vdof) noexcept
{
  return signed_vdof < 0 ? -1 : 1;
}

bool ValidSignedVDofs(const std::vector<int>& signed_vdofs,
                      const int local_size, const int expected_size) noexcept
{
  if (local_size < 0 ||
      signed_vdofs.size() != static_cast<std::size_t>(expected_size))
  {
    return false;
  }
  for (const int signed_vdof : signed_vdofs)
  {
    const int vdof = DecodeSignedVDof(signed_vdof);
    if (vdof < 0 || vdof >= local_size) { return false; }
  }
  return true;
}

bool CaptureElementVDofs(mfem::ParFiniteElementSpace& space,
                         const int expected_per_element,
                         std::vector<std::vector<int>>& destination)
{
  const mfem::ParMesh* const mesh = space.GetParMesh();
  if (mesh == nullptr || mesh->GetNE() < 0 || expected_per_element <= 0)
  {
    return false;
  }
  std::vector<std::vector<int>> next;
  next.reserve(static_cast<std::size_t>(mesh->GetNE()));
  for (int element = 0; element < mesh->GetNE(); ++element)
  {
    mfem::Array<int> vdofs;
    space.GetElementVDofs(element, vdofs);
    if (vdofs.Size() != expected_per_element) { return false; }
    const int* const data = vdofs.HostRead();
    if (data == nullptr) { return false; }
    std::vector<int> signed_vdofs(data, data + vdofs.Size());
    if (!ValidSignedVDofs(signed_vdofs, space.GetVSize(),
                          expected_per_element))
    {
      return false;
    }
    next.push_back(std::move(signed_vdofs));
  }
  destination = std::move(next);
  return true;
}

template <std::size_t size>
bool GatherSigned(const std::vector<int>& signed_vdofs,
                  const mfem::Vector& local_values,
                  mfem::Vector& element_values) noexcept
{
  if (signed_vdofs.size() != size ||
      element_values.Size() != static_cast<int>(size))
  {
    return false;
  }
  try
  {
    const double* const input = local_values.HostRead();
    if (input == nullptr && !signed_vdofs.empty()) { return false; }
    std::array<double, size> next{};
    for (std::size_t index = 0; index < size; ++index)
    {
      const int signed_vdof = signed_vdofs[index];
      const int vdof = DecodeSignedVDof(signed_vdof);
      if (vdof < 0 || vdof >= local_values.Size()) { return false; }
      next[index] = static_cast<double>(SignedVDofSign(signed_vdof)) *
          input[vdof];
    }
    double* const output = element_values.HostWrite();
    if (output == nullptr && size != 0) { return false; }
    for (std::size_t index = 0; index < size; ++index)
    {
      output[index] = next[index];
    }
    return true;
  }
  catch (const std::exception&)
  {
    return false;
  }
}

template <std::size_t size>
bool ScatterSigned(const std::vector<int>& signed_vdofs,
                   const mfem::Vector& element_values,
                   mfem::Vector& local_values) noexcept
{
  if (signed_vdofs.size() != size ||
      element_values.Size() != static_cast<int>(size))
  {
    return false;
  }
  try
  {
    const double* const input = element_values.HostRead();
    if (input == nullptr && size != 0) { return false; }
    std::array<double, size> next{};
    for (std::size_t index = 0; index < size; ++index)
    {
      const int signed_vdof = signed_vdofs[index];
      const int vdof = DecodeSignedVDof(signed_vdof);
      if (vdof < 0 || vdof >= local_values.Size()) { return false; }
      next[index] = static_cast<double>(SignedVDofSign(signed_vdof)) *
          input[index];
    }
    double* const output = local_values.HostWrite();
    if (output == nullptr && size != 0) { return false; }
    for (std::size_t index = 0; index < size; ++index)
    {
      output[DecodeSignedVDof(signed_vdofs[index])] = next[index];
    }
    return true;
  }
  catch (const std::exception&)
  {
    return false;
  }
}

} // namespace

struct BorrowedP1DGInterfaceLayout::Implementation
{
  mfem::ParMesh* mesh = nullptr;
  long mesh_sequence = -1;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> interior_space;
  trace::DGInterfaceTraceExchangePlan trace_exchange_plan;
  std::vector<std::vector<int>> trace_element_vdofs;
  std::vector<std::vector<int>> interior_element_vdofs;
  bool ready = false;

  bool IsUsable() const noexcept
  {
    return ready && mesh != nullptr && trace_collection != nullptr &&
        interior_collection != nullptr && trace_space != nullptr &&
        interior_space != nullptr && mesh->GetSequence() == mesh_sequence &&
        trace_space->GetParMesh() == mesh &&
        interior_space->GetParMesh() == mesh &&
        trace_space->FEColl() == trace_collection.get() &&
        interior_space->FEColl() == interior_collection.get() &&
        trace_space->GetVDim() == trace_components &&
        interior_space->GetVDim() == interior_components &&
        trace_space->GetOrdering() == mfem::Ordering::byNODES &&
        interior_space->GetOrdering() == mfem::Ordering::byNODES &&
        trace_space->GetSequence() == trace_exchange_plan.space_sequence &&
        trace_exchange_plan.IsConsistent() &&
        trace_element_vdofs.size() == interior_element_vdofs.size() &&
        trace_element_vdofs.size() == static_cast<std::size_t>(mesh->GetNE());
  }
};

BorrowedP1DGInterfaceLayout::BorrowedP1DGInterfaceLayout(
    mfem::ParMesh& borrowed_mesh) noexcept
{
  try
  {
    implementation_ = std::make_unique<Implementation>();
    Implementation& state = *implementation_;
    state.mesh = &borrowed_mesh;
    if (borrowed_mesh.Dimension() != dimension ||
        borrowed_mesh.SpaceDimension() != dimension || borrowed_mesh.GetNE() < 0)
    {
      return;
    }
    state.trace_collection = std::make_unique<mfem::DG_Interface_FECollection>(
        polynomial_order, dimension, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    state.interior_collection = std::make_unique<mfem::L2_FECollection>(
        polynomial_order, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    state.trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
        &borrowed_mesh, state.trace_collection.get(), trace_components,
        mfem::Ordering::byNODES);
    state.interior_space = std::make_unique<mfem::ParFiniteElementSpace>(
        &borrowed_mesh, state.interior_collection.get(), interior_components,
        mfem::Ordering::byNODES);
    state.mesh_sequence = borrowed_mesh.GetSequence();
    if (!trace::BuildDGInterfaceTraceExchangePlan(*state.trace_space,
                                                  state.trace_exchange_plan) ||
        !CaptureElementVDofs(*state.trace_space, trace_values_per_element,
                             state.trace_element_vdofs) ||
        !CaptureElementVDofs(*state.interior_space, interior_values_per_element,
                             state.interior_element_vdofs))
    {
      return;
    }
    state.ready = true;
    if (!state.IsUsable()) { state.ready = false; }
  }
  catch (const std::exception&)
  {
  }
}

BorrowedP1DGInterfaceLayout::~BorrowedP1DGInterfaceLayout() = default;

BorrowedP1DGInterfaceLayout::BorrowedP1DGInterfaceLayout(
    BorrowedP1DGInterfaceLayout&&) noexcept = default;

BorrowedP1DGInterfaceLayout& BorrowedP1DGInterfaceLayout::operator=(
    BorrowedP1DGInterfaceLayout&&) noexcept = default;

bool BorrowedP1DGInterfaceLayout::IsUsable() const noexcept
{
  return implementation_ != nullptr && implementation_->IsUsable();
}

const mfem::ParFiniteElementSpace* BorrowedP1DGInterfaceLayout::TraceSpace()
    const noexcept
{
  return IsUsable() ? implementation_->trace_space.get() : nullptr;
}

const mfem::ParFiniteElementSpace* BorrowedP1DGInterfaceLayout::InteriorSpace()
    const noexcept
{
  return IsUsable() ? implementation_->interior_space.get() : nullptr;
}

int BorrowedP1DGInterfaceLayout::TraceTrueVSize() const noexcept
{
  const mfem::ParFiniteElementSpace* const space = TraceSpace();
  return space == nullptr ? -1 : space->GetTrueVSize();
}

int BorrowedP1DGInterfaceLayout::TraceLocalVSize() const noexcept
{
  const mfem::ParFiniteElementSpace* const space = TraceSpace();
  return space == nullptr ? -1 : space->GetVSize();
}

int BorrowedP1DGInterfaceLayout::InteriorLocalVSize() const noexcept
{
  const mfem::ParFiniteElementSpace* const space = InteriorSpace();
  return space == nullptr ? -1 : space->GetVSize();
}

int BorrowedP1DGInterfaceLayout::LocalElementCount() const noexcept
{
  return IsUsable() ? implementation_->mesh->GetNE() : -1;
}

bool BorrowedP1DGInterfaceLayout::DistributeTrace(
    const mfem::Vector& owned_true_values,
    mfem::Vector& local_trace_values) const
{
  if (!IsUsable()) { return false; }
  return implementation_->trace_exchange_plan.Distribute(
      *implementation_->trace_space, owned_true_values, local_trace_values);
}

bool BorrowedP1DGInterfaceLayout::GatherTraceElement(
    const int element, const mfem::Vector& local_trace_values,
    mfem::Vector& element_values) const
{
  if (!IsUsable() || local_trace_values.Size() != TraceLocalVSize() ||
      element < 0 || element >= LocalElementCount())
  {
    return false;
  }
  return GatherSigned<trace_values_per_element>(
      implementation_->trace_element_vdofs[static_cast<std::size_t>(element)],
      local_trace_values, element_values);
}

bool BorrowedP1DGInterfaceLayout::GatherInteriorElement(
    const int element, const mfem::Vector& local_interior_values,
    mfem::Vector& element_values) const
{
  if (!IsUsable() || local_interior_values.Size() != InteriorLocalVSize() ||
      element < 0 || element >= LocalElementCount())
  {
    return false;
  }
  return GatherSigned<interior_values_per_element>(
      implementation_->interior_element_vdofs[
          static_cast<std::size_t>(element)],
      local_interior_values, element_values);
}

bool BorrowedP1DGInterfaceLayout::ScatterInteriorElement(
    const int element, const mfem::Vector& element_values,
    mfem::Vector& local_interior_values) const
{
  if (!IsUsable() || local_interior_values.Size() != InteriorLocalVSize() ||
      element < 0 || element >= LocalElementCount())
  {
    return false;
  }
  return ScatterSigned<interior_values_per_element>(
      implementation_->interior_element_vdofs[
          static_cast<std::size_t>(element)],
      element_values, local_interior_values);
}

} // namespace dg::hdg::runtime::cpu
