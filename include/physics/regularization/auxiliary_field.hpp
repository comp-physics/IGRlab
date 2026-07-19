#pragma once

#include "mfem.hpp"

#include <cstdint>
#include <type_traits>

namespace dg_igr
{
namespace regularization
{

enum class AuxiliaryFieldOwnership : std::uint8_t
{
  StateOwned,
  OperatorOwned,
  BorrowedReadOnly
};

enum class AuxiliaryFieldTopology : std::uint8_t
{
  Volume,
  Trace
};

enum class AuxiliaryValueKind : std::uint8_t
{
  Scalar,
  Vector,
  Tensor
};

enum class AuxiliaryCapability : std::uint32_t
{
  DeviceAccessible = 1u << 0,
  Restartable = 1u << 1,
  OutputVisible = 1u << 2,
  StageLocal = 1u << 3,
  TraceCoupled = 1u << 4
};

constexpr std::uint32_t CapabilityBit(AuxiliaryCapability capability)
{
  return static_cast<std::uint32_t>(capability);
}

struct AuxiliaryFieldCapabilities
{
  std::uint32_t bits = 0u;

  constexpr bool Has(AuxiliaryCapability capability) const
  {
    return (bits & CapabilityBit(capability)) != 0u;
  }
};

/** Stable semantic, restart, and output identities. These are numeric by
 * design so identity checks do not depend on pointer or string lifetime. */
struct AuxiliaryFieldIdentity
{
  std::uint64_t semantic_id = 0u;
  std::uint64_t restart_id = 0u;
  std::uint64_t output_id = 0u;
};

constexpr bool operator==(AuxiliaryFieldIdentity lhs,
                          AuxiliaryFieldIdentity rhs)
{
  return lhs.semantic_id == rhs.semantic_id &&
         lhs.restart_id == rhs.restart_id && lhs.output_id == rhs.output_id;
}

constexpr bool operator!=(AuxiliaryFieldIdentity lhs,
                          AuxiliaryFieldIdentity rhs)
{
  return !(lhs == rhs);
}

/** Provenance attached to an auxiliary field at an RK-stage boundary. */
struct AuxiliaryFieldProvenance
{
  int rk_stage = -1;
  std::uint64_t state_generation = 0u;
  std::uint64_t producer_generation = 0u;
};

constexpr bool SameStageState(AuxiliaryFieldProvenance lhs,
                              AuxiliaryFieldProvenance rhs)
{
  return lhs.rk_stage == rhs.rk_stage &&
         lhs.state_generation == rhs.state_generation;
}

struct ScalarValueTag
{
  static constexpr AuxiliaryValueKind kind = AuxiliaryValueKind::Scalar;
  static constexpr int Components(int) { return 1; }
};

struct VectorValueTag
{
  static constexpr AuxiliaryValueKind kind = AuxiliaryValueKind::Vector;
  static constexpr int Components(int dimension) { return dimension; }
};

struct TensorValueTag
{
  static constexpr AuxiliaryValueKind kind = AuxiliaryValueKind::Tensor;
  static constexpr int Components(int dimension)
  {
    return dimension * dimension;
  }
};

struct VolumeTopologyTag
{
  static constexpr AuxiliaryFieldTopology topology =
    AuxiliaryFieldTopology::Volume;
};

struct TraceTopologyTag
{
  static constexpr AuxiliaryFieldTopology topology =
    AuxiliaryFieldTopology::Trace;
};

/** Typed host descriptor for MFEM-managed auxiliary storage.
 *
 * The descriptor does not allocate or delete storage. Ownership records which
 * subsystem owns the field, while MFEM Memory remains the sole data handle.
 */
template <typename ValueTag, typename TopologyTag>
class AuxiliaryFieldDescriptor
{
public:
  using value_tag = ValueTag;
  using topology_tag = TopologyTag;

  AuxiliaryFieldDescriptor(mfem::Vector& storage,
                           int entity_count,
                           int dimension,
                           AuxiliaryFieldOwnership ownership,
                           AuxiliaryFieldProvenance provenance,
                           AuxiliaryFieldIdentity identity,
                           AuxiliaryFieldCapabilities capabilities)
    : storage_(&storage),
      entity_count_(entity_count),
      dimension_(dimension),
      ownership_(ownership),
      provenance_(provenance),
      identity_(identity),
      capabilities_(capabilities)
  {
  }

  static constexpr AuxiliaryValueKind ValueKind() { return ValueTag::kind; }
  static constexpr AuxiliaryFieldTopology Topology()
  {
    return TopologyTag::topology;
  }

  int EntityCount() const { return entity_count_; }
  int Dimension() const { return dimension_; }
  int Components() const { return ValueTag::Components(dimension_); }
  int ExpectedSize() const { return entity_count_ * Components(); }
  bool HasExpectedSize() const
  {
    return storage_ != nullptr && storage_->Size() == ExpectedSize();
  }

  AuxiliaryFieldOwnership Ownership() const { return ownership_; }
  AuxiliaryFieldProvenance Provenance() const { return provenance_; }
  AuxiliaryFieldIdentity Identity() const { return identity_; }
  AuxiliaryFieldCapabilities Capabilities() const { return capabilities_; }

  const mfem::Vector& Storage() const { return *storage_; }
  mfem::Vector& Storage() { return *storage_; }

  const mfem::Memory<mfem::real_t>& Memory() const
  {
    return storage_->GetMemory();
  }
  mfem::Memory<mfem::real_t>& Memory() { return storage_->GetMemory(); }

  const mfem::real_t* Read(bool on_device) const
  {
    return mfem::Read(Memory(), ExpectedSize(), on_device);
  }
  mfem::real_t* Write(bool on_device)
  {
    return mfem::Write(Memory(), ExpectedSize(), on_device);
  }
  mfem::real_t* ReadWrite(bool on_device)
  {
    return mfem::ReadWrite(Memory(), ExpectedSize(), on_device);
  }

private:
  mfem::Vector* storage_ = nullptr;
  int entity_count_ = 0;
  int dimension_ = 0;
  AuxiliaryFieldOwnership ownership_ = AuxiliaryFieldOwnership::BorrowedReadOnly;
  AuxiliaryFieldProvenance provenance_{};
  AuxiliaryFieldIdentity identity_{};
  AuxiliaryFieldCapabilities capabilities_{};
};

using ScalarAuxiliaryFieldDescriptor =
  AuxiliaryFieldDescriptor<ScalarValueTag, VolumeTopologyTag>;
using VectorAuxiliaryFieldDescriptor =
  AuxiliaryFieldDescriptor<VectorValueTag, VolumeTopologyTag>;
using TensorAuxiliaryFieldDescriptor =
  AuxiliaryFieldDescriptor<TensorValueTag, VolumeTopologyTag>;
using TraceAuxiliaryFieldDescriptor =
  AuxiliaryFieldDescriptor<ScalarValueTag, TraceTopologyTag>;

static_assert(std::is_same<ScalarAuxiliaryFieldDescriptor::value_tag,
                           ScalarValueTag>::value,
              "scalar auxiliary descriptor must retain its value type");
static_assert(std::is_same<VectorAuxiliaryFieldDescriptor::value_tag,
                           VectorValueTag>::value,
              "vector auxiliary descriptor must retain its value type");
static_assert(std::is_same<TensorAuxiliaryFieldDescriptor::value_tag,
                           TensorValueTag>::value,
              "tensor auxiliary descriptor must retain its value type");
static_assert(TraceAuxiliaryFieldDescriptor::Topology() ==
                AuxiliaryFieldTopology::Trace,
              "trace auxiliary descriptor must retain trace topology");

} // namespace regularization
} // namespace dg_igr
