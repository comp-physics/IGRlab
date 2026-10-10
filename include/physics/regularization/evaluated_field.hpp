#pragma once

#include "igrlab/compat.hpp"

#include "physics/regularization/axes.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <type_traits>

namespace igrlab
{
namespace regularization
{

enum class EvaluationSiteKind : std::uint8_t
{
  Volume,
  Face,
  Wall
};

/** Metadata for one producer-owned evaluated point array.
 *
 * Point count is always explicit. It is never inferred from Vector::Size(),
 * GridFunction DOFs, or a finite-element true/local-DOF extent.
 */
struct EvaluatedPointFieldMetadata
{
  int point_count = 0;
  int component_extent = 0;
  int dimension = 0;
  EvaluationSiteKind site = EvaluationSiteKind::Volume;
  std::uint64_t point_set_id = 0u;
  int rk_stage = -1;
  std::uint64_t state_generation = 0u;
  std::uint64_t producer_generation = 0u;
};

/** Non-owning binding to an explicitly allocated and filled MFEM point array.
 *
 * The factory accepts exactly mfem::Vector, so GridFunction and
 * ParGridFunction objects cannot enter this boundary as point storage.
 */
class EvaluatedPointFieldBinding
{
public:
  EvaluatedPointFieldBinding() = default;

  template <typename VectorType,
            typename std::enable_if<
              std::is_same<typename std::remove_cv<VectorType>::type,
                           mfem::Vector>::value,
              int>::type = 0>
  static EvaluatedPointFieldBinding Bind(const VectorType& storage,
                                         EvaluatedPointFieldMetadata metadata)
  {
    return EvaluatedPointFieldBinding(&storage, metadata);
  }

  static EvaluatedPointFieldBinding
  Missing(EvaluatedPointFieldMetadata metadata)
  {
    return EvaluatedPointFieldBinding(nullptr, metadata);
  }

  const mfem::Vector* Storage() const { return storage_; }
  const EvaluatedPointFieldMetadata& Metadata() const { return metadata_; }
  int PointCount() const { return metadata_.point_count; }
  int ComponentExtent() const { return metadata_.component_extent; }
  int Dimension() const { return metadata_.dimension; }
  EvaluationSiteKind Site() const { return metadata_.site; }
  std::uint64_t PointSetIdentity() const { return metadata_.point_set_id; }
  int RKStage() const { return metadata_.rk_stage; }
  std::uint64_t StateGeneration() const { return metadata_.state_generation; }
  std::uint64_t ProducerGeneration() const
  {
    return metadata_.producer_generation;
  }

  const mfem::real_t* Read(bool on_device) const
  {
    return storage_->Read(on_device);
  }

private:
  EvaluatedPointFieldBinding(const mfem::Vector* storage,
                             EvaluatedPointFieldMetadata metadata)
    : storage_(storage),
      metadata_(metadata)
  {
  }

  const mfem::Vector* storage_ = nullptr;
  EvaluatedPointFieldMetadata metadata_{};
};

template <typename VectorType,
          typename std::enable_if<
            std::is_same<typename std::remove_cv<VectorType>::type,
                         mfem::Vector>::value,
            int>::type = 0>
EvaluatedPointFieldBinding
BindEvaluatedPointField(const VectorType& storage,
                        EvaluatedPointFieldMetadata metadata)
{
  return EvaluatedPointFieldBinding::Bind(storage, metadata);
}

struct EvaluatedPointBatchMetadata
{
  int point_count = 0;
  int dimension = 0;
  EvaluationSiteKind site = EvaluationSiteKind::Volume;
  std::uint64_t point_set_id = 0u;
  int rk_stage = -1;
  std::uint64_t state_generation = 0u;

  EvaluatedPointFieldMetadata state{};
  EvaluatedPointFieldMetadata conservative_state_gradient{};
  EvaluatedPointFieldMetadata auxiliary{};
  EvaluatedPointFieldMetadata trace{};
  EvaluatedPointFieldMetadata normal{};
  EvaluatedPointFieldMetadata base_characteristic_speed{};
};

/** Inputs for exactly one operation at exactly one evaluated point set. */
struct EvaluatedPointBatch
{
  int point_count = 0;
  int dimension = 0;
  EvaluationSiteKind site = EvaluationSiteKind::Volume;
  std::uint64_t point_set_id = 0u;
  int rk_stage = -1;
  std::uint64_t state_generation = 0u;

  EvaluatedPointFieldBinding state{};
  EvaluatedPointFieldBinding conservative_state_gradient{};
  EvaluatedPointFieldBinding auxiliary{};
  EvaluatedPointFieldBinding trace{};
  EvaluatedPointFieldBinding normal{};
  EvaluatedPointFieldBinding base_characteristic_speed{};

  EvaluatedPointBatchMetadata Metadata() const
  {
    return {point_count,
            dimension,
            site,
            point_set_id,
            rk_stage,
            state_generation,
            state.Metadata(),
            conservative_state_gradient.Metadata(),
            auxiliary.Metadata(),
            trace.Metadata(),
            normal.Metadata(),
            base_characteristic_speed.Metadata()};
  }
};

struct EvaluatedPointFieldRequirements
{
  bool state = false;
  bool conservative_state_gradient = false;
  bool auxiliary = false;
  bool trace = false;
  bool normal = false;
  bool base_characteristic_speed = false;
};

struct ModelPreflightResult
{
  bool valid = true;
  const char* message = "valid";
};

/** Metadata for one writable operation result at the selected point set. */
class EvaluatedPointOutputBinding
{
public:
  EvaluatedPointOutputBinding() = default;

  template <typename VectorType,
            typename std::enable_if<
              std::is_same<typename std::remove_cv<VectorType>::type,
                           mfem::Vector>::value,
              int>::type = 0>
  static EvaluatedPointOutputBinding Bind(VectorType& storage,
                                          int point_count,
                                          int component_extent,
                                          int dimension,
                                          EvaluationSiteKind site,
                                          std::uint64_t point_set_id)
  {
    return EvaluatedPointOutputBinding(&storage,
                                       point_count,
                                       component_extent,
                                       dimension,
                                       site,
                                       point_set_id);
  }

  static EvaluatedPointOutputBinding Missing(int point_count,
                                             int component_extent,
                                             int dimension,
                                             EvaluationSiteKind site,
                                             std::uint64_t point_set_id)
  {
    return EvaluatedPointOutputBinding(nullptr,
                                       point_count,
                                       component_extent,
                                       dimension,
                                       site,
                                       point_set_id);
  }

  mfem::Vector* Storage() const { return storage_; }
  int PointCount() const { return point_count_; }
  int ComponentExtent() const { return component_extent_; }
  int Dimension() const { return dimension_; }
  EvaluationSiteKind Site() const { return site_; }
  std::uint64_t PointSetIdentity() const { return point_set_id_; }

  mfem::real_t* Write(bool on_device) const
  {
    return storage_->Write(on_device);
  }
  mfem::real_t* ReadWrite(bool on_device) const
  {
    return storage_->ReadWrite(on_device);
  }

private:
  EvaluatedPointOutputBinding(mfem::Vector* storage,
                              int point_count,
                              int component_extent,
                              int dimension,
                              EvaluationSiteKind site,
                              std::uint64_t point_set_id)
    : storage_(storage),
      point_count_(point_count),
      component_extent_(component_extent),
      dimension_(dimension),
      site_(site),
      point_set_id_(point_set_id)
  {
  }

  mfem::Vector* storage_ = nullptr;
  int point_count_ = 0;
  int component_extent_ = 0;
  int dimension_ = 0;
  EvaluationSiteKind site_ = EvaluationSiteKind::Volume;
  std::uint64_t point_set_id_ = 0u;
};

template <typename VectorType,
          typename std::enable_if<
            std::is_same<typename std::remove_cv<VectorType>::type,
                         mfem::Vector>::value,
            int>::type = 0>
EvaluatedPointOutputBinding BindEvaluatedPointOutput(VectorType& storage,
                                                     int point_count,
                                                     int component_extent,
                                                     int dimension,
                                                     EvaluationSiteKind site,
                                                     std::uint64_t point_set_id)
{
  return EvaluatedPointOutputBinding::Bind(storage,
                                           point_count,
                                           component_extent,
                                           dimension,
                                           site,
                                           point_set_id);
}

struct RegularizationOperationOutputs
{
  EvaluatedPointOutputBinding auxiliary_coefficient{};
  EvaluatedPointOutputBinding auxiliary_source{};
  EvaluatedPointOutputBinding volume_correction{};
  EvaluatedPointOutputBinding normal_correction{};
  EvaluatedPointOutputBinding wall_flux{};
  EvaluatedPointOutputBinding characteristic_bounds{};
  EvaluatedPointOutputBinding diagnostics{};
};

static_assert(std::is_trivially_copyable<EvaluatedPointFieldMetadata>::value,
              "evaluated field metadata must be trivially copyable");
static_assert(std::is_trivially_copyable<EvaluatedPointFieldBinding>::value,
              "evaluated field binding must be trivially copyable");
static_assert(std::is_trivially_copyable<EvaluatedPointBatchMetadata>::value,
              "evaluated batch metadata must be trivially copyable");
static_assert(std::is_trivially_copyable<EvaluatedPointOutputBinding>::value,
              "evaluated output binding must be trivially copyable");
static_assert(!std::is_same<mfem::GridFunction, mfem::Vector>::value,
              "raw GridFunction DOFs must remain distinct from point vectors");
#ifdef MFEM_USE_MPI
static_assert(
  !std::is_same<mfem::ParGridFunction, mfem::Vector>::value,
  "raw ParGridFunction DOFs must remain distinct from point vectors");
#endif

} // namespace regularization
} // namespace igrlab
