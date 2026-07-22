#pragma once

/**
 * @file numerics/hdg/hdg_topology.hpp
 * @brief Model-neutral HDG topology metadata and non-owning execution views.
 *
 * The owning container is a host construction aid.  HdgTopologyView contains
 * only pointers, counts, and POD metadata so a backend can mirror the arrays
 * into memory managed by MFEM, RAJA/Umpire, or a raw accelerator adapter.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#ifndef DG_IGR_HDG_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define DG_IGR_HDG_HOST_DEVICE __host__ __device__
#else
#define DG_IGR_HDG_HOST_DEVICE
#endif
#endif

namespace dg
{
namespace hdg
{

using LocalIndex = std::int32_t;
using GlobalIndex = std::int64_t;
using RankIndex = std::int32_t;

inline constexpr LocalIndex invalid_local_index = -1;
inline constexpr GlobalIndex invalid_global_index = -1;
inline constexpr RankIndex invalid_rank = -1;

enum class ReferenceGeometry : std::uint8_t
{
  unknown,
  point,
  segment,
  triangle,
  quadrilateral,
  tetrahedron,
  hexahedron,
  prism,
  pyramid
};

enum class Ownership : std::uint8_t
{
  owned,
  ghost
};

/** Backend-defined orientation code and sign for one element/face incidence.
 * The code can carry an MFEM face permutation without baking MFEM types into
 * the execution view.  A sign of -1 records an oriented quantity reversal.
 */
struct OrientationMetadata
{
  LocalIndex permutation = 0;
  std::int8_t sign = 1;
};

struct BlockExtent
{
  LocalIndex rows = 0;
  LocalIndex columns = 0;
};

enum class LocalBlock : std::uint8_t
{
  interior_jacobian,
  interior_trace_jacobian,
  trace_interior_jacobian,
  trace_jacobian
};

/** Logical local sizes.  No physical storage order is implied. */
struct LocalBlockShape
{
  LocalIndex interior_dofs = 0;
  LocalIndex trace_dofs = 0;
  LocalIndex interior_components = 0;
  LocalIndex trace_components = 0;

  DG_IGR_HDG_HOST_DEVICE constexpr LocalIndex InteriorUnknowns() const noexcept
  {
    return static_cast<LocalIndex>(
        static_cast<std::int64_t>(interior_dofs) * interior_components);
  }

  DG_IGR_HDG_HOST_DEVICE constexpr LocalIndex TraceUnknowns() const noexcept
  {
    return static_cast<LocalIndex>(
        static_cast<std::int64_t>(trace_dofs) * trace_components);
  }

  DG_IGR_HDG_HOST_DEVICE constexpr BlockExtent Extent(
      const LocalBlock block) const noexcept
  {
    const LocalIndex n_interior = InteriorUnknowns();
    const LocalIndex n_trace = TraceUnknowns();
    switch (block)
    {
      case LocalBlock::interior_jacobian:
        return {n_interior, n_interior};
      case LocalBlock::interior_trace_jacobian:
        return {n_interior, n_trace};
      case LocalBlock::trace_interior_jacobian:
        return {n_trace, n_interior};
      case LocalBlock::trace_jacobian:
        return {n_trace, n_trace};
    }
    return {};
  }
};

/** Host construction must validate this before calling the unchecked execution
 * accessors or converting local extents to size_t. */
DG_IGR_HDG_HOST_DEVICE constexpr bool IsValidLocalBlockShape(
    const LocalBlockShape shape) noexcept
{
  const std::int64_t maximum = std::numeric_limits<LocalIndex>::max();
  return shape.interior_dofs >= 0 && shape.trace_dofs >= 0 &&
         shape.interior_components >= 0 && shape.trace_components >= 0 &&
         static_cast<std::int64_t>(shape.interior_dofs) *
                 shape.interior_components <=
             maximum &&
         static_cast<std::int64_t>(shape.trace_dofs) * shape.trace_components <=
             maximum;
}

constexpr LocalIndex ReferenceDimension(const ReferenceGeometry reference) noexcept
{
  switch (reference)
  {
    case ReferenceGeometry::point:
      return 0;
    case ReferenceGeometry::segment:
      return 1;
    case ReferenceGeometry::triangle:
    case ReferenceGeometry::quadrilateral:
      return 2;
    case ReferenceGeometry::tetrahedron:
    case ReferenceGeometry::hexahedron:
    case ReferenceGeometry::prism:
    case ReferenceGeometry::pyramid:
      return 3;
    case ReferenceGeometry::unknown:
      return invalid_local_index;
  }
  return invalid_local_index;
}

constexpr bool IsValidOwnership(const Ownership ownership) noexcept
{
  return ownership == Ownership::owned || ownership == Ownership::ghost;
}

constexpr bool IsValidOrientation(const OrientationMetadata orientation) noexcept
{
  return orientation.permutation >= 0 &&
         (orientation.sign == -1 || orientation.sign == 1);
}

struct GeometryMetadata
{
  GlobalIndex global_id = invalid_global_index;
  ReferenceGeometry reference = ReferenceGeometry::unknown;
  LocalIndex reference_dimension = 0;
  LocalIndex space_dimension = 0;
  LocalIndex mapping_order = 0;
  LocalIndex quadrature_points = 0;
};

struct ElementMetadata
{
  GlobalIndex global_id = invalid_global_index;
  LocalIndex geometry = invalid_local_index;
  LocalIndex order = 0;
  LocalIndex dimension = 0;
  LocalBlockShape blocks{};
};

struct ElementFaceMetadata
{
  LocalIndex face = invalid_local_index;
  LocalIndex local_face = invalid_local_index;
  OrientationMetadata orientation{};
};

struct FaceMetadata
{
  GlobalIndex global_id = invalid_global_index;
  LocalIndex geometry = invalid_local_index;
  LocalIndex adjacent_elements[2] = {invalid_local_index,
                                    invalid_local_index};
  LocalIndex local_faces[2] = {invalid_local_index, invalid_local_index};
  OrientationMetadata orientations[2]{};
  LocalIndex trace_offset = 0;
  LocalIndex trace_count = 0;
  RankIndex owner_rank = invalid_rank;
  Ownership ownership = Ownership::owned;
  LocalIndex order = 0;
  LocalIndex dimension = 0;
};

struct TraceDofMetadata
{
  GlobalIndex global_true_dof = invalid_global_index;
  LocalIndex local_dof = invalid_local_index;
  LocalIndex face = invalid_local_index;
  LocalIndex component = 0;
  RankIndex owner_rank = invalid_rank;
  Ownership ownership = Ownership::owned;
};

/** Trivially-copyable topology view suitable for capture in backend kernels.
 * Array residency is owned by the backend; this type never allocates or
 * transfers data.
 */
struct HdgTopologyView
{
  const ElementMetadata* elements = nullptr;
  const FaceMetadata* faces = nullptr;
  const TraceDofMetadata* trace_dofs = nullptr;
  const GeometryMetadata* geometries = nullptr;
  const LocalIndex* element_face_offsets = nullptr;
  const ElementFaceMetadata* element_faces = nullptr;
  LocalIndex element_count = 0;
  LocalIndex face_count = 0;
  LocalIndex trace_dof_count = 0;
  LocalIndex geometry_count = 0;
  LocalIndex element_face_count = 0;

  DG_IGR_HDG_HOST_DEVICE constexpr const ElementMetadata& Element(
      const LocalIndex i) const noexcept
  {
    return elements[i];
  }

  DG_IGR_HDG_HOST_DEVICE constexpr const FaceMetadata& Face(
      const LocalIndex i) const noexcept
  {
    return faces[i];
  }

  DG_IGR_HDG_HOST_DEVICE constexpr const TraceDofMetadata& TraceDof(
      const LocalIndex i) const noexcept
  {
    return trace_dofs[i];
  }

  DG_IGR_HDG_HOST_DEVICE constexpr const GeometryMetadata& Geometry(
      const LocalIndex i) const noexcept
  {
    return geometries[i];
  }

  DG_IGR_HDG_HOST_DEVICE constexpr LocalIndex ElementFaceBegin(
      const LocalIndex element) const noexcept
  {
    return element_face_offsets[element];
  }

  DG_IGR_HDG_HOST_DEVICE constexpr LocalIndex ElementFaceEnd(
      const LocalIndex element) const noexcept
  {
    return element_face_offsets[element + 1];
  }
};

/** Host-owned topology used during mesh/space construction. */
struct HdgTopologyStorage
{
  std::vector<ElementMetadata> elements;
  std::vector<FaceMetadata> faces;
  std::vector<TraceDofMetadata> trace_dofs;
  std::vector<GeometryMetadata> geometries;
  std::vector<LocalIndex> element_face_offsets;
  std::vector<ElementFaceMetadata> element_faces;

  HdgTopologyView View() const noexcept
  {
    return {elements.data(),
            faces.data(),
            trace_dofs.data(),
            geometries.data(),
            element_face_offsets.data(),
            element_faces.data(),
            static_cast<LocalIndex>(elements.size()),
            static_cast<LocalIndex>(faces.size()),
            static_cast<LocalIndex>(trace_dofs.size()),
            static_cast<LocalIndex>(geometries.size()),
            static_cast<LocalIndex>(element_faces.size())};
  }

  /** Structural validation only; no mesh or formulation semantics are added. */
  bool HasConsistentRanges() const noexcept
  {
    const auto fits_local_index = [](const std::size_t size) noexcept
    { return size <= static_cast<std::size_t>(std::numeric_limits<LocalIndex>::max()); };
    const auto valid_index = [](const LocalIndex index,
                                const std::size_t size) noexcept
    {
      return index >= 0 && static_cast<std::size_t>(index) < size;
    };

    if (!fits_local_index(elements.size()) || !fits_local_index(faces.size()) ||
        !fits_local_index(trace_dofs.size()) ||
        !fits_local_index(geometries.size()) ||
        !fits_local_index(element_faces.size()))
    {
      return false;
    }
    if (element_face_offsets.size() != elements.size() + 1 ||
        element_face_offsets.empty() || element_face_offsets.front() != 0 ||
        element_face_offsets.back() !=
            static_cast<LocalIndex>(element_faces.size()))
    {
      return false;
    }
    for (std::size_t i = 1; i < element_face_offsets.size(); ++i)
    {
      if (element_face_offsets[i] < element_face_offsets[i - 1])
      {
        return false;
      }
    }
    for (const auto& geometry : geometries)
    {
      if (geometry.global_id < 0 ||
          ReferenceDimension(geometry.reference) != geometry.reference_dimension ||
          geometry.space_dimension < geometry.reference_dimension ||
          geometry.mapping_order < 0 || geometry.quadrature_points < 0)
      {
        return false;
      }
    }
    for (const auto& element : elements)
    {
      if (element.global_id < 0 || !valid_index(element.geometry, geometries.size()) ||
          element.order < 0 || element.dimension < 0 ||
          element.dimension != geometries[static_cast<std::size_t>(element.geometry)]
                                   .reference_dimension ||
          !IsValidLocalBlockShape(element.blocks))
      {
        return false;
      }
    }
    for (std::size_t face_index = 0; face_index < faces.size(); ++face_index)
    {
      const auto& face = faces[face_index];
      if (face.global_id < 0 || !valid_index(face.geometry, geometries.size()) ||
          face.order < 0 || face.dimension < 0 ||
          face.dimension != geometries[static_cast<std::size_t>(face.geometry)]
                                .reference_dimension ||
          !IsValidOwnership(face.ownership) || face.owner_rank < 0 ||
          face.trace_offset < 0 || face.trace_count < 0 ||
          static_cast<std::size_t>(face.trace_offset) > trace_dofs.size() ||
          static_cast<std::size_t>(face.trace_count) >
              trace_dofs.size() - static_cast<std::size_t>(face.trace_offset))
      {
        return false;
      }

      int adjacent_count = 0;
      for (int side = 0; side != 2; ++side)
      {
        const LocalIndex element = face.adjacent_elements[side];
        if (element == invalid_local_index)
        {
          if (face.local_faces[side] != invalid_local_index ||
              face.orientations[side].permutation != 0 ||
              face.orientations[side].sign != 1)
          {
            return false;
          }
          continue;
        }
        if (!valid_index(element, elements.size()) || face.local_faces[side] < 0 ||
            !IsValidOrientation(face.orientations[side]))
        {
          return false;
        }
        ++adjacent_count;
      }
      if (adjacent_count == 0 ||
          (adjacent_count == 2 &&
           face.adjacent_elements[0] == face.adjacent_elements[1]))
      {
        return false;
      }

      const std::size_t begin = static_cast<std::size_t>(face.trace_offset);
      const std::size_t end = begin + static_cast<std::size_t>(face.trace_count);
      for (std::size_t dof = begin; dof < end; ++dof)
      {
        if (trace_dofs[dof].face != static_cast<LocalIndex>(face_index))
        {
          return false;
        }
      }
    }

    for (std::size_t element_index = 0; element_index < elements.size();
         ++element_index)
    {
      const std::size_t begin = static_cast<std::size_t>(element_face_offsets[element_index]);
      const std::size_t end =
          static_cast<std::size_t>(element_face_offsets[element_index + 1]);
      for (std::size_t entry_index = begin; entry_index < end; ++entry_index)
      {
        const auto& entry = element_faces[entry_index];
        if (!valid_index(entry.face, faces.size()) || entry.local_face < 0 ||
            !IsValidOrientation(entry.orientation))
        {
          return false;
        }
        const auto& face = faces[static_cast<std::size_t>(entry.face)];
        bool matches_face = false;
        for (int side = 0; side != 2; ++side)
        {
          matches_face = matches_face ||
                         (face.adjacent_elements[side] ==
                              static_cast<LocalIndex>(element_index) &&
                          face.local_faces[side] == entry.local_face &&
                          face.orientations[side].permutation ==
                              entry.orientation.permutation &&
                          face.orientations[side].sign == entry.orientation.sign);
        }
        if (!matches_face)
        {
          return false;
        }
      }
    }

    for (std::size_t face_index = 0; face_index < faces.size(); ++face_index)
    {
      const auto& face = faces[face_index];
      for (int side = 0; side != 2; ++side)
      {
        if (face.adjacent_elements[side] == invalid_local_index)
        {
          continue;
        }
        const std::size_t element =
            static_cast<std::size_t>(face.adjacent_elements[side]);
        const std::size_t begin =
            static_cast<std::size_t>(element_face_offsets[element]);
        const std::size_t end =
            static_cast<std::size_t>(element_face_offsets[element + 1]);
        bool found = false;
        for (std::size_t entry = begin; entry < end; ++entry)
        {
          const auto& incidence = element_faces[entry];
          found = found ||
                  (incidence.face == static_cast<LocalIndex>(face_index) &&
                   incidence.local_face == face.local_faces[side] &&
                   incidence.orientation.permutation ==
                       face.orientations[side].permutation &&
                   incidence.orientation.sign == face.orientations[side].sign);
        }
        if (!found)
        {
          return false;
        }
      }
    }

    for (std::size_t dof_index = 0; dof_index < trace_dofs.size(); ++dof_index)
    {
      const auto& dof = trace_dofs[dof_index];
      if (dof.global_true_dof < 0 ||
          dof.local_dof != static_cast<LocalIndex>(dof_index) ||
          !valid_index(dof.face, faces.size()) || dof.component < 0 ||
          dof.owner_rank < 0 || !IsValidOwnership(dof.ownership))
      {
        return false;
      }
      const auto& face = faces[static_cast<std::size_t>(dof.face)];
      const std::size_t begin = static_cast<std::size_t>(face.trace_offset);
      const std::size_t end = begin + static_cast<std::size_t>(face.trace_count);
      if (dof_index < begin || dof_index >= end)
      {
        return false;
      }
    }
    return true;
  }
};

static_assert(std::is_trivially_copyable<OrientationMetadata>::value,
              "HDG orientation metadata must be trivially copyable");
static_assert(std::is_trivially_copyable<LocalBlockShape>::value,
              "HDG block shapes must be trivially copyable");
static_assert(std::is_trivially_copyable<GeometryMetadata>::value,
              "HDG geometry metadata must be trivially copyable");
static_assert(std::is_trivially_copyable<ElementMetadata>::value,
              "HDG element metadata must be trivially copyable");
static_assert(std::is_trivially_copyable<ElementFaceMetadata>::value,
              "HDG element-face metadata must be trivially copyable");
static_assert(std::is_trivially_copyable<FaceMetadata>::value,
              "HDG face metadata must be trivially copyable");
static_assert(std::is_trivially_copyable<TraceDofMetadata>::value,
              "HDG trace metadata must be trivially copyable");
static_assert(std::is_trivially_copyable<HdgTopologyView>::value,
              "HDG topology execution views must be trivially copyable");
static_assert(std::is_standard_layout<HdgTopologyView>::value,
              "HDG topology execution views must have standard layout");

} // namespace hdg
} // namespace dg
