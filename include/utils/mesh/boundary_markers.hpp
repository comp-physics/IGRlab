#pragma once

#include <initializer_list>
#include "mfem.hpp"

/**
 * @file utils/mesh/boundary_markers.hpp
 * @brief Boundary marker helpers for MFEM meshes.
 * @ingroup physics_bc
 */

namespace mfem
{

/**
 * @brief Build a boundary marker from a set of attribute ids {a1, a2, ...}.
 * @param[out] marker Boundary marker array (size = max bdr attribute).
 * @param[in]  pmesh  Parallel mesh providing boundary attributes.
 * @param[in]  attrs  Attribute ids to activate (1-based).
 *
 * Example: MakeMarker(pmesh, {4}) activates only attribute 4.
 */
static void MakeMarker(mfem::Array<int>& marker,
                       const mfem::ParMesh& pmesh,
                       std::initializer_list<int> attrs)
{
  const int nb = pmesh.bdr_attributes.Max();
  MFEM_VERIFY(nb > 0, "Mesh has no boundary attributes.");
  marker.SetSize(nb);
  marker = 0;
  for (int a : attrs)
  {
    MFEM_VERIFY(a >= 1 && a <= nb,
                "Requested boundary attribute " << a << " but mesh has [1.."
                                                << nb << "]");
    marker[a - 1] = 1;
  }
}

} // namespace mfem
