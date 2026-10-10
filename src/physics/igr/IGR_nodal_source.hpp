#pragma once
#include "mfem.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <cstdlib>

namespace dg_igr { namespace igr_internal {
// Private implementation interface; retained data depend only on geometry and
// reference basis. Current primitive velocities are recomputed on every call.
enum class NodalSourceMode { Baseline, Volume, Central };
NodalSourceMode GetNodalSourceMode();
inline bool NodalSourceMirrorWalls()
{
  const char* raw = std::getenv("DG_IGR_IGR_NODAL_WALL_MIRROR");
  if (!raw || std::string(raw) == "0") return false;
  MFEM_VERIFY(std::string(raw) == "1",
              "DG_IGR_IGR_NODAL_WALL_MIRROR must be 0 or 1.");
  return true;
}
int NodalSourceQuadratureMinimumOrder();
int OperatorQuadratureMinimumOrder();
// Conservative bound for MFEM's generic tensor Values/PhysDerivatives kernels,
// queried for the active CPU/CUDA/HIP device. This is not a dense-source limit.
bool SupportsSourceQuadratureTensor(int dimension, int d1d, int q1d,
                                    std::string* reason = nullptr);
const char* NodalSourceModeName(NodalSourceMode mode);
struct NodalVelocitySourceCache
{
  mfem::Vector derivative, velocity, wall_derivative;
  mfem::Array<int> neighbors, wall_elements;
  int ndof = 0, nq = 0, ne = 0, slots = 0;
  std::uint64_t rebuilds = 0, hits = 0;
  const mfem::Mesh* mesh = nullptr;
  const mfem::FiniteElementSpace* fes = nullptr;
  const mfem::FiniteElementCollection* fec = nullptr;
  long mesh_sequence = -1, nodes_sequence = -1, fes_sequence = -1;
  int order = -1, rule_order = -1;
  bool central = false;
  bool mirror_walls = false;
  int wall_element_count = 0;
  std::vector<int> wall_marker;
  std::vector<double> rule;
};
// evec uses MFEM LEXICOGRAPHIC element restriction: [node,component,element].
// source uses the existing RHS quadrature order [point,element]. The central
// face correction uses an accurately integrated physical local mass, not the
// DGSEM collocated transport mass. A non-null slip marker selects the central
// mirror trace on those physical attributes, preserving tangential velocity.
void ApplyNodalVelocitySource2D(
  const mfem::FiniteElementSpace& fes, const mfem::QuadratureSpace& qs,
  const mfem::Vector& evec, mfem::real_t alpha,
  const mfem::Vector* element_alpha, mfem::real_t rho_floor, bool central,
  NodalVelocitySourceCache& cache, mfem::Vector& source,
  const mfem::Array<int>* slip_marker = nullptr);
} }
