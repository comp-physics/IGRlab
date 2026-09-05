#pragma once
#include "mfem.hpp"
#include <memory>
#include <string>

namespace dg_igr
{
namespace igr_internal
{
// Retained first-operator local inverse. Current A and b still require an
// outer solve (FGMRES for RAS; CG is also valid for vertex-p1). Single-rank CUDA,
// uniform nodal L2 quads;
// face-hop radius 0..6 and at most 512 DOFs per local principal solve.
// Setup/factorization is FP64. fp32 uses rounded rows, rounded input scalars and
// FP32 accumulation in ordinary CUDA; the output vector remains FP64.
// kind=vertex-p1 uses a congruent GL-to-corner change of basis and solves the
// principal groups of discontinuous copies at each physical mesh vertex.
// Its composite GL rows use the same CUDA application; fp64 and radius=0 only.
std::unique_ptr<mfem::Solver> MakeScreenedPatchInversePreconditioner(
  const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes, int radius,
  const std::string& precision = "fp64", const std::string& kind = "ras");
}
}
