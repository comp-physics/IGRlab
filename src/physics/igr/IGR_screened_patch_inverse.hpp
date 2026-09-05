#pragma once
#include "mfem.hpp"
#include <memory>
#include <string>

namespace dg_igr
{
namespace igr_internal
{
// Retained first-operator restricted additive Schwarz inverse. Current A and b
// must still be solved by outer FGMRES. Rank-one CUDA, uniform nodal L2 quads;
// face-hop radius 0..6 and at most 512 DOFs per local principal solve.
// Setup/factorization is FP64. fp32 uses rounded rows, rounded input scalars and
// FP32 accumulation in ordinary CUDA; the output vector remains FP64.
std::unique_ptr<mfem::Solver> MakeScreenedPatchInversePreconditioner(
  const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes, int radius,
  const std::string& precision = "fp64");
}
}
