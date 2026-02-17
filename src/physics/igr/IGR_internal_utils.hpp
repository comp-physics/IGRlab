#pragma once

#include "utils/config/env_flags.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace dg_igr
{
namespace igr_internal
{

inline bool IsDGSpace(const mfem::FiniteElementSpace& fes)
{
  // Identify DG/L2 spaces by their FECollection type.
  const mfem::FiniteElementCollection* fec = fes.FEColl();
  return dynamic_cast<const mfem::L2_FECollection*>(fec) != nullptr ||
         dynamic_cast<const mfem::DG_FECollection*>(fec) != nullptr;
}

inline int SpaceOrder(const mfem::FiniteElementSpace& fes)
{
  MFEM_VERIFY(fes.GetNE() > 0, "Empty FES.");
  return fes.GetFE(0)->GetOrder();
}

inline mfem::IntegrationRules& FaceIntegrationRules_()
{
  static mfem::IntegrationRules face_irs(0, mfem::Quadrature1D::GaussLobatto);
  return face_irs;
}

inline double IPDGPenalty(const mfem::FiniteElementSpace& fes,
                          double Cbase = 4.0)
{
  const int p = SpaceOrder(fes);
  return Cbase * (p + 1) * (p + 1);
}

inline bool IsSPD_DefaultAssumption(const mfem::FiniteElementSpace& sfes,
                                    IGRSolver::IPKind ip_kind)
{
  // Structural policy:
  //  - H1 operator is assembled symmetrically -> SPD-friendly.
  //  - DG/L2 with SIPG is SPD-by-construction when DGDiffusionIntegrator
  //    is wired as (sigma=-1, kappa>0 penalty). IIPG/NIPG are non-SPD.
  if (IsDGSpace(sfes))
  {
    return ip_kind == IGRSolver::IPKind::SIPG;
  }
  return true; // H1: mass + diffusion -> SPD
}

inline const char* LinearSolverTypeName(IGRSolver::LinearSolverType t)
{
  switch (t)
  {
  case IGRSolver::LinearSolverType::Auto:
    return "auto";
  case IGRSolver::LinearSolverType::CG:
    return "cg";
  case IGRSolver::LinearSolverType::GMRES:
    return "gmres";
  case IGRSolver::LinearSolverType::FGMRES:
    return "fgmres";
  case IGRSolver::LinearSolverType::BiCGSTAB:
    return "bicgstab";
  case IGRSolver::LinearSolverType::MINRES:
    return "minres";
  }
  return "auto";
}

inline const char* PrecondTypeName(IGRSolver::PrecondType t)
{
  switch (t)
  {
  case IGRSolver::PrecondType::Auto:
    return "auto";
  case IGRSolver::PrecondType::BoomerAMG:
    return "boomeramg";
  case IGRSolver::PrecondType::HypreILU:
    return "hypreilu";
  case IGRSolver::PrecondType::HypreEuclid:
    return "hypreeuclid";
  case IGRSolver::PrecondType::ParaSails:
    return "parasails";
  case IGRSolver::PrecondType::HypreSmoother:
    return "hypresmoother";
  case IGRSolver::PrecondType::None:
    return "none";
  }
  return "auto";
}

inline bool RequiresSPD(IGRSolver::LinearSolverType solver)
{
  return solver == IGRSolver::LinearSolverType::CG ||
         solver == IGRSolver::LinearSolverType::MINRES;
}

inline bool IsSymmetricPC(IGRSolver::PrecondType pc)
{
  return pc == IGRSolver::PrecondType::BoomerAMG ||
         pc == IGRSolver::PrecondType::HypreSmoother ||
         pc == IGRSolver::PrecondType::None;
}

inline bool IsCudaBackendActive()
{
  return mfem::Device::Allows(mfem::Backend::CUDA_MASK);
}

inline bool PreconditionerRequiresAssembledMatrix(IGRSolver::PrecondType pc)
{
  return pc == IGRSolver::PrecondType::BoomerAMG ||
         pc == IGRSolver::PrecondType::HypreILU ||
         pc == IGRSolver::PrecondType::HypreEuclid ||
         pc == IGRSolver::PrecondType::ParaSails;
}

inline bool StrictFiniteScanEnabled()
{
  return ::dg_igr::EnvFlagEnabled("DG_IGR_STRICT_FINITE_SCAN");
}

inline bool VectorIsFiniteHostScan(const mfem::Vector& v)
{
  const int n = v.Size();
  for (int i = 0; i < n; ++i)
  {
    if (!std::isfinite(v(i)))
    {
      return false;
    }
  }
  return true;
}

inline bool VectorIsFinite(const mfem::Vector& v)
{
  if (v.Size() == 0)
  {
    return true;
  }

  // On device backends avoid elementwise host access (`v(i)`), which can
  // trigger full host staging. A device reduction keeps data on backend.
  if (mfem::Device::Allows(mfem::Backend::DEVICE_MASK) &&
      !StrictFiniteScanEnabled())
  {
    const double l2 = v.Norml2();
    return std::isfinite(l2);
  }

  return VectorIsFiniteHostScan(v);
}

inline void LogIGRFallback(const char* reason,
                           IGRSolver::LinearSolverType requested_solver,
                           IGRSolver::LinearSolverType selected_solver,
                           IGRSolver::PrecondType requested_prec,
                           IGRSolver::PrecondType selected_prec)
{
  if (!mfem::Mpi::Root())
  {
    return;
  }
  std::cout << "[igr-linear] event=fallback" << " reason=\"" << reason << "\""
            << " requested_solver=" << LinearSolverTypeName(requested_solver)
            << " selected_solver=" << LinearSolverTypeName(selected_solver)
            << " requested_precond=" << PrecondTypeName(requested_prec)
            << " selected_precond=" << PrecondTypeName(selected_prec) << "\n";
}

inline void LogIGRSelection(bool warm_start,
                            bool is_dg,
                            bool spd_ok,
                            bool using_cuda,
                            IGRSolver::LinearSolverType solver_choice,
                            IGRSolver::PrecondType prec_choice,
                            double lin_rtol,
                            double lin_atol,
                            int lin_maxit)
{
  if (!mfem::Mpi::Root())
  {
    return;
  }
  std::cout << "[igr-linear] event=select"
            << " warm_start=" << (warm_start ? 1 : 0)
            << " space=" << (is_dg ? "dg" : "h1")
            << " spd_assumed=" << (spd_ok ? 1 : 0)
            << " device=" << (using_cuda ? "cuda" : "cpu")
            << " solver=" << LinearSolverTypeName(solver_choice)
            << " precond=" << PrecondTypeName(prec_choice)
            << " rtol=" << lin_rtol << " atol=" << lin_atol
            << " maxit=" << lin_maxit << "\n";
}

} // namespace igr_internal
} // namespace dg_igr
