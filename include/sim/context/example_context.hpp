#pragma once

/**
 * @file sim/context/example_context.hpp
 * @brief Mesh + DG space context used by sim-layer examples.
 */

#include <memory>
#include <string>

#include "mfem.hpp"

#include "sim/config/case_config.hpp"
#include "sim/context/mesh_build.hpp"

namespace dg_igr_examples
{

/// @brief Aggregate DG spaces used by Euler examples.
struct DGSpaces
{
  mfem::DG_FECollection fec;
  mfem::ParFiniteElementSpace fes;
  mfem::ParFiniteElementSpace dfes;
  mfem::ParFiniteElementSpace vfes;
  std::unique_ptr<mfem::FiniteElementCollection> sigma_fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> sigma_fes;
  std::unique_ptr<mfem::H1_FECollection> h1_fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> h1_vfes;

  DGSpaces(mfem::ParMesh& pmesh,
           int order,
           int dim,
           int num_equations,
           int basis_type,
           const std::string& sigma_space,
           bool need_sigma,
           bool need_h1)
    : fec(order, dim, basis_type),
      fes(&pmesh, &fec),
      dfes(&pmesh, &fec, dim, mfem::Ordering::byNODES),
      vfes(&pmesh, &fec, num_equations, mfem::Ordering::byNODES)
  {
    if (need_sigma)
    {
      sigma_fec = BuildSigmaCollection(order, dim, sigma_space);
      sigma_fes = std::make_unique<mfem::ParFiniteElementSpace>(&pmesh, sigma_fec.get());
    }
    if (need_h1)
    {
      h1_fec = std::make_unique<mfem::H1_FECollection>(std::max(1, order), dim);
      h1_vfes = std::make_unique<mfem::ParFiniteElementSpace>(
          &pmesh, h1_fec.get(), dim, mfem::Ordering::byNODES);
    }
  }

private:
  static std::unique_ptr<mfem::FiniteElementCollection>
  BuildSigmaCollection(int order, int dim, const std::string& sigma_space)
  {
    const std::string lower = ToLowerCopy(sigma_space);
    if (lower == "h1")
    {
      return std::make_unique<mfem::H1_FECollection>(std::max(order, 3), dim);
    }
    return std::make_unique<mfem::DG_FECollection>(order, dim);
  }
};

/// @brief Bundles mesh, DG spaces, and basic dimension metadata for examples.
struct ExampleContext
{
  const RunConfig& run_cfg;
  std::unique_ptr<mfem::ParMesh> pmesh;
  int dim = 0;
  int num_equations = 0;
  DGSpaces spaces;

  static int BasisTypeForBackend(DGBackend backend)
  {
    return (backend == DGBackend::DGSEM)
               ? mfem::BasisType::GaussLobatto
               : mfem::BasisType::GaussLegendre;
  }

  ExampleContext(const RunConfig& cfg,
                 std::unique_ptr<mfem::ParMesh> pmesh_in,
                 const std::string& sigma_space,
                 bool need_sigma,
                 bool need_h1)
    : run_cfg(cfg),
      pmesh(std::move(pmesh_in)),
      dim(pmesh->Dimension()),
      num_equations(dim + 2),
      spaces(*pmesh, cfg.opts.order, dim, num_equations,
             BasisTypeForBackend(cfg.dg_backend),
             sigma_space, need_sigma, need_h1)
  {}

  mfem::ParMesh& Mesh() { return *pmesh; }
  const mfem::ParMesh& Mesh() const { return *pmesh; }
};

/// @brief Build mesh + DG spaces with standard serial/parallel refinement.
inline ExampleContext BuildExampleContext(const RunConfig& run_cfg,
                                          const std::string& sigma_space,
                                          bool need_sigma,
                                          bool need_h1)
{
  mfem::Mesh mesh = LoadSerialMesh(run_cfg.mesh_path, run_cfg.opts.ser_ref_levels);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  RefineParMesh(*pmesh, run_cfg.opts.par_ref_levels);

  return ExampleContext(run_cfg, std::move(pmesh), sigma_space, need_sigma, need_h1);
}

} // namespace dg_igr_examples
