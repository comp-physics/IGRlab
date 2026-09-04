#pragma once

/**
 * @file sim/context/example_context.hpp
 * @brief Mesh + DG space context used by sim-layer examples.
 */

#include <memory>
#include <string>

#include "mfem.hpp"

#include "sim/config/case_config.hpp"
#include "sim/context/case_context.hpp"

namespace dg_sim
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
      sigma_fes =
        std::make_unique<mfem::ParFiniteElementSpace>(&pmesh, sigma_fec.get());
    }
    if (need_h1)
    {
      h1_fec = std::make_unique<mfem::H1_FECollection>(std::max(1, order), dim);
      h1_vfes =
        std::make_unique<mfem::ParFiniteElementSpace>(&pmesh,
                                                      h1_fec.get(),
                                                      dim,
                                                      mfem::Ordering::byNODES);
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
    if (lower == "l2" || lower == "dg" || lower == "brokendg" ||
        lower == "brokenl2")
    {
      return std::make_unique<mfem::DG_FECollection>(order, dim);
    }
    MFEM_VERIFY(false,
                "Invalid IGR sigma space '" << sigma_space
                                            << "'. Use 'H1' or 'L2'.");
    return std::make_unique<mfem::DG_FECollection>(order, dim);
  }
};

/// @brief Bundles mesh, DG spaces, and basic dimension metadata for examples.
struct ExampleContext
{
  const RunConfig& run_cfg;
  CaseContext case_context;
  int dim = 0;
  int num_equations = 0;
  DGSpaces spaces;

  static int BasisTypeForBackend(DGBackend backend)
  {
    return (backend == DGBackend::DGSEM) ? mfem::BasisType::GaussLobatto
                                         : mfem::BasisType::GaussLegendre;
  }

  ExampleContext(const RunConfig& cfg,
                 CaseContext case_context_in,
                 const std::string& sigma_space,
                 bool need_sigma,
                 bool need_h1)
    : run_cfg(cfg),
      case_context(std::move(case_context_in)),
      dim(case_context.Mesh().Dimension()),
      num_equations(dim + 2),
      spaces(case_context.Mesh(),
             cfg.opts.order,
             dim,
             num_equations,
             BasisTypeForBackend(cfg.dg_backend),
             sigma_space,
             need_sigma,
             need_h1)
  {
  }

  /// @brief Compatibility adapter for callers that already own a ParMesh.
  ///
  /// New case construction should use BuildCaseContext so its mesh provenance
  /// and refinement sequence are retained.  This overload preserves existing
  /// DG test and utility callers without introducing a partition policy.
  ExampleContext(const RunConfig& cfg,
                 std::unique_ptr<mfem::ParMesh> pmesh_in,
                 const std::string& sigma_space,
                 bool need_sigma,
                 bool need_h1)
    : ExampleContext(cfg,
                     WrapExistingMesh(std::move(pmesh_in)),
                     sigma_space,
                     need_sigma,
                     need_h1)
  {
  }

  mfem::ParMesh& Mesh() { return case_context.Mesh(); }
  const mfem::ParMesh& Mesh() const { return case_context.Mesh(); }
  MPI_Comm Comm() const { return case_context.Comm(); }
  const MeshBuildMetadata& MeshMetadata() const
  {
    return case_context.MeshMetadata();
  }

private:
  static CaseContext WrapExistingMesh(std::unique_ptr<mfem::ParMesh> mesh)
  {
    MFEM_VERIFY(mesh, "ExampleContext requires a parallel mesh.");
    MeshBuildMetadata metadata;
    metadata.mesh_path_precedence = MeshPathPrecedence::AlreadyResolved;
    metadata.local_elements_after_partitioning = mesh->GetNE();
    metadata.local_elements_after_parallel_refinement = mesh->GetNE();
    return CaseContext(std::move(mesh), std::move(metadata));
  }
};

/// @brief Build mesh + DG spaces with standard serial/parallel refinement.
inline ExampleContext BuildExampleContext(const RunConfig& run_cfg,
                                          const std::string& sigma_space,
                                          bool need_sigma,
                                          bool need_h1)
{
  CaseContext case_context = BuildCaseContext(run_cfg);

  return ExampleContext(run_cfg,
                        std::move(case_context),
                        sigma_space,
                        need_sigma,
                        need_h1);
}

} // namespace dg_sim
