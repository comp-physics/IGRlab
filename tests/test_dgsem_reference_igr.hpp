#pragma once

#include "igrlab/environment.hpp"

#include "numerics/dgsem/dgsem_reference.hpp"
#include "physics/euler/conversions.hpp"
#include "physics/igr/igr_flux.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace dg_igr_tests
{

inline void RunDGSEMScalarIGRScopeFixture(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;
  constexpr double sigma_value = 0.25;
  const double tol =
    mfem::dgsem::DGSEMReferenceTolerances::pointwise_fixture;

  const int world_size = mfem::Mpi::WorldSize();
  ctx.Check(world_size >= 1,
            "DGSEM scalar-IGR fixture has an active MPI communicator");
  const int global_elements = std::max(world_size, 1);
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    global_elements, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<int> partition(static_cast<std::size_t>(global_elements), 0);
  for (int element = 0; element < global_elements; ++element)
  {
    partition[static_cast<std::size_t>(element)] = element % world_size;
  }
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh, partition.data());
  const int local_elements = pmesh.GetNE();
  int min_local_elements = 0;
  MPI_Allreduce(&local_elements,
                &min_local_elements,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_MIN,
                MPI_COMM_WORLD);
  ctx.Check(min_local_elements > 0,
            "DGSEM scalar-IGR fixture gives every tested rank owned work");
  mfem::DG_FECollection sigma_fec(1, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace sigma_fes(
    &pmesh, &sigma_fec, 1, mfem::Ordering::byNODES);
  mfem::ParGridFunction sigma(&sigma_fes);
  mfem::ConstantCoefficient coefficient(sigma_value);
  sigma.ProjectCoefficient(coefficient);
  sigma.ExchangeFaceNbrData();

  mfem::Vector primitive(num_eq);
  primitive(0) = 2.0;
  primitive(1) = 0.3;
  primitive(2) = -0.2;
  primitive(3) = 1.1;
  mfem::Vector state(num_eq);
  mfem::EulerPrimToConservative(primitive, dim, gamma, state);

  mfem::EulerFlux euler_flux(dim, gamma);
  mfem::FluxWithIGR igr_flux(euler_flux);
  igr_flux.SetSigmaField(&sigma);
  mfem::ElementTransformation* tr = sigma_fes.GetElementTransformation(0);
  const mfem::IntegrationPoint& ip = sigma_fes.GetFE(0)->GetNodes().IntPoint(0);
  tr->SetIntPoint(&ip);
  mfem::DenseMatrix base_flux(num_eq, dim);
  mfem::DenseMatrix decorated_flux(num_eq, dim);
  euler_flux.ComputeFlux(state, *tr, base_flux);
  igr_flux.ComputeFlux(state, *tr, decorated_flux);
  decorated_flux -= base_flux;

  const double expected[num_eq][dim] = {{0.0, 0.0},
                                        {sigma_value, 0.0},
                                        {0.0, sigma_value},
                                        {0.075, -0.05}};
  double max_error = 0.0;
  for (int eq = 0; eq < num_eq; ++eq)
  {
    for (int direction = 0; direction < dim; ++direction)
    {
      max_error = std::max(
        max_error,
        std::abs(decorated_flux(eq, direction) - expected[eq][direction]));
    }
  }
  ctx.CheckNear(igr_flux.GetSigmaAtPoint(*tr),
                sigma_value,
                tol,
                "DGSEM frozen scalar-IGR auxiliary read");
  ctx.Check(max_error < tol,
            "DGSEM frozen scalar-IGR tensor-flux correction");
  ctx.Check(igr_flux.GetSigmaField() == &sigma,
            "DGSEM FluxWithIGR retains the owner-provided sigma field");

  // Behavioral provenance: changing the conservative stage state does not
  // mutate sigma. Only an explicit update through the owner-held field changes
  // the auxiliary value seen by FluxWithIGR.
  mfem::Vector primitive_next(num_eq);
  primitive_next(0) = 1.5;
  primitive_next(1) = -0.4;
  primitive_next(2) = 0.1;
  primitive_next(3) = 0.9;
  mfem::Vector state_next(num_eq);
  mfem::EulerPrimToConservative(
    primitive_next, dim, gamma, state_next);
  mfem::DenseMatrix base_next(num_eq, dim);
  mfem::DenseMatrix decorated_next(num_eq, dim);
  euler_flux.ComputeFlux(state_next, *tr, base_next);
  igr_flux.ComputeFlux(state_next, *tr, decorated_next);
  decorated_next -= base_next;
  const double expected_frozen[num_eq][dim] = {{0.0, 0.0},
                                               {sigma_value, 0.0},
                                               {0.0, sigma_value},
                                               {-0.1, 0.025}};
  double frozen_state_change_max_error = 0.0;
  for (int eq = 0; eq < num_eq; ++eq)
  {
    for (int direction = 0; direction < dim; ++direction)
    {
      frozen_state_change_max_error = std::max(
        frozen_state_change_max_error,
        std::abs(decorated_next(eq, direction) -
                 expected_frozen[eq][direction]));
    }
  }
  ctx.CheckNear(igr_flux.GetSigmaAtPoint(*tr),
                sigma_value,
                tol,
                "DGSEM stage-state mutation leaves frozen sigma unchanged");
  ctx.Check(frozen_state_change_max_error < tol,
            "DGSEM changed stage state still uses owner-frozen sigma");

  constexpr double owner_updated_sigma = 0.5;
  mfem::ConstantCoefficient owner_update(owner_updated_sigma);
  sigma.ProjectCoefficient(owner_update);
  sigma.ExchangeFaceNbrData();
  igr_flux.ComputeFlux(state_next, *tr, decorated_next);
  euler_flux.ComputeFlux(state_next, *tr, base_next);
  decorated_next -= base_next;
  const double expected_updated[num_eq][dim] = {{0.0, 0.0},
                                                {owner_updated_sigma, 0.0},
                                                {0.0, owner_updated_sigma},
                                                {-0.2, 0.05}};
  double owner_update_max_error = 0.0;
  for (int eq = 0; eq < num_eq; ++eq)
  {
    for (int direction = 0; direction < dim; ++direction)
    {
      owner_update_max_error = std::max(
        owner_update_max_error,
        std::abs(decorated_next(eq, direction) -
                 expected_updated[eq][direction]));
    }
  }
  ctx.CheckNear(igr_flux.GetSigmaAtPoint(*tr),
                owner_updated_sigma,
                tol,
                "DGSEM explicit sigma-owner update reaches FluxWithIGR");
  ctx.Check(owner_update_max_error < tol,
            "DGSEM flux changes only after explicit sigma-owner update");
  const std::string scope(mfem::dgsem::kDGSEMScalarIGRScope);
  ctx.Check(scope.find("neither repairs B-SCI-001") != std::string::npos,
            "DGSEM scalar-IGR fixture makes no B-SCI-001 repair claim");
  ctx.Check(scope.find("production wall stability") != std::string::npos,
            "DGSEM scalar-IGR fixture makes no production wall-stability claim");

  const char* output_dir = igrlab::GetEnv("DG_IGR_DGSEM_REFERENCE_OUTPUT_DIR");
  if (output_dir && mfem::Mpi::Root())
  {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    std::ofstream out(std::filesystem::path(output_dir) /
                      "scalar_igr_frozen_auxiliary.json");
    out << "{\n"
        << "  \"schema\": \"dgsem-scalar-igr-frozen-auxiliary-v3\",\n"
        << "  \"ranks\": " << world_size << ",\n"
        << "  \"global_elements\": " << global_elements << ",\n"
        << "  \"minimum_local_elements\": " << min_local_elements << ",\n"
        << "  \"empty_rank_safe_assertion_executed\": true,\n"
        << "  \"sigma\": " << sigma_value << ",\n"
        << "  \"primitive_state\": [2,0.3,-0.2,1.1],\n"
        << "  \"expected_flux_correction\": "
           "[[0,0],[0.25,0],[0,0.25],[0.075,-0.05]],\n"
        << "  \"max_error\": " << max_error << ",\n"
        << "  \"state_change_sigma\": " << sigma_value << ",\n"
        << "  \"state_change_max_error\": "
        << frozen_state_change_max_error << ",\n"
        << "  \"owner_updated_sigma\": " << owner_updated_sigma << ",\n"
        << "  \"owner_update_max_error\": " << owner_update_max_error
        << ",\n"
        << "  \"provenance\": \"conservative state mutation does not mutate "
           "sigma; only an explicit update through the owner-held field "
           "changes FluxWithIGR auxiliary reads\",\n"
        << "  \"scope\": \"" << mfem::dgsem::kDGSEMScalarIGRScope
        << "\"\n}\n";
    ctx.Check(!ec && out.good(),
              "DGSEM machine-readable scalar-IGR fixture written");
  }
}

} // namespace igrlab_tests
