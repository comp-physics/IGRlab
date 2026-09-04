#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/igr/IGR.hpp"
#include "physics/igr/igr_coupling.hpp"
#include "physics/igr/igr_flux.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace
{

mfem::Mesh MakePeriodicMesh1D(int elements)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(elements, 1.0);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0})};
  return mfem::Mesh::MakePeriodic(
    mesh, mesh.CreatePeriodicVertexMapping(translations));
}

/** Test shell with communicator-scoped cumulative q_sigma telemetry. */
class SplitTelemetryDG final : public mfem::DGHyperbolicConservationLaws
{
public:
  SplitTelemetryDG(mfem::FiniteElementSpace& vfes,
                   std::unique_ptr<mfem::HyperbolicFormIntegrator> integrator,
                   MPI_Comm communicator)
    : mfem::DGHyperbolicConservationLaws(
        vfes, std::move(integrator), /*preassembleWeakDivergence=*/true),
      communicator_(communicator)
  {
  }

  void Mult(const mfem::Vector&, mfem::Vector& derivative) const override
  {
    derivative = 0.0;
    telemetry_.enabled = true;
    telemetry_.sample_count += 8;
    if (emit_failure_)
    {
      telemetry_.nonpositive_count += next_nonpositive_;
      telemetry_.nonfinite_count += next_nonfinite_;
      emit_failure_ = false;
    }
  }

  mfem::IGRFrozenSigmaTelemetry GetGlobalFrozenSigmaTelemetry() const override
  {
    mfem::IGRFrozenSigmaTelemetry global;
    int local_enabled = telemetry_.enabled ? 1 : 0;
    int global_enabled = 0;
    std::array<unsigned long long, 3> local_counts = {
      static_cast<unsigned long long>(telemetry_.sample_count),
      static_cast<unsigned long long>(telemetry_.nonpositive_count),
      static_cast<unsigned long long>(telemetry_.nonfinite_count)};
    std::array<unsigned long long, 3> global_counts = {};
    MPI_Allreduce(&local_enabled,
                  &global_enabled,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  communicator_);
    MPI_Allreduce(local_counts.data(),
                  global_counts.data(),
                  static_cast<int>(global_counts.size()),
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM,
                  communicator_);
    global.enabled = global_enabled != 0;
    global.sample_count = global_counts[0];
    global.nonpositive_count = global_counts[1];
    global.nonfinite_count = global_counts[2];
    return global;
  }

  void FailNextRHS(std::uint64_t nonpositive, std::uint64_t nonfinite)
  {
    emit_failure_ = true;
    next_nonpositive_ = nonpositive;
    next_nonfinite_ = nonfinite;
  }

private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
  mutable mfem::IGRFrozenSigmaTelemetry telemetry_;
  mutable bool emit_failure_ = false;
  mutable std::uint64_t next_nonpositive_ = 0;
  mutable std::uint64_t next_nonfinite_ = 0;
};

bool RunSplitAdmission()
{
  const int world_rank = mfem::Mpi::WorldRank();
  if (mfem::Mpi::WorldSize() != 4) { return false; }

  const int color = world_rank / 2;
  MPI_Comm communicator = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &communicator);
  int rank = -1;
  int size = 0;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);
  bool local_ok = size == 2;

  {
    constexpr int dim = 1;
    constexpr int equations = dim + 2;
    constexpr double gamma = 1.4;
    mfem::Mesh serial_mesh = MakePeriodicMesh1D(/*elements=*/4);
    mfem::ParMesh mesh(communicator, serial_mesh);
    mfem::DG_FECollection state_collection(
      /*order=*/1, dim, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace state_space(
      &mesh,
      &state_collection,
      equations,
      mfem::Ordering::byNODES);
    mfem::DG_FECollection sigma_collection(
      /*order=*/1, dim, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace sigma_space(
      &mesh, &sigma_collection, /*vdim=*/1, mfem::Ordering::byNODES);

    mfem::ParGridFunction state(&state_space);
    const int scalar_dofs = state_space.GetNDofs();
    const double velocity = 0.25;
    const double pressure = 1.0;
    const double energy =
      pressure / (gamma - 1.0) + 0.5 * velocity * velocity;
    double* values = state.HostWrite();
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      values[dof] = 1.0;
      values[scalar_dofs + dof] = velocity;
      values[2 * scalar_dofs + dof] = energy;
    }
    state.ExchangeFaceNbrData();

    dg_igr::IGRSolver igr(
      sigma_space, state_space, dim, equations, /*alpha=*/0.1);
    igr.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
    igr.SetIPKind(dg_igr::IGRIPKind::SIPG);
    igr.SetSolveMode(dg_igr::IGRSolveMode::Krylov);
    igr.SetLinearSolver(dg_igr::IGRLinearSolverType::CG);
    igr.SetPreconditioner(dg_igr::IGRPrecondType::None);
    igr.SetLinearSolveParams(
      /*rtol=*/1.0e-12, /*atol=*/0.0, /*max_it=*/400, /*print=*/0);
    igr.SetSigmaUpdatePolicy(
      dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep,
      /*interval=*/1);
    igr.SetStageRtolScale(1.0);

    mfem::EulerFlux euler_flux(dim, gamma);
    mfem::FluxWithIGR igr_flux(euler_flux);
    igr_flux.SetSigmaField(&igr.GetSigma());
    mfem::RusanovFluxWithSides numerical_flux(igr_flux);
    auto integrator = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, /*IntOrderOffset=*/2);
    SplitTelemetryDG op(state_space, std::move(integrator), communicator);
    op.SetNumericalFlux(&numerical_flux);
    mfem::EulerWithIGR wrapped(op, igr, state);

    constexpr std::uint64_t generation = 41;
    wrapped.PrepareSigmaForOutput(state,
                                  generation,
                                  /*next_rk_step=*/0,
                                  /*time=*/0.0,
                                  /*reusable_stage0=*/true);
    mfem::Vector accepted_sigma(igr.GetSigma());
    if (color == 0 && rank == 0)
    {
      op.FailNextRHS(/*nonpositive=*/3, /*nonfinite=*/2);
    }

    constexpr double original_dt = 0.01;
    constexpr double retry_dt = 0.5 * original_dt;
    wrapped.BeginRKStep(
      /*step=*/0, /*t=*/0.0, original_dt, /*stage_count=*/10);
    wrapped.SetRKStageState(
      /*stage=*/0, generation, /*stage_time=*/0.0);
    mfem::Vector derivative(state.Size());
    bool rejected = false;
    bool exact_reason = false;
    try
    {
      wrapped.Mult(state, derivative);
    }
    catch (const mfem::RecoverableRKStageAdmissionFailure& failure)
    {
      rejected = true;
      exact_reason =
        failure.Reason() == mfem::RKStageAdmissionFailureReason::
                              QSigmaNonpositiveAndNonfinite &&
        failure.GlobalNonpositiveQSigmaSampleCount() == 3 &&
        failure.GlobalNonfiniteQSigmaSampleCount() == 2 &&
        failure.Stage() == 0;
    }
    wrapped.AbortRKStep(state,
                        /*restored_time=*/0.0,
                        generation,
                        retry_dt);
    mfem::Vector rollback_error(igr.GetSigma());
    rollback_error -= accepted_sigma;

    wrapped.BeginRKStep(
      /*step=*/0, /*t=*/0.0, retry_dt, /*stage_count=*/10);
    wrapped.SetRKStageState(
      /*stage=*/0, generation, /*stage_time=*/0.0);
    bool retry_rejected = false;
    try
    {
      wrapped.Mult(state, derivative);
    }
    catch (const mfem::RecoverableRKStageAdmissionFailure&)
    {
      retry_rejected = true;
    }
    wrapped.AbortRKStep(state,
                        /*restored_time=*/0.0,
                        generation,
                        /*retry_dt=*/0.0);

    const bool expected_rejection = color == 0;
    local_ok = local_ok && rejected == expected_rejection &&
               exact_reason == expected_rejection && !retry_rejected &&
               rollback_error.Normlinf() == 0.0 &&
               wrapped.GetSigmaStageAdmissionRejectionCount() ==
                 (expected_rejection ? 1u : 0u) &&
               wrapped.GetSigmaStageAdmissionNonpositiveSampleCount() ==
                 (expected_rejection ? 3u : 0u) &&
               wrapped.GetSigmaStageAdmissionNonfiniteSampleCount() ==
                 (expected_rejection ? 2u : 0u);
  }

  MPI_Comm_free(&communicator);
  const int local_rejected = local_ok && color == 0 ? 1 : 0;
  std::array<int, 4> observed = {};
  MPI_Allgather(&local_rejected,
                1,
                MPI_INT,
                observed.data(),
                1,
                MPI_INT,
                MPI_COMM_WORLD);
  const std::array<int, 4> expected = {1, 1, 0, 0};
  const int local_pass = local_ok && observed == expected ? 1 : 0;
  int global_pass = 0;
  MPI_Allreduce(
    &local_pass, &global_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return global_pass == 1;
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const bool passed = RunSplitAdmission();
  if (mfem::Mpi::Root())
  {
    std::cout << "q-sigma-subcommunicator: "
              << (passed ? "PASS" : "FAIL") << '\n';
  }
  return passed ? 0 : 1;
}
