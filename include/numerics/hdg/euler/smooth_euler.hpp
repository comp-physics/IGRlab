#pragma once

/**
 * @file numerics/hdg/euler/smooth_euler.hpp
 * @brief Serial FP64, manufactured smooth-Euler HDG experiment.
 *
 * This is deliberately an unregistered experiment. It is neither a general
 * Euler discretization nor an MPI/device implementation. Its public value
 * types are fixed-size where physics has fixed size, so invalid physics never
 * requires a caller-owned raw output buffer.
 */

#include "numerics/hdg/core/host_reference.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace mfem
{
class SparseMatrix;
}

namespace dg::hdg::euler
{
inline constexpr int dimension = 2;
inline constexpr int conserved_components = 4;
inline constexpr double gamma = 1.4;
inline constexpr double fixed_scalar_tau = 2.5;
inline constexpr double backward_euler_dt = 0.05;

// Closed compile-time description: the experiment never chooses a model,
// stabilization, time rule, or backend per quadrature point.
struct SmoothEulerModelTag : ModelTagMarker {};
struct FixedScalarTauTag : StabilizationTagMarker {};
struct BackwardEulerTag : TimeTreatmentTagMarker {};
struct TraceStateName;
struct SerialExecutionTag;
struct HostMemoryTag;
struct NoCommunicationTag;
struct SmoothEulerDevicePolicy
{
  double tau = fixed_scalar_tau;
  double dt = backward_euler_dt;
};
using SmoothEulerTrace = TraceTuple<TracePotential<TraceStateName,
                                                   VectorTrace<conserved_components>>>;
using SmoothEulerSpec = KernelSpec<SmoothEulerModelTag,
                                  SmoothEulerTrace,
                                  FixedScalarTauTag,
                                  BackwardEulerTag,
                                  BackendPolicy<SerialExecutionTag,
                                                HostMemoryTag,
                                                NoCommunicationTag>,
                                  PrecisionPolicy<double, double, double, double, double>,
                                  SmoothEulerDevicePolicy>;
static_assert(IsKernelSpec<SmoothEulerSpec>::value,
              "smooth Euler must remain a closed static HDG KernelSpec");

using ConservedState = std::array<double, conserved_components>;
using PrimitiveState = std::array<double, conserved_components>; // rho,u,v,p
using FluxVector = std::array<double, conserved_components>;
using FluxJacobian =
    std::array<std::array<double, conserved_components>, conserved_components>;

enum class PhysicsStatus : unsigned char
{
  success,
  invalid_density,
  invalid_pressure,
  invalid_normal,
  non_finite
};

/** A value result makes failure atomic: value is untouched by failed helpers. */
struct PhysicsResult
{
  PhysicsStatus status = PhysicsStatus::success;
  bool ok() const noexcept { return status == PhysicsStatus::success; }
};

/** Conservative/primitive conversion with rho,p and all inputs checked. */
PhysicsResult ConservativeFromPrimitive(const PrimitiveState& primitive,
                                       ConservedState& conservative) noexcept;
PhysicsResult PrimitiveFromConservative(const ConservedState& conservative,
                                       PrimitiveState& primitive) noexcept;

/** F(U).n and its analytic 4x4 derivative. Failed calls preserve output. */
PhysicsResult NormalFlux(const ConservedState& conservative,
                         const std::array<double, dimension>& normal,
                         FluxVector& flux) noexcept;
PhysicsResult NormalFluxJacobian(const ConservedState& conservative,
                                 const std::array<double, dimension>& normal,
                                 FluxJacobian& jacobian) noexcept;

/** The frozen periodic manufactured state and its stationary forcing. */
PhysicsResult ManufacturedPrimitive(double x, double y,
                                   PrimitiveState& primitive) noexcept;
PhysicsResult ManufacturedConservative(double x, double y,
                                      ConservedState& conservative) noexcept;
PhysicsResult ManufacturedSource(double x, double y,
                                ConservedState& source) noexcept;
double ManufacturedMaximumWaveSpeed() noexcept;

struct NewtonEvidence
{
  int nonlinear_iterations = 0;
  int linear_iterations = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double trace_relative_residual = 0.0;
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
  double minimum_damping = 1.0;
  double maximum_wave_speed = 0.0;
  bool admissible = false;
};

struct ConvergenceEvidence
{
  int polynomial_order = 0;
  std::array<int, 3> mesh_widths{{3, 6, 12}};
  std::array<double, 3> volume_relative_l2{{0.0, 0.0, 0.0}};
  std::array<double, 3> trace_relative_l2{{0.0, 0.0, 0.0}};
  double volume_rate = 0.0;
  double trace_rate = 0.0;
  bool passed = false;
};

struct ExperimentEvidence
{
  NewtonEvidence newton{};
  std::vector<ConvergenceEvidence> convergence{};
  bool passed = false;
};

/**
 * Serial-only owner for the fixed 2-D experiment. The implementation uses
 * L2(p,2) interiors, DG_Interface(p,2) skeleton unknowns, Ordering::byNODES,
 * a SparseMatrix trace system, and local HostReferenceEngine condensation.
 */
class SmoothEulerExperiment
{
public:
  explicit SmoothEulerExperiment(int polynomial_order);
  ~SmoothEulerExperiment();
  SmoothEulerExperiment(SmoothEulerExperiment&&) noexcept;
  SmoothEulerExperiment& operator=(SmoothEulerExperiment&&) noexcept;
  SmoothEulerExperiment(const SmoothEulerExperiment&) = delete;
  SmoothEulerExperiment& operator=(const SmoothEulerExperiment&) = delete;

  /** Builds and atomically publishes a fresh verified periodic mesh/space graph. */
  bool BuildPeriodicMesh(int elements_per_direction);
  /** Stages then publishes the exact old interior state and single-valued trace. */
  bool ProjectOldStage();
  /** Assembles the current candidate volume/face/forcing local A/B/C/D data. */
  bool AssembleLinearization();
  /** Condenses actual local blocks through HostReferenceEngine and assembles S. */
  bool CondenseAndAssembleTrace();
  /** Solves the current nonsymmetric serial trace candidate. */
  bool SolveTrace();
  /** Recovers local increments through HostReferenceEngine. */
  bool RecoverInterior();
  /** Attempts the candidate Newton stage; unimplemented frozen gates fail closed. */
  bool SolveDampedNewton(NewtonEvidence& evidence);
  /** Runs the 3/6/12 ladder only when independent error gates are available. */
  bool RunConvergenceLadder(ConvergenceEvidence& evidence);

  int polynomial_order() const noexcept { return polynomial_order_; }
  int quadrature_order() const noexcept { return quadrature_order_; }
  const mfem::SparseMatrix* trace_matrix() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
  int polynomial_order_ = 0;
  int quadrature_order_ = 0;
};

/** Runs the p=1..4 serial manufactured candidate and returns all evidence. */
bool RunSmoothEulerExperiment(ExperimentEvidence& evidence);

} // namespace dg::hdg::euler
