#pragma once

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <memory>
#include <vector>

namespace dg_igr_hdg_reference_transfer_test
{

inline constexpr int n8_polynomial_order = 1;
inline constexpr int n8_projection_quadrature_order = 14;
inline constexpr int n8_observable_quadrature_order = 18;

struct EulerCommittedObservableOracle final
{
  bool available = false;
  int quadrature_order = 0;
  std::array<double, 4> conserved_integrals{};
  std::array<double, 4> conserved_l2_norms{};
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
};

/** Build the exact periodic N8 partition used by the frozen rank-one/rank-two
 * SC25 trajectory. The rank-two partition is an x-slab, not a replicated
 * serial mesh. */
bool MakeN8PeriodicMesh(MPI_Comm communicator,
                        std::unique_ptr<mfem::ParMesh>& output) noexcept;

/** Independently q14-project analytic smooth-transition U and Uhat on @p mesh.
 * Both arrays use the public generic byNODES payload layout. */
bool ProjectN8SmoothTransition(
    mfem::ParMesh& mesh, std::vector<double>& component_major_interior,
    std::vector<double>& true_trace) noexcept;

/** q18 MFEM-rule evaluation over the exported generic physical interior. */
bool EvaluateCommittedEulerObservables(
    const mfem::ParGridFunction& interior, int quadrature_order,
    EulerCommittedObservableOracle& output) noexcept;

/** q18 MFEM-rule evaluation over the public SC25 element-major U payload. */
bool EvaluateSc25ElementMajorObservables(
    const mfem::ParMesh& mesh, const std::vector<double>& element_major_u,
    int quadrature_order, EulerCommittedObservableOracle& output) noexcept;

/** Compare the generic physical field against public SC25 element-major U.
 * The maximum absolute and scaled relative component error are collective. */
bool CompareGenericFieldToSc25ElementMajor(
    const mfem::ParGridFunction& generic,
    const std::vector<double>& element_major_u,
    double& maximum_absolute_error, double& maximum_relative_error) noexcept;

} // namespace dg_igr_hdg_reference_transfer_test
