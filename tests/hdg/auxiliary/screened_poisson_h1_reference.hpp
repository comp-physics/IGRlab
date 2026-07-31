#pragma once
/** Independent serial periodic H1 oracle for HDG-AUX-CPU-03 tests. */
#include <mfem.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
namespace dg::hdg::auxiliary::testing
{
using ScalarFunction = std::function<double(const mfem::Vector&)>;
struct H1ReferenceRequest
{
  int degree = 0;
  int elements_per_direction = 0;
  double alpha = 0.0;
  ScalarFunction rho;
  ScalarFunction source;
};
struct H1ReferenceEvidence
{
  int full_iterations = 0;
  int partial_iterations = 0;
  double full_residual_norm = 0.0;
  double full_right_hand_side_norm = 0.0;
  double partial_residual_norm = 0.0;
  double partial_right_hand_side_norm = 0.0;
  double sparse_symmetry_ratio = 0.0;
  double full_probe_rayleigh = 0.0;
  double partial_probe_rayleigh = 0.0;
  double full_partial_action_relative_difference = 0.0;
  double full_partial_solution_relative_difference = 0.0;
  double h1_quadrature_relative_change = 0.0;
  std::vector<double> deterministic_probe;
  std::vector<double> full_probe_action;
  std::vector<double> partial_probe_action;
  std::vector<double> full_true_solution_probe;
  std::vector<double> partial_true_solution_probe;
};
/** Move-only owner; every object is constructed on MPI_COMM_SELF. */
struct H1ReferenceSolution
{
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::H1_FECollection> collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> space;
  std::unique_ptr<mfem::ParGridFunction> full_solution;
  std::unique_ptr<mfem::ParGridFunction> partial_solution;
  std::unique_ptr<mfem::ParGridFunction> repeat_solution;
  H1ReferenceEvidence evidence;
  H1ReferenceSolution() = default;
  H1ReferenceSolution(H1ReferenceSolution&&) noexcept = default;
  H1ReferenceSolution& operator=(H1ReferenceSolution&&) noexcept = default;
  H1ReferenceSolution(const H1ReferenceSolution&) = delete;
  H1ReferenceSolution& operator=(const H1ReferenceSolution&) = delete;
};
/** Failed calls preserve output.  This function never calls production HDG. */
bool SolveH1Reference(const H1ReferenceRequest& request,
                      H1ReferenceSolution& output) noexcept;
} // namespace dg::hdg::auxiliary::testing
