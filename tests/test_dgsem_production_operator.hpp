#pragma once

#include "igrlab/environment.hpp"

#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_operator.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/igr/igr_flux.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

namespace dg_igr_tests
{
namespace dgsem_production_test
{

inline int AbsoluteDof(int vdof)
{
  return vdof >= 0 ? vdof : -1 - vdof;
}

class ScopedEnvironment
{
public:
  ScopedEnvironment(const char* name, const char* value)
    : name_(name)
  {
    if (const char* previous = ::igrlab::GetEnv(name))
    {
      had_value_ = true;
      previous_ = previous;
    }
    if (value)
    {
      setenv(name, value, 1);
    }
    else
    {
      unsetenv(name);
    }
  }

  ~ScopedEnvironment()
  {
    if (had_value_)
    {
      setenv(name_.c_str(), previous_.c_str(), 1);
    }
    else
    {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
  std::string name_;
  std::string previous_;
  bool had_value_ = false;
};

class UnsupportedBoundaryState final : public mfem::BoundaryState
{
public:
  void Eval(const mfem::Vector& interior,
            const mfem::Vector&,
            mfem::FaceElementTransformations&,
            mfem::Vector& exterior) const override
  {
    exterior = interior;
  }
};

inline bool IsSentinel(const mfem::Vector& values, double sentinel)
{
  const double* data = values.HostRead();
  for (int i = 0; i < values.Size(); ++i)
  {
    if (data[i] != sentinel)
    {
      return false;
    }
  }
  return true;
}

inline bool EulerStateAdmissible(
  int dimension,
  double gamma,
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::Vector& local_state)
{
  const double* state = local_state.HostRead();
  mfem::Array<int> vdofs;
  for (int element = 0; element < vfes.GetNE(); ++element)
  {
    const int dof = vfes.GetFE(element)->GetDof();
    vfes.GetElementVDofs(element, vdofs);
    for (int local = 0; local < dof; ++local)
    {
      const double density =
        state[AbsoluteDof(vdofs[local])];
      if (!std::isfinite(density) || !(density > 0.0))
      {
        return false;
      }
      double momentum_squared = 0.0;
      for (int d = 0; d < dimension; ++d)
      {
        const double momentum =
          state[AbsoluteDof(vdofs[local + dof * (1 + d)])];
        momentum_squared += momentum * momentum;
      }
      const double energy = state[
        AbsoluteDof(vdofs[local + dof * (dimension + 1)])];
      const double pressure =
        (gamma - 1.0) *
        (energy - 0.5 * momentum_squared / density);
      if (!std::isfinite(pressure) || !(pressure > 0.0))
      {
        return false;
      }
    }
  }
  return true;
}

inline std::array<double, 2> ConservedMassEnergy(
  int dimension,
  int order,
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::Vector& local_state)
{
  const mfem::Vector& weights =
    mfem::dgsem::GetLGLBasis1D(order).weights;
  double local_mass = 0.0;
  double local_energy = 0.0;
  const double* state = local_state.HostRead();
  mfem::Array<int> vdofs;
  for (int element = 0; element < vfes.GetNE(); ++element)
  {
    const mfem::FiniteElement& fe = *vfes.GetFE(element);
    const int dof = fe.GetDof();
    const auto& ijk = mfem::dgsem::GetTensorDofToIJK(fe);
    mfem::ElementTransformation* transformation =
      vfes.GetElementTransformation(element);
    vfes.GetElementVDofs(element, vdofs);
    for (int local = 0; local < dof; ++local)
    {
      const mfem::IntegrationPoint& point = fe.GetNodes().IntPoint(local);
      transformation->SetIntPoint(&point);
      double quadrature_weight =
        weights(ijk[static_cast<std::size_t>(local)][0]) *
        transformation->Weight();
      if (dimension == 2)
      {
        quadrature_weight *=
          weights(ijk[static_cast<std::size_t>(local)][1]);
      }
      local_mass += quadrature_weight *
                    state[AbsoluteDof(vdofs[local])];
      local_energy += quadrature_weight * state[
        AbsoluteDof(vdofs[local + dof * (dimension + 1)])];
    }
  }
  return {{GlobalSum(local_mass), GlobalSum(local_energy)}};
}

inline void CheckMetadataNegativeGates(TestContext& ctx,
                                       int& negative_checks)
{
  using mfem::dgsem::DGSEMDeviceFaceOwnership;
  using mfem::dgsem::DGSEMProductionMetadataView;
  using mfem::dgsem::DGSEMProductionReason;
  using mfem::dgsem::ValidateDGSEMProductionMetadata;
  using mfem::dgsem::ValidateDGSEMProductionState;

  mfem::Array<int> lex(2);
  lex[0] = 0;
  lex[1] = 1;
  mfem::Array<int> vdofs(6);
  for (int i = 0; i < vdofs.Size(); ++i)
  {
    vdofs[i] = i;
  }
  mfem::DenseMatrix metrics(2, 1);
  metrics = 1.0;
  mfem::DenseMatrix inverse_mass(2, 2);
  inverse_mass = 0.0;
  inverse_mass(0, 0) = 1.0;
  inverse_mass(1, 1) = 1.0;
  mfem::DenseMatrix normals(1, 1);
  normals(0, 0) = 1.0;
  mfem::Vector weights(1);
  weights(0) = 1.0;
  mfem::Array<int> left_elements(1);
  mfem::Array<int> left_local(1);
  mfem::Array<int> right_elements(1);
  mfem::Array<int> right_local(1);
  mfem::Array<int> ownership(1);
  mfem::Array<int> shared_state_vdofs(3);
  left_elements[0] = 0;
  left_local[0] = 0;
  right_elements[0] = -1;
  right_local[0] = -1;
  ownership[0] =
    static_cast<int>(DGSEMDeviceFaceOwnership::Boundary);
  shared_state_vdofs = 0;

  DGSEMProductionMetadataView metadata;
  metadata.dimension = 1;
  metadata.order = 1;
  metadata.elements = 1;
  metadata.dofs_per_element = 2;
  metadata.equations = 3;
  metadata.local_vector_size = 6;
  metadata.face_neighbor_vector_size = 0;
  metadata.lex_to_local = &lex;
  metadata.element_vdofs = &vdofs;
  metadata.metrics = &metrics;
  metadata.inverse_mass = &inverse_mass;
  metadata.normals = &normals;
  metadata.face_weights = &weights;
  metadata.left_elements = &left_elements;
  metadata.left_local_dofs = &left_local;
  metadata.right_elements = &right_elements;
  metadata.right_local_dofs = &right_local;
  metadata.ownership = &ownership;
  metadata.shared_state_vdofs = &shared_state_vdofs;
  mfem::Vector untouched_output(6);
  untouched_output = 8765.0;

  auto expect_metadata = [&](DGSEMProductionReason expected,
                             const std::string& label)
  {
    const auto actual = ValidateDGSEMProductionMetadata(metadata);
    ctx.Check(actual == expected && IsSentinel(untouched_output, 8765.0),
              "DGSEM production negative metadata reason/no-write: " + label);
    ++negative_checks;
  };
  expect_metadata(DGSEMProductionReason::None, "valid-control");

  metadata.dimension = 3;
  expect_metadata(DGSEMProductionReason::UnsupportedDimension, "dimension");
  metadata.dimension = 1;
  metadata.order = std::numeric_limits<int>::max();
  expect_metadata(DGSEMProductionReason::UnsupportedOrder, "order");
  metadata.order = 46340;
  expect_metadata(DGSEMProductionReason::UnsupportedOrder,
                  "derivative-matrix-index-overflow");
  metadata.order = 0;
  expect_metadata(DGSEMProductionReason::UnsupportedOrder, "zero-order");
  metadata.order = 1;
  metadata.elements = std::numeric_limits<int>::max();
  expect_metadata(DGSEMProductionReason::UnsupportedTensorElement,
                  "packed-row-index-overflow");
  metadata.elements = 1;

  lex[1] = 0;
  expect_metadata(DGSEMProductionReason::InvalidLexicographicMap,
                  "lex-permutation");
  lex[1] = 1;
  vdofs[1] = 0;
  expect_metadata(DGSEMProductionReason::InvalidElementVdofMap,
                  "vdof-permutation");
  vdofs[1] = 99;
  expect_metadata(DGSEMProductionReason::InvalidElementVdofMap,
                  "vdof-range");
  vdofs[1] = 1;

  ownership[0] = 99;
  expect_metadata(DGSEMProductionReason::InvalidFaceOwnership,
                  "ownership-enum");
  ownership[0] =
    static_cast<int>(DGSEMDeviceFaceOwnership::Boundary);
  left_local[0] = 2;
  expect_metadata(DGSEMProductionReason::InvalidFaceRange,
                  "ownership-range");
  left_local[0] = 0;

  normals(0, 0) = 0.0;
  expect_metadata(DGSEMProductionReason::InvalidNormal, "zero-normal");
  normals(0, 0) = std::numeric_limits<double>::quiet_NaN();
  expect_metadata(DGSEMProductionReason::InvalidNormal, "nonfinite-normal");
  normals(0, 0) = 1.0;
  weights(0) = 0.0;
  expect_metadata(DGSEMProductionReason::InvalidWeight, "zero-weight");
  weights(0) = std::numeric_limits<double>::infinity();
  expect_metadata(DGSEMProductionReason::InvalidWeight,
                  "nonfinite-weight");
  weights(0) = 1.0;
  metrics(0, 0) = 0.0;
  expect_metadata(DGSEMProductionReason::InvalidMetric, "zero-metric");
  metrics(0, 0) = std::numeric_limits<double>::quiet_NaN();
  expect_metadata(DGSEMProductionReason::InvalidMetric,
                  "nonfinite-metric");
  metrics(0, 0) = 1.0;

  mfem::DenseMatrix states(2, 3);
  mfem::Vector sigma(2);
  states = 0.0;
  sigma = 0.0;
  for (int row = 0; row < 2; ++row)
  {
    states(row, 0) = 1.0;
    states(row, 1) = 0.1;
    states(row, 2) = 2.505;
  }
  auto expect_state = [&](DGSEMProductionReason expected,
                          double gamma,
                          double floor,
                          const std::string& label)
  {
    const auto actual = ValidateDGSEMProductionState(
      1, gamma, floor, states, sigma);
    ctx.Check(actual == expected && IsSentinel(untouched_output, 8765.0),
              "DGSEM production negative state reason/no-write: " + label);
    ++negative_checks;
  };
  expect_state(DGSEMProductionReason::None, 1.4, 1.0e-10, "valid-control");
  expect_state(DGSEMProductionReason::InvalidGamma,
               1.0,
               1.0e-10,
               "gamma");
  expect_state(DGSEMProductionReason::InvalidDensityFloor,
               1.4,
               0.0,
               "density-floor");
  states(0, 0) = 0.0;
  expect_state(DGSEMProductionReason::InvalidDensity,
               1.4,
               1.0e-10,
               "density");
  states(0, 0) = 1.0;
  states(0, 2) = 0.001;
  expect_state(DGSEMProductionReason::InvalidPressure,
               1.4,
               1.0e-10,
               "pressure");
  states(0, 2) = 2.505;
  sigma(0) = std::numeric_limits<double>::quiet_NaN();
  expect_state(DGSEMProductionReason::InvalidSigma,
               1.4,
               1.0e-10,
               "sigma");
}

inline void CheckSlipWallFloorDependency(TestContext& ctx,
                                         int& negative_checks)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::SlipWallState host_wall(2);
  mfem::Vector interior(4);
  interior = 0.0;
  interior(0) =
    0.5 * mfem::igrlab::LimiterDefaults::kRhoMin;
  interior(3) = 1.0;
  mfem::Vector normal(2);
  normal = 0.0;
  normal(0) = 1.0;
  mfem::Vector exterior;
  mfem::FaceElementTransformations* transformation =
    mesh.GetBdrFaceTransformations(0);
  MFEM_VERIFY(transformation,
              "DGSEM floor equality control requires a boundary face.");
  host_wall.Eval(interior, normal, *transformation, exterior);
  ctx.Check(exterior(0) ==
              mfem::igrlab::LimiterDefaults::kRhoMin,
            "DGSEM production rejection floor equals host SlipWall floor");
  ++negative_checks;
}

inline void CheckUnsupportedDimensionOrderNoMutation(TestContext& ctx,
                                                     int& negative_checks,
                                                     int& assemble_rejections)
{
  using mfem::dgsem::DGSEMProductionReason;
  if (mfem::Mpi::WorldSize() != 1)
  {
    return;
  }

  auto check = [&](int dimension,
                   int order,
                   DGSEMProductionReason expected,
                   const std::string& label)
  {
    std::unique_ptr<mfem::Mesh> serial_mesh;
    if (dimension == 3)
    {
      serial_mesh = std::make_unique<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(
          1, 1, 1, mfem::Element::HEXAHEDRON));
    }
    else
    {
      serial_mesh = std::make_unique<mfem::Mesh>(
        mfem::Mesh::MakeCartesian1D(1, 1.0));
    }
    mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
    const int equations = dimension + 2;
    mfem::DG_FECollection fec(
      order, dimension, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(
      &mesh, &fec, equations, mfem::Ordering::byNODES);
    mfem::EulerFlux euler_flux(dimension, 1.4);
    mfem::RusanovFluxWithSides numerical_flux(euler_flux);
    auto form =
      std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(
      vfes, std::move(form), true);
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMVolumeDissipation(mfem::DGSEMVolumeDissipation::None);
    mfem::Array<int> marker(mesh.bdr_attributes.Max());
    if (marker.Size() > 0)
    {
      marker = 1;
      oper.AddBdrFaceIntegrator(
        std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
          numerical_flux,
          std::make_unique<mfem::SlipWallState>(dimension),
          2),
        marker);
    }
    mfem::dgsem::DGSEMOperator production_operator(oper);
    const auto selection = production_operator.BindProductionPolicy(true);
    mfem::Vector state_local(vfes.GetVSize());
    state_local = 1.0;
    mfem::Vector sentinel(vfes.GetVSize());
    sentinel = 2468.0;
    const auto result = production_operator.AssembleProductionResidual(
      state_local, sentinel, true);
    ctx.Check(!selection.supported && selection.reason == expected &&
                !result.executed && result.reason == expected &&
                IsSentinel(sentinel, 2468.0),
              "DGSEM production unsupported " + label + " fails closed");
    ++negative_checks;
    ++assemble_rejections;
  };

  check(3, 1, DGSEMProductionReason::UnsupportedDimension, "dimension");
  check(1, 0, DGSEMProductionReason::UnsupportedOrder, "zero-order");
}

inline void CheckPolicyFingerprintNegatives(TestContext& ctx,
                                            int& negative_checks,
                                            int& assemble_rejections)
{
  using mfem::dgsem::DGSEMProductionReason;
  if (mfem::Mpi::WorldSize() != 1)
  {
    return;
  }

  constexpr double gamma = 1.4;
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_FECollection fec(1, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(
    &mesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector constant = EulerConsFromPrim2D(1.0, 0.1, -0.05, 1.0, gamma);
  mfem::ParGridFunction state(&vfes);
  FillConstantState(constant, state);
  mfem::Array<int> marker(mesh.bdr_attributes.Max());
  if (marker.Size() > 0)
  {
    marker = 1;
  }

  auto configure = [&](mfem::DGHyperbolicConservationLaws& oper,
                       const mfem::NumericalFlux& numerical_flux,
                       int wall_dimension,
                       bool add_boundary = true)
  {
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMVolumeDissipation(
      mfem::DGSEMVolumeDissipation::None);
    oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
    oper.SetDGSEMGamma(gamma);
    if (add_boundary && marker.Size() > 0)
    {
      oper.AddBdrFaceIntegrator(
        std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
          numerical_flux,
          std::make_unique<mfem::SlipWallState>(wall_dimension),
          2),
        marker);
    }
  };

  auto expect_rejection = [&](mfem::dgsem::DGSEMOperator& production,
                              DGSEMProductionReason expected,
                              const std::string& label)
  {
    const auto selection = production.BindProductionPolicy(true);
    mfem::Vector sentinel(vfes.GetVSize());
    sentinel = 9182.0;
    const auto result = production.AssembleProductionResidual(
      state, sentinel, true);
    ctx.Check(!selection.supported && selection.reason == expected &&
                !result.executed && result.reason == expected &&
                IsSentinel(sentinel, 9182.0),
              "DGSEM production policy rejection/no-write: " + label);
    ++negative_checks;
    ++assemble_rejections;
  };

  {
    mfem::EulerFlux euler(2, gamma);
    mfem::DGSEMSurfaceFlux surface(euler);
    surface.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::Rusanov);
    auto form =
      std::make_unique<mfem::HyperbolicFormIntegrator>(surface, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, surface, 2);
    mfem::dgsem::DGSEMOperator production(oper);
    const auto accepted = production.BindProductionPolicy(true);
    ctx.Check(accepted.supported,
              "DGSEM same-pointer surface mutation starts supported");
    surface.SetSurfaceFluxType(
      mfem::DGSEMSurfaceFluxType::DiffusiveRusanov);
    expect_rejection(production,
                     DGSEMProductionReason::UnsupportedSurfaceFlux,
                     "same-pointer-surface-mutation");
    surface.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::Rusanov);
    mfem::Vector restored(vfes.GetVSize());
    restored = -318.0;
    const auto rebound = production.BindProductionPolicy(true);
    const auto restored_result =
      production.AssembleProductionResidual(state, restored, true);
    ctx.Check(rebound.supported && restored_result.executed,
              "DGSEM same-pointer surface transition back is detected");
  }

  {
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, numerical_flux, 2);
    mfem::dgsem::DGSEMOperator production(oper);
    ctx.Check(production.BindProductionPolicy(true).supported,
              "DGSEM boundary-overlap mutation starts supported");
    if (marker.Size() > 0)
    {
      oper.AddBdrFaceIntegrator(
        std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
          numerical_flux,
          std::make_unique<mfem::SlipWallState>(2),
          2),
        marker);
    }
    expect_rejection(production,
                     DGSEMProductionReason::BoundaryOverlap,
                     "boundary-addition-overlap");
  }

  {
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, numerical_flux, 1);
    mfem::dgsem::DGSEMOperator production(oper);
    expect_rejection(production,
                     DGSEMProductionReason::BoundaryDimensionMismatch,
                     "wrong-wall-dimension");
  }

  {
    mfem::EulerFlux form_euler(2, gamma);
    mfem::EulerFlux selected_euler(2, gamma);
    mfem::RusanovFluxWithSides form_flux(form_euler);
    mfem::RusanovFluxWithSides selected_flux(selected_euler);
    auto form =
      std::make_unique<mfem::HyperbolicFormIntegrator>(form_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, selected_flux, 2);
    mfem::dgsem::DGSEMOperator production(oper);
    expect_rejection(production,
                     DGSEMProductionReason::FluxIdentityMismatch,
                     "numerical-physical-flux-identity");
  }

  {
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides selected_flux(euler);
    mfem::RusanovFluxWithSides boundary_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      selected_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, selected_flux, 2, false);
    if (marker.Size() > 0)
    {
      oper.AddBdrFaceIntegrator(
        std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
          boundary_flux,
          std::make_unique<mfem::SlipWallState>(2),
          2),
        marker);
    }
    mfem::dgsem::DGSEMOperator production(oper);
    expect_rejection(production,
                     DGSEMProductionReason::BoundaryFluxMismatch,
                     "boundary-numerical-flux-identity");
  }

  {
    mfem::EulerFlux euler(2, 1.67);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, numerical_flux, 2);
    mfem::dgsem::DGSEMOperator production(oper);
    expect_rejection(production,
                     DGSEMProductionReason::EulerGammaMismatch,
                     "underlying-euler-gamma-mismatch");
  }

  {
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, numerical_flux, 2);
    mfem::dgsem::DGSEMOperator production(oper);
    ctx.Check(production.BindProductionPolicy(true).supported,
              "DGSEM below-wall-floor state starts with supported policy");
    mfem::Vector below_floor(state);
    mfem::Array<int> vdofs;
    vfes.GetElementVDofs(0, vdofs);
    below_floor.HostWrite()[AbsoluteDof(vdofs[0])] =
      0.5 * mfem::igrlab::LimiterDefaults::kRhoMin;
    mfem::Vector sentinel(vfes.GetVSize());
    sentinel = 9182.0;
    const auto result = production.AssembleProductionResidual(
      below_floor, sentinel, true);
    ctx.Check(!result.executed &&
                result.reason ==
                  DGSEMProductionReason::DensityBelowBoundaryFloor &&
                IsSentinel(sentinel, 9182.0),
              "DGSEM below host SlipWall density floor rejects/no-write");
    ++negative_checks;
    ++assemble_rejections;
  }

  {
    mfem::DG_FECollection gl_fec(
      1, 2, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace gl_vfes(
      &mesh, &gl_fec, 4, mfem::Ordering::byNODES);
    mfem::ParGridFunction gl_state(&gl_vfes);
    FillConstantState(constant, gl_state);
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(
      gl_vfes, std::move(form), true);
    configure(oper, numerical_flux, 2);
    mfem::dgsem::DGSEMOperator production(oper);
    const auto selection = production.BindProductionPolicy(true);
    mfem::Vector sentinel(gl_vfes.GetVSize());
    sentinel = 9182.0;
    const auto result = production.AssembleProductionResidual(
      gl_state, sentinel, true);
    ctx.Check(!selection.supported &&
                selection.reason ==
                  DGSEMProductionReason::UnsupportedNodalBasis &&
                !result.executed &&
                result.reason ==
                  DGSEMProductionReason::UnsupportedNodalBasis &&
                IsSentinel(sentinel, 9182.0),
              "DGSEM same-shape Gauss-Legendre strict rejection/no-write");
    ++negative_checks;
    ++assemble_rejections;
  }
}

inline void CheckCollectiveRejectionSemantics(TestContext& ctx,
                                              int& negative_checks,
                                              int& assemble_rejections,
                                              int& collective_rejections)
{
#ifdef MFEM_USE_MPI
  using mfem::dgsem::DGSEMProductionReason;
  if (mfem::Mpi::WorldSize() != 2)
  {
    return;
  }

  MPI_Comm communicator = MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD, &communicator);
  int comparison = MPI_IDENT;
  MPI_Comm_compare(communicator, MPI_COMM_WORLD, &comparison);
  ctx.Check(comparison != MPI_IDENT,
            "DGSEM collective fixture owns a non-world communicator");
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);

  {
  constexpr double gamma = 1.4;
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(communicator, serial_mesh);
  mfem::DG_FECollection fec(1, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(
    &mesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::Vector constant = EulerConsFromPrim2D(1.0, 0.1, -0.05, 1.0, gamma);
  mfem::ParGridFunction state(&vfes);
  FillConstantState(constant, state);
  mfem::Array<int> marker(mesh.bdr_attributes.Max());
  if (marker.Size() > 0)
  {
    marker = 1;
  }

  auto configure = [&](mfem::DGHyperbolicConservationLaws& oper,
                       const mfem::NumericalFlux& numerical_flux,
                       int wall_dimension)
  {
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMVolumeDissipation(
      mfem::DGSEMVolumeDissipation::None);
    oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
    oper.SetDGSEMGamma(gamma);
    if (marker.Size() > 0)
    {
      oper.AddBdrFaceIntegrator(
        std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
          numerical_flux,
          std::make_unique<mfem::SlipWallState>(wall_dimension),
          2),
        marker);
    }
  };

  auto check_collective = [&](const mfem::dgsem::DGSEMProductionSelection& s,
                              const mfem::dgsem::DGSEMProductionResult& r,
                              DGSEMProductionReason expected,
                              bool output_unchanged,
                              const std::string& label,
                              bool expect_supported_selection)
  {
    int local_reason = static_cast<int>(r.reason);
    int minimum_reason = local_reason;
    int maximum_reason = local_reason;
    int local_unchanged = output_unchanged ? 1 : 0;
    int all_unchanged = local_unchanged;
    MPI_Allreduce(&local_reason,
                  &minimum_reason,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  communicator);
    MPI_Allreduce(&local_reason,
                  &maximum_reason,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  communicator);
    MPI_Allreduce(&local_unchanged,
                  &all_unchanged,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  communicator);
    const bool selection_matches = expect_supported_selection
                                     ? s.supported
                                     : (!s.supported && s.reason == expected);
    ctx.Check(selection_matches && !r.executed && r.reason == expected &&
                minimum_reason == maximum_reason &&
                minimum_reason == static_cast<int>(expected) &&
                all_unchanged == 1,
              "DGSEM collective rejection/no-hang/no-write: " + label);
    ++negative_checks;
    ++assemble_rejections;
    ++collective_rejections;
  };

  {
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, numerical_flux, 2);
    mfem::dgsem::DGSEMOperator production(oper);
    const auto accepted = production.BindProductionPolicy(true);
    ctx.Check(accepted.supported && accepted.ranks == 2,
              "DGSEM non-world communicator policy is supported on two ranks");

    mfem::Vector asymmetric_state(state);
    if (rank == 0)
    {
      mfem::Array<int> vdofs;
      vfes.GetElementVDofs(0, vdofs);
      asymmetric_state.HostWrite()[AbsoluteDof(vdofs[0])] = 0.0;
    }
    mfem::Vector state_output(vfes.GetVSize());
    state_output = 7412.0;
    const auto state_result = production.AssembleProductionResidual(
      asymmetric_state, state_output, true);
    check_collective(accepted,
                     state_result,
                     DGSEMProductionReason::InvalidDensity,
                     IsSentinel(state_output, 7412.0),
                     "asymmetric-invalid-state",
                     true);

    mfem::Vector output(rank == 0 ? vfes.GetVSize() - 1
                                  : vfes.GetVSize());
    output = 7412.0;
    const auto output_result =
      production.AssembleProductionResidual(state, output, true);
    check_collective(accepted,
                     output_result,
                     DGSEMProductionReason::OutputSizeMismatch,
                     IsSentinel(output, 7412.0),
                     "asymmetric-output-size",
                     true);
  }

  {
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, numerical_flux, 2);
    if (rank == 0)
    {
      oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Chandrashekar);
    }
    mfem::dgsem::DGSEMOperator production(oper);
    const bool local_strict = rank != 0;
    const auto selection = production.BindProductionPolicy(local_strict);
    mfem::Vector output(vfes.GetVSize());
    output = 7412.0;
    const auto result = production.AssembleProductionResidual(
      state, output, local_strict);
    check_collective(selection,
                     result,
                     DGSEMProductionReason::CollectivePolicyMismatch,
                     IsSentinel(output, 7412.0),
                     "asymmetric-policy-collective-strict-or",
                     false);
    ctx.Check(selection.strict,
              "DGSEM collective strict mode is fail-closed OR on subcommunicator");
  }

  {
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    configure(oper, numerical_flux, rank == 0 ? 1 : 2);
    mfem::dgsem::DGSEMOperator production(oper);
    const auto selection = production.BindProductionPolicy(true);
    mfem::Vector output(vfes.GetVSize());
    output = 7412.0;
    const auto result = production.AssembleProductionResidual(
      state, output, true);
    check_collective(selection,
                     result,
                     DGSEMProductionReason::CollectivePolicyMismatch,
                     IsSentinel(output, 7412.0),
                     "asymmetric-boundary-policy",
                     false);
  }
  }

  MPI_Comm_free(&communicator);
#else
  (void)ctx;
  (void)negative_checks;
  (void)assemble_rejections;
  (void)collective_rejections;
#endif
}

inline void RunDGSEMProductionMultStrictRejectionProbe(TestContext& ctx)
{
  if (!::igrlab::GetEnv("DG_IGR_DGSEM_MULT_STRICT_REJECTION_PROBE"))
  {
    return;
  }
  ctx.Check(mfem::Mpi::WorldSize() == 1,
            "DGSEM production Mult strict probe requires one rank");
  if (mfem::Mpi::WorldSize() != 1)
  {
    return;
  }

  constexpr int dimension = 1;
  constexpr int equations = dimension + 2;
  constexpr double gamma = 1.4;
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian1D(2, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_FECollection fec(
    1, dimension, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(
    &mesh, &fec, equations, mfem::Ordering::byNODES);
  mfem::EulerFlux euler(dimension, gamma);
  mfem::RusanovFluxWithSides numerical_flux(euler);
  auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
    numerical_flux, 2);
  mfem::DGHyperbolicConservationLaws oper(
    vfes, std::move(form), true);
  oper.SetNumericalFlux(&numerical_flux);
  oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
  oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Chandrashekar);
  oper.SetDGSEMVolumeDissipation(
    mfem::DGSEMVolumeDissipation::None);
  oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
  oper.SetDGSEMGamma(gamma);
  mfem::Array<int> marker(mesh.bdr_attributes.Max());
  if (marker.Size() > 0)
  {
    marker = 1;
    oper.AddBdrFaceIntegrator(
      std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
        numerical_flux,
        std::make_unique<mfem::SlipWallState>(dimension),
        2),
      marker);
  }

  mfem::ParGridFunction state(&vfes);
  FillConstantState(
    EulerConsFromPrim1D(1.0, 0.1, 1.0, gamma), state);
  mfem::Vector state_true;
  state.GetTrueDofs(state_true);
  mfem::Vector output(vfes.GetTrueVSize());
  output = 6421.0;
  std::cout << "[DGSEM_MULT_STRICT_PROBE] reason="
            << "unsupported-volume-flux output_sentinel_before="
            << (IsSentinel(output, 6421.0) ? 1 : 0) << '\n';
  ScopedEnvironment strict("DG_IGR_DGSEM_PRODUCTION_STRICT", "1");
  oper.Mult(state_true, output);
  MFEM_ABORT("DGSEM production Mult strict rejection probe unexpectedly returned.");
}

inline void RunDGSEMProductionSplitCommunicatorTests(TestContext& ctx)
{
#ifdef MFEM_USE_MPI
  using mfem::dgsem::DGSEMProductionReason;
  const int world_size = mfem::Mpi::WorldSize();
  ctx.Check(world_size == 4,
            "DGSEM split-communicator probe requires exactly four ranks");
  if (world_size != 4)
  {
    return;
  }

#if defined(HYPRE_USING_UMPIRE_DEVICE)
  // Four tiny MPI ranks share one CI GPU in this probe.  Size HYPRE's
  // growable device pool before its first allocation so four default 4 GiB
  // pools do not exhaust the device merely while constructing ParFES data.
  const HYPRE_Int pool_status = HYPRE_SetUmpireDevicePoolSize(
    std::size_t{256} * 1024 * 1024);
  ctx.Check(pool_status == 0,
            "DGSEM split probe configures its test-only HYPRE device pool");
#endif

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  const int group = world_rank / 2;
  MPI_Comm communicator = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, group, world_rank, &communicator);
  int local_rank = 0;
  int group_size = 0;
  MPI_Comm_rank(communicator, &local_rank);
  MPI_Comm_size(communicator, &group_size);
  int comparison = MPI_IDENT;
  MPI_Comm_compare(communicator, MPI_COMM_WORLD, &comparison);
  ctx.Check(group_size == 2 && comparison == MPI_UNEQUAL,
            "DGSEM split probe owns a true two-rank subset communicator");

  int result_reason = static_cast<int>(DGSEMProductionReason::None);
  bool result_ok = false;
  double expected_cfl = 0.0;
  double actual_cfl = 0.0;
  bool cfl_ok = false;
  double expected_fallback_cfl = 0.0;
  double actual_fallback_cfl = 0.0;
  bool fallback_cfl_ok = false;
  {
    constexpr double gamma = 1.4;
    mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mfem::ParMesh mesh(communicator, serial_mesh);
    mfem::DG_FECollection fec(1, 2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(
      &mesh, &fec, 4, mfem::Ordering::byNODES);
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMVolumeDissipation(
      mfem::DGSEMVolumeDissipation::None);
    oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
    oper.SetDGSEMGamma(gamma);
    mfem::Array<int> marker(mesh.bdr_attributes.Max());
    if (marker.Size() > 0)
    {
      marker = 1;
      oper.AddBdrFaceIntegrator(
        std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
          numerical_flux,
          std::make_unique<mfem::SlipWallState>(2),
          2),
        marker);
    }

    mfem::ParGridFunction state(&vfes);
    const mfem::Vector constant =
      EulerConsFromPrim2D(1.0, 0.1, -0.05, 1.0, gamma);
    FillConstantState(constant, state);
    mfem::Vector input(state);
    const DGSEMProductionReason expected =
      group == 0 ? DGSEMProductionReason::InvalidDensity
                 : DGSEMProductionReason::OutputSizeMismatch;
    if (group == 0 && local_rank == 0)
    {
      mfem::Array<int> vdofs;
      vfes.GetElementVDofs(0, vdofs);
      input.HostWrite()[AbsoluteDof(vdofs[0])] = 0.0;
    }
    mfem::Vector output(
      group == 1 && local_rank == 0 ? vfes.GetVSize() - 1
                                    : vfes.GetVSize());
    const double sentinel = 8200.0 + group;
    output = sentinel;

    mfem::dgsem::DGSEMOperator production(oper);
    const auto selection = production.BindProductionPolicy(true);
    const auto result = production.AssembleProductionResidual(
      input, output, true);
    result_reason = static_cast<int>(result.reason);
    int minimum_reason = result_reason;
    int maximum_reason = result_reason;
    int local_unchanged = IsSentinel(output, sentinel) ? 1 : 0;
    int all_unchanged = local_unchanged;
    MPI_Allreduce(&result_reason,
                  &minimum_reason,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  communicator);
    MPI_Allreduce(&result_reason,
                  &maximum_reason,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  communicator);
    MPI_Allreduce(&local_unchanged,
                  &all_unchanged,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  communicator);
    result_ok = selection.supported && selection.ranks == 2 &&
                !result.executed && result.reason == expected &&
                minimum_reason == maximum_reason &&
                minimum_reason == static_cast<int>(expected) &&
                all_unchanged == 1;
  }

  {
    constexpr double gamma = 1.4;
    mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mfem::ParMesh mesh(communicator, serial_mesh);
    mfem::DG_FECollection fec(1, 2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(
      &mesh, &fec, 4, mfem::Ordering::byNODES);
    mfem::EulerFlux euler(2, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMVolumeDissipation(
      mfem::DGSEMVolumeDissipation::None);
    oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
    oper.SetDGSEMGamma(gamma);
    mfem::Array<int> marker(mesh.bdr_attributes.Max());
    if (marker.Size() > 0)
    {
      marker = 1;
      oper.AddBdrFaceIntegrator(
        std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
          numerical_flux,
          std::make_unique<mfem::SlipWallState>(2),
          2),
        marker);
    }

    mfem::ParGridFunction state(&vfes);
    const mfem::Vector constant = group == 0
                                    ? EulerConsFromPrim2D(
                                        1.0, 0.05, -0.02, 0.8, gamma)
                                    : EulerConsFromPrim2D(
                                        1.0, 0.85, 0.30, 2.0, gamma);
    FillConstantState(constant, state);
    mfem::Vector input(state);
    mfem::Vector host_weak(vfes.GetVSize());
    host_weak = 0.0;
    mfem::dgsem::DGSEMOperator host(oper);
    const double local_surface_speed =
      host.AssembleSurface(input, host_weak);
    const double local_volume_speed =
      host.AssembleVolume(input, host_weak);
    const double local_expected =
      std::max(local_surface_speed, local_volume_speed);
    MPI_Allreduce(&local_expected,
                  &expected_cfl,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  communicator);

    mfem::dgsem::DGSEMOperator production(oper);
    const auto selection = production.BindProductionPolicy(true);
    mfem::Vector output(vfes.GetVSize());
    output = -8300.0 - group;
    const auto result = production.AssembleProductionResidual(
      input, output, true);
    const double direct_cfl = result.max_characteristic_speed;
    mfem::Vector admission_output(vfes.GetVSize());
    admission_output = -8350.0 - group;
    const auto admission_result =
      production.AssembleProductionResidual(input,
                                            admission_output,
                                            true,
                                            nullptr,
                                            /*admission_only=*/true);
    ctx.Check(admission_result.executed &&
                admission_result.reason == DGSEMProductionReason::None &&
                !admission_result.admission_q_sigma.enabled &&
                std::abs(admission_result.max_characteristic_speed -
                         expected_cfl) < 5.0e-13 &&
                IsSentinel(admission_output, -8350.0 - group),
              "DGSEM split-subcommunicator admission refresh reduces its "
              "CFL maximum without writing a residual");
    mfem::Vector host_local(vfes.GetVSize());
    host.ApplyInverseMass(host_weak, host_local);
    mfem::Vector host_true(vfes.GetTrueVSize());
    if (const mfem::Operator* prolongation = vfes.GetProlongationMatrix())
    {
      host_true = 0.0;
      prolongation->MultTranspose(host_local, host_true);
    }
    else
    {
      host_true = host_local;
    }
    mfem::Vector input_true;
    state.GetTrueDofs(input_true);
    mfem::Vector mult_true(vfes.GetTrueVSize());
    mult_true = -8400.0 - group;
    ScopedEnvironment split_strict(
      "DG_IGR_DGSEM_PRODUCTION_STRICT", "1");
    oper.Mult(input_true, mult_true);
    mfem::Vector mult_difference(mult_true);
    mult_difference -= host_true;
    const auto mult_selection = oper.GetDGSEMProductionSelection();
    const auto mult_bridge = oper.GetDGSEMProductionBridgeEvidence();
    const double local_mult_norm_squared =
      mult_difference * mult_difference;
    double group_mult_norm_squared = local_mult_norm_squared;
    MPI_Allreduce(&local_mult_norm_squared,
                  &group_mult_norm_squared,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  communicator);
    const bool mult_ok = mult_selection.supported &&
                         mult_selection.strict &&
                         mult_selection.ranks == 2 &&
                         mult_bridge.production_mult_calls == 1 &&
                         mult_bridge.operator_constructions == 1 &&
                         std::sqrt(group_mult_norm_squared) < 2.0e-10;
    actual_cfl = oper.GetMaxCharSpeed();
    double minimum_actual = actual_cfl;
    double maximum_actual = actual_cfl;
    MPI_Allreduce(&actual_cfl,
                  &minimum_actual,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  communicator);
    MPI_Allreduce(&actual_cfl,
                  &maximum_actual,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  communicator);
    cfl_ok = selection.supported && selection.ranks == 2 &&
             result.executed && mult_ok &&
             std::abs(direct_cfl - expected_cfl) < 5.0e-13 &&
             std::abs(actual_cfl - expected_cfl) < 5.0e-13 &&
             std::abs(maximum_actual - minimum_actual) < 5.0e-13;
    {
      ScopedEnvironment split_fallback(
        "DG_IGR_DGSEM_PRODUCTION_STRICT", nullptr);
      oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Chandrashekar);
      mfem::Vector fallback_weak(vfes.GetVSize());
      const double fallback_surface_speed =
        host.AssembleSurface(input, fallback_weak);
      const double fallback_volume_speed =
        host.AssembleVolume(input, fallback_weak);
      const double local_fallback_cfl =
        std::max(fallback_surface_speed, fallback_volume_speed);
      MPI_Allreduce(&local_fallback_cfl,
                    &expected_fallback_cfl,
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    communicator);
      mfem::Vector fallback_host_local(vfes.GetVSize());
      host.ApplyInverseMass(fallback_weak, fallback_host_local);
      mfem::Vector fallback_host_true(vfes.GetTrueVSize());
      if (const mfem::Operator* prolongation =
            vfes.GetProlongationMatrix())
      {
        fallback_host_true = 0.0;
        prolongation->MultTranspose(
          fallback_host_local, fallback_host_true);
      }
      else
      {
        fallback_host_true = fallback_host_local;
      }
      mfem::Vector fallback_true(vfes.GetTrueVSize());
      fallback_true = -8500.0 - group;
      oper.Mult(input_true, fallback_true);
      actual_fallback_cfl = oper.GetMaxCharSpeed();
      mfem::Vector fallback_difference(fallback_true);
      fallback_difference -= fallback_host_true;
      const double local_fallback_norm_squared =
        fallback_difference * fallback_difference;
      double group_fallback_norm_squared = local_fallback_norm_squared;
      MPI_Allreduce(&local_fallback_norm_squared,
                    &group_fallback_norm_squared,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    communicator);
      double fallback_minimum = actual_fallback_cfl;
      double fallback_maximum = actual_fallback_cfl;
      MPI_Allreduce(&actual_fallback_cfl,
                    &fallback_minimum,
                    1,
                    MPI_DOUBLE,
                    MPI_MIN,
                    communicator);
      MPI_Allreduce(&actual_fallback_cfl,
                    &fallback_maximum,
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    communicator);
      const auto fallback_selection =
        oper.GetDGSEMProductionSelection();
      fallback_cfl_ok = !fallback_selection.supported &&
                        !fallback_selection.strict &&
                        fallback_selection.reason ==
                          DGSEMProductionReason::UnsupportedVolumeFlux &&
                        std::sqrt(group_fallback_norm_squared) < 2.0e-10 &&
                        std::abs(actual_fallback_cfl -
                                 expected_fallback_cfl) < 5.0e-13 &&
                        std::abs(fallback_maximum - fallback_minimum) <
                          5.0e-13;
      oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    }
  }

  MPI_Comm_free(&communicator);

  int local_record[4] = {
    world_rank, group, local_rank, result_reason};
  int records[16] = {};
  MPI_Allgather(local_record,
                4,
                MPI_INT,
                records,
                4,
                MPI_INT,
                MPI_COMM_WORLD);
  double local_cfl_record[2] = {expected_cfl, actual_cfl};
  double cfl_records[8] = {};
  MPI_Allgather(local_cfl_record,
                2,
                MPI_DOUBLE,
                cfl_records,
                2,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
  double local_fallback_cfl_record[2] = {
    expected_fallback_cfl, actual_fallback_cfl};
  double fallback_cfl_records[8] = {};
  MPI_Allgather(local_fallback_cfl_record,
                2,
                MPI_DOUBLE,
                fallback_cfl_records,
                2,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
  bool exact_mapping = result_ok;
  for (int rank = 0; rank < 4; ++rank)
  {
    const int expected_reason = static_cast<int>(
      rank < 2 ? DGSEMProductionReason::InvalidDensity
               : DGSEMProductionReason::OutputSizeMismatch);
    exact_mapping =
      exact_mapping && records[4 * rank] == rank &&
      records[4 * rank + 1] == rank / 2 &&
      records[4 * rank + 2] == rank % 2 &&
      records[4 * rank + 3] == expected_reason;
  }
  ctx.Check(exact_mapping,
            "DGSEM split groups converge only to their own rejection reason");
  bool exact_cfl = cfl_ok;
  for (int rank = 0; rank < 4; ++rank)
  {
    exact_cfl =
      exact_cfl &&
      std::abs(cfl_records[2 * rank] -
               cfl_records[2 * (rank / 2) * 2]) < 5.0e-13 &&
      std::abs(cfl_records[2 * rank + 1] -
               cfl_records[2 * rank]) < 5.0e-13;
  }
  exact_cfl =
    exact_cfl && std::abs(cfl_records[0] - cfl_records[4]) > 1.0e-3;
  ctx.Check(exact_cfl,
            "DGSEM split groups reduce only their own distinct CFL maximum");
  bool exact_fallback_cfl = fallback_cfl_ok;
  for (int rank = 0; rank < 4; ++rank)
  {
    exact_fallback_cfl =
      exact_fallback_cfl &&
      std::abs(fallback_cfl_records[2 * rank] -
               fallback_cfl_records[2 * (rank / 2) * 2]) < 5.0e-13 &&
      std::abs(fallback_cfl_records[2 * rank + 1] -
               fallback_cfl_records[2 * rank]) < 5.0e-13;
  }
  exact_fallback_cfl =
    exact_fallback_cfl &&
    std::abs(fallback_cfl_records[0] - fallback_cfl_records[4]) > 1.0e-3;
  ctx.Check(exact_fallback_cfl,
            "DGSEM split ordinary fallback reduces CFL only on owning communicator");
  if (world_rank == 0)
  {
    for (int rank = 0; rank < 4; ++rank)
    {
      std::cout
        << "[DGSEM_SPLIT_MAP] world_rank=" << records[4 * rank]
        << " group=" << records[4 * rank + 1]
        << " local_rank=" << records[4 * rank + 2]
        << " reason=" << mfem::dgsem::DGSEMProductionReasonName(
             static_cast<DGSEMProductionReason>(records[4 * rank + 3]))
        << "\n";
    }
    std::cout
      << "[DGSEM_SPLIT_CFL] group=0 expected="
      << FormatSci(cfl_records[0])
      << " actual=" << FormatSci(cfl_records[1]) << "\n"
      << "[DGSEM_SPLIT_CFL] group=1 expected="
      << FormatSci(cfl_records[4])
      << " actual=" << FormatSci(cfl_records[5]) << "\n"
      << "[DGSEM_SPLIT_FALLBACK_CFL] group=0 expected="
      << FormatSci(fallback_cfl_records[0])
      << " actual=" << FormatSci(fallback_cfl_records[1]) << "\n"
      << "[DGSEM_SPLIT_FALLBACK_CFL] group=1 expected="
      << FormatSci(fallback_cfl_records[4])
      << " actual=" << FormatSci(fallback_cfl_records[5]) << "\n";
  }
#else
  (void)ctx;
#endif
}

inline void CheckDGSEMProductionMultSerialIdentityAndMove(TestContext& ctx)
{
  if (mfem::Mpi::WorldSize() != 1)
  {
    return;
  }
  static_assert(
    std::is_move_constructible<mfem::DGHyperbolicConservationLaws>::value,
    "DG production operator must remain move constructible");
  static_assert(
    !std::is_move_assignable<mfem::DGHyperbolicConservationLaws>::value,
    "DG production operator cannot safely move-assign its FE-space reference");

  constexpr int dimension = 1;
  constexpr int equations = dimension + 2;
  constexpr double gamma = 1.4;
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(2, 1.0);
  mfem::DG_FECollection fec(
    2, dimension, mfem::BasisType::GaussLobatto);
  mfem::FiniteElementSpace vfes(
    &mesh, &fec, equations, mfem::Ordering::byNODES);
  mfem::EulerFlux euler(dimension, gamma);
  mfem::RusanovFluxWithSides numerical_flux(euler);
  auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
    numerical_flux, 2);
  mfem::DGHyperbolicConservationLaws oper(
    vfes, std::move(form), true);
  oper.SetNumericalFlux(&numerical_flux);
  oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
  oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
  oper.SetDGSEMVolumeDissipation(
    mfem::DGSEMVolumeDissipation::None);
  oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
  oper.SetDGSEMGamma(gamma);
  mfem::Array<int> marker(mesh.bdr_attributes.Max());
  if (marker.Size() > 0)
  {
    marker = 1;
    oper.AddBdrFaceIntegrator(
      std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
        numerical_flux,
        std::make_unique<mfem::SlipWallState>(dimension),
        2),
      marker);
  }

  mfem::GridFunction state(&vfes);
  auto state_function = [=](const mfem::Vector& point,
                            mfem::Vector& value)
  {
    value = EulerConsFromPrim1D(
      1.0 + 0.02 * point(0),
      0.12 + 0.01 * point(0),
      1.0 + 0.03 * point(0),
      gamma);
  };
  mfem::VectorFunctionCoefficient coefficient(equations, state_function);
  state.ProjectCoefficient(coefficient);
  mfem::Vector input(state);

  mfem::dgsem::DGSEMOperator host(oper);
  mfem::Vector weak(vfes.GetVSize());
  const double face_speed = host.AssembleSurface(input, weak);
  const double volume_speed = host.AssembleVolume(input, weak);
  mfem::Vector expected(vfes.GetVSize());
  host.ApplyInverseMass(weak, expected);
  const double expected_speed = std::max(face_speed, volume_speed);

  mfem::Vector output(vfes.GetTrueVSize());
  output = -7100.0;
  oper.Mult(input, output);
  mfem::Vector difference(output);
  difference -= expected;
  const auto selection = oper.GetDGSEMProductionSelection();
  const auto bridge = oper.GetDGSEMProductionBridgeEvidence();
  ctx.Check(selection.supported && selection.strict &&
              selection.reason ==
                mfem::dgsem::DGSEMProductionReason::None &&
              difference.Norml2() < 2.0e-10 &&
              std::abs(oper.GetMaxCharSpeed() - expected_speed) < 5.0e-13 &&
              bridge.operator_ready && bridge.x_local_size == 0 &&
              bridge.z_local_size == 0 &&
              bridge.y_local_size == vfes.GetVSize() &&
              bridge.operator_constructions == 1 &&
              bridge.operator_rebinds == 0 &&
              bridge.production_mult_calls == 1,
            "DGSEM production Mult preserves serial identity-vector semantics");

  mfem::DGHyperbolicConservationLaws moved(std::move(oper));
  mfem::Vector moved_output(vfes.GetTrueVSize());
  moved_output = -7200.0;
  moved.Mult(input, moved_output);
  mfem::Vector moved_difference(moved_output);
  moved_difference -= expected;
  const auto moved_selection = moved.GetDGSEMProductionSelection();
  const auto moved_bridge = moved.GetDGSEMProductionBridgeEvidence();
  const auto moved_counters = moved.GetDGSEMProductionCacheCounters();
  ctx.Check(moved_selection.supported && moved_selection.strict &&
              moved_difference.Norml2() < 2.0e-10 &&
              std::abs(moved.GetMaxCharSpeed() - expected_speed) < 5.0e-13 &&
              moved_bridge.operator_constructions == 1 &&
              moved_bridge.operator_rebinds == 1 &&
              moved_bridge.production_mult_calls == 2 &&
              moved_counters.topology_builds == 1 &&
              moved_counters.workspace_builds == 1,
            "DGSEM persistent production bridge rebinds safely after parent move");
}

inline void CheckDGSEMProductionMixedEulerBoundaries(TestContext& ctx)
{
  if (mfem::Mpi::WorldSize() != 1)
  {
    return;
  }

  constexpr int dimension = 2;
  constexpr int equations = 4;
  constexpr int order = 2;
  constexpr double gamma = 1.4;
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  const int boundary_attributes = mesh.bdr_attributes.Max();
  ctx.Check(boundary_attributes >= 3,
            "DGSEM mixed-boundary fixture exposes inlet/outlet/wall attributes");
  if (boundary_attributes < 3)
  {
    return;
  }

  mfem::DG_FECollection fec(
    order, dimension, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(
    &mesh, &fec, equations, mfem::Ordering::byNODES);
  mfem::EulerFlux euler(dimension, gamma);
  mfem::RusanovFluxWithSides numerical_flux(euler);
  auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
    numerical_flux, 2);
  mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
  oper.SetNumericalFlux(&numerical_flux);
  oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
  oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
  oper.SetDGSEMVolumeDissipation(mfem::DGSEMVolumeDissipation::None);
  oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
  oper.SetDGSEMGamma(gamma);

  mfem::Array<int> inlet_marker(boundary_attributes);
  mfem::Array<int> outlet_marker(boundary_attributes);
  mfem::Array<int> wall_marker(boundary_attributes);
  inlet_marker = 0;
  outlet_marker = 0;
  wall_marker = 0;
  inlet_marker[0] = 1;
  outlet_marker[1] = 1;
  for (int attr = 2; attr < boundary_attributes; ++attr)
  {
    wall_marker[attr] = 1;
  }
  const mfem::Vector inlet =
    EulerConsFromPrim2D(1.08, 0.32, -0.04, 1.03, gamma);
  oper.AddBdrFaceIntegrator(
    std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
      numerical_flux,
      std::make_unique<mfem::SupersonicInletState>(inlet),
      2),
    inlet_marker);
  oper.AddBdrFaceIntegrator(
    std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
      numerical_flux,
      std::make_unique<mfem::SupersonicOutletState>(),
      2),
    outlet_marker);
  oper.AddBdrFaceIntegrator(
    std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
      numerical_flux,
      std::make_unique<mfem::SlipWallState>(dimension),
      2),
    wall_marker);

  mfem::ParGridFunction state(&vfes);
  auto state_function = [=](const mfem::Vector& point,
                            mfem::Vector& value)
  {
    value = EulerConsFromPrim2D(1.0 + 0.04 * point(0),
                               0.18 + 0.03 * point(1),
                               -0.06 + 0.02 * point(0),
                               1.0 + 0.025 * point(1),
                               gamma);
  };
  mfem::VectorFunctionCoefficient coefficient(equations, state_function);
  state.ProjectCoefficient(coefficient);

  mfem::dgsem::DGSEMOperator host(oper);
  mfem::Vector weak(vfes.GetVSize());
  mfem::Vector host_boundary_flux;
  const double face_speed =
    host.AssembleSurface(state, weak, &host_boundary_flux);
  const double volume_speed = host.AssembleVolume(state, weak);
  const int scalar_vsize = vfes.GetVSize() / equations;
  const double* weak_values = weak.HostRead();
  const double* host_boundary_values = host_boundary_flux.HostRead();
  double maximum_balance_defect = 0.0;
  for (int equation = 0; equation < equations; ++equation)
  {
    double weak_integral = 0.0;
    for (int dof = 0; dof < scalar_vsize; ++dof)
    {
      weak_integral += weak_values[dof + scalar_vsize * equation];
    }
    maximum_balance_defect = std::max(
      maximum_balance_defect,
      std::abs(weak_integral + host_boundary_values[equation]));
  }
  mfem::Vector expected(vfes.GetVSize());
  host.ApplyInverseMass(weak, expected);

  mfem::dgsem::DGSEMOperator production(oper);
  const auto selection = production.BindProductionPolicy(true);
  mfem::Vector actual(vfes.GetVSize());
  actual = -9300.0;
  mfem::Vector production_boundary_flux;
  const auto first = production.AssembleProductionResidual(
    state, actual, true, &production_boundary_flux);
  const auto first_counters = production.GetProductionCacheCounters();
  mfem::Vector repeated(vfes.GetVSize());
  repeated = -9400.0;
  const auto second = production.AssembleProductionResidual(
    state, repeated, true);
  const auto second_counters = production.GetProductionCacheCounters();
  actual -= expected;
  repeated -= expected;
  const double expected_speed = std::max(face_speed, volume_speed);
  mfem::Vector boundary_difference(production_boundary_flux);
  boundary_difference -= host_boundary_flux;
  ctx.Check(selection.supported && first.executed && second.executed &&
              actual.Norml2() < 2.0e-10 &&
              repeated.Norml2() < 2.0e-10 &&
              std::abs(first.max_characteristic_speed - expected_speed) <
                5.0e-13 &&
              std::abs(second.max_characteristic_speed - expected_speed) <
                5.0e-13,
            "DGSEM production mixed inlet/outlet/slip boundaries match host oracle");
  ctx.Check(host_boundary_flux.Normlinf() > 1.0e-12 &&
              boundary_difference.Normlinf() < 5.0e-13 &&
              maximum_balance_defect < 2.0e-10,
            "DGSEM packed physical-boundary flux matches the host oracle and "
            "closes the weak conservation balance");
  ctx.Check(first_counters.topology_builds == 1 &&
              second_counters.topology_builds ==
                first_counters.topology_builds &&
              second_counters.geometry_builds ==
                first_counters.geometry_builds &&
              second_counters.workspace_builds ==
                first_counters.workspace_builds &&
              second_counters.unchanged_cache_fast_paths ==
                first_counters.unchanged_cache_fast_paths + 1,
            "DGSEM mixed-boundary lowering and row-face CSR remain persistent");

  mfem::Vector true_state;
  state.GetTrueDofs(true_state);
  mfem::Vector mult_result(vfes.GetTrueVSize());
  ctx.Check(oper.SupportsPhysicalBoundaryFluxIntegral(),
            "DGSEM advertises exact physical-boundary flux capture");
  oper.Mult(true_state, mult_result);
  ctx.Check(!oper.HasLastPhysicalBoundaryFluxIntegral(),
            "unarmed DGSEM Mult does not create a boundary-flux capture");
  oper.ArmPhysicalBoundaryFluxIntegral();
  oper.Mult(true_state, mult_result);
  mfem::Vector captured_boundary_flux(
    oper.GetLastPhysicalBoundaryFluxIntegral());
  captured_boundary_flux -= host_boundary_flux;
  ctx.Check(oper.HasLastPhysicalBoundaryFluxIntegral() &&
              captured_boundary_flux.Normlinf() < 5.0e-13,
            "armed DGSEM Mult publishes the communicator-wide packed "
            "boundary-flux integral");
  mfem::Vector capture_snapshot(
    oper.GetLastPhysicalBoundaryFluxIntegral());
  oper.Mult(true_state, mult_result);
  mfem::Vector preserved_capture(
    oper.GetLastPhysicalBoundaryFluxIntegral());
  preserved_capture -= capture_snapshot;
  ctx.Check(oper.HasLastPhysicalBoundaryFluxIntegral() &&
              preserved_capture.Normlinf() == 0.0,
            "unarmed DGSEM Mult preserves the last completed one-shot capture");
}

inline void CheckDGSEMProductionEulerBoundaryCatalog(TestContext& ctx)
{
  if (mfem::Mpi::WorldSize() != 1)
  {
    return;
  }

  constexpr int dimension = 2;
  constexpr int equations = dimension + 2;
  constexpr int order = 2;
  constexpr double gamma = 1.4;
  constexpr double boundary_gamma = 1.32;
  constexpr double density_floor = 0.4;
  constexpr double pressure_floor = 0.55;
  constexpr double outlet_pressure = 0.64;
  const std::array<const char*, 7> labels{{
    "supersonic-inlet",
    "supersonic-outlet",
    "subsonic-inlet",
    "subsonic-outlet-pressure",
    "far-field",
    "nonreflecting-outlet",
    "slip-wall"}};

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_FECollection fec(
    order, dimension, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(
    &mesh, &fec, equations, mfem::Ordering::byNODES);
  mfem::Array<int> marker(mesh.bdr_attributes.Max());
  marker = 1;

  mfem::ParGridFunction state(&vfes);
  auto state_function = [=](const mfem::Vector& point,
                            mfem::Vector& value)
  {
    value = EulerConsFromPrim2D(0.78 + 0.04 * point(0),
                               0.11 + 0.02 * point(1),
                               -0.07 + 0.015 * point(0),
                               0.46 + 0.025 * point(1),
                               gamma);
  };
  mfem::VectorFunctionCoefficient coefficient(equations, state_function);
  state.ProjectCoefficient(coefficient);
  const mfem::Vector reference =
    EulerConsFromPrim2D(0.93, -0.08, 0.035, 0.72, boundary_gamma);

  for (int boundary_case = 0;
       boundary_case < static_cast<int>(labels.size());
       ++boundary_case)
  {
    std::unique_ptr<mfem::BoundaryState> boundary;
    switch (boundary_case)
    {
    case 0:
      boundary = std::make_unique<mfem::SupersonicInletState>(reference);
      break;
    case 1:
      boundary = std::make_unique<mfem::SupersonicOutletState>();
      break;
    case 2:
      boundary = std::make_unique<mfem::SubsonicInletState>(
        reference,
        dimension,
        boundary_gamma,
        density_floor,
        pressure_floor);
      break;
    case 3:
      boundary = std::make_unique<mfem::SubsonicOutletPressureState>(
        dimension,
        boundary_gamma,
        outlet_pressure,
        density_floor,
        pressure_floor);
      break;
    case 4:
      boundary = std::make_unique<mfem::FarFieldState>(reference,
                                                       dimension,
                                                       boundary_gamma,
                                                       density_floor,
                                                       pressure_floor);
      break;
    case 5:
      boundary = std::make_unique<mfem::NonReflectingOutletState>(
        reference,
        dimension,
        boundary_gamma,
        density_floor,
        pressure_floor);
      break;
    case 6:
      boundary = std::make_unique<mfem::SlipWallState>(dimension,
                                                       density_floor,
                                                       pressure_floor);
      break;
    default:
      MFEM_ABORT("Invalid DGSEM Euler boundary catalog case.");
    }

    mfem::EulerFlux euler(dimension, gamma);
    mfem::RusanovFluxWithSides numerical_flux(euler);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
      numerical_flux, 2);
    mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
    oper.SetNumericalFlux(&numerical_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMVolumeDissipation(mfem::DGSEMVolumeDissipation::None);
    oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
    oper.SetDGSEMGamma(gamma);
    oper.AddBdrFaceIntegrator(
      std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
        numerical_flux, std::move(boundary), 2),
      marker);

    mfem::dgsem::DGSEMOperator host(oper);
    mfem::Vector weak(vfes.GetVSize());
    mfem::Vector host_boundary_flux;
    const double face_speed =
      host.AssembleSurface(state, weak, &host_boundary_flux);
    const double volume_speed = host.AssembleVolume(state, weak);
    mfem::Vector expected(vfes.GetVSize());
    host.ApplyInverseMass(weak, expected);

    mfem::dgsem::DGSEMOperator production(oper);
    const auto selection = production.BindProductionPolicy(true);
    mfem::Vector actual(vfes.GetVSize());
    actual = -9700.0;
    mfem::Vector packed_boundary_flux;
    const auto result = production.AssembleProductionResidual(
      state, actual, true, &packed_boundary_flux);
    actual -= expected;
    packed_boundary_flux -= host_boundary_flux;
    ctx.Check(selection.supported && result.executed &&
                actual.Norml2() < 2.0e-10 &&
                packed_boundary_flux.Normlinf() < 5.0e-13 &&
                std::abs(result.max_characteristic_speed -
                         std::max(face_speed, volume_speed)) < 5.0e-13,
              std::string("DGSEM packed ") + labels[boundary_case] +
                " boundary matches the shared host boundary semantics");
  }
}

inline void CheckDGSEMProductionStaticSource(TestContext& ctx)
{
  constexpr int dimension = 1;
  constexpr int equations = dimension + 2;
  constexpr double gamma = 1.4;
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian1D(2, 1.0);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial_mesh);
  mfem::DG_FECollection fec(1, dimension, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(
    &mesh, &fec, equations, mfem::Ordering::byNODES);
  mfem::ParGridFunction state(&vfes);
  FillConstantState(EulerConsFromPrim1D(1.0, 0.0, 1.0, gamma), state);

  mfem::EulerFlux euler(dimension, gamma);
  mfem::RusanovFluxWithSides numerical_flux(euler);
  auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
    numerical_flux, 2);
  mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form), true);
  oper.SetNumericalFlux(&numerical_flux);
  oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
  oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
  oper.SetDGSEMVolumeDissipation(mfem::DGSEMVolumeDissipation::LLF);
  oper.SetDGSEMStabilization(mfem::DGSEMStabilization::None);
  oper.SetDGSEMGamma(gamma);
  mfem::Array<int> marker(mesh.bdr_attributes.Max());
  if (marker.Size() > 0)
  {
    marker = 1;
    oper.AddBdrFaceIntegrator(
      std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
        numerical_flux,
        std::make_unique<mfem::SlipWallState>(dimension),
        2),
      marker);
  }

  mfem::Vector input;
  state.GetTrueDofs(input);
  mfem::Vector baseline(vfes.GetTrueVSize());
  oper.Mult(input, baseline);
  mfem::Vector source(vfes.GetTrueVSize());
  source = 0.125;
  oper.SetStaticSource(source);
  mfem::Vector sourced(vfes.GetTrueVSize());
  oper.Mult(input, sourced);
  sourced -= baseline;
  sourced -= source;
  const auto selection = oper.GetDGSEMProductionSelection();
  ctx.Check(selection.supported &&
              selection.llf_volume_active &&
              oper.HasStaticSource() &&
              GlobalL2Norm(sourced) < 1.0e-12,
            "DGSEM strong static source is exact on the selected CPU/CUDA path");
}

inline void RunDGSEMProductionOperatorTests(TestContext& ctx)
{
  RunDGSEMProductionMultStrictRejectionProbe(ctx);
  ScopedEnvironment strict_production_mult(
    "DG_IGR_DGSEM_PRODUCTION_STRICT", "1");
  // This fixture covers runtime-sized packed kernels, including an order
  // beyond the separately compiled fused dispatch.
  ScopedEnvironment fused_residual("DG_IGR_DGSEM_FUSED_RESIDUAL", "0");
  ScopedEnvironment fused_dmma("DG_IGR_DGSEM_FUSED_DMMA", "0");
  CheckDGSEMProductionMultSerialIdentityAndMove(ctx);
  CheckDGSEMProductionMixedEulerBoundaries(ctx);
  CheckDGSEMProductionEulerBoundaryCatalog(ctx);
  CheckDGSEMProductionStaticSource(ctx);
  using mfem::dgsem::DGSEMProductionReason;
  constexpr double gamma = 1.4;
  constexpr double residual_tolerance = 2.0e-10;
  constexpr double constant_tolerance = 3.0e-10;
  const int world_size = mfem::Mpi::WorldSize();
  constexpr int validation_control_cases = 3;
  constexpr int direct_validation_rejection_cases = 21;
  ctx.Check(world_size == 1 || world_size == 2,
            "DGSEM production fixture is bounded to one/two ranks");

  int negative_checks = 0;
  int assemble_rejections = 0;
  int collective_rejections = 0;
  CheckMetadataNegativeGates(ctx, negative_checks);
  CheckSlipWallFloorDependency(ctx, negative_checks);
  CheckUnsupportedDimensionOrderNoMutation(
    ctx, negative_checks, assemble_rejections);
  CheckPolicyFingerprintNegatives(
    ctx, negative_checks, assemble_rejections);
  CheckCollectiveRejectionSemantics(
    ctx,
    negative_checks,
    assemble_rejections,
    collective_rejections);

  double max_residual_difference = 0.0;
  double max_true_residual_difference = 0.0;
  double max_mult_true_residual_difference = 0.0;
  double max_mult_cfl_difference = 0.0;
  double max_constant_residual = 0.0;
  double max_cfl_difference = 0.0;
  double max_mass_conservation = 0.0;
  double max_energy_conservation = 0.0;
  double max_trajectory_state_difference = 0.0;
  double max_trajectory_conservation_drift = 0.0;
  std::int64_t max_topology_builds = 0;
  std::int64_t max_geometry_builds = 0;
  std::int64_t max_sigma_builds = 0;
  std::int64_t max_workspace_builds = 0;
  std::int64_t max_workspace_identity_checks = 0;
  std::int64_t max_workspace_identity_failures = 0;
  std::int64_t max_unchanged_cache_fast_paths = 0;
  std::int64_t max_rebuild_preflight_entries = 0;
  std::int64_t max_state_exchanges = 0;
  std::int64_t max_sigma_exchanges = 0;
  std::int64_t max_mpi_reductions = 0;
  std::int64_t max_selection_reason_reductions = 0;
  std::int64_t max_stage_reason_reductions = 0;
  std::int64_t max_host_syncs = 0;
  std::int64_t max_mult_unchanged_cache_fast_paths = 0;
  std::int64_t max_mult_bridge_storage_reuses = 0;
  int executed_cases = 0;
  int local_face_cases = 0;
  int shared_face_cases = 0;
  int boundary_face_cases = 0;
  int explicit_mapping_cases = 0;
  int invalidation_cases = 0;
  int trajectory_cases = 0;
  int production_mult_cases = 0;
  int production_mult_mapping_cases = 0;
  int production_mult_fallback_cases = 0;
  int production_mult_update_cases = 0;

  for (int dim = 1; dim <= 2; ++dim)
  {
    const int geometry_modes = dim == 2 ? 2 : 1;
    for (int geometry_mode = 0; geometry_mode < geometry_modes;
         ++geometry_mode)
    {
      for (int order : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12})
      {
        ScopedEnvironment high_order_research(
          "DG_IGR_DGSEM_HIGH_ORDER_RESEARCH", nullptr);
        for (int sigma_mode = 0; sigma_mode <= 1; ++sigma_mode)
        {
        std::unique_ptr<mfem::Mesh> serial_mesh;
        if (dim == 1)
        {
          serial_mesh = std::make_unique<mfem::Mesh>(
            mfem::Mesh::MakeCartesian1D(2, 1.0));
        }
        else
        {
          serial_mesh = std::make_unique<mfem::Mesh>(
            mfem::Mesh::MakeCartesian2D(2,
                                        2,
                                        mfem::Element::QUADRILATERAL,
                                        true,
                                        1.0,
                                        1.0));
          if (geometry_mode == 1)
          {
            serial_mesh->Transform(
              [](const mfem::Vector& reference, mfem::Vector& physical)
              {
                physical.SetSize(2);
                physical(0) =
                  1.2 * reference(0) + 0.35 * reference(1);
                physical(1) =
                  -0.2 * reference(0) + 0.9 * reference(1);
              });
          }
        }
        mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
        const int equations = dim + 2;
        mfem::DG_FECollection fec(
          order, dim, mfem::BasisType::GaussLobatto);
        mfem::ParFiniteElementSpace vfes(
          &mesh, &fec, equations, mfem::Ordering::byNODES);
        mfem::ParGridFunction state(&vfes);

        std::unique_ptr<mfem::DG_FECollection> sigma_fec;
        std::unique_ptr<mfem::ParFiniteElementSpace> sigma_fes;
        std::unique_ptr<mfem::ParGridFunction> sigma;
        if (sigma_mode != 0)
        {
          sigma_fec = std::make_unique<mfem::DG_FECollection>(
            1, dim, mfem::BasisType::GaussLobatto);
          sigma_fes = std::make_unique<mfem::ParFiniteElementSpace>(
            &mesh, sigma_fec.get(), 1, mfem::Ordering::byNODES);
          sigma = std::make_unique<mfem::ParGridFunction>(sigma_fes.get());
          mfem::ConstantCoefficient sigma_coefficient(0.125);
          sigma->ProjectCoefficient(sigma_coefficient);
        }

        mfem::EulerFlux euler_flux(dim, gamma);
        std::unique_ptr<mfem::FluxWithIGR> igr_flux;
        const mfem::FluxFunction* physical_flux = &euler_flux;
        if (sigma)
        {
          igr_flux = std::make_unique<mfem::FluxWithIGR>(euler_flux);
          igr_flux->SetSigmaField(sigma.get());
          physical_flux = igr_flux.get();
        }
        mfem::RusanovFluxWithSides numerical_flux(*physical_flux);
        auto form_integrator =
          std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux, 2);
        mfem::DGHyperbolicConservationLaws oper(
          vfes, std::move(form_integrator), true);
        oper.SetNumericalFlux(&numerical_flux);
        oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
        oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
        oper.SetDGSEMVolumeDissipation(
          mfem::DGSEMVolumeDissipation::None);
        oper.SetDGSEMGamma(gamma);

        mfem::Array<int> marker(mesh.bdr_attributes.Max());
        if (marker.Size() > 0)
        {
          marker = 1;
          oper.AddBdrFaceIntegrator(
            std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
              numerical_flux,
              std::make_unique<mfem::SlipWallState>(dim),
              2),
            marker);
        }

        mfem::dgsem::DGSEMOperator host_operator(oper);
        mfem::dgsem::DGSEMOperator production_operator(oper);
        auto selection = production_operator.BindProductionPolicy(true);
        ctx.Check(selection.supported &&
                    selection.reason == DGSEMProductionReason::None,
                  "DGSEM production strict policy accepts p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D sigma=" + std::to_string(sigma_mode));
        if (!selection.supported)
        {
          continue;
        }
        ++executed_cases;
        local_face_cases += selection.has_local_faces ? 1 : 0;
        shared_face_cases += selection.has_shared_faces ? 1 : 0;
        boundary_face_cases += selection.has_boundary_faces ? 1 : 0;

        auto true_to_local = [&](const mfem::Vector& true_vector,
                                 mfem::Vector& local_vector)
        {
          local_vector.SetSize(vfes.GetVSize());
          const mfem::Operator* prolongation = vfes.GetProlongationMatrix();
          if (prolongation)
          {
            prolongation->Mult(true_vector, local_vector);
            ++explicit_mapping_cases;
          }
          else
          {
            local_vector = true_vector;
          }
        };
        auto local_to_true = [&](const mfem::Vector& local_vector,
                                 mfem::Vector& true_vector)
        {
          true_vector.SetSize(vfes.GetTrueVSize());
          const mfem::Operator* prolongation = vfes.GetProlongationMatrix();
          if (prolongation)
          {
            true_vector = 0.0;
            prolongation->MultTranspose(local_vector, true_vector);
          }
          else
          {
            true_vector = local_vector;
          }
        };
        auto host_residual = [&](const mfem::Vector& local_state,
                                 mfem::Vector& local_residual,
                                 double& max_speed,
                                 mfem::Vector* weak_output)
        {
          // The direct host oracle does not own FluxWithIGR's auxiliary
          // neighbor exchange.  Refresh it explicitly after tests mutate and
          // restore sigma so shared faces compare the same current field as
          // the packed production path.
          if (sigma)
          {
            sigma->ExchangeFaceNbrData();
          }
          mfem::Vector weak(vfes.GetVSize());
          const double face_speed =
            host_operator.AssembleSurface(local_state, weak);
          const double volume_speed =
            host_operator.AssembleVolume(local_state, weak);
          local_residual.SetSize(vfes.GetVSize());
          host_operator.ApplyInverseMass(weak, local_residual);
          max_speed = GlobalMax(std::max(face_speed, volume_speed));
          if (weak_output)
          {
            *weak_output = weak;
          }
        };

        mfem::Vector constant(equations);
        if (dim == 1)
        {
          constant = EulerConsFromPrim1D(1.0, 0.0, 1.0, gamma);
        }
        else
        {
          constant = EulerConsFromPrim2D(1.0, 0.0, 0.0, 1.0, gamma);
        }
        FillConstantState(constant, state);
        mfem::Vector constant_true;
        state.GetTrueDofs(constant_true);
        mfem::Vector constant_local;
        true_to_local(constant_true, constant_local);
        mfem::Vector production_constant(vfes.GetVSize());
        production_constant = -47.0;
        const auto constant_result =
          production_operator.AssembleProductionResidual(
            constant_local, production_constant, true);
        ctx.Check(constant_result.executed,
                  "DGSEM production constant stage executes p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D");
        max_constant_residual =
          std::max(max_constant_residual,
                   GlobalL2Norm(production_constant));
        const auto warm_counters =
          production_operator.GetProductionCacheCounters();
        const auto warm_workspace =
          production_operator.GetProductionWorkspaceEvidence();

        auto state_function = [=](const mfem::Vector& point,
                                  mfem::Vector& value)
        {
          const double rho =
            1.0 + 0.03 * std::sin(1.7 * point(0)) +
            (dim == 2 ? 0.02 * std::cos(1.3 * point(1)) : 0.0);
          const double u = 0.18 +
                           (dim == 2 ? 0.015 * point(1) : 0.01 * point(0));
          const double pressure =
            1.0 + 0.02 * point(0) +
            (dim == 2 ? 0.01 * point(1) : 0.0);
          if (dim == 1)
          {
            value = EulerConsFromPrim1D(rho, u, pressure, gamma);
          }
          else
          {
            value = EulerConsFromPrim2D(
              rho, u, -0.09 + 0.02 * point(0), pressure, gamma);
          }
        };
        mfem::VectorFunctionCoefficient state_coefficient(equations,
                                                           state_function);
        state.ProjectCoefficient(state_coefficient);
        mfem::Vector state_true;
        state.GetTrueDofs(state_true);
        mfem::Vector state_local;
        true_to_local(state_true, state_local);

        mfem::Vector host_local;
        mfem::Vector host_weak;
        double host_speed = 0.0;
        host_residual(state_local, host_local, host_speed, &host_weak);
        mfem::Vector production_local(vfes.GetVSize());
        production_local = -31.0;
        const auto production_result =
          production_operator.AssembleProductionResidual(
            state_local, production_local, true);
        ctx.Check(production_result.executed,
                  "DGSEM production nonconstant stage executes p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D sigma=" + std::to_string(sigma_mode));
        mfem::Vector difference(production_local);
        difference -= host_local;
        const double residual_difference = GlobalL2Norm(difference);
        max_residual_difference =
          std::max(max_residual_difference, residual_difference);
        const double cfl_difference =
          std::abs(production_result.max_characteristic_speed - host_speed);
        max_cfl_difference = std::max(max_cfl_difference, cfl_difference);
        ctx.Check(residual_difference < residual_tolerance,
                  "DGSEM production local residual matches host oracle p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D sigma=" + std::to_string(sigma_mode));
        ctx.Check(cfl_difference < 5.0e-13,
                  "DGSEM production CFL max matches host oracle p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D sigma=" + std::to_string(sigma_mode));

        const int component_stride = vfes.GetVSize() / equations;
        double local_mass_sum = 0.0;
        double local_energy_sum = 0.0;
        const double* weak_data = host_weak.HostRead();
        for (int i = 0; i < component_stride; ++i)
        {
          local_mass_sum += weak_data[i];
          local_energy_sum +=
            weak_data[i + component_stride * (dim + 1)];
        }
        const double mass_conservation = std::abs(GlobalSum(local_mass_sum));
        const double energy_conservation =
          std::abs(GlobalSum(local_energy_sum));
        max_mass_conservation =
          std::max(max_mass_conservation, mass_conservation);
        max_energy_conservation =
          std::max(max_energy_conservation, energy_conservation);
        ctx.Check(mass_conservation < residual_tolerance &&
                    energy_conservation < residual_tolerance,
                  "DGSEM production matched weak residual conserves mass/energy p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D sigma=" + std::to_string(sigma_mode));

        mfem::Vector host_true;
        mfem::Vector production_true;
        local_to_true(host_local, host_true);
        local_to_true(production_local, production_true);
        production_true -= host_true;
        max_true_residual_difference =
          std::max(max_true_residual_difference,
                   GlobalL2Norm(production_true));

        mfem::Vector mult_true(vfes.GetTrueVSize());
        mult_true = -6100.0;
        oper.Mult(state_true, mult_true);
        const auto mult_selection = oper.GetDGSEMProductionSelection();
        const auto mult_worker_first =
          oper.GetDGSEMProductionCacheCounters();
        const auto mult_workspace_first =
          oper.GetDGSEMProductionWorkspaceEvidence();
        const auto mult_bridge_first =
          oper.GetDGSEMProductionBridgeEvidence();
        mfem::Vector mult_difference(mult_true);
        mult_difference -= host_true;
        const double mult_residual_difference =
          GlobalL2Norm(mult_difference);
        const double mult_cfl_difference =
          std::abs(oper.GetMaxCharSpeed() - host_speed);
        max_mult_true_residual_difference = std::max(
          max_mult_true_residual_difference, mult_residual_difference);
        max_mult_cfl_difference =
          std::max(max_mult_cfl_difference, mult_cfl_difference);

        mfem::Vector mult_repeated(vfes.GetTrueVSize());
        mult_repeated = -6200.0;
        oper.Mult(state_true, mult_repeated);
        const auto mult_worker_second =
          oper.GetDGSEMProductionCacheCounters();
        const auto mult_workspace_second =
          oper.GetDGSEMProductionWorkspaceEvidence();
        const auto mult_bridge_second =
          oper.GetDGSEMProductionBridgeEvidence();
        mfem::Vector mult_repeated_difference(mult_repeated);
        mult_repeated_difference -= host_true;
        const bool bridge_storage_reused =
          mult_bridge_first.workspace_storage_signature ==
            mult_bridge_second.workspace_storage_signature &&
          mult_bridge_first.x_local_capacity ==
            mult_bridge_second.x_local_capacity &&
          mult_bridge_first.z_local_capacity ==
            mult_bridge_second.z_local_capacity &&
          mult_bridge_first.y_local_capacity ==
            mult_bridge_second.y_local_capacity;
        max_mult_bridge_storage_reuses = std::max<std::int64_t>(
          max_mult_bridge_storage_reuses,
          bridge_storage_reused ? 1 : 0);
        max_mult_unchanged_cache_fast_paths = std::max(
          max_mult_unchanged_cache_fast_paths,
          mult_worker_second.unchanged_cache_fast_paths);
        const bool has_prolongation =
          vfes.GetProlongationMatrix() != nullptr;
        production_mult_mapping_cases += has_prolongation ? 1 : 0;
        ++production_mult_cases;

        ctx.Check(mult_selection.supported && mult_selection.strict &&
                    mult_selection.reason == DGSEMProductionReason::None &&
                    mult_selection.dimension == dim &&
                    mult_selection.order == order &&
                    mult_residual_difference < residual_tolerance &&
                    GlobalL2Norm(mult_repeated_difference) <
                      residual_tolerance &&
                    mult_cfl_difference < 5.0e-13,
                  "DGSEM production strict Mult reaches accepted worker and "
                  "matches true host oracle p" + std::to_string(order) +
                    " " + std::to_string(dim) + "D sigma=" +
                    std::to_string(sigma_mode));
        ctx.Check(mult_worker_first.topology_builds == 1 &&
                    mult_worker_first.geometry_builds == 1 &&
                    mult_worker_first.sigma_metadata_builds == 1 &&
                    mult_worker_first.workspace_builds == 1 &&
                    mult_worker_second.topology_builds ==
                      mult_worker_first.topology_builds &&
                    mult_worker_second.geometry_builds ==
                      mult_worker_first.geometry_builds &&
                    mult_worker_second.sigma_metadata_builds ==
                      mult_worker_first.sigma_metadata_builds &&
                    mult_worker_second.workspace_builds ==
                      mult_worker_first.workspace_builds &&
                    mult_worker_second.unchanged_cache_fast_paths ==
                      mult_worker_first.unchanged_cache_fast_paths + 2 &&
                    mult_worker_second.rebuild_preflight_entries ==
                      mult_worker_first.rebuild_preflight_entries &&
                    mult_workspace_first.ready &&
                    mult_workspace_second.ready &&
                    mult_workspace_first.size_signature ==
                      mult_workspace_second.size_signature &&
                    mult_workspace_first.capacity_signature ==
                      mult_workspace_second.capacity_signature &&
                    mult_workspace_first.storage_signature ==
                      mult_workspace_second.storage_signature &&
                    mult_workspace_second.failures == 0,
                  "DGSEM production repeated Mult reuses accepted cache/workspace p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D sigma=" + std::to_string(sigma_mode));
        ctx.Check(bridge_storage_reused &&
                    mult_bridge_second.operator_ready &&
                    mult_bridge_second.operator_constructions == 1 &&
                    mult_bridge_second.operator_rebinds == 0 &&
                    mult_bridge_second.production_mult_calls == 2 &&
                    mult_bridge_second.fallback_mult_calls == 0 &&
                    mult_bridge_second.y_local_size == vfes.GetVSize() &&
                    (!has_prolongation ||
                     (mult_bridge_second.x_local_size == vfes.GetVSize() &&
                      mult_bridge_second.z_local_size == vfes.GetVSize())),
                  "DGSEM coordinator bridge preserves persistent P/P^T workspaces p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D sigma=" + std::to_string(sigma_mode));

        mfem::Vector repeated(vfes.GetVSize());
        repeated = -19.0;
        const auto repeated_result =
          production_operator.AssembleProductionResidual(
            state_local, repeated, true);
        const auto repeated_counters =
          production_operator.GetProductionCacheCounters();
        const auto repeated_workspace =
          production_operator.GetProductionWorkspaceEvidence();
        ctx.Check(repeated_result.executed &&
                    repeated_counters.topology_builds ==
                      warm_counters.topology_builds &&
                    repeated_counters.geometry_builds ==
                      warm_counters.geometry_builds &&
                    repeated_counters.sigma_metadata_builds ==
                      warm_counters.sigma_metadata_builds &&
                    repeated_counters.workspace_builds ==
                      warm_counters.workspace_builds &&
                    repeated_counters.unchanged_cache_fast_paths ==
                      warm_counters.unchanged_cache_fast_paths + 2 &&
                    repeated_counters.rebuild_preflight_entries ==
                      warm_counters.rebuild_preflight_entries &&
                    warm_workspace.ready && repeated_workspace.ready &&
                    repeated_workspace.size_signature ==
                      warm_workspace.size_signature &&
                    repeated_workspace.capacity_signature ==
                      warm_workspace.capacity_signature &&
                    repeated_workspace.storage_signature ==
                      warm_workspace.storage_signature &&
                    repeated_workspace.checks ==
                      warm_workspace.checks &&
                    repeated_workspace.failures == 0,
                  "DGSEM on-demand evidence confirms persistent workspace "
                  "shape/capacity/storage without hot-path scans p" +
                    std::to_string(order) + " " + std::to_string(dim) +
                    "D sigma=" + std::to_string(sigma_mode));

        mfem::Vector admission_sentinel(vfes.GetVSize());
        admission_sentinel = 7193.25;
        const auto counters_before_valid_admission =
          production_operator.GetProductionCacheCounters();
        const auto valid_admission =
          production_operator.AssembleProductionResidual(
            state_local,
            admission_sentinel,
            true,
            nullptr,
            /*admission_only=*/true);
        const auto counters_after_valid_admission =
          production_operator.GetProductionCacheCounters();
        const bool admission_q_ok =
          sigma ? (valid_admission.admission_q_sigma.enabled &&
                   valid_admission.admission_q_sigma.sample_count > 0 &&
                   valid_admission.admission_q_sigma.nonpositive_count == 0 &&
                   valid_admission.admission_q_sigma.nonfinite_count == 0)
                : !valid_admission.admission_q_sigma.enabled;
        ctx.Check(
          valid_admission.executed &&
            valid_admission.reason == DGSEMProductionReason::None &&
            admission_q_ok &&
            std::abs(valid_admission.max_characteristic_speed -
                     production_result.max_characteristic_speed) < 5.0e-13 &&
            IsSentinel(admission_sentinel, 7193.25) &&
            counters_after_valid_admission.admission_calls ==
              counters_before_valid_admission.admission_calls + 1 &&
            counters_after_valid_admission.stage_calls ==
              counters_before_valid_admission.stage_calls &&
            counters_after_valid_admission.output_scatters ==
              counters_before_valid_admission.output_scatters,
          "DGSEM admission-only speed refresh matches the full packed CFL "
          "maximum without writing a residual p" +
            std::to_string(order) + " " + std::to_string(dim) +
            "D sigma=" + std::to_string(sigma_mode));

        if (world_size == 1 && dim == 2 && geometry_mode == 1 &&
            order == 2 && sigma_mode == 1)
        {
          mfem::Vector host_state_true(state_true);
          mfem::Vector production_state_true(state_true);
          const auto initial_totals =
            ConservedMassEnergy(dim, order, vfes, state_local);
          bool trajectory_executed = true;
          constexpr double trajectory_dt = 1.0e-4;
          for (int step = 0; step < 4; ++step)
          {
            mfem::Vector host_stage_local;
            mfem::Vector production_stage_local;
            true_to_local(host_state_true, host_stage_local);
            true_to_local(production_state_true, production_stage_local);
            mfem::Vector host_stage_residual;
            mfem::Vector production_stage_residual(vfes.GetVSize());
            double trajectory_speed = 0.0;
            host_residual(host_stage_local,
                          host_stage_residual,
                          trajectory_speed,
                          nullptr);
            const auto trajectory_result =
              production_operator.AssembleProductionResidual(
                production_stage_local, production_stage_residual, true);
            trajectory_executed =
              trajectory_executed && trajectory_result.executed;
            mfem::Vector host_stage_true;
            mfem::Vector production_stage_true;
            local_to_true(host_stage_residual, host_stage_true);
            local_to_true(production_stage_residual, production_stage_true);
            host_state_true.Add(trajectory_dt, host_stage_true);
            production_state_true.Add(trajectory_dt, production_stage_true);
          }
          mfem::Vector host_final_local;
          mfem::Vector production_final_local;
          true_to_local(host_state_true, host_final_local);
          true_to_local(production_state_true, production_final_local);
          mfem::Vector trajectory_difference(production_state_true);
          trajectory_difference -= host_state_true;
          const double state_difference =
            GlobalL2Norm(trajectory_difference);
          max_trajectory_state_difference =
            std::max(max_trajectory_state_difference, state_difference);
          const auto host_totals =
            ConservedMassEnergy(dim, order, vfes, host_final_local);
          const auto production_totals =
            ConservedMassEnergy(dim, order, vfes, production_final_local);
          double trajectory_drift = 0.0;
          for (int component = 0; component < 2; ++component)
          {
            trajectory_drift = std::max(
              trajectory_drift,
              std::abs(host_totals[component] - initial_totals[component]));
            trajectory_drift = std::max(
              trajectory_drift,
              std::abs(production_totals[component] -
                       initial_totals[component]));
            trajectory_drift = std::max(
              trajectory_drift,
              std::abs(production_totals[component] -
                       host_totals[component]));
          }
          max_trajectory_conservation_drift = std::max(
            max_trajectory_conservation_drift, trajectory_drift);
          ctx.Check(trajectory_executed &&
                      state_difference < residual_tolerance &&
                      trajectory_drift < residual_tolerance &&
                      EulerStateAdmissible(
                        dim, gamma, vfes, host_final_local) &&
                      EulerStateAdmissible(
                        dim, gamma, vfes, production_final_local),
                    "DGSEM production short trajectory preserves state, "
                    "admissibility, and conservation parity");
          ++trajectory_cases;
        }

        if (dim == 1 && order == 1 && sigma_mode == 0)
        {
          mfem::Array<int> first_vdofs;
          vfes.GetElementVDofs(0, first_vdofs);
          mfem::Vector invalid_state(state_local);
          invalid_state.HostWrite()[AbsoluteDof(first_vdofs[0])] = 0.0;
          mfem::Vector sentinel(vfes.GetVSize());
          sentinel = 1234.5;
          const auto invalid_density =
            production_operator.AssembleProductionResidual(
              invalid_state, sentinel, true);
          ctx.Check(!invalid_density.executed &&
                      invalid_density.reason ==
                        DGSEMProductionReason::InvalidDensity &&
                      IsSentinel(sentinel, 1234.5),
                    "DGSEM production invalid density fails before output write");
          ++negative_checks;
          ++assemble_rejections;

          mfem::Vector invalid_pressure_state(state_local);
          const int local_dof = vfes.GetFE(0)->GetDof();
          double* invalid_pressure_data =
            invalid_pressure_state.HostWrite();
          invalid_pressure_data[AbsoluteDof(first_vdofs[0])] = 1.0;
          invalid_pressure_data[
            AbsoluteDof(first_vdofs[(1 + dim) * local_dof])] = 0.0;
          sentinel = 1234.5;
          const auto invalid_pressure =
            production_operator.AssembleProductionResidual(
              invalid_pressure_state, sentinel, true);
          ctx.Check(!invalid_pressure.executed &&
                      invalid_pressure.reason ==
                        DGSEMProductionReason::InvalidPressure &&
                      IsSentinel(sentinel, 1234.5),
                    "DGSEM production invalid pressure fails before output "
                    "write reason=" +
                      std::string(mfem::dgsem::DGSEMProductionReasonName(
                        invalid_pressure.reason)) +
                      " executed=" +
                      std::to_string(invalid_pressure.executed ? 1 : 0) +
                      " unchanged=" +
                      std::to_string(IsSentinel(sentinel, 1234.5) ? 1 : 0));
          ++negative_checks;
          ++assemble_rejections;

          oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Chandrashekar);
          const auto unsupported_volume =
            production_operator.BindProductionPolicy(true);
          sentinel = 1234.5;
          const auto unsupported_result =
            production_operator.AssembleProductionResidual(
              state_local, sentinel, true);
          ctx.Check(!unsupported_volume.supported &&
                      unsupported_volume.reason ==
                        DGSEMProductionReason::UnsupportedVolumeFlux &&
                      !unsupported_result.executed &&
                      IsSentinel(sentinel, 1234.5),
                    "DGSEM production unsupported volume fails closed");
          ++negative_checks;
          ++assemble_rejections;
          oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);

          oper.SetDGSEMVolumeDissipation(
            mfem::DGSEMVolumeDissipation::LLF);
          mfem::dgsem::DGSEMOperator llf_production_operator(oper);
          const auto llf_selection =
            llf_production_operator.BindProductionPolicy(true);
          mfem::Vector llf_host;
          double llf_host_speed = 0.0;
          host_residual(state_local, llf_host, llf_host_speed, nullptr);
          mfem::Vector llf_packed(vfes.GetVSize());
          llf_packed = 1234.5;
          const auto llf_result =
            llf_production_operator.AssembleProductionResidual(
              state_local, llf_packed, true);
          llf_packed -= llf_host;
          ctx.Check(llf_selection.supported &&
                      llf_selection.llf_volume_active &&
                      llf_result.executed &&
                      GlobalL2Norm(llf_packed) < residual_tolerance &&
                      std::abs(llf_result.max_characteristic_speed -
                               llf_host_speed) < 5.0e-13,
                    "typed CPU/CUDA DGSEM LLF selection matches its host oracle");
          oper.SetDGSEMVolumeDissipation(
            mfem::DGSEMVolumeDissipation::None);

          mfem::DGSEMSurfaceFlux unsupported_surface(*physical_flux);
          unsupported_surface.SetSurfaceFluxType(
            mfem::DGSEMSurfaceFluxType::DiffusiveRusanov);
          oper.SetNumericalFlux(&unsupported_surface);
          const auto unsupported_face =
            production_operator.BindProductionPolicy(true);
          sentinel = 1234.5;
          const auto unsupported_face_result =
            production_operator.AssembleProductionResidual(
              state_local, sentinel, true);
          ctx.Check(!unsupported_face.supported &&
                      unsupported_face.reason ==
                        DGSEMProductionReason::UnsupportedSurfaceFlux &&
                      !unsupported_face_result.executed &&
                      IsSentinel(sentinel, 1234.5),
                    "DGSEM production unsupported surface fails closed");
          ++negative_checks;
          ++assemble_rejections;
          oper.SetNumericalFlux(&numerical_flux);

          auto unsupported_form =
            std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux, 2);
          mfem::DGHyperbolicConservationLaws unsupported_boundary_operator(
            vfes, std::move(unsupported_form), true);
          unsupported_boundary_operator.SetNumericalFlux(&numerical_flux);
          unsupported_boundary_operator.SetVolumeMode(
            mfem::VolumeMode::DGSEMFluxDiff);
          unsupported_boundary_operator.SetDGSEMVolumeFlux(
            mfem::DGSEMVolumeFlux::Central);
          unsupported_boundary_operator.SetDGSEMVolumeDissipation(
            mfem::DGSEMVolumeDissipation::None);
          if (marker.Size() > 0)
          {
            unsupported_boundary_operator.AddBdrFaceIntegrator(
              std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
                numerical_flux,
                std::make_unique<UnsupportedBoundaryState>(),
                2),
              marker);
          }
          mfem::dgsem::DGSEMOperator unsupported_boundary_path(
            unsupported_boundary_operator);
          const auto unsupported_boundary =
            unsupported_boundary_path.BindProductionPolicy(true);
          sentinel = 1234.5;
          const auto unsupported_boundary_result =
            unsupported_boundary_path.AssembleProductionResidual(
              state_local, sentinel, true);
          ctx.Check(!unsupported_boundary.supported &&
                      unsupported_boundary.reason ==
                        DGSEMProductionReason::UnsupportedBoundary &&
                      !unsupported_boundary_result.executed &&
                      IsSentinel(sentinel, 1234.5),
                    "DGSEM production unsupported boundary fails closed");
          ++negative_checks;
          ++assemble_rejections;

          oper.SetDGSEMGamma(1.0);
          const auto invalid_gamma_policy =
            production_operator.BindProductionPolicy(true);
          sentinel = 1234.5;
          const auto invalid_gamma_result =
            production_operator.AssembleProductionResidual(
              state_local, sentinel, true);
          ctx.Check(!invalid_gamma_policy.supported &&
                      invalid_gamma_policy.reason ==
                        DGSEMProductionReason::InvalidGamma &&
                      !invalid_gamma_result.executed &&
                      IsSentinel(sentinel, 1234.5),
                    "DGSEM production invalid gamma fails closed");
          ++negative_checks;
          ++assemble_rejections;
          oper.SetDGSEMGamma(gamma);

          {
            ScopedEnvironment ordinary_fallback(
              "DG_IGR_DGSEM_PRODUCTION_STRICT", nullptr);
            oper.SetDGSEMVolumeFlux(
              mfem::DGSEMVolumeFlux::Chandrashekar);
            mfem::Vector fallback_host_local;
            double fallback_host_speed = 0.0;
            host_residual(state_local,
                          fallback_host_local,
                          fallback_host_speed,
                          nullptr);
            mfem::Vector fallback_host_true;
            local_to_true(fallback_host_local, fallback_host_true);
            mfem::Vector fallback_true(vfes.GetTrueVSize());
            fallback_true = -6300.0;
            oper.Mult(state_true, fallback_true);
            mfem::Vector fallback_difference(fallback_true);
            fallback_difference -= fallback_host_true;
            const auto fallback_selection =
              oper.GetDGSEMProductionSelection();
            const auto fallback_bridge =
              oper.GetDGSEMProductionBridgeEvidence();
            ctx.Check(!fallback_selection.supported &&
                        !fallback_selection.strict &&
                        fallback_selection.reason ==
                          DGSEMProductionReason::UnsupportedVolumeFlux &&
                        GlobalL2Norm(fallback_difference) <
                          residual_tolerance &&
                        std::abs(oper.GetMaxCharSpeed() -
                                 fallback_host_speed) < 5.0e-13 &&
                        fallback_bridge.production_mult_calls == 2 &&
                        fallback_bridge.fallback_mult_calls == 1,
                      "DGSEM production Mult ordinary fallback is exact and reason-coded");
            ++production_mult_fallback_cases;
            oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
          }
        }

        if (dim == 1 && order == 1 && sigma_mode == 1)
        {
          mfem::Vector sigma_backup(*sigma);
          sigma->HostWrite()[0] = std::numeric_limits<double>::quiet_NaN();
          mfem::Vector sentinel(vfes.GetVSize());
          sentinel = 4321.5;
          const auto invalid_sigma =
            production_operator.AssembleProductionResidual(
              state_local, sentinel, true);
          ctx.Check(!invalid_sigma.executed &&
                      invalid_sigma.reason ==
                        DGSEMProductionReason::InvalidSigma &&
                      IsSentinel(sentinel, 4321.5),
                    "DGSEM production invalid sigma fails before output write");
          ++negative_checks;
          ++assemble_rejections;
          *sigma = sigma_backup;

          *sigma = -10.0;
          mfem::Vector admission_sentinel(vfes.GetVSize());
          admission_sentinel = 8765.25;
          const auto counters_before_admission =
            production_operator.GetProductionCacheCounters();
          const auto q_admission =
            production_operator.AssembleProductionResidual(
              state_local,
              admission_sentinel,
              true,
              nullptr,
              /*admission_only=*/true);
          const auto counters_after_admission =
            production_operator.GetProductionCacheCounters();
          const std::uint64_t expected_owned_q_samples =
            static_cast<std::uint64_t>(GlobalSum(
              static_cast<double>(vfes.GetNE() *
                                  vfes.GetFE(0)->GetDof())));
          ctx.Check(q_admission.executed &&
                      q_admission.reason == DGSEMProductionReason::None &&
                      q_admission.admission_q_sigma.enabled &&
                      q_admission.admission_q_sigma.sample_count >
                        expected_owned_q_samples &&
                      q_admission.admission_q_sigma.nonpositive_count ==
                        q_admission.admission_q_sigma.sample_count &&
                      q_admission.admission_q_sigma.nonfinite_count == 0 &&
                      q_admission.admission_q_sigma.minimum_raw_q_sigma < 0.0 &&
                      q_admission.max_characteristic_speed == 0.0 &&
                      IsSentinel(admission_sentinel, 8765.25) &&
                      counters_after_admission.admission_calls ==
                        counters_before_admission.admission_calls + 1 &&
                      counters_after_admission.stage_calls ==
                        counters_before_admission.stage_calls &&
                      counters_after_admission.output_scatters ==
                        counters_before_admission.output_scatters,
                    "packed DGSEM admission-only scan detects fresh q_sigma "
                    "violations without evaluating a characteristic radius "
                    "or writing a residual");

          const mfem::IGRFrozenSigmaTelemetry q_before =
            oper.GetGlobalFrozenSigmaTelemetry();
          mfem::Vector negative_q_output(vfes.GetVSize());
          const auto negative_q_result =
            production_operator.AssembleProductionResidual(
              state_local, negative_q_output, true);
          const mfem::IGRFrozenSigmaTelemetry q_after =
            oper.GetGlobalFrozenSigmaTelemetry();
          const std::uint64_t new_q_samples =
            q_after.sample_count - q_before.sample_count;
          ctx.Check(negative_q_result.executed && q_after.enabled &&
                      new_q_samples > expected_owned_q_samples &&
                      q_after.nonpositive_count - q_before.nonpositive_count ==
                        new_q_samples &&
                      q_after.nonfinite_count == q_before.nonfinite_count &&
                      q_after.minimum_raw_q_sigma < 0.0,
                    "packed DGSEM records owned and two-sided face q_sigma "
                    "samples for stage admission");
          *sigma = sigma_backup;

          igr_flux->SetFrozenSigmaSpectralSpeed(true);
          oper.SetDGSEMVolumeDissipation(
            mfem::DGSEMVolumeDissipation::LLF);
          mfem::dgsem::DGSEMOperator frozen_spectral_operator(oper);
          const auto frozen_spectral_selection =
            frozen_spectral_operator.BindProductionPolicy(true);
          mfem::Vector frozen_spectral_host;
          mfem::Vector frozen_spectral_packed(vfes.GetVSize());
          double frozen_spectral_host_speed = 0.0;
          host_residual(state_local,
                        frozen_spectral_host,
                        frozen_spectral_host_speed,
                        nullptr);
          const auto frozen_spectral_result =
            frozen_spectral_operator.AssembleProductionResidual(
              state_local, frozen_spectral_packed, true);
          const mfem::IGRFrozenSigmaTelemetry endpoint_admission =
            oper.InspectFrozenSigmaStateForAdmission(state_true);
          const double endpoint_admission_speed = oper.GetMaxCharSpeed();
          frozen_spectral_packed -= frozen_spectral_host;
          ctx.Check(
              frozen_spectral_selection.supported &&
              frozen_spectral_selection.strict &&
              frozen_spectral_selection.frozen_sigma_spectral_speed &&
              frozen_spectral_selection.llf_volume_active &&
              frozen_spectral_result.executed &&
              endpoint_admission.enabled &&
              endpoint_admission.sample_count > 0 &&
              endpoint_admission.nonpositive_count == 0 &&
              endpoint_admission.nonfinite_count == 0 &&
              frozen_spectral_result.max_characteristic_speed >
                production_result.max_characteristic_speed &&
              std::abs(frozen_spectral_result.max_characteristic_speed -
                       frozen_spectral_host_speed) < 5.0e-13 &&
              std::abs(endpoint_admission_speed -
                       frozen_spectral_result.max_characteristic_speed) <
                5.0e-13 &&
              GlobalL2Norm(frozen_spectral_packed) < residual_tolerance,
            "typed packed DGSEM frozen-sigma endpoint admission publishes "
            "the same CFL maximum used by Rusanov/LLF and the host oracle "
            "supported=" +
              std::to_string(frozen_spectral_selection.supported ? 1 : 0) +
              " strict=" +
              std::to_string(frozen_spectral_selection.strict ? 1 : 0) +
              " spectral=" +
              std::to_string(
                frozen_spectral_selection.frozen_sigma_spectral_speed ? 1 : 0) +
              " llf=" +
              std::to_string(
                frozen_spectral_selection.llf_volume_active ? 1 : 0) +
              " executed=" +
              std::to_string(frozen_spectral_result.executed ? 1 : 0) +
              " speed=" +
              FormatSci(frozen_spectral_result.max_characteristic_speed) +
              " admitted_speed=" + FormatSci(endpoint_admission_speed) +
              " baseline=" +
              FormatSci(production_result.max_characteristic_speed) +
              " host_speed=" + FormatSci(frozen_spectral_host_speed) +
              " residual=" + FormatSci(GlobalL2Norm(frozen_spectral_packed)));
          oper.SetDGSEMVolumeDissipation(
            mfem::DGSEMVolumeDissipation::None);
          igr_flux->SetFrozenSigmaSpectralSpeed(false);

          igr_flux->SetSigmaAwareRusanov(true);
          const auto unsupported_sigma_policy =
            production_operator.BindProductionPolicy(true);
          sentinel = 4321.5;
          const auto unsupported_sigma_result =
            production_operator.AssembleProductionResidual(
              state_local, sentinel, true);
          ctx.Check(!unsupported_sigma_policy.supported &&
                      unsupported_sigma_policy.reason ==
                        DGSEMProductionReason::UnsupportedSigmaSpeedPolicy &&
                      !unsupported_sigma_result.executed &&
                      IsSentinel(sentinel, 4321.5),
                    "DGSEM production sigma speed policy fails closed");
          ++negative_checks;
          ++assemble_rejections;
          igr_flux->SetSigmaAwareRusanov(false);

          const double floor_backup = igr_flux->GetVelocityRhoFloor();
          igr_flux->SetVelocityRhoFloor(0.0);
          const auto invalid_floor_policy =
            production_operator.BindProductionPolicy(true);
          sentinel = 4321.5;
          const auto invalid_floor_result =
            production_operator.AssembleProductionResidual(
              state_local, sentinel, true);
          ctx.Check(!invalid_floor_policy.supported &&
                      invalid_floor_policy.reason ==
                        DGSEMProductionReason::InvalidDensityFloor &&
                      !invalid_floor_result.executed &&
                      IsSentinel(sentinel, 4321.5),
                    "DGSEM production invalid density floor fails closed");
          ++negative_checks;
          ++assemble_rejections;
          igr_flux->SetVelocityRhoFloor(floor_backup);

          mfem::ParFiniteElementSpace invalid_sigma_fes(
            &mesh, sigma_fec.get(), 2, mfem::Ordering::byNODES);
          mfem::ParGridFunction invalid_sigma_field(&invalid_sigma_fes);
          invalid_sigma_field = 0.0;
          igr_flux->SetSigmaField(&invalid_sigma_field);
          const auto unsupported_sigma_space =
            production_operator.BindProductionPolicy(true);
          sentinel = 4321.5;
          const auto unsupported_sigma_space_result =
            production_operator.AssembleProductionResidual(
              state_local, sentinel, true);
          ctx.Check(!unsupported_sigma_space.supported &&
                      unsupported_sigma_space.reason ==
                        DGSEMProductionReason::UnsupportedSigma &&
                      !unsupported_sigma_space_result.executed &&
                      IsSentinel(sentinel, 4321.5),
                    "DGSEM production invalid sigma space fails closed");
          ++negative_checks;
          ++assemble_rejections;
          igr_flux->SetSigmaField(sigma.get());
        }

        const auto counters =
          production_operator.GetProductionCacheCounters();
        max_topology_builds =
          std::max(max_topology_builds, counters.topology_builds);
        max_geometry_builds =
          std::max(max_geometry_builds, counters.geometry_builds);
        max_sigma_builds =
          std::max(max_sigma_builds, counters.sigma_metadata_builds);
        max_workspace_builds =
          std::max(max_workspace_builds, counters.workspace_builds);
        max_workspace_identity_checks = std::max(
          max_workspace_identity_checks,
          counters.workspace_identity_checks);
        max_workspace_identity_failures = std::max(
          max_workspace_identity_failures,
          counters.workspace_identity_failures);
        max_unchanged_cache_fast_paths = std::max(
          max_unchanged_cache_fast_paths,
          counters.unchanged_cache_fast_paths);
        max_rebuild_preflight_entries = std::max(
          max_rebuild_preflight_entries,
          counters.rebuild_preflight_entries);
        max_state_exchanges =
          std::max(max_state_exchanges, counters.state_exchanges);
        max_sigma_exchanges =
          std::max(max_sigma_exchanges, counters.sigma_exchanges);
        max_mpi_reductions =
          std::max(max_mpi_reductions, counters.mpi_reductions);
        max_selection_reason_reductions = std::max(
          max_selection_reason_reductions,
          counters.selection_reason_reductions);
        max_stage_reason_reductions = std::max(
          max_stage_reason_reductions,
          counters.stage_reason_reductions);
        max_host_syncs =
          std::max(max_host_syncs, counters.host_synchronizations);

        if (dim == 2 && geometry_mode == 1 && order == 2 && sigma_mode == 1)
        {
          production_operator.InvalidateProductionCache(
            mfem::dgsem::DGSEMProductionInvalidation::PolicyOnly);
          const auto rebound = production_operator.BindProductionPolicy(true);
          const auto policy_counters =
            production_operator.GetProductionCacheCounters();
          ctx.Check(rebound.supported &&
                      policy_counters.policy_binds ==
                        counters.policy_binds + 1 &&
                      policy_counters.topology_builds ==
                        counters.topology_builds &&
                      policy_counters.geometry_builds ==
                        counters.geometry_builds &&
                      policy_counters.sigma_metadata_builds ==
                        counters.sigma_metadata_builds &&
                      policy_counters.workspace_builds ==
                        counters.workspace_builds,
                    "DGSEM production policy-only invalidation rebuilds only policy");

          production_operator.InvalidateProductionCache(
            mfem::dgsem::DGSEMProductionInvalidation::SigmaMetadata);
          const auto sigma_rebound =
            production_operator.BindProductionPolicy(true);
          const auto sigma_counters =
            production_operator.GetProductionCacheCounters();
          ctx.Check(sigma_rebound.supported &&
                      sigma_counters.topology_builds ==
                        policy_counters.topology_builds &&
                      sigma_counters.geometry_builds ==
                        policy_counters.geometry_builds &&
                      sigma_counters.sigma_metadata_builds ==
                        policy_counters.sigma_metadata_builds + 1 &&
                      sigma_counters.workspace_builds ==
                        policy_counters.workspace_builds,
                    "DGSEM production sigma invalidation rebuilds only interpolation metadata");

          production_operator.InvalidateProductionCache(
            mfem::dgsem::DGSEMProductionInvalidation::Topology);
          mfem::Vector after_update(vfes.GetVSize());
          after_update = -777.0;
          const auto update_result =
            production_operator.AssembleProductionResidual(
              state_local, after_update, true);
          const auto update_counters =
            production_operator.GetProductionCacheCounters();
          ctx.Check(update_result.executed &&
                      update_counters.update_invalidations ==
                        sigma_counters.update_invalidations + 1 &&
                      update_counters.topology_builds ==
                        sigma_counters.topology_builds + 1 &&
                      update_counters.geometry_builds ==
                        sigma_counters.geometry_builds + 1 &&
                      update_counters.sigma_metadata_builds ==
                        sigma_counters.sigma_metadata_builds + 1 &&
                      update_counters.workspace_builds ==
                        sigma_counters.workspace_builds + 1,
                    "DGSEM production Update invalidation rebuilds all dependent layers");
          ++invalidation_cases;

          const auto mult_before_update =
            oper.GetDGSEMProductionCacheCounters();
          const auto bridge_before_update =
            oper.GetDGSEMProductionBridgeEvidence();
          oper.Update();
          mfem::Vector mult_after_update(vfes.GetTrueVSize());
          mult_after_update = -6400.0;
          oper.Mult(state_true, mult_after_update);
          mfem::Vector mult_update_difference(mult_after_update);
          mult_update_difference -= host_true;
          const auto mult_update_selection =
            oper.GetDGSEMProductionSelection();
          const auto mult_after_update_counters =
            oper.GetDGSEMProductionCacheCounters();
          const auto bridge_after_update =
            oper.GetDGSEMProductionBridgeEvidence();
          ctx.Check(mult_update_selection.supported &&
                      mult_update_selection.strict &&
                      mult_update_selection.reason ==
                        DGSEMProductionReason::None &&
                      GlobalL2Norm(mult_update_difference) <
                        residual_tolerance &&
                      std::abs(oper.GetMaxCharSpeed() - host_speed) <
                        5.0e-13 &&
                      mult_after_update_counters.update_invalidations ==
                        mult_before_update.update_invalidations + 1 &&
                      mult_after_update_counters.topology_builds ==
                        mult_before_update.topology_builds + 1 &&
                      mult_after_update_counters.geometry_builds ==
                        mult_before_update.geometry_builds + 1 &&
                      mult_after_update_counters.sigma_metadata_builds ==
                        mult_before_update.sigma_metadata_builds + 1 &&
                      mult_after_update_counters.workspace_builds ==
                        mult_before_update.workspace_builds + 1 &&
                      bridge_after_update.operator_constructions ==
                        bridge_before_update.operator_constructions &&
                      bridge_after_update.operator_rebinds == 0 &&
                      bridge_after_update.production_mult_calls ==
                        bridge_before_update.production_mult_calls + 1,
                    "DGSEM production parent Update invalidates and safely rebuilds Mult state");
          ++production_mult_update_cases;
        }
        }
      }
    }
  }

  ctx.Check(executed_cases == 66,
            "DGSEM production executes affine/sheared p1-p10/p12 1D/2D "
            "zero/frozen-sigma cases");
  ctx.Check(production_mult_cases == 66,
            "DGSEM production Mult executes all focused p1-p10/p12 1D/2D cases");
  ctx.Check(production_mult_mapping_cases > 0,
            "DGSEM production Mult executes true/local P and P^T mapping");
  ctx.Check(max_mult_true_residual_difference < residual_tolerance &&
              max_mult_cfl_difference < 5.0e-13,
            "DGSEM production Mult aggregate true residual and CFL parity");
  ctx.Check(max_mult_unchanged_cache_fast_paths > 0 &&
              max_mult_bridge_storage_reuses == 1,
            "DGSEM production repeated Mult reuses persistent worker/bridge storage");
  ctx.Check(production_mult_fallback_cases == 1,
            "DGSEM production Mult exercises one exact ordinary fallback reason");
  ctx.Check(production_mult_update_cases == 1,
            "DGSEM production Mult exercises one exact parent Update rebuild");
  ctx.Check(local_face_cases > 0 || world_size == 2,
            "DGSEM production records local-face ownership coverage");
  ctx.Check(shared_face_cases > 0 || world_size == 1,
            "DGSEM production records shared-face ownership coverage");
  ctx.Check(boundary_face_cases == executed_cases,
            "DGSEM production records boundary-face ownership coverage");
  ctx.Check(max_residual_difference < residual_tolerance,
            "DGSEM production aggregate local residual parity");
  ctx.Check(max_true_residual_difference < residual_tolerance,
            "DGSEM production aggregate P/P^T residual parity");
  ctx.Check(max_constant_residual < constant_tolerance,
            "DGSEM production aggregate constant cancellation");
  ctx.Check(max_cfl_difference < 5.0e-13,
            "DGSEM production aggregate CFL maximum parity");
  ctx.Check(max_mass_conservation < residual_tolerance &&
              max_energy_conservation < residual_tolerance,
            "DGSEM production aggregate mass/energy conservation");
  ctx.Check(max_topology_builds == 1 && max_geometry_builds == 1 &&
              max_sigma_builds == 1 && max_workspace_builds == 1,
            "DGSEM production cache layers build once before invalidation");
  ctx.Check(max_workspace_identity_checks == 0 &&
              max_workspace_identity_failures == 0,
            "DGSEM production keeps workspace evidence out of the hot path");
  ctx.Check(max_unchanged_cache_fast_paths > 0 &&
              max_rebuild_preflight_entries > 0,
            "DGSEM production accounts unchanged-cache versus rebuild paths");
  ctx.Check(max_host_syncs > 0,
            "DGSEM production explicitly accounts host synchronizations");
  if (world_size == 2)
  {
    ctx.Check(max_state_exchanges > 0 && max_mpi_reductions > 0,
              "DGSEM production accounts CPU MPI exchange and CFL reduction");
    ctx.Check(max_sigma_exchanges > 0,
              "DGSEM production accounts frozen-sigma MPI exchange");
    ctx.Check(max_selection_reason_reductions > 0 &&
                max_stage_reason_reductions > 0,
              "DGSEM production accounts collective selection/stage reasons");
  }
  const int expected_assemble_rejections =
    world_size == 1 ? 20 : 14;
  ctx.Check(assemble_rejections == expected_assemble_rejections,
            "DGSEM production counts exact no-write assembly rejections");
  ctx.Check(negative_checks == validation_control_cases +
              direct_validation_rejection_cases +
              expected_assemble_rejections,
            "DGSEM production counts exact controls and rejection cases");
  ctx.Check(collective_rejections == (world_size == 2 ? 4 : 0),
            "DGSEM production counts exact collective rejection cases");
  ctx.Check(explicit_mapping_cases > 0,
            "DGSEM production fixture explicitly executes P and P^T mapping");
  ctx.Check(invalidation_cases == 1,
            "DGSEM production fixture exercises one exact invalidation chain");
  if (world_size == 1)
  {
    ctx.Check(trajectory_cases == 1 &&
                max_trajectory_state_difference < residual_tolerance &&
                max_trajectory_conservation_drift < residual_tolerance,
              "DGSEM production fixture executes matched short trajectory");
  }

  ctx.AddParity(max_residual_difference + max_true_residual_difference +
                max_constant_residual + max_cfl_difference +
                max_mult_true_residual_difference +
                max_mult_cfl_difference);
  ctx.LogMetric("DGSEM production aggregate: residual=" +
                FormatSci(max_residual_difference) +
                " true_residual=" + FormatSci(max_true_residual_difference) +
                " constant=" + FormatSci(max_constant_residual) +
                " cfl=" + FormatSci(max_cfl_difference) +
                " mass_conservation=" + FormatSci(max_mass_conservation) +
                " energy_conservation=" + FormatSci(max_energy_conservation) +
                " trajectory_state=" + FormatSci(max_trajectory_state_difference) +
                " trajectory_conservation=" +
                FormatSci(max_trajectory_conservation_drift) +
                " mult_true_residual=" +
                FormatSci(max_mult_true_residual_difference) +
                " mult_cfl=" + FormatSci(max_mult_cfl_difference) +
                " negative_checks=" + std::to_string(negative_checks));

  const char* output_directory =
    ::igrlab::GetEnv("DG_IGR_DGSEM_PRODUCTION_OUTPUT_DIR");
  if (output_directory && mfem::Mpi::Root())
  {
    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    std::ofstream out(std::filesystem::path(output_directory) /
                      "dgsem_production_operator.json");
    out.setf(std::ios::scientific);
    out.precision(17);
    out << "{\n"
        << "  \"schema\": \"dgsem-production-operator-v2\",\n"
        << "  \"dimensions\": [1, 2],\n"
        << "  \"orders\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],\n"
        << "  \"sigma_modes\": [\"zero\", \"stage-frozen\"],\n"
        << "  \"ranks\": " << world_size << ",\n"
        << "  \"executed_cases\": " << executed_cases << ",\n"
        << "  \"max_residual_difference\": "
        << max_residual_difference << ",\n"
        << "  \"max_true_residual_difference\": "
        << max_true_residual_difference << ",\n"
        << "  \"max_constant_residual\": "
        << max_constant_residual << ",\n"
        << "  \"production_mult_cases\": " << production_mult_cases
        << ",\n"
        << "  \"production_mult_mapping_cases\": "
        << production_mult_mapping_cases << ",\n"
        << "  \"production_mult_fallback_cases\": "
        << production_mult_fallback_cases << ",\n"
        << "  \"production_mult_update_cases\": "
        << production_mult_update_cases << ",\n"
        << "  \"max_mult_true_residual_difference\": "
        << max_mult_true_residual_difference << ",\n"
        << "  \"max_mult_cfl_difference\": " << max_mult_cfl_difference
        << ",\n"
        << "  \"max_cfl_difference\": " << max_cfl_difference << ",\n"
        << "  \"max_mass_conservation\": "
        << max_mass_conservation << ",\n"
        << "  \"max_energy_conservation\": "
        << max_energy_conservation << ",\n"
        << "  \"max_trajectory_state_difference\": "
        << max_trajectory_state_difference << ",\n"
        << "  \"max_trajectory_conservation_drift\": "
        << max_trajectory_conservation_drift << ",\n"
        << "  \"trajectory_cases\": " << trajectory_cases << ",\n"
        << "  \"negative_checks\": " << negative_checks << ",\n"
        << "  \"validation_control_cases\": "
        << validation_control_cases << ",\n"
        << "  \"direct_validation_rejection_cases\": "
        << direct_validation_rejection_cases << ",\n"
        << "  \"assemble_no_write_rejection_cases\": "
        << assemble_rejections << ",\n"
        << "  \"collective_rejection_cases\": "
        << collective_rejections << ",\n"
        << "  \"invalidation_cases\": " << invalidation_cases << ",\n"
        << "  \"max_topology_builds\": " << max_topology_builds << ",\n"
        << "  \"max_geometry_builds\": " << max_geometry_builds << ",\n"
        << "  \"max_sigma_metadata_builds\": " << max_sigma_builds
        << ",\n"
        << "  \"max_workspace_builds\": " << max_workspace_builds << ",\n"
        << "  \"max_workspace_identity_checks\": "
        << max_workspace_identity_checks << ",\n"
        << "  \"max_workspace_identity_failures\": "
        << max_workspace_identity_failures << ",\n"
        << "  \"max_unchanged_cache_fast_paths\": "
        << max_unchanged_cache_fast_paths << ",\n"
        << "  \"max_rebuild_preflight_entries\": "
        << max_rebuild_preflight_entries << ",\n"
        << "  \"max_state_exchanges\": " << max_state_exchanges << ",\n"
        << "  \"max_sigma_exchanges\": " << max_sigma_exchanges << ",\n"
        << "  \"max_mult_unchanged_cache_fast_paths\": "
        << max_mult_unchanged_cache_fast_paths << ",\n"
        << "  \"max_mult_bridge_storage_reuses\": "
        << max_mult_bridge_storage_reuses << ",\n"
        << "  \"max_mpi_reductions\": " << max_mpi_reductions << ",\n"
        << "  \"max_selection_reason_reductions\": "
        << max_selection_reason_reductions << ",\n"
        << "  \"max_stage_reason_reductions\": "
        << max_stage_reason_reductions << ",\n"
        << "  \"max_host_synchronizations\": " << max_host_syncs << "\n"
        << "}\n";
    ctx.Check(out.good(),
              "DGSEM production machine-readable evidence written");
  }
}

} // namespace dgsem_production_test
} // namespace igrlab_tests
