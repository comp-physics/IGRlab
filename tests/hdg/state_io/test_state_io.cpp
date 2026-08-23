#include "sim/hdg/state_io.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>

namespace
{

bool AllRanks(const bool local)
{
  const int input = local ? 1 : 0;
  int output = 0;
  MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return output == 1;
}

class RankLocalNonFiniteEulerCoefficient final : public mfem::VectorCoefficient
{
public:
  explicit RankLocalNonFiniteEulerCoefficient(const int rank)
    : mfem::VectorCoefficient(dg_sim::hdg::hdg_euler_2d_components), rank_(rank)
  {
  }

  void Eval(mfem::Vector& value, mfem::ElementTransformation&,
            const mfem::IntegrationPoint&) override
  {
    value.SetSize(dg_sim::hdg::hdg_euler_2d_components);
    value = 0.0;
    value(0) = rank_ == 0 ? std::numeric_limits<double>::quiet_NaN() : 1.0;
    value(3) = 2.5;
  }

private:
  int rank_ = -1;
};

bool TestAssociationAndFourComponents()
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::L2_FECollection collection(1, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace volume_space(
      &mesh, &collection, dg_sim::hdg::hdg_euler_2d_components,
      mfem::Ordering::byNODES);
  mfem::ParGridFunction state(&volume_space);

  mfem::ParFiniteElementSpace scalar_space(&mesh, &collection);
  mfem::ParGridFunction scalar_state(&scalar_space);
  mfem::ParFiniteElementSpace by_vdim_space(
      &mesh, &collection, dg_sim::hdg::hdg_euler_2d_components,
      mfem::Ordering::byVDIM);
  mfem::ParGridFunction by_vdim_state(&by_vdim_space);

  mfem::Mesh other_serial_mesh = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  mfem::ParMesh other_mesh(MPI_COMM_WORLD, other_serial_mesh);

  return dg_sim::hdg::IsHdgVolumeStateAssociation(
             mesh, volume_space, state) &&
      !dg_sim::hdg::IsHdgVolumeStateAssociation(mesh, scalar_space, scalar_state) &&
      !dg_sim::hdg::IsHdgVolumeStateAssociation(
          mesh, by_vdim_space, by_vdim_state) &&
      !dg_sim::hdg::IsHdgVolumeStateAssociation(other_mesh, volume_space, state);
}

bool TestProjectionAndGenericImport()
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::L2_FECollection collection(1, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace volume_space(
      &mesh, &collection, dg_sim::hdg::hdg_euler_2d_components,
      mfem::Ordering::byNODES);
  mfem::ParGridFunction state(&volume_space);
  dg_sim::hdg::HdgStateIo state_io(mesh, volume_space, state);

  mfem::Vector values(dg_sim::hdg::hdg_euler_2d_components);
  values(0) = 1.0;
  values(1) = -2.0;
  values(2) = 3.5;
  values(3) = 11.0;
  mfem::VectorConstantCoefficient initial_condition(values);
  state_io.ProjectBorrowedInitialConditionOnce(initial_condition);

  const double* const coordinates = state_io.published_coordinates();
  const std::size_t scalar_count = state_io.published_scalar_count();
  const dg::hdg::generic::EulerFieldView imported =
      state_io.ImportPublishedInitialState();
  bool finite = coordinates != nullptr && scalar_count > 0;
  const std::size_t scalar_dofs_per_component =
      scalar_count / static_cast<std::size_t>(values.Size());
  bool projected_values = scalar_dofs_per_component > 0 &&
      scalar_count % values.Size() == 0;
  for (std::size_t coordinate = 0; coordinate < scalar_count; ++coordinate)
  {
    finite = finite && std::isfinite(coordinates[coordinate]);
    projected_values = projected_values &&
        coordinates[coordinate] ==
        values(static_cast<int>(coordinate / scalar_dofs_per_component));
  }

  return finite && projected_values &&
      imported.data == coordinates && imported.scalar_count == scalar_count &&
      imported.components == dg_sim::hdg::hdg_euler_2d_components &&
      dg::hdg::generic::IsValidField(imported, true);
}

bool TestIndependentTrueTraceProjection()
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_Interface_FECollection collection(
      1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &collection, dg_sim::hdg::hdg_euler_2d_components,
      mfem::Ordering::byNODES);
  mfem::ParGridFunction local_trace(&trace_space);
  mfem::Vector true_trace;
  mfem::Vector values(dg_sim::hdg::hdg_euler_2d_components);
  values(0) = 1.0;
  values(1) = -2.0;
  values(2) = 3.5;
  values(3) = 11.0;
  mfem::VectorConstantCoefficient initial_condition(values);

  if (!dg_sim::hdg::ProjectEulerTrueTraceCoefficientL2(
          trace_space, local_trace, true_trace, initial_condition, 5) ||
      true_trace.Size() != trace_space.GetTrueVSize() ||
      true_trace.Size() <= 0 || true_trace.CheckFinite() != 0)
  {
    return false;
  }
  const int scalar_true_dofs =
      true_trace.Size() / dg_sim::hdg::hdg_euler_2d_components;
  if (scalar_true_dofs <= 0 ||
      true_trace.Size() !=
          scalar_true_dofs * dg_sim::hdg::hdg_euler_2d_components)
  {
    return false;
  }
  for (int component = 0;
       component < dg_sim::hdg::hdg_euler_2d_components; ++component)
  {
    for (int scalar = 0; scalar < scalar_true_dofs; ++scalar)
    {
      if (std::abs(true_trace[component * scalar_true_dofs + scalar] -
                   values[component]) > 1.0e-12)
      {
        return false;
      }
    }
  }
  return true;
}

bool TestRankLocalNonFiniteTrueTraceProjectionIsCollectivelyRejected()
{
  int ranks = 0;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (ranks != 2) return true;

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_Interface_FECollection collection(
      1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &collection, dg_sim::hdg::hdg_euler_2d_components,
      mfem::Ordering::byNODES);
  mfem::ParGridFunction local_trace(&trace_space);
  mfem::Vector true_trace;
  RankLocalNonFiniteEulerCoefficient initial_condition(rank);

  const bool rejected = !dg_sim::hdg::ProjectEulerTrueTraceCoefficientL2(
      trace_space, local_trace, true_trace, initial_condition, 5);
  return AllRanks(rejected);
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  static_assert(!std::is_polymorphic<dg_sim::hdg::HdgStateIo>::value,
                "HDG state I/O must remain nonvirtual.");

  const bool passed = AllRanks(TestAssociationAndFourComponents()) &&
      AllRanks(TestProjectionAndGenericImport()) &&
      AllRanks(TestIndependentTrueTraceProjection()) &&
      AllRanks(TestRankLocalNonFiniteTrueTraceProjectionIsCollectivelyRejected());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (!passed && rank == 0)
  {
    std::cerr << "HDG state I/O test failed\n";
  }
  MPI_Finalize();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
