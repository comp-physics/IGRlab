#include "sim/hdg/state_io.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
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

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  static_assert(!std::is_polymorphic<dg_sim::hdg::HdgStateIo>::value,
                "HDG state I/O must remain nonvirtual.");

  const bool passed = AllRanks(TestAssociationAndFourComponents()) &&
      AllRanks(TestProjectionAndGenericImport());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (!passed && rank == 0)
  {
    std::cerr << "HDG state I/O test failed\n";
  }
  MPI_Finalize();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
