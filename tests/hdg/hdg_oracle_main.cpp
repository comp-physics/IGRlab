#include "numerics/hdg/scalar_hdg_oracle.hpp"

#include <mfem.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void WriteJsonNumber(std::ostream &stream, const double value)
{
   if (std::isfinite(value))
   {
      stream << value;
   }
   else
   {
      stream << "null";
   }
}

void WriteSerialJson(const std::filesystem::path &path,
                     const dg::hdg::SerialEvidence &evidence)
{
   std::ofstream stream(path);
   stream << std::setprecision(17);
   stream << "{\n"
          << "  \"schema\": \"dg_igr.hdg_cpu_oracle.serial.v1\",\n"
          << "  \"mfem_version\": \"" << mfem::GetVersionStr() << "\",\n"
          << "  \"mfem_git\": \"" << MFEM_GIT_STRING << "\",\n"
          << "  \"solver_tolerance\": 2e-12,\n"
          << "  \"comparison_tolerance\": 2e-10,\n"
          << "  \"pass\": " << (evidence.pass ? "true" : "false") << ",\n"
          << "  \"comparisons\": [\n";
   for (std::size_t i = 0; i < evidence.comparisons.size(); ++i)
   {
      const auto &check = evidence.comparisons[i];
      stream << "    {\"name\": \"" << check.name << "\", \"value\": ";
      WriteJsonNumber(stream, check.value);
      stream << ", \"tolerance\": ";
      WriteJsonNumber(stream, check.tolerance);
      stream << ", \"pass\": " << (check.pass ? "true" : "false") << "}"
             << (i + 1 == evidence.comparisons.size() ? "\n" : ",\n");
   }
   stream << "  ],\n  \"convergence_gates\": [\n";
   for (std::size_t i = 0; i < evidence.convergence_gates.size(); ++i)
   {
      const auto &gate = evidence.convergence_gates[i];
      stream << "    {\"problem\": \"" << dg::hdg::ToString(gate.problem)
             << "\", \"geometry\": \"" << dg::hdg::ToString(gate.geometry)
             << "\", \"order\": " << gate.order
             << ", \"observed_final_rate\": ";
      WriteJsonNumber(stream, gate.observed_final_rate);
      stream << ", \"required_final_rate\": ";
      WriteJsonNumber(stream, gate.required_final_rate);
      stream << ", \"margin\": ";
      WriteJsonNumber(stream, gate.margin);
      stream << ", \"all_values_finite\": "
             << (gate.all_values_finite ? "true" : "false")
             << ", \"errors_strictly_decrease\": "
             << (gate.errors_strictly_decrease ? "true" : "false")
             << ", \"pass\": " << (gate.pass ? "true" : "false") << "}"
             << (i + 1 == evidence.convergence_gates.size() ? "\n" : ",\n");
   }
   stream << "  ]\n}\n";
}

void WriteConvergenceCsv(const std::filesystem::path &path,
                         const dg::hdg::SerialEvidence &evidence)
{
   std::ofstream stream(path);
   stream << std::setprecision(17);
   stream << "problem,geometry,order,elements,h_max,l2_error,rate,"
             "recovery_error,conservation_error\n";
   for (const auto &row : evidence.convergence)
   {
      stream << dg::hdg::ToString(row.problem) << ','
             << dg::hdg::ToString(row.geometry) << ',' << row.order << ','
             << row.elements << ',' << row.h_max << ',' << row.l2_error << ','
             << row.rate << ',' << row.recovery_error << ','
             << row.conservation_error << '\n';
   }
}

void WriteMpiJson(const std::filesystem::path &path,
                  const dg::hdg::MpiEvidence &evidence)
{
   std::ofstream stream(path);
   stream << std::setprecision(17);
   stream << "{\n"
          << "  \"schema\": \"dg_igr.hdg_cpu_oracle.mpi.v1\",\n"
          << "  \"mfem_version\": \"" << mfem::GetVersionStr() << "\",\n"
          << "  \"mfem_git\": \"" << MFEM_GIT_STRING << "\",\n"
          << "  \"ranks\": " << evidence.ranks << ",\n"
          << "  \"global_elements\": " << evidence.global_elements << ",\n"
          << "  \"mfem_partition_elements\": "
          << evidence.mfem_partition_elements << ",\n"
          << "  \"owned_trace_dofs\": " << evidence.owned_trace_dofs << ",\n"
          << "  \"ghost_trace_dofs\": " << evidence.ghost_trace_dofs << ",\n"
          << "  \"shared_faces\": " << evidence.shared_faces << ",\n"
          << "  \"pass\": " << (evidence.pass ? "true" : "false") << ",\n"
          << "  \"comparisons\": [\n";
   for (std::size_t i = 0; i < evidence.comparisons.size(); ++i)
   {
      const auto &check = evidence.comparisons[i];
      stream << "    {\"name\": \"" << check.name << "\", \"value\": ";
      WriteJsonNumber(stream, check.value);
      stream << ", \"tolerance\": ";
      WriteJsonNumber(stream, check.tolerance);
      stream << ", \"pass\": " << (check.pass ? "true" : "false") << "}"
             << (i + 1 == evidence.comparisons.size() ? "\n" : ",\n");
   }
   stream << "  ],\n  \"trace_true_dof_map\": [\n";
   for (std::size_t i = 0; i < evidence.map_rank.size(); ++i)
   {
      stream << "    {\"rank\": " << evidence.map_rank[i]
             << ", \"local_dof\": " << evidence.map_local_dof[i]
             << ", \"global_true_dof\": "
             << evidence.map_global_true_dof[i]
             << ", \"owner_rank\": " << evidence.map_owner_rank[i]
             << ", \"is_owned\": "
             << (evidence.map_is_owned[i] ? "true" : "false") << "}"
             << (i + 1 == evidence.map_rank.size() ? "\n" : ",\n");
   }
   stream << "  ]\n}\n";
}

}  // namespace

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   try
   {
      if (argc != 4 || std::string(argv[1]) != "--mode")
      {
         throw std::runtime_error(
            "usage: hdg_oracle --mode serial|mpi OUTPUT_DIRECTORY");
      }
      const std::string mode = argv[2];
      const std::filesystem::path output = argv[3];
      if (rank == 0) { std::filesystem::create_directories(output); }
      MPI_Barrier(MPI_COMM_WORLD);

      bool pass = false;
      if (mode == "serial")
      {
         int ranks = 0;
         MPI_Comm_size(MPI_COMM_WORLD, &ranks);
         if (ranks != 1)
         {
            throw std::runtime_error("serial mode requires exactly one rank");
         }
         const dg::hdg::SerialEvidence evidence = dg::hdg::RunSerialOracle();
         if (rank == 0)
         {
            WriteSerialJson(output / "serial_checks.json", evidence);
            WriteConvergenceCsv(output / "convergence.csv", evidence);
            std::cout << "HDG_ORACLE_SERIAL "
                      << (evidence.pass ? "PASS" : "FAIL") << ' '
                      << output.string() << '\n';
         }
         pass = evidence.pass;
      }
      else if (mode == "mpi")
      {
         const dg::hdg::MpiEvidence evidence =
            dg::hdg::RunMpiOracle(MPI_COMM_WORLD);
         if (rank == 0)
         {
            WriteMpiJson(output /
                            ("mpi_" + std::to_string(evidence.ranks) +
                             "_checks.json"),
                         evidence);
            std::cout << "HDG_ORACLE_MPI "
                      << (evidence.pass ? "PASS" : "FAIL") << " ranks="
                      << evidence.ranks << ' ' << output.string() << '\n';
         }
         pass = evidence.pass;
      }
      else
      {
         throw std::runtime_error("unknown mode: " + mode);
      }
      int local_pass = pass ? 1 : 0;
      int global_pass = 0;
      MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
      MPI_Finalize();
      return global_pass == 1 ? 0 : 1;
   }
   catch (const std::exception &error)
   {
      std::cerr << "HDG_ORACLE_ERROR rank=" << rank << " " << error.what()
                << '\n';
      MPI_Abort(MPI_COMM_WORLD, 2);
      return 2;
   }
}
