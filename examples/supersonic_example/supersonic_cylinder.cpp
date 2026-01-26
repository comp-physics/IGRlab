//                                Supersonic Cylinder - Parallel Version
//
// Compile with: make supersonic_cylinder
//
// Sample runs:
//
//       mpirun -np 4 ex18p -p 1 -rs 2 -rp 1 -o 1 -s 3
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 3 -s 4
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 5 -s 6
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 1 -s 3 -mf
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 3 -s 3 -mf
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation in parallel.
//
//                (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u,n), [[v]]>_F = 0
//
//               where (⋅,⋅)_T is volume integration, and <⋅,⋅>_F is face
//               integration, F is the Euler flux function, and F̂ is the
//               numerical flux.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates usage of mfem::DGHyperbolicConservationLaws
//               that wraps NonlinearFormIntegrators containing element and face
//               integration schemes. In this case the system also involves an
//               external approximate Riemann solver for the DG interface flux.
//               By default, weak-divergence is pre-assembled in element-wise
//               manner, which corresponds to (I_h(F(u_h)), ∇ v). This yields
//               better performance and similar accuracy for the included test
//               problems. This can be turned off and use nonlinear assembly
//               similar to matrix-free assembly when -mf flag is provided.
//               It also demonstrates how to use GLVis for in-situ visualization
//               of vector grid function and how to set top-view.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include "supersonic_cylinder.hpp"



int main(int argc, char *argv[])
{
   // 0. Parallel setup
   mfem::Mpi::Init(argc, argv);
   const int numProcs = mfem::Mpi::WorldSize();
   const int myRank = mfem::Mpi::WorldRank();
   mfem::Hypre::Init();

   // 1. Parse command-line options.
   int problem = 1;
   const mfem::real_t specific_heat_ratio = 1.4;
   const mfem::real_t gas_constant = 1.0;

   std::string mesh_file = "";
   int IntOrderOffset = 1; // Normally 1
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   mfem::real_t t_final = 2.0;
   mfem::real_t dt = -0.01;
   mfem::real_t cfl = 0.3;
   bool visualization = true;
   bool preassembleWeakDiv = true;
   int vis_steps = 50;

   int precision = 8;
   std::cout.precision(precision);

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "mfem::Mesh file to use. If not provided, then a periodic square"
                  " mesh will be used.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See mfem::EulerInitialCondition().");
   args.AddOption(&ser_ref_levels, "-rs", "--serial-refine",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--parallel-refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  mfem::ODESolver::ExplicitTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&preassembleWeakDiv, "-ea", "--element-assembly-divergence",
                  "-mf", "--matrix-free-divergence",
                  "Weak divergence assembly level\n"
                  "    ea - Element assembly with interpolated F\n"
                  "    mf - Nonlinear assembly in matrix-free manner");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file. When the user does not provide
   //    mesh file, use the default mesh file for the problem.
   mfem::Mesh mesh = mesh_file.empty() ? mfem::EulerMesh(problem) : mfem::Mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int num_equations = dim + 2;

   // Refine the mesh to increase the resolution. In this example we do
   // 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is a
   // command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine this
   // mesh further in parallel to increase the resolution. Once the parallel
   // mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh = mfem::ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Refine the mesh to increase the resolution. In this example we do
   // 'par_ref_levels' of uniform refinement, where 'par_ref_levels' is a
   // command-line parameter.
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   std::unique_ptr<mfem::ODESolver> ode_solver = mfem::ODESolver::SelectExplicit(ode_solver_type);

   // 4. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   mfem::DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   mfem::ParFiniteElementSpace fes(&pmesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   mfem::ParFiniteElementSpace dfes(&pmesh, &fec, dim, mfem::Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_equations, mfem::Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == mfem::Ordering::byNODES, "");

   HYPRE_BigInt glob_size = vfes.GlobalTrueVSize();
   if (mfem::Mpi::Root())
   {
      std::cout << "Number of unknowns: " << glob_size << std::endl;
   }

   // 5. Define the initial conditions, save the corresponding mesh and grid
   //    functions to files. These can be opened with GLVis using:
   //    "glvis -np 4  -m euler-mesh -g euler-1-init" (for x-momentum).

   // Initialize the state.
   mfem::VectorFunctionCoefficient u0 = mfem::EulerInitialCondition(problem,
                                                        specific_heat_ratio,
                                                        gas_constant);
   mfem::ParGridFunction sol(&vfes);
   sol.ProjectCoefficient(u0);
   mfem::ParGridFunction mom(&dfes, sol.GetData() + fes.GetNDofs());
   mfem::ParGridFunction density(&fes, sol.GetData());  // Start at beginning, no offset
   mfem::ParGridFunction mom_x(&fes, sol.GetData() + 1 * fes.GetNDofs());
   mfem::ParGridFunction mom_y(&fes, sol.GetData() + 2 * fes.GetNDofs());
   mfem::ParGridFunction energy(&fes, sol.GetData() + 3 * fes.GetNDofs());
   // Output the initial solution.
   {
      std::ostringstream mesh_name;
      mesh_name << "euler-mesh." << std::setfill('0') << std::setw(6) << mfem::Mpi::WorldRank();
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equations; k++)
      {
         mfem::ParGridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         std::ostringstream sol_name;
         sol_name << "euler-" << k << "-init." << std::setfill('0') << std::setw(6)
                  << mfem::Mpi::WorldRank();
         std::ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 6. Set up the nonlinear form with euler flux and numerical flux
   mfem::EulerFlux flux(dim, specific_heat_ratio);
   mfem::RusanovFlux numericalFlux(flux);
   // mfem::HLLCFlux numericalFlux(flux, specific_heat_ratio);  // pass gamma
   // mfem::DiffusiveRusanovFlux numericalFlux(flux,
   //                                    /*beta=*/0.5,
   //                                    /*delta=*/0.0,
   //                                    /*cap=*/std::numeric_limits<mfem::real_t>::infinity(),
   //                                    /*sensor_k=*/0.0,
   //                                    /*sensor_max=*/1.0);
   // ComponentwiseUpwindFlux numericalSurfFlux(flux);
   mfem::DGHyperbolicConservationLaws euler(
      vfes, 
      std::unique_ptr<mfem::HyperbolicFormIntegrator>(
         new mfem::HyperbolicFormIntegrator(numericalFlux, IntOrderOffset)),
      preassembleWeakDiv);

   // Set up boundary markers and integrators
   // Boundary markers (attributes): 1=bottom, 2=right, 3=top, 4=left, 5=circle
   // Set up boundary condition markers
   mfem::Array<int> inlet_bdr, outlet_bdr, slip_bdr, cyl_bdr;
   mfem::MakeMarker(inlet_bdr, pmesh, {4});
   mfem::MakeMarker(outlet_bdr, pmesh, {2});
   mfem::MakeMarker(slip_bdr, pmesh, {1,3});
   mfem::MakeMarker(cyl_bdr, pmesh, {5});
   mfem::Vector Uin(4);
   // Supersonic Cyl
   Uin(0) = 1.4;
   Uin(1) = 1.4 * 1.2;
   Uin(2) = 0.0;
   Uin(3) = 8.8;
   // Vortex Case
   // Uin(0) = 1.0;
   // Uin(1) = 1.0 * 1.0;
   // Uin(2) = 0.0;
   // Uin(3) = 714.8;
   euler.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
                                    std::make_unique<mfem::SupersonicInletState>(Uin),
                                    IntOrderOffset),
      inlet_bdr);
   // euler.AddBdrFaceIntegrator(
   //    new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
   //                                  std::make_unique<mfem::SupersonicOutletState>(),
   //                                  IntOrderOffset),
   //    inlet_bdr);

   euler.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
                                    std::make_unique<mfem::SupersonicOutletState>(),
                                    IntOrderOffset),
      outlet_bdr);
   // euler.AddBdrFaceIntegrator(
   //    new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
   //                                  std::make_unique<mfem::SupersonicInletState>(Uin),
   //                                  IntOrderOffset),
   //    outlet_bdr);

   euler.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
                                    std::make_unique<mfem::SlipWallState>(dim),
                                    IntOrderOffset),
      slip_bdr);
   // euler.AddBdrFaceIntegrator(
   //    new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
   //                                  std::make_unique<mfem::SupersonicInletState>(Uin),
   //                                  IntOrderOffset),
   //    slip_bdr);

   euler.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
                                    std::make_unique<mfem::SlipWallState>(dim),
                                    IntOrderOffset),
      cyl_bdr);
   // euler.AddBdrFaceIntegrator(
   //    new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
   //                                  std::make_unique<mfem::SupersonicInletState>(Uin),
   //                                  IntOrderOffset),
   //    cyl_bdr);


   // 7. Visualize momentum with its magnitude
   mfem::socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         visualization = false;
         if (mfem::Mpi::Root())
         {
            std::cout << "Unable to connect to GLVis server at " << vishost << ':'
                 << visport << std::endl;
            std::cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout.precision(precision);
         // Plot magnitude of vector-valued momentum
         sout << "parallel " << numProcs << " " << myRank << "\n";
         // sout << "solution\n" << pmesh << mom;
         // sout << "window_title 'momentum, t = 0'\n";
         // sout << "solution\n" << pmesh << mom_x;
         // sout << "window_title 'momentum(x), t = 0'\n";
         // sout << "solution\n" << pmesh << mom_y;
         // sout << "window_title 'momentum(y), t = 0'\n";
         sout << "solution\n" << pmesh << density;
         sout << "window_title 'density, t = 0'\n";
         // sout << "solution\n" << pmesh << energy;
         // sout << "window_title 'energy, t = 0'\n";
         sout << "view 0 0\n";  // view from top
         sout << "keys jlm\n";  // turn off perspective and light, show mesh
         sout << "pause\n";
         sout << std::flush;
         if (mfem::Mpi::Root())
         {
            std::cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
         MPI_Barrier(pmesh.GetComm());
      }
   }

   // 8. Time integration

   // When dt is not specified, use CFL condition.
   // Compute h_min and initial maximum characteristic speed
   mfem::real_t hmin = mfem::infinity();
   if (cfl > 0)
   {
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         hmin = std::min(pmesh.GetElementSize(i, 1), hmin);
      }
      MPI_Allreduce(MPI_IN_PLACE, &hmin, 1,  mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN,
                    pmesh.GetComm());
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces (and all
      // elements with -mf).
      mfem::Vector z(sol.Size());
      euler.Mult(sol, z);

      mfem::real_t max_char_speed = euler.GetMaxCharSpeed();
      MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1,  mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_MAX,
                    pmesh.GetComm());
      dt = cfl * hmin / max_char_speed / (2 * order + 1);
   }

   // Start the timer.
   mfem::tic_toc.Clear();
   mfem::tic_toc.Start();

   // Init time integration
   mfem::real_t t = 0.0;
   euler.SetTime(t);
   ode_solver->Init(euler);

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done;)
   {
      mfem::real_t dt_real = std::min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0) // update time step size with CFL
      {
         mfem::real_t max_char_speed = euler.GetMaxCharSpeed();
         MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1,  mfem::MPITypeMap<mfem::real_t>::mpi_type,
                       MPI_MAX,
                       pmesh.GetComm());
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
         std::cout << "dt: " << dt << std::endl;
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);
      if (done || ti % vis_steps == 0)
      {
         if (mfem::Mpi::Root())
         {
            std::cout << "time step: " << ti << ", time: " << t << std::endl;
         }
         if (visualization)
         {
            // sout << "window_title 'momentum, t = " << t << "'\n";
            sout << "parallel " << numProcs << " " << myRank << "\n";
            // sout << "solution\n" << pmesh << mom << std::flush;
            // sout << "solution\n" << pmesh << mom_x;
            // sout << "window_title 'momentum(x), t = " << t << "'\n";
            // sout << "solution\n" << pmesh << mom_y;
            // sout << "window_title 'momentum(y), t = " << t << "'\n";
            sout << "solution\n" << pmesh << density;
            sout << "window_title 'density, t = " << t << "'\n";
            // sout << "solution\n" << pmesh << energy;
            // sout << "window_title 'energy, t = " << t << "'\n";
         }
      }
   }

   mfem::tic_toc.Stop();
   if (mfem::Mpi::Root())
   {
      std::cout << " done, " << mfem::tic_toc.RealTime() << "s." << std::endl;
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -np 4 -m euler-mesh-final -g euler-1-final" (for x-momentum).
   {
      std::ostringstream mesh_name;
      mesh_name << "euler-mesh-final." << std::setfill('0') << std::setw(6)
                << mfem::Mpi::WorldRank();
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equations; k++)
      {
         mfem::ParGridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
         std::ostringstream sol_name;
         sol_name << "euler-" << k << "-final." << std::setfill('0') << std::setw(6)
                  << mfem::Mpi::WorldRank();
         std::ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 10. Compute the L2 solution error summed for all components.
   const mfem::real_t error = sol.ComputeLpError(2, u0);
   if (mfem::Mpi::Root())
   {
      std::cout << "Solution error: " << error << std::endl;
   }

   return 0;
}
