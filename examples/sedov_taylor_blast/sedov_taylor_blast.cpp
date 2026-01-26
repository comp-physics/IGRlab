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
#include "sedov_taylor_blast.hpp"
#include "glvismulti.hpp"
#include "limiters.hpp"
#include "timesteppers.hpp"



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
   mfem::real_t t_final = 4.0;
   mfem::real_t dt = -0.01;
   mfem::real_t cfl = 0.3;
   bool visualization = true;
   bool preassembleWeakDiv = true;
   int vis_steps = 50;
   bool vtk_save = true;
   mfem::real_t dt_save = 0.01;

   int precision = 8;
   std::cout.precision(precision);
   // --- New runtime IGR/Euler control flags ---
   double C_alpha_user = -1.0;        // negative → auto (1.0 default)
   std::string igr_space = "H1";     // options: H1, L2
   std::string igr_grad  = "H1";     // options: H1, BrokenL2
   double vx_in = 3.0;               // free-stream/inlet velocity x
   double vy_in = 0.0;               // free-stream/inlet velocity y
   //DGSEM
   // int use_dgsem = 0;

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
   args.AddOption(&vtk_save, "-vtk", "--vtk_save", "-no-vtk",
                  "--no-vtk_save",
                  "Enable or disable VTK visualization.");
   args.AddOption(&dt_save, "-dt_save", "--time-step-save",
                  "Saving Time step.");
   args.AddOption(&preassembleWeakDiv, "-ea", "--element-assembly-divergence",
                  "-mf", "--matrix-free-divergence",
                  "Weak divergence assembly level\n"
                  "    ea - Element assembly with interpolated F\n"
                  "    mf - Nonlinear assembly in matrix-free manner");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&C_alpha_user, "-ca", "--c-alpha",
                  "Forward scaling constant C_alpha (default 1.0).");
   args.AddOption(&igr_space, "-is", "--igr-space",
                  "Space to solve for sigma: 'H1' or 'L2'.");
   args.AddOption(&igr_grad, "-ig", "--igr-grad",
                  "Gradient reconstruction type: 'H1' or 'BrokenL2'.");
   args.AddOption(&vx_in, "-vx", "--velocity-x",
                  "Free-stream / inflow velocity x-component.");
   args.AddOption(&vy_in, "-vy", "--velocity-y",
                  "Free-stream / inflow velocity y-component.");
   // args.AddOption(&use_dgsem, "-dgsem", "--dgsem",
   //             "Use DGSEM flux-difference for volume terms.");
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
   // L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   // Finite element space for a scalar (thermodynamic quantity)
   mfem::ParFiniteElementSpace fes(&pmesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   mfem::ParFiniteElementSpace dfes(&pmesh, &fec, dim, mfem::Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_equations, mfem::Ordering::byNODES);

   // --- Select Σ FE space based on --igr-space flag ---
   std::unique_ptr<mfem::FiniteElementCollection> sfec_ptr;
   if (igr_space == "H1" || igr_space == "h1")
   {
      sfec_ptr = std::make_unique<mfem::H1_FECollection>(std::max(order, 3), dim);
   }
   else if (igr_space == "L2" || igr_space == "l2")
   {
      sfec_ptr = std::make_unique<mfem::DG_FECollection>(order, dim);
   }
   else
   {
      sfec_ptr = std::make_unique<mfem::DG_FECollection>(order, dim);
   }
   mfem::ParFiniteElementSpace sfes(&pmesh, sfec_ptr.get());

   // --- H1 velocity space (continuous) and diagnostics (scalar DG) ---
   // Order for H1 projection: at least 1, often = order
   mfem::H1_FECollection h1fec(std::max(1, order), dim);
   mfem::ParFiniteElementSpace h1_vfes(&pmesh, &h1fec, /*vdim=*/dim, mfem::Ordering::byNODES);
   // Continuous velocity for visualization / gradients
   mfem::ParGridFunction uH1(&h1_vfes); uH1 = 0.0;
   // Scalar DG fields to *display* diagnostics (reuse your DG scalar space 'sfes')
   mfem::ParGridFunction g_umag(&h1_vfes), g_div(&h1_vfes), g_gfrob(&h1_vfes);
   g_umag = 0.0; g_div = 0.0; g_gfrob = 0.0;


  // --- RK3 with ordered limiter callbacks ---
   const double rho_min       = 5.0e-7;
   const double p_min         = 1.0e-6;
   const int    quad_order    = 0;    // 0 => auto (2*order+2)
   const int    p_bisect_iters= 12;
   mfem::RK3SSPWithLimiter rk3(vfes);
   rk3.SetEulerParams(dim, /*gamma=*/1.4);
   rk3.SetLimiterParams(rho_min, p_min, quad_order, p_bisect_iters);
   rk3.LimitInitial(true);
   // (1) BJ slope limiter first (component-wise)
   // BJOptions bj;
   // bj.dim = 2;
   // bj.qord_avg   = 0; // auto
   // bj.slope_qord = 0; // auto
   // bj.limit_rho = false;
   // bj.limit_momentum = true;
   // bj.limit_energy = false;
   // rk3.AddBarthJespersenLimiter(bj);
   // 1) Characteristic BJ slope limiter
   // CharBJOptions bj;
   // bj.dim        = 2;
   // bj.gamma      = 1.4;
   // bj.slope_qord = 2*vfes.GetFE(0)->GetOrder() + 2; // safe default
   // bj.limit_x    = true;
   // bj.limit_y    = true;
   // rk3.AddCharBJLimiter(bj);
   // 1) Face-normal characteristic BJ (monotonicity)
   // FaceBJOptions bj;
   // bj.dim        = 2;
   // bj.gamma      = 1.4;
   // bj.slope_qord = 2*vfes.GetFE(0)->GetOrder() + 2;
   // rk3.AddFaceCharBJLimiter(bj);
   // (2) Characteristic TVB (keep your previous settings)
   // CharTVBOptions tvb;
   // tvb.dim        = 2;
   // tvb.gamma      = 1.4;
   // tvb.theta      = 1.3;
   // tvb.M_tvb      = 1.0;
   // tvb.tau_flag   = 0.05;
   // tvb.slope_qord = 2*vfes.GetFE(0)->GetOrder() + 2;
   // rk3.AddCharTVBLimiter(tvb);
   // Use the face-normal characteristic TVB (direction-invariant)
   // rk3.AddCharTVBFaceNormalLimiter(tvb);
   // (2) Regular characteristic Venkat (x/y)
   mfem::VenkatOptions vk;
   vk.dim = 2; 
   vk.gamma = 1.4;
   vk.slope_qord = 2*vfes.GetFE(0)->GetOrder() + 2;
   vk.eps_kappa  = 100.0;        // increase to be less aggressive
   // rk3.AddCharVenkatXYLimiter(vk);
   // // Or face-normal variant for general meshes:
   rk3.AddCharVenkatFaceNormalLimiter(vk);
   // 2) Krivodonova hierarchical modal limiter (keeps high order away from shocks)
   // KrivoOptions krivo;
   // krivo.dim = 2;
   // krivo.gamma = 1.4;
   // krivo.face_qord = 0;    // auto: p+2
   // krivo.inflate  = 0.05;  // slight widening of neighbor bounds (try 0.0–0.1)
   // rk3.AddKrivodonovaLimiter(krivo);
   // (3) Positivity limiter last
   rk3.AddZhangShuLimiter();

   // IGR params
   const double rho_floor = rho_min;
   // --- compute nominal Δx and α from the mesh and polynomial order ---
   mfem::real_t hmin_alpha = mfem::infinity();
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      hmin_alpha = std::min(pmesh.GetElementSize(i, 1), hmin_alpha);
   }
   MPI_Allreduce(MPI_IN_PLACE, &hmin_alpha, 1,
               mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, pmesh.GetComm());
   // Effective spacing for DG(p): Δx_eff ≈ h / (p+1)
   // const double dx_eff = hmin_alpha / (order + 1);
   const double dx_eff = hmin_alpha ;
   // Choose a constant Cα (start with 1.0 for sharp resolution but more ringing, 3-10 generally)
   // const double C_alpha = 1.0;
   double C_alpha = (C_alpha_user > 0.0) ? C_alpha_user : 1.0;
   // α = Cα * (Δx_eff)^2
   const double alpha_igr = C_alpha * dx_eff * dx_eff;
   // Build IGR
   dg_igr::IGRSolver igr(sfes, vfes, dim, num_equations, alpha_igr);
   // Pick your gradient backend:
   if (igr_grad == "H1" || igr_grad == "h1")
   {
      igr.SetGradBackend(dg_igr::IGRSolver::GradBackend::H1Projection);
   }
   else if (igr_grad == "BrokenL2" || igr_grad == "broken" || igr_grad == "L2" || igr_grad == "l2")
   {
      igr.SetGradBackend(dg_igr::IGRSolver::GradBackend::BrokenDG);
   }
   else
   {
      igr.SetGradBackend(dg_igr::IGRSolver::GradBackend::BrokenDG);
   }
   igr.SetRhoFloor(rho_floor);
   // igr.SetSmoother(dg_igr::IGRSolver::SmootherType::Jacobi);
   igr.SetSmoother(dg_igr::IGRSolver::SmootherType::L1Jacobi);
   // igr.SetSweeps(/*regular=*/1000, /*warm_start=*/1000);
   igr.SetSolveMode(dg_igr::IGRSolver::SolveMode::Krylov); // default in the new header
   igr.SetLinearSolveParams(1e-8, 0.0, 1000, /*print=*/0);

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
   const int base_cells = 32;
   const int total_ref  = ser_ref_levels + par_ref_levels;
   // const int mx = base_cells << total_ref;
   // const int my = base_cells << total_ref;
   const int mx = base_cells;
   const int my = base_cells;
   mfem::VectorFunctionCoefficient u0 = mfem::SedovTaylorBlastInitialCondition(/*gamma=*/1.4, 
      /*E_blast*/ 1.0, 
      mx, 
      my);
   mfem::ParGridFunction sol(&vfes);
   sol.ProjectCoefficient(u0);
   mfem::ParGridFunction mom(&dfes, sol.GetData() + fes.GetNDofs());
   mfem::ParGridFunction density(&fes, sol.GetData());  // Start at beginning, no offset
   mfem::ParGridFunction mom_x(&fes, sol.GetData() + 1 * fes.GetNDofs());
   mfem::ParGridFunction mom_y(&fes, sol.GetData() + 2 * fes.GetNDofs());
   mfem::ParGridFunction energy(&fes, sol.GetData() + 3 * fes.GetNDofs());
   mfem::PressureCoefficient pcoef(sol, dim, specific_heat_ratio);
   mfem::ParGridFunction press(&fes);
   press.ProjectCoefficient(pcoef);
   // --- Velocity/H1 diagnostics updater (call before plotting) ---
   mfem::VelocityFromConsCoefficient ucoef(dim, /*U=*/&sol, /*rho_floor=*/rho_floor);
   auto UpdateH1VelocityAndDiagnostics = [&]() {
   // 1) Project conservative state -> continuous velocity (H1^dim)
   uH1.ProjectCoefficient(ucoef);

   // 2) Project diagnostics to scalar DG fields for easy coloring
   mfem::GradFrobNormCoeff cF(&uH1, dim);
   mfem::DivCoeff         cD(&uH1, dim);
   mfem::VelMagCoeff      cM(&uH1, dim);
   g_gfrob.ProjectCoefficient(cF);  // ||grad u||_F
   g_div  .ProjectCoefficient(cD);  // div u
   g_umag .ProjectCoefficient(cM);  // |u|
   };

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

   // 6. Set up the nonlinear form with Euler flux wrapped by IGR
   mfem::EulerFlux base_flux(dim, specific_heat_ratio);

   // IMPORTANT: mfem::FluxWithIGR must see the SAME mfem::ParGridFunction that dg_igr::IGRSolver updates.
   // Assuming dg_igr::IGRSolver::GetSigma() returns mfem::ParGridFunction&.
   // mfem::FluxWithIGR igr_flux(base_flux, &igr.GetSigma());
   mfem::FluxWithIGR igr_flux(base_flux);
   igr_flux.SetSigmaField(&igr.GetSigma());

   // Use the wrapped flux in the numerical flux (e.g., Rusanov).
   // mfem::HLLCFlux numericalFlux(igr_flux, 1.4);
   // mfem::RusanovFlux numericalFlux(igr_flux);
   // mfem::RusanovFlux numericalFlux(igr_flux);
   mfem::RusanovFluxWithSides numericalFlux(igr_flux);

   mfem::DGHyperbolicConservationLaws base_op(
      vfes,
      std::unique_ptr<mfem::HyperbolicFormIntegrator>(
         new mfem::HyperbolicFormIntegrator(numericalFlux, IntOrderOffset)),
      preassembleWeakDiv);
   // if (use_dgsem)
   //    base_op.SetVolumeMode(VolumeMode::DGSEMFluxDiff);
   // else
   //    base_op.SetVolumeMode(VolumeMode::WeakDivergence);

   // BC markers (1=bottom, 2=right, 3=top, 4=left)
   mfem::Array<int> slip_bdr;
   mfem::MakeMarker(slip_bdr, pmesh, {1,2,3,4});

   base_op.AddBdrFaceIntegrator(
      new mfem::BdrHyperbolicFluxIntegrator(numericalFlux,
                                    std::make_unique<mfem::SlipWallState>(dim),
                                    IntOrderOffset),
      slip_bdr);

   // 7. Visualization
   igr.UpdateFromState(sol, /*warm_start=*/false);
   igr.GetSigma().ExchangeFaceNbrData();
   mfem::socketstream sout;
   vis::GLVisWindow w_density, w_sigma, w_energy, w_pressure, w_momentum_x;
   vis::GLVisWindow w_uvec, w_umag, w_div, w_gfrob;
   if (visualization)
   {
      const char vishost[] = "localhost";
      const int  visport = 19916;

      // Important: if you want to *display σ*, make sure it has face-neighbor data
      // ready on this rank (no need to recompute; we just exchange).
      igr.GetSigma().ExchangeFaceNbrData();

      bool ok1 = vis::Open(w_density, vishost, visport, precision,
                           numProcs, myRank, pmesh, density,
                           "density, t = 0");
      bool ok2 = vis::Open(w_sigma,   vishost, visport, precision,
                           numProcs, myRank, pmesh, igr.GetSigma(),
                           "sigma, t = 0");
      bool ok3 = vis::Open(w_energy, vishost, visport, precision,
                           numProcs, myRank, pmesh, energy,
                           "energy, t = 0");
      bool ok4 = vis::Open(w_pressure, vishost, visport, precision,
                           numProcs, myRank, pmesh, press,
                           "pressure, t = 0");
      bool ok5 = vis::Open(w_momentum_x, vishost, visport, precision,
                           numProcs, myRank, pmesh, press,
                           "momentum_x, t = 0");

      // Compute once before first frame so fields exist
      UpdateH1VelocityAndDiagnostics();

      // bool okU = vis::Open(w_uvec,  vishost, visport, precision,
      //                      numProcs, myRank, pmesh, uH1,     "u_H1 (continuous)");
      // bool okM = vis::Open(w_umag,  vishost, visport, precision,
      //                      numProcs, myRank, pmesh, g_umag,  "|u_H1|");
      // bool okD = vis::Open(w_div,   vishost, visport, precision,
      //                      numProcs, myRank, pmesh, g_div,   "div u_H1");
      // bool okF = vis::Open(w_gfrob, vishost, visport, precision,
      //                      numProcs, myRank, pmesh, g_gfrob, "||grad u_H1||_F");

      if (!(ok1 && ok2 && ok3))
      {
         visualization = false;
         if (mfem::Mpi::Root())
         {
            std::cout << "Unable to connect to GLVis. Visualization disabled.\n";
         }
      }
   }

   // ----------------------------------------------------------------------
   //  VTK / ParaView output setup
   // ----------------------------------------------------------------------
   // mfem::ParaViewDataCollection paraview_dc("forward_facing_output", &pmesh);
   int save_idx = 0;
   mfem::real_t t_next_save = 0.0;
   std::ostringstream dc_name;
   dc_name << "sedov_taylor_blast"
         << "_p_"  << order
         << "_ref_" << (ser_ref_levels+par_ref_levels)
         << "_is_" << igr_space
         << "_ig_" << igr_grad
         << "_ca_" << C_alpha_user;

mfem::ParaViewDataCollection paraview_dc(dc_name.str(), &pmesh);
   if (vtk_save)
   {  
   // Register fields you want to write (each will become its own .pvtu series)
   paraview_dc.RegisterField("density", &density);
   paraview_dc.RegisterField("momentum_x", &mom_x);
   paraview_dc.RegisterField("momentum_y", &mom_y);
   paraview_dc.RegisterField("energy", &energy);
   paraview_dc.RegisterField("sigma", &igr.GetSigma());
   paraview_dc.RegisterField("pressure", &press);
   // Optional configuration
   paraview_dc.SetPrefixPath("vtk_output"); // output directory
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY); // smaller, faster
   paraview_dc.SetHighOrderOutput(true);         // keep DG high-order data
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.Save();
   if (mfem::Mpi::Root())
   {
      std::cout << "Saving output to VTK.\n";
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

      // Use wrapped op here so Σ exists during the probe call too
      mfem::EulerWithIGR wrapped(base_op, igr, sol);
      wrapped.Mult(sol, z);

      mfem::real_t max_char_speed = wrapped.GetMaxCharSpeed();
      MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1,  mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_MAX, pmesh.GetComm());
      dt = cfl * hmin / max_char_speed / (2 * order + 1);
   }

   // Start the timer.
   mfem::tic_toc.Clear();
   mfem::tic_toc.Start();

   mfem::real_t t = 0.0;

   // Wrap the base DG op with IGR recomputation each Mult()
   mfem::EulerWithIGR wrapped(base_op, igr, sol);
   // wrapped.SetTime(t);
   // ode_solver->Init(wrapped);
   rk3.Init(wrapped);
   rk3.LimitInitial(false);
   rk3.SetLimiterSteps(0);

   bool done = false;
   for (int ti = 0; !done;)
   {
      mfem::real_t dt_real_base = std::min(dt, t_final - t);
      mfem::real_t dt_real = dt_real_base;

      // Detect if we will land on the next save time this step
      bool hit_save_boundary = false;
      if (vtk_save && dt_save > 0.0)
      {
         const mfem::real_t gap = t_next_save - t;
         if (gap >= 0.0 && gap <= dt_real_base)
         {
            dt_real = gap; // trim to land exactly on t_next_save
            hit_save_boundary = true;
         }
      }

      rk3.Step(sol, t, dt_real);

      if (hit_save_boundary) { t = t_next_save; }

      if (cfl > 0)
      {
         mfem::real_t max_char_speed = wrapped.GetMaxCharSpeed();
         MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1,  mfem::MPITypeMap<mfem::real_t>::mpi_type,
                       MPI_MAX, pmesh.GetComm());
         dt = cfl * hmin / max_char_speed / (2 * order + 1);
         if (mfem::Mpi::Root()) { std::cout << "dt: " << dt_real << std::endl; }
      }
      ti++;

      done = (t >= t_final - 1e-8 * dt);

      if (std::isnan(dt_real)){
         break;
      }

      // (optional) GLVis still by step cadence
      if (done || ti % vis_steps == 0)
      {
         if (mfem::Mpi::Root())
         {
            std::cout << "dt: " << dt_real << " time step: " << ti << ", time: " << t << std::endl;
         }
         if (visualization)
         {
            igr.UpdateFromState(sol, /*warm_start=*/true);
            igr.GetSigma().ExchangeFaceNbrData();
            vis::Update(w_density, numProcs, myRank, pmesh, density, t, "density");
            vis::Update(w_sigma,   numProcs, myRank, pmesh, igr.GetSigma(), t, "sigma");
            vis::Update(w_energy,  numProcs, myRank, pmesh, energy, t, "energy");
            pcoef.SetState(sol);
            press.ProjectCoefficient(pcoef);
            vis::Update(w_pressure, numProcs, myRank, pmesh, press, t, "pressure");
            vis::Update(w_momentum_x,  numProcs, myRank, pmesh, mom_x, t, "momentum_x");

            // UpdateH1VelocityAndDiagnostics();
            // vis::Update(w_uvec,  numProcs, myRank, pmesh, uH1,     t, "u_H1");
            // vis::Update(w_umag,  numProcs, myRank, pmesh, g_umag,  t, "|u_H1|");
            // vis::Update(w_div,   numProcs, myRank, pmesh, g_div,   t, "div u_H1");
            // vis::Update(w_gfrob, numProcs, myRank, pmesh, g_gfrob, t, "||grad u_H1||_F");
         }
      }

      // VTK export by dt_save
      if (vtk_save && dt_save > 0.0 && hit_save_boundary)
      {
         if (mfem::Mpi::Root())
         {
            std::cout << "dt: " << dt_real << " save VTK " << "time step: " << ti << ", time: " << t << std::endl;
         }
         // refresh derived fields
         igr.UpdateFromState(sol, /*warm_start=*/true);
         igr.GetSigma().ExchangeFaceNbrData();
         pcoef.SetState(sol);
         press.ProjectCoefficient(pcoef);

         paraview_dc.SetCycle(save_idx);   
         paraview_dc.SetTime(t);           
         paraview_dc.Save();

         ++save_idx;
         t_next_save += dt_save;
      }

      // Dump final step 
      if (done && vtk_save)
      {
         igr.UpdateFromState(sol, /*warm_start=*/true);
         igr.GetSigma().ExchangeFaceNbrData();
         pcoef.SetState(sol);
         press.ProjectCoefficient(pcoef);

         paraview_dc.SetCycle(save_idx);
         paraview_dc.SetTime(t);
         paraview_dc.Save();
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
