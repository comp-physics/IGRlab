// Frozen-operator research probe. Host timings describe this probe, not GPU
// performance. Production IGR forms/coefficients/faces remain authoritative.
#include "mfem.hpp"
#include "physics/euler/conversions.hpp"
#include "physics/igr/IGR.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point t)
{ return std::chrono::duration<double>(Clock::now() - t).count(); }
double Ratio(double error, double scale)
{ return scale > 0.0 ? error / scale : (error == 0.0 ? 0.0 : INFINITY); }

struct Element
{
  mfem::Array<int> dofs;
  mfem::DenseMatrix mass, stiffness;
};

double ElementSquare(const Element& element, const mfem::Vector& x,
                     bool gradient)
{
  const auto& matrix = gradient ? element.stiffness : element.mass;
  long double value = 0.0L;
  for (int i = 0; i < element.dofs.Size(); ++i)
  {
    if (!std::isfinite(x(element.dofs[i]))) return INFINITY;
    for (int j = 0; j < element.dofs.Size(); ++j)
      value += static_cast<long double>(x(element.dofs[i])) * matrix(i, j) *
               x(element.dofs[j]);
  }
  if (!std::isfinite(value)) return INFINITY;
  return std::max(0.0, static_cast<double>(value));
}

double PhysicalNorm(const std::vector<Element>& elements,
                    const mfem::Vector& x, bool gradient = false)
{
  double square = 0.0;
  for (const auto& element : elements)
    square += ElementSquare(element, x, gradient);
  return std::sqrt(square);
}

// Exact local principal solves, restricted to the center element. Store only
// the center rows of each patch inverse; no global inverse is used to build P.
class PatchInverse final : public mfem::Solver
{
  struct Patch
  {
    std::vector<int> input;
    mfem::Array<int> output;
    mfem::DenseMatrix rows;
  };
  std::vector<std::unique_ptr<Patch>> patches_;
  mfem::Vector diagonal_;
public:
  long long storage_bytes = 0, setup_dense_entries = 0;
  int max_patch_dofs = 0;
  mutable long long calls = 0;
  mutable double apply_seconds = 0.0;

  PatchInverse(const mfem::DenseMatrix& a,
               const std::vector<Element>& elements,
               const std::vector<std::vector<int>>& distance, int radius)
    : mfem::Solver(a.Height())
  {
    if (radius < 0)
    {
      diagonal_.SetSize(height);
      for (int i = 0; i < height; ++i)
      {
        MFEM_VERIFY(std::isfinite(a(i, i)) && a(i, i) > 0.0,
                    "Point Jacobi requires positive finite diagonal.");
        diagonal_(i) = 1.0 / a(i, i);
      }
      storage_bytes = sizeof(double) * height;
      return;
    }
    for (int center = 0; center < static_cast<int>(elements.size()); ++center)
    {
      auto patch = std::make_unique<Patch>();
      patch->output = elements[center].dofs;
      std::vector<int> core_position;
      for (int e = 0; e < static_cast<int>(elements.size()); ++e)
        if (distance[center][e] <= radius)
          for (int j = 0; j < elements[e].dofs.Size(); ++j)
          {
            if (e == center)
              core_position.push_back(static_cast<int>(patch->input.size()));
            patch->input.push_back(elements[e].dofs[j]);
          }
      const int np = static_cast<int>(patch->input.size());
      MFEM_VERIFY(np <= 1024, "Patch exceeds bounded 1024-DOF probe cap.");
      max_patch_dofs = std::max(max_patch_dofs, np);
      setup_dense_entries += static_cast<long long>(np) * np;
      // Solve A_patch^T q=e_core, so q^T is the requested inverse row.
      mfem::DenseMatrix transpose(np);
      for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j)
          transpose(i, j) = a(patch->input[j], patch->input[i]);
      mfem::DenseMatrixInverse inverse(transpose);
      patch->rows.SetSize(patch->output.Size(), np);
      mfem::Vector unit(np), row(np);
      for (int i = 0; i < patch->output.Size(); ++i)
      {
        unit = 0.0;
        unit(core_position[i]) = 1.0;
        inverse.Mult(unit, row);
        for (int j = 0; j < np; ++j)
        {
          MFEM_VERIFY(std::isfinite(row(j)), "Nonfinite local inverse row.");
          patch->rows(i, j) = row(j);
        }
      }
      storage_bytes += sizeof(double) * static_cast<long long>(np) *
                         patch->output.Size() +
                       sizeof(int) * (np + patch->output.Size());
      patches_.push_back(std::move(patch));
    }
  }

  void SetOperator(const mfem::Operator& op) override
  { MFEM_VERIFY(op.Height() == height, "Frozen patch operator size changed."); }
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    const auto begin = Clock::now();
    y.SetSize(height);
    if (diagonal_.Size())
      for (int i = 0; i < height; ++i) y(i) = diagonal_(i) * x(i);
    else
      for (const auto& patch : patches_)
        for (int i = 0; i < patch->output.Size(); ++i)
        {
          double value = 0.0;
          for (int j = 0; j < static_cast<int>(patch->input.size()); ++j)
            value += patch->rows(i, j) * x(patch->input[j]);
          y(patch->output[i]) = value;
        }
    ++calls;
    apply_seconds += Seconds(begin);
  }
};

double Legendre(int p, double x)
{
  double a = 1.0, b = x;
  if (p == 0) return a;
  for (int k = 2; k <= p; ++k)
  {
    const double c = ((2.0 * k - 1.0) * x * b - (k - 1.0) * a) / k;
    a = b;
    b = c;
  }
  return b;
}

void ReportError(const char* event, const std::string& load, int radius,
                 int iterations, double damping, const mfem::Operator& op,
                 const std::vector<Element>& elements, const mfem::Vector& rhs,
                 const mfem::Vector& reference, const mfem::Vector& x)
{
  mfem::Vector residual(rhs.Size()), error(x);
  op.Mult(x, residual);
  residual -= rhs;
  error -= reference;
  std::cout << "[igr-screened] event=" << event << " load=" << load
            << " radius=" << radius << " iterations=" << iterations
            << " damping=" << damping
            << " residual_abs=" << residual.Norml2()
            << " residual_rel=" << Ratio(residual.Norml2(), rhs.Norml2())
            << " error_l2_abs=" << PhysicalNorm(elements, error)
            << " error_l2_rel=" << Ratio(PhysicalNorm(elements, error),
                                          PhysicalNorm(elements, reference))
            << " error_gradient_abs=" << PhysicalNorm(elements, error, true)
            << " error_gradient_rel=" << Ratio(PhysicalNorm(elements, error, true),
                                                PhysicalNorm(elements, reference, true))
            << '\n';
}
}

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1, "Locality probe requires one rank.");
  mfem::Device device("cpu");
  int nx = 8, ny = 4, order = 2, radius_max = 2, max_iterations = 200;
  mfem::real_t c_alpha = 1.0, shock_width_cells = 0.5, shift_cells = 0.0, damping = 1.0;
  const char* state_name = "shock-smooth";
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&nx, "-nx", "--nx", "Even number of cells in x.");
  args.AddOption(&ny, "-ny", "--ny", "Number of cells in y; 1 gives a strip.");
  args.AddOption(&order, "-p", "--order", "Euler and L2 sigma polynomial order.");
  args.AddOption(&radius_max, "-r", "--radius", "Maximum face-hop patch radius, 0..6.");
  args.AddOption(&c_alpha, "-ca", "--c-alpha", "Alpha=C*h^2 on square cells.");
  args.AddOption(&shock_width_cells, "-w", "--width-cells", "Smooth shock tanh width/h.");
  args.AddOption(&shift_cells, "-sh", "--shift-cells", "Translate the smooth shock by this many cells.");
  args.AddOption(&damping, "-d", "--damping", "Stationary correction damping.");
  args.AddOption(&max_iterations, "-mi", "--max-iterations", "FGMRES cap.");
  args.AddOption(&state_name, "-s", "--state",
                 "constant | smooth | shock-smooth | shock-aligned | shock-shifted");
  args.Parse();
  if (!args.Good()) { args.PrintUsage(std::cout); return 2; }
  const std::string state_kind(state_name);
  MFEM_VERIFY(nx >= 2 && nx % 2 == 0 && ny >= 1 && order >= 1 && order <= 8 &&
                radius_max >= 0 && radius_max <= 6 && c_alpha > 0.0 &&
                std::isfinite(c_alpha) && shock_width_cells > 0.0 &&
                std::isfinite(shock_width_cells) && std::isfinite(shift_cells) &&
                std::abs(shift_cells) < 0.5 * nx && damping > 0.0 &&
                std::isfinite(damping) && max_iterations >= 1 && max_iterations <= 1000,
              "Invalid bounded probe options.");
  MFEM_VERIFY(state_kind == "constant" || state_kind == "smooth" ||
                state_kind == "shock-smooth" || state_kind == "shock-aligned" ||
                state_kind == "shock-shifted", "Unknown prescribed state.");
  MFEM_VERIFY(shift_cells == 0.0 || state_kind == "shock-smooth",
              "--shift-cells applies only to the continuous shock-smooth state.");
  MFEM_VERIFY(static_cast<long long>(nx) * ny * (order + 1) * (order + 1) <= 1600,
              "Probe exceeds 1600 scalar DOFs; global dense sampling is a reference only.");
  const auto total_start = Clock::now();
  const double h = 1.0 / nx, alpha = c_alpha * h * h, gamma = 1.4;
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    nx, ny, mfem::Element::QUADRILATERAL, true, 1.0, ny * h);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::L2_FECollection state_collection(order, 2, mfem::BasisType::GaussLobatto);
  mfem::L2_FECollection sigma_collection(order, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace state_space(&mesh, &state_collection, 4,
                                         mfem::Ordering::byNODES);
  mfem::ParFiniteElementSpace sigma_space(&mesh, &sigma_collection);
  mfem::ParGridFunction state(&state_space);
  mfem::Vector primitive(4), conservative(4), coordinate(2), center(2);
  mfem::Array<int> vdofs;
  const double rho_left = 1.0, pressure_left = 1.0;
  const double velocity_left = 2.0 * std::sqrt(gamma);
  const double rho_right = (gamma + 1.0) / (gamma - 1.0 + 0.5);
  const double pressure_right = 1.0 + 2.0 * gamma / (gamma + 1.0) * 3.0;
  for (int e = 0; e < mesh.GetNE(); ++e)
  {
    const auto& nodes = state_space.GetFE(e)->GetNodes();
    state_space.GetElementVDofs(e, vdofs);
    mesh.GetElementCenter(e, center);
    auto* transformation = mesh.GetElementTransformation(e);
    for (int a = 0; a < nodes.GetNPoints(); ++a)
    {
      transformation->Transform(nodes.IntPoint(a), coordinate);
      double mix = 0.0;
      if (state_kind == "shock-smooth")
        mix = 0.5 * (1.0 + std::tanh((coordinate(0) - (0.5 + shift_cells * h)) /
                                     (shock_width_cells * h)));
      else if (state_kind == "shock-aligned") mix = center(0) >= 0.5 ? 1.0 : 0.0;
      else if (state_kind == "shock-shifted")
        mix = coordinate(0) >= 0.5 + 0.37 * h ? 1.0 : 0.0;
      primitive(0) = rho_left + mix * (rho_right - rho_left);
      primitive(1) = velocity_left / primitive(0); // constant mass flux
      primitive(2) = 0.0;
      primitive(3) = pressure_left + mix * (pressure_right - pressure_left);
      if (state_kind == "smooth")
      {
        const double phase = 2.0 * std::acos(-1.0) * coordinate(0);
        primitive(0) = 1.0 + 0.2 * std::sin(phase);
        primitive(1) = 1.0 + 0.15 * std::cos(phase);
        primitive(3) = 1.0;
      }
      mfem::EulerPrimToConservative(primitive, 2, gamma, conservative);
      for (int c = 0; c < 4; ++c)
        state(vdofs[c * nodes.GetNPoints() + a]) = conservative(c);
    }
  }

  dg_igr::IGRSolver igr(sigma_space, state_space, 2, 4, alpha);
  igr.SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
  igr.SetExactDGFaceOperator(true);
  // Materialize A and b via the existing lifecycle; do not use this zero-sweep
  // iterate as a solution or as the FP64 reference.
  igr.SetSolveMode(dg_igr::IGRSolveMode::SmootherSweeps);
  igr.SetSweeps(0, 0);
  const auto assembly_start = Clock::now();
  igr.UpdateFromState(state, false);
  const double assembly_seconds = Seconds(assembly_start);
  const mfem::Operator* op = igr.GetLinearOperator();
  MFEM_VERIFY(op && op->Height() == sigma_space.GetVSize() &&
                igr.GetLastExactFaceEffective(), "Actual exact-face L2 operator required.");
  const int n = op->Height(), ne = mesh.GetNE();
  mfem::DenseMatrix dense(n);
  mfem::Vector unit(n), image(n), action(n);
  const auto sampling_start = Clock::now();
  for (int j = 0; j < n; ++j)
  {
    unit = 0.0;
    unit(j) = 1.0;
    op->Mult(unit, image);
    for (int i = 0; i < n; ++i) dense(i, j) = image(i);
  }
  const double sampling_seconds = Seconds(sampling_start);
  for (int i = 0; i < n; ++i) unit(i) = std::sin(0.37 * i) + std::cos(0.19 * i);
  op->Mult(unit, image);
  dense.Mult(unit, action);
  action -= image;
  MFEM_VERIFY(Ratio(action.Norml2(), image.Norml2()) < 1.0e-11,
              "Dense sampling disagrees with independent current-operator action.");
  const auto factor_start = Clock::now();
  mfem::DenseMatrixInverse reference_inverse(dense);
  const double factor_seconds = Seconds(factor_start);

  std::vector<Element> elements(ne);
  mfem::MassIntegrator mass_integrator;
  mfem::DiffusionIntegrator stiffness_integrator;
  int source_element = 0;
  double closest = INFINITY;
  for (int e = 0; e < ne; ++e)
  {
    sigma_space.GetElementDofs(e, elements[e].dofs);
    const auto* fe = sigma_space.GetFE(e);
    auto* transformation = mesh.GetElementTransformation(e);
    mass_integrator.AssembleElementMatrix(*fe, *transformation, elements[e].mass);
    stiffness_integrator.AssembleElementMatrix(*fe, *transformation, elements[e].stiffness);
    mesh.GetElementCenter(e, center);
    const double delta = std::pow(center(0) - (0.5 + shift_cells * h - 0.5 * h), 2) +
                         std::pow(center(1) - 0.5 * ny * h, 2);
    if (delta < closest) { closest = delta; source_element = e; }
  }
  std::vector<std::vector<int>> neighbor(ne), distance(ne, std::vector<int>(ne, ne));
  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int a, b;
    mesh.GetFaceElements(f, &a, &b);
    if (a >= 0 && b >= 0) { neighbor[a].push_back(b); neighbor[b].push_back(a); }
  }
  for (int e = 0; e < ne; ++e)
  {
    std::queue<int> queue;
    distance[e][e] = 0;
    queue.push(e);
    while (!queue.empty())
    {
      const int a = queue.front(); queue.pop();
      for (int b : neighbor[a]) if (distance[e][b] == ne)
      { distance[e][b] = distance[e][a] + 1; queue.push(b); }
    }
  }

  std::vector<mfem::Vector> loads(3), references(3);
  const std::vector<std::string> names{"closure", "cell-constant", "cell-high-mode"};
  for (auto& load : loads) { load.SetSize(n); load = 0.0; }
  loads[0] = igr.GetLinearRHS();
  const auto& source = elements[source_element];
  const auto& sigma_nodes = sigma_space.GetFE(source_element)->GetNodes();
  mfem::Vector coefficients(source.dofs.Size()), local_load(source.dofs.Size());
  for (int kind = 1; kind <= 2; ++kind)
  {
    for (int i = 0; i < coefficients.Size(); ++i)
      coefficients(i) = kind == 1 ? 1.0 : Legendre(order, 2.0 * sigma_nodes.IntPoint(i).x - 1.0);
    source.mass.Mult(coefficients, local_load);
    for (int i = 0; i < source.dofs.Size(); ++i) loads[kind](source.dofs[i]) = local_load(i);
  }
  // Norm of the L2-projected forcing, including near-zero aligned-jump sources.
  mfem::Vector source_projection(n);
  for (const auto& element : elements)
  {
    mfem::DenseMatrixInverse mass_inverse(element.mass);
    mfem::Vector local(element.dofs.Size()), projected(element.dofs.Size());
    for (int i = 0; i < local.Size(); ++i) local(i) = loads[0](element.dofs[i]);
    mass_inverse.Mult(local, projected);
    for (int i = 0; i < local.Size(); ++i) source_projection(element.dofs[i]) = projected(i);
  }
  std::cout << std::setprecision(16)
            << "[igr-screened] event=setup state=" << state_kind
            << " nx=" << nx << " ny=" << ny << " p=" << order << " n=" << n
            << " c_alpha=" << c_alpha << " h=" << h << " alpha=" << alpha
            << " width_cells=" << shock_width_cells << " shift_cells=" << shift_cells
            << " source_element=" << source_element
            << " boundary=natural sigma=L2_GL state_basis=LGL gradient=BrokenDG"
            << " source_projection_l2=" << PhysicalNorm(elements, source_projection)
            << " assembly_seconds=" << assembly_seconds
            << " operator_samples=" << n << " sampling_seconds=" << sampling_seconds
            << " reference_factor_seconds=" << factor_seconds
            << " reference_explicit_bytes=" << 2LL * n * n * sizeof(double)
            << " timing=host_probe_only\n";
  for (int kind = 0; kind < 3; ++kind)
  {
    references[kind].SetSize(n);
    reference_inverse.Mult(loads[kind], references[kind]);
    op->Mult(references[kind], action);
    action -= loads[kind];
    MFEM_VERIFY(action.Norml2() <= 1.0e-9 * loads[kind].Norml2() + 1.0e-26,
                "FP64 reference failed true current-operator residual gate.");
    ReportError("reference", names[kind], -2, 0, 0.0, *op, elements,
                 loads[kind], references[kind], references[kind]);
    if (kind == 0) continue;
    const double norm = PhysicalNorm(elements, references[kind]);
    for (int radius = 0; radius <= radius_max; ++radius)
    {
      double tail = 0.0;
      for (int e = 0; e < ne; ++e)
        if (distance[source_element][e] > radius)
          tail += ElementSquare(elements[e], references[kind], false);
      std::cout << "[igr-screened] event=green-tail load=" << names[kind]
                << " radius=" << radius << " l2_tail_abs=" << std::sqrt(tail)
                << " l2_tail_rel=" << Ratio(std::sqrt(tail), norm) << '\n';
    }
  }

  for (int radius = -1; radius <= radius_max; ++radius)
  {
    const auto patch_start = Clock::now();
    PatchInverse preconditioner(dense, elements, distance, radius);
    std::cout << "[igr-screened] event=patch-setup radius=" << radius
              << " method=" << (radius < 0 ? "point-jacobi" : "restricted-patch-inverse")
              << " setup_seconds=" << Seconds(patch_start)
              << " stored_bytes=" << preconditioner.storage_bytes
              << " setup_dense_entries=" << preconditioner.setup_dense_entries
              << " max_patch_dofs=" << preconditioner.max_patch_dofs << '\n';
    for (int kind = 0; kind < 3; ++kind)
    {
      if (loads[kind].Norml2() < 1.0e-24)
      {
        std::cout << "[igr-screened] event=zero-load load=" << names[kind]
                  << " radius=" << radius << '\n';
        continue;
      }
      mfem::Vector x(n), residual(n), correction(n);
      x = 0.0;
      const auto sweeps_start = Clock::now();
      for (int sweep = 1; sweep <= 16; ++sweep)
      {
        op->Mult(x, residual);
        residual *= -1.0;
        residual += loads[kind];
        preconditioner.Mult(residual, correction);
        x.Add(damping, correction);
        if ((sweep & (sweep - 1)) == 0)
          ReportError("stationary", names[kind], radius, sweep, damping, *op,
                       elements, loads[kind], references[kind], x);
      }
      const double sweep_seconds = Seconds(sweeps_start);
      mfem::FGMRESSolver solver(MPI_COMM_WORLD);
      solver.SetOperator(*op);
      solver.SetPreconditioner(preconditioner);
      solver.SetRelTol(1.0e-8);
      solver.SetAbsTol(0.0);
      solver.SetMaxIter(max_iterations);
      solver.SetKDim(std::min(60, max_iterations));
      solver.SetPrintLevel(0);
      solver.iterative_mode = false;
      x = 0.0;
      const auto krylov_start = Clock::now();
      solver.Mult(loads[kind], x);
      const double krylov_seconds = Seconds(krylov_start);
      ReportError("fgmres", names[kind], radius, solver.GetNumIterations(), 0.0,
                   *op, elements, loads[kind], references[kind], x);
      std::cout << "[igr-screened] event=solve-time radius=" << radius
                << " load=" << names[kind] << " converged=" << solver.GetConverged()
                << " stationary_seconds=" << sweep_seconds
                << " fgmres_seconds=" << krylov_seconds << '\n';
    }
    std::cout << "[igr-screened] event=patch-total radius=" << radius
              << " calls=" << preconditioner.calls
              << " apply_seconds=" << preconditioner.apply_seconds << '\n';
  }
  std::cout << "[igr-screened] event=complete host_seconds=" << Seconds(total_start)
            << " claim=frozen_operator_screen_only\n";
  return 0;
}
