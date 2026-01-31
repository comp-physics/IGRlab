#include "numerics/limiters/bj.hpp"
#include "numerics/limiters/krivodonova.hpp"
#include "numerics/limiters/tvb.hpp"
#include "numerics/limiters/venkat.hpp"
#include "numerics/limiters/zhang_shu.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "physics/euler/conversions.hpp"
#include "physics/euler/eos.hpp"
#include "test_utils.hpp"

namespace dg_igr_tests
{

namespace
{
constexpr int kDim = 2;
constexpr int kNumEq = kDim + 2;
constexpr int kNx = 3;
constexpr int kNy = 3;
constexpr double kGamma = 1.4;
constexpr double kPi = 3.14159265358979323846;
constexpr double kBoundTol = 1.0e-2;

double GlobalSum(double local)
{
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM, MPI_COMM_WORLD);
  return global;
}

double GlobalMin(double local)
{
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN, MPI_COMM_WORLD);
  return global;
}

double GlobalMax(double local)
{
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, mfem::MPITypeMap<double>::mpi_type,
                MPI_MAX, MPI_COMM_WORLD);
  return global;
}

bool GlobalAll(bool local_ok)
{
  int local = local_ok ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, mfem::MPITypeMap<int>::mpi_type,
                MPI_LAND, MPI_COMM_WORLD);
  return global == 1;
}

mfem::Vector ConservativeFromPrimitive(double rho, double u, double v, double p)
{
  mfem::Vector prim(kNumEq);
  prim(0) = rho;
  prim(1) = u;
  prim(2) = v;
  prim(3) = p;

  mfem::Vector cons(kNumEq);
  mfem::EulerPrimToConservative(prim, kDim, kGamma, cons);
  return cons;
}

void FillConstantState(mfem::ParFiniteElementSpace& vfes,
                       const mfem::Vector& cons,
                       mfem::ParGridFunction& U)
{
  mfem::VectorConstantCoefficient coeff(cons);
  U.ProjectCoefficient(coeff);
}

void FillPerturbedState(mfem::ParFiniteElementSpace& vfes, mfem::ParGridFunction& U)
{
  class PerturbedConservativeCoefficient final : public mfem::VectorCoefficient
  {
  public:
    PerturbedConservativeCoefficient() : mfem::VectorCoefficient(kNumEq) {}

    void Eval(mfem::Vector& V,
              mfem::ElementTransformation& Tr,
              const mfem::IntegrationPoint& ip) override
    {
      mfem::Vector x(kDim);
      Tr.Transform(ip, x);

      const double sx = std::sin(2.0 * kPi * x(0));
      const double sy = std::sin(2.0 * kPi * x(1));
      const double bump = 0.1 * sx * sy;

      const double rho = 1.0 + bump;
      const double u = 0.5 + 0.1 * (x(0) - 0.5);
      const double v = -0.25 + 0.1 * (x(1) - 0.5);
      const double p = 1.0 + 0.05 * bump;

      mfem::Vector cons = ConservativeFromPrimitive(rho, u, v, p);
      V = cons;
    }
  };

  PerturbedConservativeCoefficient coeff;
  U.ProjectCoefficient(coeff);
}

bool AllFinite(const mfem::ParGridFunction& U)
{
  bool local_ok = true;
  for (int i = 0; i < U.Size(); ++i)
  {
    if (!std::isfinite(U(i)))
    {
      local_ok = false;
      break;
    }
  }
  return GlobalAll(local_ok);
}

double GlobalRelativeChange(const mfem::ParGridFunction& before,
                            const mfem::ParGridFunction& after)
{
  mfem::Vector diff(after.Size());
  diff = after;
  diff -= before;

  const double local_diff_sq = diff * diff;
  const double local_base_sq = before * before;
  const double diff_sq = GlobalSum(local_diff_sq);
  const double base_sq = GlobalSum(local_base_sq);

  const double diff_norm = std::sqrt(diff_sq);
  const double base_norm = std::sqrt(base_sq);
  return diff_norm / std::max(base_norm, 1.0e-16);
}

std::vector<mfem::Vector> ComputeElementAverages(mfem::ParFiniteElementSpace& vfes,
                                                 const mfem::ParGridFunction& U,
                                                 int quad_order)
{
  const int ne = vfes.GetNE();
  std::vector<mfem::Vector> averages(ne);

  mfem::Array<int> vdofs;
  mfem::Vector elvec;
  mfem::DenseMatrix elmat;
  mfem::Vector shape;
  mfem::Vector state(kNumEq);

  for (int e = 0; e < ne; ++e)
  {
    const mfem::FiniteElement* fe = vfes.GetFE(e);
    mfem::ElementTransformation* Tr = vfes.GetElementTransformation(e);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(e, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, kNumEq);

    const int qord = (quad_order > 0) ? quad_order : (2 * fe->GetOrder() + 2);
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(Tr->GetGeometryType(), qord);
    shape.SetSize(dof);

    mfem::Vector sum(kNumEq);
    sum = 0.0;
    double volume = 0.0;

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, state);

      const double w = ip.weight * Tr->Weight();
      sum.Add(w, state);
      volume += w;
    }

    averages[e].SetSize(kNumEq);
    averages[e] = sum;
    averages[e] /= std::max(volume, 1.0e-16);
  }

  return averages;
}

double MaxRelativeAverageChange(const std::vector<mfem::Vector>& before,
                                const std::vector<mfem::Vector>& after)
{
  double local_max = 0.0;
  for (int e = 0; e < static_cast<int>(before.size()); ++e)
  {
    mfem::Vector diff(before[e]);
    diff -= after[e];
    const double diff_norm = diff.Norml2();
    const double base_norm = before[e].Norml2();
    const double rel = diff_norm / std::max(base_norm, 1.0e-16);
    local_max = std::max(local_max, rel);
  }
  return GlobalMax(local_max);
}

struct ElementBounds
{
  double rho_min = 0.0;
  double rho_max = 0.0;
  double energy_min = 0.0;
  double energy_max = 0.0;
};

std::vector<ElementBounds> BuildStructuredNeighborBounds(
    mfem::ParFiniteElementSpace& vfes,
    const std::vector<mfem::Vector>& averages,
    int nx,
    int ny)
{
  const int ne = vfes.GetNE();
  std::vector<ElementBounds> bounds(ne);

  const double hx = 1.0 / static_cast<double>(nx);
  const double hy = 1.0 / static_cast<double>(ny);

  std::vector<int> elem_ix(ne, 0);
  std::vector<int> elem_iy(ne, 0);
  std::vector<int> grid_to_elem(nx * ny, -1);

  mfem::IntegrationPoint center_ip;
  center_ip.Set2(0.0, 0.0);
  mfem::Vector center(kDim);

  for (int e = 0; e < ne; ++e)
  {
    mfem::ElementTransformation* Tr = vfes.GetElementTransformation(e);
    Tr->Transform(center_ip, center);

    int ix = static_cast<int>(std::floor(center(0) / hx));
    int iy = static_cast<int>(std::floor(center(1) / hy));
    ix = std::max(0, std::min(nx - 1, ix));
    iy = std::max(0, std::min(ny - 1, iy));

    elem_ix[e] = ix;
    elem_iy[e] = iy;
    grid_to_elem[iy * nx + ix] = e;
  }

  auto elem_at = [&](int ix, int iy, int self) -> int
  {
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny)
    {
      return self;
    }
    const int idx = iy * nx + ix;
    const int elem = grid_to_elem[idx];
    return (elem >= 0) ? elem : self;
  };

  std::vector<double> avg_rho(ne, 0.0);
  std::vector<double> avg_energy(ne, 0.0);
  for (int e = 0; e < ne; ++e)
  {
    avg_rho[e] = averages[e](0);
    avg_energy[e] = averages[e](kDim + 1);
  }

  for (int e = 0; e < ne; ++e)
  {
    const int ix = elem_ix[e];
    const int iy = elem_iy[e];
    const std::array<int, 5> nbrs = {
        e,
        elem_at(ix - 1, iy, e),
        elem_at(ix + 1, iy, e),
        elem_at(ix, iy - 1, e),
        elem_at(ix, iy + 1, e)};

    ElementBounds b;
    b.rho_min = std::numeric_limits<double>::infinity();
    b.rho_max = -std::numeric_limits<double>::infinity();
    b.energy_min = std::numeric_limits<double>::infinity();
    b.energy_max = -std::numeric_limits<double>::infinity();

    for (int nbr : nbrs)
    {
      b.rho_min = std::min(b.rho_min, avg_rho[nbr]);
      b.rho_max = std::max(b.rho_max, avg_rho[nbr]);
      b.energy_min = std::min(b.energy_min, avg_energy[nbr]);
      b.energy_max = std::max(b.energy_max, avg_energy[nbr]);
    }

    bounds[e] = b;
  }

  return bounds;
}

double MaxBoundViolation(mfem::ParFiniteElementSpace& vfes,
                         const mfem::ParGridFunction& U,
                         int quad_order,
                         const std::vector<ElementBounds>& bounds)
{
  double local_max = 0.0;

  mfem::Array<int> vdofs;
  mfem::Vector elvec;
  mfem::DenseMatrix elmat;
  mfem::Vector shape;
  mfem::Vector state(kNumEq);

  const int ne = vfes.GetNE();
  for (int e = 0; e < ne; ++e)
  {
    const mfem::FiniteElement* fe = vfes.GetFE(e);
    mfem::ElementTransformation* Tr = vfes.GetElementTransformation(e);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(e, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, kNumEq);

    const int qord = (quad_order > 0) ? quad_order : (2 * fe->GetOrder() + 2);
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(Tr->GetGeometryType(), qord);
    shape.SetSize(dof);

    const ElementBounds& b = bounds[e];
    const double rho_range = std::max(b.rho_max - b.rho_min, 1.0e-12);
    const double energy_range = std::max(b.energy_max - b.energy_min, 1.0e-12);
    const double rho_scale = std::max({std::abs(b.rho_min), std::abs(b.rho_max), rho_range, 1.0e-6});
    const double energy_scale =
        std::max({std::abs(b.energy_min), std::abs(b.energy_max), energy_range, 1.0e-6});
    const double rho_inflate = 0.75 * rho_range + 0.05 * rho_scale + 1.0e-10;
    const double energy_inflate = 0.75 * energy_range + 0.05 * energy_scale + 1.0e-10;
    const double rho_lo = b.rho_min - rho_inflate;
    const double rho_hi = b.rho_max + rho_inflate;
    const double energy_lo = b.energy_min - energy_inflate;
    const double energy_hi = b.energy_max + energy_inflate;

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, state);

      const double rho = state(0);
      const double energy = state(kDim + 1);

      const double rho_hi_v = std::max(0.0, (rho - rho_hi) / rho_scale);
      const double rho_lo_v = std::max(0.0, (rho_lo - rho) / rho_scale);
      const double energy_hi_v = std::max(0.0, (energy - energy_hi) / energy_scale);
      const double energy_lo_v = std::max(0.0, (energy_lo - energy) / energy_scale);

      local_max = std::max(local_max, rho_hi_v);
      local_max = std::max(local_max, rho_lo_v);
      local_max = std::max(local_max, energy_hi_v);
      local_max = std::max(local_max, energy_lo_v);
    }
  }

  return GlobalMax(local_max);
}

void MinRhoPressure(mfem::ParFiniteElementSpace& vfes,
                    const mfem::ParGridFunction& U,
                    int quad_order,
                    double& min_rho,
                    double& min_p)
{
  double local_min_rho = std::numeric_limits<double>::infinity();
  double local_min_p = std::numeric_limits<double>::infinity();

  mfem::Array<int> vdofs;
  mfem::Vector elvec;
  mfem::DenseMatrix elmat;
  mfem::Vector shape;
  mfem::Vector state(kNumEq);

  const int ne = vfes.GetNE();
  for (int e = 0; e < ne; ++e)
  {
    const mfem::FiniteElement* fe = vfes.GetFE(e);
    mfem::ElementTransformation* Tr = vfes.GetElementTransformation(e);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(e, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, kNumEq);

    const int qord = (quad_order > 0) ? quad_order : (2 * fe->GetOrder() + 2);
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(Tr->GetGeometryType(), qord);
    shape.SetSize(dof);

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, state);

      local_min_rho = std::min(local_min_rho, state(0));
      local_min_p = std::min(local_min_p, mfem::EulerPressureFromConservative(
                                              state, kDim, kGamma, /*rho_floor=*/1e-16));
    }
  }

  min_rho = GlobalMin(local_min_rho);
  min_p = GlobalMin(local_min_p);
}

void CheckLimiterOnConstantState(TestContext& ctx,
                                 const std::string& name,
                                 mfem::ParFiniteElementSpace& vfes,
                                 const mfem::ParGridFunction& constant_state,
                                 void (*apply)(mfem::ParFiniteElementSpace&,
                                               mfem::ParGridFunction&))
{
  mfem::ParGridFunction U(&vfes);
  U = constant_state;
  apply(vfes, U);

  const double rel = GlobalRelativeChange(constant_state, U);
  ctx.AddParity(rel);
  ctx.Check(rel <= 1e-12, name + " preserves constant state");
}

void CheckLimiterOnPerturbedState(TestContext& ctx,
                                  const std::string& name,
                                  mfem::ParFiniteElementSpace& vfes,
                                  void (*apply)(mfem::ParFiniteElementSpace&,
                                                mfem::ParGridFunction&))
{
  mfem::ParGridFunction U_initial(&vfes);
  FillPerturbedState(vfes, U_initial);

  const int order = vfes.GetFE(0)->GetOrder();
  const int qord = 2 * order + 2;

  const std::vector<mfem::Vector> averages_before =
      ComputeElementAverages(vfes, U_initial, qord);
  const std::vector<ElementBounds> bounds =
      BuildStructuredNeighborBounds(vfes, averages_before, kNx, kNy);
  const double violation_before = MaxBoundViolation(vfes, U_initial, qord, bounds);

  mfem::ParGridFunction U_limited(&vfes);
  U_limited = U_initial;
  apply(vfes, U_limited);

  const bool finite = AllFinite(U_limited);
  ctx.Check(finite, name + " keeps values finite");

  const std::vector<mfem::Vector> averages_after =
      ComputeElementAverages(vfes, U_limited, qord);
  const double avg_change = MaxRelativeAverageChange(averages_before, averages_after);
  ctx.AddParity(avg_change);
  ctx.Check(avg_change <= 1e-8, name + " preserves element averages");

  const double violation_after = MaxBoundViolation(vfes, U_limited, qord, bounds);
  ctx.Check(violation_after <= violation_before + 1e-8,
            name + " does not worsen neighbor-bound violations");
  if (mfem::Mpi::Root() && violation_after > kBoundTol)
  {
    std::cerr << "[INFO] " << name
              << " neighbor-bound violation before=" << violation_before
              << " after=" << violation_after << "\n";
  }
  ctx.Check(violation_after <= kBoundTol,
            name + " stays within inflated neighbor bounds");

  const double rho_min = 1e-10;
  const double p_min = 1e-10;
  mfem::ApplyZhangShuPositivityLimiter(vfes, U_limited, kDim, kGamma, rho_min, p_min,
                                       /*quad_order=*/qord, /*p_bisect_iters=*/12);

  double min_rho = 0.0;
  double min_p = 0.0;
  MinRhoPressure(vfes, U_limited, qord, min_rho, min_p);
  ctx.AddParity(min_rho + min_p);
  ctx.Check(min_rho >= 0.99 * rho_min, name + " + positivity keeps density positive");
  ctx.Check(min_p >= 0.99 * p_min, name + " + positivity keeps pressure positive");
}

} // namespace

int RunLimiterTests(TestContext& ctx)
{
  // --- Zhang-Shu positivity on order-0 space ---
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
        /*nx=*/2, /*ny=*/2, mfem::Element::QUADRILATERAL, /*generate_edges=*/true,
        /*sx=*/1.0, /*sy=*/1.0);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::DG_FECollection fec(/*order=*/0, kDim);
    mfem::ParFiniteElementSpace vfes(&pmesh, &fec, kNumEq, mfem::Ordering::byNODES);
    mfem::ParGridFunction U(&vfes);
    U = 0.0;

    const mfem::Vector cons = ConservativeFromPrimitive(/*rho=*/1.0, /*u=*/0.0,
                                                        /*v=*/0.0, /*p=*/1.0);
    FillConstantState(vfes, cons, U);

    if (vfes.GetNDofs() > 0)
    {
      U(0) = -1.0;
      U(3) = 0.1;
    }

    const double rho_min = 1e-8;
    const double p_min = 1e-8;
    mfem::ApplyZhangShuPositivityLimiter(vfes, U, kDim, kGamma, rho_min, p_min,
                                         /*quad_order=*/0, /*p_bisect_iters=*/12);

    double min_rho = 0.0;
    double min_p = 0.0;
    MinRhoPressure(vfes, U, /*quad_order=*/0, min_rho, min_p);
    ctx.Check(min_rho >= 0.99 * rho_min, "Zhang-Shu limiter enforces rho floor");
    ctx.Check(min_p >= 0.99 * p_min, "Zhang-Shu limiter enforces pressure floor");
  }

  // --- Slope limiters on order-1 space ---
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/kNx, /*ny=*/kNy, mfem::Element::QUADRILATERAL, /*generate_edges=*/true,
      /*sx=*/1.0, /*sy=*/1.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mfem::DG_FECollection fec(/*order=*/1, kDim);
  mfem::ParFiniteElementSpace vfes(&pmesh, &fec, kNumEq, mfem::Ordering::byNODES);

  const mfem::Vector cons_const =
      ConservativeFromPrimitive(/*rho=*/1.0, /*u=*/0.25, /*v=*/-0.1, /*p=*/1.0);
  mfem::ParGridFunction U_const(&vfes);
  U_const = 0.0;
  FillConstantState(vfes, cons_const, U_const);

  // Component-wise Barth-Jespersen
  CheckLimiterOnConstantState(
      ctx, "Barth-Jespersen (component)", vfes, U_const,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::BJOptions opt;
        opt.dim = kDim;
        mfem::ApplyBarthJespersenLimiter(space, U, opt);
      });
  CheckLimiterOnPerturbedState(
      ctx, "Barth-Jespersen (component)", vfes,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::BJOptions opt;
        opt.dim = kDim;
        mfem::ApplyBarthJespersenLimiter(space, U, opt);
      });

  // Characteristic Barth-Jespersen (x/y)
  CheckLimiterOnConstantState(
      ctx, "Barth-Jespersen (characteristic XY)", vfes, U_const,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::CharBJOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicBJLimiter(space, U, opt);
      });
  CheckLimiterOnPerturbedState(
      ctx, "Barth-Jespersen (characteristic XY)", vfes,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::CharBJOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicBJLimiter(space, U, opt);
      });

  // Face-normal characteristic Barth-Jespersen
  CheckLimiterOnConstantState(
      ctx, "Barth-Jespersen (face-normal characteristic)", vfes, U_const,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::FaceBJOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyFaceNormalCharacteristicBJLimiter(space, U, opt);
      });
  CheckLimiterOnPerturbedState(
      ctx, "Barth-Jespersen (face-normal characteristic)", vfes,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::FaceBJOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyFaceNormalCharacteristicBJLimiter(space, U, opt);
      });

  // Characteristic TVB (x/y)
  CheckLimiterOnConstantState(
      ctx, "TVB (characteristic XY)", vfes, U_const,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::CharTVBOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicTVBLimiter(space, U, opt);
      });
  CheckLimiterOnPerturbedState(
      ctx, "TVB (characteristic XY)", vfes,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::CharTVBOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicTVBLimiter(space, U, opt);
      });

  // TVB face-normal
  CheckLimiterOnConstantState(
      ctx, "TVB (face-normal characteristic)", vfes, U_const,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::CharTVBOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicTVBFaceNormal(space, U, opt);
      });
  CheckLimiterOnPerturbedState(
      ctx, "TVB (face-normal characteristic)", vfes,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::CharTVBOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicTVBFaceNormal(space, U, opt);
      });

  // Venkatakrishnan (x/y)
  CheckLimiterOnConstantState(
      ctx, "Venkatakrishnan (characteristic XY)", vfes, U_const,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::VenkatOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicVenkatXY(space, U, opt);
      });
  CheckLimiterOnPerturbedState(
      ctx, "Venkatakrishnan (characteristic XY)", vfes,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::VenkatOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicVenkatXY(space, U, opt);
      });

  // Venkatakrishnan face-normal
  CheckLimiterOnConstantState(
      ctx, "Venkatakrishnan (face-normal characteristic)", vfes, U_const,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::VenkatOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicVenkatFaceNormal(space, U, opt);
      });
  CheckLimiterOnPerturbedState(
      ctx, "Venkatakrishnan (face-normal characteristic)", vfes,
      [](mfem::ParFiniteElementSpace& space, mfem::ParGridFunction& U) {
        mfem::VenkatOptions opt;
        opt.dim = kDim;
        opt.gamma = kGamma;
        mfem::ApplyCharacteristicVenkatFaceNormal(space, U, opt);
      });

  // Krivodonova hierarchical limiter: enforce admissibility.
  {
    mfem::KrivoOptions opt;
    opt.dim = kDim;
    opt.gamma = kGamma;

    mfem::ParGridFunction U_krivo(&vfes);
    U_krivo = 0.0;
    FillPerturbedState(vfes, U_krivo);
    mfem::ApplyKrivodonovaHierarchicalLimiter(vfes, U_krivo, opt);

    const bool finite = AllFinite(U_krivo);
    const int qord = 2 * vfes.GetFE(0)->GetOrder() + 2;
    double min_rho = 0.0;
    double min_p = 0.0;
    MinRhoPressure(vfes, U_krivo, qord, min_rho, min_p);
    ctx.Check(finite, "Krivodonova limiter keeps values finite");
    ctx.Check(min_rho > 0.0, "Krivodonova limiter keeps density positive");
    ctx.Check(min_p > 0.0, "Krivodonova limiter keeps pressure positive");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests
