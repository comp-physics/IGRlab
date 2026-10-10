#pragma once

#include "igrlab/environment.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "mfem.hpp"
#include "physics/euler/conversions.hpp"

namespace dg_igr_tests
{

inline bool EnvEnabled(const char* name)
{
  const char* env = igrlab::GetEnv(name);
  if (!env)
  {
    return false;
  }
  const std::string value(env);
  return value == "1" || value == "true" || value == "TRUE" || value == "on";
}

inline bool EnvEnabledDefault(const char* name, bool default_value)
{
  const char* env = igrlab::GetEnv(name);
  if (!env)
  {
    return default_value;
  }
  const std::string value(env);
  if (value == "0" || value == "false" || value == "FALSE" || value == "off" ||
      value == "OFF")
  {
    return false;
  }
  if (value == "1" || value == "true" || value == "TRUE" || value == "on" ||
      value == "ON")
  {
    return true;
  }
  return default_value;
}

inline std::string FormatSci(double value, int precision = 12)
{
  std::ostringstream os;
  os.setf(std::ios::scientific);
  os << std::setprecision(precision) << value;
  return os.str();
}

struct TestContext
{
  int failures = 0;
  bool parity_mode = false;
  double parity_metric = 0.0;
  bool verbose_metrics = true;

  TestContext()
    : verbose_metrics(!EnvEnabled("DG_IGR_TEST_QUIET"))
  {
  }

  void Check(bool cond, const std::string& msg)
  {
    if (!cond)
    {
      ++failures;
      std::cerr << "[FAIL] " << msg << "\n";
    }
    else if (verbose_metrics)
    {
      std::cout << "[PASS] " << msg << "\n";
    }
  }

  void CheckNear(double a, double b, double tol, const std::string& msg)
  {
    const double err = std::abs(a - b);
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    if (err > tol * scale)
    {
      ++failures;
      std::cerr << "[FAIL] " << msg << " (a=" << FormatSci(a)
                << ", b=" << FormatSci(b) << ", err=" << FormatSci(err)
                << ", tol=" << FormatSci(tol) << ")\n";
    }
    else if (verbose_metrics)
    {
      std::cout << "[PASS] " << msg << " (a=" << FormatSci(a)
                << ", b=" << FormatSci(b) << ", err=" << FormatSci(err)
                << ", tol=" << FormatSci(tol) << ")\n";
    }
  }

  void AddParity(double value) { parity_metric += value; }

  void LogMetric(const std::string& msg)
  {
    if (verbose_metrics)
    {
      std::cout << "[METRIC] " << msg << "\n";
    }
  }
};

inline mfem::Vector
EulerConsFromPrim1D(double rho, double u, double p, double gamma)
{
  mfem::Vector prim(3);
  prim(0) = rho;
  prim(1) = u;
  prim(2) = p;

  mfem::Vector cons(3);
  mfem::EulerPrimToConservative(prim, /*dim=*/1, gamma, cons);
  return cons;
}

inline mfem::Vector
EulerConsFromPrim2D(double rho, double u, double v, double p, double gamma)
{
  mfem::Vector prim(4);
  prim(0) = rho;
  prim(1) = u;
  prim(2) = v;
  prim(3) = p;

  mfem::Vector cons(4);
  mfem::EulerPrimToConservative(prim, /*dim=*/2, gamma, cons);
  return cons;
}

inline mfem::Vector Rotate2D(const mfem::Vector& v, double angle)
{
  MFEM_VERIFY(v.Size() == 2, "Rotate2D expects a 2D vector.");
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  mfem::Vector out(2);
  out(0) = c * v(0) - s * v(1);
  out(1) = s * v(0) + c * v(1);
  return out;
}

inline void FillConstantState(const mfem::Vector& state,
                              mfem::ParGridFunction& gf)
{
  mfem::VectorConstantCoefficient coeff(state);
  gf.ProjectCoefficient(coeff);
}

inline void FillConstantState(const mfem::Vector& state, mfem::GridFunction& gf)
{
  mfem::VectorConstantCoefficient coeff(state);
  gf.ProjectCoefficient(coeff);
}

inline double GlobalSum(double local)
{
  double global = 0.0;
  MPI_Allreduce(&local,
                &global,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM,
                MPI_COMM_WORLD);
  return global;
}

inline double GlobalL2Norm(const mfem::Vector& v)
{
  const double local_sq = v * v;
  const double global_sq = GlobalSum(local_sq);
  return std::sqrt(global_sq);
}

inline double GlobalMax(double local)
{
  double global = local;
  MPI_Allreduce(&local,
                &global,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MAX,
                MPI_COMM_WORLD);
  return global;
}

inline double GlobalMin(double local)
{
  double global = local;
  MPI_Allreduce(&local,
                &global,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN,
                MPI_COMM_WORLD);
  return global;
}

inline double GlobalAverage(double local)
{
  const double global_sum = GlobalSum(local);
  const int world_size = mfem::Mpi::WorldSize();
  return global_sum / static_cast<double>(world_size > 0 ? world_size : 1);
}

inline double
IntegrateComponent(const mfem::GridFunction& U, int comp, int order_offset = 2)
{
  const mfem::FiniteElementSpace* fes = U.FESpace();
  MFEM_VERIFY(fes != nullptr, "IntegrateComponent requires a valid FE space.");
  MFEM_VERIFY(comp >= 0 && comp < U.VectorDim(),
              "IntegrateComponent component index out of range.");

  double local = 0.0;
  mfem::Vector value(U.VectorDim());
  const int ne = fes->GetNE();
  for (int e = 0; e < ne; ++e)
  {
    const mfem::FiniteElement* fe = fes->GetFE(e);
    mfem::ElementTransformation* Tr = fes->GetElementTransformation(e);
    const int qord = 2 * fe->GetOrder() + order_offset;
    const mfem::IntegrationRule& ir =
      mfem::IntRules.Get(fe->GetGeomType(), qord);

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      U.GetVectorValue(*Tr, ip, value);
      local += value(comp) * ip.weight * Tr->Weight();
    }
  }

  return GlobalSum(local);
}

} // namespace igrlab_tests
