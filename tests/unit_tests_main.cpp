#include "mfem.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace dg_igr_tests
{
int RunEulerTests(TestContext& ctx);
int RunFluxTests(TestContext& ctx);
int RunLimiterTests(TestContext& ctx);
int RunDGOperatorTests(TestContext& ctx);
int RunBCTests(TestContext& ctx);
int RunIGRTests(TestContext& ctx);
int RunDGSEMBasisTests(TestContext& ctx);
int RunDGSEMFluxTests(TestContext& ctx);
int RunDGSEMOperatorTests(TestContext& ctx);
int RunDGSEMIGRTests(TestContext& ctx);
}

namespace
{

struct SuiteSelection
{
  bool euler = false;
  bool fluxes = false;
  bool limiters = false;
  bool dg_operator = false;
  bool dgsem_basis = false;
  bool dgsem_flux = false;
  bool dgsem_operator = false;
  bool dgsem_igr = false;
  bool bc = false;
  bool igr = false;
  bool parity = false;
  bool list_only = false;
  bool help = false;
  bool ok = true;
  std::string error;
};

void PrintUsage()
{
  std::cout << "dg_igr_unit_tests usage:\n"
            << "  --all        Run all suites (default).\n"
            << "  --euler      Run Euler EOS/conversion tests.\n"
            << "  --fluxes     Run numerical flux tests.\n"
            << "  --limiters   Run limiter tests.\n"
            << "  --dg-operator Run DG operator invariants.\n"
            << "  --dgsem-basis Run DGSEM basis/ordering tests.\n"
            << "  --dgsem-flux  Run DGSEM flux/volume tests.\n"
            << "  --dgsem-operator Run DGSEM operator surface/volume tests.\n"
            << "  --dgsem-igr  Run DGSEM + IGR tests.\n"
            << "  --bc         Run boundary-state policy tests.\n"
            << "  --igr        Run IGR flux decorator tests.\n"
            << "  --parity     Emit/compare MPI parity signatures (uses DG_IGR_PARITY_* env vars).\n"
            << "  --list       List available suites and exit.\n"
            << "  --help       Show this help and exit.\n";
}

SuiteSelection ParseSuites(int argc, char** argv)
{
  SuiteSelection sel;
  bool any_suite_flag = false;

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);
    if (arg == "--all")
    {
      sel.euler = true;
      sel.fluxes = true;
      sel.limiters = true;
      sel.dg_operator = true;
      sel.dgsem_basis = true;
      sel.dgsem_flux = true;
      sel.dgsem_operator = true;
      sel.dgsem_igr = true;
      sel.bc = true;
      sel.igr = true;
      any_suite_flag = true;
    }
    else if (arg == "--euler")
    {
      sel.euler = true;
      any_suite_flag = true;
    }
    else if (arg == "--fluxes")
    {
      sel.fluxes = true;
      any_suite_flag = true;
    }
    else if (arg == "--limiters")
    {
      sel.limiters = true;
      any_suite_flag = true;
    }
    else if (arg == "--dg-operator")
    {
      sel.dg_operator = true;
      any_suite_flag = true;
    }
    else if (arg == "--dgsem-basis")
    {
      sel.dgsem_basis = true;
      any_suite_flag = true;
    }
    else if (arg == "--dgsem-flux")
    {
      sel.dgsem_flux = true;
      any_suite_flag = true;
    }
    else if (arg == "--dgsem-operator")
    {
      sel.dgsem_operator = true;
      any_suite_flag = true;
    }
    else if (arg == "--dgsem-igr")
    {
      sel.dgsem_igr = true;
      any_suite_flag = true;
    }
    else if (arg == "--bc")
    {
      sel.bc = true;
      any_suite_flag = true;
    }
    else if (arg == "--igr")
    {
      sel.igr = true;
      any_suite_flag = true;
    }
    else if (arg == "--parity")
    {
      sel.parity = true;
    }
    else if (arg == "--list")
    {
      sel.list_only = true;
    }
    else if (arg == "--help" || arg == "-h")
    {
      sel.help = true;
    }
    else
    {
      sel.ok = false;
      sel.error = "Unknown argument: " + arg;
      return sel;
    }
  }

  if (!any_suite_flag && !sel.list_only && !sel.help)
  {
    sel.euler = true;
    sel.fluxes = true;
    sel.limiters = true;
    sel.dg_operator = true;
    sel.dgsem_basis = true;
    sel.dgsem_flux = true;
    sel.dgsem_operator = true;
    sel.dgsem_igr = true;
    sel.bc = true;
    sel.igr = true;
  }

  return sel;
}

int AllreduceSum(int local)
{
  int global = 0;
  MPI_Allreduce(&local, &global, 1, mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM, MPI_COMM_WORLD);
  return global;
}

double AllreduceSumDouble(double local)
{
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, mfem::MPITypeMap<double>::mpi_type,
                MPI_SUM, MPI_COMM_WORLD);
  return global;
}

bool GetEnvString(const char* name, std::string& value)
{
  const char* env = std::getenv(name);
  if (!env) { return false; }
  value = env;
  return true;
}

bool EnvEnabled(const char* name)
{
  std::string value;
  if (!GetEnvString(name, value)) { return false; }
  return value == "1" || value == "true" || value == "TRUE" || value == "on";
}

double ParityTolerance()
{
  std::string tol_str;
  if (!GetEnvString("DG_IGR_PARITY_TOL", tol_str)) { return 1.0e-9; }
  std::istringstream is(tol_str);
  double tol = 1.0e-9;
  is >> tol;
  return tol;
}

using MetricMap = std::map<std::string, double>;

bool WriteParityReference(const std::string& path, const MetricMap& metrics)
{
  std::ofstream ofs(path);
  if (!ofs) { return false; }
  ofs << "# dg_igr MPI parity reference\n";
  for (const auto& kv : metrics)
  {
    ofs << kv.first << "=" << kv.second << "\n";
  }
  return true;
}

bool ReadParityReference(const std::string& path, MetricMap& metrics_out)
{
  std::ifstream ifs(path);
  if (!ifs) { return false; }

  std::string line;
  while (std::getline(ifs, line))
  {
    if (line.empty() || line[0] == '#') { continue; }
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) { continue; }

    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);

    std::istringstream is(value);
    double parsed = 0.0;
    is >> parsed;
    metrics_out[key] = parsed;
  }
  return true;
}

void CompareParityMetrics(const MetricMap& ref,
                          const MetricMap& cur,
                          double tol,
                          dg_igr_tests::TestContext& ctx)
{
  for (const auto& kv : cur)
  {
    const auto it = ref.find(kv.first);
    if (it == ref.end())
    {
      ctx.Check(false, "Parity reference missing suite: " + kv.first);
      continue;
    }
    ctx.CheckNear(kv.second, it->second, tol,
                  "MPI parity signature for suite " + kv.first);
  }
}

} // namespace

int main(int argc, char** argv)
{
  const SuiteSelection selection = ParseSuites(argc, argv);
  if (!selection.ok)
  {
    std::cerr << selection.error << "\n";
    PrintUsage();
    return 2;
  }
  if (selection.help)
  {
    PrintUsage();
    return 0;
  }
  if (selection.list_only)
  {
    std::cout << "Available suites: euler, fluxes, limiters, dg-operator, dgsem-basis, "
                 "dgsem-flux, dgsem-operator, dgsem-igr, bc, igr\n";
    return 0;
  }

  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();

  dg_igr_tests::TestContext ctx;
  ctx.parity_mode = selection.parity;
  int local_euler_failures = 0;
  int local_flux_failures = 0;
  int local_limiter_failures = 0;
  int local_dg_operator_failures = 0;
  int local_dgsem_basis_failures = 0;
  int local_dgsem_flux_failures = 0;
  int local_dgsem_operator_failures = 0;
  int local_dgsem_igr_failures = 0;
  int local_bc_failures = 0;
  int local_igr_failures = 0;

  const int world_size = mfem::Mpi::WorldSize();
  MetricMap parity_metrics;
  auto capture_parity_metric = [&](const char* suite_name,
                                   double before_metric) {
    const double local_metric = ctx.parity_metric - before_metric;
    const double global_sum = AllreduceSumDouble(local_metric);
    const double global_avg =
        global_sum / static_cast<double>(world_size > 0 ? world_size : 1);
    parity_metrics[suite_name] = global_avg;
  };

  if (selection.euler)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunEulerTests(ctx);
    local_euler_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("euler", parity_before); }
  }
  if (selection.fluxes)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunFluxTests(ctx);
    local_flux_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("fluxes", parity_before); }
  }
  if (selection.limiters)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunLimiterTests(ctx);
    local_limiter_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("limiters", parity_before); }
  }
  if (selection.dg_operator)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunDGOperatorTests(ctx);
    local_dg_operator_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("dg-operator", parity_before); }
  }
  if (selection.dgsem_basis)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunDGSEMBasisTests(ctx);
    local_dgsem_basis_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("dgsem-basis", parity_before); }
  }
  if (selection.dgsem_flux)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunDGSEMFluxTests(ctx);
    local_dgsem_flux_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("dgsem-flux", parity_before); }
  }
  if (selection.dgsem_operator)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunDGSEMOperatorTests(ctx);
    local_dgsem_operator_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("dgsem-operator", parity_before); }
  }
  if (selection.dgsem_igr)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunDGSEMIGRTests(ctx);
    local_dgsem_igr_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("dgsem-igr", parity_before); }
  }
  if (selection.bc)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunBCTests(ctx);
    local_bc_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("bc", parity_before); }
  }
  if (selection.igr)
  {
    const int before = ctx.failures;
    const double parity_before = ctx.parity_metric;
    dg_igr_tests::RunIGRTests(ctx);
    local_igr_failures = ctx.failures - before;
    if (selection.parity) { capture_parity_metric("igr", parity_before); }
  }

  const int global_euler_failures = selection.euler ? AllreduceSum(local_euler_failures) : 0;
  const int global_flux_failures = selection.fluxes ? AllreduceSum(local_flux_failures) : 0;
  const int global_limiter_failures = selection.limiters ? AllreduceSum(local_limiter_failures) : 0;
  const int global_dg_operator_failures =
      selection.dg_operator ? AllreduceSum(local_dg_operator_failures) : 0;
  const int global_dgsem_basis_failures =
      selection.dgsem_basis ? AllreduceSum(local_dgsem_basis_failures) : 0;
  const int global_dgsem_flux_failures =
      selection.dgsem_flux ? AllreduceSum(local_dgsem_flux_failures) : 0;
  const int global_dgsem_operator_failures =
      selection.dgsem_operator ? AllreduceSum(local_dgsem_operator_failures) : 0;
  const int global_dgsem_igr_failures =
      selection.dgsem_igr ? AllreduceSum(local_dgsem_igr_failures) : 0;
  const int global_bc_failures = selection.bc ? AllreduceSum(local_bc_failures) : 0;
  const int global_igr_failures = selection.igr ? AllreduceSum(local_igr_failures) : 0;

  if (selection.parity && mfem::Mpi::Root())
  {
    std::string parity_ref_path = "dg_igr_parity_ref.txt";
    GetEnvString("DG_IGR_PARITY_REF", parity_ref_path);
    const bool parity_write = EnvEnabled("DG_IGR_PARITY_WRITE");
    const double parity_tol = ParityTolerance();

    if (parity_write)
    {
      const bool wrote = WriteParityReference(parity_ref_path, parity_metrics);
      ctx.Check(wrote, "Failed to write parity reference: " + parity_ref_path);
      if (wrote)
      {
        std::cout << "Wrote MPI parity reference: " << parity_ref_path << "\n";
      }
    }
    else
    {
      MetricMap parity_ref;
      const bool read_ok = ReadParityReference(parity_ref_path, parity_ref);
      ctx.Check(read_ok, "Failed to read parity reference: " + parity_ref_path);
      if (read_ok)
      {
        CompareParityMetrics(parity_ref, parity_metrics, parity_tol, ctx);
      }
    }
  }

  int global_failures = 0;
  MPI_Allreduce(&ctx.failures, &global_failures, 1,
                mfem::MPITypeMap<int>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

  if (mfem::Mpi::Root())
  {

    std::vector<std::string> suites;
    if (selection.euler) { suites.emplace_back("euler"); }
    if (selection.fluxes) { suites.emplace_back("fluxes"); }
    if (selection.limiters) { suites.emplace_back("limiters"); }
    if (selection.dg_operator) { suites.emplace_back("dg-operator"); }
    if (selection.bc) { suites.emplace_back("bc"); }
    if (selection.igr) { suites.emplace_back("igr"); }
    if (selection.dgsem_flux) { suites.emplace_back("dgsem-flux"); }
    if (selection.dgsem_operator) { suites.emplace_back("dgsem-operator"); }
    if (selection.dgsem_basis) { suites.emplace_back("dgsem-basis"); }
    if (selection.dgsem_igr) { suites.emplace_back("dgsem-igr"); }

    std::cout << "dg_igr unit tests suites: ";
    for (size_t i = 0; i < suites.size(); ++i)
    {
      std::cout << suites[i];
      if (i + 1 < suites.size()) { std::cout << ", "; }
    }
    std::cout << "\n";

    if (selection.euler)
    {
      std::cout << "  euler:    " << (global_euler_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_euler_failures << " failure(s))\n";
    }
    if (selection.fluxes)
    {
      std::cout << "  fluxes:   " << (global_flux_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_flux_failures << " failure(s))\n";
    }
    if (selection.limiters)
    {
      std::cout << "  limiters: " << (global_limiter_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_limiter_failures << " failure(s))\n";
    }
    if (selection.dg_operator)
    {
      std::cout << "  dg-operator: "
                << (global_dg_operator_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_dg_operator_failures << " failure(s))\n";
    }
    if (selection.dgsem_basis)
    {
      std::cout << "  dgsem-basis: "
                << (global_dgsem_basis_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_dgsem_basis_failures << " failure(s))\n";
    }
    if (selection.dgsem_flux)
    {
      std::cout << "  dgsem-flux: "
                << (global_dgsem_flux_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_dgsem_flux_failures << " failure(s))\n";
    }
    if (selection.dgsem_operator)
    {
      std::cout << "  dgsem-operator: "
                << (global_dgsem_operator_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_dgsem_operator_failures << " failure(s))\n";
    }
    if (selection.dgsem_igr)
    {
      std::cout << "  dgsem-igr: "
                << (global_dgsem_igr_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_dgsem_igr_failures << " failure(s))\n";
    }
    if (selection.bc)
    {
      std::cout << "  bc:       " << (global_bc_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_bc_failures << " failure(s))\n";
    }
    if (selection.igr)
    {
      std::cout << "  igr:      " << (global_igr_failures == 0 ? "PASS" : "FAIL")
                << " (" << global_igr_failures << " failure(s))\n";
    }
    if (selection.parity)
    {
      std::cout << "  parity signatures:";
      for (const auto& kv : parity_metrics)
      {
        std::cout << " " << kv.first << "=" << kv.second;
      }
      std::cout << "\n";
    }

    if (global_failures == 0)
    {
      std::cout << "All dg_igr unit tests passed.\n";
    }
    else
    {
      std::cout << "dg_igr unit tests failed: " << global_failures << " failure(s).\n";
    }
  }

  return (global_failures == 0) ? 0 : 1;
}
