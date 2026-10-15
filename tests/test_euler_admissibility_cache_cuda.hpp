#pragma once

#include "mfem.hpp"

#include <vector>

// Test-only CXX/CUDA boundary. This is not a production scan or arithmetic
// capability: the native experiment must establish parity with its host oracle.
namespace admissibility_cuda_probe
{
struct Sample
{
  int dof;
  int map_offset;
  int shape_offset; // -1 for a raw node occurrence
  int node;
  int element;
  int face;
  int point;
};

struct Value
{
  double state[4];
  double kinetic;
  double pressure;
  int bad_rho;
  int bad_pressure;
  int declined;
};

struct LocalSummary
{
  double min_rho;
  double min_pressure;
  int bad_rho;
  int bad_pressure;
  int first_sample;
  int declined;
};

struct Result
{
  std::vector<Value> values;
  LocalSummary summary{};
  bool environment_declined = false;
  bool sticky_flags_declined = false;
};

void FillDeviceState(mfem::Vector& state, int scalar_dofs, double rho,
                     bool invalid_first_momentum);
void WriteDeviceState(mfem::Vector& state, const std::vector<double>& values);

Result Evaluate(const mfem::Vector& state, const std::vector<int>& maps,
                const std::vector<double>& shapes,
                const std::vector<Sample>& samples, double gamma,
                double rho_min, double p_min, bool environment_supported,
                bool require_sticky_flags = false);
} // namespace admissibility_cuda_probe
