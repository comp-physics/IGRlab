#pragma once

#include "mfem.hpp"
#include "physics/euler/diagnostics.hpp"

namespace dg_igr_examples
{

/**
 * @brief Views into conservative Euler state components (2D).
 */
struct EulerStateViews
{
  mfem::ParGridFunction density;
  mfem::ParGridFunction momentum;
  mfem::ParGridFunction momentum_x;
  mfem::ParGridFunction momentum_y;
  mfem::ParGridFunction energy;

  EulerStateViews(mfem::ParFiniteElementSpace& fes,
                  mfem::ParFiniteElementSpace& dfes,
                  mfem::ParGridFunction& sol,
                  int dim)
    : density(&fes, sol.GetData()),
      momentum(&dfes, sol.GetData() + fes.GetNDofs()),
      momentum_x(&fes, sol.GetData() + 1 * fes.GetNDofs()),
      momentum_y(&fes, sol.GetData() + 2 * fes.GetNDofs()),
      energy(&fes, sol.GetData() + (dim + 1) * fes.GetNDofs())
  {
    MFEM_VERIFY(dim == 2, "EulerStateViews assumes a 2D Euler state layout.");
  }
};

/**
 * @brief Pressure projection wrapper for Euler states.
 */
struct EulerPressureView
{
  mfem::PressureCoefficient coeff;
  mfem::ParGridFunction pressure;
  mfem::ParGridFunction* state = nullptr;

  EulerPressureView(mfem::ParFiniteElementSpace& fes,
                    mfem::ParGridFunction& sol,
                    int dim,
                    double gamma)
    : coeff(sol, dim, gamma),
      pressure(&fes),
      state(&sol)
  {
    Update();
  }

  void Update()
  {
    coeff.SetState(*state);
    pressure.ProjectCoefficient(coeff);
  }
};

} // namespace dg_igr_examples
