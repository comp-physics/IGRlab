#pragma once

/**
 * @file sim/state/state_views.hpp
 * @brief Euler state view helpers (density/momentum/pressure projections).
 */

#include "mfem.hpp"
#include "physics/euler/diagnostics.hpp"

namespace dg_sim
{

/**
 * @brief Views into conservative Euler state components (2D).
 */
struct EulerStateViews
{
private:
  static mfem::real_t* ComponentData(mfem::ParGridFunction& state,
                                     const int offset) noexcept
  {
    mfem::real_t* const data = state.GetData();
    return data == nullptr ? nullptr : data + offset;
  }

public:
  mfem::ParGridFunction density;
  mfem::ParGridFunction momentum;
  mfem::ParGridFunction momentum_x;
  mfem::ParGridFunction momentum_y;
  mfem::ParGridFunction energy;

  EulerStateViews(mfem::ParFiniteElementSpace& fes,
                  mfem::ParFiniteElementSpace& dfes,
                  mfem::ParGridFunction& sol,
                  int dim)
    : density(&fes, ComponentData(sol, 0)),
      momentum(&dfes, ComponentData(sol, fes.GetNDofs())),
      momentum_x(&fes, ComponentData(sol, 1 * fes.GetNDofs())),
      momentum_y(&fes, ComponentData(sol, 2 * fes.GetNDofs())),
      energy(&fes, ComponentData(sol, (dim + 1) * fes.GetNDofs()))
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

/**
 * @brief Velocity projection wrapper for Euler states.
 */
struct EulerVelocityView
{
  mfem::VelocityFromConsCoefficient coeff;
  mfem::ParGridFunction velocity;
  mfem::ParGridFunction* state = nullptr;

  EulerVelocityView(mfem::ParFiniteElementSpace& dfes,
                    mfem::ParGridFunction& sol,
                    int dim,
                    double rho_floor)
    : coeff(dim, &sol, rho_floor),
      velocity(&dfes),
      state(&sol)
  {
    Update();
  }

  void Update()
  {
    coeff.SetState(state);
    velocity.ProjectCoefficient(coeff);
  }
};

} // namespace dg_sim
