#include "timesteppers.hpp"
#include "limiters.hpp"   // <- your ApplyZhangShuPositivityLimiter decl
#include <algorithm>

using namespace mfem;

RK3SSPWithLimiter::RK3SSPWithLimiter(ParFiniteElementSpace &vfes)
  : vfes_(vfes)
{ }

void RK3SSPWithLimiter::Init(TimeDependentOperator &f_)
{
  ODESolver::Init(f_);
  const int n = f->Width();
  y_.SetSize(n, mem_type);
  k_.SetSize(n, mem_type);

  // If user didn't set a quad order, choose something safe
  if (quad_order_ <= 0) { quad_order_ = AutoQuadOrder(); }
}

int RK3SSPWithLimiter::AutoQuadOrder() const
{
  // Use 2*order + 2 by default (good for computing averages + positivity points)
  const int order = vfes_.GetFE(0)->GetOrder();
  return 2*order + 2;
}

void RK3SSPWithLimiter::LimitVectorAsGridFunction(Vector &v)
{
  // View v as a ParGridFunction over vfes_ (no copy; external data view).
  ParGridFunction U(&vfes_, v.GetData());

  // Apply Zhang–Shu positivity limiter you already implemented
  ApplyZhangShuPositivityLimiter(
      vfes_, U, dim_, gamma_,
      rho_min_, p_min_,
      quad_order_, p_bisect_iters_);
}

void RK3SSPWithLimiter::Step(Vector &x, real_t &t, real_t &dt)
{
  // Optionally ensure admissibility of the incoming state
  if (limit_initial_) { LimitVectorAsGridFunction(x); }

  // k0 = f(t, x)
  f->SetTime(t);
  f->Mult(x, k_);

  // y = x + dt * k0, then LIMIT
  add(x, dt, k_, y_);
  LimitVectorAsGridFunction(y_);

  // k1 = f(t+dt, y)
  f->SetTime(t + dt);
  f->Mult(y_, k_);

  // y = 3/4*x + 1/4*( y + dt*k1 ), then LIMIT
  y_.Add(dt, k_);                 // y := y + dt*k1
  add(3.0/4.0, x, 1.0/4.0, y_, y_);
  LimitVectorAsGridFunction(y_);

  // k2 = f(t + dt/2, y)
  f->SetTime(t + dt/2);
  f->Mult(y_, k_);

  // x = 1/3*x + 2/3*( y + dt*k2 ), then LIMIT
  y_.Add(dt, k_);                 // y := y + dt*k2
  add(1.0/3.0, x, 2.0/3.0, y_, x);
  t += dt;

  LimitVectorAsGridFunction(x);
}
