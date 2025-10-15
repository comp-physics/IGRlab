#include "timesteppers.hpp"
#include "limiters.hpp"
#include "mfem.hpp"

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

  // If user didn't set a quad order (for helper PP), choose something safe
  if (quad_order_ <= 0) { quad_order_ = AutoQuadOrder(); }
}

int RK3SSPWithLimiter::AutoQuadOrder() const
{
  // Use 2*order + 2 by default (good for averages + positivity points)
  const int order = vfes_.GetFE(0)->GetOrder();
  return 2*order + 2;
}

void RK3SSPWithLimiter::ApplyCallbacks(Vector &v, double t)
{
  if (callbacks_.empty()) { return; }
  ParGridFunction U(&vfes_, v.GetData()); // non-owning view; no copy
  for (auto &cb : callbacks_) { cb(vfes_, U, t); }
}

// ---------- Convenience helpers: push commonly used limiters as callbacks ----------

void RK3SSPWithLimiter::AddZhangShuLimiter()
{
  // Capture parameters by value; the callback will call the PP limiter on the stage.
  const int   dim    = dim_;
  const double gamma  = gamma_;
  const double rhomin = rho_min_;
  const double pmin   = p_min_;
  const int   qord    = (quad_order_ > 0) ? quad_order_ : AutoQuadOrder();
  const int   iters   = p_bisect_iters_;

  callbacks_.emplace_back(
      [=](ParFiniteElementSpace &vf, ParGridFunction &U, double /*t*/)
      {
        ApplyZhangShuPositivityLimiter(vf, U, dim, gamma, rhomin, pmin, qord, iters);
      });
}

void RK3SSPWithLimiter::AddCharTVBLimiter(const CharTVBOptions &opts)
{
  auto local = opts; // make a self-contained copy for the stored lambda
  callbacks_.emplace_back(
      [local](ParFiniteElementSpace &vf, ParGridFunction &U, double /*t*/)
      {
        ApplyCharacteristicTVBLimiter(vf, U, local);
      });
}

void RK3SSPWithLimiter::AddBarthJespersenLimiter(const BJOptions &opts)
{
  auto local = opts;
  callbacks_.emplace_back(
      [local](ParFiniteElementSpace &vf, ParGridFunction &U, double /*t*/)
      {
        ApplyBarthJespersenLimiter(vf, U, local);
      });
}

// --------------------------------- SSPRK(3,3) ---------------------------------

void RK3SSPWithLimiter::Step(Vector &x, real_t &t, real_t &dt)
{
  // Optionally ensure the incoming state satisfies pipeline constraints
  if (limit_initial_) { ApplyCallbacks(x, t); }

  // k0 = f(t, x)
  f->SetTime(t);
  f->Mult(x, k_);

  // y = x + dt * k0, then pipeline
  add(x, dt, k_, y_);
  ApplyCallbacks(y_, t + dt);

  // k1 = f(t+dt, y)
  f->SetTime(t + dt);
  f->Mult(y_, k_);

  // y = 3/4*x + 1/4 ( y + dt * k1 ), then pipeline
  y_.Add(dt, k_);                 // y := y + dt*k1
  add(3.0/4.0, x, 1.0/4.0, y_, y_);
  ApplyCallbacks(y_, t + dt);

  // k2 = f(t + dt/2, y)
  f->SetTime(t + dt/2);
  f->Mult(y_, k_);

  // x = 1/3*x + 2/3 ( y + dt*k2 ), then pipeline
  y_.Add(dt, k_);                 // y := y + dt*k2
  add(1.0/3.0, x, 2.0/3.0, y_, x);
  t += dt;

  ApplyCallbacks(x, t);
}
