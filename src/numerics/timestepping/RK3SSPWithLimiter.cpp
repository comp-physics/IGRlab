#include "numerics/timestepping/RK3SSPWithLimiter.hpp"

#include "utils/diagnostics/non_finite.hpp"
#include "numerics/limiters/limiters.hpp"   // ApplyZhangShuPositivityLimiter, etc.
#include <algorithm>
#include <iostream>

namespace mfem {

// ------------------------------- Construction & Init ----------------------------

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

// ----------------------------- Callback application ----------------------------

void RK3SSPWithLimiter::ApplyCallbacks(Vector &v, double t, bool apply_optional)
{
  // Quick exit if no callbacks at all.
  if (always_callbacks_.empty() && (!apply_optional || optional_callbacks_.empty()))
  {
    return;
  }

  // Non-owning view of v as a ParGridFunction.
  ParGridFunction U(&vfes_, v.GetData());

  // Always-on callbacks (e.g., Zhang–Shu positivity).
  for (auto &cb : always_callbacks_)
  {
    cb(vfes_, U, t);
  }

  // Optional callbacks, only if requested this step.
  if (apply_optional)
  {
    for (auto &cb : optional_callbacks_)
    {
      cb(vfes_, U, t);
    }
  }
}

// ----------------------- Convenience helpers to add callbacks ------------------

void RK3SSPWithLimiter::AddZhangShuLimiter(bool always_on)
{
  // Capture parameters by value; the callback will call the PP limiter on the stage.
  const int    dim    = dim_;
  const double gamma  = gamma_;
  const double rhomin = rho_min_;
  const double pmin   = p_min_;
  const int    qord   = (quad_order_ > 0) ? quad_order_ : AutoQuadOrder();
  const int    iters  = p_bisect_iters_;

  auto cb = [=](ParFiniteElementSpace &vf, ParGridFunction &U, double /*t*/)
  {
    ApplyZhangShuPositivityLimiter(vf, U, dim, gamma, rhomin, pmin,
                                   qord, iters);
  };

  if (always_on)
  {
    AddAlwaysCallback(std::move(cb));
  }
  else
  {
    AddOptionalCallback(std::move(cb));
  }
}

void RK3SSPWithLimiter::AddCharTVBLimiter(const CharTVBOptions &opts)
{
  auto local = opts; // make a self-contained copy for the stored lambda
  AddOptionalCallback(
      [local](ParFiniteElementSpace &vf, ParGridFunction &U, double /*t*/)
      {
        ApplyCharacteristicTVBLimiter(vf, U, local);
      });
}

void RK3SSPWithLimiter::AddCharTVBFaceNormalLimiter(const CharTVBOptions &opts)
{
  auto local = opts; // capture by value
  AddOptionalCallback(
      [local](mfem::ParFiniteElementSpace &vf, mfem::ParGridFunction &U, double /*t*/)
      {
        ApplyCharacteristicTVBFaceNormal(vf, U, local);
      });
}

void RK3SSPWithLimiter::AddCharBJLimiter(const CharBJOptions &opts)
{
  auto local = opts;
  AddOptionalCallback(
      [local](ParFiniteElementSpace &vf, ParGridFunction &U, double /*t*/)
      {
        ApplyCharacteristicBJLimiter(vf, U, local);
      });
}

void RK3SSPWithLimiter::AddFaceCharBJLimiter(const FaceBJOptions &opts)
{
  auto local = opts; // capture by value for safety
  AddOptionalCallback(
      [local](mfem::ParFiniteElementSpace &vf, mfem::ParGridFunction &Ug, double /*t*/)
      {
        ApplyFaceNormalCharacteristicBJLimiter(vf, Ug, local);
      });
}

void RK3SSPWithLimiter::AddBarthJespersenLimiter(const BJOptions &opts)
{
  auto local = opts;
  AddOptionalCallback(
      [local](ParFiniteElementSpace &vf, ParGridFunction &U, double /*t*/)
      {
        ApplyBarthJespersenLimiter(vf, U, local);
      });
}

void RK3SSPWithLimiter::AddCharVenkatXYLimiter(const VenkatOptions &opts)
{
  auto local = opts;
  AddOptionalCallback(
      [local](mfem::ParFiniteElementSpace &vf, mfem::ParGridFunction &U, double /*t*/)
      {
        ApplyCharacteristicVenkatXY(vf, U, local);
      });
}

void RK3SSPWithLimiter::AddCharVenkatFaceNormalLimiter(const VenkatOptions &opts)
{
  auto local = opts;
  AddOptionalCallback(
      [local](mfem::ParFiniteElementSpace &vf, mfem::ParGridFunction &U, double /*t*/)
      {
        ApplyCharacteristicVenkatFaceNormal(vf, U, local);
      });
}

void RK3SSPWithLimiter::AddKrivodonovaLimiter(const KrivoOptions &opts)
{
  auto local = opts;
  AddOptionalCallback(
      [local](ParFiniteElementSpace &vf, ParGridFunction &Ustage, double /*t*/)
      {
        ApplyKrivodonovaHierarchicalLimiter(vf, Ustage, local);
      });
}

void RK3SSPWithLimiter::MaybeScanStage(Vector &v, double t, const char* label)
{
  if (!nan_scan_stages_ || stage_nan_detected_)
  {
    return;
  }

  ParGridFunction stage(&vfes_, v);
  const dg_igr::NonFiniteSummary summary = dg_igr::ScanNonFinite(stage);
  if (summary.global_non_finite > 0)
  {
    stage_nan_detected_ = true;
    if (mfem::Mpi::Root())
    {
      std::cout << "[nan-scan] Detected " << summary.global_non_finite
                << " non-finite entries at " << label << " t=" << t
                << " (finite min=" << summary.global_min
                << ", finite max=" << summary.global_max << ")\n";
    }
  }
}

// --------------------------------- SSPRK(3,3) ---------------------------------

void RK3SSPWithLimiter::Step(Vector &x, real_t &t, real_t &dt)
{
  stage_nan_detected_ = false;
  // Decide whether to apply optional limiters on this step.
  const bool apply_optional =
      (max_limiter_steps_ < 0) || (step_count_ < max_limiter_steps_);

  // Optionally ensure the incoming state satisfies pipeline constraints.
  //   - Always-callbacks: always applied if limit_initial_ is true.
  //   - Optional-callbacks: only early steps if apply_optional = true.
  if (limit_initial_)
  {
    ApplyCallbacks(x, t, apply_optional);
  }
  MaybeScanStage(x, t, "initial");

  // k0 = f(t, x)
  f->SetTime(t);
  f->Mult(x, k_);

  // ===== Stage 1: y = x + dt * k0 =====
  add(x, dt, k_, y_);
  ApplyCallbacks(y_, t + dt, apply_optional);
  MaybeScanStage(y_, t + dt, "stage1");

  // k1 = f(t+dt, y)
  f->SetTime(t + dt);
  f->Mult(y_, k_); 

  // ===== Stage 2: y = 3/4*x + 1/4 ( y + dt * k1 ) =====
  y_.Add(dt, k_);                 // y := y + dt*k1
  add(3.0/4.0, x, 1.0/4.0, y_, y_);
  ApplyCallbacks(y_, t + dt, /*apply_optional=*/false);
  MaybeScanStage(y_, t + dt, "stage2");

  // k2 = f(t + dt/2, y)
  f->SetTime(t + dt/2);
  f->Mult(y_, k_);

  // ===== Stage 3: x = 1/3*x + 2/3 ( y + dt*k2 ) =====
  y_.Add(dt, k_);                 // y := y + dt*k2
  add(1.0/3.0, x, 2.0/3.0, y_, x);
  t += dt;
  // Final limiting on x 
  ApplyCallbacks(x, t, /*apply_optional=*/false);
  MaybeScanStage(x, t, "stage3");

  ++step_count_;
}

} // namespace mfem
