#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "numerics/timestepping/ssprk_coefficients.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "sim/time/time_loop.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace
{

using dg_igr_tests::TestContext;

bool VectorsExactlyEqual(const mfem::Vector& left, const mfem::Vector& right)
{
  if (left.Size() != right.Size())
  {
    return false;
  }
  const mfem::real_t* left_values = left.HostRead();
  const mfem::real_t* right_values = right.HostRead();
  for (int i = 0; i < left.Size(); ++i)
  {
    if (left_values[i] != right_values[i])
    {
      return false;
    }
  }
  return true;
}

template <typename Coefficients>
void CheckClassicalOrderConditions(TestContext& ctx,
                                   const std::string& method_name)
{
  constexpr std::size_t s = Coefficients::stage_count;
  const auto& a = Coefficients::a;
  const auto& b = Coefficients::b;
  const auto& c = Coefficients::c;
  std::array<mfem::real_t, s> ac{};
  std::array<mfem::real_t, s> ac2{};
  std::array<mfem::real_t, s> aac{};
  for (std::size_t i = 0; i < s; ++i)
  {
    for (std::size_t j = 0; j < s; ++j)
    {
      ac[i] += a[i][j] * c[j];
      ac2[i] += a[i][j] * c[j] * c[j];
    }
  }
  for (std::size_t i = 0; i < s; ++i)
  {
    for (std::size_t j = 0; j < s; ++j)
    {
      aac[i] += a[i][j] * ac[j];
    }
  }

  mfem::real_t sum_b = 0.0;
  mfem::real_t sum_bc = 0.0;
  mfem::real_t sum_bc2 = 0.0;
  mfem::real_t sum_bac = 0.0;
  mfem::real_t sum_bc3 = 0.0;
  mfem::real_t sum_bac2 = 0.0;
  mfem::real_t sum_b_c_ac = 0.0;
  mfem::real_t sum_baac = 0.0;
  for (std::size_t i = 0; i < s; ++i)
  {
    sum_b += b[i];
    sum_bc += b[i] * c[i];
    sum_bc2 += b[i] * c[i] * c[i];
    sum_bac += b[i] * ac[i];
    sum_bc3 += b[i] * c[i] * c[i] * c[i];
    sum_bac2 += b[i] * ac2[i];
    sum_b_c_ac += b[i] * c[i] * ac[i];
    sum_baac += b[i] * aac[i];
  }

  constexpr double tol = 2.0e-14;
  ctx.CheckNear(sum_b, 1.0, tol, method_name + " order-one condition");
  if constexpr (Coefficients::order >= 2)
  {
    ctx.CheckNear(sum_bc,
                  0.5,
                  tol,
                  method_name + " order-two condition");
  }
  if constexpr (Coefficients::order >= 3)
  {
    ctx.CheckNear(sum_bc2,
                  1.0 / 3.0,
                  tol,
                  method_name + " third-order moment condition");
    ctx.CheckNear(sum_bac,
                  1.0 / 6.0,
                  tol,
                  method_name + " third-order coupling condition");
  }
  if constexpr (Coefficients::order >= 4)
  {
    ctx.CheckNear(sum_bc3,
                  0.25,
                  tol,
                  method_name + " fourth-order cubic moment condition");
    ctx.CheckNear(sum_bac2,
                  1.0 / 12.0,
                  tol,
                  method_name + " fourth-order A-c-squared condition");
    ctx.CheckNear(sum_b_c_ac,
                  0.125,
                  tol,
                  method_name + " fourth-order C-A-c condition");
    ctx.CheckNear(sum_baac,
                  1.0 / 24.0,
                  tol,
                  method_name + " fourth-order A-squared-c condition");
  }
}

class RecordingLinearOperator final
  : public mfem::TimeDependentOperator,
    public mfem::RKStageAwareOperator,
    public mfem::PhysicalBoundaryFluxIntegralProvider
{
public:
  explicit RecordingLinearOperator(int size, bool time_scaled = false)
    : mfem::TimeDependentOperator(size), time_scaled_(time_scaled)
  {
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetMemoryClass();
  }

  mfem::real_t GetMaxCharSpeed() const { return reported_characteristic_speed; }

  void Mult(const mfem::Vector& state, mfem::Vector& derivative) const override
  {
    rhs_times.push_back(GetTime());
    rhs_stages.push_back(current_stage);
    reported_characteristic_speed = initial_characteristic_speed;
    if (current_stage >= 0 &&
        current_stage < static_cast<int>(stage_characteristic_speeds.size()))
    {
      reported_characteristic_speed =
        stage_characteristic_speeds[static_cast<std::size_t>(current_stage)];
    }
    derivative = state;
    if (time_scaled_)
    {
      derivative *= GetTime();
    }
    if (boundary_flux_armed)
    {
      boundary_flux_armed = false;
      boundary_flux_available = true;
      last_boundary_flux.SetSize(4);
      mfem::real_t* flux = last_boundary_flux.HostWrite();
      for (int component = 0; component < last_boundary_flux.Size(); ++component)
      {
        const mfem::real_t magnitude =
          static_cast<mfem::real_t>((current_stage + 1) * (component + 1));
        flux[component] = ((current_stage + component) % 2 == 0)
                            ? magnitude
                            : -magnitude;
      }
      ++boundary_flux_captures;
    }
  }

  bool SupportsPhysicalBoundaryFluxIntegral() const override { return true; }

  void ArmPhysicalBoundaryFluxIntegral() const override
  {
    boundary_flux_armed = true;
    boundary_flux_available = false;
  }

  bool HasLastPhysicalBoundaryFluxIntegral() const override
  {
    return boundary_flux_available;
  }

  const mfem::Vector& GetLastPhysicalBoundaryFluxIntegral() const override
  {
    MFEM_VERIFY(boundary_flux_available,
                "Test boundary flux requested before capture completed.");
    return last_boundary_flux;
  }

  void BeginRKStep(int step,
                   mfem::real_t time,
                   mfem::real_t dt) override
  {
    BeginRKStep(step, time, dt, /*stage_count=*/3);
  }

  void BeginRKStep(int step,
                   mfem::real_t time,
                   mfem::real_t dt,
                   int stage_count) override
  {
    begin_step = step;
    begin_time = time;
    begin_dt = dt;
    begin_stage_count = stage_count;
    current_stage = -1;
    begin_steps.push_back(step);
    begin_times.push_back(time);
    begin_dts.push_back(dt);
    begin_stage_counts.push_back(stage_count);
  }

  void SetRKStage(int stage) override { current_stage = stage; }

  void SetRKStageState(int stage, std::uint64_t generation) override
  {
    current_stage = stage;
    rhs_generations.push_back(generation);
  }

  void EndRKStep(const mfem::Vector&,
                 mfem::real_t time,
                 std::uint64_t generation) override
  {
    ++end_calls;
    end_time = time;
    end_generation = generation;
  }

  void AbortRKStep(const mfem::Vector& restored_state,
                   mfem::real_t restored_time,
                   std::uint64_t restored_generation,
                   mfem::real_t retry_dt) override
  {
    ++abort_calls;
    abort_state.SetSize(restored_state.Size(),
                        restored_state.GetMemory().GetMemoryType());
    abort_state.UseDevice(restored_state.UseDevice());
    abort_state = restored_state;
    abort_time = restored_time;
    abort_generation = restored_generation;
    abort_retry_dt = retry_dt;
  }

  mutable std::vector<mfem::real_t> rhs_times;
  mutable std::vector<int> rhs_stages;
  std::vector<std::uint64_t> rhs_generations;
  std::vector<int> begin_steps;
  std::vector<mfem::real_t> begin_times;
  std::vector<mfem::real_t> begin_dts;
  std::vector<int> begin_stage_counts;
  int begin_step = -1;
  mfem::real_t begin_time = -1.0;
  mfem::real_t begin_dt = -1.0;
  int begin_stage_count = -1;
  int current_stage = -1;
  int end_calls = 0;
  mfem::real_t end_time = -1.0;
  std::uint64_t end_generation = 0;
  int abort_calls = 0;
  mfem::Vector abort_state;
  mfem::real_t abort_time = -1.0;
  std::uint64_t abort_generation = 0;
  mfem::real_t abort_retry_dt = -1.0;
  mutable bool boundary_flux_armed = false;
  mutable bool boundary_flux_available = false;
  mutable mfem::Vector last_boundary_flux;
  mutable int boundary_flux_captures = 0;
  mfem::real_t initial_characteristic_speed = 1.0;
  std::vector<mfem::real_t> stage_characteristic_speeds;
  mutable mfem::real_t reported_characteristic_speed = 1.0;

private:
  bool time_scaled_ = false;
};

void CheckAcceptedCharacteristicSpeedTransaction(
  TestContext& ctx,
  mfem::ParMesh& mesh,
  mfem::ParFiniteElementSpace& fes)
{
  RecordingLinearOperator op(fes.GetTrueVSize());
  op.stage_characteristic_speeds =
    {2.0, 3.0, 4.0, 17.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0};
  mfem::RK3SSPWithLimiter stepper(fes, mfem::SSPRKMethod::SSPRK104);
  bool reject_final_callback = false;
  stepper.AddAlwaysCallback(
    [&](mfem::ParFiniteElementSpace&,
        mfem::ParGridFunction&,
        double)
    {
      if (reject_final_callback && op.current_stage == 9)
      {
        throw mfem::RecoverableLimiterFailure(1);
      }
    });
  stepper.Init(op);
  stepper.SetCharacteristicSpeedReader(
    [&op]() { return op.GetMaxCharSpeed(); });

  mfem::Vector state(fes.GetTrueVSize());
  state.UseDevice(true);
  state = 1.0;
  mfem::real_t time = 0.0;
  mfem::real_t dt = 0.01;
  stepper.Step(state, time, dt);
  ctx.Check(stepper.HasAcceptedCharacteristicSpeed() &&
              stepper.GetAcceptedCharacteristicSpeed() == 17.0,
            "SSPRK(10,4) commits the maximum speed across every RHS stage");

  op.stage_characteristic_speeds =
    {20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0};
  reject_final_callback = true;
  bool rejected = false;
  try
  {
    stepper.Step(state, time, dt);
  }
  catch (const mfem::RecoverableLimiterFailure&)
  {
    rejected = true;
  }
  ctx.Check(rejected && stepper.HasAcceptedCharacteristicSpeed() &&
              stepper.GetAcceptedCharacteristicSpeed() == 17.0,
            "SSPRK rejected work cannot publish its attempted stage-speed "
            "maximum");

  RecordingLinearOperator loop_op(fes.GetTrueVSize());
  loop_op.stage_characteristic_speeds =
    {2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 2.0, 2.0, 2.0, 1.0};
  mfem::RK3SSPWithLimiter loop_stepper(
    fes, mfem::SSPRKMethod::SSPRK104);
  loop_stepper.Init(loop_op);
  mfem::ParGridFunction loop_state(&fes);
  loop_state = 1.0;
  dg_sim::TimeLoopConfig config;
  config.t_final = 1.25;
  config.dt = 0.1;
  config.cfl = 1.0;
  config.order = 0;
  config.vtk_save = false;
  const dg_sim::TimeLoopResult result = dg_sim::RunTimeLoop(
    loop_op, mesh, loop_state, loop_stepper, config);
  ctx.Check(result.success && result.accepted_attempts == 2 &&
              result.characteristic_speed_probes == 3 &&
              result.minimum_characteristic_speed == 1.0 &&
              result.maximum_characteristic_speed == 4.0 &&
              result.minimum_accepted_dt == 0.25 &&
              result.maximum_accepted_dt == 1.0,
            "adaptive CFL consumes and reports every accepted all-stage "
            "speed maximum, including the terminal step");
}

void CheckTerminalAlwaysCallbackOrdering(TestContext& ctx,
                                         mfem::ParFiniteElementSpace& fes)
{
  RecordingLinearOperator op(fes.GetTrueVSize());
  mfem::RK3SSPWithLimiter stepper(fes, mfem::SSPRKMethod::SSPRK33);
  stepper.LimitInitial(false);
  stepper.SetLimiterSteps(-1);

  int optional_calls = 0;
  int optional_stage = -1;
  int terminal_gate_matches = 0;
  stepper.AddOptionalCallback(
    [&](mfem::ParFiniteElementSpace&,
        mfem::ParGridFunction&,
        double)
    {
      ++optional_calls;
      optional_stage = op.current_stage;
    });
  stepper.AddAlwaysCallback(
    [&](mfem::ParFiniteElementSpace&,
        mfem::ParGridFunction&,
        double)
    {
      if (optional_stage >= 0)
      {
        terminal_gate_matches += optional_stage == op.current_stage ? 1 : 0;
        optional_stage = -1;
      }
    });
  stepper.Init(op);

  mfem::Vector state(fes.GetTrueVSize());
  state = 1.0;
  mfem::real_t time = 0.0;
  mfem::real_t dt = 0.01;
  stepper.Step(state, time, dt);
  ctx.Check(optional_calls > 0 && terminal_gate_matches == optional_calls &&
              optional_stage < 0,
            "always-on stage admission runs after each optional slope "
            "limiter on the same stage");
}

void CheckRejectedLimiterDiagnosticsStayCommitted(TestContext& ctx)
{
  constexpr double gamma = 1.4;
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian1D(1, 1.0);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial_mesh);
  mfem::L2_FECollection state_fec(/*order=*/1, /*dim=*/1);
  mfem::ParFiniteElementSpace state_fes(&mesh,
                                        &state_fec,
                                        /*vdim=*/3,
                                        mfem::Ordering::byNODES);
  mfem::ParGridFunction state(&state_fes);
  const mfem::Vector constant =
    dg_igr_tests::EulerConsFromPrim1D(1.0, 0.1, 1.0, gamma);
  dg_igr_tests::FillConstantState(constant, state);

  mfem::L2_FECollection diagnostic_fec(/*order=*/0, /*dim=*/1);
  mfem::ParFiniteElementSpace diagnostic_fes(&mesh, &diagnostic_fec);
  mfem::ParGridFunction active(&diagnostic_fes);
  mfem::ParGridFunction theta_rho(&diagnostic_fes);
  mfem::ParGridFunction theta_p(&diagnostic_fes);
  active = -7.0;
  theta_rho = -8.0;
  theta_p = -9.0;

  RecordingLinearOperator op(state.Size());
  mfem::RK3SSPWithLimiter stepper(state_fes, mfem::SSPRKMethod::SSPRK33);
  stepper.SetEulerParams(/*dim=*/1, gamma);
  stepper.LimitInitial(false);
  stepper.SetZhangShuDiagnosticsFields(&active, &theta_rho, &theta_p);
  stepper.AddZhangShuLimiter(mfem::ZhangShuPointSet::kConsumer);
  stepper.AddAlwaysCallback(
    [](mfem::ParFiniteElementSpace&, mfem::ParGridFunction&, double)
    { throw mfem::RecoverableLimiterFailure(1); });
  stepper.Init(op);

  mfem::real_t time = 0.0;
  mfem::real_t dt = 0.01;
  bool rejected = false;
  try
  {
    stepper.Step(state, time, dt);
  }
  catch (const mfem::RecoverableLimiterFailure&)
  {
    rejected = true;
  }
  ctx.Check(rejected && active.HostRead()[0] == -7.0 &&
              theta_rho.HostRead()[0] == -8.0 &&
              theta_p.HostRead()[0] == -9.0,
            "a rejected RK attempt leaves published limiter diagnostic "
            "fields at their last committed values");
}

template <typename Coefficients>
void CheckLifecycle(TestContext& ctx,
                    mfem::ParFiniteElementSpace& fes,
                    mfem::SSPRKMethod method,
                    const std::string& method_name)
{
  RecordingLinearOperator op(fes.GetTrueVSize());
  mfem::RK3SSPWithLimiter stepper(fes, method);
  std::vector<mfem::real_t> callback_times;
  stepper.AddAlwaysCallback(
    [&callback_times](mfem::ParFiniteElementSpace&,
                      mfem::ParGridFunction&,
                      double time) { callback_times.push_back(time); });
  stepper.Init(op);

  mfem::Vector state(fes.GetTrueVSize());
  state.UseDevice(true);
  state = 1.0;
  mfem::real_t time = 0.2;
  mfem::real_t dt = 0.3;
  stepper.Step(state, time, dt);

  constexpr int s = static_cast<int>(Coefficients::stage_count);
  ctx.Check(stepper.GetStageCount() == s &&
              stepper.GetOrder() == Coefficients::order &&
              stepper.GetSSPCoefficient() ==
                Coefficients::ssp_coefficient,
            method_name + " reports its typed method properties");
  ctx.Check(op.begin_step == 0 && op.begin_stage_count == s,
            method_name + " announces the complete stage count");
  ctx.CheckNear(op.begin_time,
                0.2,
                0.0,
                method_name + " announces the step start time");
  ctx.CheckNear(op.begin_dt,
                0.3,
                0.0,
                method_name + " announces the step size");
  ctx.Check(static_cast<int>(op.rhs_times.size()) == s &&
              static_cast<int>(op.rhs_stages.size()) == s &&
              static_cast<int>(op.rhs_generations.size()) == s,
            method_name + " performs exactly one lifecycle event per RHS stage");
  if (static_cast<int>(op.rhs_times.size()) == s &&
      static_cast<int>(op.rhs_stages.size()) == s &&
      static_cast<int>(op.rhs_generations.size()) == s)
  {
    for (int stage = 0; stage < s; ++stage)
    {
      ctx.Check(op.rhs_stages[stage] == stage,
                method_name + " reports RHS stage " + std::to_string(stage));
      ctx.Check(op.rhs_generations[stage] ==
                  static_cast<std::uint64_t>(stage),
                method_name + " reports current stage-state generation " +
                  std::to_string(stage));
      ctx.CheckNear(op.rhs_times[stage],
                    0.2 + Coefficients::c[stage] * 0.3,
                    2.0e-14,
                    method_name + " uses physical time for RHS stage " +
                      std::to_string(stage));
    }
  }

  ctx.Check(static_cast<int>(callback_times.size()) == s,
            method_name + " applies a limiter callback after every stage update");
  if (static_cast<int>(callback_times.size()) == s)
  {
    for (int callback = 0; callback + 1 < s; ++callback)
    {
      ctx.CheckNear(callback_times[callback],
                    0.2 + Coefficients::c[callback + 1] * 0.3,
                    2.0e-14,
                    method_name + " callback uses next-stage physical time " +
                      std::to_string(callback));
    }
    ctx.CheckNear(callback_times.back(),
                  0.5,
                  2.0e-14,
                  method_name + " final callback uses accepted endpoint time");
  }
  ctx.CheckNear(time,
                0.5,
                2.0e-14,
                method_name + " commits the physical endpoint time");
  ctx.CheckNear(op.end_time,
                time,
                0.0,
                method_name + " ends lifecycle on the accepted endpoint");
  ctx.Check(op.end_generation == static_cast<std::uint64_t>(s) &&
              stepper.GetStateGeneration() == static_cast<std::uint64_t>(s),
            method_name + " publishes accepted endpoint state generation");
  ctx.Check(stepper.GetRHSEvaluationCount() == s,
            method_name + " exposes its complete RHS evaluation count");
}

template <typename Coefficients>
void CheckPhysicalBoundaryFluxTracking(TestContext& ctx,
                                       mfem::ParFiniteElementSpace& fes,
                                       mfem::SSPRKMethod method,
                                       const std::string& method_name)
{
  RecordingLinearOperator op(fes.GetTrueVSize());
  mfem::RK3SSPWithLimiter stepper(fes, method);
  stepper.EnablePhysicalBoundaryFluxTracking();
  stepper.Init(op);

  ctx.Check(stepper.SupportsPhysicalBoundaryFluxTracking() &&
              stepper.IsPhysicalBoundaryFluxTrackingEnabled() &&
              !stepper.HasCumulativePhysicalBoundaryFluxIntegral(),
            method_name + " advertises enabled tracking before its first step");

  mfem::Vector state(fes.GetTrueVSize());
  state.UseDevice(true);
  state = 1.0;
  mfem::real_t time = 0.0;
  mfem::real_t dt = 0.3;
  stepper.Step(state, time, dt);

  constexpr int stage_count = static_cast<int>(Coefficients::stage_count);
  const mfem::Vector& signed_total =
    stepper.GetCumulativePhysicalBoundaryFluxIntegral();
  const mfem::Vector& absolute_total =
    stepper.GetCumulativePhysicalBoundaryFluxAbsoluteIntegral();
  ctx.Check(stepper.HasCumulativePhysicalBoundaryFluxIntegral() &&
              signed_total.Size() == 4 && absolute_total.Size() == 4 &&
              op.boundary_flux_captures == stage_count,
            method_name + " captures exactly one boundary flux per RHS stage");

  for (int component = 0; component < 4; ++component)
  {
    mfem::real_t expected_signed = 0.0;
    mfem::real_t expected_absolute = 0.0;
    for (int stage = 0; stage < stage_count; ++stage)
    {
      const mfem::real_t magnitude =
        static_cast<mfem::real_t>((stage + 1) * (component + 1));
      const mfem::real_t flux = ((stage + component) % 2 == 0)
                                  ? magnitude
                                  : -magnitude;
      expected_signed += dt * Coefficients::b[stage] * flux;
      expected_absolute += dt * Coefficients::b[stage] * std::abs(flux);
    }
    ctx.CheckNear(signed_total[component],
                  expected_signed,
                  2.0e-14,
                  method_name + " uses authoritative Butcher weights for "
                                "signed boundary component " +
                    std::to_string(component));
    ctx.CheckNear(absolute_total[component],
                  expected_absolute,
                  2.0e-14,
                  method_name + " accumulates the componentwise absolute "
                                "boundary scale " +
                    std::to_string(component));
  }

  const mfem::Vector signed_before_probe(signed_total);
  const mfem::Vector absolute_before_probe(absolute_total);
  mfem::Vector rhs(state.Size());
  op.Mult(state, rhs); // Representative unarmed CFL-style probe.
  ctx.Check(op.boundary_flux_captures == stage_count &&
              VectorsExactlyEqual(
                stepper.GetCumulativePhysicalBoundaryFluxIntegral(),
                signed_before_probe) &&
              VectorsExactlyEqual(
                stepper.GetCumulativePhysicalBoundaryFluxAbsoluteIntegral(),
                absolute_before_probe),
            method_name + " excludes an unarmed RHS probe from both totals");
}

void CheckActualDGStaticSourceSSPRK104(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr int components = dim + 2;
  constexpr double gamma = 1.4;
  constexpr mfem::real_t dt = 0.125;

  mfem::Mesh cartesian = mfem::Mesh::MakeCartesian2D(
    /*nx=*/4,
    /*ny=*/3,
    mfem::Element::QUADRILATERAL,
    /*generate_edges=*/true,
    /*sx=*/1.0,
    /*sy=*/1.0);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                             mfem::Vector({0.0, 1.0})};
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
    cartesian, cartesian.CreatePeriodicVertexMapping(translations));
  mfem::ParMesh mesh(MPI_COMM_SELF, periodic);
  mfem::DG_FECollection fec(
    /*order=*/1, dim, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(
    &mesh, &fec, components, mfem::Ordering::byNODES);

  const mfem::Vector initial_components = dg_igr_tests::EulerConsFromPrim2D(
    /*rho=*/1.2,
    /*u=*/0.3,
    /*v=*/-0.2,
    /*p=*/1.0,
    gamma);
  mfem::Vector source_components(components);
  source_components(0) = 0.25;
  source_components(1) = -0.1;
  source_components(2) = 0.05;
  source_components(3) = 0.4;

  mfem::ParGridFunction initial(&fes);
  mfem::ParGridFunction projected_source(&fes);
  dg_igr_tests::FillConstantState(initial_components, initial);
  dg_igr_tests::FillConstantState(source_components, projected_source);
  mfem::Vector state;
  mfem::Vector source;
  initial.GetTrueDofs(state);
  projected_source.GetTrueDofs(source);
  state.UseDevice(true);
  source.UseDevice(true);

  mfem::EulerFlux physical_flux(dim, gamma);
  mfem::RusanovFluxWithSides numerical_flux(physical_flux);
  auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(
    numerical_flux, /*IntOrderOffset=*/2);
  mfem::DGHyperbolicConservationLaws op(
    fes, std::move(form), /*preassembleWeakDivergence=*/true);
  op.SetNumericalFlux(&numerical_flux);

  mfem::Vector zero_source(source.Size());
  zero_source = 0.0;
  op.SetStaticSource(zero_source);
  ctx.Check(!op.HasStaticSource(),
            "ordinary DG discards an identically zero static source");
  op.SetStaticSource(source);
  ctx.Check(op.HasStaticSource(),
            "ordinary DG retains a nonzero projected static source");

  mfem::Vector derivative(state.Size());
  op.Mult(state, derivative);
  mfem::Vector derivative_error(derivative);
  derivative_error -= source;
  ctx.CheckNear(derivative_error.Normlinf(),
                0.0,
                2.0e-11,
                "ordinary DG adds the projected static source after inverse "
                "mass");

  mfem::RK3SSPWithLimiter stepper(fes, mfem::SSPRKMethod::SSPRK104);
  stepper.EnablePhysicalBoundaryFluxTracking();
  stepper.Init(op);
  mfem::real_t time = 0.0;
  mfem::real_t step_dt = dt;
  stepper.Step(state, time, step_dt);
  mfem::Vector expected_components(initial_components);
  expected_components.Add(dt, source_components);
  mfem::ParGridFunction expected_grid_function(&fes);
  dg_igr_tests::FillConstantState(expected_components,
                                 expected_grid_function);
  mfem::Vector expected_state;
  expected_grid_function.GetTrueDofs(expected_state);
  mfem::Vector state_error(state);
  state_error -= expected_state;
  ctx.CheckNear(state_error.Normlinf(),
                0.0,
                5.0e-11,
                "SSPRK(10,4) advances the actual DG static source exactly");
  ctx.CheckNear(time,
                dt,
                2.0e-14,
                "static-source SSPRK(10,4) reaches the requested horizon");
  ctx.Check(stepper.HasCumulativePhysicalBoundaryFluxIntegral() &&
              stepper.GetCumulativePhysicalBoundaryFluxIntegral().Normlinf() ==
                0.0,
            "periodic static-source trajectory retains zero boundary flux");
}

double IntegrateExponential(mfem::ParFiniteElementSpace& fes,
                            mfem::SSPRKMethod method,
                            int steps)
{
  RecordingLinearOperator op(fes.GetTrueVSize());
  mfem::RK3SSPWithLimiter stepper(fes, method);
  stepper.Init(op);
  mfem::Vector state(fes.GetTrueVSize());
  state.UseDevice(true);
  state = 1.0;
  mfem::real_t time = 0.0;
  mfem::real_t dt = 1.0 / static_cast<mfem::real_t>(steps);
  for (int step = 0; step < steps; ++step)
  {
    stepper.Step(state, time, dt);
  }
  return state.HostRead()[0];
}

double IntegrateNonautonomousManufactured(
  mfem::ParFiniteElementSpace& fes,
  mfem::SSPRKMethod method,
  int steps)
{
  // u'(t)=t*u(t), u(0)=1 has the exact solution exp(t^2/2).
  RecordingLinearOperator op(fes.GetTrueVSize(), /*time_scaled=*/true);
  mfem::RK3SSPWithLimiter stepper(fes, method);
  stepper.Init(op);
  mfem::Vector state(fes.GetTrueVSize());
  state.UseDevice(true);
  state = 1.0;
  mfem::real_t time = 0.0;
  mfem::real_t dt = 1.0 / static_cast<mfem::real_t>(steps);
  for (int step = 0; step < steps; ++step)
  {
    stepper.Step(state, time, dt);
  }
  return state.HostRead()[0];
}

void CheckObservedOrder(TestContext& ctx,
                        mfem::ParFiniteElementSpace& fes,
                        mfem::SSPRKMethod method,
                        int expected_order,
                        const std::string& method_name)
{
  const double exact = std::exp(1.0);
  const double coarse_error =
    std::abs(IntegrateExponential(fes, method, 4) - exact);
  const double fine_error =
    std::abs(IntegrateExponential(fes, method, 8) - exact);
  const double observed_order = std::log2(coarse_error / fine_error);
  ctx.Check(std::isfinite(observed_order) &&
              observed_order > static_cast<double>(expected_order) - 0.35,
            method_name + " realizes its expected order on y'=y");
}

void CheckNonautonomousObservedOrder(TestContext& ctx,
                                     mfem::ParFiniteElementSpace& fes,
                                     mfem::SSPRKMethod method,
                                     int expected_order,
                                     const std::string& method_name)
{
  const double exact = std::exp(0.5);
  const double coarse_error =
    std::abs(IntegrateNonautonomousManufactured(fes, method, 8) - exact);
  const double fine_error =
    std::abs(IntegrateNonautonomousManufactured(fes, method, 16) - exact);
  const double observed_order = std::log2(coarse_error / fine_error);
  ctx.Check(std::isfinite(observed_order) &&
              observed_order > static_cast<double>(expected_order) - 0.35,
            method_name +
              " realizes its expected order on nonautonomous u'=t*u");
}

void CheckSSPRK104TransactionalRollback(TestContext& ctx,
                                        mfem::ParMesh& mesh,
                                        mfem::ParFiniteElementSpace& fes,
                                        int rejected_rhs_stage,
                                        const std::string& scenario)
{
  constexpr int kInitialStep = 4;
  constexpr std::uint64_t kInitialGeneration = 13;
  constexpr int kRejectedElements = 7;
  constexpr mfem::real_t kInitialTime = 0.25;
  constexpr mfem::real_t kInitialDt = 0.125;
  constexpr mfem::real_t kRetryDt = 0.0625;
  constexpr mfem::real_t kFinalTime = 0.375;
  constexpr int kAcceptedSteps = 2;
  constexpr int kStagesPerAcceptedStep = 10;

  mfem::ParGridFunction state(&fes);
  state.UseDevice(true);
  mfem::real_t* state_values = state.HostWrite();
  for (int i = 0; i < state.Size(); ++i)
  {
    state_values[i] = 1.0 + 0.125 * static_cast<mfem::real_t>(i);
  }
  const mfem::Vector initial_state(state);

  RecordingLinearOperator op(state.Size(), /*time_scaled=*/true);
  mfem::RK3SSPWithLimiter stepper(fes, mfem::SSPRKMethod::SSPRK104);
  stepper.EnablePhysicalBoundaryFluxTracking();
  bool rejection_pending = true;
  int callback_calls = 0;
  int rejection_callback_calls = 0;
  mfem::real_t rejection_callback_time = -1.0;
  stepper.AddAlwaysCallback(
    [&](mfem::ParFiniteElementSpace&,
        mfem::ParGridFunction&,
        double callback_time)
    {
      ++callback_calls;
      if (rejection_pending && op.current_stage == rejected_rhs_stage)
      {
        rejection_pending = false;
        ++rejection_callback_calls;
        rejection_callback_time = callback_time;
        throw mfem::RecoverableLimiterFailure(kRejectedElements);
      }
    });
  stepper.Init(op);
  stepper.SetStepCount(kInitialStep);
  stepper.RestoreStateGeneration(kInitialGeneration);

  dg_sim::TimeLoopConfig config;
  config.initial_step = kInitialStep;
  config.initial_time = kInitialTime;
  config.t_final = kFinalTime;
  config.dt = kInitialDt;
  config.dt_min = 0.01;
  config.cfl = 0.0;
  config.vtk_save = false;

  int rejection_hook_calls = 0;
  bool hook_observed_exact_rollback = true;
  dg_sim::TimeLoopHooks hooks;
  hooks.on_rejected_attempt =
    [&](int step,
        mfem::real_t time,
        mfem::real_t attempted_dt,
        mfem::real_t retry_dt)
    {
      ++rejection_hook_calls;
      hook_observed_exact_rollback =
        hook_observed_exact_rollback && step == kInitialStep &&
        time == kInitialTime && attempted_dt == kInitialDt &&
        retry_dt == kRetryDt &&
        stepper.GetStepCount() == kInitialStep &&
        stepper.GetStateGeneration() == kInitialGeneration &&
        !stepper.HasCumulativePhysicalBoundaryFluxIntegral() &&
        VectorsExactlyEqual(state, initial_state);
    };

  const dg_sim::TimeLoopResult result =
    dg_sim::RunTimeLoop(op, mesh, state, stepper, config, hooks);

  mfem::ParGridFunction reference_state(&fes);
  reference_state.UseDevice(true);
  reference_state = initial_state;
  RecordingLinearOperator reference_op(reference_state.Size(),
                                       /*time_scaled=*/true);
  mfem::RK3SSPWithLimiter reference_stepper(
    fes, mfem::SSPRKMethod::SSPRK104);
  reference_stepper.EnablePhysicalBoundaryFluxTracking();
  reference_stepper.Init(reference_op);
  reference_stepper.SetStepCount(kInitialStep);
  reference_stepper.RestoreStateGeneration(kInitialGeneration);
  dg_sim::TimeLoopConfig reference_config = config;
  reference_config.dt = kRetryDt;
  const dg_sim::TimeLoopResult reference_result = dg_sim::RunTimeLoop(
    reference_op, mesh, reference_state, reference_stepper, reference_config);

  const int rejected_attempt_rhs = rejected_rhs_stage + 1;
  const long long expected_rhs =
    rejected_attempt_rhs + kAcceptedSteps * kStagesPerAcceptedStep;
  ctx.Check(rejection_callback_calls == 1 && !rejection_pending &&
              rejection_callback_time == kFinalTime,
            scenario + " rejects exactly once at the selected callback");
  ctx.Check(rejection_hook_calls == 1 && hook_observed_exact_rollback,
            scenario +
              " restores exact state, time, step, and generation before retry");
  ctx.Check(op.abort_calls == 1 && op.abort_time == kInitialTime &&
              op.abort_generation == kInitialGeneration &&
              op.abort_retry_dt == kRetryDt &&
              VectorsExactlyEqual(op.abort_state, initial_state),
            scenario + " notifies AbortRKStep with the exact rollback state");
  ctx.Check(op.begin_steps ==
              std::vector<int>({kInitialStep, kInitialStep, kInitialStep + 1}) &&
              op.begin_times == std::vector<mfem::real_t>(
                                  {kInitialTime,
                                   kInitialTime,
                                   kInitialTime + kRetryDt}) &&
              op.begin_dts == std::vector<mfem::real_t>(
                                {kInitialDt, kRetryDt, kRetryDt}) &&
              op.begin_stage_counts == std::vector<int>({10, 10, 10}) &&
              op.end_calls == kAcceptedSteps,
            scenario + " retries the uncommitted step before advancing it");
  ctx.Check(result.success && result.steps == kInitialStep + kAcceptedSteps &&
              result.segment_steps == kAcceptedSteps &&
              result.accepted_attempts == kAcceptedSteps &&
              result.rejected_attempts == 1 &&
              result.rejected_limiter_elements == kRejectedElements,
            scenario + " records accepted and rejected attempts exactly");
  ctx.Check(result.rhs_evaluations == expected_rhs &&
              stepper.GetRHSEvaluationCount() == expected_rhs &&
              static_cast<long long>(op.rhs_times.size()) == expected_rhs &&
              callback_calls == expected_rhs,
            scenario + " includes rejected RHS work in bounded accounting");
  ctx.Check(result.minimum_attempted_dt == kRetryDt &&
              result.maximum_attempted_dt == kInitialDt &&
              result.minimum_accepted_dt == kRetryDt &&
              result.maximum_accepted_dt == kRetryDt,
            scenario + " retries and completes at exactly half timestep");
  ctx.Check(stepper.GetStepCount() == kInitialStep + kAcceptedSteps &&
              stepper.GetStateGeneration() ==
                kInitialGeneration +
                  kAcceptedSteps * kStagesPerAcceptedStep &&
              result.final_time == kFinalTime,
            scenario + " commits only the two accepted half steps");
  ctx.Check(reference_result.success &&
              reference_result.rhs_evaluations ==
                kAcceptedSteps * kStagesPerAcceptedStep &&
              reference_stepper.GetStepCount() == stepper.GetStepCount() &&
              reference_stepper.GetStateGeneration() ==
                stepper.GetStateGeneration() &&
              reference_result.final_time == result.final_time &&
              VectorsExactlyEqual(state, reference_state),
            scenario + " agrees exactly with a clean half-timestep run");
  ctx.Check(op.boundary_flux_captures == expected_rhs &&
              reference_op.boundary_flux_captures ==
                kAcceptedSteps * kStagesPerAcceptedStep &&
              stepper.HasCumulativePhysicalBoundaryFluxIntegral() &&
              reference_stepper.HasCumulativePhysicalBoundaryFluxIntegral() &&
              VectorsExactlyEqual(
                stepper.GetCumulativePhysicalBoundaryFluxIntegral(),
                reference_stepper.GetCumulativePhysicalBoundaryFluxIntegral()) &&
              VectorsExactlyEqual(
                stepper.GetCumulativePhysicalBoundaryFluxAbsoluteIntegral(),
                reference_stepper
                  .GetCumulativePhysicalBoundaryFluxAbsoluteIntegral()),
            scenario + " excludes rejected-stage signed and absolute flux "
                       "work from accepted totals");
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const char* device_config = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(device_config);

  TestContext ctx;
  CheckClassicalOrderConditions<mfem::SSPRK33Coefficients>(ctx, "SSPRK(3,3)");
  CheckClassicalOrderConditions<mfem::SSPRK104Coefficients>(ctx,
                                                            "SSPRK(10,4)");
  ctx.Check(mfem::IsIntermediateRKStage(0, 3) &&
              mfem::IsIntermediateRKStage(1, 3) &&
              !mfem::IsIntermediateRKStage(2, 3) &&
              mfem::IsIntermediateRKStage(8, 10) &&
              !mfem::IsIntermediateRKStage(9, 10),
            "stage-count contract identifies every non-final RHS stage");

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian1D(1, 1.0);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial_mesh);
  mfem::L2_FECollection fec(/*order=*/0, /*dim=*/1);
  mfem::ParFiniteElementSpace fes(&mesh, &fec);

  mfem::RK3SSPWithLimiter default_stepper(fes);
  ctx.Check(default_stepper.GetMethod() == mfem::SSPRKMethod::SSPRK33,
            "historical RK stepper defaults to SSPRK(3,3)");

  CheckLifecycle<mfem::SSPRK33Coefficients>(ctx,
                                             fes,
                                             mfem::SSPRKMethod::SSPRK33,
                                             "SSPRK(3,3)");
  CheckLifecycle<mfem::SSPRK104Coefficients>(ctx,
                                              fes,
                                              mfem::SSPRKMethod::SSPRK104,
                                              "SSPRK(10,4)");
  CheckPhysicalBoundaryFluxTracking<mfem::SSPRK33Coefficients>(
    ctx, fes, mfem::SSPRKMethod::SSPRK33, "SSPRK(3,3)");
  CheckPhysicalBoundaryFluxTracking<mfem::SSPRK104Coefficients>(
    ctx, fes, mfem::SSPRKMethod::SSPRK104, "SSPRK(10,4)");
  CheckActualDGStaticSourceSSPRK104(ctx);

  CheckObservedOrder(ctx,
                     fes,
                     mfem::SSPRKMethod::SSPRK33,
                     /*expected_order=*/3,
                     "SSPRK(3,3)");
  CheckObservedOrder(ctx,
                     fes,
                     mfem::SSPRKMethod::SSPRK104,
                     /*expected_order=*/4,
                     "SSPRK(10,4)");

  CheckNonautonomousObservedOrder(ctx,
                                  fes,
                                  mfem::SSPRKMethod::SSPRK33,
                                  /*expected_order=*/3,
                                  "SSPRK(3,3)");
  CheckNonautonomousObservedOrder(ctx,
                                  fes,
                                  mfem::SSPRKMethod::SSPRK104,
                                  /*expected_order=*/4,
                                  "SSPRK(10,4)");

  CheckAcceptedCharacteristicSpeedTransaction(ctx, mesh, fes);
  CheckTerminalAlwaysCallbackOrdering(ctx, fes);
  CheckRejectedLimiterDiagnosticsStayCommitted(ctx);

  CheckSSPRK104TransactionalRollback(
    ctx,
    mesh,
    fes,
    /*rejected_rhs_stage=*/8,
    "SSPRK(10,4) late-intermediate rejection");
  CheckSSPRK104TransactionalRollback(
    ctx,
    mesh,
    fes,
    /*rejected_rhs_stage=*/9,
    "SSPRK(10,4) final-callback rejection");

  int global_failures = 0;
  MPI_Allreduce(&ctx.failures,
                &global_failures,
                1,
                mfem::MPITypeMap<int>::mpi_type,
                MPI_SUM,
                MPI_COMM_WORLD);
  return global_failures == 0 ? 0 : 1;
}
