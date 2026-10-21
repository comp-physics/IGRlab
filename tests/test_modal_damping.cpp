#include "igrlab/environment.hpp"
#include "numerics/dg/modal_damping.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/bc/boundary_state.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace
{
using dg_igr_tests::TestContext;
using mfem::real_t;

// Analytic polynomials are sampled independently of the damping transform.
real_t Legendre(int degree, real_t x)
{
  real_t previous = 1.0;
  if (degree == 0) { return previous; }
  real_t current = x;
  for (int k = 1; k < degree; ++k)
  {
    const real_t next = ((2 * k + 1) * x * current - k * previous) / (k + 1);
    previous = current;
    current = next;
  }
  return current;
}

using Polynomial = std::function<real_t(int, int, const mfem::IntegrationPoint&)>;

void Sample(mfem::FiniteElementSpace& fes,
            const Polynomial& polynomial,
            mfem::Vector& values)
{
  values.SetSize(fes.GetVSize());
  real_t* data = values.HostWrite();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    for (int c = 0; c < fes.GetVDim(); ++c)
    {
      for (int i = 0; i < dofs.Size(); ++i)
      {
        data[fes.DofToVDof(dofs[i], c)] = polynomial(e, c, nodes.IntPoint(i));
      }
    }
  }
}

void CheckVectors(TestContext& ctx, const mfem::Vector& actual,
                  const mfem::Vector& expected, real_t tolerance,
                  const std::string& message)
{
  mfem::Vector difference(actual);
  difference -= expected;
  const real_t scale = std::max(real_t(1.0), expected.Normlinf());
  ctx.Check(std::isfinite(difference.Normlinf()) &&
              difference.Normlinf() <= tolerance * scale,
            message + " (max error=" +
              dg_igr_tests::FormatSci(difference.Normlinf()) + ")");
}

void CheckCellMeans(TestContext& ctx, mfem::FiniteElementSpace& fes,
                    const mfem::Vector& derivative, const std::string& label)
{
  real_t maximum_integral = 0.0;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    const auto& fe = *fes.GetFE(e);
    const auto& rule = mfem::IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder() + 2);
    auto* transformation = fes.GetElementTransformation(e);
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    mfem::Vector shape(fe.GetDof());
    for (int c = 0; c < fes.GetVDim(); ++c)
    {
      real_t integral = 0.0;
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const auto& ip = rule.IntPoint(q);
        transformation->SetIntPoint(&ip);
        fe.CalcShape(ip, shape);
        real_t value = 0.0;
        for (int i = 0; i < dofs.Size(); ++i)
        {
          value += shape(i) * derivative(fes.DofToVDof(dofs[i], c));
        }
        integral += ip.weight * transformation->Weight() * value;
      }
      maximum_integral = std::max(maximum_integral, std::abs(integral));
    }
  }
  ctx.Check(maximum_integral < 2.0e-10,
            label + " preserves each physical cell/component mean (max integral=" +
              dg_igr_tests::FormatSci(maximum_integral) + ")");
}

void CheckP39(TestContext& ctx, int basis, int ordering)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(3, 1.0);
  mesh.GetVertex(1)[0] = 0.07;
  mesh.GetVertex(2)[0] = 0.43;
  mfem::L2_FECollection fec(39, 1, basis);
  mfem::FiniteElementSpace fes(&mesh, &fec, 3, ordering);
  mfem::ConservativeModalDamping damping(fes, 80000.0);
  const std::string label = std::string(basis == mfem::BasisType::GaussLobatto
                                        ? "p39 LGL" : "p39 GL") +
                            (ordering == mfem::Ordering::byNODES ? " byNODES" : " byVDIM");
  mfem::Vector state, rhs(fes.GetVSize()), expected;
  Sample(fes, [](int e, int c, const mfem::IntegrationPoint& ip)
  {
    const real_t x = 2.0 * ip.x - 1.0;
    const real_t a = (e + 1) * (c + 1) * 0.001;
    return 1.0 + c + a * (Legendre(29, x) + 2.0 * Legendre(30, x) -
                           3.0 * Legendre(39, x));
  }, state);
  Sample(fes, [](int e, int c, const mfem::IntegrationPoint& ip)
  {
    const real_t x = 2.0 * ip.x - 1.0;
    const real_t a = (e + 1) * (c + 1) * 0.001;
    // At p39 the measured M16 rates for k30 and k39 are 8 and 80000.
    return -16.0 * a * Legendre(30, x) + 240000.0 * a * Legendre(39, x);
  }, expected);
  rhs = 0.0;
  damping.AddMult(state, rhs);
  CheckVectors(ctx, rhs, expected, 3.0e-11,
                label + " damps the physical polynomial after mass scaling");
  CheckCellMeans(ctx, fes, rhs, label);

  mfem::Vector baseline(fes.GetVSize());
  baseline = 7.0;
  rhs = baseline;
  damping.AddMult(state, rhs);
  expected += baseline;
  CheckVectors(ctx, rhs, expected, 3.0e-11, label + " adds to existing RHS");

  Sample(fes, [](int, int c, const mfem::IntegrationPoint& ip)
  {
    const real_t x = 2.0 * ip.x - 1.0;
    return (c + 1) * (1.0 + 0.3 * Legendre(2, x) + 0.2 * Legendre(29, x));
  }, state);
  rhs = 0.0;
  damping.AddMult(state, rhs);
  ctx.Check(rhs.Normlinf() < 2.0e-8, label + " leaves all represented low modes unchanged");
}

void CheckTensor(TestContext& ctx, int dim, int basis)
{
  mfem::Mesh mesh = dim == 2
    ? mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, true, 0.7, 1.3)
    : mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON, 0.7, 1.3, 1.9);
  mfem::L2_FECollection fec(4, dim, basis);
  mfem::FiniteElementSpace fes(&mesh, &fec, dim + 2, mfem::Ordering::byVDIM);
  mfem::ConservativeModalDamping damping(fes, 80000.0);
  const std::string label = std::to_string(dim) + "D " +
    (basis == mfem::BasisType::GaussLobatto ? "LGL" : "GL");
  auto tensor = [dim](const mfem::IntegrationPoint& ip)
  {
    real_t value = Legendre(4, 2 * ip.x - 1) * Legendre(4, 2 * ip.y - 1);
    if (dim == 3) { value *= Legendre(4, 2 * ip.z - 1); }
    return value;
  };
  mfem::Vector state, expected, rhs(fes.GetVSize());
  Sample(fes, [&](int e, int c, const mfem::IntegrationPoint& ip)
  {
    return 1.0 + (e + 1) * (c + 1) * 0.001 *
      (tensor(ip) + Legendre(3, 2 * ip.x - 1));
  }, state);
  Sample(fes, [&](int e, int c, const mfem::IntegrationPoint& ip)
  {
    // q(max directional degree): one top-mode rate, not dim copies of it.
    // p4,k3 has fractional ramp origin 116/39, hence q=(1/40)^4.
    return -(e + 1) * (c + 1) * 0.001 *
      (80000.0 * tensor(ip) + 0.03125 * Legendre(3, 2 * ip.x - 1));
  }, expected);
  rhs = 0.0;
  damping.AddMult(state, rhs);
  CheckVectors(ctx, rhs, expected, 3.0e-11, label + " uses max-index tensor cutoff");
  CheckCellMeans(ctx, fes, rhs, label);
}

constexpr real_t curved_x_amplitude = 0.8;
constexpr real_t curved_y_amplitude = 0.6;

// The left cell is the identity map. The right cell is
// (1+s, t) + (0.8, 0.6) s(1-s)t(1-t), an exactly Q2 geometry.
// Its signed determinant is analytic and bounded below by 0.65.
real_t CurvedTestJacobian(int element, const mfem::IntegrationPoint& ip)
{
  if (element == 0) { return 1.0; }
  return 1.0 + curved_x_amplitude * (1.0 - 2.0 * ip.x) * ip.y * (1.0 - ip.y) +
    curved_y_amplitude * ip.x * (1.0 - ip.x) * (1.0 - 2.0 * ip.y);
}

void CheckCurvedTestGeometry(TestContext& ctx, mfem::FiniteElementSpace& fes,
                             const std::string& label)
{
  real_t maximum_geometry_error = 0.0;
  real_t minimum_determinant = 1.0;
  std::array<real_t, 2> jacobian_variation{{0.0, 0.0}};
  const auto& rule = mfem::IntRules.Get(mfem::Geometry::SQUARE, 22);
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    auto* transformation = fes.GetElementTransformation(e);
    transformation->SetIntPoint(&rule.IntPoint(0));
    const mfem::DenseMatrix first(transformation->Jacobian());
    mfem::Vector point(2);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const auto& ip = rule.IntPoint(q);
      const real_t bubble = e == 0 ? 0.0 :
        ip.x * (1.0 - ip.x) * ip.y * (1.0 - ip.y);
      const real_t bubble_x = e == 0 ? 0.0 :
        (1.0 - 2.0 * ip.x) * ip.y * (1.0 - ip.y);
      const real_t bubble_y = e == 0 ? 0.0 :
        ip.x * (1.0 - ip.x) * (1.0 - 2.0 * ip.y);
      transformation->Transform(ip, point);
      maximum_geometry_error = std::max({maximum_geometry_error,
        std::abs(point(0) - (e + ip.x + curved_x_amplitude * bubble)),
        std::abs(point(1) - (ip.y + curved_y_amplitude * bubble))});
      transformation->SetIntPoint(&ip);
      const auto& jacobian = transformation->Jacobian();
      const std::array<real_t, 4> expected{{
        1.0 + curved_x_amplitude * bubble_x, curved_y_amplitude * bubble_x,
        curved_x_amplitude * bubble_y, 1.0 + curved_y_amplitude * bubble_y}};
      for (int j = 0; j < 2; ++j)
      {
        for (int i = 0; i < 2; ++i)
        {
          maximum_geometry_error = std::max(maximum_geometry_error,
            std::abs(jacobian(i, j) - expected[i + 2 * j]));
          jacobian_variation[e] = std::max(jacobian_variation[e],
            std::abs(jacobian(i, j) - first(i, j)));
        }
      }
      const real_t determinant = jacobian(0, 0) * jacobian(1, 1) -
                                 jacobian(0, 1) * jacobian(1, 0);
      minimum_determinant = std::min(minimum_determinant, determinant);
      maximum_geometry_error = std::max(maximum_geometry_error,
        std::abs(determinant - CurvedTestJacobian(e, ip)));
    }
  }
  ctx.Check(maximum_geometry_error < 5.0e-13 && minimum_determinant > 0.6,
            label + " has the intended positive analytic Q2 geometry");
  ctx.Check(jacobian_variation[0] < 5.0e-13 && jacobian_variation[1] > 0.01,
            label + " contains both an affine and a non-affine cell");
}

void CheckCurvedPhysicalInvariants(TestContext& ctx, mfem::FiniteElementSpace& fes,
                                   const mfem::Vector& state,
                                   const mfem::Vector& source, real_t rate,
                                   const std::string& label)
{
  const real_t* state_data = state.HostRead();
  const real_t* source_data = source.HostRead();
  real_t maximum_scaled_mean = 0.0;
  real_t maximum_scaled_power = 0.0;
  real_t maximum_scaled_rate_defect = 0.0;
  real_t maximum_relative_energy_increase = 0.0;
  bool strictly_dissipative = true;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    const auto& fe = *fes.GetFE(e);
    // Higher quadrature than the physical Gram construction, using the
    // independently known signed Jacobian rather than its cached metric.
    const auto& rule = mfem::IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder() + 12);
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    mfem::Vector shape(fe.GetDof());
    for (int c = 0; c < fes.GetVDim(); ++c)
    {
      real_t mean = 0.0;
      real_t source_l1 = 0.0;
      real_t energy = 0.0;
      real_t power = 0.0;
      real_t source_energy = 0.0;
      real_t updated_energy = 0.0;
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const auto& ip = rule.IntPoint(q);
        fe.CalcShape(ip, shape);
        real_t value = 0.0;
        real_t derivative = 0.0;
        for (int i = 0; i < dofs.Size(); ++i)
        {
          const int vdof = fes.DofToVDof(dofs[i], c);
          value += shape(i) * state_data[vdof];
          derivative += shape(i) * source_data[vdof];
        }
        const real_t weight = ip.weight * CurvedTestJacobian(e, ip);
        const real_t updated = value + derivative / rate;
        mean += weight * derivative;
        source_l1 += weight * std::abs(derivative);
        energy += weight * value * value;
        power += weight * value * derivative;
        source_energy += weight * derivative * derivative;
        updated_energy += weight * updated * updated;
      }
      maximum_scaled_mean = std::max(maximum_scaled_mean,
        std::abs(mean) / std::max(real_t(1.0), source_l1));
      maximum_scaled_power = std::max(maximum_scaled_power,
        power / std::max(real_t(1.0), rate * energy));
      // For a self-adjoint damping operator with eigenrates <= rate,
      // ||source||_M^2 <= -rate <state, source>_M.
      maximum_scaled_rate_defect = std::max(maximum_scaled_rate_defect,
        (source_energy + rate * power) /
          std::max(real_t(1.0), rate * rate * energy));
      maximum_relative_energy_increase = std::max(maximum_relative_energy_increase,
        (updated_energy - energy) / std::max(real_t(1.0), energy));
      strictly_dissipative = strictly_dissipative && std::isfinite(energy) &&
        std::isfinite(power) && std::isfinite(source_energy) &&
        std::isfinite(updated_energy) && power < 0.0 && updated_energy < energy;
    }
  }
  ctx.Check(maximum_scaled_mean < 2.0e-11,
            label + " preserves every physical cell/component mean (scaled error=" +
              dg_igr_tests::FormatSci(maximum_scaled_mean) + ")");
  ctx.Check(maximum_scaled_power < 3.0e-12 && strictly_dissipative,
            label + " dissipates the independently integrated physical quadratic energy");
  ctx.Check(maximum_scaled_rate_defect < 3.0e-12,
            label + " respects the physical damping-rate bound");
  ctx.Check(maximum_relative_energy_increase < 5.0e-12 && strictly_dissipative,
            label + " is physically contractive for forward Euler at dt=1/rate");
}

void CheckCurvedTensor(TestContext& ctx, int basis, int ordering)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    2, 1, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  serial.SetCurvature(2, false, 2, mfem::Ordering::byNODES);
  serial.Transform([](const mfem::Vector& reference, mfem::Vector& physical)
  {
    physical = reference;
    if (reference(0) > 1.0)
    {
      const real_t s = reference(0) - 1.0;
      const real_t t = reference(1);
      const real_t bubble = s * (1.0 - s) * t * (1.0 - t);
      physical(0) += curved_x_amplitude * bubble;
      physical(1) += curved_y_amplitude * bubble;
    }
  });
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  mfem::L2_FECollection fec(5, 2, basis);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 4, ordering);
  const std::string label = std::string("mixed Q2 p5 ") +
    (basis == mfem::BasisType::GaussLobatto ? "LGL" : "GL") +
    (ordering == mfem::Ordering::byNODES ? " byNODES" : " byVDIM");
  CheckCurvedTestGeometry(ctx, fes, label);
  constexpr real_t rate = 80000.0;
  mfem::ConservativeModalDamping damping(fes, rate);
  mfem::Vector state, rhs(fes.GetVSize()), original_rhs;
  Sample(fes, [](int e, int c, const mfem::IntegrationPoint&)
  {
    return 0.5 + (e + 1) * (c + 1);
  }, state);
  rhs = 3.0;
  original_rhs = rhs;
  damping.AddMult(state, rhs);
  CheckVectors(ctx, rhs, original_rhs, 0.0,
                label + " gives exact zero damping for cellwise constants");

  real_t maximum_low_mode_error = 0.0;
  for (int block = 0; block < 4; ++block)
  {
    Sample(fes, [block](int e, int c, const mfem::IntegrationPoint& ip)
    {
      const int mode = 4 * block + c;
      return (e + 1) * (c + 1) *
        Legendre(mode % 4, 2.0 * ip.x - 1.0) *
        Legendre(mode / 4, 2.0 * ip.y - 1.0);
    }, state);
    rhs = 0.0;
    damping.AddMult(state, rhs);
    maximum_low_mode_error = std::max(maximum_low_mode_error,
      rhs.Normlinf() / std::max(real_t(1.0), state.Normlinf()));
  }
  ctx.Check(maximum_low_mode_error < 5.0e-9,
            label + " leaves the complete reference Q3 subspace unchanged");

  for (int probe = 0; probe < 2; ++probe)
  {
    Sample(fes, [probe](int e, int c, const mfem::IntegrationPoint& ip)
    {
      const real_t x = 2.0 * ip.x - 1.0;
      const real_t y = 2.0 * ip.y - 1.0;
      const real_t amplitude = 0.1 * (e + 1) * (c + 1);
      // The first probe is anti-dissipative under an ordinary reference
      // diagonal filter on this curved cell, despite preserving its mean.
      const real_t polynomial = probe == 0
        ? 20.0 * Legendre(3, x) + Legendre(4, x)
        : 0.6 * Legendre(3, x) * Legendre(3, y) +
          0.7 * Legendre(5, x) * Legendre(2, y) -
          0.4 * Legendre(1, x) * Legendre(5, y) +
          0.3 * Legendre(4, x) * Legendre(4, y);
      return 1.0 + c + amplitude * polynomial;
    }, state);
    rhs = 0.0;
    damping.AddMult(state, rhs);
    const std::string probe_label = label + " probe " + std::to_string(probe);
    CheckCurvedPhysicalInvariants(ctx, fes, state, rhs, rate, probe_label);

    mfem::Vector affine_expected;
    Sample(fes, [probe](int e, int c, const mfem::IntegrationPoint& ip)
    {
      const real_t x = 2.0 * ip.x - 1.0;
      const real_t y = 2.0 * ip.y - 1.0;
      const real_t amplitude = 0.1 * (e + 1) * (c + 1);
      // At p5 the independent M16 k4 weight is (11/50)^4.
      constexpr real_t q4 = 0.00234256;
      const real_t polynomial = probe == 0
        ? q4 * Legendre(4, x)
        : 0.7 * Legendre(5, x) * Legendre(2, y) -
          0.4 * Legendre(1, x) * Legendre(5, y) +
          0.3 * q4 * Legendre(4, x) * Legendre(4, y);
      return -80000.0 * amplitude * polynomial;
    }, affine_expected);
    mfem::Array<int> affine_dofs;
    fes.GetElementDofs(0, affine_dofs);
    real_t maximum_affine_error = 0.0;
    for (int c = 0; c < fes.GetVDim(); ++c)
    {
      for (int i = 0; i < affine_dofs.Size(); ++i)
      {
        const int vdof = fes.DofToVDof(affine_dofs[i], c);
        maximum_affine_error = std::max(maximum_affine_error,
          std::abs(rhs(vdof) - affine_expected(vdof)));
      }
    }
    ctx.Check(maximum_affine_error < 3.0e-11 *
                std::max(real_t(1.0), affine_expected.Normlinf()),
              probe_label + " retains the affine tensor filter on the same mesh");
  }
}

void CheckConstantsAndZeroRate(TestContext& ctx)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(2, 1.0);
  mfem::L2_FECollection constant_fec(0, 1);
  mfem::FiniteElementSpace constant_fes(&mesh, &constant_fec, 3);
  mfem::Vector constant_state, rhs(constant_fes.GetVSize()), original_rhs;
  Sample(constant_fes, [](int e, int c, const mfem::IntegrationPoint&)
  {
    return 1.0 + e + c;
  }, constant_state);
  rhs = 3.0;
  original_rhs = rhs;
  mfem::ConservativeModalDamping constants(constant_fes, 80000.0);
  constants.AddMult(constant_state, rhs);
  CheckVectors(ctx, rhs, original_rhs, 0.0,
                "p0 damping leaves the complete existing RHS exactly unchanged");

  mfem::L2_FECollection fec(4, 1, mfem::BasisType::GaussLobatto);
  mfem::FiniteElementSpace fes(&mesh, &fec, 3);
  mfem::Vector state;
  Sample(fes, [](int e, int c, const mfem::IntegrationPoint& ip)
  {
    return (e + 1) * (c + 1) * Legendre(4, 2 * ip.x - 1);
  }, state);
  rhs.SetSize(fes.GetVSize());
  rhs = 2.0;
  original_rhs = rhs;
  mfem::ConservativeModalDamping disabled(fes, 0.0);
  disabled.AddMult(state, rhs);
  CheckVectors(ctx, rhs, original_rhs, 0.0,
                "zero rate is an exact no-op even for high-mode input");
}

class DampingOnlyOperator final : public mfem::TimeDependentOperator
{
public:
  explicit DampingOnlyOperator(mfem::FiniteElementSpace& fes)
    : mfem::TimeDependentOperator(fes.GetVSize()), damping_(fes, 80000.0) {}
  void Mult(const mfem::Vector& state, mfem::Vector& rhs) const override
  {
    rhs = 0.0;
    damping_.AddMult(state, rhs);
    ++calls;
  }
  mutable int calls = 0;
private:
  mfem::ConservativeModalDamping damping_;
};

void CheckSSPRK(TestContext& ctx)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian1D(2, 0.7);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  mfem::L2_FECollection fec(39, 1, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 3, mfem::Ordering::byNODES);
  mfem::Vector state, expected;
  constexpr real_t dt_value = 1.0e-6;
  auto amplification = [](real_t rate)
  {
    const real_t z = -rate * dt_value;
    return 1.0 + z + z * z / 2.0 + z * z * z / 6.0;
  };
  Sample(fes, [](int, int c, const mfem::IntegrationPoint& ip)
  {
    const real_t x = 2 * ip.x - 1;
    return c + 1.0 + 0.01 * (Legendre(29, x) + Legendre(30, x) + Legendre(39, x));
  }, state);
  Sample(fes, [&](int, int c, const mfem::IntegrationPoint& ip)
  {
    const real_t x = 2 * ip.x - 1;
    return c + 1.0 + 0.01 * (Legendre(29, x) + amplification(8.0) * Legendre(30, x) +
                             amplification(80000.0) * Legendre(39, x));
  }, expected);
  DampingOnlyOperator op(fes);
  mfem::RK3SSPWithLimiter stepper(fes);
  stepper.Init(op);
  real_t time = 0.0;
  real_t dt = dt_value;
  stepper.Step(state, time, dt);
  ctx.Check(op.calls == 3, "SSPRK33 evaluates the damped RHS at all three stages");
  CheckVectors(ctx, state, expected, 2.0e-13,
                "SSPRK33 matches cubic modal amplification, not a post-step exponential filter");
}

void CheckPhysicalOperator(TestContext& ctx, bool preassembled, bool dgsem,
                           bool fallback = false)
{
  constexpr int dim = 2;
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    2, 1, mfem::Element::QUADRILATERAL, true, 0.7, 1.1);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  mfem::L2_FECollection fec(4, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, dim + 2, mfem::Ordering::byNODES);
  mfem::EulerFlux flux(dim, 1.4);
  mfem::RusanovFluxWithSides numerical_flux(flux);
  auto integrator = std::make_unique<mfem::HyperbolicFormIntegrator>(numerical_flux, 2);
  mfem::DGHyperbolicConservationLaws op(fes, std::move(integrator), preassembled);
  mfem::Array<int> boundary_marker(mesh.bdr_attributes.Max());
  boundary_marker = 1;
  op.AddBdrFaceIntegrator(new mfem::BdrHyperbolicFluxIntegrator(
    numerical_flux, std::make_unique<mfem::SlipWallState>(dim), 2), boundary_marker);
  if (dgsem)
  {
    op.SetNumericalFlux(&numerical_flux);
    op.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    if (fallback)
    {
      op.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Chandrashekar);
    }
  }
  const std::string label = dgsem ? (fallback ? "DGSEM host fallback" : "DGSEM packed") :
    (preassembled ? "preassembled weak DG" : "matrix-free weak DG");
  mfem::Vector state, expected, native(fes.GetTrueVSize()), damped(fes.GetTrueVSize());
  const Polynomial initial_polynomial = [](int e, int c, const mfem::IntegrationPoint& ip)
  {
    const std::array<real_t, 4> base{{1.0, 0.2, 0.1, 3.0}};
    return base[c] + 0.001 * (e + 1) * (c + 1) *
      Legendre(4, 2 * ip.x - 1) * Legendre(3, 2 * ip.y - 1);
  };
  const Polynomial damping_polynomial = [](int e, int c, const mfem::IntegrationPoint& ip)
  {
    return -80.0 * (e + 1) * (c + 1) *
      Legendre(4, 2 * ip.x - 1) * Legendre(3, 2 * ip.y - 1);
  };
  Sample(fes, initial_polynomial, state);
  Sample(fes, damping_polynomial, expected);
  ctx.Check(op.GetModalDampingRate() == 0.0, label + " defaults to damping off");
  ctx.Check(op.LimitProposedRKStepSize(0.02) == 0.02,
            label + " leaves timestep proposals unchanged when damping is off");
  op.Mult(state, native);
  op.SetModalDampingRate(80000.0);
  ctx.Check(op.GetModalDampingRate() == 80000.0,
            label + " records the physical damping rate");
  ctx.CheckNear(op.LimitProposedRKStepSize(0.02), 1.0 / 80000.0, 1.0e-15,
                label + " caps the timestep at the isolated forward-Euler damping bound");
  ctx.Check(op.LimitProposedRKStepSize(1.0e-7) == 1.0e-7,
            label + " retains a smaller CFL or save-time timestep");
  op.Mult(state, damped);
  if (dgsem)
  {
    const auto bridge = op.GetDGSEMProductionBridgeEvidence();
    ctx.Check(fallback ? bridge.fallback_mult_calls > 0 : bridge.production_mult_calls > 0,
              label + " exercises the intended physical RHS branch");
  }
  damped -= native;
  CheckVectors(ctx, damped, expected, 3.0e-11,
                label + " includes the conservative damping in its actual RHS");
  op.SetModalDampingRate(0.0);
  op.Mult(state, damped);
  CheckVectors(ctx, damped, native, 0.0, label + " disabling restores the original RHS");

  op.SetModalDampingRate(80000.0);
  mesh.UniformRefinement();
  fes.Update();
  op.Update();
  ctx.Check(op.GetModalDampingRate() == 80000.0,
            label + " retains the chosen rate across a mesh/space update");
  Sample(fes, initial_polynomial, state);
  Sample(fes, damping_polynomial, expected);
  native.SetSize(fes.GetTrueVSize());
  damped.SetSize(fes.GetTrueVSize());
  op.Mult(state, damped);
  op.SetModalDampingRate(0.0);
  op.Mult(state, native);
  damped -= native;
  CheckVectors(ctx, damped, expected, 3.0e-11,
                label + " rebuilds damping maps after refinement");
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const char* backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend);
  TestContext ctx;
  ctx.Check(mfem::ConservativeModalDamping::ModeWeight(0, 0) == 0.0,
            "piecewise constants receive no damping");
  ctx.Check(mfem::ConservativeModalDamping::ModeWeight(39, 29) == 0.0,
            "p39 cutoff preserves k29");
  ctx.CheckNear(mfem::ConservativeModalDamping::ModeWeight(39, 30), 0.0001,
                1.0e-15, "p39 k30 has the M16 ramp weight");
  ctx.CheckNear(mfem::ConservativeModalDamping::ModeWeight(39, 39), 1.0,
                1.0e-15, "highest mode gets the selected physical rate");
  for (int basis : {mfem::BasisType::GaussLegendre, mfem::BasisType::GaussLobatto})
  {
    for (int ordering : {mfem::Ordering::byNODES, mfem::Ordering::byVDIM})
    {
      CheckP39(ctx, basis, ordering);
      CheckCurvedTensor(ctx, basis, ordering);
    }
    CheckTensor(ctx, 2, basis);
    CheckTensor(ctx, 3, basis);
  }
  CheckConstantsAndZeroRate(ctx);
  CheckSSPRK(ctx);
  CheckPhysicalOperator(ctx, true, false);
  CheckPhysicalOperator(ctx, false, false);
  CheckPhysicalOperator(ctx, true, true);
  const char* previous_strict = ::igrlab::GetEnv("DG_IGR_DGSEM_PRODUCTION_STRICT");
  const std::optional<std::string> saved_strict = previous_strict
    ? std::optional<std::string>(previous_strict) : std::nullopt;
  setenv("DG_IGR_DGSEM_PRODUCTION_STRICT", "0", 1);
  CheckPhysicalOperator(ctx, true, true, true);
  if (saved_strict) { setenv("DG_IGR_DGSEM_PRODUCTION_STRICT", saved_strict->c_str(), 1); }
  else { unsetenv("DG_IGR_DGSEM_PRODUCTION_STRICT"); }
  return ctx.failures == 0 ? 0 : 1;
}
