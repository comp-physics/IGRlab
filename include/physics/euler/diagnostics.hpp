#pragma once

#include "igrlab/compat.hpp"

#include <algorithm>
#include <cmath>
#include "mfem.hpp"
#include "physics/euler/eos.hpp"
#include "utils/config/dg_igr_defaults.hpp"

/**
 * @file physics/euler/diagnostics.hpp
 * @brief Diagnostic coefficients for Euler fields.
 * @ingroup physics_euler
 */

namespace mfem
{

/**
 * @brief Utility coefficient to evaluate pressure p(U) for visualization/debug.
 */
struct PressureCoefficient : public mfem::Coefficient
{
  const mfem::ParGridFunction* U = nullptr;
  int dim = mfem::igrlab::EulerDefaults::kDim;
  double gamma = mfem::igrlab::EulerDefaults::kGamma;
  mutable mfem::Vector state; // reuse buffer

  PressureCoefficient() = default;
  PressureCoefficient(const mfem::ParGridFunction& U_, int dim_, double g_)
    : U(&U_),
      dim(dim_),
      gamma(g_),
      state(dim_ + 2)
  {
  }

  void SetState(const mfem::ParGridFunction& U_) { U = &U_; }
  void SetDimGamma(int d, double g)
  {
    dim = d;
    gamma = g;
    state.SetSize(d + 2);
  }

  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    MFEM_VERIFY(U, "PressureCoefficient: U pointer not set.");
    T.SetIntPoint(&ip);
    U->GetVectorValue(T, ip, state); // [rho, m, E]
    return PressureFromCons(state, dim, gamma, /*rho_floor=*/1e-16);
  }
};

/**
 * @brief Coefficient that returns velocity u = m/rho from conservative state.
 */
class VelocityFromConsCoefficient : public mfem::VectorCoefficient
{
public:
  VelocityFromConsCoefficient(int dim,
                              const mfem::ParGridFunction* U,
                              double rho_floor)
    : mfem::VectorCoefficient(dim),
      dim_(dim),
      U_(U),
      rho_floor_(rho_floor)
  {
  }

  void SetState(const mfem::ParGridFunction* U) { U_ = U; }
  int GetStateDimension() const { return dim_; }
  double GetDensityFloor() const { return rho_floor_; }

  void Eval(mfem::Vector& V,
            mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override
  {
    MFEM_VERIFY(U_, "VelocityFromConsCoefficient: U is null");
    T.SetIntPoint(&ip);
    mfem::Vector state;
    U_->GetVectorValue(T, ip, state); // [rho, m(0..dim-1), E]
    const double rho = std::max(state(0), rho_floor_);
    V.SetSize(dim_);
    for (int d = 0; d < dim_; ++d)
    {
      V(d) = state(1 + d) / rho;
    }
  }

private:
  int dim_;
  const mfem::ParGridFunction* U_;
  double rho_floor_;
};

// --- Diagnostics from an H1 vector velocity field (uH1) ---
struct GradFrobNormCoeff : public mfem::Coefficient
{
  const mfem::ParGridFunction* v = nullptr;
  int dim = 0;
  GradFrobNormCoeff(const mfem::ParGridFunction* vel, int d)
    : v(vel),
      dim(d)
  {
  }
  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    if (!v) return 0.0;
    T.SetIntPoint(&ip);
    mfem::DenseMatrix G(dim, dim);
    v->GetVectorGradient(T, G); // G(i,j) = d_j u_i
    double s = 0.0;
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        s += G(i, j) * G(i, j);
    return std::sqrt(s);
  }
};

struct DivCoeff : public mfem::Coefficient
{
  const mfem::ParGridFunction* v = nullptr;
  int dim = 0;
  DivCoeff(const mfem::ParGridFunction* vel, int d)
    : v(vel),
      dim(d)
  {
  }
  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    if (!v) return 0.0;
    T.SetIntPoint(&ip);
    mfem::DenseMatrix G(dim, dim);
    v->GetVectorGradient(T, G);
    double tr = 0.0;
    for (int i = 0; i < dim; i++)
      tr += G(i, i);
    return tr;
  }
};

struct VelMagCoeff : public mfem::Coefficient
{
  const mfem::ParGridFunction* v = nullptr;
  int dim = 0;
  VelMagCoeff(const mfem::ParGridFunction* vel, int d)
    : v(vel),
      dim(d)
  {
  }
  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    if (!v) return 0.0;
    T.SetIntPoint(&ip);
    mfem::Vector u;
    v->GetVectorValue(T, ip, u);
    return u.Norml2();
  }
};

// --- Diagnostics directly from conservative state (broken/DG gradient path) ---
class DivFromConservativeCoeff : public mfem::Coefficient
{
public:
  DivFromConservativeCoeff(const mfem::ParGridFunction* U,
                           int dim,
                           double rho_floor)
    : U_(U),
      dim_(dim),
      rho_floor_(rho_floor),
      state_(dim + 2),
      grad_u_(dim + 2, dim),
      gradu_(dim, dim)
  {
  }

  void SetState(const mfem::ParGridFunction* U) { U_ = U; }

  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    MFEM_VERIFY(U_ != nullptr, "DivFromConservativeCoeff: U is null");
    T.SetIntPoint(&ip);
    U_->GetVectorValue(T, ip, state_);
    U_->GetVectorGradient(T, grad_u_);

    const double rho = std::max(state_(0), rho_floor_);
    const double rho2 = rho * rho;
    for (int i = 0; i < dim_; ++i)
    {
      const double mi = state_(1 + i);
      for (int j = 0; j < dim_; ++j)
      {
        const double dmij = grad_u_(1 + i, j);
        const double drhoj = grad_u_(0, j);
        gradu_(i, j) = (rho * dmij - mi * drhoj) / rho2;
      }
    }

    double divu = 0.0;
    for (int i = 0; i < dim_; ++i)
    {
      divu += gradu_(i, i);
    }
    return divu;
  }

private:
  const mfem::ParGridFunction* U_ = nullptr;
  int dim_ = 0;
  double rho_floor_ = 1.0e-12;
  mutable mfem::Vector state_;
  mutable mfem::DenseMatrix grad_u_;
  mutable mfem::DenseMatrix gradu_;
};

class GradFrobFromConservativeCoeff : public mfem::Coefficient
{
public:
  GradFrobFromConservativeCoeff(const mfem::ParGridFunction* U,
                                int dim,
                                double rho_floor)
    : U_(U),
      dim_(dim),
      rho_floor_(rho_floor),
      state_(dim + 2),
      grad_u_(dim + 2, dim),
      gradu_(dim, dim)
  {
  }

  void SetState(const mfem::ParGridFunction* U) { U_ = U; }

  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
  {
    MFEM_VERIFY(U_ != nullptr, "GradFrobFromConservativeCoeff: U is null");
    T.SetIntPoint(&ip);
    U_->GetVectorValue(T, ip, state_);
    U_->GetVectorGradient(T, grad_u_);

    const double rho = std::max(state_(0), rho_floor_);
    const double rho2 = rho * rho;
    for (int i = 0; i < dim_; ++i)
    {
      const double mi = state_(1 + i);
      for (int j = 0; j < dim_; ++j)
      {
        const double dmij = grad_u_(1 + i, j);
        const double drhoj = grad_u_(0, j);
        gradu_(i, j) = (rho * dmij - mi * drhoj) / rho2;
      }
    }

    double sq = 0.0;
    for (int i = 0; i < dim_; ++i)
      for (int j = 0; j < dim_; ++j)
        sq += gradu_(i, j) * gradu_(i, j);
    return std::sqrt(sq);
  }

private:
  const mfem::ParGridFunction* U_ = nullptr;
  int dim_ = 0;
  double rho_floor_ = 1.0e-12;
  mutable mfem::Vector state_;
  mutable mfem::DenseMatrix grad_u_;
  mutable mfem::DenseMatrix gradu_;
};

} // namespace mfem
