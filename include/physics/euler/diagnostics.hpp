#pragma once

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
  const mfem::ParGridFunction *U = nullptr;
  int dim = mfem::dg_igr::EulerDefaults::kDim;
  double gamma = mfem::dg_igr::EulerDefaults::kGamma;
  mutable mfem::Vector state; // reuse buffer

  PressureCoefficient() = default;
  PressureCoefficient(const mfem::ParGridFunction &U_, int dim_, double g_)
    : U(&U_), dim(dim_), gamma(g_), state(dim_ + 2) {}

  void SetState(const mfem::ParGridFunction &U_) { U = &U_; }
  void SetDimGamma(int d, double g) { dim = d; gamma = g; state.SetSize(d + 2); }

  double Eval(mfem::ElementTransformation &T,
              const mfem::IntegrationPoint &ip) override
  {
    MFEM_VERIFY(U, "PressureCoefficient: U pointer not set.");
    T.SetIntPoint(&ip);
    U->GetVectorValue(T, ip, state);   // [rho, m, E]
    return PressureFromCons(state, dim, gamma, /*rho_floor=*/1e-16);
  }
};

/**
 * @brief Coefficient that returns velocity u = m/rho from conservative state.
 */
class VelocityFromConsCoefficient : public mfem::VectorCoefficient
{
public:
  VelocityFromConsCoefficient(int dim, const mfem::ParGridFunction* U,
                              double rho_floor)
    : mfem::VectorCoefficient(dim), dim_(dim), U_(U), rho_floor_(rho_floor) {}

  void SetState(const mfem::ParGridFunction* U) { U_ = U; }

  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override
  {
    MFEM_VERIFY(U_, "VelocityFromConsCoefficient: U is null");
    T.SetIntPoint(&ip);
    mfem::Vector state;
    U_->GetVectorValue(T, ip, state); // [rho, m(0..dim-1), E]
    const double rho = std::max(state(0), rho_floor_);
    V.SetSize(dim_);
    for (int d = 0; d < dim_; ++d) { V(d) = state(1 + d) / rho; }
  }

private:
  int dim_;
  const mfem::ParGridFunction* U_;
  double rho_floor_;
};

// --- Diagnostics from an H1 vector velocity field (uH1) ---
struct GradFrobNormCoeff : public mfem::Coefficient
{
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  GradFrobNormCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override
  {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::DenseMatrix G(dim, dim);
    v->GetVectorGradient(T, G); // G(i,j) = d_j u_i
    double s = 0.0; for (int i=0;i<dim;i++) for (int j=0;j<dim;j++) s += G(i,j)*G(i,j);
    return std::sqrt(s);
  }
};

struct DivCoeff : public mfem::Coefficient
{
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  DivCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override
  {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::DenseMatrix G(dim, dim);
    v->GetVectorGradient(T, G);
    double tr = 0.0; for (int i=0;i<dim;i++) tr += G(i,i);
    return tr;
  }
};

struct VelMagCoeff : public mfem::Coefficient
{
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  VelMagCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override
  {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::Vector u; v->GetVectorValue(T, ip, u);
    return u.Norml2();
  }
};

} // namespace mfem
