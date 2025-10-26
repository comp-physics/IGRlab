#pragma once
#include <functional>
#include "mfem.hpp"
#include "BdrHyperbolicFluxIntegrator.hpp"
#include "DGHyperbolicConservationLaws.hpp"
#include "fluxfunctions.hpp"
#include "IGR.hpp"

namespace mfem
{

// Helper: make a boundary marker from a set of attribute ids {a1, a2, ...}
// Example: MakeMarker(pmesh, {4}) creates a marker with only attribute 4 active.
static void MakeMarker(mfem::Array<int> &marker,
                       mfem::ParMesh &pmesh,
                       std::initializer_list<int> attrs)
{
   const int nb = pmesh.bdr_attributes.Max();
   MFEM_VERIFY(nb > 0, "Mesh has no boundary attributes.");
   marker.SetSize(nb); marker = 0;
   for (int a : attrs)
   {
      MFEM_VERIFY(a >= 1 && a <= nb,
            "Requested boundary attribute " << a << " but mesh has [1.." << nb << "]");
      marker[a-1] = 1;
   }
}


/// --------------------------------------------------------------
/// Wrapper operator: recompute Σ each Mult() and forward to DG.
/// --------------------------------------------------------------
class EulerWithIGR : public TimeDependentOperator
{
public:
  EulerWithIGR(DGHyperbolicConservationLaws& base,
               IGRSolver& igr,
               ParGridFunction& state)
    : TimeDependentOperator(base.Height()),
      base_(base), igr_(igr), state_(state) {}

  void SetTime(double t) override { base_.SetTime(t); }

  void Mult(const Vector& U, Vector& dUdt) const override
  {
    // Recompute Σ with a few warm-started sweeps;
    // FluxWithIGR reads Σ directly (as a ParGridFunction*) so
    // no explicit plumbing into DG is needed here.
    igr_.UpdateFromState(state_, /*warm_start=*/true);

    // IMPORTANT: make Σ available on face neighbors across MPI subdomains
    igr_.GetSigma().ExchangeFaceNbrData();

    // Forward to the DG operator
    base_.Mult(U, dUdt);
  }

  real_t GetMaxCharSpeed() { return base_.GetMaxCharSpeed(); }

private:
  DGHyperbolicConservationLaws& base_;
  IGRSolver&                    igr_;
  ParGridFunction&              state_;  // view of current state (vdim=num_eq)
};

class VelocityFromConsCoefficient : public mfem::VectorCoefficient
{
public:
  VelocityFromConsCoefficient(int dim, const mfem::ParGridFunction* U,
                              double rho_floor)
    : mfem::VectorCoefficient(dim), dim_(dim), U_(U), rho_floor_(rho_floor) {}

  void SetState(const mfem::ParGridFunction* U) { U_ = U; }

  virtual void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
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
struct GradFrobNormCoeff : public mfem::Coefficient {
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  GradFrobNormCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::DenseMatrix G(dim, dim);
    v->GetVectorGradient(T, G); // G(i,j) = ∂_j u_i
    double s = 0.0; for (int i=0;i<dim;i++) for (int j=0;j<dim;j++) s += G(i,j)*G(i,j);
    return std::sqrt(s);
  }
};

struct DivCoeff : public mfem::Coefficient {
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  DivCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::DenseMatrix G(dim, dim);
    v->GetVectorGradient(T, G);
    double tr = 0.0; for (int i=0;i<dim;i++) tr += G(i,i);
    return tr;
  }
};

struct VelMagCoeff : public mfem::Coefficient {
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  VelMagCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::Vector u; v->GetVectorValue(T, ip, u);
    return u.Norml2();
  }
};


} // namespace mfem
