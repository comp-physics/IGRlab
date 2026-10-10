#pragma once

#include "igrlab/environment.hpp"

#include "../include/physics/igr/detail/compact_subcell_recovery.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/igr/igr_coupling.hpp"
#include "physics/igr/igr_flux.hpp"
#include <cmath>
#include <cstdlib>
#include <string>

inline void CheckCompactQuadratureFloor(igrlab::IGRSolver& igr,
                                        mfem::ParFiniteElementSpace& sfes,
                                        const mfem::ParGridFunction& recovered)
{
  class CurrentCoefficient : public mfem::Coefficient
  {
  public:
    CurrentCoefficient(const mfem::ParGridFunction& state,int kind) : state_(state),kind_(kind) {}
    double Eval(mfem::ElementTransformation& tr,const mfem::IntegrationPoint& ip) override
    {
      mfem::Vector u(4); tr.SetIntPoint(&ip); state_.GetVectorValue(tr,ip,u);
      if(kind_==0)return 1./u(0);
      if(kind_==1)return .04/u(0);
      mfem::DenseMatrix gradient(4,2); state_.GetVectorGradient(tr,gradient);
      return igrlab::frozen_scalar_igr_algebra::SourceFromConservativeStateGradient(
        2,u.GetData(),1,gradient.GetData(),1,gradient.Height(),.04,1e-12);
    }
  private:
    const mfem::ParGridFunction& state_; int kind_;
  } reaction(recovered,0),diffusion(recovered,1),source(recovered,2);
  const auto& rule=mfem::IntRules.Get(mfem::Geometry::SQUARE,4);
  mfem::BilinearForm reference(&sfes);
  auto* mass=new mfem::MassIntegrator(reaction); mass->SetIntRule(&rule);
  auto* diff=new mfem::DiffusionIntegrator(diffusion); diff->SetIntRule(&rule);
  reference.AddDomainIntegrator(mass); reference.AddDomainIntegrator(diff);
  reference.Assemble(); reference.Finalize();
  mfem::LinearForm load(&sfes);
  auto* rhs=new mfem::DomainLFIntegrator(source); rhs->SetIntRule(&rule);
  load.AddDomainIntegrator(rhs); load.Assemble();
  mfem::Vector x(sfes.GetVSize()),expected(x.Size()),observed(x.Size());
  for(int i=0;i<x.Size();++i)x(i)=std::sin(.31*(i+1));
  MFEM_VERIFY(igr.GetLinearBaseOperator(),"Quadrature-floor test requires the exact closure's volume operator.");
  reference.Mult(x,expected); igr.GetLinearBaseOperator()->Mult(x,observed); observed-=expected;
  const double operator_error=observed.Normlinf()/std::max(expected.Normlinf(),1e-30);
  mfem::Vector rhs_error(igr.GetLinearRHS()); rhs_error-=load;
  const double source_error=rhs_error.Normlinf()/std::max(load.Normlinf(),1e-30);
  MFEM_VERIFY(std::isfinite(operator_error) && operator_error<3e-12 &&
                std::isfinite(source_error) && source_error<3e-10,
              "P1 closure quadrature floor differs from independent current-p2-state MFEM order-four assembly.");
  mfem::out<<"compact_quadrature_floor_test operator_relative_error="<<operator_error
           <<" source_relative_error="<<source_error<<'\n';
}

// Independent face integration oracle: use shared host numerical physics,
// actual face transformations and p0 weak tests, without the packed face maps.
inline mfem::Vector CompactMUSCLFaceOracle(mfem::ParGridFunction& means,
                                          const mfem::Vector& slopes,
                                          const mfem::NumericalFlux& flux)
{
  auto& mesh = *means.ParFESpace()->GetParMesh();
  mfem::Vector result(means.Size()); result = 0.0;
  std::vector<double> volumes(mesh.GetNE());
  for(int e=0;e<mesh.GetNE();++e) volumes[e]=mesh.GetElementVolume(e);
  const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT,8);
  const double* mean_data = means.HostRead(); const double* slope_data = slopes.HostRead();
  mfem::Vector left(4),right(4),normal(2),face_flux(4);
  mfem::Array<int> first,second;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    auto* tr = mesh.GetFaceElementTransformations(face);
    means.ParFESpace()->GetElementVDofs(tr->Elem1No,first);
    if (tr->Elem2No >= 0) means.ParFESpace()->GetElementVDofs(tr->Elem2No,second);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const auto& point = rule.IntPoint(q); tr->SetAllIntPoints(&point);
      mfem::CalcOrtho(tr->Jacobian(),normal);
      for (int c = 0; c < 4; ++c)
      {
        left(c) = mean_data[first[c]] + (tr->GetElement1IntPoint().x-.5)*slope_data[first[c]];
        // Every boundary in this test is a zero-gradient supersonic outlet.
        right(c) = tr->Elem2No >= 0 ? mean_data[second[c]] +
          (tr->GetElement2IntPoint().x-.5)*slope_data[second[c]] : left(c);
      }
      flux.Eval(left,right,normal,*tr,face_flux);
      for (int c = 0; c < 4; ++c)
      {
        result(first[c]) -= point.weight*face_flux(c)/volumes[tr->Elem1No];
        if (tr->Elem2No >= 0)
          result(second[c]) += point.weight*face_flux(c)/volumes[tr->Elem2No];
      }
    }
  }
  return result;
}

inline void TestCompactMUSCLFaces(MPI_Comm comm)
{
  if (!mfem::Device::Allows(mfem::Backend::CUDA_MASK)) return;
  const char* saved_q = igrlab::GetEnv("DG_IGR_DG_FACE_Q_PARALLEL");
  const bool had_q = saved_q != nullptr; const std::string original_q = saved_q ? saved_q : "";
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(8,1,mfem::Element::QUADRILATERAL,true,8.0,1.0);
  mfem::ParMesh mesh(comm,serial);
  mfem::L2_FECollection constants(0,2,mfem::BasisType::GaussLegendre);
  for (const auto setting : {std::array<int,2>{1,0},std::array<int,2>{1,4},std::array<int,2>{2,0}})
  for (auto ordering : {mfem::Ordering::byNODES,mfem::Ordering::byVDIM})
  {
    const int closure_order=setting[0],quadrature_floor=setting[1];
    mfem::L2_FECollection scalar_collection(closure_order,2,mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace fes(&mesh,&constants,4,ordering), sfes(&mesh,&scalar_collection);
    mfem::ParGridFunction means(&fes), sigma(&sfes);
    igrlab::igr_internal::CompactSubcellRecovery recovery(fes,1.4,1e-12,1e-12,false,true);
    MFEM_VERIFY(recovery.StateSpace().GetMaxElementOrder()==2 && sfes.GetMaxElementOrder()==closure_order,
                "Mixed-order closure test changed the current quadratic recovery space.");
    mfem::VectorFunctionCoefficient initial(4,[](const mfem::Vector& x,mfem::Vector& u)
    { u(0)=2+.01*x(0); u(1)=.2+.003*x(0); u(2)=0.; u(3)=5+.02*x(0); });
    recovery.InitializeMeans(initial,means);
    mfem::FunctionCoefficient sigma_coefficient([](const mfem::Vector& x) {return .03+.001*x(0);});
    sigma.ProjectCoefficient(sigma_coefficient);
    mfem::EulerFlux euler(2,1.4); mfem::FluxWithIGR physics(euler);
    physics.SetSigmaField(&sigma); physics.SetInvariantDomainFlux(true);
    physics.SetInvariantDomainFloors(1e-12,1e-12);
    physics.SetSigmaAwareCFL(true); physics.SetSigmaAwareRusanov(true);
    mfem::RusanovFluxWithSides flux(physics);
    auto form = std::make_unique<mfem::HyperbolicFormIntegrator>(flux,2);
    mfem::DGHyperbolicConservationLaws op(fes,std::move(form),true);
    op.SetNumericalFlux(&flux);
    mfem::Array<int> marker(mesh.bdr_attributes.Max()); marker=1;
    op.AddBdrFaceIntegrator(new mfem::BdrHyperbolicFluxIntegrator(
      flux,std::make_unique<mfem::SupersonicOutletState>(),2),marker);
    op.SetFaceIntegrationRule(mfem::IntRules.Get(mfem::Geometry::SEGMENT,8));
    const auto& slopes = recovery.RebuildTransportSlopes(means);
    const double expected_slope[4] = {.01,.003,0.,.02};
    mfem::Array<int> ids; mfem::Vector position(2);
    for (int e = 0; e < mesh.GetNE(); ++e)
    {
      mfem::IntegrationPoint center; center.Set2(.5,.5);
      mesh.GetElementTransformation(e)->Transform(center,position);
      fes.GetElementVDofs(e,ids);
      for (int c = 0; c < 4; ++c)
      {
        const double expected = position(0)>1 && position(0)<7 ? expected_slope[c] : 0.;
        MFEM_VERIFY(std::abs(slopes.HostRead()[ids[c]]-expected)<2e-13,
                    "MUSCL failed exact affine reconstruction or nearest-mean boundary slope.");
      }
    }
    op.SetP0FaceReferenceXSlopes(&slopes);
    mfem::Vector result(means.Size()), first_result;
    for (const char* mode : {"0","1"})
    {
      setenv("DG_IGR_DG_FACE_Q_PARALLEL",mode,1);
      op.ArmPhysicalBoundaryFluxIntegral(); op.Mult(means,result);
      mfem::Vector oracle = CompactMUSCLFaceOracle(means,slopes,flux);
      mfem::Vector difference(result); difference-=oracle;
      MFEM_VERIFY(std::isfinite(difference.Normlinf()) && difference.Normlinf()<3e-12,
                  "Packed MUSCL face residual differs from independent shared-physics quadrature.");
      mfem::Vector budget(op.GetLastPhysicalBoundaryFluxIntegral());
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
        fes.GetElementVDofs(e,ids);
        for (int c = 0; c < 4; ++c) budget(c)+=mesh.GetElementVolume(e)*result.HostRead()[ids[c]];
      }
      MFEM_VERIFY(budget.Normlinf()<3e-12,"MUSCL cell updates violate the captured physical-boundary flux budget.");
      first_result=result;
    }
    op.SetP0FaceReferenceXSlopes(nullptr); op.Mult(means,result); result-=first_result;
    MFEM_VERIFY(result.Normlinf()>1e-6,"MUSCL trace option did not change first-order transport.");

    // Exercise the real coupling wrapper, including an output closure reused
    // at stage zero and a later distinct device-resident stage vector.
    if (ordering == mfem::Ordering::byNODES)
    {
      igrlab::IGRSolver igr(sfes,recovery.StateSpace(),2,4,.04);
      igr.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);
      igr.SetIPKind(igrlab::IGRIPKind::SIPG); igr.SetExactDGFaceOperator(true);
      igr.SetSolveMode(igrlab::IGRSolveMode::Krylov);
      igr.SetLinearSolver(igrlab::IGRLinearSolverType::CG);
      igr.SetPreconditioner(igrlab::IGRPrecondType::Auto);
      igr.SetLinearSolveParams(1e-10,0.,1000,0);
      igr.SetStageRtolScale(1.0);
      igr.SetSubcellQuadratureMinimumOrder(quadrature_floor);
      physics.SetSigmaField(&igr.GetSigma());
      mfem::EulerWithIGR wrapped(op,igr,means,&recovery);
      wrapped.PrepareSigmaForOutput(means,1,0,0.,true);
      wrapped.BeginRKStep(0,0.,.001,3); wrapped.SetRKStageState(0,1,0.);
      wrapped.Mult(means,result);
      MFEM_VERIFY(wrapped.GetSigmaStage0ReuseCount()==1,"MUSCL test did not exercise current stage-zero closure reuse.");
      mfem::Vector stage0_expected = CompactMUSCLFaceOracle(means,slopes,flux);
      stage0_expected-=result;
      MFEM_VERIFY(stage0_expected.Normlinf()<3e-12,"Reused stage-zero closure omitted current MUSCL traces.");
      mfem::ParGridFunction stage(&fes); stage=means;
      double* current=stage.ReadWrite();
      mfem::forall(stage.Size(),[=] MFEM_HOST_DEVICE(int i) {current[i]*=1.07;});
      wrapped.SetRKStageState(1,2,.001); wrapped.Mult(stage,result);
      const auto& current_slopes = recovery.RebuildTransportSlopes(stage);
      mfem::Vector expected = CompactMUSCLFaceOracle(stage,current_slopes,flux); expected-=result;
      MFEM_VERIFY(expected.Normlinf()<3e-12,"Coupled stage used stale reconstructed transport traces.");
      mfem::Vector stage_sigma(igr.GetSigma());
      igr.UpdateFromState(recovery.Rebuild(stage),false,igrlab::IGRSolvePurpose::ProductionStage,true);
      MFEM_VERIFY(igr.GetLastLinearConverged() &&
                    std::isfinite(igr.GetLastLinearTrueResidualRelRHS()) &&
                    igr.GetLastLinearTrueResidualRelRHS()<=igr.GetLastIndependentCertificateRelativeLimit(),
                  "Mixed-order current recovered-state solve failed the global A*sigma-b residual certificate.");
      stage_sigma-=igr.GetSigma();
      MFEM_VERIFY(stage_sigma.Normlinf()<1e-10,"Stage closure differs from a fresh current recovered-state solve.");
      if(quadrature_floor)CheckCompactQuadratureFloor(igr,sfes,recovery.Rebuild(stage));
      wrapped.EndRKStep(stage,.001,2);
      physics.SetSigmaField(&sigma);
    }

    // Positive means whose unscaled componentwise-MC line has negative
    // internal energy. One common theta must admit both endpoints and retain
    // the mean, instead of clipping density/energy independently.
    double* data=means.HostWrite();
    for (int e=0;e<mesh.GetNE();++e)
    {
      mfem::IntegrationPoint center; center.Set2(.5,.5);
      mesh.GetElementTransformation(e)->Transform(center,position);
      fes.GetElementVDofs(e,ids); const double momentum=.2*(position(0)+.5);
      data[ids[0]]=1.; data[ids[1]]=momentum; data[ids[2]]=0.; data[ids[3]]=.5*momentum*momentum+.001;
    }
    const auto& limited=recovery.RebuildTransportSlopes(means);
    double minimum_ratio=1.;
    for(int e=1;e<mesh.GetNE()-1;++e)
    {
      fes.GetElementVDofs(e,ids); mfem::Vector left(4),right(4);
      for(int c=0;c<4;++c)
      {left(c)=data[ids[c]]-.5*limited.HostRead()[ids[c]];right(c)=data[ids[c]]+.5*limited.HostRead()[ids[c]];
       MFEM_VERIFY(std::abs(.5*(left(c)+right(c))-data[ids[c]])<2e-15,"MUSCL positivity scaling changed a mean.");}
      minimum_ratio=std::min(minimum_ratio,limited.HostRead()[ids[1]]/.2);
      MFEM_VERIFY(mfem::EulerPressureFromConservative(left,2,1.4)>1e-12 &&
                    mfem::EulerPressureFromConservative(right,2,1.4)>1e-12,
                  "MUSCL common slope scaling failed endpoint positivity.");
    }
    MFEM_VERIFY(minimum_ratio<.5,"MUSCL positivity test did not activate a nontrivial common theta.");
    op.SetP0FaceReferenceXSlopes(nullptr);
    mfem::out<<"compact_muscl_face_test closure_order="<<closure_order<<" quadrature_min_order="<<quadrature_floor
             <<" recovery_order=2 ordering="<<static_cast<int>(ordering)
             <<" shared_flux_oracle=pass boundary_budget=pass minimum_theta="<<minimum_ratio<<'\n';
  }
  if(had_q)setenv("DG_IGR_DG_FACE_Q_PARALLEL",original_q.c_str(),1);
  else unsetenv("DG_IGR_DG_FACE_Q_PARALLEL");
}

inline void TestCompactSubcellRecovery(MPI_Comm comm)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(8,1,mfem::Element::QUADRILATERAL,true,8.0,1.0);
  mfem::ParMesh mesh(comm,serial);
  mfem::L2_FECollection constants(0,2,mfem::BasisType::GaussLegendre);
  for (bool linear : {false,true})
  for (auto ordering : {mfem::Ordering::byNODES,mfem::Ordering::byVDIM})
  {
    mfem::ParFiniteElementSpace fes(&mesh,&constants,4,ordering);
    mfem::ParGridFunction means(&fes);
    igrlab::igr_internal::CompactSubcellRecovery recovery(fes,1.4,1e-12,1e-12,linear);
    mfem::VectorFunctionCoefficient initial(4,[](const mfem::Vector& x,mfem::Vector& u)
    {
      u(0) = 2+.01*x(0)+.002*x(0)*x(0);
      u(1) = .2; u(2) = 0.0; u(3) = 5.0;
    });
    recovery.InitializeMeans(initial,means);
    // The production PA closure first requests coordinate-based volume/face
    // factors after constructing recovery. Coordinate-node installation must
    // already have happened, without weakening the later fixed-mesh guard.
    MFEM_VERIFY(mesh.GetNodes(),"Recovery did not initialize PA coordinate nodes before recording geometry.");
    const long node_sequence = mesh.GetNodesSequence();
    const auto& volume_rule = mfem::IntRules.Get(mfem::Geometry::SQUARE,5);
    const auto& face_rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT,4);
    (void)mesh.GetGeometricFactors(volume_rule,mfem::GeometricFactors::JACOBIANS);
    (void)mesh.GetFaceGeometricFactors(face_rule,mfem::FaceGeometricFactors::JACOBIANS,mfem::FaceType::Interior);
    MFEM_VERIFY(mesh.GetNodesSequence() == node_sequence,
                "PA geometry materialization unexpectedly changed the fixed coordinate representation.");
    const auto& field = recovery.Rebuild(means);
    const auto& rule = mfem::IntRules.Get(mfem::Geometry::SQUARE,8);
    mfem::Vector value(4), mean(4), observed(4), x(2);
    mfem::DenseMatrix gradient(4,2);
    double integral_error = 0.0, polynomial_error = 0.0, gradient_error = 0.0;
    for (int e = 0; e < mesh.GetNE(); ++e)
    {
      auto* tr = mesh.GetElementTransformation(e);
      mfem::IntegrationPoint center; center.Set2(.5,.5);
      tr->SetIntPoint(&center); tr->Transform(center,x);
      means.GetVectorValue(e,center,observed);
      const double exact_mean = 2+.01*x(0)+.002*(x(0)*x(0)+1.0/12);
      const double center_x = x(0);
      integral_error = std::max(integral_error,std::abs(observed(0)-exact_mean));
      const bool interior = x(0) > 2 && x(0) < 6;
      mean = 0.0;
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const auto& point = rule.IntPoint(q); tr->SetIntPoint(&point); tr->Transform(point,x);
        field.GetVectorValue(e,point,value);
        for (int c = 0; c < 4; ++c) mean(c) += point.weight*value(c);
        if (interior)
        {
          const double expected = linear ? exact_mean+(x(0)-center_x)*(.01+.004*center_x)
                                         : 2+.01*x(0)+.002*x(0)*x(0);
          polynomial_error = std::max(polynomial_error,std::abs(value(0)-expected));
          field.GetVectorGradient(*tr,gradient);
          gradient_error = std::max(gradient_error,std::abs(gradient(0,0)-(.01+.004*(linear?center_x:x(0)))));
          gradient_error = std::max(gradient_error,std::abs(gradient(0,1)));
        }
      }
      for (int c = 0; c < 4; ++c) integral_error = std::max(integral_error,std::abs(mean(c)-observed(c)));
    }
    MFEM_VERIFY(integral_error < 2e-13 && polynomial_error < 2e-13 && gradient_error < 2e-12,
                "Compact conservative recovery failed independent polynomial/mean/gradient identities.");
    // Mutate the transported means in device memory. The next reconstruction
    // must use their current version; the interpolation field is not a state.
    double* changed = means.ReadWrite();
    mfem::forall(means.Size(),[=] MFEM_HOST_DEVICE(int i) { changed[i] *= 1.01; });
    const auto& current = recovery.Rebuild(means);
    for (int e = 0; e < mesh.GetNE(); ++e)
    {
      mean = 0.0;
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        current.GetVectorValue(e,rule.IntPoint(q),value);
        for (int c = 0; c < 4; ++c) mean(c) += rule.IntPoint(q).weight*value(c);
      }
      mfem::IntegrationPoint center; center.Set2(.5,.5); means.GetVectorValue(e,center,observed);
      for (int c = 0; c < 4; ++c)
        MFEM_VERIFY(std::abs(mean(c)-observed(c)) < 2e-13,"Recovery missed a changed transported mean.");
    }
    mfem::out << "compact_subcell_recovery_test linear=" << linear << " ordering=" << static_cast<int>(ordering)
              << " integral_error=" << integral_error << " polynomial_error=" << polynomial_error
              << " gradient_error=" << gradient_error << '\n';
  }
  TestCompactMUSCLFaces(comm);
}
