#pragma once
// Test-only interior LLF contribution from the SAME current physical traces,
// stress and wave-speed policy as PrimitiveDirectResidual. Exterior boundary
// fluxes remain in the complementary transport operator.
inline State FrontInteriorLLFResidual(const PrimitiveMomentProblem& pr,
                                     const State& physical,const State& sigma,TrackWork& work)
{
  const auto start=Clock::now();const int m=pr.retained.n,ne=pr.ne;
  State load(ne*m*4,0.),low(load.size(),0.);
  for(int face=1;face<ne;++face)
  {
    double al=0.,ar=0.;
    pr.physics.Flux(&physical[pr.Id(face-1,pr.p,0)],sigma[pr.SigmaIndex(face-1,pr.p)],al);
    pr.physics.Flux(&physical[pr.Id(face,0,0)],sigma[pr.SigmaIndex(face,0)],ar);
    const double speed=pr.physics.FaceSpeed(al,ar);work.flux_points+=2;++work.faces;
    for(int c=0;c<4;++c)
    {
      const double dissipative_flux=-.5*speed*(physical[pr.Id(face,0,c)]-physical[pr.Id(face-1,pr.p,c)]);
      load[((face-1)*m+m-1)*4+c]-=dissipative_flux;
      load[(face*m)*4+c]+=dissipative_flux;
    }
  }
  for(int e=0;e<ne;++e)for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<4;++c)
    low[(e*m+i)*4+c]+=pr.retained.mass_inverse(i,j)*load[(e*m+j)*4+c]/(pr.edges[e+1]-pr.edges[e]);
  State high=pr.state_transfer.Apply(pr.state_transfer.E,low,ne,4);
  for(double sum:pr.Integrals(high,pr.edges))work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(sum));
  work.transport+=Wall(start);return high;
}
