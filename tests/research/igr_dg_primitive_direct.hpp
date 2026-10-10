#pragma once
// Direct degree-p tests of the same primitive trial manifold and quadrature.
// This removes the redundant degree-3p assembly and Galerkin projection, while
// retaining the degree-3p physical serialization for reuse and independent audit.
inline void PrimitiveDirectSystem(const PrimitiveMomentProblem& pr,const std::vector<State>& low_derivatives,
                                  const State& physical,TrackWork& work,BandMatrix& a,State& b)
{
  const int m=pr.retained.n,n=pr.n,ne=pr.ne;const double alpha=pr.o.base.alpha;
  if(pr.o.closure_space!="l2"||pr.o.closure_boundary!="natural")throw std::runtime_error("direct primitive screen requires natural L2 closure");
  a=BandMatrix(ne*m,2*m-1);b.assign(ne*m,0.);double source_integral=0.;
  for(int e=0;e<ne;++e)
  {
    const double h=pr.edges[e+1]-pr.edges[e];
    for(std::size_t q=0;q<pr.weights.size();++q)
    {
      const auto u=pr.Point(physical,e,pr.shapes[q]);pr.physics.CheckState(u.data());double dr=0.,dm=0.;
      for(int i=0;i<n;++i){dr+=pr.derivatives[q][i]*physical[pr.Id(e,i,0)]/h;dm+=pr.derivatives[q][i]*physical[pr.Id(e,i,1)]/h;}
      double g=(dm-u[1]/u[0]*dr)/u[0];
      if(pr.central_source)for(int side=0;side<2;++side)
      {
        const int neighbor=e+(side?1:-1);if(neighbor<0||neighbor>=ne)continue;
        const int endpoint=side?pr.p:0,other=side?0:pr.p,low_endpoint=side?m-1:0;
        const double own=physical[pr.Id(e,endpoint,1)]/physical[pr.Id(e,endpoint,0)],outside=physical[pr.Id(neighbor,other,1)]/physical[pr.Id(neighbor,other,0)];
        double lift=0.;for(int i=0;i<m;++i)lift+=pr.low_shapes[q][i]*pr.retained.mass_inverse(i,low_endpoint)/h;
        g+=(side?-1.:1.)*.5*(own-outside)*lift;
      }
      const double source=2.*alpha*g*g,weight=pr.weights[q],ri=1./u[0];source_integral+=h*weight*source;
      for(int i=0;i<m;++i)
      {
        b[e*m+i]+=h*weight*pr.low_shapes[q][i]*source;
        for(int j=0;j<m;++j)a.At(e*m+i,e*m+j)+=weight*ri*(h*pr.low_shapes[q][i]*pr.low_shapes[q][j]+alpha*low_derivatives[q][i]*low_derivatives[q][j]/h);
      }
      ++work.source_points;
    }
  }
  if(alpha!=0.)
  {
    const State dl=pr.retained.Shape(1.,true),dr=pr.retained.Shape(0.,true);
    for(int face=1;face<ne;++face)
    {
      ++work.closure_faces;const double hl=pr.edges[face]-pr.edges[face-1],hr=pr.edges[face+1]-pr.edges[face];
      const double al=alpha/physical[pr.Id(face-1,pr.p,0)],ar=alpha/physical[pr.Id(face,0,0)];
      const double penalty=pr.o.sipg_penalty*n*n*.5*(al/hl+ar/hr);
      for(int i=0;i<2*m;++i)for(int j=0;j<2*m;++j)
      {
        const bool il=i<m,jl=j<m;const int ii=i%m,jj=j%m;
        const double ji=il?(ii==m-1?1.:0.):(ii==0?-1.:0.),jjump=jl?(jj==m-1?1.:0.):(jj==0?-1.:0.);
        if(ji==0.&&jjump==0.)continue;
        const double fi=.5*(il?al*dl[ii]/hl:ar*dr[ii]/hr),fj=.5*(jl?al*dl[jj]/hl:ar*dr[jj]/hr);
        a.At((il?face-1:face)*m+ii,(jl?face-1:face)*m+jj)+=-ji*fj-fi*jjump+penalty*ji*jjump;
      }
    }
  }
  work.current_source_integral=source_integral;++work.closure_assemblies;
}
inline State PrimitiveDirectResidual(const PrimitiveMomentProblem& pr,const std::vector<State>& low_derivatives,
                                     const State& physical,const State& sigma,TrackWork& work,std::array<double,4>& budget)
{
  const auto start=Clock::now();const int m=pr.retained.n,n=pr.n,ne=pr.ne;State load(ne*m*4,0.),low(ne*m*4,0.);
  for(int e=0;e<ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
  {
    const auto u=pr.Point(physical,e,pr.shapes[q]);double s=0.,speed=0.;for(int i=0;i<n;++i)s+=pr.shapes[q][i]*sigma[pr.SigmaIndex(e,i)];
    const auto f=pr.physics.Flux(u.data(),s,speed);++work.flux_points;
    for(int i=0;i<m;++i)for(int c=0;c<4;++c)load[(e*m+i)*4+c]+=pr.weights[q]*low_derivatives[q][i]*f[c];
  }
  budget.fill(0.);
  for(int face=0;face<=ne;++face)
  {
    std::array<double,4> flux{};
    if(face==0||face==ne)flux=pr.BoundaryFlux(physical,sigma,face==ne);
    else
    {
      double al=0.,ar=0.;const auto fl=pr.physics.Flux(&physical[pr.Id(face-1,pr.p,0)],sigma[pr.SigmaIndex(face-1,pr.p)],al);
      const auto fr=pr.physics.Flux(&physical[pr.Id(face,0,0)],sigma[pr.SigmaIndex(face,0)],ar);const double speed=pr.physics.FaceSpeed(al,ar);
      for(int c=0;c<4;++c)flux[c]=.5*(fl[c]+fr[c]-speed*(physical[pr.Id(face,0,c)]-physical[pr.Id(face-1,pr.p,c)]));
    }
    ++work.faces;
    for(int c=0;c<4;++c)
    {
      if(face>0)load[((face-1)*m+m-1)*4+c]-=flux[c];if(face<ne)load[(face*m)*4+c]+=flux[c];
      if(face==0)budget[c]=flux[c];if(face==ne)budget[c]-=flux[c];
    }
  }
  for(int e=0;e<ne;++e)for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<4;++c)
    low[(e*m+i)*4+c]+=pr.retained.mass_inverse(i,j)*load[(e*m+j)*4+c]/(pr.edges[e+1]-pr.edges[e]);
  State result=pr.state_transfer.Apply(pr.state_transfer.E,low,ne,4);const auto totals=pr.Integrals(result,pr.edges);
  for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));work.transport+=Wall(start);return result;
}
