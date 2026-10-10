#ifndef DG_IGR_HP_L2_CONDENSED_HPP
#define DG_IGR_HP_L2_CONDENSED_HPP
// Test-only exact static condensation of the current one-dimensional L2/SIPG
// closure. Include after igr_hp_front_tracking_probe.cpp in library mode.
// Euler/source construction is unchanged. This is an algebraic transformation
// of the already assembled current operator, not a local-closure approximation.
namespace
{
struct MovingL2CondensedWork
{
  double seconds=0.,local_factor_products=0.,response_products=0.;
  double trace_factor_products=0.,schur_products=0.,recovery_products=0.;
  double max_residual=0.,last_residual=0.;
  std::uint64_t calls=0,local_factors=0,trace_factors=0;
  std::uint64_t response_columns=0,source_columns=0;
  std::size_t peak_factor_entries=0,peak_response_entries=0;
};

State SolveMovingL2Condensed(const BandMatrix& a,const State& b,
                            int ne,int p,MovingL2CondensedWork& w)
{
  const auto start=Clock::now();const int n=p+1,m=n-2,nt=2*ne;
  if(ne<1||p<1||a.n!=ne*n||int(b.size())!=a.n)
    throw std::runtime_error("invalid L2 condensation dimensions");
  auto trace_dof=[&](int t){return (t/2)*n+(t%2?p:0);};
  // A volume bubble has zero traces. SIPG derivative consistency couples its
  // element interior only to the own two endpoints and nearest neighbor
  // endpoints: up to four real trace columns. Eliminating these bubbles can
  // connect four consecutive trace indices, hence half-bandwidth three.
  BandMatrix schur(nt,std::min(3,nt-1));State load(nt,0.);
  for(int i=0;i<nt;++i)
  {
    load[i]=b[trace_dof(i)];
    for(int j=std::max(0,i-schur.k);j<std::min(nt,i+schur.k+1);++j)
      schur.At(i,j)=a.Get(trace_dof(i),trace_dof(j));
  }
  struct LocalResponse{std::vector<int> traces;State response;};
  std::vector<LocalResponse> local(ne);std::size_t stored_responses=0,stored_factors=0;
  for(int e=0;e<ne&&m>0;++e)
  {
    auto& cell=local[e];const int first=e*n+1;
    for(int t=std::max(0,2*e-1);t<std::min(nt,2*e+3);++t)
    {
      bool active=false;for(int i=0;i<m;++i)active=active||a.Get(first+i,trace_dof(t))!=0.;
      if(active)cell.traces.push_back(t);
    }
    const int columns=1+int(cell.traces.size());cell.response.assign(m*columns,0.);
    for(int i=0;i<m;++i)
    {
      cell.response[i]=b[first+i];
      for(int c=1;c<columns;++c)cell.response[c*m+i]=a.Get(first+i,trace_dof(cell.traces[c-1]));
    }
    // Factor only the actual (p-1)-square interior block. Its same factor
    // supplies one current-source response and up to four face responses.
    BandFactor factor(a,first,m);factor.Solve(cell.response,columns);
    ++w.local_factors;++w.source_columns;w.response_columns+=columns-1;
    w.local_factor_products+=factor.products;
    w.response_products+=double(columns)*m*(m-1); // Both triangular solves.
    stored_factors=std::max(stored_factors,factor.l.size());
    stored_responses+=cell.response.size();
    for(int i=0;i<int(cell.traces.size());++i)
    {
      const int ti=cell.traces[i],row=trace_dof(ti);double correction=0.;
      for(int k=0;k<m;++k){correction+=a.Get(row,first+k)*cell.response[k];++w.schur_products;}
      load[ti]-=correction;
      // Compute one triangle and assign both symmetric updates. No duplicate
      // arithmetic is counted as useful work. The direct residual below is
      // evaluated against the original, uncondensed matrix.
      for(int j=0;j<=i;++j)
      {
        const int tj=cell.traces[j];double value=0.;
        for(int k=0;k<m;++k){value+=a.Get(row,first+k)*cell.response[(j+1)*m+k];++w.schur_products;}
        schur.At(ti,tj)-=value;if(ti!=tj)schur.At(tj,ti)-=value;
      }
    }
  }
  BandFactor trace_factor(schur,0,nt);trace_factor.Solve(load,1);
  ++w.trace_factors;w.trace_factor_products+=trace_factor.products;
  // Include the trace triangular solves using the actual bounded bandwidth.
  for(int i=0;i<nt;++i)w.trace_factor_products+=2.*std::min(i,trace_factor.k);
  w.peak_factor_entries=std::max(w.peak_factor_entries,
                                std::max(stored_factors,trace_factor.l.size()));
  w.peak_response_entries=std::max(w.peak_response_entries,stored_responses);
  State sigma(a.n,0.);for(int t=0;t<nt;++t)sigma[trace_dof(t)]=load[t];
  for(int e=0;e<ne&&m>0;++e)
  {
    const auto& cell=local[e];for(int i=0;i<m;++i)
    {
      double value=cell.response[i];
      for(int c=0;c<int(cell.traces.size());++c)
      {value-=cell.response[(c+1)*m+i]*load[cell.traces[c]];++w.recovery_products;}
      sigma[e*n+1+i]=value;
    }
  }
  State defect(a.n,0.);for(int i=0;i<a.n;++i)
  {defect[i]=-b[i];for(int j=std::max(0,i-a.k);j<std::min(a.n,i+a.k+1);++j)defect[i]+=a.Get(i,j)*sigma[j];}
  const double residual=Norm(defect)/std::max(Norm(b),1.e-12);
  w.last_residual=residual;w.max_residual=std::max(w.max_residual,residual);
  if(!std::isfinite(residual)||residual>1.e-8)
    throw std::runtime_error("condensed current L2 closure residual failed");
  ++w.calls;w.seconds+=Wall(start);return sigma;
}

void MovingL2CondensedSelfTest()
{
  for(int p:{1,3,7,11,15})for(double grading:{1.,3.})
    for(const std::string boundary:{"natural","dirichlet"})
  {
    TrackOptions o;o.base.elements=8;o.base.order=p;o.base.alpha=.0004;
    o.initial="entropy";o.closure_space="l2";o.closure_boundary=boundary;
    o.mode="frozen";o.grading=grading;o.sipg_penalty=4.;
    TrackingProblem pr(o);State u(pr.ne*pr.n*4,0.);
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {
      const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];
      // Positive current density varies within cells and jumps across faces.
      const double rho=1.+.1*(e%3)+.07*x+.03*x*x;
      const double velocity=.2-.3*x+.02*std::sin(double(e));
      u[pr.Id(e,i,0)]=rho;u[pr.Id(e,i,1)]=rho*velocity;
      u[pr.Id(e,i,3)]=3.+.5*rho*velocity*velocity;
    }
    TrackWork work;BandMatrix a;State b;pr.AssembleClosure(u,0.,work,a,b);
    BandFactor direct_factor(a,0,a.n);State direct=b;direct_factor.Solve(direct,1);
    MovingL2CondensedWork condensed_work;
    State result=SolveMovingL2Condensed(a,b,pr.ne,pr.p,condensed_work);
    for(std::size_t i=0;i<result.size();++i)result[i]-=direct[i];
    if(Norm(result)/std::max(Norm(direct),1.e-12)>3.e-9)
      throw std::runtime_error("condensed/direct current L2 parity failed");
    if(condensed_work.max_residual>1.e-9||condensed_work.trace_factors!=1)
      throw std::runtime_error("condensed current L2 certificate/counter failed");
    // Alpha=0 exercises the pure mass operator, zero source, and uncoupled
    // element blocks while retaining the same partition and scalar layout.
    pr.o.base.alpha=0.;pr.AssembleClosure(u,0.,work,a,b);
    if(Norm(SolveMovingL2Condensed(a,b,pr.ne,pr.p,condensed_work))>1.e-13)
      throw std::runtime_error("condensed alpha-zero parity failed");
  }
}
}
#endif
