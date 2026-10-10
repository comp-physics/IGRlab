#pragma once
// Mixed-degree conservative trial manifold: rho and retained E in Pp,
// velocity and conservative momentum tests in Pq. Signed extra momentum
// moments affect every Euler flux and the instantaneous true-Qq IGR source.
struct SignedEnrichmentProblem:EntropyCoupledProblem
{
  int base_order;
  bool enrich_density,enrich_energy;
  RecoveryProblem coarse;
  LocalTransfer coarse_transfer;
  std::unique_ptr<CurvatureProblem> initial_base;
  std::vector<State> initial_embedding;
  mutable State external_cache,internal_cache,physical_cache;
  mutable double enrichment_seconds=0.,restriction_seconds=0.,maximum_restriction_mean=0.;
  mutable std::uint64_t enrichment_cells=0,restriction_calls=0;
  mutable State last_unrestricted_rhs;
  static MemoryOptions InitialOptions(const MemoryOptions& rich,int degree)
  {
    MemoryOptions low=rich;auto& t=low.recovery.boundary.compatibility.track;
    t.quad_extra+=2*(t.base.order-3*degree);t.base.order=3*degree;
    low.retained_order=low.stress_order=degree;low.input.clear();low.start_time=0.;return low;
  }
  SignedEnrichmentProblem(const MemoryOptions& rich,int degree,double slope,double curvature,std::string volume="original",bool density=false,bool energy=false):
    EntropyCoupledProblem(rich,slope,curvature,std::move(volume)),base_order(degree),enrich_density(density),enrich_energy(energy),
    coarse(SpaceOptions(rich,degree)),coarse_transfer(retained,coarse)
  {
    if(degree<2||degree>retained.p)throw std::runtime_error("invalid signed enrichment base/rich orders");
    if(degree<retained.p)
    {
      initial_base=std::make_unique<CurvatureProblem>(InitialOptions(rich,degree),slope,curvature);
      for(double z:physics.basis.nodes)initial_embedding.push_back(initial_base->Shape(z,false));
    }
  }
  void ClearEnrichmentCache()const
  {external_cache.clear();internal_cache.clear();physical_cache.clear();cached_moments.clear();cached_physical.clear();}
  // Fixed linear projection on the EXTERNAL conservative unknowns. Momentum
  // remains Qq; only density and total-energy moments are restricted to Pp.
  State RestrictExternal(const State& high)const
  {
    if(base_order==retained.p||(enrich_density&&enrich_energy))return high;
    const auto start=Clock::now();State low=state_transfer.Apply(state_transfer.P,high,ne,4);
    const State small=coarse_transfer.Apply(coarse_transfer.P,low,ne,4),back=coarse_transfer.Apply(coarse_transfer.E,small,ne,4);
    for(int e=0;e<ne;++e)for(int c:{0,3})
    {
      if((c==0&&enrich_density)||(c==3&&enrich_energy))continue;
      double change=0.;for(int i=0;i<retained.n;++i){const int j=(e*retained.n+i)*4+c;change+=retained.physics.basis.weights[i]*(back[j]-low[j]);low[j]=back[j];}
      maximum_restriction_mean=std::max(maximum_restriction_mean,std::abs(change));
    }
    State out=state_transfer.Apply(state_transfer.E,low,ne,4);++restriction_calls;restriction_seconds+=Wall(start);return out;
  }
  // Exact linear split used by the local predictor: high momentum moment
  // coefficients are signed, but are not directly velocity-tail amplitudes.
  State CoarseExternal(const State& high)const
  {
    const State low=state_transfer.Apply(state_transfer.P,high,ne,4),small=coarse_transfer.Apply(coarse_transfer.P,low,ne,4);
    return state_transfer.Apply(state_transfer.E,coarse_transfer.Apply(coarse_transfer.E,small,ne,4),ne,4);
  }
  struct CellData{State physical,kinetic_tail;};
  CellData ExternalCell(const State& low,int e)const
  {
    CellData out;out.physical=PrimitiveMomentProblem::Cell(low,e);out.kinetic_tail.assign(retained.n,0.);
    if(base_order==retained.p||enrich_energy)return out;
    const auto start=Clock::now();const int m=retained.n;State load(m,0.),kinetic(m,0.);
    for(std::size_t q=0;q<weights.size();++q)
    {
      std::array<double,4> u{};for(int i=0;i<n;++i)for(int c=0;c<4;++c)u[c]+=shapes[q][i]*out.physical[4*i+c];
      const double K=.5*(u[1]*u[1]+u[2]*u[2])/u[0];for(int i=0;i<m;++i)load[i]+=weights[q]*low_shapes[q][i]*K;
    }
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)kinetic[i]+=retained.mass_inverse(i,j)*load[j];
    const State small=coarse_transfer.Apply(coarse_transfer.P,kinetic,1,1),represented=coarse_transfer.Apply(coarse_transfer.E,small,1,1);
    for(int i=0;i<m;++i)out.kinetic_tail[i]=kinetic[i]-represented[i];
    for(int i=0;i<n;++i)for(int j=0;j<m;++j)out.physical[4*i+3]+=state_transfer.E(i,j)*out.kinetic_tail[j];
    ++enrichment_cells;enrichment_seconds+=Wall(start);return out;
  }
  const State& InternalMoments(const State& external)const
  {
    if(base_order==retained.p||enrich_energy)return external;
    if(external!=external_cache)
    {
      const State low=state_transfer.Apply(state_transfer.P,external,ne,4);State internal=external,physical(external.size());
      for(int e=0;e<ne;++e)
      {
        const CellData cell=ExternalCell(low,e);std::copy(cell.physical.begin(),cell.physical.end(),physical.begin()+e*n*4);
        for(int i=0;i<n;++i)for(int j=0;j<retained.n;++j)internal[Id(e,i,3)]+=state_transfer.E(i,j)*cell.kinetic_tail[j];
      }
      external_cache=external;internal_cache=std::move(internal);physical_cache=std::move(physical);
    }
    // The inherited nonvirtual Reconstruct receives this INTERNAL moment key.
    // Cache installation avoids repeating its density-weighted local solve.
    if(cached_moments!=internal_cache){cached_moments=internal_cache;cached_physical=physical_cache;}
    return internal_cache;
  }
  const State& Physical(const State& external)const
  {return PrimitiveMomentProblem::Reconstruct(InternalMoments(external));}
  State Initial()const
  {
    if(base_order==retained.p)return MemoryProblem::Initial();
    if(!memory.input.empty())return RestrictExternal(MemoryProblem::Initial());
    // Start from the common base-p physical initial manifold. Extra velocity
    // modes are initially zero; momentum moments include rho_p v_p products.
    const State base_moments=initial_base->Initial(),base_physical=initial_base->Reconstruct(base_moments);State high(ne*n*4,0.);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)for(int j=0;j<initial_base->n;++j)for(int c=0;c<4;++c)
      high[Id(e,i,c)]+=initial_embedding[i][j]*base_physical[initial_base->Id(e,j,c)];
    high=state_transfer.Apply(state_transfer.E,state_transfer.Apply(state_transfer.P,high,ne,4),ne,4);return RestrictExternal(high);
  }
  void Limit(State& external,TrackWork& work)const
  {
    if(base_order==retained.p||(enrich_density&&enrich_energy)){PrimitiveMomentProblem::Limit(external,work);return;}
    const auto start=Clock::now();external=RestrictExternal(external);State low=state_transfer.Apply(state_transfer.P,external,ne,4);const int m=retained.n;
    for(int e=0;e<ne;++e)
    {
      State original(4*m);std::array<double,4> mean{};
      for(int i=0;i<m;++i)for(int c=0;c<4;++c){original[4*i+c]=low[(e*m+i)*4+c];mean[c]+=retained.physics.basis.weights[i]*original[4*i+c];}
      if(!TrackAdmitted(mean))throw std::runtime_error("signed enrichment conservative mean admission failed");
      bool accepted=false;double theta=1.;
      for(int attempt=0;attempt<49;++attempt,theta*=.5)
      {
        for(int i=0;i<m;++i)for(int c=0;c<4;++c)low[(e*m+i)*4+c]=mean[c]+theta*(original[4*i+c]-mean[c]);
        State physical;bool admitted=true;try{physical=ExternalCell(low,e).physical;}catch(const std::exception&){admitted=false;}
        if(admitted)for(const auto& shape:positive)
        {
          std::array<double,4> point{};for(int i=0;i<n;++i)for(int c=0;c<4;++c)point[c]+=shape[i]*physical[4*i+c];
          ++work.positivity_points;if(!TrackAdmitted(point)){admitted=false;break;}
        }
        if(admitted){accepted=true;if(attempt){++work.limited;work.min_theta=std::min(work.min_theta,theta);}break;}
        if(!o.positivity)throw std::runtime_error("signed enrichment physical admission failed");
      }
      if(!accepted)throw std::runtime_error("signed enrichment limiting failed");
    }
    external=state_transfer.Apply(state_transfer.E,low,ne,4);ClearEnrichmentCache();work.limiter+=Wall(start);
  }
  State Closure(const State& external,double time,TrackWork& work)const
  {return PrimitiveMomentProblem::Closure(InternalMoments(external),time,work);}
  State Residual(const State& external,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    const State& internal=InternalMoments(external);State rhs=EntropyCoupledProblem::Residual(internal,sigma,time,work,budget);last_unrestricted_rhs=rhs;
    const auto start=Clock::now();rhs=RestrictExternal(rhs);work.transport+=Wall(start);
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));return rhs;
  }
  // Differential of internal E=Ep+(Pi_q-Pi_p)K. Used only for frozen output
  // and physical entropy diagnostics, not as another evolving unknown.
  State InternalDirection(const State& external,const State& direction)const
  {
    if(base_order==retained.p||enrich_energy)return direction;
    const State& physical=Physical(external);const State low=state_transfer.Apply(state_transfer.P,direction,ne,4);State out=direction;const int m=retained.n;
    for(int e=0;e<ne;++e)
    {
      BandMatrix weighted(m,m-1);State rate(2*m,0.),drho(weights.size(),0.),load(m,0.),kt(m,0.);std::vector<std::array<double,4>> points(weights.size());
      for(std::size_t q=0;q<weights.size();++q)
      {
        const auto u=Point(physical,e,shapes[q]);points[q]={u[0],u[1]/u[0],u[2]/u[0],0.};std::array<double,2> dm{};
        for(int i=0;i<m;++i){drho[q]+=low_shapes[q][i]*low[(e*m+i)*4];for(int c=0;c<2;++c)dm[c]+=low_shapes[q][i]*low[(e*m+i)*4+c+1];}
        for(int i=0;i<m;++i){for(int j=0;j<m;++j)weighted.At(i,j)+=weights[q]*u[0]*low_shapes[q][i]*low_shapes[q][j];for(int c=0;c<2;++c)rate[c*m+i]+=weights[q]*low_shapes[q][i]*(dm[c]-drho[q]*points[q][c+1]);}
      }
      BandFactor factor(weighted,0,m);factor.Solve(rate,2);
      for(std::size_t q=0;q<weights.size();++q)
      {
        std::array<double,2> dv{};for(int i=0;i<m;++i)for(int c=0;c<2;++c)dv[c]+=low_shapes[q][i]*rate[c*m+i];const auto a=points[q];
        const double value=.5*drho[q]*(a[1]*a[1]+a[2]*a[2])+a[0]*(a[1]*dv[0]+a[2]*dv[1]);for(int i=0;i<m;++i)load[i]+=weights[q]*low_shapes[q][i]*value;
      }
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)kt[i]+=retained.mass_inverse(i,j)*load[j];
      const State small=coarse_transfer.Apply(coarse_transfer.P,kt,1,1),back=coarse_transfer.Apply(coarse_transfer.E,small,1,1);
      for(int i=0;i<n;++i)for(int j=0;j<m;++j)out[Id(e,i,3)]+=state_transfer.E(i,j)*(kt[j]-back[j]);
    }
    return out;
  }
  double EntropyRate(const State& external,const State& direction)const
  {const State internal=InternalMoments(external),rate=InternalDirection(external,direction);return EntropyCoupledProblem::EntropyRate(internal,rate);}
};
