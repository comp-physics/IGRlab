#pragma once
// Research only: current-state pressure-source VI in the existing full-Pp SIPG
// stress space. Linear QPs use a declared tightened sampled cone; polynomial
// extrema certify the original physical cone. No stress/source time lag.
struct NativeObstacleStress
{
  double tau=.5,bound=6.,margin=.001,tolerance=2.e-9;
  mutable std::uint64_t calls=0,factors=0,qp_solves=0,source_points=0,unconstrained_trials=0,unconstrained_accepts=0;
  mutable int maximum_qp_iterations=0,maximum_cuts=0,maximum_picard=0;
  mutable double seconds=0.,maximum_kkt=0.,maximum_gap=0.,maximum_nonlinear=0.;
  mutable double minimum_stress=INFINITY,minimum_work=INFINITY,minimum_pressure=INFINITY;
  mutable double physical_source_integral=0.,effective_source_integral=0.,energy_bound_ratio=0.;
  mutable State last_sigma,last_physical_load,last_effective_load,last_gradient,last_pressure,last_reaction;
  using Poly=std::vector<long double>;
  static long double Eval(const Poly& c,long double x)
  {long double p=1.,v=c.empty()?0.:c[0],r=0.;for(int j=1;j<int(c.size());++j){const long double q=((2*j-1)*x*p-(j-1)*r)/j;v+=c[j]*q;r=p;p=q;}return v;}
  static Poly Deriv(const Poly& c)
  {Poly d(std::max(1,int(c.size())-1),0.);for(int j=1;j<int(c.size());++j)for(int k=j-1;k>=0;k-=2)d[k]+=(2*k+1)*c[j];return d;}
  static State Roots(Poly c)
  {
    long double scale=0.;for(auto v:c)scale=std::max(scale,std::abs(v));
    while(c.size()>1&&std::abs(c.back())<1.e-17L*std::max(1.L,scale))c.pop_back();
    if(c.size()<2)return {};if(c.size()==2){const double x=-c[0]/c[1];return x>-1.&&x<1.?State{x}:State{};}
    State intervals=Roots(Deriv(c));intervals.insert(intervals.begin(),-1.);intervals.push_back(1.);State roots;
    for(std::size_t i=1;i+1<intervals.size();++i)if(std::abs(Eval(c,intervals[i]))<1.e-15L*std::max(1.L,scale))roots.push_back(intervals[i]);
    for(std::size_t i=0;i+1<intervals.size();++i)
    {double a=intervals[i],b=intervals[i+1];long double fa=Eval(c,a),fb=Eval(c,b);if(fa*fb>=0.)continue;
      for(int k=0;k<60;++k){const double m=.5*(a+b);const long double fm=Eval(c,m);if(fa*fm<=0.){b=m;fb=fm;}else{a=m;fa=fm;}}
      roots.push_back(.5*(a+b));}
    std::sort(roots.begin(),roots.end());roots.erase(std::unique(roots.begin(),roots.end(),[](double a,double b){return std::abs(a-b)<1.e-11;}),roots.end());return roots;
  }
  struct Row{int e;State v;double d;};
  static State Apply(const BandMatrix& a,const State& x)
  {State y(a.n,0.);for(int i=0;i<a.n;++i)for(int j=std::max(0,i-a.k);j<std::min(a.n,i+a.k+1);++j)y[i]+=a.Get(i,j)*x[j];return y;}
  static double Dot(const State& a,const State& b){double x=0.;for(std::size_t i=0;i<a.size();++i)x+=a[i]*b[i];return x;}
  static double Inf(const State& x){double a=0.;for(double v:x)a=std::max(a,std::abs(v));return a;}
  static double Step(const State& q,const State& d)
  {double a=1.;for(std::size_t i=0;i<q.size();++i)if(d[i]<0.)a=std::min(a,-q[i]/d[i]);return a;}
  State SolveQP(const BandMatrix& a,const State& b,const std::vector<Row>& rows,int m,double seed,State& dual,double& kkt,double& gap)const
  {
    const int n=a.n,nr=int(rows.size());State x(n,seed),slack(nr),lam(nr,std::max(1.e-4,Inf(b)/64.));
    auto cx=[&](const Row& r,const State& xx){double v=0.;for(int j=0;j<m;++j)v+=r.v[j]*xx[r.e*m+j];return v;};
    for(int k=0;k<nr;++k){slack[k]=rows[k].d-cx(rows[k],x);if(!(slack[k]>0.))throw std::runtime_error("obstacle QP strict seed failed");}
    ++qp_solves;
    for(int iteration=0;iteration<80;++iteration)
    {
      State rd=Apply(a,x),rp(nr);for(int i=0;i<n;++i)rd[i]-=b[i];
      for(int k=0;k<nr;++k){const auto& r=rows[k];rp[k]=cx(r,x)+slack[k]-r.d;for(int j=0;j<m;++j)rd[r.e*m+j]+=r.v[j]*lam[k];}
      gap=Dot(slack,lam);kkt=std::max(Inf(rd)/std::max(1.,Inf(b)),Inf(rp));
      if(std::max(kkt,gap/std::max(1.,std::abs(Dot(b,x))))<tolerance)
      {maximum_qp_iterations=std::max(maximum_qp_iterations,iteration);maximum_kkt=std::max(maximum_kkt,kkt);maximum_gap=std::max(maximum_gap,gap);dual=lam;return x;}
      BandMatrix K=a;
      for(int k=0;k<nr;++k){const auto& r=rows[k];const double w=lam[k]/slack[k];for(int i=0;i<m;++i)for(int j=0;j<m;++j)K.At(r.e*m+i,r.e*m+j)+=w*r.v[i]*r.v[j];}
      // Symmetric diagonal equilibration preserves the exact Newton system.
      State scale(n);for(int i=0;i<n;++i)scale[i]=1./std::sqrt(K.Get(i,i));
      for(int i=0;i<n;++i)for(int j=std::max(0,i-K.k);j<std::min(n,i+K.k+1);++j)K.At(i,j)*=scale[i]*scale[j];
      BandFactor factor(K,0,n);++factors;
      auto direction=[&](const State& rc,State& dx,State& ds,State& dl)
      {
        dx=rd;for(double&v:dx)v=-v;
        for(int k=0;k<nr;++k){const auto&r=rows[k];const double f=(rc[k]-lam[k]*rp[k])/slack[k];for(int j=0;j<m;++j)dx[r.e*m+j]+=r.v[j]*f;}
        for(int i=0;i<n;++i)dx[i]*=scale[i];factor.Solve(dx,1);for(int i=0;i<n;++i)dx[i]*=scale[i];
        ds.resize(nr);dl.resize(nr);for(int k=0;k<nr;++k){ds[k]=-rp[k]-cx(rows[k],dx);dl[k]=(-rc[k]-lam[k]*ds[k])/slack[k];}
      };
      State rc(nr),dx,ds,dl;for(int k=0;k<nr;++k)rc[k]=slack[k]*lam[k];direction(rc,dx,ds,dl);
      const double ap=Step(slack,ds),ad=Step(lam,dl);double mua=0.;for(int k=0;k<nr;++k)mua+=(slack[k]+ap*ds[k])*(lam[k]+ad*dl[k]);
      const double cent=std::pow(std::min(1.,std::max(0.,mua/gap)),3),mu=gap/nr;
      for(int k=0;k<nr;++k)rc[k]=slack[k]*lam[k]+ds[k]*dl[k]-cent*mu;
      direction(rc,dx,ds,dl);const double sp=.995*Step(slack,ds),sd=.995*Step(lam,dl);
      for(int i=0;i<n;++i)x[i]+=sp*dx[i];for(int k=0;k<nr;++k){slack[k]+=sp*ds[k];lam[k]+=sd*dl[k];if(!(slack[k]>0.)||!(lam[k]>0.))throw std::runtime_error("obstacle QP interior admission failed");}
    }
    throw std::runtime_error("obstacle QP iteration limit");
  }
  template<class Problem>
  void Update(const Problem& pr,const State& physical,TrackWork& work,const BandMatrix& A,State& effective)const
  {
    const auto started=Clock::now();const int m=pr.retained.n,ne=pr.ne,nq=int(pr.weights.size()),p=m-1;const double alpha=pr.o.base.alpha;
    std::vector<State> rho(ne,State(m)),u=rho,P=rho,g=rho,px=rho;const auto&D=pr.retained.physics.basis.D;const auto&MI=pr.retained.mass_inverse;
    for(int e=0;e<ne;++e)for(int i=0;i<m;++i)
    {const auto U=pr.Point(physical,e,pr.primitive_nodes[i]);rho[e][i]=U[0];u[e][i]=U[1]/U[0];P[e][i]=(gamma_euler-1.)*(U[3]-.5*(U[1]*U[1]+U[2]*U[2])/U[0]);}
    for(int e=0;e<ne;++e)
    {
      const double h=pr.edges[e+1]-pr.edges[e];
      for(int i=0;i<m;++i)for(int j=0;j<m;++j){g[e][i]+=D(i,j)*u[e][j]/h;px[e][i]+=D(i,j)*P[e][j]/h;}
      for(int side=0;side<2;++side)
      {const int f=e+(side?1:-1),i=side?m-1:0,j=side?0:m-1;const bool wall=f<0||f>=ne;const double normal=side?1.:-1.,uv=wall?-u[e][i]:u[f][j],pv=wall?P[e][i]:P[f][j];
        for(int k=0;k<m;++k){g[e][k]+=.5*normal*MI(k,i)/h*(uv-u[e][i]);px[e][k]+=.5*normal*MI(k,i)/h*(pv-P[e][i]);}}
    }
    auto value=[&](const State& v,const State& shape){return Dot(v,shape);};
    std::vector<State> points(ne);const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,63);
    for(auto&pts:points){pts={0.,1.};for(int k=0;k<rule.GetNPoints();++k)pts.push_back(rule.IntPoint(k).x);}
    // Polynomial conversion is a separate sufficiently exact positive rule.
    const auto& poly_rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,4*p+5);
    auto poly=[&](int e,const State& s,bool work_poly)
    {Poly out(work_poly?2*p+1:m,0.);for(int k=0;k<poly_rule.GetNPoints();++k){const auto&q=poly_rule.IntPoint(k);const State shape=pr.retained.Shape(q.x,false);const double S=0.+Dot(s,shape),v=work_poly?bound*value(P[e],shape)-tau*value(g[e],shape)*S:S;const long double z=2.*q.x-1.;long double prev=0.,cur=1.;for(int j=0;j<int(out.size());++j){out[j]+=(2*j+1)*q.weight*v*cur;const long double next=((2*j+1)*z*cur-j*prev)/(j+1);prev=cur;cur=next;}}return out;};
    State source(A.n),base(A.n,0.);double pressure_C=0.;
    auto load=[&](const State& sigma)
    {std::fill(source.begin(),source.end(),0.);physical_source_integral=0.;pressure_C=0.;std::fill(base.begin(),base.end(),0.);
      for(int e=0;e<ne;++e)for(int q=0;q<nq;++q)
      {const auto& shape=pr.low_shapes[q];const double r=value(rho[e],shape),pressure=value(P[e],shape),gg=value(g[e],shape),pg=value(px[e],shape);double S=0.;for(int i=0;i<m;++i)S+=shape[i]*sigma[e*m+i];
        if(!(r>floor_value)||!(pressure>floor_value)||S<-1.e-10)throw std::runtime_error("obstacle source state admission failed");
        const double Q=gamma_euler*pressure+(gamma_euler-1.)*S,nn=alpha*(3.*gamma_euler+1.)*std::pow(std::min(gg,0.),2),c=alpha*pg*pg/r;
        const double f=(2.*gamma_euler*pressure+(2.*gamma_euler-1.)*S)/(Q*Q),N=nn+c*f,w=(pr.edges[e+1]-pr.edges[e])*pr.weights[q];
        for(int i=0;i<m;++i){source[e*m+i]+=w*shape[i]*N;base[e*m+i]+=w*shape[i]*nn;}
        physical_source_integral+=w*N;pressure_C+=w*c*(2.*gamma_euler-1.)/std::pow(gamma_euler-1.,2);minimum_pressure=std::min(minimum_pressure,pressure);++source_points;++work.source_points;
      }
    };
    BandFactor af(A,0,A.n);++factors;
    State sigma(A.n,0.),dual;load(sigma);bool nonlinear_done=false;int picard=0;
    for(;picard<40;++picard)
    {
      State old_source=source;bool certified=false;double kkt=0.,gap=0.;
      // An unconstrained frozen-source minimizer already in the original
      // physical cone is the exact VI minimizer. This preserves zero-source,
      // small-amplitude and admissible all-compressive states without a floor.
      State trial=source;af.Solve(trial,1);++unconstrained_trials;double rawmins=INFINITY,rawminw=INFINITY;
      for(int e=0;e<ne;++e)
      {State cell(m);for(int i=0;i<m;++i)cell[i]=trial[e*m+i];for(int mode=0;mode<2;++mode){const Poly c=poly(e,cell,mode==1);State ext=Roots(Deriv(c));ext.push_back(-1.);ext.push_back(1.);for(double z:ext){const double v=Eval(c,z);(mode?rawminw:rawmins)=std::min(mode?rawminw:rawmins,v);}}}
      if(rawmins>=-1.e-10&&rawminw>=-1.e-10)
      {sigma=trial;certified=true;++unconstrained_accepts;minimum_stress=std::min(minimum_stress,rawmins);minimum_work=std::min(minimum_work,rawminw);}
      for(int cut=0;!certified&&cut<16;++cut)
      {
        std::vector<Row> rows;double local_scale=INFINITY;
        for(int e=0;e<ne;++e)for(double z:points[e]){const auto shape=pr.retained.Shape(z,false);const double pp=value(P[e],shape),gg=value(g[e],shape);if(!(pp>floor_value))throw std::runtime_error("obstacle constraint pressure admission");local_scale=std::min(local_scale,bound*pp/std::max(1.,std::abs(tau*gg)));}
        const double floor=margin*local_scale;
        for(int e=0;e<ne;++e)for(double z:points[e])
        {const State shape=pr.retained.Shape(z,false);const double pp=value(P[e],shape),gg=value(g[e],shape),scale=std::max(1.,std::abs(tau*gg));State lower=shape,upper=shape;for(int j=0;j<m;++j){lower[j]*=-1.;upper[j]*=tau*gg/scale;}rows.push_back({e,lower,-floor});rows.push_back({e,upper,(1.-margin)*bound*pp/scale});}
        sigma=SolveQP(A,source,rows,m,.1*local_scale,dual,kkt,gap);bool bad=false;double mins=INFINITY,minw=INFINITY;
        for(int e=0;e<ne;++e)
        {State cell(m);for(int i=0;i<m;++i)cell[i]=sigma[e*m+i];for(int mode=0;mode<2;++mode){const Poly c=poly(e,cell,mode==1);State extrema=Roots(Deriv(c));extrema.push_back(-1.);extrema.push_back(1.);for(double z:extrema){const double v=Eval(c,z);(mode?minw:mins)=std::min(mode?minw:mins,v);if(v<-1.e-10){const double pt=.5*(z+1.);if(std::none_of(points[e].begin(),points[e].end(),[&](double a){return std::abs(a-pt)<1.e-11;}))points[e].push_back(pt);else throw std::runtime_error("obstacle polynomial cut stalled");bad=true;}}}}
        maximum_cuts=std::max(maximum_cuts,cut);
        if(!bad){minimum_stress=std::min(minimum_stress,mins);minimum_work=std::min(minimum_work,minw);certified=true;break;}
      }
      if(!certified)throw std::runtime_error("obstacle polynomial certificate cut limit");
      load(sigma);State delta=source;for(std::size_t i=0;i<delta.size();++i)delta[i]-=old_source[i];const double residual=Inf(delta)/std::max(1.,Inf(source));
      if(residual<tolerance){maximum_nonlinear=std::max(maximum_nonlinear,residual);nonlinear_done=true;break;}
    }
    if(!nonlinear_done)throw std::runtime_error("obstacle current-source Picard limit");maximum_picard=std::max(maximum_picard,picard+1);
    effective=Apply(A,sigma);last_sigma=sigma;last_physical_load=source;last_effective_load=effective;last_reaction=source;
    last_gradient.clear();last_pressure.clear();for(int e=0;e<ne;++e){last_gradient.insert(last_gradient.end(),g[e].begin(),g[e].end());last_pressure.insert(last_pressure.end(),P[e].begin(),P[e].end());}
    effective_source_integral=0.;for(std::size_t i=0;i<effective.size();++i){effective_source_integral+=effective[i];last_reaction[i]-=effective[i];}
    State bsolve=base;af.Solve(bsolve,1);const double B=std::sqrt(std::max(0.,Dot(base,bsolve))),X=std::sqrt(std::max(0.,Dot(sigma,effective))),upper=.5*(B+std::sqrt(B*B+4.*pressure_C));
    energy_bound_ratio=std::max(energy_bound_ratio,X/std::max(upper,1.e-30));
    // Existing outer linear solve receives A Sigma, never the unconstrained
    // physical source. Its residual/mean tests certify this adapter only. The
    // source, reaction and current nonlinear KKT are serialized separately.
    work.current_source_integral=effective_source_integral;++calls;seconds+=Wall(started);
  }
};
