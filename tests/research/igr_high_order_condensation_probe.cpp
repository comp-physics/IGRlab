// Test-only exact grouping of the existing high-order H1 IGR operator.
// Euler DOFs, FE pieces, nodal quadrature and current coefficients are retained.
// Dense interiors are compared with compact banded Cholesky, not global dense
// elimination alone. The global dense system is an independent numerical oracle.
#ifndef DG_IGR_HIGH_ORDER_CONDENSATION_LIBRARY_ONLY
#define main igr_original_screened_spacetime_main
#include "screened_spacetime_probe.cpp"
#undef main
#endif
#include <fstream>
#include <limits>
#include <sstream>
#include "igr_high_order_front.hpp"

namespace
{
struct CondOptions
{
  Options base;int aggregate=1,repeats=8,trace_block=1;
  std::string factor="band",initial="smooth",output="";double front_width=.012;
  bool self_test=false,trajectory=false;
};
struct CondWork
{
  double assembly=0,factor=0,responses=0,trace=0,reconstruct=0,certificate=0;
  double factor_products=0,response_products=0,schur_products=0;
  std::size_t local_matrix_entries=0,local_factor_entries=0,response_entries=0;
  double residual=0;std::uint64_t calls=0,active_source_columns=0;
};
// Row-major compact matrix, with structural half-bandwidth p. This storage also
// supplies the original-operator residual without reconstructing a global A.
struct BandMatrix
{
  int n=0,k=0;State a;
  BandMatrix()=default;
  BandMatrix(int size,int half):n(size),k(half),a(size*(2*half+1),0.){}
  double Get(int i,int j)const
  {return (i<0||j<0||i>=n||j>=n||std::abs(i-j)>k)?0.:a[i*(2*k+1)+j-i+k];}
  double& At(int i,int j)
  {if(std::abs(i-j)>k)throw std::runtime_error("band overflow");return a[i*(2*k+1)+j-i+k];}
};
struct BandFactor
{
  int n=0,k=0;State l;double products=0;
  BandFactor(const BandMatrix& a,int first,int count):n(count),k(std::min(a.k,std::max(0,count-1))),l(count*(k+1),0.)
  {
    for(int i=0;i<n;++i)for(int j=std::max(0,i-k);j<=i;++j)
    {
      double x=a.Get(first+i,first+j);
      for(int t=std::max({0,i-k,j-k});t<j;++t){x-=Get(i,t)*Get(j,t);++products;}
      if(i==j){if(!(x>0.)||!std::isfinite(x))throw std::runtime_error("non-SPD band factor");At(i,j)=std::sqrt(x);}
      else At(i,j)=x/Get(j,j);
    }
  }
  double Get(int i,int j)const{return i<j||i-j>k?0.:l[i*(k+1)+i-j];}
  double& At(int i,int j){return l[i*(k+1)+i-j];}
  void Solve(State& b,int columns)const
  {
    for(int c=0;c<columns;++c)
    {
      for(int i=0;i<n;++i){for(int j=std::max(0,i-k);j<i;++j)b[c*n+i]-=Get(i,j)*b[c*n+j];b[c*n+i]/=Get(i,i);}
      for(int i=n-1;i>=0;--i){for(int j=i+1;j<std::min(n,i+k+1);++j)b[c*n+i]-=Get(j,i)*b[c*n+j];b[c*n+i]/=Get(i,i);}
    }
  }
};
struct LocalCond
{
  BandMatrix a;State b,x;std::array<double,4>s{};std::array<double,2>r{};
};
LocalCond LocalAssemble(const Problem& pr,const State& u,int first,int count)
{
  LocalCond c;c.a=BandMatrix(count*pr.p+1,pr.p);c.b.assign(c.a.n,0.);
  for(int e=0;e<count;++e)for(int q=0;q<pr.n;++q)
  {
    const double* state=&u[pr.Index(first+e,q,0)];pr.CheckState(state);double du[8]={};
    for(int field=0;field<4;++field)for(int j=0;j<pr.n;++j)
      du[2*field]+=pr.basis.D(q,j)*u[pr.Index(first+e,j,field)]/pr.h;
    const double rhoinv=igr::SafeInverseDensity(state[0],floor_value);
    const double source=igr::SourceFromConservativeStateGradient(2,state,1,du,2,1,pr.o.alpha,floor_value);
    if(!std::isfinite(source))throw std::runtime_error("nonfinite local source");
    c.a.At(e*pr.p+q,e*pr.p+q)+=pr.h*pr.basis.weights[q]*rhoinv;
    c.b[e*pr.p+q]+=pr.h*pr.basis.weights[q]*source;
    for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)
      c.a.At(e*pr.p+i,e*pr.p+j)+=pr.o.alpha*rhoinv*pr.basis.weights[q]*pr.basis.D(q,i)*pr.basis.D(q,j)/pr.h;
  }
  return c;
}
void Condense(LocalCond& c,const std::string& kind,CondWork& w)
{
  const int m=c.a.n-2;const int last=c.a.n-1;const auto start=Clock::now();
  c.x.assign(3*m,0.);
  double source_norm=0.;for(int i=1;i<c.a.n-1;++i)source_norm+=c.b[i]*c.b[i];if(source_norm>1.e-40)++w.active_source_columns;
  for(int i=0;i<m;++i){c.x[i]=c.a.Get(i+1,0);c.x[m+i]=c.a.Get(i+1,last);c.x[2*m+i]=c.b[i+1];}
  if(kind=="band")
  {
    BandFactor f(c.a,1,m);w.factor+=Wall(start);w.factor_products+=f.products;
    w.local_factor_entries+=f.l.size();const auto rhs_start=Clock::now();f.Solve(c.x,3);w.responses+=Wall(rhs_start);
    w.response_products+=6.*m*f.k-3.*f.k*(f.k+1.);
  }
  else
  {
    mfem::DenseMatrix d(m);for(int j=0;j<m;++j)for(int i=0;i<m;++i)d(i,j)=c.a.Get(i+1,j+1);
    mfem::DenseMatrixInverse inverse(d,true);w.factor+=Wall(start);w.factor_products+=double(m)*m*m/6.;
    w.local_factor_entries+=std::size_t(m)*m;const auto rhs_start=Clock::now();
    mfem::DenseMatrix rhs(c.x.data(),m,3),sol(m,3);inverse.Mult(rhs,sol);
    std::copy(sol.GetData(),sol.GetData()+3*m,c.x.begin());w.responses+=Wall(rhs_start);w.response_products+=3.*m*m;
  }
  const auto schur_start=Clock::now();
  for(int i=0;i<2;++i)
  {
    const int row=i?last:0;c.r[i]=c.b[row];
    for(int j=0;j<2;++j)c.s[2*i+j]=c.a.Get(row,j?last:0);
    for(int k=0;k<m;++k)
    {const double e=c.a.Get(row,k+1);if(e==0.)continue;
      c.r[i]-=e*c.x[2*m+k];for(int j=0;j<2;++j)c.s[2*i+j]-=e*c.x[j*m+k];w.schur_products+=3;}
  }
  w.responses+=Wall(schur_start);w.local_matrix_entries+=c.a.a.size();w.response_entries+=c.x.size();
}
struct TraceSystem
{
  State diagonal,edge,b;
  explicit TraceSystem(int n):diagonal(n,0.),edge(n,0.),b(n,0.){}
  int Size()const{return int(b.size());}
  void Add(int g,const LocalCond& c)
  {const int j=(g+1)%Size();diagonal[g]+=c.s[0];diagonal[j]+=c.s[3];edge[g]+=.5*(c.s[1]+c.s[2]);b[g]+=c.r[0];b[j]+=c.r[1];}
  State Apply(const State& x)const
  {State y(x.size());for(int i=0;i<Size();++i)y[i]=diagonal[i]*x[i];
    for(int i=0;i<Size();++i){int j=(i+1)%Size();y[i]+=edge[i]*x[j];y[j]+=edge[i]*x[i];}return y;}
  State Solve()const
  {
    const int n=Size();if(n==1)return State{b[0]/(diagonal[0]+2*edge[0])};
    // Eliminate all except trace zero using a tridiagonal factor. Two real
    // columns are the trace-zero response and current RHS; cyclic wrap exact.
    BandMatrix interior(n-1,1);State rhs(2*(n-1),0.);
    for(int i=1;i<n;++i){interior.At(i-1,i-1)=diagonal[i];rhs[n-1+i-1]=b[i];
      if(i<n-1){interior.At(i-1,i)=edge[i];interior.At(i,i-1)=edge[i];}}
    rhs[0]+=edge[0];rhs[n-2]+=edge[n-1];BandFactor f(interior,0,n-1);f.Solve(rhs,2);
    double schur=diagonal[0],r=b[0];
    schur-=edge[0]*rhs[0]+edge[n-1]*rhs[n-2];
    r-=edge[0]*rhs[n-1]+edge[n-1]*rhs[2*n-3];
    State x(n);x[0]=r/schur;for(int i=1;i<n;++i)x[i]=rhs[n-1+i-1]-rhs[i-1]*x[0];return x;
  }
  mfem::DenseMatrix Dense()const
  {mfem::DenseMatrix a(Size());a=0.;for(int i=0;i<Size();++i){int j=(i+1)%Size();a(i,i)+=diagonal[i];a(i,j)+=edge[i];a(j,i)+=edge[i];}return a;}
};
State Recover(const Problem& pr,const std::vector<LocalCond>& local,const State& trace,int aggregate)
{
  State sigma(pr.ne*pr.p,0.);for(int g=0;g<int(local.size());++g)
  {const auto& c=local[g];int m=c.a.n-2;double left=trace[g],right=trace[(g+1)%trace.size()];sigma[g*aggregate*pr.p]=left;
   for(int i=0;i<m;++i)sigma[g*aggregate*pr.p+i+1]=c.x[2*m+i]-left*c.x[i]-right*c.x[m+i];}return sigma;
}
double OriginalResidual(const Problem& pr,const std::vector<LocalCond>& local,const State& sigma,int aggregate)
{
  State r(sigma.size(),0.),b(sigma.size(),0.);for(int g=0;g<int(local.size());++g)
  {const auto& c=local[g];for(int i=0;i<c.a.n;++i){int row=(g*aggregate*pr.p+i)%sigma.size();b[row]+=c.b[i];r[row]-=c.b[i];
    for(int j=std::max(0,i-c.a.k);j<std::min(c.a.n,i+c.a.k+1);++j)r[row]+=c.a.Get(i,j)*sigma[(g*aggregate*pr.p+j)%sigma.size()];}}
  return Norm(r)/std::max(Norm(b),1.e-20);
}
struct CondResult{State sigma;std::vector<LocalCond> local;TraceSystem trace;CondResult(int n):trace(n){}};
// The callback form is shared only by the high-order test probes. It permits
// the hybrid probe to preserve its own collocated or overintegrated FE matrix
// exactly, rather than substituting this probe's nodal quadrature.
template <typename LocalAssembler>
CondResult SolveCondensedWithLocalAssembler(const Problem& pr,const State& u,
    int aggregate,const std::string& factor,CondWork& w,
    LocalAssembler assemble,bool certify=true)
{
  if(aggregate<1||pr.ne%aggregate)throw std::runtime_error("aggregate must divide original elements");
  if(factor!="band"&&factor!="dense")throw std::runtime_error("unknown local factor");
  CondResult result(pr.ne/aggregate);++w.calls;
  for(int g=0;g<pr.ne/aggregate;++g)
  {auto start=Clock::now();result.local.push_back(assemble(u,g*aggregate,aggregate));w.assembly+=Wall(start);
   Condense(result.local.back(),factor,w);start=Clock::now();result.trace.Add(g,result.local.back());w.trace+=Wall(start);}
  auto start=Clock::now();const State trace=result.trace.Solve();w.trace+=Wall(start);start=Clock::now();
  result.sigma=Recover(pr,result.local,trace,aggregate);w.reconstruct+=Wall(start);
  if(certify){start=Clock::now();const double r=OriginalResidual(pr,result.local,result.sigma,aggregate);w.residual=std::max(w.residual,r);w.certificate+=Wall(start);
    if(!(r<1.e-8))throw std::runtime_error("current original residual failed");}return result;
}
CondResult SolveCondensed(const Problem& pr,const State& u,int aggregate,
    const std::string& factor,CondWork& w,bool certify=true)
{
  return SolveCondensedWithLocalAssembler(pr,u,aggregate,factor,w,
    [&pr](const State& state,int first,int count){return LocalAssemble(pr,state,first,count);},certify);
}
// ElementAssembler(state, original_element, matrix, rhs) must fill exactly the
// current n-by-n element matrix and n-entry RHS in the original H1 numbering.
// Packing all entries, callback allocation and quadrature are charged to assembly.
template <typename ElementAssembler>
CondResult SolveCondensedFromElements(const Problem& pr,const State& u,
    int aggregate,const std::string& factor,CondWork& w,
    ElementAssembler element,bool certify=true)
{
  return SolveCondensedWithLocalAssembler(pr,u,aggregate,factor,w,
    [&pr,&element](const State& state,int first,int count)
    {
      LocalCond local;local.a=BandMatrix(count*pr.p+1,pr.p);local.b.assign(local.a.n,0.);
      mfem::DenseMatrix a;State b;
      for(int e=0;e<count;++e)
      {
        element(state,first+e,a,b);
        if(a.Height()!=pr.n||a.Width()!=pr.n||int(b.size())!=pr.n)
          throw std::runtime_error("element callback shape mismatch");
        for(int i=0;i<pr.n;++i)
        {local.b[e*pr.p+i]+=b[i];for(int j=0;j<pr.n;++j)local.a.At(e*pr.p+i,e*pr.p+j)+=a(i,j);}
      }
      return local;
    },certify);
}
State CondInitial(const Problem& pr,const std::string& initial,double front_width)
{
  if(initial=="smooth"||initial=="compression")return pr.Initial();
  const auto front=MakeHighOrderPeriodicFront(front_width);State u(pr.ne*pr.n*4);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {
    mfem::Vector point(4);if(initial=="constant"){point[0]=1.;point[1]=.2;point[2]=0.;point[3]=1./(gamma_euler-1.)+.02;}
    else front(pr.h*(e+pr.basis.nodes[i]),point);
    for(int c=0;c<4;++c)u[pr.Index(e,i,c)]=point[c];
  }return u;
}
struct IterationInfo{double radius=0,row_bound=0,last_residual=0;int iterations=0;bool nonnegative=true;State sigma;};
IterationInfo TraceIteration(const Problem& pr,const CondResult& exact,int aggregate,int block)
{
  const auto& t=exact.trace;const int n=t.Size();mfem::DenseMatrix a=t.Dense();
  std::vector<std::unique_ptr<mfem::DenseMatrix>> matrices;
  std::vector<std::unique_ptr<mfem::DenseMatrixInverse>> inverses;
  for(int first=0;first<n;first+=block)
  {int m=std::min(block,n-first);matrices.emplace_back(new mfem::DenseMatrix(m));for(int j=0;j<m;++j)for(int i=0;i<m;++i)(*matrices.back())(i,j)=a(first+i,first+j);
    inverses.emplace_back(new mfem::DenseMatrixInverse(*matrices.back(),true));}
  auto precondition=[&](const State& b){State y(n);for(int first=0,g=0;first<n;first+=block,++g)
    {const int m=std::min(block,n-first);mfem::Vector in(const_cast<double*>(b.data()+first),m),out(y.data()+first,m);inverses[g]->Mult(in,out);}return y;};
  auto apply=[&](const State& x){State ax=t.Apply(x),px=precondition(ax),y(n);for(int i=0;i<n;++i)y[i]=x[i]-px[i];return y;};
  IterationInfo out;
  // Generalized symmetric spectrum: B is similar to I-L^-1 A L^-T,
  // where LL^T is the block diagonal. This remains valid if a coarse, highly
  // screened high-order discretization loses the trace M-matrix property.
  State inverse_lower(n*n,0.);
  for(int first=0;first<n;first+=block)
  {
    const int m=std::min(block,n-first);State l(m*m,0.);
    for(int i=0;i<m;++i)for(int j=0;j<=i;++j)
    {double value=a(first+i,first+j);for(int k=0;k<j;++k)value-=l[i*m+k]*l[j*m+k];
      l[i*m+j]=i==j?std::sqrt(value):value/l[j*m+j];}
    for(int col=0;col<m;++col)for(int i=0;i<m;++i)
    {double value=i==col?1.:0.;for(int k=0;k<i;++k)value-=l[i*m+k]*inverse_lower[(first+k)*n+first+col];
      inverse_lower[(first+i)*n+first+col]=value/l[i*m+i];}
  }
  State transformed(n*n,0.),symmetric(n*n,0.);
  for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<n;++k)
    transformed[i*n+j]+=inverse_lower[i*n+k]*a(k,j);
  for(int i=0;i<n;++i)for(int j=0;j<n;++j)
  {symmetric[i*n+j]=i==j?1.:0.;for(int k=0;k<n;++k)symmetric[i*n+j]-=transformed[i*n+k]*inverse_lower[j*n+k];}
  for(int sweep=0;sweep<80;++sweep)
  {
    double off=0.;for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)off=std::max(off,std::abs(symmetric[i*n+j]));if(off<2.e-15)break;
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)
    {
      const double b=symmetric[i*n+j];if(std::abs(b)<1.e-16)continue;
      const double angle=.5*std::atan2(2*b,symmetric[j*n+j]-symmetric[i*n+i]);
      const double c=std::cos(angle),s=std::sin(angle),ai=symmetric[i*n+i],aj=symmetric[j*n+j];
      symmetric[i*n+i]=c*c*ai-2*s*c*b+s*s*aj;symmetric[j*n+j]=s*s*ai+2*s*c*b+c*c*aj;symmetric[i*n+j]=symmetric[j*n+i]=0.;
      for(int k=0;k<n;++k)if(k!=i&&k!=j)
      {const double x=symmetric[k*n+i],y=symmetric[k*n+j];symmetric[k*n+i]=symmetric[i*n+k]=c*x-s*y;symmetric[k*n+j]=symmetric[j*n+k]=s*x+c*y;}
    }
  }
  for(int i=0;i<n;++i)out.radius=std::max(out.radius,std::abs(symmetric[i*n+i]));
  State row_sum(n,0.);for(int j=0;j<n;++j){State unit(n,0.);unit[j]=1.;State col=apply(unit);for(int i=0;i<n;++i){row_sum[i]+=std::abs(col[i]);if(col[i]<-1.e-10)out.nonnegative=false;}}
  for(double value:row_sum)out.row_bound=std::max(out.row_bound,value);
  State x(n,0.);const double bn=std::max(Norm(t.b),1.e-20);
  for(int step=0;step<=10000;++step)
  {State r=t.Apply(x);for(int i=0;i<n;++i)r[i]=t.b[i]-r[i];out.last_residual=Norm(r)/bn;out.iterations=step;
   if(out.last_residual<1.e-10)break;if(step==10000)break;State correction=precondition(r);for(int i=0;i<n;++i)x[i]+=correction[i];}
  out.sigma=Recover(pr,exact.local,x,aggregate);return out;
}
CondOptions CondParse(int argc,char** argv)
{
  CondOptions o;o.base.elements=16;o.base.order=5;o.base.alpha=.0025;o.base.dt=.0001;o.base.final_time=.005;
  for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--self-test"){o.self_test=true;continue;}if(key=="--trajectory"){o.trajectory=true;continue;}
    if(i+1==argc)throw std::runtime_error("missing argument");std::string value=argv[++i];
    if(key=="--elements")o.base.elements=std::stoi(value);else if(key=="--order")o.base.order=std::stoi(value);
    else if(key=="--aggregate")o.aggregate=std::stoi(value);else if(key=="--repeats")o.repeats=std::stoi(value);
    else if(key=="--trace-block")o.trace_block=std::stoi(value);else if(key=="--alpha")o.base.alpha=std::stod(value);
    else if(key=="--front-width")o.front_width=std::stod(value);else if(key=="--dt")o.base.dt=std::stod(value);else if(key=="--horizon")o.base.final_time=std::stod(value);
    else if(key=="--factor")o.factor=value;else if(key=="--case")o.initial=value;else if(key=="--output")o.output=value;else throw std::runtime_error("unknown argument "+key);}
  if(o.base.elements<4||o.base.elements>128||o.base.order<2||o.base.order>7||o.aggregate<1||o.base.elements%o.aggregate||o.trace_block<1||o.repeats<1||o.repeats>1000||!(o.front_width>0.)||!(o.base.alpha>0.)||!(o.base.dt>0.)||!(o.base.final_time>0.)||!std::isfinite(o.front_width+o.base.alpha+o.base.dt+o.base.final_time)||(o.factor!="band"&&o.factor!="dense")||(o.initial!="constant"&&o.initial!="smooth"&&o.initial!="compression"&&o.initial!="shock"))throw std::runtime_error("unsupported options");
  o.base.initial_case=o.initial=="compression"?"compression":"smooth";return o;
}
State CondRhs(const Problem& pr,const State& u,int aggregate,const std::string& factor,CondWork& cw,Work& work)
{auto c=SolveCondensed(pr,u,aggregate,factor,cw);return pr.Residual(u,c.sigma,pr.ne,true,0,nullptr,nullptr,work);}
State CondRK4(const Problem& pr,const State& u,double dt,int aggregate,const std::string& factor,CondWork& cw,Work& w)
{State v(u.size());const State a=CondRhs(pr,u,aggregate,factor,cw,w);for(std::size_t i=0;i<u.size();++i)v[i]=u[i]+.5*dt*a[i];const State b=CondRhs(pr,v,aggregate,factor,cw,w);
 for(std::size_t i=0;i<u.size();++i)v[i]=u[i]+.5*dt*b[i];const State c=CondRhs(pr,v,aggregate,factor,cw,w);for(std::size_t i=0;i<u.size();++i)v[i]=u[i]+dt*c[i];const State d=CondRhs(pr,v,aggregate,factor,cw,w);
 for(std::size_t i=0;i<u.size();++i)v[i]=u[i]+dt*(a[i]+2*b[i]+2*c[i]+d[i])/6.;return v;}
void RunCond(const CondOptions& o,std::ostream& out)
{
  Problem pr(o.base);State u=CondInitial(pr,o.initial,o.front_width);Work oracle_work;auto start=Clock::now();State oracle=pr.ExactClosure(u,oracle_work);const double oracle_seconds=Wall(start);
  CondWork cw;auto exact=SolveCondensed(pr,u,o.aggregate,o.factor,cw);State delta=exact.sigma;for(std::size_t i=0;i<delta.size();++i)delta[i]-=oracle[i];const double error=Norm(delta)/std::max(Norm(oracle),1.e-20);
  if(!(error<1.e-8))throw std::runtime_error("global dense oracle mismatch");
  mfem::DenseMatrix schur=exact.trace.Dense();mfem::DenseMatrixInverse inverse(schur,true);State sx(exact.trace.Size());mfem::Vector bv(exact.trace.b.data(),sx.size()),xv(sx.data(),sx.size());inverse.Mult(bv,xv);const State tx=exact.trace.Solve();
  for(std::size_t i=0;i<sx.size();++i)sx[i]-=tx[i];const double schur_error=Norm(sx)/std::max(Norm(tx),1.e-20);
  if(!(schur_error<1.e-9))throw std::runtime_error("independent Schur oracle mismatch");
  const auto iteration=TraceIteration(pr,exact,o.aggregate,o.trace_block);const double iteration_original=OriginalResidual(pr,exact.local,iteration.sigma,o.aggregate);
  std::vector<double> times;CondWork timing;for(int k=0;k<o.repeats;++k){start=Clock::now();auto result=SolveCondensed(pr,u,o.aggregate,o.factor,timing);times.push_back(Wall(start));}
  std::sort(times.begin(),times.end());double trajectory_seconds=0.,trajectory_error=0.,conservation=0.,time=0.;int steps=0;CondWork trajectory_work;Work evolution;
  if(o.trajectory)
  {
    State control=u;const auto initial_integrals=pr.Integrals(u);start=Clock::now();
    while(time<o.base.final_time){double dt=std::min(o.base.dt,o.base.final_time-time);try{u=CondRK4(pr,u,dt,o.aggregate,o.factor,trajectory_work,evolution);}
      catch(const std::exception& error){throw std::runtime_error("trajectory rejected at accepted time="+std::to_string(time)+", steps="+std::to_string(steps)+": "+error.what());}time+=dt;++steps;}
    auto endpoint=SolveCondensed(pr,u,o.aggregate,o.factor,trajectory_work);for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i){double speed;pr.Flux(&u[pr.Index(e,i,0)],endpoint.sigma[pr.SigmaIndex(e,i)],speed);}trajectory_seconds=Wall(start);
    double ct=0.;Work w;while(ct<o.base.final_time){double dt=std::min(o.base.dt,o.base.final_time-ct);const State sigma=pr.ExactClosure(control,w);control=RK4Step(pr,control,sigma,dt,w);ct+=dt;}trajectory_error=pr.Error(u,control);
    const auto final_integrals=pr.Integrals(u);for(int c=0;c<4;++c)conservation=std::max(conservation,std::abs(final_integrals[c]-initial_integrals[c]));
    if(!(trajectory_error<1.e-8)||!(conservation<1.e-9))throw std::runtime_error("trajectory identity or conservation failed");
  }
  const double count=double(o.repeats);
  out<<std::setprecision(17)<<"{\"case\":\""<<o.initial<<"\",\"front_width\":"<<o.front_width<<",\"factor\":\""<<o.factor<<"\",\"physical_fe_elements\":"<<pr.ne<<",\"euler_order\":"<<pr.p<<",\"euler_scalar_dofs\":"<<pr.ne*pr.n<<",\"closure_dofs\":"<<pr.ne*pr.p<<",\"aggregate_elements\":"<<o.aggregate<<",\"solver_aggregates\":"<<pr.ne/o.aggregate<<",\"trace_dofs\":"<<exact.trace.Size()<<",\"interior_dofs_per_aggregate\":"<<o.aggregate*pr.p-1<<",\"stored_local_rhs_columns\":3,\"endpoint_response_columns\":2,\"active_source_rhs_columns_total\":"<<cw.active_source_columns<<",\"active_source_column_norm_threshold\":1e-20,\"trace_block\":"<<o.trace_block<<",\"alpha\":"<<pr.o.alpha<<",\"ideal_constant_density_sech\":"<<1./std::cosh(o.aggregate*pr.h/std::sqrt(pr.o.alpha))<<",\"trace_iteration_radius\":"<<iteration.radius<<",\"trace_iteration_row_bound\":"<<iteration.row_bound<<",\"trace_iteration_nonnegative\":"<<(iteration.nonnegative?"true":"false")<<",\"trace_iterations\":"<<iteration.iterations<<",\"trace_final_residual\":"<<iteration.last_residual<<",\"iterated_original_residual\":"<<iteration_original<<",\"original_residual\":"<<cw.residual<<",\"dense_oracle_relative_difference\":"<<error<<",\"schur_oracle_relative_difference\":"<<schur_error<<",\"global_dense_oracle_seconds\":"<<oracle_seconds<<",\"median_complete_closure_seconds\":"<<times[times.size()/2]<<",\"minimum_complete_closure_seconds\":"<<times.front()<<",\"maximum_complete_closure_seconds\":"<<times.back()<<",\"mean_assembly_seconds\":"<<timing.assembly/count<<",\"mean_factor_seconds\":"<<timing.factor/count<<",\"mean_response_and_schur_seconds\":"<<timing.responses/count<<",\"mean_trace_seconds\":"<<timing.trace/count<<",\"mean_reconstruction_seconds\":"<<timing.reconstruct/count<<",\"mean_certificate_seconds\":"<<timing.certificate/count<<",\"local_matrix_entries\":"<<cw.local_matrix_entries<<",\"local_factor_entries\":"<<cw.local_factor_entries<<",\"response_entries\":"<<cw.response_entries<<",\"local_factor_products\":"<<cw.factor_products<<",\"response_products\":"<<cw.response_products<<",\"schur_products\":"<<cw.schur_products<<",\"trajectory_horizon\":"<<time<<",\"trajectory_steps\":"<<steps<<",\"trajectory_seconds\":"<<trajectory_seconds<<",\"trajectory_dense_oracle_state_difference\":"<<trajectory_error<<",\"trajectory_max_current_residual\":"<<trajectory_work.residual<<",\"trajectory_conservation_absolute\":"<<conservation<<"}\n";
}
bool CondSelfTest()
{
  for(int p:{3,5,7})for(int aggregate:{1,2,4,8})for(const auto& factor:{"band","dense"})
  {CondOptions o;o.base.elements=8;o.base.order=p;o.aggregate=aggregate;o.factor=factor;o.repeats=1;o.initial="shock";o.output="";std::ostringstream sink;RunCond(o,sink);}
  CondOptions o;o.base.elements=8;o.base.order=3;o.base.final_time=.0002;o.base.dt=.0001;o.trajectory=true;o.aggregate=2;o.repeats=1;std::ostringstream sink;RunCond(o,sink);
  Problem pr(o.base);const State u=pr.Initial();CondWork direct_work,callback_work;
  auto direct=SolveCondensed(pr,u,2,"band",direct_work);
  auto callback=SolveCondensedFromElements(pr,u,2,"band",callback_work,
    [&pr](const State& state,int e,mfem::DenseMatrix& a,State& b)
    {State cell(pr.n*4);for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)cell[i*4+c]=state[pr.Index(e,i,c)];
      Work w;pr.Assemble(cell,1,false,a,b,w);});
  State error=direct.sigma;for(std::size_t i=0;i<error.size();++i)error[i]-=callback.sigma[i];
  if(Norm(error)>1.e-12*Norm(direct.sigma))throw std::runtime_error("element callback equivalence failed");
  std::cout<<"high-order exact condensation self-test passed\n";return true;
}
}
#ifndef DG_IGR_HIGH_ORDER_CONDENSATION_LIBRARY_ONLY
int main(int argc,char** argv)
{
  try{mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");auto o=CondParse(argc,argv);if(o.self_test)return CondSelfTest()?0:1;
    if(o.output.empty())RunCond(o,std::cout);else{std::ofstream file(o.output);if(!file)throw std::runtime_error("cannot open output");RunCond(o,file);}return 0;}
  catch(const std::exception& e){std::cerr<<"high-order condensation failed: "<<e.what()<<'\n';return 2;}
}
#endif
