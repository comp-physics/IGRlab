#pragma once
// Research-only rank-one retained-modal inertia changes. Conserved cell means
// are untouched; this is a new discretization, not a filter of the state.
inline State FrontTopModeDelta(const PrimitiveMomentProblem& pr,const State& high,
                              double factor,int component_mask)
{
  const int p=pr.retained.p,m=pr.retained.n;
  const State low=pr.state_transfer.Apply(pr.state_transfer.P,high,pr.ne,4);
  State delta(low.size(),0.),polynomial(m);
  for(int i=0;i<m;++i)
  {
    const double x=2.*pr.retained.physics.basis.nodes[i]-1.;
    double a=1.,b=x;
    for(int k=2;k<=p;++k){const double c=((2*k-1)*x*b-(k-1)*a)/k;a=b;b=c;}
    polynomial[i]=p?b:a;
  }
  // LGL quadrature of Pp^2 on [0,1] is 1/p, while all lower cross
  // products are exact and vanish. Divide by that discrete norm.
  for(int e=0;e<pr.ne;++e)for(int c=0;c<4;++c)if(component_mask&(1<<c))
  {
    double top=0.;for(int i=0;i<m;++i)
      top+=p*pr.retained.physics.basis.weights[i]*polynomial[i]*low[(e*m+i)*4+c];
    for(int i=0;i<m;++i)delta[(e*m+i)*4+c]=(factor-1.)*top*polynomial[i];
  }
  return pr.state_transfer.Apply(pr.state_transfer.E,delta,pr.ne,4);
}
