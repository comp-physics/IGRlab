#include "numerics/hdg/core/host_reference.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace
{
using dg::hdg::LocalBlockShape;
using dg::hdg::LocalIndex;
using dg::hdg::StridedBlockView;
using dg::hdg::core::CondensedBatchView;
using dg::hdg::core::HostReferenceEngine;
using dg::hdg::core::LocalSystemBatchView;
using dg::hdg::core::Status;

template <typename T>
StridedBlockView<T> View(T* data, LocalIndex batches, LocalIndex rows, LocalIndex cols)
{
  return {data, batches, rows, cols, rows * cols, cols, 1};
}

StridedBlockView<const double> ConstView(double* data, LocalIndex batches,
                                         LocalIndex rows, LocalIndex cols)
{
  return dg::hdg::AsConst(View(data, batches, rows, cols));
}

bool Near(double left, double right) { return std::abs(left-right) < 2.e-12; }

int TestParity()
{
  constexpr LocalIndex batches=2, ni=2, nt=2;
  std::array<double,8> a{{4.,1.,2.,3., 3.,-1.,1.,2.}};
  std::array<double,8> b{{1.,0.,0.,1., 2.,1.,1.,-1.}};
  std::array<double,8> c{{2.,-1.,1.,2., 1.,3.,-2.,1.}};
  std::array<double,8> d{{5.,.5,.5,4., 6.,1.,1.,3.}};
  std::array<double,4> f{{1.,2., 4.,-1.}}, g{{3.,4., 2.,5.}};
  std::array<double,8> schur{}, rhs{};
  LocalSystemBatchView input{{ni,nt,1,1}, ConstView(a.data(),batches,ni,ni),
      ConstView(b.data(),batches,ni,nt), ConstView(c.data(),batches,nt,ni),
      ConstView(d.data(),batches,nt,nt), ConstView(f.data(),batches,ni,1), ConstView(g.data(),batches,nt,1)};
  CondensedBatchView condensed{View(schur.data(),batches,nt,nt), View(rhs.data(),batches,nt,1)};
  HostReferenceEngine engine;
  if (engine.Condense(input,condensed) != Status::success || !engine.ready()) return 1;
  std::array<double,4> lambda{{.25,-.75, 1.5,.5}}, action{}, interior{};
  std::array<double,4> full_trace{}, full_interior{}, recovered{};
  if (engine.ApplySchur(ConstView(lambda.data(),batches,nt,1),View(action.data(),batches,nt,1)) != Status::success ||
      engine.Recover(ConstView(lambda.data(),batches,nt,1),View(interior.data(),batches,ni,1)) != Status::success) return 2;
  for (LocalIndex batch=0;batch<batches;++batch)
  {
    mfem::DenseMatrix full(ni+nt);
    mfem::Vector full_rhs(ni+nt), full_solution(ni+nt);
    for (LocalIndex row=0;row<ni;++row) {
      full_rhs(row)=f[batch*ni+row];
      for(LocalIndex col=0;col<ni;++col) full(row,col)=a[batch*ni*ni+row*ni+col];
      for(LocalIndex col=0;col<nt;++col) full(row,ni+col)=b[batch*ni*nt+row*nt+col];
    }
    for (LocalIndex row=0;row<nt;++row) {
      full_rhs(ni+row)=g[batch*nt+row];
      for(LocalIndex col=0;col<ni;++col) full(ni+row,col)=c[batch*nt*ni+row*ni+col];
      for(LocalIndex col=0;col<nt;++col) full(ni+row,ni+col)=d[batch*nt*nt+row*nt+col];
    }
    mfem::DenseMatrixInverse inverse(full); inverse.Mult(full_rhs,full_solution);
    for (LocalIndex row=0;row<nt;++row) full_trace[batch*nt+row]=full_solution(ni+row);
    for (LocalIndex row=0;row<ni;++row) full_interior[batch*ni+row]=full_solution(row);
    for(LocalIndex row=0;row<nt;++row) {
      double assembled=0.;
      for(LocalIndex col=0;col<nt;++col) assembled+=schur[batch*nt*nt+row*nt+col]*full_solution(ni+col);
      if(!Near(assembled,rhs[batch*nt+row])) return 3;
      double direct=0.;
      for(LocalIndex col=0;col<nt;++col) direct+=schur[batch*nt*nt+row*nt+col]*lambda[batch*nt+col];
      if(!Near(direct,action[batch*nt+row])) return 4;
    }
    mfem::Vector prescribed_rhs(ni);
    for(LocalIndex row=0;row<ni;++row) prescribed_rhs(row)=f[batch*ni+row];
    for(LocalIndex row=0;row<ni;++row) for(LocalIndex col=0;col<nt;++col) prescribed_rhs(row)-=b[batch*ni*nt+row*nt+col]*lambda[batch*nt+col];
    mfem::DenseMatrix local_a(ni);
    for(LocalIndex row=0;row<ni;++row) for(LocalIndex col=0;col<ni;++col) local_a(row,col)=a[batch*ni*ni+row*ni+col];
    mfem::Vector expected(ni); mfem::DenseMatrixInverse local_inverse(local_a); local_inverse.Mult(prescribed_rhs,expected);
    for(LocalIndex row=0;row<ni;++row) if(!Near(expected(row),interior[batch*ni+row])) return 5;
  }
  if(engine.Recover(ConstView(full_trace.data(),batches,nt,1),View(recovered.data(),batches,ni,1)) != Status::success) return 6;
  for(std::size_t i=0;i<recovered.size();++i) if(!Near(recovered[i],full_interior[i])) return 7;
  return 0;
}

int TestFailuresAndCensus()
{
  constexpr LocalIndex ni=2, nt=1;
  std::array<double,4> a{{1.,2.,2.,4.}}, b{{1.,1.}}, c{{1.,1.}}, d{{1.}}, f{{1.,1.}}, g{{1.}};
  std::array<double,1> sentry_s{{-9.}}, sentry_r{{-8.}};
  LocalSystemBatchView singular{{ni,nt,1,1},ConstView(a.data(),1,ni,ni),ConstView(b.data(),1,ni,nt),ConstView(c.data(),1,nt,ni),ConstView(d.data(),1,nt,nt),ConstView(f.data(),1,ni,1),ConstView(g.data(),1,nt,1)};
  HostReferenceEngine engine;
  if(engine.Condense(singular,{View(sentry_s.data(),1,nt,nt),View(sentry_r.data(),1,nt,1)}) != Status::singular || sentry_s[0]!=-9. || sentry_r[0]!=-8.) return 1;
  auto bad=ConstView(a.data(),1,ni,ni); bad.column_stride=0;
  singular.a=bad;
  if(engine.Condense(singular,{View(sentry_s.data(),1,nt,nt),View(sentry_r.data(),1,nt,1)}) != Status::invalid_view) return 2;
  singular.a=ConstView(a.data(),1,ni,ni);
  singular.f.data=nullptr;
  if(engine.Condense(singular,{View(sentry_s.data(),1,nt,nt),View(sentry_r.data(),1,nt,1)}) != Status::invalid_view) return 3;
  std::array<double,1> aa{{2.}}, bb{{1.}}, cc{{1.}}, dd{{4.}}, ff{{1.}}, gg{{1.}}, alias_rhs{{-7.}};
  LocalSystemBatchView alias{{1,1,1,1},ConstView(aa.data(),1,1,1),ConstView(bb.data(),1,1,1),ConstView(cc.data(),1,1,1),ConstView(dd.data(),1,1,1),ConstView(ff.data(),1,1,1),ConstView(gg.data(),1,1,1)};
  if(engine.Condense(alias,{View(dd.data(),1,1,1),View(alias_rhs.data(),1,1,1)}) != Status::unsupported_alias || dd[0]!=4. || alias_rhs[0]!=-7.) return 4;
  mfem::Mesh mesh=mfem::Mesh::MakeCartesian2D(1,1,mfem::Element::QUADRILATERAL,true,1.,1.);
  dg::hdg::core::ShapeCensus census;
  if(dg::hdg::core::BuildRepresentativeShapeCensus(mesh,census)!=Status::success || census.size()!=8) return 5;
  const auto p4_euler=std::find_if(census.begin(),census.end(),[](const auto& entry){return entry.polynomial_order==4 && entry.components==5;});
  return p4_euler!=census.end() && p4_euler->shape.interior_components==5 && p4_euler->shape.trace_components==5 ? 0 : 6;
}
}  // namespace

int main()
{
  const int parity=TestParity();
  return parity==0 ? TestFailuresAndCensus() : parity;
}
