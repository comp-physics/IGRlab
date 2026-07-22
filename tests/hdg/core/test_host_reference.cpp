#include "numerics/hdg/core/host_reference.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

template <std::size_t Size>
bool AllEqual(const std::array<double,Size>& values, const double expected)
{
  return std::all_of(values.begin(),values.end(),
                     [expected](const double value){return value==expected;});
}

template <typename Left, typename Right>
bool LogicalNear(const StridedBlockView<Left> left,
                 const StridedBlockView<Right> right)
{
  if(left.batch_count!=right.batch_count || left.rows!=right.rows ||
     left.columns!=right.columns) return false;
  for(LocalIndex batch=0;batch<left.batch_count;++batch)
    for(LocalIndex row=0;row<left.rows;++row)
      for(LocalIndex column=0;column<left.columns;++column)
        if(!Near(left(batch,row,column),right(batch,row,column))) return false;
  return true;
}

template <std::size_t Size>
bool PaddingEqual(const std::array<double,Size>& storage,
                  const StridedBlockView<double> view,
                  const double expected)
{
  std::array<bool,Size> logical{};
  for(LocalIndex batch=0;batch<view.batch_count;++batch)
    for(LocalIndex row=0;row<view.rows;++row)
      for(LocalIndex column=0;column<view.columns;++column)
      {
        const std::ptrdiff_t offset=
            static_cast<std::ptrdiff_t>(batch)*view.batch_stride+
            static_cast<std::ptrdiff_t>(row)*view.row_stride+
            static_cast<std::ptrdiff_t>(column)*view.column_stride;
        if(offset<0 || static_cast<std::size_t>(offset)>=Size) return false;
        logical[static_cast<std::size_t>(offset)]=true;
      }
  for(std::size_t index=0;index<Size;++index)
    if(!logical[index] && storage[index]!=expected) return false;
  return true;
}

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

int TestWritableLayoutsAndOverlap()
{
  constexpr LocalIndex batches=2, ni=2, nt=2;
  std::array<double,8> a{{4.,1.,2.,3., 3.,-1.,1.,2.}};
  std::array<double,8> b{{1.,0.,0.,1., 2.,1.,1.,-1.}};
  std::array<double,8> c{{2.,-1.,1.,2., 1.,3.,-2.,1.}};
  std::array<double,8> d{{5.,.5,.5,4., 6.,1.,1.,3.}};
  std::array<double,4> f{{1.,2., 4.,-1.}}, g{{3.,4., 2.,5.}};
  LocalSystemBatchView input{{ni,nt,1,1},ConstView(a.data(),batches,ni,ni),
      ConstView(b.data(),batches,ni,nt),ConstView(c.data(),batches,nt,ni),
      ConstView(d.data(),batches,nt,nt),ConstView(f.data(),batches,ni,1),
      ConstView(g.data(),batches,nt,1)};

  std::array<double,8> compact_schur{};
  std::array<double,4> compact_rhs{};
  HostReferenceEngine compact_engine;
  if(compact_engine.Condense(input,{View(compact_schur.data(),batches,nt,nt),
      View(compact_rhs.data(),batches,nt,1)})!=Status::success) return 1;

  std::array<double,20> padded_schur{};
  std::array<double,12> padded_rhs{};
  padded_schur.fill(-101.); padded_rhs.fill(-102.);
  auto schur_view=StridedBlockView<double>{padded_schur.data(),batches,nt,nt,11,4,1};
  auto rhs_view=StridedBlockView<double>{padded_rhs.data(),batches,nt,1,6,2,1};
  HostReferenceEngine engine;
  if(engine.Condense(input,{schur_view,rhs_view})!=Status::success ||
     !LogicalNear(ConstView(compact_schur.data(),batches,nt,nt),schur_view) ||
     !LogicalNear(ConstView(compact_rhs.data(),batches,nt,1),rhs_view) ||
     !PaddingEqual(padded_schur,schur_view,-101.) ||
     !PaddingEqual(padded_rhs,rhs_view,-102.)) return 2;

  std::array<double,4> trace{{.25,-.75,1.5,.5}};
  std::array<double,12> padded_action{},padded_interior{};
  std::array<double,4> compact_action{},compact_interior{};
  padded_action.fill(-103.); padded_interior.fill(-104.);
  auto action_view=StridedBlockView<double>{padded_action.data(),batches,nt,1,6,2,1};
  auto interior_view=StridedBlockView<double>{padded_interior.data(),batches,ni,1,8,3,1};
  if(compact_engine.ApplySchur(ConstView(trace.data(),batches,nt,1),
         View(compact_action.data(),batches,nt,1))!=Status::success ||
     compact_engine.Recover(ConstView(trace.data(),batches,nt,1),
         View(compact_interior.data(),batches,ni,1))!=Status::success ||
     engine.ApplySchur(ConstView(trace.data(),batches,nt,1),action_view)!=Status::success ||
     engine.Recover(ConstView(trace.data(),batches,nt,1),interior_view)!=Status::success ||
     !LogicalNear(ConstView(compact_action.data(),batches,nt,1),action_view) ||
     !LogicalNear(ConstView(compact_interior.data(),batches,ni,1),interior_view) ||
     !PaddingEqual(padded_action,action_view,-103.) ||
     !PaddingEqual(padded_interior,interior_view,-104.)) return 3;

  std::array<double,1> interleaved_a{{2.}},interleaved_f{{1.}};
  std::array<double,2> interleaved_b{{1.,.5}},interleaved_c{{.25,-.75}},
      interleaved_g{{2.,3.}},interleaved_rhs{};
  std::array<double,8> interleaved_d_and_s{{4.,-105.,.5,-105.,.5,-105.,3.,-105.}};
  LocalSystemBatchView interleaved_input{{1,2,1,1},
      ConstView(interleaved_a.data(),1,1,1),ConstView(interleaved_b.data(),1,1,2),
      ConstView(interleaved_c.data(),1,2,1),
      {interleaved_d_and_s.data(),1,2,2,8,4,2},
      ConstView(interleaved_f.data(),1,1,1),ConstView(interleaved_g.data(),1,2,1)};
  std::array<double,4> interleaved_reference_s{};
  std::array<double,2> interleaved_reference_rhs{};
  HostReferenceEngine interleaved_reference,interleaved_engine;
  if(interleaved_reference.Condense(interleaved_input,
         {View(interleaved_reference_s.data(),1,2,2),
          View(interleaved_reference_rhs.data(),1,2,1)})!=Status::success ||
     interleaved_engine.Condense(interleaved_input,
         {{interleaved_d_and_s.data()+1,1,2,2,8,4,2},
          View(interleaved_rhs.data(),1,2,1)})!=Status::success ||
     !LogicalNear(ConstView(interleaved_reference_s.data(),1,2,2),
         StridedBlockView<double>{interleaved_d_and_s.data()+1,1,2,2,8,4,2}) ||
     !LogicalNear(ConstView(interleaved_reference_rhs.data(),1,2,1),
         View(interleaved_rhs.data(),1,2,1)) ||
     interleaved_d_and_s[0]!=4. || interleaved_d_and_s[2]!=.5 ||
     interleaved_d_and_s[4]!=.5 || interleaved_d_and_s[6]!=3.) return 4;

  std::array<double,18> bad_schur{};
  std::array<double,8> bad_rhs{};
  bad_schur.fill(-11.); bad_rhs.fill(-12.);
  auto row_column_overlap=StridedBlockView<double>{bad_schur.data(),batches,nt,nt,8,1,1};
  if(engine.Condense(input,{row_column_overlap,View(bad_rhs.data(),batches,nt,1)})!=Status::unsupported_alias ||
     !AllEqual(bad_schur,-11.) || !AllEqual(bad_rhs,-12.)) return 5;

  bad_schur.fill(-13.); bad_rhs.fill(-14.);
  auto batch_overlap=StridedBlockView<double>{bad_schur.data(),batches,nt,nt,1,4,1};
  if(engine.Condense(input,{batch_overlap,View(bad_rhs.data(),batches,nt,1)})!=Status::unsupported_alias ||
     !AllEqual(bad_schur,-13.) || !AllEqual(bad_rhs,-14.)) return 6;

  bad_schur.fill(-15.); bad_rhs.fill(-16.);
  auto rhs_overlap=StridedBlockView<double>{bad_rhs.data(),batches,nt,1,1,1,1};
  if(engine.Condense(input,{View(bad_schur.data(),batches,nt,nt),rhs_overlap})!=Status::unsupported_alias ||
     !AllEqual(bad_schur,-15.) || !AllEqual(bad_rhs,-16.)) return 7;

  std::array<double,8> action_sentinel{},interior_sentinel{};
  action_sentinel.fill(-17.); interior_sentinel.fill(-18.);
  auto action_overlap=StridedBlockView<double>{action_sentinel.data(),batches,nt,1,1,1,1};
  auto interior_overlap=StridedBlockView<double>{interior_sentinel.data(),batches,ni,1,1,1,1};
  if(engine.ApplySchur(ConstView(trace.data(),batches,nt,1),action_overlap)!=Status::unsupported_alias ||
     !AllEqual(action_sentinel,-17.)) return 8;
  if(engine.Recover(ConstView(trace.data(),batches,nt,1),interior_overlap)!=Status::unsupported_alias ||
     !AllEqual(interior_sentinel,-18.)) return 9;
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
  const LocalIndex maximum=std::numeric_limits<LocalIndex>::max();
  double storage=3.;
  const auto huge=[&storage](LocalIndex batches_,LocalIndex rows,LocalIndex columns)
  { return StridedBlockView<const double>{&storage,batches_,rows,columns,1,1,1}; };
  const auto huge_output=[&storage](LocalIndex batches_,LocalIndex rows,LocalIndex columns)
  { return StridedBlockView<double>{&storage,batches_,rows,columns,1,1,1}; };
  LocalSystemBatchView overflow{{maximum,1,1,1},huge(maximum,maximum,maximum),
      huge(maximum,maximum,1),huge(maximum,1,maximum),huge(maximum,1,1),
      huge(maximum,maximum,1),huge(maximum,1,1)};
  if(engine.Condense(overflow,{huge_output(maximum,1,1),huge_output(maximum,1,1)})!=Status::overflow || storage!=3.) return 5;

  dg::hdg::core::ShapeCensus census;
  mfem::Mesh segment=mfem::Mesh::MakeCartesian1D(1,1.);
  if(dg::hdg::core::BuildRepresentativeShapeCensus(segment,census)!=Status::success || census.size()!=8) return 6;
  const auto segment_p1=std::find_if(census.begin(),census.end(),[](const auto& entry){return entry.polynomial_order==1 && entry.components==1;});
  if(segment_p1==census.end() || segment_p1->shape.interior_dofs!=2 ||
     segment_p1->shape.trace_dofs!=2 || segment_p1->element_count!=1) return 7;

  mfem::Mesh quad=mfem::Mesh::MakeCartesian2D(
      1,1,mfem::Element::QUADRILATERAL,false,1.,1.);
  if(dg::hdg::core::BuildRepresentativeShapeCensus(quad,census)!=Status::success || census.size()!=8) return 8;
  const auto quad_p1=std::find_if(census.begin(),census.end(),[](const auto& entry){return entry.polynomial_order==1 && entry.components==1;});
  const auto quad_p4_euler=std::find_if(census.begin(),census.end(),[](const auto& entry){return entry.polynomial_order==4 && entry.components==5;});
  if(quad_p1==census.end() || quad_p1->shape.interior_dofs!=4 ||
     quad_p1->shape.trace_dofs!=8 || quad_p1->element_count!=1 ||
     quad_p4_euler==census.end() || quad_p4_euler->shape.interior_dofs!=25 ||
     quad_p4_euler->shape.trace_dofs!=20 ||
     quad_p4_euler->shape.interior_components!=5 ||
     quad_p4_euler->shape.trace_components!=5) return 9;

  mfem::Mesh triangle=mfem::Mesh::MakeCartesian2D(
      1,1,mfem::Element::TRIANGLE,false,1.,1.);
  if(dg::hdg::core::BuildRepresentativeShapeCensus(triangle,census)!=Status::success || census.size()!=8) return 10;
  const auto triangle_p1=std::find_if(census.begin(),census.end(),[](const auto& entry){return entry.polynomial_order==1 && entry.components==1;});
  if(triangle_p1==census.end() || triangle_p1->shape.interior_dofs!=3 ||
     triangle_p1->shape.trace_dofs!=6 || triangle_p1->element_count!=2) return 11;

  mfem::Mesh hex=mfem::Mesh::MakeCartesian3D(
      1,1,1,mfem::Element::HEXAHEDRON,1.,1.,1.,false);
  if(dg::hdg::core::BuildRepresentativeShapeCensus(hex,census)!=Status::success || census.size()!=8) return 12;
  const auto hex_p1=std::find_if(census.begin(),census.end(),[](const auto& entry){return entry.polynomial_order==1 && entry.components==1;});
  if(hex_p1==census.end() || hex_p1->shape.interior_dofs!=8 ||
     hex_p1->shape.trace_dofs!=24 || hex_p1->element_count!=1) return 13;

  mfem::Mesh tetrahedron=mfem::Mesh::MakeCartesian3D(
      1,1,1,mfem::Element::TETRAHEDRON,1.,1.,1.,false);
  if(dg::hdg::core::BuildRepresentativeShapeCensus(tetrahedron,census)!=Status::success || census.size()!=8) return 14;
  const auto tetrahedron_p1=std::find_if(census.begin(),census.end(),[](const auto& entry){return entry.polynomial_order==1 && entry.components==1;});
  return tetrahedron_p1!=census.end() &&
         tetrahedron_p1->shape.interior_dofs==4 &&
         tetrahedron_p1->shape.trace_dofs==12 &&
         tetrahedron_p1->element_count==6 ? 0 : 15;
}
}  // namespace

int main()
{
  const int parity=TestParity();
  if(parity!=0) return parity;
  const int writable=TestWritableLayoutsAndOverlap();
  return writable==0 ? TestFailuresAndCensus() : 20+writable;
}
