#include "numerics/hdg/core/host_reference.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace dg::hdg::core
{
namespace
{
constexpr double pivot_scale = 64.0 * std::numeric_limits<double>::epsilon();

bool Multiply(std::size_t a, std::size_t b, std::size_t& result) noexcept
{
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) { return false; }
  result = a * b;
  return true;
}

template <typename T>
bool Valid(const StridedBlockView<T> view, std::size_t batches,
           std::size_t rows, std::size_t columns) noexcept
{
  if (view.batch_count != static_cast<LocalIndex>(batches) ||
      view.rows != static_cast<LocalIndex>(rows) ||
      view.columns != static_cast<LocalIndex>(columns) || view.data == nullptr ||
      view.batch_stride <= 0 || view.row_stride <= 0 || view.column_stride <= 0)
  {
    return false;
  }
  const auto max = std::numeric_limits<std::ptrdiff_t>::max();
  const auto fits = [max](std::size_t count, std::ptrdiff_t stride)
  { return count > 0 && count - 1 <= static_cast<std::size_t>(max / stride); };
  if (!fits(batches, view.batch_stride) || !fits(rows, view.row_stride) ||
      !fits(columns, view.column_stride)) { return false; }
  const auto batch_offset = (batches - 1) * view.batch_stride;
  const auto row_offset = (rows - 1) * view.row_stride;
  const auto column_offset = (columns - 1) * view.column_stride;
  if (batch_offset > max - row_offset ||
      column_offset > max - batch_offset - row_offset) { return false; }
  return true;
}

template <typename L, typename R>
bool Overlap(const StridedBlockView<L> left, const StridedBlockView<R> right) noexcept
{
  const auto end = [](const auto view) noexcept
  {
    const auto offset = (view.batch_count - 1) * view.batch_stride +
                        (view.rows - 1) * view.row_stride +
                        (view.columns - 1) * view.column_stride;
    return reinterpret_cast<std::uintptr_t>(view.data) +
           static_cast<std::uintptr_t>(offset) * sizeof(*view.data);
  };
  const auto first_left = reinterpret_cast<std::uintptr_t>(left.data);
  const auto first_right = reinterpret_cast<std::uintptr_t>(right.data);
  return first_left <= end(right) && first_right <= end(left);
}

bool Factor(double* lu, int* pivots, std::size_t n) noexcept
{
  double scale = 0.0;
  for (std::size_t i = 0; i < n * n; ++i)
  {
    if (!std::isfinite(lu[i])) { return false; }
    scale = std::max(scale, std::abs(lu[i]));
  }
  if (scale == 0.0) { return false; }
  for (std::size_t column = 0; column < n; ++column)
  {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < n; ++row)
    {
      if (std::abs(lu[row * n + column]) > std::abs(lu[pivot * n + column]))
      { pivot = row; }
    }
    if (std::abs(lu[pivot * n + column]) <= pivot_scale * scale) { return false; }
    pivots[column] = static_cast<int>(pivot);
    if (pivot != column)
    {
      for (std::size_t j = 0; j < n; ++j) { std::swap(lu[column*n+j], lu[pivot*n+j]); }
    }
    for (std::size_t row = column + 1; row < n; ++row)
    {
      lu[row*n+column] /= lu[column*n+column];
      for (std::size_t j = column + 1; j < n; ++j)
      { lu[row*n+j] -= lu[row*n+column] * lu[column*n+j]; }
    }
  }
  return true;
}

bool AllFinite(const double* values, std::size_t count) noexcept
{
  for (std::size_t i = 0; i < count; ++i)
  {
    if (!std::isfinite(values[i])) { return false; }
  }
  return true;
}

bool Solve(const double* lu, const int* pivots, std::size_t n,
           double* x, std::size_t nrhs) noexcept
{
  for (std::size_t row = 0; row < n; ++row)
  {
    const auto pivot = static_cast<std::size_t>(pivots[row]);
    if (pivot != row)
      for (std::size_t rhs = 0; rhs < nrhs; ++rhs) { std::swap(x[row*nrhs+rhs], x[pivot*nrhs+rhs]); }
    for (std::size_t below = row + 1; below < n; ++below)
      for (std::size_t rhs = 0; rhs < nrhs; ++rhs) { x[below*nrhs+rhs] -= lu[below*n+row] * x[row*nrhs+rhs]; }
  }
  for (std::size_t row = n; row-- > 0;)
    for (std::size_t rhs = 0; rhs < nrhs; ++rhs)
    {
      for (std::size_t col = row + 1; col < n; ++col) { x[row*nrhs+rhs] -= lu[row*n+col] * x[col*nrhs+rhs]; }
      x[row*nrhs+rhs] /= lu[row*n+row];
      if (!std::isfinite(x[row*nrhs+rhs])) { return false; }
    }
  return true;
}

template <typename T>
void CopyIn(const StridedBlockView<T> view, std::size_t batch, double* out) noexcept
{
  for (LocalIndex row = 0; row < view.rows; ++row)
    for (LocalIndex col = 0; col < view.columns; ++col)
      out[static_cast<std::size_t>(row) * view.columns + col] = view(batch, row, col);
}

void CopyOut(const double* in, std::size_t rows, std::size_t cols,
             StridedBlockView<double> view, std::size_t batch) noexcept
{
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      view(static_cast<LocalIndex>(batch), static_cast<LocalIndex>(row),
           static_cast<LocalIndex>(col)) = in[row * cols + col];
}
}  // namespace

Status HostReferenceEngine::Condense(const LocalSystemBatchView& input,
                                     const CondensedBatchView& output) noexcept
{
  if (!IsValidLocalBlockShape(input.shape) || input.shape.InteriorUnknowns() <= 0 ||
      input.shape.TraceUnknowns() <= 0 || input.a.batch_count <= 0) { return Status::invalid_shape; }
  const std::size_t ni = static_cast<std::size_t>(input.shape.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(input.shape.TraceUnknowns());
  const std::size_t batches = static_cast<std::size_t>(input.a.batch_count);
  if (!Valid(input.a,batches,ni,ni) || !Valid(input.b,batches,ni,nt) ||
      !Valid(input.c,batches,nt,ni) || !Valid(input.d,batches,nt,nt) ||
      !Valid(input.f,batches,ni,1) || !Valid(input.g,batches,nt,1) ||
      !Valid(output.schur,batches,nt,nt) || !Valid(output.right_hand_side,batches,nt,1))
    return Status::invalid_view;
  if (Overlap(output.schur,output.right_hand_side) || Overlap(output.schur,input.a) ||
      Overlap(output.schur,input.b) || Overlap(output.schur,input.c) || Overlap(output.schur,input.d) ||
      Overlap(output.schur,input.f) || Overlap(output.schur,input.g) || Overlap(output.right_hand_side,input.a) ||
      Overlap(output.right_hand_side,input.b) || Overlap(output.right_hand_side,input.c) ||
      Overlap(output.right_hand_side,input.d) || Overlap(output.right_hand_side,input.f) ||
      Overlap(output.right_hand_side,input.g)) return Status::unsupported_alias;
  std::size_t ni2=0, nint=0, ntni=0, nt2=0;
  if (!Multiply(ni,ni,ni2) || !Multiply(ni,nt,nint) || !Multiply(nt,ni,ntni) ||
      !Multiply(nt,nt,nt2)) return Status::overflow;
  std::size_t lu_count=0, b_count=0, c_count=0, d_count=0, f_count=0, rhs_count=0;
  if (!Multiply(batches,ni2,lu_count) || !Multiply(batches,nint,b_count) ||
      !Multiply(batches,ntni,c_count) || !Multiply(batches,nt2,d_count) ||
      !Multiply(batches,ni,f_count) || !Multiply(batches,nt,rhs_count)) return Status::overflow;
  try
  {
    std::vector<double> new_lu(lu_count), new_b(b_count), new_c(c_count),
        new_d(d_count), new_f(f_count), new_s(d_count), new_rhs(rhs_count), x(nint), y(ni);
    std::vector<int> new_pivots(f_count);
    for (std::size_t batch=0; batch<batches; ++batch)
    {
      double* lu=new_lu.data()+batch*ni2; CopyIn(input.a,batch,lu);
      CopyIn(input.b,batch,new_b.data()+batch*nint); CopyIn(input.c,batch,new_c.data()+batch*ntni);
      CopyIn(input.d,batch,new_d.data()+batch*nt2); CopyIn(input.f,batch,new_f.data()+batch*ni);
      if (!AllFinite(lu,ni2) || !AllFinite(new_b.data()+batch*nint,nint) || !AllFinite(new_c.data()+batch*ntni,ntni) ||
          !AllFinite(new_d.data()+batch*nt2,nt2) || !AllFinite(new_f.data()+batch*ni,ni)) return Status::non_finite;
      for (std::size_t i=0;i<nt;++i) if (!std::isfinite(input.g(batch,static_cast<LocalIndex>(i),0))) return Status::non_finite;
      if (!Factor(lu,new_pivots.data()+batch*ni,ni)) return Status::singular;
      std::copy(new_b.begin()+static_cast<std::ptrdiff_t>(batch*nint),new_b.begin()+static_cast<std::ptrdiff_t>((batch+1)*nint),x.begin());
      std::copy(new_f.begin()+static_cast<std::ptrdiff_t>(batch*ni),new_f.begin()+static_cast<std::ptrdiff_t>((batch+1)*ni),y.begin());
      if (!Solve(lu,new_pivots.data()+batch*ni,ni,x.data(),nt) || !Solve(lu,new_pivots.data()+batch*ni,ni,y.data(),1)) return Status::non_finite;
      double* s=new_s.data()+batch*nt2; double* rhs=new_rhs.data()+batch*nt;
      for (std::size_t row=0;row<nt;++row)
      {
        rhs[row]=input.g(batch,static_cast<LocalIndex>(row),0);
        for (std::size_t k=0;k<ni;++k) rhs[row]-=new_c[batch*ntni+row*ni+k]*y[k];
        for (std::size_t col=0;col<nt;++col)
        {
          s[row*nt+col]=new_d[batch*nt2+row*nt+col];
          for (std::size_t k=0;k<ni;++k) s[row*nt+col]-=new_c[batch*ntni+row*ni+k]*x[k*nt+col];
          if (!std::isfinite(s[row*nt+col])) return Status::non_finite;
        }
        if (!std::isfinite(rhs[row])) return Status::non_finite;
      }
    }
    for (std::size_t batch=0;batch<batches;++batch) { CopyOut(new_s.data()+batch*nt2,nt,nt,output.schur,batch); CopyOut(new_rhs.data()+batch*nt,nt,1,output.right_hand_side,batch); }
    shape_=input.shape; batch_count_=batches; lu_.swap(new_lu); pivots_.swap(new_pivots); b_.swap(new_b); c_.swap(new_c); d_.swap(new_d); f_.swap(new_f); ready_=true;
    return Status::success;
  }
  catch (const std::bad_alloc&) { return Status::allocation_failure; }
}

Status HostReferenceEngine::ApplySchur(const StridedBlockView<const double> trace,
                                       const StridedBlockView<double> result) const noexcept
{
  if (!ready_) return Status::not_ready;
  const std::size_t ni=static_cast<std::size_t>(shape_.InteriorUnknowns()), nt=static_cast<std::size_t>(shape_.TraceUnknowns());
  if (!Valid(trace,batch_count_,nt,1) || !Valid(result,batch_count_,nt,1)) return Status::invalid_view;
  if (batch_count_ > std::numeric_limits<std::size_t>::max() / nt) return Status::overflow;
  try {
    std::vector<double> staged(batch_count_*nt), local(ni);
    for (std::size_t batch=0;batch<batch_count_;++batch) {
      for (std::size_t i=0;i<ni;++i) { local[i]=0.; for (std::size_t j=0;j<nt;++j) { if(!std::isfinite(trace(batch,j,0))) return Status::non_finite; local[i]+=b_[batch*ni*nt+i*nt+j]*trace(batch,j,0); } }
      if (!Solve(lu_.data()+batch*ni*ni,pivots_.data()+batch*ni,ni,local.data(),1)) return Status::non_finite;
      for (std::size_t row=0;row<nt;++row) { staged[batch*nt+row]=0.; for(std::size_t col=0;col<nt;++col) staged[batch*nt+row]+=d_[batch*nt*nt+row*nt+col]*trace(batch,col,0); for(std::size_t k=0;k<ni;++k) staged[batch*nt+row]-=c_[batch*nt*ni+row*ni+k]*local[k]; if(!std::isfinite(staged[batch*nt+row])) return Status::non_finite; }
    }
    for(std::size_t batch=0;batch<batch_count_;++batch) CopyOut(staged.data()+batch*nt,nt,1,result,batch);
    return Status::success;
  } catch(const std::bad_alloc&) { return Status::allocation_failure; }
}

Status HostReferenceEngine::Recover(const StridedBlockView<const double> trace,
                                    const StridedBlockView<double> interior) const noexcept
{
  if (!ready_) return Status::not_ready;
  const std::size_t ni=static_cast<std::size_t>(shape_.InteriorUnknowns()), nt=static_cast<std::size_t>(shape_.TraceUnknowns());
  if (!Valid(trace,batch_count_,nt,1) || !Valid(interior,batch_count_,ni,1)) return Status::invalid_view;
  if (batch_count_ > std::numeric_limits<std::size_t>::max() / ni) return Status::overflow;
  try {
    std::vector<double> staged(batch_count_*ni), local(ni);
    for(std::size_t batch=0;batch<batch_count_;++batch) { for(std::size_t i=0;i<ni;++i) { local[i]=f_[batch*ni+i]; for(std::size_t j=0;j<nt;++j) { if(!std::isfinite(trace(batch,j,0))) return Status::non_finite; local[i]-=b_[batch*ni*nt+i*nt+j]*trace(batch,j,0); } } if(!Solve(lu_.data()+batch*ni*ni,pivots_.data()+batch*ni,ni,local.data(),1)) return Status::non_finite; std::copy(local.begin(),local.end(),staged.begin()+static_cast<std::ptrdiff_t>(batch*ni)); }
    for(std::size_t batch=0;batch<batch_count_;++batch) CopyOut(staged.data()+batch*ni,ni,1,interior,batch);
    return Status::success;
  } catch(const std::bad_alloc&) { return Status::allocation_failure; }
}

Status BuildRepresentativeShapeCensus(const mfem::Mesh& mesh, ShapeCensus& census) noexcept
{
  try {
    if(mesh.GetNE()<=0) return Status::invalid_shape;
    ShapeCensus next;
    for(const int components : {1,5}) for(int order=1;order<=4;++order) {
      mfem::H1_FECollection collection(order,mesh.Dimension());
      for(int element=0;element<mesh.GetNE();++element) {
        const auto* interior=collection.FiniteElementForGeometry(mesh.GetElementGeometry(element));
        mfem::Array<int> faces,orientations; mesh.GetElementFaces(element,faces,orientations); int trace_dofs=0;
        for(int face=0;face<faces.Size();++face) trace_dofs+=collection.DofForGeometry(mesh.GetFaceGeometry(faces[face]));
        const LocalBlockShape shape{interior->GetDof(),trace_dofs,components,components};
        if(!IsValidLocalBlockShape(shape)||shape.InteriorUnknowns()<=0||shape.TraceUnknowns()<=0) return Status::invalid_shape;
        const auto found=std::find_if(next.begin(),next.end(),[shape,order,components](const ShapeCensusEntry& e){return e.shape.interior_dofs==shape.interior_dofs&&e.shape.trace_dofs==shape.trace_dofs&&e.polynomial_order==order&&e.components==components;});
        if(found==next.end()) next.push_back({shape,order,components,1}); else ++found->element_count;
      }
    }
    census.swap(next); return Status::success;
  } catch(const std::bad_alloc&) { return Status::allocation_failure; }
}
}  // namespace dg::hdg::core
