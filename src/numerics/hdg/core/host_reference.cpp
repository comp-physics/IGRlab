#include "numerics/hdg/core/host_reference.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
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

template <typename T>
bool HasSeparatedStrides(const StridedBlockView<T> view,
                         const std::size_t batches,
                         const std::size_t rows,
                         const std::size_t columns) noexcept
{
  struct Dimension
  {
    std::size_t count;
    std::ptrdiff_t stride;
  };
  std::array<Dimension,3> dimensions{{
      {batches,view.batch_stride},
      {rows,view.row_stride},
      {columns,view.column_stride}}};
  std::sort(dimensions.begin(),dimensions.end(),
            [](const Dimension left, const Dimension right)
            { return left.stride < right.stride; });

  std::ptrdiff_t occupied_span=0;
  for(const Dimension dimension : dimensions)
  {
    if(dimension.count<=1) continue;
    if(dimension.stride<=occupied_span) return false;
    occupied_span += static_cast<std::ptrdiff_t>(dimension.count-1) *
                     dimension.stride;
  }
  return true;
}

Status ValidateWritable(const StridedBlockView<double> view,
                        const std::size_t batches,
                        const std::size_t rows,
                        const std::size_t columns) noexcept
{
  if (!Valid(view, batches, rows, columns)) { return Status::invalid_view; }
  if (HasSeparatedStrides(view,batches,rows,columns))
  {
    return Status::success;
  }

  std::size_t entries = 0;
  std::size_t batch_entries = 0;
  if (!Multiply(rows, columns, batch_entries) ||
      !Multiply(batches, batch_entries, entries))
  {
    return Status::overflow;
  }

  try
  {
    std::vector<std::ptrdiff_t> offsets;
    if (entries > offsets.max_size()) { return Status::overflow; }
    offsets.reserve(entries);
    for (std::size_t batch = 0; batch < batches; ++batch)
    {
      for (std::size_t row = 0; row < rows; ++row)
      {
        for (std::size_t column = 0; column < columns; ++column)
        {
          offsets.push_back(
              static_cast<std::ptrdiff_t>(batch) * view.batch_stride +
              static_cast<std::ptrdiff_t>(row) * view.row_stride +
              static_cast<std::ptrdiff_t>(column) * view.column_stride);
        }
      }
    }
    std::sort(offsets.begin(), offsets.end());
    if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end())
    {
      return Status::unsupported_alias;
    }
    return Status::success;
  }
  catch (const std::bad_alloc&)
  {
    return Status::allocation_failure;
  }
  catch (const std::length_error&)
  {
    return Status::allocation_failure;
  }
}

template <typename T>
bool LogicalEntryCount(const StridedBlockView<T> view,
                       std::size_t& entries) noexcept
{
  std::size_t entries_per_batch=0;
  return Multiply(static_cast<std::size_t>(view.rows),
                  static_cast<std::size_t>(view.columns),entries_per_batch) &&
         Multiply(static_cast<std::size_t>(view.batch_count),
                  entries_per_batch,entries);
}

template <typename T>
Status AddressRange(const StridedBlockView<T> view,
                    std::uintptr_t& first,
                    std::uintptr_t& last) noexcept
{
  const std::ptrdiff_t last_offset =
      (static_cast<std::ptrdiff_t>(view.batch_count)-1) * view.batch_stride +
      (static_cast<std::ptrdiff_t>(view.rows)-1) * view.row_stride +
      (static_cast<std::ptrdiff_t>(view.columns)-1) * view.column_stride;
  first=reinterpret_cast<std::uintptr_t>(view.data);
  const std::uintptr_t maximum=std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t element_offset=static_cast<std::uintptr_t>(last_offset);
  if(element_offset>(maximum-first)/sizeof(*view.data))
  {
    return Status::overflow;
  }
  last=first+element_offset*sizeof(*view.data);
  return Status::success;
}

template <typename T, typename Function>
void ForEachAddress(const StridedBlockView<T> view, Function&& function)
{
  const std::uintptr_t base=reinterpret_cast<std::uintptr_t>(view.data);
  for(LocalIndex batch=0;batch<view.batch_count;++batch)
    for(LocalIndex row=0;row<view.rows;++row)
      for(LocalIndex column=0;column<view.columns;++column)
      {
        const std::ptrdiff_t offset=
            static_cast<std::ptrdiff_t>(batch)*view.batch_stride+
            static_cast<std::ptrdiff_t>(row)*view.row_stride+
            static_cast<std::ptrdiff_t>(column)*view.column_stride;
        function(base+static_cast<std::uintptr_t>(offset)*sizeof(*view.data));
      }
}

template <typename L, typename R>
Status ValidateDisjoint(const StridedBlockView<L> left,
                        const StridedBlockView<R> right) noexcept
{
  std::uintptr_t first_left=0,last_left=0,first_right=0,last_right=0;
  Status status=AddressRange(left,first_left,last_left);
  if(status!=Status::success) return status;
  status=AddressRange(right,first_right,last_right);
  if(status!=Status::success) return status;
  if(last_left<first_right || last_right<first_left) return Status::success;

  std::size_t left_entries=0,right_entries=0;
  if(!LogicalEntryCount(left,left_entries) ||
     !LogicalEntryCount(right,right_entries)) return Status::overflow;
  try
  {
    std::vector<std::uintptr_t> addresses;
    const std::size_t stored_entries=std::min(left_entries,right_entries);
    if(stored_entries>addresses.max_size()) return Status::overflow;
    addresses.reserve(stored_entries);
    const bool store_left=left_entries<=right_entries;
    if(store_left)
      ForEachAddress(left,[&addresses](const std::uintptr_t address)
                     { addresses.push_back(address); });
    else
      ForEachAddress(right,[&addresses](const std::uintptr_t address)
                     { addresses.push_back(address); });
    std::sort(addresses.begin(),addresses.end());

    bool overlap=false;
    const auto probe=[&addresses,&overlap](const std::uintptr_t address)
    {
      overlap=overlap || std::binary_search(
          addresses.begin(),addresses.end(),address);
    };
    if(store_left) ForEachAddress(right,probe);
    else ForEachAddress(left,probe);
    return overlap ? Status::unsupported_alias : Status::success;
  }
  catch(const std::bad_alloc&)
  {
    return Status::allocation_failure;
  }
  catch(const std::length_error&)
  {
    return Status::allocation_failure;
  }
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
  }
  for (std::size_t row = 0; row < n; ++row)
  {
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

int CodimensionOneGeometries(
    const mfem::Geometry::Type element,
    std::array<mfem::Geometry::Type, 6>& faces) noexcept
{
  using Geometry = mfem::Geometry;
  switch (element)
  {
    case Geometry::SEGMENT:
      faces[0] = Geometry::POINT;
      faces[1] = Geometry::POINT;
      return 2;
    case Geometry::TRIANGLE:
      std::fill_n(faces.begin(), 3, Geometry::SEGMENT);
      return 3;
    case Geometry::SQUARE:
      std::fill_n(faces.begin(), 4, Geometry::SEGMENT);
      return 4;
    case Geometry::TETRAHEDRON:
      std::fill_n(faces.begin(), 4, Geometry::TRIANGLE);
      return 4;
    case Geometry::CUBE:
      std::fill_n(faces.begin(), 6, Geometry::SQUARE);
      return 6;
    case Geometry::PRISM:
      for (int face = 0;
           face < Geometry::Constants<Geometry::PRISM>::NumFaces;
           ++face)
      {
        faces[static_cast<std::size_t>(face)] =
            static_cast<Geometry::Type>(
                Geometry::Constants<Geometry::PRISM>::FaceTypes[face]);
      }
      return Geometry::Constants<Geometry::PRISM>::NumFaces;
    case Geometry::PYRAMID:
      for (int face = 0;
           face < Geometry::Constants<Geometry::PYRAMID>::NumFaces;
           ++face)
      {
        faces[static_cast<std::size_t>(face)] =
            static_cast<Geometry::Type>(
                Geometry::Constants<Geometry::PYRAMID>::FaceTypes[face]);
      }
      return Geometry::Constants<Geometry::PYRAMID>::NumFaces;
    case Geometry::POINT:
    case Geometry::INVALID:
    case Geometry::NumGeom:
      return 0;
  }
  return 0;
}
}  // namespace

Status HostReferenceEngine::Prepare(const LocalBlockShape& shape,
                                    const std::size_t batches) noexcept
{
  if (!IsValidLocalBlockShape(shape) || shape.InteriorUnknowns() <= 0 ||
      shape.TraceUnknowns() <= 0) return Status::invalid_shape;
  const std::size_t ni = static_cast<std::size_t>(shape.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(shape.TraceUnknowns());
  std::size_t ni2 = 0, nint = 0, ntni = 0, nt2 = 0;
  std::size_t lu_count = 0, b_count = 0, c_count = 0, d_count = 0;
  std::size_t f_count = 0, rhs_count = 0, work_count = 0;
  if (!Multiply(ni, ni, ni2) || !Multiply(ni, nt, nint) ||
      !Multiply(nt, ni, ntni) || !Multiply(nt, nt, nt2) ||
      !Multiply(batches, ni2, lu_count) || !Multiply(batches, nint, b_count) ||
      !Multiply(batches, ntni, c_count) || !Multiply(batches, nt2, d_count) ||
      !Multiply(batches, ni, f_count) || !Multiply(batches, nt, rhs_count) ||
      !Multiply(ni, nt + 1, work_count)) return Status::overflow;
  try
  {
    std::vector<double> next_lu(lu_count), next_b(b_count), next_c(c_count),
        next_d(d_count), next_f(f_count), next_staged_lu(lu_count),
        next_staged_b(b_count), next_staged_c(c_count), next_staged_d(d_count),
        next_staged_f(f_count), next_staged_schur(d_count),
        next_staged_rhs(rhs_count), next_work(work_count), next_recovered(f_count);
    std::vector<int> next_pivots(f_count), next_staged_pivots(f_count);
    lu_.swap(next_lu); b_.swap(next_b); c_.swap(next_c); d_.swap(next_d);
    f_.swap(next_f); pivots_.swap(next_pivots);
    staged_lu_.swap(next_staged_lu); staged_b_.swap(next_staged_b);
    staged_c_.swap(next_staged_c); staged_d_.swap(next_staged_d);
    staged_f_.swap(next_staged_f); staged_pivots_.swap(next_staged_pivots);
    staged_schur_.swap(next_staged_schur); staged_rhs_.swap(next_staged_rhs);
    work_.swap(next_work); recovered_.swap(next_recovered);
  }
  catch (const std::bad_alloc&) { return Status::allocation_failure; }
  catch (const std::length_error&) { return Status::allocation_failure; }
  shape_ = shape; batch_count_ = batches; ready_ = false;
  return Status::success;
}

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
      !Valid(input.f,batches,ni,1) || !Valid(input.g,batches,nt,1))
    return Status::invalid_view;
  std::size_t ni2=0, nint=0, ntni=0, nt2=0;
  if (!Multiply(ni,ni,ni2) || !Multiply(ni,nt,nint) || !Multiply(nt,ni,ntni) ||
      !Multiply(nt,nt,nt2)) return Status::overflow;
  std::size_t lu_count=0, b_count=0, c_count=0, d_count=0, f_count=0, rhs_count=0;
  if (!Multiply(batches,ni2,lu_count) || !Multiply(batches,nint,b_count) ||
      !Multiply(batches,ntni,c_count) || !Multiply(batches,nt2,d_count) ||
      !Multiply(batches,ni,f_count) || !Multiply(batches,nt,rhs_count)) return Status::overflow;
  const Status schur_status = ValidateWritable(output.schur,batches,nt,nt);
  if (schur_status != Status::success) { return schur_status; }
  const Status rhs_status =
      ValidateWritable(output.right_hand_side,batches,nt,1);
  if (rhs_status != Status::success) { return rhs_status; }
  Status alias_status=ValidateDisjoint(output.schur,output.right_hand_side);
  if(alias_status!=Status::success) return alias_status;
  const std::array<StridedBlockView<const double>,6> inputs{{
      input.a,input.b,input.c,input.d,input.f,input.g}};
  for(const StridedBlockView<const double> read_view : inputs)
  {
    alias_status=ValidateDisjoint(output.schur,read_view);
    if(alias_status!=Status::success) return alias_status;
    alias_status=ValidateDisjoint(output.right_hand_side,read_view);
    if(alias_status!=Status::success) return alias_status;
  }
  const bool prepared = shape_.interior_dofs == input.shape.interior_dofs &&
      shape_.trace_dofs == input.shape.trace_dofs &&
      shape_.interior_components == input.shape.interior_components &&
      shape_.trace_components == input.shape.trace_components &&
      batch_count_ == batches && lu_.size() == lu_count && b_.size() == b_count &&
      c_.size() == c_count && d_.size() == d_count && f_.size() == f_count &&
      pivots_.size() == f_count && staged_lu_.size() == lu_count &&
      staged_b_.size() == b_count && staged_c_.size() == c_count &&
      staged_d_.size() == d_count && staged_f_.size() == f_count &&
      staged_pivots_.size() == f_count && staged_schur_.size() == d_count &&
      staged_rhs_.size() == rhs_count && work_.size() >= nint + ni;
  try
  {
    std::vector<double> local_lu, local_b, local_c, local_d, local_f,
        local_s, local_rhs, local_work;
    std::vector<int> local_pivots;
    std::vector<double>& new_lu = prepared ? staged_lu_ : local_lu;
    std::vector<double>& new_b = prepared ? staged_b_ : local_b;
    std::vector<double>& new_c = prepared ? staged_c_ : local_c;
    std::vector<double>& new_d = prepared ? staged_d_ : local_d;
    std::vector<double>& new_f = prepared ? staged_f_ : local_f;
    std::vector<double>& new_s = prepared ? staged_schur_ : local_s;
    std::vector<double>& new_rhs = prepared ? staged_rhs_ : local_rhs;
    std::vector<int>& new_pivots = prepared ? staged_pivots_ : local_pivots;
    std::vector<double>& workspace = prepared ? work_ : local_work;
    if (!prepared)
    {
      new_lu.resize(lu_count); new_b.resize(b_count); new_c.resize(c_count);
      new_d.resize(d_count); new_f.resize(f_count); new_s.resize(d_count);
      new_rhs.resize(rhs_count); new_pivots.resize(f_count);
      workspace.resize(nint + ni);
    }
    double* const x = workspace.data();
    double* const y = workspace.data() + nint;
    for (std::size_t batch=0; batch<batches; ++batch)
    {
      double* lu=new_lu.data()+batch*ni2; CopyIn(input.a,batch,lu);
      CopyIn(input.b,batch,new_b.data()+batch*nint); CopyIn(input.c,batch,new_c.data()+batch*ntni);
      CopyIn(input.d,batch,new_d.data()+batch*nt2); CopyIn(input.f,batch,new_f.data()+batch*ni);
      if (!AllFinite(lu,ni2) || !AllFinite(new_b.data()+batch*nint,nint) || !AllFinite(new_c.data()+batch*ntni,ntni) ||
          !AllFinite(new_d.data()+batch*nt2,nt2) || !AllFinite(new_f.data()+batch*ni,ni)) return Status::non_finite;
      for (std::size_t i=0;i<nt;++i) if (!std::isfinite(input.g(batch,static_cast<LocalIndex>(i),0))) return Status::non_finite;
      if (!Factor(lu,new_pivots.data()+batch*ni,ni)) return Status::singular;
      std::copy(new_b.begin()+static_cast<std::ptrdiff_t>(batch*nint),new_b.begin()+static_cast<std::ptrdiff_t>((batch+1)*nint),x);
      std::copy(new_f.begin()+static_cast<std::ptrdiff_t>(batch*ni),new_f.begin()+static_cast<std::ptrdiff_t>((batch+1)*ni),y);
      if (!Solve(lu,new_pivots.data()+batch*ni,ni,x,nt) || !Solve(lu,new_pivots.data()+batch*ni,ni,y,1)) return Status::non_finite;
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
    if (prepared)
    {
      std::copy(new_lu.begin(), new_lu.end(), lu_.begin());
      std::copy(new_pivots.begin(), new_pivots.end(), pivots_.begin());
      std::copy(new_b.begin(), new_b.end(), b_.begin());
      std::copy(new_c.begin(), new_c.end(), c_.begin());
      std::copy(new_d.begin(), new_d.end(), d_.begin());
      std::copy(new_f.begin(), new_f.end(), f_.begin());
    }
    else
    { shape_=input.shape; batch_count_=batches; lu_.swap(new_lu); pivots_.swap(new_pivots); b_.swap(new_b); c_.swap(new_c); d_.swap(new_d); f_.swap(new_f); }
    ready_=true;
    return Status::success;
  }
  catch (const std::bad_alloc&) { return Status::allocation_failure; }
  catch (const std::length_error&) { return Status::allocation_failure; }
}

Status HostReferenceEngine::RefreshRightHandSide(
    const StridedBlockView<const double> f,
    const StridedBlockView<const double> g,
    const StridedBlockView<double> right_hand_side) noexcept
{
  if (!ready_) return Status::not_ready;
  const std::size_t ni =
      static_cast<std::size_t>(shape_.InteriorUnknowns());
  const std::size_t nt =
      static_cast<std::size_t>(shape_.TraceUnknowns());
  if (!Valid(f, batch_count_, ni, 1) ||
      !Valid(g, batch_count_, nt, 1))
  {
    return Status::invalid_view;
  }
  const Status rhs_status =
      ValidateWritable(right_hand_side, batch_count_, nt, 1);
  if (rhs_status != Status::success) return rhs_status;
  Status alias_status = ValidateDisjoint(right_hand_side, f);
  if (alias_status != Status::success) return alias_status;
  alias_status = ValidateDisjoint(right_hand_side, g);
  if (alias_status != Status::success) return alias_status;

  if (batch_count_ > std::numeric_limits<std::size_t>::max() / ni ||
      batch_count_ > std::numeric_limits<std::size_t>::max() / nt)
  {
    return Status::overflow;
  }
  const std::size_t f_count = batch_count_ * ni;
  const std::size_t rhs_count = batch_count_ * nt;
  try
  {
    const bool prepared = staged_f_.size() == f_count &&
        staged_rhs_.size() == rhs_count && work_.size() >= ni;
    std::vector<double> local_f, local_rhs, local_work;
    std::vector<double>& next_f = prepared ? staged_f_ : local_f;
    std::vector<double>& next_rhs = prepared ? staged_rhs_ : local_rhs;
    std::vector<double>& solve = prepared ? work_ : local_work;
    if (!prepared)
    {
      next_f.resize(f_count);
      next_rhs.resize(rhs_count);
      solve.resize(ni);
    }
    for (std::size_t batch = 0; batch < batch_count_; ++batch)
    {
      CopyIn(f, batch, next_f.data() + batch * ni);
      if (!AllFinite(next_f.data() + batch * ni, ni))
        return Status::non_finite;
      std::copy(next_f.begin() + static_cast<std::ptrdiff_t>(batch * ni),
                next_f.begin() +
                    static_cast<std::ptrdiff_t>((batch + 1) * ni),
                solve.begin());
      if (!Solve(lu_.data() + batch * ni * ni,
                 pivots_.data() + batch * ni, ni, solve.data(), 1))
      {
        return Status::non_finite;
      }
      for (std::size_t row = 0; row < nt; ++row)
      {
        const double value = g(batch, static_cast<LocalIndex>(row), 0);
        if (!std::isfinite(value)) return Status::non_finite;
        double reduced = value;
        for (std::size_t column = 0; column < ni; ++column)
        {
          reduced -= c_[batch * nt * ni + row * ni + column] *
              solve[column];
        }
        if (!std::isfinite(reduced)) return Status::non_finite;
        next_rhs[batch * nt + row] = reduced;
      }
    }
    for (std::size_t batch = 0; batch < batch_count_; ++batch)
    {
      CopyOut(next_rhs.data() + batch * nt, nt, 1,
              right_hand_side, batch);
    }
    if (prepared)
      std::copy(next_f.begin(), next_f.end(), f_.begin());
    else
      f_.swap(next_f);
    return Status::success;
  }
  catch (const std::bad_alloc&) { return Status::allocation_failure; }
  catch (const std::length_error&) { return Status::allocation_failure; }
}

Status HostReferenceEngine::ApplySchur(const StridedBlockView<const double> trace,
                                       const StridedBlockView<double> result) const noexcept
{
  if (!ready_) return Status::not_ready;
  const std::size_t ni=static_cast<std::size_t>(shape_.InteriorUnknowns()), nt=static_cast<std::size_t>(shape_.TraceUnknowns());
  if (!Valid(trace,batch_count_,nt,1)) return Status::invalid_view;
  if (batch_count_ > std::numeric_limits<std::size_t>::max() / nt) return Status::overflow;
  const Status result_status = ValidateWritable(result,batch_count_,nt,1);
  if (result_status != Status::success) { return result_status; }
  try {
    const bool prepared = staged_rhs_.size() == batch_count_ * nt &&
        work_.size() >= ni;
    std::vector<double> staged, local;
    std::vector<double>& staged_values = prepared ? staged_rhs_ : staged;
    std::vector<double>& local_values = prepared ? work_ : local;
    if (!prepared) { staged_values.resize(batch_count_*nt); local_values.resize(ni); }
    for (std::size_t batch=0;batch<batch_count_;++batch) {
      for (std::size_t i=0;i<ni;++i) { local_values[i]=0.; for (std::size_t j=0;j<nt;++j) { if(!std::isfinite(trace(batch,j,0))) return Status::non_finite; local_values[i]+=b_[batch*ni*nt+i*nt+j]*trace(batch,j,0); } }
      if (!Solve(lu_.data()+batch*ni*ni,pivots_.data()+batch*ni,ni,local_values.data(),1)) return Status::non_finite;
      for (std::size_t row=0;row<nt;++row) { staged_values[batch*nt+row]=0.; for(std::size_t col=0;col<nt;++col) staged_values[batch*nt+row]+=d_[batch*nt*nt+row*nt+col]*trace(batch,col,0); for(std::size_t k=0;k<ni;++k) staged_values[batch*nt+row]-=c_[batch*nt*ni+row*ni+k]*local_values[k]; if(!std::isfinite(staged_values[batch*nt+row])) return Status::non_finite; }
    }
    for(std::size_t batch=0;batch<batch_count_;++batch) CopyOut(staged_values.data()+batch*nt,nt,1,result,batch);
    return Status::success;
  } catch(const std::bad_alloc&) { return Status::allocation_failure; }
  catch(const std::length_error&) { return Status::allocation_failure; }
}

Status HostReferenceEngine::Recover(const StridedBlockView<const double> trace,
                                    const StridedBlockView<double> interior) const noexcept
{
  if (!ready_) return Status::not_ready;
  const std::size_t ni=static_cast<std::size_t>(shape_.InteriorUnknowns()), nt=static_cast<std::size_t>(shape_.TraceUnknowns());
  if (!Valid(trace,batch_count_,nt,1)) return Status::invalid_view;
  if (batch_count_ > std::numeric_limits<std::size_t>::max() / ni) return Status::overflow;
  const Status interior_status =
      ValidateWritable(interior,batch_count_,ni,1);
  if (interior_status != Status::success) { return interior_status; }
  try {
    const bool prepared = recovered_.size() == batch_count_ * ni &&
        work_.size() >= ni;
    std::vector<double> staged, local;
    std::vector<double>& staged_values = prepared ? recovered_ : staged;
    std::vector<double>& local_values = prepared ? work_ : local;
    if (!prepared) { staged_values.resize(batch_count_*ni); local_values.resize(ni); }
    for(std::size_t batch=0;batch<batch_count_;++batch) { for(std::size_t i=0;i<ni;++i) { local_values[i]=f_[batch*ni+i]; for(std::size_t j=0;j<nt;++j) { if(!std::isfinite(trace(batch,j,0))) return Status::non_finite; local_values[i]-=b_[batch*ni*nt+i*nt+j]*trace(batch,j,0); } } if(!Solve(lu_.data()+batch*ni*ni,pivots_.data()+batch*ni,ni,local_values.data(),1)) return Status::non_finite; std::copy(local_values.begin(),local_values.begin()+static_cast<std::ptrdiff_t>(ni),staged_values.begin()+static_cast<std::ptrdiff_t>(batch*ni)); }
    for(std::size_t batch=0;batch<batch_count_;++batch) CopyOut(staged_values.data()+batch*ni,ni,1,interior,batch);
    return Status::success;
  } catch(const std::bad_alloc&) { return Status::allocation_failure; }
  catch(const std::length_error&) { return Status::allocation_failure; }
}

Status BuildRepresentativeShapeCensus(const mfem::Mesh& mesh, ShapeCensus& census) noexcept
{
  try {
    if(mesh.GetNE()<=0) return Status::invalid_shape;
    ShapeCensus next;
    for(const int components : {1,5}) for(int order=1;order<=4;++order) {
      mfem::H1_FECollection collection(order,mesh.Dimension());
      for(int element=0;element<mesh.GetNE();++element) {
        const mfem::Geometry::Type element_geometry =
            mesh.GetElementGeometry(element);
        const auto* interior =
            collection.FiniteElementForGeometry(element_geometry);
        std::array<mfem::Geometry::Type, 6> face_geometries{};
        const int face_count =
            CodimensionOneGeometries(element_geometry,face_geometries);
        if (interior == nullptr || face_count <= 0) return Status::invalid_shape;
        int trace_dofs=0;
        for(int face=0;face<face_count;++face) {
          const auto* trace = collection.FiniteElementForGeometry(
              face_geometries[static_cast<std::size_t>(face)]);
          if (trace == nullptr || trace->GetDof() <= 0 ||
              trace_dofs > std::numeric_limits<int>::max() - trace->GetDof())
            return Status::invalid_shape;
          trace_dofs += trace->GetDof();
        }
        const LocalBlockShape shape{interior->GetDof(),trace_dofs,components,components};
        if(!IsValidLocalBlockShape(shape)||shape.InteriorUnknowns()<=0||shape.TraceUnknowns()<=0) return Status::invalid_shape;
        const auto found=std::find_if(next.begin(),next.end(),[shape,order,components](const ShapeCensusEntry& e){return e.shape.interior_dofs==shape.interior_dofs&&e.shape.trace_dofs==shape.trace_dofs&&e.polynomial_order==order&&e.components==components;});
        if(found==next.end()) next.push_back({shape,order,components,1}); else ++found->element_count;
      }
    }
    census.swap(next); return Status::success;
  } catch(const std::bad_alloc&) { return Status::allocation_failure; }
  catch(const std::length_error&) { return Status::allocation_failure; }
  catch(const std::exception&) { return Status::invalid_shape; }
  catch(...) { return Status::invalid_shape; }
}
}  // namespace dg::hdg::core
