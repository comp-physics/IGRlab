#pragma once

// Internal math/action interface shared by the tensor smoother and its tests.
// This header intentionally has no CUDA or MFEM dependency for host checks.
#include <cmath>
#include <stdexcept>
#include <vector>

namespace igrlab { namespace igr_internal {

// Compose only the d*d center rows of the stored separable patch response.
// All inputs are column-major s-by-s matrices, s=3*d. The inverse is stored
// in (y eigen-index, x eigen-index) order, just as DividePatch consumes it
// after the first patch transpose. Output is column-major (d*d)-by-(s*s),
// with x-fast center rows and x-fast patch columns. Build in FP64 first;
// reduced-precision storage is a separate, checked conversion by the owner.
inline std::vector<double> ComposeTensorCenterResponse(
    int d, const std::vector<double>& forward_x,
    const std::vector<double>& forward_y, const std::vector<double>& back_x,
    const std::vector<double>& back_y, const std::vector<double>& inverse)
{
  if(d!=4 && d!=6)
    throw std::invalid_argument("Tensor center response requires d=4 or d=6.");
  const int s=3*d,n=d*d,k=s*s;
  for(const auto* values:{&forward_x,&forward_y,&back_x,&back_y,&inverse})
  {
    if(values->size()!=static_cast<std::size_t>(k))
      throw std::invalid_argument("Tensor center response factor size mismatch.");
    for(double value:*values)
      if(!std::isfinite(value))
        throw std::invalid_argument("Nonfinite tensor center response factor.");
  }
  for(double value:inverse)
    if(value<=0.0)
      throw std::invalid_argument("Tensor center response requires a positive clipped inverse.");
  std::vector<double> response(n*k,0.0);
  for(int v=0;v<s;++v) for(int u=0;u<s;++u)
    for(int b=0;b<d;++b) for(int a=0;a<d;++a)
    {
      double sum=0.0;
      for(int i=0;i<s;++i)
      {
        double y=0.0;
        for(int j=0;j<s;++j)
          y+=back_y[d+b+s*j]*inverse[j+s*i]*forward_y[j+s*v];
        sum+=back_x[d+a+s*i]*forward_x[i+s*u]*y;
      }
      if(!std::isfinite(sum))
        throw std::overflow_error("Tensor center response composition overflow.");
      response[a+d*b+n*(u+s*v)]=sum;
    }
  return response;
}

// Apply one column-major center response to ne distinct packed patch columns
// on the default CUDA stream. The scratch output is split-major:
// partial[(part*ne+element)*d*d+center_row], with split_k*ne*d*d entries.
// warps is 1 or 4; split_k is 1, 2 or 4. For SIMT, warps selects the CTA
// thread count; TF32 uses one 16-column WMMA tile per warp. Partial K panels
// are zero padded, and only actual patch columns are read or written.
void ApplyTensorCenterResponse(int ne,int d,const double* response,
    const double* packed,double* partial,int warps=1,int split_k=1);
void ApplyTensorCenterResponse(int ne,int d,const float* response,
    const float* packed,float* partial,bool tf32,int warps=1,int split_k=1);

// Reduce the partial K sums and update each element's center exactly once,
// x += 0.8*factor[element]*sum(partial). Nonfinite arithmetic poisons the
// correction so the unchanged original-current-A outer certificate rejects
// it. This remains an internal implementation/testing interface.
void AddTensorCenterResponse(int ne,int d,const double* factor,
    const double* partial,double* x,int split_k=1);
void AddTensorCenterResponse(int ne,int d,const float* factor,
    const float* partial,float* x,int split_k=1);

} }
