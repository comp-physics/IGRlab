#pragma once

// Internal fixed-reference algebra, also used by the CUDA fixture. There is no
// dependency on MFEM or CUDA in this header.
#include <climits>
#include <cstddef>

namespace igrlab { namespace igr_internal {

inline bool TensorSharedResidualLayoutValid(int ne,int d,int restricted_d=0)
{
  if(ne<=0 || (d!=4 && d!=6) ||
      !(restricted_d==0 || (d==6 && restricted_d==4) ||
        (d==4 && restricted_d==2))) return false;
  // Fine vectors dominate the output, neighbor-map and tile indexing bounds.
  return std::size_t(ne)<=std::size_t(INT_MAX-255)/(d*d);
}

inline std::size_t TensorSharedStencilEntries(int d)
{ return d==4 || d==6 ? std::size_t(5)*d*d*d*d : 0; }

// stencil is column-major n-by-(5*n), n=d*d, in center/left/right/bottom/top
// block order. neighbors contains five element indices per element. Every RHS
// column consists of a different element's actual five neighbors. No input may
// alias the output. With restricted_d=4 (d=6) or 2 (d=4), compute only the
// leading tensor modes and write ne*restricted_d^2 entries, fusing restriction.
void ApplyTensorSharedResidual(int ne,int d,const int* neighbors,
    const double* stencil,const double* rhs,const double* x,double* residual,
    int restricted_d=0);
void ApplyTensorSharedResidual(int ne,int d,const int* neighbors,
    const float* stencil,const float* rhs,const float* x,float* residual,
    bool tf32,int restricted_d=0);

} }
