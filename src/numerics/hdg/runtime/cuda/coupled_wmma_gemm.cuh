#pragma once

// Private construction primitive. Include after cuda_runtime.h and mma.h.
// The caller supplies column-major views, including views into row-major LU
// and RHS subblocks. Output may share an allocation with either input, but its
// written entries must be disjoint from every input entry used by this launch.
// Four independent warps each own one 16x16 tile; no block-wide barrier is used.
__global__ void CoupledWmmaTf32GemmKernel(
    int rows,int columns,int inner,const float* a,int lda,long long sa,
    const float* b,int ldb,long long sb,float* out,int ldc,long long so,
    int blocks_per_element,bool subtract)
{
  constexpr unsigned mask=0xffffffffu;
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int e=blockIdx.x/blocks_per_element;
  const int tiles_rows=(rows+15)/16;
  const int tile=(blockIdx.x%blocks_per_element)*4+warp;
  const int tile_col=tile/tiles_rows;
  if(tile_col>=(columns+15)/16)return; // uniform within each warp
  const int row0=(tile%tiles_rows)*16,col0=tile_col*16;
  a+=static_cast<long long>(e)*sa;
  b+=static_cast<long long>(e)*sb;
  out+=static_cast<long long>(e)*so;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // Stage the same column-major views so adjacent lanes load/store adjacent
  // rows. All fragment bases and leading dimensions satisfy WMMA alignment.
  __shared__ __align__(32) float tile_a[4][16*8];
  __shared__ __align__(32) float tile_b[4][8*16];
  __shared__ __align__(32) float tile_c[4][16*16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,8,
      nvcuda::wmma::precision::tf32,nvcuda::wmma::col_major> af;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,8,
      nvcuda::wmma::precision::tf32,nvcuda::wmma::col_major> bf;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,8,float> cf;
  nvcuda::wmma::fill_fragment(cf,0.0f);
  bool failed=false;
  for(int first=0;first<inner;first+=8)
  {
    bool bad=false;
    for(int i=lane;i<16*8;i+=32)
    {
      const int row=row0+i%16,k=first+i/16;
      float x=row<rows&&k<inner?a[row+static_cast<long long>(k)*lda]:0.0f;
      if(!isfinite(x)){bad=true;x=0.0f;}
      x=nvcuda::wmma::__float_to_tf32(x);
      if(!isfinite(x)){bad=true;x=0.0f;}
      tile_a[warp][i]=x;
    }
    for(int i=lane;i<8*16;i+=32)
    {
      const int k=first+i%8,col=col0+i/8;
      float x=k<inner&&col<columns?b[k+static_cast<long long>(col)*ldb]:0.0f;
      if(!isfinite(x)){bad=true;x=0.0f;}
      x=nvcuda::wmma::__float_to_tf32(x);
      if(!isfinite(x)){bad=true;x=0.0f;}
      tile_b[warp][i]=x;
    }
    failed=__any_sync(mask,bad)||failed;
    __syncwarp(mask);
    nvcuda::wmma::load_matrix_sync(af,tile_a[warp],16);
    nvcuda::wmma::load_matrix_sync(bf,tile_b[warp],8);
    nvcuda::wmma::mma_sync(cf,af,bf,cf);
    // All lanes finish consuming this panel before its shared storage is reused.
    __syncwarp(mask);
  }
  nvcuda::wmma::store_matrix_sync(tile_c[warp],cf,16,nvcuda::wmma::mem_col_major);
  __syncwarp(mask);
  for(int i=lane;i<16*16;i+=32)
  {
    const int row=row0+i%16,col=col0+i/16;
    if(row<rows&&col<columns)
    {
      const long long index=row+static_cast<long long>(col)*ldc;
      const float product=tile_c[warp][i];
      const float value=subtract?out[index]-product:product;
      out[index]=failed||!isfinite(value)?__int_as_float(0x7fffffff):value;
    }
  }
#else
  // Host admission rejects pre-SM80 devices. Also fail closed if an older
  // compiled image is accidentally selected on a newer device.
  for(int i=lane;i<16*16;i+=32)
  {
    const int row=row0+i%16,col=col0+i/16;
    if(row<rows&&col<columns)
      out[row+static_cast<long long>(col)*ldc]=__int_as_float(0x7fffffff);
  }
#endif
}

inline bool CoupledWmmaTf32Supported()
{
  int device=0,major=0;
  return cudaGetDevice(&device)==cudaSuccess&&
      cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)==cudaSuccess&&major>=8;
}

inline bool CoupledWmmaTf32Gemm(
    int rows,int columns,int inner,const float* a,int lda,long long sa,
    const float* b,int ldb,long long sb,float* out,int ldc,long long so,
    int elements,bool subtract)
{
  if(rows<=0||columns<=0||inner<=0||elements<=0||
     rows>2147483632||columns>2147483632||inner>2147483632||
     lda<rows||ldb<inner||ldc<rows||sa<=0||sb<=0||so<=0||!a||!b||!out)return false;
  const long long tiles=static_cast<long long>((rows+15)/16)*((columns+15)/16);
  if(tiles>2147483647LL)return false;
  const long long blocks_per_element=(tiles+3)/4;
  if(blocks_per_element>2147483647LL/elements||cudaPeekAtLastError()!=cudaSuccess)return false;
  CoupledWmmaTf32GemmKernel<<<static_cast<unsigned>(blocks_per_element*elements),128>>>(
      rows,columns,inner,a,lda,sa,b,ldb,sb,out,ldc,so,
      static_cast<int>(blocks_per_element),subtract);
  return cudaPeekAtLastError()==cudaSuccess;
}
