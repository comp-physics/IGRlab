// Private kernels included inside IGR_local_tensor_jacobi.cu's anonymous namespace.
// Every grouped column belongs to a different element in one verified factor
// class. Padding never replicates a residual, and no packed global scratch is used.
constexpr int tensor_group_elements = 16;
constexpr int tensor_axis = 6;
constexpr int tensor_dofs = tensor_axis * tensor_axis;

// The actual quadrature mass diagonal is not collocated on Q2 geometry.
// Contract squared one-dimensional shapes with w det(J) invrho in x and y;
// the coefficient is refreshed on the operator's mass rule, not at GL6 nodes.
__global__ void RefreshProjectedTensorMass(int qaxis, const double* basis_squared,
                                          const double* geometry, const double* invrho,
                                          double* inverse_sqrt_mass)
{
  const int e = blockIdx.x, t = threadIdx.x;
  const int offset = e * qaxis * qaxis;
  __shared__ double intermediate[max_dofs];
  if (t < tensor_axis * qaxis)
  {
    const int x = t % tensor_axis, qy = t / tensor_axis;
    double value = 0.0;
    for (int qx = 0; qx < qaxis; ++qx)
    {
      const int q = offset + qx + qaxis * qy;
      value += basis_squared[qx + qaxis * x] * geometry[q] * invrho[q];
    }
    intermediate[t] = value;
  }
  __syncthreads();
  if (t < tensor_dofs)
  {
    const int x = t % tensor_axis, y = t / tensor_axis;
    double mass = 0.0;
    for (int qy = 0; qy < qaxis; ++qy)
    { mass += intermediate[x + tensor_axis * qy] * basis_squared[qy + qaxis * y]; }
    inverse_sqrt_mass[e * tensor_dofs + t] = 1.0 / sqrt(mass);
  }
}

// Setup only: form H_e Q with two directional contractions, then its Rayleigh
// diagonals. There is one shared basis per physical-boundary mask, but each
// element keeps its own positive reciprocal diagonal. Full dense blocks are
// released after this kernel and the setup checks; none are used by Mult.
__global__ void ProjectTensorDiagonals(const int* native, const int* classes,
                                      const double* blocks, const double* scaling,
                                      const double* factors, double* reciprocals,
                                      double* relative_offdiag)
{
  const int e = blockIdx.x, t = threadIdx.x;
  const double* qx = factors + std::size_t(classes[e]) * 2 * tensor_dofs;
  const double* qy = qx + tensor_dofs;
  const double* block = blocks + std::size_t(e) * tensor_dofs * tensor_dofs;
  const double* scale = scaling + e * tensor_dofs;
  __shared__ double u[tensor_dofs * tensor_dofs], v[tensor_dofs * tensor_dofs];
  __shared__ double norms[128];
  __shared__ double diagonal[tensor_dofs];
  double maximum = 0.0;
  for (int item = t; item < tensor_dofs * tensor_dofs; item += blockDim.x)
  {
    const int row = item % tensor_dofs, col = item / tensor_dofs;
    const double a = scale[row] * block[native[row] + tensor_dofs * native[col]] * scale[col];
    const double b = scale[col] * block[native[col] + tensor_dofs * native[row]] * scale[row];
    const double value = row == col ? a : 0.5 * (a + b);
    u[item] = value;
    maximum = fmax(maximum, fabs(value));
  }
  norms[t] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (t < stride) { norms[t] = fmax(norms[t], norms[t + stride]); }
    __syncthreads();
  }
  maximum = norms[0];
  __syncthreads();
  double norm_squared = 0.0;
  for (int item = t; item < tensor_dofs * tensor_dofs; item += blockDim.x)
  {
    const double normalized = u[item] / maximum;
    norm_squared += normalized * normalized;
    const int row = item % tensor_dofs, col = item / tensor_dofs;
    double value = 0.0;
    for (int k = 0; k < tensor_axis; ++k)
    { value += u[row + tensor_dofs * (k + tensor_axis * (col / tensor_axis))] *
               qx[k + tensor_axis * (col % tensor_axis)]; }
    v[item] = value;
  }
  norms[t] = norm_squared;
  __syncthreads();
  for (int item = t; item < tensor_dofs * tensor_dofs; item += blockDim.x)
  {
    const int row = item % tensor_dofs, col = item / tensor_dofs;
    double value = 0.0;
    for (int k = 0; k < tensor_axis; ++k)
    { value += v[row + tensor_dofs * (col % tensor_axis + tensor_axis * k)] *
               qy[k + tensor_axis * (col / tensor_axis)]; }
    u[item] = value;
  }
  __syncthreads();
  if (t < tensor_dofs)
  {
    double value = 0.0;
    for (int row = 0; row < tensor_dofs; ++row)
    { value += qx[row % tensor_axis + tensor_axis * (t % tensor_axis)] *
               qy[row / tensor_axis + tensor_axis * (t / tensor_axis)] *
               u[row + tensor_dofs * t]; }
    reciprocals[e * tensor_dofs + t] = 1.0 / value;
    diagonal[t] = value / maximum;
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (t < stride) { norms[t] += norms[t + stride]; }
    __syncthreads();
  }
  if (t == 0)
  {
    double diagonal_squared = 0.0;
    for (int i = 0; i < tensor_dofs; ++i) { diagonal_squared += diagonal[i] * diagonal[i]; }
    // Orthogonal invariance gives ||offdiag(Q^T H Q)||_F^2 =
    // ||H||_F^2 - ||diag(Q^T H Q)||_2^2. Clamp cancellation at roundoff.
    relative_offdiag[e] = sqrt(fmax(0.0, norms[0] - diagonal_squared) / norms[0]);
  }
}

// Precision control: retain FP64 current mass state, gather and final scaling,
// while performing the same four directional contractions in FP32.
__global__ void ApplyTensorInverseFP32(int axis, const int* map, const int* classes,
                                     const double* factors, const double* element_diagonals,
                                     const double* scaling,
                                     const double* input, double* output)
{
  const int e = blockIdx.x, t = threadIdx.x, dof = axis * axis;
  const int x = t % axis, y = t / axis;
  const double* qx = factors + std::size_t(classes[e]) * (element_diagonals ? 2 : 3) * dof;
  const double* qy = qx + dof;
  const double* denominator = element_diagonals ? element_diagonals + std::size_t(e) * dof : qy + dof;
  __shared__ float u[max_dofs], v[max_dofs];
  if (t < dof)
  { u[t] = static_cast<float>(scaling[e * dof + t] * input[map[e * dof + t]]); }
  __syncthreads();
  if (t < dof)
  {
    float value = 0.0f;
    for (int i = 0; i < axis; ++i)
    { value += static_cast<float>(qx[i + axis * x]) * u[i + axis * y]; }
    v[t] = value;
  }
  __syncthreads();
  if (t < dof)
  {
    float value = 0.0f;
    for (int j = 0; j < axis; ++j)
    { value += static_cast<float>(qy[j + axis * y]) * v[x + axis * j]; }
    u[t] = value * static_cast<float>(denominator[t]);
  }
  __syncthreads();
  if (t < dof)
  {
    float value = 0.0f;
    for (int i = 0; i < axis; ++i)
    { value += static_cast<float>(qx[x + axis * i]) * u[i + axis * y]; }
    v[t] = value;
  }
  __syncthreads();
  if (t < dof)
  {
    float value = 0.0f;
    for (int j = 0; j < axis; ++j)
    { value += static_cast<float>(qy[y + axis * j]) * v[x + axis * j]; }
    output[map[e * dof + t]] = scaling[e * dof + t] * static_cast<double>(value);
  }
}

// The shared class-dense representation and 16-element gather/scatter match the
// dense WMMA candidates. Ordinary FP32 dot products isolate the effects of
// representation/grouping from the tensor-core arithmetic.
__global__ void ApplyTensorDenseFP32(const int* map, const int* group_elements,
                                    const int* group_classes, const double* inverses,
                                    const double* scaling, const double* input,
                                    double* output)
{
  const int* elements = group_elements + blockIdx.x * tensor_group_elements;
  const double* inverse = inverses + std::size_t(group_classes[blockIdx.x]) * tensor_dofs * tensor_dofs;
  __shared__ float a[tensor_dofs * tensor_dofs];
  __shared__ float b[tensor_dofs * tensor_group_elements];
  for (int i = threadIdx.x; i < tensor_dofs * tensor_dofs; i += blockDim.x)
  { a[i] = static_cast<float>(inverse[i]); }
  for (int i = threadIdx.x; i < tensor_dofs * tensor_group_elements; i += blockDim.x)
  {
    const int row = i / tensor_group_elements, e = elements[i % tensor_group_elements];
    float value = 0.0f;
    if (e >= 0)
    {
      const int entry = e * tensor_dofs + row;
      value = static_cast<float>(scaling[entry] * input[map[entry]]);
    }
    b[i] = value;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < tensor_dofs * tensor_group_elements; i += blockDim.x)
  {
    const int row = i / tensor_group_elements, slot = i % tensor_group_elements;
    const int e = elements[slot];
    if (e >= 0)
    {
      float value = 0.0f;
      for (int k = 0; k < tensor_dofs; ++k)
      { value += a[row * tensor_dofs + k] * b[k * tensor_group_elements + slot]; }
      const int entry = e * tensor_dofs + row;
      output[map[entry]] = scaling[entry] * static_cast<double>(value);
    }
  }
}

// One CTA holds 16 elements and their 96 real nodal-line columns. Each of six
// warps multiplies the shared 6x6 axis by 16 distinct lines (16x16x8 TF32 WMMA
// with zero-padded rows/inner dimension). Repacking between the four transforms
// stays in shared memory, including the reciprocal eigensum multiplication.
__global__ void ApplyTensorFactorTF32(const int* map, const int* group_elements,
                                     const int* group_classes, const double* factors,
                                     const double* element_diagonals,
                                     const double* scaling, const double* input,
                                     double* output)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  constexpr int columns = tensor_group_elements * tensor_axis;
  const int warp = threadIdx.x / 32;
  const double* qx = factors + std::size_t(group_classes[blockIdx.x]) *
                     (element_diagonals ? 2 : 3) * tensor_dofs;
  const double* qy = qx + tensor_dofs;
  const double* denominator = qy + tensor_dofs;
  const int* elements = group_elements + blockIdx.x * tensor_group_elements;
  __shared__ __align__(32) float a[4][16 * 8];
  __shared__ __align__(32) float u[8 * columns], v[16 * columns];
  for (int i = threadIdx.x; i < 4 * 16 * 8; i += blockDim.x)
  {
    const int phase = i / (16 * 8), item = i % (16 * 8);
    const int row = item / 8, k = item % 8;
    float value = 0.0f;
    if (row < tensor_axis && k < tensor_axis)
    {
      const double* q = phase % 2 == 0 ? qx : qy;
      value = static_cast<float>(phase < 2 ? q[k + tensor_axis * row]
                                                         : q[row + tensor_axis * k]);
    }
    a[phase][item] = wmma::__float_to_tf32(value);
  }
  for (int i = threadIdx.x; i < 8 * columns; i += blockDim.x)
  {
    const int row = i / columns, column = i % columns;
    const int e = elements[column / tensor_axis], y = column % tensor_axis;
    float value = 0.0f;
    if (row < tensor_axis && e >= 0)
    {
      const int t = e * tensor_dofs + row + tensor_axis * y;
      value = static_cast<float>(scaling[t] * input[map[t]]);
    }
    u[i] = wmma::__float_to_tf32(value);
  }
  __syncthreads();
  for (int phase = 0; phase < 4; ++phase)
  {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major> left;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major> right;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    wmma::load_matrix_sync(left, a[phase], 8);
    wmma::load_matrix_sync(right, u + 16 * warp, columns);
    wmma::mma_sync(accumulator, left, right, accumulator);
    wmma::store_matrix_sync(v + 16 * warp, accumulator, columns, wmma::mem_row_major);
    __syncthreads();
    if (phase < 3)
    {
      for (int i = threadIdx.x; i < 8 * columns; i += blockDim.x)
      {
        const int row = i / columns, column = i % columns;
        const int slot = column / tensor_axis, line = column % tensor_axis;
        float value = 0.0f;
        if (row < tensor_axis)
        {
          value = v[line * columns + slot * tensor_axis + row];
          if (phase == 1)
          {
            const int e = elements[slot];
            const double* diagonal = element_diagonals && e >= 0 ?
              element_diagonals + std::size_t(e) * tensor_dofs : denominator;
            value *= e >= 0 ? static_cast<float>(diagonal[row + tensor_axis * line]) : 0.0f;
          }
        }
        u[i] = wmma::__float_to_tf32(value);
      }
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < tensor_group_elements * tensor_dofs; i += blockDim.x)
  {
    const int slot = i / tensor_dofs, t = i % tensor_dofs, e = elements[slot];
    if (e >= 0)
    {
      const float value = v[(t / tensor_axis) * columns + slot * tensor_axis + t % tensor_axis];
      const int entry = e * tensor_dofs + t;
      output[map[entry]] = scaling[entry] * static_cast<double>(value);
    }
  }
#else
  asm("trap;");
#endif
}

// Six warps each handle 16 of the group's 96 nodal-line columns. Native
// FP64 8x8x4 WMMA performs each padded 6x6 transform as two K tiles and two
// 8-column accumulators. Repacking and each element's reciprocal diagonal
// remain in shared memory throughout the four transforms.
__global__ void ApplyTensorFactorFP64(const int* map, const int* group_elements,
                                     const int* group_classes, const double* factors,
                                     const double* element_diagonals,
                                     const double* scaling, const double* input,
                                     double* output)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  constexpr int columns = tensor_group_elements * tensor_axis;
  const int warp = threadIdx.x / 32;
  const double* qx = factors + std::size_t(group_classes[blockIdx.x]) *
                     (element_diagonals ? 2 : 3) * tensor_dofs;
  const double* qy = qx + tensor_dofs;
  const double* denominator = qy + tensor_dofs;
  const int* elements = group_elements + blockIdx.x * tensor_group_elements;
  __shared__ __align__(32) double a[4][8 * 8];
  __shared__ __align__(32) double u[8 * columns], v[8 * columns];
  for (int i = threadIdx.x; i < 4 * 8 * 8; i += blockDim.x)
  {
    const int phase = i / (8 * 8), item = i % (8 * 8);
    const int row = item / 8, k = item % 8;
    double value = 0.0;
    if (row < tensor_axis && k < tensor_axis)
    {
      const double* q = phase % 2 == 0 ? qx : qy;
      value = phase < 2 ? q[k + tensor_axis * row] : q[row + tensor_axis * k];
    }
    a[phase][item] = value;
  }
  for (int i = threadIdx.x; i < 8 * columns; i += blockDim.x)
  {
    const int row = i / columns, column = i % columns;
    const int e = elements[column / tensor_axis], y = column % tensor_axis;
    double value = 0.0;
    if (row < tensor_axis && e >= 0)
    {
      const int t = e * tensor_dofs + row + tensor_axis * y;
      value = scaling[t] * input[map[t]];
    }
    u[i] = value;
  }
  __syncthreads();
  for (int phase = 0; phase < 4; ++phase)
  {
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> accumulator0, accumulator1;
    wmma::fill_fragment(accumulator0, 0.0);
    wmma::fill_fragment(accumulator1, 0.0);
    for (int k0 = 0; k0 < 8; k0 += 4)
    {
      wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> left;
      wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> right0, right1;
      wmma::load_matrix_sync(left, a[phase] + k0, 8);
      wmma::load_matrix_sync(right0, u + k0 * columns + 16 * warp, columns);
      wmma::load_matrix_sync(right1, u + k0 * columns + 16 * warp + 8, columns);
      wmma::mma_sync(accumulator0, left, right0, accumulator0);
      wmma::mma_sync(accumulator1, left, right1, accumulator1);
    }
    wmma::store_matrix_sync(v + 16 * warp, accumulator0, columns, wmma::mem_row_major);
    wmma::store_matrix_sync(v + 16 * warp + 8, accumulator1, columns, wmma::mem_row_major);
    __syncthreads();
    if (phase < 3)
    {
      for (int i = threadIdx.x; i < 8 * columns; i += blockDim.x)
      {
        const int row = i / columns, column = i % columns;
        const int slot = column / tensor_axis, line = column % tensor_axis;
        double value = 0.0;
        if (row < tensor_axis)
        {
          value = v[line * columns + slot * tensor_axis + row];
          if (phase == 1)
          {
            const int e = elements[slot];
            const double* diagonal = element_diagonals && e >= 0 ?
              element_diagonals + std::size_t(e) * tensor_dofs : denominator;
            value *= e >= 0 ? diagonal[row + tensor_axis * line] : 0.0;
          }
        }
        u[i] = value;
      }
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < tensor_group_elements * tensor_dofs; i += blockDim.x)
  {
    const int slot = i / tensor_dofs, t = i % tensor_dofs, e = elements[slot];
    if (e >= 0)
    {
      const double value = v[(t / tensor_axis) * columns + slot * tensor_axis + t % tensor_axis];
      const int entry = e * tensor_dofs + t;
      output[map[entry]] = scaling[entry] * value;
    }
  }
#else
  asm("trap;");
#endif
}

// Three row warps multiply one cached 36x36 class inverse by 16 current,
// distinct residual columns. Inner padding is 40, output-row padding is 48.
__global__ void ApplyTensorDenseTF32(const int* map, const int* group_elements,
                                    const int* group_classes, const double* inverses,
                                    const double* scaling, const double* input,
                                    double* output)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int* elements = group_elements + blockIdx.x * tensor_group_elements;
  const double* inverse = inverses + std::size_t(group_classes[blockIdx.x]) * tensor_dofs * tensor_dofs;
  __shared__ __align__(32) float a[3][16 * 8];
  __shared__ __align__(32) float b[40 * tensor_group_elements];
  __shared__ __align__(32) float c[48 * tensor_group_elements];
  for (int i = threadIdx.x; i < 40 * tensor_group_elements; i += blockDim.x)
  {
    const int row = i / tensor_group_elements, e = elements[i % tensor_group_elements];
    float value = 0.0f;
    if (row < tensor_dofs && e >= 0)
    {
      const int entry = e * tensor_dofs + row;
      value = static_cast<float>(scaling[entry] * input[map[entry]]);
    }
    b[i] = wmma::__float_to_tf32(value);
  }
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int k0 = 0; k0 < tensor_dofs; k0 += 8)
  {
    for (int i = lane; i < 16 * 8; i += 32)
    {
      const int row = 16 * warp + i / 8, k = k0 + i % 8;
      a[warp][i] = wmma::__float_to_tf32(row < tensor_dofs && k < tensor_dofs ?
        static_cast<float>(inverse[row * tensor_dofs + k]) : 0.0f);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major> left;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major> right;
    wmma::load_matrix_sync(left, a[warp], 8);
    wmma::load_matrix_sync(right, b + k0 * tensor_group_elements, tensor_group_elements);
    wmma::mma_sync(accumulator, left, right, accumulator);
    __syncwarp();
  }
  wmma::store_matrix_sync(c + 16 * warp * tensor_group_elements, accumulator,
                           tensor_group_elements, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 16 * tensor_group_elements; i += 32)
  {
    const int row = 16 * warp + i / tensor_group_elements;
    const int e = elements[i % tensor_group_elements];
    if (row < tensor_dofs && e >= 0)
    {
      const int entry = e * tensor_dofs + row;
      output[map[entry]] = scaling[entry] * static_cast<double>(c[row * tensor_group_elements + i % tensor_group_elements]);
    }
  }
#else
  asm("trap;");
#endif
}

// Five row warps use native FP64 8x8x4 WMMA. Each warp owns two 8-column
// accumulators, so one matrix tile serves all 16 current element residuals.
// Only output rows are padded (to 40); K=36 needs no inner padding.
__global__ void ApplyTensorDenseFP64(const int* map, const int* group_elements,
                                    const int* group_classes, const double* inverses,
                                    const double* scaling, const double* input,
                                    double* output)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int* elements = group_elements + blockIdx.x * tensor_group_elements;
  const double* inverse = inverses + std::size_t(group_classes[blockIdx.x]) * tensor_dofs * tensor_dofs;
  __shared__ __align__(32) double a[5][8 * 4];
  __shared__ __align__(32) double b[tensor_dofs * tensor_group_elements];
  __shared__ __align__(32) double c[40 * tensor_group_elements];
  for (int i = threadIdx.x; i < tensor_dofs * tensor_group_elements; i += blockDim.x)
  {
    const int row = i / tensor_group_elements, e = elements[i % tensor_group_elements];
    double value = 0.0;
    if (e >= 0)
    {
      const int entry = e * tensor_dofs + row;
      value = scaling[entry] * input[map[entry]];
    }
    b[i] = value;
  }
  __syncthreads();
  wmma::fragment<wmma::accumulator, 8, 8, 4, double> accumulator0, accumulator1;
  wmma::fill_fragment(accumulator0, 0.0);
  wmma::fill_fragment(accumulator1, 0.0);
  for (int k0 = 0; k0 < tensor_dofs; k0 += 4)
  {
    const int row = 8 * warp + lane / 4, k = k0 + lane % 4;
    a[warp][lane] = row < tensor_dofs ? inverse[row * tensor_dofs + k] : 0.0;
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> left;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> right0, right1;
    wmma::load_matrix_sync(left, a[warp], 4);
    wmma::load_matrix_sync(right0, b + k0 * tensor_group_elements, tensor_group_elements);
    wmma::load_matrix_sync(right1, b + k0 * tensor_group_elements + 8, tensor_group_elements);
    wmma::mma_sync(accumulator0, left, right0, accumulator0);
    wmma::mma_sync(accumulator1, left, right1, accumulator1);
    __syncwarp();
  }
  wmma::store_matrix_sync(c + 8 * warp * tensor_group_elements, accumulator0,
                           tensor_group_elements, wmma::mem_row_major);
  wmma::store_matrix_sync(c + 8 * warp * tensor_group_elements + 8, accumulator1,
                           tensor_group_elements, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 8 * tensor_group_elements; i += 32)
  {
    const int row = 8 * warp + i / tensor_group_elements;
    const int e = elements[i % tensor_group_elements];
    if (row < tensor_dofs && e >= 0)
    {
      const int entry = e * tensor_dofs + row;
      output[map[entry]] = scaling[entry] * c[row * tensor_group_elements + i % tensor_group_elements];
    }
  }
#else
  asm("trap;");
#endif
}
