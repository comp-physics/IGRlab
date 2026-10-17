#include "sim/state/detail/euler_derived_fields.hpp"
#include "physics/euler/eos.hpp"

#include <cfenv>
#include <cmath>
#include <limits>
#include <typeinfo>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
#include "mfem/general/cuda.hpp"
#include <cuda_runtime.h>
#endif

namespace dg_sim
{
namespace detail
{
struct EulerDerivedFieldPlan
{
  const mfem::ParFiniteElementSpace* input = nullptr;
  const mfem::ParFiniteElementSpace* output = nullptr;
  const mfem::Mesh* mesh = nullptr;
  const mfem::FiniteElementCollection* input_fec = nullptr;
  const mfem::FiniteElementCollection* output_fec = nullptr;
  const mfem::FiniteElement* input_fe = nullptr;
  const mfem::FiniteElement* output_fe = nullptr;
  long input_sequence = -1, output_sequence = -1;
  long mesh_sequence = -1, nodes_sequence = -1;
  int ndofs = 0, ne = 0, output_dimension = 0;
  bool admitted = false;
  mfem::Vector nodal_scale;
  mfem::Array<int> declined;
};

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
namespace
{
enum class NativeArithmeticMode { Declined, Separate, FusedMomentum };

NativeArithmeticMode NativeArithmetic()
{
#if defined(__SSE2__) && !defined(__FAST_MATH__) && !defined(__FMA__)
  const unsigned int csr = _mm_getcsr();
  const bool supported = sizeof(mfem::real_t) == sizeof(double) &&
    std::numeric_limits<mfem::real_t>::is_iec559 &&
    std::fegetround() == FE_TONEAREST && (csr & 0xe040u) == 0 &&
    (csr & 0x1f80u) == 0x1f80u;
  return supported ? NativeArithmeticMode::Separate : NativeArithmeticMode::Declined;
#elif defined(__aarch64__) && !defined(__FAST_MATH__)
  unsigned long fpcr = 0;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  // The default Arm FPCR selects round-to-nearest, gradual underflow, ordinary
  // NaNs, and masked exceptions. Reject every nondefault control, including
  // flush modes and directed rounding, before launching CUDA arithmetic.
  if (fpcr != 0 || sizeof(mfem::real_t) != sizeof(double) ||
      !std::numeric_limits<mfem::real_t>::is_iec559 ||
      std::fegetround() != FE_TONEAREST)
  { return NativeArithmeticMode::Declined; }
  static const NativeArithmeticMode compiled_mode = []
  {
    // Arm FMA is available even without an x86-style __FMA__ macro. Qualify
    // the actual host EOS expression instead of assuming its contraction mode.
    // These exact binary inputs distinguish separate squares/add from a fused
    // second momentum term. Volatile products define the independent oracle.
    const double a = 0x1.3027f08b821d1p+0, b = 0x1.11d4388ee9e79p+0;
    volatile double aa = a * a, bb = b * b;
    const double separate = aa + bb;
    const double fused = std::fma(b, b, aa);
    if (separate == fused) { return NativeArithmeticMode::Declined; }
    mfem::Vector state(4);
    state(0) = 0.5; state(1) = a; state(2) = b; state(3) = separate;
    volatile int dim = 2;
    const double pressure = mfem::EulerPressureFromConservative(state, dim, 2.0, 1e-16);
    if (pressure == 0.0) { return NativeArithmeticMode::Separate; }
    if (pressure == separate - fused) { return NativeArithmeticMode::FusedMomentum; }
    return NativeArithmeticMode::Declined;
  }();
  return compiled_mode;
#else
  return NativeArithmeticMode::Declined;
#endif
}

bool Matches(const EulerDerivedFieldPlan& p,
             const mfem::ParFiniteElementSpace& input,
             const mfem::ParFiniteElementSpace& output)
{
  const auto* mesh = input.GetMesh();
  return p.input == &input && p.output == &output && p.mesh == mesh && mesh &&
    output.GetMesh() == mesh && p.input_fec == input.FEColl() &&
    p.output_fec == output.FEColl() && p.input_sequence == input.GetSequence() &&
    p.output_sequence == output.GetSequence() &&
    p.mesh_sequence == mesh->GetSequence() &&
    p.nodes_sequence == mesh->GetNodesSequence() && p.ne == input.GetNE() &&
    p.ne == output.GetNE() && p.ndofs == input.GetNDofs() &&
    p.ndofs == output.GetNDofs() &&
    input.GetOrdering() == mfem::Ordering::byNODES &&
    output.GetOrdering() == mfem::Ordering::byNODES &&
    p.output_dimension == output.GetVDim() &&
    (p.ne == 0 || (p.input_fe == input.GetFE(0) && p.output_fe == output.GetFE(0))) &&
    (!p.admitted || (p.input_fe->GetMapType() == mfem::FiniteElement::VALUE &&
                    p.output_fe->GetMapType() == mfem::FiniteElement::VALUE));
}

void Prepare(EulerDerivedFieldPlan& p,
             const mfem::ParFiniteElementSpace& input,
             const mfem::ParFiniteElementSpace& output)
{
  p.admitted = false;
  p.input = &input; p.output = &output; p.mesh = input.GetMesh();
  p.input_fec = input.FEColl(); p.output_fec = output.FEColl();
  p.input_sequence = input.GetSequence(); p.output_sequence = output.GetSequence();
  p.mesh_sequence = p.mesh ? p.mesh->GetSequence() : -1;
  p.nodes_sequence = p.mesh ? p.mesh->GetNodesSequence() : -1;
  p.ndofs = input.GetNDofs(); p.ne = input.GetNE();
  p.output_dimension = output.GetVDim();
  p.input_fe = p.ne ? input.GetFE(0) : nullptr;
  p.output_fe = output.GetNE() ? output.GetFE(0) : nullptr;
  const auto* in_l2 = dynamic_cast<const mfem::L2_FECollection*>(p.input_fec);
  const auto* out_l2 = dynamic_cast<const mfem::L2_FECollection*>(p.output_fec);
  if (!p.mesh || output.GetMesh() != p.mesh || p.mesh->Dimension() != 2 ||
      p.mesh->SpaceDimension() != 2 || p.mesh->Nonconforming() ||
      input.Nonconforming() || output.Nonconforming() ||
      input.GetNURBSext() || output.GetNURBSext() ||
      input.IsVariableOrder() || output.IsVariableOrder() ||
      !in_l2 || !out_l2 || p.ne <= 0 || output.GetNE() != p.ne ||
      input.GetVDim() != 4 || input.GetOrdering() != mfem::Ordering::byNODES ||
      output.GetOrdering() != mfem::Ordering::byNODES ||
      output.GetNDofs() != p.ndofs ||
      in_l2->GetBasisType() != out_l2->GetBasisType() ||
      (in_l2->GetBasisType() != mfem::BasisType::GaussLegendre &&
       in_l2->GetBasisType() != mfem::BasisType::GaussLobatto))
  { return; }
  if (typeid(*p.input_fe) != typeid(mfem::L2_QuadrilateralElement) ||
      typeid(*p.output_fe) != typeid(mfem::L2_QuadrilateralElement) ||
      p.input_fe->GetMapType() != mfem::FiniteElement::VALUE ||
      p.output_fe->GetMapType() != mfem::FiniteElement::VALUE)
  { return; }
  const int dof = p.input_fe->GetDof();
  if (p.output_fe->GetDof() != dof || p.ndofs / p.ne != dof ||
      p.ndofs % p.ne != 0)
  { return; }

  // Qualify exact diagonal interpolation at every output node once. MFEM's
  // barycentric products can round a nodal diagonal away from one, so retain
  // that actual value rather than replacing it by a mathematical identity.
  // The generic path forms a dot product starting at +0.
  const auto& nodes = p.output_fe->GetNodes();
  if (nodes.GetNPoints() != dof) { return; }
  mfem::Vector shape(dof);
  p.nodal_scale.SetSize(dof);
  auto* nodal_scale = p.nodal_scale.HostWrite();
  for (int q = 0; q < dof; ++q)
  {
    p.input_fe->CalcShape(nodes.IntPoint(q), shape);
    if (!std::isfinite(shape[q]) || shape[q] <= 0.0) { return; }
    nodal_scale[q] = shape[q];
    for (int j = 0; j < dof; ++j)
      if (j != q && shape[j] != 0.0) { return; }
  }
  mfem::Array<int> vdofs;
  mfem::DofTransformation transform;
  for (int e = 0; e < p.ne; ++e)
  {
    if (input.GetFE(e) != p.input_fe || output.GetFE(e) != p.output_fe)
    { return; }
    for (const auto* space : {&input, &output})
    {
      space->GetElementVDofs(e, vdofs, transform);
      if (!transform.IsIdentity() || vdofs.Size() != dof * space->GetVDim())
      { return; }
      for (int c = 0; c < space->GetVDim(); ++c)
        for (int j = 0; j < dof; ++j)
          if (vdofs[c * dof + j] != c * p.ndofs + e * dof + j) { return; }
    }
  }
  p.declined.SetSize(1);
  p.admitted = true;
}

template <bool Pressure>
__global__ void EulerCollocatedDerivedFieldKernel(
  int ndofs, int dof, const mfem::real_t* nodal_scale,
  const mfem::real_t* state, double gamma, double floor,
  mfem::real_t* output, int* declined, bool fused_momentum)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ndofs) { return; }
  double u[4];
  for (int c = 0; c < 4; ++c)
  {
    const double value = state[i + c * ndofs];
    if (!isfinite(value)) { atomicExch(declined, 1); return; }
    // All other nodal products are signed zeros. Preserve the actual diagonal
    // product and the host dot's initial +0, including -0 canonicalization.
    u[c] = __dadd_rn(0.0, __dmul_rn(nodal_scale[i % dof], value));
    if (!isfinite(u[c])) { atomicExch(declined, 1); return; }
  }
  const double rho = u[0] < floor ? floor : u[0];
  if (Pressure)
  {
    double mom2 = __dadd_rn(0.0, __dmul_rn(u[1], u[1]));
    mom2 = fused_momentum ? __fma_rn(u[2], u[2], mom2)
                          : __dadd_rn(mom2, __dmul_rn(u[2], u[2]));
    const double kinetic = __ddiv_rn(__dmul_rn(0.5, mom2), rho);
    const double value = __dmul_rn(__dsub_rn(gamma, 1.0), __dsub_rn(u[3], kinetic));
    if (!isfinite(value)) { atomicExch(declined, 1); return; }
    output[i] = value;
  }
  else
  {
    const double x = __ddiv_rn(u[1], rho);
    const double y = __ddiv_rn(u[2], rho);
    if (!isfinite(x) || !isfinite(y)) { atomicExch(declined, 1); return; }
    output[i] = x;
    output[i + ndofs] = y;
  }
}
}
#endif

bool TryProjectEulerDerivedField(
  const mfem::ParGridFunction& state, mfem::ParGridFunction& output,
  int dim, double gamma, double rho_floor, EulerDerivedField field,
  std::shared_ptr<EulerDerivedFieldPlan>& plan)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  const unsigned long native = static_cast<unsigned long>(mfem::Backend::CPU) |
                               static_cast<unsigned long>(mfem::Backend::CUDA);
  const NativeArithmeticMode arithmetic = NativeArithmetic();
  if (!mfem::Device::Allows(mfem::Backend::CUDA) || mfem::Device::Allows(~native) ||
      arithmetic == NativeArithmeticMode::Declined || dim != 2 || !std::isfinite(gamma) ||
      !std::isfinite(rho_floor) || !state.OwnsData() || !output.OwnsData())
  { return false; }
  const auto* input_fes = state.ParFESpace();
  const auto* output_fes = output.ParFESpace();
  const bool pressure = field == EulerDerivedField::Pressure;
  if (!input_fes || !output_fes || input_fes->GetVDim() != 4 ||
      output_fes->GetVDim() != (pressure ? 1 : 2) ||
      state.Size() != input_fes->GetVSize() || output.Size() != output_fes->GetVSize())
  { return false; }
  if (!plan) { plan = std::make_shared<EulerDerivedFieldPlan>(); }
  if (!Matches(*plan, *input_fes, *output_fes))
  { Prepare(*plan, *input_fes, *output_fes); }
  if (!plan->admitted) { return false; }
  plan->declined.HostWrite()[0] = 0;
  int* declined = plan->declined.ReadWrite();
  const auto* input = state.Read();
  output.UseDevice(true);
  auto* values = output.Write();
  const auto* nodal_scale = plan->nodal_scale.Read();
  const int blocks = (plan->ndofs + 255) / 256;
  if (pressure)
  {
    EulerCollocatedDerivedFieldKernel<true><<<blocks, 256>>>(
      plan->ndofs, plan->nodal_scale.Size(), nodal_scale,
      input, gamma, rho_floor, values, declined,
      arithmetic == NativeArithmeticMode::FusedMomentum);
  }
  else
  {
    EulerCollocatedDerivedFieldKernel<false><<<blocks, 256>>>(
      plan->ndofs, plan->nodal_scale.Size(), nodal_scale,
      input, gamma, rho_floor, values, declined, false);
  }
  MFEM_GPU_CHECK(cudaGetLastError());
  // Rare exceptional values retain the host oracle's full zero*NaN behavior
  // and NaN operand selection. Only this flag crosses to the host normally.
  return plan->declined.HostRead()[0] == 0;
#else
  (void)state; (void)output; (void)dim; (void)gamma; (void)rho_floor;
  (void)field; (void)plan;
  return false;
#endif
}
}
}
