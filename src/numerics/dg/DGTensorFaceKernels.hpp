#pragma once

// Private face storage and kernels. Include inside namespace mfem, after the
// existing Euler/IGR device helpers; no public diagnostic or replay interface.
namespace dg_igr_detail
{

struct TensorFaceKey
{
  std::array<const void*, 11> objects{};
  std::array<long, 19> metadata{};
  std::array<real_t, 3> contract{};
  bool operator==(const TensorFaceKey& other) const
  {
    return objects == other.objects && metadata == other.metadata &&
           contract == other.contract;
  }
};

inline TensorFaceKey MakeTensorFaceKey(
  const FiniteElementSpace& state, const ParGridFunction* sigma,
  const NumericalFlux* numerical_flux, const Vector* slopes,
  const IntegrationRule* rule, int order_offset, real_t gamma, bool requested,
  const FluxWithIGR* igr)
{
  const auto* scalar = sigma ? sigma->ParFESpace() : nullptr;
  const auto* mesh = state.GetMesh();
  const auto* scalar_mesh = scalar ? scalar->GetMesh() : nullptr;
  TensorFaceKey key;
  key.objects = {{&state, scalar, mesh, scalar_mesh,
    state.GetNE() ? state.GetFE(0) : nullptr,
    scalar && scalar->GetNE() ? scalar->GetFE(0) : nullptr,
    state.FEColl(), scalar ? scalar->FEColl() : nullptr,
    numerical_flux, slopes, rule}};
  key.metadata = {{state.GetSequence(), scalar ? scalar->GetSequence() : -1,
    mesh ? mesh->GetSequence() : -1, mesh ? mesh->GetNodesSequence() : -1,
    scalar_mesh ? scalar_mesh->GetSequence() : -1,
    scalar_mesh ? scalar_mesh->GetNodesSequence() : -1,
    state.GetNE(), state.GetNE() ? state.GetFE(0)->GetDof() : 0,
    scalar && scalar->GetNE() ? scalar->GetFE(0)->GetDof() : 0,
    scalar ? scalar->GetVDim() : 0, state.GetVDim(),
    static_cast<long>(state.GetOrdering()),
    scalar ? static_cast<long>(scalar->GetOrdering()) : -1,
    state.GetVSize(), sigma ? sigma->Size() : 0, requested ? 1L : 0L,
    igr && igr->GetInvariantDomainFlux() ? 1L : 0L, order_offset,
    ::igrlab::EnvFlagEnabled("DG_IGR_DG_TENSOR_FACE_ROW_REUSE") ? 1L : 0L}};
  key.contract = {{gamma, igr ? igr->GetVelocityRhoFloor() : 0.0,
    igr ? igr->GetInvariantDomainPressureFloor() : 0.0}};
  return key;
}

struct TensorFaceData
{
  TensorFaceKey key;
  bool ready = false;
  int n = 0, elements = 0, interior_faces = 0, boundary_faces = 0;
  Array<int> lex_to_local;
  // Scalar global dofs indexed by the state's local node, not sigma ordering.
  Array<int> sigma_map;
  Array<int> interior_side1, interior_side2, boundary_side;
  // Four entries per element: 2*canonical_face + (right side ? 1 : 0).
  Array<int> element_faces;
  // Reference signature = 2*local_side + tangential reversal (eight in 2D).
  Array<int> side_maps;
  // A signature is sparse only when every off-pivot CalcShape value is exactly
  // zero. Keep its actual diagonal scaling rather than assuming it is one.
  Array<int> tangent_sparse, tangent_permutation, tangent_inverse;
  Array<int> boundary_rejected;
  Vector endpoints, tangents;
  // Field-major [field][element*4+side][tangent node]. State and sigma stay FP64.
  Vector traces;
  // Equation-major [equation][canonical face][face quadrature point].
  Vector flux;
};

namespace tensor_face
{

inline bool Requested()
{
  const char* value = ::igrlab::GetEnv("DG_IGR_DG_TENSOR_FACE");
  MFEM_VERIFY(!value || std::string(value) == "off" ||
                std::string(value) == "fp64",
              "DG_TENSOR_FACE must be off or fp64.");
  return value && std::string(value) == "fp64";
}

inline bool RowReuseRequested()
{
  return ::igrlab::EnvFlagEnabled("DG_IGR_DG_TENSOR_FACE_ROW_REUSE");
}

inline int CheckedSize(std::size_t count)
{
  MFEM_VERIFY(count <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Tensor DG face storage exceeds native indexing.");
  return static_cast<int>(count);
}

#ifdef DG_IGR_HAS_ACCEL_FORALL
// Each sample contracts one normal tensor line, avoiding a full-volume gather
// at every face quadrature point. Tangential interpolation follows downstream.
struct TraceKernel
{
  int n, elements;
  bool use_sigma;
  const int *local, *state_map, *sigma_map;
  const real_t *endpoints, *state, *sigma;
  real_t* traces;
  MFEM_HOST_DEVICE void operator()(int sample) const
  {
    const int tangent = sample % n, incidence = sample / n;
    const int side = incidence % 4, e = incidence / 4, dof = n * n;
    for (int field = 0; field < (use_sigma ? 5 : 4); ++field)
    {
      real_t sum = 0.0;
      for (int k = 0; k < n; ++k)
      {
        const int lex = side < 2 ? k + n * tangent : tangent + n * k;
        const int node = local[lex];
        const real_t value = field < 4
          ? state[state_map[(e * dof + node) * 4 + field]]
          : sigma[sigma_map[e * dof + node]];
        sum += endpoints[(side % 2) * n + k] * value;
      }
      traces[(field * (4 * elements) + incidence) * n + tangent] = sum;
    }
  }
};

// Share each endpoint coefficient across independent field accumulators.
// Preserve every ascending-k sum and the original FP64 product/add order.
struct TraceRowReuseKernel : TraceKernel
{
  MFEM_HOST_DEVICE void operator()(int sample) const
  {
    const int tangent = sample % n, incidence = sample / n;
    const int side = incidence % 4, e = incidence / 4, dof = n * n;
    real_t v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, vs = 0.0;
    for (int k = 0; k < n; ++k)
    {
      const int lex = side < 2 ? k + n * tangent : tangent + n * k;
      const int node = local[lex], offset = (e * dof + node) * 4;
      const real_t weight = endpoints[(side % 2) * n + k];
      v0 += weight * state[state_map[offset]];
      v1 += weight * state[state_map[offset + 1]];
      v2 += weight * state[state_map[offset + 2]];
      v3 += weight * state[state_map[offset + 3]];
      if (use_sigma) { vs += weight * sigma[sigma_map[e * dof + node]]; }
    }
    const int offset = incidence * n + tangent, stride = 4 * elements * n;
    traces[offset] = v0;
    traces[stride + offset] = v1;
    traces[2 * stride + offset] = v2;
    traces[3 * stride + offset] = v3;
    if (use_sigma) { traces[4 * stride + offset] = vs; }
  }
};

struct FluxKernel
{
  int n, elements, interior_faces, boundary_faces, max_attr;
  bool use_sigma, invariant_domain, sigma_cfl, sigma_rusanov;
  real_t gamma, rho_floor, pressure_floor;
  const int *side1, *side2, *boundary_side, *side_maps, *boundary_attr;
  const int *tangent_sparse, *tangent_permutation;
  const int* boundary_type;
  const real_t *tangents, *traces, *interior_normal, *boundary_normal;
  const real_t *boundary_gamma, *boundary_pressure, *boundary_state;
  real_t *flux, *interior_speed, *boundary_speed;
  int* boundary_rejected;
  double *interior_telemetry, *boundary_telemetry;

  MFEM_HOST_DEVICE real_t Trace(int field, int incidence, int q) const
  {
    const int map = side_maps[incidence];
    if (tangent_sparse[map])
    {
      const int t = tangent_permutation[map * n + q];
      return tangents[(map * n + q) * n + t] *
             traces[(field * (4 * elements) + incidence) * n + t];
    }
    real_t value = 0.0;
    for (int t = 0; t < n; ++t)
    {
      value += tangents[(map * n + q) * n + t] *
               traces[(field * (4 * elements) + incidence) * n + t];
    }
    return value;
  }

  MFEM_HOST_DEVICE void operator()(int fq) const
  {
    const int faces = interior_faces + boundary_faces;
    const int f = fq / n, q = fq % n;
    const bool boundary = f >= interior_faces;
    const int b = f - interior_faces;
    const int point = boundary ? b * n + q : fq;
    const int left = boundary ? boundary_side[b] : side1[f];
    const int right = boundary ? left : side2[f];
    const real_t* normal_source = boundary ? boundary_normal : interior_normal;
    const real_t normal[3] = {normal_source[2 * point],
                              normal_source[2 * point + 1], 0.0};
    real_t ul[5] = {}, ur[5] = {}, fl[5] = {}, fr[5] = {};
    for (int eq = 0; eq < 4; ++eq)
    {
      ul[eq] = Trace(eq, left, q);
      if (!boundary) { ur[eq] = Trace(eq, right, q); }
    }
    if (boundary)
    {
      boundary_rejected[point] = 0;
      const int attr = boundary_attr[b];
      real_t boundary_g = attr > 0 && attr <= max_attr ? boundary_gamma[attr] : gamma;
      if (!(boundary_g > 0.0)) { boundary_g = gamma; }
      if (attr <= 0 || attr > max_attr ||
          !EvalBoundaryGhostEuler_(boundary_type[attr], 2, 4, boundary_g,
            boundary_pressure[attr], boundary_state + 4 * attr, normal, ul, ur,
            invariant_domain, rho_floor, pressure_floor))
      {
        boundary_rejected[point] = 1;
        boundary_speed[point] = std::numeric_limits<real_t>::infinity();
        for (int eq = 0; eq < 4; ++eq)
          flux[(eq * faces + f) * n + q] =
            std::numeric_limits<real_t>::quiet_NaN();
        return;
      }
    }
    const real_t sigma_l = use_sigma ? Trace(4, left, q) : 0.0;
    const real_t sigma_r = boundary ? sigma_l :
      (use_sigma ? Trace(4, right, q) : 0.0);
    real_t speed_l = 0.0, speed_r = 0.0;
    ComputeEulerFluxDotN_(ul, normal, 2, gamma, fl, speed_l);
    ComputeEulerFluxDotN_(ur, normal, 2, gamma, fr, speed_r);
    real_t cfl_l = speed_l, cfl_r = speed_r;
    real_t dissipation_l = speed_l, dissipation_r = speed_r;
    if (use_sigma)
    {
      AddSigmaDotNTerms_(ul, normal, 2, sigma_l, rho_floor, fl);
      AddSigmaDotNTerms_(ur, normal, 2, sigma_r, rho_floor, fr);
      double* telemetry = boundary ? boundary_telemetry : interior_telemetry;
      double* slot = invariant_domain
        ? telemetry + point * kFrozenSigmaTelemetryWidth_ : nullptr;
      SelectIGRFaceSpeeds_(ul, normal, 2, gamma, speed_l, sigma_l,
        rho_floor, pressure_floor, invariant_domain, sigma_cfl, sigma_rusanov,
        slot, cfl_l, dissipation_l);
      SelectIGRFaceSpeeds_(ur, normal, 2, gamma, speed_r, sigma_r,
        rho_floor, pressure_floor, invariant_domain, sigma_cfl, sigma_rusanov,
        slot, cfl_r, dissipation_r);
    }
    const real_t scaled_speed = fmax(dissipation_l, dissipation_r) *
      sqrt(normal[0] * normal[0] + normal[1] * normal[1]);
    (boundary ? boundary_speed : interior_speed)[point] = fmax(cfl_l, cfl_r);
    for (int eq = 0; eq < 4; ++eq)
    {
      const real_t value = 0.5 *
        (scaled_speed * (ul[eq] - ur[eq]) + fl[eq] + fr[eq]);
      flux[(eq * faces + f) * n + q] = value;
    }
  }
};

// One thread owns a coefficient. Both adjacent elements use one canonical
// interface flux with opposite signs. Retain every FP64 tangential coefficient,
// including off-diagonal roundoff from reversed reference coordinates.
// Omitting exact zero products can change the footprint of nonfinite inputs;
// finite-state equivalence does not require an additional diagnostic scan.
struct LiftKernel
{
  int n, interior_faces, boundary_faces;
  const int *local, *state_map, *element_faces, *side_maps;
  const int *tangent_sparse, *tangent_inverse, *boundary_rejected;
  const real_t *endpoints, *tangents, *interior_weights, *boundary_weights, *flux;
  real_t* residual;
  MFEM_HOST_DEVICE void operator()(int entry) const
  {
    const int eq = entry % 4, node = entry / 4, dof = n * n;
    const int e = node / dof, lex = node % dof, i = lex % n, j = lex / n;
    const int faces = interior_faces + boundary_faces;
    real_t sum = 0.0;
    for (int side = 0; side < 4; ++side)
    {
      const int incidence = 4 * e + side, encoded = element_faces[incidence];
      const int f = encoded / 2, map = side_maps[incidence];
      const int tangent = side < 2 ? j : i;
      const int normal = side < 2 ? i : j;
      real_t face_sum = 0.0;
      if (f >= interior_faces)
      {
        for (int q = 0; q < n; ++q)
          if (boundary_rejected[(f - interior_faces) * n + q])
          {
            residual[state_map[(e * dof + local[lex]) * 4 + eq]] =
              std::numeric_limits<real_t>::quiet_NaN();
            return;
          }
      }
      const bool sparse = tangent_sparse[map];
      const int first_q = sparse ? tangent_inverse[map * n + tangent] : 0;
      const int last_q = sparse ? first_q + 1 : n;
      for (int q = first_q; q < last_q; ++q)
      {
        const real_t weight = f < interior_faces
          ? interior_weights[f * n + q]
          : boundary_weights[(f - interior_faces) * n + q];
        const real_t weighted_flux = (encoded % 2 ? weight : -weight) *
          flux[(eq * faces + f) * n + q];
        face_sum += tangents[(map * n + q) * n + tangent] * weighted_flux;
      }
      sum += endpoints[(side % 2) * n + normal] * face_sum;
    }
    residual[state_map[(e * dof + local[lex]) * 4 + eq]] = sum;
  }
};

// One thread owns all Euler components at one node. Face/geometry weights
// are shared while every component retains the legacy q-then-side sum order.
struct LiftRowReuseKernel : LiftKernel
{
  MFEM_HOST_DEVICE void operator()(int node) const
  {
    const int dof = n * n, e = node / dof, lex = node % dof;
    const int i = lex % n, j = lex / n;
    const int faces = interior_faces + boundary_faces;
    const int output = (e * dof + local[lex]) * 4;
    real_t sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    for (int side = 0; side < 4; ++side)
    {
      const int incidence = 4 * e + side, encoded = element_faces[incidence];
      const int f = encoded / 2, map = side_maps[incidence];
      const int tangent = side < 2 ? j : i, normal = side < 2 ? i : j;
      if (f >= interior_faces)
      {
        for (int q = 0; q < n; ++q)
          if (boundary_rejected[(f - interior_faces) * n + q])
          {
            const real_t rejected = std::numeric_limits<real_t>::quiet_NaN();
            residual[state_map[output]] = rejected;
            residual[state_map[output + 1]] = rejected;
            residual[state_map[output + 2]] = rejected;
            residual[state_map[output + 3]] = rejected;
            return;
          }
      }
      real_t face0 = 0.0, face1 = 0.0, face2 = 0.0, face3 = 0.0;
      const bool sparse = tangent_sparse[map];
      const int first_q = sparse ? tangent_inverse[map * n + tangent] : 0;
      const int last_q = sparse ? first_q + 1 : n;
      for (int q = first_q; q < last_q; ++q)
      {
        const real_t weight = f < interior_faces ? interior_weights[f * n + q]
          : boundary_weights[(f - interior_faces) * n + q];
        const real_t signed_weight = encoded % 2 ? weight : -weight;
        const real_t coefficient = tangents[(map * n + q) * n + tangent];
        const real_t w0 = signed_weight * flux[f * n + q];
        const real_t w1 = signed_weight * flux[(faces + f) * n + q];
        const real_t w2 = signed_weight * flux[(2 * faces + f) * n + q];
        const real_t w3 = signed_weight * flux[(3 * faces + f) * n + q];
        face0 += coefficient * w0;
        face1 += coefficient * w1;
        face2 += coefficient * w2;
        face3 += coefficient * w3;
      }
      const real_t endpoint = endpoints[(side % 2) * n + normal];
      sum0 += endpoint * face0;
      sum1 += endpoint * face1;
      sum2 += endpoint * face2;
      sum3 += endpoint * face3;
    }
    residual[state_map[output]] = sum0;
    residual[state_map[output + 1]] = sum1;
    residual[state_map[output + 2]] = sum2;
    residual[state_map[output + 3]] = sum3;
  }
};

struct BoundaryIntegralKernel
{
  int n, interior_faces, boundary_faces;
  const real_t *weights, *flux;
  real_t* integral;
  MFEM_HOST_DEVICE void operator()(int eq) const
  {
    real_t sum = 0.0;
    const int faces = interior_faces + boundary_faces;
    for (int b = 0; b < boundary_faces; ++b)
      for (int q = 0; q < n; ++q)
        sum += weights[b * n + q] *
          flux[(eq * faces + interior_faces + b) * n + q];
    integral[eq] = sum;
  }
};
#endif

} // namespace tensor_face
} // namespace dg_igr_detail
