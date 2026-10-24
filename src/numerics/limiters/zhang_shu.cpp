#include "igrlab/environment.hpp"
#include "numerics/limiters/zhang_shu.hpp"

#include "physics/euler/eos.hpp"
#include "limiters_common.hpp"
#include "mfem/general/reducers.hpp"
#include "utils/config/env_flags.hpp"
#include "utils/profiling/nvtx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <unordered_map>
#include <vector>

namespace
{

bool UsesCollocatedLGLMean(const mfem::ParFiniteElementSpace& vfes,
                           const mfem::FiniteElement& fe)
{
  const auto* collection =
    dynamic_cast<const mfem::L2_FECollection*>(vfes.FEColl());
  const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&fe);
  return collection != nullptr && tensor != nullptr &&
         collection->GetBasisType() == mfem::BasisType::GaussLobatto &&
         tensor->GetBasisType() == mfem::BasisType::GaussLobatto;
}

const mfem::IntegrationRule& ZhangShuAverageRule(
  const mfem::ParFiniteElementSpace& vfes,
  const mfem::FiniteElement& fe,
  mfem::ElementTransformation& transformation,
  int quadrature_order)
{
  if (UsesCollocatedLGLMean(vfes, fe))
  {
    // Match DGHyperbolicConservationLaws::ComputeInvMass exactly.  DGSEM's
    // diagonal mass is defined by the p+1 point GLL rule, including the
    // Jacobian sampled at those collocation nodes.  Contracting around a mean
    // formed with a higher-order rule is not conservative in that discrete
    // norm on curved elements.
    static mfem::IntegrationRules lgl_rules(
      0, mfem::Quadrature1D::GaussLobatto);
    const int order = std::max(1, 2 * fe.GetOrder() - 1);
    return lgl_rules.Get(transformation.GetGeometryType(), order);
  }
  return mfem::IntRules.Get(transformation.GetGeometryType(),
                            quadrature_order);
}

void ZhangShuElementAverage(const mfem::ParFiniteElementSpace& vfes,
                            const mfem::FiniteElement& fe,
                            mfem::ElementTransformation& transformation,
                            const mfem::DenseMatrix& element_state,
                            int num_equations,
                            int quadrature_order,
                            mfem::Vector& average)
{
  const mfem::IntegrationRule& rule = ZhangShuAverageRule(
    vfes, fe, transformation, quadrature_order);
  mfem::Vector shape(fe.GetDof());
  mfem::Vector point_state(num_equations);
  average.SetSize(num_equations);
  average = 0.0;
  double measure = 0.0;
  for (int q = 0; q < rule.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint& point = rule.IntPoint(q);
    transformation.SetIntPoint(&point);
    fe.CalcShape(point, shape);
    element_state.MultTranspose(shape, point_state);
    const double weight = point.weight * transformation.Weight();
    average.Add(weight, point_state);
    measure += weight;
  }
  MFEM_VERIFY(measure > 0.0,
              "Zhang-Shu limiter encountered a non-positive element measure.");
  average /= measure;
}

struct ElementFaceCacheEntry
{
  long mesh_sequence = -1;
  int ne = -1;
  std::vector<std::vector<int>> elem_faces;
};

struct LimiterAdmissionCounts
{
  int rejected;
  int active;
};

struct DeviceLimiterCache
{
  const mfem::Mesh* mesh = nullptr;
  const mfem::FiniteElementCollection* fec = nullptr;
  long mesh_sequence = -1;
  int qord = -1;
  int ne = 0;
  int dof = 0;
  int dim = 0;
  int d1d = 0;
  int num_eq = 0;
  bool collocated_lgl_average = false;
  bool tensor_points = false;
  bool compact_reference_requested = false;
  bool shared_reference_points = false;
  bool mfem_volume_points = false;
  bool mfem_face_points = false;
  bool enforce_nodes = false;
  bool enforce_vtk_points = false;
  std::vector<int> face_qords;
  const mfem::Operator* restriction = nullptr;
  const mfem::QuadratureInterpolator* volume_interpolator = nullptr;
  int volume_nq = 0;
  int face_dof = 0;
  int interior_faces = 0;
  int boundary_faces = 0;
  mfem::Vector average_weights;
  mfem::Array<int> point_offsets;
  mfem::Array<int> face_point_offsets;
  mfem::Vector point_shapes;
  mfem::Vector point_factors;
  mfem::Vector volume_values;
  mfem::Vector face_values;
  mfem::Vector element_state;
  mfem::Vector averages;
  mfem::Vector theta_rho;
  mfem::Vector theta_p;
  mfem::Vector active;
  mfem::Vector inadmissible_average;
  mfem::Vector rejected_element;
  mfem::Vector collapsed_average;
  mfem::Vector average_repair;
  mfem::Array<mfem::real_t> admissibility_reduce_workspace;
  mfem::Array<LimiterAdmissionCounts> admission_counts_workspace;

  struct FaceCopyPlan
  {
    int nq = 0;
    int num_faces = 0;
    mfem::Array<int> dst;
    mfem::Array<int> src;
  };

  std::unique_ptr<mfem::ParFiniteElementSpace> face_scalar_fes;
  const mfem::Operator* interior_face_restriction = nullptr;
  const mfem::Operator* boundary_face_restriction = nullptr;
  std::vector<const mfem::FaceQuadratureInterpolator*> interior_face_qi;
  std::vector<const mfem::FaceQuadratureInterpolator*> boundary_face_qi;
  std::vector<FaceCopyPlan> interior_side0_plans;
  std::vector<FaceCopyPlan> interior_side1_plans;
  std::vector<FaceCopyPlan> boundary_plans;
  mfem::Vector face_component_lvec;
  mfem::Vector interior_face_double;
  mfem::Vector interior_face_side0;
  mfem::Vector interior_face_side1;
  mfem::Vector boundary_face_evec;
  mfem::Vector face_q_values;

  ~DeviceLimiterCache()
  {
    if (face_scalar_fes)
    {
      // The limiter cache is static process-lifetime state, while this adapter
      // space references the caller's mesh/FEC and owns MFEM face
      // restrictions/interpolators. Do not destroy it from static teardown,
      // where those dependencies may already be gone.
      face_scalar_fes.release();
    }
  }
};

struct LimiterTimingState
{
  bool initialized = false;
  bool enabled = false;
  bool phase_enabled = false;
  int warmup_calls = 0;
  int seen_calls = 0;
  int calls = 0;
  int active_calls = 0;
  long long limited_elements = 0;
  long long element_call_opportunities = 0;
  double minimum_theta = 1.0;
  double total_seconds = 0.0;
  double avg_active_elements = 0.0;
  double avg_tensor_points = 0.0;
  double avg_mfem_volume_points = 0.0;
  double avg_mfem_face_points = 0.0;
  double avg_point_count = 0.0;
  double average_seconds = 0.0;
  double admissible_average_seconds = 0.0;
  double density_volume_seconds = 0.0;
  double density_face_seconds = 0.0;
  double density_node_seconds = 0.0;
  double density_blend_seconds = 0.0;
  double pressure_volume_seconds = 0.0;
  double pressure_face_seconds = 0.0;
  double pressure_node_seconds = 0.0;
  double pressure_blend_seconds = 0.0;
  double scatter_seconds = 0.0;
  double avg_collapsed_elements = 0.0;
  double avg_pressure_bisections = 0.0;
  double avg_volume_points = 0.0;
  double avg_face_points = 0.0;
  double avg_node_points = 0.0;
  mfem::Array<mfem::real_t> active_reduce_workspace;
  mfem::Array<mfem::real_t> theta_reduce_workspace;

  ~LimiterTimingState()
  {
    if (!enabled || calls <= 0)
    {
      return;
    }
    const double inv_calls = 1.0 / static_cast<double>(calls);
    const double limited_element_fraction =
      element_call_opportunities > 0
        ? static_cast<double>(limited_elements) /
            static_cast<double>(element_call_opportunities)
        : 0.0;
    const int rank =
      (mfem::Mpi::IsInitialized() && !mfem::Mpi::IsFinalized())
        ? mfem::Mpi::WorldRank()
        : 0;
    std::cout << "[limiter] event=run-summary"
              << " rank=" << rank
              << " calls=" << calls
              << " active_calls=" << active_calls
              << " limited_elements=" << limited_elements
              << " element_call_opportunities="
              << element_call_opportunities
              << " limited_element_fraction=" << limited_element_fraction
              << " minimum_theta=" << minimum_theta
              << " aggregation=element-call-incidence"
              << " limited_fraction_denominator="
              << "element_call_opportunities"
              << " total_s=" << total_seconds
              << " avg_s=" << total_seconds * inv_calls
              << " avg_active_elements=" << avg_active_elements * inv_calls
              << " avg_tensor_points=" << avg_tensor_points * inv_calls
              << " avg_mfem_volume_points="
              << avg_mfem_volume_points * inv_calls
              << " avg_mfem_face_points="
              << avg_mfem_face_points * inv_calls
              << " avg_points=" << avg_point_count * inv_calls;
    if (phase_enabled)
    {
      std::cout << " cpu_average_s=" << average_seconds
                << " cpu_admissible_average_s=" << admissible_average_seconds
                << " cpu_density_volume_s=" << density_volume_seconds
                << " cpu_density_face_s=" << density_face_seconds
                << " cpu_density_node_s=" << density_node_seconds
                << " cpu_density_blend_s=" << density_blend_seconds
                << " cpu_pressure_volume_s=" << pressure_volume_seconds
                << " cpu_pressure_face_s=" << pressure_face_seconds
                << " cpu_pressure_node_s=" << pressure_node_seconds
                << " cpu_pressure_blend_s=" << pressure_blend_seconds
                << " cpu_scatter_s=" << scatter_seconds
                << " avg_collapsed_elements="
                << avg_collapsed_elements * inv_calls
                << " avg_pressure_bisections="
                << avg_pressure_bisections * inv_calls
                << " avg_volume_points=" << avg_volume_points * inv_calls
                << " avg_face_points=" << avg_face_points * inv_calls
                << " avg_node_points=" << avg_node_points * inv_calls;
    }
    std::cout << "\n";
  }
};

struct CpuLimiterPhaseSample
{
  double average_seconds = 0.0;
  double admissible_average_seconds = 0.0;
  double density_volume_seconds = 0.0;
  double density_face_seconds = 0.0;
  double density_node_seconds = 0.0;
  double density_blend_seconds = 0.0;
  double pressure_volume_seconds = 0.0;
  double pressure_face_seconds = 0.0;
  double pressure_node_seconds = 0.0;
  double pressure_blend_seconds = 0.0;
  double scatter_seconds = 0.0;
  double active_elements = 0.0;
  double collapsed_elements = 0.0;
  double pressure_bisections = 0.0;
  double volume_points = 0.0;
  double face_points = 0.0;
  double node_points = 0.0;
};

struct ActiveCountReductionBody
{
  const mfem::real_t* active = nullptr;

  MFEM_HOST_DEVICE void operator()(int i, mfem::real_t& sum) const
  {
    sum += active[i] != 0.0 ? 1.0 : 0.0;
  }
};

struct LimiterAdmissionReductionBody
{
  const double* rejected = nullptr;
  const double* active = nullptr;

  MFEM_HOST_DEVICE void operator()(int i, LimiterAdmissionCounts& counts) const
  {
    counts.rejected += rejected[i] != 0.0 ? 1 : 0;
    counts.active += active[i] != 0.0 ? 1 : 0;
  }
};

struct LimiterAdmissionReducer
{
  using value_type = LimiterAdmissionCounts;

  static MFEM_HOST_DEVICE void Join(value_type& a, const value_type& b)
  {
    a.rejected += b.rejected;
    a.active += b.active;
  }

  static MFEM_HOST_DEVICE void SetInitialValue(value_type& value)
  {
    value.rejected = 0;
    value.active = 0;
  }
};

struct MinimumThetaReductionBody
{
  const mfem::real_t* theta_rho = nullptr;
  const mfem::real_t* theta_p = nullptr;

  MFEM_HOST_DEVICE void operator()(int i, mfem::real_t& minimum) const
  {
    // The two mean-preserving contractions are applied in sequence, so their
    // product is the effective contraction of the original polynomial.
    const mfem::real_t theta = theta_rho[i] * theta_p[i];
    minimum = theta < minimum ? theta : minimum;
  }
};

double CountActiveElements(const mfem::Vector* active,
                           mfem::Array<mfem::real_t>& workspace)
{
  if (active == nullptr || active->Size() <= 0)
  {
    return 0.0;
  }

  if (mfem::Device::Allows(mfem::Backend::DEVICE_MASK))
  {
    mfem::real_t local_active = 0.0;
    ActiveCountReductionBody body;
    body.active = active->Read();
    mfem::reduce(active->Size(),
                 local_active,
                 body,
                 mfem::SumReducer<mfem::real_t>(),
                 true,
                 workspace);
    return static_cast<double>(local_active);
  }

  double local_active = 0.0;
  const double* values = active->HostRead();
  for (int i = 0; i < active->Size(); ++i)
  {
    local_active += values[i] != 0.0 ? 1.0 : 0.0;
  }
  return local_active;
}

double MinimumTheta(const mfem::Vector* theta_rho,
                    const mfem::Vector* theta_p,
                    mfem::Array<mfem::real_t>& workspace)
{
  if (theta_rho == nullptr || theta_p == nullptr || theta_rho->Size() <= 0)
  {
    return 1.0;
  }
  MFEM_VERIFY(theta_rho->Size() == theta_p->Size(),
              "Limiter theta diagnostics must have matching sizes.");

  if (mfem::Device::Allows(mfem::Backend::DEVICE_MASK))
  {
    mfem::real_t local_minimum = 1.0;
    MinimumThetaReductionBody body;
    body.theta_rho = theta_rho->Read();
    body.theta_p = theta_p->Read();
    mfem::reduce(theta_rho->Size(),
                 local_minimum,
                 body,
                 mfem::MinReducer<mfem::real_t>(),
                 true,
                 workspace);
    return static_cast<double>(local_minimum);
  }

  double local_minimum = 1.0;
  const double* density = theta_rho->HostRead();
  const double* pressure = theta_p->HostRead();
  for (int i = 0; i < theta_rho->Size(); ++i)
  {
    local_minimum = std::min(local_minimum, density[i] * pressure[i]);
  }
  return local_minimum;
}

void AccumulateLimiterAggregate(LimiterTimingState& timing,
                                double local_active,
                                int local_elements,
                                double local_minimum_theta,
                                MPI_Comm communicator)
{
  double local_counts[2] = {
    local_active, static_cast<double>(local_elements)};
  double global_counts[2] = {local_counts[0], local_counts[1]};
  double global_minimum_theta = local_minimum_theta;
  if (mfem::Mpi::IsInitialized() && !mfem::Mpi::IsFinalized())
  {
    MPI_Allreduce(local_counts,
                  global_counts,
                  2,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_SUM,
                  communicator);
    MPI_Allreduce(&local_minimum_theta,
                  &global_minimum_theta,
                  1,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_MIN,
                  communicator);
  }

  const long long global_active =
    static_cast<long long>(std::llround(global_counts[0]));
  const long long global_elements =
    static_cast<long long>(std::llround(global_counts[1]));
  timing.active_calls += global_active > 0 ? 1 : 0;
  timing.limited_elements += global_active;
  timing.element_call_opportunities += global_elements;
  timing.minimum_theta = std::min(timing.minimum_theta,
                                  global_minimum_theta);
  timing.avg_active_elements += global_counts[0];
}

int GlobalRejectedElementCount(
  const mfem::ParFiniteElementSpace& vfes,
  int local_rejected_elements)
{
  int global_rejected_elements = local_rejected_elements;
  if (mfem::Mpi::IsInitialized() && !mfem::Mpi::IsFinalized())
  {
    MPI_Allreduce(&local_rejected_elements,
                  &global_rejected_elements,
                  1,
                  mfem::MPITypeMap<int>::mpi_type,
                  MPI_SUM,
                  vfes.GetComm());
  }
  return global_rejected_elements;
}

LimiterTimingState& GetLimiterTimingState()
{
  static LimiterTimingState state;
  if (!state.initialized)
  {
    state.phase_enabled =
      ::igrlab::EnvFlagEnabled("DG_IGR_LIMITER_CPU_PHASE_TIMING");
    state.enabled = ::igrlab::EnvFlagEnabled("DG_IGR_LIMITER_TIMING") ||
                    state.phase_enabled;
    if (const char* value = ::igrlab::GetEnv("DG_IGR_TIMING_WARMUP_CALLS"))
    {
      state.warmup_calls = std::max(0, std::atoi(value));
    }
    state.initialized = true;
  }
  return state;
}

void RecordLimiterTiming(double seconds,
                         const DeviceLimiterCache& cache,
                         const mfem::Vector* active,
                         const mfem::Vector* theta_rho,
                         const mfem::Vector* theta_p,
                         MPI_Comm communicator,
                         double known_local_active = -1.0)
{
  LimiterTimingState& timing = GetLimiterTimingState();
  if (!timing.enabled)
  {
    return;
  }
  if (timing.seen_calls++ < timing.warmup_calls)
  {
    return;
  }

  const double local_active = known_local_active >= 0.0
                                ? known_local_active
                                : CountActiveElements(
                                    active, timing.active_reduce_workspace);
  const double local_minimum_theta =
    MinimumTheta(theta_rho, theta_p, timing.theta_reduce_workspace);

  ++timing.calls;
  timing.total_seconds += seconds;
  AccumulateLimiterAggregate(timing,
                             local_active,
                             cache.ne,
                             local_minimum_theta,
                             communicator);
  timing.avg_tensor_points += cache.tensor_points ? 1.0 : 0.0;
  timing.avg_mfem_volume_points += cache.mfem_volume_points ? 1.0 : 0.0;
  timing.avg_mfem_face_points += cache.mfem_face_points ? 1.0 : 0.0;
  if (cache.ne > 0)
  {
    int points = cache.point_offsets[cache.ne] + cache.ne * cache.volume_nq;
    if (cache.mfem_face_points)
    {
      points += cache.face_point_offsets[cache.ne];
    }
    timing.avg_point_count += static_cast<double>(points);
  }
}

inline const std::vector<std::vector<int>>& GetElementFaces(mfem::Mesh& mesh)
{
  static std::unordered_map<const mfem::Mesh*, ElementFaceCacheEntry> cache;
  auto& entry = cache[&mesh];
  if (entry.mesh_sequence == mesh.GetSequence() && entry.ne == mesh.GetNE())
  {
    return entry.elem_faces;
  }

  entry.mesh_sequence = mesh.GetSequence();
  entry.ne = mesh.GetNE();
  entry.elem_faces.clear();
  entry.elem_faces.resize(entry.ne);

  for (int f = 0; f < mesh.GetNumFaces(); ++f)
  {
    int e1 = -1, e2 = -1;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e1 >= 0 && e1 < entry.ne)
    {
      entry.elem_faces[e1].push_back(f);
    }
    if (e2 >= 0 && e2 < entry.ne)
    {
      entry.elem_faces[e2].push_back(f);
    }
  }

  return entry.elem_faces;
}

inline std::vector<int>
BuildFaceQuadratureOrders(const mfem::ParFiniteElementSpace& vfes,
                          int volume_qord,
                          const std::vector<int>& configured_face_qords)
{
  std::vector<int> qords;
  qords.reserve(configured_face_qords.size() + 2);

  auto push_unique_positive = [&](int qord)
  {
    if (qord <= 0)
    {
      return;
    }
    if (std::find(qords.begin(), qords.end(), qord) == qords.end())
    {
      qords.push_back(qord);
    }
  };

  push_unique_positive(volume_qord);
  for (const int qord : configured_face_qords)
  {
    push_unique_positive(qord);
  }

  if (qords.size() == 1)
  {
    push_unique_positive(2 * vfes.GetFE(0)->GetOrder() + 1);
  }

  return qords;
}

void ExtractStateComponent(const mfem::Vector& source,
                           int ndofs,
                           int neq,
                           int eq,
                           mfem::Vector& component)
{
  const double* src = source.Read();
  double* dst = component.Write();
  mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int i)
  {
    dst[i] = src[i + ndofs * eq];
  });
}

void SplitDoubleValuedScalarFaceEVector(const mfem::Vector& double_face,
                                        int face_dof,
                                        int num_faces,
                                        mfem::Vector& side0,
                                        mfem::Vector& side1)
{
  if (num_faces <= 0)
  {
    return;
  }
  const double* src = double_face.Read();
  double* dst0 = side0.Write();
  double* dst1 = side1.Write();
  mfem::forall(face_dof * num_faces, [=] MFEM_HOST_DEVICE(int i)
  {
    const int d = i % face_dof;
    const int f = i / face_dof;
    dst0[d + face_dof * f] = src[d + face_dof * (2 * f)];
    dst1[d + face_dof * f] = src[d + face_dof * (1 + 2 * f)];
  });
}

void CopyFaceQValuesToElementMajor(
  const mfem::Vector& face_q_values,
  const DeviceLimiterCache::FaceCopyPlan& plan,
  int total_face_points,
  int eq,
  mfem::Vector& element_major_values)
{
  if (plan.dst.Size() == 0)
  {
    return;
  }
  const int entries = plan.dst.Size();
  const int* dst_index = plan.dst.Read();
  const int* src_index = plan.src.Read();
  const double* src = face_q_values.Read();
  double* dst = element_major_values.ReadWrite();
  mfem::forall(entries, [=] MFEM_HOST_DEVICE(int i)
  {
    dst[dst_index[i] + total_face_points * eq] = src[src_index[i]];
  });
}

void CopyDeviceVector(const mfem::Vector& src, mfem::Vector& dst)
{
  const int n = src.Size();
  dst.SetSize(n);
  dst.UseDevice(true);
  const double* src_data = src.Read();
  double* dst_data = dst.Write();
  mfem::forall(n, [=] MFEM_HOST_DEVICE(int i) { dst_data[i] = src_data[i]; });
}

bool SupportsMfemFaceLimiterPath(const mfem::ParFiniteElementSpace& vfes,
                                 const std::vector<int>& face_qords)
{
  if (face_qords.empty())
  {
    return false;
  }
  // MFEM's ParL2FaceRestriction can be fragile for this adapter space; the
  // device limiter has an element-local face-point fallback for MPI runs.
  if (mfem::Mpi::IsInitialized() && mfem::Mpi::WorldSize() > 1)
  {
    return false;
  }
  if (!::igrlab::EnvFlagEnabledDefault("DG_IGR_LIMITER_MFEM_FACE_QI", true))
  {
    return false;
  }
  if (!vfes.IsDGSpace() || vfes.IsVariableOrder() || !vfes.Conforming())
  {
    return false;
  }
  mfem::Mesh& mesh = *vfes.GetMesh();
  if (mesh.GetNE() <= 0 || mesh.Dimension() < 2 ||
      mesh.GetNumGeometries(mesh.Dimension() - 1) > 1)
  {
    return false;
  }
  return mfem::UsesTensorBasis(vfes) &&
         mfem::FaceQuadratureInterpolator::SupportsFESpace(vfes);
}

void PrepareFaceCopyPlan(DeviceLimiterCache::FaceCopyPlan& plan)
{
  plan.dst.UseDevice();
  plan.src.UseDevice();
}

static inline double ThetaForPressure(const mfem::Vector& Ubar,
                                      const mfem::Vector& Up,
                                      int dim,
                                      double gamma,
                                      double p_min,
                                      double theta_max = 1.0,
                                      int iters = 12)
{
  mfem::Vector Utest(Ubar.Size());
  Utest = Ubar;
  Utest.Add(theta_max, Up);
  Utest.Add(-theta_max, Ubar);
  if (mfem::EulerPressureFromConservative(Utest,
                                          dim,
                                          gamma,
                                          /*rho_floor=*/1e-16) >= p_min)
  {
    return theta_max;
  }

  double lo = 0.0, hi = theta_max;
  for (int it = 0; it < iters; ++it)
  {
    const double mid = 0.5 * (lo + hi);
    Utest = Ubar;
    Utest.Add(mid, Up);
    Utest.Add(-mid, Ubar);
    const double p = mfem::EulerPressureFromConservative(Utest,
                                                         dim,
                                                         gamma,
                                                         /*rho_floor=*/1e-16);
    if (p >= p_min)
    {
      lo = mid;
    }
    else
    {
      hi = mid;
    }
  }
  return lo;
}

MFEM_HOST_DEVICE inline double DevicePressure(const double* state,
                                               int dim,
                                               double gamma)
{
  const double rho = state[0] > 1.0e-16 ? state[0] : 1.0e-16;
  double momentum_sq = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    momentum_sq += state[1 + d] * state[1 + d];
  }
  return (gamma - 1.0) *
         (state[1 + dim] - 0.5 * momentum_sq / rho);
}

MFEM_HOST_DEVICE inline bool ConservativeStateIsFinite(const double* state,
                                                       int num_eq)
{
  bool finite = true;
  for (int component = 0; component < num_eq; ++component)
  {
    finite = finite && std::isfinite(state[component]);
  }
  return finite;
}

MFEM_HOST_DEVICE inline bool EulerConsumerPointIsFinite(const double* state,
                                                        int dim,
                                                        double gamma)
{
  const int num_eq = dim + 2;
  if (!ConservativeStateIsFinite(state, num_eq) ||
      !std::isfinite(state[0]))
  {
    return false;
  }
  return std::isfinite(DevicePressure(state, dim, gamma));
}

MFEM_HOST_DEVICE inline double LimiterPointBasis(const double* shapes,
                                                 const double* factors,
                                                 bool tensor_points,
                                                 int point,
                                                 int dof_index,
                                                 int dof,
                                                 int dim,
                                                 int d1d)
{
  if (!tensor_points)
  {
    return shapes[point * dof + dof_index];
  }
  const int i = dof_index % d1d;
  const int j = dim > 1 ? (dof_index / d1d) % d1d : 0;
  const int k = dim > 2 ? dof_index / (d1d * d1d) : 0;
  double value = factors[(point * dim) * d1d + i];
  if (dim > 1) { value *= factors[(point * dim + 1) * d1d + j]; }
  if (dim > 2) { value *= factors[(point * dim + 2) * d1d + k]; }
  return value;
}

MFEM_HOST_DEVICE inline double DeviceThetaForPressure(const double* average,
                                                       const double* point,
                                                       int dim,
                                                       double gamma,
                                                       double p_min,
                                                       double theta_max,
                                                       int iterations)
{
  double state[5];
  for (int c = 0; c < dim + 2; ++c)
  {
    state[c] = average[c] + theta_max * (point[c] - average[c]);
  }
  if (DevicePressure(state, dim, gamma) >= p_min)
  {
    return theta_max;
  }

  double lo = 0.0;
  double hi = theta_max;
  for (int it = 0; it < iterations; ++it)
  {
    const double mid = 0.5 * (lo + hi);
    for (int c = 0; c < dim + 2; ++c)
    {
      state[c] = average[c] + mid * (point[c] - average[c]);
    }
    if (DevicePressure(state, dim, gamma) >= p_min)
    {
      lo = mid;
    }
    else
    {
      hi = mid;
    }
  }
  return lo;
}

// Each CUDA block owns one p5 tensor cell.  Point evaluations are independent:
// their dof sums and pressure bisections retain the serial kernel's order, and
// only finite scalar minima/boolean admission flags are reduced across lanes.
// Keep the cell mean serial so curved DG and collocated DGSEM use exactly the
// existing physical/discrete mean and its accumulation order.
struct CooperativeZhangShuQuadKernel
{
  static constexpr int kD1D = 6;
  static constexpr int kDof = kD1D * kD1D;
  static constexpr int kNumEq = 4;
  static constexpr int kLanes = 32;

  int ne = 0;
  int volume_nq = 0;
  int total_face_points = 0;
  int p_bisect_iters = 0;
  double gamma = 0.0;
  double rho_min = 0.0;
  double p_min = 0.0;
  bool capture_diagnostics = false;
  bool shared_reference_points = false;
  const double* weights = nullptr;
  const int* offsets = nullptr;
  const int* face_offsets = nullptr;
  const double* factors = nullptr;
  const double* volume_values = nullptr;
  const double* face_values = nullptr;
  double* state = nullptr;
  double* averages = nullptr;
  double* theta_rho = nullptr;
  double* theta_p = nullptr;
  double* active = nullptr;
  double* inadmissible_average = nullptr;
  double* rejected_element = nullptr;
  double* collapsed_average = nullptr;
  double* average_repair = nullptr;

  MFEM_HOST_DEVICE void LoadPoint(int e,
                                  int point,
                                  int face_begin,
                                  int face_count,
                                  const double* cell_state,
                                  const double* average,
                                  double rho_theta,
                                  bool pressure_pass,
                                  double* value) const
  {
    if (point < volume_nq)
    {
      for (int c = 0; c < kNumEq; ++c)
      {
        const double original =
          volume_values[point + volume_nq * (c + kNumEq * e)];
        value[c] = pressure_pass
                     ? average[c] + rho_theta * (original - average[c])
                     : original;
      }
      return;
    }
    point -= volume_nq;
    if (point < face_count)
    {
      for (int c = 0; c < kNumEq; ++c)
      {
        const double original =
          face_values[face_begin + point + total_face_points * c];
        value[c] = pressure_pass
                     ? average[c] + rho_theta * (original - average[c])
                     : original;
      }
      return;
    }
    point += (shared_reference_points ? 0 : offsets[e]) - face_count;
    for (int c = 0; c < kNumEq; ++c) { value[c] = 0.0; }
    for (int d = 0; d < kDof; ++d)
    {
      const double basis = LimiterPointBasis(nullptr,
                                             factors,
                                             true,
                                             point,
                                             d,
                                             kDof,
                                             2,
                                             kD1D);
      for (int c = 0; c < kNumEq; ++c)
      {
        value[c] += basis * cell_state[d + kDof * c];
      }
    }
  }

  MFEM_HOST_DEVICE void operator()(int e) const
  {
    MFEM_SHARED double cell_state[kDof * kNumEq];
    MFEM_SHARED double average[kNumEq];
    MFEM_SHARED double lane_theta[kLanes];
    MFEM_SHARED int lane_finite[kLanes];
    MFEM_SHARED double rho_theta;
    MFEM_SHARED double pressure_theta;
    MFEM_SHARED int average_is_admissible;

    MFEM_FOREACH_THREAD(i, x, kDof * kNumEq)
    {
      cell_state[i] = state[i + kDof * kNumEq * e];
    }
    MFEM_SYNC_THREAD;
    if (MFEM_THREAD_ID(x) == 0)
    {
      for (int c = 0; c < kNumEq; ++c) { average[c] = 0.0; }
      for (int d = 0; d < kDof; ++d)
      {
        for (int c = 0; c < kNumEq; ++c)
        {
          average[c] += weights[e * kDof + d] *
                        cell_state[d + kDof * c];
        }
      }
      bool finite = true;
      for (int c = 0; c < kNumEq; ++c)
      {
        finite = finite && std::isfinite(average[c]);
        averages[c + kNumEq * e] = average[c];
        if (capture_diagnostics) { average_repair[e + ne * c] = 0.0; }
      }
      if (capture_diagnostics) { collapsed_average[e] = 0.0; }
      double momentum_sq = 0.0;
      for (int d = 0; d < 2; ++d)
      {
        momentum_sq += average[1 + d] * average[1 + d];
      }
      const double kinetic = finite && average[0] > rho_min
                               ? 0.5 * momentum_sq / average[0]
                               : 0.0;
      const double pressure = (gamma - 1.0) * (average[3] - kinetic);
      average_is_admissible = finite && average[0] > rho_min &&
                              std::isfinite(pressure) && pressure > p_min;
      inadmissible_average[e] = average_is_admissible ? 0.0 : 1.0;
      theta_rho[e] = 1.0;
      theta_p[e] = 1.0;
      active[e] = 0.0;
      rejected_element[e] = average_is_admissible ? 0.0 : 1.0;
    }
    MFEM_SYNC_THREAD;
    if (!average_is_admissible) { return; }

    const int face_begin = face_offsets ? face_offsets[e] : 0;
    const int face_count = face_offsets ? face_offsets[e + 1] - face_begin : 0;
    const int point_count = volume_nq + face_count +
                            offsets[e + 1] - offsets[e];
    MFEM_FOREACH_THREAD(lane, x, kLanes)
    {
      double minimum = 1.0;
      bool finite = true;
      for (int point = lane; point < point_count; point += kLanes)
      {
        double value[kNumEq];
        LoadPoint(e, point, face_begin, face_count, cell_state, average,
                  1.0, false, value);
        const bool point_finite = EulerConsumerPointIsFinite(value, 2, gamma);
        finite = finite && point_finite;
        if (point_finite && value[0] < rho_min)
        {
          const double denominator = average[0] - value[0];
          const double candidate = denominator > 0.0
                                     ? (average[0] - rho_min) / denominator
                                     : 0.0;
          minimum = candidate < minimum ? candidate : minimum;
        }
      }
      lane_theta[lane] = minimum;
      lane_finite[lane] = finite ? 1 : 0;
    }
    MFEM_SYNC_THREAD;
    if (MFEM_THREAD_ID(x) == 0)
    {
      rho_theta = 1.0;
      for (int lane = 0; lane < kLanes; ++lane)
      {
        rho_theta = lane_theta[lane] < rho_theta
                      ? lane_theta[lane] : rho_theta;
      }
      theta_rho[e] = rho_theta;
    }
    MFEM_SYNC_THREAD;
    if (rho_theta < 1.0)
    {
      MFEM_FOREACH_THREAD(i, x, kDof * kNumEq)
      {
        const int c = i / kDof;
        cell_state[i] = average[c] +
                        rho_theta * (cell_state[i] - average[c]);
      }
    }
    MFEM_SYNC_THREAD;

    MFEM_FOREACH_THREAD(lane, x, kLanes)
    {
      double minimum = 1.0;
      bool finite = lane_finite[lane] != 0;
      for (int point = lane; point < point_count; point += kLanes)
      {
        double value[kNumEq];
        LoadPoint(e, point, face_begin, face_count, cell_state, average,
                  rho_theta, true, value);
        const bool point_finite = EulerConsumerPointIsFinite(value, 2, gamma);
        finite = finite && point_finite;
        if (point_finite && DevicePressure(value, 2, gamma) < p_min)
        {
          const double candidate = DeviceThetaForPressure(
            average, value, 2, gamma, p_min, 1.0, p_bisect_iters);
          minimum = candidate < minimum ? candidate : minimum;
        }
      }
      lane_theta[lane] = minimum;
      lane_finite[lane] = finite ? 1 : 0;
    }
    MFEM_SYNC_THREAD;
    if (MFEM_THREAD_ID(x) == 0)
    {
      pressure_theta = 1.0;
      bool finite = true;
      for (int lane = 0; lane < kLanes; ++lane)
      {
        pressure_theta = lane_theta[lane] < pressure_theta
                           ? lane_theta[lane] : pressure_theta;
        finite = finite && lane_finite[lane] != 0;
      }
      theta_p[e] = pressure_theta;
      active[e] = (rho_theta < 1.0 || pressure_theta < 1.0) ? 1.0 : 0.0;
      rejected_element[e] = finite ? 0.0 : 1.0;
    }
    MFEM_SYNC_THREAD;
    if (rho_theta < 1.0 || pressure_theta < 1.0)
    {
      MFEM_FOREACH_THREAD(i, x, kDof * kNumEq)
      {
        double value = cell_state[i];
        if (pressure_theta < 1.0)
        {
          const int c = i / kDof;
          value = average[c] + pressure_theta * (value - average[c]);
        }
        state[i + kDof * kNumEq * e] = value;
      }
    }
  }
};


inline int NativeDof(const mfem::FiniteElement& fe, int lex_dof)
{
  const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&fe);
  if (tensor == nullptr)
  {
    return lex_dof;
  }
  const mfem::Array<int>& dof_map = tensor->GetDofMap();
  if (dof_map.Size() == 0)
  {
    return lex_dof;
  }
  MFEM_ASSERT(dof_map.Size() > lex_dof, "Invalid tensor dof map.");
  const int signed_dof = dof_map[lex_dof];
  return signed_dof >= 0 ? signed_dof : -1 - signed_dof;
}

bool BuildTensorPointFactors(DeviceLimiterCache& cache,
                             const std::vector<double>& shapes)
{
  const int dim = cache.dim;
  const int d1d = cache.d1d;
  const int dof = cache.dof;
  const int points = static_cast<int>(shapes.size()) / dof;
  const bool tensor_size =
    (dim == 1 && dof == d1d) ||
    (dim == 2 && dof == d1d * d1d) ||
    (dim == 3 && dof == d1d * d1d * d1d);
  if (!tensor_size)
  {
    return false;
  }

  std::vector<double> factors(points * dim * d1d, 0.0);
  double max_error = 0.0;
  for (int point = 0; point < points; ++point)
  {
    int pivot = 0;
    for (int d = 1; d < dof; ++d)
    {
      if (std::abs(shapes[point * dof + d]) >
          std::abs(shapes[point * dof + pivot]))
      {
        pivot = d;
      }
    }
    const double pivot_value = shapes[point * dof + pivot];
    if (std::abs(pivot_value) < 1.0e-15)
    {
      return false;
    }
    const int pi = pivot % d1d;
    const int pj = dim > 1 ? (pivot / d1d) % d1d : 0;
    const int pk = dim > 2 ? pivot / (d1d * d1d) : 0;
    for (int i = 0; i < d1d; ++i)
    {
      factors[(point * dim) * d1d + i] =
        shapes[point * dof + i + d1d * (pj + d1d * pk)];
    }
    if (dim > 1)
    {
      for (int j = 0; j < d1d; ++j)
      {
        factors[(point * dim + 1) * d1d + j] =
          shapes[point * dof + pi + d1d * (j + d1d * pk)] / pivot_value;
      }
    }
    if (dim > 2)
    {
      for (int k = 0; k < d1d; ++k)
      {
        factors[(point * dim + 2) * d1d + k] =
          shapes[point * dof + pi + d1d * (pj + d1d * k)] / pivot_value;
      }
    }
    for (int d = 0; d < dof; ++d)
    {
      const int i = d % d1d;
      const int j = dim > 1 ? (d / d1d) % d1d : 0;
      const int k = dim > 2 ? d / (d1d * d1d) : 0;
      double value = factors[(point * dim) * d1d + i];
      if (dim > 1) { value *= factors[(point * dim + 1) * d1d + j]; }
      if (dim > 2) { value *= factors[(point * dim + 2) * d1d + k]; }
      max_error = std::max(max_error,
                           std::abs(value - shapes[point * dof + d]));
    }
  }
  if (max_error >= 1.0e-11)
  {
    return false;
  }
  cache.point_factors.SetSize(static_cast<int>(factors.size()));
  for (int i = 0; i < cache.point_factors.Size(); ++i)
  {
    cache.point_factors[i] = factors[i];
  }
  return true;
}

DeviceLimiterCache& GetDeviceLimiterCache(
  mfem::ParFiniteElementSpace& vfes,
  int num_eq,
  int qord,
  const std::vector<int>& face_qords,
  mfem::ZhangShuPointSet point_set,
  bool enforce_vtk_points)
{
  static std::unordered_map<const mfem::ParFiniteElementSpace*,
                            std::unique_ptr<DeviceLimiterCache>> caches;
  auto& cache_ptr = caches[&vfes];
  mfem::Mesh& mesh = *vfes.GetMesh();
  const int ne = vfes.GetNE();
  const int dof = ne > 0 ? vfes.GetFE(0)->GetDof() : 0;
  const mfem::Operator* current_restriction =
    vfes.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
  const bool tensor_basis = mfem::UsesTensorBasis(vfes);
  const bool requested_mfem_volume_points =
    ne > 0 && tensor_basis &&
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_LIMITER_MFEM_VOLUME_QI", true) &&
    mfem::QuadratureInterpolator::SupportsFESpace(vfes);
  const bool requested_mfem_face_points =
    ne > 0 && SupportsMfemFaceLimiterPath(vfes, face_qords);
  const bool requested_collocated_lgl_average =
    ne > 0 && UsesCollocatedLGLMean(vfes, *vfes.GetFE(0));
  const auto* l2_fec =
    dynamic_cast<const mfem::L2_FECollection*>(vfes.FEColl());
  const bool consumer_points =
    point_set == mfem::ZhangShuPointSet::kConsumer;
  const bool requested_enforce_nodes =
    (l2_fec != nullptr &&
     l2_fec->GetBasisType() == mfem::BasisType::GaussLobatto) ||
    consumer_points;
  const bool requested_enforce_vtk_points = enforce_vtk_points;
  const bool requested_compact_reference_points =
    requested_mfem_volume_points && requested_mfem_face_points &&
    mesh.Dimension() == 2 && dof == CooperativeZhangShuQuadKernel::kDof &&
    ne > 0 && vfes.GetFE(0)->GetOrder() + 1 ==
      CooperativeZhangShuQuadKernel::kD1D &&
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_LIMITER_COOPERATIVE", true);
  const mfem::QuadratureInterpolator* current_volume_interpolator = nullptr;
  int current_volume_nq = 0;
  if (requested_mfem_volume_points)
  {
    const mfem::IntegrationRule& volume_rule =
      mfem::IntRules.Get(mesh.GetElementGeometry(0), qord);
    current_volume_nq = volume_rule.GetNPoints();
    current_volume_interpolator = vfes.GetQuadratureInterpolator(volume_rule);
    MFEM_VERIFY(current_volume_interpolator != nullptr,
                "Device positivity limiter requires volume interpolation.");
  }
  const bool rebuild =
    !cache_ptr || cache_ptr->mesh != &mesh || cache_ptr->fec != vfes.FEColl() ||
    cache_ptr->restriction != current_restriction ||
    cache_ptr->volume_interpolator != current_volume_interpolator ||
    cache_ptr->mesh_sequence != mesh.GetSequence() ||
    cache_ptr->qord != qord || cache_ptr->ne != ne ||
    cache_ptr->dof != dof || cache_ptr->num_eq != num_eq ||
    cache_ptr->face_qords != face_qords ||
    cache_ptr->collocated_lgl_average !=
      requested_collocated_lgl_average ||
    cache_ptr->mfem_volume_points != requested_mfem_volume_points ||
    cache_ptr->mfem_face_points != requested_mfem_face_points ||
    cache_ptr->enforce_nodes != requested_enforce_nodes ||
    cache_ptr->enforce_vtk_points != requested_enforce_vtk_points ||
    cache_ptr->compact_reference_requested != requested_compact_reference_points;
  if (!rebuild)
  {
    return *cache_ptr;
  }

  if (cache_ptr && mfem::Device::Allows(mfem::Backend::DEVICE_MASK))
  {
    // Rebuilding replaces vectors/restrictions owned by the previous cache.
    // In short-lived test/case setup this can happen immediately after an
    // asynchronous limiter apply, so wait before releasing device buffers.
    MFEM_DEVICE_SYNC;
  }
  cache_ptr = std::make_unique<DeviceLimiterCache>();
  DeviceLimiterCache& cache = *cache_ptr;
  cache.mesh = &mesh;
  cache.fec = vfes.FEColl();
  cache.mesh_sequence = mesh.GetSequence();
  cache.qord = qord;
  cache.ne = ne;
  cache.dof = dof;
  cache.dim = mesh.Dimension();
  cache.num_eq = num_eq;
  cache.face_qords = face_qords;
  cache.collocated_lgl_average = requested_collocated_lgl_average;
  cache.restriction = current_restriction;
  MFEM_VERIFY(cache.restriction != nullptr,
              "Device positivity limiter requires an element restriction.");

  cache.mfem_volume_points = requested_mfem_volume_points;
  cache.mfem_face_points = requested_mfem_face_points;
  cache.enforce_nodes = requested_enforce_nodes;
  cache.enforce_vtk_points = requested_enforce_vtk_points;
  cache.compact_reference_requested = requested_compact_reference_points;
  cache.shared_reference_points = requested_compact_reference_points;
  if (cache.shared_reference_points)
  {
    // With MFEM volume/face QI, the remaining points are reference FE nodes
    // and optional reference VTK points.  Reuse their exact basis data only
    // when every cell uses the identical FE object. Keep logical per-cell
    // offsets so point counts and diagnostic semantics retain the full union.
    const mfem::FiniteElement* reference_fe = vfes.GetFE(0);
    for (int e = 1; e < ne; ++e)
    {
      if (vfes.GetFE(e) != reference_fe)
      {
        cache.shared_reference_points = false;
        break;
      }
    }
  }
  if (cache.mfem_volume_points)
  {
    cache.volume_nq = current_volume_nq;
    cache.volume_interpolator = current_volume_interpolator;
    cache.volume_interpolator->SetOutputLayout(mfem::QVectorLayout::byNODES);
    cache.volume_interpolator->DisableTensorProducts(false);
  }

  if (cache.mfem_face_points)
  {
    cache.face_scalar_fes = std::make_unique<mfem::ParFiniteElementSpace>(
      vfes.GetParMesh(), vfes.FEColl(), 1, mfem::Ordering::byNODES);
    cache.interior_faces = mesh.GetNFbyType(mfem::FaceType::Interior);
    cache.boundary_faces = mesh.GetNFbyType(mfem::FaceType::Boundary);
    cache.face_dof = cache.face_scalar_fes->GetTypicalTraceElement()->GetDof();
    cache.interior_face_restriction =
      cache.face_scalar_fes->GetFaceRestriction(
        mfem::ElementDofOrdering::LEXICOGRAPHIC,
        mfem::FaceType::Interior,
        mfem::L2FaceValues::DoubleValued);
    cache.boundary_face_restriction =
      cache.face_scalar_fes->GetFaceRestriction(
        mfem::ElementDofOrdering::LEXICOGRAPHIC,
        mfem::FaceType::Boundary,
        mfem::L2FaceValues::SingleValued);
    MFEM_VERIFY(cache.interior_face_restriction != nullptr &&
                  cache.boundary_face_restriction != nullptr,
                "Device positivity limiter requires face restrictions.");
    cache.interior_face_qi.resize(face_qords.size(), nullptr);
    cache.boundary_face_qi.resize(face_qords.size(), nullptr);
    cache.interior_side0_plans.resize(face_qords.size());
    cache.interior_side1_plans.resize(face_qords.size());
    cache.boundary_plans.resize(face_qords.size());
    for (std::size_t qi = 0; qi < face_qords.size(); ++qi)
    {
      const mfem::IntegrationRule& face_rule =
        mfem::IntRules.Get(mesh.GetTypicalFaceGeometry(), face_qords[qi]);
      cache.interior_face_qi[qi] =
        cache.face_scalar_fes->GetFaceQuadratureInterpolator(
          face_rule, mfem::FaceType::Interior);
      cache.boundary_face_qi[qi] =
        cache.face_scalar_fes->GetFaceQuadratureInterpolator(
          face_rule, mfem::FaceType::Boundary);
      MFEM_VERIFY(cache.interior_face_qi[qi] != nullptr &&
                    cache.boundary_face_qi[qi] != nullptr,
                  "Device positivity limiter requires face interpolation.");
      cache.interior_face_qi[qi]->SetOutputLayout(mfem::QVectorLayout::byNODES);
      cache.boundary_face_qi[qi]->SetOutputLayout(mfem::QVectorLayout::byNODES);
      cache.interior_face_qi[qi]->DisableTensorProducts(false);
      cache.boundary_face_qi[qi]->DisableTensorProducts(false);
      const int nq = face_rule.GetNPoints();
      cache.interior_side0_plans[qi].nq = nq;
      cache.interior_side0_plans[qi].num_faces = cache.interior_faces;
      cache.interior_side1_plans[qi].nq = nq;
      cache.interior_side1_plans[qi].num_faces = cache.interior_faces;
      cache.boundary_plans[qi].nq = nq;
      cache.boundary_plans[qi].num_faces = cache.boundary_faces;
    }
  }

  cache.average_weights.SetSize(ne * dof);
  cache.point_offsets.SetSize(ne + 1);
  if (cache.mfem_face_points)
  {
    cache.face_point_offsets.SetSize(ne + 1);
  }
  std::vector<double> point_shapes;
  mfem::Vector shape;
  cache.point_offsets[0] = 0;
  if (cache.mfem_face_points)
  {
    cache.face_point_offsets[0] = 0;
  }
  const auto& elem_faces = GetElementFaces(mesh);
  int mfem_face_point_count = 0;
  const auto* interior_face_indices =
    cache.mfem_face_points
      ? &mesh.GetInvFaceIndices(mfem::FaceType::Interior)
      : nullptr;
  const auto* boundary_face_indices =
    cache.mfem_face_points
      ? &mesh.GetInvFaceIndices(mfem::FaceType::Boundary)
      : nullptr;
  for (int e = 0; e < ne; ++e)
  {
    const mfem::FiniteElement& fe = *vfes.GetFE(e);
    MFEM_VERIFY(fe.GetDof() == dof,
                "Device positivity limiter does not support variable order.");
    mfem::ElementTransformation& transformation =
      *vfes.GetElementTransformation(e);
    const mfem::IntegrationRule& volume_rule =
      mfem::IntRules.Get(transformation.GetGeometryType(), qord);
    const mfem::IntegrationRule& average_rule = ZhangShuAverageRule(
      vfes, fe, transformation, qord);
    shape.SetSize(dof);
    mfem::Vector integral(dof);
    integral = 0.0;
    double volume = 0.0;
    for (int q = 0; q < average_rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = average_rule.IntPoint(q);
      transformation.SetIntPoint(&ip);
      fe.CalcShape(ip, shape);
      const double weight = ip.weight * transformation.Weight();
      volume += weight;
      integral.Add(weight, shape);
    }
    if (!cache.mfem_volume_points)
    {
      // Average and positivity-point rules intentionally differ for DGSEM:
      // retain the caller-selected volume points while using the collocated
      // GLL rule only for the mean that defines the contraction.
      for (int q = 0; q < volume_rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& ip = volume_rule.IntPoint(q);
        fe.CalcShape(ip, shape);
        for (int d = 0; d < dof; ++d)
        {
          point_shapes.push_back(shape(NativeDof(fe, d)));
        }
      }
    }
    MFEM_VERIFY(volume > 0.0, "Element has non-positive volume.");
    for (int d = 0; d < dof; ++d)
    {
      cache.average_weights[e * dof + d] =
        integral(NativeDof(fe, d)) / volume;
    }

    for (const int face : elem_faces[e])
    {
      if (cache.mfem_face_points)
      {
        const mfem::Mesh::FaceInformation face_info =
          mesh.GetFaceInformation(face);
        if (face_info.IsInterior())
        {
          const auto it = interior_face_indices->find(face);
          MFEM_VERIFY(it != interior_face_indices->end(),
                      "Interior face is missing from MFEM face ordering.");
          const int face_index = it->second;
          const bool side0 =
            face_info.element[0].location ==
              mfem::Mesh::ElementLocation::Local &&
            face_info.element[0].index == e;
          const bool side1 =
            face_info.element[1].location ==
              mfem::Mesh::ElementLocation::Local &&
            face_info.element[1].index == e;
          MFEM_VERIFY(side0 || side1,
                      "Element face is missing from MFEM face information.");
          for (std::size_t qi = 0; qi < face_qords.size(); ++qi)
          {
            auto& plan = side0 ? cache.interior_side0_plans[qi]
                               : cache.interior_side1_plans[qi];
            const int nq = plan.nq;
            for (int q = 0; q < nq; ++q)
            {
              plan.dst.Append(mfem_face_point_count++);
              plan.src.Append(q + nq * face_index);
            }
          }
        }
        else if (face_info.IsBoundary())
        {
          const auto it = boundary_face_indices->find(face);
          MFEM_VERIFY(it != boundary_face_indices->end(),
                      "Boundary face is missing from MFEM face ordering.");
          const int face_index = it->second;
          MFEM_VERIFY(face_info.element[0].index == e,
                      "Boundary face is attached to an unexpected element.");
          for (std::size_t qi = 0; qi < face_qords.size(); ++qi)
          {
            auto& plan = cache.boundary_plans[qi];
            const int nq = plan.nq;
            for (int q = 0; q < nq; ++q)
            {
              plan.dst.Append(mfem_face_point_count++);
              plan.src.Append(q + nq * face_index);
            }
          }
        }
      }
      else
      {
        mfem::FaceElementTransformations* face_transformation =
          mesh.GetFaceElementTransformations(face);
        if (face_transformation == nullptr)
        {
          continue;
        }
        int e1 = -1;
        int e2 = -1;
        mesh.GetFaceElements(face, &e1, &e2);
        const bool first = e1 == e;
        if (!first && e2 != e)
        {
          continue;
        }
        for (const int face_qord : face_qords)
        {
          const mfem::IntegrationRule& face_rule =
            mfem::IntRules.Get(face_transformation->GetGeometryType(),
                               face_qord);
          for (int q = 0; q < face_rule.GetNPoints(); ++q)
          {
            const mfem::IntegrationPoint& face_ip = face_rule.IntPoint(q);
            face_transformation->SetAllIntPoints(&face_ip);
            const mfem::IntegrationPoint& element_ip =
              first ? face_transformation->GetElement1IntPoint()
                    : face_transformation->GetElement2IntPoint();
            fe.CalcShape(element_ip, shape);
            for (int d = 0; d < dof; ++d)
            {
              point_shapes.push_back(shape(NativeDof(fe, d)));
            }
          }
        }
      }
    }

    if (!cache.shared_reference_points || e == 0)
    {
      if (cache.enforce_nodes)
      {
        const mfem::IntegrationRule& nodes = fe.GetNodes();
        for (int q = 0; q < nodes.GetNPoints(); ++q)
        {
          fe.CalcShape(nodes.IntPoint(q), shape);
          for (int d = 0; d < dof; ++d)
          {
            point_shapes.push_back(shape(NativeDof(fe, d)));
          }
        }
      }
      if (cache.enforce_vtk_points)
      {
        const mfem::RefinedGeometry* refined =
          mfem::GlobGeometryRefiner.Refine(fe.GetGeomType(), fe.GetOrder(), 1);
        for (int q = 0; q < refined->RefPts.GetNPoints(); ++q)
        {
          fe.CalcShape(refined->RefPts.IntPoint(q), shape);
          for (int d = 0; d < dof; ++d)
          {
            point_shapes.push_back(shape(NativeDof(fe, d)));
          }
        }
      }
    }
    const int stored_points = static_cast<int>(point_shapes.size()) / dof;
    cache.point_offsets[e + 1] = cache.shared_reference_points
                                 ? (e + 1) * stored_points : stored_points;
    if (cache.mfem_face_points)
    {
      cache.face_point_offsets[e + 1] = mfem_face_point_count;
    }
  }

  cache.d1d = ne > 0 ? vfes.GetFE(0)->GetOrder() + 1 : 0;
  cache.tensor_points = ne > 0 &&
                        dynamic_cast<const mfem::TensorBasisElement*>(
                          vfes.GetFE(0)) != nullptr &&
                        BuildTensorPointFactors(cache, point_shapes);
  if (cache.shared_reference_points && !cache.tensor_points)
  {
    // A failed tensor factorization must retain the complete serial fallback
    // table. This is cold setup only; no state-dependent result is cached.
    const std::vector<double> reference_shapes(point_shapes);
    point_shapes.reserve(reference_shapes.size() * ne);
    for (int e = 1; e < ne; ++e)
    {
      point_shapes.insert(point_shapes.end(), reference_shapes.begin(),
                           reference_shapes.end());
    }
    cache.shared_reference_points = false;
  }
  cache.point_shapes.SetSize(cache.tensor_points
                               ? 0
                               : static_cast<int>(point_shapes.size()));
  for (int i = 0; i < cache.point_shapes.Size(); ++i)
  {
    cache.point_shapes[i] = point_shapes[i];
  }
  if (cache.mfem_face_points)
  {
    for (auto& plan : cache.interior_side0_plans)
    {
      PrepareFaceCopyPlan(plan);
    }
    for (auto& plan : cache.interior_side1_plans)
    {
      PrepareFaceCopyPlan(plan);
    }
    for (auto& plan : cache.boundary_plans)
    {
      PrepareFaceCopyPlan(plan);
    }
  }
  cache.element_state.SetSize(cache.restriction->Height());
  cache.volume_values.SetSize(cache.mfem_volume_points
                                ? ne * cache.volume_nq * num_eq
                                : 0);
  cache.face_values.SetSize(cache.mfem_face_points
                              ? mfem_face_point_count * num_eq
                              : 0);
  cache.averages.SetSize(ne * num_eq);
  cache.theta_rho.SetSize(ne);
  cache.theta_p.SetSize(ne);
  cache.active.SetSize(ne);
  cache.inadmissible_average.SetSize(ne);
  cache.rejected_element.SetSize(ne);
  cache.collapsed_average.SetSize(ne);
  cache.average_repair.SetSize(ne * num_eq);
  if (cache.mfem_face_points)
  {
    cache.face_component_lvec.SetSize(vfes.GetNDofs());
    cache.interior_face_double.SetSize(cache.interior_face_restriction
                                         ? cache.interior_face_restriction
                                             ->Height()
                                         : 0);
    cache.interior_face_side0.SetSize(cache.face_dof * cache.interior_faces);
    cache.interior_face_side1.SetSize(cache.face_dof * cache.interior_faces);
    cache.boundary_face_evec.SetSize(cache.boundary_face_restriction
                                       ? cache.boundary_face_restriction
                                           ->Height()
                                       : 0);
    int max_face_q_size = 0;
    for (const auto& plan : cache.interior_side0_plans)
    {
      max_face_q_size = std::max(max_face_q_size,
                                 plan.nq * plan.num_faces);
    }
    for (const auto& plan : cache.boundary_plans)
    {
      max_face_q_size = std::max(max_face_q_size,
                                 plan.nq * plan.num_faces);
    }
    cache.face_q_values.SetSize(max_face_q_size);
  }
  cache.average_weights.UseDevice(true);
  cache.point_offsets.UseDevice();
  cache.face_point_offsets.UseDevice();
  cache.point_shapes.UseDevice(true);
  cache.point_factors.UseDevice(true);
  cache.volume_values.UseDevice(true);
  cache.face_values.UseDevice(true);
  cache.element_state.UseDevice(true);
  cache.averages.UseDevice(true);
  cache.theta_rho.UseDevice(true);
  cache.theta_p.UseDevice(true);
  cache.active.UseDevice(true);
  cache.inadmissible_average.UseDevice(true);
  cache.rejected_element.UseDevice(true);
  cache.collapsed_average.UseDevice(true);
  cache.average_repair.UseDevice(true);
  cache.face_component_lvec.UseDevice(true);
  cache.interior_face_double.UseDevice(true);
  cache.interior_face_side0.UseDevice(true);
  cache.interior_face_side1.UseDevice(true);
  cache.boundary_face_evec.UseDevice(true);
  cache.face_q_values.UseDevice(true);
  return cache;
}

void PopulateMfemFaceLimiterValues(const mfem::ParFiniteElementSpace& vfes,
                                   const mfem::Vector& U,
                                   int num_eq,
                                   DeviceLimiterCache& cache)
{
  if (!cache.mfem_face_points)
  {
    return;
  }

  const int ndofs = vfes.GetNDofs();
  const int total_face_points =
    cache.ne > 0 ? cache.face_point_offsets[cache.ne] : 0;
  if (total_face_points <= 0)
  {
    return;
  }

  cache.face_values = 0.0;
  for (int eq = 0; eq < num_eq; ++eq)
  {
    ExtractStateComponent(U, ndofs, num_eq, eq, cache.face_component_lvec);

    if (cache.interior_faces > 0)
    {
      cache.interior_face_restriction->Mult(cache.face_component_lvec,
                                            cache.interior_face_double);
      SplitDoubleValuedScalarFaceEVector(cache.interior_face_double,
                                         cache.face_dof,
                                         cache.interior_faces,
                                         cache.interior_face_side0,
                                         cache.interior_face_side1);
      for (std::size_t qi = 0; qi < cache.interior_face_qi.size(); ++qi)
      {
        const int q_size =
          cache.interior_side0_plans[qi].nq *
          cache.interior_side0_plans[qi].num_faces;
        if (q_size <= 0)
        {
          continue;
        }
        cache.face_q_values.SetSize(q_size);
        cache.interior_face_qi[qi]->SetOutputLayout(
          mfem::QVectorLayout::byNODES);
        cache.interior_face_qi[qi]->Values(cache.interior_face_side0,
                                           cache.face_q_values);
        CopyFaceQValuesToElementMajor(cache.face_q_values,
                                      cache.interior_side0_plans[qi],
                                      total_face_points,
                                      eq,
                                      cache.face_values);
        cache.interior_face_qi[qi]->Values(cache.interior_face_side1,
                                           cache.face_q_values);
        CopyFaceQValuesToElementMajor(cache.face_q_values,
                                      cache.interior_side1_plans[qi],
                                      total_face_points,
                                      eq,
                                      cache.face_values);
      }
    }

    if (cache.boundary_faces > 0)
    {
      cache.boundary_face_restriction->Mult(cache.face_component_lvec,
                                            cache.boundary_face_evec);
      for (std::size_t qi = 0; qi < cache.boundary_face_qi.size(); ++qi)
      {
        const int q_size =
          cache.boundary_plans[qi].nq * cache.boundary_plans[qi].num_faces;
        if (q_size <= 0)
        {
          continue;
        }
        cache.face_q_values.SetSize(q_size);
        cache.boundary_face_qi[qi]->SetOutputLayout(
          mfem::QVectorLayout::byNODES);
        cache.boundary_face_qi[qi]->Values(cache.boundary_face_evec,
                                           cache.face_q_values);
        CopyFaceQValuesToElementMajor(cache.face_q_values,
                                      cache.boundary_plans[qi],
                                      total_face_points,
                                      eq,
                                      cache.face_values);
      }
    }
  }
}

void ApplyDeviceZhangShuPositivityLimiter(
  mfem::ParFiniteElementSpace& vfes,
  mfem::ParGridFunction& U,
  int dim,
  double gamma,
  double rho_min,
  double p_min,
  int qord,
  const std::vector<int>& face_qords,
  int p_bisect_iters,
  mfem::ZhangShuPointSet point_set,
  bool enforce_vtk_points,
  mfem::ZhangShuLimiterElementDiagnostics* diagnostics,
  bool* local_state_modified)
{
  ::igrlab::NVTXRange limiter_range("Limiter/Zhang-Shu");
  LimiterTimingState& timing = GetLimiterTimingState();
  if (timing.enabled) { MFEM_DEVICE_SYNC; }
  const auto timing_start = std::chrono::steady_clock::now();

  const int num_eq = dim + 2;
  DeviceLimiterCache& cache =
    GetDeviceLimiterCache(vfes,
                          num_eq,
                          qord,
                          face_qords,
                          point_set,
                          enforce_vtk_points);
  cache.restriction->Mult(U, cache.element_state);
  if (cache.mfem_volume_points)
  {
    cache.volume_interpolator->SetOutputLayout(mfem::QVectorLayout::byNODES);
    cache.volume_interpolator->DisableTensorProducts(false);
    cache.volume_interpolator->Values(cache.element_state,
                                      cache.volume_values);
  }
  PopulateMfemFaceLimiterValues(vfes, U, num_eq, cache);

  const int ne = cache.ne;
  const int dof = cache.dof;
  const double* weights = cache.average_weights.Read();
  const int* offsets = cache.point_offsets.Read();
  const int* face_offsets =
    cache.mfem_face_points ? cache.face_point_offsets.Read() : nullptr;
  const double* shapes = cache.point_shapes.Read();
  const double* factors = cache.point_factors.Read();
  const double* volume_values = cache.volume_values.Read();
  const double* face_values = cache.face_values.Read();
  const bool tensor_points = cache.tensor_points;
  const bool mfem_volume_points = cache.mfem_volume_points;
  const bool mfem_face_points = cache.mfem_face_points;
  const bool capture_diagnostics = diagnostics != nullptr;
  const int volume_nq = cache.volume_nq;
  const int total_face_points =
    cache.mfem_face_points && ne > 0 ? cache.face_point_offsets[ne] : 0;
  const int d1d = cache.d1d;
  double* state = cache.element_state.ReadWrite();
  double* averages = cache.averages.Write();
  double* theta_rho = cache.theta_rho.Write();
  double* theta_p = cache.theta_p.Write();
  double* active = cache.active.Write();
  double* inadmissible_average = cache.inadmissible_average.Write();
  double* rejected_element = cache.rejected_element.Write();
  double* collapsed_average = cache.collapsed_average.Write();
  double* average_repair = cache.average_repair.Write();

  // Retain a same-binary serial baseline and a fallback for other orders,
  // dimensions, and non-tensor elements.  This switch changes execution only;
  // point sets, average rules, finite admission, and pressure solves are shared.
  const bool cooperative = dim == 2 && tensor_points &&
    d1d == CooperativeZhangShuQuadKernel::kD1D &&
    ::igrlab::EnvFlagEnabledDefault("DG_IGR_LIMITER_COOPERATIVE", true);
  if (cooperative)
  {
    CooperativeZhangShuQuadKernel kernel;
    kernel.ne = ne;
    kernel.volume_nq = mfem_volume_points ? volume_nq : 0;
    kernel.total_face_points = total_face_points;
    kernel.p_bisect_iters = p_bisect_iters;
    kernel.gamma = gamma;
    kernel.rho_min = rho_min;
    kernel.p_min = p_min;
    kernel.capture_diagnostics = capture_diagnostics;
    kernel.shared_reference_points = cache.shared_reference_points;
    kernel.weights = weights;
    kernel.offsets = offsets;
    kernel.face_offsets = face_offsets;
    kernel.factors = factors;
    kernel.volume_values = volume_values;
    kernel.face_values = face_values;
    kernel.state = state;
    kernel.averages = averages;
    kernel.theta_rho = theta_rho;
    kernel.theta_p = theta_p;
    kernel.active = active;
    kernel.inadmissible_average = inadmissible_average;
    kernel.rejected_element = rejected_element;
    kernel.collapsed_average = collapsed_average;
    kernel.average_repair = average_repair;
    mfem::forall_2D(ne, CooperativeZhangShuQuadKernel::kLanes, 1, kernel);
  }
  else
  {
  mfem::forall(ne, [=] MFEM_HOST_DEVICE(int e)
  {
    double average[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (int d = 0; d < dof; ++d)
    {
      for (int c = 0; c < num_eq; ++c)
      {
        average[c] += weights[e * dof + d] *
                      state[d + dof * (c + num_eq * e)];
      }
    }
    if (capture_diagnostics)
    {
      for (int c = 0; c < num_eq; ++c)
      {
        average_repair[e + ne * c] = 0.0;
      }
      collapsed_average[e] = 0.0;
    }

    bool average_is_finite = true;
    for (int c = 0; c < num_eq; ++c)
    {
      average_is_finite = average_is_finite && std::isfinite(average[c]);
    }
    const double rho_bar = average[0];
    double momentum_sq = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      momentum_sq += average[1 + d] * average[1 + d];
    }
    const double kinetic = average_is_finite && rho_bar > rho_min
                             ? 0.5 * momentum_sq / rho_bar
                             : 0.0;
    const double pressure_bar =
      (gamma - 1.0) * (average[1 + dim] - kinetic);
    const bool average_is_admissible =
      average_is_finite && rho_bar > rho_min &&
      std::isfinite(pressure_bar) && pressure_bar > p_min;
    inadmissible_average[e] = average_is_admissible ? 0.0 : 1.0;
    bool consumer_points_are_finite = true;
    if (!average_is_admissible)
    {
      theta_rho[e] = 1.0;
      theta_p[e] = 1.0;
      active[e] = 0.0;
    }
    else
    {
      double rho_theta = 1.0;
      if (mfem_volume_points)
      {
        for (int q = 0; q < volume_nq; ++q)
        {
          double point_state[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
          for (int c = 0; c < num_eq; ++c)
          {
            point_state[c] =
              volume_values[q + volume_nq * (c + num_eq * e)];
          }
          const bool point_state_is_finite =
            EulerConsumerPointIsFinite(point_state, dim, gamma);
          consumer_points_are_finite =
            consumer_points_are_finite && point_state_is_finite;
          if (!point_state_is_finite)
          {
            continue;
          }
          const double rho = point_state[0];
          if (rho < rho_min)
          {
            const double denominator = average[0] - rho;
            const double candidate = denominator > 0.0
                                       ? (average[0] - rho_min) / denominator
                                       : 0.0;
            rho_theta = candidate < rho_theta ? candidate : rho_theta;
          }
        }
      }
      if (mfem_face_points)
      {
        for (int point = face_offsets[e]; point < face_offsets[e + 1]; ++point)
        {
          double point_state[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
          for (int c = 0; c < num_eq; ++c)
          {
            point_state[c] =
              face_values[point + total_face_points * c];
          }
          const bool point_state_is_finite =
            EulerConsumerPointIsFinite(point_state, dim, gamma);
          consumer_points_are_finite =
            consumer_points_are_finite && point_state_is_finite;
          if (!point_state_is_finite)
          {
            continue;
          }
          const double rho = point_state[0];
          if (rho < rho_min)
          {
            const double denominator = average[0] - rho;
            const double candidate = denominator > 0.0
                                       ? (average[0] - rho_min) / denominator
                                       : 0.0;
            rho_theta = candidate < rho_theta ? candidate : rho_theta;
          }
        }
      }
      for (int point = offsets[e]; point < offsets[e + 1]; ++point)
      {
        double point_state[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        for (int d = 0; d < dof; ++d)
        {
          const double basis = LimiterPointBasis(shapes,
                                                 factors,
                                                 tensor_points,
                                                 point,
                                                 d,
                                                 dof,
                                                 dim,
                                                 d1d);
          for (int c = 0; c < num_eq; ++c)
          {
            point_state[c] +=
              basis * state[d + dof * (c + num_eq * e)];
          }
        }
        const bool point_state_is_finite =
          EulerConsumerPointIsFinite(point_state, dim, gamma);
        consumer_points_are_finite =
          consumer_points_are_finite && point_state_is_finite;
        if (!point_state_is_finite)
        {
          continue;
        }
        const double rho = point_state[0];
        if (rho < rho_min)
        {
          const double denominator = average[0] - rho;
          const double candidate = denominator > 0.0
                                     ? (average[0] - rho_min) / denominator
                                     : 0.0;
          rho_theta = candidate < rho_theta ? candidate : rho_theta;
        }
      }
      if (rho_theta < 1.0)
      {
        for (int d = 0; d < dof; ++d)
        {
          for (int c = 0; c < num_eq; ++c)
          {
            const int index = d + dof * (c + num_eq * e);
            state[index] = average[c] +
                           rho_theta * (state[index] - average[c]);
          }
        }
      }

      double pressure_theta = 1.0;
      if (mfem_volume_points)
      {
        for (int q = 0; q < volume_nq; ++q)
        {
          double point_state[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
          for (int c = 0; c < num_eq; ++c)
          {
            const double original =
              volume_values[q + volume_nq * (c + num_eq * e)];
            point_state[c] = average[c] +
                             rho_theta * (original - average[c]);
          }
          const bool point_state_is_finite =
            EulerConsumerPointIsFinite(point_state, dim, gamma);
          consumer_points_are_finite =
            consumer_points_are_finite && point_state_is_finite;
          if (!point_state_is_finite)
          {
            continue;
          }
          if (DevicePressure(point_state, dim, gamma) < p_min)
          {
            const double candidate =
              DeviceThetaForPressure(average,
                                     point_state,
                                     dim,
                                     gamma,
                                     p_min,
                                     1.0,
                                     p_bisect_iters);
            pressure_theta = candidate < pressure_theta
                               ? candidate
                               : pressure_theta;
          }
        }
      }
      if (mfem_face_points)
      {
        for (int point = face_offsets[e]; point < face_offsets[e + 1]; ++point)
        {
          double point_state[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
          for (int c = 0; c < num_eq; ++c)
          {
            const double original =
              face_values[point + total_face_points * c];
            point_state[c] = average[c] +
                             rho_theta * (original - average[c]);
          }
          const bool point_state_is_finite =
            EulerConsumerPointIsFinite(point_state, dim, gamma);
          consumer_points_are_finite =
            consumer_points_are_finite && point_state_is_finite;
          if (!point_state_is_finite)
          {
            continue;
          }
          if (DevicePressure(point_state, dim, gamma) < p_min)
          {
            const double candidate =
              DeviceThetaForPressure(average,
                                     point_state,
                                     dim,
                                     gamma,
                                     p_min,
                                     1.0,
                                     p_bisect_iters);
            pressure_theta = candidate < pressure_theta
                               ? candidate
                               : pressure_theta;
          }
        }
      }
      for (int point = offsets[e]; point < offsets[e + 1]; ++point)
      {
        double point_state[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        for (int d = 0; d < dof; ++d)
        {
          const double basis = LimiterPointBasis(shapes,
                                                 factors,
                                                 tensor_points,
                                                 point,
                                                 d,
                                                 dof,
                                                 dim,
                                                 d1d);
          for (int c = 0; c < num_eq; ++c)
          {
            point_state[c] += basis *
                              state[d + dof * (c + num_eq * e)];
          }
        }
        const bool point_state_is_finite =
          EulerConsumerPointIsFinite(point_state, dim, gamma);
        consumer_points_are_finite =
          consumer_points_are_finite && point_state_is_finite;
        if (!point_state_is_finite)
        {
          continue;
        }
        if (DevicePressure(point_state, dim, gamma) < p_min)
        {
          const double candidate =
            DeviceThetaForPressure(average,
                                   point_state,
                                   dim,
                                   gamma,
                                   p_min,
                                   1.0,
                                   p_bisect_iters);
          pressure_theta = candidate < pressure_theta
                             ? candidate
                             : pressure_theta;
        }
      }
      if (pressure_theta < 1.0)
      {
        for (int d = 0; d < dof; ++d)
        {
          for (int c = 0; c < num_eq; ++c)
          {
            const int index = d + dof * (c + num_eq * e);
            state[index] = average[c] +
                           pressure_theta * (state[index] - average[c]);
          }
        }
      }
      theta_rho[e] = rho_theta;
      theta_p[e] = pressure_theta;
      active[e] = (rho_theta < 1.0 || pressure_theta < 1.0) ? 1.0 : 0.0;
    }
    rejected_element[e] =
      (average_is_admissible && consumer_points_are_finite) ? 0.0 : 1.0;
    for (int c = 0; c < num_eq; ++c)
    {
      averages[c + num_eq * e] = average[c];
    }
  });

  }

  LimiterAdmissionCounts counts{0, 0};
  if (cooperative)
  {
    mfem::reduce(ne,
                 counts,
                 LimiterAdmissionReductionBody{rejected_element, active},
                 LimiterAdmissionReducer{},
                 true,
                 cache.admission_counts_workspace);
  }
  const int local_rejected_elements = cooperative
    ? counts.rejected
    : static_cast<int>(CountActiveElements(
        &cache.rejected_element, cache.admissibility_reduce_workspace));
  const int global_rejected_elements =
    GlobalRejectedElementCount(vfes, local_rejected_elements);
  if (global_rejected_elements == 0 && (!cooperative || counts.active > 0))
  {
    cache.restriction->MultTranspose(cache.element_state, U);
  }
  if (diagnostics != nullptr)
  {
    CopyDeviceVector(cache.theta_rho, diagnostics->theta_rho);
    CopyDeviceVector(cache.theta_p, diagnostics->theta_p);
    CopyDeviceVector(cache.active, diagnostics->active);
    CopyDeviceVector(cache.inadmissible_average,
                     diagnostics->inadmissible_average);
    CopyDeviceVector(cache.collapsed_average,
                     diagnostics->collapsed_average);
    CopyDeviceVector(cache.average_repair, diagnostics->average_repair);
  }

  if (timing.enabled) { MFEM_DEVICE_SYNC; }
  // A rejected mean or non-finite consumer point commits no element update.
  // Match the host path by aggregating only completed limiter applications;
  // rejected attempts remain visible in the explicit-time summary and wall
  // clock.
  if (timing.enabled && global_rejected_elements == 0)
  {
    const double seconds =
      std::chrono::duration<double>(
        std::chrono::steady_clock::now() - timing_start)
        .count();
    RecordLimiterTiming(seconds,
                        cache,
                        &cache.active,
                        &cache.theta_rho,
                        &cache.theta_p,
                        vfes.GetComm(),
                        cooperative ? counts.active : -1.0);
  }
  if (global_rejected_elements > 0)
  {
    throw mfem::RecoverableLimiterFailure(global_rejected_elements);
  }
  if (local_state_modified != nullptr)
  {
    *local_state_modified = cooperative
      ? counts.active > 0
      : CountActiveElements(&cache.active,
                            cache.admissibility_reduce_workspace) > 0.0;
  }
}

} // namespace

namespace mfem
{

void ApplyZhangShuPositivityLimiter(ParFiniteElementSpace& vfes,
                                    ParGridFunction& U,
                                    int dim,
                                    double gamma,
                                    double rho_min,
                                    double p_min,
                                    int quad_order,
                                    const std::vector<int>& face_quad_orders,
                                    int p_bisect_iters,
                                    ZhangShuPointSet point_set,
                                    ZhangShuLimiterElementDiagnostics* diagnostics,
                                    std::optional<bool> enforce_vtk_points,
                                    bool* local_state_modified)
{
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Positivity limiter assumes byNODES ordering.");
  if (local_state_modified != nullptr)
  {
    // A rejected stage commits no conservative-state update.  Initialize the
    // optional result accordingly before any collective admission can throw.
    *local_state_modified = false;
  }

  const int num_eq = dim + 2;
  const int NE = vfes.GetNE();
  Mesh& mesh = *vfes.GetMesh();
  const auto& elem_faces = GetElementFaces(mesh);

  mfem::Array<int> vdofs;
  mfem::Vector elvec;      // element slice (size = dof*num_eq)
  mfem::DenseMatrix elmat; // (dof x num_eq) view of element DOFs
  mfem::Vector shape, Up(num_eq), Ubar(num_eq), row(num_eq);

  const int qord =
    (quad_order > 0) ? quad_order : (2 * vfes.GetFE(0)->GetOrder() + 2);
  const std::vector<int> limiter_face_qords =
    BuildFaceQuadratureOrders(vfes, qord, face_quad_orders);
  const bool effective_enforce_vtk_points =
    enforce_vtk_points.value_or(::igrlab::EnvFlagEnabledDefault(
      "DG_IGR_LIMITER_ENFORCE_VTK_POINTS", false));

  if (diagnostics)
  {
    diagnostics->SetSize(NE, num_eq);
  }

  if (Device::Allows(Backend::DEVICE_MASK))
  {
    MFEM_VERIFY(num_eq <= 5,
                "Device positivity limiter supports at most three dimensions.");
    ApplyDeviceZhangShuPositivityLimiter(vfes,
                                         U,
                                         dim,
                                         gamma,
                                         rho_min,
                                         p_min,
                                         qord,
                                         limiter_face_qords,
                                         p_bisect_iters,
                                         point_set,
                                         effective_enforce_vtk_points,
                                         diagnostics,
                                         local_state_modified);
    return;
  }

  LimiterTimingState& timing = GetLimiterTimingState();
  const auto timing_start = std::chrono::steady_clock::now();
  CpuLimiterPhaseSample phase;
  double minimum_theta = 1.0;
  using PhaseClock = std::chrono::steady_clock;
  auto phase_start = [&]()
  {
    return timing.phase_enabled ? PhaseClock::now() : PhaseClock::time_point{};
  };
  auto phase_add = [&](const PhaseClock::time_point& start, double& total)
  {
    if (timing.phase_enabled)
    {
      total += std::chrono::duration<double>(PhaseClock::now() - start).count();
    }
  };

  mfem::Vector element_averages(NE * num_eq);
  int local_rejected_elements = 0;
  // The host path scatters elements in place below. Preflight the complete
  // configured point union first so every rank reaches the collective
  // decision and a recoverable rejection leaves the full input unchanged.
  for (int i = 0; i < NE; ++i)
  {
    const FiniteElement* fe = vfes.GetFE(i);
    ElementTransformation* Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);
    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    const auto average_start = phase_start();
    ZhangShuElementAverage(vfes,
                           *fe,
                           *Tr,
                           elmat,
                           num_eq,
                           qord,
                           Ubar);
    phase_add(average_start, phase.average_seconds);
    for (int component = 0; component < num_eq; ++component)
    {
      element_averages(component + num_eq * i) = Ubar(component);
    }

    bool average_is_finite = true;
    for (int component = 0; component < num_eq; ++component)
    {
      average_is_finite =
        average_is_finite && std::isfinite(Ubar(component));
    }
    double momentum_sq = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      momentum_sq += Ubar(1 + d) * Ubar(1 + d);
    }
    const double kinetic = average_is_finite && Ubar(0) > rho_min
                             ? 0.5 * momentum_sq / Ubar(0)
                             : 0.0;
    const double pressure_bar =
      (gamma - 1.0) * (Ubar(1 + dim) - kinetic);
    const bool average_is_admissible =
      average_is_finite && Ubar(0) > rho_min &&
      std::isfinite(pressure_bar) && pressure_bar > p_min;
    bool consumer_points_are_finite = true;
    if (average_is_admissible)
    {
      auto check_consumer_point = [&](const mfem::Vector& point_state)
      {
        consumer_points_are_finite =
          consumer_points_are_finite &&
          EulerConsumerPointIsFinite(point_state.GetData(), dim, gamma);
      };

      const IntegrationRule& ir = IntRules.Get(Tr->GetGeometryType(), qord);
      shape.SetSize(dof);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
        const IntegrationPoint& ip = ir.IntPoint(q);
        Tr->SetIntPoint(&ip);
        fe->CalcShape(ip, shape);
        elmat.MultTranspose(shape, Up);
        check_consumer_point(Up);
      }

      for (const int face : elem_faces[i])
      {
        FaceElementTransformations* FTr =
          mesh.GetFaceElementTransformations(face);
        if (FTr == nullptr)
        {
          continue;
        }
        int e1 = -1;
        int e2 = -1;
        mesh.GetFaceElements(face, &e1, &e2);
        const bool elem_is_first = e1 == i;
        const bool elem_is_second = e2 == i;
        if (!elem_is_first && !elem_is_second)
        {
          continue;
        }
        for (const int fqord : limiter_face_qords)
        {
          const IntegrationRule& fir =
            IntRules.Get(FTr->GetGeometryType(), fqord);
          for (int q = 0; q < fir.GetNPoints(); ++q)
          {
            const IntegrationPoint& fip = fir.IntPoint(q);
            FTr->SetAllIntPoints(&fip);
            const IntegrationPoint& eip = elem_is_first
                                             ? FTr->GetElement1IntPoint()
                                             : FTr->GetElement2IntPoint();
            fe->CalcShape(eip, shape);
            elmat.MultTranspose(shape, Up);
            check_consumer_point(Up);
          }
        }
      }

      const auto* fec_l2 =
        dynamic_cast<const L2_FECollection*>(vfes.FEColl());
      const bool enforce_nodes =
        (fec_l2 != nullptr &&
         fec_l2->GetBasisType() == BasisType::GaussLobatto) ||
        point_set == ZhangShuPointSet::kConsumer;
      if (enforce_nodes)
      {
        for (int node = 0; node < dof; ++node)
        {
          elmat.GetRow(node, Up);
          check_consumer_point(Up);
        }
      }

      if (effective_enforce_vtk_points)
      {
        const RefinedGeometry* refined =
          GlobGeometryRefiner.Refine(fe->GetGeomType(), fe->GetOrder(), 1);
        for (int q = 0; q < refined->RefPts.GetNPoints(); ++q)
        {
          fe->CalcShape(refined->RefPts.IntPoint(q), shape);
          elmat.MultTranspose(shape, Up);
          check_consumer_point(Up);
        }
      }
    }

    if (!average_is_admissible || !consumer_points_are_finite)
    {
      ++local_rejected_elements;
      if (diagnostics && !average_is_admissible)
      {
        diagnostics->inadmissible_average(i) = 1.0;
      }
    }
  }

  const int global_rejected_elements =
    GlobalRejectedElementCount(vfes, local_rejected_elements);
  if (global_rejected_elements > 0)
  {
    throw RecoverableLimiterFailure(global_rejected_elements);
  }

  for (int i = 0; i < NE; i++)
  {
    const FiniteElement* fe = vfes.GetFE(i);
    const auto* fec_l2 = dynamic_cast<const L2_FECollection*>(vfes.FEColl());
    const bool consumer_points = point_set == ZhangShuPointSet::kConsumer;
    const bool enforce_nodes =
      (fec_l2 != nullptr && fec_l2->GetBasisType() == BasisType::GaussLobatto) ||
      consumer_points;
    const bool enforce_vtk_points = effective_enforce_vtk_points;
    ElementTransformation* Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);

    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // The collective preflight above established that this mean is admissible.
    // Reuse it so strict validation does not duplicate the average work.
    for (int component = 0; component < num_eq; ++component)
    {
      Ubar(component) = element_averages(component + num_eq * i);
    }

    // --- positivity points (use same IR as average) ---
    const IntegrationRule& ir = IntRules.Get(Tr->GetGeometryType(), qord);
    shape.SetSize(dof);

    // Step 1: density limiting
    double theta_rho = 1.0;
    auto rho_update = [&](const mfem::Vector& state)
    {
      const double rho = state(0);
      if (rho < rho_min)
      {
        const double num = Ubar(0) - rho_min;
        const double den = Ubar(0) - rho;
        if (den > 0.0)
        {
          theta_rho = std::min(theta_rho, num / den);
        }
        else
        {
          theta_rho = 0.0;
        }
      }
    };

    const auto density_volume_start = phase_start();
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up);
      rho_update(Up);
    }
    phase.volume_points += static_cast<double>(ir.GetNPoints());
    phase_add(density_volume_start, phase.density_volume_seconds);

    const auto density_face_start = phase_start();
    for (const int face : elem_faces[i])
    {
      FaceElementTransformations* FTr = mesh.GetFaceElementTransformations(face);
      if (FTr == nullptr)
      {
        continue;
      }

      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(face, &e1, &e2);
      const bool elem_is_first = (e1 == i);
      const bool elem_is_second = (e2 == i);
      if (!elem_is_first && !elem_is_second)
      {
        continue;
      }

      for (const int fqord : limiter_face_qords)
      {
        const IntegrationRule& fir =
          IntRules.Get(FTr->GetGeometryType(), fqord);
        for (int q = 0; q < fir.GetNPoints(); ++q)
        {
          phase.face_points += 1.0;
          const IntegrationPoint& fip = fir.IntPoint(q);
          FTr->SetAllIntPoints(&fip);
          const IntegrationPoint& eip = elem_is_first
                                          ? FTr->GetElement1IntPoint()
                                          : FTr->GetElement2IntPoint();
          fe->CalcShape(eip, shape);
          elmat.MultTranspose(shape, Up);
          rho_update(Up);
        }
      }
    }
    phase_add(density_face_start, phase.density_face_seconds);

    const auto density_node_start = phase_start();
    if (enforce_nodes)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, Up);
        rho_update(Up);
      }
      phase.node_points += static_cast<double>(dof);
    }
    phase_add(density_node_start, phase.density_node_seconds);

    if (enforce_vtk_points)
    {
      const mfem::RefinedGeometry* refined =
        mfem::GlobGeometryRefiner.Refine(fe->GetGeomType(), fe->GetOrder(), 1);
      for (int q = 0; q < refined->RefPts.GetNPoints(); ++q)
      {
        fe->CalcShape(refined->RefPts.IntPoint(q), shape);
        elmat.MultTranspose(shape, Up);
        rho_update(Up);
      }
    }

    const auto density_blend_start = phase_start();
    if (theta_rho < 1.0)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, row);
        igrlab::limiters_detail::BlendRow(row, Ubar, theta_rho);
        elmat.SetRow(j, row);
      }
    }
    phase_add(density_blend_start, phase.density_blend_seconds);

    // Step 2: pressure limiting
    double theta_p = 1.0;
    auto p_update = [&](const mfem::Vector& state)
    {
      const double p = EulerPressureFromConservative(state,
                                                     dim,
                                                     gamma,
                                                     /*rho_floor=*/1e-16);
      if (p < p_min)
      {
        phase.pressure_bisections += 1.0;
        theta_p = std::min(theta_p,
                           ThetaForPressure(Ubar,
                                            state,
                                            dim,
                                            gamma,
                                            p_min,
                                            /*theta_max=*/1.0,
                                            p_bisect_iters));
      }
    };

    const auto pressure_volume_start = phase_start();
    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
      const IntegrationPoint& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      elmat.MultTranspose(shape, Up); // current point state (after rho limit)
      p_update(Up);
    }
    phase_add(pressure_volume_start, phase.pressure_volume_seconds);

    const auto pressure_face_start = phase_start();
    for (const int face : elem_faces[i])
    {
      FaceElementTransformations* FTr = mesh.GetFaceElementTransformations(face);
      if (FTr == nullptr)
      {
        continue;
      }

      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(face, &e1, &e2);
      const bool elem_is_first = (e1 == i);
      const bool elem_is_second = (e2 == i);
      if (!elem_is_first && !elem_is_second)
      {
        continue;
      }

      for (const int fqord : limiter_face_qords)
      {
        const IntegrationRule& fir =
          IntRules.Get(FTr->GetGeometryType(), fqord);
        for (int q = 0; q < fir.GetNPoints(); ++q)
        {
          const IntegrationPoint& fip = fir.IntPoint(q);
          FTr->SetAllIntPoints(&fip);
          const IntegrationPoint& eip = elem_is_first
                                          ? FTr->GetElement1IntPoint()
                                          : FTr->GetElement2IntPoint();
          fe->CalcShape(eip, shape);
          elmat.MultTranspose(shape, Up);
          p_update(Up);
        }
      }
    }
    phase_add(pressure_face_start, phase.pressure_face_seconds);

    const auto pressure_node_start = phase_start();
    if (enforce_nodes)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, Up);
        p_update(Up);
      }
    }
    phase_add(pressure_node_start, phase.pressure_node_seconds);

    if (enforce_vtk_points)
    {
      const mfem::RefinedGeometry* refined =
        mfem::GlobGeometryRefiner.Refine(fe->GetGeomType(), fe->GetOrder(), 1);
      for (int q = 0; q < refined->RefPts.GetNPoints(); ++q)
      {
        fe->CalcShape(refined->RefPts.IntPoint(q), shape);
        elmat.MultTranspose(shape, Up);
        p_update(Up);
      }
    }

    const auto pressure_blend_start = phase_start();
    if (theta_p < 1.0)
    {
      for (int j = 0; j < dof; ++j)
      {
        elmat.GetRow(j, row);
        igrlab::limiters_detail::BlendRow(row, Ubar, theta_p);
        elmat.SetRow(j, row);
      }
    }
    phase_add(pressure_blend_start, phase.pressure_blend_seconds);

    if (theta_rho < 1.0 || theta_p < 1.0)
    {
      phase.active_elements += 1.0;
    }
    minimum_theta = std::min(minimum_theta, theta_rho * theta_p);

    if (diagnostics)
    {
      diagnostics->theta_rho(i) = theta_rho;
      diagnostics->theta_p(i) = theta_p;
      diagnostics->active(i) = (theta_rho < 1.0 || theta_p < 1.0) ? 1.0 : 0.0;
    }

    const auto scatter_start = phase_start();
    U.SetSubVector(vdofs, elmat.GetData());
    phase_add(scatter_start, phase.scatter_seconds);
  }

  if (timing.enabled)
  {
    if (timing.seen_calls++ >= timing.warmup_calls)
    {
      ++timing.calls;
      timing.total_seconds +=
        std::chrono::duration<double>(
          std::chrono::steady_clock::now() - timing_start)
          .count();
      AccumulateLimiterAggregate(timing,
                                 phase.active_elements,
                                 NE,
                                 minimum_theta,
                                 vfes.GetComm());
      timing.avg_tensor_points += 0.0;
      timing.avg_point_count += 0.0;
      if (timing.phase_enabled)
      {
        timing.average_seconds += phase.average_seconds;
        timing.admissible_average_seconds += phase.admissible_average_seconds;
        timing.density_volume_seconds += phase.density_volume_seconds;
        timing.density_face_seconds += phase.density_face_seconds;
        timing.density_node_seconds += phase.density_node_seconds;
        timing.density_blend_seconds += phase.density_blend_seconds;
        timing.pressure_volume_seconds += phase.pressure_volume_seconds;
        timing.pressure_face_seconds += phase.pressure_face_seconds;
        timing.pressure_node_seconds += phase.pressure_node_seconds;
        timing.pressure_blend_seconds += phase.pressure_blend_seconds;
        timing.scatter_seconds += phase.scatter_seconds;
        timing.avg_collapsed_elements += phase.collapsed_elements;
        timing.avg_pressure_bisections += phase.pressure_bisections;
        timing.avg_volume_points += phase.volume_points;
        timing.avg_face_points += phase.face_points;
        timing.avg_node_points += phase.node_points;
      }
    }
  }
  if (local_state_modified != nullptr)
  {
    *local_state_modified = phase.active_elements > 0.0;
  }
}

} // namespace mfem
