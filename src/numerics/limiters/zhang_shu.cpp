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

struct ElementFaceCacheEntry
{
  long mesh_sequence = -1;
  int ne = -1;
  std::vector<std::vector<int>> elem_faces;
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
  bool tensor_points = false;
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

  ~LimiterTimingState()
  {
    if (!enabled || calls <= 0)
    {
      return;
    }
    const double inv_calls = 1.0 / static_cast<double>(calls);
    const int rank =
      (mfem::Mpi::IsInitialized() && !mfem::Mpi::IsFinalized())
        ? mfem::Mpi::WorldRank()
        : 0;
    std::cout << "[limiter] event=run-summary"
              << " rank=" << rank
              << " calls=" << calls
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

LimiterTimingState& GetLimiterTimingState()
{
  static LimiterTimingState state;
  if (!state.initialized)
  {
    state.phase_enabled =
      dg_igr::EnvFlagEnabled("DG_IGR_LIMITER_CPU_PHASE_TIMING");
    state.enabled = dg_igr::EnvFlagEnabled("DG_IGR_LIMITER_TIMING") ||
                    state.phase_enabled;
    if (const char* value = std::getenv("DG_IGR_TIMING_WARMUP_CALLS"))
    {
      state.warmup_calls = std::max(0, std::atoi(value));
    }
    state.initialized = true;
  }
  return state;
}

void RecordLimiterTiming(double seconds,
                         const DeviceLimiterCache& cache,
                         const mfem::Vector* active)
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

  double local_active =
    CountActiveElements(active, timing.active_reduce_workspace);
  double global_active = local_active;
  if (mfem::Mpi::IsInitialized() && !mfem::Mpi::IsFinalized())
  {
    MPI_Allreduce(&local_active,
                  &global_active,
                  1,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_SUM,
                  MPI_COMM_WORLD);
  }

  ++timing.calls;
  timing.total_seconds += seconds;
  timing.avg_active_elements += global_active;
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
  if (!dg_igr::EnvFlagEnabledDefault("DG_IGR_LIMITER_MFEM_FACE_QI", true))
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
  mfem::ZhangShuPointSet point_set)
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
    dg_igr::EnvFlagEnabledDefault("DG_IGR_LIMITER_MFEM_VOLUME_QI", true) &&
    mfem::QuadratureInterpolator::SupportsFESpace(vfes);
  const bool requested_mfem_face_points =
    ne > 0 && SupportsMfemFaceLimiterPath(vfes, face_qords);
  const auto* l2_fec =
    dynamic_cast<const mfem::L2_FECollection*>(vfes.FEColl());
  const bool consumer_points =
    point_set == mfem::ZhangShuPointSet::kConsumer;
  const bool requested_enforce_nodes =
    (l2_fec != nullptr &&
     l2_fec->GetBasisType() == mfem::BasisType::GaussLobatto) ||
    consumer_points;
  const bool requested_enforce_vtk_points =
    dg_igr::EnvFlagEnabledDefault("DG_IGR_LIMITER_ENFORCE_VTK_POINTS", false);
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
    cache_ptr->mfem_volume_points != requested_mfem_volume_points ||
    cache_ptr->mfem_face_points != requested_mfem_face_points ||
    cache_ptr->enforce_nodes != requested_enforce_nodes ||
    cache_ptr->enforce_vtk_points != requested_enforce_vtk_points;
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
  cache.restriction = current_restriction;
  MFEM_VERIFY(cache.restriction != nullptr,
              "Device positivity limiter requires an element restriction.");

  cache.mfem_volume_points = requested_mfem_volume_points;
  cache.mfem_face_points = requested_mfem_face_points;
  cache.enforce_nodes = requested_enforce_nodes;
  cache.enforce_vtk_points = requested_enforce_vtk_points;
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
    shape.SetSize(dof);
    mfem::Vector integral(dof);
    integral = 0.0;
    double volume = 0.0;
    for (int q = 0; q < volume_rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& ip = volume_rule.IntPoint(q);
      transformation.SetIntPoint(&ip);
      fe.CalcShape(ip, shape);
      const double weight = ip.weight * transformation.Weight();
      volume += weight;
      integral.Add(weight, shape);
      if (!cache.mfem_volume_points)
      {
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
    cache.point_offsets[e + 1] =
      static_cast<int>(point_shapes.size()) / dof;
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
  mfem::ZhangShuLimiterElementDiagnostics* diagnostics)
{
  ::dg_igr::NVTXRange limiter_range("Limiter/Zhang-Shu");
  LimiterTimingState& timing = GetLimiterTimingState();
  if (timing.enabled) { MFEM_DEVICE_SYNC; }
  const auto timing_start = std::chrono::steady_clock::now();

  const int num_eq = dim + 2;
  DeviceLimiterCache& cache =
    GetDeviceLimiterCache(vfes, num_eq, qord, face_qords, point_set);
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
  const int volume_nq = cache.volume_nq;
  const int total_face_points =
    cache.mfem_face_points && ne > 0 ? cache.face_point_offsets[ne] : 0;
  const int d1d = cache.d1d;
  double* state = cache.element_state.ReadWrite();
  double* averages = cache.averages.Write();
  double* theta_rho = cache.theta_rho.Write();
  double* theta_p = cache.theta_p.Write();
  double* active = cache.active.Write();

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

    const double rho_bar = average[0] > rho_min ? average[0] : rho_min;
    double momentum_sq = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      momentum_sq += average[1 + d] * average[1 + d];
    }
    const double kinetic = 0.5 * momentum_sq /
                           (rho_bar > 1.0e-16 ? rho_bar : 1.0e-16);
    const double pressure_bar =
      (gamma - 1.0) * (average[1 + dim] - kinetic);
    if (rho_bar <= rho_min || pressure_bar <= p_min)
    {
      average[0] = rho_bar;
      average[1 + dim] = kinetic + p_min / (gamma - 1.0);
      for (int d = 0; d < dof; ++d)
      {
        for (int c = 0; c < num_eq; ++c)
        {
          state[d + dof * (c + num_eq * e)] = average[c];
        }
      }
      theta_rho[e] = 0.0;
      theta_p[e] = 0.0;
      active[e] = 1.0;
    }
    else
    {
      double rho_theta = 1.0;
      if (mfem_volume_points)
      {
        for (int q = 0; q < volume_nq; ++q)
        {
          const double rho =
            volume_values[q + volume_nq * (num_eq * e)];
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
          const double rho = face_values[point];
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
        double rho = 0.0;
        for (int d = 0; d < dof; ++d)
        {
          rho += LimiterPointBasis(shapes,
                                   factors,
                                   tensor_points,
                                   point,
                                   d,
                                   dof,
                                   dim,
                                   d1d) *
                 state[d + dof * (num_eq * e)];
        }
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
          if (DevicePressure(point_state, dim, gamma) < p_min)
          {
            const double candidate =
              DeviceThetaForPressure(average,
                                     point_state,
                                     dim,
                                     gamma,
                                     p_min,
                                     rho_theta,
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
          if (DevicePressure(point_state, dim, gamma) < p_min)
          {
            const double candidate =
              DeviceThetaForPressure(average,
                                     point_state,
                                     dim,
                                     gamma,
                                     p_min,
                                     rho_theta,
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
        if (DevicePressure(point_state, dim, gamma) < p_min)
        {
          const double candidate =
            DeviceThetaForPressure(average,
                                   point_state,
                                   dim,
                                   gamma,
                                   p_min,
                                   rho_theta,
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
    for (int c = 0; c < num_eq; ++c)
    {
      averages[c + num_eq * e] = average[c];
    }
  });

  cache.restriction->MultTranspose(cache.element_state, U);
  if (diagnostics != nullptr)
  {
    CopyDeviceVector(cache.theta_rho, diagnostics->theta_rho);
    CopyDeviceVector(cache.theta_p, diagnostics->theta_p);
    CopyDeviceVector(cache.active, diagnostics->active);
  }

  if (timing.enabled) { MFEM_DEVICE_SYNC; }
  if (timing.enabled)
  {
    const double seconds =
      std::chrono::duration<double>(
        std::chrono::steady_clock::now() - timing_start)
        .count();
    RecordLimiterTiming(seconds, cache, &cache.active);
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
                                    ZhangShuLimiterElementDiagnostics* diagnostics)
{
  MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES,
              "Positivity limiter assumes byNODES ordering.");

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
                                         diagnostics);
    return;
  }

  LimiterTimingState& timing = GetLimiterTimingState();
  const auto timing_start = std::chrono::steady_clock::now();
  CpuLimiterPhaseSample phase;
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

  if (diagnostics)
  {
    diagnostics->SetSize(NE);
  }

  for (int i = 0; i < NE; i++)
  {
    const FiniteElement* fe = vfes.GetFE(i);
    const auto* fec_l2 = dynamic_cast<const L2_FECollection*>(vfes.FEColl());
    const bool consumer_points = point_set == ZhangShuPointSet::kConsumer;
    const bool enforce_nodes =
      (fec_l2 != nullptr && fec_l2->GetBasisType() == BasisType::GaussLobatto) ||
      consumer_points;
    const bool enforce_vtk_points =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_LIMITER_ENFORCE_VTK_POINTS",
                                      false);
    ElementTransformation* Tr = vfes.GetElementTransformation(i);
    const int dof = fe->GetDof();
    vfes.GetElementVDofs(i, vdofs);

    U.GetSubVector(vdofs, elvec);
    elmat.UseExternalData(elvec.GetData(), dof, num_eq);

    // --- compute cell average ---
    const auto average_start = phase_start();
    dg_igr::limiters_detail::ElementAverage(
      *fe, *Tr, elmat, num_eq, qord, Ubar);
    phase_add(average_start, phase.average_seconds);

    // If the average itself is unphysical, collapse element to constant admissible state
    {
      const auto admissible_average_start = phase_start();
      const double rho_bar = std::max(Ubar(0), rho_min);
      double mom2_bar = 0.0;
      for (int d = 0; d < dim; ++d)
        mom2_bar += Ubar(1 + d) * Ubar(1 + d);
      const double k_bar = 0.5 * mom2_bar / std::max(rho_bar, 1e-16);
      double E_bar = Ubar(1 + dim);
      double p_bar = (gamma - 1.0) * (E_bar - k_bar);

      if (rho_bar <= rho_min || p_bar <= p_min)
      {
        phase.collapsed_elements += 1.0;
        phase.active_elements += 1.0;
        Ubar(0) = rho_bar;
        E_bar = k_bar + p_min / (gamma - 1.0);
        Ubar(1 + dim) = E_bar;

        for (int j = 0; j < dof; ++j)
        {
          elmat.GetRow(j, row);
          row = Ubar;
          elmat.SetRow(j, row);
        }
        U.SetSubVector(vdofs, elmat.GetData());
        if (diagnostics)
        {
          diagnostics->active(i) = 1.0;
          diagnostics->theta_rho(i) = 0.0;
          diagnostics->theta_p(i) = 0.0;
        }
        phase_add(admissible_average_start,
                  phase.admissible_average_seconds);
        continue; // done with this element
      }
      phase_add(admissible_average_start,
                phase.admissible_average_seconds);
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
        dg_igr::limiters_detail::BlendRow(row, Ubar, theta_rho);
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
                                            /*theta_max=*/theta_rho,
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
        dg_igr::limiters_detail::BlendRow(row, Ubar, theta_p);
        elmat.SetRow(j, row);
      }
    }
    phase_add(pressure_blend_start, phase.pressure_blend_seconds);

    if (theta_rho < 1.0 || theta_p < 1.0)
    {
      phase.active_elements += 1.0;
    }

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
      const double local_active = phase.active_elements;
      double global_active = local_active;
      if (Mpi::IsInitialized() && !Mpi::IsFinalized())
      {
        MPI_Allreduce(&local_active,
                      &global_active,
                      1,
                      MPITypeMap<double>::mpi_type,
                      MPI_SUM,
                      MPI_COMM_WORLD);
      }
      ++timing.calls;
      timing.total_seconds +=
        std::chrono::duration<double>(
          std::chrono::steady_clock::now() - timing_start)
          .count();
      timing.avg_active_elements += global_active;
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
}

} // namespace mfem
