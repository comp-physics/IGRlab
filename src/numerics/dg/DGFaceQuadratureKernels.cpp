#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "DGFaceQuadratureKernels.hpp"

namespace igrlab
{
namespace dg_face
{

namespace
{

void LaunchDGDiffusionCoefficientUpdate2D(
  int dof,
  int nq,
  int nf,
  mfem::Vector& side_coefficient,
  const mfem::Vector& side_q,
  const mfem::Vector& normal_dshape_base,
  const mfem::Vector& penalty_metric,
  mfem::Vector& weighted_normal_dshape,
  mfem::Vector& penalty,
  mfem::real_t kappa,
  DGDiffusionFaceScaling face_scaling,
  bool update_normals = true)
{
  const int side_q_size = 2 * nq * nf;
  const int normal_size = 2 * nq * dof * nf;
  MFEM_VERIFY(dof > 0 && nq > 0 && nf >= 0 &&
                side_q.Size() == side_q_size &&
                (!update_normals || normal_dshape_base.Size() == normal_size) &&
                penalty_metric.Size() == side_q_size,
              "Exact DG face coefficient refresh received inconsistent "
              "persistent geometry or qdata.");

  side_coefficient.SetSize(side_q_size);
  side_coefficient.UseDevice(true);
  side_coefficient = side_q;
  if (update_normals)
  {
    weighted_normal_dshape.SetSize(normal_size);
    weighted_normal_dshape.UseDevice(true);
  }
  penalty.SetSize(nq * nf);
  penalty.UseDevice(true);

  const mfem::real_t* coefficient = side_coefficient.Read();
  const mfem::real_t* normal_base = update_normals ? normal_dshape_base.Read() : nullptr;
  const mfem::real_t* metric = penalty_metric.Read();
  mfem::real_t* weighted_normal = update_normals ? weighted_normal_dshape.Write() : nullptr;
  mfem::real_t* penalty_out = penalty.Write();
  const int scaling = static_cast<int>(face_scaling);

  mfem::forall(
    nq * nf,
    [=] MFEM_HOST_DEVICE(int entry)
    {
      const int q = entry % nq;
      const int face = entry / nq;
      const int qbase = 2 * (q + nq * face);
      const mfem::real_t q0 = coefficient[qbase];
      const mfem::real_t q1 = coefficient[qbase + 1];
      const mfem::real_t metric0 = q0 * metric[qbase];
      const mfem::real_t metric1 = q1 * metric[qbase + 1];
      const mfem::real_t metric_sum = metric0 + metric1;

      if (update_normals)
      {
        mfem::real_t normal_scale0 = q0;
        mfem::real_t normal_scale1 = q1;
        if (scaling ==
              static_cast<int>(DGDiffusionFaceScaling::HarmonicWeighted) &&
            metric_sum > 0.0)
        {
          normal_scale0 *= 2.0 * metric1 / metric_sum;
          normal_scale1 *= 2.0 * metric0 / metric_sum;
        }
        for (int d = 0; d < dof; ++d)
        {
          const int nbase = 2 * (q + nq * (d + dof * face));
          weighted_normal[nbase] = normal_scale0 * normal_base[nbase];
          weighted_normal[nbase + 1] =
            normal_scale1 * normal_base[nbase + 1];
        }
      }

      mfem::real_t penalty_value = 0.5 * metric_sum;
      if (scaling == static_cast<int>(DGDiffusionFaceScaling::Max))
      {
        penalty_value = fmax(metric0, metric1);
      }
      else if (scaling == static_cast<int>(DGDiffusionFaceScaling::Sum))
      {
        penalty_value = metric_sum;
      }
      else if (scaling ==
               static_cast<int>(DGDiffusionFaceScaling::HarmonicWeighted))
      {
        penalty_value = metric_sum > 0.0
          ? 2.0 * metric0 * metric1 / metric_sum
          : 0.0;
      }
      penalty_out[entry] = kappa * penalty_value;
    });
}

void LaunchDGDiffusionNormalUpdate2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work)
{
  const int dof = info.dof, nq = info.nq, nf = info.nf;
  MFEM_VERIFY(dof > 0 && nq > 0 && nf >= 0 &&
                work.side_coefficient.Size() == 2 * nq * nf &&
                work.normal_dshape_base.Size() == 2 * nq * dof * nf &&
                work.penalty_metric.Size() == 2 * nq * nf,
              "Lazy exact DG face normals require current coefficients and "
              "the exact persistent geometry.");
  work.weighted_normal_dshape.SetSize(2 * nq * dof * nf);
  work.weighted_normal_dshape.UseDevice(true);
  const mfem::real_t* coefficient = work.side_coefficient.Read();
  const mfem::real_t* normal_base = work.normal_dshape_base.Read();
  const mfem::real_t* metric = work.penalty_metric.Read();
  mfem::real_t* weighted_normal = work.weighted_normal_dshape.Write();
  const bool harmonic =
    work.coefficient_scaling == DGDiffusionFaceScaling::HarmonicWeighted;
  mfem::forall(nq * nf, [=] MFEM_HOST_DEVICE(int entry)
  {
    const int q = entry % nq, face = entry / nq;
    const int qbase = 2 * (q + nq * face);
    mfem::real_t scale0 = coefficient[qbase];
    mfem::real_t scale1 = coefficient[qbase + 1];
    if (harmonic)
    {
      const mfem::real_t metric0 = scale0 * metric[qbase];
      const mfem::real_t metric1 = scale1 * metric[qbase + 1];
      const mfem::real_t sum = metric0 + metric1;
      if (sum > 0.0)
      {
        scale0 *= 2.0 * metric1 / sum;
        scale1 *= 2.0 * metric0 / sum;
      }
    }
    for (int d = 0; d < dof; ++d)
    {
      const int nbase = 2 * (q + nq * (d + dof * face));
      weighted_normal[nbase] = scale0 * normal_base[nbase];
      weighted_normal[nbase + 1] = scale1 * normal_base[nbase + 1];
    }
  });
}

} // namespace

void EnsureInteriorFaceDGDiffusionDenseNormals2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work)
{
  // A vector-only geometry rebuild already supplied a new, current dense
  // table. Its old scratch state must neither overwrite it nor reuse old Q.
  if (work.coefficient_geometry_revision != info.connectivity_revision ||
      work.coefficient_generation == 0 ||
      work.dense_normal_generation == work.coefficient_generation)
  { return; }
  LaunchDGDiffusionNormalUpdate2D(info, work);
  work.dense_normal_generation = work.coefficient_generation;
}

void LaunchInteriorFaceDGDiffusionApply2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  mfem::real_t ip_sigma,
  mfem::Vector& y)
{
  AddInteriorFaceDGDiffusionApply2D(info, work, scalar, ip_sigma, y);
}

bool TryLaunchInteriorFaceDGDiffusionAddToBase2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  mfem::real_t ip_sigma,
  mfem::Vector& y)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  const unsigned long alternative_backends =
    mfem::Backend::OMP_MASK | mfem::Backend::RAJA_MASK |
    mfem::Backend::OCCA_MASK | mfem::Backend::CEED_MASK |
    mfem::Backend::HIP_MASK | mfem::Backend::DEBUG_DEVICE;
  if (mfem::Device::Allows(mfem::Backend::CUDA) &&
      !mfem::Device::Allows(alternative_backends))
  {
    AddInteriorFaceDGDiffusionApplyDevice2D(
      info, work, scalar, ip_sigma, y, true);
    return true;
  }
#else
  (void)info; (void)work; (void)scalar; (void)ip_sigma; (void)y;
#endif
  return false;
}

void LaunchSharedFaceDGDiffusionApply2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  const SharedFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  const mfem::Vector& scalar_face_nbr,
  mfem::real_t ip_sigma,
  mfem::Vector& y)
{
  AddSharedFaceDGDiffusionApply2D(info,
                                  work,
                                  scalar,
                                  scalar_face_nbr,
                                  ip_sigma,
                                  y);
}

void LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& side_q,
  mfem::real_t kappa,
  DGDiffusionFaceScaling face_scaling)
{
  // A failed/aborted refresh cannot leave an earlier coefficient stamp valid.
  work.normal_factorization.coefficient_revision = 0;
  work.coefficient_geometry_revision = 0;
  work.dense_normal_generation = 0;
  const bool compact = work.compact_qdata &&
    UsesOrdinaryCUDAForDGDiffusionFaces2D() &&
    work.normal_factorization.TensorMatches(info) &&
    (face_scaling == DGDiffusionFaceScaling::Arithmetic ||
     face_scaling == DGDiffusionFaceScaling::Max ||
     face_scaling == DGDiffusionFaceScaling::Sum);
  LaunchDGDiffusionCoefficientUpdate2D(info.dof,
                                       info.nq,
                                       info.nf,
                                       work.side_coefficient,
                                       side_q,
                                       work.normal_dshape_base,
                                       work.penalty_metric,
                                       work.weighted_normal_dshape,
                                       work.penalty,
                                       kappa,
                                       face_scaling,
                                       !compact);
  if (++work.coefficient_generation == 0) { ++work.coefficient_generation; }
  work.coefficient_geometry_revision = info.connectivity_revision;
  work.coefficient_scaling = face_scaling;
  if (compact)
  {
    // Keep the exact host geometry for generic/initial paths, but release any
    // full device allocations left by an earlier dense application. N belongs
    // to an older stage and must not trigger a full, useless copy to the host.
    work.shape.DeleteDevice();
    work.normal_dshape_base.DeleteDevice();
    work.weighted_normal_dshape.DeleteDevice(false);
  }
  else { work.dense_normal_generation = work.coefficient_generation; }
  work.normal_factorization.coefficient_scaling = face_scaling;
  work.normal_factorization.coefficient_revision = info.connectivity_revision;
}

void LaunchSharedFaceDGDiffusionCoefficientUpdate2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  SharedFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& side_q,
  mfem::real_t kappa,
  DGDiffusionFaceScaling face_scaling)
{
  LaunchDGDiffusionCoefficientUpdate2D(info.dof,
                                       info.nq,
                                       info.nf,
                                       work.side_coefficient,
                                       side_q,
                                       work.normal_dshape_base,
                                       work.penalty_metric,
                                       work.weighted_normal_dshape,
                                       work.penalty,
                                       kappa,
                                       face_scaling);
}

} // namespace dg_face
} // namespace igrlab
