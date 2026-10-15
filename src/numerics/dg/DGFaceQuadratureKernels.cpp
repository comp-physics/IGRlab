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
  DGDiffusionFaceScaling face_scaling)
{
  const int side_q_size = 2 * nq * nf;
  const int normal_size = 2 * nq * dof * nf;
  MFEM_VERIFY(dof > 0 && nq > 0 && nf >= 0 &&
                side_q.Size() == side_q_size &&
                normal_dshape_base.Size() == normal_size &&
                penalty_metric.Size() == side_q_size,
              "Exact DG face coefficient refresh received inconsistent "
              "persistent geometry or qdata.");

  side_coefficient.SetSize(side_q_size);
  side_coefficient.UseDevice(true);
  side_coefficient = side_q;
  weighted_normal_dshape.SetSize(normal_size);
  weighted_normal_dshape.UseDevice(true);
  penalty.SetSize(nq * nf);
  penalty.UseDevice(true);

  const mfem::real_t* coefficient = side_coefficient.Read();
  const mfem::real_t* normal_base = normal_dshape_base.Read();
  const mfem::real_t* metric = penalty_metric.Read();
  mfem::real_t* weighted_normal = weighted_normal_dshape.Write();
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

} // namespace

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
