// Internal-only helper/kernels for DG face quadrature projection.
// This file is intentionally included after MFEM backend headers.
#pragma once

namespace dg_igr
{
namespace dg_face
{

struct FaceQFProjectInfo
{
  int dim = 0;
  int vdim = 0;
  int nd1d = 0;
  int nq1d = 0;
  int nf = 0;
  int f_vec_size = 0;
  int qf_size = 0;
};

struct FaceQFProjectDim1
{
  int vdim = 0;
  const mfem::real_t* f = nullptr;
  mfem::real_t* q = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    for (int c = 0; c < vdim; ++c)
    {
      q[c + vdim * face] = f[c + vdim * face];
    }
  }
};

struct FaceQFProjectDim2
{
  int nd1d = 0;
  int nq1d = 0;
  int vdim = 0;
  int nf = 0;
  const mfem::real_t* B = nullptr;
  const mfem::real_t* f = nullptr;
  mfem::real_t* q = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto Bm = mfem::Reshape(B, nq1d, nd1d);
    auto F = mfem::Reshape(f, nd1d, vdim, nf);
    auto Q = mfem::Reshape(q, vdim, nq1d, nf);
    for (int c = 0; c < vdim; ++c)
    {
      for (int qpt = 0; qpt < nq1d; ++qpt)
      {
        mfem::real_t val = 0.0;
        for (int d = 0; d < nd1d; ++d)
        {
          val += Bm(qpt, d) * F(d, c, face);
        }
        Q(c, qpt, face) = val;
      }
    }
  }
};

struct FaceQFProjectDim3
{
  int nd1d = 0;
  int nq1d = 0;
  int vdim = 0;
  int nf = 0;
  const mfem::real_t* B = nullptr;
  const mfem::real_t* f = nullptr;
  mfem::real_t* q = nullptr;

  MFEM_HOST_DEVICE void operator()(int face) const
  {
    auto Bm = mfem::Reshape(B, nq1d, nd1d);
    auto F = mfem::Reshape(f, nd1d, nd1d, vdim, nf);
    auto Q = mfem::Reshape(q, vdim, nq1d, nq1d, nf);
    for (int c = 0; c < vdim; ++c)
    {
      for (int q2 = 0; q2 < nq1d; ++q2)
      {
        for (int q1 = 0; q1 < nq1d; ++q1)
        {
          mfem::real_t val = 0.0;
          for (int d2 = 0; d2 < nd1d; ++d2)
          {
            const mfem::real_t b2 = Bm(q2, d2);
            for (int d1 = 0; d1 < nd1d; ++d1)
            {
              val += Bm(q1, d1) * b2 * F(d1, d2, c, face);
            }
          }
          Q(c, q1, q2, face) = val;
        }
      }
    }
  }
};

inline FaceQFProjectInfo ProjectFaceQuadratureValues(
  const mfem::FiniteElementSpace& fes,
  const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& gf,
  mfem::QuadratureFunction& qf,
  mfem::Vector* face_restriction_evec = nullptr)
{
  FaceQFProjectInfo info;
  const bool use_tensor = mfem::UsesTensorBasis(fes);
  MFEM_VERIFY(use_tensor,
              "Face quadrature projection requires tensor-product basis.");
  const mfem::ElementDofOrdering ordering =
    use_tensor ? mfem::ElementDofOrdering::LEXICOGRAPHIC
               : mfem::ElementDofOrdering::NATIVE;
  const mfem::FaceType face_type = fqs.GetFaceType();
  const mfem::Operator* R =
    fes.GetFaceRestriction(ordering, face_type,
                           mfem::L2FaceValues::SingleValued);
  MFEM_VERIFY(R, "Missing face restriction for face quadrature projection.");

  // Reuse caller-owned storage when available to avoid per-update allocations.
  mfem::Vector local_f_vec;
  mfem::Vector* f_vec_ptr = face_restriction_evec ? face_restriction_evec
                                                  : &local_f_vec;
  f_vec_ptr->SetSize(R->Height());
  R->Mult(gf, *f_vec_ptr);

  const mfem::FiniteElement* face_fe = fes.GetTypicalTraceElement();
  const mfem::IntegrationRule& ir = fqs.GetIntRule(0);
  const mfem::DofToQuad& maps = face_fe->GetDofToQuad(ir, mfem::DofToQuad::TENSOR);

  info.nd1d = maps.ndof;
  info.nq1d = maps.nqpt;
  info.vdim = fes.GetVDim();
  info.dim = fes.GetMesh()->Dimension();
  info.nf = fqs.GetNE();
  info.f_vec_size = f_vec_ptr->Size();
  info.qf_size = qf.Size();

  const mfem::real_t* B_data = maps.B.Read();
  const mfem::real_t* f_data = f_vec_ptr->Read();
  mfem::real_t* q_data = qf.Write();

  if (info.dim == 1)
  {
    FaceQFProjectDim1 kernel;
    kernel.vdim = info.vdim;
    kernel.f = f_data;
    kernel.q = q_data;
    mfem::forall(info.nf, kernel);
    return info;
  }

  if (info.dim == 2)
  {
    FaceQFProjectDim2 kernel;
    kernel.nd1d = info.nd1d;
    kernel.nq1d = info.nq1d;
    kernel.vdim = info.vdim;
    kernel.nf = info.nf;
    kernel.B = B_data;
    kernel.f = f_data;
    kernel.q = q_data;
    mfem::forall(info.nf, kernel);
    return info;
  }

  if (info.dim == 3)
  {
    FaceQFProjectDim3 kernel;
    kernel.nd1d = info.nd1d;
    kernel.nq1d = info.nq1d;
    kernel.vdim = info.vdim;
    kernel.nf = info.nf;
    kernel.B = B_data;
    kernel.f = f_data;
    kernel.q = q_data;
    mfem::forall(info.nf, kernel);
    return info;
  }

  MFEM_ABORT("Unsupported dimension in face quadrature projection.");
  return info;
}

} // namespace dg_face
} // namespace dg_igr
