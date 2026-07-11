#include "physics/igr/IGR.hpp"
#include "utils/config/env_flags.hpp"
#include "utils/profiling/nvtx.hpp"
#include "IGR_internal_utils.hpp"
#include "../../numerics/dg/DGFaceQuadratureKernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>

namespace dg_igr
{
using namespace igr_internal;

namespace
{

using Clock = std::chrono::steady_clock;

double ElapsedSeconds(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct ExactFaceTimingStats
{
  long long mult_calls = 0;
  double base_seconds = 0.0;
  double face_seconds = 0.0;
  double sum_axpy_seconds = 0.0;
  double face_setup_seconds = 0.0;
  double prolong_seconds = 0.0;
  double local_apply_seconds = 0.0;
  double shared_exchange_seconds = 0.0;
  double shared_apply_seconds = 0.0;
  double restrict_seconds = 0.0;
};

int ReadEnvInt(const char* name, int default_value)
{
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
  {
    return default_value;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  MFEM_VERIFY(end && *end == '\0',
              "Invalid integer value for " << name << ": " << raw);
  MFEM_VERIFY(value >= std::numeric_limits<int>::min() &&
                value <= std::numeric_limits<int>::max(),
              "Integer value for " << name << " is out of range: " << raw);
  return static_cast<int>(value);
}

mfem::real_t ReadEnvReal(const char* name, mfem::real_t default_value)
{
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
  {
    return default_value;
  }
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  MFEM_VERIFY(end && *end == '\0',
              "Invalid real value for " << name << ": " << raw);
  return static_cast<mfem::real_t>(value);
}

void MakeDiagonalPositive(mfem::Vector& diag)
{
  diag.UseDevice(true);
  diag.Abs();
}

std::string LogToken(std::string value)
{
  for (char& c : value)
  {
    if (c == ' ' || c == '\t' || c == '\n' || c == '=')
    {
      c = '_';
    }
  }
  return value;
}

std::unique_ptr<mfem::OperatorJacobiSmoother>
MakePositiveOperatorJacobi(const mfem::BilinearForm& a,
                           const mfem::Array<int>& ess_tdof_list)
{
  mfem::Vector diag(a.FESpace()->GetTrueVSize());
  a.AssembleDiagonal(diag);
  MakeDiagonalPositive(diag);
  return std::make_unique<mfem::OperatorJacobiSmoother>(diag, ess_tdof_list);
}

std::unique_ptr<mfem::OperatorJacobiSmoother>
MakePositiveOperatorJacobi(const mfem::Operator& op,
                           const mfem::Array<int>& ess_tdof_list)
{
  mfem::Vector diag(op.Height());
  op.AssembleDiagonal(diag);
  MakeDiagonalPositive(diag);
  return std::make_unique<mfem::OperatorJacobiSmoother>(diag, ess_tdof_list);
}

struct KrylovApplyTelemetry
{
  long long operator_apply_count = 0;
  long long preconditioner_apply_count = 0;
  long long preconditioner_operator_apply_count = 0;
  double operator_apply_seconds = 0.0;
  double preconditioner_apply_seconds = 0.0;
  double preconditioner_operator_apply_seconds = 0.0;
  bool record_seconds = false;
  bool sync = false;
  bool inside_preconditioner = false;

  void ResetCounters()
  {
    operator_apply_count = 0;
    preconditioner_apply_count = 0;
    preconditioner_operator_apply_count = 0;
    operator_apply_seconds = 0.0;
    preconditioner_apply_seconds = 0.0;
    preconditioner_operator_apply_seconds = 0.0;
    inside_preconditioner = false;
  }
};

class KrylovTelemetryOperator final : public mfem::Operator
{
public:
  KrylovTelemetryOperator(const mfem::Operator& op,
                          KrylovApplyTelemetry& telemetry)
    : mfem::Operator(op.Height(), op.Width()),
      op_(op),
      telemetry_(telemetry)
  {
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/krylov-operator-mult");
    if (!telemetry_.record_seconds)
    {
      op_.Mult(x, y);
      ++telemetry_.operator_apply_count;
      if (telemetry_.inside_preconditioner)
      {
        ++telemetry_.preconditioner_operator_apply_count;
      }
      return;
    }
    const auto start = Clock::now();
    op_.Mult(x, y);
    if (telemetry_.sync)
    {
      MFEM_DEVICE_SYNC;
    }
    const auto stop = Clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    ++telemetry_.operator_apply_count;
    telemetry_.operator_apply_seconds += seconds;
    if (telemetry_.inside_preconditioner)
    {
      ++telemetry_.preconditioner_operator_apply_count;
      telemetry_.preconditioner_operator_apply_seconds += seconds;
    }
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/krylov-operator-mult-transpose");
    if (!telemetry_.record_seconds)
    {
      op_.MultTranspose(x, y);
      ++telemetry_.operator_apply_count;
      if (telemetry_.inside_preconditioner)
      {
        ++telemetry_.preconditioner_operator_apply_count;
      }
      return;
    }
    const auto start = Clock::now();
    op_.MultTranspose(x, y);
    if (telemetry_.sync)
    {
      MFEM_DEVICE_SYNC;
    }
    const auto stop = Clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    ++telemetry_.operator_apply_count;
    telemetry_.operator_apply_seconds += seconds;
    if (telemetry_.inside_preconditioner)
    {
      ++telemetry_.preconditioner_operator_apply_count;
      telemetry_.preconditioner_operator_apply_seconds += seconds;
    }
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return op_.GetMemoryClass();
  }

private:
  const mfem::Operator& op_;
  KrylovApplyTelemetry& telemetry_;
};

class KrylovTelemetryPreconditioner final : public mfem::Solver
{
public:
  KrylovTelemetryPreconditioner(mfem::Solver& preconditioner,
                                KrylovApplyTelemetry& telemetry)
    : mfem::Solver(preconditioner.Height(), preconditioner.Width(), false),
      preconditioner_(preconditioner),
      telemetry_(telemetry)
  {
  }

  void SetOperator(const mfem::Operator&) override
  {
    height = preconditioner_.Height();
    width = preconditioner_.Width();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/krylov-preconditioner-mult");
    if (!telemetry_.record_seconds)
    {
      const bool was_inside = telemetry_.inside_preconditioner;
      telemetry_.inside_preconditioner = true;
      preconditioner_.Mult(x, y);
      telemetry_.inside_preconditioner = was_inside;
      ++telemetry_.preconditioner_apply_count;
      return;
    }
    const auto start = Clock::now();
    const bool was_inside = telemetry_.inside_preconditioner;
    telemetry_.inside_preconditioner = true;
    preconditioner_.Mult(x, y);
    telemetry_.inside_preconditioner = was_inside;
    if (telemetry_.sync)
    {
      MFEM_DEVICE_SYNC;
    }
    const auto stop = Clock::now();
    ++telemetry_.preconditioner_apply_count;
    telemetry_.preconditioner_apply_seconds +=
      std::chrono::duration<double>(stop - start).count();
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/krylov-preconditioner-mult-transpose");
    if (!telemetry_.record_seconds)
    {
      const bool was_inside = telemetry_.inside_preconditioner;
      telemetry_.inside_preconditioner = true;
      preconditioner_.MultTranspose(x, y);
      telemetry_.inside_preconditioner = was_inside;
      ++telemetry_.preconditioner_apply_count;
      return;
    }
    const auto start = Clock::now();
    const bool was_inside = telemetry_.inside_preconditioner;
    telemetry_.inside_preconditioner = true;
    preconditioner_.MultTranspose(x, y);
    telemetry_.inside_preconditioner = was_inside;
    if (telemetry_.sync)
    {
      MFEM_DEVICE_SYNC;
    }
    const auto stop = Clock::now();
    ++telemetry_.preconditioner_apply_count;
    telemetry_.preconditioner_apply_seconds +=
      std::chrono::duration<double>(stop - start).count();
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return preconditioner_.GetMemoryClass();
  }

private:
  mfem::Solver& preconditioner_;
  KrylovApplyTelemetry& telemetry_;
};

class ExactL2DGFaceOperator final : public mfem::Operator
{
public:
  ExactL2DGFaceOperator(const mfem::ParFiniteElementSpace& sfes,
                        const mfem::FaceQuadratureSpace& fqs,
                        const mfem::Vector& side_q,
                        const mfem::IntegrationRule& shared_ir,
                        const mfem::Vector& shared_side_q,
                        mfem::real_t ip_sigma,
                        mfem::real_t kappa)
    : mfem::Operator(sfes.GetTrueVSize()),
      sfes_(sfes),
      prolong_(sfes.GetProlongationMatrix()),
      restrict_(sfes.GetRestrictionOperator()),
      ip_sigma_(ip_sigma),
      local_x_gf_(const_cast<mfem::ParFiniteElementSpace*>(&sfes))
  {
    info_ = dg_face::BuildInteriorFaceDGDiffusionApplyData2D(
      sfes_,
      fqs,
      side_q,
      kappa,
      scratch_.shape,
      scratch_.weighted_normal_dshape,
      scratch_.penalty,
      scratch_.face_elements,
      scratch_.element_vdof_map);
    timing_enabled_ =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_EXACT_FACE_TIMING",
                                      false);
    if (shared_side_q.Size() > 0)
    {
      shared_info_ = dg_face::BuildSharedFaceDGDiffusionApplyData2D(
        sfes_,
        shared_ir,
        shared_side_q,
        kappa,
        shared_scratch_.shape,
        shared_scratch_.weighted_normal_dshape,
        shared_scratch_.penalty,
        shared_scratch_.face_elements,
        shared_scratch_.element_vdof_map,
        shared_scratch_.face_nbr_dof_map);
      has_shared_ = (shared_info_.nf > 0);
    }
  }

  bool TimingEnabled() const { return timing_enabled_; }

  void ResetTiming() const { timing_ = ExactFaceTimingStats{}; }

  const ExactFaceTimingStats& Timing() const { return timing_; }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return mfem::Device::GetDeviceMemoryClass();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/exact-face/Mult");
    if (!timing_enabled_)
    {
      local_y_.SetSize(sfes_.GetVSize(), y);
      local_y_ = 0.0;

      if (prolong_)
      {
        ::dg_igr::NVTXRange prolong_range("IGR/exact-face/prolong");
        prolong_->Mult(x, local_x_gf_);
      }
      else
      {
        local_x_gf_ = x;
      }

      {
        ::dg_igr::NVTXRange apply_range("IGR/exact-face/local-apply");
        dg_face::LaunchInteriorFaceDGDiffusionApply2D(info_,
                                                      scratch_,
                                                      local_x_gf_,
                                                      ip_sigma_,
                                                      local_y_);
      }
      if (has_shared_)
      {
        {
          ::dg_igr::NVTXRange exchange_range("IGR/exact-face/shared-exchange");
          local_x_gf_.ExchangeFaceNbrData();
        }
        {
          ::dg_igr::NVTXRange apply_range("IGR/exact-face/shared-apply");
          dg_face::LaunchSharedFaceDGDiffusionApply2D(
            shared_info_,
            shared_scratch_,
            local_x_gf_,
            local_x_gf_.FaceNbrData(),
            ip_sigma_,
            local_y_);
        }
      }

      if (restrict_)
      {
        ::dg_igr::NVTXRange restrict_range("IGR/exact-face/restrict");
        restrict_->Mult(local_y_, y);
      }
      else
      {
        y = local_y_;
      }
      return;
    }

    const auto setup_start = Clock::now();
    local_y_.SetSize(sfes_.GetVSize(), y);
    local_y_ = 0.0;
    timing_.face_setup_seconds += ElapsedSeconds(setup_start);

    const auto prolong_start = Clock::now();
    if (prolong_)
    {
      ::dg_igr::NVTXRange prolong_range("IGR/exact-face/prolong");
      prolong_->Mult(x, local_x_gf_);
    }
    else
    {
      local_x_gf_ = x;
    }
    timing_.prolong_seconds += ElapsedSeconds(prolong_start);

    const auto local_apply_start = Clock::now();
    {
      ::dg_igr::NVTXRange apply_range("IGR/exact-face/local-apply");
      dg_face::LaunchInteriorFaceDGDiffusionApply2D(info_,
                                                    scratch_,
                                                    local_x_gf_,
                                                    ip_sigma_,
                                                    local_y_);
    }
    timing_.local_apply_seconds += ElapsedSeconds(local_apply_start);
    if (has_shared_)
    {
      const auto shared_exchange_start = Clock::now();
      {
        ::dg_igr::NVTXRange exchange_range("IGR/exact-face/shared-exchange");
        local_x_gf_.ExchangeFaceNbrData();
      }
      timing_.shared_exchange_seconds += ElapsedSeconds(shared_exchange_start);

      const auto shared_apply_start = Clock::now();
      {
        ::dg_igr::NVTXRange apply_range("IGR/exact-face/shared-apply");
        dg_face::LaunchSharedFaceDGDiffusionApply2D(shared_info_,
                                                    shared_scratch_,
                                                    local_x_gf_,
                                                    local_x_gf_.FaceNbrData(),
                                                    ip_sigma_,
                                                    local_y_);
      }
      timing_.shared_apply_seconds += ElapsedSeconds(shared_apply_start);
    }

    const auto restrict_start = Clock::now();
    if (restrict_)
    {
      ::dg_igr::NVTXRange restrict_range("IGR/exact-face/restrict");
      restrict_->Mult(local_y_, y);
    }
    else
    {
      y = local_y_;
    }
    timing_.restrict_seconds += ElapsedSeconds(restrict_start);
    ++timing_.mult_calls;
  }

  void AssembleDiagonal(mfem::Vector& diag) const override
  {
    MFEM_VERIFY(sfes_.GetTrueVSize() == sfes_.GetVSize(),
                "Exact L2 DG face diagonal currently requires true/local "
                "dof identity.");
    diag.SetSize(Height());
    diag.UseDevice(true);
    diag = 0.0;
    dg_face::AddInteriorFaceDGDiffusionDiagonal2D(info_,
                                                  scratch_,
                                                  ip_sigma_,
                                                  diag);
    if (has_shared_)
    {
      dg_face::AddSharedFaceDGDiffusionLocalDiagonal2D(shared_info_,
                                                       shared_scratch_,
                                                       ip_sigma_,
                                                       diag);
    }
  }

private:
  const mfem::ParFiniteElementSpace& sfes_;
  const mfem::Operator* prolong_ = nullptr;
  const mfem::Operator* restrict_ = nullptr;
  mfem::real_t ip_sigma_ = 0.0;
  dg_face::InteriorFaceDGDiffusionApplyInfo info_;
  dg_face::InteriorFaceDGDiffusionApplyScratch scratch_;
  bool has_shared_ = false;
  dg_face::SharedFaceDGDiffusionApplyInfo shared_info_;
  dg_face::SharedFaceDGDiffusionApplyScratch shared_scratch_;
  mutable mfem::Vector local_y_;
  mutable mfem::ParGridFunction local_x_gf_;
  bool timing_enabled_ = false;
  mutable ExactFaceTimingStats timing_;
};

class ExactFaceSumOperator final : public mfem::Operator
{
public:
  ExactFaceSumOperator(mfem::Operator* base,
                       mfem::Operator* face,
                       bool own_base = true,
                       bool own_face = true)
    : mfem::Operator(base->Height(), base->Width()),
      base_(base),
      face_(face),
      own_base_(own_base),
      own_face_(own_face),
      exact_face_(dynamic_cast<ExactL2DGFaceOperator*>(face)),
      timing_enabled_(exact_face_ != nullptr && exact_face_->TimingEnabled()),
      scratch_(base->Height())
  {
    MFEM_VERIFY(base_->Height() == face_->Height() &&
                  base_->Width() == face_->Width(),
                "Exact face sum operator received mismatched operators.");
  }

  ~ExactFaceSumOperator() override
  {
    if (own_base_)
    {
      delete base_;
    }
    if (own_face_)
    {
      delete face_;
    }
  }

  mfem::MemoryClass GetMemoryClass() const override
  {
    return base_->GetMemoryClass() * face_->GetMemoryClass();
  }

  bool TimingEnabled() const { return timing_enabled_; }

  mfem::Operator& BaseOperator() const { return *base_; }

  void ResetTiming() const
  {
    operator_mult_calls_ = 0;
    timing_ = ExactFaceTimingStats{};
    if (exact_face_)
    {
      exact_face_->ResetTiming();
    }
  }

  long long LastMultCalls() const { return operator_mult_calls_; }

  void LogTimingSummary(MPI_Comm comm,
                        int solve_call,
                        int solve_iters,
                        double solve_seconds,
                        const std::string& precond_name) const
  {
    if (!timing_enabled_)
    {
      return;
    }

    ExactFaceTimingStats local = timing_;
    if (exact_face_)
    {
      const ExactFaceTimingStats& face_timing = exact_face_->Timing();
      local.face_setup_seconds = face_timing.face_setup_seconds;
      local.prolong_seconds = face_timing.prolong_seconds;
      local.local_apply_seconds = face_timing.local_apply_seconds;
      local.shared_exchange_seconds = face_timing.shared_exchange_seconds;
      local.shared_apply_seconds = face_timing.shared_apply_seconds;
      local.restrict_seconds = face_timing.restrict_seconds;
    }

    double seconds[] = {local.base_seconds,
                        local.face_seconds,
                        local.sum_axpy_seconds,
                        local.face_setup_seconds,
                        local.prolong_seconds,
                        local.local_apply_seconds,
                        local.shared_exchange_seconds,
                        local.shared_apply_seconds,
                        local.restrict_seconds};
    double max_seconds[sizeof(seconds) / sizeof(seconds[0])] = {};
    long long calls = local.mult_calls;
    long long max_calls = calls;
#ifdef MFEM_USE_MPI
    MPI_Allreduce(seconds,
                  max_seconds,
                  static_cast<int>(sizeof(seconds) / sizeof(seconds[0])),
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_MAX,
                  comm);
    MPI_Allreduce(&calls,
                  &max_calls,
                  1,
                  MPI_LONG_LONG,
                  MPI_MAX,
                  comm);
#else
    (void)comm;
    std::copy(seconds,
              seconds + static_cast<int>(sizeof(seconds) / sizeof(seconds[0])),
              max_seconds);
#endif
    if (!mfem::Mpi::Root())
    {
      return;
    }

    std::cout << "[igr-linear] event=exact-dg-face-timing"
              << " solve_call=" << solve_call
              << " operator_mult_calls=" << max_calls
              << " solve_iters=" << solve_iters
              << " solve_s=" << solve_seconds
              << " ranks=" << mfem::Mpi::WorldSize()
              << " precond=" << precond_name
              << " max_base_s=" << max_seconds[0]
              << " max_face_s=" << max_seconds[1]
              << " max_sum_axpy_s=" << max_seconds[2]
              << " max_face_setup_s=" << max_seconds[3]
              << " max_prolong_s=" << max_seconds[4]
              << " max_local_apply_s=" << max_seconds[5]
              << " max_shared_exchange_s=" << max_seconds[6]
              << " max_shared_apply_s=" << max_seconds[7]
              << " max_restrict_s=" << max_seconds[8]
              << " host_wall=1"
              << " sync_inserted=0"
              << std::endl;
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    ::dg_igr::NVTXRange range("IGR/exact-sum/Mult");
    ++operator_mult_calls_;
    if (!timing_enabled_)
    {
      {
        ::dg_igr::NVTXRange base_range("IGR/exact-sum/base");
        base_->Mult(x, y);
      }
      {
        ::dg_igr::NVTXRange face_range("IGR/exact-sum/face");
        face_->Mult(x, scratch_);
      }
      {
        ::dg_igr::NVTXRange axpy_range("IGR/exact-sum/axpy");
        y += scratch_;
      }
      return;
    }

    const auto base_start = Clock::now();
    {
      ::dg_igr::NVTXRange base_range("IGR/exact-sum/base");
      base_->Mult(x, y);
    }
    timing_.base_seconds += ElapsedSeconds(base_start);

    const auto face_start = Clock::now();
    {
      ::dg_igr::NVTXRange face_range("IGR/exact-sum/face");
      face_->Mult(x, scratch_);
    }
    timing_.face_seconds += ElapsedSeconds(face_start);

    const auto axpy_start = Clock::now();
    {
      ::dg_igr::NVTXRange axpy_range("IGR/exact-sum/axpy");
      y += scratch_;
    }
    timing_.sum_axpy_seconds += ElapsedSeconds(axpy_start);
    ++timing_.mult_calls;
  }

  void AssembleDiagonal(mfem::Vector& diag) const override
  {
    base_->AssembleDiagonal(diag);
    face_->AssembleDiagonal(scratch_);
    diag += scratch_;
  }

private:
  mfem::Operator* base_ = nullptr;
  mfem::Operator* face_ = nullptr;
  bool own_base_ = true;
  bool own_face_ = true;
  ExactL2DGFaceOperator* exact_face_ = nullptr;
  bool timing_enabled_ = false;
  mutable mfem::Vector scratch_;
  mutable long long operator_mult_calls_ = 0;
  mutable ExactFaceTimingStats timing_;
};

} // namespace

void IGRSolver::BuildBilinearForm_()
{
  // Any existing exact-face sum refers to the old bilinear form and face
  // layout. Drop both before rebuilding the coefficient-dependent forms.
  Ah_.Clear();
  exact_face_operator_.reset();

  const bool debug_qf =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=build-bilinear-start" << "\n";
  }
  a_ = std::make_unique<mfem::ParBilinearForm>(&sfes_);
  use_partial_assembly_ = ShouldUsePartialAssembly_();
  if (use_partial_assembly_)
  {
    a_->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
  }

  const bool use_device_qf = UseDeviceCoefficients_();
  using_device_coeffs_ = use_device_qf;
  if (use_device_qf)
  {
    EnsureDeviceQuadrature_();
  }

  mfem::Coefficient* invrho = nullptr;
  mfem::Coefficient* ck = nullptr;
  mfem::Coefficient* src = nullptr;
  if (use_device_qf)
  {
    invrho = invrho_qf_coef_.get();
    ck = ck_qf_coef_.get();
    src = src_qf_coef_.get();
  }
  else
  {
    invrho = invrho_coef_.get();
    ck = ck_coef_.get();
    src = src_coef_.get();
  }

  // Reaction/mass: (1/ρ σ, v)
  auto* mass_integ = new mfem::MassIntegrator(*invrho);
  if (use_device_qf && mass_qs_)
  {
    mass_integ->SetIntRule(&mass_qs_->GetIntRule(0));
  }
  a_->AddDomainIntegrator(mass_integ);

  // Volume diffusion term: ( (α/ρ) ∇σ, ∇v )
  auto* diff_integ = new mfem::DiffusionIntegrator(*ck);
  if (use_device_qf && sigma_qs_)
  {
    diff_integ->SetIntRule(&sigma_qs_->GetIntRule(0));
  }
  a_->AddDomainIntegrator(diff_integ);

  // If Σ-space is discontinuous, add IPDG face terms (interior only).
  const bool use_exact_face_operator = UseExactDGFaceOperator_();
  if (IsDGSpace(sfes_) && !use_exact_face_operator)
  {
    if (debug_qf && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=build-bilinear-dg-face"
                << " sigma_fqs=" << (sigma_fqs_ ? 1 : 0)
                << " ck_face_qf_coef=" << (ck_face_qf_coef_ ? 1 : 0)
                << std::endl;
    }
    const double sigma = static_cast<double>(ip_kind_); // -1(SIPG), 0, +1
    const double kappa = IPDGPenalty(sfes_);            // penalty scaling
    mfem::Coefficient* ck_face =
      use_device_qf ? ck_face_qf_coef_.get() : ck;
    auto* dg_diff = new mfem::DGDiffusionIntegrator(*ck_face, sigma, kappa);
    if (use_device_qf && sigma_fqs_)
    {
      dg_diff->SetIntRule(&sigma_fqs_->GetIntRule(0));
    }
    a_->AddInteriorFaceIntegrator(dg_diff);
    // For the IGR sigma equation, default behavior is natural boundary
    // conditions. Do not add DG boundary diffusion terms unless a specific
    // weak boundary closure is implemented.
  }
  else if (IsDGSpace(sfes_) && debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=build-bilinear-dg-face"
              << " mode=exact-side-qdata-wrapper"
              << " stock_mfem_face=0"
              << std::endl;
  }

  b_ = std::make_unique<mfem::ParLinearForm>(&sfes_);
  auto* rhs_integ = new mfem::DomainLFIntegrator(*src);
  if (use_device_qf && rhs_qs_)
  {
    rhs_integ->SetIntRule(&rhs_qs_->GetIntRule(0));
  }
  b_->AddDomainIntegrator(rhs_integ);
  b_->UseFastAssembly(use_partial_assembly_);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=build-bilinear-done" << "\n";
  }
}

bool IGRSolver::ShouldUsePartialAssembly_() const
{
  const bool device_backend =
    mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  if (device_backend)
  {
    const bool requested =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_PARTIAL_ASSEMBLY", true);
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=partial-assembly-policy"
                << " requested=" << (requested ? "on" : "off")
                << " effective=on"
                << " reason=device-required-partial-assembly"
                << " precond=" << PrecondTypeName(prec_type_) << "\n";
    }
    return true;
  }

  bool use_partial =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_PARTIAL_ASSEMBLY", false);
  if (use_partial && PreconditionerRequiresAssembledMatrix(prec_type_))
  {
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=partial-assembly-policy"
                << " requested=on"
                << " effective=off"
                << " reason=matrix-preconditioner"
                << " precond=" << PrecondTypeName(prec_type_) << "\n";
    }
    use_partial = false;
  }
  return use_partial;
}

bool IGRSolver::UseExactDGFaceOperator_() const
{
  return exact_dg_face_operator_ ||
         ::dg_igr::EnvFlagEnabledDefault(
           "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR",
           false);
}

bool IGRSolver::ShouldUseDirectIdentitySystem_(bool use_device_qf,
                                               std::string& reason) const
{
  const bool requested =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_DIRECT_IDENTITY_SYSTEM",
                                    true);
  if (!requested)
  {
    reason = "disabled-by-env";
    return false;
  }
  if (!IsDGSpace(sfes_))
  {
    reason = "space-not-dg";
    return false;
  }
  if (!use_partial_assembly_)
  {
    reason = "partial-assembly-disabled";
    return false;
  }
  if (!use_device_qf)
  {
    reason = "device-qf-disabled";
    return false;
  }
  if (ess_tdof_list_.Size() != 0)
  {
    reason = "essential-bc";
    return false;
  }
  const mfem::Operator* prolongation = sfes_.GetProlongationMatrix();
  if (!mfem::IsIdentityProlongation(prolongation))
  {
    reason = "nonidentity-prolongation";
    return false;
  }
  if (sfes_.GetTrueVSize() != sfes_.GetVSize())
  {
    reason = "true-local-size-mismatch";
    return false;
  }
  if (!a_ || !b_)
  {
    reason = "forms-not-built";
    return false;
  }
  if (a_->Height() != sfes_.GetVSize() || b_->Size() != sfes_.GetVSize())
  {
    reason = "form-size-mismatch";
    return false;
  }
  reason = "active";
  return true;
}

mfem::Operator* IGRSolver::ActiveLinearOperator_() const
{
  if (direct_identity_system_)
  {
    if (Ah_.Ptr())
    {
      return Ah_.Ptr();
    }
    return a_.get();
  }
  return Ah_.Ptr();
}

void IGRSolver::AssembleSystem(const mfem::ParGridFunction& U)
{
  ::dg_igr::NVTXRange range("IGR/assemble-system");
  const auto assemble_start = std::chrono::steady_clock::now();

  // Needed for DG/IPDG assembly: variable coeffs evaluated on shared faces
  // will read U on neighbor elements.
  // For H1 sigma spaces, no DG/IP face terms are assembled, so this exchange is
  // unnecessary. In serial runs, neighbor exchange is also a no-op with extra
  // overhead; skip it in that case as well.
  if (IsDGSpace(sfes_) && mfem::Mpi::WorldSize() > 1)
  {
    ::dg_igr::NVTXRange exchange_range("IGR/state-face-neighbor-exchange");
    const_cast<mfem::ParGridFunction&>(U).ExchangeFaceNbrData();
  }
  MFEM_VERIFY(VectorIsFinite(U),
              "IGR state contains non-finite values before linear assembly.");

  const bool device_backend = mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  const bool use_device_qf = UseDeviceCoefficients_();
  const bool exact_face_requested = UseExactDGFaceOperator_();
  const bool is_dg_space = IsDGSpace(sfes_);
  const bool use_exact_face_operator = is_dg_space && exact_face_requested;
  last_exact_face_requested_ = exact_face_requested;
  last_exact_face_effective_ = false;
  last_exact_face_reason_ =
    exact_face_requested
      ? (is_dg_space ? "pending" : "space-not-dg")
      : "not-requested";
  last_exact_face_local_nf_ = 0;
  last_exact_face_shared_nf_ = 0;
  last_exact_face_operator_mult_calls_ = 0;
  const bool debug_qf =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=assemble-use-device-qf"
              << " enabled=" << (use_device_qf ? 1 : 0) << "\n";
  }
  if (use_exact_face_operator)
  {
    if (!use_device_qf)
    {
      last_exact_face_reason_ = "requires-device-qf";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR requires "
                  "DG_IGR_IGR_DEVICE_QF=1.");
    }
    if (alpha_length_scale_policy_ == AlphaLengthScalePolicy::Local)
    {
      last_exact_face_reason_ = "requires-nonlocal-alpha";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR currently supports only "
                  "constant alpha policy.");
    }
    if (ess_bdr_.Size() != 0)
    {
      last_exact_face_reason_ = "requires-natural-boundary";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR currently supports only "
                  "natural/no essential sigma boundary conditions.");
    }
  }

  if (use_device_qf != using_device_coeffs_)
  {
    BuildBilinearForm_();
  }

  if (use_device_qf)
  {
    ::dg_igr::NVTXRange coeff_range("IGR/device-coefficients");
    UpdateDeviceCoefficients_(U);
  }
  else
  {
    ::dg_igr::NVTXRange coeff_range("IGR/host-coefficients");
    // Ensure host-visible state for coefficient evaluation.
    // The IGR coefficients call GridFunction::GetVectorValue(), which is host-only.
    // On device backends, default to syncing unless explicitly disabled.
    const bool hostsync_default = device_backend;
    if (::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_FORCE_STATE_HOSTSYNC",
                                        hostsync_default))
    {
      (void)U.HostRead();
    }

    // Refresh coefficient views of the current state and params
    invrho_coef_->SetState(U);
    invrho_coef_->SetRhoFloor(rho_floor_);

    // Build gradient backend inputs
    if (backend_ == GradBackend::H1Projection)
    {
      EnsureH1VelocitySpace_();

      // Project velocity u = m/ρ into H1^dim
      VelocityFromConsCoefficient vel_coeff(dim_, &U, rho_floor_);
      vel_coeff.SetRhoFloor(rho_floor_);
      vel_coeff.SetState(&U);

      // DG state-derived velocity is generally discontinuous; use arithmetic
      // averaging at shared H1 dofs for partition-independent projection.
      vel_h1_->ProjectDiscCoefficient(vel_coeff, mfem::GridFunction::ARITHMETIC);

      // Tell RHS coefficient to use this continuous velocity field
      src_coef_->SetVelField(vel_h1_.get());
    }
    else
    {
      // BrokenDG backend: remove any projected velocity
      src_coef_->SetVelField(nullptr);
    }

    src_coef_->SetState(U);
    src_coef_->SetAlpha(alpha_);
    src_coef_->SetAlphaCoefficient(alpha_coef_.get());
    src_coef_->SetRhoFloor(rho_floor_);
  }

  // Keep MFEM form internals consistent before re-assembly across policy modes.
  a_->Update();
  b_->Update();
  b_->UseFastAssembly(use_partial_assembly_);

  // Re-assemble values (sparsity pattern unchanged)
  {
    ::dg_igr::NVTXRange assemble_form_range("IGR/mfem-form-assemble");
    a_->Assemble();
    b_->Assemble();
  }

  // Rebuild essential dofs if user set Dirichlet BCs (for H1 spaces).
  ess_tdof_list_.SetSize(0);
  if (ess_bdr_.Size() > 0)
  {
    mfem::Array<int> dbdr(ess_bdr_);
    sfes_.GetEssentialTrueDofs(dbdr, ess_tdof_list_);
  }

  const auto assemble_stop = std::chrono::steady_clock::now();
  last_assemble_seconds_ =
    std::chrono::duration<double>(assemble_stop - assemble_start).count();
  total_assemble_seconds_ += last_assemble_seconds_;

  // Rebuild the system operator; clear handle first.
  Ah_.Clear();
  Ah_.SetType(use_partial_assembly_ ? mfem::Operator::ANY_TYPE
                                    : mfem::Operator::Hypre_ParCSR);
  direct_identity_system_ = false;
  last_direct_identity_reason_ = "not-evaluated";

  // Warm-start: FormLinearSystem pulls current sigma_ into X_
  const auto form_start = std::chrono::steady_clock::now();
  std::string direct_identity_reason;
  {
    ::dg_igr::NVTXRange form_range("IGR/form-linear-system");
    if (ShouldUseDirectIdentitySystem_(use_device_qf, direct_identity_reason))
    {
      X_.MakeRef(sigma_, 0, sigma_.Size());
      B_.MakeRef(*b_, 0, b_->Size());
      direct_identity_system_ = true;
      ++direct_identity_system_uses_;
    }
    else
    {
      a_->FormLinearSystem(ess_tdof_list_, sigma_, *b_, Ah_, X_, B_);
    }
  }
  last_direct_identity_reason_ = direct_identity_reason;

  if (use_exact_face_operator)
  {
    ::dg_igr::NVTXRange exact_setup_range("IGR/exact-face-setup");
    if (!use_partial_assembly_)
    {
      last_exact_face_reason_ = "requires-partial-assembly";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR requires partial assembly.");
    }
    mfem::Mesh* mesh = sfes_.GetMesh();
    if (!mesh)
    {
      last_exact_face_reason_ = "requires-valid-mesh";
      MFEM_VERIFY(false,
                  "DG_IGR_IGR_EXACT_DG_FACE_OPERATOR requires a valid mesh.");
    }
    auto* pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
    const mfem::Geometry::Type face_geom = mesh->GetTypicalFaceGeometry();
    const mfem::IntegrationRule& exact_face_ir =
      mfem::IntRules.Get(face_geom, 2 * SpaceOrder(sfes_));
    mfem::FaceQuadratureSpace exact_fqs(*mesh,
                                        exact_face_ir,
                                        mfem::FaceType::Interior);
    std::string reason;
    if (!dg_face::SupportsInteriorFaceSideDensityProjection2D(vfes_,
                                                              exact_fqs,
                                                              &reason))
    {
      last_exact_face_reason_ = "unsupported-interior-side-density";
      MFEM_VERIFY(false,
                  "Unsupported exact IGR side-density face operator: "
                    << reason);
    }
    if (!dg_face::SupportsInteriorFaceScalarTraceProjection2D(sfes_,
                                                              exact_fqs,
                                                              &reason))
    {
      last_exact_face_reason_ = "unsupported-interior-sigma-trace";
      MFEM_VERIFY(false,
                  "Unsupported exact IGR sigma face operator: " << reason);
    }

    mfem::Vector side_q;
    dg_face::InteriorFaceSideDensityProjectionScratch side_scratch;
    const auto side_info =
      dg_face::ProjectInteriorFaceDensityCoefficient2D(vfes_,
                                                       exact_fqs,
                                                       U,
                                                       alpha_,
                                                       rho_floor_,
                                                       side_q,
                                                       &side_scratch);
    last_exact_face_local_nf_ = side_info.nf;

    mfem::Vector shared_side_q;
    dg_face::SharedFaceSideDensityProjectionScratch shared_side_scratch;
    dg_face::SharedFaceSideDensityProjectionInfo shared_side_info;
    if (pmesh && pmesh->GetNSharedFaces() > 0)
    {
      if (!dg_face::SupportsSharedFaceSideDensityProjection2D(vfes_,
                                                              exact_face_ir,
                                                              &reason))
      {
        last_exact_face_reason_ = "unsupported-shared-side-density";
        MFEM_VERIFY(false,
                    "Unsupported exact IGR shared side-density face operator: "
                      << reason);
      }
      if (!dg_face::SupportsSharedFaceDGDiffusionApplyData2D(sfes_,
                                                             exact_face_ir,
                                                             &reason))
      {
        last_exact_face_reason_ = "unsupported-shared-sigma-trace";
        MFEM_VERIFY(false,
                    "Unsupported exact IGR shared sigma face operator: "
                      << reason);
      }
      shared_side_info =
        dg_face::ProjectSharedFaceDensityCoefficient2D(vfes_,
                                                       exact_face_ir,
                                                       U,
                                                       alpha_,
                                                       rho_floor_,
                                                       shared_side_q,
                                                       &shared_side_scratch);
      last_exact_face_shared_nf_ = shared_side_info.nf;
    }

    mfem::Operator* base_op =
      direct_identity_system_ ? static_cast<mfem::Operator*>(a_.get())
                              : Ah_.Ptr();
    MFEM_VERIFY(base_op != nullptr,
                "Exact IGR face operator requires formed base operator.");
    ExactL2DGFaceOperator* face_op = nullptr;
    const mfem::real_t face_kappa =
      static_cast<mfem::real_t>(IPDGPenalty(sfes_));
    // Keep the stage-wise exact-face refresh on the original, fully rebuilt
    // path until the device qdata update kernel has an end-to-end parity test.
    // Reusing the geometry is an optimization, not a correctness requirement.
    exact_face_operator_.reset();
    exact_face_operator_ = std::make_unique<ExactL2DGFaceOperator>(
      sfes_,
      exact_fqs,
      side_q,
      exact_face_ir,
      shared_side_q,
      static_cast<mfem::real_t>(ip_kind_),
      face_kappa);
    face_op = dynamic_cast<ExactL2DGFaceOperator*>(
      exact_face_operator_.get());
    MFEM_VERIFY(face_op != nullptr,
                "Exact DG face operator construction failed.");
    if (!direct_identity_system_)
    {
      Ah_.SetOperatorOwner(false);
    }
    Ah_.Reset(new ExactFaceSumOperator(base_op,
                                       face_op,
                                       !direct_identity_system_,
                                       false));
    last_exact_face_effective_ = true;
    last_exact_face_reason_ = "active";

    if (debug_qf && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=exact-dg-face-operator"
                << " enabled=1"
                << " nf=" << side_info.nf
                << " nq=" << side_info.nq
                << " qf_size=" << side_info.qf_size
                << " shared_nf=" << shared_side_info.nf
                << " shared_qf_size=" << shared_side_info.qf_size
                << " mode=local-and-shared-2d"
                << std::endl;
    }
  }
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=exact-dg-face-policy"
              << " requested=" << (last_exact_face_requested_ ? "on" : "off")
              << " effective=" << (last_exact_face_effective_ ? "on" : "off")
              << " reason=" << LogToken(last_exact_face_reason_)
              << " local_nf=" << last_exact_face_local_nf_
              << " shared_nf=" << last_exact_face_shared_nf_
              << " operator_uses=" << exact_face_operator_uses_
              << std::endl;
  }
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=direct-identity-system"
              << " requested="
              << (::dg_igr::EnvFlagEnabledDefault(
                    "DG_IGR_IGR_DIRECT_IDENTITY_SYSTEM",
                    true) ? "on" : "off")
              << " effective=" << (direct_identity_system_ ? "on" : "off")
              << " reason=" << LogToken(last_direct_identity_reason_)
              << " uses=" << direct_identity_system_uses_
              << std::endl;
  }
  const auto form_stop = std::chrono::steady_clock::now();
  last_form_system_seconds_ =
    std::chrono::duration<double>(form_stop - form_start).count();
  total_form_system_seconds_ += last_form_system_seconds_;
}

void IGRSolver::DoSmootherSweeps(int sweeps, bool /*warm_start*/)
{
  ::dg_igr::NVTXRange range("IGR/smoother-sweeps");
  mfem::Operator* A_op = ActiveLinearOperator_();
  MFEM_VERIFY(A_op != nullptr,
              "IGR: linear operator is not assembled before smoother sweeps.");
  mfem::HypreParMatrix* A_hypre =
    direct_identity_system_ ? nullptr : Ah_.Is<mfem::HypreParMatrix>();

  std::unique_ptr<mfem::Solver> smoother;
  if (A_hypre)
  {
    auto hypre_sm = std::make_unique<mfem::HypreSmoother>(*A_hypre);
    switch (smoot_)
    {
    case SmootherType::Jacobi:
      hypre_sm->SetType(mfem::HypreSmoother::Jacobi);
      break;
    case SmootherType::L1Jacobi:
      hypre_sm->SetType(mfem::HypreSmoother::l1Jacobi);
      break;
    case SmootherType::L1GS:
      hypre_sm->SetType(mfem::HypreSmoother::l1GS);
      break;
    case SmootherType::L1GStr:
      hypre_sm->SetType(mfem::HypreSmoother::l1GStr);
      break;
    case SmootherType::GS:
      hypre_sm->SetType(mfem::HypreSmoother::GS);
      break;
    }
    hypre_sm->SetPositiveDiagonal(true);
    smoother = std::move(hypre_sm);
  }
  else
  {
    smoother = MakePositiveOperatorJacobi(*a_, ess_tdof_list_);
  }

  mfem::Vector r(B_.Size()), z(X_.Size());
  for (int it = 0; it < sweeps; ++it)
  {
    // r = b - A x
    r = B_;
    A_op->Mult(X_, r);
    r.Neg();
    r += B_;

    // z ≈ M^{-1} r  (one application of the smoother/preconditioner)
    z = 0.0;
    smoother->Mult(r, z);

    // x <- x + z  (Richardson step)
    X_ += z;
  }

  if (direct_identity_system_)
  {
    X_.SyncAliasMemory(sigma_);
  }
  else
  {
    a_->RecoverFEMSolution(X_, *b_, sigma_);
  }
}

void IGRSolver::SolveEllipticSystem_(bool warm_start)
{
  ::dg_igr::NVTXRange range("IGR/solve-elliptic-system");
  mfem::Operator* A_op = ActiveLinearOperator_();
  MFEM_VERIFY(A_op != nullptr,
              "IGR: linear operator is not assembled before Krylov solve.");
  mfem::HypreParMatrix* A_hypre =
    direct_identity_system_ ? nullptr : Ah_.Is<mfem::HypreParMatrix>();
  const bool has_hypre_matrix = (A_hypre != nullptr);
  auto* exact_face_sum_op = dynamic_cast<ExactFaceSumOperator*>(A_op);

  MPI_Comm comm = sfes_.GetComm();
  const bool is_dg = IsDGSpace(sfes_);
  const bool spd_ok = IsSPD_DefaultAssumption(sfes_, ip_kind_);
  const bool using_cuda = IsCudaBackendActive();
  const bool prev_linear_converged = last_linear_converged_;
  const double prev_linear_final_norm = last_linear_final_norm_;

  last_linear_iters_ = 0;
  last_linear_converged_ = false;
  last_linear_final_norm_ = std::numeric_limits<double>::quiet_NaN();
  last_linear_initial_norm_ = std::numeric_limits<double>::quiet_NaN();
  last_linear_solve_seconds_ = 0.0;
  last_pc_setup_seconds_ = 0.0;
  last_chebyshev_diagonal_seconds_ = 0.0;
  last_chebyshev_setup_seconds_ = 0.0;
  last_chebyshev_eigenvalue_reused_ = false;
  last_chebyshev_smoother_reused_ = false;
  last_chebyshev_eigenvalue_estimate_ = 0.0;
  last_krylov_operator_apply_seconds_ = 0.0;
  last_krylov_preconditioner_apply_seconds_ = 0.0;
  last_krylov_preconditioner_operator_apply_seconds_ = 0.0;
  last_krylov_operator_apply_count_ = 0;
  last_krylov_preconditioner_apply_count_ = 0;
  last_krylov_preconditioner_operator_apply_count_ = 0;
  last_linear_solver_name_ = "none";
  last_precond_name_ = "none";

  // --- Choose solver (Auto policy is space-aware) ---
  const LinearSolverType requested_solver = lin_solver_type_;
  LinearSolverType solver_choice = lin_solver_type_;
  if (solver_choice == LinearSolverType::Auto)
  {
    solver_choice = spd_ok ? LinearSolverType::CG : LinearSolverType::GMRES;
  }

  // Guardrail: do not use SPD-only Krylov methods when SPD assumptions fail.
  if (RequiresSPD(solver_choice) && !spd_ok)
  {
    const LinearSolverType prev = solver_choice;
    solver_choice = LinearSolverType::GMRES;
    LogIGRFallback("operator-not-spd",
                   prev,
                   solver_choice,
                   prec_type_,
                   prec_type_);
  }

  // --- Choose preconditioner (Auto policy is space-aware) ---
  PrecondType prec_choice = prec_type_;
  if (prec_choice == PrecondType::Auto)
  {
    // CPU: keep BoomerAMG as robust default.
    // CUDA: prefer HypreSmoother for lower setup/solve overhead on current
    // inoue-style runs; fallback policy remains active for unsupported paths.
    (void)is_dg;
    prec_choice =
      using_cuda ? PrecondType::HypreSmoother : PrecondType::BoomerAMG;
  }

  // Guardrail: CG/MINRES should use symmetric/no preconditioners.
  if (RequiresSPD(solver_choice) && !IsSymmetricPC(prec_choice))
  {
    if (using_cuda)
    {
      MFEM_VERIFY(false,
                  "IGR device-closure requires symmetric preconditioning for "
                  "SPD Krylov solvers; requested preconditioner is incompatible.");
    }
    const PrecondType prev = prec_choice;
    prec_choice = PrecondType::BoomerAMG;
    LogIGRFallback("spd-solver-with-nonsymmetric-preconditioner",
                   requested_solver,
                   solver_choice,
                   prev,
                   prec_choice);
  }

  if (using_cuda)
  {
    MFEM_VERIFY(!has_hypre_matrix,
                "IGR device-closure requires partial-assembly operator path "
                "(no assembled Hypre matrix).");
    MFEM_VERIFY(prec_choice == PrecondType::HypreSmoother ||
                  prec_choice == PrecondType::OperatorChebyshev ||
                  prec_choice == PrecondType::None,
                "IGR device-closure on CUDA supports only "
                "--igr-preconditioner hypresmoother|chebyshev|none.");
  }
  else if (!has_hypre_matrix &&
           (prec_choice == PrecondType::BoomerAMG ||
            prec_choice == PrecondType::HypreILU ||
            prec_choice == PrecondType::HypreEuclid ||
            prec_choice == PrecondType::ParaSails))
  {
    const PrecondType prev = prec_choice;
    prec_choice = PrecondType::HypreSmoother;
    LogIGRFallback("operator-path-without-hypre-matrix",
                   requested_solver,
                   solver_choice,
                   prev,
                   prec_choice);
  }

  KrylovApplyTelemetry krylov_telemetry;
  krylov_telemetry.record_seconds =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_TIMING",
                                    false);
  krylov_telemetry.sync =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_SYNC",
                                    false);
  const bool use_krylov_telemetry =
    krylov_telemetry.record_seconds ||
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_KRYLOV_APPLY_COUNTS",
                                    true);

  const auto pc_setup_start = std::chrono::steady_clock::now();
  std::unique_ptr<KrylovTelemetryOperator> cheb_operator_telemetry;
  std::unique_ptr<mfem::Solver> prec;
  bool using_operator_jacobi = false;
  bool using_operator_jacobi_exact_face_diag = false;
  bool using_operator_chebyshev = false;
  bool using_operator_chebyshev_base = false;
  {
    ::dg_igr::NVTXRange pc_range("IGR/preconditioner-setup");
    if (prec_choice == PrecondType::BoomerAMG)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "BoomerAMG preconditioner requires a Hypre matrix operator.");
      auto amg = std::make_unique<mfem::HypreBoomerAMG>(*A_hypre);
      // Typical AMG-as-preconditioner settings
      amg->SetPrintLevel(0);
      amg->SetMaxIter(1);
      amg->SetTol(0.0);
      if (amg_coarsen_type_ >= 0)
      {
        amg->SetCoarsening(amg_coarsen_type_);
      }
      if (amg_relax_type_ >= 0)
      {
        amg->SetRelaxType(amg_relax_type_);
      }
      if (amg_max_levels_ >= 0)
      {
        amg->SetMaxLevels(amg_max_levels_);
      }
      if (amg_strong_threshold_ >= 0.0)
      {
        amg->SetStrengthThresh(amg_strong_threshold_);
      }
      prec = std::move(amg);
    }
    else if (prec_choice == PrecondType::HypreILU)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "HypreILU preconditioner requires a Hypre matrix operator.");
      auto ilu = std::make_unique<mfem::HypreILU>();
      ilu->SetPrintLevel(0);
      ilu->SetMaxIter(1);
      ilu->SetTol(0.0);
      ilu->SetOperator(*A_hypre);
      prec = std::move(ilu);
    }
    else if (prec_choice == PrecondType::HypreEuclid)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "HypreEuclid preconditioner requires a Hypre matrix operator.");
      auto euclid = std::make_unique<mfem::HypreEuclid>(*A_hypre);
      euclid->SetStats(0);
      prec = std::move(euclid);
    }
    else if (prec_choice == PrecondType::ParaSails)
    {
      MFEM_VERIFY(A_hypre != nullptr,
                  "ParaSails preconditioner requires a Hypre matrix operator.");
      auto parasails = std::make_unique<mfem::HypreParaSails>(*A_hypre);
      parasails->SetLogging(0);
      prec = std::move(parasails);
    }
    else if (prec_choice == PrecondType::HypreSmoother)
    {
      if (A_hypre)
      {
        auto sm = std::make_unique<mfem::HypreSmoother>(*A_hypre);
        switch (smoot_)
        {
        case SmootherType::Jacobi:
          sm->SetType(mfem::HypreSmoother::Jacobi);
          break;
        case SmootherType::L1Jacobi:
          sm->SetType(mfem::HypreSmoother::l1Jacobi);
          break;
        case SmootherType::L1GS:
          sm->SetType(mfem::HypreSmoother::l1GS);
          break;
        case SmootherType::L1GStr:
          sm->SetType(mfem::HypreSmoother::l1GStr);
          break;
        case SmootherType::GS:
          sm->SetType(mfem::HypreSmoother::GS);
          break;
        }
        sm->SetPositiveDiagonal(true);
        prec = std::move(sm);
      }
      else
      {
        const bool use_exact_face_diag =
          exact_face_sum_op &&
          ::dg_igr::EnvFlagEnabledDefault(
            "DG_IGR_IGR_EXACT_FACE_DIAGONAL_PRECOND",
            false);
        if (use_exact_face_diag)
        {
          prec = MakePositiveOperatorJacobi(*A_op, ess_tdof_list_);
          using_operator_jacobi_exact_face_diag = true;
        }
        else
        {
          prec = MakePositiveOperatorJacobi(*a_, ess_tdof_list_);
        }
        using_operator_jacobi = true;
      }
    }
    else if (prec_choice == PrecondType::OperatorChebyshev)
    {
      ::dg_igr::NVTXRange cheb_range("IGR/chebyshev-setup");
      const bool use_base_operator =
        exact_face_sum_op &&
        ::dg_igr::EnvFlagEnabledDefault(
          "DG_IGR_IGR_CHEB_BASE_OPERATOR",
          using_cuda);
      mfem::Operator* cheb_op =
        use_base_operator
          ? &exact_face_sum_op->BaseOperator()
          : A_op;
      mfem::Vector diag(cheb_op->Height());
      const auto cheb_diag_start = Clock::now();
      {
        ::dg_igr::NVTXRange diag_range("IGR/chebyshev-diagonal");
        cheb_op->AssembleDiagonal(diag);
      }
      last_chebyshev_diagonal_seconds_ = ElapsedSeconds(cheb_diag_start);
      total_chebyshev_diagonal_seconds_ +=
        last_chebyshev_diagonal_seconds_;
      const int cheb_order =
        ReadEnvInt("DG_IGR_IGR_CHEB_ORDER",
                   (using_cuda && exact_face_sum_op) ? 3 : 2);
      const int cheb_power_iterations =
        ReadEnvInt("DG_IGR_IGR_CHEB_POWER_ITERATIONS", 10);
      const mfem::real_t cheb_power_tolerance =
        ReadEnvReal("DG_IGR_IGR_CHEB_POWER_TOLERANCE", 1.0e-8);
      const int cheb_power_seed =
        ReadEnvInt("DG_IGR_IGR_CHEB_POWER_SEED", 12345);
      MFEM_VERIFY(cheb_order >= 1 && cheb_order <= 5,
                  "DG_IGR_IGR_CHEB_ORDER must be between 1 and 5 for "
                  "MFEM OperatorChebyshevSmoother.");
      MFEM_VERIFY(cheb_power_iterations > 0,
                  "DG_IGR_IGR_CHEB_POWER_ITERATIONS must be positive.");
      MFEM_VERIFY(cheb_power_tolerance > 0.0,
                  "DG_IGR_IGR_CHEB_POWER_TOLERANCE must be positive.");
      MFEM_VERIFY(cheb_power_seed != 0,
                  "DG_IGR_IGR_CHEB_POWER_SEED must be nonzero.");
      if ((lin_print_ > 0 ||
           ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false)) &&
          mfem::Mpi::Root())
      {
        std::cout << "[igr-linear] event=operator-chebyshev-config"
                  << " order=" << cheb_order
                  << " operator="
                  << (use_base_operator ? "base-pa" : "solve")
                  << " power_iterations=" << cheb_power_iterations
                  << " power_tolerance=" << cheb_power_tolerance
                  << " power_seed=" << cheb_power_seed
                  << std::endl;
      }
      mfem::Operator* cheb_smoother_op = cheb_op;
      if (use_krylov_telemetry)
      {
        cheb_operator_telemetry =
          std::make_unique<KrylovTelemetryOperator>(*cheb_op,
                                                    krylov_telemetry);
        cheb_smoother_op = cheb_operator_telemetry.get();
      }
      const auto cheb_setup_start = Clock::now();
#ifdef MFEM_USE_MPI
      mfem::PowerMethod cheb_power_method(comm);
#else
      mfem::PowerMethod cheb_power_method;
#endif
      mfem::OperatorJacobiSmoother inv_diag(diag, ess_tdof_list_, 1.0);
      mfem::ProductOperator diag_precond(&inv_diag,
                                         cheb_smoother_op,
                                         false,
                                         false);
      mfem::Vector cheb_power_vec(cheb_smoother_op->Width());
      const mfem::real_t max_eig_estimate =
        cheb_power_method.EstimateLargestEigenvalue(diag_precond,
                                                    cheb_power_vec,
                                                    cheb_power_iterations,
                                                    cheb_power_tolerance,
                                                    cheb_power_seed);
      last_chebyshev_eigenvalue_estimate_ =
        static_cast<double>(max_eig_estimate);
      auto cheb = std::make_unique<mfem::OperatorChebyshevSmoother>(
        *cheb_smoother_op,
        diag,
        ess_tdof_list_,
        cheb_order,
        max_eig_estimate);
      last_chebyshev_setup_seconds_ = ElapsedSeconds(cheb_setup_start);
      total_chebyshev_setup_seconds_ += last_chebyshev_setup_seconds_;
      ++total_chebyshev_smoother_builds_;
      prec = std::move(cheb);
      using_operator_chebyshev = true;
      using_operator_chebyshev_base = use_base_operator;
    }
    // else None -> leave prec null
  }
  const auto pc_setup_stop = std::chrono::steady_clock::now();
  last_pc_setup_seconds_ =
    std::chrono::duration<double>(pc_setup_stop - pc_setup_start).count();
  total_pc_setup_seconds_ += last_pc_setup_seconds_;
  krylov_telemetry.ResetCounters();

  if (lin_print_ > 0)
  {
    LogIGRSelection(warm_start,
                    is_dg,
                    spd_ok,
                    using_cuda,
                    solver_choice,
                    prec_choice,
                    lin_rtol_,
                    lin_atol_,
                    lin_maxit_);
  }

  // Warm start is only safe if the prior solve produced a usable iterate.
  // If not, fall back to a cold-start solve to avoid poisoning Krylov with
  // stale/non-finite data.
  bool enable_warm_start = warm_start;
  if (enable_warm_start && linear_solve_calls_ > 0)
  {
    if (!prev_linear_converged || !std::isfinite(prev_linear_final_norm))
    {
      enable_warm_start = false;
      if (lin_print_ > 0)
      {
        LogIGRFallback("warm-start-disabled-after-unsuccessful-previous-solve",
                       solver_choice,
                       solver_choice,
                       prec_choice,
                       prec_choice);
      }
    }
  }
  if (!enable_warm_start || !VectorIsFinite(X_))
  {
    X_ = 0.0;
    enable_warm_start = false;
  }
  MFEM_VERIFY(VectorIsFinite(B_),
              "IGR RHS contains non-finite values before Krylov solve.");

  const bool is_intermediate_stage = (rk_stage_index_ >= 0 && rk_stage_index_ < 2);
  const double effective_rtol =
    is_intermediate_stage ? (lin_rtol_ * stage_rtol_scale_) : lin_rtol_;

  std::unique_ptr<KrylovTelemetryOperator> solve_operator_telemetry;
  std::unique_ptr<KrylovTelemetryPreconditioner> telemetry_preconditioner;
  mfem::Operator* solve_operator = A_op;
  mfem::Solver* solve_preconditioner = prec.get();
  if (use_krylov_telemetry)
  {
    solve_operator_telemetry =
      std::make_unique<KrylovTelemetryOperator>(*A_op, krylov_telemetry);
    solve_operator = solve_operator_telemetry.get();
    if (prec)
    {
      telemetry_preconditioner =
        std::make_unique<KrylovTelemetryPreconditioner>(*prec,
                                                        krylov_telemetry);
      solve_preconditioner = telemetry_preconditioner.get();
    }
  }

  auto configure_solver = [&](mfem::IterativeSolver& solver)
  {
    solver.SetOperator(*solve_operator);
    if (solve_preconditioner)
    {
      solver.SetPreconditioner(*solve_preconditioner);
    }
    solver.SetRelTol(effective_rtol);
    solver.SetAbsTol(lin_atol_);
    solver.SetMaxIter(lin_maxit_);
    solver.SetPrintLevel(lin_print_);
    solver.iterative_mode = enable_warm_start;
  };

  if (exact_face_sum_op)
  {
    exact_face_sum_op->ResetTiming();
  }

  const auto solve_start = std::chrono::steady_clock::now();

  {
    ::dg_igr::NVTXRange krylov_range("IGR/krylov-solve");
    if (solver_choice == LinearSolverType::CG)
    {
      mfem::CGSolver solver(comm);
      configure_solver(solver);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else if (solver_choice == LinearSolverType::GMRES)
    {
      mfem::GMRESSolver solver(comm);
      configure_solver(solver);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else if (solver_choice == LinearSolverType::FGMRES)
    {
      mfem::FGMRESSolver solver(comm);
      configure_solver(solver);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else if (solver_choice == LinearSolverType::BiCGSTAB)
    {
      mfem::BiCGSTABSolver solver(comm);
      configure_solver(solver);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else if (solver_choice == LinearSolverType::MINRES)
    {
      mfem::MINRESSolver solver(comm);
      configure_solver(solver);
      solver.Mult(B_, X_);
      last_linear_iters_ = solver.GetNumIterations();
      last_linear_converged_ = solver.GetConverged();
      last_linear_final_norm_ = solver.GetFinalNorm();
      last_linear_initial_norm_ = solver.GetInitialNorm();
    }
    else
    {
      MFEM_ABORT("Unsupported IGR linear solver selection.");
    }
  }

  const auto solve_stop = std::chrono::steady_clock::now();
  last_linear_solve_seconds_ =
    std::chrono::duration<double>(solve_stop - solve_start).count();
  total_linear_solve_seconds_ += last_linear_solve_seconds_;
  last_krylov_operator_apply_seconds_ =
    krylov_telemetry.operator_apply_seconds;
  last_krylov_preconditioner_apply_seconds_ =
    krylov_telemetry.preconditioner_apply_seconds;
  last_krylov_preconditioner_operator_apply_seconds_ =
    krylov_telemetry.preconditioner_operator_apply_seconds;
  last_krylov_operator_apply_count_ =
    krylov_telemetry.operator_apply_count;
  last_krylov_preconditioner_apply_count_ =
    krylov_telemetry.preconditioner_apply_count;
  last_krylov_preconditioner_operator_apply_count_ =
    krylov_telemetry.preconditioner_operator_apply_count;
  total_krylov_operator_apply_seconds_ +=
    last_krylov_operator_apply_seconds_;
  total_krylov_preconditioner_apply_seconds_ +=
    last_krylov_preconditioner_apply_seconds_;
  total_krylov_preconditioner_operator_apply_seconds_ +=
    last_krylov_preconditioner_operator_apply_seconds_;
  total_krylov_operator_apply_count_ +=
    last_krylov_operator_apply_count_;
  total_krylov_preconditioner_apply_count_ +=
    last_krylov_preconditioner_apply_count_;
  total_krylov_preconditioner_operator_apply_count_ +=
    last_krylov_preconditioner_operator_apply_count_;
  ++linear_solve_calls_;
  total_linear_iters_ += last_linear_iters_;
  max_linear_iters_ = std::max(max_linear_iters_, last_linear_iters_);
  if (last_linear_converged_)
  {
    ++total_linear_converged_calls_;
  }

  last_linear_solver_name_ = LinearSolverTypeName(solver_choice);
  last_precond_name_ = using_operator_chebyshev
                         ? (using_operator_chebyshev_base
                              ? "operatorchebyshev-base"
                              : "operatorchebyshev")
                         : (using_operator_jacobi_exact_face_diag
                              ? "operatorjacobi-exactface"
                              : (using_operator_jacobi
                                   ? "operatorjacobi"
                                   : PrecondTypeName(prec_choice)));

  if (exact_face_sum_op)
  {
    ++exact_face_operator_uses_;
    long long local_mult_calls = exact_face_sum_op->LastMultCalls();
    long long max_mult_calls = local_mult_calls;
#ifdef MFEM_USE_MPI
    MPI_Allreduce(&local_mult_calls,
                  &max_mult_calls,
                  1,
                  MPI_LONG_LONG,
                  MPI_MAX,
                  comm);
#endif
    last_exact_face_operator_mult_calls_ = max_mult_calls;
  }
  else
  {
    last_exact_face_operator_mult_calls_ = 0;
  }

  if (exact_face_sum_op && exact_face_sum_op->TimingEnabled())
  {
    exact_face_sum_op->LogTimingSummary(comm,
                                        linear_solve_calls_,
                                        last_linear_iters_,
                                        last_linear_solve_seconds_,
                                        last_precond_name_);
  }

  if (direct_identity_system_)
  {
    X_.SyncAliasMemory(sigma_);
  }
  else
  {
    a_->RecoverFEMSolution(X_, *b_, sigma_);
  }
}

} // namespace dg_igr
