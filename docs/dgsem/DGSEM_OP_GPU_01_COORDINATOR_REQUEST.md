# DGSEM-OP-GPU-01 coordinator integration request

## Disposition

The worker-owned production cache and MFEM-native kernels were independently
accepted and merged before the current `c01c0cbc` composition base.  The
DGSEM-RESTART-01 Gauss-Lobatto cold-restart foundation was also independently
accepted and merged at that base.  The remaining coordinator-owned gate is the
narrow production-`Mult` integration below; this document records design
guidance rather than a patch to paste unchanged.

The strict evidence switch in the requested patch is
`DG_IGR_DGSEM_PRODUCTION_STRICT=1`.  A supported tuple executes the persistent
worker path on both CPU and CUDA.  An unsupported tuple logs its reason before
using the existing host oracle in ordinary mode and aborts before output writes
in strict mode.

## Historical coordinator-owned design patch

```diff
diff --git a/include/numerics/dg/DGHyperbolicConservationLaws.hpp b/include/numerics/dg/DGHyperbolicConservationLaws.hpp
--- a/include/numerics/dg/DGHyperbolicConservationLaws.hpp
+++ b/include/numerics/dg/DGHyperbolicConservationLaws.hpp
@@
   mutable real_t max_char_speed =
     0.0;            ///< updated during Mult() by the form integrator
   mutable Vector z; ///< scratch for the assembled RHS before M⁻¹
+  mutable std::unique_ptr<dgsem::DGSEMOperator> dgsem_production_operator_;
+  mutable Vector dgsem_x_local_workspace_;
+  mutable Vector dgsem_z_local_workspace_;
+  mutable Vector dgsem_y_local_workspace_;
+  mutable bool dgsem_production_dispatch_observed_ = false;
+  mutable bool dgsem_production_dispatch_supported_ = false;
+  mutable bool dgsem_production_dispatch_strict_ = false;
+  mutable bool dgsem_production_dispatch_cuda_ = false;
+  mutable dgsem::DGSEMProductionReason dgsem_production_dispatch_reason_ =
+    dgsem::DGSEMProductionReason::None;
diff --git a/src/numerics/dg/DGHyperbolicConservationLaws.cpp b/src/numerics/dg/DGHyperbolicConservationLaws.cpp
--- a/src/numerics/dg/DGHyperbolicConservationLaws.cpp
+++ b/src/numerics/dg/DGHyperbolicConservationLaws.cpp
@@
   const Vector* x_eval = &x;
-  Vector x_local;
-  Vector z_local;
-  Vector y_local;
+  Vector& x_local = dgsem_x_local_workspace_;
+  Vector& z_local = dgsem_z_local_workspace_;
+  Vector& y_local = dgsem_y_local_workspace_;
   const Operator* prolongation = nullptr;
 #ifdef MFEM_USE_MPI
-  if (Device::Allows(Backend::DEVICE_MASK))
+  if (Device::Allows(Backend::DEVICE_MASK) ||
+      vol_mode_ == VolumeMode::DGSEMFluxDiff)
@@
   if (vol_mode_ == VolumeMode::DGSEMFluxDiff)
   {
     MFEM_VERIFY(dgsem_stab_ == DGSEMStabilization::None,
                 "DGSEM stabilization hooks are not implemented. "
                 "Use --dgsem-stab none.");
-    dgsem::DGSEMOperator dgsem_op(*this);
+    if (!dgsem_production_operator_)
+    {
+      dgsem_production_operator_ =
+        std::make_unique<dgsem::DGSEMOperator>(*this);
+    }
+    dgsem::DGSEMOperator& dgsem_op = *dgsem_production_operator_;
+    const bool strict =
+      ::dg_igr::EnvFlagEnabled("DG_IGR_DGSEM_PRODUCTION_STRICT");
+    const dgsem::DGSEMProductionSelection selection =
+      dgsem_op.BindProductionPolicy(strict);
+    bool dispatch_log_root = true;
+#ifdef MFEM_USE_MPI
+    if (const auto* par_fes =
+          dynamic_cast<const ParFiniteElementSpace*>(&vfes))
+    {
+      int owner_rank = 0;
+      MPI_Comm_rank(par_fes->GetComm(), &owner_rank);
+      dispatch_log_root = owner_rank == 0;
+    }
+#endif
+    const bool dispatch_transition =
+      !dgsem_production_dispatch_observed_ ||
+      dgsem_production_dispatch_supported_ != selection.supported ||
+      dgsem_production_dispatch_strict_ != strict ||
+      dgsem_production_dispatch_cuda_ != selection.cuda ||
+      dgsem_production_dispatch_reason_ != selection.reason;
+    const auto record_dispatch = [&]()
+    {
+      dgsem_production_dispatch_observed_ = true;
+      dgsem_production_dispatch_supported_ = selection.supported;
+      dgsem_production_dispatch_strict_ = strict;
+      dgsem_production_dispatch_cuda_ = selection.cuda;
+      dgsem_production_dispatch_reason_ = selection.reason;
+    };

     const bool timing_enabled = IsDGSEMTimingEnabled_();
     const auto t_mult_start = std::chrono::steady_clock::now();
+    if (selection.supported)
+    {
+      Vector& y_eval = prolongation ? y_local : y;
+      if (y_eval.Size() != vfes.GetVSize())
+      {
+        y_eval.SetSize(vfes.GetVSize());
+      }
+      const dgsem::DGSEMProductionResult result =
+        dgsem_op.AssembleProductionResidual(*x_eval, y_eval, strict);
+      MFEM_VERIFY(result.executed,
+                  "DGSEM production execution rejected before output: "
+                    << dgsem::DGSEMProductionReasonName(result.reason));
+      if (prolongation)
+      {
+        y = 0.0;
+        prolongation->MultTranspose(y_local, y);
+      }
+      max_char_speed = result.max_characteristic_speed;
+      if (timing_enabled)
+      {
+        const double total_seconds = std::chrono::duration<double>(
+          std::chrono::steady_clock::now() - t_mult_start).count();
+        RecordDGSEMTiming_(0.0, 0.0, 0.0, total_seconds);
+      }
+      if (dispatch_transition && dispatch_log_root)
+      {
+        std::cout << "[dgsem-production] selected=mfem-native-packed"
+                  << " strict=" << strict
+                  << " backend=" << (selection.cuda ? "cuda" : "cpu")
+                  << " lvec_mode=" << (prolongation ? "P-x-PT" : "identity")
+                  << " order=" << selection.order
+                  << " dim=" << selection.dimension << '\n';
+      }
+      record_dispatch();
+      return;
+    }
+
+    MFEM_VERIFY(!strict,
+                "DGSEM production strict policy rejected tuple before output: "
+                  << dgsem::DGSEMProductionReasonName(selection.reason));
+    MFEM_VERIFY(dgsem::DGSEMOperatorSupportsDimension(dim),
+                "DGSEM host oracle cannot execute rejected dimension: "
+                  << dgsem::DGSEMProductionReasonName(selection.reason));
+    if (dispatch_transition && dispatch_log_root)
+    {
+      std::cout << "[dgsem-production] selected=host-oracle-fallback"
+                << " strict=false reason="
+                << dgsem::DGSEMProductionReasonName(selection.reason)
+                << " lvec_mode=" << (prolongation ? "P-x-PT" : "identity")
+                << '\n';
+    }
+    record_dispatch();

     const auto t_surface_start = std::chrono::steady_clock::now();
-    const real_t max_face_speed = dgsem_op.AssembleSurface(x, z);
+    const real_t max_face_speed = dgsem_op.AssembleSurface(*x_eval, z_eval);
     const auto t_surface_stop = std::chrono::steady_clock::now();

     const auto t_volume_start = std::chrono::steady_clock::now();
-    const real_t max_vol_speed = dgsem_op.AssembleVolume(x, z);
+    const real_t max_vol_speed = dgsem_op.AssembleVolume(*x_eval, z_eval);
     const auto t_volume_stop = std::chrono::steady_clock::now();

     const auto t_invmass_start = std::chrono::steady_clock::now();
-    dgsem_op.ApplyInverseMass(z, y);
+    if (prolongation)
+    {
+      if (y_local.Size() != z_eval.Size())
+      {
+        y_local.SetSize(z_eval.Size());
+      }
+      dgsem_op.ApplyInverseMass(z_eval, y_local);
+      y = 0.0;
+      prolongation->MultTranspose(y_local, y);
+    }
+    else
+    {
+      dgsem_op.ApplyInverseMass(z_eval, y);
+    }
@@
 void DGHyperbolicConservationLaws::Update()
 {
@@
   face_data_dirty_ = true;
+  if (dgsem_production_operator_)
+  {
+    dgsem_production_operator_->InvalidateProductionCache(
+      dgsem::DGSEMProductionInvalidation::Topology);
+  }
+  dgsem_production_dispatch_observed_ = false;
 }
```

The worker selection and every stage gate use the communicator owned by the
active `ParFiniteElementSpace`; a serial space uses self scope.  Within each
collective-safe phase, rank-divergent policy semantics take deterministic
precedence as `CollectivePolicyMismatch`; otherwise the stable all-rank reason
is the maximum published `DGSEMProductionReason` enum value.  Strictness is a
communicator-wide OR.
The coordinator must not add world-communicator selection, admission, fallback,
or CFL behavior around this API.

The worker rejects boundary states below
`dg_igr::LimiterDefaults::kRhoMin` before face construction.  A focused
behavioral control applies the host `SlipWallState` to a below-floor state and
requires its resulting density to equal that shared threshold exactly.  This
records the dependency on the host wall's currently private floor and makes any
future drift fail the fixture instead of silently changing wall semantics.

The coordinator should add one focused assertion per positive case that calls
`oper.Mult(state_true, mult_true)` and compares it with the already computed
host-oracle true residual.  Run those assertions with strict mode enabled on
serial CPU, two-rank CPU MPI, and one-rank CUDA.  This is the missing proof that
the accepted kernels are reached through production `Mult`, rather than only
through the worker API.

On unchanged warm admission, the worker returns after its scalar/pointer
fingerprint and owning-communicator agreement, before entering the rebuild-only
code that constructs `Vector`, `DenseMatrix`, or `std::vector` scratch.
Counters prove that each repeated production call takes that unchanged-cache
branch and does not enter rebuild preflight.  Separately, the
workspace evidence proves exact stable sizes, capacities, and host/device
storage identities for every persistent packed workspace, including exchange
storage, after the first successful call.  This proves narrowly zero
worker-owned rebuild-scratch construction on unchanged admission plus stable
persistent storage; it is not
allocator-wide MFEM/MPI/backend instrumentation and makes no performance claim.
Any allocator-level proof and any work on the deliberately deterministic
O(rows*faces) face scan remain for separately authorized profiling after the
production composition gate is independently accepted.

## Accepted output/restart foundation

DGSEM-RESTART-01 supplied the separately reviewed by-NODES Gauss-Lobatto
p1--p4 reconstruction and exact cold-restart identity path.  Candidate
`93f364fde9134c521c830935df6b33f3c9610c6d` was accepted, merged, and published
in the exact `c01c0cbc20c3fa19c6a6ba05b364595330cf9aaa` composition base.
Its CPU, two-rank MPI, reordered-piece rejection, sanitizer-bounded no-IGR
loader, and one-rank CUDA restart evidence close the blocker described by the
superseded prose above.

The accepted CUDA restart run predates production composition and therefore
proves the host-route restart path only.  Production-`Mult` CPU/MPI/CUDA,
P/P^T, CFL, strict/fallback, and lifecycle assertions remain required before
profiling or performance work can start.
