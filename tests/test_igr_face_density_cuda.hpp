#pragma once
#include "../src/physics/igr/IGR_face_density_cuda.hpp"
#include <cmath>
#include <iostream>

inline void TestFaceDensityCUDA(mfem::ParMesh& mesh)
{
  for (int p : {1, 3})
  {
    mfem::L2_FECollection collection(p, 2, mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace fes(&mesh, &collection, 4, mfem::Ordering::byVDIM);
    mfem::ParFiniteElementSpace sfes(&mesh, &collection);
    mfem::ParGridFunction state(&fes); state.UseDevice(true);
    mfem::Array<int> vdofs;
    double* initial = state.HostWrite();
    for (int e = 0; e < fes.GetNE(); ++e)
    {
      fes.GetElementVDofs(e, vdofs);
      const int q = fes.GetFE(e)->GetDof();
      for (int c = 0; c < 4; ++c)
        for (int i = 0; i < q; ++i)
          initial[vdofs[c * q + i]] = c == 0 ? .06 + .013 * e + .002 * i
                                                        : 1.0 + .2 * c + .001 * i;
    }
    mfem::Vector alpha(fes.GetNE()); alpha.UseDevice(true);
    for (int e = 0; e < fes.GetNE(); ++e) alpha(e) = .003 * (1.0 + .09 * e);
    const auto& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * p);
    mfem::FaceQuadratureSpace fqs(mesh, ir, mfem::FaceType::Interior);
    dg_igr::igr_internal::FaceDensityCUDAProjectionCache cache;
    for (bool cached : {false, true})
    for (bool local : {false, true})
      for (int change = 0; change < 2; ++change)
      {
        if (change)
        {
          // Produce a new state and alpha on the device. The tested projection
          // must consume these versions without a host-state synchronization.
          double* current = state.ReadWrite();
          mfem::forall(state.Size(), [=] MFEM_HOST_DEVICE (int i) { current[i] *= 1.23; });
          double* current_alpha = alpha.ReadWrite();
          mfem::forall(alpha.Size(), [=] MFEM_HOST_DEVICE (int i) { current_alpha[i] *= 1.11; });
        }
        constexpr double floor = .1, scalar_alpha = .007;
        mfem::Vector uncached_actual; dg_igr::dg_face::InteriorFaceSideDensityProjectionScratch uncached_scratch;
        auto& actual = cached ? cache.side_q : uncached_actual;
        auto& scratch = cached ? cache.work : uncached_scratch;
        const auto& active_fqs = cached ? cache.Prepare(fes, sfes, ir) : fqs;
        const auto info = dg_igr::igr_internal::ProjectInteriorFaceDensityCoefficientCUDA(
          fes, active_fqs, state, scalar_alpha, local ? &alpha : nullptr, floor, actual, scratch,
          cached ? &cache.info : nullptr);
        // Independent host finite-element evaluation is solely a test oracle.
        // It deliberately runs after the tested CUDA path has consumed U.
        (void)state.HostRead();
        const double* alpha_host = alpha.HostRead();
        const double* measured = actual.HostRead();
        mfem::Vector value(4); int packed = 0;
        double error = 0.0, scale = 0.0;
        for (int f = 0; f < fqs.GetNE(); ++f)
        {
          const int mesh_face = fqs.GetMeshFaceIndex(f);
          if (mesh.GetFaceInformation(mesh_face).IsShared()) continue;
          auto* tr = mesh.GetFaceElementTransformations(mesh_face);
          for (int k = 0; k < info.nq; ++k)
          {
            tr->SetAllIntPoints(&fqs.GetIntRule(f).IntPoint(k));
            for (int side = 0; side < 2; ++side)
            {
              const int e = side ? tr->Elem2No : tr->Elem1No;
              if (side) state.GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), value);
              else state.GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), value);
              const double expected = (local ? alpha_host[e] : scalar_alpha) / std::max(value(0), floor);
              const double observed = measured[side + 2 * (k + info.nq * packed)];
              MFEM_VERIFY(std::isfinite(observed) && std::isfinite(expected),
                          "CUDA face-density oracle encountered a nonfinite coefficient.");
              error = std::max(error, std::abs(observed - expected));
              scale = std::max(scale, std::abs(expected));
            }
          }
          ++packed;
        }
        MFEM_VERIFY(packed == info.nf && error < 1e-12 * scale,
                    "CUDA face-density coefficients differ from host finite-element oracle: " << error / scale);
        std::cout << "face_density_cuda_test p=" << p << " cached=" << cached << " local_alpha=" << local
                  << " changed_device_state=" << change << " relative_error=" << error / scale << '\n';
      }
    MFEM_VERIFY(cache.rebuilds == 1 && cache.hits == 3,
                "Current state or alpha changes incorrectly rebuilt cached geometry.");
    mesh.NodesUpdated();
    cache.Prepare(fes, sfes, ir);
    MFEM_VERIFY(cache.rebuilds == 2, "Node geometry update did not invalidate face cache.");
    // Same order/count and same caller rule object, but changed points/weights.
    mfem::IntegrationRule custom(ir);
    custom.IntPoint(0).weight *= 1.001;
    cache.Prepare(fes, sfes, custom);
    MFEM_VERIFY(cache.rebuilds == 3, "Quadrature weights did not invalidate face cache.");
    custom.IntPoint(0).x += .001;
    cache.Prepare(fes, sfes, custom);
    MFEM_VERIFY(cache.rebuilds == 4, "Quadrature points did not invalidate face cache.");
    const double temporary_x = custom.IntPoint(0).x + .002;
    {
      mfem::IntegrationRule temporary(custom);
      temporary.IntPoint(0).x = temporary_x;
      cache.Prepare(fes, sfes, temporary);
    }
    MFEM_VERIFY(cache.rebuilds == 5 && cache.fqs->GetIntRule(0).IntPoint(0).x == temporary_x,
                "Face cache did not preserve its owned quadrature rule.");
    // Consume the rebuilt pack after the input rule has been destroyed.
    // This checks numerical reuse after invalidation, in addition to counters.
    dg_igr::igr_internal::ProjectInteriorFaceDensityCoefficientCUDA(
      fes, *cache.fqs, state, .007, &alpha, .1, cache.side_q, cache.work, &cache.info);
    const double* rebuilt = cache.side_q.HostRead();
    const double* alpha_host = alpha.HostRead();
    mfem::Vector value(4); int packed = 0;
    for (int f = 0; f < cache.fqs->GetNE(); ++f)
    {
      const int mesh_face = cache.fqs->GetMeshFaceIndex(f);
      if (mesh.GetFaceInformation(mesh_face).IsShared()) continue;
      auto* tr = mesh.GetFaceElementTransformations(mesh_face);
      for (int k = 0; k < cache.info.nq; ++k)
      {
        tr->SetAllIntPoints(&cache.fqs->GetIntRule(f).IntPoint(k));
        for (int side = 0; side < 2; ++side)
        {
          const int e = side ? tr->Elem2No : tr->Elem1No;
          if (side) state.GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), value);
          else state.GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), value);
          const double expected = alpha_host[e] / std::max(value(0), .1);
          const double observed = rebuilt[side + 2 * (k + cache.info.nq * packed)];
          MFEM_VERIFY(std::isfinite(observed) && std::isfinite(expected) &&
                      std::abs(observed - expected) < 1e-12 * std::abs(expected),
                      "Rebuilt CUDA face-density geometry differs from host oracle.");
        }
      }
      ++packed;
    }
    mfem::ParFiniteElementSpace replaced_fes(&mesh, &collection, 4, mfem::Ordering::byVDIM);
    cache.Prepare(replaced_fes, sfes, custom);
    MFEM_VERIFY(cache.rebuilds == 6, "FES replacement did not invalidate face cache.");
    mfem::L2_FECollection changed_collection(p + 1, 2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace changed_fes(&mesh, &changed_collection, 4, mfem::Ordering::byVDIM);
    cache.Prepare(changed_fes, sfes, custom);
    MFEM_VERIFY(cache.rebuilds == 7 && cache.info.dof == (p + 2) * (p + 2),
                "Changed basis/order did not rebuild the face-density map.");
    std::cout << "face_density_cache_invalidation_test p=" << p
              << " rebuilds=" << cache.rebuilds << " hits=" << cache.hits << '\n';
  }
}
