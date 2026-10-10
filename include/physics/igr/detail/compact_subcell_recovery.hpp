#pragma once

#include "igrlab/compat.hpp"

#include "mfem.hpp"
#include "physics/euler/eos.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace igrlab
{
namespace igr_internal
{

// Bounded numerical-method bridge: p0 unknowns are independently evolved
// conservative cell means. This separate p2 field supplies current IGR data.
// It is deliberately a host implementation; every rebuild is charged to the
// application wall time. It does not fit a shock profile or evolve hidden DOFs.
class CompactSubcellRecovery
{
public:
  CompactSubcellRecovery(mfem::ParFiniteElementSpace& means_fes,
                         double gamma, double density_floor, double pressure_floor,
                         bool centered_linear = false, bool muscl_transport = false)
    : means_fes_(means_fes), collection_(2, 2, mfem::BasisType::GaussLegendre),
      recovery_fes_(means_fes.GetParMesh(), &collection_, 4, mfem::Ordering::byNODES),
      recovered_(&recovery_fes_), gamma_(gamma), density_floor_(density_floor),
      pressure_floor_(pressure_floor), centered_linear_(centered_linear),
      muscl_transport_(muscl_transport)
  {
    auto& mesh = *means_fes.GetParMesh();
    int ranks = 0; MPI_Comm_size(means_fes.GetComm(), &ranks);
    MFEM_VERIFY(ranks == 1 && mesh.Dimension() == 2 && mesh.SpaceDimension() == 2 &&
                  !mesh.Nonconforming() && means_fes.GetVDim() == 4 && mesh.GetNE() >= 5,
                "Compact subcell recovery requires rank-one 2D conforming quads and four conservative fields.");
    // PA volume/face geometry would lazily install these same coordinate
    // nodes. Do it before validating and recording the fixed mesh identity,
    // so that harmless first-use initialization is not mistaken for motion.
    mesh.EnsureNodes();
    ne_ = mesh.GetNE();
    sorted_.resize(ne_); std::iota(sorted_.begin(), sorted_.end(), 0);
    std::vector<double> left(ne_), right(ne_);
    means_map_.resize(4 * ne_); recovery_map_.resize(36 * ne_);
    double y_min = 0.0, y_max = 0.0;
    mfem::Array<int> vdofs, vertices;
    mfem::Vector physical(2);
    for (int e = 0; e < ne_; ++e)
    {
      MFEM_VERIFY(means_fes.GetFE(e)->GetGeomType() == mfem::Geometry::SQUARE &&
                    means_fes.GetFE(e)->GetOrder() == 0 && means_fes.GetFE(e)->GetDof() == 1,
                  "Compact subcell recovery requires ordinary p0 DG transport.");
      mesh.GetElementVertices(e, vertices);
      double xl = 1e300, xr = -1e300, yl = 1e300, yr = -1e300;
      for (int v : vertices)
      {
        const double* x = mesh.GetVertex(v);
        xl = std::min(xl, x[0]); xr = std::max(xr, x[0]);
        yl = std::min(yl, x[1]); yr = std::max(yr, x[1]);
      }
      MFEM_VERIFY(xr > xl && yr > yl, "Degenerate subcell geometry.");
      if (e == 0) { dx_ = xr - xl; y_min = yl; y_max = yr; }
      const double tolerance = 1e-11 * std::max({1.0, std::abs(xl), std::abs(xr),
                                                std::abs(yl), std::abs(yr)});
      MFEM_VERIFY(std::abs(xr-xl-dx_) < tolerance && std::abs(yl-y_min) < tolerance &&
                    std::abs(yr-y_max) < tolerance,
                  "Compact subcell recovery requires uniform x cells and exactly one cell across y.");
      auto* tr = mesh.GetElementTransformation(e);
      for (double x : {0.0, .5, 1.0}) for (double y : {0.0, .5, 1.0})
      {
        mfem::IntegrationPoint point; point.Set2(x, y); tr->SetIntPoint(&point);
        tr->Transform(point, physical);
        const auto& jacobian = tr->Jacobian();
        MFEM_VERIFY(std::abs(physical(0)-(xl+x*dx_)) < tolerance &&
                      std::abs(physical(1)-(yl+y*(yr-yl))) < tolerance &&
                      std::abs(jacobian(0,0)-dx_) < tolerance &&
                      std::abs(jacobian(1,1)-(yr-yl)) < tolerance &&
                      std::abs(jacobian(0,1)) < tolerance && std::abs(jacobian(1,0)) < tolerance,
                    "Compact subcell recovery requires affine quads aligned with the reference x axis.");
      }
      left[e] = xl; right[e] = xr;
      means_fes.GetElementVDofs(e, vdofs);
      for (int c = 0; c < 4; ++c)
      {
        MFEM_VERIFY(vdofs[c] >= 0, "Signed subcell mean map is unsupported.");
        means_map_[4*e+c] = vdofs[c];
      }
      recovery_fes_.GetElementVDofs(e, vdofs);
      for (int i = 0; i < 36; ++i)
      {
        MFEM_VERIFY(vdofs[i] >= 0, "Signed recovery map is unsupported.");
        recovery_map_[36*e+i] = vdofs[i];
      }
    }
    std::sort(sorted_.begin(), sorted_.end(), [&](int a, int b) { return left[a] < left[b]; });
    for (int i = 1; i < ne_; ++i)
      MFEM_VERIFY(std::abs(left[sorted_[i]]-right[sorted_[i-1]]) < 1e-10*std::max(1.0,dx_),
                  "Subcell row has a gap or overlapping cells.");
    // This first bridge uses nonperiodic zero-gradient recovery ghosts.
    int interior_faces = 0;
    for (int f = 0; f < mesh.GetNumFaces(); ++f)
      if (mesh.GetFaceInformation(f).IsInterior()) ++interior_faces;
    MFEM_VERIFY(interior_faces == ne_-1,
                "Compact subcell recovery requires a nonperiodic connected one-row mesh.");
    mesh_sequence_ = mesh.GetSequence(); nodes_sequence_ = mesh.GetNodesSequence();
    means_sequence_ = means_fes.GetSequence();
    recovered_.UseDevice(true);
    if (muscl_transport_)
    {
      transport_slopes_.SetSize(means_fes.GetVSize());
      transport_slopes_.UseDevice(true);
    }
  }

  ~CompactSubcellRecovery()
  {
    if (muscl_transport_ && transport_calls_)
      mfem::out << "[igr-subcell] event=transport-reconstruction method=muscl-mc"
                << " calls=" << transport_calls_ << " cells=" << transport_cells_
                << " limited_cells=" << transport_limited_cells_
                << " minimum_theta=" << transport_min_theta_
                << " host_seconds=" << transport_seconds_ << '\n';
  }

  mfem::ParFiniteElementSpace& StateSpace() { return recovery_fes_; }
  bool UsesMUSCLTransport() const { return muscl_transport_; }

  // Native p0 layout, with slope measured over reference x in [0,1].
  // This separate broken linear transport field preserves each mean. It must
  // be rebuilt for every RHS, even when current sigma is reused at stage zero.
  const mfem::Vector& RebuildTransportSlopes(const mfem::Vector& means)
  {
    VerifyFixedMesh();
    MFEM_VERIFY(muscl_transport_ && means.Size() == means_fes_.GetVSize(),
                "MUSCL reconstruction requires the current native p0 mean vector.");
    const auto start = std::chrono::steady_clock::now();
    ++transport_calls_;
    const double* input = means.HostRead();
    double* slopes = transport_slopes_.HostWrite();
    const auto at = [&](int i, int c)
    { return input[means_map_[4*sorted_[std::clamp(i,0,ne_-1)]+c]]; };
    for (int i = 0; i < ne_; ++i)
    {
      ++transport_cells_;
      State mean{}, slope{};
      for (int c = 0; c < 4; ++c)
      {
        mean[c] = at(i,c);
        const double a = 2*(at(i,c)-at(i-1,c));
        const double b = .5*(at(i+1,c)-at(i-1,c));
        const double d = 2*(at(i+1,c)-at(i,c));
        slope[c] = (a > 0 && b > 0 && d > 0) ? std::min({a,b,d}) :
                   (a < 0 && b < 0 && d < 0) ? std::max({a,b,d}) : 0.0;
      }
      MFEM_VERIFY(Admissible(mean), "MUSCL reconstruction received an inadmissible mean.");
      const auto endpoints_admissible = [&](double theta)
      {
        State left{}, right{};
        for (int c = 0; c < 4; ++c)
        { left[c] = mean[c]-.5*theta*slope[c]; right[c] = mean[c]+.5*theta*slope[c]; }
        return Admissible(left) && Admissible(right);
      };
      double theta = 1.0;
      if (!endpoints_admissible(theta))
      {
        double lo = 0.0, hi = 1.0;
        for (int iteration = 0; iteration < 52; ++iteration)
        {
          const double mid = .5*(lo+hi);
          if (endpoints_admissible(mid)) lo = mid; else hi = mid;
        }
        theta = .999999999*lo;
        ++transport_limited_cells_;
        transport_min_theta_ = std::min(transport_min_theta_,theta);
      }
      MFEM_VERIFY(endpoints_admissible(theta), "MUSCL endpoint admission failed after slope scaling.");
      // Convexity of the Euler admissible set certifies the whole line segment.
      for (int c = 0; c < 4; ++c) slopes[means_map_[4*sorted_[i]+c]] = theta*slope[c];
    }
    transport_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return transport_slopes_;
  }

  void InitializeMeans(mfem::VectorCoefficient& initial, mfem::ParGridFunction& means)
  {
    MFEM_VERIFY(means.ParFESpace() == &means_fes_ && initial.GetVDim() == 4,
                "Subcell mean initialization received the wrong state space.");
    const auto start = std::chrono::steady_clock::now();
    const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 15);
    double* output = means.HostWrite();
    mfem::Vector value(4), transverse(4);
    int max_panels = 0;
    for (int e = 0; e < ne_; ++e)
    {
      auto* tr = means_fes_.GetElementTransformation(e);
      State previous{}, integral{};
      bool converged = false;
      for (int panels = 1; panels <= 2048; panels *= 2)
      {
        integral.fill(0.0);
        for (int panel = 0; panel < panels; ++panel)
          for (int q = 0; q < rule.GetNPoints(); ++q)
          {
            const auto& point = rule.IntPoint(q);
            mfem::IntegrationPoint ip; ip.Set2((panel+point.x)/panels, .5);
            tr->SetIntPoint(&ip); initial.Eval(value, *tr, ip);
            for (int c = 0; c < 4; ++c) integral[c] += point.weight * value(c) / panels;
            if (panels == 1)
              for (double y : {.25, .75})
              {
                ip.y = y; tr->SetIntPoint(&ip); initial.Eval(transverse, *tr, ip);
                for (int c = 0; c < 4; ++c)
                  MFEM_VERIFY(std::abs(transverse(c)-value(c)) < 1e-12*std::max(1.0,std::abs(value(c))),
                              "Subcell initial data must be independent of y.");
              }
          }
        if (panels > 1)
        {
          double difference = 0.0;
          for (int c = 0; c < 4; ++c)
            difference = std::max(difference, std::abs(integral[c]-previous[c])/
                                                std::max(1.0,std::abs(integral[c])));
          if (difference < 1e-12) { converged = true; max_panels = std::max(max_panels,panels); break; }
        }
        previous = integral;
      }
      MFEM_VERIFY(converged && Admissible(integral),
                  "Subcell integral initialization did not converge to admissible means.");
      for (int c = 0; c < 4; ++c) output[means_map_[4*e+c]] = integral[c];
    }
    mfem::out << "[igr-subcell] event=mean-initialization method=composite-gauss8"
              << " relative_tolerance=1e-12 maximum_panels=" << max_panels
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << '\n';
  }

  const mfem::ParGridFunction& Rebuild(const mfem::ParGridFunction& means)
  {
    MFEM_VERIFY(means.ParFESpace() == &means_fes_, "Compact recovery received a different state space.");
    VerifyFixedMesh();
    const double* input = means.HostRead();
    double* output = recovered_.HostWrite();
    const auto& points = recovery_fes_.GetFE(0)->GetNodes();
    const auto at = [&](int i, int c)
    {
      return input[means_map_[4*sorted_[std::clamp(i,0,ne_-1)]+c]];
    };
    for (int i = 0; i < ne_; ++i)
    {
      State left{}, middle{}, right{};
      for (int c = 0; c < 4; ++c)
      {
        if (centered_linear_)
        {
          const double slope = .5*(at(i+1,c)-at(i-1,c));
          left[c] = at(i,c)-.5*slope; right[c] = at(i,c)+.5*slope;
          middle[c] = at(i,c);
        }
        else
        {
          left[c] = (-at(i-2,c)+7*at(i-1,c)+7*at(i,c)-at(i+1,c))/12;
          right[c] = (-at(i-1,c)+7*at(i,c)+7*at(i+1,c)-at(i+2,c))/12;
          middle[c] = 3*at(i,c)-left[c]-right[c];
        }
      }
      // A quadratic Bernstein curve lies in its control-point convex hull.
      // Subdivision permits valid curves whose initial middle control is not
      // admissible. No clipping/floor or undisclosed limited reconstruction.
      MFEM_VERIFY(AdmissibleQuadratic(left,middle,right,12),
                  "Unlimited conservative subcell recovery is inadmissible; candidate cannot advance this state.");
      const int e = sorted_[i];
      for (int j = 0; j < 9; ++j)
      {
        const double t = points.IntPoint(j).x;
        for (int c = 0; c < 4; ++c)
          output[recovery_map_[36*e+9*c+j]] = (1-t)*(1-t)*left[c] +
                                             2*t*(1-t)*middle[c] + t*t*right[c];
      }
    }
    return recovered_;
  }

private:
  using State = std::array<double,4>;
  void VerifyFixedMesh() const
  {
    const auto& mesh = *means_fes_.GetParMesh();
    MFEM_VERIFY(mesh.GetSequence() == mesh_sequence_ && mesh.GetNodesSequence() == nodes_sequence_ &&
                  means_fes_.GetSequence() == means_sequence_,
                "Subcell reconstruction received changed mesh or state space: topology="
                  << mesh_sequence_ << "->" << mesh.GetSequence() << " nodes=" << nodes_sequence_
                  << "->" << mesh.GetNodesSequence() << " fes=" << means_sequence_ << "->" << means_fes_.GetSequence());
  }
  bool Admissible(const State& state) const
  {
    for (double value : state) if (!std::isfinite(value)) return false;
    if (state[0] <= density_floor_) return false;
    mfem::Vector view(const_cast<double*>(state.data()),4);
    const double pressure = mfem::EulerPressureFromConservative(view,2,gamma_);
    return std::isfinite(pressure) && pressure > pressure_floor_;
  }
  bool AdmissibleQuadratic(const State& a, const State& b, const State& c, int depth) const
  {
    if (!Admissible(a) || !Admissible(c)) return false;
    if (Admissible(b)) return true;
    if (depth == 0) return false;
    State ab{}, bc{}, mid{};
    for (int j = 0; j < 4; ++j)
    {
      ab[j] = .5*(a[j]+b[j]); bc[j] = .5*(b[j]+c[j]); mid[j] = .5*(ab[j]+bc[j]);
    }
    return AdmissibleQuadratic(a,ab,mid,depth-1) && AdmissibleQuadratic(mid,bc,c,depth-1);
  }
  mfem::ParFiniteElementSpace& means_fes_;
  mfem::L2_FECollection collection_;
  mfem::ParFiniteElementSpace recovery_fes_;
  mfem::ParGridFunction recovered_;
  mfem::Vector transport_slopes_;
  double gamma_, density_floor_, pressure_floor_, dx_ = 0.0;
  bool centered_linear_ = false;
  bool muscl_transport_ = false;
  std::uint64_t transport_calls_ = 0, transport_cells_ = 0, transport_limited_cells_ = 0;
  double transport_min_theta_ = 1.0, transport_seconds_ = 0.0;
  int ne_ = 0;
  long mesh_sequence_ = -1, nodes_sequence_ = -1, means_sequence_ = -1;
  std::vector<int> sorted_, means_map_, recovery_map_;
};

} // namespace igr_internal
} // namespace igrlab
