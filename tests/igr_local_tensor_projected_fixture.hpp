// Included only by the native CUDA fixture, inside its anonymous namespace.
// These host oracles have no production access to factors or projected data.

constexpr double mapped_amplitude = 0.035;
constexpr double fixture_pi = 3.141592653589793238462643383279502884;

void SmoothMappedCoordinates(const Vector& x, Vector& y)
{
  y.SetSize(2);
  const bool boundary = x[0] == 0.0 || x[0] == 1.0 || x[1] == 0.0 || x[1] == 1.0;
  const double shift = boundary ? 0.0 : mapped_amplitude *
    std::sin(2.0 * fixture_pi * x[0]) * std::sin(2.0 * fixture_pi * x[1]);
  y[0] = x[0] + shift; y[1] = x[1] + shift;
}

Mesh MappedMesh(int nx, int ny, const std::string& geometry, bool reorder)
{
  // Construct every final element before interpolating the Q2 geometry. A
  // coarse warped mesh followed by refinement would define a different map.
  Mesh serial = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                     true, 1.0, 1.0, false);
  if (reorder)
  {
    int multiplier = 5;
    while (std::gcd(multiplier, serial.GetNE()) != 1) { ++multiplier; }
    Array<int> permutation(serial.GetNE());
    for (int e = 0; e < permutation.Size(); ++e)
    { permutation[e] = (multiplier * e + 3) % permutation.Size(); }
    serial.ReorderElements(permutation, false);
  }
  if (geometry == "curved-q2")
  {
    serial.SetCurvature(2, false, 2, Ordering::byVDIM);
    serial.Transform(SmoothMappedCoordinates);
  }
  else if (geometry == "shear")
  {
    serial.Transform([](const Vector& x, Vector& y)
    { y.SetSize(2); y[0] = 1.13 * x[0] + 0.23 * x[1]; y[1] = 0.87 * x[1]; });
  }
  else if (geometry == "rotated-stretch" || geometry == "orthogonal-stretch")
  {
    const double angle = geometry == "rotated-stretch" ? 0.37 : 0.0;
    serial.Transform([angle](const Vector& x, Vector& y)
    {
      y.SetSize(2);
      y[0] = 1.13 * std::cos(angle) * x[0] - 0.87 * std::sin(angle) * x[1];
      y[1] = 1.13 * std::sin(angle) * x[0] + 0.87 * std::cos(angle) * x[1];
    });
  }
  else { MFEM_VERIFY(geometry == "cartesian", "Unknown mapped fixture geometry."); }
  return serial;
}

struct SampledJacobian
{
  double minimum = std::numeric_limits<double>::infinity(), maximum = 0.0;
  std::size_t samples = 0;
};

SampledJacobian SampleMappedJacobian(Mesh& mesh)
{
  SampledJacobian result;
  const auto& corners = *mfem::Geometries.GetVertices(Geometry::SQUARE);
  const auto& rule = IntRules.Get(Geometry::SQUARE, 13);
  for (int e = 0; e < mesh.GetNE(); ++e)
  {
    auto* tr = mesh.GetElementTransformation(e);
    auto sample = [&](const IntegrationRule& points)
    {
      for (int q = 0; q < points.GetNPoints(); ++q)
      {
        tr->SetIntPoint(&points.IntPoint(q));
        const auto& j = tr->Jacobian();
        const double determinant = j(0, 0) * j(1, 1) - j(0, 1) * j(1, 0);
        MFEM_VERIFY(std::isfinite(determinant) && determinant > 0.0,
                    "Mapped fixture has a nonpositive sampled signed detJ at element " << e);
        result.minimum = std::min(result.minimum, determinant);
        result.maximum = std::max(result.maximum, determinant); ++result.samples;
      }
    };
    sample(corners); sample(rule);
    if (mesh.GetNodalFESpace()) { sample(mesh.GetNodalFESpace()->GetFE(e)->GetNodes()); }
  }
  return result;
}

void WriteMappedMesh(const Options& options)
{
  Mesh serial = MappedMesh(options.mesh_n, options.mesh_n, "curved-q2", false);
  const auto jacobian = SampleMappedJacobian(serial);
  std::ofstream output(options.write_mapped_mesh);
  MFEM_VERIFY(output, "Could not open mapped mesh output: " << options.write_mapped_mesh);
  output << std::setprecision(17); serial.Print(output); output.close();
  MFEM_VERIFY(output.good(), "Failed to write the complete mapped mesh.");
  std::cout << std::setprecision(17)
            << "local_tensor_mapped_mesh path=" << options.write_mapped_mesh
            << " mesh_n=" << options.mesh_n << " elements=" << serial.GetNE()
            << " geometry_order=2 final_mesh_before_mapping=true amplitude=" << mapped_amplitude
            << " analytic_detj_lower=" << 1.0 - 2.0 * fixture_pi * mapped_amplitude
            << " sampled_detj_min=" << jacobian.minimum
            << " sampled_detj_max=" << jacobian.maximum
            << " sampled_detj_scaled_min=" << jacobian.minimum * options.mesh_n * options.mesh_n
            << " jacobian_samples=" << jacobian.samples
            << " sample_sets=vertices,GL7,geometry-nodes boundary_fixed=true center_fixed=true"
            << " application_refinements=0 written\n";
}

struct MappedTensorFixture
{
  Mesh serial;
  ParMesh mesh;
  DG_FECollection collection;
  ParFiniteElementSpace space;
  const IntegrationRule &mass_rule, &diffusion_rule, &face_rule;
  QuadratureSpace mass_space;
  QuadratureFunction invrho, initial_invrho;
  int phase = 0;
  double alpha;
  FunctionCoefficient mass, diffusion;
  BilinearForm volume;
  FaceQuadratureSpace face_space;
  InteriorFaceDGDiffusionApplyScratch work;
  InteriorFaceDGDiffusionApplyInfo info;
  Vector side;
  const double kappa = 144.0;

  MappedTensorFixture(int nx, int ny, const std::string& geometry, double alpha_, bool reorder)
    : serial(MappedMesh(nx, ny, geometry, reorder)), mesh(MPI_COMM_WORLD, serial),
      collection(5, 2, BasisType::GaussLegendre), space(&mesh, &collection),
      mass_rule(MassIntegrator::GetRule(*space.GetFE(0), *space.GetFE(0),
                                       *mesh.GetTypicalElementTransformation())),
      diffusion_rule(DiffusionIntegrator::GetRule(*space.GetFE(0), *space.GetFE(0))),
      face_rule(IntRules.Get(Geometry::SEGMENT, diffusion_rule.GetOrder())),
      mass_space(mesh, mass_rule), invrho(mass_space), initial_invrho(mass_space), alpha(alpha_),
      mass([this](const Vector& x) { return 1.0 / Density(x); }),
      diffusion([this](const Vector& x) { return alpha / Density(x); }),
      volume(&space), face_space(mesh, face_rule, FaceType::Interior)
  {
    FillMass(mesh, invrho, mass); FillMass(mesh, initial_invrho, mass);
    AddDomain(volume, mass, diffusion, mass_rule, diffusion_rule);
    volume.SetAssemblyLevel(AssemblyLevel::PARTIAL); volume.Assemble();
    work.compact_qdata = true; work.compact_lift = true;
    FillFace(mesh, face_space, diffusion, side);
    info = BuildInteriorFaceDGDiffusionApplyData2D(
      space, face_space, side, kappa, work.shape, work.weighted_normal_dshape,
      work.penalty, work.face_elements, work.element_vdof_map,
      DGDiffusionFaceScaling::Arithmetic, &work.normal_dshape_base,
      &work.penalty_metric, &work.shape_dictionary, &work.normal_factorization);
    LaunchInteriorFaceDGDiffusionCoefficientUpdate2D(
      info, work, side, kappa, DGDiffusionFaceScaling::Arithmetic);
    MFEM_VERIFY(diffusion_rule.GetNPoints() == 36 &&
                  mass_rule.GetNPoints() == (geometry == "curved-q2" ? 49 : 36),
                "Mapped fixture no longer matches the native p5 production quadrature rules.");
  }

  double Density(const Vector& x) const
  {
    if (phase == 0) { return 1.0; }
    if (phase == 1) { return 0.45; }
    return 0.55 + 0.19 * std::sin(5.7 * x[0] + 0.3) +
           0.13 * std::cos(4.9 * x[1] - 0.2) + 0.08 * x[0] * x[1];
  }

  void SetPhase(int next) { phase = next; FillMass(mesh, invrho, mass); }

  std::unique_ptr<::igrlab::igr_internal::ElementBlockJacobi> Prepare()
  {
    auto backend = ::igrlab::igr_internal::MakeTensorElementBlockJacobi(space, initial_invrho);
    DenseTensor& blocks = backend->PrepareVolumeBlocks(volume);
    ::igrlab::igr_internal::AddExactFaceBlocks(info, work, -1.0, blocks);
    return backend;
  }
};

Vector PhysicalMassDiagonal(ParFiniteElementSpace& space, const QuadratureFunction& coefficient)
{
  const auto& fe = *space.GetFE(0);
  const auto& lex = dynamic_cast<const mfem::TensorBasisElement&>(fe).GetDofMap();
  const auto* qs = dynamic_cast<const QuadratureSpace*>(coefficient.GetSpace());
  MFEM_VERIFY(qs, "Projected oracle requires a volume quadrature space.");
  const int dof = fe.GetDof();
  Vector result(space.GetNE() * dof), shape(dof);
  const double* values = coefficient.HostRead(); double* data = result.HostWrite();
  std::fill(data, data + result.Size(), 0.0);
  for (int e = 0; e < space.GetNE(); ++e)
  {
    auto* tr = space.GetMesh()->GetElementTransformation(e);
    const auto& rule = qs->GetIntRule(e);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const auto& ip = rule.IntPoint(q); tr->SetIntPoint(&ip); fe.CalcShape(ip, shape);
      const double weight = ip.weight * tr->Weight() * values[qs->Offset(e) + q];
      MFEM_VERIFY(weight > 0.0 && std::isfinite(weight), "Nonpositive physical mass sample.");
      for (int i = 0; i < dof; ++i)
      { const double phi = shape[lex.Size() ? lex[i] : i]; data[e * dof + i] += phi * phi * weight; }
    }
  }
  return result;
}

int PhysicalBoundaryMask(ParMesh& mesh, int e)
{
  Array<int> edges, orientation;
  mesh.GetElementEdges(e, edges, orientation);
  MFEM_VERIFY(edges.Size() == 4, "Projected oracle requires quadrilateral edges.");
  int mask = 0;
  for (int edge = 0; edge < edges.Size(); ++edge)
  {
    int first = -1, second = -1; mesh.GetFaceElements(edges[edge], &first, &second);
    if ((first >= 0) != (second >= 0)) { mask |= 1 << edge; }
  }
  return mask;
}

// Test-only cyclic Jacobi sweeps differ from the production maximum-pivot
// eigensolver. Sorting and residual/orthogonality checks make the small basis
// reconstruction independent of factor packing and eigenvector signs.
void OracleSymmetricEigen(const DenseMatrix& input, Vector& values, DenseMatrix& vectors)
{
  const int n = input.Height();
  DenseMatrix a(input); vectors.SetSize(n); vectors = 0.0;
  for (int i = 0; i < n; ++i) { vectors(i, i) = 1.0; }
  double norm = 0.0;
  for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
  { norm = std::max(norm, std::abs(a(i, j))); }
  bool converged = false;
  for (int sweep = 0; sweep < 100 && !converged; ++sweep)
  {
    double maximum_off = 0.0;
    for (int q = 1; q < n; ++q) for (int p = 0; p < q; ++p)
    {
      const double off = a(p, q); maximum_off = std::max(maximum_off, std::abs(off));
      if (std::abs(off) <= 2.0e-16 * norm) { continue; }
      const double theta = 0.5 * std::atan2(2.0 * off, a(q, q) - a(p, p));
      const double c = std::cos(theta), s = std::sin(theta);
      const double app = a(p, p), aqq = a(q, q);
      a(p, p) = c * c * app - 2.0 * c * s * off + s * s * aqq;
      a(q, q) = s * s * app + 2.0 * c * s * off + c * c * aqq;
      a(p, q) = a(q, p) = 0.0;
      for (int k = 0; k < n; ++k)
      {
        if (k != p && k != q)
        {
          const double kp = a(k, p), kq = a(k, q);
          a(k, p) = a(p, k) = c * kp - s * kq;
          a(k, q) = a(q, k) = s * kp + c * kq;
        }
        const double vp = vectors(k, p), vq = vectors(k, q);
        vectors(k, p) = c * vp - s * vq; vectors(k, q) = s * vp + c * vq;
      }
    }
    converged = maximum_off <= 8.0e-16 * norm;
  }
  MFEM_VERIFY(converged, "Independent cyclic Jacobi eigensolver did not converge.");
  std::vector<int> order(n); std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int i, int j) { return a(i, i) < a(j, j); });
  const DenseMatrix unsorted(vectors); values.SetSize(n);
  for (int j = 0; j < n; ++j)
  {
    values[j] = a(order[j], order[j]);
    for (int i = 0; i < n; ++i) { vectors(i, j) = unsorted(i, order[j]); }
  }
  double error = 0.0, orthogonal = 0.0;
  for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
  {
    double aq = 0.0, qtq = 0.0;
    for (int k = 0; k < n; ++k)
    { aq += input(i, k) * vectors(k, j); qtq += vectors(k, i) * vectors(k, j); }
    error = std::max(error, std::abs(aq - vectors(i, j) * values[j]));
    orthogonal = std::max(orthogonal, std::abs(qtq - double(i == j)));
  }
  MFEM_VERIFY(error <= 1.0e-11 * norm && orthogonal < 1.0e-11,
              "Independent eigensolver residual/orthogonality failure.");
}

struct ProjectedQuality
{
  double eigen_min = std::numeric_limits<double>::infinity(), eigen_max = 0.0;
  double maximum_offdiagonal_relative = 0.0;
};

class ProjectedDenseReference
{
public:
  ProjectedDenseReference(MappedTensorFixture& fixture, const Vector& initial_blocks)
    : fixture_(fixture), dof_(36), masks_(fixture.space.GetNE()),
      initial_mass_(PhysicalMassDiagonal(fixture.space, fixture.initial_invrho)),
      h_(fixture.space.GetNE()), inverse_(fixture.space.GetNE()), diagonal_(fixture.space.GetNE())
  {
    const auto& lex = dynamic_cast<const mfem::TensorBasisElement&>(*fixture.space.GetFE(0)).GetDofMap();
    native_.resize(dof_);
    for (int i = 0; i < dof_; ++i) { native_[i] = lex.Size() ? lex[i] : i; }
    std::vector<int> count(16, 0);
    std::vector<DenseMatrix> average(16);
    for (auto& matrix : average) { matrix.SetSize(dof_); matrix = 0.0; }
    const double* blocks = initial_blocks.HostRead(); const double* mass = initial_mass_.HostRead();
    for (int e = 0; e < fixture.space.GetNE(); ++e)
    {
      const int mask = masks_[e] = PhysicalBoundaryMask(fixture.mesh, e); ++count[mask];
      h_[e].SetSize(dof_);
      for (int j = 0; j < dof_; ++j) for (int i = 0; i < dof_; ++i)
      {
        const double a = blocks[std::size_t(e) * dof_ * dof_ + native_[i] + dof_ * native_[j]];
        const double at = blocks[std::size_t(e) * dof_ * dof_ + native_[j] + dof_ * native_[i]];
        h_[e](i, j) = 0.5 * (a + at) / std::sqrt(mass[e * dof_ + i] * mass[e * dof_ + j]);
        average[mask](i, j) += h_[e](i, j);
      }
    }
    for (int mask = 0; mask < 16; ++mask)
    {
      if (!count[mask]) { continue; }
      ++classes_; average[mask] *= 1.0 / count[mask];
      DenseMatrix x(6), y(6), qx, qy; Vector eigenvalues;
      x = 0.0; y = 0.0;
      for (int j = 0; j < 6; ++j) for (int i = 0; i < 6; ++i) for (int k = 0; k < 6; ++k)
      { x(i, j) += average[mask](i + 6 * k, j + 6 * k) / 6.0;
        y(i, j) += average[mask](k + 6 * i, k + 6 * j) / 6.0; }
      // The production half-trace scalar gauge has no effect on either basis.
      OracleSymmetricEigen(x, eigenvalues, qx); OracleSymmetricEigen(y, eigenvalues, qy);
      basis_[mask].SetSize(dof_);
      for (int j = 0; j < dof_; ++j) for (int i = 0; i < dof_; ++i)
      { basis_[mask](i, j) = qx(i % 6, j % 6) * qy(i / 6, j / 6); }
    }
    std::vector<int> first(16, -1);
    for (int e = 0; e < fixture.space.GetNE(); ++e)
    {
      const int mask = masks_[e]; const auto& q = basis_[mask];
      diagonal_[e].SetSize(dof_); inverse_[e].SetSize(dof_);
      for (int k = 0; k < dof_; ++k)
      {
        long double value = 0.0L;
        for (int j = 0; j < dof_; ++j) for (int i = 0; i < dof_; ++i)
        { value += static_cast<long double>(q(i, k)) * h_[e](i, j) * q(j, k); }
        diagonal_[e][k] = double(value);
        MFEM_VERIFY(std::isfinite(diagonal_[e][k]) && diagonal_[e][k] > 0.0,
                    "Independent projected diagonal is not positive.");
      }
      for (int j = 0; j < dof_; ++j) for (int i = 0; i < dof_; ++i)
      {
        long double value = 0.0L;
        for (int k = 0; k < dof_; ++k)
        { value += static_cast<long double>(q(i, k)) * q(j, k) / diagonal_[e][k]; }
        inverse_[e](i, j) = double(value);
      }
      if (first[mask] < 0) { first[mask] = e; }
      else for (int k = 0; k < dof_; ++k)
      {
        diagonal_variation_ = std::max(diagonal_variation_,
          std::abs(diagonal_[e][k] - diagonal_[first[mask]][k]) /
          std::max(diagonal_[e][k], diagonal_[first[mask]][k]));
      }
    }
    MFEM_VERIFY(classes_ <= 16 && classes_ > 1, "Projected fixture missed shared boundary classes.");
  }

  int ClassCount() const { return classes_; }
  double DiagonalVariation() const { return diagonal_variation_; }

  OracleBudget Apply(const Vector& rhs, Vector& result, const std::string& mode) const
  {
    const Vector mass = PhysicalMassDiagonal(fixture_.space, fixture_.invrho);
    const double* m = mass.HostRead(); const double* input = rhs.HostRead();
    result.SetSize(rhs.Size()); double* output = result.HostWrite();
    Array<int> dofs; std::vector<double> z(dof_);
    long double bound2 = 0.0L, maximum_bound = 0.0L;
    for (int e = 0; e < fixture_.space.GetNE(); ++e)
    {
      fixture_.space.GetElementDofs(e, dofs);
      double max_scale = 0.0; long double z2 = 0.0L, inverse_f2 = 0.0L;
      for (int i = 0; i < dof_; ++i)
      {
        const double scale = 1.0 / std::sqrt(m[e * dof_ + i]);
        z[i] = scale * input[dofs[native_[i]]]; z2 += static_cast<long double>(z[i]) * z[i];
        max_scale = std::max(max_scale, scale);
      }
      for (int i = 0; i < dof_; ++i)
      {
        long double value = 0.0L;
        for (int j = 0; j < dof_; ++j)
        { value += static_cast<long double>(inverse_[e](i, j)) * z[j];
          inverse_f2 += static_cast<long double>(inverse_[e](i, j)) * inverse_[e](i, j); }
        output[dofs[native_[i]]] = double(value) / std::sqrt(m[e * dof_ + i]);
      }
      const long double bound = max_scale * std::sqrt(inverse_f2 * z2);
      bound2 += bound * bound; maximum_bound = std::max(maximum_bound, bound);
    }
    UnscaledOracleBudgets budgets;
    budgets.factor = {std::sqrt(bound2), maximum_bound};
    return CompactDenseReference::ScaleBudget(budgets, mode);
  }

  ProjectedQuality Quality(const Vector& current_blocks) const
  {
    ProjectedQuality result;
    const Vector mass = PhysicalMassDiagonal(fixture_.space, fixture_.invrho);
    const double* m = mass.HostRead(); const double* blocks = current_blocks.HostRead();
    for (int e = 0; e < fixture_.space.GetNE(); ++e)
    {
      const auto& q = basis_[masks_[e]];
      DenseMatrix h(dof_), hq(dof_), b(dof_), vectors; Vector values;
      for (int j = 0; j < dof_; ++j) for (int i = 0; i < dof_; ++i)
      { h(i, j) = blocks[std::size_t(e) * dof_ * dof_ + native_[i] + dof_ * native_[j]] /
                   std::sqrt(m[e * dof_ + i] * m[e * dof_ + j]); }
      mfem::Mult(h, q, hq); mfem::MultAtB(q, hq, b);
      long double off2 = 0.0L, total2 = 0.0L;
      for (int j = 0; j < dof_; ++j) for (int i = 0; i < dof_; ++i)
      {
        const double symmetric = 0.5 * (b(i, j) + b(j, i));
        total2 += static_cast<long double>(symmetric) * symmetric;
        if (i != j) { off2 += static_cast<long double>(symmetric) * symmetric; }
      }
      result.maximum_offdiagonal_relative = std::max(result.maximum_offdiagonal_relative,
                                                    std::sqrt(double(off2 / total2)));
      for (int j = 0; j < dof_; ++j) for (int i = 0; i <= j; ++i)
      { b(i, j) = b(j, i) = 0.5 * (b(i, j) + b(j, i)) /
                               std::sqrt(diagonal_[e][i] * diagonal_[e][j]); }
      OracleSymmetricEigen(b, values, vectors);
      MFEM_VERIFY(std::isfinite(values[0]) && values[0] > 0.0,
                  "Independent projected/current generalized spectrum is not positive.");
      result.eigen_min = std::min(result.eigen_min, values[0]);
      result.eigen_max = std::max(result.eigen_max, values[dof_ - 1]);
    }
    return result;
  }

private:
  MappedTensorFixture& fixture_;
  const int dof_;
  int classes_ = 0;
  double diagonal_variation_ = 0.0;
  std::vector<int> masks_, native_;
  Vector initial_mass_;
  std::vector<DenseMatrix> h_, inverse_;
  std::vector<Vector> diagonal_;
  DenseMatrix basis_[16];
};

void CheckProjectedMassDiagonal(MappedTensorFixture& fixture)
{
  const Vector oracle = PhysicalMassDiagonal(fixture.space, fixture.invrho);
  BilinearForm mass(&fixture.space);
  auto* integrator = new MassIntegrator(fixture.mass); integrator->SetIntRule(&fixture.mass_rule);
  mass.AddDomainIntegrator(integrator); mass.Assemble(); mass.Finalize();
  Vector assembled; mass.SpMat().GetDiag(assembled);
  Vector expected(assembled.Size()); Array<int> dofs;
  const auto& lex = dynamic_cast<const mfem::TensorBasisElement&>(*fixture.space.GetFE(0)).GetDofMap();
  for (int e = 0; e < fixture.space.GetNE(); ++e)
  {
    fixture.space.GetElementDofs(e, dofs);
    for (int i = 0; i < 36; ++i) { expected[dofs[lex.Size() ? lex[i] : i]] = oracle[e * 36 + i]; }
  }
  MFEM_VERIFY(RelativeDifference(assembled, expected) < 1.0e-12,
              "Integrated phi_i^2 physical mass differs from assembled MFEM mass diagonal.");
}

void QualifyProjected(const std::vector<std::string>& modes)
{
  ReferenceSelection reference_policy("projected");
  for (const std::string geometry : {"cartesian", "rotated-stretch", "shear", "curved-q2"})
  {
    MappedTensorFixture fixture(5, 4, geometry, 0.017, true);
    const Vector initial = AssembledDiagonal(fixture.space, fixture.mass, fixture.diffusion,
      fixture.mass_rule, fixture.diffusion_rule, fixture.face_rule, fixture.kappa);
    ProjectedDenseReference reference(fixture, initial);
    if (geometry == "rotated-stretch")
    {
      MappedTensorFixture stretched(5, 4, "orthogonal-stretch", 0.017, true);
      const Vector unrotated = AssembledDiagonal(stretched.space, stretched.mass, stretched.diffusion,
        stretched.mass_rule, stretched.diffusion_rule, stretched.face_rule, stretched.kappa);
      MFEM_VERIFY(RelativeDifference(initial, unrotated) < 5.0e-11,
                  "Rigid rotation changed scalar physical blocks for compatible constant coefficients.");
    }
    if (geometry == "curved-q2")
    { MFEM_VERIFY(reference.DiagonalVariation() > 1.0e-7,
                  "Curved fixture does not distinguish element diagonals within a shared class."); }
    const auto jacobian = SampleMappedJacobian(fixture.mesh);
    const int n = fixture.space.GetVSize();
    Vector rhs(n), second(n), actual(n), action2(n), expected(n), expected2(n), first_action(n);
    FillSignedResidual(rhs, 0); FillSignedResidual(second, 0, true); first_action = 0.0;
    std::vector<Vector> physical(3), exact(3);
    ProjectedQuality quality[3];
    double noncollocated_mass_difference = 0.0;
    for (int phase = 0; phase < 3; ++phase)
    {
      fixture.SetPhase(phase); CheckProjectedMassDiagonal(fixture);
      physical[phase] = AssembledDiagonal(fixture.space, fixture.mass, fixture.diffusion,
        fixture.mass_rule, fixture.diffusion_rule, fixture.face_rule, fixture.kappa);
      exact[phase] = SolveBlocks(fixture.space, physical[phase], rhs);
      quality[phase] = reference.Quality(physical[phase]);
      if (geometry == "curved-q2" && phase == 2)
      {
        QuadratureSpace collocated_space(fixture.mesh, IntRules.Get(Geometry::SQUARE, 11));
        QuadratureFunction collocated(collocated_space); FillMass(fixture.mesh, collocated, fixture.mass);
        noncollocated_mass_difference = RelativeDifference(
          PhysicalMassDiagonal(fixture.space, collocated), PhysicalMassDiagonal(fixture.space, fixture.invrho));
        MFEM_VERIFY(noncollocated_mass_difference > 1.0e-8,
                    "Q2 heterogeneous fixture does not distinguish GL7 mass from nodal GL6 scaling.");
      }
    }
    for (const auto& mode : modes)
    {
      ExecutionSelection select(mode); fixture.SetPhase(0);
      auto backend = ::igrlab::igr_internal::MakeTensorElementBlockJacobi(fixture.space, fixture.initial_invrho);
      DenseTensor& built = backend->PrepareVolumeBlocks(fixture.volume);
      ::igrlab::igr_internal::AddExactFaceBlocks(fixture.info, fixture.work, -1.0, built);
      const Vector production(const_cast<double*>(built.HostRead()), built.TotalSize());
      const double assembly_error = RelativeDifference(production, initial);
      MFEM_VERIFY(assembly_error < 5.0e-11,
                  "Projected preparation changed the exact physical MFEM volume/face block.");
      backend->FactorBlocks(); MFEM_VERIFY(built.TotalSize() == 0, "Projected setup retained dense blocks.");
      for (const int phase : {0, 1, 2, 0})
      {
        fixture.SetPhase(phase);
        backend->RefreshMassScaling(fixture.invrho);
        backend->Mult(rhs, actual); backend->Mult(second, action2);
        if (phase == 2) { CheckFusedSolutionUpdate(*backend, rhs, mode == "wmma-factor-fp64"); }
        const auto budget = reference.Apply(rhs, expected, mode);
        const auto budget2 = reference.Apply(second, expected2, mode);
        const auto error = MeasureError(actual, expected), error2 = MeasureError(action2, expected2);
        CheckCorrection(error, budget, mode); CheckCorrection(error2, budget2, mode);
        const long double quadratic = Dot(rhs, actual), quadratic2 = Dot(second, action2);
        MFEM_VERIFY(quadratic > 0.0L && quadratic2 > 0.0L, "Projected action lost positive quadratic form.");
        const long double scale = std::sqrt(quadratic) * std::sqrt(quadratic2);
        const double symmetry = double(std::abs(Dot(rhs, action2) - Dot(second, actual)) / scale);
        const double symmetry_limit = IsFP64(mode) ? 5.0e-10 :
          double((budget.absolute_l2 * std::sqrt(Dot(second, second)) +
                  budget2.absolute_l2 * std::sqrt(Dot(rhs, rhs))) / scale) + 5.0e-10;
        MFEM_VERIFY(symmetry <= symmetry_limit, "Projected action exceeded arithmetic symmetry envelope.");
        const double physical_error = RelativeDifference(actual, exact[phase]);
        if ((geometry == "cartesian" || geometry == "rotated-stretch") && phase < 2)
        { CheckCorrection(MeasureError(actual, exact[phase]), budget, mode); }
        if (phase == 0)
        {
          if (first_action.Size() == n && first_action.Norml2() > 0.0)
          { MFEM_VERIFY(RelativeDifference(actual, first_action) < (IsFP64(mode) ? 5.0e-10 : 1.0e-6),
                        "Returning to initial density did not restore the projected action."); }
          first_action = actual;
        }
        std::cout << std::setprecision(17)
                  << "local_tensor_projected_qualification geometry=" << geometry << " backend=" << mode
                  << " p=5 phase=" << phase << " mass_q1d=" << int(std::sqrt(fixture.mass_rule.GetNPoints()))
                  << " diffusion_q1d=6 boundary_classes=" << reference.ClassCount()
                  << " shared_class_diagonal_variation=" << reference.DiagonalVariation()
                  << " gl6_vs_actual_mass_relative=" << noncollocated_mass_difference
                  << " physical_block_relative=" << assembly_error
                  << " surrogate_relative_l2=" << error.relative_l2
                  << " surrogate_relative_max=" << error.relative_max
                  << " roundoff_envelope_relative_l2=" << double(budget.absolute_l2 / error.norm_l2)
                  << " symmetry_relative=" << symmetry << " positive_quadratic=true"
                  << " exact_current_inverse_relative=" << physical_error
                  << " generalized_eigen_min=" << quality[phase].eigen_min
                  << " generalized_eigen_max=" << quality[phase].eigen_max
                  << " generalized_condition=" << quality[phase].eigen_max / quality[phase].eigen_min
                  << " normalized_offdiagonal_relative=" << quality[phase].maximum_offdiagonal_relative
                  << " sampled_detj_min=" << jacobian.minimum
                  << " oracle=independent-mfem-actual-mass-shared-basis-element-diagonal"
                  << " geometry_accuracy_threshold=none dense_after_setup=0 passed\n";
      }
      first_action = 0.0;
    }
  }
  std::cout << "local_tensor_projected_test passed modes=" << modes.size()
            << " initial_physical_blocks_unchanged=true surrogate_spd=true"
            << " gl7_mass_diagonal=true class_basis_shared=true per_element_diagonal=true\n";
}

void BenchmarkProjected(const Options& options, const std::vector<std::string>& modes)
{
  ReferenceSelection reference_policy("projected");
  // Qualify arithmetic against the independent dense oracle before spending
  // time on a large mesh. Profiling remains a single selected backend.
  if (!options.profile) { QualifyProjected(modes); }
  constexpr double alpha = 6.357828776041666e-7;
  MappedTensorFixture fixture(options.mesh_n, options.mesh_n, "curved-q2", alpha, false);
  const auto jacobian = SampleMappedJacobian(fixture.mesh);
  fixture.SetPhase(2);
  const int n = fixture.space.GetVSize();
  Vector rhs(n), result(n), baseline;
  FillSignedResidual(rhs, 0); rhs.UseDevice(true); result.UseDevice(true);
  (void)rhs.Read(); (void)result.Write(); (void)fixture.invrho.Read();
  MultTimer timer(options.repetitions);
  std::vector<std::vector<double>> apply_medians(modes.size()), refresh_medians(modes.size());
  std::vector<ErrorMetrics> errors(modes.size());
  std::cout << std::setprecision(17)
            << "local_tensor_projected_benchmark event=configuration"
            << " source=" << Provenance("IGRLAB_BENCHMARK_SOURCE")
            << " build=" << Provenance("IGRLAB_BENCHMARK_BUILD")
            << " mesh_n=" << options.mesh_n << " elements=" << fixture.space.GetNE()
            << " p=5 sigma_dofs=" << n << " geometry_order=2 amplitude=" << mapped_amplitude
            << " mass_q1d=7 diffusion_q1d=6 face_q1d=6 alpha=" << alpha << " kappa=" << fixture.kappa
            << " sampled_detj_min=" << jacobian.minimum
            << " sampled_detj_scaled_min=" << jacobian.minimum * options.mesh_n * options.mesh_n
            << " jacobian_samples=" << jacobian.samples
            << " repetitions=" << options.repetitions << " warmup=" << options.warmup
            << " rounds=" << options.rounds << " modes=" << modes.size()
            << " scope=complete-production-Mult gather_scaling_scatter_included=true"
            << " coefficient_refresh_scope=separate-CUDA-event setup_timed_separately=true"
            << " setup_includes_initial_refresh=true"
            << " residual=signed-nonconstant current_density=positive-heterogeneous"
            << " profile=" << (options.profile ? "true" : "false") << '\n';
  // One reference action is stored, with only one backend alive at a time.
  if (!options.profile)
  {
    ExecutionSelection select("simt-fp64"); fixture.phase = 0;
    auto backend = fixture.Prepare(); backend->FactorBlocks(); fixture.phase = 2;
    backend->RefreshMassScaling(fixture.invrho); backend->Mult(rhs, baseline);
    MFEM_GPU_CHECK(cudaDeviceSynchronize());
  }
  for (int round = 0; round < options.rounds; ++round)
  {
    std::vector<int> order(modes.size()); std::iota(order.begin(), order.end(), 0);
    std::rotate(order.begin(), order.begin() + round % order.size(), order.end());
    if (round % 2) { std::reverse(order.begin(), order.end()); }
    for (int position = 0; position < int(order.size()); ++position)
    {
      const int index = order[position]; const auto& mode = modes[index];
      ExecutionSelection select(mode); fixture.phase = 0;
      const auto begin = std::chrono::steady_clock::now();
      auto backend = fixture.Prepare(); backend->FactorBlocks(); fixture.phase = 2;
      backend->RefreshMassScaling(fixture.invrho);
      MFEM_GPU_CHECK(cudaDeviceSynchronize());
      const double setup_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      const auto refresh = Summarize(timer.MeasureRefresh(*backend, fixture.invrho, options.warmup));
      const auto apply = Summarize(timer.Measure(*backend, rhs, result, options.warmup));
      apply_medians[index].push_back(apply.median); refresh_medians[index].push_back(refresh.median);
      std::cout << "local_tensor_projected_benchmark event=" << (options.profile ? "profile" : "round")
                << " round=" << round << " order=" << position << " backend=" << mode
                << " apply_ms_min=" << apply.min << " apply_ms_median=" << apply.median
                << " apply_ms_max=" << apply.max
                << " refresh_ms_min=" << refresh.min << " refresh_ms_median=" << refresh.median
                << " refresh_ms_max=" << refresh.max << " setup_seconds=" << setup_seconds;
      if (!options.profile)
      {
        errors[index] = MeasureErrorWith<double>(result, baseline);
        if (IsFP64(mode))
        { MFEM_VERIFY(errors[index].relative_l2 < 5.0e-10 && errors[index].relative_max < 5.0e-10,
                      "Full-size FP64 factor action differs from projected SIMT FP64 baseline."); }
        MFEM_VERIFY(DotDouble(rhs, result) > 0.0, "Full-size projected correction lost positive quadratic action.");
        std::cout << " correction_relative_l2=" << errors[index].relative_l2
                  << " correction_relative_max=" << errors[index].relative_max
                  << " oracle=projected-simt-fp64 small_independent_oracle=passed"
                  << " positive_quadratic=true tf32_full_difference=reported";
      }
      else { std::cout << " oracle=not-run other_backend_Mult_calls=0"; }
      std::cout << " live_backend_instances=1 physical_accuracy_claim=false passed" << std::endl;
    }
  }
  if (options.profile) { return; }
  for (int i = 0; i < int(modes.size()); ++i)
  {
    const auto apply = Summarize(apply_medians[i]), refresh = Summarize(refresh_medians[i]);
    std::cout << "local_tensor_projected_benchmark event=summary backend=" << modes[i]
              << " mesh_n=" << options.mesh_n << " p=5 sigma_dofs=" << n
              << " apply_ms_median=" << apply.median << " refresh_ms_median=" << refresh.median
              << " apply_ms_min=" << apply.min << " apply_ms_max=" << apply.max
              << " refresh_ms_min=" << refresh.min << " refresh_ms_max=" << refresh.max
              << " correction_relative_l2=" << errors[i].relative_l2
              << " correction_relative_max=" << errors[i].relative_max
              << " statistic=per-round-medians whole_application_speed_claim=false\n";
  }
}
