#include "igrlab/compat.hpp"
#include "io/vtk_output.hpp"

#include "utils/config/env_flags.hpp"
#include "vtk_field_cache.hpp"
#include <iostream>
#include <map>
#include <typeinfo>

namespace mfem
{
namespace io
{
namespace
{
class CachedEulerDataCollection final : public ParaViewDataCollection
{
  struct Field
  {
    GridFunction* source;
    detail::VTKFieldEvaluation evaluation;
    std::unique_ptr<detail::CachedVTKScalar> scalar;
    std::unique_ptr<detail::CachedVTKVector> vector;
    Field(GridFunction& field, detail::VTKReferenceShapes& shapes)
      : source(&field), evaluation(field, shapes) { }
  };
public:
  CachedEulerDataCollection(const std::string& name, Mesh* mesh)
    : ParaViewDataCollection(name, mesh) { SetOwnData(false); }
  ~CachedEulerDataCollection() override
  {
    // Coefficients belong to these private unique_ptrs; input GFs/mesh are
    // borrowed. Base maps may retain pointers but must never delete them.
    SetOwnData(false);
  }
  void RegisterField(const std::string& name, GridFunction* field) override
  {
    DeregisterField(name);
    if (!field) { ParaViewDataCollection::RegisterField(name, field); return; }
    auto value = std::make_unique<Field>(*field, shapes_);
    Route(name, *value);
    fields_[name] = std::move(value);
  }
  void DeregisterField(const std::string& name) override
  {
    SetOwnData(false);
    ParaViewDataCollection::DeregisterField(name);
    DeregisterCoeffField(name); DeregisterVCoeffField(name);
    fields_.erase(name); pressure_.erase(name);
  }
  void RegisterPressure(const std::string& name, Coefficient* coefficient)
  {
    DeregisterField(name);
    if (coefficient && typeid(*coefficient) == typeid(PressureCoefficient))
    {
      auto value = std::make_unique<detail::CachedVTKPressure>(
        static_cast<PressureCoefficient&>(*coefficient), shapes_);
      value->Reset();
      RegisterCoeffField(name, value.get());
      pressure_[name] = std::move(value);
    }
    else { RegisterCoeffField(name, coefficient); }
  }
  void Save() override
  {
    SetOwnData(false);
    ResetFrame();
    for (auto& field : fields_) { Route(field.first, *field.second); }
    struct ResetAfterSave
    {
      CachedEulerDataCollection& collection;
      ~ResetAfterSave() { collection.ResetFrame(); }
    } reset{*this};
    ParaViewDataCollection::Save();
  }
private:
  void ResetFrame()
  {
    for (auto& field : fields_) { field.second->evaluation.Reset(); }
    for (auto& pressure : pressure_) { pressure.second->Reset(); }
    shapes_.Reset();
  }
  void Route(const std::string& name, Field& field)
  {
    ParaViewDataCollection::DeregisterField(name);
    DeregisterCoeffField(name); DeregisterVCoeffField(name);
    if (bdr_output || !detail::CanCacheVTKField(*field.source, GetMesh()))
    { ParaViewDataCollection::RegisterField(name, field.source); return; }
    const int dimension = field.source->VectorDim();
    if (dimension == 1)
    {
      if (!field.scalar) { field.scalar = std::make_unique<detail::CachedVTKScalar>(field.evaluation); }
      RegisterCoeffField(name, field.scalar.get());
    }
    else
    {
      if (!field.vector || field.vector->GetVDim() != dimension)
      { field.vector = std::make_unique<detail::CachedVTKVector>(dimension, field.evaluation); }
      RegisterVCoeffField(name, field.vector.get());
    }
  }
  detail::VTKReferenceShapes shapes_;
  std::map<std::string, std::unique_ptr<Field>> fields_;
  std::map<std::string, std::unique_ptr<detail::CachedVTKPressure>> pressure_;
};
} // namespace

std::unique_ptr<ParaViewDataCollection> MakeEulerVTKDataCollection(
  const std::string& name, Mesh* mesh)
{
  const bool cached = ::igrlab::EnvFlagEnabledDefault("DG_IGR_VTK_CACHED_EVALUATION", false);
  if (::igrlab::GetEnv("DG_IGR_VTK_CACHED_EVALUATION"))
  {
    std::cout << "[vtk-writer] event=setup requested=" << (cached ? "cached" : "native")
              << " selected=" << (cached ? "cached" : "native")
              << " interpolation=gridfunction native_mesh_writer=1 precision=unchanged" << std::endl;
  }
  if (cached) { return std::make_unique<CachedEulerDataCollection>(name, mesh); }
  return std::make_unique<ParaViewDataCollection>(name, mesh);
}

void ApplyVTKConfig(mfem::ParaViewDataCollection& dc,
                    const VTKOutputConfig& cfg)
{
  dc.SetPrefixPath(cfg.prefix_path);
  if (cfg.levels_of_detail >= 0)
  {
    dc.SetLevelsOfDetail(cfg.levels_of_detail);
  }
  dc.SetDataFormat(cfg.format);
  dc.SetHighOrderOutput(cfg.high_order_output);
}

void RegisterEulerFields(mfem::ParaViewDataCollection& dc,
                         const EulerVTKFields& fields)
{
  if (fields.density)
  {
    dc.RegisterField("density", fields.density);
  }
  if (fields.momentum)
  {
    dc.RegisterField("momentum", fields.momentum);
  }
  if (fields.energy)
  {
    dc.RegisterField("energy", fields.energy);
  }
  if (fields.velocity)
  {
    dc.RegisterField("velocity", fields.velocity);
  }
  if (fields.sigma)
  {
    // MFEM's registration API predates const field handles; ParaView output
    // only reads the registered GridFunction.
    dc.RegisterField("sigma",
                     const_cast<mfem::ParGridFunction*>(fields.sigma));
  }
  if (fields.pressure_coefficient &&
      ::igrlab::EnvFlagEnabledDefault(
        "DG_IGR_VTK_PRESSURE_COEFFICIENT", false))
  {
    if (auto* cached = dynamic_cast<CachedEulerDataCollection*>(&dc))
    { cached->RegisterPressure("pressure", fields.pressure_coefficient); }
    else { dc.RegisterCoeffField("pressure", fields.pressure_coefficient); }
  }
  else if (fields.pressure)
  {
    dc.RegisterField("pressure", fields.pressure);
  }
  if (fields.velocity_h1)
  {
    dc.RegisterField("velocity_h1", fields.velocity_h1);
  }
  if (fields.speed_h1)
  {
    dc.RegisterField("speed_h1", fields.speed_h1);
  }
  if (fields.div_h1)
  {
    dc.RegisterField("div_h1", fields.div_h1);
  }
  if (fields.grad_frob_h1)
  {
    dc.RegisterField("grad_frob_h1", fields.grad_frob_h1);
  }
  if (fields.div_l2)
  {
    dc.RegisterField("div_l2", fields.div_l2);
  }
  if (fields.grad_frob_l2)
  {
    dc.RegisterField("grad_frob_l2", fields.grad_frob_l2);
  }
  if (fields.mpi_rank)
  {
    dc.RegisterField("mpi_rank", fields.mpi_rank);
  }
  if (fields.zhang_shu_active)
  {
    dc.RegisterField("zhang_shu_active", fields.zhang_shu_active);
  }
  if (fields.zhang_shu_theta_rho)
  {
    dc.RegisterField("zhang_shu_theta_rho", fields.zhang_shu_theta_rho);
  }
  if (fields.zhang_shu_theta_p)
  {
    dc.RegisterField("zhang_shu_theta_p", fields.zhang_shu_theta_p);
  }
}

} // namespace io
} // namespace mfem
