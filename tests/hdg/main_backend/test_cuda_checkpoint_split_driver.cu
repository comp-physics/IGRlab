#include "igrlab/compat.hpp"
#include "sim/hdg/main_backend.hpp"

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

bool inject_after_two_commits = false;

struct Arguments
{
  std::string mesh;
  std::string checkpoint;
  std::string restart;
  std::string vtk_root;
  std::string inspect;
  bool inject_after_two = false;
  bool perturb_initial = false;
  bool perturb_source = false;
};

bool ParseArguments(const int argc, char* argv[], Arguments& arguments)
{
  bool have_mesh = false;
  bool have_checkpoint = false;
  bool have_restart = false;
  bool have_vtk_root = false;
  bool have_inspect = false;
  for (int index = 1; index < argc; ++index)
  {
    const std::string option{argv[index]};
    const auto value = [&](std::string& destination, bool& present)
    {
      if (present || index + 1 >= argc) return false;
      destination = argv[++index];
      present = !destination.empty();
      return present;
    };
    if (option == "--mesh")
    {
      if (!value(arguments.mesh, have_mesh)) return false;
    }
    else if (option == "--checkpoint")
    {
      if (!value(arguments.checkpoint, have_checkpoint)) return false;
    }
    else if (option == "--restart")
    {
      if (!value(arguments.restart, have_restart)) return false;
    }
    else if (option == "--vtk-root")
    {
      if (!value(arguments.vtk_root, have_vtk_root)) return false;
    }
    else if (option == "--inspect-checkpoint")
    {
      if (!value(arguments.inspect, have_inspect)) return false;
    }
    else if (option == "--inject-after-two" &&
             !arguments.inject_after_two)
    {
      arguments.inject_after_two = true;
    }
    else if (option == "--perturb-initial" &&
             !arguments.perturb_initial)
    {
      arguments.perturb_initial = true;
    }
    else if (option == "--perturb-source" &&
             !arguments.perturb_source)
    {
      arguments.perturb_source = true;
    }
    else
    {
      return false;
    }
  }
  if (have_inspect)
  {
    return !have_mesh && !have_checkpoint && !have_restart &&
        !have_vtk_root && !arguments.inject_after_two &&
        !arguments.perturb_initial && !arguments.perturb_source;
  }
  return have_mesh && have_checkpoint && have_vtk_root;
}

class SmoothEulerCoefficient final : public mfem::VectorCoefficient
{
public:
  explicit SmoothEulerCoefficient(const bool perturbed)
      : mfem::VectorCoefficient(4), perturbed_(perturbed) {}

  void Eval(mfem::Vector& state,
            mfem::ElementTransformation& transformation,
            const mfem::IntegrationPoint& integration_point) override
  {
    mfem::Vector point;
    transformation.Transform(integration_point, point);
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double rho = 1.0 + (perturbed_ ? 1.0e-4 : 0.0) + 0.01 *
        std::sin(2.0 * pi * point[0]) * std::sin(2.0 * pi * point[1]);
    constexpr double velocity_x = 0.02;
    constexpr double velocity_y = -0.01;
    constexpr double pressure = 1.0;
    state.SetSize(4);
    state[0] = rho;
    state[1] = rho * velocity_x;
    state[2] = rho * velocity_y;
    state[3] = pressure / 0.4 + 0.5 * rho *
        (velocity_x * velocity_x + velocity_y * velocity_y);
  }

private:
  bool perturbed_ = false;
};

class SmoothSourceCoefficient final : public mfem::VectorCoefficient
{
public:
  explicit SmoothSourceCoefficient(const bool perturbed)
      : mfem::VectorCoefficient(4), scale_(perturbed ? 1.01 : 1.0) {}

  void Eval(mfem::Vector& source,
            mfem::ElementTransformation& transformation,
            const mfem::IntegrationPoint& integration_point) override
  {
    mfem::Vector point;
    transformation.Transform(integration_point, point);
    const double modulation = 1.0 + 0.1 * point[0] - 0.05 * point[1];
    source.SetSize(4);
    source[0] = scale_ * 1.0e-6 * modulation;
    source[1] = scale_ * 2.0e-7 * modulation;
    source[2] = scale_ * -1.0e-7 * modulation;
    source[3] = scale_ * 2.5e-6 * modulation;
  }

private:
  double scale_ = 1.0;
};

dg_sim::hdg::HdgEulerCaseData MakeCaseData(
    const dg_sim::CaseContext& context, const int order,
    const bool perturb_initial, const bool perturb_source)
{
  dg_sim::hdg::HdgEulerCaseData data;
  data.initial_condition =
      std::make_unique<SmoothEulerCoefficient>(perturb_initial);
  data.static_source =
      std::make_unique<SmoothSourceCoefficient>(perturb_source);
  data.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  const int boundary_attributes = context.Mesh().bdr_attributes.Size() == 0
      ? 0
      : context.Mesh().bdr_attributes.Max();
  if (boundary_attributes > 0)
  {
    mfem::Array<int> periodic_marker(boundary_attributes);
    periodic_marker = 1;
    data.boundary_plan.AddPeriodic(std::move(periodic_marker));
  }
  return data;
}

std::string Hex(const dg::hdg::runtime::StateHash& hash)
{
  constexpr char digits[] = "0123456789abcdef";
  std::string result(hash.bytes.size() * 2, '0');
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    result[2 * index] = digits[hash.bytes[index] >> 4U];
    result[2 * index + 1] = digits[hash.bytes[index] & 0x0fU];
  }
  return result;
}

std::uint64_t DoubleBits(const double value) noexcept
{
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value),
                "checkpoint summary requires binary64");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int InspectCheckpoint(const std::filesystem::path& path)
{
  using namespace dg::hdg::runtime::app;
  std::vector<std::uint8_t> bytes;
  std::string io_detail;
  if (detail::ReadCurrentGenerationArtifact(
          path, generic_hdg_cuda_checkpoint_maximum_bytes_v1,
          bytes, io_detail) != DurableIoStatus::ok)
  {
    std::cerr << "checkpoint inspection read failed: " << io_detail << '\n';
    return 1;
  }

  // The fieldwise v1 prefix is magic[16], version[4], size[8], followed by
  // the 32-byte numerical identity. The official decoder validates every
  // other field, checksum, rank tuple, descriptor, and payload hash.
  constexpr std::size_t numerical_identity_offset = 16 + 4 + 8;
  if (bytes.size() < numerical_identity_offset +
          dg::hdg::runtime::StateHash{}.bytes.size())
  {
    std::cerr << "checkpoint inspection prefix is truncated\n";
    return 1;
  }
  dg::hdg::runtime::StateHash identity{};
  std::copy_n(bytes.data() + numerical_identity_offset,
              identity.bytes.size(), identity.bytes.data());
  GenericHdgCudaCheckpointArtifactV1 artifact{};
  const CheckpointCodecStatus status = DecodeGenericHdgCudaCheckpointV1(
      {bytes.data(), bytes.size()}, identity, 1, artifact);
  if (status != CheckpointCodecStatus::ok || artifact.ranks.size() != 1)
  {
    std::cerr << "checkpoint inspection decode failed: "
              << static_cast<int>(status) << '\n';
    return 1;
  }

  const auto& time = artifact.time_state;
  const auto& cursor = time.cursor;
  const auto& rank = artifact.ranks.front();
  constexpr std::size_t legacy_interior_capacity =
      static_cast<std::size_t>(
          dg::hdg::runtime::cuda_w8a::maximum_rank_elements) *
      dg::hdg::runtime::cuda_w8a::maximum_interior_dofs_per_element;
  constexpr std::size_t legacy_trace_capacity =
      dg::hdg::runtime::cuda_w8a::maximum_true_trace_dofs;
  if (rank.interior.size() <= legacy_interior_capacity ||
      rank.owned_true_trace.size() <= legacy_trace_capacity)
  {
    std::cerr << "checkpoint inspection did not exceed both legacy DTO "
                 "payload capacities\n";
    return 1;
  }
  std::ostringstream summary;
  summary << "checkpoint-summary"
          << " identity=" << Hex(artifact.numerical_identity)
          << " time=" << std::hex << DoubleBits(cursor.current_time)
          << " step=" << std::dec << cursor.committed_steps
          << " next_sequence=" << cursor.next_sequence
          << " next_fixed_cycle=" << cursor.next_fixed_cycle
          << " next_publication_cycle=" << cursor.next_publication_cycle
          << " next_explicit_index=" << cursor.next_explicit_index
          << " output_cycle=" << time.output_cycle
          << " mode=" << static_cast<int>(time.mode)
          << " adaptive=" << static_cast<int>(
                 time.adaptive_controller.has_value());
  if (time.adaptive_controller)
  {
    const auto& controller = *time.adaptive_controller;
    summary << " controller_revision=" << controller.revision
            << " accepted=" << controller.accepted_steps
            << " rejected=" << controller.rejected_attempts
            << " consecutive_rejections="
            << controller.consecutive_rejections
            << " next_dt=" << std::hex << DoubleBits(controller.next_dt)
            << " last_dt=" << DoubleBits(controller.last_attempted_dt)
            << " last_error=" <<
                 DoubleBits(controller.last_normalized_error)
            << " last_accepted=" << std::dec
            << static_cast<int>(controller.last_step_accepted);
  }
  summary << " descriptor_time=" << std::hex
          << DoubleBits(rank.descriptor.coordinates.time)
          << " descriptor_dt=" << DoubleBits(rank.descriptor.coordinates.dt)
          << " descriptor_step=" << std::dec
          << rank.descriptor.coordinates.step
          << " descriptor_generation="
          << rank.descriptor.coordinates.generation
          << " descriptor_cursor=" << rank.descriptor.coordinates.cursor
          << " interior_count=" << rank.interior.size()
          << " trace_count=" << rank.owned_true_trace.size()
          << " local_interior_hash=" << Hex(rank.local_interior_hash)
          << " local_trace_hash=" << Hex(rank.local_trace_hash)
          << " aggregate_interior_hash="
          << Hex(rank.descriptor.interior_hash)
          << " aggregate_trace_hash=" << Hex(rank.descriptor.trace_hash);
  std::cout << summary.str() << '\n';
  return 0;
}

int Run(const Arguments& arguments)
{
  constexpr int order = 2;
  dg_sim::CaseConfig case_config;
  case_config.case_name = "cuda_checkpoint_split_test";
  case_config.gamma = mfem::igrlab::EulerDefaults::kGamma;
  case_config.case_dir = DG_IGR_HDG_REPOSITORY_ROOT;
  case_config.vtk_config.prefix_path = arguments.vtk_root;

  dg_sim::CommonOptions options;
  options.mesh_file = arguments.mesh;
  options.ser_ref_levels = 0;
  options.par_ref_levels = 0;
  options.order = order;
  options.device = "cuda";
  options.discretization = "hdg";
  options.hdg_cuda_engine = "ordinary";
  options.regularization = "igr";
  options.hdg_igr_alpha = 0.2;
  options.hdg_igr_beta = 0.8;
  options.time_integrator = "rodas3";
  options.time_atol = 1.0e6;
  options.time_rtol = 1.0e6;
  options.dt = -1.0e-5;
  options.dt_min = 1.0e-8;
  options.cfl = 0.25;
  options.t_final = 4.0e-5;
  options.vtk_save = true;
  options.dt_save = 1.0e-5;
  options.glvis = false;
  options.analysis_steps = 0;
  options.hdg_checkpoint = arguments.checkpoint;
  options.hdg_restart_checkpoint = arguments.restart;

  inject_after_two_commits = arguments.inject_after_two;
  const dg_sim::hdg::HdgEulerCaseDataFactory factory =
      [order, &arguments](const dg_sim::CaseContext& context)
      { return MakeCaseData(context, order, arguments.perturb_initial,
                            arguments.perturb_source); };
  const dg_sim::hdg::HdgEulerMainRequest request{
      case_config, options, factory, MPI_COMM_WORLD, {}};
  return dg_sim::hdg::RunEulerMainBackend(request);
}

} // namespace

extern "C" bool dg_igr_hdg_cuda_prepare_boundary_failure_for_test(
    const std::uint64_t committed_steps,
    const double committed_time) noexcept
{
  return inject_after_two_commits && committed_steps == 2 &&
      std::isfinite(committed_time) && committed_time > 0.0;
}

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  Arguments arguments;
  if (!ParseArguments(argc, argv, arguments))
  {
    if (mfem::Mpi::Root())
    {
      std::cerr
          << "Usage: " << argv[0]
          << " --mesh PATH --checkpoint PATH --vtk-root PATH"
             " [--restart PATH] [--inject-after-two]"
             " [--perturb-initial] [--perturb-source]\n"
             "   or: --inspect-checkpoint PATH\n";
    }
    return 2;
  }
  if (!arguments.inspect.empty())
  {
    return mfem::Mpi::Root() ? InspectCheckpoint(arguments.inspect) : 0;
  }
  mfem::Device device("cuda");
  (void)device;
  return Run(arguments);
}
