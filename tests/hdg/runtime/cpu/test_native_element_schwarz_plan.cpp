#include "numerics/hdg/runtime/cuda/native_element_schwarz_plan.hpp"

#include <array>
#include <iostream>

namespace
{
using namespace dg::hdg::runtime::cuda_w8a::detail;

static_assert(NativeElementSchwarzIgrPatchRows(1) == 40);
static_assert(NativeElementSchwarzIgrPatchRows(2) == 60);
static_assert(NativeElementSchwarzIgrPatchRows(3) == 80);
static_assert(NativeElementSchwarzIgrPatchRows(0) == 0);
static_assert(NativeElementSchwarzIgrPatchRows(4) == 0);
static_assert(NativeElementSchwarzEncodeDof(7, -1) == -8);
static_assert(NativeElementSchwarzDecodedDof(-8) == 7);

bool CheckExactTwoElementPlan()
{
  // Two two-dof patches overlap on true dof 1.  The second patch sees that
  // shared dof with the opposite element orientation.
  const std::array<int, 4> signed_element_local{{0, 1, -2, 2}};
  const std::array<int, 3> local_true{{0, 1, 2}};
  const std::array<int, 8> sources{{0, 1, 2, 3, 4, 5, 6, 7}};
  const std::array<int, 8> sections{{0, 0, 0, 0, 0, 0, 0, 0}};
  const std::array<int, 8> targets{{0, 1, 2, 3, 3, 4, 5, 6}};
  const NativeElementSchwarzPlan plan = BuildNativeElementSchwarzPlan({
      2, 2, 3, 3, 7,
      signed_element_local.data(), signed_element_local.size(),
      local_true.data(), local_true.size(), sources.data(), sections.data(),
      targets.data(), targets.size()});
  return plan.valid &&
      plan.signed_element_true_dofs ==
          std::vector<int>({0, 1, -2, 2}) &&
      plan.true_multiplicity == std::vector<int>({1, 2, 1}) &&
      plan.true_incidence_slots ==
          std::vector<int>({0, -1, 1, 2, 3, -1}) &&
      plan.minimum_multiplicity == 1 && plan.maximum_multiplicity == 2 &&
      plan.maximum_contributors == 2 &&
      plan.first_element_sources[3] == 3 &&
      plan.second_element_sources[3] == 4 &&
      plan.first_element_sources[4] == 3 &&
      plan.second_element_sources[4] == 4;
}

bool CheckSignedWeightedApplyModel()
{
  const std::array<int, 4> signed_element_local{{0, 1, -2, 2}};
  const std::array<int, 3> local_true{{0, 1, 2}};
  const std::array<int, 8> sources{{0, 1, 2, 3, 4, 5, 6, 7}};
  const std::array<int, 8> sections{{0, 0, 0, 0, 0, 0, 0, 0}};
  const std::array<int, 8> targets{{0, 1, 2, 3, 3, 4, 5, 6}};
  const NativeElementSchwarzPlan plan = BuildNativeElementSchwarzPlan({
      2, 2, 3, 3, 7,
      signed_element_local.data(), signed_element_local.size(),
      local_true.data(), local_true.size(), sources.data(), sections.data(),
      targets.data(), targets.size()});
  if (!plan.valid) return false;

  // Diagonal element Schur blocks. The shared true diagonal is the signed sum
  // 4 + 6 = 10; both exact principal patches must see that full value.
  const std::array<double, 8> local_schur{{2.0, 0.0, 0.0, 4.0,
                                           6.0, 0.0, 0.0, 3.0}};
  std::array<double, 8> signed_sources{};
  for (int source = 0; source < 8; ++source)
  {
    const int element = source / 4;
    const int entry = source % 4;
    const int row = entry / 2;
    const int column = entry % 2;
    const int row_sign = NativeElementSchwarzDofSign(
        signed_element_local[static_cast<std::size_t>(2 * element + row)]);
    const int column_sign = NativeElementSchwarzDofSign(
        signed_element_local[
            static_cast<std::size_t>(2 * element + column)]);
    signed_sources[static_cast<std::size_t>(source)] =
        row_sign * column_sign *
        local_schur[static_cast<std::size_t>(source)];
  }
  std::array<double, 8> patches{};
  for (int target = 0; target < 8; ++target)
  {
    const int element = target / 4;
    const int entry = target % 4;
    const int row = entry / 2;
    const int column = entry % 2;
    const int first = plan.first_element_sources[
        static_cast<std::size_t>(target)];
    const int second = plan.second_element_sources[
        static_cast<std::size_t>(target)];
    const int row_sign = NativeElementSchwarzDofSign(
        plan.signed_element_true_dofs[
            static_cast<std::size_t>(2 * element + row)]);
    const int column_sign = NativeElementSchwarzDofSign(
        plan.signed_element_true_dofs[
            static_cast<std::size_t>(2 * element + column)]);
    patches[static_cast<std::size_t>(target)] = row_sign * column_sign *
        (signed_sources[static_cast<std::size_t>(first)] +
         (second >= 0 ? signed_sources[static_cast<std::size_t>(second)]
                      : 0.0));
  }
  if (patches != std::array<double, 8>{{2.0, 0.0, 0.0, 10.0,
                                        10.0, 0.0, 0.0, 3.0}})
    return false;

  const std::array<double, 3> input{{2.0, 20.0, 6.0}};
  std::array<double, 4> element_solution{};
  for (int slot = 0; slot < 4; ++slot)
  {
    const int truth = NativeElementSchwarzDecodedDof(
        plan.signed_element_true_dofs[static_cast<std::size_t>(slot)]);
    const int sign = NativeElementSchwarzDofSign(
        plan.signed_element_true_dofs[static_cast<std::size_t>(slot)]);
    const int element = slot / 2;
    const int row = slot % 2;
    element_solution[static_cast<std::size_t>(slot)] = sign *
        input[static_cast<std::size_t>(truth)] /
        patches[static_cast<std::size_t>(4 * element + 3 * row)];
  }
  std::array<double, 3> output{};
  for (int truth = 0; truth < 3; ++truth)
  {
    for (int incidence = 0; incidence < 2; ++incidence)
    {
      const int slot = plan.true_incidence_slots[
          static_cast<std::size_t>(2 * truth + incidence)];
      if (slot >= 0)
        output[static_cast<std::size_t>(truth)] +=
            NativeElementSchwarzDofSign(
                plan.signed_element_true_dofs[
                    static_cast<std::size_t>(slot)]) *
            element_solution[static_cast<std::size_t>(slot)];
    }
    output[static_cast<std::size_t>(truth)] /=
        plan.true_multiplicity[static_cast<std::size_t>(truth)];
  }
  return output == std::array<double, 3>{{1.0, 2.0, 2.0}};
}

bool CheckFailClosedPlans()
{
  const std::array<int, 3> signed_element_local{{0, 0, 0}};
  const std::array<int, 1> local_true{{0}};
  const std::array<int, 3> sources{{0, 1, 2}};
  const std::array<int, 3> sections{{0, 0, 0}};
  const std::array<int, 3> targets{{0, 0, 0}};
  const NativeElementSchwarzPlan three_contributors =
      BuildNativeElementSchwarzPlan({
          3, 1, 1, 1, 1,
          signed_element_local.data(), signed_element_local.size(),
          local_true.data(), local_true.size(), sources.data(),
          sections.data(), targets.data(), targets.size()});

  auto offd_sections = sections;
  offd_sections[1] = 1;
  const NativeElementSchwarzPlan offd = BuildNativeElementSchwarzPlan({
      3, 1, 1, 1, 1,
      signed_element_local.data(), signed_element_local.size(),
      local_true.data(), local_true.size(), sources.data(),
      offd_sections.data(), targets.data(), targets.size()});
  return !three_contributors.valid && !offd.valid;
}
} // namespace

int main()
{
  if (!CheckExactTwoElementPlan() || !CheckSignedWeightedApplyModel() ||
      !CheckFailClosedPlans())
  {
    std::cerr << "native element Schwarz topology-plan checks failed\n";
    return 1;
  }
  std::cout << "native element Schwarz topology-plan checks passed\n";
  return 0;
}
