#include "numerics/hdg/runtime/cuda/native_element_schwarz_plan.hpp"
#include "numerics/hdg/runtime/cuda/trace_csr_graph_plan.hpp"

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
static_assert(NativeElementSchwarzIgrFieldSplitLayout(1).valid);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(1).euler_rows == 32);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(1).sigma_rows == 8);
static_assert(
    NativeElementSchwarzIgrFieldSplitLayout(1).required_doubles == 1424);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(1).warp_valid);
static_assert(
    NativeElementSchwarzIgrFieldSplitLayout(1).warp_required_doubles == 1504);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(2).euler_rows == 48);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(2).sigma_rows == 12);
static_assert(
    NativeElementSchwarzIgrFieldSplitLayout(2).required_doubles == 3144);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(2).warp_valid);
static_assert(
    NativeElementSchwarzIgrFieldSplitLayout(2).warp_required_doubles == 3264);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(3).euler_rows == 64);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(3).sigma_rows == 16);
static_assert(
    NativeElementSchwarzIgrFieldSplitLayout(3).required_doubles == 5536);
static_assert(NativeElementSchwarzIgrFieldSplitLayout(3).warp_valid);
static_assert(
    NativeElementSchwarzIgrFieldSplitLayout(3).warp_required_doubles == 5696);
static_assert(!NativeElementSchwarzIgrFieldSplitLayout(0).valid);
static_assert(!NativeElementSchwarzIgrFieldSplitLayout(4).valid);
static_assert(NativeElementSchwarzEncodeDof(7, -1) == -8);
static_assert(NativeElementSchwarzDecodedDof(-8) == 7);

bool CheckFieldSplitLayout()
{
  for (int order = 1; order <= 3; ++order)
  {
    const NativeElementSchwarzFieldSplitLayout layout =
        NativeElementSchwarzIgrFieldSplitLayout(order);
    if (!layout.valid || layout.patch_rows != 20 * (order + 1) ||
        layout.euler_rows != 16 * (order + 1) ||
        layout.sigma_rows != 4 * (order + 1) ||
        layout.euler_lu_offset != 0 ||
        layout.sigma_lu_offset !=
            static_cast<std::size_t>(layout.euler_rows) *
                layout.euler_rows ||
        layout.lower_coupling_offset != layout.sigma_lu_offset +
            static_cast<std::size_t>(layout.sigma_rows) *
                layout.sigma_rows ||
        layout.left_scale_offset != layout.lower_coupling_offset +
            static_cast<std::size_t>(layout.sigma_rows) *
                layout.euler_rows ||
        layout.right_scale_offset !=
            layout.left_scale_offset + layout.patch_rows ||
        layout.warp_inverse_diagonal_offset != layout.required_doubles ||
        layout.warp_rhs_permutation_offset !=
            layout.warp_inverse_diagonal_offset + layout.patch_rows ||
        layout.warp_required_doubles !=
            layout.warp_rhs_permutation_offset + layout.patch_rows ||
        layout.required_doubles > layout.available_doubles ||
        !layout.warp_valid ||
        layout.warp_required_doubles > layout.available_doubles)
      return false;

    int euler_rows = 0;
    int sigma_rows = 0;
    std::array<bool, 80> visited{};
    for (int row = 0; row < layout.patch_rows; ++row)
    {
      const int packed = NativeElementSchwarzIgrFieldSplitPackedRow(
          order, row);
      if (packed < 0 || packed >= layout.patch_rows ||
          visited[static_cast<std::size_t>(packed)] ||
          NativeElementSchwarzIgrFieldSplitPatchRow(order, packed) != row)
        return false;
      visited[static_cast<std::size_t>(packed)] = true;
      if (NativeElementSchwarzIgrPatchRowIsSigma(order, row))
      {
        if (packed < layout.euler_rows) return false;
        ++sigma_rows;
      }
      else
      {
        if (packed >= layout.euler_rows) return false;
        ++euler_rows;
      }
    }
    if (euler_rows != layout.euler_rows || sigma_rows != layout.sigma_rows)
      return false;
  }
  return true;
}

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
      plan.patch_rows == 2 &&
      plan.minimum_multiplicity == 1 && plan.maximum_multiplicity == 2 &&
      plan.maximum_contributors == 2 &&
      plan.first_element_sources[3] == 3 &&
      plan.second_element_sources[3] == 4 &&
      plan.first_element_sources[4] == 3 &&
      plan.second_element_sources[4] == 4;
}

bool SamePlan(const NativeElementSchwarzPlan& left,
              const NativeElementSchwarzPlan& right)
{
  return left.signed_element_true_dofs ==
          right.signed_element_true_dofs &&
      left.first_element_sources == right.first_element_sources &&
      left.second_element_sources == right.second_element_sources &&
      left.true_multiplicity == right.true_multiplicity &&
      left.true_incidence_slots == right.true_incidence_slots &&
      left.patch_rows == right.patch_rows &&
      left.minimum_multiplicity == right.minimum_multiplicity &&
      left.maximum_multiplicity == right.maximum_multiplicity &&
      left.maximum_contributors == right.maximum_contributors &&
      left.valid == right.valid;
}

bool CheckDirectPlanMatchesCsrDerivedPlan()
{
  // Both trace dofs are shared, exercising two contributors for diagonal and
  // off-diagonal entries. The second element uses opposite orientations.
  const std::array<int, 4> signed_element_local{{0, 1, -1, -2}};
  const std::array<int, 2> local_true{{0, 1}};
  const std::array<int, 8> sources{{0, 1, 2, 3, 4, 5, 6, 7}};
  const std::array<int, 8> sections{{0, 0, 0, 0, 0, 0, 0, 0}};
  const std::array<int, 8> targets{{0, 1, 2, 3, 0, 1, 2, 3}};
  const NativeElementSchwarzPlan csr = BuildNativeElementSchwarzPlan({
      2, 2, 2, 2, 4,
      signed_element_local.data(), signed_element_local.size(),
      local_true.data(), local_true.size(), sources.data(), sections.data(),
      targets.data(), targets.size()});
  const NativeElementSchwarzPlan direct =
      BuildDirectNativeElementSchwarzPlan({
          2, 2, 2, 2,
          signed_element_local.data(), signed_element_local.size(),
          local_true.data(), local_true.size()});
  if (!csr.valid || !direct.valid || !SamePlan(csr, direct) ||
      direct.first_element_sources !=
          std::vector<int>({0, 1, 2, 3, 0, 1, 2, 3}) ||
      direct.second_element_sources !=
          std::vector<int>({4, 5, 6, 7, 4, 5, 6, 7}) ||
      direct.true_incidence_slots != std::vector<int>({0, 2, 1, 3}))
    return false;

  // A boundary/interior mix also checks one-contributor entries and both
  // multiplicity values.
  const std::array<int, 4> mixed_signed{{0, 1, -2, 2}};
  const std::array<int, 3> mixed_true{{0, 1, 2}};
  const std::array<int, 8> mixed_targets{{0, 1, 2, 3, 3, 4, 5, 6}};
  const NativeElementSchwarzPlan mixed_csr = BuildNativeElementSchwarzPlan({
      2, 2, 3, 3, 7, mixed_signed.data(), mixed_signed.size(),
      mixed_true.data(), mixed_true.size(), sources.data(), sections.data(),
      mixed_targets.data(), mixed_targets.size()});
  const NativeElementSchwarzPlan mixed_direct =
      BuildDirectNativeElementSchwarzPlan({
          2, 2, 3, 3, mixed_signed.data(), mixed_signed.size(),
          mixed_true.data(), mixed_true.size()});
  if (!mixed_csr.valid || !mixed_direct.valid ||
      !SamePlan(mixed_csr, mixed_direct) ||
      mixed_direct.minimum_multiplicity != 1 ||
      mixed_direct.maximum_multiplicity != 2)
    return false;

  // Local and true numbering need not coincide on a borrowed topology.
  const std::array<int, 3> permuted_true{{2, 0, 1}};
  const NativeElementSchwarzPlan permuted_csr =
      BuildNativeElementSchwarzPlan({
          2, 2, 3, 3, 7, mixed_signed.data(), mixed_signed.size(),
          permuted_true.data(), permuted_true.size(), sources.data(),
          sections.data(), mixed_targets.data(), mixed_targets.size()});
  const NativeElementSchwarzPlan permuted_direct =
      BuildDirectNativeElementSchwarzPlan({
          2, 2, 3, 3, mixed_signed.data(), mixed_signed.size(),
          permuted_true.data(), permuted_true.size()});
  return permuted_csr.valid && permuted_direct.valid &&
      SamePlan(permuted_csr, permuted_direct) &&
      permuted_direct.minimum_multiplicity == 1 &&
      permuted_direct.maximum_multiplicity == 2;
}

bool CheckDirectPlanFailsClosed()
{
  const std::array<int, 3> three_incidence{{0, 0, 0}};
  const std::array<int, 1> one_truth{{0}};
  const NativeElementSchwarzPlan too_many =
      BuildDirectNativeElementSchwarzPlan({
          3, 1, 1, 1, three_incidence.data(), three_incidence.size(),
          one_truth.data(), one_truth.size()});

  const std::array<int, 2> duplicate_in_element{{0, 0}};
  const NativeElementSchwarzPlan duplicate =
      BuildDirectNativeElementSchwarzPlan({
          1, 2, 1, 1, duplicate_in_element.data(),
          duplicate_in_element.size(), one_truth.data(), one_truth.size()});

  const std::array<int, 1> invalid_local{{1}};
  const NativeElementSchwarzPlan bad_local =
      BuildDirectNativeElementSchwarzPlan({
          1, 1, 1, 1, invalid_local.data(), invalid_local.size(),
          one_truth.data(), one_truth.size()});

  const std::array<int, 1> out_of_range_truth{{1}};
  const std::array<int, 1> valid_local{{0}};
  const NativeElementSchwarzPlan bad_truth =
      BuildDirectNativeElementSchwarzPlan({
          1, 1, 1, 1, valid_local.data(), valid_local.size(),
          out_of_range_truth.data(), out_of_range_truth.size()});

  const NativeElementSchwarzPlan bad_count =
      BuildDirectNativeElementSchwarzPlan({
          1, 1, 1, 1, valid_local.data(), 0,
          one_truth.data(), one_truth.size()});
  return !too_many.valid && !duplicate.valid && !bad_local.valid &&
      !bad_truth.valid && !bad_count.valid;
}

bool CheckDirectTraceCsrGraph()
{
  const std::array<int, 4> signed_element_local{{0, 1, -2, 2}};
  const TraceCsrGraphPlan graph = BuildTraceCsrGraphPlan({
      2, 2, 3, signed_element_local.data(), signed_element_local.size()});
  if (!graph.valid || graph.legacy_columns_sorted ||
      graph.offsets != std::vector<int>({0, 2, 5, 7}) ||
      graph.columns != std::vector<int>({0, 1, 1, 2, 0, 2, 1}))
    return false;

  // Repeated incidence within one element is harmless for the union graph.
  const std::array<int, 3> duplicate{{0, 0, 1}};
  const TraceCsrGraphPlan duplicate_graph = BuildTraceCsrGraphPlan({
      1, 3, 2, duplicate.data(), duplicate.size()});
  if (!duplicate_graph.valid ||
      duplicate_graph.offsets != std::vector<int>({0, 2, 4}) ||
      duplicate_graph.columns != std::vector<int>({0, 1, 1, 0}))
    return false;

  const std::array<int, 1> invalid_local{{2}};
  const TraceCsrGraphPlan bad_local = BuildTraceCsrGraphPlan({
      1, 1, 2, invalid_local.data(), invalid_local.size()});
  const TraceCsrGraphPlan bad_count = BuildTraceCsrGraphPlan({
      1, 1, 1, invalid_local.data(), 0});
  const TraceCsrGraphPlan unreferenced_row = BuildTraceCsrGraphPlan({
      1, 1, 2, duplicate.data(), 1});
  const std::array<int, 2> cancelling_diagonal{{0, -1}};
  const TraceCsrGraphPlan cancelled = BuildTraceCsrGraphPlan({
      1, 2, 1, cancelling_diagonal.data(), cancelling_diagonal.size()});
  return !bad_local.valid && !bad_count.valid && !unreferenced_row.valid &&
      !cancelled.valid;
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

bool CheckMatrixFreeActionAndRhsModel()
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

  const std::array<double, 8> schur{{2.0, 1.0, 3.0, 4.0,
                                     6.0, 5.0, 7.0, 8.0}};
  const std::array<double, 4> element_rhs{{1.0, 2.0, 3.0, 4.0}};
  const std::array<double, 3> input{{2.0, 3.0, 5.0}};
  std::array<double, 9> assembled{};
  std::array<double, 3> assembled_rhs{};
  for (int element = 0; element < 2; ++element)
    for (int row = 0; row < 2; ++row)
    {
      const int row_slot = 2 * element + row;
      const int row_truth = NativeElementSchwarzDecodedDof(
          plan.signed_element_true_dofs[static_cast<std::size_t>(row_slot)]);
      const int row_sign = NativeElementSchwarzDofSign(
          plan.signed_element_true_dofs[static_cast<std::size_t>(row_slot)]);
      assembled_rhs[static_cast<std::size_t>(row_truth)] += row_sign *
          element_rhs[static_cast<std::size_t>(row_slot)];
      for (int column = 0; column < 2; ++column)
      {
        const int column_slot = 2 * element + column;
        const int column_truth = NativeElementSchwarzDecodedDof(
            plan.signed_element_true_dofs[
                static_cast<std::size_t>(column_slot)]);
        const int column_sign = NativeElementSchwarzDofSign(
            plan.signed_element_true_dofs[
                static_cast<std::size_t>(column_slot)]);
        assembled[static_cast<std::size_t>(3 * row_truth + column_truth)] +=
            row_sign * column_sign * schur[static_cast<std::size_t>(
                4 * element + 2 * row + column)];
      }
    }

  std::array<double, 4> element_action{};
  for (int element = 0; element < 2; ++element)
    for (int row = 0; row < 2; ++row)
      for (int column = 0; column < 2; ++column)
      {
        const int column_slot = 2 * element + column;
        const int truth = NativeElementSchwarzDecodedDof(
            plan.signed_element_true_dofs[
                static_cast<std::size_t>(column_slot)]);
        const int sign = NativeElementSchwarzDofSign(
            plan.signed_element_true_dofs[
                static_cast<std::size_t>(column_slot)]);
        element_action[static_cast<std::size_t>(2 * element + row)] +=
            schur[static_cast<std::size_t>(
                4 * element + 2 * row + column)] * sign *
            input[static_cast<std::size_t>(truth)];
      }
  std::array<double, 3> matrix_free_action{};
  for (int truth = 0; truth < 3; ++truth)
    for (int incidence = 0; incidence < 2; ++incidence)
    {
      const int slot = plan.true_incidence_slots[
          static_cast<std::size_t>(2 * truth + incidence)];
      if (slot >= 0)
        matrix_free_action[static_cast<std::size_t>(truth)] +=
            NativeElementSchwarzDofSign(
                plan.signed_element_true_dofs[
                    static_cast<std::size_t>(slot)]) *
            element_action[static_cast<std::size_t>(slot)];
    }
  std::array<double, 3> assembled_action{};
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      assembled_action[static_cast<std::size_t>(row)] +=
          assembled[static_cast<std::size_t>(3 * row + column)] *
          input[static_cast<std::size_t>(column)];
  return assembled_rhs == std::array<double, 3>{{1.0, -1.0, 4.0}} &&
      assembled_action == std::array<double, 3>{{7.0, 11.0, 19.0}} &&
      matrix_free_action == assembled_action;
}
} // namespace

int main()
{
  if (!CheckFieldSplitLayout() || !CheckExactTwoElementPlan() ||
      !CheckDirectPlanMatchesCsrDerivedPlan() ||
      !CheckDirectPlanFailsClosed() ||
      !CheckDirectTraceCsrGraph() ||
      !CheckSignedWeightedApplyModel() ||
      !CheckFailClosedPlans() || !CheckMatrixFreeActionAndRhsModel())
  {
    std::cerr << "native element Schwarz topology-plan checks failed\n";
    return 1;
  }
  std::cout << "native element Schwarz topology-plan checks passed\n";
  return 0;
}
