#include "numerics/hdg/runtime/runtime_abi.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"
#include "numerics/hdg/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace
{

using namespace dg::hdg::runtime;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "shared runtime ABI must compile in strict C++17 mode");

bool Check(const bool condition, const char* name)
{
  if (!condition) { std::cerr << "FAILED: " << name << '\n'; }
  return condition;
}

std::uint8_t HexNibble(const char value)
{
  if (value >= '0' && value <= '9')
  {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f')
  {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  return static_cast<std::uint8_t>(value - 'A' + 10);
}

std::vector<std::uint8_t> HexBytes(const std::string_view text)
{
  std::vector<std::uint8_t> bytes(text.size() / 2);
  for (std::size_t index = 0; index < bytes.size(); ++index)
  {
    bytes[index] = static_cast<std::uint8_t>(
        (HexNibble(text[2 * index]) << 4U) |
        HexNibble(text[2 * index + 1]));
  }
  return bytes;
}

Sha256Digest HexDigest(const std::string_view text)
{
  Sha256Digest digest{};
  const std::vector<std::uint8_t> bytes = HexBytes(text);
  if (bytes.size() == digest.size())
  {
    std::copy(bytes.begin(), bytes.end(), digest.begin());
  }
  return digest;
}

StateHash FilledHash(const std::uint8_t first)
{
  StateHash hash{};
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    hash.bytes[index] = static_cast<std::uint8_t>(first + index);
  }
  return hash;
}

bool SameTransfer(const TransferEventV1& left,
                  const TransferEventV1& right)
{
  return left.ordinal == right.ordinal &&
      left.call_site_id == right.call_site_id &&
      left.direction == right.direction && left.category == right.category &&
      left.phase == right.phase && left.success == right.success &&
      left.bytes == right.bytes &&
      left.source_capacity_bytes == right.source_capacity_bytes &&
      left.destination_capacity_bytes == right.destination_capacity_bytes &&
      left.source_storage_token == right.source_storage_token &&
      left.destination_storage_token == right.destination_storage_token &&
      left.endpoint_token == right.endpoint_token &&
      left.attempt_cursor == right.attempt_cursor &&
      left.target_step == right.target_step &&
      left.generation == right.generation && left.rank == right.rank;
}

bool SameAllocation(const AllocationEventV1& left,
                    const AllocationEventV1& right)
{
  return left.ordinal == right.ordinal &&
      left.call_site_id == right.call_site_id &&
      left.operation == right.operation && left.phase == right.phase &&
      left.success == right.success &&
      left.requested_bytes == right.requested_bytes &&
      left.requested_alignment == right.requested_alignment &&
      left.capacity_bytes == right.capacity_bytes &&
      left.storage_token == right.storage_token &&
      left.attempt_cursor == right.attempt_cursor &&
      left.target_step == right.target_step &&
      left.generation == right.generation && left.rank == right.rank;
}

bool CheckSha256()
{
  bool passed = true;
  passed = Check(
               ComputeSha256(std::string_view{}) ==
                   HexDigest("e3b0c44298fc1c149afbf4c8996fb924"
                             "27ae41e4649b934ca495991b7852b855"),
               "sha256_empty_fips_vector") &&
      passed;
  const Sha256Digest abc =
      HexDigest("ba7816bf8f01cfea414140de5dae2223"
                "b00361a396177a9cb410ff61f20015ad");
  passed = Check(ComputeSha256("abc") == abc, "sha256_abc_fips_vector") &&
      passed;

  Sha256 chunked;
  Sha256Digest chunked_digest{};
  Sha256Digest repeated_digest{};
  const bool updated = chunked.Update("a", 1) && chunked.Update("b", 1) &&
      chunked.Update("c", 1);
  const bool finalized = chunked.Finalize(chunked_digest) &&
      chunked.Finalize(repeated_digest);
  passed = Check(updated && finalized && chunked_digest == abc &&
                     repeated_digest == abc,
                 "sha256_chunked_equals_one_shot_and_finalize_is_stable") &&
      passed;
  return passed;
}

bool CheckRuntimeSelection()
{
  const RuntimeTupleDescriptor cpu_serial =
      CpuSerialRuntimeSelection::Descriptor();
  const RuntimeTupleDescriptor cpu_rank2 =
      CpuRank2RuntimeSelection::Descriptor();
  const RuntimeTupleDescriptor cuda_rank1 =
      CudaRank1OneGpuRuntimeSelection::Descriptor();
  const RuntimeTupleDescriptor cuda_rank2 =
      CudaRank2OneGpuHostStagedRuntimeSelection::Descriptor();
  const RuntimeTupleDescriptor partitioned_cpu_rank1 =
      PartitionedBackwardEulerCpuRank1RuntimeSelection::Descriptor();
  const RuntimeTupleDescriptor partitioned_cpu_rank2 =
      PartitionedBackwardEulerCpuRank2RuntimeSelection::Descriptor();
  const RuntimeTupleDescriptor partitioned_cuda_rank1 =
      PartitionedBackwardEulerCudaRank1RuntimeSelection::Descriptor();
  const RuntimeTupleDescriptor partitioned_cuda_rank2 =
      PartitionedBackwardEulerCudaRank2RuntimeSelection::Descriptor();
  return Check(
      conventional_persistent_card_id ==
              "exp.hdg.euler.conventional.smooth_persistent.w8a.v1" &&
          conventional_persistent_case_id ==
              "conventional-smooth-persistent-v1" &&
          cpu_serial.backend.view() == "cpu.serial" &&
          cpu_rank2.backend.view() == "cpu.rank2" &&
          cuda_rank1.backend.view() == "cuda.rank1.one_gpu" &&
          cuda_rank2.backend.view() ==
              "cuda.rank2.one_gpu.host_staged" &&
          cpu_serial.model.view() == "euler.compressible.2d" &&
          cpu_serial.formulation.view() == "hdg.conventional" &&
          cpu_serial.trace.view() == "mfem.dg_interface.trace" &&
          cpu_serial.stabilization.view() == "tau.fixed" &&
          cpu_serial.time_treatment.view() == "backward_euler" &&
          cpu_serial.precision.view() == "fp64" &&
          partitioned_cpu_rank1.backend.view() == "cpu.serial" &&
          partitioned_cpu_rank2.backend.view() == "cpu.rank2" &&
          partitioned_cuda_rank1.backend.view() == "cuda.rank1.one_gpu" &&
          partitioned_cuda_rank2.backend.view() ==
              "cuda.rank2.one_gpu.host_staged" &&
          partitioned_cpu_rank1.time_treatment.view() ==
              "partitioned_backward_euler" &&
          partitioned_cpu_rank2.time_treatment.view() ==
              "partitioned_backward_euler" &&
          partitioned_cuda_rank1.time_treatment.view() ==
              "partitioned_backward_euler" &&
          partitioned_cuda_rank2.time_treatment.view() ==
              "partitioned_backward_euler",
      "runtime_selection_exact_card_and_partitioned_backend_ids");
}

bool CheckTransferCodec()
{
  const TransferEventV1 value{0x0102030405060708ULL,
                              0x0a0b0c0dU,
                              TransferDirection::DeviceToDevice,
                              TransferCategory::ValidationHistory,
                              FailurePhase::TerminalEvidence,
                              1,
                              0x1112131415161718ULL,
                              0x2122232425262728ULL,
                              0x3132333435363738ULL,
                              0x4142434445464748ULL,
                              0x5152535455565758ULL,
                              0x6162636465666768ULL,
                              0x7172737475767778ULL,
                              0x8182838485868788ULL,
                              0x9192939495969798ULL,
                              -2};
  const std::vector<std::uint8_t> fixture = HexBytes(
      "08070605040302010d0c0b0a02060b01"
      "18171615141312112827262524232221"
      "38373635343332314847464544434241"
      "58575655545352516867666564636261"
      "78777675747372718887868584838281"
      "9897969594939291feffffff");
  SerializedTransferEventV1 serialized{};
  TransferEventV1 decoded{};
  bool passed = Check(SerializeTransferEventV1(value, serialized) &&
                          fixture.size() == serialized.size() &&
                          std::equal(fixture.begin(), fixture.end(),
                                     serialized.begin()) &&
                          DecodeTransferEventV1(
                              serialized.data(), serialized.size(), decoded) &&
                          SameTransfer(value, decoded),
                      "transfer_event_fieldwise_little_endian_fixture");

  const TransferEventV1 sentinel = decoded;
  for (const std::size_t offset : {std::size_t{12}, std::size_t{13},
                                   std::size_t{14}, std::size_t{15}})
  {
    SerializedTransferEventV1 invalid = serialized;
    invalid[offset] = 0xffU;
    TransferEventV1 unchanged = sentinel;
    passed = Check(!DecodeTransferEventV1(
                           invalid.data(), invalid.size(), unchanged) &&
                       SameTransfer(unchanged, sentinel),
                   "transfer_decoder_rejects_unknown_ordinal_or_success") &&
        passed;
  }
  TransferEventV1 unchanged = sentinel;
  passed = Check(!DecodeTransferEventV1(serialized.data(),
                                        serialized.size() - 1, unchanged) &&
                     SameTransfer(unchanged, sentinel) &&
                     !DecodeTransferEventV1(nullptr, serialized.size(),
                                            unchanged),
                 "transfer_decoder_rejects_bad_length_and_null") &&
      passed;

  TransferEventV1 invalid_value = value;
  invalid_value.success = 2;
  SerializedTransferEventV1 unchanged_bytes{};
  unchanged_bytes.fill(0x5aU);
  const SerializedTransferEventV1 before = unchanged_bytes;
  passed = Check(!SerializeTransferEventV1(invalid_value, unchanged_bytes) &&
                     unchanged_bytes == before,
                 "transfer_serializer_rejects_invalid_value_atomically") &&
      passed;
  return passed;
}

bool CheckAllocationCodec()
{
  const AllocationEventV1 value{0x0102030405060708ULL,
                                0x0a0b0c0dU,
                                AllocationOperation::Deallocate,
                                FailurePhase::Commit,
                                1,
                                0x1112131415161718ULL,
                                0x2122232425262728ULL,
                                0x3132333435363738ULL,
                                0x4142434445464748ULL,
                                0x5152535455565758ULL,
                                0x6162636465666768ULL,
                                0x7172737475767778ULL,
                                -2};
  const std::vector<std::uint8_t> fixture = HexBytes(
      "08070605040302010d0c0b0a010901"
      "18171615141312112827262524232221"
      "38373635343332314847464544434241"
      "58575655545352516867666564636261"
      "7877767574737271feffffff");
  SerializedAllocationEventV1 serialized{};
  AllocationEventV1 decoded{};
  bool passed = Check(SerializeAllocationEventV1(value, serialized) &&
                          fixture.size() == serialized.size() &&
                          std::equal(fixture.begin(), fixture.end(),
                                     serialized.begin()) &&
                          DecodeAllocationEventV1(
                              serialized.data(), serialized.size(), decoded) &&
                          SameAllocation(value, decoded),
                      "allocation_event_fieldwise_little_endian_fixture");

  const AllocationEventV1 sentinel = decoded;
  for (const std::size_t offset : {std::size_t{12}, std::size_t{13},
                                   std::size_t{14}})
  {
    SerializedAllocationEventV1 invalid = serialized;
    invalid[offset] = 0xffU;
    AllocationEventV1 unchanged = sentinel;
    passed = Check(!DecodeAllocationEventV1(
                           invalid.data(), invalid.size(), unchanged) &&
                       SameAllocation(unchanged, sentinel),
                   "allocation_decoder_rejects_unknown_ordinal_or_success") &&
        passed;
  }
  AllocationEventV1 unchanged = sentinel;
  passed = Check(!DecodeAllocationEventV1(serialized.data(),
                                          serialized.size() - 1, unchanged) &&
                     SameAllocation(unchanged, sentinel) &&
                     !DecodeAllocationEventV1(nullptr, serialized.size(),
                                              unchanged),
                 "allocation_decoder_rejects_bad_length_and_null") &&
      passed;

  AllocationEventV1 invalid_value = value;
  invalid_value.operation = static_cast<AllocationOperation>(9);
  SerializedAllocationEventV1 unchanged_bytes{};
  unchanged_bytes.fill(0xa5U);
  const SerializedAllocationEventV1 before = unchanged_bytes;
  passed = Check(!SerializeAllocationEventV1(invalid_value, unchanged_bytes) &&
                     unchanged_bytes == before,
                 "allocation_serializer_rejects_invalid_value_atomically") &&
      passed;
  return passed;
}

bool CheckRankManifestAndCheckpointBinding()
{
  std::array<RankPayloadTupleV1, 2> tuples{{
      {0, 3, FilledHash(0)},
      {1, 5, FilledHash(32)},
  }};
  const std::vector<std::uint8_t> fixed_preimage = HexBytes(
      "44474947522d4844472d52414e4b2d5041594c4f41442d5631"
      "0102000000"
      "000000000300000000000000"
      "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
      "010000000500000000000000"
      "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f");
  const Sha256Digest fixed_digest =
      HexDigest("70bfc8edc8e5f38f7f02393148d82f63"
                "4173ec5b6c4ecc3916233e7aab57bb28");

  std::size_t preimage_size = 0;
  const bool sized = RankPayloadManifestSizeV1(tuples.size(), preimage_size);
  std::vector<std::uint8_t> encoded(preimage_size);
  RankPayloadHashV1 from_tuples{};
  RankPayloadHashV1 from_bytes{};
  bool passed = Check(
      sized && preimage_size == 118 &&
          SerializeRankPayloadManifestV1(RankPayloadKind::Interior,
                                         tuples.data(), tuples.size(),
                                         encoded.data(), encoded.size()) &&
          encoded == fixed_preimage &&
          HashRankPayloadManifestV1(RankPayloadKind::Interior,
                                    tuples.data(), tuples.size(), from_tuples) &&
          DecodeAndHashRankPayloadManifestV1(encoded.data(), encoded.size(),
                                             from_bytes) &&
          from_tuples.kind == RankPayloadKind::Interior &&
          from_bytes.kind == RankPayloadKind::Interior &&
          from_tuples.hash.bytes == fixed_digest &&
          from_bytes.hash == from_tuples.hash,
      "rank_manifest_two_unequal_ranks_fixed_preimage_and_digest");

  const RankPayloadHashV1 sentinel{RankPayloadKind::TrueTrace,
                                  FilledHash(0xa0U)};
  auto rejects = [&](std::vector<std::uint8_t> invalid,
                     const char* name) {
    RankPayloadHashV1 unchanged = sentinel;
    return Check(!DecodeAndHashRankPayloadManifestV1(
                         invalid.data(), invalid.size(), unchanged) &&
                     unchanged.kind == sentinel.kind &&
                     unchanged.hash == sentinel.hash,
                 name);
  };

  std::vector<std::uint8_t> bad_domain = encoded;
  bad_domain[0] ^= 1U;
  passed = rejects(bad_domain, "rank_manifest_rejects_bad_domain") && passed;
  std::vector<std::uint8_t> bad_kind = encoded;
  bad_kind[25] = 3;
  passed = rejects(bad_kind, "rank_manifest_rejects_bad_kind") && passed;
  std::vector<std::uint8_t> zero_count(encoded.begin(), encoded.begin() + 30);
  zero_count[26] = 0;
  zero_count[27] = 0;
  zero_count[28] = 0;
  zero_count[29] = 0;
  passed = rejects(zero_count, "rank_manifest_rejects_zero_count") && passed;
  std::vector<std::uint8_t> too_short(encoded.begin(), encoded.end() - 1);
  passed = rejects(too_short, "rank_manifest_rejects_short_length") && passed;
  std::vector<std::uint8_t> too_long = encoded;
  too_long.push_back(0);
  passed = rejects(too_long, "rank_manifest_rejects_long_length") && passed;

  std::vector<std::uint8_t> out_of_order = encoded;
  out_of_order[30] = 1;
  passed = rejects(out_of_order, "rank_manifest_rejects_out_of_order_rank") &&
      passed;
  std::vector<std::uint8_t> duplicate = encoded;
  duplicate[74] = 0;
  passed = rejects(duplicate, "rank_manifest_rejects_duplicate_rank") &&
      passed;
  std::vector<std::uint8_t> missing = encoded;
  missing[74] = 2;
  passed = rejects(missing, "rank_manifest_rejects_missing_rank") && passed;

  std::vector<std::uint8_t> huge_count(encoded.begin(), encoded.begin() + 30);
  huge_count[26] = 0xffU;
  huge_count[27] = 0xffU;
  huge_count[28] = 0xffU;
  huge_count[29] = 0xffU;
  std::size_t ignored = 0;
  passed = Check(!RankPayloadManifestSizeV1(
                         std::numeric_limits<std::size_t>::max(), ignored) &&
                     rejects(huge_count,
                             "rank_manifest_rejects_overflow_or_size_mismatch"),
                 "rank_manifest_checks_overflow_before_hash") &&
      passed;

  RankPayloadHashV1 true_trace{};
  const bool trace_hashed = HashRankPayloadManifestV1(
      RankPayloadKind::TrueTrace, tuples.data(), tuples.size(), true_trace);
  CheckpointDescriptor checkpoint{};
  checkpoint.schema_version = checkpoint_schema_version;
  checkpoint.interior_hash = FilledHash(0xe0U);
  checkpoint.trace_hash = FilledHash(0xc0U);
  const StateHash local_interior = FilledHash(0x40U);
  const StateHash local_trace = FilledHash(0x60U);
  CheckpointMetadataV1 metadata{};
  passed = Check(trace_hashed &&
                     MakeCheckpointMetadataV1(checkpoint, from_tuples,
                                              true_trace, local_interior,
                                              local_trace, metadata) &&
                     metadata.checkpoint.interior_hash == from_tuples.hash &&
                     metadata.checkpoint.trace_hash == true_trace.hash &&
                     metadata.local_interior_hash == local_interior &&
                     metadata.local_true_trace_hash == local_trace,
                 "checkpoint_metadata_binds_rank_common_and_keeps_local_hashes") &&
      passed;

  CheckpointMetadataV1 unchanged_metadata = metadata;
  passed = Check(!MakeCheckpointMetadataV1(
                         checkpoint, true_trace, from_tuples, local_interior,
                         local_trace, unchanged_metadata) &&
                     unchanged_metadata.checkpoint.interior_hash ==
                         metadata.checkpoint.interior_hash &&
                     unchanged_metadata.local_interior_hash ==
                         metadata.local_interior_hash,
                 "checkpoint_metadata_rejects_swapped_payload_tokens_atomically") &&
      passed;
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckSha256() && passed;
  passed = CheckRuntimeSelection() && passed;
  passed = CheckTransferCodec() && passed;
  passed = CheckAllocationCodec() && passed;
  passed = CheckRankManifestAndCheckpointBinding() && passed;
  return passed ? 0 : 1;
}
