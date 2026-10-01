// Diagnostic interposer for the installed v105 OpenVX build. The runner must also
// validate libOpenVX.so SHA-256 before loading this library. No SDK headers or
// libraries are needed to build it, and it never changes SDK object state.
#include "dlfcn.h"
#include "elf.h"
#include "link.h"
#include "sys/uio.h"
#include "unistd.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>

#ifndef SPECFERRY_VERIFY_TRACE_BUILD_ID
#define SPECFERRY_VERIFY_TRACE_BUILD_ID "92958fb5a5db393d27dd5e28badb521363d8dee1"
#endif

namespace {
constexpr char expected_build_id[] = SPECFERRY_VERIFY_TRACE_BUILD_ID;
constexpr std::size_t node_count_offset = 0xa8;
constexpr std::size_t nodes_offset = 0xb0;
constexpr std::size_t node_capacity_offset = 0xb8;
constexpr std::size_t topology_offset = 0x110;
constexpr std::size_t verified_offset = 0x11c;
constexpr std::size_t execution_table_offset = 0x101c0;
constexpr std::size_t node_kernel_offset = 0xb0;
constexpr std::size_t kernel_name_offset = 0xa8;
constexpr std::size_t kernel_name_capacity = 256;
constexpr std::size_t kernel_enum_offset = 0x1a8;
constexpr std::size_t node_target_index_offset = 0x390;
constexpr std::size_t node_operations_offset = 0x3e8;
constexpr std::size_t operation_count_offset = 0x10;
// Diagnostic budgets, not SDK limits. The SDK starts with 2048 node slots and
// grows its arrays; reject implausible snapshots without changing SDK execution.
constexpr std::uint32_t maximum_snapshot_nodes = 4096;
constexpr std::uint32_t maximum_snapshot_capacity = 65536;

using GraphStatus = int (*)(void *);
using OperationTable = void (*)(void *);
using CreateLayer = int (*)(void *, void **);
using GenerateCommands = int (*)(void *, void *);

struct Calls {
  GraphStatus verify;
  OperationTable operation_table;
  CreateLayer create_layer;
  GraphStatus initialize_nodes;
  GraphStatus verify_tiling;
  GenerateCommands generate_commands;
  GraphStatus prepare_memory;
};

struct Runtime {
  Calls calls;
  std::FILE *output;
  std::FILE *nodes_output;
  std::map<void *, bool> operation_integrity;
  std::mutex integrity_mutex;
};

struct VerifyScope {
  std::uint64_t sequence = 0;
  void *graph = nullptr;
};

std::atomic<std::uint64_t> next_sequence{0};
thread_local VerifyScope current_verify;

[[noreturn]] void refuse_trace(const char *reason) {
  std::fprintf(stderr, "SpecFerry Verify trace refused: %s\n", reason);
  std::fflush(stderr);
  std::_Exit(125);
}

struct BuildIdQuery {
  std::uintptr_t base;
  bool matches = false;
};

std::size_t aligned_note_size(std::size_t size) { return (size + 3) & ~std::size_t(3); }

int inspect_build_id(dl_phdr_info *info, std::size_t, void *opaque) {
  auto &query = *static_cast<BuildIdQuery *>(opaque);
  if (info->dlpi_addr != query.base) {
    return 0;
  }
  constexpr char digits[] = "0123456789abcdef";
  for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
    const auto &header = info->dlpi_phdr[index];
    if (header.p_type != PT_NOTE) {
      continue;
    }
    const auto *bytes = reinterpret_cast<const unsigned char *>(info->dlpi_addr + header.p_vaddr);
    std::size_t offset = 0;
    while (offset + sizeof(ElfW(Nhdr)) <= header.p_memsz) {
      ElfW(Nhdr) note;
      std::memcpy(&note, bytes + offset, sizeof(note));
      offset += sizeof(note);
      const auto name_size = aligned_note_size(note.n_namesz);
      const auto value_size = aligned_note_size(note.n_descsz);
      if (name_size > header.p_memsz - offset || value_size > header.p_memsz - offset - name_size) {
        break;
      }
      const auto *name = bytes + offset;
      const auto *value = name + name_size;
      if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
          std::memcmp(name, "GNU", 4) == 0 && note.n_descsz * 2 == sizeof(expected_build_id) - 1) {
        query.matches = true;
        for (std::size_t byte = 0; byte < note.n_descsz; ++byte) {
          if (expected_build_id[2 * byte] != digits[value[byte] >> 4] ||
              expected_build_id[2 * byte + 1] != digits[value[byte] & 15]) {
            query.matches = false;
            break;
          }
        }
        return 1;
      }
      offset += name_size + value_size;
    }
  }
  return 1;
}

void *find_symbol(const char *name, void *expected_base = nullptr) {
  dlerror();
  auto *address = dlsym(RTLD_NEXT, name);
  if (!address || dlerror()) {
    refuse_trace(name);
  }
  if (expected_base) {
    Dl_info info{};
    if (!dladdr(address, &info) || info.dli_fbase != expected_base) {
      refuse_trace("hook symbols belong to different shared libraries");
    }
  }
  return address;
}

Runtime initialize() {
  auto *verify = find_symbol("vxVerifyGraph");
  Dl_info info{};
  if (!dladdr(verify, &info)) {
    refuse_trace("cannot identify the real vxVerifyGraph library");
  }
  BuildIdQuery build{reinterpret_cast<std::uintptr_t>(info.dli_fbase)};
  dl_iterate_phdr(inspect_build_id, &build);
  if (!build.matches) {
    refuse_trace("libOpenVX build ID does not match the pinned internal layout");
  }
  const auto base = info.dli_fbase;
  Calls calls{
      reinterpret_cast<GraphStatus>(verify),
      reinterpret_cast<OperationTable>(find_symbol("vxoGraph_GenerateOperationTable", base)),
      reinterpret_cast<CreateLayer>(find_symbol("vxnneExecutionLayer_Create", base)),
      reinterpret_cast<GraphStatus>(find_symbol("vxoGraph_InitializeAllNodeKernels", base)),
      reinterpret_cast<GraphStatus>(find_symbol("vxoGraph_VerifyTiling", base)),
      reinterpret_cast<GenerateCommands>(find_symbol("vxnneExecutionLayer_GenerateCommands", base)),
      reinterpret_cast<GraphStatus>(find_symbol("vxoGraph_PrepareParamMemory", base)),
  };
  const auto *path = std::getenv("SPECFERRY_VERIFY_TRACE");
  if (!path || !*path) {
    refuse_trace("SPECFERRY_VERIFY_TRACE must name an unused output file");
  }
  auto *output = std::fopen(path, "wx");
  if (!output) {
    refuse_trace("cannot create Verify trace output");
  }
  const std::string nodes_path = std::string(path) + "-nodes.tsv";
  auto *nodes_output = std::fopen(nodes_path.c_str(), "wx");
  if (!nodes_output) {
    refuse_trace("cannot create Verify node snapshot output");
  }
  std::fputs("event\tapi\tseconds\tunix_seconds\tverify_id\tgraph\tstatus\tsnapshot_valid\t"
             "verified\texecution_table\tfirst_node_operations\targument\tnode_count\t"
             "first_node_snapshot_valid\n",
             output);
  std::fflush(output);
  std::fputs("event\tunix_seconds\tverify_id\tgraph\tnode_count\tnode_capacity\t"
             "topology_position\tnode_index\tnode\toperations\toperation_count\t"
             "operation_pointer_valid\toperation_count_valid\tcondition\tscanned_nodes\t"
             "read_errors\tnull_operations\tzero_operations\ttotal_operations\t"
             "first_null_topology\tfirst_null_node_index\tscan_complete\tscan_seconds\t"
             "kernel_pointer_valid\tkernel\tkernel_name_read_valid\tkernel_name_terminated\t"
             "kernel_name\tkernel_enum_valid\tkernel_enum\ttarget_index_valid\ttarget_index\n",
             nodes_output);
  std::fflush(nodes_output);
  return {calls, output, nodes_output, {}, {}};
}

Runtime &runtime() {
  // Resolve every hook before the first real call. The stream stays open until
  // process exit so SDK destructor activity can still be traced safely.
  static Runtime state = initialize();
  return state;
}

template <typename T> bool read_field(const void *object, std::size_t offset, T &value) {
  if (!object) {
    return false;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(object);
  constexpr auto maximum_address = std::numeric_limits<std::uintptr_t>::max();
  if (offset > maximum_address - base || sizeof(value) - 1 > maximum_address - base - offset) {
    return false;
  }
  iovec local{&value, sizeof(value)};
  iovec remote{reinterpret_cast<void *>(base + offset), sizeof(value)};
  // Read only this process. A damaged SDK pointer becomes an invalid snapshot
  // instead of causing the diagnostic itself to dereference unmapped memory.
  return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == sizeof(value);
}

struct Snapshot {
  bool valid = false;
  bool first_node_valid = false;
  std::uint32_t node_count = 0;
  std::uint32_t verified = 0;
  void *execution_table = nullptr;
  void *first_node_operations = nullptr;
};

Snapshot snapshot(void *graph) {
  Snapshot result;
  std::uint32_t capacity = 0, first_index = 0;
  void *nodes = nullptr, *topology = nullptr, *first_node = nullptr;
  if (!read_field(graph, verified_offset, result.verified) ||
      !read_field(graph, execution_table_offset, result.execution_table) ||
      !read_field(graph, node_count_offset, result.node_count)) {
    return result;
  }
  result.valid = true;
  // Before Verify builds the topology, graph fields remain useful even when
  // the first topological node is unavailable. Keep their validity separate.
  if (!result.node_count || !read_field(graph, nodes_offset, nodes) ||
      !read_field(graph, node_capacity_offset, capacity) ||
      !read_field(graph, topology_offset, topology) || !read_field(topology, 0, first_index) ||
      first_index >= capacity || !read_field(nodes, first_index * sizeof(void *), first_node) ||
      !read_field(first_node, node_operations_offset, result.first_node_operations)) {
    return result;
  }
  result.first_node_valid = true;
  return result;
}

struct NodeKernelSnapshot {
  void *kernel = nullptr;
  bool pointer_valid = false;
  bool name_read_valid = false;
  bool name_terminated = false;
  // Escaping any byte as \\xHH needs at most four characters. Preserve a bounded
  // unterminated name, but mark it explicitly instead of reading past the field.
  char escaped_name[kernel_name_capacity * 4 + 1]{};
  std::int32_t enumeration = 0;
  bool enum_valid = false;
  std::uint32_t target_index = 0;
  bool target_index_valid = false;
};

NodeKernelSnapshot snapshot_node_kernel(void *node) {
  NodeKernelSnapshot result;
  result.target_index_valid = read_field(node, node_target_index_offset, result.target_index);
  result.pointer_valid = read_field(node, node_kernel_offset, result.kernel);
  if (!result.pointer_valid || !result.kernel) {
    return result;
  }
  result.enum_valid = read_field(result.kernel, kernel_enum_offset, result.enumeration);
  char raw_name[kernel_name_capacity]{};
  result.name_read_valid = read_field(result.kernel, kernel_name_offset, raw_name);
  if (!result.name_read_valid) {
    return result;
  }
  constexpr char digits[] = "0123456789abcdef";
  std::size_t output = 0;
  for (const unsigned char byte : raw_name) {
    if (!byte) {
      result.name_terminated = true;
      break;
    }
    if (byte == '\\' || byte == '\t' || byte == '\r' || byte == '\n') {
      result.escaped_name[output++] = '\\';
      result.escaped_name[output++] = byte == '\t'   ? 't'
                                      : byte == '\r' ? 'r'
                                      : byte == '\n' ? 'n'
                                                     : '\\';
    } else if (byte < 0x20 || byte >= 0x7f) {
      result.escaped_name[output++] = '\\';
      result.escaped_name[output++] = 'x';
      result.escaped_name[output++] = digits[byte >> 4];
      result.escaped_name[output++] = digits[byte & 15];
    } else {
      result.escaped_name[output++] = static_cast<char>(byte);
    }
  }
  return result;
}

void record_node_kernel(std::FILE *output, const NodeKernelSnapshot &kernel) {
  char pointer_text[32]{}, enum_text[24]{}, target_text[24]{};
  if (kernel.pointer_valid) {
    std::snprintf(pointer_text, sizeof(pointer_text), "%p", kernel.kernel);
  }
  if (kernel.enum_valid) {
    std::snprintf(enum_text, sizeof(enum_text), "%d", kernel.enumeration);
  }
  if (kernel.target_index_valid) {
    std::snprintf(target_text, sizeof(target_text), "%u", kernel.target_index);
  }
  // These are configured identities, not evidence that a backend executed the
  // node. Keep unreadable numeric fields blank rather than reporting zero.
  std::fprintf(
      output, "\t%d\t%s\t%d\t%s\t%s\t%d\t%s\t%d\t%s\n", kernel.pointer_valid, pointer_text,
      kernel.name_read_valid, kernel.name_read_valid ? (kernel.name_terminated ? "1" : "0") : "",
      kernel.escaped_name, kernel.enum_valid, enum_text, kernel.target_index_valid, target_text);
}

struct NodeOperationsSnapshot {
  std::int64_t node_index = -1;
  void *node = nullptr;
  void *operations = nullptr;
  std::uint32_t operation_count = 0;
  bool operation_pointer_valid = false;
  bool operation_count_valid = false;
  const char *condition = "topology_read_failed";
  NodeKernelSnapshot kernel;
};

NodeOperationsSnapshot snapshot_node_operations(void *nodes, void *topology, std::uint32_t position,
                                                std::uint32_t capacity) {
  NodeOperationsSnapshot result;
  std::uint32_t index = 0;
  if (!read_field(topology, static_cast<std::size_t>(position) * sizeof(index), index)) {
    return result;
  }
  result.node_index = index;
  if (index >= capacity) {
    result.condition = "node_index_out_of_bounds";
    return result;
  }
  if (!read_field(nodes, static_cast<std::size_t>(index) * sizeof(void *), result.node)) {
    result.condition = "node_read_failed";
    return result;
  }
  if (!result.node) {
    result.condition = "null_node";
    return result;
  }
  result.kernel = snapshot_node_kernel(result.node);
  if (!read_field(result.node, node_operations_offset, result.operations)) {
    result.condition = "operations_read_failed";
    return result;
  }
  result.operation_pointer_valid = true;
  if (!result.operations) {
    result.condition = "null_operations";
    return result;
  }
  if (!read_field(result.operations, operation_count_offset, result.operation_count)) {
    result.condition = "operation_count_read_failed";
    return result;
  }
  result.operation_count_valid = true;
  result.condition = result.operation_count ? "ok" : "zero_operations";
  return result;
}

void record_node_operations(void *graph) {
  auto &state = runtime();
  const auto started = std::chrono::steady_clock::now();
  const auto timestamp =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  std::uint32_t count = 0, capacity = 0;
  void *nodes = nullptr, *topology = nullptr;
  const char *condition = "complete";
  bool scan_allowed = false;
  if (!read_field(graph, node_count_offset, count) ||
      !read_field(graph, node_capacity_offset, capacity)) {
    condition = "graph_fields_read_failed";
  } else if (count > capacity) {
    condition = "node_count_exceeds_capacity";
  } else if (count > maximum_snapshot_nodes || capacity > maximum_snapshot_capacity) {
    condition = "diagnostic_limit_exceeded";
  } else if (!count) {
    condition = "empty_graph";
    scan_allowed = true;
  } else if (!read_field(graph, nodes_offset, nodes) ||
             !read_field(graph, topology_offset, topology) || !nodes || !topology) {
    condition = "graph_arrays_unavailable";
  } else {
    scan_allowed = true;
  }

  std::uint32_t scanned_nodes = 0, read_errors = 0, null_operations = 0, zero_operations = 0;
  std::uint64_t total_operations = 0;
  std::int64_t first_null_topology = -1, first_null_node_index = -1;
  flockfile(state.nodes_output);
  if (scan_allowed) {
    for (std::uint32_t position = 0; position < count; ++position) {
      const auto node = snapshot_node_operations(nodes, topology, position, capacity);
      ++scanned_nodes;
      if (!node.operation_pointer_valid || (node.operations && !node.operation_count_valid)) {
        ++read_errors;
      } else if (!node.operations) {
        ++null_operations;
        if (first_null_topology < 0) {
          first_null_topology = position;
          first_null_node_index = node.node_index;
        }
      } else {
        total_operations += node.operation_count;
        if (!node.operation_count) {
          ++zero_operations;
        }
      }
      std::fprintf(state.nodes_output,
                   "node\t%.6f\t%llu\t%p\t%u\t%u\t%u\t%lld\t%p\t%p\t%u\t%d\t%d\t%s"
                   "\t\t\t\t\t\t\t\t\t",
                   timestamp, static_cast<unsigned long long>(current_verify.sequence), graph,
                   count, capacity, position, static_cast<long long>(node.node_index), node.node,
                   node.operations, node.operation_count, node.operation_pointer_valid,
                   node.operation_count_valid, node.condition);
      record_node_kernel(state.nodes_output, node.kernel);
    }
  }
  const bool complete = scan_allowed && scanned_nodes == count && !read_errors;
  {
    std::lock_guard<std::mutex> lock(state.integrity_mutex);
    state.operation_integrity[graph] = complete && count && !null_operations && !zero_operations;
  }
  if (scan_allowed && read_errors) {
    condition = "incomplete_node_reads";
  }
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::fprintf(state.nodes_output,
               "summary\t%.6f\t%llu\t%p\t%u\t%u\t\t\t\t\t\t\t\t%s\t%u\t%u\t%u\t%u\t%llu"
               "\t%lld\t%lld\t%d\t%.9g\t\t\t\t\t\t\t\t\t\n",
               timestamp, static_cast<unsigned long long>(current_verify.sequence), graph, count,
               capacity, condition, scanned_nodes, read_errors, null_operations, zero_operations,
               static_cast<unsigned long long>(total_operations),
               static_cast<long long>(first_null_topology),
               static_cast<long long>(first_null_node_index), complete, elapsed);
  std::fflush(state.nodes_output);
  funlockfile(state.nodes_output);
}

void record(const char *event, const char *api, void *graph, bool has_status = false,
            int status = 0, double elapsed = 0, const void *argument = nullptr) {
  auto &state = runtime();
  const auto timestamp =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  const auto fields = snapshot(graph);
  char status_text[24]{};
  if (has_status) {
    std::snprintf(status_text, sizeof(status_text), "%d", status);
  }
  flockfile(state.output);
  std::fprintf(state.output, "%s\t%s\t%.9g\t%.6f\t%llu\t%p\t%s\t%d\t%u\t%p\t%p\t%p\t%u\t%d\n",
               event, api, elapsed, timestamp,
               static_cast<unsigned long long>(current_verify.sequence), graph, status_text,
               fields.valid, fields.verified, fields.execution_table, fields.first_node_operations,
               argument, fields.node_count, fields.first_node_valid);
  std::fflush(state.output);
  funlockfile(state.output);
}

template <typename Action>
int trace_status(const char *api, void *graph, void *argument, Action action) {
  record("begin", api, graph, false, 0, 0, argument);
  const auto started = std::chrono::steady_clock::now();
  const auto status = action();
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  record("end", api, graph, true, status, elapsed, argument);
  return status;
}
} // namespace

extern "C" int vxVerifyGraph(void *graph) {
  auto &state = runtime();
  {
    // A later model may reuse the same address; evidence from its predecessor is invalid.
    std::lock_guard<std::mutex> lock(state.integrity_mutex);
    state.operation_integrity.erase(graph);
  }
  const auto previous = current_verify;
  current_verify = {++next_sequence, graph};
  const auto status =
      trace_status("vxVerifyGraph", graph, nullptr, [&] { return state.calls.verify(graph); });
  current_verify = previous;
  return status;
}

extern "C" void vxoGraph_GenerateOperationTable(void *graph) {
  auto &state = runtime();
  record("begin", "vxoGraph_GenerateOperationTable", graph);
  record_node_operations(graph);
  const auto started = std::chrono::steady_clock::now();
  state.calls.operation_table(graph);
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  record("end", "vxoGraph_GenerateOperationTable", graph, false, 0, elapsed);
}

extern "C" int vxnneExecutionLayer_Create(void *graph, void **output) {
  auto &state = runtime();
  record("begin", "vxnneExecutionLayer_Create", graph);
  const auto started = std::chrono::steady_clock::now();
  const auto status = state.calls.create_layer(graph, output);
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  void *created = nullptr;
  read_field(output, 0, created);
  record("end", "vxnneExecutionLayer_Create", graph, true, status, elapsed, created);
  return status;
}

extern "C" int vxoGraph_InitializeAllNodeKernels(void *graph) {
  auto &state = runtime();
  return trace_status("vxoGraph_InitializeAllNodeKernels", graph, nullptr,
                      [&] { return state.calls.initialize_nodes(graph); });
}

extern "C" int vxoGraph_VerifyTiling(void *graph) {
  auto &state = runtime();
  return trace_status("vxoGraph_VerifyTiling", graph, nullptr,
                      [&] { return state.calls.verify_tiling(graph); });
}

extern "C" int vxnneExecutionLayer_GenerateCommands(void *context, void *layer) {
  auto &state = runtime();
  return trace_status("vxnneExecutionLayer_GenerateCommands", current_verify.graph, layer,
                      [&] { return state.calls.generate_commands(context, layer); });
}

extern "C" int vxoGraph_PrepareParamMemory(void *graph) {
  auto &state = runtime();
  return trace_status("vxoGraph_PrepareParamMemory", graph, nullptr,
                      [&] { return state.calls.prepare_memory(graph); });
}

namespace {
struct ProcessScope {
  std::uint64_t sequence = 0;
  void *graph = nullptr;
};

thread_local ProcessScope current_process;
std::atomic<std::uint64_t> next_process{0};

void record_execution(const char *api, int status, void *argument = nullptr, int command = -1,
                      int hardware = -1, int interface_status = -1) {
  // This stream records calls that actually execute, separately from Verify's configured targets.
  static std::FILE *output = [] {
    runtime(); // Check the pinned OpenVX build before reading any private layout.
    const std::string path = std::string(std::getenv("SPECFERRY_VERIFY_TRACE")) + "-execution.tsv";
    auto *file = std::fopen(path.c_str(), "wx");
    if (!file) {
      refuse_trace("cannot create execution trace output");
    }
    std::fputs("unix_seconds\tprocess_"
               "id\tgraph\tapi\tstatus\targument\tcommand\thardware\tinterface_status\n",
               file);
    return file;
  }();
  const auto timestamp =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  flockfile(output);
  std::fprintf(output, "%.6f\t%llu\t%p\t%s\t%d\t%p\t%d\t%d\t%d\n", timestamp,
               static_cast<unsigned long long>(current_process.sequence), current_process.graph,
               api, status, argument, command, hardware, interface_status);
  std::fflush(output);
  funlockfile(output);
}
} // namespace

extern "C" int vxProcessGraph(void *graph) {
  static const auto execute = reinterpret_cast<GraphStatus>(find_symbol("vxProcessGraph"));
  const auto previous = current_process;
  current_process = {++next_process, graph};
  record_execution("vxProcessGraph.begin", 0);
  // In this pinned build, a nonzero graph+0x124 branches to ProcessInternal even
  // when captured commands exist. Keep this distinct from merely naming a target.
  std::uint32_t internal = 1;
  if (read_field(graph, 0x124, internal)) {
    record_execution("process_internal_flag", static_cast<int>(internal));
  }
  const auto fields = snapshot(graph);
  void *capture = nullptr;
  if (fields.valid && read_field(fields.execution_table, 0x16280, capture)) {
    record_execution("compiled_capture", static_cast<int>(fields.verified), capture);
  }
  const auto *guard = std::getenv("SPECFERRY_VERIFY_REQUIRE_OPERATIONS");
  const bool permitted =
      !guard || std::strcmp(guard, "1") != 0 || runtime().operation_integrity[graph];
  // Opt-in protection for same-process rebuild probes: return a normal API error
  // before execution if compilation lost entries. The caller can release normally.
  const auto status = permitted ? execute(graph) : -1;
  if (!permitted) {
    record_execution("compile_integrity_guard", status);
  }
  record_execution("vxProcessGraph.end", status);
  current_process = previous;
  return status;
}

extern "C" int vxnneExecutionLayer_Execute(void *layer) {
  static const auto execute =
      reinterpret_cast<GraphStatus>(find_symbol("vxnneExecutionLayer_Execute"));
  const auto status = execute(layer);
  if (current_process.graph) {
    record_execution("vxnneExecutionLayer_Execute", status, layer);
  }
  return status;
}

extern "C" int vxnneOperation_ExecuteCommands(void *operation, void *commands) {
  static const auto execute =
      reinterpret_cast<GenerateCommands>(find_symbol("vxnneOperation_ExecuteCommands"));
  const auto status = execute(operation, commands);
  if (current_process.graph) {
    record_execution("vxnneOperation_ExecuteCommands", status, operation);
  }
  return status;
}

// Signature and the four leading interface fields follow installed HAL/gc_hal_base.h
// and HAL/shared/gc_hal_driver_shared.h. Never modify the request or returned status.
extern "C" int gcoOS_DeviceControl(void *os, std::uint32_t code, void *input,
                                   std::size_t input_size, void *output, std::size_t output_size) {
  using Control = int (*)(void *, std::uint32_t, void *, std::size_t, void *, std::size_t);
  static const auto control = reinterpret_cast<Control>(find_symbol("gcoOS_DeviceControl"));
  std::int32_t command = -1, hardware = -1, interface_status = -1;
  if (current_process.graph && input_size >= 16) {
    read_field(input, 0, command);
    read_field(input, 4, hardware);
  }
  const auto status = control(os, code, input, input_size, output, output_size);
  if (current_process.graph) {
    if (output_size >= 16) {
      read_field(output, 12, interface_status);
    }
    record_execution("gcoOS_DeviceControl", status, nullptr, command, hardware, interface_status);
  }
  return status;
}

extern "C" int vxoGraph_Submit(void *graph) {
  static const auto submit = reinterpret_cast<GraphStatus>(find_symbol("vxoGraph_Submit"));
  const auto status = submit(graph);
  if (current_process.graph) {
    record_execution("vxoGraph_Submit", status, graph);
  }
  return status;
}

// Diagnostic-only ABI confirmed against pinned libGAL's gcoVX_Replay wrapper:
// buffer pointer plus uint32 capture length, forwarded unchanged to ReplayCmd.
extern "C" int gcoVX_Replay(void *buffer, std::uint32_t bytes) {
  using Replay = int (*)(void *, std::uint32_t);
  static const auto replay = reinterpret_cast<Replay>(find_symbol("gcoVX_Replay"));
  const auto status = replay(buffer, bytes);
  if (current_process.graph) {
    record_execution("gcoVX_Replay", status, buffer, static_cast<int>(bytes));
  }
  return status;
}

extern "C" int gcoOS_WaitSignal(void *os, void *signal, std::uint32_t timeout) {
  using WaitSignal = int (*)(void *, void *, std::uint32_t);
  static const auto wait = reinterpret_cast<WaitSignal>(find_symbol("gcoOS_WaitSignal"));
  const auto status = wait(os, signal, timeout);
  if (current_process.graph) {
    record_execution("gcoOS_WaitSignal", status, signal);
  }
  return status;
}
