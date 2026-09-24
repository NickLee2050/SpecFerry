#pragma once

#include "np101/memory_budget.hpp"
#include "vsi_nn_pub.h"

#include <map>

namespace specferry::np101 {
void check(vsi_status status, const char *operation);

// The graph must be released before its context. Neither handle is copyable.
class Context {
public:
  Context();

  ~Context();

  Context(const Context &) = delete;

  Context &operator=(const Context &) = delete;

  vsi_nn_context_t get() const { return handle_; }

  void close();

  MemoryBudget &memory() { return memory_; }

  void save_memory_report() const;

private:
  MemoryBudget memory_;
  vsi_nn_context_t handle_ = nullptr;
};

class Graph {
public:
  explicit Graph(Context &context, unsigned tensors, unsigned nodes);

  ~Graph();

  Graph(const Graph &) = delete;

  Graph &operator=(const Graph &) = delete;

  vsi_nn_graph_t *get() const { return handle_; }

  void close();

  MemoryBudget::Lease reserve(bool constant, std::size_t bytes);
  void record_storage(vsi_nn_tensor_id_t id, MemoryBudget::Lease lease);
  void alias_storage(vsi_nn_tensor_id_t id, const Graph &owner, vsi_nn_tensor_id_t source);
  void save_memory_report() const;

private:
  Context &context_;
  std::map<vsi_nn_tensor_id_t, MemoryBudget::Lease> allocations_;
  vsi_nn_graph_t *handle_ = nullptr;
};
} // namespace specferry::np101
