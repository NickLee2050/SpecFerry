#pragma once

#include "vsi_nn_pub.h"
#include <stdexcept>
#include <string>

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

private:
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

private:
  vsi_nn_graph_t *handle_ = nullptr;
};
} // namespace specferry::np101
