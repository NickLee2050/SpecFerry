#include "np101/context.hpp"
#include "np101/diagnostics.hpp"

#include <stdexcept>
#include <string>

namespace specferry::np101 {
void check(vsi_status status, const char *operation) {
  if (status != VSI_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed: SDK status=" + std::to_string(status));
  }
}

Context::Context()
    : handle_(sdk_call("vsi_nn_CreateContext", [] { return vsi_nn_CreateContext(); })) {
  if (!handle_) {
    throw std::runtime_error("vsi_nn_CreateContext returned null");
  }
  try {
    check(vxGetStatus(reinterpret_cast<vx_reference>(handle_->c)), "vxGetStatus(context)");
  } catch (...) {
    sdk_call("vsi_nn_ReleaseContext", [&] { vsi_nn_ReleaseContext(&handle_); });
    throw;
  }
}

Context::~Context() {
  if (handle_) {
    vsi_nn_ReleaseContext(&handle_);
  }
}

void Context::close() {
  if (handle_) {
    sdk_call("vsi_nn_ReleaseContext", [&] { vsi_nn_ReleaseContext(&handle_); });
  }
  if (handle_) {
    throw std::runtime_error("vsi_nn_ReleaseContext did not clear handle");
  }
}

Graph::Graph(Context &context, unsigned tensors, unsigned nodes)
    : handle_(sdk_call("vsi_nn_CreateGraph",
                       [&] { return vsi_nn_CreateGraph(context.get(), tensors, nodes); })) {
  if (!handle_) {
    throw std::runtime_error("vsi_nn_CreateGraph returned null");
  }
}

Graph::~Graph() {
  if (handle_) {
    vsi_nn_ReleaseGraph(&handle_);
  }
}

void Graph::close() {
  if (handle_) {
    sdk_call("vsi_nn_ReleaseGraph", [&] { vsi_nn_ReleaseGraph(&handle_); });
  }
  if (handle_) {
    throw std::runtime_error("vsi_nn_ReleaseGraph did not clear handle");
  }
}
} // namespace specferry::np101
