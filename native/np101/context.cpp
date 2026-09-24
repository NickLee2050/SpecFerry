#include "np101/context.hpp"
#include "np101/diagnostics.hpp"

#include <stdexcept>
#include <string>
#include <utility>

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
    : context_(context), handle_(sdk_call("vsi_nn_CreateGraph", [&] {
        return vsi_nn_CreateGraph(context.get(), tensors, nodes);
      })) {
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
  allocations_.clear();
}

MemoryBudget::Lease Graph::reserve(bool constant, std::size_t bytes) {
  return context_.memory().reserve(constant, bytes);
}

void Graph::record_storage(vsi_nn_tensor_id_t id, MemoryBudget::Lease lease) {
  allocations_.emplace(id, std::move(lease));
}

void Graph::alias_storage(vsi_nn_tensor_id_t id, const Graph &owner, vsi_nn_tensor_id_t source) {
  allocations_.at(id) = owner.allocations_.at(source);
}

} // namespace specferry::np101
