#include "np101/kv_cache.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"

#include <array>
#include <chrono>
#include <stdexcept>

namespace specferry::np101 {
namespace {
struct TensorView {
  vx_tensor tensor = nullptr;

  ~TensorView() {
    if (tensor) {
      vxReleaseTensor(&tensor);
    }
  }
};

void expose_output(Graph &graph, vx_node node) {
  auto parameter = vxGetParameterByIndex(node, 1);
  if (!parameter) {
    throw std::runtime_error("copy node has no destination parameter");
  }
  auto status = vxAddParameterToGraph(graph.get()->g, parameter);
  vxReleaseParameter(&parameter);
  check(status, "expose KV copy destination");
}

vx_tensor source_tensor(Graph &producer, vsi_nn_tensor_id_t id) {
  auto *tensor = vsi_nn_GetTensor(producer.get(), id);
  if (!tensor || !tensor->t || tensor->attr.vtl || tensor->attr.is_const ||
      tensor->attr.is_created_from_handle || tensor->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16 ||
      tensor->attr.dim_num != 3 || tensor->attr.size[0] != 256 || tensor->attr.size[1] != 1 ||
      tensor->attr.size[2] != 2) {
    throw std::invalid_argument("KV producer must expose ordinary FP16 [2, 1, 256] tensors");
  }
  return tensor->t;
}
} // namespace

struct KvCache::Impl {
  unsigned capacity;
  Graph storage;
  vsi_nn_tensor_id_t keys;
  vsi_nn_tensor_id_t values;
  // Destruction order: copy graph, views, then parent storage.
  std::vector<std::unique_ptr<TensorView>> key_views;
  std::vector<std::unique_ptr<TensorView>> value_views;
  Graph copy;
  std::size_t completed_writes = 0;
  std::size_t revalidations = 0;
  double write_seconds = 0;
  double revalidation_seconds = 0;

  Impl(Context &context, Graph &producer, vsi_nn_tensor_id_t key, vsi_nn_tensor_id_t value,
       unsigned requested_capacity)
      : capacity(requested_capacity), storage(context, 2, 0), copy(context, 0, 0) {
    if (capacity < 1 || capacity > maximum_capacity || producer.get()->ctx != context.get()) {
      throw std::invalid_argument("KV cache requires one context and capacity in [1, 512]");
    }
    auto source_key = source_tensor(producer, key);
    auto source_value = source_tensor(producer, value);
    const TensorSpec spec{DataType::Float16, {256, capacity, 2}};
    const std::vector<std::uint8_t> zeros(spec.bytes());
    keys = add_tensor(storage, spec, false, zeros);
    values = add_tensor(storage, spec, false, zeros);
    for (unsigned position = 0; position < capacity; ++position) {
      key_views.push_back(view(keys, position));
      value_views.push_back(view(values, position));
    }

    // Only the new token is copied. Views alias the parent allocation; this
    // intentionally bypasses whole-cache SCATTER_ND_UPDATE outputs.
    auto key_copy = vxTensorCopyNode(copy.get()->g, source_key, key_views[0]->tensor);
    if (!key_copy) {
      throw std::runtime_error("cannot create key-to-slot copy node");
    }
    auto value_copy = vxTensorCopyNode(copy.get()->g, source_value, value_views[0]->tensor);
    if (!value_copy) {
      vxReleaseNode(&key_copy);
      throw std::runtime_error("cannot create value-to-slot copy node");
    }
    try {
      expose_output(copy, key_copy);
      expose_output(copy, value_copy);
    } catch (...) {
      vxReleaseNode(&value_copy);
      vxReleaseNode(&key_copy);
      throw;
    }
    vxReleaseNode(&value_copy);
    vxReleaseNode(&key_copy);
    check(vxVerifyGraph(copy.get()->g), "verify KV slot-copy graph");
  }

  std::unique_ptr<TensorView> view(vsi_nn_tensor_id_t parent, unsigned position) {
    std::array<vsi_size_t, 3> start{0, position, 0};
    std::array<vsi_size_t, 3> end{256, position + 1, 2};
    auto result = std::make_unique<TensorView>();
    result->tensor = vsi_nn_CreateViewTensor(storage.get(), start.data(), end.data(),
                                             vsi_nn_GetTensor(storage.get(), parent));
    check(vxGetStatus(reinterpret_cast<vx_reference>(result->tensor)), "create KV slot view");
    return result;
  }
};

KvCache::KvCache(Context &context, Graph &producer, vsi_nn_tensor_id_t key,
                 vsi_nn_tensor_id_t value, unsigned capacity)
    : impl_(std::make_unique<Impl>(context, producer, key, value, capacity)) {}

KvCache::~KvCache() = default;

void KvCache::write(unsigned position) {
  if (!impl_ || position >= impl_->capacity) {
    throw std::out_of_range("KV write position exceeds capacity or cache is closed");
  }
  auto graph = impl_->copy.get()->g;
  const auto start = std::chrono::steady_clock::now();
  check(vxSetGraphParameterByIndex(
            graph, 0, reinterpret_cast<vx_reference>(impl_->key_views[position]->tensor)),
        "select key slot");
  check(vxSetGraphParameterByIndex(
            graph, 1, reinterpret_cast<vx_reference>(impl_->value_views[position]->tensor)),
        "select value slot");
  if (!vxIsGraphVerified(graph)) {
    const auto verify_start = std::chrono::steady_clock::now();
    ++impl_->revalidations;
    check(vxVerifyGraph(graph), "reverify KV slot-copy graph");
    impl_->revalidation_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - verify_start).count();
  }
  check(vxProcessGraph(graph), "write KV slot");
  ++impl_->completed_writes;
  impl_->write_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

vsi_nn_tensor_id_t KvCache::retain_keys(Graph &reader) {
  if (!impl_) {
    throw std::logic_error("closed KV cache");
  }
  return retain_tensor(impl_->storage, impl_->keys, reader);
}

vsi_nn_tensor_id_t KvCache::retain_values(Graph &reader) {
  if (!impl_) {
    throw std::logic_error("closed KV cache");
  }
  return retain_tensor(impl_->storage, impl_->values, reader);
}

std::vector<std::uint8_t> KvCache::read_keys() {
  if (!impl_) {
    throw std::logic_error("closed KV cache");
  }
  return read_tensor(impl_->storage, impl_->keys);
}

std::vector<std::uint8_t> KvCache::read_values() {
  if (!impl_) {
    throw std::logic_error("closed KV cache");
  }
  return read_tensor(impl_->storage, impl_->values);
}

unsigned KvCache::capacity() const { return impl_ ? impl_->capacity : 0; }

std::size_t KvCache::writes() const { return impl_ ? impl_->completed_writes : 0; }

std::size_t KvCache::revalidations() const { return impl_ ? impl_->revalidations : 0; }

double KvCache::write_seconds() const { return impl_ ? impl_->write_seconds : 0; }

double KvCache::revalidation_seconds() const { return impl_ ? impl_->revalidation_seconds : 0; }

void KvCache::close() {
  if (impl_) {
    impl_->copy.close();
    impl_->key_views.clear();
    impl_->value_views.clear();
    impl_->storage.close();
    impl_.reset();
  }
}
} // namespace specferry::np101
