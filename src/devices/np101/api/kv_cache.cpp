#include "devices/np101/api/diagnostics.hpp"
#include "devices/np101/api/kv_cache.hpp"
#include "devices/np101/api/tensor.hpp"
#include "devices/np101/api/tensor_spec.hpp"

#include <array>
#include <optional>
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

vx_tensor source_tensor(Graph &producer, vsi_nn_tensor_id_t id, const KvSpec &spec) {
  auto *tensor = vsi_nn_GetTensor(producer.get(), id);
  if (!tensor || !tensor->t || tensor->attr.vtl || tensor->attr.is_const ||
      tensor->attr.is_created_from_handle || tensor->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16 ||
      tensor->attr.dim_num != 3 || tensor->attr.size[0] != spec.head_dim ||
      tensor->attr.size[1] != 1 || tensor->attr.size[2] != spec.heads) {
    throw std::invalid_argument("KV producer shape/dtype differs from the configured slot");
  }
  return tensor->t;
}
} // namespace

struct KvCache::Impl {
  KvSpec spec;
  unsigned capacity;
  Graph storage;
  vsi_nn_tensor_id_t keys;
  vsi_nn_tensor_id_t values;
  // Destruction order: copy graph, views, then parent storage.
  std::vector<std::unique_ptr<TensorView>> key_views;
  std::vector<std::unique_ptr<TensorView>> value_views;
  std::unique_ptr<Graph> copy;
  KvWriteMode mode;
  Graph &producer;
  vsi_nn_tensor_id_t source_key_id, source_value_id;
  std::vector<std::uint8_t> key_buffer, value_buffer;
  std::optional<unsigned> selected_position = 0;
  std::size_t completed_writes = 0;
  std::size_t revalidations = 0;

  Impl(Context &context, Graph &producer, vsi_nn_tensor_id_t key, vsi_nn_tensor_id_t value,
       KvSpec requested_spec, KvWriteMode write_mode)
      : spec(requested_spec), capacity(spec.capacity), storage(context, 2, 0), mode(write_mode),
        producer(producer), source_key_id(key), source_value_id(value) {
    spec.validate();
    if (mode != KvWriteMode::CopyGraph && mode != KvWriteMode::MappedHost) {
      throw std::invalid_argument("unknown KV write mode");
    }
    if (producer.get()->ctx != context.get()) {
      throw std::invalid_argument("KV cache requires one context");
    }
    auto source_key = source_tensor(producer, key, spec);
    auto source_value = source_tensor(producer, value, spec);
    const auto tensor_spec = spec.tensor();
    const std::vector<std::uint8_t> zeros(tensor_spec.bytes());
    keys = add_tensor(storage, tensor_spec, false, zeros);
    values = add_tensor(storage, tensor_spec, false, zeros);
    if (mode == KvWriteMode::MappedHost) {
      key_buffer.resize(spec.slot().bytes());
      value_buffer.resize(spec.slot().bytes());
      return;
    }
    copy = std::make_unique<Graph>(context, 0, 0);
    for (unsigned position = 0; position < capacity; ++position) {
      key_views.push_back(view(keys, position));
      value_views.push_back(view(values, position));
    }

    // Only the new token is copied. Views alias the parent allocation; this
    // intentionally bypasses whole-cache SCATTER_ND_UPDATE outputs.
    auto key_copy = vxTensorCopyNode(copy->get()->g, source_key, key_views[0]->tensor);
    if (!key_copy) {
      throw std::runtime_error("cannot create key-to-slot copy node");
    }
    auto value_copy = vxTensorCopyNode(copy->get()->g, source_value, value_views[0]->tensor);
    if (!value_copy) {
      vxReleaseNode(&key_copy);
      throw std::runtime_error("cannot create value-to-slot copy node");
    }
    try {
      expose_output(*copy, key_copy);
      expose_output(*copy, value_copy);
    } catch (...) {
      vxReleaseNode(&value_copy);
      vxReleaseNode(&key_copy);
      throw;
    }
    vxReleaseNode(&value_copy);
    vxReleaseNode(&key_copy);
    check(sdk_call("vxVerifyGraph", [&] { return vxVerifyGraph(copy->get()->g); }),
          "verify KV slot-copy graph");
  }

  std::unique_ptr<TensorView> view(vsi_nn_tensor_id_t parent, unsigned position) {
    std::array<vsi_size_t, 3> start{0, position, 0};
    std::array<vsi_size_t, 3> end{spec.head_dim, position + 1, spec.heads};
    auto result = std::make_unique<TensorView>();
    result->tensor = vsi_nn_CreateViewTensor(storage.get(), start.data(), end.data(),
                                             vsi_nn_GetTensor(storage.get(), parent));
    check(vxGetStatus(reinterpret_cast<vx_reference>(result->tensor)), "create KV slot view");
    return result;
  }
};

KvCache::KvCache(Context &context, Graph &producer, vsi_nn_tensor_id_t key,
                 vsi_nn_tensor_id_t value, KvSpec spec, KvWriteMode mode)
    : impl_(std::make_unique<Impl>(context, producer, key, value, spec, mode)) {}

KvCache::~KvCache() = default;

void KvCache::write(unsigned position) {
  if (!impl_ || position >= impl_->capacity) {
    throw std::out_of_range("KV write position exceeds capacity or cache is closed");
  }
  if (impl_->mode == KvWriteMode::MappedHost) {
    // The synchronous producer has completed. Mapping/unmapping is the host
    // access boundary; readers retain the same parent tensors for their lifetime.
    const auto &spec = impl_->spec;
    const std::vector<std::uint32_t> start{0, 0, 0}, end{spec.head_dim, 1, spec.heads};
    const std::vector<std::uint32_t> slot_start{0, position, 0};
    const std::vector<std::uint32_t> slot_end{spec.head_dim, position + 1, spec.heads};
    read_tensor_region_into(impl_->producer, impl_->source_key_id, start, end, impl_->key_buffer);
    read_tensor_region_into(impl_->producer, impl_->source_value_id, start, end,
                            impl_->value_buffer);
    upload_tensor_region(impl_->storage, impl_->keys, slot_start, slot_end, impl_->key_buffer);
    upload_tensor_region(impl_->storage, impl_->values, slot_start, slot_end, impl_->value_buffer);
    ++impl_->completed_writes;
    return;
  }
  auto graph = impl_->copy->get()->g;
  // Construction already binds slot zero. Rebinding identical destinations can
  // invalidate the verified graph before the first token, without changing data.
  if (impl_->selected_position != position) {
    impl_->selected_position.reset();
    check(sdk_call("vxSetGraphParameterByIndex.key",
                   [&] {
                     return vxSetGraphParameterByIndex(
                         graph, 0,
                         reinterpret_cast<vx_reference>(impl_->key_views[position]->tensor));
                   }),
          "select key slot");
    check(sdk_call("vxSetGraphParameterByIndex.value",
                   [&] {
                     return vxSetGraphParameterByIndex(
                         graph, 1,
                         reinterpret_cast<vx_reference>(impl_->value_views[position]->tensor));
                   }),
          "select value slot");
    impl_->selected_position = position;
  }
  if (!vxIsGraphVerified(graph)) {
    ++impl_->revalidations;
    check(sdk_call("vxVerifyGraph", [&] { return vxVerifyGraph(graph); }),
          "reverify KV slot-copy graph");
  }
  check(sdk_call("vxProcessGraph", [&] { return vxProcessGraph(graph); }), "write KV slot");
  ++impl_->completed_writes;
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

std::vector<std::uint8_t> KvCache::read_keys(TensorReadback method) {
  if (!impl_) {
    throw std::logic_error("closed KV cache");
  }
  return read_tensor(impl_->storage, impl_->keys, method);
}

std::vector<std::uint8_t> KvCache::read_values(TensorReadback method) {
  if (!impl_) {
    throw std::logic_error("closed KV cache");
  }
  return read_tensor(impl_->storage, impl_->values, method);
}

KvSpec KvCache::spec() const {
  if (!impl_) {
    throw std::logic_error("closed KV cache");
  }
  return impl_->spec;
}

unsigned KvCache::capacity() const { return impl_ ? impl_->capacity : 0; }

std::size_t KvCache::writes() const { return impl_ ? impl_->completed_writes : 0; }

std::size_t KvCache::revalidations() const { return impl_ ? impl_->revalidations : 0; }

void KvCache::close() {
  if (impl_) {
    if (impl_->copy) {
      impl_->copy->close();
      impl_->copy.reset();
    }
    impl_->key_views.clear();
    impl_->value_views.clear();
    impl_->storage.close();
    impl_.reset();
  }
}
} // namespace specferry::np101
