#include "models/opt/decoder.hpp"
#include "models/opt/layer.hpp"
#include "np101/diagnostics.hpp"
#include "np101/kv_cache.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

namespace {
class Projection : public GraphBuilder {
public:
  Tensor hidden, query, key, value;

  Projection(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
             TensorBinding input)
      : GraphBuilder(context, 128, 96) {
    hidden = bind(input, config.hidden_spec());
    const auto slot = config.kv_spec().slot().shape;
    const auto projected = build_projections(*this, weights, config, layer, hidden);
    query = reshape(projected.query, slot);
    key = reshape(projected.key, slot);
    value = reshape(projected.value, slot);
    compile({hidden}, {query, key, value});
  }
};

class DecoderTail : public GraphBuilder {
public:
  Tensor valid_length, output;
  std::map<std::string, Tensor> outputs;

  DecoderTail(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
              Projection &producer, KvCache &cache)
      : GraphBuilder(context, 128, 96) {
    auto hidden = share(producer, producer.hidden);
    auto query = share(producer, producer.query);
    Tensor keys{cache.retain_keys(graph), config.kv_spec().tensor()};
    Tensor values{cache.retain_values(graph), keys.spec};
    valid_length = tensor({DataType::Int32, {1}});
    auto attention = attention_core(*this, query, keys, values, valid_length, 1.0f);
    outputs = build_decoder_tail(*this, weights, config, layer, hidden,
                                 reshape(attention.attended, config.hidden_spec().shape));
    outputs.emplace("probabilities", attention.probabilities);
    output = outputs.at("output");
    std::vector<Tensor> snapshots;
    for (const auto &entry : outputs) {
      snapshots.push_back(entry.second);
    }
    compile({hidden, query, keys, values, valid_length}, snapshots);
  }
};

struct Layer {
  Projection producer;
  KvCache cache;
  DecoderTail tail;
  const std::string qkv_label, cache_label, tail_label;

  Layer(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
        TensorBinding input)
      : producer(context, weights, config, layer, input),
        cache(context, producer.graph, producer.key.id, producer.value.id, config.kv_spec()),
        tail(context, weights, config, layer, producer, cache),
        qkv_label(config.prefix(layer) + "qkv"), cache_label(config.prefix(layer) + "kv"),
        tail_label(config.prefix(layer) + "attention_ffn") {}

  void step(unsigned position) {
    {
      TimingLabel component(TimingField::Component, qkv_label);
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(producer.graph.get()); }),
            "OPT Q/K/V projections");
    }
    {
      TimingLabel component(TimingField::Component, cache_label);
      cache.write(position);
    }
    {
      TimingLabel component(TimingField::Component, tail_label);
      const auto length = static_cast<std::int32_t>(position + 1);
      std::vector<std::uint8_t> bytes(sizeof(length));
      std::memcpy(bytes.data(), &length, sizeof(length));
      upload_tensor(tail.graph, tail.valid_length.id, bytes);
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(tail.graph.get()); }),
            "OPT attention and feed-forward");
    }
  }

  std::vector<std::uint8_t> read(const std::string &name) {
    if (name == "keys") {
      return cache.read_keys();
    }
    if (name == "values") {
      return cache.read_values();
    }
    if (name == "query") {
      return read_tensor(producer.graph, producer.query.id);
    }
    if (name == "key") {
      return read_tensor(producer.graph, producer.key.id);
    }
    if (name == "value") {
      return read_tensor(producer.graph, producer.value.id);
    }
    return read_tensor(tail.graph, tail.outputs.at(name).id);
  }

  void close() {
    tail.graph.close();
    cache.close();
    producer.graph.close();
  }
};
} // namespace

struct DecoderSlice::Impl {
  Config config;
  Graph storage;
  vsi_nn_tensor_id_t input;
  bool bound_input;
  std::vector<unsigned> selected;
  std::vector<std::unique_ptr<Layer>> layers;
  unsigned completed = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights, const Config &spec,
       const std::vector<unsigned> &indices, const TensorBinding *external = nullptr)
      : config(spec), storage(context, 1, 0), input(VSI_NN_TENSOR_ID_NA), bound_input(external),
        selected(indices) {
    if (!external) {
      input = add_tensor(storage, config.hidden_spec(), false,
                         std::vector<std::uint8_t>(config.hidden_spec().bytes()));
    }
    try {
      for (auto layer : selected) {
        TimingLabel component(TimingField::Component, config.prefix(layer) + "initialize");
        const auto first = external ? *external : TensorBinding{storage, input};
        const auto binding = layers.empty() ? first
                                            : TensorBinding{layers.back()->tail.graph,
                                                            layers.back()->tail.output.id};
        layers.push_back(std::make_unique<Layer>(context, weights, config, layer, binding));
      }
    } catch (...) {
      while (!layers.empty()) {
        layers.pop_back();
      }
      throw;
    }
  }

  ~Impl() {
    while (!layers.empty()) {
      layers.pop_back();
    }
  }
};

DecoderSlice::DecoderSlice(Context &context, const WeightStore &weights, const Config &config,
                           const std::vector<unsigned> &layers) {
  validate_weights(weights, config, layers);
  impl_ = std::make_unique<Impl>(context, weights, config, layers);
}

DecoderSlice::~DecoderSlice() = default;

DecoderSlice::DecoderSlice(Context &context, const WeightStore &weights, const Config &config,
                           const std::vector<unsigned> &layers, TensorBinding input) {
  validate_weights(weights, config, layers);
  impl_ = std::make_unique<Impl>(context, weights, config, layers, &input);
}

void DecoderSlice::step(const std::vector<std::uint8_t> &hidden) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("OPT slice is closed or failed");
  }
  if (impl_->bound_input) {
    throw std::logic_error("use step_bound for an externally bound OPT slice");
  }
  if (hidden.size() != impl_->config.hidden_spec().bytes()) {
    throw std::invalid_argument("OPT input size mismatch");
  }
  if (impl_->completed >= impl_->config.capacity) {
    throw std::out_of_range("OPT KV capacity exhausted");
  }
  impl_->failed = true;
  upload_tensor(impl_->storage, impl_->input, hidden);
  for (auto &layer : impl_->layers) {
    layer->step(impl_->completed);
  }
  ++impl_->completed;
  impl_->failed = false;
}

void DecoderSlice::step_bound() {
  if (!impl_ || impl_->failed || !impl_->bound_input) {
    throw std::logic_error("OPT slice has no valid external input binding");
  }
  if (impl_->completed >= impl_->config.capacity) {
    throw std::out_of_range("OPT KV capacity exhausted");
  }
  impl_->failed = true;
  for (auto &layer : impl_->layers) {
    layer->step(impl_->completed);
  }
  ++impl_->completed;
  impl_->failed = false;
}

TensorBinding DecoderSlice::output_binding() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("OPT slice is closed or failed");
  }
  auto &tail = impl_->layers.back()->tail;
  return {tail.graph, tail.output.id};
}

void DecoderSlice::reset() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("OPT slice is closed or failed");
  }
  impl_->completed = 0;
}

unsigned DecoderSlice::length() const { return impl_ ? impl_->completed : 0; }

std::size_t DecoderSlice::cache_writes() const {
  std::size_t count = 0;
  if (impl_) {
    for (const auto &layer : impl_->layers) {
      count += layer->cache.writes();
    }
  }
  return count;
}

std::size_t DecoderSlice::cache_revalidations() const {
  std::size_t count = 0;
  if (impl_) {
    for (const auto &layer : impl_->layers) {
      count += layer->cache.revalidations();
    }
  }
  return count;
}

double DecoderSlice::cache_write_seconds() const {
  double seconds = 0;
  if (impl_) {
    for (const auto &layer : impl_->layers) {
      seconds += layer->cache.write_seconds();
    }
  }
  return seconds;
}

double DecoderSlice::cache_revalidation_seconds() const {
  double seconds = 0;
  if (impl_) {
    for (const auto &layer : impl_->layers) {
      seconds += layer->cache.revalidation_seconds();
    }
  }
  return seconds;
}

std::vector<std::uint8_t> DecoderSlice::read(unsigned layer, const std::string &name) {
  if (!impl_ || impl_->failed || !impl_->completed) {
    throw std::logic_error("OPT slice has no valid output");
  }
  const auto found = std::find(impl_->selected.begin(), impl_->selected.end(), layer);
  if (found == impl_->selected.end()) {
    throw std::out_of_range("layer not selected");
  }
  return impl_->layers.at(found - impl_->selected.begin())->read(name);
}

void DecoderSlice::close() {
  if (impl_) {
    while (!impl_->layers.empty()) {
      impl_->layers.back()->close();
      impl_->layers.pop_back();
    }
    impl_->storage.close();
    impl_.reset();
  }
}
} // namespace specferry::models::opt
