#include "models/opt/decoder.hpp"
#include "models/opt/model.hpp"
#include "np101/diagnostics.hpp"
#include "np101/tensor.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

InputEmbedding::InputEmbedding(Context &context, const WeightStore &weights,
                               const ModelConfig &config, Vocabulary &table)
    : GraphBuilder(context, 12, 4), config_(config), table_(table) {
  config.validate();
  if (table.width() != config.embedding || table.rows() != config.vocabulary) {
    throw std::invalid_argument("OPT embedding table dimensions disagree");
  }
  auto embedding = bind(table.embedding(), {DataType::Float16, {config.embedding, 1}});
  position_ = tensor({DataType::Int32, {1}});
  const auto &record = weights.find("decoder.embed_positions.weight");
  auto positions = weight(weights, record, record.spec);
  auto projected = project(embedding, weights, weights.find("decoder.project_in.weight"));
  output_ = binary(VSI_NN_OP_ADD, projected, gather(positions, position_, 1));
  compile({embedding, position_}, {output_});
}

void InputEmbedding::run(std::int32_t token, unsigned position) {
  TimingLabel component(TimingField::Component, "embedding.projection");
  config_.validate_token(token);
  if (position >= config_.decoder.capacity) {
    throw std::out_of_range("OPT position exceeds cache capacity");
  }
  table_.lookup(token);
  const auto index = static_cast<std::int32_t>(position + config_.position_offset);
  std::vector<std::uint8_t> bytes(sizeof(index));
  std::memcpy(bytes.data(), &index, sizeof(index));
  upload_tensor(graph, position_.id, bytes);
  check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.get()); }),
        "OPT embedding projection and learned position");
}

TensorBinding InputEmbedding::binding() { return {graph, output_.id}; }

std::vector<std::uint8_t> InputEmbedding::read() { return read_tensor(graph, output_.id); }

OutputProjection::OutputProjection(Context &context, const WeightStore &weights,
                                   const ModelConfig &config, TensorBinding hidden)
    : GraphBuilder(context, 3, 1) {
  auto input = bind(hidden, config.decoder.hidden_spec());
  output_ = project(input, weights, weights.find("decoder.project_out.weight"));
  // OPT-350M is post-norm: the last layer already normalized its output.
  compile({input}, {output_});
}

void OutputProjection::run() {
  TimingLabel component(TimingField::Component, "output_projection");
  check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.get()); }),
        "OPT output projection");
}

TensorBinding OutputProjection::binding() { return {graph, output_.id}; }

struct Model::Impl {
  ModelConfig config;
  unsigned layer_count;
  bool failed = false, prediction_valid = false;
  Vocabulary table;
  InputEmbedding input;
  DecoderSlice decoder;
  OutputProjection output;
  VocabularyHead head;

  static std::vector<unsigned> indices(unsigned count) {
    std::vector<unsigned> result(count);
    std::iota(result.begin(), result.end(), 0);
    return result;
  }

  Impl(Context &context, const WeightStore &weights, const ModelConfig &spec, unsigned layers,
       SamplingOptions sampling)
      : config(spec), layer_count(layers),
        table(context, weights, weights.find("decoder.embed_tokens.weight"), spec.block_rows),
        input(context, weights, spec, table),
        decoder(context, weights, spec.decoder, indices(layers), input.binding()),
        output(context, weights, spec, decoder.output_binding()),
        head(context, table, output.binding(), sampling) {}
};

Model::Model(Context &context, const WeightStore &weights, const ModelConfig &config,
             unsigned layers, SamplingOptions sampling) {
  validate_model_weights(weights, config, layers);
  impl_ = std::make_unique<Impl>(context, weights, config, layers, sampling);
}

Model::~Model() = default;

void Model::require_live() const {
  if (!impl_ || impl_->failed) {
    throw std::logic_error(
        "OPT model is closed or failed; recreate after partial execution failure");
  }
}

void Model::consume(std::int32_t token) {
  require_live();
  impl_->config.validate_token(token);
  if (length() >= impl_->config.decoder.capacity) {
    throw std::out_of_range("OPT KV capacity exhausted");
  }
  impl_->failed = true;
  impl_->prediction_valid = false;
  impl_->input.run(token, length());
  impl_->decoder.step_bound();
  impl_->failed = false;
}

std::int32_t Model::predict() {
  require_live();
  if (!length()) {
    throw std::logic_error("OPT prediction requires a consumed token");
  }
  impl_->failed = true;
  impl_->output.run();
  auto token = impl_->head.select();
  impl_->prediction_valid = true;
  impl_->failed = false;
  return token;
}

void Model::reset() {
  require_live();
  impl_->decoder.reset();
  impl_->head.reset();
  impl_->prediction_valid = false;
}

unsigned Model::length() const { return impl_ ? impl_->decoder.length() : 0; }

std::size_t Model::cache_writes() const { return impl_ ? impl_->decoder.cache_writes() : 0; }

double Model::cache_write_seconds() const {
  return impl_ ? impl_->decoder.cache_write_seconds() : 0;
}

double Model::cache_revalidation_seconds() const {
  return impl_ ? impl_->decoder.cache_revalidation_seconds() : 0;
}

std::vector<std::uint8_t> Model::read(const std::string &name, unsigned layer) {
  require_live();
  if (!length() || ((name == "logits" || name == "projected") && !impl_->prediction_valid)) {
    throw std::logic_error("OPT diagnostic output is not valid");
  }
  if (name == "embedding") {
    return impl_->input.read();
  }
  if (name == "projected") {
    auto source = impl_->output.binding();
    return read_tensor(source.owner, source.id);
  }
  if (name == "logits") {
    return impl_->head.read_logits();
  }
  return impl_->decoder.read(layer, name);
}

Generation Model::generate(const std::vector<std::int32_t> &prompt, unsigned maximum_new_tokens,
                           const std::function<void(std::int32_t)> &on_token) {
  require_live();
  if (impl_->layer_count != impl_->config.decoder.layers) {
    throw std::logic_error("text generation requires every decoder layer");
  }
  const auto &config = impl_->config;
  const auto prompt_length = std::max<std::size_t>(1, prompt.size());
  return inference::generate_tokens(
      {config.vocabulary, config.decoder.capacity, config.bos, config.eos},
      {[this] { reset(); },
       [this, prompt_length](std::int32_t token) {
         TimingLabel phase(TimingField::Phase, length() < prompt_length ? "prefill" : "decode");
         consume(token);
       },
       [this, prompt_length] {
         TimingLabel phase(TimingField::Phase,
                           length() == prompt_length ? "first_prediction" : "decode");
         return predict();
       }},
      prompt, maximum_new_tokens, on_token);
}

void Model::close() {
  if (impl_) {
    impl_->head.graph.close();
    impl_->output.graph.close();
    impl_->decoder.close();
    impl_->input.graph.close();
    impl_->table.close();
    impl_.reset();
  }
}
} // namespace specferry::models::opt
