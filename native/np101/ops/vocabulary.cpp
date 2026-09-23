#include "np101/diagnostics.hpp"
#include "np101/ops/vocabulary.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace specferry::np101::ops {
namespace {
constexpr auto f16 = DataType::Float16;

class Lookup : public GraphBuilder {
public:
  Tensor index;

  Lookup(Context &context, TensorBinding table, TensorSpec spec, TensorBinding result)
      : GraphBuilder(context, 3, 1), index(tensor({DataType::Int32, {1}})) {
    auto values = bind(table, spec);
    auto output = bind(result, {f16, {spec.shape[0], 1}});
    node(VSI_NN_OP_GATHER, {values, index}, output)->nn_param.gather.axis = 1;
    compile({values, index}, {output});
  }
};
} // namespace

struct Vocabulary::Impl {
  unsigned rows, width, block_rows;
  Graph storage;
  vsi_nn_tensor_id_t output;
  std::vector<Tensor> blocks;
  std::vector<std::unique_ptr<Lookup>> lookups;

  Impl(Context &context, const WeightStore &store, const WeightRecord &table, unsigned block_size)
      : rows(table.spec.shape[1]), width(table.spec.shape[0]), block_rows(block_size),
        storage(context, 1 + (rows - 1) / block_rows + 1, 0),
        output(
            add_tensor(storage, {f16, {width, 1}}, false, std::vector<std::uint8_t>(width * 2))) {
    for (unsigned first = 0; first < rows; first += block_rows) {
      TensorSpec spec{f16, {width, std::min(block_rows, rows - first)}};
      auto bytes = store.read(table, std::size_t(first) * width * 2, spec.bytes());
      // Mutable SDK storage permits the already validated retained-tensor path.
      // The application treats these weights as immutable after initialization.
      auto id = add_tensor(storage, spec, false, bytes);
      blocks.push_back({id, spec});
      lookups.push_back(std::make_unique<Lookup>(context, TensorBinding{storage, id}, spec,
                                                 TensorBinding{storage, output}));
    }
  }
};

Vocabulary::Vocabulary(Context &context, const WeightStore &store, const WeightRecord &table,
                       unsigned block_rows) {
  if (table.spec.type != f16 || table.spec.shape.size() != 2 || !block_rows || block_rows > 4096 ||
      table.spec.shape[1] > std::numeric_limits<std::int32_t>::max() ||
      std::uint64_t(table.spec.shape[0]) * block_rows * 2 > 8 * 1024 * 1024) {
    throw std::invalid_argument("vocabulary requires bounded FP16 row blocks");
  }
  table.spec.bytes();
  impl_ = std::make_unique<Impl>(context, store, table, block_rows);
}

Vocabulary::~Vocabulary() = default;

void Vocabulary::lookup(std::int32_t token) {
  TimingLabel component(TimingField::Component, "embedding.lookup");
  if (!impl_ || token < 0 || unsigned(token) >= impl_->rows) {
    throw std::out_of_range("token exceeds vocabulary or table is closed");
  }
  auto &lookup = *impl_->lookups.at(token / impl_->block_rows);
  const auto local = static_cast<std::int32_t>(token % impl_->block_rows);
  std::vector<std::uint8_t> bytes(sizeof(local));
  std::memcpy(bytes.data(), &local, sizeof(local));
  upload_tensor(lookup.graph, lookup.index.id, bytes);
  check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(lookup.graph.get()); }),
        "vocabulary lookup");
}

TensorBinding Vocabulary::embedding() {
  if (!impl_) {
    throw std::logic_error("closed vocabulary");
  }
  return {impl_->storage, impl_->output};
}

TensorBinding Vocabulary::block(unsigned index) {
  if (!impl_) {
    throw std::logic_error("closed vocabulary");
  }
  return {impl_->storage, impl_->blocks.at(index).id};
}

TensorSpec Vocabulary::block_spec(unsigned index) const {
  if (!impl_) {
    throw std::logic_error("closed vocabulary");
  }
  return impl_->blocks.at(index).spec;
}

unsigned Vocabulary::blocks() const { return impl_ ? impl_->blocks.size() : 0; }

unsigned Vocabulary::rows() const { return impl_ ? impl_->rows : 0; }

unsigned Vocabulary::width() const { return impl_ ? impl_->width : 0; }

unsigned Vocabulary::block_rows() const { return impl_ ? impl_->block_rows : 0; }

void Vocabulary::close() {
  if (impl_) {
    for (auto &lookup : impl_->lookups) {
      lookup->graph.close();
    }
    impl_->lookups.clear();
    impl_->storage.close();
    impl_.reset();
  }
}

VocabularyHead::VocabularyHead(Context &context, Vocabulary &table, TensorBinding source,
                               SamplingOptions sampling)
    : GraphBuilder(context, table.blocks() * 16 + 8, table.blocks() * 12 + 4),
      vocabulary_size_(table.rows()), sampling_(sampling) {
  if (!table.blocks()) {
    throw std::invalid_argument("head requires a live vocabulary");
  }
  auto input = bind(source, {f16, {table.width(), 1}});
  std::vector<Tensor> inputs{input};
  std::vector<Tensor> scores, tokens;
  for (unsigned index = 0; index < table.blocks(); ++index) {
    auto weights = bind(table.block(index), table.block_spec(index));
    inputs.push_back(weights);
    auto logits = matmul(input, weights, {weights.spec.shape[1], 1}, false, true);
    logits_.push_back(logits);
    if (sampling_.enabled) {
      continue;
    }
    auto local = argmax(logits, 0);
    scores.push_back(reshape(gather(logits, local, 0), {1}));
    tokens.push_back(binary(VSI_NN_OP_ADD, local, integer(index * table.block_rows())));
  }
  if (sampling_.enabled) {
    auto combined = logits_.front();
    for (unsigned index = 1; index < logits_.size(); ++index) {
      combined = concat(combined, logits_[index]);
    }
    // Direct FP16 sampling returned invalid/wrong indices in the installed SDK.
    // FP16 -> FP32 is exact; checkpoint weights and head products remain FP16.
    auto logits = convert(combined, DataType::Float32);
    seed_ = tensor({DataType::Int32, {4}});
    inputs.push_back(seed_);
    token_ = tensor({DataType::Int32, {1, 1}});
    node(VSI_NN_OP_RANDOM_MULTINOMIAL, {logits, seed_}, token_)
        ->nn_param.random_multinomial.sample_num = 1;
  } else {
    auto best_scores = scores.front(), best_tokens = tokens.front();
    for (unsigned index = 1; index < scores.size(); ++index) {
      best_scores = concat(best_scores, scores[index]);
      best_tokens = concat(best_tokens, tokens[index]);
    }
    token_ = gather(best_tokens, argmax(best_scores, 0), 0);
  }
  auto outputs = logits_;
  outputs.push_back(token_);
  compile(inputs, outputs);
}

std::int32_t VocabularyHead::select() {
  TimingLabel component(TimingField::Component, "lm_head");
  if (sampling_.enabled) {
    if (draws_ == std::numeric_limits<std::uint32_t>::max()) {
      throw std::out_of_range("sampling draw counter exhausted");
    }
    // The SDK treats these INT32 tensor bytes as seed words. Advancing the
    // second word prevents reusing one uniform draw at every decoded position.
    const std::array<std::uint32_t, 4> words{sampling_.seed, draws_, 0, 0};
    std::vector<std::uint8_t> bytes(sizeof(words));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    upload_tensor(graph, seed_.id, bytes);
  }
  check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.get()); }),
        "vocabulary projection and token selection");
  auto bytes = read_tensor(graph, token_.id);
  std::int32_t result;
  std::memcpy(&result, bytes.data(), sizeof(result));
  if (result < 0 || unsigned(result) >= vocabulary_size_) {
    throw std::runtime_error("head returned a token outside the vocabulary");
  }
  if (sampling_.enabled) {
    ++draws_;
  }
  return result;
}

void VocabularyHead::reset() { draws_ = 0; }

std::vector<std::uint8_t> VocabularyHead::read_logits() {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(std::size_t(vocabulary_size_) * 2);
  for (auto logits : logits_) {
    auto part = read_tensor(graph, logits.id);
    bytes.insert(bytes.end(), part.begin(), part.end());
  }
  return bytes;
}
} // namespace specferry::np101::ops
