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

Tensor lookup_blocks(GraphBuilder &g, const std::vector<Tensor> &blocks, Tensor tokens) {
  if (blocks.empty() || tokens.spec.type != DataType::Int32 || tokens.spec.shape.size() != 1) {
    throw std::invalid_argument("blocked lookup requires FP16 blocks and an INT32 token vector");
  }
  const auto width = blocks.front().spec.shape.at(0), count = tokens.spec.shape[0];
  Tensor result{};
  unsigned offset = 0;
  for (const auto &block : blocks) {
    if (block.spec.type != f16 || block.spec.shape.size() != 2 || block.spec.shape[0] != width) {
      throw std::invalid_argument("vocabulary block widths/dtypes differ");
    }
    auto local = g.binary(VSI_NN_OP_SUBTRACT, tokens, g.integer(offset));
    auto bounded = g.binary(VSI_NN_OP_MINIMUM, g.binary(VSI_NN_OP_MAXIMUM, local, g.integer(0)),
                            g.integer(block.spec.shape[1] - 1));
    auto embedding = g.gather(block, bounded, 1);
    auto lower = g.tensor({DataType::Bool8, {count}});
    auto upper = g.tensor(lower.spec);
    g.node(VSI_NN_OP_RELATIONAL_OPS, {local, g.integer(0)}, lower)->nn_param.relational_ops.op =
        VSI_NN_RELATIONAL_OPS_GREAT_EQUAL;
    g.node(VSI_NN_OP_RELATIONAL_OPS, {local, g.integer(block.spec.shape[1])}, upper)
        ->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_LESS;
    auto selected = g.tensor(embedding.spec), masked = g.tensor(embedding.spec);
    g.node(VSI_NN_OP_SELECT, {g.reshape(lower, {1, count}), embedding, g.scalar(0, f16)}, selected);
    g.node(VSI_NN_OP_SELECT, {g.reshape(upper, {1, count}), selected, g.scalar(0, f16)}, masked);
    result = offset ? g.binary(VSI_NN_OP_ADD, result, masked) : masked;
    offset += block.spec.shape[1];
  }
  return result;
}

Tensor greedy_token(GraphBuilder &g, const std::vector<Tensor> &logits, unsigned block_rows) {
  if (logits.empty() || !block_rows) {
    throw std::invalid_argument("greedy selection requires nonempty ordered logit blocks");
  }
  std::vector<Tensor> scores, tokens;
  for (unsigned index = 0; index < logits.size(); ++index) {
    auto local = g.argmax(logits[index], 0);
    scores.push_back(g.reshape(g.gather(logits[index], local, 0), {1}));
    tokens.push_back(g.binary(VSI_NN_OP_ADD, local, g.integer(index * block_rows)));
  }
  auto best_scores = scores.front(), best_tokens = tokens.front();
  for (unsigned index = 1; index < scores.size(); ++index) {
    best_scores = g.concat(best_scores, scores[index]);
    best_tokens = g.concat(best_tokens, tokens[index]);
  }
  return g.gather(best_tokens, g.argmax(best_scores, 0), 0);
}

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
  for (unsigned index = 0; index < table.blocks(); ++index) {
    auto weights = bind(table.block(index), table.block_spec(index));
    inputs.push_back(weights);
    auto logits = matmul(input, weights, {weights.spec.shape[1], 1}, false, true);
    logits_.push_back(logits);
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
    token_ = greedy_token(*this, logits_, table.block_rows());
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
