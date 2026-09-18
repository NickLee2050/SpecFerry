#include "np101/attention.hpp"
#include "np101/kv_cache.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace specferry::np101 {
namespace {
using Shape = std::vector<std::uint32_t>;
constexpr auto f16 = DataType::Float16;
constexpr auto f32 = DataType::Float32;
constexpr auto i32 = DataType::Int32;
constexpr unsigned capacity = KvCache::maximum_capacity;
const std::string weight_prefix = "model.layers.3.self_attn.";

struct Tensor {
  vsi_nn_tensor_id_t id;
  TensorSpec spec;
};

std::vector<std::uint8_t> integer_bytes(std::int32_t value) {
  std::vector<std::uint8_t> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

// Small construction helpers shared by this mixer's two fixed compute graphs.
// All arithmetic operators are listed in the chip team's reference guide.
class AttentionGraph {
public:
  std::deque<Shape> parameters;
  Graph graph;

  explicit AttentionGraph(Context &context) : graph(context, 100, 80) {}

  Tensor tensor(TensorSpec spec) { return {add_tensor(graph, spec), std::move(spec)}; }

  Tensor constant(TensorSpec spec, const std::vector<std::uint8_t> &bytes) {
    return {add_tensor(graph, spec, true, bytes), std::move(spec)};
  }

  Tensor share(AttentionGraph &owner, Tensor tensor) {
    return {retain_tensor(owner.graph, tensor.id, graph), tensor.spec};
  }

  Tensor floats(const std::vector<float> &values, Shape shape, DataType type) {
    vsi_nn_dtype_t dtype{};
    dtype.vx_type = type == f16 ? VSI_NN_TYPE_FLOAT16 : VSI_NN_TYPE_FLOAT32;
    dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    TensorSpec spec{type, std::move(shape)};
    if (values.size() != spec.elements()) {
      throw std::invalid_argument("attention constant size mismatch");
    }
    std::vector<std::uint8_t> bytes(spec.bytes());
    for (std::size_t index = 0; index < values.size(); ++index) {
      check(vsi_nn_Float32ToDtype(values[index], bytes.data() + index * dtype_bytes(type), &dtype),
            "encode attention constant");
    }
    return constant(spec, bytes);
  }

  Tensor scalar(float value, DataType type = f32) { return floats({value}, {1}, type); }

  Tensor weight(const WeightStore &weights, const std::string &name, Shape shape) {
    const auto &record = weights.find(weight_prefix + name);
    if (record.spec.type != f16 || record.spec.shape != shape) {
      throw std::invalid_argument("unexpected attention weight contract: " + name);
    }
    return constant(record.spec, weights.read(record, 0, record.bytes));
  }

  vsi_nn_node_t *node(vsi_nn_op_t op, std::initializer_list<Tensor> inputs, Tensor output) {
    auto *result = vsi_nn_AddNode(graph.get(), op, 0, 0, nullptr);
    if (!result || result->input.num < inputs.size() || result->output.num < 1) {
      throw std::runtime_error("attention AddNode failed");
    }
    std::fill_n(result->input.tensors, result->input.num, VSI_NN_TENSOR_ID_NA);
    unsigned slot = 0;
    for (const auto &input : inputs) {
      result->input.tensors[slot++] = input.id;
    }
    result->output.tensors[0] = output.id;
    if (op == VSI_NN_OP_MULTIPLY) {
      result->nn_param.multiply.scale = 1.0f;
    }
    return result;
  }

  Tensor unary(vsi_nn_op_t op, Tensor input, DataType type) {
    auto result = tensor({type, input.spec.shape});
    node(op, {input}, result);
    return result;
  }

  Tensor convert(Tensor input, DataType type) { return unary(VSI_NN_OP_DATACONVERT, input, type); }

  Tensor binary(vsi_nn_op_t op, Tensor left, Tensor right) {
    auto result = tensor(left.spec);
    node(op, {left, right}, result);
    return result;
  }

  Tensor reshape(Tensor input, Shape shape) {
    auto result = tensor({input.spec.type, shape});
    parameters.push_back(std::move(shape));
    auto *operation = node(VSI_NN_OP_RESHAPE, {input}, result);
    operation->nn_param.reshape.size = parameters.back().data();
    operation->nn_param.reshape.dim_num = parameters.back().size();
    return result;
  }

  Tensor slice(Tensor input, Shape start, Shape length) {
    auto result = tensor({input.spec.type, length});
    auto *operation = node(VSI_NN_OP_SLICE, {input}, result);
    operation->nn_param.slice.dims = start.size();
    parameters.push_back(std::move(start));
    operation->nn_param.slice.start = parameters.back().data();
    parameters.push_back(std::move(length));
    operation->nn_param.slice.length = parameters.back().data();
    return result;
  }

  Tensor concat(Tensor left, Tensor right) {
    auto shape = left.spec.shape;
    shape[0] += right.spec.shape[0];
    auto result = tensor({left.spec.type, shape});
    node(VSI_NN_OP_CONCAT, {left, right}, result)->nn_param.concat.axis = 0;
    return result;
  }

  Tensor matmul(Tensor left, Tensor right, Shape shape, bool transpose_right) {
    auto result = tensor({left.spec.type, std::move(shape)});
    auto *operation = node(VSI_NN_OP_MATRIXMUL, {left, right}, result);
    operation->nn_param.matrixmul.transpose[0] = false;
    operation->nn_param.matrixmul.transpose[1] = transpose_right;
    return result;
  }

  Tensor project(const WeightStore &weights, Tensor input, const std::string &name, unsigned rows) {
    return matmul(input, weight(weights, name + ".weight", {input.spec.shape[0], rows}), {rows, 1},
                  true);
  }

  Tensor rms_norm(const WeightStore &weights, Tensor input, const std::string &name) {
    auto value = convert(input, f32);
    auto squares = binary(VSI_NN_OP_MULTIPLY, value, value);
    auto shape = squares.spec.shape;
    shape[0] = 1;
    auto mean = tensor({f32, shape});
    parameters.push_back({0});
    auto *operation = node(VSI_NN_OP_REDUCE, {squares}, mean);
    operation->nn_param.reduce.type = VSI_NN_REDUCE_MEAN;
    operation->nn_param.reduce.axis = reinterpret_cast<const int32_t *>(parameters.back().data());
    operation->nn_param.reduce.axis_num = 1;
    operation->nn_param.reduce.keep_dim = TRUE;
    auto inverse = unary(VSI_NN_OP_RSQRT, binary(VSI_NN_OP_ADD, mean, scalar(1e-6f)), f32);
    auto normalized = binary(VSI_NN_OP_MULTIPLY, value, inverse);
    auto scale =
        binary(VSI_NN_OP_ADD, convert(weight(weights, name + ".weight", {256}), f32), scalar(1));
    return convert(binary(VSI_NN_OP_MULTIPLY, normalized, reshape(scale, {256, 1})), f16);
  }

  void compile(std::initializer_list<Tensor> inputs, std::initializer_list<Tensor> outputs) {
    std::vector<vsi_nn_tensor_id_t> input_ids, output_ids;
    for (const auto &input : inputs) {
      input_ids.push_back(input.id);
    }
    for (const auto &output : outputs) {
      output_ids.push_back(output.id);
    }
    if (!vsi_nn_SetGraphInputs(graph.get(), input_ids.data(), input_ids.size()) ||
        !vsi_nn_SetGraphOutputs(graph.get(), output_ids.data(), output_ids.size())) {
      throw std::runtime_error("attention IO declaration failed");
    }
    check(vsi_nn_SetupGraph(graph.get(), FALSE), "attention SetupGraph");
    check(vsi_nn_VerifyGraph(graph.get()), "attention VerifyGraph");
  }
};

class ProjectionGraph : public AttentionGraph {
public:
  Tensor hidden, position, query, key, value, gate;

  ProjectionGraph(Context &context, const WeightStore &weights) : AttentionGraph(context) {
    hidden = tensor({f16, {1024, 1}});
    position = tensor({i32, {1}});
    auto packed = reshape(project(weights, hidden, "q_proj", 4096), {512, 8});
    query = rms_norm(weights, slice(packed, {0, 0}, {256, 8}), "q_norm");
    gate = slice(packed, {256, 0}, {256, 8});
    key = rms_norm(weights, reshape(project(weights, hidden, "k_proj", 512), {256, 2}), "k_norm");
    value = reshape(project(weights, hidden, "v_proj", 512), {256, 1, 2});
    auto cosine = rope_table(false);
    auto sine = rope_table(true);
    query = reshape(rotate(query, cosine, sine), {256, 4, 2});
    key = reshape(rotate(key, cosine, sine), {256, 1, 2});
    compile({hidden, position}, {query, key, value, gate});
  }

private:
  Tensor rope_table(bool sine) {
    std::vector<float> table(capacity * 64);
    for (unsigned position = 0; position < capacity; ++position) {
      for (unsigned dim = 0; dim < 64; ++dim) {
        const float frequency = 1.0f / std::pow(10000000.0f, float(2 * (dim % 32)) / 64);
        const float angle = position * frequency;
        table[position * 64 + dim] = sine ? std::sin(angle) : std::cos(angle);
      }
    }
    auto data = floats(table, {64, capacity}, f16);
    auto row = tensor({f16, {64, 1}});
    node(VSI_NN_OP_GATHER, {data, position}, row)->nn_param.gather.axis = 1;
    return row;
  }

  Tensor rotate(Tensor input, Tensor cosine, Tensor sine) {
    auto heads = input.spec.shape[1];
    auto rotary = slice(input, {0, 0}, {64, heads});
    auto first = slice(rotary, {0, 0}, {32, heads});
    auto second = slice(rotary, {32, 0}, {32, heads});
    auto turned = concat(binary(VSI_NN_OP_MULTIPLY, second, scalar(-1, f16)), first);
    auto rotated = binary(VSI_NN_OP_ADD, binary(VSI_NN_OP_MULTIPLY, rotary, cosine),
                          binary(VSI_NN_OP_MULTIPLY, turned, sine));
    return concat(rotated, slice(input, {64, 0}, {192, heads}));
  }
};

class ReaderGraph : public AttentionGraph {
public:
  Tensor valid_length, output, probabilities;

  ReaderGraph(Context &context, const WeightStore &weights, ProjectionGraph &producer,
              KvCache &cache)
      : AttentionGraph(context) {
    auto query = share(producer, producer.query);
    auto gate = share(producer, producer.gate);
    Tensor keys{cache.retain_keys(graph), {f16, {256, capacity, 2}}};
    Tensor values{cache.retain_values(graph), keys.spec};
    valid_length = tensor({i32, {1}});

    // Group four Q heads under each KV head. Both GEMMs read the compact FP16
    // parent cache directly; there is no full-cache gather, expansion or cast.
    auto scores = matmul(query, keys, {capacity, 4, 2}, true);
    scores = convert(binary(VSI_NN_OP_MULTIPLY, scores, scalar(1.0f / 16, f16)), f32);
    // Keep masking/softmax in the validated [token, head] SDK layout. The
    // installed SDK's rank-three softmax did not honor the requested token axis.
    scores = reshape(scores, {capacity, 8});
    std::vector<std::uint8_t> position_bytes(capacity * sizeof(std::int32_t));
    for (unsigned position = 0; position < capacity; ++position) {
      auto bytes = integer_bytes(position);
      std::copy(bytes.begin(), bytes.end(), position_bytes.begin() + position * bytes.size());
    }
    auto positions = constant({i32, {capacity, 1}}, position_bytes);
    auto valid = tensor({DataType::Bool8, positions.spec.shape});
    node(VSI_NN_OP_RELATIONAL_OPS, {positions, valid_length}, valid)->nn_param.relational_ops.op =
        VSI_NN_RELATIONAL_OPS_LESS;
    auto masked = tensor(scores.spec);
    node(VSI_NN_OP_SELECT, {valid, scores, scalar(-1e9f)}, masked);
    auto probability_fp32 = tensor(masked.spec);
    auto *softmax = node(VSI_NN_OP_SOFTMAX, {masked}, probability_fp32);
    softmax->nn_param.softmax.axis = 0;
    softmax->nn_param.softmax.beta = 1.0f;
    probabilities = reshape(convert(probability_fp32, f16), {capacity, 4, 2});
    auto attended = reshape(matmul(probabilities, values, {256, 4, 2}, false), {256, 8});
    auto gated = binary(VSI_NN_OP_MULTIPLY, attended, unary(VSI_NN_OP_SIGMOID, gate, f16));
    output = project(weights, reshape(gated, {2048, 1}), "o_proj", 1024);
    compile({query, gate, keys, values, valid_length}, {output, probabilities});
  }
};
} // namespace

struct Attention::Impl {
  ProjectionGraph producer;
  KvCache cache;
  ReaderGraph reader;
  unsigned length = 0;
  bool failed = false;
  bool has_output = false;

  Impl(Context &context, const WeightStore &weights)
      : producer(context, weights),
        cache(context, producer.graph, producer.key.id, producer.value.id),
        reader(context, weights, producer, cache) {}
};

Attention::Attention(Context &context, const WeightStore &weights)
    : impl_(std::make_unique<Impl>(context, weights)) {}

Attention::~Attention() = default;

void Attention::step(const std::vector<std::uint8_t> &hidden_fp16) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("attention is closed or must be recreated after failure");
  }
  if (impl_->length >= capacity) {
    throw std::out_of_range("attention KV capacity exhausted");
  }
  if (hidden_fp16.size() != 1024 * 2) {
    throw std::invalid_argument("attention expects one FP16 hidden vector");
  }
  impl_->failed = true;
  impl_->has_output = false;
  auto &producer = impl_->producer;
  upload_tensor(producer.graph, producer.hidden.id, hidden_fp16);
  upload_tensor(producer.graph, producer.position.id, integer_bytes(impl_->length));
  check(vsi_nn_RunGraph(producer.graph.get()), "attention projections");
  impl_->cache.write(impl_->length);
  upload_tensor(impl_->reader.graph, impl_->reader.valid_length.id,
                integer_bytes(impl_->length + 1));
  check(vsi_nn_RunGraph(impl_->reader.graph.get()), "attention cache reader");
  ++impl_->length;
  impl_->has_output = true;
  impl_->failed = false;
}

void Attention::reset() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed attention must be recreated");
  }
  // Storage remains allocated. Every reused position is overwritten before it
  // becomes visible; the mask excludes the stale suffix.
  impl_->length = 0;
  impl_->failed = false;
  impl_->has_output = false;
}

void Attention::truncate(unsigned length) {
  if (!impl_ || impl_->failed || length > impl_->length) {
    throw std::out_of_range("cannot extend or truncate an invalid attention cache");
  }
  impl_->length = length;
  impl_->has_output = false;
}

unsigned Attention::length() const { return impl_ ? impl_->length : 0; }

std::vector<std::uint8_t> Attention::read(const std::string &name) {
  if (!impl_ || impl_->failed || !impl_->has_output) {
    throw std::logic_error("attention has no completed output");
  }
  if (name == "keys") {
    return impl_->cache.read_keys();
  }
  if (name == "values") {
    return impl_->cache.read_values();
  }
  if (name == "query" || name == "key" || name == "value") {
    auto &producer = impl_->producer;
    auto tensor = name == "query" ? producer.query : name == "key" ? producer.key : producer.value;
    return read_tensor(producer.graph, tensor.id);
  }
  if (name == "output" || name == "probabilities") {
    auto &reader = impl_->reader;
    return read_tensor(reader.graph, name == "output" ? reader.output.id : reader.probabilities.id);
  }
  throw std::invalid_argument("unknown attention output: " + name);
}

std::size_t Attention::cache_writes() const { return impl_ ? impl_->cache.writes() : 0; }

std::size_t Attention::cache_revalidations() const {
  return impl_ ? impl_->cache.revalidations() : 0;
}

void Attention::close() {
  if (impl_) {
    impl_->reader.graph.close();
    impl_->cache.close();
    impl_->producer.graph.close();
    impl_.reset();
  }
}
} // namespace specferry::np101
