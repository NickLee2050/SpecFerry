#include "np101/component_spec.hpp"
#include "np101/context.hpp"
#include "np101/kv_cache.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;

std::vector<std::uint8_t> fp16(const std::vector<float> &values) {
  vsi_nn_dtype_t dtype{};
  dtype.vx_type = VSI_NN_TYPE_FLOAT16;
  dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
  std::vector<std::uint8_t> bytes(values.size() * 2);
  for (std::size_t index = 0; index < values.size(); ++index) {
    check(vsi_nn_Float32ToDtype(values[index], bytes.data() + index * 2, &dtype), "encode FP16");
  }
  return bytes;
}

void require_equal(const std::vector<std::uint8_t> &actual, const std::vector<float> &expected,
                   const char *label) {
  auto bytes = fp16(expected);
  if (actual != bytes) {
    std::size_t first = 0;
    while (first < actual.size() && first < bytes.size() && actual[first] == bytes[first]) {
      ++first;
    }
    throw std::runtime_error(std::string(label) + " mismatch at byte " + std::to_string(first));
  }
}

void run(const std::filesystem::path &directory, KvSpec spec) {
  spec.validate();
  const auto capacity = spec.capacity;
  Context context;
  Graph producer(context, 2, 0);
  const auto token = spec.slot();
  auto key = add_tensor(producer, token, false, std::vector<std::uint8_t>(token.bytes()));
  auto value = add_tensor(producer, token, false, std::vector<std::uint8_t>(token.bytes()));
  KvCache cache(context, producer, key, value, spec);

  // Compile this reader once, before any write. Reading its output first avoids
  // accidentally making parent-tensor host readback a coherency prerequisite.
  Graph reader(context, 3, 1);
  for (const TensorSpec &wrong : {TensorSpec{DataType::Float16, {spec.head_dim * spec.heads, 1}},
                                  TensorSpec{DataType::Float32, token.shape}}) {
    bool rejected = false;
    try {
      bind_tensor({producer, key}, reader, wrong);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error("binding accepted an incompatible shape or dtype");
    }
  }
  auto keys = cache.retain_keys(reader);
  auto values = cache.retain_values(reader);
  auto output = add_tensor(reader, spec.tensor());
  auto *add = vsi_nn_AddNode(reader.get(), VSI_NN_OP_ADD, 2, 1, nullptr);
  if (!add) {
    throw std::runtime_error("cannot create fixed KV reader");
  }
  add->input.tensors[0] = keys;
  add->input.tensors[1] = values;
  add->output.tensors[0] = output;
  std::array<vsi_nn_tensor_id_t, 2> inputs{keys, values};
  if (!vsi_nn_SetGraphInputs(reader.get(), inputs.data(), inputs.size()) ||
      !vsi_nn_SetGraphOutputs(reader.get(), &output, 1)) {
    throw std::runtime_error("cannot declare fixed KV reader IO");
  }
  check(vsi_nn_SetupGraph(reader.get(), FALSE), "setup fixed KV reader");
  check(vsi_nn_VerifyGraph(reader.get()), "verify fixed KV reader");

  std::vector<float> expected_keys(spec.tensor().elements(), 0.0f);
  std::vector<float> expected_values(expected_keys.size(), 0.0f);
  std::vector<float> expected_sum(expected_keys.size(), 0.0f);
  const std::array<unsigned, 8> positions{
      0, 1 % capacity, 3 % capacity, 7 % capacity, capacity / 2, capacity - 1, 1 % capacity, 0};
  for (unsigned pass = 0; pass < 2; ++pass) {
    for (unsigned iteration = 0; iteration < positions.size(); ++iteration) {
      unsigned position = positions[iteration];
      std::cout << "write pass=" << pass << " position=" << position << std::endl;
      std::vector<float> next_key(token.elements()), next_value(token.elements());
      for (unsigned head = 0; head < spec.heads; ++head) {
        for (unsigned dim = 0; dim < spec.head_dim; ++dim) {
          auto index = head * spec.head_dim + dim;
          next_key[index] = 0.125f * (1 + pass + iteration) + 0.25f * head;
          next_value[index] = 0.5f + 0.5f * head + 0.0625f * (dim % 4);
          auto offset = (head * capacity + position) * spec.head_dim + dim;
          expected_keys[offset] = next_key[index];
          expected_values[offset] = next_value[index];
          expected_sum[offset] = next_key[index] + next_value[index];
        }
      }
      upload_tensor(producer, key, fp16(next_key));
      upload_tensor(producer, value, fp16(next_value));
      cache.write(position);
      check(vsi_nn_RunGraph(reader.get()), "run fixed KV reader");
      if (pass == 0 || iteration + 1 == positions.size()) {
        require_equal(read_tensor(reader, output), expected_sum, "fixed reader");
        require_equal(cache.read_keys(), expected_keys, "key storage");
        require_equal(cache.read_values(), expected_values, "value storage");
      }
    }
  }
  bool rejected = false;
  try {
    cache.write(capacity);
  } catch (const std::out_of_range &) {
    rejected = true;
  }
  if (!rejected || cache.writes() != 16) {
    throw std::runtime_error("invalid slot was not rejected before submission");
  }
  const auto revalidations = cache.revalidations();
  reader.close();
  cache.close();
  producer.close();
  context.close();
  std::ofstream report(directory / "cache.json");
  report << "{\"status\":\"numerical_pass\",\"writes\":16,"
            "\"cache_payload_bytes\":"
         << spec.token_bytes() * capacity << ",\"write_payload_bytes\":" << spec.token_bytes()
         << ","
            "\"copy_graph_revalidations\":"
         << revalidations
         << ","
            "\"fixed_reader_verifications\":1,\"released\":true,"
            "\"hardware_execution_proven\":false,\"device_residency_verified\":false}\n";
  if (!report) {
    throw std::runtime_error("cannot write cache report");
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2 && argc != 5) {
    std::cerr << "usage: np101_kv_cache_check OUTPUT_DIRECTORY [HEADS HEAD_DIM CAPACITY]\n";
    return 1;
  }
  try {
    std::filesystem::create_directories(argv[1]);
    KvSpec spec{2, 256, 512};
    if (argc == 5) {
      auto dimension = [](const char *argument) {
        auto shape = parse_shape(argument);
        if (shape.size() != 1) {
          throw std::invalid_argument("expected one positive dimension");
        }
        return shape.front();
      };
      spec = {dimension(argv[2]), dimension(argv[3]), dimension(argv[4])};
    }
    run(argv[1], spec);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
