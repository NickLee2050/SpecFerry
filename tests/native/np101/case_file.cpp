#include "case_file.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace specferry::testing {
namespace {
std::vector<std::string> split(const std::string &text) {
  std::vector<std::string> values;
  std::istringstream stream(text);
  std::string value;
  if (text.empty() || text.back() == ',') {
    throw std::invalid_argument("empty list entry");
  }
  while (std::getline(stream, value, ',')) {
    if (value.empty()) {
      throw std::invalid_argument("empty list entry");
    }
    values.push_back(value);
  }
  return values;
}

void require_name(const std::string &name) {
  if (name.empty() ||
      name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") !=
          std::string::npos ||
      name == "." || name == "..") {
    throw std::invalid_argument("invalid fixture identifier: " + name);
  }
}

std::uint32_t unsigned_value(const std::string &text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("invalid nonnegative integer: " + text);
  }
  auto value = std::stoull(text);
  if (value > UINT32_MAX) {
    throw std::invalid_argument("integer exceeds uint32 range");
  }
  return static_cast<std::uint32_t>(value);
}
} // namespace

CaseDefinition load_case(const std::filesystem::path &path) {
  std::ifstream file(path);
  std::string line;
  if (!std::getline(file, line) || line != "specferry-np101-case 1") {
    throw std::invalid_argument("invalid case header");
  }
  CaseDefinition result;
  result.root = std::filesystem::canonical(path).parent_path();
  std::set<std::string> names;
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string command;
    fields >> command;
    if (command == "tensor") {
      std::string name, dtype, storage, shape, initial;
      fields >> name >> dtype >> storage >> shape >> initial;
      require_name(name);
      if (!names.insert(name).second ||
          (storage != "constant" && storage != "mutable" && storage != "handle")) {
        throw std::invalid_argument("duplicate tensor or invalid storage: " + name);
      }
      np101::TensorSpec spec{np101::parse_dtype(dtype), np101::parse_shape(shape)};
      if (spec.bytes() > 1024ULL * 1024 * 1024) {
        throw std::invalid_argument("fixture tensor exceeds 1 GiB limit");
      }
      result.tensors.push_back({name, spec, storage, initial});
    } else if (command == "node") {
      NodeDefinition node;
      std::string inputs, parameters;
      fields >> node.operation >> inputs >> node.output >> parameters;
      node.inputs = split(inputs);
      if (parameters != "-") {
        for (const auto &value : split(parameters)) {
          node.parameters.push_back(unsigned_value(value));
        }
      }
      result.nodes.push_back(node);
    } else if (command == "input") {
      InputSequence input;
      std::string files;
      fields >> input.tensor >> files;
      input.files = split(files);
      result.inputs.push_back(input);
    } else if (command == "output") {
      std::string name;
      fields >> name;
      result.outputs.push_back(name);
    } else if (command == "feedback") {
      if (!result.feedback_input.empty()) {
        throw std::invalid_argument("only one feedback connection is supported per fixture");
      }
      fields >> result.feedback_output >> result.feedback_input;
    } else if (command == "steps" || command == "reset_after") {
      std::string value;
      fields >> value;
      auto number = unsigned_value(value);
      if (command == "steps") {
        result.steps = number;
      } else {
        result.reset_after = number;
      }
    } else {
      throw std::invalid_argument("unknown case directive: " + command);
    }
    if (fields.fail()) {
      throw std::invalid_argument("missing fields: " + line);
    }
    std::string extra;
    if (fields >> extra) {
      throw std::invalid_argument("extra fields: " + line);
    }
  }
  if (result.tensors.empty() || result.nodes.empty() || result.outputs.empty() ||
      result.steps == 0 || result.steps > 32 || result.reset_after >= result.steps) {
    throw std::invalid_argument("incomplete fixture or invalid step count");
  }
  auto require_tensor = [&](const std::string &name) {
    if (!names.count(name)) {
      throw std::invalid_argument("unknown tensor: " + name);
    }
  };
  std::set<std::string> produced;
  for (const auto &node : result.nodes) {
    require_tensor(node.output);
    if (!produced.insert(node.output).second) {
      throw std::invalid_argument("tensor has multiple producers: " + node.output);
    }
    for (const auto &input : node.inputs) {
      require_tensor(input);
    }
  }
  for (const auto &input : result.inputs) {
    require_tensor(input.tensor);
    if (input.files.size() != result.steps) {
      throw std::invalid_argument("input sequence length mismatch");
    }
  }
  for (const auto &output : result.outputs) {
    require_tensor(output);
  }
  if (!result.feedback_input.empty()) {
    require_tensor(result.feedback_input);
    require_tensor(result.feedback_output);
  }
  return result;
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path &root, const std::string &name,
                                     std::size_t expected_bytes) {
  require_name(name);
  auto path = std::filesystem::canonical(root / name);
  if (path.parent_path() != std::filesystem::canonical(root) ||
      std::filesystem::file_size(path) != expected_bytes) {
    throw std::invalid_argument("fixture path or byte count mismatch: " + name);
  }
  std::vector<std::uint8_t> data(expected_bytes);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(data.data()), data.size());
  if (!input) {
    throw std::runtime_error("cannot read fixture: " + name);
  }
  return data;
}
} // namespace specferry::testing
