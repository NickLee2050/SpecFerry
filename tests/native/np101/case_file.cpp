#include "case_file.hpp"

#include <cstring>
#include <fstream>
#include <map>
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
  if (!std::getline(file, line) ||
      (line != "specferry-np101-case 1" && line != "specferry-np101-case 2")) {
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
      if (!names.insert(name).second || (storage != "constant" && storage != "mutable")) {
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
    } else if (command == "bounds") {
      std::string tensor, minimum, maximum;
      fields >> tensor >> minimum >> maximum;
      auto low = unsigned_value(minimum);
      auto high = unsigned_value(maximum);
      if (low > high || high > INT32_MAX) {
        throw std::invalid_argument("invalid integer bounds");
      }
      result.integer_bounds.push_back(
          {tensor, static_cast<std::int32_t>(low), static_cast<std::int32_t>(high)});
    } else if (command == "steps") {
      std::string value;
      fields >> value;
      result.steps = unsigned_value(value);
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
      result.steps == 0 || result.steps > 32) {
    throw std::invalid_argument("incomplete fixture or invalid step count");
  }
  auto require_tensor = [&](const std::string &name) {
    if (!names.count(name)) {
      throw std::invalid_argument("unknown tensor: " + name);
    }
  };
  std::set<std::string> produced;
  std::map<std::string, const TensorDefinition *> definitions;
  std::set<std::string> available;
  for (const auto &tensor : result.tensors) {
    definitions.emplace(tensor.name, &tensor);
    if (tensor.initial_file != "-") {
      available.insert(tensor.name);
    }
    if (tensor.storage == "constant" && tensor.initial_file == "-") {
      throw std::invalid_argument("constant tensor lacks initialization: " + tensor.name);
    }
  }
  std::set<std::string> input_names;
  for (const auto &input : result.inputs) {
    require_tensor(input.tensor);
    if (!input_names.insert(input.tensor).second ||
        definitions.at(input.tensor)->storage == "constant") {
      throw std::invalid_argument("duplicate or constant graph input: " + input.tensor);
    }
    available.insert(input.tensor);
  }
  for (const auto &node : result.nodes) {
    require_tensor(node.output);
    if (!produced.insert(node.output).second || input_names.count(node.output) ||
        definitions.at(node.output)->storage == "constant") {
      throw std::invalid_argument("invalid or multiple tensor producers: " + node.output);
    }
    for (const auto &input : node.inputs) {
      require_tensor(input);
      if (!available.count(input)) {
        throw std::invalid_argument("node reads an uninitialized tensor: " + input);
      }
    }
    available.insert(node.output);
  }
  for (const auto &input : result.inputs) {
    require_tensor(input.tensor);
    if (input.files.size() != result.steps) {
      throw std::invalid_argument("input sequence length mismatch");
    }
  }
  for (const auto &output : result.outputs) {
    require_tensor(output);
    if (!produced.count(output)) {
      throw std::invalid_argument("output is not produced by a node: " + output);
    }
  }
  std::set<std::string> bounded;
  for (const auto &bounds : result.integer_bounds) {
    require_tensor(bounds.tensor);
    const auto &tensor = *definitions.at(bounds.tensor);
    if (!bounded.insert(bounds.tensor).second || tensor.spec.type != np101::DataType::Int32 ||
        produced.count(bounds.tensor) ||
        (tensor.initial_file == "-" && !input_names.count(bounds.tensor))) {
      throw std::invalid_argument("bounds require an initialized INT32 input");
    }
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

void validate_case_data(const CaseDefinition &test) {
  auto validate = [&](const TensorDefinition &tensor, const std::string &filename) {
    auto data = read_bytes(test.root, filename, tensor.spec.bytes());
    for (const auto &bounds : test.integer_bounds) {
      if (bounds.tensor != tensor.name) {
        continue;
      }
      for (std::size_t offset = 0; offset < data.size(); offset += sizeof(std::int32_t)) {
        std::int32_t value;
        std::memcpy(&value, data.data() + offset, sizeof(value));
        if (value < bounds.minimum || value > bounds.maximum) {
          throw std::invalid_argument("index outside declared bounds: " + tensor.name);
        }
      }
    }
  };
  for (const auto &tensor : test.tensors) {
    if (tensor.initial_file != "-") {
      validate(tensor, tensor.initial_file);
    }
    for (const auto &input : test.inputs) {
      if (input.tensor == tensor.name) {
        for (const auto &filename : input.files) {
          validate(tensor, filename);
        }
      }
    }
  }
}
} // namespace specferry::testing
