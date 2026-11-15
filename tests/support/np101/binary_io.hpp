#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace specferry::testing {
template <class T> std::vector<std::uint8_t> as_bytes(const std::vector<T> &values) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

inline void write_bytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (!output) {
    throw std::runtime_error("cannot write bytes: " + path.string());
  }
}

inline void write_text(const std::filesystem::path &path, std::string_view text) {
  std::ofstream output(path);
  output << text;
  output.close();
  if (!output) {
    throw std::runtime_error("cannot write text: " + path.string());
  }
}

inline std::vector<std::uint8_t> read_bytes(const std::filesystem::path &root,
                                            const std::string &name, std::size_t expected_bytes) {
  const auto path = root / name;
  if (std::filesystem::file_size(path) != expected_bytes) {
    throw std::runtime_error("wrong fixture byte count: " + name);
  }
  std::vector<std::uint8_t> data(expected_bytes);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char *>(data.data()), data.size())) {
    throw std::runtime_error("cannot read fixture: " + name);
  }
  return data;
}
} // namespace specferry::testing
