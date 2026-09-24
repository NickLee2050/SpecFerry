#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specferry::testing {
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
