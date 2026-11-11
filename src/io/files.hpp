#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace specferry::io {
void write_bytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes,
                 const char *error);
} // namespace specferry::io
