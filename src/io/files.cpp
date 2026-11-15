#include "io/files.hpp"

#include <fstream>
#include <stdexcept>

namespace specferry::io {
void write_bytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes,
                 const char *error) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (!output) {
    throw std::runtime_error(error);
  }
}
} // namespace specferry::io
