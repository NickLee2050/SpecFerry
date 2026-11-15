#pragma once

#include <istream>
#include <string>

namespace specferry::np101 {
// Read and check the package's SDK records before constructing a Context.
void verify_compiled_sdk(std::istream &manifest, const std::string &profile, unsigned libraries);
} // namespace specferry::np101
