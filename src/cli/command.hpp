#pragma once

#include <exception>
#include <iostream>

namespace specferry::cli {
// Command boundaries translate errors once. Model sessions preserve their own failure records.
template <class Execute> int command(int argc, int expected, const char *usage, Execute execute) {
  if (argc != expected) {
    std::cerr << usage << '\n';
    return 2;
  }
  try {
    return execute();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
} // namespace specferry::cli
