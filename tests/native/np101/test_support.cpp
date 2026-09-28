#include "np101/context.hpp"
#include "test_support.hpp"
#include "vsi_nn_pub.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace specferry::testing {
std::vector<std::uint8_t> encode_floats(const std::vector<float> &values, np101::DataType type) {
  if (type != np101::DataType::Float16 && type != np101::DataType::Float32) {
    throw std::invalid_argument("test data requires FP16 or FP32");
  }
  vsi_nn_dtype_t dtype{};
  dtype.vx_type = type == np101::DataType::Float16 ? VSI_NN_TYPE_FLOAT16 : VSI_NN_TYPE_FLOAT32;
  dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
  const auto width = np101::dtype_bytes(type);
  std::vector<std::uint8_t> bytes(values.size() * width);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      throw std::invalid_argument("nonfinite test input");
    }
    np101::check(vsi_nn_Float32ToDtype(values[index], bytes.data() + index * width, &dtype),
                 "encode test float");
  }
  return bytes;
}
} // namespace specferry::testing
