#pragma once

#include <stdexcept>
#include <string_view>

namespace specferry::models::qwen3_5 {

// Block prefill may skip scalar producers and output publication. Their retained
// tensors remain bound for decode, but are not snapshots of the latest token.
struct DecoderReadbackState {
  bool scalar_mixer = false;
  bool published_output = false;

  void require_available(std::string_view name) const {
    if (name == "output" && !published_output) {
      throw std::logic_error("decoder output was not published by the latest execution");
    }
    if ((name == "normalized" || name == "mixer") && !scalar_mixer) {
      throw std::logic_error("scalar mixer diagnostics are unavailable after block prefill");
    }
  }
};

} // namespace specferry::models::qwen3_5
