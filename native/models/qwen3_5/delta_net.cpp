#include "models/qwen3_5/compiled.hpp"
#include "models/qwen3_5/delta_net.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "vsi_nn_pub.h"

#include <map>
#include <stdexcept>

namespace specferry::models::qwen3_5 {
using namespace np101;
using namespace np101::ops;

namespace {
class CompiledStep {
public:
  GraphBuilder execution;
  std::unique_ptr<GraphBuilder> commit;
  Tensor input, recurrent, convolution;
  std::map<std::string, Tensor> outputs;

  CompiledStep(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
               const CompiledBlock &artifact, TensorBinding incoming, TensorBinding destination,
               bool deferred_commit)
      : execution(context, 16, 1) {
    validate_compiled_mixer(artifact, weights, config, layer);
    input = execution.bind(incoming, config.hidden_spec());
    recurrent = execution.tensor(config.delta.recurrent());
    convolution = execution.tensor(config.delta.convolution());
    const std::vector<Tensor> inputs{input, recurrent, convolution};
    std::vector<Tensor> sdk_inputs(3);
    for (unsigned index = 0; index < 3; ++index) {
      sdk_inputs[artifact.input_order[index]] = inputs[index];
    }
    const auto sdk_outputs = execution.compiled(artifact.graph, sdk_inputs);
    std::vector<Tensor> logical;
    for (const auto slot : artifact.output_order) {
      logical.push_back(sdk_outputs[slot]);
    }
    execution.compile(inputs, logical);
    outputs = {{"output", logical[0]}, {"recurrent", logical[1]}, {"convolution", logical[2]}};
    if (deferred_commit) {
      return;
    }
    commit = std::make_unique<GraphBuilder>(context, 6, 3);
    // Fixed ordinary copy graph avoids NBG input/output aliases and alternating imports.
    const std::vector<Tensor> destinations{commit->bind(destination, config.hidden_spec()),
                                           commit->share(execution, recurrent),
                                           commit->share(execution, convolution)};
    std::vector<Tensor> sources;
    for (unsigned index = 0; index < 3; ++index) {
      const auto source = commit->share(execution, logical[index]);
      commit->node(VSI_NN_OP_DATACONVERT, {source}, destinations[index]);
      sources.push_back(source);
    }
    commit->compile(sources, destinations);
  }

  void run() {
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(execution.graph.get()); }),
          "compiled DeltaNet RunGraph");
    if (commit) {
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(commit->graph.get()); }),
            "compiled DeltaNet state commit");
    }
  }

  void close() {
    if (commit) {
      commit->graph.close();
    }
    execution.graph.close();
  }
};
} // namespace

struct DeltaNet::Impl {
  Config config;
  CompiledStep compiled;
  std::size_t completed_steps = 0;
  bool failed = false, pending_commit = false;

  Impl(Context &context, const WeightStore &weights, const Config &configuration, unsigned layer,
       TensorBinding input, TensorBinding output, const CompiledBlock &artifact,
       bool deferred_commit)
      : config(configuration),
        compiled(context, weights, configuration, layer, artifact, input, output, deferred_commit) {
  }
};

DeltaNet::DeltaNet(Context &context, const WeightStore &weights, const Config &config,
                   unsigned layer, TensorBinding input, TensorBinding output,
                   const CompiledBlock &compiled, bool deferred_commit) {
  validate_layer_weights(weights, config, layer, false);
  if (config.mixer(layer) != MixerKind::DeltaNet) {
    throw std::invalid_argument("layer is not DeltaNet");
  }
  impl_ = std::make_unique<Impl>(context, weights, config, layer, input, output, compiled,
                                 deferred_commit);
  reset();
}

DeltaNet::~DeltaNet() = default;

TensorSpec DeltaNet::recurrent_spec() const {
  if (!impl_) {
    throw std::logic_error("closed DeltaNet");
  }
  return impl_->config.delta.recurrent();
}

TensorSpec DeltaNet::convolution_spec() const {
  if (!impl_) {
    throw std::logic_error("closed DeltaNet");
  }
  return impl_->config.delta.convolution();
}

void DeltaNet::reset(const std::vector<std::uint8_t> &recurrent,
                     const std::vector<std::uint8_t> &convolution) {
  if (!impl_ || impl_->pending_commit) {
    throw std::logic_error("reset requires a live DeltaNet without a pending state commit");
  }
  if ((!recurrent.empty() && recurrent.size() != recurrent_spec().bytes()) ||
      (!convolution.empty() && convolution.size() != convolution_spec().bytes())) {
    throw std::invalid_argument("DeltaNet initial state size mismatch");
  }
  impl_->failed = true;
  const auto state =
      recurrent.empty() ? std::vector<std::uint8_t>(recurrent_spec().bytes()) : recurrent;
  const auto history =
      convolution.empty() ? std::vector<std::uint8_t>(convolution_spec().bytes()) : convolution;
  auto &step = impl_->compiled;
  upload_tensor(step.execution.graph, step.recurrent.id, state);
  upload_tensor(step.execution.graph, step.convolution.id, history);
  impl_->completed_steps = 0;
  impl_->failed = false;
}

void DeltaNet::step() {
  if (!impl_ || impl_->failed || impl_->pending_commit) {
    throw std::logic_error("DeltaNet must be initialized/reset before stepping");
  }
  impl_->failed = true;
  impl_->compiled.run();
  impl_->pending_commit = !impl_->compiled.commit;
  ++impl_->completed_steps;
  impl_->failed = false;
}

TensorBinding DeltaNet::output() {
  if (!impl_ || impl_->compiled.commit || impl_->failed) {
    throw std::logic_error("direct DeltaNet output requires compiled deferred state commit");
  }
  return {impl_->compiled.execution.graph, impl_->compiled.outputs.at("output").id};
}

std::vector<TensorCopyBinding> DeltaNet::state_copies() {
  if (!impl_ || impl_->compiled.commit || impl_->failed) {
    throw std::logic_error("external state copies require compiled deferred state commit");
  }
  auto &step = impl_->compiled;
  return {{{step.execution.graph, step.outputs.at("recurrent").id},
           {step.execution.graph, step.recurrent.id},
           recurrent_spec()},
          {{step.execution.graph, step.outputs.at("convolution").id},
           {step.execution.graph, step.convolution.id},
           convolution_spec()}};
}

void DeltaNet::confirm_state_commit() {
  if (!impl_ || impl_->compiled.commit || impl_->failed || !impl_->pending_commit) {
    throw std::logic_error("DeltaNet has no pending external state commit");
  }
  impl_->pending_commit = false;
}

std::vector<std::uint8_t> DeltaNet::read(const std::string &name, TensorReadback method) {
  if (!impl_ || impl_->failed || impl_->pending_commit || impl_->completed_steps == 0) {
    throw std::logic_error("DeltaNet has no completed output");
  }
  auto &step = impl_->compiled;
  return read_tensor(step.execution.graph, step.outputs.at(name).id, method);
}

std::size_t DeltaNet::steps() const { return impl_ ? impl_->completed_steps : 0; }

void DeltaNet::close() {
  if (impl_) {
    impl_->compiled.close();
    impl_.reset();
  }
}
} // namespace specferry::models::qwen3_5
