#include "case_file.hpp"
#include "models/opt/config.hpp"
#include "models/opt/model.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/vocabulary.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void save(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) {
    throw std::runtime_error("cannot save IO observation");
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: np101_opt_io_check DEPLOYMENT FIXTURE OUTPUT\n";
    return 2;
  }
  namespace fs = std::filesystem;
  using namespace specferry::np101;
  using namespace specferry::np101::ops;
  using namespace specferry::models::opt;
  const fs::path fixture(argv[2]), output(argv[3]);
  std::unique_ptr<SdkTimings> timings;
  try {
    fs::create_directories(output);
    timings = std::make_unique<SdkTimings>(output);
    auto config = read_model_config(fixture);
    WeightStore weights(argv[1]);
    weights.verify();
    Context context;
    Vocabulary table(context, weights, weights.find("decoder.embed_tokens.weight"),
                     config.block_rows);
    InputEmbedding input(context, weights, config, table);
    Graph hidden(context, 1, 0);
    auto hidden_id = add_tensor(hidden, config.decoder.hidden_spec(), false,
                                std::vector<std::uint8_t>(config.decoder.hidden_spec().bytes()));
    OutputProjection projection(context, weights, config, {hidden, hidden_id});
    VocabularyHead head(context, table, projection.binding());
    std::ifstream cases(fixture / "cases.txt");
    std::ofstream tokens(output / "tokens.txt");
    std::int32_t token;
    unsigned position, index = 0;
    while (cases >> token >> position) {
      TimingLabel request(TimingField::Request, "case." + std::to_string(index));
      TimingLabel phase(TimingField::Phase, "selection");
      input.run(token, position);
      save(output / ("embedding." + std::to_string(index) + ".bin"), input.read());
      upload_tensor(hidden, hidden_id,
                    specferry::testing::read_bytes(fixture,
                                                   "hidden." + std::to_string(index) + ".bin",
                                                   config.decoder.hidden_spec().bytes()));
      projection.run();
      tokens << head.select() << '\n';
      auto projected = projection.binding();
      save(output / ("projected." + std::to_string(index) + ".bin"),
           read_tensor(projected.owner, projected.id));
      save(output / ("logits." + std::to_string(index) + ".bin"), head.read_logits());
      ++index;
    }
    if (!cases.eof() || !index || !tokens) {
      throw std::runtime_error("invalid IO cases or token output");
    }
    unsigned rejected = 0;
    for (auto invalid : {-1, static_cast<int>(config.vocabulary)}) {
      try {
        input.run(invalid, 0);
      } catch (const std::invalid_argument &) {
        ++rejected;
      }
    }
    try {
      input.run(0, config.decoder.capacity);
    } catch (const std::out_of_range &) {
      ++rejected;
    }
    if (rejected != 3) {
      throw std::runtime_error("IO accepted an invalid token/position");
    }
    TimingLabel phase(TimingField::Phase, "release");
    head.graph.close();
    projection.graph.close();
    hidden.close();
    input.graph.close();
    table.close();
    context.close();
    std::ofstream(output / "execution.json")
        << "{\"status\":\"executed\",\"released\":true,\"bounds_rejected\":true,\"cases\":" << index
        << "}\n";
    timings->save();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    if (timings) {
      try {
        timings->save();
      } catch (const std::exception &logging_error) {
        std::cerr << "Cannot save SDK timings: " << logging_error.what() << '\n';
      }
    }
    return 1;
  }
}
