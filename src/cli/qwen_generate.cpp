#include "cli/command.hpp"
#include "models/qwen3_5/execution.hpp"

int main(int argc, char **argv) {
  return specferry::cli::command(
      argc, 5, "usage: specferry_qwen_generate DEPLOYMENT MIXERS REQUEST OUTPUT",
      [&] { return specferry::models::qwen3_5::execute({argv[1], argv[3], argv[4]}, argv[2]); });
}
