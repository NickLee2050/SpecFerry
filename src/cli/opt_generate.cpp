#include "cli/command.hpp"
#include "models/opt/execution.hpp"

int main(int argc, char **argv) {
  return specferry::cli::command(
      argc, 4, "usage: specferry_opt_generate DEPLOYMENT REQUEST OUTPUT",
      [&] { return specferry::models::opt::execute({argv[1], argv[2], argv[3]}); });
}
