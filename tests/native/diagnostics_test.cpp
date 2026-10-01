#include "inference/generation.hpp"
#include "np101/diagnostics.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace specferry::np101;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Action> void rejects(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected, "expected diagnostic error");
}

class Environment {
public:
  Environment() {
    const auto *value = std::getenv("SPECFERRY_SDK_TIMING");
    present_ = value != nullptr;
    previous_ = value ? value : "";
  }

  ~Environment() {
    if (present_) {
      setenv("SPECFERRY_SDK_TIMING", previous_.c_str(), 1);
    } else {
      unsetenv("SPECFERRY_SDK_TIMING");
    }
  }

  void set(const char *value) {
    const int status =
        value ? setenv("SPECFERRY_SDK_TIMING", value, 1) : unsetenv("SPECFERRY_SDK_TIMING");
    require(status == 0, "cannot configure timing environment");
  }

private:
  bool present_;
  std::string previous_;
};

using Table = std::map<std::array<std::string, 4>, std::vector<std::string>>;

Table read_summary(const std::filesystem::path &directory) {
  std::ifstream input(directory / "sdk-summary.tsv");
  std::string line;
  require(bool(std::getline(input, line)), "summary missing header");
  require(line == "request\tphase\trole\tapi\tcount\ttotal_seconds\tmin_seconds\tmax_seconds"
                  "\tcompleted\texceptions\tnested_count",
          "unexpected summary columns");
  Table rows;
  while (std::getline(input, line)) {
    std::vector<std::string> columns;
    std::istringstream fields(line);
    std::string field;
    while (std::getline(fields, field, '\t')) {
      columns.push_back(field);
    }
    require(columns.size() == 11, "invalid summary row");
    const auto count = std::stoull(columns[4]);
    const auto total = std::stod(columns[5]);
    const auto minimum = std::stod(columns[6]);
    const auto maximum = std::stod(columns[7]);
    require(count > 0 && minimum >= 0 && maximum >= minimum && total >= maximum &&
                total <= count * maximum + 1e-12,
            "invalid duration aggregation");
    require(std::stoull(columns[8]) + std::stoull(columns[9]) == count,
            "completed and exceptional calls must partition count");
    const std::array<std::string, 4> key{columns[0], columns[1], columns[2], columns[3]};
    require(rows.emplace(key, std::move(columns)).second, "duplicate summary key");
  }
  require(input.eof(), "cannot read summary");
  return rows;
}

void scopes_and_summary(Environment &environment, const std::filesystem::path &directory) {
  environment.set("summary");
  {
    SdkTimings timings(directory);
    rejects([&] { SdkTimings nested(directory); });
    SdkScope request("request", "r0");
    SdkScope phase("phase", "prefill");
    SdkScope role("role", "decoder");
    for (unsigned index = 0; index < 2; ++index) {
      require(sdk_call("mock-run", [] { return 7; }) == 7, "call result changed");
    }
    rejects([&] { sdk_call("mock-run", [] { throw std::runtime_error("mock failure"); }); });
    rejects([&] {
      SdkScope changed_phase("phase", "inner");
      SdkScope changed_role("role", "temporary");
      sdk_call("exception", [] { throw std::runtime_error("scope failure"); });
    });
    {
      SdkScope empty_phase("phase", "");
      SdkScope empty_role("role", "");
      sdk_call("mock-run", [] {});
    }
    sdk_call("outer", [&] {
      SdkScope nested_role("role", "nested");
      sdk_call("inner", [] {});
      rejects([&] { timings.flush(); });
    });
    rejects([&] { SdkScope invalid("unknown", "x"); });
    rejects([&] { SdkScope invalid("request", "x\ty"); });
    rejects([&] { SdkTimer invalid("x\ny"); });
    require(!std::filesystem::exists(directory / "sdk-summary.tsv"),
            "summary must not write during calls");
    timings.flush();
    // Timers may continue after an explicit flush; the next flush is cumulative.
    sdk_call("after-flush", [] {});
  }
  require(!SdkTimings::enabled(), "timing session did not deactivate");
  const auto rows = read_summary(directory);
  const auto &runs = rows.at({"r0", "prefill", "decoder", "mock-run"});
  require(rows.size() == 5 && runs[4] == "4" && runs[8] == "3" && runs[9] == "1",
          "counts or scope restoration failed");
  require(rows.at({"r0", "inner", "temporary", "exception"})[9] == "1",
          "exceptional scope was not captured");
  require(rows.at({"r0", "prefill", "nested", "inner"})[10] == "1" &&
              rows.at({"r0", "prefill", "decoder", "outer"})[10] == "0",
          "nested calls not distinguished");
  require(std::stod(rows.at({"r0", "prefill", "decoder", "outer"})[5]) >=
              std::stod(rows.at({"r0", "prefill", "nested", "inner"})[5]),
          "inclusive outer timing omitted inner call");
  require(rows.at({"r0", "prefill", "decoder", "after-flush"})[4] == "1",
          "destructor did not flush later calls");
  require(!std::filesystem::exists(directory / "sdk-calls.tsv"), "summary created per-call trace");
}

void generation_phases(Environment &environment, const std::filesystem::path &directory) {
  using namespace specferry::inference;
  environment.set("summary");
  {
    SdkTimings timings(directory);
    SdkScope phase("phase", "caller");
    GenerationExecutor executor{[] { sdk_call("reset", [] {}); },
                                [](std::int32_t) { sdk_call("consume", [] {}); },
                                [] { return sdk_call("predict", [] { return 3; }); }};
    const auto result = generate_tokens({10, 8, 2, 2}, executor, {2, 4}, 2);
    require(result.consumed == 3 && result.tokens.size() == 2,
            "instrumentation changed generation");
    sdk_call("after-generation", [] {});
    executor.prefill = [](const std::vector<std::int32_t> &) { sdk_call("block-prefill", [] {}); };
    generate_tokens({10, 8, 2, 2}, executor, {2, 4}, 1);
    timings.flush();
  }
  const auto rows = read_summary(directory);
  require(rows.at({"", "reset", "", "reset"})[4] == "2" &&
              rows.at({"", "prefill", "", "consume"})[4] == "2" &&
              rows.at({"", "decode", "", "consume"})[4] == "1" &&
              rows.at({"", "prefill", "", "predict"})[4] == "2" &&
              rows.at({"", "decode", "", "predict"})[4] == "1" &&
              rows.at({"", "prefill", "", "block-prefill"})[4] == "1" &&
              rows.at({"", "caller", "", "after-generation"})[4] == "1",
          "generation phase classification failed");
}

void calls_and_errors(Environment &environment, const std::filesystem::path &directory) {
  environment.set("calls");
  {
    SdkTimings timings(directory);
    sdk_call("returned", [] {});
    rejects([&] { sdk_call("exception", [] { throw std::runtime_error("mock failure"); }); });
    timings.flush();
  }
  std::ifstream input(directory / "sdk-calls.tsv");
  std::string line;
  std::getline(input, line);
  require(line == "event\tapi\tseconds\tunix_seconds\toutcome", "calls schema changed");
  unsigned begins = 0, ends = 0, returned = 0, exceptions = 0;
  while (std::getline(input, line)) {
    begins += line.find("begin\t") == 0;
    ends += line.find("end\t") == 0;
    const auto outcome = line.substr(line.rfind('\t') + 1);
    returned += outcome == "returned";
    exceptions += outcome == "exception";
  }
  require(begins == 2 && ends == 2 && returned == 1 && exceptions == 1,
          "call begin/end or outcome records missing");
  environment.set("invalid");
  rejects([&] { SdkTimings invalid(directory); });
  environment.set("summary");
  {
    SdkTimings unwritable(directory / "missing");
    sdk_call("mock", [] {});
    rejects([&] { unwritable.flush(); });
    // Destructor must suppress the same I/O failure and deactivate the session.
  }
  require(!SdkTimings::enabled(), "failed flush kept timing session active");
}

void disabled(Environment &environment, const std::filesystem::path &directory) {
  for (const auto *mode : {static_cast<const char *>(nullptr), "off"}) {
    environment.set(mode);
    {
      SdkTimings timings(directory);
      SdkScope phase("phase", "disabled");
      SdkTimer timer("direct-disabled");
      require(!SdkTimings::enabled() && sdk_call("mock", [] { return 9; }) == 9,
              "disabled timer changed execution");
      timings.flush();
    }
  }
  require(std::filesystem::is_empty(directory), "disabled timing wrote a file");
}
} // namespace

int main() {
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root =
      std::filesystem::temp_directory_path() / ("specferry-sdk-timing-" + std::to_string(unique));
  try {
    Environment environment;
    for (const auto *name : {"disabled", "summary", "generation", "calls"}) {
      std::filesystem::create_directories(root / name);
    }
    disabled(environment, root / "disabled");
    scopes_and_summary(environment, root / "summary");
    generation_phases(environment, root / "generation");
    calls_and_errors(environment, root / "calls");
    std::filesystem::remove_all(root);
    std::cout << "SDK timing scopes, aggregation, unwind, nesting, output and generation passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << " (evidence: " << root << ")\n";
    return 1;
  }
}
