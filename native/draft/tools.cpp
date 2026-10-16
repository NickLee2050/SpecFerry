// Host tools for the framework-independent draft protocol and completion API.
#include "client.hpp"
#include <curl/curl.h>
#include <json-c/json.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Json = std::unique_ptr<json_object, decltype(&json_object_put)>;
using Options = std::map<std::string, std::string>;

Json own(json_object *value) { return Json(value, json_object_put); }

std::string encode(json_object *value) {
  return json_object_to_json_string_ext(value, JSON_C_TO_STRING_PLAIN);
}

json_object *field(json_object *object, const char *name) {
  json_object *value = nullptr;
  if (!object || !json_object_object_get_ex(object, name, &value)) {
    throw std::runtime_error(std::string("Missing JSON field: ") + name);
  }
  return value;
}

void string_field(json_object *object, const char *name, const std::string &value) {
  json_object_object_add(object, name, json_object_new_string_len(value.data(), value.size()));
}

Json parse(const std::string &text) {
  std::unique_ptr<json_tokener, decltype(&json_tokener_free)> parser(json_tokener_new(),
                                                                     json_tokener_free);
  json_tokener_set_flags(parser.get(), JSON_TOKENER_STRICT);
  auto value = own(json_tokener_parse_ex(parser.get(), text.data(), text.size()));
  if (json_tokener_get_error(parser.get()) != json_tokener_success) {
    throw std::runtime_error("Invalid JSON");
  }
  for (size_t i = json_tokener_get_parse_end(parser.get()); i < text.size(); ++i) {
    if (text[i] != ' ' && text[i] != '\t' && text[i] != '\r' && text[i] != '\n') {
      throw std::runtime_error("Trailing JSON data");
    }
  }
  return value;
}

std::string read(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot read: " + path);
  }
  if (std::filesystem::file_size(path) > 4 * 1024 * 1024) {
    throw std::runtime_error("Input exceeds 4 MiB");
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write(const std::filesystem::path &path, const std::string &value) {
  std::ofstream output(path);
  output << value;
  if (!output) {
    throw std::runtime_error("Cannot write: " + path.string());
  }
}

std::string option(const Options &options, const std::string &key,
                   const std::string &fallback = "") {
  auto found = options.find(key);
  if (found != options.end()) {
    return found->second;
  }
  if (!fallback.empty()) {
    return fallback;
  }
  throw std::runtime_error("Required option: " + key);
}

long integer(const Options &options, const std::string &key, long fallback, long low, long high) {
  const auto text = option(options, key, std::to_string(fallback));
  size_t used = 0;
  long value = std::stol(text, &used);
  if (used != text.size() || value < low || value > high) {
    throw std::runtime_error("Invalid option: " + key);
  }
  return value;
}

std::string fingerprint(const std::string &directory) {
  char hash[65];
  if (!sf_draft_tokenizer_fingerprint(directory.c_str(), hash)) {
    throw std::runtime_error(sf_draft_last_error());
  }
  return hash;
}

int probe(const Options &options) {
  const auto directory = option(options, "--model-dir");
  const auto url = option(options, "--dlm-url");
  const auto tokens = parse(read(option(options, "--prefix-file")));
  if (!json_object_is_type(tokens.get(), json_type_array)) {
    throw std::runtime_error("Prefix file must contain an integer token array");
  }
  const auto k = integer(options, "--draft-tokens", 4, 1, 32);
  const auto timeout = integer(options, "--timeout-ms", 30000, 1, 3600000);
  std::vector<int32_t> prefix;
  for (size_t i = 0; i < json_object_array_length(tokens.get()); ++i) {
    auto *token = json_object_array_get_idx(tokens.get(), i);
    if (!json_object_is_type(token, json_type_int) || json_object_get_int64(token) < 0 ||
        json_object_get_int64(token) >= 50272) {
      throw std::runtime_error("Prefix token is outside the OPT vocabulary");
    }
    prefix.push_back(json_object_get_int(token));
  }
  std::unique_ptr<void, decltype(&sf_draft_destroy)> client(
      sf_draft_create(directory.c_str(), url.c_str(), 50272, 2048, timeout, nullptr),
      sf_draft_destroy);
  if (!client) {
    throw std::runtime_error(sf_draft_last_error());
  }
  int32_t output[32];
  sf_draft_propose(client.get(), prefix.data(), prefix.size(), k, output, 32);
  auto result = parse(sf_draft_last_result(client.get()));
  string_field(result.get(), "tokenizer_sha256", fingerprint(directory));
  std::cout << encode(result.get()) << '\n';
  return std::string(json_object_get_string(field(result.get(), "status"))) == "ok" ? 0 : 1;
}

struct Stream {
  Clock::time_point started = Clock::now();
  double first = -1, last = 0;
  size_t first_count = 0;
  std::string pending, event_data, text, finish, error;
  bool done = false;
  Json usage = own(nullptr), ids = own(json_object_new_array());
  std::ofstream events;

  explicit Stream(const std::filesystem::path &directory) : events(directory / "events.jsonl") {
    if (!events) {
      throw std::runtime_error("Cannot open event log");
    }
  }

  double seconds() const { return std::chrono::duration<double>(Clock::now() - started).count(); }

  void event() {
    if (event_data.empty()) {
      return;
    }
    if (done) {
      throw std::runtime_error("SSE data after DONE");
    }
    if (event_data == "[DONE]") {
      done = true;
      event_data.clear();
      return;
    }
    auto value = parse(event_data);
    auto record = own(json_object_new_object());
    json_object_object_add(record.get(), "seconds", json_object_new_double(seconds()));
    json_object_object_add(record.get(), "event", json_object_get(value.get()));
    events << encode(record.get()) << '\n';
    json_object *problem = nullptr;
    if (json_object_object_get_ex(value.get(), "error", &problem)) {
      throw std::runtime_error("TLM returned an SSE error");
    }
    json_object *count = nullptr;
    if (json_object_object_get_ex(value.get(), "usage", &count) && count) {
      usage = own(json_object_get(count));
    }
    auto *choices = field(value.get(), "choices");
    if (!json_object_is_type(choices, json_type_array)) {
      throw std::runtime_error("Invalid SSE choices");
    }
    for (size_t i = 0; i < json_object_array_length(choices); ++i) {
      auto *choice = json_object_array_get_idx(choices, i);
      json_object *piece = nullptr;
      if (json_object_object_get_ex(choice, "text", &piece) && piece) {
        if (!json_object_is_type(piece, json_type_string)) {
          throw std::runtime_error("Invalid SSE text");
        }
        std::string delta(json_object_get_string(piece), json_object_get_string_len(piece));
        text += delta;
        std::cout << delta << std::flush;
      }
      json_object *tokens = nullptr;
      if (json_object_object_get_ex(choice, "token_ids", &tokens) && tokens) {
        if (!json_object_is_type(tokens, json_type_array)) {
          throw std::runtime_error("Invalid SSE token IDs");
        }
        auto n = json_object_array_length(tokens);
        if (n) {
          if (first < 0) {
            first = seconds();
            first_count = n;
          }
          last = seconds();
        }
        for (size_t j = 0; j < n; ++j) {
          auto *token = json_object_array_get_idx(tokens, j);
          if (!json_object_is_type(token, json_type_int) || json_object_get_int64(token) < 0 ||
              json_object_get_int64(token) >= 50272) {
            throw std::runtime_error("Invalid generated OPT token ID");
          }
          json_object_array_add(ids.get(), json_object_get(token));
        }
      }
      json_object *reason = nullptr;
      if (json_object_object_get_ex(choice, "finish_reason", &reason) && reason) {
        finish = json_object_get_string(reason);
      }
    }
    event_data.clear();
  }

  void append(const char *bytes, size_t n) {
    pending.append(bytes, n);
    size_t offset = 0, end = 0;
    while ((end = pending.find('\n', offset)) != std::string::npos) {
      auto line = pending.substr(offset, end - offset);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.empty()) {
        event();
      } else if (line.rfind("data:", 0) == 0) {
        auto data = line.substr(5);
        if (!data.empty() && data.front() == ' ') {
          data.erase(0, 1);
        }
        if (!event_data.empty()) {
          event_data += '\n';
        }
        event_data += data;
      }
      offset = end + 1;
    }
    pending.erase(0, offset);
    if (pending.size() + event_data.size() > 1024 * 1024) {
      throw std::runtime_error("SSE event exceeds 1 MiB");
    }
  }
};

size_t receive(char *data, size_t size, size_t n, void *opaque) {
  auto &stream = *static_cast<Stream *>(opaque);
  try {
    stream.append(data, size * n);
    return size * n;
  } catch (const std::exception &error) {
    stream.error = error.what();
    return 0;
  }
}

int generate(const Options &options) {
  const auto prompt = read(option(options, "--prompt-file"));
  auto base = option(options, "--base-url", "http://127.0.0.1:8011");
  while (!base.empty() && base.back() == '/') {
    base.pop_back();
  }
  const auto url = base + "/v1/completions";
  const auto limit = integer(options, "--max-tokens", 128, 1, 2047);
  const auto timeout = integer(options, "--timeout-seconds", 1800, 1, 86400);
  const std::filesystem::path directory = option(options, "--output-dir");
  if (std::filesystem::exists(directory)) {
    throw std::runtime_error("Choose a new output directory");
  }
  std::filesystem::create_directories(directory);
  auto payload = own(json_object_new_object());
  string_field(payload.get(), "model", option(options, "--model", "specferry-tlm-spec"));
  string_field(payload.get(), "prompt", prompt);
  string_field(payload.get(), "cache_salt",
               std::to_string(Clock::now().time_since_epoch().count()));
  json_object_object_add(payload.get(), "temperature", json_object_new_int(0));
  json_object_object_add(payload.get(), "max_tokens", json_object_new_int64(limit));
  json_object_object_add(payload.get(), "ignore_eos", json_object_new_boolean(false));
  json_object_object_add(payload.get(), "stream", json_object_new_boolean(true));
  json_object_object_add(payload.get(), "return_token_ids", json_object_new_boolean(true));
  auto *stream_options = json_object_new_object();
  json_object_object_add(stream_options, "include_usage", json_object_new_boolean(true));
  json_object_object_add(payload.get(), "stream_options", stream_options);
  const auto body = encode(payload.get());
  write(directory / "request.json", body + '\n');
  write(directory / "input.txt", prompt);
  auto report = own(json_object_new_object());
  string_field(report.get(), "status", "running");
  string_field(report.get(), "base_url", base);
  write(directory / "report.json", encode(report.get()) + '\n');
  Stream stream(directory);
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
  if (!curl) {
    throw std::runtime_error("Cannot create HTTP client");
  }
  std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(
      curl_slist_append(nullptr, "Content-Type: application/json"), curl_slist_free_all);
  curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
  curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 5L);
  curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, timeout);
  curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
  curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(body.size()));
  curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, receive);
  curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &stream);
  auto code = curl_easy_perform(curl.get());
  long http_status = 0;
  curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_status);
  const auto count = json_object_array_length(stream.ids.get());
  bool complete = code == CURLE_OK && http_status == 200 && stream.done && stream.usage &&
                  !stream.finish.empty() && stream.pending.empty() && stream.event_data.empty();
  if (complete) {
    auto *reported = field(stream.usage.get(), "completion_tokens");
    complete = json_object_is_type(reported, json_type_int) &&
               json_object_get_int64(reported) == static_cast<int64_t>(count);
  }
  string_field(report.get(), "status", complete ? "complete" : "failed");
  string_field(report.get(), "finish_reason", stream.finish);
  string_field(report.get(), "error", stream.error);
  json_object_object_add(report.get(), "http_status", json_object_new_int64(http_status));
  json_object_object_add(report.get(), "curl_code", json_object_new_int(code));
  json_object_object_add(report.get(), "done_received", json_object_new_boolean(stream.done));
  json_object_object_add(report.get(), "total_seconds", json_object_new_double(stream.seconds()));
  json_object_object_add(report.get(), "output_tokens", json_object_new_int64(count));
  json_object_object_add(report.get(), "ttft_seconds",
                         stream.first < 0 ? nullptr : json_object_new_double(stream.first));
  json_object_object_add(report.get(), "end_to_end_tokens_per_second",
                         json_object_new_double(count / stream.seconds()));
  json_object_object_add(
      report.get(), "decode_tokens_per_second",
      stream.last > stream.first && stream.first >= 0
          ? json_object_new_double((count - stream.first_count) / (stream.last - stream.first))
          : nullptr);
  json_object_object_add(report.get(), "usage", json_object_get(stream.usage.get()));
  json_object_object_add(report.get(), "token_ids", json_object_get(stream.ids.get()));
  write(directory / "output.txt", stream.text);
  write(directory / "report.json", encode(report.get()) + '\n');
  std::cerr << '\n'
            << (complete ? "Saved: " : "Incomplete stream; evidence: ") << directory << '\n';
  return complete ? 0 : 1;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error(
          "Usage: specferry {fingerprint|probe|generate} [--option value ...]");
    }
    const std::string mode = argv[1];
    const std::map<std::string, std::vector<std::string>> allowed = {
        {"fingerprint", {"--model-dir"}},
        {"probe", {"--model-dir", "--dlm-url", "--prefix-file", "--draft-tokens", "--timeout-ms"}},
        {"generate",
         {"--base-url", "--model", "--prompt-file", "--max-tokens", "--timeout-seconds",
          "--output-dir"}}};
    if (!allowed.count(mode)) {
      throw std::runtime_error("Unknown command: " + mode);
    }
    Options options;
    for (int i = 2; i < argc; i += 2) {
      const std::string key = argv[i];
      const auto &names = allowed.at(mode);
      if (i + 1 >= argc || std::find(names.begin(), names.end(), key) == names.end() ||
          !options.emplace(key, argv[i + 1]).second) {
        throw std::runtime_error("Invalid or duplicate option: " + key);
      }
    }
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
      throw std::runtime_error("Cannot initialize HTTP library");
    }
    if (mode == "fingerprint") {
      std::cout << fingerprint(option(options, "--model-dir")) << '\n';
      return 0;
    }
    return mode == "probe" ? probe(options) : generate(options);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
