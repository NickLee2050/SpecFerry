// External greedy draft transport. The inference framework owns verification and KV state.
#include "client.hpp"
#include <curl/curl.h>
#include <json-c/json.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr size_t kMaxReplyBytes = 65536;
constexpr const char *kProtocol = "specferry.draft.v1";
thread_local std::string last_error;

struct JsonDeleter {
  void operator()(json_object *value) const { json_object_put(value); }
};

using Json = std::unique_ptr<json_object, JsonDeleter>;

std::string encode(json_object *value) {
  return json_object_to_json_string_ext(value, JSON_C_TO_STRING_PLAIN);
}

void set(json_object *object, const char *key, const std::string &value) {
  json_object_object_add(object, key, json_object_new_string_len(value.data(), value.size()));
}

json_object *get(json_object *object, const char *key) {
  json_object *value = nullptr;
  if (!json_object_object_get_ex(object, key, &value)) {
    throw std::runtime_error("missing_reply_field");
  }
  return value;
}

std::string hex(const unsigned char *data, size_t size) {
  const char *digits = "0123456789abcdef";
  std::string result(size * 2, '\0');
  for (size_t i = 0; i < size; ++i) {
    result[2 * i] = digits[data[i] >> 4];
    result[2 * i + 1] = digits[data[i] & 15];
  }
  return result;
}

std::string digest(const std::string &data) {
  unsigned char result[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(data.data()), data.size(), result);
  return hex(result, sizeof(result));
}

std::string tokenizer_digest(const std::string &directory) {
  std::string bytes;
  for (const auto *name : {"vocab.json", "merges.txt"}) {
    std::ifstream input(directory + "/" + name, std::ios::binary);
    if (!input) {
      throw std::runtime_error("missing_opt_tokenizer_files");
    }
    std::string data{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    bytes += name;
    bytes += '\0';
    uint64_t length = data.size();
    for (int shift = 56; shift >= 0; shift -= 8) {
      bytes += static_cast<char>((length >> shift) & 255);
    }
    bytes += data;
  }
  return digest(bytes);
}

size_t receive(char *data, size_t size, size_t count, void *opaque) {
  auto &body = *static_cast<std::string *>(opaque);
  if (size && count > (kMaxReplyBytes - body.size()) / size) {
    return 0;
  }
  body.append(data, size * count);
  return size * count;
}

Json parse(const std::string &body) {
  std::unique_ptr<json_tokener, decltype(&json_tokener_free)> tokener(json_tokener_new(),
                                                                      json_tokener_free);
  json_tokener_set_flags(tokener.get(), JSON_TOKENER_STRICT);
  Json result(json_tokener_parse_ex(tokener.get(), body.data(), body.size()));
  if (json_tokener_get_error(tokener.get()) != json_tokener_success ||
      !json_object_is_type(result.get(), json_type_object)) {
    throw std::runtime_error("invalid_reply_json");
  }
  for (size_t i = json_tokener_get_parse_end(tokener.get()); i < body.size(); ++i) {
    if (body[i] != ' ' && body[i] != '\n' && body[i] != '\r' && body[i] != '\t') {
      throw std::runtime_error("trailing_reply_data");
    }
  }
  return result;
}

class Client {
public:
  Client(const std::string &directory, const std::string &url, int vocab_size, int max_length,
         long timeout_ms, const std::string &audit_path)
      : url_(url), vocab_size_(vocab_size), max_length_(max_length), timeout_ms_(timeout_ms),
        audit_path_(audit_path), tokenizer_sha256_(tokenizer_digest(directory)) {
    static std::once_flag initialized;
    std::call_once(initialized, []() {
      if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        throw std::runtime_error("curl_initialization_failed");
      }
    });
    if (vocab_size <= 0 || max_length <= 0 || timeout_ms <= 0) {
      throw std::runtime_error("invalid_client_limits");
    }
    std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> parsed(curl_url(), curl_url_cleanup);
    if (curl_url_set(parsed.get(), CURLUPART_URL, url.c_str(), 0) != CURLUE_OK) {
      throw std::runtime_error("invalid_device_url");
    }
    char *scheme = nullptr;
    if (curl_url_get(parsed.get(), CURLUPART_SCHEME, &scheme, 0) != CURLUE_OK) {
      throw std::runtime_error("missing_device_url_scheme");
    }
    std::string protocol(scheme);
    curl_free(scheme);
    if (protocol != "http" && protocol != "https") {
      throw std::runtime_error("unsupported_device_url_scheme");
    }
    char *user = nullptr;
    if (curl_url_get(parsed.get(), CURLUPART_USER, &user, 0) == CURLUE_OK) {
      curl_free(user);
      throw std::runtime_error("embedded_device_credentials_forbidden");
    }
    curl_.reset(curl_easy_init());
    if (!curl_) {
      throw std::runtime_error("curl_handle_failed");
    }
  }

  int propose(const int32_t *prefix, size_t length, size_t k, int32_t *output, size_t capacity) {
    Json audit(json_object_new_object());
    json_object_object_add(audit.get(), "prefix_tokens", json_object_new_int64(length));
    json_object_object_add(audit.get(), "requested_tokens", json_object_new_int64(k));
    auto started = std::chrono::steady_clock::now();
    int count = 0;
    try {
      if (!prefix || !output || length == 0 || length > static_cast<size_t>(max_length_) ||
          k > 32 || k > capacity || length + k > static_cast<size_t>(max_length_)) {
        throw std::runtime_error("invalid_prefix_or_draft_limit");
      }
      Json request(json_object_new_object());
      auto *tokens = json_object_new_array_ext(length);
      json_object_object_add(request.get(), "prefix_token_ids", tokens);
      for (size_t i = 0; i < length; ++i) {
        if (prefix[i] < 0 || prefix[i] >= vocab_size_) {
          throw std::runtime_error("invalid_prefix_token");
        }
        json_object_array_add(tokens, json_object_new_int(prefix[i]));
      }
      unsigned char random[16];
      if (RAND_bytes(random, sizeof(random)) != 1) {
        throw std::runtime_error("request_id_generation_failed");
      }
      set(request.get(), "protocol", kProtocol);
      set(request.get(), "request_id", hex(random, sizeof(random)));
      set(request.get(), "prefix_sha256", digest(encode(tokens)));
      set(request.get(), "tokenizer_sha256", tokenizer_sha256_);
      set(request.get(), "draft_mode", "greedy");
      json_object_object_add(request.get(), "max_draft_tokens", json_object_new_int64(k));
      for (const char *key : {"request_id", "prefix_sha256"}) {
        set(audit.get(), key, json_object_get_string(get(request.get(), key)));
      }
      const auto body = encode(request.get());
      std::string response;
      curl_easy_reset(curl_.get());
      curl_easy_setopt(curl_.get(), CURLOPT_URL, url_.c_str());
      curl_easy_setopt(curl_.get(), CURLOPT_PROXY, "");
      curl_easy_setopt(curl_.get(), CURLOPT_PROTOCOLS_STR, "http,https");
      curl_easy_setopt(curl_.get(), CURLOPT_TIMEOUT_MS, timeout_ms_);
      curl_easy_setopt(curl_.get(), CURLOPT_CONNECTTIMEOUT_MS, std::min(1000L, timeout_ms_));
      curl_easy_setopt(curl_.get(), CURLOPT_NOSIGNAL, 1L);
      curl_easy_setopt(curl_.get(), CURLOPT_POST, 1L);
      curl_easy_setopt(curl_.get(), CURLOPT_POSTFIELDS, body.c_str());
      curl_easy_setopt(curl_.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                       static_cast<curl_off_t>(body.size()));
      curl_easy_setopt(curl_.get(), CURLOPT_WRITEFUNCTION, receive);
      curl_easy_setopt(curl_.get(), CURLOPT_WRITEDATA, &response);
      curl_slist *raw_headers = curl_slist_append(nullptr, "Content-Type: application/json");
      if (const char *token = std::getenv("SPECFERRY_DLM_TOKEN")) {
        raw_headers =
            curl_slist_append(raw_headers, (std::string("Authorization: Bearer ") + token).c_str());
      }
      std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(raw_headers,
                                                                          curl_slist_free_all);
      curl_easy_setopt(curl_.get(), CURLOPT_HTTPHEADER, headers.get());
      json_object_object_add(audit.get(), "request_bytes", json_object_new_int64(body.size()));
      json_object_object_add(audit.get(), "prepare_seconds", json_object_new_double(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()));
      const auto transport = curl_easy_perform(curl_.get());
      const auto received = std::chrono::steady_clock::now();
      long status = 0;
      curl_easy_getinfo(curl_.get(), CURLINFO_RESPONSE_CODE, &status);
      json_object_object_add(audit.get(), "http_status", json_object_new_int64(status));
      json_object_object_add(audit.get(), "response_bytes", json_object_new_int64(response.size()));
      for (auto field : {std::make_pair("connect_seconds", CURLINFO_CONNECT_TIME),
                         std::make_pair("http_seconds", CURLINFO_TOTAL_TIME)}) {
        double seconds = 0;
        curl_easy_getinfo(curl_.get(), field.second, &seconds);
        json_object_object_add(audit.get(), field.first, json_object_new_double(seconds));
      }
      if (transport != CURLE_OK) {
        json_object_object_add(audit.get(), "curl_code", json_object_new_int(transport));
        throw std::runtime_error("device_transport_failure_or_timeout");
      }
      if (status != 200) {
        throw std::runtime_error("device_http_error");
      }
      auto reply = parse(response);
      for (const char *key :
           {"protocol", "request_id", "prefix_sha256", "tokenizer_sha256", "draft_mode"}) {
        auto *actual = get(reply.get(), key);
        if (!json_object_is_type(actual, json_type_string) ||
            std::string(json_object_get_string(actual), json_object_get_string_len(actual)) !=
                json_object_get_string(get(request.get(), key))) {
          throw std::runtime_error("stale_or_mismatched_reply");
        }
      }
      auto *draft = get(reply.get(), "draft_token_ids");
      if (!json_object_is_type(draft, json_type_array) || json_object_array_length(draft) > k) {
        throw std::runtime_error("invalid_candidate_count_or_type");
      }
      std::vector<int32_t> validated;
      for (size_t i = 0; i < json_object_array_length(draft); ++i) {
        auto *token = json_object_array_get_idx(draft, i);
        if (!json_object_is_type(token, json_type_int) || json_object_get_int64(token) < 0 ||
            json_object_get_int64(token) >= vocab_size_) {
          throw std::runtime_error("invalid_candidate_token");
        }
        validated.push_back(json_object_get_int(token));
      }
      std::copy(validated.begin(), validated.end(), output);
      count = static_cast<int>(validated.size());
      set(audit.get(), "status", "ok");
      json_object_object_add(audit.get(), "draft_token_ids", json_object_get(draft));
      // Optional remote telemetry is untrusted and never affects token decisions.
      // HTTP time includes queueing/device work; it must not be called network RTT.
      json_object *timing = nullptr;
      if (json_object_object_get_ex(reply.get(), "timing", &timing) &&
          json_object_is_type(timing, json_type_object)) {
        auto *reported = json_object_new_object();
        for (const char *field : {"draft_seconds", "synchronization_seconds", "server_seconds"}) {
          json_object *value = nullptr;
          if (json_object_object_get_ex(timing, field, &value) &&
              (json_object_is_type(value, json_type_double) ||
               json_object_is_type(value, json_type_int))) {
            double seconds = json_object_get_double(value);
            if (std::isfinite(seconds) && seconds >= 0) {
              json_object_object_add(reported, field, json_object_new_double(seconds));
            }
          }
        }
        json_object_object_add(audit.get(), "reported_dlm_timing", reported);
      }
      json_object_object_add(audit.get(), "response_validation_seconds", json_object_new_double(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - received).count()));
    } catch (const std::exception &error) {
      set(audit.get(), "status", "fallback");
      set(audit.get(), "error_type", error.what());
    }
    auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    json_object_object_add(audit.get(), "wall_seconds", json_object_new_double(elapsed));
    last_result_ = encode(audit.get());
    if (!audit_path_.empty()) {
      std::ofstream stream(audit_path_, std::ios::app);
      stream << last_result_ << '\n';
    }
    return count;
  }

  const char *last_result() const { return last_result_.c_str(); }

private:
  std::string url_;
  int vocab_size_, max_length_;
  long timeout_ms_;
  std::string audit_path_, tokenizer_sha256_;
  std::string last_result_ = "{}";
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl_{nullptr, curl_easy_cleanup};
};
} // namespace

extern "C" {
void *sf_draft_create(const char *directory, const char *url, int vocab_size, int max_length,
                      long timeout_ms, const char *audit_path) {
  try {
    if (!directory || !url) {
      throw std::runtime_error("missing_client_configuration");
    }
    return new Client(directory, url, vocab_size, max_length, timeout_ms,
                      audit_path ? audit_path : "");
  } catch (const std::exception &error) {
    last_error = error.what();
    return nullptr;
  }
}

const char *sf_draft_last_error() { return last_error.c_str(); }

int sf_draft_tokenizer_fingerprint(const char *directory, char output[65]) {
  try {
    if (!directory || !output) {
      throw std::runtime_error("missing_fingerprint_arguments");
    }
    auto value = tokenizer_digest(directory);
    std::memcpy(output, value.c_str(), 65);
    return 1;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 0;
  }
}

int sf_draft_propose(void *handle, const int32_t *prefix, size_t length, size_t k, int32_t *output,
                     size_t capacity) {
  try {
    if (!handle) {
      return 0;
    }
    return static_cast<Client *>(handle)->propose(prefix, length, k, output, capacity);
  } catch (...) {
    return 0;
  }
}

const char *sf_draft_last_result(void *handle) {
  return handle ? static_cast<Client *>(handle)->last_result() : "{}";
}

void sf_draft_destroy(void *handle) { delete static_cast<Client *>(handle); }
}
