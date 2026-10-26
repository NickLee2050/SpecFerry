#include "np101/weights.hpp"
#include "openssl/evp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <future>
#include <iomanip>
#include <ios>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace specferry::np101 {
namespace {
std::uint64_t parse_offset(const std::string &text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("invalid weight offset or size");
  }
  return std::stoull(text);
}

bool valid_digest(const std::string &text) {
  return text.size() == 64 && text.find_first_not_of("0123456789abcdef") == std::string::npos;
}

template <class Worker> void run_verification_workers(std::size_t jobs, Worker worker) {
  const auto count =
      std::min<std::size_t>(jobs, std::min(4u, std::max(1u, std::thread::hardware_concurrency())));
  std::atomic<std::size_t> next{0};
  std::vector<std::future<void>> workers;
  for (std::size_t index = 0; index < count; ++index) {
    workers.push_back(std::async(std::launch::async, [&] { worker(next); }));
  }
  for (auto &result : workers) {
    result.get();
  }
}
} // namespace

std::string file_sha256(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  using Digest = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  Digest context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!file || !context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("cannot hash file: " + path.string());
  }
  std::array<char, 64 * 1024> block;
  while (file.read(block.data(), block.size()) || file.gcount()) {
    if (EVP_DigestUpdate(context.get(), block.data(), file.gcount()) != 1) {
      throw std::runtime_error("file SHA256 update failed");
    }
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned size = 0;
  if (!file.eof() || EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1) {
    throw std::runtime_error("file SHA256 finalization failed");
  }
  std::ostringstream encoded;
  for (unsigned index = 0; index < size; ++index) {
    encoded << std::hex << std::setw(2) << std::setfill('0') << unsigned(digest[index]);
  }
  return encoded.str();
}

void verify_files(const std::vector<FileChecksum> &files) {
  struct PendingFile {
    const FileChecksum *checksum;
    std::uintmax_t bytes;
  };

  std::vector<PendingFile> ordered;
  for (const auto &file : files) {
    if (!valid_digest(file.sha256)) {
      throw std::invalid_argument("invalid file checksum: " + file.path.string());
    }
    ordered.push_back({&file, std::filesystem::file_size(file.path)});
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto &left, const auto &right) { return left.bytes > right.bytes; });
  run_verification_workers(ordered.size(), [&](std::atomic<std::size_t> &next) {
    for (auto index = next.fetch_add(1); index < ordered.size(); index = next.fetch_add(1)) {
      const auto &file = *ordered[index].checksum;
      if (file_sha256(file.path) != file.sha256) {
        throw std::runtime_error("file checksum mismatch: " + file.path.string());
      }
    }
  });
}

WeightStore::WeightStore(const std::filesystem::path &directory) {
  auto root = std::filesystem::canonical(directory);
  data_path_ = std::filesystem::canonical(root / "weights.bin");
  auto index_path = std::filesystem::canonical(root / "weights.index");
  if (data_path_.parent_path() != root || index_path.parent_path() != root) {
    throw std::invalid_argument("weight pack paths escape the deployment directory");
  }
  std::ifstream index(index_path);
  std::string line;
  if (!std::getline(index, line) || line != "specferry-np101-weights 1") {
    throw std::invalid_argument("unsupported weight index header");
  }
  auto file_bytes = std::filesystem::file_size(data_path_);
  std::uint64_t previous_end = 0;
  std::set<std::string> names;
  while (std::getline(index, line)) {
    std::istringstream fields(line);
    std::string name, dtype, shape, offset, bytes, digest, extra;
    if (!(fields >> name >> dtype >> shape >> offset >> bytes >> digest) || fields >> extra) {
      throw std::invalid_argument("invalid weight index record");
    }
    if (name.empty() ||
        name.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-/") !=
            std::string::npos ||
        !names.insert(name).second || !valid_digest(digest)) {
      throw std::invalid_argument("invalid or duplicate weight name/digest: " + name);
    }
    auto type = parse_dtype(dtype);
    if (type != DataType::Float16 && type != DataType::Float32) {
      throw std::invalid_argument("weight pack only supports FP16/FP32");
    }
    WeightRecord record{
        name, {type, parse_shape(shape)}, parse_offset(offset), parse_offset(bytes), digest};
    auto padding = (64 - previous_end % 64) % 64;
    if (previous_end > file_bytes || padding > file_bytes - previous_end ||
        record.offset != previous_end + padding || record.offset > file_bytes ||
        record.bytes > file_bytes - record.offset || record.bytes != record.spec.bytes()) {
      throw std::invalid_argument(
          "weight index has invalid dimensions or overlapping/truncated ranges");
    }
    previous_end = record.offset + record.bytes;
    records_.push_back(record);
  }
  if (records_.empty() || previous_end != file_bytes) {
    throw std::invalid_argument("empty weight index or unaccounted payload bytes");
  }
}

const WeightRecord &WeightStore::find(const std::string &name) const {
  for (const auto &record : records_) {
    if (record.name == name) {
      return record;
    }
  }
  throw std::out_of_range("unknown weight: " + name);
}

std::vector<std::uint8_t> WeightStore::read(const WeightRecord &record,
                                            std::uint64_t relative_offset,
                                            std::size_t bytes) const {
  const auto &owned = find(record.name);
  if (&owned != &record) {
    throw std::invalid_argument("weight record does not belong to this store");
  }
  if (relative_offset > record.bytes || bytes > record.bytes - relative_offset ||
      bytes > 8 * 1024 * 1024) {
    throw std::out_of_range("weight read exceeds record or 8 MiB chunk limit");
  }
  std::ifstream input(data_path_, std::ios::binary);
  input.seekg(record.offset + relative_offset);
  std::vector<std::uint8_t> result(bytes);
  input.read(reinterpret_cast<char *>(result.data()), result.size());
  if (!input) {
    throw std::runtime_error("weight pack read failed");
  }
  return result;
}

void WeightStore::verify() const {
  using Digest = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  std::vector<const WeightRecord *> ordered;
  for (const auto &record : records_) {
    ordered.push_back(&record);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto *left, const auto *right) { return left->bytes > right->bytes; });
  run_verification_workers(ordered.size(), [&](std::atomic<std::size_t> &next) {
    // Each worker owns its stream, digest and at most 8 MiB of read storage.
    // Verification never shares a seek position or invokes an SDK operation.
    std::ifstream input(data_path_, std::ios::binary);
    Digest context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    std::vector<char> block(std::min<std::uint64_t>(8 * 1024 * 1024, ordered.front()->bytes));
    if (!input || !context) {
      throw std::runtime_error("cannot initialize weight verification");
    }
    for (auto index = next.fetch_add(1); index < ordered.size(); index = next.fetch_add(1)) {
      const auto &record = *ordered[index];
      if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("cannot initialize SHA256");
      }
      input.seekg(record.offset);
      std::uint64_t offset = 0;
      while (offset < record.bytes) {
        const auto bytes = std::min<std::uint64_t>(block.size(), record.bytes - offset);
        input.read(block.data(), bytes);
        if (!input) {
          throw std::runtime_error("weight pack read failed: " + record.name);
        }
        if (EVP_DigestUpdate(context.get(), block.data(), bytes) != 1) {
          throw std::runtime_error("SHA256 update failed");
        }
        offset += bytes;
      }
      std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
      unsigned size = 0;
      if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1) {
        throw std::runtime_error("SHA256 finalization failed");
      }
      std::ostringstream encoded;
      for (unsigned byte = 0; byte < size; ++byte) {
        encoded << std::hex << std::setw(2) << std::setfill('0') << unsigned(digest[byte]);
      }
      if (encoded.str() != record.sha256) {
        throw std::runtime_error("weight checksum mismatch: " + record.name);
      }
    }
  });
}
} // namespace specferry::np101
