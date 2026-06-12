#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>

namespace wp {

using addr_t = uint64_t;
using content_t = uint32_t;

constexpr unsigned DIRECT_BITS = 20;

struct PackedInputs {
  std::vector<uint64_t> data;
  size_t W = 0;
  size_t n = 0;
  uint32_t entrySize = 0;

  void init(size_t numSamples, uint32_t bits) {
    entrySize = bits;
    W = (bits + 63) / 64;
    n = numSamples;
    data.assign(numSamples * W, 0);
  }

  inline uint64_t bit(size_t s, uint32_t i) const {
    return (data[s * W + (i >> 6)] >> (i & 63)) & 1ULL;
  }

  inline void setBit(size_t s, uint32_t i) {
    data[s * W + (i >> 6)] |= (1ULL << (i & 63));
  }

  inline const uint64_t* row(size_t s) const { return data.data() + s * W; }
};

class Ram {
public:
  void init(uint32_t addressSize) {
    addressSize_ = addressSize;
    distinct_ = 0;
    direct_ = (addressSize <= DIRECT_BITS);
    if (direct_) {
      counts_.assign(size_t(1) << addressSize, 0);
    } else {
      counts_.clear();
      hash_.clear();
    }
  }

  inline void train(addr_t a) {
    if (direct_) {
      content_t& c = counts_[a];
      if (c == 0) ++distinct_;
      ++c;
    } else {
      content_t& c = hash_[a];
      if (c == 0) ++distinct_;
      ++c;
    }
  }

  inline content_t vote(addr_t a) const {
    if (direct_) return counts_[a];
    auto it = hash_.find(a);
    return it == hash_.end() ? 0 : it->second;
  }

  inline uint64_t distinctSeen() const { return distinct_; }
  inline uint32_t addressSize() const { return addressSize_; }
  inline bool isDirect() const { return direct_; }

private:
  std::vector<content_t> counts_;
  std::unordered_map<addr_t, content_t> hash_;
  uint64_t distinct_ = 0;
  uint32_t addressSize_ = 0;
  bool direct_ = false;
};

class Discriminator {
public:
  void init(uint32_t numRAMs, uint32_t addressSize) {
    numRAMs_ = numRAMs;
    addressSize_ = addressSize;
    rams_.resize(numRAMs);
    for (auto& r : rams_) r.init(addressSize);
  }

  inline void train(const PackedInputs& in, size_t s, const uint32_t* idx) {
    for (uint32_t r = 0; r < numRAMs_; ++r) {
      addr_t a = gather(in, s, idx + size_t(r) * addressSize_);
      rams_[r].train(a);
    }
  }

  inline void votes(const PackedInputs& in, size_t s, const uint32_t* idx,
                    content_t* out) const {
    for (uint32_t r = 0; r < numRAMs_; ++r) {
      addr_t a = gather(in, s, idx + size_t(r) * addressSize_);
      out[r] = rams_[r].vote(a);
    }
  }

  long modelBytes() const {
    long bytesPerAddr = (long(addressSize_) + 7) / 8;
    long total = 0;
    for (const auto& r : rams_) total += long(r.distinctSeen()) * bytesPerAddr;
    return total;
  }

  uint32_t numRAMs() const { return numRAMs_; }

private:
  inline addr_t gather(const PackedInputs& in, size_t s, const uint32_t* idx) const {
    addr_t a = 0;
    for (uint32_t k = 0; k < addressSize_; ++k) {
      a |= addr_t(in.bit(s, idx[k])) << k;
    }
    return a;
  }

  std::vector<Ram> rams_;
  uint32_t numRAMs_ = 0;
  uint32_t addressSize_ = 0;
};

struct WbinData {
  uint32_t entrySize = 0;
  uint32_t addressSize = 0;
  uint32_t numRAMs = 0;
  uint32_t numClasses = 0;
  uint64_t nTrain = 0;
  uint64_t nTest = 0;
  std::vector<uint32_t> mapping;
  std::vector<uint32_t> trainLabels;
  std::vector<uint32_t> testLabels;
  PackedInputs trainBits;
  PackedInputs testBits;
};

class Wisard {
public:
  void build(uint32_t numClasses, uint32_t numRAMs, uint32_t addressSize,
             const std::vector<uint32_t>& mapping) {
    numClasses_ = numClasses;
    numRAMs_ = numRAMs;
    addressSize_ = addressSize;
    mapping_ = mapping;
    discriminators_.resize(numClasses);
    for (auto& d : discriminators_) d.init(numRAMs, addressSize);
  }

  void train(const PackedInputs& in, const std::vector<uint32_t>& labels);

  void classify(const PackedInputs& in, std::vector<uint32_t>& out) const;

  long modelBytes() const {
    long total = 0;
    for (const auto& d : discriminators_) total += d.modelBytes();
    return total;
  }

  uint32_t numClasses() const { return numClasses_; }
  uint32_t numRAMs() const { return numRAMs_; }
  uint32_t addressSize() const { return addressSize_; }

private:
  uint32_t classifyOne(const PackedInputs& in, size_t s,
                       content_t* voteBuf) const;

  std::vector<Discriminator> discriminators_;
  std::vector<uint32_t> mapping_;
  uint32_t numClasses_ = 0;
  uint32_t numRAMs_ = 0;
  uint32_t addressSize_ = 0;
};

namespace io {
bool read(const std::string& path, WbinData& out, std::string& err);
bool write(const std::string& path, const WbinData& in, std::string& err);
}

}  // namespace wp
