// BloomRAM: a RAM node that uses a Bloom filter instead of a hash map.
// Training adds the addressed bits to the filter; classification queries membership.

class BloomRAM {
public:
  BloomRAM() : ignoreZero(false), base(2) {}

  BloomRAM(const std::vector<int>& addresses, int numBits, int numHashes,
           bool ignoreZero = false, int base = 2, const std::string& hashMode = "murmur")
    : addresses(addresses), filter(numBits, numHashes, hashMode),
      ignoreZero(ignoreZero), base(base) {}

  void initSimHash() {
    filter.initSimHash((int)addresses.size());
  }

  void train(const BinInput& image) {
    std::vector<int> key = getKey(image);
    if (ignoreZero && isZeroKey(key)) return;
    filter.add(key);
  }

  int getVote(const BinInput& image) const {
    std::vector<int> key = getKey(image);
    if (ignoreZero && isZeroKey(key)) return 0;
    return filter.getMinCount(key);
  }

  // Soft vote: count of matching hash positions (0 to numHashes).
  int getSoftVote(const BinInput& image) const {
    std::vector<int> key = getKey(image);
    if (ignoreZero && isZeroKey(key)) return 0;
    return filter.count(key);
  }

  void reset() {
    filter.reset();
  }

  int getAddressSize() const { return (int)addresses.size(); }

private:
  std::vector<int> getKey(const BinInput& image) const {
    std::vector<int> key(addresses.size());
    for (size_t i = 0; i < addresses.size(); i++) {
      key[i] = image[addresses[i]];
    }
    return key;
  }

  bool isZeroKey(const std::vector<int>& key) const {
    for (int v : key) {
      if (v != 0) return false;
    }
    return true;
  }

  std::vector<int> addresses;
  BloomFilter filter;
  bool ignoreZero;
  int base;
};
