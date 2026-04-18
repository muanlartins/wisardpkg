// Counting Bloom filter for Bloom WiSARD RAM nodes.
// Stores integer counters instead of bits, enabling bleaching (threshold-based voting).
// Supports MurmurHash3 double-hashing and SimHash LSH.

class BloomFilter {
public:
  BloomFilter() : numBits(0), numHashes(0), hashMode("murmur") {}

  BloomFilter(int numBits, int numHashes, const std::string& hashMode = "murmur")
    : numBits(numBits), numHashes(numHashes), hashMode(hashMode) {
    counters.resize(numBits, 0);
  }

  void initSimHash(int inputDim) {
    simHasher = SimHasher(numBits, numHashes, inputDim);
  }

  void add(const std::vector<int>& key) {
    std::vector<int> positions = computeHashes(key);
    for (int pos : positions) {
      counters[pos]++;
    }
  }

  bool query(const std::vector<int>& key) const {
    std::vector<int> positions = computeHashes(key);
    for (int pos : positions) {
      if (counters[pos] == 0) return false;
    }
    return true;
  }

  // Minimum counter value across hash positions (counting Bloom filter semantics).
  // This is the best estimate of how many times the key was inserted.
  int getMinCount(const std::vector<int>& key) const {
    std::vector<int> positions = computeHashes(key);
    int minVal = counters[positions[0]];
    for (int i = 1; i < (int)positions.size(); i++) {
      if (counters[positions[i]] < minVal) {
        minVal = counters[positions[i]];
      }
    }
    return minVal;
  }

  // Count how many of the k hash positions are non-zero (for soft matching).
  int count(const std::vector<int>& key) const {
    std::vector<int> positions = computeHashes(key);
    int c = 0;
    for (int pos : positions) {
      if (counters[pos] > 0) c++;
    }
    return c;
  }

  void reset() {
    std::fill(counters.begin(), counters.end(), 0);
  }

  int getNumBits() const { return numBits; }
  int getNumHashes() const { return numHashes; }

private:
  std::vector<int> computeHashes(const std::vector<int>& key) const {
    if (hashMode == "simhash") {
      return simHasher.computeHashes(key);
    }

    // MurmurHash3 double-hashing: h(i) = (h1 + i * h2) % numBits
    // Pack key into bytes for hashing.
    int keyBytes = (int)(key.size() * sizeof(int));
    uint32_t h1 = MurmurHash3_32(key.data(), keyBytes, 0);
    uint32_t h2 = MurmurHash3_32(key.data(), keyBytes, h1);

    std::vector<int> positions(numHashes);
    for (int i = 0; i < numHashes; i++) {
      positions[i] = (int)((h1 + (uint32_t)i * h2) % (uint32_t)numBits);
    }
    return positions;
  }

  int numBits;
  int numHashes;
  std::string hashMode;
  std::vector<int> counters;
  SimHasher simHasher;
};
