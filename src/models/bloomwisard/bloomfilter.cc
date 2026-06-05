// Counting Bloom filter for Bloom WiSARD RAM nodes.
// Stores integer counters instead of bits, enabling bleaching (threshold-based voting).
// Supports MurmurHash3 double-hashing, SimHash LSH, and H3 universal hashing
// (Carter & Wegman, 1979) — the hash family used by BTHOWeN (Susskind et al., PACT 2022).

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

  // Initialise H3 random constants (numHashes rows, keyLength columns) for a given
  // RAM tuple size. Each hash is XOR over the columns where the bitvector key is 1,
  // reduced modulo numBits. This matches the reference BTHOWeN implementation.
  void initH3(int keyLength) {
    h3Constants.assign(numHashes, std::vector<uint64_t>(keyLength));
    for (int i = 0; i < numHashes; i++) {
      for (int j = 0; j < keyLength; j++) {
        h3Constants[i][j] = (uint64_t)((uint64_t)rand() * RAND_MAX + (uint64_t)rand());
      }
    }
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

  // In-memory footprint in bytes: the counting Bloom filter stores one int
  // counter per position (counters), plus the H3 constant matrix when in h3 mode.
  long getsizeof() const {
    long size = sizeof(BloomFilter);
    size += (long)counters.size() * sizeof(int);
    size += (long)h3Constants.size() * sizeof(std::vector<uint64_t>);
    for (const auto& row : h3Constants) size += (long)row.size() * sizeof(uint64_t);
    return size;
  }

private:
  std::vector<int> computeHashes(const std::vector<int>& key) const {
    if (hashMode == "simhash") {
      return simHasher.computeHashes(key);
    }

    if (hashMode == "h3") {
      // H3: for each hash function i, XOR all h3Constants[i][j] where key[j] == 1,
      // then reduce modulo numBits. h3Constants must be initialised via initH3().
      std::vector<int> positions(numHashes);
      for (int i = 0; i < numHashes; i++) {
        uint64_t h = 0;
        for (size_t j = 0; j < key.size(); j++) {
          if (key[j]) h ^= h3Constants[i][j];
        }
        positions[i] = (int)(h % (uint64_t)numBits);
      }
      return positions;
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
  std::vector<std::vector<uint64_t>> h3Constants;  // [numHashes][keyLength], used when hashMode == "h3"
};
