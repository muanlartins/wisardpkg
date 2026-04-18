// Locality-Sensitive Hashing alternatives for Bloom WiSARD.
// SimHash: generates a binary signature from random hyperplanes, then uses
// MurmurHash3 with different seeds to produce independent Bloom filter positions.

#ifndef LSH_H
#define LSH_H

#include <vector>
#include <cstdlib>
#include <cmath>

class SimHasher {
public:
  SimHasher() : numHashes(0), inputDim(0), numBits(0), numHyperplanes(0) {}

  SimHasher(int numBits, int numHashes, int inputDim)
    : numBits(numBits), numHashes(numHashes), inputDim(inputDim) {
    // Number of hyperplanes determines the SimHash signature length.
    // Use enough hyperplanes for a meaningful signature (at least 64 bits).
    numHyperplanes = std::max(64, inputDim * 2);

    // Generate random hyperplanes for SimHash signature.
    hyperplanes.resize(numHyperplanes);
    for (int h = 0; h < numHyperplanes; h++) {
      hyperplanes[h].resize(inputDim);
      for (int d = 0; d < inputDim; d++) {
        hyperplanes[h][d] = (rand() % 2 == 0) ? 1.0 : -1.0;
      }
    }
  }

  // Compute hash positions using SimHash signature + MurmurHash3.
  // 1. Compute a binary SimHash signature from the input (preserves similarity).
  // 2. Hash the signature with MurmurHash3 using different seeds for each position.
  // This produces independent positions while preserving locality sensitivity.
  std::vector<int> computeHashes(const std::vector<int>& key) const {
    // Step 1: Compute SimHash binary signature
    // Each bit is the sign of dot(key, hyperplane[h])
    int sigBytes = (numHyperplanes + 7) / 8;
    std::vector<uint8_t> signature(sigBytes, 0);

    for (int h = 0; h < numHyperplanes; h++) {
      double dot = 0.0;
      int dimLimit = std::min((int)key.size(), (int)hyperplanes[h].size());
      for (int d = 0; d < dimLimit; d++) {
        dot += key[d] * hyperplanes[h][d];
      }
      if (dot >= 0) {
        signature[h / 8] |= (1 << (h % 8));
      }
    }

    // Step 2: Hash the signature with MurmurHash3, different seed per position
    std::vector<int> positions(numHashes);
    for (int i = 0; i < numHashes; i++) {
      uint32_t hashVal = MurmurHash3_32(signature.data(), sigBytes, (uint32_t)(i + 1));
      positions[i] = (int)(hashVal % (uint32_t)numBits);
    }
    return positions;
  }

private:
  int numBits;
  int numHashes;
  int inputDim;
  int numHyperplanes;
  std::vector<std::vector<double>> hyperplanes;
};

#endif // LSH_H
