// BloomDiscriminator: a set of BloomRAM nodes forming one class's memory.

class BloomDiscriminator {
public:
  BloomDiscriminator() : entrySize(0), count(0) {}

  BloomDiscriminator(std::vector<std::vector<int>> mapping, int entrySize,
                     int numBits, int numHashes, bool ignoreZero = false,
                     int base = 2, const std::string& hashMode = "murmur")
    : entrySize(entrySize), count(0) {
    for (size_t i = 0; i < mapping.size(); i++) {
      rams.push_back(BloomRAM(mapping[i], numBits, numHashes, ignoreZero, base, hashMode));
      if (hashMode == "simhash") {
        rams.back().initSimHash();
      }
    }
  }

  void train(const BinInput& image) {
    count++;
    for (size_t i = 0; i < rams.size(); i++) {
      rams[i].train(image);
    }
  }

  std::vector<int> classify(const BinInput& image) const {
    std::vector<int> votes(rams.size());
    for (size_t i = 0; i < rams.size(); i++) {
      votes[i] = rams[i].getVote(image);
    }
    return votes;
  }

  void reset() {
    count = 0;
    for (size_t i = 0; i < rams.size(); i++) {
      rams[i].reset();
    }
  }

  int getNumberOfRAMS() const { return (int)rams.size(); }
  int getNumberOfTrainings() const { return count; }

private:
  std::vector<BloomRAM> rams;
  int entrySize;
  int count;
};
