// BloomWisard: WiSARD classifier using Bloom filter RAMs instead of hash maps.
// Supports MurmurHash3 (standard) and SimHash (locality-sensitive) hashing.

class BloomWisard: public ClassificationModel {
public:
  BloomWisard(nl::json c = {}) {
    nl::json value;

    value = c["classificationMethod"];
    if (value.is_null()) {
      classificationMethod = new Bleaching();
    } else {
      classificationMethod = ClassificationMethods::load(value);
    }

    value = c["verbose"];
    verbose = value.is_null() ? false : value.get<bool>();

    value = c["ignoreZero"];
    ignoreZero = value.is_null() ? false : value.get<bool>();

    base = 2;

    value = c["mappingGenerator"];
    if (value.is_null()) {
      mappingGenerator = new RandomMapping();
    } else {
      mappingGenerator = MappingGeneratorHelper::load(value);
    }

    value = c["monoMapping"];
    mappingGenerator->monoMapping = value.is_null() ? false : value.get<bool>();

    value = c["completeAddressing"];
    mappingGenerator->completeAddressing = value.is_null() ? true : value.get<bool>();

    numBits = 1024;
    numHashes = 3;
    hashMode = "murmur";
  }

  BloomWisard(unsigned int addressSize, int numBits, int numHashes, nl::json c = {})
    : BloomWisard(c) {
    mappingGenerator->setTupleSize(addressSize);
    this->numBits = numBits;
    this->numHashes = numHashes;
  }

  ~BloomWisard() {
    discriminators.clear();
  }

  void train(const DataSet& dataset) {
    for (size_t i = 0; i < dataset.size(); i++) {
      if (verbose) std::cout << "\rtraining " << i + 1 << " of " << dataset.size();
      if (discriminators.find(dataset.getLabel(i)) == discriminators.end()) {
        makeDiscriminator(dataset.getLabel(i), dataset[i].size());
      }
      discriminators[dataset.getLabel(i)].train(dataset[i]);
    }
  }

  std::vector<std::string> classify(const DataSet& images) const {
    std::vector<std::string> labels(images.size());
    for (unsigned int i = 0; i < images.size(); i++) {
      if (verbose) std::cout << "\rclassifying " << i + 1 << " of " << images.size();
      labels[i] = classify(images[i]);
    }
    if (verbose) std::cout << "\r" << std::endl;
    return labels;
  }

  std::string classify(const BinInput& input) const {
    std::map<std::string, int> candidates = rank(input);
    return classificationMethod->getBiggestCandidate(candidates);
  }

  std::map<std::string, int> rank(const BinInput& image) const {
    std::map<std::string, std::vector<int>> allvotes;

    for (auto& i : discriminators) {
      allvotes[i.first] = i.second.classify(image);
    }

    return classificationMethod->run(allvotes);
  }

  std::vector<std::map<std::string, int>> rank(const DataSet& images) const {
    std::vector<std::map<std::string, int>> out(images.size());
    for (unsigned int i = 0; i < images.size(); i++) {
      out[i] = rank(images[i]);
    }
    return out;
  }

  // Returns the raw per-RAM vote vector for each class, before any bleaching
  // is applied. Each vector entry is the counting-Bloom-filter min-count for
  // that RAM. Lets Python drive a custom bleach-search loop (used by BTHOWeN).
  std::map<std::string, std::vector<int>> getRawVotes(const BinInput& image) const {
    std::map<std::string, std::vector<int>> allvotes;
    for (auto& i : discriminators) {
      allvotes[i.first] = i.second.classify(image);
    }
    return allvotes;
  }

  std::vector<std::map<std::string, std::vector<int>>> getRawVotes(const DataSet& images) const {
    std::vector<std::map<std::string, std::vector<int>>> out(images.size());
    for (unsigned int i = 0; i < images.size(); i++) {
      out[i] = getRawVotes(images[i]);
    }
    return out;
  }

  // Number of RAMs per discriminator (uniform across discriminators after training).
  // Returns 0 if the model has not yet been trained.
  int getNumberOfRAMS() const {
    if (discriminators.empty()) return 0;
    return discriminators.begin()->second.getNumberOfRAMS();
  }

  int getNumBits() const { return numBits; }
  int getNumHashes() const { return numHashes; }
  std::string getHashMode() const { return hashMode; }

  void reset() {
    for (auto& d : discriminators) {
      d.second.reset();
    }
  }

  std::string json(std::string filename = "") const {
    nl::json config = {
      {"version", __version__},
      {"verbose", verbose},
      {"ignoreZero", ignoreZero},
      {"numBits", numBits},
      {"numHashes", numHashes},
      {"hashMode", hashMode}
    };
    return config.dump();
  }

  // In-memory footprint in bytes. Mirrors Wisard::getsizeof(): the model struct
  // plus, per discriminator, the label string and the discriminator's full size
  // (its BloomRAM nodes and their Bloom-filter counters). The previous version
  // returned only sizeof(BloomWisard) (~88 B), ignoring all filter storage.
  long getsizeof() const override {
    long size = sizeof(BloomWisard);
    for (const auto& d : discriminators) {
      size += (long)d.first.size() + d.second.getsizeof();
    }
    return size;
  }

  // Deployed bit-table: numRAMs * numBits, 1 bit per filter position (the
  // fixed-budget representation actually shipped). On the same yardstick as
  // Wisard/ClusWisard::deployedSizeBytes() (their lossless seen-address set).
  long deployedSizeBytes() const override {
    return ((long)getNumberOfRAMS() * (long)getNumBits() + 7) / 8;
  }

  void setHashMode(const std::string& mode) {
    hashMode = mode;
  }

protected:
  void makeDiscriminator(std::string label, int entrySize) {
    if (mappingGenerator->getEntrySize() < 2) {
      mappingGenerator->setEntrySize(entrySize);
    }
    discriminators[label] = BloomDiscriminator(
      mappingGenerator->getMapping(label), entrySize,
      numBits, numHashes, ignoreZero, base, hashMode);
  }

  std::map<std::string, BloomDiscriminator> discriminators;
  ClassificationBase* classificationMethod;
  MappingGenerator* mappingGenerator;
  bool verbose;
  bool ignoreZero;
  int base;
  int numBits;
  int numHashes;
  std::string hashMode;
};
