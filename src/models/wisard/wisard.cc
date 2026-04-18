class Wisard: public ClassificationModel {
public:
  Wisard(nl::json c){
    nl::json value;

    value = c["classificationMethod"];
    if(value.is_null()){
      classificationMethod = new Bleaching();
    }
    else{
      classificationMethod = ClassificationMethods::load(value);
    }

    value = c["verbose"];
    verbose = value.is_null() ? false : value.get<bool>();

    value = c["balanced"];
    balanced = value.is_null() ? false : value.get<bool>();

    value = c["ignoreZero"];
    ignoreZero = value.is_null() ? false : value.get<bool>();

    value = c["base"];
    base = value.is_null() ? 2 : value.get<int>();

    // Mapping

    value = c["mappingGenerator"];
    if(value.is_null()){
      mappingGenerator = new RandomMapping();
    }
    else{
      mappingGenerator = MappingGeneratorHelper::load(value);
    }

    value = c["indexes"];
    std::vector<int> indexes = value.is_null() ? std::vector<int>(0) : value.get<std::vector<int>>();

    if(indexes.size() > 0){
      mappingGenerator->setIndexes(indexes);
    }

    value = c["mapping"];
    std::map<std::string, std::vector<std::vector<int>>> mapping = value.is_null() ? std::map<std::string, std::vector<std::vector<int>>>() : value.get<std::map<std::string, std::vector<std::vector<int>>>>();

    if (mapping.size() > 0){
      mappingGenerator->setMappings(mapping);
    }

    value = c["monoMapping"];
    mappingGenerator->monoMapping = value.is_null() ? false : value.get<bool>();

    value = c["completeAddressing"];
    mappingGenerator->completeAddressing = value.is_null() ? true : value.get<bool>();

    softBleaching = false;
    crossClassScoring = false;
    negativeEvidence = false;
    negativeAlpha = 0.0;
    negativeMode = "uniform";
    useRAMWeights = false;
    useSharedDiscriminator = false;
    sharedBeta = 0.0;
    sharedDiscriminatorInitialized = false;
    attentionWeighting = false;
  }

  Wisard(unsigned int addressSize, nl::json c={}) : Wisard(c){
    mappingGenerator->setTupleSize(addressSize);
  }

  Wisard(std::string config) : Wisard(nl::json::parse(config)){
    nl::json c = nl::json::parse(config);

    nl::json classes = c["classes"];
    nl::json dConfig = {
      {"ignoreZero", ignoreZero},
      {"base", base}
    };
    for(nl::json::iterator it = classes.begin(); it != classes.end(); ++it){
      nl::json d = it.value();
      d.merge_patch(dConfig);
      discriminators[it.key()] = Discriminator(d);
    }
  }

  ~Wisard(){
    discriminators.clear();
  }

  void train(const DataSet& dataset) {
    for(size_t i=0; i<dataset.size(); i++){
      if(verbose) std::cout << "\rtraining " << i+1 << " of " << dataset.size();
      if(discriminators.find(dataset.getLabel(i)) == discriminators.end()){
        makeDiscriminator(dataset.getLabel(i), dataset[i].size());
      }
      discriminators[dataset.getLabel(i)].train(dataset[i]);

      if(useSharedDiscriminator){
        if(!sharedDiscriminatorInitialized){
          sharedDiscrim = Discriminator(mappingGenerator->getMapping("__shared__"), dataset[i].size(), ignoreZero, base);
          sharedDiscriminatorInitialized = true;
        }
        sharedDiscrim.train(dataset[i]);
      }
    }
  }

  std::vector<std::string> classify(const DataSet& images) const{
    std::vector<std::string> labels(images.size());

    for(unsigned int i=0; i<images.size(); i++){
      if(verbose) std::cout << "\rclassifying " << i+1 << " of " << images.size();
      labels[i] = classify(images[i]);
    }
    if(verbose) std::cout << "\r" << std::endl;
    return labels;
  }

  std::string classify(const BinInput& input) const {
    std::map<std::string,int> candidates = rank(input);
    return classificationMethod->getBiggestCandidate(candidates);
  }

  void untrain(const DataSet& images){
    for(unsigned int i=0; i<images.size(); i++){
      if(verbose) std::cout << "\runtraining " << i+1 << " of " << images.size();
        auto d = discriminators.find(images.getLabel(i));
        if(d != discriminators.end()){
          d->second.untrain(images[i]);
        }
    }
    if(verbose) std::cout << "\r" << std::endl;
  }

  void reset() {
    for (auto& d : discriminators) {
      d.second.reset();
    }
  }

  void trainSingle(const BinInput& input, const std::string& label) {
    if (discriminators.find(label) == discriminators.end()) {
      makeDiscriminator(label, input.size());
    }
    discriminators[label].train(input);

    if(useSharedDiscriminator){
      if(!sharedDiscriminatorInitialized){
        sharedDiscrim = Discriminator(mappingGenerator->getMapping("__shared__"), input.size(), ignoreZero, base);
        sharedDiscriminatorInitialized = true;
      }
      sharedDiscrim.train(input);
    }
  }

  void untrainSingle(const BinInput& input, const std::string& label) {
    auto d = discriminators.find(label);
    if (d != discriminators.end()) {
      d->second.untrain(input);
    }
  }

  nl::json getMappingJson() const {
    nl::json config;
    config["mapping"] = nl::json(mappingGenerator->getMappings());
    config["monoMapping"] = mappingGenerator->monoMapping;
    config["completeAddressing"] = mappingGenerator->completeAddressing;
    return config;
  }

  std::map<std::string,std::vector<int>> getMentalImages(){
    std::map<std::string,std::vector<int>> images;
    for(std::map<std::string, Discriminator>::iterator d=discriminators.begin(); d!=discriminators.end(); ++d){
      images[d->first] = d->second.getMentalImage();
    }
    return images;
  }

  std::string json(std::string filename="") const {
    nl::json config = {
      {"version", __version__},
      {"verbose", verbose},
      {"ignoreZero", ignoreZero},
      {"classificationMethod", ClassificationMethods::json(classificationMethod)},
      {"mappingGenerator", MappingGeneratorHelper::json(mappingGenerator)},
      {"base", base}
    };
    nl::json c;
    bool isSave = filename.size() > 0;
    for(auto& d : discriminators){
      c[d.first] = d.second.getJson(isSave);
    }
    config["classes"] = c;
    if(isSave){
      std::string outfile = filename + config_sufix;
      std::ofstream dataFile;
      dataFile.open(outfile, std::ios::app);
      dataFile << config.dump();
      dataFile.close();
      return filename;
    }
    return config.dump();
  }

  std::map<std::string, int> rank(const BinInput& image) const{
    std::map<std::string,std::vector<int>> allvotes;

    float totalTrainned = 0;
    if(balanced){
      for(auto& i: discriminators){
        totalTrainned += i.second.getNumberOfTrainings();
      }
    }

    for(auto& i: discriminators){
      allvotes[i.first] = i.second.classify(image,totalTrainned);
    }

    // Multi-resolution: reweight votes by RAM address size before Bleaching.
    if(mappingGenerator->multiResolution){
      for(auto& entry: allvotes){
        auto it = discriminators.find(entry.first);
        if(it != discriminators.end()){
          std::vector<int> sizes = it->second.getTupleSizes();
          for(size_t j = 0; j < entry.second.size() && j < sizes.size(); j++){
            entry.second[j] *= sizes[j];
          }
        }
      }
    }

    // Weighted RAM voting: multiply each RAM's vote by its learned weight.
    if(useRAMWeights){
      for(auto& entry: allvotes){
        auto wit = ramWeights.find(entry.first);
        if(wit != ramWeights.end()){
          for(size_t j = 0; j < entry.second.size() && j < wit->second.size(); j++){
            entry.second[j] = (int)(entry.second[j] * wit->second[j]);
          }
        }
      }
    }

    // Attention-like weighting: weight each RAM by agreement with consensus.
    if(attentionWeighting){
      for(auto& entry: allvotes){
        std::vector<int>& votes = entry.second;
        double mean = 0.0;
        double maxVal = 0.0;
        for(size_t j = 0; j < votes.size(); j++){
          mean += votes[j];
          if(votes[j] > maxVal) maxVal = votes[j];
        }
        if(votes.size() > 0) mean /= votes.size();

        for(size_t j = 0; j < votes.size(); j++){
          double w = 1.0 - std::abs((double)votes[j] - mean) / (maxVal + 1e-9);
          votes[j] = (int)(votes[j] * w);
        }
      }
    }

    // Cross-class scoring: normalize each RAM's vote by the total across all classes.
    if(crossClassScoring){
      size_t n_rams = 0;
      for(auto& entry: allvotes){ n_rams = std::max(n_rams, entry.second.size()); }
      int n_classes = (int)allvotes.size();

      for(size_t j = 0; j < n_rams; j++){
        int total = 0;
        for(auto& entry: allvotes){
          if(j < entry.second.size()) total += entry.second[j];
        }
        if(total > 0){
          for(auto& entry: allvotes){
            if(j < entry.second.size()){
              entry.second[j] = entry.second[j] * n_classes / (total + 1);
            }
          }
        }
      }
    }

    // Compute per-class totals via softBleaching or classification method.
    std::map<std::string, int> labels;

    if(softBleaching){
      int bleaching = 0;
      bool looping = true;

      while(looping){
        int min = 0;
        bool firstTime = true;

        for(auto& entry: allvotes){
          labels[entry.first] = 0;
          for(size_t j = 0; j < entry.second.size(); j++){
            if(entry.second[j] > bleaching){
              labels[entry.first] += (entry.second[j] - bleaching);
              if(firstTime || entry.second[j] < min){
                min = entry.second[j];
                firstTime = false;
              }
            }
          }
        }

        bleaching = min;

        int biggest = 0;
        bool ambiguity = false;
        for(auto& l: labels){
          if(l.second > biggest){ biggest = l.second; ambiguity = false; }
          else if((biggest - l.second) < 1){ ambiguity = true; }
        }

        looping = ambiguity && biggest > 1;
      }
    }
    else{
      labels = classificationMethod->run(allvotes);
    }

    // Shared discriminator: subtract background response from all class scores.
    if(useSharedDiscriminator && sharedDiscriminatorInitialized){
      std::vector<int> sharedVotes = sharedDiscrim.classify(image);
      int sharedResponse = 0;
      for(size_t j = 0; j < sharedVotes.size(); j++){
        if(sharedVotes[j] > 0) sharedResponse++;
      }
      for(auto& l: labels){
        l.second = (int)(l.second - sharedBeta * sharedResponse);
      }
    }

    // Negative evidence: penalize each class based on other classes' responses.
    if(negativeEvidence){
      int totalAll = 0;
      for(auto& l: labels) totalAll += l.second;

      std::map<std::string, int> adjusted;
      for(auto& l: labels){
        int selfScore = l.second;
        int othersSum = totalAll - selfScore;

        if(negativeMode == "uniform"){
          adjusted[l.first] = (int)(selfScore - negativeAlpha * othersSum);
        }
        else if(negativeMode == "max_competitor"){
          int maxComp = 0;
          for(auto& o: labels){
            if(o.first != l.first && o.second > maxComp) maxComp = o.second;
          }
          adjusted[l.first] = (int)(selfScore - negativeAlpha * maxComp);
        }
        else{ // normalized
          adjusted[l.first] = (int)(selfScore * 1000.0 / (1.0 + negativeAlpha * othersSum));
        }
      }
      labels = adjusted;
    }

    return labels;
  }

  std::vector<std::map<std::string, int>> rank(const DataSet& images) const{
    std::vector<std::map<std::string, int>> out(images.size());

    for(unsigned int i=0; i<images.size(); i++){
        out[i] = rank(images[i]);
    }
    return out;
  }

  // Compute per-RAM quality weights from training data.
  // Metrics: "entropy", "information_gain", "purity"
  void computeRAMWeights(const DataSet& dataset, const std::string& metric = "entropy"){
    // For each discriminator, track per-RAM activation counts per class.
    // activations[discrim_label][ram_idx][sample_class] = count of times RAM fired
    std::map<std::string, std::vector<std::map<std::string, int>>> activations;
    std::map<std::string, std::vector<int>> totalActivations; // total fires per RAM per discrim

    for(auto& d: discriminators){
      int nRAMs = d.second.getNumberOfRAMS();
      activations[d.first].resize(nRAMs);
      totalActivations[d.first].resize(nRAMs, 0);
    }

    // Pass each training sample through all discriminators.
    for(size_t i = 0; i < dataset.size(); i++){
      const std::string& sampleClass = dataset.getLabel(i);
      for(auto& d: discriminators){
        std::vector<int> votes = d.second.classify(dataset[i]);
        for(size_t j = 0; j < votes.size(); j++){
          if(votes[j] > 0){
            activations[d.first][j][sampleClass]++;
            totalActivations[d.first][j]++;
          }
        }
      }
    }

    // Compute weights from activation patterns.
    ramWeights.clear();
    int nClasses = (int)discriminators.size();

    for(auto& d: discriminators){
      int nRAMs = d.second.getNumberOfRAMS();
      ramWeights[d.first].resize(nRAMs, 1.0);

      for(int j = 0; j < nRAMs; j++){
        int total = totalActivations[d.first][j];
        if(total == 0){
          ramWeights[d.first][j] = 0.0;
          continue;
        }

        if(metric == "entropy"){
          // Weight = 1 - normalized entropy of class distribution of activations
          double entropy = 0.0;
          for(auto& ac: activations[d.first][j]){
            double p = (double)ac.second / total;
            if(p > 0) entropy -= p * std::log2(p);
          }
          double maxEntropy = (nClasses > 1) ? std::log2((double)nClasses) : 1.0;
          ramWeights[d.first][j] = 1.0 - entropy / maxEntropy;
        }
        else if(metric == "information_gain"){
          // MI between binary RAM output (fires/doesn't) and class label
          double nSamples = (double)dataset.size();
          double pFire = total / nSamples;
          double pNoFire = 1.0 - pFire;
          double mi = 0.0;

          for(auto& ac: activations[d.first][j]){
            // Count total samples of this class
            int classTotal = 0;
            for(size_t s = 0; s < dataset.size(); s++){
              if(dataset.getLabel(s) == ac.first) classTotal++;
            }
            double pClass = classTotal / nSamples;
            // p(fire, class)
            double pJoint = ac.second / nSamples;
            if(pJoint > 0 && pFire > 0 && pClass > 0){
              mi += pJoint * std::log2(pJoint / (pFire * pClass));
            }
            // p(no_fire, class)
            double pJointNo = (classTotal - ac.second) / nSamples;
            if(pJointNo > 0 && pNoFire > 0 && pClass > 0){
              mi += pJointNo * std::log2(pJointNo / (pNoFire * pClass));
            }
          }
          ramWeights[d.first][j] = mi;
        }
        else if(metric == "purity"){
          // Fraction of activations belonging to this discriminator's class
          auto it = activations[d.first][j].find(d.first);
          int correctCount = (it != activations[d.first][j].end()) ? it->second : 0;
          ramWeights[d.first][j] = (double)correctCount / total;
        }
      }
    }

    // Normalize weights to [0, 1] range per discriminator
    for(auto& entry: ramWeights){
      double maxW = 0.0;
      for(double w: entry.second) if(w > maxW) maxW = w;
      if(maxW > 0){
        for(double& w: entry.second) w /= maxW;
      }
    }

    useRAMWeights = true;
  }

  void setRAMWeights(const std::map<std::string, std::vector<double>>& weights){
    ramWeights = weights;
    useRAMWeights = true;
  }

  std::map<std::string, std::vector<double>> getRAMWeights() const{
    return ramWeights;
  }

  void pruneRAMs(double threshold){
    for(auto& entry: ramWeights){
      for(size_t i = 0; i < entry.second.size(); i++){
        if(entry.second[i] < threshold) entry.second[i] = 0.0;
      }
    }
  }

  std::map<std::string,std::vector<int>> getTupleSizes() const{
    std::map<std::string,std::vector<int>> sizes;

    for(auto& i: discriminators){
      sizes[i.first] = i.second.getTupleSizes();
    }
    return sizes;
  }

  long getsizeof() const{
    long size = sizeof(Wisard);
    for(auto& d: discriminators){
      size += d.first.size() + d.second.getsizeof();
    }
    return size;
  }

protected:
  void makeDiscriminator(std::string label, int entrySize){
    if (mappingGenerator->getEntrySize() < 2){
      mappingGenerator->setEntrySize(entrySize);
    }
    discriminators[label] = Discriminator(mappingGenerator->getMapping(label), entrySize, ignoreZero, base);
  }

  void checkInputSizes(const int imageSize, const int labelsSize){
    if(imageSize != labelsSize){
      throw Exception("The size of data is not the same of the size of labels!");
    }
  }

  std::map<std::string, Discriminator> discriminators;
  ClassificationBase* classificationMethod;
  MappingGenerator* mappingGenerator;
  bool verbose;
  bool ignoreZero;
  int base;
  bool balanced;
  bool softBleaching;
  bool crossClassScoring;

  // Negative evidence
  bool negativeEvidence;
  double negativeAlpha;
  std::string negativeMode;

  // Weighted RAM voting
  std::map<std::string, std::vector<double>> ramWeights;
  bool useRAMWeights;

  // Shared discriminator
  bool useSharedDiscriminator;
  double sharedBeta;
  Discriminator sharedDiscrim;
  bool sharedDiscriminatorInitialized;

  // Attention-like weighting
  bool attentionWeighting;
};
