class StochasticThermometer : public BinBase {
public:
  StochasticThermometer(const size_t thermometerSize)
    : thermometerSize(thermometerSize), fitted(false) {
    std::srand(std::time(NULL));
  }

  void fit(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return;
    size_t n_samples = data.size();
    size_t n_features = data[0].size();

    valueRanges.resize(n_features);

    for (size_t f = 0; f < n_features; f++) {
      std::vector<double> values(n_samples);
      for (size_t i = 0; i < n_samples; i++) {
        values[i] = data[i][f];
      }
      std::sort(values.begin(), values.end());

      valueRanges[f].resize(thermometerSize);
      for (size_t k = 0; k < thermometerSize; k++) {
        double percentile = (double)(k + 1) / (double)(thermometerSize + 1);
        size_t index = (size_t)(percentile * (double)(n_samples - 1));
        if (index >= n_samples) index = n_samples - 1;
        valueRanges[f][k] = values[index];
      }
    }

    fitted = true;
  }

  double optimize(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::string>& labels,
    int addressSize,
    double validationSize = 0.2,
    int rounds = 5,
    int stepsPerThreshold = 100,
    int numThreads = 0,
    double maxShiftRatio = 0.5,
    int earlyStopWindow = 0
  ) {
    if (!fitted) {
      throw Exception("StochasticThermometer must be fitted before optimize!");
    }
    if (data.size() != labels.size()) {
      throw Exception("Data and labels must have the same size!");
    }

    size_t n = data.size();
    size_t valCount = (size_t)(validationSize * n);
    if (valCount == 0) valCount = 1;
    size_t trainCount = n - valCount;

    // Shuffle indices for train/validation split
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; i++) indices[i] = i;
    for (size_t i = n - 1; i > 0; i--) {
      size_t j = std::rand() % (i + 1);
      std::swap(indices[i], indices[j]);
    }

    std::vector<std::vector<double>> trainData(trainCount), valData(valCount);
    std::vector<std::string> trainLabels(trainCount), valLabels(valCount);
    for (size_t i = 0; i < trainCount; i++) {
      trainData[i] = data[indices[i]];
      trainLabels[i] = labels[indices[i]];
    }
    for (size_t i = 0; i < valCount; i++) {
      valData[i] = data[indices[trainCount + i]];
      valLabels[i] = labels[indices[trainCount + i]];
    }

    size_t n_features = valueRanges.size();

    // Pre-binarize all samples once into BinInput (compact bitpacked form)
    std::vector<BinInput> trainBinInputs(trainCount);
    std::vector<BinInput> valBinInputs(valCount);
    for (size_t i = 0; i < trainCount; i++) {
      trainBinInputs[i] = transform(trainData[i]);
    }
    for (size_t i = 0; i < valCount; i++) {
      valBinInputs[i] = transform(valData[i]);
    }

    // Create a master Wisard with a fixed random mapping.
    Wisard masterWisard(addressSize);

    // Collect unique labels and force discriminator/mapping creation
    std::vector<std::string> uniqueLabels;
    for (size_t i = 0; i < trainCount; i++) {
      bool found = false;
      for (size_t j = 0; j < uniqueLabels.size(); j++) {
        if (uniqueLabels[j] == trainLabels[i]) { found = true; break; }
      }
      if (!found) {
        masterWisard.trainSingle(trainBinInputs[i], trainLabels[i]);
        uniqueLabels.push_back(trainLabels[i]);
      }
    }
    masterWisard.reset();

    // Extract the mapping so thread-local Wisards use the same bit-to-RAM assignment
    nl::json mappingConfig = masterWisard.getMappingJson();

    // Determine thread count
    int nThreads = numThreads > 0 ? numThreads : (int)std::thread::hardware_concurrency();
    if (nThreads < 1) nThreads = 1;

    // Pre-create thread-local Wisards with the same mapping
    std::vector<Wisard> threadWisards;
    threadWisards.reserve(nThreads);
    for (int t = 0; t < nThreads; t++) {
      threadWisards.emplace_back(addressSize, mappingConfig);
      // Force discriminator creation by training one sample per class
      for (size_t j = 0; j < uniqueLabels.size(); j++) {
        threadWisards.back().trainSingle(trainBinInputs[0], uniqueLabels[j]);
      }
      threadWisards.back().reset();
    }

    // Pre-allocate thread-local BinInput copies (reused across coordinates)
    std::vector<std::vector<BinInput>> threadTrainBins(nThreads);
    std::vector<std::vector<BinInput>> threadValBins(nThreads);

    // Initial accuracy using master Wisard
    for (size_t i = 0; i < trainCount; i++) {
      masterWisard.trainSingle(trainBinInputs[i], trainLabels[i]);
    }
    double bestAcc = classifyValidation(masterWisard, valBinInputs, valLabels);

    // Build list of all (feature, threshold_index) pairs
    std::vector<std::pair<size_t, size_t>> coords;
    for (size_t f = 0; f < n_features; f++) {
      for (size_t k = 0; k < valueRanges[f].size(); k++) {
        coords.push_back({f, k});
      }
    }

    int halfSteps = stepsPerThreshold / 2;

    for (int round = 0; round < rounds; round++) {
      // Randomize threshold processing order each round
      for (size_t i = coords.size() - 1; i > 0; i--) {
        size_t j = std::rand() % (i + 1);
        std::swap(coords[i], coords[j]);
      }

      for (size_t ci = 0; ci < coords.size(); ci++) {
        size_t f = coords[ci].first;
        size_t k = coords[ci].second;
        size_t bitPos = f * thermometerSize + k;

        double originalValue = valueRanges[f][k];

        double gapLeft = (k > 0) ? (valueRanges[f][k] - valueRanges[f][k - 1]) : valueRanges[f][k];
        double gapRight = (k < valueRanges[f].size() - 1) ? (valueRanges[f][k + 1] - valueRanges[f][k]) : gapLeft;
        double gap = std::min(gapLeft, gapRight);

        if (gap <= 0.0) continue;

        // Pre-generate all candidate thresholds (left + right)
        // maxShiftRatio controls search range: 0.5 = up to 50% of gap, 0.2 = up to 20%
        std::vector<double> candidateThresholds(stepsPerThreshold);
        for (int s = 0; s < halfSteps; s++) {
          double bandLow = (double)s / (double)halfSteps * maxShiftRatio;
          double bandHigh = (double)(s + 1) / (double)halfSteps * maxShiftRatio;
          double randInBand = bandLow + ((double)std::rand() / RAND_MAX) * (bandHigh - bandLow);
          candidateThresholds[s] = originalValue - randInBand * gap;
        }
        for (int s = 0; s < halfSteps; s++) {
          double bandLow = (double)s / (double)halfSteps * maxShiftRatio;
          double bandHigh = (double)(s + 1) / (double)halfSteps * maxShiftRatio;
          double randInBand = bandLow + ((double)std::rand() / RAND_MAX) * (bandHigh - bandLow);
          candidateThresholds[halfSteps + s] = originalValue + randInBand * gap;
        }

        // Evaluate all candidates in parallel.
        // Each thread gets its own Wisard and BinInput copies.
        std::vector<double> candidateAccuracies(stepsPerThreshold, -1.0);

        // Snapshot master BinInputs for this coordinate
        for (int t = 0; t < nThreads; t++) {
          threadTrainBins[t] = trainBinInputs;
          threadValBins[t] = valBinInputs;
        }

        // Parallel evaluation of candidates
        auto evaluateBatch = [&](int threadId, int startIdx, int endIdx) {
          Wisard& w = threadWisards[threadId];
          std::vector<BinInput>& tBins = threadTrainBins[threadId];
          std::vector<BinInput>& vBins = threadValBins[threadId];

          for (int s = startIdx; s < endIdx; s++) {
            double threshold = candidateThresholds[s];

            // Modify the single bit for this threshold
            for (size_t i = 0; i < trainCount; i++) {
              tBins[i].set(bitPos, (trainData[i][f] > threshold) ? 1 : 0);
            }
            for (size_t i = 0; i < valCount; i++) {
              vBins[i].set(bitPos, (valData[i][f] > threshold) ? 1 : 0);
            }

            // Reset, train, classify
            w.reset();
            for (size_t i = 0; i < trainCount; i++) {
              w.trainSingle(tBins[i], trainLabels[i]);
            }

            int correct = 0;
            for (size_t i = 0; i < valCount; i++) {
              if (w.classify(vBins[i]) == valLabels[i]) correct++;
            }
            candidateAccuracies[s] = (double)correct / (double)valCount;
          }
        };

        // Evaluate candidates in parallel, with optional per-threshold early stopping.
        // earlyStopWindow > 0: evaluate left scan first; if no improvement in the
        // first earlyStopWindow candidates, skip the right scan entirely.
        if (earlyStopWindow > 0) {
          // Evaluate left scan
          parallelFor(0, halfSteps, nThreads, evaluateBatch, threadWisards);

          // Check if any left candidate improved
          bool leftImproved = false;
          int windowEnd = std::min(earlyStopWindow, halfSteps);
          for (int s = 0; s < windowEnd; s++) {
            if (candidateAccuracies[s] > bestAcc) { leftImproved = true; break; }
          }

          // If early window showed improvement, or no early stop window check needed,
          // evaluate the rest
          if (leftImproved) {
            parallelFor(halfSteps, stepsPerThreshold, nThreads, evaluateBatch, threadWisards);
          }
        } else {
          // No early stopping: evaluate all candidates
          parallelFor(0, stepsPerThreshold, nThreads, evaluateBatch, threadWisards);
        }

        // Find best candidate
        double bestValue = originalValue;
        double bestThresholdAcc = bestAcc;
        for (int s = 0; s < stepsPerThreshold; s++) {
          if (candidateAccuracies[s] > bestThresholdAcc) {
            bestThresholdAcc = candidateAccuracies[s];
            bestValue = candidateThresholds[s];
          }
        }

        // Commit the best threshold to master state
        valueRanges[f][k] = bestValue;
        for (size_t i = 0; i < trainCount; i++) {
          int newBit = (trainData[i][f] > bestValue) ? 1 : 0;
          if (newBit != trainBinInputs[i].get(bitPos)) {
            masterWisard.untrainSingle(trainBinInputs[i], trainLabels[i]);
            trainBinInputs[i].set(bitPos, newBit);
            masterWisard.trainSingle(trainBinInputs[i], trainLabels[i]);
          }
        }
        for (size_t i = 0; i < valCount; i++) {
          valBinInputs[i].set(bitPos, (valData[i][f] > bestValue) ? 1 : 0);
        }
        bestAcc = bestThresholdAcc;
      }
    }

    return bestAcc;
  }

  BinInput transform(const std::vector<double>& data) {
    if (!fitted) {
      throw Exception("StochasticThermometer must be fitted before transform!");
    }

    BinInput out(data.size() * thermometerSize);
    size_t k = 0;
    for (size_t i = 0; i < data.size(); i++) {
      for (size_t j = 0; j < valueRanges[i].size(); j++) {
        if (data[i] > valueRanges[i][j]) {
          out.set(k, 1);
        } else {
          out.set(k, 0);
        }
        k++;
      }
    }
    return out;
  }

  size_t getSize() const {
    return thermometerSize;
  }

  std::vector<std::vector<double>> getThresholds() const {
    return valueRanges;
  }

  void setThresholds(const std::vector<std::vector<double>>& thresholds) {
    valueRanges = thresholds;
    fitted = true;
  }

private:
  double classifyValidation(
    const Wisard& wisard,
    const std::vector<BinInput>& valBinInputs,
    const std::vector<std::string>& valLabels
  ) {
    int correct = 0;
    for (size_t i = 0; i < valBinInputs.size(); i++) {
      if (wisard.classify(valBinInputs[i]) == valLabels[i]) correct++;
    }
    return (double)correct / (double)valBinInputs.size();
  }

  // Distribute work [startIdx, endIdx) across threads
  template<typename Func>
  void parallelFor(
    int startIdx, int endIdx, int nThreads,
    Func& func,
    std::vector<Wisard>& wisards
  ) {
    int total = endIdx - startIdx;
    if (total <= 0) return;

    int actualThreads = std::min(nThreads, total);
    if (actualThreads <= 1) {
      func(0, startIdx, endIdx);
      return;
    }

    std::vector<std::thread> threads;
    threads.reserve(actualThreads);
    int chunk = (total + actualThreads - 1) / actualThreads;

    for (int t = 0; t < actualThreads; t++) {
      int lo = startIdx + t * chunk;
      int hi = std::min(lo + chunk, endIdx);
      if (lo >= endIdx) break;
      threads.emplace_back(func, t, lo, hi);
    }

    for (auto& t : threads) {
      t.join();
    }
  }

protected:
  size_t thermometerSize;
  std::vector<std::vector<double>> valueRanges;
  bool fitted;
};
