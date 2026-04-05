class SupervisedThermometer : public BinBase {
public:
  SupervisedThermometer(
    const size_t thermometerSize = 32,
    const std::string& method = "class_conditional",
    const size_t minBitsPerFeature = 2
  ) : thermometerSize(thermometerSize), method(method),
      minBitsPerFeature(minBitsPerFeature), totalSize(0), fitted(false) {}

  void fit(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::string>& labels
  ) {
    if (data.empty()) return;
    size_t n_samples = data.size();
    size_t n_features = data[0].size();

    if (method == "mi_allocation") {
      fitMIAllocation(data, labels, n_samples, n_features);
    } else if (method == "entropy_weighted") {
      fitEntropyWeighted(data, labels, n_samples, n_features);
    } else {
      fitClassConditional(data, labels, n_samples, n_features);
    }

    totalSize = 0;
    for (size_t f = 0; f < valueRanges.size(); f++) {
      totalSize += valueRanges[f].size();
    }
    fitted = true;
  }

  BinInput transform(const std::vector<double>& data) {
    if (!fitted) {
      throw Exception("SupervisedThermometer must be fitted before transform!");
    }

    BinInput out(totalSize);
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
    return totalSize > 0 ? totalSize : thermometerSize;
  }

  std::vector<size_t> getSizes() const {
    return featureSizes;
  }

  std::vector<std::vector<double>> getThresholds() const {
    return valueRanges;
  }

  void setThresholds(const std::vector<std::vector<double>>& thresholds) {
    valueRanges = thresholds;
    featureSizes.resize(thresholds.size());
    totalSize = 0;
    for (size_t i = 0; i < thresholds.size(); i++) {
      featureSizes[i] = thresholds[i].size();
      totalSize += featureSizes[i];
    }
    fitted = true;
  }

private:
  // ======== Class-Conditional: per-class quantiles merged ========
  void fitClassConditional(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::string>& labels,
    size_t n_samples, size_t n_features
  ) {
    std::map<std::string, std::vector<size_t>> classIndices;
    for (size_t i = 0; i < n_samples; i++) {
      classIndices[labels[i]].push_back(i);
    }
    size_t n_classes = classIndices.size();

    std::vector<std::pair<std::string, size_t>> classAlloc;
    size_t base = thermometerSize / n_classes;
    size_t remainder = thermometerSize % n_classes;
    size_t c = 0;
    for (auto& ci : classIndices) {
      classAlloc.push_back({ci.first, base + (c < remainder ? 1 : 0)});
      c++;
    }

    valueRanges.resize(n_features);
    featureSizes.assign(n_features, thermometerSize);
    for (size_t f = 0; f < n_features; f++) {
      std::vector<double> allThresholds;
      allThresholds.reserve(thermometerSize);

      for (auto& ca : classAlloc) {
        const std::vector<size_t>& indices = classIndices[ca.first];
        size_t nThresholds = ca.second;
        if (nThresholds == 0 || indices.empty()) continue;

        std::vector<double> classValues(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
          classValues[i] = data[indices[i]][f];
        }
        std::sort(classValues.begin(), classValues.end());

        for (size_t k = 0; k < nThresholds; k++) {
          double percentile = (double)(k + 1) / (double)(nThresholds + 1);
          size_t index = (size_t)(percentile * (double)(classValues.size() - 1));
          if (index >= classValues.size()) index = classValues.size() - 1;
          allThresholds.push_back(classValues[index]);
        }
      }

      std::sort(allThresholds.begin(), allThresholds.end());
      valueRanges[f] = allThresholds;
    }
  }

  // ======== MI Allocation: MI-based bit budget per feature ========
  void fitMIAllocation(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::string>& labels,
    size_t n_samples, size_t n_features
  ) {
    size_t totalBudget = n_features * thermometerSize;

    std::vector<double> miScores(n_features);
    for (size_t f = 0; f < n_features; f++) {
      std::vector<double> values(n_samples);
      for (size_t i = 0; i < n_samples; i++) values[i] = data[i][f];
      miScores[f] = computeMI(values, labels, thermometerSize);
    }

    featureSizes = allocateBits(miScores, totalBudget, minBitsPerFeature, n_features);

    valueRanges.resize(n_features);
    for (size_t f = 0; f < n_features; f++) {
      std::vector<double> values(n_samples);
      for (size_t i = 0; i < n_samples; i++) values[i] = data[i][f];
      std::sort(values.begin(), values.end());

      size_t bits = featureSizes[f];
      valueRanges[f].resize(bits);
      for (size_t k = 0; k < bits; k++) {
        double percentile = (double)(k + 1) / (double)(bits + 1);
        size_t index = (size_t)(percentile * (double)(n_samples - 1));
        if (index >= n_samples) index = n_samples - 1;
        valueRanges[f][k] = values[index];
      }
    }
  }

  // ======== Entropy-Weighted: concentrate thresholds at class overlap ========
  void fitEntropyWeighted(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::string>& labels,
    size_t n_samples, size_t n_features
  ) {
    size_t windowSize = std::max((size_t)10, (size_t)std::sqrt((double)n_samples));
    windowSize = std::min(windowSize, n_samples / 4);

    featureSizes.assign(n_features, thermometerSize);
    valueRanges.resize(n_features);
    for (size_t f = 0; f < n_features; f++) {
      std::vector<size_t> sortedIdx(n_samples);
      for (size_t i = 0; i < n_samples; i++) sortedIdx[i] = i;
      std::sort(sortedIdx.begin(), sortedIdx.end(),
        [&](size_t a, size_t b) { return data[a][f] < data[b][f]; });

      std::vector<double> weights(n_samples);
      size_t halfW = windowSize / 2;

      for (size_t i = 0; i < n_samples; i++) {
        size_t lo = (i > halfW) ? i - halfW : 0;
        size_t hi = std::min(n_samples, i + halfW + 1);
        if (hi - lo > windowSize) {
          if (lo == 0) hi = lo + windowSize;
          else lo = hi - windowSize;
        }

        std::map<std::string, int> windowCounts;
        for (size_t j = lo; j < hi; j++) {
          windowCounts[labels[sortedIdx[j]]]++;
        }

        double wSize = (double)(hi - lo);
        double entropy = 0.0;
        for (auto& lc : windowCounts) {
          double p = (double)lc.second / wSize;
          if (p > 0.0) entropy -= p * std::log2(p);
        }
        weights[i] = entropy;
      }

      double maxW = *std::max_element(weights.begin(), weights.end());
      double floor = (maxW > 0.0) ? 0.01 * maxW : 1.0;
      for (size_t i = 0; i < n_samples; i++) weights[i] += floor;

      std::vector<double> cumWeights(n_samples);
      cumWeights[0] = weights[0];
      for (size_t i = 1; i < n_samples; i++) {
        cumWeights[i] = cumWeights[i - 1] + weights[i];
      }
      double totalWeight = cumWeights[n_samples - 1];

      valueRanges[f].resize(thermometerSize);
      for (size_t k = 0; k < thermometerSize; k++) {
        double target = (double)(k + 1) / (double)(thermometerSize + 1) * totalWeight;
        auto it = std::lower_bound(cumWeights.begin(), cumWeights.end(), target);
        size_t index = (size_t)(it - cumWeights.begin());
        if (index >= n_samples) index = n_samples - 1;
        valueRanges[f][k] = data[sortedIdx[index]][f];
      }
    }
  }

  // ======== MI utilities ========
  double computeMI(
    const std::vector<double>& featureValues,
    const std::vector<std::string>& labels,
    size_t nBins
  ) {
    size_t n = featureValues.size();
    if (n == 0 || nBins == 0) return 0.0;

    std::vector<size_t> sortedIdx(n);
    for (size_t i = 0; i < n; i++) sortedIdx[i] = i;
    std::sort(sortedIdx.begin(), sortedIdx.end(),
      [&](size_t a, size_t b) { return featureValues[a] < featureValues[b]; });

    std::vector<size_t> binAssignment(n);
    for (size_t i = 0; i < n; i++) {
      binAssignment[sortedIdx[i]] = (i * nBins) / n;
    }

    std::map<std::string, size_t> labelIndex;
    for (size_t i = 0; i < n; i++) {
      if (labelIndex.find(labels[i]) == labelIndex.end()) {
        size_t idx = labelIndex.size();
        labelIndex[labels[i]] = idx;
      }
    }
    size_t nLabels = labelIndex.size();

    std::vector<std::vector<size_t>> joint(nBins, std::vector<size_t>(nLabels, 0));
    for (size_t i = 0; i < n; i++) {
      joint[binAssignment[i]][labelIndex[labels[i]]]++;
    }

    std::vector<size_t> binCounts(nBins, 0);
    std::vector<size_t> labelCounts(nLabels, 0);
    for (size_t b = 0; b < nBins; b++) {
      for (size_t l = 0; l < nLabels; l++) {
        binCounts[b] += joint[b][l];
        labelCounts[l] += joint[b][l];
      }
    }

    double mi = 0.0;
    double dn = (double)n;
    for (size_t b = 0; b < nBins; b++) {
      if (binCounts[b] == 0) continue;
      for (size_t l = 0; l < nLabels; l++) {
        if (joint[b][l] == 0) continue;
        double pbl = (double)joint[b][l] / dn;
        double pb = (double)binCounts[b] / dn;
        double pl = (double)labelCounts[l] / dn;
        mi += pbl * std::log2(pbl / (pb * pl));
      }
    }
    return std::max(mi, 0.0);
  }

  std::vector<size_t> allocateBits(
    const std::vector<double>& miScores,
    size_t totalBudget, size_t minBits, size_t n_features
  ) {
    std::vector<size_t> sizes(n_features, minBits);
    size_t floorTotal = minBits * n_features;

    if (floorTotal >= totalBudget) {
      size_t perFeature = totalBudget / n_features;
      size_t rem = totalBudget % n_features;
      for (size_t f = 0; f < n_features; f++) {
        sizes[f] = perFeature + (f < rem ? 1 : 0);
      }
      return sizes;
    }

    size_t remaining = totalBudget - floorTotal;
    double totalMI = 0.0;
    for (size_t f = 0; f < n_features; f++) totalMI += miScores[f];

    if (totalMI <= 0.0) {
      size_t perFeature = totalBudget / n_features;
      size_t rem = totalBudget % n_features;
      for (size_t f = 0; f < n_features; f++) {
        sizes[f] = perFeature + (f < rem ? 1 : 0);
      }
      return sizes;
    }

    std::vector<double> fractional(n_features);
    size_t allocated = 0;
    for (size_t f = 0; f < n_features; f++) {
      fractional[f] = (miScores[f] / totalMI) * (double)remaining;
      sizes[f] += (size_t)fractional[f];
      allocated += (size_t)fractional[f];
    }

    size_t leftover = remaining - allocated;
    if (leftover > 0) {
      std::vector<size_t> order(n_features);
      for (size_t f = 0; f < n_features; f++) order[f] = f;
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return (fractional[a] - (size_t)fractional[a]) > (fractional[b] - (size_t)fractional[b]);
      });
      for (size_t i = 0; i < leftover && i < n_features; i++) {
        sizes[order[i]]++;
      }
    }
    return sizes;
  }

protected:
  size_t thermometerSize;
  std::string method;
  size_t minBitsPerFeature;
  std::vector<size_t> featureSizes;
  std::vector<std::vector<double>> valueRanges;
  size_t totalSize;
  bool fitted;
};
