class LogarithmicThermometer : public BinBase {
public:
  LogarithmicThermometer(const size_t thermometerSize) : thermometerSize(thermometerSize), fitted(false) {}

  void fit(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return;
    size_t n_samples = data.size();
    size_t n_features = data[0].size();

    valueRanges.resize(n_features);

    for (size_t f = 0; f < n_features; f++) {
      double minv = data[0][f], maxv = data[0][f];
      for (size_t i = 1; i < n_samples; i++) {
        if (data[i][f] < minv) minv = data[i][f];
        if (data[i][f] > maxv) maxv = data[i][f];
      }
      double shift = (minv <= 0.0) ? (-minv + 1.0) : 0.0;
      double logMin = std::log(minv + shift);
      double logMax = std::log(maxv + shift);
      if (logMax <= logMin) logMax = logMin + 1e-9;

      valueRanges[f].resize(thermometerSize);
      for (size_t k = 0; k < thermometerSize; k++) {
        double frac = (double)(k + 1) / (double)(thermometerSize + 1);
        double logThr = logMin + frac * (logMax - logMin);
        valueRanges[f][k] = std::exp(logThr) - shift;
      }
    }
    fitted = true;
  }

  BinInput transform(const std::vector<double>& data) {
    if (!fitted) {
      throw Exception("LogarithmicThermometer must be fitted before transform!");
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

protected:
  size_t thermometerSize;
  std::vector<std::vector<double>> valueRanges;
  bool fitted;
};
