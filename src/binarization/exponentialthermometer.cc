class ExponentialThermometer : public BinBase {
public:
  ExponentialThermometer(const size_t thermometerSize) : thermometerSize(thermometerSize), fitted(false) {}

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

      double mu = math::mean(values);

      valueRanges[f].resize(thermometerSize);
      for (size_t k = 0; k < thermometerSize; k++) {
        double p = (double)(k + 1) / (double)(thermometerSize + 1);
        if (mu > 0.0) {
          // Inverse CDF of exponential: F^{-1}(p) = -ln(1-p) / lambda = -ln(1-p) * mu
          valueRanges[f][k] = -std::log(1.0 - p) * mu;
        } else {
          valueRanges[f][k] = 0.0;
        }
      }
    }

    fitted = true;
  }

  BinInput transform(const std::vector<double>& data) {
    if (!fitted) {
      throw Exception("ExponentialThermometer must be fitted before transform!");
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
