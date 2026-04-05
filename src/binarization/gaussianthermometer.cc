class GaussianThermometer : public BinBase {
public:
  GaussianThermometer(const size_t thermometerSize) : thermometerSize(thermometerSize), fitted(false) {}

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
      double sigma = (n_samples > 1) ? math::stdev(values) : 0.0;

      valueRanges[f].resize(thermometerSize);
      for (size_t k = 0; k < thermometerSize; k++) {
        double p = (double)(k + 1) / (double)(thermometerSize + 1);
        if (sigma > 0.0) {
          valueRanges[f][k] = mu + sigma * probit(p);
        } else {
          valueRanges[f][k] = mu;
        }
      }
    }

    fitted = true;
  }

  BinInput transform(const std::vector<double>& data) {
    if (!fitted) {
      throw Exception("GaussianThermometer must be fitted before transform!");
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
  // Peter Acklam's rational approximation for the inverse normal CDF (probit)
  // Accurate to approximately 1.15e-9 across the full range (0, 1)
  static double probit(double p) {
    static const double a[] = {
      -3.969683028665376e+01,  2.209460984245205e+02,
      -2.759285104469687e+02,  1.383577518672690e+02,
      -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
      -5.447609879822406e+01,  1.615858368580409e+02,
      -1.556989798598866e+02,  6.680131188771972e+01,
      -1.328068155288572e+01
    };
    static const double c[] = {
      -7.784894002430293e-03, -3.223964580411365e-01,
      -2.400758277161838e+00, -2.549732539343734e+00,
       4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
       7.784695709041462e-03,  3.224671290700398e-01,
       2.445134137142996e+00,  3.754408661907416e+00
    };

    static const double p_low  = 0.02425;
    static const double p_high = 1.0 - p_low;

    double q, r;

    if (p < p_low) {
      // Lower tail
      q = std::sqrt(-2.0 * std::log(p));
      return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= p_high) {
      // Central region
      q = p - 0.5;
      r = q * q;
      return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
             (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
      // Upper tail (symmetry)
      q = std::sqrt(-2.0 * std::log(1.0 - p));
      return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
  }

protected:
  size_t thermometerSize;
  std::vector<std::vector<double>> valueRanges;
  bool fitted;
};
