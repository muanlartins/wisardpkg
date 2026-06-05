// CircularThermometer — for periodic values (lat/lon, hour, day-of-week).
// Each bit_i has a receptive field centered at c_i = (i/N) * period, of width period/2.
// Bit fires when the value lies within the circular-distance window of the center.
class CircularThermometer : public BinBase {
public:
  CircularThermometer(const size_t thermometerSize,
                      const double minimum = 0.0,
                      const double maximum = 6.283185307179586)
    : thermometerSize(thermometerSize), minv(minimum), maxv(maximum) {
    period = maximum - minimum;
    if (period <= 0.0) {
      throw Exception("CircularThermometer requires maximum > minimum");
    }
  }

  BinInput transform(const std::vector<double>& data) {
    BinInput out(data.size() * thermometerSize);
    size_t k = 0;
    double halfPeriod = period / 2.0;
    double quarterPeriod = period / 4.0;
    for (size_t i = 0; i < data.size(); i++) {
      // wrap value into [0, period)
      double v = data[i] - minv;
      v = v - std::floor(v / period) * period;
      for (size_t j = 0; j < thermometerSize; j++) {
        double center = ((double)j / (double)thermometerSize) * period;
        double diff = std::fabs(v - center);
        if (diff > halfPeriod) diff = period - diff;  // shortest arc
        out.set(k, diff < quarterPeriod ? 1 : 0);
        k++;
      }
    }
    return out;
  }

  size_t getSize() const {
    return thermometerSize;
  }

  double getMinimum() const { return minv; }
  double getMaximum() const { return maxv; }

protected:
  size_t thermometerSize;
  double minv;
  double maxv;
  double period;
};
