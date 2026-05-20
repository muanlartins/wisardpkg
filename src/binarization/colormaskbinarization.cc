// Hand-tuned colour-mask binariser for objects with a small known palette
// (Where's Waldo: red+white sweater, dark hair/glasses/beanie).
//
// Per pixel, emits 3 bits along channel-interleaved RGB triples:
//   bit 0: is_red    — high R, low G, low B
//   bit 1: is_white  — all three channels high
//   bit 2: is_dark   — all three channels low
//
// Input is assumed to be a flat row-major (H*W*3) vector of values in [0,1],
// channels interleaved as R,G,B per pixel. Output BinInput has 3*pixel_count bits.
class ColorMaskBinarization : public BinBase {
private:
  double redMin;
  double redChannelGap;
  double whiteMin;
  double darkMax;

public:
  ColorMaskBinarization(double redMin = 0.5,
                        double redChannelGap = 0.10,
                        double whiteMin = 0.75,
                        double darkMax = 0.20)
    : redMin(redMin), redChannelGap(redChannelGap), whiteMin(whiteMin), darkMax(darkMax) {}

  BinInput transform(const std::vector<double>& data) {
    if (data.size() % 3 != 0)
      throw Exception("ColorMaskBinarization expects RGB triples; input size must be divisible by 3");
    size_t n_px = data.size() / 3;
    BinInput out(n_px * 3);
    for (size_t i = 0; i < n_px; i++) {
      double r = data[3 * i];
      double g = data[3 * i + 1];
      double b = data[3 * i + 2];
      bool is_red   = (r > redMin) && (r - g > redChannelGap) && (r - b > redChannelGap);
      bool is_white = (r > whiteMin) && (g > whiteMin) && (b > whiteMin);
      bool is_dark  = (r < darkMax) && (g < darkMax) && (b < darkMax);
      out.set(3 * i,     is_red   ? 1 : 0);
      out.set(3 * i + 1, is_white ? 1 : 0);
      out.set(3 * i + 2, is_dark  ? 1 : 0);
    }
    return out;
  }

  double getRedMin()        const { return redMin; }
  double getRedChannelGap() const { return redChannelGap; }
  double getWhiteMin()      const { return whiteMin; }
  double getDarkMax()       const { return darkMax; }
};
