// Spatially-aware mapping for image-like inputs.
//
// Standard RandomMapping shuffles all input bits uniformly across RAMs. For
// image data, that destroys the 2D co-occurrence that makes structural patterns
// (stripes, edges, blob configurations) detectable. Local2DMapping confines each
// RAM's input bits to a contiguous (windowHeight × windowWidth × bitsPerPixel)
// window of the image, with the windows tiled across the image with optional
// overlap.
//
// Input layout assumption: the flat bit vector is row-major over pixels, with
// the `bitsPerPixel` bits for each pixel contiguous. That is, the bit at
// (row=h, col=w, bit=k) lives at index  h*W*bitsPerPixel + w*bitsPerPixel + k,
// for 0 <= h < imageHeight, 0 <= w < imageWidth, 0 <= k < bitsPerPixel.
// This matches what you get by flattening an (H, W, C) uint8 array per pixel
// after a per-pixel binariser like ColorMaskBinarization, or an (H, W, C*B)
// array after per-channel thermometer encoding (set bitsPerPixel = C * B).
class Local2DMapping : public MappingGenerator {
private:
  unsigned int imageHeight;
  unsigned int imageWidth;
  unsigned int bitsPerPixel;
  unsigned int windowHeight;
  unsigned int windowWidth;
  unsigned int stride;          // 0 means stride = window size (no overlap)
  unsigned int ramsPerWindow;   // number of RAMs sampled from each window

public:
  Local2DMapping(unsigned int imageHeight,
                 unsigned int imageWidth,
                 unsigned int bitsPerPixel,
                 unsigned int windowHeight,
                 unsigned int windowWidth,
                 unsigned int tupleSize,
                 unsigned int stride = 0,
                 unsigned int ramsPerWindow = 1,
                 bool monoMapping = false) {
    if (imageHeight < 1 || imageWidth < 1 || bitsPerPixel < 1)
      throw Exception("Local2DMapping: image dimensions and bitsPerPixel must be >= 1");
    if (windowHeight < 1 || windowWidth < 1)
      throw Exception("Local2DMapping: window dimensions must be >= 1");
    if (windowHeight > imageHeight || windowWidth > imageWidth)
      throw Exception("Local2DMapping: window must fit inside image");
    unsigned int bitsPerWindow = windowHeight * windowWidth * bitsPerPixel;
    if (tupleSize < 2)
      throw Exception("Local2DMapping: tupleSize must be >= 2");
    if (tupleSize > bitsPerWindow)
      throw Exception("Local2DMapping: tupleSize must not exceed bits in a window");
    if (ramsPerWindow < 1)
      throw Exception("Local2DMapping: ramsPerWindow must be >= 1");

    this->imageHeight   = imageHeight;
    this->imageWidth    = imageWidth;
    this->bitsPerPixel  = bitsPerPixel;
    this->windowHeight  = windowHeight;
    this->windowWidth   = windowWidth;
    this->stride        = stride == 0 ? std::min(windowHeight, windowWidth) : stride;
    this->ramsPerWindow = ramsPerWindow;
    this->tupleSize     = tupleSize;
    this->monoMapping   = monoMapping;
    this->completeAddressing = false;  // we always emit fixed tupleSize, no padding semantics
    this->multiResolution    = false;

    // Pre-fill `indexes` so getEntrySize()/checkEntrySize work consistently.
    unsigned int entrySize = imageHeight * imageWidth * bitsPerPixel;
    this->indexes = entrySizeToIndexes(entrySize);
  }

  Local2DMapping(nl::json config) {
    imageHeight   = config["imageHeight"].get<unsigned int>();
    imageWidth    = config["imageWidth"].get<unsigned int>();
    bitsPerPixel  = config["bitsPerPixel"].get<unsigned int>();
    windowHeight  = config["windowHeight"].get<unsigned int>();
    windowWidth   = config["windowWidth"].get<unsigned int>();
    stride        = config["stride"].get<unsigned int>();
    ramsPerWindow = config["ramsPerWindow"].get<unsigned int>();
    tupleSize     = config["tupleSize"].get<unsigned int>();
    monoMapping   = config["monoMapping"].get<bool>();
    completeAddressing = false;
    multiResolution    = false;
    auto m_it = config.find("mapping");
    if (m_it != config.end()) {
      mapping = m_it->get<std::map<std::string, std::vector<std::vector<int>>>>();
    }
    indexes = entrySizeToIndexes(imageHeight * imageWidth * bitsPerPixel);
  }

  std::vector<std::vector<int>> getMapping(const std::string label) override {
    auto it = mapping.find(label);
    if (it != mapping.end()) return it->second;
    if (monoMapping && !mapping.empty()) return mapping.begin()->second;

    std::random_device rd;
    std::mt19937 mt(rd());

    std::vector<std::vector<int>> result;
    std::vector<int> windowBits(windowHeight * windowWidth * bitsPerPixel);

    for (unsigned int top = 0; top + windowHeight <= imageHeight; top += stride) {
      for (unsigned int left = 0; left + windowWidth <= imageWidth; left += stride) {
        // Collect the absolute bit indices that live in this window.
        unsigned int p = 0;
        for (unsigned int dh = 0; dh < windowHeight; dh++) {
          for (unsigned int dw = 0; dw < windowWidth; dw++) {
            unsigned int pixelBase = ((top + dh) * imageWidth + (left + dw)) * bitsPerPixel;
            for (unsigned int k = 0; k < bitsPerPixel; k++) {
              windowBits[p++] = (int)(pixelBase + k);
            }
          }
        }

        // Emit ramsPerWindow RAMs, each a random tupleSize-sized subset of windowBits.
        for (unsigned int r = 0; r < ramsPerWindow; r++) {
          std::shuffle(windowBits.begin(), windowBits.end(), mt);
          result.emplace_back(windowBits.begin(), windowBits.begin() + tupleSize);
        }
      }
    }

    mapping[label] = result;
    return result;
  }

  MappingGenerator* clone() const override {
    return new Local2DMapping(imageHeight, imageWidth, bitsPerPixel,
                              windowHeight, windowWidth, tupleSize,
                              stride, ramsPerWindow, monoMapping);
  }

  std::string json() const override {
    nl::json config = {
      {"imageHeight",   imageHeight},
      {"imageWidth",    imageWidth},
      {"bitsPerPixel",  bitsPerPixel},
      {"windowHeight",  windowHeight},
      {"windowWidth",   windowWidth},
      {"stride",        stride},
      {"ramsPerWindow", ramsPerWindow},
      {"tupleSize",     tupleSize},
      {"monoMapping",   monoMapping},
      {"mapping",       mapping},
    };
    return config.dump();
  }

  std::string className() const override {
    return "Local2DMapping";
  }

  // Accessors (helpful from Python for diagnostics).
  unsigned int getImageHeight()   const { return imageHeight; }
  unsigned int getImageWidth()    const { return imageWidth; }
  unsigned int getBitsPerPixel()  const { return bitsPerPixel; }
  unsigned int getWindowHeight() const { return windowHeight; }
  unsigned int getWindowWidth()  const { return windowWidth; }
  unsigned int getStride()        const { return stride; }
  unsigned int getRamsPerWindow() const { return ramsPerWindow; }
  unsigned int getNumberOfRAMs()  const {
    unsigned int rowsOfWindows = ((imageHeight - windowHeight) / stride) + 1;
    unsigned int colsOfWindows = ((imageWidth  - windowWidth ) / stride) + 1;
    return rowsOfWindows * colsOfWindows * ramsPerWindow;
  }
};
