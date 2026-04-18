
class BloomWisardWrapper: public BloomWisard {
public:
  BloomWisardWrapper(int addressSize, int numBits, int numHashes, py::kwargs kwargs)
    : BloomWisard(addressSize, numBits, numHashes) {
    for (auto arg : kwargs) {
      if (std::string(py::str(arg.first)).compare("classificationMethod") == 0) {
        classificationMethod = arg.second.cast<ClassificationBase*>();
        classificationMethod = classificationMethod->clone();
      }

      if (std::string(py::str(arg.first)).compare("verbose") == 0)
        verbose = arg.second.cast<bool>();

      if (std::string(py::str(arg.first)).compare("ignoreZero") == 0)
        ignoreZero = arg.second.cast<bool>();

      if (std::string(py::str(arg.first)).compare("completeAddressing") == 0)
        mappingGenerator->completeAddressing = arg.second.cast<bool>();

      if (std::string(py::str(arg.first)).compare("monoMapping") == 0)
        mappingGenerator->monoMapping = arg.second.cast<bool>();

      if (std::string(py::str(arg.first)).compare("hashMode") == 0)
        hashMode = arg.second.cast<std::string>();
    }
  }
};
