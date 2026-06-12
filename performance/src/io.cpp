#include "wisard.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace wp {
namespace io {

namespace {

template <class T>
bool readPod(std::ifstream& f, T& v) {
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  return bool(f);
}

template <class T>
bool readVec(std::ifstream& f, std::vector<T>& v, size_t count) {
  v.resize(count);
  if (count == 0) return true;
  f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
  return bool(f);
}

template <class T>
void writePod(std::ofstream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
void writeVec(std::ofstream& f, const std::vector<T>& v) {
  if (!v.empty()) f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

}  // namespace

bool read(const std::string& path, WbinData& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open file: " + path;
    return false;
  }

  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, "WBIN", 4) != 0) {
    err = "bad magic (expected WBIN)";
    return false;
  }

  uint32_t version = 0;
  if (!readPod(f, version)) { err = "truncated header"; return false; }
  if (version != 1) { err = "unsupported version"; return false; }

  if (!readPod(f, out.entrySize) || !readPod(f, out.addressSize) ||
      !readPod(f, out.numRAMs) || !readPod(f, out.numClasses)) {
    err = "truncated header fields";
    return false;
  }
  if (!readPod(f, out.nTrain) || !readPod(f, out.nTest)) {
    err = "truncated sample counts";
    return false;
  }

  if (!readVec(f, out.mapping, size_t(out.numRAMs) * out.addressSize)) {
    err = "truncated mapping";
    return false;
  }
  if (!readVec(f, out.trainLabels, size_t(out.nTrain))) {
    err = "truncated train labels";
    return false;
  }
  if (!readVec(f, out.testLabels, size_t(out.nTest))) {
    err = "truncated test labels";
    return false;
  }

  out.trainBits.init(out.nTrain, out.entrySize);
  out.testBits.init(out.nTest, out.entrySize);

  if (out.nTrain) {
    f.read(reinterpret_cast<char*>(out.trainBits.data.data()),
           out.trainBits.data.size() * sizeof(uint64_t));
    if (!f) { err = "truncated train bits"; return false; }
  }
  if (out.nTest) {
    f.read(reinterpret_cast<char*>(out.testBits.data.data()),
           out.testBits.data.size() * sizeof(uint64_t));
    if (!f) { err = "truncated test bits"; return false; }
  }

  return true;
}

bool write(const std::string& path, const WbinData& in, std::string& err) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open file for writing: " + path;
    return false;
  }

  f.write("WBIN", 4);
  writePod<uint32_t>(f, 1);
  writePod(f, in.entrySize);
  writePod(f, in.addressSize);
  writePod(f, in.numRAMs);
  writePod(f, in.numClasses);
  writePod(f, in.nTrain);
  writePod(f, in.nTest);

  writeVec(f, in.mapping);
  writeVec(f, in.trainLabels);
  writeVec(f, in.testLabels);
  writeVec(f, in.trainBits.data);
  writeVec(f, in.testBits.data);

  if (!f) {
    err = "write failure";
    return false;
  }
  return true;
}

}  // namespace io
}  // namespace wp
