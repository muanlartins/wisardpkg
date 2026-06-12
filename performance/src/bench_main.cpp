#include "wisard.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifdef WP_OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

long peakRssBytes() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return long(ru.ru_maxrss);
#else
  return long(ru.ru_maxrss) * 1024L;
#endif
}

void applyThreads(int threads) {
#ifdef WP_OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#else
  (void)threads;
#endif
}

int activeThreads(int requested) {
#ifdef WP_OPENMP
  int t = 1;
#pragma omp parallel
  {
#pragma omp single
    t = omp_get_num_threads();
  }
  return t;
#else
  (void)requested;
  return 1;
#endif
}

double median(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t m = v.size() / 2;
  return (v.size() & 1) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

uint32_t computeNumRAMs(uint32_t entrySize, uint32_t addressSize) {
  uint32_t n = entrySize / addressSize;
  if (entrySize % addressSize) ++n;
  return n;
}

std::vector<uint32_t> randomMapping(uint32_t entrySize, uint32_t addressSize,
                                    uint32_t numRAMs, std::mt19937& rng) {
  uint32_t padded = numRAMs * addressSize;
  std::vector<uint32_t> indexes(padded);
  for (uint32_t i = 0; i < entrySize; ++i) indexes[i] = i;
  std::uniform_int_distribution<uint32_t> pick(0, entrySize - 1);
  for (uint32_t i = entrySize; i < padded; ++i) indexes[i] = pick(rng);
  std::shuffle(indexes.begin(), indexes.end(), rng);
  return indexes;
}

double scoreAccuracy(const std::vector<uint32_t>& pred,
                     const std::vector<uint32_t>& truth) {
  if (pred.empty()) return 0.0;
  size_t correct = 0;
  for (size_t i = 0; i < pred.size(); ++i)
    if (pred[i] == truth[i]) ++correct;
  return double(correct) / double(pred.size());
}

struct RunMetrics {
  double train_ms = 0;
  double infer_ms_total = 0;
  double accuracy = 0;
  long model_bytes = 0;
};

RunMetrics runOnce(wp::Wisard& w, const wp::WbinData& d,
                   std::vector<uint32_t>& pred) {
  RunMetrics m;
  auto t0 = Clock::now();
  w.train(d.trainBits, d.trainLabels);
  auto t1 = Clock::now();
  w.classify(d.testBits, pred);
  auto t2 = Clock::now();
  m.train_ms = msSince(t0, t1);
  m.infer_ms_total = msSince(t1, t2);
  m.accuracy = scoreAccuracy(pred, d.testLabels);
  m.model_bytes = w.modelBytes();
  return m;
}

void emitJson(const std::string& path, const RunMetrics& m, size_t nTest,
              uint32_t addressSize, uint32_t numRAMs, int reps, int threads,
              long rss) {
  double per = nTest ? m.infer_ms_total / double(nTest) : 0.0;
  double sps = m.infer_ms_total > 0 ? double(nTest) / (m.infer_ms_total / 1000.0) : 0.0;
  char buf[1024];
  std::snprintf(buf, sizeof(buf),
                "{\n"
                "  \"train_ms\": %.6f,\n"
                "  \"infer_ms_total\": %.6f,\n"
                "  \"infer_ms_per_sample\": %.6f,\n"
                "  \"throughput_sps\": %.3f,\n"
                "  \"peak_rss_bytes\": %ld,\n"
                "  \"model_bytes\": %ld,\n"
                "  \"accuracy\": %.6f,\n"
                "  \"address_size\": %u,\n"
                "  \"num_rams\": %u,\n"
                "  \"reps\": %d,\n"
                "  \"threads\": %d\n"
                "}\n",
                m.train_ms, m.infer_ms_total, per, sps, rss, m.model_bytes,
                m.accuracy, addressSize, numRAMs, reps, threads);
  if (path.empty()) {
    std::fputs(buf, stdout);
  } else {
    std::ofstream f(path);
    f << buf;
  }
}

int selftest(int threads) {
  applyThreads(threads);

  const uint32_t numClasses = 3;
  const uint32_t prefixBits = 12;
  const uint32_t noiseBits = 36;
  const uint32_t entrySize = prefixBits + noiseBits;
  const uint32_t addressSize = 8;
  const size_t nTrain = 600;
  const size_t nTest = 150;

  std::mt19937 rng(12345);
  std::bernoulli_distribution noise(0.5);
  std::bernoulli_distribution flip(0.05);
  std::uniform_int_distribution<uint32_t> classPick(0, numClasses - 1);

  wp::WbinData d;
  d.entrySize = entrySize;
  d.addressSize = addressSize;
  d.numRAMs = computeNumRAMs(entrySize, addressSize);
  d.numClasses = numClasses;
  d.nTrain = nTrain;
  d.nTest = nTest;
  d.mapping = randomMapping(entrySize, addressSize, d.numRAMs, rng);

  auto fabricate = [&](size_t count, wp::PackedInputs& bits,
                       std::vector<uint32_t>& labels) {
    bits.init(count, entrySize);
    labels.resize(count);
    for (size_t s = 0; s < count; ++s) {
      uint32_t cls = classPick(rng);
      labels[s] = cls;
      for (uint32_t b = 0; b < prefixBits; ++b) {
        uint32_t expected = ((b % numClasses) == cls) ? 1u : 0u;
        if (flip(rng)) expected ^= 1u;
        if (expected) bits.setBit(s, b);
      }
      for (uint32_t b = 0; b < noiseBits; ++b) {
        if (noise(rng)) bits.setBit(s, prefixBits + b);
      }
    }
  };

  fabricate(nTrain, d.trainBits, d.trainLabels);
  fabricate(nTest, d.testBits, d.testLabels);

  wp::Wisard w;
  w.build(numClasses, d.numRAMs, addressSize, d.mapping);

  std::vector<uint32_t> pred;
  RunMetrics m = runOnce(w, d, pred);

  int th = activeThreads(threads);
  emitJson("", m, nTest, addressSize, d.numRAMs, 1, th, peakRssBytes());

  bool pass = m.accuracy > 0.90;
  std::printf("%s accuracy=%.4f (threshold 0.90)\n", pass ? "PASS" : "FAIL",
              m.accuracy);
  return pass ? 0 : 1;
}

int runBench(const std::string& input, int threads, int reps,
             const std::string& out) {
  applyThreads(threads);

  wp::WbinData d;
  std::string err;
  if (!wp::io::read(input, d, err)) {
    std::fprintf(stderr, "error reading %s: %s\n", input.c_str(), err.c_str());
    return 1;
  }

  std::vector<double> trainTimes, inferTimes;
  RunMetrics last;
  std::vector<uint32_t> pred;
  for (int r = 0; r < reps; ++r) {
    wp::Wisard w;
    w.build(d.numClasses, d.numRAMs, d.addressSize, d.mapping);
    last = runOnce(w, d, pred);
    trainTimes.push_back(last.train_ms);
    inferTimes.push_back(last.infer_ms_total);
  }

  RunMetrics agg = last;
  agg.train_ms = median(trainTimes);
  agg.infer_ms_total = median(inferTimes);

  int th = activeThreads(threads);
  emitJson(out, agg, d.nTest, d.addressSize, d.numRAMs, reps, th, peakRssBytes());
  if (!out.empty()) {
    std::printf("accuracy=%.4f model_bytes=%ld train_ms=%.3f infer_ms=%.3f\n",
                agg.accuracy, agg.model_bytes, agg.train_ms, agg.infer_ms_total);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, out;
  int threads = 0;
  int reps = 1;
  bool doSelftest = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (a == "--input") input = next();
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--reps") reps = std::max(1, std::atoi(next().c_str()));
    else if (a == "--out") out = next();
    else if (a == "--selftest") doSelftest = true;
    else if (a == "-h" || a == "--help") {
      std::printf(
          "usage: wisard_bench --input FILE.wbin [--threads N] [--reps R] "
          "[--out RESULT.json]\n"
          "       wisard_bench --selftest [--threads N]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      return 1;
    }
  }

  if (doSelftest) return selftest(threads);

  if (input.empty()) {
    std::fprintf(stderr, "error: --input required (or use --selftest)\n");
    return 1;
  }
  return runBench(input, threads, reps, out);
}
