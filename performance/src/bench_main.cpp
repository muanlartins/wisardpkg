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
#include <sstream>
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

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t m = v.size() / 2;
  return (v.size() & 1) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

double percentile(std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  size_t idx = size_t(q * (sorted.size() - 1) + 0.5);
  if (idx >= sorted.size()) idx = sorted.size() - 1;
  return sorted[idx];
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

struct LatencyDist {
  double p50 = 0, p99 = 0, mean = 0, max = 0;
};

// Single-threaded per-sample latency. Timing each classify individually adds a
// steady_clock pair per sample, so these percentiles are an upper bound on the
// true per-sample cost (clock overhead is included, identically for every
// contender that measures the same way). Throughput is reported separately from
// the un-instrumented bulk pass.
LatencyDist measureLatency(const wp::Wisard& w, const wp::WbinData& d) {
  std::vector<wp::content_t> buf(size_t(w.numClasses()) * w.numRAMs());
  std::vector<double> lat;
  lat.reserve(d.testBits.n);
  volatile uint32_t sink = 0;
  for (size_t s = 0; s < d.testBits.n; ++s) {
    auto a = Clock::now();
    sink ^= w.predictOne(d.testBits, s, buf.data());
    auto b = Clock::now();
    lat.push_back(msSince(a, b));
  }
  (void)sink;
  std::sort(lat.begin(), lat.end());
  LatencyDist L;
  L.p50 = percentile(lat, 0.50);
  L.p99 = percentile(lat, 0.99);
  L.max = lat.empty() ? 0.0 : lat.back();
  double sum = 0;
  for (double x : lat) sum += x;
  L.mean = lat.empty() ? 0.0 : sum / lat.size();
  return L;
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

std::string arrJson(const std::vector<double>& v) {
  std::ostringstream s;
  s << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.6f", v[i]);
    s << b;
    if (i + 1 < v.size()) s << ", ";
  }
  s << "]";
  return s.str();
}

void emitJson(const std::string& path, const RunMetrics& m,
              const std::vector<double>& trainReps,
              const std::vector<double>& inferReps, const LatencyDist& lat,
              size_t nTest, uint32_t addressSize, uint32_t numRAMs, int reps,
              int threads, long rss) {
  double per = nTest ? m.infer_ms_total / double(nTest) : 0.0;
  double sps =
      m.infer_ms_total > 0 ? double(nTest) / (m.infer_ms_total / 1000.0) : 0.0;
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(6);
  s << "{\n"
    << "  \"train_ms\": " << m.train_ms << ",\n"
    << "  \"train_ms_reps\": " << arrJson(trainReps) << ",\n"
    << "  \"infer_ms_total\": " << m.infer_ms_total << ",\n"
    << "  \"infer_ms_reps\": " << arrJson(inferReps) << ",\n"
    << "  \"infer_ms_per_sample\": " << per << ",\n"
    << "  \"throughput_sps\": " << sps << ",\n"
    << "  \"latency_ms_p50\": " << lat.p50 << ",\n"
    << "  \"latency_ms_p99\": " << lat.p99 << ",\n"
    << "  \"latency_ms_mean\": " << lat.mean << ",\n"
    << "  \"latency_ms_max\": " << lat.max << ",\n"
    << "  \"peak_rss_bytes\": " << rss << ",\n"
    << "  \"model_bytes\": " << m.model_bytes << ",\n"
    << "  \"accuracy\": " << m.accuracy << ",\n"
    << "  \"address_size\": " << addressSize << ",\n"
    << "  \"num_rams\": " << numRAMs << ",\n"
    << "  \"reps\": " << reps << ",\n"
    << "  \"threads\": " << threads << "\n"
    << "}\n";
  if (path.empty()) {
    std::fputs(s.str().c_str(), stdout);
  } else {
    std::ofstream f(path);
    f << s.str();
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
  LatencyDist lat = measureLatency(w, d);

  int th = activeThreads(threads);
  emitJson("", m, {m.train_ms}, {m.infer_ms_total}, lat, nTest, addressSize,
           d.numRAMs, 1, th, peakRssBytes());

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
  wp::Wisard w;
  for (int r = 0; r < reps; ++r) {
    w = wp::Wisard();
    w.build(d.numClasses, d.numRAMs, d.addressSize, d.mapping);
    last = runOnce(w, d, pred);
    trainTimes.push_back(last.train_ms);
    inferTimes.push_back(last.infer_ms_total);
  }

  LatencyDist lat = measureLatency(w, d);

  RunMetrics agg = last;
  agg.train_ms = median(trainTimes);
  agg.infer_ms_total = median(inferTimes);

  int th = activeThreads(threads);
  emitJson(out, agg, trainTimes, inferTimes, lat, d.nTest, d.addressSize,
           d.numRAMs, reps, th, peakRssBytes());
  if (!out.empty()) {
    std::printf(
        "accuracy=%.4f model_bytes=%ld train_ms=%.3f infer_ms=%.3f "
        "p99_ms=%.4f\n",
        agg.accuracy, agg.model_bytes, agg.train_ms, agg.infer_ms_total,
        lat.p99);
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
