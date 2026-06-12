#include "wisard.hpp"

#include <algorithm>
#include <vector>

#ifdef WP_OPENMP
#include <omp.h>
#endif

namespace wp {

void Wisard::train(const PackedInputs& in, const std::vector<uint32_t>& labels) {
  const uint32_t* idx = mapping_.data();

#ifdef WP_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int c = 0; c < int(numClasses_); ++c) {
    Discriminator& d = discriminators_[c];
    for (size_t s = 0; s < in.n; ++s) {
      if (labels[s] == uint32_t(c)) d.train(in, s, idx);
    }
  }
}

uint32_t Wisard::classifyOne(const PackedInputs& in, size_t s,
                             content_t* voteBuf) const {
  const uint32_t* idx = mapping_.data();
  const uint32_t C = numClasses_;
  const uint32_t R = numRAMs_;

  for (uint32_t c = 0; c < C; ++c) {
    discriminators_[c].votes(in, s, idx, voteBuf + size_t(c) * R);
  }

  std::vector<int> score(C, 0);
  int bleaching = 0;
  while (true) {
    int minPos = 0;
    bool first = true;
    for (uint32_t c = 0; c < C; ++c) {
      const content_t* v = voteBuf + size_t(c) * R;
      int sc = 0;
      for (uint32_t r = 0; r < R; ++r) {
        int val = int(v[r]);
        if (val > bleaching) {
          ++sc;
          if (first || val < minPos) {
            minPos = val;
            first = false;
          }
        }
      }
      score[c] = sc;
    }

    int biggest = 0;
    bool ambiguity = false;
    for (uint32_t c = 0; c < C; ++c) {
      if (score[c] > biggest) {
        biggest = score[c];
        ambiguity = false;
      } else if ((biggest - score[c]) < 1) {
        ambiguity = true;
      }
    }

    if (!(ambiguity && biggest > 1)) break;
    bleaching = minPos;
  }

  uint32_t winner = 0;
  int biggest = 0;
  for (uint32_t c = 0; c < C; ++c) {
    if (score[c] >= biggest) {
      biggest = score[c];
      winner = c;
    }
  }
  return winner;
}

void Wisard::classify(const PackedInputs& in, std::vector<uint32_t>& out) const {
  out.assign(in.n, 0);
  const size_t voteSize = size_t(numClasses_) * numRAMs_;

#ifdef WP_OPENMP
#pragma omp parallel
  {
    std::vector<content_t> voteBuf(voteSize);
#pragma omp for schedule(dynamic, 64)
    for (long s = 0; s < long(in.n); ++s) {
      out[s] = classifyOne(in, size_t(s), voteBuf.data());
    }
  }
#else
  std::vector<content_t> voteBuf(voteSize);
  for (size_t s = 0; s < in.n; ++s) {
    out[s] = classifyOne(in, s, voteBuf.data());
  }
#endif
}

}  // namespace wp
