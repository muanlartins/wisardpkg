// MurmurHash3 - public domain hash function by Austin Appleby.
// Simplified for wisardpkg Bloom filter double-hashing.
// We only need two independent 32-bit hashes from an input to do
// double hashing: h(i) = (h1 + i * h2) % numBits.

#ifndef MURMUR3_H
#define MURMUR3_H

#include <cstdint>

inline uint32_t murmur3_rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

inline uint32_t murmur3_fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

// MurmurHash3_x86_32: produces a single 32-bit hash.
// Call with different seeds to get independent hashes for double hashing.
inline uint32_t MurmurHash3_32(const void* key, int len, uint32_t seed) {
  const uint8_t* data = (const uint8_t*)key;
  const int nblocks = len / 4;

  uint32_t h1 = seed;
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  // body
  const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);
  for (int i = -nblocks; i; i++) {
    uint32_t k1 = blocks[i];
    k1 *= c1;
    k1 = murmur3_rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = murmur3_rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
  }

  // tail
  const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
  uint32_t k1 = 0;
  switch (len & 3) {
    case 3: k1 ^= (uint32_t)tail[2] << 16; // fallthrough
    case 2: k1 ^= (uint32_t)tail[1] << 8;  // fallthrough
    case 1: k1 ^= (uint32_t)tail[0];
            k1 *= c1; k1 = murmur3_rotl32(k1, 15); k1 *= c2; h1 ^= k1;
  }

  // finalization
  h1 ^= len;
  h1 = murmur3_fmix32(h1);
  return h1;
}

#endif // MURMUR3_H
