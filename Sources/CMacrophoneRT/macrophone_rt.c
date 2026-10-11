#include "macrophone_rt.h"

#include <stdatomic.h>
#include <stdlib.h>

struct macrophone_ring {
    float *buf;
    uint64_t mask;
    _Atomic uint64_t write_pos; // total samples ever written
    _Atomic uint64_t read_pos;  // total samples ever read
    _Atomic uint64_t dropped;
    _Atomic uint64_t first_host_time;
    _Atomic bool armed;
};

macrophone_ring *macrophone_ring_create(uint32_t min_capacity) {
    uint64_t cap = 1;
    while (cap < min_capacity) cap <<= 1;
    macrophone_ring *ring = calloc(1, sizeof *ring);
    if (!ring) return NULL;
    ring->buf = calloc(cap, sizeof(float));
    if (!ring->buf) {
        free(ring);
        return NULL;
    }
    ring->mask = cap - 1;
    return ring;
}

void macrophone_ring_destroy(macrophone_ring *ring) {
    if (!ring) return;
    free(ring->buf);
    free(ring);
}

void macrophone_ring_write_abl(macrophone_ring *ring, const AudioBufferList *abl, uint32_t frames, uint64_t host_time) {
    if (!ring || !abl || abl->mNumberBuffers == 0 || frames == 0) return;

    if (atomic_load_explicit(&ring->armed, memory_order_acquire)) {
        atomic_store_explicit(&ring->first_host_time, host_time, memory_order_relaxed);
        atomic_store_explicit(&ring->armed, false, memory_order_release);
    }

    const uint64_t w = atomic_load_explicit(&ring->write_pos, memory_order_relaxed);
    const uint64_t r = atomic_load_explicit(&ring->read_pos, memory_order_acquire);
    const uint64_t space = (ring->mask + 1) - (w - r);
    uint32_t n = frames;
    if (n > space) {
        atomic_fetch_add_explicit(&ring->dropped, n - space, memory_order_relaxed);
        n = (uint32_t)space;
    }

    // AVAudioEngine delivers deinterleaved float32 (one buffer per channel); interleaved buffers
    // contribute their first channel. Channels are averaged into mono.
    const uint32_t nbuf = abl->mNumberBuffers;
    const float scale = 1.0f / (float)nbuf;
    for (uint32_t i = 0; i < n; i++) {
        float acc = 0.0f;
        for (uint32_t b = 0; b < nbuf; b++) {
            const AudioBuffer *ab = &abl->mBuffers[b];
            const float *src = (const float *)ab->mData;
            const uint32_t stride = ab->mNumberChannels ? ab->mNumberChannels : 1;
            if (src) acc += src[(size_t)i * stride];
        }
        ring->buf[(w + i) & ring->mask] = acc * scale;
    }
    atomic_store_explicit(&ring->write_pos, w + n, memory_order_release);
}

uint32_t macrophone_ring_read(macrophone_ring *ring, float *dst, uint32_t max) {
    const uint64_t r = atomic_load_explicit(&ring->read_pos, memory_order_relaxed);
    const uint64_t w = atomic_load_explicit(&ring->write_pos, memory_order_acquire);
    const uint64_t avail = w - r;
    const uint32_t n = avail < max ? (uint32_t)avail : max;
    for (uint32_t i = 0; i < n; i++) dst[i] = ring->buf[(r + i) & ring->mask];
    atomic_store_explicit(&ring->read_pos, r + n, memory_order_release);
    return n;
}

void macrophone_ring_discard(macrophone_ring *ring) {
    const uint64_t w = atomic_load_explicit(&ring->write_pos, memory_order_acquire);
    atomic_store_explicit(&ring->read_pos, w, memory_order_release);
}

void macrophone_ring_arm(macrophone_ring *ring) {
    atomic_store_explicit(&ring->first_host_time, 0, memory_order_relaxed);
    atomic_store_explicit(&ring->armed, true, memory_order_release);
}

uint64_t macrophone_ring_first_host_time(const macrophone_ring *ring) {
    return atomic_load_explicit(&ring->first_host_time, memory_order_relaxed);
}

uint64_t macrophone_ring_dropped(const macrophone_ring *ring) {
    return atomic_load_explicit(&ring->dropped, memory_order_relaxed);
}
