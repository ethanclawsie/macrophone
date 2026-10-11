#pragma once

#include <CoreAudio/CoreAudioTypes.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Single-producer / single-consumer lock-free ring of mono float samples.
///
/// The producer (`macrophone_ring_write_abl`) runs on the Core Audio I/O thread: no locks, no allocation,
/// no Objective-C or Swift runtime. Everything else is called from the consumer side.
typedef struct macrophone_ring macrophone_ring;

macrophone_ring *macrophone_ring_create(uint32_t min_capacity);
void macrophone_ring_destroy(macrophone_ring *ring);

/// Producer: append `frames` frames of float32 audio, downmixed to mono. Drops samples on overflow.
void macrophone_ring_write_abl(macrophone_ring *ring, const AudioBufferList *abl, uint32_t frames, uint64_t host_time);

/// Consumer: read up to `max` samples into `dst`; returns the number read.
uint32_t macrophone_ring_read(macrophone_ring *ring, float *dst, uint32_t max);

/// Consumer: discard everything buffered so far (safe while the producer is running).
void macrophone_ring_discard(macrophone_ring *ring);

/// Consumer: arm capture of the host time of the next buffer written (see `macrophone_ring_first_host_time`).
void macrophone_ring_arm(macrophone_ring *ring);

/// mach host time of the first buffer written after `macrophone_ring_arm`, or 0 if none has arrived yet.
uint64_t macrophone_ring_first_host_time(const macrophone_ring *ring);

uint64_t macrophone_ring_dropped(const macrophone_ring *ring);

#ifdef __cplusplus
}
#endif
