/*
 * Copyright 2026 Claude / got-bringup
 * SPDX-License-Identifier: MIT
 */

/*
 * GPU-side timing for Honeykrisp.
 *
 * WHY THIS EXISTS
 * ---------------
 * Asahi exposes no GPU utilisation counter: not in debugfs, not as drm-engine-*
 * in fdinfo, and MangoHud has no AGX backend. Until now the only signal
 * available for "is the GPU busy?" was macsmc_hwmon Total System Power, which
 * is a whole-SoC number and cannot separate CPU from GPU, let alone vertex from
 * fragment from compute.
 *
 * But the firmware can timestamp work directly. drm_asahi_cmd_render carries
 * ts_vtx and ts_frag (start and end), and drm_asahi_cmd_compute carries ts --
 * the firmware writes the GPU timebase into a bound timestamp buffer at the
 * start and end of each phase. The gallium driver already uses this for
 * AGX_DBG_STATS; Honeykrisp wired up only ts_frag.end, and only to service
 * vkCmdWriteTimestamp. Every other slot went unused.
 *
 * This module claims those unused slots for profiling, so a run can be
 * described as: of N ms of wall time, the GPU spent V ms in vertex, F ms in
 * fragment, C ms in compute, and was busy at all for U ms.
 *
 * U -- the UNION of all the intervals -- is the number that matters. Vertex,
 * fragment and compute overlap on a TBDR by design, so the individual sums can
 * exceed wall time and say nothing on their own. The union cannot. If the union
 * is near 100%, the GPU is saturated and the only way forward is less work. If
 * it is well under, the GPU is idling inside the frame and the problem is
 * submission latency or serialisation, not capacity -- a completely different
 * investigation.
 *
 * USAGE
 *   HK_GPUTIME=<seconds>   report every <seconds> seconds to stderr (try 5)
 *
 * COST
 * Two u64 firmware writes per phase (which the hardware does anyway when asked)
 * plus a ~1 MB buffer walk four times a second. It does not meaningfully
 * perturb what it measures, which is the whole point of doing it in firmware
 * rather than by bracketing submits on the CPU.
 *
 * LIMITS
 *   - A command whose end slot is already claimed by a real vkCmdWriteTimestamp
 *     is not profiled; those are counted and reported as "skipped".
 *   - Intervals are collected in a ring. If it wraps before a scan, samples are
 *     lost, not corrupted. The ring is sized for ~27 s at Ghost of Tsushima's
 *     observed 610 control streams/second.
 */

#ifndef HK_GPUTIME_H
#define HK_GPUTIME_H

#include <stdbool.h>
#include <stdint.h>
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"

struct hk_device;
struct agx_bo;

/* Slots the firmware may write per command, in u64 units. Padded to 8 (one
 * 64-byte line) so that firmware writes for adjacent commands never share a
 * cache line with each other or with our scan.
 */
#define HK_GPUTIME_SLOT_VTX_START  0
#define HK_GPUTIME_SLOT_VTX_END    1
#define HK_GPUTIME_SLOT_FRAG_START 2
#define HK_GPUTIME_SLOT_FRAG_END   3
#define HK_GPUTIME_SLOT_COMP_START 4
#define HK_GPUTIME_SLOT_COMP_END   5
#define HK_GPUTIME_SLOTS_PER_BLOCK 8

enum hk_gputime_kind {
   HK_GPUTIME_VTX = 0,
   HK_GPUTIME_FRAG,
   HK_GPUTIME_COMP,
   HK_GPUTIME_KINDS,
};

struct hk_gputime_interval {
   uint64_t start, end;  /* raw firmware timebase */
   uint32_t kind;
};

struct hk_gputime {
   bool enabled;

   struct agx_bo *bo;
   uint64_t *blocks;      /* mapped, nr_blocks * HK_GPUTIME_SLOTS_PER_BLOCK */
   uint32_t handle;       /* kernel timestamp object handle, not a GEM handle */
   uint32_t nr_blocks;
   uint32_t next;         /* round-robin cursor, atomically incremented */

   simple_mtx_t lock;     /* guards everything below */
   struct util_dynarray intervals;
   uint64_t last_scan_ns;
   uint64_t report_start_ns;
   uint64_t report_period_ns;
   uint64_t skipped;      /* commands whose end slot the app had claimed */
   uint64_t presents;     /* frames delivered, counted at vkQueuePresentKHR */
};

void hk_gputime_init(struct hk_device *dev);
void hk_gputime_finish(struct hk_device *dev);

/*
 * Reserve a block for one command. Returns the block index, or -1 if profiling
 * is off. The caller fills the drm_asahi_timestamp fields it owns.
 */
int hk_gputime_reserve(struct hk_device *dev);

/* Byte offset of a slot within the timestamp object. */
static inline uint32_t
hk_gputime_offset(int block, unsigned slot)
{
   return (block * HK_GPUTIME_SLOTS_PER_BLOCK + slot) * sizeof(uint64_t);
}

/*
 * Called after each submit. Harvests completed timestamps and, once the report
 * period has elapsed, prints a summary. Cheap enough to call unconditionally.
 */
void hk_gputime_tick(struct hk_device *dev);

#endif
