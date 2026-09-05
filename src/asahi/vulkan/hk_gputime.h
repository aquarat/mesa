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
#include <vulkan/vulkan_core.h>
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

/*
 * Where a compute dispatch came from. Honeykrisp issues compute for far more
 * than vkCmdDispatch: on AGX the geometry-shader/pre-rasterization path runs
 * the VERTEX shader as a compute dispatch, and tessellation is emulated
 * entirely in compute. With ~1750 dispatches per frame in Ghost of Tsushima and
 * 90% of GPU time in compute, which of these it is decides everything.
 */
enum hk_disp_kind {
   HK_DISP_APP = 0,   /* vkCmdDispatch* -- the application's own compute */
   HK_DISP_META,      /* vk_meta copies/blits, which also go via vkCmdDispatch */
   HK_DISP_GS,        /* geometry shader / pre-rasterization, incl. VS-in-CS */
   HK_DISP_TESS,      /* tessellation emulation */
   HK_DISP_PRECOMP,   /* libagx helper kernels (clears, fills, prefix sums...) */
   HK_DISP_KINDS,
};

/*
 * Per-shader accounting.
 *
 * "The application's own compute shaders are 72% of GPU time" is not yet an
 * actionable statement: it does not say whether that is five shaders run 250
 * times each or two hundred run six times. Optimisation cannot start until the
 * work is ranked, so every dispatch is attributed to the shader that ran.
 *
 * The key is the agx_shader_info pointer, which is stable for the lifetime of a
 * compiled variant and unique per variant. Alongside the dynamic counts we keep
 * the compiler's own static estimates (agx2_stats: ALU/FSCIB/IC cycles, GPRs,
 * achievable occupancy, spills) so the report can rank by
 *
 *     invocations x cycles-per-invocation
 *
 * which is the closest thing to a cost model available without per-dispatch
 * timestamps -- the firmware timestamps a whole control stream, and a control
 * stream here holds ~32 dispatches.
 */
#define HK_GPUTIME_MAX_SHADERS 1024
#define HK_GPUTIME_MAX_PRECOMP 128

struct agx_shader_info;

struct hk_gputime_shader {
   const struct agx_shader_info *info; /* identity; NULL means a free slot */
   uint32_t id;                        /* small stable index, for reports */
   uint32_t kind;                      /* enum hk_disp_kind of first use */
   uint64_t dispatches;
   uint64_t indirect;                  /* dispatches with an indirect grid */
   uint64_t threads;                   /* invocations launched (direct only) */

   /* MEASURED firmware GPU time for control streams that held only this
    * shader, and how many such streams there were. This is the number the
    * cycle estimate cannot produce: it includes memory stalls, spill traffic
    * and everything else the static model is blind to.
    */
   uint64_t measured_ticks;
   uint64_t measured_streams;
};

/* A control stream whose dispatches came from more than one shader. */
#define HK_GPUTIME_MIXED ((const struct agx_shader_info *)(uintptr_t)1)

struct hk_gputime_precomp {
   uint64_t dispatches;
   uint64_t threads;
   uint64_t indirect;
};

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
   uint64_t dispatches;   /* compute dispatches inside those commands */
   uint64_t draws;        /* draws inside the render commands */
   uint64_t disp_kind[HK_DISP_KINDS];

   /* Recorded at the CDM chokepoint rather than at submit, so it can be
    * compared against `dispatches` (which is summed at submit) to show whether
    * record-time attribution and submit-time counting agree.
    */
   uint64_t disp_recorded;

   /* CDM commands actually built at submit. Compared against the number of
    * harvested compute intervals, this says whether timestamps are being lost
    * between command creation and harvest -- which decides whether per-command
    * attribution means anything.
    */
   uint64_t cdm_commands;
   uint64_t cdm_timestamped;
   uint64_t isolate_splits;   /* times the isolate path ended a stream */

   /* vkCmdPipelineBarrier2 currently ends BOTH control streams outright
    * ("the big hammer", hk_cmd_buffer.c). A stream end is not a cache
    * operation, it is a full drain and a firmware round trip, and the game
    * runs ~56 compute streams per frame against ~1750 dispatches -- so if the
    * barriers driving those ends are compute-only, they could be served by an
    * in-stream CDM barrier instead and the drain avoided.
    *
    * Whether that is worth doing depends entirely on how vkd3d-proton spells
    * its barriers: D3D12 resource barriers can translate to ALL_COMMANDS,
    * which no in-stream barrier can honour. So count them before changing
    * anything.
    */
   uint64_t barriers;         /* vkCmdPipelineBarrier2 calls */
   uint64_t barriers_compute; /* ...whose stages are all compute-servable */
   uint64_t barriers_gfx_open; /* ...rejected only because gfx was open */
   /* Compute-only barriers that would actually save a drain: ones where a
    * compute stream is open AND has work in it. A barrier that ends an empty
    * stream costs nothing, so counting all compute-only barriers overstates
    * the prize. This is the number of stream ends per frame that
    * HK_PERFTEST=csbarrier can remove.
    */
   uint64_t barriers_avoidable;

   /* Commands submitted, and how many had to wait on the OTHER subqueue.
    * Cross-subqueue overlap is only possible for the ones that did not, so
    * this says directly how much of the workload the dependency tracking is
    * actually freeing rather than leaving serialised.
    */
   uint64_t cmds_submitted[2];
   uint64_t cmds_crossed[2];

   /* Set when HK_GPUTIME_ISOLATE=1: end the compute control stream after every
    * dispatch, so each one is timed individually. This perturbs -- it turns ~56
    * control streams per frame into ~1780 -- but a stream costs about 2 us to
    * start (measured, tests/disptest.c) against the ~2400 us these streams
    * actually take, so the ranking it produces is trustworthy even though the
    * absolute frame time is not.
    */
   bool isolate;

   /* Which shader owns the command that reserved each block, so a harvested
    * interval can be charged to it. Indexed by block.
    */
   const struct agx_shader_info **block_shader;

   simple_mtx_t shader_lock; /* guards the two tables below */
   struct hk_gputime_shader shaders[HK_GPUTIME_MAX_SHADERS];
   uint32_t nr_shaders;
   uint64_t shader_overflow; /* dispatches dropped when the table filled */
   struct hk_gputime_precomp precomp[HK_GPUTIME_MAX_PRECOMP];
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

/*
 * Attribute one dispatch, called from the two wrappers that between them reach
 * every CDM launch. Both are past their early-returns, so the totals reconcile
 * exactly with cs->stats.cmds.
 *
 * `threads` is the launch grid in invocations, valid only when !indirect: an
 * indirect grid lives in GPU memory and the CPU cannot know it here.
 */
void hk_gputime_note_shader(struct hk_device *dev, enum hk_disp_kind kind,
                            const struct agx_shader_info *info,
                            uint64_t threads, bool indirect);

void hk_gputime_note_precomp(struct hk_device *dev, unsigned prog,
                             uint64_t threads, bool indirect);

/* Classify one vkCmdPipelineBarrier2 for the report. `gfx_open` says whether a
 * graphics control stream was in progress, which alone forces the hammer. */
void hk_gputime_note_barrier(struct hk_device *dev, bool compute_only,
                             bool gfx_open, bool ends_real_work);

/* One submitted hardware command: `compute` selects the subqueue, `crossed`
 * says whether it had to wait on the other one. */
void hk_gputime_note_submit(struct hk_device *dev, bool compute, bool crossed);

/* Record which shader owns the command that reserved this block. */
void hk_gputime_set_block_owner(struct hk_device *dev, int blk,
                                const struct agx_shader_info *info);

#endif
