/*
 * Copyright 2026 Claude / got-bringup
 * SPDX-License-Identifier: MIT
 */

#include "hk_gputime.h"

#include <stdio.h>
#include <stdlib.h>

#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_device.h"
#include "hk_device.h"
#include "libagx_shaders.h"
#include "shader_enums.h"

#include "util/os_time.h"
#include "util/u_atomic.h"

/* ~27 s of headroom at the 610 control streams/second observed in Ghost of
 * Tsushima, 2.7 s at ten times that rate. 1 MB.
 */
#define HK_GPUTIME_BLOCKS 16384

void
hk_gputime_init(struct hk_device *dev)
{
   struct hk_gputime *gt = &dev->gputime;
   memset(gt, 0, sizeof(*gt));

   const char *env = getenv("HK_GPUTIME");
   if (!env || !env[0])
      return;

   double period_s = atof(env);
   if (period_s <= 0.0)
      return;

   size_t size =
      (size_t)HK_GPUTIME_BLOCKS * HK_GPUTIME_SLOTS_PER_BLOCK * sizeof(uint64_t);

   /* The kernel requires timestamp buffers be SHARED, and we read them from the
    * CPU every scan, so writeback caching too.
    */
   gt->bo = agx_bo_create(&dev->dev, size, 0, AGX_BO_WRITEBACK | AGX_BO_SHARED,
                          "GPU time profiling");
   if (!gt->bo) {
      fprintf(stderr, "[hk gputime] could not allocate the timestamp buffer; "
                      "profiling disabled\n");
      return;
   }

   if (agx_bind_timestamps(&dev->dev, gt->bo, &gt->handle) || !gt->handle) {
      fprintf(stderr, "[hk gputime] could not bind the timestamp buffer; "
                      "profiling disabled\n");
      agx_bo_unreference(&dev->dev, gt->bo);
      gt->bo = NULL;
      return;
   }

   gt->blocks = agx_bo_map(gt->bo);
   memset(gt->blocks, 0, size);

   gt->nr_blocks = HK_GPUTIME_BLOCKS;
   gt->isolate = getenv("HK_GPUTIME_ISOLATE") != NULL;
   gt->block_shader = calloc(gt->nr_blocks, sizeof(*gt->block_shader));
   if (!gt->block_shader) {
      agx_bo_unreference(&dev->dev, gt->bo);
      gt->bo = NULL;
      return;
   }
   gt->report_period_ns = (uint64_t)(period_s * 1000000000.0);
   gt->last_scan_ns = gt->report_start_ns = os_time_get_nano();
   simple_mtx_init(&gt->lock, mtx_plain);
   simple_mtx_init(&gt->shader_lock, mtx_plain);
   util_dynarray_init(&gt->intervals, NULL);
   gt->enabled = true;

   fprintf(stderr,
           "[hk gputime] firmware GPU timing active, reporting every %.2f s "
           "(timebase %llu Hz)%s\n",
           period_s,
           (unsigned long long)dev->dev.params.command_timestamp_frequency_hz,
           gt->isolate ? "  [ISOLATE: one dispatch per control stream, "
                         "absolute times are perturbed]" : "");
}

void
hk_gputime_set_block_owner(struct hk_device *dev, int blk,
                           const struct agx_shader_info *info)
{
   struct hk_gputime *gt = &dev->gputime;
   if (gt->enabled && blk >= 0 && (uint32_t)blk < gt->nr_blocks)
      gt->block_shader[blk] = info;
}

void
hk_gputime_finish(struct hk_device *dev)
{
   struct hk_gputime *gt = &dev->gputime;
   if (!gt->enabled)
      return;

   gt->enabled = false;
   util_dynarray_fini(&gt->intervals);
   simple_mtx_destroy(&gt->lock);
   simple_mtx_destroy(&gt->shader_lock);
   agx_bo_unreference(&dev->dev, gt->bo);
   gt->bo = NULL;
   free(gt->block_shader);
   gt->block_shader = NULL;
}

/*
 * Names for the libagx helper kernels. Generated from the enum in
 * libagx_shaders.h; the static_assert below fires if that enum grows, which is
 * the only way this can go stale.
 */
static const char *libagx_program_name[] = {
   [LIBAGX_DECOMPRESS_0] = "decompress_0",
   [LIBAGX_DECOMPRESS_1] = "decompress_1",
   [LIBAGX_DECOMPRESS_2] = "decompress_2",
   [LIBAGX_FAST_CLEAR_0] = "fast_clear_0",
   [LIBAGX_FAST_CLEAR_1] = "fast_clear_1",
   [LIBAGX_FAST_CLEAR_2] = "fast_clear_2",
   [LIBAGX_FAST_CLEAR_3] = "fast_clear_3",
   [LIBAGX_FAST_CLEAR_4] = "fast_clear_4",
   [LIBAGX_FAST_CLEAR_5] = "fast_clear_5",
   [LIBAGX_FAST_CLEAR_6] = "fast_clear_6",
   [LIBAGX_FAST_CLEAR_7] = "fast_clear_7",
   [LIBAGX_FAST_CLEAR_8] = "fast_clear_8",
   [LIBAGX_FAST_CLEAR_9] = "fast_clear_9",
   [LIBAGX_FAST_CLEAR_10] = "fast_clear_10",
   [LIBAGX_FAST_CLEAR_11] = "fast_clear_11",
   [LIBAGX_FAST_CLEAR_12] = "fast_clear_12",
   [LIBAGX_FAST_CLEAR_13] = "fast_clear_13",
   [LIBAGX_FAST_CLEAR_14] = "fast_clear_14",
   [LIBAGX_FILL] = "fill",
   [LIBAGX_COPY_UINT4] = "copy_uint4",
   [LIBAGX_COPY_UCHAR] = "copy_uchar",
   [LIBAGX_FILL_UINT4] = "fill_uint4",
   [LIBAGX_PREDICATE_INDIRECT_0] = "predicate_indirect_0",
   [LIBAGX_PREDICATE_INDIRECT_1] = "predicate_indirect_1",
   [LIBAGX_DRAW_WITHOUT_ADJ] = "draw_without_adj",
   [LIBAGX_DRAW_ROBUST_INDEX_0] = "draw_robust_index_0",
   [LIBAGX_DRAW_ROBUST_INDEX_1] = "draw_robust_index_1",
   [LIBAGX_DRAW_ROBUST_INDEX_2] = "draw_robust_index_2",
   [LIBAGX_INCREMENT_IA] = "increment_ia",
   [LIBAGX_INCREMENT_IA_RESTART] = "increment_ia_restart",
   [LIBAGX_UNROLL_RESTART_0] = "unroll_restart_0",
   [LIBAGX_UNROLL_RESTART_1] = "unroll_restart_1",
   [LIBAGX_UNROLL_RESTART_2] = "unroll_restart_2",
   [LIBAGX_UNROLL_RESTART_3] = "unroll_restart_3",
   [LIBAGX_UNROLL_RESTART_4] = "unroll_restart_4",
   [LIBAGX_UNROLL_RESTART_5] = "unroll_restart_5",
   [LIBAGX_UNROLL_RESTART_6] = "unroll_restart_6",
   [LIBAGX_UNROLL_RESTART_7] = "unroll_restart_7",
   [LIBAGX_UNROLL_RESTART_8] = "unroll_restart_8",
   [LIBAGX_UNROLL_RESTART_9] = "unroll_restart_9",
   [LIBAGX_UNROLL_RESTART_10] = "unroll_restart_10",
   [LIBAGX_GS_SETUP_INDIRECT] = "gs_setup_indirect",
   [LIBAGX_PREFIX_SUM_GEOM] = "prefix_sum_geom",
   [LIBAGX_PREFIX_SUM_TESS_0] = "prefix_sum_tess_0",
   [LIBAGX_PREFIX_SUM_TESS_1] = "prefix_sum_tess_1",
   [LIBAGX_PREFIX_SUM_TESS_REDUCE] = "prefix_sum_tess_reduce",
   [LIBAGX_PREFIX_SUM_TESS_SCAN_0] = "prefix_sum_tess_scan_0",
   [LIBAGX_PREFIX_SUM_TESS_SCAN_1] = "prefix_sum_tess_scan_1",
   [LIBAGX_COPY_QUERY] = "copy_query",
   [LIBAGX_RESET_QUERY] = "reset_query",
   [LIBAGX_COPY_QUERY_GL] = "copy_query_gl",
   [LIBAGX_COPY_XFB_COUNTERS] = "copy_xfb_counters",
   [LIBAGX_INCREMENT_STATISTIC] = "increment_statistic",
   [LIBAGX_INCREMENT_CS_INVOCATIONS] = "increment_cs_invocations",
   [LIBAGX_WRITE_U32S] = "write_u32s",
   [LIBAGX_COPY_TIMESTAMP] = "copy_timestamp",
   [LIBAGX_WRITE_U32] = "write_u32",
   [LIBAGX_TESS_SETUP_INDIRECT] = "tess_setup_indirect",
   [LIBAGX_TESS_ISOLINE_0] = "tess_isoline_0",
   [LIBAGX_TESS_ISOLINE_1] = "tess_isoline_1",
   [LIBAGX_TESS_TRI_0] = "tess_tri_0",
   [LIBAGX_TESS_TRI_1] = "tess_tri_1",
   [LIBAGX_TESS_QUAD_0] = "tess_quad_0",
   [LIBAGX_TESS_QUAD_1] = "tess_quad_1",
   [LIBAGX_HELPER] = "helper",
};
static_assert(ARRAY_SIZE(libagx_program_name) == LIBAGX_NUM_PROGRAMS,
              "libagx program names are out of date with libagx_shaders.h");

void
hk_gputime_note_barrier(struct hk_device *dev, bool compute_only,
                        bool gfx_open, bool ends_real_work)
{
   struct hk_gputime *gt = &dev->gputime;
   if (!gt->enabled)
      return;

   simple_mtx_lock(&gt->lock);
   gt->barriers++;
   if (compute_only) {
      if (gfx_open) {
         gt->barriers_gfx_open++;
      } else {
         gt->barriers_compute++;
         if (ends_real_work)
            gt->barriers_avoidable++;
      }
   }
   simple_mtx_unlock(&gt->lock);
}

void
hk_gputime_note_precomp(struct hk_device *dev, unsigned prog, uint64_t threads,
                        bool indirect)
{
   struct hk_gputime *gt = &dev->gputime;
   if (!gt->enabled)
      return;

   p_atomic_inc(&gt->disp_kind[HK_DISP_PRECOMP]);
   p_atomic_inc(&gt->disp_recorded);

   if (prog >= HK_GPUTIME_MAX_PRECOMP)
      return;

   struct hk_gputime_precomp *p = &gt->precomp[prog];
   p_atomic_inc(&p->dispatches);
   if (indirect)
      p_atomic_inc(&p->indirect);
   else
      p_atomic_add(&p->threads, threads);
}

/*
 * Find or create the table entry for a shader. Open addressing on the pointer:
 * there are a few hundred live shaders at most, the table is oversized, and the
 * probe is a handful of cache lines. Called under shader_lock.
 */
static struct hk_gputime_shader *
hk_gputime_shader_entry(struct hk_gputime *gt, const struct agx_shader_info *info,
                        enum hk_disp_kind kind)
{
   uint32_t mask = HK_GPUTIME_MAX_SHADERS - 1;
   uint32_t h = (uint32_t)(((uintptr_t)info >> 4) * 2654435761u) & mask;

   for (uint32_t probe = 0; probe <= mask; ++probe) {
      struct hk_gputime_shader *e = &gt->shaders[(h + probe) & mask];

      if (e->info == info)
         return e;

      if (e->info == NULL) {
         e->info = info;
         e->id = gt->nr_shaders++;
         e->kind = kind;
         return e;
      }
   }

   return NULL;
}

void
hk_gputime_note_shader(struct hk_device *dev, enum hk_disp_kind kind,
                       const struct agx_shader_info *info, uint64_t threads,
                       bool indirect)
{
   struct hk_gputime *gt = &dev->gputime;
   if (!gt->enabled)
      return;

   p_atomic_inc(&gt->disp_kind[kind]);
   p_atomic_inc(&gt->disp_recorded);

   simple_mtx_lock(&gt->shader_lock);
   struct hk_gputime_shader *e = hk_gputime_shader_entry(gt, info, kind);
   if (e) {
      e->dispatches++;
      if (indirect)
         e->indirect++;
      else
         e->threads += threads;
   } else {
      gt->shader_overflow++;
   }
   simple_mtx_unlock(&gt->shader_lock);
}

int
hk_gputime_reserve(struct hk_device *dev)
{
   struct hk_gputime *gt = &dev->gputime;
   if (!gt->enabled)
      return -1;

   uint32_t idx = p_atomic_inc_return(&gt->next) - 1;
   return idx % gt->nr_blocks;
}

/*
 * Move one completed start/end pair out of the ring and into the interval list.
 *
 * A slot the firmware has not written reads as zero, so a zero end means the
 * command is still in flight -- leave the pair alone and pick it up next scan.
 * Only a consumed pair is cleared, which is what makes a partially-written
 * block safe to observe.
 */
static void
harvest(struct hk_gputime *gt, uint64_t *blk, unsigned s, unsigned e,
        enum hk_gputime_kind kind, uint32_t block_idx)
{
   uint64_t start = blk[s], end = blk[e];

   if (!start || !end)
      return;

   blk[s] = 0;
   blk[e] = 0;

   if (end <= start)
      return;

   struct hk_gputime_interval iv = {.start = start, .end = end, .kind = kind};
   util_dynarray_append(&gt->intervals, iv);

   /* Charge the elapsed time to the shader that owned this control stream, if
    * exactly one did. This is MEASURED time, so unlike the cycle estimate it
    * includes memory stalls and spill traffic -- which is the whole reason a
    * shader can dominate the frame while the estimate calls it a minority.
    */
   if (kind == HK_GPUTIME_COMP && gt->block_shader) {
      const struct agx_shader_info *owner = gt->block_shader[block_idx];
      gt->block_shader[block_idx] = NULL;

      if (owner && owner != HK_GPUTIME_MIXED) {
         simple_mtx_lock(&gt->shader_lock);
         struct hk_gputime_shader *ent =
            hk_gputime_shader_entry(gt, owner, HK_DISP_APP);
         if (ent) {
            ent->measured_ticks += end - start;
            ent->measured_streams++;
         }
         simple_mtx_unlock(&gt->shader_lock);
      }
   }
}

/* lock held */
static void
hk_gputime_scan(struct hk_gputime *gt)
{
   for (uint32_t b = 0; b < gt->nr_blocks; ++b) {
      uint64_t *blk = gt->blocks + (size_t)b * HK_GPUTIME_SLOTS_PER_BLOCK;

      harvest(gt, blk, HK_GPUTIME_SLOT_VTX_START, HK_GPUTIME_SLOT_VTX_END,
              HK_GPUTIME_VTX, b);
      harvest(gt, blk, HK_GPUTIME_SLOT_FRAG_START, HK_GPUTIME_SLOT_FRAG_END,
              HK_GPUTIME_FRAG, b);
      harvest(gt, blk, HK_GPUTIME_SLOT_COMP_START, HK_GPUTIME_SLOT_COMP_END,
              HK_GPUTIME_COMP, b);
   }
}

static const char *
hk_disp_kind_name(uint32_t kind)
{
   static const char *n[HK_DISP_KINDS] = {"app", "meta", "gs", "tess",
                                          "precomp"};
   return kind < HK_DISP_KINDS ? n[kind] : "?";
}

/*
 * Cost model.
 *
 * The firmware timestamps a control stream, not a dispatch, and a compute
 * control stream here holds ~32 dispatches -- so there is no way to measure one
 * shader directly without splitting streams, which would perturb the thing
 * being measured beyond usefulness. Instead: invocations (known exactly, from
 * the launch grid) times the compiler's own per-invocation cycle estimate.
 *
 * agx2_stats carries three estimates, one per issue pipe: alu, fscib (F16/F32
 * and the SCIB path) and ic. A shader is limited by whichever pipe is busiest,
 * so the max is the right per-invocation figure -- not the sum, which would
 * assume the pipes never overlap.
 *
 * This is an estimate and is blind to memory stalls, which is exactly why the
 * occupancy column ("thr", max threads in flight per core, which register
 * pressure caps) is printed next to it: a shader with a low estimate and low
 * occupancy is a memory-bound shader the model will under-rank.
 */
static uint64_t
hk_shader_cycles_per_invocation(const struct agx_shader_info *info)
{
   uint32_t c = MAX2(info->stats.alu, info->stats.fscib);
   return MAX2(c, info->stats.ic);
}

static int
cmp_shader_cost(const void *a, const void *b)
{
   const struct hk_gputime_shader *x = a, *y = b;

   /* Measured time beats the estimate whenever it exists: the estimate cannot
    * see memory stalls, and a spilling shader is exactly where the two diverge.
    */
   if (x->measured_ticks || y->measured_ticks) {
      if (x->measured_ticks != y->measured_ticks)
         return x->measured_ticks > y->measured_ticks ? -1 : 1;
   }

   uint64_t cx = x->threads * hk_shader_cycles_per_invocation(x->info);
   uint64_t cy = y->threads * hk_shader_cycles_per_invocation(y->info);

   if (cx != cy)
      return cx > cy ? -1 : 1;
   return (x->dispatches < y->dispatches) - (x->dispatches > y->dispatches);
}

/* lock held */
static void
hk_gputime_report_shaders(struct hk_gputime *gt, uint64_t frames, double hz)
{
   double per_frame = frames ? 1.0 / (double)frames : 0.0;

   simple_mtx_lock(&gt->shader_lock);

   /* Compact the live entries out of the open-addressed table so they can be
    * sorted without disturbing the probe sequence.
    */
   /* static, not on the stack: 40 KB of frame in a driver thread is a poor
    * trade for a table that is only ever touched under shader_lock. */
   static struct hk_gputime_shader live[HK_GPUTIME_MAX_SHADERS];
   unsigned n = 0;
   uint64_t total_cycles = 0, total_measured = 0;

   for (unsigned i = 0; i < HK_GPUTIME_MAX_SHADERS; ++i) {
      struct hk_gputime_shader *e = &gt->shaders[i];
      if (!e->info || !e->dispatches)
         continue;

      live[n++] = *e;
      total_cycles += e->threads * hk_shader_cycles_per_invocation(e->info);
      total_measured += e->measured_ticks;

      /* Keep the identity and the id, drop the counts: the report is per
       * period, but a shader must keep the same id across periods to be
       * followed from one report to the next.
       */
      e->dispatches = 0;
      e->indirect = 0;
      e->threads = 0;
      e->measured_ticks = 0;
      e->measured_streams = 0;
   }

   qsort(live, n, sizeof(*live), cmp_shader_cost);

   if (n) {
      fprintf(stderr,
              "[hk gputime]   top compute shaders (%u live, %.3f Gcycle "
              "estimated total, %.1f ms measured over %s streams)\n",
              n, total_cycles / 1.0e9, (total_measured / hz) * 1000.0,
              total_measured ? "single-shader" : "no");
      fprintf(stderr,
              "[hk gputime]     %-4s %-5s %-8s %-10s %-8s %9s %7s  %5s %5s "
              "%5s %5s %5s %9s %6s %5s %-5s %s\n",
              "id", "kind", "disp/fr", "invoc/fr", "est%", "meas ms/fr",
              "meas%", "cyc", "inst", "gprs", "thr", "loop", "spill:fill",
              "scrat", "wg", "stage", "spirv");

      for (unsigned i = 0; i < MIN2(n, 20); ++i) {
         struct hk_gputime_shader *e = &live[i];
         const struct agx_shader_info *in = e->info;
         uint64_t cyc = hk_shader_cycles_per_invocation(in);
         uint64_t cost = e->threads * cyc;

         /* The SPIR-V hash is the join key to the disassembly: run the game
          * once with AGX_MESA_DEBUG=shaders (and MESA_SHADER_CACHE_DISABLE=true,
          * or nothing is compiled the second time) and the NIR header above
          * each dump carries the same source_blake3.
          */
         char blake[BLAKE3_HEX_LEN];
         _mesa_blake3_format(blake, in->source_blake3);
         blake[12] = '\0';

         double meas_ms = (e->measured_ticks / hz) * 1000.0;

         fprintf(stderr,
                 "[hk gputime]     %-4u %-5s %8.1f %10.0f %7.1f%% %9.2f %6.1f%%"
                 "  %5llu %5u %5u %5u %5u %4u:%-4u %6u %5u %-5s %s%s\n",
                 e->id, hk_disp_kind_name(e->kind),
                 e->dispatches * per_frame, e->threads * per_frame,
                 total_cycles ? 100.0 * cost / total_cycles : 0.0,
                 meas_ms * per_frame,
                 total_measured ? 100.0 * e->measured_ticks / total_measured
                                : 0.0,
                 (unsigned long long)cyc, in->stats.instrs, in->stats.gprs,
                 in->stats.threads,
                 /* Hardware loop count. A shader with loops can execute
                  * unboundedly more instructions than `inst` suggests, which is
                  * exactly where the static cycle estimate stops meaning
                  * anything -- measured locally at 2894 us for a single
                  * 32-invocation dispatch with a data-dependent loop
                  * (tests/cstest.c). */
                 in->stats.loops, in->stats.spills, in->stats.fills,
                 /* Scratch is private memory the shader addresses indirectly.
                  * It is reported separately from spill/fill because the two
                  * have the same symptom -- memory traffic per invocation the
                  * static cycle estimate is blind to -- but completely
                  * different cures: spills want less register pressure,
                  * scratch wants the indirectly-indexed array gone. */
                 in->stats.scratch,
                 in->workgroup_size[0] * in->workgroup_size[1] *
                    in->workgroup_size[2],
                 _mesa_shader_stage_to_abbrev(in->stage), blake,
                 e->indirect ? " (indirect)" : "");
      }
   }

   if (gt->shader_overflow) {
      fprintf(stderr, "[hk gputime]   %llu dispatch(es) lost, shader table full\n",
              (unsigned long long)gt->shader_overflow);
      gt->shader_overflow = 0;
   }

   /* The libagx helper kernels. This is the bucket that was unattributed: the
    * driver's own clears, fills, copies, prefix sums and query bookkeeping.
    */
   struct precomp_row {
      unsigned prog;
      struct hk_gputime_precomp p;
   };
   static struct precomp_row pc[HK_GPUTIME_MAX_PRECOMP];
   unsigned np = 0;
   uint64_t pc_total = 0;

   for (unsigned i = 0; i < HK_GPUTIME_MAX_PRECOMP; ++i) {
      if (!gt->precomp[i].dispatches)
         continue;

      pc[np].prog = i;
      pc[np].p = gt->precomp[i];
      np++;
      pc_total += gt->precomp[i].dispatches;
      memset(&gt->precomp[i], 0, sizeof(gt->precomp[i]));
   }

   /* Sort by dispatch count; the struct's first member is the count's home so
    * reuse the interval comparator shape via a small local sort instead.
    */
   for (unsigned i = 1; i < np; ++i) {
      for (unsigned j = i; j > 0 && pc[j].p.dispatches > pc[j - 1].p.dispatches;
           --j) {
         struct precomp_row t = pc[j];
         pc[j] = pc[j - 1];
         pc[j - 1] = t;
      }
   }

   if (np) {
      fprintf(stderr,
              "[hk gputime]   driver helper kernels: %llu dispatches "
              "(%.1f/frame) across %u kernels\n",
              (unsigned long long)pc_total, pc_total * per_frame, np);

      for (unsigned i = 0; i < MIN2(np, 12); ++i) {
         const char *name = pc[i].prog < ARRAY_SIZE(libagx_program_name)
                               ? libagx_program_name[pc[i].prog]
                               : "?";
         fprintf(stderr,
                 "[hk gputime]     %-24s %8.1f disp/fr  %10.0f invoc/fr"
                 "  %5.1f%%\n",
                 name, pc[i].p.dispatches * per_frame,
                 pc[i].p.threads * per_frame,
                 pc_total ? 100.0 * pc[i].p.dispatches / pc_total : 0.0);
      }
   }

   simple_mtx_unlock(&gt->shader_lock);
}

static int
cmp_interval(const void *a, const void *b)
{
   uint64_t x = ((const struct hk_gputime_interval *)a)->start;
   uint64_t y = ((const struct hk_gputime_interval *)b)->start;
   return (x > y) - (x < y);
}

/* lock held */
static void
hk_gputime_report(struct hk_device *dev, uint64_t now_ns)
{
   struct hk_gputime *gt = &dev->gputime;

   double wall_ms = (now_ns - gt->report_start_ns) / 1.0e6;
   double hz = (double)dev->dev.params.command_timestamp_frequency_hz;

   unsigned n =
      util_dynarray_num_elements(&gt->intervals, struct hk_gputime_interval);
   struct hk_gputime_interval *iv = util_dynarray_begin(&gt->intervals);

   uint64_t sum[HK_GPUTIME_KINDS] = {0};
   uint32_t cnt[HK_GPUTIME_KINDS] = {0};

   for (unsigned i = 0; i < n; ++i) {
      sum[iv[i].kind] += iv[i].end - iv[i].start;
      cnt[iv[i].kind]++;
   }

   /* The union of every interval, regardless of kind. Vertex, fragment and
    * compute overlap by design, so this -- not the individual sums -- is what
    * says whether the GPU was actually saturated.
    */
   qsort(iv, n, sizeof(*iv), cmp_interval);

   /* Walk the merged intervals, and while doing so measure the GAPS between
    * them -- the stretches where the GPU had nothing running at all.
    *
    * "GPU busy 78.6%" says a fifth of the frame is idle but not why, and the
    * two candidate answers want opposite fixes. Many small gaps means the
    * pipeline is being drained repeatedly -- control stream boundaries, of
    * which this game has ~137 per frame -- and the fix is to stop ending
    * streams. A few large gaps means the CPU is not feeding the GPU, and the
    * fix is somewhere in submission or in the game's own pacing.
    *
    * The histogram is coarse on purpose: the distinction above only needs
    * orders of magnitude.
    */
   uint64_t busy = 0, idle = 0, idle_max = 0;
   uint32_t gaps = 0;
   uint32_t gap_hist[4] = {0}; /* <10us, <100us, <1ms, >=1ms */
   uint64_t gap_time[4] = {0};
   uint64_t prev_end = 0;
   bool have_prev = false;

   for (unsigned i = 0; i < n;) {
      uint64_t s = iv[i].start, e = iv[i].end;
      unsigned j = i + 1;
      for (; j < n && iv[j].start <= e; ++j)
         e = MAX2(e, iv[j].end);

      if (have_prev && s > prev_end) {
         uint64_t g = s - prev_end;
         double us = (g / hz) * 1.0e6;
         unsigned b = us < 10.0 ? 0 : us < 100.0 ? 1 : us < 1000.0 ? 2 : 3;
         gap_hist[b]++;
         gap_time[b] += g;
         idle += g;
         idle_max = MAX2(idle_max, g);
         gaps++;
      }
      prev_end = e;
      have_prev = true;

      busy += e - s;
      i = j;
   }

   static const char *names[HK_GPUTIME_KINDS] = {"vtx ", "frag", "comp"};

   fprintf(stderr, "[hk gputime] %.2f s wall   %llu frames  %.1f fps\n",
           wall_ms / 1000.0, (unsigned long long)gt->presents,
           wall_ms > 0 ? gt->presents * 1000.0 / wall_ms : 0.0);
   for (unsigned k = 0; k < HK_GPUTIME_KINDS; ++k) {
      double ms = (sum[k] / hz) * 1000.0;
      fprintf(stderr,
              "[hk gputime]   %s  %8.1f ms  %5.1f%% of wall  %6u cmds  "
              "%7.1f us each\n",
              names[k], ms, wall_ms > 0 ? 100.0 * ms / wall_ms : 0.0, cnt[k],
              cnt[k] ? (ms * 1000.0) / cnt[k] : 0.0);
   }

   /* Per-COMMAND cost says little on its own: a command is a whole control
    * stream and may hold one dispatch or a hundred. Per-DISPATCH cost is what
    * distinguishes the driver issuing many small internal dispatches from the
    * application issuing a few expensive ones.
    */
   if (gt->dispatches || gt->draws) {
      double comp_ms = (sum[HK_GPUTIME_COMP] / hz) * 1000.0;
      fprintf(stderr,
              "[hk gputime]   dispatch origin: app %llu, meta %llu, "
              "gs/prerast %llu, tess %llu, precomp %llu  (recorded %llu)\n",
              (unsigned long long)gt->disp_kind[HK_DISP_APP],
              (unsigned long long)gt->disp_kind[HK_DISP_META],
              (unsigned long long)gt->disp_kind[HK_DISP_GS],
              (unsigned long long)gt->disp_kind[HK_DISP_TESS],
              (unsigned long long)gt->disp_kind[HK_DISP_PRECOMP],
              (unsigned long long)gt->disp_recorded);
      fprintf(stderr,
              "[hk gputime]   CDM commands built %llu, timestamped %llu, "
              "intervals harvested %u, isolate splits %llu\n",
              (unsigned long long)gt->cdm_commands,
              (unsigned long long)gt->cdm_timestamped, cnt[HK_GPUTIME_COMP],
              (unsigned long long)gt->isolate_splits);
      fprintf(stderr,
              "[hk gputime]   %llu dispatches (%.1f/cmd, %.1f us each)  "
              "%llu draws\n",
              (unsigned long long)gt->dispatches,
              cnt[HK_GPUTIME_COMP] ? (double)gt->dispatches / cnt[HK_GPUTIME_COMP] : 0.0,
              gt->dispatches ? (comp_ms * 1000.0) / gt->dispatches : 0.0,
              (unsigned long long)gt->draws);
      fprintf(stderr,
              "[hk gputime]   pipeline barriers %llu (%.1f/frame): "
              "%llu compute-only (%.1f/frame ending real work), "
              "%llu compute-only but gfx open, "
              "%llu need the hammer\n",
              (unsigned long long)gt->barriers,
              gt->presents ? (double)gt->barriers / gt->presents : 0.0,
              (unsigned long long)gt->barriers_compute,
              gt->presents ? (double)gt->barriers_avoidable / gt->presents
                           : 0.0,
              (unsigned long long)gt->barriers_gfx_open,
              (unsigned long long)(gt->barriers - gt->barriers_compute -
                                   gt->barriers_gfx_open));
   }

   double busy_ms = (busy / hz) * 1000.0;
   fprintf(stderr,
           "[hk gputime]   GPU busy (union) %.1f ms = %.1f%% of wall%s\n",
           busy_ms, wall_ms > 0 ? 100.0 * busy_ms / wall_ms : 0.0,
           gt->skipped ? "" : "");
   if (gaps) {
      static const char *gap_names[4] = {"<10us", "<100us", "<1ms", ">=1ms"};
      fprintf(stderr,
              "[hk gputime]   GPU idle %.1f ms in %u gaps (%.1f/frame, "
              "largest %.0f us)\n",
              (idle / hz) * 1000.0, gaps,
              gt->presents ? (double)gaps / gt->presents : 0.0,
              (idle_max / hz) * 1.0e6);
      fprintf(stderr, "[hk gputime]     ");
      for (unsigned b = 0; b < 4; ++b) {
         fprintf(stderr, "%s %u (%.1f ms)%s", gap_names[b], gap_hist[b],
                 (gap_time[b] / hz) * 1000.0, b == 3 ? "\n" : "   ");
      }
   }

   if (gt->skipped) {
      fprintf(stderr,
              "[hk gputime]   %llu command(s) not profiled "
              "(app claimed the timestamp slot)\n",
              (unsigned long long)gt->skipped);
   }

   hk_gputime_report_shaders(gt, gt->presents, hz);

   util_dynarray_clear(&gt->intervals);
   gt->skipped = 0;
   gt->disp_recorded = 0;
   gt->cdm_commands = 0;
   gt->cdm_timestamped = 0;
   gt->isolate_splits = 0;
   gt->presents = 0;
   gt->dispatches = 0;
   gt->draws = 0;
   gt->barriers = 0;
   gt->barriers_compute = 0;
   gt->barriers_gfx_open = 0;
   gt->barriers_avoidable = 0;
   memset(gt->disp_kind, 0, sizeof(gt->disp_kind));
   gt->report_start_ns = now_ns;
}

void
hk_gputime_tick(struct hk_device *dev)
{
   struct hk_gputime *gt = &dev->gputime;
   if (!gt->enabled)
      return;

   uint64_t now = os_time_get_nano();

   /* Scan four times a second: often enough that the ring cannot wrap, rarely
    * enough that walking it costs nothing worth measuring.
    */
   if (now - gt->last_scan_ns < 250000000ull)
      return;

   simple_mtx_lock(&gt->lock);
   gt->last_scan_ns = now;
   hk_gputime_scan(gt);

   if (now - gt->report_start_ns >= gt->report_period_ns)
      hk_gputime_report(dev, now);

   simple_mtx_unlock(&gt->lock);
}
