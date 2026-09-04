/*
 * Copyright 2026 Claude / got-bringup
 * SPDX-License-Identifier: MIT
 */

#include "hk_gputime.h"

#include <stdio.h>
#include <stdlib.h>

#include "agx_bo.h"
#include "agx_device.h"
#include "hk_device.h"

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
   gt->report_period_ns = (uint64_t)(period_s * 1000000000.0);
   gt->last_scan_ns = gt->report_start_ns = os_time_get_nano();
   simple_mtx_init(&gt->lock, mtx_plain);
   util_dynarray_init(&gt->intervals, NULL);
   gt->enabled = true;

   fprintf(stderr,
           "[hk gputime] firmware GPU timing active, reporting every %.2f s "
           "(timebase %llu Hz)\n",
           period_s,
           (unsigned long long)dev->dev.params.command_timestamp_frequency_hz);
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
   agx_bo_unreference(&dev->dev, gt->bo);
   gt->bo = NULL;
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
        enum hk_gputime_kind kind)
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
}

/* lock held */
static void
hk_gputime_scan(struct hk_gputime *gt)
{
   for (uint32_t b = 0; b < gt->nr_blocks; ++b) {
      uint64_t *blk = gt->blocks + (size_t)b * HK_GPUTIME_SLOTS_PER_BLOCK;

      harvest(gt, blk, HK_GPUTIME_SLOT_VTX_START, HK_GPUTIME_SLOT_VTX_END,
              HK_GPUTIME_VTX);
      harvest(gt, blk, HK_GPUTIME_SLOT_FRAG_START, HK_GPUTIME_SLOT_FRAG_END,
              HK_GPUTIME_FRAG);
      harvest(gt, blk, HK_GPUTIME_SLOT_COMP_START, HK_GPUTIME_SLOT_COMP_END,
              HK_GPUTIME_COMP);
   }
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

   uint64_t busy = 0;
   for (unsigned i = 0; i < n;) {
      uint64_t s = iv[i].start, e = iv[i].end;
      unsigned j = i + 1;
      for (; j < n && iv[j].start <= e; ++j)
         e = MAX2(e, iv[j].end);
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

   double busy_ms = (busy / hz) * 1000.0;
   fprintf(stderr,
           "[hk gputime]   GPU busy (union) %.1f ms = %.1f%% of wall%s\n",
           busy_ms, wall_ms > 0 ? 100.0 * busy_ms / wall_ms : 0.0,
           gt->skipped ? "" : "");
   if (gt->skipped) {
      fprintf(stderr,
              "[hk gputime]   %llu command(s) not profiled "
              "(app claimed the timestamp slot)\n",
              (unsigned long long)gt->skipped);
   }

   util_dynarray_clear(&gt->intervals);
   gt->skipped = 0;
   gt->presents = 0;
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
