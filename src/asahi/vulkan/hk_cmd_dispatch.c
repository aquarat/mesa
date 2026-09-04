/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "libagx/query.h"
#include "vulkan/vulkan_core.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "agx_tilebuffer.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "pool.h"

void
hk_cmd_buffer_begin_compute(struct hk_cmd_buffer *cmd,
                            const VkCommandBufferBeginInfo *pBeginInfo)
{
}

void
hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd)
{
   memset(&cmd->state.cs, 0, sizeof(cmd->state.cs));
}

void
hk_cmd_bind_compute_shader(struct hk_cmd_buffer *cmd,
                           struct hk_api_shader *shader)
{
   cmd->state.cs.shader = shader;
}

/*
 * A CDM barrier with only the bits we ask for.
 *
 * agx_cdm_barrier() (libagx_dgc.h:372) sets every bit of the 23-bit barrier
 * word after EVERY launch, under a comment saying the bits are not understood
 * and this is "to be safe". That is why consecutive dispatches never overlap,
 * and it costs dearly: this GPU needs ~4096 threads in flight to cover a
 * 428-905 ns dependent load, and the game supplies ~213 per dispatch.
 *
 * Omitting the barrier entirely does not work -- it emits no barrier BLOCK at
 * all, and the command stream appears to need one for sequencing even when no
 * cache maintenance is wanted. That hangs the GPU (observed as
 * "XIO: fatal IO error 110 (Connection timed out)" from the session's X server).
 *
 * So emit a real barrier block with a chosen subset of the cache bits.
 * AGX_CDM_BARRIER_MASK sweeps that subset so the minimal correct set can be
 * found by measurement rather than guessed. Bit 3 is USC cache invalidate, the
 * only one genxml names; 0-20 are the rest.
 */
static void
hk_cdm_barrier_masked(struct hk_device *dev, struct hk_cs *cs, uint32_t mask)
{
   assert(cs->type == HK_CS_CDM);
   assert(cs->current + AGX_CDM_BARRIER_LENGTH < cs->end &&
          "caller must ensure space");

   uint32_t *out = (uint32_t *)cs->current;
   *out = (mask & 0x001FFFFFu) | (3u /* CDM Block Type: Barrier */ << 29);
   cs->current = (uint8_t *)cs->current + AGX_CDM_BARRIER_LENGTH;
   cs->stats.flushes++;
}

static uint32_t
hk_overlap_max(struct hk_device *dev)
{
   if (unlikely(!dev->overlap_max_valid)) {
      const char *e = getenv("AGX_OVERLAP_MAX");
      dev->overlap_max = e ? (uint32_t)strtoul(e, NULL, 0) : 0u;
      dev->overlap_max_valid = true;
   }
   return dev->overlap_max;
}

/* Cached so the sweep costs one getenv per device, not one per dispatch. */
static uint32_t
hk_weak_barrier_mask(struct hk_device *dev)
{
   if (unlikely(!dev->weak_barrier_mask_valid)) {
      /* 0x1f: bits 0-4, which include bit 3 (USC cache inval) and bit 4 --
       * the descriptor/uniform state maintenance that must happen between ANY
       * two dispatches. Leaving it out hangs the GPU. It deliberately excludes
       * bit 7, the data-coherency bit, which is the one that serialises and is
       * only required between DEPENDENT dispatches -- and those are separated
       * by a barrier, which ends the control stream.
       *
       * Found by sweeping a single dependency pattern (got-bringup/
       * tests/coherence.c) rather than by reasoning about the bits, whose
       * meanings remain unknown.
       */
      const char *e = getenv("AGX_CDM_BARRIER_MASK");
      dev->weak_barrier_mask = e ? (uint32_t)strtoul(e, NULL, 0) : 0x1fu;
      dev->weak_barrier_mask_valid = true;
   }
   return dev->weak_barrier_mask;
}

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   assert(cs->type == HK_CS_CDM);
   assert(cs->current + AGX_CDM_BARRIER_LENGTH < cs->end &&
          "caller must ensure space");

   cs->current = agx_cdm_barrier(cs->current, dev->dev.chip);
   cs->stats.flushes++;
}

void
hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                            struct agx_cdm_launch_word_0_packed launch,
                            uint32_t usc, struct agx_grid grid,
                            struct agx_workgroup wg, enum agx_barrier barrier)
{
   hk_ensure_cs_has_space(cs->cmd, cs, 0x2000 /* TODO */);

   /* Settle any deferred flush before a dispatch that is NOT overlappable.
    * Skipping the flush between two application dispatches is legal -- Vulkan
    * orders them only through a barrier, which ends the stream. It is not legal
    * to let the driver's own helper kernels, or an indirect dispatch reading a
    * GPU-written grid, observe writes that were never flushed. A garbage
    * indirect grid is not a wrong pixel, it is a dispatch of billions of
    * workgroups, which is how the blanket version hung the GPU.
    */
   if (unlikely(cs->pending_flush) &&
       ((barrier & AGX_BARRIER_ALL) || agx_is_indirect(grid))) {
      hk_cdm_cache_flush(dev, cs);
      cs->pending_flush = false;
      cs->weak_run = 0;
      hk_ensure_cs_has_space(cs->cmd, cs, 0x2000 /* TODO */);
   }

   cs->stats.cmds++;

   cs->current =
      agx_cdm_launch(cs->current, dev->dev.chip, grid, wg, launch, usc);

   /* The AGX_PREGFX/AGX_POSTGFX bits only select the target control stream, so
    * mask them out: only AGX_BARRIER_ALL asks for cache maintenance. Callers
    * that know the next dispatch in this control stream is independent pass
    * AGX_BARRIER_NONE and we skip the (expensive, fully conservative) flush.
    *
    * HK_PERFTEST=forcebarrier restores the old unconditional behaviour so this
    * can be A/B tested without a rebuild.
    */
   /* HK_PERFTEST=noflush skips every CDM cache flush. This is DELIBERATELY
    * INCORRECT and will render wrong; its only purpose is to put a hard upper
    * bound on what the per-dispatch barrier tax costs. Measured on Ghost of
    * Tsushima: 966 dispatches and 971 flushes per frame at ~6 fps.
    */
   if (HK_PERF(dev, NOFLUSH))
      return;

   if ((barrier & AGX_BARRIER_ALL) || HK_PERF(dev, FORCEBARRIER))
      hk_cdm_cache_flush(dev, cs);
   else if (HK_PERF(dev, OVERLAP)) {
      /* Still emit a barrier BLOCK, just a weak one: the stream seems to need
       * the block for sequencing, and the cache bits are what serialise.
       *
       * AGX_OVERLAP_MAX bounds how many dispatches may go by on a weak barrier
       * before a full one is forced. It exists to bisect a failure: if the game
       * survives a depth of 8 but not 64, the trouble is how long writes stay
       * unflushed, not the idea of overlapping at all. 0 (the default) means
       * unbounded.
       */
      hk_cdm_barrier_masked(dev, cs, hk_weak_barrier_mask(dev));
      cs->pending_flush = true;
      cs->weak_run++;

      uint32_t cap = hk_overlap_max(dev);
      if (cap && cs->weak_run >= cap) {
         hk_cdm_cache_flush(dev, cs);
         cs->pending_flush = false;
         cs->weak_run = 0;
      }
   }
}

void
hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                     struct agx_shader_info *info, uint32_t usc,
                     struct agx_grid grid, struct agx_workgroup local_size,
                     enum agx_barrier barrier, enum hk_disp_kind kind)
{
   /* Every CDM launch in the driver reaches the chokepoint below through either
    * this function or hk_dispatch_precomp, and both are past their early
    * returns here, so the origin counts reconcile exactly with cs->stats.cmds.
    */
   if (unlikely(dev->gputime.enabled)) {
      bool indirect = agx_is_indirect(grid);
      uint64_t threads =
         indirect ? 0
                  : (uint64_t)grid.count[0] * grid.count[1] * grid.count[2];

      hk_gputime_note_shader(dev, kind, info, threads, indirect);

      /* Tag the stream so the firmware timestamp can be charged to a shader.
       * A stream holding work from two different shaders is marked MIXED and
       * simply not attributed, rather than attributed to whichever came first.
       */
      if (cs->gputime_shader == NULL)
         cs->gputime_shader = info;
      else if (cs->gputime_shader != info)
         cs->gputime_shader = HK_GPUTIME_MIXED;
   }

   struct agx_cdm_launch_word_0_packed launch;
   agx_pack(&launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = info->texture_state_count;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(info->sampler_state_count, false);
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }

   hk_dispatch_with_usc_launch(dev, cs, launch, usc, grid, local_size, barrier);

   /* HK_GPUTIME_ISOLATE: give every dispatch its own control stream so the
    * firmware times it individually. Only the plain compute stream is split --
    * pre_gfx/post_gfx are ordered against a graphics command and splitting them
    * would change what runs when.
    */
   /* NOTE: this ends the control stream and a fresh one is created for the next
    * dispatch -- verified, the hk_cs pointers differ -- but it does NOT produce
    * one submitted command per dispatch. Measured with 1216 dispatches: 13 CDM
    * commands built. Something downstream coalesces them, so per-dispatch
    * timing is not achieved by splitting here and the attribution stays at
    * ~101 dispatches per timed command. Left in place, and reported honestly by
    * the "CDM commands built / timestamped / intervals harvested" line, until
    * the coalescing is understood.
    */
   if (unlikely(dev->gputime.isolate) && cs->cmd &&
       cs == cs->cmd->current_cs.cs) {
      p_atomic_inc(&dev->gputime.isolate_splits);
      hk_cmd_buffer_end_compute(cs->cmd);
   }
}

static void
dispatch(struct hk_cmd_buffer *cmd, struct agx_grid grid)
{
   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   if (agx_is_shader_empty(&s->b))
      return;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   struct agx_workgroup local_size =
      agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                    s->b.info.workgroup_size[2]);

   uint64_t stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT);

   if (hk_stat_enabled(stat)) {
      perf_debug(cmd, "CS invocation statistic");
      uint64_t grid = cmd->state.cs.descriptors.root.cs.group_count_addr;

      libagx_increment_cs_invocations(cmd, agx_1d(1), AGX_BARRIER_ALL, grid,
                                      stat, agx_workgroup_threads(local_size));
   }

   if (!agx_is_indirect(grid)) {
      grid.count[0] *= local_size.x;
      grid.count[1] *= local_size.y;
      grid.count[2] *= local_size.z;
   }

   /* vk_meta implements image copies and blits by calling vkCmdDispatch, so
    * without this they would be indistinguishable from the application's own
    * compute -- and mislabelled as it.
    */
   /* Dispatches not separated by a barrier or event may run concurrently per
    * Vulkan, and hk_CmdPipelineBarrier2 ends the control stream when ordering
    * is actually requested -- so the per-dispatch flush is not what provides
    * that ordering, it only prevents overlap. See HK_PERF_OVERLAP.
    */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   enum agx_barrier barrier =
      HK_PERF(dev, OVERLAP) ? AGX_BARRIER_NONE : AGX_BARRIER_ALL;

   hk_dispatch_with_local_size(cmd, cs, s, grid, local_size, barrier,
                               cmd->in_meta ? HK_DISP_META : HK_DISP_APP);
   cs->stats.calls++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                   uint32_t baseGroupY, uint32_t baseGroupZ,
                   uint32_t groupCountX, uint32_t groupCountY,
                   uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = baseGroupX;
   desc->root.cs.base_group[1] = baseGroupY;
   desc->root.cs.base_group[2] = baseGroupZ;

   /* We don't want to key the shader to whether we're indirectly dispatching,
    * so treat everything as indirect.
    */
   VkDispatchIndirectCommand group_count = {
      .x = groupCountX,
      .y = groupCountY,
      .z = groupCountZ,
   };

   desc->root.cs.group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);

   dispatch(cmd, agx_3d(groupCountX, groupCountY, groupCountZ));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;

   uint64_t dispatch_addr = hk_buffer_address(buffer, offset, true);
   assert(dispatch_addr != 0);

   desc->root.cs.group_count_addr = dispatch_addr;
   dispatch(cmd, agx_grid_indirect(dispatch_addr));
}
