/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_device.h"
#include "util/rwlock.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "agx_bg_eot.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "decode.h"
#include "vk_cmd_queue.h"
#include "vk_dispatch_table.h"

#include "hk_private.h"

#include "hk_descriptor_table.h"
#include "hk_gputime.h"
#include "hk_queue.h"
#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct hk_physical_device;
struct vk_pipeline_cache;

typedef void (*hk_internal_builder_t)(struct nir_builder *b, const void *key);

struct hk_internal_key {
   hk_internal_builder_t builder;
   size_t key_size;
   uint8_t key[];
};

struct hk_internal_shaders {
   simple_mtx_t lock;
   struct hash_table *ht;
};

struct hk_rc_sampler {
   struct agx_sampler_packed key;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct hk_sampler_heap {
   simple_mtx_t lock;

   struct hk_descriptor_table table;

   /* Map of agx_sampler_packed to hk_rc_sampler */
   struct hash_table *ht;
};

struct hk_device {
   struct vk_device vk;
   struct agx_device dev;
   struct agxdecode_ctx *decode_ctx;

   struct hk_descriptor_table occlusion_queries;
   struct hk_sampler_heap samplers;

   struct vk_meta_device meta;
   struct agx_bg_eot_cache bg_eot;

   struct {
      struct agx_bo *bo;
      uint64_t heap;
   } rodata;

   struct hk_internal_shaders prolog_epilog;
   struct hk_internal_shaders kernels;
   struct hk_api_shader *null_fs;

   /* Indirected for common secondary emulation */
   struct vk_device_dispatch_table cmd_dispatch;

   /* Heap used for GPU-side memory allocation for geometry/tessellation.
    *
    * Control streams accessing the heap must be serialized. This is not
    * expected to be a legitimate problem. If it is, we can rework later.
    */
   struct agx_bo *heap;
   util_once_flag heap_init_once;

   struct {
      struct agx_scratch vs, fs, cs;
      simple_mtx_t lock;
   } scratch;

   uint32_t perftest;

   /* Firmware GPU timing, HK_GPUTIME=<seconds>. Off unless asked for. */
   struct hk_gputime gputime;

   /* AGX_CDM_BARRIER_MASK, resolved once. See hk_cdm_barrier_masked(). */
   uint32_t weak_barrier_mask;
   bool weak_barrier_mask_valid;
   uint32_t overlap_max;
   bool overlap_max_valid;

   struct {
      struct u_rwlock lock;
      struct util_dynarray list;
      struct util_dynarray counts;
   } external_bos;
};

VK_DEFINE_HANDLE_CASTS(hk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

enum hk_perftest {
   HK_PERF_NOTESS = BITFIELD_BIT(0),
   HK_PERF_NOBORDER = BITFIELD_BIT(1),
   HK_PERF_NOBARRIER = BITFIELD_BIT(2),
   HK_PERF_BATCH = BITFIELD_BIT(3),
   HK_PERF_NOROBUST = BITFIELD_BIT(4),
   HK_PERF_FORCEBARRIER = BITFIELD_BIT(5),
   HK_PERF_NOFLUSH = BITFIELD_BIT(6),

   /* Let independent vkCmdDispatch calls overlap.
    *
    * dispatch() issues every application dispatch with AGX_BARRIER_ALL, which
    * emits a full conservative CDM cache flush after it and so serialises
    * consecutive dispatches. Vulkan does not require that: dispatches not
    * separated by a barrier or event may execute concurrently, and
    * hk_CmdPipelineBarrier2 already ends the control stream when the
    * application does ask for ordering.
    *
    * It matters because this GPU needs roughly 4096 threads in flight to hide
    * memory latency (measured: 428 ns for a dependent load in cache, 905 ns
    * from DRAM, and a 7.4x throughput gain going from 64 to 4096 threads).
    * Ghost of Tsushima's hot compute shader dispatches ~213 invocations at a
    * time, 412 times a frame, so serialising them leaves the machine at about a
    * fifth of its achievable throughput. Measured directly with identical work
    * and no barriers between it: one dispatch of 64 groups takes 2.66 ms, the
    * same work as 64 dispatches of 1 group takes 17.20 ms -- 6.5x.
    */
   HK_PERF_OVERLAP = BITFIELD_BIT(7),

   /* Opt out of the above, which is on by default. */
   HK_PERF_NOOVERLAP = BITFIELD_BIT(8),
   HK_PERF_CSBARRIER = BITFIELD_BIT(9),
};

#define HK_PERF(dev, flag) unlikely((dev)->perftest &HK_PERF_##flag)

static inline struct hk_physical_device *
hk_device_physical(struct hk_device *dev)
{
   return (struct hk_physical_device *)dev->vk.physical;
}

VkResult hk_device_init_meta(struct hk_device *dev);
void hk_device_finish_meta(struct hk_device *dev);

VkResult hk_sampler_heap_add(struct hk_device *dev,
                             struct agx_sampler_packed desc,
                             struct hk_rc_sampler **out);

void hk_sampler_heap_remove(struct hk_device *dev, struct hk_rc_sampler *rc);

static inline struct agx_scratch *
hk_device_scratch_locked(struct hk_device *dev, mesa_shader_stage stage)
{
   simple_mtx_assert_locked(&dev->scratch.lock);

   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      return &dev->scratch.fs;
   case MESA_SHADER_VERTEX:
      return &dev->scratch.vs;
   default:
      return &dev->scratch.cs;
   }
}

static inline void
hk_device_alloc_scratch(struct hk_device *dev, mesa_shader_stage stage,
                        unsigned size)
{
   simple_mtx_lock(&dev->scratch.lock);
   agx_scratch_alloc(hk_device_scratch_locked(dev, stage), size, 0);
   simple_mtx_unlock(&dev->scratch.lock);
}
