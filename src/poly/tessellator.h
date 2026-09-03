/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/libcl/libcl.h"

enum poly_tess_partitioning {
   POLY_TESS_PARTITIONING_FRACTIONAL_ODD,
   POLY_TESS_PARTITIONING_FRACTIONAL_EVEN,
   POLY_TESS_PARTITIONING_INTEGER,
};

enum poly_tess_mode {
   /* Do not actually tessellate, just write the index counts */
   POLY_TESS_MODE_COUNT,

   /* Tessellate using the count buffers to allocate indices */
   POLY_TESS_MODE_WITH_COUNTS,
};

/* Workgroup size of the two parallel tessellation prefix sum kernels. Fixed at
 * 1024 because poly_work_group_scan_inclusive_add assumes a full 32 subgroups.
 */
#define POLY_TESS_SCAN_WG 1024

/* Number of per-patch counts prefix summed by a single workgroup in the
 * parallel (two pass) tessellation prefix sum. Must be <= POLY_TESS_SCAN_WG:
 * one element per invocation, with the surplus invocations contributing zero.
 * Lowering this is useful to force multi-block behaviour when testing.
 */
#define POLY_TESS_SCAN_BLOCK 1024

/* Only worth scanning in parallel past this many patches. Below it the single
 * workgroup kernel finishes in a handful of iterations and the extra dispatch
 * (plus its cache flush) costs more than it saves. Measured on an M1 Max the
 * crossover sits between 4 and 8 blocks.
 */
#define POLY_TESS_SCAN_PARALLEL_MIN (8 * POLY_TESS_SCAN_BLOCK)

/* Number of workgroups (blocks) the parallel prefix sum uses for a given number
 * of patches. Always at least 1: with zero patches we still need a single
 * workgroup to allocate an empty index buffer and write the draw descriptor.
 */
static inline uint32_t
poly_tess_scan_blocks(uint32_t nr_patches)
{
   uint32_t blocks =
      (nr_patches + (POLY_TESS_SCAN_BLOCK - 1)) / POLY_TESS_SCAN_BLOCK;

   return blocks == 0 ? 1 : blocks;
}

/* Scratch words appended after the counts array, holding one partial sum per
 * block. Sized poly_tess_scan_blocks(nr_patches) words.
 */
static inline uint32_t
poly_tess_counts_size_B(uint32_t nr_patches)
{
   return (nr_patches + poly_tess_scan_blocks(nr_patches)) * 4;
}

struct poly_tess_point {
   uint32_t u;
   uint32_t v;
};
static_assert(sizeof(struct poly_tess_point) == 8,
              "struct poly_tess_point must be 8 bytes");

struct poly_tess_params {
   /* Heap to allocate tessellator outputs in */
   DEVICE(struct poly_heap) heap;

   /* Patch coordinate buffer, indexed as:
    *
    *    coord_allocs[patch_ID] + vertex_in_patch
    */
   DEVICE(struct poly_tess_point) patch_coord_buffer;

   /* Per-patch index within the heap for the tess coords, written by the
    * tessellator based on the allocated memory.
    */
   DEVICE(uint32_t) coord_allocs;

   /* Space for output draws from the tessellator. API draw calls. */
   DEVICE(uint32_t) out_draws;

   /* Tessellation control shader output buffer. */
   DEVICE(float) tcs_buffer;

   /* Count buffer. # of indices per patch written here, then prefix summed. */
   DEVICE(uint32_t) counts;

   /* Allocated index buffer for all patches, if we're prefix summing counts */
   DEVICE(uint32_t) index_buffer;

   /* Address of the tess eval invocation counter for implementing pipeline
    * statistics, if active. Zero if inactive. Incremented by tessellator.
    */
   DEVICE(uint32_t) statistic;

   /* Bitfield of TCS per-vertex outputs */
   uint64_t tcs_per_vertex_outputs;

   /* Default tess levels used in OpenGL when there is no TCS in the pipeline.
    * Unused in Vulkan and OpenGL ES.
    */
   float tess_level_outer_default[4];
   float tess_level_inner_default[2];

   /* Number of vertices in the input patch */
   uint32_t input_patch_size;

   /* Number of vertices in the TCS output patch */
   uint32_t output_patch_size;

   /* Number of patch constants written by TCS */
   uint32_t tcs_patch_constants;

   /* Number of input patches per instance of the VS/TCS */
   uint32_t patches_per_instance;

   /* Stride between tessellation facotrs in the TCS output buffer. */
   uint32_t tcs_stride_el;

   /* Number of patches being tessellated */
   uint32_t nr_patches;

   /* Partitioning and points mode. These affect per-patch setup code but not
    * the hot tessellation loop so we make them dynamic to reduce tessellator
    * variants.
    */
   enum poly_tess_partitioning partitioning;
   uint32_t points_mode;
   uint32_t isolines;

   /* When fed into a geometry shader, triangles should be counter-clockwise.
    * The tessellator always produces clockwise triangles, but we can swap
    * dynamically in the TES.
    */
   uint32_t ccw;
} PACKED;
static_assert(sizeof(struct poly_tess_params) == 34 * 4,
              "struct poly_tess_params must be 34 words");
