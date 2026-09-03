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

/* The tessellation control shader is dispatched as a compute shader with one
 * thread per output control point, so a workgroup would naturally be the output
 * patch size. AGX runs 32 threads per subgroup, and the overwhelmingly common
 * patch sizes are 3 and 4, so such a workgroup wastes almost the whole machine
 * (and burns a threadgroup slot per patch).
 *
 * Instead we pack as many *whole* patches as fit into a single subgroup, so the
 * packed workgroup is still at most one subgroup. That keeps the barriers
 * correct: poly_nir_lower_tcs either drops a TCS barrier outright (relying on
 * subgroup lockstep), demotes it to SCOPE_SUBGROUP, or leaves it at its
 * original workgroup scope. In every case the resulting synchronisation is a
 * superset of the per-patch barrier the shader asked for, since the patch is a
 * subset of the packed workgroup which is a subset of the subgroup.
 *
 * Packing more than a subgroup's worth would be wrong, since then the implicit
 * lockstep that replaces the dropped barriers no longer holds.
 */
#define POLY_TCS_SUBGROUP_SIZE (32)

static inline uint32_t
poly_tcs_patches_per_workgroup(uint32_t output_patch_size)
{
   if (output_patch_size == 0 || output_patch_size >= POLY_TCS_SUBGROUP_SIZE)
      return 1;

   return POLY_TCS_SUBGROUP_SIZE / output_patch_size;
}

static inline uint32_t
poly_tcs_workgroup_size(uint32_t output_patch_size)
{
   return output_patch_size * poly_tcs_patches_per_workgroup(output_patch_size);
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
