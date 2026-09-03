/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl_vk.h"
#include "poly/cl/restart.h"
#include "poly/geometry.h"
#include "poly/prim.h"
#include "poly/tessellator.h"
#include "util/macros.h"
#include "util/u_math.h"

KERNEL(1)
libagx_increment_ia(global uint32_t *ia_vertices,
                    global uint32_t *ia_primitives,
                    global uint32_t *vs_invocations, global uint32_t *c_prims,
                    global uint32_t *c_invs, constant uint32_t *draw,
                    enum mesa_prim prim, unsigned verts_per_patch)
{
   poly_increment_ia(ia_vertices, ia_primitives, vs_invocations, c_prims,
                     c_invs, draw, prim, verts_per_patch);
}

KERNEL(1024)
libagx_increment_ia_restart(global uint32_t *ia_vertices,
                            global uint32_t *ia_primitives,
                            global uint32_t *vs_invocations,
                            global uint32_t *c_prims, global uint32_t *c_invs,
                            constant uint32_t *draw, uint64_t index_buffer,
                            uint32_t index_buffer_range_el,
                            uint32_t restart_index, uint32_t index_size_B,
                            enum mesa_prim prim, unsigned verts_per_patch)
{
   uint tid = cl_global_id.x;
   unsigned count = draw[0];
   local uint scratch;

   uint start = draw[2];
   uint partial = 0;

   /* Count non-restart indices */
   for (uint i = tid; i < count; i += 1024) {
      uint index = poly_load_index(index_buffer, index_buffer_range_el,
                                   start + i, index_size_B);

      if (index != restart_index)
         partial++;
   }

   /* Accumulate the partials across the workgroup */
   scratch = 0;
   barrier(CLK_LOCAL_MEM_FENCE);
   atomic_add(&scratch, partial);
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Elect a single thread from the workgroup to increment the counters */
   if (tid == 0) {
      poly_increment_counters(ia_vertices, vs_invocations, NULL,
                              scratch * draw[1]);
   }

   /* TODO: We should vectorize this */
   if ((ia_primitives || c_prims || c_invs) && tid == 0) {
      uint accum = 0;
      int last_restart = -1;
      for (uint i = 0; i < count; ++i) {
         uint index = poly_load_index(index_buffer, index_buffer_range_el,
                                      start + i, index_size_B);

         if (index == restart_index) {
            accum += poly_decomposed_prims_for_vertices_with_tess(
               prim, i - last_restart - 1, verts_per_patch);
            last_restart = i;
         }
      }

      {
         accum += poly_decomposed_prims_for_vertices_with_tess(
            prim, count - last_restart - 1, verts_per_patch);
      }

      poly_increment_counters(ia_primitives, c_prims, c_invs, accum * draw[1]);
   }
}

KERNEL(1024)
libagx_unroll_restart(global struct poly_heap *heap, uint64_t index_buffer,
                      constant uint *in_draw, global uint32_t *out_draw,
                      uint32_t max_draws, uint32_t restart_index,
                      uint32_t index_buffer_size_el, uint32_t index_size_log2,
                      uint32_t flatshade_first, uint mode__11)
{
   uint32_t index_size_B = 1 << index_size_log2;
   enum mesa_prim mode = poly_uncompact_prim(mode__11);

   POLY_DECL_UNROLL_RESTART_SCRATCH(scratch, 1024);
   poly_unroll_restart(out_draw, heap, in_draw, index_buffer,
                       index_buffer_size_el, index_size_B, restart_index,
                       flatshade_first, mode, scratch);
}

KERNEL(1)
libagx_gs_setup_indirect(
   uint64_t index_buffer, constant uint *draw,
   global struct poly_vertex_params *vp /* output */,
   global struct poly_geometry_params *p /* output */,
   global struct poly_heap *heap,
   uint64_t vs_outputs /* Vertex (TES) output mask */,
   uint32_t index_size_B /* 0 if no index bffer */,
   uint32_t index_buffer_range_el,
   uint32_t prim /* Input primitive type, enum mesa_prim */,
   int is_prefix_summing, uint max_indices, enum poly_gs_shape shape)
{
   poly_gs_setup_indirect(index_buffer, draw, vp, p, heap,
                          vs_outputs, index_size_B, index_buffer_range_el, prim,
                          is_prefix_summing, max_indices, shape);
}

KERNEL(1024)
libagx_prefix_sum_geom(constant struct poly_geometry_params *p)
{
   local uint scratch[32];
   poly_prefix_sum(scratch, p->count_buffer, p->input_primitives,
                   p->count_buffer_stride / 4, cl_group_id.x, 1024);
}

/*
 * Once the total number of indices is known, allocate the index buffer and
 * write out the API indexed draw. Must be called from exactly one invocation of
 * the whole dispatch.
 */
static void
libagx_tess_finish_prefix_sum(global struct poly_tess_params *p, uint total,
                              global uint *c_prims, global uint *c_invs,
                              uint increment_stats)
{
   /* Allocate 4-byte indices */
   uint32_t elsize_B = sizeof(uint32_t);
   uint32_t size_B = total * elsize_B;
   uint alloc_B = poly_heap_alloc_offs(p->heap, size_B);
   p->index_buffer = (global uint32_t *)(((uintptr_t)p->heap->base) + alloc_B);

   /* ...and now we can generate the API indexed draw */
   global uint32_t *desc = p->out_draws;

   desc[0] = total;              /* count */
   desc[1] = 1;                  /* instance_count */
   desc[2] = alloc_B / elsize_B; /* start */
   desc[3] = 0;                  /* index_bias */
   desc[4] = 0;                  /* start_instance */

   /* If necessary, increment clipper statistics too. This is only used when
    * there's no geometry shader following us. See poly_nir_lower_gs.c for more
    * info on the emulation. We just need to calculate the # of primitives
    * tessellated.
    */
   if (increment_stats) {
      uint prims = p->points_mode ? total
                   : p->isolines  ? (total / 2)
                                  : (total / 3);

      poly_increment_counters(c_prims, c_invs, NULL, prims);
   }
}

KERNEL(1024)
libagx_prefix_sum_tess(global struct poly_tess_params *p, global uint *c_prims,
                       global uint *c_invs, uint increment_stats__2)
{
   local uint scratch[32];
   poly_prefix_sum(scratch, p->counts, p->nr_patches, 1 /* words */,
                   0 /* word */, 1024);

   /* After prefix summing, we know the total # of indices, so allocate the
    * index buffer now. Elect a thread for the allocation.
    */
   barrier(CLK_LOCAL_MEM_FENCE);
   if (cl_local_id.x != 0)
      return;

   /* The last element of an inclusive prefix sum is the total sum */
   uint total = p->nr_patches > 0 ? p->counts[p->nr_patches - 1] : 0;

   libagx_tess_finish_prefix_sum(p, total, c_prims, c_invs,
                                 increment_stats__2);
}

/*
 * Parallel version of libagx_prefix_sum_tess, spread across many workgroups
 * instead of looping in a single one. Two passes:
 *
 *  1. Reduce: each block of POLY_TESS_SCAN_BLOCK counts is summed to a single
 *     partial, stashed in the scratch words that follow the counts array.
 *
 *  2. Scan: each block sums the partials of the blocks before it to get its
 *     base, scans its own block, and writes base + scan.
 *
 * The result is the exact same ordered inclusive prefix sum the serial kernel
 * produced - poly_draw() depends on counts[patch - 1] being the ordered
 * exclusive prefix so that primitives are emitted in patch order.
 */
KERNEL(POLY_TESS_SCAN_WG)
libagx_prefix_sum_tess_reduce(global struct poly_tess_params *p)
{
   local uint scratch[32];
   uint tid = cl_local_id.x;
   uint nr_patches = p->nr_patches;
   uint i = (cl_group_id.x * POLY_TESS_SCAN_BLOCK) + tid;

   bool live = (tid < POLY_TESS_SCAN_BLOCK) && (i < nr_patches);
   uint sum = poly_work_group_reduce_add(live ? p->counts[i] : 0, scratch);

   if (tid == 0)
      p->counts[nr_patches + cl_group_id.x] = sum;
}

KERNEL(POLY_TESS_SCAN_WG)
libagx_prefix_sum_tess_scan(global struct poly_tess_params *p,
                            global uint *c_prims, global uint *c_invs,
                            uint increment_stats__2)
{
   local uint scratch[32];
   uint tid = cl_local_id.x;
   uint block = cl_group_id.x;
   uint nr_patches = p->nr_patches;
   uint nr_blocks = poly_tess_scan_blocks(nr_patches);
   global uint *partials = p->counts + nr_patches;

   /* Sum the block partials, both for the blocks strictly before us (our base)
    * and for all blocks (the grand total). nr_blocks is uniform so the loop
    * trip count is uniform, which the barriers inside the reduction require.
    */
   uint pre = 0, all = 0;
   for (uint off = 0; off < nr_blocks; off += POLY_TESS_SCAN_WG) {
      uint b = off + tid;
      uint value = (b < nr_blocks) ? partials[b] : 0;

      all += value;
      pre += (b < block) ? value : 0;
   }

   uint base = poly_work_group_reduce_add(pre, scratch);
   uint total = poly_work_group_reduce_add(all, scratch);

   /* Ordered inclusive scan of our own block, offset by the base */
   uint i = (block * POLY_TESS_SCAN_BLOCK) + tid;
   bool live = (tid < POLY_TESS_SCAN_BLOCK) && (i < nr_patches);
   uint scan = poly_work_group_scan_inclusive_add(live ? p->counts[i] : 0,
                                                  scratch)[0];

   if (live)
      p->counts[i] = base + scan;

   /* Elect a single invocation of the whole dispatch for the allocation */
   if (block != 0 || tid != 0)
      return;

   libagx_tess_finish_prefix_sum(p, total, c_prims, c_invs,
                                 increment_stats__2);
}
