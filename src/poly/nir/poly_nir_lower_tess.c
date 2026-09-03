/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "poly/cl/libpoly.h"
#include "poly/geometry.h"
#include "poly/tessellator.h"
#include "poly/nir/poly_nir.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

struct lower_tcs_state {
   /* Whether barriers targeting only shader_out may be dropped entirely */
   bool can_ignore_shader_out_barriers;

   /* Number of patches packed into a single TCS workgroup. Must match the
    * workgroup size the driver dispatches with, which is
    * patches_per_workgroup * tcs_vertices_out.
    */
   unsigned patches_per_workgroup;
};

/*
 * Index of the patch handled by this thread within its workgroup. With K
 * patches packed per workgroup, thread t handles patch t / output_patch_size.
 *
 * There is deliberately no bounds check here: the grid is dispatched in threads
 * (patches * output_patch_size), so the hardware launches a partial final
 * threadgroup containing exactly the leftover patches. Every launched thread
 * therefore belongs to a real patch, whether or not the patch count is a
 * multiple of K and whether or not the patch size divides the subgroup size.
 */
static nir_def *
tcs_patch_in_workgroup(nir_builder *b, const struct lower_tcs_state *state)
{
   if (state->patches_per_workgroup == 1)
      return nir_imm_int(b, 0);

   return nir_udiv_imm(b, nir_channel(b, nir_load_local_invocation_id(b), 0),
                       b->shader->info.tess.tcs_vertices_out);
}

static nir_def *
tcs_unrolled_id(nir_builder *b, const struct lower_tcs_state *state)
{
   return poly_tcs_unrolled_id(
      b, nir_load_tess_param_buffer_poly(b), nir_load_workgroup_id(b),
      tcs_patch_in_workgroup(b, state),
      nir_imm_int(b, state->patches_per_workgroup));
}

uint64_t
poly_tcs_per_vertex_outputs(const nir_shader *nir)
{
   return nir->info.outputs_written &
          ~(VARYING_BIT_TESS_LEVEL_INNER | VARYING_BIT_TESS_LEVEL_OUTER |
            VARYING_BIT_BOUNDING_BOX0 | VARYING_BIT_BOUNDING_BOX1);
}

unsigned
poly_tcs_output_stride(const nir_shader *nir)
{
   return poly_tcs_out_stride(util_last_bit(nir->info.patch_outputs_written),
                              nir->info.tess.tcs_vertices_out,
                              poly_tcs_per_vertex_outputs(nir));
}

static nir_def *
tcs_out_addr(nir_builder *b, nir_intrinsic_instr *intr, nir_def *vertex_id,
             const struct lower_tcs_state *state)
{
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);

   nir_def *offset = nir_get_io_offset_src(intr)->ssa;
   nir_def *addr = poly_tcs_out_address(
      b, nir_load_tess_param_buffer_poly(b), tcs_unrolled_id(b, state),
      vertex_id,
      nir_iadd_imm(b, offset, sem.location),
      nir_imm_int(b, util_last_bit(b->shader->info.patch_outputs_written)),
      nir_imm_int(b, b->shader->info.tess.tcs_vertices_out),
      nir_imm_int64(b, poly_tcs_per_vertex_outputs(b->shader)));

   addr = nir_iadd_imm(b, addr, nir_intrinsic_component(intr) * 4);

   return addr;
}

static nir_def *
lower_tes_load(nir_builder *b, nir_intrinsic_instr *intr)
{
   gl_varying_slot location = nir_intrinsic_io_semantics(intr).location;
   nir_src *offset_src = nir_get_io_offset_src(intr);

   nir_def *vertex = nir_imm_int(b, 0);
   nir_def *offset = offset_src ? offset_src->ssa : nir_imm_int(b, 0);

   if (intr->intrinsic == nir_intrinsic_load_per_vertex_input)
      vertex = intr->src[0].ssa;

   nir_def *addr = poly_tes_in_address(b, nir_load_tess_param_buffer_poly(b),
                                       nir_load_vertex_id(b), vertex,
                                       nir_iadd_imm(b, offset, location));

   if (nir_intrinsic_has_component(intr))
      addr = nir_iadd_imm(b, addr, nir_intrinsic_component(intr) * 4);

   return nir_load_global_constant(b, intr->def.num_components,
                                   intr->def.bit_size, addr, .align_mul = 4);
}

static nir_def *
tcs_load_input(nir_builder *b, nir_intrinsic_instr *intr,
               const struct lower_tcs_state *state)
{
   nir_def *base = nir_imul(
      b, tcs_unrolled_id(b, state),
      poly_tcs_patch_vertices_in(b, nir_load_tess_param_buffer_poly(b)));
   nir_def *vertex = nir_iadd(b, base, intr->src[0].ssa);

   return poly_load_per_vertex_input(b, intr, vertex);
}

static nir_def *
lower_tcs_impl(nir_builder *b, nir_intrinsic_instr *intr,
               const struct lower_tcs_state *state)
{
   bool can_ignore_shader_out_barriers = state->can_ignore_shader_out_barriers;

   switch (intr->intrinsic) {
   case nir_intrinsic_barrier: {
      /* We lowered the TCS outputs from shader_out to global memory,
       * preserve barriers if they don't target shader outputs or modify their
       * execution/memory scopes and modes accordingly to match subgroups since
       * a patch fits in a subgroup.
       *
       * When several patches are packed into one workgroup they all still fit
       * in a single subgroup, so a subgroup barrier remains a superset of the
       * per-patch barrier the shader asked for. */
      nir_variable_mode modes = nir_intrinsic_memory_modes(intr);

      /* Skip barriers that don't handle shader outputs */
      if (!(modes & nir_var_shader_out))
         return NULL;

      modes &= ~nir_var_shader_out;

      /* A barrier that only targeted shader outputs can be dropped entirely by
       * drivers that don't need it. */
      if (can_ignore_shader_out_barriers && !modes)
         return NIR_LOWER_INSTR_PROGRESS_REPLACE;

      /* Keep the original scopes while the barrier still targets other modes;
       * otherwise a subgroup scope suffices since a patch fits in a subgroup.
       * Drivers that can't ignore the shader-output barrier must preserve it
       * against the lowered global memory. */
      mesa_scope execution_scope =
         modes ? nir_intrinsic_execution_scope(intr) : SCOPE_SUBGROUP;
      mesa_scope memory_scope =
         modes ? nir_intrinsic_memory_scope(intr) : SCOPE_SUBGROUP;
      if (!can_ignore_shader_out_barriers)
         modes |= nir_var_mem_global;

      nir_barrier(b, .execution_scope = execution_scope,
                  .memory_scope = memory_scope,
                  .memory_semantics = nir_intrinsic_memory_semantics(intr),
                  .memory_modes = modes);
      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   }

   case nir_intrinsic_load_primitive_id:
      return poly_tcs_patch_id(b, nir_load_workgroup_id(b),
                               tcs_patch_in_workgroup(b, state),
                               nir_imm_int(b, state->patches_per_workgroup));

   case nir_intrinsic_load_instance_id:
      return nir_channel(b, nir_load_workgroup_id(b), 1);

   case nir_intrinsic_load_invocation_id: {
      unsigned patch_size = b->shader->info.tess.tcs_vertices_out;
      if (patch_size == 1)
         return nir_imm_int(b, 0);

      nir_def *local = nir_channel(b, nir_load_local_invocation_id(b), 0);
      if (state->patches_per_workgroup == 1)
         return local;

      /* local % patch_size, reusing the division from tcs_patch_in_workgroup
       * rather than emitting a second magic-number divide. */
      return nir_isub(
         b, local,
         nir_imul_imm(b, tcs_patch_in_workgroup(b, state), patch_size));
   }

   case nir_intrinsic_load_per_vertex_input:
      return tcs_load_input(b, intr, state);

   case nir_intrinsic_load_patch_vertices_in:
      return poly_tcs_patch_vertices_in(b, nir_load_tess_param_buffer_poly(b));

   case nir_intrinsic_load_tess_level_outer_default:
      return poly_tess_level_outer_default(b,
                                           nir_load_tess_param_buffer_poly(b));

   case nir_intrinsic_load_tess_level_inner_default:
      return poly_tess_level_inner_default(b,
                                           nir_load_tess_param_buffer_poly(b));

   case nir_intrinsic_load_output: {
      nir_def *addr = tcs_out_addr(b, intr, nir_undef(b, 1, 32), state);
      return nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                             addr, .align_mul = 4);
   }

   case nir_intrinsic_load_per_vertex_output: {
      nir_def *addr = tcs_out_addr(b, intr, intr->src[0].ssa, state);
      return nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                             addr, .align_mul = 4);
   }

   case nir_intrinsic_store_output: {
      /* Only vec2, make sure we can't overwrite */
      assert(intr->src[0].ssa->num_components <= 2 ||
             nir_intrinsic_io_semantics(intr).location !=
                VARYING_SLOT_TESS_LEVEL_INNER);

      nir_store_global(b, intr->src[0].ssa,
                       tcs_out_addr(b, intr, nir_undef(b, 1, 32), state),
                       .write_mask = nir_intrinsic_write_mask(intr));
      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   }

   case nir_intrinsic_store_per_vertex_output: {
      nir_store_global(b, intr->src[0].ssa,
                       tcs_out_addr(b, intr, intr->src[1].ssa, state),
                       .write_mask = nir_intrinsic_write_mask(intr));
      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   }

   default:
      return NULL;
   }
}

static bool
lower_tcs(nir_builder *b, nir_intrinsic_instr *intr, void *data_)
{
   const struct lower_tcs_state *data = data_;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *repl = lower_tcs_impl(b, intr, data);
   if (!repl)
      return false;

   if (repl != NIR_LOWER_INSTR_PROGRESS_REPLACE)
      nir_def_rewrite_uses(&intr->def, repl);

   nir_instr_remove(&intr->instr);
   return true;
}

bool
poly_nir_lower_tcs(nir_shader *tcs, bool can_ignore_shader_out_barriers,
                   unsigned patches_per_workgroup)
{
   assert(patches_per_workgroup >= 1);
   assert(patches_per_workgroup == 1 ||
          patches_per_workgroup * tcs->info.tess.tcs_vertices_out <=
             POLY_TCS_SUBGROUP_SIZE);

   struct lower_tcs_state state = {
      .can_ignore_shader_out_barriers = can_ignore_shader_out_barriers,
      .patches_per_workgroup = patches_per_workgroup,
   };

   return nir_shader_intrinsics_pass(tcs, lower_tcs, nir_metadata_control_flow,
                                     &state);
}

static nir_def *
lower_tes_impl(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_tess_coord_xy:
      return poly_load_tess_coord(b, nir_load_tess_param_buffer_poly(b),
                                  nir_load_vertex_id(b));

   case nir_intrinsic_load_primitive_id:
      return poly_tes_patch_id(b, nir_load_tess_param_buffer_poly(b),
                               nir_load_vertex_id(b));

   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_tess_level_inner:
   case nir_intrinsic_load_tess_level_outer:
      return lower_tes_load(b, intr);

   case nir_intrinsic_load_patch_vertices_in:
      return poly_tes_patch_vertices_in(b, nir_load_tess_param_buffer_poly(b));

   default:
      return NULL;
   }
}

static bool
lower_tes(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *repl = lower_tes_impl(b, intr, data);

   if (repl) {
      nir_def_replace(&intr->def, repl);
      return true;
   } else {
      return false;
   }
}

static bool
lower_tes_indexing(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vertex_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *p = nir_load_tess_param_buffer_poly(b);
   nir_def *id = nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
   nir_def_replace(&intr->def, poly_load_tes_index(b, p, id));
   return true;
}

bool
poly_nir_lower_tes(nir_shader *tes, bool to_hw_vs)
{
   nir_lower_tess_coord_z(
      tes, tes->info.tess._primitive_mode == TESS_PRIMITIVE_TRIANGLES);

   nir_shader_intrinsics_pass(tes, lower_tes, nir_metadata_control_flow, NULL);

   /* Points mode renders as points, make sure we write point size for the HW */
   if (tes->info.tess.point_mode && to_hw_vs) {
      nir_lower_default_point_size(tes);
   }

   if (to_hw_vs) {
      /* We lower to a HW VS, so update the shader info so the compiler does the
       * right thing.
       */
      tes->info.stage = MESA_SHADER_VERTEX;
      memset(&tes->info.vs, 0, sizeof(tes->info.vs));
      tes->info.vs.tes_poly = true;
   } else {
      /* If we're running as a compute shader, we need to load from the index
       * buffer manually. Fortunately, this doesn't require a shader key:
       * tess-as-compute always use U32 index buffers.
       */
      nir_shader_intrinsics_pass(tes, lower_tes_indexing,
                                 nir_metadata_control_flow, NULL);
   }

   nir_lower_idiv(tes, &(nir_lower_idiv_options){.allow_fp16 = true});
   return nir_progress(true, nir_shader_get_entrypoint(tes), nir_metadata_none);
}
