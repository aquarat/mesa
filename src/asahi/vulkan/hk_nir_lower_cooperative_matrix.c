/*
 * Copyright © 2023 Bas Nieuwenhuizen
 * Copyright © 2024 Collabora, Ltd.
 * Copyright © 2026 Google LLC.
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_cooperative_matrix for AGX. Derived from
 * panvk_nir_lower_cooperative_matrix.c, retargeted at the 8x8x8 SIMD-group
 * matrix multiply-accumulate (simd_matrix_fmadd16/32).
 *
 * The hardware tile is 8x8 over the 32-lane SIMD-group, two elements per lane,
 * in the "morton" layout used by Metal's simdgroup_matrix (as reverse
 * engineered for metal-flash-attention, Apple patent US11256518B2): the tile
 * is four 4x4 quadrants of eight lanes each; inside a quadrant lane pairs walk
 * the rows and each lane owns two horizontally adjacent elements:
 *
 *    q   = lane >> 3
 *    row = (q >> 1) * 4 + ((lane >> 1) & 3)
 *    col = (q & 1) * 4 + (lane & 1) * 2 + {0, 1}
 *
 * A, B, C and D all use this layout, so a cooperative matrix of any use is
 * lowered to a per-lane 2-vector and loads/stores are pure addressing.
 * Larger shapes are split into 8x8 tiles by
 * nir_lower_cooperative_matrix_flexible_dimensions first.
 */

#include "hk_shader.h"
#include "nir_builder.h"
#include "util/hash_table.h"
#include "util/u_debug.h"

#define HK_CMAT_HW_DIM 8
#define HK_CMAT_LANES  32

struct lower_cmat_ctx {
   struct hash_table *type_mapping;
};

/* Number of matrix elements each lane holds (the SPIR-V cmat length). */
static unsigned
get_cmat_length(struct glsl_cmat_description desc)
{
   assert(desc.rows == HK_CMAT_HW_DIM && desc.cols == HK_CMAT_HW_DIM);
   return (desc.rows * desc.cols) / HK_CMAT_LANES;
}

static const struct glsl_type *
remap_matrix_type(struct lower_cmat_ctx *ctx, const struct glsl_type *orig)
{
   struct hash_entry *entry = _mesa_hash_table_search(ctx->type_mapping, orig);
   if (entry)
      return entry->data;

   const struct glsl_type *new_type = orig;
   const struct glsl_type *leaf = glsl_without_array(orig);

   if (glsl_type_is_cmat(leaf)) {
      struct glsl_cmat_description desc = *glsl_get_cmat_description(leaf);
      new_type = glsl_type_wrap_in_arrays(
         glsl_vector_type(desc.element_type, get_cmat_length(desc)), orig);
   }

   _mesa_hash_table_insert(ctx->type_mapping, orig, (void *)new_type);
   return new_type;
}

static bool
remap_type_in_place(struct lower_cmat_ctx *ctx, const struct glsl_type **type)
{
   const struct glsl_type *new_type = remap_matrix_type(ctx, *type);
   if (new_type == *type)
      return false;

   *type = new_type;
   return true;
}

static struct glsl_cmat_description
cmat_src_desc(nir_src src)
{
   return *glsl_get_cmat_description(nir_src_as_deref(src)->type);
}

static nir_def *
load_cmat_src(nir_builder *b, nir_src src)
{
   nir_deref_instr *deref = nir_src_as_deref(src);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(deref->type);

   return nir_build_load_deref(b, get_cmat_length(desc),
                               glsl_base_type_bit_size(desc.element_type),
                               &deref->def, 0);
}

static void
store_cmat_src(nir_builder *b, nir_src dst, nir_def *val)
{
   nir_store_deref(b, nir_src_as_deref(dst), val, ~0);
}

/* (row, col of element 0) owned by this lane, see the layout comment above. */
static void
lane_origin(nir_builder *b, nir_def *lane, nir_def **row, nir_def **col)
{
   nir_def *q = nir_ushr_imm(b, lane, 3);
   *row = nir_iadd(b, nir_imul_imm(b, nir_iand_imm(b, nir_ushr_imm(b, q, 1), 1), 4),
                   nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3));
   *col = nir_iadd(b, nir_imul_imm(b, nir_iand_imm(b, q, 1), 4),
                   nir_imul_imm(b, nir_iand_imm(b, lane, 1), 2));
}

/*
 * HK_CMAT_VECLOAD selects how a lane moves its two elements of a tile when
 * they are contiguous:
 *
 *   0 (default) two scalar accesses, one per element
 *   1           one two-component access of the element type
 *   2           one 32-bit access, for a 4-byte-aligned 16-bit pair
 *
 * 1 and 2 are one instruction instead of two and measure smaller (fewer
 * instructions, fewer registers, same occupancy), but on G13G they are also
 * measurably *slower* in ggml's mul_mm once its inner loop stops re-issuing
 * the same tile load (-10 % on llama-bench pp512, -8 % on a ViT-H graph), so
 * the scalar form is what ships. See PHASE5 for the numbers.
 */
static unsigned
hk_cmat_vecload(void)
{
   static int mode = -1;
   if (mode < 0)
      mode = debug_get_num_option("HK_CMAT_VECLOAD", 0);

   return mode;
}

static bool
lower_cmat_load_store(nir_builder *b, nir_intrinsic_instr *intr,
                      const struct lower_cmat_ctx *ctx)
{
   const bool is_load = intr->intrinsic == nir_intrinsic_cmat_load;
   const struct glsl_cmat_description desc = cmat_src_desc(intr->src[!is_load]);
   const bool row_major =
      nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_ROW_MAJOR;
   const unsigned length = get_cmat_length(desc);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[is_load]);

   nir_def *lane = nir_load_subgroup_invocation(b);

   const unsigned ptr_vec = glsl_get_vector_elements(deref->type);
   const struct glsl_type *elem_type = glsl_scalar_type(desc.element_type);
   const unsigned elem_size_B = glsl_base_type_bit_size(desc.element_type) / 8;
   deref = nir_build_deref_cast(b, &deref->def, deref->modes, elem_type,
                                elem_size_B);

   const unsigned idx_bits = deref->def.bit_size;
   /* The buffer pointee may be a vector (multicomponent), so its stride is in
    * pointee units - scale it to the scalar elements we address.
    */
   nir_def *stride =
      nir_imul_imm(b, nir_u2uN(b, intr->src[2].ssa, idx_bits), ptr_vec);

   nir_def *row, *col0;
   lane_origin(b, lane, &row, &col0);

   nir_def *src = is_load ? NULL : load_cmat_src(b, intr->src[!is_load]);

   /* A lane owns two *horizontally* adjacent elements (col, col+1) of the
    * tile, so in the row-major orientation its whole share of the tile is one
    * contiguous vector in memory and can be moved with a single access. In
    * the column-major orientation the same two elements are a row stride
    * apart, so that case keeps the scalar path.
    *
    * The address is still computed in scalar elements (the row stride is
    * arbitrary), so the vector access is only naturally aligned when the
    * pointee is itself a vector: col is always even, hence so is
    * ptr_vec * stride * row + col when ptr_vec is. Claim no more than that -
    * agx lowers an under-aligned vec2 to a two-component 16-bit access, which
    * is still one instruction, rather than to two accesses.
    */
   if (row_major && length > 1 && hk_cmat_vecload()) {
      nir_def *idx =
         nir_iadd(b, nir_imul(b, nir_u2uN(b, row, idx_bits), stride),
                  nir_u2uN(b, col0, idx_bits));

      nir_deref_instr *e = nir_build_deref_ptr_as_array(b, deref, idx);
      const unsigned align = (ptr_vec % 2) ? elem_size_B : 2 * elem_size_B;

      /* Mode 2: when the pair is 4-byte aligned and 32 bits wide, move it as
       * one 32-bit scalar instead of a two-component 16-bit access. Both are
       * a single instruction; neither was faster here.
       */
      if (length * elem_size_B == 4 && align >= 4 && hk_cmat_vecload() == 2) {
         e = nir_build_deref_cast_with_alignment(b, &e->def, deref->modes,
                                                 glsl_uint_type(), 4, 4, 0);

         if (is_load) {
            nir_def *v = nir_load_deref(b, e);
            store_cmat_src(b, intr->src[0],
                           nir_unpack_32_2x16(b, v));
         } else {
            nir_store_deref(b, e, nir_pack_32_2x16(b, src), ~0);
         }

         nir_instr_remove(&intr->instr);
         return true;
      }

      e = nir_build_deref_cast_with_alignment(
         b, &e->def, deref->modes,
         glsl_vector_type(desc.element_type, length), length * elem_size_B,
         align, 0);

      if (is_load)
         store_cmat_src(b, intr->src[0], nir_load_deref(b, e));
      else
         nir_store_deref(b, e, src, ~0);

      nir_instr_remove(&intr->instr);
      return true;
   }

   nir_def *elems[NIR_MAX_VEC_COMPONENTS];
   if (!is_load) {
      for (unsigned p = 0; p < length; p++)
         elems[p] = nir_channel(b, src, p);
   }

   for (unsigned p = 0; p < length; p++) {
      nir_def *col = nir_iadd_imm(b, col0, p);
      nir_def *outer = row_major ? row : col;
      nir_def *inner = row_major ? col : row;
      nir_def *idx =
         nir_iadd(b, nir_imul(b, nir_u2uN(b, outer, idx_bits), stride),
                  nir_u2uN(b, inner, idx_bits));

      nir_deref_instr *e = nir_build_deref_ptr_as_array(b, deref, idx);

      if (is_load)
         elems[p] = nir_load_deref(b, e);
      else
         nir_store_deref(b, e, elems[p], ~0);
   }

   if (is_load)
      store_cmat_src(b, intr->src[0], nir_vec(b, elems, length));

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_instr(nir_builder *b, nir_instr *instr, struct lower_cmat_ctx *ctx)
{
   /* Remap deref types. Processed in reverse, so the intrinsics below still
    * see the original cmat-typed derefs when reading the description.
    */
   if (instr->type == nir_instr_type_deref) {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      return remap_type_in_place(ctx, &deref->type);
   }

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_construct: {
      nir_def *r = nir_replicate(
         b, intr->src[1].ssa, get_cmat_length(cmat_src_desc(intr->src[0])));
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_length: {
      const unsigned length = get_cmat_length(nir_intrinsic_cmat_desc(intr));
      nir_def_replace(&intr->def, nir_imm_int(b, length));
      return true;
   }

   case nir_intrinsic_cmat_extract: {
      nir_def *mat = load_cmat_src(b, intr->src[0]);
      nir_def *elem = nir_vector_extract(b, mat, intr->src[1].ssa);
      nir_def_replace(&intr->def, elem);
      return true;
   }

   case nir_intrinsic_cmat_insert: {
      nir_def *mat = load_cmat_src(b, intr->src[2]);
      nir_def *r = nir_vector_insert(b, mat, intr->src[1].ssa, intr->src[3].ssa);
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_copy: {
      nir_build_copy_deref(b, intr->src[0].ssa, intr->src[1].ssa);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store:
      return lower_cmat_load_store(b, intr, ctx);

   case nir_intrinsic_cmat_muladd: {
      struct glsl_cmat_description a_desc = cmat_src_desc(intr->src[1]);
      struct glsl_cmat_description c_desc = cmat_src_desc(intr->src[3]);
      unsigned ab_bits = glsl_base_type_bit_size(a_desc.element_type);
      unsigned c_bits = glsl_base_type_bit_size(c_desc.element_type);

      nir_def *a = load_cmat_src(b, intr->src[1]);
      nir_def *b_mat = load_cmat_src(b, intr->src[2]);
      nir_def *acc = load_cmat_src(b, intr->src[3]);

      /* Mixed f16 x f16 + f32 is supported natively: the per-operand size
       * flags are honoured and bit 26 (the fmadd32 variant) follows the
       * accumulator, verified bit-exact on G13G. HK_CMAT_WIDEN=1 instead
       * widens A and B to f32 elementwise (exact, same layout) and uses the
       * all-f32 form, kept for experiments.
       */
      if (ab_bits != c_bits) {
         static int widen = -1;
         if (widen < 0)
            widen = debug_get_bool_option("HK_CMAT_WIDEN", false);

         if (widen) {
            a = nir_f2f32(b, a);
            b_mat = nir_f2f32(b, b_mat);
         }
      }

      acc = nir_cmat_muladd_agx(b, a, b_mat, acc);
      store_cmat_src(b, intr->src[0], acc);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_unary_op:
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op: {
      nir_def *a = load_cmat_src(b, intr->src[1]);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);

      nir_def *r;
      if (intr->intrinsic == nir_intrinsic_cmat_unary_op) {
         r = nir_build_alu1(b, nir_intrinsic_alu_op(intr), a);
      } else {
         nir_def *y = intr->intrinsic == nir_intrinsic_cmat_binary_op
                         ? load_cmat_src(b, intr->src[2])
                         : intr->src[2].ssa;
         r = nir_build_alu2(b, nir_intrinsic_alu_op(intr), a, y);
      }

      b->fp_math_ctrl = nir_fp_fast_math;
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_bitcast: {
      store_cmat_src(b, intr->src[0], load_cmat_src(b, intr->src[1]));
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_convert: {
      struct glsl_cmat_description dst = cmat_src_desc(intr->src[0]);
      struct glsl_cmat_description src = cmat_src_desc(intr->src[1]);
      nir_def *ret = load_cmat_src(b, intr->src[1]);

      /* Same layout for every use and type, so a convert is elementwise. */
      if (dst.element_type != src.element_type) {
         b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
         nir_op op = nir_type_conversion_op(
            nir_get_nir_type_for_glsl_base_type(src.element_type),
            nir_get_nir_type_for_glsl_base_type(dst.element_type),
            nir_rounding_mode_undef);
         ret = nir_build_alu1(b, op, ret);
         b->fp_math_ctrl = nir_fp_fast_math;
      }

      store_cmat_src(b, intr->src[0], ret);
      nir_instr_remove(instr);
      return true;
   }

   default:
      return false;
   }
}

static bool
lower_cmat_impl(nir_function_impl *impl, struct lower_cmat_ctx *ctx)
{
   bool progress = false;

   nir_foreach_function_temp_variable(var, impl) {
      if (remap_type_in_place(ctx, &var->type))
         progress = true;
   }

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_reverse_safe(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         if (lower_cmat_instr(&b, instr, ctx))
            progress = true;
      }
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

bool
hk_nir_lower_cooperative_matrix(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE ||
       !nir->info.cs.has_cooperative_matrix)
      return false;

   /* Everything becomes 8x8 tiles first. */
   struct nir_lower_coopmat_args args = {
      .m_gran = HK_CMAT_HW_DIM,
      .n_gran = HK_CMAT_HW_DIM,
      .k_gran = HK_CMAT_HW_DIM,
   };
   bool progress = nir_lower_cooperative_matrix_flexible_dimensions(nir, &args);

   if (progress) {
      NIR_PASS(_, nir, nir_opt_deref);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables,
               nir_var_function_temp | nir_var_shader_temp, NULL);
   }

   struct lower_cmat_ctx ctx = {
      .type_mapping = _mesa_pointer_hash_table_create(NULL),
   };

   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp) {
      if (remap_type_in_place(&ctx, &var->type))
         progress = true;
   }

   nir_foreach_function_impl(impl, nir) {
      if (lower_cmat_impl(impl, &ctx))
         progress = true;
   }

   _mesa_hash_table_destroy(ctx.type_mapping, NULL);
   return progress;
}
