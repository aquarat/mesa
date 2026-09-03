/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "util/u_debug.h"

#define AGX_MAX_PENDING (8)

/*
 * Returns whether an instruction is asynchronous and needs a scoreboard slot
 */
static bool
instr_is_async(agx_instr *I)
{
   return agx_opcodes_info[I->op].immediates & AGX_IMMEDIATE_SCOREBOARD;
}

struct slot {
   /* Set of registers this slot is currently writing */
   BITSET_DECLARE(writes, AGX_NUM_REGS);

   /* Number of pending messages on this slot. Must not exceed
    * AGX_MAX_PENDING for correct results.
    */
   uint8_t nr_pending;

   /* Position of the earliest instruction that will force a wait on this slot,
    * i.e. the first consumer of any message pending on it. UINT32_MAX if the
    * slot is empty.
    */
   uint32_t drain;
};

/*
 * For every asynchronous instruction in the block, find the position of the
 * first instruction that touches one of its destinations. That is where the
 * wait draining it will be inserted, so it tells us which messages want to
 * share a scoreboard slot.
 */
static void
compute_drain_points(agx_block *block, unsigned nr_instrs, uint32_t *first_use)
{
   uint32_t *next_access = malloc(AGX_NUM_REGS * sizeof(uint32_t));

   for (unsigned i = 0; i < AGX_NUM_REGS; ++i)
      next_access[i] = UINT32_MAX;

   unsigned pos = nr_instrs;

   agx_foreach_instr_in_block_rev(block, I) {
      assert(pos > 0);
      pos--;

      if (instr_is_async(I)) {
         uint32_t f = UINT32_MAX;

         agx_foreach_dest(I, d) {
            if (agx_is_null(I->dest[d]) ||
                I->dest[d].type != AGX_INDEX_REGISTER)
               continue;

            unsigned n = agx_index_size_16(I->dest[d]);
            for (unsigned r = 0; r < n; ++r)
               f = MIN2(f, next_access[I->dest[d].value + r]);
         }

         first_use[pos] = f;
      }

      agx_foreach_src(I, s) {
         if (I->src[s].type != AGX_INDEX_REGISTER)
            continue;

         unsigned n = agx_index_size_16(I->src[s]);
         for (unsigned r = 0; r < n; ++r)
            next_access[I->src[s].value + r] = pos;
      }

      agx_foreach_dest(I, d) {
         if (agx_is_null(I->dest[d]) || I->dest[d].type != AGX_INDEX_REGISTER)
            continue;

         unsigned n = agx_index_size_16(I->dest[d]);
         for (unsigned r = 0; r < n; ++r)
            next_access[I->dest[d].value + r] = pos;
      }
   }

   free(next_access);
}

/*
 * Pick a scoreboard slot for an asynchronous instruction whose first consumer
 * is at position first_use.
 *
 * A wait drains every message pending on a slot, so a message should share a
 * slot with messages that are consumed at the same place. Failing that, an
 * empty slot is ideal. Failing that, join the slot that will be drained
 * latest, so we do not make an earlier consumer wait on a message it does not
 * need.
 */
static unsigned
choose_slot(const struct slot *slots, unsigned nr_slots, uint32_t first_use,
            unsigned mode)
{
   if (mode == 0) {
      /* Upstream behaviour: first free slot, else whatever was there. */
      for (unsigned slot = 0; slot < nr_slots; ++slot) {
         if (slots[slot].nr_pending == 0)
            return slot;
      }

      return 0;
   }

   if (mode >= 2) {
      for (unsigned slot = 0; slot < nr_slots; ++slot) {
         if (slots[slot].nr_pending && slots[slot].drain == first_use &&
             slots[slot].nr_pending < AGX_MAX_PENDING)
            return slot;
      }
   }

   for (unsigned slot = 0; slot < nr_slots; ++slot) {
      if (slots[slot].nr_pending == 0)
         return slot;
   }

   unsigned best = 0;
   for (unsigned slot = 1; slot < nr_slots; ++slot) {
      if (slots[slot].drain > slots[best].drain)
         best = slot;
   }

   return best;
}

/*
 * Insert waits within a block to stall after every async instruction. Useful
 * for debugging.
 */
static void
agx_insert_waits_trivial(agx_context *ctx, agx_block *block)
{
   agx_foreach_instr_in_block_safe(block, I) {
      if (instr_is_async(I)) {
         agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
         agx_wait(&b, I->scoreboard);
      }
   }
}

/*
 * Insert waits within a block, assuming scoreboard slots have already been
 * assigned. This waits for everything at the end of the block, rather than
 * doing something more intelligent/global. This should be optimized.
 *
 * XXX: Do any instructions read their sources asynchronously?
 */
static void
agx_insert_waits_local(agx_context *ctx, agx_block *block)
{
   struct slot slots[2] = {0};
   unsigned mode = debug_get_num_option("AGX_WAIT_SLOT", 2);
   unsigned nr_instrs = 0;
   uint32_t *first_use = NULL;

   slots[0].drain = slots[1].drain = UINT32_MAX;

   if (mode) {
      agx_foreach_instr_in_block(block, I)
         nr_instrs++;

      first_use = calloc(nr_instrs ?: 1, sizeof(uint32_t));
      compute_drain_points(block, nr_instrs, first_use);
   }

   unsigned pos = 0;

   agx_foreach_instr_in_block_safe(block, I) {
      uint8_t wait_mask = 0;
      unsigned cur = pos++;

      /* Check for read-after-write */
      agx_foreach_src(I, s) {
         if (I->src[s].type != AGX_INDEX_REGISTER)
            continue;

         unsigned nr_read = agx_index_size_16(I->src[s]);
         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (BITSET_TEST_COUNT(slots[slot].writes, I->src[s].value, nr_read))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Check for write-after-write */
      agx_foreach_dest(I, d) {
         if (I->dest[d].type != AGX_INDEX_REGISTER)
            continue;

         unsigned nr_writes = agx_index_size_16(I->dest[d]);
         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (BITSET_TEST_COUNT(slots[slot].writes, I->dest[d].value,
                                  nr_writes))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Check for barriers */
      if (I->op == AGX_OPCODE_THREADGROUP_BARRIER ||
          I->op == AGX_OPCODE_MEMORY_BARRIER) {

         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (slots[slot].nr_pending)
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Assign a scoreboard slot */
      if (instr_is_async(I)) {
         I->scoreboard = choose_slot(slots, ARRAY_SIZE(slots),
                                     first_use ? first_use[cur] : UINT32_MAX,
                                     mode);
      }

      /* Check for slot overflow */
      if (instr_is_async(I) &&
          slots[I->scoreboard].nr_pending >= AGX_MAX_PENDING)
         wait_mask |= BITSET_BIT(I->scoreboard);

      /* Insert the appropriate waits, clearing the slots */
      u_foreach_bit(slot, wait_mask) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_wait(&b, slot);

         BITSET_ZERO(slots[slot].writes);
         slots[slot].nr_pending = 0;
         slots[slot].drain = UINT32_MAX;
      }

      /* Record access */
      if (instr_is_async(I)) {
         agx_foreach_dest(I, d) {
            if (agx_is_null(I->dest[d]))
               continue;

            assert(I->dest[d].type == AGX_INDEX_REGISTER);
            BITSET_SET_COUNT(slots[I->scoreboard].writes, I->dest[d].value,
                             agx_index_size_16(I->dest[d]));
         }

         slots[I->scoreboard].nr_pending++;

         if (first_use) {
            slots[I->scoreboard].drain =
               MIN2(slots[I->scoreboard].drain, first_use[cur]);
         }
      }
   }

   free(first_use);

   /* If there are outstanding messages, wait for them. We don't do this for the
    * exit block, though, since nothing else will execute in the shader so
    * waiting is pointless.
    */
   if (block != agx_exit_block(ctx)) {
      agx_builder b = agx_init_builder(ctx, agx_after_block_logical(block));

      for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
         if (slots[slot].nr_pending) {
            agx_wait(&b, slot);
            ctx->nr_eob_waits++;
         }
      }
   }
}

/*
 * Assign scoreboard slots to asynchronous instructions and insert waits for the
 * appropriate hazard tracking.
 */
void
agx_insert_waits(agx_context *ctx)
{
   agx_foreach_block(ctx, block) {
      if (agx_compiler_debug & AGX_DBG_WAIT)
         agx_insert_waits_trivial(ctx, block);
      else
         agx_insert_waits_local(ctx, block);
   }
}
