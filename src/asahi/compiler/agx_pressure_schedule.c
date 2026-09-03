/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

/* Bottom-up local scheduler to reduce register pressure and hide memory
 * latency.
 *
 * The scheduler is a bottom-up list scheduler. Its primary heuristic is
 * register pressure: of the instructions that are ready, choose the one that
 * reduces liveness the most. On top of that we model latency: an asynchronous
 * instruction (one that takes a scoreboard slot, i.e. a memory access) is not
 * considered "ready" until enough instructions have been scheduled after it to
 * cover its latency. That pushes async instructions away from their consumers,
 * so independent work lands in between and the hardware can keep multiple
 * messages in flight instead of draining the scoreboard after every load.
 *
 * Hiding latency costs registers, so we only accept a schedule if it either
 * lowers peak pressure (the old criterion) or lowers our stall estimate while
 * keeping peak pressure under the occupancy budget.
 */

#include "util/dag.h"
#include "util/sparse_bitset.h"
#include "util/u_debug.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/* Modelled latency of an asynchronous (scoreboarded) instruction, in
 * instructions. Real memory latency is far higher, but every instruction we
 * hoist across costs registers, so this is a budget rather than a measurement.
 */
#define AGX_ASYNC_LATENCY_DEFAULT (16)

/* Peak pressure (in 16-bit registers) we allow a latency-motivated schedule to
 * reach. 104 half-registers is the limit for the maximum 1024 threads/core.
 */
#define AGX_SCHED_MAX_REGS_DEFAULT (88)

struct sched_ctx {
   /* Dependency graph */
   struct dag *dag;

   /* Live set */
   struct u_sparse_bitset live;

   /* Size in 16-bit registers of each SSA value, indexed by value */
   uint8_t *size16;

   /* Bottom-up step counter: how many instructions have been scheduled after
    * (in program order) the one we are about to choose.
    */
   uint32_t step;

   /* Tunables */
   unsigned async_latency;
};

struct sched_node {
   struct dag_node dag;

   /* Instruction this node represents */
   agx_instr *instr;

   /* Earliest step at which this node may be scheduled without stalling its
    * consumers. Bottom-up, so larger means "further from the end of the
    * block".
    */
   uint32_t ready;
};

/*
 * Latency of an instruction's result, in instructions. Only asynchronous
 * instructions (device/stack/texture accesses) have a latency worth modelling;
 * everything else is interlocked by the hardware.
 */
static unsigned
instr_latency(const agx_instr *I, unsigned async_latency)
{
   if (agx_opcodes_info[I->op].immediates & AGX_IMMEDIATE_SCOREBOARD)
      return async_latency;
   else
      return 1;
}

static void
add_dep(struct sched_node *a, struct sched_node *b, uintptr_t latency)
{
   assert(a != b && "no self-dependencies");

   if (a && b)
      dag_add_edge_max_data(&a->dag, &b->dag, latency);
}

static void
serialize(struct sched_node *a, struct sched_node **b)
{
   add_dep(a, *b, 1);
   *b = a;
}

static struct dag *
create_dag(agx_context *ctx, agx_block *block, void *memctx, unsigned latency)
{
   struct dag *dag = dag_create(ctx);

   struct sched_node **last_write =
      calloc(ctx->alloc, sizeof(struct sched_node *));
   struct sched_node *coverage = NULL;
   struct sched_node *preload = NULL;

   /* Last memory load, to serialize stores against */
   struct sched_node *memory_load = NULL;

   /* Last memory store, to serialize loads and stores against */
   struct sched_node *memory_store = NULL;

   agx_foreach_instr_in_block(block, I) {
      /* Don't touch control flow */
      if (instr_after_logical_end(I))
         break;

      struct sched_node *node = rzalloc(memctx, struct sched_node);
      node->instr = I;
      dag_init_node(dag, &node->dag);

      /* Reads depend on writes, no other hazards in SSA */
      agx_foreach_ssa_src(I, s) {
         struct sched_node *def = last_write[I->src[s].value];

         if (def)
            add_dep(node, def, instr_latency(def->instr, latency));
      }

      agx_foreach_ssa_dest(I, d) {
         assert(I->dest[d].value < ctx->alloc);
         last_write[I->dest[d].value] = node;
      }

      /* Classify the instruction and add dependencies according to the class */
      enum agx_schedule_class dep = agx_opcodes_info[I->op].schedule_class;
      assert(dep != AGX_SCHEDULE_CLASS_INVALID && "invalid instruction seen");

      bool barrier = dep == AGX_SCHEDULE_CLASS_BARRIER;
      bool discards =
         I->op == AGX_OPCODE_SAMPLE_MASK || I->op == AGX_OPCODE_ZS_EMIT;

      if (dep == AGX_SCHEDULE_CLASS_STORE)
         add_dep(node, memory_load, 1);
      else if (dep == AGX_SCHEDULE_CLASS_ATOMIC || barrier)
         serialize(node, &memory_load);

      if (dep == AGX_SCHEDULE_CLASS_LOAD || dep == AGX_SCHEDULE_CLASS_STORE ||
          dep == AGX_SCHEDULE_CLASS_ATOMIC || barrier)
         serialize(node, &memory_store);

      if (dep == AGX_SCHEDULE_CLASS_COVERAGE || barrier)
         serialize(node, &coverage);

      /* Make sure side effects happen before a discard */
      if (discards)
         add_dep(node, memory_store, 1);

      if (dep == AGX_SCHEDULE_CLASS_PRELOAD)
         serialize(node, &preload);
      else
         add_dep(node, preload, 1);
   }

   free(last_write);

   return dag;
}

/*
 * Calculate the change in register pressure from scheduling a given
 * instruction. Equivalently, calculate the difference in the number of live
 * registers before and after the instruction, given the live set after the
 * instruction. This calculation follows immediately from the dataflow
 * definition of liveness:
 *
 *      live_in = (live_out - KILL) + GEN
 */
static signed
calculate_pressure_delta(agx_instr *I, struct u_sparse_bitset *live)
{
   signed delta = 0;

   /* Destinations must be unique */
   agx_foreach_ssa_dest(I, d) {
      if (u_sparse_bitset_test(live, I->dest[d].value))
         delta -= agx_index_size_16(I->dest[d]);
   }

   agx_foreach_ssa_src(I, src) {
      /* Filter duplicates */
      bool dupe = false;

      for (unsigned i = 0; i < src; ++i) {
         if (agx_is_equiv(I->src[i], I->src[src])) {
            dupe = true;
            break;
         }
      }

      if (!dupe && !u_sparse_bitset_test(live, I->src[src].value))
         delta += agx_index_size_16(I->src[src]);
   }

   return delta;
}

/*
 * Number of 16-bit registers held by a live set. Used to turn the scheduler's
 * relative pressure numbers into absolute ones, so we can compare against the
 * occupancy budget.
 */
static unsigned
live_halfregs(struct u_sparse_bitset *s, const uint8_t *size16)
{
   unsigned sum = 0;

   rb_tree_foreach(struct u_sparse_bitset_node, node, &s->tree, node) {
      unsigned b;

      BITSET_FOREACH_SET(b, node->vals, U_SPARSE_BITSET_BITS_PER_NODE) {
         sum += size16[node->offset + b];
      }
   }

   return sum;
}

/*
 * Estimate how many cycles a given instruction order stalls waiting on
 * asynchronous results. For each value produced asynchronously, we charge the
 * gap between its modelled latency and the distance to its first consumer,
 * which is where the wait will land.
 */
static unsigned
estimate_stall(agx_instr **order, unsigned n, unsigned *def_pos, unsigned alloc,
               unsigned latency)
{
   unsigned stall = 0;

   for (unsigned i = 0; i < alloc; ++i)
      def_pos[i] = ~0u;

   for (unsigned i = 0; i < n; ++i) {
      agx_instr *I = order[i];

      agx_foreach_ssa_src(I, s) {
         unsigned v = I->src[s].value;
         assert(v < alloc);

         if (def_pos[v] == ~0u)
            continue;

         unsigned p = def_pos[v];
         unsigned l = instr_latency(order[p], latency);

         if (i - p < l)
            stall += l - (i - p);

         /* The wait is emitted at the first consumer, so later consumers are
          * free.
          */
         def_pos[v] = ~0u;
      }

      agx_foreach_ssa_dest(I, d)
         def_pos[I->dest[d].value] = i;
   }

   return stall;
}

/*
 * Choose the next instruction, bottom-up. Among the instructions that are
 * ready (i.e. far enough from their consumers to cover their latency), choose
 * the one that has the best effect on liveness, while hoisting sample_mask. If
 * nothing is ready, we have to stall anyway, so fall back to pure pressure.
 */
static struct sched_node *
choose_instr(struct sched_ctx *s, bool require_ready)
{
   int32_t min_delta = INT32_MAX;
   struct sched_node *best = NULL;

   list_for_each_entry(struct sched_node, n, &s->dag->heads, dag.link) {
      if (require_ready && n->ready > s->step)
         continue;

      /* Heuristic: hoist sample_mask/zs_emit. This allows depth/stencil tests
       * to run earlier, and potentially to discard the entire quad invocation
       * earlier, reducing how much redundant fragment shader we run.
       *
       * Since we schedule backwards, we make that happen by only choosing
       * sample_mask when all other instructions have been exhausted.
       */
      if (n->instr->op == AGX_OPCODE_SAMPLE_MASK ||
          n->instr->op == AGX_OPCODE_ZS_EMIT) {

         /* Only ever chosen in the unrestricted pass, so that a latency stall
          * never causes us to sink sample_mask past real work.
          */
         if (!require_ready && !best) {
            best = n;
            assert(min_delta == INT32_MAX);
         }

         continue;
      }

      /* Heuristic: sink wait_pix to increase parallelism. Since wait_pix does
       * not read or write registers, this has no effect on pressure.
       */
      if (n->instr->op == AGX_OPCODE_WAIT_PIX)
         return n;

      int32_t delta = calculate_pressure_delta(n->instr, &s->live);

      if (delta < min_delta) {
         best = n;
         min_delta = delta;
      }
   }

   return best;
}

static void
pressure_schedule_block(agx_context *ctx, agx_block *block, struct sched_ctx *s,
                        unsigned *def_pos, unsigned max_regs)
{
   signed pressure = 0;
   signed orig_max_pressure = 0;
   unsigned nr_ins = 0;

   u_sparse_bitset_dup(&s->live, &block->live_out);
   unsigned base = live_halfregs(&s->live, s->size16);

   agx_foreach_instr_in_block_rev(block, I) {
      pressure += calculate_pressure_delta(I, &s->live);
      orig_max_pressure = MAX2(pressure, orig_max_pressure);
      agx_liveness_ins_update(&s->live, I);
      nr_ins++;
   }

   u_sparse_bitset_dup(&s->live, &block->live_out);

   signed max_pressure = 0;
   pressure = 0;

   struct sched_node **schedule = calloc(nr_ins, sizeof(struct sched_node *));
   nr_ins = 0;
   s->step = 0;

   while (!list_is_empty(&s->dag->heads)) {
      struct sched_node *node = choose_instr(s, true);

      /* Nothing is ready: we have to stall, so pick on pressure alone. */
      if (!node)
         node = choose_instr(s, false);

      assert(node != NULL);

      pressure += calculate_pressure_delta(node->instr, &s->live);
      max_pressure = MAX2(pressure, max_pressure);

      /* Propagate readiness to the producers before the edges are cleared */
      util_dynarray_foreach(&node->dag.edges, struct dag_edge, edge) {
         if (!edge->child)
            continue;

         struct sched_node *child =
            rb_node_data(struct sched_node, edge->child, dag);

         child->ready = MAX2(child->ready, s->step + edge->data);
      }

      dag_prune_head(s->dag, &node->dag);

      schedule[nr_ins++] = node;
      agx_liveness_ins_update(&s->live, node->instr);
      s->step++;
   }

   /* The old criterion: strictly less pressure is always good. */
   bool accept = max_pressure < orig_max_pressure;

   /* Otherwise, accept if we hid more latency and stayed within budget. */
   if (!accept && (unsigned)(base + max_pressure) <= max_regs) {
      agx_instr **order = calloc(nr_ins, sizeof(agx_instr *));
      agx_instr **orig = calloc(nr_ins, sizeof(agx_instr *));
      unsigned i = 0;

      /* schedule[] is in reverse program order */
      for (unsigned j = 0; j < nr_ins; ++j)
         order[nr_ins - 1 - j] = schedule[j]->instr;

      agx_foreach_instr_in_block(block, I) {
         if (instr_after_logical_end(I))
            break;

         orig[i++] = I;
      }
      assert(i == nr_ins);

      unsigned stall_new = estimate_stall(order, nr_ins, def_pos, ctx->alloc,
                                          s->async_latency);
      unsigned stall_old = estimate_stall(orig, nr_ins, def_pos, ctx->alloc,
                                          s->async_latency);

      accept = stall_new < stall_old;

      free(order);
      free(orig);
   }

   if (!accept) {
      free(schedule);
      return;
   }

   /* Apply the schedule */
   for (unsigned i = 0; i < nr_ins; ++i) {
      agx_remove_instruction(schedule[i]->instr);
      list_add(&schedule[i]->instr->link, &block->instructions);
   }

   free(schedule);
}

void
agx_pressure_schedule(agx_context *ctx)
{
   unsigned latency =
      debug_get_num_option("AGX_SCHED_LATENCY", AGX_ASYNC_LATENCY_DEFAULT);
   unsigned max_regs =
      debug_get_num_option("AGX_SCHED_MAX_REGS", AGX_SCHED_MAX_REGS_DEFAULT);

   agx_compute_liveness(ctx);
   void *memctx = ralloc_context(ctx);

   /* Record the size of every SSA value so we can measure absolute pressure */
   uint8_t *size16 = rzalloc_array(memctx, uint8_t, ctx->alloc);
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         size16[I->dest[d].value] = agx_index_size_16(I->dest[d]);
   }

   unsigned *def_pos = rzalloc_array(memctx, unsigned, ctx->alloc);

   agx_foreach_block(ctx, block) {
      struct sched_ctx sctx = {
         .dag = create_dag(ctx, block, memctx, latency),
         .size16 = size16,
         .async_latency = latency,
      };
      u_sparse_bitset_init(&sctx.live, ctx->alloc, memctx);

      pressure_schedule_block(ctx, block, &sctx, def_pos, max_regs);
   }

   /* Clean up after liveness analysis */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         I->src[s].kill = false;
   }

   ralloc_free(memctx);
}
