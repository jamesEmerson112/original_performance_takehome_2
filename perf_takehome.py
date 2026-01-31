"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class ListScheduler:
    """
    VLIW instruction scheduler with automatic dependency detection.

    Collects operations, detects RAW/WAW/WAR hazards, and schedules
    them into packed VLIW bundles that maximize engine utilization.

    Usage:
        sched = ListScheduler(vlen=8)
        sched.add_op("valu", ("%", v_tmp1, v_val, v_two))
        sched.add_op("valu", ("==", v_tmp1, v_tmp1, v_zero))
        sched.add_op("flow", ("vselect", v_tmp2, v_tmp1, v_one, v_two))
        for bundle in sched.schedule():
            self.add_packed(bundle)
    """

    SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

    def __init__(self, vlen: int = 8):
        self.vlen = vlen
        self.ops = []  # [(engine, slot, op_id), ...]
        self.op_id_counter = 0
        self.reads = {}   # op_id -> set[int] addresses read
        self.writes = {}  # op_id -> set[int] addresses written
        self.successors = defaultdict(set)
        self.predecessors = defaultdict(set)

    def _parse_reads_writes(self, engine: str, slot: tuple) -> tuple:
        """Extract read/write addresses from instruction tuple."""
        reads, writes = set(), set()

        if engine == "alu":
            # (op, dest, src1, src2)
            op, dest, src1, src2 = slot
            reads.update([src1, src2])
            writes.add(dest)

        elif engine == "valu":
            if slot[0] == "vbroadcast":
                # ("vbroadcast", dest, src) - scalar src, vector dest
                _, dest, src = slot
                reads.add(src)
                writes.update(range(dest, dest + self.vlen))
            elif slot[0] == "multiply_add":
                # ("multiply_add", dest, a, b, c)
                _, dest, a, b, c = slot
                for base in [a, b, c]:
                    reads.update(range(base, base + self.vlen))
                writes.update(range(dest, dest + self.vlen))
            else:
                # (op, dest, src1, src2) - all vectors
                op, dest, src1, src2 = slot
                reads.update(range(src1, src1 + self.vlen))
                reads.update(range(src2, src2 + self.vlen))
                writes.update(range(dest, dest + self.vlen))

        elif engine == "load":
            if slot[0] == "const":
                # ("const", dest, val) - no reads
                _, dest, _ = slot
                writes.add(dest)
            elif slot[0] == "load":
                # ("load", dest, addr)
                _, dest, addr = slot
                reads.add(addr)
                writes.add(dest)
            elif slot[0] == "vload":
                # ("vload", dest, addr) - scalar addr, vector dest
                _, dest, addr = slot
                reads.add(addr)
                writes.update(range(dest, dest + self.vlen))
            elif slot[0] == "load_offset":
                # ("load_offset", dest, addr, offset)
                _, dest, addr, offset = slot
                reads.add(addr)
                writes.add(dest)

        elif engine == "store":
            if slot[0] == "store":
                # ("store", addr, src)
                _, addr, src = slot
                reads.update([addr, src])
            elif slot[0] == "vstore":
                # ("vstore", addr, src) - scalar addr, vector src
                _, addr, src = slot
                reads.add(addr)
                reads.update(range(src, src + self.vlen))

        elif engine == "flow":
            if slot[0] == "vselect":
                # ("vselect", dest, cond, a, b) - all vectors
                _, dest, cond, a, b = slot
                for base in [cond, a, b]:
                    reads.update(range(base, base + self.vlen))
                writes.update(range(dest, dest + self.vlen))
            elif slot[0] == "select":
                # ("select", dest, cond, a, b) - scalar
                _, dest, cond, a, b = slot
                reads.update([cond, a, b])
                writes.add(dest)
            elif slot[0] in ("pause", "halt"):
                pass  # No dependencies
            elif slot[0] in ("cond_jump", "cond_jump_rel"):
                _, cond, _ = slot
                reads.add(cond)
            elif slot[0] == "add_imm":
                _, dest, a, _ = slot
                reads.add(a)
                writes.add(dest)
            elif slot[0] == "coreid":
                _, dest = slot
                writes.add(dest)

        elif engine == "debug":
            pass  # Debug ops have no dependencies

        return reads, writes

    def add_op(self, engine: str, slot: tuple) -> int:
        """
        Add an operation with automatic dependency detection.

        Returns the op_id for this operation.
        """
        op_id = self.op_id_counter
        self.op_id_counter += 1

        reads, writes = self._parse_reads_writes(engine, slot)
        self.reads[op_id] = reads
        self.writes[op_id] = writes
        self.ops.append((engine, slot, op_id))

        # Detect dependencies with all previous ops
        for prev_engine, prev_slot, prev_id in self.ops[:-1]:
            prev_writes = self.writes[prev_id]
            prev_reads = self.reads[prev_id]

            has_dep = False
            # RAW: this op reads what prev wrote
            if reads & prev_writes:
                has_dep = True
            # WAW: both write to same address
            if writes & prev_writes:
                has_dep = True
            # WAR: this op writes what prev reads
            if writes & prev_reads:
                has_dep = True

            if has_dep:
                self.successors[prev_id].add(op_id)
                self.predecessors[op_id].add(prev_id)

        return op_id

    def _priority(self, op_id: int) -> tuple:
        """
        Calculate scheduling priority for an operation.

        Priority (higher = scheduled first):
        1. FLOW operations get highest priority (bottleneck)
        2. Operations with more successors (critical path)
        3. Earlier op_id (preserve program order for ties)
        """
        engine = self.ops[op_id][0]

        # Engine priority: FLOW > STORE > LOAD > VALU > ALU
        engine_prio = {
            "flow": 100,
            "store": 80,
            "load": 60,
            "valu": 40,
            "alu": 20,
            "debug": 0,
        }.get(engine, 0)

        # Critical path: count transitive successors
        visited = set()
        stack = [op_id]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                stack.extend(self.successors[curr])

        critical_path = len(visited) - 1

        return (engine_prio, critical_path, -op_id)

    def schedule(self) -> list:
        """
        Schedule all operations into VLIW bundles.

        Returns a list of instruction bundles:
        [{"flow": [...], "valu": [...], "alu": [...], ...}, ...]
        """
        if not self.ops:
            return []

        scheduled = set()
        # Initially ready: ops with no predecessors
        ready = {op_id for _, _, op_id in self.ops if not self.predecessors[op_id]}
        bundles = []

        while ready:
            bundle = defaultdict(list)
            slots_left = dict(self.SLOT_LIMITS)
            this_cycle = []

            # Sort by priority (descending)
            for op_id in sorted(ready, key=self._priority, reverse=True):
                engine, slot, _ = self.ops[op_id]

                # Check if we have slots available for this engine
                if slots_left.get(engine, 0) > 0:
                    bundle[engine].append(slot)
                    slots_left[engine] -= 1
                    this_cycle.append(op_id)

            # Update state
            for op_id in this_cycle:
                ready.remove(op_id)
                scheduled.add(op_id)

                # Check if successors are now ready
                for succ_id in self.successors[op_id]:
                    if succ_id not in scheduled:
                        # All predecessors must be scheduled
                        if all(pred_id in scheduled for pred_id in self.predecessors[succ_id]):
                            ready.add(succ_id)

            if bundle:
                bundles.append(dict(bundle))

        # Verify all ops scheduled
        if len(scheduled) != len(self.ops):
            unscheduled = [op_id for _, _, op_id in self.ops if op_id not in scheduled]
            raise RuntimeError(f"Scheduling failed: {len(unscheduled)} ops not scheduled")

        return bundles

    def clear(self):
        """Reset the scheduler for reuse."""
        self.ops = []
        self.op_id_counter = 0
        self.reads = {}
        self.writes = {}
        self.successors = defaultdict(set)
        self.predecessors = defaultdict(set)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def add_packed(self, instr_dict):
        """Add a packed VLIW instruction bundle"""
        self.instrs.append(instr_dict)

    def emit_vectorized_hash(self, v_hash, v_tmp1, v_tmp2, v_hash_consts):
        """
        Emit vectorized hash for 8 elements.
        6 stages x 3 ops = 18 valu ops.

        v_hash_consts: list of (v_c1, v_c3) vector constant addresses for each stage
        """
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            # Stage: tmp1 = hash op1 c1, tmp2 = hash op3 c3, hash = tmp1 op2 tmp2
            # Pack first two ops together
            self.add_packed({
                "valu": [
                    (op1, v_tmp1, v_hash, v_c1),
                    (op3, v_tmp2, v_hash, v_c3),
                ]
            })
            # Then the combining op
            self.add_packed({
                "valu": [(op2, v_hash, v_tmp1, v_tmp2)]
            })

    def emit_vectorized_hash_2way(self, v_hash_a, v_hash_b, v_tmp1_a, v_tmp2_a, v_tmp1_b, v_tmp2_b, v_hash_consts):
        """
        2-way batched hash: process both chunks in 12 cycles instead of 24.

        Each stage does 4 VALU ops (2 for each chunk) then 2 combine ops.
        Total: 6 stages × 2 cycles = 12 cycles for BOTH chunks.
        """
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            # Pack both chunks' first two ops together (4 VALU)
            self.add_packed({
                "valu": [
                    (op1, v_tmp1_a, v_hash_a, v_c1),
                    (op3, v_tmp2_a, v_hash_a, v_c3),
                    (op1, v_tmp1_b, v_hash_b, v_c1),
                    (op3, v_tmp2_b, v_hash_b, v_c3),
                ]
            })
            # Both combine ops (2 VALU)
            self.add_packed({
                "valu": [
                    (op2, v_hash_a, v_tmp1_a, v_tmp2_a),
                    (op2, v_hash_b, v_tmp1_b, v_tmp2_b),
                ]
            })

    def emit_vectorized_hash_3way(self, v_hashes, v_tmps, v_hash_consts):
        """
        3-way batched hash: process 3 chunks in 12 cycles using all 6 VALU slots.

        v_hashes: [v_hash_a, v_hash_b, v_hash_c]
        v_tmps: [v_tmp1_a, v_tmp2_a, v_tmp1_b, v_tmp2_b, v_tmp1_c, v_tmp2_c]
        """
        v_hash_a, v_hash_b, v_hash_c = v_hashes
        v_tmp1_a, v_tmp2_a, v_tmp1_b, v_tmp2_b, v_tmp1_c, v_tmp2_c = v_tmps

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            # All 6 VALU slots filled (2 ops per chunk × 3 chunks)
            self.add_packed({
                "valu": [
                    (op1, v_tmp1_a, v_hash_a, v_c1),
                    (op3, v_tmp2_a, v_hash_a, v_c3),
                    (op1, v_tmp1_b, v_hash_b, v_c1),
                    (op3, v_tmp2_b, v_hash_b, v_c3),
                    (op1, v_tmp1_c, v_hash_c, v_c1),
                    (op3, v_tmp2_c, v_hash_c, v_c3),
                ]
            })
            # 3 combine ops
            self.add_packed({
                "valu": [
                    (op2, v_hash_a, v_tmp1_a, v_tmp2_a),
                    (op2, v_hash_b, v_tmp1_b, v_tmp2_b),
                    (op2, v_hash_c, v_tmp1_c, v_tmp2_c),
                ]
            })

    def emit_pipelined_gather_and_hash(self, v_idx_a, v_val_a, v_idx_b, v_val_b,
                                        v_node_val, v_node_val_b, gather_addrs,
                                        forest_values_p, zero_const,
                                        v_tmp1, v_tmp2, v_tmp1_b, v_tmp2_b, v_hash_consts):
        """
        Pipelined gather + hash: overlap B's gather with A's hash.

        LOAD and VALU use different engines, so they can run in parallel.
        This saves ~4 cycles per pair.
        """
        # === Gather A: 2 ALU + 4 LOAD cycles ===
        # Cycle 1: A's ALU copy indices
        self.add_packed({"alu": [
            ("+", gather_addrs + 0, v_idx_a + 0, zero_const),
            ("+", gather_addrs + 1, v_idx_a + 1, zero_const),
            ("+", gather_addrs + 2, v_idx_a + 2, zero_const),
            ("+", gather_addrs + 3, v_idx_a + 3, zero_const),
            ("+", gather_addrs + 4, v_idx_a + 4, zero_const),
            ("+", gather_addrs + 5, v_idx_a + 5, zero_const),
            ("+", gather_addrs + 6, v_idx_a + 6, zero_const),
            ("+", gather_addrs + 7, v_idx_a + 7, zero_const),
        ]})
        # Cycle 2: A's ALU address calculation
        self.add_packed({"alu": [
            ("+", gather_addrs + 0, forest_values_p, gather_addrs + 0),
            ("+", gather_addrs + 1, forest_values_p, gather_addrs + 1),
            ("+", gather_addrs + 2, forest_values_p, gather_addrs + 2),
            ("+", gather_addrs + 3, forest_values_p, gather_addrs + 3),
            ("+", gather_addrs + 4, forest_values_p, gather_addrs + 4),
            ("+", gather_addrs + 5, forest_values_p, gather_addrs + 5),
            ("+", gather_addrs + 6, forest_values_p, gather_addrs + 6),
            ("+", gather_addrs + 7, forest_values_p, gather_addrs + 7),
        ]})
        # Cycles 3-6: A's LOAD
        self.add_packed({"load": [("load", v_node_val + 0, gather_addrs + 0), ("load", v_node_val + 1, gather_addrs + 1)]})
        self.add_packed({"load": [("load", v_node_val + 2, gather_addrs + 2), ("load", v_node_val + 3, gather_addrs + 3)]})
        self.add_packed({"load": [("load", v_node_val + 4, gather_addrs + 4), ("load", v_node_val + 5, gather_addrs + 5)]})
        self.add_packed({"load": [("load", v_node_val + 6, gather_addrs + 6), ("load", v_node_val + 7, gather_addrs + 7)]})

        # === A's XOR (1 VALU) ===
        self.add_packed({"valu": [("^", v_val_a, v_val_a, v_node_val)]})

        # === B's gather ALU (2 cycles) overlapped with A's hash start ===
        # Cycle 7: B's ALU copy + A's hash stage 1 first two ops
        self.add_packed({
            "alu": [
                ("+", gather_addrs + 0, v_idx_b + 0, zero_const),
                ("+", gather_addrs + 1, v_idx_b + 1, zero_const),
                ("+", gather_addrs + 2, v_idx_b + 2, zero_const),
                ("+", gather_addrs + 3, v_idx_b + 3, zero_const),
                ("+", gather_addrs + 4, v_idx_b + 4, zero_const),
                ("+", gather_addrs + 5, v_idx_b + 5, zero_const),
                ("+", gather_addrs + 6, v_idx_b + 6, zero_const),
                ("+", gather_addrs + 7, v_idx_b + 7, zero_const),
            ],
            "valu": [
                (HASH_STAGES[0][0], v_tmp1, v_val_a, v_hash_consts[0][0]),
                (HASH_STAGES[0][3], v_tmp2, v_val_a, v_hash_consts[0][1]),
            ],
        })
        # Cycle 8: B's ALU addr + A's hash stage 1 combine
        self.add_packed({
            "alu": [
                ("+", gather_addrs + 0, forest_values_p, gather_addrs + 0),
                ("+", gather_addrs + 1, forest_values_p, gather_addrs + 1),
                ("+", gather_addrs + 2, forest_values_p, gather_addrs + 2),
                ("+", gather_addrs + 3, forest_values_p, gather_addrs + 3),
                ("+", gather_addrs + 4, forest_values_p, gather_addrs + 4),
                ("+", gather_addrs + 5, forest_values_p, gather_addrs + 5),
                ("+", gather_addrs + 6, forest_values_p, gather_addrs + 6),
                ("+", gather_addrs + 7, forest_values_p, gather_addrs + 7),
            ],
            "valu": [(HASH_STAGES[0][2], v_val_a, v_tmp1, v_tmp2)],
        })

        # === B's LOAD (4 cycles) overlapped with A's hash stages 2-5 ===
        for load_idx, stage_idx in enumerate([1, 2, 3, 4]):
            op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
            v_c1, v_c3 = v_hash_consts[stage_idx]
            # First cycle of stage: LOAD + 2 VALU ops
            self.add_packed({
                "load": [
                    ("load", v_node_val_b + load_idx*2, gather_addrs + load_idx*2),
                    ("load", v_node_val_b + load_idx*2+1, gather_addrs + load_idx*2+1),
                ],
                "valu": [
                    (op1, v_tmp1, v_val_a, v_c1),
                    (op3, v_tmp2, v_val_a, v_c3),
                ],
            })
            # Second cycle of stage: combine op only
            self.add_packed({"valu": [(op2, v_val_a, v_tmp1, v_tmp2)]})

        # === A's hash stage 6 (last stage) ===
        op1, val1, op2, op3, val3 = HASH_STAGES[5]
        v_c1, v_c3 = v_hash_consts[5]
        self.add_packed({"valu": [(op1, v_tmp1, v_val_a, v_c1), (op3, v_tmp2, v_val_a, v_c3)]})
        self.add_packed({"valu": [(op2, v_val_a, v_tmp1, v_tmp2)]})

        # === B's XOR and hash ===
        self.add_packed({"valu": [("^", v_val_b, v_val_b, v_node_val_b)]})
        self.emit_vectorized_hash(v_val_b, v_tmp1_b, v_tmp2_b, v_hash_consts)

    def emit_compute_next_index(self, v_idx, v_val, v_tmp1, v_tmp2, v_one, v_two, v_zero, v_n_nodes):
        """
        Compute next index for 8 elements:
        idx = 2*idx + (1 if val%2==0 else 2)
        idx = 0 if idx >= n_nodes else idx

        Optimized: use arithmetic instead of vselect for direction.
        offset = (val % 2) + 1 gives 1 when even, 2 when odd.
        """
        # tmp1 = val % 2 (0 or 1)
        self.add_packed({"valu": [("%", v_tmp1, v_val, v_two)]})
        # tmp2 = tmp1 + 1 (1 or 2)
        self.add_packed({"valu": [("+", v_tmp2, v_tmp1, v_one)]})
        # idx = idx * 2 + tmp2 (using multiply_add)
        self.add_packed({"valu": [("multiply_add", v_idx, v_idx, v_two, v_tmp2)]})
        # tmp1 = (idx < n_nodes)
        self.add_packed({"valu": [("<", v_tmp1, v_idx, v_n_nodes)]})
        # idx = select(tmp1, idx, 0) - wrap to 0 if >= n_nodes
        self.add_packed({"flow": [("vselect", v_idx, v_tmp1, v_idx, v_zero)]})

    def emit_compute_next_index_2way(self, v_idx_a, v_val_a, v_idx_b, v_val_b,
                                      v_tmp1_a, v_tmp2_a, v_tmp1_b, v_tmp2_b,
                                      v_one, v_two, v_zero, v_n_nodes):
        """
        2-way interleaved index computation using arithmetic instead of vselect.

        Optimized: offset = (val % 2) + 1 eliminates one vselect per chunk.
        """
        # Cycle 1: A1, B1 - both mod operations (2 VALU)
        self.add_packed({"valu": [
            ("%", v_tmp1_a, v_val_a, v_two),
            ("%", v_tmp1_b, v_val_b, v_two),
        ]})

        # Cycle 2: A2, B2 - compute offset = (val%2) + 1 (2 VALU)
        self.add_packed({"valu": [
            ("+", v_tmp2_a, v_tmp1_a, v_one),
            ("+", v_tmp2_b, v_tmp1_b, v_one),
        ]})

        # Cycle 3: A3, B3 - idx = idx*2 + offset using multiply_add (2 VALU)
        self.add_packed({"valu": [
            ("multiply_add", v_idx_a, v_idx_a, v_two, v_tmp2_a),
            ("multiply_add", v_idx_b, v_idx_b, v_two, v_tmp2_b),
        ]})

        # Cycle 4: A4, B4 - compare with n_nodes (2 VALU)
        self.add_packed({"valu": [
            ("<", v_tmp1_a, v_idx_a, v_n_nodes),
            ("<", v_tmp1_b, v_idx_b, v_n_nodes),
        ]})

        # Cycle 6: A6 (FLOW) - wrap if needed
        self.add_packed({
            "flow": [("vselect", v_idx_a, v_tmp1_a, v_idx_a, v_zero)],
        })

        # Cycle 7: B6 (FLOW) - wrap if needed
        self.add_packed({
            "flow": [("vselect", v_idx_b, v_tmp1_b, v_idx_b, v_zero)],
        })

    def emit_xor_hash_index_scheduled(self, v_val_a, v_val_b, v_node_val_a, v_node_val_b,
                                        v_idx_a, v_idx_b,
                                        v_tmp1_a, v_tmp2_a, v_tmp1_b, v_tmp2_b,
                                        v_one, v_two, v_zero, v_n_nodes, v_hash_consts):
        """
        XOR + Hash + Index computation for a chunk pair using 2-way batching.

        For 2-way, the scheduler doesn't help much because the hash dependencies
        are strict. Keep the original manual implementation which is well-optimized.
        """
        # XOR both chunks (2 VALU ops fit in one cycle)
        self.add_packed({"valu": [
            ("^", v_val_a, v_val_a, v_node_val_a),
            ("^", v_val_b, v_val_b, v_node_val_b),
        ]})

        # 2-way batched hash (12 cycles for both instead of 24)
        self.emit_vectorized_hash_2way(
            v_val_a, v_val_b,
            v_tmp1_a, v_tmp2_a, v_tmp1_b, v_tmp2_b,
            v_hash_consts
        )

        # Compute next index - 2-way interleaved (7 cycles instead of 14)
        self.emit_compute_next_index_2way(
            v_idx_a, v_val_a, v_idx_b, v_val_b,
            v_tmp1_a, v_tmp2_a, v_tmp1_b, v_tmp2_b,
            v_one, v_two, v_zero, v_n_nodes
        )

    def emit_xor_hash_index_broadcast(self, chunk_indices, v_indices, v_values,
                                         v_broadcast_node_val, v_tmps, v_hash_consts,
                                         v_one, v_two, v_zero, v_n_nodes, needs_wrap=True):
        """
        Optimized XOR + Hash + Index for unique_count=1 (all chunks use same node value).

        Instead of copying the broadcast value to N separate registers, XOR directly
        with the single broadcast value. This saves N-1 VALU copies.

        v_broadcast_node_val: single broadcast vector used for ALL chunks
        v_tmps: list of (v_tmp1, v_tmp2) pairs - reused across chunks within scheduler
        """
        n = len(chunk_indices)
        sched = ListScheduler(vlen=VLEN)

        v_vals = [v_values + c * VLEN for c in chunk_indices]
        v_idxs = [v_indices + c * VLEN for c in chunk_indices]

        # XOR all chunks with the SAME broadcast value
        for i in range(n):
            sched.add_op("valu", ("^", v_vals[i], v_vals[i], v_broadcast_node_val))

        # Hash operations
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            for i in range(n):
                v_tmp1, v_tmp2 = v_tmps[i % len(v_tmps)]  # Reuse temps across chunks
                sched.add_op("valu", (op1, v_tmp1, v_vals[i], v_c1))
                sched.add_op("valu", (op3, v_tmp2, v_vals[i], v_c3))
                sched.add_op("valu", (op2, v_vals[i], v_tmp1, v_tmp2))

        # Index computation (using multiply_add)
        for i in range(n):
            v_tmp1, v_tmp2 = v_tmps[i % len(v_tmps)]
            sched.add_op("valu", ("%", v_tmp1, v_vals[i], v_two))
            sched.add_op("valu", ("+", v_tmp2, v_tmp1, v_one))
            sched.add_op("valu", ("multiply_add", v_idxs[i], v_idxs[i], v_two, v_tmp2))
            if needs_wrap:
                sched.add_op("valu", ("<", v_tmp1, v_idxs[i], v_n_nodes))
                sched.add_op("flow", ("vselect", v_idxs[i], v_tmp1, v_idxs[i], v_zero))

        for bundle in sched.schedule():
            self.add_packed(bundle)

    def emit_xor_hash_index_nway_scheduled(self, chunk_indices, v_indices, v_values,
                                            v_node_vals, v_tmps, v_hash_consts,
                                            v_one, v_two, v_zero, v_n_nodes,
                                            skip_xor_hash=False, needs_wrap=True):
        """
        N-way XOR + Hash + Index using ListScheduler.

        Processes N chunks together for better VALU utilization:
        - 6-way fills all 6 VALU slots during hash stages (100% utilization)
        - The scheduler automatically finds opportunities to overlap FLOW with VALU

        chunk_indices: list of chunk indices to process (e.g., [0,1,2,3,4,5])
        v_node_vals: list of N node value addresses
        v_tmps: list of N (v_tmp1, v_tmp2) tuples
        skip_xor_hash: if True, only do index computation (XOR+hash already done)
        needs_wrap: if False, skip the wrap vselect (idx >= n_nodes check)
        """
        n = len(chunk_indices)
        sched = ListScheduler(vlen=VLEN)

        # Get addresses for each chunk
        v_vals = [v_values + c * VLEN for c in chunk_indices]
        v_idxs = [v_indices + c * VLEN for c in chunk_indices]

        if not skip_xor_hash:
            # === XOR operations (N VALU) ===
            for i in range(n):
                sched.add_op("valu", ("^", v_vals[i], v_vals[i], v_node_vals[i]))

            # === Hash operations (6 stages × 3 ops × N chunks) ===
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                v_c1, v_c3 = v_hash_consts[hi]
                for i in range(n):
                    v_tmp1, v_tmp2 = v_tmps[i]
                    sched.add_op("valu", (op1, v_tmp1, v_vals[i], v_c1))
                    sched.add_op("valu", (op3, v_tmp2, v_vals[i], v_c3))
                    sched.add_op("valu", (op2, v_vals[i], v_tmp1, v_tmp2))

        # === Index computation ===
        # Optimized: use arithmetic instead of vselect for direction choice
        # offset = (val % 2) + 1 gives 1 when even, 2 when odd
        # Use multiply_add to combine idx*2 + offset into one op
        for i in range(n):
            v_tmp1, v_tmp2 = v_tmps[i]
            sched.add_op("valu", ("%", v_tmp1, v_vals[i], v_two))       # tmp1 = val % 2 (0 or 1)
            sched.add_op("valu", ("+", v_tmp2, v_tmp1, v_one))          # tmp2 = tmp1 + 1 (1 or 2)
            sched.add_op("valu", ("multiply_add", v_idxs[i], v_idxs[i], v_two, v_tmp2))  # idx = idx*2 + offset
            if needs_wrap:
                sched.add_op("valu", ("<", v_tmp1, v_idxs[i], v_n_nodes))   # tmp1 = (idx < n_nodes)
                sched.add_op("flow", ("vselect", v_idxs[i], v_tmp1, v_idxs[i], v_zero))  # wrap if needed

        # Schedule and emit
        for bundle in sched.schedule():
            self.add_packed(bundle)

    def emit_vselect_from_prebroadcast(self, v_result, v_idx, v_cache_broadcast,
                                         v_start_idx, unique_count, v_tmps, v_cond, v_zero, v_cmp_consts):
        """
        Select from pre-broadcast cache values using cascade.
        v_cache_broadcast: array of pre-broadcast vector values (unique_count × VLEN words)
        v_start_idx: pre-broadcast start_idx vector
        v_zero: pre-broadcast vector of zeros (for copying)
        v_cmp_consts: pre-broadcast comparison constants [0, 1, 2, ..., 31] × VLEN
        """
        if unique_count == 1:
            # Just copy the single pre-broadcast value
            self.add_packed({"valu": [("+", v_result, v_cache_broadcast, v_zero)]})
            return

        v_offset = v_tmps[0]

        # Compute offset = v_idx - start_idx
        self.add_packed({"valu": [("-", v_offset, v_idx, v_start_idx)]})

        # Initialize result with first cache entry
        self.add_packed({"valu": [("+", v_result, v_cache_broadcast, v_zero)]})

        # For each unique index > 0, conditionally update
        # Using pre-broadcast comparison constants (no broadcasts in this loop!)
        for u in range(1, unique_count):
            # Compare offset against pre-broadcast u
            self.add_packed({"valu": [("==", v_cond, v_offset, v_cmp_consts + u * VLEN)]})
            # vselect: if offset==u, use cache[u], else keep current result
            self.add_packed({"flow": [("vselect", v_result, v_cond, v_cache_broadcast + u * VLEN, v_result)]})

    def emit_fused_unique2_complete(self, chunk_indices, v_indices, v_values,
                                       v_cache_broadcast, v_start_idx, v_diff, v_tmps,
                                       v_hash_consts, v_one, v_two, v_zero, v_n_nodes,
                                       needs_wrap=False):
        """
        Fused selection + XOR + hash + index for unique_count=2.

        Combines all operations into a single ListScheduler call for better packing.
        This eliminates scheduler overhead between selection and compute phases.
        Temps are reused across chunks via modulo (scheduler handles dependencies).

        Operations per chunk:
        - Selection: 3 VALU (offset, multiply, add)
        - XOR: 1 VALU
        - Hash: 18 VALU (6 stages × 3 ops)
        - Index: 3 VALU (mod, +1, multiply_add)
        Total: 25 VALU per chunk
        """
        n = len(chunk_indices)
        n_tmps = len(v_tmps)
        sched = ListScheduler(vlen=VLEN)

        v_vals = [v_values + c * VLEN for c in chunk_indices]
        v_idxs = [v_indices + c * VLEN for c in chunk_indices]

        # === Selection: node_value = cache[0] + offset * diff ===
        # We compute into v_tmp2 (will be XOR'd immediately after)
        for i in range(n):
            v_tmp1, v_tmp2 = v_tmps[i % n_tmps]  # Reuse temps across chunks
            v_idx = v_idxs[i]

            sched.add_op("valu", ("-", v_tmp1, v_idx, v_start_idx))           # offset = idx - start_idx
            sched.add_op("valu", ("*", v_tmp2, v_tmp1, v_diff))               # tmp2 = offset * diff
            sched.add_op("valu", ("+", v_tmp1, v_cache_broadcast, v_tmp2))    # tmp1 = cache[0] + tmp2 (node_val)
            # XOR immediately
            sched.add_op("valu", ("^", v_vals[i], v_vals[i], v_tmp1))         # val ^= node_val

        # === Hash: 6 stages × 3 ops ===
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            for i in range(n):
                v_tmp1, v_tmp2 = v_tmps[i % n_tmps]
                sched.add_op("valu", (op1, v_tmp1, v_vals[i], v_c1))
                sched.add_op("valu", (op3, v_tmp2, v_vals[i], v_c3))
                sched.add_op("valu", (op2, v_vals[i], v_tmp1, v_tmp2))

        # === Index computation (using multiply_add) ===
        for i in range(n):
            v_tmp1, v_tmp2 = v_tmps[i % n_tmps]
            sched.add_op("valu", ("%", v_tmp1, v_vals[i], v_two))
            sched.add_op("valu", ("+", v_tmp2, v_tmp1, v_one))
            sched.add_op("valu", ("multiply_add", v_idxs[i], v_idxs[i], v_two, v_tmp2))
            if needs_wrap:
                sched.add_op("valu", ("<", v_tmp1, v_idxs[i], v_n_nodes))
                sched.add_op("flow", ("vselect", v_idxs[i], v_tmp1, v_idxs[i], v_zero))

        for bundle in sched.schedule():
            self.add_packed(bundle)

    def emit_select_unique2_arithmetic(self, chunk_indices, v_indices, v_node_vals,
                                          v_cache_broadcast, v_start_idx, v_diff, v_tmps, v_zero):
        """
        Arithmetic selection for unique_count=2 - replaces FLOW-heavy vselect with VALU.

        For unique_count=2, indices are either start_idx or start_idx+1.
        Formula: node_value = cache[0] + offset * (cache[1] - cache[0])
        where offset = idx - start_idx (0 or 1).

        This is ALL VALU operations, no FLOW needed!

        v_cache_broadcast: cache[0] at +0*VLEN, cache[1] at +1*VLEN
        v_diff: pre-computed (cache[1] - cache[0]) broadcast
        v_tmps: list of (v_tmp1, v_tmp2) pairs for each chunk
        """
        n = len(chunk_indices)
        sched = ListScheduler(vlen=VLEN)

        # For each chunk: offset = idx - start_idx, result = cache[0] + offset * diff
        for i in range(n):
            v_idx = v_indices + chunk_indices[i] * VLEN
            v_tmp1, v_tmp2 = v_tmps[i]
            v_result = v_node_vals[i]

            sched.add_op("valu", ("-", v_tmp1, v_idx, v_start_idx))           # offset = idx - start_idx
            sched.add_op("valu", ("*", v_tmp2, v_tmp1, v_diff))               # tmp2 = offset * diff
            sched.add_op("valu", ("+", v_result, v_cache_broadcast, v_tmp2))  # result = cache[0] + tmp2

        for bundle in sched.schedule():
            self.add_packed(bundle)

    def emit_last_group_with_next_gather(self,
                                          # Current round last group params
                                          curr_chunks, v_indices, v_values,
                                          curr_node_vals, curr_tmps, v_hash_consts,
                                          v_one, v_two, v_zero, v_n_nodes,
                                          curr_needs_wrap,
                                          # Next round first group params
                                          next_chunks, next_addrs, next_node_vals,
                                          forest_values_p):
        """
        Cross-round pipelining: overlap current round's last group compute with
        next round's first group gather.

        Key insight: After index computation updates v_indices, we can start
        computing addresses and loading for the next round while still in the
        same scheduler call.

        Operations:
        - Current: XOR + hash + index (all VALU)
        - Next: address computation (ALU) + loads (LOAD)

        The scheduler respects dependencies: address ALU reads v_indices after
        index computation writes to them.
        """
        n_curr = len(curr_chunks)
        n_next = len(next_chunks)
        sched = ListScheduler(vlen=VLEN)

        # Get addresses for current compute
        v_vals = [v_values + c * VLEN for c in curr_chunks]
        v_idxs = [v_indices + c * VLEN for c in curr_chunks]

        # === Current round: XOR ===
        for i in range(n_curr):
            sched.add_op("valu", ("^", v_vals[i], v_vals[i], curr_node_vals[i]))

        # === Current round: Hash ===
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            for i in range(n_curr):
                v_tmp1, v_tmp2 = curr_tmps[i]
                sched.add_op("valu", (op1, v_tmp1, v_vals[i], v_c1))
                sched.add_op("valu", (op3, v_tmp2, v_vals[i], v_c3))
                sched.add_op("valu", (op2, v_vals[i], v_tmp1, v_tmp2))

        # === Current round: Index computation ===
        for i in range(n_curr):
            v_tmp1, v_tmp2 = curr_tmps[i]
            sched.add_op("valu", ("%", v_tmp1, v_vals[i], v_two))
            sched.add_op("valu", ("+", v_tmp2, v_tmp1, v_one))
            sched.add_op("valu", ("multiply_add", v_idxs[i], v_idxs[i], v_two, v_tmp2))
            if curr_needs_wrap:
                sched.add_op("valu", ("<", v_tmp1, v_idxs[i], v_n_nodes))
                sched.add_op("flow", ("vselect", v_idxs[i], v_tmp1, v_idxs[i], v_zero))

        # === Next round first group: Address ALU ===
        # These read v_indices after index computation updates them
        for i in range(n_next):
            v_idx_i = v_indices + next_chunks[i] * VLEN
            addrs_i = next_addrs[i]
            for j in range(8):
                sched.add_op("alu", ("+", addrs_i + j, forest_values_p, v_idx_i + j))

        # === Next round first group: LOAD ===
        # These depend on address ALU completing
        for i in range(n_next):
            addrs_i = next_addrs[i]
            v_node_i = next_node_vals[i]
            for j in range(8):
                sched.add_op("load", ("load", v_node_i + j, addrs_i + j))

        for bundle in sched.schedule():
            self.add_packed(bundle)

    def emit_vselect_2way_interleaved(self, v_results, v_idxs, v_cache_broadcast,
                                        v_start_idx, unique_count, v_offsets, v_conds, v_zero, v_cmp_consts):
        """
        2-way interleaved vselect cascade - overlaps FLOW of chunk A with VALU of chunk B.

        This is the PRIORITY SCHEDULER approach:
        - FLOW operations (vselect) are the critical path
        - During FLOW cycles, we fill empty VALU slots with the other chunk's work

        v_results: [v_result_a, v_result_b]
        v_idxs: [v_idx_a, v_idx_b]
        v_offsets: [v_offset_a, v_offset_b]
        v_conds: [v_cond_a, v_cond_b]
        """
        v_result_a, v_result_b = v_results
        v_idx_a, v_idx_b = v_idxs
        v_offset_a, v_offset_b = v_offsets
        v_cond_a, v_cond_b = v_conds

        if unique_count == 1:
            # Both just copy the single pre-broadcast value (2 VALU ops, 1 cycle)
            self.add_packed({"valu": [
                ("+", v_result_a, v_cache_broadcast, v_zero),
                ("+", v_result_b, v_cache_broadcast, v_zero),
            ]})
            return

        # Phase 1: Compute offsets and initialize results for BOTH chunks (4 VALU ops, fits in 1 cycle)
        self.add_packed({"valu": [
            ("-", v_offset_a, v_idx_a, v_start_idx),
            ("-", v_offset_b, v_idx_b, v_start_idx),
            ("+", v_result_a, v_cache_broadcast, v_zero),
            ("+", v_result_b, v_cache_broadcast, v_zero),
        ]})

        # Phase 2: Interleaved cascade - overlap A's FLOW with B's VALU and vice versa
        # For each unique value u, we do:
        #   - A's compare (VALU)
        #   - A's vselect (FLOW) + B's compare (VALU) <- KEY: fill FLOW cycle with VALU!
        #   - B's vselect (FLOW) + A's next compare (VALU) if available

        for u in range(1, unique_count):
            # A's compare
            self.add_packed({"valu": [("==", v_cond_a, v_offset_a, v_cmp_consts + u * VLEN)]})

            # A's vselect (FLOW) + B's compare (VALU) - PRIORITY SCHEDULER: FLOW first, fill with VALU
            self.add_packed({
                "flow": [("vselect", v_result_a, v_cond_a, v_cache_broadcast + u * VLEN, v_result_a)],
                "valu": [("==", v_cond_b, v_offset_b, v_cmp_consts + u * VLEN)],  # Fill empty VALU slots!
            })

            # B's vselect (FLOW only for last u, else can overlap with next A's compare)
            self.add_packed({
                "flow": [("vselect", v_result_b, v_cond_b, v_cache_broadcast + u * VLEN, v_result_b)],
            })

    def emit_pipelined_gather_6way(self, chunk_indices, v_indices, gather_addrs_all,
                                     v_node_vals, forest_values_p, zero_const):
        """
        6-way pipelined gather: compute addresses for all 6 chunks first,
        then pipeline the loads with address computation for next group.

        This saves cycles by:
        1. Batching ALU address computation (12 ALU slots vs 8 ops needed per chunk)
        2. Overlapping LOADs (2/cycle bottleneck) with next iteration's ALU prep

        Returns after all 6 chunks have their node values loaded.
        """
        n = len(chunk_indices)

        # Phase 1: Compute all addresses (batch ALU operations)
        # 8 ops per chunk, 12 slots available = can do ~1.5 chunks per cycle
        for i in range(n):
            v_idx_i = v_indices + chunk_indices[i] * VLEN
            addrs_i = gather_addrs_all[i]

            # Copy indices to gather addresses (8 ALU ops)
            self.add_packed({"alu": [
                ("+", addrs_i + 0, v_idx_i + 0, zero_const),
                ("+", addrs_i + 1, v_idx_i + 1, zero_const),
                ("+", addrs_i + 2, v_idx_i + 2, zero_const),
                ("+", addrs_i + 3, v_idx_i + 3, zero_const),
                ("+", addrs_i + 4, v_idx_i + 4, zero_const),
                ("+", addrs_i + 5, v_idx_i + 5, zero_const),
                ("+", addrs_i + 6, v_idx_i + 6, zero_const),
                ("+", addrs_i + 7, v_idx_i + 7, zero_const),
            ]})

            # Add forest base to get actual addresses (8 ALU ops)
            self.add_packed({"alu": [
                ("+", addrs_i + 0, forest_values_p, addrs_i + 0),
                ("+", addrs_i + 1, forest_values_p, addrs_i + 1),
                ("+", addrs_i + 2, forest_values_p, addrs_i + 2),
                ("+", addrs_i + 3, forest_values_p, addrs_i + 3),
                ("+", addrs_i + 4, forest_values_p, addrs_i + 4),
                ("+", addrs_i + 5, forest_values_p, addrs_i + 5),
                ("+", addrs_i + 6, forest_values_p, addrs_i + 6),
                ("+", addrs_i + 7, forest_values_p, addrs_i + 7),
            ]})

        # Phase 2: Pipelined loads (2 loads per cycle)
        # We can interleave loads from different chunks to maximize throughput
        # Total: n * 4 cycles (8 loads per chunk, 2 per cycle)
        for load_pair in range(4):  # 4 pairs of 2 loads per chunk
            for i in range(n):
                addrs_i = gather_addrs_all[i]
                v_node_i = v_node_vals[i]
                self.add_packed({"load": [
                    ("load", v_node_i + load_pair * 2, addrs_i + load_pair * 2),
                    ("load", v_node_i + load_pair * 2 + 1, addrs_i + load_pair * 2 + 1),
                ]})

    def emit_pipelined_gather_and_xor_hash_6way(self, chunk_indices, v_indices, v_values,
                                                  gather_addrs_all, v_node_vals, v_tmps,
                                                  forest_values_p, zero_const,
                                                  v_hash_consts, v_one, v_two, v_zero, v_n_nodes):
        """
        Fully pipelined 6-way: overlap LOAD with VALU hash computation.

        Key insight: While loading chunks B-F, we can do hash on chunks that finished loading.

        Pipeline structure:
        - Compute all addresses first (ALU phase)
        - Start loading chunk A
        - While loading chunk B, do XOR+partial hash on A
        - While loading chunk C, continue hash on A, start hash on B
        - etc.
        """
        n = len(chunk_indices)
        v_vals = [v_values + c * VLEN for c in chunk_indices]
        v_idxs = [v_indices + c * VLEN for c in chunk_indices]

        # Phase 1: Compute ALL addresses for all 6 chunks (12 ALU cycles total)
        for i in range(n):
            v_idx_i = v_idxs[i]
            addrs_i = gather_addrs_all[i]

            self.add_packed({"alu": [
                ("+", addrs_i + 0, v_idx_i + 0, zero_const),
                ("+", addrs_i + 1, v_idx_i + 1, zero_const),
                ("+", addrs_i + 2, v_idx_i + 2, zero_const),
                ("+", addrs_i + 3, v_idx_i + 3, zero_const),
                ("+", addrs_i + 4, v_idx_i + 4, zero_const),
                ("+", addrs_i + 5, v_idx_i + 5, zero_const),
                ("+", addrs_i + 6, v_idx_i + 6, zero_const),
                ("+", addrs_i + 7, v_idx_i + 7, zero_const),
            ]})
            self.add_packed({"alu": [
                ("+", addrs_i + 0, forest_values_p, addrs_i + 0),
                ("+", addrs_i + 1, forest_values_p, addrs_i + 1),
                ("+", addrs_i + 2, forest_values_p, addrs_i + 2),
                ("+", addrs_i + 3, forest_values_p, addrs_i + 3),
                ("+", addrs_i + 4, forest_values_p, addrs_i + 4),
                ("+", addrs_i + 5, forest_values_p, addrs_i + 5),
                ("+", addrs_i + 6, forest_values_p, addrs_i + 6),
                ("+", addrs_i + 7, forest_values_p, addrs_i + 7),
            ]})

        # Phase 2: Load chunk A (4 cycles)
        addrs_a = gather_addrs_all[0]
        v_node_a = v_node_vals[0]
        for lp in range(4):
            self.add_packed({"load": [
                ("load", v_node_a + lp * 2, addrs_a + lp * 2),
                ("load", v_node_a + lp * 2 + 1, addrs_a + lp * 2 + 1),
            ]})

        # Phase 3: For each remaining chunk, load it while doing XOR+hash on previous
        # Use ListScheduler to overlap LOAD with VALU optimally
        for i in range(1, n):
            addrs_i = gather_addrs_all[i]
            v_node_i = v_node_vals[i]
            prev_idx = i - 1
            v_val_prev = v_vals[prev_idx]
            v_node_prev = v_node_vals[prev_idx]
            v_tmp1_prev, v_tmp2_prev = v_tmps[prev_idx]

            # XOR for previous chunk (VALU) + Load pair 0 for current (LOAD)
            self.add_packed({
                "valu": [("^", v_val_prev, v_val_prev, v_node_prev)],
                "load": [
                    ("load", v_node_i + 0, addrs_i + 0),
                    ("load", v_node_i + 1, addrs_i + 1),
                ],
            })

            # Hash stages 0-1 interleaved with loads 1-2
            for hi in range(2):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                v_c1, v_c3 = v_hash_consts[hi]
                self.add_packed({
                    "valu": [
                        (op1, v_tmp1_prev, v_val_prev, v_c1),
                        (op3, v_tmp2_prev, v_val_prev, v_c3),
                    ],
                    "load": [
                        ("load", v_node_i + (hi + 1) * 2, addrs_i + (hi + 1) * 2),
                        ("load", v_node_i + (hi + 1) * 2 + 1, addrs_i + (hi + 1) * 2 + 1),
                    ],
                })
                self.add_packed({"valu": [(op2, v_val_prev, v_tmp1_prev, v_tmp2_prev)]})

            # Load pair 3 + hash stage 2
            op1, val1, op2, op3, val3 = HASH_STAGES[2]
            v_c1, v_c3 = v_hash_consts[2]
            self.add_packed({
                "valu": [
                    (op1, v_tmp1_prev, v_val_prev, v_c1),
                    (op3, v_tmp2_prev, v_val_prev, v_c3),
                ],
                "load": [
                    ("load", v_node_i + 6, addrs_i + 6),
                    ("load", v_node_i + 7, addrs_i + 7),
                ],
            })
            self.add_packed({"valu": [(op2, v_val_prev, v_tmp1_prev, v_tmp2_prev)]})

            # Remaining hash stages 3-5 (no more loads for this chunk)
            for hi in range(3, 6):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                v_c1, v_c3 = v_hash_consts[hi]
                self.add_packed({"valu": [
                    (op1, v_tmp1_prev, v_val_prev, v_c1),
                    (op3, v_tmp2_prev, v_val_prev, v_c3),
                ]})
                self.add_packed({"valu": [(op2, v_val_prev, v_tmp1_prev, v_tmp2_prev)]})

        # Phase 4: XOR + Hash for last chunk (no overlapping load)
        last_idx = n - 1
        v_val_last = v_vals[last_idx]
        v_node_last = v_node_vals[last_idx]
        v_tmp1_last, v_tmp2_last = v_tmps[last_idx]

        self.add_packed({"valu": [("^", v_val_last, v_val_last, v_node_last)]})
        for hi in range(6):
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            v_c1, v_c3 = v_hash_consts[hi]
            self.add_packed({"valu": [
                (op1, v_tmp1_last, v_val_last, v_c1),
                (op3, v_tmp2_last, v_val_last, v_c3),
            ]})
            self.add_packed({"valu": [(op2, v_val_last, v_tmp1_last, v_tmp2_last)]})

        # Phase 5: Index computation for all chunks using ListScheduler
        self.emit_xor_hash_index_nway_scheduled(
            chunk_indices, v_indices, v_values,
            # Note: we pass dummy node_vals since XOR+hash already done
            # We need to skip XOR+hash in the scheduled function
            v_node_vals, v_tmps,
            v_hash_consts, v_one, v_two, v_zero, v_n_nodes,
            skip_xor_hash=True  # New parameter!
        )

    def emit_gather_batched(self, chunk_indices, v_indices, gather_addrs_list,
                            v_node_vals, forest_values_p, zero_const):
        """
        Emit gather operations with optimized ALU: direct add instead of copy+add.

        Key insight: Instead of copy + add base, directly compute addr = base + idx.
        This halves the ALU operations!
        """
        group_size = len(chunk_indices)

        # Phase 1: Compute addresses directly (base + idx)
        # 8 ops per chunk, 12 slots = can do 1.5 chunks per cycle
        add_ops = []
        for i in range(group_size):
            v_idx_i = v_indices + chunk_indices[i] * VLEN
            addrs_i = gather_addrs_list[i]
            for j in range(8):
                add_ops.append(("+", addrs_i + j, forest_values_p, v_idx_i + j))

        # Emit first batch of ALU ops (need at least chunk 0 done before loading)
        self.add_packed({"alu": add_ops[0:12]})  # Does chunks 0 + partial 1

        # Phase 2: Interleave remaining ALU with loads
        add_idx = 12
        load_chunk = 0
        load_pair = 0

        while load_chunk < group_size:
            bundle = {}

            # Add 2 loads per cycle
            addrs_i = gather_addrs_list[load_chunk]
            v_node_i = v_node_vals[load_chunk]
            bundle["load"] = [
                ("load", v_node_i + load_pair * 2, addrs_i + load_pair * 2),
                ("load", v_node_i + load_pair * 2 + 1, addrs_i + load_pair * 2 + 1),
            ]

            # Add up to 12 ALU ops in parallel if available
            if add_idx < len(add_ops):
                batch_end = min(add_idx + 12, len(add_ops))
                bundle["alu"] = add_ops[add_idx:batch_end]
                add_idx = batch_end

            self.add_packed(bundle)

            # Advance load position
            load_pair += 1
            if load_pair >= 4:
                load_pair = 0
                load_chunk += 1

    def emit_gather_alu_only(self, chunk_indices, v_indices, gather_addrs_list,
                             forest_values_p):
        """Emit only the ALU portion of gather (address computation)."""
        group_size = len(chunk_indices)

        add_ops = []
        for i in range(group_size):
            v_idx_i = v_indices + chunk_indices[i] * VLEN
            addrs_i = gather_addrs_list[i]
            for j in range(8):
                add_ops.append(("+", addrs_i + j, forest_values_p, v_idx_i + j))

        for batch_start in range(0, len(add_ops), 12):
            batch = add_ops[batch_start:batch_start + 12]
            self.add_packed({"alu": batch})

    def emit_gather_load_only(self, group_size, gather_addrs_list, v_node_vals):
        """Emit only the LOAD portion of gather."""
        for i in range(group_size):
            addrs_i = gather_addrs_list[i]
            v_node_i = v_node_vals[i]
            self.add_packed({"load": [("load", v_node_i + 0, addrs_i + 0), ("load", v_node_i + 1, addrs_i + 1)]})
            self.add_packed({"load": [("load", v_node_i + 2, addrs_i + 2), ("load", v_node_i + 3, addrs_i + 3)]})
            self.add_packed({"load": [("load", v_node_i + 4, addrs_i + 4), ("load", v_node_i + 5, addrs_i + 5)]})
            self.add_packed({"load": [("load", v_node_i + 6, addrs_i + 6), ("load", v_node_i + 7, addrs_i + 7)]})

    def emit_compute_with_prefetch(self, compute_chunks, compute_v_indices, compute_v_values,
                                    compute_node_vals, compute_tmps, v_hash_consts,
                                    v_one, v_two, v_zero, v_n_nodes,
                                    prefetch_chunks, prefetch_addrs, prefetch_node_vals,
                                    needs_wrap=True, include_prefetch_alu=False,
                                    forest_values_p=None):
        """
        Overlap current group's XOR+Hash+Index (VALU) with next group's LOAD.

        The ListScheduler will interleave LOAD and VALU operations since they're
        independent engines.

        needs_wrap: if False, skip the wrap vselect (idx >= n_nodes check)
        include_prefetch_alu: if True, also compute prefetch addresses (requires forest_values_p)
        """
        n_compute = len(compute_chunks)
        n_prefetch = len(prefetch_chunks) if prefetch_chunks else 0

        sched = ListScheduler(vlen=VLEN)

        # Get addresses for compute
        v_vals = [compute_v_values + c * VLEN for c in compute_chunks]
        v_idxs = [compute_v_indices + c * VLEN for c in compute_chunks]

        # === Optionally add ALU for prefetch addresses ===
        if include_prefetch_alu and n_prefetch > 0:
            for i in range(n_prefetch):
                v_idx_i = compute_v_indices + prefetch_chunks[i] * VLEN
                addrs_i = prefetch_addrs[i]
                for j in range(8):
                    sched.add_op("alu", ("+", addrs_i + j, forest_values_p, v_idx_i + j))

        # === Add LOAD operations for prefetch (next group) ===
        for i in range(n_prefetch):
            addrs_i = prefetch_addrs[i]
            v_node_i = prefetch_node_vals[i]
            for j in range(8):
                sched.add_op("load", ("load", v_node_i + j, addrs_i + j))

        # === Add XOR operations ===
        for i in range(n_compute):
            sched.add_op("valu", ("^", v_vals[i], v_vals[i], compute_node_vals[i]))

        # === Add Hash operations ===
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            for i in range(n_compute):
                v_tmp1, v_tmp2 = compute_tmps[i]
                sched.add_op("valu", (op1, v_tmp1, v_vals[i], v_c1))
                sched.add_op("valu", (op3, v_tmp2, v_vals[i], v_c3))
                sched.add_op("valu", (op2, v_vals[i], v_tmp1, v_tmp2))

        # === Add Index computation (using multiply_add) ===
        for i in range(n_compute):
            v_tmp1, v_tmp2 = compute_tmps[i]
            sched.add_op("valu", ("%", v_tmp1, v_vals[i], v_two))
            sched.add_op("valu", ("+", v_tmp2, v_tmp1, v_one))
            sched.add_op("valu", ("multiply_add", v_idxs[i], v_idxs[i], v_two, v_tmp2))
            if needs_wrap:
                sched.add_op("valu", ("<", v_tmp1, v_idxs[i], v_n_nodes))
                sched.add_op("flow", ("vselect", v_idxs[i], v_tmp1, v_idxs[i], v_zero))

        # Schedule and emit
        for bundle in sched.schedule():
            self.add_packed(bundle)

    def emit_vselect_all_chunks_batched(self, n_vectors, v_indices, v_cache_broadcast, v_start_idx,
                                         unique_count, v_offsets, v_results, v_compares, v_zero, v_cmp_consts, VLEN):
        """
        Batched vselect for ALL chunks at once.

        Key insight: FLOW operations must be sequential (1 per cycle), but VALU can be batched.
        We overlap FLOW cycles with VALU work for next iteration.

        v_offsets: scratch space for 32 offset vectors (256 words)
        v_results: scratch space for 32 result vectors (256 words)
        v_compares: scratch space for 32 compare result vectors (256 words)
        """
        if unique_count == 1:
            # Just initialize all results with the single cache value
            for batch_start in range(0, n_vectors, 6):
                batch_ops = []
                for c in range(batch_start, min(batch_start + 6, n_vectors)):
                    batch_ops.append(("+", v_results + c * VLEN, v_cache_broadcast, v_zero))
                self.add_packed({"valu": batch_ops})
            return

        # Phase 1: Batch compute all offsets (32 ops / 6 = 6 cycles)
        for batch_start in range(0, n_vectors, 6):
            batch_ops = []
            for c in range(batch_start, min(batch_start + 6, n_vectors)):
                batch_ops.append(("-", v_offsets + c * VLEN, v_indices + c * VLEN, v_start_idx))
            self.add_packed({"valu": batch_ops})

        # Phase 2: Batch initialize all results (32 ops / 6 = 6 cycles)
        for batch_start in range(0, n_vectors, 6):
            batch_ops = []
            for c in range(batch_start, min(batch_start + 6, n_vectors)):
                batch_ops.append(("+", v_results + c * VLEN, v_cache_broadcast, v_zero))
            self.add_packed({"valu": batch_ops})

        # Phase 3: For each unique value, batch compares then do sequential vselects
        # Overlap FLOW with next iteration's VALU where possible
        for u in range(1, unique_count):
            # Batch compute all compares (32 ops / 6 = 6 cycles)
            for batch_start in range(0, n_vectors, 6):
                batch_ops = []
                for c in range(batch_start, min(batch_start + 6, n_vectors)):
                    batch_ops.append(("==", v_compares + c * VLEN, v_offsets + c * VLEN, v_cmp_consts + u * VLEN))
                self.add_packed({"valu": batch_ops})

            # Sequential vselects for all chunks (32 FLOW cycles)
            for c in range(n_vectors):
                self.add_packed({
                    "flow": [("vselect", v_results + c * VLEN, v_compares + c * VLEN,
                              v_cache_broadcast + u * VLEN, v_results + c * VLEN)],
                })

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized vectorized kernel using dedup/broadcast strategy.

        Key optimization: Instead of loading 256 tree nodes per round,
        we cache unique node values and use vselect to distribute them.

        - Type A rounds (R0-R5, R11-R15): Known small unique counts (1,2,4,8,16,32)
        - Type B rounds (R6-R10): Dynamic deduplication
        """
        # ===== SCRATCH SPACE ALLOCATION =====

        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Init vars from memory header
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Vector registers for indices and values (256 elements = 32 vectors of 8)
        n_vectors = batch_size // VLEN  # 32
        v_indices = self.alloc_scratch("v_indices", batch_size)
        v_values = self.alloc_scratch("v_values", batch_size)

        # Pre-broadcast cache for unique node values (max 32 unique × VLEN = 256 words)
        # This allows us to broadcast once per round instead of per chunk
        v_cache_broadcast = self.alloc_scratch("v_cache_broadcast", 32 * VLEN)

        # Temporary for start_idx broadcast (used in offset computation)
        v_start_idx = self.alloc_scratch("v_start_idx", VLEN)

        # Scalar cache for loading node values before broadcasting
        node_cache_scalar = self.alloc_scratch("node_cache_scalar", 32)

        # Vector temporaries
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_cond = self.alloc_scratch("v_cond", VLEN)

        # Extra temporaries for 2-way interleaved processing
        v_node_val_b = self.alloc_scratch("v_node_val_b", VLEN)
        v_offset_a = self.alloc_scratch("v_offset_a", VLEN)
        v_offset_b = self.alloc_scratch("v_offset_b", VLEN)
        v_cond_a = self.alloc_scratch("v_cond_a", VLEN)
        v_cond_b = self.alloc_scratch("v_cond_b", VLEN)
        # Extra temporaries for 2-way index computation (need separate tmp1/tmp2 for A and B)
        v_tmp1_b = self.alloc_scratch("v_tmp1_b", VLEN)
        v_tmp2_b = self.alloc_scratch("v_tmp2_b", VLEN)

        # Extra temporaries for 3-way processing (chunk C)
        v_node_val_c = self.alloc_scratch("v_node_val_c", VLEN)
        v_tmp1_c = self.alloc_scratch("v_tmp1_c", VLEN)
        v_tmp2_c = self.alloc_scratch("v_tmp2_c", VLEN)

        # Extra temporaries for 6-way processing (chunks D, E, F)
        v_node_val_d = self.alloc_scratch("v_node_val_d", VLEN)
        v_tmp1_d = self.alloc_scratch("v_tmp1_d", VLEN)
        v_tmp2_d = self.alloc_scratch("v_tmp2_d", VLEN)
        v_node_val_e = self.alloc_scratch("v_node_val_e", VLEN)
        v_tmp1_e = self.alloc_scratch("v_tmp1_e", VLEN)
        v_tmp2_e = self.alloc_scratch("v_tmp2_e", VLEN)
        v_node_val_f = self.alloc_scratch("v_node_val_f", VLEN)
        v_tmp1_f = self.alloc_scratch("v_tmp1_f", VLEN)
        v_tmp2_f = self.alloc_scratch("v_tmp2_f", VLEN)

        # Collect all node_val and tmp registers for N-way processing (Group A)
        v_node_vals_all = [v_node_val, v_node_val_b, v_node_val_c, v_node_val_d, v_node_val_e, v_node_val_f]
        v_tmps_all = [(v_tmp1, v_tmp2), (v_tmp1_b, v_tmp2_b), (v_tmp1_c, v_tmp2_c),
                      (v_tmp1_d, v_tmp2_d), (v_tmp1_e, v_tmp2_e), (v_tmp1_f, v_tmp2_f)]

        # Group B registers for software pipelining (second set)
        v_node_val_g = self.alloc_scratch("v_node_val_g", VLEN)
        v_tmp1_g = self.alloc_scratch("v_tmp1_g", VLEN)
        v_tmp2_g = self.alloc_scratch("v_tmp2_g", VLEN)
        v_node_val_h = self.alloc_scratch("v_node_val_h", VLEN)
        v_tmp1_h = self.alloc_scratch("v_tmp1_h", VLEN)
        v_tmp2_h = self.alloc_scratch("v_tmp2_h", VLEN)
        v_node_val_i = self.alloc_scratch("v_node_val_i", VLEN)
        v_tmp1_i = self.alloc_scratch("v_tmp1_i", VLEN)
        v_tmp2_i = self.alloc_scratch("v_tmp2_i", VLEN)
        v_node_val_j = self.alloc_scratch("v_node_val_j", VLEN)
        v_tmp1_j = self.alloc_scratch("v_tmp1_j", VLEN)
        v_tmp2_j = self.alloc_scratch("v_tmp2_j", VLEN)
        v_node_val_k = self.alloc_scratch("v_node_val_k", VLEN)
        v_tmp1_k = self.alloc_scratch("v_tmp1_k", VLEN)
        v_tmp2_k = self.alloc_scratch("v_tmp2_k", VLEN)
        v_node_val_l = self.alloc_scratch("v_node_val_l", VLEN)
        v_tmp1_l = self.alloc_scratch("v_tmp1_l", VLEN)
        v_tmp2_l = self.alloc_scratch("v_tmp2_l", VLEN)

        v_node_vals_grpB = [v_node_val_g, v_node_val_h, v_node_val_i, v_node_val_j, v_node_val_k, v_node_val_l]
        v_tmps_grpB = [(v_tmp1_g, v_tmp2_g), (v_tmp1_h, v_tmp2_h), (v_tmp1_i, v_tmp2_i),
                       (v_tmp1_j, v_tmp2_j), (v_tmp1_k, v_tmp2_k), (v_tmp1_l, v_tmp2_l)]

        # Temporary addresses for batched gather - Group A (8 per chunk)
        gather_addrs = self.alloc_scratch("gather_addrs", 8)
        gather_addrs_b = self.alloc_scratch("gather_addrs_b", 8)
        gather_addrs_c = self.alloc_scratch("gather_addrs_c", 8)
        gather_addrs_d = self.alloc_scratch("gather_addrs_d", 8)
        gather_addrs_e = self.alloc_scratch("gather_addrs_e", 8)
        gather_addrs_f = self.alloc_scratch("gather_addrs_f", 8)
        gather_addrs_all = [gather_addrs, gather_addrs_b, gather_addrs_c,
                           gather_addrs_d, gather_addrs_e, gather_addrs_f]

        # Temporary addresses for gather - Group B (for pipelining)
        gather_addrs_g = self.alloc_scratch("gather_addrs_g", 8)
        gather_addrs_h = self.alloc_scratch("gather_addrs_h", 8)
        gather_addrs_i = self.alloc_scratch("gather_addrs_i", 8)
        gather_addrs_j = self.alloc_scratch("gather_addrs_j", 8)
        gather_addrs_k = self.alloc_scratch("gather_addrs_k", 8)
        gather_addrs_l = self.alloc_scratch("gather_addrs_l", 8)
        gather_addrs_grpB = [gather_addrs_g, gather_addrs_h, gather_addrs_i,
                            gather_addrs_j, gather_addrs_k, gather_addrs_l]


        # Broadcast vectors for constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # For unique_count=2 arithmetic: v_diff = cache[1] - cache[0]
        v_diff = self.alloc_scratch("v_diff", VLEN)

        # Vector constants for hash stages (6 stages × 2 constants each)
        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)
            v_hash_consts.append((v_c1, v_c3))

        # Pre-broadcast comparison constants 0-3 for vselect cascade (only need 4 since unique_count <= 4)
        v_cmp_consts = self.alloc_scratch("v_cmp_consts", 4 * VLEN)

        # Broadcast constants to vectors (batched: 4 ops in 1 cycle)
        self.add_packed({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ]})

        # Pre-create scalar constants for hash
        hash_scalar_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            hash_scalar_consts.append((c1, c3))

        # Broadcast hash constants to vectors (batched: 12 ops in 2 cycles)
        self.add_packed({"valu": [
            ("vbroadcast", v_hash_consts[0][0], hash_scalar_consts[0][0]),
            ("vbroadcast", v_hash_consts[0][1], hash_scalar_consts[0][1]),
            ("vbroadcast", v_hash_consts[1][0], hash_scalar_consts[1][0]),
            ("vbroadcast", v_hash_consts[1][1], hash_scalar_consts[1][1]),
            ("vbroadcast", v_hash_consts[2][0], hash_scalar_consts[2][0]),
            ("vbroadcast", v_hash_consts[2][1], hash_scalar_consts[2][1]),
        ]})
        self.add_packed({"valu": [
            ("vbroadcast", v_hash_consts[3][0], hash_scalar_consts[3][0]),
            ("vbroadcast", v_hash_consts[3][1], hash_scalar_consts[3][1]),
            ("vbroadcast", v_hash_consts[4][0], hash_scalar_consts[4][0]),
            ("vbroadcast", v_hash_consts[4][1], hash_scalar_consts[4][1]),
            ("vbroadcast", v_hash_consts[5][0], hash_scalar_consts[5][0]),
            ("vbroadcast", v_hash_consts[5][1], hash_scalar_consts[5][1]),
        ]})

        # Pre-create scalar constants for comparison (0-3)
        cmp_scalar_consts = [self.scratch_const(i) for i in range(4)]

        # Broadcast comparison constants 0-3 (4 ops in 1 cycle)
        self.add_packed({"valu": [
            ("vbroadcast", v_cmp_consts + 0 * VLEN, cmp_scalar_consts[0]),
            ("vbroadcast", v_cmp_consts + 1 * VLEN, cmp_scalar_consts[1]),
            ("vbroadcast", v_cmp_consts + 2 * VLEN, cmp_scalar_consts[2]),
            ("vbroadcast", v_cmp_consts + 3 * VLEN, cmp_scalar_consts[3]),
        ]})

        # ===== INITIALIZATION: Load all indices/values into scratch (4-way batched) =====
        # Using 4 temps (tmp1, tmp2, tmp3, tmp_addr) for better throughput
        for chunk in range(0, n_vectors, 4):
            off1 = chunk * VLEN
            off2 = (chunk + 1) * VLEN
            off3 = (chunk + 2) * VLEN
            off4 = (chunk + 3) * VLEN

            # Load indices for 4 chunks
            self.add_packed({"load": [("const", tmp1, off1), ("const", tmp2, off2)]})
            self.add_packed({"load": [("const", tmp3, off3), ("const", tmp_addr, off4)]})
            self.add_packed({"alu": [
                ("+", tmp1, self.scratch["inp_indices_p"], tmp1),
                ("+", tmp2, self.scratch["inp_indices_p"], tmp2),
                ("+", tmp3, self.scratch["inp_indices_p"], tmp3),
                ("+", tmp_addr, self.scratch["inp_indices_p"], tmp_addr),
            ]})
            self.add_packed({"load": [
                ("vload", v_indices + off1, tmp1),
                ("vload", v_indices + off2, tmp2),
            ]})
            self.add_packed({"load": [
                ("vload", v_indices + off3, tmp3),
                ("vload", v_indices + off4, tmp_addr),
            ]})

            # Load values for 4 chunks
            self.add_packed({"load": [("const", tmp1, off1), ("const", tmp2, off2)]})
            self.add_packed({"load": [("const", tmp3, off3), ("const", tmp_addr, off4)]})
            self.add_packed({"alu": [
                ("+", tmp1, self.scratch["inp_values_p"], tmp1),
                ("+", tmp2, self.scratch["inp_values_p"], tmp2),
                ("+", tmp3, self.scratch["inp_values_p"], tmp3),
                ("+", tmp_addr, self.scratch["inp_values_p"], tmp_addr),
            ]})
            self.add_packed({"load": [
                ("vload", v_values + off1, tmp1),
                ("vload", v_values + off2, tmp2),
            ]})
            self.add_packed({"load": [
                ("vload", v_values + off3, tmp3),
                ("vload", v_values + off4, tmp_addr),
            ]})

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting optimized loop"))

        # Cross-round pipelining state: tracks if first group was pre-loaded by previous round
        # first_group_preloaded: (node_vals_list, addrs_list) or None
        first_group_preloaded = None

        # ===== MAIN LOOP: Process each round =====
        for rnd in range(rounds):
            # Determine gather_depth = rnd % (forest_height + 1)
            gather_depth = rnd % (forest_height + 1)

            if gather_depth <= 5:
                # Type A: Known unique count = 2^gather_depth
                unique_count = 1 << gather_depth  # 1, 2, 4, 8, 16, 32

                # For unique_count <= 2: use vselect/broadcast
                # For unique_count > 2: use direct load (faster due to FLOW bottleneck)
                use_vselect = (unique_count <= 2)

                if use_vselect:
                    # Calculate starting index for this depth level
                    start_idx = (1 << gather_depth) - 1

                    # Load unique node values (batched: 2 loads per cycle)
                    for u in range(0, unique_count, 2):
                        if u + 1 < unique_count:
                            self.add_packed({"load": [
                                ("const", tmp1, start_idx + u),
                                ("const", tmp2, start_idx + u + 1),
                            ]})
                            self.add_packed({"alu": [
                                ("+", tmp1, self.scratch["forest_values_p"], tmp1),
                                ("+", tmp2, self.scratch["forest_values_p"], tmp2),
                            ]})
                            self.add_packed({"load": [
                                ("load", node_cache_scalar + u, tmp1),
                                ("load", node_cache_scalar + u + 1, tmp2),
                            ]})
                        else:
                            self.add_packed({"load": [("const", tmp1, start_idx + u)]})
                            self.add_packed({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
                            self.add_packed({"load": [("load", node_cache_scalar + u, tmp1)]})

                    # Pre-broadcast all cache values (batched: 6 per cycle)
                    for batch_start in range(0, unique_count, 6):
                        batch_ops = []
                        for u in range(batch_start, min(batch_start + 6, unique_count)):
                            batch_ops.append(("vbroadcast", v_cache_broadcast + u * VLEN, node_cache_scalar + u))
                        if batch_start == 0 and len(batch_ops) < 6:
                            start_const = self.scratch_const(start_idx)
                            batch_ops.append(("vbroadcast", v_start_idx, start_const))
                        self.add_packed({"valu": batch_ops})

                    if unique_count >= 6:
                        start_const = self.scratch_const(start_idx)
                        self.add_packed({"valu": [("vbroadcast", v_start_idx, start_const)]})

                if use_vselect:
                    needs_wrap = (gather_depth >= forest_height)

                    if unique_count == 1:
                        # Optimized path: process ALL 32 chunks in one scheduler call
                        # XOR directly with the single broadcast value (no per-chunk copy needed)
                        all_chunks = list(range(n_vectors))
                        self.emit_xor_hash_index_broadcast(
                            all_chunks, v_indices, v_values,
                            v_cache_broadcast, v_tmps_all + v_tmps_grpB,  # Reuse all temps
                            v_hash_consts, v_one, v_two, v_zero, v_n_nodes,
                            needs_wrap=needs_wrap
                        )
                    elif unique_count == 2:
                        # Precompute v_diff = cache[1] - cache[0]
                        self.add_packed({"valu": [("-", v_diff, v_cache_broadcast + VLEN, v_cache_broadcast)]})

                        # Combine GroupA and GroupB temp registers (12 pairs, reused via modulo)
                        v_tmps_12way = v_tmps_all + v_tmps_grpB

                        # Process ALL 32 chunks at once - temps are reused via modulo
                        all_chunks = list(range(n_vectors))
                        self.emit_fused_unique2_complete(
                            all_chunks, v_indices, v_values,
                            v_cache_broadcast, v_start_idx, v_diff,
                            v_tmps_12way,
                            v_hash_consts, v_one, v_two, v_zero, v_n_nodes,
                            needs_wrap=needs_wrap
                        )
                    else:
                        # Vselect cascade for unique_count > 2
                        v_node_vals_12way = v_node_vals_all + v_node_vals_grpB
                        v_tmps_12way = v_tmps_all + v_tmps_grpB

                        for group_start in range(0, n_vectors, 12):
                            group_size = min(12, n_vectors - group_start)
                            chunk_indices = list(range(group_start, group_start + group_size))

                            for i in range(0, group_size, 2):
                                if i + 1 < group_size:
                                    v_idx_a = v_indices + chunk_indices[i] * VLEN
                                    v_idx_b = v_indices + chunk_indices[i+1] * VLEN
                                    self.emit_vselect_2way_interleaved(
                                        [v_node_vals_12way[i], v_node_vals_12way[i+1]],
                                        [v_idx_a, v_idx_b],
                                        v_cache_broadcast,
                                        v_start_idx, unique_count,
                                        [v_offset_a, v_offset_b],
                                        [v_cond_a, v_cond_b],
                                        v_zero, v_cmp_consts
                                    )
                                else:
                                    v_idx_a = v_indices + chunk_indices[i] * VLEN
                                    self.emit_vselect_from_prebroadcast(
                                        v_node_vals_12way[i], v_idx_a, v_cache_broadcast,
                                        v_start_idx, unique_count, [v_tmp1, v_tmp2],
                                        v_cond, v_zero, v_cmp_consts
                                    )

                            # XOR + hash + index
                            self.emit_xor_hash_index_nway_scheduled(
                                chunk_indices, v_indices, v_values,
                                v_node_vals_12way[:group_size], v_tmps_12way[:group_size],
                                v_hash_consts, v_one, v_two, v_zero, v_n_nodes,
                                needs_wrap=needs_wrap
                            )
                else:
                    # DIRECT LOAD with pipelining (same as Type B)
                    groups = []
                    for group_start in range(0, n_vectors, 6):
                        group_size = min(6, n_vectors - group_start)
                        chunk_indices = list(range(group_start, group_start + group_size))
                        groups.append((chunk_indices, group_size))

                    # Check if next round uses direct load (for cross-round pipelining)
                    next_gather_depth = (rnd + 1) % (forest_height + 1) if rnd + 1 < rounds else -1
                    next_is_direct = next_gather_depth > 5 or (next_gather_depth >= 0 and (1 << next_gather_depth) > 2)

                    # First group: skip if pre-loaded, otherwise gather
                    chunk_indices, group_size = groups[0]
                    if first_group_preloaded is not None:
                        pass  # Already loaded by previous round
                    else:
                        self.emit_gather_batched(
                            chunk_indices, v_indices, gather_addrs_all[:group_size],
                            v_node_vals_all[:group_size], self.scratch["forest_values_p"], zero_const
                        )

                    first_group_preloaded = None  # Consumed

                    # Middle groups: compute current while loading next
                    for g_idx in range(len(groups) - 1):
                        curr_chunks, curr_size = groups[g_idx]
                        next_chunks, next_size = groups[g_idx + 1]

                        if g_idx % 2 == 0:
                            curr_nodes = v_node_vals_all[:curr_size]
                            curr_tmps = v_tmps_all[:curr_size]
                            next_nodes = v_node_vals_grpB[:next_size]
                            next_addrs = gather_addrs_grpB[:next_size]
                        else:
                            curr_nodes = v_node_vals_grpB[:curr_size]
                            curr_tmps = v_tmps_grpB[:curr_size]
                            next_nodes = v_node_vals_all[:next_size]
                            next_addrs = gather_addrs_all[:next_size]

                        # Wrap only needed when gather_depth >= forest_height
                        needs_wrap = (gather_depth >= forest_height)
                        # Combined ALU + compute + prefetch in one scheduler
                        self.emit_compute_with_prefetch(
                            curr_chunks, v_indices, v_values,
                            curr_nodes, curr_tmps, v_hash_consts,
                            v_one, v_two, v_zero, v_n_nodes,
                            next_chunks, next_addrs, next_nodes,
                            needs_wrap=needs_wrap,
                            include_prefetch_alu=True,
                            forest_values_p=self.scratch["forest_values_p"]
                        )

                    # Last group: compute, optionally with cross-round prefetch
                    last_chunks, last_size = groups[-1]
                    if (len(groups) - 1) % 2 == 0:
                        last_nodes = v_node_vals_all[:last_size]
                        last_tmps = v_tmps_all[:last_size]
                    else:
                        last_nodes = v_node_vals_grpB[:last_size]
                        last_tmps = v_tmps_grpB[:last_size]

                    needs_wrap = (gather_depth >= forest_height)

                    if next_is_direct:
                        # Cross-round pipelining
                        next_first_chunks = groups[0][0]
                        next_first_size = groups[0][1]
                        self.emit_last_group_with_next_gather(
                            last_chunks, v_indices, v_values,
                            last_nodes, last_tmps, v_hash_consts,
                            v_one, v_two, v_zero, v_n_nodes,
                            needs_wrap,
                            next_first_chunks, gather_addrs_all[:next_first_size],
                            v_node_vals_all[:next_first_size],
                            self.scratch["forest_values_p"]
                        )
                        first_group_preloaded = True
                    else:
                        self.emit_xor_hash_index_nway_scheduled(
                            last_chunks, v_indices, v_values,
                            last_nodes, last_tmps,
                            v_hash_consts, v_one, v_two, v_zero, v_n_nodes,
                            needs_wrap=needs_wrap
                        )
            else:
                # Type B: High diversity rounds (R6-R10)
                # Use pipelining: overlap group N's LOAD with group N-1's compute

                groups = []
                for group_start in range(0, n_vectors, 6):
                    group_size = min(6, n_vectors - group_start)
                    chunk_indices = list(range(group_start, group_start + group_size))
                    groups.append((chunk_indices, group_size))

                # Check if next round is also Type B (for cross-round pipelining)
                next_gather_depth = (rnd + 1) % (forest_height + 1) if rnd + 1 < rounds else -1
                next_is_type_b = next_gather_depth > 5

                # First group: skip if pre-loaded by previous round, otherwise gather
                chunk_indices, group_size = groups[0]
                if first_group_preloaded is not None:
                    # First group already loaded by previous round's last group
                    # Just use the pre-loaded data (in GroupA)
                    pass
                else:
                    self.emit_gather_batched(
                        chunk_indices, v_indices, gather_addrs_all[:group_size],
                        v_node_vals_all[:group_size], self.scratch["forest_values_p"], zero_const
                    )

                # Reset preloaded state (consumed)
                first_group_preloaded = None

                # Middle groups: compute current while loading next
                for g_idx in range(len(groups) - 1):
                    curr_chunks, curr_size = groups[g_idx]
                    next_chunks, next_size = groups[g_idx + 1]

                    # Alternate register sets
                    if g_idx % 2 == 0:
                        curr_nodes = v_node_vals_all[:curr_size]
                        curr_tmps = v_tmps_all[:curr_size]
                        next_nodes = v_node_vals_grpB[:next_size]
                        next_addrs = gather_addrs_grpB[:next_size]
                    else:
                        curr_nodes = v_node_vals_grpB[:curr_size]
                        curr_tmps = v_tmps_grpB[:curr_size]
                        next_nodes = v_node_vals_all[:next_size]
                        next_addrs = gather_addrs_all[:next_size]

                    # Compute current + prefetch next (ALU + VALU + LOAD all in one scheduler)
                    # The scheduler handles ALU -> LOAD dependency automatically
                    # Wrap only needed when gather_depth >= forest_height
                    needs_wrap = (gather_depth >= forest_height)
                    self.emit_compute_with_prefetch(
                        curr_chunks, v_indices, v_values,
                        curr_nodes, curr_tmps, v_hash_consts,
                        v_one, v_two, v_zero, v_n_nodes,
                        next_chunks, next_addrs, next_nodes,
                        needs_wrap=needs_wrap,
                        include_prefetch_alu=True,
                        forest_values_p=self.scratch["forest_values_p"]
                    )

                # Last group: compute, and optionally pre-load next round's first group
                last_chunks, last_size = groups[-1]
                if (len(groups) - 1) % 2 == 0:
                    last_nodes = v_node_vals_all[:last_size]
                    last_tmps = v_tmps_all[:last_size]
                else:
                    last_nodes = v_node_vals_grpB[:last_size]
                    last_tmps = v_tmps_grpB[:last_size]

                needs_wrap = (gather_depth >= forest_height)

                if next_is_type_b:
                    # Cross-round pipelining: overlap last group compute with next round's first gather
                    # With 6 groups, last uses GroupB (since (6-1)%2=1), so prefetch into GroupA
                    next_first_chunks = groups[0][0]
                    next_first_size = groups[0][1]
                    self.emit_last_group_with_next_gather(
                        last_chunks, v_indices, v_values,
                        last_nodes, last_tmps, v_hash_consts,
                        v_one, v_two, v_zero, v_n_nodes,
                        needs_wrap,
                        next_first_chunks, gather_addrs_all[:next_first_size],
                        v_node_vals_all[:next_first_size],
                        self.scratch["forest_values_p"]
                    )
                    first_group_preloaded = True
                else:
                    # Normal last group compute
                    self.emit_xor_hash_index_nway_scheduled(
                        last_chunks, v_indices, v_values,
                        last_nodes, last_tmps,
                        v_hash_consts, v_one, v_two, v_zero, v_n_nodes,
                        needs_wrap=needs_wrap
                    )

        # ===== FINALIZATION: Store values back to memory (4-way batched) =====
        for chunk in range(0, n_vectors, 4):
            off1 = chunk * VLEN
            off2 = (chunk + 1) * VLEN
            off3 = (chunk + 2) * VLEN
            off4 = (chunk + 3) * VLEN
            # Batch const + ALU + vstore for quads
            self.add_packed({"load": [("const", tmp1, off1), ("const", tmp2, off2)]})
            self.add_packed({"load": [("const", tmp3, off3), ("const", tmp_addr, off4)]})
            self.add_packed({"alu": [
                ("+", tmp1, self.scratch["inp_values_p"], tmp1),
                ("+", tmp2, self.scratch["inp_values_p"], tmp2),
                ("+", tmp3, self.scratch["inp_values_p"], tmp3),
                ("+", tmp_addr, self.scratch["inp_values_p"], tmp_addr),
            ]})
            self.add_packed({"store": [
                ("vstore", tmp1, v_values + off1),
                ("vstore", tmp2, v_values + off2),
            ]})
            self.add_packed({"store": [
                ("vstore", tmp3, v_values + off3),
                ("vstore", tmp_addr, v_values + off4),
            ]})

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
