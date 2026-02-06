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

    def get_vconst(self, val):
        """Get or create a vector constant (8 copies of val)."""
        key = ("vconst", val)
        if key not in self.const_map:
            addr = self.alloc_scratch(f"vc_{val:x}", VLEN)
            scalar_addr = self.scratch_const(val)
            self.vconst_inits.append((addr, scalar_addr))
            self.const_map[key] = addr
        return self.const_map[key]

    def init_hash_vconsts(self):
        """Pre-allocate all vector constants needed for hash function."""
        self.hash_vconsts = {}
        self.hash_collapsible = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                multiplier = 1 + (1 << val3)
                self.hash_collapsible[hi] = True
                self.hash_vconsts[(hi, "mul")] = self.get_vconst(multiplier)
                self.hash_vconsts[(hi, "add")] = self.get_vconst(val1)
            else:
                self.hash_collapsible[hi] = False
                self.hash_vconsts[(hi, 1)] = self.get_vconst(val1)
                self.hash_vconsts[(hi, 3)] = self.get_vconst(val3)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """Scratch-resident SIMD kernel with G=8, multiply_add hash, and
        aggressive interleaving. All idx/val kept in scratch across all rounds."""
        self.vconst_inits = []
        G = 8  # groups per iteration (64 elements)

        # --- Scratch allocation ---
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        tmp_addrs = [tmp1, tmp2] + [self.alloc_scratch(f"tmp_addr_{i}") for i in range(5)]

        # Store address scratch (for overlapping store prep with last compute)
        store_addr1 = self.alloc_scratch("store_a1")
        store_addr2 = self.alloc_scratch("store_a2")

        # Scratch-resident: all idx and val live in scratch
        n_groups = batch_size // VLEN  # 32 groups of 8
        all_idx = [self.alloc_scratch(f"sidx_{g}", VLEN) for g in range(n_groups)]
        all_val = [self.alloc_scratch(f"sval_{g}", VLEN) for g in range(n_groups)]

        # Working vectors
        v_node = [self.alloc_scratch(f"v_node_g{g}", VLEN) for g in range(G)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_g{g}", VLEN) for g in range(G)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_g{g}", VLEN) for g in range(G)]

        # Node addresses for scattered loads
        node_addrs = [[self.alloc_scratch(f"na_g{g}_{i}") for i in range(VLEN)]
                      for g in range(G)]
        next_node_addrs = [[self.alloc_scratch(f"nna_g{g}_{i}") for i in range(VLEN)]
                           for g in range(G)]

        # Load header values
        for i in range(0, len(init_vars), 2):
            ops = [("const", tmp_addrs[i], i)]
            if i + 1 < len(init_vars):
                ops.append(("const", tmp_addrs[i+1], i+1))
            self.instrs.append({"load": ops})
        for i in range(0, len(init_vars), 2):
            ops = [("load", self.scratch[init_vars[i]], tmp_addrs[i])]
            if i + 1 < len(init_vars):
                ops.append(("load", self.scratch[init_vars[i+1]], tmp_addrs[i+1]))
            self.instrs.append({"load": ops})

        # Scalar constants
        zero_const = self.alloc_scratch("const_0")
        one_const = self.alloc_scratch("const_1")
        two_const = self.alloc_scratch("const_2")
        self.instrs.append({"load": [("const", zero_const, 0), ("const", one_const, 1)]})
        self.instrs.append({"load": [("const", two_const, 2)]})
        self.const_map[0] = zero_const
        self.const_map[1] = one_const
        self.const_map[2] = two_const

        # Vector constants
        v_one = self.get_vconst(1)
        v_two = self.get_vconst(2)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.init_hash_vconsts()

        # Emit all vbroadcasts (overlap with offset constant creation)
        all_vbc = [("vbroadcast", va, sa) for va, sa in self.vconst_inits]
        all_vbc.append(("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Pre-calculate which offset constants we'll need
        offset_vals = sorted(set(g * VLEN for g in range(n_groups + 1)))
        # Remove constants we already have (0, 1, 2 will be in const_map after scalar constants)
        new_offset_consts = [(v, self.alloc_scratch(f"off_{v}")) for v in offset_vals if v > 2]

        # Interleave vbroadcast (VALU) with offset constant creation (LOAD)
        vbc_idx = 0
        const_idx = 0
        while vbc_idx < len(all_vbc) or const_idx < len(new_offset_consts):
            instr = {}
            # Add up to 6 vbroadcasts
            if vbc_idx < len(all_vbc):
                vbc_batch = all_vbc[vbc_idx:min(vbc_idx + 6, len(all_vbc))]
                instr["valu"] = vbc_batch
                vbc_idx += len(vbc_batch)
            # Add up to 2 const loads
            if const_idx < len(new_offset_consts):
                const_batch = []
                for _ in range(min(2, len(new_offset_consts) - const_idx)):
                    v, scratch_loc = new_offset_consts[const_idx]
                    const_batch.append(("const", scratch_loc, v))
                    self.const_map[v] = scratch_loc
                    const_idx += 1
                instr["load"] = const_batch
            self.instrs.append(instr)

        # Offset constants now created during vbroadcast overlap above

        # --- Initial load: values only (indices are known zero) ---
        # Build list of all value load operations
        init_loads = []
        for g in range(0, n_groups, 2):
            init_loads.append(("inp_values_p", g, all_val[g], all_val[g + 1]))

        idx_init_ops = [("vbroadcast", all_idx[g], zero_const) for g in range(n_groups)]

        def take_idx_inits(max_ops=6):
            nonlocal idx_init_ops
            if not idx_init_ops:
                return []
            ops = idx_init_ops[:max_ops]
            idx_init_ops = idx_init_ops[max_ops:]
            return ops

        # First: ALU for pair 0 (+ idx init)
        ptr0, g0, _, _ = init_loads[0]
        off1_0 = self.const_map[g0 * VLEN]
        off2_0 = self.const_map[(g0 + 1) * VLEN]
        instr0 = {"alu": [
            ("+", tmp1, self.scratch[ptr0], off1_0),
            ("+", tmp2, self.scratch[ptr0], off2_0),
        ]}
        vb = take_idx_inits(6)
        if vb:
            instr0["valu"] = vb
        self.instrs.append(instr0)

        # Interleaved: load pair N + ALU pair N+1 (+ idx init)
        for si in range(len(init_loads) - 1):
            _, _, da, db = init_loads[si]
            ptr_next, g_next, _, _ = init_loads[si + 1]
            off1_n = self.const_map[g_next * VLEN]
            off2_n = self.const_map[(g_next + 1) * VLEN]
            instr = {
                "load": [("vload", da, tmp1), ("vload", db, tmp2)],
                "alu": [
                    ("+", tmp1, self.scratch[ptr_next], off1_n),
                    ("+", tmp2, self.scratch[ptr_next], off2_n),
                ],
            }
            vb = take_idx_inits(6)
            if vb:
                instr["valu"] = vb
            self.instrs.append(instr)

        # Last: load final pair (+ idx init)
        _, _, da_last, db_last = init_loads[-1]
        instr_last = {"load": [
            ("vload", da_last, tmp1), ("vload", db_last, tmp2),
        ]}
        vb = take_idx_inits(6)
        if vb:
            instr_last["valu"] = vb
        self.instrs.append(instr_last)

        self.add("flow", ("pause",))

        n_iters = n_groups // G  # 4 iterations per round

        # --- Helpers ---
        def get_groups(it):
            base = it * G
            return (
                [all_idx[base + g] for g in range(G)],
                [all_val[base + g] for g in range(G)],
            )

        def emit_node_addr_calc_g8(idx_addrs, na):
            """Compute 64 scattered load addresses for G=8."""
            fvp = self.scratch["forest_values_p"]
            # 64 ALU ops, 12/cycle = 6 cycles
            for batch_start in range(0, G, 2):
                # 2 groups per batch = 16 ALU ops, but max 12/cycle
                g0 = batch_start
                g1 = batch_start + 1
                ops = ([("+", na[g0][i], fvp, idx_addrs[g0] + i) for i in range(VLEN)]
                     + [("+", na[g1][i], fvp, idx_addrs[g1] + i) for i in range(4)])
                self.instrs.append({"alu": ops})
                ops = [("+", na[g1][i], fvp, idx_addrs[g1] + i) for i in range(4, VLEN)]
                self.instrs.append({"alu": ops})

        def emit_scattered_loads_g8(na, dst):
            """Emit 64 scattered loads for G=8."""
            for g in range(G):
                for i in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", dst[g] + i, na[g][i]),
                        ("load", dst[g] + i + 1, na[g][i + 1]),
                    ]})

        def emit_hash_g8(val_addrs):
            """Hash with G=8 and interleaved stages."""
            for hi in range(6):
                if self.hash_collapsible[hi]:
                    vc_mul = self.hash_vconsts[(hi, "mul")]
                    vc_add = self.hash_vconsts[(hi, "add")]
                    # 8 multiply_add, ceil(8/6) = 2 cycles
                    self.instrs.append({"valu": [
                        ("multiply_add", val_addrs[g], val_addrs[g], vc_mul, vc_add) for g in range(6)
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", val_addrs[g], val_addrs[g], vc_mul, vc_add) for g in range(6, G)
                    ]})
                else:
                    op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                    vc1 = self.hash_vconsts[(hi, 1)]
                    vc3 = self.hash_vconsts[(hi, 3)]
                    # op13: 16 ops in 3 cycles (6,6,4)
                    self.instrs.append({"valu": [
                        (op1, v_tmp1[0], val_addrs[0], vc1), (op3, v_tmp2[0], val_addrs[0], vc3),
                        (op1, v_tmp1[1], val_addrs[1], vc1), (op3, v_tmp2[1], val_addrs[1], vc3),
                        (op1, v_tmp1[2], val_addrs[2], vc1), (op3, v_tmp2[2], val_addrs[2], vc3),
                    ]})
                    self.instrs.append({"valu": [
                        (op1, v_tmp1[3], val_addrs[3], vc1), (op3, v_tmp2[3], val_addrs[3], vc3),
                        (op1, v_tmp1[4], val_addrs[4], vc1), (op3, v_tmp2[4], val_addrs[4], vc3),
                        (op1, v_tmp1[5], val_addrs[5], vc1), (op3, v_tmp2[5], val_addrs[5], vc3),
                    ]})
                    self.instrs.append({"valu": [
                        (op1, v_tmp1[6], val_addrs[6], vc1), (op3, v_tmp2[6], val_addrs[6], vc3),
                        (op1, v_tmp1[7], val_addrs[7], vc1), (op3, v_tmp2[7], val_addrs[7], vc3),
                        (op2, val_addrs[0], v_tmp1[0], v_tmp2[0]),
                        (op2, val_addrs[1], v_tmp1[1], v_tmp2[1]),
                    ]})
                    # op2 g2-7: g2-5 written 2 cycles ago ✓, g6-7 written prev cycle ✓
                    self.instrs.append({"valu": [
                        (op2, val_addrs[2], v_tmp1[2], v_tmp2[2]),
                        (op2, val_addrs[3], v_tmp1[3], v_tmp2[3]),
                        (op2, val_addrs[4], v_tmp1[4], v_tmp2[4]),
                        (op2, val_addrs[5], v_tmp1[5], v_tmp2[5]),
                        (op2, val_addrs[6], v_tmp1[6], v_tmp2[6]),
                        (op2, val_addrs[7], v_tmp1[7], v_tmp2[7]),
                    ]})

        def emit_idx_update_g8(idx_addrs, val_addrs):
            """Optimized idx update for G=8 using multiply_add.
            new_idx = 2*idx + 1 + (val&1), wrap to 0 if >= n_nodes."""
            # Cy1: madd g0-2 (2*idx+1), & g0-2 (val&1)
            self.instrs.append({"valu": [
                ("multiply_add", v_tmp2[0], idx_addrs[0], v_two, v_one),
                ("multiply_add", v_tmp2[1], idx_addrs[1], v_two, v_one),
                ("multiply_add", v_tmp2[2], idx_addrs[2], v_two, v_one),
                ("&", v_tmp1[0], val_addrs[0], v_one),
                ("&", v_tmp1[1], val_addrs[1], v_one),
                ("&", v_tmp1[2], val_addrs[2], v_one),
            ]})
            # Cy2: madd g3-5, & g3-5
            self.instrs.append({"valu": [
                ("multiply_add", v_tmp2[3], idx_addrs[3], v_two, v_one),
                ("multiply_add", v_tmp2[4], idx_addrs[4], v_two, v_one),
                ("multiply_add", v_tmp2[5], idx_addrs[5], v_two, v_one),
                ("&", v_tmp1[3], val_addrs[3], v_one),
                ("&", v_tmp1[4], val_addrs[4], v_one),
                ("&", v_tmp1[5], val_addrs[5], v_one),
            ]})
            # Cy3: madd g6-7, & g6-7, add g0-1 (madd+& from Cy1 available ✓)
            self.instrs.append({"valu": [
                ("multiply_add", v_tmp2[6], idx_addrs[6], v_two, v_one),
                ("multiply_add", v_tmp2[7], idx_addrs[7], v_two, v_one),
                ("&", v_tmp1[6], val_addrs[6], v_one),
                ("&", v_tmp1[7], val_addrs[7], v_one),
                ("+", idx_addrs[0], v_tmp2[0], v_tmp1[0]),
                ("+", idx_addrs[1], v_tmp2[1], v_tmp1[1]),
            ]})
            # Cy4: add g2-7 (g2-5 madd from Cy1-2, g6-7 from Cy3 all available ✓)
            self.instrs.append({"valu": [
                ("+", idx_addrs[2], v_tmp2[2], v_tmp1[2]),
                ("+", idx_addrs[3], v_tmp2[3], v_tmp1[3]),
                ("+", idx_addrs[4], v_tmp2[4], v_tmp1[4]),
                ("+", idx_addrs[5], v_tmp2[5], v_tmp1[5]),
                ("+", idx_addrs[6], v_tmp2[6], v_tmp1[6]),
                ("+", idx_addrs[7], v_tmp2[7], v_tmp1[7]),
            ]})
            # Cy5: cmp g0-5 (idx from Cy3-4 available ✓)
            self.instrs.append({"valu": [
                ("<", v_tmp1[0], idx_addrs[0], v_n_nodes),
                ("<", v_tmp1[1], idx_addrs[1], v_n_nodes),
                ("<", v_tmp1[2], idx_addrs[2], v_n_nodes),
                ("<", v_tmp1[3], idx_addrs[3], v_n_nodes),
                ("<", v_tmp1[4], idx_addrs[4], v_n_nodes),
                ("<", v_tmp1[5], idx_addrs[5], v_n_nodes),
            ]})
            # Cy6: cmp g6-7, wrap g0-3
            self.instrs.append({"valu": [
                ("<", v_tmp1[6], idx_addrs[6], v_n_nodes),
                ("<", v_tmp1[7], idx_addrs[7], v_n_nodes),
                ("*", idx_addrs[0], idx_addrs[0], v_tmp1[0]),
                ("*", idx_addrs[1], idx_addrs[1], v_tmp1[1]),
                ("*", idx_addrs[2], idx_addrs[2], v_tmp1[2]),
                ("*", idx_addrs[3], idx_addrs[3], v_tmp1[3]),
            ]})
            # Cy7: wrap g4-7
            self.instrs.append({"valu": [
                ("*", idx_addrs[4], idx_addrs[4], v_tmp1[4]),
                ("*", idx_addrs[5], idx_addrs[5], v_tmp1[5]),
                ("*", idx_addrs[6], idx_addrs[6], v_tmp1[6]),
                ("*", idx_addrs[7], idx_addrs[7], v_tmp1[7]),
            ]})

        def emit_pipelined_iter_g8(it, next_it_idx_addrs):
            """Compute iteration it while loading tree values for next iteration.
            G=8: 64 scattered loads + 64 addr calcs pipelined with compute."""
            cur_idx, cur_val = get_groups(it)
            fvp = self.scratch["forest_values_p"]

            # Scattered load queue: 64 loads for next iteration
            nlq = [(g, i) for g in range(G) for i in range(VLEN)]
            li = 0

            def next_loads(n=2):
                nonlocal li
                ops = []
                for _ in range(n):
                    if li < len(nlq):
                        g, i = nlq[li]
                        ops.append(("load", v_node[g] + i, next_node_addrs[g][i]))
                        li += 1
                return ops

            # Addr calc queue: 64 addr calcs for next iteration
            alq = [(g, i) for g in range(G) for i in range(VLEN)]
            ai = 0

            def next_addr_calcs(n=12):
                nonlocal ai
                ops = []
                for _ in range(n):
                    if ai < len(alq):
                        g, i = alq[ai]
                        ops.append(("+", next_node_addrs[g][i], fvp, next_it_idx_addrs[g] + i))
                        ai += 1
                return ops

            # ---- Hash interleaved with addr calc and scattered loads ----
            # Cy1: XOR g0-5 + addr calc
            self.instrs.append({
                "valu": [("^", cur_val[g], cur_val[g], v_node[g]) for g in range(6)],
                "alu": next_addr_calcs(12),
            })
            # Cy2: XOR g6-7 + H0 madd g0-3 + addr calc + first loads
            vc_mul0 = self.hash_vconsts[(0, "mul")]
            vc_add0 = self.hash_vconsts[(0, "add")]
            self.instrs.append({
                "valu": [
                    ("^", cur_val[6], cur_val[6], v_node[6]),
                    ("^", cur_val[7], cur_val[7], v_node[7]),
                    ("multiply_add", cur_val[0], cur_val[0], vc_mul0, vc_add0),
                    ("multiply_add", cur_val[1], cur_val[1], vc_mul0, vc_add0),
                    ("multiply_add", cur_val[2], cur_val[2], vc_mul0, vc_add0),
                    ("multiply_add", cur_val[3], cur_val[3], vc_mul0, vc_add0),
                ],
                "alu": next_addr_calcs(12),
                "load": next_loads(),  # uses addrs from Cy1
            })
            # Cy3: H0 madd g4-7 + H1 op13 g0 + addr calc + first loads
            hi = 1
            op1_1, _, op2_1, op3_1, _ = HASH_STAGES[hi]
            vc1_1, vc3_1 = self.hash_vconsts[(hi, 1)], self.hash_vconsts[(hi, 3)]
            self.instrs.append({
                "valu": [
                    ("multiply_add", cur_val[4], cur_val[4], vc_mul0, vc_add0),
                    ("multiply_add", cur_val[5], cur_val[5], vc_mul0, vc_add0),
                    ("multiply_add", cur_val[6], cur_val[6], vc_mul0, vc_add0),
                    ("multiply_add", cur_val[7], cur_val[7], vc_mul0, vc_add0),
                    (op1_1, v_tmp1[0], cur_val[0], vc1_1),
                    (op3_1, v_tmp2[0], cur_val[0], vc3_1),
                ],
                "alu": next_addr_calcs(12),
                "load": next_loads(),
            })
            # Cy4: H1 op13 g1-3 + addr calc + loads
            self.instrs.append({
                "valu": [
                    (op1_1, v_tmp1[1], cur_val[1], vc1_1), (op3_1, v_tmp2[1], cur_val[1], vc3_1),
                    (op1_1, v_tmp1[2], cur_val[2], vc1_1), (op3_1, v_tmp2[2], cur_val[2], vc3_1),
                    (op1_1, v_tmp1[3], cur_val[3], vc1_1), (op3_1, v_tmp2[3], cur_val[3], vc3_1),
                ],
                "alu": next_addr_calcs(12),
                "load": next_loads(),
            })
            # Cy5: H1 op13 g4-6 + addr calc + loads
            self.instrs.append({
                "valu": [
                    (op1_1, v_tmp1[4], cur_val[4], vc1_1), (op3_1, v_tmp2[4], cur_val[4], vc3_1),
                    (op1_1, v_tmp1[5], cur_val[5], vc1_1), (op3_1, v_tmp2[5], cur_val[5], vc3_1),
                    (op1_1, v_tmp1[6], cur_val[6], vc1_1), (op3_1, v_tmp2[6], cur_val[6], vc3_1),
                ],
                "alu": next_addr_calcs(12),
                "load": next_loads(),
            })
            # Cy6: H1 op13 g7 + H1 op2 g0-3 + remaining addr calcs + loads
            remaining_ac = next_addr_calcs(12)  # gets remaining 4
            instr6 = {
                "valu": [
                    (op1_1, v_tmp1[7], cur_val[7], vc1_1), (op3_1, v_tmp2[7], cur_val[7], vc3_1),
                    (op2_1, cur_val[0], v_tmp1[0], v_tmp2[0]),
                    (op2_1, cur_val[1], v_tmp1[1], v_tmp2[1]),
                    (op2_1, cur_val[2], v_tmp1[2], v_tmp2[2]),
                    (op2_1, cur_val[3], v_tmp1[3], v_tmp2[3]),
                ],
                "load": next_loads(),
            }
            if remaining_ac:
                instr6["alu"] = remaining_ac
            self.instrs.append(instr6)
            # Cy7: H1 op2 g4-7 + H2 madd g0-1
            # g4-6 op13 from Cy5 ✓, g7 from Cy6 ✓
            vc_mul2 = self.hash_vconsts[(2, "mul")]
            vc_add2 = self.hash_vconsts[(2, "add")]
            self.instrs.append({
                "valu": [
                    (op2_1, cur_val[4], v_tmp1[4], v_tmp2[4]),
                    (op2_1, cur_val[5], v_tmp1[5], v_tmp2[5]),
                    (op2_1, cur_val[6], v_tmp1[6], v_tmp2[6]),
                    (op2_1, cur_val[7], v_tmp1[7], v_tmp2[7]),
                    ("multiply_add", cur_val[0], cur_val[0], vc_mul2, vc_add2),
                    ("multiply_add", cur_val[1], cur_val[1], vc_mul2, vc_add2),
                ],
                "load": next_loads(),
            })
            # Cy8: H2 madd g2-7
            self.instrs.append({
                "valu": [
                    ("multiply_add", cur_val[2], cur_val[2], vc_mul2, vc_add2),
                    ("multiply_add", cur_val[3], cur_val[3], vc_mul2, vc_add2),
                    ("multiply_add", cur_val[4], cur_val[4], vc_mul2, vc_add2),
                    ("multiply_add", cur_val[5], cur_val[5], vc_mul2, vc_add2),
                    ("multiply_add", cur_val[6], cur_val[6], vc_mul2, vc_add2),
                    ("multiply_add", cur_val[7], cur_val[7], vc_mul2, vc_add2),
                ],
                "load": next_loads(),
            })
            # Cy9: H3 op13 g0-2 + loads
            hi = 3
            op1_3, _, op2_3, op3_3, _ = HASH_STAGES[hi]
            vc1_3, vc3_3 = self.hash_vconsts[(hi, 1)], self.hash_vconsts[(hi, 3)]
            self.instrs.append({
                "valu": [
                    (op1_3, v_tmp1[0], cur_val[0], vc1_3), (op3_3, v_tmp2[0], cur_val[0], vc3_3),
                    (op1_3, v_tmp1[1], cur_val[1], vc1_3), (op3_3, v_tmp2[1], cur_val[1], vc3_3),
                    (op1_3, v_tmp1[2], cur_val[2], vc1_3), (op3_3, v_tmp2[2], cur_val[2], vc3_3),
                ],
                "load": next_loads(),
            })
            # Cy10: H3 op13 g3-5 + loads
            self.instrs.append({
                "valu": [
                    (op1_3, v_tmp1[3], cur_val[3], vc1_3), (op3_3, v_tmp2[3], cur_val[3], vc3_3),
                    (op1_3, v_tmp1[4], cur_val[4], vc1_3), (op3_3, v_tmp2[4], cur_val[4], vc3_3),
                    (op1_3, v_tmp1[5], cur_val[5], vc1_3), (op3_3, v_tmp2[5], cur_val[5], vc3_3),
                ],
                "load": next_loads(),
            })
            # Cy11: H3 op13 g6-7 + H3 op2 g0-1 + loads
            self.instrs.append({
                "valu": [
                    (op1_3, v_tmp1[6], cur_val[6], vc1_3), (op3_3, v_tmp2[6], cur_val[6], vc3_3),
                    (op1_3, v_tmp1[7], cur_val[7], vc1_3), (op3_3, v_tmp2[7], cur_val[7], vc3_3),
                    (op2_3, cur_val[0], v_tmp1[0], v_tmp2[0]),
                    (op2_3, cur_val[1], v_tmp1[1], v_tmp2[1]),
                ],
                "load": next_loads(),
            })
            # Cy12: H3 op2 g2-7
            self.instrs.append({
                "valu": [
                    (op2_3, cur_val[2], v_tmp1[2], v_tmp2[2]),
                    (op2_3, cur_val[3], v_tmp1[3], v_tmp2[3]),
                    (op2_3, cur_val[4], v_tmp1[4], v_tmp2[4]),
                    (op2_3, cur_val[5], v_tmp1[5], v_tmp2[5]),
                    (op2_3, cur_val[6], v_tmp1[6], v_tmp2[6]),
                    (op2_3, cur_val[7], v_tmp1[7], v_tmp2[7]),
                ],
                "load": next_loads(),
            })
            # Cy13: H4 madd g0-5
            vc_mul4 = self.hash_vconsts[(4, "mul")]
            vc_add4 = self.hash_vconsts[(4, "add")]
            self.instrs.append({
                "valu": [
                    ("multiply_add", cur_val[0], cur_val[0], vc_mul4, vc_add4),
                    ("multiply_add", cur_val[1], cur_val[1], vc_mul4, vc_add4),
                    ("multiply_add", cur_val[2], cur_val[2], vc_mul4, vc_add4),
                    ("multiply_add", cur_val[3], cur_val[3], vc_mul4, vc_add4),
                    ("multiply_add", cur_val[4], cur_val[4], vc_mul4, vc_add4),
                    ("multiply_add", cur_val[5], cur_val[5], vc_mul4, vc_add4),
                ],
                "load": next_loads(),
            })
            # Cy14: H4 madd g6-7 + H5 op13 g0-1
            hi = 5
            op1_5, _, op2_5, op3_5, _ = HASH_STAGES[hi]
            vc1_5, vc3_5 = self.hash_vconsts[(hi, 1)], self.hash_vconsts[(hi, 3)]
            self.instrs.append({
                "valu": [
                    ("multiply_add", cur_val[6], cur_val[6], vc_mul4, vc_add4),
                    ("multiply_add", cur_val[7], cur_val[7], vc_mul4, vc_add4),
                    (op1_5, v_tmp1[0], cur_val[0], vc1_5), (op3_5, v_tmp2[0], cur_val[0], vc3_5),
                    (op1_5, v_tmp1[1], cur_val[1], vc1_5), (op3_5, v_tmp2[1], cur_val[1], vc3_5),
                ],
                "load": next_loads(),
            })
            # Cy15: H5 op13 g2-4
            self.instrs.append({
                "valu": [
                    (op1_5, v_tmp1[2], cur_val[2], vc1_5), (op3_5, v_tmp2[2], cur_val[2], vc3_5),
                    (op1_5, v_tmp1[3], cur_val[3], vc1_5), (op3_5, v_tmp2[3], cur_val[3], vc3_5),
                    (op1_5, v_tmp1[4], cur_val[4], vc1_5), (op3_5, v_tmp2[4], cur_val[4], vc3_5),
                ],
                "load": next_loads(),
            })
            # Cy16: H5 op13 g5-7 + H5 op2 g0
            self.instrs.append({
                "valu": [
                    (op1_5, v_tmp1[5], cur_val[5], vc1_5), (op3_5, v_tmp2[5], cur_val[5], vc3_5),
                    (op1_5, v_tmp1[6], cur_val[6], vc1_5), (op3_5, v_tmp2[6], cur_val[6], vc3_5),
                    (op1_5, v_tmp1[7], cur_val[7], vc1_5), (op3_5, v_tmp2[7], cur_val[7], vc3_5),
                ],
                "load": next_loads(),
            })
            # Cy17: H5 op2 g0-5 (g0-1 from Cy14, g2-4 from Cy15, g5 from Cy16)
            self.instrs.append({
                "valu": [
                    (op2_5, cur_val[0], v_tmp1[0], v_tmp2[0]),
                    (op2_5, cur_val[1], v_tmp1[1], v_tmp2[1]),
                    (op2_5, cur_val[2], v_tmp1[2], v_tmp2[2]),
                    (op2_5, cur_val[3], v_tmp1[3], v_tmp2[3]),
                    (op2_5, cur_val[4], v_tmp1[4], v_tmp2[4]),
                    (op2_5, cur_val[5], v_tmp1[5], v_tmp2[5]),
                ],
                "load": next_loads(),
            })
            # Cy18: H5 op2 g6-7 + idx madd g0-1, & g0-1
            self.instrs.append({
                "valu": [
                    (op2_5, cur_val[6], v_tmp1[6], v_tmp2[6]),
                    (op2_5, cur_val[7], v_tmp1[7], v_tmp2[7]),
                    ("multiply_add", v_tmp2[0], cur_idx[0], v_two, v_one),
                    ("&", v_tmp1[0], cur_val[0], v_one),
                    ("multiply_add", v_tmp2[1], cur_idx[1], v_two, v_one),
                    ("&", v_tmp1[1], cur_val[1], v_one),
                ],
                "load": next_loads(),
            })
            # Cy19: idx madd+& g2-4
            self.instrs.append({
                "valu": [
                    ("multiply_add", v_tmp2[2], cur_idx[2], v_two, v_one),
                    ("&", v_tmp1[2], cur_val[2], v_one),
                    ("multiply_add", v_tmp2[3], cur_idx[3], v_two, v_one),
                    ("&", v_tmp1[3], cur_val[3], v_one),
                    ("multiply_add", v_tmp2[4], cur_idx[4], v_two, v_one),
                    ("&", v_tmp1[4], cur_val[4], v_one),
                ],
                "load": next_loads(),
            })
            # Cy20: idx madd+& g5-6 + add g0-1 (reads madd/& from Cy18 ✓)
            self.instrs.append({
                "valu": [
                    ("multiply_add", v_tmp2[5], cur_idx[5], v_two, v_one),
                    ("&", v_tmp1[5], cur_val[5], v_one),
                    ("multiply_add", v_tmp2[6], cur_idx[6], v_two, v_one),
                    ("&", v_tmp1[6], cur_val[6], v_one),
                    ("+", cur_idx[0], v_tmp2[0], v_tmp1[0]),
                    ("+", cur_idx[1], v_tmp2[1], v_tmp1[1]),
                ],
                "load": next_loads(),
            })
            # Cy21: idx madd+& g7 + add g2-5 (g2-4 from Cy19 ✓, g5 from Cy20 ✓)
            self.instrs.append({
                "valu": [
                    ("multiply_add", v_tmp2[7], cur_idx[7], v_two, v_one),
                    ("&", v_tmp1[7], cur_val[7], v_one),
                    ("+", cur_idx[2], v_tmp2[2], v_tmp1[2]),
                    ("+", cur_idx[3], v_tmp2[3], v_tmp1[3]),
                    ("+", cur_idx[4], v_tmp2[4], v_tmp1[4]),
                    ("+", cur_idx[5], v_tmp2[5], v_tmp1[5]),
                ],
                "load": next_loads(),
            })
            # Cy22: add g6-7 + cmp g0-3 (g6 from Cy20 ✓, g7 from Cy21 ✓)
            self.instrs.append({
                "valu": [
                    ("+", cur_idx[6], v_tmp2[6], v_tmp1[6]),
                    ("+", cur_idx[7], v_tmp2[7], v_tmp1[7]),
                    ("<", v_tmp1[0], cur_idx[0], v_n_nodes),
                    ("<", v_tmp1[1], cur_idx[1], v_n_nodes),
                    ("<", v_tmp1[2], cur_idx[2], v_n_nodes),
                    ("<", v_tmp1[3], cur_idx[3], v_n_nodes),
                ],
                "load": next_loads(),
            })
            # Cy23: cmp g4-7 + wrap g0-1
            self.instrs.append({
                "valu": [
                    ("<", v_tmp1[4], cur_idx[4], v_n_nodes),
                    ("<", v_tmp1[5], cur_idx[5], v_n_nodes),
                    ("<", v_tmp1[6], cur_idx[6], v_n_nodes),
                    ("<", v_tmp1[7], cur_idx[7], v_n_nodes),
                    ("*", cur_idx[0], cur_idx[0], v_tmp1[0]),
                    ("*", cur_idx[1], cur_idx[1], v_tmp1[1]),
                ],
                "load": next_loads(),
            })
            # Cy24: wrap g2-7
            ld = next_loads()
            instr = {"valu": [
                ("*", cur_idx[2], cur_idx[2], v_tmp1[2]),
                ("*", cur_idx[3], cur_idx[3], v_tmp1[3]),
                ("*", cur_idx[4], cur_idx[4], v_tmp1[4]),
                ("*", cur_idx[5], cur_idx[5], v_tmp1[5]),
                ("*", cur_idx[6], cur_idx[6], v_tmp1[6]),
                ("*", cur_idx[7], cur_idx[7], v_tmp1[7]),
            ]}
            if ld: instr["load"] = ld
            self.instrs.append(instr)

            # Remaining loads
            while li < len(nlq):
                ld = next_loads()
                if ld:
                    self.instrs.append({"load": ld})

        def emit_compute_iter_g8(it, prep_store=False):
            """Interleaved compute for G=8: 24 VALU cycles.
            Uses same schedule as pipelined version but without extra loads/ALU.
            If prep_store=True, uses idle ALU cycles to pre-compute first store addresses."""
            V, _ = get_groups(it)  # Wait, get_groups returns (idx, val)
            cur_idx, cur_val = get_groups(it)
            V = cur_val
            I = cur_idx

            # Stage shortcuts
            vc_mul0 = self.hash_vconsts[(0, "mul")]
            vc_add0 = self.hash_vconsts[(0, "add")]
            op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
            vc1_1 = self.hash_vconsts[(1, 1)]
            vc3_1 = self.hash_vconsts[(1, 3)]
            vc_mul2 = self.hash_vconsts[(2, "mul")]
            vc_add2 = self.hash_vconsts[(2, "add")]
            op1_3, _, op2_3, op3_3, _ = HASH_STAGES[3]
            vc1_3 = self.hash_vconsts[(3, 1)]
            vc3_3 = self.hash_vconsts[(3, 3)]
            vc_mul4 = self.hash_vconsts[(4, "mul")]
            vc_add4 = self.hash_vconsts[(4, "add")]
            op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
            vc1_5 = self.hash_vconsts[(5, 1)]
            vc3_5 = self.hash_vconsts[(5, 3)]
            T1, T2 = v_tmp1, v_tmp2

            # Cy1: XOR g0-5 (+ store prep if requested)
            instr1 = {"valu": [("^", V[g], V[g], v_node[g]) for g in range(6)]}
            if prep_store:
                # Pre-compute first store address pair (values group 0, 1)
                off1_0 = self.scratch_const(0 * VLEN)
                off2_0 = self.scratch_const(1 * VLEN)
                instr1["alu"] = [
                    ("+", store_addr1, self.scratch["inp_values_p"], off1_0),
                    ("+", store_addr2, self.scratch["inp_values_p"], off2_0),
                ]
            self.instrs.append(instr1)
            # Cy2: XOR g6-7 + H0 madd g0-3
            self.instrs.append({"valu": [
                ("^", V[6], V[6], v_node[6]), ("^", V[7], V[7], v_node[7]),
                ("multiply_add", V[0], V[0], vc_mul0, vc_add0),
                ("multiply_add", V[1], V[1], vc_mul0, vc_add0),
                ("multiply_add", V[2], V[2], vc_mul0, vc_add0),
                ("multiply_add", V[3], V[3], vc_mul0, vc_add0),
            ]})
            # Cy3: H0 madd g4-7 + H1 op13 g0
            self.instrs.append({"valu": [
                ("multiply_add", V[4], V[4], vc_mul0, vc_add0),
                ("multiply_add", V[5], V[5], vc_mul0, vc_add0),
                ("multiply_add", V[6], V[6], vc_mul0, vc_add0),
                ("multiply_add", V[7], V[7], vc_mul0, vc_add0),
                (op1_1, T1[0], V[0], vc1_1), (op3_1, T2[0], V[0], vc3_1),
            ]})
            # Cy4: H1 op13 g1-3
            self.instrs.append({"valu": [
                (op1_1, T1[1], V[1], vc1_1), (op3_1, T2[1], V[1], vc3_1),
                (op1_1, T1[2], V[2], vc1_1), (op3_1, T2[2], V[2], vc3_1),
                (op1_1, T1[3], V[3], vc1_1), (op3_1, T2[3], V[3], vc3_1),
            ]})
            # Cy5: H1 op13 g4-6
            self.instrs.append({"valu": [
                (op1_1, T1[4], V[4], vc1_1), (op3_1, T2[4], V[4], vc3_1),
                (op1_1, T1[5], V[5], vc1_1), (op3_1, T2[5], V[5], vc3_1),
                (op1_1, T1[6], V[6], vc1_1), (op3_1, T2[6], V[6], vc3_1),
            ]})
            # Cy6: H1 op13 g7 + H1 op2 g0-3
            self.instrs.append({"valu": [
                (op1_1, T1[7], V[7], vc1_1), (op3_1, T2[7], V[7], vc3_1),
                (op2_1, V[0], T1[0], T2[0]), (op2_1, V[1], T1[1], T2[1]),
                (op2_1, V[2], T1[2], T2[2]), (op2_1, V[3], T1[3], T2[3]),
            ]})
            # Cy7: H1 op2 g4-7 + H2 madd g0-1
            self.instrs.append({"valu": [
                (op2_1, V[4], T1[4], T2[4]), (op2_1, V[5], T1[5], T2[5]),
                (op2_1, V[6], T1[6], T2[6]), (op2_1, V[7], T1[7], T2[7]),
                ("multiply_add", V[0], V[0], vc_mul2, vc_add2),
                ("multiply_add", V[1], V[1], vc_mul2, vc_add2),
            ]})
            # Cy8: H2 madd g2-7
            self.instrs.append({"valu": [
                ("multiply_add", V[2], V[2], vc_mul2, vc_add2),
                ("multiply_add", V[3], V[3], vc_mul2, vc_add2),
                ("multiply_add", V[4], V[4], vc_mul2, vc_add2),
                ("multiply_add", V[5], V[5], vc_mul2, vc_add2),
                ("multiply_add", V[6], V[6], vc_mul2, vc_add2),
                ("multiply_add", V[7], V[7], vc_mul2, vc_add2),
            ]})
            # Cy9: H3 op13 g0-2
            self.instrs.append({"valu": [
                (op1_3, T1[0], V[0], vc1_3), (op3_3, T2[0], V[0], vc3_3),
                (op1_3, T1[1], V[1], vc1_3), (op3_3, T2[1], V[1], vc3_3),
                (op1_3, T1[2], V[2], vc1_3), (op3_3, T2[2], V[2], vc3_3),
            ]})
            # Cy10: H3 op13 g3-5
            self.instrs.append({"valu": [
                (op1_3, T1[3], V[3], vc1_3), (op3_3, T2[3], V[3], vc3_3),
                (op1_3, T1[4], V[4], vc1_3), (op3_3, T2[4], V[4], vc3_3),
                (op1_3, T1[5], V[5], vc1_3), (op3_3, T2[5], V[5], vc3_3),
            ]})
            # Cy11: H3 op13 g6-7 + H3 op2 g0-1
            self.instrs.append({"valu": [
                (op1_3, T1[6], V[6], vc1_3), (op3_3, T2[6], V[6], vc3_3),
                (op1_3, T1[7], V[7], vc1_3), (op3_3, T2[7], V[7], vc3_3),
                (op2_3, V[0], T1[0], T2[0]), (op2_3, V[1], T1[1], T2[1]),
            ]})
            # Cy12: H3 op2 g2-7
            self.instrs.append({"valu": [
                (op2_3, V[2], T1[2], T2[2]), (op2_3, V[3], T1[3], T2[3]),
                (op2_3, V[4], T1[4], T2[4]), (op2_3, V[5], T1[5], T2[5]),
                (op2_3, V[6], T1[6], T2[6]), (op2_3, V[7], T1[7], T2[7]),
            ]})
            # Cy13: H4 madd g0-5
            self.instrs.append({"valu": [
                ("multiply_add", V[0], V[0], vc_mul4, vc_add4),
                ("multiply_add", V[1], V[1], vc_mul4, vc_add4),
                ("multiply_add", V[2], V[2], vc_mul4, vc_add4),
                ("multiply_add", V[3], V[3], vc_mul4, vc_add4),
                ("multiply_add", V[4], V[4], vc_mul4, vc_add4),
                ("multiply_add", V[5], V[5], vc_mul4, vc_add4),
            ]})
            # Cy14: H4 madd g6-7 + H5 op13 g0-1
            self.instrs.append({"valu": [
                ("multiply_add", V[6], V[6], vc_mul4, vc_add4),
                ("multiply_add", V[7], V[7], vc_mul4, vc_add4),
                (op1_5, T1[0], V[0], vc1_5), (op3_5, T2[0], V[0], vc3_5),
                (op1_5, T1[1], V[1], vc1_5), (op3_5, T2[1], V[1], vc3_5),
            ]})
            # Cy15: H5 op13 g2-4
            self.instrs.append({"valu": [
                (op1_5, T1[2], V[2], vc1_5), (op3_5, T2[2], V[2], vc3_5),
                (op1_5, T1[3], V[3], vc1_5), (op3_5, T2[3], V[3], vc3_5),
                (op1_5, T1[4], V[4], vc1_5), (op3_5, T2[4], V[4], vc3_5),
            ]})
            # Cy16: H5 op13 g5-7
            self.instrs.append({"valu": [
                (op1_5, T1[5], V[5], vc1_5), (op3_5, T2[5], V[5], vc3_5),
                (op1_5, T1[6], V[6], vc1_5), (op3_5, T2[6], V[6], vc3_5),
                (op1_5, T1[7], V[7], vc1_5), (op3_5, T2[7], V[7], vc3_5),
            ]})
            # Cy17: H5 op2 g0-5
            self.instrs.append({"valu": [
                (op2_5, V[0], T1[0], T2[0]), (op2_5, V[1], T1[1], T2[1]),
                (op2_5, V[2], T1[2], T2[2]), (op2_5, V[3], T1[3], T2[3]),
                (op2_5, V[4], T1[4], T2[4]), (op2_5, V[5], T1[5], T2[5]),
            ]})
            # Cy18: H5 op2 g6-7 + idx madd g0-1, & g0-1
            self.instrs.append({"valu": [
                (op2_5, V[6], T1[6], T2[6]), (op2_5, V[7], T1[7], T2[7]),
                ("multiply_add", T2[0], I[0], v_two, v_one), ("&", T1[0], V[0], v_one),
                ("multiply_add", T2[1], I[1], v_two, v_one), ("&", T1[1], V[1], v_one),
            ]})
            # Cy19: idx madd+& g2-4
            self.instrs.append({"valu": [
                ("multiply_add", T2[2], I[2], v_two, v_one), ("&", T1[2], V[2], v_one),
                ("multiply_add", T2[3], I[3], v_two, v_one), ("&", T1[3], V[3], v_one),
                ("multiply_add", T2[4], I[4], v_two, v_one), ("&", T1[4], V[4], v_one),
            ]})
            # Cy20: idx madd+& g5-6, + g0-1
            self.instrs.append({"valu": [
                ("multiply_add", T2[5], I[5], v_two, v_one), ("&", T1[5], V[5], v_one),
                ("multiply_add", T2[6], I[6], v_two, v_one), ("&", T1[6], V[6], v_one),
                ("+", I[0], T2[0], T1[0]), ("+", I[1], T2[1], T1[1]),
            ]})
            # Cy21: idx madd+& g7, + g2-5
            self.instrs.append({"valu": [
                ("multiply_add", T2[7], I[7], v_two, v_one), ("&", T1[7], V[7], v_one),
                ("+", I[2], T2[2], T1[2]), ("+", I[3], T2[3], T1[3]),
                ("+", I[4], T2[4], T1[4]), ("+", I[5], T2[5], T1[5]),
            ]})
            # Cy22: + g6-7, < g0-3
            self.instrs.append({"valu": [
                ("+", I[6], T2[6], T1[6]), ("+", I[7], T2[7], T1[7]),
                ("<", T1[0], I[0], v_n_nodes), ("<", T1[1], I[1], v_n_nodes),
                ("<", T1[2], I[2], v_n_nodes), ("<", T1[3], I[3], v_n_nodes),
            ]})
            # Cy23: < g4-7, wrap g0-1
            self.instrs.append({"valu": [
                ("<", T1[4], I[4], v_n_nodes), ("<", T1[5], I[5], v_n_nodes),
                ("<", T1[6], I[6], v_n_nodes), ("<", T1[7], I[7], v_n_nodes),
                ("*", I[0], I[0], T1[0]), ("*", I[1], I[1], T1[1]),
            ]})
            # Cy24: wrap g2-7
            self.instrs.append({"valu": [
                ("*", I[2], I[2], T1[2]), ("*", I[3], I[3], T1[3]),
                ("*", I[4], I[4], T1[4]), ("*", I[5], I[5], T1[5]),
                ("*", I[6], I[6], T1[6]), ("*", I[7], I[7], T1[7]),
            ]})

        def emit_compute_iter_d0_g8(it, wrap=True, idx_zero=False):
            """Compute for depth-0 rounds: XOR directly with v_tree[0], skip vbroadcast.
            Saves 2 cycles per d0 iteration by not needing emit_broadcast_tree_value.
            If wrap=False, skip the idx wrap check (valid for depths < forest_height).
            If idx_zero=True, assume idx is 0 and use a cheaper update."""
            cur_idx, cur_val = get_groups(it)
            V = cur_val
            I = cur_idx

            # Stage shortcuts
            vc_mul0 = self.hash_vconsts[(0, "mul")]
            vc_add0 = self.hash_vconsts[(0, "add")]
            op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
            vc1_1 = self.hash_vconsts[(1, 1)]
            vc3_1 = self.hash_vconsts[(1, 3)]
            vc_mul2 = self.hash_vconsts[(2, "mul")]
            vc_add2 = self.hash_vconsts[(2, "add")]
            op1_3, _, op2_3, op3_3, _ = HASH_STAGES[3]
            vc1_3 = self.hash_vconsts[(3, 1)]
            vc3_3 = self.hash_vconsts[(3, 3)]
            vc_mul4 = self.hash_vconsts[(4, "mul")]
            vc_add4 = self.hash_vconsts[(4, "add")]
            op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
            vc1_5 = self.hash_vconsts[(5, 1)]
            vc3_5 = self.hash_vconsts[(5, 3)]
            T1, T2 = v_tmp1, v_tmp2

            # Cy1: XOR g0-5 with v_tree[0] directly (no v_node needed)
            self.instrs.append({"valu": [("^", V[g], V[g], v_tree[0]) for g in range(6)]})
            # Cy2: XOR g6-7 + H0 madd g0-3
            self.instrs.append({"valu": [
                ("^", V[6], V[6], v_tree[0]), ("^", V[7], V[7], v_tree[0]),
                ("multiply_add", V[0], V[0], vc_mul0, vc_add0),
                ("multiply_add", V[1], V[1], vc_mul0, vc_add0),
                ("multiply_add", V[2], V[2], vc_mul0, vc_add0),
                ("multiply_add", V[3], V[3], vc_mul0, vc_add0),
            ]})
            # Cy3-Cy24: same as emit_compute_iter_g8 from Cy3 onwards
            self.instrs.append({"valu": [
                ("multiply_add", V[4], V[4], vc_mul0, vc_add0),
                ("multiply_add", V[5], V[5], vc_mul0, vc_add0),
                ("multiply_add", V[6], V[6], vc_mul0, vc_add0),
                ("multiply_add", V[7], V[7], vc_mul0, vc_add0),
                (op1_1, T1[0], V[0], vc1_1), (op3_1, T2[0], V[0], vc3_1),
            ]})
            self.instrs.append({"valu": [
                (op1_1, T1[1], V[1], vc1_1), (op3_1, T2[1], V[1], vc3_1),
                (op1_1, T1[2], V[2], vc1_1), (op3_1, T2[2], V[2], vc3_1),
                (op1_1, T1[3], V[3], vc1_1), (op3_1, T2[3], V[3], vc3_1),
            ]})
            self.instrs.append({"valu": [
                (op1_1, T1[4], V[4], vc1_1), (op3_1, T2[4], V[4], vc3_1),
                (op1_1, T1[5], V[5], vc1_1), (op3_1, T2[5], V[5], vc3_1),
                (op1_1, T1[6], V[6], vc1_1), (op3_1, T2[6], V[6], vc3_1),
            ]})
            self.instrs.append({"valu": [
                (op1_1, T1[7], V[7], vc1_1), (op3_1, T2[7], V[7], vc3_1),
                (op2_1, V[0], T1[0], T2[0]), (op2_1, V[1], T1[1], T2[1]),
                (op2_1, V[2], T1[2], T2[2]), (op2_1, V[3], T1[3], T2[3]),
            ]})
            self.instrs.append({"valu": [
                (op2_1, V[4], T1[4], T2[4]), (op2_1, V[5], T1[5], T2[5]),
                (op2_1, V[6], T1[6], T2[6]), (op2_1, V[7], T1[7], T2[7]),
                ("multiply_add", V[0], V[0], vc_mul2, vc_add2),
                ("multiply_add", V[1], V[1], vc_mul2, vc_add2),
            ]})
            self.instrs.append({"valu": [
                ("multiply_add", V[2], V[2], vc_mul2, vc_add2),
                ("multiply_add", V[3], V[3], vc_mul2, vc_add2),
                ("multiply_add", V[4], V[4], vc_mul2, vc_add2),
                ("multiply_add", V[5], V[5], vc_mul2, vc_add2),
                ("multiply_add", V[6], V[6], vc_mul2, vc_add2),
                ("multiply_add", V[7], V[7], vc_mul2, vc_add2),
            ]})
            self.instrs.append({"valu": [
                (op1_3, T1[0], V[0], vc1_3), (op3_3, T2[0], V[0], vc3_3),
                (op1_3, T1[1], V[1], vc1_3), (op3_3, T2[1], V[1], vc3_3),
                (op1_3, T1[2], V[2], vc1_3), (op3_3, T2[2], V[2], vc3_3),
            ]})
            self.instrs.append({"valu": [
                (op1_3, T1[3], V[3], vc1_3), (op3_3, T2[3], V[3], vc3_3),
                (op1_3, T1[4], V[4], vc1_3), (op3_3, T2[4], V[4], vc3_3),
                (op1_3, T1[5], V[5], vc1_3), (op3_3, T2[5], V[5], vc3_3),
            ]})
            self.instrs.append({"valu": [
                (op1_3, T1[6], V[6], vc1_3), (op3_3, T2[6], V[6], vc3_3),
                (op1_3, T1[7], V[7], vc1_3), (op3_3, T2[7], V[7], vc3_3),
                (op2_3, V[0], T1[0], T2[0]), (op2_3, V[1], T1[1], T2[1]),
            ]})
            self.instrs.append({"valu": [
                (op2_3, V[2], T1[2], T2[2]), (op2_3, V[3], T1[3], T2[3]),
                (op2_3, V[4], T1[4], T2[4]), (op2_3, V[5], T1[5], T2[5]),
                (op2_3, V[6], T1[6], T2[6]), (op2_3, V[7], T1[7], T2[7]),
            ]})
            self.instrs.append({"valu": [
                ("multiply_add", V[0], V[0], vc_mul4, vc_add4),
                ("multiply_add", V[1], V[1], vc_mul4, vc_add4),
                ("multiply_add", V[2], V[2], vc_mul4, vc_add4),
                ("multiply_add", V[3], V[3], vc_mul4, vc_add4),
                ("multiply_add", V[4], V[4], vc_mul4, vc_add4),
                ("multiply_add", V[5], V[5], vc_mul4, vc_add4),
            ]})
            self.instrs.append({"valu": [
                ("multiply_add", V[6], V[6], vc_mul4, vc_add4),
                ("multiply_add", V[7], V[7], vc_mul4, vc_add4),
                (op1_5, T1[0], V[0], vc1_5), (op3_5, T2[0], V[0], vc3_5),
                (op1_5, T1[1], V[1], vc1_5), (op3_5, T2[1], V[1], vc3_5),
            ]})
            self.instrs.append({"valu": [
                (op1_5, T1[2], V[2], vc1_5), (op3_5, T2[2], V[2], vc3_5),
                (op1_5, T1[3], V[3], vc1_5), (op3_5, T2[3], V[3], vc3_5),
                (op1_5, T1[4], V[4], vc1_5), (op3_5, T2[4], V[4], vc3_5),
            ]})
            self.instrs.append({"valu": [
                (op1_5, T1[5], V[5], vc1_5), (op3_5, T2[5], V[5], vc3_5),
                (op1_5, T1[6], V[6], vc1_5), (op3_5, T2[6], V[6], vc3_5),
                (op1_5, T1[7], V[7], vc1_5), (op3_5, T2[7], V[7], vc3_5),
            ]})
            self.instrs.append({"valu": [
                (op2_5, V[0], T1[0], T2[0]), (op2_5, V[1], T1[1], T2[1]),
                (op2_5, V[2], T1[2], T2[2]), (op2_5, V[3], T1[3], T2[3]),
                (op2_5, V[4], T1[4], T2[4]), (op2_5, V[5], T1[5], T2[5]),
            ]})
            if idx_zero:
                # idx is known 0: new_idx = 1 + (val & 1)
                self.instrs.append({"valu": [
                    (op2_5, V[6], T1[6], T2[6]), (op2_5, V[7], T1[7], T2[7]),
                    ("&", T1[0], V[0], v_one), ("&", T1[1], V[1], v_one),
                    ("&", T1[2], V[2], v_one), ("&", T1[3], V[3], v_one),
                ]})
                self.instrs.append({"valu": [
                    ("&", T1[4], V[4], v_one), ("&", T1[5], V[5], v_one),
                    ("&", T1[6], V[6], v_one), ("&", T1[7], V[7], v_one),
                    ("+", I[0], T1[0], v_one), ("+", I[1], T1[1], v_one),
                ]})
                self.instrs.append({"valu": [
                    ("+", I[2], T1[2], v_one), ("+", I[3], T1[3], v_one),
                    ("+", I[4], T1[4], v_one), ("+", I[5], T1[5], v_one),
                    ("+", I[6], T1[6], v_one), ("+", I[7], T1[7], v_one),
                ]})
            else:
                self.instrs.append({"valu": [
                    (op2_5, V[6], T1[6], T2[6]), (op2_5, V[7], T1[7], T2[7]),
                    ("multiply_add", T2[0], I[0], v_two, v_one), ("&", T1[0], V[0], v_one),
                    ("multiply_add", T2[1], I[1], v_two, v_one), ("&", T1[1], V[1], v_one),
                ]})
                self.instrs.append({"valu": [
                    ("multiply_add", T2[2], I[2], v_two, v_one), ("&", T1[2], V[2], v_one),
                    ("multiply_add", T2[3], I[3], v_two, v_one), ("&", T1[3], V[3], v_one),
                    ("multiply_add", T2[4], I[4], v_two, v_one), ("&", T1[4], V[4], v_one),
                ]})
                self.instrs.append({"valu": [
                    ("multiply_add", T2[5], I[5], v_two, v_one), ("&", T1[5], V[5], v_one),
                    ("multiply_add", T2[6], I[6], v_two, v_one), ("&", T1[6], V[6], v_one),
                    ("+", I[0], T2[0], T1[0]), ("+", I[1], T2[1], T1[1]),
                ]})
                self.instrs.append({"valu": [
                    ("multiply_add", T2[7], I[7], v_two, v_one), ("&", T1[7], V[7], v_one),
                    ("+", I[2], T2[2], T1[2]), ("+", I[3], T2[3], T1[3]),
                    ("+", I[4], T2[4], T1[4]), ("+", I[5], T2[5], T1[5]),
                ]})
                if wrap:
                    self.instrs.append({"valu": [
                        ("+", I[6], T2[6], T1[6]), ("+", I[7], T2[7], T1[7]),
                        ("<", T1[0], I[0], v_n_nodes), ("<", T1[1], I[1], v_n_nodes),
                        ("<", T1[2], I[2], v_n_nodes), ("<", T1[3], I[3], v_n_nodes),
                    ]})
                    self.instrs.append({"valu": [
                        ("<", T1[4], I[4], v_n_nodes), ("<", T1[5], I[5], v_n_nodes),
                        ("<", T1[6], I[6], v_n_nodes), ("<", T1[7], I[7], v_n_nodes),
                        ("*", I[0], I[0], T1[0]), ("*", I[1], I[1], T1[1]),
                    ]})
                    self.instrs.append({"valu": [
                        ("*", I[2], I[2], T1[2]), ("*", I[3], I[3], T1[3]),
                        ("*", I[4], I[4], T1[4]), ("*", I[5], I[5], T1[5]),
                        ("*", I[6], I[6], T1[6]), ("*", I[7], I[7], T1[7]),
                    ]})
                else:
                    self.instrs.append({"valu": [
                        ("+", I[6], T2[6], T1[6]), ("+", I[7], T2[7], T1[7]),
                    ]})

        def emit_compute_iter_d1_g8(it, wrap=True):
            """Fused d1 selection + compute: saves 1 cycle vs separate calls.
            Total: 27 cycles (vs 4+24=28 for separate emit_broadcast + emit_compute).
            Overlaps the tail of d1 mask+madd selection with the start of XOR+hash.
            If wrap=False, skip the idx wrap check (valid for depths < forest_height)."""
            cur_idx, cur_val = get_groups(it)
            V = cur_val
            I = cur_idx

            # Stage shortcuts
            vc_mul0 = self.hash_vconsts[(0, "mul")]
            vc_add0 = self.hash_vconsts[(0, "add")]
            op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
            vc1_1 = self.hash_vconsts[(1, 1)]
            vc3_1 = self.hash_vconsts[(1, 3)]
            vc_mul2 = self.hash_vconsts[(2, "mul")]
            vc_add2 = self.hash_vconsts[(2, "add")]
            op1_3, _, op2_3, op3_3, _ = HASH_STAGES[3]
            vc1_3 = self.hash_vconsts[(3, 1)]
            vc3_3 = self.hash_vconsts[(3, 3)]
            vc_mul4 = self.hash_vconsts[(4, "mul")]
            vc_add4 = self.hash_vconsts[(4, "add")]
            op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
            vc1_5 = self.hash_vconsts[(5, 1)]
            vc3_5 = self.hash_vconsts[(5, 3)]
            T1, T2 = v_tmp1, v_tmp2

            # d1 selection: v_node[g] = v_tree_d1_diff * (idx&1) + v_tree[2]
            # Fused with compute start

            # Cy1: mask g0-5 (& idx with v_one → v_tmp1[0..5])
            self.instrs.append({"valu": [("&", T1[g], I[g], v_one) for g in range(6)]})
            # Cy2: mask g6-7 + madd g0-3 (selection: v_node = diff*mask + tree[2])
            self.instrs.append({"valu": [
                ("&", T1[6], I[6], v_one), ("&", T1[7], I[7], v_one),
                ("multiply_add", v_node[0], v_tree_d1_diff, T1[0], v_tree[2]),
                ("multiply_add", v_node[1], v_tree_d1_diff, T1[1], v_tree[2]),
                ("multiply_add", v_node[2], v_tree_d1_diff, T1[2], v_tree[2]),
                ("multiply_add", v_node[3], v_tree_d1_diff, T1[3], v_tree[2]),
            ]})
            # Cy3: madd g4-7 + XOR g0-1 (v_node[0..3] available from Cy2 ✓)
            self.instrs.append({"valu": [
                ("multiply_add", v_node[4], v_tree_d1_diff, T1[4], v_tree[2]),
                ("multiply_add", v_node[5], v_tree_d1_diff, T1[5], v_tree[2]),
                ("multiply_add", v_node[6], v_tree_d1_diff, T1[6], v_tree[2]),
                ("multiply_add", v_node[7], v_tree_d1_diff, T1[7], v_tree[2]),
                ("^", V[0], V[0], v_node[0]),
                ("^", V[1], V[1], v_node[1]),
            ]})
            # Cy4: XOR g2-5 + H0 madd g0-1 (v_node[4..7] from Cy3 ✓, V[0..1] XOR from Cy3 ✓)
            self.instrs.append({"valu": [
                ("^", V[2], V[2], v_node[2]), ("^", V[3], V[3], v_node[3]),
                ("^", V[4], V[4], v_node[4]), ("^", V[5], V[5], v_node[5]),
                ("multiply_add", V[0], V[0], vc_mul0, vc_add0),
                ("multiply_add", V[1], V[1], vc_mul0, vc_add0),
            ]})
            # Cy5: XOR g6-7 + H0 madd g2-5 (V[2..5] XOR from Cy4 ✓)
            self.instrs.append({"valu": [
                ("^", V[6], V[6], v_node[6]), ("^", V[7], V[7], v_node[7]),
                ("multiply_add", V[2], V[2], vc_mul0, vc_add0),
                ("multiply_add", V[3], V[3], vc_mul0, vc_add0),
                ("multiply_add", V[4], V[4], vc_mul0, vc_add0),
                ("multiply_add", V[5], V[5], vc_mul0, vc_add0),
            ]})
            # Cy6: H0 madd g6-7 + H1 op13 g0-1 (V[6..7] XOR from Cy5 ✓, V[0..1] H0 from Cy4 ✓)
            self.instrs.append({"valu": [
                ("multiply_add", V[6], V[6], vc_mul0, vc_add0),
                ("multiply_add", V[7], V[7], vc_mul0, vc_add0),
                (op1_1, T1[0], V[0], vc1_1), (op3_1, T2[0], V[0], vc3_1),
                (op1_1, T1[1], V[1], vc1_1), (op3_1, T2[1], V[1], vc3_1),
            ]})
            # Cy7: H1 op13 g2-4 (V[2..4] H0 from Cy5 ✓)
            self.instrs.append({"valu": [
                (op1_1, T1[2], V[2], vc1_1), (op3_1, T2[2], V[2], vc3_1),
                (op1_1, T1[3], V[3], vc1_1), (op3_1, T2[3], V[3], vc3_1),
                (op1_1, T1[4], V[4], vc1_1), (op3_1, T2[4], V[4], vc3_1),
            ]})
            # Cy8: H1 op13 g5-7 (V[5] H0 from Cy5 ✓, V[6..7] H0 from Cy6 ✓)
            self.instrs.append({"valu": [
                (op1_1, T1[5], V[5], vc1_1), (op3_1, T2[5], V[5], vc3_1),
                (op1_1, T1[6], V[6], vc1_1), (op3_1, T2[6], V[6], vc3_1),
                (op1_1, T1[7], V[7], vc1_1), (op3_1, T2[7], V[7], vc3_1),
            ]})
            # Cy9: H1 op2 g0-5 (T1/T2[0..1] from Cy6 ✓, [2..4] from Cy7 ✓, [5] from Cy8... wait)
            # T1/T2[5] written in Cy8, read in Cy9 - that's next cycle, OK
            self.instrs.append({"valu": [
                (op2_1, V[0], T1[0], T2[0]), (op2_1, V[1], T1[1], T2[1]),
                (op2_1, V[2], T1[2], T2[2]), (op2_1, V[3], T1[3], T2[3]),
                (op2_1, V[4], T1[4], T2[4]), (op2_1, V[5], T1[5], T2[5]),
            ]})
            # Cy10: H1 op2 g6-7 + H2 madd g0-3 (T1/T2[6..7] from Cy8 ✓, V[0..3] from Cy9 ✓)
            self.instrs.append({"valu": [
                (op2_1, V[6], T1[6], T2[6]), (op2_1, V[7], T1[7], T2[7]),
                ("multiply_add", V[0], V[0], vc_mul2, vc_add2),
                ("multiply_add", V[1], V[1], vc_mul2, vc_add2),
                ("multiply_add", V[2], V[2], vc_mul2, vc_add2),
                ("multiply_add", V[3], V[3], vc_mul2, vc_add2),
            ]})
            # Cy11: H2 madd g4-7 (V[4..5] from Cy9 ✓, V[6..7] from Cy10 ✓)
            self.instrs.append({"valu": [
                ("multiply_add", V[4], V[4], vc_mul2, vc_add2),
                ("multiply_add", V[5], V[5], vc_mul2, vc_add2),
                ("multiply_add", V[6], V[6], vc_mul2, vc_add2),
                ("multiply_add", V[7], V[7], vc_mul2, vc_add2),
            ]})
            # Cy12: H3 op13 g0-2
            self.instrs.append({"valu": [
                (op1_3, T1[0], V[0], vc1_3), (op3_3, T2[0], V[0], vc3_3),
                (op1_3, T1[1], V[1], vc1_3), (op3_3, T2[1], V[1], vc3_3),
                (op1_3, T1[2], V[2], vc1_3), (op3_3, T2[2], V[2], vc3_3),
            ]})
            # Cy13: H3 op13 g3-5
            self.instrs.append({"valu": [
                (op1_3, T1[3], V[3], vc1_3), (op3_3, T2[3], V[3], vc3_3),
                (op1_3, T1[4], V[4], vc1_3), (op3_3, T2[4], V[4], vc3_3),
                (op1_3, T1[5], V[5], vc1_3), (op3_3, T2[5], V[5], vc3_3),
            ]})
            # Cy14: H3 op13 g6-7 + H3 op2 g0-1
            self.instrs.append({"valu": [
                (op1_3, T1[6], V[6], vc1_3), (op3_3, T2[6], V[6], vc3_3),
                (op1_3, T1[7], V[7], vc1_3), (op3_3, T2[7], V[7], vc3_3),
                (op2_3, V[0], T1[0], T2[0]), (op2_3, V[1], T1[1], T2[1]),
            ]})
            # Cy15: H3 op2 g2-7
            self.instrs.append({"valu": [
                (op2_3, V[2], T1[2], T2[2]), (op2_3, V[3], T1[3], T2[3]),
                (op2_3, V[4], T1[4], T2[4]), (op2_3, V[5], T1[5], T2[5]),
                (op2_3, V[6], T1[6], T2[6]), (op2_3, V[7], T1[7], T2[7]),
            ]})
            # Cy16: H4 madd g0-5
            self.instrs.append({"valu": [
                ("multiply_add", V[0], V[0], vc_mul4, vc_add4),
                ("multiply_add", V[1], V[1], vc_mul4, vc_add4),
                ("multiply_add", V[2], V[2], vc_mul4, vc_add4),
                ("multiply_add", V[3], V[3], vc_mul4, vc_add4),
                ("multiply_add", V[4], V[4], vc_mul4, vc_add4),
                ("multiply_add", V[5], V[5], vc_mul4, vc_add4),
            ]})
            # Cy17: H4 madd g6-7 + H5 op13 g0-1
            self.instrs.append({"valu": [
                ("multiply_add", V[6], V[6], vc_mul4, vc_add4),
                ("multiply_add", V[7], V[7], vc_mul4, vc_add4),
                (op1_5, T1[0], V[0], vc1_5), (op3_5, T2[0], V[0], vc3_5),
                (op1_5, T1[1], V[1], vc1_5), (op3_5, T2[1], V[1], vc3_5),
            ]})
            # Cy18: H5 op13 g2-4
            self.instrs.append({"valu": [
                (op1_5, T1[2], V[2], vc1_5), (op3_5, T2[2], V[2], vc3_5),
                (op1_5, T1[3], V[3], vc1_5), (op3_5, T2[3], V[3], vc3_5),
                (op1_5, T1[4], V[4], vc1_5), (op3_5, T2[4], V[4], vc3_5),
            ]})
            # Cy19: H5 op13 g5-7
            self.instrs.append({"valu": [
                (op1_5, T1[5], V[5], vc1_5), (op3_5, T2[5], V[5], vc3_5),
                (op1_5, T1[6], V[6], vc1_5), (op3_5, T2[6], V[6], vc3_5),
                (op1_5, T1[7], V[7], vc1_5), (op3_5, T2[7], V[7], vc3_5),
            ]})
            # Cy20: H5 op2 g0-5
            self.instrs.append({"valu": [
                (op2_5, V[0], T1[0], T2[0]), (op2_5, V[1], T1[1], T2[1]),
                (op2_5, V[2], T1[2], T2[2]), (op2_5, V[3], T1[3], T2[3]),
                (op2_5, V[4], T1[4], T2[4]), (op2_5, V[5], T1[5], T2[5]),
            ]})
            # Cy21: H5 op2 g6-7 + idx madd g0-1, & g0-1
            self.instrs.append({"valu": [
                (op2_5, V[6], T1[6], T2[6]), (op2_5, V[7], T1[7], T2[7]),
                ("multiply_add", T2[0], I[0], v_two, v_one), ("&", T1[0], V[0], v_one),
                ("multiply_add", T2[1], I[1], v_two, v_one), ("&", T1[1], V[1], v_one),
            ]})
            # Cy22: idx madd+& g2-4
            self.instrs.append({"valu": [
                ("multiply_add", T2[2], I[2], v_two, v_one), ("&", T1[2], V[2], v_one),
                ("multiply_add", T2[3], I[3], v_two, v_one), ("&", T1[3], V[3], v_one),
                ("multiply_add", T2[4], I[4], v_two, v_one), ("&", T1[4], V[4], v_one),
            ]})
            # Cy23: idx madd+& g5-6, + g0-1
            self.instrs.append({"valu": [
                ("multiply_add", T2[5], I[5], v_two, v_one), ("&", T1[5], V[5], v_one),
                ("multiply_add", T2[6], I[6], v_two, v_one), ("&", T1[6], V[6], v_one),
                ("+", I[0], T2[0], T1[0]), ("+", I[1], T2[1], T1[1]),
            ]})
            # Cy24: idx madd+& g7, + g2-5
            self.instrs.append({"valu": [
                ("multiply_add", T2[7], I[7], v_two, v_one), ("&", T1[7], V[7], v_one),
                ("+", I[2], T2[2], T1[2]), ("+", I[3], T2[3], T1[3]),
                ("+", I[4], T2[4], T1[4]), ("+", I[5], T2[5], T1[5]),
            ]})
            if wrap:
                # Cy25: + g6-7, < g0-3
                self.instrs.append({"valu": [
                    ("+", I[6], T2[6], T1[6]), ("+", I[7], T2[7], T1[7]),
                    ("<", T1[0], I[0], v_n_nodes), ("<", T1[1], I[1], v_n_nodes),
                    ("<", T1[2], I[2], v_n_nodes), ("<", T1[3], I[3], v_n_nodes),
                ]})
                # Cy26: < g4-7, wrap g0-1
                self.instrs.append({"valu": [
                    ("<", T1[4], I[4], v_n_nodes), ("<", T1[5], I[5], v_n_nodes),
                    ("<", T1[6], I[6], v_n_nodes), ("<", T1[7], I[7], v_n_nodes),
                    ("*", I[0], I[0], T1[0]), ("*", I[1], I[1], T1[1]),
                ]})
                # Cy27: wrap g2-7
                self.instrs.append({"valu": [
                    ("*", I[2], I[2], T1[2]), ("*", I[3], I[3], T1[3]),
                    ("*", I[4], I[4], T1[4]), ("*", I[5], I[5], T1[5]),
                    ("*", I[6], I[6], T1[6]), ("*", I[7], I[7], T1[7]),
                ]})
            else:
                # Cy25: + g6-7 only (no wrap needed)
                self.instrs.append({"valu": [
                    ("+", I[6], T2[6], T1[6]), ("+", I[7], T2[7], T1[7]),
                ]})

        def emit_compute_iter_d1_pipelined_g8(it, next_it_idx_addrs):
            """Fused d1 selection + compute + addr calc + scattered loads for next round.
            Overlaps the addr calc and scattered loads with the d1 compute.
            Saves ~40 cycles vs separate d1_compute + addr_calc + scattered_loads."""
            cur_idx, cur_val = get_groups(it)
            V = cur_val
            I = cur_idx
            fvp = self.scratch["forest_values_p"]

            # Stage shortcuts
            vc_mul0 = self.hash_vconsts[(0, "mul")]
            vc_add0 = self.hash_vconsts[(0, "add")]
            op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
            vc1_1 = self.hash_vconsts[(1, 1)]
            vc3_1 = self.hash_vconsts[(1, 3)]
            vc_mul2 = self.hash_vconsts[(2, "mul")]
            vc_add2 = self.hash_vconsts[(2, "add")]
            op1_3, _, op2_3, op3_3, _ = HASH_STAGES[3]
            vc1_3 = self.hash_vconsts[(3, 1)]
            vc3_3 = self.hash_vconsts[(3, 3)]
            vc_mul4 = self.hash_vconsts[(4, "mul")]
            vc_add4 = self.hash_vconsts[(4, "add")]
            op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
            vc1_5 = self.hash_vconsts[(5, 1)]
            vc3_5 = self.hash_vconsts[(5, 3)]
            T1, T2 = v_tmp1, v_tmp2

            # Scattered load queue: 64 loads for next iteration
            nlq = [(g, i) for g in range(G) for i in range(VLEN)]
            li = 0

            def next_loads(n=2):
                nonlocal li
                ops = []
                for _ in range(n):
                    if li < len(nlq):
                        g, i = nlq[li]
                        ops.append(("load", v_node[g] + i, next_node_addrs[g][i]))
                        li += 1
                return ops

            # Addr calc queue: 64 addr calcs for next iteration
            alq = [(g, i) for g in range(G) for i in range(VLEN)]
            ai = 0

            def next_addr_calcs(n=12):
                nonlocal ai
                ops = []
                for _ in range(n):
                    if ai < len(alq):
                        g, i = alq[ai]
                        ops.append(("+", next_node_addrs[g][i], fvp, next_it_idx_addrs[g] + i))
                        ai += 1
                return ops

            # Cy1: mask g0-5 + addr calc
            self.instrs.append({
                "valu": [("&", T1[g], I[g], v_one) for g in range(6)],
                "alu": next_addr_calcs(12),
            })
            # Cy2: mask g6-7 + madd g0-3 + addr calc
            self.instrs.append({
                "valu": [
                    ("&", T1[6], I[6], v_one), ("&", T1[7], I[7], v_one),
                    ("multiply_add", v_node[0], v_tree_d1_diff, T1[0], v_tree[2]),
                    ("multiply_add", v_node[1], v_tree_d1_diff, T1[1], v_tree[2]),
                    ("multiply_add", v_node[2], v_tree_d1_diff, T1[2], v_tree[2]),
                    ("multiply_add", v_node[3], v_tree_d1_diff, T1[3], v_tree[2]),
                ],
                "alu": next_addr_calcs(12),
            })
            # Cy3: madd g4-7 + XOR g0-1 + addr calc + first loads
            self.instrs.append({
                "valu": [
                    ("multiply_add", v_node[4], v_tree_d1_diff, T1[4], v_tree[2]),
                    ("multiply_add", v_node[5], v_tree_d1_diff, T1[5], v_tree[2]),
                    ("multiply_add", v_node[6], v_tree_d1_diff, T1[6], v_tree[2]),
                    ("multiply_add", v_node[7], v_tree_d1_diff, T1[7], v_tree[2]),
                    ("^", V[0], V[0], v_node[0]),
                    ("^", V[1], V[1], v_node[1]),
                ],
                "alu": next_addr_calcs(12),
                "load": next_loads(),
            })
            # Cy4: XOR g2-5 + H0 madd g0-1 + addr calc + loads
            self.instrs.append({
                "valu": [
                    ("^", V[2], V[2], v_node[2]), ("^", V[3], V[3], v_node[3]),
                    ("^", V[4], V[4], v_node[4]), ("^", V[5], V[5], v_node[5]),
                    ("multiply_add", V[0], V[0], vc_mul0, vc_add0),
                    ("multiply_add", V[1], V[1], vc_mul0, vc_add0),
                ],
                "alu": next_addr_calcs(12),
                "load": next_loads(),
            })
            # Cy5: XOR g6-7 + H0 madd g2-5 + addr calc + loads
            self.instrs.append({
                "valu": [
                    ("^", V[6], V[6], v_node[6]), ("^", V[7], V[7], v_node[7]),
                    ("multiply_add", V[2], V[2], vc_mul0, vc_add0),
                    ("multiply_add", V[3], V[3], vc_mul0, vc_add0),
                    ("multiply_add", V[4], V[4], vc_mul0, vc_add0),
                    ("multiply_add", V[5], V[5], vc_mul0, vc_add0),
                ],
                "alu": next_addr_calcs(12),
                "load": next_loads(),
            })
            # Cy6: H0 madd g6-7 + H1 op13 g0-1 + remaining addr calc + loads
            remaining_ac = next_addr_calcs(12)
            instr6 = {
                "valu": [
                    ("multiply_add", V[6], V[6], vc_mul0, vc_add0),
                    ("multiply_add", V[7], V[7], vc_mul0, vc_add0),
                    (op1_1, T1[0], V[0], vc1_1), (op3_1, T2[0], V[0], vc3_1),
                    (op1_1, T1[1], V[1], vc1_1), (op3_1, T2[1], V[1], vc3_1),
                ],
                "load": next_loads(),
            }
            if remaining_ac:
                instr6["alu"] = remaining_ac
            self.instrs.append(instr6)
            # Cy7: H1 op13 g2-4 + loads
            self.instrs.append({
                "valu": [
                    (op1_1, T1[2], V[2], vc1_1), (op3_1, T2[2], V[2], vc3_1),
                    (op1_1, T1[3], V[3], vc1_1), (op3_1, T2[3], V[3], vc3_1),
                    (op1_1, T1[4], V[4], vc1_1), (op3_1, T2[4], V[4], vc3_1),
                ],
                "load": next_loads(),
            })
            # Cy8: H1 op13 g5-7 + loads
            self.instrs.append({
                "valu": [
                    (op1_1, T1[5], V[5], vc1_1), (op3_1, T2[5], V[5], vc3_1),
                    (op1_1, T1[6], V[6], vc1_1), (op3_1, T2[6], V[6], vc3_1),
                    (op1_1, T1[7], V[7], vc1_1), (op3_1, T2[7], V[7], vc3_1),
                ],
                "load": next_loads(),
            })
            # Cy9: H1 op2 g0-5 + loads
            self.instrs.append({
                "valu": [
                    (op2_1, V[0], T1[0], T2[0]), (op2_1, V[1], T1[1], T2[1]),
                    (op2_1, V[2], T1[2], T2[2]), (op2_1, V[3], T1[3], T2[3]),
                    (op2_1, V[4], T1[4], T2[4]), (op2_1, V[5], T1[5], T2[5]),
                ],
                "load": next_loads(),
            })
            # Cy10: H1 op2 g6-7 + H2 madd g0-3 + loads
            self.instrs.append({
                "valu": [
                    (op2_1, V[6], T1[6], T2[6]), (op2_1, V[7], T1[7], T2[7]),
                    ("multiply_add", V[0], V[0], vc_mul2, vc_add2),
                    ("multiply_add", V[1], V[1], vc_mul2, vc_add2),
                    ("multiply_add", V[2], V[2], vc_mul2, vc_add2),
                    ("multiply_add", V[3], V[3], vc_mul2, vc_add2),
                ],
                "load": next_loads(),
            })
            # Cy11: H2 madd g4-7 + loads
            self.instrs.append({
                "valu": [
                    ("multiply_add", V[4], V[4], vc_mul2, vc_add2),
                    ("multiply_add", V[5], V[5], vc_mul2, vc_add2),
                    ("multiply_add", V[6], V[6], vc_mul2, vc_add2),
                    ("multiply_add", V[7], V[7], vc_mul2, vc_add2),
                ],
                "load": next_loads(),
            })
            # Cy12: H3 op13 g0-2 + loads
            self.instrs.append({
                "valu": [
                    (op1_3, T1[0], V[0], vc1_3), (op3_3, T2[0], V[0], vc3_3),
                    (op1_3, T1[1], V[1], vc1_3), (op3_3, T2[1], V[1], vc3_3),
                    (op1_3, T1[2], V[2], vc1_3), (op3_3, T2[2], V[2], vc3_3),
                ],
                "load": next_loads(),
            })
            # Cy13: H3 op13 g3-5 + loads
            self.instrs.append({
                "valu": [
                    (op1_3, T1[3], V[3], vc1_3), (op3_3, T2[3], V[3], vc3_3),
                    (op1_3, T1[4], V[4], vc1_3), (op3_3, T2[4], V[4], vc3_3),
                    (op1_3, T1[5], V[5], vc1_3), (op3_3, T2[5], V[5], vc3_3),
                ],
                "load": next_loads(),
            })
            # Cy14: H3 op13 g6-7 + H3 op2 g0-1 + loads
            self.instrs.append({
                "valu": [
                    (op1_3, T1[6], V[6], vc1_3), (op3_3, T2[6], V[6], vc3_3),
                    (op1_3, T1[7], V[7], vc1_3), (op3_3, T2[7], V[7], vc3_3),
                    (op2_3, V[0], T1[0], T2[0]), (op2_3, V[1], T1[1], T2[1]),
                ],
                "load": next_loads(),
            })
            # Cy15: H3 op2 g2-7 + loads
            self.instrs.append({
                "valu": [
                    (op2_3, V[2], T1[2], T2[2]), (op2_3, V[3], T1[3], T2[3]),
                    (op2_3, V[4], T1[4], T2[4]), (op2_3, V[5], T1[5], T2[5]),
                    (op2_3, V[6], T1[6], T2[6]), (op2_3, V[7], T1[7], T2[7]),
                ],
                "load": next_loads(),
            })
            # Cy16: H4 madd g0-5 + loads
            self.instrs.append({
                "valu": [
                    ("multiply_add", V[0], V[0], vc_mul4, vc_add4),
                    ("multiply_add", V[1], V[1], vc_mul4, vc_add4),
                    ("multiply_add", V[2], V[2], vc_mul4, vc_add4),
                    ("multiply_add", V[3], V[3], vc_mul4, vc_add4),
                    ("multiply_add", V[4], V[4], vc_mul4, vc_add4),
                    ("multiply_add", V[5], V[5], vc_mul4, vc_add4),
                ],
                "load": next_loads(),
            })
            # Cy17: H4 madd g6-7 + H5 op13 g0-1 + loads
            self.instrs.append({
                "valu": [
                    ("multiply_add", V[6], V[6], vc_mul4, vc_add4),
                    ("multiply_add", V[7], V[7], vc_mul4, vc_add4),
                    (op1_5, T1[0], V[0], vc1_5), (op3_5, T2[0], V[0], vc3_5),
                    (op1_5, T1[1], V[1], vc1_5), (op3_5, T2[1], V[1], vc3_5),
                ],
                "load": next_loads(),
            })
            # Cy18: H5 op13 g2-4 + loads
            self.instrs.append({
                "valu": [
                    (op1_5, T1[2], V[2], vc1_5), (op3_5, T2[2], V[2], vc3_5),
                    (op1_5, T1[3], V[3], vc1_5), (op3_5, T2[3], V[3], vc3_5),
                    (op1_5, T1[4], V[4], vc1_5), (op3_5, T2[4], V[4], vc3_5),
                ],
                "load": next_loads(),
            })
            # Cy19: H5 op13 g5-7 + loads
            self.instrs.append({
                "valu": [
                    (op1_5, T1[5], V[5], vc1_5), (op3_5, T2[5], V[5], vc3_5),
                    (op1_5, T1[6], V[6], vc1_5), (op3_5, T2[6], V[6], vc3_5),
                    (op1_5, T1[7], V[7], vc1_5), (op3_5, T2[7], V[7], vc3_5),
                ],
                "load": next_loads(),
            })
            # Cy20: H5 op2 g0-5 + loads
            self.instrs.append({
                "valu": [
                    (op2_5, V[0], T1[0], T2[0]), (op2_5, V[1], T1[1], T2[1]),
                    (op2_5, V[2], T1[2], T2[2]), (op2_5, V[3], T1[3], T2[3]),
                    (op2_5, V[4], T1[4], T2[4]), (op2_5, V[5], T1[5], T2[5]),
                ],
                "load": next_loads(),
            })
            # Cy21: H5 op2 g6-7 + idx madd g0-1, & g0-1 + loads
            self.instrs.append({
                "valu": [
                    (op2_5, V[6], T1[6], T2[6]), (op2_5, V[7], T1[7], T2[7]),
                    ("multiply_add", T2[0], I[0], v_two, v_one), ("&", T1[0], V[0], v_one),
                    ("multiply_add", T2[1], I[1], v_two, v_one), ("&", T1[1], V[1], v_one),
                ],
                "load": next_loads(),
            })
            # Cy22: idx madd+& g2-4 + loads
            self.instrs.append({
                "valu": [
                    ("multiply_add", T2[2], I[2], v_two, v_one), ("&", T1[2], V[2], v_one),
                    ("multiply_add", T2[3], I[3], v_two, v_one), ("&", T1[3], V[3], v_one),
                    ("multiply_add", T2[4], I[4], v_two, v_one), ("&", T1[4], V[4], v_one),
                ],
                "load": next_loads(),
            })
            # Cy23: idx madd+& g5-6, + g0-1 + loads
            self.instrs.append({
                "valu": [
                    ("multiply_add", T2[5], I[5], v_two, v_one), ("&", T1[5], V[5], v_one),
                    ("multiply_add", T2[6], I[6], v_two, v_one), ("&", T1[6], V[6], v_one),
                    ("+", I[0], T2[0], T1[0]), ("+", I[1], T2[1], T1[1]),
                ],
                "load": next_loads(),
            })
            # Cy24: idx madd+& g7, + g2-5 + loads
            self.instrs.append({
                "valu": [
                    ("multiply_add", T2[7], I[7], v_two, v_one), ("&", T1[7], V[7], v_one),
                    ("+", I[2], T2[2], T1[2]), ("+", I[3], T2[3], T1[3]),
                    ("+", I[4], T2[4], T1[4]), ("+", I[5], T2[5], T1[5]),
                ],
                "load": next_loads(),
            })
            # Cy25: + g6-7, < g0-3 + loads
            self.instrs.append({
                "valu": [
                    ("+", I[6], T2[6], T1[6]), ("+", I[7], T2[7], T1[7]),
                    ("<", T1[0], I[0], v_n_nodes), ("<", T1[1], I[1], v_n_nodes),
                    ("<", T1[2], I[2], v_n_nodes), ("<", T1[3], I[3], v_n_nodes),
                ],
                "load": next_loads(),
            })
            # Cy26: < g4-7, wrap g0-1 + loads
            self.instrs.append({
                "valu": [
                    ("<", T1[4], I[4], v_n_nodes), ("<", T1[5], I[5], v_n_nodes),
                    ("<", T1[6], I[6], v_n_nodes), ("<", T1[7], I[7], v_n_nodes),
                    ("*", I[0], I[0], T1[0]), ("*", I[1], I[1], T1[1]),
                ],
                "load": next_loads(),
            })
            # Cy27: wrap g2-7 + remaining loads
            ld = next_loads()
            instr27 = {"valu": [
                ("*", I[2], I[2], T1[2]), ("*", I[3], I[3], T1[3]),
                ("*", I[4], I[4], T1[4]), ("*", I[5], I[5], T1[5]),
                ("*", I[6], I[6], T1[6]), ("*", I[7], I[7], T1[7]),
            ]}
            if ld: instr27["load"] = ld
            self.instrs.append(instr27)

            # Remaining loads after compute
            while li < len(nlq):
                ld = next_loads()
                if ld:
                    self.instrs.append({"load": ld})

        # --- Broadcast support: preload tree values for depths 0-1 ---
        MAX_BC_DEPTH = 1  # broadcast for depths 0-1
        # Preload tree[0], tree[1], tree[2] as scalar values
        tree_cache = [self.alloc_scratch(f"tc_{i}") for i in range(3)]
        # tree[0] is at forest_values_p + 0
        self.instrs.append({"load": [
            ("load", tree_cache[0], self.scratch["forest_values_p"]),
        ]})
        # tree[1] at forest_values_p+1, tree[2] at forest_values_p+2
        tc_addr1 = self.alloc_scratch("tc_a1")
        tc_addr2 = self.alloc_scratch("tc_a2")
        self.instrs.append({"alu": [
            ("+", tc_addr1, self.scratch["forest_values_p"], one_const),
            ("+", tc_addr2, self.scratch["forest_values_p"], two_const),
        ]})
        self.instrs.append({"load": [
            ("load", tree_cache[1], tc_addr1),
            ("load", tree_cache[2], tc_addr2),
        ]})

        # Broadcast vector constants for tree values
        v_tree = [self.alloc_scratch(f"v_tree_{i}", VLEN) for i in range(3)]
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree[0], tree_cache[0]),
            ("vbroadcast", v_tree[1], tree_cache[1]),
            ("vbroadcast", v_tree[2], tree_cache[2]),
        ]})

        # Precompute diff and base for depth-1 selection
        # tree[1] if idx is odd (idx&1=1), tree[2] if idx is even (idx&1=0)
        # result = (tree[1]-tree[2]) * (idx&1) + tree[2] = multiply_add(diff, mask, tree2)
        v_tree_d1_diff = self.alloc_scratch("v_td1_diff", VLEN)
        self.instrs.append({"valu": [
            ("-", v_tree_d1_diff, v_tree[1], v_tree[2]),
        ]})

        def get_round_depth(round_idx):
            return round_idx % (forest_height + 1)

        def is_broadcast_round(round_idx):
            return get_round_depth(round_idx) <= MAX_BC_DEPTH

        def emit_broadcast_tree_value(depth, idx_addrs):
            """Set v_node[g] = tree value based on idx, using broadcast selection."""
            if depth == 0:
                # All elements access tree[0]
                for g in range(0, G, 6):
                    end = min(g + 6, G)
                    self.instrs.append({"valu": [
                        ("vbroadcast", v_node[gg], tree_cache[0]) for gg in range(g, end)
                    ]})
            elif depth == 1:
                # tree[1] if idx&1==1, tree[2] if idx&1==0
                # v_node = multiply_add(v_tree_d1_diff, idx&1, v_tree[2])
                # First compute masks, then multiply_add
                for g in range(0, G, 6):
                    end = min(g + 6, G)
                    self.instrs.append({"valu": [
                        ("&", v_tmp1[gg], idx_addrs[gg], v_one) for gg in range(g, end)
                    ]})
                for g in range(0, G, 6):
                    end = min(g + 6, G)
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node[gg], v_tree_d1_diff, v_tmp1[gg], v_tree[2]) for gg in range(g, end)
                    ]})

        # ================================================================
        # MAIN LOOP: 16 rounds × 4 iterations, with broadcast for d0-d1
        # ================================================================
        need_tree_load = True  # whether v_node needs to be loaded for current iter

        for round_idx in range(rounds):
            depth = get_round_depth(round_idx)
            bc = depth <= MAX_BC_DEPTH

            for it in range(n_iters):
                is_very_last = (round_idx == rounds - 1 and it == n_iters - 1)
                cur_idx, cur_val = get_groups(it)

                # Determine next iteration
                next_it = it + 1
                next_round = round_idx
                if next_it >= n_iters:
                    next_it = 0
                    next_round = round_idx + 1
                next_bc = (next_round < rounds and
                           get_round_depth(next_round) <= MAX_BC_DEPTH)

                if bc:
                    # Check if this is a d1 transition to scattered round
                    is_d1_transition = (depth == 1 and not is_very_last
                                        and not next_bc and next_it == 0)
                    if depth == 0:
                        # d0: XOR directly with v_tree[0], skip vbroadcast + wrap
                        emit_compute_iter_d0_g8(it, wrap=False, idx_zero=True)
                    elif is_d1_transition:
                        # d1 transition: fused selection + compute + addr calc + loads
                        next_idx, _ = get_groups(next_it)
                        emit_compute_iter_d1_pipelined_g8(it, next_idx)
                    else:
                        # d1: fused selection + compute (no transition, no wrap)
                        emit_compute_iter_d1_g8(it, wrap=False)
                    # If next iteration needs scattered loads (d0 case only now)
                    if not is_d1_transition and not is_very_last and not next_bc and next_it == 0:
                        # Transitioning from broadcast to scattered round (d0 case)
                        next_idx, _ = get_groups(next_it)
                        emit_node_addr_calc_g8(next_idx, next_node_addrs)
                        emit_scattered_loads_g8(next_node_addrs, v_node)
                elif need_tree_load:
                    # First scattered iter: load then compute (with pipeline to next)
                    emit_node_addr_calc_g8(cur_idx, node_addrs)
                    emit_scattered_loads_g8(node_addrs, v_node)
                    if is_very_last or (next_bc and next_it == 0):
                        emit_compute_iter_g8(it, prep_store=is_very_last)
                    else:
                        # Pipeline compute with loading next iteration
                        next_idx, _ = get_groups(next_it)
                        emit_pipelined_iter_g8(it, next_idx)
                elif is_very_last:
                    emit_compute_iter_g8(it, prep_store=True)
                else:
                    # Pipelined scattered iteration
                    next_idx, _ = get_groups(next_it)
                    if next_bc and next_it == 0:
                        # Next round is broadcast, don't load tree values
                        emit_compute_iter_g8(it)
                    else:
                        emit_pipelined_iter_g8(it, next_idx)

                # Track whether next iteration needs tree loading
                if bc:
                    need_tree_load = False  # broadcast rounds manage their own
                elif it == 0 and not bc:
                    need_tree_load = False  # we just did the initial load
                else:
                    need_tree_load = False  # pipeline handles it

        # --- Final store: write all idx and val back to memory (interleaved) ---
        # Build list of all store operations: (base_ptr_name, data_addrs) pairs
        store_ops = []
        for g in range(0, n_groups, 2):
            store_ops.append(("inp_values_p", g, all_val[g], all_val[g + 1]))
        for g in range(0, n_groups, 2):
            store_ops.append(("inp_indices_p", g, all_idx[g], all_idx[g + 1]))

        # First store uses pre-computed addresses from last compute iteration
        # (store_addr1, store_addr2 were computed during prep_store)
        _, _, d0a, d0b = store_ops[0]

        # Compute next pair addresses
        ptr1, g1, _, _ = store_ops[1]
        off1_1 = self.scratch_const(g1 * VLEN)
        off2_1 = self.scratch_const((g1 + 1) * VLEN)
        self.instrs.append({
            "store": [("vstore", store_addr1, d0a), ("vstore", store_addr2, d0b)],
            "alu": [
                ("+", tmp1, self.scratch[ptr1], off1_1),
                ("+", tmp2, self.scratch[ptr1], off2_1),
            ],
        })
        # Interleaved: store pair N + ALU pair N+1 (starting from pair 1)
        for si in range(1, len(store_ops) - 1):
            _, _, da, db = store_ops[si]
            ptr_next, g_next, _, _ = store_ops[si + 1]
            off1_n = self.scratch_const(g_next * VLEN)
            off2_n = self.scratch_const((g_next + 1) * VLEN)
            self.instrs.append({
                "store": [("vstore", tmp1, da), ("vstore", tmp2, db)],
                "alu": [
                    ("+", tmp1, self.scratch[ptr_next], off1_n),
                    ("+", tmp2, self.scratch[ptr_next], off2_n),
                ],
            })
        # Last: store final pair
        _, _, da_last, db_last = store_ops[-1]
        self.instrs.append({"store": [
            ("vstore", tmp1, da_last), ("vstore", tmp2, db_last),
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
