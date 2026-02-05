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
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            self.hash_vconsts[(hi, 1)] = self.get_vconst(val1)
            self.hash_vconsts[(hi, 3)] = self.get_vconst(val3)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Highly optimized SIMD implementation with aggressive pipelining.
        Processes 32 elements at a time (4 groups of 8).
        Eliminates vselect for wrap using multiplication.
        Uses double buffering with deep pipelining to overlap loads and compute.
        """
        self.vconst_inits = []
        G = 4  # Groups per iteration

        # Scalar scratch for initialization
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Header variables
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.instrs.append({"load": [("const", tmp1, i)]})
            self.instrs.append({"load": [("load", self.scratch[v], tmp1)]})

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Double buffering: two sets of vector registers (0 and 1)
        # Each buffer has G groups
        v_idx = [[self.alloc_scratch(f"v_idx_{s}g{g}", VLEN) for g in range(G)] for s in range(2)]
        v_val = [[self.alloc_scratch(f"v_val_{s}g{g}", VLEN) for g in range(G)] for s in range(2)]
        v_node = [[self.alloc_scratch(f"v_node_{s}g{g}", VLEN) for g in range(G)] for s in range(2)]
        v_tmp1 = [[self.alloc_scratch(f"v_tmp1_{s}g{g}", VLEN) for g in range(G)] for s in range(2)]
        v_tmp2 = [[self.alloc_scratch(f"v_tmp2_{s}g{g}", VLEN) for g in range(G)] for s in range(2)]

        # Scalar addresses for each buffer
        idx_addr = [[self.alloc_scratch(f"idx_addr_{s}g{g}") for g in range(G)] for s in range(2)]
        val_addr = [[self.alloc_scratch(f"val_addr_{s}g{g}") for g in range(G)] for s in range(2)]

        # Node addresses for scattered loads (2 sets for double buffering)
        node_addrs = [[[self.alloc_scratch(f"na_{s}g{g}_{i}") for i in range(VLEN)] for g in range(G)] for s in range(2)]

        # Vector constants
        v_zero = self.get_vconst(0)
        v_one = self.get_vconst(1)
        v_two = self.get_vconst(2)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Pre-allocate hash constants
        self.init_hash_vconsts()

        # Initialize vector constants
        for vaddr, saddr in self.vconst_inits:
            self.instrs.append({"valu": [("vbroadcast", vaddr, saddr)]})
        self.instrs.append({"valu": [("vbroadcast", v_n_nodes, self.scratch["n_nodes"])]})

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting 4-group pipelined SIMD loop"))

        n_groups = batch_size // VLEN  # 32 groups of 8
        n_iters = n_groups // G  # 8 iterations per round

        def emit_load_phase(s, base_offset):
            """Emit load instructions for buffer s."""
            offsets = [base_offset + g * VLEN for g in range(G)]
            offset_consts = [self.scratch_const(off) for off in offsets]

            # Calculate addresses (8 ALU ops)
            self.instrs.append({"alu": [
                ("+", idx_addr[s][g], self.scratch["inp_indices_p"], offset_consts[g])
                for g in range(G)
            ] + [
                ("+", val_addr[s][g], self.scratch["inp_values_p"], offset_consts[g])
                for g in range(G)
            ]})

            # Load indices and values (4 vloads, 2/cycle = 2 cycles)
            for g in range(0, G, 2):
                self.instrs.append({"load": [
                    ("vload", v_idx[s][g], idx_addr[s][g]),
                    ("vload", v_idx[s][g+1], idx_addr[s][g+1]),
                ]})
            for g in range(0, G, 2):
                self.instrs.append({"load": [
                    ("vload", v_val[s][g], val_addr[s][g]),
                    ("vload", v_val[s][g+1], val_addr[s][g+1]),
                ]})

            # Compute node addresses (32 ALU ops, 12/cycle = 3 cycles)
            for g in range(G):
                self.instrs.append({"alu": [
                    ("+", node_addrs[s][g][i], self.scratch["forest_values_p"], v_idx[s][g] + i)
                    for i in range(VLEN)
                ]})

            # Load node values (32 scattered loads, 2/cycle = 16 cycles)
            for g in range(G):
                for i in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", v_node[s][g] + i, node_addrs[s][g][i]),
                        ("load", v_node[s][g] + i + 1, node_addrs[s][g][i + 1]),
                    ]})

        def emit_compute_phase(s):
            """Emit compute instructions for buffer s (4 groups)."""
            # XOR: 4 groups (fits in 6 VALU slots)
            self.instrs.append({"valu": [
                ("^", v_val[s][g], v_val[s][g], v_node[s][g]) for g in range(G)
            ]})

            # Hash: 6 stages
            for hi in range(6):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                vc1 = self.hash_vconsts[(hi, 1)]
                vc3 = self.hash_vconsts[(hi, 3)]

                # Part 1: op1 and op3 for 4 groups (8 ops, split into 6+2)
                self.instrs.append({"valu": [
                    (op1, v_tmp1[s][0], v_val[s][0], vc1),
                    (op3, v_tmp2[s][0], v_val[s][0], vc3),
                    (op1, v_tmp1[s][1], v_val[s][1], vc1),
                    (op3, v_tmp2[s][1], v_val[s][1], vc3),
                    (op1, v_tmp1[s][2], v_val[s][2], vc1),
                    (op3, v_tmp2[s][2], v_val[s][2], vc3),
                ]})
                self.instrs.append({"valu": [
                    (op1, v_tmp1[s][3], v_val[s][3], vc1),
                    (op3, v_tmp2[s][3], v_val[s][3], vc3),
                ]})

                # Part 2: combine for 4 groups
                self.instrs.append({"valu": [
                    (op2, v_val[s][g], v_tmp1[s][g], v_tmp2[s][g]) for g in range(G)
                ]})

            # Index calculation: idx = 2*idx + 1 + (val & 1)
            self.instrs.append({"valu": [
                ("&", v_tmp1[s][0], v_val[s][0], v_one),
                ("*", v_idx[s][0], v_idx[s][0], v_two),
                ("&", v_tmp1[s][1], v_val[s][1], v_one),
                ("*", v_idx[s][1], v_idx[s][1], v_two),
                ("&", v_tmp1[s][2], v_val[s][2], v_one),
                ("*", v_idx[s][2], v_idx[s][2], v_two),
            ]})
            self.instrs.append({"valu": [
                ("&", v_tmp1[s][3], v_val[s][3], v_one),
                ("*", v_idx[s][3], v_idx[s][3], v_two),
            ]})
            self.instrs.append({"valu": [
                ("+", v_tmp2[s][g], v_tmp1[s][g], v_one) for g in range(G)
            ]})
            self.instrs.append({"valu": [
                ("+", v_idx[s][g], v_idx[s][g], v_tmp2[s][g]) for g in range(G)
            ]})

            # Wrap using multiply: idx = idx * (idx < n_nodes)
            self.instrs.append({"valu": [
                ("<", v_tmp1[s][g], v_idx[s][g], v_n_nodes) for g in range(G)
            ]})
            self.instrs.append({"valu": [
                ("*", v_idx[s][g], v_idx[s][g], v_tmp1[s][g]) for g in range(G)
            ]})

            # Store (8 vstores, 2/cycle = 4 cycles)
            for g in range(0, G, 2):
                self.instrs.append({"store": [
                    ("vstore", idx_addr[s][g], v_idx[s][g]),
                    ("vstore", idx_addr[s][g+1], v_idx[s][g+1]),
                ]})
            for g in range(0, G, 2):
                self.instrs.append({"store": [
                    ("vstore", val_addr[s][g], v_val[s][g]),
                    ("vstore", val_addr[s][g+1], v_val[s][g+1]),
                ]})

        def emit_pipelined_compute_with_load(compute_s, load_s, load_base_offset):
            """Emit compute for buffer compute_s while loading into buffer load_s."""
            load_offsets = [load_base_offset + g * VLEN for g in range(G)]
            load_offset_consts = [self.scratch_const(off) for off in load_offsets]

            # XOR + calculate load addresses
            self.instrs.append({
                "valu": [("^", v_val[compute_s][g], v_val[compute_s][g], v_node[compute_s][g]) for g in range(G)],
                "alu": [("+", idx_addr[load_s][g], self.scratch["inp_indices_p"], load_offset_consts[g]) for g in range(G)]
                     + [("+", val_addr[load_s][g], self.scratch["inp_values_p"], load_offset_consts[g]) for g in range(G)],
            })

            # Hash stage 0 + vload indices
            hi = 0
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            vc1 = self.hash_vconsts[(hi, 1)]
            vc3 = self.hash_vconsts[(hi, 3)]
            self.instrs.append({
                "valu": [
                    (op1, v_tmp1[compute_s][0], v_val[compute_s][0], vc1),
                    (op3, v_tmp2[compute_s][0], v_val[compute_s][0], vc3),
                    (op1, v_tmp1[compute_s][1], v_val[compute_s][1], vc1),
                    (op3, v_tmp2[compute_s][1], v_val[compute_s][1], vc3),
                    (op1, v_tmp1[compute_s][2], v_val[compute_s][2], vc1),
                    (op3, v_tmp2[compute_s][2], v_val[compute_s][2], vc3),
                ],
                "load": [
                    ("vload", v_idx[load_s][0], idx_addr[load_s][0]),
                    ("vload", v_idx[load_s][1], idx_addr[load_s][1]),
                ],
            })
            self.instrs.append({
                "valu": [
                    (op1, v_tmp1[compute_s][3], v_val[compute_s][3], vc1),
                    (op3, v_tmp2[compute_s][3], v_val[compute_s][3], vc3),
                ],
                "load": [
                    ("vload", v_idx[load_s][2], idx_addr[load_s][2]),
                    ("vload", v_idx[load_s][3], idx_addr[load_s][3]),
                ],
            })
            self.instrs.append({
                "valu": [(op2, v_val[compute_s][g], v_tmp1[compute_s][g], v_tmp2[compute_s][g]) for g in range(G)],
                "load": [
                    ("vload", v_val[load_s][0], val_addr[load_s][0]),
                    ("vload", v_val[load_s][1], val_addr[load_s][1]),
                ],
            })
            self.instrs.append({
                "load": [
                    ("vload", v_val[load_s][2], val_addr[load_s][2]),
                    ("vload", v_val[load_s][3], val_addr[load_s][3]),
                ],
            })

            # Hash stage 1 + compute node addresses (32 ALU ops needed)
            hi = 1
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            vc1 = self.hash_vconsts[(hi, 1)]
            vc3 = self.hash_vconsts[(hi, 3)]
            for g in range(G):
                self.instrs.append({
                    "valu": [
                        (op1, v_tmp1[compute_s][g], v_val[compute_s][g], vc1),
                        (op3, v_tmp2[compute_s][g], v_val[compute_s][g], vc3),
                    ] if g < G else [],
                    "alu": [
                        ("+", node_addrs[load_s][g][i], self.scratch["forest_values_p"], v_idx[load_s][g] + i)
                        for i in range(VLEN)
                    ],
                })
            self.instrs.append({
                "valu": [(op2, v_val[compute_s][g], v_tmp1[compute_s][g], v_tmp2[compute_s][g]) for g in range(G)],
            })

            # Hash stages 2-5 + load node values (32 scattered loads)
            node_load_queue = [(g, i) for g in range(G) for i in range(VLEN)]
            load_idx = 0

            for hi in range(2, 6):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                vc1 = self.hash_vconsts[(hi, 1)]
                vc3 = self.hash_vconsts[(hi, 3)]

                # Part 1a: first 6 hash ops + 2 loads
                load_ops = []
                for _ in range(2):
                    if load_idx < len(node_load_queue):
                        g, i = node_load_queue[load_idx]
                        load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                        load_idx += 1
                self.instrs.append({
                    "valu": [
                        (op1, v_tmp1[compute_s][0], v_val[compute_s][0], vc1),
                        (op3, v_tmp2[compute_s][0], v_val[compute_s][0], vc3),
                        (op1, v_tmp1[compute_s][1], v_val[compute_s][1], vc1),
                        (op3, v_tmp2[compute_s][1], v_val[compute_s][1], vc3),
                        (op1, v_tmp1[compute_s][2], v_val[compute_s][2], vc1),
                        (op3, v_tmp2[compute_s][2], v_val[compute_s][2], vc3),
                    ],
                    "load": load_ops,
                })

                # Part 1b: remaining 2 hash ops + 2 loads
                load_ops = []
                for _ in range(2):
                    if load_idx < len(node_load_queue):
                        g, i = node_load_queue[load_idx]
                        load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                        load_idx += 1
                self.instrs.append({
                    "valu": [
                        (op1, v_tmp1[compute_s][3], v_val[compute_s][3], vc1),
                        (op3, v_tmp2[compute_s][3], v_val[compute_s][3], vc3),
                    ],
                    "load": load_ops,
                })

                # Part 2: combine + 2 loads
                load_ops = []
                for _ in range(2):
                    if load_idx < len(node_load_queue):
                        g, i = node_load_queue[load_idx]
                        load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                        load_idx += 1
                self.instrs.append({
                    "valu": [(op2, v_val[compute_s][g], v_tmp1[compute_s][g], v_tmp2[compute_s][g]) for g in range(G)],
                    "load": load_ops,
                })

            # Index calculation with remaining loads
            load_ops = []
            for _ in range(2):
                if load_idx < len(node_load_queue):
                    g, i = node_load_queue[load_idx]
                    load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                    load_idx += 1
            self.instrs.append({
                "valu": [
                    ("&", v_tmp1[compute_s][0], v_val[compute_s][0], v_one),
                    ("*", v_idx[compute_s][0], v_idx[compute_s][0], v_two),
                    ("&", v_tmp1[compute_s][1], v_val[compute_s][1], v_one),
                    ("*", v_idx[compute_s][1], v_idx[compute_s][1], v_two),
                    ("&", v_tmp1[compute_s][2], v_val[compute_s][2], v_one),
                    ("*", v_idx[compute_s][2], v_idx[compute_s][2], v_two),
                ],
                "load": load_ops,
            })

            load_ops = []
            for _ in range(2):
                if load_idx < len(node_load_queue):
                    g, i = node_load_queue[load_idx]
                    load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                    load_idx += 1
            self.instrs.append({
                "valu": [
                    ("&", v_tmp1[compute_s][3], v_val[compute_s][3], v_one),
                    ("*", v_idx[compute_s][3], v_idx[compute_s][3], v_two),
                ],
                "load": load_ops,
            })

            load_ops = []
            for _ in range(2):
                if load_idx < len(node_load_queue):
                    g, i = node_load_queue[load_idx]
                    load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                    load_idx += 1
            self.instrs.append({
                "valu": [("+", v_tmp2[compute_s][g], v_tmp1[compute_s][g], v_one) for g in range(G)],
                "load": load_ops,
            })

            load_ops = []
            for _ in range(2):
                if load_idx < len(node_load_queue):
                    g, i = node_load_queue[load_idx]
                    load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                    load_idx += 1
            self.instrs.append({
                "valu": [("+", v_idx[compute_s][g], v_idx[compute_s][g], v_tmp2[compute_s][g]) for g in range(G)],
                "load": load_ops,
            })

            # Wrap + remaining loads
            load_ops = []
            for _ in range(2):
                if load_idx < len(node_load_queue):
                    g, i = node_load_queue[load_idx]
                    load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                    load_idx += 1
            self.instrs.append({
                "valu": [("<", v_tmp1[compute_s][g], v_idx[compute_s][g], v_n_nodes) for g in range(G)],
                "load": load_ops,
            })

            load_ops = []
            for _ in range(2):
                if load_idx < len(node_load_queue):
                    g, i = node_load_queue[load_idx]
                    load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                    load_idx += 1
            instr = {"valu": [("*", v_idx[compute_s][g], v_idx[compute_s][g], v_tmp1[compute_s][g]) for g in range(G)]}
            if load_ops:
                instr["load"] = load_ops
            self.instrs.append(instr)

            # Any remaining loads
            while load_idx < len(node_load_queue):
                load_ops = []
                for _ in range(2):
                    if load_idx < len(node_load_queue):
                        g, i = node_load_queue[load_idx]
                        load_ops.append(("load", v_node[load_s][g] + i, node_addrs[load_s][g][i]))
                        load_idx += 1
                if load_ops:
                    self.instrs.append({"load": load_ops})

            # Store
            for g in range(0, G, 2):
                self.instrs.append({"store": [
                    ("vstore", idx_addr[compute_s][g], v_idx[compute_s][g]),
                    ("vstore", idx_addr[compute_s][g+1], v_idx[compute_s][g+1]),
                ]})
            for g in range(0, G, 2):
                self.instrs.append({"store": [
                    ("vstore", val_addr[compute_s][g], v_val[compute_s][g]),
                    ("vstore", val_addr[compute_s][g+1], v_val[compute_s][g+1]),
                ]})

        # Main pipelined loop
        for round_idx in range(rounds):
            # Prologue: Load first iteration
            emit_load_phase(0, 0)

            # Steady state: compute current while loading next
            for it in range(n_iters - 1):
                current_buf = it % 2
                next_buf = 1 - current_buf
                next_offset = (it + 1) * G * VLEN
                emit_pipelined_compute_with_load(current_buf, next_buf, next_offset)

            # Epilogue: compute last iteration (no more loads)
            last_buf = (n_iters - 1) % 2
            emit_compute_phase(last_buf)

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
