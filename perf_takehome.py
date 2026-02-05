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
        Optimized SIMD implementation with software pipelining.
        Processes 16 elements at a time (2 groups of 8).
        Overlaps memory loads with computation using double buffering.
        """
        self.vconst_inits = []

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
        # Each set handles 2 groups (A and B) = 16 elements
        v_idx = [
            [self.alloc_scratch(f"v_idx_{s}a", VLEN), self.alloc_scratch(f"v_idx_{s}b", VLEN)]
            for s in range(2)
        ]
        v_val = [
            [self.alloc_scratch(f"v_val_{s}a", VLEN), self.alloc_scratch(f"v_val_{s}b", VLEN)]
            for s in range(2)
        ]
        v_node_val = [
            [self.alloc_scratch(f"v_node_{s}a", VLEN), self.alloc_scratch(f"v_node_{s}b", VLEN)]
            for s in range(2)
        ]
        v_tmp1 = [
            [self.alloc_scratch(f"v_tmp1_{s}a", VLEN), self.alloc_scratch(f"v_tmp1_{s}b", VLEN)]
            for s in range(2)
        ]
        v_tmp2 = [
            [self.alloc_scratch(f"v_tmp2_{s}a", VLEN), self.alloc_scratch(f"v_tmp2_{s}b", VLEN)]
            for s in range(2)
        ]
        v_tmp3 = [
            [self.alloc_scratch(f"v_tmp3_{s}a", VLEN), self.alloc_scratch(f"v_tmp3_{s}b", VLEN)]
            for s in range(2)
        ]
        v_cond = [
            [self.alloc_scratch(f"v_cond_{s}a", VLEN), self.alloc_scratch(f"v_cond_{s}b", VLEN)]
            for s in range(2)
        ]

        # Scalar addresses for each buffer
        idx_addr = [[self.alloc_scratch(f"idx_addr_{s}a"), self.alloc_scratch(f"idx_addr_{s}b")] for s in range(2)]
        val_addr = [[self.alloc_scratch(f"val_addr_{s}a"), self.alloc_scratch(f"val_addr_{s}b")] for s in range(2)]

        # Node addresses for scattered loads (one set, reused)
        node_addrs_a = [self.alloc_scratch(f"node_addr_a{i}") for i in range(VLEN)]
        node_addrs_b = [self.alloc_scratch(f"node_addr_b{i}") for i in range(VLEN)]

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
        self.add("debug", ("comment", "Starting pipelined SIMD loop"))

        n_groups = batch_size // VLEN  # 32 groups of 8
        n_iters = n_groups // 2  # 16 dual-group iterations per round

        def emit_load_phase(s, offset_a, offset_b):
            """Emit load instructions for buffer s."""
            offset_const_a = self.scratch_const(offset_a)
            offset_const_b = self.scratch_const(offset_b)

            # Calculate addresses
            self.instrs.append({"alu": [
                ("+", idx_addr[s][0], self.scratch["inp_indices_p"], offset_const_a),
                ("+", val_addr[s][0], self.scratch["inp_values_p"], offset_const_a),
                ("+", idx_addr[s][1], self.scratch["inp_indices_p"], offset_const_b),
                ("+", val_addr[s][1], self.scratch["inp_values_p"], offset_const_b),
            ]})

            # Load indices and values
            self.instrs.append({"load": [
                ("vload", v_idx[s][0], idx_addr[s][0]),
                ("vload", v_val[s][0], val_addr[s][0]),
            ]})
            self.instrs.append({"load": [
                ("vload", v_idx[s][1], idx_addr[s][1]),
                ("vload", v_val[s][1], val_addr[s][1]),
            ]})

            # Compute node addresses
            self.instrs.append({"alu": [
                ("+", node_addrs_a[i], self.scratch["forest_values_p"], v_idx[s][0] + i)
                for i in range(VLEN)
            ] + [
                ("+", node_addrs_b[i], self.scratch["forest_values_p"], v_idx[s][1] + i)
                for i in range(4)
            ]})
            self.instrs.append({"alu": [
                ("+", node_addrs_b[i], self.scratch["forest_values_p"], v_idx[s][1] + i)
                for i in range(4, VLEN)
            ]})

            # Load node values
            for i in range(0, VLEN, 2):
                self.instrs.append({"load": [
                    ("load", v_node_val[s][0] + i, node_addrs_a[i]),
                    ("load", v_node_val[s][0] + i + 1, node_addrs_a[i + 1]),
                ]})
            for i in range(0, VLEN, 2):
                self.instrs.append({"load": [
                    ("load", v_node_val[s][1] + i, node_addrs_b[i]),
                    ("load", v_node_val[s][1] + i + 1, node_addrs_b[i + 1]),
                ]})

        def emit_compute_phase(s):
            """Emit compute instructions for buffer s."""
            va, vb = 0, 1  # indices into the group arrays

            # XOR
            self.instrs.append({"valu": [
                ("^", v_val[s][va], v_val[s][va], v_node_val[s][va]),
                ("^", v_val[s][vb], v_val[s][vb], v_node_val[s][vb]),
            ]})

            # Hash: 6 stages
            for hi in range(6):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                vc1 = self.hash_vconsts[(hi, 1)]
                vc3 = self.hash_vconsts[(hi, 3)]
                self.instrs.append({"valu": [
                    (op1, v_tmp1[s][va], v_val[s][va], vc1),
                    (op3, v_tmp2[s][va], v_val[s][va], vc3),
                    (op1, v_tmp1[s][vb], v_val[s][vb], vc1),
                    (op3, v_tmp2[s][vb], v_val[s][vb], vc3),
                ]})
                self.instrs.append({"valu": [
                    (op2, v_val[s][va], v_tmp1[s][va], v_tmp2[s][va]),
                    (op2, v_val[s][vb], v_tmp1[s][vb], v_tmp2[s][vb]),
                ]})

            # Index calculation
            self.instrs.append({"valu": [
                ("%", v_tmp1[s][va], v_val[s][va], v_two),
                ("*", v_idx[s][va], v_idx[s][va], v_two),
                ("%", v_tmp1[s][vb], v_val[s][vb], v_two),
                ("*", v_idx[s][vb], v_idx[s][vb], v_two),
            ]})
            self.instrs.append({"valu": [
                ("==", v_cond[s][va], v_tmp1[s][va], v_zero),
                ("==", v_cond[s][vb], v_tmp1[s][vb], v_zero),
            ]})
            self.instrs.append({"flow": [("vselect", v_tmp3[s][va], v_cond[s][va], v_one, v_two)]})
            self.instrs.append({"flow": [("vselect", v_tmp3[s][vb], v_cond[s][vb], v_one, v_two)]})
            self.instrs.append({"valu": [
                ("+", v_idx[s][va], v_idx[s][va], v_tmp3[s][va]),
                ("+", v_idx[s][vb], v_idx[s][vb], v_tmp3[s][vb]),
            ]})

            # Wrap indices
            self.instrs.append({"valu": [
                ("<", v_cond[s][va], v_idx[s][va], v_n_nodes),
                ("<", v_cond[s][vb], v_idx[s][vb], v_n_nodes),
            ]})
            self.instrs.append({"flow": [("vselect", v_idx[s][va], v_cond[s][va], v_idx[s][va], v_zero)]})
            self.instrs.append({"flow": [("vselect", v_idx[s][vb], v_cond[s][vb], v_idx[s][vb], v_zero)]})

            # Store
            self.instrs.append({"store": [
                ("vstore", idx_addr[s][va], v_idx[s][va]),
                ("vstore", val_addr[s][va], v_val[s][va]),
            ]})
            self.instrs.append({"store": [
                ("vstore", idx_addr[s][vb], v_idx[s][vb]),
                ("vstore", val_addr[s][vb], v_val[s][vb]),
            ]})

        def emit_pipelined_compute_with_load(compute_s, load_s, load_offset_a, load_offset_b):
            """Emit compute for buffer compute_s while loading into buffer load_s."""
            va, vb = 0, 1
            load_offset_const_a = self.scratch_const(load_offset_a)
            load_offset_const_b = self.scratch_const(load_offset_b)

            # XOR + calculate load addresses
            self.instrs.append({
                "valu": [
                    ("^", v_val[compute_s][va], v_val[compute_s][va], v_node_val[compute_s][va]),
                    ("^", v_val[compute_s][vb], v_val[compute_s][vb], v_node_val[compute_s][vb]),
                ],
                "alu": [
                    ("+", idx_addr[load_s][0], self.scratch["inp_indices_p"], load_offset_const_a),
                    ("+", val_addr[load_s][0], self.scratch["inp_values_p"], load_offset_const_a),
                    ("+", idx_addr[load_s][1], self.scratch["inp_indices_p"], load_offset_const_b),
                    ("+", val_addr[load_s][1], self.scratch["inp_values_p"], load_offset_const_b),
                ],
            })

            # Hash stage 0 + vload indices/values for load buffer
            hi = 0
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            vc1 = self.hash_vconsts[(hi, 1)]
            vc3 = self.hash_vconsts[(hi, 3)]
            self.instrs.append({
                "valu": [
                    (op1, v_tmp1[compute_s][va], v_val[compute_s][va], vc1),
                    (op3, v_tmp2[compute_s][va], v_val[compute_s][va], vc3),
                    (op1, v_tmp1[compute_s][vb], v_val[compute_s][vb], vc1),
                    (op3, v_tmp2[compute_s][vb], v_val[compute_s][vb], vc3),
                ],
                "load": [
                    ("vload", v_idx[load_s][0], idx_addr[load_s][0]),
                    ("vload", v_val[load_s][0], val_addr[load_s][0]),
                ],
            })
            self.instrs.append({
                "valu": [
                    (op2, v_val[compute_s][va], v_tmp1[compute_s][va], v_tmp2[compute_s][va]),
                    (op2, v_val[compute_s][vb], v_tmp1[compute_s][vb], v_tmp2[compute_s][vb]),
                ],
                "load": [
                    ("vload", v_idx[load_s][1], idx_addr[load_s][1]),
                    ("vload", v_val[load_s][1], val_addr[load_s][1]),
                ],
            })

            # Hash stage 1 + compute node addresses for load buffer
            hi = 1
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            vc1 = self.hash_vconsts[(hi, 1)]
            vc3 = self.hash_vconsts[(hi, 3)]
            self.instrs.append({
                "valu": [
                    (op1, v_tmp1[compute_s][va], v_val[compute_s][va], vc1),
                    (op3, v_tmp2[compute_s][va], v_val[compute_s][va], vc3),
                    (op1, v_tmp1[compute_s][vb], v_val[compute_s][vb], vc1),
                    (op3, v_tmp2[compute_s][vb], v_val[compute_s][vb], vc3),
                ],
                "alu": [
                    ("+", node_addrs_a[i], self.scratch["forest_values_p"], v_idx[load_s][0] + i)
                    for i in range(VLEN)
                ] + [
                    ("+", node_addrs_b[i], self.scratch["forest_values_p"], v_idx[load_s][1] + i)
                    for i in range(4)
                ],
            })
            self.instrs.append({
                "valu": [
                    (op2, v_val[compute_s][va], v_tmp1[compute_s][va], v_tmp2[compute_s][va]),
                    (op2, v_val[compute_s][vb], v_tmp1[compute_s][vb], v_tmp2[compute_s][vb]),
                ],
                "alu": [
                    ("+", node_addrs_b[i], self.scratch["forest_values_p"], v_idx[load_s][1] + i)
                    for i in range(4, VLEN)
                ],
            })

            # Hash stages 2-5 + load node values
            node_load_idx = 0
            for hi in range(2, 6):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                vc1 = self.hash_vconsts[(hi, 1)]
                vc3 = self.hash_vconsts[(hi, 3)]

                # Part 1: hash ops + 2 node loads
                load_ops = []
                if node_load_idx < VLEN:
                    load_ops = [
                        ("load", v_node_val[load_s][0] + node_load_idx, node_addrs_a[node_load_idx]),
                        ("load", v_node_val[load_s][0] + node_load_idx + 1, node_addrs_a[node_load_idx + 1]),
                    ]
                    node_load_idx += 2

                instr = {
                    "valu": [
                        (op1, v_tmp1[compute_s][va], v_val[compute_s][va], vc1),
                        (op3, v_tmp2[compute_s][va], v_val[compute_s][va], vc3),
                        (op1, v_tmp1[compute_s][vb], v_val[compute_s][vb], vc1),
                        (op3, v_tmp2[compute_s][vb], v_val[compute_s][vb], vc3),
                    ],
                }
                if load_ops:
                    instr["load"] = load_ops
                self.instrs.append(instr)

                # Part 2: hash combine + 2 more node loads
                load_ops = []
                if node_load_idx < VLEN:
                    load_ops = [
                        ("load", v_node_val[load_s][0] + node_load_idx, node_addrs_a[node_load_idx]),
                        ("load", v_node_val[load_s][0] + node_load_idx + 1, node_addrs_a[node_load_idx + 1]),
                    ]
                    node_load_idx += 2

                instr = {
                    "valu": [
                        (op2, v_val[compute_s][va], v_tmp1[compute_s][va], v_tmp2[compute_s][va]),
                        (op2, v_val[compute_s][vb], v_tmp1[compute_s][vb], v_tmp2[compute_s][vb]),
                    ],
                }
                if load_ops:
                    instr["load"] = load_ops
                self.instrs.append(instr)

            # Continue with remaining node loads for group B during index calculation
            node_load_idx_b = 0

            # mod/mul + load node_val_b[0:2]
            self.instrs.append({
                "valu": [
                    ("%", v_tmp1[compute_s][va], v_val[compute_s][va], v_two),
                    ("*", v_idx[compute_s][va], v_idx[compute_s][va], v_two),
                    ("%", v_tmp1[compute_s][vb], v_val[compute_s][vb], v_two),
                    ("*", v_idx[compute_s][vb], v_idx[compute_s][vb], v_two),
                ],
                "load": [
                    ("load", v_node_val[load_s][1] + 0, node_addrs_b[0]),
                    ("load", v_node_val[load_s][1] + 1, node_addrs_b[1]),
                ],
            })

            # equality + load node_val_b[2:4]
            self.instrs.append({
                "valu": [
                    ("==", v_cond[compute_s][va], v_tmp1[compute_s][va], v_zero),
                    ("==", v_cond[compute_s][vb], v_tmp1[compute_s][vb], v_zero),
                ],
                "load": [
                    ("load", v_node_val[load_s][1] + 2, node_addrs_b[2]),
                    ("load", v_node_val[load_s][1] + 3, node_addrs_b[3]),
                ],
            })

            # vselect + load node_val_b[4:6]
            self.instrs.append({
                "flow": [("vselect", v_tmp3[compute_s][va], v_cond[compute_s][va], v_one, v_two)],
                "load": [
                    ("load", v_node_val[load_s][1] + 4, node_addrs_b[4]),
                    ("load", v_node_val[load_s][1] + 5, node_addrs_b[5]),
                ],
            })

            # vselect + load node_val_b[6:8]
            self.instrs.append({
                "flow": [("vselect", v_tmp3[compute_s][vb], v_cond[compute_s][vb], v_one, v_two)],
                "load": [
                    ("load", v_node_val[load_s][1] + 6, node_addrs_b[6]),
                    ("load", v_node_val[load_s][1] + 7, node_addrs_b[7]),
                ],
            })

            # add
            self.instrs.append({"valu": [
                ("+", v_idx[compute_s][va], v_idx[compute_s][va], v_tmp3[compute_s][va]),
                ("+", v_idx[compute_s][vb], v_idx[compute_s][vb], v_tmp3[compute_s][vb]),
            ]})

            # compare
            self.instrs.append({"valu": [
                ("<", v_cond[compute_s][va], v_idx[compute_s][va], v_n_nodes),
                ("<", v_cond[compute_s][vb], v_idx[compute_s][vb], v_n_nodes),
            ]})

            # vselect wrap
            self.instrs.append({"flow": [("vselect", v_idx[compute_s][va], v_cond[compute_s][va], v_idx[compute_s][va], v_zero)]})
            self.instrs.append({"flow": [("vselect", v_idx[compute_s][vb], v_cond[compute_s][vb], v_idx[compute_s][vb], v_zero)]})

            # store
            self.instrs.append({"store": [
                ("vstore", idx_addr[compute_s][va], v_idx[compute_s][va]),
                ("vstore", val_addr[compute_s][va], v_val[compute_s][va]),
            ]})
            self.instrs.append({"store": [
                ("vstore", idx_addr[compute_s][vb], v_idx[compute_s][vb]),
                ("vstore", val_addr[compute_s][vb], v_val[compute_s][vb]),
            ]})

        # Main pipelined loop
        # Pipeline strategy:
        # 1. Load iter 0 data into buffer 0
        # 2. For each iteration i (0 to n-2):
        #    - Compute buffer (i%2) while loading buffer ((i+1)%2) with iter i+1 data
        # 3. Compute final buffer

        for round_idx in range(rounds):
            # Prologue: Load first iteration
            emit_load_phase(0, 0, VLEN)

            # Steady state: compute current while loading next
            for it in range(n_iters - 1):
                current_buf = it % 2
                next_buf = 1 - current_buf
                next_offset_a = (it + 1) * 2 * VLEN
                next_offset_b = next_offset_a + VLEN
                emit_pipelined_compute_with_load(current_buf, next_buf, next_offset_a, next_offset_b)

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
