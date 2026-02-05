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
        Optimized SIMD implementation processing 16 elements at a time (2 groups of 8).
        Uses VLIW packing to maximize parallelism across engines.
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

        # Vector registers for two groups (A and B)
        v_idx_a = self.alloc_scratch("v_idx_a", VLEN)
        v_val_a = self.alloc_scratch("v_val_a", VLEN)
        v_node_val_a = self.alloc_scratch("v_node_val_a", VLEN)
        v_tmp1_a = self.alloc_scratch("v_tmp1_a", VLEN)
        v_tmp2_a = self.alloc_scratch("v_tmp2_a", VLEN)
        v_tmp3_a = self.alloc_scratch("v_tmp3_a", VLEN)
        v_cond_a = self.alloc_scratch("v_cond_a", VLEN)

        v_idx_b = self.alloc_scratch("v_idx_b", VLEN)
        v_val_b = self.alloc_scratch("v_val_b", VLEN)
        v_node_val_b = self.alloc_scratch("v_node_val_b", VLEN)
        v_tmp1_b = self.alloc_scratch("v_tmp1_b", VLEN)
        v_tmp2_b = self.alloc_scratch("v_tmp2_b", VLEN)
        v_tmp3_b = self.alloc_scratch("v_tmp3_b", VLEN)
        v_cond_b = self.alloc_scratch("v_cond_b", VLEN)

        # Scalar addresses
        idx_addr_a = self.alloc_scratch("idx_addr_a")
        val_addr_a = self.alloc_scratch("val_addr_a")
        idx_addr_b = self.alloc_scratch("idx_addr_b")
        val_addr_b = self.alloc_scratch("val_addr_b")

        # Node addresses for scattered loads
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
        self.add("debug", ("comment", "Starting SIMD loop"))

        n_groups = batch_size // VLEN  # 32 groups of 8

        for round_idx in range(rounds):
            for group in range(0, n_groups, 2):
                offset_a = group * VLEN
                offset_b = (group + 1) * VLEN
                offset_const_a = self.scratch_const(offset_a)
                offset_const_b = self.scratch_const(offset_b)

                # === LOAD PHASE ===
                # Calculate all 4 addresses
                self.instrs.append({"alu": [
                    ("+", idx_addr_a, self.scratch["inp_indices_p"], offset_const_a),
                    ("+", val_addr_a, self.scratch["inp_values_p"], offset_const_a),
                    ("+", idx_addr_b, self.scratch["inp_indices_p"], offset_const_b),
                    ("+", val_addr_b, self.scratch["inp_values_p"], offset_const_b),
                ]})

                # Load indices and values for both groups
                self.instrs.append({"load": [
                    ("vload", v_idx_a, idx_addr_a),
                    ("vload", v_val_a, val_addr_a),
                ]})
                self.instrs.append({"load": [
                    ("vload", v_idx_b, idx_addr_b),
                    ("vload", v_val_b, val_addr_b),
                ]})

                # Compute all 16 node addresses (12 per cycle max)
                self.instrs.append({"alu": [
                    ("+", node_addrs_a[i], self.scratch["forest_values_p"], v_idx_a + i)
                    for i in range(VLEN)
                ] + [
                    ("+", node_addrs_b[i], self.scratch["forest_values_p"], v_idx_b + i)
                    for i in range(4)
                ]})
                self.instrs.append({"alu": [
                    ("+", node_addrs_b[i], self.scratch["forest_values_p"], v_idx_b + i)
                    for i in range(4, VLEN)
                ]})

                # Load all 16 node values (2 per cycle)
                for i in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", v_node_val_a + i, node_addrs_a[i]),
                        ("load", v_node_val_a + i + 1, node_addrs_a[i + 1]),
                    ]})
                for i in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", v_node_val_b + i, node_addrs_b[i]),
                        ("load", v_node_val_b + i + 1, node_addrs_b[i + 1]),
                    ]})

                # === COMPUTE PHASE ===
                # XOR both groups
                self.instrs.append({"valu": [
                    ("^", v_val_a, v_val_a, v_node_val_a),
                    ("^", v_val_b, v_val_b, v_node_val_b),
                ]})

                # Hash: 6 stages, each needs 2 cycles
                for hi in range(6):
                    op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                    vc1 = self.hash_vconsts[(hi, 1)]
                    vc3 = self.hash_vconsts[(hi, 3)]
                    # Part 1: parallel ops for both groups
                    self.instrs.append({"valu": [
                        (op1, v_tmp1_a, v_val_a, vc1),
                        (op3, v_tmp2_a, v_val_a, vc3),
                        (op1, v_tmp1_b, v_val_b, vc1),
                        (op3, v_tmp2_b, v_val_b, vc3),
                    ]})
                    # Part 2: combine
                    self.instrs.append({"valu": [
                        (op2, v_val_a, v_tmp1_a, v_tmp2_a),
                        (op2, v_val_b, v_tmp1_b, v_tmp2_b),
                    ]})

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                self.instrs.append({"valu": [
                    ("%", v_tmp1_a, v_val_a, v_two),
                    ("*", v_idx_a, v_idx_a, v_two),
                    ("%", v_tmp1_b, v_val_b, v_two),
                    ("*", v_idx_b, v_idx_b, v_two),
                ]})
                self.instrs.append({"valu": [
                    ("==", v_cond_a, v_tmp1_a, v_zero),
                    ("==", v_cond_b, v_tmp1_b, v_zero),
                ]})
                self.instrs.append({"flow": [("vselect", v_tmp3_a, v_cond_a, v_one, v_two)]})
                self.instrs.append({"flow": [("vselect", v_tmp3_b, v_cond_b, v_one, v_two)]})
                self.instrs.append({"valu": [
                    ("+", v_idx_a, v_idx_a, v_tmp3_a),
                    ("+", v_idx_b, v_idx_b, v_tmp3_b),
                ]})

                # Wrap indices
                self.instrs.append({"valu": [
                    ("<", v_cond_a, v_idx_a, v_n_nodes),
                    ("<", v_cond_b, v_idx_b, v_n_nodes),
                ]})
                self.instrs.append({"flow": [("vselect", v_idx_a, v_cond_a, v_idx_a, v_zero)]})
                self.instrs.append({"flow": [("vselect", v_idx_b, v_cond_b, v_idx_b, v_zero)]})

                # Store results
                self.instrs.append({"store": [
                    ("vstore", idx_addr_a, v_idx_a),
                    ("vstore", val_addr_a, v_val_a),
                ]})
                self.instrs.append({"store": [
                    ("vstore", idx_addr_b, v_idx_b),
                    ("vstore", val_addr_b, v_val_b),
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
