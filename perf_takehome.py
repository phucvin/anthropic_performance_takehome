"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

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


class Scheduler:
    def __init__(self, slot_limits):
        self.slot_limits = slot_limits
        self.instrs = []
        self.reg_ready = defaultdict(int)
        self.reg_last_read = defaultdict(int)
        self.max_cycle = -1

    def schedule(self, slots):
        for engine, slot in slots:
            self.schedule_slot(engine, slot)
        return self.instrs

    def schedule_slot(self, engine, slot):
        if engine == "debug":
             # Skip debug instructions
             return

        inputs, outputs = self.get_rw_regs(engine, slot)

        min_cycle = 0
        for reg in inputs:
            min_cycle = max(min_cycle, self.reg_ready[reg])
        for reg in outputs:
            min_cycle = max(min_cycle, self.reg_last_read[reg])
            # WAW dependency check: must wait for previous write to complete
            min_cycle = max(min_cycle, self.reg_ready[reg])

        op = slot[0]
        if op in ("pause", "halt"):
            min_cycle = max(min_cycle, self.max_cycle)

        cycle = min_cycle
        while True:
            if cycle >= len(self.instrs):
                self.instrs.append(defaultdict(list))

            instr = self.instrs[cycle]
            if len(instr.get(engine, [])) < self.slot_limits.get(engine, 0):
                instr[engine].append(slot)
                break
            cycle += 1

        self.max_cycle = max(self.max_cycle, cycle)

        for reg in outputs:
            self.reg_ready[reg] = cycle + 1
        for reg in inputs:
            self.reg_last_read[reg] = cycle

    def get_rw_regs(self, engine, slot):
        inputs = []
        outputs = []

        def vec(start):
            return list(range(start, start + VLEN))

        op = slot[0]
        args = slot[1:]

        if engine == "alu":
            # (op, dest, src1, src2)
            outputs.append(args[0])
            inputs.append(args[1])
            inputs.append(args[2])
        elif engine == "valu":
            if op == "vbroadcast":
                outputs.extend(vec(args[0]))
                inputs.append(args[1])
            elif op == "multiply_add":
                outputs.extend(vec(args[0]))
                inputs.extend(vec(args[1]))
                inputs.extend(vec(args[2]))
                inputs.extend(vec(args[3]))
            else:
                outputs.extend(vec(args[0]))
                inputs.extend(vec(args[1]))
                inputs.extend(vec(args[2]))
        elif engine == "load":
            if op == "load":
                outputs.append(args[0])
                inputs.append(args[1])
            elif op == "load_offset":
                outputs.append(args[0] + args[2])
                inputs.append(args[1] + args[2])
            elif op == "vload":
                outputs.extend(vec(args[0]))
                inputs.append(args[1])
            elif op == "const":
                outputs.append(args[0])
        elif engine == "store":
            if op == "store":
                inputs.append(args[0])
                inputs.append(args[1])
            elif op == "vstore":
                inputs.append(args[0])
                inputs.extend(vec(args[1]))
        elif engine == "flow":
            if op == "select":
                outputs.append(args[0])
                inputs.append(args[1])
                inputs.append(args[2])
                inputs.append(args[3])
            elif op == "vselect":
                outputs.extend(vec(args[0]))
                inputs.extend(vec(args[1]))
                inputs.extend(vec(args[2]))
                inputs.extend(vec(args[3]))
            elif op == "add_imm":
                outputs.append(args[0])
                inputs.append(args[1])
            elif op == "cond_jump" or op == "cond_jump_rel":
                inputs.append(args[0])

        return inputs, outputs

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        scheduler = Scheduler(SLOT_LIMITS)
        return scheduler.schedule(slots)

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

    def scratch_const_vec(self, val, name=None):
        if val not in self.vec_const_map:
            s_addr = self.scratch_const(val)
            v_addr = self.alloc_scratch(name, VLEN)
            self.add("valu", ("vbroadcast", v_addr, s_addr))
            self.vec_const_map[val] = v_addr
        return self.vec_const_map[val]

    def build_hash_vec(self, val_hash_addr, tmp1, tmp2):
        slots = []
        # Pre-broadcast hash constants
        consts_vec = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
             if val1 not in consts_vec: consts_vec[val1] = self.scratch_const_vec(val1)
             if val3 not in consts_vec: consts_vec[val3] = self.scratch_const_vec(val3)
             # Also need shift amounts as vectors for <<, >>
             if op1 in ("<<", ">>") and val1 not in consts_vec: consts_vec[val1] = self.scratch_const_vec(val1)
             if op3 in ("<<", ">>") and val3 not in consts_vec: consts_vec[val3] = self.scratch_const_vec(val3)

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("valu", (op1, tmp1, val_hash_addr, consts_vec[val1])))
            slots.append(("valu", (op3, tmp2, val_hash_addr, consts_vec[val3])))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # Initial loads
        tmp1 = self.alloc_scratch("tmp1")
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Vector constants
        v_zero = self.scratch_const_vec(0)
        v_one = self.scratch_const_vec(1)
        v_two = self.scratch_const_vec(2)
        v_n_nodes = self.scratch_const_vec(n_nodes)

        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        # Load cached forest nodes (levels 0 to 3)
        # 1+2+4+8 = 15 nodes.
        CACHE_LEVELS = 0

        forest_scratch_base = self.scratch_ptr
        total_cached_nodes = max(0, 2**(CACHE_LEVELS) - 1)
        self.alloc_scratch("cached_forest", total_cached_nodes)

        forest_ptr = self.alloc_scratch("forest_ptr")
        self.add("flow", ("add_imm", forest_ptr, self.scratch["forest_values_p"], 0))

        # Load in chunks of 8
        for i in range(0, total_cached_nodes, VLEN):
             self.add("load", ("vload", forest_scratch_base + i, forest_ptr))
             self.add("flow", ("add_imm", forest_ptr, forest_ptr, VLEN))

        if total_cached_nodes > 0:
            self.alloc_scratch("padding", (VLEN - (total_cached_nodes % VLEN)) % VLEN)

        self.add("flow", ("pause",))

        body = []

        # Hoisted arrays
        hoisted_idx = [self.alloc_scratch(f"h_idx_{i}", VLEN) for i in range(0, batch_size, VLEN)]
        hoisted_val = [self.alloc_scratch(f"h_val_{i}", VLEN) for i in range(0, batch_size, VLEN)]

        curr_indices_p = self.alloc_scratch("curr_indices_p")
        curr_values_p = self.alloc_scratch("curr_values_p")

        # Hoist Loads
        body.append(("flow", ("add_imm", curr_indices_p, self.scratch["inp_indices_p"], 0)))
        body.append(("flow", ("add_imm", curr_values_p, self.scratch["inp_values_p"], 0)))
        for i in range(len(hoisted_idx)):
             body.append(("load", ("vload", hoisted_idx[i], curr_indices_p)))
             body.append(("load", ("vload", hoisted_val[i], curr_values_p)))
             body.append(("flow", ("add_imm", curr_indices_p, curr_indices_p, VLEN)))
             body.append(("flow", ("add_imm", curr_values_p, curr_values_p, VLEN)))

        # Pre-allocate unique temps per chunk (reusing sets to save scratch)
        NUM_SETS = 3
        temp_sets = []
        for s in range(NUM_SETS):
             t = {}
             t['v_node_val'] = self.alloc_scratch(f"v_node_val_s{s}", VLEN)
             t['v_addr'] = self.alloc_scratch(f"v_addr_s{s}", VLEN)
             t['v_tmp1'] = self.alloc_scratch(f"v_tmp1_s{s}", VLEN)
             t['v_tmp2'] = self.alloc_scratch(f"v_tmp2_s{s}", VLEN)
             t['v_tmp3'] = self.alloc_scratch(f"v_tmp3_s{s}", VLEN)
             t['v_offset'] = self.alloc_scratch(f"v_offset_s{s}", VLEN)
             t['v_bits'] = [self.alloc_scratch(f"v_bit_{d}_s{s}", VLEN) for d in range(CACHE_LEVELS)]

             t_tree = []
             for d in range(CACHE_LEVELS):
                 layer_width = 2**(CACHE_LEVELS - 1 - d)
                 layer = [self.alloc_scratch(f"tree_tmp_{d}_{j}_s{s}", VLEN) for j in range(layer_width)]
                 t_tree.append(layer)
             t['tree_temps'] = t_tree
             temp_sets.append(t)

        chunk_temps = [temp_sets[i % NUM_SETS] for i in range(len(hoisted_idx))]

        # Pre-allocate broadcast temps
        max_level_nodes = int(2**(CACHE_LEVELS - 1)) if CACHE_LEVELS > 0 else 0
        broadcast_temps = [self.alloc_scratch(f"bc_tmp_{n}", VLEN) for n in range(max_level_nodes)]

        for round in range(rounds):
            level = round % (forest_height + 1)
            is_cached = level < CACHE_LEVELS

            if is_cached:
                 start_node = 2**level - 1
                 v_start_node = self.scratch_const_vec(start_node)
                 level_base = forest_scratch_base + start_node

                 level_vecs = []
                 for n in range(2**level):
                     v = broadcast_temps[n]
                     body.append(("valu", ("vbroadcast", v, level_base + n)))
                     level_vecs.append(v)

            for i in range(len(hoisted_idx)):
                v_idx_curr = hoisted_idx[i]
                v_val_curr = hoisted_val[i]
                temps = chunk_temps[i]
                v_node_val = temps['v_node_val']
                v_addr = temps['v_addr']
                v_tmp1 = temps['v_tmp1']
                v_tmp2 = temps['v_tmp2']
                v_tmp3 = temps['v_tmp3']

                if is_cached:
                    v_offset = temps['v_offset']
                    v_bits = temps['v_bits']
                    tree_temps = temps['tree_temps']
                    # offset = idx - start_node
                    body.append(("valu", ("-", v_offset, v_idx_curr, v_start_node)))

                    # Extract bits?
                    # vselect(cond, a, b). cond != 0 -> a.
                    # bits: bit 0 selects between pairs.
                    # bit k selects at layer k.
                    # We need bits 0..level-1.
                    # bit j = (offset >> j) & 1.

                    # Since we do this for each batch, we compute bits for each batch.
                    # But we can reuse v_bits scratch?

                    # Tree construction
                    current_layer = level_vecs
                    for d in range(level):
                        # Extract bit d
                        v_bit = v_bits[d]
                        # bit = (offset >> d) & 1
                        # We need constant vector for d? No, shift by constant.
                        # shift vector?
                        # valu ">>", dest, src, shift_amt (vector).
                        # We need shift_amt vector.
                        v_shift = self.scratch_const_vec(d)
                        body.append(("valu", (">>", v_bit, v_offset, v_shift)))
                        # & 1
                        body.append(("valu", ("&", v_bit, v_bit, v_one)))

                        next_layer = []
                        for j in range(len(current_layer)//2):
                            left = current_layer[2*j+1]
                            right = current_layer[2*j]
                            res = tree_temps[d][j]
                            body.append(("flow", ("vselect", res, v_bit, left, right)))
                            next_layer.append(res)
                        current_layer = next_layer

                    # Final result is current_layer[0]
                    # Copy to v_node_val? Or alias?
                    # We can move it. Or just use it in Hash.
                    # But Hash expects v_node_val.
                    # Copy: valu + 0?
                    # Or just:
                    body.append(("valu", ("+", v_node_val, current_layer[0], v_zero)))

                else:
                    body.append(("valu", ("+", v_addr, v_forest_base, v_idx_curr)))
                    for k in range(VLEN):
                        body.append(("load", ("load_offset", v_node_val, v_addr, k)))

                # Hash
                body.append(("valu", ("^", v_val_curr, v_val_curr, v_node_val)))
                body.extend(self.build_hash_vec(v_val_curr, v_tmp1, v_tmp2))

                # Update idx
                body.append(("valu", ("&", v_tmp1, v_val_curr, v_one)))
                body.append(("valu", ("==", v_tmp1, v_tmp1, v_zero)))
                body.append(("flow", ("vselect", v_tmp3, v_tmp1, v_one, v_two)))
                body.append(("valu", ("*", v_idx_curr, v_idx_curr, v_two)))
                body.append(("valu", ("+", v_idx_curr, v_idx_curr, v_tmp3)))

                # Wrap
                body.append(("valu", ("<", v_tmp1, v_idx_curr, v_n_nodes)))
                body.append(("flow", ("vselect", v_idx_curr, v_tmp1, v_idx_curr, v_zero)))

        # Store Back
        body.append(("flow", ("add_imm", curr_indices_p, self.scratch["inp_indices_p"], 0)))
        body.append(("flow", ("add_imm", curr_values_p, self.scratch["inp_values_p"], 0)))
        for i in range(len(hoisted_idx)):
             body.append(("store", ("vstore", curr_indices_p, hoisted_idx[i])))
             body.append(("store", ("vstore", curr_values_p, hoisted_val[i])))
             body.append(("flow", ("add_imm", curr_indices_p, curr_indices_p, VLEN)))
             body.append(("flow", ("add_imm", curr_values_p, curr_values_p, VLEN)))

        body.append(("flow", ("pause",)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)

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
