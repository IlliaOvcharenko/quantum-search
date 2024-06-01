import sys, os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import math

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives.sampler import Sampler
from fire import Fire


def run_quantum_search(
    array_len: int | None = None,
    n_grover_iterations: int | None = None,
):
    np.random.seed(42)


    array = np.arange(array_len)
    np.random.shuffle(array)
    item = np.random.choice(array)
    array = array.tolist()

    print(f"Input array: {array}")
    print(f"Item to search: {item}")
    print(f"Correct answer should be: {array.index(item)}")

    n_index_qbits = math.ceil(math.log(len(array), 2))
    n_value_qbits = math.ceil(math.log(max(array), 2))
    to_bits = lambda i: bin(i)[2:][::-1]
    to_int = lambda b: int(b[::-1], 2)

    index_qbits = [i for i in range(n_index_qbits)]
    value_qbits = [i + n_index_qbits for i in range(n_value_qbits)]
    phase_qbit = n_index_qbits + n_value_qbits

    if n_grover_iterations is None:
        n_grover_iterations = math.floor(math.sqrt(len(array)))


    qc = QuantumCircuit(
        QuantumRegister(n_index_qbits, "index"),
        QuantumRegister(n_value_qbits, "value"),
        QuantumRegister(1, "phase"),
        icr := ClassicalRegister(n_index_qbits, "meas-index"),
    )
    qc.h(index_qbits)

    qc.x(phase_qbit)
    qc.h(phase_qbit)

    def add_oracle(global_qc):
        qc = QuantumCircuit(
            QuantumRegister(n_index_qbits, "index"),
            QuantumRegister(n_value_qbits, "value"),
            QuantumRegister(1, "phase"),
        )
        for array_idx in range(len(array)):
            qc.barrier()
            array_idx_bit = to_bits(array_idx)[:n_index_qbits]
            array_idx_bit += "0" * (n_index_qbits - len(array_idx_bit))
            array_item = array[array_idx]
            array_item_bit = to_bits(array_item)[:n_value_qbits]
            array_item_bit += "0" * (n_value_qbits - len(array_item_bit))

            zeros = [index_qbits[i] for i, b in enumerate(array_idx_bit) if b == "0"]
            ones = [value_qbits[i] for i, b in enumerate(array_item_bit) if b == "1"]

            for value_idx in ones:
                if zeros:
                    qc.x(zeros)
                qc.mcx(index_qbits, value_idx)
                if zeros:
                    qc.x(zeros)

        global_qc = global_qc.compose(qc)

        item_bit = to_bits(item)
        item_bit += "0" * (n_value_qbits - len(item_bit))
        zeros = [value_qbits[i] for i, b in enumerate(item_bit) if b == "0"]
        if zeros:
            global_qc.x(zeros)
        global_qc.mcx(value_qbits, phase_qbit)
        if zeros:
            global_qc.x(zeros)

        global_qc = global_qc.compose(qc.inverse())
        return global_qc


    def add_diffuser(global_qc):
        qc = QuantumCircuit(
            QuantumRegister(n_index_qbits, "index"),
            QuantumRegister(n_value_qbits, "value"),
            QuantumRegister(1, "phase"),
        )
        qc.h(index_qbits)
        qc.x(index_qbits)
        qc.mcx(index_qbits, phase_qbit)
        qc.x(index_qbits)
        qc.h(index_qbits)
        global_qc = global_qc.compose(qc)
        return global_qc


    for _ in range(n_grover_iterations):
        qc = add_oracle(qc)
        qc = add_diffuser(qc)

    qc.measure(index_qbits, icr)
    sampler = Sampler()
    job = sampler.run(qc, shots=64)
    result = job.result()
    index_dist = result.quasi_dists[0]


    index_pred = max(index_dist.items(), key=lambda x: x[1])
    print(f"Search result, index: {index_pred[0]}, " \
          f"empirical prob to collapse in this state: {index_pred[1]}")
    print(f"Prob to collapse into a correct state: {index_dist[array.index(item)]}")
    print()
    return index_pred


import psutil
import os
import gc

from datetime import datetime
from functools import partial

class MeasureResult:
    def __init__(self, return_value, func_name = None):
        self.return_value = return_value
        self.measurements = {}
        self.func_name = func_name

    def print(self):
        if self.func_name:
            print(f"func: {self.func_name}")
        for k, v in self.measurements.items():
            print(f"\t{k} - {v}")
        print()

    # def __call__():
    #     return self.func()

def get_func_name(f):
    if isinstance(f, partial):
        return f.func.__name__ + " with args " + str(f.args) + " with kwargs " + str(f.keywords)
    return f.__name__

def define_name(name, measurements, depth=0):
    try_name = name
    if depth != 0:
        try_name += f"_{depth}"

    if try_name in measurements:
        return define_name(name, measurements, depth=depth+1)
    else:
        return try_name

def measure_exec_time(f):
    def nested_func():
        start_t = datetime.now().timestamp()
        rv = f()
        end_t = datetime.now().timestamp()
        exec_time = end_t - start_t

        if not isinstance(rv, MeasureResult):
            rv = MeasureResult(rv, func_name=get_func_name(f))

        measure_name = define_name("exec_time", rv.measurements)
        rv.measurements[measure_name] = exec_time
        return rv

    return nested_func

def measure_ram_usage(f):
    def nested_func():
        pid = os.getpid()
        current_process = psutil.Process(pid)
        start_ram = current_process.memory_info().rss / (1024 * 1024)
        # print(start_ram)
        # exit(0)
        rv = f()
        end_ram = current_process.memory_info().rss / (1024 * 1024)
        ram_usage = end_ram - start_ram

        if not isinstance(rv, MeasureResult):
            rv = MeasureResult(rv, func_name=get_func_name(f))

        measure_name = define_name("ram_usage", rv.measurements)
        rv.measurements[measure_name] = ram_usage
        return rv
    return nested_func

def apply_measurements(f, ms):
    m = ms.pop()
    if not ms:
        return m(f)
    return apply_measurements(m(f), ms)

def print_measure(f):
    def nested_func():
        rv = f()
        if isinstance(rv, MeasureResult):
            rv.print()
        else:
            print(f"there is nothing to print yet :(")
        return rv
    return nested_func

def plot_with_number_of_input_data():
    array_len_param = [v for v in range(10, 71, 10)]
    # array_len_param = [v for v in range(10, 31, 10)]
    print(array_len_param)
    measurements = []

    for al in array_len_param:
        run_quantum_search_with_measurements = apply_measurements(
            partial(run_quantum_search, array_len=al),
            [print_measure, measure_exec_time, measure_ram_usage, measure_exec_time],
        )
        meas = run_quantum_search_with_measurements()
        measurements.append(meas)
        gc.collect()


    plt.figure(figsize=(15, 10))
    plt.plot(array_len_param, [m.measurements["exec_time"] for m in measurements], "-*")
    plt.xlabel("N, input array len")
    plt.ylabel("execution time, s")
    plt.grid()
    plt.savefig(f"figs/n-exec-time.png", bbox_inches="tight")

    plt.figure(figsize=(15, 10))
    plt.plot(array_len_param, [m.measurements["ram_usage"] for m in measurements], "-*")
    plt.xlabel("N, input array len")
    plt.ylabel("RAM usage, s")
    plt.grid()
    plt.savefig(f"figs/n-ram-usage.png", bbox_inches="tight")
    # for m in measurements:
    #     m.print()
    # run_quantum_search_with_measurements = apply_measurements(
    #     partial(run_quantum_search, array_len=10),
    #     [measure_exec_time, measure_ram_usage, measure_exec_time],
    # )
    # m = run_quantum_search_with_measurements()
    # m.print()

def plot_with_number_of_grover_iter():
    array_len = 32
    n_iter_param = [v for v in range(1, 8+1)]
    # n_iter_param = [v for v in range(10)]

    pred = []
    confidence = []

    for ni in n_iter_param:
        print(f"num of iter: {ni}")
        result = run_quantum_search(array_len=array_len, n_grover_iterations=ni)
        pred.append(result[0])
        confidence.append(result[1])
    print(pred)
    print(confidence)

    plt.figure(figsize=(15, 10))
    plt.plot(n_iter_param, confidence, "-*")
    plt.xlabel("number of grover iterations")
    plt.ylabel("prob to collapse in a correct state")
    plt.axvline(math.floor(math.sqrt(array_len)), c="red")
    plt.grid()
    plt.savefig(f"figs/grover-iter-prob.png", bbox_inches="tight")


def plot_linear_vs_sqrt():
    x = np.arange(100)
    plt.figure(figsize=(15, 10))
    plt.plot(x, x, "-", label="linear search", c="b")
    plt.plot(x, [math.floor(math.sqrt(v)) for v in x], "-", label="quantum search", c="r")
    plt.xlabel("number of input data")
    plt.ylabel("number of required iterations")
    plt.legend()
    plt.grid()
    plt.savefig(f"figs/linear-vs-sqrt.png", bbox_inches="tight")


def main():
    # plot_with_number_of_input_data()
    # plot_with_number_of_grover_iter()
    plot_linear_vs_sqrt()


if __name__ == "__main__":
    Fire(main)
