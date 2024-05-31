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


def main(
    array_len: int | None = None,
    n_grover_iterations: int | None = None,

    save_circuit_plot: bool = False,
    circuit_plot_prefix: str = "",
):
    np.random.seed(42)

    if array_len is None:
        array = [4, 7, 2, 1, 3, 0, 5, 6]
        # item = 5 # answer should be 6
        # item = 4 # answer should be 0
        item = 0 # answer should be 5
    else:

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

    print(f"N qubits for index: {n_index_qbits}")
    print(f"N qubits for value: {n_value_qbits}")

    if n_grover_iterations is None:
        n_grover_iterations = math.floor(math.sqrt(len(array)))

    print(f"Num of grover iterations: {n_grover_iterations}")

    qc = QuantumCircuit(
        QuantumRegister(n_index_qbits, "index"),
        QuantumRegister(n_value_qbits, "value"),
        QuantumRegister(1, "phase"),
        icr := ClassicalRegister(n_index_qbits, "meas-index"),
    )
    qc.h(index_qbits)
    # qc.x(index_qbits[0])
    # qc.x(index_qbits[1])

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

    if save_circuit_plot:
        circuit_plot_prefix = circuit_plot_prefix + "-" if circuit_plot_prefix != "" \
                              else circuit_plot_prefix
        qc.draw("mpl")
        plt.savefig(f"figs/{circuit_plot_prefix}qc.png", bbox_inches="tight")

    qc.measure(index_qbits, icr)
    sampler = Sampler()
    job = sampler.run(qc, shots=64)
    result = job.result()
    index_dist = result.quasi_dists[0]
    index_pred = max(index_dist.items(), key=lambda x: x[1])
    print(f"Search result, index: {index_pred[0]}, " \
          f"empirical prob to collapse in this state: {index_pred[1]}")
    print(f"Prob to collapse into a correct state: {index_dist[array.index(item)]}")


if __name__ == "__main__":
    Fire(main)
