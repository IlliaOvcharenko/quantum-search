import sys, os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import math

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (QiskitRuntimeService,
                                SamplerV2 as Sampler)
from fire import Fire


def main()
    ibm_token = open("credentials/ibm-token.txt").read().replace("\n", "")
    ibm_quantum_service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)

    array = [4, 7, 2, 1, 3]
    # item = 5 # answer should be 6
    # item = 4 # answer should be 0
    item = 2 # answer should be 5

    # array = [4, 7, 2, 1, 3, 0, 5, 6]
    # # item = 5 # answer should be 6
    # # item = 4 # answer should be 0
    # item = 0 # answer should be 5

    # array = np.arange(50)
    # np.random.seed(42)
    # np.random.shuffle(array)
    # array = array.tolist()
    # item = 10
    print(f"Input array: {array}")
    print(f"Item to search: {item}")
    print(f"Answer: {array.index(item)}")
    # evalue)

    n_index_qbits = math.ceil(math.log(len(array), 2))
    n_value_qbits = math.ceil(math.log(max(array), 2))
    to_bits = lambda i: bin(i)[2:][::-1]
    to_int = lambda b: int(b[::-1], 2)

    # print(array)
    # print([to_bits(v) for v in array])
    # print([to_int(to_bits(v)) for v in array])
    # exit(0)

    index_qbits = [i for i in range(n_index_qbits)]
    value_qbits = [i + n_index_qbits for i in range(n_value_qbits)]
    phase_qbit = n_index_qbits + n_value_qbits

    print(f"N qubits for index: {n_index_qbits}")
    print(f"N qubits for value: {n_value_qbits}")

    # n_grover_iterations = math.floor(math.sqrt(len(array)))
    n_grover_iterations = 1
    print(f"Num of grover iterations: {n_grover_iterations}")
    # print(index_qbits)
    # print(value_qbits)
    # print(phase_qbit)
    # exit(0)

    qc = QuantumCircuit(
        QuantumRegister(n_index_qbits, "index"),
        QuantumRegister(n_value_qbits, "value"),
        QuantumRegister(1, "phase"),
        icr := ClassicalRegister(n_index_qbits, "meas-index"),
        # ClassicalRegister(3, "meas-value"),
        # pcr := ClassicalRegister(1, "meas-phase"),
    )
    # TODO hadamar for index
    # TODO ket minus for phase

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
            # vcr := ClassicalRegister(3, "meas-index"),
            # ClassicalRegister(3, "meas-value"),
            # pcr := ClassicalRegister(3, "meas-phase"),
        )
        for array_idx in range(len(array)):
            # qc.barrier()
            array_idx_bit = to_bits(array_idx)[:n_index_qbits]
            array_idx_bit += "0" * (n_index_qbits - len(array_idx_bit))
            array_item = array[array_idx]
            array_item_bit = to_bits(array_item)[:n_value_qbits]
            array_item_bit += "0" * (n_value_qbits - len(array_item_bit))

            zeros = [index_qbits[i] for i, b in enumerate(array_idx_bit) if b == "0"]
            ones = [value_qbits[i] for i, b in enumerate(array_item_bit) if b == "1"]
            # print(array_idx)
            # print(array_idx_bit)
            # print(array_item)
            # print(array_item_bit)
            # print(zeros)
            # print(ones)
            # print()

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

    # qc.draw("mpl")
    # plt.savefig("figs/oracle-qc.png", bbox_inches="tight")

    # qc.h(phase_qbit)
    qc.measure(index_qbits, icr)
    qc.draw("mpl")
    plt.savefig("figs/oracle-qc-array-8.png", bbox_inches="tight")
    # qc = qc.measure_all(inplace=False)
    # sampler = Sampler()
    # job = sampler.run(qc, shots=64)
    # result = job.result()
    # print(max(result.quasi_dists[0].items(), key=lambda x: x[1]))
    # print(result.quasi_dists[0][array.index(item)])

    backend = ibm_quantum_service.get_backend("ibm_sherbrooke")
    # backend = ibm_quantum_service.least_busy(operational=True, simulator=False)
    print(f"Selected quantum computer: {backend}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_opt = pm.run(qc)
    # qc_opt.draw("mpl")
    # plt.savefig("figs/oracle-qc-opt-array-8.png", bbox_inches="tight")

    sampler = Sampler(backend=backend)
    job = sampler.run([qc_opt], shots=5)
    print(f"Job ID is {job.job_id()}")
    # pub_result = job.result()[0]
    # print(pub_result)

if __name__ == "__main__":
    Fire(main)
