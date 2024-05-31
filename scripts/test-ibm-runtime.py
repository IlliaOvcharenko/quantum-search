import sys, os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
# from qiskit.primitives.sampler import Sampler

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (QiskitRuntimeService,
                                SamplerV2 as Sampler)
from fire import Fire


def main():
    ibm_token = open("credentials/ibm-token.txt").read().replace("\n", "")
    ibm_quantum_service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
    # print(ibm_quantum_service.backends(simulator=True))

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    # qc.draw("mpl")
    # plt.savefig("figs/test-qc.png", bbox_inches="tight")
    qc_measured = qc.measure_all(inplace=False)
    # qc_measured.draw("mpl")
    # plt.savefig("figs/test-qc-measured.png", bbox_inches="tight")

    # sampler = Sampler()
    # job = sampler.run(qc_measured, shots=1)
    # result = job.result()
    # print(result)


    backend = ibm_quantum_service.least_busy(operational=True, simulator=False)
    print(backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_measured_opt = pm.run(qc_measured)

    sampler = Sampler(backend=backend)
    job = sampler.run([qc_measured_opt], shots=10)
    print(f"Job ID is {job.job_id()}")
    pub_result = job.result()[0]
    print(pub_result)


if __name__ == "__main__":
    Fire(main)
