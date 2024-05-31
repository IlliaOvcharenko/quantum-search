import sys, os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
# from qiskit.primitives.sampler import Sampler

from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit import transpile
from fire import Fire


def main():
    sim = AerSimulator()

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    # qc.draw("mpl")
    # plt.savefig("figs/test-qc.png", bbox_inches="tight")
    qc_measured = qc.measure_all(inplace=False)
    # qc_measured.draw("mpl")
    # plt.savefig("figs/test-qc-measured.png", bbox_inches="tight")

    qc_measured_opt = transpile(qc_measured, sim, optimization_level=0)

    sampler = Sampler()
    job = sampler.run([qc_measured_opt], shots=10)
    print(f"Job ID is {job.job_id()}")
    pub_result = job.result()
    print(pub_result[0].data.meas.get_counts())


if __name__ == "__main__":
    Fire(main)
