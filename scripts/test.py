import sys, os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives.sampler import Sampler
from fire import Fire

def main():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.draw("mpl")
    plt.savefig("figs/test-qc.png", bbox_inches="tight")
    qc_measured = qc.measure_all(inplace=False)
    qc_measured.draw("mpl")
    plt.savefig("figs/test-qc-measured.png", bbox_inches="tight")
    sampler = Sampler()
    job = sampler.run(qc_measured, shots=1)
    result = job.result()
    print(result)


if __name__ == "__main__":
    Fire(main)
