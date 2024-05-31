from fire import Fire
from qiskit_ibm_runtime import QiskitRuntimeService

def main(job_id: str):
    # cscs2wbx35wg00810gxg
    # csbx9fxca010008x7b20

    ibm_token = open("credentials/ibm-token.txt").read().replace("\n", "")
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token=ibm_token
    )
    job = service.job(job_id)
    job_result = job.result()
    print(job_result[0].data.meas.get_counts())


if __name__ == "__main__":
    Fire(main)
