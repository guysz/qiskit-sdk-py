import numpy as np
import sys
from functools import reduce
sys.path.append("../")
import qiskit.backends.local.qasm_simulator_cpp as qs

SIM_EXECUTABLE = '../out/src/qasm-simulator-cpp/qasm_simulator_cpp'
qobj1 = {
  "id": "first_try",
  "config": {
      "data": ["density_matrix"],
      "backend_name": "local_qasm_simulator",

  },
  "circuits": [
    {

      "name": "my_first_try",
      "compiled_circuit": {
        "header": {
                    "clbit_labels": [],
                    "number_of_clbits": 0,
                    "number_of_qubits": 3,
                    "qubit_labels": [["q", 0], ["q", 1], ["q", 2]]
                },
        "operations": [
                    {"name": "id", "qubits": [0]},
                    {"name": "id", "qubits": [1]},
                    {"name": "id", "qubits": [2]},
                    {"name": "snapshot", "params": [0]}
                ]
      }
    }
  ]
}

def extract_density_matrix(sim_output, circ_num):
    return sim_output['result'][circ_num]['data']['snapshots']['0']['density_matrix']

def frobenius(density_matrix):
    return np.linalg.norm(density_matrix, 'fro')

def generate_intervals(beta, num_intervals):
    start = (sum(beta)-1)/3
    end = sum(beta)/3
    return  np.linspace(start, end, num=num_intervals, dtype=float)

def inner_runner(qobj, shots, noise_params):

    if "config" not in qobj:
        qobj["config"] = {}

    # Add "shots" to qobj
    qobj["config"]["shots"] = shots

    # Add noise params to qobj
    qobj["config"]["noise_params"] = noise_params

    # Run simulator
    sim_output = qs.run(qobj, SIM_EXECUTABLE)

    # Return the frobenius norm of the resulted density_matrix
    return frobenius(extract_density_matrix(sim_output, 0))

def outer_runner(qobj, beta, shots=1024, num_intervals=100):
    x_axis = generate_intervals(beta, num_intervals)
    return x_axis, [inner_runner(qobj, shots, {"id": {"p_pauli": [beta[0]-x, beta[1]-x, beta[2]-x]}}) for x in x_axis]
