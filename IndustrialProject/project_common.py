import sys
import numpy as np
import copy
from enum import Enum
sys.path.append("../")
import qiskit.backends.local.qasm_simulator_cpp as qs
from qiskit.tools.visualization import plot_circuit, circuit_drawer
sys.path.append('../test/python/')
from _random_circuit_generator import RandomCircuitGenerator
sim_path = '../../yael_branch/qiskit-sdk-py/out/qiskit_simulator'

GateErrors = [
    # 'measure',
    # 'reset',
    # 'X90',
    'id',
    'CX',
    'U'
]

EXACT_SIM_STR = 'local_qasm_simulator_cpp'
CLIFFORD_SIM_STR = 'local_clifford_simulator_cpp'
RELAXATION_RATE = 1.0
THERMAL_POPULATIONS = [1.0, 0.0]


class ExperimentType(Enum):
    EXACT_ONLY = 0
    WITH_CLIFFORD = 1


def generate_counts_vec(sim_output, circuit_index, num_of_qubits):
    """Returns ndarray of length 2^num_of_qubits which contains result counts devided by shots_num"""
    vec = np.zeros(2 ** num_of_qubits, dtype=int)
    for i, count in sim_output['result'][circuit_index]['data']['counts'].items():
        vec[int(i.replace(' ', ''), 2)] = count

    return vec[np.newaxis] / sim_output['result'][circuit_index]['shots']


def extract_density_matrix(sim_output, circuit_index):
    return sim_output['result'][circuit_index]['data']['density_matrix']


def add_measurements(qobj):
    new_qobj = copy.deepcopy(qobj)
    for circuit in new_qobj['circuits']:
        qubits_num = circuit['compiled_circuit']['header']['number_of_qubits']
        circuit['compiled_circuit']['operations'] += [
            {'name': 'measure', 'qubits': [i], 'clbits': [i]} for i in range(qubits_num)]
    return new_qobj


def add_density_matrix(qobj):
    new_qobj = copy.deepcopy(qobj)
    new_qobj['config']['data'] = ['density_matrix']
    return new_qobj


def metric_fro_exact_clifford(exact_result, clifford_result, circuit_index, num_of_qubits):
    exact_vec = generate_counts_vec(exact_result, circuit_index, num_of_qubits)
    clifford_vec = generate_counts_vec(clifford_result, circuit_index, num_of_qubits)
    exact_matrix = exact_vec.T * exact_vec
    clifford_matrix = clifford_vec.T * clifford_vec
    return np.linalg.norm(exact_matrix - clifford_matrix)


def metric_fro_exact_exact(result1, result2, circuit_index, num_of_qubits):
    return np.linalg.norm(
        extract_density_matrix(result2, circuit_index) - extract_density_matrix(result1, circuit_index))


def unify_circuits(circuit1, circuit2):
    new_circuit1 = copy.deepcopy(circuit1)
    new_circuit2 = copy.deepcopy(circuit2)
    qubits_num1 = circuit1['compiled_circuit']['header']['number_of_qubits']
    qubits_num2 = circuit2['compiled_circuit']['header']['number_of_qubits']
    qubits_num = qubits_num2 + qubits_num1

    for operation in new_circuit2['compiled_circuit']['operations']:
        if 'qubits' in operation:
            operation['qubits'] = [j + qubits_num1 for j in operation['qubits']]
    new_circuit1['compiled_circuit']['operations'] += new_circuit2['compiled_circuit']['operations']

    new_circuit1['compiled_circuit']['header']['number_of_qubits'] = qubits_num
    new_circuit1['compiled_circuit']['header']['number_of_clbits'] = qubits_num
    new_circuit1['compiled_circuit']['header']['clbit_labels'] = [['c', qubits_num]]
    for i in range(qubits_num2):
        new_circuit1['compiled_circuit']['header']['qubit_labels'].append(['q', qubits_num1 + i])

    return new_circuit1


def calculate_gate_time(gamma):     # TODO: write it more generic (with r and [a,b])
    p = gamma / 2 - 0.5 * np.sqrt(1 - gamma) + 0.5
    return -(np.log(1 - p))


def amplitude_damping_matrices(g):
    """
    Returns two matrices of size 2x2:
    E0 = [
            [1,0],
            [0, sqrt(1-g)]
         ]

    E1 = [
            [0, sqrt(g)],
            [0,0]
         ]
    """
    assert 0 <= g < 1, "Gamma (g) must be between zero to one"
    return [[[1, 0], [0, np.sqrt(1 - g)]], [[0, np.sqrt(g)], [0, 0]]]


def set_amplitude_damping(qobj, gamma):
    E0_E1 = amplitude_damping_matrices(gamma)
    qobj['config']['noise_params'] = {gate: {'operator_sum': E0_E1} for gate in GateErrors}


def set_relaxation(qobj, gamma):
    gate_time = calculate_gate_time(gamma)
    qobj['config']['noise_params'] = {
        'relaxation_rate': RELAXATION_RATE,
        'thermal_populations': THERMAL_POPULATIONS,
        **{gate: {'gate_time': gate_time} for gate in GateErrors}
    }


def generate_circuit(num_qubits, operations, with_measure):
    return {
        'compiled_circuit':
            {
                'header':
                    {
                        'number_of_qubits': num_qubits,
                        'number_of_clbits': num_qubits,
                        "qubit_labels": [["q", n] for n in range(num_qubits)],
                        "clbit_labels": [["c", num_qubits]]
                    },
                'operations': add_measurements(operations) if with_measure else operations
            }
    }


def generate_random_circuits(num_qubits, num_gates, num_circuits, gates):
    circuit_generator = RandomCircuitGenerator(max_qubits=num_qubits,
                                               min_qubits=num_qubits,
                                               max_depth=num_gates,
                                               min_depth=num_gates)
    circuit_generator.add_circuits(num_circuits, False, gates)
    circuits = [{'compiled_circuit': circ} for circ in circuit_generator.get_circuits('qobj')]
    circuit_drawers = [circuit_drawer(circ) for circ in circuit_generator.get_circuits('QuantumCircuit')]
    return circuits, circuit_drawers


def create_basic_qobj(name, shots, seed):
    return {
        'id': name,
        'config': {
            'shots': shots,
            'seed': seed
        },
        'circuits': []
    }


def run(qobj, gamma, backend, noise_callback):
    qobj['config']['backend'] = backend
    noise_callback(qobj, gamma)
    return qs.run(qobj, sim_path)


def _run_experiment(qobj, gamma_range, sim1_str, sim2_str, sim1_noise_callback, sim2_noise_callback):
    res1 = []
    res2 = []
    for gamma in gamma_range:
        res1.append(run(qobj, gamma, sim1_str, sim1_noise_callback))
        res2.append(run(qobj, gamma, sim2_str, sim2_noise_callback))
    return res1, res2


def _run_experiment_exact_only(qobj, gamma_range):
    return _run_experiment(add_density_matrix(qobj), gamma_range,
                           EXACT_SIM_STR, EXACT_SIM_STR, set_amplitude_damping, set_relaxation)


def _run_experiment_with_clifford(qobj, gamma_range):
    return _run_experiment(add_measurements(qobj), gamma_range,
                           EXACT_SIM_STR, CLIFFORD_SIM_STR, set_amplitude_damping, set_relaxation)


def run_experiment(qobj, gamma_range, experiment_type):
    return {
        ExperimentType.EXACT_ONLY: _run_experiment_exact_only,
        ExperimentType.WITH_CLIFFORD: _run_experiment_with_clifford
    }[experiment_type](qobj, gamma_range)
