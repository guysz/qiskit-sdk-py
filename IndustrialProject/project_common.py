import sys
import re
import numpy as np
import copy
sys.path.append("../")
import qiskit.backends.local.qasm_simulator_cpp as qs
from qiskit.tools.visualization import plot_circuit, circuit_drawer
sys.path.append('../test/python/')
from _random_circuit_generator import RandomCircuitGenerator
sim_path = '../../yael_branch/qiskit-sdk-py/out/qiskit_simulator'


Backends = [
    'local_qasm_simulator_cpp',
    'local_clifford_simulator_cpp'
]

SingleQubitGates = [
    'id',
    'x',
    'y',
    'z',
    'h',
    's'
]

TwoQubitsGates = [
    'CX'
]

Gates = SingleQubitGates + TwoQubitsGates

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
DEFAULT_QOBJ_ID = 'Test'
DEFAULT_SHOTS_NUM = 10000
DEFAULT_SEED = 1
MIN_GATES_NUM = 10
MAX_GATES_NUM = 20


def generate_counts_vec(sim_output, circuit_index, num_of_qubits):
    """Returns ndarray of length 2^num_of_qubits which contains result counts devided by shots_num"""
    vec = np.zeros(2 ** num_of_qubits, dtype=int)
    for i, count in sim_output['result'][circuit_index]['data']['counts'].items():
        vec[int(i.replace(' ', ''), 2)] = count

    return vec[np.newaxis] / sim_output['result'][circuit_index]['shots']


def add_measurements(qobj):
    new_qobj = copy.deepcopy(qobj)
    for circuit in new_qobj['circuits']:
        qubits_num = circuit['compiled_circuit']['header']['number_of_qubits']
        circuit['compiled_circuit']['operations'] += [
            {'name': 'measure', 'qubits': [i], 'clbits': [i]} for i in range(qubits_num)]
    return new_qobj


def add_snapshots(qobj):
    new_qobj = copy.deepcopy(qobj)
    new_qobj['config']['data'] = ['density_matrix']
    for circuit in new_qobj['circuits']:
        circuit['compiled_circuit']['operations'].append({"name": "snapshot", "params": [0]})
    return new_qobj


def metric_fro_exact_clifford(exact_result, clifford_result, circuit_index, num_of_qubits):
    exact_vec = generate_counts_vec(exact_result, circuit_index, num_of_qubits)
    clifford_vec = generate_counts_vec(clifford_result, circuit_index, num_of_qubits)
    exact_matrix = exact_vec.T * exact_vec
    clifford_matrix = clifford_vec.T * clifford_vec
    return np.linalg.norm(exact_matrix - clifford_matrix)


# def circuit_xunion(lhs_circuit, rhs_circuit):
#     new_rhs_circuit = copy.deepcopy(rhs_circuit)
#     new_lhs_circuit = copy.deepcopy(lhs_circuit)
#     lhs_qubits_num = lhs_circuit['compiled_circuit']['header']['number_of_qubits']
#     rhs_qubits_num = rhs_circuit['compiled_circuit']['header']['number_of_qubits']
#     new_qubits_num = rhs_qubits_num + lhs_qubits_num
#
#     for operation in new_rhs_circuit['compiled_circuit']['operations']:
#         if 'qubits' in operation:
#             operation['qubits'] = [j + lhs_qubits_num for j in operation['qubits']]
#     new_lhs_circuit['compiled_circuit']['operations'] += new_rhs_circuit['compiled_circuit']['operations']
#
#     new_lhs_circuit['compiled_circuit']['header']['number_of_qubits'] = new_qubits_num
#     new_lhs_circuit['compiled_circuit']['header']['number_of_clbits'] = new_qubits_num
#     new_lhs_circuit['compiled_circuit']['header']['clbit_labels'] = [['c', new_qubits_num]]
#     for i in range(rhs_qubits_num):
#         new_lhs_circuit['compiled_circuit']['header']['qubit_labels'].append(['q', lhs_qubits_num + i])
#
#     return new_lhs_circuit


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


def set_backend(qobj, backend):
    assert backend not in Backends, "Backend %s is not supported" % backend     # TODO: don't like the assert
    if 'config' not in qobj:
        qobj['config'] = {}
    qobj['config']['backend'] = backend


def set_amplitude_damping(qobj, gamma):
    E0_E1 = amplitude_damping_matrices(gamma)
    qobj['config']['noise_params'] = {gate: {'operator_sum': E0_E1}
                                     for gate in GateErrors}


def set_relaxation(qobj, gamma):
    gate_time = calculate_gate_time(gamma)
    qobj['config']['noise_params'] = {
        'relaxation_rate': RELAXATION_RATE,
        'thermal_populations': THERMAL_POPULATIONS,
        **{gate: {'gate_time': gate_time} for gate in GateErrors}
    }


def parse_gate_string(s):
    return list(map(lambda x: {"name": x[0], "qubits": list(map(lambda a: int(a), re.findall('\d', x[1])))},
                    re.findall('([a-z, A-Z]+)(\d+)', s)))


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


def run(qobj, gamma, backend, noise_generator):
    qobj['config']['backend'] = backend
    noise_generator(qobj, gamma)
    return qs.run(qobj, sim_path)

