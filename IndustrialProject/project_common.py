import sys
import re
import numpy as np
import copy
import random
sys.path.append("../")
import qiskit.backends.local.qasm_simulator_cpp as qs
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
        vec[int(i, 2)] = count

    return vec[np.newaxis] / sim_output['result'][circuit_index]['shots']


def metric_fro_exact_clifford(exact_result, clifford_result, circuit_index, num_of_qubits):
    exact_vec = generate_counts_vec(exact_result, circuit_index, num_of_qubits)
    clifford_vec = generate_counts_vec(clifford_result, circuit_index, num_of_qubits)
    exact_matrix = exact_vec.T * exact_vec
    clifford_matrix = clifford_vec.T * clifford_vec
    return np.linalg.norm(exact_matrix - clifford_matrix)


def circuit_xunion(lhs_circuit, rhs_circuit):
    new_rhs_circuit = copy.deepcopy(rhs_circuit)
    new_lhs_circuit = copy.deepcopy(lhs_circuit)
    lhs_qubits_num = lhs_circuit['compiled_circuit']['header']['number_of_qubits']
    rhs_qubits_num = rhs_circuit['compiled_circuit']['header']['number_of_qubits']
    new_qubits_num = rhs_qubits_num + lhs_qubits_num

    for operation in new_rhs_circuit['compiled_circuit']['operations']:
        if 'qubits' in operation:
            operation['qubits'] = [j + lhs_qubits_num for j in operation['qubits']]
    new_lhs_circuit['compiled_circuit']['operations'] += new_rhs_circuit['compiled_circuit']['operations']

    new_lhs_circuit['compiled_circuit']['header']['number_of_qubits'] = new_qubits_num
    new_lhs_circuit['compiled_circuit']['header']['number_of_clbits'] = new_qubits_num
    new_lhs_circuit['compiled_circuit']['header']['clbit_labels'] = [['c', new_qubits_num]]
    for i in range(rhs_qubits_num):
        new_lhs_circuit['compiled_circuit']['header']['qubit_labels'].append(['q', lhs_qubits_num + i])

    return new_lhs_circuit


def calculate_gate_time(gamma):
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


def rand_gate_string(qubits_num, gates_num):
    res = ''
    for _ in range(gates_num):
        gate = random.choice(Gates)
        if gate in SingleQubitGates:
            qubits = random.sample(range(qubits_num), 1)
        else:
            qubits = random.sample(range(qubits_num), 2)
        res += gate + ''.join([str(i) for i in qubits])
    return res


def generate_circuit(num_qubits, gate_string, with_measurements):
    gates = parse_gate_string(gate_string)
    if with_measurements:
        gates += [{"name": "measure", "qubits": [n], "clbits": [n]} for n in range(num_qubits)]
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
                'operations': gates
            }
    }


def create_basic_qobj(name, shots, seed):
    return {
        'id': name,
        'config': {
            'shots': shots,
            'seed': seed
        },
        'circuits': []
    }


# def create_qobj(name, shots, seed):
#     qobj = create_basic_qobj(name, shots, seed)
#     circuit = {
#         'compiled_circuit': {
#             "header": {
#                 "number_of_clbits": 0,
#                 "number_of_qubits": 0,
#                 "clbit_labels": [],
#                 "qubit_labels": []
#             },
#             "operations": []
#         }
#     }
#     for _ in range(CIRCUITS_NUM):
#         circuit = enlarge_circuit(circuit)
#         qobj['circuits'].append(circuit)
#
#     return qobj
