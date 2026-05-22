import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
import torch
SIMULATION_DEVICE = "lightning.qubit"

def pauli_coeff_l1_norm(H):
    """
    H: qml.Hamiltonian with decomposition H = sum_j coeffs[j] * ops[j]
    Returns ||H||_* = sum_j |coeff_j|
    """
    return float(np.sum(np.abs(np.array(H.coeffs, dtype=np.float64))))

# Class-based (pickleable) versions for parallelization

class QAOAMaxcutCost:
    """
    Pickleable QAOA cost evaluator for MaxCut problem.
    Can be pickled/passed to thread pools unlike closure-based functions.
    """
    def __init__(self, graph, n_layers=1, normalized=False):
        self.graph = graph
        self.n_layers = n_layers
        self.normalized = normalized
        self.n_qubits = len(graph.nodes)
        self.cost_h, self.mixer_h = qaoa.maxcut(graph)
        
        if normalized:
            self.coeff_l1 = float(sum(abs(c) for c in self.cost_h.coeffs))
        else:
            self.coeff_l1 = None
    
    def __call__(self, params, **kwargs):
        """Evaluate cost for given parameters"""
        def qaoa_layer(gamma, alpha):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(alpha, self.mixer_h)
        
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.n_layers, params[0], params[1])
            return qml.expval(self.cost_h)
        
        # Use explicit wire list for GPU compatibility
        dev = qml.device(SIMULATION_DEVICE, wires=list(range(self.n_qubits)))
        cost_qnode = qml.QNode(circuit, dev, diff_method="adjoint", interface="torch")
        expval = cost_qnode(params)
        
        if self.normalized and self.coeff_l1 is not None:
            expval = expval / self.coeff_l1
        return expval

class QAOAMaxCutShot:
    """
    Pickleable QAOA cost evaluator for MIS (Maximum Independent Set) problem.
    Can be pickled/passed to thread pools unlike closure-based functions.
    """
    def __init__(self, graph, n_layers=1, n_shots=5000):
        self.graph = graph
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.n_qubits = len(graph.nodes)
        self.cost_h, self.mixer_h = qaoa.maxcut(graph)
    
    def __call__(self, params, **kwargs):
        """Evaluate cost for given parameters"""
        def qaoa_layer(gamma, beta):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(beta, self.mixer_h)

        dev = qml.device(SIMULATION_DEVICE, wires=list(range(self.n_qubits)))
        
        @qml.qnode(dev)
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.n_layers, params[0], params[1])
            return qml.sample(wires=range(self.n_qubits))
        
        # Use explicit wire list for GPU compatibility
        
        circuit_sample = qml.set_shots(circuit, shots=self.n_shots)
        bitstrings = circuit_sample(params)
        
        return bitstrings

class QAOAMISCost:
    """
    Pickleable QAOA cost evaluator for MIS (Maximum Independent Set) problem.
    Can be pickled/passed to thread pools unlike closure-based functions.
    """
    def __init__(self, graph, n_layers=1, normalized=False):
        self.graph = graph
        self.n_layers = n_layers
        self.normalized = normalized
        self.n_qubits = len(graph.nodes)
        self.cost_h, self.mixer_h = qaoa.max_independent_set(graph, constrained=False)
        
        if normalized:
            self.coeff_l1 = float(sum(abs(c) for c in self.cost_h.coeffs))
        else:
            self.coeff_l1 = None
    
    def __call__(self, params, **kwargs):
        """Evaluate cost for given parameters"""
        def qaoa_layer(gamma, beta):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(beta, self.mixer_h)
        
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.n_layers, params[0], params[1])
            return qml.expval(self.cost_h)
        
        # Use explicit wire list for GPU compatibility
        dev = qml.device(SIMULATION_DEVICE, wires=list(range(self.n_qubits)))
        cost_qnode = qml.QNode(circuit, dev, diff_method="adjoint", interface="torch")
        expval = cost_qnode(params)
        
        if self.normalized and self.coeff_l1 is not None:
            expval = expval / self.coeff_l1
        return expval

class QAOAMISShot:
    """
    Pickleable QAOA cost evaluator for MIS (Maximum Independent Set) problem.
    Can be pickled/passed to thread pools unlike closure-based functions.
    """
    def __init__(self, graph, n_layers=1, n_shots=5000):
        self.graph = graph
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.n_qubits = len(graph.nodes)
        self.cost_h, self.mixer_h = qaoa.max_independent_set(graph, constrained=False)
    
    def __call__(self, params, **kwargs):
        """Evaluate cost for given parameters"""
        def qaoa_layer(gamma, beta):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(beta, self.mixer_h)

        dev = qml.device(SIMULATION_DEVICE, wires=list(range(self.n_qubits)))
        
        @qml.qnode(dev)
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.n_layers, params[0], params[1])
            return qml.sample(wires=range(self.n_qubits))
        
        # Use explicit wire list for GPU compatibility
        
        circuit_sample = qml.set_shots(circuit, shots=self.n_shots)
        bitstrings = circuit_sample(params)
        
        return bitstrings

class QAOAMaxCliqueCost:
    """
    Pickleable QAOA cost evaluator for MaxClique problem.
    Can be pickled/passed to thread pools unlike closure-based functions.
    """
    def __init__(self, graph, n_layers=1, normalized=False):
        self.graph = graph
        self.n_layers = n_layers
        self.normalized = normalized
        self.n_qubits = len(graph.nodes)
        self.cost_h, self.mixer_h = qaoa.max_clique(graph, constrained=False)
        
        if normalized:
            self.coeff_l1 = float(sum(abs(c) for c in self.cost_h.coeffs))
        else:
            self.coeff_l1 = None
    
    def __call__(self, params, **kwargs):
        """Evaluate cost for given parameters"""
        def qaoa_layer(gamma, beta):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(beta, self.mixer_h)
        
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.n_layers, params[0], params[1])
            return qml.expval(self.cost_h)
        
        # Use explicit wire list for GPU compatibility
        dev = qml.device(SIMULATION_DEVICE, wires=list(range(self.n_qubits)))
        cost_qnode = qml.QNode(circuit, dev, diff_method="adjoint", interface="torch")
        expval = cost_qnode(params)
        
        if self.normalized and self.coeff_l1 is not None:
            expval = expval / self.coeff_l1
        return expval
    
class QAOAMaxCliqueShot:
    """
    Pickleable QAOA shot sampler for MaxClique problem (unconstrained).
    Can be pickled/passed to thread pools unlike closure-based functions.
    Returns bitstring samples from the QAOA circuit.
    """
    def __init__(self, graph, n_layers=1, n_shots=5000):
        self.graph = graph
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.n_qubits = len(graph.nodes)
        self.cost_h, self.mixer_h = qaoa.max_clique(graph, constrained=False)
    
    def __call__(self, params, **kwargs):
        """Sample bitstrings from the QAOA circuit"""
        def qaoa_layer(gamma, beta):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(beta, self.mixer_h)

        dev = qml.device(SIMULATION_DEVICE, wires=list(range(self.n_qubits)))
        
        @qml.qnode(dev)
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.n_layers, params[0], params[1])
            return qml.sample(wires=range(self.n_qubits))
        
        circuit_sample = qml.set_shots(circuit, shots=self.n_shots)
        bitstrings = circuit_sample(params)
        
        return bitstrings

class QAOAMVCCost:
    """
    Pickleable QAOA cost evaluator for MVC (Minimum Vertex Cover) problem (unconstrained).
    Can be pickled/passed to thread pools unlike closure-based functions.
    """
    def __init__(self, graph, n_layers=1, normalized=False):
        self.graph = graph
        self.n_layers = n_layers
        self.normalized = normalized
        self.n_qubits = len(graph.nodes)
        self.cost_h, self.mixer_h = qaoa.min_vertex_cover(graph, constrained=False)
        
        if normalized:
            self.coeff_l1 = float(sum(abs(c) for c in self.cost_h.coeffs))
        else:
            self.coeff_l1 = None
    
    def __call__(self, params, **kwargs):
        """Evaluate cost for given parameters"""
        def qaoa_layer(gamma, beta):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(beta, self.mixer_h)
        
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.n_layers, params[0], params[1])
            return qml.expval(self.cost_h)
        
        # Use explicit wire list for GPU compatibility
        dev = qml.device(SIMULATION_DEVICE, wires=list(range(self.n_qubits)))
        cost_qnode = qml.QNode(circuit, dev, diff_method="adjoint", interface="torch")
        expval = cost_qnode(params)
        
        if self.normalized and self.coeff_l1 is not None:
            expval = expval / self.coeff_l1
        return expval

class QAOAMVCShot:
    """
    Pickleable QAOA shot sampler for MVC (Minimum Vertex Cover) problem (unconstrained).
    Can be pickled/passed to thread pools unlike closure-based functions.
    Returns bitstring samples from the QAOA circuit.
    """
    def __init__(self, graph, n_layers=1, n_shots=5000):
        self.graph = graph
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.n_qubits = len(graph.nodes)
        self.cost_h, self.mixer_h = qaoa.min_vertex_cover(graph, constrained=False)
    
    def __call__(self, params, **kwargs):
        """Sample bitstrings from the QAOA circuit"""
        def qaoa_layer(gamma, beta):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(beta, self.mixer_h)

        dev = qml.device(SIMULATION_DEVICE, wires=list(range(self.n_qubits)))
        
        @qml.qnode(dev)
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.n_layers, params[0], params[1])
            return qml.sample(wires=range(self.n_qubits))
        
        circuit_sample = qml.set_shots(circuit, shots=self.n_shots)
        bitstrings = circuit_sample(params)
        
        return bitstrings