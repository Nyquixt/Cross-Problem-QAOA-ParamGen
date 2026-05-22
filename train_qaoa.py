import os
import pickle
import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
from joblib import Parallel, delayed
from dataset import generate_graphs
from gurobi_solvers import _get_gurobi_env, maxcut, mis, maxclique, mvc, mds
from utils import mis_empirical_stats_from_samples, maxcut_empirical_stats_from_samples, maxclique_empirical_stats_from_samples, mvc_empirical_stats_from_samples

def qaoa_maxcut(graph, n_layers=1, n_shots=5000, params=None, opt_val=None, verbose=False): # transfer params from MC as initialization
    cost_h, mixer_h = qaoa.maxcut(graph)
    #############################
    ### QAOA layer definition ###
    wires = range(len(graph.nodes))
    # Defines the device and the QAOA cost function
    dev = qml.device('lightning.qubit', wires=len(wires))
    
    # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
    def qaoa_layer(gamma, beta):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(beta, mixer_h)
    
    # Creates the actual quantum circuit for the QAOA algorithm
    @qml.qnode(dev)
    def circuit(params, return_samples=False):
        for w in wires: # apply Hadamards to get the n qubit |+> state
            qml.Hadamard(wires=w)
            
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        if return_samples: return qml.sample()
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def circuit_sample(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.sample(wires=wires)
    #############################
    
    #############################
    ##### QAOA optimization #####
    if verbose: print(f"\np={n_layers}")
    
    if params is not None:
        theta = np.array(params, requires_grad=True) # use provided initialization
    else:
        theta = np.random.rand(2, n_layers, requires_grad=True) * 5e-2 # Initial guess parameters
        
    angle = [theta] # Store the values of the circuit parameter
    cost = [circuit(theta)] # Store the values of the cost function
    
    opt = qml.AdamOptimizer(stepsize=0.01) # Our optimizer!
    max_iterations = 500 # Maximum number of calls to the optimizer 
    conv_tol = 1e-8 # Convergence threshold to stop our optimization procedure
    
    for n in range(max_iterations):
        theta, prev_cost = opt.step_and_cost(circuit, theta)
        cost.append(circuit(theta))
        angle.append(theta)
    
        conv = np.abs(cost[-1] - prev_cost)
        if verbose:
            if (n) % 20 == 0:
                print(f"Step = {n},  Cost function = {cost[-1]:.8f} ")
        if conv <= conv_tol:
            if verbose:
                print(f"Terminated at step = {n}")
            break

    params_maxcut = angle[-1]
    # print optimal parameters and most frequently sampled bitstring
    # exp_val = circuit(params_maxcut)
    circuit_sample_5000 = qml.set_shots(circuit_sample, shots=n_shots)
    bitstrings = circuit_sample_5000(params_maxcut)
    # sampled_ints = [bitstring_to_int(string) for string in bitstrings]
    # counts = np.bincount(np.array(sampled_ints))
    # exp_val, feas_rate = mis_empirical_expval(counts, graph)
    exp_val, feas_rate, opt_rate = maxcut_empirical_stats_from_samples(bitstrings, graph, opt_val)
    if verbose:
        print(f"\nFinal value of the cost function = {cost[-1]:.8f}")
        print(f"\nEmpirical expectation value = {exp_val:.8f}")
        print(f"\nFeasibility rate = {feas_rate:.8f}")
        print(f"\nOptimal rate = {opt_rate:.8f}")
        print(f"Optimal value of gamma = {angle[-1][0]}")
        print(f"Optimal value of beta = {angle[-1][1]}\n")

    return {
        "exp_value": exp_val,
        "feas_rate": feas_rate,
        "opt_rate": opt_rate,
        "params": params_maxcut,
        "gamma": angle[-1][0],
        "beta": angle[-1][1],
        "n_steps": n
    }


def qaoa_mis(graph, n_layers=1, n_shots=5000, params=None, opt_val=None, verbose=False): # transfer params from MC as initialization
    cost_h, mixer_h = qaoa.max_independent_set(graph, constrained=False)
    #############################
    ### QAOA layer definition ###
    wires = range(len(graph.nodes))
    # Defines the device and the QAOA cost function
    dev = qml.device('lightning.qubit', wires=len(wires))               # analytic
    
    # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
    def qaoa_layer(gamma, beta):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(beta, mixer_h)
    
    # Creates the actual quantum circuit for the QAOA algorithm
    @qml.qnode(dev)
    def circuit_expval(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def circuit_sample(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.sample(wires=wires)
    #############################
    
    #############################
    ##### QAOA optimization #####
    if verbose: print(f"\np={n_layers}")
    
    if params is not None:
        theta = np.array(params, requires_grad=True) # use provided initialization
    else:
        theta = np.random.rand(2, n_layers, requires_grad=True) * 5e-2 # Initial guess parameters
        
    angle = [theta] # Store the values of the circuit parameter
    cost = [circuit_expval(theta)] # Store the values of the cost function
    
    opt = qml.AdamOptimizer(stepsize=0.01) # Our optimizer!
    max_iterations = 500 # Maximum number of calls to the optimizer 
    conv_tol = 1e-8 # Convergence threshold to stop our optimization procedure
    
    for n in range(max_iterations):
        theta, prev_cost = opt.step_and_cost(circuit_expval, theta)
        cost.append(circuit_expval(theta))
        angle.append(theta)
    
        conv = np.abs(cost[-1] - prev_cost)
        if verbose:
            if (n) % 20 == 0:
                print(f"Step = {n},  Cost function = {cost[-1]:.8f} ")
        if conv <= conv_tol:
            if verbose:
                print(f"Terminated at step = {n}")
            break

    params_mis = angle[-1]
    circuit_sample_5000 = qml.set_shots(circuit_sample, shots=n_shots)
    bitstrings = circuit_sample_5000(params_mis)
    # sampled_ints = [bitstring_to_int(string) for string in bitstrings]
    # counts = np.bincount(np.array(sampled_ints))
    # exp_val, feas_rate = mis_empirical_expval(counts, graph)
    exp_val, feas_rate, opt_rate = mis_empirical_stats_from_samples(bitstrings, graph, opt_val)
    if verbose:
        print(f"\nFinal value of the cost function = {cost[-1]:.8f}")
        print(f"\nEmpirical expectation value = {exp_val:.8f}")
        print(f"\nFeasibility rate = {feas_rate:.8f}")
        print(f"\nOptimal rate = {opt_rate:.8f}")
        print(f"Optimal value of gamma = {angle[-1][0]}")
        print(f"Optimal value of beta = {angle[-1][1]}\n")

    return {
        "exp_value": exp_val,
        "feas_rate": feas_rate,
        "opt_rate": opt_rate,
        "params": params_mis,
        "gamma": angle[-1][0],
        "beta": angle[-1][1],
        "n_steps": n
    }

def qaoa_maxclique(graph, n_layers=1, n_shots=5000, params=None, opt_val=None, verbose=False): # transfer params from MC as initialization
    cost_h, mixer_h = qaoa.max_clique(graph, constrained=False)
    #############################
    ### QAOA layer definition ###
    wires = range(len(graph.nodes))
    # Defines the device and the QAOA cost function
    dev = qml.device('lightning.qubit', wires=len(wires))               # analytic
    
    # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
    def qaoa_layer(gamma, beta):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(beta, mixer_h)
    
    # Creates the actual quantum circuit for the QAOA algorithm
    @qml.qnode(dev)
    def circuit_expval(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def circuit_sample(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.sample(wires=wires)
    #############################
    
    #############################
    ##### QAOA optimization #####
    if verbose: print(f"\np={n_layers}")
    
    if params is not None:
        theta = np.array(params, requires_grad=True) # use provided initialization
    else:
        theta = np.random.rand(2, n_layers, requires_grad=True) * 5e-2 # Initial guess parameters
        
    angle = [theta] # Store the values of the circuit parameter
    cost = [circuit_expval(theta)] # Store the values of the cost function
    
    opt = qml.AdamOptimizer(stepsize=0.01) # Our optimizer!
    max_iterations = 500 # Maximum number of calls to the optimizer 
    conv_tol = 1e-8 # Convergence threshold to stop our optimization procedure
    
    for n in range(max_iterations):
        theta, prev_cost = opt.step_and_cost(circuit_expval, theta)
        cost.append(circuit_expval(theta))
        angle.append(theta)
    
        conv = np.abs(cost[-1] - prev_cost)
        if verbose:
            if (n) % 20 == 0:
                print(f"Step = {n},  Cost function = {cost[-1]:.8f} ")
        if conv <= conv_tol:
            if verbose:
                print(f"Terminated at step = {n}")
            break

    params_maxclique = angle[-1]
    circuit_sample_5000 = qml.set_shots(circuit_sample, shots=n_shots)
    bitstrings = circuit_sample_5000(params_maxclique)
    # sampled_ints = [bitstring_to_int(string) for string in bitstrings]
    # counts = np.bincount(np.array(sampled_ints))
    # exp_val, feas_rate = maxclique_empirical_expval(counts, graph)
    exp_val, feas_rate, opt_rate = maxclique_empirical_stats_from_samples(bitstrings, graph, opt_val)
    if verbose:
        print(f"\nFinal value of the cost function = {cost[-1]:.8f}")
        print(f"\nEmpirical expectation value = {exp_val:.8f}")
        print(f"\nFeasibility rate = {feas_rate:.8f}")
        print(f"\nOptimal rate = {opt_rate:.8f}")
        print(f"Optimal value of gamma = {angle[-1][0]}")
        print(f"Optimal value of beta = {angle[-1][1]}\n")

    return {
        "exp_value": exp_val,
        "feas_rate": feas_rate,
        "opt_rate": opt_rate,
        "params": params_maxclique,
        "gamma": angle[-1][0],
        "beta": angle[-1][1],
        "n_steps": n
    }

def qaoa_mvc(graph, n_layers=1, n_shots=5000, params=None, opt_val=None, verbose=False): # transfer params from MC as initialization
    cost_h, mixer_h = qaoa.min_vertex_cover(graph, constrained=False)
    #############################
    ### QAOA layer definition ###
    wires = range(len(graph.nodes))
    # Defines the device and the QAOA cost function
    dev = qml.device('lightning.qubit', wires=len(wires))               # analytic
    
    # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
    def qaoa_layer(gamma, beta):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(beta, mixer_h)
    
    # Creates the actual quantum circuit for the QAOA algorithm
    @qml.qnode(dev)
    def circuit_expval(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def circuit_sample(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.sample(wires=wires)
    #############################
    
    #############################
    ##### QAOA optimization #####
    if verbose: print(f"\np={n_layers}")
    
    if params is not None:
        theta = np.array(params, requires_grad=True) # use provided initialization
    else:
        theta = np.random.rand(2, n_layers, requires_grad=True) * 5e-2 # Initial guess parameters
        
    angle = [theta] # Store the values of the circuit parameter
    cost = [circuit_expval(theta)] # Store the values of the cost function
    
    opt = qml.AdamOptimizer(stepsize=0.01) # Our optimizer!
    max_iterations = 500 # Maximum number of calls to the optimizer 
    conv_tol = 1e-8 # Convergence threshold to stop our optimization procedure
    
    for n in range(max_iterations):
        theta, prev_cost = opt.step_and_cost(circuit_expval, theta)
        cost.append(circuit_expval(theta))
        angle.append(theta)
    
        conv = np.abs(cost[-1] - prev_cost)
        if verbose:
            if (n) % 20 == 0:
                print(f"Step = {n},  Cost function = {cost[-1]:.8f} ")
        if conv <= conv_tol:
            if verbose:
                print(f"Terminated at step = {n}")
            break

    params_mvc = angle[-1]
    circuit_sample_5000 = qml.set_shots(circuit_sample, shots=n_shots)
    bitstrings = circuit_sample_5000(params_mvc)
    # sampled_ints = [bitstring_to_int(string) for string in bitstrings]
    # counts = np.bincount(np.array(sampled_ints))
    # exp_val, feas_rate = mvc_empirical_expval(counts, graph)
    exp_val, feas_rate, opt_rate = mvc_empirical_stats_from_samples(bitstrings, graph, opt_val)
    if verbose:
        print(f"\nFinal value of the cost function = {cost[-1]:.8f}")
        print(f"\nEmpirical expectation value = {exp_val:.8f}")
        print(f"\nFeasibility rate = {feas_rate:.8f}")
        print(f"\nOptimal rate = {opt_rate:.8f}")
        print(f"Optimal value of gamma = {angle[-1][0]}")
        print(f"Optimal value of beta = {angle[-1][1]}\n")

    return {
        "exp_value": exp_val,
        "feas_rate": feas_rate,
        "opt_rate": opt_rate,
        "params": params_mvc,
        "gamma": angle[-1][0],
        "beta": angle[-1][1],
        "n_steps": n
    }

def qaoa_mds(graph, n_layers=1, n_shots=5000, params=None, verbose=False): # transfer params from MC as initialization
    cost_h, mixer_h = mds_unconstrained(graph, penalty=3.0)
    #############################
    ### QAOA layer definition ###
    wires = range(len(graph.nodes))
    # Defines the device and the QAOA cost function
    dev = qml.device('lightning.qubit', wires=len(wires))               # analytic
    
    # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
    def qaoa_layer(gamma, beta):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(beta, mixer_h)
    
    # Creates the actual quantum circuit for the QAOA algorithm
    @qml.qnode(dev)
    def circuit_expval(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def circuit_sample(params):
        for w in wires:
            qml.Hadamard(w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.sample(wires=wires)
    #############################
    
    #############################
    ##### QAOA optimization #####
    if verbose: print(f"\np={n_layers}")
    
    if params is not None:
        theta = np.array(params, requires_grad=True) # use provided initialization
    else:
        theta = np.random.rand(2, n_layers, requires_grad=True) * 5e-2 # Initial guess parameters
        
    angle = [theta] # Store the values of the circuit parameter
    cost = [circuit_expval(theta)] # Store the values of the cost function
    
    opt = qml.AdamOptimizer(stepsize=0.01) # Our optimizer!
    max_iterations = 500 # Maximum number of calls to the optimizer 
    conv_tol = 1e-8 # Convergence threshold to stop our optimization procedure
    
    for n in range(max_iterations):
        theta, prev_cost = opt.step_and_cost(circuit_expval, theta)
        cost.append(circuit_expval(theta))
        angle.append(theta)
    
        conv = np.abs(cost[-1] - prev_cost)
        if verbose:
            if (n) % 20 == 0:
                print(f"Step = {n},  Cost function = {cost[-1]:.8f} ")
        if conv <= conv_tol:
            if verbose:
                print(f"Terminated at step = {n}")
            break

    params_mds = angle[-1]
    circuit_sample_5000 = qml.set_shots(circuit_sample, shots=n_shots)
    bitstrings = circuit_sample_5000(params_mds)
    sampled_ints = [bitstring_to_int(string) for string in bitstrings]
    counts = np.bincount(np.array(sampled_ints))
    exp_val, feas_rate = mds_empirical_expval(counts, graph)
    if verbose:
        print(f"\nFinal value of the cost function = {cost[-1]:.8f}")
        print(f"\nEmpirical expectation value = {exp_val:.8f}")
        print(f"\nFeasibility rate = {feas_rate:.8f}")
        print(f"Optimal value of gamma = {angle[-1][0]}")
        print(f"Optimal value of beta = {angle[-1][1]}\n")

    return {
        "exp_value": exp_val,
        "feas_rate": feas_rate,
        "params": params_mds,
        "gamma": angle[-1][0],
        "beta": angle[-1][1],
        "n_steps": n
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy")
    parser.add_argument("--problem", type=str, default="maxcut")
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args()
    
    PROBLEM = args.problem
    N_LAYERS = args.p
    DATASET = args.dataset

    test_path = f'./dataset_{DATASET}/testset/graphs.pkl'
    
    with open(test_path, 'rb') as infile:
        test_graphs = pickle.load(infile)
        if DATASET == 'main':
            test_graphs = [graph["graph"] for graph in test_graphs]
    
    print("Getting optimal solutions...")
    def solve_graph(graph):
        env = _get_gurobi_env()
        env.setParam('OutputFlag', 0)
        env.start()
        solver_args = {
            "env": env,
            "graph": graph,
            "max_time": 100.0
        }
        if PROBLEM == 'maxcut':
            return maxcut(**solver_args)
        elif PROBLEM == 'mis':
            return mis(**solver_args)
        elif PROBLEM == 'maxclique':
            return maxclique(**solver_args)
        elif PROBLEM == 'mvc':
            return mvc(**solver_args)
        elif PROBLEM == 'mds':
            return mds(**solver_args)

    test_solutions = Parallel(n_jobs=-1)(delayed(solve_graph)(g) for g in test_graphs)

    def run_qaoa(graph, i, solution):
        if PROBLEM == 'maxcut':
            output = qaoa_maxcut(graph, n_layers=N_LAYERS, opt_val=solution)
        if PROBLEM == 'mis':
            output = qaoa_mis(graph, n_layers=N_LAYERS, opt_val=solution)
        if PROBLEM == 'maxclique':
            output = qaoa_maxclique(graph, n_layers=N_LAYERS, opt_val=solution)
        if PROBLEM == 'mvc':
            output = qaoa_mvc(graph, n_layers=N_LAYERS, opt_val=solution)
        if PROBLEM == 'mds':
            output = qaoa_mds(graph, n_layers=N_LAYERS, opt_val=solution)
        exp_val = output["exp_value"]
        n_steps = output["n_steps"]
        feas_rate = output["feas_rate"]
        opt_rate = output["opt_rate"]
        print(f"Graph {i}: QAOA exp value = {exp_val:.4f}, Optimal value = {solution:.4f}")
        approximation_ratio = (exp_val) / solution
        return approximation_ratio, n_steps, feas_rate, opt_rate

    results = Parallel(n_jobs=-1)(
        delayed(run_qaoa)(graph, i, test_solutions[i]) for i, graph in enumerate(test_graphs)
    )

    approximation_ratios, n_steps_list, feas_rate_list, opt_rate_list = zip(*results)
    
    avg_ar = sum(approximation_ratios) / len(approximation_ratios)
    avg_steps = sum(n_steps_list) / len(n_steps_list)
    avg_feas_rate = sum(feas_rate_list) / len(feas_rate_list)
    avg_opt_rate = sum(opt_rate_list) / len(opt_rate_list)
    print(f" > Approximation Ratio: {avg_ar:.4f}")
    print(f" > Average n_steps: {avg_steps:.2f}")
    print(f" > Average Feasibility Rate: {avg_feas_rate:.4f}")
    print(f" > Average Optimal Rate: {avg_opt_rate:.4f}")

    with open(f'{PROBLEM}_{N_LAYERS}_{DATASET}.txt', 'w') as outfile:
        outfile.write(f" > Approximation Ratio: {(sum(approximation_ratios)/ len(approximation_ratios)):.4f}\n")
        outfile.write(f" > Average n_steps: {(sum(n_steps_list)/ len(n_steps_list)):.4f}")
        outfile.write(f"\n > Average Feasibility Rate: {avg_feas_rate:.4f}")
        outfile.write(f"\n > Average Optimal Rate: {avg_opt_rate:.4f}")

        