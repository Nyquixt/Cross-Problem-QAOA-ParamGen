import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Union
from joblib import Parallel, delayed

import itertools
import networkx as nx
import pennylane as qml
from pennylane import qaoa

def mds_unconstrained(graph, penalty=3.0):
    """
    Unconstrained MDS as a penalty Hamiltonian + X mixer, in the same style as
    PennyLane's unconstrained MIS/MVC/MaxClique helpers (x_mixer + |+...+> init). :contentReference[oaicite:1]{index=1}

    Args:
        graph: networkx.Graph
        penalty: lambda (>=1) penalty for undominated vertices

    Returns:
        (cost_h, mixer_h): qml.Hamiltonian, qml.Hamiltonian
    """
    # Ensure wires are 0..n-1
    G = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    n = G.number_of_nodes()
    wires = list(range(n))

    coeffs = []
    ops = []

    # ---- Objective: minimize sum_i x_i, with x_i = (1 - Z_i)/2 ----
    # sum_i x_i = n/2 * I  - 1/2 * sum_i Z_i
    # Put constant shift as Identity(0) (PennyLane requires a wire for Identity)
    coeffs.append(0.5 * n)
    ops.append(qml.Identity(0))

    for i in wires:
        coeffs.append(-0.5)
        ops.append(qml.PauliZ(i))

    # ---- Constraints: for each v, penalize undominated:
    # prod_{u in N[v]} (1 - x_u) = prod_{u in N[v]} (1 + Z_u)/2
    # = (1/2^k) * prod_{u in N[v]} (I + Z_u)
    # Expands into sum_{S subset N[v]} (1/2^k) * (prod_{u in S} Z_u)
    for v in wires:
        Nv = set(G.neighbors(v))
        Nv.add(v)  # closed neighborhood
        Nv = sorted(Nv)
        k = len(Nv)
        base = penalty / (2 ** k)

        # Expand all subsets S ⊆ N[v]
        for r in range(k + 1):
            for subset in itertools.combinations(Nv, r):
                if len(subset) == 0:
                    coeffs.append(base)
                    ops.append(qml.Identity(0))
                else:
                    op = qml.PauliZ(subset[0])
                    for u in subset[1:]:
                        op = op @ qml.PauliZ(u)
                    coeffs.append(base)
                    ops.append(op)

    cost_h = qml.Hamiltonian(coeffs, ops)

    # Unconstrained-style mixer is x_mixer over all wires (same convention as MVC/MaxClique). :contentReference[oaicite:2]{index=2}
    mixer_h = qaoa.x_mixer(wires)

    return cost_h, mixer_h

def train_step(graph_cost, net, optimizer):
    optimizer.zero_grad()
    loss = net(graph_cost, intermediate_steps=False)
    loss.backward()
    optimizer.step()

    return loss

def test_step(graph_cost, net, optimizer, steps):
    if steps == 0:
        params_loss = net(graph_cost, intermediate_steps=True)
        params, loss = params_loss[:-1], params_loss[-1]
        return params, loss
    net.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = net(graph_cost, intermediate_steps=False)
        loss.backward()
        optimizer.step()
    net.eval()
    params_loss = net(graph_cost, intermediate_steps=True)
    params, loss = params_loss[:-1], params_loss[-1]
    return params, loss

def train_step_batch(graph_costs, lstm, optimizer, step_optimizer=True):
    """
    graph_costs: list of callables, length B
    step_optimizer: if True, performs optimizer.step(); if False, only computes gradients
    """
    lstm.train()
    optimizer.zero_grad(set_to_none=True)

    losses = []
    for gc in graph_costs:
        loss_i = lstm(graph_cost=gc)   # should return scalar tensor
        losses.append(loss_i.view(1))

    loss = torch.mean(torch.cat(losses, dim=0))     # average over batch
    loss.backward()
    if step_optimizer:
        optimizer.step()
    return loss.detach()

def train_step_batch_parallel(graph_costs, lstm, optimizer, step_optimizer=True, n_jobs=2):
    """
    Parallel version: computes losses and gradients in separate processes.
    Aggregates gradients before applying optimizer step.
    
    graph_costs: list of callables, length B
    lstm: RNN model
    optimizer: torch optimizer
    step_optimizer: if True, performs optimizer.step(); if False, only computes gradients
    n_jobs: number of parallel processes (default 2, increase cautiously)
    """
    def evaluate_single_with_grads(gc, lstm_copy):
        """Evaluate single graph cost and compute gradients in separate process"""
        loss_i = lstm_copy(graph_cost=gc)
        
        # Compute gradients w.r.t. parameters
        params = list(lstm_copy.parameters())
        grads = torch.autograd.grad(loss_i, params, create_graph=False)
        
        return loss_i.detach().cpu().item(), grads
    
    # Evaluate in parallel, collecting losses and gradients
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(evaluate_single_with_grads)(gc, lstm) for gc in graph_costs
    )
    
    # Unpack results
    losses, all_grads_per_process = zip(*results)
    
    # Aggregate gradients
    lstm.train()
    optimizer.zero_grad(set_to_none=True)
    
    for param_idx, param in enumerate(lstm.parameters()):
        # Sum gradients across all processes
        grad_sum = None
        for grads in all_grads_per_process:
            grad = grads[param_idx]
            if grad is not None:
                if grad_sum is None:
                    grad_sum = grad.clone()
                else:
                    grad_sum += grad
        
        if grad_sum is not None:
            # Attach averaged gradient to parameter
            param.grad = grad_sum / len(all_grads_per_process)
    
    # Backward pass and optimization
    if step_optimizer:
        optimizer.step()
    
    return torch.tensor(losses).mean().detach()

def train_step_g2v(graph_cost, g_cond, net, optimizer):
    optimizer.zero_grad()
    loss = net(graph_cost, g_cond, intermediate_steps=False)
    loss.backward()
    optimizer.step()

    return loss

def train_step_g2v_batch(graph_costs, embeddings, lstm, optimizer, step_optimizer=True):
    """
    embeddings: list/array of graph embeddings, length B
    step_optimizer: if True, performs optimizer.step(); if False, only computes gradients
    """
    lstm.train()
    optimizer.zero_grad(set_to_none=True)

    losses = []
    for gc, emb in zip(graph_costs, embeddings):
        loss_i = lstm(graph_cost=gc, g_cond=emb)    # scalar tensor
        losses.append(loss_i.view(1))

    loss = torch.mean(torch.cat(losses, dim=0))
    loss.backward()
    if step_optimizer:
        optimizer.step()
    return loss.detach()

def train_step_g2v_batch_parallel(graph_costs, embeddings, lstm, optimizer, step_optimizer=True, n_jobs=2):
    """
    Parallel version: computes losses and gradients in separate processes.
    Aggregates gradients before applying optimizer step.
    
    graph_costs: list of callables, length B
    embeddings: list/array of graph embeddings, length B
    lstm: RNN model
    optimizer: torch optimizer
    step_optimizer: if True, performs optimizer.step(); if False, only computes gradients
    n_jobs: number of parallel processes (default 2, increase cautiously)
    """
    def evaluate_single_with_grads(gc, emb, lstm_copy):
        """Evaluate single graph cost with embedding and compute gradients in separate process"""
        loss_i = lstm_copy(graph_cost=gc, g_cond=emb)
        
        # Compute gradients w.r.t. parameters
        params = list(lstm_copy.parameters())
        grads = torch.autograd.grad(loss_i, params, create_graph=False)
        
        return loss_i.detach().cpu().item(), grads
    
    # Evaluate in parallel, collecting losses and gradients
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(evaluate_single_with_grads)(gc, emb, lstm) 
        for gc, emb in zip(graph_costs, embeddings)
    )
    
    # Unpack results
    losses, all_grads_per_process = zip(*results)
    
    # Aggregate gradients
    lstm.train()
    optimizer.zero_grad(set_to_none=True)
    
    for param_idx, param in enumerate(lstm.parameters()):
        # Sum gradients across all processes
        grad_sum = None
        for grads in all_grads_per_process:
            grad = grads[param_idx]
            if grad is not None:
                if grad_sum is None:
                    grad_sum = grad.clone()
                else:
                    grad_sum += grad
        
        if grad_sum is not None:
            # Attach averaged gradient to parameter
            param.grad = grad_sum / len(all_grads_per_process)
    
    # Backward pass and optimization
    if step_optimizer:
        optimizer.step()
    
    return torch.tensor(losses).mean().detach()

def test_step_g2v(graph_cost, g_cond, net, optimizer, steps):
    if steps == 0:
        params_loss = net(graph_cost, g_cond, intermediate_steps=True)
        params, loss = params_loss[:-1], params_loss[-1]
        return params, loss
    net.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = net(graph_cost, g_cond, intermediate_steps=False)
        loss.backward()
        optimizer.step()
    net.eval()
    params_loss = net(graph_cost, g_cond, intermediate_steps=True)
    params, loss = params_loss[:-1], params_loss[-1]
    return params, loss

def shuffle_two_lists(a, b, seed=None):
    assert len(a) == len(b)
    rng = random.Random(seed)
    idx = list(range(len(a)))
    rng.shuffle(idx)
    a_shuf = [a[i] for i in idx]
    b_shuf = [b[i] for i in idx]
    return a_shuf, b_shuf

def shuffle_three_lists(a, b, c, seed=None):
    assert len(a) == len(b)
    rng = random.Random(seed)
    idx = list(range(len(a)))
    rng.shuffle(idx)
    a_shuf = [a[i] for i in idx]
    b_shuf = [b[i] for i in idx]
    c_shuf = [c[i] for i in idx]
    return a_shuf, b_shuf, c_shuf

# MAXCUT utilities

def qaoa_ar_landscape_2d(
    graph_cost,
    opt_value: float,
    *,
    n_points: int = 500,
    n_gamma: Optional[int] = None,
    n_beta: Optional[int] = None,
    gamma_range=(-2*np.pi, 2*np.pi),
    beta_range=(-np.pi, np.pi),
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Evaluate a p=1 QAOA landscape over (gamma, beta) and return a 2D grid of approximation ratios.

    Args:
        graph_cost: callable(params) -> scalar expval/objective to MAXIMIZE.
                   Expected params shape is (2, 1): [[gamma],[beta]].
        opt_value: optimal objective value (C*) for approximation ratio = expval / opt_value
        n_points: approximate total #grid points if n_gamma/n_beta not provided
        n_gamma, n_beta: grid resolution (overrides n_points if both provided)
        gamma_range: (min, max) for gamma
        beta_range: (min, max) for beta
        device, dtype: torch placement for params passed into graph_cost

    Returns:
        ar_grid: (n_gamma, n_beta) numpy array of approximation ratios
        gammas: (n_gamma,) numpy linspace
        betas: (n_beta,) numpy linspace
    """
    if opt_value == 0:
        raise ValueError("opt_value is 0; approximation ratio would be undefined.")

    if n_gamma is None or n_beta is None:
        n_gamma = max(int(np.sqrt(n_points)), 2)
        n_beta = max(int(np.ceil(n_points / n_gamma)), 2)

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma, dtype=np.float64)
    betas = np.linspace(beta_range[0], beta_range[1], n_beta, dtype=np.float64)

    dev = torch.device(device) if device is not None else None
    ar_grid = np.zeros((n_gamma, n_beta), dtype=np.float64)

    with torch.no_grad():
        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                params = torch.tensor([[g], [b]], dtype=dtype, device=dev)  # (2,1)
                expval = -graph_cost(params)

                # robust scalar extraction
                if torch.is_tensor(expval):
                    expval_val = float(expval.detach().cpu().reshape(-1)[0].item())
                else:
                    expval_val = float(np.array(expval).reshape(-1)[0])

                ar_grid[i, j] = expval_val / float(opt_value)

    return ar_grid, gammas, betas

def plot_ar_landscape_with_traj(
    ar_grid,
    gammas,  # y-axis values
    betas,   # x-axis values
    traj,    # shape (T, 2) : columns [gamma, beta]
    *,
    title="QAOA Approximation Ratio Landscape",
):
    ar_grid = np.asarray(ar_grid)
    gammas = np.asarray(gammas)
    betas = np.asarray(betas)
    traj = np.asarray(traj)

    # traj[:,0]=gamma (y), traj[:,1]=beta (x)
    traj_gamma = traj[:, 0]
    traj_beta = traj[:, 1]

    plt.figure()
    plt.imshow(
        ar_grid,
        origin="lower",
        aspect="auto",
        extent=[betas[0], betas[-1], gammas[0], gammas[-1]],  # x=beta, y=gamma
    )
    plt.colorbar(label="Approximation ratio")
    plt.xlabel("beta")
    plt.ylabel("gamma")
    plt.title(title)

    # Trajectory line
    plt.plot(traj_beta, traj_gamma, linewidth=2)

    # Start / end markers
    plt.scatter([traj_beta[0]], [traj_gamma[0]], s=80, marker="o", edgecolors="k", label="start")
    plt.scatter([traj_beta[-1]], [traj_gamma[-1]], s=80, marker="X", edgecolors="k", label="end")

    plt.legend()
    plt.tight_layout()
    plt.show()


def maxcut_empirical_expval(count_array, graph):
    """
    Suppose 'count_array' is length 2^n, indexed by integer-coded bitstring (0..2^n-1).
    Returns empirical expectation value for MaxCut objective.
    MaxCut objective: number of edges crossing the cut defined by bitstring.
    """
    total_shots = sum(count_array)
    sum_of_weighted_costs = 0.0
    n = graph.number_of_nodes()
    for index, freq in enumerate(count_array):
        bitstring = f"{index:0{n}b}"
        # MaxCut: count edges crossing the cut
        cut = 0
        for u, v in graph.edges():
            if bitstring[u] != bitstring[v]:
                cut += 1
        sum_of_weighted_costs += freq * cut
    return sum_of_weighted_costs / total_shots

def maxcut_percent_of_optimal(sampled_bitstrings, optimal_solution, graph):
    """
    Single-instance MaxCut metric.

    Args:
        sampled_bitstrings: array/tensor of shape (shots, n) for one instance.
        optimal_solution: scalar optimal MaxCut value for this instance.
        graph: networkx graph for this instance.

    Returns:
        float percentage in [0, 100] of all sampled bitstrings that achieve
        the optimal MaxCut value for this instance.
    """
    if torch.is_tensor(sampled_bitstrings):
        samples = sampled_bitstrings.detach().cpu().numpy()
    else:
        samples = np.asarray(sampled_bitstrings)

    if samples.ndim != 2:
        raise ValueError("sampled_bitstrings must have shape (shots, n).")

    if samples.dtype.kind in {"U", "S", "O"}:
        samples_str = samples.astype(str)
    else:
        samples_str = samples.astype(int).astype(str)

    shots = samples_str.shape[0]
    if shots == 0:
        return 0.0

    opt_cut = int(round(float(optimal_solution)))
    optimal_count = 0

    for sample in samples_str:
        cut = 0
        for u, v in graph.edges():
            if sample[u] != sample[v]:
                cut += 1
        if cut == opt_cut:
            optimal_count += 1

    return 100.0 * optimal_count / shots

def maxcut_value(bits, G, selected_bit="1"):
    """
    Cut value for bitstring bits (0/1).
    bits[u] indicates partition assignment. Edge is cut if endpoints differ.
    selected_bit is not really needed here; kept for API consistency.
    """
    val = 0
    for u, v in G.edges():
        if str(bits[u]) != str(bits[v]):
            val += 1
    return val

def maxcut_empirical_stats_from_samples(sampled_bitstrings, graph, optimal_solution=None):
    """
    sampled_bitstrings: array/tensor (shots, n) with {0,1}
    optimal_solution: (optional) scalar optimal MaxCut value (MAXIMUM)
    Returns:
      mean_cut_value,
      feasibility_rate (=1.0),
      percent_optimal_over_all_shots
    """
    # Ensure integer-labeled graph for indexing
    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")

    if torch.is_tensor(sampled_bitstrings):
        samples = sampled_bitstrings.detach().cpu().numpy()
    else:
        samples = np.asarray(sampled_bitstrings)

    if samples.ndim != 2:
        raise ValueError("sampled_bitstrings must have shape (shots, n).")

    shots, n = samples.shape
    if shots == 0:
        return 0.0, 1.0, 0.0

    opt_val = None if optimal_solution is None else int(round(float(optimal_solution)))

    total = 0.0
    optimal = 0

    for s in samples:
        bits = "".join(str(int(x)) for x in s.tolist())
        val = maxcut_value(bits, graph)
        total += val
        if opt_val is not None and val == opt_val:
            optimal += 1

    mean_cut = total / shots
    feas_rate = 1.0
    pct_opt = optimal / shots if opt_val is not None else 0.0

    return mean_cut, feas_rate, pct_opt
    
# MIS utilities
def bitstring_to_int(bit_string_sample):
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample[::-1])

def mis_size(bits, selected_bit="1"):
        return sum(1 for b in bits if b == selected_bit)
    
def mis_feasible(bits, G, selected_bit="1"):
    # bits: 0/1 array length n
    for u, v in G.edges():
        if bits[u] == selected_bit and bits[v] == selected_bit:
            return False
    return True

def mis_empirical_expval(count_array, graph):
    # Suppose 'count_array' is e.g. length 2^n, 
    # indexed by integer-coded bitstring (0..2^n-1).
    total_shots = sum(count_array)
    feasible_shots = 0
    sum_of_weighted_costs = 0.0

    for index, freq in enumerate(count_array):
        if freq == 0: continue
        bitstring = f"{index:0{graph.number_of_nodes()}b}"
        if mis_feasible(bitstring, graph):
            feasible_shots += int(freq)
            sum_of_weighted_costs += freq * sum(int(c) for c in bitstring)
    exp_val = (sum_of_weighted_costs / feasible_shots) if feasible_shots > 0 else 0.0
    feasibility_rate = feasible_shots / total_shots
    return exp_val, feasibility_rate

def mis_percent_of_optimal(sampled_bitstrings, optimal_solution, graph=None, selected_bit="1"):
    """
    Single-instance MIS metric.

    Args:
        sampled_bitstrings: array/tensor of shape (shots, n) for one instance.
        optimal_solution: scalar optimal MIS size for this instance.
        graph: optional networkx graph for this instance. If provided, samples
            must be MIS-feasible to be counted as optimal.
        selected_bit: bit value treated as selected vertex ("1" by default).

    Returns:
        float percentage in [0, 100] of all sampled bitstrings that achieve
        the optimal MIS value for this instance.
    """
    if torch.is_tensor(sampled_bitstrings):
        samples = sampled_bitstrings.detach().cpu().numpy()
    else:
        samples = np.asarray(sampled_bitstrings)

    if samples.ndim != 2:
        raise ValueError("sampled_bitstrings must have shape (shots, n).")

    if samples.dtype.kind in {"U", "S", "O"}:
        samples_str = samples.astype(str)
    else:
        samples_str = samples.astype(int).astype(str)

    shots = samples_str.shape[0]
    if shots == 0:
        return 0.0

    target = str(selected_bit)
    opt_size = int(round(float(optimal_solution)))
    optimal_count = 0

    for sample in samples_str:
        if graph is not None and not mis_feasible(sample, graph, selected_bit=target):
            continue
        if mis_size(sample, selected_bit=target) == opt_size:
            optimal_count += 1

    return 100.0 * optimal_count / shots

def maxclique_feasible(bits, G, selected_bit="1"):
    # bits: 0/1 array or string length n
    # Check if the selected vertices form a complete subgraph (clique)
    selected = [i for i, b in enumerate(bits) if str(b) == selected_bit]
    
    if len(selected) < 2:
        return True  # Single vertex or empty set is a clique
    
    # Check if all pairs of selected vertices are connected
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if not G.has_edge(selected[i], selected[j]):
                return False
    
    return True

def maxclique_percent_of_optimal(sampled_bitstrings, optimal_solution, graph=None, selected_bit="1"):
    """
    Single-instance MaxClique metric.

    Args:
        sampled_bitstrings: array/tensor of shape (shots, n) for one instance.
        optimal_solution: scalar optimal clique size for this instance.
        graph: optional networkx graph for this instance. If provided, samples
            must be MaxClique-feasible to be counted as optimal.
        selected_bit: bit value treated as selected vertex ("1" by default).

    Returns:
        float percentage in [0, 100] of all sampled bitstrings that achieve
        the optimal MaxClique value for this instance.
    """
    if torch.is_tensor(sampled_bitstrings):
        samples = sampled_bitstrings.detach().cpu().numpy()
    else:
        samples = np.asarray(sampled_bitstrings)

    if samples.ndim != 2:
        raise ValueError("sampled_bitstrings must have shape (shots, n).")

    if samples.dtype.kind in {"U", "S", "O"}:
        samples_str = samples.astype(str)
    else:
        samples_str = samples.astype(int).astype(str)

    shots = samples_str.shape[0]
    if shots == 0:
        return 0.0

    target = str(selected_bit)
    opt_size = int(round(float(optimal_solution)))
    optimal_count = 0

    for sample in samples_str:
        if graph is not None and not maxclique_feasible(sample, graph, selected_bit=target):
            continue
        if mis_size(sample, selected_bit=target) == opt_size:
            optimal_count += 1

    return 100.0 * optimal_count / shots

def mis_empirical_stats_from_samples(sampled_bitstrings, graph, optimal_solution=None, selected_bit="1"):
    if torch.is_tensor(sampled_bitstrings):
        samples = sampled_bitstrings.detach().cpu().numpy()
    else:
        samples = np.asarray(sampled_bitstrings)

    target = str(selected_bit)
    shots = samples.shape[0]

    feasible = 0
    sum_sizes = 0
    optimal = 0

    opt_size = None if optimal_solution is None else int(round(float(optimal_solution)))

    for s in samples:
        bits = "".join(str(int(x)) for x in s.tolist())
        if mis_feasible(bits, graph, selected_bit=target):
            feasible += 1
            size = bits.count(target)
            sum_sizes += size
            if opt_size is not None and size == opt_size:
                optimal += 1

    feas_rate = feasible / shots if shots else 0.0
    mean_over_feas = (sum_sizes / feasible) if feasible else 0.0
    pct_opt = optimal / shots if shots else 0.0

    return mean_over_feas, feas_rate, pct_opt
    
def maxclique_empirical_expval(count_array, graph):
    """
    Suppose 'count_array' is e.g. length 2^n, 
    indexed by integer-coded bitstring (0..2^n-1).
    Returns (exp_val, feasibility_rate) for MaxClique objective.
    MaxClique objective: size of the clique in each bitstring.
    """
    total_shots = sum(count_array)
    feasible_shots = 0
    sum_of_weighted_costs = 0.0

    for index, freq in enumerate(count_array):
        if freq == 0: continue
        bitstring = f"{index:0{graph.number_of_nodes()}b}"
        if maxclique_feasible(bitstring, graph):
            feasible_shots += int(freq)
            sum_of_weighted_costs += freq * sum(int(c) for c in bitstring)
    exp_val = (sum_of_weighted_costs / feasible_shots) if feasible_shots > 0 else 0.0
    feasibility_rate = feasible_shots / total_shots
    return exp_val, feasibility_rate

def mvc_feasible(bits, G, selected_bit="1"):
    # bits: 0/1 array or string length n
    # Check if the selected vertices form a valid vertex cover
    # A vertex cover must have at least one endpoint of every edge selected
    for u, v in G.edges():
        if str(bits[u]) != selected_bit and str(bits[v]) != selected_bit:
            return False
    return True

def mvc_percent_of_optimal(sampled_bitstrings, optimal_solution, graph=None, selected_bit="1"):
    """
    Single-instance MVC metric.

    Args:
        sampled_bitstrings: array/tensor of shape (shots, n) for one instance.
        optimal_solution: scalar optimal vertex-cover size for this instance.
        graph: optional networkx graph for this instance. If provided, samples
            must be MVC-feasible to be counted as optimal.
        selected_bit: bit value treated as selected vertex ("1" by default).

    Returns:
        float percentage in [0, 100] of all sampled bitstrings that achieve
        the optimal MVC value for this instance.
    """
    if torch.is_tensor(sampled_bitstrings):
        samples = sampled_bitstrings.detach().cpu().numpy()
    else:
        samples = np.asarray(sampled_bitstrings)

    if samples.ndim != 2:
        raise ValueError("sampled_bitstrings must have shape (shots, n).")

    if samples.dtype.kind in {"U", "S", "O"}:
        samples_str = samples.astype(str)
    else:
        samples_str = samples.astype(int).astype(str)

    shots = samples_str.shape[0]
    if shots == 0:
        return 0.0

    target = str(selected_bit)
    opt_size = int(round(float(optimal_solution)))
    optimal_count = 0

    for sample in samples_str:
        if graph is not None and not mvc_feasible(sample, graph, selected_bit=target):
            continue
        if mis_size(sample, selected_bit=target) == opt_size:
            optimal_count += 1

    return 100.0 * optimal_count / shots

def maxclique_empirical_stats_from_samples(sampled_bitstrings, graph, optimal_solution=None, selected_bit="1"):
    """
    sampled_bitstrings: array/tensor (shots, n) with {0,1}
    optimal_solution: (optional) scalar optimal clique size
    Returns:
      mean_size_over_feasible,
      feasibility_rate,
      percent_optimal_over_all_shots
    """
    # Ensure integer-labeled graph for indexing
    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")

    if torch.is_tensor(sampled_bitstrings):
        samples = sampled_bitstrings.detach().cpu().numpy()
    else:
        samples = np.asarray(sampled_bitstrings)

    if samples.ndim != 2:
        raise ValueError("sampled_bitstrings must have shape (shots, n).")

    shots, n = samples.shape
    if shots == 0:
        return 0.0, 0.0, 0.0

    target = str(selected_bit)
    opt_size = None if optimal_solution is None else int(round(float(optimal_solution)))

    feasible = 0
    sum_sizes = 0
    optimal = 0

    for s in samples:
        bits = "".join(str(int(x)) for x in s.tolist())
        if maxclique_feasible(bits, graph, selected_bit=target):
            feasible += 1
            size = bits.count(target)
            sum_sizes += size
            if opt_size is not None and size == opt_size:
                optimal += 1

    feas_rate = feasible / shots
    mean_over_feas = (sum_sizes / feasible) if feasible else 0.0
    pct_opt = optimal / shots

    return mean_over_feas, feas_rate, pct_opt
    
def mvc_empirical_expval(count_array, graph):
    """
    Suppose 'count_array' is e.g. length 2^n, 
    indexed by integer-coded bitstring (0..2^n-1).
    Returns (exp_val, feasibility_rate) for MVC objective.
    MVC objective: size of the vertex cover in each bitstring.
    """
    total_shots = sum(count_array)
    feasible_shots = 0
    sum_of_weighted_costs = 0.0

    for index, freq in enumerate(count_array):
        if freq == 0: continue
        bitstring = f"{index:0{graph.number_of_nodes()}b}"
        if mvc_feasible(bitstring, graph):
            feasible_shots += int(freq)
            sum_of_weighted_costs += freq * sum(int(c) for c in bitstring)
    exp_val = (sum_of_weighted_costs / feasible_shots) if feasible_shots > 0 else 0.0
    feasibility_rate = feasible_shots / total_shots
    return exp_val, feasibility_rate

def mvc_empirical_stats_from_samples(sampled_bitstrings, graph, optimal_solution=None, selected_bit="1"):
    """
    sampled_bitstrings: array/tensor (shots, n) with {0,1}
    optimal_solution: (optional) scalar optimal MVC size (MINIMUM)
    Returns:
      mean_size_over_feasible,
      feasibility_rate,
      percent_optimal_over_all_shots
    """
    # Ensure integer-labeled graph for indexing
    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")

    if torch.is_tensor(sampled_bitstrings):
        samples = sampled_bitstrings.detach().cpu().numpy()
    else:
        samples = np.asarray(sampled_bitstrings)

    if samples.ndim != 2:
        raise ValueError("sampled_bitstrings must have shape (shots, n).")

    shots, n = samples.shape
    if shots == 0:
        return 0.0, 0.0, 0.0

    target = str(selected_bit)
    opt_size = None if optimal_solution is None else int(round(float(optimal_solution)))

    feasible = 0
    sum_sizes = 0
    optimal = 0

    for s in samples:
        bits = "".join(str(int(x)) for x in s.tolist())
        if mvc_feasible(bits, graph, selected_bit=target):
            feasible += 1
            size = bits.count(target)
            sum_sizes += size
            if opt_size is not None and size == opt_size:
                optimal += 1

    feas_rate = feasible / shots
    mean_over_feas = (sum_sizes / feasible) if feasible else 0.0
    pct_opt = optimal / shots

    return mean_over_feas, feas_rate, pct_opt

def mds_feasible(bits, G, selected_bit="1"):
    """
    Dominating set feasibility:
    Every vertex v must be dominated by the selected set S, i.e.,
    v is selected OR at least one neighbor of v is selected.
    """
    n = G.number_of_nodes()
    for v in range(n):
        if str(bits[v]) == selected_bit:
            continue
        # v not selected, must have a selected neighbor
        dominated = False
        for u in G.neighbors(v):
            if str(bits[u]) == selected_bit:
                dominated = True
                break
        if not dominated:
            return False
    return True


def mds_empirical_expval(count_array, graph, selected_bit="1"):
    """
    Suppose 'count_array' is e.g. length 2^n, indexed by integer-coded bitstring (0..2^n-1).
    Returns (exp_val, feasibility_rate) for MDS objective.

    MDS objective: size of dominating set in each bitstring (we want MINIMUM).
    Here we report exp_val as the mean size over feasible samples.
    """
    total_shots = sum(count_array)
    feasible_shots = 0
    sum_of_weighted_sizes = 0.0

    n = graph.number_of_nodes()
    for index, freq in enumerate(count_array):
        if freq == 0:
            continue
        bitstring = f"{index:0{n}b}"
        if mds_feasible(bitstring, graph, selected_bit=selected_bit):
            feasible_shots += int(freq)
            sum_of_weighted_sizes += freq * bitstring.count(selected_bit)

    exp_val = (sum_of_weighted_sizes / feasible_shots) if feasible_shots > 0 else 0.0
    feasibility_rate = feasible_shots / total_shots if total_shots > 0 else 0.0
    return exp_val, feasibility_rate

def postselected_mis_from_probs(probs, graph):
    """
    probs: length 2^n array with p(x)
    Returns (E[C | feasible], feasibility_rate)
    where C(x)=|S| (cardinality of selected vertices).
    """

    def mis_is_feasible_bits(bits, graph):
        # bits: array-like of 0/1 length n
        for u, v in graph.edges():
            if bits[u] == 1 and bits[v] == 1:
                return False
        return True

    n = graph.number_of_nodes()
    probs = np.asarray(probs, dtype=np.float64)

    feasible_mass = 0.0
    feasible_weighted_cost = 0.0

    for idx, p in enumerate(probs):
        if p == 0.0:
            continue
        bits = np.array([(idx >> k) & 1 for k in range(n)], dtype=np.int8)
        if mis_is_feasible_bits(bits, graph):
            c = bits.sum()  # MIS objective = size of set
            feasible_mass += p
            feasible_weighted_cost += p * c

    if feasible_mass == 0.0:
        return 0.0, 0.0
    return feasible_weighted_cost / feasible_mass, feasible_mass

def postselected_mis_from_shots(samples, G, opt_size, selected_bit="1"):
    """
    samples: np.ndarray shape (shots, n) with entries {0,1} OR {"0","1"}
    Returns: (mean_AR_over_feasible, best_AR_over_feasible, feas_rate, num_feasible)
    """

    shots = samples.shape[0]
    feas_ars = []

    # make sure we compare as strings ("0"/"1") for simplicity
    if samples.dtype != object and samples.dtype != str:
        samples_str = samples.astype(int).astype(str)
    else:
        samples_str = samples

    for s in samples_str:
        if mis_feasible(s, G, selected_bit=selected_bit):
            val = mis_size(s, selected_bit=selected_bit)
            feas_ars.append(val / opt_size)

    num_feasible = len(feas_ars)
    feas_rate = num_feasible / shots

    if num_feasible == 0:
        return 0.0, 0

    return float(np.mean(feas_ars)), feas_rate

def mis_mean_ar_landscape_2d(
    sampler_fn, # callable(params, n_shots=...) -> (shots, n) tensor/array
    graph,
    opt_size,
    *,
    n_points=500,
    n_gamma=None,
    n_beta=None,
    gamma_range=(-2*np.pi, 2*np.pi),
    beta_range=(-np.pi, np.pi),
    n_shots=5000,
    selected_bit="1",
    device=None,
    dtype=torch.float32,
):
    """
    Returns:
      ar_grid:   (n_gamma, n_beta) mean AR over feasible samples
      feas_grid: (n_gamma, n_beta) feasibility rate
      gammas:    (n_gamma,)
      betas:     (n_beta,)
    """
    if opt_size == 0:
        raise ValueError("opt_size is 0; AR undefined.")

    if n_gamma is None or n_beta is None:
        n_gamma = max(int(np.sqrt(n_points)), 2)
        n_beta  = max(int(np.ceil(n_points / n_gamma)), 2)

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma, dtype=np.float64)
    betas  = np.linspace(beta_range[0],  beta_range[1],  n_beta,  dtype=np.float64)

    dev = torch.device(device) if device is not None else None
    ar_grid = np.zeros((n_gamma, n_beta), dtype=np.float64)
    feas_grid = np.zeros((n_gamma, n_beta), dtype=np.float64)

    with torch.no_grad():
        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                params = torch.tensor([[g], [b]], dtype=dtype, device=dev)  # (2,1) for p=1
                samples = sampler_fn(params, n_shots=n_shots)

                if torch.is_tensor(samples):
                    samples = samples.detach().cpu().numpy()
                else:
                    samples = np.asarray(samples)

                ar, feas = postselected_mis_from_shots(
                    samples, graph, opt_size, selected_bit=selected_bit
                )
                ar_grid[i, j] = ar
                feas_grid[i, j] = feas

    return ar_grid, feas_grid, gammas, betas

def plot_mis_ar_and_feas(
    ar_grid,
    feas_grid,
    gammas,
    betas,
    *,
    traj=None,            # optional (T,2): [gamma, beta]
    feas_floor=0.02,      # mask AR where feas rate is too low
    title="MIS landscapes (mean AR over feasible)",
):
    ar = np.asarray(ar_grid, dtype=float)
    feas = np.asarray(feas_grid, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)

    ar_masked = np.ma.array(ar, mask=(feas < feas_floor)) if feas_floor is not None else ar
    extent = [betas[0], betas[-1], gammas[0], gammas[-1]]  # x=beta, y=gamma

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    im0 = axes[0].imshow(ar_masked, origin="lower", aspect="auto", extent=extent)
    axes[0].set_title(f"Mean AR over feasible (masked < {feas_floor:.2f})" if feas_floor else "Mean AR over feasible")
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("gamma")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(feas, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0)
    axes[1].set_title("Feasibility rate")
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("gamma")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if traj is not None:
        traj = np.asarray(traj, dtype=float)
        tg, tb = traj[:, 0], traj[:, 1]  # gamma (y), beta (x)

        for ax in axes:
            ax.plot(tb, tg, linewidth=2)
            ax.scatter([tb[0]], [tg[0]], s=80, marker="o", edgecolors="k", label="start")
            ax.scatter([tb[-1]], [tg[-1]], s=90, marker="X", edgecolors="k", label="end")
            ax.legend(loc="best")

    fig.suptitle(title)
    plt.show()


if __name__ == '__main__':
    a = [1,2,3,4,5]
    b = [1,2,3,4,5]
    c = [1,2,3,4,5]

    # a, b = shuffle_two_lists(a, b)
    # print(a, b)

    a, b, c = shuffle_three_lists(a, b, c)
    print(a, b, c)