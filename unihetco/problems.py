from docplex.mp.model import Model
from collections import defaultdict

def build_maxcut_model(edges, weights=None):
    """
    Build MaxCut model (minimization form).

    Args:
        edges (list of tuples): edges (i, j)
        weights (dict): {(i, j): weight}

    Returns:
        model, variables
    """
    mdl = Model('MaxCut')
    nodes = sorted(set(u for e in edges for u in e))

    # Default weight = 1 if not provided
    weights = weights or {e: 1 for e in edges}

    x = {i: mdl.binary_var(name=f"x_{i}") for i in nodes}

    obj = mdl.sum(weights[(i, j)] * (x[i] + x[j] - 2 * x[i] * x[j]) for (i, j) in edges)
    mdl.minimize(-obj)

    # obj = mdl.sum(weights[(i, j)] * (2 * x[i] * x[j] - x[i] - x[j]) for (i, j) in edges)
    # mdl.minimize(obj)

    return mdl, x

def build_mis_model(edges):
    """
    Build Maximum Independent Set model (minimization form).

    Args:
        edges (list of tuples): edges (i, j)

    Returns:
        model, variables
    """
    mdl = Model('MIS')
    nodes = sorted(set(u for e in edges for u in e))
    x = {i: mdl.binary_var(name=f"x_{i}") for i in nodes}

    # Objective: minimize negative count
    mdl.minimize(-mdl.sum(x[i] for i in nodes))

    # Constraints: no two adjacent nodes selected
    for (i, j) in edges:
        mdl.add_constraint(x[i] + x[j] <= 1)

    return mdl, x

def build_mvc_model(edges):
    """
    Build a minimum vertex cover model using DOcplex.

    Args:
        edges: list of (i, j) tuples

    Returns:
        model, x_vars: DOcplex model and binary vars for each node
    """
    mdl = Model("MVC")

    # Get unique nodes
    nodes = sorted(set(u for e in edges for u in e))

    # Binary variable: x[i] = 1 if node i is in the cover
    x = {i: mdl.binary_var(name=f"x_{i}") for i in nodes}

    # Objective: minimize number of nodes in the cover
    mdl.minimize(mdl.sum(x[i] for i in nodes))

    # Constraints: each edge must be covered by at least one endpoint
    for u, v in edges:
        mdl.add_constraint(-x[u] - x[v] <= -1)

    return mdl, x

def build_bp_model(edges, weights=None):
    """
    Build a balanced graph partitioning model in DOcplex (QP form, no auxiliary vars).

    Args:
        edges: list of (i, j) tuples
        weights: dict {(i, j): weight} or None (default = 1)

    Returns:
        model, x_vars (binary partition indicators)
    """
    mdl = Model("BalancedPartition")
    
    # Identify all nodes
    nodes = sorted(set(u for e in edges for u in e))
    n = len(nodes)
    
    # Default weight = 1 if not provided
    weights = weights or {e: 1 for e in edges}
    
    # Binary variables: x[i] = 1 if node i in group A, else 0
    x = {i: mdl.binary_var(name=f"x_{i}") for i in nodes}

    # Quadratic objective: w_ij * (x_i + x_j - 2 * x_i * x_j)
    obj = mdl.sum(
        weights[(i, j)] * (x[i] + x[j] - 2 * x[i] * x[j])
        for (i, j) in edges
    )
    mdl.minimize(obj)

    # Balance constraint: |A| = n/2
    mdl.add_constraint(mdl.sum(x[i] for i in nodes) == n // 2)

    return mdl, x

def build_mds_model(edges):
    """
    Solves the Minimum Dominating Set problem using Docplex.

    Args:
        G (networkx.Graph): Input undirected graph.

    Returns:
        solution (dict): A dict {node: 0 or 1} indicating which nodes are in the dominating set.
        objective_value (float): Total number of nodes in the dominating set.
    """
    mdl = Model("MinimumDominatingSet")

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    all_nodes = sorted(set(u for edge in edges for u in edge))

    # Binary variable x_i = 1 if node i is in the dominating set
    x = {i: mdl.binary_var(name=f"x_{i}") for i in all_nodes}

    # For every node i, ensure it is dominated (by itself or one of its neighbors)
    for i in all_nodes:
        neighborhood = [x[i]] + [x[j] for j in adj[i]]
        mdl.add_constraint(-mdl.sum(neighborhood) <= -1, ctname=f"dom_{i}")

    # Objective: minimize number of selected nodes
    mdl.minimize(mdl.sum(x[i] for i in all_nodes))

    return mdl, x

def build_maxclique_model(edges):
    """
    Build Maximum Clique model (minimization form).
    """
    mdl = Model('MaxClique')
    nodes = sorted(set(u for e in edges for u in e))
    x = {i: mdl.binary_var(name=f"x_{i}") for i in nodes}

    # Objective: minimize negative clique size  <=>  maximize clique size
    mdl.minimize(-mdl.sum(x[i] for i in nodes))

    # Constraints: for every NON-edge (u,v), cannot select both
    # Build undirected edge set
    E = set()
    for u, v in edges:
        if u == v: 
            continue
        a, b = (u, v) if u < v else (v, u)
        E.add((a, b))

    # Add non-edge constraints
    for i_idx in range(len(nodes)):
        for j_idx in range(i_idx + 1, len(nodes)):
            u, v = nodes[i_idx], nodes[j_idx]
            if (u, v) not in E:          # non-edge in original graph
                mdl.add_constraint(x[u] + x[v] <= 1)

    return mdl, x