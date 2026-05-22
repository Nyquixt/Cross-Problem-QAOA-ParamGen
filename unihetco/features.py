# build features for nodes in graphs

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected
from torch import Tensor

def _zscore_per_feature(x: Tensor, eps: float = 1e-8) -> Tensor:
    if x.numel() == 0:
        return x
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + eps
    return (x - mean) / std

def get_graph_features(
    G: nx.Graph,
    clustering=True,
    betweenness=True,
    closeness=True,
    core=True,
    eigenvec=True,
    pagerank=True,
    undirected=True,
    use_edge_weight=False,
    normalize=True
):
    
    # 1) Make a working copy and ensure undirected if requested
    if undirected and G.is_directed():
        G = G.to_undirected()
    else:
        G = G.copy()

    G.remove_edges_from(nx.selfloop_edges(G))

    # 2) Relabel nodes to 0..N-1 so we can index them directly
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    N = G.number_of_nodes()
    nodes = list(G.nodes())  # [0, 1, ..., N-1] after relabel

    # 3) Decide which edge weight attribute (if any) to use
    weight_key = "weight" if use_edge_weight else None

    # ---------- Node-level structural features ---------- #
    feats = []

    # Degree (unnormalized)
    deg = np.array([G.degree(n) for n in nodes], dtype=float)
    feats.append(deg.reshape(-1, 1))

    # Clustering coefficient
    if clustering:
        cc = nx.clustering(G, weight=weight_key)
        cc_vec = np.array([cc[n] for n in nodes], dtype=float)
        feats.append(cc_vec.reshape(-1, 1))

    # Betweenness centrality (can be expensive for large graphs)
    if betweenness:
        bc = nx.betweenness_centrality(G, normalized=True, weight=weight_key)
        bc_vec = np.array([bc[n] for n in nodes], dtype=float)
        feats.append(bc_vec.reshape(-1, 1))

    # Closeness centrality
    if closeness:
        cl= nx.closeness_centrality(G)
        cl_vec = np.array([cl[n] for n in nodes], dtype=float)
        feats.append(cl_vec.reshape(-1, 1))

    # Core number (k-core index)
    if core:
        cn = nx.core_number(G)
        cn_vec = np.array([cn[n] for n in nodes], dtype=float)
        feats.append(cn_vec.reshape(-1, 1))

    # eigenvector centrality
    if eigenvec:
        # use weight=None for unweighted, or "weight" if edges have 'weight'
        ev = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6, weight=None)
        ev_vec = np.array([ev[n] for n in nodes], dtype=float)
        feats.append(ev_vec.reshape(-1, 1))

    # PageRank
    if pagerank:
        try:
            pr = nx.pagerank(G, alpha=0.85, weight=weight_key)
        except nx.PowerIterationFailedConvergence:
            # fallback: unweighted PageRank if weighted failed
            pr = nx.pagerank(G, alpha=0.85, weight=None)
        pr_vec = np.array([pr[n] for n in nodes], dtype=float)
        feats.append(pr_vec.reshape(-1, 1))

    # Stack all features: [N, F]
    X = np.concatenate(feats, axis=1)  # shape: [num_nodes, num_features]

    # Optional z-score normalization per feature dimension
    if normalize:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

    x = torch.from_numpy(X).to(torch.float)

    # ---------- Edges & edge weights ---------- #
    # Build edge list with optional weights
    edges = []
    weights = []
    for u, v, attr in G.edges(data=True):
        edges.append((u, v))
        w = attr.get("weight", 1.0) if use_edge_weight else 1.0
        weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    edge_weight = torch.tensor(weights, dtype=torch.float)               # [E]

    # Make edges undirected (both directions), carrying edge_weight along
    edge_index, edge_weight = to_undirected(
        edge_index, edge_attr=edge_weight, num_nodes=N
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    return data

def build_qc_node_features(
    Q: Tensor,
    c: Tensor,
    *,
    zero_tol: float = 0.0,
    normalize_degree: bool = True,
    make_undirected: bool = True,
) -> Data:
    """
    Build a QC graph (PyG Data) from Q only, with c already absorbed into Q.

    - Nodes: variables x_j, j = 0..N-1
    - Node feature: degree_j = # of nonzero Q_ij in row j (|Q_ij| > zero_tol),
                    including diagonal entries (self-loops).
    - Edges: one edge (i -> j) for each nonzero Q_ij, with edge_weight = Q_ij.
             If make_undirected=True, edges are symmetrized via to_undirected.

    Args:
        Q: [N, N] tensor (float), quadratic coefficients with c absorbed on diag.
        zero_tol: treat |Q_ij| <= zero_tol as zero (ignored).
        normalize_degree: if True, z-score normalize the degree vector.
        make_undirected: if True, symmetrize edge_index / edge_weight.

    Returns:
        data: torch_geometric.data.Data with:
            - data.x          : [N, 1] degree feature
            - data.edge_index : [2, E]
            - data.edge_weight: [E]
    """
    assert Q.dim() == 2 and Q.size(0) == Q.size(1)
    N = Q.size(0)
    c = c.view(-1)
    assert c.numel() == N

    device, dtype = Q.device, Q.dtype

    # Absorb c into diagonal
    Q_eff = Q.clone()
    diag_idx = torch.arange(N, device=device)
    Q_eff[diag_idx, diag_idx] += c.to(device=device, dtype=dtype)

    # Nonzero pattern
    mask = Q_eff.abs() > zero_tol
    row_idx, col_idx = mask.nonzero(as_tuple=True)  # [E], [E]

    edge_index = torch.stack([row_idx, col_idx], dim=0)       # [2, E]
    edge_weight = Q_eff[row_idx, col_idx].to(dtype=dtype)     # [E]

    if make_undirected:
        edge_index, edge_weight = to_undirected(
            edge_index, edge_attr=edge_weight, num_nodes=N
        )

    # Degree feature
    deg = mask.sum(dim=1).to(dtype=dtype)  # [N]
    if normalize_degree and deg.numel() > 0:
        mean, std = deg.mean(), deg.std() + 1e-8
        deg = (deg - mean) / std
    x = deg.view(-1, 1)  # [N, 1]

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

def build_ab_hetero_features(
    A: Tensor,
    b: Tensor,
    *,
    zero_tol: float = 0.0,
    add_reverse_edges: bool = True,
    normalize: bool = True,
    compute_closeness: bool = False,
    compute_core: bool = False,
    compute_betweenness: bool = False,
) -> HeteroData:
    """
    Build a bipartite HeteroData for Ax <= b.

    Node types:
      'constr': i = 0..M-1  (constraints)
      'var':    j = 0..N-1  (variables)

    Edge type:
      ('constr', 'A', 'var') with edge_weight = A_ij for each nonzero.
    """
    assert A.dim() == 2, "A must be 2D [M, N]"
    M, N = A.shape

    A = A.clone()
    b = b.view(-1)
    assert b.numel() == M, "b must have length M"

    device = A.device
    dtype = A.dtype

    # Nonzero pattern
    mask = A.abs() > zero_tol
    row_idx, col_idx = mask.nonzero(as_tuple=True)  # [E], [E]
    E = row_idx.numel()

    data = HeteroData()

    # ---------- SPECIAL CASE: no constraints / no nonzero A_ij ----------
    # (complete MaxClique graph, etc.)
    if M == 0 or E == 0:
        # constraint features: just b (could be empty if M=0)
        x_constr = b.to(device=device, dtype=dtype).view(-1, 1)  # [M, 1]

        # variable features: all degrees are zero (no constraints touch them)
        x_var = torch.zeros(N, 1, dtype=dtype, device=device)   # [N, 1]

        if normalize:
            # x_constr is typically small; you can normalize if you want
            if x_var.numel() > 0:
                x_var = _zscore_per_feature(x_var)

        data["constr"].x = x_constr
        data["constr"].num_nodes = M

        data["var"].x = x_var
        data["var"].num_nodes = N

        # Empty edge sets so HeteroConv still has the edge types defined
        empty_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        empty_edge_weight = torch.empty(0, dtype=dtype, device=device)

        data["constr", "A", "var"].edge_index = empty_edge_index
        data["constr", "A", "var"].edge_weight = empty_edge_weight

        if add_reverse_edges:
            data["var", "rev_A", "constr"].edge_index = empty_edge_index.clone()
            data["var", "rev_A", "constr"].edge_weight = empty_edge_weight.clone()

        return data

    # ---------- GENERAL CASE: there ARE constraints ----------

    # Build a bipartite NetworkX graph for structural metrics
    G = nx.Graph()
    # 0 .. M-1 -> constr, M .. M+N-1 -> var
    for i in range(M):
        G.add_node(i, kind="constr")
    for j in range(N):
        G.add_node(M + j, kind="var")

    for i, j in zip(row_idx.tolist(), col_idx.tolist()):
        w = float(A[i, j].item())
        G.add_edge(i, M + j, weight=w)

    num_nodes_total = M + N
    nodes_order = list(range(num_nodes_total))

    # degree
    deg = np.array([G.degree(n) for n in nodes_order], dtype=float).reshape(-1, 1)

    # closeness
    if compute_closeness:
        closeness = nx.closeness_centrality(G)
        closeness_vec = np.array([closeness[n] for n in nodes_order], dtype=float).reshape(-1, 1)
    else:
        closeness_vec = np.zeros((num_nodes_total, 1), dtype=float)

    # core number
    if compute_core:
        core = nx.core_number(G)
        core_vec = np.array([core[n] for n in nodes_order], dtype=float).reshape(-1, 1)
    else:
        core_vec = np.zeros((num_nodes_total, 1), dtype=float)

    # betweenness (optional, can be expensive)
    if compute_betweenness:
        bc = nx.betweenness_centrality(G, normalized=True, weight="weight")
        bc_vec = np.array([bc[n] for n in nodes_order], dtype=float).reshape(-1, 1)
    else:
        bc_vec = np.zeros((num_nodes_total, 1), dtype=float)

    # For now, you said you're only really using degree, but we keep room to extend:
    struct_feats = deg  # or np.concatenate([deg, bc_vec, closeness_vec, core_vec], axis=1)

    struct_feats_all = torch.from_numpy(struct_feats).to(device=device, dtype=dtype)
    struct_constr = struct_feats_all[:M]   # [M, F_struct]
    struct_var = struct_feats_all[M:]      # [N, F_struct]

    # A,b-based local variable stats (deg_var)
    edge_vals = A[row_idx.to(device), col_idx.to(device)]
    abs_edge_vals = edge_vals.abs()

    deg_var = torch.zeros(N, dtype=dtype, device=device)
    deg_var = deg_var.index_add(0, col_idx.to(device), torch.ones_like(abs_edge_vals))

    # Constraint features: [b]  (you simplified to only b)
    x_constr = b.to(device=device, dtype=dtype).view(-1, 1)

    # Variable features: [deg_var] (you simplified to only deg)
    x_var = deg_var.view(-1, 1)

    if normalize:
        x_var = _zscore_per_feature(x_var)

    data["constr"].x = x_constr
    data["constr"].num_nodes = M

    data["var"].x = x_var
    data["var"].num_nodes = N

    # Forward edges
    row_idx_t, col_idx_t = row_idx.to(device), col_idx.to(device)
    edge_index_cv = torch.stack([row_idx_t, col_idx_t], dim=0)  # [2, E]
    data["constr", "A", "var"].edge_index = edge_index_cv
    data["constr", "A", "var"].edge_weight = edge_vals

    if add_reverse_edges:
        data["var", "rev_A", "constr"].edge_index = torch.stack(
            [col_idx_t, row_idx_t], dim=0
        )
        data["var", "rev_A", "constr"].edge_weight = edge_vals.clone()

    return data