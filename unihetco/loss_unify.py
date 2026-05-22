import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from typing import Optional
from torch_scatter import scatter_add

# c as self-loop
def objective_from_graph(
    x: torch.Tensor,
    data: Data, # PyG Batch or single Data (decision nodes only)
    *,
    undirected: bool = True,
    return_per_graph: bool = False,
    normalize: bool = True,
    node_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Objective for self-loop variant:
        f(x) = sum_{i!=j} Q_ij x_i x_j + sum_i (diag_i) x_i^2
    where diag_i includes the absorbed linear term c_i as a self-loop weight.

    Notes:
      - For binary x, x_i^2 == x_i, so self-loops implement c^T x exactly.
      - For relaxed x in [0,1], self-loops behave like c_i * x_i^2 (not c_i * x_i).

    Inputs:
      - x: concatenated over ALL nodes in `data` (decision nodes), shape [N] or [N,1]
      - data.edge_index: [2, E]
      - data.edge_attr:  [E] (defaults to 1.0 if missing)
      - data.batch:      [N] (optional; if present, computes per-graph totals)

    Returns:
      - scalar if return_per_graph=False
      - tensor [B] per-graph if return_per_graph=True
    """
    x = x.view(-1)
    device = x.device

    row, col = data.edge_index.to(device)
    w = getattr(data, "edge_weight", None)
    if w is None:
        w = torch.ones(row.size(0), dtype=x.dtype, device=device)
    else:
        w = w.to(device=device, dtype=x.dtype).view(-1)

    # Masks
    is_self = (row == col)
    is_off = ~is_self

    # Terms
    off_terms  = w[is_off] * x[row[is_off]] * x[col[is_off]]
    self_terms = w[is_self] * x[row[is_self]]

    if hasattr(data, "batch") and data.batch is not None:
        B = int(data.batch.max().item() + 1) if data.batch.numel() > 0 else 1
        nb = data.batch.to(device)

        out_off  = torch.zeros(B, dtype=x.dtype, device=device)
        out_self = torch.zeros(B, dtype=x.dtype, device=device)

        # attribute to graph of the source endpoint (same for undirected)
        out_off.index_add_(0, nb[row[is_off]],  off_terms)
        out_self.index_add_(0, nb[row[is_self]], self_terms)

        if undirected:
            out_off *= 0.5  # only halve off-diagonals (likely stored both directions)
        
        f_raw = out_off + out_self
        if not normalize:
            return f_raw if return_per_graph else f_raw.sum()

        # |V| per graph
        nV = torch.zeros(B, dtype=x.dtype, device=device)
        nV.index_add_(0, nb, torch.ones_like(nb, dtype=x.dtype))

        # |E| per graph
        nE = torch.zeros(B, dtype=x.dtype, device=device)
        if is_off.any():
            ones_off = torch.ones(is_off.sum(), dtype=x.dtype, device=device)
            nE.index_add_(0, nb[row[is_off]], ones_off)
        if undirected:
            nE *= 0.5

        use_edge_norm = (f_raw.abs() > (node_threshold * nV)) & (nE > 0)
        denom_edges = nE.clamp_min(1.0)
        denom_nodes = nV.clamp_min(1.0)
        denom = torch.where(use_edge_norm, denom_edges, denom_nodes)

        f_norm = f_raw / denom
        return f_norm if return_per_graph else f_norm.mean()

    # single-graph
    off_sum  = off_terms.sum()
    self_sum = self_terms.sum()
    if undirected:
        off_sum *= 0.5
    f_raw = off_sum + self_sum

    if not normalize:
        return f_raw

    N = torch.tensor(float(getattr(data, "num_nodes", x.numel())), device=device, dtype=x.dtype)
    nE = torch.tensor(float(is_off.sum().item()), device=device, dtype=x.dtype)
    if undirected:
        nE *= 0.5

    use_edge_norm = (f_raw.abs() > (node_threshold * N)) and (nE > 0)
    denom = (nE.clamp_min(1.0) if use_edge_norm else N.clamp_min(1.0))
    return f_raw / denom

def penalty_from_hetero_constraints(  # constraint graph as hetero
    p: torch.Tensor,
    hetero: Optional[HeteroData],
    *,
    lambda_k: float = 1.0,
    penalty_kind: str = "hinge",  # 'hinge' or 'squared'
    tau: float = 1.0,             # only used if switch to a smooth hinge
    return_per_graph: bool = False,
    normalize_by_num_graphs: bool = True,
) -> torch.Tensor:
    """
    Constraint penalty for a bipartite HeteroData with node types:
        'constr'  (rows of A, one node per constraint)
        'var'     (columns of A, one node per variable)

    and edge type:
        ('constr', 'A', 'var')  with edge_weight = A_ij.

    We assume:
      - p has shape [num_vars] or [num_vars, 1] (same order as hetero['var'] nodes).
      - hetero['constr'].x stores the RHS b (shape [M] or [M,1]).

    Computes (per constraint i):
        lhs_i = sum_j A_ij * p_j
        viol_i = lhs_i - b_i
      then applies a hinge/squared penalty and aggregates per graph.

    Row scaling is ignored (no normalization by row norm).
    """
    # ----- Early exit: no graph or no constraint edges -----
    if (
        hetero is None
        or ("constr", "A", "var") not in (hetero.edge_types if hetero is not None else [])
        or hetero["constr", "A", "var"].edge_index.numel() == 0
    ):
        if return_per_graph:
            # Infer number of graphs B from var.batch if possible
            B = 1
            if hetero is not None and "var" in hetero.node_types:
                var_batch = getattr(hetero["var"], "batch", None)
                if var_batch is not None and var_batch.numel() > 0:
                    B = int(var_batch.to(p.device).max().item() + 1)
            return lambda_k * p.new_zeros(B)
        return lambda_k * p.new_tensor(0.0)

    device = p.device

    # ----- Variables & constraints counts -----
    num_vars = hetero["var"].num_nodes
    num_constr = hetero["constr"].num_nodes

    # Flatten p to [num_vars]
    p = p.to(device)
    if p.dim() > 1:
        p_vec = p.view(-1)
    else:
        p_vec = p
    assert p_vec.numel() == num_vars, "p must match number of 'var' nodes."

    # ----- A_ij from hetero edges -----
    edge_index = hetero["constr", "A", "var"].edge_index.to(device)  # [2, E]
    row = edge_index[0]  # constraint indices (0..num_constr-1)
    col = edge_index[1]  # variable indices   (0..num_vars-1)

    # coefficients A_ij are stored in edge_weight
    a_ej = hetero["constr", "A", "var"].edge_weight
    if a_ej is None:
        a_ej = torch.ones(row.size(0), device=device, dtype=p.dtype)
    else:
        a_ej = a_ej.to(device=device, dtype=p.dtype).view(-1)

    # ----- LHS per constraint: sum_j A_ij * p_j -----
    contrib = a_ej * p_vec[col]  # [E]
    lhs = torch.zeros(num_constr, dtype=p.dtype, device=device)
    lhs.index_add_(0, row, contrib)

    # ----- RHS b per constraint -----
    if getattr(hetero["constr"], "x", None) is None:
        raise ValueError("hetero['constr'].x must store the RHS b for each constraint.")

    bx = hetero["constr"].x.to(device=device, dtype=p.dtype)
    if bx.dim() == 1:
        b_vec = bx
    elif bx.dim() == 2 and bx.size(1) == 1:
        b_vec = bx.view(-1)
    else:
        # If you later add extra constr features, adjust this slice accordingly.
        # For now we expect [M] or [M,1].
        raise ValueError(
            f"Unexpected constr.x shape {tuple(bx.shape)}; expected [M] or [M,1] for b."
        )

    assert b_vec.numel() == num_constr, "b vector must match number of 'constr' nodes."

    # ----- Violations & per-constraint penalties (no row scaling) -----
    viol = lhs - b_vec  # > 0 means violation of A p <= b

    if penalty_kind == "hinge":
        pen_per_c = F.relu(viol)
        # or smooth hinge: pen_per_c = F.softplus(viol / tau)
    elif penalty_kind == "squared":
        pen_per_c = F.relu(viol) ** 2
    else:
        raise ValueError("penalty_kind must be 'hinge' or 'squared'.")

    # ----- Aggregate per graph using batch info -----
    constr_batch = getattr(hetero["constr"], "batch", None)
    var_batch = getattr(hetero["var"], "batch", None)

    if constr_batch is not None and var_batch is not None:
        constr_batch = constr_batch.to(device)
        var_batch = var_batch.to(device)

        # Total number of graphs in the batch according to var nodes
        B = int(var_batch.max().item() + 1) if var_batch.numel() > 0 else 1

        pen_sum = torch.zeros(B, dtype=p.dtype, device=device)

        # Only add where constraints exist; graphs with no constraints stay at 0
        if constr_batch.numel() > 0:
            pen_sum.index_add_(0, constr_batch, pen_per_c)

        per_graph = pen_sum  # sum per graph

        if return_per_graph:
            return lambda_k * per_graph

        total = per_graph.mean() if normalize_by_num_graphs and B > 0 else per_graph.sum()
        return lambda_k * total

    # ----- Single-graph case (no batch info) -----
    total = pen_per_c.mean()
    return lambda_k * total

def fractionality_loss(
    p: torch.Tensor,             # (N,) or (N,1) relaxed probabilities for var nodes
    batch: torch.Tensor,         # (N,) graph id for each variable node
    *,
    lambda_f = 0.1,
    squared: bool = False,       # use (x(1-x)) or squared
    return_per_graph: bool = False,
):
    """
    Batch-aware fractionality loss:
        L = sum_i  x_i (1 - x_i)     or squared.

    Args:
        p: (N,) or (N,1) tensor, relaxed probs.
        batch: (N,) tensor of graph indices.
        squared: if True use (x(1-x))^2.
        return_per_graph: if True returns (B,) tensor; else a single scalar.

    Returns:
        (B,) tensor or scalar.
    """
    if p.dim() == 2:
        p = p.squeeze(-1)

    # Core fractionality term per node
    f = p * (1.0 - p)
    if squared:
        f = f * f  # (x(1-x))^2

    # Aggregate per graph
    if batch is None:
        batch = torch.zeros(p.shape[0], dtype=torch.long).to(p.device)
    
    B = int(batch.max().item()) + 1
    per_graph = scatter_add(f, batch, dim=0, dim_size=B)

    if return_per_graph:
        return lambda_f * per_graph
    else:
        return lambda_f * per_graph.sum()   # or .mean()

def generalized_qp_loss(
    p: Tensor,
    qc_graph, # graph over variables only (for Q,c)
    star_graph, # star-expanded bipartite graph for Ax<=b
    *,
    lambda_k: float = 1.0,
    lambda_f: float = 0.1,
    normalize: bool = True,
    penalty_kind: str = "hinge"
) -> Tensor:
    """
    Loss(p) = [p^T Q p + c^T p] + lambda_k * penalty(Ax<=b from star expansion) + entropy_weight * H(p)
    All terms computed directly on graphs (no explicit matrices).
    """

    if normalize:
        obj = objective_from_graph(
            p, qc_graph, normalize=True,
            return_per_graph=True
        )
    
        pen = penalty_from_hetero_constraints(
            p, star_graph,
            lambda_k=lambda_k,
            penalty_kind=penalty_kind,
            normalize_by_num_graphs=True,
            return_per_graph=True
        ) # v2
    else:
        obj = objective_from_graph(
            p, qc_graph, normalize=False,
            return_per_graph=True
        )
    
        pen = penalty_from_hetero_constraints(
            p, star_graph,
            lambda_k=lambda_k,
            penalty_kind=penalty_kind,
            normalize_by_num_graphs=False,
            return_per_graph=True
        ) # v2

    frac = fractionality_loss(
        p, qc_graph.batch,
        lambda_f=lambda_f,
        return_per_graph=True
    )

    if pen.numel() == 0:
        return (obj + frac).mean()
    
    # if obj.shape != pen.shape:
    #     print(star_graph)
    #     print(star_graph['constr'].batch.max())
    return (obj + pen + frac).mean()