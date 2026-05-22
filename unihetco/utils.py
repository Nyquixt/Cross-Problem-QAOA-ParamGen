from loss_unify import generalized_qp_loss
from copy import deepcopy
import copy
import torch
from torch_geometric.data import Data, Batch

from torch_scatter import scatter_add
import pandas as pd
import math

class EarlyStopping:
    """
    Early stopping on a scalar metric.

    mode:
      - "min": lower is better (e.g., val_loss)
      - "max": higher is better (e.g., val_acc, approx ratio)
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "min"):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode

        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0
        self.best_state = None
        self.best_epoch = -1

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        else:
            return value > (self.best + self.min_delta)

    def step(self, value: float, model: torch.nn.Module, epoch: int) -> bool:
        """
            Returns True if training should stop.
        """
        if self._is_improvement(value):
            self.best = value
            self.num_bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    def restore_best(self, model: torch.nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def propagate(x, edge_index):
    row, col = edge_index
    out = scatter_add(x[col], row, dim=0)
    return out

# get the k-hop neighbors of node sample
def get_mask(x, edge_index, hops):
    for _ in range(hops):
        x = propagate(x, edge_index)
    mask = (x > 0).float()
    return mask

def as_batch(single: Data) -> Batch:
    # keep device consistent
    dev = single.edge_index.device if hasattr(single, "edge_index") else None
    B = Batch.from_data_list([single])
    if dev is not None:
        B = B.to(dev)
    return B

def finetune_one_graph(
    problem_graph, 
    qc_graph, 
    ab_graph, 
    model,
    lambda_k=1.0,
    lambda_f=1.0,
    n_updates=5
):
    ft_model = deepcopy(model)
    

    if n_updates > 0:
        ft_model.train()
        # optimizer = torch.optim.Adam(ft_model.parameters(), lr=0.0001, weight_decay=0.0)
        optimizer = torch.optim.RMSprop(ft_model.parameters(), lr=0.0001, weight_decay=0.0)
        
        for _ in range(n_updates):
            p = ft_model(problem_graph, qc_graph, ab_graph)
                    
            loss = generalized_qp_loss(
                p, qc_graph, ab_graph,
                lambda_k=lambda_k, lambda_f=lambda_f,
                penalty_kind="hinge",
                entropy_weight=0.0, normalize=False
            )
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            optimizer.step()
    
    final_p = ft_model(problem_graph, qc_graph, ab_graph)
    
    return final_p

@torch.no_grad()
def collect_node_probs(model, loader, device):
    """
    Returns a dict with per-batch tensors:
      - 'probs':  (N_total_vars,) concatenated probabilities (variable nodes only if var_only)
      - 'graph_id': (N_total_vars,) graph index per node
      - 'node_id_global': (N_total_vars,) global node index within the batch stream
    """
    model.eval()
    all_probs = []
    all_graph_ids = []
    all_node_ids = []

    # global node id across all batches
    running_node_offset = 0

    for batch in loader:
        problem_graph_batch = batch["problem_graph_batch"].to(device)
        qc_graph_batch = batch["qc_graph_batch"].to(device)
        if batch["ab_graph_batch"] is None:
            ab_graph_batch = None
        else:
            ab_graph_batch = batch["ab_graph_batch"].to(device)

        # Forward pass: whatever your model returns for per-node scores
        p = model(problem_graph_batch, qc_graph_batch, ab_graph_batch)
        graph_ids = problem_graph_batch.batch

        num_nodes = int(problem_graph_batch.num_nodes)
        node_ids = torch.arange(num_nodes, device=device) + running_node_offset
        running_node_offset += num_nodes

        all_probs.append(p.squeeze().detach().float().cpu())
        all_graph_ids.append(graph_ids.detach().cpu())
        all_node_ids.append(node_ids.detach().cpu())

        probs_data = torch.cat(all_probs, dim=0).numpy()
        graph_ids_data = torch.cat(all_graph_ids, dim=0).numpy()
        node_ids_data = torch.cat(all_node_ids, dim=0).numpy()

        # print(probs_data.shape, graph_ids_data.shape, node_ids_data.shape)

    return {
        "probs": probs_data,
        "graph_id": graph_ids_data,
        "node_id_global": node_ids_data,
    }

@torch.no_grad()
def collect_node_probs_hetero(model, loader, device):
    """
    Returns a dict with per-batch tensors:
      - 'probs':  (N_total_vars,) concatenated probabilities (variable nodes only if var_only)
      - 'graph_id': (N_total_vars,) graph index per node
      - 'node_id_global': (N_total_vars,) global node index within the batch stream
    """
    model.eval()
    all_probs = []
    all_graph_ids = []
    all_node_ids = []

    # global node id across all batches
    running_node_offset = 0

    for batch in loader:
        hetero_graph_batch = batch["hetero_graph_batch"].to(device)
        p = model(hetero_graph_batch)
        graph_ids = hetero_graph_batch['var'].batch.to(device)   # [N_var_batch]
        num_nodes = int(hetero_graph_batch['var'].x.size(0))

        node_ids = torch.arange(num_nodes, device=device) + running_node_offset
        running_node_offset += num_nodes

        all_probs.append(p.squeeze().detach().float().cpu())
        all_graph_ids.append(graph_ids.detach().cpu())
        all_node_ids.append(node_ids.detach().cpu())

        probs_data = torch.cat(all_probs, dim=0).numpy()
        graph_ids_data = torch.cat(all_graph_ids, dim=0).numpy()
        node_ids_data = torch.cat(all_node_ids, dim=0).numpy()
    
        # print(probs_data.shape, graph_ids_data.shape, node_ids_data.shape)

    return {
        "probs": probs_data,
        "graph_id": graph_ids_data,
        "node_id_global": node_ids_data,
    }

def to_df(split_name, out_dict):
    df = pd.DataFrame({
        "split": split_name,
        "graph_id": out_dict["graph_id"],
        "node_global": out_dict["node_id_global"],
        "prob": out_dict["probs"],
    })
    return df

def hetero_to_graph_batch(hetero_batch):
    """
    Extracts the ('var','prob','var') relation and produces a 
    standard PyG graph Batch with:
        - x: [N_var, F]
        - edge_index: [2, E_prob]
        - batch: [N_var]
    """
    # Node features of variable nodes
    x = hetero_batch['var'].x            # [N_var, F]

    # Batch vector of variable nodes
    batch = hetero_batch['var'].batch    # [N_var]

    # Problem-graph edges: var --prob-- var
    edge_index = hetero_batch['var', 'prob', 'var'].edge_index

    # Build Data object
    data = Data(x=x, edge_index=edge_index, batch=batch)

    return Batch.from_data_list([data])