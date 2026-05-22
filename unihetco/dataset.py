from problems import *
from problem_utils import extract_docplex_matrices
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import random
from typing import List, Optional
import networkx as nx
from features import get_graph_features, build_qc_node_features, build_ab_hetero_features

def build_model_by_name(name, *args, **kwargs):
    func_name = f"build_{name}_model"
    try:
        func = globals()[func_name]
    except KeyError:
        raise ValueError(f"No function named '{func_name}' found")
    return func(*args, **kwargs)

def keep_largest_component(G, relabel=True):
    """Return the induced subgraph on the largest connected component (undirected)."""
    if G.number_of_nodes() == 0:
        return G.copy()
    # Largest by number of nodes
    largest_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_nodes).copy()
    if relabel:
        H = nx.convert_node_labels_to_integers(H, first_label=0)  # stable and fast
    return H

class FixedTrainDataset(Dataset):
    def __init__(self, 
                 graphs: List[nx.Graph],
                 problem: str = "mis",
                 *,
                 base_seed: Optional[int] = None):
        super().__init__()
        self.graphs = graphs
        self.problem = problem
        self.base_seed = base_seed
        self.rng = random.Random(base_seed)

    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx):
        # graph = nx.to_undirected(self.graphs[idx]["graph"])
        graph = nx.to_undirected(self.graphs[idx])
        graph = keep_largest_component(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # PyG Data problem graph
        problem_graph = get_graph_features(
            graph, 
            clustering=True,
            betweenness=True,
            closeness=False,
            core=True,
            eigenvec=False,
            pagerank=False,
            undirected=True,
            use_edge_weight=False,
            normalize=True
        )
        
        # building Qc and Ab graphs
        edges = list(graph.edges)
        model_args = {"edges": edges}

        if self.problem == "maxcut":
            for (u, v) in graph.edges(): # set edge weight = 1 by default
                graph.edges[u, v]["weight"] = 1.

            weights = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
            model_args["weights"] = weights

        mdl, _ = build_model_by_name(self.problem, **model_args)

        Q, c, A, b, _, _ = extract_docplex_matrices(mdl)

        qc_graph = build_qc_node_features(
            torch.from_numpy(Q).float(), 
            torch.from_numpy(c).float()
        )
        
        if self.problem == "maxcut":
            ab_graph = None
        else:
            ab_graph = build_ab_hetero_features(
                torch.from_numpy(A).float(), 
                torch.from_numpy(b).float()
            )
        
        return {
            "problem": self.problem,
            "problem_graph": problem_graph,
            "qc_graph": qc_graph,
            "ab_graph": ab_graph,
            "opt_val": None
        }

class FixedTestDataset(Dataset):
    def __init__(self, 
                 graphs: List[nx.Graph],
                 opt_vals: List[int],
                 problem: str = "mis",
                 *,
                 base_seed: Optional[int] = None):
        super().__init__()
        assert len(graphs) == len(opt_vals)
        self.graphs = graphs
        self.opt_vals = opt_vals
        self.problem = problem
        self.base_seed = base_seed
        self.rng = random.Random(base_seed)

    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx):
        # graph = nx.to_undirected(self.graphs[idx]["graph"])
        graph = nx.to_undirected(self.graphs[idx])
        graph = keep_largest_component(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        opt = self.opt_vals[idx]

        # PyG Data problem graph
        problem_graph = get_graph_features(
            graph, 
            clustering=True,
            betweenness=True,
            closeness=False,
            core=True,
            eigenvec=False,
            pagerank=False,
            undirected=True,
            use_edge_weight=False,
            normalize=True
        )
        
        # building Qc and Ab graphs
        edges = list(graph.edges)
        model_args = {"edges": edges}

        if self.problem == "maxcut":
            for (u, v) in graph.edges(): # set edge weight = 1 by default
                graph.edges[u, v]["weight"] = 1.

            weights = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
            model_args["weights"] = weights

        mdl, _ = build_model_by_name(self.problem, **model_args)

        Q, c, A, b, _, _ = extract_docplex_matrices(mdl)

        qc_graph = build_qc_node_features(
            torch.from_numpy(Q).float(), 
            torch.from_numpy(c).float()
        )
        
        if self.problem == "maxcut":
            ab_graph = None
        else:
            ab_graph = build_ab_hetero_features(
                torch.from_numpy(A).float(), 
                torch.from_numpy(b).float()
            )
        
        return {
            "problem": self.problem,
            "problem_graph": problem_graph,
            "qc_graph": qc_graph,
            "ab_graph": ab_graph,
            "opt_val": opt
        }
        
def collate_fn(batch):
    problems = [item["problem"] for item in batch]
    problem_graphs = [item["problem_graph"] for item in batch]
    qc_graphs = [item["qc_graph"] for item in batch]
    if batch[0]["ab_graph"] is None:
        ab_graphs = None
    else:
        ab_graphs = [item["ab_graph"] for item in batch]
    opt_vals = [item["opt_val"] for item in batch]

    problem_graph_batch = Batch.from_data_list(problem_graphs)
    qc_graph_batch = Batch.from_data_list(qc_graphs)
    if ab_graphs is None:
        ab_graph_batch = None
    else:
        ab_graph_batch = Batch.from_data_list(ab_graphs)
    if opt_vals[0] is not None:
        opt_vals = torch.tensor(opt_vals)

    return {
        "problems": problems,
        "problem_graph_batch": problem_graph_batch,
        "qc_graph_batch": qc_graph_batch,
        "ab_graph_batch": ab_graph_batch,
        "opt_vals": opt_vals
    }