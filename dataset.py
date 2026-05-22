import networkx as nx
from torch.utils.data import Dataset

import random

def _make_choose_n(n_nodes_range, rng):
    # normalize n_nodes_range to (min_n, max_n) OR explicit choices
    if (
        isinstance(n_nodes_range, (tuple, list))
        and len(n_nodes_range) == 2
        and all(isinstance(x, int) for x in n_nodes_range)
    ):
        n_min, n_max = n_nodes_range
        return lambda: rng.randint(n_min, n_max)
    choices = list(n_nodes_range)
    return lambda: rng.choice(choices)

def generate_graphs(n_graphs, n_nodes_range, seed=None):
    """
    Generate connected Erdos-Renyi graphs.

    Args:
        n_graphs: int, number of graphs to return
        n_nodes_range: tuple (min_n, max_n) or list of possible n values
        seed: int or None

    Returns:
        list[nx.Graph] of length n_graphs, all connected
    """
    rng = random.Random(seed)
    choose_n = _make_choose_n(n_nodes_range, rng)
    
    graphs = []
    attempt = 0
    while len(graphs) < n_graphs:
        attempt += 1
        n = choose_n()
        k = rng.randint(3, n - 1)
        p_edge = k / n

        # vary seed per attempt to avoid repeating identical graphs
        g_seed = rng.randrange(2**31 - 1) if seed is not None else None

        G = nx.erdos_renyi_graph(n, p=p_edge, seed=g_seed)

        # ensure connected (and nontrivial)
        if n <= 1 or not nx.is_connected(G):
            continue

        graphs.append(G)

    return graphs

def generate_random_regular_graphs(n_graphs, n_nodes_range, d_range=(3, 7), seed=None):
    """
    Generate connected random d-regular graphs.

    Constraints enforced:
      - d < n
      - (n*d) % 2 == 0
      - graph is connected
    """
    rng = random.Random(seed)
    choose_n = _make_choose_n(n_nodes_range, rng)

    graphs = []
    while len(graphs) < n_graphs:
        n = choose_n()
        d_min, d_max = d_range
        if n <= d_min:
            continue

        d = rng.randint(d_min, min(d_max, n - 1))
        if d >= n or (n * d) % 2 != 0:
            continue

        g_seed = rng.randrange(2**31 - 1) if seed is not None else None

        try:
            G = nx.random_regular_graph(d, n, seed=g_seed)
        except nx.NetworkXError:
            continue

        if n <= 1 or not nx.is_connected(G):
            continue

        graphs.append(G)

    return graphs


def generate_watts_strogatz_graphs(n_graphs, n_nodes_range, k_range=(3, 10), p_rewire_range=(0.05, 0.5), seed=None):
    """
    Generate connected Watts–Strogatz small-world graphs.

    Params:
      - n sampled from n_nodes_range
      - k sampled from k_range (even k is typical; we enforce k even)
      - p rewiring sampled uniformly from p_rewire_range
    """
    rng = random.Random(seed)
    choose_n = _make_choose_n(n_nodes_range, rng)

    graphs = []
    while len(graphs) < n_graphs:
        n = choose_n()
        if n <= 3:
            continue

        k_min, k_max = k_range
        k_max = min(k_max, n - 1)
        if k_min > k_max:
            continue

        k = rng.randint(k_min, k_max)
        if k % 2 == 1:
            k = k - 1
        if k < 2:
            continue

        p0, p1 = p_rewire_range
        p = rng.uniform(p0, p1)

        g_seed = rng.randrange(2**31 - 1) if seed is not None else None
        G = nx.watts_strogatz_graph(n, k, p, seed=g_seed)

        if n <= 1 or not nx.is_connected(G):
            continue

        graphs.append(G)

    return graphs

def generate_barabasi_albert_graphs(n_graphs, n_nodes_range, m_range=(1, 5), seed=None):
    """
    Generate connected Barabási–Albert preferential attachment graphs.

    Params:
      - n sampled from n_nodes_range
      - m sampled from m_range, with 1 <= m < n
    """
    rng = random.Random(seed)
    choose_n = _make_choose_n(n_nodes_range, rng)

    graphs = []
    while len(graphs) < n_graphs:
        n = choose_n()
        if n <= 2:
            continue

        m_min, m_max = m_range
        m_max = min(m_max, n - 1)
        if m_min > m_max:
            continue

        m = rng.randint(m_min, m_max)
        if not (1 <= m < n):
            continue

        g_seed = rng.randrange(2**31 - 1) if seed is not None else None
        G = nx.barabasi_albert_graph(n, m, seed=g_seed)

        # BA graphs are connected for valid (n,m), but keep the check for safety
        if n <= 1 or not nx.is_connected(G):
            continue

        graphs.append(G)

    return graphs

class GraphDataset(Dataset):
    def __init__(self, graphs, graph_cost, embeddings=None):
        super().__init__()
        self.graphs = graphs
        self.graph_cost = graph_cost
        if embeddings is not None:
            self.embeddings = embeddings

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        if hasattr(self, 'embeddings'):
            return {
                "graph": self.graphs[index],
                "graph_cost": self.graph_cost[index],
                "embedding": self.embeddings[index]
            }
        
        return {
            "graph": self.graphs[index],
            "graph_cost": self.graph_cost[index]
        }
    
def collate_fn(batch):
    graphs = [item["graph"] for item in batch]
    graph_costs = [item["graph_cost"] for item in batch]

    if "embedding" in batch[0].keys():
        embeddings = [item["embedding"] for item in batch]
        return {
            "graphs": graphs,
            "graph_costs": graph_costs,
            "embeddings": embeddings
        }
    return {
        "graphs": graphs,
        "graph_costs": graph_costs
    }