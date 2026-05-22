import torch_geometric.nn as geom_nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_max

# for graphgnnv2
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN, LayerNorm as LN
from torch_geometric.nn import GINConv, HeteroConv, global_mean_pool
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
from torch_geometric.utils import add_remaining_self_loops, dropout_adj
from utils import get_mask
from attention import MixEncoderLayer, prepare_attention_data

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv, 
    "GAT": geom_nn.GATConv, 
    "GraphConv": geom_nn.GraphConv, 
    "SAGEConv": geom_nn.SAGEConv
}

class GraphGNN(nn.Module):
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers=2, 
        layer_name="GraphConv", 
        dropout=0.0
    ):
        super(GraphGNN, self).__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        c_in, c_out = in_channels, hidden_channels
        layers = []
        
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(c_in, c_out),
                geom_nn.norm.LayerNorm(c_out),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            c_in = hidden_channels
        layers += [gnn_layer(c_in, out_channels)]
        self.layers = nn.ModuleList(layers)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight
        if graph.edge_weight is not None:
            edge_weight = graph.edge_weight
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                if edge_weight is not None:
                    x = layer(x, edge_index, edge_weight=edge_weight)
                else:
                    x = layer(x, edge_index)
            else:
                x = layer(x)
        x = F.normalize(x, dim=1)
        return x

class GraphGNNV2(torch.nn.Module):
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        num_layers=4, 
        heads=8, 
        concat=True
    ):
        super(GraphGNNV2, self).__init__()
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.momentum = 0.1
        self.numlayers = num_layers
        self.heads = heads
        self.concat = concat

        self.conv1 = GINConv(Sequential(
            Linear(
                self.in_channels, 
                self.heads * self.hidden_channels
            ),
            ReLU(),
            Linear(
                self.heads * self.hidden_channels, 
                self.heads * self.hidden_channels
            ),
            ReLU(),
            BN(
                self.heads * self.hidden_channels, 
                momentum=self.momentum
            ),
        ), train_eps=True)

        # self.ln1 = torch.nn.LayerNorm(self.heads * self.hidden_channels)
        self.bn1 = BN(self.heads * self.hidden_channels)  

        # graphconv layers
        self.convs = torch.nn.ModuleList()        
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(Sequential(
                Linear(
                    self.heads * self.hidden_channels, 
                    self.heads * self.hidden_channels
                ),
                ReLU(),
                Linear(
                    self.heads * self.hidden_channels, 
                    self.heads * self.hidden_channels
                ),
                ReLU(),
                BN(
                    self.heads * self.hidden_channels, 
                    momentum=self.momentum
                ),
            ),train_eps=True))     
        
        # BN layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(BN(
                self.heads * self.hidden_channels, 
                momentum=self.momentum
            ))

        if self.concat:
            self.lin1 = Linear(self.heads * self.hidden_channels, self.hidden_channels)
        else:
            self.lin1 = Linear(self.hidden_channels, self.hidden_channels)
        self.gnorm = GraphSizeNorm()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.ln1.reset_parameters()
        self.bn1.reset_parameters()
        self.lin1.reset_parameters()
        # self.lin2.reset_parameters()

    def forward(self, graph, edge_dropout=None):
        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch

        if edge_dropout is not None:
            edge_index = dropout_adj(edge_index, edge_attr=(torch.ones(edge_index.shape[1], device=x.device)).long(), 
                                     p=edge_dropout, force_undirected=True)[0]
            edge_index = add_remaining_self_loops(edge_index, num_nodes=batch.shape[0])[0]

        if x.dim() == 1:
            x = x.unsqueeze(-1) # (N, 1)
        mask = get_mask(x, edge_index, 1).to(x.dtype)
        x = F.leaky_relu(self.conv1(x, edge_index))# +x
        x = self.gnorm(x)
        x = self.bn1(x)
        
        for conv, bn in zip(self.convs, self.bns):
            if (x.dim() > 1):
                x = x + F.leaky_relu(conv(x, edge_index))
                mask = get_mask(mask, edge_index, 1).to(x.dtype)
                x = self.gnorm(x)
                x = bn(x)
        
        x = F.leaky_relu(self.lin1(x)) # may be omit this?
        return x, mask

class BipartiteHeteroGNN(torch.nn.Module):
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        num_layers,
        layer_name="GraphConv",
        heads=4,
        momentum=0.1, 
        dropout=0.1
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.momentum = momentum

        out_dim = heads * hidden_channels

        if len(in_channels) == 2:
            self.conv1 = HeteroConv(
                {
                    ("constr", "A", "var"): gnn_layer_by_name[layer_name](in_channels[0], out_dim),
                    ("var", "rev_A", "constr"):  gnn_layer_by_name[layer_name](in_channels[1], out_dim),
                },
                aggr="mean",  # how to aggregate across edge types
            )
        else:
            self.conv1 = HeteroConv(
                {
                    ("constr", "A", "var"): gnn_layer_by_name[layer_name](in_channels, out_dim),
                    ("var", "rev_A", "constr"):  gnn_layer_by_name[layer_name](in_channels, out_dim),
                },
                aggr="mean",  # how to aggregate across edge types
            )

        self.bn1 = torch.nn.ModuleDict(
            {
                # "constr": BN(out_dim, momentum=momentum),
                # "var": BN(out_dim, momentum=momentum),
                "constr": LN(out_dim),
                "var": LN(out_dim),
            }
        )
        
        # --- Hetero GIN layers ---
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            # Share the same MLP for both directions
            conv = HeteroConv(
                {
                    ("constr", "A", "var"): gnn_layer_by_name[layer_name](out_dim, out_dim),
                    ("var", "rev_A", "constr"): gnn_layer_by_name[layer_name](out_dim, out_dim),
                },
                aggr="mean",
            )
            self.convs.append(conv)

        # --- BN per node type, per layer ---
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(
                torch.nn.ModuleDict(
                    {
                        # "constr": BN(out_dim, momentum=momentum),
                        # "var": BN(out_dim, momentum=momentum),
                        "constr": LN(out_dim),
                        "var": LN(out_dim),
                    }
                )
            )

        # optional per-type norm (your old self.gnorm) 
        # self.gnorms = torch.nn.ModuleDict(
        #     {
        #         "constr": torch.nn.LayerNorm(out_dim),
        #         "var": torch.nn.LayerNorm(out_dim)
        #     }
        # )

        self.lin1 = torch.nn.ModuleDict(
            {
                "constr": Linear(out_dim, hidden_channels),
                "var": Linear(out_dim, hidden_channels)
            }
        )

    def forward(
        self, 
        x_dict, 
        edge_index_dict, 
        edge_weight_dict=None, 
        mask_dict=None
    ):
        """
        x_dict: {
          'constr': [Nc, F],
          'var':    [Nv, F],
        }

        edge_index_dict: {
          ('constr','A','var'):      [2, E_cv],
          ('var','rev_A','constr'):  [2, E_vc],
        }

        mask_dict (optional): {
          'constr': mask_constr,  # [Nc, 1] or [Nc, F]
          'var':    mask_var,     # [Nv, 1] or [Nv, F]
        }
        """
        if edge_weight_dict is None:
            x_dict = self.conv1(x_dict, edge_index_dict)
        else:
            x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)

        new_x_dict = {}
        for ntype, x in x_dict.items():
            h = x
            h = F.leaky_relu(x_dict[ntype])
            h = self.bn1[ntype](h)
            new_x_dict[ntype] = h
        x_dict = new_x_dict
            
        new_x_dict = {}
        for ntype, x in x_dict.items():
            h = x

            if h.dim() > 1:
                # residual + nonlinearity
                h = h + F.leaky_relu(x_dict[ntype])
                # h = self.gnorms[ntype](h)
                h = self.bn1[ntype](h)
                new_x_dict[ntype] = h
        x_dict = new_x_dict
        
        for conv, bn_dict in zip(self.convs, self.bns):
            # hetero message passing:
            if edge_weight_dict is None:
                out_dict = conv(x_dict, edge_index_dict)
            else:
                out_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)  # {'constr': Hc, 'var': Hv}

            new_x_dict = {}
            for ntype, x in x_dict.items():
                h = x

                if h.dim() > 1:
                    # residual + nonlinearity
                    h = h + F.leaky_relu(out_dict[ntype])

                    # ----- MASKING (per node type) -----
                    if mask_dict is not None and ntype in mask_dict:
                        base_mask = mask_dict[ntype]
                        if ntype == "var":
                            # messages for var come from ('constr','A','var')
                            eidx = edge_index_dict[("constr", "A", "var")]
                        else:  # 'constr'
                            # messages for constr come from ('var','rev_A','constr')
                            eidx = edge_index_dict[("var", "rev_A", "constr")]

                        # adapt your original pattern: mask = get_mask(mask, edge_index, 1)
                        m = get_mask(base_mask, eidx, 1).to(h.dtype)
                        h = h * m

                    # ----- norms per node type -----
                    # h = self.gnorms[ntype](h)
                    h = bn_dict[ntype](h)

                new_x_dict[ntype] = h

            x_dict = new_x_dict

        x_dict['constr'] = F.leaky_relu(self.lin1['constr'](x_dict['constr']))
        x_dict['var'] = F.leaky_relu(self.lin1['var'](x_dict['var']))

        return x_dict

class UnifiedQPModel(nn.Module):
    def __init__(
        self, 
        graph_in_channels, 
        qc_in_channels,
        ab_in_channels,
        hidden_channels, 
        out_channels,  
        qc_layer_name="GraphConv",
        ab_layer_name="GraphConv",
        use_attn=False, 
        use_graph_embed=False,
        n_graph_layers=6, 
        n_qc_layers=2, 
        n_ab_layers=1,
        return_features=False
    ):
        super(UnifiedQPModel, self).__init__()
        self.use_attn = use_attn
        self.use_graph_embed = use_graph_embed
        self.hidden_channels = hidden_channels
        self.return_features = return_features
        
        # problem graph, input is Data
        self.graph_model = GraphGNNV2(graph_in_channels, hidden_channels, num_layers=n_graph_layers) # v2

        # Qc graph, input is Data
        self.qc_model = GraphGNN(qc_in_channels, hidden_channels, hidden_channels, layer_name=qc_layer_name, num_layers=n_qc_layers)

        # Ab graph, input is HeteroData
        self.ab_model = BipartiteHeteroGNN(
            ab_in_channels, hidden_channels, 
            layer_name=ab_layer_name, num_layers=n_ab_layers
        )

        if self.use_attn:
            self.attn = MixEncoderLayer(
                hidden_channels,
                hidden_channels,
                4,
                hidden_channels // 4,
                hidden_channels // 4
            )

        # decoders
        self.mlp1 = nn.Sequential(
            nn.Linear(5 * hidden_channels if use_graph_embed else 3 * hidden_channels, hidden_channels),
            nn.LeakyReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),  # out_features = 1
            nn.LeakyReLU()
        )

    def construct_graph_embedding(self, node_embeddings, batch, eps=1e-12):
        mu = global_mean_pool(node_embeddings, batch)
        sq = global_mean_pool(node_embeddings**2, batch)
        var = (sq - mu**2).clamp_min(eps)
        std = torch.sqrt(var)
        return torch.cat([mu, std], dim=-1) # B, 2H
    
    def forward(self, problem_graph, qc_graph, ab_graph):

        graph_batch = problem_graph.batch
        if graph_batch is None:
            N_size = problem_graph.x.size(0)
            graph_batch = torch.zeros(N_size, dtype=torch.long).to(problem_graph.x.device)
        else:
            N_size = int(graph_batch.max().item()) + 1

        qc_features = self.qc_model(qc_graph) #v1

        graph_features, _ = self.graph_model(problem_graph) # v2

        if ab_graph is None:
            ab_v_features = graph_features.new_zeros(graph_features.size(0), graph_features.size(1), device=problem_graph.x.device)
            ab_e_features = None
        else:
            ab_features = self.ab_model(ab_graph.x_dict, ab_graph.edge_index_dict, ab_graph.edge_weight_dict)
            ab_v_features, ab_e_features = ab_features['var'], ab_features['constr']

        if self.use_graph_embed:
            problem_whole_graph_features = self.construct_graph_embedding(graph_features, graph_batch) # B, 2D
            problem_node_graph_features = torch.cat([graph_features, problem_whole_graph_features[graph_batch]], dim=-1) # N, 3D
            
            fusion = torch.cat([problem_node_graph_features, qc_features, ab_v_features], dim=-1)
        else:
            fusion = torch.cat([graph_features, qc_features, ab_v_features], dim=-1)
        
        if self.return_features: return fusion # return fusion features if requested
        
        out = self.mlp1(fusion)
        # if self.return_features: return out # return fusion features if requested

        if self.use_attn:
            batch_var = ab_graph["var"].batch
            batch_constr = ab_graph["constr"].batch
            query, k_v, cost_mat = prepare_attention_data(
                out, ab_e_features,
                batch_var, batch_constr,
                ab_graph["constr", "A", "var"].edge_index
            )
            # print(query.shape, k_v.shape, cost_mat.shape)
            dec_out_batched = self.attn(query, cost_mat, k_v)
            dec_out_flat = torch.empty_like(out)
            for b in range(N_size):
                idx_v = (batch_var == b).nonzero(as_tuple=True)[0]
                n_b = idx_v.numel()
                if n_b > 0:
                    dec_out_flat[idx_v] = dec_out_batched[b, :n_b]
            out = dec_out_flat            

        out = self.mlp2(out)

        batch_max = scatter_max(out, graph_batch, 0, dim_size= N_size)[0]
        batch_max = torch.index_select(batch_max, 0, graph_batch)  
        batch_min = scatter_min(out, graph_batch, 0, dim_size= N_size)[0]
        batch_min = torch.index_select(batch_min, 0, graph_batch)
        out = (out - batch_min) / (batch_max + 1e-6 - batch_min)
        return out