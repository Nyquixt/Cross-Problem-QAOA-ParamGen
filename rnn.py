import torch.nn as nn
import torch

class LSTMNet(nn.Module):
    def __init__(self, p, device, T=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.p = p
        self.device = device
        self.T = T
        self.cell = nn.LSTMCell(input_size=1 + 2 * p, hidden_size=2 * self.p)

    def run_iteration(self, inputs, graph_cost):
        prev_cost = inputs[0]    # shape (1, 1)
        prev_params = inputs[1]   # shape (1, 2*p)
        prev_h = inputs[2]        # shape (1, 2*p)
        prev_c = inputs[3]       # shape (1, 2*p)

        new_input = torch.cat([prev_cost.to(self.device), prev_params.to(self.device)], dim=1).to(self.device)
        new_h, new_c = self.cell(new_input, (prev_h, prev_c))
        new_params = new_h
        _params = new_params.view(2, self.p) # (1, 2*p) -> (2, p)
        _cost = graph_cost(_params)

        new_cost = _cost.view(1, 1).float()
        return [new_cost, new_params, new_h, new_c]

    def forward(self, graph_cost, intermediate_steps=False):
        
        initial_cost = torch.zeros((1, 1)).to(self.device)

        gamma = torch.rand((1, self.p), device=self.device) * 5e-2
        beta = torch.rand((1, self.p), device=self.device) * 5e-2
        initial_params = torch.cat([gamma, beta], dim=1)
        
        initial_h = torch.zeros((1, 2 * self.p)).to(self.device)
        initial_c = torch.zeros((1, 2 * self.p)).to(self.device)

        outputs = []
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device=self.device)
        current_output = [initial_cost, initial_params, initial_h, initial_c]

        for i in range(self.T):
            current_output = self.run_iteration(current_output, graph_cost)
            outputs.append(current_output)

        # Weighted sum of costs
        loss = torch.zeros((1, 1), device=self.device)
        for i in range(self.T):
            loss += weights[i] * outputs[i][0] # outputs[i][0] is the cost from iteration i

        if intermediate_steps:
            param_list = [initial_params] + [outputs[i][1] for i in range(self.T)]
            return param_list + [loss]
        
        return loss

class ConditionedLSTM(nn.Module): # condition at the start hidden and cell states, add normalized graph embedding to hidden state at every step
    def __init__(self, p, device, graph_dim, hidden_size=64, T=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.device = device
        self.graph_dim = graph_dim
        self.hidden_size = hidden_size
        self.T = T

        self.cell = nn.LSTMCell(input_size=1 + 2 * p, hidden_size=hidden_size)

        # Map graph embedding to initial hidden and cell states
        self.init_h = nn.Linear(graph_dim, hidden_size)
        self.init_c = nn.Linear(graph_dim, hidden_size)
        
        # Project hidden state to parameters (2p)
        self.param_head = nn.Linear(hidden_size, 2 * p)
        
        # Project graph embedding to hidden size for adding to hidden state at each step
        self.graph_to_hidden = nn.Linear(graph_dim, hidden_size)

    def run_iteration(self, inputs, graph_cost, g_embed_normalized):
        prev_cost, prev_params, prev_h, prev_c = inputs
        new_input = torch.cat([prev_cost.to(self.device), prev_params.to(self.device)], dim=1).to(self.device)

        new_h, new_c = self.cell(new_input, (prev_h, prev_c))
        
        # Add normalized graph embedding to hidden state
        new_h = new_h + g_embed_normalized
        
        # Project hidden state to parameters
        new_params = self.param_head(new_h)
        # new_params = torch.pi * torch.tanh(new_params)

        _params = new_params.view(2, self.p)
        _cost = graph_cost(_params)
        new_cost = _cost.view(1, 1).float()

        return [new_cost, new_params, new_h, new_c]

    def forward(self, graph_cost, g_cond, intermediate_steps=False):
        g_cond = torch.from_numpy(g_cond).unsqueeze(0)
        g = g_cond.to(self.device)  # shape (1, graph_dim)

        initial_cost = torch.zeros((1, 1), device=self.device)

        gamma = torch.rand((1, self.p), device=self.device) * 5e-2
        beta = torch.rand((1, self.p), device=self.device) * 5e-2
        initial_params = torch.cat([gamma, beta], dim=1)

        # condition via initial hidden/cell
        initial_h = self.init_h(g)
        initial_c = self.init_c(g)
        
        # Project and normalize graph embedding
        g_embed_projected = self.graph_to_hidden(g)
        g_embed_normalized = g_embed_projected / (torch.norm(g_embed_projected, dim=1, keepdim=True) + 1e-8)

        outputs = []
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device=self.device)
        current_output = [initial_cost, initial_params, initial_h, initial_c]

        for _ in range(self.T):
            current_output = self.run_iteration(current_output, graph_cost, g_embed_normalized)
            outputs.append(current_output)

        loss = torch.zeros((1, 1), device=self.device)
        for i in range(self.T):
            loss += weights[i] * outputs[i][0]
        
        if intermediate_steps:
            param_list = [initial_params] + [outputs[i][1] for i in range(self.T)]
            return param_list + [loss]

        return loss